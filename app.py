from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, Response
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import psycopg2
from werkzeug.security import generate_password_hash, check_password_hash
import re
import os
import logging
import json
import requests
import sqlite3
from xml.etree import ElementTree
from ratelimit import limits, sleep_and_retry
from sentence_transformers import SentenceTransformer
import numpy as np
from scipy.spatial.distance import cosine
from dotenv import load_dotenv
from datetime import datetime, timedelta
from collections import Counter
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import sendgrid
from sendgrid.helpers.mail import Mail, Email, To, Content
import traceback
import time
import tenacity
import email_validator
from email_validator import validate_email, EmailNotValidError
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
import asyncio
import aiohttp
import hashlib

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'your-secret-key')
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize embedding model
embedding_model = None

# Initialize scheduler
scheduler = BackgroundScheduler()
scheduler.start()

# SendGrid client
sendgrid_api_key = os.environ.get('SENDGRID_API_KEY', '').strip()
if not sendgrid_api_key:
    logger.error("SENDGRID_API_KEY not set in environment variables")
sg = sendgrid.SendGridAPIClient(sendgrid_api_key) if sendgrid_api_key else None

# Biomedical vocabulary for synonyms
BIOMEDICAL_VOCAB = {
    "diabetes": ["Diabetes Mellitus", "insulin resistance", "type 2 diabetes"],
    "weight loss": ["obesity", "body weight reduction", "fat loss"],
    "treatment": ["therapy", "intervention", "management"],
    "disease": ["disorder", "condition", "pathology"],
    "statins": ["HMG-CoA reductase inhibitors", "atorvastatin", "simvastatin"],
    "heart disease": ["cardiovascular disease", "coronary artery disease", "myocardial infarction"],
    "cardiovascular": ["heart-related", "circulatory"],
    "blood pressure": ["hypertension", "BP"],
    "hypertension": ["high blood pressure", "elevated BP"]
}

# Initialize SQLite database for search progress and cache
def init_progress_db():
    conn = sqlite3.connect('search_progress.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS search_progress
                 (user_id TEXT, query TEXT, status TEXT, timestamp REAL)''')
    c.execute('''CREATE TABLE IF NOT EXISTS grok_cache
                 (query TEXT PRIMARY KEY, response TEXT, timestamp REAL)''')
    c.execute('''CREATE TABLE IF NOT EXISTS embedding_cache
                 (pmid TEXT PRIMARY KEY, embedding BLOB, timestamp REAL)''')
    conn.commit()
    conn.close()

init_progress_db()

def update_search_progress(user_id, query, status):
    conn = sqlite3.connect('search_progress.db')
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO search_progress (user_id, query, status, timestamp) VALUES (?, ?, ?, ?)",
              (user_id, query, status, time.time()))
    conn.commit()
    conn.close()
    logger.info(f"Search progress updated: user={user_id}, query={query}, status={status}")

def get_search_progress(user_id, query):
    conn = sqlite3.connect('search_progress.db')
    c = conn.cursor()
    c.execute("SELECT status, timestamp FROM search_progress WHERE user_id = ? AND query = ? ORDER BY timestamp DESC LIMIT 1",
              (user_id, query))
    result = c.fetchone()
    conn.close()
    return result if result else (None, None)

def cache_grok_response(query, response):
    conn = sqlite3.connect('search_progress.db')
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO grok_cache (query, response, timestamp) VALUES (?, ?, ?)",
              (query, response, time.time()))
    conn.commit()
    conn.close()
    logger.info(f"Cached Grok response for query: {query[:50]}...")

def get_cached_grok_response(query):
    conn = sqlite3.connect('search_progress.db')
    c = conn.cursor()
    c.execute("SELECT response, timestamp FROM grok_cache WHERE query = ? AND timestamp > ?",
              (query, time.time() - 604800))  # Cache valid for 7 days
    result = c.fetchone()
    conn.close()
    return result[0] if result else None

def cache_embedding(pmid, embedding):
    conn = sqlite3.connect('search_progress.db')
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO embedding_cache (pmid, embedding, timestamp) VALUES (?, ?, ?)",
              (pmid, embedding.tobytes(), time.time()))
    conn.commit()
    conn.close()

def get_cached_embedding(pmid):
    conn = sqlite3.connect('search_progress.db')
    c = conn.cursor()
    c.execute("SELECT embedding, timestamp FROM embedding_cache WHERE pmid = ? AND timestamp > ?",
              (pmid, time.time() - 604800))
    result = c.fetchone()
    conn.close()
    if result:
        return np.frombuffer(result[0])
    return None

def load_embedding_model():
    global embedding_model
    if embedding_model is None:
        logger.info("Loading sentence-transformers model...")
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Model loaded.")
    return embedding_model

def generate_embedding(text):
    model = load_embedding_model()
    return model.encode(text, convert_to_numpy=True)

async def query_grok_api_async(query, context, prompt="Process the provided context according to the user's prompt."):
    try:
        cache_key = hashlib.md5((query + context + prompt).encode()).hexdigest()
        cached_response = get_cached_grok_response(cache_key)
        if cached_response:
            logger.info(f"Using cached Grok response for query: {query[:50]}...")
            return cached_response
        
        api_key = os.environ.get('XAI_API_KEY')
        if not api_key:
            logger.error("XAI_API_KEY not set")
            return "Error: xAI API key not configured"
        
        async with aiohttp.ClientSession() as session:
            url = "https://api.x.ai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "model": "grok-3",
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"Based on the following context, answer the prompt: {query}\n\nContext: {context}"}
                ],
                "max_tokens": 1000
            }
            logger.info(f"Sending async Grok API request: prompt={query[:50]}..., context_length={len(context)}")
            async with session.post(url, headers=headers, json=data, timeout=300) as response:
                response.raise_for_status()
                response_json = await response.json()
                response_text = response_json['choices'][0]['message']['content']
                logger.info(f"Async Grok API response received: length={len(response_text)}")
                cache_grok_response(cache_key, response_text)
                return response_text
    except Exception as e:
        logger.error(f"Async Grok API call failed: {str(e)}")
        if isinstance(e, aiohttp.ClientResponseError):
            logger.error(f"HTTP Status: {e.status}, Message: {e.message}")
        elif isinstance(e, aiohttp.ClientConnectionError):
            logger.error("Connection error, possibly network issue")
        return None

@tenacity.retry(
    stop=tenacity.stop_after_attempt(2),
    wait=tenacity.wait_exponential(multiplier=1, min=2, max=15),
    retry=tenacity.retry_if_exception_type((requests.exceptions.RequestException, Exception)),
    before_sleep=lambda retry_state: logger.info(f"Retrying Grok API call, attempt {retry_state.attempt_number}")
)
def query_grok_api(query, context, prompt="Process the provided context according to the user's prompt."):
    loop = asyncio.get_event_loop()
    try:
        return loop.run_until_complete(query_grok_api_async(query, context, prompt))
    except Exception as e:
        logger.error(f"Grok API call failed after retries: {str(e)}")
        raise

def get_db_connection():
    conn = psycopg2.connect(os.environ['DATABASE_URL'])
    return conn

class User(UserMixin):
    def __init__(self, id, email):
        self.id = id
        self.email = email

@login_manager.user_loader
def load_user(user_id):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, email FROM users WHERE id = %s", (user_id,))
    user = cur.fetchone()
    cur.close()
    conn.close()
    if user:
        return User(user[0], user[1])
    return None

def validate_user_email(email):
    try:
        validate_email(email, check_deliverability=False)
        return True
    except EmailNotValidError as e:
        logger.error(f"Invalid email address: {email}, error: {str(e)}")
        return False

def run_notification_rule(rule_id, user_id, rule_name, keywords, timeframe, prompt_text, email_format, user_email, test_mode=False):
    logger.info(f"Running notification rule {rule_id} ({rule_name}) for user {user_id}, keywords: {keywords}, timeframe: {timeframe}, test_mode: {test_mode}, recipient: {user_email}")
    if not validate_user_email(user_email):
        raise ValueError(f"Invalid recipient email address: {user_email}")
    
    query = keywords
    keywords_with_synonyms, date_range, _ = extract_keywords_and_date(query)
    start_year = datetime.now().year - {'daily': 1, 'weekly': 1, 'monthly': 1, 'annually': 1}[timeframe]
    search_query = build_pubmed_query(keywords_with_synonyms, date_range or f"{start_year}/01/01[dp] TO {datetime.now().strftime('%Y/%m/%d')}[dp]")
    
    try:
        api_key = os.environ.get('PUBMED_API_KEY')
        esearch_result = esearch(search_query, retmax=20, api_key=api_key)
        pmids = esearch_result['esearchresult']['idlist']
        logger.info(f"Notification rule {rule_id} query: {search_query}, PMIDs: {len(pmids)}")
        if not pmids:
            logger.info(f"No new results for rule {rule_id}")
            content = "No new results found for this rule."
            if not sg:
                raise Exception("SendGrid API key not configured. Please contact support.")
            message = Mail(
                from_email=Email("noreply@firesidetechnologies.com"),
                to_emails=To(user_email),
                subject=f"PubMedResearcher {'Test ' if test_mode else ''}Notification: {rule_name}",
                plain_text_content=content
            )
            logger.info(f"Sending email for rule {rule_id}, recipient: {user_email}, subject: {message.subject}")
            response = sg.send(message)
            response_headers = {k: v for k, v in response.headers.items()}
            logger.info(f"Email sent for rule {rule_id}, test_mode: {test_mode}, recipient: {user_email}, status: {response.status_code}, message_id: {response_headers.get('X-Message-Id', 'Not provided')}, response_body: {response.body.decode('utf-8') if response.body else 'No body'}, headers: {response_headers}")
            
            if test_mode:
                return {
                    "results": [],
                    "email_content": content,
                    "status": "success",
                    "email_sent": True,
                    "message_id": response_headers.get('X-Message-Id', 'Not provided')
                }
            return
        
        efetch_xml = efetch(pmids, api_key=api_key)
        results = parse_efetch_xml(efetch_xml)
        
        context = "\n".join([f"Title: {r['title']}\nAbstract: {r['abstract'] or ''}\nAuthors: {r['authors']}\nJournal: {r['journal']}\nDate: {r['publication_date']}" for r in results])
        output = query_grok_api(prompt_text or "Summarize the provided research articles.", context)
        
        if email_format == "list":
            content = "\n".join([f"- {r['title']} ({r['publication_date']})\n  {r['abstract'][:100] or 'No abstract'}..." for r in results])
        elif email_format == "detailed":
            content = "\n".join([f"Title: {r['title']}\nAuthors: {r['authors']}\nJournal: {r['journal']}\nDate: {r['publication_date']}\nAbstract: {r['abstract'] or 'No abstract'}\n" for r in results])
        else:
            content = output
        
        if not sg:
            raise Exception("SendGrid API key not configured. Please contact support.")
        message = Mail(
            from_email=Email("noreply@firesidetechnologies.com"),
            to_emails=To(user_email),
            subject=f"PubMedResearcher {'Test ' if test_mode else ''}Notification: {rule_name}",
            plain_text_content=content
        )
        logger.info(f"Sending email for rule {rule_id}, recipient: {user_email}, subject: {message.subject}")
        response = sg.send(message)
        response_headers = {k: v for k, v in response.headers.items()}
        logger.info(f"Email sent for rule {rule_id}, test_mode: {test_mode}, recipient: {user_email}, status: {response.status_code}, message_id: {response_headers.get('X-Message-Id', 'Not provided')}, response_body: {response.body.decode('utf-8') if response.body else 'No body'}, headers: {response_headers}")
        
        if test_mode:
            return {
                "results": results,
                "email_content": content,
                "status": "success",
                "email_sent": True,
                "message_id": response_headers.get('X-Message-Id', 'Not provided')
            }
        
    except Exception as e:
        logger.error(f"Error running notification rule {rule_id}: {str(e)}\n{traceback.format_exc()}")
        if test_mode:
            try:
                if not sg:
                    raise Exception("SendGrid API key not configured. Please contact support.")
                message = Mail(
                    from_email=Email("noreply@firesidetechnologies.com"),
                    to_emails=To(user_email),
                    subject=f"PubMedResearcher Test Notification Failed: {rule_name}",
                    plain_text_content=f"Error testing notification rule: {str(e)}"
                )
                logger.info(f"Sending error email for rule {rule_id}, recipient: {user_email}, subject: {message.subject}")
                response = sg.send(message)
                response_headers = {k: v for k, v in response.headers.items()}
                logger.info(f"Error email sent for rule {rule_id}, test_mode: {test_mode}, recipient: {user_email}, status: {response.status_code}, message_id: {response_headers.get('X-Message-Id', 'Not provided')}, response_body: {response.body.decode('utf-8') if response.body else 'No body'}, headers: {response_headers}")
                email_sent = True
            except Exception as email_e:
                logger.error(f"Failed to send error email for rule {rule_id}: {str(email_e)}\n{traceback.format_exc()}")
                error_detail = ""
                if hasattr(email_e, 'body') and email_e.body:
                    try:
                        error_body = json.loads(email_e.body.decode('utf-8'))
                        error_detail = f": {error_body.get('errors', [{}])[0].get('message', 'No details provided')}"
                    except json.JSONDecodeError:
                        error_detail = f": {email_e.body.decode('utf-8')}"
                email_sent = False
            error_message = (
                f"Email sending failed due to unverified sender identity. Please verify noreply@firesidetechnologies.com in SendGrid{error_detail}."
                if "403" in str(e) or "Forbidden" in str(e)
                else "Email sending failed due to invalid API key configuration. Please contact support."
                if "Invalid header value" in str(e) or "API key not configured" in str(e)
                else f"Email sending failed: {str(e)}{error_detail}"
            )
            return {
                "results": [],
                "email_content": error_message,
                "status": "error",
                "email_sent": email_sent,
                "message_id": None
            }
        raise

def schedule_notification_rules():
    scheduler.remove_all_jobs()
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT n.id, n.user_id, n.rule_name, n.keywords, n.timeframe, n.prompt_text, n.email_format, u.email "
        "FROM notifications n JOIN users u ON n.user_id = u.id"
    )
    rules = cur.fetchall()
    cur.close()
    conn.close()
    
    for rule in rules:
        rule_id, user_id, rule_name, keywords, timeframe, prompt_text, email_format, user_email = rule
        cron_trigger = {
            'daily': CronTrigger(hour=8, minute=0),
            'weekly': CronTrigger(day_of_week='mon', hour=8, minute=0),
            'monthly': CronTrigger(day=1, hour=8, minute=0),
            'annually': CronTrigger(month=1, day=1, hour=8, minute=0)
        }[timeframe]
        scheduler.add_job(
            run_notification_rule,
            trigger=cron_trigger,
            args=[rule_id, user_id, rule_name, keywords, timeframe, prompt_text, email_format, user_email],
            id=f"notification_{rule_id}",
            replace_existing=True
        )
    logger.info(f"Scheduled {len(rules)} notification rules")

@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('search'))
    return render_template('index.html', username=None)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT id FROM users WHERE email = %s", (email,))
        if cur.fetchone():
            flash('Email already registered.', 'error')
        else:
            password_hash = generate_password_hash(password)
            cur.execute("INSERT INTO users (email, password_hash) VALUES (%s, %s)", (email, password_hash))
            conn.commit()
            flash('Registration successful! Please log in.', 'success')
            cur.close()
            conn.close()
            return redirect(url_for('login'))
        cur.close()
        conn.close()
    return render_template('register.html', username=None)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT id, email, password_hash FROM users WHERE email = %s", (email,))
        user = cur.fetchone()
        cur.close()
        conn.close()
        if user and check_password_hash(user[2], password):
            login_user(User(user[0], user[1]))
            return redirect(url_for('search'))
        flash('Invalid email or password.', 'error')
    return render_template('login.html', username=None)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

def extract_keywords_and_date(query, search_older=False, start_year=None):
    query_lower = query.lower()
    tokens = word_tokenize(query)
    tagged = pos_tag(tokens)
    stop_words = set(stopwords.words('english')).union({'what', 'can', 'tell', 'me', 'is', 'new', 'in', 'the', 'of', 'for', 'any', 'articles', 'that', 'show', 'between', 'only', 'related', 'to'})
    
    # Improved phrase detection
    keywords = []
    current_phrase = []
    for word, tag in tagged:
        if word.lower() in stop_words:
            if current_phrase:
                keywords.append('+'.join(current_phrase))
                current_phrase = []
            continue
        if tag.startswith('NN') or tag.startswith('JJ'):  # Nouns or adjectives
            current_phrase.append(word.lower())
        else:
            if current_phrase:
                keywords.append('+'.join(current_phrase))
                current_phrase = []
    if current_phrase:
        keywords.append('+'.join(current_phrase))
    
    # Fallback to single words if no phrases detected
    if not keywords:
        keywords = [word for word in query_lower.split() if word not in stop_words and len(word) > 1]
    
    keywords = keywords[:5]  # Limit to 5 keywords/phrases
    
    # Get synonyms
    keywords_with_synonyms = []
    for kw in keywords:
        synonyms = BIOMEDICAL_VOCAB.get(kw.lower(), [])[:2]  # Limit to 2 synonyms
        keywords_with_synonyms.append((kw, synonyms))
    
    today = datetime.now()
    default_start_year = today.year - 5
    date_range = None
    
    # Check for explicit date range in query
    if since_match := re.search(r'\bsince\s+(20\d{2})\b', query_lower):
        start_year = int(since_match.group(1))
        date_range = f"{start_year}/01/01[dp] TO {today.strftime('%Y/%m/%d')}[dp]"
    elif year_match := re.search(r'\b(20\d{2})\b', query_lower):
        year = int(year_match.group(1))
        date_range = f"{year}/01/01[dp] TO {year}/12/31[dp]"
    elif 'past year' in query_lower:
        date_range = f"{(today - timedelta(days=365)).strftime('%Y/%m/%d')}[dp] TO {today.strftime('%Y/%m/%d')}[dp]"
    elif 'past week' in query_lower:
        date_range = f"{(today - timedelta(days=7)).strftime('%Y/%m/%d')}[dp] TO {today.strftime('%Y/%m/%d')}[dp]"
    else:
        start_year_int = int(start_year) if search_older and start_year else default_start_year
        date_range = f"{start_year_int}/01/01[dp] TO {today.strftime('%Y/%m/%d')}[dp]"
    
    logger.info(f"Extracted keywords: {keywords_with_synonyms}, Date range: {date_range}")
    return keywords_with_synonyms, date_range, start_year_int if search_older and start_year else default_start_year

def build_pubmed_query(keywords_with_synonyms, date_range):
    query_parts = []
    for keyword, synonyms in keywords_with_synonyms:
        terms = [f'"{keyword}"[All Fields]'] + [f'"{syn}"[All Fields]' for syn in synonyms]
        term_query = " OR ".join(terms)
        query_parts.append(f"({term_query})")
    
    query = " AND ".join(query_parts) if query_parts else ""
    query = f"({query}) AND {date_range}" if query else date_range
    
    logger.info(f"Built PubMed query: {query}")
    return query

@sleep_and_retry
@limits(calls=10, period=1)
def esearch(query, retmax=20, api_key=None):
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": retmax,
        "retmode": "json",
        "sort": "relevance",
        "api_key": api_key
    }
    logger.info(f"PubMed ESearch query: {query}")
    response = requests.get(url, params=params)
    response.raise_for_status()
    result = response.json()
    logger.info(f"ESearch result: {len(result['esearchresult']['idlist'])} PMIDs")
    return result

@sleep_and_retry
@limits(calls=10, period=1)
def efetch(pmids, api_key=None):
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": "pubmed",
        "id": ",".join(map(str, pmids)),
        "retmode": "xml",
        "api_key": api_key
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.content

def parse_efetch_xml(xml_content):
    root = ElementTree.fromstring(xml_content)
    articles = []
    for article in root.findall(".//PubmedArticle"):
        pmid = article.find(".//PMID").text if article.find(".//PMID") is not None else ""
        title = article.find(".//ArticleTitle").text if article.find(".//ArticleTitle") is not None else ""
        abstract = article.find(".//AbstractText")
        abstract = abstract.text or "" if abstract is not None else ""
        authors = [author.find("LastName").text for author in article.findall(".//Author") 
                   if author.find("LastName") is not None]
        journal = article.find(".//Journal/Title")
        journal = journal.text if journal is not None else ""
        pub_date = article.find(".//PubDate/Year")
        pub_date = pub_date.text if pub_date is not None else ""
        logger.info(f"Parsed article: PMID={pmid}, Date={pub_date}, Abstract={'Present' if abstract else 'Missing'}")
        articles.append({
            "id": pmid,
            "title": title,
            "abstract": abstract,
            "authors": ", ".join(authors),
            "journal": journal,
            "publication_date": pub_date
        })
    return articles

def parse_prompt(prompt_text):
    if not prompt_text:
        return {
            'summary_result_count': 20,
            'display_result_count': 20,
            'limit_presentation': False
        }
    
    prompt_text_lower = prompt_text.lower()
    
    summary_result_count = 20
    if match := re.search(r'(?:top|return|summarize|include|limit\s+to|show\s+only)\s+(\d+)\s+(?:articles|results)', prompt_text_lower):
        summary_result_count = min(int(match.group(1)), 20)
    elif 'top' in prompt_text_lower:
        summary_result_count = 3
    
    display_result_count = 20
    limit_presentation = ('show only' in prompt_text_lower or 'present only' in prompt_text_lower)
    
    logger.info(f"Parsed prompt: summary_result_count={summary_result_count}, display_result_count={display_result_count}, limit_presentation={limit_presentation}")
    
    return {
        'summary_result_count': summary_result_count,
        'display_result_count': display_result_count,
        'limit_presentation': limit_presentation
    }

def rank_results(query, results, prompt_params=None):
    display_result_count = prompt_params.get('display_result_count', 20) if prompt_params else 20
    
    try:
        articles_context = []
        for i, result in enumerate(results):
            article_text = f"Article {i+1}: Title: {result['title']}\nAbstract: {result['abstract'] or ''}\nAuthors: {result['authors']}\nJournal: {result['journal']}\nDate: {result['publication_date']}"
            articles_context.append(article_text)
        
        context = "\n\n".join(articles_context)
        ranking_prompt = f"""
Given the query '{query}', rank the following articles by relevance.
Focus on articles that directly address the query's topic and intent.
Exclude articles that are unrelated to the query.
Return a JSON list of article indices (1-based) in order of relevance, with a brief explanation for each.
Ensure the response is valid JSON. Example:
[
    {{"index": 1, "explanation": "Directly discusses the query topic with high relevance"}},
    {{"index": 2, "explanation": "Relevant but less specific to the query"}}
]
Articles:
{context}
"""
        cache_key = hashlib.md5((query + context + ranking_prompt).encode()).hexdigest()
        response = query_grok_api(query, context, prompt=ranking_prompt)
        if not response:
            raise ValueError("Grok API returned None")
        
        logger.info(f"Grok ranking response: {response[:200]}...")
        ranking = json.loads(response)
        if isinstance(ranking, dict) and 'articles' in ranking:
            ranking = ranking['articles']
        if not isinstance(ranking, list):
            raise ValueError("Grok response is not a list")
        
        ranked_indices = []
        for item in ranking:
            if isinstance(item, dict) and 'index' in item:
                index = item['index']
                if isinstance(index, (int, str)) and str(index).isdigit():
                    index = int(index) - 1
                    if 0 <= index < len(results):
                        ranked_indices.append(index)
        
        missing_indices = [i for i in range(len(results)) if i not in ranked_indices]
        ranked_indices.extend(missing_indices)
        
        ranked_results = [results[i] for i in ranked_indices[:display_result_count]]
        logger.info(f"Grok ranked {len(ranked_results)} results: indices {ranked_indices[:display_result_count]}")
        return ranked_results
    except Exception as e:
        logger.error(f"Grok ranking failed: {str(e)}")
        return embedding_based_ranking(query, results, prompt_params)

def embedding_based_ranking(query, results, prompt_params=None):
    display_result_count = prompt_params.get('display_result_count', 20) if prompt_params else 20
    query_embedding = generate_embedding(query)
    current_year = datetime.now().year
    
    embeddings = []
    texts = []
    for result in results:
        embedding = get_cached_embedding(result['id'])
        if embedding is None:
            texts.append(f"{result['title']} {result['abstract'] or ''}")
        else:
            embeddings.append(embedding)
    
    if texts:
        model = load_embedding_model()
        new_embeddings = model.encode(texts, convert_to_numpy=True)
        for i, (result, emb) in enumerate(zip(results[len(embeddings):], new_embeddings)):
            cache_embedding(result['id'], emb)
            embeddings.append(emb)
    
    scores = []
    for i, (emb, result) in enumerate(zip(embeddings, results)):
        similarity = 1 - cosine(query_embedding, emb) if emb is not None else 0.0
        pub_year = int(result['publication_date']) if result['publication_date'].isdigit() else 2000
        recency_bonus = (pub_year - 2000) / (current_year - 2000)
        weighted_score = (0.8 * similarity) + (0.2 * recency_bonus)
        scores.append((i, weighted_score, pub_year))
    
    scores.sort(key=lambda x: (x[1], x[2]), reverse=True)
    ranked_indices = [i for i, _, _ in scores]
    
    ranked_results = [results[i] for i in ranked_indices[:display_result_count]]
    logger.info(f"Embedding-based ranked {len(ranked_results)} results with indices {ranked_indices[:display_result_count]}")
    return ranked_results

def generate_prompt_output(query, results, prompt_text, prompt_params, is_fallback=False):
    if not results:
        return f"No results found for '{query}'{' outside the specified timeframe' if is_fallback else ''}."
    
    logger.info(f"Initial results count: {len(results)}, is_fallback: {is_fallback}")
    
    summary_result_count = prompt_params.get('summary_result_count', 20) if prompt_params else 20
    context_results = results[:summary_result_count]
    logger.info(f"Context results count for summary: {len(context_results)}")
    
    if not context_results:
        return f"No results found for '{query}'{' outside the specified timeframe' if is_fallback else ''} matching criteria."
    
    context = "\n".join([f"Title: {r['title']}\nAbstract: {r['abstract'] or ''}\nAuthors: {r['authors']}\nJournal: {r['journal']}\nDate: {r['publication_date']}" for r in context_results])
    cache_key = hashlib.md5((query + context + prompt_text).encode()).hexdigest()
    try:
        output = query_grok_api(prompt_text, context, prompt=prompt_text)
        if not output:
            logger.warning("Grok API returned None for summary, using fallback")
            output = f"Fallback: Unable to generate AI summary. Top {summary_result_count} results include: " + "; ".join([f"{r['title']} ({r['publication_date']})" for r in context_results])
    except Exception as e:
        logger.error(f"Error generating AI summary: {str(e)}")
        output = f"Fallback: Unable to generate AI summary due to error: {str(e)}. Top {summary_result_count} results include: " + "; ".join([f"{r['title']} ({r['publication_date']})" for r in context_results])
    
    paragraphs = output.split('\n\n')
    formatted_output = ''.join(f'<p>{p}</p>' for p in paragraphs if p.strip())
    logger.info(f"Generated prompt output: length={len(formatted_output)}, is_fallback: {is_fallback}")
    return formatted_output

@app.route('/search_progress', methods=['GET'])
@login_required
def search_progress():
    def stream_progress():
        try:
            if not current_user or not hasattr(current_user, 'id'):
                yield f"data: {{'status': 'error: User not authenticated'}}\n\n"
                return
            query = request.args.get('query', '')
            last_status = None
            while True:
                status, timestamp = get_search_progress(current_user.id, query)
                if status and status != last_status:
                    yield f"data: {{'status': '{status}'}}\n\n"
                    last_status = status
                if status in ["complete", "error"]:
                    break
                time.sleep(1)
        except Exception as e:
            logger.error(f"Error in search_progress: {str(e)}")
            yield f"data: {{'status': 'error: {str(e)}'}}\n\n"
    
    return Response(stream_progress(), mimetype='text/event-stream')

@app.route('/search', methods=['GET', 'POST'])
@login_required
def search():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('SELECT id, prompt_name, prompt_text FROM prompts WHERE user_id = %s', (current_user.id,))
    prompts = [{'id': str(p[0]), 'prompt_name': p[1], 'prompt_text': p[2]} for p in cur.fetchall()]
    cur.close()
    conn.close()
    logger.info(f"Loaded prompts: {len(prompts)} prompts for user {current_user.id}")

    prompt_id = request.args.get('prompt_id', request.form.get('prompt_id', ''))
    prompt_text = request.args.get('prompt_text', request.form.get('prompt_text', ''))
    query = request.args.get('query', request.form.get('query', ''))
    search_older = request.form.get('search_older', 'off') == 'on'
    start_year = request.form.get('start_year', None)

    selected_prompt_text = prompt_text
    if prompt_id and not prompt_text:
        for prompt in prompts:
            if prompt['id'] == prompt_id:
                selected_prompt_text = prompt['prompt_text']
                break
        else:
            logger.warning(f"Prompt ID {prompt_id} not found in prompts")
            selected_prompt_text = ''

    logger.info(f"Search request: prompt_id={prompt_id}, prompt_text={prompt_text[:50]}..., selected_prompt_text={selected_prompt_text[:50]}..., query={query[:50]}..., search_older={search_older}, start_year={start_year}")

    if request.method == 'POST':
        if not query:
            update_search_progress(current_user.id, query, "error: Query cannot be empty")
            return render_template('search.html', error="Query cannot be empty", prompts=prompts, prompt_id=prompt_id, prompt_text=selected_prompt_text, results=[], fallback_results=[], prompt_output='', fallback_prompt_output='', username=current_user.email, has_prompt=bool(selected_prompt_text), prompt_params={}, summary_result_count=20, search_older=search_older, start_year=start_year)
        
        update_search_progress(current_user.id, query, "contacting PubMed")
        
        keywords_with_synonyms, date_range, start_year_int = extract_keywords_and_date(query, search_older, start_year)
        if not keywords_with_synonyms:
            update_search_progress(current_user.id, query, "error: No valid keywords found")
            return render_template('search.html', error="No valid keywords found", prompts=prompts, prompt_id=prompt_id, prompt_text=selected_prompt_text, results=[], fallback_results=[], prompt_output='', fallback_prompt_output='', username=current_user.email, has_prompt=bool(selected_prompt_text), prompt_params={}, summary_result_count=20, search_older=search_older, start_year=start_year)
        
        prompt_params = parse_prompt(selected_prompt_text)
        summary_result_count = prompt_params['summary_result_count']
        display_result_count = prompt_params['display_result_count']
        
        api_key = os.environ.get('PUBMED_API_KEY')
        try:
            search_query = build_pubmed_query(keywords_with_synonyms, date_range)
        except Exception as e:
            logger.error(f"Error building PubMed query: {str(e)}")
            update_search_progress(current_user.id, query, f"error: Query processing failed: {str(e)}")
            return render_template('search.html', error=f"Query processing failed: {str(e)}", prompts=prompts, prompt_id=prompt_id, prompt_text=selected_prompt_text, results=[], fallback_results=[], prompt_output='', fallback_prompt_output='', username=current_user.email, has_prompt=bool(selected_prompt_text), prompt_params={}, summary_result_count=20, search_older=search_older, start_year=start_year)
        
        try:
            update_search_progress(current_user.id, query, "executing search")
            esearch_result = esearch(search_query, retmax=20, api_key=api_key)
            pmids = esearch_result['esearchresult']['idlist']
            results = []
            if pmids:
                update_search_progress(current_user.id, query, "fetching article details")
                efetch_xml = efetch(pmids, api_key=api_key)
                results = parse_efetch_xml(efetch_xml)
            
            # Filter results by date range
            primary_results = [
                r for r in results
                if r['publication_date'] and r['publication_date'].isdigit() and int(r['publication_date']) >= start_year_int
            ]
            fallback_results = [
                r for r in results
                if r['publication_date'] and r['publication_date'].isdigit() and int(r['publication_date']) < start_year_int
            ]
            
            if primary_results or fallback_results:
                conn = get_db_connection()
                cur = conn.cursor()
                cur.execute("INSERT INTO search_cache (query, results) VALUES (%s, %s)", 
                            (query, json.dumps(primary_results + fallback_results)))
                conn.commit()
                cur.close()
                conn.close()
            
            if not primary_results and not fallback_results:
                logger.info("No results found")
                update_search_progress(current_user.id, query, "error: No results found")
                return render_template('search.html', error="No results found. Try broadening your query.", prompts=prompts, prompt_id=prompt_id, prompt_text=selected_prompt_text, results=[], fallback_results=[], prompt_output='', fallback_prompt_output='', username=current_user.email, has_prompt=bool(selected_prompt_text), prompt_params={}, summary_result_count=20, search_older=search_older, start_year=start_year)
            
            update_search_progress(current_user.id, query, "ranking results")
            ranked_results = []
            ranked_fallback_results = []
            if primary_results:
                ranked_results = rank_results(query, primary_results, prompt_params)
            if fallback_results and not primary_results:
                ranked_fallback_results = rank_results(query, fallback_results, prompt_params)
            
            logger.info(f"Ranked results: {len(ranked_results)} primary, {len(ranked_fallback_results)} fallback")
            
            update_search_progress(current_user.id, query, "complete" if not selected_prompt_text else "awaiting_summary")
            
            return render_template(
                'search.html', 
                results=ranked_results,
                fallback_results=ranked_fallback_results,
                query=query, 
                prompts=prompts, 
                prompt_id=prompt_id,
                prompt_text=selected_prompt_text,
                prompt_output='' if selected_prompt_text else None,
                fallback_prompt_output='' if selected_prompt_text else None,
                summary_result_count=summary_result_count,
                target_year=None,
                username=current_user.email,
                has_prompt=bool(selected_prompt_text),
                prompt_params=prompt_params,
                search_older=search_older,
                start_year=start_year
            )
        except Exception as e:
            logger.error(f"PubMed API error: {str(e)}")
            update_search_progress(current_user.id, query, f"error: Search failed: {str(e)}")
            return render_template('search.html', error=f"Search failed: {str(e)}", prompts=prompts, prompt_id=prompt_id, prompt_text=selected_prompt_text, results=[], fallback_results=[], prompt_output='', fallback_prompt_output='', username=current_user.email, has_prompt=bool(selected_prompt_text), prompt_params={}, summary_result_count=20, search_older=search_older, start_year=start_year)
    
    return render_template('search.html', prompts=prompts, prompt_id=prompt_id, prompt_text=selected_prompt_text, results=[], fallback_results=[], prompt_output='', fallback_prompt_output='', username=current_user.email, has_prompt=bool(selected_prompt_text), prompt_params={}, summary_result_count=20, search_older=False, start_year=None)

@app.route('/search_summary', methods=['POST'])
@login_required
def search_summary():
    query = request.form.get('query', '')
    prompt_text = request.form.get('prompt_text', '')
    results = json.loads(request.form.get('results', '[]'))
    fallback_results = json.loads(request.form.get('fallback_results', '[]'))
    prompt_params = json.loads(request.form.get('prompt_params', '{}'))
    
    try:
        prompt_output = generate_prompt_output(query, results, prompt_text, prompt_params)
        fallback_prompt_output = generate_prompt_output(query, fallback_results, prompt_text, prompt_params, is_fallback=True) if fallback_results else ''
        
        if prompt_output.startswith("Fallback:"):
            flash("AI summarization failed for primary results.", "warning")
        if fallback_prompt_output and fallback_prompt_output.startswith("Fallback:"):
            flash("AI summarization failed for fallback results.", "warning")
        
        update_search_progress(current_user.id, query, "complete")
        
        return jsonify({
            'status': 'success',
            'prompt_output': prompt_output,
            'fallback_prompt_output': fallback_prompt_output
        })
    except Exception as e:
        logger.error(f"Error generating AI summary: {str(e)}")
        update_search_progress(current_user.id, query, f"error: AI summary failed: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f"AI summary failed: {str(e)}"
        })

@app.route('/prompt', methods=['GET', 'POST'])
@login_required
def prompt():
    if request.method == 'POST':
        prompt_name = request.form.get('prompt_name')
        prompt_text = request.form.get('prompt_text')
        if not prompt_name or not prompt_text:
            flash('Prompt name and text cannot be empty.', 'error')
        else:
            conn = get_db_connection()
            cur = conn.cursor()
            try:
                cur.execute('INSERT INTO prompts (user_id, prompt_name, prompt_text) VALUES (%s, %s, %s)', 
                            (current_user.id, prompt_name, prompt_text))
                conn.commit()
                flash('Prompt saved successfully.', 'success')
            except Exception as e:
                conn.rollback()
                flash(f'Failed to save prompt: {str(e)}', 'error')
            finally:
                cur.close()
                conn.close()
    
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('SELECT id, prompt_name, prompt_text, created_at FROM prompts WHERE user_id = %s ORDER BY created_at DESC', 
                (current_user.id,))
    prompts = [{'id': str(p[0]), 'prompt_name': p[1], 'prompt_text': p[2], 'created_at': p[3]} for p in cur.fetchall()]
    cur.close()
    conn.close()
    return render_template('prompt.html', prompts=prompts, username=current_user.email)

@app.route('/prompt/edit/<int:id>', methods=['GET', 'POST'])
@login_required
def edit_prompt(id):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('SELECT id, prompt_name, prompt_text FROM prompts WHERE id = %s AND user_id = %s', 
                (id, current_user.id))
    prompt = cur.fetchone()
    
    if not prompt:
        cur.close()
        conn.close()
        flash('Prompt not found or you do not have permission to edit it.', 'error')
        return redirect(url_for('prompt'))
    
    if request.method == 'POST':
        prompt_name = request.form.get('prompt_name')
        prompt_text = request.form.get('prompt_text')
        if not prompt_name or not prompt_text:
            flash('Prompt name and text cannot be empty.', 'error')
        else:
            try:
                cur.execute('UPDATE prompts SET prompt_name = %s, prompt_text = %s WHERE id = %s AND user_id = %s', 
                            (prompt_name, prompt_text, id, current_user.id))
                conn.commit()
                flash('Prompt updated successfully.', 'success')
                cur.close()
                conn.close()
                return redirect(url_for('prompt'))
            except Exception as e:
                conn.rollback()
                flash(f'Failed to update prompt: {str(e)}', 'error')
    
    cur.close()
    conn.close()
    return render_template('prompt_edit.html', prompt={'id': prompt[0], 'prompt_name': prompt[1], 'prompt_text': prompt[2]}, username=current_user.email)

@app.route('/prompt/delete/<int:id>', methods=['POST'])
@login_required
def delete_prompt(id):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('SELECT id FROM prompts WHERE id = %s AND user_id = %s', (id, current_user.id))
    prompt = cur.fetchone()
    
    if not prompt:
        cur.close()
        conn.close()
        flash('Prompt not found or you do not have permission to delete it.', 'error')
        return redirect(url_for('prompt'))
    
    try:
        cur.execute('DELETE FROM prompts WHERE id = %s AND user_id = %s', (id, current_user.id))
        conn.commit()
        flash('Prompt deleted successfully.', 'success')
    except Exception as e:
        conn.rollback()
        flash(f'Failed to delete prompt: {str(e)}', 'error')
    finally:
        cur.close()
        conn.close()
    
    return redirect(url_for('prompt'))

@app.route('/notifications', methods=['GET', 'POST'])
@login_required
def notifications():
    conn = get_db_connection()
    cur = conn.cursor()
    if request.method == 'POST':
        rule_name = request.form.get('rule_name')
        keywords = request.form.get('keywords')
        timeframe = request.form.get('timeframe')
        prompt_text = request.form.get('prompt_text')
        email_format = request.form.get('email_format')
        
        if not all([rule_name, keywords, timeframe, email_format]):
            flash('All fields except prompt text are required.', 'error')
        elif timeframe not in ['daily', 'weekly', 'monthly', 'annually']:
            flash('Invalid timeframe selected.', 'error')
        elif email_format not in ['summary', 'list', 'detailed']:
            flash('Invalid email format selected.', 'error')
        else:
            try:
                cur.execute(
                    "INSERT INTO notifications (user_id, rule_name, keywords, timeframe, prompt_text, email_format) "
                    "VALUES (%s, %s, %s, %s, %s, %s)",
                    (current_user.id, rule_name, keywords, timeframe, prompt_text, email_format)
                )
                conn.commit()
                flash('Notification rule created successfully.', 'success')
                schedule_notification_rules()
            except Exception as e:
                conn.rollback()
                logger.error(f"Error creating notification: {str(e)}")
                flash(f'Failed to create notification rule: {str(e)}', 'error')
    
    cur.execute(
        "SELECT id, rule_name, keywords, timeframe, prompt_text, email_format, created_at "
        "FROM notifications WHERE user_id = %s ORDER BY created_at DESC",
        (current_user.id,)
    )
    notifications = [
        {
            'id': n[0],
            'rule_name': n[1],
            'keywords': n[2],
            'timeframe': n[3],
            'prompt_text': n[4],
            'email_format': n[5],
            'created_at': n[6]
        } for n in cur.fetchall()
    ]
    cur.close()
    conn.close()
    return render_template('notifications.html', notifications=notifications, username=current_user.email)

@app.route('/notifications/edit/<int:id>', methods=['GET', 'POST'])
@login_required
def edit_notification(id):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, rule_name, keywords, timeframe, prompt_text, email_format "
        "FROM notifications WHERE id = %s AND user_id = %s",
        (id, current_user.id)
    )
    notification = cur.fetchone()
    
    if not notification:
        cur.close()
        conn.close()
        flash('Notification rule not found or you do not have permission to edit it.', 'error')
        return redirect(url_for('notifications'))
    
    if request.method == 'POST':
        rule_name = request.form.get('rule_name')
        keywords = request.form.get('keywords')
        timeframe = request.form.get('timeframe')
        prompt_text = request.form.get('prompt_text')
        email_format = request.form.get('email_format')
        
        if not all([rule_name, keywords, timeframe, email_format]):
            flash('All fields except prompt text are required.', 'error')
        elif timeframe not in ['daily', 'weekly', 'monthly', 'annually']:
            flash('Invalid timeframe selected.', 'error')
        elif email_format not in ['summary', 'list', 'detailed']:
            flash('Invalid email format selected.', 'error')
        else:
            try:
                cur.execute(
                    "UPDATE notifications SET rule_name = %s, keywords = %s, timeframe = %s, prompt_text = %s, email_format = %s "
                    "WHERE id = %s AND user_id = %s",
                    (rule_name, keywords, timeframe, prompt_text, email_format, id, current_user.id)
                )
                conn.commit()
                flash('Notification rule updated successfully.', 'success')
                schedule_notification_rules()
                cur.close()
                conn.close()
                return redirect(url_for('notifications'))
            except Exception as e:
                conn.rollback()
                logger.error(f"Error updating notification: {str(e)}")
                flash(f'Failed to update notification rule: {str(e)}', 'error')
    
    cur.close()
    conn.close()
    return render_template('notification_edit.html', notification={
        'id': notification[0],
        'rule_name': notification[1],
        'keywords': notification[2],
        'timeframe': notification[3],
        'prompt_text': notification[4],
        'email_format': notification[5]
    }, username=current_user.email)

@app.route('/notifications/delete/<int:id>', methods=['POST'])
@login_required
def delete_notification(id):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('SELECT id FROM notifications WHERE id = %s AND user_id = %s', (id, current_user.id))
    notification = cur.fetchone()
    
    if not notification:
        cur.close()
        conn.close()
        flash('Notification rule not found or you do not have permission to delete it.', 'error')
        return redirect(url_for('notifications'))
    
    try:
        cur.execute('DELETE FROM notifications WHERE id = %s AND user_id = %s', (id, current_user.id))
        conn.commit()
        flash('Notification rule deleted successfully.', 'success')
        schedule_notification_rules()
    except Exception as e:
        conn.rollback()
        logger.error(f"Error deleting notification: {str(e)}")
        flash(f'Failed to delete notification rule: {str(e)}', 'error')
    finally:
        cur.close()
        conn.close()
    
    return redirect(url_for('notifications'))

@app.route('/notifications/test/<int:id>', methods=['GET'])
@login_required
def test_notification(id):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT n.id, n.user_id, n.rule_name, n.keywords, n.timeframe, n.prompt_text, n.email_format, u.email "
        "FROM notifications n JOIN users u ON n.user_id = u.id WHERE n.id = %s AND n.user_id = %s",
        (id, current_user.id)
    )
    rule = cur.fetchone()
    cur.close()
    conn.close()
    
    if not rule:
        flash('Notification rule not found or you do not have permission to test it.', 'error')
        return jsonify({"status": "error", "message": "Rule not found"}), 404
    
    rule_id, user_id, rule_name, keywords, timeframe, prompt_text, email_format, user_email = rule
    try:
        test_result = run_notification_rule(
            rule_id, user_id, rule_name, keywords, timeframe, prompt_text, email_format, user_email, test_mode=True
        )
        return jsonify(test_result)
    except Exception as e:
        logger.error(f"Error testing notification rule {rule_id}: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/notifications/test_email', methods=['POST'])
@login_required
def test_email():
    if not sg:
        logger.error("SendGrid API key not configured for test email")
        return jsonify({"status": "error", "message": "SendGrid API key not configured"}), 500
    
    email = request.form.get('email', current_user.email)
    if not validate_user_email(email):
        logger.error(f"Invalid email for test: {email}")
        return jsonify({"status": "error", "message": f"Invalid email address: {email}"}), 400
    
    try:
        message = Mail(
            from_email=Email("noreply@firesidetechnologies.com"),
            to_emails=To(email),
            subject="PubMedResearcher Test Email",
            plain_text_content="This is a test email to verify SendGrid integration."
        )
        logger.info(f"Sending test email, recipient: {email}, subject: {message.subject}")
        response = sg.send(message)
        response_headers = {k: v for k, v in response.headers.items()}
        logger.info(f"Test email sent, recipient: {email}, status: {response.status_code}, message_id: {response_headers.get('X-Message-Id', 'Not provided')}, response_body: {response.body.decode('utf-8') if response.body else 'No body'}, headers: {response_headers}")
        return jsonify({
            "status": "success",
            "message": f"Test email sent to {email}. Check your inbox and spam/junk folder.",
            "message_id": response_headers.get('X-Message-Id', 'Not provided')
        })
    except Exception as e:
        logger.error(f"Error sending test email: {str(e)}\n{traceback.format_exc()}")
        error_detail = ""
        if hasattr(e, 'body') and e.body:
            try:
                error_body = json.loads(e.body.decode('utf-8'))
                error_detail = f": {error_body.get('errors', [{}])[0].get('message', 'No details provided')}"
            except json.JSONDecodeError:
                error_detail = f": {e.body.decode('utf-8')}"
        return jsonify({"status": "error", "message": f"Failed to send test email: {str(e)}{error_detail}"}), 500

# Schedule notifications on startup
schedule_notification_rules()

if __name__ == '__main__':
    app.run(debug=True)