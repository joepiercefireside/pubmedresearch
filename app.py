from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, Response
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import psycopg2
from werkzeug.security import generate_password_hash, check_password_hash
import spacy
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
from datetime import datetime
from collections import Counter
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import sendgrid
from sendgrid.helpers.mail import Mail, Email, To, Content
from openai import OpenAI
import traceback
import openai
import httpx.__version__
import time

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'your-secret-key')
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Log dependency versions
logger.info(f"openai version: {openai.__version__}")
logger.info(f"httpx version: {httpx.__version__}")

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Initialize embedding model for fallback ranking
embedding_model = None

# Initialize scheduler
scheduler = BackgroundScheduler()
scheduler.start()

# SendGrid client
sg = sendgrid.SendGridAPIClient(os.environ.get('SENDGRID_API_KEY'))

# Initialize SQLite database for search progress
def init_progress_db():
    conn = sqlite3.connect('search_progress.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS search_progress
                 (user_id TEXT, query TEXT, status TEXT, timestamp REAL)''')
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

# xAI Grok API call with increased retries
def query_grok_api(query, context, prompt="Process the provided context according to the user's prompt."):
    try:
        api_key = os.environ.get('XAI_API_KEY')
        if not api_key:
            logger.error("XAI_API_KEY not set")
            return "Error: xAI API key not configured"
        
        client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
        logger.info(f"Sending Grok API request: prompt={query[:50]}..., context_length={len(context)}")
        
        completion = client.chat.completions.create(
            model="grok-3-latest",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Based on the following context, answer the prompt: {query}\n\nContext: {context}"}
            ],
            max_tokens=1000,
            timeout=30  # Increase timeout
        )
        response = completion.choices[0].message.content
        logger.info(f"Grok API response received: length={len(response)}")
        return response
    except Exception as e:
        logger.error(f"Error querying xAI Grok API: {str(e)}\n{traceback.format_exc()}")
        # Fallback: Basic requests-based API call
        try:
            url = "https://api.x.ai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "model": "grok-3-latest",
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"Based on the following context, answer the prompt: {query}\n\nContext: {context}"}
                ],
                "max_tokens": 1000
            }
            response = requests.post(url, headers=headers, json=data, proxies=None, timeout=30)
            response.raise_for_status()
            response_text = response.json()['choices'][0]['message']['content']
            logger.info(f"Fallback Grok API response received: length={len(response_text)}")
            return response_text
        except Exception as fallback_e:
            logger.error(f"Fallback API call failed: {str(fallback_e)}")
            return f"Fallback: Unable to generate AI summary. Please check API key or endpoint."

# Database connection
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

def run_notification_rule(rule_id, user_id, rule_name, keywords, timeframe, prompt_text, email_format, user_email):
    logger.info(f"Running notification rule {rule_id} ({rule_name}) for user {user_id}, keywords: {keywords}")
    keywords_list = [k.strip() for k in keywords.split(',')]
    date_filters = {
        'daily': 'last+1+day[dp]',
        'weekly': 'last+7+days[dp]',
        'monthly': 'last+30+days[dp]',
        'annually': 'last+365+days[dp]'
    }
    intent = {'date': date_filters[timeframe]}
    search_query = build_pubmed_query(keywords_list, intent)
    try:
        api_key = os.environ.get('PUBMED_API_KEY')
        esearch_result = esearch(search_query, retmax=20, api_key=api_key)
        pmids = esearch_result['esearchresult']['idlist']
        if not pmids:
            logger.info(f"No new results for rule {rule_id}")
            return
        
        efetch_xml = efetch(pmids, api_key=api_key)
        results = parse_efetch_xml(efetch_xml)
        
        # Use xAI Grok API for prompt processing
        context = "\n".join([f"Title: {r['title']}\nAbstract: {r['abstract']}\nAuthors: {r['authors']}\nJournal: {r['journal']}\nDate: {r['publication_date']}" for r in results])
        output = query_grok_api(prompt_text or "Summarize the provided research articles.", context)
        
        if email_format == "list":
            content = "\n".join([f"- {r['title']} ({r['publication_date']})\n  {r['abstract'][:100]}..." for r in results])
        elif email_format == "detailed":
            content = "\n".join([f"Title: {r['title']}\nAuthors: {r['authors']}\nJournal: {r['journal']}\nDate: {r['publication_date']}\nAbstract: {r['abstract']}\n" for r in results])
        else:
            content = output
        
        message = Mail(
            from_email=Email("notifications@pubmedresearcher.com"),
            to_emails=To(user_email),
            subject=f"PubMedResearcher Notification: {rule_name}",
            plain_text_content=content
        )
        response = sg.send(message)
        logger.info(f"Email sent for rule {rule_id}, status: {response.status_code}")
    except Exception as e:
        logger.error(f"Error running notification rule {rule_id}: {str(e)}")

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

def extract_keywords_and_intent(query):
    doc = nlp(query.lower())
    stop_words = nlp.Defaults.stop_words | {'about', 'articles', 'from', 'on', 'this', 'year', 'provide', 'summary', 'recent', 'months', 'treatments'}
    
    # Extract keywords
    keywords = []
    for chunk in doc.noun_chunks:
        phrase = chunk.text.strip()
        if (len(phrase) > 1 and 
            all(token.text not in stop_words and token.is_alpha for token in nlp(phrase)) and
            not any(word in phrase for word in ['recent', 'latest', 'new'])):
            keywords.append(phrase)
    
    for token in doc:
        if (token.is_alpha and 
            token.text not in stop_words and 
            len(token.text) > 1 and 
            not any(token.text in phrase for phrase in keywords)):
            keywords.append(token.text)
    
    query_lower = query.lower()
    if 'cidp' in query_lower or 'chronic inflammatory demyelinating polyneuropathy' in query_lower:
        if 'chronic inflammatory demyelinating polyneuropathy' not in keywords:
            keywords = ['chronic inflammatory demyelinating polyneuropathy'] + [k for k in keywords if k != 'cidp']
    
    keywords = keywords[:3]
    
    # Extract intent, timeframe, and author
    intent = {'focus': None, 'date': None, 'author': None}
    current_year = str(datetime.now().year)
    current_date = datetime.now().strftime('%Y/%m/%d')
    
    # Intent detection
    if 'treatment' in query_lower or 'therapy' in query_lower or 'therapeutic' in query_lower:
        intent['focus'] = 'treatment'
    elif 'diagnosis' in query_lower or 'diagnostic' in query_lower:
        intent['focus'] = 'diagnosis'
    elif 'prevention' in query_lower or 'preventive' in query_lower:
        intent['focus'] = 'prevention'
    elif 'review' in query_lower or 'meta-analysis' in query_lower:
        intent['focus'] = 'review'
    
    # Author detection
    author_match = re.search(r'(?:by|author:?)\s+([\w\s]+?)(?:\s+(?:in|from|since|\d{4}|$))', query_lower)
    if author_match:
        intent['author'] = author_match.group(1).strip()
    
    # Timeframe detection
    if 'this year' in query_lower or 'current year' in query_lower:
        intent['date'] = f"{current_year}[dp]"
    elif 'last year' in query_lower or 'previous year' in query_lower:
        intent['date'] = f"{int(current_year)-1}[dp]"
    elif year_match := re.search(r'\b(20\d{2})\b', query_lower):
        intent['date'] = f"{year_match.group(1)}[dp]"
    elif 'past month' in query_lower or 'last month' in query_lower or 'last 30 days' in query_lower:
        intent['date'] = 'last+30+days[dp]'
    elif 'past week' in query_lower or 'last week' in query_lower or 'last 7 days' in query_lower:
        intent['date'] = 'last+7+days[dp]'
    elif 'past day' in query_lower or 'last day' in query_lower or 'yesterday' in query_lower:
        intent['date'] = 'last+1+day[dp]'
    elif since_match := re.search(r'since\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})', query_lower):
        month = {'january': '01', 'february': '02', 'march': '03', 'april': '04', 'may': '05', 'june': '06',
                 'july': '07', 'august': '08', 'september': '09', 'october': '10', 'november': '11', 'december': '12'}[since_match.group(1)]
        year = since_match.group(2)
        intent['date'] = f"{year}/{month}/01:{current_date}[dp]"
    elif quarter_match := re.search(r'(?:last|past)\s+(q[1-4]|quarter\s*[1-4])\s*(?:of\s*(\d{4}))?', query_lower):
        quarter = quarter_match.group(1).replace('quarter', 'q').lower()
        year = quarter_match.group(2) or current_year
        quarter_ranges = {
            'q1': f"{year}/01/01:{year}/03/31[dp]",
            'q2': f"{year}/04/01:{year}/06/30[dp]",
            'q3': f"{year}/07/01:{year}/09/30[dp]",
            'q4': f"{year}/10/01:{year}/12/31[dp]"
        }
        intent['date'] = quarter_ranges[quarter]
    elif half_match := re.search(r'(?:first|second)\s+half\s+of\s+(\d{4})', query_lower):
        year = half_match.group(1)
        if 'first' in query_lower:
            intent['date'] = f"{year}/01/01:{year}/06/30[dp]"
        else:
            intent['date'] = f"{year}/07/01:{year}/12/31[dp]"
    elif re.search(r'(?:last|past)\s+(\d+)\s+years?', query_lower):
        years = re.search(r'(?:last|past)\s+(\d+)\s+years?', query_lower).group(1)
        intent['date'] = f"last+{years}+years[dp]"
    elif re.search(r'(?:last|past)\s+(\d+)\s+months?', query_lower):
        months = re.search(r'(?:last|past)\s+(\d+)\s+months?', query_lower).group(1)
        intent['date'] = f"last+{int(months)*30}+days[dp]"
    elif re.search(r'(?:last|past)\s+(\d+)\s+weeks?', query_lower):
        weeks = re.search(r'(?:last|past)\s+(\d+)\s+weeks?', query_lower).group(1)
        intent['date'] = f"last+{int(weeks)*7}+days[dp]"
    elif re.search(r'(?:last|past)\s+(\d+)\s+days?', query_lower):
        days = re.search(r'(?:last|past)\s+(\d+)\s+days?', query_lower).group(1)
        intent['date'] = f"last+{days}+days[dp]"
    elif 'recent' in query_lower:
        intent['date'] = 'last+5+years[dp]'
    
    # spaCy-based date parsing
    for ent in doc.ents:
        if ent.label_ == 'DATE':
            text = ent.text.lower()
            if 'today' in text:
                intent['date'] = 'last+1+day[dp]'
            elif 'yesterday' in text:
                intent['date'] = 'last+1+day[dp]'
            elif 'last month' in text and not intent['date']:
                intent['date'] = 'last+30+days[dp]'
            elif 'last week' in text and not intent['date']:
                intent['date'] = 'last+7+days[dp]'
    
    logger.info(f"Extracted keywords: {keywords}, Intent: {intent}")
    if not keywords:
        keywords = [word for word in query_lower.split() 
                    if word not in stop_words and len(word) > 1][:3]
        logger.info(f"Fallback keywords: {keywords}")
    
    return keywords, intent

def build_pubmed_query(keywords, intent):
    query_parts = []
    for kw in keywords:
        if kw == 'chronic inflammatory demyelinating polyneuropathy':
            query_parts.append('(cidp OR chronic+inflammatory+demyelinating+polyneuropathy)')
        else:
            query_parts.append(f"({kw.replace(' ', '+')})")
    query = " OR ".join(query_parts)
    if intent.get('focus'):
        focus_terms = {
            'treatment': '(treatment OR therapy OR therapeutic)',
            'diagnosis': '(diagnosis OR diagnostic)',
            'prevention': '(prevention OR preventive)',
            'review': '(review OR meta-analysis)'
        }
        query += f" AND {focus_terms[intent['focus']]}"
    if intent.get('author'):
        query += f" AND {intent['author']}[au]"
    if intent.get('date'):
        query += f" {intent['date']}"
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
        "sort": "date",
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
        pmid = article.find(".//PMID").text
        title = article.find(".//ArticleTitle").text
        abstract = article.find(".//AbstractText")
        abstract = abstract.text if abstract is not None else ""
        authors = [author.find("LastName").text for author in article.findall(".//Author") 
                   if author.find("LastName") is not None]
        journal = article.find(".//Journal/Title")
        journal = journal.text if journal is not None else ""
        pub_date = article.find(".//PubDate/Year")
        pub_date = pub_date.text if pub_date is not None else ""
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
            'result_count': 20,
            'limit_presentation': False
        }
    
    prompt_text_lower = prompt_text.lower()
    
    # Extract number of results
    result_count = 20
    if match := re.search(r'(?:top|return|summarize|include|limit\s+to|show\s+only)\s+(\d+)\s+(?:articles|results)', prompt_text_lower):
        result_count = min(int(match.group(1)), 20)  # Cap at 20 due to PubMed API
    elif 'top' in prompt_text_lower:
        result_count = 3  # Default for "top"
    
    # Check for limiting presentation (only for display, not context)
    limit_presentation = ('show only' in prompt_text_lower or 'present only' in prompt_text_lower)
    
    logger.info(f"Parsed prompt: result_count={result_count}, limit_presentation={limit_presentation}")
    
    return {
        'result_count': result_count,
        'limit_presentation': limit_presentation
    }

def grok_llm_ranking(query, results, embeddings, intent=None, prompt_params=None):
    """
    Rank results using Grok API for relevance scoring, with fallback to embedding-based ranking.
    """
    result_count = prompt_params.get('result_count', 20) if prompt_params else 20
    ranked_results = []
    ranked_embeddings = []
    
    try:
        # Prepare context for Grok
        articles_context = []
        for i, result in enumerate(results):
            article_text = f"Title: {result['title']}\nAbstract: {result['abstract']}\nAuthors: {result['authors']}\nJournal: {result['journal']}\nDate: {result['publication_date']}"
            articles_context.append(f"Article {i+1}: {article_text}")
        
        context = "\n\n".join(articles_context)
        ranking_prompt = f"""
        Given the query '{query}' and the following articles, rank the articles by relevance to the query from most to least relevant.
        Return a JSON list of article indices (1-based) in order of relevance, along with a brief explanation for each.
        Ensure the response is valid JSON. Example:
        [
            {{"index": 1, "explanation": "Most relevant due to focus on query topic"}},
            {{"index": 2, "explanation": "Relevant but less specific"}}
        ]
        Articles:
        {context}
        """
        
        response = query_grok_api(query, context, prompt=ranking_prompt)
        logger.info(f"Grok ranking response: {response[:200]}...")
        
        # Parse Grok's JSON response
        try:
            ranking = json.loads(response)
            if not isinstance(ranking, list):
                raise ValueError("Grok response is not a list")
            
            # Convert 1-based indices to 0-based
            ranked_indices = []
            for item in ranking:
                if isinstance(item, dict) and 'index' in item:
                    index = item['index']
                    if isinstance(index, (int, str)) and str(index).isdigit():
                        index = int(index) - 1
                        if 0 <= index < len(results):
                            ranked_indices.append(index)
            
            # Add missing indices
            missing_indices = [i for i in range(len(results)) if i not in ranked_indices]
            ranked_indices.extend(missing_indices)
            
            ranked_results = [results[i] for i in ranked_indices]
            ranked_embeddings = [embeddings[i] for i in ranked_indices]
            logger.info(f"Grok ranked {len(ranked_results)} results: indices {ranked_indices}")
            return ranked_results[:result_count], ranked_embeddings[:result_count]
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Error parsing Grok ranking response: {str(e)}")
            # Fall through to fallback ranking
    except Exception as e:
        logger.error(f"Grok ranking failed: {str(e)}")
    
    # Fallback: Use embedding-based ranking
    logger.info("Falling back to embedding-based ranking")
    query_embedding = generate_embedding(query)
    current_year = datetime.now().year
    scores = []
    
    focus_keywords = {
        'treatment': ['treatment', 'therapy', 'therapeutic', 'intervention'],
        'diagnosis': ['diagnosis', 'diagnostic', 'detection'],
        'prevention': ['prevention', 'preventive', 'prophylaxis'],
        'review': ['review', 'meta-analysis', 'systematic']
    }
    focus_weight = 0.2 if intent and intent.get('focus') else 0
    focus_terms = focus_keywords.get(intent.get('focus'), []) if intent else []
    
    for i, (emb, result) in enumerate(zip(embeddings, results)):
        similarity = 1 - cosine(query_embedding, emb) if emb is not None else 0.0
        pub_year = int(result['publication_date']) if result['publication_date'].isdigit() else 2000
        recency_bonus = (pub_year - 2000) / (current_year - 2000)
        
        focus_score = 0
        if focus_terms:
            abstract_lower = result['abstract'].lower()
            focus_score = sum(1 for term in focus_terms if term in abstract_lower) / len(focus_terms)
        
        author_score = 0
        if intent and intent.get('author'):
            author_lower = intent['author'].lower()
            if author_lower in result['authors'].lower():
                author_score = 0.3
        
        weighted_score = (0.7 * similarity) + (0.2 * recency_bonus) + (focus_weight * focus_score) + author_score
        scores.append((i, weighted_score))
    
    scores.sort(key=lambda x: x[1], reverse=True)
    ranked_indices = [i for i, _ in scores]
    ranked_indices = ranked_indices[:max(result_count, len(results))]
    
    ranked_results = [results[i] for i in ranked_indices]
    ranked_embeddings = [embeddings[i] for i in ranked_indices]
    logger.info(f"Fallback ranked {len(ranked_results)} results")
    return ranked_results, ranked_embeddings

def generate_summary(abstract, query, title=None, authors=None, journal=None, publication_date=None):
    if not abstract and not title:
        return {"text": "No content available to summarize.", "metadata": {}, "embedding": None}
    text = f"{title} {abstract or ''} {authors or ''} {journal or ''}".strip()
    embedding = generate_embedding(text) if text else None
    summary_text = abstract[:200] if abstract else f"Title: {title}"
    summary = {
        "text": summary_text,
        "metadata": {
            "authors": authors or "Unknown",
            "journal": journal or "Unknown",
            "publication_date": str(publication_date) if publication_date else "Unknown"
        },
        "embedding": embedding
    }
    return summary

def generate_prompt_output(query, results, prompt_text, prompt_params):
    if not results:
        return f"No results found for '{query}'."
    
    # Log initial results
    logger.info(f"Initial results count: {len(results)}")
    
    # Apply year filter if specified, but relax if too few results
    query_lower = query.lower()
    year_match = re.search(r'\b(20\d{2})\b', query_lower)
    target_year = year_match.group(1) if year_match else str(datetime.now().year) if 'this year' in query_lower else None
    result_count = prompt_params.get('result_count', 20) if prompt_params else 20
    
    filtered_results = results
    if target_year:
        filtered_results = [r for r in results if r['publication_date'] == target_year]
        logger.info(f"After year filter ({target_year}): {len(filtered_results)} results")
        if len(filtered_results) < result_count:
            # Relax filter to include more recent years
            filtered_results = results[:result_count]  # Use top results regardless of year
            logger.info(f"Relaxed year filter, using top {len(filtered_results)} results")
    
    # Ensure at least result_count results (or all available) are used
    context_results = filtered_results[:result_count]
    logger.info(f"Context results count: {len(context_results)}")
    
    if not context_results:
        return f"No results found for '{query}' matching criteria."
    
    # Prepare context from results
    context = "\n".join([f"Title: {r['title']}\nAbstract: {r['abstract']}\nAuthors: {r['authors']}\nJournal: {r['journal']}\nDate: {r['publication_date']}" for r in context_results])
    
    # Use xAI Grok API to generate response
    output = query_grok_api(prompt_text or "Summarize the provided research articles.", context)
    
    logger.info(f"Generated prompt output: length={len(output)}")
    return output

@app.route('/search_progress', methods=['GET'])
@login_required
def search_progress():
    def stream_progress():
        try:
            if not current_user or not hasattr(current_user, 'id'):
                yield f"data: {{'status': 'error: User not authenticated'}}\n\n"
                return
            query = request.args.get('query', '')
            while True:
                status, timestamp = get_search_progress(current_user.id, query)
                if status:
                    yield f"data: {{'status': '{status}'}}\n\n"
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
    prompts = [{'id': p[0], 'prompt_name': p[1], 'prompt_text': p[2]} for p in cur.fetchall()]
    cur.close()
    conn.close()

    if request.method == 'POST':
        query = request.form.get('query')
        prompt_id = request.form.get('prompt_id')
        prompt_text = request.form.get('prompt_text')
        if not query:
            return render_template('search.html', error="Query cannot be empty", prompts=prompts, prompt_id=prompt_id, prompt_text=prompt_text, username=current_user.email)
        
        # Update progress
        update_search_progress(current_user.id, query, "contacting PubMed")
        
        keywords, intent = extract_keywords_and_intent(query)
        if not keywords:
            update_search_progress(current_user.id, query, "error: No valid keywords found")
            return render_template('search.html', error="No valid keywords found", prompts=prompts, prompt_id=prompt_id, prompt_text=prompt_text, username=current_user.email)
        
        selected_prompt_text = prompt_text if prompt_text else next((p['prompt_text'] for p in prompts if str(p['id']) == prompt_id), None)
        prompt_params = parse_prompt(selected_prompt_text)
        result_count = prompt_params['result_count']
        limit_presentation = prompt_params['limit_presentation']
        
        query_lower = query.lower()
        year_match = re.search(r'\b(20\d{2})\b', query_lower)
        target_year = year_match.group(1) if year_match else str(datetime.now().year) if 'this year' in query_lower else None
        
        results = None
        api_key = os.environ.get('PUBMED_API_KEY')
        search_query = build_pubmed_query(keywords, intent)
        try:
            update_search_progress(current_user.id, query, "executing search")
            esearch_result = esearch(search_query, retmax=20, api_key=api_key)
            pmids = esearch_result['esearchresult']['idlist']
            if not pmids:
                logger.info("No results with initial query")
                update_search_progress(current_user.id, query, "error: No results found")
                return render_template('search.html', error="No results found. Try broadening your query.", prompts=prompts, prompt_id=prompt_id, prompt_text=prompt_text, target_year=target_year, username=current_user.email)
            
            update_search_progress(current_user.id, query, "fetching article details")
            efetch_xml = efetch(pmids, api_key=api_key)
            results = parse_efetch_xml(efetch_xml)
            
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute("INSERT INTO search_cache (query, results) VALUES (%s, %s)", 
                        (query, json.dumps(results)))
            conn.commit()
            cur.close()
            conn.close()
        except Exception as e:
            logger.error(f"PubMed API error: {str(e)}")
            update_search_progress(current_user.id, query, f"error: Search failed: {str(e)}")
            return render_template('search.html', error=f"Search failed: {str(e)}", prompts=prompts, prompt_id=prompt_id, prompt_text=prompt_text, target_year=target_year, username=current_user.email)
        
        high_relevance = []
        embeddings = []
        summaries = []
        for r in results:
            text = f"{r['title']} {r['abstract'] or ''} {r['authors'] or ''} {r['journal'] or ''}".strip()
            embedding = generate_embedding(text) if text else None
            summary = generate_summary(
                r['abstract'], query, 
                title=r['title'], authors=r['authors'], 
                journal=r['journal'], publication_date=r['publication_date']
            )
            high_relevance.append({
                'id': r['id'],
                'title': r['title'],
                'abstract': r['abstract'],
                'score': 0,
                'authors': r['authors'],
                'journal': r['journal'],
                'publication_date': r['publication_date'],
                'keywords': None
            })
            summaries.append(summary)
            embeddings.append(embedding)
        
        update_search_progress(current_user.id, query, "ranking results")
        high_relevance, embeddings = grok_llm_ranking(query, high_relevance, embeddings, intent, prompt_params)
        
        update_search_progress(current_user.id, query, "creating AI response")
        prompt_output = generate_prompt_output(
            query, high_relevance, selected_prompt_text, prompt_params
        )
        
        # Determine results to display
        display_results = high_relevance if limit_presentation else results
        result_summaries = list(zip(display_results, summaries[:len(display_results)]))
        
        update_search_progress(current_user.id, query, "complete")
        
        return render_template(
            'search.html', 
            result_summaries=result_summaries,
            query=query, 
            prompts=prompts, 
            prompt_id=prompt_id,
            prompt_text=prompt_text,
            prompt_output=prompt_output,
            target_year=target_year,
            username=current_user.email
        )
    
    return render_template('search.html', prompts=prompts, prompt_id='', prompt_text='', username=current_user.email)

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
    prompts = [{'id': p[0], 'prompt_name': p[1], 'prompt_text': p[2], 'created_at': p[3]} for p in cur.fetchall()]
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

# Schedule notifications on startup
schedule_notification_rules()

if __name__ == '__main__':
    app.run(debug=True)