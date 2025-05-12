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
from openai import OpenAI
import traceback
import openai
import httpx.__version__
import time
import tenacity

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

# xAI Grok API call with enhanced retries
@tenacity.retry(
    stop=tenacity.stop_after_attempt(5),
    wait=tenacity.wait_exponential(multiplier=1, min=2, max=10),
    retry=tenacity.retry_if_exception_type((requests.exceptions.RequestException, Exception)),
    before_sleep=lambda retry_state: logger.info(f"Retrying Grok API call, attempt {retry_state.attempt_number}")
)
def query_grok_api(query, context, prompt="Process the provided context according to the user's prompt."):
    try:
        api_key = os.environ.get('XAI_API_KEY')
        if not api_key:
            logger.error("XAI_API_KEY not set")
            return "Error: xAI API key not configured"
        
        client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
        logger.info(f"Sending Grok API request: prompt={query[:50]}..., context_length={len(context)}")
        
        completion = client.chat.completions.create(
            model="grok-3",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Based on the following context, answer the prompt: {query}\n\nContext: {context}"}
            ],
            max_tokens=1000,
            timeout=30
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
                "model": "grok-3",
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

def run_notification_rule(rule_id, user_id, rule_name, keywords, timeframe, prompt_text, email_format, user_email, test_mode=False):
    logger.info(f"Running notification rule {rule_id} ({rule_name}) for user {user_id}, keywords: {keywords}, test_mode: {test_mode}")
    keywords_list = [k.strip() for k in keywords.split(',')]
    
    # Calculate precise date ranges for PubMed [dp]
    today = datetime.now()
    date_filters = {
        'daily': f"{(today - timedelta(days=1)).strftime('%Y/%m/%d')}[dp] TO {today.strftime('%Y/%m/%d')}[dp]",
        'weekly': f"{(today - timedelta(days=7)).strftime('%Y/%m/%d')}[dp] TO {today.strftime('%Y/%m/%d')}[dp]",
        'monthly': f"{(today - timedelta(days=31)).strftime('%Y/%m/%d')}[dp] TO {today.strftime('%Y/%m/%d')}[dp]",
        'annually': f"{(today - timedelta(days=365)).strftime('%Y/%m/%d')}[dp] TO {today.strftime('%Y/%m/%d')}[dp]"
    }
    intent = {'date': date_filters[timeframe]}
    search_query = build_pubmed_query(keywords_list, intent)
    
    try:
        api_key = os.environ.get('PUBMED_API_KEY')
        esearch_result = esearch(search_query, retmax=20, api_key=api_key)
        pmids = esearch_result['esearchresult']['idlist']
        if not pmids:
            logger.info(f"No new results for rule {rule_id}")
            content = "No new results found for this rule."
            # Send email even if no results (to test SendGrid)
            message = Mail(
                from_email=Email("notifications@pubmedresearcher.com"),
                to_emails=To(user_email),
                subject=f"PubMedResearcher {'Test ' if test_mode else ''}Notification: {rule_name}",
                plain_text_content=content
            )
            response = sg.send(message)
            logger.info(f"Email sent for rule {rule_id}, test_mode: {test_mode}, status: {response.status_code}, no results")
            
            if test_mode:
                return {
                    "results": [],
                    "email_content": content,
                    "status": "success",
                    "email_sent": True
                }
            return
        
        efetch_xml = efetch(pmids, api_key=api_key)
        results = parse_efetch_xml(efetch_xml)
        
        # Prepare email content
        context = "\n".join([f"Title: {r['title']}\nAbstract: {r['abstract'] or ''}\nAuthors: {r['authors']}\nJournal: {r['journal']}\nDate: {r['publication_date']}" for r in results])
        output = query_grok_api(prompt_text or "Summarize the provided research articles.", context)
        
        if email_format == "list":
            content = "\n".join([f"- {r['title']} ({r['publication_date']})\n  {r['abstract'][:100] or 'No abstract'}..." for r in results])
        elif email_format == "detailed":
            content = "\n".join([f"Title: {r['title']}\nAuthors: {r['authors']}\nJournal: {r['journal']}\nDate: {r['publication_date']}\nAbstract: {r['abstract'] or 'No abstract'}\n" for r in results])
        else:
            content = output
        
        # Send email (in both test and non-test mode)
        message = Mail(
            from_email=Email("notifications@pubmedresearcher.com"),
            to_emails=To(user_email),
            subject=f"PubMedResearcher {'Test ' if test_mode else ''}Notification: {rule_name}",
            plain_text_content=content
        )
        response = sg.send(message)
        logger.info(f"Email sent for rule {rule_id}, test_mode: {test_mode}, status: {response.status_code}")
        
        if test_mode:
            return {
                "results": results,
                "email_content": content,
                "status": "success",
                "email_sent": True
            }
        
    except Exception as e:
        logger.error(f"Error running notification rule {rule_id}: {str(e)}")
        if test_mode:
            # Attempt to send an error email
            try:
                message = Mail(
                    from_email=Email("notifications@pubmedresearcher.com"),
                    to_emails=To(user_email),
                    subject=f"PubMedResearcher Test Notification Failed: {rule_name}",
                    plain_text_content=f"Error testing notification rule: {str(e)}"
                )
                response = sg.send(message)
                logger.info(f"Error email sent for rule {rule_id}, test_mode: {test_mode}, status: {response.status_code}")
                email_sent = True
            except Exception as email_e:
                logger.error(f"Failed to send error email for rule {rule_id}: {str(email_e)}")
                email_sent = False
            return {
                "results": [],
                "email_content": f"Error: {str(e)}",
                "status": "error",
                "email_sent": email_sent
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

def extract_keywords_and_intent(query):
    intent_prompt = """
Analyze the following medical research query and extract its intent and keywords for a PubMed API search.
- Identify the core topic (e.g., disease, condition).
- Identify the focus, which must be one of: 'treatment', 'diagnosis', 'prevention', 'review', or 'relationship' (for queries about impact, association, or effect).
- Extract only explicit terms from the query (e.g., 'weight loss', 'heart disease'), avoiding inferred terms unless directly mentioned.
- Exclude terms like 'impact', 'relationship', or 'association' from keywords; map them to 'focus: relationship'.
- Identify the timeframe (e.g., specific year, recent).
- Identify any author names if present.
- Return a JSON object with:
  - 'keywords': List of search terms (prioritize explicit phrases, include MeSH terms if applicable).
  - 'intent': Dictionary with 'topic', 'focus', 'date', 'author' (null if not specified).
Ensure terms are relevant to medical research and suitable for PubMed's Boolean query syntax.
Query: {0}
Example output for "articles on the relationship between weight loss and heart disease":
{{
  "keywords": ["weight loss", "heart disease"],
  "intent": {{
    "topic": "heart disease",
    "focus": "relationship",
    "date": null,
    "author": null
  }}
}}
Example output for "what can you tell me is new in the treatment of diabetes in 2025":
{{
  "keywords": ["diabetes", "new treatment"],
  "intent": {{
    "topic": "diabetes",
    "focus": "treatment",
    "date": "2025[dp]",
    "author": null
  }}
}}
""".format(query)
    try:
        response = query_grok_api(query, "", prompt=intent_prompt)
        result = json.loads(response)
        keywords = result.get('keywords', [])
        intent = result.get('intent', {'topic': None, 'focus': None, 'date': None, 'author': None})
        logger.info(f"Extracted keywords: {keywords}, Intent: {intent}")
        if not keywords:
            logger.warning("No keywords extracted, using fallback")
            keywords = [word for word in query.lower().split() if word not in {'what', 'can', 'tell', 'me', 'is', 'new', 'in', 'the', 'of', 'for', 'any', 'articles', 'that', 'show', 'relationship', 'impact', 'between'} and len(word) > 1][:3]
        return keywords, intent
    except Exception as e:
        logger.error(f"Error extracting intent with Grok: {str(e)}")
        # Fallback to simple keyword extraction
        query_lower = query.lower()
        keywords = [word for word in query_lower.split() if word not in {'what', 'can', 'tell', 'me', 'is', 'new', 'in', 'the', 'of', 'for', 'any', 'articles', 'that', 'show', 'relationship', 'impact', 'between'} and len(word) > 1][:3]
        intent = {'topic': None, 'focus': None, 'date': None, 'author': None}
        year_match = re.search(r'\b(20\d{2})\b', query_lower)
        if year_match:
            intent['date'] = f"{year_match.group(1)}[dp]"
        if 'relationship' in query_lower or 'impact' in query_lower or 'association' in query_lower:
            intent['focus'] = 'relationship'
        elif 'treatment' in query_lower or 'therapy' in query_lower:
            intent['focus'] = 'treatment'
        logger.info(f"Fallback keywords: {keywords}, Intent: {intent}")
        return keywords, intent

def build_pubmed_query(keywords, intent):
    query_parts = []
    for kw in keywords:
        if kw == 'chronic inflammatory demyelinating polyneuropathy':
            query_parts.append('(cidp OR chronic+inflammatory+demyelinating+polyneuropathy)')
        else:
            # Use MeSH terms if applicable
            kw = kw.replace(' ', '+')
            query_parts.append(f"({kw}[MeSH Terms] OR {kw})")
    # Use AND to ensure all keywords are included
    query = " AND ".join(query_parts)
    if intent.get('focus'):
        focus_terms = {
            'treatment': '(treatment OR therapy OR therapeutic)',
            'diagnosis': '(diagnosis OR diagnostic)',
            'prevention': '(prevention OR preventive)',
            'review': '(review OR meta-analysis)',
            'relationship': '(relationship OR association OR impact OR effect)'
        }
        if intent['focus'] in focus_terms:
            query += f" AND {focus_terms[intent['focus']]}"
        else:
            logger.warning(f"Invalid focus: {intent['focus']}, skipping focus terms")
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
    result_count = prompt_params.get('result_count', 20) if prompt_params else 20
    ranked_results = []
    ranked_embeddings = []
    
    try:
        # Prepare context for Grok
        articles_context = []
        for i, result in enumerate(results):
            article_text = f"Title: {result['title']}\nAbstract: {result['abstract'] or ''}\nAuthors: {result['authors']}\nJournal: {result['journal']}\nDate: {result['publication_date']}"
            articles_context.append(f"Article {i+1}: {article_text}")
        
        context = "\n\n".join(articles_context)
        ranking_prompt = f"""
Given the query '{query}', rank the following articles by relevance.
Focus on articles that directly address the query's topic and intent (e.g., treatments for diabetes, relationships between weight loss and heart disease).
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
        
        response = query_grok_api(query, context, prompt=ranking_prompt)
        logger.info(f"Grok ranking response: {response[:200]}...")
        
        # Parse Grok's JSON response
        try:
            ranking = json.loads(response)
            # Handle case where response is {"articles": [...]}
            if isinstance(ranking, dict) and 'articles' in ranking:
                ranking = ranking['articles']
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
            
            ranked_results = [results[i] for i in ranked_indices[:result_count]]
            ranked_embeddings = [embeddings[i] for i in ranked_indices[:result_count]]
            logger.info(f"Grok ranked {len(ranked_results)} results: indices {ranked_indices[:result_count]}")
            return ranked_results, ranked_embeddings, ranked_indices
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Error parsing Grok ranking response: {str(e)}")
            # Fall through to fallback ranking
    except Exception as e:
        logger.error(f"Grok ranking failed: {str(e)}")
    
    # Fallback: Use embedding-based ranking with secondary date sorting
    logger.info("Falling back to embedding-based ranking")
    query_embedding = generate_embedding(query)
    current_year = datetime.now().year
    scores = []
    
    focus_keywords = {
        'treatment': ['treatment', 'therapy', 'therapeutic'],
        'diagnosis': ['diagnosis', 'diagnostic', 'detection'],
        'prevention': ['prevention', 'preventive', 'prophylaxis'],
        'review': ['review', 'meta-analysis', 'systematic'],
        'relationship': ['relationship', 'association', 'impact', 'effect']
    }
    focus_weight = 0.2 if intent and intent.get('focus') else 0
    focus_terms = focus_keywords.get(intent.get('focus'), []) if intent else []
    
    for i, (emb, result) in enumerate(zip(embeddings, results)):
        similarity = 1 - cosine(query_embedding, emb) if emb is not None else 0.0
        pub_year = int(result['publication_date']) if result['publication_date'].isdigit() else 2000
        recency_bonus = (pub_year - 2000) / (current_year - 2000)
        
        focus_score = 0
        if focus_terms and result['abstract']:
            abstract_lower = result['abstract'].lower()
            focus_score = sum(1 for term in focus_terms if term in abstract_lower) / len(focus_terms)
        
        author_score = 0
        if intent and intent.get('author'):
            author_lower = intent['author'].lower()
            if author_lower in result['authors'].lower():
                author_score = 0.3
        
        # Primary: relevance score; Secondary: publication year for tie-breaking
        weighted_score = (0.7 * similarity) + (0.2 * recency_bonus) + (focus_weight * focus_score) + author_score
        scores.append((i, weighted_score, pub_year))
    
    # Sort by weighted_score (descending) and pub_year (descending) for ties
    scores.sort(key=lambda x: (x[1], x[2]), reverse=True)
    ranked_indices = [i for i, _, _ in scores]
    
    ranked_results = [results[i] for i in ranked_indices[:result_count]]
    ranked_embeddings = [embeddings[i] for i in ranked_indices[:result_count]]
    logger.info(f"Fallback ranked {len(ranked_results)} results with indices {ranked_indices[:result_count]}")
    return ranked_results, ranked_embeddings, ranked_indices

def generate_summary(abstract, query, title=None, authors=None, journal=None, publication_date=None):
    if not abstract and not title:
        return {"text": "No content available to summarize.", "metadata": {}, "embedding": None}
    text = f"{title} {abstract or ''} {authors or ''} {journal or ''}".strip()
    embedding = generate_embedding(text) if text else None
    summary_text = abstract[:300] if abstract else f"Title: {title}"
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
    
    # Apply year filter if specified
    query_lower = query.lower()
    year_match = re.search(r'\b(20\d{2})\b', query_lower)
    target_year = year_match.group(1) if year_match else str(datetime.now().year) if 'this year' in query_lower else None
    result_count = prompt_params.get('result_count', 20) if prompt_params else 20
    
    filtered_results = results
    if target_year:
        filtered_results = [r for r in results if r['publication_date'] == target_year]
        logger.info(f"After year filter ({target_year}): {len(filtered_results)} results")
        if not filtered_results:
            flash(f"No results found for {target_year}. Displaying all available results.", "warning")
            filtered_results = results  # Fall back to all results with warning
    
    # Ensure at least result_count results (or all available) are used for summarization
    context_results = filtered_results[:result_count]
    logger.info(f"Context results count: {len(context_results)}")
    
    if not context_results:
        return f"No results found for '{query}' matching criteria."
    
    # Prepare context from results
    context = "\n".join([f"Title: {r['title']}\nAbstract: {r['abstract'] or ''}\nAuthors: {r['authors']}\nJournal: {r['journal']}\nDate: {r['publication_date']}" for r in context_results])
    
    # Use xAI Grok API to generate response
    output = query_grok_api(prompt_text or "Summarize the provided research articles.", context)
    
    # Format output with paragraph breaks
    paragraphs = output.split('\n\n')
    formatted_output = ''.join(f'<p>{p}</p>' for p in paragraphs if p.strip())
    logger.info(f"Generated prompt output: length={len(formatted_output)}")
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

    # Handle prompt_id from GET (e.g., page reload) or POST (form submission)
    prompt_id = request.args.get('prompt_id', request.form.get('prompt_id', ''))
    prompt_text = request.args.get('prompt_text', request.form.get('prompt_text', ''))
    query = request.args.get('query', request.form.get('query', ''))

    # If a prompt_id is selected, fetch its text
    selected_prompt_text = prompt_text
    if prompt_id and not prompt_text:  # Only override if prompt_text is empty
        for prompt in prompts:
            if prompt['id'] == prompt_id:
                selected_prompt_text = prompt['prompt_text']
                break
        else:
            logger.warning(f"Prompt ID {prompt_id} not found in prompts")
            selected_prompt_text = ''

    logger.info(f"Search request: prompt_id={prompt_id}, prompt_text={prompt_text[:50]}..., selected_prompt_text={selected_prompt_text[:50]}..., query={query[:50]}...")

    if request.method == 'POST':
        if not query:
            update_search_progress(current_user.id, query, "error: Query cannot be empty")
            return render_template('search.html', error="Query cannot be empty", prompts=prompts, prompt_id=prompt_id, prompt_text=selected_prompt_text, results=[], prompt_output='', username=current_user.email)
        
        # Update progress
        update_search_progress(current_user.id, query, "contacting PubMed")
        
        keywords, intent = extract_keywords_and_intent(query)
        if not keywords:
            update_search_progress(current_user.id, query, "error: No valid keywords found")
            return render_template('search.html', error="No valid keywords found", prompts=prompts, prompt_id=prompt_id, prompt_text=selected_prompt_text, results=[], prompt_output='', username=current_user.email)
        
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
                return render_template('search.html', error="No results found. Try broadening your query.", prompts=prompts, prompt_id=prompt_id, prompt_text=selected_prompt_text, results=[], prompt_output='', target_year=target_year, username=current_user.email)
            
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
            return render_template('search.html', error=f"Search failed: {str(e)}", prompts=prompts, prompt_id=prompt_id, prompt_text=selected_prompt_text, results=[], prompt_output='', target_year=target_year, username=current_user.email)
        
        # Rank results
        update_search_progress(current_user.id, query, "ranking results")
        ranked_results, _, ranked_indices = grok_llm_ranking(query, results, [generate_embedding(f"{r['title']} {r['abstract'] or ''}") for r in results], intent, prompt_params)
        
        # Sort the full results list using all ranked indices
        results = [results[i] for i in ranked_indices]
        logger.info(f"Full results list sorted by ranked indices: {len(results)} results")
        
        update_search_progress(current_user.id, query, "creating AI response")
        prompt_output = generate_prompt_output(query, ranked_results, selected_prompt_text, prompt_params)
        
        # Ensure partial results are displayed even if Grok fails
        if prompt_output.startswith("Fallback:"):
            flash("AI summarization failed, displaying raw results.", "warning")
        
        update_search_progress(current_user.id, query, "complete")
        
        return render_template(
            'search.html', 
            results=results,
            query=query, 
            prompts=prompts, 
            prompt_id=prompt_id,
            prompt_text=selected_prompt_text,
            prompt_output=prompt_output,
            target_year=target_year,
            username=current_user.email
        )
    
    return render_template('search.html', prompts=prompts, prompt_id=prompt_id, prompt_text=selected_prompt_text, results=[], prompt_output='', username=current_user.email)

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

# Schedule notifications on startup
schedule_notification_rules()

if __name__ == '__main__':
    app.run(debug=True)