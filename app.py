from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, Response, session, send_from_directory, make_response
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import psycopg2
from werkzeug.security import generate_password_hash, check_password_hash
import os
import json
import logging
import sqlite3
import hashlib
from datetime import datetime, timedelta
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import sendgrid
from sendgrid.helpers.mail import Mail, Email, To, Content
import time
import re
from email_validator import validate_email, EmailNotValidError
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import numpy as np
from scipy.spatial.distance import cosine
from utils import esearch, efetch, parse_efetch_xml, extract_keywords_and_date, build_pubmed_query, SearchHandler, PubMedSearchHandler, GoogleScholarSearchHandler
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

nltk_data_dir = os.path.join(os.path.dirname(__file__), 'nltk_data')
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)
nltk.data.path.append(nltk_data_dir)

try:
    logger.info("Downloading NLTK resources")
    nltk.download('punkt', download_dir=nltk_data_dir)
    nltk.download('stopwords', download_dir=nltk_data_dir)
    nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_dir)
    logger.info("NLTK resources downloaded successfully")
except Exception as e:
    logger.error(f"Failed to download NLTK resources: {str(e)}")

from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'your-secret-key')
app.config['SESSION_COOKIE_SECURE'] = True  # Ensure cookies are secure
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

scheduler = BackgroundScheduler()
scheduler.start()

sendgrid_api_key = os.environ.get('SENDGRID_API_KEY', '').strip()
if not sendgrid_api_key:
    logger.error("SENDGRID_API_KEY not set in environment variables")
sg = sendgrid.SendGridAPIClient(api_key=sendgrid_api_key) if sendgrid_api_key else None

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
logger.info("Embedding model loaded at startup.")

def datetimeformat(value, format='%Y-%m-%d %H:%M:%S'):
    if isinstance(value, float):
        return datetime.fromtimestamp(value).strftime(format)
    return value
app.jinja_env.filters['datetimeformat'] = datetimeformat

def load_embedding_model():
    global embedding_model
    return embedding_model

def generate_embedding(text):
    model = load_embedding_model()
    embedding = model.encode(text, convert_to_numpy=True)
    if embedding.shape[0] != 384:
        logger.error(f"Generated embedding has incorrect dimension: {embedding.shape[0]}")
        return None
    return embedding

def init_progress_db():
    conn = sqlite3.connect('search_progress.db')
    c = conn.cursor()
    try:
        c.execute('''CREATE TABLE IF NOT EXISTS search_progress
                     (user_id TEXT, query TEXT DEFAULT '', status TEXT NOT NULL, timestamp REAL NOT NULL)''')
        c.execute('''CREATE TABLE IF NOT EXISTS grok_cache
                     (query TEXT PRIMARY KEY, response TEXT NOT NULL, timestamp REAL NOT NULL)''')
        c.execute('''CREATE TABLE IF NOT EXISTS search_history
                     (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL, query TEXT NOT NULL, prompt_text TEXT, sources TEXT NOT NULL, result_ids TEXT NOT NULL, timestamp REAL NOT NULL)''')
        c.execute('''CREATE TABLE IF NOT EXISTS chat_history
                     (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL, session_id TEXT NOT NULL, message TEXT NOT NULL, is_user BOOLEAN NOT NULL, timestamp REAL NOT NULL)''')
        c.execute('''CREATE TABLE IF NOT EXISTS user_settings
                     (user_id TEXT PRIMARY KEY, chat_memory_retention_hours INTEGER DEFAULT 24)''')
        c.execute('''CREATE TABLE IF NOT EXISTS embedding_cache
                     (pmid TEXT PRIMARY KEY, embedding BLOB, timestamp REAL)''')
        conn.commit()
    except sqlite3.Error as e:
        logger.error(f"Error initializing SQLite database: {str(e)}")
        raise
    finally:
        c.close()
        conn.close()

init_progress_db()

def update_search_progress(user_id, query, status):
    try:
        conn = sqlite3.connect('search_progress.db')
        c = conn.cursor()
        c.execute("INSERT OR REPLACE INTO search_progress (user_id, query, status, timestamp) VALUES (?, ?, ?, ?)",
                  (user_id, query or '', status, time.time()))
        conn.commit()
    except sqlite3.Error as e:
        logger.error(f"Error updating search progress: {str(e)}")
    finally:
        c.close()
        conn.close()
    logger.info(f"Search progress updated: user={user_id}, query={query}, status={status}")

def cache_grok_response(query, response):
    conn = sqlite3.connect('search_progress.db')
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO grok_cache (query, response, timestamp) VALUES (?, ?, ?)",
              (query, response, time.time()))
    conn.commit()
    c.close()
    conn.close()
    logger.info(f"Cached Grok response for query: {query[:50]}...")

def get_cached_grok_response(query):
    conn = sqlite3.connect('search_progress.db')
    c = conn.cursor()
    c.execute("SELECT response, timestamp FROM grok_cache WHERE query = ? AND timestamp > ?",
              (query, time.time() - 604800))  # Cache valid for 7 days
    result = c.fetchone()
    c.close()
    conn.close()
    return result[0] if result else None

def cache_embedding(pmid, embedding):
    conn = sqlite3.connect('search_progress.db')
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO embedding_cache (pmid, embedding, timestamp) VALUES (?, ?, ?)",
              (pmid, embedding.tobytes(), time.time()))
    conn.commit()
    c.close()
    conn.close()

def get_cached_embedding(pmid):
    conn = sqlite3.connect('search_progress.db')
    c = conn.cursor()
    c.execute("SELECT embedding, timestamp FROM embedding_cache WHERE pmid = ? AND timestamp > ?",
              (pmid, time.time() - 604800))  # Cache valid for 7 days
    result = c.fetchone()
    c.close()
    conn.close()
    if result:
        embedding = np.frombuffer(result[0], dtype=np.float32)
        if embedding.shape[0] != 384:
            logger.warning(f"Invalid embedding dimension for PMID {pmid}: expected 384, got {embedding.shape[0]}")
            return None
        return embedding
    return None

def save_search_results(user_id, query, results):
    conn = get_db_connection()
    cur = conn.cursor()
    result_ids = []
    try:
        for result in results[:50]:
            result_id = hashlib.md5(json.dumps(result, sort_keys=True).encode()).hexdigest()
            result_data = json.dumps(result)  # Ensure string serialization
            cur.execute(
                "INSERT INTO search_results (user_id, query, source_id, result_data, created_at) VALUES (%s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                (str(user_id), query, result.get('source_id', 'unknown'), result_data, datetime.now())
            )
            result_ids.append(result_id)
        conn.commit()
    except Exception as e:
        logger.error(f"Error saving search results: {str(e)}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()
    return result_ids

def get_search_results(user_id, query):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT result_data FROM search_results WHERE user_id = %s AND query = %s ORDER BY created_at DESC LIMIT 50",
            (str(user_id), query)
        )
        results = []
        for row in cur.fetchall():
            try:
                result_data = row[0]
                if isinstance(result_data, str):
                    results.append(json.loads(result_data))
                else:
                    results.append(result_data)
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding JSON for search result: {str(e)}")
                continue
        return results
    except Exception as e:
        logger.error(f"Error retrieving search results: {str(e)}")
        return []
    finally:
        cur.close()
        conn.close()

def save_search_history(user_id, query, prompt_text, sources, results):
    result_ids = save_search_results(user_id, query, results)
    
    conn = sqlite3.connect('search_progress.db')
    c = conn.cursor()
    try:
        c.execute(
            "INSERT INTO search_history (user_id, query, prompt_text, sources, result_ids, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, query, prompt_text, json.dumps(sources), json.dumps(result_ids), time.time())
        )
        conn.commit()
    except sqlite3.Error as e:
        logger.error(f"Error saving search history: {str(e)}")
        conn.rollback()
    finally:
        c.close()
        conn.close()
    logger.info(f"Saved search history for user={user_id}, query={query}")
    return result_ids

def get_search_history(user_id):
    conn = sqlite3.connect('search_progress.db')
    c = conn.cursor()
    try:
        c.execute("SELECT id, query, prompt_text, sources, result_ids, timestamp FROM search_history WHERE user_id = ? ORDER BY timestamp DESC",
                  (user_id,))
        results = [
            {'id': row[0], 'query': row[1], 'prompt_text': row[2], 'sources': json.loads(row[3]), 'result_ids': json.loads(row[4]), 'timestamp': row[5]}
            for row in c.fetchall()
        ]
        return results
    except sqlite3.Error as e:
        logger.error(f"Error retrieving search history: {str(e)}")
        return []
    finally:
        c.close()
        conn.close()

def save_chat_message(user_id, session_id, message, is_user):
    conn = sqlite3.connect('search_progress.db')
    c = conn.cursor()
    try:
        c.execute("INSERT INTO chat_history (user_id, session_id, message, is_user, timestamp) VALUES (?, ?, ?, ?, ?)",
                  (user_id, session_id, message, is_user, time.time()))
        conn.commit()
    except sqlite3.Error as e:
        logger.error(f"Error saving chat message: {str(e)}")
    finally:
        c.close()
        conn.close()

def get_chat_history(user_id, session_id, retention_hours):
    conn = sqlite3.connect('search_progress.db')
    c = conn.cursor()
    try:
        cutoff_time = time.time() - (retention_hours * 3600)
        c.execute("SELECT message, is_user, timestamp FROM chat_history WHERE user_id = ? AND session_id = ? AND timestamp > ? ORDER BY timestamp ASC",
                  (user_id, session_id, cutoff_time))
        messages = [{'message': row[0], 'is_user': row[1], 'timestamp': row[2]} for row in c.fetchall()]
        return messages
    except sqlite3.Error as e:
        logger.error(f"Error retrieving chat history: {str(e)}")
        return []
    finally:
        c.close()
        conn.close()

def get_user_settings(user_id):
    conn = sqlite3.connect('search_progress.db')
    c = conn.cursor()
    try:
        c.execute("SELECT chat_memory_retention_hours FROM user_settings WHERE user_id = ?", (user_id,))
        result = c.fetchone()
        return result[0] if result else 24
    except sqlite3.Error as e:
        logger.error(f"Error retrieving user settings: {str(e)}")
        return 24
    finally:
        c.close()
        conn.close()

def update_user_settings(user_id, retention_hours):
    conn = sqlite3.connect('search_progress.db')
    c = conn.cursor()
    try:
        c.execute("INSERT OR REPLACE INTO user_settings (user_id, chat_memory_retention_hours) VALUES (?, ?)",
                  (user_id, retention_hours))
        conn.commit()
    except sqlite3.Error as e:
        logger.error(f"Error updating user settings: {str(e)}")
    finally:
        c.close()
        conn.close()

def get_db_connection():
    conn = psycopg2.connect(os.environ['DATABASE_URL'])
    return conn

class User(UserMixin):
    def __init__(self, id, email, admin=False, status='trial'):
        self.id = str(id)
        self.email = email
        self.admin = admin
        self.status = status

@login_manager.user_loader
def load_user(user_id):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT id, email, admin, status FROM users WHERE id = %s", (user_id,))
        user = cur.fetchone()
        if user:
            return User(user[0], user[1], user[2], user[3])
        return None
    except Exception as e:
        logger.error(f"Error loading user: {str(e)}")
        return None
    finally:
        cur.close()
        conn.close()

def validate_user_email(email):
    try:
        validate_email(email, check_deliverability=False)
        return True
    except EmailNotValidError as e:
        logger.error(f"Invalid email address: {email}, error: {str(e)}")
        return False

def parse_prompt(prompt_text):
    if not prompt_text:
        return {
            'summary_result_count': 3,
            'display_result_count': 10,
            'limit_presentation': False
        }
    
    prompt_text_lower = prompt_text.lower()
    summary_result_count = 3
    if match := re.search(r'(?:top|return|summarize|include|limit\s+to|show\s+only)\s+(\d+)\s+(?:articles|results)', prompt_text_lower):
        summary_result_count = min(int(match.group(1)), 10)
    elif 'top' in prompt_text_lower:
        summary_result_count = 3
    
    display_result_count = 10
    limit_presentation = ('show only' in prompt_text_lower or 'present only' in prompt_text_lower)
    
    logger.info(f"Parsed prompt: summary_result_count={summary_result_count}, display_result_count={display_result_count}, limit_presentation={limit_presentation}")
    
    return {
        'summary_result_count': summary_result_count,
        'display_result_count': display_result_count,
        'limit_presentation': limit_presentation
    }

def query_grok_api(prompt, context):
    try:
        api_key = os.environ.get('XAI_API_KEY')
        if not api_key:
            raise ValueError("XAI_API_KEY not set")
        client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
        completion = client.chat.completions.create(
            model="grok-3",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": context}
            ],
            max_tokens=4096
        )
        return completion.choices[0].message.content
    except Exception as e:
        logger.error(f"Error querying Grok API: {str(e)}")
        raise

def rank_results(query, results, prompt_params=None):
    display_result_count = prompt_params.get('display_result_count', 10) if prompt_params else 10
    try:
        articles_context = []
        for i, result in enumerate(results[:10]):
            article_text = f"Article {i+1}: Title: {result['title']}\nAbstract: {result.get('abstract', '')}\nAuthors: {result.get('authors', 'N/A')}\nJournal: {result.get('journal', 'N/A')}\nDate: {result.get('publication_date', 'N/A')}"
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
        cached_response = get_cached_grok_response(cache_key)
        if cached_response:
            response = cached_response
        else:
            response = query_grok_api(ranking_prompt, context)
            try:
                json.loads(response)
                cache_grok_response(cache_key, response)
            except json.JSONDecodeError as je:
                logger.error(f"Invalid JSON from Grok: {str(je)}")
                raise
        logger.info(f"Grok ranking response: {response[:200]}...")
        ranking = json.loads(response)
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
    display_result_count = prompt_params.get('display_result_count', 10) if prompt_params else 10
    query_embedding = generate_embedding(query)
    if query_embedding is None:
        logger.error("Failed to generate query embedding")
        return results[:display_result_count]
    current_year = datetime.now().year
    
    results = results[:10]
    embeddings = []
    texts = []
    for result in results:
        pmid = result.get('url', '').split('/')[-2] if 'pubmed' in result.get('url', '') else f"{result['title']}_{result['publication_date']}"
        embedding = get_cached_embedding(pmid)
        if embedding is None:
            texts.append(f"{result['title']} {result.get('abstract', '')}")
        else:
            embeddings.append(embedding)
    
    if texts:
        model = load_embedding_model()
        new_embeddings = model.encode(texts, convert_to_numpy=True)
        for i, (result, emb) in enumerate(zip(results[len(embeddings):], new_embeddings)):
            pmid = result.get('url', '').split('/')[-2] if 'pubmed' in result.get('url', '') else f"{result['title']}_{result['publication_date']}"
            if emb.shape[0] == 384:
                cache_embedding(pmid, emb)
                embeddings.append(emb)
            else:
                logger.error(f"Generated embedding for {pmid} has incorrect dimension: {emb.shape[0]}")
                embeddings.append(None)
    
    scores = []
    for i, (emb, result) in enumerate(zip(embeddings, results)):
        if emb is not None and emb.shape[0] == 384:
            similarity = 1 - cosine(query_embedding, emb)
        else:
            similarity = 0.0
        pub_year = int(result['publication_date'].split('-')[0]) if result['publication_date'] and '-' in result['publication_date'] else 2000
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
    
    summary_result_count = prompt_params.get('summary_result_count', 3) if prompt_params else 3
    context_results = results[:summary_result_count]
    logger.info(f"Context results count for summary: {len(context_results)}")
    
    if not context_results:
        return f"No results found for '{query}'{' outside the specified timeframe' if is_fallback else ''} matching criteria."
    
    context = "\n".join([f"Source: {r['source_id']}\nTitle: {r['title']}\nAbstract: {r.get('abstract', '')}\nAuthors: {r.get('authors', 'N/A')}\nJournal: {r.get('journal', 'N/A')}\nDate: {r.get('publication_date', 'N/A')}" for r in context_results])
    
    MAX_CONTEXT_LENGTH = 12000
    if len(context) > MAX_CONTEXT_LENGTH:
        context = context[:MAX_CONTEXT_LENGTH] + "... [truncated]"
        logger.warning(f"Context truncated to {MAX_CONTEXT_LENGTH} characters for query: {query[:50]}...")
    
    try:
        cache_key = hashlib.md5((query + context + prompt_text).encode()).hexdigest()
        cached_response = get_cached_grok_response(cache_key)
        if cached_response:
            output = cached_response
        else:
            output = query_grok_api(prompt_text or "Summarize the provided research articles.", context)
            cache_grok_response(cache_key, output)
        
        paragraphs = output.split('\n\n')
        formatted_output = ''.join(f'<p>{p}</p>' for p in paragraphs if p.strip())
        logger.info(f"Generated prompt output: length={len(formatted_output)}, is_fallback: {is_fallback}")
        return formatted_output
    except Exception as e:
        logger.error(f"Error generating AI summary: {str(e)}")
        output = f"Fallback: Unable to generate AI summary due to error: {str(e)}. Top results include: " + "; ".join([f"{r['title']} ({r['publication_date']})" for r in context_results])
        return ''.join(f'<p>{p}</p>' for p in output.split('\n\n') if p.strip())

def run_notification_rule(rule_id, user_id, rule_name, keywords, timeframe, prompt_text, email_format, user_email, sources, test_mode=False):
    logger.info(f"Running notification rule {rule_id} ({rule_name}) for user {user_id}, keywords: {keywords}, timeframe: {timeframe}, sources: {sources}, test_mode={test_mode}, recipient: {user_email}")
    if not validate_user_email(user_email):
        raise ValueError(f"Invalid recipient email address: {user_email}")
    
    query = keywords
    today = datetime.now()
    timeframe_ranges = {
        'daily': (today - timedelta(hours=24)).strftime('%Y/%m/%d'),
        'weekly': (today - timedelta(days=7)).strftime('%Y/%m/%d'),
        'monthly': (today - timedelta(days=31)).strftime('%Y/%m/%d'),
        'annually': (today - timedelta(days=365)).strftime('%Y/%m/%d')
    }
    start_date = timeframe_ranges[timeframe]
    date_range = f"{start_date}:{today.strftime('%Y/%m/%d')}"
    
    search_handlers = {
        'pubmed': PubMedSearchHandler(),
        'googlescholar': GoogleScholarSearchHandler()
    }
    
    try:
        keywords_with_synonyms, _, start_year_int = extract_keywords_and_date(query)
        results = []
        for source_id in sources:
            if source_id not in search_handlers:
                logger.warning(f"Unknown source in notification: {source_id}")
                continue
            handler = search_handlers[source_id]
            primary_results, _ = handler.search(query, keywords_with_synonyms, date_range, start_year_int)
            if primary_results:
                ranked_results = rank_results(query, primary_results, {'display_result_count': 10})
                results.extend([dict(r, source_id=source_id) for r in ranked_results[:10]])
        
        if results:
            save_search_results(user_id, query, results)
        
        logger.info(f"Notification rule {rule_id} retrieved {len(results)} results")
        if not results:
            content = "No new results found for this rule."
            if not sg:
                raise Exception("SendGrid API key not configured.")
            message = Mail(
                from_email=Email("noreply@firesidetechnologies.com"),
                to_emails=To(user_email),
                subject=f"PubMedResearcher {'Test ' if test_mode else ''}Notification: {rule_name}",
                plain_text_content=content
            )
            response = sg.send(message)
            response_headers = {k: v for k, v in response.headers.items()}
            logger.info(f"Email sent for rule {rule_id}, status: {response.status_code}, message_id: {response_headers.get('X-Message-Id', 'Not provided')}")
            
            if test_mode:
                return {
                    "results": [],
                    "email_content": content,
                    "status": "success",
                    "email_sent": True,
                    "message_id": response_headers.get('X-Message-Id', 'Not provided')
                }
            return
        
        context = "\n".join([f"Title: {r['title']}\nAbstract: {r.get('abstract', '')}\nAuthors: {r.get('authors', 'N/A')}\nDate: {r.get('publication_date', 'N/A')}" for r in results])
        output = query_grok_api(prompt_text or "Summarize the provided research articles.", context)
        
        if email_format == "list":
            content = "\n".join([f"- {r['title']} ({r.get('publication_date', 'N/A')})\n  {r.get('abstract', '')[:100] or 'No abstract'}..." for r in results])
        elif email_format == "detailed":
            content = "\n".join([f"Title: {r['title']}\nAuthors: {r.get('authors', 'N/A')}\nJournal: {r.get('journal', 'N/A')}\nDate: {r.get('publication_date', 'N/A')}\nAbstract: {r.get('abstract', '') or 'No abstract'}\n" for r in results])
        else:
            content = output
        
        if not sg:
            raise Exception("SendGrid API key not configured.")
        message = Mail(
            from_email=Email("noreply@firesidetechnologies.com"),
            to_emails=To(user_email),
            subject=f"PubMedResearcher {'Test ' if test_mode else ''}Notification: {rule_name}",
            plain_text_content=content
        )
        response = sg.send(message)
        response_headers = {k: v for k, v in response.headers.items()}
        logger.info(f"Email sent for rule {rule_id}, status: {response.status_code}, message_id: {response_headers.get('X-Message-Id', 'Not provided')}")
        
        if test_mode:
            return {
                "results": results,
                "email_content": content,
                "status": "success",
                "email_sent": True,
                "message_id": response_headers.get('X-Message-Id', 'Not provided')
            }
        
    except Exception as e:
        logger.error(f"Error running notification rule {rule_id}: {str(e)}")
        if test_mode:
            try:
                if not sg:
                    raise Exception("SendGrid API key not configured.")
                message = Mail(
                    from_email=Email("noreply@firesidetechnologies.com"),
                    to_emails=To(user_email),
                    subject=f"PubMedResearcher Test Notification Failed: {rule_name}",
                    plain_text_content=f"Error testing notification rule: {str(e)}"
                )
                response = sg.send(message)
                response_headers = {k: v for k, v in response.headers.items()}
                logger.info(f"Error email sent for rule {rule_id}, status: {response.status_code}, message_id: {response_headers.get('X-Message-Id', 'Not provided')}")
                email_sent = True
            except Exception as email_e:
                logger.error(f"Failed to send error email for rule {rule_id}: {str(email_e)}")
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
                "message_id": response_headers.get('X-Message-Id', 'Not provided') if email_sent else None
            }
        raise

def schedule_notification_rules():
    scheduler.remove_all_jobs()
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT n.id, n.user_id, n.rule_name, n.keywords, n.timeframe, n.prompt_text, n.email_format, u.email, n.sources "
            "FROM notifications n JOIN users u ON n.user_id = u.id"
        )
        rules = cur.fetchall()
        for rule in rules:
            rule_id, user_id, rule_name, keywords, timeframe, prompt_text, email_format, user_email, sources = rule
            sources = json.loads(sources) if sources else ['pubmed']
            cron_trigger = {
                'daily': CronTrigger(hour=8, minute=0),
                'weekly': CronTrigger(day_of_week='mon', hour=8, minute=0),
                'monthly': CronTrigger(day=1, hour=8, minute=0),
                'annually': CronTrigger(month=1, day=1, hour=8, minute=0)
            }[timeframe]
            scheduler.add_job(
                run_notification_rule,
                trigger=cron_trigger,
                args=[rule_id, user_id, rule_name, keywords, timeframe, prompt_text, email_format, user_email, sources],
                id=f"notification_{rule_id}",
                replace_existing=True
            )
        logger.info(f"Scheduled {len(rules)} notification rules")
    except Exception as e:
        logger.error(f"Error scheduling notification rules: {str(e)}")
    finally:
        cur.close()
        conn.close()

@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('search'))
    response = make_response(render_template('index.html', username=None))
    response.headers['X-Content-Type-Options'] = 'nosniff'
    return response

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        conn = get_db_connection()
        cur = conn.cursor()
        try:
            cur.execute("SELECT id FROM users WHERE email = %s", (email,))
            if cur.fetchone():
                flash('Email already registered.', 'error')
            else:
                password_hash = generate_password_hash(password)
                cur.execute("INSERT INTO users (email, password_hash) VALUES (%s, %s)", (email, password_hash))
                conn.commit()
                flash('Registration successful! Please log in.', 'success')
                return redirect(url_for('login'))
        except Exception as e:
            logger.error(f"Error during registration: {str(e)}")
            conn.rollback()
            flash('Registration failed. Please try again.', 'error')
        finally:
            cur.close()
            conn.close()
    response = make_response(render_template('register.html', username=None))
    response.headers['X-Content-Type-Options'] = 'nosniff'
    return response

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        conn = get_db_connection()
        cur = conn.cursor()
        try:
            cur.execute("SELECT id, email, password_hash, status, admin FROM users WHERE email = %s", (email,))
            user = cur.fetchone()
            if user and check_password_hash(user[2], password):
                if user[3] == 'inactive':
                    flash('Your account is inactive. Please contact support at pubmedresearch@firesidetechnologies.com.', 'error')
                else:
                    login_user(User(user[0], user[1], user[4], user[3]))
                    response = make_response(redirect(url_for('search')))
                    response.headers['X-Content-Type-Options'] = 'nosniff'
                    return response
            flash('Invalid email or password.', 'error')
        except Exception as e:
            logger.error(f"Error during login: {str(e)}")
            flash('Login failed. Please try again.', 'error')
        finally:
            cur.close()
            conn.close()
    response = make_response(render_template('login.html', username=None))
    response.headers['X-Content-Type-Options'] = 'nosniff'
    return response

@app.route('/logout')
@login_required
def logout():
    logout_user()
    response = make_response(redirect(url_for('login')))
    response.headers['X-Content-Type-Options'] = 'nosniff'
    return response

@app.route('/static/<path:filename>')
def static_files(filename):
    response = send_from_directory(app.static_folder, filename)
    response.headers['Cache-Control'] = 'public, max-age=31536000'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    return response

@app.route('/search_progress', methods=['GET'])
def search_progress():
    def stream_progress():
        try:
            if current_user is None or not hasattr(current_user, 'is_authenticated') or not current_user.is_authenticated:
                logger.debug("Skipping progress updates for unauthenticated user")
                return
            query = request.args.get('query', '')
            last_status = None
            while True:
                conn = sqlite3.connect('search_progress.db')
                c = conn.cursor()
                try:
                    c.execute("SELECT status, timestamp FROM search_progress WHERE user_id = ? AND query = ? ORDER BY timestamp DESC LIMIT 1",
                              (current_user.id, query))
                    result = c.fetchone()
                    if result and result[0] != last_status:
                        escaped_status = result[0].replace("'", "\\'")
                        yield 'data: {"status": "' + escaped_status + '"}\n\n'
                        last_status = result[0]
                    if result and result[0].startswith(("complete", "error")):
                        break
                except sqlite3.Error as e:
                    logger.error(f"Error in search_progress: {str(e)}")
                    escaped_error = str(e).replace("'", "\\'")
                    yield 'data: {"status": "error: ' + escaped_error + '"}\n\n'
                    break
                finally:
                    c.close()
                    conn.close()
                time.sleep(1)
        except Exception as e:
            logger.error(f"Error in search_progress stream: {str(e)}")
            yield 'data: {"status": "error: ' + str(e).replace("'", "\\'") + '"}\n\n'
    
    response = Response(stream_progress(), mimetype='text/event-stream')
    response.headers['X-Content-Type-Options'] = 'nosniff'
    return response

@app.route('/search', methods=['GET', 'POST'])
@login_required
def search():
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute('SELECT id, prompt_name, prompt_text FROM prompts WHERE user_id = %s', (current_user.id,))
        prompts = [{'id': str(p[0]), 'prompt_name': p[1], 'prompt_text': p[2]} for p in cur.fetchall()]
    except Exception as e:
        logger.error(f"Error loading prompts: {str(e)}")
        prompts = []
    finally:
        cur.close()
        conn.close()
    logger.info(f"Loaded prompts: {len(prompts)} prompts for user {current_user.id}")
    session.pop('latest_search_result_ids', None)

    page = {source: int(request.args.get(f'page_{source}', 1)) for source in ['pubmed', 'googlescholar']}
    per_page = 20
    sort_by = request.form.get('sort_by', request.args.get('sort_by', 'relevance'))
    prompt_id = request.form.get('prompt_id', request.args.get('prompt_id', ''))
    prompt_text = request.form.get('prompt_text', request.args.get('prompt_text', ''))
    query = request.form.get('query', request.args.get('query', ''))
    search_older = request.form.get('search_older', 'off') == 'on' or request.args.get('search_older', 'False') == 'True'
    start_year = request.form.get('start_year', request.args.get('start_year', None))
    if start_year == "None" or not start_year:
        start_year = None
    else:
        try:
            start_year = int(start_year)
        except ValueError:
            start_year = None
    sources_selected = request.form.getlist('sources') or request.args.getlist('sources') or []
    logger.debug(f"Sources selected: type={type(sources_selected)}, value={sources_selected}")

    selected_prompt_text = prompt_text
    if prompt_id and not prompt_text:
        for prompt in prompts:
            if str(prompt['id']) == prompt_id:
                selected_prompt_text = prompt['prompt_text']
                break
        else:
            logger.warning(f"Prompt ID {prompt_id} not found in prompts")
            selected_prompt_text = ''

    logger.info(f"Search request: prompt_id={prompt_id}, prompt_text={prompt_text[:50] if prompt_text else 'None'}..., query={query[:50] if query else 'None'}..., search_older={search_older}, start_year={start_year}, sources={sources_selected}, page={page}, sort_by={sort_by}")

    search_handlers = {
        'pubmed': PubMedSearchHandler(),
        'googlescholar': GoogleScholarSearchHandler()
    }

    if request.method == 'POST' or (request.method == 'GET' and query and sources_selected):
        if not query:
            update_search_progress(current_user.id, query, "error: Query cannot be empty")
            response = make_response(render_template('search.html', error="Query cannot be empty", prompts=prompts, prompt_id=prompt_id, 
                                   prompt_text=selected_prompt_text, sources=[], total_results={}, total_pages={}, page=page, per_page=per_page, 
                                   username=current_user.email, has_prompt=bool(selected_prompt_text), prompt_params={}, 
                                   summary_result_count=3, search_older=search_older, start_year=start_year, sort_by=sort_by, 
                                   pubmed_results=[], pubmed_fallback_results=[], sources_selected=sources_selected, combined_summary=''))
            response.headers['X-Content-Type-Options'] = 'nosniff'
            return response

        if not sources_selected:
            update_search_progress(current_user.id, query, "error: At least one search source must be selected")
            response = make_response(render_template('search.html', error="At least one search source must be selected", prompts=prompts, prompt_id=prompt_id, 
                                   prompt_text=selected_prompt_text, sources=[], total_results={}, total_pages={}, page=page, per_page=per_page, 
                                   username=current_user.email, has_prompt=bool(selected_prompt_text), prompt_params={}, 
                                   summary_result_count=3, search_older=search_older, start_year=start_year, sort_by=sort_by, 
                                   pubmed_results=[], pubmed_fallback_results=[], sources_selected=sources_selected, combined_summary=''))
            response.headers['X-Content-Type-Options'] = 'nosniff'
            return response

        update_search_progress(current_user.id, query, "contacting APIs")

        try:
            keywords_with_synonyms, date_range, start_year_int = extract_keywords_and_date(query, search_older, start_year)
            if not keywords_with_synonyms:
                update_search_progress(current_user.id, query, "error: No valid keywords found")
                response = make_response(render_template('search.html', error="No valid keywords found", prompts=prompts, prompt_id=prompt_id, 
                                       prompt_text=selected_prompt_text, sources=[], total_results={}, total_pages={}, page=page, per_page=per_page, 
                                       username=current_user.email, has_prompt=bool(selected_prompt_text), prompt_params={}, 
                                       summary_result_count=3, search_older=search_older, start_year=start_year, sort_by=sort_by, 
                                       pubmed_results=[], pubmed_fallback_results=[], sources_selected=sources_selected, combined_summary=''))
                response.headers['X-Content-Type-Options'] = 'nosniff'
                return response

            prompt_params = parse_prompt(selected_prompt_text) or {}
            prompt_params['sort_by'] = sort_by
            summary_result_count = prompt_params.get('summary_result_count', 3)

            sources = []
            total_results = {}
            pubmed_results = []
            googlescholar_results = []
            pubmed_fallback_results = []
            all_results = []
            all_ranked_results = []

            for source_id in sources_selected:
                if source_id not in search_handlers:
                    logger.warning(f"Unknown source: {source_id}")
                    continue

                handler = search_handlers[source_id]
                update_search_progress(current_user.id, query, f"executing {handler.name} search")

                primary_results, fallback_results = handler.search(query, keywords_with_synonyms, date_range, start_year_int)

                primary_results = primary_results or [][:20]
                fallback_results = fallback_results or [][:20]

                ranked_results = []
                if primary_results:
                    update_search_progress(current_user.id, query, f"ranking {handler.name} results")
                    ranked_results = rank_results(query, primary_results, prompt_params)

                source_summary = ""
                if selected_prompt_text and ranked_results:
                    update_search_progress(current_user.id, query, f"generating {handler.name} summary")
                    context = "\n".join([f"Title: {r['title']}\nAbstract: {r.get('abstract', '')}\nAuthors: {r.get('authors', 'N/A')}\nDate: {r.get('publication_date', 'N/A')}" 
                                        for r in ranked_results[:summary_result_count]])
                    try:
                        response = query_grok_api(selected_prompt_text, context)
                        source_summary = response
                    except Exception as e:
                        logger.error(f"Error generating {source_id} summary: {str(e)}")
                        source_summary = f"Unable to generate summary: {str(e)}"

                source_data = {
                    'id': handler.source_id,
                    'name': handler.name,
                    'results': {
                        'ranked': ranked_results,
                        'all': primary_results,
                        'fallback': fallback_results
                    },
                    'summary': source_summary
                }

                if source_id == 'pubmed':
                    if primary_results or fallback_results:
                        conn = get_db_connection()
                        cur = conn.cursor()
                        try:
                            cur.execute("INSERT INTO search_cache (query, results, created_at) VALUES (%s, %s, %s)", 
                                        (query, json.dumps(primary_results + fallback_results), datetime.now()))
                            conn.commit()
                        except Exception as e:
                            logger.error(f"Error caching search results: {str(e)}")
                            conn.rollback()
                        finally:
                            cur.close()
                            conn.close()
                        pubmed_results = primary_results
                        pubmed_fallback_results = fallback_results
                elif source_id == 'googlescholar':
                    googlescholar_results = primary_results

                source_total = len(primary_results) + len(fallback_results)
                total_results[source_id] = source_total
                sources.append(source_data)

                all_results.extend([dict(r, source_id=source_id) for r in primary_results])
                all_results.extend([dict(r, source_id=source_id) for r in fallback_results])
                all_ranked_results.extend([dict(r, source_id=source_id) for r in ranked_results[:summary_result_count]])

            total_pages = {source_id: (total_results.get(source_id, 0) + per_page - 1) // per_page for source_id in total_results}

            for source in sources:
                source_page = page.get(source['id'], 1)
                start_idx = (source_page - 1) * per_page
                end_idx = start_idx + per_page
                source['results']['ranked'] = source['results']['ranked'][start_idx:end_idx]
                source['results']['all'] = source['results']['all'][start_idx:end_idx]
                source['results']['fallback'] = source['results']['fallback'][start_idx:end_idx]

            result_ids = save_search_history(current_user.id, query, selected_prompt_text, sources_selected, all_results)
            session['latest_search_result_ids'] = json.dumps(result_ids[:10])
            session['latest_query'] = query

            update_search_progress(current_user.id, query, "complete")

            logger.debug("Rendering search template for POST/GET request")
            response = make_response(render_template(
                'search.html', 
                sources=sources,
                total_results=total_results,
                total_pages=total_pages,
                page=page,
                per_page=per_page,
                query=query, 
                prompts=prompts, 
                prompt_id=prompt_id,
                prompt_text=selected_prompt_text,
                summary_result_count=summary_result_count,
                username=current_user.email,
                has_prompt=bool(selected_prompt_text),
                prompt_params=prompt_params,
                search_older=search_older,
                start_year=start_year,
                sort_by=sort_by,
                pubmed_results=pubmed_results,
                googlescholar_results=googlescholar_results,
                pubmed_fallback_results=pubmed_fallback_results,
                sources_selected=sources_selected,
                combined_summary=''  # Deprecated, use source-specific summaries
            ))
            response.headers['X-Content-Type-Options'] = 'nosniff'
            return response
        except Exception as e:
            logger.error(f"API error in POST/GET: {str(e)}")
            update_search_progress(current_user.id, query, f"error: Search failed: {str(e)}")
            response = make_response(render_template('search.html', error=f"Search failed: {str(e)}", prompts=prompts, prompt_id=prompt_id, 
                                   prompt_text=selected_prompt_text, sources=[], total_results={}, total_pages={}, page=page, per_page=per_page, 
                                   username=current_user.email, has_prompt=bool(selected_prompt_text), prompt_params={}, 
                                   summary_result_count=3, search_older=search_older, start_year=start_year, sort_by=sort_by, 
                                   pubmed_results=[], pubmed_fallback_results=[], sources_selected=sources_selected, combined_summary=''))
            response.headers['X-Content-Type-Options'] = 'nosniff'
            return response

    logger.debug("Rendering search template for GET request")
    response = make_response(render_template(
        'search.html', 
        prompts=prompts, 
        prompt_id=prompt_id, 
        prompt_text=selected_prompt_text, 
        sources=[],
        total_results={},
        total_pages={},
        page=page,
        per_page=per_page,
        username=current_user.email, 
        has_prompt=bool(selected_prompt_text), 
        prompt_params={}, 
        summary_result_count=3, 
        search_older=False, 
        start_year=None,
        sort_by=sort_by,
        pubmed_results=[],
        pubmed_fallback_results=[],
        sources_selected=sources_selected,
        combined_summary=''
    ))
    response.headers['X-Content-Type-Options'] = 'nosniff'
    return response

@app.route('/chat', methods=['GET', 'POST'])
@login_required
def chat():
    if not current_user.is_authenticated:
        response = make_response(redirect(url_for('login')))
        response.headers['X-Content-Type-Options'] = 'nosniff'
        return response
    
    session_id = session.get('chat_session_id', str(hashlib.md5(str(time.time()).encode()).hexdigest()))
    session['chat_session_id'] = session_id
    
    retention_hours = get_user_settings(current_user.id)
    chat_history = get_chat_history(current_user.id, session_id, retention_hours)
    
    if request.method == 'POST':
        user_message = request.form.get('message', '')
        new_retention = request.form.get('retention_hours')
        
        if new_retention:
            try:
                retention_hours = int(new_retention)
                if retention_hours < 1 or retention_hours > 720:
                    flash("Retention hours must be between 1 and 720.", "error")
                else:
                    update_user_settings(current_user.id, retention_hours)
                    flash("Chat memory retention updated.", "success")
            except ValueError:
                flash("Invalid retention hours.", "error")
        
        if user_message:
            save_chat_message(current_user.id, session_id, user_message, True)
            
            search_results = get_search_results(current_user.id, session.get('latest_query', ''))
            query = session.get('latest_query', '')
            context = "\n".join([f"Source: {r['source_id']}\nTitle: {r['title']}\nAbstract: {r.get('abstract', '')}\nAuthors: {r.get('authors', 'N/A')}\nDate: {r.get('publication_date', 'N/A')}\nURL: {r.get('url', 'N/A')}" for r in search_results[:10]])
            
            with open('static/templates/chatbot_prompt.txt', 'r', encoding='utf-8') as f:
                system_prompt = f.read()
            
            history_context = "\n".join([f"{'User' if msg['is_user'] else 'Assistant'}: {msg['message']}" for msg in chat_history[-5:]])
            full_context = f"Search Query: {query}\n\nSearch Results:\n{context}\n\nChat History:\n{history_context}\n\nUser Query: {user_message}"
            
            try:
                response = query_grok_api(system_prompt, full_context)
                save_chat_message(current_user.id, session_id, response, False)
                chat_history.append({'message': user_message, 'is_user': True, 'timestamp': time.time()})
                chat_history.append({'message': response, 'is_user': False, 'timestamp': time.time()})
            except Exception as e:
                flash(f"Error generating chat response: {str(e)}", "error")
        
        response = make_response(render_template('chat.html', chat_history=chat_history, username=current_user.email, retention_hours=retention_hours))
        response.headers['X-Content-Type-Options'] = 'nosniff'
        return response
    
    response = make_response(render_template('chat.html', chat_history=chat_history, username=current_user.email, retention_hours=retention_hours))
    response.headers['X-Content-Type-Options'] = 'nosniff'
    return response

@app.route('/chat_message', methods=['POST'])
@login_required
def chat_message():
    if not current_user.is_authenticated:
        return jsonify({'status': 'error', 'message': 'User not authenticated'}), 401
    
    session_id = session.get('chat_session_id', str(hashlib.md5(str(time.time()).encode()).hexdigest()))
    session['chat_session_id'] = session_id
    
    user_message = request.form.get('message', '')
    if not user_message:
        return jsonify({'status': 'error', 'message': 'Message cannot be empty'}), 400
    
    try:
        save_chat_message(current_user.id, session_id, user_message, True)
        
        retention_hours = get_user_settings(current_user.id)
        chat_history = get_chat_history(current_user.id, session_id, retention_hours)
        
        query = session.get('latest_query', '')
        search_results = get_search_results(current_user.id, query)
        context = "\n".join([f"Source: {r['source_id']}\nTitle: {r['title']}\nAbstract: {r.get('abstract', '')}\nAuthors: {r.get('authors', 'N/A')}\nDate: {r.get('publication_date', 'N/A')}\nURL: {r.get('url', 'N/A')}" for r in search_results[:10]])
        
        with open('static/templates/chatbot_prompt.txt', 'r', encoding='utf-8') as f:
            system_prompt = f.read()
        
        history_context = "\n".join([f"{'User' if msg['is_user'] else 'Assistant'}: {msg['message']}" for msg in chat_history[-5:]])
        full_context = f"Search Query: {query}\n\nSearch Results:\n{context}\n\nChat History:\n{history_context}\n\nUser Query: {user_message}"
        
        response = query_grok_api(system_prompt, full_context)
        if "summary" in user_message.lower() and "top" in user_message.lower():
            formatted_response = ""
            for source_id in ['pubmed', 'googlescholar']:
                source_results = [r for r in search_results if r['source_id'] == source_id][:3]
                if source_results:
                    context = "\n".join([f"Title: {r['title']}\nAbstract: {r.get('abstract', '')}" for r in source_results])
                    summary_prompt = f"Summarize the abstracts of the following {source_id} articles in simple terms. Provide one paragraph per article, up to 3 paragraphs."
                    summary = query_grok_api(summary_prompt, context)
                    formatted_response += f"{source_id.capitalize()} Summaries:\n{summary}\n\n"
            response = formatted_response.strip() or response
        
        save_chat_message(current_user.id, session_id, response, False)
        
        return jsonify({'status': 'success', 'message': response})
    except Exception as e:
        logger.error(f"Error in chat_message: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/previous_searches', methods=['GET'])
@login_required
def previous_searches():
    search_history = get_search_history(current_user.id)
    response = make_response(render_template('previous_searches.html', searches=search_history, username=current_user.email))
    response.headers['X-Content-Type-Options'] = 'nosniff'
    return response

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
                logger.error(f"Failed to save prompt: {str(e)}")
                conn.rollback()
                flash(f'Failed to save prompt: {str(e)}', 'error')
            finally:
                cur.close()
                conn.close()
    
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute('SELECT id, prompt_name, prompt_text, created_at FROM prompts WHERE user_id = %s ORDER BY created_at DESC', 
                    (current_user.id,))
        prompts = [{'id': str(p[0]), 'prompt_name': p[1], 'prompt_text': p[2], 'created_at': p[3]} for p in cur.fetchall()]
    except Exception as e:
        logger.error(f"Error loading prompts: {str(e)}")
        prompts = []
    finally:
        cur.close()
        conn.close()
    response = make_response(render_template('prompt.html', prompts=prompts, username=current_user.email))
    response.headers['X-Content-Type-Options'] = 'nosniff'
    return response

@app.route('/prompt/edit/<int:id>', methods=['GET', 'POST'])
@login_required
def edit_prompt(id):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute('SELECT id, prompt_name, prompt_text FROM prompts WHERE id = %s AND user_id = %s', 
                    (id, current_user.id))
        prompt = cur.fetchone()
        
        if not prompt:
            flash('Prompt not found or you do not have permission to edit it.', 'error')
            response = make_response(redirect(url_for('prompt')))
            response.headers['X-Content-Type-Options'] = 'nosniff'
            return response
        
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
                    response = make_response(redirect(url_for('prompt')))
                    response.headers['X-Content-Type-Options'] = 'nosniff'
                    return response
                except Exception as e:
                    logger.error(f"Failed to update prompt: {str(e)}")
                    conn.rollback()
                    flash(f'Failed to update prompt: {str(e)}', 'error')
        
        response = make_response(render_template('prompt_edit.html', prompt={'id': prompt[0], 'prompt_name': prompt[1], 'prompt_text': prompt[2]}, username=current_user.email))
        response.headers['X-Content-Type-Options'] = 'nosniff'
        return response
    except Exception as e:
        logger.error(f"Error in edit_prompt: {str(e)}")
        flash('An error occurred. Please try again.', 'error')
        response = make_response(redirect(url_for('prompt')))
        response.headers['X-Content-Type-Options'] = 'nosniff'
        return response
    finally:
        cur.close()
        conn.close()

@app.route('/prompt/delete/<int:id>', methods=['POST'])
@login_required
def delete_prompt(id):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute('SELECT id FROM prompts WHERE id = %s AND user_id = %s', (id, current_user.id))
        prompt = cur.fetchone()
        
        if not prompt:
            flash('Prompt not found or you do not have permission to delete it.', 'error')
            response = make_response(redirect(url_for('prompt')))
            response.headers['X-Content-Type-Options'] = 'nosniff'
            return response
        
        cur.execute('DELETE FROM prompts WHERE id = %s AND user_id = %s', (id, current_user.id))
        conn.commit()
        flash('Prompt deleted successfully.', 'success')
    except Exception as e:
        logger.error(f"Error deleting prompt: {str(e)}")
        conn.rollback()
        flash(f'Failed to delete prompt: {str(e)}', 'error')
    finally:
        cur.close()
        conn.close()
    
    response = make_response(redirect(url_for('prompt')))
    response.headers['X-Content-Type-Options'] = 'nosniff'
    return response

@app.route('/notifications', methods=['GET', 'POST'])
@login_required
def notifications():
    conn = get_db_connection()
    cur = conn.cursor()
    
    pre_query = request.args.get('keywords', '')
    pre_prompt = request.args.get('prompt_text', '')
    
    if request.method == 'POST':
        rule_name = request.form.get('rule_name')
        keywords = request.form.get('keywords')
        timeframe = request.form.get('timeframe')
        prompt_text = request.form.get('prompt_text')
        email_format = request.form.get('email_format')
        sources = request.form.getlist('sources')
        
        if not all([rule_name, keywords, timeframe, sources, email_format]):
            flash('All required fields must be filled.', 'error')
        elif timeframe not in ['daily', 'weekly', 'monthly', 'annually']:
            flash('Invalid timeframe selected.', 'error')
        elif email_format not in ['summary', 'list', 'detailed']:
            flash('Invalid email format selected.', 'error')
        else:
            try:
                cur.execute(
                    """
                    INSERT INTO notifications (user_id, rule_name, keywords, timeframe, prompt_text, email_format, sources)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (current_user.id, rule_name, keywords, timeframe, prompt_text, email_format, json.dumps(sources))
                )
                conn.commit()
                flash('Notification rule created successfully.', 'success')
                schedule_notification_rules()
            except Exception as e:
                logger.error(f"Error creating notification: {str(e)}")
                conn.rollback()
                flash(f'Failed to create notification rule: {str(e)}', 'error')
    
    try:
        cur.execute(
            "SELECT id, rule_name, keywords, timeframe, prompt_text, email_format, created_at, sources "
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
                'created_at': n[6],
                'sources': json.loads(n[7]) if n[7] else ['pubmed']
            } for n in cur.fetchall()
        ]
    except Exception as e:
        logger.error(f"Error loading notifications: {str(e)}")
        notifications = []
    finally:
        cur.close()
        conn.close()
    response = make_response(render_template('notifications.html', notifications=notifications, username=current_user.email, pre_query=pre_query, pre_prompt=pre_prompt))
    response.headers['X-Content-Type-Options'] = 'nosniff'
    return response

@app.route('/notifications/edit/<int:id>', methods=['GET', 'POST'])
@login_required
def edit_notification(id):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT id, rule_name, keywords, timeframe, prompt_text, email_format, sources "
            "FROM notifications WHERE id = %s AND user_id = %s",
            (id, current_user.id)
        )
        notification = cur.fetchone()
        
        if not notification:
            flash('Notification rule not found or you do not have permission to edit it.', 'error')
            response = make_response(redirect(url_for('notifications')))
            response.headers['X-Content-Type-Options'] = 'nosniff'
            return response
        
        if request.method == 'POST':
            rule_name = request.form.get('rule_name')
            keywords = request.form.get('keywords')
            timeframe = request.form.get('timeframe')
            prompt_text = request.form.get('prompt_text')
            email_format = request.form.get('email_format')
            sources = request.form.getlist('sources')
            
            if not all([rule_name, keywords, timeframe, sources, email_format]):
                flash('All required fields must be filled.', 'error')
            elif timeframe not in ['daily', 'weekly', 'monthly', 'annually']:
                flash('Invalid timeframe selected.', 'error')
            elif email_format not in ['summary', 'list', 'detailed']:
                flash('Invalid email format selected.', 'error')
            else:
                try:
                    cur.execute(
                        """
                        UPDATE notifications SET rule_name = %s, keywords = %s, timeframe = %s, prompt_text = %s, email_format = %s, sources = %s
                        WHERE id = %s AND user_id = %s
                        """,
                        (rule_name, keywords, timeframe, prompt_text, email_format, json.dumps(sources), id, current_user.id)
                    )
                    conn.commit()
                    flash('Notification rule updated successfully.', 'success')
                    schedule_notification_rules()
                    response = make_response(redirect(url_for('notifications')))
                    response.headers['X-Content-Type-Options'] = 'nosniff'
                    return response
                except Exception as e:
                    logger.error(f"Error updating notification: {str(e)}")
                    conn.rollback()
                    flash(f'Failed to update notification rule: {str(e)}', 'error')
        
        notification_data = {
            'id': notification[0],
            'rule_name': notification[1],
            'keywords': notification[2],
            'timeframe': notification[3],
            'prompt_text': notification[4],
            'email_format': notification[5],
            'sources': json.loads(notification[6]) if notification[6] else ['pubmed']
        }
        response = make_response(render_template('notification_edit.html', notification=notification_data, username=current_user.email))
        response.headers['X-Content-Type-Options'] = 'nosniff'
        return response
    except Exception as e:
        logger.error(f"Error in edit_notification: {str(e)}")
        flash('An error occurred. Please try again.', 'error')
        response = make_response(redirect(url_for('notifications')))
        response.headers['X-Content-Type-Options'] = 'nosniff'
        return response
    finally:
        cur.close()
        conn.close()

@app.route('/notifications/delete/<int:id>', methods=['POST'])
@login_required
def delete_notification(id):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute('SELECT id FROM notifications WHERE id = %s AND user_id = %s', (id, current_user.id))
        notification = cur.fetchone()
        
        if not notification:
            flash('Notification rule not found or you do not have permission to delete it.', 'error')
            response = make_response(redirect(url_for('notifications')))
            response.headers['X-Content-Type-Options'] = 'nosniff'
            return response
        
        cur.execute('DELETE FROM notifications WHERE id = %s AND user_id = %s', (id, current_user.id))
        conn.commit()
        flash('Notification rule deleted successfully.', 'success')
        schedule_notification_rules()
    except Exception as e:
        logger.error(f"Error deleting notification: {str(e)}")
        conn.rollback()
        flash(f'Failed to delete notification rule: {str(e)}', 'error')
    finally:
        cur.close()
        conn.close()
    
    response = make_response(redirect(url_for('notifications')))
    response.headers['X-Content-Type-Options'] = 'nosniff'
    return response

@app.route('/notifications/test/<int:id>', methods=['GET'])
@login_required
def test_notification(id):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT n.id, n.user_id, n.rule_name, n.keywords, n.timeframe, n.prompt_text, n.email_format, u.email, n.sources "
            "FROM notifications n JOIN users u ON n.user_id = u.id WHERE n.id = %s AND n.user_id = %s",
            (id, current_user.id)
        )
        rule = cur.fetchone()
        
        if not rule:
            flash('Notification rule not found or you do not have permission to test it.', 'error')
            return jsonify({"status": "error", "message": "Rule not found"}), 404
        
        rule_id, user_id, rule_name, keywords, timeframe, prompt_text, email_format, user_email, sources = rule
        sources = json.loads(sources) if sources else ['pubmed']
        test_result = run_notification_rule(
            rule_id, user_id, rule_name, keywords, timeframe, prompt_text, email_format, user_email, sources, test_mode=True
        )
        return jsonify(test_result)
    except Exception as e:
        logger.error(f"Error testing notification rule {id}: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        cur.close()
        conn.close()

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
        logger.info(f"Test email sent, recipient: {email}, status: {response.status_code}, message_id: {response_headers.get('X-Message-Id', 'Not provided')}")
        return jsonify({
            "status": "success",
            "message": f"Test email sent to {email}. Check your inbox and spam/junk folder.",
            "message_id": response_headers.get('X-Message-Id', 'Not provided')
        })
    except Exception as e:
        logger.error(f"Error sending test email: {str(e)}")
        error_detail = ""
        if hasattr(e, 'body') and e.body:
            try:
                error_body = json.loads(e.body.decode('utf-8'))
                error_detail = f": {error_body.get('errors', [{}])[0].get('message', 'No details provided')}"
            except json.JSONDecodeError:
                error_detail = f": {e.body.decode('utf-8')}"
        return jsonify({"status": "error", "message": f"Failed to send test email: {str(e)}{error_detail}"}), 500

@app.route('/help')
def help():
    response = make_response(render_template('help.html', username=current_user.email if current_user.is_authenticated else None))
    response.headers['X-Content-Type-Options'] = 'nosniff'
    return response

schedule_notification_rules()

if __name__ == '__main__':
    app.run(debug=True)