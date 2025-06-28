from flask import Flask, make_response, render_template
from flask_login import LoginManager
import os
import logging
import psycopg2
import sendgrid
from sendgrid import SendGridAPIClient
from datetime import datetime
import time
from apscheduler.schedulers.background import BackgroundScheduler
from openai import OpenAI
import tenacity
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import numpy as np

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

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'your-secret-key')
app.config['SESSION_COOKIE_SECURE'] = True
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

scheduler = BackgroundScheduler()
scheduler.start()

sendgrid_api_key = os.environ.get('SENDGRID_API_KEY', '').strip()
if not sendgrid_api_key:
    logger.error("SENDGRID_API_KEY not set in environment variables")
sg = SendGridAPIClient(sendgrid_api_key) if sendgrid_api_key else None

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

def get_db_connection():
    try:
        conn = psycopg2.connect(os.environ['DATABASE_URL'])
        return conn
    except psycopg2.Error as e:
        logger.error(f"Failed to connect to database: {str(e)}")
        raise

def init_search_progress_table():
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute('''
            CREATE TABLE IF NOT EXISTS search_progress (
                user_id TEXT,
                query TEXT DEFAULT '',
                status TEXT NOT NULL,
                timestamp REAL NOT NULL,
                UNIQUE(user_id, query)
            )
        ''')
        conn.commit()
        logger.info("search_progress table initialized successfully")
    except psycopg2.Error as e:
        logger.error(f"Error initializing search_progress table: {str(e)}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()

def init_progress_db():
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute('''
            CREATE TABLE IF NOT EXISTS search_progress (
                user_id TEXT,
                query TEXT DEFAULT '',
                status TEXT NOT NULL,
                timestamp REAL NOT NULL,
                UNIQUE(user_id, query)
            )
        ''')
        cur.execute('''
            CREATE TABLE IF NOT EXISTS grok_cache (
                query TEXT PRIMARY KEY,
                response TEXT NOT NULL,
                timestamp REAL NOT NULL
            )
        ''')
        cur.execute('''
            CREATE TABLE IF NOT EXISTS search_history (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                query TEXT NOT NULL,
                prompt_text TEXT,
                sources TEXT NOT NULL,
                result_ids TEXT NOT NULL,
                timestamp REAL NOT NULL
            )
        ''')
        cur.execute('''
            CREATE TABLE IF NOT EXISTS chat_history (
                id SERIAL PRIMARY KEY,
                user_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                message TEXT NOT NULL,
                is_user BOOLEAN NOT NULL,
                timestamp REAL NOT NULL,
                search_id TEXT
            )
        ''')
        cur.execute('''
            CREATE TABLE IF NOT EXISTS user_settings (
                user_id TEXT PRIMARY KEY,
                chat_memory_retention_hours INTEGER DEFAULT 24
            )
        ''')
        cur.execute('''
            CREATE TABLE IF NOT EXISTS embedding_cache (
                pmid TEXT PRIMARY KEY,
                embedding BYTEA,
                timestamp REAL
            )
        ''')
        conn.commit()
        logger.info("PostgreSQL database initialized successfully")
    except psycopg2.Error as e:
        logger.error(f"Error initializing PostgreSQL database: {str(e)}")
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()

try:
    init_progress_db()
    init_search_progress_table()  # Ensure search_progress table exists
except Exception as e:
    logger.error(f"Failed to initialize database at startup: {str(e)}")

def update_search_progress(user_id, query, status):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO search_progress (user_id, query, status, timestamp) VALUES (%s, %s, %s, %s)
            ON CONFLICT (user_id, query) DO UPDATE SET status = %s, timestamp = %s
        """, (user_id, query or '', status, time.time(), status, time.time()))
        conn.commit()
    except psycopg2.Error as e:
        logger.error(f"Error updating search progress: {str(e)}")
        raise
    finally:
        cur.close()
        conn.close()
    logger.info(f"Search progress updated: user={user_id}, query={query}, status={status}")

def cache_grok_response(query, response):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO grok_cache (query, response, timestamp) VALUES (%s, %s, %s)
            ON CONFLICT (query) DO UPDATE SET response = %s, timestamp = %s
        """, (query, response, time.time(), response, time.time()))
        conn.commit()
    except psycopg2.Error as e:
        logger.error(f"Error caching Grok response: {str(e)}")
    finally:
        cur.close()
        conn.close()
    logger.info(f"Cached Grok response for query: {query[:50]}...")

def get_cached_grok_response(query):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT response, timestamp FROM grok_cache WHERE query = %s AND timestamp > %s",
                    (query, time.time() - 604800))  # Cache valid for 7 days
        result = cur.fetchone()
        return result[0] if result else None
    except psycopg2.Error as e:
        logger.error(f"Error retrieving cached Grok response: {str(e)}")
        return None
    finally:
        cur.close()
        conn.close()

def cache_embedding(pmid, embedding):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO embedding_cache (pmid, embedding, timestamp) VALUES (%s, %s, %s)
            ON CONFLICT (pmid) DO UPDATE SET embedding = %s, timestamp = %s
        """, (pmid, embedding.tobytes(), time.time(), embedding.tobytes(), time.time()))
        conn.commit()
    except psycopg2.Error as e:
        logger.error(f"Error caching embedding: {str(e)}")
    finally:
        cur.close()
        conn.close()

def get_cached_embedding(pmid):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT embedding, timestamp FROM embedding_cache WHERE pmid = %s AND timestamp > %s",
                    (pmid, time.time() - 604800))  # Cache valid for 7 days
        result = cur.fetchone()
        if result:
            embedding = np.frombuffer(result[0], dtype=np.float32)
            if embedding.shape[0] != 384:
                logger.warning(f"Invalid embedding dimension for PMID {pmid}: expected 384, got {embedding.shape[0]}")
                return None
            return embedding
        return None
    except psycopg2.Error as e:
        logger.error(f"Error retrieving cached embedding: {str(e)}")
        return None
    finally:
        cur.close()
        conn.close()

@app.route('/help')
def help():
    response = make_response(render_template('help.html'))
    response.headers['X-Content-Type-Options'] = 'nosniff'
    return response

@tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_exponential(multiplier=1, min=2, max=60),
    retry=tenacity.retry_if_exception_type(Exception),
    before_sleep=lambda retry_state: logger.info(f"Retrying Grok API, attempt {retry_state.attempt_number}")
)
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
            max_tokens=8192
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error querying Grok API: {str(e)}")
        raise