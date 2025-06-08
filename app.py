from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, Response, session
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import psycopg2
from werkzeug.security import generate_password_hash, check_password_hash
import os
import logging
import json
import sqlite3
import hashlib
from datetime import datetime, timedelta
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import sendgrid
from sendgrid.helpers.mail import Mail, Email, To, Content
import time
import re
import email_validator
from email_validator import validate_email, EmailNotValidError
from openai import OpenAI
from utils import esearch, efetch, parse_e_fetch_xml, search_fda_label_api, extract_keywords_and_date, build_pubmed_query, SearchHandler, PubMedSearchHandler, FDASearchHandler, GoogleScholarSearchHandler

# Configure logging before NLTK imports
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import nltk
import nltk.data

# Set explicit NLTK data path
nltk_data_dir = os.path.join(os.path.dirname(__file__), 'nltk_data')
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)
nltk.data.path.append(nltk_data_dir)

# Download NLTK resources with error handling
try:
    logger.info("Downloading NLTK resources")
    nltk.download('punkt', download_dir=nltk_data_dir)
    nltk.download('stopwords', download_dir=nltk_data_dir)
    nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_dir)
    nltk.download('wordnet', download_dir=nltk_data_dir)
    logger.info("NLTK resources downloaded successfully")
except Exception as e:
    logger.error(f"Failed to download NLTK resources: {str(e)}")

from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'your-secret-key')
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

scheduler = BackgroundScheduler()
scheduler.start()

sendgrid_api_key = os.environ.get('SENDGRID_API_KEY', '').strip()
if not sendgrid_api_key:
    logger.error("SENDGRID_API_KEY not set in environment variables")
sg = sendgrid.SendGridAPIClient(sendgrid_api_key) if sendgrid_api_key else None

# Add datetimeformat filter
def datetimeformat(value, format='%Y-%m-%d %H:%M:%S'):
    try:
        dt = datetime.fromtimestamp(value)
        return dt.strftime(format)
    except (ValueError, TypeError):
        return str(value)

app.jinja_env.filters['datetimeformat'] = datetimeformat

def init_progress_db():
    conn = sqlite3.connect('search_progress.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS search_progress
                 (user_id TEXT, query TEXT, status TEXT, timestamp REAL)''')
    c.execute('''CREATE TABLE IF NOT EXISTS grok_cache
                 (query TEXT PRIMARY KEY, response TEXT, timestamp REAL)''')
    c.execute('''CREATE TABLE IF NOT EXISTS search_history
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, query TEXT, prompt_text TEXT, sources TEXT, result_ids TEXT, timestamp REAL)''')
    c.execute('''CREATE TABLE IF NOT EXISTS search_results
                 (id TEXT PRIMARY KEY, source_id TEXT, result_data TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS chat_history
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, session_id TEXT, message TEXT, is_user BOOLEAN, timestamp REAL)''')
    c.execute('''CREATE TABLE IF NOT EXISTS user_settings
                 (user_id TEXT PRIMARY KEY, chat_memory_retention_hours INTEGER DEFAULT 24)''')
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

def save_search_history(user_id, query, prompt_text, sources, results):
    conn = sqlite3.connect('search_progress.db')
    c = conn.cursor()
    result_ids = []
    for result in results:
        result_id = hashlib.md5(json.dumps(result, sort_keys=True).encode()).hexdigest()
        c.execute("INSERT OR REPLACE INTO search_results (id, source_id, result_data) VALUES (?, ?, ?)",
                  (result_id, result.get('source_id', 'unknown'), json.dumps(result)))
        result_ids.append(result_id)
    c.execute("INSERT INTO search_history (user_id, query, prompt_text, sources, result_ids, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
              (user_id, query, prompt_text, json.dumps(sources), json.dumps(result_ids), time.time()))
    conn.commit()
    conn.close()
    logger.info(f"Saved search history for user={user_id}, query={query}")
    return result_ids

def get_search_history(user_id):
    conn = sqlite3.connect('search_progress.db')
    c = conn.cursor()
    c.execute("SELECT id, query, prompt_text, sources, result_ids, timestamp FROM search_history WHERE user_id = ? ORDER BY timestamp DESC",
              (user_id,))
    results = [
        {'id': row[0], 'query': row[1], 'prompt_text': row[2], 'sources': json.loads(row[3]), 'result_ids': json.loads(row[4]), 'timestamp': row[5]}
        for row in c.fetchall()
    ]
    conn.close()
    return results

def get_search_results(result_ids):
    conn = sqlite3.connect('search_progress.db')
    c = conn.cursor()
    results = []
    for result_id in result_ids:
        c.execute("SELECT result_data FROM search_results WHERE id = ?", (result_id,))
        result = c.fetchone()
        if result:
            results.append(json.loads(result[0]))
    conn.close()
    return results

def save_chat_message(user_id, session_id, message, is_user):
    conn = sqlite3.connect('search_progress.db')
    c = conn.cursor()
    c.execute("INSERT INTO chat_history (user_id, session_id, message, is_user, timestamp) VALUES (?, ?, ?, ?, ?)",
              (user_id, session_id, message, is_user, time.time()))
    conn.commit()
    conn.close()

def get_chat_history(user_id, session_id, retention_hours):
    conn = sqlite3.connect('search_progress.db')
    c = conn.cursor()
    cutoff_time = time.time() - (retention_hours * 3600)
    c.execute("SELECT message, is_user, timestamp FROM chat_history WHERE user_id = ? AND session_id = ? AND timestamp > ? ORDER BY timestamp ASC",
              (user_id, session_id, cutoff_time))
    messages = [{'message': row[0], 'is_user': row[1], 'timestamp': row[2]} for row in c.fetchall()]
    conn.close()
    return messages

def get_user_settings(user_id):
    conn = sqlite3.connect('search_progress.db')
    c = conn.cursor()
    c.execute("SELECT chat_memory_retention_hours FROM user_settings WHERE user_id = ?", (user_id,))
    result = c.fetchone()
    conn.close()
    return result[0] if result else 24

def update_user_settings(user_id, retention_hours):
    conn = sqlite3.connect('search_progress.db')
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO user_settings (user_id, chat_memory_retention_hours) VALUES (?, ?)",
              (user_id, retention_hours))
    conn.commit()
    conn.close()

def get_db_connection():
    conn = psycopg2.connect(os.environ['DATABASE_URL'])
    return conn

class User(UserMixin):
    def __init__(self, id, email, admin=False):
        self.id = id
        self.email = email
        self.admin = admin

@login_manager.user_loader
def load_user(user_id):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, email, admin FROM users WHERE id = %s", (user_id,))
    user = cur.fetchone()
    cur.close()
    conn.close()
    if user:
        return User(user[0], user[1], user[2])
    return None

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
            'summary_result_count': 20,
            'display_result_count': 80,
            'limit_presentation': False
        }
    
    prompt_text_lower = prompt_text.lower()
    summary_result_count = 20
    if match := re.search(r'(?:top|return|summarize|include|limit\s+to|show\s+only)\s+(\d+)\s+(?:articles|results)', prompt_text_lower):
        summary_result_count = min(int(match.group(1)), 20)
    elif 'top' in prompt_text_lower:
        summary_result_count = 3
    
    display_result_count = 80
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
            max_tokens=1000
        )
        return completion.choices[0].message.content
    except Exception as e:
        logger.error(f"Error querying Grok API: {str(e)}")
        raise

def run_notification_rule(rule_id, user_id, rule_name, keywords, timeframe, prompt_text, email_format, user_email, sources, test_mode=False):
    logger.info(f"Running notification rule {rule_id} ({rule_name}) for user {user_id}, keywords: {keywords}, timeframe: {timeframe}, sources: {sources}, test_mode: {test_mode}, recipient: {user_email}")
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
    date_range = f"{start_date}[dp]:{today.strftime('%Y/%m/%d')}[dp]"
    
    search_handlers = {
        'pubmed': PubMedSearchHandler(),
        'fda': FDASearchHandler(),
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
            if primary_results is not None:
                results.extend([dict(r, source_id=source_id) for r in primary_results])
        
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
        
        context = "\n".join([f"Title: {r['title']}\nAbstract: {r.get('abstract', r.get('summary', ''))}\nAuthors: {r.get('authors', 'N/A')}\nDate: {r.get('publication_date', r.get('date', 'N/A'))}" for r in results])
        output = query_grok_api(prompt_text or "Summarize the provided research articles.", context)
        
        if email_format == "list":
            content = "\n".join([f"- {r['title']} ({r.get('publication_date', r.get('date', 'N/A'))})\n  {r.get('abstract', r.get('summary', ''))[:100] or 'No abstract'}..." for r in results])
        elif email_format == "detailed":
            content = "\n".join([f"Title: {r['title']}\nAuthors: {r.get('authors', 'N/A')}\nJournal: {r.get('journal', 'N/A')}\nDate: {r.get('publication_date', r.get('date', 'N/A'))}\nAbstract: {r.get('abstract', r.get('summary', '')) or 'No abstract'}\n" for r in results])
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
                "message_id": None
            }
        raise

def schedule_notification_rules():
    scheduler.remove_all_jobs()
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT n.id, n.user_id, n.rule_name, n.keywords, n.timeframe, n.prompt_text, n.email_format, u.email, n.sources "
        "FROM notifications n JOIN users u ON n.user_id = u.id"
    )
    rules = cur.fetchall()
    cur.close()
    conn.close()
    
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
    # Clean up session to reduce cookie size
    session.pop('latest_search_result_ids', None)
    import psutil
    logger.debug(f"Memory usage: {psutil.Process().memory_info().rss / 1024**2:.2f} MB")

    page = int(request.args.get('page', 1))
    per_page = 20
    sort_by = request.form.get('sort_by', request.args.get('sort_by', 'relevance'))
    filter_sources = request.form.getlist('filter_sources') or request.args.getlist('filter_sources') or []
    logger.debug(f"Filter sources initialized: type={type(filter_sources)}, value={filter_sources}")

    prompt_id = request.form.get('prompt_id', request.args.get('prompt_id', ''))
    prompt_text = request.form.get('prompt_text', request.args.get('prompt_text', ''))
    query = request.form.get('query', request.args.get('query', ''))
    search_older = request.form.get('search_older', 'off') == 'on' or request.args.get('search_older', 'False') == 'True'
    start_year = request.form.get('start_year', request.args.get('start_year', None))
    # Validate start_year
    if start_year == "None" or not start_year:
        start_year = None
    else:
        try:
            start_year = int(start_year)
        except ValueError:
            start_year = None
    sources_selected = request.form.getlist('sources') or request.args.getlist('sources') or []
    logger.debug(f"Sources selected initialized: type={type(sources_selected)}, value={sources_selected}")

    selected_prompt_text = prompt_text
    if prompt_id and not prompt_text:
        for prompt in prompts:
            if prompt['id'] == prompt_id:
                selected_prompt_text = prompt['prompt_text']
                break
        else:
            logger.warning(f"Prompt ID {prompt_id} not found in prompts")
            selected_prompt_text = ''

    logger.info(f"Search request: prompt_id={prompt_id}, prompt_text={prompt_text[:50]}..., query={query[:50]}..., search_older={search_older}, start_year={start_year}, sources={sources_selected}, page={page}, sort_by={sort_by}, filter_sources={filter_sources}")

    search_handlers = {
        'pubmed': PubMedSearchHandler(),
        'fda': FDASearchHandler(),
        'googlescholar': GoogleScholarSearchHandler()
    }

    if request.method == 'POST':
        if not query:
            update_search_progress(current_user.id, query, "error: Query cannot be empty")
            return render_template('search.html', error="Query cannot be empty", prompts=prompts, prompt_id=prompt_id, prompt_text=selected_prompt_text, sources=[], total_results=0, page=page, per_page=per_page, username=current_user.email, has_prompt=bool(selected_prompt_text), prompt_params={}, summary_result_count=5, search_older=search_older, start_year=start_year, sort_by=sort_by, filter_sources=filter_sources, pubmed_results=[], pubmed_fallback_results=[])

        if not sources_selected:
            update_search_progress(current_user.id, query, "error: At least one search source must be selected")
            return render_template('search.html', error="At least one search source must be selected", prompts=prompts, prompt_id=prompt_id, prompt_text=selected_prompt_text, sources=[], total_results=0, page=page, per_page=per_page, username=current_user.email, has_prompt=bool(selected_prompt_text), prompt_params={}, summary_result_count=5, search_older=search_older, start_year=start_year, sort_by=sort_by, filter_sources=filter_sources, pubmed_results=[], pubmed_fallback_results=[])

        update_search_progress(current_user.id, query, "contacting APIs")

        try:
            keywords_with_synonyms, date_range, start_year_int = extract_keywords_and_date(query, search_older, start_year)
            if not keywords_with_synonyms:
                update_search_progress(current_user.id, query, "error: No valid keywords found")
                return render_template('search.html', error="No valid keywords found", prompts=prompts, prompt_id=prompt_id, prompt_text=selected_prompt_text, sources=[], total_results=0, page=page, per_page=per_page, username=current_user.email, has_prompt=bool(selected_prompt_text), prompt_params={}, summary_result_count=5, search_older=search_older, start_year=start_year, sort_by=sort_by, filter_sources=filter_sources, pubmed_results=[], pubmed_fallback_results=[])

            prompt_params = parse_prompt(selected_prompt_text) or {}
            prompt_params['sort_by'] = sort_by
            summary_result_count = prompt_params.get('summary_result_count', 20)

            sources = []
            total_results = 0
            pubmed_results = []
            fda_results = []
            googlescholar_results = []
            pubmed_fallback_results = []
            all_results = []

            for source_id in sources_selected:
                if source_id not in search_handlers:
                    logger.warning(f"Unknown source: {source_id}")
                    continue

                handler = search_handlers[source_id]
                update_search_progress(current_user.id, query, f"executing {handler.name} search")

                primary_results, fallback_results = handler.search(query, keywords_with_synonyms, date_range, start_year_int)

                primary_results = primary_results or [][:20]  # Limit to 20 results
                fallback_results = fallback_results or [][:20]

                ranked_results = []
                if primary_results:
                    update_search_progress(current_user.id, query, f"ranking {handler.name} results")
                    ranked_results = handler.rank_results(query, primary_results, prompt_params)

                summary = ""
                if selected_prompt_text and (ranked_results or primary_results):
                    update_search_progress(current_user.id, query, f"generating {handler.name} summary")
                    summary = handler.generate_summary(query, ranked_results, selected_prompt_text, prompt_params)

                source_data = {
                    'id': handler.source_id,
                    'name': handler.name,
                    'results': {
                        'ranked': ranked_results,
                        'all': primary_results,
                        'fallback': fallback_results
                    },
                    'summary': summary
                }
                logger.debug(f"Summary for {handler.name}: {summary[:200]}...")

                if source_id == 'pubmed':
                    if primary_results or fallback_results:
                        conn = get_db_connection()
                        cur = conn.cursor()
                        cur.execute("INSERT INTO search_cache (query, results) VALUES (%s, %s)", 
                                    (query, json.dumps(primary_results + fallback_results)))
                        conn.commit()
                        cur.close()
                        conn.close()
                        pubmed_results = primary_results
                        pubmed_fallback_results = fallback_results
                elif source_id == 'fda':
                    fda_results = primary_results
                elif source_id == 'googlescholar':
                    googlescholar_results = primary_results

                total_results += len(primary_results) + len(fallback_results)
                sources.append(source_data)

                all_results.extend([dict(r, source_id=source_id) for r in primary_results])
                all_results.extend([dict(r, source_id=source_id) for r in fallback_results])

            logger.debug(f"Before filtering: sources={len(sources)}, total_results={total_results}, filter_sources={filter_sources}, sources_selected={sources_selected}")
            if isinstance(filter_sources, list) and filter_sources and isinstance(sources_selected, list) and len(sources_selected) > 1:
                sources = [s for s in sources if s['id'] in filter_sources]
                total_results = sum(len(s['results']['all']) + len(s['results']['fallback']) for s in sources)
                all_results = [r for r in all_results if r['source_id'] in filter_sources]
            logger.debug(f"After filtering: sources={len(sources)}, total_results={total_results}")

            start_idx = (page - 1) * per_page
            end_idx = start_idx + per_page

            for source in sources:
                source['results']['ranked'] = source['results']['ranked'][start_idx:end_idx]
                source['results']['all'] = source['results']['all'][start_idx:end_idx]
                source['results']['fallback'] = source['results']['fallback'][start_idx:end_idx]

            total_pages = (total_results + per_page - 1) // per_page

            result_ids = save_search_history(current_user.id, query, selected_prompt_text, sources_selected, all_results)
            session['latest_search_result_ids'] = json.dumps(result_ids[:10])
            session['latest_query'] = query

            update_search_progress(current_user.id, query, "complete")

            return render_template(
                'search.html', 
                sources=sources,
                total_results=total_results,
                page=page,
                per_page=per_page,
                total_pages=total_pages,
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
                filter_sources=filter_sources,
                pubmed_results=pubmed_results,
                fda_results=fda_results,
                googlescholar_results=googlescholar_results,
                pubmed_fallback_results=pubmed_fallback_results
            )
        except Exception as e:
            logger.error(f"API error: {str(e)}")
            update_search_progress(current_user.id, query, f"error: Search failed: {str(e)}")
            return render_template('search.html', error=f"Search failed: {str(e)}", prompts=prompts, prompt_id=prompt_id, prompt_text=selected_prompt_text, sources=[], total_results=0, page=page, per_page=per_page, username=current_user.email, has_prompt=bool(selected_prompt_text), prompt_params={}, summary_result_count=5, search_older=search_older, start_year=start_year, sort_by=sort_by, filter_sources=filter_sources, pubmed_results=[], pubmed_fallback_results=[])

@app.route('/search_summary', methods=['POST'])
@login_required
def search_summary():
    query = request.form.get('query', '')
    prompt_text = request.form.get('prompt_text', '')
    results = json.loads(request.form.get('results', '[]'))
    fallback_results = json.loads(request.form.get('fallback_results', '[]'))
    prompt_params = json.loads(request.form.get('prompt_params', '{}'))
    source_id = request.form.get('source_id', 'pubmed')
    
    logger.info(f"Received summary request for query: {query[:50]}... with prompt: {prompt_text[:50]}... source: {source_id}")
    
    try:
        handler = next((h for h in [PubMedSearchHandler(), FDASearchHandler(), GoogleScholarSearchHandler()] if h.source_id == source_id), None)
        if not handler:
            raise ValueError(f"Unknown source: {source_id}")
        
        prompt_output = handler.generate_summary(query, results, prompt_text, prompt_params)
        fallback_prompt_output = handler.generate_summary(query, fallback_results, prompt_text, prompt_params) if fallback_results and source_id == 'pubmed' else ''
        
        logger.info(f"Summary generated: prompt_output length={len(prompt_output)}, fallback_output length={len(fallback_prompt_output) if fallback_prompt_output else 0}")
        
        if prompt_output.startswith("Unable to generate"):
            flash(f"AI summarization failed for {source_id} results.", "warning")
        if fallback_prompt_output and fallback_prompt_output.startswith("Unable to generate"):
            flash("AI summarization failed for PubMed fallback results.", "warning")
        
        update_search_progress(current_user.id, query, "complete")
        
        return jsonify({
            'status': 'success',
            'prompt_output': prompt_output,
            'fallback_prompt_output': fallback_prompt_output
        })
    except Exception as e:
        logger.error(f"Error generating AI summary for {source_id}: {str(e)}")
        update_search_progress(current_user.id, query, f"error: AI summary failed: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f"AI summary failed: {str(e)}"
        })

@app.route('/chat', methods=['GET', 'POST'])
@login_required
def chat():
    if not current_user.is_authenticated:
        return redirect(url_for('login'))
    
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
            
            result_ids = json.loads(session.get('latest_search_result_ids', '[]'))
            search_results = get_search_results(result_ids)
            query = session.get('latest_query', '')
            context = "\n".join([f"Title: {r['title']}\nAbstract: {r.get('abstract', r.get('summary', ''))}\nAuthors: {r.get('authors', 'N/A')}\nDate: {r.get('publication_date', r.get('date', 'N/A'))}" for r in search_results])
            
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
        
        return render_template('chat.html', chat_history=chat_history, username=current_user.email, retention_hours=retention_hours)
    
    return render_template('chat.html', chat_history=chat_history, username=current_user.email, retention_hours=retention_hours)

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
        
        result_ids = json.loads(session.get('latest_search_result_ids', '[]'))
        search_results = get_search_results(result_ids)
        query = session.get('latest_query', '')
        context = "\n".join([f"Title: {r['title']}\nAbstract: {r.get('abstract', r.get('summary', ''))}\nAuthors: {r.get('authors', 'N/A')}\nDate: {r.get('publication_date', r.get('date', 'N/A'))}" for r in search_results])
        
        with open('static/templates/chatbot_prompt.txt', 'r', encoding='utf-8') as f:
            system_prompt = f.read()
        
        history_context = "\n".join([f"{'User' if msg['is_user'] else 'Assistant'}: {msg['message']}" for msg in chat_history[-5:]])
        full_context = f"Search Query: {query}\n\nSearch Results:\n{context}\n\nChat History:\n{history_context}\n\nUser Query: {user_message}"
        
        response = query_grok_api(system_prompt, full_context)
        save_chat_message(current_user.id, session_id, response, False)
        
        return jsonify({'status': 'success', 'message': response})
    except Exception as e:
        logger.error(f"Error in chat_message: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/previous_searches', methods=['GET'])
@login_required
def previous_searches():
    search_history = get_search_history(current_user.id)
    return render_template('previous_searches.html', searches=search_history, username=current_user.email)

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
                conn.rollback()
                logger.error(f"Error creating notification: {str(e)}")
                flash(f'Failed to create notification rule: {str(e)}', 'error')
    
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
    cur.close()
    conn.close()
    return render_template('notifications.html', notifications=notifications, username=current_user.email, pre_query=pre_query, pre_prompt=pre_prompt)

@app.route('/notifications/edit/<int:id>', methods=['GET', 'POST'])
@login_required
def edit_notification(id):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, rule_name, keywords, timeframe, prompt_text, email_format, sources "
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
                cur.close()
                conn.close()
                return redirect(url_for('notifications'))
            except Exception as e:
                conn.rollback()
                logger.error(f"Error updating notification: {str(e)}")
                flash(f'Failed to update notification rule: {str(e)}', 'error')
    
    cur.close()
    conn.close()
    notification_data = {
        'id': notification[0],
        'rule_name': notification[1],
        'keywords': notification[2],
        'timeframe': notification[3],
        'prompt_text': notification[4],
        'email_format': notification[5],
        'sources': json.loads(notification[6]) if notification[6] else ['pubmed']
    }
    return render_template('notification_edit.html', notification=notification_data, username=current_user.email)

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
        "SELECT n.id, n.user_id, n.rule_name, n.keywords, n.timeframe, n.prompt_text, n.email_format, u.email, n.sources "
        "FROM notifications n JOIN users u ON n.user_id = u.id WHERE n.id = %s AND n.user_id = %s",
        (id, current_user.id)
    )
    rule = cur.fetchone()
    cur.close()
    conn.close()
    
    if not rule:
        flash('Notification rule not found or you do not have permission to test it.', 'error')
        return jsonify({"status": "error", "message": "Rule not found"}), 404
    
    rule_id, user_id, rule_name, keywords, timeframe, prompt_text, email_format, user_email, sources = rule
    sources = json.loads(sources) if sources else ['pubmed']
    try:
        test_result = run_notification_rule(
            rule_id, user_id, rule_name, keywords, timeframe, prompt_text, email_format, user_email, sources, test_mode=True
        )
        return jsonify(test_result)
    except Exception as e:
        logger.error(f"Error testing notification rule {rule_id}: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/help')
def help():
    return render_template('help.html', username=current_user.email if current_user.is_authenticated else None)

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

schedule_notification_rules()

if __name__ == '__main__':
    app.run(debug=True)