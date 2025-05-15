from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import psycopg2
from werkzeug.security import generate_password_hash, check_password_hash
import requests
import os
import logging
from dotenv import load_dotenv
import aiohttp
import asyncio

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'your-secret-key')
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database connection
def get_db_connection():
    conn = psycopg2.connect(os.environ['DATABASE_URL'])
    return conn

# User class for Flask-Login
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

# Helper function to extract keywords
def extract_keywords(query):
    return query.split()  # Simple keyword extraction

# PubMed search function
def search_pubmed(query, start_year=None):
    api_key = os.environ.get('PUBMED_API_KEY')
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    date_range = f"{start_year}/01/01:3000/12/31[dp]" if start_year else ""
    search_term = f"{query} AND {date_range}" if date_range else query
    params = {
        "db": "pubmed",
        "term": search_term,
        "retmax": 10,
        "retmode": "json",
        "api_key": api_key
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        pmids = data['esearchresult']['idlist']
        if pmids:
            fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
            fetch_params = {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml", "api_key": api_key}
            fetch_response = requests.get(fetch_url, params=fetch_params)
            if fetch_response.status_code == 200:
                from xml.etree import ElementTree
                root = ElementTree.fromstring(fetch_response.content)
                results = []
                for article in root.findall(".//PubmedArticle"):
                    title = article.find(".//ArticleTitle").text if article.find(".//ArticleTitle") is not None else "No title"
                    abstract = article.find(".//AbstractText")
                    abstract = abstract.text if abstract is not None else "No abstract"
                    results.append({"title": title, "abstract": abstract})
                return results
    return []

# Grok API call for ranking or responses
async def query_grok_api(query, context, prompt="Rank these articles by relevance to the query or provide a response based on the prompt."):
    api_key = os.environ.get('XAI_API_KEY')
    if not api_key:
        logger.error("XAI_API_KEY not set")
        return None
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
                {"role": "user", "content": f"Query: {query}\n\nContext: {context}"}
            ],
            "max_tokens": 1000
        }
        try:
            async with session.post(url, headers=headers, json=data, timeout=30) as response:
                response.raise_for_status()
                response_json = await response.json()
                return response_json['choices'][0]['message']['content']
        except Exception as e:
            logger.error(f"Grok API call failed: {str(e)}")
            return None

# Routes
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

@app.route('/search', methods=['GET', 'POST'])
@login_required
async def search():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('SELECT id, prompt_name, prompt_text FROM prompts WHERE user_id = %s', (current_user.id,))
    prompts = [{'id': str(p[0]), 'prompt_name': p[1], 'prompt_text': p[2]} for p in cur.fetchall()]
    cur.close()
    conn.close()

    if request.method == 'POST':
        query = request.form.get('query', '')
        prompt_id = request.form.get('prompt_id', '')
        start_year = request.form.get('start_year', None)
        if not query:
            flash('Query cannot be empty.', 'error')
        else:
            try:
                # Get prompt text if selected
                prompt_text = next((p['prompt_text'] for p in prompts if p['id'] == prompt_id), None)

                # Extract keywords and search PubMed
                keywords = extract_keywords(query)
                search_query = " AND ".join(keywords)
                results = search_pubmed(search_query, start_year)

                if results:
                    # Prepare context for AI
                    context = "\n".join([f"Title: {r['title']}\nAbstract: {r['abstract']}" for r in results])

                    # Rank results using Grok AI
                    ranking_prompt = f"Rank these articles by relevance to '{query}'"
                    ranked_response = await query_grok_api(query, context, prompt=ranking_prompt)
                    ranked_results = results  # Simplified; assumes Grok returns usable ranking

                    # Generate AI response if prompt provided
                    ai_response = None
                    if prompt_text:
                        response_prompt = f"{prompt_text}\n\nContext: {context}"
                        ai_response = await query_grok_api(query, context, prompt=response_prompt)

                    return render_template('search.html', results=results, ranked_results=ranked_results, query=query, prompts=prompts, ai_response=ai_response, prompt_text=prompt_text, username=current_user.email)
                else:
                    flash('No results found.', 'error')
            except Exception as e:
                flash(f'Search failed: {str(e)}', 'error')
    return render_template('search.html', prompts=prompts, username=current_user.email)

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

@app.route('/notifications', methods=['GET', 'POST'])
@login_required
def notifications():
    if request.method == 'POST':
        rule_name = request.form.get('rule_name')
        keywords = request.form.get('keywords')
        timeframe = request.form.get('timeframe')
        prompt_text = request.form.get('prompt_text', '')
        email_format = request.form.get('email_format')
        if not all([rule_name, keywords, timeframe, email_format]):
            flash('All fields except prompt text are required.', 'error')
        else:
            conn = get_db_connection()
            cur = conn.cursor()
            try:
                cur.execute('INSERT INTO notifications (user_id, rule_name, keywords, timeframe, prompt_text, email_format) VALUES (%s, %s, %s, %s, %s, %s)', 
                            (current_user.id, rule_name, keywords, timeframe, prompt_text, email_format))
                conn.commit()
                flash('Notification rule added successfully.', 'success')
            except Exception as e:
                conn.rollback()
                flash(f'Failed to add notification rule: {str(e)}', 'error')
            finally:
                cur.close()
                conn.close()
    
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('SELECT id, rule_name, keywords, timeframe, prompt_text, email_format, created_at FROM notifications WHERE user_id = %s ORDER BY created_at DESC', 
                (current_user.id,))
    notifications = [{'id': str(n[0]), 'rule_name': n[1], 'keywords': n[2], 'timeframe': n[3], 'prompt_text': n[4], 'email_format': n[5], 'created_at': n[6]} for n in cur.fetchall()]
    cur.close()
    conn.close()
    return render_template('notifications.html', notifications=notifications, username=current_user.email)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)