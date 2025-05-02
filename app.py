from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import psycopg2
from werkzeug.security import generate_password_hash, check_password_hash
import spacy
import re
import os
from collections import Counter

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'your-secret-key')
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Database connection function
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

@app.route('/')
def index():
    return render_template('index.html')

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
            return redirect(url_for('login'))
        cur.close()
        conn.close()
    return render_template('register.html')

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
            return redirect(url_for('index'))
        flash('Invalid email or password.', 'error')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

def extract_keywords(query):
    doc = nlp(query.lower())
    stop_words = nlp.Defaults.stop_words | {'information', 'show', 'shows', 'affect', 'affects'}
    keywords = [token.text for token in doc if token.is_alpha and token.text not in stop_words and len(token.text) > 2]
    # Handle multi-word phrases (e.g., "weight loss")
    phrases = [chunk.text for chunk in doc.noun_chunks if chunk.text not in stop_words]
    keywords.extend(phrases)
    # Deduplicate and prioritize phrases
    keywords = list(dict.fromkeys(keywords))
    return keywords

def score_abstract(abstract, keywords):
    abstract_words = re.findall(r'\w+', abstract.lower())
    keyword_counts = sum(abstract_words.count(keyword.lower()) for keyword in keywords)
    return keyword_counts / (len(abstract_words) + 1)  # Normalize by abstract length

def generate_summary(results, query):
    if not results:
        return "No relevant articles found to summarize."
    # Simple TextRank-like summary: extract top sentences from top 3 abstracts
    abstracts = [result['abstract'] for result in results[:3] if result['abstract']]
    if not abstracts:
        return "No abstracts available to summarize."
    doc = nlp(' '.join(abstracts))
    sentences = [sent.text for sent in doc.sents]
    # Score sentences by keyword overlap
    keyword_scores = [(sent, sum(sent.lower().count(keyword.lower()) for keyword in extract_keywords(query))) for sent in sentences]
    top_sentences = sorted(keyword_scores, key=lambda x: x[1], reverse=True)[:2]
    summary = ' '.join(sent for sent, score in top_sentences if score > 0)
    return summary or "Summary could not be generated from available abstracts."

@app.route('/search', methods=['GET', 'POST'])
@login_required
def search():
    if request.method == 'POST':
        query = request.form.get('query')
        if not query:
            return render_template('search.html', error="Query cannot be empty")
        nlp = spacy.load('en_core_web_sm')
        doc = nlp(query)
        keywords = [token.text for token in doc if token.is_alpha and not token.is_stop][:5]
        if not keywords:
            return render_template('search.html', error="No valid keywords found")
        ts_query = ' & '.join(keywords)
        conn = get_db_connection()
        try:
            results = conn.execute(
                "SELECT id, title, abstract, ts_rank(to_tsvector('english', title || ' ' || abstract), to_tsquery(?)) AS rank "
                "FROM articles WHERE to_tsvector('english', title || ' ' || abstract) @@ to_tsquery(?) "
                "ORDER BY rank DESC LIMIT 5",
                (ts_query, ts_query)
            ).fetchall()
            summaries = []
            for r in results:
                summary = generate_summary([{'abstract': r[2]}], query)[:200]
                summaries.append(summary)
        except Exception as e:
            conn.close()
            return render_template('search.html', error=f"Search failed: {str(e)}")
        conn.close()
        return render_template('search.html', results=results, query=query, summaries=summaries)
    return render_template('search.html')

@app.route('/prompt', methods=['GET', 'POST'])
@login_required
def prompt():
    if request.method == 'POST':
        prompt_text = request.form.get('prompt_text')
        if not prompt_text:
            flash('Prompt cannot be empty.', 'error')
        else:
            conn = get_db_connection()
            conn.execute('INSERT INTO prompts (user_id, prompt_text) VALUES (?, ?)', 
                         (current_user.id, prompt_text))
            conn.commit()
            conn.close()
            flash('Prompt saved successfully.', 'success')
    conn = get_db_connection()
    prompts = conn.execute('SELECT id, prompt_text, created_at FROM prompts WHERE user_id = ?', 
                           (current_user.id,)).fetchall()
    conn.close()
    prompts = [{'id': p[0], 'prompt_text': p[1], 'created_at': p[2]} for p in prompts]
    return render_template('prompt.html', prompts=prompts)

@app.route('/notifications', methods=['GET', 'POST'])
@login_required
def notifications():
    conn = get_db_connection()
    cur = conn.cursor()
    if request.method == 'POST':
        keywords = request.form.get('keywords')
        cur.execute("INSERT INTO notifications (user_id, keywords) VALUES (%s, %s) ON CONFLICT (user_id) DO UPDATE SET keywords = %s", 
                    (current_user.id, keywords, keywords))
        conn.commit()
        flash('Notification settings updated.', 'success')
        return redirect(url_for('notifications'))
    cur.execute("SELECT keywords FROM notifications WHERE user_id = %s", (current_user.id,))
    notifications = cur.fetchone()
    cur.close()
    conn.close()
    return render_template('notifications.html', notifications=notifications)

if __name__ == '__main__':
    app.run(debug=True)