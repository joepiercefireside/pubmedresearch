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

# Load spaCy model globally with minimal pipeline
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

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
    return keywords[:3]  # Limit to 3 keywords

def score_abstract(abstract, keywords):
    abstract_words = re.findall(r'\w+', abstract.lower())
    keyword_counts = sum(abstract_words.count(keyword.lower()) for keyword in keywords)
    return keyword_counts / (len(abstract_words) + 1)  # Normalize by abstract length

def generate_summary(abstract, query):
    if not abstract:
        return "No abstract available to summarize."
    # Use first 200 characters of abstract as summary
    return abstract[:200]

@app.route('/search', methods=['GET', 'POST'])
@login_required
def search():
    # Fetch user's prompts for dropdown
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('SELECT id, prompt_name, prompt_text FROM prompts WHERE user_id = %s', 
                (current_user.id,))
    prompts = cur.fetchall()
    cur.close()
    conn.close()
    prompts = [{'id': p[0], 'prompt_name': p[1], 'prompt_text': p[2]} for p in prompts]

    if request.method == 'POST':
        query = request.form.get('query')
        selected_prompt_id = request.form.get('prompt_id')
        if not query:
            return render_template('search.html', error="Query cannot be empty", prompts=prompts)
        keywords = extract_keywords(query)
        if not keywords:
            return render_template('search.html', error="No valid keywords found", prompts=prompts)
        ts_query = ' & '.join(keywords)
        conn = get_db_connection()
        cur = conn.cursor()
        try:
            cur.execute(
                "SELECT id, title, abstract, ts_rank(to_tsvector('english', title || ' ' || abstract), to_tsquery(%s)) AS rank "
                "FROM articles WHERE to_tsvector('english', title || ' ' || abstract) @@ to_tsquery(%s) "
                "ORDER BY rank DESC LIMIT 3",
                (ts_query, ts_query)
            )
            results = cur.fetchall()
            summaries = []
            for r in results:
                summary = generate_summary(r[2], query)
                summaries.append(summary)
            # If a prompt is selected, include its text in the response
            selected_prompt = next((p for p in prompts if str(p['id']) == selected_prompt_id), None)
            prompt_text = selected_prompt['prompt_text'] if selected_prompt else None
        except Exception as e:
            cur.close()
            conn.close()
            return render_template('search.html', error=f"Search failed: {str(e)}", prompts=prompts)
        cur.close()
        conn.close()
        return render_template('search.html', results=results, query=query, summaries=summaries, prompts=prompts, prompt_text=prompt_text)
    return render_template('search.html', prompts=prompts)

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
    cur.execute('SELECT id, prompt_name, prompt_text, created_at FROM prompts WHERE user_id = %s', 
                (current_user.id,))
    prompts = cur.fetchall()
    cur.close()
    conn.close()
    prompts = [{'id': p[0], 'prompt_name': p[1], 'prompt_text': p[2], 'created_at': p[3]} for p in prompts]
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