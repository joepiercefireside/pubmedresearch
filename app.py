from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
import os
import requests
import xml.etree.ElementTree as ET
from psycopg2.extras import RealDictCursor
import psycopg2
import bcrypt
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key')
app.config['DATABASE_URL'] = os.environ.get('DATABASE_URL')

# User model
class User(UserMixin):
    def __init__(self, id, email):
        self.id = id
        self.email = email

# Database connection
def get_db_connection():
    conn = psycopg2.connect(app.config['DATABASE_URL'])
    return conn

# Initialize database
def init_db():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            email VARCHAR(255) UNIQUE NOT NULL,
            password_hash VARCHAR(255)
        );
        CREATE TABLE IF NOT EXISTS articles (
            pmid VARCHAR(50) PRIMARY KEY,
            title TEXT,
            abstract TEXT,
            authors JSONB,
            keywords JSONB,
            publication_date DATE
        );
        CREATE TABLE IF NOT EXISTS prompts (
            id SERIAL PRIMARY KEY,
            user_id INTEGER REFERENCES users(id),
            prompt_name VARCHAR(255),
            prompt_text TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS notifications (
            id SERIAL PRIMARY KEY,
            user_id INTEGER REFERENCES users(id),
            keywords JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    cur.close()
    conn.close()

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

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

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT id, email, password_hash FROM users WHERE email = %s", (email,))
        user = cur.fetchone()
        cur.close()
        conn.close()
        if user and bcrypt.checkpw(password.encode('utf-8'), user[2].encode('utf-8')):
            login_user(User(user[0], user[1]))
            return redirect(url_for('index'))
        flash('Invalid email or password', 'danger')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT id FROM users WHERE email = %s", (email,))
        if cur.fetchone():
            flash('Email already registered', 'danger')
            cur.close()
            conn.close()
            return render_template('register.html')
        password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        cur.execute("INSERT INTO users (email, password_hash) VALUES (%s, %s) RETURNING id", (email, password_hash))
        user_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()
        login_user(User(user_id, email))
        return redirect(url_for('index'))
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))

@app.route('/')
@login_required
def index():
    return render_template('index.html')

@app.route('/search', methods=['GET', 'POST'])
@login_required
def search():
    if request.method == 'POST':
        query = request.form.get('query')
        prompt_id = request.form.get('prompt_id')
        # Placeholder for LLM search
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT * FROM articles WHERE to_tsvector('english', title || ' ' || abstract) @@ to_tsquery(%s)", (query,))
        results = cur.fetchall()
        cur.close()
        conn.close()
        return render_template('search.html', results=results, query=query)
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, prompt_name FROM prompts WHERE user_id = %s", (current_user.id,))
    prompts = cur.fetchall()
    cur.close()
    conn.close()
    return render_template('search.html', prompts=prompts)

@app.route('/prompt', methods=['GET', 'POST'])
@login_required
def prompt():
    if request.method == 'POST':
        prompt_name = request.form.get('prompt_name')
        prompt_text = request.form.get('prompt_text')
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("INSERT INTO prompts (user_id, prompt_name, prompt_text) VALUES (%s, %s, %s)", 
                    (current_user.id, prompt_name, prompt_text))
        conn.commit()
        cur.close()
        conn.close()
        flash('Prompt created successfully.', 'success')
        return redirect(url_for('prompt'))
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("SELECT id, prompt_name, prompt_text, created_at FROM prompts WHERE user_id = %s", (current_user.id,))
    prompts = cur.fetchall()
    cur.close()
    conn.close()
    return render_template('prompt.html', prompts=prompts)

@app.route('/notifications', methods=['GET', 'POST'])
@login_required
def notifications():
    if request.method == 'POST':
        keywords = request.form.get('keywords').split(',')
        keywords_json = [k.strip() for k in keywords]
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("INSERT INTO notifications (user_id, keywords) VALUES (%s, %s)", 
                    (current_user.id, keywords_json))
        conn.commit()
        cur.close()
        conn.close()
        flash('Notification settings saved.', 'success')
        return redirect(url_for('notifications'))
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("SELECT id, keywords, created_at FROM notifications WHERE user_id = %s", (current_user.id,))
    notifications = cur.fetchall()
    cur.close()
    conn.close()
    return render_template('notifications.html', notifications=notifications)

if __name__ == '__main__':
    init_db()
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))