from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import psycopg2
from werkzeug.security import generate_password_hash, check_password_hash
import spacy
import re
import os
import logging
import json
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from scipy.spatial.distance import cosine

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'your-secret-key')
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load spaCy model globally with minimal pipeline
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# Initialize MobileBERT model and tokenizer lazily
tokenizer = None
model = None

def load_mobilebert_model():
    global tokenizer, model
    if tokenizer is None or model is None:
        logger.info("Loading MobileBERT model...")
        tokenizer = AutoTokenizer.from_pretrained('google/mobilebert-uncased')
        model = AutoModel.from_pretrained('google/mobilebert-uncased')
        logger.info("MobileBERT model loaded.")
    return tokenizer, model

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
    stop_words = nlp.Defaults.stop_words
    keywords = [token.text for token in doc if token.is_alpha and token.text not in stop_words and len(token.text) > 1]
    intent = {'recent': 'recent' in query.lower()}
    logger.info(f"Extracted keywords: {keywords}, Intent: {intent}")
    if not keywords:
        keywords = [word for word in query.lower().split() if word not in stop_words and len(word) > 1]
        logger.info(f"Fallback keywords: {keywords}")
    return keywords[:3], intent

def parse_prompt(prompt_text):
    """Parse prompt for result count and output type."""
    if not prompt_text:
        return 20, "summary"
    prompt_text = prompt_text.lower()
    # Extract result count
    match = re.search(r'return\s+(\d+)\s+results', prompt_text)
    result_count = int(match.group(1)) if match else 20
    # Determine output type
    if "summary article" in prompt_text:
        output_type = "article"
    elif "letter" in prompt_text:
        output_type = "letter"
    elif "answer" in prompt_text:
        output_type = "answer"
    else:
        output_type = "summary"
    logger.info(f"Parsed prompt: result_count={result_count}, output_type={output_type}")
    return result_count, output_type

def generate_embedding(text):
    """Generate embedding for a single text."""
    tokenizer, model = load_mobilebert_model()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def mock_llm_ranking(query, results, embeddings):
    """Mock LLM ranking using cosine similarity of embeddings."""
    query_embedding = generate_embedding(query)
    scores = []
    for i, emb in enumerate(embeddings):
        if emb is not None:
            similarity = 1 - cosine(query_embedding, emb)
            scores.append((i, similarity))
        else:
            scores.append((i, 0.0))
    scores.sort(key=lambda x: x[1], reverse=True)
    ranked_indices = [i for i, _ in scores]
    ranked_results = [results[i] for i in ranked_indices]
    ranked_embeddings = [embeddings[i] for i in ranked_indices]
    return ranked_results, ranked_embeddings

def generate_summary(abstract, query, prompt_text=None, title=None, authors=None, journal=None, publication_date=None):
    if not abstract and not title:
        return {"text": "No content available to summarize.", "metadata": {}, "embedding": None}
    # Generate embedding
    text = f"{title} {abstract or ''} {authors or ''} {journal or ''}".strip()
    embedding = generate_embedding(text) if text else None
    # Generate summary based on prompt
    summary_text = abstract[:200] if abstract else f"Title: {title}"
    if prompt_text:
        logger.info(f"Processing prompt: {prompt_text}")
        prompt_text_lower = prompt_text.lower()
        if "insights" in prompt_text_lower:
            summary_text = abstract[:300] if abstract else f"Title: {title} (Insights mode)"
        elif "google" in prompt_text_lower:
            summary_text = abstract[:100] if abstract else f"Title: {title} (Google mode)"
        elif "expert" in prompt_text_lower:
            summary_text = f"Expert Summary: {abstract[:250] if abstract else title}" if abstract else f"Title: {title} (Expert mode)"
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

def generate_prompt_output(query, results, prompt_text, output_type):
    """Generate flexible prompt-driven output (summary, article, letter, answer)."""
    if not results:
        return f"No results found for '{query}'."
    # Mock LLM output (replace with xAI API later)
    combined_text = "\n".join([f"{r[1]}: {r[2] or 'No abstract'}" for r in results])
    if output_type == "article":
        output = f"Summary Article for '{query}':\n\nBased on recent PubMed data, the following insights were found:\n{combined_text[:1000]}\n\nThis article summarizes key findings."
    elif output_type == "letter":
        output = f"Dear Researcher,\n\nRegarding '{query}', PubMed data suggests:\n{combined_text[:500]}\n\nSincerely,\nPubMed Research Team"
    elif output_type == "answer":
        output = f"Answer to '{query}':\n\n{combined_text[:200]}"
    else:
        output = f"Summary for '{query}':\n\n{combined_text[:300]}"
    logger.info(f"Generated prompt output: type={output_type}, length={len(output)}")
    return output

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
        logger.info(f"Search query: {query}, selected prompt ID: {selected_prompt_id}")
        if not query:
            return render_template('search.html', error="Query cannot be empty", prompts=prompts)
        keywords, intent = extract_keywords(query)
        if not keywords:
            return render_template('search.html', error="No valid keywords found", prompts=prompts)
        
        # Parse prompt for result count and output type
        selected_prompt = next((p for p in prompts if str(p['id']) == selected_prompt_id), None)
        logger.info(f"Selected prompt: {selected_prompt}")
        result_count, output_type = parse_prompt(selected_prompt['prompt_text'] if selected_prompt else None)
        
        # Use OR and prefix matching
        ts_query = ' | '.join([f"{kw}:*" for kw in keywords])
        logger.info(f"TS query: {ts_query}, Intent: {intent}, Result count: {result_count}")
        conn = get_db_connection()
        cur = conn.cursor()
        results = []
        try:
            # Keyword-based search
            sql = (
                "SELECT id, title, abstract, ts_rank(tsv, to_tsquery(%s)) AS rank, "
                "authors, journal, publication_date "
                "FROM articles WHERE tsv @@ to_tsquery(%s) "
            )
            params = [ts_query, ts_query]
            if intent.get('recent'):
                sql += "AND publication_date > NOW() - INTERVAL '5 years' "
            sql += f"ORDER BY rank DESC LIMIT {result_count}"
            cur.execute(sql, params)
            results = cur.fetchall()
            logger.info(f"Full-text search results count: {len(results)}")
            if results:
                logger.info(f"Sample title: {results[0][1][:50]}")
            
            # Fallback to LIKE search
            if not results:
                logger.info("Falling back to LIKE search")
                like_conditions = ' OR '.join([f"title ILIKE %s OR abstract ILIKE %s" for _ in keywords])
                like_params = [f"%{kw}%" for kw in keywords for _ in (1, 2)]
                sql = (
                    f"SELECT id, title, abstract, 0 AS rank, authors, journal, publication_date "
                    f"FROM articles WHERE {like_conditions} "
                )
                if intent.get('recent'):
                    sql += "AND publication_date > NOW() - INTERVAL '5 years' "
                sql += f"LIMIT {result_count}"
                cur.execute(sql, like_params)
                results = cur.fetchall()
                logger.info(f"LIKE search results count: {len(results)}")
                if results:
                    logger.info(f"Sample LIKE title: {results[0][1][:50]}")
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            cur.close()
            conn.close()
            return render_template('search.html', error=f"Search failed: {str(e)}", prompts=prompts)
        
        # Convert results to objects and generate embeddings
        high_relevance = [
            {
                'id': r[0],
                'title': r[1],
                'abstract': r[2],
                'score': r[3],
                'authors': r[4],
                'journal': r[5],
                'publication_date': r[6],
                'keywords': None
            } for r in results
        ]
        embeddings = []
        summaries = []
        for r in results:
            summary = generate_summary(
                r[2], query, 
                selected_prompt['prompt_text'] if selected_prompt else None,
                title=r[1], authors=r[4], journal=r[5], publication_date=r[6]
            )
            summaries.append(summary)
            embeddings.append(summary['embedding'])
        
        # Mock LLM ranking
        high_relevance, embeddings = mock_llm_ranking(query, high_relevance, embeddings)
        
        # Generate prompt-driven output
        prompt_output = generate_prompt_output(query, results, selected_prompt['prompt_text'] if selected_prompt else None, output_type)
        
        prompt_text = selected_prompt['prompt_text'] if selected_prompt else None
        cur.close()
        conn.close()
        return render_template('search.html', high_relevance=high_relevance, query=query, summaries=summaries, prompts=prompts, prompt_text=prompt_text, prompt_output=prompt_output)
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