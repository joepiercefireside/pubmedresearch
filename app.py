from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import psycopg2
from werkzeug.security import generate_password_hash, check_password_hash
import spacy
import re
import os
import logging
import json
import requests
from xml.etree import ElementTree
from ratelimit import limits, sleep_and_retry
from sentence_transformers import SentenceTransformer
import numpy as np
from scipy.spatial.distance import cosine
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'your-secret-key')
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load spaCy model with parser enabled
nlp = spacy.load('en_core_web_sm', disable=['ner'])

# Initialize embedding model
embedding_model = None

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
    stop_words = nlp.Defaults.stop_words | {'about', 'articles', 'from', 'on'}
    
    # Extract noun phrases
    keywords = []
    for chunk in doc.noun_chunks:
        phrase = chunk.text.strip()
        if (len(phrase) > 1 and 
            all(token.text not in stop_words and token.is_alpha for token in nlp(phrase))):
            keywords.append(phrase)
    
    # Add single keywords not in phrases
    for token in doc:
        if (token.is_alpha and 
            token.text not in stop_words and 
            len(token.text) > 1 and 
            not any(token.text in phrase for phrase in keywords)):
            keywords.append(token.text)
    
    keywords = keywords[:3]
    
    # Detect date intent
    intent = {}
    query_lower = query.lower()
    current_year = str(datetime.now().year)  # Get current year (2025)
    if 'this year' in query_lower:
        intent['date'] = f"{current_year}[dp]"
    elif year_match := re.search(r'\b(20\d{2})\b', query_lower):
        intent['date'] = f"{year_match.group(1)}[dp]"
    elif 'recent' in query_lower:
        intent['date'] = 'last+5+years[dp]'
    elif re.search(r'last\s+(\d+)\s+years?', query_lower):
        years = re.search(r'last\s+(\d+)\s+years?', query_lower).group(1)
        intent['date'] = f"last+{years}+years[dp]"
    elif re.search(r'past\s+(\d+)\s+years?', query_lower):
        years = re.search(r'past\s+(\d+)\s+years?', query_lower).group(1)
        intent['date'] = f"last+{years}+years[dp]"
    
    logger.info(f"Extracted keywords: {keywords}, Intent: {intent}")
    if not keywords:
        keywords = [word for word in query_lower.split() 
                    if word not in stop_words and len(word) > 1][:3]
        logger.info(f"Fallback keywords: {keywords}")
    
    return keywords, intent

def build_pubmed_query(keywords, intent):
    query_parts = [f"({kw.replace(' ', '+')})" for kw in keywords]
    query = " AND ".join(query_parts)
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
        "sort": "date",  # Sort by date (most recent first)
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
        return 20, "summary", False
    prompt_text = prompt_text.lower()
    match = re.search(r'return\s+(\d+)\s+results', prompt_text)
    result_count = int(match.group(1)) if match else 20
    is_multi_paragraph = "multi-paragraph" in prompt_text or "multiparagraph" in prompt_text
    if "summary article" in prompt_text or "summary" in prompt_text:
        output_type = "summary"
    elif "letter" in prompt_text:
        output_type = "letter"
    elif "answer" in prompt_text or "question" in prompt_text:
        output_type = "answer"
    else:
        output_type = "summary"
    logger.info(f"Parsed prompt: result_count={result_count}, output_type={output_type}, multi_paragraph={is_multi_paragraph}")
    return result_count, output_type, is_multi_paragraph

def mock_llm_ranking(query, results, embeddings):
    query_embedding = generate_embedding(query)
    current_year = datetime.now().year
    scores = []
    for i, (emb, result) in enumerate(zip(embeddings, results)):
        similarity = 1 - cosine(query_embedding, emb) if emb is not None else 0.0
        # Weight recency: add bonus for newer articles
        pub_year = int(result['publication_date']) if result['publication_date'].isdigit() else 2000
        recency_bonus = (pub_year - 2000) / (current_year - 2000)  # Normalize to 0-1
        weighted_score = 0.7 * similarity + 0.3 * recency_bonus
        scores.append((i, weighted_score))
    scores.sort(key=lambda x: x[1], reverse=True)
    ranked_indices = [i for i, _ in scores]
    ranked_results = [results[i] for i in ranked_indices]
    ranked_embeddings = [embeddings[i] for i in ranked_indices]
    return ranked_results, ranked_embeddings

def generate_summary(abstract, query, prompt_text=None, title=None, authors=None, journal=None, publication_date=None):
    if not abstract and not title:
        return {"text": "No content available to summarize.", "metadata": {}, "embedding": None}
    text = f"{title} {abstract or ''} {authors or ''} {journal or ''}".strip()
    embedding = generate_embedding(text) if text else None
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

def generate_prompt_output(query, results, prompt_text, output_type, is_multi_paragraph):
    if not results:
        return f"No results found for '{query}'."
    
    # Filter results by year if specified in query
    year_match = re.search(r'\b(20\d{2})\b', query.lower()) or ('this year' in query.lower() and str(datetime.now().year))
    target_year = year_match.group(1) if isinstance(year_match, re.Match) else str(datetime.now().year)
    if target_year:
        filtered_results = [r for r in results if r['publication_date'] == target_year]
        if not filtered_results:
            return f"No results found for '{query}' in {target_year}. Try broadening the search."
        results = filtered_results
    
    # Combine results for context
    combined_text = "\n".join([f"{r['title']}: {r['abstract'] or 'No abstract'}" for r in results])
    
    # Generate output based on type
    if output_type == "summary":
        if is_multi_paragraph:
            output = f"Multi-Paragraph Summary for '{query}' (Year: {target_year or 'Recent'}):\n\n"
            for i, result in enumerate(results, 1):
                title = result['title']
                abstract = result['abstract'] or "No abstract available."
                output += f"Paragraph {i}: Research on {query} from \"{title}\" (published {result['publication_date']}) highlights advancements. "
                output += f"{abstract[:300]}... This study advances our understanding of {query}.\n\n"
            output += f"This summary synthesizes {len(results)} PubMed findings, focusing on recent developments."
        else:
            output = f"Summary for '{query}' (Year: {target_year or 'Recent'}):\n\n{combined_text}\n\nThis summarizes key findings."
    elif output_type == "letter":
        output = f"Dear Researcher,\n\nRegarding '{query}', PubMed data suggests:\n{combined_text}\n\nSincerely,\nPubMed Research Team"
    elif output_type == "answer":
        question = prompt_text if prompt_text and ("question" in prompt_text.lower() or prompt_text.strip().endswith("?")) else query
        output = f"Answer to '{question}':\n\nBased on recent PubMed data:\n{combined_text}\n\nThis addresses the latest findings."
    else:
        output = f"Summary for '{query}' (Year: {target_year or 'Recent'}):\n\n{combined_text}"
    
    logger.info(f"Generated prompt output: type={output_type}, multi_paragraph={is_multi_paragraph}, length={len(output)}")
    return output

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
        if not query:
            return render_template('search.html', error="Query cannot be empty", prompts=prompts)
        
        keywords, intent = extract_keywords(query)
        if not keywords:
            return render_template('search.html', error="No valid keywords found", prompts=prompts)
        
        selected_prompt = next((p for p in prompts if str(p['id']) == prompt_id), None)
        result_count, output_type, is_multi_paragraph = parse_prompt(selected_prompt['prompt_text'] if selected_prompt else None)
        
        # Bypass cache for testing
        results = None
        # Check cache (commented out for testing)
        # conn = get_db_connection()
        # cur = conn.cursor()
        # cur.execute("SELECT results FROM search_cache WHERE query = %s AND created_at > NOW() - INTERVAL '1 day'", (query,))
        # cached = cur.fetchone()
        # if cached:
        #     logger.info("Using cached results")
        #     results = json.loads(cached[0])
        #     cur.close()
        #     conn.close()
        
        if not results:
            # PubMed API search
            api_key = os.environ.get('PUBMED_API_KEY')
            search_query = build_pubmed_query(keywords, intent)
            try:
                esearch_result = esearch(search_query, retmax=result_count, api_key=api_key)
                pmids = esearch_result['esearchresult']['idlist']
                # Fallback if no results
                if not pmids:
                    logger.info("No results with initial query, trying broader query")
                    search_query = build_pubmed_query(keywords, {})
                    esearch_result = esearch(search_query, retmax=result_count, api_key=api_key)
                    pmids = esearch_result['esearchresult']['idlist']
                
                if not pmids:
                    return render_template('search.html', error="No results found", prompts=prompts)
                
                efetch_xml = efetch(pmids, api_key=api_key)
                results = parse_efetch_xml(efetch_xml)
                
                # Cache results
                conn = get_db_connection()
                cur = conn.cursor()
                cur.execute("INSERT INTO search_cache (query, results) VALUES (%s, %s)", 
                            (query, json.dumps(results)))
                conn.commit()
                cur.close()
                conn.close()
            except Exception as e:
                logger.error(f"PubMed API error: {str(e)}")
                return render_template('search.html', error=f"Search failed: {str(e)}", prompts=prompts)
        
        # Generate embeddings and summaries
        high_relevance = []
        embeddings = []
        summaries = []
        for r in results:
            text = f"{r['title']} {r['abstract'] or ''} {r['authors'] or ''} {r['journal'] or ''}".strip()
            embedding = generate_embedding(text) if text else None
            summary = generate_summary(
                r['abstract'], query, 
                selected_prompt['prompt_text'] if selected_prompt else None,
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
        
        # Rank results
        high_relevance, embeddings = mock_llm_ranking(query, high_relevance, embeddings)
        
        # Generate prompt output
        prompt_output = generate_prompt_output(
            query, high_relevance, 
            selected_prompt['prompt_text'] if selected_prompt else None, 
            output_type, is_multi_paragraph
        )
        
        # Zip results and summaries
        result_summaries = list(zip(high_relevance, summaries))
        
        return render_template(
            'search.html', 
            result_summaries=result_summaries,
            query=query, 
            prompts=prompts, 
            prompt_text=selected_prompt['prompt_text'] if selected_prompt else None, 
            prompt_output=prompt_output
        )
    
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
    prompts = [{'id': p[0], 'prompt_name': p[1], 'prompt_text': p[2], 'created_at': p[3]} for p in cur.fetchall()]
    cur.close()
    conn.close()
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