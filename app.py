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
from collections import Counter
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import sendgrid
from sendgrid.helpers.mail import Mail, Email, To, Content

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'your-secret-key')
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load spaCy model
nlp = spacy.load('en_core_web_sm', disable=['ner'])

# Initialize embedding model
embedding_model = None

# Initialize scheduler
scheduler = BackgroundScheduler()
scheduler.start()

# SendGrid client
sg = sendgrid.SendGridAPIClient(os.environ.get('SENDGRID_API_KEY'))

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

def run_notification_rule(rule_id, user_id, rule_name, keywords, timeframe, prompt_text, email_format, user_email):
    logger.info(f"Running notification rule {rule_id} ({rule_name}) for user {user_id}, keywords: {keywords}")
    keywords_list = [k.strip() for k in keywords.split(',')]
    date_filters = {
        'daily': 'last+1+day[dp]',
        'weekly': 'last+7+days[dp]',
        'monthly': 'last+30+days[dp]',
        'annually': 'last+365+days[dp]'
    }
    intent = {'date': date_filters[timeframe]}
    search_query = build_pubmed_query(keywords_list, intent)
    try:
        api_key = os.environ.get('PUBMED_API_KEY')
        esearch_result = esearch(search_query, retmax=20, api_key=api_key)
        pmids = esearch_result['esearchresult']['idlist']
        if not pmids:
            logger.info(f"No new results for rule {rule_id}")
            return
        
        efetch_xml = efetch(pmids, api_key=api_key)
        results = parse_efetch_xml(efetch_xml)
        
        output_type = "summary" if prompt_text and "summary" in prompt_text.lower() else email_format
        is_multi_paragraph = "multi-paragraph" in (prompt_text or "").lower() or "two or three paragraph" in (prompt_text or "").lower()
        is_cumulative = True
        prompt_output = generate_prompt_output(keywords, results, prompt_text, output_type, is_multi_paragraph, is_cumulative)
        
        if email_format == "list":
            content = "\n".join([f"- {r['title']} ({r['publication_date']})\n  {r['abstract'][:100]}..." for r in results])
        elif email_format == "detailed":
            content = "\n".join([f"Title: {r['title']}\nAuthors: {r['authors']}\nJournal: {r['journal']}\nDate: {r['publication_date']}\nAbstract: {r['abstract']}\n" for r in results])
        else:
            content = prompt_output
        
        message = Mail(
            from_email=Email("notifications@pubmedresearcher.com"),
            to_emails=To(user_email),
            subject=f"PubMedResearcher Notification: {rule_name}",
            plain_text_content=content
        )
        response = sg.send(message)
        logger.info(f"Email sent for rule {rule_id}, status: {response.status_code}")
    except Exception as e:
        logger.error(f"Error running notification rule {rule_id}: {str(e)}")

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
    stop_words = nlp.Defaults.stop_words | {'about', 'articles', 'from', 'on', 'this', 'year', 'provide', 'summary'}
    
    keywords = []
    for chunk in doc.noun_chunks:
        phrase = chunk.text.strip()
        if (len(phrase) > 1 and 
            all(token.text not in stop_words and token.is_alpha for token in nlp(phrase))):
            keywords.append(phrase)
    
    for token in doc:
        if (token.is_alpha and 
            token.text not in stop_words and 
            len(token.text) > 1 and 
            not any(token.text in phrase for phrase in keywords)):
            keywords.append(token.text)
    
    query_lower = query.lower()
    if 'cidp' in query_lower or 'chronic inflammatory demyelinating polyneuropathy' in query_lower:
        if 'chronic inflammatory demyelinating polyneuropathy' not in keywords:
            keywords = ['chronic inflammatory demyelinating polyneuropathy'] + [k for k in keywords if k != 'cidp']
    
    keywords = keywords[:3]
    
    intent = {}
    current_year = str(datetime.now().year)
    if 'this year' in query_lower:
        intent['date'] = f"{current_year}[dp]"
    elif year_match := re.search(r'\b(20\d{2})\b', query_lower):
        intent['date'] = f"{year_match.group(1)}[dp]"
    elif 'past month' in query_lower or 'last month' in query_lower:
        intent['date'] = 'last+30+days[dp]'
    elif 'past week' in query_lower or 'last week' in query_lower:
        intent['date'] = 'last+7+days[dp]'
    elif 'recent' in query_lower:
        intent['date'] = 'last+5+years[dp]'
    elif re.search(r'(?:last|past)\s+(\d+)\s+years?', query_lower):
        years = re.search(r'(?:last|past)\s+(\d+)\s+years?', query_lower).group(1)
        intent['date'] = f"last+{years}+years[dp]"
    elif re.search(r'(?:last|past)\s+(\d+)\s+months?', query_lower):
        months = re.search(r'(?:last|past)\s+(\d+)\s+months?', query_lower).group(1)
        intent['date'] = f"last+{int(months)*30}+days[dp]"
    elif re.search(r'(?:last|past)\s+(\d+)\s+weeks?', query_lower):
        weeks = re.search(r'(?:last|past)\s+(\d+)\s+weeks?', query_lower).group(1)
        intent['date'] = f"last+{int(weeks)*7}+days[dp]"
    
    logger.info(f"Extracted keywords: {keywords}, Intent: {intent}")
    if not keywords:
        keywords = [word for word in query_lower.split() 
                    if word not in stop_words and len(word) > 1][:3]
        logger.info(f"Fallback keywords: {keywords}")
    
    return keywords, intent

def build_pubmed_query(keywords, intent):
    query_parts = []
    for kw in keywords:
        if kw == 'chronic inflammatory demyelinating polyneuropathy':
            query_parts.append('(cidp OR chronic+inflammatory+demyelinating+polyneuropathy)')
        else:
            query_parts.append(f"({kw.replace(' ', '+')})")
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
        "sort": "date",
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

def : 1.0
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
        return 20, "summary", False, True
    prompt_text = prompt_text.lower()
    match = re.search(r'return\s+(\d+)\s+results', prompt_text)
    result_count = int(match.group(1)) if match else 20
    is_multi_paragraph = ("multi-paragraph" in prompt_text or 
                         "multiparagraph" in prompt_text or 
                         "two or three paragraph" in prompt_text)
    is_cumulative = not ("each result" in prompt_text or "per result" in prompt_text)
    if "summary" in prompt_text:
        output_type = "summary"
    elif "letter" in prompt_text:
        output_type = "letter"
    elif "answer" in prompt_text or "question" in prompt_text or prompt_text.strip().endswith("?"):
        output_type = "answer"
    else:
        output_type = "summary"
    logger.info(f"Parsed prompt: result_count={result_count}, output_type={output_type}, multi_paragraph={is_multi_paragraph}, cumulative={is_cumulative}")
    return result_count, output_type, is_multi_paragraph, is_cumulative

def mock_llm_ranking(query, results, embeddings):
    query_embedding = generate_embedding(query)
    current_year = datetime.now().year
    scores = []
    for i, (emb, result) in enumerate(zip(embeddings, results)):
        similarity = 1 - cosine(query_embedding, emb) if emb is not None else 0.0
        pub_year = int(result['publication_date']) if result['publication_date'].isdigit() else 2000
        recency_bonus = (pub_year - 2000) / (current_year - 2000)
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

def generate_prompt_output(query, results, prompt_text, output_type, is_multi_paragraph, is_cumulative):
    if not results:
        return f"No results found for '{query}'."
    
    query_lower = query.lower()
    year_match = re.search(r'\b(20\d{2})\b', query_lower)
    target_year = year_match.group(1) if year_match else str(datetime.now().year) if 'this year' in query_lower else None
    if target_year:
        filtered_results = [r for r in results if r['publication_date'] == target_year]
        if not filtered_results:
            return f"No results found for '{query}' in {target_year}. Try broadening the search to recent years."
        results = filtered_results
    
    all_abstracts = " ".join(r['abstract'] or "" for r in results)
    doc = nlp(all_abstracts)
    key_concepts = []
    for chunk in doc.noun_chunks:
        if chunk.text.lower() not in nlp.Defaults.stop_words and len(chunk.text) > 3:
            key_concepts.append(chunk.text)
    concept_counts = Counter(key_concepts).most_common(5)
    key_concepts_str = ", ".join([concept for concept, _ in concept_counts]) if concept_counts else "no key concepts identified"
    logger.info(f"Key concepts extracted: {key_concepts_str}")
    
    if output_type == "summary":
        if is_cumulative:
            combined_text = " ".join([r['abstract'] or r['title'] for r in results])
            if is_multi_paragraph:
                output = f"Multi-Paragraph Summary for '{query}' (Year: {target_year or 'Recent'}):\n\n"
                output += f"Paragraph 1: Research on {query} reveals a focus on {key_concepts_str}. Studies collectively indicate that {combined_text[:300]}... These findings highlight advancements in {query}.\n\n"
                output += f"Paragraph 2: Further insights from the literature emphasize {key_concepts_str.split(', ')[-1] if concept_counts else 'ongoing research'}. The combined evidence suggests {combined_text[300:600]}... This underscores the importance of continued investigation.\n\n"
                output += f"This summary synthesizes {len(results)} PubMed findings, integrating key themes from recent research."
            else:
                output = f"Summary for '{query}' (Year: {target_year or 'Recent'}):\n\n"
                output += f"Research on {query} centers on {key_concepts_str}. The collective findings indicate {combined_text[:300]}... This summary integrates {len(results)} PubMed articles to highlight key advancements."
        else:
            combined_text = "\n".join([f"{r['title']}: {r['abstract'] or 'No abstract'}" for r in results])
            if is_multi_paragraph:
                output = f"Multi-Paragraph Summary for '{query}' (Year: {target_year or 'Recent'}):\n\n"
                for i, result in enumerate(results, 1):
                    title = result['title']
                    abstract = result['abstract'] or "No abstract available."
                    output += f"Paragraph {i}: Research on {query} from \"{title}\" (published {result['publication_date']}) highlights advancements. "
                    output += f"{abstract[:300]}... This study advances our understanding of {query}.\n\n"
                output += f"This summary covers {len(results)} PubMed findings."
            else:
                output = f"Summary for '{query}' (Year: {target_year or 'Recent'}):\n\n{combined_text}\n\nThis summarizes key findings."
    elif output_type == "letter":
        combined_text = "\n".join([f"{r['title']}: {r['abstract'] or 'No abstract'}" for r in results])
        output = f"Dear Researcher,\n\nRegarding '{query}', PubMed data suggests:\n{combined_text}\n\nSincerely,\nPubMed Research Team"
    elif output_type == "answer":
        question = prompt_text if prompt_text and ("question" in prompt_text.lower() or prompt_text.strip().endswith("?")) else query
        combined_text = " ".join([r['abstract'] or r['title'] for r in results])
        relevant_articles = [r['title'] for r in results if any(kc.lower() in r['abstract'].lower() for kc in key_concepts)]
        output = f"Answer to '{question}' (Year: {target_year or 'Recent'}):\n\n"
        output += f"Based on {len(results)} PubMed articles, the response to '{question}' centers on {key_concepts_str}. "
        output += f"The collective findings indicate {combined_text[:300]}... "
        if relevant_articles:
            output += f"Key articles addressing this include: {', '.join(relevant_articles[:2])}."
        else:
            output += "No specific articles directly address this question, but the literature provides relevant context."
        output += f"\n\nThis answer synthesizes the latest PubMed findings."
    
    logger.info(f"Generated prompt output: type={output_type}, multi_paragraph={is_multi_paragraph}, cumulative={is_cumulative}, length={len(output)}")
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
        prompt_text = request.form.get('prompt_text')  # Get edited prompt text
        if not query:
            return render_template('search.html', error="Query cannot be empty", prompts=prompts, prompt_id=prompt_id, prompt_text=prompt_text)
        
        keywords, intent = extract_keywords(query)
        if not keywords:
            return render_template('search.html', error="No valid keywords found", prompts=prompts, prompt_id=prompt_id, prompt_text=prompt_text)
        
        # Use submitted prompt_text if provided, else fall back to selected prompt
        selected_prompt_text = prompt_text if prompt_text else next((p['prompt_text'] for p in prompts if str(p['id']) == prompt_id), None)
        result_count, output_type, is_multi_paragraph, is_cumulative = parse_prompt(selected_prompt_text)
        
        query_lower = query.lower()
        year_match = re.search(r'\b(20\d{2})\b', query_lower)
        target_year = year_match.group(1) if year_match else str(datetime.now().year) if 'this year' in query_lower else None
        
        results = None
        if not results:
            api_key = os.environ.get('PUBMED_API_KEY')
            search_query = build_pubmed_query(keywords, intent)
            try:
                esearch_result = esearch(search_query, retmax=result_count, api_key=api_key)
                pmids = esearch_result['esearchresult']['idlist']
                if not pmids:
                    logger.info("No results with initial query, trying broader query")
                    intent.pop('date', None)
                    search_query = build_pubmed_query(keywords, intent)
                    esearch_result = esearch(search_query, retmax=result_count, api_key=api_key)
                    pmids = esearch_result['esearchresult']['idlist']
                
                if not pmids:
                    return render_template('search.html', error="No results found", prompts=prompts, prompt_id=prompt_id, prompt_text=prompt_text, target_year=target_year)
                
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
                return render_template('search.html', error=f"Search failed: {str(e)}", prompts=prompts, prompt_id=prompt_id, prompt_text=prompt_text, target_year=target_year)
        
        high_relevance = []
        embeddings = []
        summaries = []
        for r in results:
            text = f"{r['title']} {r['abstract'] or ''} {r['authors'] or ''} {r['journal'] or ''}".strip()
            embedding = generate_embedding(text) if text else None
            summary = generate_summary(
                r['abstract'], query, 
                selected_prompt_text,
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
        
        high_relevance, embeddings = mock_llm_ranking(query, high_relevance, embeddings)
        
        prompt_output = generate_prompt_output(
            query, high_relevance, 
            selected_prompt_text, 
            output_type, is_multi_paragraph, is_cumulative
        )
        
        result_summaries = list(zip(high_relevance, summaries))
        
        return render_template(
            'search.html', 
            result_summaries=result_summaries,
            query=query, 
            prompts=prompts, 
            prompt_id=prompt_id,
            prompt_text=prompt_text,
            prompt_output=prompt_output,
            target_year=target_year
        )
    
    return render_template('search.html', prompts=prompts, prompt_id='', prompt_text='')

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
    prompts = [{'id': p[0], 'prompt_name': p[1], 'prompt_text': p[2], 'created_at': p[3]} for p in cur.fetchall()]
    cur.close()
    conn.close()
    return render_template('prompt.html', prompts=prompts)

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
    return render_template('prompt_edit.html', prompt={'id': prompt[0], 'prompt_name': prompt[1], 'prompt_text': prompt[2]})

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
    return render_template('notifications.html', notifications=notifications)

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
    })

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

# Schedule notifications on startup
schedule_notification_rules()

if __name__ == '__main__':
    app.run(debug=True)