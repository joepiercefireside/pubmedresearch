from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, Response, session
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import psycopg2
from psycopg2.extras import execute_values
from werkzeug.security import generate_password_hash, check_password_hash
import os
import logging
import asyncio
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from scipy.spatial.distance import cosine
import urllib.parse
import sqlite3
import json
import psutil
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
import traceback
from openai import OpenAI
import tenacity
import time
import pdfplumber
import aiohttp
import re
import io
from heapq import heappush, heappop
import threading
import concurrent.futures

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'your-secret-key')
app.config['PERMANENT_SESSION_LIFETIME'] = 86400  # 24 hours
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
logger = logging.getLogger(__name__)

# Initialize embedding model and tokenizer lazily
tokenizer = None
model = None

# Initialize SQLite database for progress tracking
def init_progress_db():
    conn = sqlite3.connect('progress.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS progress
                 (user_id TEXT, url TEXT, library_id INTEGER, links_found INTEGER, links_scanned INTEGER, items_crawled INTEGER, status TEXT, current_url TEXT)''')
    conn.commit()
    conn.close()

init_progress_db()

@tenacity.retry(stop=tenacity.stop_after_attempt(3), wait=tenacity.wait_exponential(multiplier=1, min=4, max=10))
async def load_embedding_model():
    global tokenizer, model
    if tokenizer is None or model is None:
        logger.info("Loading all-MiniLM-L6-v2 model...")
        try:
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2', cache_dir='/tmp/hf_cache')
            model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2', cache_dir='/tmp/hf_cache')
            logger.info("all-MiniLM-L6-v2 model loaded.")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}\n{traceback.format_exc()}")
            try:
                os.environ["HF_HUB_DISABLE_XET"] = "1"
                tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2', cache_dir='/tmp/hf_cache')
                model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2', cache_dir='/tmp/hf_cache')
                logger.info("Fallback: all-MiniLM-L6-v2 model loaded without hf_xet.")
            except Exception as e2:
                logger.error(f"Fallback failed: {str(e2)}\n{traceback.format_exc()}")
                raise
    return tokenizer, model

async def unload_embedding_model():
    global tokenizer, model
    tokenizer = None
    model = None
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    logger.info("Unloaded all-MiniLM-L6-v2 model to free memory.")

def get_db_connection():
    try:
        conn = psycopg2.connect(os.environ['DATABASE_URL'])
        logger.info("Database connection established.")
        return conn
    except KeyError as e:
        logger.error(f"Missing DATABASE_URL: {e}")
        raise
    except psycopg2.Error as e:
        logger.error(f"Database connection failed: {e}")
        raise

class User(UserMixin):
    def __init__(self, id, email):
        self.id = id
        self.email = email

@login_manager.user_loader
def load_user(user_id):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT id, email FROM users WHERE id = %s", (int(user_id),))
        user = cur.fetchone()
        cur.close()
        conn.close()
        if user:
            logger.info(f"User loaded: ID {user_id}")
            return User(user[0], user[1])
        logger.warning(f"No user found for ID {user_id}")
        return None
    except Exception as e:
        logger.error(f"Error loading user {user_id}: {e}")
        return None

def clean_content(html):
    try:
        soup = BeautifulSoup(html, 'html.parser')
        for element in soup(['script', 'style', 'header', 'footer', 'nav', 'img', 'video', 'audio']):
            element.decompose()
        text = soup.get_text(separator=' ', strip=True)
        return ' '.join(text.split())[:1000]
    except Exception as e:
        logger.error(f"Error cleaning content: {e}\n{traceback.format_exc()}")
        return ""

async def extract_pdf_text(url):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200 or 'application/pdf' not in response.headers.get('Content-Type', ''):
                    return None
                pdf_data = await response.read()
                with pdfplumber.open(io.BytesIO(pdf_data)) as pdf:
                    text = ''
                    for page in pdf.pages:
                        text += page.extract_text() or ''
                    return ' '.join(text.split())[:1000] if text else None
    except Exception as e:
        logger.error(f"Error extracting PDF text from {url}: {e}\n{traceback.format_exc()}")
        return None

async def analyze_page_for_links(page):
    try:
        api_key = os.environ.get('XAI_API_KEY')
        if not api_key:
            logger.error("XAI_API_KEY not set")
            return []
        
        html = await page.content()
        prompt = """
        Analyze the provided HTML to identify interactive elements (e.g., buttons, links, dynamic lists, endless scroll triggers, images with onclick events) that could lead to pages with textual content when clicked or scrolled. Prioritize buttons with labels like 'Browse', 'Learn More', 'Resources', or links within dynamic lists. Suggest specific actions (e.g., click selectors, scroll) to uncover more links, including multi-step paths (e.g., click a button to load a page, then click image links). Return a JSON list of actions, each with 'type' ('click' or 'scroll'), 'selector' (CSS selector for click or empty for scroll), and 'priority' (1 for high, 2 for medium, 3 for low). Ensure the response is valid JSON.
        Example response: [
            {"type": "click", "selector": "button.browse", "priority": 1},
            {"type": "scroll", "selector": "", "priority": 2}
        ]
        """
        client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
        completion = client.chat.completions.create(
            model="grok-3-latest",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"HTML: {html[:4000]}"}
            ],
            max_tokens=1000
        )
        raw_response = completion.choices[0].message.content
        logger.debug(f"Grok API raw response: {raw_response}")
        try:
            actions = json.loads(raw_response)
            if not isinstance(actions, list):
                logger.error("Grok API response is not a list")
                return []
            return sorted(actions, key=lambda x: x.get('priority', 3))
        except json.JSONDecodeError:
            logger.error(f"Failed to parse Grok API response as JSON: {raw_response}")
            return []
    except Exception as e:
        logger.error(f"Error analyzing page for links: {str(e)}\n{traceback.format_exc()}")
        return []

async def generate_embedding(text, tokenizer, model):
    try:
        process = psutil.Process()
        logger.info(f"Memory usage before embedding: {process.memory_info().rss / 1024 / 1024:.2f} MB")
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        logger.info(f"Memory usage after embedding: {process.memory_info().rss / 1024 / 1024:.2f} MB")
        return embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}\n{traceback.format_exc()}")
        return None

@tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_exponential(multiplier=1, min=2, max=60),
    retry=tenacity.retry_if_exception_type(Exception),
    before_sleep=lambda retry_state: logger.info(f"Retrying Grok API call, attempt {retry_state.attempt_number}")
)
def query_grok_api(query, context, prompt="Process the provided context according to the user's prompt."):
    try:
        api_key = os.environ.get('XAI_API_KEY')
        if not api_key:
            logger.error("XAI_API_KEY not set in environment variables")
            return "Error: xAI API key not configured"

        # Truncate context if too long
        if len(context) > 10000:
            context = context[:10000] + "... [truncated]"
            logger.warning(f"Context truncated to 10,000 characters for query: {query[:50]}...")

        client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
        logger.info(f"Sending Grok API request: prompt={query[:50]}..., context_length={len(context)}")
        
        completion = client.chat.completions.create(
            model="grok-3-latest",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Based on the following context, answer the question: {query}\n\nContext: {context}"}
            ],
            max_tokens=500,
            timeout=30  # Add timeout to prevent hanging
        )
        
        response_content = completion.choices[0].message.content
        logger.info(f"Grok API response received: length={len(response_content)}")
        return response_content
    
    except Exception as e:
        logger.error(f"Grok API call failed: {str(e)}")
        raise  # Re-raise to trigger retry

def normalize_url(url):
    """Normalize URL by adding https:// if missing and ensuring proper format."""
    if not url:
        return None
    url = url.strip()
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    try:
        parsed = urllib.parse.urlparse(url)
        if not parsed.netloc:
            return None
        # Ensure www. prefix for consistency
        if not parsed.netloc.startswith('www.'):
            parsed = parsed._replace(netloc='www.' + parsed.netloc)
        return parsed.geturl()
    except ValueError:
        return None

async def crawl_website(start_url, user_id, library_id):
    logger.info(f"Starting crawl for {start_url} by user {user_id} in library {library_id}")
    process = psutil.Process()
    logger.info(f"Memory usage before crawl: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    
    visited_urls = set()
    to_visit = []
    def add_to_visit(url, priority):
        heappush(to_visit, (priority, url))
    
    education_keywords = ['/education', '/learning-center', '/resources']
    def get_url_priority(url):
        for keyword in education_keywords:
            if keyword in url.lower():
                return 0
        return 1
    
    add_to_visit(start_url, get_url_priority(start_url))
    base_domain = urllib.parse.urlparse(start_url).netloc
    links_found = 1
    links_scanned = 0
    items_crawled = 0
    max_items = 50
    crawled_data = []
    
    conn = sqlite3.connect('progress.db')
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO progress (user_id, url, library_id, links_found, links_scanned, items_crawled, status, current_url) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
              (str(user_id), start_url, library_id, links_found, links_scanned, items_crawled, "running", start_url))
    conn.commit()
    
    try:
        embedding_tokenizer, embedding_model = await load_embedding_model()
    except Exception as e:
        logger.error(f"Failed to load embedding model: {str(e)}")
        c.execute("UPDATE progress SET status = ?, current_url = ? WHERE user_id = ? AND url = ? AND library_id = ?",
                  (f"error: {str(e)}", "", str(user_id), start_url, library_id))
        conn.commit()
        conn.close()
        return crawled_data
    
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True, args=['--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage'])
            try:
                while to_visit and items_crawled < max_items:
                    try:
                        priority, url = heappop(to_visit)
                        if url in visited_urls or not url.startswith(('http://', 'https://')):
                            logger.debug(f"Skipping invalid or visited URL: {url}")
                            continue
                        if re.search(r'\.(jpg|jpeg|png|gif|bmp|svg|ico|mp4|avi|mov|wmv|flv|webm|mp3|wav|ogg|css|js|woff|woff2|ttf|eot)$', url, re.IGNORECASE):
                            logger.debug(f"Skipping non-textual URL: {url}")
                            continue
                        visited_urls.add(url)
                        links_scanned += 1
                        logger.debug(f"Scanning URL: {url} (priority: {priority})")
                        
                        c.execute("UPDATE progress SET current_url = ? WHERE user_id = ? AND url = ? AND library_id = ?",
                                  (url, str(user_id), start_url, library_id))
                        conn.commit()
                        
                        page = await browser.new_page()
                        response = await page.goto(url, timeout=30000, wait_until='networkidle')
                        if response is None or response.status >= 400:
                            logger.warning(f"Failed to load {url}: Status {response.status if response else 'None'}")
                            await page.close()
                            continue
                        
                        for _ in range(3):
                            await page.evaluate('window.scrollTo(0, document.body.scrollHeight);')
                            await page.wait_for_timeout(2000)
                        
                        actions = await analyze_page_for_links(page)
                        for action in actions[:5]:
                            try:
                                if action['type'] == 'click' and action['selector']:
                                    elements = await page.query_selector_all(action['selector'])
                                    for element in elements[:2]:
                                        try:
                                            await element.click(timeout=5000)
                                            await page.wait_for_timeout(3000)
                                            new_links = await page.evaluate('''() => {
                                                return Array.from(document.querySelectorAll('a[href]')).map(a => a.href);
                                            }''')
                                            for link in new_links:
                                                absolute_url = urllib.parse.urljoin(url, link)
                                                parsed_url = urllib.parse.urlparse(absolute_url)
                                                if parsed_url.netloc == base_domain and absolute_url not in visited_urls and absolute_url not in [u[1] for u in to_visit] and parsed_url.scheme in ('http', 'https'):
                                                    add_to_visit(absolute_url, get_url_priority(absolute_url))
                                                    links_found += 1
                                                    logger.debug(f"New link found via click: {absolute_url}, links_found={links_found}")
                                        except Exception as e:
                                            logger.debug(f"Error clicking element {action['selector']}: {e}")
                                elif action['type'] == 'scroll':
                                    await page.evaluate('window.scrollTo(0, document.body.scrollHeight);')
                                    await page.wait_for_timeout(2000)
                            except Exception as e:
                                logger.debug(f"Error performing action {action}: {e}")
                        
                        content_type = response.headers.get('content-type', '').lower()
                        cleaned_content = None
                        if 'text/html' in content_type:
                            content = await page.content()
                            cleaned_content = clean_content(content)
                            if not cleaned_content or len(cleaned_content.strip()) < 100:
                                logger.warning(f"No valid textual content found for {url}")
                                await page.close()
                                continue
                        elif 'application/pdf' in content_type:
                            cleaned_content = await extract_pdf_text(url)
                            if not cleaned_content or len(cleaned_content.strip()) < 100:
                                logger.warning(f"No valid textual content in PDF at {url}")
                                await page.close()
                                continue
                        else:
                            logger.debug(f"Skipping unsupported content type {content_type} for {url}")
                            await page.close()
                            continue
                        
                        embedding = await generate_embedding(cleaned_content, embedding_tokenizer, embedding_model)
                        if embedding is None:
                            logger.warning(f"Failed to generate embedding for {url}")
                            await page.close()
                            continue
                        
                        items_crawled += 1
                        crawled_data.append((url, cleaned_content, embedding.tolist(), None, library_id))
                        logger.debug(f"Content found for {url}, items_crawled={items_crawled}")
                        
                        links = await page.evaluate('''() => {
                            const urls = [];
                            document.querySelectorAll('a[href], button, [role="link"], [onclick], [data-href], [data-nav], [data-url], [data-link], meta[content][http-equiv="refresh"], link[rel="sitemap"]').forEach(el => {
                                let url = el.href || el.getAttribute('data-href') || el.getAttribute('data-nav') || el.getAttribute('data-url') || el.getAttribute('data-link');
                                if (!url && el.getAttribute('onclick')) {
                                    const match = el.getAttribute('onclick').match(/(?:location\.href|window\.open|navigateTo|window\.location\.assign|window\.location\.replace)\(['"]([^'"]+)['"]/);
                                    if (match) url = match[1];
                                }
                                if (!url && el.tagName === 'META' && el.getAttribute('http-equiv') === 'refresh') {
                                    const content = el.getAttribute('content');
                                    const match = content.match(/url=(.+)$/i);
                                    if (match) url = match[1];
                                }
                                if (!url && el.tagName === 'LINK' && el.getAttribute('rel') === 'sitemap') {
                                    url = el.href;
                                }
                                if (url) urls.push(url);
                            });
                            const scripts = document.querySelectorAll('script');
                            scripts.forEach(script => {
                                const text = script.textContent || script.src;
                                const matches = text.match(/(?:location\.href|window\.location|navigateTo|open)\(['"]([^'"]+)['"]/g);
                                if (matches) {
                                    matches.forEach(match => {
                                        const urlMatch = match.match(/['"]([^'"]+)['"]/);
                                        if (urlMatch) urls.push(urlMatch[1]);
                                    });
                                }
                            });
                            return [...new Set(urls)];
                        }''')
                        for link in links:
                            absolute_url = urllib.parse.urljoin(url, link)
                            parsed_url = urllib.parse.urlparse(absolute_url)
                            if parsed_url.netloc == base_domain and absolute_url not in visited_urls and absolute_url not in [u[1] for u in to_visit] and parsed_url.scheme in ('http', 'https'):
                                add_to_visit(absolute_url, get_url_priority(absolute_url))
                                links_found += 1
                                logger.debug(f"New link found: {absolute_url}, links_found={links_found}")
                        
                        c.execute("UPDATE progress SET links_found = ?, links_scanned = ?, items_crawled = ?, status = ? WHERE user_id = ? AND url = ? AND library_id = ?",
                                  (links_found, links_scanned, items_crawled, "running", str(user_id), start_url, library_id))
                        conn.commit()
                        
                        await page.close()
                        logger.info(f"Memory usage after scanning {url}: {process.memory_info().rss / 1024 / 1024:.2f} MB")
                    except Exception as e:
                        logger.error(f"Error crawling {url}: {str(e)}\n{traceback.format_exc()}")
                        c.execute("UPDATE progress SET status = ? WHERE user_id = ? AND url = ? AND library_id = ?",
                                  (f"error: {str(e)}", str(user_id), start_url, library_id))
                        conn.commit()
                        continue
                logger.info(f"Crawl complete: items_crawled={items_crawled}")
                c.execute("UPDATE progress SET status = ?, current_url = ? WHERE user_id = ? AND url = ? AND library_id = ?",
                          ("complete", "", str(user_id), start_url, library_id))
                conn.commit()
            finally:
                await browser.close()
                conn.close()
        if not crawled_data and links_found == links_scanned:
            flash("No new textual content found; all URLs already exist or lack meaningful text.", 'info')
        return crawled_data
    except Exception as e:
        logger.error(f"Unexpected error in crawl_website: {str(e)}\n{traceback.format_exc()}")
        c.execute("UPDATE progress SET status = ?, current_url = ? WHERE user_id = ? AND url = ? AND library_id = ?",
                  (f"error: {str(e)}", "", str(user_id), start_url, library_id))
        conn.commit()
        conn.close()
        return crawled_data
    finally:
        await unload_embedding_model()

@app.route('/')
def index():
    try:
        logger.info("Accessing index page")
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering index page: {e}\n{traceback.format_exc()}")
        return "Internal Server Error", 500

@app.route('/register', methods=['GET', 'POST'])
def register():
    try:
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
    except Exception as e:
        logger.error(f"Error in register endpoint: {e}\n{traceback.format_exc()}")
        return "Internal Server Error", 500

@app.route('/login', methods=['GET', 'POST'])
def login():
    try:
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
                login_user(User(user[0], user[1]), remember=True)
                session.permanent = True
                logger.info(f"User {email} logged in successfully")
                return redirect(url_for('index'))
            flash('Invalid email or password.', 'error')
        return render_template('login.html')
    except Exception as e:
        logger.error(f"Error in login endpoint: {e}\n{traceback.format_exc()}")
        return "Internal Server Error", 500

@app.route('/logout')
@login_required
def logout():
    try:
        logout_user()
        logger.info("User logged out")
        return redirect(url_for('login'))
    except Exception as e:
        logger.error(f"Error in logout endpoint: {e}\n{traceback.format_exc()}")
        return "Internal Server Error", 500

@app.route('/libraries', methods=['GET', 'POST'])
@login_required
def libraries():
    try:
        if not current_user.is_authenticated:
            logger.error("Unauthenticated user accessing libraries endpoint")
            flash('Please log in to access libraries.', 'error')
            return redirect(url_for('login'))
        
        conn = get_db_connection()
        cur = conn.cursor()
        if request.method == 'POST':
            name = request.form.get('name')
            if not name:
                flash('Library name cannot be empty.', 'error')
            else:
                cur.execute("INSERT INTO libraries (user_id, name) VALUES (%s, %s)", (int(current_user.id), name))
                conn.commit()
                flash('Library created successfully.', 'success')
                return redirect(url_for('libraries'))
        
        cur.execute("SELECT id, name FROM libraries WHERE user_id = %s", (int(current_user.id),))
        libraries = cur.fetchall()
        cur.close()
        conn.close()
        return render_template('libraries.html', libraries=libraries)
    except Exception as e:
        logger.error(f"Error in libraries endpoint: {e}\n{traceback.format_exc()}")
        flash(f"Error: {str(e)}", 'error')
        return render_template('libraries.html', libraries=[])

@app.route('/libraries/view/<int:library_id>')
@login_required
def view_library(library_id):
    try:
        if not current_user.is_authenticated:
            logger.error("Unauthenticated user accessing view_library endpoint")
            flash('Please log in to view libraries.', 'error')
            return redirect(url_for('login'))
        
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT id, name FROM libraries WHERE id = %s AND user_id = %s", (library_id, int(current_user.id)))
        library = cur.fetchone()
        if not library:
            flash('Library not found.', 'error')
            return redirect(url_for('libraries'))
        
        cur.execute("SELECT id, url, content FROM documents WHERE library_id = %s", (library_id,))
        contents = cur.fetchall()
        cur.close()
        conn.close()
        return render_template('library_view.html', library=library, contents=contents)
    except Exception as e:
        logger.error(f"Error in view_library endpoint: {e}\n{traceback.format_exc()}")
        flash(f"Error: {str(e)}", 'error')
        return redirect(url_for('libraries'))

@app.route('/libraries/delete_content/<int:content_id>')
@login_required
def delete_library_content(content_id):
    try:
        if not current_user.is_authenticated:
            logger.error("Unauthenticated user accessing delete_library_content endpoint")
            flash('Please log in to delete content.', 'error')
            return redirect(url_for('login'))
        
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT library_id FROM documents WHERE id = %s", (content_id,))
        library_id = cur.fetchone()
        if not library_id:
            flash('Content not found.', 'error')
            return redirect(url_for('libraries'))
        
        cur.execute("DELETE FROM documents WHERE id = %s AND library_id IN (SELECT id FROM libraries WHERE user_id = %s)", (content_id, int(current_user.id)))
        conn.commit()
        cur.close()
        conn.close()
        flash('Content deleted successfully.', 'success')
        return redirect(url_for('view_library', library_id=library_id[0]))
    except Exception as e:
        logger.error(f"Error in delete_library_content endpoint: {e}\n{traceback.format_exc()}")
        flash(f"Error: {str(e)}", 'error')
        return redirect(url_for('libraries'))

@app.route('/libraries/delete/<int:library_id>')
@login_required
def delete_library(library_id):
    try:
        if not current_user.is_authenticated:
            logger.error("Unauthenticated user accessing delete_library endpoint")
            flash('Please log in to delete libraries.', 'error')
            return redirect(url_for('login'))
        
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT id FROM libraries WHERE id = %s AND user_id = %s", (library_id, int(current_user.id)))
        if not cur.fetchone():
            flash('Library not found.', 'error')
            return redirect(url_for('libraries'))
        
        cur.execute("DELETE FROM documents WHERE library_id = %s", (library_id,))
        cur.execute("DELETE FROM libraries WHERE id = %s AND user_id = %s", (library_id, int(current_user.id)))
        conn.commit()
        cur.close()
        conn.close()
        flash('Library and its contents deleted successfully.', 'success')
        return redirect(url_for('libraries'))
    except Exception as e:
        logger.error(f"Error in delete_library endpoint: {e}\n{traceback.format_exc()}")
        flash(f"Error: {str(e)}", 'error')
        return redirect(url_for('libraries'))

@app.route('/add_library', methods=['POST'])
@login_required
def add_library():
    try:
        if not current_user.is_authenticated:
            logger.error("Unauthenticated user accessing add_library endpoint")
            return jsonify({'status': 'error', 'message': 'Please log in to add a library'}), 401
        
        name = request.form.get('name')
        if not name:
            flash('Library name cannot be empty.', 'error')
            return jsonify({'status': 'error', 'message': 'Library name cannot be empty'}), 400
        
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("INSERT INTO libraries (user_id, name) VALUES (%s, %s) RETURNING id", (int(current_user.id), name))
        library_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()
        return jsonify({'status': 'success', 'library_id': library_id, 'library_name': name})
    except Exception as e:
        logger.error(f"Error in add_library endpoint: {e}\n{traceback.format_exc()}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/prompts', methods=['GET', 'POST'])
@login_required
def prompts():
    try:
        if not current_user.is_authenticated:
            logger.error("Unauthenticated user accessing prompts endpoint")
            flash('Please log in to access prompts.', 'error')
            return redirect(url_for('login'))
        
        conn = get_db_connection()
        cur = conn.cursor()
        if request.method == 'POST':
            name = request.form.get('name')
            content = request.form.get('content')
            if not name or not content:
                flash('Prompt name and content cannot be empty.', 'error')
            else:
                cur.execute("INSERT INTO prompts (user_id, name, content) VALUES (%s, %s, %s)", (int(current_user.id), name, content))
                conn.commit()
                flash('Prompt created successfully.', 'success')
                return redirect(url_for('prompts'))
        
        cur.execute("SELECT id, name, content FROM prompts WHERE user_id = %s", (int(current_user.id),))
        prompts = cur.fetchall()
        cur.close()
        conn.close()
        return render_template('prompts.html', prompts=prompts)
    except Exception as e:
        logger.error(f"Error in prompts endpoint: {e}\n{traceback.format_exc()}")
        flash(f"Error: {str(e)}", 'error')
        return render_template('prompts.html', prompts=[])

@app.route('/prompts/edit/<int:prompt_id>', methods=['GET', 'POST'])
@login_required
def edit_prompt(prompt_id):
    try:
        if not current_user.is_authenticated:
            logger.error("Unauthenticated user accessing edit_prompt endpoint")
            flash('Please log in to edit prompts.', 'error')
            return redirect(url_for('login'))
        
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT id, name, content FROM prompts WHERE id = %s AND user_id = %s", (prompt_id, int(current_user.id)))
        prompt = cur.fetchone()
        if not prompt:
            flash('Prompt not found.', 'error')
            return redirect(url_for('prompts'))
        
        if request.method == 'POST':
            name = request.form.get('name')
            content = request.form.get('content')
            if not name or not content:
                flash('Prompt name and content cannot be empty.', 'error')
            else:
                cur.execute("UPDATE prompts SET name = %s, content = %s WHERE id = %s AND user_id = %s",
                            (name, content, prompt_id, int(current_user.id)))
                conn.commit()
                flash('Prompt updated successfully.', 'success')
                return redirect(url_for('prompts'))
        
        cur.close()
        conn.close()
        return render_template('prompt_edit.html', prompt=prompt)
    except Exception as e:
        logger.error(f"Error in edit_prompt endpoint: {e}\n{traceback.format_exc()}")
        flash(f"Error: {str(e)}", 'error')
        return redirect(url_for('prompts'))

@app.route('/prompts/delete/<int:prompt_id>')
@login_required
def delete_prompt(prompt_id):
    try:
        if not current_user.is_authenticated:
            logger.error("Unauthenticated user accessing delete_prompt endpoint")
            flash('Please log in to delete prompts.', 'error')
            return redirect(url_for('login'))
        
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("DELETE FROM prompts WHERE id = %s AND user_id = %s", (prompt_id, int(current_user.id)))
        conn.commit()
        cur.close()
        conn.close()
        flash('Prompt deleted successfully.', 'success')
        return redirect(url_for('prompts'))
    except Exception as e:
        logger.error(f"Error in delete_prompt endpoint: {e}\n{traceback.format_exc()}")
        flash(f"Error: {str(e)}", 'error')
        return redirect(url_for('prompts'))

@app.route('/crawl', methods=['GET', 'POST'])
@login_required
def crawl():
    try:
        if not current_user.is_authenticated:
            logger.error("Unauthenticated user accessing crawl endpoint")
            flash('Please log in to start a crawl.', 'error')
            return redirect(url_for('login'))
        
        logger.debug(f"User authenticated: ID {current_user.id}, email {current_user.email}")
        
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT id, name FROM libraries WHERE user_id = %s", (int(current_user.id),))
        libraries = cur.fetchall()
        cur.close()
        conn.close()
        
        if request.method == 'POST':
            start_url = normalize_url(request.form.get('url'))
            library_id = request.form.get('library_id')
            if not start_url or not library_id:
                flash('URL and library selection cannot be empty.', 'error')
                return render_template('crawl.html', libraries=libraries)
            
            logger.info(f"Starting crawl for {start_url} by user {current_user.id} in library {library_id}")
            try:
                # Run crawl_website in a thread to avoid event loop conflicts
                def run_crawl(user_id, start_url, library_id):
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        return loop.run_until_complete(crawl_website(start_url, user_id, int(library_id)))
                    finally:
                        loop.close()
                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    crawled_data = executor.submit(run_crawl, current_user.id, start_url, library_id).result()
                
                if crawled_data:
                    conn = get_db_connection()
                    cur = conn.cursor()
                    query = """
                    INSERT INTO documents (url, content, embedding, file_path, library_id)
                    VALUES %s
                    ON CONFLICT (url, library_id) DO NOTHING
                    """
                    execute_values(cur, query, crawled_data)
                    conn.commit()
                    cur.close()
                    conn.close()
                    flash(f"Stored {len(crawled_data)} items in library.", 'success')
                else:
                    flash("No new textual content found; all URLs already exist or lack meaningful text.", 'info')
                
                return redirect(url_for('crawl', url=start_url, library_id=library_id))
            except psycopg2.errors.UniqueViolation as e:
                logger.info(f"Duplicate URL detected: {str(e)}")
                flash("All pages from that link have already been added to the library.", 'info')
                return redirect(url_for('crawl'))
            except Exception as e:
                logger.error(f"Error during crawl: {str(e)}\n{traceback.format_exc()}")
                flash(f"Error during crawl: {str(e)}", 'error')
                return render_template('crawl.html', libraries=libraries)
        
        return render_template('crawl.html', libraries=libraries)
    except Exception as e:
        logger.error(f"Error in crawl endpoint: {e}\n{traceback.format_exc()}")
        flash(f"Error: {str(e)}", 'error')
        return render_template('crawl.html', libraries=[])

@app.route('/crawl_progress', methods=['GET'])
@login_required
def crawl_progress():
    try:
        if not current_user.is_authenticated:
            logger.error("Unauthenticated user in crawl_progress endpoint")
            return jsonify({"status": "error", "message": "User not authenticated"}), 401
        conn = sqlite3.connect('progress.db')
        c = conn.cursor()
        c.execute("SELECT links_found, links_scanned, items_crawled, status FROM progress WHERE user_id = ? AND url = ? AND library_id = ? ORDER BY rowid DESC LIMIT 1",
                  (str(current_user.id), request.args.get('url'), request.args.get('library_id')))
        result = c.fetchone()
        conn.close()
        data = {
            "links_found": result[0] if result else 0,
            "links_scanned": result[1] if result else 0,
            "items_crawled": result[2] if result else 0,
            "status": result[3] if result else "waiting"
        }
        return jsonify(data)
    except Exception as e:
        logger.error(f"Error in crawl_progress: {e}\n{traceback.format_exc()}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/current_url', methods=['GET'])
@login_required
def current_url():
    try:
        if not current_user.is_authenticated:
            logger.error("Unauthenticated user in current_url endpoint")
            return jsonify({"status": "error", "message": "User not authenticated"}), 401
        conn = sqlite3.connect('progress.db')
        c = conn.cursor()
        c.execute("SELECT current_url FROM progress WHERE user_id = ? AND url = ? AND library_id = ? ORDER BY rowid DESC LIMIT 1",
                  (str(current_user.id), request.args.get('url'), request.args.get('library_id')))
        result = c.fetchone()
        conn.close()
        return jsonify({"current_url": result[0] if result and result[0] else ""})
    except Exception as e:
        logger.error(f"Error in current_url: {e}\n{traceback.format_exc()}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/search', methods=['GET', 'POST'])
@login_required
def search():
    try:
        if not current_user.is_authenticated:
            logger.error("Unauthenticated user accessing search endpoint")
            flash('Please log in to perform a search.', 'error')
            return redirect(url_for('login'))
        
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT id, name FROM libraries WHERE user_id = %s", (int(current_user.id),))
        libraries = cur.fetchall()
        cur.execute("SELECT id, name, content FROM prompts WHERE user_id = %s", (int(current_user.id),))
        prompts = cur.fetchall()
        cur.close()
        conn.close()
        
        if request.method == 'POST':
            query = request.form.get('query')
            library_id = request.form.get('library_id')
            prompt_id = request.form.get('prompt_id')
            if not query or not library_id or not prompt_id:
                flash('Query, library, and prompt selection cannot be empty.', 'error')
                return render_template('search.html', libraries=libraries, prompts=prompts)
            
            logger.info(f"Search query: {query} in library {library_id}")
            def run_embedding():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    tokenizer, model = loop.run_until_complete(load_embedding_model())
                    query_embedding = loop.run_until_complete(generate_embedding(query, tokenizer, model))
                    loop.run_until_complete(unload_embedding_model())
                    return query_embedding
                finally:
                    loop.close()
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                query_embedding = executor.submit(run_embedding).result()
            
            if query_embedding is None:
                flash('Failed to generate query embedding.', 'error')
                return render_template('search.html', libraries=libraries, prompts=prompts)
            
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute("""
            SELECT id, url, content, file_path, embedding
            FROM documents
            WHERE library_id = %s
            ORDER BY embedding <=> %s::vector
            LIMIT 5
            """, (int(library_id), query_embedding.tolist()))
            results = cur.fetchall()
            
            cur.execute("SELECT content FROM prompts WHERE id = %s AND user_id = %s", (int(prompt_id), int(current_user.id)))
            prompt = cur.fetchone()
            cur.close()
            conn.close()
            
            if not prompt:
                flash('Selected prompt not found.', 'error')
                return render_template('search.html', libraries=libraries, prompts=prompts)
            
            context = "\n\n".join([result[2] for result in results])
            prompt_answer = query_grok_api(query, context, prompt[0])
            if prompt_answer.startswith("Error") or prompt_answer.startswith("Fallback"):
                prompt_answer = f"{prompt_answer}\n\nRelevant Documents:"
            
            documents = [
                {
                    "url": result[1] or result[3],
                    "snippet": result[2][:100] + ("..." if len(result[2]) > 100 else "")
                }
                for result in results
            ]
            
            if not results:
                documents = [{"url": "", "snippet": "No relevant content found."}]
            
            return render_template('search.html', libraries=libraries, prompts=prompts, query=query, prompt_answer=prompt_answer, documents=documents)
        
        return render_template('search.html', libraries=libraries, prompts=prompts)
    except Exception as e:
        logger.error(f"Error in search endpoint: {e}\n{traceback.format_exc()}")
        return "Internal Server Error", 500

@app.route('/test_playwright')
def test_playwright():
    try:
        logger.info("Starting Playwright test")
        def run_playwright():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                async def inner():
                    async with async_playwright() as p:
                        browser = await p.chromium.launch()
                        page = await browser.new_page()
                        await page.goto('https://example.com')
                        content = await page.content()
                        await browser.close()
                        return content
                return loop.run_until_complete(inner())
            finally:
                loop.close()
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            content = executor.submit(run_playwright).result()
        
        logger.info("Playwright test completed successfully")
        return f"Playwright test successful: {len(content)} bytes"
    except Exception as e:
        logger.error(f"Playwright test failed: {str(e)}\n{traceback.format_exc()}")
        return f"Playwright test failed: {str(e)}"

@app.route('/health')
def health():
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    app.run(debug=True)