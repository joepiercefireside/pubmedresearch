from flask import Flask, request, jsonify, render_template, flash, url_for
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import json
import asyncio
import logging

app = Flask(__name__)
app.secret_key = 'your-secret-key'  # Replace with a secure key
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Mock User class for Flask-Login (replace with actual user model)
class User(UserMixin):
    def __init__(self, id):
        self.id = id

@login_manager.user_loader
def load_user(user_id):
    return User(user_id)

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Placeholder for async AI summary generation (replace with actual implementation)
async def generate_prompt_output(query, results, prompt_text, prompt_params, is_fallback=False):
    # Simulate AI summary generation
    return f"Summary for {query}: {prompt_text}"

# Placeholder for search progress update (replace with actual implementation)
def update_search_progress(user_id, query, status):
    pass

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['GET', 'POST'])
@login_required
def search():
    if request.method == 'POST':
        query = request.form.get('query', '')
        prompt_id = request.form.get('prompt_id', '')
        prompt_text = request.form.get('prompt_text', '')
        search_older = 'search_older' in request.form
        start_year = request.form.get('start_year', '')

        # Mock search results (replace with actual PubMed API call)
        results = [{'id': '1', 'title': 'Sample Result', 'authors': 'Author', 'journal': 'Journal', 'publication_date': '2023', 'abstract': 'Abstract'}]
        ranked_results = results[:10]
        fallback_results = []
        prompts = [{'id': '1', 'prompt_name': 'Summarize', 'prompt_text': 'Summarize the top 3 results'}]
        prompt_params = {}

        has_prompt = bool(prompt_text or prompt_id)
        return render_template('search.html', query=query, prompt_id=prompt_id, prompt_text=prompt_text,
                               search_older=search_older, start_year=start_year, results=results,
                               ranked_results=ranked_results, fallback_results=fallback_results,
                               prompts=prompts, prompt_params=prompt_params, has_prompt=has_prompt,
                               username=current_user.id)
    return render_template('search.html', prompts=[], username=current_user.id)

@app.route('/search_summary', methods=['POST'])
@login_required
async def search_summary():
    query = request.form.get('query', '')
    prompt_text = request.form.get('prompt_text', '')
    results = json.loads(request.form.get('results', '[]'))
    fallback_results = json.loads(request.form.get('fallback_results', '[]'))
    prompt_params = json.loads(request.form.get('prompt_params', '{}'))
    
    try:
        prompt_output = await generate_prompt_output(query, results, prompt_text, prompt_params)
        fallback_prompt_output = await generate_prompt_output(query, fallback_results, prompt_text, prompt_params, is_fallback=True) if fallback_results else ''
        
        if prompt_output.startswith("Fallback:"):
            flash("AI summarization used fallback.", "warning")
        if fallback_prompt_output and fallback_prompt_output.startswith("Fallback:"):
            flash("AI summarization used fallback for fallback results.", "warning")
        
        update_search_progress(current_user.id, query, "complete")
        
        return jsonify({
            'status': 'success',
            'prompt_output': prompt_output,
            'fallback_prompt_output': fallback_prompt_output
        })
    except Exception as e:
        logger.error(f"Error generating AI summary: {str(e)}")
        update_search_progress(current_user.id, query, f"error: AI summary failed: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f"AI summary failed: {str(e)}"
        })

@app.route('/prompt')
@login_required
def prompt():
    return render_template('prompt.html', username=current_user.id)

@app.route('/notifications')
@login_required
def notifications():
    return render_template('notifications.html', username=current_user.id)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user_id = request.form.get('username')  # Simplified login
        user = User(user_id)
        login_user(user)
        return redirect(url_for('index'))
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/register')
def register():
    return render_template('register.html')

if __name__ == '__main__':
    app.run(debug=True)