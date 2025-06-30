import os
from flask import send_from_directory
from core import app
from auth import *  # Register authentication routes
from search import *  # Register search routes
from features import *  # Register chat, prompt, notification routes

app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'your-secret-key')

@app.route('/robots.txt')
def robots_txt():
    return send_from_directory('static', 'robots.txt')

@app.route('/debug_session', methods=['GET'])
@login_required
def debug_session():
    return jsonify({'selected_search_ids': session.get('selected_search_ids', [])})

if __name__ == '__main__':
    app.run()