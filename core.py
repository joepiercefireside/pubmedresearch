import os
from flask import Flask
from flask_login import LoginManager
from apscheduler.schedulers.background import BackgroundScheduler
import logging
import sendgrid
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'default-secret-key')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add Jinja2 filter for datetime formatting
@app.template_filter('datetimeformat')
def datetimeformat_filter(timestamp):
    from datetime import datetime
    if isinstance(timestamp, (int, float)):
        return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
    return timestamp

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

scheduler = BackgroundScheduler()
scheduler.start()

sg = sendgrid.SendGridAPIClient(os.getenv('SENDGRID_API_KEY'))

def update_search_progress(user_id, query, message):
    logger.info(f"Search progress for user {user_id}, query '{query}': {message}")

def query_grok_api(prompt, context):
    # Placeholder for actual Grok API call
    return f"AI response to '{prompt}' based on context: {context[:50]}..."

def generate_embedding(text):
    # Placeholder for embedding generation
    return [0.1, 0.2, 0.3]

def get_db_connection():
    # Placeholder for database connection
    import psycopg2
    return psycopg2.connect(os.getenv('DATABASE_URL'))

from features import *

@login_manager.user_loader
def load_user(user_id):
    # Placeholder for user loading logic
    from auth import User
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, email FROM users WHERE id = %s", (user_id,))
    user_data = cur.fetchone()
    cur.close()
    conn.close()
    return User(user_data[0], user_data[1]) if user_data else None

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=os.getenv('FLASK_DEBUG', 'False') == 'True')