from flask import send_from_directory
from core import app
from auth import *  # Register authentication routes
from search import *  # Register search routes
from features import *  # Register chat, prompt, notification routes

@app.route('/robots.txt')
def robots_txt():
    return send_from_directory('static', 'robots.txt')

if __name__ == '__main__':
    app.run()