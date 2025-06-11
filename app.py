from core import app
from auth import *  # Register authentication routes
from search import *  # Register search routes
from features import *  # Register chat, prompt, notification routes

if __name__ == '__main__':
    app.run()