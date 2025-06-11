from flask import render_template, request, redirect, url_for, flash, send_from_directory, make_response
from flask_login import UserMixin, login_user, logout_user, current_user, login_required
import psycopg2
from werkzeug.security import generate_password_hash, check_password_hash
from email_validator import validate_email, EmailNotValidError
from core import app, login_manager, logger, get_db_connection

class User(UserMixin):
    def __init__(self, id, email, admin=False, status='active'):
        self.id = str(id)
        self.email = email
        self.admin = admin
        self.status = status

@login_manager.user_loader
def load_user(user_id):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT id, email, admin, status FROM users WHERE id = %s", (user_id,))
        user = cur.fetchone()
        if user:
            return User(user[0], user[1], user[2], user[3])
        return None
    except Exception as e:
        logger.error(f"Error loading user: {str(e)}")
        return None
    finally:
        cur.close()
        conn.close()

def validate_user_email(email):
    try:
        validate_email(email, check_deliverability=False)
        return True
    except EmailNotValidError as e:
        logger.error(f"Invalid email address: {email}, error: {str(e)}")
        return False

@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('search'))
    response = make_response(render_template('index.html', username=None))
    response.headers['X-Content-Type-Options'] = 'nosniff'
    return response

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        conn = get_db_connection()
        cur = conn.cursor()
        try:
            cur.execute("SELECT id FROM users WHERE email = %s", (email,))
            if cur.fetchone():
                flash('Email already registered.', 'error')
            else:
                password_hash = generate_password_hash(password)
                cur.execute("INSERT INTO users (email, password_hash) VALUES (%s, %s)", (email, password_hash))
                conn.commit()
                flash('Registration successful! Please log in.', 'success')
                return redirect(url_for('login'))
        except Exception as e:
            logger.error(f"Error during registration: {str(e)}")
            conn.rollback()
            flash('Registration failed. Please try again.', 'error')
        finally:
            cur.close()
            conn.close()
    response = make_response(render_template('register.html', username=None))
    response.headers['X-Content-Type-Options'] = 'nosniff'
    return response

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        conn = get_db_connection()
        cur = conn.cursor()
        try:
            cur.execute("SELECT id, email, password_hash, status, admin FROM users WHERE email = %s", (email,))
            user = cur.fetchone()
            if user and check_password_hash(user[2], password):
                if user[3] == 'inactive':
                    flash('Your account is inactive. Please contact support at pubmedresearch@pubmedresearch.com.', 'error')
                else:
                    login_user(User(user[0], user[1], user[4], user[3]))
                    response = make_response(redirect(url_for('search')))
                    response.headers['X-Content-Type-Options'] = 'nosniff'
                    return response
            flash('Invalid email or password.', 'error')
        except Exception as e:
            logger.error(f"Error during login: {str(e)}")
            flash('Login failed. Please try again.', 'error')
        finally:
            cur.close()
            conn.close()
    response = make_response(render_template('login.html', username=None))
    response.headers['X-Content-Type-Options'] = 'nosniff'
    return response

@app.route('/logout')
@login_required
def logout():
    logout_user()
    response = make_response(redirect(url_for('login')))
    response.headers['X-Content-Type-Options'] = 'nosniff'
    return response

@app.route('/static/<path:filename>')
def static_files(filename):
    response = send_from_directory(app.static_folder, filename)
    response.headers['Cache-Control'] = 'public, max-age=31536000'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    return response