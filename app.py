# app.py
import os
import secrets
import urllib
import pickle
from pathlib import Path
import joblib
import requests
import validators
import numpy as np
import pandas as pd
from functools import wraps
from textblob import TextBlob
from flask import Flask, request, render_template, redirect, url_for, session, flash
from flask_cors import CORS
from flask.logging import create_logger
from flask_mysqldb import MySQL
from werkzeug.security import generate_password_hash, check_password_hash
from newspaper import Article, Config
from newsapi import NewsApiClient
import logging

# --- Config & app setup ---
secret = secrets.token_urlsafe(32)
app = Flask(__name__, template_folder='templates')
CORS(app)
app.config['SECRET_KEY'] = secret
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'fakenewsapp_db'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'
log = create_logger(app)
log.setLevel(logging.INFO)

# MySQL and external APIs
mysql = MySQL(app)
newsapi = NewsApiClient(api_key='2aa3ac1960ca48b2a5260ebe34c37e96')

# Globals for model artifacts (loaded at startup if available)
CLEANER = None
MODEL = None
LABEL_ENCODER = None
MODELS_DIR = Path("models")

def find_first_existing(paths):
    for p in paths:
        p = Path(p)
        if p.exists():
            return p
    return None

def load_model_artifacts():
    global CLEANER, MODEL, LABEL_ENCODER
    # Candidate filenames (keeps backward compatibility with various saved names)
    vect_candidates = [
        MODELS_DIR / "vectorizer.joblib",
        MODELS_DIR / "TfidfVectorizer.joblib",
        Path("TfidfVectorizer-new.sav"),
        Path("TfidfVectorizer.sav"),
        MODELS_DIR / "TfidfVectorizer-new.sav",
        MODELS_DIR / "TfidfVectorizer.sav",
    ]
    model_candidates = [
        MODELS_DIR / "pac.joblib",
        MODELS_DIR / "pac.pkl",
        MODELS_DIR / "ClassifierModel.joblib",
        Path("ClassifierModel-new.sav"),
        Path("ClassifierModel.sav"),
        MODELS_DIR / "ClassifierModel-new.sav",
        MODELS_DIR / "ClassifierModel.sav",
    ]
    le_candidates = [
        MODELS_DIR / "label_encoder.joblib",
        MODELS_DIR / "label_encoder.pkl",
        MODELS_DIR / "le.joblib",
    ]

    vect_path = find_first_existing(vect_candidates)
    model_path = find_first_existing(model_candidates)
    le_path = find_first_existing(le_candidates)

    if not vect_path or not model_path:
        log.warning("Model artifacts not found on startup. vect: %s model: %s", vect_path, model_path)
        CLEANER, MODEL, LABEL_ENCODER = None, None, None
        return

    # Load vectorizer
    try:
        CLEANER = joblib.load(vect_path)
        log.info("Loaded vectorizer from %s", vect_path)
    except Exception:
        try:
            CLEANER = pickle.load(open(vect_path, 'rb'))
            log.info("Loaded vectorizer (pickle) from %s", vect_path)
        except Exception as e:
            CLEANER = None
            log.exception("Failed loading vectorizer %s: %s", vect_path, e)

    # Load model
    try:
        MODEL = joblib.load(model_path)
        log.info("Loaded model from %s", model_path)
    except Exception:
        try:
            MODEL = pickle.load(open(model_path, 'rb'))
            log.info("Loaded model (pickle) from %s", model_path)
        except Exception as e:
            MODEL = None
            log.exception("Failed loading model %s: %s", model_path, e)

    # Load label encoder (optional)
    if le_path and le_path.exists():
        try:
            LABEL_ENCODER = joblib.load(le_path)
            log.info("Loaded label encoder from %s", le_path)
        except Exception:
            try:
                LABEL_ENCODER = pickle.load(open(le_path, 'rb'))
                log.info("Loaded label encoder (pickle) from %s", le_path)
            except Exception as e:
                LABEL_ENCODER = None
                log.exception("Failed loading label encoder %s: %s", le_path, e)
    else:
        LABEL_ENCODER = None
        log.info("Label encoder not found (optional)")

# Attempt load at startup
load_model_artifacts()

# Helper: login decorator (session-based)
def is_logged_in(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if session.get('logged_in'):
            return f(*args, **kwargs)
        else:
            flash('Please login to gain access of this page', 'danger')
            return redirect(url_for('login'))
    return wrap

# Routes
@app.route('/', methods=['GET', 'POST'])
def main():
    try:
        data = newsapi.get_top_headlines(language='en', country="us", category='general', page_size=10)
        l1 = []
        l2 = []
        for i in data.get('articles', []):
            l1.append(i.get('title'))
            l2.append(i.get('url'))
        return render_template('main.html', l1=l1, l2=l2)
    except Exception as e:
        log.warning("newsapi failure on main(): %s", e)
        return render_template('main.html', l1=[], l2=[])

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/articles')
def articles():
    data = newsapi.get_top_headlines(language='en', country="us", category='general', page_size=20)
    articles = data.get('articles', [])
    return render_template('articles.html', articles=articles)

@app.route('/login', methods=['GET'])
def login():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login_post():
    if session.get('logged_in'):
        return redirect(url_for('history'))

    email = request.form.get('email')
    password = request.form.get('password')

    try:
        cursor = mysql.connection.cursor()
        cursor.execute('SELECT * FROM users WHERE email = %s', (email,))
        account = cursor.fetchone()
        cursor.close()

        if not account:
            flash('Account with this email address does not exist! Please try again!', 'danger')
            return render_template('login.html', email=email)

        password_db = account['password_hash']
        if check_password_hash(password_db, password):
            session['logged_in'] = True
            session['username'] = account['username']
            session['id'] = account['id']
            flash('You have successfully logged in!', 'success')
            return redirect(url_for('main'))
        else:
            flash('Wrong password! Please try again!', 'danger')
            return render_template('login.html', email=email)
    except Exception as e:
        log.error("Database error during login: %s", e)
        flash('We are having trouble connecting to the database! Please try again later!', 'danger')
        return render_template('login.html', email=email)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email')
        username = request.form.get('username')
        password = request.form.get('password')

        if not email or not username or not password:
            flash('Please fill out the form to register!', 'danger')
            return render_template('register.html', email=email, username=username)

        if len(password) < 8:
            flash('Please use a stronger password (*Password must have at least 8 characters)', 'danger')
            return render_template('register.html', email=email, username=username)

        password_hash = generate_password_hash(password)
        try:
            cursor = mysql.connection.cursor()
            cursor.execute('SELECT * FROM users WHERE email = %s', (email,))
            account = cursor.fetchone()
            if account:
                cursor.close()
                flash('Email already exists!', 'danger')
                return render_template('register.html', username=username)
            cursor.execute("INSERT INTO users(email, username, password_hash) VALUES(%s,%s,%s)",
                           (email, username, password_hash))
            mysql.connection.commit()
            cursor.close()
            flash('You have successfully registered and you are allowed to login', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            log.error("Database error during register: %s", e)
            flash('Registration failed due to a server error. Please try again later.', 'danger')
            return render_template('register.html', email=email, username=username)

    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('main'))

@app.route('/history', methods=['GET', 'POST'])
@is_logged_in
def history():
    userID = session.get('id')
    cursor = mysql.connection.cursor()
    cursor.execute('SELECT * FROM history WHERE userID = %s ORDER BY historyDate DESC', (userID,))
    history = cursor.fetchall()
    cursor.close()
    if history:
        return render_template('history.html', history=history, record=True)
    else:
        msg = 'No History Found'
        return render_template('history.html', msg=msg, record=False)

# Improved predict route: safer fetching, model file fallbacks, better logging
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        # Provide empty values so template doesn't error on initial load
        return render_template('predict.html', prediction_text='', url_input='', language_error='')

    url = request.form.get('news', '').strip()
    if not url:
        flash('Please enter a news site URL', 'danger')
        return redirect(url_for('main'))

    # Ensure scheme present
    parsed = urllib.parse.urlparse(url)
    if not parsed.scheme:
        url = 'http://' + url

    if not validators.url(url):
        flash('Please enter a valid news site URL', 'danger')
        return redirect(url_for('main'))

    # Setup newspaper config with a sensible UA and timeout
    user_agent = request.headers.get('User-Agent') or 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
    cfg = Config()
    cfg.browser_user_agent = user_agent
    cfg.request_timeout = 10

    try:
        # Try newspaper first
        article = Article(url, config=cfg)
        article.download()
        article.parse()

        # If parse failed to extract text, fallback to requests + feed html to newspaper
        if not article.text or len(article.text.strip()) < 50:
            try:
                r = requests.get(url, headers={"User-Agent": user_agent}, timeout=10)
                r.raise_for_status()
                article.set_html(r.text)
                article.parse()
            except Exception as re:
                log.warning("Requests fallback failed for %s: %s", url, re)

        parsed_text = article.text or ""
        if not parsed_text or len(parsed_text.strip()) == 0:
            flash('Invalid news article! Could not extract text. Try a different article.', 'danger')
            return redirect(url_for('main'))

        # detect language (TextBlob can fail; fallback to 'en')
        try:
            b = TextBlob(parsed_text)
            lang = b.detect_language()
        except Exception:
            lang = 'en'
        if lang != 'en':
            language_error = "We currently do not support this language"
            return render_template('predict.html', language_error=language_error, url_input=url, prediction_text='')

        # Ensure model artifacts are loaded (load again if None)
        global CLEANER, MODEL, LABEL_ENCODER
        if CLEANER is None or MODEL is None:
            load_model_artifacts()

        if CLEANER is None or MODEL is None:
            flash('Model files are missing or failed to load. Please run the training script to create model files.', 'danger')
            return redirect(url_for('main'))

        # Predict
        news = parsed_text
        news_to_predict = [news]
        cleaned_text = CLEANER.transform(news_to_predict)
        pred = MODEL.predict(cleaned_text)

        # Map numeric label back to original label if encoder exists
        if LABEL_ENCODER is not None:
            try:
                pred_label = LABEL_ENCODER.inverse_transform(pred)[0]
            except Exception:
                pred_label = str(pred[0])
        else:
            pred_label = str(pred[0])

        # Normalize outcome to the UI expectation "True"/"False"
        if str(pred_label).lower() in ("real", "true", "1", "0"):
            outcome = "True"
        else:
            outcome = "False"

        if session.get('logged_in'):
            saveHistory(session.get('id'), url, outcome)

        return render_template('predict.html', prediction_text=outcome, url_input=url, news=news)

    except Exception as e:
        # Log detailed exception so you can inspect in terminal
        log.exception("Predict error for URL %s: %s", url, e)
        flash('We currently do not support this website or the article could not be parsed! Please try again', 'danger')
        return redirect(url_for('main'))

def saveHistory(userID, url, outcome):
    try:
        cursor = mysql.connection.cursor()
        cursor.execute("INSERT INTO history(historyURL, historyLabel, userID) VALUES(%s,%s,%s)", (url, outcome, userID))
        mysql.connection.commit()
        cursor.close()
    except Exception as e:
        log.error("Failed saving history: %s", e)

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(port=port, debug=True, use_reloader=False)