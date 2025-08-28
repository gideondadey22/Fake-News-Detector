# Importing the Libraries
import os
import secrets
import urllib
import pickle
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

# MySQL and external APIs
mysql = MySQL(app)
newsapi = NewsApiClient(api_key='2aa3ac1960ca48b2a5260ebe34c37e96')

# -----------------------
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

# -----------------------
# Routes
@app.route('/', methods=['GET', 'POST'])
def main():
    data = newsapi.get_top_headlines(language='en', country="us", category='general', page_size=10)
    l1 = []
    l2 = []
    for i in data.get('articles', []):
        l1.append(i.get('title'))
        l2.append(i.get('url'))
    return render_template('main.html', l1=l1, l2=l2)

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

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """
    GET: render the prediction page (empty).
    POST: expect form field 'news' containing the article URL.
    """
    # When visiting the page first time show the form
    if request.method == 'GET':
        # Ensure template variables used in predict.html are defined to avoid Jinja errors
        return render_template('predict.html', prediction_text='', url_input='', language_error='')

    # POST: extract URL from the form
    url = request.form.get('news', '').strip()

    if not url:
        flash('Please enter a news site URL', 'danger')
        return redirect(url_for('main'))

    # If user omitted scheme (http/https) add it
    parsed = urllib.parse.urlparse(url)
    if not parsed.scheme:
        url = 'http://' + url
        parsed = urllib.parse.urlparse(url)

    # validate URL
    if not validators.url(url):
        flash('Please enter a valid news site URL', 'danger')
        return redirect(url_for('main'))

    user_agent = request.headers.get('User-Agent') or 'Mozilla/5.0'
    config = Config()
    config.browser_user_agent = user_agent

    try:
        article = Article(str(url), config=config)
        article.download()
        article.parse()

        parsed_text = article.text
        if not parsed_text:
            flash('Invalid news article! Please try again', 'danger')
            return redirect(url_for('main'))

        # detect language
        try:
            b = TextBlob(parsed_text)
            lang = b.detect_language()
        except Exception:
            # if language detection fails, assume english (or show error)
            lang = 'en'

        if lang != "en":
            language_error = "We currently do not support this language"
            return render_template('predict.html', language_error=language_error, url_input=url, prediction_text='')

        # run NLP from newspaper (for e.g. keywords if needed)
        try:
            article.nlp()
            news = article.text
        except Exception:
            news = parsed_text

        if not news:
            flash('Invalid URL! Please try again', 'danger')
            return redirect(url_for('main'))

        # load vectorizer & model (make sure filenames match actual saved files)
        cleaner = pickle.load(open('TfidfVectorizer-new.sav', 'rb'))
        model = pickle.load(open('ClassifierModel-new.sav', 'rb'))

        news_to_predict = pd.Series(np.array([news]))
        cleaned_text = cleaner.transform(news_to_predict)
        pred = model.predict(cleaned_text)
        pred_outcome = format(pred[0])

        # Map model output to display text
        if pred_outcome in ("0", "REAL", "Real", "TRUE", "True"):
            outcome = "True"
        else:
            outcome = "False"

        if session.get('logged_in'):
            saveHistory(session.get('id'), url, outcome)

        return render_template('predict.html', prediction_text=outcome, url_input=url, news=news)

    except Exception as e:
        log.error("Predict error: %s", e)
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