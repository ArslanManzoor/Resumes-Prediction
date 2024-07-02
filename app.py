import os
import re
import json
import sqlite3
from flask import Flask, request, jsonify, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import plotly.graph_objs as go
import plotly
from pdfminer.high_level import extract_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

app = Flask(__name__)

# Define the database schema
def create_db():
    conn = sqlite3.connect('resumes.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS resumes (
            id INTEGER PRIMARY KEY,
            name TEXT,
            city TEXT,
            country TEXT,
            age INTEGER,
            gender TEXT,
            job_title TEXT,
            email TEXT,
            cv_path TEXT
        )
    ''')
    conn.commit()
    conn.close()

# Read job descriptions from a .txt file and train a simple classifier to predict job fields
def train_classifier():
    job_titles = []
    job_descriptions = []

    with open('job_descriptions.txt', 'r') as file:
        for line in file:
            if ':' in line:
                job_title, job_description = line.split(':', 1)
                job_titles.append(job_title.strip())
                job_descriptions.append(job_description.strip())

    vectorizer = TfidfVectorizer(stop_words='english')
    clf = MultinomialNB()

    pipeline = make_pipeline(vectorizer, clf)
    pipeline.fit(job_descriptions, job_titles)

    return pipeline

model = train_classifier()
create_db()

# Function to extract email from text
def extract_email(text):
    email = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    return email.group(0) if email else None

@app.route('/')
def index():
    conn = sqlite3.connect('resumes.db')
    c = conn.cursor()
    c.execute('SELECT DISTINCT job_title FROM resumes')
    job_titles = c.fetchall()
    c.execute('SELECT DISTINCT country FROM resumes')
    countries = c.fetchall()
    c.execute('SELECT DISTINCT gender FROM resumes')
    genders = c.fetchall()
    conn.close()
    return render_template('index.html', job_titles=job_titles, countries=countries, genders=genders)

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    name = request.form['name']
    city = request.form['city']
    country = request.form['country']
    age = request.form['age']
    gender = request.form['gender']
    
    if file and file.filename.endswith('.pdf'):
        filename = secure_filename(file.filename)
        file_path = os.path.join('uploads', filename)
        file.save(file_path)

        text = extract_text(file_path)
        email = extract_email(text)
        job_title = model.predict([text])[0]

        conn = sqlite3.connect('resumes.db')
        c = conn.cursor()
        c.execute('''
            INSERT INTO resumes (name, city, country, age, gender, job_title, email, cv_path) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (name, city, country, age, gender, job_title, email, file_path))
        conn.commit()
        conn.close()

        return redirect(url_for('index'))

    return 'Invalid file format. Please upload a PDF file.', 400

@app.route('/show_results')
def show_results():
    page = request.args.get('page', 1, type=int)
    per_page = 10
    offset = (page - 1) * per_page

    with sqlite3.connect('resumes.db') as conn:
        c = conn.cursor()
        c.execute('SELECT COUNT(*) FROM resumes')
        total_records = c.fetchone()[0]
        total_pages = (total_records + per_page - 1) // per_page

        c.execute('SELECT * FROM resumes LIMIT ? OFFSET ?', (per_page, offset))
        resumes = c.fetchall()

    return render_template('results.html', resumes=resumes, page=page, total_pages=total_pages)

@app.route('/analysis')
def analysis():
    with sqlite3.connect('resumes.db') as conn:
        c = conn.cursor()
        c.execute('SELECT DISTINCT gender FROM resumes')
        genders = c.fetchall()

        c.execute('SELECT DISTINCT job_title FROM resumes')
        job_titles = c.fetchall()

        c.execute('SELECT DISTINCT country FROM resumes')
        countries = c.fetchall()

    return render_template('analysis.html', genders=genders, job_titles=job_titles, countries=countries)

@app.route('/get_bar_data', methods=['POST'])
def get_bar_data():
    gender = request.form.get('gender')
    job_title = request.form.get('job_title')
    country = request.form.get('country')

    query = 'SELECT country, gender, COUNT(*) as count FROM resumes WHERE 1=1'
    params = []

    if gender:
        query += ' AND gender = ?'
        params.append(gender)
    if job_title:
        query += ' AND job_title = ?'
        params.append(job_title)
    if country:
        query += ' AND country = ?'
        params.append(country)

    query += ' GROUP BY country, gender ORDER BY count DESC LIMIT 10'

    with sqlite3.connect('resumes.db') as conn:
        c = conn.cursor()
        c.execute(query, params)
        data = c.fetchall()

    countries = [row[0] for row in data]
    genders = [row[1] for row in data]
    counts = [row[2] for row in data]

    fig = go.Figure()

    for gender in set(genders):
        x = [country for country, gen, count in data if gen == gender]
        y = [count for country, gen, count in data if gen == gender]
        fig.add_trace(go.Bar(x=x, y=y, name=gender, text=y, textposition='auto'))

    fig.update_layout(title='Top 10 Countries by Resume Count',
                      xaxis_title='Country',
                      yaxis_title='Count',
                      barmode='group')

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON

@app.route('/get_line_data', methods=['POST'])
def get_line_data():
    gender = request.form.get('gender')
    job_title = request.form.get('job_title')
    country = request.form.get('country')

    if not gender:
        query = '''
            SELECT job_title, SUM(CASE WHEN gender = 'Male' THEN 1 ELSE 0 END) AS male_count,
                   SUM(CASE WHEN gender = 'Female' THEN 1 ELSE 0 END) AS female_count
            FROM resumes
            GROUP BY job_title
            ORDER BY COUNT(*) DESC
            LIMIT 10
        '''
    else:
        query = '''
            SELECT job_title, SUM(CASE WHEN gender = ? THEN 1 ELSE 0 END) AS count
            FROM resumes
            WHERE gender = ?
            GROUP BY job_title
            ORDER BY COUNT(*) DESC
            LIMIT 10
        '''

    with sqlite3.connect('resumes.db') as conn:
        c = conn.cursor()
        if not gender:
            c.execute(query)
        else:
            c.execute(query, (gender, gender))
        data = c.fetchall()

    job_titles = [row[0] for row in data]
    counts = [row[1] for row in data]

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=job_titles, y=counts, mode='lines+markers', name='Job Titles', text=counts, textposition='top center'))

    fig.update_layout(title='Top 10 Job Titles by Gender',
                      xaxis_title='Job Title',
                      yaxis_title='Count')

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON

@app.route('/get_counts', methods=['GET'])
def get_counts():
    gender = request.args.get('gender')
    job_title = request.args.get('job_title')
    country = request.args.get('country')

    query = 'SELECT COUNT(*) FROM resumes WHERE 1=1'
    params = []

    if gender:
        query += ' AND gender = ?'
        params.append(gender)
    if job_title:
        query += ' AND job_title = ?'
        params.append(job_title)
    if country:
        query += ' AND country = ?'
        params.append(country)

    with sqlite3.connect('resumes.db') as conn:
        c = conn.cursor()
        c.execute(query, params)
        total_resumes = c.fetchone()[0]

        if not gender:
            c.execute('SELECT COUNT(*) FROM resumes WHERE gender = "Male"')
            total_males = c.fetchone()[0]
            c.execute('SELECT COUNT(*) FROM resumes WHERE gender = "Female"')
            total_females = c.fetchone()[0]
        else:
            c.execute('SELECT COUNT(*) FROM resumes WHERE gender = ?', (gender,))
            total_males = c.fetchone()[0]
            c.execute('SELECT COUNT(*) FROM resumes WHERE gender = ?', (gender,))
            total_females = c.fetchone()[0]

    return jsonify({
        'total_resumes': total_resumes,
        'total_males': total_males,
        'total_females': total_females
    })

if __name__ == '__main__':
    app.run(debug=True)
