from flask import Flask, render_template, request, jsonify, session, redirect, url_for, Response
from google import genai
from google.genai import types
import os
import threading
from queue import Queue
import uuid
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
news_api_key = os.getenv("NEWS_API_KEY")


# Initialize the Flask app
app = Flask(__name__)
app.secret_key = 'your-secret-key'  # Required for session management

# Configure the Google Generative AI client
client = genai.Client(api_key=api_key)  # Ensure api_key is set in your environment

# Global storage for workflow progress and results
workflows = {}
workflow_lock = threading.Lock()

grounding_tool = types.Tool(
    google_search=types.GoogleSearch()
)

config = types.GenerateContentConfig(
    tools=[grounding_tool]
)

def Content_Agent(company, workflow_id):
    with workflow_lock:
        workflows[workflow_id]['messages'].append("Fetching company information...")
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"""I have an interview coming with the company named {company}. Can you provide me info about the company that can help
me with my interview Prep. give me detailed info.""",
        config=config,
    )
    with workflow_lock:
        workflows[workflow_id]['messages'].append("Content generation completed.")
    return response.text

def Content_cleaner(content, workflow_id):
    with workflow_lock:
        workflows[workflow_id]['messages'].append("Cleaning content...")
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"""Can you please clean this content and return only the text as the output. Remove unnecessary symbols and make it presentable.
Content = {content}"""
    )
    with workflow_lock:
        workflows[workflow_id]['messages'].append("Content was cleaned.")
    return response.text

def HTML_Transform(cleaned_content, workflow_id):
    with workflow_lock:
        workflows[workflow_id]['messages'].append("Transforming content to HTML...")
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"""I will provide you with some content, your goal is create an HTML Page for that content. and return me the HTML code for the same. Start directly with <!DOCTYPE html>...
Content = {cleaned_content}"""
    )
    with workflow_lock:
        workflows[workflow_id]['messages'].append("HTML code was generated.")
    return response.text

def HTML_Validator(code, workflow_id):
    with workflow_lock:
        workflows[workflow_id]['messages'].append("Validating HTML code...")
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"""I will provide you with some HTML code. Your task is to look and correct any errors if any. The page must be complete, remove broken links etc. And return
just the corrected HTML code as the output nothing else
HTML code: {code}"""
    )
    with workflow_lock:
        workflows[workflow_id]['messages'].append("HTML code was validated.")
    return response.text

def UI_Agent(html_code, workflow_id):
    with workflow_lock:
        workflows[workflow_id]['messages'].append("Styling HTML with CSS...")
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"""I will provide you with some HTML code. Your goal is to style it using CSS. Enhance the UI, make it presentable. Start directly with <!DOCTYPE html>...
HTML CODE = {html_code}"""
    )
    with workflow_lock:
        workflows[workflow_id]['messages'].append("HTML code was styled.")
    return response.text

def Final_Cleaner(data, workflow_id):
    with workflow_lock:
        workflows[workflow_id]['messages'].append("Finalizing HTML output...")
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"""Your task is to return just the HTML code present in the data as the output only. Also improve if there is any errors in the code. Return JUST the HTML code. Start directly with <!DOCTYPE html>...
The Data = {data}"""
    )
    with workflow_lock:
        workflows[workflow_id]['messages'].append("Final cleanup completed.")
    return response.text

function = [
    lambda x, wid: Content_Agent(x, wid),
    lambda x, wid: HTML_Transform(x, wid),
    lambda x, wid: HTML_Validator(x, wid),
    lambda x, wid: UI_Agent(x, wid),
    lambda x, wid: UI_Agent(x, wid),
    lambda x, wid: Final_Cleaner(x, wid)
]

def Pipeline(data, functions, workflow_id):
    for func in functions:
        data = func(data, workflow_id)
    return data

def WorkFlow(name, workflow_id):
    with workflow_lock:
        workflows[workflow_id] = {'status': 'running', 'messages': [], 'result': None}
    final_code = Pipeline(name, function, workflow_id)
    with workflow_lock:
        workflows[workflow_id]['result'] = final_code
        workflows[workflow_id]['status'] = 'complete'
    return final_code

# Routes
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/search', methods=['POST'])
def search():
    name = request.form['company_name']
    workflow_id = str(uuid.uuid4())  # Generate unique workflow ID
    try:
        # Store workflow ID in session
        session['workflow_id'] = workflow_id
        # Start workflow in a separate thread
        threading.Thread(target=WorkFlow, args=(name, workflow_id), daemon=True).start()
        return render_template('progress.html')
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/check_progress')
def check_progress():
    workflow_id = session.get('workflow_id')
    if not workflow_id or workflow_id not in workflows:
        return jsonify({
            'status': 'error',
            'messages': ['No active workflow found.'],
            'result': ''
        })
    with workflow_lock:
        if workflows[workflow_id]['status'] == 'complete':
            return jsonify({
                'status': 'complete',
                'messages': workflows[workflow_id]['messages'],
                'redirect': url_for('result')
            })
        return jsonify({
            'status': workflows[workflow_id]['status'],
            'messages': workflows[workflow_id]['messages'],
            'result': ''
        })

@app.route('/result')
def result():
    workflow_id = session.get('workflow_id')
    if not workflow_id or workflow_id not in workflows:
        return render_template('error.html', error="No workflow result found.")
    with workflow_lock:
        html_content = workflows[workflow_id].get('result', '')
    if not html_content:
        return render_template('error.html', error="No HTML content available.")
    return Response(html_content, mimetype='text/html')

@app.route('/news', methods=['POST'])
def news():
    from newsapi.newsapi_client import NewsApiClient
    name = request.form['company_name']
    newsapi = NewsApiClient(api_key=news_api_key)
    top_headlines = newsapi.get_everything(q=f'{name}', language='en')
    content = top_headlines['articles']
    content = [news for news in content if news['urlToImage'] is not None]
    return render_template('news.html', articles=content)

# Run the app
if __name__ == "__main__":
    app.run(debug=True)