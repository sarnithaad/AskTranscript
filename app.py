from flask import Flask, request, render_template_string
from transcript_utils import load_transcript
from embed_tfidf import TFIDFSearch
from embed_openai import OpenAISearch
from embed_huggingface import HuggingFaceSearch

app = Flask(__name__)
chunks = load_transcript('transcript.txt')

@app.route('/', methods=['GET', 'POST'])
def index():
    result = ""
    if request.method == 'POST':
        q = request.form['question']
        mode = request.form['mode']
        searcher = {'tfidf': TFIDFSearch, 'llm1': OpenAISearch, 'llm2': HuggingFaceSearch}[mode](chunks)
        t, a = searcher.query(q)
        result = f"[{t}], {a}"
    return render_template_string('''
    <form method="post">
      <input name="question" placeholder="Ask a question">
      <select name="mode">
        <option value="tfidf">TF-IDF</option>
        <option value="llm1">OpenAI</option>
        <option value="llm2">HuggingFace</option>
      </select>
      <button>Ask</button>
    </form>
    <p>{{result}}</p>
    ''', result=result)