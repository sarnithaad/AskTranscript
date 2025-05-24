AskTranscript

Overview:
  This project implements a Semantic Search System that allows users to perform question answering (Q&A) on video transcripts using multiple search methods including:
    TF-IDF Vectorizer for keyword-based search
    OpenAI Embeddings for semantic search using large language models
    Hugging Face Embeddings as an alternative semantic search method

The system provides both:
  Command Line Interface (CLI) for flexible usage
  Web Interface (Flask-based) for easy interaction through a browser

Approach:
1. Transcript Processing:
   Transcripts are segmented into chunks or paragraphs for indexing.
2. Vectorization:
   Depending on the method selected:
     TF-IDF converts text chunks into sparse vectors based on word frequency.
     OpenAI and Hugging Face generate dense semantic embeddings.
3. Search:
   When a user inputs a query, the system:
     Converts the query into the corresponding vector.
     Calculates similarity scores with transcript vectors.
     Returns the most relevant transcript chunks as answers.
4. User Interfaces: 
     CLI mode for direct terminal use.
     Flask web app for GUI-based querying.
     Optional integration with Streamlit or Gradio for enhanced UI (optional).

Dependencies:
Python 3.8 or higher
Libraries:
  `numpy`
  `scikit-learn`
  `flask`
  `requests`
  `openai`
  `transformers`
  `sentence-transformers`
  `gunicorn` (for deployment)
  `pytest` (for unit testing)

You can install dependencies via:
  pip install -r requirements.txt
  
Setup and Usage:
1. Clone the repository
  git clone https://github.com/yourusername/semantic-search-transcript-qa.git
  cd semantic-search-transcript-qa
2. Create and activate virtual environment
  .\venv\Scripts\activate
3. Install dependencies
  pip install -r requirements.txt
4. Configure API Keys
  If using OpenAI embeddings, add your API key as an environment variable:
    set OPENAI_API_KEY="your_openai_api_key"     
5. Run CLI example
  python cli_search.py --method openai --transcript data/sample_transcript.txt --query "What is the main topic?"
6. Run the Web App
  export FLASK_APP=app.py
  flask run
  Then open http://127.0.0.1:5000 in your browser.
7. Optional: Run with Docker
Build and run the Docker container:
  docker build -t semantic-search-qa .
  docker run -p 5000:5000 semantic-search-qa
  Open the web app at http://localhost:5000.

Testing:
Run unit tests with:
  pytest tests/
