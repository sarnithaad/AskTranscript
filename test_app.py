from transcript_utils import load_transcript
from embed_tfidf import TFIDFSearch

chunks = load_transcript("transcript.txt")
def test_tfidf():
    s = TFIDFSearch(chunks)
    t, a = s.query("What is data science?")
    assert "data science" in a.lower()
