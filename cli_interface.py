import sys
from transcript_utils import load_transcript
from embed_tfidf import TFIDFSearch
from embed_openai import OpenAISearch
from embed_huggingface import HuggingFaceSearch

path, mode = sys.argv[1], sys.argv[2]
chunks = load_transcript(path)
print("Transcript loaded, please ask your question (press 8 for exit):")

searcher = {'tfidf': TFIDFSearch, 'llm1': OpenAISearch, 'llm2': HuggingFaceSearch}[mode](chunks)

while True:
    q = input("Q: ")
    if q == '8': break
    t, a = searcher.query(q)
    print(f"[{t}], {a}")