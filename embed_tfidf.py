from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TFIDFSearch:
    def __init__(self, chunks):
        self.timestamps, self.texts = zip(*chunks)
        self.vectorizer = TfidfVectorizer()
        self.vectors = self.vectorizer.fit_transform(self.texts)

    def query(self, question):
        q_vec = self.vectorizer.transform([question])
        sims = cosine_similarity(q_vec, self.vectors).flatten()
        idx = sims.argmax()
        return self.timestamps[idx], self.texts[idx]