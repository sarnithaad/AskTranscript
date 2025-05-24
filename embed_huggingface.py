from sentence_transformers import SentenceTransformer, util

class HuggingFaceSearch:
    def __init__(self, chunks):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.timestamps, self.texts = zip(*chunks)
        self.embeddings = self.model.encode(self.texts, convert_to_tensor=True)

    def query(self, question):
        q_vec = self.model.encode(question, convert_to_tensor=True)
        sims = util.cos_sim(q_vec, self.embeddings)[0]
        idx = sims.argmax().item()
        return self.timestamps[idx], self.texts[idx]