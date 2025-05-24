import openai
import os
from dotenv import load_dotenv
import numpy as np
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

class OpenAISearch:
    def __init__(self, chunks):
        self.timestamps, self.texts = zip(*chunks)
        self.embeddings = [self.embed(t) for t in self.texts]

    def embed(self, text):
        return openai.Embedding.create(input=text, model="text-embedding-ada-002")['data'][0]['embedding']

    def query(self, question):
        q_vec = np.array(self.embed(question))
        sims = [np.dot(q_vec, np.array(e)) for e in self.embeddings]
        idx = np.argmax(sims)
        return self.timestamps[idx], self.texts[idx]