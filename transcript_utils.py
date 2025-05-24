import re

def load_transcript(path):
    chunks = []
    with open(path, 'r') as f:
        for line in f:
            match = re.match(r'\[(.*?)\],?\s*(.*)', line)
            if match:
                timestamp, chunk = match.groups()
                chunks.append((timestamp.strip(), chunk.strip()))
    return chunks