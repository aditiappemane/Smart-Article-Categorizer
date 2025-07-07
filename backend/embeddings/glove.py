import numpy as np
import os

def load_glove_vectors(glove_path: str):
    embeddings = {}
    with open(glove_path, 'r', encoding='utf8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

def document_vector(text: str, embeddings: dict, dim: int = 300):
    words = text.split()
    vectors = [embeddings[w] for w in words if w in embeddings]
    if not vectors:
        return np.zeros(dim)
    return np.mean(vectors, axis=0) 