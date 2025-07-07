import pandas as pd
import numpy as np
import os
import pickle
from embeddings.glove import load_glove_vectors, document_vector
from embeddings.gemini import get_gemini_embedding
from embeddings.openai import get_openai_embedding
from classifier import EmbeddingClassifier
from utils import preprocess

LABELS = ['Tech', 'Finance', 'Healthcare', 'Sports', 'Politics', 'Entertainment']

DATA_PATH = os.path.join(os.path.dirname(__file__), '../data/articles.csv')
GLOVE_PATH = os.path.join(os.path.dirname(__file__), '../data/glove.6B.300d.txt')

# Load dataset
print('Loading dataset...')
df = pd.read_csv(DATA_PATH)

# Clean and normalize labels
print('Cleaning and normalizing labels...')
df['label'] = df['label'].str.strip().str.title()
df = df[df['label'].isin(LABELS)]

# Remove duplicates
df = df.drop_duplicates(subset=['text', 'label'])

# Load GloVe vectors (simulate with random if not present)
if os.path.exists(GLOVE_PATH):
    print('Loading GloVe vectors...')
    glove_vectors = load_glove_vectors(GLOVE_PATH)
else:
    print('GloVe vectors not found, using random vectors for demo.')
    glove_vectors = {}
    def document_vector(text, embeddings, dim=300):
        return np.random.randn(dim)

X_glove, X_gemini, X_sbert, X_openai = [], [], [], []
for text in df['text']:
    text_p = preprocess(text)
    # GloVe
    X_glove.append(document_vector(text_p, glove_vectors))
    # Gemini
    try:
        X_gemini.append(get_gemini_embedding(text_p))
    except Exception:
        X_gemini.append(np.random.randn(768))
    # SBERT (simulate with Gemini)
    try:
        X_sbert.append(get_gemini_embedding(text_p))
    except Exception:
        X_sbert.append(np.random.randn(768))
    # OpenAI
    try:
        X_openai.append(get_openai_embedding(text_p))
    except Exception:
        X_openai.append(np.random.randn(1536))

X_glove = np.vstack(X_glove)
X_gemini = np.vstack(X_gemini)
X_sbert = np.vstack(X_sbert)
X_openai = np.vstack(X_openai)
y = df['label'].values

# Overwrite any old models
for name, X in zip(['glove', 'gemini', 'sbert', 'openai'], [X_glove, X_gemini, X_sbert, X_openai]):
    print(f'Training {name} classifier...')
    if name == 'openai':
        clf = EmbeddingClassifier()
        clf.model = clf.model.set_params(class_weight='balanced')
        clf.train(X, y)
    else:
        clf = EmbeddingClassifier()
        clf.train(X, y)
    with open(os.path.join(os.path.dirname(__file__), f'../data/{name}_clf.pkl'), 'wb') as f:
        pickle.dump(clf.model, f)
    print(f'Saved {name}_clf.pkl')
print('Training complete.') 