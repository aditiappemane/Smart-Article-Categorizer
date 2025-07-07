from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict
import os
import numpy as np
from backend.embeddings.glove import load_glove_vectors, document_vector
from backend.embeddings.gemini import get_gemini_embedding
from backend.embeddings.openai import get_openai_embedding
from backend.classifier import EmbeddingClassifier
from backend.utils import preprocess
from fastapi.middleware.cors import CORSMiddleware
import pickle

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load resources on startup ---
GLOVE_PATH = os.path.join(os.path.dirname(__file__), '../data/glove.6B.300d.txt')
if os.path.exists(GLOVE_PATH):
    glove_vectors = load_glove_vectors(GLOVE_PATH)
else:
    glove_vectors = {}

# Load trained classifiers
MODEL_PATHS = {
    'GloVe': os.path.join(os.path.dirname(__file__), '../data/glove_clf.pkl'),
    'Gemini': os.path.join(os.path.dirname(__file__), '../data/gemini_clf.pkl'),
    'SBERT': os.path.join(os.path.dirname(__file__), '../data/sbert_clf.pkl'),
    'OpenAI': os.path.join(os.path.dirname(__file__), '../data/openai_clf.pkl'),
}
models = {}
for key, path in MODEL_PATHS.items():
    try:
        with open(path, 'rb') as f:
            models[key] = pickle.load(f)
    except Exception:
        models[key] = None

LABELS = ['Tech', 'Finance', 'Healthcare', 'Sports', 'Politics', 'Entertainment']

class ArticleRequest(BaseModel):
    text: str

class ClassificationResponse(BaseModel):
    predictions: Dict[str, Dict[str, float]]  # {model: {label: confidence}}

@app.get("/ping")
def ping():
    return {"status": "ok"}

def merge_probs(classes, probs):
    # Defensive: merge duplicate keys and normalize
    from collections import defaultdict
    merged = defaultdict(float)
    for label, prob in zip(classes, probs):
        merged[label] += float(prob)
    total = sum(merged.values())
    if total > 0:
        for k in merged:
            merged[k] /= total
    # Ensure all LABELS are present
    for label in LABELS:
        merged.setdefault(label, 0.0)
    return dict(merged)

@app.post("/classify", response_model=ClassificationResponse)
def classify_article(req: ArticleRequest):
    text = preprocess(req.text)
    results = {}
    # GloVe
    if glove_vectors and models['GloVe']:
        glove_emb = document_vector(text, glove_vectors)
        probs = models['GloVe'].predict_proba([glove_emb])[0]
        results['GloVe'] = merge_probs(models['GloVe'].classes_, probs)
    else:
        results['GloVe'] = {label: 0.0 for label in LABELS}
    # Gemini (BERT-like)
    if models['Gemini']:
        try:
            gemini_emb = get_gemini_embedding(text)
            probs = models['Gemini'].predict_proba([gemini_emb])[0]
            results['Gemini'] = merge_probs(models['Gemini'].classes_, probs)
        except Exception:
            results['Gemini'] = {label: 0.0 for label in LABELS}
    else:
        results['Gemini'] = {label: 0.0 for label in LABELS}
    # SBERT (simulated)
    if models['SBERT']:
        try:
            sbert_emb = get_gemini_embedding(text)
            probs = models['SBERT'].predict_proba([sbert_emb])[0]
            results['SBERT'] = merge_probs(models['SBERT'].classes_, probs)
        except Exception:
            results['SBERT'] = {label: 0.0 for label in LABELS}
    else:
        results['SBERT'] = {label: 0.0 for label in LABELS}
    # OpenAI Ada
    if models['OpenAI']:
        try:
            openai_emb = get_openai_embedding(text)
            probs = models['OpenAI'].predict_proba([openai_emb])[0]
            results['OpenAI'] = merge_probs(models['OpenAI'].classes_, probs)
        except Exception:
            results['OpenAI'] = {label: 0.0 for label in LABELS}
    else:
        results['OpenAI'] = {label: 0.0 for label in LABELS}
    return {"predictions": results}

@app.get("/embedding-viz")
def embedding_viz():
    # Placeholder: return embedding cluster data
    return {"embedding_clusters": []} 