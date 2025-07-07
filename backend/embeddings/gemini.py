import os
import requests
from dotenv import load_dotenv
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_EMBED_URL = "https://generativelanguage.googleapis.com/v1beta/models/embedding-001:embedContent?key=" + GEMINI_API_KEY

def get_gemini_embedding(text: str) -> list:
    headers = {"Content-Type": "application/json"}
    data = {"content": {"parts": [{"text": text}]}}
    response = requests.post(GEMINI_EMBED_URL, json=data, headers=headers)
    if response.status_code == 200:
        return response.json()["embedding"]["values"]
    else:
        return [0.0]*768  # fallback 