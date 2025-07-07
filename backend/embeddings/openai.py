import os
import requests
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBED_URL = "https://api.openai.com/v1/embeddings"

headers = {
    "Authorization": f"Bearer {OPENAI_API_KEY}",
    "Content-Type": "application/json"
}

def get_openai_embedding(text: str, model: str = "text-embedding-ada-002") -> list:
    data = {"input": text, "model": model}
    response = requests.post(OPENAI_EMBED_URL, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()["data"][0]["embedding"]
    else:
        return [0.0]*1536  # fallback 