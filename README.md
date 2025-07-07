# Smart Article Categorizer

A system to classify articles into 6 categories (Tech, Finance, Healthcare, Sports, Politics, Entertainment) using multiple embedding models and compare their performance.

## Features
- Embedding models: GloVe, Gemini (BERT/SBERT), OpenAI Ada
- Logistic Regression classifiers for each embedding
- Web UI for real-time predictions and model comparison
- Embedding cluster visualization
- Performance analysis and recommendations

## Project Structure

```
backend/         # FastAPI backend, embeddings, classifiers
  embeddings/    # Embedding logic (GloVe, Gemini, OpenAI)
frontend/        # React frontend app
notebooks/       # Analysis and comparison notebooks
 data/           # Datasets and pre-trained vectors
README.md        # Project overview
requirements.txt # Python dependencies
.env             # API keys (Gemini, OpenAI)
```

## Performance Analysis

After training, evaluate each classifier (GloVe, Gemini, SBERT, OpenAI Ada) on a test set using the following metrics:
- Accuracy
- Precision
- Recall
- F1-score

### Example Results Table

| Model      | Accuracy | Precision | Recall | F1-score |
|------------|----------|-----------|--------|----------|
| GloVe      | 0.83     | 0.82      | 0.81   | 0.81     |
| Gemini     | 0.85     | 0.84      | 0.85   | 0.84     |
| SBERT      | 0.88     | 0.87      | 0.88   | 0.87     |
| OpenAI Ada | 0.90     | 0.89      | 0.90   | 0.89     |


## Recommendations

- **Use Real Embeddings:** For best results, always use real pre-trained embeddings (e.g., GloVe, OpenAI Ada) rather than random vectors.
- **Model Selection:** Choose the embedding model with the highest F1-score for your use case. In most cases, transformer-based embeddings (OpenAI Ada, Gemini/SBERT) outperform GloVe on news classification tasks.
- **Class Imbalance:** If your dataset is imbalanced, use `class_weight='balanced'` in your classifier (as done for OpenAI Ada) to improve minority class performance.
- **API Keys:** Store all API keys (Gemini, OpenAI) in your `.env` file and never commit them to version control.
- **Frontend/Backend Security:** Restrict CORS origins in production to only allow your frontend domain.
- **Scalability:** For larger datasets, consider using batch processing for embedding extraction and model training.
- **Visualization:** Use embedding cluster visualizations (e.g., UMAP/t-SNE) to gain insights into how articles are grouped by topic.
- **Continuous Evaluation:** Regularly evaluate your models on new data and retrain as needed to maintain accuracy.
- **Deployment:** Use Docker or cloud services for scalable deployment of both backend and frontend.
- **Documentation:** Keep your README and code comments up to date for easier collaboration and maintenance.

 

 
