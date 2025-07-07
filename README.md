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



### Recommendations
- **Best Performing Model:** Based on the above metrics, select the model with the highest F1-score or best trade-off for your use case.
- **Analysis:** Discuss why certain embeddings performed better (e.g., OpenAI Ada may outperform GloVe due to richer context understanding).
- **Suggestions:** If class imbalance is present, using `class_weight='balanced'` (as done for OpenAI Ada) can improve results.

 