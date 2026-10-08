# Fallback Plan

## Dataset unavailable (Kaggle/IMDb removed or inaccessible)
- Use a locally cached copy of the downloaded dataset.
- Switch to another publicly available movie dataset with similar fields.
  (e.g. TMDb 5000) with similar fields; adapt column mapping.
- As a last resort, use a small hand-built sample CSV for demos and tests.

## Model underperforms or is unavailable
- Retrain from the cached dataset if the saved model file is missing or corrupt.
- Tune Random Forest hyperparameters or reduce features.
- Fall back to simpler baselines: Gradient Boosting, k-NN, or content-based
  filtering (cosine similarity on genre/director/actors).
- Final fallback: rank by genre match and average rating.

## External service/API unavailable
- Run fully offline from local data; treat API enrichment as optional.
- Cache API responses locally and degrade gracefully when missing.
- If Kaggle API credentials fail, download the dataset manually.

## Environment issues
- Pin dependency versions in `requirements.txt` if a release breaks things.
- Use a fresh virtual environment to reproduce problems.
