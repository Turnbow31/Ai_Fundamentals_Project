# Ai_Fundamentals_Project

Final project: a movie recommendation tool that suggests movies based on a
user's preferences. It will use a Kaggle/IMDb movie dataset and a Random Forest
model, using attributes such as genre, director, actors, runtime, and ratings.

> Status: initial repository structure only. No model or application logic yet.

## Structure

- `src/` – application and machine learning code
- `data/` – dataset documentation (see `data/README.md`); raw data is not committed
- `docs/fallback-plan.md` – fallback options if the dataset, model, or a service is unavailable
- `tests/` – future tests
- `.env.example` – template for environment variables

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # then fill in your own values; never commit .env
```

Download the dataset as described in `data/README.md`.
