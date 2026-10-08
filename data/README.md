# Data

Raw data files are not committed (see `.gitignore`). Place them in this directory.

## Source

- Primary: a movie dataset from [Kaggle](https://www.kaggle.com/datasets) built from
  [IMDb](https://www.imdb.com/) data (genre, director, actors, runtime, ratings).
  The exact dataset will be chosen and recorded here (name, URL, license, date downloaded).
- Official IMDb non-commercial datasets: https://developer.imdb.com/non-commercial-datasets/

## Download

Download manually from Kaggle, or with the Kaggle CLI using credentials from `.env`:

```bash
kaggle datasets download -d <owner>/<dataset> -p data --unzip
```

Respect each dataset's license and terms of use.
