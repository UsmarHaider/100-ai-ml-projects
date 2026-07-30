# Movie Recommendation with NLP

![NLP](https://img.shields.io/badge/domain-NLP-blue) ![Task](https://img.shields.io/badge/task-Recommender%20System-orange)

## Overview
A beginner-friendly walkthrough of recommender systems built on movie metadata. It progresses from a simple demographic chart (IMDB weighted rating) to **content-based filtering** using NLP over plot overviews and structured metadata (cast, crew, keywords, genres) with TF-IDF and cosine similarity.

## Dataset
- **TMDB 5000 Movie Metadata**: https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata (`tmdb_5000_movies.csv`, `tmdb_5000_credits.csv`)
- **The Movies Dataset** (referenced): https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset
- ~4,800 movies; metadata fields include budget, genres, keywords, overview, popularity, cast, crew, `vote_average`, `vote_count`.
- Data is **not** committed. Download with:
  ```bash
  kaggle datasets download tmdb/tmdb-movie-metadata
  ```
  The notebook reads the CSVs from `../input/tmdb-movie-metadata/`.

## Approach
1. **Demographic filtering** — compute the IMDB weighted rating `WR = (v/(v+m))·R + (m/(m+v))·C`, where `C` is the mean rating and `m` is the 90th-percentile vote count (≈481 qualifying movies), then rank.
2. **Content-based — plot description recommender:** `TfidfVectorizer(stop_words='english')` over each movie's `overview` (~20,000-term vocabulary), then cosine similarity via `linear_kernel`; recommend the top-10 most similar titles.
3. **Content-based — metadata recommender:** parse stringified `cast`/`crew`/`keywords`/`genres`, extract the director and top-3 cast, normalize tokens (lowercase, strip spaces), build a combined "soup" of metadata, and compute similarity over that.

## Results
This is an **unsupervised recommender** — it produces ranked recommendation lists rather than supervised accuracy/error metrics. The notebook demonstrates outputs such as the top-rated chart and `get_recommendations('The Dark Knight Rises')` / `get_recommendations('The Avengers')` similarity lists.

The captured run log for this project is empty. **Run the notebook to reproduce the recommendation outputs** — there are no classification/regression metrics to report; quality is assessed qualitatively (the notebook notes that the plot-only recommender returns mostly same-franchise titles, motivating the richer metadata model).

## Key Takeaways
- The IMDB weighted rating fairly handles the "high rating, few votes" problem for popularity charts.
- TF-IDF + cosine similarity over plot overviews gives reasonable but franchise-biased recommendations.
- Adding cast/director/keyword/genre metadata produces more taste-aligned suggestions than plot text alone.
- A natural extension is collaborative filtering, which needs user-item interaction data rather than item metadata.

## How to Run
```bash
pip install pandas numpy scikit-learn matplotlib
jupyter notebook notebook.ipynb
```
Download the TMDB dataset (see Dataset) and place the CSVs under `../input/tmdb-movie-metadata/`.

## Credit
> Based on ["Getting Started with a Movie Recommendation System"](https://www.kaggle.com/code/ibtesama/getting-started-with-a-movie-recommendation-system) by Ibtesam Ahmed on Kaggle (8068 votes).
> Datasets: [TMDB 5000 Movie Metadata](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata), [The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset). Adapted and documented for this portfolio.
