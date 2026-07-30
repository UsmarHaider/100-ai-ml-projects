# Netflix Movies & TV Shows — Text Clustering & Recommender

![ML](https://img.shields.io/badge/domain-NLP%20%2F%20Unsupervised-blueviolet)
![Task](https://img.shields.io/badge/task-Clustering%20%2B%20Recommender-orange)

## Overview
Clusters the Netflix catalogue by content and builds a content-based recommender. Textual
attributes (director, cast, country, genre, description) are vectorized and clustered with
both K-Means and hierarchical clustering, then cosine similarity drives "if you liked X,
you may also enjoy…" recommendations.

## Dataset
- **Netflix Movies and TV Shows** — [Kaggle](https://www.kaggle.com/datasets/bansodesandeep/netflix-movies-and-tv-shows)
- ~7,800 titles with `type`, `title`, `director`, `cast`, `country`, `date_added`,
  `release_year`, `rating`, `duration`, `listed_in`, `description`.
- Included in `data/` (`Netflix_movies_and_tv_shows_clustering.csv`).

## Approach
1. **Text preprocessing** — concatenate director + cast + country + genre + description into
   one field; strip non-ASCII; remove stopwords (NLTK) and punctuation; lemmatize; tokenize.
2. **Vectorization** — `TfidfVectorizer` (`max_features=20000`) on the token corpus.
3. **Dimensionality reduction** — `PCA` (cumulative-variance plot) reducing to 4,000 components.
4. **K-Means clustering** — elbow method + silhouette sweep over `k = 2..30`; final fit at
   **k = 6** (`init='k-means++'`, `random_state=33`), with per-cluster word clouds.
5. **Hierarchical clustering** — Ward linkage dendrogram, `AgglomerativeClustering(n_clusters=12)`.
6. **Recommender** — `CountVectorizer` + cosine similarity to return the top-10 most similar titles.

## Results
Metrics extracted from the run log for the K-Means (k = 6) solution.

| Metric | Value |
|---|---|
| K-Means distortion (inertia) | 6375.24 |
| Silhouette score | 0.0088 |

The recommender successfully returns top-10 similar titles for queries such as
"A Man Called God", "Stranger Things", "Peaky Blinders" and "Lucifer" (see log).

## Key Takeaways
- High-dimensional sparse TF-IDF text yields a low silhouette score — content clusters
  overlap heavily, which is expected for free-text catalogue data.
- PCA to 4,000 components keeps clustering tractable on the 20k-feature TF-IDF matrix.
- Cosine similarity on combined metadata gives an effective, lightweight content-based
  recommender without needing user-interaction data.
- Hierarchical (Ward) clustering with 12 clusters offers a finer-grained alternative to the
  6-cluster K-Means view, useful for inspecting niche genre groupings.

## How to Run
```bash
pip install numpy pandas matplotlib seaborn scikit-learn nltk wordcloud beautifulsoup4 scipy
jupyter notebook notebook.ipynb
```
Data is already in `data/Netflix_movies_and_tv_shows_clustering.csv`.

## Credit
> Based on ["🍿🥤Netflix Movies and TV Shows: Clustering"](https://www.kaggle.com/code/bansodesandeep/netflix-movies-and-tv-shows-clustering) by Sandeep Bansode on Kaggle (824 votes).
> Dataset: [Netflix Movies and TV Shows](https://www.kaggle.com/datasets/bansodesandeep/netflix-movies-and-tv-shows).
> Adapted and documented for this portfolio.
