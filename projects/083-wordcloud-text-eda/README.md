# Text EDA and Word Clouds

![Domain](https://img.shields.io/badge/domain-NLP-blue)
![Task](https://img.shields.io/badge/task-EDA%20%2B%20Topic%20Modelling-green)

## Overview
An exploratory text-analysis walkthrough on the *Spooky Author Identification* corpus
(short prose excerpts by Edgar Allan Poe, Mary Shelley and HP Lovecraft). It covers
text EDA with frequency plots and word clouds, classic NLP preprocessing, and two
unsupervised topic-modelling techniques (LDA and NMF) applied to the raw text.

## Dataset
- **Spooky Author Identification** — [Kaggle competition](https://www.kaggle.com/competitions/spooky-author-identification)
- `train.csv`: text excerpts labelled by author initials (`EAP`, `HPL`, `MWS`).
- Data is **not** committed here. Download via the competition page (requires accepting
  the competition rules):
  ```bash
  kaggle competitions download -c spooky-author-identification
  ```
  Place `train.csv` where the notebook expects it (`../input/train.csv`).

## Approach
1. **EDA** — load `train.csv`, count word frequencies per author, render word clouds
   using image masks (`wordcloud` + `STOPWORDS`).
2. **Preprocessing** — tokenization, NLTK English stop-word removal (153 stopwords),
   and `WordNetLemmatizer` lemmatization via a custom `LemmaCountVectorizer`.
3. **Vectorization** — Bag-of-Words term frequencies with
   `CountVectorizer(max_df=0.95, min_df=2, stop_words='english')`; TF-IDF discussed.
4. **Topic modelling**
   - **LDA**: `LatentDirichletAllocation(n_components=11, max_iter=5, learning_method='online', learning_offset=50.0, random_state=0)`.
   - **NMF**: `NMF` over the TF-IDF matrix for comparison.
   Top words per topic are printed for interpretation.

## Results
This notebook is an exploratory/topic-modelling tutorial; it produces word clouds and
per-topic top-word lists rather than a single predictive score. The run log for this
project is empty, so no executed metrics were captured.

| Output | Description |
|---|---|
| Word frequency plots | Most common terms per author |
| Word clouds | Masked clouds per author |
| LDA topics | 11 topics with top words |
| NMF topics | Topic decomposition of TF-IDF matrix |

Run the notebook to reproduce the topic-word tables and visualizations.

## Key Takeaways
- LDA and NMF surface broadly thematic word groupings even without labels.
- Lemmatization plus aggressive stop-word/`max_df` filtering noticeably cleans the
  vocabulary before vectorization.
- A natural extension is to use these topic features (or TF-IDF directly) as inputs to a
  supervised author classifier.

## How to Run
```bash
pip install numpy pandas scikit-learn nltk wordcloud matplotlib plotly scipy
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"
jupyter notebook notebook.ipynb
```
Download the competition data (see Dataset) and place `train.csv` at `../input/train.csv`.

## Credit
> Based on ["Spooky NLP and Topic Modelling tutorial"](https://www.kaggle.com/code/arthurtok/spooky-nlp-and-topic-modelling-tutorial) by Anisotropic on Kaggle (2459 votes).
> Dataset: [Spooky Author Identification](https://www.kaggle.com/competitions/spooky-author-identification). Adapted and documented for this portfolio.
