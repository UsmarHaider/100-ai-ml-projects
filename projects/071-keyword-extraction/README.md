# Keyword Extraction from Text

![NLP](https://img.shields.io/badge/domain-NLP-blue) ![Task](https://img.shields.io/badge/task-Keyword%20Extraction-orange)

## Overview
Automatically extract the terms that best describe a document. The notebook applies and compares several unsupervised keyword-extraction techniques on NIPS research papers: statistical TF-IDF, the graph-based TextRank algorithm, and RAKE.

## Dataset
- **NIPS Papers**: https://www.kaggle.com/datasets/benhamner/nips-papers
- `papers.csv` with columns including `id`, `year`, `title`, `event_type`, `pdf_name`, `abstract`, `paper_text`. The work mainly uses `paper_text` (title + abstract + body).
- Data is **not** committed. Download with:
  ```bash
  kaggle datasets download benhamner/nips-papers
  ```
  The notebook reads `/kaggle/input/nips-papers/papers.csv`.

## Approach
1. **Preprocessing** — lowercase, strip tags/special characters/digits, remove stopwords (NLTK + custom list), drop tokens shorter than 3 characters, lemmatize (WordNet).
2. **TF-IDF (scikit-learn)** — `CountVectorizer(max_df=0.95, max_features=10000, ngram_range=(1,3))` → `TfidfTransformer(smooth_idf=True, use_idf=True)`; rank terms by TF-IDF score and return the top-10 per document.
3. **TextRank (gensim)** — `gensim.summarization.keywords` with POS filtering (nouns/adjectives) and lemmatization, on both small example text and full papers.
4. **RAKE (rake-nltk)** — Rapid Automatic Keyword Extraction returning ranked candidate phrases with scores.

## Results
This is an **unsupervised extraction** task with no ground-truth keyword labels, so it reports **no accuracy/F1/RMSE-style metrics**. Output is the ranked top-N keywords/phrases per document from each method, compared qualitatively.

The captured run log for this project is empty. **Run the notebook to reproduce the keyword outputs** for each method (TF-IDF scored terms, TextRank phrases, RAKE ranked phrases). The notebook markdown discusses trade-offs — e.g., TF-IDF surfaces overlapping sub-phrases of the same concept because it ignores context.

## Key Takeaways
- TF-IDF is fast and intuitive but, being context-free, produces redundant near-duplicate keywords (e.g. "matrix factorization", "negative matrix", "nmf").
- TextRank captures the main topical phrases but can miss some salient terms.
- RAKE yields interpretable multi-word phrases with relevance scores; ranking quirks remain.
- No single method dominates; combining or post-filtering candidates improves results.

## How to Run
```bash
pip install numpy pandas nltk scikit-learn gensim rake-nltk
jupyter notebook notebook.ipynb
```
Download the NIPS Papers dataset (see Dataset) and place `papers.csv` under `/kaggle/input/nips-papers/` (or adjust the path).

## Credit
> Based on ["Extract Keywords"](https://www.kaggle.com/code/akhatova/extract-keywords) by Alfiia Akhatova on Kaggle (183 votes).
> Dataset: [NIPS Papers](https://www.kaggle.com/datasets/benhamner/nips-papers). Adapted and documented for this portfolio.
