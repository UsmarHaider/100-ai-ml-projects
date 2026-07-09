# Amazon Product Reviews Sentiment Analysis

![NLP](https://img.shields.io/badge/domain-NLP-blue) ![Task](https://img.shields.io/badge/task-Sentiment%20Classification-green)

## Overview
Three-class sentiment analysis (Negative / Neutral / Positive) on Amazon musical-instrument reviews. Star ratings are mapped to sentiment labels, text is cleaned and TF-IDF vectorized, the class imbalance is fixed with SMOTE, and several classifiers are compared.

## Dataset
- [Amazon Music Reviews](https://www.kaggle.com/datasets/eswarchandt/amazon-music-reviews) (`eswarchandt/amazon-music-reviews`, ~13 MB).
- **Included in `data/`** as `Musical_instruments_reviews.csv` (and `Musical_Instruments_5.json`). 10,261 reviews × 9 columns; key fields `reviewText`, `summary`, `overall`.

## Approach
1. **Labeling** (`overall` → `sentiment`): rating 3 → Neutral, ratings 1–2 → Negative, ratings 4–5 → Positive.
2. **EDA**: helpfulness vs. sentiment (violin plot), sentiment-by-year trend, TextBlob `polarity` distribution, per-class WordClouds.
3. **Preprocessing**: lowercase, remove stopwords (`wordcloud.STOPWORDS`), Porter stemming.
4. **Vectorization & balancing**: `TfidfVectorizer`, then `SMOTE` to oversample minority classes — original `{Positive: 9022, Neutral: 772, Negative: 467}` resampled to `{9022, 9022, 9022}`.
5. **Models compared**: Logistic Regression, Decision Tree, KNN, SVC, Naive Bayes; final tuned `LogisticRegression(C=10000, random_state=0)`.

## Results
Real values from the run log.

Initial model comparison (test accuracy):

| Model | Test Accuracy |
|-------|--------------|
| Logistic Regression | 0.881 |
| SVC | 0.880 |
| KNN | 0.872 |
| Decision Tree | 0.819 |
| Naive Bayes | 0.803 |

Final tuned Logistic Regression (`C=10000`): **mean accuracy 0.942**, test-set accuracy **0.94**.

Classification report (test set, n=6767):

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|------|---------|
| 0 (Negative) | 0.93 | 1.00 | 0.96 | 2326 |
| 1 (Neutral) | 0.91 | 0.98 | 0.95 | 2232 |
| 2 (Positive) | 1.00 | 0.84 | 0.91 | 2209 |
| **accuracy** | | | **0.94** | 6767 |
| macro avg | 0.95 | 0.94 | 0.94 | 6767 |
| weighted avg | 0.95 | 0.94 | 0.94 | 6767 |

## Key Takeaways
- The raw data is heavily skewed toward positive reviews; SMOTE oversampling is what makes the minority Negative/Neutral classes learnable.
- A well-regularized Logistic Regression (`C=10000`) beats the tree/KNN/Naive Bayes baselines and ties/edges out SVC.
- The Positive class shows perfect precision (1.00) but lower recall (0.84) — the model is conservative about predicting Positive after rebalancing.
- Reported accuracy is measured on the SMOTE-balanced split; real-world skewed traffic would warrant evaluating on the original imbalanced distribution too.

## How to Run
```bash
pip install scikit-learn imbalanced-learn textblob nltk wordcloud plotly seaborn matplotlib pandas numpy
jupyter notebook notebook.ipynb
```
Point the read path at the local `data/Musical_instruments_reviews.csv`.

## Credit
> Based on ["Sentiment Analysis | Amazon reviews"](https://www.kaggle.com/code/benroshan/sentiment-analysis-amazon-reviews) by Ben Roshan on Kaggle (567 votes).
> Dataset: [Amazon Music Reviews](https://www.kaggle.com/datasets/eswarchandt/amazon-music-reviews). Adapted and documented for this portfolio.
