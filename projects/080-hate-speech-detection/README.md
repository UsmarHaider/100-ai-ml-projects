# Hate Speech Detection

![NLP](https://img.shields.io/badge/domain-NLP-blue) ![Task](https://img.shields.io/badge/task-Binary%20Classification-green)

## Overview
Detect hate speech in tweets — a tweet is labeled `1` if it carries racist/sexist sentiment, else `0`. The notebook builds a TF-IDF + SMOTE pipeline and compares four classifiers, then tunes the best ones; F1-score is the target metric (matching the source competition).

## Dataset
- **Twitter Sentiment Analysis (Hatred Speech)**: https://www.kaggle.com/datasets/arkhoshghalb/twitter-sentiment-analysis-hatred-speech
- Train: 31,962 labeled tweets (`id`, `label`, `tweet`); Test: 17,197 tweets (`id`, `tweet`). The classes are highly imbalanced toward non-hate. No missing values or duplicates.
- Included in `data/` (`train.csv`, `test.csv`).

## Approach
1. **EDA** — word clouds for hate vs. non-hate tweets ('user' added to stopwords).
2. **Feature engineering** — tweet length, hashtag/mention/punctuation counts, word count (explored but excluded, as they hurt performance).
3. **Text normalization** — lowercase, strip `@user`, tokenize (NLTK), keep word tokens, Porter stemming.
4. **Vectorization** — `TfidfVectorizer(tokenizer=custom, stop_words='english')` (TF-IDF chosen over Bag-of-Words; n-grams gave no gain).
5. **Class balancing** — **SMOTE** on the training matrix (preferred over random oversampling).
6. **Models** — Logistic Regression, Multinomial Naive Bayes, Random Forest, XGBoost; then hyperparameter tuning of Random Forest and XGBoost.

## Results
Validation accuracy / F1 extracted from the captured run log (20% holdout). The competition metric is F1-score:

| Model | Val accuracy | Val F1 |
| --- | --- | --- |
| Logistic Regression | 0.931 | 0.624 |
| Multinomial Naive Bayes | 0.925 | 0.615 |
| Random Forest (default) | 0.964 | 0.707 |
| XGBoost (default) | 0.952 | 0.639 |
| Random Forest (tuned) | 0.962 | **0.713** |
| XGBoost (tuned) | 0.962 | 0.701 |

The notebook also records public-leaderboard F1 scores in code comments: tuned Random Forest **0.727**, tuned XGBoost 0.692. The final submission uses the tuned Random Forest retrained on the full training set.

## Key Takeaways
- Accuracy is misleading under heavy imbalance — all models score >0.92 accuracy but F1 separates them (Random Forest best at ~0.71).
- SMOTE outperformed random oversampling; TF-IDF outperformed Bag-of-Words here.
- The hand-engineered count features did not help and were dropped — TF-IDF text features carried the signal.
- Random Forest tended to overfit and XGBoost to underfit; tuning narrowed the gap and lifted RF's validation F1.

## How to Run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost nltk wordcloud
jupyter notebook notebook.ipynb
```
Data is already in `data/`.

## Credit
> Based on ["Twitter Sentiment Analysis: NLP, SMOTE"](https://www.kaggle.com/code/pradneshlachake/twitter-sentiment-analysis-nlp-smote) by Pradnesh Lachake on Kaggle (96 votes).
> Dataset: [Twitter Sentiment Analysis (Hatred Speech)](https://www.kaggle.com/datasets/arkhoshghalb/twitter-sentiment-analysis-hatred-speech). Adapted and documented for this portfolio.
