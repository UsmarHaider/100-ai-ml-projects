# Mental Health Text Classification

![NLP](https://img.shields.io/badge/domain-NLP-blue) ![Task](https://img.shields.io/badge/task-Multiclass%20Classification-green)

## Overview
Classify a person's written statement into one of seven mental-health categories — Normal, Depression, Suicidal, Anxiety, Stress, Bipolar, Personality disorder. The notebook builds a classic NLP pipeline (TF-IDF + engineered features, class rebalancing) and compares four machine-learning classifiers.

## Dataset
- **Sentiment Analysis for Mental Health**: https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health
- `Combined Data.csv` with `statement` and `status` columns; ~53,000 rows (52,681 with non-null statements). The target is **imbalanced** (Normal/Depression/Suicidal dominate).
- Data is **not** committed. Download with:
  ```bash
  kaggle datasets download suchintikasarkar/sentiment-analysis-for-mental-health
  ```
  The notebook reads `/kaggle/input/sentiment-analysis-for-mental-health/Combined Data.csv`.

## Approach
1. **Cleaning** — drop missing rows; lowercase; remove URLs, markdown links, @handles, punctuation/special characters.
2. **Feature engineering** — `num_of_characters` and `num_of_sentences` per statement.
3. **Tokenize & stem** — NLTK `word_tokenize` + `PorterStemmer` (stopwords intentionally kept, as negations/emotion words carry signal).
4. **Vectorization** — `TfidfVectorizer(ngram_range=(1,2), max_features=50000)` on stemmed tokens, combined (via sparse `hstack`) with the numeric features.
5. **Rebalancing** — `RandomOverSampler` on the training set (chosen over SMOTE / undersampling).
6. **Models (hyperparameters tuned with GridSearchCV):** Bernoulli Naive Bayes (`alpha=0.1`), Decision Tree (`max_depth=9`), Logistic Regression (`liblinear`, `penalty='l1'`, `C=10`), XGBoost (`learning_rate=0.2`, `max_depth=7`, `n_estimators=500`).

## Results
Test-set accuracy extracted from the captured run log (80/20 split, 10,537 test rows):

| Model | Accuracy |
| --- | --- |
| Bernoulli Naive Bayes | 0.641 |
| Decision Tree | 0.618 |
| Logistic Regression | 0.765 |
| **XGBoost** | **0.808** |

XGBoost per-class F1 (from the log): Normal 0.93, Bipolar 0.85, Anxiety 0.84, Depression 0.75, Personality disorder 0.75, Stress 0.72, Suicidal 0.71 (weighted-avg F1 ≈ 0.81).

## Key Takeaways
- XGBoost on TF-IDF + simple length features is the clear winner (~81% accuracy).
- The `Normal` class is easiest (F1 0.93); `Depression` and `Suicidal` overlap heavily in vocabulary, the hardest pair to separate.
- Keeping stopwords and random oversampling both improved results on this imbalanced, emotion-laden corpus.

## How to Run
```bash
pip install pandas numpy matplotlib seaborn nltk scikit-learn imbalanced-learn xgboost wordcloud scipy
jupyter notebook notebook.ipynb
```
Download the dataset (see Dataset) and place `Combined Data.csv` under `/kaggle/input/sentiment-analysis-for-mental-health/` (or adjust the path).

## Credit
> Based on ["Mental Health Sentiment Analysis | NLP | ML"](https://www.kaggle.com/code/annastasy/mental-health-sentiment-analysis-nlp-ml) by Anna Balatska on Kaggle (385 votes).
> Dataset: [Sentiment Analysis for Mental Health](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health). Adapted and documented for this portfolio.
