# Spam SMS Detection (NLP for Beginners)

![NLP](https://img.shields.io/badge/domain-NLP-green)
![Task](https://img.shields.io/badge/task-text%20classification-orange)

## Overview
Classify SMS messages as **spam** or **ham** (legitimate) from their text. The notebook is a beginner-friendly tour of text vectorization (Bag-of-Words, TF-IDF) and compares Multinomial Naive Bayes against Logistic Regression. Spam filtering is a canonical, high-value text-classification task.

## Dataset
- **SMS Spam Collection** — [kaggle.com/datasets/uciml/sms-spam-collection-dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- 5,572 labeled SMS messages: **4,825 ham** and **747 spam**.
- **Included in `data/spam.csv`** (latin-1 encoded; the notebook also references `/kaggle/input/sms-spam-collection-dataset/spam.csv`).

## Approach
1. **Load & clean** — read `spam.csv`, drop empty columns, rename to `label`/`message`, map `ham/spam` → `0/1`.
2. **EDA** — message-length feature shows spam messages tend to be longer; word-frequency counts per class.
3. **Text preprocessing** — `text_process()` removes punctuation and stopwords (NLTK English list extended with chat tokens like `u`, `ur`, `im`, `dont`).
4. **Vectorization** — `CountVectorizer` (Bag-of-Words) → optional `TfidfTransformer` (TF-IDF weighting); train/test split via `train_test_split`.
5. **Models**
   - **Multinomial Naive Bayes** on the document-term matrix, and as a `Pipeline(CountVectorizer → TfidfTransformer → MultinomialNB)`.
   - **Logistic Regression** (`solver='liblinear'`) on the same features.
6. **Evaluation** — accuracy, confusion matrix, and ROC-AUC; closes with a discussion of tuning `CountVectorizer` (stop_words, ngram_range, min_df, max_df).

## Results
Real values extracted from the recorded run:

| Model | Accuracy | ROC-AUC |
| --- | --- | --- |
| Multinomial NB (CountVectorizer) | **0.9828** | — |
| NB Pipeline (Count → TF-IDF → NB) | **0.9670** | — |
| Logistic Regression | **0.9842** | **0.9836** |

Logistic Regression on Bag-of-Words counts is the top performer here at ~98.4% accuracy.

## Key Takeaways
- Simple linear/Bayesian models on Bag-of-Words counts already exceed 98% accuracy on this dataset.
- Applying TF-IDF on top of counts slightly **lowered** NB accuracy here (0.983 → 0.967), a reminder that TF-IDF is not always a win.
- Message length is a genuinely discriminative feature — spam messages skew longer.
- The vectorizer itself (stopwords, n-grams, min_df/max_df) is worth tuning alongside the model.

## How to Run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk
python -c "import nltk; nltk.download('stopwords')"
jupyter notebook notebook.ipynb
```
The dataset is already in `data/spam.csv`.

## Credit
> Based on ["Natural Language Processing (NLP) for Beginners"](https://www.kaggle.com/code/faressayah/natural-language-processing-nlp-for-beginners) by Fares Sayah on Kaggle (1,419 votes).
> Dataset: [SMS Spam Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset). Adapted and documented for this portfolio.
