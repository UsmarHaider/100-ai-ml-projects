# Toxic Comment Classification (NB-SVM Baseline)

![NLP](https://img.shields.io/badge/domain-NLP-green)
![Task](https://img.shields.io/badge/task-multi--label%20classification-orange)

## Overview
Multi-label classification of Wikipedia comments across six toxicity types: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`. The model is **NB-SVM** — Naive Bayes log-count ratios used as features for a logistic-regression (SVM-equivalent) classifier — a famously strong, fast linear baseline introduced by Wang & Manning.

## Dataset
- **Competition:** [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge)
- `train.csv` (comment text + 6 binary labels), `test.csv`, `sample_submission.csv`. A comment may carry zero, one, or several labels; the notebook also derives a `none` indicator.
- Data is **not** committed. Download via the competition:
  ```bash
  kaggle competitions download -c jigsaw-toxic-comment-classification-challenge
  unzip jigsaw-toxic-comment-classification-challenge.zip
  ```
  The notebook reads from `../input/`.

## Approach
1. **Clean** — fill empty comments with `"unknown"`.
2. **Tokenize** — regex tokenizer that separates punctuation/special characters.
3. **Vectorize** — `TfidfVectorizer(ngram_range=(1,2), min_df=3, max_df=0.9, strip_accents='unicode', sublinear_tf=1, smooth_idf=1, use_idf=1)` → sparse term-document matrix. (The author notes TF-IDF beats binarized features, improving leaderboard log-loss from 0.59 to 0.55.)
4. **NB-SVM** — for each label, compute the Naive Bayes log-count ratio `r = log(pr(1)/pr(0))`, multiply features by `r`, then fit `LogisticRegression(C=4, dual=True)`.
5. **Predict** — one model per label; `predict_proba` fills a `(n_test, 6)` matrix; write `submission.csv`.

## Results
This is a per-label probability model scored by **mean column-wise ROC-AUC** on the competition leaderboard. The notebook produces a submission file rather than printing a local metric, and the run log contains no executed outputs.

| Metric | Value |
| --- | --- |
| Leaderboard log-loss (author's note) | ~0.55 with TF-IDF (vs. 0.59 binarized) |
| Local validation metric | Run the notebook to reproduce |

Run the notebook to reproduce metrics / generate the submission.

## Key Takeaways
- NB-SVM is a remarkably strong linear baseline — fast, interpretable, and competitive on this task.
- Multiplying TF-IDF features by Naive Bayes log-count ratios injects class priors directly into the feature space.
- Multi-label is handled as six independent binary one-vs-rest logistic regressions.
- Switching from binarized features to TF-IDF measurably improved the score.

## How to Run
```bash
pip install pandas numpy scikit-learn
jupyter notebook notebook.ipynb
```
Download the Jigsaw competition data (see Dataset) so `train.csv`/`test.csv`/`sample_submission.csv` resolve under `../input/`.

## Credit
> Based on ["NB-SVM strong linear baseline"](https://www.kaggle.com/code/jhoward/nb-svm-strong-linear-baseline) by Jeremy Howard on Kaggle (3,108 votes).
> Competition: [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge). Adapted and documented for this portfolio.
