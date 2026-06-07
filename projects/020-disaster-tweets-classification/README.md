# Disaster Tweets Classification

![NLP](https://img.shields.io/badge/domain-NLP-green)
![Task](https://img.shields.io/badge/task-binary%20text%20classification-orange)

## Overview
Decide whether a tweet is about a **real disaster** (`target=1`) or not (`target=0`). The notebook is the official "Getting Started" tutorial for the NLP competition: count-vectorize the tweet text and fit a linear Ridge classifier — a minimal, transparent baseline you can iterate on.

## Dataset
- **Competition:** [Natural Language Processing with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started)
- `train.csv` (tweet `text` + binary `target`), `test.csv`, `sample_submission.csv`.
- Data is **not** committed. Download via the competition:
  ```bash
  kaggle competitions download -c nlp-getting-started
  unzip nlp-getting-started.zip
  ```
  The notebook reads from `/kaggle/input/nlp-getting-started/`.

## Approach
1. **Load** — read `train.csv` and `test.csv`; inspect example disaster vs. non-disaster tweets.
2. **Vectorize** — `CountVectorizer` builds word-count vectors. `fit_transform` on train, `transform` on test so both use the same vocabulary.
3. **Model** — `linear_model.RidgeClassifier` (ridge-regularized linear classifier — pushes weights toward 0 without dropping words).
4. **Validate** — 3-fold `cross_val_score` with `scoring="f1"` (F1 is the competition metric).
5. **Predict & submit** — fit on full training data, predict on the test vectors, write `submission.csv`.

## Results
The competition is scored by **F1**. The notebook computes 3-fold cross-validated F1 (`scores`) but the run log is empty (no executed outputs). The author's markdown estimates the baseline scores **roughly 0.65** F1 on the leaderboard.

| Metric | Value |
| --- | --- |
| Cross-validated F1 (3-fold) | Run the notebook to reproduce |
| Estimated leaderboard F1 (author's note) | ~0.65 |

Run the notebook to reproduce the metrics.

## Key Takeaways
- A bare `CountVectorizer` + `RidgeClassifier` already reaches ~0.65 F1 — a solid floor to beat.
- Using `transform` (not `fit_transform`) on the test set keeps train/test vocabularies aligned.
- The competition metric is F1, so cross-validation is scored with F1 directly.
- Clear next steps to improve: TF-IDF weighting, LSA, or RNN/transformer models.

## How to Run
```bash
pip install numpy pandas scikit-learn
jupyter notebook notebook.ipynb
```
Download the competition data (see Dataset) so the `/kaggle/input/nlp-getting-started/` paths resolve (or edit the paths to a local folder).

## Credit
> Based on ["NLP Getting Started Tutorial"](https://www.kaggle.com/code/philculliton/nlp-getting-started-tutorial) by Phil Culliton on Kaggle (5,023 votes).
> Competition: [Natural Language Processing with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started). Adapted and documented for this portfolio.
