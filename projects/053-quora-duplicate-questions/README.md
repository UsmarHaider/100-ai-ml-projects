# Quora Duplicate Question Detection

![NLP](https://img.shields.io/badge/domain-NLP-blue) ![Task](https://img.shields.io/badge/task-Binary%20Classification-green)

## Overview
Quora receives many questions that are semantically duplicates of ones already asked. The goal is to predict, for a pair of questions, whether the two are duplicates. This notebook is an exploratory data analysis plus an XGBoost baseline built on hand-crafted text-overlap features, scored against log loss.

## Dataset
- **Quora Question Pairs** competition: https://www.kaggle.com/competitions/quora-question-pairs
- Training set: ~404K question pairs with fields `id`, `qid1`, `qid2`, `question1`, `question2`, `is_duplicate`. ~37% of training pairs are duplicates; the test set (~2.3M rows, many auto-generated) has roughly 16.5% positives.
- Data is **not** committed here. Download via the competition page (requires accepting the rules):
  ```bash
  kaggle competitions download -c quora-question-pairs
  ```
  Place `train.csv` and `test.csv` where the notebook expects them (`../input/`).

## Approach
1. **EDA** — question-frequency histograms, character/word-count distributions, word cloud, and punctuation/capitalization statistics over the question text.
2. **Naive baseline** — submit the mean duplicate rate as a constant probability to calibrate against the leaderboard distribution.
3. **Feature engineering**
   - `word_match_share`: fraction of shared non-stopword tokens between the two questions.
   - `tfidf_word_match_share`: the same overlap weighted by custom inverse-frequency weights (smoothed, rare words ignored).
4. **Class rebalancing** — oversample the negative class so the training positive rate (~37%) matches the test rate (~17%), aligning XGBoost's probability outputs with the log-loss leaderboard metric.
5. **Model** — XGBoost on the two overlap features.

## Results
The kernel title reports the public leaderboard score for the XGBoost submission.

| Metric | Value |
| --- | --- |
| Public LB log loss (XGBoost) | **0.35460** |
| Naive constant-probability submission | ~0.55 LB |

The notebook also computes ROC AUC of the individual `word_match` and `tfidf_word_match` features. The captured run log contains only nbconvert messages (no metric stdout), so the numbers above come from the kernel title and notebook markdown. Run the notebook to reproduce the per-feature AUC values.

## Key Takeaways
- Simple lexical-overlap features already carry meaningful signal for duplicate detection.
- The train/test class-balance mismatch matters a lot for a log-loss metric — rebalancing is what aligns predictions with the leaderboard.
- TF-IDF weighting did not improve overall AUC over raw word overlap in this baseline, but adds complementary information for the model.
- Next steps: richer semantic features (embeddings, fuzzy/edit-distance, shared n-grams) and a deeper model.

## How to Run
```bash
pip install numpy pandas matplotlib seaborn scikit-learn nltk wordcloud xgboost
jupyter notebook notebook.ipynb
```
Download the competition data (see Dataset) and place `train.csv`/`test.csv` under `../input/`.

## Credit
> Based on ["Data Analysis & XGBoost Starter (0.35460 LB)"](https://www.kaggle.com/code/anokas/data-analysis-xgboost-starter-0-35460-lb) by anokas on Kaggle (3251 votes).
> Dataset: [Quora Question Pairs competition](https://www.kaggle.com/competitions/quora-question-pairs). Adapted and documented for this portfolio.
