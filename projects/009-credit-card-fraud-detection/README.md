# Credit Card Fraud Detection

![Domain](https://img.shields.io/badge/domain-Machine%20Learning-blue)
![Task](https://img.shields.io/badge/task-Imbalanced%20Classification-red)

## Overview
Detects fraudulent credit-card transactions on a highly imbalanced dataset (~0.17% fraud).
The notebook is a deep dive into handling class imbalance correctly: it contrasts Random
Under-Sampling and SMOTE oversampling, stresses scaling/splitting before resampling to
avoid leakage, and compares multiple classifiers plus a small neural network.

## Dataset
- **Credit Card Fraud Detection** — [Kaggle dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (`mlg-ulb/creditcardfraud`, ~144 MB)
- 284,807 transactions; features `V1`–`V28` (PCA-anonymized) plus `Time`, `Amount`; binary
  label `Class` (1 = fraud). Frauds are ~0.17% of rows.
- Data is **not** committed. Download with:
  ```bash
  kaggle datasets download mlg-ulb/creditcardfraud
  ```
  The notebook reads `../input/creditcard.csv`.

## Approach
1. **Scaling** — `RobustScaler` on `Amount` and `Time` (less sensitive to outliers); drop
   the originals.
2. **Stratified split** — `StratifiedKFold` to carve out an original train/test set that
   preserves the fraud ratio (used as the true held-out test).
3. **Random Under-Sampling** — build a 50/50 balanced subset; correlation analysis and
   IQR-based extreme-outlier removal on the most fraud-correlated features.
4. **Classifier comparison** — Logistic Regression, KNN, SVC, Decision Tree; tuned with
   `GridSearchCV`; scored via `cross_val_score` and `roc_auc_score` / ROC curves.
5. **SMOTE oversampling** — `imblearn` `SMOTE` inside a cross-validation pipeline
   (`RandomizedSearchCV`), evaluated on the untouched original test set.
6. **Neural network** — a small Keras `Sequential` model compared under both under-sampling
   and over-sampling to illustrate the effect on the minority class.

## Results
This notebook computes accuracy, **ROC-AUC**, precision/recall/F1, and confusion matrices
across several classifiers and resampling strategies. The run log for this project is
empty, so no executed metric values were captured here.

| Metric | Where reported |
|---|---|
| Cross-val accuracy (per classifier) | `cross_val_score(...).mean()` |
| ROC-AUC | `roc_auc_score(...)` for LR / KNN / SVC / Decision Tree |
| Precision / Recall / F1 | `classification_report`, `classification_report_imbalanced` |
| NN comparison | under-sample vs SMOTE confusion matrices |

Run the notebook to reproduce the AUC scores and per-model classification reports.

## Key Takeaways
- On imbalanced data, accuracy is misleading — **ROC-AUC / recall on the fraud class** are
  the metrics that matter.
- Scale and split **before** resampling; SMOTE must be applied only inside cross-validation
  folds to avoid leakage into the validation data.
- Random Under-Sampling discards data and can overfit; SMOTE evaluated on the original test
  set gives a more realistic picture of fraud-detection performance.

## How to Run
```bash
pip install numpy pandas scikit-learn imbalanced-learn tensorflow matplotlib seaborn
jupyter notebook notebook.ipynb
```
Download the dataset (see Dataset) and place `creditcard.csv` at `../input/creditcard.csv`.

## Credit
> Based on ["Credit Fraud || Dealing with Imbalanced Datasets"](https://www.kaggle.com/code/janiobachmann/credit-fraud-dealing-with-imbalanced-datasets) by Janio Martinez Bachmann on Kaggle (15064 votes).
> Dataset: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). Adapted and documented for this portfolio.
