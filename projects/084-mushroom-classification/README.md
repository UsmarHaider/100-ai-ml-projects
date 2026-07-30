# Mushroom Edibility Classification — XGBoost

![ML](https://img.shields.io/badge/domain-Machine%20Learning-blue)
![Task](https://img.shields.io/badge/task-Binary%20Classification-green)

## Overview
Classifies mushrooms as **edible** or **poisonous** from their physical characteristics.
Built on the large Kaggle Playground Series (S4E8) synthetic mushroom dataset, the pipeline
handles heavy missing data, encodes high-cardinality categoricals, and trains a gradient-boosted
tree model evaluated with Matthews Correlation Coefficient (MCC).

## Dataset
- **Mushroom Classification** — [Kaggle dataset](https://www.kaggle.com/datasets/uciml/mushroom-classification)
  and the [Playground Series S4E8 competition](https://www.kaggle.com/competitions/playground-series-s4e8).
- Training frame: **3,116,945 rows × 21 columns** (per run log). Target `class` is binary
  (edible / poisonous). 3 numeric features (`cap-diameter`, `stem-height`, `stem-width`)
  and 18 categorical features.
- Several columns have very high missingness — e.g. `veil-type` ~94.9%, `spore-print-color`
  ~91.4%, `veil-color` ~87.9%, `stem-root` ~88.5%, `stem-surface` ~63.6%, `gill-spacing` ~40.4%.
- Included in `data/` (`mushrooms.csv`).

## Approach
1. **Missing-data handling** — drop columns above a 0.95 missingness threshold; impute the rest.
2. **Encoding** — `LabelEncoder` / `OrdinalEncoder` and `category_encoders` for categorical
   variables; correlation analysis via the `dython` library.
3. **Split** — stratified `train_test_split` (test_size=0.2, random_state=42, `stratify=y`).
4. **Model** — `XGBClassifier` with key hyperparameters:
   `objective='binary:logistic'`, `n_estimators=100`, `max_depth=14`, `min_child_weight=7`,
   `alpha=0.1`, `gamma=1e-6`, `subsample=0.8`, `colsample_bytree=0.6`. Trained with logloss +
   MCC eval metrics on a validation set.
5. **Evaluation** — Matthews Correlation Coefficient on the test split.
6. **Explainability** — LIME (`LimeTabularExplainer`) on individual predictions.

## Results
Metrics extracted from the run log.

| Metric | Value |
|---|---|
| Test MCC (final) | **0.9828** |
| Best validation MCC (during training) | ~0.9829 (round 44) |
| Validation logloss (best) | ~0.0386 |

The model reaches an MCC of ~0.983 — near-perfect agreement between predicted and true labels.

## Key Takeaways
- A deep XGBoost (`max_depth=14`) handles the mixed numeric/high-cardinality-categorical
  feature space very well, hitting MCC ≈ 0.983.
- Aggressive missing-column dropping (>95%) plus encoding the rest is sufficient — the
  poison/edible signal survives heavy imputation.
- Validation MCC plateaus around round ~40, so fewer estimators or early stopping would
  train faster with no accuracy loss.
- LIME adds per-prediction transparency, valuable for a safety-critical edible/poisonous call.

## How to Run
```bash
pip install numpy pandas matplotlib scikit-learn xgboost lightgbm category_encoders dython lime
jupyter notebook notebook.ipynb
```
Data is already in `data/mushrooms.csv` (the competition CSVs can be pulled with
`kaggle competitions download -c playground-series-s4e8`).

## Credit
> Based on ["🍄Mushroom🎉Classification📈Analysis"](https://www.kaggle.com/code/satyaprakashshukl/mushroom-classification-analysis) by Satya on Kaggle (484 votes).
> Dataset: [Mushroom Classification](https://www.kaggle.com/datasets/uciml/mushroom-classification) ·
> [Playground Series S4E8](https://www.kaggle.com/competitions/playground-series-s4e8).
> Adapted and documented for this portfolio.
