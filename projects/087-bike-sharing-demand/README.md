# Bike Sharing Demand — EDA & Regression (ML + DL Ensemble)

![ML](https://img.shields.io/badge/domain-Machine%20Learning-blue)
![Task](https://img.shields.io/badge/task-Regression-green)

## Overview
Predicts the hourly count of rental bikes from calendar and weather features in the Kaggle
**Bike Sharing Demand** competition. The notebook combines exploratory analysis with both a
deep neural network and tree-based regressors, then averages them into an ensemble submission.

## Dataset
- **Bike Sharing Demand** — [Kaggle competition](https://www.kaggle.com/competitions/bike-sharing-demand)
- Training set ~10,886 rows; test set 6,493 rows (confirmed in the run log).
- Features: `datetime`, `season`, `holiday`, `workingday`, `weather`, `temp`, `atemp`,
  `humidity`, `windspeed`; target `count` (with `casual` + `registered`).
- **Not committed** — download with the Kaggle CLI:
  ```bash
  kaggle competitions download -c bike-sharing-demand
  ```

## Approach
1. **EDA** — missing-value check (`missingno`), feature distributions, datetime parsing into
   hour / day / month / year components.
2. **Deep learning model** — Keras `Sequential` MLP: Dense(32) → Dense(64) → Dense(128) →
   Dense(256) → Dense(64) → output, ReLU activations, trained on the engineered features.
3. **Machine learning models** — compared `CatBoostRegressor`, `XGBRegressor`
   (`n_estimators=100`, `max_depth=5`) and `RandomForestRegressor` via 10-fold `KFold`.
   Final ML model: `RandomForestRegressor(n_estimators=300, n_jobs=-1)` after sweeping
   `n_estimators ∈ {10…300}`.
4. **Ensemble** — average the DL and RF predictions to form the final `count` submission.

## Results
Metrics quoted from the run log. The notebook reports the
training fit (R²) and the Kaggle leaderboard RMSLE-style scores for each submission type.

| Metric | Value |
|---|---|
| RandomForest train R² (`relation_square`) | up to ~0.9927 |
| Kaggle score — **Merge (ML+DL avg)** | **0.46969** |
| Kaggle score — Machine (RF) | 0.48616 |
| Kaggle score — Deep (MLP) | 0.55899 |

Lower competition score is better; the averaged ML+DL ensemble (0.46969) beat either model alone.

## Key Takeaways
- Averaging a deep MLP with a random forest produced a better leaderboard score than either
  individual model — a simple, effective blend.
- Random forest accuracy rises with `n_estimators` but with diminishing returns; 300 trees
  was the chosen trade-off.
- The very high train R² (~0.99) signals the tree model can overfit the training fit, so the
  held-out competition score is the metric that matters.

## How to Run
```bash
pip install numpy pandas matplotlib seaborn missingno tensorflow scikit-learn xgboost catboost
jupyter notebook notebook.ipynb
```
Download the competition data into the notebook's input path first (command above).

## Credit
> Based on ["🚴‍♀️Bike Sharing EDA📈&Regression Model Predict💡"](https://www.kaggle.com/code/kagleo123/bike-sharing-eda-regression-model-predict) by wngus_ on Kaggle (144 votes).
> Dataset: [Bike Sharing Demand competition](https://www.kaggle.com/competitions/bike-sharing-demand).
> Adapted and documented for this portfolio.
