# XGBoost Hyperparameter Tuning Guide (Hyperopt)

![ML](https://img.shields.io/badge/domain-Machine%20Learning-blue)
![Task](https://img.shields.io/badge/task-Classification%20%2B%20HPO-green)

## Overview
A hands-on guide to tuning XGBoost hyperparameters with Bayesian optimization (Hyperopt's
Tree-structured Parzen Estimator). A binary `XGBClassifier` is trained on the Wholesale
Customers dataset, and the notebook searches the key XGBoost knobs to minimize validation loss.

## Dataset
- **Wholesale Customers Data Set** — [Kaggle](https://www.kaggle.com/datasets/binovi/wholesale-customers-data-set)
- 440 rows; clients of a wholesale distributor with annual spend across product categories
  (Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen) plus `Channel` and `Region`.
- Target: `Channel`, remapped to binary `{1, 2} → {1, 0}`.
- Included in `data/` (`Wholesale customers data.csv`).

## Approach
1. **Label prep** — convert the `Channel` target to binary (`2 → 0`, `1 → 1`).
2. **Split** — `train_test_split(test_size=0.3, random_state=0)`.
3. **Search space** (Hyperopt `hp`):
   - `max_depth`: quniform 3–18
   - `gamma`: uniform 1–9
   - `reg_alpha`: quniform 40–180
   - `reg_lambda`: uniform 0–1
   - `colsample_bytree`: uniform 0.5–1
   - `min_child_weight`: quniform 0–10
   - fixed `n_estimators=180`, `seed=0`
4. **Objective** — train `XGBClassifier` with sampled params, evaluate on the test split,
   return validation loss as the value to minimize.
5. **Optimization** — `fmin` with `algo=tpe.suggest`, `max_evals=100`, tracked via `Trials()`;
   print the best hyperparameters found.

## Results
The notebook has no executed cell outputs and the captured run log is empty, so there are
no real metric values to quote. The code computes:

- `accuracy_score` of the XGBoost classifier on the test split
- the **best hyperparameter set** returned by Hyperopt after 100 TPE evaluations

**Run the notebook to reproduce the metrics** (test accuracy and the chosen hyperparameters).

## Key Takeaways
- Hyperopt's TPE explores the XGBoost search space far more efficiently than grid search for
  the same evaluation budget.
- Tuning is framed around the high-impact regularization knobs (`max_depth`, `gamma`,
  `reg_alpha`/`reg_lambda`, `colsample_bytree`, `min_child_weight`) that most affect overfitting.
- The dataset is small (440 rows), so this is best read as a reusable tuning *template* to
  drop onto larger problems.

## How to Run
```bash
pip install numpy pandas scikit-learn xgboost hyperopt
jupyter notebook notebook.ipynb
```
Data is already in `data/Wholesale customers data.csv`.

## Credit
> Based on ["A Guide on XGBoost hyperparameters tuning"](https://www.kaggle.com/code/prashant111/a-guide-on-xgboost-hyperparameters-tuning) by Prashant Banerjee on Kaggle (1,995 votes).
> Dataset: [Wholesale Customers Data Set](https://www.kaggle.com/datasets/binovi/wholesale-customers-data-set).
> Adapted and documented for this portfolio.
