# Credit Risk Modeling — Home Credit Default Risk

![ML](https://img.shields.io/badge/domain-Machine%20Learning-blue)
![Classification](https://img.shields.io/badge/task-Binary%20Classification-green)

## Overview
A gentle, end-to-end introduction to the **Home Credit Default Risk**
competition: predict the probability that a loan applicant will default. The
notebook covers EDA, handling of imbalanced classes, feature engineering, and a
Logistic Regression / Random Forest baseline scored by **ROC AUC**.

## Dataset
- **Home Credit Default Risk** competition — [kaggle.com/competitions/home-credit-default-risk](https://www.kaggle.com/competitions/home-credit-default-risk)
- Main files: `application_train.csv` (with `TARGET`) and
  `application_test.csv`; plus 6 auxiliary tables and a column-description file.
- Not committed to this repo. Accept the competition rules, then:
  ```bash
  kaggle competitions download -c home-credit-default-risk
  ```

## Approach
1. **EDA**: target imbalance (most loans repaid), missing-value audit, column
   typing, anomaly handling (e.g. `DAYS_EMPLOYED` sentinel values).
2. **Encoding**: `LabelEncoder` for 2-category columns, one-hot encoding for
   the rest; `align(join='inner', axis=1)` to keep train/test columns matched.
3. **Correlation analysis**: `EXT_SOURCE_1/2/3` and `DAYS_BIRTH` (age) are the
   strongest signals.
4. **Feature engineering**:
   - **Polynomial features** on the `EXT_SOURCE` columns + `DAYS_BIRTH`.
   - **Domain features** (income/credit/annuity ratios, inspired by Aguiar).
5. **Preprocessing**: median `Imputer` + `MinMaxScaler` to [0, 1].
6. **Models**:
   - **Logistic Regression** (`C = 0.0001`), `predict_proba` for AUC.
   - **Random Forest** (`n_estimators=100`, `random_state=50`).
   - Both also retrained on the engineered feature set for comparison.
7. **Submission**: `SK_ID_CURR` + predicted `TARGET` probability.

## Results
No run log was captured for this project (no run output was captured).
The notebook's markdown states the expected competition leaderboard scores
(ROC AUC) for the baselines:

| Model | Public LB ROC AUC |
|-------|-------------------|
| Logistic Regression (C=0.0001) | ~0.671 |
| Random Forest (100 trees) | ~0.678 |
| Random guess (reference) | 0.500 |

Run the notebook and submit to the competition to reproduce these AUC scores.

## Key Takeaways
- With imbalanced classes, **ROC AUC** is the right metric — accuracy is
  misleading (always predicting "no default" scores ~92% accuracy but is
  useless).
- The three `EXT_SOURCE` external-credit scores are the most predictive
  features; engineering polynomial and ratio features off them helps.
- Random Forest edges out Logistic Regression; gradient boosting (LightGBM)
  is the natural next step.

## How to Run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
jupyter notebook notebook.ipynb
```
Note: the notebook uses the legacy `sklearn.preprocessing.Imputer`; on modern
scikit-learn replace it with `sklearn.impute.SimpleImputer`. Download the
competition data (command above) into the path the notebook expects
(`../input/`).

## Credit
> Based on ["Start Here: A Gentle Introduction"](https://www.kaggle.com/code/willkoehrsen/start-here-a-gentle-introduction)
> by Will Koehrsen on Kaggle (9,237 votes).
> Competition: [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk).
> Adapted and documented for this portfolio.
