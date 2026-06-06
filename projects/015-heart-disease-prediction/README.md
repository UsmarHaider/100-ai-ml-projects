# Heart Disease Prediction

![ML](https://img.shields.io/badge/domain-Machine%20Learning-blue) ![Task](https://img.shields.io/badge/task-Binary%20Classification-green)

## Overview
Cardiovascular disease is the leading cause of death globally, and early detection enables timely intervention. This project performs exploratory data analysis on a clinical heart-disease dataset and trains several classifiers to predict whether a patient has heart disease from 13 routine clinical measurements.

## Dataset
- [Heart Disease](https://www.kaggle.com/datasets/data855/heart-disease) (ref `data855/heart-disease`).
- 303 records, 14 columns (13 features + binary `target`). Features include age, sex, chest pain type, resting blood pressure, cholesterol, fasting blood sugar, resting ECG, max heart rate, exercise-induced angina, ST depression/slope, number of major vessels, and thalassemia.
- After dropping 7 faulty rows (invalid `ca` and `thal` values flagged in the dataset discussion) the working set is 296 records.
- Included in `data/` (`heart.csv`).

## Approach
1. **Preprocessing**: drop the 7 erroneous rows; rename abbreviated columns to readable medical names; map coded categories to labels (e.g. chest pain type, thalassemia) for EDA.
2. **EDA**: group features into numeric, binary and nominal; explore distributions and a correlation heatmap.
3. **Feature prep**: one-hot encode categorical features for modeling.
4. **Models**:
   - Scikit-learn classifiers including Logistic Regression with `RandomizedSearchCV` hyperparameter tuning (best params found: `C≈0.284`, `penalty=l2`, `solver=liblinear`).
   - Gradient-boosting models: CatBoost, LightGBM and XGBoost.
5. **Evaluation**: precision / recall / F1 / accuracy via `classification_report`; model explainability section.

## Results
Metrics from the recorded run on the held-out test set (74 samples):

| Model | Accuracy | Macro F1 | Notes |
|-------|----------|----------|-------|
| Logistic Regression (tuned) | 0.86 | 0.86 | class 1 recall 0.91 |
| Boosting (XGB/LGBM/CatBoost ensemble report) | 0.86 | 0.86 | class 1 recall 0.94 |

Per-class on the tuned Logistic Regression: precision 0.91/0.82, recall 0.82/0.91 for classes 0/1.

## Key Takeaways
- A well-tuned Logistic Regression matches gradient-boosting models on this small dataset (~0.86 test accuracy).
- Cleaning the 7 flagged rows is important; the raw dataset encodes some NaNs as out-of-range category codes.
- With only 296 rows, model variance is high — cross-validation and more data would firm up the estimates.

## How to Run
```bash
pip install pandas numpy scipy scikit-learn seaborn matplotlib xgboost lightgbm catboost
jupyter notebook notebook.ipynb
```
Data is already in `data/heart.csv`. Adjust the read path if you keep it elsewhere.

## Credit
> Based on ["Heart Disease Predictions"](https://www.kaggle.com/code/desalegngeb/heart-disease-predictions) by A Legacy Grandmaster! on Kaggle (4093 votes).
> Dataset: [Heart Disease](https://www.kaggle.com/datasets/data855/heart-disease). Adapted and documented for this portfolio.
