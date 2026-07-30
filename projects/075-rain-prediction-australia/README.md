# Rain Prediction in Australia — Logistic Regression

![ML](https://img.shields.io/badge/domain-Machine%20Learning-blue)
![Task](https://img.shields.io/badge/task-Binary%20Classification-green)

## Overview
Predicts whether it will rain tomorrow (`RainTomorrow`) at locations across Australia
from daily weather observations. The task is a binary classification problem, solved
end-to-end with a Logistic Regression classifier — from EDA through feature engineering,
scaling, model training, threshold analysis, ROC-AUC, cross-validation, and grid search.

## Dataset
- **Rain in Australia** — [Kaggle](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package)
- ~145,000 daily records, 23 raw columns (numeric + categorical), target `RainTomorrow`.
- Mix of numerical (temperature, rainfall, humidity, pressure, wind speed) and
  categorical features (`Location`, `WindGustDir`, `WindDir9am`, `WindDir3pm`, `RainToday`).
- Included in `data/` (`weatherAUS.csv`).

## Approach
1. **EDA** — separate categorical vs numerical variables, inspect cardinality, missing
   values, frequency distributions, outliers (boxplots) and distributions (histograms).
2. **Feature engineering** — parse `Date` into `Year`/`Month`/`Day`; drop `RISK_MM` and `Date`.
3. **Missing values** — impute numerical with median, categorical with mode.
4. **Outliers** — top-coding/capping on skewed numeric variables.
5. **Encoding** — one-hot encoding for categorical variables (with NA indicator dummies).
6. **Scaling** — `MinMaxScaler` / feature scaling on the train and test splits.
7. **Model** — `LogisticRegression` (default `C=1`), with experiments at `C=100` and `C=0.01`.
8. **Evaluation** — accuracy, confusion matrix, classification report, threshold tuning,
   ROC-AUC, 5-fold cross-validation, and `GridSearchCV` hyperparameter optimization.

## Results
Metrics quoted from the notebook's markdown analysis cells.

| Metric | Value |
|---|---|
| Training accuracy | 0.8476 |
| Test accuracy | 0.8501 |
| Null accuracy (majority class) | 0.7759 |
| Mean cross-validation accuracy | 0.8474 |
| GridSearchCV best accuracy | 0.8507 |

The model comfortably beats the null-accuracy baseline (0.7759) and shows no overfitting
(train ≈ test accuracy). ROC-AUC trends toward 1, indicating good separability.

## Key Takeaways
- Logistic Regression reaches ~85% accuracy with the default `C=1`; behaviour is
  near-identical on train and test, so the model is well regularized.
- The majority of records predict "no rain," so the null-accuracy comparison is essential
  to claim genuine signal — the model clears it by ~7 points.
- `GridSearchCV` only nudged accuracy from 0.8501 to 0.8507; larger gains would likely
  come from non-linear models (gradient boosting) or richer feature engineering.

## How to Run
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
jupyter notebook notebook.ipynb
```
Data is already in `data/weatherAUS.csv`. Update the read path if running outside Kaggle.

## Credit
> Based on ["Logistic Regression Classifier Tutorial"](https://www.kaggle.com/code/prashant111/logistic-regression-classifier-tutorial) by Prashant Banerjee on Kaggle (2,791 votes).
> Dataset: [Rain in Australia](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package).
> Adapted and documented for this portfolio.
