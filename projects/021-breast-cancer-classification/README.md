# Breast Cancer Classification

![ML](https://img.shields.io/badge/domain-Machine%20Learning-blue) ![Task](https://img.shields.io/badge/task-Binary%20Classification-green)

## Overview
Classifies breast tumors as malignant or benign from the Wisconsin Diagnostic Breast Cancer (WDBC) dataset of cell-nucleus measurements. The notebook walks through EDA, then compares Logistic Regression, Decision Tree and Random Forest models, using feature importance to select a compact, high-accuracy predictor set.

## Dataset
- [Breast Cancer Wisconsin (Diagnostic)](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data) (ref `uciml/breast-cancer-wisconsin-data`).
- 569 records. The `diagnosis` target is M (malignant) / B (benign). 30 numeric features = mean, standard error and "worst" of 10 cell-nucleus measurements (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension).
- Included in `data/` (`data.csv`).

## Approach
1. **Preprocessing**: drop the empty `Unnamed: 32` column; encode `diagnosis` to 1 (M) / 0 (B).
2. **EDA**: per-feature histograms split by diagnosis; identify that mean radius, perimeter, area, compactness, concavity and concave points separate the classes well.
3. **Split**: simple 70:30 train/test split.
4. **Models** (each wrapped in a helper that reports train accuracy + 5-fold KFold CV):
   - Logistic Regression on the 6 "mean" predictors.
   - Decision Tree.
   - Random Forest (`n_estimators=100`, `min_samples_split=25`, `max_depth=7`, `max_features=2`), then refit on the top-5 features by importance: `concave points_mean`, `area_mean`, `radius_mean`, `perimeter_mean`, `concavity_mean`.
5. **Evaluation**: training accuracy and KFold cross-validation score.

## Results
The notebook reports the following for the best model (Random Forest on the top-5 features), per its concluding markdown:

| Model | Test Accuracy | CV Score |
|-------|---------------|----------|
| Random Forest (top-5 features) | ~95% | ~93% |

The notebook's executed run log contained only nbconvert messages, not numeric cell outputs. Re-run the notebook to regenerate the per-model accuracy and KFold scores printed by `classification_model`.

## Key Takeaways
- A handful of nucleus-geometry features (concave points, area, radius, perimeter, concavity) carry most of the predictive signal.
- Random Forest on the reduced feature set beats single predictors and avoids overfitting better than a lone Decision Tree.
- Further gains likely from hyperparameter search and trying gradient boosting / SVM.

## How to Run
```bash
pip install pandas numpy scikit-learn matplotlib mpld3
jupyter notebook notebook.ipynb
```
Note: the original uses the deprecated `sklearn.cross_validation` API; on modern scikit-learn replace it with `sklearn.model_selection`. Data is already in `data/data.csv`.

## Credit
> Based on ["Breast cancer prediction"](https://www.kaggle.com/code/buddhiniw/breast-cancer-prediction) by Buddhini W on Kaggle (2272 votes).
> Dataset: [Breast Cancer Wisconsin (Diagnostic)](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data). Adapted and documented for this portfolio.
