# Wine Quality Prediction

![ML](https://img.shields.io/badge/domain-Machine%20Learning-blue) ![Task](https://img.shields.io/badge/task-Binary%20Classification-green)

## Overview
A beginner-friendly classification project on the red-wine physicochemical dataset. The continuous quality score is binarized into "good" vs. "bad", and three classifiers are compared, then improved with cross-validation and grid search.

## Dataset
- [Red Wine Quality (Cortez et al., 2009)](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009) (ref `uciml/red-wine-quality-cortez-et-al-2009`).
- 1,599 records, 11 physicochemical features (fixed/volatile acidity, citric acid, residual sugar, chlorides, free/total sulfur dioxide, density, pH, sulphates, alcohol) and an integer `quality` score (3–8).
- Included in `data/` (`winequality-red.csv`).

## Approach
1. **EDA**: bar plots of each feature vs. quality (alcohol and sulphates rise with quality; volatile acidity falls).
2. **Target binarization**: `quality` cut into bins `(2, 6.5, 8)` → `bad` (0) / `good` (1); `LabelEncoder`.
3. **Split & scale**: `train_test_split` (test_size=0.2, random_state=42); `StandardScaler`.
4. **Models**:
   - Random Forest (`n_estimators=200`).
   - SGD Classifier.
   - Support Vector Classifier (SVC), then tuned via `GridSearchCV` (`C`, `kernel` in {linear, rbf}, `gamma`; best ≈ `C=1.2, gamma=0.9, kernel=rbf`).
5. **Evaluation**: `classification_report`, confusion matrix, and 10-fold `cross_val_score`.

## Results
Accuracies reported in the notebook's markdown commentary:

| Model | Accuracy |
|-------|----------|
| Random Forest | ~87% |
| SGD Classifier | ~84% |
| SVC (default) | ~86% |
| SVC (GridSearchCV tuned) | ~90% |
| Random Forest (10-fold CV) | ~91% |

The recorded run log contained only nbconvert messages, not numeric cell outputs; the figures above are the notebook author's stated results. Re-run to regenerate the exact classification reports.

## Key Takeaways
- Binarizing quality turns a hard ordinal problem into a tractable binary one; the classes are imbalanced (far more "bad").
- Grid-search tuning lifts SVC from ~86% to ~90%, and cross-validation gives a more honest ~91% for Random Forest.
- Alcohol and sulphates are among the most discriminative features for high-quality wine.

## How to Run
```bash
pip install pandas numpy scikit-learn seaborn matplotlib
jupyter notebook notebook.ipynb
```
Data is already in `data/winequality-red.csv`.

## Credit
> Based on ["Prediction of quality of Wine"](https://www.kaggle.com/code/vishalyo990/prediction-of-quality-of-wine) by Vishal Kumar on Kaggle (2624 votes).
> Dataset: [Red Wine Quality](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009). Adapted and documented for this portfolio.
