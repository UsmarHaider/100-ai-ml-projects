# Employee Attrition Prediction (IBM HR Analytics)

![ML](https://img.shields.io/badge/domain-Machine%20Learning-blue) ![Task](https://img.shields.io/badge/task-Binary%20Classification-green)

## Overview
Predicts whether an employee will leave the company (attrition) from the IBM HR Analytics dataset, and surfaces the drivers of attrition. The notebook does thorough EDA, encodes the categorical HR fields, and benchmarks seven classifiers on an imbalanced target.

## Dataset
- [IBM HR Analytics Employee Attrition & Performance](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) (ref `pavansubhasht/ibm-hr-analytics-attrition-dataset`).
- 1,470 records, 35 columns. Target `Attrition` (Yes/No) is imbalanced: 1,233 No / 237 Yes (~16% positive). Features cover demographics, compensation, job role/level, satisfaction scores, tenure and overtime.
- Included in `data/` (`WA_Fn-UseC_-HR-Employee-Attrition.csv`).

## Approach
1. **EDA**: per-column unique-value and category breakdowns; distributions of attrition across job role, department, overtime, marital status, etc.
2. **Encoding**: map `Attrition` to 1/0; one-hot encode categoricals (expands to 136 columns); drop constant columns (`EmployeeCount`, `StandardHours`, `Over18`).
3. **Split**: train/test split (1,029 train / 441 test; class balance ~16% in both).
4. **Models** (each reports train + test confusion matrix, accuracy and classification report): Logistic Regression, Support Vector Machine, Random Forest (`GridSearchCV`, 648 candidates × 5 folds), XGBoost, LightGBM, CatBoost, AdaBoost (`GridSearchCV`).
5. **Evaluation**: accuracy, precision/recall/F1, and ROC-AUC compared across all models.

## Results
From the recorded run — test-set ROC-AUC by model (the meaningful metric given ~16% positives):

| Model | Test ROC-AUC | Test Accuracy |
|-------|--------------|---------------|
| XGBoost | 0.620 | ~0.83 |
| LightGBM | 0.609 | ~0.85 |
| AdaBoost | 0.599 | ~0.82 |
| CatBoost | 0.586 | ~0.85 |
| Logistic Regression | 0.546 | ~0.85 |
| Random Forest | 0.543 | ~0.86 |
| SVM | 0.500 | ~0.84 |

Note the accuracy paradox: every model scores ~83–86% accuracy simply by leaning toward the majority "No" class (recall on the leaving class is low, 0.11–0.41), while ROC-AUC reveals XGBoost (0.620) as the best ranker. Several tree models hit 100% train accuracy (overfitting).

## Key Takeaways
- With a ~16% positive rate, accuracy is misleading; ROC-AUC and recall on the attrition class are the right metrics.
- XGBoost gave the best discrimination (AUC 0.620), though all models struggled to recall actual leavers.
- Class-imbalance handling (resampling, class weights, threshold tuning) and richer feature engineering are the obvious next steps.

## How to Run
```bash
pip install pandas numpy scikit-learn seaborn matplotlib xgboost lightgbm catboost
jupyter notebook notebook.ipynb
```
Data is already in `data/WA_Fn-UseC_-HR-Employee-Attrition.csv`.

## Credit
> Based on ["IBM HR Analytics💼Employee Attrition & Performance"](https://www.kaggle.com/code/faressayah/ibm-hr-analytics-employee-attrition-performance) by Fares Sayah on Kaggle (1759 votes).
> Dataset: [IBM HR Analytics Employee Attrition](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset). Adapted and documented for this portfolio.
