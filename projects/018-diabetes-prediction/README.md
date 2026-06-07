# Diabetes Prediction (Pima Indians)

![ML](https://img.shields.io/badge/domain-Machine%20Learning-blue) ![Task](https://img.shields.io/badge/task-Binary%20Classification-green)

## Overview
Predicts whether a patient has diabetes from routine diagnostic measurements in the Pima Indians Diabetes dataset. The notebook is an end-to-end study: careful missing-value imputation, EDA, feature engineering, and a tuned LightGBM model boosted past 90% accuracy by ensembling with KNN.

## Dataset
- [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) (ref `uciml/pima-indians-diabetes-database`).
- 768 records, 8 numeric features (Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age) and a binary `Outcome`.
- Included in `data/` (`diabetes.csv`).

## Approach
1. **Missing-value handling**: zero values in Insulin (48.7%), SkinThickness (29.6%), BloodPressure (4.6%), BMI (1.4%) and Glucose (0.65%) are treated as missing and replaced with informed estimates.
2. **EDA**: target balance, per-feature distributions (healthy vs. diabetic), Plotly visualizations.
3. **Feature engineering**: ~16 new features created to lift accuracy.
4. **Encoding/scaling**: `StandardScaler` + `LabelEncoder`; correlation matrix review.
5. **Models**:
   - LightGBM tuned with `RandomizedSearchCV` (5-fold CV) with a discrimination-threshold analysis.
   - A `VotingClassifier` combining the tuned LightGBM with `KNeighborsClassifier`, tuned via `GridSearchCV`.
6. **Evaluation**: 5-fold cross-validated accuracy, ROC/AUC, precision-recall.

## Results
Cross-validated accuracies stated in the notebook (matching the kernel title "0.906"):

| Model | 5-fold Accuracy |
|-------|-----------------|
| LightGBM (RandomSearch tuned) | 89.8% |
| LightGBM + KNN (VotingClassifier) | 90.6% |

## Key Takeaways
- Correctly treating physiologically-impossible zeros as missing values is the single most impactful preprocessing step.
- Feature engineering plus a KNN/LightGBM voting ensemble pushes accuracy past the 90% mark.
- Discrimination-threshold tuning matters for a medical screening task where recall on the positive class is valuable.

## How to Run
```bash
pip install pandas numpy scipy scikit-learn seaborn matplotlib plotly squarify lightgbm yellowbrick
jupyter notebook notebook.ipynb
```
Data is already in `data/diabetes.csv`.

## Credit
> Based on ["Pima Indians Diabetes - EDA & Prediction (0.906)"](https://www.kaggle.com/code/vincentlugat/pima-indians-diabetes-eda-prediction-0-906) by Vincent Lugat on Kaggle (2274 votes).
> Dataset: [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database). Adapted and documented for this portfolio.
