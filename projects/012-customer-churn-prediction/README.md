# Customer Churn Prediction

![Domain](https://img.shields.io/badge/domain-Machine%20Learning-blue)
![Task](https://img.shields.io/badge/task-Binary%20Classification-green)

## Overview
Predicts whether a telecom customer will churn, using the Telco Customer Churn dataset.
The notebook does thorough EDA, encodes and scales the features, then benchmarks a wide set
of classifiers and combines the best ones in a soft-voting ensemble.

## Dataset
- **Telco Customer Churn** — [Kaggle dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (`blastchar/telco-customer-churn`)
- **7,043** customers × 21 columns; target `Churn` (Yes/No). Features cover demographics,
  account info, and subscribed services.
- **Included in `data/`**: `WA_Fn-UseC_-Telco-Customer-Churn.csv`.

## Approach
1. **Preprocessing** — coerce `TotalCharges` to numeric, encode categoricals
   (`LabelEncoder` / dummies), `StandardScaler` on numeric columns.
2. **Split** — `train_test_split(test_size=0.30, random_state=40, stratify=y)`
   → 4,933 train / 2,110 test.
3. **Models benchmarked** — KNN (`n_neighbors=11`), SVM (`SVC`), Random Forest
   (`n_estimators=500, max_leaf_nodes=30, oob_score=True`), Logistic Regression,
   Decision Tree, AdaBoost, Gradient Boosting.
4. **Ensemble** — soft-voting `VotingClassifier` over Gradient Boosting + Logistic
   Regression + AdaBoost.
5. **Evaluation** — accuracy, `classification_report`, and confusion matrices per model.

## Results
Real values extracted from the captured run log on the
2,110-row test set (class 1 = churn):

| Model | Test accuracy | Churn (class 1) F1 |
|---|---|---|
| KNN (k=11) | 0.775 | 0.55 |
| SVM | 0.808 | 0.58 |
| Random Forest | 0.814 | 0.59 |
| Logistic Regression | 0.809 | 0.62 |
| Decision Tree | 0.725 | 0.50 |
| AdaBoost | 0.81 | 0.60 |
| Gradient Boosting | 0.808 | 0.60 |
| **Voting (GB + LR + AdaBoost)** | **0.817** | **0.63** |

The soft-voting ensemble is the best model at **81.7% accuracy** (weighted F1 0.81).

## Key Takeaways
- The soft-voting ensemble edges out every individual model, reaching **81.7% accuracy**.
- All models predict the majority "no churn" class far better (F1 ~0.85–0.88) than the
  churn class (F1 ~0.50–0.63) — the minority class is the hard part.
- For a churn use case, recall on the churn class matters most; class weighting / SMOTE or
  threshold tuning would likely improve it beyond raw accuracy.

## How to Run
```bash
pip install pandas numpy scikit-learn seaborn matplotlib plotly missingno
jupyter notebook notebook.ipynb
```
Data is already in `data/`; update the read path if needed (the notebook reads
`../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv`).

## Credit
> Based on ["CUSTOMER CHURN PREDICTION 📈"](https://www.kaggle.com/code/bhartiprasad17/customer-churn-prediction) by BHARTI PRASAD on Kaggle (3209 votes).
> Dataset: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn). Adapted and documented for this portfolio.
