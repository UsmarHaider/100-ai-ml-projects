# Loan Approval / Default Prediction (LendingClub)

![ML](https://img.shields.io/badge/domain-Machine%20Learning-blue) ![Task](https://img.shields.io/badge/task-Binary%20Classification-green)

## Overview
A credit-risk model for LendingClub peer-to-peer loans: predict whether a borrower will fully repay or default ("charged-off"). Identifying risky applicants lets the lender cut credit losses. The notebook combines extensive EDA/cleaning with a neural network and tree-based classifiers.

## Dataset
- LendingClub loan data. Primary source: [wordsforthewise/lending-club](https://www.kaggle.com/datasets/wordsforthewise/lending-club). Mirrors/alternatives used by the community include [adarshsng/lending-club-loan-data-csv](https://www.kaggle.com/datasets/adarshsng/lending-club-loan-data-csv), [ethon0426/lending-club-20072020q1](https://www.kaggle.com/datasets/ethon0426/lending-club-20072020q1), and [janiobachmann/lending-club-first-dataset](https://www.kaggle.com/datasets/janiobachmann/lending-club-first-dataset).
- Target `loan_status` (Fully Paid vs. Charged Off). ~392k train / ~130k test rows after cleaning (per the run log: 262,143 train and 130,423 test samples; positive class ≈ 80%). Features include loan amount, term, interest rate, grade/subgrade, employment, home ownership, annual income, purpose, zip code, and more.
- **Not committed** (multi-GB). Download with:
  ```bash
  kaggle datasets download wordsforthewise/lending-club
  ```

## Approach
1. **Cleaning / feature engineering**: drop high-cardinality/leaky columns (`emp_title`, `emp_length`, `title`, `grade`, `address`, `issue_d`); impute `mort_acc` from correlated features; `dropna`; one-hot encode categoricals and `zip_code`.
2. **Split & scale**: `train_test_split` (test_size=0.33); `MinMaxScaler`.
3. **Models**:
   - **Neural network** (Keras functional `Model`): Dense ReLU hidden layers with `Dropout` ([0.1, 0, 0.1, 0]) and BatchNormalization, sigmoid output, `Adam` optimizer, `binary_crossentropy` loss, AUC metric.
   - **XGBoost** classifier (with an optional `RandomizedSearchCV` tuning block).
   - **Random Forest** classifier.
4. **Evaluation**: accuracy, confusion matrix, classification report, and ROC-AUC per model.

## Results
From the recorded run (test set ≈ 130k rows):

| Model | Test Accuracy | ROC-AUC |
|-------|---------------|---------|
| Random Forest | ~88.9% | 0.725 |
| XGBoost | ~88.9% | 0.734 |
| Neural Network (ANN) | ~88.9% | 0.905 |

All models reach ~89% accuracy, but accuracy is inflated by the ~80% "fully paid" majority class — the ANN's 0.905 ROC-AUC is the more meaningful separator. Tree models showed near-perfect train accuracy (overfitting) on some configurations.

## Key Takeaways
- On an imbalanced credit dataset, ROC-AUC (and recall on the default class) matters far more than raw accuracy.
- The neural network achieved by far the best ranking quality (AUC 0.905) versus the tree models (0.72–0.73).
- Careful removal of leaky / high-cardinality columns is essential before modeling.

## How to Run
```bash
pip install pandas numpy scipy scikit-learn seaborn matplotlib hvplot xgboost tensorflow
# download the dataset (see above), place CSV under ../input or adjust the read path
jupyter notebook notebook.ipynb
```

## Credit
> Based on ["🏦 Lending Club Loan 💰 Defaulters 🏃 Prediction"](https://www.kaggle.com/code/faressayah/lending-club-loan-defaulters-prediction) by Fares Sayah on Kaggle (1782 votes).
> Dataset: [LendingClub](https://www.kaggle.com/datasets/wordsforthewise/lending-club). Adapted and documented for this portfolio.
