# Titanic Survival Prediction

![Domain](https://img.shields.io/badge/domain-Machine%20Learning-blue)
![Task](https://img.shields.io/badge/task-Binary%20Classification-green)

## Overview
The canonical end-to-end Titanic data-science workflow: acquire, wrangle, analyze,
feature-engineer, then train and compare several classifiers to predict passenger survival.
A thorough tutorial covering correlation analysis, missing-value imputation, feature
creation, and model benchmarking.

## Dataset
- **Titanic - Machine Learning from Disaster** — [Kaggle competition](https://www.kaggle.com/competitions/titanic)
- `train.csv` (891 passengers, ~38% survived) and `test.csv`; features include `Pclass`,
  `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked`.
- Data is **not** committed. Download via the competition page:
  ```bash
  kaggle competitions download -c titanic
  ```
  The notebook reads `../input/train.csv` and `../input/test.csv`.

## Approach
1. **Analysis** — survival correlation by `Pclass`, `Sex`, `SibSp`, `Parch`; visualize age
   and fare distributions.
2. **Feature engineering** — extract `Title` from `Name`; create `FamilySize` =
   `SibSp + Parch + 1` and an `IsAlone` flag; band `Age` (`pd.cut`, 5 bins) and `Fare`
   (`pd.qcut`, 4 bins); engineer `Age*Class`.
3. **Cleaning/encoding** — impute missing `Age`/`Embarked`/`Fare`; map `Sex`, `Title`,
   `Embarked`, banded features to integers; drop `Ticket`, `Cabin`, `Name`, `PassengerId`.
4. **Models** — train and score (on training data) Logistic Regression, SVC, Linear SVC,
   `KNeighborsClassifier(n_neighbors=3)`, Gaussian Naive Bayes, Perceptron, SGD,
   Decision Tree, and Random Forest; rank them in a comparison table.

## Results
The notebook reports each model's **training-set accuracy** via `model.score(...)*100`
rounded to 2 decimals, then assembles a sorted model-comparison table. The run log for this
project is empty, so no executed accuracy values were captured here.

| Metric | Where reported |
|---|---|
| Training accuracy (per model) | `acc_log`, `acc_svc`, `acc_knn`, `acc_random_forest`, ... |
| Model ranking table | sorted `models` DataFrame by `Score` |
| Logistic regression coefficients | `coeff_df` feature-correlation table |

Run the notebook to reproduce the per-model accuracy table and generate `submission.csv`.

## Key Takeaways
- Feature engineering (`Title`, `FamilySize`/`IsAlone`, banded `Age`/`Fare`) drives most of
  the gains over raw columns.
- Reported scores are training-set self-scores; for an unbiased estimate use
  cross-validation or the Kaggle leaderboard (the public LB is the true test metric).
- Tree-ensemble models (Random Forest, Decision Tree) typically top the comparison table on
  this dataset.

## How to Run
```bash
pip install pandas numpy scikit-learn seaborn matplotlib
jupyter notebook notebook.ipynb
```
Download the competition data (see Dataset) and place the CSVs at `../input/`.

## Credit
> Based on ["Titanic Data Science Solutions"](https://www.kaggle.com/code/startupsci/titanic-data-science-solutions) by Manav Sehgal on Kaggle (39960 votes).
> Dataset: [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic). Adapted and documented for this portfolio.
