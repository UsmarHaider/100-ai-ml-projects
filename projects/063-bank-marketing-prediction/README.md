# Bank Term Deposit Marketing Prediction

![ML](https://img.shields.io/badge/domain-Machine%20Learning-blue)
![Classification](https://img.shields.io/badge/task-Binary%20Classification-green)

## Overview
Predict whether a bank client will subscribe to a **term deposit** following a
marketing call. The notebook combines rich EDA ("intelligent targeting") with a
classification pipeline, comparing several models and tuning the
precision/recall trade-off for an imbalanced target.

## Dataset
- **Bank Marketing Dataset** — [kaggle.com/datasets/janiobachmann/bank-marketing-dataset](https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset)
- `bank.csv`: client demographics, campaign contact details, and the binary
  target `deposit` (≈87% "no" / 13% "yes"); no missing values.
- Included in `data/` (`bank.csv`, ~0.9 MB).

## Approach
1. **EDA**: analysis by occupation, marital status, education (clustering),
   campaign duration, and correlations driving the decision.
2. **Preprocessing**: custom `LabelEncoder` / `CategoricalEncoder` transformers,
   `StandardScaler`, combined in a `FeatureUnion` / `Pipeline`.
3. **Stratified sampling**: `StratifiedShuffleSplit` to preserve the 87/13
   class ratio in train/test.
4. **Model survey**: Logistic Regression, KNN, Linear SVM, Gradient Boosting,
   Decision Tree, Random Forest (`n_estimators=18`), Neural Net (`MLPClassifier`,
   `alpha=1`), Naive Bayes — compared by training score.
5. **Overfitting control**: 3-fold `cross_val_score` to fairly rank models
   (Decision Tree / Random Forest flagged as overfitting at ~99–100% train acc).
6. **Threshold tuning**: precision-recall curve and ROC analysis on the
   `GradientBoostingClassifier`; best precision/recall balance near a 0.13
   threshold.
7. **Ensemble**: soft `VotingClassifier` of Gradient Boosting + Naive Bayes +
   Neural Net.

## Results
No run log was captured for this project (no run output was captured).
The notebook's markdown/plots report these **ROC AUC** scores
(via `cross_val_predict` decision functions):

| Classifier | ROC AUC |
|------------|---------|
| **Gradient Boosting** | **~91.72%** |
| Neural Net (MLP) | ~91.54% |
| Naive Bayes | ~80.33% |

The notebook concludes "**GradientBoosting Classifier Wins**" and also reports
precision, recall, F1, and a confusion matrix. Run the notebook to reproduce
the exact precision/recall/F1 and the final VotingClassifier accuracy.

## Key Takeaways
- The target is imbalanced, so ROC AUC and the precision/recall trade-off
  matter more than raw accuracy.
- Tree/forest models overfit badly on the training set; cross-validation
  exposes this.
- Gradient Boosting is the strongest single model; call `duration` is highly
  predictive but cannot be used in a realistic pre-call model (it is only known
  after the call).

## How to Run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn plotly squarify
jupyter notebook notebook.ipynb
```
The dataset ships in `data/bank.csv`. Note: the notebook uses the legacy
`plotly.plotly` API — modern Plotly imports differ (`plotly.offline` /
`plotly.graph_objects`).

## Credit
> Based on ["Bank Marketing Campaign || Opening a Term Deposit"](https://www.kaggle.com/code/janiobachmann/bank-marketing-campaign-opening-a-term-deposit)
> by Janio Martinez Bachmann on Kaggle (2,385 votes).
> Dataset: [Bank Marketing Dataset](https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset).
> Adapted and documented for this portfolio.
