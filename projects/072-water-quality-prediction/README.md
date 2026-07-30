# Water Potability Prediction

![ML](https://img.shields.io/badge/domain-Machine%20Learning-blue)
![Classification](https://img.shields.io/badge/task-Binary%20Classification-green)

## Overview
Predict whether water is safe to drink (`Potability`) from nine physico-chemical
measurements (pH, hardness, solids, chloramines, sulfate, conductivity, organic
carbon, trihalomethanes, turbidity). The notebook benchmarks **seven
classifiers** to find the best predictor.

## Dataset
- **Water Quality / Potability** — [kaggle.com/datasets/adityakadiwal/water-potability](https://www.kaggle.com/datasets/adityakadiwal/water-potability)
- `water_potability.csv`: **3,276 rows**, 9 numeric features + binary
  `Potability` target (1,998 not-potable / 1,278 potable).
- Missing values in `ph` (491), `Sulfate` (781), `Trihalomethanes` (162).
- Included in `data/` (`water_potability.csv`, ~0.5 MB).

## Approach
1. **Imputation**: fill missing `ph`, `Sulfate`, `Trihalomethanes` with the
   column mean.
2. **Scaling**: `StandardScaler` on the features.
3. **Split**: `train_test_split(test_size=0.33, random_state=42)`.
4. **Classifiers compared**: Logistic Regression, Decision Tree, Random Forest,
   XGBoost, KNN, SVM (`SVC(kernel='rbf', random_state=42)`), AdaBoost
   (`learning_rate=0.002, n_estimators=205`).
5. **Metrics**: accuracy score + full `classification_report` per model, then a
   summary comparison table.

## Results
Real metrics from the captured Kaggle run log, test
set accuracy (reports appear in notebook order):

| Model | Test Accuracy |
|-------|---------------|
| Logistic Regression | 0.628 |
| Decision Tree | 0.645 |
| Random Forest | 0.628 |
| **XGBoost** | **0.671** |
| KNN | 0.653 |
| SVM (RBF) | 0.689 |
| AdaBoost | 0.634 |

The notebook concludes **SVM achieves the highest accuracy** (≈0.69). Note the
minority (potable) class has low recall for several models — see the per-class
reports in the log.

## Key Takeaways
- The dataset is only moderately separable: best accuracy hovers around 65–69%.
- Linear/forest models tend to predict the majority (not-potable) class,
  yielding ~0 recall on the potable class; SVM and XGBoost handle it better.
- Class imbalance and missing-value imputation choices materially affect
  results — resampling (e.g. SMOTE) or better imputation are natural next steps.

## How to Run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
jupyter notebook notebook.ipynb
```
The dataset ships in `data/water_potability.csv`. The notebook reads from
`/kaggle/input/water-potability/`; update the path when running locally.

## Credit
> Based on ["Water Quality Prediction ( 7 model )"](https://www.kaggle.com/code/imakash3011/water-quality-prediction-7-model)
> by Akash Patel on Kaggle (945 votes).
> Dataset: [Water Quality](https://www.kaggle.com/datasets/adityakadiwal/water-potability).
> Adapted and documented for this portfolio.
