# Flight Ticket Price Prediction

![ML](https://img.shields.io/badge/domain-Machine%20Learning-blue)
![Regression](https://img.shields.io/badge/task-Regression-orange)

## Overview
Predict airline ticket fares from flight attributes (airline, route, stops,
departure/arrival time, class, duration, days-left-to-departure). The notebook
performs extensive EDA, then benchmarks **10 regression models** and selects the
best performer.

## Dataset
- **Flight Price Prediction** — [kaggle.com/datasets/shubhambathwal/flight-price-prediction](https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction)
- `Clean_Dataset.csv`: **300,153 rows**, 11 columns (8 categorical, 3 numeric
  incl. target `price`).
- Not committed to this repo (~53 MB). Download with:
  ```bash
  kaggle datasets download shubhambathwal/flight-price-prediction
  ```

## Approach
1. **EDA**: price vs. airline, class (Economy vs. Business), stops, departure/
   arrival time, source/destination city, duration, and days-left.
2. **Encoding**: `LabelEncoder` on every object column.
3. **Split**: 70/30 train/test (`random_state=42`).
4. **Scaling**: `MinMaxScaler` to [0, 1].
5. **Models benchmarked** (default hyperparameters unless noted): Linear
   Regression, Ridge, Lasso(α=0.1), Decision Tree, Random Forest, Extra Trees,
   Gradient Boosting (lr=0.1, n=100, depth=3), XGBoost, KNN(k=5), Bagging.
6. **Metrics**: MAE, MSE, RMSE, R², Adjusted R², MAPE, RMSLE per model;
   ranked by Adjusted R²; the top model is refit and visualized.

## Results
Real metrics from the captured Kaggle run log, test
set:

| Model | RMSE | R² | MAPE |
|-------|------|-----|------|
| **Extra Trees** | **2800.27** | **0.9848** | **7.62%** |
| Random Forest | 2855.65 | 0.9842 | 7.90% |
| Bagging | 2978.25 | 0.9828 | 8.21% |
| Decision Tree | 3682.49 | 0.9737 | 8.64% |
| XGBoost | 3368.90 | 0.9779 | 14.42% |
| KNN | 3819.27 | 0.9717 | 11.11% |
| Gradient Boosting | 4713.93 | 0.9568 | 20.67% |
| Linear / Ridge / Lasso | ~7005 | 0.9047 | 43.89% |

**Extra Trees Regressor wins** with R² ≈ 0.985 and the lowest RMSE/MAPE.

## Key Takeaways
- Tree ensembles (Extra Trees, Random Forest, Bagging) massively outperform
  linear models here — fare structure is highly non-linear.
- Linear/Ridge/Lasso plateau at R² ≈ 0.90 with ~44% MAPE.
- `class` (Economy vs. Business) and `days_left` are dominant price signals;
  fares spike sharply when only ~2 days remain.

## How to Run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
jupyter notebook notebook.ipynb
```
Download the dataset (command above) and place `Clean_Dataset.csv` where the
notebook expects it (`/kaggle/input/flight-price-prediction/`).

## Credit
> Based on ["Flight Fare Prediction | 10 ML Models"](https://www.kaggle.com/code/varunsaikanuri/flight-fare-prediction-10-ml-models)
> by Varun Sai Kanuri on Kaggle (714 votes).
> Dataset: [Flight Price Prediction](https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction).
> Adapted and documented for this portfolio.
