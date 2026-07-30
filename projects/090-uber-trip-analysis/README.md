# Uber Fare Prediction — Comparing Regression Models

![ML](https://img.shields.io/badge/domain-Machine%20Learning-blue)
![Task](https://img.shields.io/badge/task-Regression-green)

## Overview
Predicts Uber ride `fare_amount` from pickup/dropoff coordinates, passenger count and trip
time. The notebook builds a full preprocessing pipeline (distance computation, outlier
removal, encoding, scaling) and benchmarks five linear regression variants to find the best.

## Dataset
- **Uber Fares Dataset** — [Kaggle](https://www.kaggle.com/datasets/yasserh/uber-fares-dataset)
- 200,000 samples, 7 raw features (per run log): `fare_amount`, `pickup_datetime`,
  pickup/dropoff longitude & latitude, `passenger_count`.
- Included in `data/` (`uber.csv`).

## Approach
1. **Cleaning** — drop nulls; constrain coordinates to valid lat/long ranges; remove 109 duplicates.
2. **Feature engineering** — derive trip `Distance` (geopy/haversine) from coordinates;
   extract `year`, `weekday`, `Monthly_Quarter`, `Hourly_Segments` from `pickup_datetime`.
3. **Outlier removal** — drops to 163,203 samples (retaining ~81.6% of data after cleanup).
4. **Encoding & scaling** — dummy-encode categoricals; standardize features; drop highly
   correlated features via a correlation matrix.
5. **Models compared** — Multiple Linear Regression, Ridge, Lasso, Elastic-Net, and
   Polynomial Regression, each evaluated on train/test R², RSS, MSE and RMSE.

## Results
Metrics extracted from the run log — test set unless noted.

| Model | Test R² | Test RMSE |
|---|---|---|
| Multiple Linear Regression | 0.3282 | 4.137 |
| Ridge Regression | 0.3282 | 4.137 |
| Lasso Regression | 0.2623 | 4.335 |
| Elastic-Net Regression | 0.2395 | 4.402 |
| **Polynomial Regression** | **0.3721** | **4.000** |

Polynomial Regression gave the best test fit (R² ≈ 0.372, RMSE ≈ 4.00).

## Key Takeaways
- Distance is the dominant predictor of fare; engineering it from raw coordinates is the
  single most important step.
- Linear, Ridge and Lasso land in a similar R² band; the polynomial expansion captures
  mild non-linearity and edges ahead.
- Overall R² is modest (~0.37), suggesting tree-based or gradient-boosted models could
  meaningfully improve accuracy on this data.
- Aggressive outlier removal (dropping ~18% of rows) keeps the linear models stable but
  also caps how much variance any linear fit can explain here.

## How to Run
```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn geopy statsmodels
jupyter notebook notebook.ipynb
```
Data is already in `data/uber.csv`.

## Credit
> Based on ["Uber Fare Prediction - (Comparing Best ML Models)"](https://www.kaggle.com/code/yasserh/uber-fare-prediction-comparing-best-ml-models) by M Yasser H on Kaggle (454 votes).
> Dataset: [Uber Fares Dataset](https://www.kaggle.com/datasets/yasserh/uber-fares-dataset).
> Adapted and documented for this portfolio.
