# COVID-19 Visualization & Case Forecasting

![ML](https://img.shields.io/badge/domain-Machine%20Learning-blue)
![Task](https://img.shields.io/badge/task-Time%20Series%20Regression-green)

## Overview
Visualizes the global spread of COVID-19 and forecasts future confirmed-case counts using
regression on the time axis. Three models — Support Vector Regression, polynomial linear
regression, and Bayesian Ridge polynomial regression — are fit to the latest data window and
projected forward.

## Dataset
- Johns Hopkins University CSSE COVID-19 time-series data (confirmed cases, deaths,
  recoveries), as used by the original Kaggle kernel.
- The notebook reads `days_since_1_22` engineered from the case time series; modelling uses
  a recent window (data from 8/1/2022 onward).
- **Not committed.** Source data: the
  [JHU CSSE COVID-19 repository](https://github.com/CSSEGISandData/COVID-19), or the original
  Kaggle kernel's attached dataset linked in Credit.

## Approach
1. **Aggregation** — build worldwide confirmed/deaths/recoveries time series; engineer
   `days_since_1/22/2020` as the regression feature.
2. **Train/test split** — use the recent window (skip earlier days) and `train_test_split`.
3. **SVR model** — `SVR(kernel='poly', degree=3, C=0.1, gamma=0.01, epsilon=1, shrinking=True)`;
   optimal params explored via `RandomizedSearchCV` (commented out).
4. **Polynomial regression** — `PolynomialFeatures(degree=3)` + `LinearRegression`.
5. **Bayesian Ridge** — `PolynomialFeatures(degree=3)` + `BayesianRidge`, tuned via
   `RandomizedSearchCV` over tol / alpha / lambda.
6. **Evaluation & forecast** — MAE and MSE on the test window; project the next ~10 days for
   each model and visualize against actuals.

## Results
The notebook had no executed cell outputs and the captured run log is empty, so no real
metric values are available. The code computes, for each model:

- **MAE** (`mean_absolute_error`) on the test window
- **MSE** (`mean_squared_error`) on the test window
- 10-day forward case-count forecasts (SVR, polynomial, Bayesian Ridge)

**Run the notebook to reproduce the metrics** (MAE/MSE per model and the forecast tables).

## Key Takeaways
- Treating cumulative cases as a function of "days since start" turns forecasting into a
  simple 1-D regression — easy to fit but sensitive to the chosen data window.
- Three model families are compared so their forecast divergence illustrates model
  uncertainty rather than relying on a single projection.
- Polynomial degree and the SVR `C`/`gamma`/`epsilon` choices strongly drive the projection;
  the included `RandomizedSearchCV` scaffolding supports retuning as data grows.

## How to Run
```bash
pip install numpy pandas matplotlib scikit-learn
jupyter notebook notebook.ipynb
```
Place the JHU CSSE time-series CSVs in the input path the notebook reads from.

## Credit
> Based on ["Coronavirus (COVID-19) Visualization & Prediction"](https://www.kaggle.com/code/therealcyberlord/coronavirus-covid-19-visualization-prediction) by Xingyu Bian on Kaggle (8,990 votes).
> Data: [JHU CSSE COVID-19 dataset](https://github.com/CSSEGISandData/COVID-19).
> Adapted and documented for this portfolio.
