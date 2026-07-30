# Energy Consumption Forecasting

![ML](https://img.shields.io/badge/domain-Machine%20Learning-blue)
![Time Series](https://img.shields.io/badge/task-Time%20Series%20Regression-orange)

## Overview
Forecast hourly electricity demand for the PJM East (PJME) region using
machine learning instead of classical time-series methods. The notebook frames
forecasting as a supervised regression problem with calendar features and trains
an **XGBoost** regressor, scored by RMSE.

## Dataset
- **Hourly Energy Consumption** — [kaggle.com/datasets/robikscube/hourly-energy-consumption](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption)
- Uses `PJME_hourly.csv`: hourly megawatt (MW) consumption indexed by
  `Datetime`.
- Not committed to this repo (~44 MB). Download with:
  ```bash
  kaggle datasets download robikscube/hourly-energy-consumption
  ```

## Approach
1. **Index**: parse `Datetime` to a `DatetimeIndex`.
2. **Train/test split by date**: train < 2015-01-01, test ≥ 2015-01-01.
3. **Feature creation** from the time index: `hour`, `dayofweek`, `quarter`,
   `month`, `year`, `dayofyear`, `dayofmonth`, `weekofyear`.
4. **Model**: `XGBRegressor` with `n_estimators=1000`, `max_depth=3`,
   `learning_rate=0.01`, `early_stopping_rounds=50`, `objective='reg:linear'`,
   evaluated on both train and test sets each round.
5. **Analysis**: feature importance bar chart, prediction overlay on the raw
   series, and the worst-predicted days by mean absolute error.

## Results
Real metrics from the captured Kaggle run log:

| Metric | Value |
|--------|-------|
| **RMSE on test set** | **3721.75 MW** |
| Validation RMSE (round 999) | train 3052.74 / test 3722.92 |

Training converges steadily over 1,000 boosting rounds (test RMSE falls from
~31,657 at round 0 to ~3,723 by round 999).

## Key Takeaways
- Pure calendar features + XGBoost capture strong daily/seasonal demand cycles
  without any lag features.
- `hour` and seasonal features dominate importance; demand peaks track the
  daily load curve and summer/winter seasons.
- Suggested improvements (from the notebook): more robust cross-validation and
  richer features such as weather forecasts and holidays.

## How to Run
```bash
pip install pandas numpy matplotlib seaborn xgboost scikit-learn
jupyter notebook notebook.ipynb
```
Download the dataset (command above) and place `PJME_hourly.csv` where the
notebook expects it (`../input/hourly-energy-consumption/`). Note: the
`reg:linear` objective is deprecated in modern XGBoost — use
`reg:squarederror`.

## Credit
> Based on ["Time Series Forecasting with Machine Learning [YT]"](https://www.kaggle.com/code/robikscube/time-series-forecasting-with-machine-learning-yt)
> by Rob Mulla on Kaggle (4,980 votes).
> Dataset: [Hourly Energy Consumption](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption).
> Adapted and documented for this portfolio.
