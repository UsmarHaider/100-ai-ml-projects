# Sales Forecasting — Traditional Time Series

![ML](https://img.shields.io/badge/domain-Machine%20Learning-blue) ![Task](https://img.shields.io/badge/task-Time%20Series%20Forecasting-purple)

## Overview
A tour of traditional time-series forecasting on the "Predict Future Sales" competition data (daily sales for store-item combinations from the Russian software firm 1C). The notebook builds intuition for stationarity, seasonality and classical models (AR/MA/ARIMA, ETS, Prophet) before extending to hierarchical forecasting.

## Dataset
- Kaggle competition: [Predict Future Sales](https://www.kaggle.com/competitions/competitive-data-science-predict-future-sales) (ref `competitive-data-science-predict-future-sales`).
- Files: `sales_train.csv` (daily sales), `items.csv`, `item_categories.csv`, `shops.csv`, `test.csv`, `sample_submission.csv`. Task: predict monthly sales per store-item.
- **Not committed.** Accept the competition rules and download:
  ```bash
  kaggle competitions download -c competitive-data-science-predict-future-sales
  ```

## Approach
1. **Load & aggregate**: merge sales with item/category/shop metadata; roll daily sales up to a monthly series.
2. **Single-series analysis**:
   - **Stationarity**: Augmented Dickey-Fuller test (`adfuller`), rolling mean/std.
   - **Decomposition**: `seasonal_decompose` (additive and multiplicative, period=12) into trend / seasonality / remainder.
   - **Models**: AR, MA, ARMA, **ARIMA** / **SARIMAX**, with `p`/`q` selection via AIC (`arma_order_select_ic`); ETS (exponential smoothing); **Prophet**; UCM.
3. **Hierarchical time series**: bottom-up, Average Historical Proportions (AHP), Proportions of the Historical Averages (PHA) and Forecast Proportions (FP) reconciliation across the store-item hierarchy.

## Results
This kernel is primarily educational and reports model diagnostics rather than a single leaderboard score. Metrics/diagnostics it computes: the ADF stationarity statistic and p-value, AIC for ARIMA order selection, decomposition components, and model fit summaries (the recorded run shows the SARIMAX optimizer converging, final objective `F ≈ 10.568`). Run the notebook to reproduce the per-model AIC, ADF results and forecast plots.

## Key Takeaways
- Differencing/transformations to achieve stationarity is the prerequisite for ARIMA-family models.
- AIC offers a principled way to choose ARIMA `(p,q)` orders.
- For a competition with thousands of store-item series, hierarchical reconciliation (bottom-up / AHP / PHA / FP) is the practical path beyond modeling one series at a time.

## How to Run
```bash
pip install pandas numpy scipy statsmodels seaborn matplotlib fbprophet
# download competition data (see above), place under ../input or adjust read paths
jupyter notebook notebook.ipynb
```
Note: `statsmodels.tsa.arima_model.ARIMA` and `fbprophet` are legacy; on current versions use `statsmodels.tsa.arima.model.ARIMA` and the `prophet` package.

## Credit
> Based on ["Time series Basics : Exploring traditional TS"](https://www.kaggle.com/code/jagangupta/time-series-basics-exploring-traditional-ts) by Jagan on Kaggle (4720 votes).
> Data: [Predict Future Sales competition](https://www.kaggle.com/competitions/competitive-data-science-predict-future-sales). Adapted and documented for this portfolio.
