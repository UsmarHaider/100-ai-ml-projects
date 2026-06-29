# Stock Market Analysis & Price Prediction with LSTM

![ML](https://img.shields.io/badge/domain-Machine%20Learning-blue) ![Task](https://img.shields.io/badge/task-Time%20Series%20%2F%20Regression-purple)

## Overview
Explores technology-stock price data (Apple, Amazon, Google, Microsoft), analyzing returns, moving averages, correlation and investment risk, then forecasts Apple's closing price with an LSTM recurrent neural network.

## Dataset
- Live data is pulled at runtime via `yfinance` / `pandas_datareader` (the recorded run fetched ~251 daily records for 2022-01 to 2023-01 with OHLC, Adj Close and Volume columns).
- Static Kaggle datasets referenced by the original kernel (included in `data/`): portfolio quotes (`portfolio_data.csv`), Apple historical data (`HistoricalQuotes.csv`), and Tesla (`Tesla.csv - Tesla.csv.csv`).
  - [AMZN/DPZ/BTC/NFLX adjusted May 2013–May 2019](https://www.kaggle.com/datasets/hershyandrew/amzn-dpz-btc-ntfx-adjusted-may-2013may2019)
  - [Apple AAPL historical stock data](https://www.kaggle.com/datasets/tarunpaparaju/apple-aapl-historical-stock-data)
  - [Tesla stock price](https://www.kaggle.com/datasets/rpaguirre/tesla-stock-price)

## Approach
1. **Acquire data** with `yfinance` for the four tech tickers; inspect with `.describe()` / `.info()`.
2. **Exploratory analysis** (questions 1–5): closing price over time, trading volume, 10/20-day moving averages, daily returns distribution, inter-stock return correlation, and value-at-risk.
3. **Forecasting (Apple)**:
   - Use the `Close` series; train/test split at 95% (`training_data_len = ceil(len*0.95)`).
   - Scale to [0,1] with `MinMaxScaler`; build 60-step lookback windows; reshape to `(samples, timesteps, 1)`.
   - **LSTM model** (Keras `Sequential`): `LSTM(128, return_sequences=True)` → `LSTM(64)` → `Dense(25)` → `Dense(1)`; `adam` optimizer, `mean_squared_error` loss; `batch_size=1`, `epochs=1`.
4. **Evaluation**: inverse-transform predictions and compute RMSE against actual closing prices.

## Results
The model computes **RMSE** (root mean squared error) between predicted and actual Apple closing prices, plus the EDA statistics (moving averages, daily-return distribution, correlation matrix). The recorded run captured the data pipeline and scaled training array but did not print a clean final RMSE value. Run the notebook to reproduce the RMSE metric and the train/validation/prediction price plot.

## Key Takeaways
- LSTMs can capture short-term temporal structure in price series using a sliding lookback window.
- With `epochs=1` this is a demonstration setup; more epochs, multivariate inputs and walk-forward validation would improve and properly validate the forecast.
- Price prediction is not investment advice — correlation/risk EDA is as valuable as the forecast itself.

## How to Run
```bash
pip install pandas numpy scikit-learn seaborn matplotlib tensorflow keras yfinance pandas_datareader
jupyter notebook notebook.ipynb
```
The notebook fetches live prices via `yfinance`; static CSVs are also provided in `data/`.

## Credit
> Based on ["📊Stock Market Analysis 📈 + Prediction using LSTM"](https://www.kaggle.com/code/faressayah/stock-market-analysis-prediction-using-lstm) by Fares Sayah on Kaggle (13132 votes).
> Datasets: [AMZN/DPZ/BTC/NFLX](https://www.kaggle.com/datasets/hershyandrew/amzn-dpz-btc-ntfx-adjusted-may-2013may2019), [Apple AAPL](https://www.kaggle.com/datasets/tarunpaparaju/apple-aapl-historical-stock-data), [Tesla](https://www.kaggle.com/datasets/rpaguirre/tesla-stock-price). Adapted and documented for this portfolio.
