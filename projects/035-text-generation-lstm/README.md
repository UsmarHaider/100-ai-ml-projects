# Recurrent Neural Networks: LSTM & GRU for Sequence Prediction

![NLP](https://img.shields.io/badge/domain-Sequence%20Modeling-blue) ![Task](https://img.shields.io/badge/task-Regression%20%2F%20Forecasting-green)

## Overview
A hands-on intro to recurrent neural networks that builds and compares stacked LSTM and GRU models for sequence prediction. The walkthrough uses IBM daily stock prices as the example sequence, forecasting the 2017 "High" price from a 60-step sliding window of history.

> Note: the source kernel is titled an RNN tutorial and applies LSTM/GRU to a time-series sequence. The same windowed-sequence machinery underpins text generation; here the worked example is numeric sequence forecasting.

## Dataset
- [Stock Time Series 2005-2017](https://www.kaggle.com/datasets/szrlee/stock-time-series-20050101-to-20171231) (`szrlee/stock-time-series-20050101-to-20171231`, ~9 MB).
- **Included in `data/`** — per-ticker CSVs (e.g. `IBM_2006-01-01_to_2018-01-01.csv`) plus combined files. The notebook reads the IBM series and uses the `High` column.
- Train = dates up to 2016; test = 2017 onward.

## Approach
1. **Preprocessing**: select `High` price, `MinMaxScaler` to [0,1], build sliding windows of 60 timesteps → 1 output; reshape to `(samples, 60, 1)`.
2. **LSTM model** (`Sequential`): four `LSTM(50)` layers (first three `return_sequences=True`) each followed by `Dropout(0.2)`, then `Dense(1)`. Optimizer `rmsprop`, loss `mean_squared_error`, 50 epochs, batch size 32.
3. **GRU model** (`Sequential`): four `GRU(50, activation='tanh')` layers with `Dropout(0.2)` and `Dense(1)`. Optimizer `SGD(lr=0.01, decay=1e-7, momentum=0.9)`, loss `mean_squared_error`, 50 epochs, batch size 150.
4. **Evaluation**: inverse-transform predictions and compute RMSE via `return_rmse` (`sqrt(mean_squared_error)`); plot predicted vs. real prices.

## Results
The notebook computes **RMSE** (`math.sqrt(mean_squared_error(test, predicted))`) for both the LSTM and GRU models and plots predicted vs. actual prices, but this folder has no executed run log and the numeric outputs are not committed.

Run the notebook to reproduce the metrics (test-set RMSE for the LSTM model and the GRU model).

## Key Takeaways
- The 60-step sliding window turns a 1-D series into supervised `(X, y)` pairs suitable for RNNs.
- Stacked recurrent layers with `Dropout(0.2)` regularize a deep 4-layer encoder.
- LSTM vs. GRU is a deliberate comparison: GRU has fewer gates/parameters and is often faster to train, with comparable accuracy.
- Scaling with `MinMaxScaler` (and inverse-transforming before RMSE) is essential for stable RNN training on price data.

## How to Run
```bash
pip install tensorflow keras scikit-learn numpy pandas matplotlib
jupyter notebook notebook.ipynb
```
Data is already in `data/`; adjust the read path (`../input/IBM_2006-01-01_to_2018-01-01.csv`) to point at the local `data/` copy if needed.

## Credit
> Based on ["Intro to Recurrent Neural Networks LSTM | GRU"](https://www.kaggle.com/code/thebrownviking20/intro-to-recurrent-neural-networks-lstm-gru) by Siddharth Yadav on Kaggle (3109 votes).
> Dataset: [Stock Time Series 2005-2017](https://www.kaggle.com/datasets/szrlee/stock-time-series-20050101-to-20171231). Adapted and documented for this portfolio.
