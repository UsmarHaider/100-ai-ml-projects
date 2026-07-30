# Stock Market News Sentiment Analysis

![Domain](https://img.shields.io/badge/domain-NLP%20%2F%20Finance-blue)
![Task](https://img.shields.io/badge/task-Time--series%20Prediction-green)

## Overview
A starter kernel for the **Two Sigma: Using News to Predict Stock Movements** competition.
It demonstrates how to use the competition's custom `twosigmanews` Python module to access
market and news data, iterate through trading days, and submit confidence-value predictions
without leaking future data. It is a scaffolding/getting-started notebook rather than a
finished model.

## Dataset
- **Two Sigma Financial News** — [Kaggle competition](https://www.kaggle.com/competitions/two-sigma-financial-news)
- Data is **only** accessible inside the Kaggle kernel environment through the
  `kaggle.competitions.twosigmanews` module (it cannot be downloaded as plain files).
- Provides `market_train_df` and `news_train_df` for training, plus per-day observation
  frames during prediction.

## Approach
1. **Environment setup** — `from kaggle.competitions import twosigmanews; env = twosigmanews.make_env()`.
2. **Load training data** — `env.get_training_data()` returns market and news DataFrames.
3. **Prediction loop** — iterate `env.get_prediction_days()`, which yields
   `(market_obs_df, news_obs_df, predictions_template_df)` per trading day.
4. **Predict** — fill the template's `confidenceValue` column (the starter uses random
   values in `[-1, 1]`) and call `env.predict(predictions_df)`.
5. **Submit** — `env.write_submission_file()` writes `submission.csv`.

The starter intentionally uses random predictions as a placeholder; the real modelling
(`train_my_model` / `make_my_predictions`) is left to the user.

## Results
This is an official getting-started scaffold and produces no predictive metric — the
baseline predictions are random. The run log for this project is empty.

The competition is scored on a custom Sharpe-like metric of
`confidenceValue * marketReturn` summed across assets and days. Implement a real model
and submit through the Kaggle kernel to obtain a leaderboard score.

## Key Takeaways
- The `twosigmanews` API enforces no-future-leakage by streaming one day at a time; you
  can only iterate `get_prediction_days()` once per session.
- Predictions are continuous confidence values in `[-1, 1]`, not hard labels.
- Natural next steps: join news sentiment features onto market rows and replace the random
  generator with a trained regressor/classifier.

## How to Run
This notebook must run inside the Kaggle kernel for the
[Two Sigma Financial News competition](https://www.kaggle.com/competitions/two-sigma-financial-news),
because the `twosigmanews` module and its data are only available there.
```bash
# inside the Kaggle kernel environment
pip install numpy
jupyter notebook notebook.ipynb
```

## Credit
> Based on ["Two Sigma News Official Getting Started Kernel"](https://www.kaggle.com/code/dster/two-sigma-news-official-getting-started-kernel) by DJ Sterling on Kaggle (3933 votes).
> Dataset: [Two Sigma Financial News](https://www.kaggle.com/competitions/two-sigma-financial-news). Adapted and documented for this portfolio.
