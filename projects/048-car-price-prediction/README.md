# Used Car Price Prediction

![ML](https://img.shields.io/badge/domain-Machine%20Learning-blue)
![Regression](https://img.shields.io/badge/task-Regression-orange)

## Overview
A Chinese automaker (Geely Auto) wants to understand which factors drive car
prices in the US market. This notebook builds an interpretable **multiple
linear regression** model and uses **Recursive Feature Elimination (RFE)** plus
manual p-value / VIF pruning with `statsmodels` to find the significant
predictors.

## Dataset
- **Car Price Assignment data** — [kaggle.com/datasets/goyalshalini93/car-data](https://www.kaggle.com/datasets/goyalshalini93/car-data)
- 205 rows, 26 columns (mix of categorical and numeric car specs).
- Included in `data/` (`CarPrice_Assignment.csv` + the data dictionary
  `Data Dictionary - carprices.xlsx`).

## Approach
1. **Cleaning**: split brand out of `CarName`, fix misspelled brand names
   (e.g. `maxda`→`mazda`, `vw`→`volkswagen`), check for duplicates.
2. **EDA**: price distribution (right-skewed), categorical vs. price box plots,
   numeric scatter/pair plots, correlation heatmap.
3. **Feature engineering**:
   - `fueleconomy = 0.55*citympg + 0.45*highwaympg`.
   - `carsrange` bucket (Budget / Medium / Highend) from average price per
     company.
4. **Encoding**: `pd.get_dummies(drop_first=True)` for all categoricals.
5. **Split & scale**: 70/30 split, `MinMaxScaler` on numeric columns.
6. **Modeling** (iterative):
   - `sklearn` `LinearRegression` + `RFE` to select 10 features.
   - `statsmodels` OLS rebuilt 7 times, dropping features by **p-value > 0.05**
     and high **VIF** at each step.
7. **Diagnostics**: residual histogram (approx. normal), y_test vs. y_pred
   scatter, final OLS summary.

## Results
No run log was captured for this project. The notebook's markdown reports the
final OLS fit statistics:

| Metric | Value |
|--------|-------|
| R-squared (train) | 0.899 |
| Adjusted R-squared (train) | 0.896 |
| F-statistic | 308.0 (Prob ≈ 0.0) |
| Test R² | computed via `r2_score(y_test, y_pred)` |

Run the notebook to reproduce the exact `r2_score` on the test set.

## Key Takeaways
- ~90% of price variance is explained by a handful of features; the model is
  fit for interpretation, not just prediction.
- `curbweight`, `enginesize`, `horsepower`, `carwidth` and the `highend`
  bucket are the strongest price drivers.
- Dropping high-VIF / high-p-value terms (curbweight, sedan, wagon, etc.) keeps
  the model parsimonious and avoids multicollinearity.

## How to Run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels
jupyter notebook notebook.ipynb
```
The dataset ships in `data/`. The notebook reads from `../input/`; update the
path to `data/CarPrice_Assignment.csv` when running locally.

## Credit
> Based on ["Car Price Prediction (Linear Regression - RFE)"](https://www.kaggle.com/code/goyalshalini93/car-price-prediction-linear-regression-rfe)
> by Shalini Goyal on Kaggle (4,289 votes).
> Dataset: [Car data](https://www.kaggle.com/datasets/goyalshalini93/car-data).
> Adapted and documented for this portfolio.
