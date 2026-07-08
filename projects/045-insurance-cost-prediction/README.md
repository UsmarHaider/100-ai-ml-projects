# Medical Insurance Cost Prediction

![ML](https://img.shields.io/badge/domain-Machine%20Learning-blue)
![Regression](https://img.shields.io/badge/task-Regression-orange)

## Overview
Predict individual medical insurance charges from demographic and health
attributes (age, sex, BMI, children, smoker status, region). The notebook is a
hands-on **linear regression** tutorial: it derives the model from the normal
equation, builds it from scratch with NumPy, and verifies it against
scikit-learn.

## Dataset
- **Medical Cost Personal Dataset** — [kaggle.com/datasets/mirichoi0218/insurance](https://www.kaggle.com/datasets/mirichoi0218/insurance)
- 1,338 rows, 6 independent features (`age`, `sex`, `bmi`, `children`,
  `smoker`, `region`) and the target `charges`.
- No missing values.
- Included in `data/` (`insurance.csv`, ~54 KB).

## Approach
1. **EDA**: distribution of charges (right-skewed), violin/box plots vs. sex,
   smoker, region, children; correlation heatmap.
2. **Target transform**: `np.log(charges)` to make the target approximately
   normal (Box-Cox was tried via `scipy.stats.boxcox` but log transform
   performed better).
3. **Encoding**: `pd.get_dummies(drop_first=True)` one-hot encoding of the
   categorical columns (handles the dummy-variable trap).
4. **Split**: `train_test_split(test_size=0.3, random_state=23)`.
5. **Models**:
   - **From-scratch OLS** via the normal equation
     `theta = (XᵀX)⁻¹ Xᵀy` (NumPy `linalg.inv` / `matmul`).
   - **scikit-learn `LinearRegression`** for verification — coefficients match.
6. **Validation**: residual normality, Q-Q plot, homoscedasticity check, and
   Variance Inflation Factor (VIF) for multicollinearity.

## Results
The notebook computes **MSE / J(θ)** and **R²** on the test set (both the
normal-equation and scikit-learn versions). A run log was not captured for this
project, but the notebook's markdown reports:

| Metric | Value |
|--------|-------|
| R² (test set) | ~0.78 (77.95%) |
| VIF | < 5 (no critical multicollinearity) |

Run the notebook to reproduce the exact MSE and R² values from the cell
outputs.

## Key Takeaways
- A log transform of the skewed target meaningfully helps the linear fit.
- `smoker` is the dominant driver of charges; sex and region are weak.
- Residual diagnostics show the linear assumption is only partially met
  (curved actual-vs-predicted, mild heteroscedasticity) — a non-linear model
  (tree/GBM) would likely improve accuracy.
- Hand-coded OLS and scikit-learn agree exactly, validating the math.

## How to Run
```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn
jupyter notebook notebook.ipynb
```
The dataset ships in `data/insurance.csv`. The notebook reads from `../input/`
(Kaggle layout); update the path to `data/insurance.csv` when running locally.

## Credit
> Based on ["Linear Regression Tutorial"](https://www.kaggle.com/code/sudhirnl7/linear-regression-tutorial)
> by Sudhir Kumar on Kaggle (2,932 votes).
> Dataset: [Medical Cost Personal Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance).
> Adapted and documented for this portfolio.
