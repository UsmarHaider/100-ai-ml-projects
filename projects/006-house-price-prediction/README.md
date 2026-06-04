# House Price Prediction

![Domain](https://img.shields.io/badge/domain-Machine%20Learning-blue)
![Task](https://img.shields.io/badge/task-Regression-green)

## Overview
Predicts residential house sale prices using **TensorFlow Decision Forests (TF-DF)**. A
Random Forest regressor is trained on the Ames housing data with no manual preprocessing —
TF-DF natively handles numeric, categorical, and missing features — making it a strong
tabular baseline.

## Dataset
- **House Prices - Advanced Regression Techniques** — [Kaggle competition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
- **1,460** training rows × **81** columns (79 features + `Id` + `SalePrice` label);
  mixed numeric/categorical with missing values.
- Data is **not** committed. Download via the competition page:
  ```bash
  kaggle competitions download -c house-prices-advanced-regression-techniques
  ```
  The notebook reads `../input/house-prices-advanced-regression-techniques/{train,test}.csv`.

## Approach
1. **Load & inspect** — read `train.csv`, drop `Id`, explore `SalePrice` and numeric-feature
   distributions.
2. **Split** — custom `split_dataset(test_ratio=0.30)` into train/validation Pandas frames,
   then convert to `tf.data.Dataset` with `tfdf.keras.pd_dataframe_to_tf_dataset(..., task=REGRESSION)`.
3. **Model** — `tfdf.keras.RandomForestModel(task=REGRESSION)`, compiled with
   `metrics=["mse"]`; no feature preprocessing required.
4. **Evaluate** — out-of-bag RMSE vs number of trees via `make_inspector().training_logs()`;
   validation metrics via `rf.evaluate(valid_ds, return_dict=True)`.
5. **Predict & submit** — predict on `test.csv` and write `submission.csv` with
   `Id`/`SalePrice`.

## Results
The notebook reports **RMSE** (out-of-bag, and on the validation split via `evaluate`) plus
the model inspector's variable importances. The run log for this project is empty, so no
executed RMSE value was captured here.

| Metric | Where reported |
|---|---|
| Out-of-bag RMSE | `make_inspector().training_logs()` (plotted vs #trees) |
| Validation RMSE / MSE | `rf.evaluate(valid_ds, return_dict=True)` |
| Variable importances | `inspector.variable_importances()` |

Run the notebook to reproduce the RMSE values and the competition submission.

## Key Takeaways
- TF-DF Random Forests handle raw mixed-type tabular data with **no preprocessing**, an
  ideal first baseline before neural networks.
- Out-of-bag RMSE gives a built-in validation signal without holding out extra data.
- Switching to `GradientBoostedTreesModel` or tuning hyperparameters is the natural next
  step for leaderboard improvement.

## How to Run
```bash
pip install tensorflow tensorflow_decision_forests pandas seaborn matplotlib
jupyter notebook notebook.ipynb
```
Download the competition data (see Dataset) and place the CSVs under
`../input/house-prices-advanced-regression-techniques/`.

## Credit
> Based on ["House Prices Prediction using TFDF"](https://www.kaggle.com/code/gusthema/house-prices-prediction-using-tfdf) by Gusthema on Kaggle (15022 votes).
> Dataset: [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques). Adapted and documented for this portfolio.
