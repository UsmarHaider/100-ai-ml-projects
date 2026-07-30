# BigMart Sales — EDA & Feature Engineering

![ML](https://img.shields.io/badge/domain-Machine%20Learning-blue)
![Task](https://img.shields.io/badge/task-Regression%20(prep)-green)

## Overview
Explores and prepares the BigMart retail dataset to predict `Item_Outlet_Sales` —
the sales of each product at a given outlet. This notebook focuses on the exploratory
analysis and feature engineering pipeline that produces clean, model-ready train/test
tables for a downstream regression model.

## Dataset
- **BigMart Sales Data** — [Kaggle](https://www.kaggle.com/datasets/brijbhushannanda1979/bigmart-sales-data)
- `Train.csv` and `Test.csv`; ~8,500 training rows, 12 columns including item attributes
  (`Item_Weight`, `Item_Fat_Content`, `Item_Visibility`, `Item_Type`, `Item_MRP`) and
  outlet attributes (`Outlet_Size`, `Outlet_Location_Type`, `Outlet_Type`, establishment year).
- Target: `Item_Outlet_Sales` (continuous).
- Included in `data/` (`Train.csv`, `Test.csv`).

## Approach
1. **Combine** train and test into one frame (with a `source` flag) for consistent processing.
2. **Target analysis** — distribution, skewness and kurtosis of `Item_Outlet_Sales`.
3. **EDA** — categorical frequency counts; boxplots of sales vs `Outlet_Size`,
   `Outlet_Location_Type`, `Outlet_Type`, `Outlet_Identifier`; time view by establishment year.
4. **Missing-value imputation**
   - `Item_Weight` → mean weight per `Item_Identifier` (pivot table).
   - `Outlet_Size` → mode per `Outlet_Type`.
   - `Item_Visibility` zeros → mean visibility for that product.
5. **Feature engineering**
   - `Item_Type_Combined` from `Item_Identifier` prefix (Food / Drinks / Non-Consumable).
   - Consolidate `Item_Fat_Content` labels; mark non-consumables as "Non-Edible".
   - `Outlet_Years = 2009 - Outlet_Establishment_Year`.
6. **Encoding** — `LabelEncoder` + one-hot (`get_dummies`) for categorical fields;
   split back into train/test and export model-ready CSVs.

## Results
This notebook is an **EDA + feature-engineering pipeline**; it does not fit a model or
compute a predictive score. The target (`Item_Outlet_Sales`) is regression-oriented
(RMSE is the standard metric for this problem). Run the notebook to reproduce the cleaned,
encoded train/test files, then attach a regressor (e.g. linear regression, random forest,
or XGBoost) to produce RMSE.

## Key Takeaways
- `Outlet_Type` and `Item_MRP` are the most visually predictive of sales in the boxplots.
- Domain-aware imputation (per-item weight, per-outlet-type size) is cleaner than blanket
  mean/mode fills.
- Deriving `Outlet_Years` and a coarse `Item_Type_Combined` reduces noise from
  high-cardinality identifiers.

## How to Run
```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn
jupyter notebook notebook.ipynb
```
Data is already in `data/` (`Train.csv`, `Test.csv`).

## Credit
> Based on ["BigMart Sales Prediction"](https://www.kaggle.com/code/hiralmshah/bigmart-sales-prediction) by Hiral on Kaggle (351 votes).
> Dataset: [BigMart Sales Data](https://www.kaggle.com/datasets/brijbhushannanda1979/bigmart-sales-data).
> Adapted and documented for this portfolio.
