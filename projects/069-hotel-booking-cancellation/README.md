# Hotel Booking Cancellation Prediction

![ML](https://img.shields.io/badge/domain-Machine%20Learning-blue)
![Classification](https://img.shields.io/badge/task-Binary%20Classification-green)

## Overview
Predict whether a hotel booking will be cancelled (`is_canceled`) from booking
details such as lead time, deposit type, market segment, and guest history. The
notebook does extensive EDA, then benchmarks many classifiers plus a neural
network.

## Dataset
- **Hotel Booking Demand** — [kaggle.com/datasets/jessemostipak/hotel-booking-demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)
- `hotel_bookings.csv`: ~119k bookings across City and Resort hotels with the
  binary target `is_canceled`.
- Included in `data/` (`hotel_bookings.csv`, ~16 MB).

## Approach
1. **Cleaning**: `fillna(0)`, drop leakage / useless columns, handle `adr`
   missing values with the column mean.
2. **EDA**: country-wise guests, price (ADR) by hotel type and month,
   length-of-stay distribution, and feature correlation with `is_canceled`.
3. **Encoding**: separate categorical and numeric frames; encode categoricals
   (dropping `reservation_status_date`, `arrival_date_month`).
4. **Split & scale**: 70/30 `train_test_split`, `StandardScaler`.
5. **Classifiers benchmarked**: Logistic Regression, KNN, Decision Tree,
   Random Forest, AdaBoost (base = Decision Tree), Gradient Boosting, XGBoost
   (`lr=0.1, max_depth=5, n_estimators=180`), CatBoost, Extra Trees, LightGBM,
   and a soft `VotingClassifier`.
6. **Deep learning**: a Keras ANN trained on the one-hot target
   (`to_categorical`).
7. **Metrics**: accuracy, confusion matrix, and classification report per model.

## Results
The captured Kaggle run log recorded the Keras ANN
training, which reached:

| Metric | Value |
|--------|-------|
| **ANN validation accuracy** | **~0.9934 (99.34%)** |
| ANN final training accuracy | ~0.9934 |

(Kernel title advertises **99.5% accuracy**.) The sklearn classifier accuracy
prints were not captured in the log — run the notebook to reproduce the
per-model accuracy table and confusion matrices.

## Key Takeaways
- Lead time, deposit type, and previous cancellations are strong cancellation
  signals.
- Tree ensembles (Random Forest, XGBoost, CatBoost, LightGBM) and the ANN reach
  very high accuracy on this dataset.
- Caution: some columns (e.g. `reservation_status`) can leak the target and are
  dropped — keep an eye on leakage when interpreting the ~99% figure.

## How to Run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost catboost lightgbm tensorflow keras
jupyter notebook notebook.ipynb
```
The dataset ships in `data/hotel_bookings.csv`. The notebook reads from
`../input/hotel-booking-demand/`; update the path when running locally.

## Credit
> Based on ["Hotel Booking Prediction (99.5% acc)"](https://www.kaggle.com/code/niteshyadav3103/hotel-booking-prediction-99-5-acc)
> by Nitesh Yadav on Kaggle (1,504 votes).
> Dataset: [Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand).
> Adapted and documented for this portfolio.
