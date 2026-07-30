# Student Performance Prediction (Game-Play)

![ML](https://img.shields.io/badge/domain-Machine%20Learning-blue)
![Classification](https://img.shields.io/badge/task-Binary%20Classification-green)

## Overview
Predict whether students answer each of 18 in-game checkpoint questions
correctly, from their game-play event logs (the *Predict Student Performance
from Game Play* competition). The notebook is an **XGBoost baseline** that
trains one model per question with GroupKFold cross-validation and optimizes a
shared F1 threshold.

## Dataset
- **Predict Student Performance from Game Play** competition —
  [kaggle.com/competitions/predict-student-performance-from-game-play](https://www.kaggle.com/competitions/predict-student-performance-from-game-play)
- `train.csv` (~4.7 GB of event logs) + `train_labels.csv` (correct/incorrect
  per question per session). Inference uses the competition's `jo_wilder` API.
- Not committed to this repo. Accept the rules, then:
  ```bash
  kaggle competitions download -c predict-student-performance-from-game-play
  ```

## Approach
1. **Chunked loading**: read `train.csv` in **10 pieces** to avoid memory
   errors, feature-engineering each piece before reading the next.
2. **Feature engineering** per `(session_id, level_group)`:
   - `nunique` of categoricals (`event_name`, `fqid`, `room_fqid`, `text`);
   - `mean` and `std` of numerics (`elapsed_time`, `level`, coords,
     `hover_duration`, …);
   - `sum` of 9 engineered event-type flags + `elapsed_time`.
   Result: **31 features** over **23,562 users**.
3. **Modeling**: an `XGBClassifier` per question (1–18), trained on the matching
   `level_group` (`0-4`→Q1-3, `5-12`→Q4-13, `13-22`→Q14-18). Params:
   `learning_rate=0.05`, `max_depth=4`, `n_estimators=1000`,
   `early_stopping_rounds=50`, `subsample=0.8`, `colsample_bytree=0.4`,
   `tree_method='hist'`.
4. **CV**: 5-fold `GroupKFold` grouped by session; out-of-fold predictions
   collected for all questions.
5. **Threshold search**: sweep 0.40–0.80 to maximize macro F1, then apply the
   optimal threshold; inference via the `jo_wilder` iterator API.

## Results
Real metrics from the captured Kaggle run log:

| Metric | Value |
|--------|-------|
| **Overall CV macro F1** | **0.6792** |
| Best F1 question (Q3) | 0.6221 |
| Lowest F1 question (Q12) | 0.4354 |
| Test submission mean (`correct`) | 0.667 |

(Kernel title references the public LB ≈ 0.680.)

## Key Takeaways
- Per-question models trained on the correct `level_group` match the
  competition's staged inference API.
- A single tuned F1 threshold across all questions is a simple, effective
  baseline.
- Easy headroom: more features, more KFold models, or sequence models
  (RNN/Transformer) on the raw event stream.

## How to Run
```bash
pip install pandas numpy scikit-learn xgboost
jupyter notebook notebook.ipynb
```
This notebook depends on the competition's `jo_wilder` inference module and is
designed to run inside a Kaggle kernel. Download the competition data (command
above) for local feature engineering.

## Credit
> Based on ["XGBoost Baseline - [0.680]"](https://www.kaggle.com/code/cdeotte/xgboost-baseline-0-680)
> by Chris Deotte on Kaggle (1,306 votes).
> Competition: [Predict Student Performance from Game Play](https://www.kaggle.com/competitions/predict-student-performance-from-game-play).
> Adapted and documented for this portfolio.
