# Unsupervised Anomaly Detection (Time Series)

![ML](https://img.shields.io/badge/domain-Machine%20Learning-blue)
![Anomaly Detection](https://img.shields.io/badge/task-Anomaly%20Detection-red)

## Overview
A comparative walkthrough of **unsupervised anomaly-detection** techniques on a
real time-series signal (ambient temperature with a known system failure). The
notebook implements and visualizes six different approaches to understand what
each method detects and where it fails.

## Dataset
- **Numenta Anomaly Benchmark (NAB)** — [kaggle.com/datasets/boltzmannbrain/nab](https://www.kaggle.com/datasets/boltzmannbrain/nab)
- Uses `realKnownCause/ambient_temperature_system_failure.csv`
  (timestamped temperature readings).
- Included in `data/` (the full NAB collection: `realKnownCause`,
  `realAWSCloudwatch`, `realTraffic`, `artificialWithAnomaly`, etc.).

## Approach
1. **Preprocessing**: parse timestamps, convert °F→°C.
2. **Feature engineering**: `hours`, `daylight` (7–22h), `DayOfTheWeek`,
   `WeekDay`, and a 4-way `categories` feature (weekday/weekend × day/night).
   Assumed `outliers_fraction = 0.01`.
3. **Models implemented**:
   - **K-Means clustering** (15 centroids via elbow method) on PCA-reduced
     (2-component) standardized features; points far from their centroid are
     flagged.
   - **EllipticEnvelope (Gaussian)** fit per category.
   - **Markov chain** over 5 discretized temperature states (code included;
     `pyemma` not available on Kaggle so it is commented out).
   - **Isolation Forest** (`contamination = outliers_fraction`).
   - **One-Class SVM** (`nu = 0.95 * outliers_fraction`).
   - **RNN** (LSTM) — learns from 50 previous points to predict the next; large
     prediction error ⇒ anomaly.

## Results
This is an **unsupervised, exploratory** notebook — it produces labeled-anomaly
visualizations rather than a single scored metric. The captured run log
 contains only nbconvert/TensorFlow build messages, no
metric values.

Each method reports an **anomaly count** (`value_counts()` of its flag column)
and overlays detected anomalies on the time series. Run the notebook to
reproduce the per-method anomaly counts and plots.

## Key Takeaways
- K-Means flags abnormally low/high temperatures but misses the extreme peak.
- Category-wise Gaussian envelopes add context sensitivity and catch extremes.
- Isolation Forest and One-Class SVM give similar collective-anomaly results;
  SVM additionally flags some mid-range values.
- Markov chains target *sequential* anomalies (unusual transitions) rather than
  extreme values — complementary to the others.
- No single method is universally best; the right choice depends on whether
  anomalies are point, contextual, or sequential.

## How to Run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow keras
jupyter notebook notebook.ipynb
```
NAB data ships in `data/`. The notebook reads from `../input/`; update paths to
`data/realKnownCause/...` when running locally. (Markov chains additionally
require `pyemma`, which is optional.)

## Credit
> Based on ["Unsupervised Anomaly Detection"](https://www.kaggle.com/code/victorambonati/unsupervised-anomaly-detection)
> by Victor Ambonati on Kaggle (1,445 votes).
> Dataset: [Numenta Anomaly Benchmark (NAB)](https://www.kaggle.com/datasets/boltzmannbrain/nab).
> Adapted and documented for this portfolio.
