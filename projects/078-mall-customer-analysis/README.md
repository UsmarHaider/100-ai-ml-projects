# Mall Customer Segmentation — KMeans Clustering

![ML](https://img.shields.io/badge/domain-Machine%20Learning-blue)
![Task](https://img.shields.io/badge/task-Clustering-orange)

## Overview
Segments mall customers into actionable groups using K-Means clustering on demographic
and behavioural attributes. Customer segmentation lets a retailer target marketing and
loyalty programs at distinct spender profiles rather than treating all shoppers alike.

## Dataset
- **Mall Customer Segmentation Data** — [Kaggle](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)
- 200 customers, 5 columns: `CustomerID`, `Gender`, `Age`, `Annual Income (k$)`,
  `Spending Score (1-100)`. No missing values (confirmed in the run log).
- Included in `data/` (`Mall_Customers.csv`).

## Approach
1. **EDA** — distribution plots for `Age`, `Annual Income`, `Spending Score`; pairplot by
   gender; scatter plots of feature pairs.
2. **Elbow method** — fit `KMeans` for `k = 1..14` and plot inertia to find the elbow,
   repeated for each feature set.
3. **Clustering models** (`KMeans`, `init='k-means++'`, `n_init=10`, `max_iter=300`,
   `tol=1e-4`, `algorithm='elkan'`, `random_state=111`):
   - Age vs Spending Score → **k = 4** (and a k = 5 variant)
   - Annual Income vs Spending Score → **k = 5**
   - Age + Annual Income + Spending Score (3-D) → **k = 6**
4. **Visualization** — decision-boundary mesh plots in 2-D and a 3-D Plotly scatter of the
   six-cluster solution.

## Results
This is an unsupervised clustering study; the notebook does not compute a labelled accuracy
metric. The clustering is selected via the **elbow method** (inertia vs k) and visualized.
The run log confirms the dataset shape (200 rows × 5 columns, no nulls).

Reported cluster counts:

| Feature set | Chosen k |
|---|---|
| Age vs Spending Score | 4 (also 5) |
| Annual Income vs Spending Score | 5 |
| Age + Income + Spending (3-D) | 6 |

Run the notebook to reproduce the inertia curves and cluster plots.

## Key Takeaways
- The Income-vs-Spending segmentation yields the classic 5 groups (e.g. high-income/high-spend,
  high-income/low-spend, etc.) that are most useful for marketing.
- The elbow method gives a clear, interpretable choice of `k` for each feature pairing.
- A silhouette score could be added to quantitatively compare candidate `k` values.

## How to Run
```bash
pip install numpy pandas matplotlib seaborn plotly scikit-learn
jupyter notebook notebook.ipynb
```
Data is already in `data/Mall_Customers.csv`.

## Credit
> Based on ["KMeans Clustering for Customer Data"](https://www.kaggle.com/code/heeraldedhia/kmeans-clustering-for-customer-data) by Heeral Dedhia on Kaggle (1,063 votes).
> Dataset: [Customer Segmentation Tutorial in Python](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python).
> Adapted and documented for this portfolio.
