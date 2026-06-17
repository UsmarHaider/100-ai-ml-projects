# Customer Segmentation with K-Means

![ML](https://img.shields.io/badge/domain-Machine%20Learning-blue) ![Task](https://img.shields.io/badge/task-Clustering%20(Unsupervised)-orange)

## Overview
Unsupervised segmentation of a grocery firm's customers using their personality/spending records. Grouping customers into behavioural segments lets the business tailor products and campaigns. The pipeline cleans and engineers features, reduces dimensionality with PCA, then clusters and profiles the segments.

## Dataset
- [Customer Personality Analysis](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis) (ref `imakash3011/customer-personality-analysis`).
- 2,240 records, 29 columns (tab-separated): demographics (Year_Birth, Education, Marital_Status, Income, Kidhome, Teenhome), enrolment date, recency, spend per category (Wines, Fruits, Meat, Fish, Sweets, Gold), purchase channels, and campaign responses.
- Included in `data/` (`marketing_campaign.csv`, read with `sep="\t"`).

## Approach
1. **Cleaning**: drop rows with missing `Income` (2,240 → 2,216); parse `Dt_Customer` as datetime.
2. **Feature engineering**: derive `Customer_For` (days since enrolment), age, total spend, family composition (`Living_With`), etc.; remove outliers (→ 2,212 rows).
3. **Encoding & scaling**: `LabelEncoder` for categoricals (`Education`, `Living_With`); `StandardScaler` on all features.
4. **Dimensionality reduction**: PCA to 3 components for visualization and clustering input.
5. **Clustering**: `KElbowVisualizer` (yellowbrick) to pick `k`, then `KMeans`; Agglomerative clustering also explored.
6. **Profiling**: characterize each cluster by income, spend and demographics to derive marketing personas.

## Results
The notebook is unsupervised, so there is no accuracy metric. The recorded run confirms the data pipeline: 2,240 → 2,216 (drop NA income) → 2,212 (drop outliers) rows, all features encoded and scaled, with the Elbow method used to choose the cluster count.

Metrics the code computes: the KElbow distortion curve (to select `k`) and cluster profiles. Run the notebook to reproduce the chosen number of clusters and the per-segment profile plots.

## Key Takeaways
- PCA-then-KMeans on scaled features produces interpretable customer segments suitable for targeted marketing.
- The Elbow method drives the choice of `k`; segments are validated by profiling income and category spend.
- Outlier removal and dropping the small set of missing-income rows materially improve cluster quality.

## How to Run
```bash
pip install pandas numpy scikit-learn seaborn matplotlib yellowbrick
jupyter notebook notebook.ipynb
```
Data is already in `data/marketing_campaign.csv` (tab-separated).

## Credit
> Based on ["Customer Segmentation: Clustering 🛍️🛒🛒"](https://www.kaggle.com/code/karnikakapoor/customer-segmentation-clustering) by Karnika Kapoor on Kaggle (7124 votes).
> Dataset: [Customer Personality Analysis](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis). Adapted and documented for this portfolio.
