# World Happiness Report 2021 — EDA & Clustering

![ML](https://img.shields.io/badge/domain-Machine%20Learning-blue)
![Task](https://img.shields.io/badge/task-EDA%20%2B%20Clustering-orange)

## Overview
An in-depth exploratory data analysis of the 2021 World Happiness Report, enriched with
country population data, to understand what makes a country happy or unhappy. The notebook
is built around high-quality, custom-styled visualizations and finishes with an unsupervised
clustering of countries by their happiness drivers.

## Dataset
- **World Happiness Report 2021** — [Kaggle](https://www.kaggle.com/datasets/ajaypalsinghlo/world-happiness-report-2021)
  (also the historic 2005+ scores).
- **Population by Country 2020** — [Kaggle](https://www.kaggle.com/datasets/tanuprabhu/population-by-country-2020)
  (merged in for population/density/median-age context).
- Features include `Ladder score`, `Logged GDP per capita`, `Social support`,
  `Healthy life expectancy`, `Freedom to make life choices`, `Generosity`,
  `Perceptions of corruption`, plus merged population, density and median age.
- Included in `data/` (`world-happiness-report-2021.csv`, `world-happiness-report.csv`,
  `population_by_country_2020.csv`).

## Approach
1. **EDA** — happiest/unhappiest countries (mean Ladder score), temporal trends since 2005,
   region/continent comparisons, and the relationship of happiness to GDP, life expectancy,
   freedom, corruption, population density and median age.
2. **Custom visualization** — styled Matplotlib/Seaborn (gridspec, themed backgrounds),
   GeoPandas world map, slope charts, waffle chart, and correlation heatmap.
3. **Feature merge** — join happiness data with 2020 population data on country (alpha-3 codes).
4. **Clustering** — `StandardScaler` → `KMeans(n_clusters=3)` on the happiness driver features
   (Ladder score, GDP, social support, life expectancy, freedom, generosity, corruption),
   with `PCA` available for visualization; mean Ladder score reported per cluster.

## Results
This is primarily an **exploratory + unsupervised** notebook with no labelled prediction
metric, and the captured run log is empty (no executed numeric outputs captured). The
clustering produces **3 country clusters** whose mean `Ladder score` differs (high / middle /
low happiness tiers). The narrative findings (from the conclusion cell): happier countries
tend to be wealthier, less populous, older, less corrupt, more free, and to have stronger
social support.

**Run the notebook to reproduce** the per-cluster mean Ladder scores and all visualizations.

## Key Takeaways
- GDP per capita and healthy life expectancy are the clearest correlates of happiness;
  Western Europe and North America/ANZ lead, Sub-Saharan Africa and South Asia trail.
- Temporal analysis shows "the unhappy tend to stay unhappy," with some regions trending down.
- K-Means on the standardized happiness drivers cleanly separates countries into 3 tiers,
  validating the EDA narrative quantitatively.

## How to Run
```bash
pip install numpy pandas matplotlib seaborn scikit-learn geopandas pycountry pywaffle
jupyter notebook notebook.ipynb
```
Data is already in `data/` (three CSVs listed above).

## Credit
> Based on ["[Awesome EDA] 2021 Happiness & Population"](https://www.kaggle.com/code/joshuaswords/awesome-eda-2021-happiness-population) by Josh on Kaggle (701 votes).
> Datasets: [World Happiness Report 2021](https://www.kaggle.com/datasets/ajaypalsinghlo/world-happiness-report-2021) ·
> [Population by Country 2020](https://www.kaggle.com/datasets/tanuprabhu/population-by-country-2020).
> Adapted and documented for this portfolio.
