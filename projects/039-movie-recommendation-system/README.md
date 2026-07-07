# Movie Recommendation System (Netflix Prize)

![ML](https://img.shields.io/badge/domain-Machine%20Learning-blue) ![Task](https://img.shields.io/badge/task-Recommender%20System-red)

## Overview
Builds movie recommendations from the Netflix Prize ratings data using two complementary techniques: matrix-factorization collaborative filtering (SVD) and item-item Pearson correlation. The main engineering challenge is loading and slicing the very large ratings dataset efficiently.

## Dataset
- [Netflix Prize Data](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data) (ref `netflix-inc/netflix-prize-data`).
- Four `combined_data_*.txt` files, each ~20M+ rows; together 17K+ movies and 500K+ customers. Plus `movie_titles.csv` (Movie_Id, Year, Name).
- **Not committed** (~2 GB). Download with:
  ```bash
  kaggle datasets download netflix-inc/netflix-prize-data
  ```

## Approach
1. **Load & reshape**: read `combined_data_1.txt` (others optional) into `Cust_Id` / `Rating`; detect the `movie_id:` separator rows and forward-fill a `Movie_Id` column.
2. **Data cleaning / slicing**: drop sparse movies and inactive customers using count/mean benchmarks to make the data tractable in memory.
3. **Pivot**: build a customer × movie ratings matrix (`pd.pivot_table`).
4. **Models**:
   - **Collaborative filtering** with the `surprise` library: `Reader` + `Dataset.load_from_df`, `SVD` matrix factorization, evaluated by `cross_validate` (RMSE, MAE), then fit on the full trainset to estimate a target user's scores.
   - **Pearson's R correlation**: `df_p.corrwith(target_movie)` to surface the top-10 most-correlated movies, joined back to titles and filtered by minimum rating count.

## Results
The model evaluates SVD via 5-fold `cross_validate` reporting **RMSE** and **MAE**, and produces ranked top-10 recommendation lists (per-user SVD estimates and per-movie Pearson correlations). No run log was captured for this project, so no executed metric values are available. Run the notebook to reproduce the SVD RMSE/MAE and the recommendation tables.

## Key Takeaways
- Memory management (dropping sparse users/movies, careful dtypes) is the real bottleneck on Netflix-scale data.
- SVD collaborative filtering and correlation-based item similarity are complementary: one personalizes per user, the other finds "similar to this movie".
- Filtering by a minimum rating count avoids spurious high correlations from rarely-rated movies.

## How to Run
```bash
pip install pandas numpy scipy matplotlib seaborn scikit-surprise
# download the dataset (see above), place under ../input or adjust read paths
jupyter notebook notebook.ipynb
```

## Credit
> Based on ["Netflix - Movie recommendation"](https://www.kaggle.com/code/laowingkin/netflix-movie-recommendation) by DLao on Kaggle (3420 votes).
> Dataset: [Netflix Prize Data](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data). Adapted and documented for this portfolio.
