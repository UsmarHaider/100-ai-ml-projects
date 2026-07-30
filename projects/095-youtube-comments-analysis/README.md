# YouTube Comments Sentiment Analysis

![Domain](https://img.shields.io/badge/domain-NLP-blue)
![Task](https://img.shields.io/badge/task-Sentiment%20Analysis-green)

## Overview
A lightweight, lexicon-based sentiment analysis of YouTube comments. Each comment is scored
for polarity with TextBlob, bucketed into positive / neutral / negative, and the resulting
distribution is explored and visualized. It is an introductory EDA-style sentiment notebook
rather than a trained classifier.

## Dataset
- **Trending YouTube Video Statistics and Comments** — [Kaggle dataset](https://www.kaggle.com/datasets/datasnaek/youtube) (`datasnaek/youtube`, ~148 MB)
- The notebook reads `UScomments.csv` (`../input/UScomments.csv`) and samples 2,000 random
  comments for analysis.
- Data is **not** committed here. Download with:
  ```bash
  kaggle datasets download datasnaek/youtube
  ```

## Approach
1. **Load** — `pd.read_csv('../input/UScomments.csv', encoding='utf8', error_bad_lines=False)`.
2. **Sample** — draw 2,000 random comments (`data.sample(2000)`) for faster analysis.
3. **Polarity scoring** — `TextBlob(comment).sentiment.polarity` for each comment
   (wrapped in try/except, defaulting to 0 on failure).
4. **Bucketing** — map polarity to categories: `> 0 → 1` (positive), `== 0 → 0` (neutral),
   `< 0 → -1` (negative).
5. **Analysis** — split into positive / negative / neutral subsets, count with
   `value_counts()`, and bar-plot the distribution.

## Results
This notebook performs lexicon-based polarity scoring and exploratory visualization; it
does not train or evaluate a model, so there is no accuracy/F1 metric. The run log for this
project is empty.

| Output | Description |
|---|---|
| Polarity column (`pol`) | TextBlob polarity per comment |
| Category counts | Count of positive / neutral / negative comments |
| Bar chart | Distribution of sentiment categories |

Run the notebook to reproduce the comment-level polarity and the category distribution.

## Key Takeaways
- TextBlob gives a quick, training-free sentiment baseline suitable for EDA.
- The neutral bucket (`polarity == 0`) tends to be large because the lexicon assigns 0 to
  many short or out-of-vocabulary comments.
- A supervised classifier or a transformer model would capture context and sarcasm that the
  lexicon approach misses.

## How to Run
```bash
pip install pandas numpy matplotlib seaborn textblob
python -m textblob.download_corpora
jupyter notebook notebook.ipynb
```
Download the dataset (see Dataset) and place `UScomments.csv` at `../input/UScomments.csv`.

## Credit
> Based on ["Sentiment Analysis of YouTube Comments"](https://www.kaggle.com/code/adepvenugopal/sentiment-analysis-of-youtube-comments) by Venugopal Adep on Kaggle (150 votes).
> Dataset: [Trending YouTube Video Statistics and Comments](https://www.kaggle.com/datasets/datasnaek/youtube). Adapted and documented for this portfolio.
