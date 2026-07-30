# Airline Tweets Sentiment Analysis

![Domain](https://img.shields.io/badge/domain-NLP-blue)
![Task](https://img.shields.io/badge/task-Sentiment%20Classification-green)

## Overview
Sentiment classification of US-airline tweets (Feb 2015) using a linear Support Vector
Machine. After EDA of sentiment distribution and negative-reason breakdowns per airline,
the neutral class is dropped and a binary negative-vs-positive classifier is trained on
bag-of-words features.

## Dataset
- **Twitter US Airline Sentiment** — [Kaggle dataset](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)
  (also [bansodesandeep/us-airline-data](https://www.kaggle.com/datasets/bansodesandeep/us-airline-data))
- **14,640** tweets, 15 columns; label column `airline_sentiment`
  (negative / neutral / positive) across 6 airlines.
- **Included in `data/`**: `Tweets.csv` and `database.sqlite`.

## Approach
1. **EDA** — sentiment counts per airline, percent-negative per airline, negative-reason
   frequencies, time trends, and positive/negative word clouds.
2. **Cleaning** — `tweet_to_words()`: strip URLs/`@mentions`/`RT`, lowercase, regex
   tokenization, NLTK stop-word removal.
3. **Label reduction** — drop all `neutral` rows; `LabelEncoder` on the remaining
   negative/positive labels.
4. **Vectorization** — `CountVectorizer` (a tuned variant uses
   `stop_words='english', ngram_range=(1,2), min_df=0.1, max_df=0.7, max_features=100`).
5. **Model** — `SVC(kernel='linear', random_state=10)` trained with a default
   `train_test_split` (≈75/25), evaluated with `classification_report` and a confusion
   matrix.

## Results
Real values extracted from the captured run log on the
binary (negative vs positive) test split of 2,886 tweets:

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| negative | 0.94 | 0.94 | 0.94 | 2323 |
| positive | 0.77 | 0.76 | 0.76 | 563 |
| **accuracy** | | | **0.91** | 2886 |
| macro avg | 0.85 | 0.85 | 0.85 | 2886 |
| weighted avg | 0.91 | 0.91 | 0.91 | 2886 |

Train/test sizes from the log: 8,655 train / 2,886 test (after dropping neutral).

## Key Takeaways
- The linear SVM reaches **91% accuracy** on the binary task; the negative class (majority)
  is predicted very reliably (F1 0.94).
- Class imbalance hurts the minority positive class (F1 0.76) — class weighting or
  resampling could help.
- Dropping the neutral class simplifies the problem; a full 3-class model would be a
  harder, more useful extension.

## How to Run
```bash
pip install pandas numpy scikit-learn nltk seaborn matplotlib wordcloud mlxtend
python -c "import nltk; nltk.download('stopwords')"
jupyter notebook notebook.ipynb
```
Data is already in `data/`; update the read path if needed (the notebook reads
`../input/twitter-airline-sentiment/Tweets.csv`).

## Credit
> Based on ["Sentiment Analysis: Support Vector Machine"](https://www.kaggle.com/code/bansodesandeep/sentiment-analysis-support-vector-machine) by Sandeep Bansode on Kaggle (422 votes).
> Datasets: [Twitter US Airline Sentiment](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment), [US Airline Data](https://www.kaggle.com/datasets/bansodesandeep/us-airline-data). Adapted and documented for this portfolio.
