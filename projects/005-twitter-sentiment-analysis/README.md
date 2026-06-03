# Sentiment Analysis: VADER vs. RoBERTa

![NLP](https://img.shields.io/badge/domain-NLP-green)
![Task](https://img.shields.io/badge/task-sentiment%20analysis-orange)
![Model](https://img.shields.io/badge/model-VADER%20%2B%20RoBERTa-purple)

## Overview
Compare two sentiment-analysis approaches on text reviews: a lexicon/bag-of-words scorer (**VADER**) and a transformer (**RoBERTa**) fine-tuned for tweet sentiment. The notebook scores the same reviews with both, then visualizes where the rule-based and context-aware models disagree — a great illustration of why transformers capture context that bag-of-words misses.

## Dataset
- **Amazon Fine Food Reviews** — [kaggle.com/datasets/snap/amazon-fine-food-reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)
- ~568,454 reviews; the notebook samples the **first 500** (`df.head(500)`) for speed, using the `Text` and `Score` (1–5 stars) columns.
- Data is **not** committed. Download with:
  ```bash
  kaggle datasets download snap/amazon-fine-food-reviews
  unzip amazon-fine-food-reviews.zip
  ```
  The notebook reads `../input/amazon-fine-food-reviews/Reviews.csv`.

## Approach
1. **NLTK basics** — tokenize, POS-tag, and named-entity-chunk an example review.
2. **VADER** — NLTK `SentimentIntensityAnalyzer` produces neg/neu/pos/compound scores for every review (bag-of-words, stopwords removed).
3. **RoBERTa** — `cardiffnlp/twitter-roberta-base-sentiment` via Hugging Face Transformers; tokenize, run the model, `softmax` the logits into `roberta_neg/neu/pos`.
4. **Combine & compare** — merge VADER and RoBERTa scores into one DataFrame; `seaborn` barplots and pairplots compare scores against the star rating; inspect 1-star reviews scored positive and 5-star reviews scored negative.
5. **Extra** — the Hugging Face `pipeline("sentiment-analysis")` for quick predictions.

## Results
This is an exploratory comparison rather than a benchmarked classifier — it produces score distributions and comparison plots, not a single headline accuracy. From the run log, RoBERTa scores align strongly with sentiment, e.g. for a negative example review:

| Model | neg | neu | pos |
| --- | --- | --- | --- |
| RoBERTa | 0.976 | 0.021 | 0.003 |

The run log shows two reviews skipped (`Broke for id 83`, `Broke for id 187`) — long texts that exceed RoBERTa's token limit. Run the notebook to reproduce the full comparison plots.

## Key Takeaways
- RoBERTa accounts for word **context**, so it handles negation and sarcasm better than VADER's bag-of-words scoring.
- Plotting model scores against star ratings reveals systematic disagreements between the two methods.
- VADER is instant and dependency-light; RoBERTa is heavier but more accurate.
- Inputs longer than RoBERTa's max length raise `RuntimeError` and must be truncated or skipped (as the notebook does with a try/except).

## How to Run
```bash
pip install pandas numpy matplotlib seaborn nltk transformers torch scipy tqdm
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt'); nltk.download('averaged_perceptron_tagger'); nltk.download('maxent_ne_chunker'); nltk.download('words')"
jupyter notebook notebook.ipynb
```
Download the Amazon Fine Food Reviews dataset (see Dataset). The RoBERTa weights download automatically from Hugging Face on first run.

## Credit
> Based on ["Sentiment Analysis Python 🤗 [Youtube Tutorial]"](https://www.kaggle.com/code/robikscube/sentiment-analysis-python-youtube-tutorial) by Rob Mulla on Kaggle (5,984 votes).
> Dataset: [Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews). Model: [cardiffnlp/twitter-roberta-base-sentiment](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment). Adapted and documented for this portfolio.
