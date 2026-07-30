# Song Lyrics Analysis & Generation

![NLP](https://img.shields.io/badge/domain-NLP-blue) ![Task](https://img.shields.io/badge/task-Text%20Generation%20%26%20Sentiment-orange)

## Overview
Analyze and generate lyrics in the style of Portishead. The notebook combines sentiment analysis (TextBlob polarity) and visualization with a character-of-text-style **LSTM language model** trained on the band's lyrics to generate new lines word-by-word.

## Dataset
- **Portishead Songs Dataset**: https://www.kaggle.com/datasets/cluesec/portishead-songs-dataset
- A small corpus of 71 tracks; the lyrics file has columns `track_name` and `lyrics_text` (61 non-null lyric entries).
- Included in `data/` (`portishead_lyrics.csv`, `portishead_data.csv`).

## Approach
1. **Cleaning** — lowercase and strip punctuation/non-word characters from each track's lyrics.
2. **Sentiment analysis** — TextBlob polarity per song; plot the polarity distribution and an interactive Plotly scatter of sentiment vs. word count; word cloud of frequent terms.
3. **Sequence preparation** — Keras `Tokenizer` over the cleaned lyrics; build n-gram input sequences, pad to the max sequence length (`padding='pre'`); split into predictors and one-hot next-word labels.
4. **Language model** — `Sequential([Embedding(total_words, 64), LSTM(100), Dense(total_words, softmax)])`, compiled with Adam and categorical cross-entropy.
5. **Training & generation** — fit for 50 epochs; generate lyrics by iteratively predicting the next word from a seed text.

## Results
Values extracted from the captured run log (final training epoch, 50/50):

| Metric | Value |
| --- | --- |
| Training accuracy (next-word) | **~0.9745** |
| Training loss | **~0.1186** |

These are **training-set** next-word metrics on a very small corpus (no held-out validation split). The model also produces generated lyric text from a random seed (the log shows a "Seed Text" → generated continuation). The TextBlob sentiment outputs are visualizations rather than scored metrics. **Run the notebook to reproduce the generated samples and sentiment plots.**

## Key Takeaways
- A compact Embedding→LSTM→Dense stack can memorize/model a small single-artist corpus quickly (high training accuracy by 50 epochs).
- With only 71 tracks and no validation split, the high training accuracy reflects fit/memorization rather than measured generalization.
- TextBlob gives a quick, dependency-light sentiment signal for relating emotional tone to lyric length.

## How to Run
```bash
pip install pandas numpy matplotlib seaborn plotly wordcloud tensorflow textblob
jupyter notebook notebook.ipynb
```
Data is already in `data/`.

## Credit
> Based on ["Sentiment Analysis and Lyrics Generation with LSTM"](https://www.kaggle.com/code/cluesec/sentiment-analysis-and-lyrics-generation-with-lstm) by ClueSec on Kaggle (49 votes).
> Dataset: [Portishead Songs Dataset](https://www.kaggle.com/datasets/cluesec/portishead-songs-dataset). Adapted and documented for this portfolio.
