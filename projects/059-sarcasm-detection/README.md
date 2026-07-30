# Sarcasm Detection in Headlines

![NLP](https://img.shields.io/badge/domain-NLP-blue) ![Task](https://img.shields.io/badge/task-Binary%20Classification-green)

## Overview
Detect whether a news headline is sarcastic. Using a clean, label-reliable corpus (sarcastic headlines from The Onion vs. real headlines from HuffPost), the notebook trains recurrent neural networks on top of two different word-embedding schemes — **Word2Vec** (trained on the corpus) and pretrained **GloVe** — and compares them.

## Dataset
- **News Headlines Dataset for Sarcasm Detection**: https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection
- **GloVe Twitter embeddings** (200d): https://www.kaggle.com/datasets/icw123/glove-twitter
- ~28,000 headlines, balanced across sarcastic (1) / not-sarcastic (0). Fields: `headline`, `is_sarcastic` (`article_link` dropped).
- Headline data is included in `data/` (`Sarcasm_Headlines_Dataset.json`, `Sarcasm_Headlines_Dataset_v2.json`). The GloVe embeddings are **not** committed (~3.7 GB) — download with:
  ```bash
  kaggle datasets download icw123/glove-twitter
  ```

## Approach
1. **Cleaning** — strip HTML (BeautifulSoup), remove URLs/bracketed text, drop stopwords + punctuation.
2. **EDA** — class balance, word clouds per class, character/word-length and average-word-length distributions.
3. **Word2Vec model**
   - Train gensim `Word2Vec` (`size=200`, `window=5`, `min_count=1`) on the headline tokens.
   - Tokenize/pad sequences to length 20; build an embedding weight matrix.
   - Network: `Embedding(200d)` → `Bidirectional(LSTM(128), dropout/recurrent_dropout=0.3, return_sequences)` → `Bidirectional(GRU(32), dropout/recurrent_dropout=0.1)` → `Dense(1, sigmoid)`. Adam (lr=0.01), binary cross-entropy, 3 epochs, batch size 128.
4. **GloVe model** — same idea using pretrained `glove.twitter.27B.200d` vectors with `max_features=35000`, `maxlen=200`, and an analogous RNN classifier.
5. **Evaluation** — train/test accuracy, confusion matrix, accuracy/loss curves; compare Word2Vec vs. GloVe.

## Results
The kernel title reports the headline accuracy.

| Metric | Value |
| --- | --- |
| Reported accuracy (kernel title) | **~83%** |

The captured run log for this project is empty, so per-model train/test accuracies and the confusion matrix were not preserved here. **Run the notebook to reproduce the metrics** — it prints train/test accuracy for each embedding scheme and plots accuracy/loss curves. The notebook markdown notes the Word2Vec model shows signs of overfitting on the test split.

## Key Takeaways
- A clean, single-source-labeled corpus avoids the label noise typical of hashtag-collected Twitter sarcasm datasets.
- Bi-directional LSTM+GRU stacks on word embeddings are a solid baseline for short-text binary classification.
- Pretrained GloVe vs. corpus-trained Word2Vec is the central comparison; the small corpus and aggressive learning rate make overfitting a risk worth watching.

## How to Run
```bash
pip install numpy pandas seaborn matplotlib nltk wordcloud beautifulsoup4 gensim keras tensorflow scikit-learn
jupyter notebook notebook.ipynb
```
Headline JSON files are in `data/`; download the GloVe embeddings (see Dataset) for the second model.

## Credit
> Based on ["Sarcasm Detection with GloVe/Word2Vec(83%Accuracy)"](https://www.kaggle.com/code/madz2000/sarcasm-detection-with-glove-word2vec-83-accuracy) by Madhav Mathur on Kaggle (545 votes).
> Datasets: [News Headlines Dataset for Sarcasm Detection](https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection), [GloVe Twitter](https://www.kaggle.com/datasets/icw123/glove-twitter). Adapted and documented for this portfolio.
