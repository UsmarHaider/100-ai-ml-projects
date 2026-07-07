# Topic Modeling (LDA) on Elon Musk Tweets

![NLP](https://img.shields.io/badge/domain-NLP-blue) ![Task](https://img.shields.io/badge/task-Topic%20Modeling-green)

## Overview
Unsupervised topic discovery over a corpus of Elon Musk tweets using Latent Dirichlet Allocation (LDA). LDA represents each document as a soft mixture of latent topics and each topic as a probability distribution over words, revealing the recurring themes in the tweets.

## Dataset
- [Elon Musk's Tweets](https://www.kaggle.com/datasets/kulgen/elon-musks-tweets) (`kulgen/elon-musks-tweets`, ~0.43 MB).
- **Included in `data/`** as `data_elonmusk.csv` (columns include `Tweet`, `Time`).

## Approach
1. **Preprocessing**: parse tweet timestamps, build a corpus of tweet texts, lowercase and tokenize on whitespace, remove NLTK English stopwords + punctuation + `RT`/`rt`.
2. **Corpus build**: `gensim.corpora.Dictionary` from tokens, then bag-of-words vectors (`doc2bow`); serialized to disk as `.dict` / `.mm`.
3. **Transformation**: `TfidfModel` to weight the BoW corpus.
4. **Model**: `gensim.models.LdaModel` with `num_topics=5` over the corpus (with TF-IDF folded in).
5. **Interpretation**: `show_topics` / `show_topic` to list top words per topic; topic-correlation `clustermap` (seaborn) and `pyLDAvis` interactive visualization.

## Results
This is an unsupervised model with no labeled ground truth, so no accuracy/F1 metric is computed. Outputs are the **5 discovered topics** (top-N weighted words each), a topic-correlation heatmap, and a `pyLDAvis` visualization; no run log is present in this folder.

Run the notebook to reproduce the topic-word distributions and visualizations.

## Key Takeaways
- LDA is fully unsupervised — the number of topics (5 here) is a chosen hyperparameter, not learned.
- Aggressive stopword/punctuation/`RT` removal is critical for tweet data so topics aren't dominated by noise tokens.
- TF-IDF weighting before LDA down-weights ubiquitous terms and sharpens topics.
- `pyLDAvis` and a topic-correlation clustermap make the otherwise opaque topics interpretable. Tuning `num_topics` and adding coherence scoring would strengthen model selection.

## How to Run
```bash
pip install gensim nltk pyLDAvis plotly seaborn matplotlib pandas numpy
python -c "import nltk; nltk.download('stopwords')"
jupyter notebook notebook.ipynb
```
Point the `datafile` path at `data/data_elonmusk.csv`.

## Credit
> Based on ["Topic Modelling (LDA) on Elon Tweets"](https://www.kaggle.com/code/errearanhas/topic-modelling-lda-on-elon-tweets) by Renato Aranha on Kaggle (338 votes).
> Dataset: [Elon Musk's Tweets](https://www.kaggle.com/datasets/kulgen/elon-musks-tweets). Adapted and documented for this portfolio.
