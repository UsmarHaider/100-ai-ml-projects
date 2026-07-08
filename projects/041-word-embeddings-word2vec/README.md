# Word Embeddings with Gensim Word2Vec

![NLP](https://img.shields.io/badge/domain-NLP-blue) ![Task](https://img.shields.io/badge/task-Word%20Embeddings-green)

## Overview
A practical Word2Vec tutorial that emphasizes the often-skipped data-preparation step. It trains word embeddings on Simpsons dialogue with Gensim, where words sharing similar contexts end up with similar vectors, enabling similarity queries and analogy arithmetic.

## Dataset
- [Dialogue Lines of The Simpsons](https://www.kaggle.com/datasets/pierremegret/dialogue-lines-of-the-simpsons) (`pierremegret/dialogue-lines-of-the-simpsons`, ~9 MB).
- **Included in `data/`** as `simpsons_dataset.csv` (columns include `raw_character_text`, `spoken_words`).

## Approach
1. **Cleaning**: regex strip non-alphabetic characters and lowercase each line; lemmatize and remove stopwords with spaCy (`nlp.pipe`, NER/parser disabled for speed); drop empty rows.
2. **Bigrams**: detect common phrases with Gensim `Phrases(min_count=30)` and apply via `Phraser` (e.g. `homer_simpson`, `moe_'s`).
3. **Model**: `gensim.models.Word2Vec(min_count=20, window=2, size=300, sample=6e-5, alpha=0.03, min_alpha=0.0007, negative=20, workers=cores-1)`.
4. **Training**: `build_vocab(sentences)` then `train(..., epochs=30)`.
5. **Exploration**: `most_similar` (e.g. for `homer`, `marge`, `bart`), `similarity` pairs, `doesnt_match` outlier detection, analogy arithmetic (`woman + homer - marge`), and t-SNE scatter plots of neighboring vectors.

## Results
Word2Vec is an unsupervised embedding model, so there is no classification metric. The notebook's outputs are **qualitative similarity/analogy queries** (nearest-neighbor word lists, pairwise cosine similarities, outlier detection, t-SNE projections); no run log is present in this folder.

Run the notebook to reproduce the embedding-space queries and t-SNE visualizations.

## Key Takeaways
- Data preparation (lemmatization, stopword removal, bigram detection) materially changes embedding quality — the tutorial's central point.
- A small `window=2` with skip-gram-style negative sampling (`negative=20`) and 300-dim vectors works well for conversational dialogue.
- `Phraser` lets multi-word entities like `homer_simpson` become single tokens, sharpening similarity results.
- The `min_count=20` floor and `sample=6e-5` subsampling keep rare/very-frequent tokens from dominating.

## How to Run
```bash
pip install gensim spacy scikit-learn matplotlib seaborn pandas numpy
python -m spacy download en_core_web_sm
jupyter notebook notebook.ipynb
```
Note: the notebook loads `spacy.load('en')` and `../input/simpsons_dataset.csv`; update to `en_core_web_sm` and the local `data/simpsons_dataset.csv` path. Newer Gensim renames `size=` to `vector_size=`.

## Credit
> Based on ["Gensim Word2Vec Tutorial"](https://www.kaggle.com/code/pierremegret/gensim-word2vec-tutorial) by Pierre Megret on Kaggle (2206 votes).
> Dataset: [Dialogue Lines of The Simpsons](https://www.kaggle.com/datasets/pierremegret/dialogue-lines-of-the-simpsons). Adapted and documented for this portfolio.
