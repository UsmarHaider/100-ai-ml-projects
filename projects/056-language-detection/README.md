# Language Detection

![NLP](https://img.shields.io/badge/domain-NLP-blue) ![Task](https://img.shields.io/badge/task-Multiclass%20Classification-green)

## Overview
Given short, noisy text snippets (most contain a primary language plus a few foreign words), the task is to identify the language. The notebook focuses on **memory-efficient** character n-gram approaches and compares several classification/divergence methods rather than a single heavyweight model.

## Dataset
- **Language Identification dataset**: https://www.kaggle.com/datasets/zarajamshaid/language-identification-datasst
- A single `dataset.csv` with columns `Text` and `language`, covering 22 languages (e.g. English, Spanish, Latin/Italian, Dutch, Swedish, Chinese, Japanese, Korean, Russian, Thai, Arabic, ...). The notebook splits 80/20 into train/test.
- Included in `data/` (`dataset.csv`, ~12.4 MB).

## Approach
1. **Train/test split** — `train_test_split(test_size=0.2, random_state=42)`.
2. **Feature extraction with character n-grams** (`CountVectorizer(analyzer='char')`):
   - **Unigrams** — sufficient to separate languages with unique scripts (Chinese ~3,249 symbols, Japanese ~2,054, Korean ~1,407).
   - **Bigrams** and a **Uni+Bigram mixture** (`ngram_range=(1,2)`, `min_df=1e-2`) for languages sharing the Latin/Cyrillic alphabets.
   - Reduced feature sets via "top-K grams per language" to stay memory-efficient.
3. **Per-language frequency profiles** — relative character distributions built per language.
4. **Classification / scoring methods compared:** Naive Bayes, k-Nearest Neighbours, Sum of Squared Differences, Kullback–Leibler Divergence, and a Kolmogorov–Smirnov test, then an F1-score comparison across methods.

## Results
The notebook reports **F1-scores** per method for the language-identification task and a comparison table/heatmap of character distributions across European languages.

The captured run log for this project is empty, and the notebook's executed metric outputs were not preserved here. **Run the notebook to reproduce the metrics** — it computes per-method F1-scores and per-language confusion behavior.

## Key Takeaways
- Languages with unique scripts are trivially separable from character unigrams alone.
- Romance languages (Spanish/Italian/Portuguese/...) sharing the Latin alphabet are the hard cases and require distributional (bigram/mixture) features.
- Restricting to the most frequent n-grams per language keeps the feature space small without sacrificing much accuracy — useful for low-memory deployment.

## How to Run
```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy
jupyter notebook notebook.ipynb
```
The dataset is already in `data/dataset.csv`.

## Credit
> Based on ["Language Detection"](https://www.kaggle.com/code/martinkk5575/language-detection) by Martin kleine Kalvelage on Kaggle (95 votes).
> Dataset: [Language Identification dataset](https://www.kaggle.com/datasets/zarajamshaid/language-identification-datasst). Adapted and documented for this portfolio.
