# IMDB Movie Review Sentiment Analysis

![NLP](https://img.shields.io/badge/domain-NLP-green)
![Task](https://img.shields.io/badge/task-text%20classification-orange)

## Overview
Classify IMDB movie reviews as **positive** or **negative** from their raw text. The pipeline applies classic NLP text normalization, two competing vectorization schemes (Bag-of-Words and TF-IDF), and a Logistic Regression classifier. It is a clean reference for end-to-end text classification without deep learning.

## Dataset
- **IMDB Dataset of 50K Movie Reviews** — [kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- 50,000 reviews with a `review` text column and a balanced `sentiment` label (25k positive / 25k negative).
- Split used: first **40,000** reviews for training, last **10,000** for testing.
- Data is **not** committed. Download with:
  ```bash
  kaggle datasets download lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
  unzip imdb-dataset-of-50k-movie-reviews.zip
  ```
  The notebook reads `../input/IMDB Dataset.csv`.

## Approach
1. **Text normalization**
   - Strip HTML with BeautifulSoup and remove text inside square brackets.
   - Remove special characters via regex.
   - Porter **stemming** of every token.
   - Remove English **stopwords** (ToktokTokenizer + NLTK stopword list).
2. **Vectorization** (two variants, both with `ngram_range=(1,3)`)
   - **Bag-of-Words**: `CountVectorizer`.
   - **TF-IDF**: `TfidfVectorizer`.
3. **Labels** — `LabelBinarizer` maps positive/negative to 1/0.
4. **Model** — `LogisticRegression(penalty='l2', C=1, max_iter=500)` trained separately on the BoW and TF-IDF features.
5. **Evaluation** — `accuracy_score`, plus `classification_report` and `confusion_matrix`.

## Results
The notebook computes accuracy for both feature sets (`lr_bow_score`, `lr_tfidf_score`) and prints classification reports and confusion matrices. No executed metric outputs are present in the run log.

| Model | Metric | Value |
| --- | --- | --- |
| Logistic Regression (BoW) | Accuracy | Run the notebook to reproduce |
| Logistic Regression (TF-IDF) | Accuracy | Run the notebook to reproduce |

Run the notebook to reproduce the metrics (it reports accuracy, precision, recall, F1, and confusion matrices).

## Key Takeaways
- Heavy text normalization (HTML stripping, stemming, stopword removal) plus simple linear models is a strong, fast baseline for binary sentiment.
- Comparing BoW against TF-IDF on identical splits isolates the effect of term weighting.
- Using n-grams up to trigrams captures short phrases like "not good".
- No deep learning required — the whole pipeline is scikit-learn + NLTK.

## How to Run
```bash
pip install numpy pandas seaborn matplotlib nltk scikit-learn wordcloud beautifulsoup4 spacy textblob
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')"
jupyter notebook notebook.ipynb
```
Download the IMDB dataset (see Dataset) and place `IMDB Dataset.csv` where the notebook expects it.

## Credit
> Based on ["Sentiment Analysis of IMDB Movie Reviews"](https://www.kaggle.com/code/lakshmi25npathi/sentiment-analysis-of-imdb-movie-reviews) by Lakshmipathi N on Kaggle (3,719 votes).
> Dataset: [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews). Adapted and documented for this portfolio.
