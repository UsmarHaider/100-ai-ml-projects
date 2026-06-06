# 100 AI/ML Projects — CV · NLP · ML

![Progress](https://img.shields.io/badge/published-11%2F100-brightgreen) ![CV](https://img.shields.io/badge/Computer%20Vision-4-blue) ![NLP](https://img.shields.io/badge/NLP-4-orange) ![ML](https://img.shields.io/badge/Machine%20Learning-3-purple) ![Python](https://img.shields.io/badge/python-3.x-yellow)

**New projects published regularly** — building up to 100 end-to-end data science projects across computer vision, natural language processing and classic machine learning. Each project folder contains:

- `notebook.ipynb` — a complete, runnable Jupyter notebook (curated from top-voted Kaggle work)
- `README.md` — a full report: problem, dataset, approach, results and how to run
- `data/` — the dataset itself when small enough to commit (✅), otherwise a download link/command (🔗)
- `source.json` — provenance metadata (original author, kernel URL, votes, dataset links)

> **Attribution.** These notebooks are adapted from publicly shared Kaggle kernels; every project README credits its original author with links to the source kernel and dataset. Original works remain under their authors' licenses (typically Apache 2.0).

## 🖼 Computer Vision (4 published)

Image classification, detection, segmentation, GANs, OCR and medical imaging.

| Project | Original author | ⭐ Kaggle votes | Data |
|---|---|---|---|
| [MNIST Handwritten Digit Classification](projects/001-mnist-digit-classification) | [DATAI](https://www.kaggle.com/code/kanncaa1/pytorch-tutorial-for-deep-learning-lovers) | 5,193 | 🔗 |
| [CIFAR-10 Image Classification](projects/004-cifar10-image-classification) | [Farzad Nekouei](https://www.kaggle.com/code/farzadnekouei/cifar-10-image-classification-with-cnn) | 2,198 | 🔗 |
| [Fashion MNIST Clothing Classification](projects/007-fashion-mnist-classification) | [Pavan Sanagapati](https://www.kaggle.com/code/pavansanagapati/a-simple-cnn-model-beginner-guide) | 1,613 | 🔗 |
| [Cats vs Dogs Image Classification](projects/010-cats-vs-dogs-classification) | [Uysim](https://www.kaggle.com/code/uysimty/keras-cnn-dog-or-cat-classification) | 4,696 | 🔗 |

## 📝 Natural Language Processing (4 published)

Sentiment analysis, text classification, transformers, NER, summarization and topic modeling.

| Project | Original author | ⭐ Kaggle votes | Data |
|---|---|---|---|
| [IMDB Movie Review Sentiment Analysis](projects/002-imdb-sentiment-analysis) | [Lakshmipathi N](https://www.kaggle.com/code/lakshmi25npathi/sentiment-analysis-of-imdb-movie-reviews) | 3,719 | 🔗 |
| [Twitter Sentiment Analysis](projects/005-twitter-sentiment-analysis) | [Rob Mulla](https://www.kaggle.com/code/robikscube/sentiment-analysis-python-youtube-tutorial) | 5,984 | 🔗 |
| [Spam Email/SMS Detection](projects/008-spam-email-detection) | [Fares Sayah](https://www.kaggle.com/code/faressayah/natural-language-processing-nlp-for-beginners) | 1,419 | ✅ |
| [Fake News Detection](projects/011-fake-news-detection) | [Awsaf](https://www.kaggle.com/code/awsaf49/detect-fake-text-kerasnlp-tf-torch-jax-train) | 1,501 | 🔗 |

## 📊 Machine Learning & Data Science (3 published)

Regression, classification, clustering, time series, recommenders and EDA.

| Project | Original author | ⭐ Kaggle votes | Data |
|---|---|---|---|
| [Titanic Survival Prediction](projects/003-titanic-survival-prediction) | [Manav Sehgal](https://www.kaggle.com/code/startupsci/titanic-data-science-solutions) | 39,960 | 🔗 |
| [House Price Prediction](projects/006-house-price-prediction) | [Gusthema](https://www.kaggle.com/code/gusthema/house-prices-prediction-using-tfdf) | 15,022 | 🔗 |
| [Credit Card Fraud Detection](projects/009-credit-card-fraud-detection) | [Janio Martinez Bachmann](https://www.kaggle.com/code/janiobachmann/credit-fraud-dealing-with-imbalanced-datasets) | 15,064 | 🔗 |

## Getting started

```bash
git clone https://github.com/UsmarHaider/100-ai-ml-projects.git
pip install jupyter numpy pandas scikit-learn matplotlib seaborn
cd projects/<any-project>
jupyter notebook notebook.ipynb
```

Projects marked 🔗 need their dataset downloaded first — the exact `kaggle datasets download` command is in each project's README (requires the [Kaggle CLI](https://www.kaggle.com/docs/api)).
