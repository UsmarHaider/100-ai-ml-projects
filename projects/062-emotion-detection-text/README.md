# Emotion Detection from Text

![NLP](https://img.shields.io/badge/domain-NLP-blue) ![Task](https://img.shields.io/badge/task-Multiclass%20Classification-green)

## Overview
Classify short tweets into one of six emotions (sadness, joy, love, anger, fear, surprise). The notebook is a walkthrough of the Hugging Face stack — tokenization concepts, then two ways to use **DistilBERT**: as a frozen feature extractor feeding a logistic-regression head, and as a fully fine-tuned classifier.

## Dataset
- **Emotion dataset**: https://www.kaggle.com/datasets/parulpandey/emotion-dataset
- 6 classes; splits — train (16,000), validation (2,000), test (2,000). Fields: `text`, `label`.
- Included in `data/` (`training.csv`, `validation.csv`, `test.csv`).

## Approach
1. **Data assembly** — load the three CSVs into a Hugging Face `DatasetDict` with a typed `ClassLabel` feature.
2. **Tokenization concepts** — demonstrates character, word, and subword (WordPiece) tokenization; uses `distilbert-base-uncased` (vocab 30,522, max length 512) with padding/truncation and attention masks.
3. **Approach A — feature extraction:** run frozen DistilBERT, take the `[CLS]` last-hidden-state (768-dim) per tweet as features; visualize with t-SNE; train a `LogisticRegression(max_iter=2000)` head, benchmarked against a `DummyClassifier(strategy="most_frequent")`.
4. **Approach B — fine-tuning:** `AutoModelForSequenceClassification` (`num_labels=6`) trained with the `Trainer` API — 3 epochs, learning rate 2e-5, batch size 64, weight decay 0.01; metrics = accuracy and weighted F1.
5. **Error analysis** — sort validation examples by per-sample cross-entropy loss; inference via a `pipeline`.

## Results
Values extracted from the captured run log (feature-extraction stage):

| Model | Validation accuracy |
| --- | --- |
| DummyClassifier (most frequent) | 0.352 |
| DistilBERT embeddings + Logistic Regression | **0.6335** |

The fine-tuned DistilBERT `Trainer` reports accuracy and weighted F1 per epoch and on the validation set; those exact final values were not captured in the run log (the log ends at the prediction tensors). The notebook markdown notes fine-tuning clearly outperforms feature extraction, with `love`↔`joy` and `surprise`↔`joy`/`fear` the main remaining confusions. **Run the notebook to reproduce the fine-tuned accuracy/F1.**

## Key Takeaways
- Frozen-embedding + logistic regression already nearly doubles the majority-class baseline (0.352 → 0.63).
- t-SNE shows positive emotions (joy/love) cleanly separated, while negative emotions overlap — matching the confusion-matrix errors.
- Full fine-tuning is the larger win and reduces the joy-related confusions, at higher compute cost.

## How to Run
```bash
pip install pandas numpy torch transformers datasets scikit-learn plotly panel matplotlib seaborn
jupyter notebook notebook.ipynb
```
Data is already in `data/`.

## Credit
> Based on ["Twitter Emotion Classification"](https://www.kaggle.com/code/shtrausslearning/twitter-emotion-classification) by Andrey Shtrauss on Kaggle (498 votes).
> Dataset: [Emotion dataset](https://www.kaggle.com/datasets/parulpandey/emotion-dataset). Adapted and documented for this portfolio.
