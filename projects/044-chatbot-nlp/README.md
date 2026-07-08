# Chatbot for Mental Health Conversations

![NLP](https://img.shields.io/badge/domain-NLP-blue) ![Task](https://img.shields.io/badge/task-Intent%20Classification-green)

## Overview
An intent-based conversational chatbot for mental-health support. User messages are classified into predefined intents (e.g. `sad`, `greeting`, `anxious`), and the bot replies with an empathetic response drawn from that intent's response set.

## Dataset
- [Mental Health Conversational Data](https://www.kaggle.com/datasets/elvis23/mental-health-conversational-data) (`elvis23/mental-health-conversational-data`, ~0.04 MB).
- **Included in `data/`** as `intents.json` — a list of intents, each with a `tag`, example `patterns`, and candidate `responses`.

## Approach
1. **Data preparation**: load `intents.json`, flatten into a DataFrame of `(tag, pattern, responses)` rows so each example pattern maps to its intent tag.
2. **EDA**: Plotly bar charts of intent distribution and average pattern/response counts per intent.
3. **Intent model**: `train_test_split` (80/20), `TfidfVectorizer` on the patterns, and a Support Vector Machine (`sklearn.svm.SVC`) classifier trained to predict the intent tag.
4. **Evaluation**: `classification_report` (precision/recall/F1 per intent) visualized with a grouped Plotly bar plot.
5. **Deployment**: `predict_intent(user_input)` vectorizes the message, predicts the intent, and returns a response sampled from that intent.

## Results
The notebook computes a `classification_report` (precision, recall, F1-score per intent) and visualizes the scores, but no executed run log is present in this folder and the committed notebook does not embed the numeric report.

Run the notebook to reproduce the metrics: per-intent **precision, recall, F1-score** from the SVM intent classifier.

## Key Takeaways
- The "intent" abstraction (tag + patterns + responses) cleanly separates understanding (classification) from generation (response lookup).
- TF-IDF + linear SVM is a lightweight, fast baseline well-suited to the small pattern set.
- With few patterns per intent, the train/test split is tiny — metrics will be noisy; more patterns or data augmentation would help.
- This is a retrieval/template chatbot, not generative; an embedding-based or transformer intent classifier is a natural upgrade.

## How to Run
```bash
pip install scikit-learn plotly pandas numpy
jupyter notebook notebook.ipynb
```
Point the `intents.json` path at the local `data/intents.json`.

## Credit
> Based on ["Chatbot for Mental Health Conversations"](https://www.kaggle.com/code/jocelyndumlao/chatbot-for-mental-health-conversations) by Jocelyn Dumlao on Kaggle (1835 votes).
> Dataset: [Mental Health Conversational Data](https://www.kaggle.com/datasets/elvis23/mental-health-conversational-data). Adapted and documented for this portfolio.
