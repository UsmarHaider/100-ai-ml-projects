# News Article Category Classification

![NLP](https://img.shields.io/badge/domain-NLP-blue) ![Task](https://img.shields.io/badge/task-Text%20Classification-green)

## Overview
Classify news articles into their topical category (politics, entertainment, sports, etc.) from the headline and a short description. Automatic categorization powers news feeds, recommendation, and content moderation. The approach pairs pretrained GloVe word embeddings with three deep text classifiers (TextCNN, Bidirectional GRU + Conv, and LSTM + Attention) and ensembles them.

## Dataset
- [News Category Dataset](https://www.kaggle.com/datasets/rmisra/news-category-dataset) (rmisra/news-category-dataset, ~83 MB) — HuffPost headlines with `headline`, `short_description`, and `category` fields.
- [GloVe: Global Vectors for Word Representation](https://www.kaggle.com/datasets/rtatman/glove-global-vectors-for-word-representation) (rtatman/glove-global-vectors-for-word-representation, ~1.16 GB) — uses `glove.6B.100d.txt` (100-dim embeddings).
- Categories: the notebook reports the count via `cates.ngroups`; `THE WORLDPOST` is merged into `WORLDPOST` to remove a duplicate class. Records with fewer than 5 tokens are dropped.
- No `data/` committed. Download with:
  ```bash
  kaggle datasets download rmisra/news-category-dataset
  kaggle datasets download rtatman/glove-global-vectors-for-word-representation
  ```

## Approach
1. **Text prep**: concatenate `headline + " " + short_description` into a single `text` field.
2. **Tokenization**: Keras `Tokenizer` fit on all text; sequences padded/truncated to `maxlen = 50`. Rows with `word_length < 5` removed.
3. **Label encoding**: categories mapped to integer ids and one-hot encoded with `np_utils.to_categorical`.
4. **Embeddings**: build an embedding matrix from `glove.6B.100d.txt` (`EMBEDDING_DIM = 100`) into a non-trainable Keras `Embedding` layer.
5. **Split**: `train_test_split` with `test_size = 0.2`, `random_state = 29`.
6. **Models** (all `adam`, `categorical_crossentropy`, `batch_size = 128`, `epochs = 20`):
   - **TextCNN**: parallel `Conv1D(64)` with kernel sizes `[2, 3, 4]`, each followed by `MaxPooling1D(3)` and `Dropout(0.5)`, concatenated → `Flatten` → `Dropout(0.5)` → softmax `Dense`.
   - **Bidirectional GRU + Conv**: `SpatialDropout1D(0.2)` → `Bidirectional(GRU(128, dropout=0.1, recurrent_dropout=0.1))` → `Conv1D(64, k=3)` → concat of `GlobalAveragePooling1D` + `GlobalMaxPooling1D` → softmax `Dense`.
   - **LSTM with Attention**: `LSTM(300, dropout=0.25, recurrent_dropout=0.25)` → `Dropout(0.25)` → custom `Attention` layer → `Dense(256, relu)` → `Dropout(0.25)` → `BatchNormalization` → softmax `Dense`.
7. **Ensemble**: average the softmax predictions of the three models on the validation set.
8. **Evaluation**: per-model validation accuracy (`evaluate_accuracy`), ensemble accuracy (`evaluate_accuracy_ensemble`), and a confusion matrix for the Attention LSTM.

## Results
The committed notebook has **no executed cell outputs**, so per-model and ensemble accuracies must be regenerated. The kernel title claims a validation accuracy of **~0.65** ([source title](https://www.kaggle.com/code/hengzheng/news-category-classifier-val-acc-0-65)).

| Model | Validation accuracy |
|-------|--------------------|
| TextCNN | run notebook to reproduce |
| Bidirectional GRU + Conv | run notebook to reproduce |
| LSTM with Attention | run notebook to reproduce |
| Ensemble (avg of 3) | run notebook to reproduce |
| Author's title claim | ~0.65 (`val_acc`) |

Run the notebook to reproduce metrics; it reports per-model validation accuracy, an ensemble accuracy, training/validation accuracy and loss curves per model, and a confusion matrix.

## Key Takeaways
- Combining frozen GloVe embeddings with multiple architectures (CNN, recurrent, attention) and ensembling is a strong, simple recipe for short-text classification.
- Light data cleaning matters: merging the duplicate `THE WORLDPOST`/`WORLDPOST` label and dropping <5-token rows reduces noise and label leakage.
- A short `maxlen = 50` is sufficient because inputs are only headline + short description.
- Possible improvements: trainable/fine-tuned embeddings, class-imbalance handling (categories are uneven), early stopping or `ModelCheckpoint` (imported but unused), and a stronger transformer baseline (e.g. DistilBERT).

## How to Run
```bash
pip install numpy pandas matplotlib scikit-learn tensorflow keras
jupyter notebook notebook.ipynb
```
Place the data so the notebook's `../input/` paths resolve, e.g. `../input/news-category-dataset/News_Category_Dataset.json` and `../input/glove-global-vectors-for-word-representation/glove.6B.100d.txt` (unzip the Kaggle downloads above). Note: the notebook uses the legacy `keras` API (`keras.engine.topology`, `np_utils`, `acc` history keys); a Keras 2.x / TF 1.x-era environment is recommended, or adapt the imports for modern TensorFlow.

## Credit
> Based on ["News Category Classifier (val_acc ~ 0.65)"](https://www.kaggle.com/code/hengzheng/news-category-classifier-val-acc-0-65) by heng on Kaggle (647 votes).
> Datasets: [News Category Dataset](https://www.kaggle.com/datasets/rmisra/news-category-dataset) and [GloVe: Global Vectors for Word Representation](https://www.kaggle.com/datasets/rtatman/glove-global-vectors-for-word-representation). Adapted and documented for this portfolio.
