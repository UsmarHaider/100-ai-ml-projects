# Transformer Models for Text Classification

![Domain](https://img.shields.io/badge/domain-NLP-blue)
![Task](https://img.shields.io/badge/task-Text%20Classification-green)

## Overview
Fine-tunes a HuggingFace 🤗 Transformer (RoBERTa, with switchable BERT / DistilBERT /
XLNet / XLM backbones) for 5-class movie-review sentiment classification, wrapping the
`transformers` model inside the `fastai` training loop so it benefits from discriminative
learning rates and gradual unfreezing.

## Dataset
- **Sentiment Analysis on Movie Reviews** — [Kaggle competition](https://www.kaggle.com/competitions/sentiment-analysis-on-movie-reviews)
- `train.tsv.zip` / `test.tsv.zip`: Rotten Tomatoes phrases with 5 sentiment labels
  (0 = negative … 4 = positive).
- Data is **not** committed. Download via the competition page:
  ```bash
  kaggle competitions download -c sentiment-analysis-on-movie-reviews
  ```
  The notebook reads from `DATA_ROOT/train.tsv.zip` and `test.tsv.zip`.

## Approach
1. **Setup** — `pip install transformers`; a `MODEL_CLASSES` dict maps a model-type string
   to its (model, tokenizer, config) classes for BERT, RoBERTa, DistilBERT, XLNet, XLM.
2. **Config** — `model_type = 'roberta'`, `pretrained_model_name = 'roberta-base'`, batch
   size `bs = 16`, fixed `seed` via `seed_all()`.
3. **Tokenization** — custom `TransformersBaseTokenizer` / `TransformersVocab` adapt the
   HF tokenizer (with model-specific BOS/EOS, padding side, `add_prefix_space` for RoBERTa)
   to a fastai `databunch` built with `TextList.from_df(... split_by_rand_pct(0.1) ...)`.
4. **Model** — `RobertaForSequenceClassification` with `config.num_labels = 5`, wrapped in
   a `CustomTransformerModel`; trained with a `Learner` using `AdamW` (`correct_bias=False`)
   and metrics `[accuracy, error_rate]`.
5. **Training schedule** (gradual unfreezing + `fit_one_cycle`):
   - `freeze_to(-1)` → `fit_one_cycle(1, max_lr=2e-3, moms=(0.8,0.7))`
   - `freeze_to(-2)` → `fit_one_cycle(1, max_lr=slice(...), moms=(0.8,0.9))`
   - `freeze_to(-3)` → `fit_one_cycle(1, max_lr=slice(...), moms=(0.8,0.9))`
   - `unfreeze()` → `fit_one_cycle(2, max_lr=slice(...), moms=(0.8,0.9))`

## Results
The notebook evaluates on classification **accuracy** (plus `error_rate`) shown per epoch
via fastai's `ShowGraph`. The run log for this project is empty, so no executed accuracy
value was captured here.

| Metric | Where reported |
|---|---|
| Accuracy | fastai `Learner` metric, per epoch |
| Error rate | fastai `Learner` metric, per epoch |

Run the notebook to reproduce the per-epoch accuracy and the competition submission.

## Key Takeaways
- A single `MODEL_CLASSES` dictionary makes the architecture (BERT/RoBERTa/XLNet/…) a
  one-line swap.
- Combining HF transformers with fastai unlocks discriminative LRs, slanted triangular LRs
  and gradual unfreezing for transfer learning.
- FP16 mode is wired in but noted as not reliably working in the original kernel.

## How to Run
```bash
pip install numpy pandas torch fastai transformers
jupyter notebook notebook.ipynb
```
Download the competition data (see Dataset) and point `DATA_ROOT` at the `.tsv.zip` files.
GPU strongly recommended.

## Credit
> Based on ["Fastai with 🤗 Transformers (BERT, RoBERTa, ...)"](https://www.kaggle.com/code/maroberti/fastai-with-transformers-bert-roberta) by Maximilien Roberti on Kaggle (1230 votes).
> Dataset: [Sentiment Analysis on Movie Reviews](https://www.kaggle.com/competitions/sentiment-analysis-on-movie-reviews). Adapted and documented for this portfolio.
