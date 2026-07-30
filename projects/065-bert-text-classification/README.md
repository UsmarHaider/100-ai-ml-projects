# Text Classification with BERT (DistilBERT + LoRA)

![NLP](https://img.shields.io/badge/domain-NLP-blue) ![Task](https://img.shields.io/badge/task-Binary%20Sentiment%20Classification-green)

## Overview
Binary sentiment classification (positive vs. negative) on IMDB movie reviews. Instead of full fine-tuning, the notebook applies **LoRA** (Low-Rank Adaptation) to **DistilBERT**, training only a small fraction of parameters for an efficient, GPU-friendly transformer pipeline.

## Dataset
- **IMDB reviews**, loaded directly via the Hugging Face `datasets` hub (`load_dataset("imdb")`) — no Kaggle dataset attached.
- Subsampled for efficiency: 10,000 training reviews and 2,500 evaluation reviews (balanced 50/50 positive/negative; average length ~230 words).
- No `data/` folder is needed — the dataset is downloaded at runtime:
  ```python
  from datasets import load_dataset
  dataset = load_dataset("imdb")
  ```

## Approach
1. **EDA** — text-length distribution (mean ~230 words → `max_length=256`) and word clouds.
2. **Tokenization** — `DistilBertTokenizerFast` (`distilbert-base-uncased`), truncate/pad to 256 tokens.
3. **Model** — `DistilBertForSequenceClassification` (`num_labels=2`) wrapped with PEFT **LoRA**: `r=32`, `lora_alpha=32`, `lora_dropout=0.05`, `bias="none"`, target modules `q_lin`, `v_lin`.
4. **Training** — Hugging Face `Trainer`: 3 epochs, learning rate 3e-5, per-device batch size 16 with `gradient_accumulation_steps=2` (effective 32), `weight_decay=0.01`, mixed precision (`fp16=True`), `load_best_model_at_end` on weighted F1.
5. **Metrics** — accuracy and weighted F1 via the `evaluate` library.

## Results
Values extracted from the captured run log (final `trainer.evaluate()` on the 2,500-sample test set):

| Metric | Value |
| --- | --- |
| Accuracy | **0.8612** |
| Weighted F1 | **0.8612** |
| Eval loss | 0.3084 |
| Trainable params (LoRA) | 1,181,954 of 68,136,964 (**1.73%**) |

## Key Takeaways
- LoRA trains only ~1.7% of the model's parameters yet reaches ~86% accuracy/F1, a strong efficiency/performance trade-off.
- Mixed precision + gradient accumulation make it feasible to fine-tune on 10k samples within ~20–40 minutes on a single GPU.
- Capping sequence length at 256 (vs. 512) is justified by the ~230-word average review length and further reduces cost.

## How to Run
```bash
pip install torch transformers datasets peft evaluate accelerate wandb scikit-learn matplotlib seaborn wordcloud pandas
jupyter notebook notebook.ipynb
```
The IMDB dataset is fetched automatically on first run; no manual data placement required.

## Credit
> Based on ["Sentiment Analysis ▴LoRA ▴DistilBERT"](https://www.kaggle.com/code/tahmidmir/sentiment-analysis-lora-distilbert) by Mir Tahmid on Kaggle (145 votes).
> Dataset: IMDB reviews via the Hugging Face `datasets` hub. Adapted and documented for this portfolio.
