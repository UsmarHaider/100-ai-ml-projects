# Question Answering with BERT

![NLP](https://img.shields.io/badge/domain-NLP-blue) ![Task](https://img.shields.io/badge/task-Extractive%20QA-green)

## Overview
Extractive question answering: given a question and a context paragraph, the model predicts the start and end token positions of the answer span. The notebook first demonstrates inference with a SQuAD-finetuned BERT, then fine-tunes DistilBERT on SQuAD from scratch.

## Dataset
- [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) — Stanford Question Answering Dataset, loaded directly via `datasets.load_dataset("squad")` (no manual download needed).
- The kernel is self-contained on Kaggle; no dataset is committed in `data/`. `load_dataset("squad")` fetches the data at runtime.

## Approach
1. **Inference demo**: `BertForQuestionAnswering` + `BertTokenizer` from `kaporter/bert-base-uncased-finetuned-squad`; encode `question + context`, take argmax of start/end logits, decode the span.
2. **Data labeling**: map character-level answer spans to token positions; handle long contexts by splitting into overlapping windows (`return_overflowing_tokens=True`, `max_length` 160 then 512, `padding="max_length"`).
3. **Fine-tuning**: `distilbert-base-uncased` chosen over full BERT for speed/size; trained with a PyTorch loop over the tokenized SQuAD train split.
4. **Metrics**: Exact Match (EM) and F1 over the validation set via the `evaluate` library.

## Results
Real values from the run log:

| Epoch | Avg training loss | Exact Match | F1 |
|-------|-------------------|-------------|------|
| 1 | 2.52 | 23.8 | 60.48 |
| 2 | 1.11 | 24.8 | 61.57 |

After 2 epochs the DistilBERT model reaches **EM 24.8 / F1 61.57** on the evaluation set.

## Key Takeaways
- QA is framed as predicting two token classifiers (answer start and answer end) over the context.
- Long contexts must be windowed with overlap so the answer span is never split across the truncation boundary.
- DistilBERT trades some accuracy for much faster training — F1 ~61.6 after only 2 epochs; more epochs and full BERT would push EM/F1 higher (BERT-base typically reaches ~80 F1 on SQuAD).
- The large F1-vs-EM gap shows the model often overlaps the gold answer without matching it exactly.

## How to Run
```bash
pip install torch transformers datasets evaluate pandas numpy
jupyter notebook notebook.ipynb
```
`datasets.load_dataset("squad")` downloads SQuAD automatically; a GPU is recommended for fine-tuning.

## Credit
> Based on ["Question-Answering using Bert"](https://www.kaggle.com/code/arunmohan003/question-answering-using-bert) by arunmohan_003 on Kaggle (330 votes).
> Dataset: [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) via Hugging Face `datasets`. Adapted and documented for this portfolio.
