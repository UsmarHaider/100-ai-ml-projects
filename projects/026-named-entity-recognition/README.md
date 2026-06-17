# Named Entity Recognition (NER) Tutorial

![NLP](https://img.shields.io/badge/domain-NLP-blue) ![Task](https://img.shields.io/badge/task-Token%20Classification-green)

## Overview
An end-to-end tour of Named Entity Recognition, progressing from classical rule-based methods (POS tagging, chunking, IOB tags) to library models (NLTK, spaCy) and finally to a BERT token-classification model fine-tuned on an annotated corpus. NER locates and classifies entities such as persons, organizations and locations in unstructured text.

## Dataset
- [Entity Annotated Corpus](https://www.kaggle.com/datasets/abhinavwalia95/entity-annotated-corpus) (`abhinavwalia95/entity-annotated-corpus`, ~164 MB) — primary corpus for BERT fine-tuning.
- Auxiliary inputs referenced for demos: the [NLP Getting Started](https://www.kaggle.com/competitions/nlp-getting-started) competition `train.csv`, plus CORD-19 / Wikibooks for illustrative text.
- Not committed here. Download with:
  ```bash
  kaggle datasets download abhinavwalia95/entity-annotated-corpus
  kaggle competitions download -c nlp-getting-started
  ```

## Approach
1. **Classical NLP**: NLTK `pos_tag` + `word_tokenize`, `RegexpParser` noun-phrase chunking, IOB tags via `tree2conlltags`, and `ne_chunk` for entity extraction.
2. **spaCy**: `en_core_web_sm` statistical NER, with `displacy` visualization of recognized entities.
3. **BERT (zero-shot)**: Hugging Face `pipeline("ner", model="dslim/bert-base-NER")` applied directly to text.
4. **BERT fine-tuning**: `AutoModelForTokenClassification` from `dslim/bert-base-NER`, tokenizer `MAX_LEN=128`, batch size 32, optimizer `AdamW` (lr 3e-5) with a linear LR scheduler, 3 epochs. Training/validation tracked by loss and F1.

## Results
Real values from the run log, reported per epoch as training/validation loss and F1:

| Epoch | Train loss | Train F1 | Val loss | Val F1 |
|-------|-----------|----------|----------|--------|
| 1 | 0.459 | 0.335 | 0.344 | 0.347 |
| 2 | 0.326 | 0.365 | 0.317 | 0.355 |
| 3 | 0.314 | 0.369 | 0.311 | 0.356 |

Validation F1 plateaus around **0.356** over 3 epochs in this run.

## Key Takeaways
- The notebook is structured as a teaching path: classical chunking → spaCy → transformer fine-tuning, making each abstraction layer explicit.
- Starting fine-tuning from `dslim/bert-base-NER` (already an NER checkpoint) gives a head start over a vanilla BERT.
- The low absolute F1 over only 3 epochs suggests the run was illustrative; more epochs, label-set alignment, and subword/label re-alignment would raise scores.
- A linear LR scheduler with `AdamW` is the standard recipe for transformer token classification.

## How to Run
```bash
pip install transformers torch nltk spacy svgling pandas numpy
python -m spacy download en_core_web_sm
jupyter notebook notebook.ipynb
```
Place the entity-annotated corpus where the notebook expects `../input/entity-annotated-corpus` (or adjust the path).

## Credit
> Based on ["NER - Named Entity Recognition Tutorial"](https://www.kaggle.com/code/eneszvo/ner-named-entity-recognition-tutorial) by Enes Zvorničanin on Kaggle (220 votes).
> Datasets: [Entity Annotated Corpus](https://www.kaggle.com/datasets/abhinavwalia95/entity-annotated-corpus), [NLP Getting Started](https://www.kaggle.com/competitions/nlp-getting-started). Adapted and documented for this portfolio.
