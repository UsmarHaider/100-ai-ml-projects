# Detect AI-Generated Text (Fake / LLM Text Detection)

![NLP](https://img.shields.io/badge/domain-NLP-green)
![Task](https://img.shields.io/badge/task-binary%20classification-orange)
![Model](https://img.shields.io/badge/model-DeBERTa%20v3-purple)

## Overview
Binary-classify essays as **human-written** vs. **AI/LLM-generated**. The notebook fine-tunes a **DeBERTa-v3** transformer with KerasNLP (backend-agnostic across TF/Torch/JAX) on external augmentation datasets, using stratified K-fold cross-validation. Reliable AI-text detection is increasingly important for academic integrity and content moderation.

## Dataset
- **Competition:** [LLM - Detect AI Generated Text](https://www.kaggle.com/competitions/llm-detect-ai-generated-text)
- **External training data** (used because the competition's own train set is tiny):
  - [argugpt](https://www.kaggle.com/datasets/alejopaullier/argugpt) — committed as `data/argugpt.csv`
  - [daigt-external-dataset](https://www.kaggle.com/datasets/alejopaullier/daigt-external-dataset) — committed as `data/daigt_external_dataset.csv`
  - [daigt-proper-train-dataset](https://www.kaggle.com/datasets/thedrcat/daigt-proper-train-dataset) (~353 MB, **not** committed)
- `data/` also contains `machine-train.csv`, `machine-dev.csv`, `machine-test.csv`. Get the remaining pieces with:
  ```bash
  kaggle competitions download -c llm-detect-ai-generated-text
  kaggle datasets download thedrcat/daigt-proper-train-dataset
  ```

## Approach
1. **Config (`CFG`)** — preset DeBERTa-v3, `sequence_length=200`, `num_folds=5`, `selected_folds=[0,1]`, `epochs=3`, device **TPU**, seed 42.
2. **Data** — load competition + external essays, combine, and build a `StratifiedKFold` split.
3. **Preprocessing** — `keras_nlp.models.DebertaV3Preprocessor` tokenizes/pads text to length 200; wrapped in a `tf.data` pipeline (`build_dataset`, batched).
4. **Model** — KerasNLP DeBERTa-v3 classifier head; compiled with an **AUC** metric and a custom learning-rate schedule with warmup; **Weights & Biases** logging and callbacks (LR schedule, checkpointing).
5. **Training** — loop over selected folds; predict on the held-out validation fold.

## Results
Real values extracted from the recorded training run (the captured run log, 3 epochs):

| Metric | Final epoch (train) | Final epoch (validation) |
| --- | --- | --- |
| AUC | **0.9987** | **0.9988** |
| Loss | 0.0721 | 0.0852 |

Best validation AUC during the run reached **0.9996** (epoch 2). The model separates human vs. AI text almost perfectly on the validation fold.

## Key Takeaways
- DeBERTa-v3 fine-tuned on rich external data achieves near-perfect validation AUC (~0.999) for AI-text detection.
- KerasNLP keeps the code backend-agnostic (TF/Torch/JAX) and TPU-friendly.
- Stratified K-fold plus heavy external augmentation compensates for the small official train set.
- Such high in-domain AUC can be optimistic — generalization to unseen LLM generators is the real challenge.

## How to Run
```bash
pip install keras_nlp==0.6.3 keras-core==0.1.7 tensorflow pandas numpy scikit-learn wandb
jupyter notebook notebook.ipynb
```
The external CSVs are in `data/`; download the competition data and the large `daigt-proper-train-dataset` if needed (see Dataset). A TPU/GPU is strongly recommended.

## Credit
> Based on ["Detect Fake Text: KerasNLP [TF/Torch/JAX][Train]"](https://www.kaggle.com/code/awsaf49/detect-fake-text-kerasnlp-tf-torch-jax-train) by Awsaf on Kaggle (1,501 votes).
> Datasets: [argugpt](https://www.kaggle.com/datasets/alejopaullier/argugpt), [daigt-external-dataset](https://www.kaggle.com/datasets/alejopaullier/daigt-external-dataset), [daigt-proper-train-dataset](https://www.kaggle.com/datasets/thedrcat/daigt-proper-train-dataset); competition: [LLM - Detect AI Generated Text](https://www.kaggle.com/competitions/llm-detect-ai-generated-text). Adapted and documented for this portfolio.
