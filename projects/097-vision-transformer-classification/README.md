# Image Classification with Vision Transformers (ViT)

![CV](https://img.shields.io/badge/domain-Computer%20Vision-blue)
![Task](https://img.shields.io/badge/task-image%20classification-orange)
![Model](https://img.shields.io/badge/model-Vision%20Transformer-purple)

## Overview
A tutorial-plus-baseline that explains how Vision Transformers work and then fine-tunes a pretrained ViT to classify cassava leaf disease into 5 categories. ViT replaces convolutions with self-attention over image patches, treating an image as a sequence of "16×16 words". This baseline shows how to apply a transformer backbone to a real Kaggle image-classification competition.

## Dataset
- **Competition:** [Cassava Leaf Disease Classification](https://www.kaggle.com/competitions/cassava-leaf-disease-classification) — 5 disease/health classes of cassava leaf images.
- **Pretrained ViT weights:** [vit-base-models-pretrained-pytorch](https://www.kaggle.com/datasets/abhinand05/vit-base-models-pretrained-pytorch) (~1.39 GB).
- **Tutorial illustrations:** [vittutorialillustrations](https://www.kaggle.com/datasets/abhinand05/vittutorialillustrations) — committed locally in `data/` (architecture/attention diagrams).
- Competition data and pretrained weights are **not** committed. Get them via:
  ```bash
  kaggle competitions download -c cassava-leaf-disease-classification
  kaggle datasets download abhinand05/vit-base-models-pretrained-pytorch
  ```

## Approach
1. **Setup** — install `torch-xla` (TPU) and `timm` (PyTorch image models with pretrained ViT weights); seed everything for reproducibility.
2. **Data** — read competition `train_images/` with a stratified split; standard image transforms via `torchvision.transforms`.
3. **Model** — pretrained **ViT-base** loaded through `timm`, with the classification head adapted to 5 classes.
4. **Training** — custom per-batch / per-epoch training loop on **TPU** (`XLA_USE_BF16=1`), with validation each epoch and checkpointing on validation-loss improvement.

## Results
Real values extracted from the recorded training run:

| Epoch | Train loss | Train acc | Valid loss | Valid acc |
| --- | --- | --- | --- | --- |
| 7 | 0.301 | 0.941 | 0.406 | 0.855 |
| 8 | 0.285 | 0.941 | **0.406** (best, checkpointed) | 0.871 |
| 9 | 0.289 | 0.945 | 0.418 | 0.871 |
| 10 | 0.281 | 0.945 | 0.453 | 0.840 |

Best validation loss ≈ **0.406**, peak validation accuracy ≈ **0.871**. Total run time ≈ **33m 39s**.

## Key Takeaways
- A pretrained ViT-base fine-tuned via `timm` reaches ~87% validation accuracy on cassava leaf disease within ~10 epochs.
- TPU + bfloat16 makes transformer training on images practical.
- Validation loss bottoms out around epoch 8 and then drifts up — early stopping / checkpoint-on-best is the right call (and is used here).
- The notebook doubles as a clear conceptual walkthrough of patch embeddings, positional embeddings, and multi-headed self-attention.

## How to Run
```bash
pip install torch torchvision timm numpy pandas matplotlib scikit-learn pillow tqdm
# TPU runtime additionally requires torch-xla (see the notebook's setup cell)
jupyter notebook notebook.ipynb
```
Download the cassava competition data and pretrained ViT weights (see Dataset). The diagram images are already in `data/`.

## Credit
> Based on ["Vision Transformer (ViT): Tutorial + Baseline"](https://www.kaggle.com/code/abhinand05/vision-transformer-vit-tutorial-baseline) by Abhinand on Kaggle (2,293 votes).
> Datasets: [pretrained ViT weights](https://www.kaggle.com/datasets/abhinand05/vit-base-models-pretrained-pytorch), [tutorial illustrations](https://www.kaggle.com/datasets/abhinand05/vittutorialillustrations); competition: [Cassava Leaf Disease Classification](https://www.kaggle.com/competitions/cassava-leaf-disease-classification). Adapted and documented for this portfolio.
