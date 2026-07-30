# Transfer Learning with ResNet50 — Imbalanced Garbage Classification

![CV](https://img.shields.io/badge/domain-Computer%20Vision-blue) ![Task](https://img.shields.io/badge/task-Image%20Classification-orange) ![Transfer Learning](https://img.shields.io/badge/method-Transfer%20Learning-brightgreen)

## Overview
Classifies waste images into six recyclable categories to support automated waste
segregation. The project tackles class imbalance head-on and compares a ResNet50 built
from scratch against a fine-tuned ImageNet-pretrained ResNet50 (transfer learning).

## Dataset
- [Trash Type Image Dataset](https://www.kaggle.com/datasets/farzadnekouei/trash-type-image-dataset).
- **6 classes** with a noticeable imbalance: cardboard 403, glass 501, metal 410,
  paper 594, plastic 482, trash 137 (~2,527 images total).
- Data is **not** committed here. Download with:
  ```bash
  kaggle datasets download farzadnekouei/trash-type-image-dataset
  ```

## Approach
1. **Dataset analysis** — gather all image paths into a DataFrame and quantify the
   per-class imbalance.
2. **Stratified split** — `train_test_split(test_size=0.2, stratify=label, random_state=42)`
   to preserve class ratios.
3. **Augmentation & rescaling** — training generator applies on-the-fly augmentation +
   `rescale=1/255`; validation generator only rescales. Images resized to **384×384**,
   `class_mode='categorical'`, batch size 32, fed via `flow_from_dataframe`.
4. **Class imbalance** — `compute_class_weight` (scikit-learn) supplies inverse-frequency
   class weights to the loss.
5. **Models** —
   - *From scratch*: a custom ResNet50 architecture (Conv/BatchNorm/Add residual blocks).
   - *Transfer learning*: `ResNet50(weights='imagenet')` backbone with ResNet-specific
     `preprocess_input`, `GlobalAveragePooling2D` head → dense softmax over 6 classes.
6. **Training** — Adam optimizer, `EarlyStopping` and `ReduceLROnPlateau` callbacks,
   up to 50 epochs.

## Results
From the recorded transfer-learning run (early-stopped at epoch 45):

| Metric | Value |
| --- | --- |
| Peak validation accuracy | **0.9545** (epoch 42) |
| Final-epoch val accuracy | 0.9526 |
| Test accuracy | **0.95** |
| Macro avg F1 | 0.94 |
| Weighted avg F1 | 0.95 |

Per-class test report (support 506):

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| cardboard | 0.97 | 0.96 | 0.97 | 81 |
| glass | 0.97 | 0.94 | 0.95 | 100 |
| metal | 0.95 | 0.98 | 0.96 | 82 |
| paper | 0.93 | 0.96 | 0.95 | 119 |
| plastic | 0.96 | 0.93 | 0.94 | 97 |
| trash | 0.86 | 0.93 | 0.89 | 27 |

## Key Takeaways
- Transfer learning with a pretrained ResNet50 reaches ~95% test accuracy on a small,
  imbalanced dataset — far stronger than training from scratch.
- Class weights + augmentation keep the minority `trash` class usable (F1 0.89) despite
  only 137 samples.
- Validation accuracy plateaus while train accuracy approaches 1.0, so early stopping and
  LR reduction are essential to curb overfitting.

## How to Run
```bash
pip install tensorflow keras scikit-learn numpy pandas matplotlib seaborn
jupyter notebook notebook.ipynb
```
Point `dataset_path` at the downloaded `TrashType_Image_Dataset` folder.

## Credit
> Based on ["Imbalanced Garbage Classification | ResNet50"](https://www.kaggle.com/code/farzadnekouei/imbalanced-garbage-classification-resnet50) by Farzad Nekouei on Kaggle (968 votes).
> Dataset: [Trash Type Image Dataset](https://www.kaggle.com/datasets/farzadnekouei/trash-type-image-dataset). Adapted and documented for this portfolio.
