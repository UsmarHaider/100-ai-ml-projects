# Cats vs Dogs Image Classification

![CV](https://img.shields.io/badge/domain-Computer%20Vision-blue)
![Task](https://img.shields.io/badge/task-Binary%20Classification-green)
![Framework](https://img.shields.io/badge/framework-Keras%2FTensorFlow-orange)

## Overview
This project builds a CNN that classifies images as either a cat or a dog using the Kaggle Dogs vs. Cats dataset. It demonstrates an end-to-end Keras image pipeline with `ImageDataGenerator`-based augmentation, callbacks, and submission generation for the competition test set.

## Dataset
- [Dogs vs. Cats](https://www.kaggle.com/competitions/dogs-vs-cats) (Kaggle competition).
- ~25,000 labeled training images (≈12,000 cats + ≈12,000 dogs); labels are parsed from filenames (`dog.*` / `cat.*`). A separate unlabeled `test1` set is used for prediction.
- The notebook resizes all images to 128x128x3 and splits the labeled data 80/20 into train/validation (`random_state=42`).
- Not committed. Download with:
  ```bash
  kaggle competitions download -c dogs-vs-cats
  ```
  The notebook reads from `../input/train/train/` and `../input/test1/test1/`.

## Approach
1. **Data prep**: build a DataFrame of filenames + categories; map `0->cat`, `1->dog`; 80/20 train/validation split.
2. **Augmentation** (train `ImageDataGenerator`): rescale 1/255, rotation ±15°, shear 0.1, zoom 0.2, horizontal flip, width/height shift 0.1; validation/test only rescaled.
3. **Model** (Sequential CNN):
   - 3 blocks of `[Conv2D(32/64/128, 3x3, ReLU) -> BatchNorm -> MaxPool2D(2) -> Dropout(0.25)]`
   - `Flatten` → `Dense(512, ReLU)` → `BatchNorm` → `Dropout(0.5)` → `Dense(2, softmax)`
   - Compiled with `rmsprop`, `categorical_crossentropy`, accuracy metric.
4. **Training**: `fit_generator` for up to 50 epochs, batch size 15, with `EarlyStopping(patience=10)` and `ReduceLROnPlateau` (monitor `val_acc`, factor 0.5, patience 2, `min_lr=1e-5`).
5. **Inference**: predict on `test1`, take `argmax`, map back to labels, and write `submission.csv`.

## Results
The committed notebook has no saved cell outputs and the run log is empty.

| Set | Metrics computed |
|-----|------------------|
| Train / Validation | `accuracy`, `loss`, `val_accuracy`, `val_loss` (tracked per epoch) |

No real numeric values are available in the notebook or logs. Run the notebook to reproduce the metrics (training/validation accuracy and loss are plotted from `history`).

## Key Takeaways
- A 3-block CNN with BatchNorm and dropout is a solid baseline for binary natural-image classification.
- Augmentation plus `EarlyStopping`/`ReduceLROnPlateau` callbacks help curb overfitting on the relatively large image set.
- Possible improvements: transfer learning (e.g. VGG/ResNet), larger batch sizes, and higher input resolution.

## How to Run
```bash
pip install numpy pandas matplotlib scikit-learn keras tensorflow
jupyter notebook notebook.ipynb
```
Download the Dogs vs. Cats competition data and place the images under `../input/train/train/` and `../input/test1/test1/`.

## Credit
> Based on ["Keras CNN Dog or Cat Classification"](https://www.kaggle.com/code/uysimty/keras-cnn-dog-or-cat-classification) by Uysim on Kaggle (4696 votes).
> Dataset: [Dogs vs. Cats competition](https://www.kaggle.com/competitions/dogs-vs-cats). Adapted and documented for this portfolio.
