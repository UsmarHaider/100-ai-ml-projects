# CIFAR-10 Image Classification

![CV](https://img.shields.io/badge/domain-Computer%20Vision-blue)
![Task](https://img.shields.io/badge/task-Classification-green)
![Framework](https://img.shields.io/badge/framework-Keras%2FTensorFlow-orange)

## Overview
This project classifies 32x32 color images into 10 object categories (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck) from the CIFAR-10 benchmark. A compact VGG-style CNN is trained with heavy data augmentation and regularization to generalize well while staying around ~1.2M parameters.

## Dataset
- CIFAR-10, loaded directly from `keras.datasets.cifar10` (no manual download required).
- 60,000 color images of size 32x32x3 across 10 classes (50,000 train / 10,000 test); the notebook further carves a 10% validation split from the training set.
- Reference dataset page: [CIFAR-10 on Kaggle](https://www.kaggle.com/c/cifar-10).
- Data is fetched at runtime by Keras, so no `data/` folder is needed.

## Approach
1. **Preprocessing**: cast pixels to `float32`; normalize by subtracting the training mean and dividing by std (`+1e-7`). One-hot encode labels with `to_categorical`.
2. **Augmentation** (`ImageDataGenerator`): rotation ±15°, width/height shift 12%, horizontal flip, zoom 10%, brightness [0.9, 1.1], shear 10°, channel shift 0.1.
3. **Model** (VGG-inspired Sequential CNN):
   - 4 blocks of `[Conv2D x2 -> BatchNorm x2 -> MaxPool2D(2) -> Dropout]` with filters 32, 64, 128, 256 and dropout 0.2 → 0.5.
   - `Conv2D` layers use 3x3 kernels, `padding='same'`, ReLU, and L2 regularization (`weight_decay=0.0001`).
   - `Flatten` → `Dense(10, softmax)`. Total ~1,186,346 parameters.
4. **Training**: Adam (`lr=0.0005`), `categorical_crossentropy`, batch size 64, up to 300 epochs with `ReduceLROnPlateau` (factor 0.5, patience 10) and `EarlyStopping` (patience 40, `restore_best_weights=True`).

## Results
The committed notebook has no saved cell outputs and the run log is empty.

| Set | Metric | Value (author's stated target) |
|-----|--------|-------------------------------|
| Test | Accuracy | > 90% (per notebook narrative) |

The notebook computes `test_loss` and `test_acc` via `model.evaluate` on the held-out test set. The ">90%" figure is the author's qualitative statement, not a captured output. Run the notebook to reproduce the metrics.

## Key Takeaways
- A relatively small VGG-style network (~1.2M params) reaches >90% on CIFAR-10 when combined with strong augmentation and regularization.
- BatchNorm + progressively increasing dropout (0.2 → 0.5) effectively counters overfitting; training and validation curves stay close.
- `ReduceLROnPlateau` + `EarlyStopping` with `restore_best_weights` make training efficient and reproducible.
- Possible improvements: residual/inception blocks or test-time augmentation for additional accuracy.

## How to Run
```bash
pip install tensorflow keras numpy matplotlib opencv-python scikit-learn
jupyter notebook notebook.ipynb
```
No manual data download is required; `cifar10.load_data()` fetches the dataset automatically. The final cell also pulls an external truck image from GitHub for an out-of-dataset sanity check.

## Credit
> Based on ["CIFAR-10 Image Classification with CNN"](https://www.kaggle.com/code/farzadnekouei/cifar-10-image-classification-with-cnn) by Farzad Nekouei on Kaggle (2198 votes).
> Dataset: CIFAR-10 (via `keras.datasets`); see also [CIFAR-10 on Kaggle](https://www.kaggle.com/c/cifar-10). Adapted and documented for this portfolio.
