# Chest X-Ray Pneumonia Detection

![CV](https://img.shields.io/badge/domain-Computer%20Vision-blue)
![Task](https://img.shields.io/badge/task-Binary%20Classification-green)
![Framework](https://img.shields.io/badge/framework-Keras%2FTensorFlow-orange)

## Overview
This project detects pneumonia from chest X-ray images, a binary classification problem (PNEUMONIA vs. NORMAL). A custom 5-block convolutional neural network is trained from scratch on grayscale X-rays with data augmentation to handle class imbalance.

## Dataset
- [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) by Paul Timothy Mooney.
- 5,863 JPEG chest X-rays in 2 categories (Pneumonia / Normal), organized into `train` / `test` / `val` folders with per-class subfolders.
- Images are loaded grayscale and resized to 150x150.
- Not committed. Download with:
  ```bash
  kaggle datasets download paultimothymooney/chest-xray-pneumonia
  ```
  The notebook reads from `../input/chest-xray-pneumonia/chest_xray/chest_xray/{train,test,val}`.

## Approach
1. **Preprocessing**: read images grayscale, resize to 150x150, normalize to [0, 1], reshape to `(150, 150, 1)`.
2. **Augmentation** (`ImageDataGenerator`): rotation ±30°, zoom 0.2, width/height shift 0.1, horizontal flip — to expand data and counter class imbalance.
3. **Model** (Sequential CNN from scratch):
   - 5 conv blocks with filters 32, 64, 64, 128, 256, each `Conv2D(3x3, padding='same', ReLU)` → (Dropout on some) → `BatchNormalization` → `MaxPool2D(2)`.
   - `Flatten` → `Dense(128, ReLU)` → `Dropout(0.2)` → `Dense(1, sigmoid)`.
   - Compiled with `rmsprop`, `binary_crossentropy`, accuracy.
4. **Training**: `model.fit` on augmented data, batch size 32, 12 epochs, with `ReduceLROnPlateau` (monitor `val_accuracy`, factor 0.3, patience 2, `min_lr=1e-6`).
5. **Evaluation**: test accuracy/loss via `model.evaluate`, `classification_report`, and confusion matrix; correct/incorrect samples are visualized.

## Results
The committed notebook has no saved cell outputs and the run log is empty. The original kernel title advertises **92.6% accuracy**.

| Set | Metric | Value |
|-----|--------|-------|
| Test | Accuracy | 92.6% (per the original kernel title) |

The notebook itself computes test loss/accuracy (`model.evaluate`), per-class precision/recall/F1 (`classification_report`), and a confusion matrix, but no numeric outputs are saved in this copy. Run the notebook to reproduce the metrics.

## Key Takeaways
- A from-scratch CNN with BatchNorm + dropout can reach ~92.6% on this dataset without transfer learning.
- The dataset is imbalanced (more pneumonia than normal); augmentation and `ReduceLROnPlateau` help stabilize training.
- The provided `val` split is tiny, so test-set metrics are the more meaningful signal.
- Possible improvements: transfer learning, class weighting, and reporting recall/specificity (clinically important for screening).

## How to Run
```bash
pip install numpy pandas matplotlib seaborn opencv-python scikit-learn keras tensorflow
jupyter notebook notebook.ipynb
```
Download the Chest X-Ray Pneumonia dataset and place it under `../input/chest-xray-pneumonia/` as referenced in the notebook.

## Credit
> Based on ["Pneumonia Detection using CNN(92.6% Accuracy)"](https://www.kaggle.com/code/madz2000/pneumonia-detection-using-cnn-92-6-accuracy) by Madhav Mathur on Kaggle (5633 votes).
> Dataset: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia). Adapted and documented for this portfolio.
