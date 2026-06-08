# Skin Cancer Lesion Classification (HAM10000)

![CV](https://img.shields.io/badge/domain-Computer%20Vision-blue)
![Task](https://img.shields.io/badge/task-Multi--class%20Classification-green)
![Framework](https://img.shields.io/badge/framework-Keras%2FTensorFlow-orange)

## Overview
This project classifies dermatoscopic skin-lesion images into 7 cancer/lesion types from the HAM10000 dataset using a convolutional neural network. It walks through a complete pipeline: metadata handling, EDA, image resizing, normalization, a custom CNN, and evaluation with a confusion matrix.

## Dataset
- [Skin Cancer MNIST: HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) (kmader).
- 10,015 dermatoscopic images plus `HAM10000_metadata.csv`; images come from two folders merged via an image-id → path dictionary.
- 7 classes: Melanocytic nevi, Melanoma, Benign keratosis-like lesions, Basal cell carcinoma, Actinic keratoses, Vascular lesions, Dermatofibroma (heavily imbalanced toward Melanocytic nevi).
- Images resized from 450x600 to 100x75; 80/20 train/test split, then 10% of train held out for validation.
- Not committed. Download with:
  ```bash
  kaggle datasets download kmader/skin-cancer-mnist-ham10000
  ```
  The notebook reads from `../input/` (metadata CSV + `*/*.jpg`).

## Approach
1. **Metadata + EDA**: build image-path dictionary, map lesion codes to readable names, fill missing `age` with the mean, explore class/age/sex/localization distributions.
2. **Preprocessing**: load and resize images to 100x75; normalize by subtracting mean and dividing by std; one-hot encode the 7 labels; reshape to `(75, 100, 3)`.
3. **Augmentation** (`ImageDataGenerator`): rotation ±10°, zoom 0.1, width/height shift 0.1.
4. **Model** (Sequential CNN): `In -> [[Conv2D->ReLU]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out`
   - `Conv2D(32) x2` → `MaxPool2D` → `Dropout(0.25)`
   - `Conv2D(64) x2` → `MaxPool2D` → `Dropout(0.40)`
   - `Flatten` → `Dense(128, ReLU)` → `Dropout(0.5)` → `Dense(7, softmax)`
5. **Training**: Adam (`lr=0.001`), `categorical_crossentropy`, batch size 10, 50 epochs, with `ReduceLROnPlateau` (monitor `val_acc`, factor 0.5, patience 3, `min_lr=1e-5`).
6. **Evaluation**: test + validation accuracy/loss (`model.evaluate`), confusion matrix, and per-class error fractions.

## Results
The committed notebook has no saved cell outputs and the run log is empty. The original kernel title advertises **77.0344% accuracy**.

| Set | Metric | Value |
|-----|--------|-------|
| Test | Accuracy | 77.0344% (per the original kernel title) |

The notebook computes test and validation accuracy/loss via `model.evaluate` and a confusion matrix, but no numeric outputs are saved in this copy. Run the notebook to reproduce the metrics. The author notes the most-misclassified class is Basal cell carcinoma, followed by Vascular lesions.

## Key Takeaways
- A modest 4-conv CNN on low-resolution (100x75) images reaches ~77% across 7 imbalanced classes.
- Severe class imbalance (Melanocytic nevi dominates) is the main challenge; minority lesion types are harder.
- Possible improvements: class weighting/oversampling, higher input resolution, transfer learning, and reporting per-class recall (clinically critical for melanoma).

## How to Run
```bash
pip install numpy pandas matplotlib seaborn pillow scikit-learn keras tensorflow
jupyter notebook notebook.ipynb
```
Download the HAM10000 dataset and make the metadata CSV and image folders available under `../input/`.

## Credit
> Based on ["Step wise Approach : CNN Model (77.0344% Accuracy)"](https://www.kaggle.com/code/sid321axn/step-wise-approach-cnn-model-77-0344-accuracy) by Manu Siddhartha on Kaggle (1570 votes).
> Dataset: [Skin Cancer MNIST: HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000). Adapted and documented for this portfolio.
