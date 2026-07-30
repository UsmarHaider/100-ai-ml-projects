# Satellite Image Classification (CNN)

![CV](https://img.shields.io/badge/domain-Computer%20Vision-blue) ![Task](https://img.shields.io/badge/task-Image%20Classification-orange)

## Overview
Classifies 256×256 satellite images into four land-cover categories — Cloudy, Desert,
Green_Area, and Water — with a custom convolutional neural network built in Keras.

## Dataset
- [Satellite Image Classification](https://www.kaggle.com/datasets/mahmoudreda55/satellite-image-classification).
- **4 classes** (Cloudy / Desert / Green_Area / Water), images of size 256×256.
- Data is **not** committed here. Download with:
  ```bash
  kaggle datasets download mahmoudreda55/satellite-image-classification
  ```

## Approach
1. **Index data** — walk the four class folders into a DataFrame of `image_path` / `label`
   and persist it as `image_dataset.csv`.
2. **Split** — `train_test_split(test_size=0.2, random_state=42)`.
3. **Augmentation** — training `ImageDataGenerator` with `rescale=1/255`, shear 0.2,
   zoom 0.2, horizontal + vertical flip, rotation 45°, `fill_mode='nearest'`; test
   generator only rescales. Both use `flow_from_dataframe`, `target_size=(255,255)`,
   batch size 32, `class_mode='categorical'`.
4. **Model (Sequential CNN)** — `Conv2D(32)` → `Conv2D(32)` → `MaxPool` →
   `Conv2D(64)` → `MaxPool` → `Conv2D(128)` → `MaxPool` → `Flatten` →
   `Dense(128, relu)` → `Dropout(0.5)` → `Dense(4, softmax)`.
5. **Training** — Adam, `categorical_crossentropy`, accuracy metric, 5 epochs.
6. **Evaluation & inference** — `model.evaluate` on the test generator, a confusion
   matrix, single-image prediction, and predictions on images fetched from URLs.

## Results
The notebook trains for 5 epochs, plots train/validation loss and accuracy curves,
computes a **confusion matrix** over the four classes, and reports test loss/accuracy via
`model.evaluate`. No executed metric values are saved in the run log.

Run the notebook to reproduce the accuracy, loss curves, and confusion matrix.

## Key Takeaways
- A compact 3-block CNN handles 4-class land-cover classification on visually distinct
  categories.
- Aggressive augmentation (flips, 45° rotation, zoom) is appropriate since satellite
  imagery is orientation-invariant.
- The notebook ends with a practical demo: classifying arbitrary satellite images fetched
  from public URLs, showing the model generalizing beyond the dataset.
- The model is saved to `Model.h5` for reuse without retraining.
- Improvements: more epochs, a held-out validation split distinct from the test set, and
  transfer learning for finer-grained land cover.

## How to Run
```bash
pip install tensorflow keras scikit-learn numpy pandas matplotlib pillow requests
jupyter notebook notebook.ipynb
```
Point the `labels` dictionary at the downloaded `data/` class folders.

## Credit
> Based on ["🛰️Satellite Image Classification | CNN"](https://www.kaggle.com/code/sujithmandala/satellite-image-classification-cnn) by Sujith K Mandala on Kaggle (389 votes).
> Dataset: [Satellite Image Classification](https://www.kaggle.com/datasets/mahmoudreda55/satellite-image-classification). Adapted and documented for this portfolio.
