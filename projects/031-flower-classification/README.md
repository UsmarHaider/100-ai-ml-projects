# Flower Species Image Classification

![CV](https://img.shields.io/badge/domain-Computer%20Vision-blue)
![Task](https://img.shields.io/badge/task-Image%20Classification-green)

## Overview
Classify photographs of flowers into five species (daisy, sunflower, tulip, dandelion, rose).
Automatic flower recognition is a classic fine-grained image classification problem and a good
testbed for convolutional networks. The approach is a from-scratch CNN trained on augmented
150x150 RGB images.

## Dataset
- **Flowers Recognition** — [kaggle.com/datasets/alxmamaev/flowers-recognition](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition)
- 5 classes: daisy, sunflower, tulip, dandelion, rose (~4,300 images total).
- Images are loaded with OpenCV, resized to **150x150x3**, normalized to `[0,1]`, label-encoded
  and one-hot encoded. Split 75% train / 25% validation.
- Data is **not** committed. Download with:
  ```bash
  kaggle datasets download alxmamaev/flowers-recognition
  ```
  Unzip so that `flowers/flowers/<class>/...` is reachable, then adjust the directory paths
  in the notebook.

## Approach
1. **Load & resize** every image to 150x150 with OpenCV; build `X` (pixels) and `Z` (labels).
2. **Encode labels** with `LabelEncoder` + `to_categorical` (5 classes); scale pixels by `1/255`.
3. **Split** with `train_test_split(test_size=0.25, random_state=42)`.
4. **Augment** the training set with `ImageDataGenerator` (rotation 10 deg, zoom 0.1,
   width/height shift 0.2, horizontal flip).
5. **Model** — Sequential CNN:
   - Conv2D(32, 5x5, same, ReLU) + MaxPool(2x2)
   - Conv2D(64, 3x3, same, ReLU) + MaxPool(2x2)
   - Conv2D(96, 3x3, same, ReLU) + MaxPool(2x2)
   - Conv2D(96, 3x3, same, ReLU) + MaxPool(2x2)
   - Flatten -> Dense(512, ReLU) -> Dense(5, softmax)
6. **Training** — Adam (lr=0.001), categorical cross-entropy, batch size 128, 50 epochs,
   `ReduceLROnPlateau(monitor='val_acc', patience=3, factor=0.1)`.

## Results
The notebook reports training/validation **accuracy** and **loss** curves and visualizes
correctly vs. misclassified validation images. The committed notebook has no executed metric
outputs and no run log is available.

| Metric | Value |
|--------|-------|
| Validation accuracy | Run the notebook to reproduce the metric |
| Training/validation loss | Run the notebook to reproduce the metric |

Run the notebook to reproduce the metrics.

## Key Takeaways
- A compact 4-block CNN is sufficient for a 5-class flower task at 150x150 resolution.
- Augmentation plus an LR annealer are the main regularizers against overfitting on a small dataset.
- Possible improvements: transfer learning (ResNet/EfficientNet), class-balanced sampling, and
  reporting a confusion matrix / per-class F1.

## How to Run
```bash
pip install tensorflow keras numpy pandas matplotlib seaborn scikit-learn opencv-python tqdm pillow
# place the flowers dataset so flowers/flowers/<class>/ is reachable
jupyter notebook notebook.ipynb
```

## Credit
> Based on ["Flower Recognition CNN Keras"](https://www.kaggle.com/code/rajmehra03/flower-recognition-cnn-keras) by Raj Mehrotra on Kaggle (3253 votes).
> Dataset: [Flowers Recognition](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition). Adapted and documented for this portfolio.
