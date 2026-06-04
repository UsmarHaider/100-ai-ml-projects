# Fashion MNIST Clothing Classification

![CV](https://img.shields.io/badge/domain-Computer%20Vision-blue)
![Task](https://img.shields.io/badge/task-Classification-green)
![Framework](https://img.shields.io/badge/framework-Keras%2FTensorFlow-orange)

## Overview
This project classifies grayscale clothing images into 10 categories (T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot) using the Fashion-MNIST dataset, a drop-in replacement for the original MNIST. It serves as a beginner-friendly walk-through of building and training a simple CNN with Keras.

## Dataset
- [Fashion MNIST](https://www.kaggle.com/datasets/zalando-research/fashionmnist) (Zalando Research), ~200 MB.
- 70,000 grayscale 28x28 images (60,000 train / 10,000 test), 10 classes; provided as CSV (`fashion-mnist_train.csv`, `fashion-mnist_test.csv`) with a label column + 784 pixel columns.
- The notebook splits the training set 80/20 into train/validation (`random_state=12345`).
- Not committed. Download with:
  ```bash
  kaggle datasets download zalando-research/fashionmnist
  ```
  The notebook reads from `../input/fashionmnist/`.

## Approach
1. **Preprocessing**: read CSVs into NumPy `float32` arrays; scale pixels to [0, 1] by dividing by 255; reshape to `(28, 28, 1)`; 80/20 train/validation split.
2. **Model** (Sequential CNN):
   - `Conv2D(32, 3x3, ReLU)` → `MaxPooling2D(2)` → `Dropout(0.2)`
   - `Flatten` → `Dense(32, ReLU)` → `Dense(10, softmax)`
3. **Training**: Adam (`lr=0.001`), `sparse_categorical_crossentropy`, batch size 4096, 75 epochs, validated on the held-out split.
4. **Evaluation**: `model.evaluate` on the test set plus a `classification_report` (precision/recall/F1 per class).

## Results
The committed notebook has no saved cell outputs and the run log is empty.

| Set | Metrics computed |
|-----|------------------|
| Test | `Test Loss`, `Test Accuracy` (via `model.evaluate`) |
| Test | Per-class precision / recall / F1 (`classification_report`) |

No real numeric values are available in the notebook or logs. Run the notebook to reproduce the metrics. The author notes the classifier underperforms on class 6 (Shirt) and is weaker on classes 2 (Pullover) and 4 (Coat).

## Key Takeaways
- Even a single-conv-block CNN learns Fashion-MNIST reasonably; the architecture is intentionally minimal for teaching.
- Shirts (class 6) are the hardest category, often confused with other tops.
- A very large batch size (4096) is used with 75 epochs; smaller batches or deeper conv stacks would likely improve accuracy.
- Possible improvements: more convolution/pooling blocks, batch normalization, and data augmentation.

## How to Run
```bash
pip install pandas numpy matplotlib scikit-learn keras tensorflow
jupyter notebook notebook.ipynb
```
Download the Fashion MNIST dataset and make the CSVs available under `../input/fashionmnist/`.

## Credit
> Based on ["A Simple CNN Model Beginner Guide !!!!!!"](https://www.kaggle.com/code/pavansanagapati/a-simple-cnn-model-beginner-guide) by Pavan Sanagapati on Kaggle (1613 votes).
> Dataset: [Fashion MNIST](https://www.kaggle.com/datasets/zalando-research/fashionmnist). Adapted and documented for this portfolio.
