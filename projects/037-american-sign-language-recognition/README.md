# American Sign Language Recognition

![CV](https://img.shields.io/badge/domain-Computer%20Vision-blue)
![Task](https://img.shields.io/badge/task-Image%20Classification-green)

## Overview
Recognize static American Sign Language (ASL) hand gestures for letters A-Z from 28x28 grayscale
images. Gesture recognition is a step toward assistive communication tools for the deaf and hard
of hearing. The approach is a CNN trained on the Sign Language MNIST dataset.

## Dataset
- **Sign Language MNIST** — [kaggle.com/datasets/datamunge/sign-language-mnist](https://www.kaggle.com/datasets/datamunge/sign-language-mnist)
- **24 classes** (letters A-Z excluding J and Z, which require motion).
- 27,455 training and 7,172 test cases; each row is a label + 784 pixels (28x28 grayscale, 0-255).
- Data is **not** committed (~201 MB). Download with:
  ```bash
  kaggle datasets download datamunge/sign-language-mnist
  ```
  so `sign_mnist_train.csv` and `sign_mnist_test.csv` are reachable.

## Approach
1. **Load** train/test CSVs; separate the `label` column; reshape pixels to `(-1, 28, 28, 1)`.
2. **Encode labels** with `LabelBinarizer` (24-way one-hot).
3. **Augment** the training set with `ImageDataGenerator` (rescale `1/255`, height/width shift 0.2,
   zoom 0.2, horizontal flip, `fill_mode='nearest'`); test set rescaled by `1/255`.
4. **Model** — Sequential CNN:
   - Conv2D(128, 5x5, same, ReLU) + MaxPool(3x3, stride 2)
   - Conv2D(64, 2x2, same, ReLU) + MaxPool(2x2, stride 2)
   - Conv2D(32, 2x2, same, ReLU) + MaxPool(2x2, stride 2)
   - Flatten -> Dense(512, ReLU) -> Dropout(0.25) -> Dense(24, softmax)
5. **Training** — Adam, categorical cross-entropy, batch size 200, 35 epochs.
6. **Evaluation** — `model.evaluate` reports test loss and accuracy.

## Results
The notebook prints test accuracy via `model.evaluate` (`MODEL ACCURACY = {acc*100}%`). The
committed notebook has no executed outputs and no run log is available. The original kernel title
claims **99.40% accuracy**.

| Metric | Value |
|--------|-------|
| Test accuracy | Reported by `model.evaluate`; kernel title states 99.40% |
| Test loss | Reported by `model.evaluate` |

Run the notebook to reproduce the metrics.

## Key Takeaways
- A shallow 3-conv CNN reaches very high accuracy on the well-aligned 28x28 ASL-MNIST images.
- Augmentation guards against memorization despite the small per-image resolution.
- Note: the dataset excludes the dynamic letters J and Z, so this is not full-alphabet recognition;
  a video/temporal model would be needed for those.

## How to Run
```bash
pip install tensorflow keras numpy pandas matplotlib seaborn scikit-learn
# place sign_mnist_train.csv and sign_mnist_test.csv under ../input/sign-language-mnist/
jupyter notebook notebook.ipynb
```

## Credit
> Based on ["Sign-Language Classification CNN (99.40% Accuracy)"](https://www.kaggle.com/code/sayakdasgupta/sign-language-classification-cnn-99-40-accuracy) by sayak on Kaggle (900 votes).
> Dataset: [Sign Language MNIST](https://www.kaggle.com/datasets/datamunge/sign-language-mnist). Adapted and documented for this portfolio.
