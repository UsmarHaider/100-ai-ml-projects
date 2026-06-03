# MNIST Handwritten Digit Classification

![CV](https://img.shields.io/badge/domain-Computer%20Vision-blue)
![Task](https://img.shields.io/badge/task-Classification-green)
![Framework](https://img.shields.io/badge/framework-PyTorch-orange)

## Overview
This project recognizes handwritten digits (0-9) from the classic MNIST/Digit Recognizer dataset. It is structured as a PyTorch tutorial that builds up from linear regression to a convolutional neural network, showing how added depth and convolution improve classification accuracy. The goal is a clean, didactic walk-through of the core PyTorch building blocks applied to image classification.

## Dataset
- [Digit Recognizer](https://www.kaggle.com/competitions/digit-recognizer) (Kaggle competition, MNIST handwritten digits).
- 28x28 grayscale images (784 pixels) flattened in `train.csv`, with a `label` column of 10 classes (0-9).
- 80/20 train/test split inside the notebook (`train_test_split`, `test_size=0.2`, `random_state=42`).
- Not committed. Download the competition data:
  ```bash
  kaggle competitions download -c digit-recognizer
  ```
  Place `train.csv` where the notebook expects it (the notebook reads `../input/train.csv`).

## Approach
1. **Preprocessing**: load `train.csv`, separate pixels/labels, normalize pixels by dividing by 255, split 80/20, wrap in `TensorDataset` + `DataLoader` (`batch_size=100`).
2. **Linear / Logistic Regression** (warm-ups): single `nn.Linear` layers, `CrossEntropyLoss`, SGD optimizer.
3. **ANN**: `784 -> 150 -> 150 -> 150 -> 10` with ReLU, Tanh, and ELU activations; SGD (`lr=0.02`), CrossEntropyLoss.
4. **CNN** (main model):
   - `Conv2d(1->16, kernel=5)` + ReLU + `MaxPool2d(2)`
   - `Conv2d(16->32, kernel=5)` + ReLU + `MaxPool2d(2)`
   - Flatten + `Linear(32*4*4 -> 10)`
   - SGD (`lr=0.1`), CrossEntropyLoss, ~2500 iterations.

## Results
The notebook prints test accuracy at intervals during training but the committed copy has no saved cell outputs, and the run log is empty.

| Model | Metric | Value (author's stated target) |
|-------|--------|-------------------------------|
| Logistic Regression | Test accuracy | ~85% (per notebook narrative) |
| ANN | Test accuracy | ~95% (per notebook narrative) |
| CNN | Test accuracy | ~98% (per notebook narrative) |

These figures are the author's qualitative descriptions in the markdown, not captured cell outputs. Run the notebook to reproduce the metrics (accuracy is computed against the held-out test split each 50 iterations).

## Key Takeaways
- A clear progression: logistic regression < ANN < CNN in accuracy, demonstrating the value of non-linearity and convolution.
- Normalizing pixels to [0, 1] and using `CrossEntropyLoss` (which folds in softmax) keeps the model code minimal.
- Possible improvements: more epochs/iterations, learning-rate scheduling, data augmentation, and padding in conv layers to preserve spatial information.

## How to Run
```bash
pip install numpy pandas matplotlib torch scikit-learn
jupyter notebook notebook.ipynb
```
Download the Digit Recognizer competition data and make `train.csv` available at the path the notebook reads (`../input/train.csv`).

## Credit
> Based on ["Pytorch Tutorial for Deep Learning Lovers"](https://www.kaggle.com/code/kanncaa1/pytorch-tutorial-for-deep-learning-lovers) by DATAI on Kaggle (5193 votes).
> Dataset: [Digit Recognizer competition](https://www.kaggle.com/competitions/digit-recognizer). Adapted and documented for this portfolio.
