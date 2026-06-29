# Traffic Sign Recognition (GTSRB)

![CV](https://img.shields.io/badge/domain-Computer%20Vision-blue)
![Task](https://img.shields.io/badge/task-Image%20Classification-green)

## Overview
Classify German traffic signs into 43 categories from the GTSRB benchmark. Reliable traffic-sign
recognition underpins driver-assistance and autonomous-driving systems. The approach is a deep
convolutional network trained on 32x32 RGB sign crops.

## Dataset
- **GTSRB - German Traffic Sign Recognition Benchmark** — [kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)
- **43 classes**, more than 50,000 images combined across train and test.
- Images read with PIL, resized to **32x32x3**, cast to `float32`, labels one-hot encoded.
- Data is **not** committed. Download with:
  ```bash
  kaggle datasets download meowmeowmeowmeowmeow/gtsrb-german-traffic-sign
  ```
  Unzip so the `Train/<class_id>/` folders are reachable from `../input/gtsrb-german-traffic-sign/Train`.

## Approach
1. **Load** all training images by walking the 43 class folders; resize each to 32x32.
2. **Preprocess** — `float32` images, `to_categorical` labels (43 classes).
3. **Split** — 80% train, then the remaining 20% split into validation/test (stratified train split).
4. **Model** — Sequential CNN:
   - Conv2D(32, 3x3, same, ReLU) + MaxPool + BatchNorm + Dropout(0.3)
   - Conv2D(128, 3x3, same, ReLU) + MaxPool + BatchNorm + Dropout(0.3)
   - Conv2D(512, 3x3, same, ReLU) + Dropout(0.3) + Conv2D(512, 3x3, same, ReLU) + MaxPool + BatchNorm
   - Flatten -> Dense(4000, ReLU) -> Dense(4000, ReLU) -> Dense(1000, ReLU) -> Dense(43, softmax)
5. **Training** — Adam, categorical cross-entropy, batch size 64, 20 epochs.
6. **Evaluation** — accuracy on the held-out test split via `accuracy_score`.

## Results
Metrics below are extracted from the actual notebook run log:

| Metric | Value |
|--------|-------|
| Test set accuracy | **99.52%** |
| Final training accuracy | 99.03% |
| Final validation accuracy | 99.29% |
| Best epoch val_loss / val_accuracy | 0.0310 / 0.9926 |

These confirm the kernel title's claim of 99% accuracy.

## Key Takeaways
- Stacking BatchNorm + Dropout after each conv block lets a wide CNN train stably to >99% accuracy.
- Very large dense layers (2x 4000 units) add capacity but most discriminative power comes from the
  convolutional stack; they are a candidate for trimming.
- Validation accuracy is occasionally volatile across epochs (one dip to ~92.6%), suggesting a
  learning-rate schedule could smooth convergence.

## How to Run
```bash
pip install tensorflow numpy pandas matplotlib seaborn pillow scikit-learn
# place GTSRB so ../input/gtsrb-german-traffic-sign/Train/<id>/ is reachable
jupyter notebook notebook.ipynb
```

## Credit
> Based on ["Traffic Sign Detection using CNNs 99% Accuracy"](https://www.kaggle.com/code/raghav2002sharma/traffic-sign-detection-using-cnns-99-accuracy) by Raghav Sharma on Kaggle (193 votes).
> Dataset: [GTSRB - German Traffic Sign](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign). Adapted and documented for this portfolio.
