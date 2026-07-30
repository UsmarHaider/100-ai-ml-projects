# Age and Gender Detection from Faces

![CV](https://img.shields.io/badge/domain-Computer%20Vision-blue)
![Task](https://img.shields.io/badge/task-multi--output%20(classification%20%2B%20regression)-orange)

## Overview
Predict both **gender** (binary classification) and **age** (regression) directly from a face image. The approach uses a single multi-output convolutional network that shares a feature extractor and branches into two heads. Automated face attribute estimation is widely used in retail analytics, access control, and demographic research.

## Dataset
- **UTKFace** — [kaggle.com/datasets/jangedoo/utkface-new](https://www.kaggle.com/datasets/jangedoo/utkface-new)
- ~23,708 face images. Labels are encoded in each filename as `age_gender_race_*.jpg` (e.g. `1_0_0_239389.jpg` → age 1, gender 0/Male). Race is ignored for this task.
- Images are converted to grayscale and resized to **128×128**.
- Data is **not** committed. Download with:
  ```bash
  kaggle datasets download jangedoo/utkface-new
  unzip utkface-new.zip
  ```
  The notebook expects the images under `UTKFace/`.

## Approach
1. **Parse labels** — read filenames, split on `_` to extract age and gender into a DataFrame.
2. **Subsample** — `train_test_split(test_size=0.85)` keeps a smaller subset because the full set is slow to train.
3. **Feature extraction** — load each image grayscale, resize to 128×128, stack into a `(N, 128, 128, 1)` array, normalize by `/255`.
4. **Model** — Keras functional multi-output CNN:
   - `Conv2D(64, 3×3, relu)` → `BatchNorm` → `MaxPool(3×3)`
   - `Conv2D(128, 3×3, relu)` → `MaxPool(2×2)`
   - `Conv2D(256, 3×3, relu)` → `MaxPool(2×2)` → `Flatten`
   - Two dense branches with `Dropout(0.4)`
   - **gender_output**: `Dense(1, sigmoid)`; **age_output**: `Dense(1, relu)`
5. **Training** — `optimizer='adam'`, losses `['binary_crossentropy', 'mae']`, `batch_size=10`, `epochs=20`, `validation_split=0.1`.

## Results
The notebook trains the model and plots gender/age train vs. validation loss curves, then runs sample predictions. No executed metric outputs are present in the run log.

| Metric | Value |
| --- | --- |
| Gender accuracy | Run the notebook to reproduce |
| Age MAE | Run the notebook to reproduce |

The author's markdown notes the model **overfits** by the end of training and suggests adding regularization. Run the notebook to reproduce the metrics.

## Key Takeaways
- A shared CNN backbone with two task-specific heads handles classification and regression jointly.
- Filename-encoded labels make UTKFace easy to ingest without a separate annotation file.
- Grayscale + downsampling and using only ~15% of the data trade accuracy for training speed.
- The model overfits; dropout, data augmentation, or more training data would likely help.

## How to Run
```bash
pip install numpy pandas matplotlib seaborn pillow tensorflow scikit-learn
jupyter notebook notebook.ipynb
```
Download UTKFace (see Dataset) and place the images under `UTKFace/` next to the notebook.

## Credit
> Based on ["Age and Gender Detection"](https://www.kaggle.com/code/uzzivirus/age-and-gender-detection) by Uzair Siddiqui on Kaggle (641 votes).
> Dataset: [UTKFace](https://www.kaggle.com/datasets/jangedoo/utkface-new). Adapted and documented for this portfolio.
