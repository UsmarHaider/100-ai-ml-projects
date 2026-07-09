# COVID-19 Chest X-Ray Classification

![CV](https://img.shields.io/badge/domain-Computer%20Vision-blue)
![Task](https://img.shields.io/badge/task-Binary%20Classification-green)

## Overview
Classify chest X-ray images as **negative (Normal)** vs. **positive (COVID-19 pneumonia)** using
transfer learning. The notebook demonstrates a deep-learning screening prototype on the CoronaHack
chest X-ray dataset, with heavy augmentation to offset the scarcity of COVID-positive cases.

## Dataset
- **CoronaHack - Chest X-Ray Dataset** — [kaggle.com/datasets/praveengovi/coronahack-chest-xraydataset](https://www.kaggle.com/datasets/praveengovi/coronahack-chest-xraydataset)
- Metadata in `Chest_xray_Corona_Metadata.csv` with `Label`, `Label_2_Virus_category`,
  `Dataset_type` (TRAIN/TEST); images under `Coronahack-Chest-XRay-Dataset/.../train` and `/test`.
- Built target: Normal -> negative/0, COVID-19 pneumonia -> positive/1 (other pneumonia filtered out).
- Data is **not** committed. Download with:
  ```bash
  kaggle datasets download praveengovi/coronahack-chest-xraydataset
  ```
  so the metadata CSV and `train/`-`test/` image folders are reachable under `../input/coronahack-chest-xraydataset/`.

## Approach
1. **Metadata prep** — load CSV, impute missing values with `'unknown'`, split TRAIN/TEST, derive a
   binary `target` (Normal vs. COVID-19).
2. **Class balancing** — augment the small COVID-positive set with `ImageDataGenerator`
   (`shear_range=0.2`, `zoom_range=0.2`), generating ~20 augmented copies per positive image.
3. **Loading** — read images at **255x255x3**, scale by `1/255`, convert to tensors, build
   `tf.data` datasets, batch (BATCH_SIZE=16, shuffle buffer 1000).
4. **Model** — **ResNet50** (ImageNet weights, `include_top=False`, `trainable=False`) +
   `GlobalAveragePooling2D` -> Dense(128) -> Dropout(0.2) -> Dense(1, sigmoid).
5. **Training** — Adam, binary cross-entropy, accuracy metric, **10 epochs**, `EarlyStopping`
   on `val_loss` (patience 2).
6. **Evaluation** — `predict_classes` on the test arrays for predictions.

## Results
The notebook trains for up to 10 epochs and reports training/validation **accuracy** and **loss**,
then predicts on the test set. The committed notebook has no executed metric outputs and no run log
is available.

| Metric | Value |
|--------|-------|
| Validation accuracy | Run the notebook to reproduce the metric |
| Validation loss | Run the notebook to reproduce the metric |

Run the notebook to reproduce the metrics.

## Key Takeaways
- A frozen ResNet50 backbone with a small trainable head is a fast baseline for X-ray screening.
- The COVID-positive class is heavily under-represented; targeted augmentation is essential to avoid
  a degenerate majority-class classifier.
- Given the class imbalance, accuracy alone is misleading — sensitivity/recall and a confusion matrix
  would be more informative additions.

## How to Run
```bash
pip install tensorflow numpy pandas matplotlib seaborn pillow scikit-learn tqdm
# place the CoronaHack dataset under ../input/coronahack-chest-xraydataset/
jupyter notebook notebook.ipynb
```

## Credit
> Based on ["Deep Learning and Transfer Learning on COVID-19"](https://www.kaggle.com/code/digvijayyadav/deep-learning-and-transfer-learning-on-covid-19) by Digvijay Yadav on Kaggle (222 votes).
> Dataset: [CoronaHack - Chest X-Ray Dataset](https://www.kaggle.com/datasets/praveengovi/coronahack-chest-xraydataset). Adapted and documented for this portfolio.
