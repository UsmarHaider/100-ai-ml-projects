# Blood Cell Image Classification

![CV](https://img.shields.io/badge/domain-Computer%20Vision-blue)
![Task](https://img.shields.io/badge/task-image%20classification-orange)

## Overview
Classify microscope images of blood cells into white-blood-cell subtypes (Neutrophil, Eosinophil, Monocyte, Lymphocyte) and into the broader Mononuclear vs. Polynuclear grouping. Automating this normally manual diagnostic step can speed up blood analysis and reduce human error.

## Dataset
- **Blood Cells** — [kaggle.com/datasets/paultimothymooney/blood-cells](https://www.kaggle.com/datasets/paultimothymooney/blood-cells)
- 410 original images plus ~12,500 augmented images, organized into `dataset2-master/images/TRAIN` and `TEST` folders by subtype. The original set is class-imbalanced; the augmented set is balanced by oversampling.
- Data is **not** committed. Download with:
  ```bash
  kaggle datasets download paultimothymooney/blood-cells
  unzip blood-cells.zip
  ```
  The notebook reads from `../input/dataset2-master/...`.

## Approach
1. **Inspect** — parse a BCCD XML annotation to draw cell bounding boxes; plot one example per subtype.
2. **Load augmented data** — walk `TRAIN`/`TEST` folders, map each subtype folder to a 5-class label and a 2-class (Mono/Poly) label, read with OpenCV and resize to **60×80×3**.
3. **Encode labels** — `to_categorical` for both the 5-class and 2-class targets.
4. **Preprocess** — normalize pixel values by `/255`; visualize RGB intensity histograms before/after.
5. **Model** — Keras `Sequential` CNN (Conv2D / MaxPool2D / BatchNormalization / Dropout / Dense) optimized with `RMSprop`, using `EarlyStopping`, `ReduceLROnPlateau`, and `ModelCheckpoint` callbacks; evaluation via `accuracy_score` and a confusion matrix.

## Results
The notebook reports **accuracy** and a **confusion matrix** on the held-out test split. No executed metric outputs are present in the run log (the notebook uses several deprecated APIs — `sklearn.cross_validation`, `scipy.misc.imresize`, `keras.utils.np_utils` — that may need updating before it runs).

| Metric | Value |
| --- | --- |
| Test accuracy | Run the notebook to reproduce |
| Confusion matrix | Run the notebook to reproduce |

Run the notebook to reproduce the metrics.

## Key Takeaways
- Using the augmented, class-balanced image set avoids the imbalance present in the 410 original images.
- The task is framed two ways: fine-grained 4/5-class subtype and coarse Mononuclear vs. Polynuclear.
- A compact CNN on small (60×80) images keeps training tractable.
- Code relies on legacy Keras/scikit-learn/scipy APIs and will need modernization to run on current versions.

## How to Run
```bash
pip install numpy pandas keras tensorflow scikit-learn opencv-python scipy matplotlib seaborn tqdm
jupyter notebook notebook.ipynb
```
Download the Blood Cells dataset (see Dataset) so paths resolve under `../input/dataset2-master/`.

## Credit
> Based on ["Identify Blood Cell Subtypes From Images"](https://www.kaggle.com/code/paultimothymooney/identify-blood-cell-subtypes-from-images) by Paul Mooney on Kaggle (834 votes).
> Dataset: [Blood Cells](https://www.kaggle.com/datasets/paultimothymooney/blood-cells). Adapted and documented for this portfolio.
