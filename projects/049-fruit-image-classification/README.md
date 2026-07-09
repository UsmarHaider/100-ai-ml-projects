# Fruit Image Classification (PCA + Classical ML)

![CV](https://img.shields.io/badge/domain-Computer%20Vision-blue)
![Task](https://img.shields.io/badge/task-Classification-green)

## Overview
Classify fruit images from the Fruits-360 dataset using **classical machine learning** rather than
deep nets: raw pixels are flattened, scaled, optionally reduced with PCA, and fed to SVM, K-NN, and
Decision Tree classifiers for both binary and multi-class tasks. It is a clean study of how feature
dimensionality and model choice affect accuracy.

## Dataset
- **Fruits 360** — [kaggle.com/datasets/moltean/fruits](https://www.kaggle.com/datasets/moltean/fruits)
- Pre-split into `Training/` and `Test/` folders (~75% / 25%), one folder per fruit class.
- Each image is read with OpenCV, resized to **100x100x3**, flattened to a feature vector, and
  standardized with `StandardScaler` (mean subtraction).
- Data is **not** committed. Download with:
  ```bash
  kaggle datasets download moltean/fruits
  ```
  so that `fruits-360/Training/<class>/` and `fruits-360/Test/<class>/` are reachable under `../input/`.

## Approach
1. **Load & vectorize** — read selected classes, resize to 100x100, flatten to image-feature vectors.
2. **Scale** — `StandardScaler` on the flattened pixels.
3. **Dimensionality reduction / visualization** — `PCA` to 2D and 3D for plotting; PCA reconstructions
   at n=2/10/50 components; explained-variance analysis.
4. **Classifiers** (binary, then multi-class):
   - **Linear SVM** — `SVC(kernel='linear', gamma='auto', probability=True)`
   - **K-Nearest Neighbors** — `KNeighborsClassifier`
   - **Decision Tree** — `DecisionTreeClassifier`
5. **Validation** — `cross_val_score(cv=5)` for K-fold accuracy; variants trained on full features
   and on 2D-PCA features to visualize decision boundaries.
6. **Evaluation** — `accuracy_score`, normalized confusion matrices, and ROC curve / `roc_auc_score`
   for the binary SVM.

## Results
The notebook prints accuracy per model (e.g. `"Accuracy with SVM: {:.2f}%"`) and K-fold mean +/- std,
plus confusion matrices and ROC/AUC for the binary case. The committed notebook has no stored metric
outputs and no run log is available.

| Metric | Value |
|--------|-------|
| SVM / K-NN / Decision Tree accuracy | Run the notebook to reproduce the metric |
| SVM 5-fold CV accuracy (mean +/- std) | Run the notebook to reproduce the metric |
| Binary SVM ROC-AUC | Run the notebook to reproduce the metric |

Run the notebook to reproduce the metrics.

## Key Takeaways
- On the clean, well-centered Fruits-360 images, classical classifiers on flattened pixels are
  surprisingly competitive without any CNN.
- PCA trades accuracy for speed; 2D PCA is mainly useful for visualizing class separability and
  decision boundaries.
- A CNN is the natural next step for harder, less curated fruit imagery (the author notes this).

## How to Run
```bash
pip install scikit-learn numpy pandas matplotlib seaborn opencv-python
# place fruits-360/Training and fruits-360/Test under ../input/
jupyter notebook notebook.ipynb
```

## Credit
> Based on ["Fruit Classification: PCA, SVM, KNN, Decision Tree"](https://www.kaggle.com/code/waltermaffy/fruit-classification-pca-svm-knn-decision-tree) by Walter Maffione on Kaggle (1198 votes).
> Dataset: [Fruits 360](https://www.kaggle.com/datasets/moltean/fruits). Adapted and documented for this portfolio.
