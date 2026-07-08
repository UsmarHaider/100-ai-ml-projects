# Malaria Cell Image Detection

![CV](https://img.shields.io/badge/domain-Computer%20Vision-blue)
![Task](https://img.shields.io/badge/task-Binary%20Classification-green)

## Overview
Detect malaria infection from microscopy images of red blood cells by classifying each cell image as
**Parasitized** or **Uninfected**. Automating this screening can speed diagnosis where microscopists
are scarce. The approach is transfer learning with a pretrained ImageNet CNN.

## Dataset
- **Cell Images for Detecting Malaria** — [kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria)
- **2 classes**: Parasitized, Uninfected (~27,558 cell images, roughly balanced).
- Images loaded via `flow_from_directory`, resized to **224x224x3**, rescaled.
- Data is **not** committed. Download with:
  ```bash
  kaggle datasets download iarunava/cell-images-for-detecting-malaria
  ```
  so the `cell_images/cell_images/<Parasitized|Uninfected>/` folders are reachable.

## Approach
1. **Data generators** — `ImageDataGenerator` with horizontal/vertical flip and other augmentations;
   `flow_from_directory` for train and validation batches (`class_mode='categorical'`).
2. **Model** — transfer learning via a `get_model()` factory supporting Xception, ResNet50,
   InceptionV3, InceptionResNetV2, DenseNet201, NASNetMobile/Large. The notebook instantiates
   **NASNetMobile** with `input_shape=(224, 224, 3)`, `num_class=2`, plus a custom classification head.
3. **Training** — Adam (lr=0.0001), categorical cross-entropy, accuracy metric, **10 epochs**, with
   adjusted **class weights** and callbacks (ModelCheckpoint, EarlyStopping, ReduceLROnPlateau).
4. **Evaluation** — accuracy plus `precision`, `recall`, `f1_score`, confusion matrix,
   `classification_report`, and `roc_auc_score` are imported for reporting.

## Results
The notebook computes accuracy, precision, recall, F1, ROC-AUC and a confusion matrix. The committed
notebook has no executed metric outputs and no run log is available. The original kernel title claims
**~95% accuracy**.

| Metric | Value |
|--------|-------|
| Accuracy | Kernel title states ~95%; run to reproduce |
| Precision / Recall / F1 | Computed in notebook |
| ROC-AUC | Computed in notebook |

Run the notebook to reproduce the metrics.

## Key Takeaways
- Transfer learning (NASNetMobile) with a lightweight head fits a binary cell-classification task well.
- Class weighting plus EarlyStopping/ReduceLROnPlateau stabilize training within only 10 epochs.
- Beyond accuracy, the notebook tracks precision/recall/F1/ROC-AUC, which matter more for a medical
  screening task where false negatives are costly.

## How to Run
```bash
pip install tensorflow keras numpy pandas scikit-learn matplotlib seaborn opencv-python pillow mlxtend tqdm
# place cell_images/cell_images/<class>/ under ../input/
jupyter notebook notebook.ipynb
```

## Credit
> Based on ["Malaria Detection using Keras (Accuracy 95%)"](https://www.kaggle.com/code/anjanatiha/malaria-detection-using-keras-accuracy-95) by Anjana Tiha on Kaggle (121 votes).
> Dataset: [Cell Images for Detecting Malaria](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria). Adapted and documented for this portfolio.
