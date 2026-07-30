# Food Image Classification — Food-101 (InceptionV3)

![CV](https://img.shields.io/badge/domain-Computer%20Vision-blue) ![Task](https://img.shields.io/badge/task-Image%20Classification-orange) ![Transfer Learning](https://img.shields.io/badge/method-Transfer%20Learning-brightgreen)

## Overview
Multiclass food classification on the Food-101 dataset using transfer learning with a
pretrained InceptionV3. The workflow starts with a 3-class proof of concept, then scales
the same fine-tuning recipe up to 11 classes.

## Dataset
- [Food-101](https://www.kaggle.com/datasets/dansbecker/food-101) — **101,000 images
  across 101 food categories** (1,000 images per class), max side length 512px. Includes
  `meta/train.txt`, `meta/test.txt`, `meta/classes.txt`.
- Data is **not** committed here. Download with:
  ```bash
  kaggle datasets download dansbecker/food-101
  ```

## Approach
1. **Prepare splits** — use `train.txt` / `test.txt` to copy images into `train/` and
   `test/` folders, then carve out `train_mini` / `test_mini` subsets (first **3 classes**:
   samosa, pizza, omelette) for fast experimentation, later scaling to **11 classes**.
2. **Augmentation** — `ImageDataGenerator` with `rescale=1/255` (+ shear/zoom/flip),
   `flow_from_directory`, `target_size=(299, 299)`, `class_mode='categorical'`,
   batch size 16.
3. **Model** — `InceptionV3(weights='imagenet', include_top=False)` →
   `GlobalAveragePooling2D` → `Dense(128, relu)` → `Dropout(0.2)` →
   `Dense(n_classes, softmax)` with L2 regularization (0.005).
4. **Training** — `SGD(lr=0.0001, momentum=0.9)`, `categorical_crossentropy`, accuracy
   metric, 30 epochs, `ModelCheckpoint` (best-only) + `CSVLogger`.
5. **Inference** — load the best checkpoint and predict softmax classes for new
   downloaded images.

## Results
| Model | Accuracy |
| --- | --- |
| InceptionV3, 3-class subset | **>95%** (author-reported in notebook markdown) |

The notebook plots accuracy/loss curves for both the 3-class and 11-class models and
shows example predictions (e.g. a pizza+apple-pie image scored pizza 75.4% / apple pie
18%). No run log is available here, so only the author's reported figure is quoted.

Run the notebook to reproduce the accuracy/loss curves for the 3- and 11-class models.

## Key Takeaways
- Fine-tuning InceptionV3 with a small GAP→Dense head reaches >95% on an easy 3-class
  subset; accuracy drops as classes are added (11-class).
- A low SGD learning rate (1e-4) + momentum and L2 regularization stabilize fine-tuning
  of the large backbone.
- Starting on a 3-class subset is a practical way to iterate on architecture before
  committing compute to the full 101 classes.

## How to Run
```bash
pip install tensorflow keras numpy pandas matplotlib pillow
jupyter notebook notebook.ipynb
```
Point the data paths at the downloaded `food-101/` folder (images + `meta/` files).

## Credit
> Based on ["Multiclass Food Classification using TensorFlow"](https://www.kaggle.com/code/theimgclist/multiclass-food-classification-using-tensorflow) by Avinash on Kaggle (885 votes).
> Dataset: [Food-101](https://www.kaggle.com/datasets/dansbecker/food-101). Adapted and documented for this portfolio.
