# Brain Tumor MRI Classification

![CV](https://img.shields.io/badge/domain-Computer%20Vision-blue)
![Task](https://img.shields.io/badge/task-Binary%20Classification-green)
![Framework](https://img.shields.io/badge/framework-Keras%2FTensorFlow-orange)

## Overview
This project classifies brain MRI scans as tumor (`YES`) vs. no tumor (`NO`). It combines a computer-vision preprocessing step that auto-crops the brain region from each scan with a VGG-16 transfer-learning classifier, addressing a small-data binary medical-imaging problem.

## Dataset
- [Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection) — MRI scans in `yes`/`no` folders.
- [Keras Pretrained Models](https://www.kaggle.com/datasets/gaborfodor/keras-pretrained-models) (~1 GB) — supplies the VGG-16 ImageNet weights (`vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5`).
- The notebook splits images into TRAIN / VAL / TEST folders (5 images per class held out for TEST; ~80% train, remainder validation).
- Not committed. Download with:
  ```bash
  kaggle datasets download navoneel/brain-mri-images-for-brain-tumor-detection
  kaggle datasets download gaborfodor/keras-pretrained-models
  ```
  The notebook reads from `../input/brain-mri-images-for-brain-tumor-detection/...` and `../input/keras-pretrained-models/...`.

## Approach
1. **Brain cropping (OpenCV)**: grayscale + Gaussian blur → threshold → erode/dilate → find largest contour → locate extreme points → crop the brain rectangle out of the scan.
2. **Preprocessing**: resize to 224x224 and apply VGG-16 `preprocess_input`.
3. **Augmentation** (`ImageDataGenerator`): rotation ±15°, width/height shift 0.1, shear 0.1, brightness [0.5, 1.5], horizontal + vertical flip.
4. **Model** (transfer learning):
   - Base: `VGG16(include_top=False, input_shape=(224,224,3))` with ImageNet weights, frozen.
   - Head: `Flatten` → `Dropout(0.5)` → `Dense(1, sigmoid)`.
   - Compiled with `RMSprop(lr=1e-4)`, `binary_crossentropy`, accuracy.
5. **Training**: `fit_generator` for up to 30 epochs (steps/epoch 50, validation steps 25) with `EarlyStopping` (monitor `val_acc`, patience 6).
6. **Evaluation**: accuracy + confusion matrix on validation and test sets; misclassified test images are displayed.

## Results
The committed notebook has no saved cell outputs and the run log is empty. The author reports the following in the notebook markdown:

| Set | Metric | Value (author's stated) |
|-----|--------|-------------------------|
| Validation | Accuracy | ~88% |
| Test | Accuracy | ~80% |

These are the author's stated results, not captured cell outputs. The notebook computes accuracy via `accuracy_score` and confusion matrices for both sets. Run the notebook to reproduce the metrics.

## Key Takeaways
- Auto-cropping the brain region (contour + extreme points) normalizes wildly varying MRI framing before classification — a key CV preprocessing trick.
- VGG-16 transfer learning with a tiny sigmoid head works on a very small dataset thanks to augmentation.
- The gap between validation (~88%) and the tiny test set (~80%) reflects how limited the data is; more images would stabilize the estimate.
- Possible improvements: more training images, fine-tuning VGG-16, and a larger held-out test set.

## How to Run
```bash
pip install numpy tqdm opencv-python imutils matplotlib plotly scikit-learn keras tensorflow
jupyter notebook notebook.ipynb
```
Download both datasets above and place them under `../input/` as referenced in the notebook.

## Credit
> Based on ["Brain Tumor Detection v1.0 || CNN, VGG-16"](https://www.kaggle.com/code/ruslankl/brain-tumor-detection-v1-0-cnn-vgg-16) by Ruslan Klymentiev on Kaggle (4334 votes).
> Datasets: [Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection), [Keras Pretrained Models](https://www.kaggle.com/datasets/gaborfodor/keras-pretrained-models). Adapted and documented for this portfolio.
