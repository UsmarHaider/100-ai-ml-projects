# Face Mask Detection (with Social Distancing)

![CV](https://img.shields.io/badge/domain-Computer%20Vision-blue)
![Task](https://img.shields.io/badge/task-Detection%20%2B%20Classification-green)
![Framework](https://img.shields.io/badge/framework-Keras%2FTensorFlow-orange)

## Overview
This project detects whether people in an image are wearing a face mask and flags social-distancing violations. Faces are located with a Haar cascade detector, each face crop is classified as `MASK` / `NO MASK` by a VGG19 transfer-learning model, and pairwise distances between detected faces are checked against a minimum threshold.

## Dataset
- [Face Mask 12K Images Dataset](https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset) — `Train`/`Validation`/`Test` folders of `WithMask` / `WithoutMask` images, resized to 128x128.
- [Face Mask Detection](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection) — sample images for face-detection demos.
- [Haar Cascades](https://www.kaggle.com/datasets/lalitharajesh/haarcascades) — included in `data/` (e.g. `haarcascade_frontalface_default.xml`, ~9 MB).
- The two image datasets are **not** committed. Download with:
  ```bash
  kaggle datasets download ashishjangra27/face-mask-12k-images-dataset
  kaggle datasets download andrewmvd/face-mask-detection
  ```
  The notebook reads them from `../input/face-mask-12k-images-dataset/...` and `../input/face-mask-detection/images/...`. The Haar cascade XML files are available in this folder's `data/`.

## Approach
1. **Face detection**: `cv2.CascadeClassifier` with `haarcascade_frontalface_default.xml`; `detectMultiScale` returns face bounding boxes.
2. **Social distancing**: compute Euclidean distance between every pair of detected faces; if below `MIN_DISTANCE=130`, mark both boxes as violations (red).
3. **Mask classifier (transfer learning)**:
   - Base: `VGG19(weights='imagenet', include_top=False, input_shape=(128,128,3))` with all base layers frozen.
   - Head: `Flatten` → `Dense(2, sigmoid)`.
   - Compiled with Adam, `categorical_crossentropy`, accuracy.
4. **Augmentation**: train `ImageDataGenerator` with rescale 1/255, horizontal flip, zoom 0.2, shear 0.2; target size 128x128, batch size 32.
5. **Training**: `fit_generator` for 20 epochs; model saved as `masknet.h5`.
6. **Integration**: for each detected face, crop → resize 128x128 → classify → overlay `MASK`/`NO MASK` label and a green/red box.

## Results
The committed notebook has no saved cell outputs and the run log is empty.

| Set | Metric | Value (author's stated) |
|-----|--------|-------------------------|
| Test | Accuracy | ~98% (per notebook narrative) |

The "98%" is the author's qualitative statement in the markdown, not a captured `model.evaluate` output. Run the notebook to reproduce the metric (accuracy is computed via `model.evaluate_generator` on the test set).

## Key Takeaways
- Combining a classic Haar cascade face detector with a deep mask classifier yields a practical two-stage pipeline.
- Freezing VGG19 and training only a small head makes the classifier fast to train and data-efficient.
- Haar cascade detection struggles on masked/angled faces, which can limit end-to-end performance.
- Possible improvements: a learned face detector (e.g. MTCNN/SSD), fine-tuning deeper VGG19 layers, and a calibrated real-world distance metric.

## How to Run
```bash
pip install numpy pandas opencv-python scipy matplotlib keras tensorflow
jupyter notebook notebook.ipynb
```
The Haar cascade XML files are provided in `data/`. Download the two image datasets above and place them under `../input/` as referenced in the notebook.

## Credit
> Based on ["Mask and social distancing detection using VGG19"](https://www.kaggle.com/code/nageshsingh/mask-and-social-distancing-detection-using-vgg19) by Nagesh Singh Chauhan on Kaggle (1117 votes).
> Datasets: [Face Mask 12K Images](https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset), [Face Mask Detection](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection), [Haar Cascades](https://www.kaggle.com/datasets/lalitharajesh/haarcascades). Adapted and documented for this portfolio.
