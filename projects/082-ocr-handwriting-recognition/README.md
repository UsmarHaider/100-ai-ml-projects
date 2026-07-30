# Offline Handwritten Text OCR

![CV](https://img.shields.io/badge/domain-Computer%20Vision-blue) ![Task](https://img.shields.io/badge/task-OCR%20%2F%20Classification-orange)

## Overview
Reads handwritten words from an image by combining a CNN character classifier with
OpenCV-based character segmentation. Each word image is split into individual characters,
each character is classified, and the predictions are stitched back into the final word.

## Dataset
- Character classifier training: [Handwritten Characters](https://www.kaggle.com/datasets/vaibhao/handwritten-characters)
  (A–Z + 0–9, **35 classes** after dropping symbols `# $ & @`).
- Word images for inference: [Handwriting Recognition](https://www.kaggle.com/datasets/landlord/handwriting-recognition).
- Data is **not** committed here. Download with:
  ```bash
  kaggle datasets download vaibhao/handwritten-characters
  kaggle datasets download landlord/handwriting-recognition
  ```

## Approach
1. **Load & preprocess** — read grayscale character crops, resize to **32×32**, cap at
   4,000 train / 1,000 val images per class, shuffle, normalize to `[0,1]`, reshape to
   `(-1, 32, 32, 1)`; labels one-hot encoded with `LabelBinarizer`.
2. **CNN classifier (Sequential)** — `Conv2D(32)` → `MaxPool` → `Conv2D(64)` →
   `MaxPool` → `Conv2D(128)` → `MaxPool` → `Dropout(0.25)` → `Flatten` →
   `Dense(128, relu)` → `Dropout(0.2)` → `Dense(35, softmax)`.
3. **Training** — Adam, `categorical_crossentropy`, accuracy metric, 50 epochs,
   batch size 32, with train/validation accuracy and loss plots.
4. **Segmentation + recognition** — threshold + dilate the word image, find external
   contours, sort them left-to-right (`sort_contours`), crop and classify each character,
   then `get_word` joins letters into the predicted word.

## Results
The notebook trains the 35-class character CNN with `accuracy` as the metric and plots
training vs. validation accuracy/loss curves, then prints the recognized word for several
sample images. No executed metric values are stored in the run log.

Run the notebook to reproduce the classifier accuracy curves and the per-word recognition
outputs.

## Key Takeaways
- Decomposing OCR into segmentation + per-character classification keeps the model simple
  (a small CNN) but ties accuracy to contour detection quality.
- Limitations noted in-kernel: fails when OpenCV can't isolate a character contour and on
  cursive/connected text; needs more training examples per character.
- Scaling to paragraphs would chain line → word → character segmentation before
  classification.

## How to Run
```bash
pip install tensorflow keras opencv-python imutils scikit-learn numpy pandas matplotlib seaborn
jupyter notebook notebook.ipynb
```
Point the `Train/`, `Validation/`, and word-image paths at the downloaded datasets.

## Credit
> Based on ["Offline Handwritten Text OCR"](https://www.kaggle.com/code/aman10kr/offline-handwritten-text-ocr) by Aman Kumar on Kaggle (1694 votes).
> Datasets: [Handwritten Characters](https://www.kaggle.com/datasets/vaibhao/handwritten-characters), [Handwriting Recognition](https://www.kaggle.com/datasets/landlord/handwriting-recognition). Adapted and documented for this portfolio.
