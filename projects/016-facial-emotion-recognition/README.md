# Facial Emotion Recognition (FER2013)

![CV](https://img.shields.io/badge/domain-Computer%20Vision-blue)
![Task](https://img.shields.io/badge/task-Classification-green)
![Framework](https://img.shields.io/badge/framework-Keras%2FTensorFlow-orange)

## Overview
This project recognizes facial emotions from 48x48 grayscale face images in the FER2013 dataset across 7 emotion classes. It uses VGG19 as a feature extractor with a lightweight classification head (transfer learning) to map faces to emotions.

## Dataset
- [Facial Expression Recognition (FER) Challenge](https://www.kaggle.com/datasets/ashishpatel26/facial-expression-recognitionferchallenge) (~287 MB), provided as `fer2013.csv`.
- 35,887 grayscale 48x48 face images; pixels stored as space-separated strings in the CSV.
- 7 classes. The label map used in the notebook is `{0: anger, 1: disgust, 2: fear, 3: happiness, 4: sadness, 5: surprise, 6: neutral}`.
- 90/10 stratified train/validation split (`random_state=42`).
- Not committed. Download with:
  ```bash
  kaggle datasets download ashishpatel26/facial-expression-recognitionferchallenge
  ```
  The notebook reads `../input/facial-expression-recognitionferchallenge/fer2013/fer2013/fer2013.csv`.

## Approach
1. **Preprocessing**: parse pixel strings into 48x48 arrays, convert grayscale → RGB (`cv2.cvtColor`), label-encode + one-hot encode emotions, normalize pixels to [0, 1].
2. **Augmentation** (`ImageDataGenerator`): rotation ±15°, width/height shift 0.15, shear 0.15, zoom 0.15, horizontal flip.
3. **Model** (transfer learning):
   - Base: `VGG19(weights='imagenet', include_top=False, input_shape=(48,48,3))`.
   - Head: take VGG19's second-to-last layer → `GlobalAveragePooling2D` → `Dense(7, softmax)`.
4. **Training**: Adam (`lr=0.0001`, β1=0.9, β2=0.999), `categorical_crossentropy`, batch size 32, 25 epochs, with `EarlyStopping` (monitor `val_accuracy`, patience 11, `restore_best_weights=True`) and `ReduceLROnPlateau` (factor 0.5, patience 7, `min_lr=1e-7`).
5. **Evaluation**: confusion matrix + `classification_report` on the validation set.

## Results
Extracted from the last Kaggle run log. Validation set = 3,589 images; 1,135 wrong predictions.

| Set | Metric | Value |
|-----|--------|-------|
| Validation | Accuracy | 0.68 |
| Validation | Macro avg F1 | 0.66 |
| Validation | Weighted avg F1 | 0.68 |

Per-class F1 (validation): happiness (class 3) 0.88 and surprise (class 5) 0.79 were strongest; fear/sadness classes were weakest (F1 ≈ 0.53–0.56).

## Key Takeaways
- VGG19 + a small pooling/softmax head reaches ~68% validation accuracy on FER2013, a notoriously noisy 7-class problem.
- The model is strong on "happiness" but weak on visually ambiguous emotions like sad vs. neutral — partly due to class imbalance and inherently hard/ambiguous labels.
- Possible improvements: class weighting/oversampling for minority emotions, fine-tuning deeper VGG19 layers, or face alignment/denoising preprocessing.

## How to Run
```bash
pip install numpy pandas opencv-python scikit-plot seaborn matplotlib scikit-learn tensorflow keras
jupyter notebook notebook.ipynb
```
Download the FER2013 dataset and make `fer2013.csv` available at the path the notebook reads under `../input/`.

## Credit
> Based on ["Facial Emotion Recognition | VGG19 - FER2013"](https://www.kaggle.com/code/enesztrk/facial-emotion-recognition-vgg19-fer2013) by Enes Ozturk on Kaggle (1118 votes).
> Dataset: [FER Challenge](https://www.kaggle.com/datasets/ashishpatel26/facial-expression-recognitionferchallenge). Adapted and documented for this portfolio.
