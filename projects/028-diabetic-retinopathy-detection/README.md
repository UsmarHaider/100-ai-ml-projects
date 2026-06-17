# Diabetic Retinopathy Detection — Eye Image Preprocessing (APTOS 2019)

![CV](https://img.shields.io/badge/domain-Computer%20Vision-blue)
![Task](https://img.shields.io/badge/task-Preprocessing%20%2F%20EDA-lightgrey)
![Framework](https://img.shields.io/badge/framework-OpenCV%20%2B%20Keras-orange)

## Overview
This project focuses on **image preprocessing and exploration** for diabetic retinopathy grading on the APTOS 2019 retina-fundus dataset. Rather than training a final classifier, it demonstrates techniques to make subtle retinal pathology far more visible: Ben Graham's lighting normalization, auto-cropping of uninformative black borders, and circular cropping — all of which boost downstream model quality.

## Dataset
- [APTOS 2019 Blindness Detection](https://www.kaggle.com/competitions/aptos2019-blindness-detection) (competition) — `train.csv`/`test.csv` with `id_code` and a 0–4 `diagnosis` severity level, plus fundus images in `train_images/` and `test_images/`.
- [Diabetic Retinopathy (resized)](https://www.kaggle.com/datasets/tanlikesmath/diabetic-retinopathy-resized) — past-competition (2015) data in `.jpeg`.
- [Retinopathy Train 2015 (PNG)](https://www.kaggle.com/datasets/donkeys/retinopathy-train-2015) — higher-quality `.png` version of the 2015 data.
- 5 severity classes (`NUM_CLASSES = 5`); images processed at `IMG_SIZE = 512`. The notebook does a 15% stratified train/validation split (`SEED = 77`) but uses it mainly for exploration.
- Not committed. Download with:
  ```bash
  kaggle competitions download -c aptos2019-blindness-detection
  kaggle datasets download tanlikesmath/diabetic-retinopathy-resized
  kaggle datasets download donkeys/retinopathy-train-2015
  ```
  The notebook reads these from `../input/...`.

## Approach
1. **Load + explore**: read `train.csv`/`test.csv`, shuffle, stratified 15% validation split; visualize 5 samples per severity level (original RGB and grayscale).
2. **Ben Graham preprocessing**: subtract a heavily Gaussian-blurred version of the image (`cv2.addWeighted(img, 4, GaussianBlur(img, sigmaX), -4, 128)`) to normalize lighting and amplify lesions (hemorrhages, exudates, cotton-wool spots).
3. **Auto-cropping**: `crop_image_from_gray` removes uninformative black borders using a grayscale tolerance mask, with a guard for images so dark they would crop to nothing.
4. **Color Ben + crop** (`load_ben_color`): read → RGB → crop → resize to 512x512 → Ben Graham weighting (`sigmaX` 10/30/50 compared).
5. **Circle crop** (`circle_crop`): mask a centered circle, crop, then apply Ben Graham preprocessing.
6. **Cross-dataset check**: apply the same pipeline to the 2015 `.jpeg` and `.png` datasets and compare image quality.
- Notebook imports `ResNet50` preprocessing utilities and metric helpers (`f1_score`, `fbeta_score`), but the committed notebook does not include a full training/evaluation loop.

## Results
This is a preprocessing/EDA notebook: it produces qualitative image comparisons rather than model metrics, and the run log is empty. No accuracy/F1/loss values are computed in the committed cells.

| Output | Type |
|--------|------|
| Original vs. grayscale vs. Ben Graham vs. cropped vs. circle-cropped images | Qualitative visual comparison |

Run the notebook to regenerate the preprocessing visualizations. (Classification metrics such as F1 would require adding the modeling step the imports anticipate.)

## Key Takeaways
- Ben Graham's blur-subtraction lighting correction dramatically improves visibility of diabetic-retinopathy lesions.
- Auto-cropping black borders preserves informative pixels when images are downscaled — important since lesions are tiny.
- Label noise is real: ophthalmologists disagree on severity, so the underlying task is inherently noisy.
- `.png` vs `.jpeg` made little difference here; lost detail was mostly intrinsic to the noisy source images.

## How to Run
```bash
pip install numpy pandas matplotlib scikit-image imgaug tqdm pillow opencv-python scikit-learn keras tensorflow
jupyter notebook notebook.ipynb
```
Download the APTOS 2019 competition data and the two supplementary 2015 datasets, and place them under `../input/` as referenced in the notebook.

## Credit
> Based on ["APTOS : Eye Preprocessing in Diabetic Retinopathy"](https://www.kaggle.com/code/ratthachat/aptos-eye-preprocessing-in-diabetic-retinopathy) by Neuron Engineer on Kaggle (2328 votes).
> Datasets: [APTOS 2019 Blindness Detection](https://www.kaggle.com/competitions/aptos2019-blindness-detection), [Diabetic Retinopathy (resized)](https://www.kaggle.com/datasets/tanlikesmath/diabetic-retinopathy-resized), [Retinopathy Train 2015 (PNG)](https://www.kaggle.com/datasets/donkeys/retinopathy-train-2015). Adapted and documented for this portfolio.
