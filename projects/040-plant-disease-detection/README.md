# Plant Leaf Disease Detection (Plant Pathology 2020)

![CV](https://img.shields.io/badge/domain-Computer%20Vision-blue)
![Task](https://img.shields.io/badge/task-Multi--class%20Classification-green)

## Overview
Diagnose apple-leaf conditions from photographs into four categories — healthy, scab, rust, and
multiple diseases. Early disease detection helps protect agricultural yield. The notebook combines
EDA and OpenCV image processing with transfer learning from pretrained ImageNet CNNs (DenseNet and
EfficientNet), then ensembles them.

## Dataset
- **Plant Pathology 2020 - FGVC7** (competition) — [kaggle.com/competitions/plant-pathology-2020-fgvc7](https://www.kaggle.com/competitions/plant-pathology-2020-fgvc7)
- 4 target columns: `healthy`, `scab`, `rust`, `multiple_diseases`; images provided as JPGs with
  `train.csv` / `test.csv` / `sample_submission.csv`.
- Data is **not** committed. Accept the competition rules and download:
  ```bash
  kaggle competitions download -c plant-pathology-2020-fgvc7
  ```
  so that `images/`, `train.csv`, and `test.csv` are reachable under `../input/plant-pathology-2020-fgvc7/`.

## Approach
1. **EDA** — visualize sample leaves; analyze per-channel (R/G/B) intensity distributions
   (insight: the blue channel tends to track disease).
2. **Image processing demos** — Canny edge detection, flipping, convolution, blurring with OpenCV.
3. **Pipeline** — TPU-oriented `tf.data` input: decode JPEG, resize to **512x512**, augment;
   labels split with `train_test_split(test_size=0.15, random_state=2020)`.
4. **Models** (transfer learning, `GlobalAveragePooling2D` + softmax head, 4 classes):
   - **DenseNet121** (ImageNet weights)
   - **EfficientNet** (B7 family via `efficientnet.tfkeras`)
   - **EfficientNet NoisyStudent** weights
5. **Training** — `EPOCHS = 20`, custom learning-rate schedule (`build_lrfn`: warmup to `lr_max`
   then exponential decay) via `LearningRateScheduler`.
6. **Ensembling** — average predictions across the trained models for the final submission.

## Results
The notebook is structured for leaderboard submission; competition scoring is mean column-wise
**ROC AUC**. The committed notebook has no executed metric outputs and no run log is available.

| Metric | Value |
|--------|-------|
| Validation / LB ROC AUC | Run the notebook to reproduce the metric |

Run the notebook to reproduce the metrics.

## Key Takeaways
- Channel-distribution EDA motivates the modeling: color cues (notably blue) separate healthy vs.
  diseased leaves.
- Transfer learning from DenseNet/EfficientNet at high resolution (512x512) outperforms training
  from scratch on a small labeled set.
- Ensembling complementary backbones plus a warmup-decay LR schedule is the main accuracy lever.
- Requires a TPU/`strategy.scope()` and the `efficientnet` package for full reproduction.

## How to Run
```bash
pip install tensorflow keras efficientnet numpy scipy pandas opencv-python seaborn matplotlib plotly scikit-learn tqdm
# download the competition data under ../input/plant-pathology-2020-fgvc7/
jupyter notebook notebook.ipynb
```

## Credit
> Based on ["Plant Pathology 2020 : EDA + Models"](https://www.kaggle.com/code/tarunpaparaju/plant-pathology-2020-eda-models) by Tarun Paparaju on Kaggle (2207 votes).
> Dataset/competition: [Plant Pathology 2020 - FGVC7](https://www.kaggle.com/competitions/plant-pathology-2020-fgvc7). Adapted and documented for this portfolio.
