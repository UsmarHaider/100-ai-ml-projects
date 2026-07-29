# Image Segmentation with U-Net (Severstal Steel Defect Detection)

![CV](https://img.shields.io/badge/domain-Computer%20Vision-blue)
![Task](https://img.shields.io/badge/task-Semantic%20Segmentation-green)

## Overview
Segment surface defects on steel sheets into 4 defect classes from the Severstal competition. The
approach is a **U-Net with a pretrained ResNet-18 encoder** (from `segmentation_models.pytorch`),
trained with BCE loss and evaluated with Dice and IoU. Segmentation localizes exactly where each
defect occurs, not just whether one exists.

## Dataset
- **Severstal: Steel Defect Detection** (competition) — [kaggle.com/competitions/severstal-steel-defect-detection](https://www.kaggle.com/competitions/severstal-steel-defect-detection)
- Images are **256x1600** grayscale-style steel photos; masks encoded as run-length (RLE) over
  **4 defect classes** (mask shape 256x1600x4 per image).
- Supporting datasets (model weights/scripts) are partly **included in `data/`**:
  - `data/model.py` — the single-file U-Net model script (from `rishabhiitbhu/unetmodelscript`).
  - `data/pretrainedmodels-0.7.4/` — the `pretrained-models.pytorch` dependency.
- ResNet encoder weights and the competition images are **not** committed:
  ```bash
  kaggle competitions download -c severstal-steel-defect-detection
  kaggle datasets download pytorch/resnet18      # encoder weights
  ```

## Approach
1. **RLE utilities** — `mask2rle` / `make_mask` convert between RLE strings and dense
   256x1600x4 masks.
2. **Dataset & augmentation** — `SteelDataset` with albumentations (`HorizontalFlip`,
   `ShiftScaleRotate`, `GaussNoise`, `Normalize`, `Resize`); stratified `train_test_split`
   (test_size=0.2, on `defects`, random_state=69).
3. **Model** — `Unet("resnet18", encoder_weights="imagenet", classes=4, activation=None)`.
4. **Trainer config**:
   - Loss: `BCEWithLogitsLoss`
   - Optimizer: Adam, `lr=5e-4`
   - Scheduler: `ReduceLROnPlateau(mode="min", patience=3)`
   - Batch size 4 (train/val), gradient accumulation `32 // 4 = 8` steps
   - `num_epochs = 20`, phases train/val
5. **Metrics** — per-batch **Dice** (positive/negative split) and **IoU**, logged per epoch; best
   model checkpointed on validation loss.

## Results
The kernel title indicates a public leaderboard Dice **> 0.88**. The trainer logs train/val loss,
Dice, and IoU per epoch, but the committed notebook has no stored outputs and no run log is available.

| Metric | Value |
|--------|-------|
| Public LB Dice | Kernel title states > 0.88 |
| Train/Val Dice, IoU, BCE loss | Run the notebook to reproduce the metrics |

Run the notebook to reproduce the metrics.

## Key Takeaways
- A ResNet-18-encoder U-Net is a strong, lightweight baseline for multi-class defect segmentation.
- Gradient accumulation lets an effective batch size of 32 fit despite a physical batch of 4 on
  large 256x1600 images.
- RLE encode/decode correctness is critical; the kernel notes a bugfix in Dice computation across
  versions.
- The author packaged `segmentation_models.pytorch` as a dataset because the competition disallowed
  internet — hence the committed `data/model.py`.

## How to Run
```bash
pip install torch torchvision numpy pandas opencv-python albumentations matplotlib tqdm
# data/model.py and data/pretrainedmodels-0.7.4 are included;
# download competition images + resnet18 weights as shown above
jupyter notebook notebook.ipynb
```

## Credit
> Based on ["UNet starter kernel (Pytorch) LB>0.88"](https://www.kaggle.com/code/rishabhiitbhu/unet-starter-kernel-pytorch-lb-0-88) by Rishabh Agrahari on Kaggle (1935 votes).
> Competition: [Severstal: Steel Defect Detection](https://www.kaggle.com/competitions/severstal-steel-defect-detection).
> Helper datasets: [unetmodelscript](https://www.kaggle.com/datasets/rishabhiitbhu/unetmodelscript), [pretrainedmodels](https://www.kaggle.com/datasets/rishabhiitbhu/pretrainedmodels). Adapted and documented for this portfolio.
