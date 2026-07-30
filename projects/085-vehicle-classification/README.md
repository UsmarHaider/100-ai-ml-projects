# Vehicle Type Classification — Stanford Cars (fastai ResNet34)

![CV](https://img.shields.io/badge/domain-Computer%20Vision-blue) ![Task](https://img.shields.io/badge/task-Image%20Classification-orange) ![fastai](https://img.shields.io/badge/framework-fastai-9cf)

## Overview
Fine-grained vehicle classification across **196 car categories** using fastai's default
image-classification workflow on a ResNet34 backbone. Tests how a few lines of transfer
learning perform on a hard, visually similar multi-class problem.

## Dataset
- [Stanford Cars Dataset (by classes folder)](https://www.kaggle.com/datasets/jutrera/stanford-car-dataset-by-classes-folder)
  — **16,185 images across 196 classes** of cars.
- Data is **not** committed here. Download with:
  ```bash
  kaggle datasets download jutrera/stanford-car-dataset-by-classes-folder
  ```

## Approach
1. **Data bunch** — `ImageDataBunch.from_folder` on the `car_data/train` directory with a
   20% validation split (`valid_pct=0.2`).
2. **Transforms** — `get_transforms(do_flip=False, flip_vert=False, max_rotate=0,
   max_lighting=0.3)`, `size=224`, batch size 64, normalized with ImageNet stats.
3. **Model** — `create_cnn(data, models.resnet34, metrics=accuracy)` (ImageNet-pretrained
   ResNet34).
4. **Training** — `fit_one_cycle(35)` (one-cycle learning-rate schedule, 35 epochs).
5. **Analysis** — `ClassificationInterpretation` plots the 196×196 confusion matrix.

## Results
| Metric | Value |
| --- | --- |
| Validation accuracy | **~65%** (author-reported in notebook markdown) |

The author notes 65% is well above random chance for 196 fine-grained, visually similar
categories given how little code is involved. No run log is available here.

Run the notebook to reproduce the accuracy and confusion matrix.

## Key Takeaways
- A pretrained ResNet34 with fastai's one-cycle policy reaches ~65% top-1 across 196
  near-identical car classes with minimal code.
- Flips/rotations are disabled because car orientation/text is discriminative.
- The confusion matrix is the key diagnostic here: most errors are between visually
  similar makes/models/years rather than across unrelated vehicle types.
- The author notes this dataset is harder than Caltech-256 or Simpsons Characters with
  the same recipe, reflecting the fine-grained nature of the task.
- Improvements: a deeper backbone (ResNet50+), progressive resizing, and longer
  fine-tuning with discriminative learning rates.

## How to Run
```bash
pip install fastai torch torchvision numpy pandas matplotlib
jupyter notebook notebook.ipynb
```
Note: this notebook targets the **fastai v1** API; point `img_dir` at the downloaded
`car_data/car_data/train` folder.

## Credit
> Based on ["Stanford Cars Dataset with Fastai V1"](https://www.kaggle.com/code/paultimothymooney/stanford-cars-dataset-with-fastai-v1) by Paul Mooney on Kaggle (113 votes).
> Dataset: [Stanford Cars Dataset (by classes folder)](https://www.kaggle.com/datasets/jutrera/stanford-car-dataset-by-classes-folder). Adapted and documented for this portfolio.
