# Dog Breed Identification

![CV](https://img.shields.io/badge/domain-Computer%20Vision-blue)
![Task](https://img.shields.io/badge/task-Multi--class%20Classification-green)

## Overview
Identify the breed of a dog from a photograph across **120 breeds** using transfer learning from a
pretrained ResNet-50. Fine-grained breed classification is challenging because many breeds look
similar; a strong ImageNet backbone provides the visual features.

## Dataset
- **Stanford Dogs Dataset** — [kaggle.com/datasets/jessicali9530/stanford-dogs-dataset](https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset)
- **120 breed classes**, ~20,580 images organized one folder per breed.
- Loaded with `tf.keras.utils.image_dataset_from_directory` (80/20 train/validation split, seed 123),
  resized to a fixed `img_size`, batch size 32, with augmentation.
- Data is **not** committed. Download with:
  ```bash
  kaggle datasets download jessicali9530/stanford-dogs-dataset
  ```
  so the per-breed image folders are reachable.

## Approach
1. **Data pipeline** — `image_dataset_from_directory` for train/val, on-the-fly augmentation,
   batch size 32.
2. **Model** — transfer learning: a TensorFlow Hub **ResNet-V2-50** feature layer
   (`hub.KerasLayer`) followed by `Dense(120, softmax)`.
3. **Compile** — Adam optimizer, accuracy metric (categorical cross-entropy objective).
4. **Training** — `model.fit(epochs=30)` with three callbacks: **EarlyStopping**,
   **ReduceLROnPlateau**, and **ModelCheckpoint** (save best on validation loss) to curb overfitting.
5. **Evaluation** — `model.evaluate(val)` reports loss and accuracy; accuracy/loss curves plotted.

## Results
The notebook prints final accuracy via `model.evaluate` (`f"Accuracy is: {round(accuracy*100,2)}%"`)
and plots train/validation accuracy and loss. The committed notebook has no stored metric outputs and
no run log is available.

| Metric | Value |
|--------|-------|
| Validation accuracy | Run the notebook to reproduce the metric |
| Validation loss | Run the notebook to reproduce the metric |

Run the notebook to reproduce the metrics.

## Key Takeaways
- A frozen ResNet-V2-50 feature extractor plus a single softmax head handles 120-way fine-grained
  classification with minimal trainable parameters.
- The trio of EarlyStopping + ReduceLROnPlateau + ModelCheckpoint is the main defense against
  overfitting on a moderate-size dataset.
- Fine-tuning the top backbone blocks and adding test-time augmentation are natural accuracy levers.

## How to Run
```bash
pip install tensorflow tensorflow-hub numpy pandas matplotlib
# place the Stanford Dogs breed folders under the input directory
jupyter notebook notebook.ipynb
```

## Credit
> Based on ["Predicting Dog Species Using Resnet50"](https://www.kaggle.com/code/siddhantojha17/predicting-dog-species-using-resnet50) by SiddhantOjha on Kaggle (121 votes).
> Dataset: [Stanford Dogs Dataset](https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset). Adapted and documented for this portfolio.
