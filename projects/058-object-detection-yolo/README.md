# Object Detection with YOLO v3

![CV](https://img.shields.io/badge/domain-Computer%20Vision-blue)
![Task](https://img.shields.io/badge/task-Object%20Detection-green)

## Overview
Implement the **YOLO v3** object detector from scratch in TensorFlow and run inference using
pretrained Darknet weights to detect and localize objects across the 80 COCO classes. Unlike
classifiers, YOLO predicts both class labels and bounding-box locations in a single forward pass.

## Dataset
- **Data for YOLO v3 kernel** — [kaggle.com/datasets/aruchomu/data-for-yolo-v3-kernel](https://www.kaggle.com/datasets/aruchomu/data-for-yolo-v3-kernel)
- Contains the pretrained **`yolov3.weights`** (Darknet), **`coco.names`** (80 class labels), and
  sample images for detection. This is an inference notebook — no training data is required.
- Data is **not** committed (~284 MB). Download with:
  ```bash
  kaggle datasets download aruchomu/data-for-yolo-v3-kernel
  ```
  so `coco.names`, `yolov3.weights`, and the sample images are reachable under `../input/`.

## Approach
1. **Configuration** — `_MODEL_SIZE = (416, 416)`; COCO anchors `[(10,13),(16,30),(33,23), ...]`
   (9 anchors, 3 per scale) computed via k-means on COCO.
2. **Building blocks** — `batch_norm`, fixed-padding conv (`conv2d_fixed_padding`), Leaky ReLU.
3. **Backbone** — **Darknet-53** feature extractor with residual blocks (last 3 classification
   layers omitted).
4. **Detection head** — `yolo_convolution_block`s feeding **3 detection layers** at 3 scales; each
   predicts `n_anchors * (5 + n_classes)` values; upsampling + concatenation with earlier feature
   maps for multi-scale detection.
5. **Post-processing** — sigmoid box decoding, then **non-max suppression** (`non_max_suppression`)
   with `max_output_size`, `iou_threshold`, and `confidence_threshold`.
6. **Weights** — `load_weights` parses the Darknet binary `yolov3.weights` into the TF graph;
   `load_class_names('coco.names')` provides the 80 labels. Detections drawn on the sample images.

## Results
This is an inference/visualization notebook (no training, no accuracy metric). It outputs detected
bounding boxes with class labels and confidence scores overlaid on the sample images. There are no
numeric metrics to report and no run log is available.

| Output | Description |
|--------|-------------|
| Bounding boxes + labels + scores | Drawn on the input images after NMS |

Run the notebook to reproduce the detection visualizations.

## Key Takeaways
- A full YOLO v3 graph (Darknet-53 + 3-scale detection head) can be reconstructed cleanly in
  TensorFlow's low-level API.
- Multi-scale detection with anchor priors is what lets YOLO handle objects of very different sizes.
- Correctly parsing the original Darknet weight binary into the TF layer order is the trickiest part
  of reproducing pretrained inference.
- The notebook uses the older `tf.layers` API (TensorFlow 1.x style); a TF2 port would need
  `tf.compat.v1` or rewriting in Keras layers.

## How to Run
```bash
pip install "tensorflow<2" numpy pillow seaborn
# place coco.names, yolov3.weights, and sample images under ../input/
jupyter notebook notebook.ipynb
```

## Credit
> Based on ["Yolo v3 Object Detection in Tensorflow"](https://www.kaggle.com/code/aruchomu/yolo-v3-object-detection-in-tensorflow) by heartkilla on Kaggle (6063 votes).
> Dataset: [Data for YOLO v3 kernel](https://www.kaggle.com/datasets/aruchomu/data-for-yolo-v3-kernel). Adapted and documented for this portfolio.
