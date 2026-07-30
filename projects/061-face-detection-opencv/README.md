# Face Detection with OpenCV

![CV](https://img.shields.io/badge/domain-Computer%20Vision-blue) ![Task](https://img.shields.io/badge/task-Face%20Detection-orange)

## Overview
Classic, training-free face detection using OpenCV's Haar feature-based cascade
classifier. The pipeline detects faces both in a single group photo and across a
sampled batch of images from the Labeled Faces in the Wild (LFW) dataset, drawing
bounding boxes around each detected face.

## Dataset
- [LFW People](https://www.kaggle.com/datasets/atulanandjha/lfwpeople) (~232 MB) — 13,233 face photographs.
- A Haar cascade XML (`haarcascade_frontalface_default.xml`) plus a sample group
  photo are expected under an OpenCV `haarcascade-frontal-faces` input folder.
- Data is **not** committed here. Download with:
  ```bash
  kaggle datasets download atulanandjha/lfwpeople
  ```

## Approach
1. **Load cascade** — wrap `cv2.CascadeClassifier` in a `FaceDetector` class exposing a
   `detect(image, scaleFactor, minNeighbors, minSize)` method that returns face rectangles.
2. **Single image detection** — convert the national-team group photo to grayscale and
   run `detectMultiScale`, experimenting with `scaleFactor` (1.9 → 1.3) and `minNeighbors=3`.
3. **Batch fetching** — `FetchLFW` streams images out of `lfwfunneled.tgz`, randomly
   samples `dim*dim` images (e.g. a 10×10 = 100-image gallery), resizes by 0.4, and
   stacks them into a photo gallery.
4. **Batch detection** — run the detector on each grayscale image with
   `scaleFactor=1.1, minNeighbors=5, minSize=(30,30)`, draw one rectangle per image, and
   count total faces found.

## Results
No classification metrics are computed — this is a detector demo. The notebook prints
the count of scanned images and detected faces, e.g. `100 images have been scaned` /
`<N> faces have been detected`, and renders before/after galleries.

Run the notebook to reproduce the detection counts and visualizations.

## Key Takeaways
- Haar cascades give fast, zero-training face detection but are sensitive to
  `scaleFactor` / `minNeighbors`; loosening them trades recall for false positives.
- Works well on near-frontal, well-lit faces (LFW is funneled/aligned).
- The notebook also illustrates streaming images directly out of a `.tgz` archive and
  building a tiled photo gallery via NumPy `hstack`/`vstack` for compact visualization.
- Possible improvements: switch to a DNN/SSD or MTCNN detector for profile faces and
  cluttered scenes, and verify detections against LFW's ground-truth identities.

## How to Run
```bash
pip install numpy pandas matplotlib opencv-python
jupyter notebook notebook.ipynb
```
Place the LFW `lfwfunneled.tgz` and the Haar cascade XML where the notebook's
`../input/...` paths expect them (or edit the paths).

## Credit
> Based on ["Face Detection with OpenCV"](https://www.kaggle.com/code/serkanpeldek/face-detection-with-opencv) by Serkan Peldek on Kaggle (2090 votes).
> Dataset: [LFW People](https://www.kaggle.com/datasets/atulanandjha/lfwpeople). Adapted and documented for this portfolio.
