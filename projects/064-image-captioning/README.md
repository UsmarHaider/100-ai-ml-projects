# Image Captioning with CNN + LSTM (Flickr8k)

![CV](https://img.shields.io/badge/domain-Computer%20Vision-blue) ![NLP](https://img.shields.io/badge/domain-NLP-green) ![Task](https://img.shields.io/badge/task-Image%20Captioning-orange)

## Overview
Generates natural-language captions for images by combining computer vision and
sequence modeling. A pretrained CNN encodes each image into a feature vector, and an
LSTM decoder generates the caption word-by-word — the classic encoder–decoder ("Show
and Tell") framework trained on Flickr8k.

## Dataset
- [Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k) — 8,091 images, each
  with 5 human captions (`captions.txt`).
- Alternate larger source: [Flickr Image Dataset](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset).
- Data is **not** committed here. Download with:
  ```bash
  kaggle datasets download adityajn105/flickr8k
  ```

## Approach
1. **Caption preprocessing** — lowercase, strip special characters/extra spaces, drop
   single characters, wrap each caption with `startseq … endseq`.
2. **Tokenization** — Keras `Tokenizer` builds the vocabulary (`vocab_size`); sequences
   are padded to `max_length`. Images split 85% / 15% into train / test by unique image.
3. **Image feature extraction** — **DenseNet201** (ImageNet) with the final layer
   removed; the GlobalAveragePooling output gives a **1920-dim** embedding per image,
   precomputed and cached for all images.
4. **Caption model** —
   - Image branch: `Dense(256, relu)` → reshape to `(1, 256)`.
   - Text branch: `Embedding(vocab_size, 256)`.
   - Concatenate → `LSTM(256)` → `Dropout(0.5)` → add image features →
     `Dense(128, relu)` → `Dropout(0.5)` → `Dense(vocab_size, softmax)`.
5. **Training** — Adam, `categorical_crossentropy`, batch size 64, up to 50 epochs with
   a custom data generator, `ModelCheckpoint`, `EarlyStopping` (patience 5), and
   `ReduceLROnPlateau` (factor 0.2, patience 3).
6. **Inference** — greedy decode word-by-word until `endseq` and display sample images
   with their generated captions.

## Results
The notebook trains with `categorical_crossentropy` loss and plots train vs. validation
loss curves; the author notes the model overfits on this small dataset. No BLEU score is
computed and no quantitative metric is reported in the markdown.

Run the notebook to reproduce the loss curves; add a BLEU evaluation for a numeric score.

## Key Takeaways
- DenseNet201 features + a lightweight LSTM decoder are enough to produce relevant
  captions on Flickr8k.
- Adding the image features back into the LSTM output (residual-style) slightly improves
  over the original 2014 architecture.
- The model overfits with 8k images and shows repetitive phrasing; improvements: train on
  Flickr30k/40k, add an attention mechanism, and evaluate with BLEU.

## How to Run
```bash
pip install numpy pandas tensorflow matplotlib seaborn tqdm
jupyter notebook notebook.ipynb
```
Place the Flickr8k `Images/` folder and `captions.txt` where the notebook expects them
(edit the `../input/flickr8k` paths if needed).

## Credit
> Based on ["Flickr8K Image Captioning using CNNs+LSTMs"](https://www.kaggle.com/code/quadeer15sh/flickr8k-image-captioning-using-cnns-lstms) by Quadeer Shaikh on Kaggle (2198 votes).
> Dataset: [Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k). Adapted and documented for this portfolio.
