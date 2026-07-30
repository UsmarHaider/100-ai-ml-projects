# Image Denoising with Autoencoders (PyTorch)

![CV](https://img.shields.io/badge/domain-Computer%20Vision-blue) ![Task](https://img.shields.io/badge/task-Denoising-purple) ![PyTorch](https://img.shields.io/badge/framework-PyTorch-red)

## Overview
Removes noise from face images using a convolutional autoencoder, benchmarked against the
classical BM3D filter. The notebook studies three noise types (Gaussian, masking
/Bernoulli, Poisson) and measures restoration quality with PSNR and MSE.

## Dataset
- [FFHQ Face Data Set](https://www.kaggle.com/datasets/greatgamedota/ffhq-face-data-set) —
  high-quality face images loaded via `torchvision.datasets.ImageFolder`.
- Data is **not** committed here. Download with:
  ```bash
  kaggle datasets download greatgamedota/ffhq-face-data-set
  ```

## Approach
1. **Preprocess** — `Resize(255)` → `ToTensor`; split into train/val `DataLoader`s
   (batch size 16, shuffled).
2. **Noise models** — additive Gaussian (`torch.normal`), masking/Bernoulli, and Poisson
   noise applied on the fly, with `PSNR` / `MSE` helper functions for evaluation.
3. **Classical baseline** — BM3D (Block Matching and 3D filtering) for comparison.
4. **Autoencoder** — encoder of `Conv2d` blocks (3 → 32 → 64 → 128) and decoder of
   `ConvTranspose2d` blocks (128 → 128 → 64 → 32 → 3), trained to reconstruct the clean
   image from the noisy input.
5. **Training** — Adam (`lr` 1e-3, also 5e-3 / 1e-2 variants), reconstruction loss with
   both **MSELoss** and **L1Loss** variants, up to 30 epochs; PSNR/MSE tracked on
   train and validation each evaluation.

## Results
Author-reported values (run locally and recorded in the notebook markdown):

| Stage | PSNR (dB) |
| --- | --- |
| Noisy input — Gaussian (baseline) | ~15.0 |
| Noisy input — masking/Bernoulli (baseline) | ~7.7–13.6 |
| Noisy input — Poisson (baseline) | ~11–14 |
| Autoencoder, **MSE loss** (mean ± std) | **31.13 ± 1.19** |
| Autoencoder, **L1/MAE loss** (mean ± std) | **31.07 ± 1.28** |

The autoencoder roughly **doubles** the PSNR versus the noisy input, decisively beating
the noisy baseline. (These numbers are copied from the author's local run; re-running may
vary.)

## Key Takeaways
- A simple Conv/ConvTranspose autoencoder lifts PSNR from ~15 dB to ~31 dB on Gaussian
  noise — a large quality gain.
- MSE and L1 reconstruction losses give nearly identical PSNR here (31.1 vs 31.1).
- BM3D works well for Gaussian noise but is weaker on masking/Poisson noise, motivating
  the learned approach.

## How to Run
```bash
pip install torch torchvision numpy matplotlib bm3d torchsummary
jupyter notebook notebook.ipynb
```
Point `data_dir` at the downloaded FFHQ image folder. Training is GPU-intensive.

## Credit
> Based on ["Image Denoising using AutoEncoder (PyTorch🔥)"](https://www.kaggle.com/code/alaaeddinebenzekri/image-denoising-using-autoencoder-pytorch) by Alaa Eddine Ben Zekri on Kaggle (166 votes).
> Dataset: [FFHQ Face Data Set](https://www.kaggle.com/datasets/greatgamedota/ffhq-face-data-set). Adapted and documented for this portfolio.
