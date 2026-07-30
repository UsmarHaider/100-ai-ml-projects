# Image Generation with GANs (Generative Dog Images)

![CV](https://img.shields.io/badge/domain-Computer%20Vision-blue) ![Task](https://img.shields.io/badge/task-Generative%20%2F%20GAN-purple) ![PyTorch](https://img.shields.io/badge/framework-PyTorch-red)

## Overview
An introduction to Generative Adversarial Networks (GANs) that trains a DCGAN-style
generator/discriminator pair to synthesize 64×64 dog images for Kaggle's *Generative Dog
Images* competition. Covers GAN theory, data augmentation, and a full PyTorch training
loop.

## Dataset
- [Generative Dog Images competition](https://www.kaggle.com/competitions/generative-dog-images)
  — the Stanford Dogs dataset (120 breeds), read from an `all-dogs/` image folder.
- Data is **not** committed here. Accept the competition rules and download from the
  competition page above (`kaggle competitions download -c generative-dog-images`).

## Approach
1. **Preprocessing / augmentation** — `Resize(64)` → `CenterCrop(64)` →
   `RandomHorizontalFlip(0.5)` → random `ColorJitter`/`RandomRotation(20°)` (p=0.2) →
   `ToTensor` → `Normalize(0.5, 0.5)`. Loaded via `ImageFolder` + `DataLoader`
   (batch size 32).
2. **Weight init** — custom `weights_init` (Conv ~ N(0, 0.02), BatchNorm ~ N(1, 0.02)).
3. **Generator** — stacked `ConvTranspose2d` blocks (noise `nz=128` → 1024 → 512 → 256 →
   128 → 64 → 3) with BatchNorm + ReLU and a final `Tanh`.
4. **Discriminator** — strided `Conv2d` blocks (3 → 32 → 64 → 128 → 256 → 1) with
   LeakyReLU(0.2), BatchNorm on deeper layers, sigmoid output.
5. **Training** — `BCELoss`, two Adam optimizers (`LR_G=0.0005`, `LR_D=0.0003`,
   `beta1=0.5`), label smoothing (`real_label=0.95`), up to 200 epochs; tracks
   generator/discriminator loss and saves fixed-noise samples per epoch.

## Results
This is a generative task — there is no accuracy metric. The competition is scored by
**MiFID** (Memorization-informed Fréchet Inception Distance). The notebook tracks and
plots generator vs. discriminator BCE loss per epoch and saves sample grids; the
baseline training cell ships with `EPOCH = 0` so it must be raised to actually train.

Run the notebook (set epochs > 0) to reproduce loss curves and generated samples.

## Key Takeaways
- A standard DCGAN with light augmentation and label smoothing is a solid baseline for
  64×64 image synthesis.
- Asymmetric learning rates (`G > D`) and a near-1 real label help stabilize adversarial
  training.
- The notebook walks through GAN theory (generator vs. discriminator, the adversarial
  game, backprop on each network) before the implementation, making it a useful primer.
- Dataset caveats noted in-kernel: some images contain multiple dogs, people, or text,
  which adds noise to the generative target.
- Improvements noted in-kernel: RaLSGAN loss and longer training for better MiFID.

## How to Run
```bash
pip install torch torchvision numpy matplotlib tqdm
jupyter notebook notebook.ipynb
```
Download the competition data into an `all-dogs/` folder and set `EPOCH`/`epochs > 0`.

## Credit
> Based on ["GAN Introduction"](https://www.kaggle.com/code/jesucristo/gan-introduction) by Nanashi on Kaggle (1950 votes).
> Competition: [Generative Dog Images](https://www.kaggle.com/competitions/generative-dog-images). Adapted and documented for this portfolio.
