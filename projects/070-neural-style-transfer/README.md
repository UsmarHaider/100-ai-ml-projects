# Neural Style Transfer (VGG19)

![CV](https://img.shields.io/badge/domain-Computer%20Vision-blue) ![Task](https://img.shields.io/badge/task-Style%20Transfer-purple)

## Overview
Implements Gatys-style neural style transfer: recombine the *content* of one photograph
with the *artistic style* of a painting by optimizing a generated image against feature
representations extracted from a pretrained VGG19 network.

## Dataset
- Style images: [Best Artworks of All Time](https://www.kaggle.com/datasets/ikarus777/best-artworks-of-all-time) (e.g. a Picasso painting).
- Content images: [Image Classification dataset](https://www.kaggle.com/datasets/duttadebadri/image-classification) (a travel/adventure photo).
- Pretrained weights: [VGG19](https://www.kaggle.com/datasets/crawford/vgg19) (~625 MB, `vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5`).
- Data is **not** committed here. Download with:
  ```bash
  kaggle datasets download ikarus777/best-artworks-of-all-time
  kaggle datasets download duttadebadri/image-classification
  kaggle datasets download crawford/vgg19
  ```

## Approach
1. **Preprocess** — load content + style images, resize to 400px height, mean-center via
   `vgg19.preprocess_input`, and create a trainable `combination_image` placeholder.
2. **Feature model** — build VGG19 (`include_top=False`, ImageNet weights) over the
   concatenated `[content, style, combination]` tensor; tap intermediate layers.
3. **Losses** —
   - *Content loss*: MSE between `block5_conv2` features of content vs. combination.
   - *Style loss*: MSE between **Gram matrices** of `block1_conv1 … block5_conv1` (equal
     weight per layer).
   - Total loss = `content_weight * content + style_weight * style`
     (`content_weight=0.025`, `style_weight=1.0`).
4. **Optimization** — compute gradients of the loss w.r.t. the combination image with
   Keras backend and minimize using **SciPy L-BFGS-B** (`fmin_l_bfgs_b`) over several
   iterations, then `deprocess_image` for display.

## Results
There is no accuracy metric for style transfer; quality is the loss value of the
optimized image. The recorded run log shows the L-BFGS-B objective decreasing across
iterations, ending at a final function value of **F ≈ 4.94 × 10²¹** (the optimizer hit
its per-iteration evaluation limit). Lower loss and more iterations yield a more
stylized image.

Run the notebook to reproduce the loss trajectory and the stylized output image.

## Key Takeaways
- Gram-matrix style loss across multiple VGG19 layers captures texture/color; a single
  deep content layer preserves structure.
- The content/style weight ratio (0.025 vs 1.0) controls how strongly the painting's
  style dominates.
- L-BFGS-B converges faster per iteration than plain SGD for this optimization-as-image
  problem, but each step is expensive.

## How to Run
```bash
pip install tensorflow keras numpy scipy matplotlib pandas
jupyter notebook notebook.ipynb
```
Provide the VGG19 `.h5` weights and content/style images at the paths the notebook
expects (edit the `../input/...` paths if needed).

## Credit
> Based on ["Style Transfer Deep Learning Algorithm"](https://www.kaggle.com/code/basu369victor/style-transfer-deep-learning-algorithm) by Victor Basu on Kaggle (933 votes).
> Datasets: [Best Artworks of All Time](https://www.kaggle.com/datasets/ikarus777/best-artworks-of-all-time), [Image Classification](https://www.kaggle.com/datasets/duttadebadri/image-classification), [VGG19](https://www.kaggle.com/datasets/crawford/vgg19). Adapted and documented for this portfolio.
