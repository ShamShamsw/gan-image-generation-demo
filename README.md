# Beginner Project 19: GAN Image Generation Demo

**Time:** 3-5 hours  
**Difficulty:** Intermediate Beginner  
**Focus:** adversarial training fundamentals, generator-discriminator dynamics, training stability monitoring, and generated image quality visualization

---

## Why This Project?

Generative Adversarial Networks (GANs) are a core concept for modern synthetic data and image generation systems.
This project gives you a practical beginner workflow where you can:

- train a lightweight GAN-style setup end-to-end with only NumPy,
- observe the adversarial game between generator and discriminator losses,
- track simple quality indicators as training progresses,
- visualize generated image samples and learning curves,
- and save reproducible training artifacts for analysis.

---

## More Projects

You can access this project and more in this separate repository:

[student-interview-prep](https://github.com/ShamShamsw/student-interview-prep.git)

---

## What You Will Build

You will build a command-line GAN image generation demo that:

1. Defines a synthetic grayscale image dataset with Gaussian-blob style real samples.
2. Implements a small generator network that maps latent noise vectors into 16x16 images.
3. Implements a discriminator network that predicts whether an image is real or generated.
4. Trains generator and discriminator in alternating steps with adversarial losses.
5. Tracks epoch-level metrics including discriminator loss, generator loss, and quality score.
6. Saves visual artifacts (loss curves, quality trend, and generated sample grid).
7. Persists a complete session summary to JSON for reproducibility.

---

## Requirements

- Python 3.11+
- `numpy`
- `matplotlib`
- `seaborn`

Install with:

```bash
pip install -r requirements.txt
```

---

## Example Session

```text
======================================================================
   GAN IMAGE GENERATION DEMO - NUMPY ADVERSARIAL TRAINING
======================================================================

Configuration:
   Image size: 16x16 (grayscale)
   Latent dimension: 24
   Hidden dimension: 64
   Epochs: 250
   Batch size: 64
   Generator learning rate: 0.025
   Discriminator learning rate: 0.015
   Snapshot interval: every 25 epochs
   Random seed: 42

Dataset profile:
   Name: Synthetic 16x16 grayscale blobs
   Description: Random Gaussian blobs plus low-intensity noise
   Pixel range: (0.0, 1.0)

Session behavior:
   1) Sample synthetic real-image batches each epoch.
   2) Train discriminator to separate real and generated images.
   3) Train generator to fool discriminator.
   4) Track GAN losses and a quality heuristic over epochs.
   5) Save generated sample grid and learning plots.

Session summary:
   Status: completed
   Training epochs: 250
   Final discriminator loss: 0.2387
   Final generator loss: 3.1464
   Mean quality (last 20 epochs): 0.197
   Max quality observed: 0.670
   Final D(real) mean: 0.833
   Final D(fake) mean: 0.050
   Final sample statistics:
      Mean pixel intensity: 0.471
      Pixel std deviation: 0.109
      Snapshots saved: 10
   Saved plots: data/runs/gan_losses.png, data/runs/quality_curve.png
   Saved sample grid: data/runs/generated_samples_grid.png
Saved run artifact: data/runs/latest_gan_demo.json
```

---

## Build Order

Follow this order for clean architecture:

1. `storage.py`
2. `models.py`
3. `operations.py`
4. `display.py`
5. `main.py`

---
