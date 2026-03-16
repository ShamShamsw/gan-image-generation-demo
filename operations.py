"""operations.py – dataset generation and GAN training steps."""

import numpy as np

from models import Discriminator, Generator

_EPS = 1e-8


# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------

def generate_real_samples(batch_size: int, image_size: int,
                           rng: np.random.RandomState) -> np.ndarray:
    """Return a batch of synthetic 16×16 greyscale images.

    Each image contains 1–3 random Gaussian blobs plus low-intensity noise.
    Pixel values are clipped to [0, 1].
    """
    n_pixels = image_size * image_size
    xs, ys = np.meshgrid(np.arange(image_size), np.arange(image_size),
                          indexing="ij")
    images = np.empty((batch_size, n_pixels), dtype=float)

    for i in range(batch_size):
        img = np.zeros((image_size, image_size))
        n_blobs = rng.randint(1, 4)
        for _ in range(n_blobs):
            cx = rng.uniform(2.0, image_size - 2.0)
            cy = rng.uniform(2.0, image_size - 2.0)
            sigma = rng.uniform(1.5, 3.5)
            amplitude = rng.uniform(0.5, 1.0)
            img += amplitude * np.exp(
                -((xs - cx) ** 2 + (ys - cy) ** 2) / (2.0 * sigma ** 2)
            )
        img += rng.uniform(0.0, 0.05, size=(image_size, image_size))
        images[i] = np.clip(img, 0.0, 1.0).ravel()

    return images


# ---------------------------------------------------------------------------
# Training steps
# ---------------------------------------------------------------------------

def train_discriminator_step(
    disc: Discriminator,
    real_imgs: np.ndarray,
    fake_imgs: np.ndarray,
    lr: float,
) -> tuple[float, float, float]:
    """One discriminator update step.

    Trains D to output 1 for real images and 0 for generated images.

    Returns
    -------
    (d_loss, d_real_mean, d_fake_mean)
    """
    # --- Real images: target = 1 ---
    d_real = disc.forward(real_imgs)
    loss_real = -np.mean(np.log(d_real + _EPS))
    # Per-sample gradients of -log(D(x)); backward() will average over batch
    d_loss_real = -(1.0 / (d_real + _EPS))
    disc.backward(d_loss_real, lr=lr, update_weights=True)

    # --- Fake images: target = 0 ---
    d_fake = disc.forward(fake_imgs)
    loss_fake = -np.mean(np.log(1.0 - d_fake + _EPS))
    # Per-sample gradients of -log(1 - D(x)); backward() will average over batch
    d_loss_fake = (1.0 / (1.0 - d_fake + _EPS))
    disc.backward(d_loss_fake, lr=lr, update_weights=True)

    d_loss = (loss_real + loss_fake) / 2.0
    return d_loss, float(d_real.mean()), float(d_fake.mean())


def train_generator_step(
    gen: Generator,
    disc: Discriminator,
    z: np.ndarray,
    lr: float,
) -> float:
    """One generator update step.

    Trains G to produce images that D classifies as real (target = 1).

    Returns
    -------
    Generator loss scalar.
    """
    fake_imgs = gen.forward(z)
    d_fake = disc.forward(fake_imgs)

    g_loss = -np.mean(np.log(d_fake + _EPS))

    # Gradient of loss w.r.t. D output (per-sample); backward() will average
    d_loss_d_out = -(1.0 / (d_fake + _EPS))

    # Propagate through D (weights frozen) to get gradient w.r.t. fake images
    d_x = disc.backward(d_loss_d_out, lr=lr, update_weights=False)

    # Propagate through G and update its weights
    gen.backward(d_x, lr=lr)

    return float(g_loss)


# ---------------------------------------------------------------------------
# Quality metric
# ---------------------------------------------------------------------------

def compute_quality(disc: Discriminator, gen: Generator,
                    z: np.ndarray) -> float:
    """Return the mean discriminator score for a batch of generated images.

    A higher score means G is better at fooling D.
    """
    fake_imgs = gen.forward(z)
    d_fake = disc.forward(fake_imgs)
    return float(d_fake.mean())
