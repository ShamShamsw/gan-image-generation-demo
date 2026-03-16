"""main.py – entry point for the GAN image generation demo."""

import numpy as np

from display import (
    plot_losses,
    plot_quality,
    plot_sample_grid,
    print_epoch,
    print_header,
    print_summary,
)
from models import Discriminator, Generator
from operations import (
    compute_quality,
    generate_real_samples,
    train_discriminator_step,
    train_generator_step,
)
from storage import save_json

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIG = {
    "image_size": 16,
    "latent_dim": 24,
    "hidden_dim": 64,
    "epochs": 250,
    "batch_size": 64,
    "lr_gen": 0.025,
    "lr_disc": 0.015,
    "snapshot_interval": 25,
    "seed": 42,
}

# Output paths
RUNS_DIR = "data/runs"
PATH_LOSSES = f"{RUNS_DIR}/gan_losses.png"
PATH_QUALITY = f"{RUNS_DIR}/quality_curve.png"
PATH_GRID = f"{RUNS_DIR}/generated_samples_grid.png"
PATH_JSON = f"{RUNS_DIR}/latest_gan_demo.json"

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    cfg = CONFIG
    print_header(cfg)

    rng = np.random.RandomState(cfg["seed"])

    image_size: int = cfg["image_size"]
    n_pixels: int = image_size * image_size
    latent_dim: int = cfg["latent_dim"]
    hidden_dim: int = cfg["hidden_dim"]
    epochs: int = cfg["epochs"]
    batch_size: int = cfg["batch_size"]
    lr_gen: float = cfg["lr_gen"]
    lr_disc: float = cfg["lr_disc"]
    snapshot_interval: int = cfg["snapshot_interval"]

    # Build models
    gen = Generator(latent_dim, hidden_dim, n_pixels, rng)
    disc = Discriminator(n_pixels, hidden_dim, rng)

    # Metric accumulators
    d_losses: list[float] = []
    g_losses: list[float] = []
    quality_scores: list[float] = []
    snapshots_saved: int = 0

    last_d_real_mean: float = 0.0
    last_d_fake_mean: float = 0.0

    # Fixed noise for quality evaluation / snapshots (reproducible)
    z_eval = rng.randn(batch_size, latent_dim)

    for epoch in range(1, epochs + 1):
        # Sample real images and latent vectors
        real_imgs = generate_real_samples(batch_size, image_size, rng)
        z = rng.randn(batch_size, latent_dim)

        # Generate fake images for D training
        fake_imgs = gen.forward(z)

        # --- Train discriminator ---
        d_loss, d_real_mean, d_fake_mean = train_discriminator_step(
            disc, real_imgs, fake_imgs, lr=lr_disc
        )

        # --- Train generator ---
        z_g = rng.randn(batch_size, latent_dim)
        g_loss = train_generator_step(gen, disc, z_g, lr=lr_gen)

        # --- Quality metric (evaluated on fixed noise) ---
        quality = compute_quality(disc, gen, z_eval)

        d_losses.append(d_loss)
        g_losses.append(g_loss)
        quality_scores.append(quality)
        last_d_real_mean = d_real_mean
        last_d_fake_mean = d_fake_mean

        # Print progress every snapshot_interval epochs
        if epoch % snapshot_interval == 0 or epoch == epochs:
            print_epoch(epoch, epochs, d_loss, g_loss, quality)
            snapshots_saved += 1

    # ---------------------------------------------------------------------------
    # Produce final generated samples
    # ---------------------------------------------------------------------------
    z_final = rng.randn(32, latent_dim)
    final_samples = gen.forward(z_final)

    sample_mean = float(final_samples.mean())
    sample_std = float(final_samples.std())

    # ---------------------------------------------------------------------------
    # Save plots
    # ---------------------------------------------------------------------------
    plot_losses(d_losses, g_losses, PATH_LOSSES)
    plot_quality(quality_scores, PATH_QUALITY)
    plot_sample_grid(final_samples, image_size, PATH_GRID, grid_cols=8)

    # ---------------------------------------------------------------------------
    # Build and persist session summary
    # ---------------------------------------------------------------------------
    mean_quality_last20 = float(np.mean(quality_scores[-20:]))
    max_quality = float(np.max(quality_scores))

    results = {
        "status": "completed",
        "epochs": epochs,
        "final_d_loss": round(d_losses[-1], 4),
        "final_g_loss": round(g_losses[-1], 4),
        "mean_quality_last20": round(mean_quality_last20, 3),
        "max_quality": round(max_quality, 3),
        "final_d_real_mean": round(last_d_real_mean, 3),
        "final_d_fake_mean": round(last_d_fake_mean, 3),
        "sample_mean": round(sample_mean, 3),
        "sample_std": round(sample_std, 3),
        "snapshots_saved": snapshots_saved,
        "plot_losses": PATH_LOSSES,
        "plot_quality": PATH_QUALITY,
        "plot_grid": PATH_GRID,
        "config": cfg,
    }

    save_json(results, PATH_JSON)
    print_summary(results)
    print(f"Saved run artifact: {PATH_JSON}")


if __name__ == "__main__":
    main()
