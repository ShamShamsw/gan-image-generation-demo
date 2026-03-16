"""display.py – printing and plotting helpers for the GAN demo."""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from storage import ensure_directory


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

_SEP = "=" * 70


def print_header(config: dict) -> None:
    """Print the session banner and configuration table."""
    print(_SEP)
    print("   GAN IMAGE GENERATION DEMO - NUMPY ADVERSARIAL TRAINING")
    print(_SEP)
    print()
    print("Configuration:")
    print(f"   Image size: {config['image_size']}x{config['image_size']} (grayscale)")
    print(f"   Latent dimension: {config['latent_dim']}")
    print(f"   Hidden dimension: {config['hidden_dim']}")
    print(f"   Epochs: {config['epochs']}")
    print(f"   Batch size: {config['batch_size']}")
    print(f"   Generator learning rate: {config['lr_gen']}")
    print(f"   Discriminator learning rate: {config['lr_disc']}")
    print(f"   Snapshot interval: every {config['snapshot_interval']} epochs")
    print(f"   Random seed: {config['seed']}")
    print()
    print("Dataset profile:")
    print(f"   Name: Synthetic {config['image_size']}x{config['image_size']} grayscale blobs")
    print("   Description: Random Gaussian blobs plus low-intensity noise")
    print("   Pixel range: (0.0, 1.0)")
    print()
    print("Session behavior:")
    print("   1) Sample synthetic real-image batches each epoch.")
    print("   2) Train discriminator to separate real and generated images.")
    print("   3) Train generator to fool discriminator.")
    print("   4) Track GAN losses and a quality heuristic over epochs.")
    print("   5) Save generated sample grid and learning plots.")
    print()


def print_epoch(epoch: int, epochs: int, d_loss: float, g_loss: float,
                quality: float) -> None:
    """Print a one-line epoch summary."""
    print(
        f"   Epoch {epoch:>3}/{epochs}  "
        f"D-loss: {d_loss:.4f}  "
        f"G-loss: {g_loss:.4f}  "
        f"Quality: {quality:.3f}"
    )


def print_summary(results: dict) -> None:
    """Print the final session summary."""
    r = results
    print()
    print("Session summary:")
    print(f"   Status: {r['status']}")
    print(f"   Training epochs: {r['epochs']}")
    print(f"   Final discriminator loss: {r['final_d_loss']:.4f}")
    print(f"   Final generator loss: {r['final_g_loss']:.4f}")
    print(f"   Mean quality (last 20 epochs): {r['mean_quality_last20']:.3f}")
    print(f"   Max quality observed: {r['max_quality']:.3f}")
    print(f"   Final D(real) mean: {r['final_d_real_mean']:.3f}")
    print(f"   Final D(fake) mean: {r['final_d_fake_mean']:.3f}")
    print("   Final sample statistics:")
    print(f"      Mean pixel intensity: {r['sample_mean']:.3f}")
    print(f"      Pixel std deviation: {r['sample_std']:.3f}")
    print(f"      Snapshots saved: {r['snapshots_saved']}")
    print(f"   Saved plots: {r['plot_losses']}, {r['plot_quality']}")
    print(f"   Saved sample grid: {r['plot_grid']}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_losses(d_losses: list[float], g_losses: list[float],
                filepath: str) -> None:
    """Save a figure with discriminator and generator loss curves."""
    ensure_directory(os.path.dirname(filepath))
    sns.set_theme(style="darkgrid")
    fig, ax = plt.subplots(figsize=(8, 4))
    epochs = range(1, len(d_losses) + 1)
    ax.plot(epochs, d_losses, label="Discriminator loss", linewidth=1.5)
    ax.plot(epochs, g_losses, label="Generator loss", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("GAN Training Losses")
    ax.legend()
    fig.tight_layout()
    fig.savefig(filepath, dpi=100)
    plt.close(fig)


def plot_quality(quality_scores: list[float], filepath: str) -> None:
    """Save a figure with the quality score trend over epochs."""
    ensure_directory(os.path.dirname(filepath))
    sns.set_theme(style="darkgrid")
    fig, ax = plt.subplots(figsize=(8, 4))
    epochs = range(1, len(quality_scores) + 1)
    ax.plot(epochs, quality_scores, color="mediumseagreen", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Quality (D(fake) mean)")
    ax.set_title("Generator Quality Score Over Training")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(filepath, dpi=100)
    plt.close(fig)


def plot_sample_grid(images: np.ndarray, image_size: int,
                     filepath: str, grid_cols: int = 8) -> None:
    """Save a grid of generated sample images.

    Parameters
    ----------
    images:
        Array of shape (n_samples, image_size * image_size).
    image_size:
        Side length of each square image (pixels).
    filepath:
        Destination path for the PNG file.
    grid_cols:
        Number of columns in the grid.
    """
    ensure_directory(os.path.dirname(filepath))
    n = images.shape[0]
    grid_rows = (n + grid_cols - 1) // grid_cols

    fig, axes = plt.subplots(grid_rows, grid_cols,
                              figsize=(grid_cols * 1.5, grid_rows * 1.5))
    axes = np.array(axes).reshape(grid_rows, grid_cols)

    for idx in range(grid_rows * grid_cols):
        row, col = divmod(idx, grid_cols)
        ax = axes[row, col]
        if idx < n:
            ax.imshow(images[idx].reshape(image_size, image_size),
                      cmap="gray", vmin=0, vmax=1)
        ax.axis("off")

    fig.suptitle("Generated Image Samples", fontsize=12)
    fig.tight_layout()
    fig.savefig(filepath, dpi=100)
    plt.close(fig)
