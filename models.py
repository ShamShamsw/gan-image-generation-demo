"""models.py – NumPy-only Generator and Discriminator for the GAN demo."""

import numpy as np


# ---------------------------------------------------------------------------
# Activation helpers
# ---------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _sigmoid_grad(s: np.ndarray) -> np.ndarray:
    """Gradient of sigmoid given *s = sigmoid(x)*."""
    return s * (1.0 - s)


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(float)


def _leaky_relu(x: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    return np.where(x > 0, x, alpha * x)


def _leaky_relu_grad(x: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    return np.where(x > 0, 1.0, alpha)


# ---------------------------------------------------------------------------
# Generator  (latent_dim → hidden_dim → image_size)
# ---------------------------------------------------------------------------

class Generator:
    """Two-layer MLP that maps a latent noise vector to a flat image.

    Architecture: Linear → ReLU → Linear → Sigmoid
    """

    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int,
                 rng: np.random.RandomState) -> None:
        scale = 0.05
        self.W1 = rng.randn(latent_dim, hidden_dim) * scale
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.randn(hidden_dim, output_dim) * scale
        self.b2 = np.zeros(output_dim)

        # Cache populated during forward pass (needed for backward)
        self._z: np.ndarray | None = None
        self._h1_pre: np.ndarray | None = None
        self._h1: np.ndarray | None = None
        self._out: np.ndarray | None = None

    def forward(self, z: np.ndarray) -> np.ndarray:
        """Return generated images for latent vectors *z* (batch, latent_dim)."""
        self._z = z
        self._h1_pre = z @ self.W1 + self.b1
        self._h1 = _relu(self._h1_pre)
        self._out = _sigmoid(self._h1 @ self.W2 + self.b2)
        return self._out

    def backward(self, d_out: np.ndarray, lr: float) -> None:
        """Update weights given upstream gradient *d_out* (batch, output_dim)."""
        batch = self._z.shape[0]

        d_out_pre = d_out * _sigmoid_grad(self._out)
        dW2 = self._h1.T @ d_out_pre / batch
        db2 = d_out_pre.mean(axis=0)

        d_h1 = d_out_pre @ self.W2.T
        d_h1_pre = d_h1 * _relu_grad(self._h1_pre)
        dW1 = self._z.T @ d_h1_pre / batch
        db1 = d_h1_pre.mean(axis=0)

        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1


# ---------------------------------------------------------------------------
# Discriminator  (image_size → hidden_dim → 1)
# ---------------------------------------------------------------------------

class Discriminator:
    """Two-layer MLP that outputs a real/fake probability for a flat image.

    Architecture: Linear → LeakyReLU → Linear → Sigmoid
    """

    def __init__(self, input_dim: int, hidden_dim: int,
                 rng: np.random.RandomState) -> None:
        scale = 0.05
        self.W1 = rng.randn(input_dim, hidden_dim) * scale
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.randn(hidden_dim, 1) * scale
        self.b2 = np.zeros(1)

        # Cache
        self._x: np.ndarray | None = None
        self._h1_pre: np.ndarray | None = None
        self._h1: np.ndarray | None = None
        self._out: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Return discriminator probabilities for images *x* (batch, input_dim)."""
        self._x = x
        self._h1_pre = x @ self.W1 + self.b1
        self._h1 = _leaky_relu(self._h1_pre)
        self._out = _sigmoid(self._h1 @ self.W2 + self.b2)
        return self._out

    def backward(self, d_loss_d_out: np.ndarray, lr: float,
                 update_weights: bool = True) -> np.ndarray:
        """Backpropagate *d_loss_d_out* through the discriminator.

        Parameters
        ----------
        d_loss_d_out:
            Gradient of the scalar loss w.r.t. the sigmoid output,
            shape (batch, 1).
        lr:
            Learning rate used when *update_weights* is True.
        update_weights:
            When False, only return the input gradient (used during G training).

        Returns
        -------
        Gradient of the loss w.r.t. the discriminator input x,
        shape (batch, input_dim).
        """
        batch = self._x.shape[0]

        d_out_pre = d_loss_d_out * _sigmoid_grad(self._out)
        dW2 = self._h1.T @ d_out_pre / batch
        db2 = d_out_pre.mean(axis=0)

        d_h1 = d_out_pre @ self.W2.T
        d_h1_pre = d_h1 * _leaky_relu_grad(self._h1_pre)
        dW1 = self._x.T @ d_h1_pre / batch
        db1 = d_h1_pre.mean(axis=0)

        d_x = d_h1_pre @ self.W1.T

        if update_weights:
            self.W2 -= lr * dW2
            self.b2 -= lr * db2
            self.W1 -= lr * dW1
            self.b1 -= lr * db1

        return d_x
