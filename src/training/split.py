"""Train/val index split (NumPy only — no torch/jax)."""

from __future__ import annotations

import numpy as np


def split_indices(
    n: int,
    *,
    val_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if n < 1:
        raise ValueError("n must be >= 1")
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(n)
    if n < 5 or val_fraction <= 0:
        return perm, np.zeros((0,), dtype=np.int64)
    n_val = int(round(n * float(val_fraction)))
    n_val = min(max(n_val, 1), n - 1)
    val = np.sort(perm[:n_val])
    train = np.sort(perm[n_val:])
    return train, val


def batched_slices(n: int, batch_size: int) -> list[tuple[int, int]]:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    return [(i, min(i + batch_size, n)) for i in range(0, n, batch_size)]
