"""Train/val/test index split (NumPy only — no torch/jax)."""

from __future__ import annotations

import numpy as np


def split_indices(
    n: int,
    *,
    val_fraction: float,
    seed: int,
    test_fraction: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n < 1:
        raise ValueError("n must be >= 1")
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(n)
    empty = np.zeros((0,), dtype=np.int64)
    if n < 5:
        return perm, empty, empty
    vf = float(val_fraction)
    tf = float(test_fraction)
    if vf <= 0 and tf <= 0:
        return perm, empty, empty
    n_test = int(round(n * tf)) if tf > 0 else 0
    n_val = int(round(n * vf)) if vf > 0 else 0
    n_test = max(n_test, 0)
    n_val = max(n_val, 0)
    while n_test + n_val >= n:
        if n_test > 0:
            n_test -= 1
        elif n_val > 0:
            n_val -= 1
        else:
            break
    test = np.sort(perm[:n_test])
    val = np.sort(perm[n_test : n_test + n_val])
    train = np.sort(perm[n_test + n_val :])
    return train, val, test


def batched_slices(n: int, batch_size: int) -> list[tuple[int, int]]:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    return [(i, min(i + batch_size, n)) for i in range(0, n, batch_size)]
