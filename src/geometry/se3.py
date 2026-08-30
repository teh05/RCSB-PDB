"""SE(3) rigid transforms for equivariance checks.

Coordinates are stored as row vectors with shape (N, 3). A rotation matrix R
that acts on column vectors (R @ x) is applied as ``coords @ R.T``.
"""

from __future__ import annotations

import numpy as np


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Sample a matrix in SO(3) via QR with positive determinant."""
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    q = np.asarray(q, dtype=np.float64)
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def rotate(coords: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    return np.asarray(coords, dtype=np.float64) @ np.asarray(rotation, dtype=np.float64).T


def translate(coords: np.ndarray, shift: np.ndarray) -> np.ndarray:
    t = np.asarray(shift, dtype=np.float64).reshape(1, 3)
    return np.asarray(coords, dtype=np.float64) + t
