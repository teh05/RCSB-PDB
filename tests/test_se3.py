from __future__ import annotations

import numpy as np

from src.geometry.se3 import random_rotation, rotate, translate


def test_random_rotation_is_so3():
    rng = np.random.default_rng(0)
    for _ in range(8):
        r = random_rotation(rng)
        assert r.shape == (3, 3)
        assert np.allclose(r @ r.T, np.eye(3), atol=1e-6)
        assert np.allclose(np.linalg.det(r), 1.0, atol=1e-6)


def test_rotate_then_translate():
    rng = np.random.default_rng(1)
    coords = rng.normal(size=(5, 3))
    r = random_rotation(rng)
    t = rng.normal(size=3)
    out = translate(rotate(coords, r), t)
    expected = coords @ r.T + t
    assert np.allclose(out, expected)
