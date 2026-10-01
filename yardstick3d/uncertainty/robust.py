from __future__ import annotations

import numpy as np

from yardstick3d.types import FloatArray

KERNELS = ("l2", "huber", "cauchy", "tukey", "geman_mcclure")


def robust_weight(r_norm: FloatArray, kernel: str = "huber", delta: float = 1.0) -> FloatArray:
    """IRLS weight w(u) for standardized residual u = ||r|| / δ."""
    u = np.asarray(r_norm, dtype=np.float64)
    k = kernel.lower()
    d = max(float(delta), 1e-12)
    x = np.abs(u) / d
    if k == "l2":
        return np.ones_like(x)
    if k == "huber":
        w = np.ones_like(x)
        mask = x > 1.0
        w[mask] = 1.0 / np.maximum(x[mask], 1e-12)
        return w
    if k == "cauchy":
        return 1.0 / (1.0 + x**2)
    if k == "tukey":
        w = np.zeros_like(x)
        mask = x < 1.0
        w[mask] = (1.0 - x[mask] ** 2) ** 2
        return w
    if k == "geman_mcclure":
        return 1.0 / (1.0 + x**2) ** 2
    raise ValueError(f"Unknown kernel {kernel}")
