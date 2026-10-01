from __future__ import annotations

import numpy as np

from yardstick3d.types import FloatArray


def ate_se3(pred_centers: FloatArray, gt_centers: FloatArray) -> float:
    """RMSE of camera centers with NO Sim(3) alignment. Metric claim metric."""
    p = np.asarray(pred_centers, dtype=np.float64)
    g = np.asarray(gt_centers, dtype=np.float64)
    if p.shape != g.shape:
        raise ValueError("shape mismatch")
    return float(np.sqrt(np.mean(np.sum((p - g) ** 2, axis=1))))


def rpe_translation(pred_centers: FloatArray, gt_centers: FloatArray) -> float:
    p = np.asarray(pred_centers, dtype=np.float64)
    g = np.asarray(gt_centers, dtype=np.float64)
    dp = p[1:] - p[:-1]
    dg = g[1:] - g[:-1]
    return float(np.sqrt(np.mean(np.sum((dp - dg) ** 2, axis=1))))
