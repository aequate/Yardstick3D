from __future__ import annotations

import numpy as np

from yardstick3d.types import FloatArray


def umeyama(src: FloatArray, dst: FloatArray, with_scale: bool) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares similarity: dst ≈ s R src + t. If not with_scale, s=1 (SE3)."""
    a = np.asarray(src, dtype=np.float64).reshape(-1, 3)
    b = np.asarray(dst, dtype=np.float64).reshape(-1, 3)
    if a.shape != b.shape or a.shape[0] < 3:
        raise ValueError("need >=3 corresponding 3D points")
    mu_a = a.mean(axis=0)
    mu_b = b.mean(axis=0)
    a0 = a - mu_a
    b0 = b - mu_b
    cov = (b0.T @ a0) / a.shape[0]
    u, d, vt = np.linalg.svd(cov)
    s_det = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        s_det[2, 2] = -1
    R = u @ s_det @ vt
    if with_scale:
        var_a = float(np.sum(a0**2) / a.shape[0])
        scale = float(np.trace(np.diag(d) @ s_det) / max(var_a, 1e-18))
        if scale <= 0:
            scale = 1.0
    else:
        scale = 1.0
    t = mu_b - scale * (R @ mu_a)
    return scale, R, t


def apply_se3(points: FloatArray, R: np.ndarray, t: np.ndarray) -> FloatArray:
    p = np.asarray(points, dtype=np.float64)
    return (R @ p.reshape(-1, 3).T).T + t


def apply_sim3(points: FloatArray, s: float, R: np.ndarray, t: np.ndarray) -> FloatArray:
    p = np.asarray(points, dtype=np.float64)
    return s * (R @ p.reshape(-1, 3).T).T + t


def ate_rmse(pred: FloatArray, gt: FloatArray) -> float:
    p = np.asarray(pred, dtype=np.float64)
    g = np.asarray(gt, dtype=np.float64)
    return float(np.sqrt(np.mean(np.sum((p - g) ** 2, axis=1))))
