from __future__ import annotations

import numpy as np

from yardstick3d.geometry.sim3 import Sim3
from yardstick3d.types import FloatArray


def camera_centers_from_w2c(T_w2c: FloatArray) -> FloatArray:
    """OpenCV world-to-camera [R|t]: x_cam = R X_world + t, so C = -R^T t."""
    T = np.asarray(T_w2c, dtype=np.float64)
    if T.ndim == 2:
        T = T[None, ...]
    R = T[:, :3, :3]
    t = T[:, :3, 3]
    # C = -R^T t.  (R^T t)_j = sum_i R_ij t_i = einsum("nij,ni->nj", R, t)
    centers = np.einsum("nij,ni->nj", R, -t)
    return centers


def w2c_from_centers(R_w2c: FloatArray, centers: FloatArray) -> FloatArray:
    """Build [R|t] from rotations and camera centers. t = -R C."""
    R = np.asarray(R_w2c, dtype=np.float64)
    C = np.asarray(centers, dtype=np.float64)
    if R.ndim == 2:
        R = R[None, ...]
        C = C[None, ...]
    t = -np.einsum("nij,nj->ni", R, C)
    T = np.zeros((R.shape[0], 3, 4), dtype=np.float64)
    T[:, :3, :3] = R
    T[:, :3, 3] = t
    return T


def invert_w2c(T_w2c: FloatArray) -> FloatArray:
    """Return T_c2w 4x4."""
    T = np.asarray(T_w2c, dtype=np.float64)
    single = T.ndim == 2
    if single:
        T = T[None, ...]
    n = T.shape[0]
    out = np.repeat(np.eye(4)[None, ...], n, axis=0)
    R = T[:, :3, :3]
    t = T[:, :3, 3]
    Rt = np.transpose(R, (0, 2, 1))
    out[:, :3, :3] = Rt
    out[:, :3, 3] = -np.einsum("nji,ni->nj", Rt, t)
    return out[0] if single else out


def apply_sim3_w2c(T_w2c: FloatArray, sim: Sim3) -> FloatArray:
    """World points transform as X' = s R X + t. Camera centers transform identically.

    Camera-frame observations are unchanged except for metric scale of depth.
    R_w2c' = R_w2c R_sim^T,  C' = s R_sim C + t_sim,  t' = -R_w2c' C'.
    """
    T = np.asarray(T_w2c, dtype=np.float64)
    single = T.ndim == 2
    if single:
        T = T[None, ...]
    R_w2c = T[:, :3, :3]
    C = camera_centers_from_w2c(T)
    R_sim = np.asarray(sim.R, dtype=np.float64)
    C_m = float(sim.scale) * (R_sim @ C.T).T + np.asarray(sim.t).reshape(3)
    R_new = R_w2c @ R_sim.T
    T_new = w2c_from_centers(R_new, C_m)
    return T_new[0] if single else T_new
