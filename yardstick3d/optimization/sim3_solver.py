from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from yardstick3d.geometry.sim3 import Sim3
from yardstick3d.types import FloatArray


def solve_sim3_from_displacements(
    src_disp: FloatArray,
    dst_disp: FloatArray,
    weights: FloatArray | None = None,
) -> Sim3:
    """Align displacement vectors: dst ≈ s R src (no translation from displacements).

    Translation remains unobservable from pure displacements. Returns t=0.
    """
    A = np.asarray(src_disp, dtype=np.float64).reshape(-1, 3)
    B = np.asarray(dst_disp, dtype=np.float64).reshape(-1, 3)
    if weights is None:
        w = np.ones(A.shape[0])
    else:
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
    W = np.sqrt(w)[:, None]
    A_w = A * W
    B_w = B * W
    H = A_w.T @ B_w
    U, _S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    RA = (R @ A.T).T
    numer = float(np.sum(w[:, None] * RA * B))
    denom = float(np.sum(w[:, None] * RA * RA))
    s = numer / max(denom, 1e-18)
    if s < 0:
        s = abs(s)
        R = Rotation.from_rotvec(np.array([0.0, 0.0, np.pi])).as_matrix() @ R
    return Sim3(scale=float(s), R=R, t=np.zeros(3))
