from __future__ import annotations

import numpy as np

from yardstick3d.types import FloatArray


def displacements(centers: FloatArray) -> FloatArray:
    C = np.asarray(centers, dtype=np.float64)
    return C[1:] - C[:-1]


def path_length(centers: FloatArray) -> float:
    d = displacements(centers)
    return float(np.sum(np.linalg.norm(d, axis=1)))


def find_frame_index(timestamps: FloatArray, t: float, max_dt: float = np.inf) -> int | None:
    ts = np.asarray(timestamps, dtype=np.float64)
    k = int(np.argmin(np.abs(ts - t)))
    if abs(ts[k] - t) > max_dt:
        return None
    return k
