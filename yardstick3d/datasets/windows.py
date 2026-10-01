"""GT-independent ADVIO window generation (inference channels only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from yardstick3d.datasets.advio import InferenceChannels, corelocation_interval_distance
from yardstick3d.types import FloatArray

FROZEN_V1 = {
    "t_origin_s": 0.0,
    "duration_s": 16.0,
    "stride_s": 16.0,
    "n_frames": 8,
    "min_gps_path_m": 10.0,
    "gps_max_extrap_s": 1.0,
}

WINDOW_KEYS = ("t0", "t1", "frame_indices", "gps_path_m")


def linspace_frame_indices(
    timestamps: FloatArray, t0: float, t1: float, n_keep: int
) -> np.ndarray | None:
    m = (timestamps >= t0) & (timestamps <= t1)
    idx = np.flatnonzero(m)
    if idx.size < n_keep:
        return None
    take = np.linspace(0, idx.size - 1, n_keep).round().astype(int)
    out = idx[take]
    if np.unique(out).size < n_keep:
        return None
    return out


def gps_samples_span_window(
    location_t: FloatArray, t0: float, t1: float, max_extrap_s: float
) -> bool:
    t = np.asarray(location_t, dtype=np.float64)
    if t.size < 2:
        return False
    if t0 < float(t[0]) - max_extrap_s or t1 > float(t[-1]) + max_extrap_s:
        return False
    return bool(np.any(t <= t0 + max_extrap_s) and np.any(t >= t1 - max_extrap_s))


def generate_windows(
    inf: InferenceChannels,
    *,
    duration_s: float = 16.0,
    stride_s: float = 16.0,
    n_frames: int = 8,
    min_gps_path_m: float = 10.0,
    t_origin_s: float = 0.0,
    gps_max_extrap_s: float = 1.0,
) -> list[dict[str, Any]]:
    inf.assert_imu_denied()
    ts = np.asarray(inf.frame_timestamps, dtype=np.float64)
    if ts.size < n_frames or duration_s <= 0.0 or stride_s <= 0.0:
        return []
    t_end = float(ts[-1])
    out: list[dict[str, Any]] = []
    t0 = float(t_origin_s)
    while t0 + duration_s <= t_end + 1e-12:
        t1 = t0 + duration_s
        if gps_samples_span_window(inf.location_t, t0, t1, gps_max_extrap_s):
            dist = corelocation_interval_distance(inf, t0, t1)
            idx = linspace_frame_indices(ts, t0, t1, n_frames)
            if dist is not None and dist >= min_gps_path_m and idx is not None:
                out.append(
                    {
                        "t0": float(t0),
                        "t1": float(t1),
                        "frame_indices": [int(i) for i in idx],
                        "gps_path_m": float(dist),
                    }
                )
        t0 += stride_s
    return out


def windows_json_payload(
    sequence_id: str,
    windows: list[dict[str, Any]],
    *,
    role: str,
    config: str = "configs/splits/advio_windows_v1.yaml",
    parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "name": "advio_windows_v1",
        "config": config,
        "sequence": sequence_id,
        "role": role,
        "gt_used": False,
        "imu_used": False,
        "cue_source": "corelocation_endpoint_interp",
        "n_windows": len(windows),
        "parameters": dict(parameters or FROZEN_V1),
        "windows": windows,
    }
