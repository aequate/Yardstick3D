"""Header-timescale guard for ROS bag sensor topics.

Added after cross-backbone v1: the MARS-LVIG u-blox `receiver_lla` header stamps
were in GPS time (+18 s vs bag record time) while camera stamps were UTC, and
Stage-1 used both raw. This guard compares each topic's header stamp with the
bag record time and refuses unexplained offsets. A GPST->UTC correction uses the
published leap-second constant; it is never fitted.
"""
from __future__ import annotations

import numpy as np

GPST_MINUS_UTC_S = 18.0  # leap seconds, valid since 2017-01-01
DEFAULT_TOL_S = 0.5


def header_offset_stats(header_s, record_s) -> dict:
    d = np.asarray(header_s, float) - np.asarray(record_s, float)
    if d.size == 0:
        raise ValueError("no samples")
    return {"n": int(d.size), "median": float(np.median(d)), "min": float(d.min()), "max": float(d.max())}


def classify_timescale(median_offset_s: float, tol_s: float = DEFAULT_TOL_S) -> str:
    """'utc' if header ~ record time, 'gpst' if ~ +18 s leap offset, else 'unknown'."""
    if abs(median_offset_s) < tol_s:
        return "utc"
    if abs(median_offset_s - GPST_MINUS_UTC_S) < tol_s:
        return "gpst"
    return "unknown"


def to_utc(header_s, timescale: str):
    """Convert header stamps to UTC using the published constant only."""
    t = np.asarray(header_s, float)
    if timescale == "utc":
        return t
    if timescale == "gpst":
        return t - GPST_MINUS_UTC_S
    raise ValueError(f"cannot convert timescale {timescale!r}")


def require_common_timescale(topics: dict, tol_s: float = DEFAULT_TOL_S) -> dict:
    """topics: {name: (header_s, record_s)}. Raise unless every topic is utc or gpst.

    Returns {name: timescale}. Callers must apply `to_utc` before pairing streams.
    """
    out = {}
    for name, (h, r) in topics.items():
        s = header_offset_stats(h, r)
        ts = classify_timescale(s["median"], tol_s)
        if ts == "unknown":
            raise RuntimeError(f"{name}: header-record offset {s['median']:+.3f} s is unexplained")
        out[name] = ts
    return out
