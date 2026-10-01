"""Speed integration. Method frozen BEFORE looking at real metric scores.

Primary method: trapezoidal integration of linearly interpolated speed.
Invalid CoreLocation speeds (NaN, <0, Apple kCLLocationSpeedInvalid typically -1)
are NOT treated as 0 m/s; they are dropped and the integral uses only valid samples.
If an interval has no valid sample, it contributes no constraint.
"""

from __future__ import annotations

import numpy as np

from yardstick3d.types import FloatArray

SPEED_INTEGRATION_METHOD = "trapezoidal_linear"
INVALID_SPEED = -1.0


def invalid_speed_mask(speed: FloatArray) -> np.ndarray:
    s = np.asarray(speed, dtype=np.float64)
    return (~np.isfinite(s)) | (s < 0.0) | np.isclose(s, INVALID_SPEED)


def integrate_speed(
    t_i: float,
    t_j: float,
    t_samples: FloatArray,
    speed: FloatArray,
    method: str = SPEED_INTEGRATION_METHOD,
) -> tuple[float, int] | None:
    """Return (distance_m, n_valid) over [t_i, t_j], or None if unusable."""
    if t_j <= t_i:
        return None
    t = np.asarray(t_samples, dtype=np.float64)
    v = np.asarray(speed, dtype=np.float64)
    bad = invalid_speed_mask(v)
    t = t[~bad]
    v = v[~bad]
    if t.size == 0:
        return None
    t0, t1 = float(t_i), float(t_j)
    inside = (t >= t0) & (t <= t1)
    # Endpoints via linear interpolation when neighbors exist.
    v0 = _interp_speed(t0, t, v)
    v1 = _interp_speed(t1, t, v)
    ts = [t0]
    vs = [v0]
    for ti, vi in zip(t[inside], v[inside]):
        ts.append(float(ti))
        vs.append(float(vi))
    ts.append(t1)
    vs.append(v1)
    order = np.argsort(ts)
    ts_a = np.asarray(ts, dtype=np.float64)[order]
    vs_a = np.asarray(vs, dtype=np.float64)[order]
    finite = np.isfinite(vs_a)
    if finite.sum() < 2:
        return None
    ts_a = ts_a[finite]
    vs_a = vs_a[finite]
    if ts_a[-1] <= ts_a[0]:
        return None
    if method == "zero_order_hold":
        d = 0.0
        for k in range(len(ts_a) - 1):
            d += vs_a[k] * (ts_a[k + 1] - ts_a[k])
    elif method in ("trapezoidal_linear", "linear"):
        d = float(np.sum(0.5 * (vs_a[1:] + vs_a[:-1]) * np.diff(ts_a)))
    else:
        raise ValueError(method)
    if not np.isfinite(d) or d < 0:
        return None
    return float(d), int(finite.sum())


def _interp_speed(tq: float, t: FloatArray, v: FloatArray) -> float:
    if t.size == 0:
        return float("nan")
    if tq <= t[0]:
        return float(v[0]) if abs(tq - t[0]) < 1.0 else float("nan")
    if tq >= t[-1]:
        return float(v[-1]) if abs(tq - t[-1]) < 1.0 else float("nan")
    k = int(np.searchsorted(t, tq, side="right") - 1)
    k = min(max(k, 0), len(t) - 2)
    w = (tq - t[k]) / max(t[k + 1] - t[k], 1e-12)
    return float((1.0 - w) * v[k] + w * v[k + 1])
