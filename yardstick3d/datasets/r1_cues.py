"""Site replication R1 - secondary-arm cues (u-blox Doppler speed, u-blox altitude). Pure numpy.

Contract: configs/prospective_site_replication_r1.yaml (secondary_arms). Inputs are UTC-converted u-blox
samples (receiver_pvt / receiver_lla) and frozen backbone predictions ONLY. No reference value, no fitted
time offset (GNSS time -> UTC uses the published 18.0 s), no quality filtering.
"""
from __future__ import annotations

import numpy as np

from yardstick3d.datasets import r1_contract as r1c


# ------------------------------- GNSS time ----------------------------------
def gnss_week_tow_to_utc(week, tow, leap_s: float = r1c.GPST_MINUS_UTC_S):
    """GPS week + time-of-week (GPST) -> UTC unix seconds: GPS epoch + week*604800 + tow - 18.0 s.

    Vectorised. A sample with a negative/non-finite week or a tow outside [0, 604800] is returned as NaN
    (dropped downstream like any non-finite sample); the conversion is never fitted.
    """
    w = np.asarray(week, dtype=np.float64)
    s = np.asarray(tow, dtype=np.float64)
    ok = np.isfinite(w) & np.isfinite(s) & (w >= 0) & (s >= 0) & (s <= r1c.SECONDS_PER_WEEK)
    out = r1c.GPS_EPOCH_UNIX_S + w * r1c.SECONDS_PER_WEEK + s - float(leap_s)
    out = np.where(ok, out, np.nan)
    return float(out) if out.ndim == 0 else out


def pvt_speed_3d(vn, ve, vd):
    """|v| = sqrt(vel_n^2 + vel_e^2 + vel_d^2); NaN where any component is non-finite (dropped later)."""
    vn, ve, vd = (np.asarray(x, dtype=np.float64) for x in (vn, ve, vd))
    s = np.sqrt(vn * vn + ve * ve + vd * vd)
    return np.where(np.isfinite(vn) & np.isfinite(ve) & np.isfinite(vd), s, np.nan)


def finite_sorted_samples(t, x):
    """Drop non-finite (t, x) samples, sort by time, drop duplicate stamps (keep first)."""
    t = np.asarray(t, dtype=np.float64).ravel()
    x = np.asarray(x, dtype=np.float64).ravel()
    ok = np.isfinite(t) & np.isfinite(x)
    t, x = t[ok], x[ok]
    o = np.argsort(t, kind="stable")
    t, x = t[o], x[o]
    keep = np.concatenate([[True], np.diff(t) > 0.0]) if t.size else np.zeros(0, bool)
    return t[keep], x[keep]


# ------------------------------- PVT timescale gate --------------------------
def pvt_timescale_gate(pvt_utc, record_t, delta: float) -> dict:
    """Stage-1 PVT gate: residual d = t_utc - (record + delta).

    PASS iff |median d| < 0.25 s AND (P99 - P1) of d < 0.5 s. FAIL => the speed arm is NOT RUN for that
    sequence (disclosed); the primary arm is unaffected. delta is the camera-measured recorder offset (A2).
    """
    u = np.asarray(pvt_utc, dtype=np.float64)
    r = np.asarray(record_t, dtype=np.float64)
    d = u - (r + float(delta))
    d = d[np.isfinite(d)]
    rec = {"n": int(d.size), "delta_s": float(delta),
           "rule": f"|median| < {r1c.PVT_GATE_MEDIAN_MAX_S} s and P99-P1 < {r1c.PVT_GATE_SPREAD_MAX_S} s"}
    if d.size == 0:
        return {**rec, "median": None, "p1": None, "p99": None, "spread": None, "status": "FAIL",
                "reason": "no finite PVT samples"}
    med = float(np.median(d))
    p1, p99 = float(np.percentile(d, 1)), float(np.percentile(d, 99))
    spread = p99 - p1
    ok = abs(med) < r1c.PVT_GATE_MEDIAN_MAX_S and spread < r1c.PVT_GATE_SPREAD_MAX_S
    reason = None if ok else (f"|median| {abs(med):.3f} s >= {r1c.PVT_GATE_MEDIAN_MAX_S} s"
                              if not abs(med) < r1c.PVT_GATE_MEDIAN_MAX_S
                              else f"spread {spread:.3f} s >= {r1c.PVT_GATE_SPREAD_MAX_S} s")
    return {**rec, "median": med, "p1": p1, "p99": p99, "spread": spread,
            "status": "PASS" if ok else "FAIL", "reason": reason}


# ------------------------------- speed arm ----------------------------------
def speed_interval_lengths(frame_t, pvt_t, speed, max_gap_s: float = r1c.SPEED_GAP_MAX_S):
    """b_k^v = integral of linearly-interpolated |v| over [t_k, t_{k+1}] (exact trapezoid).

    Returns (b, usable): b has length n_frames-1 (NaN where unusable). An interval is unusable iff it is not
    covered by samples (t_k before the first or t_{k+1} after the last finite sample) or any consecutive
    sample gap > max_gap_s overlaps it. Non-finite samples are dropped first (so they can open a gap).
    """
    ft = np.asarray(frame_t, dtype=np.float64)
    t, v = finite_sorted_samples(pvt_t, speed)
    n = max(ft.size - 1, 0)
    b = np.full(n, np.nan)
    usable = np.zeros(n, dtype=bool)
    if t.size < 2:
        return b, usable
    gaps = np.diff(t)
    for k in range(n):
        a, c = float(ft[k]), float(ft[k + 1])
        if not (np.isfinite(a) and np.isfinite(c)) or c <= a:
            continue
        if a < t[0] or c > t[-1]:
            continue
        lo = max(int(np.searchsorted(t, a, side="right")) - 1, 0)  # last sample <= a
        hi = int(np.searchsorted(t, c, side="left"))                 # first sample >= c
        if np.any(gaps[lo:hi] > max_gap_s):                         # pairs (j, j+1), lo <= j < hi
            continue
        inner = t[(t > a) & (t < c)]
        pts = np.concatenate([[a], inner, [c]])
        vals = np.interp(pts, t, v)
        b[k] = float(np.sum(0.5 * (vals[1:] + vals[:-1]) * np.diff(pts)))
        usable[k] = True
    return b, usable


def speed_constraints(frame_t, pvt_t, speed, max_gap_s: float = r1c.SPEED_GAP_MAX_S,
                      min_coverage: float = r1c.COVERAGE_MIN, sigma: float = r1c.CUE_SIGMA,
                      max_dt_match: float = r1c.MAX_DT_CUE_S, cue_topic: str = r1c.PVT_TOPIC):
    """PathDisplacementConstraint s*a_k ~= b_k^v (the primary arm's constraint type and parameters).

    Mirrors mars_lvig.constraints_from_ublox_displacement: unusable intervals are DROPPED (never zero-
    filled). Returns (ConstraintSet, info); info["valid"] iff >= 80 % of the intervals are usable.
    """
    from yardstick3d.constraints.base import ConstraintSet
    from yardstick3d.constraints.speed import PathDisplacementConstraint

    r1c.check_cue_topic(cue_topic)
    ft = np.asarray(frame_t, dtype=np.float64)
    b, usable = speed_interval_lengths(ft, pvt_t, speed, max_gap_s)
    cs = ConstraintSet()
    for k in np.flatnonzero(usable):
        cs.add(PathDisplacementConstraint(t_i=float(ft[k]), t_j=float(ft[k + 1]), length_m=float(b[k]),
                                          sigma=float(sigma), max_dt_match=float(max_dt_match)))
    n = int(b.size)
    cov = float(usable.sum() / n) if n else 0.0
    info = {"n_intervals": n, "n_kept": int(usable.sum()), "n_dropped": int(n - usable.sum()),
            "coverage": cov, "valid": bool(n > 0 and cov >= min_coverage),
            "cue_lengths_m": [float(x) if np.isfinite(x) else None for x in b], "cue_topic": cue_topic}
    return cs, info


def speed_displacement_ratio(b_speed, usable_speed, b_disp, cov_disp, min_cov: float = r1c.COVERAGE_MIN) -> dict:
    """GT-free diagnostic: sum_k b_k^v / sum_k b_k over intervals usable for BOTH cues."""
    bs, bd = np.asarray(b_speed, float), np.asarray(b_disp, float)
    m = np.asarray(usable_speed, bool) & (np.asarray(cov_disp, float) >= min_cov) & np.isfinite(bs) & np.isfinite(bd)
    den = float(bd[m].sum())
    return {"n_intervals": int(m.sum()), "ratio": float(bs[m].sum() / den) if m.any() and den > 0 else None}


# ------------------------------- altitude arm --------------------------------
def interp_no_gap(t, x, tq, max_gap_s: float = r1c.ALT_GAP_MAX_S):
    """Linear interpolation at tq; NaN outside the sample span or across a gap > max_gap_s."""
    t, x = finite_sorted_samples(t, x)
    q = np.atleast_1d(np.asarray(tq, dtype=np.float64))
    out = np.full(q.shape, np.nan)
    if t.size == 0:
        return out
    for i, tv in enumerate(q):
        if not np.isfinite(tv) or tv < t[0] or tv > t[-1]:
            continue
        j = int(np.searchsorted(t, tv, side="left"))
        if j < t.size and t[j] == tv:
            out[i] = x[j]
            continue
        if t[j] - t[j - 1] > max_gap_s:
            continue
        w = (tv - t[j - 1]) / (t[j] - t[j - 1])
        out[i] = (1.0 - w) * x[j - 1] + w * x[j]
    return out


def up_axis_from_nadir(R_w2c) -> np.ndarray:
    """GT-free vertical from a downward camera: e_up = -normalize(mean_k R_k^T [0,0,1]).

    R_w2c: (N,3,3) (or (N,3,4) / (N,4,4) poses) OpenCV world-to-camera; R_k^T [0,0,1] is camera k's optical
    axis in the backbone world frame, which approximates gravity-down up to platform tilt (disclosed).
    """
    R = np.asarray(R_w2c, dtype=np.float64)
    if R.ndim == 2:
        R = R[None]
    R = R[:, :3, :3]
    axes = R[:, 2, :]  # R^T [0,0,1] = third row of R
    m = axes.mean(axis=0)
    nrm = float(np.linalg.norm(m))
    if not np.isfinite(nrm) or nrm < 1e-12:
        return np.full(3, np.nan)
    return -m / nrm


def altitude_arm(frame_t, cue_t, cue_alt, centers, R_w2c, dh_min: float = r1c.ALT_DH_MIN_M,
                 ratio_min: float = r1c.ALT_RATIO_MIN, max_gap_s: float = r1c.ALT_GAP_MAX_S,
                 cue_topic: str = r1c.LLA_TOPIC) -> dict:
    """Altitude-arm scale s_alt = dh / dv with the pre-registered observability rule.

    dh = h(t_last) - h(t_first) (receiver_lla altitude, linear interpolation, no interpolation across a gap
    > 0.5 s); dv = e_up . (C_last - C_first). OBSERVABLE iff |dh| >= 5.0 m AND |dv|/||C_last - C_first||
    >= 0.2; then s_alt <= 0 or non-finite -> FAILURE (counted, no substitution). Otherwise NOT_OBSERVABLE
    (no scale applied). Decided from cue + prediction only.
    """
    r1c.check_cue_topic(cue_topic)
    ft = np.asarray(frame_t, dtype=np.float64)
    C = np.asarray(centers, dtype=np.float64)
    out = {"status": r1c.ALT_NOT_OBSERVABLE, "s_alt": None, "dh": None, "dv": None, "ratio": None,
           "chord": None, "e_up": None, "reason": None, "cue_topic": cue_topic}
    h = interp_no_gap(cue_t, cue_alt, [ft[0], ft[-1]], max_gap_s)
    if not np.all(np.isfinite(h)):
        out["reason"] = f"altitude unavailable at first/last frame (span or gap > {max_gap_s} s)"
        return out
    dh = float(h[1] - h[0])
    e_up = up_axis_from_nadir(R_w2c)
    D = C[-1] - C[0]
    chord = float(np.linalg.norm(D))
    dv = float(e_up @ D) if np.all(np.isfinite(e_up)) else float("nan")
    ratio = abs(dv) / chord if (np.isfinite(dv) and chord > 0) else float("nan")
    out.update({"dh": dh, "dv": dv if np.isfinite(dv) else None, "ratio": ratio if np.isfinite(ratio) else None,
                "chord": chord, "e_up": [float(x) for x in e_up] if np.all(np.isfinite(e_up)) else None})
    if not abs(dh) >= dh_min:
        out["reason"] = f"|dh| {abs(dh):.3f} m < {dh_min} m"
        return out
    if not ratio >= ratio_min:
        out["reason"] = f"|dv|/chord {ratio:.3f} < {ratio_min}"
        return out
    s = dh / dv if dv != 0 else float("nan")
    if not (np.isfinite(s) and s > 0):
        out.update({"status": r1c.ALT_FAILURE, "reason": f"s_alt {s} non-positive or non-finite (counted, no substitution)",
                    "s_alt": float(s) if np.isfinite(s) else None})
        return out
    out.update({"status": r1c.ALT_OBSERVABLE, "s_alt": float(s)})
    return out


def vertical_dh_window(cue_t, cue_alt, t0: float, dur: float = r1c.WINDOW_DURATION_S,
                       edge: float = r1c.VERTICAL_EDGE_S) -> float:
    """dh_window = median altitude over [t0+dur-edge, t0+dur] - median over [t0, t0+edge] (UTC); NaN if empty."""
    t, a = finite_sorted_samples(cue_t, cue_alt)
    first = a[(t >= t0) & (t <= t0 + edge)]
    last = a[(t >= t0 + dur - edge) & (t <= t0 + dur)]
    if first.size == 0 or last.size == 0:
        return float("nan")
    return float(np.median(last) - np.median(first))
