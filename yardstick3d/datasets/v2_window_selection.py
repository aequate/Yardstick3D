"""Cross-backbone v2 window selection (GT-free, cue-only, deterministic).

Implements `window_selection` of configs/prospective_cross_backbone_v2.DRAFT.yaml.
Inputs are camera timestamps and UTC-converted receiver_lla fixes ONLY. No
reference position, image content or backbone output may enter this module.
Thresholds are the pre-registered constants below; never tuned on data.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ---- pre-registered constants (YAML window_selection) ----
WINDOW_DURATION_S = 16.0
FRAMES_PER_WINDOW = 8
GRID_DT_S = 0.1
SPEED_HALF_STENCIL_S = 1.0
CUE_GAP_MAX_S = 0.5
MAX_DT_IMAGE_S = 0.06
CUE_COVERAGE_MIN = 0.8
VALID_FRAC_MIN = 0.8
V_MEAN_MIN = 1.5
SV_CV_MIN = 0.20
SV_DV_MIN = 1.5
SV_S_MIN = 0.10
CC_CV_MAX = 0.10
CC_S_MAX = 0.05
TIMING_TAUS_S = (-18.0, 18.0)
EDGE_MARGIN_S = 2.0  # 1 s speed-stencil margin each side
N_TEST = 4
N_VALIDATION = 2
N_CC = 2

ROLE_TEST = "prospective_test"
ROLE_VALIDATION = "validation"
ROLE_CC = "cc_control"


class SequenceIneligible(RuntimeError):
    """Cue-only ineligibility (fewer than n_test SV windows)."""


@dataclass(frozen=True)
class WindowStats:
    k: int
    t0: float  # elapsed seconds from first image (UTC)
    eligible: bool
    cls: str  # 'SV' | 'CC' | 'other' | 'ineligible'
    frames_matched: bool
    cue_coverage: float
    valid_frac: float
    v_mean: float
    cv: float
    dv: float
    S: float
    reasons: tuple = field(default_factory=tuple)


@dataclass(frozen=True)
class Selection:
    starts: list  # [{"t0","k","role","S","cls"}] in role order test, validation, cc
    n_sv_candidates: int

    def by_role(self, role: str) -> list:
        return [s for s in self.starts if s["role"] == role]


def unique_sorted_cue(cue_t, cue_xyz):
    """Sort by time, drop non-finite rows and duplicate timestamps (keep first)."""
    t = np.asarray(cue_t, float)
    p = np.asarray(cue_xyz, float).reshape(len(t), -1)
    ok = np.isfinite(t) & np.all(np.isfinite(p), axis=1)
    t, p = t[ok], p[ok]
    order = np.argsort(t, kind="stable")
    t, p = t[order], p[order]
    keep = np.concatenate([[True], np.diff(t) > 0.0]) if t.size else np.zeros(0, bool)
    return t[keep], p[keep]


def interp_on_grid(cue_t, cue_p, grid_t, gap_max: float = CUE_GAP_MAX_S) -> np.ndarray:
    """Linear interpolation of cue positions onto grid_t; NaN outside the span or in a gap > gap_max."""
    cue_t = np.asarray(cue_t, float)
    cue_p = np.asarray(cue_p, float)
    g = np.asarray(grid_t, float)
    out = np.full((g.size, cue_p.shape[1]), np.nan)
    if cue_t.size < 2:
        return out
    j = np.clip(np.searchsorted(cue_t, g, side="right") - 1, 0, cue_t.size - 2)
    dt = cue_t[j + 1] - cue_t[j]
    ok = (g >= cue_t[0]) & (g <= cue_t[-1]) & (dt <= gap_max)
    w = np.where(ok, (g - cue_t[j]) / np.where(dt > 0, dt, 1.0), 0.0)
    val = (1 - w)[:, None] * cue_p[j] + w[:, None] * cue_p[j + 1]
    out[ok] = val[ok]
    return out


def horizontal_speed_profile(cue_t, cue_enu, t_ref: float, t_last: float, dt: float = GRID_DT_S):
    """v_h on a 0.1 s UTC grid anchored at t_ref (first image, UTC).

    v_h(t) = ||p_EN(t+1) - p_EN(t-1)|| / 2 (centered 2 s difference); NaN if either
    end of the stencil is outside the cue span or inside a cue gap > 0.5 s.
    Returns (grid_elapsed_s, v_h).
    """
    n = int(np.floor((t_last - t_ref) / dt + 1e-9)) + 1
    idx = np.arange(n)
    grid_t = t_ref + idx * dt
    p = interp_on_grid(cue_t, np.asarray(cue_enu)[:, :2], grid_t)
    h = int(round(SPEED_HALF_STENCIL_S / dt))
    v = np.full(n, np.nan)
    if n > 2 * h:
        d = p[2 * h:] - p[: n - 2 * h]
        v[h : n - h] = np.linalg.norm(d, axis=1) / (2 * SPEED_HALF_STENCIL_S)
    return idx * dt, v


def polyline_length_3d(cue_t, cue_p, ta: float, tb: float) -> float:
    """3D polyline length of fixes strictly inside (ta, tb) plus interpolated endpoints; NaN if the span is left."""
    if ta < cue_t[0] or tb > cue_t[-1] or tb <= ta:
        return float("nan")
    m = (cue_t > ta) & (cue_t < tb)
    t_pts = np.concatenate([[ta], cue_t[m], [tb]])
    pts = np.stack([np.interp(t_pts, cue_t, cue_p[:, c]) for c in range(cue_p.shape[1])], axis=1)
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


def timing_sensitivity(cue_t, cue_xyz, t_ref: float, t0: float, taus=TIMING_TAUS_S) -> float:
    """S = max_tau |L(tau)/L(0) - 1|; taus whose window leaves the cue span are skipped; NaN if none remain."""
    L0 = polyline_length_3d(cue_t, cue_xyz, t_ref + t0, t_ref + t0 + WINDOW_DURATION_S)
    if not np.isfinite(L0) or L0 <= 0:
        return float("nan")
    vals = []
    for tau in taus:
        L = polyline_length_3d(cue_t, cue_xyz, t_ref + t0 + tau, t_ref + t0 + WINDOW_DURATION_S + tau)
        if np.isfinite(L):
            vals.append(abs(L / L0 - 1.0))
    return float(max(vals)) if vals else float("nan")


def _cue_interval_coverage(cue_t, frame_t) -> float:
    """Fraction of the 7 intervals fully covered: both endpoints inside the cue span and no
    cue gap > 0.5 s intersecting the interval (most conservative reading, see prereg notes)."""
    cov = 0
    gaps = np.diff(cue_t)
    for a, b in zip(frame_t[:-1], frame_t[1:]):
        if a < cue_t[0] or b > cue_t[-1]:
            continue
        lo = max(int(np.searchsorted(cue_t, a, side="right")) - 1, 0)
        hi = min(int(np.searchsorted(cue_t, b, side="left")), cue_t.size - 1)
        if hi <= lo or np.all(gaps[lo:hi] <= CUE_GAP_MAX_S):
            cov += 1
    return cov / 7.0


def frames_for_window(image_t, t_ref, t0):
    """Snap 8 uniform ideal times over [t0, t0+16] to the nearest image; None unless all within 0.06 s and unique."""
    ideal = t_ref + t0 + np.linspace(0.0, WINDOW_DURATION_S, FRAMES_PER_WINDOW)
    j = np.clip(np.searchsorted(image_t, ideal), 1, image_t.size - 1)
    left, right = image_t[j - 1], image_t[j]
    idx = np.where(np.abs(ideal - left) <= np.abs(right - ideal), j - 1, j)
    d = np.abs(image_t[idx] - ideal)
    if np.any(d > MAX_DT_IMAGE_S) or len(set(idx.tolist())) != idx.size:
        return None, ideal
    return idx, ideal


def candidate_starts(t_last_elapsed: float) -> list:
    """t0 = 16 k, k >= 1, with t0 + 16 + 2 <= last image elapsed."""
    out = []
    k = 1
    while WINDOW_DURATION_S * k + WINDOW_DURATION_S + EDGE_MARGIN_S <= t_last_elapsed + 1e-9:
        out.append(k)
        k += 1
    return out


def classify_window(k, cv, dv, S, v_mean, valid_frac, frames_ok, cue_cov) -> tuple:
    """Return (eligible, class, reasons) by the pre-registered rule (NaN comparisons are False)."""
    reasons = []
    if not frames_ok:
        reasons.append("frames_not_matched_0.06s")
    if not cue_cov >= CUE_COVERAGE_MIN:
        reasons.append("cue_coverage<0.8")
    if not valid_frac >= VALID_FRAC_MIN:
        reasons.append("valid_frac<0.8")
    if not v_mean >= V_MEAN_MIN:
        reasons.append("v_mean<1.5")
    if reasons:
        return False, "ineligible", tuple(reasons)
    if cv >= SV_CV_MIN and dv >= SV_DV_MIN and S >= SV_S_MIN:
        return True, "SV", ()
    if cv < CC_CV_MAX and S < CC_S_MAX:
        return True, "CC", ()
    return True, "other", ()


def window_statistics(image_t, cue_t, cue_xyz) -> list:
    """Per-candidate WindowStats. image_t and cue_t must both already be UTC seconds."""
    image_t = np.sort(np.asarray(image_t, float))
    cue_t, cue_xyz = unique_sorted_cue(cue_t, cue_xyz)
    if image_t.size < 2 or cue_t.size < 2:
        return []
    t_ref = float(image_t[0])
    t_last = float(image_t[-1] - t_ref)
    grid_el, v = horizontal_speed_profile(cue_t, cue_xyz, t_ref, float(image_t[-1]))
    out = []
    for k in candidate_starts(t_last):
        t0 = WINDOW_DURATION_S * k
        j0, j1 = int(round(t0 / GRID_DT_S)), int(round((t0 + WINDOW_DURATION_S) / GRID_DT_S))
        vw = v[j0 : j1 + 1]
        finite = np.isfinite(vw)
        valid_frac = float(finite.mean())
        if finite.any():
            vf = vw[finite]
            v_mean = float(vf.mean())
            cv = float(vf.std(ddof=0) / v_mean) if v_mean > 0 else float("nan")
            dv = float(np.percentile(vf, 95) - np.percentile(vf, 5))
        else:
            v_mean = cv = dv = float("nan")
        idx, ideal = frames_for_window(image_t, t_ref, t0)
        frames_ok = idx is not None
        cue_cov = _cue_interval_coverage(cue_t, image_t[idx] if frames_ok else ideal)
        S = timing_sensitivity(cue_t, cue_xyz, t_ref, t0)
        elig, cls, reasons = classify_window(k, cv, dv, S, v_mean, valid_frac, frames_ok, cue_cov)
        out.append(WindowStats(k, float(t0), elig, cls, frames_ok, float(cue_cov), valid_frac, v_mean, cv, dv, S, reasons))
    return out


def select_windows(stats: list, n_test=N_TEST, n_validation=N_VALIDATION, n_cc=N_CC) -> Selection:
    """Deterministic role assignment. Raises SequenceIneligible if fewer than n_test SV windows."""
    sv = sorted((w for w in stats if w.cls == "SV"), key=lambda w: (-w.S, w.t0))
    other = sorted((w for w in stats if w.cls == "other"), key=lambda w: w.t0)
    cc = sorted((w for w in stats if w.cls == "CC"), key=lambda w: w.t0)
    taken: dict = {}  # k -> entry

    def free(w):
        return w.k not in taken and (w.k - 1) not in taken and (w.k + 1) not in taken

    def take(w, role):
        taken[w.k] = {"t0": w.t0, "k": w.k, "role": role, "S": w.S, "cls": w.cls}

    n_taken = 0
    for w in sv:
        if n_taken >= n_test:
            break
        if free(w):
            take(w, ROLE_TEST)
            n_taken += 1
    if n_taken < n_test:
        raise SequenceIneligible(f"only {n_taken} non-adjacent SV windows (< {n_test}) => INELIGIBLE (cue-only)")
    n_val = 0
    for w in sv:
        if n_val >= n_validation:
            break
        if free(w):
            take(w, ROLE_VALIDATION)
            n_val += 1
    for w in other:
        if n_val >= n_validation:
            break
        if free(w):
            take(w, ROLE_VALIDATION)
            n_val += 1
    n_c = 0
    for w in cc:
        if n_c >= n_cc:
            break
        if free(w):
            take(w, ROLE_CC)
            n_c += 1
    order = {ROLE_TEST: 0, ROLE_VALIDATION: 1, ROLE_CC: 2}
    starts = sorted(taken.values(), key=lambda s: (order[s["role"]], s["t0"]))
    return Selection(starts=starts, n_sv_candidates=len(sv))


def select_windows_pooled(stats_by_seq: dict, rank_order: list, n_test=N_TEST, n_validation=N_VALIDATION,
                          n_cc=N_CC) -> Selection:
    """v5 amendment A3: the same role rules over the POOL of sequences (fixed a-priori rank order).

    Candidates are (sequence, k). SV ranking: S descending, ties -> sequence rank, then earlier t0.
    'other' and CC ordering: (sequence rank, t0). Non-adjacency applies within a sequence only.
    Raises SequenceIneligible (=> v5 STOP) if fewer than n_test pooled SV windows.
    """
    if sorted(stats_by_seq) != sorted(rank_order):
        raise ValueError("stats_by_seq keys != rank_order (the pool is fixed)")
    rank = {s: i for i, s in enumerate(rank_order)}
    pool = [(seq, w) for seq in rank_order for w in stats_by_seq[seq]]
    sv = sorted((p for p in pool if p[1].cls == "SV"), key=lambda p: (-p[1].S, rank[p[0]], p[1].t0))
    other = sorted((p for p in pool if p[1].cls == "other"), key=lambda p: (rank[p[0]], p[1].t0))
    cc = sorted((p for p in pool if p[1].cls == "CC"), key=lambda p: (rank[p[0]], p[1].t0))
    taken: dict = {}  # (seq, k) -> entry

    def free(p):
        seq, w = p
        return all((seq, w.k + d) not in taken for d in (-1, 0, 1))

    def take(p, role):
        seq, w = p
        taken[(seq, w.k)] = {"sequence": seq, "t0": w.t0, "k": w.k, "role": role, "S": w.S, "cls": w.cls}

    def fill(cands, role, n, count=0):
        for p in cands:
            if count >= n:
                break
            if free(p):
                take(p, role)
                count += 1
        return count

    n_taken = fill(sv, ROLE_TEST, n_test)
    if n_taken < n_test:
        raise SequenceIneligible(f"only {n_taken} non-adjacent pooled SV windows (< {n_test}) => v5 STOP")
    fill(other, ROLE_VALIDATION, n_validation, fill(sv, ROLE_VALIDATION, n_validation))
    fill(cc, ROLE_CC, n_cc)
    order = {ROLE_TEST: 0, ROLE_VALIDATION: 1, ROLE_CC: 2}
    starts = sorted(taken.values(), key=lambda s: (order[s["role"]], rank[s["sequence"]], s["t0"]))
    return Selection(starts=starts, n_sv_candidates=len(sv))
