"""Cross-backbone v2 shared, pure (numpy-only) helpers.

Used by scripts/cross_backbone_v2_da3.py, build_vggt_pack_v2.py, cross_backbone_v2_score.py.
No bag I/O, no reference access. Contract: configs/prospective_cross_backbone_v5.yaml (FROZEN; v5 = v4 + A3 pooled SV).
Windows may carry a "sequence" key (v5 pool); windows without it belong to SEQUENCE (v2-v4 single-sequence layout).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / "configs/prospective_cross_backbone_v5.yaml"  # v5 = v4 + A3 (pooled SV)
REGISTRY_PATH = ROOT / "configs/experiment_registry.json"
CONTRACT_SHA = "79d67798ba7b770c4a10eeb1d2c262a38b88237b37d3a735a6bfdee0a7867b60"
SEQUENCE = "HKisland_GNSS02"  # rank 1; default for windows without a "sequence" key
POOL = ("HKisland_GNSS02", "HKisland_GNSS03", "HKisland_GNSS01")  # contract dataset.pool (fixed rank order)
PACK_ID = "vggt_offbox_cross_backbone_v5"

# frozen scalar parameters (contract: reference_handling / matching / solver / nominal prior)
MAX_DT_CUE_S = 0.06
MAX_DT_REF_S = 0.15
FIXED_VALUES = (50,)
REF_VALID_REQUIRED = 7
N_FRAMES = 8
WINDOW_DURATION_S = 16.0
V_CRUISE_TABLE_MPS = 6.0  # dataset-table cruise speed for HKisland_GNSS02 (contract: sequence_selection.rank_order)
V_CRUISE_TABLE_BY_SEQ = {"HKisland_GNSS02": 6.0, "HKisland_GNSS03": 9.0, "HKisland_GNSS01": 3.0}  # dataset.pool
SHIFT_TAUS_S = (-18.0, 18.0)

ROLE_TEST = "prospective_test"
ROLE_VALIDATION = "validation"
ROLE_CC = "cc_control"
SECTION_OF_ROLE = {ROLE_TEST: "sv_test", ROLE_VALIDATION: "validation", ROLE_CC: "cc_control"}


def seq_of(w: dict) -> str:
    return w.get("sequence", SEQUENCE)


def frame_relpath(w: dict, k: int) -> str:
    """Stage-1 frame file (relative to stage1/frames): v5 namespaces frames by sequence."""
    return f"{w['sequence']}/{int(k):08d}.png" if "sequence" in w else f"{int(k):08d}.png"


def cue_csv_name(seq: str | None) -> str:
    return f"cue_{seq}.csv" if seq else "cue_receiver_lla.csv"


def v_cruise_for(w: dict) -> float:
    return V_CRUISE_TABLE_BY_SEQ[w["sequence"]] if "sequence" in w else V_CRUISE_TABLE_MPS


def sha256_file(path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def check_contract(path: Path = CONTRACT_PATH, expected: str = CONTRACT_SHA, registry: Path | None = REGISTRY_PATH) -> str:
    """Raise unless the contract bytes hash to the frozen sha (and, if given, it is in the registry)."""
    path = Path(path)
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"v2 contract hash mismatch: {actual} != {expected}")
    if registry is not None:
        registry = Path(registry)
        if not registry.exists() or expected not in registry.read_text(encoding="utf-8"):
            raise ValueError(f"v2 contract sha256 not recorded in {registry}")
    return actual


def clean_json(obj):
    if isinstance(obj, dict):
        return {str(k): clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_json(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if np.isfinite(f) else None
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return clean_json(obj.tolist())
    return obj


def dump_json(path: Path, obj, exclusive: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x" if exclusive else "w", encoding="utf-8") as f:
        f.write(json.dumps(clean_json(obj), indent=2, allow_nan=False))


def guard_receipt(path: Path) -> None:
    if Path(path).exists():
        raise FileExistsError(f"scoring receipt exists; refusing replay: {path}")


# ------------------------------- cue / chords -------------------------------
def window_origin(cue_t, cue_llh, t_start: float):
    """Per-window ENU origin: first finite cue fix at/after t_start (else the first finite fix)."""
    t = np.asarray(cue_t, float)
    llh = np.asarray(cue_llh, float)
    ok = np.all(np.isfinite(llh[:, :2]), axis=1)
    cand = np.flatnonzero(ok & (t >= t_start))
    i = int(cand[0]) if cand.size else int(np.flatnonzero(ok)[0])
    alt = float(llh[i, 2]) if np.isfinite(llh[i, 2]) else 0.0
    return float(llh[i, 0]), float(llh[i, 1]), alt


def chords(centers) -> np.ndarray:
    """Visual chords a_k = ||P_{k+1} - P_k||."""
    c = np.asarray(centers, float)
    return np.linalg.norm(np.diff(c, axis=0), axis=1)


def _rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    r = np.empty(x.size, float)
    r[order] = np.arange(x.size, dtype=float)
    xs = x[order]
    i = 0
    while i < x.size:  # average ranks over ties
        j = i
        while j + 1 < x.size and xs[j + 1] == xs[i]:
            j += 1
        r[order[i : j + 1]] = 0.5 * (i + j)
        i = j + 1
    return r


def spearman(a, b) -> float | None:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return None
    ra, rb = _rank(a[ok]), _rank(b[ok])
    if np.std(ra) == 0 or np.std(rb) == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def rho_diagnostic(centers, frame_t, cue_t, cue_enu, taus=SHIFT_TAUS_S) -> dict:
    """GT-free Spearman rho(a_k, b_k) between visual chords and cue lengths, plus cue shifted by +/-tau.

    Inputs are backbone centres, UTC frame times and the UTC cue only. Disclosure only (contract:
    gt_free_pre_scoring_diagnostic); never used for gating or selection.
    """
    from yardstick3d.datasets import mars_lvig as ml

    a = chords(centers)
    out = {}
    for tau in (0.0, *taus):
        b, cov = ml.cue_polyline_lengths(np.asarray(cue_t, float) + tau, cue_enu, np.asarray(frame_t, float))
        b = np.where(cov >= 0.8, b, np.nan)
        out["rho_tau_0" if tau == 0.0 else f"rho_tau_{tau:+.0f}"] = spearman(a, b)
    out["n_chords"] = int(a.size)
    return out


def nominal_speed_prior_scale(chord_sum: float, v_cruise: float = V_CRUISE_TABLE_MPS,
                              duration: float = WINDOW_DURATION_S) -> float | None:
    """s_prior = v_cruise * 16 s / sum_k a_k (no sensor)."""
    if not (np.isfinite(chord_sum) and chord_sum > 0):
        return None
    return float(v_cruise * duration / chord_sum)


# ------------------------------- reference rule -----------------------------
def _nearest(t_sorted: np.ndarray, q: np.ndarray):
    if t_sorted.size == 0:
        return np.full(q.shape, -1, int), np.full(q.shape, np.inf)
    j = np.clip(np.searchsorted(t_sorted, q), 1, max(t_sorted.size - 1, 1))
    if t_sorted.size == 1:
        idx = np.zeros(q.shape, int)
    else:
        idx = np.where(np.abs(q - t_sorted[j - 1]) <= np.abs(t_sorted[j] - q), j - 1, j)
    return idx, np.abs(t_sorted[idx] - q)


def ref_valid_frames(frame_t, ref_t, info_t, info_status, fixed_values=FIXED_VALUES, max_dt=MAX_DT_REF_S) -> dict:
    """Per-frame REF-VALID (contract reference_handling.per_frame_match).

    A frame is REF-VALID iff a reference position sample (ref_t, UTC) AND a fix-status sample (info_t, UTC)
    lie within max_dt of the frame's UTC stamp AND the status is in fixed_values.
    Returns mask + the log fields (n_matched, n_fixed, max_abs_dt_s).
    """
    q = np.asarray(frame_t, float)
    rt = np.asarray(ref_t, float)
    it = np.asarray(info_t, float)
    st = np.asarray(info_status)
    ro, io = np.argsort(rt, kind="stable"), np.argsort(it, kind="stable")
    ri, rd = _nearest(rt[ro], q)
    ii, idt = _nearest(it[io], q)
    ref_ok = rd <= max_dt
    info_ok = idt <= max_dt
    matched = ref_ok & info_ok
    status = st[io][ii] if st.size else np.zeros(q.shape, int)
    fixed = matched & np.isin(status, list(fixed_values))
    used = np.concatenate([rd[matched], idt[matched]]) if matched.any() else np.array([])
    return {
        "mask": fixed,
        "ref_index": ro[ri] if rt.size else np.full(q.shape, -1, int),
        "n_matched": int(matched.sum()),
        "n_fixed": int(fixed.sum()),
        "max_abs_dt_s": float(used.max()) if used.size else None,
    }


def window_evaluable(mask, required: int = REF_VALID_REQUIRED) -> bool:
    """Window evaluable iff >= 7 of 8 frames are REF-VALID."""
    return int(np.asarray(mask, bool).sum()) >= required


class SubsetPred:
    """Restrict a prediction to the REF-VALID frame subset for ATE (evaluate_grounding uses only centers())."""

    def __init__(self, pred, mask):
        self._c = np.asarray(pred.centers(), float)[np.asarray(mask, bool)]

    def centers(self):
        return self._c


def median(vals):
    v = [x for x in vals if x is not None and np.isfinite(x)]
    return float(np.median(v)) if v else None
