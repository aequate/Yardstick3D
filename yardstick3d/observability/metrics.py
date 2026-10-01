from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from yardstick3d.constraints.base import ConstraintSet
from yardstick3d.optimization.scale_solver import collect_scale_observations
from yardstick3d.types import PredictionBundle


@dataclass
class ObservabilityReport:
    score: float
    observable: bool
    fisher_s: float
    predicted_scale_std: float
    n_observations: int
    path_length: float
    mean_baseline: float
    reasons: list[str]


def motion_excitation(prediction: PredictionBundle) -> dict[str, float]:
    C = prediction.centers()
    d = C[1:] - C[:-1]
    lengths = np.linalg.norm(d, axis=1)
    pl = float(np.sum(lengths))
    span = float(np.linalg.norm(C.max(axis=0) - C.min(axis=0)))
    return {
        "path_length": pl,
        "max_step": float(np.max(lengths) if len(lengths) else 0.0),
        "mean_step": float(np.mean(lengths) if len(lengths) else 0.0),
        "bbox_span": span,
        "n_frames": float(prediction.n_frames()),
    }


def interval_scales(
    prediction: PredictionBundle, constraints: ConstraintSet
) -> tuple[np.ndarray, np.ndarray]:
    """Implied interval scales s_k = b_k / a_k and GPS (or cue) sigmas."""
    obs = collect_scale_observations(prediction, constraints)
    s: list[float] = []
    sig: list[float] = []
    for _name, a, b, var in obs:
        if a > 1e-15:
            s.append(float(b / a))
            sig.append(float(np.sqrt(var)))
    return np.asarray(s, dtype=np.float64), np.asarray(sig, dtype=np.float64)


def _std_over_median(s: np.ndarray) -> float:
    if s.size < 2:
        return float("nan")
    med = float(np.median(s))
    if abs(med) < 1e-15:
        return float("nan")
    return float(np.std(s, ddof=0) / abs(med))


def pairwise_scale_cv(prediction: PredictionBundle, constraints: ConstraintSet) -> float:
    """GT-free coefficient of variation of implied interval scales s_k = b_k / a_k.

    High CV: inconsistent GPS/visual ratios (shape error, GPS spikes, or misalignment).
    Low CV: a single global scale can explain the metric intervals.
    """
    s, _sig = interval_scales(prediction, constraints)
    return _std_over_median(s)


def robust_cv(s: np.ndarray) -> float:
    """MAD / |median| of implied interval scales s_k (GT-free)."""
    s = np.asarray(s, dtype=np.float64)
    s = s[np.isfinite(s)]
    if s.size < 2:
        return float("nan")
    med = float(np.median(s))
    if abs(med) < 1e-15:
        return float("nan")
    mad = float(np.median(np.abs(s - med)))
    return mad / abs(med)


def log_iqr_over_median(s: np.ndarray) -> float:
    """IQR(log s_k) / |median(log s_k)| (GT-free)."""
    s = np.asarray(s, dtype=np.float64)
    s = s[np.isfinite(s) & (s > 0)]
    if s.size < 2:
        return float("nan")
    log_s = np.log(s)
    med = float(np.median(log_s))
    if abs(med) < 1e-15:
        return float("nan")
    q75, q25 = np.percentile(log_s, [75.0, 25.0])
    return float((q75 - q25) / abs(med))


def local_scale_dispersion(
    prediction: PredictionBundle, constraints: ConstraintSet
) -> dict[str, float]:
    """GT-free local-scale dispersion of interval ratios s_k = b_k / a_k."""
    s, sig = interval_scales(prediction, constraints)
    return {
        "pairwise_scale_cv": _std_over_median(s),
        "robust_cv": robust_cv(s),
        "log_iqr_over_median": log_iqr_over_median(s),
        "n_intervals": float(s.size),
        "mean_gps_sigma": float(np.mean(sig)) if sig.size else float("nan"),
    }


def metric_observability_score(
    prediction: PredictionBundle,
    constraints: ConstraintSet,
    fisher_s: float | None = None,
    min_fisher: float = 1.0,
    min_path: float = 1e-3,
) -> ObservabilityReport:
    """MOS ∈ [0, 1] from Fisher information of scale and motion excitation.

    This is a *predicted* reliability, not a guarantee. Calibrate empirically.
    Status: HYPOTHESIS until real-data reliability diagrams exist.
    """
    obs = collect_scale_observations(prediction, constraints)
    a = np.array([o[1] for o in obs], dtype=np.float64) if obs else np.zeros(0)
    var = np.array([o[3] for o in obs], dtype=np.float64) if obs else np.zeros(0)
    if fisher_s is None:
        if len(obs) == 0:
            fisher_s = 0.0
        else:
            fisher_s = float(np.sum((a * a) / np.maximum(var, 1e-18)))
    mot = motion_excitation(prediction)
    reasons: list[str] = []
    if len(obs) == 0:
        reasons.append("no_scale_observations")
    if mot["path_length"] < min_path:
        reasons.append("insufficient_parallax")
    if fisher_s < min_fisher:
        reasons.append("weak_fisher")
    # Squash Fisher to (0,1): 1 - exp(-I / I0) with I0=50 as a nominal well-posed clip.
    info = 1.0 - np.exp(-max(fisher_s, 0.0) / 50.0)
    motion = 1.0 - np.exp(-mot["path_length"] / max(mot["bbox_span"] + mot["path_length"], 1e-6) * 4.0)
    n_term = 1.0 - np.exp(-len(obs) / 4.0)
    score = float(np.clip(info * 0.6 + motion * 0.2 + n_term * 0.2, 0.0, 1.0))
    if reasons:
        score *= 0.25
    std = float(np.sqrt(1.0 / max(fisher_s, 1e-18)))
    observable = score >= 0.45 and "no_scale_observations" not in reasons and "singular" not in reasons
    return ObservabilityReport(
        score=score,
        observable=observable,
        fisher_s=float(fisher_s),
        predicted_scale_std=std,
        n_observations=len(obs),
        path_length=mot["path_length"],
        mean_baseline=mot["mean_step"],
        reasons=reasons,
    )
