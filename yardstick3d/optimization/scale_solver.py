from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from yardstick3d.constraints.base import ConstraintSet
from yardstick3d.types import (
    ConstraintValidity,
    MetricState,
    PredictionBundle,
    SolverDiagnostics,
)
from yardstick3d.uncertainty.robust import robust_weight


@dataclass
class ScaleSolveResult:
    scale: float
    scale_var: float
    diagnostics: SolverDiagnostics
    observations: list[tuple[str, float, float, float]]


def naive_scale_ratio(a: np.ndarray, b: np.ndarray) -> float:
    """Path-length / TLS-style ratio (VI3 α0 / Eq. 7 special case)."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na = float(np.linalg.norm(a))
    if na < 1e-15:
        return float("nan")
    return float(np.sign(np.dot(a, b)) * np.linalg.norm(b) / na)


def collect_scale_observations(
    prediction: PredictionBundle,
    constraints: ConstraintSet,
) -> list[tuple[str, float, float, float]]:
    """List of (name, a, b, variance) for s * a ≈ b."""
    state = MetricState()
    obs: list[tuple[str, float, float, float]] = []
    for c in constraints.constraints:
        if c.validity(prediction, state) != ConstraintValidity.VALID:
            continue
        so = c.scale_observation(prediction)
        if so is None:
            continue
        a, b, var = so
        if not np.isfinite(a) or not np.isfinite(b) or var <= 0:
            continue
        obs.append((c.name, float(a), float(b), float(var)))
    return obs


def solve_global_scale(
    prediction: PredictionBundle,
    constraints: ConstraintSet,
    kernel: str = "huber",
    huber_delta: float = 1.345,
    max_iters: int = 25,
    min_obs: int = 1,
    reject_z: float = 6.0,
) -> ScaleSolveResult:
    """Weighted IRLS for min_s Σ w_k (s a_k - b_k)^2 / σ_k^2, s > 0.

    Closed form per iteration:
        s = Σ w a b / σ²  /  Σ w a² / σ²
    Hessian H = 2 Σ w a² / σ²,  Var(s) ≈ 1 / Σ w a² / σ²
    """
    obs = collect_scale_observations(prediction, constraints)
    diag = SolverDiagnostics()
    if len(obs) < min_obs:
        diag.message = "insufficient_observations"
        diag.n_rejected = len(constraints.constraints)
        return ScaleSolveResult(scale=float("nan"), scale_var=float("inf"), diagnostics=diag, observations=obs)

    a = np.array([o[1] for o in obs], dtype=np.float64)
    b = np.array([o[2] for o in obs], dtype=np.float64)
    var = np.array([o[3] for o in obs], dtype=np.float64)
    names = [o[0] for o in obs]

    inv_var = 1.0 / np.maximum(var, 1e-18)
    w = np.ones_like(a)
    ratios = np.divide(b, a, out=np.full_like(a, np.nan), where=np.abs(a) > 1e-15)
    finite_ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
    if finite_ratios.size:
        s = float(np.median(finite_ratios))
    else:
        s = naive_scale_ratio(a * np.sqrt(inv_var), b * np.sqrt(inv_var))
    if not np.isfinite(s) or s <= 0:
        s = 1.0

    cost_hist: list[float] = []
    accepted = np.ones(len(a), dtype=bool)
    for it in range(max_iters):
        r = s * a - b
        r_std = r / np.sqrt(np.maximum(var, 1e-18))
        w = robust_weight(r_std, kernel=kernel, delta=huber_delta)
        # Hard gating only after IRLS has moved toward the inlier mode.
        accepted = np.ones(len(a), dtype=bool) if it < 4 else (np.abs(r_std) < reject_z)
        ww = w * inv_var * accepted.astype(np.float64)
        denom = float(np.sum(ww * a * a))
        numer = float(np.sum(ww * a * b))
        if denom < 1e-18:
            diag.message = "singular_hessian"
            diag.n_iterations = it + 1
            diag.cost_history = cost_hist
            return ScaleSolveResult(scale=float("nan"), scale_var=float("inf"), diagnostics=diag, observations=obs)
        s_new = numer / denom
        if s_new <= 0:
            s_new = abs(s_new) if abs(s_new) > 1e-12 else s
        cost = float(np.sum(ww * (s_new * a - b) ** 2))
        cost_hist.append(cost)
        if abs(s_new - s) / max(abs(s), 1e-9) < 1e-8:
            s = s_new
            diag.converged = True
            diag.n_iterations = it + 1
            break
        s = s_new
        diag.n_iterations = it + 1
    else:
        diag.n_iterations = max_iters

    r = s * a - b
    r_std = r / np.sqrt(np.maximum(var, 1e-18))
    ww = w * inv_var * accepted.astype(np.float64)
    fisher = float(np.sum(ww * a * a))
    scale_var = 1.0 / max(fisher, 1e-18)
    diag.cost = cost_hist[-1] if cost_hist else float("nan")
    diag.cost_history = cost_hist
    diag.hessian_scale = fisher
    diag.condition_number = fisher
    diag.n_accepted = int(np.sum(accepted))
    diag.n_rejected = int(np.sum(~accepted)) + (len(constraints.constraints) - len(obs))
    breakdown: dict[str, float] = {}
    for name, rr in zip(names, r_std):
        breakdown[name] = breakdown.get(name, 0.0) + float(rr**2)
    n_per: dict[str, int] = {}
    for name in names:
        n_per[name] = n_per.get(name, 0) + 1
    diag.residual_breakdown = {k: v / max(n_per[k], 1) for k, v in breakdown.items()}
    diag.message = "ok" if diag.converged else "max_iters"
    return ScaleSolveResult(scale=float(s), scale_var=float(scale_var), diagnostics=diag, observations=obs)
