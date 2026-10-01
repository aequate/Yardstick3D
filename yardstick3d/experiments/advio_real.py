from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

from yardstick3d.evaluation.alignment import apply_se3, apply_sim3, ate_rmse, umeyama
from yardstick3d.evaluation.leakage import assert_no_gt_in_constraints
from yardstick3d.evaluation.oracle import oracle_report, oracle_scale_path_length
from yardstick3d.evaluation.scale import relative_scale_error, scale_error
from yardstick3d.evaluation.trajectory import rpe_translation
from yardstick3d.evaluation.taxonomy import grounding_taxonomy
from yardstick3d.geometry.trajectory import path_length
from yardstick3d.optimization.grounder import MetricGrounder
from yardstick3d.types import PredictionBundle


def config_hash(cfg: dict[str, Any]) -> str:
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def evaluate_grounding(
    pred: PredictionBundle,
    gt_centers: np.ndarray,
    result_scale: float,
) -> dict[str, Any]:
    c_pred = pred.centers()
    s_or = oracle_scale_path_length(c_pred, gt_centers)
    raw = c_pred
    grounded = c_pred * result_scale if np.isfinite(result_scale) else raw * np.nan
    oracle_c = c_pred * s_or if np.isfinite(s_or) else raw * np.nan

    def pack(name: str, pts: np.ndarray) -> dict[str, float]:
        if not np.all(np.isfinite(pts)):
            return {f"{name}_ate_se3": float("nan"), f"{name}_ate_sim3": float("nan"), f"{name}_ate_none": float("nan")}
        s1, R1, t1 = umeyama(pts, gt_centers, with_scale=False)
        s2, R2, t2 = umeyama(pts, gt_centers, with_scale=True)
        return {
            f"{name}_ate_none": ate_rmse(pts, gt_centers),
            f"{name}_ate_se3": ate_rmse(apply_se3(pts, R1, t1), gt_centers),
            f"{name}_ate_sim3": ate_rmse(apply_sim3(pts, s2, R2, t2), gt_centers),
            f"{name}_rpe": rpe_translation(apply_se3(pts, R1, t1), gt_centers),
            f"{name}_umeyama_s_se3": s1,
            f"{name}_umeyama_s_sim3": s2,
            f"{name}_path": path_length(pts),
        }

    out = {"oracle_scale": s_or, "path_gt": path_length(gt_centers), "path_pred": path_length(c_pred)}
    out.update(pack("raw", raw))
    out.update(pack("grounded", grounded))
    out.update(pack("oracle", oracle_c))
    out.update(oracle_report(result_scale, s_or))
    # Preserve historical path oracle fields. The ATE ceiling requires the
    # ATE-minimizing similarity scale, not the GT/predicted path-length ratio.
    optimal_scale, _, _ = umeyama(c_pred, gt_centers, with_scale=True)
    out["oracle_scale_ate_optimal"] = optimal_scale
    out["oracle_ate_optimal"] = out["raw_ate_sim3"]
    out["oracle_scale_definition_legacy"] = "GT/predicted path-length ratio"
    if np.isfinite(result_scale):
        out["taxonomy"] = grounding_taxonomy(
            out["raw_ate_se3"], out["grounded_ate_se3"], out["raw_ate_sim3"],
            result_scale, optimal_scale)
    if np.isfinite(result_scale) and np.isfinite(s_or):
        out["scale_error_log"] = scale_error(result_scale, s_or)
        out["scale_error_rel"] = relative_scale_error(result_scale, s_or)
    return out


def run_solvers(pred: PredictionBundle, constraints) -> dict[str, Any]:
    assert_no_gt_in_constraints(constraints)
    g = MetricGrounder()
    rows = {}
    for name in ("none", "naive", "ls", "robust"):
        r = g.solve(pred, constraints, solver=name)  # type: ignore[arg-type]
        rows[name] = {
            "scale": r.scale,
            "scale_std": r.scale_std,
            "observable": bool(r.observable),
            "observability_score": r.observability_score,
            "accepted": r.accepted_constraints,
            "rejected": r.rejected_constraints,
            "message": r.diagnostics.message,
        }
    return rows
