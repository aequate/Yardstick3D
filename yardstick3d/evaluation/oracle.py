"""Evaluation-only scale. Must never be imported by constraint builders or the grounder."""

from __future__ import annotations

import numpy as np

from yardstick3d.evaluation.scale import relative_scale_error, scale_error
from yardstick3d.geometry.trajectory import path_length
from yardstick3d.types import FloatArray


def oracle_scale_path_length(pred_centers: FloatArray, gt_centers: FloatArray) -> float:
    lp = path_length(pred_centers)
    lg = path_length(gt_centers)
    if lp < 1e-12:
        return float("nan")
    return float(lg / lp)


def oracle_report(s_hat: float, s_oracle: float) -> dict[str, float]:
    return {
        "oracle_scale": float(s_oracle) if np.isfinite(s_oracle) else float("nan"),
        "hat_over_oracle": float(s_hat / s_oracle) if np.isfinite(s_hat) and np.isfinite(s_oracle) and s_oracle else float("nan"),
        "scale_error_log_vs_oracle": scale_error(s_hat, s_oracle),
        "scale_error_rel_vs_oracle": relative_scale_error(s_hat, s_oracle),
    }
