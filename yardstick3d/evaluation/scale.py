from __future__ import annotations

import numpy as np


def scale_error(s_hat: float, s_true: float) -> float:
    """E_s = |log(s_hat / s_true)|."""
    if not np.isfinite(s_hat) or not np.isfinite(s_true) or s_hat <= 0 or s_true <= 0:
        return float("nan")
    return float(abs(np.log(s_hat / s_true)))


def relative_scale_error(s_hat: float, s_true: float) -> float:
    if not np.isfinite(s_hat) or not np.isfinite(s_true) or s_true == 0:
        return float("nan")
    return float(abs(s_hat - s_true) / abs(s_true))
