"""Evaluation-only diagnostics. None of these quantities are deployment features."""
from __future__ import annotations

import math
from collections.abc import Sequence


def grounding_taxonomy(raw_ate: float, grounded_ate: float, oracle_ate: float,
                       scale: float, oracle_scale: float, *, epsilon: float = 1e-12,
                       minimum_leverage: float | None = None) -> dict:
    """Separate geometry leverage from solver capture; never clip bad outcomes.

    `minimum_leverage` is an explicitly supplied analysis policy, not a fitted
    groundability threshold. Without it capture is algebraically reported but
    its substantive interpretation is not authorized. Invalid denominators
    return None so JSON cannot accidentally serialize infinities as evidence.
    """
    if epsilon <= 0 or not math.isfinite(epsilon):
        raise ValueError("epsilon must be finite and positive")
    if minimum_leverage is not None and not 0 < minimum_leverage <= 1:
        raise ValueError("minimum_leverage must be in (0, 1]")
    if any(not math.isfinite(x) or x < 0 for x in (raw_ate, grounded_ate, oracle_ate)):
        raise ValueError("ATEs must be finite and nonnegative")
    gap = raw_ate - oracle_ate
    improvement = raw_ate - grounded_ate
    leverage = gap / (raw_ate + epsilon) if raw_ate > epsilon else None
    capture = improvement / (gap + epsilon) if gap > epsilon else None
    valid_scale = all(math.isfinite(x) and x > 0 for x in (scale, oracle_scale))
    ratio = scale / oracle_scale if valid_scale else None
    return {
        "raw_ate": raw_ate, "grounded_ate": grounded_ate, "oracle_ate": oracle_ate,
        "oracle_scale_leverage": leverage,
        "solver_ceiling_capture": capture,
        "solver_ceiling_capture_interpretable": bool(capture is not None and
            minimum_leverage is not None and leverage >= minimum_leverage),
        "minimum_leverage_policy": minimum_leverage,
        "scale_relative_error": abs(ratio - 1) if ratio is not None else None,
        "scale_log_error": abs(math.log(ratio)) if ratio is not None else None,
        "scale_signed_log_error": math.log(ratio) if ratio is not None else None,
        "scale_ratio": ratio,
        "absolute_grounded_improvement": improvement,
    }


def temporal_overlap_pairs(windows: Sequence[dict]) -> list[tuple[int, int]]:
    """Positive-duration overlap; a shared endpoint is not an overlapping clip."""
    return [(i, j) for i, a in enumerate(windows) for j, b in enumerate(windows)
            if i < j and max(a["t0"], b["t0"]) < min(a["t1"], b["t1"])]
