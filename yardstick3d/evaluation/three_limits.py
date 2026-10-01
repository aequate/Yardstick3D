"""Three-limits scale-factor decomposition (evaluation-only, pure arithmetic).

Implements the exact multiplicative identity:

    s_hat / s_opt = (s_hat / s_cue) * (Lg / Lt) * (s_path / s_opt)

All functions are pure arithmetic on caller-supplied scalars. They import
nothing GT-derived, touch no caches, and perform no I/O, so they cannot
leak ground truth into a solver. Invalid inputs (non-finite, non-positive)
yield ``None`` so JSON serializers cannot accidentally emit infinities as
evidence -- the same guard policy as
:func:`yardstick3d.evaluation.taxonomy.grounding_taxonomy`.

These helpers separate *scale factors only*. They do not decompose ATE,
which has no additive decomposition.
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping


def _is_positive_finite(x: object) -> bool:
    try:
        v = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return math.isfinite(v) and v > 0.0


def cue_supported_scale(Lg: float, Lv: float) -> float | None:
    """Naive cue-transfer scale ``s_cue = Lg / Lv``.

    ``Lg`` is the integrated cue path (e.g. ``cue_path_m``); ``Lv`` is the
    predicted sampled path (e.g. ``path_pred``). Returns ``None`` unless
    both are positive and finite.
    """
    if not (_is_positive_finite(Lg) and _is_positive_finite(Lv)):
        return None
    return float(Lg) / float(Lv)


def estimator_departure(s_hat: float, s_cue: float) -> float | None:
    """Solver aggregation departure from naive cue transfer: ``s_hat / s_cue``.

    ``s_hat`` is the solver's scalar output; ``s_cue`` is
    :func:`cue_supported_scale`. ``None`` unless both are positive finite.
    """
    if not (_is_positive_finite(s_hat) and _is_positive_finite(s_cue)):
        return None
    return float(s_hat) / float(s_cue)


def estimator_departure_signed_log_error(s_hat: float, s_cue: float) -> float | None:
    """Signed log estimator departure ``log(s_hat / s_cue)``; ``None`` if invalid."""
    ratio = estimator_departure(s_hat, s_cue)
    if ratio is None:
        return None
    return math.log(ratio)


def estimator_departure_log_error(s_hat: float, s_cue: float) -> float | None:
    """Absolute log estimator departure ``|log(s_hat / s_cue)|``; ``None`` if invalid."""
    signed = estimator_departure_signed_log_error(s_hat, s_cue)
    if signed is None:
        return None
    return abs(signed)


def cue_discrepancy(Lg: float, Lt: float) -> float | None:
    """Cue/path-sampling discrepancy ``Lg / Lt``; ``None`` unless both positive finite."""
    if not (_is_positive_finite(Lg) and _is_positive_finite(Lt)):
        return None
    return float(Lg) / float(Lt)


def path_vs_optimal_gap(s_path: float, s_opt: float) -> float | None:
    """Path-oracle versus ATE-optimal scale ``s_path / s_opt``; ``None`` if invalid."""
    if not (_is_positive_finite(s_path) and _is_positive_finite(s_opt)):
        return None
    return float(s_path) / float(s_opt)


def decompose_scale_error(
    s_hat: float,
    s_cue: float,
    Lg: float,
    Lt: float,
    s_path: float,
    s_opt: float,
) -> dict:
    """Evaluate the exact identity ``s_hat/s_opt = departure * cue * gap``.

    Returns the three factors, the direct total ``s_hat / s_opt``, the
    factor product, the four signed logs (additive form), and the
    multiplicative residual ``product / total - 1`` (zero up to roundoff
    when all inputs are valid). Any invalid input nulls every field except
    ``identity_exact``, which is then ``False``.
    """
    departure = estimator_departure(s_hat, s_cue)
    cue = cue_discrepancy(Lg, Lt)
    gap = path_vs_optimal_gap(s_path, s_opt)
    total = estimator_departure(s_hat, s_opt)  # same ratio form: s_hat / s_opt
    fields = (departure, cue, gap, total)
    if any(f is None for f in fields):
        return {
            "estimator_departure": departure,
            "cue_discrepancy": cue,
            "path_vs_optimal_gap": gap,
            "total_scale_ratio": total,
            "factor_product": None,
            "log_total": None,
            "log_estimator_departure": None,
            "log_cue_discrepancy": None,
            "log_path_vs_optimal_gap": None,
            "identity_residual": None,
            "identity_exact": False,
        }
    assert departure is not None and cue is not None and gap is not None and total is not None
    product = departure * cue * gap
    return {
        "estimator_departure": departure,
        "cue_discrepancy": cue,
        "path_vs_optimal_gap": gap,
        "total_scale_ratio": total,
        "factor_product": product,
        "log_total": math.log(total),
        "log_estimator_departure": math.log(departure),
        "log_cue_discrepancy": math.log(cue),
        "log_path_vs_optimal_gap": math.log(gap),
        "identity_residual": product / total - 1.0,
        "identity_exact": True,
    }


def aggregate_scale_observations(
    a: Iterable[float], b: Iterable[float]
) -> Mapping[str, float | None]:
    """Four deterministic aggregations of per-interval ``s * a_k ~= b_k`` pairs.

    - ``sum_ratio``: ``sum(b) / sum(a)`` -- the path-sum (cue-transfer) ratio
      ``s_cue = Lg / Lv`` when ``a``/``b`` are the interval visual/cue lengths.
    - ``norm_ratio``: ``||b||_2 / ||a||_2`` -- the frozen ADVIO ``naive``
      solver (``naive_scale_ratio``). For positive lengths the sign factor is +1.
    - ``median_ratio``: ``median(b_k / a_k)`` over intervals with ``a_k > 0``
      and ``b_k / a_k > 0`` -- the Mobile-GVIO prospective ``b0_scale`` and
      the robust-solver starting point. ``None`` when no valid interval exists.
    - ``ls_ratio``: ``sum(a b) / sum(a^2)`` -- unweighted least squares.

    A uniform cue scale (all ``b_k = alpha * a_k``) makes all four equal
    ``alpha``; otherwise they differ in general. In particular a median of
    interval ratios is not a path-sum ratio. Non-finite entries are skipped;
    empty input yields ``None`` fields.
    """
    aval = [float(v) for v in list(a)]
    bval = [float(v) for v in list(b)]
    if len(aval) != len(bval) or not aval:
        return {"sum_ratio": None, "norm_ratio": None, "median_ratio": None, "ls_ratio": None}
    pairs = [(x, y) for x, y in zip(aval, bval) if math.isfinite(x) and math.isfinite(y)]
    if not pairs:
        return {"sum_ratio": None, "norm_ratio": None, "median_ratio": None, "ls_ratio": None}
    xs = [x for x, _ in pairs]
    ys = [y for _, y in pairs]
    sum_a = sum(xs)
    sum_b = sum(ys)
    sum_ratio = sum_b / sum_a if sum_a > 0 and math.isfinite(sum_b) else None
    na = math.sqrt(sum(x * x for x in xs))
    nb = math.sqrt(sum(y * y for y in ys))
    norm_ratio = nb / na if na > 1e-15 and math.isfinite(nb) else None
    ratios = sorted(y / x for x, y in pairs if x > 1e-15 and (y / x) > 0 and math.isfinite(y / x))
    if not ratios:
        median_ratio = None
    elif len(ratios) % 2:
        median_ratio = ratios[len(ratios) // 2]
    else:
        median_ratio = 0.5 * (ratios[len(ratios) // 2 - 1] + ratios[len(ratios) // 2])
    denom = sum(x * x for x in xs)
    ls_ratio = sum(x * y for x, y in pairs) / denom if denom > 1e-18 else None
    return {
        "sum_ratio": sum_ratio,
        "norm_ratio": norm_ratio,
        "median_ratio": median_ratio,
        "ls_ratio": ls_ratio,
    }
