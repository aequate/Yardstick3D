"""Deterministic unit tests for three-limits scale-factor helpers.

Pure arithmetic only: handcrafted scalars, no GT, no I/O, no randomness.
"""

from __future__ import annotations

import math

import pytest

from yardstick3d.evaluation.three_limits import (
    aggregate_scale_observations,
    cue_discrepancy,
    cue_supported_scale,
    decompose_scale_error,
    estimator_departure,
    estimator_departure_log_error,
    estimator_departure_signed_log_error,
    path_vs_optimal_gap,
)


def test_identity_holds_exactly_on_paired_values():
    s_hat, s_cue, Lg, Lt, s_path, s_opt = 11.0, 12.0, 24.0, 26.0, 13.0, 10.0
    out = decompose_scale_error(s_hat, s_cue, Lg, Lt, s_path, s_opt)
    assert out["identity_exact"] is True
    assert out["total_scale_ratio"] == pytest.approx(
        math.prod(
            [out["estimator_departure"], out["cue_discrepancy"], out["path_vs_optimal_gap"]]
        )
    )
    assert out["factor_product"] == pytest.approx(out["total_scale_ratio"])
    assert abs(out["identity_residual"]) < 1e-12
    # Log-additive form.
    assert out["log_total"] == pytest.approx(
        out["log_estimator_departure"] + out["log_cue_discrepancy"] + out["log_path_vs_optimal_gap"]
    )


def test_estimator_departure_vs_total_scale_error():
    # Faithful transfer of a biased cue: departure is 1 while total error is not.
    assert estimator_departure(12.0, 12.0) == 1.0
    assert estimator_departure_log_error(12.0, 12.0) == 0.0
    assert estimator_departure_signed_log_error(12.0, 6.0) == math.log(2.0)
    out = decompose_scale_error(
        s_hat=12.0, s_cue=12.0, Lg=24.0, Lt=26.0, s_path=13.0, s_opt=10.0
    )
    assert out["estimator_departure"] == 1.0
    assert out["total_scale_ratio"] != 1.0  # cue + path-vs-optimal gaps remain


def test_individual_factors_and_logs():
    assert cue_supported_scale(24.0, 12.0) == 2.0
    assert cue_discrepancy(24.0, 26.0) == 24.0 / 26.0
    assert path_vs_optimal_gap(13.0, 10.0) == 1.3
    assert estimator_departure_log_error(11.0, 12.0) == abs(math.log(11.0 / 12.0))


def test_guards_return_none_on_invalid_inputs():
    for fn, args in [
        (cue_supported_scale, (0.0, 1.0)),
        (cue_supported_scale, (-2.0, 1.0)),
        (cue_supported_scale, (float("nan"), 1.0)),
        (cue_supported_scale, (1.0, float("inf"))),
        (estimator_departure, (1.0, 0.0)),
        (estimator_departure, (float("nan"), 1.0)),
        (cue_discrepancy, (1.0, -1.0)),
        (path_vs_optimal_gap, (1.0, 0.0)),
        (estimator_departure_log_error, (1.0, 0.0)),
        (estimator_departure_signed_log_error, (-1.0, 1.0)),
    ]:
        assert fn(*args) is None
    bad = decompose_scale_error(1.0, 0.0, 1.0, 1.0, 1.0, 1.0)
    assert bad["identity_exact"] is False
    assert bad["factor_product"] is None
    assert bad["identity_residual"] is None


def test_median_ratio_is_not_path_sum_ratio():
    # a = visual chords, b = cue lengths: all four aggregations differ.
    agg = aggregate_scale_observations([2.0, 4.0], [3.0, 3.0])
    assert agg["sum_ratio"] == 1.0  # 6/6
    assert agg["norm_ratio"] == math.sqrt(18.0) / math.sqrt(20.0)
    assert agg["median_ratio"] == 1.125  # median(1.5, 0.75)
    assert agg["ls_ratio"] == 18.0 / 20.0
    assert len({agg["sum_ratio"], agg["norm_ratio"], agg["median_ratio"], agg["ls_ratio"]}) == 4


def test_uniform_cue_scale_collapses_aggregations():
    agg = aggregate_scale_observations([2.0, 4.0, 6.0], [1.84, 3.68, 5.52])
    for key in ("sum_ratio", "norm_ratio", "median_ratio", "ls_ratio"):
        assert agg[key] == pytest.approx(agg["sum_ratio"])
        assert agg[key] == pytest.approx(0.92)


def test_aggregation_guards():
    assert aggregate_scale_observations([], [])["median_ratio"] is None
    assert aggregate_scale_observations([0.0, 0.0], [1.0, 2.0])["median_ratio"] is None
    assert aggregate_scale_observations([1.0], [1.0, 2.0])["sum_ratio"] is None
