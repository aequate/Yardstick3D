import pytest
from yardstick3d.evaluation.taxonomy import grounding_taxonomy, temporal_overlap_pairs


def test_scale_limited_exact_recovery():
    r = grounding_taxonomy(10, 1, 1, 2, 2, minimum_leverage=.5)
    assert r["oracle_scale_leverage"] == pytest.approx(.9)
    assert r["solver_ceiling_capture"] == pytest.approx(1)
    assert r["solver_ceiling_capture_interpretable"]
    assert r["scale_log_error"] == 0
    assert r["absolute_grounded_improvement"] == 9


def test_high_capture_low_leverage_is_not_success():
    r = grounding_taxonomy(10, 9.224, 9.2, 1, 2, minimum_leverage=.5)
    assert r["solver_ceiling_capture"] == pytest.approx(.97)
    assert r["oracle_scale_leverage"] == pytest.approx(.08)
    assert not r["solver_ceiling_capture_interpretable"]


@pytest.mark.parametrize("raw,oracle", [(0, 0), (1, 1), (1, 2)])
def test_vacuous_capture(raw, oracle):
    assert grounding_taxonomy(raw, 1, oracle, 1, 1)["solver_ceiling_capture"] is None


def test_failures_not_clipped_and_policy_not_invented():
    r = grounding_taxonomy(2, 3, 1, -1, 2)
    assert r["solver_ceiling_capture"] < 0
    assert r["scale_log_error"] is None
    assert not r["solver_ceiling_capture_interpretable"]
    assert grounding_taxonomy(1, 1, 2, 1, 1)["oracle_scale_leverage"] < 0


def test_dependence_overlap_audit():
    assert temporal_overlap_pairs([{"t0": 0, "t1": 16}, {"t0": 16, "t1": 32}]) == []
    assert temporal_overlap_pairs([{"t0": 0, "t1": 16}, {"t0": 8, "t1": 24}]) == [(0, 1)]


def test_leverage_does_not_certify_absolute_shape_quality():
    r = grounding_taxonomy(1000, 100, 100, 1, 1)
    assert r["oracle_scale_leverage"] == pytest.approx(.9)
    assert r["solver_ceiling_capture"] == pytest.approx(1)
    assert r["oracle_ate"] == 100  # Still a 100 m geometry floor.


def test_ate_optimal_oracle_differs_from_path_oracle_and_preserves_gauge():
    import numpy as np
    from yardstick3d.types import PredictionBundle
    from yardstick3d.experiments.advio_real import evaluate_grounding
    gt = np.array([[0,0,0], [1,0,0], [2,0,0], [3,0,0]], dtype=float)
    centers = np.array([[0,0,0], [1,2,0], [2,-2,0], [3,0,0]], dtype=float)
    def evaluate(multiplier):
        return evaluate_grounding(PredictionBundle(timestamps=np.arange(4),
            T_w2c=np.zeros((4,3,4)), camera_centers=centers*multiplier), gt, 1/multiplier)
    a,b = evaluate(1),evaluate(10)
    assert a['oracle_ate_optimal'] < a['oracle_ate_se3']
    assert a['oracle_ate_optimal'] == pytest.approx(b['oracle_ate_optimal'])
    assert a['taxonomy']['oracle_scale_leverage'] != pytest.approx(b['taxonomy']['oracle_scale_leverage'])
