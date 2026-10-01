"""Counterexamples to interpreting cue agreement as universal shape identification."""
import numpy as np
from yardstick3d.evaluation.alignment import umeyama, apply_sim3, ate_rmse


def test_sparse_positions_do_not_identify_between_cue_shape():
    t = np.linspace(0, 4, 17)
    reference = np.column_stack([t, np.zeros_like(t), np.zeros_like(t)])
    distorted = reference.copy()
    distorted[:, 1] = 2 * np.sin(np.pi * t)
    cue_indices = np.arange(0, 17, 4)
    np.testing.assert_allclose(distorted[cue_indices], reference[cue_indices], atol=1e-14)
    s, r, tr = umeyama(distorted, reference, True)
    assert ate_rmse(apply_sim3(distorted, s, r, tr), reference) > 0.5


def test_raw_gauge_can_raise_leverage_without_improving_shape():
    t = np.linspace(0, 4, 17)
    gt = np.column_stack([t, np.zeros_like(t), np.zeros_like(t)])
    pred = np.column_stack([t, np.sin(np.pi * t), np.zeros_like(t)])
    leverage, floors = [], []
    for multiplier in [1., 1000.]:
        p = multiplier * pred
        s, r, tr = umeyama(p, gt, True)
        floor = ate_rmse(apply_sim3(p, s, r, tr), gt)
        _, rr, tt = umeyama(p, gt, False)
        raw = ate_rmse(apply_sim3(p, 1., rr, tt), gt)
        leverage.append((raw - floor) / raw)
        floors.append(floor)
    np.testing.assert_allclose(floors[0], floors[1], rtol=1e-10)
    assert floors[0] > 0.5
    assert leverage[1] > .99
    assert leverage[1] > leverage[0] + .5
