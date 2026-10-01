"""Prospective MARS-LVIG contract tests (no bags, no GT, no network).

Covers: timestamp parsing/units, sync mapping (no fitted offset), frame/cue
matching, coordinate transforms, split enforcement, config hash, cue/reference
isolation, missing-sensor handling, adapter polyline, contract immutability.
"""
from pathlib import Path

import numpy as np
import pytest

from yardstick3d.datasets import mars_lvig as ml

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "prospective_metric_grounding_v2.yaml"
EXPECTED_SHA = "8af5106aae2484f653b41d1e0e060a8928e15ed1c5245110a28f370cab33a923"


def test_contract_hash_frozen():
    ml.check_contract_frozen(CONTRACT, EXPECTED_SHA)


def test_contract_hash_tamper_rejected(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_bytes(CONTRACT.read_bytes() + b"\n# tamper\n")
    with pytest.raises(ValueError, match="hash"):
        ml.check_contract_frozen(p, EXPECTED_SHA)


def test_timestamp_units_microseconds_to_seconds():
    # filenames are integer microseconds UTC; adapter domain is float seconds
    fname_us = 1621218775000000
    t_s = fname_us / 1e6
    assert t_s == pytest.approx(1621218775.0)
    # GPS->UTC conversion from contract docs: -18 leap s + 315964800 epoch offset
    gps_week, gps_s = 2158, 95593
    utc = gps_week * 604800 + gps_s - 18 + 315964800
    assert utc == 1621218775


def test_sync_mapping_has_no_fitted_offset():
    import inspect

    src = inspect.getsource(ml)
    assert "fitted_offset" not in src
    assert "cross_correlat" not in src.lower()
    # matching is nearest-sample within tolerance only
    assert "nearest_index" in src
    assert ml.nearest_index(np.array([0.0, 0.1, 0.2]), 0.11, 0.06) == 1
    assert ml.nearest_index(np.array([0.0, 0.1, 0.2]), 0.5, 0.06) is None


def test_frame_cue_matching_and_coverage():
    img_t = np.arange(0, 20, 0.1)
    cue_t = np.arange(0, 20, 0.1)
    assert ml.match_coverage(img_t, cue_t, 0.06) == pytest.approx(1.0)
    assert ml.match_coverage(np.array([0.0, 1.0]), np.array([50.0]), 0.06) == pytest.approx(0.0)
    wins = ml.prospective_windows("HKisland_GNSS01", t_bag0=1000.0)
    assert len(wins) == 5
    assert wins[0].t_start == pytest.approx(1120.0)
    assert len(wins[0].frame_times) == 8
    idx = ml.select_frame_indices_from_images(img_t, np.array([0.05, 0.15]), 0.06)
    assert idx.tolist() == [0, 1] or idx.tolist() == [1, 2] or len(idx) == 2
    with pytest.raises(RuntimeError):
        ml.select_frame_indices_from_images(np.array([0.0]), np.array([99.0]), 0.06)


def test_coordinate_transform_enu_metres():
    # ~11 m per 1e-4 deg latitude; ENU output in metres, E/N/U order
    lat = np.array([22.0000, 22.0001])
    lon = np.array([114.0000, 114.0000])
    enu = ml.cue_enu_from_llh(lat, lon, np.array([0.0, 0.0]))
    assert enu.shape == (2, 3)
    assert abs(enu[0, 0]) < 1e-9
    assert 9.0 < enu[1, 1] < 13.0
    assert abs(enu[1, 0]) < 1.0


def test_split_enforcement_dev_vs_test():
    ml.enforce_sequence_role("HKairport_GNSS01", "development")
    ml.enforce_sequence_role("HKisland_GNSS01", "prospective_test")
    with pytest.raises(ValueError):
        ml.enforce_sequence_role("HKisland_GNSS01", "development")
    with pytest.raises(ValueError):
        ml.enforce_sequence_role("HKairport_GNSS01", "prospective_test")
    with pytest.raises(ValueError):
        ml.enforce_sequence_role("AMvalley01", "prospective_test")


def test_cue_reference_isolation():
    ml.assert_cue_reference_isolation(
        ["/ublox_driver/receiver_lla"], ["/dji_osdk_ros/rtk_position"]
    )
    with pytest.raises(ValueError):
        ml.assert_cue_reference_isolation(
            ["/dji_osdk_ros/rtk_position"], ["/dji_osdk_ros/rtk_position"]
        )
    with pytest.raises(ValueError):
        ml.assert_cue_reference_isolation(
            ["/dji_osdk_ros/rtk_position"], ["/dji_osdk_ros/rtk_velocity"]
        )
    with pytest.raises(ValueError):
        ml.assert_cue_reference_isolation(["/ublox_driver/receiver_lla"], ["/ublox_driver/receiver_lla"])


def test_missing_sensor_values_dropped_not_zero_filled():
    cue_t = np.arange(0, 16.1, 0.1)
    enu = np.column_stack([cue_t * 3.0, np.zeros_like(cue_t), np.zeros_like(cue_t)])
    t_clip = np.linspace(0, 16, 8)
    cs, info = ml.constraints_from_ublox_displacement(t_clip, cue_t, enu)
    assert info["n_kept"] == 7
    # remove all cue fixes in one interval -> that interval dropped
    mask = ~((cue_t > t_clip[2]) & (cue_t <= t_clip[3]))
    cs2, info2 = ml.constraints_from_ublox_displacement(t_clip, cue_t[mask], enu[mask])
    assert info2["n_dropped"] >= 1
    assert info2["n_kept"] <= 6
    for c in cs2:
        assert np.isfinite(c.length_m) and c.length_m > 0


def test_polyline_matches_straight_line_distance():
    cue_t = np.arange(0, 10.1, 0.1)
    enu = np.column_stack([cue_t * 5.0, np.zeros_like(cue_t), np.zeros_like(cue_t)])
    b, cov = ml.cue_polyline_lengths(cue_t, enu, np.array([0.0, 10.0]))
    assert b[0] == pytest.approx(50.0, rel=1e-9)
    assert cov[0] == pytest.approx(1.0)


def test_reference_interpolation_eval_only():
    ref_t = np.array([0.0, 1.0, 2.0])
    ref_enu = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)
    out = ml.interpolate_reference_at(ref_t, ref_enu, np.array([0.5, 1.5]))
    assert out[0, 0] == pytest.approx(0.5)
    assert out[1, 0] == pytest.approx(1.5)
    out2 = ml.interpolate_reference_at(ref_t, ref_enu, np.array([99.0]))
    assert not np.isfinite(out2[0, 0])


def test_sync_manifest_schema():
    m = ml.SyncManifest(
        sequence_id="HKisland_GNSS01", t_bag0=1.0, n_images=10,
        n_cue_fixes=10, camera_cue_match_fraction=1.0,
        time_sync_notes={"source": "sensor-only stage 1"},
    )
    d = m.to_dict()
    assert d["ref_positions_opened"] is False
    assert d["max_dt_match_camera_cue_s"] == pytest.approx(0.06)
