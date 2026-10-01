from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from yardstick3d.datasets.advio import load_advio
from yardstick3d.datasets.advio_cues import constraints_from_corelocation_displacement
from yardstick3d.evaluation.leakage import assert_no_gt_in_constraints

FIXTURE = Path(__file__).parent / "fixtures" / "advio_mini"


def test_advio_fixture_loads_without_speed_column():
    seq = load_advio(FIXTURE, "mini", imu_allowed=False)
    assert seq.inference.speed_column_present is False
    assert seq.inference.speed_mps is None
    assert seq.inference.lat.shape[0] == 4
    assert seq.evaluation.centers.shape == (4, 3)


def test_imu_not_loaded():
    seq = load_advio(FIXTURE, "mini", imu_allowed=False)
    seq.inference.assert_imu_denied()
    assert seq.inference.notes["imu_loaded"] is False


def test_cannot_set_imu_notes():
    seq = load_advio(FIXTURE, "mini", imu_allowed=False)
    seq.inference.notes["gyro"] = "nope"
    with pytest.raises(RuntimeError):
        seq.inference.assert_imu_denied()


def test_cues_do_not_see_gt():
    seq = load_advio(FIXTURE, "mini", imu_allowed=False)
    cs = constraints_from_corelocation_displacement(seq.inference, seq.inference.frame_timestamps)
    assert len(cs) >= 1
    assert_no_gt_in_constraints(cs)
    for c in cs:
        assert "gt" not in c.metadata()["name"]


def test_invalid_speed_not_zero():
    from yardstick3d.sensors.speed_integrate import integrate_speed, invalid_speed_mask

    t = np.array([0.0, 1.0, 2.0])
    v = np.array([-1.0, np.nan, -1.0])
    assert invalid_speed_mask(v).all()
    assert integrate_speed(0.0, 2.0, t, v) is None


def _synth_inf(t_gps: np.ndarray, lat: np.ndarray, lon: np.ndarray):
    from yardstick3d.datasets.advio import InferenceChannels

    n = len(t_gps)
    return InferenceChannels(
        sequence_id="synth",
        frame_timestamps=np.array([0.0, 2.0]),
        frame_indices=np.array([1.0, 2.0]),
        video_path=None,
        location_t=t_gps,
        lat=lat,
        lon=lon,
        hacc_m=np.full(n, 5.0),
        alt_m=np.full(n, 10.0),
        vacc_m=np.full(n, 6.0),
        speed_mps=None,
        speed_column_present=False,
        baro_t=np.array([0.0]),
        baro_pressure=np.array([101.0]),
        baro_rel_alt_m=np.array([0.0]),
        imu_allowed=False,
    )


def test_corelocation_interval_includes_straddling_segments():
    from yardstick3d.datasets.advio import corelocation_interval_distance
    from yardstick3d.sensors.geodesy import geodetic_to_enu

    t_gps = np.array([0.5, 1.5, 2.5, 3.5])
    lat0, lon0 = 60.18680, 24.82200
    lat = np.full(4, lat0)
    lon = lon0 + np.array([0.0, 1e-5, 2e-5, 3e-5])
    inf = _synth_inf(t_gps, lat, lon)
    d = corelocation_interval_distance(inf, 0.0, 2.0)
    assert d is not None
    xyz = geodetic_to_enu(inf.lat, inf.lon, inf.alt_m)
    p0 = np.array([np.interp(0.0, t_gps, xyz[:, k]) for k in range(3)])
    p1 = np.array([np.interp(2.0, t_gps, xyz[:, k]) for k in range(3)])
    expected = float(
        np.linalg.norm(xyz[0] - p0) + np.linalg.norm(xyz[1] - xyz[0]) + np.linalg.norm(p1 - xyz[1])
    )
    assert d == pytest.approx(expected, rel=1e-9)
    interior_only = float(np.linalg.norm(xyz[1] - xyz[0]))
    assert d > interior_only + 0.05


def test_corelocation_interval_aligned_samples_unchanged():
    from yardstick3d.datasets.advio import corelocation_interval_distance
    from yardstick3d.sensors.geodesy import geodetic_to_enu

    seq = load_advio(FIXTURE, "mini", imu_allowed=False)
    d = corelocation_interval_distance(seq.inference, 0.0, 1.0)
    xyz = geodetic_to_enu(seq.inference.lat, seq.inference.lon, seq.inference.alt_m)
    assert d == pytest.approx(float(np.linalg.norm(xyz[1] - xyz[0])), rel=1e-9)
