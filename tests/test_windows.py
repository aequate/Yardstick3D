from __future__ import annotations

from pathlib import Path

import numpy as np

from yardstick3d.datasets.advio import InferenceChannels, corelocation_interval_distance, load_advio
from yardstick3d.datasets.windows import FROZEN_V1, WINDOW_KEYS, generate_windows

FIXTURE = Path(__file__).parent / "fixtures" / "advio_mini"
YAML = Path(__file__).resolve().parents[1] / "configs" / "splits" / "advio_windows_v1.yaml"


def _synth_inf(
    frame_t: np.ndarray,
    t_gps: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
) -> InferenceChannels:
    n = len(t_gps)
    nf = len(frame_t)
    return InferenceChannels(
        sequence_id="synth",
        frame_timestamps=np.asarray(frame_t, dtype=np.float64),
        frame_indices=np.arange(1, nf + 1, dtype=np.float64),
        video_path=None,
        location_t=np.asarray(t_gps, dtype=np.float64),
        lat=np.asarray(lat, dtype=np.float64),
        lon=np.asarray(lon, dtype=np.float64),
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


def test_frozen_yaml_v1_parameters():
    text = YAML.read_text(encoding="utf-8")
    assert "duration_s: 16" in text
    assert "stride_s: 16" in text
    assert "n_frames: 8" in text
    assert "min_gps_path_m: 10" in text
    assert "t_origin_s: 0" in text
    assert FROZEN_V1["duration_s"] == 16.0
    assert FROZEN_V1["stride_s"] == 16.0
    assert FROZEN_V1["n_frames"] == 8
    assert FROZEN_V1["min_gps_path_m"] == 10.0


def test_mini_default_duration_yields_no_windows():
    seq = load_advio(FIXTURE, "mini", imu_allowed=False)
    assert generate_windows(seq.inference) == []


def test_mini_injected_thresholds():
    seq = load_advio(FIXTURE, "mini", imu_allowed=False)
    inf = seq.inference
    path = corelocation_interval_distance(inf, 0.0, 2.0)
    assert path is not None and path > 1.0
    wins = generate_windows(
        inf, duration_s=2.0, stride_s=2.0, n_frames=2, min_gps_path_m=1.0
    )
    assert len(wins) >= 1
    w = wins[0]
    assert tuple(w.keys()) == WINDOW_KEYS
    assert w["t0"] == 0.0 and w["t1"] == 2.0
    assert len(w["frame_indices"]) == 2
    assert w["gps_path_m"] == path
    assert all(k not in w for k in ("gt", "centers", "pose"))


def test_min_gps_path_filters_stationary():
    seq = load_advio(FIXTURE, "mini", imu_allowed=False)
    wins = generate_windows(
        seq.inference, duration_s=2.0, stride_s=2.0, n_frames=2, min_gps_path_m=1e6
    )
    assert wins == []


def test_gps_span_required():
    frame_t = np.linspace(0.0, 32.0, 33)
    t_gps = np.array([40.0, 41.0, 42.0])
    lat = np.array([60.18680, 60.18700, 60.18720])
    lon = np.array([24.82200, 24.82220, 24.82240])
    inf = _synth_inf(frame_t, t_gps, lat, lon)
    wins = generate_windows(inf, duration_s=16.0, stride_s=16.0, n_frames=8, min_gps_path_m=1.0)
    assert wins == []


def test_moving_synth_emits_nonoverlap_windows():
    frame_t = np.linspace(0.0, 32.0, 33)
    t_gps = np.linspace(0.0, 32.0, 33)
    lat0, lon0 = 60.18680, 24.82200
    lat = lat0 + 3e-5 * np.arange(33)
    lon = lon0 + 3e-5 * np.arange(33)
    inf = _synth_inf(frame_t, t_gps, lat, lon)
    wins = generate_windows(inf, duration_s=16.0, stride_s=16.0, n_frames=8, min_gps_path_m=10.0)
    assert len(wins) == 2
    assert wins[0]["t0"] == 0.0 and wins[0]["t1"] == 16.0
    assert wins[1]["t0"] == 16.0 and wins[1]["t1"] == 32.0
    assert wins[0]["t1"] <= wins[1]["t0"]
    assert all(len(w["frame_indices"]) == 8 for w in wins)
    assert all(w["gps_path_m"] >= 10.0 for w in wins)
    assert generate_windows(inf, duration_s=16.0, stride_s=16.0, n_frames=8, min_gps_path_m=10.0) == wins
