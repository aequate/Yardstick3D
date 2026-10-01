"""Prospective UrbanNav v3 contract tests (no bags, no GT, no network)."""
from pathlib import Path

import numpy as np
import pytest

from yardstick3d.datasets import urbannav as un

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "prospective_metric_grounding_v3.yaml"
EXPECTED_SHA = "79635e9f4ff4a1c5dc12cd8529d3f4078441cc26e64295155c0a74c500e2a320"


def test_contract_hash_frozen():
    un.check_contract_frozen(CONTRACT, EXPECTED_SHA)


def test_contract_hash_tamper_rejected(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_bytes(CONTRACT.read_bytes() + b"\n# tamper\n")
    with pytest.raises(ValueError, match="hash"):
        un.check_contract_frozen(p, EXPECTED_SHA)


def test_gps2utc_conversion():
    assert un.gps2utc(2158, 95593) == pytest.approx(1621218775.0)


def test_sync_mapping_has_no_fitted_offset():
    import inspect

    src = inspect.getsource(un)
    assert "fitted_offset" not in src
    assert "cross_correlat" not in src.lower()
    assert un.nearest_index(np.array([0.0, 0.1, 0.2]), 0.11, 0.15) == 1
    assert un.nearest_index(np.array([0.0]), np.array(99.0), 0.15) is None


def test_windows_validation_vs_test_roles():
    wins = un.prospective_windows(t_bag0=1000.0)
    assert len(wins) == 7
    assert [w.role for w in wins].count("validation") == 2
    assert [w.role for w in wins].count("prospective_test") == 5
    assert wins[0].t_start == pytest.approx(1090.0)
    assert wins[-1].t_end == pytest.approx(1000.0 + 630.0 + 16.0)
    un.enforce_window_role(90.0, "validation")
    un.enforce_window_role(270.0, "prospective_test")
    with pytest.raises(ValueError):
        un.enforce_window_role(270.0, "validation")
    with pytest.raises(ValueError):
        un.enforce_window_role(90.0, "prospective_test")


def test_frame_cue_matching_and_coverage():
    img_t = np.arange(0, 20, 1 / 15)
    cue_t = np.arange(0, 20, 1.0)
    cov = un.match_coverage(img_t, cue_t, 0.6)
    assert 0.95 < cov <= 1.0  # tail images past last 1 Hz cue fix legitimately uncovered
    assert un.match_coverage(np.arange(0, 19, 1 / 15), cue_t, 0.6) == pytest.approx(1.0)
    assert un.match_coverage(np.array([0.0]), np.array([50.0]), 0.6) == pytest.approx(0.0)
    idx = un.select_frame_indices_from_images(img_t, np.array([0.03, 0.1]), 0.15)
    assert len(idx) == 2
    with pytest.raises(RuntimeError):
        un.select_frame_indices_from_images(np.array([0.0]), np.array([99.0]), 0.15)


def test_coordinate_transform_enu_metres():
    lat = np.array([22.0000, 22.0001])
    lon = np.array([114.0000, 114.0000])
    enu = un.cue_enu_from_llh(lat, lon, np.array([0.0, 0.0]))
    assert enu.shape == (2, 3)
    assert 9.0 < enu[1, 1] < 13.0


def test_cue_reference_isolation():
    un.assert_cue_reference_isolation(["ublox_f9p_nmea_gga"], ["span_cpt_ie_inspvax"])
    with pytest.raises(ValueError):
        un.assert_cue_reference_isolation(["span_cpt_ie_inspvax"], ["span_cpt_ie_inspvax"])
    with pytest.raises(ValueError):
        un.assert_cue_reference_isolation(["span_cpt_ie_gt_raw"], ["span_cpt_ie_inspvax"])
    with pytest.raises(ValueError):
        un.assert_cue_reference_isolation(["ublox_f9p_nmea_gga"], ["ublox_f9p_nmea_gga"])


def test_nmea_gga_known_answer_and_gating():
    line = "$GNGGA,023255.00,2218.07395,N,11410.74198,E,2,12,0.62,9.0,M,-1.6,M,,0000*51"
    out = un.parse_nmea_gga_cue(line + "\n", "170521")
    assert len(out["t"]) == 1
    assert out["lat_deg"][0] == pytest.approx(22 + 18.07395 / 60, rel=1e-9)
    assert out["lon_deg"][0] == pytest.approx(114 + 10.74198 / 60, rel=1e-9)
    assert out["alt_m"][0] == pytest.approx(9.0)
    assert out["quality"][0] == 2 and out["nsat"][0] == 12
    import calendar, datetime
    assert out["t"][0] == calendar.timegm(datetime.datetime(2021, 5, 17, 2, 32, 55).timetuple())
    # RTK-fixed quality and void fixes are dropped, never used
    bad = "$GNGGA,023256.00,2218.07395,N,11410.74198,E,4,12,0.62,9.0,M,-1.6,M,,0000*51\n"
    assert len(un.parse_nmea_gga_cue(bad, "170521")["t"]) == 0
    void = "$GNGGA,023256.00,,,,,0,00,99.9,,M,,M,,0000*00\n"
    assert len(un.parse_nmea_gga_cue(void, "170521")["t"]) == 0


def test_rmc_date_extraction():
    txt = "$GNRMC,023255.00,A,2218.07395,N,11410.74198,E,5.5,90.0,170521,,,A*00\n"
    assert un.extract_rmc_date(txt) == "170521"
    with pytest.raises(RuntimeError):
        un.extract_rmc_date("$GNGGA,023255.00,,,,,0,00,99.9,,M,,M,,*00\n")


def test_missing_sensor_values_dropped():
    cue_t = np.arange(0, 17, 1.0)
    enu = np.column_stack([cue_t * 8.0, np.zeros_like(cue_t), np.zeros_like(cue_t)])
    t_clip = np.linspace(0, 16, 8)
    cs, info = un.constraints_from_ublox_spp_displacement(t_clip, cue_t, enu)
    assert info["n_kept"] + info["n_dropped"] == 7
    assert info["n_kept"] >= 5
    mask = ~((cue_t > t_clip[3]) & (cue_t <= t_clip[4]))
    cs2, info2 = un.constraints_from_ublox_spp_displacement(t_clip, cue_t[mask], enu[mask])
    assert info2["n_dropped"] >= info["n_dropped"] + 1
    for c in cs2:
        assert np.isfinite(c.length_m) and c.length_m > 0


def test_polyline_1hz_matches_distance():
    cue_t = np.arange(0, 11, 1.0)
    enu = np.column_stack([cue_t * 8.0, np.zeros_like(cue_t), np.zeros_like(cue_t)])
    b, cov = un.cue_polyline_lengths(cue_t, enu, np.array([0.0, 10.0]))
    assert b[0] == pytest.approx(80.0, rel=1e-9)


def test_reference_interpolation_eval_only():
    ref_t = np.array([0.0, 1.0, 2.0])
    ref_enu = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)
    out = un.interpolate_reference_at(ref_t, ref_enu, np.array([0.5, 1.5]))
    assert out[0, 0] == pytest.approx(0.5)
    assert not np.isfinite(un.interpolate_reference_at(ref_t, ref_enu, np.array([99.0]))[0, 0])


def test_sync_manifest_schema():
    m = un.SyncManifest(t_bag0=1.0, n_images_left=5, n_cue_fixes=5,
                        camera_cue_match_fraction=1.0,
                        time_reference_notes={"source": "stage1"})
    d = m.to_dict()
    assert d["ref_positions_opened"] is False
    assert d["max_dt_match_camera_cue_s"] == pytest.approx(0.15)
