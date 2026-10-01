"""Tests required before freeze by configs/prospective_cross_backbone_v2.DRAFT.yaml (integrity.required_tests_before_freeze)."""
from __future__ import annotations

import importlib.util
import json
import struct
from pathlib import Path

import numpy as np
import pytest

from yardstick3d.datasets import timescale as ts
from yardstick3d.datasets import v2_sync_guards as sg
from yardstick3d.datasets import v2_window_selection as ws

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("cb_v2_stage1", ROOT / "scripts" / "cross_backbone_v2_stage1.py")
s1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s1)

T0 = 1_700_000_000.0
LAT0, LON0 = 22.3, 114.2


def _llh(e, n, u):
    lat = LAT0 + n / 111_320.0
    lon = LON0 + e / (111_320.0 * np.cos(np.radians(LAT0)))
    return np.stack([lat, lon, 30.0 + u], axis=1)


def _track(speed_fn, dur=400.0, noise=0.0, seed=0):
    t = np.arange(0.0, dur, 0.1)
    v = speed_fn(t)
    e = np.concatenate([[0.0], np.cumsum(v[:-1] * 0.1)])
    rng = np.random.default_rng(seed)
    e = e + noise * rng.standard_normal(t.size)
    n = noise * rng.standard_normal(t.size)
    return T0 + t, _llh(e, n, np.zeros_like(t))


def _sine(t):
    return 3.0 + 2.5 * np.sin(2 * np.pi * t / 40.0)


def _enu(llh):
    from yardstick3d.datasets import mars_lvig as ml

    return ml.cue_enu_from_llh(llh[:, 0], llh[:, 1], llh[:, 2])


def _cam(dur=400.0):
    return T0 + np.arange(0.0, dur, 0.1)


def _streams(cue_offset=18.0, cam_offset=0.0, rtk_offset=0.0, speed_fn=_sine, noise=0.0):
    cue_utc, llh = _track(speed_fn, noise=noise)
    cam_utc = _cam()
    rec_cam = cam_utc + 0.003
    rec_cue = cue_utc + 0.004
    rtk_utc = cue_utc[::1]
    return dict(
        cam_header=cam_utc + cam_offset, cam_record=rec_cam,
        cue_header=cue_utc + cue_offset, cue_record=rec_cue, cue_llh=llh,
        rtk_header=rtk_utc + rtk_offset, rtk_record=rtk_utc + 0.002,
    )


# ---------------------------------------------------------------- timescale
def test_timescale_gpst_fixture_classify_and_convert():
    rec = np.linspace(0, 100, 101) + T0
    hdr = rec + 18.0
    assert ts.classify_timescale(float(np.median(hdr - rec))) == "gpst"
    classes = ts.require_common_timescale({"cue": (hdr, rec), "cam": (rec + 0.01, rec)})
    assert classes == {"cue": "gpst", "cam": "utc"}
    assert np.allclose(ts.to_utc(hdr, classes["cue"]), rec)
    with pytest.raises(RuntimeError):
        ts.require_common_timescale({"x": (rec + 5.0, rec)})


# ------------------------------------------------------------ Stage-1 gate
def test_gate_classifies_gpst_cue_and_converts():
    d = _streams(cue_offset=18.0)
    gate = sg.stage1_timescale_gate({
        s1.CAMERA_TOPIC: (d["cam_header"], d["cam_record"]),
        s1.CUE_TOPIC: (d["cue_header"], d["cue_record"]),
        s1.RTK_POSITION_TOPIC: (d["rtk_header"], d["rtk_record"]),
    })
    assert gate.classes[s1.CUE_TOPIC] == "gpst"
    assert gate.classes[s1.CAMERA_TOPIC] == "utc"
    cue = sg.stamp_stream(s1.CUE_TOPIC, d["cue_header"], gate)
    assert np.allclose(cue.t_utc, d["cue_header"] - 18.0)
    assert gate.cross_topic_max_diff_s < 0.25


def test_plan_gpst_cue_gives_same_windows_as_utc_cue():
    a = s1.plan_stage1(**_streams(cue_offset=18.0))
    b = s1.plan_stage1(**_streams(cue_offset=0.0))
    assert a["gate"]["classes"][s1.CUE_TOPIC] == "gpst" and b["gate"]["classes"][s1.CUE_TOPIC] == "utc"
    assert [w["t0"] for w in a["windows"]] == [w["t0"] for w in b["windows"]]
    assert [w["role"] for w in a["windows"]] == [w["role"] for w in b["windows"]]


@pytest.mark.parametrize("kw", [{"cue_offset": 5.0}, {"cam_offset": 5.0}, {"rtk_offset": 5.0}])
def test_offset_5s_stops(kw):
    with pytest.raises(sg.Stage1Stop):
        s1.plan_stage1(**_streams(**kw))


def test_gate_spread_and_cross_topic_rules_stop():
    d = _streams(cue_offset=0.0)
    h = d["cue_header"].copy()
    h[len(h) // 2:] += 18.0  # mid-bag timescale switch; median still utc
    h[: len(h) // 2 + 5] = d["cue_header"][: len(h) // 2 + 5]
    with pytest.raises(sg.Stage1Stop):
        sg.stage1_timescale_gate({"c": (h, d["cue_record"]), "cam": (d["cam_header"], d["cam_record"])})
    with pytest.raises(sg.Stage1Stop):  # both 'utc' but medians differ by 0.4 s >= 0.25 s
        sg.stage1_timescale_gate({"a": (T0 + np.arange(50) + 0.4, T0 + np.arange(50)), "b": (T0 + np.arange(50), T0 + np.arange(50))})


# ---------------------------------------------------------- pairing guard
def test_pairing_rejects_unconverted_stamps():
    d = _streams(cue_offset=18.0)
    gate = sg.stage1_timescale_gate({
        s1.CAMERA_TOPIC: (d["cam_header"], d["cam_record"]),
        s1.CUE_TOPIC: (d["cue_header"], d["cue_record"]),
    })
    cam = sg.stamp_stream(s1.CAMERA_TOPIC, d["cam_header"], gate)
    cue = sg.stamp_stream(s1.CUE_TOPIC, d["cue_header"], gate)
    idx, ok = sg.pair_nearest(cam, cue, 0.06)
    assert ok.mean() > 0.99
    # raw arrays / forged streams / unclassified topics / tampering are all refused
    with pytest.raises(TypeError):
        sg.pair_nearest(d["cam_header"], d["cue_header"], 0.06)
    with pytest.raises(TypeError):
        sg.pair_nearest(cam, d["cue_header"], 0.06)
    with pytest.raises(TypeError):
        sg.UtcStream("cue", d["cue_header"], "gpst", d["cue_header"])  # no private token
    with pytest.raises(sg.Stage1Stop):
        sg.stamp_stream("/some/other/topic", d["cue_header"], gate)
    forged = sg.stamp_stream(s1.CUE_TOPIC, d["cue_header"], gate)
    object.__setattr__(forged, "t_utc", d["cue_header"])  # unconverted GPST stamps
    with pytest.raises(ValueError):
        sg.pair_nearest(cam, forged, 0.06)
    # and the raw (unconverted) pairing would in fact have failed to match
    raw_pair = np.abs(d["cam_header"][:, None][:50] - d["cue_header"][None, :]).min(axis=1)
    assert raw_pair.min() > 10.0


# ------------------------------------------------ window selection (cue-only)
def test_selection_determinism_and_input_order_invariance():
    cue_t, llh = _track(_sine)
    enu = _enu(llh)
    img = _cam()
    st1 = ws.window_statistics(img, cue_t, enu)
    st2 = ws.window_statistics(img[::-1].copy(), cue_t[::-1].copy(), enu[::-1].copy())
    assert st1 == st2
    r1, r2 = ws.select_windows(st1), ws.select_windows(st2)
    assert r1 == r2 and ws.select_windows(st1) == r1
    roles = [s["role"] for s in r1.starts]
    assert roles.count(ws.ROLE_TEST) == 4 and roles.count(ws.ROLE_VALIDATION) == 2
    ks = sorted(s["k"] for s in r1.starts)
    assert all(b - a > 1 for a, b in zip(ks, ks[1:]))  # non-adjacent in any role
    tests = r1.by_role(ws.ROLE_TEST)
    ranked = sorted((w for w in st1 if w.cls == "SV"), key=lambda w: (-w.S, w.t0))
    assert {s["k"] for s in tests} <= {w.k for w in ranked}
    assert all(w.S >= 0.10 for w in st1 if w.cls == "SV")


def test_hover_to_cruise_is_sv():
    def hover_cruise(t):
        return np.clip((t - 38.0) / 4.0, 0.0, 1.0) * 5.0

    cue_t, llh = _track(hover_cruise, dur=200.0)
    st = {w.k: w for w in ws.window_statistics(_cam(200.0), cue_t, _enu(llh))}
    w = st[2]  # [32, 48]: hover -> ramp -> cruise
    assert w.eligible and w.cls == "SV", w
    assert w.cv >= 0.20 and w.dv >= 1.5 and w.S >= 0.10
    assert st[1].cls == "ineligible"  # pure hover: v_mean < 1.5


def test_constant_3ms_plus_noise_is_cc_and_never_sv():
    cue_t, llh = _track(lambda t: np.full_like(t, 3.0), dur=300.0, noise=0.04)
    stats = ws.window_statistics(_cam(300.0), cue_t, _enu(llh))
    el = [w for w in stats if w.eligible]
    assert el and all(w.cls == "CC" for w in el), [(w.k, w.cls, w.cv, w.S) for w in el]
    with pytest.raises(ws.SequenceIneligible):
        ws.select_windows(stats)


def test_gap_makes_window_ineligible_and_thresholds_are_frozen_constants():
    cue_t, llh = _track(_sine, dur=200.0)
    keep = ~((cue_t - T0 > 60.0) & (cue_t - T0 < 66.0))  # 6 s cue gap inside k=3 [48,64] / k=4 [64,80]
    st = {w.k: w for w in ws.window_statistics(_cam(200.0), cue_t[keep], _enu(llh)[keep])}
    assert not st[3].eligible and not st[4].eligible
    assert (ws.SV_CV_MIN, ws.SV_DV_MIN, ws.SV_S_MIN, ws.V_MEAN_MIN) == (0.20, 1.5, 0.10, 1.5)
    assert (ws.CC_CV_MAX, ws.CC_S_MAX, ws.N_TEST, ws.N_VALIDATION, ws.N_CC) == (0.10, 0.05, 4, 2, 2)


def test_cc_controls_and_validation_fill_with_other():
    def mk(k, cls, S=0.2):
        return ws.WindowStats(k, 16.0 * k, True, cls, True, 1.0, 1.0, 3.0, 0.3, 2.0, S)

    stats = [mk(1, "SV", .5), mk(2, "SV", .9), mk(3, "SV", .4), mk(5, "SV", .3), mk(7, "SV", .2), mk(9, "SV", .15),
             mk(11, "SV", .1), mk(13, "other"), mk(15, "CC"), mk(17, "CC"), mk(19, "CC")]
    sel = ws.select_windows(stats)
    ks = {r: [x["k"] for x in sel.starts if x["role"] == r] for r in (ws.ROLE_TEST, ws.ROLE_VALIDATION, ws.ROLE_CC)}
    assert ks[ws.ROLE_TEST] == [2, 5, 7, 9]  # ranked by S desc; 1 and 3 skipped (adjacent to 2)
    assert ks[ws.ROLE_VALIDATION] == [11, 13]  # next SV, then earliest eligible 'other' fill
    assert ks[ws.ROLE_CC] == [15, 17]  # earliest first, n_cc = 2
    with pytest.raises(ws.SequenceIneligible):
        ws.select_windows([mk(1, "SV"), mk(3, "SV"), mk(5, "SV")])


# ------------------------------------------------ rtk_position header-only
class _Tracker:
    def __init__(self, data):
        self._d = data
        self.touched = []

    def __getitem__(self, key):
        self.touched.append(key)
        return self._d[key]

    def __len__(self):  # a header-only parser has no need for the length
        raise AssertionError("len() must not be called")

    def __iter__(self):
        raise AssertionError("iteration must not be used")


def test_rtk_header_parser_never_touches_bytes_beyond_12():
    header = struct.pack("<III", 7, 1_700_000_123, 250_000_000)
    body_a = header + b"\xff" * 200
    body_b = header + b"\x00" * 3
    tr = _Tracker(body_a)
    assert sg.parse_ros1_header_only(tr) == (7, 1_700_000_123, 250_000_000)
    assert len(tr.touched) == 1
    sl = tr.touched[0]
    assert isinstance(sl, slice) and (sl.stop is not None and sl.stop <= 12)
    assert sg.parse_ros1_header_only(body_b) == sg.parse_ros1_header_only(body_a)  # tail-independent
    assert sg.rtk_header_stamp_s(body_a) == pytest.approx(1_700_000_123.25)
    with pytest.raises(ValueError):
        sg.parse_ros1_header_only(header[:11])
    with pytest.raises(ValueError):
        sg.parse_ros1_header_only(struct.pack("<III", 0, 1, 2_000_000_000))


def test_stage1_script_never_reads_reference_values():
    text = (ROOT / "scripts" / "cross_backbone_v2_stage1.py").read_text(encoding="utf-8")
    for bad in ("rtk_velocity", "rtk_yaw", "rtk_info_position", "rtk_info_yaw", "rtk_connection_status"):
        assert text.count(f'"/dji_osdk_ros/{bad}"') == 1  # only inside the FORBIDDEN list
    assert "deserialize_ros1" in text and "RTK_POSITION_TOPIC" in text
    assert "_headers(reader, conns[RTK_POSITION_TOPIC])" in text
    for t in s1.FORBIDDEN_VALUE_TOPICS:
        if t != s1.RTK_POSITION_TOPIC:
            with pytest.raises(RuntimeError):
                s1._guard_topic(t)
    # the plan function takes no reference argument at all
    assert not any("ref" in a and "t_ref" not in a for a in s1.plan_stage1.__code__.co_varnames[: s1.plan_stage1.__code__.co_argcount])


# ------------------------------------------------------ contract / time_sync
def test_contract_loader_refuses_draft_and_placeholders(tmp_path):
    with pytest.raises(SystemExit):
        s1.load_contract(ROOT / "configs" / "prospective_cross_backbone_v2.DRAFT.yaml")
    y = tmp_path / "c.yaml"
    reg = tmp_path / "reg.json"
    y.write_text('status: frozen_pre_data\ndataset:\n  bag_filename: "<PLACEHOLDER>"\n  x: 1\n  gpst_minus_utc_s: 18.0\n', encoding="utf-8")
    reg.write_text(json.dumps({"sha": __import__("hashlib").sha256(y.read_bytes()).hexdigest()}))
    with pytest.raises(SystemExit):
        s1.load_contract(y, reg)
    y.write_text('status: frozen_pre_data\ndataset:\n  bag_filename: "abc.bag"  # c\n  gpst_minus_utc_s: 18.0\n', encoding="utf-8")
    with pytest.raises(SystemExit):  # sha not in registry
        s1.load_contract(y, reg)
    reg.write_text(json.dumps({"sha": __import__("hashlib").sha256(y.read_bytes()).hexdigest()}))
    with pytest.raises(SystemExit):  # no rank_order -> fail closed
        s1.load_contract(y, reg)
    y.write_text(
        "status: frozen_pre_data\n"
        "dataset:\n"
        '  bag_filename: "abc_GNSS01.bag"  # c\n'
        "  gpst_minus_utc_s: 18.0\n"
        "sequence_selection:\n"
        "  rank_order:\n"
        "    - abc_GNSS01  # x\n"
        '  rule: "r"\n',
        encoding="utf-8",
    )
    reg.write_text(json.dumps({"sha": __import__("hashlib").sha256(y.read_bytes()).hexdigest()}))
    assert s1.load_contract(y, reg)["bag"].name == "abc_GNSS01.bag"


def test_time_sync_checks():
    rec = T0 + np.arange(100.0)
    hdr = rec + 18.0  # gpst header
    ok = s1.check_time_sync_topic("/dji_osdk_ros/time_sync_gps_utc", hdr, rec, list(rec + 0.1), [])
    assert ok["timescale"] == "gpst" and abs(ok["decoded_residual_median_s"]) < 2.0
    with pytest.raises(sg.Stage1Stop):
        s1.check_time_sync_topic("/dji_osdk_ros/time_sync_gps_utc", hdr, rec, list(rec + 5.0), [])
    with pytest.raises(sg.Stage1Stop):  # no recognizable UTC field -> fail closed
        s1.check_time_sync_topic("/dji_osdk_ros/time_sync_fc_time_utc", rec, rec, [None] * 100, [])
    tod = list(rec % 86400.0)
    assert s1.check_time_sync_topic("/dji_osdk_ros/time_sync_nmea_msg", rec, rec, [], tod)["count"] == 100
    with pytest.raises(sg.Stage1Stop):
        s1.check_time_sync_topic("/dji_osdk_ros/time_sync_nmea_msg", rec, rec, [], [x + 3.0 for x in tod])
    with pytest.raises(sg.Stage1Stop):  # unexplained header offset
        s1.check_time_sync_topic("/dji_osdk_ros/time_sync_gps_utc", rec + 5.0, rec, list(rec), [])
    assert s1.check_time_sync_topic("/ublox_driver/time_pulse_info", hdr, rec, [], [])["timescale"] == "gpst"
    assert s1.gprmc_tod_s("$GPRMC,010203.50,A,2218.0,N") == pytest.approx(3723.5)


def test_v3_a1_integer_second_labels_and_dji_decoders():
    from types import SimpleNamespace as NS
    rec = T0 + np.arange(100.0) + 0.437
    # observed on HKisland_GNSS02: fc_time_utc +0.563 s, gps_utc +1.062 s (next-PPS labels) -> PASS under A1
    for off in (0.563, 1.062):
        r = s1.check_time_sync_topic("/dji_osdk_ros/time_sync_gps_utc", rec, rec, list(rec + off), [])
        assert r["decoded_residual_median_s"] == pytest.approx(off)
    for bad in (18.0, -18.0, 2.5):  # GPST/leap-second error or gross offset still STOPs
        with pytest.raises(sg.Stage1Stop):
            s1.check_time_sync_topic("/dji_osdk_ros/time_sync_gps_utc", rec, rec, list(rec + bad), [])
    jitter = list(rec + 0.5 + np.where(np.arange(100) % 2, 0.4, -0.4))  # unstable phase: spread >= 0.5 s
    with pytest.raises(sg.Stage1Stop):
        s1.check_time_sync_topic("/dji_osdk_ros/time_sync_fc_time_utc", rec, rec, jitter, [])
    fc = NS(stamp=NS(sec=1, nanosec=0), fc_timestamp_us=0, fc_utc_yymmdd=231024, fc_utc_hhmmss=75157)
    assert s1.extract_utc_field(fc) == 1698133917.0
    assert s1.extract_utc_field(NS(stamp=NS(sec=1, nanosec=0), UTCTimeData="UTC 231024 075158 5 ")) == 1698133918.0


def test_time_sync_pairs_sparse_messages_with_own_stamp():
    rec = T0 + np.arange(100.0) * 0.2
    tod = [((r + 0.3) % 86400.0) if i % 5 == 0 else None for i, r in enumerate(rec)]  # GPRMC is 1 in 5 sentences
    r = s1.check_time_sync_topic("/dji_osdk_ros/time_sync_nmea_msg", rec, rec, [], tod)
    assert r["decoded_residual_median_s"] == pytest.approx(0.3, abs=1e-6)
    dec = [(x + 1.0) if i % 3 == 0 else None for i, x in enumerate(rec)]
    r = s1.check_time_sync_topic("/dji_osdk_ros/time_sync_gps_utc", rec, rec, dec, [])
    assert r["decoded_residual_median_s"] == pytest.approx(1.0, abs=1e-6)


def test_cue_only_fallback_moves_to_next_rank(tmp_path, monkeypatch):
    monkeypatch.setattr(s1, "OUT", tmp_path)
    c0 = s1.load_contract()
    assert c0["sequence"] == "HKisland_GNSS02" and c0["fallbacks_used"] == 0
    for q in ("HKisland_GNSS02", "HKisland_GNSS03", "HKisland_GNSS01"):
        (tmp_path / f"{s1.INELIGIBLE_PREFIX}{q}.json").write_text("{}")
    with pytest.raises(SystemExit):
        s1.load_contract()  # 3 fallbacks > max 2 -> STOP


def _gate_v4(cam_off, cue_off, rtk_off, n=400):
    rec = T0 + np.arange(n) * 0.1
    return sg.stage1_timescale_gate_v4("cam", (rec + cam_off, rec), "cue", (rec + cue_off, rec), {"rtk": (rec + rtk_off, rec)})


def test_v4_a2_recorder_clock_offset_observed_patterns():
    # measured header-record medians (s): GNSS02 cam +0.41 / cue +18.498; GNSS01 cam -0.588 / cue +17.510; v1 airport cam -0.06 / cue +18.022
    for cam, cue in ((0.41, 18.498), (-0.588, 17.510), (-0.06, 18.022)):
        g = _gate_v4(cam, cue, 0.0)
        assert g.classes == {"cam": "utc", "cue": "gpst", "rtk": "recorder"}
        assert g.recorder_offset_s == pytest.approx(cam)
        rtk = sg.stamp_stream("rtk", T0 + np.arange(3.0), g)
        assert np.allclose(rtk.t_utc - (T0 + np.arange(3.0)), cam)  # UTC = recorder stamp + delta
    with pytest.raises(sg.Stage1Stop):
        _gate_v4(0.41, 18.0, 0.0)          # cue disagrees with camera by 0.41 s >= 0.25 s
    with pytest.raises(sg.Stage1Stop):
        _gate_v4(0.41, 18.41, 0.3)         # rtk not on recorder clock
    with pytest.raises(sg.Stage1Stop):
        _gate_v4(2.5, 20.5, 0.0)           # |delta| >= 2.0 s


def test_v4_time_sync_uses_recorder_offset():
    rec = T0 + np.arange(100.0) + 0.437
    r = s1.check_time_sync_topic("/dji_osdk_ros/time_sync_gps_utc", rec, rec, list(rec + 1.062), [], recorder_offset_s=0.41)
    assert r["recorder_clock"] and r["decoded_residual_median_s"] == pytest.approx(1.062 - 0.41)
