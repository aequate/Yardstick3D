"""Cross-backbone v5 (amendment A3: pooled SV windows) tests. Synthetic fixtures only: no GT, no GPU, no network."""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

from yardstick3d.datasets import v2_scoring as v2
from yardstick3d.datasets import v2_window_selection as ws

ROOT = Path(__file__).resolve().parents[1]
POOL = ["HKisland_GNSS02", "HKisland_GNSS03", "HKisland_GNSS01"]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


s5 = _load("cb_v5_stage1_t", "scripts/cross_backbone_v5_stage1.py")
score_mod = _load("cb_v5_score_t", "scripts/cross_backbone_v2_score.py")
pack_mod = _load("cb_v5_pack_t", "scripts/build_vggt_pack_v2.py")
da3_mod = _load("cb_v5_da3_t", "scripts/cross_backbone_v2_da3.py")
tp = _load("cb_v5_pipe_t", "tests/test_cross_backbone_v2_pipeline.py")


def _w(k, cls, S=0.0):
    elig = cls != "ineligible"
    return ws.WindowStats(k, 16.0 * k, elig, cls, True, 1.0, 1.0, 3.0 if elig else 0.5, 0.3, 2.0, S,
                          () if elig else ("v_mean<1.5",))


# ------------------------------- contract -------------------------------------
def test_v5_changes_only_pooling_not_thresholds():
    """Every v4 line survives verbatim in v5 except the header, identity, dataset-sequence lines and the
    four rules A3 replaces (selection minimum, primary endpoint, failure policy, immutability)."""
    v4 = (ROOT / "configs/prospective_cross_backbone_v4.yaml").read_text(encoding="utf-8").splitlines()
    v5 = set(v2.CONTRACT_PATH.read_text(encoding="utf-8").splitlines())
    allowed = ("#", "name:", "supersedes:", "  sequence:", "  bag_drive_id:", "  bag_bytes_documented:", "  bag_sha256:",
               "    minimum:", "primary_endpoint:", '  - "Stage-1 timescale/time_sync STOP', "contract_immutability:")
    changed = [line for line in v4 if line not in v5]
    assert changed and all(line.startswith(allowed) for line in changed), [x for x in changed if not x.startswith(allowed)]
    assert (ws.N_TEST, ws.N_VALIDATION, ws.N_CC) == (4, 2, 2)
    assert (ws.SV_CV_MIN, ws.SV_DV_MIN, ws.SV_S_MIN, ws.V_MEAN_MIN) == (0.20, 1.5, 0.10, 1.5)


# ------------------------------- pooled selection ------------------------------
def _census_2_2_1():
    return {
        "HKisland_GNSS02": [_w(1, "ineligible"), _w(2, "other"), _w(6, "SV", 1.847), _w(10, "CC", 0.01),
                            _w(11, "CC", 0.02), _w(24, "SV", 1.567), _w(25, "SV", 0.9)],  # k25 adjacent to k24
        "HKisland_GNSS03": [_w(9, "SV", 1.2), _w(19, "SV", 0.5), _w(4, "CC", 0.0)],
        "HKisland_GNSS01": [_w(30, "SV", 1.2), _w(3, "other"), _w(12, "CC", 0.01)],
    }


def test_pooled_selection_ranking_ties_adjacency_and_roles():
    sel = ws.select_windows_pooled(_census_2_2_1(), POOL)
    test = [(s["sequence"], s["k"]) for s in sel.by_role(ws.ROLE_TEST)]
    # S desc; tie 1.2 -> sequence rank (GNSS03 before GNSS01)
    assert sorted(test) == sorted([("HKisland_GNSS02", 6), ("HKisland_GNSS02", 24),
                                   ("HKisland_GNSS03", 9), ("HKisland_GNSS01", 30)])
    val = [(s["sequence"], s["k"], s["cls"]) for s in sel.by_role(ws.ROLE_VALIDATION)]
    # k25 is adjacent to selected k24 (same sequence) -> skipped; next SV = GNSS03 k19, then 'other' by (rank, t0)
    assert val == [("HKisland_GNSS02", 2, "other"), ("HKisland_GNSS03", 19, "SV")]
    cc = [(s["sequence"], s["k"]) for s in sel.by_role(ws.ROLE_CC)]
    assert cc == [("HKisland_GNSS02", 10), ("HKisland_GNSS03", 4)]  # k11 adjacent to k10
    assert sel.n_sv_candidates == 6
    # determinism
    assert ws.select_windows_pooled(_census_2_2_1(), POOL).starts == sel.starts


def test_pooled_adjacency_is_within_sequence_only():
    cen = {"HKisland_GNSS02": [_w(5, "SV", 0.9), _w(6, "SV", 0.8)],
           "HKisland_GNSS03": [_w(5, "SV", 0.7), _w(6, "SV", 0.6)],
           "HKisland_GNSS01": [_w(5, "SV", 0.5), _w(8, "SV", 0.4)]}
    test = [(s["sequence"], s["k"]) for s in ws.select_windows_pooled(cen, POOL).by_role(ws.ROLE_TEST)]
    assert sorted(test) == sorted([("HKisland_GNSS02", 5), ("HKisland_GNSS03", 5),
                                   ("HKisland_GNSS01", 5), ("HKisland_GNSS01", 8)])


def test_pooled_stop_below_four_and_fixed_pool():
    cen = {"HKisland_GNSS02": [_w(6, "SV", 1.0), _w(7, "SV", 0.9)], "HKisland_GNSS03": [_w(9, "SV", 1.0)],
           "HKisland_GNSS01": [_w(3, "CC", 0.0)]}
    with pytest.raises(ws.SequenceIneligible, match="only 2 non-adjacent pooled SV"):
        ws.select_windows_pooled(cen, POOL)
    with pytest.raises(ValueError):
        ws.select_windows_pooled({"HKisland_GNSS02": []}, POOL)


def test_census_roundtrip_and_pooled_windows_carry_frames():
    cen = _census_2_2_1()
    records = {}
    for seq, stats in cen.items():
        stats = list(stats) + [ws.WindowStats(40, 640.0, False, "ineligible", False, 0.0, float("nan"), float("nan"),
                                              float("nan"), float("nan"), float("nan"), ("frames_not_matched_0.06s",))]
        rec = {"candidate_stats": json.loads(json.dumps(
            [tp.v2.clean_json(w.__dict__ | {"reasons": list(w.reasons)}) for w in stats])),
            "candidate_frames": {str(w.k): {"frame_indices_bag": [w.k * 10 + i for i in range(8)],
                                            "frame_times_utc": [w.t0 + 2.0 * i for i in range(8)]}
                                 for w in stats if w.frames_matched}}
        back = s5.stats_from_census(rec)
        assert [b.k for b in back] == [w.k for w in stats] and math.isnan(back[-1].S)
        assert back[0] == stats[0]
        records[seq] = rec
    wins, n_sv = s5.pooled_windows(records, POOL)
    assert n_sv == 6 and len(wins) == 8
    for w in wins:
        assert w["window_id"] == f"{w['sequence']}:k{w['k']}"
        assert w["frame_indices_bag"][0] == w["k"] * 10


# ------------------------------- reference sub-bag -----------------------------
def test_reference_subbag_copies_raw_bytes_only(tmp_path):
    from rosbags.rosbag1 import Reader, Writer
    from rosbags.typesys import Stores, get_typestore

    store = get_typestore(Stores.ROS1_NOETIC)
    NavSatFix = store.types["sensor_msgs/msg/NavSatFix"]
    Header = store.types["std_msgs/msg/Header"]
    Time = store.types["builtin_interfaces/msg/Time"]
    Status = store.types["sensor_msgs/msg/NavSatStatus"]
    UInt8 = store.types["std_msgs/msg/UInt8"]
    src = tmp_path / "src.bag"
    raws = {}
    with Writer(src) as wr:
        c1 = wr.add_connection("/dji_osdk_ros/rtk_position", "sensor_msgs/msg/NavSatFix", typestore=store)
        c2 = wr.add_connection("/dji_osdk_ros/rtk_info_position", "std_msgs/msg/UInt8", typestore=store)
        c3 = wr.add_connection("/left_camera/image/compressed", "std_msgs/msg/UInt8", typestore=store)
        for i in range(5):
            m = NavSatFix(Header(i, Time(100 + i, 0), "gps"), Status(0, 1), 22.3 + i, 114.2, 30.0,
                          np.zeros(9), 0)
            raw = store.serialize_ros1(m, "sensor_msgs/msg/NavSatFix")
            raws.setdefault("pos", []).append(bytes(raw))
            wr.write(c1, (100 + i) * 10**9 + 7, raw)
            raw2 = store.serialize_ros1(UInt8(50), "std_msgs/msg/UInt8")
            wr.write(c2, (100 + i) * 10**9 + 9, raw2)
            wr.write(c3, (100 + i) * 10**9, raw2)
    dst = tmp_path / "ref.bag"
    counts = s5.copy_reference_topics(src, dst)
    assert counts == {"/dji_osdk_ros/rtk_position": 5, "/dji_osdk_ros/rtk_info_position": 5}
    with Reader(dst) as rd:
        topics = {c.topic for c in rd.connections}
        assert topics == set(s5.REF_TOPICS)  # camera topic not copied
        got = [(rec, bytes(raw)) for c, rec, raw in rd.messages() if c.topic == "/dji_osdk_ros/rtk_position"]
    assert [g[1] for g in got] == raws["pos"]  # byte-identical
    assert [g[0] for g in got] == [(100 + i) * 10**9 + 7 for i in range(5)]  # record time preserved
    # the scorer's reference reader works on the sub-bag unchanged (recorder clock: UTC = header + delta)
    ref = score_mod.read_reference(dst, "recorder", 0.41)
    assert np.allclose(ref["pos_t"], 100 + np.arange(5) + 0.41) and np.allclose(ref["info_t"], 100 + np.arange(5) + 9e-9 + 0.41)
    assert ref["info_status"].tolist() == [50] * 5
    bad = tmp_path / "bad.bag"
    with Writer(bad) as wr:
        wr.add_connection("/dji_osdk_ros/rtk_position", "sensor_msgs/msg/NavSatFix", typestore=store)
    with pytest.raises(RuntimeError, match="reference topics absent"):
        s5.copy_reference_topics(bad, tmp_path / "x.bag")


def test_refbag_never_deserializes():
    src = (ROOT / "scripts/cross_backbone_v5_stage1.py").read_text(encoding="utf-8")
    body = src.split("def copy_reference_topics", 1)[1].split("\ndef ", 1)[0]
    assert "deserialize" not in body.replace("no deserialization", "").replace("Never deserializes", "")


# ------------------------------- pooled scoring --------------------------------
def _pooled_synthetic():
    """Two sequences with different tracks; each window must use ITS OWN sequence's cue + reference."""
    wins_a, t_a, llh_a, ref_a, preds_a = tp._synthetic(drop_frames_win1=(), drop_frames_win2=())
    t_b = t_a + 5000.0
    enu_b = np.stack([np.linspace(0, 900, t_a.size), np.zeros(t_a.size), np.zeros(t_a.size)], axis=1)
    llh_b = tp._llh(enu_b[:, 0], enu_b[:, 1], enu_b[:, 2])
    ref_b = {"pos_t": t_b, "lat": llh_b[:, 0], "lon": llh_b[:, 1], "alt": llh_b[:, 2], "info_t": t_b,
             "info_status": np.full(t_b.size, 50)}
    wins = []
    for w in wins_a[:2]:
        wins.append(dict(w, sequence="HKisland_GNSS02"))
    for w in wins_a[:2]:
        ft = [x + 5000.0 for x in w["frame_times_utc"]]
        wins.append(dict(w, sequence="HKisland_GNSS03", frame_times_utc=ft))
    preds = {}
    for bb, sc in (("da3-base", 0.4), ("vggt-1b", 0.55)):
        preds[bb] = {}
        for i, w in enumerate(wins):
            ft = np.asarray(w["frame_times_utc"])
            tt, ee = (t_a, None) if w["sequence"] == "HKisland_GNSS02" else (t_b, enu_b)
            if ee is None:
                p = preds_a[bb][i].centers() / sc
            else:
                p = np.stack([np.interp(ft, tt, ee[:, c]) for c in range(3)], axis=1)
            preds[bb][i] = tp._pred(p, sc, ft)
    cue_t = score_mod.per_sequence({"HKisland_GNSS02": t_a, "HKisland_GNSS03": t_b})
    cue_llh = score_mod.per_sequence({"HKisland_GNSS02": llh_a, "HKisland_GNSS03": llh_b})
    ref = score_mod.per_sequence({"HKisland_GNSS02": ref_a, "HKisland_GNSS03": ref_b})
    return wins, cue_t, cue_llh, ref, preds


def test_pooled_score_core_uses_each_sequence_cue_reference_and_prior():
    wins, cue_t, cue_llh, ref, preds = _pooled_synthetic()
    loaders = {bb: (lambda i, bb=bb: preds[bb][i]) for bb in preds}
    sections, ref_log = score_mod.score_core(wins, cue_t, cue_llh, ref, loaders)
    assert all(L["evaluable"] for L in ref_log) and [L["sequence"] for L in ref_log][2] == "HKisland_GNSS03"
    for bb in preds:
        rows = sections["sv_test"][bb]["rows"]
        assert [r["sequence"] for r in rows] == ["HKisland_GNSS02"] * 2 + ["HKisland_GNSS03"] * 2
        for r in rows:  # cue == reference per sequence => grounded collapses; wrong-sequence pairing would not
            assert r["eval_naive"]["grounded_ate_se3"] < 0.05 * r["eval_none"]["raw_ate_se3"]
            assert r["three_limits"]["Lg_over_Lt"] == pytest.approx(1.0, rel=1e-3)
            assert r["nominal_speed_prior"]["v_cruise_mps"] == v2.V_CRUISE_TABLE_BY_SEQ[r["sequence"]]


def test_seq_sources_v5_manifest(tmp_path):
    wins, *_ = _pooled_synthetic()
    man = {"sequences": {
        "HKisland_GNSS02": {"gate": {"classes": {score_mod.REF_TOPIC: "recorder"}, "recorder_offset_s": 0.41},
                            "reference_source": {"kind": "sealed_sub_bag", "path": "ref_HKisland_GNSS02.bag", "sha256": "a" * 64}},
        "HKisland_GNSS03": {"gate": {"classes": {score_mod.REF_TOPIC: "recorder"}, "recorder_offset_s": -0.1},
                            "reference_source": {"kind": "local_bag", "path": "data/mars_lvig/HKisland_GNSS03.bag", "sha256": "b" * 64}}}}
    src = score_mod.seq_sources(tmp_path, man, wins, Path("unused"))
    assert src["HKisland_GNSS02"] == (tmp_path / "ref_HKisland_GNSS02.bag", "recorder", 0.41, "a" * 64)
    assert src["HKisland_GNSS03"][0] == ROOT / "data/mars_lvig/HKisland_GNSS03.bag" and src["HKisland_GNSS03"][2] == -0.1


def test_pooled_score_end_to_end_awaits_then_refuses_replay(tmp_path):
    wins, *_ = _pooled_synthetic()
    s1 = tmp_path / "stage1"
    s1.mkdir()
    (s1 / "stage1_manifest.json").write_text(json.dumps({"contract_sha256": v2.CONTRACT_SHA, "sequences": {},
                                                         "reference_values_read": False}))
    (s1 / "windows.json").write_text(json.dumps({"windows": wins}))
    called = []
    assert score_mod.score(out=tmp_path, pack=tmp_path / "pack", reference_reader=lambda *a: called.append(a)) == 3
    assert not called and not (tmp_path / "SCORING_RECEIPT.json").exists()


# ------------------------------- pack / DA3 helpers ----------------------------
def test_pack_builds_multi_sequence(tmp_path):
    wins, *_ = _pooled_synthetic()
    src = tmp_path / "stage1"
    for w in wins:
        d = src / "frames" / w["sequence"]
        d.mkdir(parents=True, exist_ok=True)
        for k in w["frame_indices_bag"]:
            (d / f"{k:08d}.png").write_bytes(b"\x89PNG-%s-%d" % (w["sequence"].encode(), k))
    (src / "stage1_manifest.json").write_text(json.dumps({"contract_sha256": v2.CONTRACT_SHA}))
    (src / "windows.json").write_text(json.dumps({"windows": wins}))
    pack = tmp_path / "pack"
    assert pack_mod.build(src=src, pack=pack) == 0
    pv = score_mod._load_pack_verify(pack)
    entries = [e for e, _ in pv.verify_inputs(pack)]
    assert entries == [score_mod.w_entry(w) for w in wins]
    man = json.loads((pack / "input_hashes.json").read_text())
    assert man["sequences"] == ["HKisland_GNSS02", "HKisland_GNSS03"] and man["pack_id"] == "vggt_offbox_cross_backbone_v5"
    lock = json.loads((pack / "model_lock.json").read_text())
    assert lock["sequence"] == "HKisland_GNSS02+HKisland_GNSS03" and lock["contract_sha256"] == v2.CONTRACT_SHA
    blob = "".join(p.read_text() for p in pack.rglob("*.json") if p.name != "model_lock.json")
    assert not any(tok in blob for tok in ("prospective_test", "cc_control", "validation", "window_id", '"S"'))
    # the same frame bytes as Stage 1 (sequence-namespaced source)
    e0 = entries[2]
    assert (pack / e0 / "f00.png").read_bytes() == (src / "frames" / v2.frame_relpath(wins[2], wins[2]["frame_indices_bag"][0])).read_bytes()


def test_da3_cue_for_uses_sequence_csv(tmp_path):
    (tmp_path / "cue_HKisland_GNSS03.csv").write_text("t_utc,lat_deg,lon_deg,alt_m\n5.0,22.3,114.2,30.0\n")
    (tmp_path / "cue_receiver_lla.csv").write_text("t_utc,lat_deg,lon_deg,alt_m\n1.0,22.3,114.2,30.0\n")
    cache = {}
    assert da3_mod.cue_for(tmp_path, {"sequence": "HKisland_GNSS03"}, cache)[0].tolist() == [5.0]
    assert da3_mod.cue_for(tmp_path, {}, cache)[0].tolist() == [1.0]
    assert v2.frame_relpath({"sequence": "S"}, 7) == "S/00000007.png" and v2.frame_relpath({}, 7) == "00000007.png"
