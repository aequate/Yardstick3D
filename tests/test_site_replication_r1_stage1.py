"""Site replication R1 - Stage 0/1 on SYNTHETIC rosbags written in tmp (no MARS-LVIG byte is read).

Synthetic flight: 400 s; camera header = UTC (hardware), LLA header = GPST (UTC + 18 s), rtk header = recorder
clock, recorder clock = UTC - 0.4 s; receiver_pvt carries GNSS week/tow (nested GnssTimeMsg) + NED velocity.
Sequence A: PVT on UTC (gate PASS). Sequence B: PVT 1 s late (gate FAIL -> speed arm NOT RUN, primary unaffected).
Horizontal sine speed for 300 s (SV windows), then a hover-climb at 1 m/s (vertical windows).
"""
from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from yardstick3d.datasets import r1_contract as r1c

ROOT = Path(__file__).resolve().parents[1]
POOL = list(r1c.POOL)
T0 = 1_700_000_000.0
LAT0, LON0 = 22.3, 113.9
REC_LAG = 0.4  # recorder clock = UTC - 0.4 s  => camera delta ~ +0.4 s

PVT_MSGDEF = """GnssTimeMsg time
uint8 fix_type
bool valid_fix
bool diff_soln
uint8 carr_soln
uint8 num_sv
float64 latitude
float64 longitude
float64 altitude
float64 height_msl
float64 h_acc
float64 v_acc
float64 p_dop
float64 vel_n
float64 vel_e
float64 vel_d
float64 vel_acc
================================================================================
MSG: gnss_comm/GnssTimeMsg
uint32 week
float64 tow
"""


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


s1 = _load("r1_stage1_t", "scripts/site_replication_r1_stage1.py")


def _profile(dur=400.0):
    t = np.arange(0.0, dur, 0.1)
    v = np.where(t < 300.0, 3.0 + 2.5 * np.sin(2 * np.pi * t / 40.0), 0.5)
    e = np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * 0.1)])
    u = np.clip(t - 300.0, 0.0, None)  # 1 m/s climb after t = 300 s
    climb = (t >= 300.0).astype(float)
    return t, v, e, u, climb


def write_bag(path: Path, t_base: float, pvt_offset_s: float = 0.0):
    import cv2
    from rosbags.rosbag1 import Writer
    from rosbags.typesys import Stores, get_types_from_msg, get_typestore

    store = get_typestore(Stores.ROS1_NOETIC)
    store.register(get_types_from_msg(PVT_MSGDEF, "gnss_comm/msg/GnssPVTSolnMsg"))
    T = store.types
    Header, Time = T["std_msgs/msg/Header"], T["builtin_interfaces/msg/Time"]
    png = cv2.imencode(".png", np.full((4, 4, 3), 128, np.uint8))[1].ravel()

    def hdr(i, s, fid="f"):
        sec = int(np.floor(s))
        return Header(i, Time(sec, int(round((s - sec) * 1e9)) % 1_000_000_000), fid)

    def ns(s):
        return int(round(s * 1e9))

    t, v, e, u, climb = _profile()
    utc = t_base + t
    lat = LAT0 + 0.0 / 111_320.0 + 0 * e
    lon = LON0 + e / (111_320.0 * np.cos(np.radians(LAT0)))
    gpst = utc + pvt_offset_s + 18.0 - r1c.GPS_EPOCH_UNIX_S
    week = np.floor(gpst / r1c.SECONDS_PER_WEEK)
    tow = gpst - week * r1c.SECONDS_PER_WEEK
    with Writer(path) as wr:
        c_cam = wr.add_connection(r1c.CAMERA_TOPIC, "sensor_msgs/msg/CompressedImage", typestore=store)
        c_lla = wr.add_connection(r1c.LLA_TOPIC, "sensor_msgs/msg/NavSatFix", typestore=store)
        c_rtk = wr.add_connection(r1c.REF_TOPICS[0], "sensor_msgs/msg/NavSatFix", typestore=store)
        c_inf = wr.add_connection(r1c.REF_TOPICS[1], "std_msgs/msg/UInt8", typestore=store)
        c_pvt = wr.add_connection(r1c.PVT_TOPIC, "gnss_comm/msg/GnssPVTSolnMsg", typestore=store)
        Img, Fix, Status, U8 = (T["sensor_msgs/msg/CompressedImage"], T["sensor_msgs/msg/NavSatFix"],
                                T["sensor_msgs/msg/NavSatStatus"], T["std_msgs/msg/UInt8"])
        Pvt, GTime = T["gnss_comm/msg/GnssPVTSolnMsg"], T["gnss_comm/msg/GnssTimeMsg"]
        msgs = []
        for i in range(t.size):
            rec = utc[i] - REC_LAG
            msgs.append((ns(rec + 0.003), c_cam, store.serialize_ros1(Img(hdr(i, utc[i]), "png", png), Img.__msgtype__)))
            fix = Fix(hdr(i, utc[i] + 18.0), Status(0, 1), lat[i], lon[i], 30.0 + u[i], np.zeros(9), 0)
            msgs.append((ns(rec + 0.004), c_lla, store.serialize_ros1(fix, Fix.__msgtype__)))
            if i % 2 == 0:
                rtk = Fix(hdr(i, rec + 0.002), Status(0, 1), lat[i], lon[i], 30.0 + u[i], np.zeros(9), 0)
                msgs.append((ns(rec + 0.002), c_rtk, store.serialize_ros1(rtk, Fix.__msgtype__)))
                msgs.append((ns(rec + 0.006), c_inf, store.serialize_ros1(U8(50), U8.__msgtype__)))
            p = Pvt(GTime(int(week[i]), float(tow[i])), 3, True, False, 0, 12, lat[i], lon[i], 30.0 + u[i],
                    30.0 + u[i], 1.0, 1.5, 1.2, 0.0, float(v[i]), float(-climb[i]), 0.1)
            msgs.append((ns(rec + 0.005), c_pvt, store.serialize_ros1(p, Pvt.__msgtype__)))
        for ts_, conn, raw in sorted(msgs, key=lambda m: m[0]):
            wr.write(conn, ts_, raw)
    return path


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    """Full dry-run Stage 0/1 over the 2-sequence synthetic pool."""
    root = tmp_path_factory.mktemp("r1root")
    bags = {POOL[0]: write_bag(root / "a.bag", T0, 0.0), POOL[1]: write_bag(root / "b.bag", T0 + 20_000.0, 1.0)}
    ctx = s1.Ctx(root, dry=True)
    rc = {}
    for seq, bag in bags.items():  # every pool sequence gets its Stage-0 decision before any census
        rc[("stage0", seq)] = s1.stage0(ctx, seq, bag)
    for seq, bag in bags.items():
        rc[("census", seq)] = s1.census(ctx, seq, bag)
    rc["select"] = s1.select(ctx)
    wj = json.loads((ctx.out / "windows.json").read_text())
    for seq in sorted({w["sequence"] for w in wj["windows"]}):
        rc[("frames", seq)] = s1.frames(ctx, seq, bags[seq])
        rc[("refbag", seq)] = s1.refbag(ctx, seq, bags[seq])
    rc["finalize"] = s1.finalize(ctx)
    return ctx, bags, rc


def _j(ctx, name):
    return json.loads((ctx.out / name).read_text())


def test_stage0_records_bytes_sha_and_topics_without_reading_messages(tmp_path, monkeypatch):
    from rosbags.rosbag1 import Reader

    bag = write_bag(tmp_path / "x.bag", T0)

    def boom(*a, **k):
        raise AssertionError("stage0 read a message")

    monkeypatch.setattr(Reader, "messages", boom)
    ctx = s1.Ctx(tmp_path, dry=True)
    assert s1.stage0(ctx, POOL[0], bag) == 0
    rec = _j(ctx, f"stage0_{POOL[0]}.json")
    assert rec["bag_sha256"] == s1.sha256_file(bag) and rec["bag_bytes"] == bag.stat().st_size
    assert set(r1c.REQUIRED_TOPICS) <= set(rec["topics"]) and rec["missing_required_topics"] == []
    assert rec["messages_deserialized"] is False and rec["dry_run"] is True
    assert rec["topics"][r1c.PVT_TOPIC]["msgtype"] == "gnss_comm/msg/GnssPVTSolnMsg"
    with pytest.raises(SystemExit, match="refusing silent rerun"):
        s1.stage0(ctx, POOL[0], bag)



def test_unavailable_needs_failed_attempts_24h_apart_and_census_needs_all_stage0_decisions(tmp_path, monkeypatch):
    clock = {"t": 1_000_000.0}
    monkeypatch.setattr(s1, "_utcnow", lambda: clock["t"])
    ctx = s1.Ctx(tmp_path, dry=True)
    good = write_bag(tmp_path / "a.bag", T0)
    gone = tmp_path / "b.bag"  # never written: access fails
    assert s1.stage0(ctx, POOL[0], good) == 0
    # census refused until EVERY pool sequence has a Stage-0 decision
    with pytest.raises(SystemExit, match="no Stage-0 decision"):
        s1.census(ctx, POOL[0], good)
    # --unavailable on an accessible bag is refused; on a first failure it is refused (no retry yet)
    with pytest.raises(SystemExit, match="is accessible"):
        s1.stage0(s1.Ctx(tmp_path / "other", dry=True), POOL[1], good, unavailable="quota")
    assert s1.stage0(ctx, POOL[1], gone) == 4  # attempt 1 logged, no decision
    with pytest.raises(SystemExit, match=">= 24 h"):
        s1.stage0(ctx, POOL[1], gone, unavailable="quota")  # attempt 2 at the same time: too early
    clock["t"] += 24 * 3600.0 + 1.0
    assert s1.stage0(ctx, POOL[1], gone, unavailable="Drive quota exceeded on retry") == 0
    u = _j(ctx, f"stage0_UNAVAILABLE_{POOL[1]}.json")
    assert u["status"] == "UNAVAILABLE" and "bag missing" in u["access_failure"] and len(u["attempts"]) == 3
    assert u["attempts"][-1]["utc_s"] - u["attempts"][0]["utc_s"] >= 24 * 3600.0
    # now census may run for the available bag; the UNAVAILABLE sequence is never censused
    assert s1.census(ctx, POOL[0], good) == 0
    with pytest.raises(SystemExit, match="UNAVAILABLE"):
        s1.census(ctx, POOL[1], gone)
    # --unavailable can never be used once any census output exists
    ctx2 = s1.Ctx(tmp_path / "late", dry=True)
    ctx2.out.mkdir(parents=True)
    (ctx2.out / f"census_{POOL[0]}.json").write_text("{}")
    with pytest.raises(SystemExit, match="census output already exists"):
        s1.stage0(ctx2, POOL[1], gone, unavailable="quota")
    # a tampered attempt log invalidates the UNAVAILABLE decision
    with open(ctx.out / f"stage0_attempts_{POOL[1]}.json", "a") as f:
        f.write(" ")
    with pytest.raises(SystemExit, match="attempt log"):
        s1.require_stage0_decisions(ctx)


@pytest.mark.parametrize("name", ["HKairport_GNSS02.bag", "hkairport_gnss03.bag", "copy_of_HKAIRPORT_GNSS02_x.bag"])
def test_dry_run_refuses_bags_named_like_a_pool_sequence(tmp_path, monkeypatch, name):
    import hashlib

    bag = write_bag(tmp_path / name, T0)
    touched = []
    monkeypatch.setattr(s1, "sha256_file", lambda p: touched.append(p) or hashlib.sha256(b"").hexdigest())
    ctx = s1.Ctx(tmp_path / "root", dry=True)
    for call in (lambda: s1.stage0(ctx, POOL[0], bag), lambda: s1.census(ctx, POOL[0], bag),
                 lambda: s1.frames(ctx, POOL[0], bag), lambda: s1.refbag(ctx, POOL[0], bag)):
        with pytest.raises(SystemExit, match="pool sequence"):
            call()
    with pytest.raises(SystemExit, match="pool sequence"):
        ctx.bag(POOL[0], bag)
    with pytest.raises(SystemExit, match="pool sequence"):
        ctx.bag(POOL[1], None)  # default path data/mars_lvig/<SEQ>.bag
    assert touched == [] and not any(ctx.out.glob("stage0_*"))


def test_dry_run_refuses_oversized_bag_and_data_dir(tmp_path, monkeypatch):
    big = tmp_path / "synthetic.bag"
    with open(big, "wb") as f:  # sparse 60 MiB file: refused from stat() alone
        f.seek(60 * 2**20)
        f.write(b"\0")
    ctx = s1.Ctx(tmp_path / "root", dry=True)
    with pytest.raises(SystemExit, match="bytes >"):
        s1.stage0(ctx, POOL[0], big)
    monkeypatch.setattr(s1, "DRY_BAG_MAX_BYTES", 10 * 2**30)
    assert ctx.bag(POOL[0], big) == big
    with pytest.raises(SystemExit, match="under data/"):
        ctx.bag(POOL[0], ROOT / "data" / "mars_lvig" / "synthetic.bag")
    # the same bag is accepted when NOT a dry run (registered contract path): the guard is dry-only
    real = s1.Ctx.__new__(s1.Ctx)
    real.dry, real.root = False, tmp_path
    assert real.bag(POOL[0], tmp_path / f"{POOL[0]}.bag").name == f"{POOL[0]}.bag"


def test_census_gates_pvt_decode_and_speed_arm_gate(run):
    ctx, bags, rc = run
    assert rc[("census", POOL[0])] == 0 and rc[("census", POOL[1])] == 0  # PVT FAIL never stops the primary arm
    a, b = _j(ctx, f"census_{POOL[0]}.json"), _j(ctx, f"census_{POOL[1]}.json")
    for c in (a, b):
        assert c["reference_values_read"] is False and c["gate"]["recorder_offset_s"] == pytest.approx(0.397, abs=1e-3)
        assert c["gate"]["classes"][r1c.LLA_TOPIC] == "gpst" and c["gate"]["classes"][r1c.REF_TOPICS[0]] == "recorder"
    assert a["pvt"]["decoded"] and a["pvt"]["gate"]["status"] == "PASS" and a["pvt"]["speed_arm"] == "RUN"
    assert abs(a["pvt"]["gate"]["median"]) < 0.01
    assert b["pvt"]["gate"]["status"] == "FAIL" and b["pvt"]["speed_arm"] == "NOT_RUN"
    assert b["pvt"]["gate"]["median"] == pytest.approx(1.0, abs=0.01)
    with open(ctx.out / a["pvt"]["csv"], newline="") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == ["record_t", "t_utc", "vel_n", "vel_e", "vel_d"] and len(rows) == 4000
    r0 = rows[100]
    assert float(r0["t_utc"]) - float(r0["record_t"]) == pytest.approx(REC_LAG - 0.005, abs=1e-5)
    with open(ctx.out / a["cue_csv"], newline="") as f:
        cue = list(csv.DictReader(f))
    assert list(cue[0]) == ["t_utc", "lat_deg", "lon_deg", "alt_m"]
    assert float(cue[-1]["alt_m"]) == pytest.approx(30.0 + 99.9, abs=1e-3)
    vc = {v["k"]: v for v in a["vertical_census"]}
    assert vc[20]["dh_window"] == pytest.approx(15.0, abs=0.2) and abs(vc[5]["dh_window"]) < 1e-6


def test_select_primary_minimum_rule_and_vertical_windows(run):
    ctx, _, rc = run
    assert rc["select"] == 0
    wj = _j(ctx, "windows.json")
    wins = wj["windows"]
    roles = [w["role"] for w in wins]
    assert roles.count(r1c.ROLE_TEST) == wj["n_test"] >= 2 and wj["n_test_target"] == 4
    assert wj["minimum_rule_applied"] == (wj["n_test"] < 4)
    assert wj["speed_arm_by_sequence"] == {POOL[0]: "RUN", POOL[1]: "NOT_RUN"}
    vert = [w for w in wins if w["role"] == r1c.ROLE_VERTICAL]
    assert 1 <= len(vert) <= r1c.N_VERTICAL and all(abs(w["dh_window"]) >= 10.0 and 16 * w["k"] >= 300 for w in vert)
    taken = [(w["sequence"], w["k"]) for w in wins]
    assert len(set(taken)) == len(taken)
    for w in vert:
        others = {x for x in taken if x != (w["sequence"], w["k"])}
        assert not any((w["sequence"], w["k"] + d) in others for d in (-1, 1))
    for w in wins:
        assert len(w["frame_times_utc"]) == 8 and len(w["frame_indices_bag"]) == 8
    blob = json.dumps(wj)
    assert "rtk" not in blob.replace("rtk_position_access", "")


def test_frames_refbag_finalize_manifest_hashes_every_output(run):
    from rosbags.rosbag1 import Reader

    ctx, bags, rc = run
    assert rc["finalize"] == 0
    man = _j(ctx, "stage1_manifest.json")
    files = {p.relative_to(ctx.out).as_posix() for p in ctx.out.rglob("*") if p.is_file()} - {"stage1_manifest.json"}
    assert set(man["output_sha256"]) == files
    for name, dg in man["output_sha256"].items():
        assert s1.sha256_file(ctx.out / name) == dg
    assert man["dry_run"] is True and man["reference_values_read"] is False
    for seq in man["sequences_used"]:
        src = man["sequences"][seq]["reference_source"]
        assert src["kind"] == "sealed_sub_bag"
        with Reader(ctx.out / src["path"]) as rd:
            assert {c.topic for c in rd.connections} == set(r1c.REF_TOPICS)
        assert _j(ctx, f"refbag_{seq}.json")["deserialized"] is False
    need = {f"{w['sequence']}/{int(i):08d}.png" for w in man["windows"] for i in w["frame_indices_bag"]}
    assert set(man["frame_sha256"]) == need


def test_refusals_rerun_unregistered_real_out_and_tampered_bag(run, tmp_path):
    ctx, bags, _ = run
    with pytest.raises(SystemExit, match="refusing silent rerun"):
        s1.census(ctx, POOL[0], bags[POOL[0]])
    with pytest.raises(SystemExit, match="refusing silent rerun"):
        s1.select(ctx)
    # an unregistered contract is refused (an empty registry, since the real one now records the freeze)
    with pytest.raises(SystemExit, match="not frozen"):
        s1.Ctx(tmp_path, dry=False, registry=tmp_path / "no_registry.json")
    with pytest.raises(SystemExit, match="real R1 output"):
        s1.Ctx(ROOT, dry=True)
    with pytest.raises(SystemExit, match="not in the R1 pool"):
        s1.stage0(s1.Ctx(tmp_path, dry=True), "HKisland_GNSS02", bags[POOL[0]])
    # a bag whose bytes differ from the Stage-0 record is refused before any decode
    ctx2 = s1.Ctx(tmp_path, dry=True)
    bag = write_bag(tmp_path / "c.bag", T0)
    s1.stage0(ctx2, POOL[0], bag)
    s1.stage0(ctx2, POOL[1], write_bag(tmp_path / "d.bag", T0 + 20_000.0))
    with open(bag, "ab") as f:
        f.write(b"\0")
    with pytest.raises(SystemExit, match="size"):
        s1.census(ctx2, POOL[0], bag)


def test_scorer_and_pack_refuse_dry_run_stage1(run, tmp_path):
    ctx, _, _ = run
    score_mod = _load("r1_score_dry_t", "scripts/site_replication_r1_score.py")
    with pytest.raises(RuntimeError, match="dry run"):
        score_mod._verify_stage1(ctx.out, ctx.sha)
    pack_mod = _load("r1_pack_dry_t", "scripts/build_vggt_pack_site_replication_r1.py")
    with pytest.raises(SystemExit, match="dry-run"):
        pack_mod.build(src=ctx.out, pack=tmp_path / "pack", check=False)
    assert not (tmp_path / "pack").exists()
