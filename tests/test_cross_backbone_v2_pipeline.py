"""Cross-backbone v2 pipeline tests (DA3 script, VGGT pack builder, one-shot scorer, gates 6-8).

Synthetic fixtures only: no bag, no GT, no network, no GPU.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from yardstick3d.datasets import v2_scoring as v2
from yardstick3d.types import PredictionBundle

ROOT = Path(__file__).resolve().parents[1]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


score_mod = _load("cb_v2_score_t", "scripts/cross_backbone_v2_score.py")
pack_mod = _load("cb_v2_pack_t", "scripts/build_vggt_pack_v2.py")
gate_mod = _load("cb_v2_gate_t", "scripts/claim1_gate_metrics_v2.py")
da3_mod = _load("cb_v2_da3_t", "scripts/cross_backbone_v2_da3.py")

T0 = 1_700_000_000.0
LAT0, LON0 = 22.3, 114.2


# ------------------------------- synthetic data -----------------------------
def _llh(e, n, u):
    lat = LAT0 + n / 111_320.0
    lon = LON0 + e / (111_320.0 * np.cos(np.radians(LAT0)))
    return np.stack([lat, lon, 30.0 + u], axis=1)


def _track(dur=140.0, dt=0.1):
    """Speed-varying track (3..9 m/s, mean 6): returns (t_utc, enu[E,N,U])."""
    t = np.arange(0.0, dur, dt)
    v = 6.0 + 3.0 * np.sin(2 * np.pi * t / 16.0)
    e = np.concatenate([[0.0], np.cumsum(v[:-1] * dt)])
    n = 20.0 * np.sin(t / 10.0)
    return T0 + t, np.stack([e, n, np.zeros_like(t)], axis=1)


def _windows(starts_roles):
    wins = []
    for t0, role, cls in starts_roles:
        ideal = T0 + t0 + np.linspace(0.0, 16.0, 8)
        wins.append({"t0": float(t0), "k": int(t0 // 16), "role": role, "cls": cls, "S": 0.2,
                     "frame_times_utc": [float(x) for x in ideal],
                     "frame_indices_sorted": [int(10 * (t0 + 16 * i / 7)) for i in range(8)],
                     "frame_indices_bag": [int(10 * (t0 + 16 * i / 7)) for i in range(8)]})
    return wins


def _pred(enu_at_frames, scale, timestamps):
    c = np.asarray(enu_at_frames, float) * scale
    T = np.tile(np.eye(4)[:3], (len(c), 1, 1))
    T[:, :, 3] = -c  # R = I  =>  C = -t
    return PredictionBundle(timestamps=np.asarray(timestamps, float), T_w2c=T, camera_centers=c)


def _synthetic(drop_frames_win1=(0, 1), drop_frames_win2=(0,)):
    t, enu = _track()
    llh = _llh(enu[:, 0], enu[:, 1], enu[:, 2])
    wins = _windows([(16, "prospective_test", "SV"), (48, "prospective_test", "SV"),
                     (80, "validation", "SV"), (112, "cc_control", "CC")])
    # reference: same track at 10 Hz; fix status 50 except near dropped frames of selected windows
    st = np.full(t.size, 50)
    for wi, drops in ((1, drop_frames_win1), (2, drop_frames_win2)):
        for f in drops:
            tf = wins[wi]["frame_times_utc"][f]
            st[np.abs(t - tf) < 0.2] = 34
    ref = {"pos_t": t, "lat": llh[:, 0], "lon": llh[:, 1], "alt": llh[:, 2], "info_t": t, "info_status": st}
    preds = {}
    for bb, sc in (("da3-base", 0.4), ("vggt-1b", 0.55)):
        preds[bb] = {}
        for i, w in enumerate(wins):
            ft = np.asarray(w["frame_times_utc"])
            p = np.stack([np.interp(ft, t, enu[:, c]) for c in range(3)], axis=1)
            preds[bb][i] = _pred(p, sc, ft)
    return wins, t, llh, ref, preds


# ---------------------------------- contract --------------------------------
def test_contract_sha_check():
    assert v2.sha256_file(v2.CONTRACT_PATH) == v2.CONTRACT_SHA
    assert v2.check_contract() == v2.CONTRACT_SHA  # incl. registry membership
    assert v2.CONTRACT_SHA.startswith("79d67798")


def test_contract_tamper_rejected(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_bytes(v2.CONTRACT_PATH.read_bytes() + b"\n# tamper\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        v2.check_contract(p, v2.CONTRACT_SHA, None)
    reg = tmp_path / "reg.json"
    reg.write_text("{}")
    with pytest.raises(ValueError, match="not recorded"):
        v2.check_contract(v2.CONTRACT_PATH, v2.CONTRACT_SHA, reg)


def test_frozen_constants_match_contract_text():
    text = v2.CONTRACT_PATH.read_text(encoding="utf-8")
    assert "max_dt_match_camera_ref_s: 0.15" in text and v2.MAX_DT_REF_S == 0.15
    assert "fixed_values: [50]" in text and v2.FIXED_VALUES == (50,)
    assert "max_dt_match_camera_cue_s: 0.06" in text and v2.MAX_DT_CUE_S == 0.06
    assert "ref_valid_frames_required: 7" in text and v2.REF_VALID_REQUIRED == 7
    assert "HKisland_GNSS02" in text and v2.SEQUENCE == "HKisland_GNSS02"
    assert "6 m/s" in text and v2.V_CRUISE_TABLE_MPS == 6.0


# ------------------------------- replay refusal -----------------------------
def test_replay_refused_when_receipt_exists(tmp_path):
    (tmp_path / "SCORING_RECEIPT.json").write_text("{}")
    with pytest.raises(FileExistsError):
        score_mod.score(out=tmp_path, pack=tmp_path / "pack")
    with pytest.raises(FileExistsError):
        v2.guard_receipt(tmp_path / "SCORING_RECEIPT.json")
    v2.guard_receipt(tmp_path / "absent.json")


def _stage1_dir(out: Path, wins):
    s1 = out / "stage1"
    s1.mkdir(parents=True)
    (s1 / "stage1_manifest.json").write_text(json.dumps(
        {"contract_sha256": v2.CONTRACT_SHA, "reference_values_read": False,
         "gate": {"classes": {score_mod.REF_TOPIC: "utc"}}, "frame_sha256": {}}))
    (s1 / "windows.json").write_text(json.dumps({"windows": wins}))
    return s1


def test_score_refuses_unless_both_arms_present(tmp_path):
    wins, *_ = _synthetic()
    _stage1_dir(tmp_path, wins)
    called = []
    rc = score_mod.score(out=tmp_path, pack=tmp_path / "pack",
                         reference_reader=lambda *a: called.append(1))
    assert rc == 3 and not called  # reference never opened
    assert not (tmp_path / "SCORING_RECEIPT.json").exists()
    assert not (tmp_path / "rho_diagnostics.json").exists()
    assert not (tmp_path / "paired_results.json").exists()


def test_verify_arms_requires_vggt_even_if_da3_present(tmp_path):
    wins, *_ = _synthetic()
    (tmp_path / "da3_manifest.json").write_text(json.dumps(
        {"contract_sha256": v2.CONTRACT_SHA, "contains_ground_truth": False, "windows": []}))
    with pytest.raises(score_mod.ArmsNotReady):
        score_mod.verify_arms(tmp_path, tmp_path / "pack", wins)


# --------------------------------- pack builder -----------------------------
FORBIDDEN_SUBSTR = ("gt", "ground_truth_", "rtk", "reference", "cue", "ublox", "lla", "enu", "cls")


def _fake_stage1(tmp_path, wins):
    src = tmp_path / "stage1"
    (src / "frames").mkdir(parents=True)
    (src / "stage1_manifest.json").write_text(json.dumps({"contract_sha256": v2.CONTRACT_SHA}))
    for w in wins:
        for k in w["frame_indices_bag"]:
            (src / "frames" / f"{k:08d}.png").write_bytes(b"\x89PNG-fake-%d" % k)
    (src / "windows.json").write_text(json.dumps({"windows": wins}))
    return src


def test_pack_builds_and_contains_no_gt_or_cue(tmp_path):
    wins, *_ = _synthetic()
    src = _fake_stage1(tmp_path, wins)
    pack = tmp_path / "pack"
    assert pack_mod.build(src=src, pack=pack) == 0
    pv = score_mod._load_pack_verify(pack)
    windows = pv.verify_inputs(pack)  # frozen-v1 verify semantics
    assert len(windows) == 4
    manifest = json.loads((pack / "input_hashes.json").read_text())
    assert manifest["pack_id"] == "vggt_offbox_cross_backbone_v5"
    assert manifest["contains_ground_truth"] is False and manifest["sequences"] == ["HKisland_GNSS02"]
    lock = json.loads((pack / "model_lock.json").read_text())
    assert lock["contract_sha256"] == v2.CONTRACT_SHA and lock["pack_id"] == v2.PACK_ID
    v1lock = json.loads((pack_mod.TEMPLATE_PACK / "model_lock.json").read_text())
    for k in ("source_url", "source_revision", "model_id", "model_revision", "checkpoint",
              "checkpoint_sha256", "checkpoint_bytes", "preprocess", "compute_dtype", "seed"):
        assert lock[k] == v1lock[k]
    # runner files are byte-identical to v1
    for n in pack_mod.TEMPLATE_FILES:
        assert (pack / n).read_bytes() == (pack_mod.TEMPLATE_PACK / n).read_bytes()
    # no GT / cue / reference / role material anywhere in the pack
    names = [p.name.lower() for p in pack.rglob("*") if p.is_file()]
    assert not any(n.endswith((".csv", ".bag")) for n in names)
    keys = set()
    for p in pack.rglob("*.json"):
        if p.name == "model_lock.json":
            continue
        def walk(o):
            if isinstance(o, dict):
                for k, v in o.items():
                    keys.add(str(k).lower())
                    walk(v)
            elif isinstance(o, list):
                for v in o:
                    walk(v)
        walk(json.loads(p.read_text()))
    bad = {k for k in keys for s in FORBIDDEN_SUBSTR if s in k and "images" not in k}
    assert not bad, bad
    blob = "".join(p.read_text() for p in pack.rglob("*.json") if p.name != "model_lock.json")
    assert not any(tok in blob for tok in ("prospective_test", "cc_control", "validation", "SV"))
    for rec in (r for _, r in windows):
        assert set(rec) <= {"t0", "t1", "timestamps", "frame_indices", "images", "pose_convention", "is_metric"}
    # timestamps in the pack are the Stage-1 UTC frame times
    for (entry, rec), w in zip(windows, wins):
        assert np.allclose(rec["timestamps"], w["frame_times_utc"])
        assert entry == score_mod.pack_entry(w["t0"])


def test_pack_refuses_rebuild_and_wrong_stage1_contract(tmp_path):
    wins, *_ = _synthetic()
    src = _fake_stage1(tmp_path, wins)
    pack = tmp_path / "pack"
    pack_mod.build(src=src, pack=pack)
    with pytest.raises(SystemExit):
        pack_mod.build(src=src, pack=pack)
    (src / "stage1_manifest.json").write_text(json.dumps({"contract_sha256": "0" * 64}))
    with pytest.raises(SystemExit):
        pack_mod.build(src=src, pack=tmp_path / "pack2")


def test_model_lock_pins_must_match_contract():
    v1lock = json.loads((pack_mod.TEMPLATE_PACK / "model_lock.json").read_text())
    bad = dict(v1lock, source_revision="f" * 40)
    with pytest.raises(RuntimeError):
        pack_mod.build_model_lock(bad, v2.CONTRACT_PATH.read_text(encoding="utf-8"), v2.CONTRACT_SHA)


# --------------------------------- REF-VALID rule ---------------------------
def _ref_arrays(ft, status=50, ref_offset=0.0, info_offset=0.0):
    t = np.arange(ft[0] - 2, ft[-1] + 2, 0.05)
    st = np.full(t.size, status) if np.isscalar(status) else status(t)
    return t + ref_offset, t + info_offset, st


def test_ref_valid_all_fixed():
    ft = np.linspace(100, 116, 8)
    rt, it, st = _ref_arrays(ft)
    r = v2.ref_valid_frames(ft, rt, it, st)
    assert r["mask"].all() and r["n_fixed"] == 8 and r["n_matched"] == 8
    assert r["max_abs_dt_s"] <= 0.025 + 1e-9
    assert v2.window_evaluable(r["mask"])


def test_ref_valid_status_and_dt_rules():
    ft = np.linspace(100, 116, 8)
    rt, it, _ = _ref_arrays(ft)
    for bad_status in (0, 16, 34):  # anything except 50 is not RTK fixed
        r = v2.ref_valid_frames(ft, rt, it, np.full(it.size, bad_status))
        assert r["n_fixed"] == 0 and r["n_matched"] == 8 and not v2.window_evaluable(r["mask"])
    # reference offset by > 0.15 s => unmatched
    rt2, it2, st2 = _ref_arrays(ft, ref_offset=0.5)
    # samples still exist but 0.5 s shifted: nearest within 0.05 grid => dt ~0.5 mod... use sparse ref
    sparse = np.array([100.0, 104.0, 108.0, 112.0, 116.0])
    r = v2.ref_valid_frames(ft, sparse, it, np.full(it.size, 50))
    assert r["n_matched"] < 8  # frames ~2.29 s apart from samples fail the 0.15 s match
    # status stream missing near a frame => that frame not REF-VALID even though position matches
    holes = np.concatenate([it[it < ft[3] - 0.5], it[it > ft[3] + 0.5]])
    r = v2.ref_valid_frames(ft, rt, holes, np.full(holes.size, 50))
    assert not r["mask"][3] and r["n_fixed"] == 7 and v2.window_evaluable(r["mask"])


def test_ref_valid_seven_of_eight_boundary():
    ft = np.linspace(100, 116, 8)
    rt, it, _ = _ref_arrays(ft)
    one = np.where(np.abs(it - ft[2]) < 0.2, 34, 50)
    two = np.where((np.abs(it - ft[2]) < 0.2) | (np.abs(it - ft[5]) < 0.2), 34, 50)
    r7 = v2.ref_valid_frames(ft, rt, it, one)
    r6 = v2.ref_valid_frames(ft, rt, it, two)
    assert r7["n_fixed"] == 7 and v2.window_evaluable(r7["mask"])
    assert r6["n_fixed"] == 6 and not v2.window_evaluable(r6["mask"])


def test_status_and_stamp_helpers():
    class M:
        data = 50
    assert score_mod.decode_status(M()) == 50
    with pytest.raises(RuntimeError):
        score_mod.decode_status(object())
    h = np.array([118.0, 119.0])
    r = np.array([100.0, 101.0])
    t, cls = score_mod.to_utc_stamps(h, r)
    assert cls == "gpst" and np.allclose(t, r)  # +18 s published constant, never fitted
    with pytest.raises(RuntimeError):
        score_mod.to_utc_stamps(h + 5.0, r)
    t, cls = score_mod.to_utc_stamps(None, r)
    assert cls == "record_time+recorder_offset" and np.allclose(t, r)
    t, cls = score_mod.to_utc_stamps(None, r, None, 0.41)  # v4 A2: header-less recorder topic
    assert np.allclose(t, r + 0.41)
    t, cls = score_mod.to_utc_stamps(r, r, "recorder", -0.588)
    assert cls == "recorder" and np.allclose(t, r - 0.588)


# ------------------------------- rho / nominal prior ------------------------
def test_rho_diagnostic_and_nominal_prior():
    from yardstick3d.datasets import mars_lvig as ml

    wins, t, llh, ref, preds = _synthetic()
    w = wins[0]
    ft = np.asarray(w["frame_times_utc"])
    enu = ml.cue_enu_from_llh(llh[:, 0], llh[:, 1], llh[:, 2], *v2.window_origin(t, llh, ft[0]))
    rho = v2.rho_diagnostic(preds["da3-base"][0].centers(), ft, t, enu)
    assert rho["n_chords"] == 7
    assert rho["rho_tau_0"] == pytest.approx(1.0)
    assert "rho_tau_-18" in rho and "rho_tau_+18" in rho
    # constant-speed sanity of the prior formula: 6 m/s * 16 s / sum(a)
    assert v2.nominal_speed_prior_scale(48.0) == pytest.approx(2.0)
    assert v2.nominal_speed_prior_scale(0.0) is None
    assert v2.spearman([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert v2.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert v2.spearman([1, 1, 1, 1], [1, 2, 3, 4]) is None


# ---------------------------------- scoring core ----------------------------
def test_score_core_sections_ref_valid_and_gates():
    wins, t, llh, ref, preds = _synthetic()
    loaders = {bb: (lambda i, bb=bb: preds[bb][i]) for bb in preds}
    sections, ref_log = score_mod.score_core(wins, t, llh, ref, loaders)
    assert set(sections) == {"sv_test", "validation", "cc_control"}
    assert [L["evaluable"] for L in ref_log] == [True, False, True, True]  # win1: 6/8, win2: 7/8
    for bb in ("da3-base", "vggt-1b"):
        rows = sections["sv_test"][bb]["rows"]
        assert "unavailable" in rows[1] and "eval_naive" in rows[0]  # excluded, not replaced
        assert len(sections["validation"][bb]["rows"]) == 1 and "eval_naive" in sections["validation"][bb]["rows"][0]
        assert len(sections["cc_control"][bb]["rows"]) == 1
        val = sections["validation"][bb]["rows"][0]
        assert val["n_ref_valid"] == 7  # ATE on REF-VALID subset only
        r0 = rows[0]
        assert r0["n_ref_valid"] == 8
        # cue equals reference => naive scale recovers 1/pred_scale and grounded ATE collapses vs raw
        assert r0["eval_naive"]["grounded_ate_se3"] < 0.05 * r0["eval_none"]["raw_ate_se3"]
        assert r0["three_limits"]["Lg_over_Lt"] == pytest.approx(1.0, rel=1e-3)
        assert r0["nominal_speed_prior"]["s_prior"] > 0
        assert r0["nominal_speed_prior"]["grounded_ate_se3"] is not None
        assert sections["sv_test"][bb]["summary"]["n_scored"] == 1
    # identical REF-VALID exclusion for both backbones
    assert [("unavailable" in r) for r in sections["sv_test"]["da3-base"]["rows"]] == \
           [("unavailable" in r) for r in sections["sv_test"]["vggt-1b"]["rows"]]
    # gates use SV test windows only; excluded window disclosed; validation/CC never enter
    g = gate_mod.gates({"sections": json.loads(json.dumps(sections, default=float))})
    for bb, blk in g.items():
        assert blk["n_test"] == 2 and blk["n_scored"] == 1
        assert [e["window"] for e in blk["excluded_ref_valid"]] == [1]
        assert all(r["role"] == "prospective_test" for r in blk["windows"])
        assert blk["n_evaluable_test"] == 1
        assert blk["gate6_pass"] is True and blk["gate7_pass"] is True


# ---------------------------------- gate metrics ----------------------------
def _row(idx, raw, gr, orc, role="prospective_test"):
    return {"window": idx, "role": role, "cue": {"n_kept": 7, "n_dropped": 0},
            "solver": {"naive": 1.0}, "three_limits": {"s_cue": 1.0},
            "eval_naive": {"raw_ate_se3": raw, "grounded_ate_se3": gr, "oracle_ate_se3": orc,
                           "raw_ate_sim3": orc, "oracle_scale": 1.0, "scale_error_rel_vs_oracle": 0.0}}


def _paired(rows_a, rows_b, extra=None):
    sections = {"sv_test": {"da3-base": {"rows": rows_a}, "vggt-1b": {"rows": rows_b}}}
    sections.update(extra or {})
    return {"sections": sections}


def test_gate_metrics_pass_fail_and_thresholds():
    good = [_row(0, 10.0, 2.5, 2.0), _row(1, 8.0, 2.4, 2.0), _row(2, 12.0, 3.0, 2.0), _row(3, 9.0, 2.6, 2.0)]
    poor = [_row(0, 10.0, 9.0, 2.0), _row(1, 8.0, 7.5, 2.0), _row(2, 12.0, 11.0, 2.0), _row(3, 9.0, 8.0, 2.0)]
    g = gate_mod.gates(_paired(good, poor))
    assert g["da3-base"]["gate6_pass"] and g["da3-base"]["gate7_pass"]
    assert not g["vggt-1b"]["gate6_pass"] and not g["vggt-1b"]["gate7_pass"]
    a = g["da3-base"]
    assert a["median_capture_R"] == pytest.approx(np.median([(10 - 2.5) / 8, (8 - 2.4) / 6, (12 - 3) / 10, (9 - 2.6) / 7]))
    assert a["median_grounded_ate"] == pytest.approx(np.median([2.5, 2.4, 3.0, 2.6]))
    assert a["grounded_over_oracle"] == pytest.approx(a["median_grounded_ate"] / 2.0)


def test_gate_metrics_shape_limited_and_ref_valid_exclusion_and_scope():
    rows = [_row(0, 10.0, 2.5, 2.0), _row(1, 2.1, 2.05, 2.0),  # leverage 0.1 < 0.25 => shape-limited
            {"window": 2, "role": "prospective_test", "unavailable": "REF-VALID frames 6/8 < 7"},
            _row(3, 9.0, 2.6, 2.0)]
    other = {"validation": {"da3-base": {"rows": [_row(9, 50.0, 49.0, 2.0, "validation")]},
                            "vggt-1b": {"rows": []}},
             "cc_control": {"da3-base": {"rows": [_row(8, 50.0, 49.0, 2.0, "cc_control")]}, "vggt-1b": {"rows": []}}}
    g = gate_mod.gates(_paired(rows, rows, other))["da3-base"]
    assert g["n_test"] == 4 and g["n_scored"] == 3 and g["n_evaluable_test"] == 2
    assert g["shape_limited_test_windows"] == [1]
    assert [e["window"] for e in g["excluded_ref_valid"]] == [2]
    assert {r["window"] for r in g["windows"]} == {0, 1, 3}  # validation/CC rows never in gates
    assert g["gate6_pass"] is True
    # no evaluable window => gates fail, no crash
    none = gate_mod.gates(_paired([_row(0, 2.1, 2.05, 2.0)], []))["da3-base"]
    assert none["gate6_pass"] is False and none["gate7_pass"] is False and none["median_capture_R"] is None
    # null metrics (non-finite) are counted as non-evaluable, never substituted
    nul = _row(0, None, None, None)
    assert gate_mod.gates(_paired([nul], []))["da3-base"]["n_evaluable_test"] == 0


# ---------------------------------- DA3 helpers -----------------------------
def test_da3_stage1_check_and_cue_loader(tmp_path):
    s1 = tmp_path / "stage1"
    (s1 / "frames").mkdir(parents=True)
    (s1 / "frames" / "00000001.png").write_bytes(b"x")
    good = {"contract_sha256": v2.CONTRACT_SHA, "reference_values_read": False,
            "frame_sha256": {"00000001.png": v2.sha256_file(s1 / "frames" / "00000001.png")}}
    (s1 / "stage1_manifest.json").write_text(json.dumps(good))
    (s1 / "windows.json").write_text(json.dumps({"windows": []}))
    assert da3_mod.check_stage1(s1) == {"windows": []}
    (s1 / "frames" / "00000001.png").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="frame hash"):
        da3_mod.check_stage1(s1)
    (s1 / "stage1_manifest.json").write_text(json.dumps(dict(good, contract_sha256="0" * 64)))
    with pytest.raises(RuntimeError, match="contract sha"):
        da3_mod.check_stage1(s1)
    (s1 / "cue.csv").write_text("t_utc,lat_deg,lon_deg,alt_m\n1.0,22.3,114.2,30.0\n1.1,22.3001,114.2,30.0\n")
    t, llh = da3_mod.load_cue(s1 / "cue.csv")
    assert t.tolist() == [1.0, 1.1] and llh.shape == (2, 3)
