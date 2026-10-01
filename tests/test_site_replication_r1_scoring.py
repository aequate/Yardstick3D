"""Site replication R1 - one-shot scorer, verdict functions and VGGT pack (synthetic fixtures only).

The reference is MOCKED (reference_reader); no bag, no data/ file and no v5 artifact is opened. The contract is a
byte-identical tmp copy of the draft, registered in a tmp registry (the real registry is never touched).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from yardstick3d.datasets import r1_contract as r1c
from yardstick3d.datasets import v2_scoring as v2
from yardstick3d.io.prediction_cache import save_prediction
from yardstick3d.types import PredictionBundle

ROOT = Path(__file__).resolve().parents[1]
POOL = list(r1c.POOL)
LAT0, LON0 = 22.3, 113.9
BASE = {POOL[0]: 1_700_000_000.0, POOL[1]: 1_700_050_000.0}
MODEL_ID = "vggt-1b/facebook/VGGT-1B"


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sm = _load("r1_score_t", "scripts/site_replication_r1_score.py")
pack_mod = _load("r1_pack_t", "scripts/build_vggt_pack_site_replication_r1.py")


# ------------------------------------------------------------------ fixtures
def _track():
    t = np.arange(0.0, 200.0, 0.1)
    vh = np.where(t < 120.0, 6.0 + 3.0 * np.sin(2 * np.pi * t / 16.0), 1.0)
    e = np.concatenate([[0.0], np.cumsum(0.5 * (vh[1:] + vh[:-1]) * 0.1)])
    n = 20.0 * np.sin(t / 10.0)
    u = 1.5 * np.clip(t - 120.0, 0.0, None)
    return t, np.stack([e, n, u], axis=1)


def _llh(enu):
    lat = LAT0 + enu[:, 1] / 111_320.0
    lon = LON0 + enu[:, 0] / (111_320.0 * np.cos(np.radians(LAT0)))
    return np.stack([lat, lon, 30.0 + enu[:, 2]], axis=1)


WINDOWS = [(POOL[0], 16, r1c.ROLE_TEST, "SV"), (POOL[0], 48, r1c.ROLE_TEST, "SV"),
           (POOL[1], 16, r1c.ROLE_TEST, "SV"), (POOL[1], 48, r1c.ROLE_TEST, "SV"),
           (POOL[0], 80, r1c.ROLE_VALIDATION, "SV"), (POOL[1], 80, r1c.ROLE_VALIDATION, "other"),
           (POOL[0], 112, r1c.ROLE_CC, "CC"),
           (POOL[0], 144, r1c.ROLE_VERTICAL, "V"), (POOL[1], 160, r1c.ROLE_VERTICAL, "V")]


def _wins():
    out = []
    for seq, t0, role, cls in WINDOWS:
        idx = [int(round(10 * (t0 + 16 * i / 7))) for i in range(8)]
        out.append({"sequence": seq, "t0": float(t0), "k": t0 // 16, "role": role, "cls": cls,
                    "S": 0.2 if cls == "SV" else None, "window_id": f"{seq}:k{t0 // 16}",
                    "frame_times_utc": [BASE[seq] + t0 + 16.0 * i / 7 for i in range(8)],
                    "frame_indices_sorted": idx, "frame_indices_bag": idx})
    return out


def _rot(axis, ang):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


BACKBONE_FRAME = {"da3-base": (0.4, np.eye(3)), "vggt-1b": (0.55, _rot([0.3, 0.4, 0.5], 0.9))}


def _pose(w, sc, Q, up=False):
    t, enu = _track()
    ft = np.asarray(w["frame_times_utc"]) - BASE[w["sequence"]]
    p = np.stack([np.interp(ft, t, enu[:, c]) for c in range(3)], axis=1)
    p = p + np.random.default_rng(int(w["t0"]) + 7 * POOL.index(w["sequence"])).normal(0.0, 0.15, p.shape)
    C = sc * (Q @ p.T).T
    D = np.eye(3) if up else np.diag([1.0, -1.0, -1.0])  # up=True: camera optical axis points UP (FAILURE case)
    R = np.tile((Q @ D).T, (8, 1, 1))  # nadir camera in the backbone world
    T = np.zeros((8, 3, 4))
    T[:, :, :3] = R
    T[:, :, 3] = -np.einsum("nij,nj->ni", R, C)
    return T


def _ref(seq):
    t, enu = _track()
    llh = _llh(enu)
    tt = BASE[seq] + t
    return {"pos_t": tt, "lat": llh[:, 0], "lon": llh[:, 1], "alt": llh[:, 2], "info_t": tt,
            "info_status": np.full(t.size, 50)}


def make_env(tmp_path: Path, register: bool = True, speed_not_run=frozenset(), da3_up=frozenset()) -> dict:
    tmp_path.mkdir(parents=True, exist_ok=True)
    contract = tmp_path / "contract.yaml"
    contract.write_bytes(r1c.CONTRACT_PATH.read_bytes())
    sha = r1c.contract_sha(contract)
    reg = tmp_path / "registry.json"
    reg.write_text(json.dumps({"experiments": [{"contract_sha256": sha}] if register else []}))
    out, pack = tmp_path / "r1", tmp_path / "pack"
    s1 = out / "stage1"
    wins = _wins()
    t, enu = _track()
    llh = _llh(enu)
    vel = np.gradient(enu, t, axis=0)
    seqs = {}
    for seq in POOL:
        s1.mkdir(parents=True, exist_ok=True)
        cue = s1 / f"cue_{seq}.csv"
        cue.write_text("t_utc,lat_deg,lon_deg,alt_m\n" + "".join(
            f"{BASE[seq] + a:.6f},{b[0]:.9f},{b[1]:.9f},{b[2]:.4f}\n" for a, b in zip(t, llh)))
        pvt = s1 / f"pvt_{seq}.csv"
        if seq in speed_not_run:
            pvt = None
        else:
            pvt.write_text("record_t,t_utc,vel_n,vel_e,vel_d\n" + "".join(
                f"{BASE[seq] + a - 0.4:.6f},{BASE[seq] + a:.6f},{v[1]:.6f},{v[0]:.6f},{-v[2]:.6f}\n" for a, v in zip(t, vel)))
        ref = s1 / f"ref_{seq}.bag"
        ref.write_bytes(b"sealed sub-bag placeholder " + seq.encode())  # never opened: reader is mocked
        seqs[seq] = {"cue_csv": cue.name, "cue_csv_sha256": v2.sha256_file(cue),
                     "pvt": {"speed_arm": "RUN", "csv": pvt.name, "csv_sha256": v2.sha256_file(pvt)} if pvt else
                     {"speed_arm": "NOT_RUN", "decoded": True, "gate": {"status": "FAIL"}, "reason": "PVT gate FAIL"},
                     "gate": {"classes": {sm.cb.REF_TOPIC: "recorder"}, "recorder_offset_s": 0.4},
                     "reference_source": {"kind": "sealed_sub_bag", "path": ref.name, "sha256": v2.sha256_file(ref)}}
    for w in wins:
        d = s1 / "frames" / w["sequence"]
        d.mkdir(parents=True, exist_ok=True)
        for k in w["frame_indices_bag"]:
            (d / f"{k:08d}.png").write_bytes(b"\x89PNG-%s-%d" % (w["sequence"].encode(), k))
    (s1 / "windows.json").write_text(json.dumps({"contract_sha256": sha, "dry_run": False, "windows": wins}))
    files = {p.relative_to(s1).as_posix(): v2.sha256_file(p) for p in sorted(s1.rglob("*")) if p.is_file()}
    (s1 / "stage1_manifest.json").write_text(json.dumps({
        "contract_sha256": sha, "dry_run": False, "reference_values_read": False, "minimum_rule_applied": False,
        "windows_json_sha256": v2.sha256_file(s1 / "windows.json"), "sequences": seqs, "output_sha256": files,
        "windows": wins}))
    # VGGT pack (R1 builder) + synthetic VGGT outputs in the pack's exact output format
    assert pack_mod.build(src=s1, pack=pack, contract=contract, registry=reg, check=register) == 0
    sm.bind(contract, reg)
    lock = json.loads((pack / "model_lock.json").read_text())
    man_sha = v2.sha256_file(pack / "input_hashes.json")
    for w in wins:
        sc, Q = BACKBONE_FRAME["vggt-1b"]
        npz = pack / "outputs" / f"{sm.cb.w_entry(w)}.npz"
        npz.parent.mkdir(parents=True, exist_ok=True)
        K = np.tile(np.array([[500.0, 0, 259], [0, 500.0, 259], [0, 0, 1]]), (8, 1, 1))
        np.savez(npz, timestamps=np.asarray(w["frame_times_utc"]), T_w2c=_pose(w, sc, Q), K=K,
                 depth_z=np.ones((8, 2, 2), np.float32), depth_conf=np.ones((8, 2, 2), np.float32),
                 is_metric=np.array(False), model_id=np.array(MODEL_ID))
        npz.with_suffix(".json").write_text(json.dumps({
            "contains_ground_truth": False, "is_metric": False, "npz_sha256": v2.sha256_file(npz),
            "input_manifest_sha256": man_sha, "source_revision": lock["source_revision"],
            "model_revision": lock["model_revision"], "model_id": MODEL_ID, "n_frames": 8}))
    # DA3 arm
    rows = []
    for i, w in enumerate(wins):
        sc, Q = BACKBONE_FRAME["da3-base"]
        dest = out / "predictions_da3" / f"{i:04d}.npz"
        save_prediction(PredictionBundle(timestamps=np.asarray(w["frame_times_utc"]), T_w2c=_pose(w, sc, Q, i in da3_up)), dest,
                        meta={"contract_sha256": sha, "window": i})
        rows.append({"window": i, "sequence": w["sequence"], "role": w["role"], "prediction": str(dest),
                     "prediction_sha256": v2.sha256_file(dest)})
    (out / "da3_manifest.json").write_text(json.dumps({"contract_sha256": sha, "contains_ground_truth": False,
                                                       "windows": rows}))
    return {"contract": contract, "registry": reg, "out": out, "pack": pack, "sha": sha, "wins": wins}


def _reader_factory(env, calls):
    out = env["out"]

    def reader(path, rtk_class, offset):
        # ORDER: receipt + its sha256 file + diagnostics exist; REF-VALID log not yet
        rec = out / "SCORING_RECEIPT.json"
        assert rec.exists() and (out / "SCORING_RECEIPT.sha256").exists() and (out / "r1_diagnostics.json").exists()
        assert (out / "SCORING_RECEIPT.sha256").read_text().split()[0] == hashlib.sha256(rec.read_bytes()).hexdigest()
        assert not (out / "ref_valid_log.json").exists()
        assert rtk_class == "recorder" and offset == 0.4
        seq = Path(path).stem.replace("ref_", "")
        calls.append(seq)
        return _ref(seq)

    return reader


def _score(env, reader):
    return sm.score(out=env["out"], pack=env["pack"], contract=env["contract"], registry=env["registry"],
                    reference_reader=reader)


# ------------------------------------------------------------------ end to end
def test_scoring_order_receipt_and_ref_valid_log_before_any_metric(tmp_path, monkeypatch):
    env = make_env(tmp_path)
    out, calls, metric_calls = env["out"], [], []

    def before_metric():
        assert (out / "SCORING_RECEIPT.sha256").exists() and (out / "ref_valid_log.json").exists()
        metric_calls.append(1)

    real_eval_window, real_eval = sm.cb.evaluate_window, sm.evaluate_grounding
    monkeypatch.setattr(sm.cb, "evaluate_window", lambda *a, **k: (before_metric(), real_eval_window(*a, **k))[1])
    monkeypatch.setattr(sm, "evaluate_grounding", lambda *a, **k: (before_metric(), real_eval(*a, **k))[1])
    assert _score(env, _reader_factory(env, calls)) == 0
    assert calls == POOL and metric_calls
    # receipt never rewritten: its sha file still matches
    rec = out / "SCORING_RECEIPT.json"
    assert (out / "SCORING_RECEIPT.sha256").read_text().split()[0] == hashlib.sha256(rec.read_bytes()).hexdigest()
    comp = json.loads((out / "SCORING_COMPLETION.json").read_text())
    assert comp["result_sha256"] == v2.sha256_file(out / "r1_results.json")
    receipt = json.loads(rec.read_text())
    assert receipt["contract_sha256"] == env["sha"] and receipt["evaluation_count"] == 1
    assert receipt["r1_diagnostics_sha256"] == v2.sha256_file(out / "r1_diagnostics.json")


def test_end_to_end_verdicts_all_arms_and_sim3_coreport(tmp_path):
    env = make_env(tmp_path)
    assert _score(env, _reader_factory(env, [])) == 0
    res = json.loads((env["out"] / "r1_results.json").read_text())
    v = res["verdicts"]
    assert v["primary"]["R1-6_pass"] and v["primary"]["R1-7_pass"] and not v["primary"]["small_n_binding_caveat"]
    assert v["speed_doppler"]["verdict"] == "REPLICATED" and v["speed_doppler"]["n_valid"] == {"da3-base": 4, "vggt-1b": 4}
    assert v["altitude_ublox"]["verdict"] == "REPLICATED" and v["altitude_ublox"]["n_observable"]["vggt-1b"] == 2
    for bb, (sc, _) in BACKBONE_FRAME.items():
        p_rows = res["arms"]["primary"]["sv_test"][bb]["rows"]
        assert len(p_rows) == 4 and all(set(r["solver"]) == {"none", "naive", "ls", "robust"} for r in p_rows)
        assert all(r["sim3_coreport"]["oracle_ate_sim3_min"] is not None for r in p_rows)
        s_rows = res["arms"]["speed_doppler"]["sv_test"][bb]["rows"]
        assert all(r["cue_topic"] == r1c.PVT_TOPIC and r["solver"]["naive"] == pytest.approx(1 / sc, rel=0.03) for r in s_rows)
        a_v = res["arms"]["altitude_ublox"]["vertical"][bb]["rows"]
        assert all(r["altitude"]["status"] == r1c.ALT_OBSERVABLE and r["applied_scale"] == pytest.approx(1 / sc, rel=0.03)
                   for r in a_v)
        a_t = res["arms"]["altitude_ublox"]["sv_test"][bb]["rows"]  # level flight: reported, not scored
        assert all(r["altitude"]["status"] == r1c.ALT_NOT_OBSERVABLE and "eval_naive" not in r for r in a_t)
        assert "vertical" not in res["arms"]["primary"] and "vertical" not in res["arms"]["speed_doppler"]
        assert "sim3_coreport" in v["primary"]["backbones"][bb]
    diag = json.loads((env["out"] / "r1_diagnostics.json").read_text())
    assert diag["reference_values_read"] is False
    assert all(d["speed_over_displacement"]["ratio"] == pytest.approx(1.0, abs=0.02) for d in diag["speed"])


def test_scorer_refuses_replay(tmp_path):
    env = make_env(tmp_path)
    assert _score(env, _reader_factory(env, [])) == 0
    calls = []
    with pytest.raises(FileExistsError, match="refusing replay"):
        _score(env, _reader_factory(env, calls))
    assert calls == []


def test_scorer_awaits_when_an_arm_is_missing_and_writes_nothing(tmp_path):
    env = make_env(tmp_path)
    victim = next((env["pack"] / "outputs").rglob("*.npz"))
    victim.unlink()
    calls = []
    assert _score(env, _reader_factory(env, calls)) == 3
    (env["out"] / "da3_manifest.json").unlink()
    assert _score(env, _reader_factory(env, calls)) == 3
    assert calls == []
    for name in ("SCORING_RECEIPT.json", "SCORING_RECEIPT.sha256", "r1_diagnostics.json", "ref_valid_log.json"):
        assert not (env["out"] / name).exists()


def test_scorer_refuses_unregistered_contract_and_tampered_stage1(tmp_path):
    env = make_env(tmp_path)
    env["registry"].write_text('{"experiments": []}')
    with pytest.raises(r1c.ContractNotRegistered):
        _score(env, _reader_factory(env, []))
    env["registry"].write_text(json.dumps({"experiments": [{"contract_sha256": env["sha"]}]}))
    with open(env["out"] / "stage1" / f"pvt_{POOL[0]}.csv", "a") as f:
        f.write("1,1,0,0,0\n")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        _score(env, _reader_factory(env, []))
    assert not (env["out"] / "SCORING_RECEIPT.json").exists()


# ------------------------------------------------------------------ verdict functions
def _row(i, raw, gr, orc, s=1.0, **extra):
    return {"window": i, "role": r1c.ROLE_TEST, "solver": {"naive": s}, "three_limits": {"s_cue": None},
            "eval_naive": {"raw_ate_se3": raw, "grounded_ate_se3": gr, "oracle_ate_se3": orc,
                           "oracle_ate_optimal": orc * 0.9, "oracle_scale_ate_optimal": 1.0}, **extra}


def test_speed_arm_verdict_rules():
    good = [_row(0, 10.0, 1.0, 0.8), _row(1, 8.0, 0.9, 0.7)]
    bad = [_row(0, 10.0, 9.0, 0.8), _row(1, 8.0, 7.5, 0.7)]
    assert sm.speed_arm_verdict({"da3-base": good, "vggt-1b": good}, True)["verdict"] == "REPLICATED"
    assert sm.speed_arm_verdict({"da3-base": good, "vggt-1b": bad}, True)["verdict"] == "NOT_REPLICATED"
    assert sm.speed_arm_verdict({"da3-base": good, "vggt-1b": good}, False)["verdict"] == "INCONCLUSIVE"
    one = [good[0], {"window": 1, "unavailable": "speed coverage 0.571 < 0.8"}]
    r = sm.speed_arm_verdict({"da3-base": good, "vggt-1b": one}, True)
    assert r["verdict"] == "INCONCLUSIVE" and r["n_valid"]["vggt-1b"] == 1
    # >= 2 valid but none with leverage >= 0.25 -> R1-6 cannot pass -> NOT_REPLICATED
    flat = [_row(0, 1.0, 0.9, 0.9), _row(1, 1.0, 0.9, 0.85)]
    assert sm.speed_arm_verdict({"da3-base": flat, "vggt-1b": flat}, True)["verdict"] == "NOT_REPLICATED"


def test_altitude_arm_verdict_rules_failure_counts_as_raw():
    obs = {"status": r1c.ALT_OBSERVABLE}
    fail = {"status": r1c.ALT_FAILURE}
    good = [_row(0, 10.0, 1.0, 0.8, altitude=obs), _row(1, 8.0, 0.9, 0.7, altitude=obs)]
    assert sm.altitude_arm_verdict({"da3-base": good, "vggt-1b": good})["verdict"] == "REPLICATED"
    # FAILURE windows enter with grounded == raw ATE: 1 good + 2 failures -> median R = 0 -> NOT_REPLICATED
    mixed = [good[0], _row(1, 8.0, 8.0, 0.7, s=-2.0, altitude=fail), _row(2, 9.0, 9.0, 0.6, s=None, altitude=fail)]
    r = sm.altitude_arm_verdict({"da3-base": good, "vggt-1b": mixed})
    assert r["verdict"] == "NOT_REPLICATED" and r["n_failure"]["vggt-1b"] == 2
    notobs = {"window": 3, "altitude": {"status": r1c.ALT_NOT_OBSERVABLE}, "not_scored": "NOT_OBSERVABLE"}
    r = sm.altitude_arm_verdict({"da3-base": good, "vggt-1b": [good[0], notobs]})
    assert r["verdict"] == "INCONCLUSIVE" and r["n_observable"] == {"da3-base": 2, "vggt-1b": 1}


def test_primary_gates_small_n_caveat_and_sim3_coreport():
    good = [_row(0, 10.0, 1.0, 0.8), _row(1, 8.0, 0.9, 0.7)]
    sections = {"sv_test": {"da3-base": {"rows": good}, "vggt-1b": {"rows": good}}}
    g = sm.primary_gates(sections, n_test=2)
    assert g["R1-6_pass"] and g["R1-7_pass"] and g["small_n_binding_caveat"]
    assert g["outcome_if_integrity_pass"].startswith("PRIMARY_REPLICATED")
    co = g["backbones"]["vggt-1b"]["sim3_coreport"]
    assert co["median_oracle_ate_sim3_min"] == pytest.approx(0.675) and co["gate7_vs_sim3_oracle_pass_reported_only"]
    bad = {"sv_test": {"da3-base": {"rows": good}, "vggt-1b": {"rows": [_row(0, 10.0, 9.0, 0.8)]}}}
    assert sm.primary_gates(bad, 4)["outcome_if_integrity_pass"] == "PRIMARY_NOT_REPLICATED"


# ------------------------------------------------------------------ pack / isolation
def test_pack_r1_identity_pins_unchanged_and_no_roles(tmp_path):
    env = make_env(tmp_path)
    pack = env["pack"]
    man = json.loads((pack / "input_hashes.json").read_text())
    assert man["pack_id"] == r1c.PACK_ID == "vggt_offbox_site_replication_r1" and man["sequences"] == POOL
    lock = json.loads((pack / "model_lock.json").read_text())
    tmpl = json.loads((pack_mod.TEMPLATE_PACK / "model_lock.json").read_text())
    for k in pack_mod.bp.PIN_KEYS + ("checkpoint_bytes",):
        assert lock[k] == tmpl[k]
    assert lock["contract_sha256"] == env["sha"] and lock["contract"].endswith("prospective_site_replication_r1.yaml")
    assert lock["pins_asserted_against"]["sha256"] == v2.CONTRACT_SHA
    blob = "".join(p.read_text() for p in pack.rglob("*.json") if p.name != "model_lock.json" and "outputs" not in p.parts)
    assert not any(tok in blob for tok in ("prospective_test", "cc_control", "validation", "vertical", '"S"', "dh_window"))
    pv = sm.cb._load_pack_verify(pack)
    assert [e for e, _ in pv.verify_inputs(pack)] == [sm.cb.w_entry(w) for w in env["wins"]]


def test_r1_binding_never_mutates_frozen_v2_module():
    before = (v2.PACK_ID, v2.CONTRACT_SHA, v2.POOL, dict(v2.V_CRUISE_TABLE_BY_SEQ), dict(v2.SECTION_OF_ROLE))
    view = sm.bind(r1c.CONTRACT_PATH, None)
    assert view.PACK_ID == r1c.PACK_ID and view.v_cruise_for({"sequence": POOL[1]}) == 9.0
    assert view.REF_VALID_REQUIRED == v2.REF_VALID_REQUIRED and view.FIXED_VALUES == v2.FIXED_VALUES
    assert (v2.PACK_ID, v2.CONTRACT_SHA, v2.POOL, v2.V_CRUISE_TABLE_BY_SEQ, v2.SECTION_OF_ROLE) == before
    fresh = _load("cb_fresh_t", "scripts/cross_backbone_v2_score.py")
    assert fresh.v2 is v2 and fresh.CONTRACT_SHA == v2.CONTRACT_SHA and sm.cb is not fresh


# ------------------------------------------------------------------ pre-freeze review fixes
def test_no_reference_subbag_byte_is_read_before_the_receipt(tmp_path, monkeypatch):
    """MAJ-1: Stage-1 verification must skip ref_*.bag; the sealed sub-bags are hashed only after the receipt."""
    import builtins
    import io
    import os

    env = make_env(tmp_path)
    out, events = env["out"], []
    real_open = builtins.open

    def spy(file, *a, **k):
        try:
            name = Path(os.fspath(file)).name
        except TypeError:
            name = ""
        if name.startswith("ref_") and name.endswith(".bag"):
            events.append((name, (out / "SCORING_RECEIPT.json").exists() and (out / "SCORING_RECEIPT.sha256").exists()))
        return real_open(file, *a, **k)

    monkeypatch.setattr(builtins, "open", spy)
    monkeypatch.setattr(io, "open", spy)
    calls = []
    assert _score(env, _reader_factory(env, calls)) == 0
    assert calls == POOL
    assert {n for n, _ in events} == {f"ref_{s}.bag" for s in POOL}  # hashed (post-receipt) ...
    assert all(after for _, after in events), events                   # ... and never before the receipt
    # a tampered sub-bag is caught AFTER the receipt (even with a custom reader), before the reader is called
    env2 = make_env(tmp_path / "t")
    with open(env2["out"] / "stage1" / f"ref_{POOL[1]}.bag", "ab") as f:
        f.write(b"x")
    calls2 = []
    with pytest.raises(RuntimeError, match="reference source hash mismatch"):
        _score(env2, _reader_factory(env2, calls2))
    assert calls2 == [] and (env2["out"] / "SCORING_RECEIPT.sha256").exists()
    assert not (env2["out"] / "ref_valid_log.json").exists()


def test_altitude_failure_window_scored_at_raw_ate_and_counted_in_medians(tmp_path):
    """MIN-4 (kills review mutant M20): a FAILURE V window gets applied scale 1.0 (grounded == raw ATE) and is
    counted in the altitude-arm denominator and medians, which here turns the arm NOT_REPLICATED."""
    wins = _wins()
    i_fail = next(i for i, w in enumerate(wins) if w["role"] == r1c.ROLE_VERTICAL and w["sequence"] == POOL[1])
    i_obs = next(i for i, w in enumerate(wins) if w["role"] == r1c.ROLE_VERTICAL and w["sequence"] == POOL[0])
    env = make_env(tmp_path, da3_up={i_fail})
    assert _score(env, _reader_factory(env, [])) == 0
    res = json.loads((env["out"] / "r1_results.json").read_text())
    rows = {r["window"]: r for r in res["arms"]["altitude_ublox"]["vertical"]["da3-base"]["rows"]}
    f, o = rows[i_fail], rows[i_obs]
    assert f["altitude"]["status"] == r1c.ALT_FAILURE and f["altitude"]["s_alt"] < 0
    assert f["applied_scale"] == 1.0 and f["solver"]["naive"] == f["altitude"]["s_alt"]
    assert f["eval_naive"]["grounded_ate_se3"] == pytest.approx(f["eval_none"]["grounded_ate_se3"])
    assert f["eval_naive"]["grounded_ate_se3"] == pytest.approx(f["eval_naive"]["raw_ate_se3"], rel=1e-6)
    assert o["altitude"]["status"] == r1c.ALT_OBSERVABLE and o["applied_scale"] == pytest.approx(2.5, rel=0.03)
    v = res["verdicts"]["altitude_ublox"]
    assert v["n_observable"] == {"da3-base": 2, "vggt-1b": 2} and v["n_failure"] == {"da3-base": 1, "vggt-1b": 0}
    blk = v["backbones"]["da3-base"]
    ev = {r["window"]: r for r in blk["windows"] if r["evaluable"]}
    assert i_fail in ev and ev[i_fail]["capture_R"] == pytest.approx(0.0, abs=1e-6)
    assert not blk["gate6_pass"] and v["backbones"]["vggt-1b"]["gate6_pass"]
    assert v["verdict"] == "NOT_REPLICATED"
    diag = json.loads((env["out"] / "r1_diagnostics.json").read_text())
    assert {d["window"]: d["status"] for d in diag["altitude"]["da3-base"]}[i_fail] == r1c.ALT_FAILURE


def test_end_to_end_speed_arm_not_run_for_one_sequence(tmp_path):
    """MIN-5: PVT gate FAIL on GNSS03 -> speed arm NOT_RUN there (rows unavailable, disclosed); the primary arm and
    the other sequence are unaffected; the arm verdict is computed from the 2 valid GNSS02 test windows."""
    env = make_env(tmp_path, speed_not_run={POOL[1]})
    assert _score(env, _reader_factory(env, [])) == 0
    res = json.loads((env["out"] / "r1_results.json").read_text())
    assert res["speed_arm_by_sequence"] == {POOL[0]: True, POOL[1]: False}
    v = res["verdicts"]
    assert v["primary"]["R1-6_pass"] and v["primary"]["R1-7_pass"]
    assert v["speed_doppler"]["n_valid"] == {"da3-base": 2, "vggt-1b": 2}
    assert v["speed_doppler"]["verdict"] == "REPLICATED"
    for bb in BACKBONE_FRAME:
        assert len([r for r in res["arms"]["primary"]["sv_test"][bb]["rows"] if "eval_naive" in r]) == 4
        for r in res["arms"]["speed_doppler"]["sv_test"][bb]["rows"]:
            if r["sequence"] == POOL[1]:
                assert "eval_naive" not in r and "NOT RUN" in r["unavailable"]
            else:
                assert "eval_naive" in r
    diag = json.loads((env["out"] / "r1_diagnostics.json").read_text())
    assert diag["speed_arm_by_sequence"] == {POOL[0]: True, POOL[1]: False}
    assert all(d["arm_run"] is (d["sequence"] == POOL[0]) for d in diag["speed"])
    comp = json.loads((env["out"] / "SCORING_COMPLETION.json").read_text())
    ex = [e for e in comp["exclusions"] if e["arm"] == "speed_doppler"]
    assert ex and all(e["sequence"] == POOL[1] for e in ex)
    # every arm NOT_RUN -> INCONCLUSIVE end to end
    env2 = make_env(tmp_path / "none", speed_not_run=set(POOL))
    assert _score(env2, _reader_factory(env2, [])) == 0
    v2_ = json.loads((env2["out"] / "r1_results.json").read_text())["verdicts"]
    assert v2_["speed_doppler"]["verdict"] == "INCONCLUSIVE" and v2_["primary"]["R1-6_pass"]


def test_stale_diagnostics_archived_and_logged_on_rerun_then_refused_after_receipt(tmp_path, monkeypatch):
    """MIN-9: a crash between diagnostics and receipt does not brick the run; the stale file is archived (never
    deleted) and logged; once a receipt exists every rerun is refused."""
    env = make_env(tmp_path)
    out = env["out"]
    real_dump = sm.dump

    def crash_on_receipt(path, obj, exclusive=False):
        if Path(path).name == "SCORING_RECEIPT.json":
            raise KeyboardInterrupt("simulated crash before the receipt")
        return real_dump(path, obj, exclusive=exclusive)

    monkeypatch.setattr(sm, "dump", crash_on_receipt)
    calls = []
    with pytest.raises(KeyboardInterrupt):
        _score(env, _reader_factory(env, calls))
    assert calls == [] and (out / "r1_diagnostics.json").exists() and not (out / "SCORING_RECEIPT.json").exists()
    stale = (out / "r1_diagnostics.json").read_bytes()
    monkeypatch.setattr(sm, "dump", real_dump)
    assert _score(env, _reader_factory(env, calls)) == 0 and calls == POOL
    archived = sorted(out.glob("r1_diagnostics.*.json"))
    assert len(archived) == 1 and archived[0].read_bytes() == stale
    log = [json.loads(x) for x in (out / "r1_rerun_log.jsonl").read_text().splitlines()]
    assert len(log) == 1 and log[0]["archived_as"] == archived[0].name
    assert log[0]["sha256"] == hashlib.sha256(stale).hexdigest() and log[0]["receipt_existed"] is False
    receipt = json.loads((out / "SCORING_RECEIPT.json").read_text())
    assert receipt["rerun_log_sha256"] == v2.sha256_file(out / "r1_rerun_log.jsonl")
    # receipt exists -> refuse; nothing archived, nothing logged
    with pytest.raises(FileExistsError, match="refusing replay"):
        _score(env, _reader_factory(env, []))
    with pytest.raises(FileExistsError, match="refusing replay"):
        sm.archive_stale_diagnostics(out)
    assert len(list(out.glob("r1_diagnostics.*.json"))) == 1
    assert len((out / "r1_rerun_log.jsonl").read_text().splitlines()) == 1
