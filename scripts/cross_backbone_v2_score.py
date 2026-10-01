"""Cross-backbone v2 - one-time paired scoring (DA3-BASE vs VGGT-1B).

Contract: configs/prospective_cross_backbone_v5.yaml (FROZEN; v5 = v4 + A3 pooled SV windows). Refuses unless BOTH arms are present and
verified (DA3 manifest + prediction hashes; VGGT pack inputs/outputs/metadata). Order of operations:

  1. replay guard (SCORING_RECEIPT.json must not exist)
  2. verify both arms (exit 3 = AWAITING; nothing is written, no reference is opened)
  3. GT-free rho(a,b) diagnostic recorded (rho_diagnostics.json, exclusive create) BEFORE scoring
  4. SCORING_RECEIPT.json written atomically ('x') BEFORE any reference value is read
  5. reference (rtk_position + rtk_info_position) opened once; REF-VALID log written BEFORE metrics
  6. metrics (identical REF-VALID frame subset for both backbones and all solvers) -> paired_results.json

Usage: python scripts/cross_backbone_v2_score.py score
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from yardstick3d.datasets import timescale as ts  # noqa: E402
from yardstick3d.datasets import v2_scoring as v2  # noqa: E402
from yardstick3d.datasets import v2_sync_guards as sg  # noqa: E402

OUT = ROOT / "artifacts/cross_backbone_v5"  # v5 pooled (A3)
PACK = ROOT / "artifacts/vggt_pack_cross_backbone_v5"
BAG = ROOT / "data/mars_lvig" / f"{v2.SEQUENCE}.bag"  # v2-v4 single-sequence default
CONTRACT = v2.CONTRACT_PATH
CONTRACT_SHA = v2.CONTRACT_SHA
SEQ = v2.SEQUENCE
REF_TOPIC = "/dji_osdk_ros/rtk_position"
REF_STATUS_TOPIC = "/dji_osdk_ros/rtk_info_position"
BACKBONES = ("da3-base", "vggt-1b")
SOLVERS = ("none", "naive", "ls", "robust")

sha256_file = v2.sha256_file
_clean = v2.clean_json
dump = v2.dump_json
_guard_receipt = v2.guard_receipt
median = v2.median


class ArmsNotReady(RuntimeError):
    """One arm absent/unverified: scoring is withheld (AWAITING)."""


def _load_pack_verify(pack: Path):
    spec = importlib.util.spec_from_file_location("cc_v2_pack_verify", pack / "verify.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pack_entry(t0: float, seq: str = SEQ) -> str:
    return f"{seq}/t{int(round(t0))}_{int(round(t0 + v2.WINDOW_DURATION_S))}"


def w_entry(w: dict) -> str:
    return pack_entry(float(w["t0"]), v2.seq_of(w))


# ------------------------------ arm verification ----------------------------
def verify_arms(out: Path, pack: Path, wins: list) -> tuple[dict, dict, dict]:
    """Both arms present + verified, else ArmsNotReady. Returns (da3_by_idx, vggt_pred_paths_by_idx, manifest)."""
    da3_path = out / "da3_manifest.json"
    if not da3_path.exists():
        raise ArmsNotReady("DA3 manifest absent")
    da3 = json.loads(da3_path.read_text())
    if da3.get("contract_sha256") != CONTRACT_SHA or da3.get("contains_ground_truth") is not False:
        raise ArmsNotReady("DA3 manifest contract/GT attestation invalid")
    if len(da3["windows"]) != len(wins):
        raise ArmsNotReady("DA3 window count != Stage-1 window count")
    by_idx = {}
    for r in da3["windows"]:
        p = ROOT / r["prediction"] if not Path(r["prediction"]).is_absolute() else Path(r["prediction"])
        if not p.exists() or sha256_file(p) != r["prediction_sha256"]:
            raise ArmsNotReady(f"DA3 prediction missing/hash mismatch: window {r['window']}")
        by_idx[int(r["window"])] = {**r, "path": p}
    if sorted(by_idx) != list(range(len(wins))):
        raise ArmsNotReady("DA3 windows are not 0..n-1")

    if not (pack / "input_hashes.json").exists() or not (pack / "verify.py").exists():
        raise ArmsNotReady("VGGT pack absent")
    pv = _load_pack_verify(pack)
    manifest = json.loads((pack / "input_hashes.json").read_text())
    if manifest.get("pack_id") != v2.PACK_ID:
        raise ArmsNotReady(f"unexpected pack_id {manifest.get('pack_id')!r}")
    lock = json.loads((pack / "model_lock.json").read_text())
    if lock.get("contract_sha256") != CONTRACT_SHA:
        raise ArmsNotReady("pack model_lock contract sha != frozen v2 contract sha")
    pack_windows = dict(pv.verify_inputs(pack))
    entries = [w_entry(w) for w in wins]
    if sorted(pack_windows) != sorted(entries) or len(set(entries)) != len(entries):
        raise ArmsNotReady("pack window list != Stage-1 window list")
    vggt = {}
    for i, (w, e) in enumerate(zip(wins, entries)):
        p = pack / "outputs" / f"{e}.npz"
        if not p.exists():
            raise ArmsNotReady(f"VGGT output absent: {e}")
        if not np.allclose(pack_windows[e]["timestamps"], w["frame_times_utc"], atol=1e-6, rtol=0):
            raise ArmsNotReady(f"pack timestamps != Stage-1 UTC frame times: {e}")
        pv.verify_output(p, np.asarray(pack_windows[e]["timestamps"]))
        pv.verify_metadata(p, pack)
        vggt[i] = p
    return by_idx, vggt, manifest


# ------------------------------ reference reading ---------------------------
def decode_status(msg):
    """Fix-status integer from an rtk_info_position message (std_msgs UInt8-like `.data`, or a named field)."""
    for name in ("data", "position_solution", "positionSolution", "status", "fix_status"):
        if hasattr(msg, name):
            v = getattr(msg, name)
            if hasattr(v, "status"):  # NavSatStatus-like
                v = v.status
            return int(v)
    raise RuntimeError(f"cannot decode fix status from {type(msg).__name__} (fail closed)")


def to_utc_stamps(header_s, record_s, known_class: str | None = None, recorder_offset_s: float = 0.0):
    """Convert header stamps to UTC via the published constant (never fitted). Returns (t_utc, class).

    known_class: the Stage-1 gate class for the topic when it exists; else classified against record time
    (unknown => RuntimeError). Header-less topics: pass header_s=None -> record time is used as-is.
    """
    rec = np.asarray(record_s, float)
    if header_s is None:  # v4 A2: header-less recorder topic -> record time + recorder offset
        return rec + recorder_offset_s, "record_time+recorder_offset"
    h = np.asarray(header_s, float)
    if known_class == "recorder":  # v4 A2
        return h + recorder_offset_s, "recorder"
    cls = known_class or ts.classify_timescale(ts.header_offset_stats(h, rec)["median"], sg.TOL_CLASS_S)
    if cls == "unknown":
        raise RuntimeError("reference header-record offset unexplained (fail closed)")
    return ts.to_utc(h, cls), cls


def read_reference(bag: Path, rtk_class: str, recorder_offset_s: float = 0.0) -> dict:
    """Open the withheld DJI RTK reference ONCE (scoring only): positions + fix-status, all in UTC."""
    from rosbags.rosbag1 import Reader
    from rosbags.typesys import Stores, get_typestore

    store = get_typestore(Stores.ROS1_NOETIC)
    with Reader(bag) as reader:
        conns = {c.topic: c for c in reader.connections}
        for t in (REF_TOPIC, REF_STATUS_TOPIC):
            if t not in conns:
                raise RuntimeError(f"missing reference topic {t}")
        ph, pr, lat, lon, alt = [], [], [], [], []
        c = conns[REF_TOPIC]
        for _, rec, raw in reader.messages(connections=[c]):
            m = store.deserialize_ros1(raw, c.msgtype)
            la, lo, al = float(m.latitude), float(m.longitude), float(m.altitude)
            if not (np.isfinite(la) and np.isfinite(lo) and np.isfinite(al)):
                continue
            ph.append(sg.rtk_header_stamp_s(raw))
            pr.append(rec / 1e9)
            lat.append(la)
            lon.append(lo)
            alt.append(al)
        ih, ir, st = [], [], []
        c = conns[REF_STATUS_TOPIC]
        has_header = None
        for _, rec, raw in reader.messages(connections=[c]):
            m = store.deserialize_ros1(raw, c.msgtype)
            has_header = hasattr(m, "header")
            ih.append(sg.rtk_header_stamp_s(raw) if has_header else float("nan"))
            ir.append(rec / 1e9)
            st.append(decode_status(m))
    pos_t, pos_cls = to_utc_stamps(ph, pr, rtk_class, recorder_offset_s)
    info_t, info_cls = to_utc_stamps(ih if has_header else None, ir, "recorder" if has_header else None, recorder_offset_s)
    return {"pos_t": pos_t, "lat": np.asarray(lat), "lon": np.asarray(lon), "alt": np.asarray(alt),
            "info_t": info_t, "info_status": np.asarray(st), "pos_class": pos_cls, "info_class": info_cls}


# ------------------------------- scoring core -------------------------------
def load_cue(path: Path):
    import csv

    t, llh = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            t.append(float(row["t_utc"]))
            llh.append((float(row["lat_deg"]), float(row["lon_deg"]), float(row["alt_m"])))
    return np.asarray(t), np.asarray(llh)


def _per_seq(obj, w):
    """obj is either one value (single-sequence v2-v4) or per_sequence({sequence: value}) (v5 pool)."""
    if isinstance(obj, dict) and "__per_sequence__" in obj:
        return obj[v2.seq_of(w)]
    return obj


def per_sequence(d: dict) -> dict:
    return {"__per_sequence__": True, **d}


def ref_valid_log(wins: list, ref) -> list[dict]:
    """Per-window REF-VALID decision from frame stamps + reference availability ONLY (no metric)."""
    logs = []
    for i, w in enumerate(wins):
        r = _per_seq(ref, w)
        rv = v2.ref_valid_frames(w["frame_times_utc"], r["pos_t"], r["info_t"], r["info_status"],
                                 v2.FIXED_VALUES, v2.MAX_DT_REF_S)
        logs.append({"window": i, "sequence": v2.seq_of(w), "role": w["role"], "n_matched": rv["n_matched"], "n_fixed": rv["n_fixed"],
                     "max_abs_dt_s": rv["max_abs_dt_s"], "evaluable": v2.window_evaluable(rv["mask"]),
                     "ref_valid_mask": [bool(x) for x in rv["mask"]], "_rv": rv})
    return logs


def evaluate_window(w: dict, i: int, pred, rv: dict, cue_t, cue_llh, ref: dict) -> dict:
    from yardstick3d.datasets import mars_lvig as ml
    from yardstick3d.evaluation.three_limits import cue_discrepancy, cue_supported_scale, estimator_departure
    from yardstick3d.experiments.advio_real import evaluate_grounding
    from yardstick3d.geometry.trajectory import path_length
    from yardstick3d.optimization.grounder import MetricGrounder

    ts_ = np.asarray(w["frame_times_utc"], float)
    mask = rv["mask"]
    lat0, lon0, alt0 = v2.window_origin(cue_t, cue_llh, float(ts_[0]))
    cue_alt = np.where(np.isfinite(cue_llh[:, 2]), cue_llh[:, 2], 0.0)
    cue_enu = ml.cue_enu_from_llh(cue_llh[:, 0], cue_llh[:, 1], cue_alt, lat0, lon0, alt0)
    ref_enu = ml.cue_enu_from_llh(ref["lat"], ref["lon"], ref["alt"], lat0, lon0, alt0)  # SAME origin
    gt_c = ref_enu[rv["ref_index"][mask]]  # nearest sample within 0.15 s (REF-VALID frames only)
    sub = v2.SubsetPred(pred, mask)

    cs, cinfo = ml.constraints_from_ublox_displacement(ts_, cue_t, cue_enu, max_dt_match=v2.MAX_DT_CUE_S)
    g = MetricGrounder()
    row = {"window": i, "sequence": v2.seq_of(w), "role": w["role"], "cls": w.get("cls"), "t0": w["t0"],
           "S_timing": w.get("S"),
           "n_ref_valid": int(mask.sum()),
           "cue": {"n_kept": cinfo["n_kept"], "n_dropped": cinfo["n_dropped"]}, "solver": {}}
    for name in SOLVERS:
        r = g.solve(pred, cs, solver=name)
        s = float(r.scale) if np.isfinite(r.scale) else None
        row["solver"][name] = s
        row[f"eval_{name}"] = evaluate_grounding(sub, gt_c, r.scale)
        if s is None:
            row.setdefault("failures", []).append(f"{name}: non-finite scale (counted, no substitution)")
    # three limits on the REF-VALID subset (cue polyline between consecutive REF-VALID frames)
    b, cov = ml.cue_polyline_lengths(cue_t, cue_enu, ts_[mask])
    okb = b[cov >= 0.8]
    Lg = float(np.sum(okb)) if okb.size and np.all(np.isfinite(okb)) else float("nan")
    Lt, Lv = path_length(gt_c), path_length(sub.centers())
    s_cue = cue_supported_scale(Lg, Lv) if np.isfinite(Lg) else None
    s_hat = row["solver"]["naive"]
    row["three_limits"] = {
        "s_cue": s_cue, "Lg_over_Lt": cue_discrepancy(Lg, Lt),
        "estimator_departure": estimator_departure(s_hat, s_cue) if (s_hat is not None and s_cue is not None) else None,
    }
    # nominal-speed-prior secondary metric (no sensor): s_prior = v_cruise * 16 / sum_k a_k over the full window
    csum = float(np.sum(v2.chords(pred.centers())))
    vc = v2.v_cruise_for(w)
    sp = v2.nominal_speed_prior_scale(csum, vc)
    ev_p = evaluate_grounding(sub, gt_c, sp) if sp is not None else None
    row["nominal_speed_prior"] = {"v_cruise_mps": vc, "chord_sum": csum, "s_prior": sp,
                                  "grounded_ate_se3": ev_p["grounded_ate_se3"] if ev_p else None}
    return row


def summarize(rows: list[dict]) -> dict:
    scored = [r for r in rows if "eval_naive" in r]
    m = {
        "n_windows": len(rows), "n_scored": len(scored),
        "median_raw_ate_se3": median([r["eval_none"]["raw_ate_se3"] for r in scored]),
        "median_grounded_naive_ate_se3": median([r["eval_naive"]["grounded_ate_se3"] for r in scored]),
        "median_grounded_ls_ate_se3": median([r["eval_ls"]["grounded_ate_se3"] for r in scored]),
        "median_grounded_robust_ate_se3": median([r["eval_robust"]["grounded_ate_se3"] for r in scored]),
        "median_oracle_ate": median([r["eval_naive"]["oracle_ate_optimal"] for r in scored]),
        "median_oracle_scale": median([r["eval_naive"]["oracle_scale_ate_optimal"] for r in scored]),
        "median_Lg_over_Lt": median([r["three_limits"]["Lg_over_Lt"] for r in scored]),
        "median_nominal_prior_ate_se3": median([r["nominal_speed_prior"]["grounded_ate_se3"] for r in scored]),
    }
    raw, gr = m["median_raw_ate_se3"], m["median_grounded_naive_ate_se3"]
    m["raw_to_grounded_reduction"] = float(1.0 - gr / raw) if (raw and gr is not None and raw > 0) else None
    return m


def score_core(wins: list, cue_t, cue_llh, ref, loaders: dict) -> tuple[dict, list]:
    """Pure scoring core. loaders: {backbone: callable(window_idx) -> PredictionBundle}.

    cue_t / cue_llh / ref: single values (v2-v4), or per_sequence({seq: value}) for the v5 pool
    (each window uses its own sequence's cue and reference).

    The REF-VALID decision (and its log) is fixed from frame stamps + reference availability BEFORE any
    metric; excluded windows are recorded as unavailable for BOTH backbones, never replaced.
    Returns (sections, ref_log).
    """
    logs = ref_valid_log(wins, ref)
    ref_log = [{k: v for k, v in L.items() if k != "_rv"} for L in logs]
    sections = {s: {} for s in v2.SECTION_OF_ROLE.values()}
    for bb in BACKBONES:
        per_role: dict = {s: [] for s in sections}
        for i, w in enumerate(wins):
            sec = v2.SECTION_OF_ROLE[w["role"]]
            L = logs[i]
            if not L["evaluable"]:
                per_role[sec].append({"window": i, "sequence": v2.seq_of(w), "role": w["role"], "t0": w["t0"],
                                      "unavailable": f"REF-VALID frames {L['n_fixed']}/8 < {v2.REF_VALID_REQUIRED}"})
                continue
            per_role[sec].append(evaluate_window(w, i, loaders[bb](i), L["_rv"], _per_seq(cue_t, w),
                                                 _per_seq(cue_llh, w), _per_seq(ref, w)))
        for sec, rows in per_role.items():
            sections[sec][bb] = {"rows": rows, "summary": summarize(rows)}
    return sections, ref_log


def seq_sources(s1: Path, s1_manifest: dict, wins: list, bag: Path) -> dict:
    """{seq: (reference bag path, rtk class, recorder offset, sha256|None)}; v5 manifests are per-sequence."""
    if "sequences" not in s1_manifest:  # v2-v4 single-sequence layout
        g = s1_manifest["gate"]
        return {SEQ: (bag, g["classes"][REF_TOPIC], float(g.get("recorder_offset_s", 0.0)), None)}
    out = {}
    for seq in sorted({v2.seq_of(w) for w in wins}, key=v2.POOL.index):
        info = s1_manifest["sequences"][seq]
        src = info["reference_source"]
        p = s1 / src["path"] if src["kind"] == "sealed_sub_bag" else ROOT / src["path"]
        out[seq] = (p, info["gate"]["classes"][REF_TOPIC], float(info["gate"]["recorder_offset_s"]), src["sha256"])
    return out


def score(out: Path | None = None, pack: Path | None = None, bag: Path | None = None,
          contract: Path | None = None, registry: Path | None = v2.REGISTRY_PATH, reference_reader=None) -> int:
    from yardstick3d.io.prediction_cache import load_prediction

    out = Path(out or OUT)
    pack = Path(pack or PACK)
    bag = Path(bag or BAG)
    v2.check_contract(contract or CONTRACT, CONTRACT_SHA, registry)
    receipt = out / "SCORING_RECEIPT.json"
    _guard_receipt(receipt)  # 1. replay refusal

    s1 = out / "stage1"
    s1_manifest = json.loads((s1 / "stage1_manifest.json").read_text())
    if s1_manifest.get("contract_sha256") != CONTRACT_SHA:
        raise RuntimeError("stage1 manifest contract sha != frozen v2 contract sha")
    wins = json.loads((s1 / "windows.json").read_text())["windows"]
    try:
        da3_by_idx, vggt_paths, pack_manifest = verify_arms(out, pack, wins)  # 2. both arms verified
    except ArmsNotReady as exc:
        print(f"AWAITING: {exc}; scoring withheld (no reference opened)")
        return 3

    pooled = "sequences" in s1_manifest
    seqs = sorted({v2.seq_of(w) for w in wins}, key=lambda s: v2.POOL.index(s) if s in v2.POOL else 0)
    if pooled:
        cues = {s: load_cue(s1 / v2.cue_csv_name(s)) for s in seqs}
        cue_t = per_sequence({s: c[0] for s, c in cues.items()})
        cue_llh = per_sequence({s: c[1] for s, c in cues.items()})
    else:
        cue_t, cue_llh = load_cue(s1 / "cue_receiver_lla.csv")
    loaders = {
        "da3-base": lambda i: load_prediction(da3_by_idx[i]["path"]),
        "vggt-1b": lambda i: load_prediction(vggt_paths[i]),
    }

    # 3. GT-free rho(a,b) diagnostic, recorded BEFORE scoring (exclusive create)
    from yardstick3d.datasets import mars_lvig as ml

    rho = {}
    for bb in BACKBONES:
        rho[bb] = []
        for i, w in enumerate(wins):
            tsw = np.asarray(w["frame_times_utc"], float)
            ct, cl = _per_seq(cue_t, w), _per_seq(cue_llh, w)
            lat0, lon0, alt0 = v2.window_origin(ct, cl, float(tsw[0]))
            alt = np.where(np.isfinite(cl[:, 2]), cl[:, 2], 0.0)
            enu = ml.cue_enu_from_llh(cl[:, 0], cl[:, 1], alt, lat0, lon0, alt0)
            rho[bb].append({"window": i, "sequence": v2.seq_of(w), "role": w["role"],
                            **v2.rho_diagnostic(loaders[bb](i).centers(), tsw, ct, enu)})
    dump(out / "rho_diagnostics.json", {"contract_sha256": CONTRACT_SHA, "reference_values_read": False,
                                        "disclosure_only": True, "rho": rho}, exclusive=True)

    # 4. receipt written BEFORE the reference is opened
    receipt_payload = {
        "contract_sha256": CONTRACT_SHA, "sequences": seqs, "evaluation_count": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "evaluator_revision": "yardstick3d.experiments.advio_real.evaluate_grounding + three_limits (v2 REF-VALID subset)",
        "da3_manifest_sha256": sha256_file(out / "da3_manifest.json"),
        "stage1_manifest_sha256": sha256_file(s1 / "stage1_manifest.json"),
        "rho_diagnostics_sha256": sha256_file(out / "rho_diagnostics.json"),
        "vggt_input_manifest_sha256": sha256_file(pack / "input_hashes.json"),
        "vggt_prediction_sha256": {str(k): sha256_file(v) for k, v in vggt_paths.items()},
        "da3_prediction_sha256": {str(k): r["prediction_sha256"] for k, r in da3_by_idx.items()},
        "evaluated_windows": [f"{w['role']}:{v2.seq_of(w)}:{w['t0']}" for w in wins],
        "exclusions": [],
    }
    dump(receipt, receipt_payload, exclusive=True)

    # 5. reference opened once; REF-VALID log before metrics
    reader = reference_reader or read_reference
    sources = seq_sources(s1, s1_manifest, wins, bag)
    if pooled:
        for s, src in sources.items():  # reference bytes must be the frozen ones (local bag or sealed sub-bag)
            if reference_reader is None and sha256_file(src[0]) != src[3]:
                raise RuntimeError(f"reference source hash mismatch for {s}: {src[0]}")
        ref = per_sequence({s: reader(src[0], src[1], src[2]) for s, src in sources.items()})
    else:
        src = sources[SEQ]
        ref = reader(src[0], src[1], src[2])
    sections, ref_log = score_core(wins, cue_t, cue_llh, ref, loaders)
    dump(out / "ref_valid_log.json", {"fixed_values": list(v2.FIXED_VALUES), "max_dt_s": v2.MAX_DT_REF_S,
                                      "required": v2.REF_VALID_REQUIRED, "windows": ref_log})

    results = {
        "contract_sha256": CONTRACT_SHA, "sequences": seqs, "backbones": list(BACKBONES),
        "split": {"sv_test_windows": 4, "validation_windows": 2, "cc_control_windows": 2},
        "sections": sections, "ref_valid_log": ref_log,
        "primary_solver": "naive", "claim_level": 0, "reference_values_read": True,
        "notes": ["validation and cc_control sections are NOT claims and enter no gate",
                  "ATE computed on REF-VALID frames only, identical subset for both backbones and all solvers"],
    }
    dump(out / "paired_results.json", results)
    receipt_payload["result_sha256"] = sha256_file(out / "paired_results.json")
    receipt_payload["exclusions"] = [
        {"backbone": bb, "window": r["window"], "sequence": r.get("sequence"), "reason": r["unavailable"]}
        for sec in sections.values() for bb, blk in sec.items() for r in blk["rows"] if "unavailable" in r
    ]
    dump(receipt, receipt_payload)  # overwrite with result hash + exclusions
    print("paired scoring complete")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["score"])
    ap.parse_args()
    return score()


if __name__ == "__main__":
    raise SystemExit(main())
