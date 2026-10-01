"""Site replication R1 - one-shot scorer (primary arm + speed (Doppler) arm + altitude arm).

Contract: configs/prospective_site_replication_r1.yaml (scoring.order). The frozen v2/v5 scorer
(scripts/cross_backbone_v2_score.py) is loaded as a PRIVATE module instance and re-pointed at the R1 identity
(yardstick3d.datasets.r1_compat.V2ViewR1); its functions (verify_arms, read_reference, ref_valid_log,
evaluate_window, summarize, seq_sources) are reused UNCHANGED. Its score() is NOT used: unlike v5 it rewrote
the receipt after results; R1 never rewrites the receipt (completion record goes to a separate file).

Order (exactly):
  1. replay guard (SCORING_RECEIPT.json / SCORING_RECEIPT.sha256 must not exist)
  2. verify both arms (DA3 manifest, VGGT pack outputs) -> exit 3 AWAITING, nothing written, no reference
  3. GT-free diagnostics (rho, speed/displacement ratio, speed validity, altitude observability)
     -> r1_diagnostics.json, exclusive create (a stale one from an aborted pre-receipt run is archived as
     r1_diagnostics.<utc>.json + logged in r1_rerun_log.jsonl; once a receipt exists, refuse)
  4. SCORING_RECEIPT.json exclusive create, then SCORING_RECEIPT.sha256 (sha256 of the receipt bytes),
     exclusive create - both BEFORE any reference byte is read
  5. sealed reference sub-bags sha-verified (first reference byte read; Stage-1 verification skips them) and
     opened once
  6. ref_valid_log.json written BEFORE any metric
  7. metrics: primary (4 solvers) on test/validation/CC; speed arm (4 solvers, primary naive) on
     test/validation/CC; altitude arm (s_alt through the same evaluate_grounding) on test/validation/CC/vertical
  8. r1_results.json with per-arm verdicts (primary_gates / speed_arm_verdict / altitude_arm_verdict),
     then SCORING_COMPLETION.json

Usage: python scripts/site_replication_r1_score.py score
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import median as _median

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from yardstick3d.datasets import r1_contract as r1c  # noqa: E402
from yardstick3d.datasets import r1_cues as cues  # noqa: E402
from yardstick3d.datasets import v2_scoring as v2  # noqa: E402
from yardstick3d.datasets.r1_compat import V2ViewR1  # noqa: E402
from yardstick3d.experiments.advio_real import evaluate_grounding  # noqa: E402


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cb = _load("r1_cb_v2_score", "scripts/cross_backbone_v2_score.py")       # private instance; file untouched
claim1 = _load("r1_claim1_gates", "scripts/claim1_gate_metrics_v2.py")   # v5 gate arithmetic, unchanged
if (claim1.LEVERAGE_MIN, claim1.R_MIN, claim1.RED_MIN, claim1.ORACLE_RATIO_MAX) != (
        r1c.LEVERAGE_MIN_M, r1c.R_MIN, r1c.REDUCTION_MIN, r1c.ORACLE_RATIO_MAX):
    raise RuntimeError("frozen claim1 gate thresholds differ from the R1 contract constants")

OUT = ROOT / r1c.OUT_REL
PACK = ROOT / r1c.PACK_REL
CONTRACT = r1c.CONTRACT_PATH
REGISTRY = r1c.REGISTRY_PATH
BACKBONES = cb.BACKBONES
SOLVERS = cb.SOLVERS
PRIMARY_ROLES = (r1c.ROLE_TEST, r1c.ROLE_VALIDATION, r1c.ROLE_CC)
ALL_ROLES = (*PRIMARY_ROLES, r1c.ROLE_VERTICAL)
ArmsNotReady = cb.ArmsNotReady
dump = v2.dump_json
sha256_file = v2.sha256_file


def bind(contract: Path = CONTRACT, registry: Path | None = REGISTRY) -> V2ViewR1:
    """Point the private v2-scorer instance at the R1 identity (pack id, pool, cruise table, contract sha)."""
    view = V2ViewR1(contract, registry)
    cb.v2 = view
    cb.CONTRACT = Path(contract)
    cb.CONTRACT_SHA = view.CONTRACT_SHA
    cb.SEQ = view.SEQUENCE
    return view


def _sections(roles) -> dict:
    return {r1c.SECTION_OF_ROLE[r]: {} for r in roles}


def _med(vals):
    v = [float(x) for x in vals if x is not None and np.isfinite(x)]
    return float(_median(v)) if v else None


def load_pvt_csv(path: Path) -> dict:
    import csv

    cols = {k: [] for k in ("record_t", "t_utc", "vel_n", "vel_e", "vel_d")}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            for k in cols:
                cols[k].append(float(row[k]))
    return {k: np.asarray(v, float) for k, v in cols.items()}


def _window_enu(w, cue_t, cue_llh):
    from yardstick3d.datasets import mars_lvig as ml

    tsw = np.asarray(w["frame_times_utc"], float)
    lat0, lon0, alt0 = v2.window_origin(cue_t, cue_llh, float(tsw[0]))
    alt = np.where(np.isfinite(cue_llh[:, 2]), cue_llh[:, 2], 0.0)
    return ml.cue_enu_from_llh(cue_llh[:, 0], cue_llh[:, 1], alt, lat0, lon0, alt0), (lat0, lon0, alt0)


def _gt_subset(w, rv, cue_t, cue_llh, ref):
    """Reference centres of the REF-VALID frames in the window's cue ENU frame (as cb.evaluate_window)."""
    from yardstick3d.datasets import mars_lvig as ml

    _, (lat0, lon0, alt0) = _window_enu(w, cue_t, cue_llh)
    ref_enu = ml.cue_enu_from_llh(ref["lat"], ref["lon"], ref["alt"], lat0, lon0, alt0)
    return ref_enu[rv["ref_index"][rv["mask"]]]


# --------------------------- 3. GT-free diagnostics ---------------------------
def gt_free_diagnostics(wins, cue_by_seq, pvt_by_seq, speed_run, loaders) -> dict:
    """Cue + prediction ONLY. Speed validity is cue-only; altitude decisions are per backbone."""
    from yardstick3d.datasets import mars_lvig as ml

    speed, rho, alt = [], {bb: [] for bb in BACKBONES}, {bb: [] for bb in BACKBONES}
    for i, w in enumerate(wins):
        seq = w["sequence"]
        ct, cl = cue_by_seq[seq]
        tsw = np.asarray(w["frame_times_utc"], float)
        enu, _ = _window_enu(w, ct, cl)
        if w["role"] in PRIMARY_ROLES:
            rec = {"window": i, "sequence": seq, "role": w["role"]}
            if not speed_run.get(seq):
                rec.update({"arm_run": False, "valid": False, "reason": "speed arm NOT RUN for sequence (PVT gate/decode)"})
            else:
                p = pvt_by_seq[seq]
                spd = cues.pvt_speed_3d(p["vel_n"], p["vel_e"], p["vel_d"])
                _, info = cues.speed_constraints(tsw, p["t_utc"], spd)
                b_v, usable = cues.speed_interval_lengths(tsw, p["t_utc"], spd)
                b_d, cov_d = ml.cue_polyline_lengths(ct, enu, tsw)
                rec.update({"arm_run": True, "valid": info["valid"], "coverage": info["coverage"],
                            "n_kept": info["n_kept"], "cue_lengths_m": info["cue_lengths_m"],
                            "speed_over_displacement": cues.speed_displacement_ratio(b_v, usable, b_d, cov_d)})
            speed.append(rec)
        for bb in BACKBONES:
            pred = loaders[bb](i)
            if w["role"] in PRIMARY_ROLES:
                rho[bb].append({"window": i, "sequence": seq, "role": w["role"],
                                **v2.rho_diagnostic(pred.centers(), tsw, ct, enu)})
            T = np.asarray(pred.T_w2c, float)
            alt[bb].append({"window": i, "sequence": seq, "role": w["role"],
                            **cues.altitude_arm(tsw, ct, cl[:, 2], pred.centers(), T[:, :3, :3])})
    return {"speed": speed, "rho": rho, "altitude": alt}


# ------------------------------ 7. arm metrics --------------------------------
def evaluate_speed_window(w, i, pred, rv, cue_t, cue_llh, ref, pvt) -> dict:
    """Speed arm: same grounder + solvers on PathDisplacementConstraint(b_k^v); same REF-VALID subset."""
    from yardstick3d.optimization.grounder import MetricGrounder

    tsw = np.asarray(w["frame_times_utc"], float)
    spd = cues.pvt_speed_3d(pvt["vel_n"], pvt["vel_e"], pvt["vel_d"])
    cs, info = cues.speed_constraints(tsw, pvt["t_utc"], spd)
    gt_c = _gt_subset(w, rv, cue_t, cue_llh, ref)
    sub = v2.SubsetPred(pred, rv["mask"])
    g = MetricGrounder()
    row = {"window": i, "sequence": w["sequence"], "role": w["role"], "t0": w["t0"], "cue_topic": r1c.PVT_TOPIC,
           "n_ref_valid": int(rv["mask"].sum()),
           "cue": {"n_kept": info["n_kept"], "n_dropped": info["n_dropped"], "coverage": info["coverage"]},
           "solver": {}, "three_limits": {"s_cue": None}}
    for name in SOLVERS:
        r = g.solve(pred, cs, solver=name)
        s = float(r.scale) if np.isfinite(r.scale) else None
        row["solver"][name] = s
        row[f"eval_{name}"] = evaluate_grounding(sub, gt_c, r.scale)
        if s is None:
            row.setdefault("failures", []).append(f"{name}: non-finite scale (counted, no substitution)")
    return row


def evaluate_altitude_window(w, i, pred, rv, cue_t, cue_llh, ref, decision: dict) -> dict:
    """Altitude arm, using the PRE-RECORDED GT-free decision. OBSERVABLE -> s_alt; FAILURE -> raw ATE (scale 1,
    counted against the medians); NOT_OBSERVABLE -> no scale applied, not scored."""
    row = {"window": i, "sequence": w["sequence"], "role": w["role"], "t0": w["t0"], "cue_topic": r1c.LLA_TOPIC,
           "altitude": decision, "cue": {"status": decision["status"]}, "three_limits": {"s_cue": None}}
    if decision["status"] == r1c.ALT_NOT_OBSERVABLE:
        row["not_scored"] = "NOT_OBSERVABLE (no scale applied)"
        return row
    applied = float(decision["s_alt"]) if decision["status"] == r1c.ALT_OBSERVABLE else 1.0
    gt_c = _gt_subset(w, rv, cue_t, cue_llh, ref)
    sub = v2.SubsetPred(pred, rv["mask"])
    row["applied_scale"] = applied
    row["solver"] = {"naive": decision["s_alt"]}
    row["eval_none"] = evaluate_grounding(sub, gt_c, 1.0)
    row["eval_naive"] = evaluate_grounding(sub, gt_c, applied)
    return row


def sim3_coreport(row: dict) -> dict:
    """Sim(3) min-ATE oracle + scale error vs the ATE-optimal scale (R1 metrics; v5 caveats 3, 11)."""
    e = row.get("eval_naive") or {}
    s_hat, s_opt = (row.get("solver") or {}).get("naive"), e.get("oracle_scale_ate_optimal")
    err = abs(s_hat - s_opt) / s_opt if (s_hat is not None and s_opt and np.isfinite(s_opt)) else None
    return {"oracle_ate_sim3_min": e.get("oracle_ate_optimal"), "oracle_scale_ate_optimal": s_opt,
            "scale_error_rel_vs_ate_optimal": err}


# --------------------------------- verdicts -----------------------------------
def _gate_block(rows_by_bb: dict) -> dict:
    """claim1 gate arithmetic (v5, unchanged) on {bb: rows}; plus the Sim(3) min-ATE co-report."""
    g = claim1.gates({"sections": {claim1.SECTION: {bb: {"rows": rows} for bb, rows in rows_by_bb.items()}}})
    for bb, blk in g.items():
        ev = {r["window"] for r in blk["windows"] if r["evaluable"] and r["capture_R"] is not None}
        src = [r for r in rows_by_bb[bb] if r.get("window") in ev and "eval_naive" in r]
        med_g = _med(r["eval_naive"]["grounded_ate_se3"] for r in src)
        med_s3 = _med(r["eval_naive"].get("oracle_ate_optimal") for r in src)
        blk["sim3_coreport"] = {
            "median_oracle_ate_sim3_min": med_s3,
            "grounded_over_sim3_oracle": med_g / med_s3 if (med_g is not None and med_s3) else None,
            "gate7_vs_sim3_oracle_pass_reported_only": bool(med_g is not None and med_s3 is not None
                                                            and med_g <= r1c.ORACLE_RATIO_MAX * med_s3),
            "median_scale_error_rel_vs_ate_optimal": _med(sim3_coreport(r)["scale_error_rel_vs_ate_optimal"] for r in src),
        }
    return g


def primary_gates(sections: dict, n_test: int) -> dict:
    """R1-6 / R1-7 per backbone over evaluable SV test windows (binding: v5 path-length oracle; Sim(3) co-reported)."""
    g = _gate_block({bb: blk["rows"] for bb, blk in sections["sv_test"].items()})
    r16 = all(b["gate6_pass"] for b in g.values())
    r17 = all(b["gate7_pass"] for b in g.values())
    return {"backbones": g, "R1-6_pass": r16, "R1-7_pass": r17,
            "outcome_if_integrity_pass": "PRIMARY_REPLICATED (subject to R1-1..R1-5, R1-9, R1-10 and R1-12)"
            if (r16 and r17) else "PRIMARY_NOT_REPLICATED",
            "n_test": n_test,
            "small_n_binding_caveat": n_test < r1c.N_TEST_TARGET}


def speed_arm_verdict(sv_rows_by_bb: dict, arm_ran: bool) -> dict:
    """REPLICATED iff BOTH backbones pass R1-6/R1-7 thresholds over evaluable SV test windows; INCONCLUSIVE iff
    the arm did not run or < 2 SV test windows are valid for it (valid = coverage >= 0.8 AND REF-evaluable)."""
    n_valid = {bb: sum(1 for r in rows if "eval_naive" in r) for bb, rows in sv_rows_by_bb.items()}
    if not arm_ran:
        return {"verdict": "INCONCLUSIVE", "reason": "speed arm did not run (PVT gate FAIL/undecodable on every used sequence)",
                "n_valid": n_valid}
    if any(n < r1c.MIN_ARM_WINDOWS for n in n_valid.values()):
        return {"verdict": "INCONCLUSIVE", "reason": f"< {r1c.MIN_ARM_WINDOWS} valid SV test windows", "n_valid": n_valid}
    g = _gate_block(sv_rows_by_bb)
    ok = all(b["gate6_pass"] and b["gate7_pass"] for b in g.values())
    return {"verdict": "REPLICATED" if ok else "NOT_REPLICATED", "n_valid": n_valid, "backbones": g}


def altitude_arm_verdict(v_rows_by_bb: dict) -> dict:
    """REPLICATED iff, for BOTH backbones, >= 2 V windows are OBSERVABLE (OBSERVABLE or FAILURE, REF-evaluable) and
    the R1-6/R1-7 thresholds pass over them (FAILURE = raw ATE); INCONCLUSIVE if < 2 for either backbone."""
    obs = {bb: [r for r in rows if "eval_naive" in r] for bb, rows in v_rows_by_bb.items()}
    n_obs = {bb: len(rows) for bb, rows in obs.items()}
    n_fail = {bb: sum(1 for r in rows if r["altitude"]["status"] == r1c.ALT_FAILURE) for bb, rows in obs.items()}
    if any(n < r1c.MIN_ARM_WINDOWS for n in n_obs.values()):
        return {"verdict": "INCONCLUSIVE", "reason": f"< {r1c.MIN_ARM_WINDOWS} observable V windows",
                "n_observable": n_obs, "n_failure": n_fail}
    g = _gate_block(obs)
    for bb, rows in obs.items():  # disclosure: medians over ALL observable windows (no leverage filter)
        g[bb]["all_observable_coreport"] = {
            "median_reduction": _med((r["eval_naive"]["raw_ate_se3"] - r["eval_naive"]["grounded_ate_se3"])
                                     / r["eval_naive"]["raw_ate_se3"] for r in rows if r["eval_naive"]["raw_ate_se3"]),
            "median_grounded_ate": _med(r["eval_naive"]["grounded_ate_se3"] for r in rows),
            "median_oracle_ate": _med(r["eval_naive"]["oracle_ate_se3"] for r in rows)}
    ok = all(b["gate6_pass"] and b["gate7_pass"] for b in g.values())
    return {"verdict": "REPLICATED" if ok else "NOT_REPLICATED", "n_observable": n_obs, "n_failure": n_fail,
            "backbones": g}


# ---------------------------------- scoring -----------------------------------
def _unavail(w, i, reason):
    return {"window": i, "sequence": w["sequence"], "role": w["role"], "t0": w["t0"], "unavailable": reason}


def compute_metrics(wins, logs, cue_by_seq, pvt_by_seq, speed_run, ref, loaders, diag) -> dict:
    alt_dec = {bb: {d["window"]: d for d in diag["altitude"][bb]} for bb in BACKBONES}
    spd_dec = {d["window"]: d for d in diag["speed"]}
    primary, speed, altitude = _sections(PRIMARY_ROLES), _sections(PRIMARY_ROLES), _sections(ALL_ROLES)
    for bb in BACKBONES:
        rows_p = {s: [] for s in primary}
        rows_s = {s: [] for s in speed}
        rows_a = {s: [] for s in altitude}
        for i, w in enumerate(wins):
            seq, sec, L = w["sequence"], r1c.SECTION_OF_ROLE[w["role"]], logs[i]
            ct, cl = cue_by_seq[seq]
            r = cb._per_seq(ref, w)
            if not L["evaluable"]:
                why = f"REF-VALID frames {L['n_fixed']}/8 < {v2.REF_VALID_REQUIRED}"
                for rows in (rows_p, rows_s) if w["role"] in PRIMARY_ROLES else ():
                    rows[sec].append(_unavail(w, i, why))
                rows_a[sec].append(_unavail(w, i, why))
                continue
            pred = loaders[bb](i)
            if w["role"] in PRIMARY_ROLES:
                row = cb.evaluate_window(w, i, pred, L["_rv"], ct, cl, r)
                row["sim3_coreport"] = sim3_coreport(row)
                rows_p[sec].append(row)
                d = spd_dec[i]
                if not d["arm_run"]:
                    rows_s[sec].append(_unavail(w, i, d["reason"]))
                elif not d["valid"]:
                    rows_s[sec].append(_unavail(w, i, f"speed coverage {d['coverage']:.3f} < {r1c.COVERAGE_MIN}"))
                else:
                    row = evaluate_speed_window(w, i, pred, L["_rv"], ct, cl, r, pvt_by_seq[seq])
                    row["sim3_coreport"] = sim3_coreport(row)
                    rows_s[sec].append(row)
            row = evaluate_altitude_window(w, i, pred, L["_rv"], ct, cl, r, alt_dec[bb][i])
            if "eval_naive" in row:
                row["sim3_coreport"] = sim3_coreport(row)
            rows_a[sec].append(row)
        for s, rows in rows_p.items():
            primary[s][bb] = {"rows": rows, "summary": cb.summarize(rows)}
        for s, rows in rows_s.items():
            speed[s][bb] = {"rows": rows}
        for s, rows in rows_a.items():
            altitude[s][bb] = {"rows": rows}
    return {"primary": primary, "speed_doppler": speed, "altitude_ublox": altitude}


def reference_entries(man: dict) -> set:
    """Stage-1 output names that hold reference bytes (sealed sub-bags): never read before the receipt."""
    names = {info["reference_source"]["path"] for info in man.get("sequences", {}).values()
             if (info.get("reference_source") or {}).get("kind") == "sealed_sub_bag"}
    names |= {n for n in man.get("output_sha256", {}) if Path(n).name.startswith("ref_") and n.endswith(".bag")}
    return names


def _verify_stage1(s1: Path, sha: str) -> tuple[dict, list]:
    """Pre-receipt Stage-1 verification. Reference sub-bags are SKIPPED here (contract: no reference byte before the
    receipt); they are sha-checked in verify_reference_sources() after SCORING_RECEIPT.json + .sha256 exist."""
    man = json.loads((s1 / "stage1_manifest.json").read_text(encoding="utf-8"))
    if man.get("contract_sha256") != sha:
        raise RuntimeError("stage1 manifest contract sha != registered R1 contract sha")
    if man.get("dry_run") is not False or man.get("reference_values_read") is not False:
        raise RuntimeError("stage1 manifest is a dry run or lacks the no-reference attestation")
    deferred = reference_entries(man)
    for name, dg in man.get("output_sha256", {}).items():
        if name in deferred:
            continue
        if sha256_file(s1 / name) != dg:
            raise RuntimeError(f"stage1 output hash mismatch: {name}")
    if sha256_file(s1 / "windows.json") != man["windows_json_sha256"]:
        raise RuntimeError("windows.json changed after Stage 1")
    wins = json.loads((s1 / "windows.json").read_text(encoding="utf-8"))["windows"]
    if any(w["role"] not in ALL_ROLES or "sequence" not in w for w in wins):
        raise RuntimeError("unexpected window role / missing sequence")
    return man, wins


def verify_reference_sources(s1: Path, man: dict, sources: dict, receipt_sha: Path) -> None:
    """Post-receipt only: each sealed sub-bag must match BOTH its reference_source sha and its output_sha256 entry."""
    if not receipt_sha.exists():
        raise RuntimeError("reference sub-bag hash requested before SCORING_RECEIPT.sha256 exists")
    for s, src in sources.items():
        rs = man["sequences"][s]["reference_source"]
        if rs["kind"] != "sealed_sub_bag":
            raise RuntimeError(f"R1 reference must be a sealed sub-bag ({s})")
        digest = sha256_file(src[0])
        listed = man.get("output_sha256", {}).get(rs["path"])
        if digest != src[3] or (listed is not None and digest != listed):
            raise RuntimeError(f"reference source hash mismatch for {s}: {src[0]}")


def archive_stale_diagnostics(out: Path) -> dict | None:
    """MIN-9 recovery: r1_diagnostics.json without a receipt = an aborted pre-receipt run. It is archived as
    r1_diagnostics.<utc>.json and logged in r1_rerun_log.jsonl (never deleted). Diagnostics are GT-free, so a
    rerun reveals nothing new; the receipt records the log sha."""
    diag = out / "r1_diagnostics.json"
    if not diag.exists():
        return None
    if (out / "SCORING_RECEIPT.json").exists() or (out / "SCORING_RECEIPT.sha256").exists():
        raise FileExistsError(f"scoring receipt exists; refusing replay: {out}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    dst = out / f"r1_diagnostics.{stamp}.json"
    k = 1
    while dst.exists():
        dst, k = out / f"r1_diagnostics.{stamp}_{k}.json", k + 1
    digest = sha256_file(diag)
    diag.rename(dst)
    entry = {"utc": datetime.now(timezone.utc).isoformat(), "event": "aborted pre-receipt run; diagnostics archived",
             "archived_as": dst.name, "sha256": digest, "receipt_existed": False}
    with open(out / "r1_rerun_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")
    return entry


def score(out: Path | None = None, pack: Path | None = None, contract: Path = CONTRACT,
          registry: Path | None = REGISTRY, reference_reader=None) -> int:
    from yardstick3d.io.prediction_cache import load_prediction

    out, pack = Path(out or OUT), Path(pack or PACK)
    sha = r1c.check_registered(contract, registry)  # frozen contract only
    bind(contract, registry)
    receipt, receipt_sha = out / "SCORING_RECEIPT.json", out / "SCORING_RECEIPT.sha256"
    v2.guard_receipt(receipt)  # 1. replay refusal
    v2.guard_receipt(receipt_sha)

    s1 = out / "stage1"
    man, wins = _verify_stage1(s1, sha)
    try:
        da3_by_idx, vggt_paths, _ = cb.verify_arms(out, pack, wins)  # 2.
    except ArmsNotReady as exc:
        print(f"AWAITING: {exc}; scoring withheld (no reference opened, nothing written)")
        return 3

    seqs = sorted({w["sequence"] for w in wins}, key=r1c.POOL.index)
    cue_by_seq, pvt_by_seq, speed_run = {}, {}, {}
    for s in seqs:
        info = man["sequences"][s]
        cue_by_seq[s] = cb.load_cue(s1 / info["cue_csv"])
        pv = info.get("pvt") or {}
        speed_run[s] = pv.get("speed_arm") == "RUN"
        if speed_run[s]:
            if sha256_file(s1 / pv["csv"]) != pv["csv_sha256"]:
                raise RuntimeError(f"pvt csv hash mismatch for {s}")
            pvt_by_seq[s] = load_pvt_csv(s1 / pv["csv"])
    loaders = {"da3-base": lambda i: load_prediction(da3_by_idx[i]["path"]),
               "vggt-1b": lambda i: load_prediction(vggt_paths[i])}

    # 3. GT-free diagnostics, exclusive create, BEFORE the receipt (a stale file from an aborted run is archived)
    archive_stale_diagnostics(out)
    diag = gt_free_diagnostics(wins, cue_by_seq, pvt_by_seq, speed_run, loaders)
    dump(out / "r1_diagnostics.json", {"contract_sha256": sha, "reference_values_read": False,
                                       "speed_arm_by_sequence": {s: speed_run[s] for s in seqs}, **diag},
         exclusive=True)

    # 4. receipt + its sha256 file, BEFORE any reference byte
    payload = {
        "contract_sha256": sha, "experiment": r1c.EXPERIMENT_NAME, "sequences": seqs, "evaluation_count": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "evaluator_revision": "advio_real.evaluate_grounding (+ v2 REF-VALID subset); claim1 gate arithmetic",
        "da3_manifest_sha256": sha256_file(out / "da3_manifest.json"),
        "stage1_manifest_sha256": sha256_file(s1 / "stage1_manifest.json"),
        "r1_diagnostics_sha256": sha256_file(out / "r1_diagnostics.json"),
        "vggt_input_manifest_sha256": sha256_file(pack / "input_hashes.json"),
        "vggt_prediction_sha256": {str(k): sha256_file(v) for k, v in vggt_paths.items()},
        "da3_prediction_sha256": {str(k): r["prediction_sha256"] for k, r in da3_by_idx.items()},
        "evaluated_windows": [f"{w['role']}:{w['sequence']}:{w['t0']}" for w in wins],
        "rerun_log_sha256": sha256_file(out / "r1_rerun_log.jsonl") if (out / "r1_rerun_log.jsonl").exists() else None,
    }
    dump(receipt, payload, exclusive=True)
    digest = hashlib.sha256(receipt.read_bytes()).hexdigest()
    with open(receipt_sha, "x", encoding="utf-8") as f:
        f.write(f"{digest}  SCORING_RECEIPT.json\n")

    # 5. reference opened once (sealed sub-bags, sha-verified)
    sources = cb.seq_sources(s1, man, wins, Path("unused"))
    verify_reference_sources(s1, man, sources, receipt_sha)  # first reference byte read: AFTER the receipt
    reader = reference_reader or cb.read_reference
    ref = cb.per_sequence({s: reader(src[0], src[1], src[2]) for s, src in sources.items()})

    # 6. REF-VALID log BEFORE metrics
    logs = cb.ref_valid_log(wins, ref)
    ref_log = [{k: v for k, v in L.items() if k != "_rv"} for L in logs]
    dump(out / "ref_valid_log.json", {"fixed_values": list(v2.FIXED_VALUES), "max_dt_s": v2.MAX_DT_REF_S,
                                      "required": v2.REF_VALID_REQUIRED, "windows": ref_log}, exclusive=True)

    # 7. metrics
    m = compute_metrics(wins, logs, cue_by_seq, pvt_by_seq, speed_run, ref, loaders, diag)

    # 8. verdicts + results
    n_test = sum(1 for w in wins if w["role"] == r1c.ROLE_TEST)
    verdicts = {
        "primary": primary_gates(m["primary"], n_test),
        "speed_doppler": speed_arm_verdict({bb: blk["rows"] for bb, blk in m["speed_doppler"]["sv_test"].items()},
                                           any(speed_run.values())),
        "altitude_ublox": altitude_arm_verdict({bb: blk["rows"] for bb, blk in m["altitude_ublox"]["vertical"].items()}),
    }
    results = {"contract_sha256": sha, "experiment": r1c.EXPERIMENT_NAME, "sequences": seqs,
               "backbones": list(BACKBONES), "primary_solver": "naive",
               "split": {r: sum(1 for w in wins if w["role"] == r) for r in ALL_ROLES},
               "minimum_rule_applied": man.get("minimum_rule_applied"),
               "speed_arm_by_sequence": speed_run, "arms": m, "verdicts": verdicts, "ref_valid_log": ref_log,
               "reference_values_read": True,
               "notes": ["secondary arms never enter the Claim-1 gates",
                         "validation / cc_control sections are not claims; vertical windows score only the altitude arm",
                         "ATE on REF-VALID frames only, identical subset for both backbones, all solvers and all arms"]}
    dump(out / "r1_results.json", results, exclusive=True)
    exclusions = [{"arm": arm, "backbone": bb, "window": r["window"], "sequence": r["sequence"], "reason": r["unavailable"]}
                  for arm, secs in m.items() for sec in secs.values() for bb, blk in sec.items()
                  for r in blk["rows"] if "unavailable" in r]
    dump(out / "SCORING_COMPLETION.json", {"receipt_sha256": digest, "result_sha256": sha256_file(out / "r1_results.json"),
                                           "exclusions": exclusions}, exclusive=True)
    print("R1 scoring complete:", {k: v.get("verdict", v.get("outcome_if_integrity_pass")) for k, v in verdicts.items()})
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["score"])
    ap.parse_args(argv)
    return score()


if __name__ == "__main__":
    raise SystemExit(main())
