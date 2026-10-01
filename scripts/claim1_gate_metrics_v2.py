"""Compute pre-registered Claim-1 gates 6-8 from the one-shot v5 (pooled) paired_results.json (read-only; no rescoring).

SV test windows ONLY (section `sv_test`); validation and CC-control sections are never gated.
Thresholds are the v1 pre-registered values, unchanged:
leverage >= 0.25 m, median R >= 0.70, median reduction >= 30 %, median grounded <= 1.5 x median oracle.
Windows excluded at scoring by the REF-VALID rule (`unavailable`) are disclosed (gate 9), never replaced.
"""
import json
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "artifacts/cross_backbone_v5/paired_results.json"
OUT = ROOT / "artifacts/cross_backbone_v5/claim1_gate_metrics.json"
LEVERAGE_MIN, R_MIN, RED_MIN, ORACLE_RATIO_MAX = 0.25, 0.70, 0.30, 1.5
PRIMARY = "eval_naive"
SECTION = "sv_test"


def _window_row(w):
    e = w[PRIMARY]
    raw, gr, orc = e["raw_ate_se3"], e["grounded_ate_se3"], e["oracle_ate_se3"]
    if raw is None or gr is None or orc is None:  # non-finite metric (JSON null): counted, never substituted
        lev, ok = None, False
    else:
        lev = raw - orc
        ok = lev >= LEVERAGE_MIN
    return {
        "window": w["window"], "role": w.get("role"), "cue": w.get("cue"),
        "raw_ate_se3": raw, "grounded_ate_se3": gr, "oracle_ate_se3": orc,
        "sim3_ate": e.get("raw_ate_sim3"), "leverage": lev, "evaluable": bool(ok),
        "capture_R": (raw - gr) / lev if ok else None,
        "reduction": (raw - gr) / raw if (raw and gr is not None) else None,
        "scale_hat": w["solver"]["naive"], "oracle_scale": e.get("oracle_scale"),
        "scale_error_rel_vs_oracle": e.get("scale_error_rel_vs_oracle"),
        "s_cue": w["three_limits"]["s_cue"],
        "nominal_prior_ate_se3": (w.get("nominal_speed_prior") or {}).get("grounded_ate_se3"),
    }


def gates(d):
    """d: v2 paired_results dict. Returns {backbone: gate block} over the SV test section only."""
    out = {}
    for b, blk in d["sections"][SECTION].items():
        rows_all = blk["rows"]
        excluded = [{"window": w["window"], "reason": w["unavailable"]} for w in rows_all if "unavailable" in w]
        rows = [_window_row(w) for w in rows_all if PRIMARY in w]
        ev = [r for r in rows if r["evaluable"] and r["capture_R"] is not None]
        med_R = median(r["capture_R"] for r in ev) if ev else None
        med_red = median(r["reduction"] for r in ev) if ev else None
        med_g = median(r["grounded_ate_se3"] for r in ev) if ev else None
        med_o = median(r["oracle_ate_se3"] for r in ev) if ev else None
        g6 = bool(ev) and med_R >= R_MIN and med_red >= RED_MIN
        g7 = bool(ev) and med_g <= ORACLE_RATIO_MAX * med_o
        out[b] = {
            "windows": rows,
            "n_test": len(rows_all), "n_scored": len(rows), "n_evaluable_test": len(ev),
            "excluded_ref_valid": excluded,
            "shape_limited_test_windows": [r["window"] for r in rows if not r["evaluable"]],
            "median_capture_R": med_R, "median_reduction": med_red,
            "median_grounded_ate": med_g, "median_oracle_ate": med_o,
            "median_leverage": median(r["leverage"] for r in ev) if ev else None,
            "grounded_over_oracle": med_g / med_o if (ev and med_o) else None,
            "gate6_pass": g6, "gate7_pass": g7,
        }
    return out


if __name__ == "__main__":
    d = json.loads(RES.read_text())
    g = gates(d)
    payload = {"source": str(RES.relative_to(ROOT)), "primary_solver": "naive", "section": SECTION,
               "thresholds": {"leverage_min_m": LEVERAGE_MIN, "median_R_min": R_MIN,
                              "median_reduction_min": RED_MIN, "grounded_over_oracle_max": ORACLE_RATIO_MAX},
               "backbones": g,
               "gate6_pass_all": all(v["gate6_pass"] for v in g.values()),
               "gate7_pass_all": all(v["gate7_pass"] for v in g.values())}
    OUT.write_text(json.dumps(payload, indent=2))
    for b, v in g.items():
        print(b, {k: (round(x, 4) if isinstance(x, float) else x) for k, x in v.items() if k != "windows"})
