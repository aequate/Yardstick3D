"""Site replication R1 - GT-free window selection (primary pooled selection + vertical windows).

Primary: the frozen v5 amendment-A3 rule (v2_window_selection.select_windows_pooled, imported unchanged)
with the R1 minimum change: n_test = min(4, #available SV) if >= 2, else STOP.
Vertical: contract secondary_arms.altitude_ublox.windows.vertical_class, selected AFTER the primary
selection with the same within-sequence adjacency definition as v5.
Inputs: per-window cue-only statistics. No reference, image content or backbone output enters here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from yardstick3d.datasets import r1_contract as r1c
from yardstick3d.datasets import v2_window_selection as ws


@dataclass(frozen=True)
class R1Selection:
    status: str  # "OK" | "STOP"
    n_test: int
    n_sv_candidates: int
    starts: list
    minimum_rule_applied: bool
    reason: str | None = None

    def by_role(self, role: str) -> list:
        return [s for s in self.starts if s["role"] == role]


def select_r1(census_by_seq: dict, rank: list, n_target: int = r1c.N_TEST_TARGET,
              n_min: int = r1c.N_TEST_MIN) -> R1Selection:
    """Pooled selection with the R1 minimum rule.

    select_windows_pooled is greedy and prefix-consistent: its test set for n_test = n is the first n
    greedily accepted (non-adjacent) SV windows. So the number of available SV test windows is the largest
    n <= n_target for which it succeeds; n < n_min => STOP (no fallback, no replacement).
    """
    last_exc = None
    for n in range(n_target, n_min - 1, -1):
        try:
            sel = ws.select_windows_pooled(census_by_seq, list(rank), n_test=n,
                                           n_validation=r1c.N_VALIDATION, n_cc=r1c.N_CC)
        except ws.SequenceIneligible as exc:
            last_exc = exc
            continue
        return R1Selection("OK", n, sel.n_sv_candidates, sel.starts, n < n_target)
    n_sv = sum(1 for s in rank for w in census_by_seq[s] if w.cls == "SV")
    return R1Selection("STOP", 0, n_sv, [], True,
                       f"fewer than {n_min} non-adjacent pooled SV windows => R1 STOP ({last_exc})")


def vertical_eligible(v: dict, dh_min: float = r1c.VERTICAL_DH_MIN_M) -> tuple[bool, tuple]:
    """V-class eligibility: frames matched, cue coverage >= 0.8, valid_frac >= 0.8 (no v_mean), |dh| >= 10 m."""
    reasons = []
    if not v.get("frames_matched"):
        reasons.append("frames_not_matched_0.06s")
    if not (v.get("cue_coverage") is not None and v["cue_coverage"] >= r1c.COVERAGE_MIN):
        reasons.append("cue_coverage<0.8")
    if not (v.get("valid_frac") is not None and v["valid_frac"] >= r1c.COVERAGE_MIN):
        reasons.append("valid_frac<0.8")
    dh = v.get("dh_window")
    if dh is None or not np.isfinite(dh) or not abs(dh) >= dh_min:
        reasons.append(f"|dh_window|<{dh_min}")
    return (not reasons), tuple(reasons)


def select_vertical(census_by_seq: dict, rank: list, taken, n: int = r1c.N_VERTICAL,
                    dh_min: float = r1c.VERTICAL_DH_MIN_M) -> list:
    """Deterministic greedy V-window selection.

    census_by_seq: {seq: [{"k","t0","frames_matched","cue_coverage","valid_frac","dh_window"}]}.
    taken: iterable of (sequence, k) already selected in ANY role (the primary selection).
    Ranking: |dh_window| desc, ties -> sequence rank, then earlier t0. A candidate is skipped if (seq, k-1),
    (seq, k) or (seq, k+1) is already taken or chosen (v5 adjacency, within a sequence).
    """
    rk = {s: i for i, s in enumerate(rank)}
    occupied = {(str(s), int(k)) for s, k in taken}
    cands = []
    for seq in rank:
        for v in census_by_seq.get(seq, []):
            ok, _ = vertical_eligible(v, dh_min)
            if ok:
                cands.append((seq, v))
    cands.sort(key=lambda p: (-abs(float(p[1]["dh_window"])), rk[p[0]], float(p[1]["t0"])))
    out = []
    for seq, v in cands:
        if len(out) >= n:
            break
        k = int(v["k"])
        if any((seq, k + d) in occupied for d in (-1, 0, 1)):
            continue
        occupied.add((seq, k))
        out.append({"sequence": seq, "k": k, "t0": float(v["t0"]), "role": r1c.ROLE_VERTICAL,
                    "cls": "V", "S": None, "dh_window": float(v["dh_window"])})
    return out
