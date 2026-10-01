"""Site replication R1 - pooled selection with the minimum rule + vertical-window selection (synthetic only)."""
from __future__ import annotations

import random

import numpy as np
import pytest

from yardstick3d.datasets import r1_contract as r1c
from yardstick3d.datasets import r1_cues as cues
from yardstick3d.datasets import r1_selection as r1s
from yardstick3d.datasets import v2_window_selection as ws

POOL = list(r1c.POOL)


def _w(k, cls, S=0.2):
    elig = cls != "ineligible"
    return ws.WindowStats(k, 16.0 * k, elig, cls, True, 1.0, 1.0, 5.0, 0.3, 2.0, S, ())


def _census(sv_a, sv_b, n=20):
    """Each sequence: n windows, SV at the given k with S = 0.3 - 0.01 k (b slightly lower), rest 'other'/CC."""
    out = {}
    for seq, sv, off in ((POOL[0], sv_a, 0.0), (POOL[1], sv_b, 0.005)):
        out[seq] = [_w(k, "SV", 0.3 - 0.01 * k - off) if k in sv else _w(k, "CC" if k % 5 == 0 else "other")
                    for k in range(1, n + 1)]
    return out


@pytest.mark.parametrize("sv_a,sv_b,n_expected", [
    ({2, 6}, {3, 9}, 4),        # 4 SV -> 4
    ({2, 6}, {3}, 3),           # 3 -> 3 (minimum rule)
    ({2}, {3}, 2),              # 2 -> 2 (minimum rule)
    ({2}, set(), 0),            # 1 -> STOP
    (set(), set(), 0),          # 0 -> STOP
    ({2, 3, 4}, {7}, 3),        # adjacency: k=3 is never taken next to k=2/4 -> 3 non-adjacent
])
def test_minimum_rule(sv_a, sv_b, n_expected):
    sel = r1s.select_r1(_census(sv_a, sv_b), POOL)
    if n_expected == 0:
        assert sel.status == "STOP" and sel.starts == [] and "STOP" in sel.reason
        return
    assert sel.status == "OK" and sel.n_test == n_expected
    assert sel.minimum_rule_applied == (n_expected < 4)
    test = sel.by_role(r1c.ROLE_TEST)
    assert len(test) == n_expected and all(s["cls"] == "SV" for s in test)
    taken = [(s["sequence"], s["k"]) for s in sel.starts]
    assert len(set(taken)) == len(taken)
    for seq, k in taken:  # never adjacent within a sequence, in any role
        assert (seq, k + 1) not in taken and (seq, k - 1) not in taken
    assert len(sel.by_role(r1c.ROLE_VALIDATION)) == 2 and len(sel.by_role(r1c.ROLE_CC)) <= 2


def test_with_4_sv_r1_equals_frozen_v5_pooled_rule():
    c = _census({2, 6, 12}, {3, 9, 15})
    sel = r1s.select_r1(c, POOL)
    v5 = ws.select_windows_pooled(c, POOL, n_test=4, n_validation=2, n_cc=2)
    assert sel.starts == v5.starts and not sel.minimum_rule_applied
    # top-4 S across the pool (interleaved sequences); listed in (role, rank, t0) order
    assert [(s["sequence"], s["k"]) for s in sel.by_role(r1c.ROLE_TEST)] == [(POOL[0], 2), (POOL[0], 6), (POOL[1], 3), (POOL[1], 9)]
    # one bag unavailable: the rule runs on the remaining bag alone
    single = r1s.select_r1({POOL[1]: c[POOL[1]]}, [POOL[1]])
    assert single.status == "OK" and single.n_test == 3 and {s["sequence"] for s in single.starts} == {POOL[1]}


def _vcensus(dh_by_k, n=20, **override):
    rows = []
    for k in range(1, n + 1):
        r = {"k": k, "t0": 16.0 * k, "frames_matched": True, "cue_coverage": 1.0, "valid_frac": 1.0,
             "dh_window": dh_by_k.get(k, 0.5)}
        r.update(override.get(k, {}))
        rows.append(r)
    return rows


def test_vertical_selection_deterministic_and_never_neighbours_selected_windows():
    cen = {POOL[0]: _vcensus({4: 12.0, 5: -25.0, 6: 11.0, 9: 14.0, 10: 14.0, 15: 30.0}),
           POOL[1]: _vcensus({4: -12.0, 8: 20.0})}
    taken = [(POOL[0], 14), (POOL[0], 2), (POOL[1], 9)]  # primary windows (any role)
    v = r1s.select_vertical(cen, POOL, taken)
    got = [(x["sequence"], x["k"]) for x in v]
    # |dh| desc: GNSS02 k15 neighbours taken k14 -> skipped; k5 (-25, descent) ; GNSS03 k8 neighbours k9 -> skipped;
    # k9/k10 tie 14 -> k9 (earlier); k10 neighbours k9 -> skipped; GNSS02 k4/GNSS03 k4 tie 12 -> rank order
    assert got == [(POOL[0], 5), (POOL[0], 9), (POOL[1], 4)]
    assert all(x["role"] == r1c.ROLE_VERTICAL and x["cls"] == "V" for x in v)
    occupied = set(taken)
    for seq, k in got:
        assert not any((seq, k + d) in occupied for d in (-1, 0, 1))
        occupied.add((seq, k))
    for _ in range(5):  # deterministic under input reordering
        shuffled = {s: random.sample(rows, len(rows)) for s, rows in cen.items()}
        assert r1s.select_vertical(shuffled, POOL, list(reversed(taken))) == v
    assert len(r1s.select_vertical({POOL[0]: _vcensus({k: 50.0 for k in range(1, 21)}), POOL[1]: []},
                                   POOL, [])) == r1c.N_VERTICAL


def test_vertical_eligibility_no_vmean_requirement_and_rules():
    ok, why = r1s.vertical_eligible({"frames_matched": True, "cue_coverage": 0.8, "valid_frac": 0.8, "dh_window": -10.0})
    assert ok and why == ()  # no v_mean key at all: hovering climbs are eligible
    for bad, reason in (({"frames_matched": False}, "frames"), ({"cue_coverage": 0.79}, "cue_coverage"),
                        ({"valid_frac": 0.5}, "valid_frac"), ({"dh_window": 9.99}, "dh_window"),
                        ({"dh_window": float("nan")}, "dh_window"), ({"dh_window": None}, "dh_window")):
        v = {"frames_matched": True, "cue_coverage": 1.0, "valid_frac": 1.0, "dh_window": 20.0, **bad}
        ok, why = r1s.vertical_eligible(v)
        assert not ok and any(reason in w for w in why)


def test_vertical_windows_from_climb_track_are_selected_gt_free():
    """dh_window from a synthetic altitude profile: the climb/descent windows are the ones selected."""
    t = np.arange(0.0, 400.0, 0.1)
    h = 80.0 + np.interp(t, [0, 100, 140, 250, 290, 400], [0, 0, 40, 40, 0, 0])  # climb 1 m/s, descent 1 m/s
    rows = [{"k": k, "t0": 16.0 * k, "frames_matched": True, "cue_coverage": 1.0, "valid_frac": 1.0,
             "dh_window": cues.vertical_dh_window(t, h, 16.0 * k)} for k in range(1, 23)]
    v = r1s.select_vertical({POOL[0]: rows, POOL[1]: []}, POOL, [])
    ks = [x["k"] for x in v]
    # k7 = [112, 128] inside the climb, k16 = [256, 272] inside the descent (|dh| = 15 m each); k17 ties k16
    # but neighbours it; partial windows k6/k8 (11.5 m) neighbour k7. Level windows never qualify.
    assert ks == [7, 16] and v[0]["dh_window"] == pytest.approx(15.0) and v[1]["dh_window"] == pytest.approx(-15.0)
