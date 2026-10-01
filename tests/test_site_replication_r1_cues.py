"""Site replication R1 - contract constants, topic allow-list and secondary-arm cue tests (synthetic only).

Covers configs/prospective_site_replication_r1.yaml integrity.required_tests_before_freeze items:
GNSS week/tow -> UTC; Doppler speed integration; speed-arm grounding; altitude arm; topic allow-list.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from yardstick3d.datasets import r1_contract as r1c
from yardstick3d.datasets import r1_cues as cues

ROOT = Path(__file__).resolve().parents[1]
TEXT = r1c.CONTRACT_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------- contract
def test_contract_constants_match_yaml_text():
    c = r1c.load_contract()
    assert c["rank_order"] == list(r1c.POOL) and c["sha256"] == r1c.contract_sha()
    assert {p["sequence"]: p["v_cruise_table_mps"] for p in c["pool"]} == r1c.V_CRUISE_TABLE_BY_SEQ
    expected = [
        f"tow - {r1c.GPST_MINUS_UTC_S} s (published constant; never fitted)",
        f"|median(t_utc - (record + delta))| < {r1c.PVT_GATE_MEDIAN_MAX_S} s AND (P99 - P1) of that residual < {r1c.PVT_GATE_SPREAD_MAX_S} s",
        f"usable iff no sample gap > {r1c.SPEED_GAP_MAX_S} s inside it",
        f">= {int(r1c.COVERAGE_MIN * 100)}% of its 7 intervals are usable",
        f"OBSERVABLE iff |dh| >= {r1c.ALT_DH_MIN_M} m AND |dv| / ||C_last - C_first|| >= {r1c.ALT_RATIO_MIN}",
        f"no interpolation across a gap > {r1c.ALT_GAP_MAX_S} s",
        f"|dh_window| >= {r1c.VERTICAL_DH_MIN_M} m",
        "median altitude over [t0+15, t0+16] - median over [t0, t0+1]",
        f"n_vertical = up to {r1c.N_VERTICAL} pooled",
        f"first n_test = {r1c.N_TEST_TARGET} SV windows",
        f"ALL SV windows it selects (>= {r1c.N_TEST_MIN}) are the test set",
        f"validation = next {r1c.N_VALIDATION} SV", f"CC controls up to {r1c.N_CC}",
        f"median R >= {r1c.R_MIN:.2f} and median reduction >= {int(r1c.REDUCTION_MIN * 100)}%",
        f"(leverage >= {r1c.LEVERAGE_MIN_M} m)", f"median grounded <= {r1c.ORACLE_RATIO_MAX} x median oracle",
        f"< {r1c.MIN_ARM_WINDOWS} SV test windows are valid", f"< {r1c.MIN_ARM_WINDOWS} V windows are OBSERVABLE",
        f"output_dir: {r1c.OUT_REL}", f"topic: {r1c.PVT_TOPIC}", f"topic: {r1c.LLA_TOPIC}",
        "fixed_values [50], 0.15 s match, >= 7/8 REF-VALID",
    ]
    for s in expected:
        assert s in TEXT, s
    assert r1c.EXPERIMENT_NAME in TEXT and r1c.CAMERA_TOPIC in TEXT
    assert r1c.GPS_EPOCH_UNIX_S == 315964800.0 and r1c.SECONDS_PER_WEEK == 7 * 86400


def test_contract_registration_is_required(tmp_path):
    reg = tmp_path / "reg.json"
    reg.write_text('{"experiments": []}')
    with pytest.raises(r1c.ContractNotRegistered):
        r1c.check_registered(r1c.CONTRACT_PATH, reg)
    with pytest.raises(r1c.ContractNotRegistered):
        r1c.check_registered(r1c.CONTRACT_PATH, None)
    reg.write_text('{"experiments": [{"contract_sha256": "%s"}]}' % r1c.contract_sha())
    assert r1c.check_registered(r1c.CONTRACT_PATH, reg) == r1c.contract_sha()
    tampered = tmp_path / "c.yaml"
    tampered.write_bytes(r1c.CONTRACT_PATH.read_bytes() + b"\n# tamper\n")
    with pytest.raises(r1c.ContractNotRegistered):
        r1c.check_registered(tampered, reg)


# ---------------------------------------------------------------- topic allow-list
def _forbidden_topics():
    block = re.search(r"forbidden_for_all_arms: \[(.*?)\]", TEXT).group(1)
    out = [t.strip().strip('"') for t in block.split(",")]
    return [t.replace("*", x) for t in out for x in ("position", "info_position", "velocity")] if out else []


def test_forbidden_topics_rejected_by_every_cue_constructor():
    topics = sorted(set(_forbidden_topics()) | set(r1c.REF_TOPICS) | {"/dji_osdk_ros/anything"})
    assert len(topics) >= 10
    ft = np.linspace(0, 16, 8)
    for t in topics:
        with pytest.raises(r1c.ForbiddenCueTopic):
            r1c.check_cue_topic(t)
        with pytest.raises(r1c.ForbiddenCueTopic):
            cues.speed_constraints(ft, ft, np.ones(8), cue_topic=t)
        with pytest.raises(r1c.ForbiddenCueTopic):
            cues.altitude_arm(ft, ft, np.zeros(8), np.zeros((8, 3)), np.tile(np.eye(3), (8, 1, 1)), cue_topic=t)
    with pytest.raises(r1c.ForbiddenCueTopic):  # not on the allow-list either
        r1c.check_cue_topic("/ublox_driver/receiver_nmea")
    assert r1c.check_cue_topic(r1c.PVT_TOPIC) == r1c.PVT_TOPIC and r1c.check_cue_topic(r1c.LLA_TOPIC)


def test_no_dji_topic_reaches_any_cue_path_in_code():
    """Static: R1 code names DJI topics only via r1_contract.REF_TOPICS (raw copy / scorer)."""
    for rel in ("yardstick3d/datasets/r1_cues.py", "yardstick3d/datasets/r1_selection.py",
                "scripts/site_replication_r1_stage1.py", "scripts/site_replication_r1_score.py"):
        src = (ROOT / rel).read_text(encoding="utf-8")
        code = "\n".join(line for line in src.splitlines() if not line.lstrip().startswith("#"))
        code = re.sub(r'"""(.*?)"""', "", code, flags=re.S)
        assert "/dji_osdk_ros" not in code, rel
    s1 = (ROOT / "scripts/site_replication_r1_stage1.py").read_text(encoding="utf-8")
    body = s1.split("def decode_pvt", 1)[1].split("\ndef ", 1)[0]
    assert "r1c.check_cue_topic(r1c.PVT_TOPIC)" in body and "r1c.PVT_TOPIC" in body
    assert "r1c.check_cue_topic(r1c.LLA_TOPIC)" in s1.split("def census", 1)[1].split("\ndef ", 1)[0]


# ---------------------------------------------------------------- GNSS time
def test_gnss_week_tow_to_utc_constant_and_rollover():
    assert cues.gnss_week_tow_to_utc(2288, 252818.0) == pytest.approx(1_700_000_000.0, abs=1e-6)
    assert cues.gnss_week_tow_to_utc(2288, 252818.0, leap_s=0.0) - cues.gnss_week_tow_to_utc(2288, 252818.0) == 18.0
    # week rollover: the last second of week W and the first second of week W+1 are 1 s apart
    a = cues.gnss_week_tow_to_utc([2287, 2288], [604799.5, 0.5])
    assert a[1] - a[0] == pytest.approx(1.0, abs=1e-9)
    assert cues.gnss_week_tow_to_utc(2287, 604800.0) == cues.gnss_week_tow_to_utc(2288, 0.0)
    # 10-bit-style week 0 boundary and invalid inputs (NaN, never guessed)
    assert cues.gnss_week_tow_to_utc(0, 18.0) == r1c.GPS_EPOCH_UNIX_S
    bad = cues.gnss_week_tow_to_utc([-1, 2288, 2288, np.nan], [0.0, -1.0, 604801.0, 0.0])
    assert np.all(np.isnan(bad))


def test_pvt_speed_3d_and_nonfinite():
    s = cues.pvt_speed_3d([3.0, np.nan, 1.0], [4.0, 1.0, 2.0], [0.0, 0.0, 2.0])
    assert s[0] == 5.0 and np.isnan(s[1]) and s[2] == pytest.approx(3.0)


def test_pvt_timescale_gate_pass_and_fail():
    rng = np.random.default_rng(0)
    rec = 1_700_000_000.0 + np.arange(0, 300, 0.1)
    delta = 0.41
    ok = cues.pvt_timescale_gate(rec + delta + 0.02 * rng.standard_normal(rec.size), rec, delta)
    assert ok["status"] == "PASS" and abs(ok["median"]) < 0.01
    off = cues.pvt_timescale_gate(rec + delta + 0.3, rec, delta)
    assert off["status"] == "FAIL" and "median" in off["reason"]
    spread = cues.pvt_timescale_gate(rec + delta + rng.uniform(-0.3, 0.3, rec.size), rec, delta)
    assert spread["status"] == "FAIL" and "spread" in spread["reason"]
    assert cues.pvt_timescale_gate([np.nan], [1.0], 0.0)["status"] == "FAIL"
    # a camera-delta-free comparison would FAIL: the gate uses record + delta, never a fitted offset
    assert cues.pvt_timescale_gate(rec + delta, rec, 0.0)["status"] == "FAIL"


# ---------------------------------------------------------------- speed integration
def test_speed_integration_constant_and_accelerating():
    ft = np.linspace(100.0, 116.0, 8)
    t = np.arange(99.0, 118.0, 0.1)
    b, ok = cues.speed_interval_lengths(ft, t, np.full(t.size, 5.0))
    assert ok.all() and np.allclose(b, 5.0 * np.diff(ft))
    acc = 0.7 * (t - 99.0)  # linear speed: trapezoid with interpolated endpoints is exact
    b, ok = cues.speed_interval_lengths(ft, t, acc)
    exact = 0.35 * ((ft[1:] - 99.0) ** 2 - (ft[:-1] - 99.0) ** 2)
    assert ok.all() and np.allclose(b, exact, rtol=1e-9)
    # quadratic speed: still converges (sample-rate trapezoid), < 0.1 % error
    q = 0.05 * (t - 99.0) ** 2
    b, _ = cues.speed_interval_lengths(ft, t, q)
    ex = (0.05 / 3) * ((ft[1:] - 99.0) ** 3 - (ft[:-1] - 99.0) ** 3)
    assert np.allclose(b, ex, rtol=1e-3)


def test_speed_gap_makes_interval_unusable_and_coverage_rule():
    ft = np.linspace(100.0, 116.0, 8)  # intervals of 16/7 s
    t = np.arange(99.0, 118.0, 0.1)
    v = np.full(t.size, 4.0)
    keep = ~((t > 103.0) & (t < 103.65))  # one gap of 0.7 s inside interval 1 ([102.29, 104.57])
    b, ok = cues.speed_interval_lengths(ft, t[keep], v[keep])
    assert ok.tolist() == [True, False, True, True, True, True, True] and np.isnan(b[1])
    keep2 = ~((t > 103.0) & (t < 103.45))  # a 0.5 s gap is NOT > 0.5 s -> usable
    _, ok2 = cues.speed_interval_lengths(ft, t[keep2], v[keep2])
    assert ok2.all()
    v3 = v.copy()
    v3[(t > 106.0) & (t < 106.65)] = np.nan  # non-finite samples are dropped -> they open a gap
    _, ok3 = cues.speed_interval_lengths(ft, t, v3)
    assert ok3.sum() == 6
    _, info = cues.speed_constraints(ft, t[keep], v[keep])  # 6/7 = 0.857 >= 0.8 -> valid
    assert info["valid"] and info["n_kept"] == 6 and info["n_dropped"] == 1
    both = keep & ~((t > 110.0) & (t < 110.7))
    _, info = cues.speed_constraints(ft, t[both], v[both])  # 5/7 = 0.714 -> invalid
    assert not info["valid"] and info["n_kept"] == 5
    _, ok4 = cues.speed_interval_lengths(ft, t[t < 112.0], v[t < 112.0])  # not covered at the end
    assert ok4.tolist()[-2:] == [False, False]


def test_speed_arm_grounding_recovers_known_scale():
    from yardstick3d.optimization.grounder import MetricGrounder

    t = np.arange(0.0, 40.0, 0.1)
    v = 6.0 + 3.0 * np.sin(2 * np.pi * t / 16.0)
    e = np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * 0.1)])
    p = np.stack([e, 2.0 * np.sin(t / 15.0), np.zeros_like(t)], axis=1)
    vn = np.gradient(p[:, 1], t)
    ft = 10.0 + np.linspace(0.0, 16.0, 8)
    c = np.stack([np.interp(ft, t, p[:, k]) for k in range(3)], axis=1)
    pred = _bundle(c * 0.4, ft)
    cs, info = cues.speed_constraints(ft, t, cues.pvt_speed_3d(vn, v, np.zeros_like(t)))
    assert info["valid"] and len(cs) == 7
    for solver in ("naive", "ls", "robust"):
        s = MetricGrounder().solve(pred, cs, solver=solver).scale
        assert s == pytest.approx(2.5, rel=0.01), solver


# ---------------------------------------------------------------- altitude arm
def _bundle(centers, ts, R=None):
    from yardstick3d.types import PredictionBundle

    c = np.asarray(centers, float)
    R = np.tile(np.diag([1.0, -1.0, -1.0]), (len(c), 1, 1)) if R is None else np.asarray(R, float)
    T = np.zeros((len(c), 3, 4))
    T[:, :, :3] = R
    T[:, :, 3] = -np.einsum("nij,nj->ni", R, c)
    return PredictionBundle(timestamps=np.asarray(ts, float), T_w2c=T)


def _rot(axis, ang):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def _tilted_nadir_R_w2c(Q, tilts_deg, azim_deg):
    """World->camera rotations of a downward camera tilted by tilt_k toward azimuth_k, in a world rotated by Q."""
    out = []
    nadir = np.diag([1.0, -1.0, -1.0])  # camera->ENU: optical axis = -Up
    for th, az in zip(np.radians(tilts_deg), np.radians(azim_deg)):
        tilt = _rot([-np.sin(az), np.cos(az), 0.0], th)
        out.append((Q @ tilt @ nadir).T)
    return np.asarray(out)


def test_up_axis_from_tilted_nadir_cameras():
    Q = _rot([0.3, -0.5, 0.8], 1.1)
    rng = np.random.default_rng(3)
    tilts = rng.uniform(5.0, 10.0, 8)
    R = _tilted_nadir_R_w2c(Q, tilts, np.arange(8) * 45.0)
    e_up = cues.up_axis_from_nadir(R)
    err = np.degrees(np.arccos(np.clip(e_up @ (Q @ [0, 0, 1.0]), -1, 1)))
    assert err < 2.0  # tilts spread in azimuth largely cancel
    R1 = _tilted_nadir_R_w2c(Q, np.full(8, 8.0), np.zeros(8))  # constant one-sided 8 deg tilt: bias == tilt
    e1 = cues.up_axis_from_nadir(R1)
    assert np.degrees(np.arccos(e1 @ (Q @ [0, 0, 1.0]))) == pytest.approx(8.0, abs=1e-6)
    T = np.concatenate([R1, np.zeros((8, 3, 1))], axis=2)  # (N,3,4) poses accepted
    assert np.allclose(cues.up_axis_from_nadir(T), e1)


@pytest.mark.parametrize("sc", [0.37, 2.2])
def test_altitude_arm_recovers_scale_on_tilted_climb(sc):
    """5-10 deg tilted nadir cameras, arbitrary world rotation + scale: s_alt within 3 % of 1/sc."""
    Q = _rot([0.2, 0.9, -0.4], 2.0)
    rng = np.random.default_rng(1)
    ft = 1000.0 + np.linspace(0.0, 16.0, 8)
    p = np.stack([0.8 * (ft - ft[0]), 0.3 * (ft - ft[0]), 1.5 * (ft - ft[0])], axis=1)  # mostly vertical climb
    R = _tilted_nadir_R_w2c(Q, rng.uniform(5.0, 10.0, 8), np.arange(8) * 45.0 + 10.0)
    C = sc * (Q @ p.T).T
    t = np.arange(990.0, 1030.0, 0.2)
    h = 40.0 + 1.5 * np.clip(t - ft[0], 0.0, None)
    out = cues.altitude_arm(ft, t, h, C, R)
    assert out["status"] == r1c.ALT_OBSERVABLE and out["dh"] == pytest.approx(24.0)
    assert out["s_alt"] == pytest.approx(1.0 / sc, rel=0.03)
    # pure vertical motion, constant one-sided 8 deg tilt: bias 1/cos(8 deg) - 1 = 0.98 %
    pv = np.stack([0 * ft, 0 * ft, 1.5 * (ft - ft[0])], axis=1)
    out2 = cues.altitude_arm(ft, t, h, sc * (Q @ pv.T).T, _tilted_nadir_R_w2c(Q, np.full(8, 8.0), np.zeros(8)))
    assert out2["s_alt"] == pytest.approx(1.0 / sc, rel=0.015)


def test_altitude_observability_rule_level_flight_ratio_failure_and_gap():
    ft = 500.0 + np.linspace(0.0, 16.0, 8)
    t = np.arange(490.0, 530.0, 0.2)
    level = np.stack([6.0 * (ft - ft[0]), 0 * ft, 0 * ft], axis=1)
    out = cues.altitude_arm(ft, t, np.full(t.size, 80.0) + 0.3 * np.sin(t), level, _nadir(8))
    assert out["status"] == r1c.ALT_NOT_OBSERVABLE and "|dh|" in out["reason"] and out["s_alt"] is None
    # |dh| = 6 m but |dv|/chord small (96 m horizontal): NOT_OBSERVABLE by the ratio rule
    shallow = np.stack([6.0 * (ft - ft[0]), 0 * ft, 0.375 * (ft - ft[0])], axis=1)
    out = cues.altitude_arm(ft, t, 80.0 + 0.375 * (t - ft[0]), shallow, _nadir(8))
    assert out["status"] == r1c.ALT_NOT_OBSERVABLE and "dv|/chord" in out["reason"]
    # camera looking UP (wrong vertical) on a climb: s_alt < 0 -> FAILURE (counted, no substitution)
    climb = np.stack([1.0 * (ft - ft[0]), 0 * ft, 1.0 * (ft - ft[0])], axis=1)
    up_R = np.tile(np.eye(3), (8, 1, 1))  # optical axis = +Up
    out = cues.altitude_arm(ft, t, 80.0 + 1.0 * (t - ft[0]), climb, up_R)
    assert out["status"] == r1c.ALT_FAILURE and out["s_alt"] < 0
    # altitude gap > 0.5 s around the first frame: no interpolation -> NOT_OBSERVABLE
    keep = ~((t > ft[0] - 0.4) & (t < ft[0] + 0.4))
    out = cues.altitude_arm(ft, t[keep], (80.0 + 1.0 * (t - ft[0]))[keep], climb, _nadir(8))
    assert out["status"] == r1c.ALT_NOT_OBSERVABLE and "gap" in out["reason"]
    out = cues.altitude_arm(ft, t, 80.0 + 1.0 * (t - ft[0]), climb, _nadir(8))
    assert out["status"] == r1c.ALT_OBSERVABLE and out["s_alt"] == pytest.approx(1.0)


def _nadir(n):
    return np.tile(np.diag([1.0, -1.0, -1.0]), (n, 1, 1))


def test_vertical_dh_window():
    t = np.arange(0.0, 100.0, 0.1)
    h = 50.0 + 1.0 * t
    assert cues.vertical_dh_window(t, h, 20.0) == pytest.approx(15.0, abs=1e-6)
    assert cues.vertical_dh_window(t, -h, 20.0) == pytest.approx(-15.0, abs=1e-6)
    assert np.isnan(cues.vertical_dh_window(t, h, 200.0))


# ---------------------------------------------------------------- freeze pins
def _repo_module_file(name: str):
    parts = name.split(".")
    for cand in (ROOT.joinpath(*parts).with_suffix(".py"), ROOT.joinpath(*parts) / "__init__.py"):
        if cand.is_file():
            return cand
    return None


def _static_repo_deps(path: Path) -> set:
    """Repo files a module imports (import / from-import) or loads via importlib (string literal '<rel>.py')."""
    import ast
    import re

    out = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and re.fullmatch(r"[\w/.-]+\.py", node.value):
            if (ROOT / node.value).is_file():
                out.add(ROOT / node.value)
        for n in names:
            f = _repo_module_file(n)
            if f is not None:
                out.add(f)
            parts = n.split(".")  # parent packages' __init__ run on import
            for i in range(1, len(parts)):
                init = ROOT.joinpath(*parts[:i]) / "__init__.py"
                if init.is_file():
                    out.add(init)
    return out


def test_freeze_pins_cover_every_repo_file_the_r1_code_loads():
    pins = r1c.freeze_pins()
    seeds = [*ROOT.glob("scripts/*site_replication_r1*.py"), *ROOT.glob("yardstick3d/datasets/r1_*.py")]
    assert len(seeds) == 8
    seen, todo = set(), list(seeds)
    while todo:  # transitive closure over repo imports + importlib-loaded scripts
        p = todo.pop()
        if p not in seen:
            seen.add(p)
            todo.extend(_static_repo_deps(p) - seen)
    closure = {p.relative_to(ROOT).as_posix() for p in seen}
    assert not closure - set(pins), sorted(closure - set(pins))
    must = {"configs/prospective_site_replication_r1.yaml", "yardstick3d/optimization/grounder.py",
            "scripts/cross_backbone_v5_stage1.py", "scripts/cross_backbone_v2_score.py",
            "scripts/cross_backbone_v2_stage1.py", "scripts/cross_backbone_v2_da3.py", "scripts/build_vggt_pack_v2.py",
            "scripts/claim1_gate_metrics_v2.py", "yardstick3d/datasets/mars_lvig.py",
            "yardstick3d/datasets/timescale.py", "configs/prospective_cross_backbone_v5.yaml",
            "vggt_runner/verify.py", "vggt_runner/model_lock.json",
            *(p.relative_to(ROOT).as_posix() for p in ROOT.glob("yardstick3d/datasets/v2_*.py")),
            *(p.relative_to(ROOT).as_posix() for p in seeds)}
    assert must <= set(pins), sorted(must - set(pins))
    assert pins["configs/prospective_site_replication_r1.yaml"] == r1c.contract_sha()
    assert all(len(v) == 64 for v in pins.values()) and not any(k.startswith(("data/", "artifacts/cross_backbone_v5"))
                                                                for k in pins)
    assert r1c.verify_freeze_pins(pins) == []
    tampered = {**pins, "yardstick3d/optimization/grounder.py": "0" * 64}
    del tampered["scripts/cross_backbone_v2_score.py"]
    assert r1c.verify_freeze_pins(tampered) == ["scripts/cross_backbone_v2_score.py",
                                                "yardstick3d/optimization/grounder.py"]
