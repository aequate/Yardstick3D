from __future__ import annotations


from yardstick3d.datasets.synthetic import generate_scene, make_constraints
from yardstick3d.evaluation.scale import relative_scale_error
from yardstick3d.optimization.grounder import MetricGrounder


def test_exact_scale_recovery_speed():
    scene = generate_scene(n_frames=20, regime="line", true_scale=2.7, seed=0)
    cs = make_constraints(scene, "speed", sparsity=1.0, noise_std=0.0, seed=0)
    r = MetricGrounder().solve(scene.prediction, cs, solver="ls")
    assert relative_scale_error(r.scale, scene.true_scale) < 1e-8


def test_exact_scale_recovery_each_cue():
    for cue, regime in [
        ("speed", "line"),
        ("gnss", "line"),
        ("altitude", "vertical"),
        ("baseline", "line"),
        ("range", "line"),
    ]:
        scene = generate_scene(
            n_frames=16,
            regime=regime,
            true_scale=1.9,
            seed=5,
            unknown_rotation=(cue != "altitude"),
        )
        cs = make_constraints(scene, cue, sparsity=1.0, noise_std=0.0, seed=5)
        r = MetricGrounder().solve(scene.prediction, cs, solver="robust")
        err = relative_scale_error(r.scale, scene.true_scale)
        assert err < 1e-6, f"{cue} err={err} hat={r.scale}"


def test_naive_vs_ls_noiseless_match():
    scene = generate_scene(n_frames=12, regime="line", true_scale=3.3, seed=2)
    cs = make_constraints(scene, "speed", sparsity=1.0, seed=2)
    g = MetricGrounder()
    a = g.solve(scene.prediction, cs, solver="naive")
    b = g.solve(scene.prediction, cs, solver="ls")
    assert abs(a.scale - b.scale) / b.scale < 1e-6


def test_heterogeneous_fusion_exact():
    scene = generate_scene(n_frames=20, regime="dynamic", true_scale=2.1, seed=8)
    cs = make_constraints(scene, "all", sparsity=1.0, noise_std=0.0, seed=8)
    r = MetricGrounder().solve(scene.prediction, cs, solver="robust")
    assert relative_scale_error(r.scale, scene.true_scale) < 1e-5


def test_outlier_frac_actually_corrupts_measurements():
    scene = generate_scene(n_frames=24, regime="line", true_scale=2.7, seed=0)
    cs0 = make_constraints(scene, "speed", sparsity=1.0, noise_std=0.05, outlier_frac=0.0, seed=17)
    cs2 = make_constraints(scene, "speed", sparsity=1.0, noise_std=0.05, outlier_frac=0.2, seed=17)
    v0 = [c.speed_mps for c in cs0]
    v2 = [c.speed_mps for c in cs2]
    assert max(abs(a - b) for a, b in zip(v0, v2)) > 1.0


def test_robust_beats_ls_with_outliers():
    scene = generate_scene(n_frames=40, regime="line", true_scale=2.4, seed=9)
    cs = make_constraints(scene, "speed", sparsity=1.0, noise_std=0.02, outlier_frac=0.2, seed=9)
    g = MetricGrounder()
    ls = g.solve(scene.prediction, cs, solver="ls")
    rb = g.solve(scene.prediction, cs, solver="robust")
    e_ls = relative_scale_error(ls.scale, scene.true_scale)
    e_rb = relative_scale_error(rb.scale, scene.true_scale)
    assert e_rb < e_ls or e_rb < 0.05
    assert e_rb < 0.15
