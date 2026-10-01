from __future__ import annotations

import json

from yardstick3d.datasets.synthetic import generate_scene, make_constraints
from yardstick3d.evaluation.leakage import assert_no_gt_in_constraints
from yardstick3d.optimization.grounder import MetricGrounder


def test_result_json_schema():
    scene = generate_scene(n_frames=8, regime="line", true_scale=2.0, seed=0)
    cs = make_constraints(scene, "speed", seed=0)
    assert_no_gt_in_constraints(cs)
    r = MetricGrounder().solve(scene.prediction, cs, solver="robust")
    d = r.to_json_dict()
    assert "scale" in d
    assert "observability_score" in d
    json.dumps(d)


def test_registry():
    from yardstick3d.constraints.registry import CONSTRAINT_REGISTRY, create_constraint

    c = create_constraint("speed", t_i=0.0, t_j=0.1, speed_mps=1.0)
    assert c.name == "speed"
    assert "gnss_displacement" in CONSTRAINT_REGISTRY
