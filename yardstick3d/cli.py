from __future__ import annotations

import argparse
import json
from pathlib import Path

from yardstick3d.experiments.synthetic_benchmark import run_synthetic_benchmark
from yardstick3d.optimization.grounder import MetricGrounder
from yardstick3d.datasets.synthetic import generate_scene, make_constraints


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="yardstick3d")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("synthetic-benchmark", help="Controlled scale-recovery sweeps")
    b.add_argument("--out", default="artifacts/synthetic")
    b.add_argument("--seed", type=int, default=0)

    g = sub.add_parser("ground", help="Ground a synthetic or cached prediction")
    g.add_argument("--regime", default="line")
    g.add_argument("--constraint", default="speed")
    g.add_argument("--solver", default="robust")
    g.add_argument("--true-scale", type=float, default=2.5)
    g.add_argument("--n-frames", type=int, default=30)
    g.add_argument("--sparsity", type=float, default=1.0)
    g.add_argument("--noise", type=float, default=0.0)

    sub.add_parser("info", help="Print hardware/adapter status")

    args = p.parse_args(argv)
    if args.cmd == "synthetic-benchmark":
        summary = run_synthetic_benchmark(Path(args.out), seed=args.seed)
        print(json.dumps(summary, indent=2))
        return 0
    if args.cmd == "ground":
        scene = generate_scene(
            n_frames=args.n_frames, regime=args.regime, true_scale=args.true_scale, seed=0
        )
        cs = make_constraints(scene, cue=args.constraint, sparsity=args.sparsity, noise_std=args.noise)
        result = MetricGrounder().solve(scene.prediction, cs, solver=args.solver)
        payload = result.to_json_dict()
        payload["true_scale"] = scene.true_scale
        print(json.dumps(payload, indent=2))
        return 0
    if args.cmd == "info":
        from yardstick3d.adapters.vggt import VGGT_BLOCKER

        print(json.dumps({"vggt": VGGT_BLOCKER, "local_backend": "dummy-synthetic"}, indent=2))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
