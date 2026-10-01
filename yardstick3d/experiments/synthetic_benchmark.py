from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from yardstick3d.datasets.synthetic import generate_scene, make_constraints
from yardstick3d.evaluation.scale import relative_scale_error, scale_error
from yardstick3d.optimization.grounder import MetricGrounder
from yardstick3d.visualization.plots import plot_information_budget, plot_sweep, save_table


SOLVERS = ("none", "naive", "ls", "robust")


def _finite(x: float) -> float | None:
    return float(x) if np.isfinite(x) else None


def _run_once(
    regime: str,
    cue: str,
    sparsity: float,
    noise: float,
    outliers: float,
    time_offset: float,
    solver: str,
    seed: int,
    true_scale: float = 2.7,
    n_frames: int = 24,
) -> dict:
    scene = generate_scene(
        n_frames=n_frames,
        dt=0.1,
        regime=regime,
        true_scale=true_scale,
        seed=seed,
        unknown_rotation=(cue != "altitude"),
    )
    cs = make_constraints(
        scene,
        cue=cue,
        sparsity=sparsity,
        noise_std=noise,
        outlier_frac=outliers,
        time_offset_s=time_offset,
        seed=seed + 17,
    )
    result = MetricGrounder().solve(
        scene.prediction,
        cs,
        solver=solver,  # type: ignore[arg-type]
        provenance={
            "regime": regime,
            "cue": cue,
            "sparsity": sparsity,
            "noise": noise,
            "outliers": outliers,
            "time_offset": time_offset,
            "solver": solver,
            "seed": seed,
        },
    )
    return {
        "regime": regime,
        "cue": cue,
        "sparsity": sparsity,
        "noise": noise,
        "outliers": outliers,
        "time_offset": time_offset,
        "solver": solver,
        "n_constraints": len(cs),
        "scale_hat": float(result.scale) if np.isfinite(result.scale) else None,
        "true_scale": float(scene.true_scale),
        "scale_error_log": _finite(scale_error(result.scale, scene.true_scale)),
        "scale_error_rel": _finite(relative_scale_error(result.scale, scene.true_scale)),
        "scale_std": float(result.scale_std) if np.isfinite(result.scale_std) else None,
        "observability_score": float(result.observability_score),
        "observable": bool(result.observable),
        "accepted": int(result.accepted_constraints),
        "rejected": int(result.rejected_constraints),
        "converged": bool(result.diagnostics.converged),
        "message": result.diagnostics.message,
        "budget": float(len(cs) / max(scene.prediction.n_frames(), 1)),
    }


def run_synthetic_benchmark(out_dir: Path, seed: int = 0) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    # Exact recovery
    for cue in ("speed", "gnss", "altitude", "baseline", "range"):
        regime = "vertical" if cue == "altitude" else "line"
        rows.append(_run_once(regime, cue, 1.0, 0.0, 0.0, 0.0, "robust", seed))

    # Solver comparison, noisy speed
    for solver in SOLVERS:
        rows.append(_run_once("line", "speed", 1.0, 0.15, 0.0, 0.0, solver, seed))

    # Sparsity
    for sp in (1.0, 0.5, 0.2, 0.1, 0.05):
        rows.append(_run_once("line", "speed", sp, 0.0, 0.0, 0.0, "robust", seed, n_frames=40))

    # Noise
    for sig in (0.0, 0.05, 0.1, 0.25, 0.5):
        rows.append(_run_once("line", "speed", 1.0, sig, 0.0, 0.0, "robust", seed))
        rows.append(_run_once("line", "speed", 1.0, sig, 0.0, 0.0, "naive", seed))

    # Outliers
    for of in (0.0, 0.05, 0.1, 0.2):
        rows.append(_run_once("line", "speed", 1.0, 0.05, of, 0.0, "robust", seed))
        rows.append(_run_once("line", "speed", 1.0, 0.05, of, 0.0, "ls", seed))
        rows.append(_run_once("line", "speed", 1.0, 0.05, of, 0.0, "naive", seed))

    # Heterogeneous
    rows.append(_run_once("dynamic", "all", 0.3, 0.1, 0.05, 0.0, "robust", seed))
    rows.append(_run_once("dynamic", "speed", 0.3, 0.1, 0.05, 0.0, "robust", seed))

    # Degeneracy
    for regime in ("stationary", "pure_rotation", "line", "circle", "planar_drive", "vertical"):
        rows.append(_run_once(regime, "speed", 1.0, 0.0, 0.0, 0.0, "robust", seed))

    # Time offset
    for off in (0.0, 0.01, 0.05, 0.1, 0.25, 0.5):
        rows.append(_run_once("line", "speed", 1.0, 0.0, 0.0, off, "robust", seed))

    save_table(rows, out_dir / "synthetic_results.json")

    # Plots
    sp_vals = [1.0, 0.5, 0.2, 0.1, 0.05]
    y_sp = [
        next(
            r["scale_error_log"]
            for r in rows
            if r["sparsity"] == sp
            and r["cue"] == "speed"
            and r["noise"] == 0.0
            and r["outliers"] == 0.0
            and r["solver"] == "robust"
            and r["time_offset"] == 0.0
            and r["regime"] == "line"
        )
        for sp in sp_vals
    ]
    plot_sweep(
        sp_vals,
        {"robust speed": y_sp},
        xlabel="cue sparsity (fraction of intervals kept)",
        ylabel=r"$E_s$",
        title="Scale error vs sparsity (synthetic, noiseless)",
        out_path=out_dir / "sparsity.png",
    )

    sigs = [0.0, 0.05, 0.1, 0.25, 0.5]
    y_noise = {}
    for solver in ("robust", "naive"):
        y_noise[solver] = [
            next(
                r["scale_error_log"]
                for r in rows
                if r["noise"] == sig
                and r["solver"] == solver
                and r["cue"] == "speed"
                and r["outliers"] == 0.0
                and r["sparsity"] == 1.0
                and r["regime"] == "line"
            )
            for sig in sigs
        ]
    plot_sweep(
        sigs,
        y_noise,
        xlabel=r"speed noise $\sigma$ (m/s)",
        ylabel=r"$E_s$",
        title="Scale error vs speed noise",
        out_path=out_dir / "noise.png",
    )

    ofs = [0.0, 0.05, 0.1, 0.2]
    y_out = {}
    for solver in ("robust", "ls", "naive"):
        y_out[solver] = [
            next(
                r["scale_error_log"]
                for r in rows
                if r["outliers"] == of
                and r["solver"] == solver
                and r["cue"] == "speed"
                and abs(r["noise"] - 0.05) < 1e-12
                and r["regime"] == "line"
            )
            for of in ofs
        ]
    plot_sweep(
        ofs,
        y_out,
        xlabel="outlier fraction",
        ylabel=r"$E_s$",
        title="Scale error vs outliers",
        out_path=out_dir / "outliers.png",
    )

    budgets = []
    errs = []
    for r in rows:
        if r["solver"] == "robust" and r["cue"] == "speed" and r["regime"] == "line" and r["outliers"] == 0.0:
            if np.isfinite(r["scale_error_log"]):
                budgets.append(r["budget"])
                errs.append(r["scale_error_log"])
    if budgets:
        order = np.argsort(budgets)
        plot_information_budget(
            list(np.array(budgets)[order]),
            {"speed/robust": list(np.array(errs)[order])},
            out_dir / "information_budget.png",
        )

    exact = [r for r in rows if r["noise"] == 0 and r["outliers"] == 0 and r["sparsity"] == 1.0 and r["time_offset"] == 0.0 and r["solver"] == "robust" and r["regime"] in ("line", "vertical")]
    exact_ok = all(np.isfinite(r["scale_error_rel"]) and r["scale_error_rel"] < 1e-6 for r in exact)

    summary = {
        "n_rows": len(rows),
        "exact_recovery_pass": exact_ok,
        "exact_cases": [
            {"cue": r["cue"], "rel": r["scale_error_rel"], "hat": r["scale_hat"]} for r in exact
        ],
        "degeneracy": [
            {
                "regime": r["regime"],
                "observable": r["observable"],
                "score": r["observability_score"],
                "rel": r["scale_error_rel"],
            }
            for r in rows
            if r["solver"] == "robust"
            and r["cue"] == "speed"
            and r["noise"] == 0
            and r["outliers"] == 0
            and r["sparsity"] == 1.0
            and r["time_offset"] == 0
            and r["regime"] in ("stationary", "pure_rotation", "line")
        ],
        "out_dir": str(out_dir),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
