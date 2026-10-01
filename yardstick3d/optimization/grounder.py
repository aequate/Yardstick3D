from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from yardstick3d.constraints.base import ConstraintSet
from yardstick3d.geometry.cameras import camera_centers_from_w2c
from yardstick3d.geometry.sim3 import Sim3, apply_sim3_points
from yardstick3d.observability.metrics import metric_observability_score
from yardstick3d.optimization.scale_solver import naive_scale_ratio, solve_global_scale
from yardstick3d.types import GroundingResult, MetricState, PredictionBundle, SolverDiagnostics


SolverName = Literal["none", "naive", "ls", "robust"]


@dataclass
class MetricGrounder:
    kernel: str = "huber"
    huber_delta: float = 1.345
    min_observability: float = 0.45

    def solve(
        self,
        prediction: PredictionBundle,
        constraints: ConstraintSet,
        solver: SolverName = "robust",
        provenance: dict[str, Any] | None = None,
    ) -> GroundingResult:
        if solver == "none":
            state = MetricState(scale=1.0)
            diag = SolverDiagnostics(converged=True, message="identity")
            scale_var = float("inf")
        elif solver == "naive":
            from yardstick3d.optimization.scale_solver import collect_scale_observations

            obs = collect_scale_observations(prediction, constraints)
            if not obs:
                state = MetricState(scale=float("nan"))
                diag = SolverDiagnostics(message="insufficient_observations")
                scale_var = float("inf")
            else:
                a = np.array([o[1] for o in obs])
                b = np.array([o[2] for o in obs])
                s = naive_scale_ratio(a, b)
                state = MetricState(scale=float(s) if np.isfinite(s) else float("nan"))
                diag = SolverDiagnostics(
                    converged=np.isfinite(s),
                    n_accepted=len(obs),
                    message="naive_ratio",
                )
                scale_var = float("nan")
        else:
            kernel = "l2" if solver == "ls" else self.kernel
            res = solve_global_scale(
                prediction,
                constraints,
                kernel=kernel,
                huber_delta=self.huber_delta,
            )
            state = MetricState(scale=res.scale)
            diag = res.diagnostics
            scale_var = res.scale_var

        obs_rep = metric_observability_score(
            prediction,
            constraints,
            fisher_s=None if not np.isfinite(diag.hessian_scale) else diag.hessian_scale,
        )
        sim = Sim3(scale=state.scale if np.isfinite(state.scale) else 1.0, R=state.rotation, t=state.translation)
        centers = camera_centers_from_w2c(prediction.T_w2c)
        traj = apply_sim3_points(centers, sim) if np.isfinite(state.scale) else centers * np.nan
        depth = None
        if prediction.depth_z is not None and np.isfinite(state.scale):
            depth = prediction.depth_z * state.scale * state.depth_scale
        points = None
        if prediction.points_world is not None and np.isfinite(state.scale):
            points = apply_sim3_points(prediction.points_world, sim)

        return GroundingResult(
            state=state,
            trajectory_metric=traj,
            depth_metric=depth,
            points_metric=points,
            scale=float(state.scale),
            scale_std=float(np.sqrt(scale_var)) if np.isfinite(scale_var) else float("nan"),
            observability_score=obs_rep.score,
            observable=obs_rep.observable and np.isfinite(state.scale),
            diagnostics=diag,
            accepted_constraints=diag.n_accepted,
            rejected_constraints=diag.n_rejected,
            residual_breakdown=diag.residual_breakdown,
            provenance=provenance or {},
        )
