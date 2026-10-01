from __future__ import annotations

from dataclasses import dataclass, field

from yardstick3d.types import ConstraintValidity, MetricConstraint, MetricState, PredictionBundle


@dataclass
class ConstraintSet:
    constraints: list[MetricConstraint] = field(default_factory=list)

    def add(self, constraint: MetricConstraint) -> None:
        self.constraints.append(constraint)

    def __iter__(self):
        return iter(self.constraints)

    def __len__(self) -> int:
        return len(self.constraints)

    def valid(self, prediction: PredictionBundle, state: MetricState) -> list[MetricConstraint]:
        out: list[MetricConstraint] = []
        for c in self.constraints:
            if c.validity(prediction, state) == ConstraintValidity.VALID:
                out.append(c)
        return out
