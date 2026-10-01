"""Site replication R1 - read-only view of the frozen v2/v5 scoring helpers with R1 identity.

The frozen v5 scripts (scripts/cross_backbone_v2_score.py, scripts/cross_backbone_v2_da3.py) read their
experiment identity (pack id, pool, cruise-speed table, contract check) through a module-level name `v2`.
R1 loads a PRIVATE instance of those scripts via importlib and points that instance's `v2` at this view.
Nothing in yardstick3d.datasets.v2_scoring (or any frozen file) is modified: every other attribute is
delegated unchanged, so thresholds, REF-VALID rule, chords, rho and SubsetPred are the frozen ones.
"""
from __future__ import annotations

import types
from pathlib import Path

from yardstick3d.datasets import r1_contract as r1c
from yardstick3d.datasets import v2_scoring as v2


class V2ViewR1(types.ModuleType):
    PACK_ID = r1c.PACK_ID
    POOL = r1c.POOL
    SEQUENCE = r1c.POOL[0]
    V_CRUISE_TABLE_BY_SEQ = dict(r1c.V_CRUISE_TABLE_BY_SEQ)
    SECTION_OF_ROLE = dict(r1c.SECTION_OF_ROLE)
    CONTRACT_PATH = r1c.CONTRACT_PATH
    REGISTRY_PATH = r1c.REGISTRY_PATH

    def __init__(self, contract_path: Path = r1c.CONTRACT_PATH, registry: Path | None = r1c.REGISTRY_PATH):
        super().__init__("v2_scoring_view_r1")
        self.CONTRACT_PATH = Path(contract_path)
        self.REGISTRY_PATH = registry
        self.CONTRACT_SHA = r1c.contract_sha(contract_path)

    def __getattr__(self, name):  # only reached for names not overridden above
        return getattr(v2, name)

    @staticmethod
    def seq_of(w: dict) -> str:
        return w["sequence"]  # every R1 window carries its sequence (pooled layout)

    @staticmethod
    def v_cruise_for(w: dict) -> float:
        return r1c.V_CRUISE_TABLE_BY_SEQ[w["sequence"]]

    def check_contract(self, path=None, expected=None, registry="__default__"):
        """R1 freeze check: the contract bytes must be registered (and equal `expected` if given)."""
        reg = self.REGISTRY_PATH if registry == "__default__" else registry
        sha = r1c.check_registered(path or self.CONTRACT_PATH, reg) if reg is not None else r1c.contract_sha(path or self.CONTRACT_PATH)
        if expected is not None and sha != expected:
            raise ValueError(f"R1 contract hash mismatch: {sha} != {expected}")
        return sha
