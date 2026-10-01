from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def plot_sweep(
    x: list[float] | np.ndarray,
    y: dict[str, list[float]],
    xlabel: str,
    ylabel: str,
    title: str,
    out_path: str | Path,
    logx: bool = False,
) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for name, vals in y.items():
        ax.plot(x, vals, marker="o", label=name)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if logx:
        ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_information_budget(
    budget: list[float],
    scale_err: dict[str, list[float]],
    out_path: str | Path,
) -> None:
    plot_sweep(
        budget,
        scale_err,
        xlabel="Metric information budget B = n_cues / n_frames",
        ylabel=r"$E_s = |\log(\hat s / s^\star)|$",
        title="How much metric information is needed? (synthetic)",
        out_path=out_path,
        logx=True,
    )


def save_table(rows: list[dict[str, Any]], out_path: str | Path) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    import json

    def _default(o: Any):
        if isinstance(o, (np.bool_, np.integer)):
            return o.item()
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))

    Path(out_path).write_text(json.dumps(rows, indent=2, default=_default), encoding="utf-8")
