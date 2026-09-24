#!/usr/bin/env python3
"""Plot POST-task effective-width growth from actual task-0 initialization."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SOURCE_HASH = "273c3fa3b52dc0c954eaaa58a4e711183c259770"
GEOMETRIES = ("rgb32", "dup64", "dup64_half", "gray32", "avg16")
OPTIMIZERS = ("adam", "sgd")
SEEDS = tuple(range(300, 305))
COLORS = {"rgb32": "#242424", "dup64": "#cc5a30", "dup64_half": "#d39724",
          "gray32": "#3576a8", "avg16": "#518c61"}
DEFAULT_ANALYSIS = Path("/home/issan/Projects/claude/wt/effdisp_imagegeom_0924/results/effdisp_imagegeom_0924")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run(analysis: Path) -> dict:
    posthoc = analysis / "posthoc_normalization"
    metadata = json.loads((posthoc / "metadata.json").read_text())
    if metadata.get("status") != "COMPLETE_POSTHOC" or metadata.get("source_git_hash") != SOURCE_HASH:
        raise ValueError("growth source is not the completed frozen-source posthoc analysis")
    source = posthoc / "growth_POSTtask_per_task.csv"
    if sha256(source) != metadata["output_sha256"][source.name]:
        raise ValueError("growth CSV differs from posthoc output manifest")
    data = pd.read_csv(source)
    if (len(data) != 2500 or data.duplicated(["geometry", "optimizer", "seed", "task"]).any()
            or set(data.geometry) != set(GEOMETRIES) or set(data.optimizer) != set(OPTIMIZERS)
            or set(data.seed) != set(SEEDS) or set(data.task) != set(range(1, 51))):
        raise ValueError("growth table does not contain all 50 series and task1–50")
    if not np.isfinite(data[["R0_actual", "R_posttask", "R_posttask_over_R0"]].to_numpy(dtype=float)).all():
        raise ValueError("nonfinite effective-width growth")
    zero = data[data.task == 1][["geometry", "optimizer", "seed", "R0_actual"]].copy()
    zero["task"] = 0
    zero["R_posttask"] = zero.R0_actual
    zero["R_posttask_over_R0"] = 1.0
    complete = pd.concat([zero, data], ignore_index=True)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, layout="constrained")
    for ax, optimizer in zip(axes, OPTIMIZERS):
        selected = complete[complete.optimizer == optimizer]
        for geometry in GEOMETRIES:
            block = selected[selected.geometry == geometry]
            grouped = block.groupby("task")["R_posttask_over_R0"]
            median = grouped.median()
            low = grouped.min()
            high = grouped.max()
            if len(median) != 51 or not grouped.size().eq(5).all():
                raise ValueError(f"incomplete seed envelope: {geometry}/{optimizer}")
            x = median.index.to_numpy(dtype=int)
            color = COLORS[geometry]
            ax.fill_between(x, low.to_numpy(), high.to_numpy(), color=color, alpha=.13, linewidth=0)
            ax.plot(x, median.to_numpy(), color=color, lw=1.7, label=geometry)
        ax.set_title(optimizer.upper())
        ax.set_ylabel("POST-task $R_t/R_0$")
        ax.grid(alpha=.2)
        ax.legend(ncol=5, fontsize=8)
    axes[-1].set_xlabel("POST-task endpoint $t$ (actual initialization at $t=0$)")
    fig.suptitle("Effective first-layer width relative to each run's actual task-0 width\n"
                 "Line: five-seed median; band: seed min–max")
    target = analysis / "figures" / "initial_width_growth.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=180)
    plt.close(fig)
    record = {"status": "COMPLETE", "source_git_hash": SOURCE_HASH,
              "growth_csv": str(source), "growth_csv_sha256": sha256(source),
              "figure": str(target), "figure_sha256": sha256(target),
              "script": str(Path(__file__)), "script_sha256": sha256(Path(__file__)),
              "series": 50, "seeds_per_condition": 5,
              "definition": "POST-task R_t/R0 for t1..50; t0=1 from actual R0; line seed median, band seed min-max"}
    manifest = analysis / "initial_width_growth_figure.json"
    manifest.write_text(json.dumps(record, indent=2) + "\n")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, default=DEFAULT_ANALYSIS)
    args = parser.parse_args()
    print(json.dumps(run(args.analysis), indent=2))


if __name__ == "__main__":
    main()
