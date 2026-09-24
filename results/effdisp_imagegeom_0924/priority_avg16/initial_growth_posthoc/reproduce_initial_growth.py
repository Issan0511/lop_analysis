#!/usr/bin/env python3
"""Post hoc actual-initialization growth of effective first-layer width.

Reads only the saved priority avg16 versus original rgb32 analysis tables.
R_0 is the actual t000 width; R_t for t>0 is the POST-task endpoint R_next.
No training or registered 50-series verdict is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/issan/Projects/obsidian-research-data/effdisp_imagegeom_0924/priority_avg16/analysis")
SOURCE_HASH = "273c3fa3b52dc0c954eaaa58a4e711183c259770"
SEEDS = tuple(range(300, 305))
GEOMETRIES = ("rgb32", "avg16")
OPTIMIZERS = ("adam", "sgd")
LATE = tuple(range(41, 51))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stats(values) -> dict:
    x = np.asarray(values, dtype=np.float64)
    if len(x) != 5 or not np.isfinite(x).all():
        raise ValueError("five finite matched seeds are required")
    return {"n_seed": 5, "median": float(np.median(x)), "min": float(np.min(x)),
            "max": float(np.max(x))}


def check_close(a: float, b: float, label: str) -> float:
    error = abs(a-b)
    if not math.isclose(a, b, rel_tol=2e-12, abs_tol=2e-10):
        raise ValueError(f"{label}: snapshot width mismatch {a} vs {b}")
    return error


def calculate(input_dir: Path):
    source = json.loads((input_dir / "metadata.json").read_text())
    if source.get("status") != "COMPLETE_PRIORITY_TWO_CONDITION" or source.get("source_git_hash") != SOURCE_HASH:
        raise ValueError("input is not completed frozen-source priority analysis")
    transitions = pd.read_csv(input_dir / "transitions.csv")
    points = pd.read_csv(input_dir / "checkpoint_points.csv")
    expected = {(g, o, s) for g in GEOMETRIES for o in OPTIMIZERS for s in SEEDS}
    if (len(transitions) != 1000 or len(points) != 120
            or transitions.duplicated(["geometry", "optimizer", "seed", "task"]).any()
            or points.duplicated(["geometry", "optimizer", "seed", "task"]).any()
            or set(zip(transitions.geometry, transitions.optimizer, transitions.seed)) != expected
            or set(zip(points.geometry, points.optimizer, points.seed)) != expected):
        raise ValueError("priority tables do not contain exactly 20 complete series")
    late_rows, series_rows = [], []
    max_checkpoint_error = 0.0
    max_adjacent_error = 0.0
    for geometry in GEOMETRIES:
        for optimizer in OPTIMIZERS:
            for seed in SEEDS:
                block = transitions[(transitions.geometry == geometry)
                                    & (transitions.optimizer == optimizer)
                                    & (transitions.seed == seed)].sort_values("task")
                cp = points[(points.geometry == geometry) & (points.optimizer == optimizer)
                            & (points.seed == seed)].set_index("task")
                if (len(block) != 50 or list(block.task) != list(range(1, 51))
                        or set(cp.index) != {0, 10, 20, 30, 40, 50}):
                    raise ValueError(f"task schedule incomplete: {geometry}/{optimizer}/{seed}")
                t0, t50 = block.iloc[0], block.iloc[-1]
                R0, R50 = float(t0.R), float(t50.R_next)
                if not (np.isfinite(R0) and np.isfinite(R50) and R0 > 0 and R50 > 0):
                    raise ValueError("nonpositive or nonfinite endpoint effective width")
                max_checkpoint_error = max(max_checkpoint_error,
                    check_close(R0, float(cp.loc[0, "R_endpoint"]), "t0"),
                    check_close(R50, float(cp.loc[50, "R_endpoint"]), "t50"))
                prev_post = block.R_next.to_numpy(dtype=np.float64)[:-1]
                next_pre = block.R.to_numpy(dtype=np.float64)[1:]
                max_adjacent_error = max(max_adjacent_error, float(np.max(np.abs(prev_post-next_pre))))
                if not np.allclose(prev_post, next_pre, rtol=2e-12, atol=2e-10):
                    raise ValueError("adjacent pre/post effective widths disagree")
                late = block[block.task.isin(LATE)]
                if len(late) != 10:
                    raise ValueError("late POST-task endpoints incomplete")
                for row in late.itertuples(index=False):
                    width = float(row.R_next)
                    if not np.isfinite(width) or width <= 0:
                        raise ValueError("invalid late POST-task width")
                    late_rows.append({"geometry": geometry, "optimizer": optimizer, "seed": seed,
                                      "task": int(row.task), "R0_actual": R0,
                                      "R_posttask": width, "R_posttask_over_R0": width/R0,
                                      "source_snapshot": row.source_next})
                late_width_median = float(np.median(late.R_next.to_numpy(dtype=np.float64)))
                late_growth_median = float(np.median(late.R_next.to_numpy(dtype=np.float64)/R0))
                series_rows.append({"geometry": geometry, "optimizer": optimizer, "seed": seed,
                                    "R0_actual": R0, "R50_posttask": R50,
                                    "R50_over_R0": R50/R0,
                                    "R_posttask_41_50_median": late_width_median,
                                    "R_posttask_over_R0_41_50_median": late_growth_median,
                                    "source_t000": t0.source_prev,
                                    "source_t050": t50.source_next})
    series = pd.DataFrame(series_rows)
    paired_rows = []
    metrics = ("R50_over_R0", "R_posttask_over_R0_41_50_median")
    for optimizer in OPTIMIZERS:
        for seed in SEEDS:
            avg = series[(series.geometry == "avg16") & (series.optimizer == optimizer)
                         & (series.seed == seed)].iloc[0]
            rgb = series[(series.geometry == "rgb32") & (series.optimizer == optimizer)
                         & (series.seed == seed)].iloc[0]
            for metric in metrics:
                av, rv = float(avg[metric]), float(rgb[metric])
                paired_rows.append({"optimizer": optimizer, "seed": seed, "metric": metric,
                                    "avg16_growth": av, "rgb32_growth": rv,
                                    "paired_growth_ratio_avg16_over_rgb32": av/rv})
    paired = pd.DataFrame(paired_rows)
    absolute_groups = []
    absolute_metrics = ("R0_actual", "R50_posttask", "R_posttask_41_50_median",
                        "R50_over_R0", "R_posttask_over_R0_41_50_median")
    for (geometry, optimizer), block in series.groupby(["geometry", "optimizer"]):
        for metric in absolute_metrics:
            absolute_groups.append({"geometry": geometry, "optimizer": optimizer,
                                    "metric": metric, **stats(block[metric])})
    paired_groups = []
    for (optimizer, metric), block in paired.groupby(["optimizer", "metric"]):
        paired_groups.append({"optimizer": optimizer, "metric": metric,
                              **stats(block.paired_growth_ratio_avg16_over_rgb32)})
    audit = {"status": "PASS", "series": 20, "transitions_read": len(transitions),
             "late_posttask_rows": len(late_rows), "checkpoint_points_read": len(points),
             "max_t0_t50_checkpoint_abs_error": max_checkpoint_error,
             "max_adjacent_pre_post_abs_error": max_adjacent_error}
    return (pd.DataFrame(late_rows), series, paired, pd.DataFrame(absolute_groups),
            pd.DataFrame(paired_groups), audit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=ROOT / "initial_growth_posthoc")
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"refusing existing nonempty output: {args.out}")
    late, series, paired, absolute, paired_groups, audit = calculate(args.input)
    args.out.mkdir(parents=True, exist_ok=True)
    outputs = {"late_posttask_widths.csv": late, "series_width_growth.csv": series,
               "paired_growth.csv": paired, "absolute_width_groups.csv": absolute,
               "paired_growth_groups.csv": paired_groups}
    for name, table in outputs.items():
        table.to_csv(args.out / name, index=False)
    definitions = {"status": "POSTHOC_DESCRIPTIVE_PRIORITY_COMPARISON",
                   "R0_actual": "effective width of saved W1 at task0, from transition1 R and t000 checkpoint",
                   "R_t_posttask": "effective width after task t; transition t R_next, not transition t R",
                   "R50_over_R0": "R after task50 divided by same series actual task0 width",
                   "late_growth": "median over POST-task endpoints41..50 of R_t/R0 within the same series",
                   "paired_growth": "avg16 growth divided by rgb32 growth in matched seed, then median/min/max over five seeds",
                   "scope": "Separately prioritized avg16 versus completed original RGB32 only; not final registered 50-series analysis or asymptotic convergence."}
    (args.out / "definitions.json").write_text(json.dumps(definitions, indent=2) + "\n")
    (args.out / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    archived_script = args.out / "reproduce_initial_growth.py"
    shutil.copy2(Path(__file__), archived_script)
    metadata = {"status": "COMPLETE_POSTHOC", "input": str(args.input),
                "source_git_hash": SOURCE_HASH, "script_path": str(archived_script),
                "script_sha256": sha256(archived_script),
                "input_sha256": {name: sha256(args.input / name) for name in
                                 ("metadata.json", "transitions.csv", "checkpoint_points.csv")},
                "output_sha256": {name: sha256(args.out / name) for name in
                                  (*outputs, "definitions.json", "audit.json")}}
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("Absolute width/growth by condition (five-seed medians):")
    print(absolute.pivot_table(index=["geometry", "optimizer"], columns="metric", values="median").to_string())
    print("\nPaired avg16/RGB32 growth ratios (five-seed median/min/max):")
    print(paired_groups.to_string(index=False))
    print("\nAudit:", json.dumps(audit))


if __name__ == "__main__":
    main()
