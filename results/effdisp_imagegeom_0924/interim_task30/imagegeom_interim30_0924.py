#!/usr/bin/env python3
"""Descriptive task-30 image-geometry checkpoint; not a preregistered late verdict."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/home/issan/Projects/claude/wt/effdisp_imagegeom_0924")
sys.path.insert(0, str(REPO))
from analysis.effdisp_imagegeom_0924.analyze import projection_terms  # noqa: E402

RAW = Path("/home/issan/Projects/obsidian-research-data/effdisp_imagegeom_0924/raw")
OUT = Path("/home/issan/Projects/obsidian-research-data/effdisp_imagegeom_0924/interim_task30")
SOURCE_HASH = "273c3fa3b52dc0c954eaaa58a4e711183c259770"
GEOMETRIES = ("rgb32", "dup64", "dup64_half")
OPTIMIZERS = ("adam", "sgd")
SEEDS = tuple(range(300, 305))
WINDOW = tuple(range(26, 31))
POINTS = (10, 20, 30)
N = 1200


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot(raw: Path, geometry: str, optimizer: str, seed: int, task: int):
    path = raw / geometry / optimizer / f"seed{seed}" / f"t{task:03d}.npz"
    with np.load(path, allow_pickle=False) as f:
        if int(f["task"]) != task or int(f["step"]) != task*30000:
            raise ValueError(f"misaligned snapshot: {path}")
        w, b = np.asarray(f["W1"], dtype=np.float64), np.asarray(f["b1"], dtype=np.float64)
    if not np.isfinite(w).all() or not np.isfinite(b).all():
        raise ValueError(f"nonfinite snapshot: {path}")
    return w, b, path


def effective_w(w: np.ndarray, geometry: str) -> np.ndarray:
    if geometry == "rgb32":
        if w.shape != (100, 3072):
            raise ValueError("rgb32 W1 shape")
        return w
    if w.shape != (100, 12288):
        raise ValueError(f"{geometry} W1 shape")
    scale = 0.5 if geometry == "dup64_half" else 1.0
    return scale * w.reshape(100, 3, 32, 2, 32, 2).sum(axis=(3, 5), dtype=np.float64).reshape(100, 3072)


def raw_width(w: np.ndarray) -> float:
    return float(np.linalg.norm(w))


def bank_and_audit(raw: Path, seed: int) -> np.ndarray:
    paths = {g: raw / "banks" / g / f"seed{seed}.npz" for g in GEOMETRIES}
    with np.load(paths["rgb32"], allow_pickle=False) as f:
        x = np.asarray(f["x"], dtype=np.float32)
        ids = np.asarray(f["indices"])
    if x.shape != (N, 3072) or len(np.unique(ids)) != N:
        raise ValueError(f"invalid RGB bank for seed {seed}")
    expanded = np.repeat(np.repeat(x.reshape(N, 3, 32, 32), 2, 2), 2, 3).reshape(N, 12288)
    for geometry, scale in (("dup64", 1.0), ("dup64_half", .5)):
        with np.load(paths[geometry], allow_pickle=False) as f:
            transformed = np.asarray(f["x"], dtype=np.float32)
            transformed_ids = np.asarray(f["indices"])
        if not np.array_equal(ids, transformed_ids) or not np.array_equal(transformed, expanded*scale):
            raise ValueError(f"{geometry} bank differs from exact matched transform for seed {seed}")
    return x.astype(np.float64)


def finite_median(values) -> float:
    a = np.asarray(values, dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(np.median(a)) if len(a) else math.nan


def calculate(raw: Path):
    meta = json.loads((raw / "metadata.json").read_text())
    if meta.get("git_hash") != SOURCE_HASH or meta.get("tasks") != 50 or meta.get("steps_per_task") != 30000:
        raise ValueError("source hash or canonical schedule mismatch")
    if tuple(meta.get("seeds", [])) != SEEDS:
        raise ValueError("seeds mismatch")
    if sha256(REPO / "src/effdisp_imagegeom_0924.py") != meta.get("source_sha256"):
        raise ValueError("local source differs from saved source SHA256")
    for g in GEOMETRIES:
        for o in OPTIMIZERS:
            for seed in SEEDS:
                status = json.loads((raw / g / o / f"seed{seed}" / "status.json").read_text())
                if status.get("last_saved_task", -1) < 30 or status.get("status") == "DIVERGED":
                    raise ValueError(f"task30 unavailable: {g}/{o}/{seed}")
    diag = pd.read_csv(raw / "per_task.csv")
    if diag.duplicated(["geometry", "optimizer", "seed", "task"]).any():
        raise ValueError("duplicate per_task row")
    diag = diag.set_index(["geometry", "optimizer", "seed", "task"])
    point_rows, transition_rows, bank_sha = [], [], {}
    for seed in SEEDS:
        x = bank_and_audit(raw, seed)
        mean = x.mean(0, dtype=np.float64)
        centered = x - mean
        bank_sha[str(seed)] = sha256(raw / "banks" / "rgb32" / f"seed{seed}.npz")
        for geometry in GEOMETRIES:
            for optimizer in OPTIMIZERS:
                states = {task: snapshot(raw, geometry, optimizer, seed, task)
                          for task in (*POINTS[:2], 25, *WINDOW)}
                for task in POINTS[:2]:
                    w, b, path = states[task]
                    we = effective_w(w, geometry)
                    projection = centered @ we.T / math.sqrt(N)
                    point_rows.append({"geometry": geometry, "optimizer": optimizer, "seed": seed,
                                       "task": task, "R": float(np.linalg.norm(projection)),
                                       "raw_W_F": raw_width(w), "snapshot": str(path)})
                wprev, bprev, _ = states[25]
                pprev = None
                for task in WINDOW:
                    wnext, bnext, path = states[task]
                    terms, pprev = projection_terms(
                        centered, mean, effective_w(wprev, geometry), bprev,
                        effective_w(wnext, geometry), bnext, prev_projection=pprev)
                    c, d = terms["c"], terms["d_eff"]
                    rstar = d / (2*abs(c)) if np.isfinite(c) and c < 0 and d > 0 else math.nan
                    rratio = terms["R"] / rstar if np.isfinite(rstar) and rstar > 0 else math.nan
                    key = (geometry, optimizer, seed, task)
                    if key not in diag.index:
                        raise ValueError(f"missing per_task {key}")
                    dr = diag.loc[key]
                    if dr.status != "OK" or not np.isfinite(dr.end_acc):
                        raise ValueError(f"invalid per_task {key}")
                    transition_rows.append({"geometry": geometry, "optimizer": optimizer, "seed": seed,
                                            "task": task, "R_prev": terms["R"], "R_end": terms["R_next"],
                                            "raw_W_prev_F": raw_width(wprev),
                                            "raw_W_end_F": raw_width(wnext),
                                            "D_eff": d, "c": c, "Rstar_local": rstar,
                                            "R_over_Rstar_local": rratio,
                                            "end_acc": float(dr.end_acc),
                                            "V": terms["V"], "Q": terms["Q"], "X": terms["X"],
                                            "identity_residual": terms["identity_residual"],
                                            "source_prev": str(states[task-1][2]),
                                            "source_end": str(path)})
                    if task == 30:
                        point_rows.append({"geometry": geometry, "optimizer": optimizer, "seed": seed,
                                           "task": 30, "R": terms["R_next"],
                                           "raw_W_F": raw_width(wnext), "snapshot": str(path)})
                    wprev, bprev = wnext, bnext
    points = pd.DataFrame(point_rows)
    trans = pd.DataFrame(transition_rows)
    if len(points) != 90 or len(trans) != 150:
        raise ValueError("incomplete point/transition output")
    if np.abs(trans.identity_residual).max() > 1e-8*max(1., trans.V.max()):
        raise ValueError("projection identity mismatch")
    fields = ("R_prev", "raw_W_prev_F", "D_eff", "c", "Rstar_local",
              "R_over_Rstar_local", "end_acc")
    series = []
    for (g, o, s), block in trans.groupby(["geometry", "optimizer", "seed"]):
        row = {"geometry": g, "optimizer": o, "seed": s, "window": "task26-30",
               "n_negative_c": int(np.count_nonzero(block.c < 0))}
        row.update({field: finite_median(block[field]) for field in fields})
        p = points[(points.geometry == g) & (points.optimizer == o) & (points.seed == s)]
        for task in POINTS:
            pr = p[p.task == task].iloc[0]
            row[f"R_t{task}"] = float(pr.R)
            row[f"raw_W_F_t{task}"] = float(pr.raw_W_F)
        series.append(row)
    series = pd.DataFrame(series)
    pairs = []
    comparison_fields = ("R_prev", "raw_W_prev_F", "D_eff", "Rstar_local",
                         "R_over_Rstar_local", "end_acc", "R_t10", "R_t20", "R_t30",
                         "raw_W_F_t10", "raw_W_F_t20", "raw_W_F_t30")
    for o in OPTIMIZERS:
        for seed in SEEDS:
            base = series[(series.geometry == "rgb32") & (series.optimizer == o) & (series.seed == seed)].iloc[0]
            for g in GEOMETRIES[1:]:
                arm = series[(series.geometry == g) & (series.optimizer == o) & (series.seed == seed)].iloc[0]
                for field in comparison_fields:
                    a, b = float(arm[field]), float(base[field])
                    pairs.append({"geometry": g, "reference_geometry": "rgb32", "optimizer": o,
                                  "seed": seed, "metric": field, "ratio": a/b if np.isfinite(a) and np.isfinite(b) and b > 0 else math.nan,
                                  "difference": a-b, "value": a, "reference_value": b})
                pairs.append({"geometry": g, "reference_geometry": "rgb32", "optimizer": o,
                              "seed": seed, "metric": "c", "ratio": math.nan,
                              "difference": float(arm.c-base.c), "value": float(arm.c),
                              "reference_value": float(base.c)})
    pairs = pd.DataFrame(pairs)
    groups = pd.DataFrame([{"geometry": g, "optimizer": o, "metric": field,
                            "n_seed": len(block), "n_valid_ratio": int(np.isfinite(block.ratio).sum()),
                            "paired_ratio_median": finite_median(block.ratio),
                            "paired_difference_median": finite_median(block.difference)}
                           for (g, o, field), block in pairs.groupby(["geometry", "optimizer", "metric"])])
    return points, trans, series, pairs, groups, meta, bank_sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"refusing nonempty output: {args.out}")
    points, trans, series, pairs, groups, source, bank_sha = calculate(args.raw)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, frame in (("task_points.csv", points), ("window_transitions.csv", trans),
                        ("series_summary.csv", series), ("paired_ratios.csv", pairs),
                        ("group_summary.csv", groups)):
        frame.to_csv(args.out / name, index=False)
    manifest = {"status": "DESCRIPTIVE_INTERIM_TASK30", "source_git_hash": SOURCE_HASH,
                "source_sha256": source["source_sha256"], "script_path": str(Path(__file__)),
                "script_sha256": sha256(Path(__file__)), "raw_root": str(args.raw),
                "raw_metadata_sha256": sha256(args.raw / "metadata.json"),
                "rgb_bank_sha256": bank_sha,
                "geometries": list(GEOMETRIES), "optimizers": list(OPTIMIZERS), "seeds": list(SEEDS),
                "task_points": list(POINTS), "transition_window": list(WINDOW),
                "definitions": {"R": "sqrt(tr(W1 Sigma W1.T)) on the centered fixed 1200-image bank",
                                "raw_W_F": "Frobenius norm of stored full ambient W1",
                                "D_eff": "sqrt(tr(DeltaW1 Sigma DeltaW1.T))",
                                "c": "X/sqrt(Vprev Q)",
                                "Rstar_local": "D_eff/(2*abs(c)) only when c<0; descriptive local ratio, not convergence",
                                "R_over_Rstar_local": "R_prev/Rstar_local",
                                "window_statistic": "median across tasks26..30 within seed, then paired geometry/rgb32 ratio, then five-seed median"},
                "caveat": "Interim descriptive comparison; task26-30 is not registered late41-50 and no convergence verdict is made."}
    (args.out / "metadata.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(groups[(groups.metric.isin(("R_prev", "raw_W_prev_F", "D_eff", "Rstar_local", "R_over_Rstar_local", "R_t30", "raw_W_F_t30")))].to_string(index=False))


if __name__ == "__main__":
    main()
