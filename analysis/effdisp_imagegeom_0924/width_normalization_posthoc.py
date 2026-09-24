#!/usr/bin/env python3
"""Post hoc width/displacement normalizations for final image geometry output.

No training, registered verdicts, or raw-snapshot loading. Run after the
50-series final analyzer has produced its complete output tables.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/home/issan/Projects/claude/wt/effdisp_imagegeom_0924")
DEFAULT_INPUT = REPO / "results/effdisp_imagegeom_0924"
SOURCE_HASH = "273c3fa3b52dc0c954eaaa58a4e711183c259770"
GEOMETRIES = {"rgb32": 3072, "dup64": 12288, "dup64_half": 12288,
              "gray32": 1024, "avg16": 768}
OPTIMIZERS = ("adam", "sgd")
SEEDS = tuple(range(300, 305))
WINDOWS = {"early": (6, 15), "late": (41, 50)}
H = 100
METRICS = ("R", "R_div_sqrt_d", "R_div_sqrt_trace", "raw_W_RMS", "A_width",
           "D_eff", "D_div_sqrt_d", "D_div_sqrt_trace", "delta_W_RMS", "A_displacement")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_median(values) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if len(values) else math.nan


def expected_cases(geometries=GEOMETRIES):
    return {(g, o, s) for g in geometries for o in OPTIMIZERS for s in SEEDS}


def load_final_tables(input_dir: Path):
    input_dir = input_dir.resolve()
    metadata = json.loads((input_dir / "analysis_metadata.json").read_text())
    if (metadata.get("status") not in ("COMPLETE", "TERMINAL_WITH_DIVERGENCE")
            or metadata.get("series") != 50 or metadata.get("tasks_per_series") != 50
            or metadata.get("source_git_hash") != SOURCE_HASH):
        raise ValueError("final analysis metadata is not the registered terminal 50-series result")
    completion = pd.read_csv(input_dir / "completion.csv")
    if (len(completion) != 50 or completion.duplicated(["geometry", "optimizer", "seed"]).any()
            or set(zip(completion.geometry, completion.optimizer, completion.seed)) != expected_cases()
            or not completion.status.isin(("COMPLETE", "DIVERGED")).all()):
        raise ValueError("final completion table does not identify 50 terminal series")
    transitions = pd.read_csv(input_dir / "transitions.csv")
    covstats = pd.read_csv(input_dir / "covstats.csv")
    if (len(covstats) != 25 or covstats.duplicated(["geometry", "seed"]).any()
            or set(zip(covstats.geometry, covstats.seed)) != {(g, s) for g in GEOMETRIES for s in SEEDS}):
        raise ValueError("covstats does not identify five geometries times five seeds")
    if (transitions.duplicated(["geometry", "optimizer", "seed", "task"]).any()
            or not set(zip(transitions.geometry, transitions.optimizer, transitions.seed)).issubset(expected_cases())):
        raise ValueError("transition table has duplicate or foreign series")
    for case in completion.itertuples(index=False):
        part = transitions[(transitions.geometry == case.geometry)
                           & (transitions.optimizer == case.optimizer)
                           & (transitions.seed == case.seed)]
        last = int(case.last_saved_task)
        if len(part) != last or set(part.task) != set(range(1, last + 1)):
            raise ValueError(f"transition count does not match terminal snapshot boundary: {case.geometry}/{case.optimizer}/{case.seed}")
    return transitions, covstats, completion, metadata


def normalized_tasks(transitions: pd.DataFrame, covstats: pd.DataFrame,
                     geometries=GEOMETRIES) -> tuple[pd.DataFrame, dict]:
    needed = {"geometry", "optimizer", "seed", "task", "R", "d_eff", "raw_W_sq", "raw_D_sq", "V", "Q"}
    if absent := needed - set(transitions.columns):
        raise ValueError(f"transitions missing {sorted(absent)}")
    need_cov = {"geometry", "seed", "dimension", "cov_trace"}
    if absent := need_cov - set(covstats.columns):
        raise ValueError(f"covstats missing {sorted(absent)}")
    cov = covstats[["geometry", "seed", "dimension", "cov_trace"]]
    frame = transitions.merge(cov, on=["geometry", "seed"], how="left", validate="many_to_one")
    if len(frame) != len(transitions):
        raise ValueError("covariance merge changed transition count")
    if not np.isfinite(frame[["R", "d_eff", "raw_W_sq", "raw_D_sq", "V", "Q",
                              "dimension", "cov_trace"]].to_numpy(dtype=np.float64)).all():
        raise ValueError("nonfinite primary transition or covariance quantity")
    for g, d in geometries.items():
        if not (frame.loc[frame.geometry == g, "dimension"] == d).all():
            raise ValueError(f"dimension mismatch for {g}")
    if (frame[["R", "d_eff", "raw_W_sq", "raw_D_sq", "V", "Q"]] < 0).any().any() or (frame.cov_trace <= 0).any():
        raise ValueError("negative norm/squared norm or nonpositive input trace")
    d = frame.dimension.to_numpy(dtype=np.float64)
    trace = frame.cov_trace.to_numpy(dtype=np.float64)
    R = frame.R.to_numpy(dtype=np.float64)
    D = frame.d_eff.to_numpy(dtype=np.float64)
    raw_w_sq = frame.raw_W_sq.to_numpy(dtype=np.float64)
    raw_d_sq = frame.raw_D_sq.to_numpy(dtype=np.float64)
    raw_w_rms = np.sqrt(raw_w_sq / (H*d))
    delta_w_rms = np.sqrt(raw_d_sq / (H*d))
    w_denom = raw_w_rms * np.sqrt(H*trace)
    delta_denom = delta_w_rms * np.sqrt(H*trace)
    A_w = np.divide(R, w_denom, out=np.full(len(frame), np.nan), where=w_denom > 0)
    A_d = np.divide(D, delta_denom, out=np.full(len(frame), np.nan), where=delta_denom > 0)
    result = frame[["geometry", "optimizer", "seed", "task", "dimension", "cov_trace"]].copy()
    result["R"] = R
    result["R_div_sqrt_d"] = R / np.sqrt(d)
    result["R_div_sqrt_trace"] = R / np.sqrt(trace)
    result["raw_W_RMS"] = raw_w_rms
    result["A_width"] = A_w
    result["D_eff"] = D
    result["D_div_sqrt_d"] = D / np.sqrt(d)
    result["D_div_sqrt_trace"] = D / np.sqrt(trace)
    result["delta_W_RMS"] = delta_w_rms
    result["A_displacement"] = A_d
    # Audit the squared identities against the independently saved V and Q.
    reconstructed_v = raw_w_rms**2 * H * trace * A_w**2
    reconstructed_q = delta_w_rms**2 * H * trace * A_d**2
    width_defined = np.isfinite(A_w)
    displacement_defined = np.isfinite(A_d)
    width_error = np.abs(reconstructed_v[width_defined] - frame.V.to_numpy(dtype=np.float64)[width_defined])
    displacement_error = np.abs(reconstructed_q[displacement_defined] - frame.Q.to_numpy(dtype=np.float64)[displacement_defined])
    width_scale = np.maximum(1, frame.V.to_numpy(dtype=np.float64)[width_defined])
    displacement_scale = np.maximum(1, frame.Q.to_numpy(dtype=np.float64)[displacement_defined])
    max_width_rel = float(np.max(width_error/width_scale)) if len(width_error) else math.nan
    max_displacement_rel = float(np.max(displacement_error/displacement_scale)) if len(displacement_error) else math.nan
    if (np.isfinite(max_width_rel) and max_width_rel > 1e-9
            or np.isfinite(max_displacement_rel) and max_displacement_rel > 1e-9):
        raise ValueError("per-task V/Q decomposition exceeds float64 tolerance")
    audit = {"rows": len(result), "width_defined_rows": int(width_defined.sum()),
             "displacement_defined_rows": int(displacement_defined.sum()),
             "max_width_relative_identity_residual": max_width_rel,
             "max_displacement_relative_identity_residual": max_displacement_rel}
    return result, audit


def seed_windows(tasks: pd.DataFrame, completion: pd.DataFrame,
                 geometries=GEOMETRIES) -> pd.DataFrame:
    blocks = {(g, o, int(s)): part for (g, o, s), part in tasks.groupby(["geometry", "optimizer", "seed"])}
    status = {(r.geometry, r.optimizer, int(r.seed)): r.status for r in completion.itertuples(index=False)}
    rows = []
    for g in geometries:
        for o in OPTIMIZERS:
            for s in SEEDS:
                block = blocks.get((g, o, s))
                for name, (first, last) in WINDOWS.items():
                    part = (block[(block.task >= first) & (block.task <= last)]
                            if block is not None else pd.DataFrame())
                    has_tasks = len(part) == 10 and set(part.task) == set(range(first, last+1))
                    row = {"geometry": g, "optimizer": o, "seed": s, "window": name,
                           "first_task": first, "last_task": last,
                           "series_status": status[(g, o, s)], "has_all_tasks": bool(has_tasks)}
                    for metric in METRICS:
                        values = part[metric].to_numpy(dtype=np.float64) if has_tasks else np.array([])
                        row[metric] = finite_median(values) if len(values) == 10 and np.isfinite(values).all() else math.nan
                        row[f"n_finite_{metric}"] = int(np.isfinite(values).sum())
                    rows.append(row)
    return pd.DataFrame(rows)


def pair_and_group(windows: pd.DataFrame, geometries=GEOMETRIES):
    lookup = {(r.geometry, r.optimizer, int(r.seed), r.window): r for r in windows.itertuples(index=False)}
    paired = []
    for geometry in geometries:
        if geometry == "rgb32":
            continue
        for optimizer in OPTIMIZERS:
            for seed in SEEDS:
                for window in WINDOWS:
                    a = lookup[(geometry, optimizer, seed, window)]
                    b = lookup[("rgb32", optimizer, seed, window)]
                    for metric in METRICS:
                        av, bv = float(getattr(a, metric)), float(getattr(b, metric))
                        valid = np.isfinite(av) and np.isfinite(bv) and bv > 0
                        paired.append({"geometry": geometry, "reference_geometry": "rgb32",
                                       "optimizer": optimizer, "seed": seed, "window": window,
                                       "metric": metric, "geometry_series_status": a.series_status,
                                       "reference_series_status": b.series_status,
                                       "window_pair_available": bool(valid),
                                       "geometry_value": av, "reference_value": bv,
                                       "paired_ratio": av/bv if valid else math.nan})
    paired = pd.DataFrame(paired)
    groups = []
    for (geometry, optimizer, window, metric), part in paired.groupby(["geometry", "optimizer", "window", "metric"]):
        values = part.paired_ratio.to_numpy(dtype=np.float64)
        valid = values[np.isfinite(values)]
        groups.append({"geometry": geometry, "optimizer": optimizer, "window": window,
                       "metric": metric, "n_seed": 5, "n_valid": len(valid),
                       "paired_ratio_seed_median": float(np.median(valid)) if len(valid) else math.nan,
                       "paired_ratio_seed_min": float(np.min(valid)) if len(valid) else math.nan,
                       "paired_ratio_seed_max": float(np.max(valid)) if len(valid) else math.nan})
    return paired, pd.DataFrame(groups)


def actual_initial_width_growth(transitions: pd.DataFrame, completion: pd.DataFrame,
                                geometries=GEOMETRIES):
    """Use POST-task R_next for every t>0 and actual task-0 R from transition 1."""
    if absent := {"R_next", "R"} - set(transitions.columns):
        raise ValueError(f"growth needs {sorted(absent)}")
    blocks = {(g, o, int(s)): block.sort_values("task") for (g, o, s), block in
              transitions.groupby(["geometry", "optimizer", "seed"])}
    status = {(r.geometry, r.optimizer, int(r.seed)): r.status for r in completion.itertuples(index=False)}
    per_task, series = [], []
    max_adjacent_error = 0.0
    for geometry in geometries:
        for optimizer in OPTIMIZERS:
            for seed in SEEDS:
                block = blocks.get((geometry, optimizer, seed))
                R0 = math.nan
                if block is not None and len(block):
                    if int(block.task.iloc[0]) != 1:
                        raise ValueError("transition 1 required to identify actual task-0 width")
                    R0 = float(block.R.iloc[0])
                    if not np.isfinite(R0) or R0 <= 0:
                        raise ValueError("actual task-0 effective width must be finite and positive")
                    pre = block.R.to_numpy(dtype=np.float64)
                    post = block.R_next.to_numpy(dtype=np.float64)
                    if not np.isfinite(pre).all() or not np.isfinite(post).all() or np.any(post < 0):
                        raise ValueError("nonfinite or negative POST-task width")
                    if len(block) > 1:
                        adjacent_error = np.abs(post[:-1] - pre[1:])
                        max_adjacent_error = max(max_adjacent_error, float(np.max(adjacent_error)))
                        if not np.allclose(post[:-1], pre[1:], rtol=2e-12, atol=2e-10):
                            raise ValueError("adjacent POST-task and next pre-task widths disagree")
                    for row in block.itertuples(index=False):
                        width = float(row.R_next)
                        per_task.append({"geometry": geometry, "optimizer": optimizer,
                                         "seed": seed, "task": int(row.task),
                                         "series_status": status[(geometry, optimizer, seed)],
                                         "R0_actual": R0, "R_posttask": width,
                                         "R_posttask_over_R0": width/R0})
                item = {"geometry": geometry, "optimizer": optimizer, "seed": seed,
                        "series_status": status[(geometry, optimizer, seed)],
                        "R0_actual": R0}
                for label, (first, last) in WINDOWS.items():
                    part = (block[(block.task >= first) & (block.task <= last)]
                            if block is not None else pd.DataFrame())
                    available = len(part) == 10 and set(part.task) == set(range(first, last+1))
                    widths = part.R_next.to_numpy(dtype=np.float64) if available else np.array([])
                    available = bool(available and np.isfinite(widths).all() and np.isfinite(R0))
                    item[f"{label}_POSTtask_window_available"] = available
                    item[f"R_POSTtask_{first}_{last}_median"] = float(np.median(widths)) if available else math.nan
                    item[f"R_POSTtask_over_R0_{first}_{last}_median"] = float(np.median(widths/R0)) if available else math.nan
                t50 = (block[block.task == 50] if block is not None else pd.DataFrame())
                R50 = float(t50.R_next.iloc[0]) if len(t50) == 1 else math.nan
                item["R50_POSTtask"] = R50
                item["R50_POSTtask_over_R0"] = R50/R0 if np.isfinite(R50) and np.isfinite(R0) else math.nan
                series.append(item)
    per_task = pd.DataFrame(per_task, columns=["geometry", "optimizer", "seed", "task",
                                               "series_status", "R0_actual", "R_posttask",
                                               "R_posttask_over_R0"])
    series = pd.DataFrame(series)
    growth_metrics = ("R50_POSTtask_over_R0", "R_POSTtask_over_R0_6_15_median",
                      "R_POSTtask_over_R0_41_50_median")
    lookup = {(r.geometry, r.optimizer, int(r.seed)): r for r in series.itertuples(index=False)}
    paired = []
    for geometry in geometries:
        if geometry == "rgb32":
            continue
        for optimizer in OPTIMIZERS:
            for seed in SEEDS:
                a = lookup[(geometry, optimizer, seed)]
                b = lookup[("rgb32", optimizer, seed)]
                for metric in growth_metrics:
                    av, bv = float(getattr(a, metric)), float(getattr(b, metric))
                    valid = np.isfinite(av) and np.isfinite(bv) and bv > 0
                    paired.append({"geometry": geometry, "reference_geometry": "rgb32",
                                   "optimizer": optimizer, "seed": seed, "metric": metric,
                                   "geometry_series_status": a.series_status,
                                   "reference_series_status": b.series_status,
                                   "geometry_growth": av, "reference_growth": bv,
                                   "paired_growth_ratio": av/bv if valid else math.nan})
    paired = pd.DataFrame(paired)
    groups = []
    for (geometry, optimizer, metric), part in paired.groupby(["geometry", "optimizer", "metric"]):
        values = part.paired_growth_ratio.to_numpy(dtype=np.float64)
        valid = values[np.isfinite(values)]
        groups.append({"geometry": geometry, "optimizer": optimizer, "metric": metric,
                       "n_seed": 5, "n_valid": len(valid),
                       "paired_growth_seed_median": float(np.median(valid)) if len(valid) else math.nan,
                       "paired_growth_seed_min": float(np.min(valid)) if len(valid) else math.nan,
                       "paired_growth_seed_max": float(np.max(valid)) if len(valid) else math.nan})
    absolute = []
    for (geometry, optimizer), block in series.groupby(["geometry", "optimizer"]):
        for metric in ("R0_actual", "R50_POSTtask", "R_POSTtask_6_15_median",
                       "R_POSTtask_41_50_median", *growth_metrics):
            values = block[metric].to_numpy(dtype=np.float64)
            valid = values[np.isfinite(values)]
            absolute.append({"geometry": geometry, "optimizer": optimizer, "metric": metric,
                             "n_seed": 5, "n_valid": len(valid),
                             "seed_median": float(np.median(valid)) if len(valid) else math.nan,
                             "seed_min": float(np.min(valid)) if len(valid) else math.nan,
                             "seed_max": float(np.max(valid)) if len(valid) else math.nan})
    audit = {"growth_per_task_rows": len(per_task), "growth_series_rows": len(series),
             "growth_paired_rows": len(paired),
             "max_adjacent_POSTtask_pretask_abs_error": max_adjacent_error}
    return per_task, series, paired, pd.DataFrame(groups), pd.DataFrame(absolute), audit


def definitions():
    return {"status": "POSTHOC_DESCRIPTIVE_ONLY; no preregistered verdict",
            "order": "per-task correction -> median of tasks 6–15 or 41–50 within seed -> geometry/rgb32 paired ratio -> median/min/max over five seeds",
            "R": "sqrt(V) for W at transition start; V=tr(W Sigma W.T)",
            "D_eff": "sqrt(Q) for task transition; Q=tr(DeltaW Sigma DeltaW.T)",
            "R_div_sqrt_d": "R/sqrt(input dimension d)",
            "R_div_sqrt_trace": "R/sqrt(tr Sigma)",
            "raw_W_RMS": "sqrt(||W||F^2/(h*d)), h=100",
            "A_width": "R/[raw_W_RMS*sqrt(h*tr Sigma)]",
            "D_div_sqrt_d": "D_eff/sqrt(d)",
            "D_div_sqrt_trace": "D_eff/sqrt(tr Sigma)",
            "delta_W_RMS": "sqrt(||DeltaW||F^2/(h*d)), h=100",
            "A_displacement": "D_eff/[delta_W_RMS*sqrt(h*tr Sigma)]",
            "exact_squared_identities": ["V=raw_W_RMS^2*h*trSigma*A_width^2",
                                         "Q=delta_W_RMS^2*h*trSigma*A_displacement^2"],
            "actual_initial_width_growth": "R0 is actual transition1 previous width; R_t for t>0 is POST-task transition t R_next. Correct POST-task R_t/R0 per task first, then median over tasks6–15 or41–50 within seed, then geometry/rgb32 paired ratio, then five-seed median/min/max. R50 uses transition50 R_next.",
            "caveat": "sqrt(d) is a fixed-coordinate-amplitude benchmark, not a causal correction. For native fan-in uniform initialization E[R^2]=h*trSigma/(3d). Correlated inputs, optimizer coordinates, information content, and initialization also differ. Do not multiply medians to claim a decomposition. Width-growth ratios are descriptive, not convergence verdicts."}


def selftest_priority(priority: Path) -> dict:
    transitions = pd.read_csv(priority / "transitions.csv")
    covstats = pd.read_csv(priority / "covstats.csv")
    geometries = {"rgb32": 3072, "avg16": 768}
    tasks, audit = normalized_tasks(transitions, covstats, geometries)
    completion = pd.DataFrame([{"geometry": g, "optimizer": o, "seed": s, "status": "COMPLETE"}
                               for g in geometries for o in OPTIMIZERS for s in SEEDS])
    windows = seed_windows(tasks, completion, geometries)
    paired, _ = pair_and_group(windows, geometries)
    prior = pd.read_csv(priority / "width_normalization_posthoc.csv")
    mapping = {"R": "R", "R_div_sqrt_d": "R_div_sqrt_d",
               "R_div_sqrt_trace": "R_div_sqrt_input_trace", "raw_W_RMS": "raw_W_RMS",
               "A_width": "width_alignment_factor"}
    errors = []
    for row in prior.itertuples(index=False):
        metric = next(k for k, v in mapping.items() if v == row.field)
        found = paired[(paired.geometry == "avg16") & (paired.optimizer == row.optimizer)
                       & (paired.seed == row.seed) & (paired.window == "late")
                       & (paired.metric == metric)].iloc[0]
        errors.extend((abs(found.geometry_value-row.avg16_value),
                       abs(found.reference_value-row.rgb32_value),
                       abs(found.paired_ratio-row.paired_ratio)))
    maximum = max(errors)
    if maximum > 2e-12:
        raise ValueError(f"priority posthoc reproduction failed: max abs error {maximum}")
    _, growth_series, growth_paired, _, _, growth_audit = actual_initial_width_growth(
        transitions, completion, geometries)
    previous_growth = pd.read_csv(priority / "initial_growth_posthoc" / "series_width_growth.csv")
    previous_pairs = pd.read_csv(priority / "initial_growth_posthoc" / "paired_growth.csv")
    growth_errors = []
    series_mapping = {"R0_actual": "R0_actual", "R50_POSTtask": "R50_posttask",
                      "R50_POSTtask_over_R0": "R50_over_R0",
                      "R_POSTtask_over_R0_41_50_median": "R_posttask_over_R0_41_50_median"}
    for old in previous_growth.itertuples(index=False):
        new = growth_series[(growth_series.geometry == old.geometry)
                            & (growth_series.optimizer == old.optimizer)
                            & (growth_series.seed == old.seed)].iloc[0]
        growth_errors.extend(abs(float(new[k])-float(getattr(old, v)))
                             for k, v in series_mapping.items())
    pair_mapping = {"R50_over_R0": "R50_POSTtask_over_R0",
                    "R_posttask_over_R0_41_50_median": "R_POSTtask_over_R0_41_50_median"}
    for old in previous_pairs.itertuples(index=False):
        new = growth_paired[(growth_paired.geometry == "avg16")
                            & (growth_paired.optimizer == old.optimizer)
                            & (growth_paired.seed == old.seed)
                            & (growth_paired.metric == pair_mapping[old.metric])].iloc[0]
        growth_errors.append(abs(float(new.paired_growth_ratio)-float(old.paired_growth_ratio_avg16_over_rgb32)))
    growth_maximum = max(growth_errors)
    if growth_maximum > 2e-12:
        raise ValueError(f"priority initial-growth reproduction failed: max abs error {growth_maximum}")
    return {"status": "PASS", "compared_rows": len(prior),
            "max_abs_reproduction_error": maximum,
            "compared_growth_series": len(previous_growth),
            "compared_growth_pairs": len(previous_pairs),
            "max_abs_growth_reproduction_error": growth_maximum, **audit, **growth_audit}


def run(input_dir: Path, out: Path) -> dict:
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty output: {out}")
    transitions, covstats, completion, source = load_final_tables(input_dir)
    tasks, decomposition_audit = normalized_tasks(transitions, covstats)
    windows = seed_windows(tasks, completion)
    paired, groups = pair_and_group(windows)
    growth_tasks, growth_series, growth_pairs, growth_groups, growth_absolute, growth_audit = (
        actual_initial_width_growth(transitions, completion))
    out.mkdir(parents=True, exist_ok=True)
    outputs = {"per_task_normalized.csv": tasks, "seed_windows.csv": windows,
               "paired_ratios.csv": paired, "groups.csv": groups,
               "growth_POSTtask_per_task.csv": growth_tasks,
               "growth_POSTtask_series.csv": growth_series,
               "growth_POSTtask_paired.csv": growth_pairs,
               "growth_POSTtask_groups.csv": growth_groups,
               "growth_POSTtask_absolute_groups.csv": growth_absolute}
    for name, table in outputs.items():
        table.to_csv(out / name, index=False)
    (out / "definitions.json").write_text(json.dumps(definitions(), indent=2) + "\n")
    audit = {"status": "PASS", "source_analysis_status": source["status"],
             "source_git_hash": source["source_git_hash"],
             "expected_series": 50, "window_rows": len(windows),
             "paired_rows": len(paired), "group_rows": len(groups),
             **decomposition_audit, **growth_audit}
    (out / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    inputs = {name: sha256(input_dir / name) for name in
              ("analysis_metadata.json", "completion.csv", "transitions.csv", "covstats.csv")}
    result = {"status": "COMPLETE_POSTHOC", "source_analysis": str(input_dir),
              "source_git_hash": SOURCE_HASH, "script_path": str(Path(__file__)),
              "script_sha256": sha256(Path(__file__)), "input_sha256": inputs,
              "output_sha256": {name: sha256(out / name) for name in
                                (*outputs, "definitions.json", "audit.json")},
              "warning": "Descriptive normalizations, not registered verdicts or causal dimension corrections."}
    (out / "metadata.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_INPUT / "width_normalization_posthoc")
    parser.add_argument("--selftest-priority", type=Path,
                        help="validate formulas against existing two-condition priority CSVs; writes nothing")
    args = parser.parse_args()
    result = (selftest_priority(args.selftest_priority) if args.selftest_priority else
              run(args.input, args.out))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
