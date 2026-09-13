#!/usr/bin/env python3
"""Registered descriptive unit summary for the ELU response-anchor run."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "elu_response_anchor_0913"
SOURCE = OUT / "units.npz"
CSV_OUT = OUT / "selected_unit_summary.csv"
MD_OUT = OUT / "selected_unit_summary.md"
TASKS = tuple(range(21, 26))
BRANCHES = ("A", "B", "C", "D", "AF", "BF")
ALL_STEPS = (0, 1, 2, 5, 10, 20, 25, 50, 75, 150, 375, 750, 1500, 3000, 6000)
REPORT_STEPS = (0, 1, 20, 75, 6000)
INTERVAL_WINDOWS = {0: "baseline_zero", 1: "0_to_1", 20: "10_to_20", 75: "50_to_75", 6000: "3000_to_6000"}
USED_KEYS = (
    "q_lowgate", "response_gate_mean", "effective_argument_mean",
    "actual_activation_mean", "actual_activation_std", "W2_row_norm",
    "interval_dW1_row_norm", "interval_dW2_row_norm", "interval_db2",
    "interval_dW3_column_norm", "interval_delta_z2_mean", "interval_delta_z2_std",
)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fmt(value: float) -> str:
    return f"{value:.6g}"


def require_source() -> tuple[dict, dict]:
    validation = json.loads((OUT / "validation.json").read_text())
    provenance = json.loads((OUT / "provenance.json").read_text())
    if validation.get("status") != "PASS" or provenance.get("status") != "COMPLETE":
        raise RuntimeError("unit summary requires PASS validation and COMPLETE provenance")
    actual = sha(SOURCE)
    for label, record in (("validation", validation), ("provenance", provenance)):
        expected = record.get("result_sha256", {}).get(SOURCE.name)
        if expected != actual:
            raise RuntimeError(f"{label} units.npz hash mismatch: {expected} != {actual}")
    return validation, provenance


def load_arrays() -> dict[str, np.ndarray]:
    with np.load(SOURCE, allow_pickle=False) as archive:
        required = {"tasks", "steps", "model_seed", "model_branch", "selected_unit_id", *USED_KEYS}
        missing = required.difference(archive.files)
        if missing:
            raise RuntimeError(f"units.npz missing keys: {sorted(missing)}")
        arrays = {key: np.asarray(archive[key]) for key in required}
    if tuple(arrays["tasks"].astype(int)) != TASKS or tuple(arrays["steps"].astype(int)) != ALL_STEPS:
        raise RuntimeError("registered task/probe axes mismatch")
    seeds = arrays["model_seed"].astype(int)
    branches = arrays["model_branch"].astype(str)
    expected_models = [(seed, branch) for seed in range(3) for branch in BRANCHES]
    if list(zip(seeds.tolist(), branches.tolist())) != expected_models:
        raise RuntimeError("registered model axis is not seed-major A,B,C,D,AF,BF")
    ids = arrays["selected_unit_id"].astype(int)
    if ids.shape != (18, 20) or any(len(set(row.tolist())) != 20 for row in ids):
        raise RuntimeError("selected-unit IDs are not 20 unique IDs per model")
    for seed in range(3):
        rows = ids[seed * 6:(seed + 1) * 6]
        if not all(np.array_equal(rows[0], row) for row in rows[1:]):
            raise RuntimeError(f"selected-unit IDs differ across branches for seed {seed}")
    for key in USED_KEYS:
        if arrays[key].shape != (5, 15, 18, 100) or not np.isfinite(arrays[key]).all():
            raise RuntimeError(f"invalid registered unit array {key}: {arrays[key].shape}")
    return arrays


def summarize(arrays: dict[str, np.ndarray]) -> list[dict]:
    task_index = {int(task): i for i, task in enumerate(arrays["tasks"])}
    step_index = {int(step): i for i, step in enumerate(arrays["steps"])}
    rows = []
    for task in TASKS:
        ti = task_index[task]
        for step in REPORT_STEPS:
            ui = step_index[step]
            for model_index, (seed, branch) in enumerate(zip(arrays["model_seed"].astype(int), arrays["model_branch"].astype(str))):
                selected = arrays["selected_unit_id"][model_index].astype(int)
                for population, unit_ids in (("selected20", selected), ("all100", np.arange(100))):
                    take = lambda key: arrays[key][ti, ui, model_index, unit_ids]
                    q = take("q_lowgate")
                    db = take("interval_db2")
                    dz = take("interval_delta_z2_mean")
                    rows.append({
                        "analysis_label": "REGISTERED_DESCRIPTIVE_SECONDARY",
                        "seed": int(seed), "branch": branch, "task": task, "step": step,
                        "interval_window": INTERVAL_WINDOWS[step], "population": population,
                        "unit_count": int(len(unit_ids)), "q_mean": float(q.mean()),
                        "sink_count_q_ge_0_95": int((q >= .95).sum()),
                        "response_gate_mean": float(take("response_gate_mean").mean()),
                        "effective_argument_mean": float(take("effective_argument_mean").mean()),
                        "actual_activation_mean": float(take("actual_activation_mean").mean()),
                        "actual_activation_std_mean": float(take("actual_activation_std").mean()),
                        "W2_row_norm_mean": float(take("W2_row_norm").mean()),
                        "interval_dW2_row_norm_mean": float(take("interval_dW2_row_norm").mean()),
                        "interval_db2_mean": float(db.mean()), "interval_db2_abs_mean": float(np.abs(db).mean()),
                        "interval_dW3_column_norm_mean": float(take("interval_dW3_column_norm").mean()),
                        "interval_realized_dz_mean": float(dz.mean()),
                        "interval_realized_dz_abs_mean": float(np.abs(dz).mean()),
                        "interval_realized_dz_input_sd_mean": float(take("interval_delta_z2_std").mean()),
                    })
    if len(rows) != 3 * 6 * 5 * 5 * 2:
        raise RuntimeError(f"summary row count {len(rows)} != 900")
    keys = {(r["seed"], r["branch"], r["task"], r["step"], r["population"]) for r in rows}
    if len(keys) != len(rows):
        raise RuntimeError("duplicate selected-unit summary keys")
    return rows


def mean_row(rows: list[dict], branch: str, task: int, step: int, population: str = "selected20") -> dict[str, float]:
    found = [r for r in rows if r["branch"] == branch and r["task"] == task and r["step"] == step and r["population"] == population]
    if len(found) != 3:
        raise RuntimeError(f"expected three seed rows for {branch} task{task} step{step} {population}")
    numeric = (
        "q_mean", "sink_count_q_ge_0_95", "effective_argument_mean", "actual_activation_std_mean",
        "W2_row_norm_mean", "interval_dW2_row_norm_mean", "interval_db2_abs_mean",
        "interval_dW3_column_norm_mean", "interval_realized_dz_abs_mean",
    )
    return {key: float(np.mean([r[key] for r in found], dtype=np.float64)) for key in numeric}


def markdown(rows: list[dict], arrays: dict[str, np.ndarray], validation: dict, provenance: dict) -> str:
    source_hash = sha(SOURCE)
    afbf = np.isin(arrays["model_branch"].astype(str), ("AF", "BF"))
    freeze_max = float(np.abs(arrays["interval_dW1_row_norm"][:, :, afbf, :]).max())
    if freeze_max != 0:
        raise RuntimeError(f"AF/BF registered W1 interval change is nonzero: {freeze_max}")

    lines = [
        "# ELU response-anchor selected-unit summary",
        "",
        "**REGISTERED DESCRIPTIVE SECONDARY.** This summarizes the completed preregistered run. Seeds remain the replicate unit; units, tasks, and probe steps are not independent replicates. No inferential or new causal label is assigned here.",
        "",
        "## Source and windows",
        "",
        f"Source file: `results/elu_response_anchor_0913/{SOURCE.name}` (SHA256 `{source_hash}`). Validation status `{validation['status']}`; provenance status `{provenance['status']}`.",
        "",
        "Exact source keys: `" + "`, `".join(USED_KEYS) + "`, plus `tasks`, `steps`, `model_seed`, `model_branch`, and `selected_unit_id`.",
        "",
        "Reported tasks are 21-25. Reported probe steps are 0, 1, 20, 75, and 6000. Interval arrays always compare with the preceding point on the full registered probe grid: step 0 is stored zero; 0->1; 10->20; 50->75; and 3000->6000. Thus the step-20, step-75, and step-6000 interval columns are not cumulative from task start.",
        "",
        "Each CSV coordinate has one `selected20` row and one `all100` row. `actual_activation_std_mean` is the mean across units of each unit's SD over the 1,200 fixed inputs. `interval_realized_dz_*` comes directly from input-wise changes in unshifted z2 over the registered interval.",
        "",
        "## B gate state and corrected feature spread",
        "",
        "The table gives arithmetic means across the three seed-level rows for the fixed selected-20 cohort. It displays gate re-sinking (`q`) alongside corrected activation spread; it does not classify either as a mediated effect.",
        "",
        "| task | step | A q | B q | A sink /20 | B sink /20 | A corrected act SD | B corrected act SD |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for task in TASKS:
        for step in REPORT_STEPS:
            a, b = mean_row(rows, "A", task, step), mean_row(rows, "B", task, step)
            lines.append(f"| {task} | {step} | {fmt(a['q_mean'])} | {fmt(b['q_mean'])} | {fmt(a['sink_count_q_ge_0_95'])} | {fmt(b['sink_count_q_ge_0_95'])} | {fmt(a['actual_activation_std_mean'])} | {fmt(b['actual_activation_std_mean'])} |")

    lines += [
        "",
        "## Six-branch task-21 selected-unit dynamics",
        "",
        "Three-seed arithmetic means. `|dz|` and parameter changes use the exact interval named above.",
        "",
        "| branch | step | q | sink /20 | effective arg mean | corrected act SD | W2 norm | interval dW2 | interval abs(db2) | interval dW3 | interval abs(dz) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for branch in BRANCHES:
        for step in REPORT_STEPS:
            r = mean_row(rows, branch, 21, step)
            lines.append(f"| {branch} | {step} | {fmt(r['q_mean'])} | {fmt(r['sink_count_q_ge_0_95'])} | {fmt(r['effective_argument_mean'])} | {fmt(r['actual_activation_std_mean'])} | {fmt(r['W2_row_norm_mean'])} | {fmt(r['interval_dW2_row_norm_mean'])} | {fmt(r['interval_db2_abs_mean'])} | {fmt(r['interval_dW3_column_norm_mean'])} | {fmt(r['interval_realized_dz_abs_mean'])} |")

    lines += [
        "",
        "## Five-task selected-unit endpoints",
        "",
        "Three-seed arithmetic means at step 6000.",
        "",
        "| task | branch | q | sink /20 | effective arg mean | corrected act SD | W2 norm | interval dW2 | interval abs(db2) | interval dW3 | interval abs(dz) |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for task in TASKS:
        for branch in BRANCHES:
            r = mean_row(rows, branch, task, 6000)
            lines.append(f"| {task} | {branch} | {fmt(r['q_mean'])} | {fmt(r['sink_count_q_ge_0_95'])} | {fmt(r['effective_argument_mean'])} | {fmt(r['actual_activation_std_mean'])} | {fmt(r['W2_row_norm_mean'])} | {fmt(r['interval_dW2_row_norm_mean'])} | {fmt(r['interval_db2_abs_mean'])} | {fmt(r['interval_dW3_column_norm_mean'])} | {fmt(r['interval_realized_dz_abs_mean'])} |")

    lines += [
        "",
        "## Layer-1 freeze check",
        "",
        f"Across every registered task, all 15 probe steps, both AF/BF branches, all three seeds, and all 100 W1 rows, the maximum `interval_dW1_row_norm` is `{freeze_max:.1f}`. This verifies the saved interval measurements for the registered layer-1 freeze; Adam moments are outside this unit-summary file.",
        "",
        "The CSV retains seed identity and contains all six branches, all five tasks, all five requested report steps, and both selected-20 and all-100 populations (900 rows). Branch differences are descriptive outcomes of the registered intervention.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    validation, provenance = require_source()
    arrays = load_arrays()
    rows = summarize(arrays)
    with CSV_OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    MD_OUT.write_text(markdown(rows, arrays, validation, provenance), encoding="utf-8")
    print(json.dumps({
        "status": "COMPLETE", "analysis_label": "REGISTERED_DESCRIPTIVE_SECONDARY",
        "source": str(SOURCE), "source_sha256": sha(SOURCE), "csv_rows": len(rows),
        "csv": str(CSV_OUT), "markdown": str(MD_OUT),
    }, indent=2))


if __name__ == "__main__":
    main()
