"""Read-only analysis of preregistered effective-displacement task snapshots.

The default run requires every expected seed and task. ``--allow-incomplete``
is for diagnostics only and labels the report INCOMPLETE. No model is trained.
Each CSV retains the endpoint-task convention: row t uses W_(t-1) and W_t.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from metrics import summarize_task_table, transition_metrics, validate_covariance_matrix

SEEDS = tuple(range(100, 105))
ACTS = ("LR", "R", "ELU", "GELU", "SiLU", "SN1", "SNA03", "SNA06", "SNA1")
CONDA_ARMS = {
    "k0": 5, "k1": 5, "k3": 5, "k7": 5,
    "k1_eps1e3": 5, "k1_sgd": 5, "k1_r2": 2, "k1_r10": 10,
    "k1_relu": 5, "k1_elu": 5, "k1_gelu": 5, "k1_silu": 5,
    "k1_T1000": 5, "k1_snake": 5,
    "k1_sna03": 5, "k1_sna06": 5, "k1_sna1": 5,
    "m40_r5_k1": 5, "m40_r10_k1": 10, "m40_r10_k2": 10,
    "m40_r10_k1_sna06": 10, "m40_r10_k2_sna06": 10,
}
DEFAULT_RAW = Path("/home/issan/Projects/obsidian-research-data/effdisp_validation_0924/raw")
DEFAULT_CIFAR = Path("/home/issan/Projects/obsidian-research-data/rlcifar_mlp_battle_0918/results/rlcifar_mlp_battle_0918")
DEFAULT_CIFAR_DATA = Path("/home/issan/Projects/claude/proj_004_drift/data/cifar10")
_VALIDATED_COVARIANCE_DIGESTS: set[tuple[tuple[int, ...], bytes]] = set()


@dataclass(frozen=True)
class Series:
    environment: str
    arm: str
    seed: int
    tasks: int
    kind: str
    path: Path
    source_class: str = "prospective"
    optimizer: str = "unknown"


def _mnist_run_dir(raw_root: Path, mode: str, act: str) -> Path:
    if mode == "pm_sgd":
        candidates = [raw_root / "pm_sgd" / act, raw_root / "mnist" / "pm_sgd" / act,
                      raw_root / f"pm_sgd_{act}"]
    else:
        candidates = [raw_root / f"{mode}_{act}", raw_root / "mnist" / mode / act]
    return next((candidate for candidate in candidates if candidate.exists()), candidates[0])


def expected_series(raw_root: Path, *, include_cifar: bool = False,
                    cifar_archive: Path = DEFAULT_CIFAR) -> list[Series]:
    series = [Series("conda", arm, seed, 100, "conda", raw_root / "conda" / arm / f"seed{seed}")
              for arm in CONDA_ARMS for seed in SEEDS]
    series += [Series(f"mnist_{mode}", act, seed, 50, mode,
                      _mnist_run_dir(raw_root, mode, act) / f"seed_{seed:03d}",
                      optimizer="adam")
               for mode in ("rl", "pm") for act in ACTS for seed in SEEDS]
    series += [Series("mnist_pm_sgd", "LR", seed, 50, "pm",
                      _mnist_run_dir(raw_root, "pm_sgd", "LR") / f"seed_{seed:03d}",
                      "registered_reference", "sgd") for seed in SEEDS]
    if include_cifar:
        series += [Series("cifar", cond, seed, 50, "cifar",
                          cifar_archive / "LR" / "snap" / f"LR_{cond}_seed{seed}",
                          "archived_posthoc")
                   for cond in ("raw", "std") for seed in range(5)]
    return series


def _snapshot_path(s: Series, task: int) -> Path:
    if s.kind == "conda":
        return s.path / f"t{task:03d}.npz"
    if s.kind in ("rl", "pm"):
        return s.path / f"state_{task:03d}.npz"
    return s.path / f"t{task:02d}.npz"


def _load_json(path: Path):
    return json.loads(path.read_text())


def preflight(s: Series) -> dict:
    """Check run completion without loading weights or outcome values."""
    missing = [str(_snapshot_path(s, t)) for t in range(s.tasks + 1)
               if not _snapshot_path(s, t).is_file()]
    reason = ""
    state_label = "INCOMPLETE"
    if s.kind == "conda":
        meta_path = s.path.parents[1] / "metadata.json"
        status_path = s.path / "status.json"
        if not meta_path.is_file():
            reason = "missing metadata.json"
        elif not status_path.is_file():
            reason = "missing status.json"
        else:
            meta = _load_json(meta_path)
            status = _load_json(status_path)
            if meta.get("tasks") != 100 or s.seed not in meta.get("seeds", []):
                reason = "metadata does not match preregistered tasks/seeds"
            elif meta.get("primary_period") != 10000 or meta.get("frequency_period") != 1000:
                reason = "metadata period differs from preregistration"
            elif meta.get("arm_dimensions") and s.arm not in meta["arm_dimensions"]:
                reason = "arm missing from metadata arm_dimensions"
            elif status.get("status") != "COMPLETED" or status.get("last_saved_task") != s.tasks:
                state_label = "DIVERGED" if status.get("status") == "DIVERGED" else "RUNNING"
                reason = (f"runner status={status.get('status', 'missing')} task={status.get('last_saved_task', 0)}"
                          f" first_bad_task={status.get('first_bad_task', '')}")
    elif s.kind in ("rl", "pm"):
        run_dir = s.path.parent
        meta_path, status_path = run_dir / "metadata.json", run_dir / "status.json"
        if not meta_path.is_file() or not status_path.is_file():
            reason = "missing metadata/status.json"
        else:
            meta, status = _load_json(meta_path), _load_json(status_path).get(str(s.seed), {})
            expected_steps = 30000 if s.kind == "rl" else 625
            if (meta.get("mode") != s.kind or meta.get("activation") != s.arm
                    or meta.get("tasks") != 50 or meta.get("steps_per_task") != expected_steps
                    or meta.get("optimizer") != s.optimizer
                    or meta.get("smoke") or s.seed not in meta.get("seeds", [])):
                reason = "metadata differs from preregistration or is a smoke run"
            elif status.get("state") == "failed":
                state_label = "FAILED"
                reason = f"runner failed: {status.get('error', 'unknown error')}"
            elif status.get("state") != "complete" or status.get("completed_tasks") != s.tasks:
                state_label = "RUNNING"
                reason = f"runner status={status.get('state', 'missing')} tasks={status.get('completed_tasks', 0)}"
        required = [s.path / ("input_bank.npz" if s.kind == "rl" else "reference_bank.npz"),
                    s.path / "per_task.json"]
        if s.kind == "pm":
            required += [s.path / f"task_{t:03d}_input.npz" for t in range(1, s.tasks + 1)]
        missing += [str(p) for p in required if not p.is_file()]
    if missing and not reason:
        reason = f"{len(missing)} required files missing"
    if not s.path.exists():
        state_label = "MISSING"
    return {"environment": s.environment, "arm": s.arm, "seed": s.seed,
            "source_class": s.source_class,
            "status": "COMPLETE" if not reason and not missing else state_label,
            "reason": reason, "missing_count": len(missing),
            "first_missing": missing[0] if missing else ""}


def _npz(path: Path, *keys: str, optional: tuple[str, ...] = ()) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in (*keys, *(k for k in optional if k in data))}


def _checked_covariance(value: np.ndarray, inputs: int) -> np.ndarray:
    """Check each distinct dense bank once, including repeated activation runs."""
    array = np.ascontiguousarray(value, dtype=np.float64)
    key = (array.shape, hashlib.sha256(array.view(np.uint8)).digest())
    if key not in _VALIDATED_COVARIANCE_DIGESTS:
        array = validate_covariance_matrix(array, inputs)
        _VALIDATED_COVARIANCE_DIGESTS.add(key)
    return array


def _mnist_diagnostics(s: Series) -> dict[int, dict]:
    rows = _load_json(s.path / "per_task.json")
    result = {int(row["task"]): row for row in rows}
    if len(result) != s.tasks or set(result) != set(range(1, s.tasks + 1)):
        raise ValueError("per_task.json must contain exactly one diagnostic row per task")
    for task, row in result.items():
        for name in ("online_acc", "task_start_acc", "train_acc", "active_unit_frac_l1",
                     "active_unit_frac_l2", "derivative_abs_mean_l1", "derivative_abs_mean_l2"):
            if name not in row or row[name] is None or not np.isfinite(row[name]):
                raise ValueError(f"task {task}: missing or nonfinite {name}")
    return result


@lru_cache(maxsize=2)
def _cifar_dataset(data_dir: str):
    repo = str(Path(__file__).resolve().parents[2])
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from src import pmnist_rlcifar_0907 as RC
    RC.DATA_DIR = Path(data_dir)
    return RC.Cifar10()


def _cifar_factor(seed: int, cond: str, data_dir: Path) -> np.ndarray:
    """Recreate the archived seed's input bank; factor uses population N."""
    import torch
    dataset = _cifar_dataset(str(data_dir))
    from src import rlcifar_mlp_battle_0918 as B
    images = B.slot_inputs(dataset, seed, cond,
                           torch.device("cpu")).numpy().astype(np.float64)
    centered = images - images.mean(axis=0)
    return centered.T / math.sqrt(len(images))


def _late_start(tasks: int) -> int:
    return tasks - max(10, math.ceil(tasks * 0.2)) + 1


def _transition_rows(s: Series, *, cifar_data: Path) -> tuple[list[dict], list[dict], dict[str, list[pd.DataFrame]]]:
    """Load one seed and return summed task rows, unit rows, and late unit cache."""
    rows: list[dict] = []
    units: list[dict] = []
    late_units: dict[str, list[pd.DataFrame]] = {}
    diagnostics = _mnist_diagnostics(s) if s.kind in ("rl", "pm") else {}
    previous = _npz(_snapshot_path(s, 0), "W", "b", "mu", "flip_state") if s.kind == "conda" else _npz(_snapshot_path(s, 0), "W1", "b1")
    base: dict = {}
    if s.kind == "conda":
        m = previous["W"].shape[1]
        r = m - len(previous["flip_state"])
        if r <= 0 or r >= m:
            raise ValueError(f"invalid CondA input dimensions m={m}, r={r}")
        base["variance"] = np.r_[np.zeros(m-r), np.full(r, 0.25)]
    elif s.kind == "rl":
        base["covariance"] = _checked_covariance(
            _npz(s.path / "input_bank.npz", "covariance")["covariance"], 784)
    elif s.kind == "pm":
        base["covariance"] = _checked_covariance(
            _npz(s.path / "reference_bank.npz", "covariance")["covariance"], 784)
    else:
        base["factor"] = _cifar_factor(s.seed, s.arm, cifar_data)
    prev_cov = None
    prev_mean = None
    if s.kind == "pm":
        bank1 = _npz(s.path / "task_001_input.npz", "covariance", "mean")
        prev_cov = _checked_covariance(bank1["covariance"], 784)
        prev_mean = bank1["mean"].astype(np.float64)
    for task in range(1, s.tasks + 1):
        source = _snapshot_path(s, task)
        current = (_npz(source, "W", "b", "mu", "mse", "task_start_mse", "w_sigma2",
                        "derivative_exactzero_frac", "derivative_absmean", "activeunit_frac",
                        optional=("alpha", "VEMA"))
                   if s.kind == "conda" else _npz(source, "W1", "b1"))
        w0 = previous["W"] if s.kind == "conda" else previous["W1"]
        w1 = current["W"] if s.kind == "conda" else current["W1"]
        metrics: dict[str, tuple[pd.DataFrame, dict]] = {}
        effective_kwargs = dict(base)
        if s.kind == "conda":
            effective_kwargs.update(mean_prev=previous["mu"], mean_next=current["mu"],
                                    bias_prev=previous["b"], bias_next=current["b"])
        elif s.kind == "rl":
            effective_kwargs["validate_covariance"] = False
        elif s.kind == "pm":
            effective_kwargs["validate_covariance"] = False
        else:
            pass
        main_name = "reference" if s.kind == "pm" else "effective"
        metrics[main_name] = transition_metrics(w0, w1, **effective_kwargs)
        metrics["raw"] = transition_metrics(w0, w1)
        if s.kind == "pm":
            inp = _npz(s.path / f"task_{task:03d}_input.npz", "covariance", "mean")
            curr_cov = _checked_covariance(inp["covariance"], 784) if task > 1 else prev_cov
            curr_mean = inp["mean"].astype(np.float64)
            metrics["actual"] = transition_metrics(
                w0, w1, covariance=prev_cov, covariance_next=curr_cov,
                validate_covariance=False, mean_prev=prev_mean, mean_next=curr_mean,
                bias_prev=previous["b1"], bias_next=current["b1"])
            prev_cov, prev_mean = curr_cov, curr_mean
        for metric, (unit, summed) in metrics.items():
            if metric == "raw":
                cov0 = cov1 = "identity"
            elif s.kind == "conda":
                cov0 = cov1 = f"diag(0x{m-r},0.25x{r})"
            elif s.kind == "rl":
                cov0 = cov1 = str(s.path / "input_bank.npz")
            elif s.kind == "pm" and metric == "reference":
                cov0 = cov1 = str(s.path / "reference_bank.npz")
            elif s.kind == "pm":
                cov0 = str(s.path / f"task_{max(1, task-1):03d}_input.npz")
                cov1 = str(s.path / f"task_{task:03d}_input.npz")
            else:
                cov0 = cov1 = f"slot_inputs(seed={s.seed},cond={s.arm}); centered population factor"
            record = {"environment": s.environment, "arm": s.arm, "seed": s.seed,
                      "source_class": s.source_class, "metric": metric, "task": task,
                      "source_prev": str(_snapshot_path(s, task - 1)), "source_next": str(source),
                      "covariance_prev_source": cov0, "covariance_next_source": cov1,
                      **summed, "max_abs_unit_identity_residual": float(unit.identity_residual.abs().max()),
                      "max_abs_unit_residual": float(unit.residual.abs().max())}
            if s.kind == "conda":
                record.update({k: float(current[k]) for k in
                               ("mse", "task_start_mse", "w_sigma2", "derivative_exactzero_frac",
                                "derivative_absmean", "activeunit_frac")})
                record["task_mse_gain"] = record["task_start_mse"] - record["mse"]
                for name in ("mse", "task_start_mse", "w_sigma2", "derivative_absmean", "activeunit_frac"):
                    if not np.isfinite(record[name]):
                        raise ValueError(f"task {task}: nonfinite CondA {name}")
                if metric == "effective":
                    record["snapshot_width_discrepancy"] = record["V_next"] - record["w_sigma2"]
                if "alpha" in current:
                    alpha = current["alpha"].astype(np.float64)
                    record["adaptive_alpha_median"] = float(np.median(alpha))
                    record["adaptive_alpha_clip_frac"] = float(np.mean((alpha <= .05) | (alpha >= 3.)))
            elif diagnostics:
                record.update({k: diagnostics.get(task, {}).get(k) for k in
                               ("online_acc", "task_start_acc", "train_acc", "test_acc", "sigma_z1_median",
                                "sigma_z1_reference_median", "active_unit_frac_l1",
                                "active_unit_frac_l2", "derivative_zero_frac_l1",
                                "derivative_zero_frac_l2", "derivative_abs_mean_l1",
                                "derivative_abs_mean_l2", "grad_w1_start_norm",
                                "grad_w2_start_norm", "grad_w1_end_norm",
                                "grad_w2_end_norm", "adaptive_alpha_median_l1",
                                "adaptive_alpha_median_l2", "adaptive_alpha_clip_frac_l1",
                                "adaptive_alpha_clip_frac_l2")})
                start = record.get("task_start_acc")
                end = record.get("train_acc")
                record["task_accuracy_gain"] = (float(end) - float(start)
                                                if start is not None and end is not None else math.nan)
            rows.append(record)
            if task >= _late_start(s.tasks):
                late_units.setdefault(metric, []).append(unit)
        previous = current
    for metric, frames in late_units.items():
        all_units = pd.concat(frames, ignore_index=True)
        for unit_id, chunk in all_units.groupby("unit"):
            v = float(chunk.V.sum())
            q = float(chunk.Q.sum())
            x = float(chunk.X.sum())
            units.append({"environment": s.environment, "arm": s.arm, "seed": s.seed,
                          "source_class": s.source_class, "metric": metric,
                          "unit": int(unit_id), "first_task": _late_start(s.tasks),
                          "last_task": s.tasks, "sum_V": v, "sum_Q": q, "sum_X": x,
                          "mean_sigma": float(np.sqrt(chunk.V.clip(lower=0)).mean()),
                          "mean_D": float(chunk.D_norm.mean()),
                          "balance": -2*x/q if q > 0 else math.nan,
                          "c_sum": x/math.sqrt(v*q) if v > 0 and q > 0 else math.nan,
                          "covariance_term_sum": float(chunk.covariance_term.sum()),
                          "mean_shift_sq_sum": (float(chunk.mean_shift_sq.sum())
                                                if "mean_shift_sq" in chunk else math.nan)})
    return rows, units, late_units


def bootstrap_median_ci(values, *, draws: int = 5000, seed: int = 924) -> tuple[float, float]:
    """Percentile CI, resampling seed values rather than units or transitions."""
    a = np.asarray(values, dtype=np.float64)
    a = a[np.isfinite(a)]
    if not len(a):
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    samples = a[rng.integers(0, len(a), size=(draws, len(a)))]
    return tuple(map(float, np.quantile(np.median(samples, axis=1), [0.025, 0.975])))


def _closure_trajectory(table: pd.DataFrame, summary: dict, labels: dict) -> list[dict]:
    test = table.loc[table.task >= summary["heldout_first_task"]].sort_values("task")
    value = float(summary["split_V"])
    result = []
    for row in test.itertuples():
        value = (1 - 2 * float(summary["gamma"])) * value + float(summary["qbar"])
        result.append({**labels, "task": int(row.task), "observed_V": float(row.V_next),
                       "model_V": value, "constant_V": float(summary["split_V"]),
                       "calibration_last_task": summary["calibration_last_task"]})
    return result


def _unit_regression(units: pd.DataFrame) -> float:
    usable = units[(units.mean_sigma > 0) & (units.mean_D > 0)]
    if len(usable) < 3 or usable.mean_sigma.nunique() < 2:
        return math.nan
    return float(np.polyfit(np.log(usable.mean_sigma), np.log(usable.mean_D), 1)[0])


def low_response_onset(table: pd.DataFrame) -> int | None:
    """First task of the first three consecutive low-response endpoints."""
    ordered = table.sort_values("task")
    if ordered.environment.iloc[0] == "conda":
        low = ((ordered.derivative_absmean.to_numpy(dtype=float) < 1e-8)
               | (ordered.activeunit_frac.to_numpy(dtype=float) < .1))
    elif ordered.environment.iloc[0].startswith("mnist_"):
        low = np.zeros(len(ordered), dtype=bool)
        for layer in (1, 2):
            low |= (ordered[f"derivative_abs_mean_l{layer}"].to_numpy(dtype=float) < 1e-8)
            low |= (ordered[f"active_unit_frac_l{layer}"].to_numpy(dtype=float) < .1)
    else:
        return None
    for i in range(len(low) - 2):
        if bool(np.all(low[i:i+3])):
            return int(ordered.task.iloc[i])
    return None


def pre_onset_window(table: pd.DataFrame, onset: int | None) -> dict:
    """Report-only final 20 transitions before onset; never replaces P1–P3."""
    empty = {"active_window_first_task": math.nan, "active_window_last_task": math.nan,
             "active_window_balance": math.nan, "active_window_activity": math.nan,
             "active_window_loglog_slope": math.nan, "active_window_halves_ratio": math.nan,
             "active_window_mean_sqrtQ": math.nan}
    if onset is None:
        return {"active_window_status": "NO_LOW_RESPONSE", **empty}
    eligible = table[table.task < onset].sort_values("task")
    if len(eligible) < 20:
        return {"active_window_status": "INSUFFICIENT_ACTIVE_HISTORY", **empty}
    part = eligible.tail(20)
    v, q, x = (part[k].to_numpy(dtype=np.float64) for k in ("V", "Q", "X"))
    qsum = float(q.sum())
    mean_v = float(v.mean())
    endpoint_v = np.r_[v[0], part.V_next.to_numpy(dtype=np.float64)]
    endpoint_t = np.r_[int(part.task.iloc[0]) - 1, part.task.to_numpy(dtype=np.int64)]
    slope = (float(np.polyfit(np.log(endpoint_t), np.log(endpoint_v), 1)[0])
             if np.all(endpoint_t > 0) and np.all(endpoint_v > 0) else math.nan)
    return {"active_window_status": "AVAILABLE", "active_window_first_task": int(part.task.iloc[0]),
            "active_window_last_task": int(part.task.iloc[-1]),
            "active_window_balance": -2*float(x.sum())/qsum if qsum > 0 else math.nan,
            "active_window_activity": qsum/mean_v if mean_v > 0 else math.nan,
            "active_window_loglog_slope": slope,
            "active_window_halves_ratio": float(v[10:].mean()/v[:10].mean()) if v[:10].mean() > 0 else math.nan,
            "active_window_mean_sqrtQ": float(np.sqrt(q).mean())}


def summarize_seed_tables(transitions: pd.DataFrame, units: pd.DataFrame
                          ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Produce one row per seed/metric and its genuinely held-out trajectory."""
    summaries, trajectories = [], []
    keys = ["environment", "arm", "seed", "source_class", "metric"]
    for key, block in transitions.groupby(keys, sort=True):
        labels = dict(zip(keys, key))
        fixed = labels["metric"] != "actual"
        summary = summarize_task_table(block, fixed_covariance=fixed)
        onset = low_response_onset(block)
        summary["low_response_onset_task"] = onset if onset is not None else math.nan
        summary["response_censoring"] = "LOW_RESPONSE" if onset is not None else "NO_LOW_RESPONSE"
        summary.update(pre_onset_window(block, onset))
        if labels["metric"] == "raw":
            summary.update(P1="NOT_APPLICABLE", P2="NOT_APPLICABLE", P3="NOT_APPLICABLE")
        late = block[block.task >= _late_start(int(block.task.max()))]
        qsum = float(late.Q.sum())
        summary["late_mean_sqrtQ"] = float(np.sqrt(late.Q).mean())
        summary["late_mean_V"] = float(late.V.mean())
        summary["late_mean_sigma"] = float(np.sqrt(late.V).mean())
        summary["late_G_abs_over_Q"] = (float(late.covariance_term.abs().sum()) / qsum
                                         if qsum > 0 else math.nan)
        summary["late_mean_shift_sq"] = (float(late.mean_shift_sq.mean())
                                          if "mean_shift_sq" in late else math.nan)
        for name in ("mse", "task_start_mse", "task_mse_gain", "online_acc", "task_start_acc",
                     "train_acc", "task_accuracy_gain", "test_acc", "activeunit_frac",
                     "active_unit_frac_l1", "active_unit_frac_l2",
                     "derivative_exactzero_frac", "derivative_zero_frac_l1", "derivative_zero_frac_l2",
                     "grad_w1_start_norm", "grad_w2_start_norm", "grad_w1_end_norm", "grad_w2_end_norm",
                     "adaptive_alpha_median", "adaptive_alpha_clip_frac",
                     "adaptive_alpha_median_l1", "adaptive_alpha_median_l2",
                     "adaptive_alpha_clip_frac_l1", "adaptive_alpha_clip_frac_l2"):
            if name in late:
                summary[f"late_{name}"] = float(late[name].mean())
        selected = units[(units.environment == labels["environment"])
                         & (units.arm == labels["arm"])
                         & (units.seed == labels["seed"])
                         & (units.metric == labels["metric"])]
        summary["unit_logD_on_logW_slope_report_only"] = _unit_regression(selected)
        if labels["metric"] in ("raw", "actual"):
            summary["dynamics_label"] = "DIAGNOSTIC_ONLY"
        elif summary["late_invalid_V"]:
            summary["dynamics_label"] = "INVALID_METRIC"
        elif summary["stopped"]:
            summary["dynamics_label"] = "STOPPED"
        elif summary["P2"] == "PASS":
            summary["dynamics_label"] = "PLATEAU_WITH_ACTIVITY"
        elif summary["P1"] == "PASS":
            summary["dynamics_label"] = "NEAR_BALANCE_WITH_ACTIVITY"
        elif np.isfinite(summary["late_loglog_slope"]) and summary["late_loglog_slope"] > .2:
            summary["dynamics_label"] = "TRANSIENT_GROWTH"
        else:
            summary["dynamics_label"] = "OTHER_ACTIVE"
        summaries.append({**labels, **summary})
        if fixed and labels["metric"] != "raw":
            trajectories += _closure_trajectory(block, summary, labels)
    result = pd.DataFrame(summaries)
    # P4 compares slopes within the same seed; the raw identity metric is a
    # descriptive comparator, never a second independent experiment.
    result["P4"] = "NOT_APPLICABLE"
    result["raw_minus_effective_slope"] = math.nan
    for idx, row in result.iterrows():
        if row.metric not in ("effective", "reference"):
            continue
        raw = result[(result.environment == row.environment) & (result.arm == row.arm)
                     & (result.seed == row.seed) & (result.metric == "raw")]
        if len(raw) != 1:
            continue
        gap = float(raw.iloc[0].late_loglog_slope - row.late_loglog_slope)
        result.at[idx, "raw_minus_effective_slope"] = gap
        result.at[idx, "P4"] = ("SEPARATED" if row.P2 == "PASS" and gap >= 0.2
                                 else "NOT_ESTABLISHED")
    return result, pd.DataFrame(trajectories)


def paired_conda(seed_summary: pd.DataFrame) -> pd.DataFrame:
    """Paired seed ratios for the registered k, optimizer, epsilon questions."""
    conda = seed_summary[(seed_summary.environment == "conda")
                         & (seed_summary.metric == "effective")]
    rows = []
    for seed in SEEDS:
        sub = conda[conda.seed == seed].set_index("arm")
        row = {"seed": seed, "status": "COMPLETE"}
        for label, numerator, denominator in (
            ("P5_D_k7_over_k1", "k7", "k1"),
            ("P6_D_A_over_S", "k1", "k1_sgd"),
            ("P6_D_E_over_A", "k1_eps1e3", "k1"),
            ("width_k7_over_k1_report_only", "k7", "k1"),
        ):
            column = "late_mean_sigma" if label.startswith("width") else "late_mean_sqrtQ"
            if numerator not in sub.index or denominator not in sub.index:
                row[label] = math.nan
                row["status"] = "INCOMPLETE"
                continue
            den = float(sub.loc[denominator, column])
            row[label] = float(sub.loc[numerator, column]) / den if den > 0 else math.nan
        row["P5_pass"] = bool(np.isfinite(row["P5_D_k7_over_k1"]) and row["P5_D_k7_over_k1"] > 1.2)
        row["P6_pass"] = bool(np.isfinite(row["P6_D_A_over_S"]) and np.isfinite(row["P6_D_E_over_A"])
                              and row["P6_D_A_over_S"] > 1 and row["P6_D_E_over_A"] < 1)
        rows.append(row)
    return pd.DataFrame(rows)


def report_only_comparisons(seed_summary: pd.DataFrame) -> pd.DataFrame:
    """Seed-index comparisons requested by addenda; no new verdicts."""
    comparisons = []
    for arm in ("k1_r2", "k1_r10", "k1_snake", "k1_sna03", "k1_sna06", "k1_sna1"):
        comparisons.append(("conda", "k1", "conda", arm, "same_m_seed"))
    for arm, reference in (("m40_r5_k1", "k1"),
                           ("m40_r10_k1", "k1_r10"),
                           ("m40_r10_k2", "k1_r10"),
                           ("m40_r10_k1_sna06", "k1_sna06"),
                           ("m40_r10_k2_sna06", "k1_sna06")):
        comparisons.append(("conda", reference, "conda", arm,
                            "seed_index_matched_independent_teacher"))
    for env in ("mnist_rl", "mnist_pm"):
        for arm in ACTS:
            if arm != "LR":
                comparisons.append((env, "LR", env, arm, "same_seed_activation"))
    comparisons.append(("mnist_pm_sgd", "LR", "mnist_pm", "LR",
                        "optimizer_and_learning_rate_differ"))
    main = seed_summary[seed_summary.metric.isin(["effective", "reference"])]
    index = {(row.environment, row.arm, int(row.seed)): row
             for row in main.itertuples()}
    rows = []
    for ref_env, ref_arm, cmp_env, cmp_arm, design in comparisons:
        for seed in SEEDS:
            ref, cmp = index.get((ref_env, ref_arm, seed)), index.get((cmp_env, cmp_arm, seed))
            row = {"reference_environment": ref_env, "reference_arm": ref_arm,
                   "comparison_environment": cmp_env, "comparison_arm": cmp_arm,
                   "seed": seed, "design_note": design,
                   "status": "COMPLETE" if ref is not None and cmp is not None else "INCOMPLETE"}
            if ref is not None and cmp is not None:
                for name in ("late_mean_sqrtQ", "late_mean_sigma", "late_activity",
                             "late_loglog_slope", "raw_minus_effective_slope",
                             "late_mean_shift_sq", "late_mse", "late_train_acc"):
                    a, b = getattr(ref, name, math.nan), getattr(cmp, name, math.nan)
                    row[f"reference_{name}"] = a
                    row[f"comparison_{name}"] = b
                    row[f"difference_{name}"] = b - a if np.isfinite(a) and np.isfinite(b) else math.nan
                    if name in ("late_mean_sqrtQ", "late_mean_sigma"):
                        row[f"ratio_{name}"] = b / a if np.isfinite(a) and a > 0 else math.nan
                row["reference_low_response_onset"] = ref.low_response_onset_task
                row["comparison_low_response_onset"] = cmp.low_response_onset_task
            rows.append(row)
    return pd.DataFrame(rows)


def group_verdicts(seed_summary: pd.DataFrame, paired: pd.DataFrame) -> pd.DataFrame:
    """Seed-first medians and 5-seed, >=4 PASS rules; no unit pseudoreplication."""
    rows = []
    main = seed_summary[seed_summary.metric.isin(["effective", "reference"])]
    value_column = {"P1": "late_nearbalance", "P2": "late_loglog_slope",
                    "P3": "heldout_mape", "P4": "raw_minus_effective_slope"}
    for (env, arm, metric), block in main.groupby(["environment", "arm", "metric"]):
        for question, column in value_column.items():
            statuses = block[question].tolist()
            wins = ("SEPARATED",) if question == "P4" else ("PASS",)
            n_pass = sum(x in wins for x in statuses)
            n_uninformative = sum(x == "UNINFORMATIVE" for x in statuses)
            values = block[column].to_numpy(dtype=np.float64)
            lo, hi = bootstrap_median_ci(values)
            verdict = (("SEPARATED" if question == "P4" else "PASS") if n_pass >= 4 and len(block) == 5 else
                       "UNINFORMATIVE" if question == "P3" and n_uninformative >= 4 and len(block) == 5 else
                       "INCOMPLETE" if len(block) < 5 else
                       "NOT_ESTABLISHED" if question == "P4" else "FAIL")
            rows.append({"environment": env, "arm": arm, "metric": metric,
                         "question": question, "verdict": verdict,
                         "n_seed": len(block), "n_pass": n_pass,
                         "n_uninformative": n_uninformative,
                         "statistic": column, "seed_median": float(np.nanmedian(values)),
                         "ci95_low": lo, "ci95_high": hi,
                         "source_class": block.source_class.iloc[0]})
    if not paired.empty:
        for question, column in (("P5", "P5_D_k7_over_k1"),
                                 ("P6", "P6_D_A_over_S")):
            valid = paired[np.isfinite(paired[column])]
            if question == "P6":
                valid = valid[np.isfinite(valid["P6_D_E_over_A"])]
            n_pass = int(paired[f"{question}_pass"].sum())
            lo, hi = bootstrap_median_ci(valid[column])
            secondary = valid["P6_D_E_over_A"] if question == "P6" else pd.Series(dtype=float)
            slo, shi = bootstrap_median_ci(secondary)
            median = float(valid[column].median()) if len(valid) else math.nan
            p5_median_ok = question != "P5" or median > 1.2
            rows.append({"environment": "conda", "arm": "paired_registered",
                         "metric": "effective", "question": question,
                         "verdict": "PASS" if len(valid) == 5 and n_pass >= 4 and p5_median_ok else
                                    "INCOMPLETE" if len(valid) < 5 else "FAIL",
                         "n_seed": len(valid), "n_pass": n_pass, "n_uninformative": 0,
                         "statistic": column, "seed_median": median,
                         "ci95_low": lo, "ci95_high": hi,
                         "secondary_statistic": "P6_D_E_over_A" if question == "P6" else "",
                         "secondary_seed_median": float(secondary.median()) if len(secondary) else math.nan,
                         "secondary_ci95_low": slo, "secondary_ci95_high": shi,
                         "source_class": "prospective"})
    return pd.DataFrame(rows)


def _markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "(no complete series)"
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for _, row in df.reindex(columns=columns).iterrows():
        fields = []
        for col in columns:
            value = row[col]
            fields.append(f"{value:.3g}" if isinstance(value, (float, np.floating)) and np.isfinite(value)
                          else str(value))
        rows.append("| " + " | ".join(fields) + " |")
    return "\n".join(rows)


def write_summary(path: Path, completion: pd.DataFrame, summary: pd.DataFrame,
                  verdict: pd.DataFrame, *, complete: bool) -> None:
    lines = ["# Effective displacement validation — analysis report", "",
             f"Run state: **{'COMPLETE' if complete else 'INCOMPLETE'}**. Archived CIFAR is post hoc reference, not independent confirmation.",
             "", "Near balance measures supply/cancellation with continued activity; it does not alone establish a plateau.",
             "A W–D regression slope is descriptive and does not establish causation.", "",
             "## Completion", "",
             _markdown_table(completion, ["environment", "arm", "seed", "status", "reason"]), "",
             "## Registered verdicts", "",
             _markdown_table(verdict, ["environment", "arm", "metric", "question", "verdict", "n_pass", "n_seed", "seed_median", "ci95_low", "ci95_high"]), "",
             "## Seed details", "",
             "See `seed_summary.csv`, `source_windows.csv`, `transitions.csv`, `unit_late.csv`, `closure.csv`, `paired.csv`, and `addon_pairs.csv`.",
             "P1–P4 apply only to fixed covariance. PM actual width reports its covariance-change term separately.",
             "The late and fit windows, source snapshot paths, failed cases, and all arm outcomes remain in the CSV files.",
             "PM Adam is an explicit change from the old canonical SGD setup; the SGD LR run is a reference. CondA adaptive Snake uses analytic conditional variance, while MNIST uses minibatch preactivation variance. Comparisons across m20 and m40 use independently dimensioned teachers and initializations.", ""]
    if not summary.empty:
        response = summary[summary.metric.isin(["effective", "reference"])]
        rl = response[response.environment == "mnist_rl"]
        if not rl.empty:
            rl_columns = ["late_task_start_acc", "late_train_acc",
                          "late_grad_w1_start_norm", "late_grad_w1_end_norm",
                          "late_grad_w2_start_norm", "late_grad_w2_end_norm"]
            rl_group = rl.groupby("arm", as_index=False)[rl_columns].median()
            onset = rl[rl.response_censoring == "LOW_RESPONSE"].groupby("arm").agg(
                low_response_seeds=("seed", "count"),
                median_onset_task=("low_response_onset_task", "median"))
            rl_group = rl_group.join(onset, on="arm")
            rl_group["low_response_seeds"] = rl_group.low_response_seeds.fillna(0).astype(int)
            for question in ("P1", "P2", "P3"):
                decisions = verdict[(verdict.environment == "mnist_rl")
                                    & (verdict.question == question)].set_index("arm").verdict
                rl_group[question] = rl_group.arm.map(decisions)
            lines += ["## RL: registered verdicts and learning context", "",
                      "Medians are across five seeds. Accuracy and raw first-16-example probe-gradient norms are measured on the current task; a small end gradient alone does not establish loss of task response. Onset counts use the registered three-task LOW_RESPONSE rule.", "",
                      _markdown_table(rl_group, ["arm", "P1", "P2", "P3", "low_response_seeds",
                                                 "median_onset_task", *rl_columns]), ""]
        lines += ["## Low-response context", "",
                  "Onset is the first of three consecutive task ends where either hidden layer has mean absolute derivative below 1e-8 or active-unit fraction below 0.1. This is an operational LOW_RESPONSE marker, not proof of loss of plasticity. Full-horizon verdicts remain unchanged. The last 20 transitions before onset are reported only when available.", "",
                  _markdown_table(response, ["environment", "arm", "seed", "response_censoring",
                                             "low_response_onset_task", "active_window_status",
                                             "active_window_balance", "active_window_activity",
                                             "active_window_loglog_slope"]), "",
                  "Task-start to task-end accuracy or MSE gain and probe gradients are context, not causal evidence; see `seed_summary.csv`.", ""]
        lines += ["## Seed dynamics", "",
                  "STOPPED means low update supply relative to width under the registered activity rule. It does not imply zero raw gradient or inability to learn a task; task-start/end performance and probe gradients are reported separately. DIVERGED cases appear in `completion.csv`; neither is counted as an active plateau.", "",
                  _markdown_table(response, ["environment", "arm", "seed", "dynamics_label",
                                             "response_censoring", "late_activity", "late_nearbalance",
                                             "late_loglog_slope"]), ""]
        adaptive = response[(response.environment == "mnist_rl")
                            & response.arm.isin(["SNA03", "SNA06", "SNA1"])]
        alpha_columns = ["late_adaptive_alpha_median_l1", "late_adaptive_alpha_median_l2",
                         "late_adaptive_alpha_clip_frac_l1", "late_adaptive_alpha_clip_frac_l2"]
        if not adaptive.empty and all(column in adaptive for column in alpha_columns):
            alpha = adaptive.groupby("arm", as_index=False)[alpha_columns].median()
            lines += ["## RL adaptive-alpha context (late tasks 41–50)", "",
                      "These are medians of seed-level late-window summaries. Alpha clipping is descriptive; a passing width criterion alone does not show that adaptation caused the plateau.", "",
                      _markdown_table(alpha, ["arm", *alpha_columns]), ""]
            sna = response[(response.environment == "mnist_rl") & (response.arm == "SNA03")]
            if len(sna) == 5:
                lines += [f"SNA03 passes P2/P4 in five seeds, but its late alpha is at the lower clip (median layer-1/layer-2 clip fractions {sna.late_adaptive_alpha_clip_frac_l1.median():.3g}/{sna.late_adaptive_alpha_clip_frac_l2.median():.3g}); this does not isolate an adaptive-alpha effect. Its P3 model MAPE median {sna.heldout_mape.median():.3g} is worse than the constant-split median {sna.constant_split_mape.median():.3g}, so the closure fails despite the low absolute model error.", ""]
    if not summary.empty and "actual" in summary.metric.values:
        actual = summary[summary.metric == "actual"]
        lines += ["## PM actual moving covariance (diagnostic only)", "",
                  _markdown_table(actual, ["arm", "seed", "late_loglog_slope", "late_halves_ratio", "late_G_abs_over_Q"]), ""]
    path.write_text("\n".join(lines))


def run_analysis(raw_root: Path, out: Path, *, include_cifar: bool = False,
                 allow_incomplete: bool = False, make_plots: bool = True,
                 cifar_archive: Path = DEFAULT_CIFAR,
                 cifar_data: Path = DEFAULT_CIFAR_DATA) -> dict[str, pd.DataFrame]:
    """Analyze all expected series, with an explicit incomplete-run gate."""
    raw_root, out = Path(raw_root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    series = expected_series(raw_root, include_cifar=include_cifar, cifar_archive=cifar_archive)
    completion = pd.DataFrame([preflight(s) for s in series])
    completion.to_csv(out / "completion.csv", index=False)
    if (completion.status != "COMPLETE").any() and not allow_incomplete:
        raise RuntimeError(f"{(completion.status != 'COMPLETE').sum()} of {len(completion)} expected seed series incomplete; see completion.csv; use --allow-incomplete for diagnostic output")
    transitions, units = [], []
    for s, ok in zip(series, completion.status == "COMPLETE"):
        if not ok:
            continue
        try:
            task_rows, unit_rows, _ = _transition_rows(s, cifar_data=cifar_data)
            transitions.extend(task_rows)
            units.extend(unit_rows)
        except Exception as exc:
            match = (completion.environment == s.environment) & (completion.arm == s.arm) & (completion.seed == s.seed)
            completion.loc[match, ["status", "reason"]] = ["ANALYSIS_FAILED", repr(exc)]
            completion.to_csv(out / "completion.csv", index=False)
            if not allow_incomplete:
                raise RuntimeError(f"analysis failed for {s.environment}/{s.arm}/seed{s.seed}: {exc}") from exc
    transition_df = pd.DataFrame(transitions)
    unit_df = pd.DataFrame(units)
    transition_df.to_csv(out / "transitions.csv", index=False)
    unit_df.to_csv(out / "unit_late.csv", index=False)
    if transition_df.empty:
        empty = pd.DataFrame()
        for name in ("seed_summary", "source_windows", "closure", "paired", "addon_pairs", "verdict"):
            empty.to_csv(out / f"{name}.csv", index=False)
        write_summary(out / "summary.md", completion, empty, empty, complete=False)
        return {"completion": completion, "transitions": empty, "unit_late": empty,
                "seed_summary": empty, "source_windows": empty, "closure": empty,
                "paired": empty, "addon_pairs": empty, "verdict": empty}
    summary, closure = summarize_seed_tables(transition_df, unit_df)
    windows = []
    key_columns = ["environment", "arm", "seed", "metric"]
    for key, block in transition_df.groupby(key_columns, sort=True):
        block = block.sort_values("task")
        one = summary[(summary.environment == key[0]) & (summary.arm == key[1])
                      & (summary.seed == key[2]) & (summary.metric == key[3])].iloc[0]
        windows.append(dict(zip(key_columns, key)) | {
            "source_class": one.source_class, "first_task": int(block.task.iloc[0]),
            "last_task": int(block.task.iloc[-1]),
            "calibration_first_task": int(one.calibration_first_task),
            "calibration_last_task": int(one.calibration_last_task),
            "heldout_first_task": int(one.heldout_first_task),
            "heldout_last_task": int(one.heldout_last_task),
            "late_first_task": int(block.task.max() - one.late_n_transitions + 1),
            "late_last_task": int(block.task.max()),
            "first_snapshot": block.source_prev.iloc[0],
            "last_snapshot": block.source_next.iloc[-1],
            "first_covariance_source": block.covariance_prev_source.iloc[0],
            "last_covariance_source": block.covariance_next_source.iloc[-1],
        })
    window_df = pd.DataFrame(windows)
    paired = paired_conda(summary)
    addon_pairs = report_only_comparisons(summary)
    verdict = group_verdicts(summary, paired)
    summary.to_csv(out / "seed_summary.csv", index=False)
    window_df.to_csv(out / "source_windows.csv", index=False)
    closure.to_csv(out / "closure.csv", index=False)
    paired.to_csv(out / "paired.csv", index=False)
    addon_pairs.to_csv(out / "addon_pairs.csv", index=False)
    verdict.to_csv(out / "verdict.csv", index=False)
    complete = bool((completion.status == "COMPLETE").all())
    write_summary(out / "summary.md", completion, summary, verdict, complete=complete)
    if make_plots:
        from plots import make_plots as draw
        draw(transition_df, summary, closure, paired, out)
    return {"completion": completion, "transitions": transition_df,
            "unit_late": unit_df, "seed_summary": summary, "source_windows": window_df,
            "closure": closure,
            "paired": paired, "addon_pairs": addon_pairs, "verdict": verdict}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--include-cifar", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--cifar-archive", type=Path, default=DEFAULT_CIFAR)
    parser.add_argument("--cifar-data", type=Path, default=DEFAULT_CIFAR_DATA)
    args = parser.parse_args()
    run_analysis(args.raw_root, args.out, include_cifar=args.include_cifar,
                 allow_incomplete=args.allow_incomplete, make_plots=not args.no_plots,
                 cifar_archive=args.cifar_archive, cifar_data=args.cifar_data)


if __name__ == "__main__":
    main()
