"""Preregistered Phase 0b: longer horizon and depth/capacity separation.

Stages are deliberately explicit so the calibrated width can be frozen before
the expensive full run::

    OMP_NUM_THREADS=1 python -m src.mlp2_phase0b --calibrate
    OMP_NUM_THREADS=1 python -m src.mlp2_phase0b --preflight
    OMP_NUM_THREADS=1 python -m src.mlp2_phase0b

Raw 1k-step trajectories are restart checkpoints for the analysis as well as
the audit log.  A complete arm is therefore reused automatically on rerun.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from .common import ROOT, load_config, pick_device
from .mlp2_phase0 import (
    LOG_LAYER_KEYS,
    PhaseRecorder,
    S0Replay,
    _sha_file,
    exact_layer_record,
    identity_sanity_pass,
    require_omp,
    setup_arm,
    spearman,
    train_arm,
    write_arm_logs,
    write_csv,
)
from .ratchet_log import record_steps as legacy_record_steps
from .ratchet_log import full_support_ro, teacher_f64


SMOKE_STEPS = 30_000
ARM_ORDER = ("L1w100", "L1wide", "L2")


def _base_cfg(cfg: dict) -> dict:
    """Adapt Phase 0b's section name to the unchanged simulation helpers."""
    out = copy.deepcopy(cfg)
    out["phase0"] = copy.deepcopy(cfg["phase0b"])
    return out


def validate_config(cfg: dict, *, full: bool) -> None:
    C, A, P = cfg["common"], cfg["condA"], cfg["phase0b"]
    names = [a["name"] for a in cfg["arms"]]
    if names != list(ARM_ORDER):
        raise ValueError(f"arms must be {ARM_ORDER}, got {names}")
    hidden = {a["name"]: a["hidden"] for a in cfg["arms"]}
    if hidden["L1w100"] != [100] or hidden["L2"] != [100, 100]:
        raise ValueError("fixed arms differ from the preregistration")
    if int(A["m"]) != 20 or int(A["f"]) != 15:
        raise ValueError("Phase 0b requires condA m=20, f=15")
    if list(A["T_values"]) != [10_000] or list(A["encodings"]) != ["std"]:
        raise ValueError("Phase 0b requires T=10000 and std encoding")
    if int(P["exact_support"]) != 2 ** (int(A["m"]) - int(A["f"])):
        raise ValueError("phase0b.exact_support does not match full support")
    if str(P["ci_method"]) != "studentized":
        raise ValueError("Phase 0b CI must be studentized")
    if full:
        if int(C["total_steps"]) != 5_000_000 or list(C["seeds"]) != list(range(10)):
            raise ValueError("full Phase 0b requires 5M steps and seeds 0..9")
        if str(C["device"]) != "cpu":
            raise ValueError("full Phase 0b is CPU-only")
        wide = hidden["L1wide"]
        if len(wide) != 1 or not isinstance(wide[0], int):
            raise ValueError("run --calibrate before the full run")
        if wide[0] not in [int(v) for v in cfg["calibration"]["widths"]]:
            raise ValueError("calibrated width is outside the registered grid")


def _arm(cfg: dict, name: str) -> dict:
    return next(a for a in cfg["arms"] if a["name"] == name)


def _window_indices(steps: np.ndarray, period: int, window: list[int]) -> np.ndarray:
    task = steps // period
    return np.flatnonzero((steps > 0) & (steps % period == 0)
                          & (task >= int(window[0])) & (task <= int(window[1])))


class UnfitRecorder:
    """Calibration recorder that intentionally exposes only unfitness."""

    def __init__(self, steps: list[int], sigma_tol: float):
        self.steps = np.asarray(steps, dtype=np.int64)
        self.index = {int(v): i for i, v in enumerate(self.steps)}
        self.unfit: np.ndarray | None = None
        self.sigma_tol = float(sigma_tol)

    def __call__(self, st: dict, step: int) -> None:
        i = self.index.get(int(step))
        if i is None:
            return
        # Do not even compute the forbidden calibration quantities (dead, M,
        # or effective rank).  This is the exact-support unfitness calculation
        # only, written separately so accidental inspection is impossible.
        with torch.no_grad():
            X = full_support_ro(st["env"]).double()
            y = teacher_f64(st["teacher"], X)
            cur = X
            for W, b in zip(st["net"].Ws, st["net"].bs):
                cur = torch.relu(torch.einsum("rhd,prd->prh", W.double(), cur)
                                 + b.double())
            yhat = ((cur * st["net"].v.double()).sum(dim=-1)
                    + st["net"].c.double())
            values = (((yhat - y).var(dim=0, unbiased=False)
                       / y.var(dim=0, unbiased=False)).cpu().numpy())
        if self.unfit is None:
            self.unfit = np.empty((len(self.steps), len(values)), dtype=np.float64)
        self.unfit[i] = values


def _phase0_l2_target(path: Path, early: list[int]) -> tuple[float, list[float]]:
    per_seed: dict[int, list[float]] = {}
    seen: set[tuple[int, int]] = set()
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            if row["arm"] != "L2" or int(row["task_end"]) != 1:
                continue
            task, seed = int(row["task"]), int(row["seed"])
            key = (seed, task)
            if not (int(early[0]) <= task <= int(early[1])) or key in seen:
                continue
            seen.add(key)
            per_seed.setdefault(seed, []).append(float(row["unfit"]))
    if sorted(per_seed) != list(range(10)) or any(len(v) != 10 for v in per_seed.values()):
        raise ValueError("committed Phase 0 target is incomplete")
    seed_values = [float(np.mean(per_seed[s])) for s in range(10)]
    return float(np.median(seed_values)), seed_values


def calibrate(cfg_path: Path, cfg: dict, device: str, outdir: Path) -> dict:
    require_omp(cfg)
    K, P = cfg["calibration"], cfg["phase0b"]
    target_path = Path(ROOT) / "results/mlp2_phase0_0829/layer_stats.csv"
    target, target_seeds = _phase0_l2_target(target_path, list(K["window_tasks"]))
    floor = float(P["unfit_floor"])
    widths = [int(v) for v in K["widths"]]
    rows, medians = [], {}
    period, total = int(P["task_period"]), int(K["steps"])
    steps = [period * t for t in range(int(K["window_tasks"][0]),
                                      int(K["window_tasks"][1]) + 1)]
    for width in widths:
        c = _base_cfg(cfg)
        c["common"]["seeds"] = [int(v) for v in K["seeds"]]
        arm = {"name": "L1wide", "hidden": [width]}
        st = setup_arm(c, arm, device)
        rec = UnfitRecorder(steps, float(P["sigma_degenerate_tol"]))
        print(f"[calibration W={width}] seeds={c['common']['seeds']} steps={total:,}",
              flush=True)
        elapsed = train_arm(st, rec, steps, total, outdir, [])
        if rec.unfit is None:
            raise RuntimeError("calibration recorder was not called")
        seed_values = rec.unfit.mean(axis=0)
        median = float(np.median(seed_values))
        medians[width] = median
        # Calibration compares the observed early values themselves.  The
        # 1e-12 floor is registered for the Phase 0b G0/log endpoint only.
        logdiff = abs(math.log10(median) - math.log10(target))
        projected = elapsed * 5_000_000 / total
        row = dict(width=width, seed0_early_unfit=seed_values[0],
                   seed1_early_unfit=seed_values[1], seed2_early_unfit=seed_values[2],
                   median_early_unfit=median, phase0_l2_target=target,
                   phase0_l2_target_seed_values=json.dumps(target_seeds),
                   abs_log10_difference=logdiff,
                   within_tolerance=int(logdiff <= float(K["target_log10_tolerance"])),
                   selected=0, elapsed_sec=elapsed,
                   steps_per_sec=total / elapsed,
                   projected_5m_same_3seeds_sec=projected,
                   projected_5m_10seeds_linear_sec=projected * 10 / len(K["seeds"]))
        rows.append(row)
        print(f"[calibration W={width}] median={median:.6g} log10_diff={logdiff:.3f} "
              f"elapsed={elapsed:.1f}s", flush=True)
    eligible = [w for w in widths if abs(math.log10(medians[w])
                                         - math.log10(target))
                <= float(K["target_log10_tolerance"])]
    selected = min(eligible) if eligible else int(K["fallback_width"])
    reached = bool(eligible)
    for row in rows:
        row["selected"] = int(row["width"] == selected)
        row["calibration_reached"] = int(reached)
    outdir.mkdir(parents=True, exist_ok=True)
    write_csv(outdir / "calibration.csv", rows)

    text = cfg_path.read_text(encoding="utf-8")
    if "[CALIBRATED]" in text:
        cfg_path.write_text(text.replace("[CALIBRATED]", f"[{selected}]", 1), encoding="utf-8")
    else:
        current = _arm(cfg, "L1wide")["hidden"]
        if current != [selected]:
            raise RuntimeError(f"config already frozen at {current}, calibration selected {selected}")
    result = dict(selected_width=selected, calibration_reached=reached,
                  target=target, widths=medians)
    (outdir / "calibration.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"CALIBRATION DONE: W*={selected} reached={reached}", flush=True)
    return result


def _find_worktree_file(relative: Path) -> Path:
    local = Path(ROOT) / relative
    if local.exists():
        return local
    try:
        raw = subprocess.check_output(
            ["git", "worktree", "list", "--porcelain"], cwd=ROOT, text=True)
        roots = [Path(line.split(" ", 1)[1]) for line in raw.splitlines()
                 if line.startswith("worktree ")]
    except (OSError, subprocess.CalledProcessError):
        roots = []
    for root in roots:
        candidate = root / relative
        if candidate.exists():
            return candidate
    raise FileNotFoundError(relative)


def _s0_preflight(device: str, outdir: Path) -> dict:
    cfg0_path = Path(ROOT) / "configs/mlp2_phase0_0829.yaml"
    cfg0 = load_config(str(cfg0_path))
    require_omp(cfg0)
    total, period = int(cfg0["common"]["total_steps"]), int(cfg0["phase0"]["task_period"])
    st = setup_arm(cfg0, _arm(cfg0, "L1"), device)
    reference = Path(ROOT) / cfg0["sanity"]["s0_bit_equality_ref"]
    steps = legacy_record_steps(total, period, 100, 1000)
    replay = S0Replay(reference, steps, list(cfg0["common"]["seeds"]), None)
    elapsed = train_arm(st, replay, steps, total, outdir, [])
    checkpoint = _find_worktree_file(
        Path(cfg0["sanity"]["s0_bit_equality_ref"]).parent.parent
        / "ckpts" / f"A_w100_step{total}.pt")
    result = replay.finish(st, checkpoint)
    result["elapsed_sec"] = elapsed
    return result


def _ci_components(values: np.ndarray, draws: np.ndarray, statistic: str,
                   se_tol: float, degenerate_frac_max: float,
                   width_ratio_max: float) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) != draws.shape[1] or not np.isfinite(values).all():
        raise ValueError("guarded CI requires one finite scalar per seed")
    stat = np.mean if statistic == "mean" else np.median
    point = float(stat(values))
    n = len(values)

    def jk_se(matrix: np.ndarray) -> np.ndarray:
        if statistic == "mean":
            return matrix.std(axis=1, ddof=1) / math.sqrt(n)
        jk = np.stack([np.median(np.delete(matrix, i, axis=1), axis=1)
                       for i in range(n)], axis=1)
        return np.sqrt((n - 1) / n
                       * np.square(jk - jk.mean(axis=1, keepdims=True)).sum(axis=1))

    original = values[None, :]
    se0 = float(jk_se(original)[0])
    samples = values[draws]
    boot = stat(samples, axis=1)
    se = jk_se(samples)
    degenerate_fraction = float(np.mean(~np.isfinite(se) | (se < se_tol)))
    good = np.isfinite(boot) & np.isfinite(se) & (se >= se_tol)
    if se0 >= se_tol and good.any():
        pivots = (boot[good] - point) / se[good]
        qlo, qhi = np.quantile(pivots, [0.025, 0.975])
        student_lo, student_hi = point - qhi * se0, point - qlo * se0
    else:
        student_lo = student_hi = point
    pct_lo, pct_hi = np.quantile(boot[np.isfinite(boot)], [0.025, 0.975])
    width = float(student_hi - student_lo)
    ratio = width / max(abs(point), se_tol)
    degenerate = bool(degenerate_fraction > degenerate_frac_max
                      or ratio > width_ratio_max)
    nonzero = values[values != 0]
    positives = int((nonzero > 0).sum())
    if len(nonzero):
        tail = sum(math.comb(len(nonzero), i) for i in range(0, min(positives,
                   len(nonzero) - positives) + 1)) / (2 ** len(nonzero))
        sign_p = min(1.0, 2.0 * tail)
    else:
        sign_p = 1.0
    return dict(point=point, studentized_ci_lo=float(student_lo),
                studentized_ci_hi=float(student_hi), percentile_ci_lo=float(pct_lo),
                percentile_ci_hi=float(pct_hi), se0=se0,
                degenerate_se_fraction=degenerate_fraction,
                studentized_width_ratio=float(ratio), ci_degenerate=int(degenerate),
                boot_ok=int(good.sum()), sign_test_p=float(sign_p),
                n_seed=n, statistic=statistic)


def guarded_ci(values: np.ndarray, draws: np.ndarray, cfg: dict,
               statistic: str = "median") -> dict:
    P = cfg["phase0b"]
    return _ci_components(values, draws, statistic,
                          float(P["degenerate_se_tol"]),
                          float(P["degenerate_frac_max"]),
                          float(P["degenerate_width_ratio_max"]))


def _s5_selftest(cfg: dict) -> dict:
    n = len(cfg["common"]["seeds"])
    rng = np.random.default_rng(int(cfg["phase0b"]["bootstrap_seed"]))
    draws = rng.integers(0, n, size=(int(cfg["phase0b"]["bootstrap_B"]), n))
    result = guarded_ci(np.zeros(n), draws, cfg)
    return dict(pass_=bool(result["ci_degenerate"]), result=result)


def _s6_floor_check(cfg: dict, device: str, outdir: Path) -> dict:
    c = _base_cfg(cfg)
    c["common"]["seeds"] = [0, 1, 2]
    total = int(cfg["sanity"]["s6_floor_check_steps"])
    st = setup_arm(c, _arm(cfg, "L2"), device)
    elapsed = train_arm(st, lambda _st, _step: None, [], total, outdir, [])
    rec, sanity = exact_layer_record(st, float(cfg["phase0b"]["sigma_degenerate_tol"]))
    unfit = rec["run"]["unfit"].detach().cpu().numpy()
    # A conservative first-order bound for the 32-term float64 exact average
    # and its variance ratio.  This checks the registered floor against the
    # arithmetic actually used by exact_layer_record, not against float32 SGD.
    bound = float(np.max(np.finfo(np.float64).eps * 32 * (1.0 + np.abs(unfit))))
    floor = float(cfg["phase0b"]["unfit_floor"])
    return dict(pass_=bool(sanity["run_finite"] and floor > bound), floor=floor,
                numerical_precision_bound=bound, floor_over_bound=floor / bound,
                steps=total, seeds=[0, 1, 2], elapsed_sec=elapsed)


def preflight(cfg: dict, device: str, outdir: Path) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    omp = require_omp(cfg)
    print("[S0] replaying legacy L=1 trajectory", flush=True)
    s0 = _s0_preflight(device, outdir / "s0")
    print(f"[S0] {'PASS' if s0['pass_'] else 'FAIL'} ({s0['elapsed_sec']:.1f}s)", flush=True)
    s5 = _s5_selftest(cfg)
    print(f"[S5] {'PASS' if s5['pass_'] else 'FAIL'}", flush=True)
    print("[S6] L2 seeds 0..2 x 200k floor check", flush=True)
    s6 = _s6_floor_check(cfg, device, outdir / "s6")
    print(f"[S6] {'PASS' if s6['pass_'] else 'FAIL'} ({s6['elapsed_sec']:.1f}s)", flush=True)
    result = dict(pass_=bool(omp["pass_"] and s0["pass_"] and s5["pass_"] and s6["pass_"]),
                  S0=s0, S3=omp, S5=s5, S6=s6)
    (outdir / "preflight.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    if not result["pass_"]:
        raise RuntimeError(f"preflight failed: {result}")
    return result


def _complete_arm_logs(outdir: Path, arm: str, seeds: list[int], total: int,
                       every: int = 1000) -> bool:
    expected = np.arange(0, total + 1, every, dtype=np.int64)
    if expected[-1] != total:
        expected = np.append(expected, total)
    for seed in seeds:
        path = outdir / "logs" / f"{arm}_seed{seed}.npz"
        if not path.exists():
            return False
        try:
            with np.load(path, allow_pickle=False) as z:
                if (not np.array_equal(z["step"], expected)
                        or int(z["seed"]) != seed or str(z["arm"]) != arm):
                    return False
        except (OSError, ValueError, KeyError):
            return False
    return True


def _task_rows_from_logs(cfg: dict, outdir: Path) -> list[dict]:
    period = int(cfg["phase0b"]["task_period"])
    rows = []
    for arm in ARM_ORDER:
        hidden = [int(v) for v in _arm(cfg, arm)["hidden"]]
        for seed in cfg["common"]["seeds"]:
            path = outdir / "logs" / f"{arm}_seed{seed}.npz"
            with np.load(path, allow_pickle=False) as z:
                idx = np.flatnonzero((z["step"] > 0) & (z["step"] % period == 0))
                for i in idx:
                    for li, width in enumerate(hidden, start=1):
                        row = dict(arm=arm, run_id=str(z["run_id"]), seed=int(seed),
                                   step=int(z["step"][i]), task=int(z["step"][i] // period),
                                   task_end=1, layer=li)
                        for key in LOG_LAYER_KEYS:
                            value = z[f"layer{li}_{key}"][i]
                            row[key] = (int(value) if key in ("n_na", "strict_dead", "alive")
                                        else float(value))
                        row["strict_dead_frac"] = row["strict_dead"] / width
                        for key in ("signal_var", "residual_var", "unfit", "eval_loss_exact"):
                            row[key] = float(z[key][i])
                        rows.append(row)
    return rows


def _arm_arrays(cfg: dict, outdir: Path, arm: str) -> dict:
    seeds = list(cfg["common"]["seeds"])
    period = int(cfg["phase0b"]["task_period"])
    result: dict[str, object] = {"layers": []}
    per_seed = []
    for seed in seeds:
        with np.load(outdir / "logs" / f"{arm}_seed{seed}.npz", allow_pickle=False) as z:
            idx = np.flatnonzero((z["step"] > 0) & (z["step"] % period == 0))
            item = {"steps": z["step"][idx].copy(), "unfit": z["unfit"][idx].copy(),
                    "layers": [{k: z[f"layer{li}_{k}"][idx].copy()
                                for k in LOG_LAYER_KEYS}
                               for li in range(1, len(_arm(cfg, arm)["hidden"]) + 1)]}
            per_seed.append(item)
    result["steps"] = per_seed[0]["steps"]
    result["unfit"] = np.stack([v["unfit"] for v in per_seed], axis=1)
    for li in range(len(per_seed[0]["layers"])):
        result["layers"].append({k: np.stack([v["layers"][li][k] for v in per_seed], axis=1)
                                 for k in LOG_LAYER_KEYS})
    return result


def _cluster_regression(rows_by_arm: dict[str, dict], x_name: str, floor: float,
                        rng: np.random.Generator, B: int, se_tol: float,
                        degenerate_frac_max: float,
                        width_ratio_max: float) -> list[dict]:
    clusters: list[tuple[str, int, np.ndarray, np.ndarray]] = []
    for ai, arm in enumerate(ARM_ORDER):
        data = rows_by_arm[arm]
        yall = np.log10(np.maximum(data["unfit"], floor))
        layer = data["layers"][-1]
        xall = np.asarray(layer[x_name], dtype=np.float64)
        if x_name == "eff_rank":
            xall = np.log10(np.maximum(xall, floor))
        for seed in range(yall.shape[1]):
            y = yall[:, seed]
            x = xall[:, seed]
            X = np.column_stack((np.ones(len(y)), x,
                                 np.full(len(y), ai == 1, dtype=float),
                                 np.full(len(y), ai == 2, dtype=float)))
            clusters.append((arm, seed, X.T @ X, X.T @ y))

    def fit(indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
        A = sum((clusters[i][2] for i in indices), np.zeros((4, 4)))
        b = sum((clusters[i][3] for i in indices), np.zeros(4))
        beta = np.linalg.solve(A, b)
        bread = np.linalg.inv(A)
        scores = [clusters[i][3] - clusters[i][2] @ beta for i in indices]
        meat = sum((np.outer(s, s) for s in scores), np.zeros((4, 4)))
        cov = bread @ meat @ bread * len(indices) / (len(indices) - 1)
        return beta, np.sqrt(np.maximum(np.diag(cov), 0))

    original = list(range(len(clusters)))
    beta0, se0 = fit(original)
    arm_indices = [[i for i, c in enumerate(clusters) if c[0] == arm] for arm in ARM_ORDER]
    boot_beta = np.empty((B, 4)); boot_se = np.empty((B, 4))
    for b in range(B):
        # Stratified seed-cluster resampling preserves the arm factor without
        # pairing seed labels across arms.
        ids = []
        for choices in arm_indices:
            ids.extend(np.asarray(choices)[rng.integers(0, len(choices), len(choices))].tolist())
        boot_beta[b], boot_se[b] = fit(ids)

    rows = []
    for j, contrast in ((2, "L1wide-L1w100"), (3, "L2-L1w100")):
        good = np.isfinite(boot_beta[:, j]) & np.isfinite(boot_se[:, j]) & (boot_se[:, j] >= se_tol)
        deg_frac = float(1 - good.mean())
        if se0[j] >= se_tol and good.any():
            piv = (boot_beta[good, j] - beta0[j]) / boot_se[good, j]
            qlo, qhi = np.quantile(piv, [0.025, 0.975])
            slo, shi = beta0[j] - qhi * se0[j], beta0[j] - qlo * se0[j]
        else:
            slo = shi = beta0[j]
        plo, phi = np.quantile(boot_beta[:, j], [0.025, 0.975])
        ratio = float((shi - slo) / max(abs(beta0[j]), se_tol))
        deg = bool(deg_frac > degenerate_frac_max or ratio > width_ratio_max)
        rows.append(dict(metric="arm_effect", arm=contrast, x=x_name,
                         point=float(beta0[j]), studentized_ci_lo=float(slo),
                         studentized_ci_hi=float(shi), percentile_ci_lo=float(plo),
                         percentile_ci_hi=float(phi), ci_degenerate=int(deg),
                         degenerate_se_fraction=deg_frac,
                         studentized_width_ratio=ratio, boot_ok=int(good.sum()),
                         sign_test_p="", sign_test_note="not defined for unpaired arm contrast",
                         h0_rejected=int(not deg and not (slo <= 0 <= shi))))
    return rows


def analyze(cfg: dict, outdir: Path, sanity: dict, elapsed: dict[str, float]) -> dict:
    P = cfg["phase0b"]
    B, seed0 = int(P["bootstrap_B"]), int(P["bootstrap_seed"])
    nseed = len(cfg["common"]["seeds"])
    rng = np.random.default_rng(seed0)
    draws = rng.integers(0, nseed, size=(B, nseed))
    floor = float(P["unfit_floor"])
    data = {arm: _arm_arrays(cfg, outdir, arm) for arm in ARM_ORDER}
    verdict_rows, details = [], {"arms": {}, "wall_trends": [], "wall_levels": [],
                                "arm_effects": []}

    for arm in ARM_ORDER:
        a = data[arm]
        steps = np.asarray(a["steps"])
        early_i = _window_indices(steps, int(P["task_period"]), list(P["early_tasks"]))
        late_i = _window_indices(steps, int(P["task_period"]), list(P["late_tasks"]))
        early_seed = np.asarray(a["unfit"])[early_i].mean(axis=0)
        late_seed = np.maximum(np.asarray(a["unfit"])[late_i].mean(axis=0), floor)
        early_floor = np.maximum(early_seed, floor)
        dlog = np.log10(late_seed) - np.log10(early_floor)
        interval = guarded_ci(dlog, draws, cfg, "median")
        early_median = float(np.median(early_seed))
        regime = "IN_SCOPE" if early_median >= float(P["regime_threshold"]) else "INTERPOLATING"
        onset = late_seed >= float(P["onset_threshold"])
        n_onset = int(onset.sum())
        verdict = ("LOP_PRESENT" if n_onset >= int(P["onset_present_min"])
                   else "LOP_ABSENT" if n_onset == 0 else "LOP_PARTIAL")
        row = dict(metric="G0", arm=arm, x="", regime=regime, verdict=verdict,
                   n_onset=n_onset, onset_seeds=json.dumps(np.flatnonzero(onset).tolist()),
                   late_unfit_values=json.dumps(late_seed.tolist()),
                   late_unfit_median=float(np.median(late_seed)),
                   late_unfit_min=float(late_seed.min()), late_unfit_max=float(late_seed.max()),
                   early_unfit_seed_median=early_median, dU_log=interval["point"],
                   **interval)
        verdict_rows.append(row)
        details["arms"][arm] = dict(regime=regime, verdict=verdict,
                                    early_seed=early_seed.tolist(), late_seed=late_seed.tolist(),
                                    dU_log_seed=dlog.tolist(), dU_log_ci=interval)

        task = steps / int(P["task_period"])
        for li, layer in enumerate(a["layers"], start=1):
            D = -np.asarray(layer["median_M"], dtype=np.float64)
            seed_rho = np.array([spearman(task, D[:, s]) for s in range(nseed)])
            trend_ci = guarded_ci(seed_rho, draws, cfg, "median")
            trend = dict(metric="wall_trend", arm=arm, layer=li,
                         seed_values=seed_rho.tolist(), increase=int(
                             not trend_ci["ci_degenerate"] and trend_ci["studentized_ci_lo"] > 0),
                         **trend_ci)
            details["wall_trends"].append(trend)
            verdict_rows.append(dict(regime="", verdict="REPORT_ONLY", n_onset="",
                                     x=f"layer{li}", **trend))
            for window_name, indices in (("early", early_i), ("late", late_i)):
                seed_values = D[indices].mean(axis=0)
                level = dict(arm=arm, layer=li, window=window_name,
                             seed_values=seed_values.tolist(),
                             median=float(np.median(seed_values)),
                             q25=float(np.quantile(seed_values, .25)),
                             q75=float(np.quantile(seed_values, .75)))
                details["wall_levels"].append(level)

        if len(a["layers"]) == 2:
            for window_name, indices in (("early", early_i), ("late", late_i)):
                d1 = -np.asarray(a["layers"][0]["median_M"])[indices].mean(axis=0)
                d2 = -np.asarray(a["layers"][1]["median_M"])[indices].mean(axis=0)
                comp = guarded_ci(d2 - d1, draws, cfg, "median")
                verdict_rows.append(dict(metric="wall_layer_contrast", arm=arm, x=window_name,
                                         regime="", verdict="REPORT_ONLY", n_onset="",
                                         **comp))

    for x_name in ("eff_rank", "alive"):
        effects = _cluster_regression(data, x_name, floor, rng, B,
                                      float(P["degenerate_se_tol"]),
                                      float(P["degenerate_frac_max"]),
                                      float(P["degenerate_width_ratio_max"]))
        details["arm_effects"].extend(effects)
        for effect in effects:
            verdict_rows.append(dict(regime="", verdict="H0_REJECTED" if effect["h0_rejected"]
                                     else "H0_NOT_REJECTED", n_onset="", **effect))

    # Stable union schema for heterogeneous registered endpoints.
    fields = []
    for row in verdict_rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    normalized = [{key: row.get(key, "") for key in fields} for row in verdict_rows]
    write_csv(outdir / "verdict.csv", normalized)
    write_csv(outdir / "layer_stats.csv", _task_rows_from_logs(cfg, outdir))

    g0s = [r for r in verdict_rows if r["metric"] == "G0"]
    lines = ["# mlp2_phase0b_0829 summary", "", "## G-pre / G0", "",
             "| arm | regime | verdict | n_onset | early seed median | late seed median [min, max] | dU_log median | CI degenerate |",
             "|---|---|---|---:|---:|---:|---:|---:|"]
    for r in g0s:
        lines.append(f"| {r['arm']} | {r['regime']} | {r['verdict']} | {r['n_onset']} | "
                     f"{r['early_unfit_seed_median']:.6g} | {r['late_unfit_median']:.6g} "
                     f"[{r['late_unfit_min']:.6g}, {r['late_unfit_max']:.6g}] | "
                     f"{r['dU_log']:.6g} | {r['ci_degenerate']} |")
    lines += ["", "全 seed の late U_k は verdict.csv に保存。腕間は非ペアとして扱った。", "",
              "## Arm effects", "",
              "回帰は全タスク末尾点を用い、腕ごとに独立な seed-cluster bootstrap を行った。",
              "", "| x | contrast | coefficient | studentized 95% CI | degenerate | decision |",
              "|---|---|---:|---:|---:|---|"]
    for r in details["arm_effects"]:
        decision = "H0_REJECTED" if r["h0_rejected"] else "H0_NOT_REJECTED"
        lines.append(f"| {r['x']} | {r['arm']} | {r['point']:.6g} | "
                     f"[{r['studentized_ci_lo']:.6g}, {r['studentized_ci_hi']:.6g}] | "
                     f"{r['ci_degenerate']} | {decision} |")
    lines += ["", "## Wall depth D = -median(M)", "",
              "| arm | layer | median seed Spearman(task,D) | studentized 95% CI | degenerate | increase |",
              "|---|---:|---:|---:|---:|---:|"]
    for r in details["wall_trends"]:
        lines.append(f"| {r['arm']} | {r['layer']} | {r['point']:.6g} | "
                     f"[{r['studentized_ci_lo']:.6g}, {r['studentized_ci_hi']:.6g}] | "
                     f"{r['ci_degenerate']} | {r['increase']} |")
    lines += ["", "## Sanity", "",
              f"- S0: **{'PASS' if sanity['S0']['pass_'] else 'FAIL'}**",
              f"- S1/S2: **{'PASS' if sanity['S1_S2_all_pass'] else 'FAIL'}**",
              f"- S3: **{'PASS' if sanity['S3']['pass_'] else 'FAIL'}**",
              f"- S5: **{'PASS' if sanity['S5']['pass_'] else 'FAIL'}**",
              f"- S6: **{'PASS' if sanity['S6']['pass_'] else 'FAIL'}**", ""]
    (outdir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    details["elapsed_sec"] = elapsed
    return details


def _provenance(cfg_path: Path, cfg: dict, outdir: Path, sanity: dict,
                analysis: dict, elapsed: dict, started: float) -> dict:
    files = [outdir / name for name in ("calibration.csv", "verdict.csv", "summary.md",
                                        "layer_stats.csv", "config_used.yaml")]
    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                           text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--short"], cwd=ROOT,
                                        text=True).splitlines()
    except (OSError, subprocess.CalledProcessError):
        git_hash, dirty = None, []
    spec = Path(ROOT) / cfg["spec"]
    return dict(experiment="mlp2_phase0b_0829",
                created=time.strftime("%Y-%m-%d %H:%M:%S %z"), command=sys.argv,
                elapsed_sec=round(time.time() - started, 3), arm_elapsed_sec=elapsed,
                cwd=os.getcwd(), python=sys.version, platform=platform.platform(),
                torch=torch.__version__, numpy=np.__version__, device=cfg["common"]["device"],
                git_hash=git_hash, git_dirty=dirty, config=str(cfg_path),
                config_sha256=_sha_file(cfg_path), spec=str(spec), spec_sha256=_sha_file(spec),
                sanity=sanity, analysis=analysis,
                output_sha256={p.name: _sha_file(p) for p in files if p.exists()})


def run_full(cfg_path: Path, cfg: dict, device: str, outdir: Path, *, smoke: bool) -> dict:
    started = time.time()
    outdir.mkdir(parents=True, exist_ok=True)
    require_omp(cfg)
    preflight_path = Path(ROOT) / "results/_preflight_mlp2_phase0b_0829/preflight.json"
    if smoke:
        preflight_result = dict(pass_=True, S0={"pass_": True}, S3=require_omp(cfg),
                                S5=_s5_selftest(cfg), S6={"pass_": True})
    else:
        if not preflight_path.exists():
            raise FileNotFoundError("run --preflight before the full run")
        preflight_result = json.loads(preflight_path.read_text(encoding="utf-8"))
        if not preflight_result.get("pass_"):
            raise RuntimeError("saved preflight did not pass")
    with (outdir / "config_used.yaml").open("w") as fh:
        yaml.safe_dump(cfg, fh, allow_unicode=True, sort_keys=False)

    C, P = cfg["common"], cfg["phase0b"]
    total = SMOKE_STEPS if smoke else int(C["total_steps"])
    seeds = [0] if smoke else [int(v) for v in C["seeds"]]
    probe_steps = list(range(0, total + 1, int(C["lop_every"])))
    if probe_steps[-1] != total:
        probe_steps.append(total)
    elapsed, identity = {}, {}
    for arm in ARM_ORDER:
        if _complete_arm_logs(outdir, arm, seeds, total, int(C["lop_every"])):
            elapsed[arm] = 0.0
            identity[arm] = {"pass_": True, "resumed_from_complete_logs": True}
            print(f"[{arm}] complete logs found; resuming after arm", flush=True)
            continue
        c = _base_cfg(cfg)
        c["common"]["seeds"] = seeds
        arm_cfg = _arm(cfg, arm)
        print(f"[{arm}] hidden={arm_cfg['hidden']} seeds={seeds} steps={total:,}", flush=True)
        st = setup_arm(c, arm_cfg, device)
        _, before = exact_layer_record(st, float(P["sigma_degenerate_tol"]))
        before["pass_"] = identity_sanity_pass(before, float(cfg["sanity"]["s1_identity_tol"]))
        if not before["pass_"]:
            raise RuntimeError(f"{arm} preflight identity failed")
        rec = PhaseRecorder(probe_steps, st, float(P["sigma_degenerate_tol"]),
                            float(cfg["sanity"]["s1_identity_tol"]))
        elapsed[arm] = train_arm(st, rec, probe_steps, total, outdir,
                                 [] if smoke else list(C.get("checkpoints", [])))
        arm_sanity = rec.sanity()
        identity[arm] = arm_sanity
        if not arm_sanity["pass_"]:
            raise RuntimeError(f"{arm} S1/S2 failed: {arm_sanity}")
        write_arm_logs(outdir, arm, st, rec)
        print(f"[{arm}] complete in {elapsed[arm]:.1f}s", flush=True)
        del rec, st

    sanity = dict(S0=preflight_result["S0"], S3=preflight_result["S3"],
                  S5=preflight_result["S5"], S6=preflight_result["S6"],
                  S1_S2=identity,
                  S1_S2_all_pass=bool(all(v["pass_"] for v in identity.values())))
    if smoke:
        result = dict(smoke=True, elapsed_sec=elapsed)
        (outdir / "smoke_sanity.json").write_text(
            json.dumps(dict(pass_=sanity["S1_S2_all_pass"], sanity=sanity,
                            analysis=result), indent=2, default=str), encoding="utf-8")
    else:
        result = analyze(cfg, outdir, sanity, elapsed)
        prov = _provenance(cfg_path, cfg, outdir, sanity, result, elapsed, started)
        (outdir / "provenance.json").write_text(
            json.dumps(prov, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"ALL DONE -> {outdir}", flush=True)
    return dict(sanity=sanity, analysis=result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mlp2_phase0b_0829.yaml")
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--outdir")
    args = parser.parse_args()
    if sum((args.calibrate, args.preflight, args.smoke, args.analyze_only)) > 1:
        parser.error("stage flags are mutually exclusive")
    cfg_path = Path(args.config).resolve()
    cfg = load_config(str(cfg_path))
    device = pick_device(cfg)
    if device != "cpu":
        raise ValueError("Phase 0b is CPU-only")
    outdir = (Path(args.outdir).resolve() if args.outdir else
              Path(ROOT) / "results" / ("_preflight_mlp2_phase0b_0829" if args.preflight
                                          else "_smoke_mlp2_phase0b_0829" if args.smoke
                                          else "mlp2_phase0b_0829"))
    validate_config(cfg, full=not args.calibrate)
    if args.calibrate:
        calibrate(cfg_path, cfg, device, outdir)
    elif args.preflight:
        preflight(cfg, device, outdir)
    elif args.analyze_only:
        preflight_result = json.loads((Path(ROOT) / "results/_preflight_mlp2_phase0b_0829/preflight.json").read_text())
        sanity = dict(S0=preflight_result["S0"], S3=preflight_result["S3"],
                      S5=preflight_result["S5"], S6=preflight_result["S6"],
                      S1_S2={}, S1_S2_all_pass=True)
        analyze(cfg, outdir, sanity, {})
    else:
        run_full(cfg_path, cfg, device, outdir, smoke=args.smoke)


if __name__ == "__main__":
    main()
