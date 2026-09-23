#!/usr/bin/env python3
"""Read-only, seed-paired analysis of the preregistered image geometry run.

The first-layer covariance is represented by the centered 1200-image bank;
no d-by-d covariance is formed. All arithmetic after loading snapshots is
float64. Production analysis requires all 50 complete series.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DEFAULT_RAW = Path("/home/issan/Projects/obsidian-research-data/effdisp_imagegeom_0924/raw")
SEEDS = tuple(range(300, 305))
GEOMETRIES = {"rgb32": 3072, "dup64": 12288, "dup64_half": 12288,
              "gray32": 1024, "avg16": 768}
OPTIMIZERS = ("adam", "sgd")
TASKS = 50
N_IMAGES = 1200
BOOTSTRAP_DRAWS = 5000
BOOTSTRAP_SEED = 924
EARLY = (6, 15)
LATE = (41, 50)


def _prior_metrics():
    """Load the exact previous metrics implementation without a sys.path alias."""
    path = REPO / "analysis/effdisp_validation_0924/metrics.py"
    spec = importlib.util.spec_from_file_location("effdisp_validation_metrics_0924", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load prior metrics from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PRIOR = _prior_metrics()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def load_npz(path: Path, *keys: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in keys}


def bootstrap_median_ci(values, *, draws: int = BOOTSTRAP_DRAWS,
                        seed: int = BOOTSTRAP_SEED) -> tuple[float, float]:
    a = np.asarray(values, dtype=np.float64)
    a = a[np.isfinite(a)]
    if not len(a):
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    samples = a[rng.integers(0, len(a), size=(draws, len(a)))]
    return tuple(map(float, np.quantile(np.median(samples, axis=1), (.025, .975))))


def projection_terms(centered: np.ndarray, mean: np.ndarray,
                     w_prev: np.ndarray, b_prev: np.ndarray,
                     w_next: np.ndarray, b_next: np.ndarray,
                     top_left: np.ndarray | None = None,
                     prev_projection: np.ndarray | None = None
                     ) -> tuple[dict[str, float], np.ndarray]:
    """Population V/Q/X and mean channel via the same centered image rows.

    ``prev_projection`` is C Wprev.T / sqrt(N), cached across consecutive
    transitions. It avoids a second large matrix multiply per task.
    """
    c = np.asarray(centered, dtype=np.float64)
    mu = np.asarray(mean, dtype=np.float64)
    w0 = np.asarray(w_prev, dtype=np.float64)
    w1 = np.asarray(w_next, dtype=np.float64)
    b0 = np.asarray(b_prev, dtype=np.float64)
    b1 = np.asarray(b_next, dtype=np.float64)
    if c.ndim != 2 or w0.shape != w1.shape or w0.shape[1] != c.shape[1]:
        raise ValueError("centered bank and weight shapes disagree")
    if mu.shape != (c.shape[1],) or b0.shape != b1.shape or b0.shape != (w0.shape[0],):
        raise ValueError("mean or bias shape disagrees with weights")
    if not all(np.isfinite(a).all() for a in (c, mu, w0, w1, b0, b1)):
        raise ValueError("nonfinite bank or parameter")
    scale = 1 / math.sqrt(c.shape[0])
    p0 = (c @ w0.T) * scale if prev_projection is None else np.asarray(prev_projection, dtype=np.float64)
    if p0.shape != (c.shape[0], w0.shape[0]):
        raise ValueError("previous projection shape disagrees with weights")
    d = w1 - w0
    dp = (c @ d.T) * scale
    p1 = p0 + dp
    v, q, x, vn = (float(np.einsum("ij,ij->", a, b, dtype=np.float64))
                   for a, b in ((p0, p0), (dp, dp), (p0, dp), (p1, p1)))
    raw0 = float(np.einsum("ij,ij->", w0, w0, dtype=np.float64))
    raw1 = float(np.einsum("ij,ij->", w1, w1, dtype=np.float64))
    raw_d = float(np.einsum("ij,ij->", d, d, dtype=np.float64))
    raw_x = float(np.einsum("ij,ij->", w0, d, dtype=np.float64))
    mean_shift = d @ mu + (b1 - b0)
    mean_shift_sq = float(np.dot(mean_shift, mean_shift))
    top_q = math.nan
    if top_left is not None and q > 0:
        u = np.asarray(top_left, dtype=np.float64)
        if u.shape[0] != c.shape[0] or u.ndim != 2:
            raise ValueError("top-left singular vectors have wrong shape")
        top = u.T @ dp
        top_q = float(np.einsum("ij,ij->", top, top, dtype=np.float64) / q)
    result = {"V": v, "Q": q, "X": x, "V_next": vn,
              "delta_V": vn - v, "identity_residual": vn - v - q - 2*x,
              "R": math.sqrt(v), "R_next": math.sqrt(vn), "d_eff": math.sqrt(q),
              "c": x / math.sqrt(v*q) if v > 0 and q > 0 else math.nan,
              "rho": math.sqrt(q/v) if v > 0 else math.nan,
              "balance": -2*x/q if q > 0 else math.nan,
              "mean_shift_sq": mean_shift_sq,
              "raw_W_sq": raw0, "raw_W_next_sq": raw1, "raw_D_sq": raw_d,
              "raw_X": raw_x, "top10_Q_fraction": top_q}
    return result, p1


def covariance_spectrum(x: np.ndarray, *, reference: dict | None = None,
                        scale: float = 1.0) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """Float64 singular spectrum without a d-by-d covariance.

    For exact pixel duplication, ``reference`` contains the rgb32 singular
    values and left vectors; the singulars scale by 2 (dup64) or 1 (half).
    Direct SVD on the native bank sets the 1e-10 singular-value rank cutoff.
    """
    xx = np.asarray(x, dtype=np.float64)
    if xx.ndim != 2 or xx.shape[0] != N_IMAGES or not np.isfinite(xx).all():
        raise ValueError("bank must contain 1200 finite image rows")
    if reference is None:
        centered = xx - xx.mean(0)
        # economy SVD returns U and singular values; Vt is intentionally dropped.
        u, s, _ = np.linalg.svd(centered / math.sqrt(N_IMAGES), full_matrices=False)
    else:
        s = np.asarray(reference["singular_values"], dtype=np.float64) * scale
        u = np.asarray(reference["top_left"], dtype=np.float64)
    eigen = s*s
    rank = int(np.count_nonzero(s > s[0] * 1e-10)) if len(s) and s[0] > 0 else 0
    trace = float(eigen.sum(dtype=np.float64))
    trace2 = float(np.dot(eigen, eigen))
    stats = {"dimension": int(xx.shape[1]), "n_images": N_IMAGES,
             "numeric_rank": rank, "rank_cutoff_relative_singular": 1e-10,
             "cov_trace": trace, "cov_trace_sq": trace2,
             "cov_participation_ratio": trace*trace/trace2 if trace2 > 0 else math.nan,
             "top10_cov_trace_fraction": float(eigen[:10].sum()/trace) if trace > 0 else math.nan,
             "largest_cov_eigenvalue": float(eigen[0]) if len(eigen) else math.nan}
    return stats, s, u[:, :10]


def low_response_onset(table: pd.DataFrame) -> int | None:
    ordered = table.sort_values("task")
    low = np.zeros(len(ordered), dtype=bool)
    for layer in (1, 2):
        low |= ordered[f"mean_abs_dphi_l{layer}"].to_numpy(dtype=float) < 1e-8
        low |= ordered[f"activeunit_frac_l{layer}"].to_numpy(dtype=float) < .1
    for i in range(len(low) - 2):
        if bool(np.all(low[i:i+3])):
            return int(ordered.task.iloc[i])
    return None


def seed_window_medians(transitions: pd.DataFrame, completion: pd.DataFrame | None = None) -> pd.DataFrame:
    """A registered window is usable when its ten observed rows are finite."""
    rows = []
    blocks = {(g, o, int(s)): block for (g, o, s), block in
              transitions.groupby(["geometry", "optimizer", "seed"])}
    cases = ([(r.geometry, r.optimizer, int(r.seed), r.status)
              for r in completion.itertuples(index=False)] if completion is not None else
             [(g, o, s, "UNKNOWN") for g, o, s in blocks])
    columns = ("R", "d_eff", "c", "rho", "train_acc", "top10_Q_fraction",
               "mean_shift_sq", "raw_W_sq", "raw_D_sq")
    for geometry, optimizer, seed, series_status in cases:
        block = blocks.get((geometry, optimizer, seed))
        for label, (first, last) in (("early", EARLY), ("late", LATE)):
            part = (block[(block.task >= first) & (block.task <= last)]
                    if block is not None else pd.DataFrame())
            has_all_tasks = len(part) == last-first+1 and set(part.task) == set(range(first, last+1))
            finite = has_all_tasks and np.isfinite(part[list(columns)].to_numpy(dtype=np.float64)).all()
            if not has_all_tasks and series_status == "COMPLETE":
                raise ValueError(f"complete case missing {label} transitions for {geometry}/{optimizer}/{seed}")
            row = {"geometry": geometry, "optimizer": optimizer, "seed": int(seed),
                   "window": label, "first_task": first, "last_task": last,
                   "series_status": series_status, "available": bool(finite),
                   "unavailable_reason": "" if finite else "NONFINITE" if has_all_tasks else "MISSING_TASKS"}
            row.update({column: float(np.median(part[column].to_numpy(dtype=np.float64))) if finite else math.nan
                        for column in columns})
            rows.append(row)
    return pd.DataFrame(rows, columns=["geometry", "optimizer", "seed", "window",
                                      "first_task", "last_task", "series_status", "available",
                                      "unavailable_reason", *columns])


def paired_contrasts(windows: pd.DataFrame) -> pd.DataFrame:
    """All geometry/rgb32 and Adam/SGD comparisons, paired by seed."""
    lookup = {(r.geometry, r.optimizer, int(r.seed), r.window): r
              for r in windows.itertuples(index=False)}
    rows = []
    metrics = ("R", "d_eff", "c", "rho", "train_acc")
    for seed in SEEDS:
        for window in ("early", "late"):
            for optimizer in OPTIMIZERS:
                for geometry in GEOMETRIES:
                    if geometry == "rgb32":
                        continue
                    rows += _pair_rows(lookup, metrics, "geometry_over_rgb32", geometry,
                                       optimizer, "rgb32", optimizer, seed, window)
            for geometry in GEOMETRIES:
                rows += _pair_rows(lookup, metrics, "adam_over_sgd", geometry,
                                   "adam", geometry, "sgd", seed, window)
    return pd.DataFrame(rows)


def _pair_rows(lookup, metrics, kind, numerator_geometry, numerator_optimizer,
               denominator_geometry, denominator_optimizer, seed, window) -> list[dict]:
    num = lookup.get((numerator_geometry, numerator_optimizer, seed, window))
    den = lookup.get((denominator_geometry, denominator_optimizer, seed, window))
    num_status = getattr(num, "series_status", "UNKNOWN") if num is not None else "UNKNOWN"
    den_status = getattr(den, "series_status", "UNKNOWN") if den is not None else "UNKNOWN"
    valid = (num is not None and den is not None
             and bool(getattr(num, "available", True)) and bool(getattr(den, "available", True)))
    rows = []
    for metric in metrics:
        a = float(getattr(num, metric)) if valid else math.nan
        b = float(getattr(den, metric)) if valid else math.nan
        rows.append({"contrast": kind, "geometry": numerator_geometry,
                     "optimizer": numerator_optimizer, "reference_geometry": denominator_geometry,
                     "reference_optimizer": denominator_optimizer, "seed": seed,
                     "window": window, "metric": metric,
                     "availability": "AVAILABLE" if valid else "MISSING_OR_NONFINITE_WINDOW",
                     "numerator_series_status": num_status,
                     "reference_series_status": den_status,
                     "terminal_divergence": num_status == "DIVERGED" or den_status == "DIVERGED",
                     "value": a, "reference_value": b, "difference": a-b,
                     # Signed c differences are primary; no near-zero c ratios.
                     "ratio": a/b if metric != "c" and np.isfinite(b) and b > 0 else math.nan})
    return rows


def paired_groups(contrasts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["contrast", "geometry", "optimizer", "reference_geometry",
            "reference_optimizer", "window", "metric"]
    for key, block in contrasts.groupby(keys):
        for statistic in ("difference", "ratio"):
            values = block[statistic].to_numpy(dtype=np.float64)
            lo, hi = bootstrap_median_ci(values)
            finite = values[np.isfinite(values)]
            rows.append({**dict(zip(keys, key)), "statistic": statistic,
                         "n_seed": len(block), "n_valid": len(finite),
                         "seed_median": float(np.median(finite)) if len(finite) else math.nan,
                         "ci95_low": lo, "ci95_high": hi})
    return pd.DataFrame(rows)


def window_groups(windows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = ("R", "d_eff", "c", "rho", "train_acc", "top10_Q_fraction")
    for (geometry, optimizer, window), block in windows.groupby(["geometry", "optimizer", "window"]):
        for metric in metrics:
            values = block[metric].to_numpy(dtype=np.float64)
            lo, hi = bootstrap_median_ci(values)
            finite = values[np.isfinite(values)]
            rows.append({"geometry": geometry, "optimizer": optimizer, "window": window,
                         "metric": metric, "n_seed": len(block), "n_valid": len(finite),
                         "seed_median": float(np.median(finite)) if len(finite) else math.nan,
                         "ci95_low": lo, "ci95_high": hi})
    return pd.DataFrame(rows, columns=["geometry", "optimizer", "window", "metric",
                                      "n_seed", "n_valid", "seed_median", "ci95_low", "ci95_high"])


def preregistered_e_verdicts(contrasts: pd.DataFrame) -> pd.DataFrame:
    """E1–E5 use only prespecified windows and 4-of-5 paired seeds."""
    rows = []
    def ratios(geometry, optimizer, window, metric):
        part = contrasts[(contrasts.contrast == "geometry_over_rgb32")
                         & (contrasts.geometry == geometry) & (contrasts.optimizer == optimizer)
                         & (contrasts.window == window) & (contrasts.metric == metric)]
        return part.set_index("seed").reindex(SEEDS).ratio.to_numpy(dtype=float)
    e1_r = ratios("dup64_half", "sgd", "late", "R")
    e1_d = ratios("dup64_half", "sgd", "late", "d_eff")
    e1 = np.isfinite(e1_r) & np.isfinite(e1_d) & (e1_r >= .8) & (e1_r <= 1.2) & (e1_d >= .8) & (e1_d <= 1.2)
    rows.append(_e_row("E1", e1, "late halfdup/SGD R and d_eff ratios both in [0.8,1.2]"))
    e2 = ratios("dup64_half", "adam", "early", "d_eff")
    rows.append(_e_row("E2", np.isfinite(e2) & (e2 > 1.2), "early halfdup/Adam d_eff ratio >1.2"))
    e3a = ratios("dup64", "adam", "early", "d_eff")
    e3s = ratios("dup64", "sgd", "early", "d_eff")
    e3_adam_count = int(np.count_nonzero(np.isfinite(e3a) & (e3a > 1)))
    e3_sgd_count = int(np.count_nonzero(np.isfinite(e3s) & (e3s > 1)))
    rows.append({"question": "E3", "verdict": "PASS" if min(e3_adam_count, e3_sgd_count) >= 4 else "FAIL",
                 "n_pass": min(e3_adam_count, e3_sgd_count), "n_seed": len(SEEDS),
                 "adam_n_pass": e3_adam_count, "sgd_n_pass": e3_sgd_count,
                 "criterion": "early rawdup d_eff ratio >1, independently in 4/5 seeds per optimizer"})
    rows.append({"question": "E4", "verdict": "REPORT_ONLY", "n_pass": math.nan,
                 "n_seed": len(SEEDS), "criterion": "gray32/avg16 direction unspecified; rank and performance reported"})
    e5_all = True
    for optimizer in OPTIMIZERS:
        for geometry in GEOMETRIES:
            if geometry == "rgb32":
                continue
            part = contrasts[(contrasts.contrast == "geometry_over_rgb32")
                             & (contrasts.geometry == geometry) & (contrasts.optimizer == optimizer)
                             & (contrasts.window == "late") & (contrasts.metric == "c")]
            differences = part.set_index("seed").reindex(SEEDS).difference.to_numpy(dtype=float)
            count = int(np.count_nonzero(np.isfinite(differences) & (np.abs(differences) <= .05)))
            e5_all &= count >= 4
            rows.append({"question": "E5_component", "geometry": geometry, "optimizer": optimizer,
                         "verdict": "PASS" if count >= 4 else "FAIL", "n_pass": count,
                         "n_seed": len(SEEDS), "criterion": "late |c_geometry-c_rgb32| <=0.05"})
    rows.append({"question": "E5", "verdict": "PASS" if e5_all else "FAIL",
                 "n_pass": int(e5_all), "n_seed": len(SEEDS),
                 "criterion": "all four late c geometry comparisons pass in each optimizer"})
    return pd.DataFrame(rows)


def _e_row(question: str, passed: np.ndarray, criterion: str) -> dict:
    count = int(np.count_nonzero(passed))
    return {"question": question, "verdict": "PASS" if count >= 4 else "FAIL",
            "n_pass": count, "n_seed": len(SEEDS), "criterion": criterion}


def expected_series(raw: Path) -> list[tuple[str, str, int, Path]]:
    return [(geometry, optimizer, seed, raw / geometry / optimizer / f"seed{seed}")
            for geometry in GEOMETRIES for optimizer in OPTIMIZERS for seed in SEEDS]


def preflight(raw: Path) -> tuple[pd.DataFrame, dict]:
    """Require 50 terminal trajectories, including documented divergence."""
    meta_path = raw / "metadata.json"
    if not meta_path.is_file():
        raise FileNotFoundError(meta_path)
    meta = load_json(meta_path)
    problems = []
    done_path = raw / "DONE.json"
    done = None
    if not done_path.is_file():
        problems.append("runner DONE.json is absent")
    else:
        try:
            done = load_json(done_path)
            if not isinstance(done, dict):
                raise ValueError("DONE.json must be an object")
            if (done.get("expected_cases") != 50 or done.get("git_hash") != meta.get("git_hash")
                    or done.get("source_sha256") != meta.get("source_sha256")):
                problems.append("runner DONE.json does not match registered cases and source")
        except (OSError, ValueError) as exc:
            problems.append(f"runner DONE.json is unreadable: {exc}")
    if not meta.get("finished_at_utc"):
        problems.append("runner metadata does not have a finish timestamp")
    args = meta.get("args", {})
    if (meta.get("tasks") != TASKS or tuple(meta.get("seeds", [])) != SEEDS
            or meta.get("epochs") != 400 or meta.get("batch") != 16
            or meta.get("images_per_seed") != N_IMAGES
            or meta.get("steps_per_task") != 30000
            or args.get("tasks") != TASKS or tuple(args.get("seeds", [])) != SEEDS):
        problems.append("metadata tasks/seeds differ from preregistration")
    source_hash = meta.get("git_hash")
    source_sha = meta.get("source_sha256")
    if not isinstance(source_hash, str) or len(source_hash) != 40:
        problems.append("metadata missing full source git hash")
    if not isinstance(source_sha, str) or len(source_sha) != 64:
        problems.append("metadata missing source SHA256")
    spec = REPO / "specs/spec_effdisp_imagegeom_0924.md"
    expected_spec_sha = hashlib.sha256(spec.read_bytes()).hexdigest()
    if meta.get("spec_sha256") != expected_spec_sha:
        problems.append("metadata spec SHA256 differs from registered spec")
    source_file = REPO / "src/effdisp_imagegeom_0924.py"
    if source_file.is_file() and source_sha != hashlib.sha256(source_file.read_bytes()).hexdigest():
        problems.append("metadata source SHA256 differs from local training source")
    records = []
    failed_statuses = []
    for geometry, optimizer, seed, sd in expected_series(raw):
        basic = (sd / "status.json", raw / "banks" / geometry / f"seed{seed}.npz",
                 raw / "labels" / f"seed{seed}.npz")
        missing = [p for p in basic if not p.is_file()]
        state = "MISSING" if not sd.exists() else "INCOMPLETE"
        reason = f"{len(missing)} required files missing" if missing else ""
        last, first_nonfinite = -1, None
        if not missing:
            try:
                status = load_json(sd / "status.json")
                if not isinstance(status, dict):
                    raise ValueError("status.json must be an object")
                last = status.get("last_saved_task")
                first_nonfinite = status.get("first_nonfinite_task")
                identity_ok = all(status.get(k) == v for k, v in
                                  (("geometry", geometry), ("optimizer", optimizer), ("seed", seed)))
                complete = (status.get("status") == "COMPLETED" and last == TASKS
                            and first_nonfinite is None)
                diverged = (status.get("status") == "DIVERGED" and type(last) is int
                            and type(first_nonfinite) is int and 0 <= last < TASKS
                            and first_nonfinite == last + 1)
                if identity_ok and (complete or diverged):
                    missing += [sd / f"t{task:03d}.npz" for task in range(last + 1)
                                if not (sd / f"t{task:03d}.npz").is_file()]
                    extra = [sd / f"t{task:03d}.npz" for task in range(last + 1, TASKS + 1)
                             if (sd / f"t{task:03d}.npz").exists()]
                    if not missing and not extra:
                        state = "COMPLETE" if complete else "DIVERGED"
                        reason = "" if complete else f"first_nonfinite_task={first_nonfinite}"
                        if diverged:
                            failed_statuses.append(status)
                    else:
                        reason = f"{len(missing)} missing and {len(extra)} unexpected snapshots"
                else:
                    reason = f"nonterminal or inconsistent runner status={status.get('status')}"
            except (OSError, ValueError, TypeError) as exc:
                reason = f"bad status JSON: {exc}"
        records.append({"geometry": geometry, "optimizer": optimizer, "seed": seed,
                        "status": state, "last_saved_task": last,
                        "first_nonfinite_task": first_nonfinite,
                        "reason": reason, "missing_count": len(missing),
                        "first_missing": str(missing[0]) if missing else ""})
    completion = pd.DataFrame(records)
    if len(completion) != 50:
        problems.append(f"expected 50 series, found {len(completion)}")
    if not completion.status.isin(("COMPLETE", "DIVERGED")).all():
        problems.append(f"{int((~completion.status.isin(('COMPLETE', 'DIVERGED'))).sum())} series nonterminal")
    if done is not None:
        failed = done.get("failed_cases")
        valid_failed_list = isinstance(failed, list) and all(isinstance(item, dict) for item in failed)
        if (done.get("status") != ("FAILED" if failed_statuses else "SUCCESS")
                or done.get("completed_cases") != 50-len(failed_statuses)
                or not valid_failed_list
                or sorted(failed, key=lambda x: (x.get("geometry"), x.get("optimizer"), x.get("seed")))
                != sorted(failed_statuses, key=lambda x: (x.get("geometry"), x.get("optimizer"), x.get("seed")))):
            problems.append("runner DONE failed list/count does not exactly match terminal statuses")
    # Trainer metadata must identify all five geometries and both optimizers.
    if (set(meta.get("geometries", [])) != set(GEOMETRIES)
            or meta.get("geometry_dims") != GEOMETRIES):
        problems.append("metadata geometry names/dimensions differ from preregistration")
    if set(meta.get("optimizers", [])) != set(OPTIMIZERS):
        problems.append("metadata optimizer set differs from preregistration")
    settings = meta.get("optimizer_settings", {})
    if (settings.get("adam") != {"lr": 0.001, "betas": [0.9, 0.999], "eps": 1e-8}
            or settings.get("sgd") != {"lr": 0.01, "momentum": 0}):
        problems.append("metadata optimizer settings differ from preregistration")
    cases = meta.get("cases", [])
    expected_cases = {(g, o, s, (GEOMETRIES[g], 100, 100, 10))
                      for g in GEOMETRIES for o in OPTIMIZERS for s in SEEDS}
    actual_cases = {(item.get("geometry"), item.get("optimizer"), item.get("seed"),
                     tuple(item.get("dims", []))) for item in cases}
    if len(cases) != 50 or actual_cases != expected_cases:
        problems.append("metadata cases do not enumerate exactly the registered 50 series")
    if problems:
        completion.attrs["preflight_errors"] = problems
    return completion, meta


def load_diagnostics(raw: Path, completion: pd.DataFrame) -> pd.DataFrame:
    path = raw / "per_task.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    table = pd.read_csv(path)
    required = {"geometry", "optimizer", "seed", "task", "start_acc", "end_acc",
                "online_acc", "start_ce", "end_ce", "mean_abs_dphi_l1",
                "mean_abs_dphi_l2", "activeunit_frac_l1", "activeunit_frac_l2"}
    if missing := required - set(table.columns):
        raise ValueError(f"per_task.csv missing columns: {sorted(missing)}")
    expected = set()
    for case in completion.itertuples(index=False):
        end = TASKS if case.status == "COMPLETE" else int(case.first_nonfinite_task)
        expected.update((case.geometry, case.optimizer, case.seed, t)
                        for t in range(1, end+1))
    if len(table) != len(expected):
        raise ValueError(f"per_task.csv requires {len(expected)} registered rows, found {len(table)}")
    actual = set(zip(table.geometry, table.optimizer, table.seed, table.task))
    if actual != expected or table.duplicated(["geometry", "optimizer", "seed", "task"]).any():
        raise ValueError("per_task.csv has missing or duplicate series/task rows")
    if "status" not in table:
        raise ValueError("per_task.csv lacks status")
    expected_status = {(c.geometry, c.optimizer, c.seed, t):
                       ("DIVERGED" if c.status == "DIVERGED" and t == c.first_nonfinite_task else "OK")
                       for c in completion.itertuples(index=False)
                       for t in range(1, (TASKS if c.status == "COMPLETE" else int(c.first_nonfinite_task))+1)}
    for row in table.itertuples(index=False):
        key = (row.geometry, row.optimizer, int(row.seed), int(row.task))
        if row.status != expected_status[key]:
            raise ValueError(f"per_task.csv inconsistent terminal row {key}: {row.status}")
    finite_columns = required - {"geometry", "optimizer", "end_acc", "online_acc", "end_ce",
                                 "mean_abs_dphi_l1", "mean_abs_dphi_l2",
                                 "activeunit_frac_l1", "activeunit_frac_l2"}
    for name in finite_columns:
        if not np.isfinite(pd.to_numeric(table[name], errors="raise").to_numpy(dtype=float)).all():
            raise ValueError(f"nonfinite per_task {name}")
    for name in ("end_acc", "online_acc", "end_ce", "mean_abs_dphi_l1",
                 "mean_abs_dphi_l2", "activeunit_frac_l1", "activeunit_frac_l2"):
        good = table.status == "OK"
        if not np.isfinite(pd.to_numeric(table.loc[good, name], errors="raise").to_numpy(dtype=float)).all():
            raise ValueError(f"nonfinite successful per_task {name}")
    return table


def _state(path: Path, geometry: str, optimizer: str, task: int) -> dict[str, np.ndarray]:
    names = ("W1", "b1", "W2", "b2", "W3", "b3", "task", "step", "optimizer_step")
    state = load_npz(path, *names)
    d = GEOMETRIES[geometry]
    expected_shapes = {"W1": (100, d), "b1": (100,), "W2": (100, 100),
                       "b2": (100,), "W3": (10, 100), "b3": (10,)}
    for key, shape in expected_shapes.items():
        if state[key].shape != shape or state[key].dtype != np.float32:
            raise ValueError(f"{path}: {key} shape/dtype mismatch")
        if not np.isfinite(state[key]).all():
            raise ValueError(f"{path}: nonfinite {key}")
    if int(state["task"]) != task or int(state["step"]) != task*30000:
        raise ValueError(f"{path}: task/global step misaligned")
    if int(state["optimizer_step"]) != (task*30000 if optimizer == "adam" else 0):
        raise ValueError(f"{path}: optimizer step misaligned")
    with np.load(path, allow_pickle=False) as archive:
        for kind in ("adam_m", "adam_v"):
            for key, shape in expected_shapes.items():
                name = f"{kind}_{key}"
                if name not in archive:
                    raise ValueError(f"{path}: missing moment {name}")
                value = archive[name]
                expected = shape if optimizer == "adam" else ()
                if value.shape != expected or value.dtype != np.float32 or not np.isfinite(value).all():
                    raise ValueError(f"{path}: invalid moment {name}")
                if optimizer == "sgd" and float(value) != 0:
                    raise ValueError(f"{path}: nonzero SGD moment {name}")
    return state


def _bank(raw: Path, geometry: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    bank = load_npz(raw / "banks" / geometry / f"seed{seed}.npz", "x", "indices")
    x, indices = bank["x"], bank["indices"]
    if x.shape != (N_IMAGES, GEOMETRIES[geometry]) or x.dtype != np.float32:
        raise ValueError(f"{geometry}/{seed}: image bank shape/dtype mismatch")
    if indices.shape != (N_IMAGES,) or len(np.unique(indices)) != N_IMAGES:
        raise ValueError(f"{geometry}/{seed}: invalid subset indices")
    if not np.isfinite(x).all():
        raise ValueError(f"{geometry}/{seed}: nonfinite image bank")
    return x, indices


def _verify_duplication(rgb: np.ndarray, dup: np.ndarray, half: np.ndarray) -> None:
    reshaped = rgb.reshape(N_IMAGES, 3, 32, 32)
    expected = np.repeat(np.repeat(reshaped, 2, axis=2), 2, axis=3).reshape(N_IMAGES, -1)
    if not np.array_equal(dup, expected) or not np.array_equal(half, expected*0.5):
        raise ValueError("dup64/dup64_half banks are not exact pixel transforms of rgb32")


def _verify_geometry(rgb: np.ndarray, gray: np.ndarray, avg: np.ndarray) -> None:
    """Audit the saved information-changing transforms against the raw bank."""
    z = rgb.reshape(N_IMAGES, 3, 32, 32)
    expected_gray = (z[:, 0] * np.float32(.299) + z[:, 1] * np.float32(.587)
                     + z[:, 2] * np.float32(.114)).reshape(N_IMAGES, -1)
    expected_avg = z.reshape(N_IMAGES, 3, 16, 2, 16, 2).mean(axis=(3, 5)).reshape(N_IMAGES, -1)
    if not np.allclose(gray, expected_gray, rtol=2e-7, atol=2e-7):
        raise ValueError("gray32 bank disagrees with RGB channel weights")
    if not np.allclose(avg, expected_avg, rtol=2e-7, atol=2e-7):
        raise ValueError("avg16 bank disagrees with nonoverlapping 2x2 spatial average")


def _series_transitions(raw: Path, geometry: str, optimizer: str, seed: int,
                        centered: np.ndarray, mean: np.ndarray, top_left: np.ndarray,
                        diagnostics: pd.DataFrame, last_saved_task: int) -> pd.DataFrame:
    sd = raw / geometry / optimizer / f"seed{seed}"
    previous = _state(sd / "t000.npz", geometry, optimizer, 0)
    pprev = None
    rows = []
    diag = diagnostics[(diagnostics.geometry == geometry) & (diagnostics.optimizer == optimizer)
                       & (diagnostics.seed == seed)].set_index("task")
    for task in range(1, last_saved_task+1):
        current = _state(sd / f"t{task:03d}.npz", geometry, optimizer, task)
        terms, pprev = projection_terms(centered, mean, previous["W1"], previous["b1"],
                                        current["W1"], current["b1"], top_left, pprev)
        diagrow = diag.loc[task]
        rows.append({"geometry": geometry, "optimizer": optimizer, "seed": seed,
                     "task": task, "source_prev": str(sd / f"t{task-1:03d}.npz"),
                     "source_next": str(sd / f"t{task:03d}.npz"),
                     "train_acc": float(diagrow.end_acc),
                     "task_start_acc": float(diagrow.start_acc),
                     "online_acc": float(diagrow.online_acc),
                     "train_ce": float(diagrow.end_ce),
                     "task_start_ce": float(diagrow.start_ce),
                     "mean_abs_dphi_l1": float(diagrow.mean_abs_dphi_l1),
                     "mean_abs_dphi_l2": float(diagrow.mean_abs_dphi_l2),
                     "activeunit_frac_l1": float(diagrow.activeunit_frac_l1),
                     "activeunit_frac_l2": float(diagrow.activeunit_frac_l2),
                     **terms})
        previous = current
    return pd.DataFrame(rows)


def _closure(table: pd.DataFrame, summary: dict, labels: dict) -> list[dict]:
    value = float(summary["split_V"])
    result = []
    test = table[table.task >= summary["heldout_first_task"]].sort_values("task")
    for row in test.itertuples():
        value = (1 - 2*float(summary["gamma"])) * value + float(summary["qbar"])
        result.append({**labels, "task": int(row.task), "observed_V": float(row.V_next),
                       "model_V": value, "constant_V": float(summary["split_V"])})
    return result


def summarize_seeds(transitions: pd.DataFrame,
                    completion: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, trajectories = [], []
    completion_lookup = {} if completion is None else {
        (r.geometry, r.optimizer, int(r.seed)): r for r in completion.itertuples(index=False)}
    for (geometry, optimizer, seed), table in transitions.groupby(["geometry", "optimizer", "seed"]):
        if (geometry, optimizer, int(seed)) in completion_lookup and completion_lookup[(geometry, optimizer, int(seed))].status == "DIVERGED":
            continue
        table = table.sort_values("task")
        summary = PRIOR.summarize_task_table(table)
        raw = pd.DataFrame({"task": table.task, "V": table.raw_W_sq,
                            "Q": table.raw_D_sq, "X": table.raw_X,
                            "V_next": table.raw_W_next_sq})
        raw_summary = PRIOR.summarize_task_table(raw)
        gap = float(raw_summary["late_loglog_slope"] - summary["late_loglog_slope"])
        summary["raw_minus_effective_slope"] = gap
        summary["P4"] = "SEPARATED" if summary["P2"] == "PASS" and gap >= .2 else "NOT_ESTABLISHED"
        onset = low_response_onset(table)
        summary["low_response_onset_task"] = onset if onset is not None else math.nan
        summary["response_censoring"] = "LOW_RESPONSE" if onset is not None else "NO_LOW_RESPONSE"
        late = table[(table.task >= LATE[0]) & (table.task <= LATE[1])]
        summary["late_median_R"] = float(late.R.median())
        summary["late_median_d_eff"] = float(late.d_eff.median())
        summary["late_median_c"] = float(late.c.median())
        summary["late_median_rho"] = float(late.rho.median())
        summary["late_median_train_acc"] = float(late.train_acc.median())
        summary["late_median_top10_Q_fraction"] = float(late.top10_Q_fraction.median())
        summary["late_mean_shift_sq"] = float(late.mean_shift_sq.mean())
        if summary["late_invalid_V"]:
            label = "INVALID_METRIC"
        elif onset is not None and onset <= TASKS:
            label = "LOW_RESPONSE"
        elif summary["stopped"]:
            label = "STOPPED"
        elif summary["P2"] == "PASS":
            label = "PLATEAU_WITH_ACTIVITY"
        elif summary["P1"] == "PASS":
            label = "NEAR_BALANCE_WITH_ACTIVITY"
        elif np.isfinite(summary["late_loglog_slope"]) and summary["late_loglog_slope"] > .2:
            label = "TRANSIENT_GROWTH"
        else:
            label = "OTHER_ACTIVE"
        summary["dynamics_label"] = label
        labels = {"geometry": geometry, "optimizer": optimizer, "seed": int(seed)}
        rows.append({**labels, "status": "COMPLETE", "last_saved_task": TASKS,
                     "first_nonfinite_task": math.nan, **summary})
        trajectories += _closure(table, summary, labels)
    if completion is not None:
        for c in completion.itertuples(index=False):
            if c.status != "DIVERGED":
                continue
            rows.append({"geometry": c.geometry, "optimizer": c.optimizer, "seed": int(c.seed),
                         "status": "DIVERGED", "last_saved_task": int(c.last_saved_task),
                         "first_nonfinite_task": int(c.first_nonfinite_task),
                         "n_transitions": int(c.last_saved_task),
                         "P1": "DIVERGED", "P2": "DIVERGED", "P3": "DIVERGED",
                         "P4": "DIVERGED", "dynamics_label": "DIVERGED",
                         "response_censoring": "DIVERGED",
                         "late_nearbalance": math.nan, "late_activity": math.nan,
                         "late_loglog_slope": math.nan, "heldout_mape": math.nan,
                         "raw_minus_effective_slope": math.nan})
    return (pd.DataFrame(rows),
            pd.DataFrame(trajectories, columns=["geometry", "optimizer", "seed", "task",
                                                "observed_V", "model_V", "constant_V"]))


def p_group_verdicts(seed_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    columns = {"P1": "late_nearbalance", "P2": "late_loglog_slope",
               "P3": "heldout_mape", "P4": "raw_minus_effective_slope"}
    for (geometry, optimizer), block in seed_summary.groupby(["geometry", "optimizer"]):
        for question, column in columns.items():
            outcomes = block[question].tolist()
            wins = "SEPARATED" if question == "P4" else "PASS"
            n_pass = int(sum(v == wins for v in outcomes))
            n_uninformative = int(sum(v == "UNINFORMATIVE" for v in outcomes))
            n_diverged = int(sum(v == "DIVERGED" for v in outcomes))
            values = block[column].to_numpy(dtype=np.float64)
            lo, hi = bootstrap_median_ci(values)
            finite = values[np.isfinite(values)]
            verdict = (wins if len(block) == 5 and n_pass >= 4 else
                       "UNINFORMATIVE" if question == "P3" and n_uninformative >= 4 else
                       "INCOMPLETE" if len(block) < 5 else
                       "NOT_ESTABLISHED" if question == "P4" else "FAIL")
            rows.append({"geometry": geometry, "optimizer": optimizer,
                         "question": question, "verdict": verdict,
                         "n_seed": len(block), "n_pass": n_pass,
                         "n_diverged": n_diverged,
                         "n_uninformative": n_uninformative,
                         "statistic": column,
                         "seed_median": float(np.median(finite)) if len(finite) else math.nan,
                         "ci95_low": lo, "ci95_high": hi})
    return pd.DataFrame(rows)


def _source_manifest(raw: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(p for p in raw.rglob("*") if p.is_file()):
        digest = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
        rows.append({"source": str(path), "relative_source": str(path.relative_to(raw)),
                     "bytes": path.stat().st_size, "sha256": digest.hexdigest()})
    return pd.DataFrame(rows)


def _markdown(df: pd.DataFrame, columns: tuple[str, ...]) -> str:
    if df.empty:
        return "(no rows)"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in df.itertuples(index=False):
        data = row._asdict()
        cells = []
        for column in columns:
            value = data.get(column, "")
            cells.append(f"{value:.3g}" if isinstance(value, (float, np.floating)) and np.isfinite(value)
                         else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _write_report(path: Path, completion: pd.DataFrame, verdicts: pd.DataFrame,
                  groups: pd.DataFrame, covstats: pd.DataFrame) -> None:
    p = verdicts[verdicts.question.str.startswith("P")]
    e = verdicts[verdicts.question.isin(["E1", "E2", "E3", "E4", "E5"])]
    failed = completion[completion.status == "DIVERGED"]
    lines = ["# CIFAR image geometry and effective displacement", "",
             f"Analysis status: **TERMINAL** ({50-len(failed)} complete, {len(failed)} DIVERGED of 50 registered seed series).", "",
             "DIVERGED seeds remain nonpasses for 50-task P1–P4. A fully observed finite early or late registered window before failure remains usable for its window comparison; missing windows are invalid. No shortened horizon is treated as 50 tasks.", "",
             "## Terminal failures", "",
             _markdown(failed, ("geometry", "optimizer", "seed", "last_saved_task", "first_nonfinite_task")), "",
             "First-layer V/Q/X use float64 centered projections of the saved 1200-image bank; no ambient covariance matrix is formed.",
             "P1–P4 use the earlier fixed-covariance implementation and thresholds. Identity residuals audit arithmetic and are not evidence for the hypothesis.",
             "STOPPED and LOW_RESPONSE remain distinct from an active plateau. P3 uses tasks 6–25 for calibration and 26–50 for held-out prediction.",
             "dup64 and dup64_half preserve information; gray32 and avg16 also change information and task difficulty.",
             "Raw01 inputs are not the earlier standardized CIFAR condition. All geometries use the same global learning rates, including the first layer.", "",
             "## Registered P1–P4", "",
             _markdown(p, ("geometry", "optimizer", "question", "verdict", "n_pass", "seed_median", "ci95_low", "ci95_high")), "",
             "## Registered E1–E5", "",
             _markdown(e, ("question", "verdict", "n_pass", "criterion")), "",
             "## Output tables", "",
             "`transitions.csv` has observed endpoint tasks and source snapshot paths. `seed_summary.csv` has all 50 seeds, including explicit DIVERGED rows; `closure.csv` contains only complete held-out predictions. `windows.csv` marks each registered window's status and availability. `contrasts.csv` retains invalid pairs as missing with NaN rather than dropping their seeds; `paired_groups.csv` and `groups.csv` bootstrap valid seed-level values (5000 draws, seed 924). `covstats.csv` and `spectrum.csv` report rank, participation ratio, and covariance spectrum. `source_manifest.csv` hashes all raw inputs.", "",
             "If P2 fails, reported R is an observed finite-window width rather than an established fixed-point height.", ""]
    path.write_text("\n".join(lines))


def analyze(raw: Path, out: Path) -> dict:
    raw, out = Path(raw).resolve(), Path(out).resolve()
    completion, meta = preflight(raw)
    out.mkdir(parents=True, exist_ok=True)
    completion.to_csv(out / "completion.csv", index=False)
    errors = completion.attrs.get("preflight_errors", [])
    if errors:
        raise ValueError("strict preflight failed: " + "; ".join(errors))
    diagnostics = load_diagnostics(raw, completion)
    completion_lookup = {(r.geometry, r.optimizer, int(r.seed)): r
                         for r in completion.itertuples(index=False)}
    transitions, covstats, spectra = [], [], []
    for seed in SEEDS:
        labels = load_npz(raw / "labels" / f"seed{seed}.npz", "labels")["labels"]
        if labels.shape != (TASKS, N_IMAGES) or not np.issubdtype(labels.dtype, np.integer):
            raise ValueError(f"seed{seed}: invalid label schedule shape/dtype")
        if np.any((labels < 0) | (labels >= 10)):
            raise ValueError(f"seed{seed}: labels outside 0..9")
        banks = {geometry: _bank(raw, geometry, seed) for geometry in GEOMETRIES}
        first_indices = banks["rgb32"][1]
        for geometry, (_, indices) in banks.items():
            if not np.array_equal(indices, first_indices):
                raise ValueError(f"seed{seed}: bank indices differ for {geometry}")
        _verify_duplication(banks["rgb32"][0], banks["dup64"][0],
                            banks["dup64_half"][0])
        _verify_geometry(banks["rgb32"][0], banks["gray32"][0], banks["avg16"][0])
        rgb_reference = None
        for geometry in GEOMETRIES:
            x = banks[geometry][0]
            xx = x.astype(np.float64)
            mean = xx.mean(0)
            centered = xx - mean
            if geometry == "dup64":
                stats, singular, top_left = covariance_spectrum(x, reference=rgb_reference, scale=2.0)
            elif geometry == "dup64_half":
                stats, singular, top_left = covariance_spectrum(x, reference=rgb_reference, scale=1.0)
            else:
                stats, singular, top_left = covariance_spectrum(x)
                if geometry == "rgb32":
                    rgb_reference = {"singular_values": singular, "top_left": top_left}
            covstats.append({"geometry": geometry, "seed": seed, **stats,
                             "input_mean_sq": float(np.dot(mean, mean)),
                             "source_bank": str(raw / "banks" / geometry / f"seed{seed}.npz")})
            spectra += [{"geometry": geometry, "seed": seed, "component": i+1,
                         "cov_eigenvalue": float(value*value)}
                        for i, value in enumerate(singular)]
            for optimizer in OPTIMIZERS:
                last = int(completion_lookup[(geometry, optimizer, seed)].last_saved_task)
                transitions.append(_series_transitions(raw, geometry, optimizer, seed,
                                                       centered, mean, top_left, diagnostics, last))
    observed = [part for part in transitions if not part.empty]
    table = (pd.concat(observed, ignore_index=True) if observed else
             pd.DataFrame(columns=["geometry", "optimizer", "seed", "task", "identity_residual",
                                   "V_next", "R_next", "d_eff"]))
    if (len(table) != int(completion.last_saved_task.sum())
            or table.duplicated(["geometry", "optimizer", "seed", "task"]).any()):
        raise ValueError("observed transition table is incomplete or duplicated")
    # Arithmetic audits use the scale of the width, not a fixed absolute bound.
    residual = np.abs(table.identity_residual.to_numpy(dtype=float))
    tolerance = 1e-9 * np.maximum(1.0, table.V_next.to_numpy(dtype=float))
    if np.any(residual > tolerance):
        raise ValueError("projection identity residual exceeds float64 audit tolerance")
    seed_summary, closure = summarize_seeds(table, completion)
    windows = seed_window_medians(table, completion)
    contrasts = paired_contrasts(windows)
    group_contrasts = paired_groups(contrasts)
    groups = window_groups(windows)
    verdicts = pd.concat([p_group_verdicts(seed_summary),
                          preregistered_e_verdicts(contrasts)], ignore_index=True)
    results = {"transitions.csv": table, "seed_summary.csv": seed_summary,
               "closure.csv": closure, "windows.csv": windows,
               "contrasts.csv": contrasts, "paired_groups.csv": group_contrasts,
               "groups.csv": groups, "verdicts.csv": verdicts,
               "covstats.csv": pd.DataFrame(covstats), "spectrum.csv": pd.DataFrame(spectra)}
    for filename, frame in results.items():
        frame.to_csv(out / filename, index=False)
    manifest = _source_manifest(raw)
    manifest.to_csv(out / "source_manifest.csv", index=False)
    _write_report(out / "summary.md", completion, verdicts, groups, pd.DataFrame(covstats))
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from analysis.effdisp_imagegeom_0924.plots import make_plots
    figures = make_plots(table, seed_summary, windows, contrasts,
                         pd.DataFrame(covstats), pd.DataFrame(spectra), out / "figures")
    n_diverged = int((completion.status == "DIVERGED").sum())
    metadata = {"status": "COMPLETE" if n_diverged == 0 else "TERMINAL_WITH_DIVERGENCE",
                "completed_series": 50-n_diverged, "diverged_series": n_diverged,
                "raw": str(raw), "output": str(out),
                "source_git_hash": meta["git_hash"], "source_sha256": meta["source_sha256"],
                "spec_sha256": meta["spec_sha256"], "series": 50,
                "tasks_per_series": TASKS, "seed_bootstrap_draws": BOOTSTRAP_DRAWS,
                "seed_bootstrap_rng": BOOTSTRAP_SEED, "early": list(EARLY), "late": list(LATE),
                "covariance_representation": "centered image projections C@W.T/sqrt(1200), float64",
                "rank_rule": "singular > maximum singular * 1e-10; duplicate singulars derived from verified exact transform",
                "top10_Q_fraction": "||U_top10.T C D.T||_F² / ||C D.T||_F²; undefined when Q=0",
                "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "prior_metrics_sha256": hashlib.sha256((REPO / "analysis/effdisp_validation_0924/metrics.py").read_bytes()).hexdigest(),
                "tables": list(results) + ["completion.csv", "source_manifest.csv"],
                "figures": [str(p) for p in figures],
                "max_abs_identity_residual": float(residual.max()) if len(residual) else None}
    (out / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--out", type=Path, default=REPO / "results/effdisp_imagegeom_0924")
    args = parser.parse_args()
    print(json.dumps(analyze(args.raw, args.out), indent=2))


if __name__ == "__main__":
    main()
