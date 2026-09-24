#!/usr/bin/env python3
"""Analyze the separately prioritized avg16 run against completed rgb32.

This is a two-condition priority result, not the registered 50-series final
geometry analysis. Run only after priority_avg16/launch.json says SUCCESS.
CPU only; set OPENBLAS_NUM_THREADS=2 and OMP_NUM_THREADS=2 at launch.
"""

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
ROOT = Path("/home/issan/Projects/obsidian-research-data/effdisp_imagegeom_0924")
SOURCE_HASH = "273c3fa3b52dc0c954eaaa58a4e711183c259770"
SEEDS = tuple(range(300, 305))
OPTIMIZERS = ("adam", "sgd")
GEOMETRIES = ("rgb32", "avg16")
WINDOWS = {"early": (6, 15), "late": (41, 50)}
DIMENSIONS = {"rgb32": 3072, "avg16": 768}
TASKS = 50
sys.path.insert(0, str(REPO))
from analysis.effdisp_imagegeom_0924 import analyze as A  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def median_finite(values) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if len(values) else math.nan


def preflight(root: Path) -> dict:
    main = root / "raw"
    priority = root / "priority_avg16"
    launch = A.load_json(priority / "launch.json")
    if launch.get("status") != "SUCCESS" or launch.get("returncode") != 0:
        raise RuntimeError("priority avg16 run has not completed successfully")
    if launch.get("source_git_hash") != SOURCE_HASH:
        raise ValueError("priority launch source hash differs from frozen source")
    m0, m1 = A.load_json(main / "metadata.json"), A.load_json(priority / "raw" / "metadata.json")
    if m0.get("git_hash") != SOURCE_HASH or m1.get("git_hash") != SOURCE_HASH:
        raise ValueError("training source hashes differ")
    source_sha = sha256(REPO / "src/effdisp_imagegeom_0924.py")
    if m0.get("source_sha256") != source_sha or m1.get("source_sha256") != source_sha:
        raise ValueError("training source SHA256 differs")
    for label, meta in (("main", m0), ("priority", m1)):
        if (meta.get("tasks") != TASKS or meta.get("epochs") != 400 or
                meta.get("batch") != 16 or meta.get("images_per_seed") != 1200 or
                meta.get("steps_per_task") != 30000 or tuple(meta.get("seeds", [])) != SEEDS):
            raise ValueError(f"{label} metadata differs from canonical schedule")
        settings = meta.get("optimizer_settings", {})
        if settings.get("adam") != {"lr": .001, "betas": [.9, .999], "eps": 1e-8} or settings.get("sgd") != {"lr": .01, "momentum": 0}:
            raise ValueError(f"{label} optimizer metadata differs")
    if m0.get("spec_sha256") != m1.get("spec_sha256"):
        raise ValueError("runs used different preregistered specification bytes")
    if m0.get("spec_sha256") != sha256(REPO / "specs/spec_effdisp_imagegeom_0924.md"):
        raise ValueError("saved specification SHA256 differs from frozen local specification")
    done = A.load_json(priority / "raw" / "DONE.json")
    if (done.get("status") != "SUCCESS" or done.get("expected_cases") != 10 or
            done.get("completed_cases") != 10 or done.get("failed_cases") != [] or
            done.get("git_hash") != SOURCE_HASH or done.get("source_sha256") != source_sha):
        raise ValueError("priority DONE.json is incomplete or source mismatched")
    expected_priority_cases = {(g, o, s) for g in ("avg16",) for o in OPTIMIZERS for s in SEEDS}
    actual_priority_cases = {(r.get("geometry"), r.get("optimizer"), r.get("seed")) for r in m1.get("cases", [])}
    if len(m1.get("cases", [])) != 10 or actual_priority_cases != expected_priority_cases:
        raise ValueError("priority metadata does not identify the 10 avg16 series")
    for geometry in GEOMETRIES:
        raw = main if geometry == "rgb32" else priority / "raw"
        for optimizer in OPTIMIZERS:
            for seed in SEEDS:
                series = raw / geometry / optimizer / f"seed{seed}"
                status = A.load_json(series / "status.json")
                if (status.get("status") != "COMPLETED" or status.get("last_saved_task") != TASKS or
                        status.get("first_nonfinite_task") is not None or
                        (status.get("geometry"), status.get("optimizer"), status.get("seed")) != (geometry, optimizer, seed)):
                    raise ValueError(f"series incomplete: {series}")
                missing = [series / f"t{task:03d}.npz" for task in range(TASKS+1)
                           if not (series / f"t{task:03d}.npz").is_file()]
                if missing:
                    raise FileNotFoundError(missing[0])
    return {"main": m0, "priority": m1, "launch": launch, "main_raw": main,
            "priority_raw": priority / "raw"}


def diagnostics(raw: Path, geometry: str) -> pd.DataFrame:
    path = raw / "per_task.csv"
    table = pd.read_csv(path)
    table = table[table.geometry == geometry].copy()
    expected = {(geometry, o, s, t) for o in OPTIMIZERS for s in SEEDS for t in range(1, TASKS+1)}
    actual = set(zip(table.geometry, table.optimizer, table.seed, table.task))
    if len(table) != len(expected) or actual != expected or table.duplicated(["geometry", "optimizer", "seed", "task"]).any():
        raise ValueError(f"incomplete or duplicate diagnostics: {path} {geometry}")
    needed = ("start_acc", "end_acc", "online_acc", "start_ce", "end_ce",
              "mean_abs_dphi_l1", "mean_abs_dphi_l2", "activeunit_frac_l1", "activeunit_frac_l2")
    if not table.status.eq("OK").all() or not np.isfinite(table[list(needed)].to_numpy(dtype=np.float64)).all():
        raise ValueError(f"invalid diagnostics: {path} {geometry}")
    return table


def matched_banks(main: Path, priority: Path, seed: int) -> tuple[dict[str, np.ndarray], dict]:
    rgb, rgb_ids = A._bank(main, "rgb32", seed)
    avg, avg_ids = A._bank(priority, "avg16", seed)
    if not np.array_equal(rgb_ids, avg_ids):
        raise ValueError(f"seed{seed} subset IDs differ between runs")
    z = rgb.reshape(1200, 3, 32, 32)
    expected_avg = z.reshape(1200, 3, 16, 2, 16, 2).mean(axis=(3, 5)).reshape(1200, -1)
    if not np.allclose(avg, expected_avg, rtol=2e-7, atol=2e-7):
        raise ValueError(f"seed{seed} avg16 bank disagrees with RGB block averaging")
    labels0 = A.load_npz(main / "labels" / f"seed{seed}.npz", "labels")["labels"]
    labels1 = A.load_npz(priority / "labels" / f"seed{seed}.npz", "labels")["labels"]
    if labels0.shape != (TASKS, 1200) or not np.array_equal(labels0, labels1):
        raise ValueError(f"seed{seed} task label schedules differ")
    return {"rgb32": rgb, "avg16": avg}, {
        "seed": seed, "same_indices": True, "same_labels": True,
        "rgb_bank_sha256": sha256(main / "banks" / "rgb32" / f"seed{seed}.npz"),
        "avg_bank_sha256": sha256(priority / "banks" / "avg16" / f"seed{seed}.npz"),
        "main_labels_sha256": sha256(main / "labels" / f"seed{seed}.npz"),
        "priority_labels_sha256": sha256(priority / "labels" / f"seed{seed}.npz")}


def enrich_transitions(frame: pd.DataFrame, geometry: str, origin: str) -> pd.DataFrame:
    frame = frame.copy()
    n_params = 100 * DIMENSIONS[geometry]
    frame["raw_W_F"] = np.sqrt(frame.raw_W_sq)
    frame["raw_W_RMS"] = frame.raw_W_F / math.sqrt(n_params)
    frame["raw_D_F"] = np.sqrt(frame.raw_D_sq)
    frame["Rstar_local"] = np.nan
    negative = (frame.c < 0) & np.isfinite(frame.c) & (frame.d_eff > 0)
    frame.loc[negative, "Rstar_local"] = (frame.loc[negative, "d_eff"] /
                                          (2*frame.loc[negative, "c"].abs()))
    frame["R_over_Rstar_local"] = frame.R / frame.Rstar_local
    frame["condition_origin"] = origin
    return frame


def seed_summary(frame: pd.DataFrame) -> dict:
    result = A.PRIOR.summarize_task_table(frame)
    onset = A.low_response_onset(frame)
    result["low_response_onset_task"] = onset if onset is not None else math.nan
    result["dynamics_label"] = ("LOW_RESPONSE" if onset is not None else
                                "STOPPED" if result["stopped"] else
                                "ACTIVE_P2" if result["P2"] == "PASS" else "ACTIVE_OTHER")
    return result


def aggregate(transitions: pd.DataFrame, seed_summaries: pd.DataFrame):
    metrics = ("R", "d_eff", "c", "raw_W_F", "raw_W_RMS", "raw_D_F",
               "Rstar_local", "R_over_Rstar_local", "train_acc", "top10_Q_fraction")
    windows = []
    for (geometry, optimizer, seed), block in transitions.groupby(["geometry", "optimizer", "seed"]):
        for name, (first, last) in WINDOWS.items():
            part = block[(block.task >= first) & (block.task <= last)]
            if len(part) != 10:
                raise ValueError(f"incomplete {name} window: {geometry}/{optimizer}/{seed}")
            row = {"geometry": geometry, "optimizer": optimizer, "seed": seed,
                   "window": name, "first_task": first, "last_task": last,
                   "n_negative_c": int((part.c < 0).sum())}
            for metric in metrics:
                values = part[metric].to_numpy(dtype=np.float64)
                row[metric] = median_finite(values)
                row[f"n_valid_{metric}"] = int(np.isfinite(values).sum())
            windows.append(row)
    windows = pd.DataFrame(windows)
    pairs = []
    for optimizer in OPTIMIZERS:
        for seed in SEEDS:
            for window in WINDOWS:
                a = windows[(windows.geometry == "avg16") & (windows.optimizer == optimizer)
                            & (windows.seed == seed) & (windows.window == window)].iloc[0]
                b = windows[(windows.geometry == "rgb32") & (windows.optimizer == optimizer)
                            & (windows.seed == seed) & (windows.window == window)].iloc[0]
                for metric in metrics:
                    va, vb = float(a[metric]), float(b[metric])
                    pairs.append({"optimizer": optimizer, "seed": seed, "window": window,
                                  "metric": metric, "avg16_value": va, "rgb32_value": vb,
                                  "difference": va-vb,
                                  "ratio": va/vb if metric != "c" and np.isfinite(va) and np.isfinite(vb) and vb > 0 else math.nan})
    pairs = pd.DataFrame(pairs)
    groups = []
    for (optimizer, window, metric), block in pairs.groupby(["optimizer", "window", "metric"]):
        for statistic in ("ratio", "difference"):
            values = block[statistic].to_numpy(dtype=np.float64)
            lo, hi = A.bootstrap_median_ci(values)
            groups.append({"optimizer": optimizer, "window": window, "metric": metric,
                           "statistic": statistic, "n_seed": 5,
                           "n_valid": int(np.isfinite(values).sum()),
                           "paired_seed_median": median_finite(values),
                           "ci95_low": lo, "ci95_high": hi})
    p_groups = []
    for (geometry, optimizer), block in seed_summaries.groupby(["geometry", "optimizer"]):
        if len(block) != 5:
            raise ValueError("P1–P3 group lost a seed")
        for question in ("P1", "P2", "P3"):
            n_pass = int(block[question].eq("PASS").sum())
            n_uninformative = int(block[question].eq("UNINFORMATIVE").sum())
            verdict = ("PASS" if n_pass >= 4 else "UNINFORMATIVE" if question == "P3" and n_uninformative >= 4 else "FAIL")
            p_groups.append({"geometry": geometry, "optimizer": optimizer, "question": question,
                             "n_seed": 5, "n_pass": n_pass, "n_uninformative": n_uninformative,
                             "n_low_response": int(block.dynamics_label.eq("LOW_RESPONSE").sum()),
                             "verdict": verdict})
    return windows, pairs, pd.DataFrame(groups), pd.DataFrame(p_groups)


def analyze(root: Path, out: Path) -> dict:
    provenance = preflight(root)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty output: {out}")
    main, priority = provenance["main_raw"], provenance["priority_raw"]
    diag = {"rgb32": diagnostics(main, "rgb32"), "avg16": diagnostics(priority, "avg16")}
    transitions, summaries, covstats, spectra, matching, checkpoint_rows = [], [], [], [], [], []
    for seed in SEEDS:
        banks, record = matched_banks(main, priority, seed)
        matching.append(record)
        for geometry in GEOMETRIES:
            x = banks[geometry].astype(np.float64)
            mean = x.mean(0)
            centered = x-mean
            stats, singular, top_left = A.covariance_spectrum(x)
            covstats.append({"geometry": geometry, "seed": seed, **stats,
                             "input_mean_sq": float(np.dot(mean, mean))})
            spectra.extend({"geometry": geometry, "seed": seed, "component": i+1,
                            "cov_eigenvalue": float(v*v)} for i, v in enumerate(singular))
            raw = main if geometry == "rgb32" else priority
            for optimizer in OPTIMIZERS:
                frame = A._series_transitions(raw, geometry, optimizer, seed,
                                              centered, mean, top_left, diag[geometry], TASKS)
                frame = enrich_transitions(frame, geometry,
                                           "main_rgb32" if geometry == "rgb32" else "priority_avg16")
                if len(frame) != TASKS or np.abs(frame.identity_residual).max() > 1e-8*max(1., frame.V_next.max()):
                    raise ValueError(f"projection identity or trajectory failed: {geometry}/{optimizer}/{seed}")
                summary = seed_summary(frame)
                summaries.append({"geometry": geometry, "optimizer": optimizer, "seed": seed,
                                  "condition_origin": frame.condition_origin.iloc[0], **summary})
                for task in (0, 10, 20, 30, 40, 50):
                    if task == 0:
                        endpoint_R = float(frame.R.iloc[0])
                        raw_F = math.sqrt(float(frame.raw_W_sq.iloc[0]))
                    else:
                        endpoint = frame[frame.task == task].iloc[0]
                        endpoint_R = float(endpoint.R_next)
                        raw_F = math.sqrt(float(endpoint.raw_W_next_sq))
                    checkpoint_rows.append({"geometry": geometry, "optimizer": optimizer,
                                            "seed": seed, "task": task, "R_endpoint": endpoint_R,
                                            "raw_W_F_endpoint": raw_F,
                                            "raw_W_RMS_endpoint": raw_F/math.sqrt(100*DIMENSIONS[geometry])})
                transitions.append(frame)
    table = pd.concat(transitions, ignore_index=True)
    seed_summaries = pd.DataFrame(summaries)
    windows, pairs, paired_groups, p_groups = aggregate(table, seed_summaries)
    checkpoint_points = pd.DataFrame(checkpoint_rows)
    checkpoint_pairs = []
    for optimizer in OPTIMIZERS:
        for seed in SEEDS:
            for task in (0, 10, 20, 30, 40, 50):
                a = checkpoint_points[(checkpoint_points.geometry == "avg16")
                                      & (checkpoint_points.optimizer == optimizer)
                                      & (checkpoint_points.seed == seed)
                                      & (checkpoint_points.task == task)].iloc[0]
                b = checkpoint_points[(checkpoint_points.geometry == "rgb32")
                                      & (checkpoint_points.optimizer == optimizer)
                                      & (checkpoint_points.seed == seed)
                                      & (checkpoint_points.task == task)].iloc[0]
                for metric in ("R_endpoint", "raw_W_F_endpoint", "raw_W_RMS_endpoint"):
                    va, vb = float(a[metric]), float(b[metric])
                    checkpoint_pairs.append({"optimizer": optimizer, "seed": seed, "task": task,
                                             "metric": metric, "avg16_value": va, "rgb32_value": vb,
                                             "ratio": va/vb if vb > 0 else math.nan,
                                             "difference": va-vb})
    out.mkdir(parents=True, exist_ok=True)
    outputs = {"transitions.csv": table, "seed_summary.csv": seed_summaries,
               "windows.csv": windows, "paired_avg16_over_rgb32.csv": pairs,
               "paired_groups.csv": paired_groups, "p_groups.csv": p_groups,
               "checkpoint_points.csv": checkpoint_points,
               "checkpoint_pairs.csv": pd.DataFrame(checkpoint_pairs),
               "covstats.csv": pd.DataFrame(covstats), "spectrum.csv": pd.DataFrame(spectra),
               "matched_inputs.csv": pd.DataFrame(matching)}
    for name, frame in outputs.items():
        frame.to_csv(out / name, index=False)
    interpretation = ["# Priority avg16 versus rgb32", "",
                      "Separate priority analysis of the completed rgb32 main arm and separately launched avg16 arm.",
                      "This is not the registered 50-series final image-geometry verdict.", "",
                      "Both conditions use the same five image IDs and task-label schedules per seed, 50 tasks, 30000 updates/task and frozen training source.",
                      "The avg16 transform removes spatial information; its native fan-in initialization does not match the RGB initial function. The same learning rate also acts in different optimizer coordinates. Paired ratios are descriptive and do not isolate input dimension alone.",
                      "raw_W_RMS = ||W1||F / sqrt(100*d) removes the elementary sqrt(d) factor from comparing raw Frobenius norms. The effective R and D use each condition's own fixed input covariance.",
                      "Rstar_local = D_eff/(2|c|) is reported only when c<0; R/Rstar is a local diagnostic, not proof of convergence.",
                      "Early is task6–15, late is task41–50. Window values are medians within seed; avg16/rgb32 ratios pair seeds before the five-seed median and bootstrap CI (5000 resamples, RNG924).",
                      "`checkpoint_points.csv` gives task-end R, raw Frobenius and raw RMS at task0/10/20/30/40/50; `checkpoint_pairs.csv` pairs the same seed and task. These trajectories show observed approach only, not an inferred limiting value.",
                      "P1–P3 use the prior fixed-covariance thresholds separately per condition; P3 calibrates on task6–25 and predicts task26–50. Group PASS requires four of five seeds. A P2 failure means observed width is not called a fixed-point height.", "",
                      "## P1–P3 per condition", "", p_groups.to_string(index=False), "",
                      "## Paired late-window metrics", "",
                      paired_groups[(paired_groups.window == "late") & (paired_groups.statistic == "ratio")].to_string(index=False), ""]
    (out / "summary.md").write_text("\n".join(interpretation))
    metadata = {"status": "COMPLETE_PRIORITY_TWO_CONDITION", "source_git_hash": SOURCE_HASH,
                "source_sha256": provenance["main"]["source_sha256"],
                "main_raw": str(main), "priority_raw": str(priority),
                "main_metadata_sha256": sha256(main / "metadata.json"),
                "priority_metadata_sha256": sha256(priority / "metadata.json"),
                "priority_launch_sha256": sha256(root / "priority_avg16" / "launch.json"),
                "script_sha256": sha256(Path(__file__)), "script_path": str(Path(__file__)),
                "seed_count": 5, "condition_count": 4, "series_count": 20,
                "tasks_per_series": TASKS, "windows": WINDOWS,
                "bootstrap_draws": A.BOOTSTRAP_DRAWS, "bootstrap_seed": A.BOOTSTRAP_SEED,
                "condition_note": "RGB completed in main run; avg16 completed in separately prioritized run; not final 50-series verdict",
                "raw_RMS_formula": "||W1||F/sqrt(100*d)",
                "files": {name: sha256(out / name) for name in outputs},
                "summary_sha256": sha256(out / "summary.md")}
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=ROOT / "priority_avg16" / "analysis")
    args = parser.parse_args()
    if os.environ.get("OPENBLAS_NUM_THREADS") != "2" or os.environ.get("OMP_NUM_THREADS") != "2":
        raise RuntimeError("launch with OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2")
    print(json.dumps(analyze(args.root, args.out), indent=2))


if __name__ == "__main__":
    main()
