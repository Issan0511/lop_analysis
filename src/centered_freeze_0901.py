"""P-1: condA・centered × ``freeze_bias``.

The expensive arm is only the frozen arm.  Its paired free reference is the
committed ``mlp2_phase1_0829/L1w100_A1`` trajectory.  A mandatory 30k replay
with ``freeze_bias=false`` proves that this small, legacy one-layer harness is
the same trajectory before the 5M frozen run is allowed.

Commands::

    OMP_NUM_THREADS=1 python -m src.centered_freeze_0901 --preflight
    OMP_NUM_THREADS=1 python -m src.centered_freeze_0901 --smoke
    OMP_NUM_THREADS=1 python -m src.centered_freeze_0901
    OMP_NUM_THREADS=1 python -m src.centered_freeze_0901 --analyze-only
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from analysis.center_selfcov.slopes import paired_boot_ci
from .common import ROOT, build_runs, group_runs, load_config, pick_device
from .ratchet_log import exact_record, full_support_ro, teacher_f64
from .train import train_group


CONFIG = Path(ROOT) / "configs" / "centered_freeze_0901.yaml"


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_omp(cfg: dict) -> dict:
    expected = str(int(cfg["centered_freeze"]["omp_num_threads"]))
    actual = os.environ.get("OMP_NUM_THREADS")
    if actual != expected:
        raise RuntimeError(f"OMP_NUM_THREADS must be {expected}, got {actual!r}")
    return {"pass_": True, "expected": expected, "actual": actual}


def validate_config(cfg: dict, *, full: bool) -> None:
    C, A, P = cfg["common"], cfg["condA"], cfg["centered_freeze"]
    if pick_device(cfg) != "cpu":
        raise ValueError("centered_freeze_0901 is CPU-only")
    if (int(A["m"]), int(A["f"]), int(A["target_hidden"])) != (20, 15, 100):
        raise ValueError("registered condA dimensions differ")
    if (list(A["T_values"]), list(A["widths"]), list(A["encodings"]),
            list(A["batch_values"])) != ([10_000], [100], ["centered"], [1]):
        raise ValueError("registered condA regime differs")
    if float(A["center_alpha"]) != 0.01 or float(C["lr_main"]) != 0.01:
        raise ValueError("registered alpha/lr differs")
    if A.get("freeze_bias") is not True:
        raise ValueError("the full config must register freeze_bias=true")
    if (float(P["near_zero_max"]), float(P["decisive_reduction_min"])) != (0.05, 0.80):
        raise ValueError("registered P1 thresholds differ")
    if (int(P["bootstrap_B"]), int(P["bootstrap_seed"])) != (10_000, 20_260_901):
        raise ValueError("registered bootstrap differs")
    if list(P["late_tasks"]) != [451, 500]:
        raise ValueError("registered late window differs")
    if full and (int(C["total_steps"]) != 5_000_000
                 or list(C["seeds"]) != list(range(10))):
        raise ValueError("full run requires 5M steps and seeds 0..9")


def exact_scalars(st: dict) -> tuple[dict[str, np.ndarray], dict]:
    """Exact 32-support endpoints without mutating the environment or RNG."""
    rec, sanity = exact_record(st, as_f64=True, _with_sanity=True)
    with torch.no_grad():
        X = full_support_ro(st["env"]).double()
        y = teacher_f64(st["teacher"], X)
        x_in = X - st["centered"].double()[None, :, None] * st["running_mean"].double()[None]
        pre = torch.einsum("rhd,prd->prh", st["net"].W.double(), x_in)
        pre = pre + st["net"].b.double()
        act = torch.relu(pre)
        yhat = (act * st["net"].v.double()).sum(dim=-1) + st["net"].c.double()
        residual = yhat - y
        signal_var = y.var(dim=0, unbiased=False)
        residual_var = residual.var(dim=0, unbiased=False)
        unfit = residual_var / signal_var
        eval_loss = residual.square().mean(dim=0)
    if not np.array_equal(eval_loss.cpu().numpy(), rec["eval_loss_exact"]):
        raise RuntimeError("independent exact eval disagrees with ratchet exact_record")
    p_hat, bias = rec["p_hat"], rec["b"]
    out = {
        "strict_dead": (p_hat == 0).sum(axis=1).astype(np.int64),
        "strict_dead_frac": (p_hat == 0).mean(axis=1),
        "eval_loss_exact": rec["eval_loss_exact"],
        "unfit": unfit.cpu().numpy(),
        "signal_var": signal_var.cpu().numpy(),
        "residual_var": residual_var.cpu().numpy(),
        "mu_norm": rec["mu_norm"],
        "b_mean": bias.mean(axis=1),
        "b_min": bias.min(axis=1),
        "b_max": bias.max(axis=1),
        "b_maxabs": np.abs(bias).max(axis=1),
    }
    return out, sanity


class ExactTaskRecorder:
    def __init__(self, *, keep_arrays: bool = False):
        self.rows: list[dict] = []
        self.keep_arrays = keep_arrays
        self.arrays: dict[int, dict[str, np.ndarray]] = {}
        self.sanity_rows: list[dict] = []

    def __call__(self, st: dict, step: int) -> None:
        values, sanity = exact_scalars(st)
        if self.keep_arrays:
            raw = exact_record(st, as_f64=True)
            self.arrays[int(step)] = {
                "p_hat": raw["p_hat"].copy(),
                "eval_loss_exact": raw["eval_loss_exact"].copy(),
                "unfit": values["unfit"].copy(),
            }
        self.sanity_rows.append({"step": int(step), **sanity})
        for ri, run in enumerate(st["runs"]):
            row = {"step": int(step), "run_id": run["run_id"],
                   "seed": int(run["seed"]), "arm": "frozen"}
            for key, array in values.items():
                value = array[ri]
                row[key] = int(value) if key == "strict_dead" else float(value)
            self.rows.append(row)

    def dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


def baseline_path(cfg: dict, seed: int) -> Path:
    arm = cfg["centered_freeze"]["baseline_arm"]
    return Path(ROOT) / cfg["baseline_dir"] / "logs" / f"{arm}_seed{seed}.npz"


def preflight(cfg: dict, outdir: Path) -> dict:
    """30k free replay against the committed free trajectory (S0)."""
    require_omp(cfg)
    P = cfg["centered_freeze"]
    steps = int(P["preflight_steps"])
    every = int(P["preflight_probe_every"])
    replay_cfg = copy.deepcopy(cfg)
    replay_cfg["condA"]["freeze_bias"] = False
    replay_cfg["common"]["total_steps"] = steps
    replay_cfg["common"]["lop_every"] = every
    replay_cfg["common"]["loss_bin"] = every
    replay_cfg["common"]["checkpoints"] = []
    runs = build_runs(replay_cfg)
    groups = group_runs(runs)
    if len(groups) != 1:
        raise RuntimeError("expected one vectorized condA group")
    recorder = ExactTaskRecorder(keep_arrays=True)
    replay_dir = outdir / "free_replay"
    for gkey, gruns in groups.items():
        train_group(gkey, gruns, replay_cfg, "cpu", str(replay_dir),
                    total_steps=steps, ckpts=[], probe=recorder,
                    probe_steps=range(0, steps + 1, every))

    differences = []
    max_abs = {"eval_loss_exact": 0.0, "unfit": 0.0}
    for ri, seed in enumerate(replay_cfg["common"]["seeds"]):
        path = baseline_path(cfg, int(seed))
        if not path.exists():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as z:
            for step, actual in recorder.arrays.items():
                indices = np.flatnonzero(z["step"] == step)
                if len(indices) != 1:
                    differences.append({"seed": int(seed), "step": step,
                                        "field": "step", "detail": str(len(indices))})
                    continue
                i = int(indices[0])
                expected_p = z["layer1_p_hat"][i]
                if not np.array_equal(actual["p_hat"][ri].astype(expected_p.dtype), expected_p):
                    differences.append({"seed": int(seed), "step": step,
                                        "field": "p_hat", "detail": "array mismatch"})
                for key in ("eval_loss_exact", "unfit"):
                    delta = abs(float(actual[key][ri]) - float(z[key][i]))
                    max_abs[key] = max(max_abs[key], delta)
                    if delta > 1e-12:
                        differences.append({"seed": int(seed), "step": step,
                                            "field": key, "detail": f"abs={delta:.3g}"})
    result = {
        "pass_": not differences,
        "steps": steps,
        "probe_every": every,
        "n_seeds": len(replay_cfg["common"]["seeds"]),
        "n_probes": len(recorder.arrays),
        "max_abs": max_abs,
        "differences": differences[:100],
    }
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "preflight.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    if not result["pass_"]:
        raise RuntimeError(f"S0 preflight failed: {differences[:3]}")
    print(f"S0 PASS: {result['n_seeds']} seeds × {result['n_probes']} probes", flush=True)
    return result


def run_frozen(cfg: dict, outdir: Path, *, smoke: bool) -> pd.DataFrame:
    require_omp(cfg)
    validate_config(cfg, full=not smoke)
    P, C = cfg["centered_freeze"], cfg["common"]
    total = int(P["preflight_steps"]) if smoke else int(C["total_steps"])
    every = int(P["preflight_probe_every"] if smoke else P["probe_every"])
    run_cfg = copy.deepcopy(cfg)
    if smoke:
        run_cfg["common"]["seeds"] = [0]
        run_cfg["common"]["total_steps"] = total
        run_cfg["common"]["lop_every"] = every
        run_cfg["common"]["loss_bin"] = every
        run_cfg["common"]["checkpoints"] = [0, total]

    runs = build_runs(run_cfg)
    pd.DataFrame(runs).to_csv(outdir / "runs.csv", index=False)
    with (outdir / "config_used.yaml").open("w") as fh:
        yaml.safe_dump(run_cfg, fh, allow_unicode=True, sort_keys=False)
    recorder = ExactTaskRecorder()
    run_started = time.time()
    for gkey, gruns in group_runs(runs).items():
        started = time.time()
        train_group(gkey, gruns, run_cfg, "cpu", str(outdir), total_steps=total,
                    ckpts=run_cfg["common"]["checkpoints"], probe=recorder,
                    probe_steps=range(0, total + 1, every))
        print(f"frozen {gkey}: {len(gruns)} seeds, {time.time()-started:.1f}s", flush=True)
    frame = recorder.dataframe()
    frame.to_csv(outdir / "task_end_metrics.csv", index=False)
    sanity = {
        "S2_bias_exact_zero": bool((frame["b_maxabs"] == 0.0).all()),
        "S3_all_finite": bool(np.isfinite(frame.select_dtypes(include=[np.number])).all().all()),
        "S3_support_identity": bool(all(
            row["n_mismatch_beyond_tol"] == 0
            and row["max_p_hat_quantization_abs_err"] == 0.0
            and row["n_nonfinite_all_stats"] == 0
            for row in recorder.sanity_rows)),
        "n_rows": int(len(frame)), "n_probes": int(frame.step.nunique()),
        "n_seeds": int(frame.seed.nunique()),
        "training_elapsed_sec": round(time.time() - run_started, 3),
    }
    (outdir / "run_sanity.json").write_text(
        json.dumps(sanity, indent=2, ensure_ascii=False), encoding="utf-8")
    if not all(sanity[k] for k in ("S2_bias_exact_zero", "S3_all_finite",
                                   "S3_support_identity")):
        raise RuntimeError(f"frozen run sanity failed: {sanity}")
    return frame


def load_free(cfg: dict) -> pd.DataFrame:
    rows = []
    for seed in cfg["common"]["seeds"]:
        path = baseline_path(cfg, int(seed))
        if not path.exists():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as z:
            p = z["layer1_p_hat"]
            for i, step in enumerate(z["step"]):
                rows.append({
                    "step": int(step), "seed": int(seed), "arm": "free",
                    "strict_dead": int((p[i] == 0).sum()),
                    "strict_dead_frac": float((p[i] == 0).mean()),
                    "eval_loss_exact": float(z["eval_loss_exact"][i]),
                    "unfit": float(z["unfit"][i]),
                })
    return pd.DataFrame(rows)


def classify_p1(frozen_mean: float, reduction: float, diff_ci: dict,
                near_zero: float, decisive_reduction: float) -> str:
    decreases = bool(diff_ci["hi"] < 0)
    if decreases and frozen_mean <= near_zero and reduction >= decisive_reduction:
        return "BIAS_ROUTE_DECISIVE"
    if decreases:
        return "BIAS_ROUTE_PARTIAL"
    if diff_ci["lo"] > 0:
        return "BIAS_FREEZE_INCREASES_DEATH"
    return "BIAS_ROUTE_NOT_SUPPORTED"


def _window_by_seed(frame: pd.DataFrame, lo_step: int, hi_step: int) -> pd.DataFrame:
    return (frame[(frame.step >= lo_step) & (frame.step <= hi_step)]
            .groupby("seed")[["strict_dead_frac", "unfit", "eval_loss_exact"]]
            .mean().sort_index())


def markdown_table(frame: pd.DataFrame) -> str:
    """Small dependency-free Markdown renderer for committed summaries."""
    columns = [str(column) for column in frame.columns]

    def render(value) -> str:
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.8g}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    rows = ["| " + " | ".join(columns) + " |",
            "| " + " | ".join("---" for _ in columns) + " |"]
    rows.extend("| " + " | ".join(render(value) for value in row) + " |"
                for row in frame.itertuples(index=False, name=None))
    return "\n".join(rows)


def analyze(cfg: dict, outdir: Path) -> dict:
    frozen = pd.read_csv(outdir / "task_end_metrics.csv")
    free_all = load_free(cfg)
    period = int(cfg["condA"]["T_values"][0])
    frozen_steps = set(frozen.step.unique())
    free = free_all[free_all.step.isin(frozen_steps)].copy()
    expected_steps = set(range(0, int(cfg["common"]["total_steps"]) + 1,
                               int(cfg["centered_freeze"]["probe_every"])))
    if frozen_steps != expected_steps:
        raise RuntimeError("frozen task-end grid is incomplete")
    if set(free.seed.unique()) != set(frozen.seed.unique()):
        raise RuntimeError("free/frozen seed sets differ")

    late0, late1 = [int(v) for v in cfg["centered_freeze"]["late_tasks"]]
    lo_step, hi_step = late0 * period, late1 * period
    fw = _window_by_seed(free, lo_step, hi_step)
    zw = _window_by_seed(frozen, lo_step, hi_step)
    seeds = fw.index.intersection(zw.index)
    final_step = int(cfg["common"]["total_steps"])
    ff = free[free.step == final_step].set_index("seed").loc[seeds]
    zf = frozen[frozen.step == final_step].set_index("seed").loc[seeds]

    rng = np.random.default_rng(int(cfg["centered_freeze"]["bootstrap_seed"]))
    B = int(cfg["centered_freeze"]["bootstrap_B"])
    dead_ci = paired_boot_ci(zw.loc[seeds, "strict_dead_frac"],
                             fw.loc[seeds, "strict_dead_frac"], rng, n_boot=B)
    final_dead_ci = paired_boot_ci(zf["strict_dead_frac"], ff["strict_dead_frac"],
                                   rng, n_boot=B)
    unfit_ci = paired_boot_ci(zw.loc[seeds, "unfit"], fw.loc[seeds, "unfit"],
                              rng, n_boot=B)
    free_dead = float(fw.loc[seeds, "strict_dead_frac"].mean())
    frozen_dead = float(zw.loc[seeds, "strict_dead_frac"].mean())
    reduction = ((free_dead - frozen_dead) / free_dead
                 if free_dead > 0 else float("nan"))
    p1 = classify_p1(
        frozen_dead, reduction, dead_ci,
        float(cfg["centered_freeze"]["near_zero_max"]),
        float(cfg["centered_freeze"]["decisive_reduction_min"]),
    )
    p2 = ("FROZEN_BETTER" if unfit_ci["hi"] < 0 else
          "FROZEN_WORSE" if unfit_ci["lo"] > 0 else "NULL")

    endpoints = pd.DataFrame({
        "seed": seeds,
        "dead_free_late": fw.loc[seeds, "strict_dead_frac"].values,
        "dead_frozen_late": zw.loc[seeds, "strict_dead_frac"].values,
        "dead_free_final": ff["strict_dead_frac"].values,
        "dead_frozen_final": zf["strict_dead_frac"].values,
        "unfit_free_late": fw.loc[seeds, "unfit"].values,
        "unfit_frozen_late": zw.loc[seeds, "unfit"].values,
        "unfit_free_final": ff["unfit"].values,
        "unfit_frozen_final": zf["unfit"].values,
    })
    endpoints.to_csv(outdir / "paired_endpoints.csv", index=False)

    verdict = pd.DataFrame([
        {"pred": "P1", "scope": "late tasks 451-500 strict_dead_frac",
         "verdict": p1,
         "evidence": (f"free {free_dead:.6f} -> frozen {frozen_dead:.6f}; "
                      f"reduction {reduction:.6f}; frozen-free {dead_ci['mean']:+.6f} "
                      f"CI [{dead_ci['lo']:+.6f}, {dead_ci['hi']:+.6f}]")},
        {"pred": "P1-final", "scope": "step 5M strict_dead_frac (supportive)",
         "verdict": "REPORT_ONLY",
         "evidence": (f"free {ff.strict_dead_frac.mean():.6f} -> "
                      f"frozen {zf.strict_dead_frac.mean():.6f}; frozen-free "
                      f"{final_dead_ci['mean']:+.6f} CI "
                      f"[{final_dead_ci['lo']:+.6f}, {final_dead_ci['hi']:+.6f}]")},
        {"pred": "P2", "scope": "late tasks 451-500 exact-support unfit",
         "verdict": p2,
         "evidence": (f"free {fw.unfit.mean():.6g} -> frozen {zw.unfit.mean():.6g}; "
                      f"frozen-free {unfit_ci['mean']:+.6g} CI "
                      f"[{unfit_ci['lo']:+.6g}, {unfit_ci['hi']:+.6g}]")},
    ])
    verdict.to_csv(outdir / "verdict.csv", index=False)

    combined = pd.concat([
        free[["step", "seed", "arm", "strict_dead_frac", "unfit"]],
        frozen[["step", "seed", "arm", "strict_dead_frac", "unfit"]],
    ], ignore_index=True)
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    for arm, color in (("free", "tab:red"), ("frozen", "tab:blue")):
        g = combined[combined.arm == arm]
        for ax, key in zip(axes, ("strict_dead_frac", "unfit")):
            summary = g.groupby("step")[key].agg(["mean", "sem"])
            ax.plot(summary.index, summary["mean"], label=arm, color=color, lw=1.5)
            ax.fill_between(summary.index, summary["mean"] - summary["sem"],
                            summary["mean"] + summary["sem"], color=color, alpha=0.2)
    axes[0].axhline(float(cfg["centered_freeze"]["near_zero_max"]), color="gray",
                    ls="--", lw=1)
    axes[0].set_ylabel("strict_dead_frac")
    axes[1].set_ylabel("exact-support unfit")
    axes[1].set_xlabel("step")
    axes[1].set_yscale("log")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle("condA centered: free vs freeze_bias")
    fig.tight_layout()
    fig.savefig(outdir / "fig_centered_freeze.png", dpi=160)
    plt.close(fig)

    summary = [
        "# centered_freeze_0901 — P-1 result", "",
        "## Verdict", "", markdown_table(verdict), "",
        "## Paired seed endpoints", "", markdown_table(endpoints), "",
        "## Interpretation", "",
    ]
    if p1 == "BIAS_ROUTE_DECISIVE":
        summary.append("- condA・centered の終盤死は、b を 0 に凍結すると事前登録した"
                       "「ほぼ消失」域まで減った。**この設定では b 経路が決定的**である。")
    elif p1 == "BIAS_ROUTE_PARTIAL":
        summary.append("- b 凍結で死は有意に減ったが、事前登録した決定打の水準には届かなかった。"
                       "**b 経路は寄与するが単独ではない**。")
    else:
        summary.append("- b 凍結は事前登録した方向で死を減らさなかった。"
                       "**centered 死の b 経路説は支持されない**。")
    summary += [
        f"- 機能の副次判定は **{p2}**。ただし freeze_bias は表現力も変えるため、"
        "dead の機能コストを単独では同定しない。",
        "- スコープは condA・1層幅100・center_alpha=0.01・T=10,000・batch=1・"
        "plain SGD・5M step に限定する。", "",
        "## Sanity", "",
        "- S0: 30k free replay は既存 L1w100_A1 と一致。",
        "- S2: frozen の全記録点で b は厳密に 0。",
        "- S3: 全支持点記録の恒等式・1/32 量子化・有限性を通過。",
    ]
    (outdir / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    return {"P1": p1, "P2": p2, "free_dead_late": free_dead,
            "frozen_dead_late": frozen_dead, "reduction": reduction,
            "dead_diff_ci": dead_ci, "final_dead_diff_ci": final_dead_ci,
            "unfit_diff_ci": unfit_ci}


def provenance(cfg_path: Path, cfg: dict, outdir: Path, analysis: dict,
               started: float, command: list[str]) -> dict:
    inputs = {str(baseline_path(cfg, int(seed)).relative_to(ROOT)):
              sha_file(baseline_path(cfg, int(seed)))
              for seed in cfg["common"]["seeds"]}
    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                           text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--short"], cwd=ROOT,
                                        text=True).splitlines()
    except (OSError, subprocess.CalledProcessError):
        git_hash, dirty = None, []
    outputs = {}
    for name in ("task_end_metrics.csv", "paired_endpoints.csv", "verdict.csv",
                 "summary.md", "run_sanity.json", "config_used.yaml",
                 "fig_centered_freeze.png"):
        path = outdir / name
        if path.exists():
            outputs[name] = sha_file(path)
    checkpoint_sha256 = {
        str(path.relative_to(outdir)): sha_file(path)
        for path in sorted((outdir / "ckpts").glob("*.pt"))
    }
    run_sanity_path = outdir / "run_sanity.json"
    run_sanity = (json.loads(run_sanity_path.read_text(encoding="utf-8"))
                  if run_sanity_path.exists() else {})
    preflight_path = Path(ROOT) / "results" / "_preflight_centered_freeze_0901" / "preflight.json"
    return {
        "experiment": "centered_freeze_0901",
        "created": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "training_command": "OMP_NUM_THREADS=1 .venv/bin/python -m src.centered_freeze_0901",
        "analysis_command": command,
        "analysis_elapsed_sec": round(time.time() - started, 3),
        "training_elapsed_sec": run_sanity.get("training_elapsed_sec"),
        "cwd": os.getcwd(), "python": sys.version, "platform": platform.platform(),
        "torch": torch.__version__, "numpy": np.__version__, "device": "cpu",
        "git_hash": git_hash, "git_dirty": dirty,
        "config": str(cfg_path), "config_sha256": sha_file(cfg_path),
        "spec": cfg["spec"], "spec_sha256": sha_file(Path(ROOT) / cfg["spec"]),
        "preflight": str(preflight_path.relative_to(ROOT)),
        "preflight_sha256": sha_file(preflight_path) if preflight_path.exists() else None,
        "baseline_inputs": inputs, "analysis": analysis, "output_sha256": outputs,
        "checkpoint_sha256": checkpoint_sha256,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--outdir")
    args = parser.parse_args()
    if sum((args.preflight, args.smoke, args.analyze_only)) > 1:
        parser.error("stage flags are mutually exclusive")
    cfg_path = Path(args.config).resolve()
    cfg = load_config(str(cfg_path))
    validate_config(cfg, full=not args.smoke and not args.preflight)
    main_dir = Path(ROOT) / cfg["centered_freeze"]["output_dir"]
    preflight_dir = Path(ROOT) / "results" / "_preflight_centered_freeze_0901"
    smoke_dir = Path(ROOT) / "results" / "_smoke_centered_freeze_0901"
    outdir = (Path(args.outdir).resolve() if args.outdir else
              preflight_dir if args.preflight else smoke_dir if args.smoke else main_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    if args.preflight:
        preflight(cfg, outdir)
        return
    if args.smoke:
        run_frozen(cfg, outdir, smoke=True)
        print(f"SMOKE PASS -> {outdir}", flush=True)
        return
    preflight_file = preflight_dir / "preflight.json"
    if not preflight_file.exists():
        raise FileNotFoundError("run --preflight before the full run")
    preflight_result = json.loads(preflight_file.read_text(encoding="utf-8"))
    if not preflight_result.get("pass_"):
        raise RuntimeError("saved S0 preflight did not pass")
    started = time.time()
    if not args.analyze_only:
        run_frozen(cfg, outdir, smoke=False)
    result = analyze(cfg, outdir)
    (outdir / "provenance.json").write_text(
        json.dumps(provenance(cfg_path, cfg, outdir, result, started, sys.argv),
                   indent=2, ensure_ascii=False), encoding="utf-8")
    print(pd.read_csv(outdir / "verdict.csv").to_string(index=False), flush=True)
    print(f"ALL DONE -> {outdir}", flush=True)


if __name__ == "__main__":
    main()
