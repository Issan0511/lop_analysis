"""Test whether fast fitting extinguishes the growth of ``dead_frac``.

The analysis reads the existing ``center_selfcov_0814`` CSVs and deterministically
replays only the condA input stream to recover the exact 32-point target variance
for each task.  It never trains or updates a network.

Run from the repository root::

    OMP_NUM_THREADS=1 .venv/bin/python -m analysis.fit_death_race.analyze
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from src.ratchet_log import full_support_ro, teacher_f64
from src.train import setup_group


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results" / "center_selfcov_0814"
DEFAULT_OUT = ROOT / "results" / "fit_death_race_0830"
WIDTHS = (5, 100)
PERIOD = 10_000
N_TASKS = 100
SEEDS = tuple(range(5))


def load_config() -> dict:
    with (SOURCE / "config_used.yaml").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def replay_signal_variance(width: int, cfg: dict) -> pd.DataFrame:
    """Replay inputs (not learning) and return exact Var[y] per task and seed."""
    ckpt = torch.load(
        SOURCE / "ckpts" / f"A_w{width}_step1000000.pt",
        map_location="cpu",
        weights_only=False,
    )
    runs = ckpt["runs"]
    st = setup_group(("A", width, 1, "none"), runs, cfg, torch.device("cpu"))
    rows: list[dict] = []
    for task in range(1, N_TASKS + 1):
        # Exactly reproduces the input-generator consumption of 10,000 online
        # steps while avoiding all network forward/backward/update operations.
        st["env"].segment(PERIOD)
        X = full_support_ro(st["env"]).double()
        y = teacher_f64(st["teacher"], X)
        signal_var = y.var(dim=0, unbiased=False)
        for run_index, run in enumerate(runs):
            if run["enc"] != "centered":
                continue
            rows.append(dict(
                width=width,
                seed=int(run["seed"]),
                task=task,
                signal_var=float(signal_var[run_index]),
            ))

    if st["env"].t != int(ckpt["env"]["t"]):
        raise RuntimeError(f"w{width}: replay t mismatch")
    if not torch.equal(st["env"].flip_state, ckpt["env"]["flip_state"]):
        raise RuntimeError(f"w{width}: replay final flip_state mismatch")
    return pd.DataFrame(rows)


def load_series(cfg: dict) -> pd.DataFrame:
    frames = []
    for width in WIDTHS:
        raw = pd.read_csv(SOURCE / f"lop_metrics_A_w{width}.csv")
        raw = raw[raw["run_id"].str.contains("_centered_") & (raw["step"] > 0)].copy()
        raw["width"] = width
        raw["seed"] = raw["run_id"].str.extract(r"_s(\d+)$")[0].astype(int)
        raw["task"] = np.ceil(raw["step"] / PERIOD).astype(int)
        signal = replay_signal_variance(width, cfg)
        raw = raw.merge(signal, on=["width", "seed", "task"], validate="many_to_one")
        raw["nmse"] = raw["eval_loss"] / raw["signal_var"].clip(lower=1e-12)
        raw["log10_nmse"] = np.log10(raw["nmse"].clip(lower=1e-12))
        frames.append(raw[[
            "width", "seed", "step", "task", "eval_loss", "signal_var",
            "nmse", "log10_nmse", "dead_frac",
        ]])
    data = pd.concat(frames, ignore_index=True).sort_values(["width", "seed", "step"])
    expected = len(WIDTHS) * len(SEEDS) * 1000
    if len(data) != expected:
        raise RuntimeError(f"unexpected series length: {len(data)} != {expected}")
    return data


def ols_slope(group: pd.DataFrame, first_task: int, last_task: int) -> float:
    x = group.loc[group["task"].between(first_task, last_task), "task"].to_numpy(float)
    y = group.loc[group["task"].between(first_task, last_task), "dead_frac"].to_numpy(float)
    return float(np.polyfit(x, y, 1)[0])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(pd.Series(x).rank().corr(pd.Series(y).rank()))


def summarize(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    task_end = data[data["step"] % PERIOD == 0].copy()
    seed_rows = []
    slope_rows = []
    for (width, seed), group in task_end.groupby(["width", "seed"], sort=True):
        group = group.sort_values("task")
        early = group[group["task"].between(1, 10)]
        row1 = group[group["task"] == 1].iloc[0]
        row50 = group[group["task"] == 50].iloc[0]
        row100 = group[group["task"] == 100].iloc[0]
        first_fit = group.loc[group["nmse"] <= 0.01, "task"]
        seed_rows.append(dict(
            width=int(width), seed=int(seed),
            task1_eval_loss=float(row1.eval_loss),
            task1_nmse=float(row1.nmse),
            early_median_nmse_t1_10=float(np.median(early["nmse"])),
            task1_dead=float(row1.dead_frac),
            task50_dead=float(row50.dead_frac),
            final_dead=float(row100.dead_frac),
            final_eval_loss=float(row100.eval_loss),
            final_nmse=float(row100.nmse),
            first_task_nmse_le_001=(int(first_fit.iloc[0]) if len(first_fit) else None),
        ))
        for window, lo, hi in (("t1_50", 1, 50), ("t51_100", 51, 100),
                               ("t76_100", 76, 100)):
            slope_rows.append(dict(
                width=int(width), seed=int(seed), window=window,
                dead_frac_slope_per_task=ols_slope(group, lo, hi),
            ))

    seeds = pd.DataFrame(seed_rows)
    slopes = pd.DataFrame(slope_rows)
    correlations = []
    for width, group in seeds.groupby("width"):
        correlations.append(dict(
            width=int(width),
            statistic="spearman_early_nmse_vs_final_dead",
            value=spearman(
                group["early_median_nmse_t1_10"].to_numpy(float),
                group["final_dead"].to_numpy(float),
            ),
            n_seed=len(group),
        ))

    # Exploratory within-task temporal association.  The first 1k probe and the
    # mean normalized loss in a task are compared with that task's net change in
    # dead_frac. This is not a causal estimate: task difficulty and time are
    # shared causes, and the support-based dead label can change identity.
    for (width, seed), group in data.groupby(["width", "seed"]):
        by_task = group.sort_values("step").groupby("task").agg(
            first_nmse=("nmse", "first"),
            mean_nmse=("nmse", "mean"),
            end_dead=("dead_frac", "last"),
        ).reset_index()
        dead_change = by_task["end_dead"].diff().to_numpy(float)[1:]
        correlations.append(dict(
            width=int(width), seed=int(seed),
            statistic="spearman_first1k_nmse_vs_within_task_dead_change",
            value=spearman(
                by_task["first_nmse"].to_numpy(float)[1:], dead_change,
            ),
            n_seed=1,
        ))
        correlations.append(dict(
            width=int(width), seed=int(seed),
            statistic="spearman_mean_nmse_vs_within_task_dead_change",
            value=spearman(
                by_task["mean_nmse"].to_numpy(float)[1:], dead_change,
            ),
            n_seed=1,
        ))
    corr = pd.DataFrame(correlations)

    late = slopes[slopes["window"] == "t51_100"]
    w5 = late[late["width"] == 5]["dead_frac_slope_per_task"].to_numpy(float)
    w100 = late[late["width"] == 100]["dead_frac_slope_per_task"].to_numpy(float)
    med5, med100 = float(np.median(w5)), float(np.median(w100))
    rng = np.random.default_rng(20260830)
    B = 20_000
    draw5 = np.median(w5[rng.integers(0, len(w5), (B, len(w5)))], axis=1)
    draw100 = np.median(w100[rng.integers(0, len(w100), (B, len(w100)))], axis=1)
    diff = draw5 - draw100
    lo, hi = np.quantile(diff, [0.025, 0.975])
    summary = dict(
        late_window="task 51-100",
        w5_median_dead_slope_per_task=med5,
        w100_median_dead_slope_per_task=med100,
        slope_ratio_w5_over_w100=med5 / med100,
        median_slope_difference_w5_minus_w100=med5 - med100,
        bootstrap_difference_ci95=[float(lo), float(hi)],
        bootstrap_B=B,
        w100_mean_dead_task1=float(seeds.loc[seeds.width == 100, "task1_dead"].mean()),
        w100_mean_dead_task50=float(seeds.loc[seeds.width == 100, "task50_dead"].mean()),
        w100_mean_dead_task100=float(seeds.loc[seeds.width == 100, "final_dead"].mean()),
        w5_mean_dead_task50=float(seeds.loc[seeds.width == 5, "task50_dead"].mean()),
        w5_mean_dead_task100=float(seeds.loc[seeds.width == 5, "final_dead"].mean()),
    )
    for width in WIDTHS:
        width_all = data[data.width == width]
        width_end = task_end[task_end.width == width]
        first1k = (width_all.sort_values("step").groupby(["seed", "task"], as_index=False)
                   .first())
        summary[f"w{width}_task_end_nmse_median"] = float(width_end.nmse.median())
        summary[f"w{width}_task_end_nmse_max"] = float(width_end.nmse.max())
        summary[f"w{width}_first1k_nmse_median"] = float(first1k.nmse.median())
        summary[f"w{width}_first1k_nmse_max"] = float(first1k.nmse.max())
    return task_end, seeds, slopes, corr, summary


def plot(data: pd.DataFrame, seeds: pd.DataFrame, out: Path) -> None:
    task_end = data[data["step"] % PERIOD == 0]
    colors = {5: "#bf616a", 100: "#5e81ac"}
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.5))
    for width in WIDTHS:
        group = task_end[task_end.width == width]
        pivot_nmse = group.pivot(index="task", columns="seed", values="nmse")
        pivot_dead = group.pivot(index="task", columns="seed", values="dead_frac")
        for pivot, ax in ((pivot_nmse, axes[0, 0]), (pivot_dead, axes[0, 1])):
            med = pivot.median(axis=1)
            lo = pivot.quantile(0.25, axis=1)
            hi = pivot.quantile(0.75, axis=1)
            ax.plot(pivot.index, med, color=colors[width], label=f"w{width}", lw=2)
            ax.fill_between(pivot.index, lo, hi, color=colors[width], alpha=0.18)
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_ylabel("task-end eval_loss / Var(y)")
    axes[0, 0].set_xlabel("task")
    axes[0, 0].legend(frameon=False)
    axes[0, 1].set_ylabel("task-end dead_frac")
    axes[0, 1].set_xlabel("task")
    axes[0, 1].legend(frameon=False)

    for width, marker in ((5, "o"), (100, "s")):
        group = seeds[seeds.width == width]
        axes[1, 0].scatter(
            group.early_median_nmse_t1_10, group.final_dead,
            color=colors[width], marker=marker, s=55, label=f"w{width}",
        )
        for row in group.itertuples():
            axes[1, 0].annotate(str(row.seed), (row.early_median_nmse_t1_10, row.final_dead),
                                xytext=(4, 3), textcoords="offset points", fontsize=8)
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_xlabel("median normalized MSE, tasks 1-10")
    axes[1, 0].set_ylabel("final dead_frac")
    axes[1, 0].legend(frameon=False)

    late = task_end[task_end.task.between(51, 100)]
    vals = []
    for width in WIDTHS:
        for _, group in late[late.width == width].groupby("seed"):
            vals.append((width, ols_slope(group, 51, 100)))
    for j, width in enumerate(WIDTHS):
        y = np.array([value for w, value in vals if w == width])
        axes[1, 1].scatter(np.full(len(y), j), y, color=colors[width], s=50)
        axes[1, 1].plot([j - 0.18, j + 0.18], [np.median(y)] * 2,
                        color="black", lw=2)
    axes[1, 1].axhline(0, color="#777777", lw=0.8)
    axes[1, 1].set_xticks([0, 1], ["w5", "w100"])
    axes[1, 1].set_ylabel("dead_frac slope / task (tasks 51-100)")

    fig.suptitle("Fit-versus-death race: existing centered runs", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "fit_death_race.png", dpi=180)
    fig.savefig(out / "fit_death_race.pdf")
    plt.close(fig)


def write_report(seeds: pd.DataFrame, corr: pd.DataFrame, summary: dict, out: Path) -> None:
    w100 = seeds[seeds.width == 100]
    w5 = seeds[seeds.width == 5]
    seed_corr = corr[corr.statistic == "spearman_early_nmse_vs_final_dead"]
    temporal_early = corr[
        corr.statistic == "spearman_first1k_nmse_vs_within_task_dead_change"
    ].groupby("width")["value"].median()
    temporal_mean = corr[
        corr.statistic == "spearman_mean_nmse_vs_within_task_dead_change"
    ].groupby("width")["value"].median()
    lines = [
        "# Fit–death race validation (post-hoc, no training)", "",
        "## Verdict", "",
        "- **Strong form rejected:** fitting does not create an absorbing state in which `dead_frac` stops.",
        "- **Weak rate form supported descriptively:** the late fractional-death trend is slower at w100, but remains positive.",
        "- The CSV alone cannot establish causal direction or bistability; `dead_frac` is a task-dependent support label, not unit identity survival.", "",
        "## Decisive numbers", "",
        f"- w100: all task-1 raw eval losses are at most {w100.task1_eval_loss.max():.5g}; mean dead_frac moves "
        f"{summary['w100_mean_dead_task1']:.3f} -> {summary['w100_mean_dead_task50']:.3f} -> "
        f"{summary['w100_mean_dead_task100']:.3f} at tasks 1, 50, 100.",
        f"- Across all 100 tasks, median task-end normalized MSE is {summary['w100_task_end_nmse_median']:.5f} at w100 "
        f"versus {summary['w5_task_end_nmse_median']:.3f} at w5. Task switches reheat the residual: at the first 1k probe "
        f"the medians are {summary['w100_first1k_nmse_median']:.4f} and {summary['w5_first1k_nmse_median']:.3f}, respectively.",
        f"- Late median OLS slope (tasks 51-100): w5={summary['w5_median_dead_slope_per_task']:.6f}, "
        f"w100={summary['w100_median_dead_slope_per_task']:.6f} dead_frac/task; ratio="
        f"{summary['slope_ratio_w5_over_w100']:.2f}x.",
        f"- Difference w5-w100={summary['median_slope_difference_w5_minus_w100']:.6f}, independent seed-cluster "
        f"bootstrap 95% CI [{summary['bootstrap_difference_ci95'][0]:.6f}, "
        f"{summary['bootstrap_difference_ci95'][1]:.6f}] (B={summary['bootstrap_B']:,}).", "",
        "## Seed ordering", "",
        "Early fit is the median normalized MSE over tasks 1-10; normalization is eval_loss / exact Var(y).", "",
        "| width | seed | early normalized MSE | final dead_frac |",
        "|---:|---:|---:|---:|",
    ]
    for row in seeds.sort_values(["width", "seed"]).itertuples():
        lines.append(f"| {row.width} | {row.seed} | {row.early_median_nmse_t1_10:.6g} | {row.final_dead:.3f} |")
    lines += ["",
        f"- Spearman(early error, final dead_frac): w5={float(seed_corr.loc[seed_corr.width == 5, 'value'].iloc[0]):.3f}, "
        f"w100={float(seed_corr.loc[seed_corr.width == 100, 'value'].iloc[0]):.3f} (n=5 each; descriptive only).",
        f"- The proposed seed-3 versus seed-0 ordering holds for w5. It is not monotone over all five seeds: seed 4 has the smallest task-1 normalized error ({w5.loc[w5.seed == 4, 'task1_nmse'].iloc[0]:.4f}) but ends at dead_frac=0.8, and seed 2 has the worst early-10-task error but ends at 0.6.",
        f"- Median seed-wise Spearman(first-1k NMSE, same-task net dead change): "
        f"w5={temporal_early.loc[5]:.3f}, w100={temporal_early.loc[100]:.3f}; using the task mean gives "
        f"w5={temporal_mean.loc[5]:.3f}, w100={temporal_mean.loc[100]:.3f}. These weak associations are exploratory and non-causal.", "",
        "## Replay validation", "",
        "For each width, only the input generator was advanced in 10,000-step segments. After 100 segments, `env.t` and the full final `flip_state` matched the saved 1M checkpoint exactly. No network training or update was executed.",
    ]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    cfg = load_config()
    data = load_series(cfg)
    task_end, seeds, slopes, corr, summary = summarize(data)
    data.to_csv(args.out / "normalized_series.csv", index=False)
    task_end.to_csv(args.out / "task_end_series.csv", index=False)
    seeds.to_csv(args.out / "per_seed_summary.csv", index=False)
    slopes.to_csv(args.out / "dead_slope_by_seed.csv", index=False)
    corr.to_csv(args.out / "correlations.csv", index=False)
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    plot(data, seeds, args.out)
    write_report(seeds, corr, summary, args.out)


if __name__ == "__main__":
    main()
