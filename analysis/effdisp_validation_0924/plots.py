"""Static, seed-level visual summaries of effective-displacement CSV outputs."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _grid(n: int):
    cols = min(4, max(1, n))
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.1 * cols, 3.0 * rows),
                             squeeze=False, layout="constrained")
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    return fig, axes.ravel()


def plot_growth(transitions: pd.DataFrame, out: Path) -> list[Path]:
    """All arms: raw and centered/reference/actual V by endpoint task."""
    paths = []
    for environment, block in transitions.groupby("environment"):
        arms = sorted(block.arm.unique())
        fig, axes = _grid(len(arms))
        for ax, arm in zip(axes, arms):
            part = block[block.arm == arm]
            for metric, color, linestyle in (("raw", "#777777", "--"),
                                             ("effective", "#2070a8", "-"),
                                             ("reference", "#2070a8", "-"),
                                             ("actual", "#cb6528", ":")):
                selected = part[part.metric == metric]
                if selected.empty:
                    continue
                med = selected.groupby("task").V_next.median()
                ax.plot(med.index, med.to_numpy(), color=color, ls=linestyle,
                        label=metric, lw=1.5)
            ax.set_title(arm)
            ax.set_xlabel("endpoint task")
            ax.set_ylabel("V = squared width / raw norm")
            ax.set_yscale("symlog", linthresh=1e-8)
            ax.grid(alpha=.2)
        axes[0].legend(fontsize=8)
        fig.suptitle(f"{environment}: raw and effective first-layer growth")
        path = out / f"growth_{environment}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_paired(summary: pd.DataFrame, paired: pd.DataFrame, out: Path) -> Path | None:
    conda = summary[(summary.environment == "conda") & (summary.metric == "effective")]
    if conda.empty:
        return None
    arms = sorted(conda.arm.unique())
    arm_groups = ([a for a in arms if not a.startswith("m40_")],
                  [a for a in arms if a.startswith("m40_")])
    fig, axes = plt.subplots(1, 2, figsize=(max(9, len(arms) * .7), 4), layout="constrained")
    for ax, col, title in ((axes[0], "late_mean_sqrtQ", "late effective displacement"),
                           (axes[1], "late_mean_sigma", "late effective width")):
        for seed, part in conda.groupby("seed"):
            ordered = part.set_index("arm").reindex(arms)
            for group in arm_groups:
                if not group:
                    continue
                idx = [arms.index(arm) for arm in group]
                first_group = next(g for g in arm_groups if g)
                ax.plot(idx, ordered.iloc[idx][col], marker="o", ms=2.5,
                        lw=.8, alpha=.5, label=f"seed {seed}" if group is first_group else None)
        ax.set_xticks(range(len(arms)), arms, rotation=65, ha="right")
        ax.set_title(title)
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    fig.suptitle("CondA: paired seeds within input dimension; m40 teacher and initialization differ")
    path = out / "paired_D_width_conda.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_activity(summary: pd.DataFrame, out: Path) -> Path | None:
    main = summary[summary.metric.isin(["effective", "reference"])]
    if main.empty:
        return None
    envs = sorted(main.environment.unique())
    fig, axes = _grid(len(envs))
    for ax, env in zip(axes, envs):
        part = main[main.environment == env]
        for arm, block in part.groupby("arm"):
            ax.scatter(block.late_activity, block.late_nearbalance,
                       s=15, alpha=.75, label=arm)
        ax.axvline(.1, color="black", lw=.7, ls=":")
        ax.axhline(.9, color="black", lw=.7, ls=":")
        ax.axhline(1.1, color="black", lw=.7, ls=":")
        ax.set_title(env)
        ax.set_xlabel("late sum Q / mean V")
        ax.set_ylabel("late -2 sum X / sum Q")
        if (part.late_activity > 0).all():
            ax.set_xscale("log")
        ax.grid(alpha=.2)
        ax.legend(fontsize=6, ncol=2)
    fig.suptitle("Activity and balance are separate conditions")
    path = out / "activity_balance.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_closure(closure: pd.DataFrame, out: Path) -> list[Path]:
    paths = []
    if closure.empty:
        return paths
    for environment, block in closure.groupby("environment"):
        arms = sorted(block.arm.unique())
        fig, axes = _grid(len(arms))
        for ax, arm in zip(axes, arms):
            part = block[block.arm == arm]
            for col, color, label in (("observed_V", "#222222", "observed"),
                                      ("model_V", "#2070a8", "closure"),
                                      ("constant_V", "#999999", "constant")):
                med = part.groupby("task")[col].median()
                ax.plot(med.index, med.to_numpy(), color=color, lw=1.3, label=label)
            ax.set_title(arm)
            ax.set_xlabel("held-out endpoint task")
            ax.set_ylabel("V")
            ax.grid(alpha=.2)
        axes[0].legend(fontsize=8)
        fig.suptitle(f"{environment}: calibration-only closure vs held-out width")
        path = out / f"closure_{environment}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths


def make_plots(transitions: pd.DataFrame, summary: pd.DataFrame,
               closure: pd.DataFrame, paired: pd.DataFrame, out: Path) -> list[Path]:
    out = Path(out)
    paths = plot_growth(transitions, out)
    paths += plot_closure(closure, out)
    for path in (plot_paired(summary, paired, out), plot_activity(summary, out)):
        if path is not None:
            paths.append(path)
    return paths
