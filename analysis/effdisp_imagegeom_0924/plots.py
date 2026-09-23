"""Static figures for seed-paired image geometry summaries."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

GEOMETRIES = ("rgb32", "dup64", "dup64_half", "gray32", "avg16")
OPTIMIZERS = ("adam", "sgd")
COLORS = {"rgb32": "#242424", "dup64": "#cc5a30", "dup64_half": "#d39724",
          "gray32": "#3576a8", "avg16": "#518c61"}


def _save(fig, path: Path) -> Path:
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _trajectory(transitions: pd.DataFrame, column: str, ylabel: str,
                path: Path) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True, layout="constrained")
    for ax, optimizer in zip(axes, OPTIMIZERS):
        block = transitions[transitions.optimizer == optimizer]
        for geometry in GEOMETRIES:
            part = block[block.geometry == geometry]
            med = part.groupby("task")[column].median()
            ax.plot(med.index, med.to_numpy(), color=COLORS[geometry], lw=1.5,
                    label=geometry)
        ax.set_title(optimizer.upper())
        ax.set_ylabel(ylabel)
        ax.grid(alpha=.2)
        ax.legend(ncol=5, fontsize=8)
    axes[-1].set_xlabel("endpoint task")
    fig.suptitle("First-layer effective width and displacement on the fixed image bank")
    return _save(fig, path)


def _paired(contrasts: pd.DataFrame, path: Path) -> Path:
    part = contrasts[(contrasts.contrast == "geometry_over_rgb32")
                     & (contrasts.metric == "d_eff")]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, layout="constrained")
    for i, optimizer in enumerate(OPTIMIZERS):
        for j, window in enumerate(("early", "late")):
            ax = axes[i, j]
            block = part[(part.optimizer == optimizer) & (part.window == window)]
            for x, geometry in enumerate(GEOMETRIES[1:]):
                values = block[block.geometry == geometry].sort_values("seed").ratio.to_numpy(dtype=float)
                ax.scatter(np.full(len(values), x), values, color=COLORS[geometry],
                           s=22, alpha=.7)
                if np.isfinite(values).any():
                    ax.plot([x-.2, x+.2], [np.nanmedian(values)]*2, color="black", lw=1.2)
            ax.axhline(1, color="#777777", lw=.8, ls="--")
            ax.set_xticks(range(4), GEOMETRIES[1:], rotation=25, ha="right")
            ax.set_title(f"{optimizer.upper()} · tasks {6 if window == 'early' else 41}–{15 if window == 'early' else 50}")
            ax.set_ylabel("paired d_eff / rgb32 d_eff")
            ax.grid(alpha=.2)
    fig.suptitle("Five paired seeds per geometry and optimizer")
    return _save(fig, path)


def _spectrum(spectrum: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    for geometry in GEOMETRIES:
        part = spectrum[(spectrum.geometry == geometry) & (spectrum.component <= 100)]
        med = part.groupby("component").cov_eigenvalue.median()
        ax.plot(med.index, med.to_numpy(), color=COLORS[geometry], lw=1.5,
                label=geometry)
    ax.set_yscale("log")
    ax.set_xlabel("principal component")
    ax.set_ylabel("population covariance eigenvalue")
    ax.set_title("Image bank covariance spectrum, median over five seeds")
    ax.legend(ncol=3)
    ax.grid(alpha=.2)
    return _save(fig, path)


def _activity(seed_summary: pd.DataFrame, path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, layout="constrained")
    for ax, optimizer in zip(axes, OPTIMIZERS):
        part = seed_summary[seed_summary.optimizer == optimizer]
        for geometry in GEOMETRIES:
            arm = part[part.geometry == geometry]
            ax.scatter(arm.late_activity, arm.late_nearbalance, color=COLORS[geometry],
                       s=28, alpha=.75, label=geometry)
        ax.axvline(.1, color="#555555", lw=.7, ls=":")
        ax.axhline(.9, color="#555555", lw=.7, ls=":")
        ax.axhline(1.1, color="#555555", lw=.7, ls=":")
        ax.set_title(optimizer.upper())
        ax.set_xlabel("late sum Q / mean V")
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("late -2 sum X / sum Q")
    fig.suptitle("Activity and near balance are separate conditions")
    return _save(fig, path)


def _top10(windows: pd.DataFrame, path: Path) -> Path:
    part = windows[windows.window == "late"]
    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    centers = np.arange(len(GEOMETRIES))
    width = .36
    for shift, optimizer in ((-width/2, "adam"), (width/2, "sgd")):
        vals = [part[(part.geometry == geometry) & (part.optimizer == optimizer)]
                .top10_Q_fraction.median() for geometry in GEOMETRIES]
        ax.bar(centers + shift, vals, width=width, label=optimizer.upper(),
               color="#3379a8" if optimizer == "adam" else "#d1763a")
    ax.set_xticks(centers, GEOMETRIES)
    ax.set_ylim(0, 1)
    ax.set_ylabel("fraction of Q in top 10 bank PCs")
    ax.set_title("Late tasks 41–50, seed-median of within-seed medians")
    ax.legend()
    ax.grid(axis="y", alpha=.2)
    return _save(fig, path)


def make_plots(transitions: pd.DataFrame, seed_summary: pd.DataFrame,
               windows: pd.DataFrame, contrasts: pd.DataFrame,
               covstats: pd.DataFrame, spectrum: pd.DataFrame, out: Path) -> list[Path]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    figures = [_spectrum(spectrum, out / "covariance_spectrum.png")]
    if not transitions.empty:
        figures.extend([_trajectory(transitions, "R_next", "R = sqrt(V)", out / "effective_width.png"),
                        _trajectory(transitions, "d_eff", "d_eff = sqrt(Q)", out / "effective_displacement.png")])
    if not windows.empty and windows.available.any():
        figures.extend([_paired(contrasts, out / "paired_displacement_ratios.png"),
                        _activity(seed_summary, out / "activity_balance.png"),
                        _top10(windows, out / "top10_Q_fraction.png")])
    return figures
