#!/usr/bin/env python3
"""Standalone research figures: paired-seed conditions, no fitted coefficients."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/chargpt_phase2conditions_1024'


def main():
    df = pd.read_csv(OUT/'states.csv')
    df = df[(df.layer == 0) & (df.metric == 'SGD') & df.arm.isin(['C3', 'ReLUcap'])]
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axs = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    panels = [('P2_ba', 'Phase-2 vs CE: balanced accuracy'),
              ('fit_error', 'Whitened distance to fixed quadratic LS'),
              ('Zhat', 'Gaussian-model Z (64-probe estimate)'),
              ('ce_relative', 'CE correction / quadratic push (L2 norm)'),
              ('task_departure_relative', 'Task departure / context mean (L2 norm)'),
              ('pmax', 'Mean maximum predicted probability')]
    for ax, (key, title) in zip(axs.flat, panels):
        for arm, color in [('C3', '#0068a0'), ('ReLUcap', '#bb5b14')]:
            vals = df[df.arm == arm].pivot(index='task', columns='seed', values=key)
            ax.fill_between(vals.index, vals.min(axis=1), vals.max(axis=1), color=color, alpha=.14)
            ax.plot(vals.index, vals.median(axis=1), color=color, marker='o', markersize=3, label=arm)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Task endpoint')
        ax.set_xticks([1, 3, 5, 7, 9])
        ax.axvspan(4.5, 5.5, color='grey', alpha=.07)
        ax.grid(alpha=.18)
        if key == 'P2_ba':
            ax.axhline(.5, color='grey', ls='--', lw=1)
            ax.set_ylim(0, 1)
    axs[0, 0].legend(frameon=False)
    fig.suptitle('L0 own-up-row push, SGD: line = median, band = range across 10 seeds', fontsize=12)
    for ext in ['png', 'svg']:
        fig.savefig(OUT/f'conditions.{ext}', dpi=180)
    plt.close(fig)
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    panels = [('P2_vs_ls_task_ba', 'Phase-2 mean vs individual LS task'),
              ('ls_task_ba', 'Individual LS task vs actual CE'),
              ('sq_full_ba', 'Actual-head quadratic push vs actual CE')]
    for ax, (key, title) in zip(axs, panels):
        for arm, color in [('C3', '#0068a0'), ('ReLUcap', '#bb5b14')]:
            vals = df[df.arm == arm].pivot(index='task', columns='seed', values=key)
            ax.fill_between(vals.index, vals.min(axis=1), vals.max(axis=1), color=color, alpha=.14)
            ax.plot(vals.index, vals.median(axis=1), color=color, marker='o', markersize=3, label=arm)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Task endpoint')
        ax.set_ylim(0, 1)
        ax.axhline(.5, color='grey', ls='--', lw=1)
        ax.grid(alpha=.18)
    axs[0].legend(frameon=False)
    fig.suptitle('Direction agreement through the three approximation steps; 10 seeds', fontsize=12)
    for ext in ['png', 'svg']:
        fig.savefig(OUT/f'approximation_steps.{ext}', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
