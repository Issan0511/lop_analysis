"""Render the audit's measured results; no fitting or model selection."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def number(row, key):
    return float(row[key])


def select(rows, **values):
    return [r for r in rows if all(
        r[k] == v if isinstance(v, str) else np.isclose(float(r[k]), v, rtol=1e-10, atol=1e-14)
        for k, v in values.items())]


def panel(ax, title, xlabel, ylabel):
    ax.set_title(title, loc='left', fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=.18)
    ax.spines[['top', 'right']].set_visible(False)


def save(fig, out, name):
    fig.savefig(out / (name + '.png'), dpi=180, bbox_inches='tight')
    fig.savefig(out / (name + '.pdf'), bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=Path('results/logistic_v_height_0929'))
    out = parser.parse_args().out
    plt.rcParams.update({'font.size': 11, 'axes.labelsize': 11, 'legend.fontsize': 9,
                         'pdf.fonttype': 42, 'savefig.facecolor': 'white'})
    scalar = read(out / 'scalar_sweep.csv')
    warm = read(out / 'scalar_warm.csv')
    anis = read(out / 'anisotropic_end.csv')
    saved = read(out / 'saved_summary.csv')
    colors = ['#2465a8', '#cd6836', '#258570', '#85629c']
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 9.1), layout='constrained')
    ax = axes[0, 0]
    for opt, label, color, style in zip(
            ['adam', 'gd', 'instant', 'no_first'],
            ['Adam (.9, .999)', 'Gradient descent', 'Instant normalization', 'Adam (0, .999)'],
            colors, ['-', '-', '--', ':']):
        rows = sorted(select(scalar, group='core', optimizer=opt, eps=0., t=4000),
                      key=lambda r: number(r, 'v'))
        ax.plot([number(r, 'v') for r in rows], [number(r, 'z') for r in rows],
                label=label, color=color, linestyle=style, linewidth=2)
    ax.set_xscale('log')
    panel(ax, 'A  Scalar height: optimizer-dependent', 'Frozen readout v', 'Height z at step 4000')
    ax.set_ylim(-.1, 4.2)
    ax.legend(loc='lower left')
    ax.text(.97, .97, r'$z_0=0,\ \eta=0.001,\ \epsilon=0$',
            transform=ax.transAxes, va='top', ha='right', fontsize=10,
            bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .9})

    ax = axes[0, 1]
    for target, label, color in [(1., 'Separable target = 1', colors[0]),
                                  (.8, 'Finite optimum: target = .8', colors[1])]:
        rows = sorted(select(warm, target=target, moment_policy='retained', t_after=4000),
                      key=lambda r: number(r, 'gain'))
        ax.plot([number(r, 'gain') for r in rows], [number(r, 'delta_z') for r in rows],
                'o-', color=color, label=label, linewidth=2)
    ax.set_xscale('log')
    ax.set_xticks([.1, 1, 10], ['0.1', '1', '10'])
    ax.axhline(0, color='.4', linewidth=1)
    panel(ax, 'B  Intervention: scalar limitation', 'Readout multiplier after step 4000',
          'Height change over next 4000 steps')
    ax.legend(loc='lower left')
    ax.text(.97, .97, 'Adam moments retained', transform=ax.transAxes,
            ha='right', va='top', fontsize=10)

    ax = axes[1, 0]
    for opt, label, color, style in [('adam', 'Adam: A = 10', colors[0], 'o-'),
                                    ('instantaneous', 'Instant normalization: A = 10', colors[2], 's--')]:
        rows = sorted(select(anis, A=10., p=1/64, optimizer=opt), key=lambda r: number(r, 'v'))
        ax.plot([number(r, 'v') for r in rows], [number(r, 'centered_tail_over_s') for r in rows],
                style, color=color, label=label, linewidth=2)
    ax.axhline(1, color=colors[3], linestyle=':', linewidth=2, label='A = 1: both optimizers')
    ax.set_xscale('log')
    ax.set_xticks([.1, 1, 10], ['0.1', '1', '10'])
    panel(ax, 'C  Four-point construction (post-hoc)', 'Frozen readout v',
          r'Centered tail $T=(\max z-\mathrm{median}\ z)/s$')
    ax.legend(loc='center left')
    ax.text(.97, .97, 'p = 1/64; step 4000', transform=ax.transAxes,
            ha='right', va='top', fontsize=10)
    ax.set_ylim(.65, 7)

    ax = axes[1, 1]
    for seed, color in [(0, colors[0]), (1, colors[1])]:
        for cohort, style, name in [('original_switch_alive', 'o-', 'switch-alive units'),
                                     ('all_units', 's--', 'all 100 units')]:
            rows = [select(saved, arm=arm, seed=str(seed), cohort=cohort)[0]
                    for arm in ['VF0.1', 'VF1', 'VF10']]
            ax.plot([.1, 1, 10], [number(r, 'tail') for r in rows], style, color=color,
                    label=f'Seed {seed}, {name}', linewidth=1.7, markersize=5)
    ax.set_xscale('log')
    ax.set_xticks([.1, 1, 10], ['0.1', '1', '10'])
    panel(ax, 'D  Saved RL-MNIST: shape effect persists', 'Readout multiplier at task 51',
          r'Centered tail $T$ (tasks 151-200)')
    ax.legend(loc='upper right')
    ax.set_ylim(2.4, 4.05)
    fig.suptitle('Frozen readout, hidden height, and distribution shape', fontsize=16, fontweight='bold')
    save(fig, out, 'overview')

    trace = read(out / 'anisotropic_trace.csv')
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6), layout='constrained')
    for v, color in zip([.1, 1., 10.], colors):
        rows = sorted(select(trace, A=10., p=1/64, optimizer='adam', v=v),
                      key=lambda r: number(r, 'step'))
        steps = [number(r, 'step') for r in rows]
        axes[0].plot(steps, [number(r, 'height_ratio_rare_common') for r in rows],
                     'o-', color=color, markersize=3, label=f'v = {v:g}')
        axes[1].plot(steps, [number(r, 'last_step_over_lr_rare') for r in rows],
                     '-', color=color, linewidth=2, label=f'v = {v:g}, rare axis')
        axes[1].plot(steps, [number(r, 'last_step_over_lr_common') for r in rows],
                     '--', color=color, linewidth=2, label=f'v = {v:g}, common axis')
    for ax in axes:
        ax.set_xscale('log')
        ax.legend(fontsize=9)
    panel(axes[0], 'Relative growth changes the direction of w', 'Adam step', r'Height ratio $R=A w_1/w_2$')
    panel(axes[1], 'History slows the high-gain coordinate first', 'Adam step', r'Coordinate step / $\eta$')
    fig.suptitle('Deterministic four-point example: A = 10, p = 1/64', fontsize=14, fontweight='bold')
    save(fig, out, 'anisotropic_dynamics')


if __name__ == '__main__':
    main()
