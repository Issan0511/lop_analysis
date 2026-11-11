#!/usr/bin/env python3
"""POSTHOC EXISTING DATA: archived stage1_ctrl S/Qall trajectories.

Reads NPZ files sequentially. No training and no source-data modification.
Unit summaries are formed within each seed before averaging across 10 seeds.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess

os.environ.setdefault('MPLCONFIGDIR', '/tmp/loading_bridge_matplotlib')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW = Path('/home/issan/Projects/obsidian-research-data/stage1_ctrl_1111/results/stage1_ctrl_1111/raw')
ACTS = ('LR', 'R', 'ELU', 'SILU', 'GELU')
NAMES = {'LR': 'Leaky ReLU', 'R': 'ReLU', 'ELU': 'ELU', 'SILU': 'SiLU', 'GELU': 'GELU'}
METRICS = ('m_end_median', 's_end_median', 'r_end_median', 'dphi_end_mean',
           'pplus_end_mean', 'closed_end_fraction', 'headnorm_start',
           'headnorm_end', 'conf0', 'accuracy_end', 'accuracy_fresh', 'deficit')
UNIT_METRICS = ('m_median', 's_median', 'r_median', 'dphi_mean', 'pplus_mean', 'closed_fraction')
COLORS = {'S': '#4c566a', 'Qall': '#007c91', 'Qbranch': '#d77a18'}
RECORD_FIELDS = ('online_acc', 'online_loss', 'loss_first', 'eval', 'units',
                 'upts', 'push', 'conf0', 'vnorm')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def unit_summary(u):
    return dict(zip(UNIT_METRICS, [float(np.median(u[0])), float(np.median(u[1])),
                float(np.median(u[0] / np.maximum(u[1], 1e-12))),
                float(u[3].mean()), float(u[2].mean()), float((u[2] == 0).mean())]))


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def summarize(rows, grouping, metrics):
    groups = {}
    for row in rows:
        key = tuple(row[k] for k in grouping)
        groups.setdefault(key, []).append(row)
    result = []
    for key, group in groups.items():
        row = dict(zip(grouping, key))
        row['n_seed'] = len(group)
        assert len({r['seed'] for r in group}) == len(group) == 10
        for metric in metrics:
            values = np.array([r[metric] for r in group], dtype=float)
            assert np.all(np.isfinite(values)), (key, metric)
            row[metric + '_mean'] = float(values.mean())
            row[metric + '_sd'] = float(values.std(ddof=1))
        result.append(row)
    return result


def collect(raw):
    rows, branch_rows, sources = [], [], []
    task1_matches, task2_matches, branch_body_matches = [], [], []
    for act in ACTS:
        for seed in range(10):
            bp = raw / f'branch_adam_lr0.001_{act}_s{seed}.npz'
            qp = raw / f'persist_Q_adam_lr0.001_{act}_s{seed}.npz'
            for path in (bp, qp):
                meta_path = path.with_suffix('.json')
                meta = json.loads(meta_path.read_text())
                sources.append({'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest(path),
                                'metadata_path': str(meta_path), 'metadata_sha256': digest(meta_path),
                                'source_git_commit': meta['git_head']})
            with np.load(bp) as b, np.load(qp) as q:
                for field in RECORD_FIELDS:
                    same = np.array_equal(b[f'S|t01|{field}'], q[f'Qall|t01|{field}'])
                    task1_matches.append(same)
                    assert same, (act, seed, field, 'task1 differs')
                for task in range(1, 31):
                    fresh = float(b[f'fa|t{task:02d}|eval'][-1, 0])
                    for arm, d, source in (('S', b, bp), ('Qall', q, qp)):
                        prefix = f'{arm}|t{task:02d}|'
                        u = unit_summary(d[prefix + 'units'][-1])
                        norm = d[prefix + 'vnorm']
                        acc = float(d[prefix + 'eval'][-1, 0])
                        row = {'activation': act, 'arm': arm, 'seed': seed, 'task': task}
                        row.update(dict(zip(METRICS[:6], (u[k] for k in UNIT_METRICS))))
                        row.update(headnorm_start=float(norm[0]), headnorm_end=float(norm[1]),
                                   conf0=float(d[prefix + 'conf0']), accuracy_end=acc,
                                   accuracy_fresh=fresh, deficit=fresh - acc,
                                   source_npz=str(source), source_prefix=prefix)
                        rows.append(row)
                if act == 'ELU':
                    for field in RECORD_FIELDS:
                        same = np.array_equal(b[f'Q|t02|{field}'], q[f'Qall|t02|{field}'])
                        task2_matches.append(same)
                        assert same, (seed, field, 'task2 Qbranch/Qall differs')
                    for task in (2, 30):
                        same_body = np.array_equal(b[f'S|t{task:02d}|units'][0], b[f'Q|t{task:02d}|units'][0])
                        branch_body_matches.append(same_body)
                        assert same_body, (seed, task, 'branch initial body differs')
                        for label, key, d, source in (('S', 'S', b, bp), ('Qbranch', 'Q', b, bp),
                                                       ('Qall', 'Qall', q, qp)):
                            prefix = f'{key}|t{task:02d}|'
                            units, steps = d[prefix + 'units'], d[prefix + 'upts']
                            assert np.array_equal(steps, [0, 1, 10, 50, 200, 1000, 4000])
                            for step, u in zip(steps, units):
                                row = dict(activation=act, arm=label, seed=seed, task=task, step=int(step))
                                row.update(unit_summary(u))
                                row.update(source_npz=str(source), source_key=prefix + 'units')
                                branch_rows.append(row)
    assert len(rows) == 3000 and len(branch_rows) == 420
    return rows, branch_rows, sources, {'task1_pairs': 50, 'task1_fields_compared': len(task1_matches),
                                      'task1_all_exactly_equal': all(task1_matches),
                                      'elu_task2_fields_compared': len(task2_matches),
                                      'elu_task2_qbranch_qall_exactly_equal': all(task2_matches),
                                      'elu_branch_initial_body_checks': len(branch_body_matches),
                                      'elu_branch_initial_body_exactly_equal': all(branch_body_matches)}


def style():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'axes.titlesize': 11, 'axes.labelsize': 9,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.grid': True, 'grid.alpha': .16, 'grid.linewidth': .6,
                         'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'savefig.facecolor': 'white'})


def select(summary, act, arm):
    return sorted([r for r in summary if r['activation'] == act and r['arm'] == arm], key=lambda r: r['task'])


def curve(ax, records, metric, color, linestyle='-', shade=False, label=None):
    x = np.array([r['task'] for r in records])
    y = np.array([r[metric + '_mean'] for r in records])
    ax.plot(x, y, color=color, lw=1.7, ls=linestyle, label=label)
    if shade:
        sd = np.array([r[metric + '_sd'] for r in records])
        ax.fill_between(x, y - sd, y + sd, color=color, alpha=.12, lw=0)


def overview(summary, out):
    fields = [('m_end_median', 'End preactivation mean\n(unit median)'),
              ('s_end_median', 'End preactivation SD\n(unit median)'),
              ('r_end_median', 'End mean / SD\n(unit median)'),
              ('dphi_end_mean', 'End mean |activation derivative|'),
              ('pplus_end_mean', 'End open-input fraction'),
              ('deficit', 'End accuracy deficit\n(fresh minus arm)')]
    fig, axs = plt.subplots(6, 5, figsize=(15, 13.4), sharex=True, sharey='row')
    for col, act in enumerate(ACTS):
        for row, (metric, label) in enumerate(fields):
            ax = axs[row, col]
            for arm in ('S', 'Qall'):
                curve(ax, select(summary, act, arm), metric, COLORS[arm], shade=True)
            ax.set_xlim(1, 30)
            ax.set_xticks([1, 10, 20, 30])
            if row == 0:
                ax.set_title(NAMES[act], weight='bold')
            if col == 0:
                ax.set_ylabel(label)
            if row == len(fields) - 1:
                ax.set_xlabel('Task')
                ax.axhline(0, color='#333333', lw=.7, alpha=.6)
    fig.suptitle('POSTHOC EXISTING DATA | Repeated readout rescaling changes the body trajectory',
                 fontsize=15, weight='bold', y=.985)
    fig.legend([Line2D([], [], color=COLORS[a], lw=2) for a in ('S', 'Qall')],
               ['S: uninterrupted control', 'Qall: rescale at every switch'], ncol=2,
               loc='upper center', bbox_to_anchor=(.5, .957), frameon=False)
    fig.text(.5, .017, 'Lines: mean of 10 seed summaries. Bands: between-seed SD, not confidence intervals. '
             'Task 1 is identical; interventions begin at task 2.\n'
             'All 100 units included. Deficit uses the matched fresh arm; no ReLU eligibility truncation. '
             'Archived stage1_ctrl_1111 data; no new training.', ha='center', fontsize=9)
    fig.subplots_adjust(left=.085, right=.988, top=.908, bottom=.075, hspace=.15, wspace=.11)
    for suffix in ('png', 'pdf'):
        fig.savefig(out / f'longitudinal_overview.{suffix}', dpi=180)
    plt.close(fig)


def readout_and_history(summary, branch_summary, out):
    fig, axs = plt.subplots(3, 4, figsize=(15.5, 9.4))
    for col, act in enumerate(('ELU', 'R')):
        for arm in ('S', 'Qall'):
            records = select(summary, act, arm)
            for metric, ls in (('headnorm_start', '--'), ('headnorm_end', '-')):
                curve(axs[0, col], records, metric, COLORS[arm], ls)
            curve(axs[1, col], records, 'conf0', COLORS[arm])
            curve(axs[2, col], records, 'pplus_end_mean', COLORS[arm])
        axs[0, col].set_title(NAMES[act] + ': accumulated history', weight='bold')
        for row in range(3):
            axs[row, col].set_xlim(1, 30)
            axs[row, col].set_xticks([1, 10, 20, 30])
            axs[row, col].set_xlabel('Task')
        axs[0, col].set_ylabel('Centered head norm')
        axs[1, col].set_ylabel('Mean max softmax\n(after switch intervention)')
        axs[2, col].set_ylabel('End open-input fraction')
        qlast = select(summary, act, 'Qall')[-1]
        axs[0, col].annotate(f"Qall task 30: {qlast['headnorm_start_mean']:.2f} → {qlast['headnorm_end_mean']:.2f}",
                             xy=(.04, .92), xycoords='axes fraction', color=COLORS['Qall'], fontsize=9,
                             va='top', bbox=dict(facecolor='white', alpha=.85, edgecolor='none', pad=2))
    branch_fields = [('m_median', 'Preactivation mean\n(unit median)'),
                     ('s_median', 'Preactivation SD\n(unit median)'),
                     ('r_median', 'Mean / SD\n(unit median)')]
    for col, task in enumerate((2, 30), 2):
        for row, (metric, label) in enumerate(branch_fields):
            ax = axs[row, col]
            arms = ('S', 'Qbranch') if task == 2 else ('S', 'Qbranch', 'Qall')
            for arm in arms:
                records = sorted([r for r in branch_summary if r['task'] == task and r['arm'] == arm], key=lambda r: r['step'])
                ax.plot([r['step'] for r in records], [r[metric + '_mean'] for r in records],
                        color=COLORS[arm], lw=1.8, marker='o', ms=3,
                        ls='--' if arm == 'Qbranch' else '-')
            ax.set_xscale('symlog', linthresh=1)
            ax.set_xticks([0, 10, 200, 4000], labels=['0', '10', '200', '4000'])
            ax.set_xlabel('Updates within task (symlog)')
            ax.set_ylabel(label)
        axs[0, col].set_title(f'ELU task {task}\nSingle versus repeated Q', weight='bold', fontsize=10)
        if task == 2:
            axs[0, col].text(.03, .1, 'Qbranch = Qall at task 2', transform=axs[0, col].transAxes,
                            color=COLORS['Qbranch'], fontsize=9)
    fig.suptitle('POSTHOC EXISTING DATA | Readout regrowth and single-switch versus accumulated history',
                 fontsize=14, weight='bold', y=.986)
    handles = [Line2D([], [], color=COLORS[a], lw=2, ls='--' if a == 'Qbranch' else '-')
               for a in ('S', 'Qall', 'Qbranch')]
    handles += [Line2D([], [], color='#444444', lw=1.7, ls=ls) for ls in ('--', '-')]
    fig.legend(handles, ['S control', 'Qall repeated', 'Qbranch: one intervention from S',
                         'Head norm: start', 'Head norm: end'], ncol=5, loc='upper center',
               bbox_to_anchor=(.5, .956), frameon=False, fontsize=9)
    fig.add_artist(Line2D([.507, .507], [.10, .89], transform=fig.transFigure,
                          color='#999999', lw=.8, alpha=.6))
    fig.text(.5, .018, 'Mean of 10 seeds; all units included. Qall head norm is reset only at switches and regrows during training.\n'
             'Qbranch starts from the same body as S. At task 30, Qall already has a different body at update 0. '
             'Confidence is stored only at update 0.', ha='center', fontsize=9)
    fig.subplots_adjust(left=.06, right=.99, top=.88, bottom=.105, hspace=.43, wspace=.42)
    for suffix in ('png', 'pdf'):
        fig.savefig(out / f'longitudinal_readout_history.{suffix}', dpi=180)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--raw', type=Path, default=DEFAULT_RAW)
    p.add_argument('--out', type=Path, default=ROOT / 'results/loading_bridge_1111')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rows, branches, sources, checks = collect(a.raw)
    summary = summarize(rows, ('activation', 'arm', 'task'), METRICS)
    branch_summary = summarize(branches, ('activation', 'arm', 'task', 'step'), UNIT_METRICS)
    write_csv(a.out / 'longitudinal_seed.csv', rows)
    write_csv(a.out / 'longitudinal_summary.csv', summary)
    write_csv(a.out / 'longitudinal_branch_seed.csv', branches)
    write_csv(a.out / 'longitudinal_branch_summary.csv', branch_summary)
    style()
    overview(summary, a.out)
    readout_and_history(summary, branch_summary, a.out)
    provenance = dict(kind='POSTHOC_EXISTING_DATA_NO_TRAINING', script=str(Path(__file__).resolve()),
                      script_sha256=digest(Path(__file__)), sources=sources, checks=checks,
                      rows=len(rows), branch_rows=len(branches), seeds=list(range(10)),
                      numpy=np.__version__, matplotlib=matplotlib.__version__,
                      base_git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                      units='all 100 units; no ReLU eligibility truncation',
                      aggregate='unit summary within seed, then seed mean and sample SD (ddof=1)',
                      headnorm='class-centered head Frobenius norm; start is after intervention, end after 4000 updates',
                      derivative='stored diagnostic absolute derivative, not optimizer step or a direct mediation measure')
    (a.out / 'longitudinal_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    lines = ['# POSTHOC EXISTING DATA: stage1_ctrl longitudinal audit', '',
             'No new training. All five activations, seeds 0–9, tasks 1–30. Task 1 exists in every Qall file and exactly matches S for all nine stored record fields (450 checks). The ELU task-2 Qbranch/Qall records also exactly match for all nine fields (90 checks). S/Qbranch initial unit summaries match at tasks 2 and 30 (20 checks).', '',
             'Each row in longitudinal_seed.csv is a seed/task/arm. Summaries contain seed means and between-seed sample SD, not confidence intervals. All 100 units are included; the registered ReLU eligibility filter is not applied to this post-hoc description. The deficit uses the same seed/task fresh arm.', '',
             'm and s are each unit\'s preactivation mean and population SD over 1,200 inputs; the reported m, s, and r=m/s are medians over units, formed before the seed mean. The r statistic is the median of unit-wise ratios, not the ratio of the two medians. Derivative and p+ are means over units and inputs; p+ is the fraction with preactivation > 0. End statistics are after 4,000 updates. Head norm is the Frobenius norm after class-centering each head column.', '',
             'The stored ELU derivative diagnostic uses the mathematical derivative; it does not directly measure Adam updates or establish causal mediation.', '',
             'Task 30, means over 10 seeds:', '',
             '| Activation | Qall norm start | Qall norm end | Qall confidence at start | S confidence at start |',
             '|---|---:|---:|---:|---:|']
    for act in ACTS:
        q, s = select(summary, act, 'Qall')[-1], select(summary, act, 'S')[-1]
        lines.append(f"| {NAMES[act]} | {q['headnorm_start_mean']:.4f} | {q['headnorm_end_mean']:.4f} | {q['conf0_mean']:.4f} | {s['conf0_mean']:.4f} |")
    lines += ['', 'Qall start norm 1.7393 is the post-intervention task-30 value, not its end value. Norm regrowth is not itself a measurement of label alignment or kappa. Confidence is available only at update 0.', '',
              'The ELU task-2 Qbranch and Qall curves coincide. At task 30, Qbranch shares the initial body with S, while Qall has a different history before update 0. Its wider body and higher opening/derivative cannot be attributed solely to changes during task 30.', '',
              'Plots: longitudinal_overview.png/.pdf and longitudinal_readout_history.png/.pdf. The second plot uses a symlog update axis for the single-switch branches.', '',
              'Source files and SHA256 hashes: longitudinal_provenance.json. Reproduce with the project Python environment and analysis/loading_bridge_1111/longitudinal.py.']
    (a.out / 'longitudinal_report.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'rows': len(rows), 'branch_rows': len(branches), **checks, 'out': str(a.out)}))


if __name__ == '__main__':
    main()
