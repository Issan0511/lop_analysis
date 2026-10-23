#!/usr/bin/env python3
"""Descriptive CPU-only figures for COMPLETE cnn_response_chb_1021 results.

No torch import, GPU access, result polling, statistical tests, window selection,
or exclusion of failed/nonfinite seeds. Run only after all four arms COMPLETE.
Input files are read-only; every generated artifact is placed under --out.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ARMS = ('CH_self', 'CH_swap', 'CHB_self', 'CHB_swap')
SEEDS = tuple(range(10, 20))
LAYERS = ('c1', 'c2', 'f1', 'f2')
ALL_LAYERS = (*LAYERS, 'head')
STEPS = (0, 100, 750, 3000, 15000, 30000)
WINDOWS = {'first3000': (0, 3000), 'steps1001_3000': (1000, 3000),
           'all': (0, 30000), 'last3000': (27000, 30000)}
WINDOW_LABELS = {'first3000': 'Primary: updates 1–3000', 'steps1001_3000': 'Secondary: updates 1001–3000',
                 'all': 'Secondary: updates 1–30000', 'last3000': 'Secondary: updates 27001–30000'}
UPDATE_FIELDS = ('delta_w_relative_norm', 'delta_b_rms', 'adam_denom_weight_mean',
                 'grad_over_adam_denom_weight_rms', 'grad_weight_rms')
UPDATE_LABELS = ('Relative weight update ‖ΔW‖ / ‖W‖', 'Bias update RMS (includes B)',
                 'Mean Adam weight denominator', 'RMS(gradient / Adam denominator)', 'Weight-gradient RMS')
ARM_COLORS = dict(zip(ARMS, ('#4477AA', '#66CCEE', '#CC6677', '#AA3377')))
ARM_LABELS = {'CH_self': 'CH′ self', 'CH_swap': 'CH′ ← CH′+B response',
              'CHB_self': 'CH′+B self', 'CHB_swap': 'CH′+B ← CH′ response'}


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def read_csv(path):
    with Path(path).open(newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    rows = list(rows)
    require(bool(rows), f'Empty output table: {path}')
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def flag(value):
    if isinstance(value, bool):
        return value
    require(str(value).strip().lower() in ('true', 'false', '0', '1'), f'Unrecognized failed flag: {value}')
    return str(value).strip().lower() in ('true', '1')


def strict_mean(values, axis=None):
    """Arithmetic mean with every observation retained; NaN/inf propagate."""
    with np.errstate(invalid='ignore', over='ignore', divide='ignore'):
        return np.asarray(values).mean(axis=axis, dtype=np.float64)


def nonfinite_counts(values):
    value = np.asarray(values)
    return dict(n=int(value.size), n_finite=int(np.isfinite(value).sum()),
                n_nan=int(np.isnan(value).sum()), n_posinf=int(np.isposinf(value).sum()),
                n_neginf=int(np.isneginf(value).sum()))


def json_safe(value):
    """Explicit strings preserve undefined/infinite values in strict JSON."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return 'NaN' if math.isnan(value) else ('Infinity' if value > 0 else '-Infinity')
    return value


class Resolver:
    """Read original results or their exact archived equivalents, optionally."""
    def __init__(self, root, manifest=None):
        self.root = Path(root).resolve()
        self.entries = []
        if manifest is not None:
            source = read_json(manifest)
            self.entries = source['files'] if isinstance(source, dict) else source
        self.used = {}

    def path(self, relative):
        target = self.root / relative
        if target.exists():
            self.used[str(relative)] = {'path': str(target), 'bytes': target.stat().st_size}
            return target
        suffix = '/results/cnn_response_chb_1021/' + str(relative)
        matches = [entry for entry in self.entries if entry.get('source') == str(target)
                   or str(entry.get('source', '')).endswith(suffix)
                   or entry.get('source_relative') == 'results/cnn_response_chb_1021/' + str(relative)]
        require(len(matches) == 1, f'Missing or ambiguous archive entry: {target}')
        entry = matches[0]
        saved = Path(entry['backup'])
        require(saved.exists() and saved.stat().st_size == entry['bytes'] and digest(saved) == entry['sha256'],
                f'Archive bytes/hash mismatch: {saved}')
        self.used[str(relative)] = {'path': str(saved), 'bytes': saved.stat().st_size,
                                    'archive_sha256': entry['sha256']}
        return saved


def load_complete_results(resolver):
    """Fail closed before writing any result artifact if an arm is incomplete."""
    results = {}
    for arm in ARMS:
        provenance = read_json(resolver.path(f'{arm}/provenance.json'))
        done = read_json(resolver.path(f'{arm}/done.json'))
        require(provenance.get('status') == 'COMPLETE', f'{arm}: provenance is not COMPLETE')
        identity = done['identity']
        require(provenance['identity'] == identity, f'{arm}: identity mismatch')
        require(identity['steps'] == 30000 and identity['data']['seeds'] == list(SEEDS),
                f'{arm}: not the complete registered bundle')
        require(identity.get('arm') == arm, f'{arm}: mislabeled identity')
        for name, key in (('summary.csv', 'summary_sha256'), ('batch_curves.npz', 'curves_sha256'),
                          ('state.pt', 'final_state_sha256')):
            actual = resolver.path(f'{arm}/{name}')
            require(digest(actual) == done[key], f'{arm}: {name} differs from completion hash')
        rows = read_csv(resolver.path(f'{arm}/summary.csv'))
        require(len(rows) == 10 and sorted(int(row['seed']) for row in rows) == list(SEEDS),
                f'{arm}: summary missing/duplicate seeds')
        summary = {int(row['seed']): row for row in rows}
        with np.load(resolver.path(f'{arm}/batch_curves.npz'), allow_pickle=False) as saved:
            arrays = {key: saved[key].copy() for key in ('seeds', 'accuracy', 'ce', 'updates', 'B_fire_fraction')}
        shapes = {'seeds': (10,), 'accuracy': (10, 30000), 'ce': (10, 30000),
                  'updates': (30000, 10, 5, 5), 'B_fire_fraction': (30000, 10, 4)}
        for key, shape in shapes.items():
            require(arrays[key].shape == shape, f'{arm}: unexpected {key} shape')
        require(arrays['seeds'].tolist() == list(SEEDS), f'{arm}: curve seed order changed')
        accuracy, ce = arrays['accuracy'], arrays['ce']
        require(np.isfinite(accuracy).all() and ((accuracy >= 0) & (accuracy <= 1)).all(),
                f'{arm}: accuracy violates zero-on-failure protocol')
        require((np.isfinite(ce) | np.isposinf(ce)).all() and (ce >= 0).all(),
                f'{arm}: CE contains NaN/negative inf, violating +inf-on-failure protocol')
        # Match the registered independent audit: retain float32 source arrays
        # and request float64 accumulation, rather than pre-casting to float64.
        # NumPy can use a different summation kernel after pre-casting (1 ULP
        # observed for CH_swap/seed11/CE_all). Exact comparison stays exact.
        window_means = {name: {'accuracy': accuracy[:, lo:hi].mean(axis=1, dtype=np.float64),
                               'ce': ce[:, lo:hi].mean(axis=1, dtype=np.float64)}
                        for name, (lo, hi) in WINDOWS.items()}
        for name in WINDOWS:
            for key, prefix in (('accuracy', ''), ('ce', 'ce_')):
                actual = np.array([float(summary[seed][prefix+name]) for seed in SEEDS])
                require(np.array_equal(window_means[name][key], actual),
                        f'{arm}/{prefix+name}: summary does not match registered curve reduction')
        failed = []
        onset = []
        for r, seed in enumerate(SEEDS):
            failed.append(flag(summary[seed]['failed']))
            indices = np.flatnonzero(np.isposinf(ce[r]))
            first = int(indices[0]+1) if indices.size else None
            onset.append(first)
            require(bool(indices.size) == failed[-1], f'{arm}/{seed}: failed flag disagrees with CE')
            if first is not None:
                require(np.isposinf(ce[r, first-1:]).all() and (accuracy[r, first-1:] == 0).all(),
                        f'{arm}/{seed}: failed seed was excluded or restarted')
        require(sorted(done['failed_seeds']) == [s for s, f in zip(SEEDS, failed) if f],
                f'{arm}: done.failed_seeds inconsistent')
        diagnostics = read_csv(resolver.path(f'{arm}/diagnostics.csv'))
        expected = {(seed, step) for seed in SEEDS for step in STEPS}
        keys = [(int(row['seed']), int(row['update'])) for row in diagnostics]
        require(len(keys) == len(expected) and set(keys) == expected, f'{arm}: missing/duplicate diagnostic rows')
        require(all(row['arm'] == arm for row in diagnostics), f'{arm}: mixed diagnostics')
        diag = {(int(row['seed']), int(row['update'])): row for row in diagnostics}
        require(all(flag(diag[seed, 30000]['failed']) == failed[r] for r, seed in enumerate(SEEDS)),
                f'{arm}: final diagnostic failures differ')
        results[arm] = dict(done=done, provenance=provenance, summary=summary, arrays=arrays,
                            diagnostics=diag, failed=failed, failure_onset=onset, window_means=window_means)
    base = {k: v for k, v in results[ARMS[0]]['done']['identity'].items() if k != 'arm'}
    require(all({k: v for k, v in results[a]['done']['identity'].items() if k != 'arm'} == base for a in ARMS),
            'Arm data, code, runtime or protocol identities differ')
    for key in ('labels_sha256', 'streams_sha256'):
        require(len({results[a]['done'][key] for a in ARMS}) == 1, f'Paired arms differ in {key}')
    return results


def diagnostic_values(results, arm, key):
    """Shape: seed × registered diagnostic step; no silent missing-column default."""
    return np.array([[float(results[arm]['diagnostics'][seed, step][key]) for step in STEPS] for seed in SEEDS])


def export_tables(results, out):
    windows, update_windows, epoch_accuracy, epoch_updates = [], [], [], []
    diagnostic_long, diagnostic_groups, final = [], [], []
    for arm, data in results.items():
        a = data['arrays']
        ep_acc = strict_mean(a['accuracy'].reshape(10, 400, 75), axis=2)
        ep_ce = strict_mean(a['ce'].reshape(10, 400, 75), axis=2)
        ep_update = strict_mean(a['updates'].reshape(400, 75, 10, 5, 5), axis=1)
        ep_fire = strict_mean(a['B_fire_fraction'].reshape(400, 75, 10, 4), axis=1)
        data['epoch_accuracy'], data['epoch_ce'] = ep_acc, ep_ce
        data['epoch_updates'], data['epoch_fire'] = ep_update, ep_fire
        for r, seed in enumerate(SEEDS):
            common = dict(arm=arm, seed=seed, failed_final=data['failed'][r],
                          failure_first_update=data['failure_onset'][r])
            for name, (lo, hi) in WINDOWS.items():
                windows.append(common | dict(window=name, first_update=lo+1, last_update=hi,
                    accuracy=float(data['window_means'][name]['accuracy'][r]), ce=float(data['window_means'][name]['ce'][r]),
                    ce_infinite_updates=int(np.isposinf(a['ce'][r, lo:hi]).sum())))
                for l, layer in enumerate(ALL_LAYERS):
                    vals = a['updates'][lo:hi, r, l]
                    update_windows.append(common | dict(window=name, layer=layer,
                        **{key: float(strict_mean(vals[:, j])) for j, key in enumerate(UPDATE_FIELDS)},
                        B_fire_fraction=float(strict_mean(a['B_fire_fraction'][lo:hi, r, l])) if l < 4 else float('nan'),
                        B_applicable=l < 4 and arm.startswith('CHB'), **nonfinite_counts(vals)))
            for e in range(400):
                epoch_accuracy.append(common | dict(epoch=e+1, first_update=75*e+1, last_update=75*(e+1),
                                                    accuracy=float(ep_acc[r, e]), ce=float(ep_ce[r, e])))
                for l, layer in enumerate(ALL_LAYERS):
                    epoch_updates.append(common | dict(epoch=e+1, last_update=75*(e+1), layer=layer,
                        **{key: float(ep_update[e, r, l, j]) for j, key in enumerate(UPDATE_FIELDS)},
                        B_fire_fraction=float(ep_fire[e, r, l]) if l < 4 else float('nan'),
                        B_applicable=l < 4 and arm.startswith('CHB')))
            for step in STEPS:
                row = data['diagnostics'][seed, step]
                for key, value in row.items():
                    if key in ('arm', 'seed', 'update', 'failed'):
                        continue
                    diagnostic_long.append(common | dict(update=step, failed_at_snapshot=flag(row['failed']),
                                                          metric=key, value=float(value)))
            end = data['diagnostics'][seed, 30000]
            final.append(common | {key: float(value) for key, value in end.items()
                                   if key not in ('arm', 'seed', 'update', 'failed')}
                         | {f'accuracy_{name}': float(data['summary'][seed][name]) for name in WINDOWS}
                         | {f'ce_{name}': float(data['summary'][seed]['ce_'+name]) for name in WINDOWS})
        metric_keys = [key for key in data['diagnostics'][SEEDS[0], 0] if key not in ('arm', 'seed', 'update', 'failed')]
        for key in metric_keys:
            values = diagnostic_values(results, arm, key)
            for t, step in enumerate(STEPS):
                diagnostic_groups.append(dict(arm=arm, update=step, metric=key,
                    mean_all10=float(strict_mean(values[:, t])), **nonfinite_counts(values[:, t])))
    tables = {'per_seed_windows.csv': windows, 'per_seed_layer_windows.csv': update_windows,
              'per_seed_epoch_accuracy.csv': epoch_accuracy, 'per_seed_epoch_layer_updates.csv': epoch_updates,
              'diagnostics_long.csv': diagnostic_long, 'diagnostic_group_means.csv': diagnostic_groups,
              'per_seed_final_metrics.csv': final}
    for filename, rows in tables.items():
        write_csv(out/filename, rows)
    initial, late = [], []
    for receiver, self_arm, swap_arm, donor_arm in (('CH', 'CH_self', 'CH_swap', 'CHB_self'),
                                                 ('CHB', 'CHB_self', 'CHB_swap', 'CH_self')):
        for seed in SEEDS:
            for layer in LAYERS:
                key = f'{layer}_effective_native_mean_d_mean'
                raw = f'{layer}_raw_native_mean_d_mean'
                s0 = results[self_arm]['diagnostics'][seed, 0]
                x0 = results[swap_arm]['diagnostics'][seed, 0]
                d0 = results[donor_arm]['diagnostics'][seed, 0]
                record = dict(receiver=receiver, seed=seed, layer=layer,
                    self_raw_gate=float(s0[raw]), swap_raw_gate=float(x0[raw]),
                    self_effective_gate=float(s0[key]), swap_effective_gate=float(x0[key]),
                    donor_initial_raw_gate=float(d0[raw]),
                    raw_swap_minus_self=float(x0[raw])-float(s0[raw]),
                    effective_swap_minus_self=float(x0[key])-float(s0[key]),
                    effective_swap_minus_donor=float(x0[key])-float(d0[raw]))
                if layer in ('c1', 'c2'):
                    # Winners belong to each receiver's artificial activity. Donor
                    # own pool winners are not the objects being exchanged.
                    pool_key = f'{layer}_effective_pool_mean_d_mean'
                    record.update(self_receiver_pool_gate=float(s0[pool_key]),
                                  swap_receiver_pool_gate=float(x0[pool_key]),
                                  receiver_pool_gate_difference=float(x0[pool_key])-float(s0[pool_key]))
                initial.append(record)
                for step in (3000, 15000, 30000):
                    sr = results[self_arm]['diagnostics'][seed, step]
                    xr = results[swap_arm]['diagnostics'][seed, step]
                    late.append(dict(receiver=receiver, seed=seed, layer=layer, update=step,
                        self_failed=flag(sr['failed']), swap_failed=flag(xr['failed']),
                        self_effective_gate=float(sr[key]), swap_effective_gate=float(xr[key]),
                        swap_minus_self=float(xr[key])-float(sr[key]),
                        swap_gate_agreement_donor=float(xr[f'{layer}_effective_gate_agreement_donor_mean']),
                        swap_gate_agreement_receiver=float(xr[f'{layer}_effective_gate_agreement_receiver_mean']),
                        swap_delta_rms=float(xr[f'{layer}_effective_delta_rms_mean']),
                        swap_fixed_offset_rms=float(xr[f'{layer}_effective_fixed_anchor_offset_rms_mean']),
                        swap_current_activity_offset_rms=float(xr[f'{layer}_effective_activity_offset_rms_mean'])))
    write_csv(out/'window_arm_means.csv', [dict(arm=arm, window=window,
        accuracy_fraction=float(strict_mean(data['window_means'][window]['accuracy'])),
        accuracy_percent=100*float(strict_mean(data['window_means'][window]['accuracy'])),
        cross_entropy=float(strict_mean(data['window_means'][window]['ce'])),
        n_seeds=10, failed_seeds=sum(data['failed']))
        for arm, data in results.items() for window in WINDOWS])
    write_csv(out/'initial_manipulation_per_seed.csv', initial)
    write_csv(out/'late_response_per_seed.csv', late)
    write_csv(out/'nonfinite_diagnostic_counts.csv', [row for row in diagnostic_groups if row['n_finite'] < row['n']]
              or [dict(arm='ALL', update='', metric='No nonfinite diagnostic values', mean_all10='',
                       n=0, n_finite=0, n_nan=0, n_posinf=0, n_neginf=0)])
    return {'table_rows': {name: len(rows) for name, rows in tables.items()},
            'initial_max_raw_gate_residual': float(np.max(np.abs([r['raw_swap_minus_self'] for r in initial]))),
            'initial_max_effective_donor_gate_residual': float(np.max(np.abs([r['effective_swap_minus_donor'] for r in initial])))}


def setup_style():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.titlesize': 10,
                         'axes.labelsize': 9, 'figure.dpi': 110, 'savefig.dpi': 180,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.grid': True, 'grid.alpha': .18, 'pdf.fonttype': 42,
                         'svg.fonttype': 'none'})


def save_figure(fig, out, stem, caption):
    fig.text(.02, .008, caption, ha='left', va='bottom', fontsize=8, wrap=True)
    fig.savefig(out/f'{stem}.png', bbox_inches='tight')
    fig.savefig(out/f'{stem}.pdf', bbox_inches='tight')
    plt.close(fig)


def curve(ax, x, values, color, label=None, linestyle='-', alpha=.16):
    """Draw all seeds and strict all-seed mean; nonfinite values are explicit gaps."""
    values = np.asarray(values, dtype=float)
    require(values.ndim == 2 and values.shape[0] == 10, 'Every curve must retain all 10 seeds')
    plotted = np.where(np.isfinite(values), values, np.nan)
    for row in plotted:
        ax.plot(x, row, color=color, lw=.55, alpha=alpha, linestyle=linestyle)
    mean = strict_mean(values, axis=0)
    ax.plot(x, np.where(np.isfinite(mean), mean, np.nan), color=color,
            lw=1.7, label=label, linestyle=linestyle)
    return int((~np.isfinite(values)).sum()), int(values.size)


def note_nonfinite(ax, count, total):
    if count:
        ax.text(.98, .03, f'Nonfinite: {count}/{total}\n(gaps, retained in tables/means)',
                transform=ax.transAxes, ha='right', va='bottom', fontsize=6.5, color='#8B0000')


def diagnostic_axis(ax):
    ax.set_xscale('symlog', linthresh=100, linscale=.5)
    ax.set_xticks(STEPS, ['0', '100', '750', '3k', '15k', '30k'])
    ax.tick_params(axis='x', labelsize=7)
    ax.set_xlim(0, 30000)


def plot_behavior(results, out):
    x = np.arange(1, 401)*75
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5), sharex=True, sharey=True)
    for ax, arm in zip(axes.flat, ARMS):
        curve(ax, x, 100*results[arm]['epoch_accuracy'], ARM_COLORS[arm], 'All-seed mean')
        ax.axvspan(0, 3000, color='#DDCC77', alpha=.18, zorder=-2)
        ax.axvspan(27000, 30000, color='#BBBBBB', alpha=.16, zorder=-2)
        failed = [(seed, start) for seed, start in zip(SEEDS, results[arm]['failure_onset']) if start is not None]
        if failed:
            ax.scatter([start for _, start in failed], [0]*len(failed), color='#AA0000', marker='x', s=40, zorder=5)
            ax.text(.98, .98, 'Failed seeds: '+','.join(str(seed) for seed, _ in failed),
                    ha='right', va='top', transform=ax.transAxes, color='#AA0000', fontsize=8)
        ax.set(title=ARM_LABELS[arm], ylim=(-2, 102), xlim=(0, 30000))
    for ax in axes[:, 0]: ax.set_ylabel('Online accuracy (%)')
    for ax in axes[-1]: ax.set_xlabel('Update (nonoverlapping 75-update epoch means)')
    fig.suptitle('Task 31: every seed retained, full 30000-update trajectory')
    fig.tight_layout(rect=(0, .06, 1, .95))
    save_figure(fig, out, '01_learning_all_seeds',
        'Thin lines: seeds10–19; thick line: mean of all10. Gold: registered primary1–3000; gray: last3000.\n'
        'Failed seeds remain accuracy0 and CE+∞ after their first failed update; red crosses show onset. No smoothing/window search.')
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8), sharey=True)
    colors = plt.get_cmap('tab10')(np.arange(10))
    for ax, window in zip(axes.flat, WINDOWS):
        ys = np.array([[float(results[arm]['summary'][seed][window])*100 for arm in ARMS] for seed in SEEDS])
        for r, seed in enumerate(SEEDS):
            offset = (r-4.5)*.022
            for start in (0, 2):
                ax.plot(np.arange(start, start+2)+offset, ys[r, start:start+2], color=colors[r], lw=.65, alpha=.65)
            for c, arm in enumerate(ARMS):
                ax.scatter(c+offset, ys[r, c], marker='x' if results[arm]['failed'][r] else 'o',
                           s=22, linewidths=.9, color=colors[r], zorder=3)
        ax.scatter(np.arange(4), strict_mean(ys, axis=0), marker='_', color='black', s=230, linewidths=2, zorder=5)
        ax.set(xticks=np.arange(4), xticklabels=['CH′ self', 'CH′ swap', 'CH′+B self', 'CH′+B swap'],
               title=WINDOW_LABELS[window], ylim=(-2, 102), xlim=(-.3, 3.3))
        ax.tick_params(axis='x', labelsize=8)
    for ax in axes[:, 0]: ax.set_ylabel('Online accuracy (%)')
    handles = [Line2D([0], [0], marker='o', color=colors[r], linestyle='none', label=str(seed)) for r, seed in enumerate(SEEDS)]
    fig.legend(handles=handles, title='Seed (same color throughout)', ncol=10, loc='lower center', bbox_to_anchor=(.5, .045), fontsize=8)
    fig.suptitle('Registered windows: all40 arm×seed observations')
    fig.tight_layout(rect=(0, .12, 1, .95))
    save_figure(fig, out, '02_registered_windows',
        'Connected points share receiver history and seed; black bars are all10-seed means. × marks a seed that failed anywhere in the full task.\n'
        'Only1–3000 is primary. Other fixed windows are descriptive; no additional tests, selection, or exclusion.')


def plot_diagnostic_grid(results, out, stem, title, series, ylabel, ylim=None, magnitude=False):
    fig, axes = plt.subplots(4, 4, figsize=(14, 10), sharex=True, sharey='row')
    for row, layer in enumerate(LAYERS):
        for col, arm in enumerate(ARMS):
            ax = axes[row, col]
            missing, total = 0, 0
            for suffix, label, color, style in series:
                values = diagnostic_values(results, arm, f'{layer}_{suffix}')
                bad, count = curve(ax, STEPS, values, color, label=label, linestyle=style)
                missing += bad; total += count
            diagnostic_axis(ax)
            if ylim is not None: ax.set_ylim(*ylim)
            if magnitude:
                ax.set_yscale('symlog', linthresh=1e-6, linscale=.5)
            if row == 0: ax.set_title(ARM_LABELS[arm])
            if col == 0: ax.set_ylabel(layer+'\n'+ylabel)
            if row == 3: ax.set_xlabel('Update')
            note_nonfinite(ax, missing, total)
    if magnitude:
        for ax in axes[:, 0]:
            ax.autoscale(enable=True, axis='y')
            ax.set_ylim(bottom=0)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .045), ncol=len(series))
    fig.suptitle(title)
    fig.tight_layout(rect=(0, .1, 1, .96))
    save_figure(fig, out, stem,
        'Thin lines show every seed; thick lines are arithmetic means over all10 seeds. No nonfinite value is excluded from a mean.\n'
        'X axis is symlog(linthresh100); only the six registered snapshots are shown. Y scales are shared within each layer row. Gates are not full Jacobians.')


def plot_update_diagnostics(results, out):
    x = np.arange(1, 401)*75
    fig, axes = plt.subplots(5, 5, figsize=(16, 13), sharex=True, sharey='row')
    for metric, label in enumerate(UPDATE_LABELS):
        for l, layer in enumerate(ALL_LAYERS):
            ax = axes[metric, l]
            bad, total = 0, 0
            for arm in ARMS:
                values = results[arm]['epoch_updates'][:, :, l, metric].T
                n, count = curve(ax, x, values, ARM_COLORS[arm], ARM_LABELS[arm], alpha=.08)
                bad += n; total += count
            ax.set_yscale('symlog', linthresh=1e-10, linscale=.5)
            ax.set_xlim(0, 30000)
            ax.set_xticks([0, 10000, 20000, 30000], ['0', '10k', '20k', '30k'])
            if metric == 0: ax.set_title(layer)
            if l == 0: ax.set_ylabel(label, fontsize=8)
            if metric == 4: ax.set_xlabel('Update')
            ax.tick_params(axis='both', labelsize=7)
            note_nonfinite(ax, bad, total)
    for ax in axes[:, 0]:
        ax.autoscale(enable=True, axis='y')
        ax.set_ylim(bottom=0)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc='lower center', bbox_to_anchor=(.5, .045), fontsize=9)
    fig.suptitle('Adam, gradients and parameter updates: full task, all layers')
    fig.tight_layout(rect=(0, .095, 1, .965))
    save_figure(fig, out, '06_adam_and_updates',
        'Every75-update epoch is summarized without smoothing. Thin lines: every seed; thick: all10 mean. Y scales are shared per metric row, symlog(linthresh1e−10).\n'
        'Δb includes the subsequent B projection for CH′+B recipients. The Adam denominator is sqrt(v/(1−β₂ᵗ))+ε. These are diagnostics, not isolated causal effects.')
    fig, axes = plt.subplots(2, 4, figsize=(13, 6), sharex=True, sharey=True)
    for row, arm in enumerate(('CHB_self', 'CHB_swap')):
        for l, layer in enumerate(LAYERS):
            ax = axes[row, l]
            bad, total = curve(ax, x, results[arm]['epoch_fire'][:, :, l].T, ARM_COLORS[arm])
            ax.set(ylim=(-.02, 1.02), xlim=(0, 30000))
            ax.set_xticks([0, 10000, 20000, 30000], ['0', '10k', '20k', '30k'])
            if row == 0: ax.set_title(layer)
            if l == 0: ax.set_ylabel(ARM_LABELS[arm]+'\nB firing fraction')
            if row == 1: ax.set_xlabel('Update')
            note_nonfinite(ax, bad, total)
    fig.suptitle('Continuing bias projection in CH′+B recipients')
    fig.tight_layout(rect=(0, .095, 1, .95))
    save_figure(fig, out, '07_bias_projection',
        'Fraction of hidden bias coordinates clipped per update, then averaged over each75-update epoch. Thin lines: all seeds; thick line: all10 mean.\n'
        'CH′ recipients have no B treatment and are omitted only from this B-specific figure; their zero traces remain in the exported tables.')


def plot_cross_entropy(results, out):
    """Display the prespecified CE outcomes alongside accuracy, without clipping."""
    x = np.arange(1, 401)*75
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5), sharex=True, sharey=True)
    for ax, arm in zip(axes.flat, ARMS):
        missing, total = curve(ax, x, results[arm]['epoch_ce'], ARM_COLORS[arm])
        ax.axvspan(0, 3000, color='#DDCC77', alpha=.18, zorder=-2)
        ax.axvspan(27000, 30000, color='#BBBBBB', alpha=.16, zorder=-2)
        ax.set(title=ARM_LABELS[arm], xlim=(0, 30000))
        ax.set_yscale('symlog', linthresh=1, linscale=.5)
        note_nonfinite(ax, missing, total)
    axes[0, 0].autoscale(enable=True, axis='y'); axes[0, 0].set_ylim(bottom=0)
    for ax in axes[:, 0]: ax.set_ylabel('Online cross-entropy (lower is better)')
    for ax in axes[-1]: ax.set_xlabel('Update (nonoverlapping75-update epoch means)')
    fig.suptitle('Cross-entropy: accuracy improvement does not imply CE improvement')
    fig.tight_layout(rect=(0, .065, 1, .95))
    save_figure(fig, out, '08_cross_entropy_all_seeds',
        'Thin lines: all seeds; thick: all10-seed arithmetic mean. No finite CE value is clipped or excluded. Shared symlog Y axis, linthresh1.\n'
        'Gold: primary1–3000; gray: last3000. Large CE can coexist with high accuracy when incorrect predictions are very confident.')
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8), sharey=True)
    colors = plt.get_cmap('tab10')(np.arange(10))
    for ax, window in zip(axes.flat, WINDOWS):
        values = np.stack([results[arm]['window_means'][window]['ce'] for arm in ARMS], axis=1)
        for r, seed in enumerate(SEEDS):
            offset = (r-4.5)*.022
            for start in (0, 2):
                ax.plot(np.arange(start, start+2)+offset,
                        np.where(np.isfinite(values[r, start:start+2]), values[r, start:start+2], np.nan),
                        color=colors[r], lw=.65, alpha=.65)
            for col, arm in enumerate(ARMS):
                if np.isfinite(values[r, col]):
                    ax.scatter(col+offset, values[r, col], s=22, color=colors[r],
                               marker='x' if results[arm]['failed'][r] else 'o', zorder=3)
        means = strict_mean(values, axis=0)
        ax.scatter(np.arange(4), np.where(np.isfinite(means), means, np.nan),
                   marker='_', color='black', s=230, linewidths=2, zorder=5)
        ax.set(xticks=np.arange(4), xticklabels=['CH′ self', 'CH′ swap', 'CH′+B self', 'CH′+B swap'],
               title=WINDOW_LABELS[window], xlim=(-.3, 3.3))
        ax.set_yscale('symlog', linthresh=1, linscale=.5)
        ax.tick_params(axis='x', labelsize=8)
        note_nonfinite(ax, int((~np.isfinite(values)).sum()), int(values.size))
    axes[0, 0].autoscale(enable=True, axis='y'); axes[0, 0].set_ylim(bottom=0)
    for ax in axes[:, 0]: ax.set_ylabel('Online cross-entropy (lower is better)')
    handles = [Line2D([0], [0], marker='o', color=colors[r], linestyle='none', label=str(seed)) for r, seed in enumerate(SEEDS)]
    fig.legend(handles=handles, title='Seed', ncol=10, loc='lower center', bbox_to_anchor=(.5, .045), fontsize=8)
    fig.suptitle('Registered windows: CE for every arm and seed')
    fig.tight_layout(rect=(0, .12, 1, .95))
    save_figure(fig, out, '09_registered_window_cross_entropy',
        'Connected points share receiver history and seed; black bars: all10 means. No finite value is clipped. Shared symlog Y axis, linthresh1.\n'
        'CE is secondary and must be reported alongside the accuracy-defined primary verdict. No new tests or outcome-selected windows.')


def write_notes(out, results, summary):
    failure = [f'{arm}: '+(', '.join(f'{seed}@{onset}' for seed, onset in zip(SEEDS, data['failure_onset']) if onset is not None) or 'none')
               for arm, data in results.items()]
    text = """# Supplementary descriptive diagnostics

These figures and tables were produced only after every registered arm was COMPLETE. They do not modify the primary verdict and introduce no statistical tests or data-dependent windows.

## Reading order

1. `initial_manipulation_per_seed.csv` and `03_raw_effective_gates`: establish whether the effective activation derivative means match the initial donor while receiver raw gates remain unchanged at update0. The CSV includes each seed and layer. A scalar mean alone does not demonstrate semantic feature alignment or the full Jacobian exchange; use the registered preflight for the elementwise intervention checks.
2. `01_learning_all_seeds`, `02_registered_windows`, `08_cross_entropy_all_seeds`, `09_registered_window_cross_entropy` and `window_arm_means.csv`: inspect the entire task, every seed, and the fixed primary/secondary windows. The primary verdict belongs to the registered independent main report. Accuracy improvement and CE worsening may coexist. The registered primary verdict is defined by accuracy and does not assert improvement on all metrics.
3. `04_gate_agreement` and `05_delta_and_offsets`: inspect response movement away from the fixed fields. Gate agreement can be high simply because many units share a closed gate. Fixed anchor offsets, current activity displacement and raw preactivation delta are distinct quantities.
4. `06_adam_and_updates` and `07_bias_projection`: inspect receiver optimization and continued B. Similar update norms do not isolate an effective-step-size causal mechanism.
5. `per_seed_final_metrics.csv` and `late_response_per_seed.csv`: retain all final values, failure status, and layer-level late behavior.

## Reductions and missing values

- Every scientific comparison includes seeds10–19. `mean_all10` and thick figure lines use ordinary arithmetic means; no `nanmean`, failed-seed exclusion, or successful-seed rerun occurs.
- Accuracy tables use fractions0–1; behavior figures use percentages0–100. Accuracy and CE come from all30000 saved pre-update online measurements. Failure policy remains accuracy0/CE+∞ from failure onset onward. The script rejects curves that violate that protocol.
- Diagnostics use the six registered snapshots. The raw/effective mean derivative is1 for native clamp input≥0 (including exactly0), and0 below0; the strict opening fraction is a separate diagnostic available in CSV.
- For convolutional diagnostics, stored scalar means average the per-position image-axis statistics over channels and spatial positions. In particular, the plotted displacement magnitude is the mean of per-position image RMS values, not an RMS across all channels/positions.
- All layers retain their own scales. Activity effective rank follows the original stored channel/unit convention; convolutional rank is not the rank of the entire flattened spatial feature vector.
- At convolutional sites, pooled derivatives are evaluated at the receiver's actual artificial-activity pool winners. They are not derivatives at the donor's own pool winners.
- `activity_offset_rms` is current artificial activity minus native clamp of current raw z. `fixed_anchor_offset_rms` is clamp(initial receiver z) minus clamp(initial donor z). These must not be conflated.
- Update traces are reported as fixed nonoverlapping75-update epoch means. Weight relative update uses the pre-update weight norm. Bias update includes B after Adam where present. Denominator means and RMS(gradient/denominator) are descriptive recorded quantities.
- Head B columns are NaN with `B_applicable=False`; no head bias projection exists. CH′ hidden B fractions are zero and `B_applicable=False`.
- CSV preserves literal `nan`, `inf`, and `-inf`. They remain in means. Figures display nonfinite values as gaps and annotate their counts. `nonfinite_diagnostic_counts.csv` distinguishes undefined ratios from a failed seed; the failure flags are authoritative for the experiment's failure policy.
- The log-like figure axes use fixed, stated symlog thresholds solely for visualization, retaining exact zeros. They do not change exported data or inference.

## Interpretation limits

The intervention concerns activation derivative gates at fixed corresponding image/channel/position indices. It does not exchange weights, head, optimizer history, centering prescription, pool selection, or the full Jacobian. Do not divide the effect by the known natural CH′+B benefit to claim a mediated fraction. The one-task intervention does not establish long-term prevention or identify which of the four layers is necessary.

## Validation

All four completion records, provenance identities, all40 seed rows, label/batch pairing, curves and final-state SHA256 values were checked before output. All four registered windows were independently reconstructed from float32 batch curves with `mean(axis=1,dtype=float64)`, exactly matching the registered independent audit. Pre-casting to float64 before summation is intentionally avoided because NumPy may choose another summation kernel. No torch or GPU library is imported.

Failure onset (seed@update):
"""
    text += '\n'.join('- '+entry for entry in failure)+'\n\n'
    text += '## Accuracy and cross-entropy together\n\n| Arm | First3000 accuracy (%) | First3000 CE | Last3000 accuracy (%) | Last3000 CE |\n|---|---:|---:|---:|---:|\n'
    for arm, data in results.items():
        first, last = data['window_means']['first3000'], data['window_means']['last3000']
        text += f"| {arm} | {100*strict_mean(first['accuracy']):.5f} | {strict_mean(first['ce']):.6g} | {100*strict_mean(last['accuracy']):.5f} | {strict_mean(last['ce']):.6g} |\n"
    text += '\nAn accuracy-defined rescue is not an all-metric rescue; retain the CE increase in interpretation.\n\n'
    text += f"Initial maximum raw gate mean residual within receiver: {summary['initial_max_raw_gate_residual']!r}.\n\n"
    text += f"Initial maximum swapped effective minus donor raw gate mean residual: {summary['initial_max_effective_donor_gate_residual']!r}.\n"
    (out/'READING_GUIDE.md').write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True, help='Original results/cnn_response_chb_1021 directory')
    parser.add_argument('--out', type=Path, required=True, help='Directory for supplemental CSV, PNG, PDF and reading guide')
    parser.add_argument('--archive-manifest', type=Path, help='Optional exact backup manifest if original raw files were moved')
    args = parser.parse_args()
    require(args.out.resolve() != args.results.resolve(), 'Use a separate supplemental output directory')
    resolver = Resolver(args.results, args.archive_manifest)
    results = load_complete_results(resolver)
    # Output creation occurs only after successful four-arm completion validation.
    args.out.mkdir(parents=True, exist_ok=True)
    summary = export_tables(results, args.out)
    setup_style()
    plot_behavior(results, args.out)
    plot_diagnostic_grid(results, args.out, '03_raw_effective_gates',
        'All4 hidden layers: raw versus effective activation derivative gates',
        [('raw_native_mean_d_mean', 'Raw gate', '#EE7733', '--'),
         ('effective_native_mean_d_mean', 'Effective gate', '#0077BB', '-')],
        'Mean native derivative', ylim=(-.02, 1.02))
    plot_diagnostic_grid(results, args.out, '04_gate_agreement',
        'Current effective gates versus the two frozen reference gate fields',
        [('effective_gate_agreement_receiver_mean', 'Initial receiver agreement', '#009988', '--'),
         ('effective_gate_agreement_donor_mean', 'Initial donor agreement', '#AA3377', '-')],
        'Gate agreement', ylim=(-.02, 1.02))
    plot_diagnostic_grid(results, args.out, '05_delta_and_offsets',
        'Field displacement and activity offsets: separate quantities',
        [('effective_delta_rms_mean', 'Current raw-z delta', '#4477AA', '-'),
         ('effective_fixed_anchor_offset_rms_mean', 'Fixed anchor offset', '#CC6677', '--'),
         ('effective_activity_offset_rms_mean', 'Current activity displacement', '#228833', ':')],
        'Mean per-position image RMS', magnitude=True)
    plot_update_diagnostics(results, args.out)
    plot_cross_entropy(results, args.out)
    write_notes(args.out, results, summary)
    for record in resolver.used.values():
        record['sha256'] = digest(record['path'])
    manifest = dict(script=str(Path(__file__).resolve()), script_sha256=digest(__file__),
                    command=sys.argv, inputs=resolver.used, figures_and_tables={}, summary=summary,
                    numpy=np.__version__, matplotlib=matplotlib.__version__, seeds=list(SEEDS),
                    arms=list(ARMS), diagnostic_steps=list(STEPS), windows=WINDOWS,
                    statistical_tests='none; descriptive supplement only',
                    nonfinite_policy='retain in tables and ordinary means; annotated plotting gaps')
    for path in sorted(args.out.iterdir()):
        if path.is_file() and path.name != 'supplement_manifest.json':
            manifest['figures_and_tables'][path.name] = {'bytes': path.stat().st_size, 'sha256': digest(path)}
    (args.out/'supplement_manifest.json').write_text(json.dumps(json_safe(manifest), indent=2, allow_nan=False)+'\n')
    print(f'Wrote supplementary tables and9 paired PNG/PDF figures to {args.out}')


if __name__ == '__main__':
    main()
