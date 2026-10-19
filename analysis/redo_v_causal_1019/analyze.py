#!/usr/bin/env python3
"""Seed-paired analysis of preregistered outgoing-reset interventions."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

os.environ.setdefault('MPLCONFIGDIR', '/tmp/redo_v_causal_matplotlib_1019')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
CORE_ARMS = ('zero', 'keep', 'rand01', 'rand03', 'rand10')
ARMS = CORE_ARMS + ('center', 'center_random', 'gauge4', 'gauge025')
COLORS = dict(zero='#222222', keep='#9874af', rand01='#87b5c8', rand03='#3888a8', rand10='#c55e32',
              center='#258143', center_random='#8db17e', gauge4='#cc912d', gauge025='#7b5ab2')
METRICS = ('task3_near99_mean', 'task3_fullopen_mean', 'task3_p_mean',
           'task3_mean_mean', 'task3_sd_mean', 'task3_activation_rms_mean',
           'task3_activation_sd_mean', 'task3_gate_diversity_mean',
           'global_near99_mean', 'global_fullopen_mean', 'global_p_mean',
           'global_mean_mean', 'global_sd_mean', 'v_centered_norm_mean',
           'ce_demand_mean_mean', 'ce_brake_mean_mean', 'ce_upward_mean_mean')


def read_csv(path):
    output = []
    with path.open() as handle:
        for row in csv.DictReader(handle):
            parsed = {}
            for key, value in row.items():
                if value is None or value == '':
                    parsed[key] = None
                    continue
                try:
                    parsed[key] = float(value)
                except ValueError:
                    parsed[key] = value
            output.append(parsed)
    return output


def write_csv(path, rows):
    if rows:
        with path.open('w') as handle:
            writer = csv.DictWriter(handle, list(dict.fromkeys(k for row in rows for k in row)))
            writer.writeheader()
            writer.writerows(rows)


def load_npz(path):
    with np.load(path) as handle:
        return {k: handle[k] for k in handle.files}


def sign_test(difference):
    positive = int((difference > 0).sum())
    negative = int((difference < 0).sum())
    n = positive + negative
    p = min(1., 2 * sum(math.comb(n, k) for k in range(min(positive, negative) + 1)) / 2 ** n) if n else 1.
    return {'positive': positive, 'negative': negative, 'ties': int((difference == 0).sum()), 'sign_p_two_sided': p}


def contrast(values, reference, seed=1019):
    ids = sorted(set(values) & set(reference))
    delta = np.asarray([values[s] - reference[s] for s in ids], dtype=float)
    if not len(delta) or not np.isfinite(delta).all():
        raise ValueError('Empty or nonfinite paired endpoint')
    rng = np.random.default_rng(seed)
    boot = delta[rng.integers(0, len(delta), size=(20_000, len(delta)))].mean(axis=1)
    return {'n_seeds': len(delta), 'mean_difference': float(delta.mean()),
            'median_difference': float(np.median(delta)),
            'ci95_low': float(np.quantile(boot, .025)), 'ci95_high': float(np.quantile(boot, .975)),
            **sign_test(delta)}


def primary_verdict(result):
    if result['ci95_high'] < -.10:
        return 'registered_meaningful_suppression'
    if result['ci95_high'] < 0:
        return 'suppression_smaller_than_registered_threshold'
    if result['ci95_low'] > 0:
        return 'opposite_direction_for_this_contrast'
    return 'unresolved_interval_includes_zero'


def validate_groups(raw, arms):
    results = []
    sources = []
    for start in (0, 10):
        base = raw / f'zero_s{start}-{start + 9}'
        baseline_fingerprint = json.loads((base / 'trace/fingerprints.json').read_text())
        baseline_batches = json.loads((base / 'trace/batch_hashes.json').read_text())
        baseline_cohort = load_npz(base / 'trace/cohort.npz')
        for arm in arms:
            folder = raw / f'{arm}_s{start}-{start + 9}'
            fingerprint = json.loads((folder / 'trace/fingerprints.json').read_text())
            batches = json.loads((folder / 'trace/batch_hashes.json').read_text())
            cohort = load_npz(folder / 'trace/cohort.npz')
            reproduction = json.loads((folder / 'reproduction.json').read_text())
            excluded = ('P3',) if arm in ('center', 'center_random') else ('P4',)
            unchanged = all(fingerprint['intervention_post']['all'][key] == fingerprint['zero_post']['all'][key]
                            for key in fingerprint['zero_post']['all'] if key not in excluded)
            row = {'arm': arm, 'start_seed': start,
                   'pre_reset_equal': fingerprint['reset_pre'] == baseline_fingerprint['reset_pre'],
                   'complete_reset_equal': fingerprint['zero_post'] == baseline_fingerprint['zero_post'],
                   'unchanged_except_registered_parameters': unchanged,
                   'batch_hashes_equal': batches == baseline_batches,
                   'cohort_masks_equal': bool(np.array_equal(cohort['masks'], baseline_cohort['masks'])),
                   'old_W3_equal': bool(np.array_equal(cohort['old_W3'], baseline_cohort['old_W3'])),
                   'reference_prefix_exact': bool(reproduction['exact'])}
            if not all(row[k] for k in row if k not in ('arm', 'start_seed')):
                raise AssertionError(f'Pairing/replay checks failed: {row}')
            final = load_npz(folder / 'final_state.npz')
            row['all_final_parameters_and_moments_finite'] = all(np.isfinite(value).all() for value in final.values())
            if not row['all_final_parameters_and_moments_finite']:
                raise AssertionError(f'Nonfinite final state: {folder}')
            if arm.startswith('gauge'):
                row['pre_gauge_equals_zero_step50'] = fingerprint['gauge_pre'] == baseline_fingerprint['task3_step50']
                if not row['pre_gauge_equals_zero_step50']:
                    raise AssertionError(f'Gauge prefix differs from zero: {folder}')
            coordinate_path = folder / 'trace/coordinate_checks.json'
            if coordinate_path.exists():
                coordinate = json.loads(coordinate_path.read_text())
                row['initial_coordinate_function_invariance'] = all(q['logit_max_error'] <= 1e-5 and q['gate_sign_changed'] == 0 for q in coordinate)
                if not row['initial_coordinate_function_invariance']:
                    raise AssertionError(f'Coordinate invariance failed: {folder}')
            results.append(row)
            for path in (folder / 'trace/per_seed.csv', folder / 'trace/shock.csv', folder / 'trace/online_segments.csv', folder / 'replay/per_task.csv'):
                sources.append({'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    return results, sources


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', type=Path, default=ROOT / 'results/redo_v_causal_1019/raw')
    parser.add_argument('--out', type=Path, default=ROOT / 'results/redo_v_causal_1019')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        difference = {seed: 0. for seed in range(20)}
        result = contrast(difference, {seed: 1. for seed in range(20)})
        assert result['mean_difference'] == -1 and result['ci95_low'] == result['ci95_high'] == -1
        assert result['sign_p_two_sided'] == 2 / (2 ** 20)
        assert primary_verdict(result) == 'registered_meaningful_suppression'
        assert sign_test(np.zeros(20))['sign_p_two_sided'] == 1
        print(json.dumps({'status': 'passed', 'checks': ['paired bootstrap constant effect', 'exact sign test', 'ties', 'preregistered threshold']}))
        return
    args.out.mkdir(parents=True, exist_ok=True)
    checks, sources = validate_groups(args.raw, ARMS)
    units, performance, shocks, deltas, segments = [], [], [], [], []
    ce_max = 0.
    ce_brake_min = math.inf
    for arm in ARMS:
        for start in (0, 10):
            folder = args.raw / f'{arm}_s{start}-{start + 9}'
            seed_rows = read_csv(folder / 'trace/per_seed.csv')
            for row in seed_rows:
                row['seed'] = int(row['seed'])
                row['task'] = int(row['task'])
                row['step'] = int(row['step'])
            units.extend(seed_rows)
            for row in read_csv(folder / 'replay/per_task.csv'):
                row['arm'] = arm
                row['seed'] = int(row['seed'])
                row['task'] = int(row['task'])
                performance.append(row)
            for row in read_csv(folder / 'trace/shock.csv'):
                shocks.append({'arm': arm, **row})
            for row in read_csv(folder / 'trace/online_segments.csv'):
                row['seed'] = int(row['seed'])
                segments.append({'arm': arm, **row})
            trace = load_npz(folder / 'trace/units.npz')
            if 'ce_closure_max' in trace:
                ce_max = max(ce_max, float(np.nanmax(trace['ce_closure_max'])))
                ce_brake_min = min(ce_brake_min, float(np.nanmin(trace['ce_brake_min'])))
            dd = load_npz(folder / 'trace/deltas.npz')
            cohort = load_npz(folder / 'trace/cohort.npz')['masks'][:, 1].astype(bool)
            for j, task in enumerate(dd['task']):
                if task != 3:
                    continue
                for r, seed in enumerate(range(start, start + 10)):
                    for label, mask in [('first_reset', cohort[r]), ('survivor', ~cohort[r])]:
                        result = {'arm': arm, 'seed': seed, 'task': 3, 'step': int(dd['step'][j]), 'cohort': label}
                        for key in ('own_w', 'own_b', 'upstream', 'actual', 'adam_current', 'adam_memory', 'sgd_direction'):
                            result[key] = float(dd[key][j, r, mask].mean())
                        result['closure_abs_max'] = float(np.max(np.abs(dd['closure'][j, r, mask])))
                        result['adam_closure_abs_max'] = float(np.max(np.abs(dd['adam_closure'][j, r, mask])))
                        deltas.append(result)
    if ce_max > 1e-8 or ce_brake_min < -1e-8:
        raise AssertionError(f'CE identity/convexity check failed: {ce_max}, {ce_brake_min}')

    # Every reported endpoint is fixed first-reset cohort, irrespective of outcome.
    grouped = []
    fields = ('arm', 'cohort', 'event', 'task', 'step')
    unique = dict.fromkeys(tuple(row[k] for k in fields) for row in units)
    for key in unique:
        rows = [row for row in units if tuple(row[k] for k in fields) == key]
        result = dict(zip(fields, key))
        result['n_seeds'] = len(rows)
        for metric in METRICS:
            values = [row[metric] for row in rows if row.get(metric) is not None and np.isfinite(row[metric])]
            if values:
                result[metric] = float(np.mean(values))
        grouped.append(result)
    contrasts = []
    paired_values = []
    for step in (50, 780):
        for metric in METRICS:
            reference = {row['seed']: row[metric] for row in units if row['arm'] == 'zero' and
                         row['cohort'] == 'first_reset' and row['event'] == 'after_step' and row['task'] == 3 and
                         row['step'] == step and row.get(metric) is not None and np.isfinite(row[metric])}
            if not reference:
                continue
            for arm in ARMS[1:]:
                values = {row['seed']: row[metric] for row in units if row['arm'] == arm and
                          row['cohort'] == 'first_reset' and row['event'] == 'after_step' and row['task'] == 3 and
                          row['step'] == step and row.get(metric) is not None and np.isfinite(row[metric])}
                if set(reference) != set(range(20)) or set(values) != set(reference):
                    raise AssertionError(f'Missing/duplicate seed endpoint {arm}/{step}/{metric}')
                result = {'arm': arm, 'reference': 'zero', 'step': step, 'cohort': 'first_reset', 'metric': metric,
                          'arm_mean': float(np.mean(list(values.values()))),
                          'reference_mean': float(np.mean(list(reference.values()))), **contrast(values, reference)}
                result['registered_primary'] = arm == 'rand10' and step == 50 and metric == 'task3_near99_mean'
                contrasts.append(result)
                for seed in sorted(values):
                    paired_values.append({'arm': arm, 'step': step, 'metric': metric, 'seed': seed,
                                          'value': values[seed], 'reference_value': reference[seed], 'difference': values[seed] - reference[seed]})
    perf_summary, perf_contrasts = [], []
    for task in sorted(set(row['task'] for row in performance)):
        for arm in ARMS:
            rows = [row for row in performance if row['task'] == task and row['arm'] == arm]
            perf_summary.append({'arm': arm, 'task': task, 'n_seeds': len(rows),
                                 **{key: float(np.mean([row[key] for row in rows])) for key in ('online_acc', 'train_acc', 'test_acc')}})
        if task != 3:
            continue
        for key in ('online_acc', 'train_acc', 'test_acc'):
            reference = {row['seed']: row[key] for row in performance if row['task'] == task and row['arm'] == 'zero'}
            for arm in ARMS[1:]:
                values = {row['seed']: row[key] for row in performance if row['task'] == task and row['arm'] == arm}
                if set(values) != set(range(20)) or set(reference) != set(values):
                    raise AssertionError(f'Missing performance endpoint {arm}/{key}')
                perf_contrasts.append({'arm': arm, 'task': task, 'metric': key, **contrast(values, reference)})
    primary = next(row for row in contrasts if row['registered_primary'])
    route_contrasts = []
    for reference_arm in ('zero', 'center_random'):
        for metric in ('task3_near99_mean', 'online_acc'):
            if metric == 'task3_near99_mean':
                selected = [r for r in units if r['cohort'] == 'first_reset' and r['event'] == 'after_step' and r['task'] == 3 and r['step'] == 50]
            else:
                selected = [r for r in performance if r['task'] == 3]
            values = {r['seed']: r[metric] for r in selected if r['arm'] == 'center'}
            reference = {r['seed']: r[metric] for r in selected if r['arm'] == reference_arm}
            result = {'arm': 'center', 'reference': reference_arm, 'metric': metric,
                      'step': 50 if metric == 'task3_near99_mean' else 780, **contrast(values, reference)}
            result['registered_support'] = result['ci95_high'] < 0 if metric == 'task3_near99_mean' else (result['mean_difference'] >= .005 and result['ci95_low'] > 0)
            route_contrasts.append(result)
    gauge_contrasts = []
    for arm in ('gauge4', 'gauge025'):
        for metric in ('online_steps1_50', 'online_steps51_780', 'online_steps1_780'):
            values = {r['seed']: r[metric] for r in segments if r['arm'] == arm}
            reference = {r['seed']: r[metric] for r in segments if r['arm'] == 'zero'}
            if set(values) != set(range(20)) or set(reference) != set(range(20)):
                raise AssertionError(f'Missing gauge online endpoint seeds: {arm}/{metric}')
            result = {'arm': arm, 'reference': 'zero', 'metric': metric, **contrast(values, reference)}
            if metric == 'online_steps1_50' and any(values[s] != reference[s] for s in values):
                raise AssertionError(f'Gauge online first50 prefix mismatch: {arm}')
            result['registered_primary_gauge'] = metric == 'online_steps51_780'
            gauge_contrasts.append(result)
    verdict = {'analysis_rng': 1019, 'bootstrap_replicates': 20_000, 'bootstrap': 'paired-seed percentile95% interval',
               'seed_unit': '20 paired seed-specific fixed first-reset L2 cohorts; no unit-level pseudoreplication',
               'primary': primary, 'primary_verdict': primary_verdict(primary),
               'checks': checks, 'ce_closure_max': ce_max, 'ce_brake_min': ce_brake_min,
               'source_files': sources, 'contrasts': contrasts, 'performance_contrasts': perf_contrasts,
               'route_contrasts': route_contrasts, 'route_registered_support': all(r['registered_support'] for r in route_contrasts),
               'gauge_contrasts': gauge_contrasts,
               'limitations': ['Nonzero outgoing initialization changes initial predictions; shock is measured, not matched.',
                               'Random-direction norm replacement is a specific outgoing intervention, not an exhaustive test of all initialization schemes.',
                               'A alone does not establish positive-sidedness as a mediator of worse accuracy.',
                               'Center changes the mean-input update route and Adam geometry. Any gating/accuracy co-movement does not uniquely identify gating mediation.',
                               'Gauge preserves initial function and gates but changes future Adam dynamics; later gates can diverge.',
                               'Task3 is one early hard task in this real-label MLP protocol, not a claim across architectures or late training.',
                               'CE split is a local raw-gradient diagnostic, not an Adam-update decomposition; actual-step decomposition is reported separately.',
                               'Secondary contrasts are diagnostic, without multiple-comparison-adjusted global superiority claims.']}
    (args.out / 'verdict.json').write_text(json.dumps(verdict, ensure_ascii=False, indent=2))
    for name, rows in [('per_seed', units), ('summary', grouped), ('contrasts', contrasts), ('paired_differences', paired_values),
                       ('performance', performance), ('performance_summary', perf_summary), ('performance_contrasts', perf_contrasts),
                       ('shock', shocks), ('actual_deltas', deltas), ('checks', checks), ('verdict', [primary])]:
        write_csv(args.out / f'{name}.csv', rows)
    for name, rows in [('online_segments', segments), ('route_contrasts', route_contrasts), ('gauge_contrasts', gauge_contrasts)]:
        write_csv(args.out / f'{name}.csv', rows)
    make_figure(args.out, grouped, contrasts, perf_summary)
    make_report(args.out, primary, verdict, grouped, perf_summary, perf_contrasts, shocks)
    print(json.dumps({'primary': primary, 'primary_verdict': verdict['primary_verdict'], 'ce_closure_max': ce_max,
                      'performance_task3': [r for r in perf_summary if r['task'] == 3]}, ensure_ascii=False, indent=2))


def make_figure(out, grouped, contrasts, perf_summary):
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), layout='constrained')
    for arm in CORE_ARMS:
        rows = sorted((r for r in grouped if r['arm'] == arm and r['cohort'] == 'first_reset' and
                       (r['event'] == 'reset_post' or (r['task'] == 3 and r['event'] == 'after_step'))),
                      key=lambda r: 0 if r['event'] == 'reset_post' else r['step'])
        x = [0 if r['event'] == 'reset_post' else r['step'] for r in rows]
        axes[0].plot(x, [r['task3_near99_mean'] for r in rows], '.-', color=COLORS[arm], label=arm)
    axes[0].set(xlabel='Task-3 updates', ylabel='Fraction of cohort with opening >= .99', ylim=(-.02, 1.02))
    axes[0].set_xscale('symlog', linthresh=1)
    axes[0].legend(frameon=False, fontsize=8)
    rows = [next(r for r in contrasts if r['arm'] == arm and r['step'] == 50 and r['metric'] == 'task3_near99_mean') for arm in CORE_ARMS[1:]]
    for i, row in enumerate(rows):
        axes[1].errorbar(row['mean_difference'], i, xerr=[[row['mean_difference'] - row['ci95_low']], [row['ci95_high'] - row['mean_difference']]],
                         fmt='o', color=COLORS[row['arm']], capsize=3)
    axes[1].set_yticks(range(len(rows)), [r['arm'] for r in rows])
    axes[1].axvline(0, color='#777777', lw=.8)
    axes[1].axvline(-.10, color='#bbbbbb', ls='--', lw=.8)
    axes[1].set(xlabel='High-open fraction difference vs zero', title='Step 50: paired95% bootstrap CI')
    for i, arm in enumerate(ARMS):
        row = next(r for r in perf_summary if r['arm'] == arm and r['task'] == 3)
        axes[2].plot(i, row['online_acc'], 'o', color=COLORS[arm])
        axes[2].plot(i, row['test_acc'], 's', color=COLORS[arm], fillstyle='none')
    axes[2].set_xticks(range(len(ARMS)), ARMS, rotation=60, fontsize=7)
    axes[2].set(ylabel='Accuracy', title='Task 3: online (o), end test (square)')
    fig.savefig(out / 'outgoing_interventions.png', dpi=180)
    fig.savefig(out / 'outgoing_interventions.pdf')
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), layout='constrained')
    for arm in ('zero', 'center', 'center_random', 'gauge4', 'gauge025'):
        rows = sorted((r for r in grouped if r['arm'] == arm and r['cohort'] == 'first_reset' and r['task'] == 3 and r['event'] == 'after_step'), key=lambda r: r['step'])
        for axis, key in zip(axes, ('task3_near99_mean', 'task3_mean_mean', 'v_centered_norm_mean')):
            axis.plot([r['step'] for r in rows], [r[key] for r in rows], '.-', color=COLORS[arm], label=arm)
            axis.set_xscale('symlog', linthresh=1)
            axis.set_xlabel('Task-3 updates')
    axes[0].set(ylabel='Fraction of cohort with opening >= .99', ylim=(-.02, 1.02))
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].set(ylabel='Mean preactivation on task-3 images')
    axes[2].set(ylabel='Class-centered outgoing norm')
    fig.suptitle('Coordinate controls: gauge step-50 points precede rescaling', fontsize=10)
    fig.savefig(out / 'coordinate_interventions.png', dpi=180)
    fig.savefig(out / 'coordinate_interventions.pdf')
    plt.close(fig)


def make_report(out, primary, verdict, grouped, perf_summary, perf_contrasts, shocks):
    labels = {'registered_meaningful_suppression': '事前基準を満たす正の片側化の抑制',
              'suppression_smaller_than_registered_threshold': '抑制方向だが10ポイント以上の抑制を支持するCI基準は未達',
              'opposite_direction_for_this_contrast': '今回の対照では予想と反対方向',
              'unresolved_interval_includes_zero': '区間がゼロを含み未解決'}
    text = '# ReDoのvリセットと正の片側化：初回リセットへの介入（1019）\n\n'
    text += f"主解析は **{labels[verdict['primary_verdict']]}**。rand10−zeroの50更新時・高開口率個体割合の対応差は{100 * primary['mean_difference']:+.2f}ポイント、95% bootstrap CI [{100 * primary['ci95_low']:+.2f}, {100 * primary['ci95_high']:+.2f}]。20 seedを反復単位とし、各seedの初回交換群を固定して比較した。\n\n"
    text += '5+1実ラベルCIFAR100のReLU MLP。初回の完全なReDoリセット後、交換された第2層の読み出し列だけを変更。zeroは元のReDo、keepは古い列の保持（momentは元のReDo同様ゼロ）、rand01/03/10は同じ独立ランダム方向で旧クラス中心化normの0.1/0.3/1.0倍。ランダム腕はクラス共通成分を除去した。\n\n'
    text += '| 腕 | 50更新・p≥.99割合 | 780更新・p≥.99割合 | 第3課題online | 第3課題末train | 第3課題末test |\n|---|---:|---:|---:|---:|---:|\n'
    for arm in ARMS:
        a, b = [next(r for r in grouped if r['arm'] == arm and r['cohort'] == 'first_reset' and r['event'] == 'after_step' and r['task'] == 3 and r['step'] == step) for step in (50, 780)]
        p = next(r for r in perf_summary if r['arm'] == arm and r['task'] == 3)
        text += f"| {arm} | {100 * a['task3_near99_mean']:.2f}% | {100 * b['task3_near99_mean']:.2f}% | {100 * p['online_acc']:.2f}% | {100 * p['train_acc']:.2f}% | {100 * p['test_acc']:.2f}% |\n"
    text += '\n開口率は第3課題の全2500学習画像。各seedの割合を等重み平均。onlineは780 minibatchの平均。成績差の対応CIはperformance_contrasts.csv。主比較以外は診断的副解析として全腕を報告した。\n\n'
    text += '| 腕−zero | Online差・ポイント[95%CI] | Test差・ポイント[95%CI] |\n|---|---:|---:|\n'
    for arm in ARMS[1:]:
        cols = []
        for key in ('online_acc', 'test_acc'):
            row = next(r for r in perf_contrasts if r['arm'] == arm and r['metric'] == key)
            cols.append(f"{100 * row['mean_difference']:+.2f} [{100 * row['ci95_low']:+.2f}, {100 * row['ci95_high']:+.2f}]")
        text += f'| {arm} | ' + ' | '.join(cols) + ' |\n'
    text += '\nkeep/randの非ゼロ読み出し腕では初期予測は一致していない。vを非ゼロ化すると初期logitとCEが変わるため、片側化と成績の同時変化だけで「片側化が成績差を媒介した」とは結論しない。入力側再初期化・旧特徴消失・初期予測の変化と、読み出し初期化の役割を区別する。\n\n'
    text += '| 腕 | 初回介入直後の次課題CE | zero後との差・logit RMS |\n|---|---:|---:|\n'
    for arm in CORE_ARMS:
        selected = [r for r in shocks if r['arm'] == arm and r['state'] == 'intervention_post' and r['scope'] == 'upcoming_task3']
        text += f"| {arm} | {np.mean([r['loss'] for r in selected]):.5f} | {np.mean([r['logit_diff_rms_from_zero_post'] for r in selected]):.5f} |\n"
    text += '\n上表は固定256画像の次課題プローブ。正確な初期衝撃はseed別shock.csvに残す。\n\n'
    text += '**平均入力の経路への介入B。** centerはz=w·(h−μ)+b′と書き換え、介入瞬間の前活性・予測を保ってからμを固定する。center_randomは同じnormのランダムμを使う比較。読み出しゼロは維持する。\n\n'
    text += '| centerの比較先 | 50更新high99差・ポイント[95%CI] | Online差・ポイント[95%CI] |\n|---|---:|---:|\n'
    for reference in ('zero', 'center_random'):
        columns = []
        for metric in ('task3_near99_mean', 'online_acc'):
            row = next(r for r in verdict['route_contrasts'] if r['reference'] == reference and r['metric'] == metric)
            columns.append(f"{100 * row['mean_difference']:+.2f} [{100 * row['ci95_low']:+.2f}, {100 * row['ci95_high']:+.2f}]")
        text += f'| {reference} | ' + ' | '.join(columns) + ' |\n'
    support = '満たした' if verdict['route_registered_support'] else '満たさなかった'
    text += f'\n両比較でhigh99差のCI上端<0、online差の平均≥+0.5ポイントかつCI下端>0という事前の支持条件を**{support}**。この書き換えは平均入力を通る更新経路とAdamの座標を変える。支持条件を満たしても、非線形ゲートが成績差を一意に媒介した証明とはしない。\n\n'
    text += '**正の同次性による再尺度化C。** gauge4/025はzeroと同じ最初の50更新後に、交換群のincomingを1/c倍、outgoingをc倍し、勾配履歴としてmomentを変換。介入瞬間の関数とゲートを保つ。\n\n'
    text += '| 腕−zero | 1〜50更新online差 | 51〜780更新online差・ポイント[95%CI] |\n|---|---:|---:|\n'
    for arm in ('gauge4', 'gauge025'):
        before = next(r for r in verdict['gauge_contrasts'] if r['arm'] == arm and r['metric'] == 'online_steps1_50')
        after = next(r for r in verdict['gauge_contrasts'] if r['arm'] == arm and r['metric'] == 'online_steps51_780')
        text += f"| {arm} | {100 * before['mean_difference']:.8f} | {100 * after['mean_difference']:+.2f} [{100 * after['ci95_low']:+.2f}, {100 * after['ci95_high']:+.2f}] |\n"
    text += '\nCは活動/readoutの尺度に対するAdam学習の依存を調べる。介入後のゲート軌道は固定されないため、その後までゲートを保った純粋なv介入ではない。図のgauge腕の50更新点は再尺度化直前の値で、変換直後は生データのgauge_postイベントに保存した。\n\n'
    text += f"再現性：zeroの元保存結果への一致、全腕のreset前・完全reset後・minibatch系列・交換mask一致、登録された座標変換以外の不変を検査済み。B/Cは介入瞬間の予測とゲートの不変も検査。CE分解の閉包最大誤差{verdict['ce_closure_max']:.3g}、自己抑制項の最小値{verdict['ce_brake_min']:.3g}。CE診断は固定256画像・float64での局所勾配であり、Adam実更新そのものではない。\n\n"
    text += 'この介入Aの対象は初回交換後の第3課題まで。長期・CNNへの一般化や、正の片側化から成績低下への矢印は別途必要。元の非ゼロnormで効果がない場合も、すべての読み出し介入が無効とはしない。\n\n'
    text += '数値正本：verdict.json、contrasts.csv、per_seed.csv、performance_contrasts.csv。初期衝撃はshock.csv、個々の実更新分解はactual_deltas.csv。会話中の観測値再計算はposthoc/summary.md。\n'
    (out / 'summary.md').write_text(text)


if __name__ == '__main__':
    main()
