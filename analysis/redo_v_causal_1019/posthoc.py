#!/usr/bin/env python3
"""Reproduce earlier chat summaries from immutable archived arrays; no training."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = Path('/home/issan/Projects/obsidian-research-data')


def arrays(path):
    with np.load(path) as handle:
        return {k: handle[k] for k in handle.files}


def csv_write(path, rows):
    if not rows:
        return
    with path.open('w') as handle:
        writer = csv.DictWriter(handle, list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)


def source_record(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}


def summarize_seed_means(rows, group_keys, value_keys):
    output = []
    for group in dict.fromkeys(tuple(row[k] for k in group_keys) for row in rows):
        selected = [row for row in rows if tuple(row[k] for k in group_keys) == group]
        result = dict(zip(group_keys, group))
        result['seeds'] = len(selected)
        for key in value_keys:
            values = np.asarray([row[key] for row in selected], dtype=float)
            finite = values[np.isfinite(values)]
            result[key] = float(finite.mean()) if len(finite) else None
        output.append(result)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=ROOT / 'results/redo_v_causal_1019/posthoc')
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    sources = []
    vpath = ARCHIVE / 'redo_v_opening_1019/raw/individuals.npz'
    gpath = ARCHIVE / 'redo_open_gradient_1019/raw/individuals.npz'
    vdata, gdata = arrays(vpath), arrays(gpath)
    sources.extend(source_record(p) for p in (vpath, gpath))

    # Per-seed normalized weights avoid silently pooling seeds with large norms.
    weighted = []
    for method in ('redo', 'none'):
        for stage in np.unique(gdata['stage']):
            for seed in range(20):
                mask = (gdata['method'] == method) & (gdata['stage'] == stage) & (gdata['seed'] == seed)
                if not mask.any():
                    continue
                norm = gdata['v_norm'][mask]
                p = gdata['p_full_current'][mask]
                rms = gdata['delta_rms'][mask]
                row = {'method': method, 'stage': str(stage), 'seed': seed, 'n': int(mask.sum())}
                for name, power in [('uniform', 0), ('v', 1), ('v2', 2)]:
                    weight = norm ** power
                    denominator = weight.sum()
                    row[f'opening_{name}'] = float(np.dot(weight, p) / denominator)
                    row[f'delta_rms_{name}'] = float(np.dot(weight, rms) / denominator)
                    row[f'high99_weight_share_{name}'] = float(weight[p >= .99].sum() / denominator)
                weighted.append(row)
    weight_keys = [key for key in weighted[0] if key not in ('method', 'stage', 'seed')]
    weighted_summary = summarize_seed_means(weighted, ['method', 'stage'], weight_keys)

    # Individual-level means and counts explicitly use the fixed global probe.
    cohorts = []
    cohort_counts = []
    for stage in ('post_reset', 't3_s10', 't3_s50', 't3_s780'):
        for cohort in ('recycled', 'survivors'):
            mask = (vdata['method'] == 'redo') & (vdata['stage'] == stage) & (vdata['cohort'] == cohort)
            p = vdata['p_global'][mask]
            cohort_counts.append({'stage': stage, 'cohort': cohort, 'n': int(mask.sum()),
                                  'high99': int((p >= .99).sum()), 'full': int((p == 1).sum()),
                                  'opening_pooled_mean': float(p.mean())})
            for seed in range(20):
                selected = mask & (vdata['seed'] == seed)
                row = {'stage': stage, 'cohort': cohort, 'seed': seed, 'n': int(selected.sum())}
                for key in ('v_norm', 'raw_v_norm', 'p_global', 'p_task3', 'activation_rms', 'logit_contribution_rms'):
                    row[key] = float(vdata[key][selected].mean())
                cohorts.append(row)
    cohort_keys = [key for key in cohorts[0] if key not in ('stage', 'cohort', 'seed')]
    cohort_summary = summarize_seed_means(cohorts, ['stage', 'cohort'], cohort_keys)

    geometry = []
    context_opening = []
    context_performance = []
    for method in ('redo', 'none'):
        for start in (0, 10):
            folder = ARCHIVE / f'redo_positive_lineage_1019/raw/{method}_s{start}-{start + 9}'
            path = folder / 'replay/per_task.csv'
            with path.open() as handle:
                for row in csv.DictReader(handle):
                    if int(row['task']) in (1, 2, 3, 29, 30):
                        context_performance.append({'method': method, 'seed': int(row['seed']), 'task': int(row['task']),
                                                    **{key: float(row[key]) for key in ('online_acc', 'train_acc', 'test_acc')}})
            sources.append(source_record(path))
            meta = json.loads((folder / 'trace/events.json').read_text())
            units = arrays(folder / 'trace/units.npz')
            sources.extend(source_record(p) for p in (folder / 'trace/events.json', folder / 'trace/units.npz'))
            for index, event in enumerate(meta['events']):
                if event['event'] != 'task_end' or event['task'] not in (1, 2, 3, 29, 30):
                    continue
                for r, seed in enumerate(meta['seeds']):
                    p = units['g_l2_p'][index, r]
                    pc = units['current_l2_p'][index, r]
                    context_opening.append({'method': method, 'seed': seed, 'task': event['task'],
                                            'global_fullopen_count': int((p == 1).sum()),
                                            'global_high99_count': int((p >= .99).sum()), 'global_p_mean': float(p.mean()),
                                            'current_fullopen_count': int((pc == 1).sum()),
                                            'current_high99_count': int((pc >= .99).sum()), 'current_p_mean': float(pc.mean())})
    context_performance_summary = summarize_seed_means(context_performance, ['method', 'task'], ['online_acc', 'train_acc', 'test_acc'])
    context_opening_summary = summarize_seed_means(context_opening, ['method', 'task'], [key for key in context_opening[0] if key not in ('method', 'seed', 'task')])
    for start in (0, 10):
        folder = ARCHIVE / f'redo_positive_lineage_1019/raw/redo_s{start}-{start + 9}/trace'
        meta_path, units_path = folder / 'events.json', folder / 'units.npz'
        meta = json.loads(meta_path.read_text())
        units = arrays(units_path)
        sources.extend(source_record(p) for p in (meta_path, units_path))
        before = next(i for i, e in enumerate(meta['events']) if e['task'] == 2 and e['event'] == 'reset_pre')
        masks = units['reset_mask'][before, :, 1].astype(bool)
        for stage, target_event, target_step in [('post_reset', 'reset_outgoing_2', 780), ('t3_s780', 'after_step', 780)]:
            task = 2 if stage == 'post_reset' else 3
            index = next(i for i, e in enumerate(meta['events']) if e['task'] == task and e['event'] == target_event and e['step'] == target_step)
            path = folder / f'deep_{index:04d}.npz'
            deep = arrays(path)
            sources.append(source_record(path))
            h, z, w, bias = [deep[key].astype(float) for key in ('h1', 'z2', 'W2', 'b2')]
            activity = np.maximum(z, 0)
            mu = h.mean(axis=1)
            wn = np.linalg.norm(w, axis=2)
            inner = np.einsum('rij,rj->ri', w, mu)
            cosine = np.divide(inner, wn * np.linalg.norm(mu, axis=1)[:, None], out=np.zeros_like(inner),
                               where=wn * np.linalg.norm(mu, axis=1)[:, None] > 0)
            metrics = {'mean_z': z.mean(axis=1), 'sd_z': z.std(axis=1), 'rms_z': np.sqrt((z * z).mean(axis=1)),
                       'activation_rms': np.sqrt((activity * activity).mean(axis=1)), 'activation_sd': activity.std(axis=1),
                       'incoming_norm': wn, 'bias': bias, 'w_dot_mu': inner, 'cosine_w_mu': cosine,
                       'mean_squared_z_mean': z.mean(axis=1) ** 2, 'mean_z_variance': z.var(axis=1)}
            for r, seed in enumerate(meta['seeds']):
                for label, mask in [('recycled', masks[r]), ('survivors', ~masks[r])]:
                    row = {'stage': stage, 'cohort': label, 'seed': seed, 'n': int(mask.sum())}
                    row.update({key: float(value[r, mask].mean()) for key, value in metrics.items()})
                    geometry.append(row)
    geometry_keys = [key for key in geometry[0] if key not in ('stage', 'cohort', 'seed')]
    geometry_summary = summarize_seed_means(geometry, ['stage', 'cohort'], geometry_keys)
    lookup_w = {(r['method'], r['stage']): r for r in weighted_summary}
    lookup_c = {(r['stage'], r['cohort']): r for r in cohort_summary}
    lookup_g = {(r['stage'], r['cohort']): r for r in geometry_summary}
    full = next(r for r in cohort_counts if r['stage'] == 't3_s780' and r['cohort'] == 'recycled')
    zero, end = lookup_g[('post_reset', 'recycled')], lookup_g[('t3_s780', 'recycled')]
    report = {
        'tier': 'reproduced_posthoc_observations_no_intervention',
        'aggregation': 'Unless explicitly pooled, calculate unit means or normalized weights within each seed, then equal-weight arithmetic mean across 20 seeds.',
        'v_definition': 'L2 across 100 classes after removing each column class-common component',
        'weighted_probe': 'full current task training images; 2500 images on hard tasks3 and29',
        'geometry_probe': 'fixed global1000 training images (10 per class)',
        'gradient_definition': 'per-input dL/dz RMS across saved256-image current-task probe, without mean-loss 1/256 factor',
        'weighted_summary': weighted_summary, 'cohort_summary': cohort_summary,
        'cohort_counts': cohort_counts, 'geometry_summary': geometry_summary,
        'sources': list({record['path']: record for record in sources}.values()),
        'context_performance_summary': context_performance_summary, 'context_opening_summary': context_opening_summary,
        'interpretation': ['Positive one-sidedness is observed after complete ReDo reset; its outgoing-reset cause is not established by these observations.',
                           'The activity RMS rise mostly accompanies mean-position growth, with a smaller width increase.',
                           'v2-weighted opening is a normalized scalar sensitivity summary, not the complete SGD/Adam adaptation kernel.',
                           'Further v-weighting of dL/dz is descriptive and does not equal actual parameter updates.',
                           'Fully open on1000 finite probe images does not prove globally affine behavior on all possible inputs.']}
    (out / 'verdict.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    for name, rows in [('weighted_per_seed', weighted), ('weighted_summary', weighted_summary), ('cohorts_per_seed', cohorts),
                       ('cohorts_summary', cohort_summary), ('cohort_counts', cohort_counts), ('geometry_per_seed', geometry), ('geometry_summary', geometry_summary),
                       ('context_performance_per_seed', context_performance), ('context_performance_summary', context_performance_summary),
                       ('context_opening_per_seed', context_opening), ('context_opening_summary', context_opening_summary)]:
        csv_write(out / f'{name}.csv', rows)
    text = '# ReDoの正の片側化：会話中の観測値の再計算（1019）\n\n'
    text += '新規学習なし。保存された個体配列・深部状態から再計算した。特記なければ各seed内で平均・重みの正規化を行い、20 seedを等重みに平均する。vは100クラス共通成分を除いた出力重みのL2norm。\n\n'
    text += '| 時点 | 開口率・単純平均 | v重み | v²重み | 全開寄り個体のv²割合 |\n|---|---:|---:|---:|---:|\n'
    for stage in ('t3_s780', 't29_end'):
        row = lookup_w[('redo', stage)]
        text += f"| {stage} | {row['opening_uniform']:.6f} | {row['opening_v']:.6f} | {row['opening_v2']:.6f} | {row['high99_weight_share_v2']:.6f} |\n"
    text += '\n開口率は各課題の全学習画像。全開寄りはp≥0.99。v²重み開口率は入力bias経路などの局所感度の要約であり、入力共分散・誤差方向・読み出し学習・Adam状態を含む適応全体を表さない。\n\n'
    text += '| 時点 | dL/dz RMS・単純平均 | v重み | v²重み |\n|---|---:|---:|---:|\n'
    for stage in ('t3_s780', 't29_end'):
        row = lookup_w[('redo', stage)]
        text += f"| {stage} | {row['delta_rms_uniform']:.6f} | {row['delta_rms_v']:.6f} | {row['delta_rms_v2']:.6f} |\n"
    text += '\ndL/dzには既にvが含まれる。追加重み付けは記述統計であり実更新量ではない。\n\n'
    text += f"第3課題末の初回リセット群{full['n']}個体のうち、固定global1000画像でp≥0.99は{full['high99']}個体、厳密なp=1は{full['full']}個体。個体をプールした平均開口率は{full['opening_pooled_mean']:.6f}。\n\n"
    text += '| リセット群・global1000 | リセット直後 | 第3課題末 |\n|---|---:|---:|\n'
    for key in ('mean_z', 'sd_z', 'activation_rms', 'activation_sd', 'incoming_norm', 'bias', 'w_dot_mu', 'cosine_w_mu'):
        text += f'| {key} | {zero[key]:.6f} | {end[key]:.6f} |\n'
    text += '\n活動RMSの増大と分布幅の増大を同一視しない。正側への平均移動が大きい。有限プローブ上の全開は、その集合で折れ目を使わないことを意味するが、全入力上での線形化や成績低下の因果証明ではない。\n\n'
    text += '| 第3課題末・global1000 | リセット群 | 継続群 |\n|---|---:|---:|\n'
    for key in ('v_norm', 'activation_rms', 'logit_contribution_rms'):
        a, b = lookup_c[('t3_s780', 'recycled')], lookup_c[('t3_s780', 'survivors')]
        text += f'| {key} | {a[key]:.6f} | {b[key]:.6f} |\n'
    text += '\n予測へのRMS寄与には入力共通成分も含まれるので、特徴の識別力と同じではない。\n\n'
    text += '| 課題末 | ReDo online | 無リセット online | 差・ポイント | ReDo全開個体数/100 | 無リセット全開個体数/100 |\n|---|---:|---:|---:|---:|---:|\n'
    for task in (1, 2, 3, 29, 30):
        a, b = [next(r for r in context_performance_summary if r['method'] == method and r['task'] == task) for method in ('redo', 'none')]
        c, d = [next(r for r in context_opening_summary if r['method'] == method and r['task'] == task) for method in ('redo', 'none')]
        text += f"| {task} | {100*a['online_acc']:.4f}% | {100*b['online_acc']:.4f}% | {100*(a['online_acc']-b['online_acc']):+.4f} | {c['global_fullopen_count']:.2f} | {d['global_fullopen_count']:.2f} |\n"
    text += '\n全開個体数は固定global1000画像・全100個体のseed平均。課題1/2は初回リセット前の段階対照であり、成熟した前段表現を持つ交換直後との厳密な一致対照ではない。第3課題のReDo−無リセット差と、後期の優良活性化手法に対する差を混同しない。第29課題ではReDoは無リセットを上回る。\n\n原典のパス・SHA256はverdict.json、seed別は各per_seed.csv。\n'
    (out / 'summary.md').write_text(text)
    print(json.dumps({'counts': full, 'weighted_task3': lookup_w[('redo', 't3_s780')],
                      'geometry_reset': zero, 'geometry_end': end}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
