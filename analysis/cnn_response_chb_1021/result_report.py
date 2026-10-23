#!/usr/bin/env python3
"""Independent CPU audit and report from saved CH/CHB curves (no torch import)."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUN = 'cnn_response_chb_1021'
ARMS = ('CH_self', 'CH_swap', 'CHB_self', 'CHB_swap')
SEEDS = list(range(10, 20))
UPDATES = (0, 100, 750, 3000, 15000, 30000)
WINDOWS = {'first3000': slice(0, 3000), 'steps1001_3000': slice(1000, 3000),
           'all': slice(None), 'last3000': slice(-3000, None)}
PAIRS = (('rescue', 'CH_swap', 'CH_self'), ('reverse', 'CHB_self', 'CHB_swap'))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def resolve(path):
    if path.exists():
        return path
    manifest = json.loads((ROOT/'results'/RUN/'backup_manifest.json').read_text())
    entries = manifest['files'] if isinstance(manifest, dict) else manifest
    rel = str(path.relative_to(ROOT))
    entry = next(e for e in entries if e.get('source_relative') == rel or e.get('source') == str(path))
    saved = Path(entry['backup'])
    require(saved.stat().st_size == entry['bytes'] and sha(saved) == entry['sha256'], f'Archive mismatch: {path}')
    return saved


def sha(path):
    h = hashlib.sha256()
    with resolve(path).open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''):
            h.update(b)
    return h.hexdigest()


def read_csv(path):
    with resolve(path).open() as f:
        return list(csv.DictReader(f))


def read_json(path):
    return json.loads(resolve(path).read_text())


def finite_json(obj):
    if isinstance(obj, dict):
        return {k: finite_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [finite_json(v) for v in obj]
    if isinstance(obj, (float, np.floating)) and not math.isfinite(obj):
        return None
    return obj


def exact_sign(values):
    positive = sum(float(x) > 0 for x in values)
    negative = sum(float(x) < 0 for x in values)
    n = positive + negative
    p = min(1., 2 * sum(math.comb(n, k) for k in range(min(positive, negative)+1)) / 2**n) if n else 1.
    return {'n_positive': positive, 'n_negative': negative,
            'n_tie': len(values)-n, 'p_two_sided': p}


def effect(values, bootstrap):
    return {'mean_difference': float(values.mean()), **exact_sign(values),
            'seed_differences': dict(zip(map(str, SEEDS), map(float, values))),
            'ci95_seed_bootstrap': np.percentile(values[bootstrap].mean(1), [2.5, 97.5]).tolist()}


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '|'+'|'.join(['---']*len(headers))+'|',
                      *['| '+' | '.join(str(v) for v in row)+' |' for row in rows]])


def run(results, out):
    verdict = read_json(results/'verdict.json')
    preflight_path = results/'checks'/'preflight.json'
    preflight = read_json(preflight_path)
    require(preflight['all_pass'] is True, 'GPU preflight did not pass')
    input_files = [results/'verdict.json', preflight_path, ROOT/'specs'/f'spec_{RUN}.md']
    summaries, arrays, means, diagnostics, failures, identities, commits, done_by_arm = {}, {}, {}, {}, {}, {}, {}, {}
    for arm in ARMS:
        folder = results/arm
        prov, done, anchor = [read_json(folder/name) for name in ('provenance.json', 'done.json', 'anchor.json')]
        require(prov['status'] == 'COMPLETE' and anchor['pass'], f'{arm}: incomplete/invalid initial anchor')
        require(prov['identity'] == done['identity'], f'{arm}: provenance identity mismatch')
        require(prov['identity']['arm'] == arm, f'{arm}: wrong arm identity')
        require(done['identity']['steps'] == 30000 and done['identity']['data']['seeds'] == SEEDS, f'{arm}: wrong protocol')
        for name, key in (('summary.csv', 'summary_sha256'), ('batch_curves.npz', 'curves_sha256'), ('state.pt', 'final_state_sha256')):
            require(sha(folder/name) == done[key], f'{arm}: bad terminal hash {name}')
        summaries[arm] = read_csv(folder/'summary.csv')
        require([int(r['seed']) for r in summaries[arm]] == SEEDS, f'{arm}: missing/order/duplicate seeds')
        require(all(r['arm'] == arm for r in summaries[arm]), f'{arm}: mixed summary arms')
        with np.load(resolve(folder/'batch_curves.npz')) as data:
            require(data['seeds'].tolist() == SEEDS, f'{arm}: wrong curve seed identities')
            require(data['accuracy'].shape == data['ce'].shape == (10, 30000), f'{arm}: incomplete curves')
            acc, ce = data['accuracy'].copy(), data['ce'].copy()
            require(np.isfinite(acc).all() and ((acc >= 0) & (acc <= 1)).all(), f'{arm}: invalid accuracy')
            require(not np.isnan(ce).any() and not np.isneginf(ce).any(), f'{arm}: invalid CE failure representation')
            require(data['updates'].shape == (30000, 10, 5, 5), f'{arm}: incomplete update diagnostics')
            require(data['B_fire_fraction'].shape == (30000, 10, 4), f'{arm}: incomplete B diagnostics')
        failed = []
        for i, seed in enumerate(SEEDS):
            is_failed = summaries[arm][i]['failed'].lower() == 'true'
            bad_at = np.flatnonzero(~np.isfinite(ce[i]))
            require(is_failed == bool(len(bad_at)), f'{arm}/{seed}: failure flag differs from curve')
            if len(bad_at):
                first = int(bad_at[0])
                require((acc[i, first:] == 0).all() and np.isposinf(ce[i, first:]).all(), f'{arm}/{seed}: failure policy not retained')
                failed.append({'seed': seed, 'first_failed_update': first+1})
        require(sorted(done['failed_seeds']) == sorted(r['seed'] for r in failed), f'{arm}: failed seed identity mismatch')
        means[arm] = {}
        arrays[arm] = {}
        for metric, window in WINDOWS.items():
            values = acc[:, window].mean(1, dtype=np.float64)
            arrays[arm][metric] = values
            for prefix, curve in (('', acc), ('ce_', ce)):
                expected = curve[:, window].mean(1, dtype=np.float64)
                actual = np.array([float(r[prefix+metric]) for r in summaries[arm]])
                require(np.array_equal(expected, actual), f'{arm}: summary differs from saved curve {prefix+metric}')
                means[arm][prefix+metric] = float(expected.mean())
        epochrows = read_csv(folder/'epochs.csv')
        require(all(r['arm'] == arm for r in epochrows), f'{arm}: mixed epoch arms')
        require(len(epochrows) == 4000 and {(int(r['seed']), int(r['epoch'])) for r in epochrows}
                == {(s, e) for s in SEEDS for e in range(1, 401)}, f'{arm}: incomplete epochs')
        ds = read_csv(folder/'diagnostics.csv')
        require(all(r['arm'] == arm for r in ds), f'{arm}: mixed diagnostic arms')
        require(len(ds) == 60 and {(int(r['seed']), int(r['update'])) for r in ds}
                == {(s, u) for s in SEEDS for u in UPDATES}, f'{arm}: incomplete diagnostic checkpoints')
        units = read_csv(folder/'unit_diagnostics.csv')
        require(len(units) == 60*2*(16+16+100+100), f'{arm}: incomplete unit records')
        require(all(r['arm'] == arm for r in units), f'{arm}: mixed unit arms')
        unit_keys = {(int(r['seed']), int(r['update']), r['site'], r['kind'], int(r['unit'])) for r in units}
        expected_units = {(s, u, layer, kind, j) for s in SEEDS for u in UPDATES
                          for layer, width in (('c1', 16), ('c2', 16), ('f1', 100), ('f2', 100))
                          for kind in ('raw', 'effective') for j in range(width)}
        require(unit_keys == expected_units, f'{arm}: duplicate/missing unit identities')
        diagnostics[arm] = {str(u): {k: float(np.mean([float(r[k]) for r in ds if int(r['update']) == u]))
                                     for k in ds[0] if k not in ('arm', 'seed', 'update', 'failed')} for u in UPDATES}
        failures[arm] = failed
        identities[arm] = {k: v for k, v in prov['identity'].items() if k != 'arm'}
        commits[arm] = prov['git_hash']
        done_by_arm[arm] = done
        input_files.extend(folder/name for name in ('summary.csv', 'provenance.json', 'done.json', 'anchor.json',
                                                     'batch_curves.npz', 'epochs.csv', 'diagnostics.csv', 'unit_diagnostics.csv'))
    base = identities[ARMS[0]]
    require(all(v == base for v in identities.values()), 'Arm conditions differ')
    require(len(set(commits.values())) == 1, 'Arms used different implementation commits')
    require(len({d['labels_sha256'] for d in done_by_arm.values()}) == 1
            and len({d['streams_sha256'] for d in done_by_arm.values()}) == 1, 'Paired labels/batches differ')
    contract = preflight['contract']
    require(contract['identity'] == base and contract['implementation_commit'] == commits[ARMS[0]], 'Preflight/run mismatch')
    for path, expected in base['code_sha256'].items():
        blob = subprocess.check_output(['git', '-C', str(ROOT), 'show', commits[ARMS[0]]+':'+path])
        require(hashlib.sha256(blob).hexdigest() == expected, f'Immutable code hash mismatch: {path}')
    boots = np.random.default_rng(1021).integers(0, 10, size=(20000, 10))
    effects = {metric: {name: effect(arrays[plus][metric]-arrays[minus][metric], boots)
                       for name, plus, minus in PAIRS} for metric in WINDOWS}
    primary = effects['first3000']
    previous = 0.
    for i, name in enumerate(sorted(primary, key=lambda k: primary[k]['p_two_sided'])):
        item = primary[name]
        item['p_holm'] = min(1., max(previous, (2-i)*item['p_two_sided']))
        previous = item['p_holm']
        item['significant'] = item['p_holm'] <= .05
        mean, pos, neg = item['mean_difference'], item['n_positive'], item['n_negative']
        item['direction'] = ('tie' if mean == 0 and pos == neg else 'predicted' if mean > 0 and pos > neg
                             else 'opposite' if mean < 0 and neg > pos else 'heterogeneous')
        saved = verdict['tests'][name]
        for key in ('mean_difference', 'n_positive', 'n_negative', 'n_tie', 'p_two_sided', 'p_holm'):
            require(abs(item[key]-saved[key]) < 1e-14, f'Registered statistic mismatch: {name}/{key}')
        require(np.allclose(item['ci95_seed_bootstrap'], saved['ci95_seed_bootstrap'], rtol=0, atol=1e-14), 'Bootstrap mismatch')
        require(item['direction'] == saved['direction'] and item['significant'] == saved['significant'], 'Saved direction/significance mismatch')
        require(item['seed_differences'] == saved['seed_differences'], 'Saved per-seed differences mismatch')
    supported = sum(t['significant'] and t['direction'] == 'predicted' for t in primary.values())
    label = ('HETEROGENEOUS' if any(t['significant'] and t['direction'] == 'heterogeneous' for t in primary.values())
             else 'OPPOSITE' if any(t['significant'] and t['direction'] == 'opposite' for t in primary.values())
             else 'BIDIRECTIONAL' if supported == 2 else 'ONE_DIRECTION' if supported == 1 else 'NONE')
    require(label == verdict['primary'], 'Independent verdict differs')
    report = {'experiment': RUN, 'status': 'COMPLETE_AUDITED', 'primary': label,
              'audit_script_sha256': sha(Path(__file__)),
              'primary_matches_registered': True, 'effects': effects, 'means': means,
              'per_seed_summary': summaries, 'diagnostics_seed_means': diagnostics, 'failures': failures,
              'failure_count': sum(map(len, failures.values())), 'run_commit': commits[ARMS[0]],
              'registered_spec_commit': contract['registered_spec_commit'],
              'audit': {'paired_streams_equal': True, 'all_seed_curves_complete': True,
                        'all_identities_match': True, 'immutable_code_hashes_match': True,
                        'all_initial_anchors_exact': True, 'summary_matches_saved_curves_exactly': True},
              'input_sha256': {str(p.relative_to(ROOT)): sha(p) for p in input_files},
              'undefined_numeric_values_serialized_as_null': True,
              'no_new_forward_or_training': True}
    out.mkdir(parents=True, exist_ok=True)
    (out/'report.json').write_text(json.dumps(finite_json(report), ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    lines = ['# CNN CH′/CH′＋B 全4層応答交換：登録走の結果', '',
             f"主判定 **`{label}`**。保存された4腕×10 seed×30000更新から主評価を独立再計算し、登録集計と一致した。",
             f"失敗seedは{report['failure_count']}件。全seedを含み、失敗後は登録どおり精度0・CE+∞として保持する。", '',
             '## 主評価：課題31の更新1–3000', '',
             table(['比較', '判定用平均差(pt)', '95% seed bootstrap CI(pt)', '正/負/tie', 'Holm p'],
                   [[name, f"{100*d['mean_difference']:+.5f}", ', '.join(f'{100*x:+.5f}' for x in d['ci95_seed_bootstrap']),
                     f"{d['n_positive']}/{d['n_negative']}/{d['n_tie']}", f"{d['p_holm']:.8g}"] for name, d in primary.items()]), '',
             'rescue=CH_swap−CH_self、reverse=CHB_self−CHB_swap（正が悪化）。正確両側符号検定2本をHolm補正。CIはseed1021・20000回。', '',
             table(['seed', *ARMS, 'rescue差(pt)', 'reverse差(pt)'],
                   [[s, *[f"{100*arrays[a]['first3000'][i]:.5f}" for a in ARMS],
                     *[f"{100*primary[n]['seed_differences'][str(s)]:+.5f}" for n in ('rescue', 'reverse')]] for i, s in enumerate(SEEDS)]), '',
             '## 副窓：主判定を置き換えない記述統計', '',
             table(['腕', *WINDOWS], [[a, *[f'{100*means[a][k]:.5f}' for k in WINDOWS]] for a in ARMS]), '',
             '各値は更新前online accuracy（%）。1001–3000はAdam初期過渡への感度記述。全体・末尾の不利益も保持する。', '',
             '### 交差エントロピー（CE）も全seedを保持する', '',
             table(['腕', *WINDOWS], [[a, *[f'{means[a]["ce_"+k]:.6f}' for k in WINDOWS]] for a in ARMS]), '',
             '更新前の平均CE。accuracyの主判定は、CEや全指標の改善を意味しない。巨大だが有限な損失も除外・上限処理せず、同じ固定窓で併記する。', '',
             '## 実際に使われた応答と限界', '',
             '次は各層の実効native平均微分（convは受け手の実際のpool winner上、FCは全unit平均）のseed平均。', '']
    for layer in ('c1', 'c2', 'f1', 'f2'):
        key = layer+'_effective_'+('pool_mean_d_mean' if layer.startswith('c') else 'native_mean_d_mean')
        lines += [f'### {layer}', '', table(['更新', *ARMS],
                  [[u, *[f"{diagnostics[a][str(u)][key]:.6g}" for a in ARMS]] for u in UPDATES]), '']
    lines += ['介入はReLU局所微分ゲートへの操作。全Jacobian、自然なCHB利益の媒介率、ゲート密度対配置、責任層は同定しない。受け手のAdam・EMA・Bを維持するため、適応状態との相互作用を含む。後半には凍結参照からの変位と固定オフセットの影響も含む。', '',
              '旧LR/LRcのc1のみの結果は別実験として保持し、合算・撤回しない。非有意は同等性や機構不在を意味しない。', '',
              '## 再現記録', '', f"実装commit `{report['run_commit']}`、登録spec `{report['registered_spec_commit']}`。",
              '全腕identity、対応ラベル・batch列、terminal hash、全seed曲線、400 epoch、全診断時点、実装のGit blob hashを照合した。詳細・全層診断・失敗箇所・input hashはreport.json。', '']
    (out/'report.md').write_text('\n'.join(lines))
    print(json.dumps({'status': report['status'], 'primary': label, 'failure_count': report['failure_count'], 'report': str(out/'report.md')}, ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, default=ROOT/'results'/RUN)
    p.add_argument('--out', type=Path, default=ROOT/'results'/RUN/'report')
    args = p.parse_args()
    run(args.results.resolve(), args.out.resolve())
