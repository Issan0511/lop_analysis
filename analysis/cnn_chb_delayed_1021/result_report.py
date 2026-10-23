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
RUN = 'cnn_chb_delayed_1021'
ARMS = ('CH_continue', 'CH_addB')
SEEDS = list(range(10, 20))
UPDATES = (0, 1, 100, 750, 3000, 15000, 30000)
WINDOWS = {'first3000': slice(0, 3000), 'steps1001_3000': slice(1000, 3000),
           'all': slice(None), 'last3000': slice(-3000, None)}



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
            require(data['updates'].shape == (30000, 10, 5, 7), f'{arm}: incomplete update diagnostics')
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
                arrays[arm][prefix+metric] = expected
        epochrows = read_csv(folder/'epochs.csv')
        require(all(r['arm'] == arm for r in epochrows), f'{arm}: mixed epoch arms')
        require(len(epochrows) == 4000 and {(int(r['seed']), int(r['epoch'])) for r in epochrows}
                == {(s, e) for s in SEEDS for e in range(1, 401)}, f'{arm}: incomplete epochs')
        ds = read_csv(folder/'diagnostics.csv')
        require(all(r['arm'] == arm for r in ds), f'{arm}: mixed diagnostic arms')
        require(len(ds) == 70 and {(int(r['seed']), int(r['update'])) for r in ds}
                == {(s, u) for s in SEEDS for u in UPDATES}, f'{arm}: incomplete diagnostic checkpoints')
        units = read_csv(folder/'unit_diagnostics.csv')
        require(len(units) == 70*(16+16+100+100), f'{arm}: incomplete unit records')
        require(all(r['arm'] == arm for r in units), f'{arm}: mixed unit arms')
        unit_keys = {(int(r['seed']), int(r['update']), r['site'], r['kind'], int(r['unit'])) for r in units}
        expected_units = {(s, u, layer, kind, j) for s in SEEDS for u in UPDATES
                          for layer, width in (('c1', 16), ('c2', 16), ('f1', 100), ('f2', 100))
                          for kind in ('raw',) for j in range(width)}
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
    # A second implementation uses direct comparisons for signs, independently
    # from the training module's failure-policy benefit vector.
    boots=np.random.default_rng(1021).integers(0,10,size=(20000,10))
    primary={}
    effects={}
    per_seed=[]
    for window in WINDOWS:
        effects[window]={}
        for metric,prefix,scale in (('accuracy','',100.),('ce','ce_',1.)):
            a=arrays['CH_continue'][prefix+window];b=arrays['CH_addB'][prefix+window]
            pos=(b>a) if metric=='accuracy' else (b<a)
            neg=(b<a) if metric=='accuracy' else (b>a)
            signs=pos.astype(int)-neg.astype(int)
            finite=bool(np.isfinite(a).all() and np.isfinite(b).all())
            with np.errstate(invalid='ignore'):
                delta=b-a
            effects[window][metric]={'addB_minus_continue':delta.tolist(),
                'mean_addB_minus_continue':float(delta.mean()) if finite else None}
            for i,seed in enumerate(SEEDS):
                per_seed.append(dict(window=window,metric=metric,seed=seed,CH_continue=float(a[i]),
                                     CH_addB=float(b[i]),addB_minus_continue=float(delta[i])))
            if window=='first3000':
                benefit=delta if metric=='accuracy' else -delta
                primary[metric]=exact_sign(signs)|{'finite':finite, 'benefit_signs':signs.tolist(),
                    'mean_benefit':float(benefit.mean()) if finite else None,
                    'ci95_seed_bootstrap':np.percentile(benefit[boots].mean(1),[2.5,97.5]).tolist() if finite else None}
    previous=0.
    for rank,key in enumerate(sorted(primary,key=lambda k:primary[k]['p_two_sided'])):
        t=primary[key];t['p_holm']=min(1.,max(previous,(2-rank)*t['p_two_sided']));previous=t['p_holm']
        t['verdict']='INCONCLUSIVE'
        if not t['finite']: t['verdict']='INCONCLUSIVE_NONFINITE'
        elif t['p_holm']<=.05:
            m=t['mean_benefit'];p=t['n_positive'];n=t['n_negative']
            t['verdict']='IMPROVES' if m>0 and p>n else 'WORSENS' if m<0 and n>p else 'HETEROGENEOUS'
    require(primary==verdict['tests'],'Independent primary differs from registered summary')
    # Audit every saved projection, not just successful seeds or checkpoints.
    max_excess=0.;hits=0;total=0
    for epoch in range(1,401):
        path=results/'CH_addB'/'projection'/f'epoch{epoch:03d}.npz'
        with np.load(resolve(path)) as z:
            for layer in ('c1','c2','f1','f2'):
                width=z[layer+'_s'];before=z[layer+'_before'];after=z[layer+'_after'];hit=z[layer+'_hit']
                finite=np.isfinite(width)&np.isfinite(before)&np.isfinite(after)
                require(np.array_equal(hit,(np.abs(before.astype(float))>.2*width)), 'B hit inconsistent')
                require(np.array_equal(after[~hit],before[~hit],equal_nan=True),'In-bound bias changed')
                excess=np.maximum(0,np.abs(after.astype(float))-.2*width)/np.maximum(1,.2*width)
                if finite.any():max_excess=max(max_excess,float(excess[finite].max()))
                hits+=int(hit.sum());total+=hit.size
        input_files.append(path)
    require(max_excess<=2e-7,'B cap exceeds float32 rounding')
    report={'experiment':RUN,'status':'COMPLETE_AUDITED','primary':primary,'effects':effects,'means':means,
            'per_seed':per_seed,'diagnostics_seed_means':diagnostics,'failures':failures,
            'failure_count':sum(map(len,failures.values())), 'run_commit':commits[ARMS[0]],
            'registered_spec_commit':contract['registered_spec_commit'], 'audit_script_sha256':sha(Path(__file__)),
            'audit':{'primary_matches_registered':True,'paired_streams_equal':True,'all_curves_complete':True,
                     'immutable_code_hashes_match':True,'all_source_qualification_passed':True,
                     'projection_max_relative_excess':max_excess,'projection_total_hit_fraction':hits/total},
            'input_sha256':{str(p.relative_to(ROOT)):sha(p) for p in input_files},
            'scope':'delayed B from degraded CH; no artificial fields, complete restoration or mediation claim',
            'undefined_numeric_values_serialized_as_null':True,'no_new_forward_or_training':True}
    out.mkdir(parents=True,exist_ok=True)
    (out/'report.json').write_text(json.dumps(finite_json(report),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    with (out/'per_seed_windows.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(per_seed[0]));writer.writeheader();writer.writerows(per_seed)
    lines=['# 劣化後のCHにBを追加：直接回復実験','',
           f"精度: **{primary['accuracy']['verdict']}** / CE: **{primary['ce']['verdict']}**。全2腕×10seed×30000更新。失敗 {report['failure_count']} seed。",'',
           '## first3000の登録判定','',table(['項目','正の便益','95% CI','正/負/tie','Holm p','判定'],
           [[k,t['mean_benefit'],t['ci95_seed_bootstrap'],f"{t['n_positive']}/{t['n_negative']}/{t['n_tie']}",t['p_holm'],t['verdict']] for k,t in primary.items()]),'',
           '精度便益=addB−継続（表では0–1単位）、CE便益=継続−addB。両側exact sign testを2項目Holm補正。', '']
    for metric,prefix,scale in (('accuracy','',100.),('CE','ce_',1.)):
        lines += [f'## {metric}：全窓・全seed','']
        for window in WINDOWS:
            lines += [f'### {window}','',table(['seed','CH_continue','CH_addB','addB−continue'],
                [[seed,*(f'{scale*arrays[a][prefix+window][i]:.6f}' for a in ARMS),
                  f"{scale*(arrays['CH_addB'][prefix+window][i]-arrays['CH_continue'][prefix+window][i]):+.6f}"] for i,seed in enumerate(SEEDS)]), '',
                f"seed平均: 継続 {scale*means['CH_continue'][prefix+window]:.6f} / addB {scale*means['CH_addB'][prefix+window]:.6f}",'']
    lines += ['## 自然応答・幅・biasの診断','']
    for layer in ('c1','c2','f1','f2'):
        lines += [f'### {layer}','',table(['update','arm','native d','width','bias/width','r'],
                [[u,a,*(f'{diagnostics[a][str(u)][layer+k]:.6g}' for k in ('_native_mean_d_mean','_image_width_mean','_bias_over_width_mean','_r_mean'))] for u in UPDATES for a in ARMS]),'']
    lines += ['初回Bは最初のAdam/EMA後。学習開始からのCHB予防とは別。改善は劣化したCH継続に対するもの。自然Bはbias/forward/応答/尺度/Adam軌道を同時に変え、応答だけの媒介率や責任層は同定しない。旧人工応答交換・capを保持する。','',
              f"登録/実行commit: `{report['run_commit']}`。全曲線・全窓・全seed・paired streams・終端SHA・400回の射影保存を独立照合。",'']
    (out/'report.md').write_text('\n'.join(lines))
    print(json.dumps({'status':report['status'],'primary':{k:v['verdict'] for k,v in primary.items()},'failure_count':report['failure_count']}))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--results',type=Path,default=ROOT/'results'/RUN)
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    run(args.results,args.out or args.results/'report')
