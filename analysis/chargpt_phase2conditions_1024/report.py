#!/usr/bin/env python3
"""Registered aggregates of saved-state condition measurements."""
import importlib.util
import json
from pathlib import Path

sp = importlib.util.spec_from_file_location('conditions', Path(__file__).with_name('measure.py'))
R = importlib.util.module_from_spec(sp)
sp.loader.exec_module(R)
np, OUT, dump, csvwrite = R.np, R.OUT, R.dump, R.csvwrite


def median(a):
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.median(a)) if len(a) else None


def ratio(a, b):
    return float(np.linalg.norm(a)/max(np.linalg.norm(b), 1e-300))


def fmt(v, digits=3):
    return 'NA' if v is None else f'{v:.{digits}f}'


def span(values, digits=3):
    vs = [float(x) for x in values if x is not None and np.isfinite(x)]
    return 'NA' if not vs else fmt(min(vs), digits)+'–'+fmt(max(vs), digits)


def zeros(a):
    return R.S.numeric_zero(a, 1e-10)


def main():
    states, components, comparisons = [], [], []
    for arm, seed, task, step in R.EARLY+R.SILU:
        tag = f'{arm}_seed{seed}_task{task}_u{step}'
        meta = json.loads((OUT/'raw'/f'{tag}.json').read_text())
        early = meta['early']
        with np.load(OUT/'raw'/f'{tag}.npz') as arrays:
            for layer in ([0, 1] if early else [2, 3]):
                mask = arrays[f'L{layer}_mask'].astype(bool)
                for metric in R.METRICS:
                    prefix = f'L{layer}_{metric}_'
                    p = {k[len(prefix):]: arrays[k] for k in arrays.files if k.startswith(prefix)}
                    truth = p['actual' if early else 'actual_full']
                    truthz = zeros(truth)
                    common = mask & (truthz != 0)
                    base = dict(arm=arm, seed=seed, task=task, step=step, layer=layer, metric=metric,
                                route='own_up' if early else 'all_parameters')
                    row = dict(**base, **meta['stats'], n_responsive=int(mask.sum()), n_comparison=int(common.sum()),
                               actual_mean=float(truth.mean()), actual_sink=float((truth[mask] > 0).mean()),
                               r_median=median(arrays[f'L{layer}_r'][mask]),
                               ce_relative=ratio(p['ce'][mask], p['sq_full'][mask]),
                               task_departure_relative=ratio(p['task_departure'][mask], p['ctx_mean'][mask]),
                               fit_relative=ratio(p['fit_gap'][mask], p['ls_task'][mask]),
                               null_relative=ratio(p['null'][mask], truth[mask]),
                               fit_ce_cancellation=ratio((p['fit_gap']+p['ce'])[mask], p['fit_gap'][mask])+0.)
                    if early:
                        for name in ['Zhat', 'Zhat_first', 'Zhat_last']:
                            zz = p[name][mask]
                            row[name] = median(zz)
                            valid = np.isfinite(zz)
                            row[name+'_valid'] = int(valid.sum())
                            row[name+'_ge3'] = float((zz[valid] >= 3).mean()) if valid.any() else None
                    pred = {name: p[name] for name in ['P2', 'P2ctx', 'ls_task', 'sq_range', 'sq_full', 'actual']}
                    for part in R.PARTS+([] if early else ['direct']):
                        pred['without_'+part] = truth-p[part]
                    for name, value in pred.items():
                        cc = R.E.compare(zeros(value), truthz, common)
                        comparisons.append(dict(**base, predictor=name, target='actual', **cc))
                        row[name+'_ba'] = cc['ba']
                        row[name+'_rho'] = cc['rho']
                        row[name+'_agreement'] = cc['agreement']
                    for name in ['ls_task', 'sq_range', 'sq_full']:
                        yy = zeros(p[name])
                        cc = R.E.compare(zeros(p['P2']), yy, mask & (yy != 0))
                        comparisons.append(dict(**base, predictor='P2', target=name, **cc))
                        row['P2_vs_'+name+'_ba'] = cc['ba']
                    states.append(row)
                    for name in ['iid_mean', 'ctx_mean', 'ls_task', 'sq_range', 'sq_full', 'actual']+R.PARTS[1:]+([] if early else ['direct', 'actual_full']):
                        value = p[name][mask]
                        components.append(dict(**base, component=name, l2=float(np.linalg.norm(value)),
                                               relative_actual=ratio(value, truth[mask]), median=median(value),
                                               mean=float(np.mean(value)), positive_fraction=float((value > 0).mean())))
    csvwrite(OUT/'states.csv', states)
    csvwrite(OUT/'components.csv', components)
    csvwrite(OUT/'comparisons.csv', comparisons)
    windows = []
    keys = ['fit_error', 'Zhat', 'Zhat_first', 'Zhat_last', 'Zhat_ge3', 'ce_relative', 'task_departure_relative',
            'pmax', 'logit_rms', 'head_ce_ratio', 'fit_relative', 'null_relative', 'P2_ba', 'P2ctx_ba', 'ls_task_ba',
            'sq_range_ba', 'sq_full_ba', 'P2_rho', 'P2_vs_ls_task_ba', 'P2_vs_sq_range_ba', 'P2_vs_sq_full_ba']
    for arm in ['C3', 'ReLUcap']:
        for seed in range(10):
            for layer in [0, 1]:
                for metric in R.METRICS:
                    selected = [r for r in states if (r['arm'], r['seed'], r['layer'], r['metric']) == (arm, seed, layer, metric)]
                    row = dict(arm=arm, seed=seed, layer=layer, metric=metric)
                    for window, tasks in [('early', range(1, 5)), ('late', range(6, 10))]:
                        rr = [r for r in selected if r['task'] in tasks]
                        for key in keys:
                            row[key+'_'+window] = median([r[key] for r in rr])
                    for key in ['fit_error', 'Zhat', 'Zhat_first', 'Zhat_last', 'ce_relative', 'task_departure_relative']:
                        a, b = row[key+'_early'], row[key+'_late']
                        row[key+'_ratio'] = None if a is None or a == 0 or b is None else b/a
                    windows.append(row)
    csvwrite(OUT/'windows.csv', windows)
    mainrows = [r for r in windows if r['arm'] == 'C3' and r['layer'] == 0 and r['metric'] == 'SGD']
    judge = dict(scope='C3 L0 own-up SGD; observational diagnostic, not causal identification', seed_count=10)
    thresholds = [('J1', 'FIT_CLOSER', 'fit_error', .8, False, .35),
                  ('J2', 'GEOMETRY_MORE_STABLE', 'Zhat', 1.25, True, .65),
                  ('J3', 'CE_RELATIVELY_SMALLER', 'ce_relative', .8, False, .35),
                  ('J4', 'TASK_DEPARTURE_SMALLER', 'task_departure_relative', .8, False, .50)]
    for key, label, field, threshold, greater, prob in thresholds:
        passed = [r['seed'] for r in mainrows if r[field+'_ratio'] is not None and
                  (r[field+'_ratio'] >= threshold if greater else r[field+'_ratio'] <= threshold)]
        judge[key] = dict(label=label, passed=len(passed) >= 8, seeds=passed, count=len(passed), threshold=threshold,
                          prediction_probability=prob, brier=(prob-(len(passed) >= 8))**2)
    judge['J2']['split_half_counts'] = {field: sum(r[field+'_ratio'] >= 1.25 for r in mainrows) for field in ['Zhat_first', 'Zhat_last']}
    dump(OUT/'judge.json', judge)
    text = ['# 相2の方向対応が立ち上がる条件：保存状態の実測', '',
            'C3/ReLUcap各10 seed×課題1〜9の180状態、SiLU/SiLUcap seed1000の課題50/100末4状態。学習更新0。', '',
            '## 事前登録した判定', '', '|判定|達成seed|結果|予測確率|Brier|', '|---|---:|---|---:|---:|']
    for key, *_ in thresholds:
        j = judge[key]
        text.append(f"|{key} {j['label']}|{j['count']}/10|{'成立' if j['passed'] else '不成立'}|{j['prediction_probability']:.2f}|{j['brier']:.4f}|")
    text += ['', '早期=課題1〜4、成立後=6〜9。状態内のunit集約→seed内の窓中央値→10seed範囲。J2 split-half: '+str(judge['J2']['split_half_counts']), '',
             '## 主：C3 L0自身のup行、SGD', '', '|量|課題1〜4|課題6〜9|後/前のseed範囲|', '|---|---:|---:|---:|']
    for key in keys:
        text.append('|'+key+'|'+span([r[key+'_early'] for r in mainrows])+'|'+span([r[key+'_late'] for r in mainrows])+'|'+span([r.get(key+'_ratio') for r in mainrows])+'|')
    text += ['', '|seed|fit距離 前→後|Zhat 前→後|CE比 前→後|個別課題偏差比 前→後|相2 BA 前→後|', '|---:|---|---|---|---|---|']
    for r in mainrows:
        text.append('|'+str(r['seed'])+'|'+ '|'.join(fmt(r[k+'_early'])+'→'+fmt(r[k+'_late']) for k in ['fit_error', 'Zhat', 'ce_relative', 'task_departure_relative', 'P2_ba'])+'|')
    text += ['', '## 同じ実測への模型ごとの方向一致', '', '|腕・層・metric|窓|P2 BA|文脈P2 BA|個別LS BA|実head二乗 BA|P2対個別LS BA|', '|---|---|---:|---:|---:|---:|---:|']
    for arm in ['C3', 'ReLUcap']:
        for layer in [0, 1]:
            for metric in R.METRICS:
                rr = [r for r in windows if (r['arm'], r['layer'], r['metric']) == (arm, layer, metric)]
                for window in ['early', 'late']:
                    text.append(f'|{arm} L{layer} {metric}|{window}|'+ '|'.join(span([r[k+'_'+window] for r in rr]) for k in ['P2_ba', 'P2ctx_ba', 'ls_task_ba', 'sq_full_ba', 'P2_vs_ls_task_ba'])+'|')
    text += ['', '## 帳簿の各項の大きさと差引き（主、C3 L0 SGD）', '',
             '比=各項のunit L2 norm / 実CEのunit L2 norm。項の相殺があるため寄与率ではない。差引き後BAは「元の実測」を正解としている。', '',
             '|項|窓|L2比 seed範囲|正のunit割合 seed範囲|差引き後BA seed範囲|', '|---|---|---:|---:|---:|']
    for part in R.PARTS:
        for window, tasks in [('early', range(1, 5)), ('late', range(6, 10))]:
            norms, signs, bas = [], [], []
            for seed in range(10):
                rr = [r for r in components if (r['arm'], r['seed'], r['layer'], r['metric'], r['component']) == ('C3', seed, 0, 'SGD', part) and r['task'] in tasks]
                norms.append(median([r['relative_actual'] for r in rr]))
                signs.append(median([r['positive_fraction'] for r in rr]))
                ss = [r for r in states if (r['arm'], r['seed'], r['layer'], r['metric']) == ('C3', seed, 0, 'SGD') and r['task'] in tasks]
                bas.append(median([r['without_'+part+'_ba'] for r in ss]))
            text.append(f'|{part}|{window}|{span(norms)}|{span(signs)}|{span(bas)}|')
    text += ['', '## SiLU、全パラメータ経路、1 seedの記録', '',
             '|腕|課題|層|metric|r中央値|P2 BA|個別LS BA|実head二乗 BA|fit距離|CE比|null/実測|', '|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in states:
        if r['arm'] not in ['SiLU', 'SiLUcap']:
            continue
        text.append(f"|{r['arm']}|{r['task']}|L{r['layer']}|{r['metric']}|"+'|'.join(fmt(r[k]) for k in ['r_median', 'P2_ba', 'ls_task_ba', 'sq_full_ba', 'fit_error', 'ce_relative', 'null_relative'])+'|')
    gate = json.loads((OUT/'gate.json').read_text())
    checkvalues = [v for m in json.loads((OUT/'checks.json').read_text()) for v in m['checks'].values()]
    refvalues = [v for m in gate['states'] for v in m['reference']['checks'].values()]
    text += ['', '## 検算と限界', '', f'前回5状態の配列再現：最大相対誤差{max(refvalues):.3g}。184状態の恒等式・勾配・r射影：最大相対誤差{max(checkvalues):.3g}。', '',
             'Zhatは固定64 Gaussian方向からの推定。固定12unitのAをJVPから直接作り、trace・全Gaussian二次形式・fit分解を独立検算した。正確なZと上界はgate.json。Gaussian模型の確率上界を実Markov課題の保証に読み替えない。', '',
             'Y=65(Q−U)は固定した一様点二次模型。fit距離はCE最適headへの距離ではない。βはDirichlet生成則から固定し、実測の係数fitは無し。文脈平均は学習済み特徴とprobeを固定した模型で、ラベルとの独立性を実環境で実証したものではない。', '',
             'C3の比較はL0自身のup行の経路。SiLUは全パラメータ経路で直接共有headの項を含む。両者を同じ経路の比較と呼ばない。切替は新ラベルの独立一様平均で、次の課題入力を再生したものではない。実際の長期更新・沈降の原因割合ではない。', '',
             '## 再現', '', 'spec登録3324927d。コード・結果：analysis/chargpt_phase2conditions_1024、results/chargpt_phase2conditions_1024。入力・実行commitはprovenance.json、raw/log退避先はbackup_manifest.json。']
    (OUT/'summary.md').write_text('\n'.join(text)+'\n')
    print(json.dumps(judge, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
