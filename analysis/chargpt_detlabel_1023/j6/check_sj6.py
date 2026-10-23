#!/usr/bin/env python3
"""S-J6（spec_chargpt_detlabel_1023 §8）の照合: 写した J6 の解析を A2（C3-2、α 0.01）seed 0 の課題末 1・9・50・100 に当てた出力を、
loadsign の states.csv・components_states.csv（と loading_sources_states.csv）と比べ、恒等式と autograd の差を集める。

先に次を回しておく（J6_ARM=A2 J6_OUT=<CHK>/A2）: measure.py gate → measure.py run --seeds 0 --tasks 1 9 50 100 →
components.py gate → run（同じ部分集合）→ loading_sources.py gate → run。
判定: 符号の割合（s_*）と整数（n_*）は完全一致、連続量は相対 1e−10 以内（|a−b| / max(|b|, 1e−300)。両方 NaN は一致）。
  列名が _error で終わる列（検算の残り。丸めの大きさの数）は値の再現の対象にせず、両方が 1e−10 以下であることを見る（compute_J の読み）。
恒等式: autograd ≤ 1e−10（最大絶対誤差 / 最大信号）、direct = self + other + residual ≤ 1e−10、exact = direct + radial ≤ 1e−10、
softmax 恒等式 ≤ 1e−10（各状態）。
"""
import csv
import json
import math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
CHK = ROOT/'results/chargpt_detlabel_1023/checks/J6'
OUT = CHK/'A2'
REF = ROOT/'results/chargpt_loadsign_1023'
TASKS = [1, 9, 50, 100]
SKIP = {'arm', 'seed', 'task', 'layer', 'previous_error', 'autograd_error'}


def relmax(d, *scales):
    return float(np.abs(d).max())/max(max(float(np.abs(s).max()) for s in scales), 1e-300)


def compare(name, ours_path, ref_path):
    ours = {int(r['task']): r for r in csv.DictReader(ours_path.open())}
    ref = {int(r['task']): r for r in csv.DictReader(ref_path.open()) if r['arm'] == 'C3' and int(r['seed']) == 0 and int(r['task']) in TASKS}
    assert sorted(ours) == TASKS and sorted(ref) == TASKS, (name, sorted(ours), sorted(ref))
    cols = [c for c in ref[1] if c not in SKIP]
    missing = [c for c in cols if c not in ours[1]]
    extra = [c for c in ours[1] if c not in ref[1] and c not in SKIP]
    res = dict(file=name, n_states=len(TASKS), n_columns=len(cols), missing_columns=missing, extra_columns=extra,
               n_sign_values=0, n_sign_mismatch=0, n_int_values=0, n_int_mismatch=0, n_cont_values=0, n_cont_fail=0,
               max_cont_rel=0.0, max_cont_rel_where=None, n_error_cols=0, n_error_col_fail=0, max_error_col_value=0.0, n_bit_identical=0, n_nan_both=0, failures=[])
    for t in TASKS:
        for c in cols:
            if c in missing: continue
            a, b = ours[t][c], ref[t][c]
            if c.startswith('s_') or c.startswith('n_') or c.endswith('_missing'):
                kind = 'sign' if c.startswith('s_') else 'int'
                res[f'n_{kind}_values'] += 1
                fa, fb = float(a), float(b)
                same = (fa == fb) or (math.isnan(fa) and math.isnan(fb))
                if not same:
                    res[f'n_{kind}_mismatch'] += 1; res['failures'].append(dict(task=t, col=c, ours=a, ref=b))
                continue
            fa, fb = float(a), float(b)
            if c.endswith('_error'):
                # 検算の残り（丸めの大きさ）。値の再現の対象ではなく、両方が許容 1e−10 以下であることを見る。
                res['n_error_cols'] += 1; res['max_error_col_value'] = max(res['max_error_col_value'], fa, fb)
                if not (fa <= 1e-10 and fb <= 1e-10):
                    res['n_error_col_fail'] += 1; res['failures'].append(dict(task=t, col=c, ours=a, ref=b))
                continue
            res['n_cont_values'] += 1
            if math.isnan(fa) and math.isnan(fb): res['n_nan_both'] += 1; continue
            if a == b: res['n_bit_identical'] += 1
            r = abs(fa-fb)/max(abs(fb), 1e-300)
            if r > res['max_cont_rel']: res['max_cont_rel'], res['max_cont_rel_where'] = r, dict(task=t, col=c, ours=a, ref=b)
            if not r <= 1e-10:
                res['n_cont_fail'] += 1; res['failures'].append(dict(task=t, col=c, ours=a, ref=b, rel=r))
    res['passed'] = not missing and res['n_sign_mismatch'] == 0 and res['n_int_mismatch'] == 0 and res['n_cont_fail'] == 0 and res['n_error_col_fail'] == 0
    return res


def identities():
    gate = json.loads((OUT/'gate.json').read_text())
    st = {int(r['task']): r for r in csv.DictReader((OUT/'states.csv').open())}
    co = {int(r['task']): r for r in csv.DictReader((OUT/'components_states.csv').open())}
    raw = np.load(OUT/'raw/A2_seed0.npz'); rc = np.load(OUT/'raw_components/A2_seed0.npz')
    rows = []
    for g in gate['rows']:
        t = int(g['task'])
        ex, di, ra = raw[f't{t}_original_exact_K'], raw[f't{t}_original_direct_K'], raw[f't{t}_radial']
        e_exact = relmax(ex-di-ra, ex, di, ra)
        e_comp = max(relmax(rc[f't{t}_{e}_direct']-rc[f't{t}_{e}_self']-rc[f't{t}_{e}_other']-rc[f't{t}_{e}_residual'],
                            rc[f't{t}_{e}_direct'], rc[f't{t}_{e}_self'], rc[f't{t}_{e}_other'], rc[f't{t}_{e}_residual'])
                     for e in ['original', 'loaderror'])
        rows.append(dict(task=t, autograd=float(g['autograd_error']), exact_eq_direct_plus_radial=e_exact,
                         direct_eq_self_other_residual=e_comp, softmax_identity=float(co[t]['softmax_identity_error']),
                         token_sum=float(co[t]['token_sum_error']), stage1_vs_stage2=float(co[t]['stage1_error']),
                         stage1_additivity_column=float(st[t]['additivity_error'])))
    worst = {k: max(r[k] for r in rows) for k in rows[0] if k != 'task'}
    return dict(rows=rows, worst=worst, passed=all(v <= 1e-10 for v in worst.values()))


def main():
    res = dict(identities=identities(),
               states=compare('states.csv', OUT/'states.csv', REF/'states.csv'),
               components=compare('components_states.csv', OUT/'components_states.csv', REF/'components_states.csv'),
               loading_sources=compare('loading_sources_states.csv', OUT/'loading_sources_states.csv', REF/'loading_sources_states.csv'))
    res['passed'] = all(v['passed'] for v in res.values())
    (CHK/'sj6_check.json').write_text(json.dumps(res, indent=2, ensure_ascii=False)+'\n')
    for k in ['states', 'components', 'loading_sources']:
        v = res[k]
        print(k, 'PASS' if v['passed'] else 'FAIL', {x: v[x] for x in ['n_columns', 'missing_columns', 'n_sign_values', 'n_sign_mismatch', 'n_int_values',
              'n_int_mismatch', 'n_cont_values', 'n_cont_fail', 'n_bit_identical', 'n_error_cols', 'n_error_col_fail', 'max_error_col_value', 'n_nan_both', 'max_cont_rel', 'max_cont_rel_where']})
    print('identities', 'PASS' if res['identities']['passed'] else 'FAIL', res['identities']['worst'])
    print('ALL', 'PASS' if res['passed'] else 'FAIL')


if __name__ == '__main__':
    main()
