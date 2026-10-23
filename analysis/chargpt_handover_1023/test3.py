#!/usr/bin/env python3
"""chargpt_handover_1023 検査 3: 帰還を塞いだ腕（lnex1）の L0 を対照と並べる（spec §3.4）。新しい学習はしない。

入力は chargpt_seed10_1022 の C の再生（byakko）の出力 results/chargpt_seed10_1022/revival/{C1,C2,C3}/seed*/<腕>/。
  C1: C3-2 の control・lnex1・nocommon（課題 S〜E、seed 0・1・2・4〜9）
  C3: ReLUcap-2 の control・lnex1（記録）
  C2: C3-2 の control・gamma03f・gamma03（課題 25 末から、記録）
量: layer_stats.csv の u = n_per_task（10000）の行。主は probe = uniform（C1 判定と同じ）の L0 の dead005（= frac_p_lt005）と r_med。
出力: results/chargpt_handover_1023/test3_rows.csv・test3.json（K7 を含む）。
"""
import sys
sys.dont_write_bytecode = True
import csv, hashlib, importlib.util, json, math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REV = ROOT / 'results/chargpt_seed10_1022/revival'
OUT = ROOT / 'results/chargpt_handover_1023'
_sp = importlib.util.spec_from_file_location('plan', ROOT / 'analysis/chargpt_seed10_1022/revival10_plan.py')
PLAN = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(PLAN)
PARTS = {'C1': ('C3', ['control', 'lnex1', 'nocommon']), 'C3': ('ReLUcap', ['control', 'lnex1']), 'C2': ('C3', ['control', 'gamma03f', 'gamma03'])}
QTY = ['frac_p_lt005', 'dead', 'r_med', 'mu_med']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sign_test_p(npos, nneg):
    """両側 exact 符号検定（二項 1/2）。0 の差は除いてから呼ぶ（judge.py と同じ式）。"""
    n = npos + nneg
    if n == 0: return float('nan')
    k = min(npos, nneg)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def windows(part):
    if part == 'C1': return {k: (S, E) for k, (R, S, E) in PLAN.C1_TABLE.items()}
    if part == 'C3': return {k: (S, E) for k, (R, S, E) in PLAN.C3_TABLE.items()}
    return {k: (PLAN.C2_FROM, PLAN.C2_TO) for k in range(10)}


def read_unit(part, seed, arm):
    d = REV / part / f'seed{seed}' / arm
    pv = json.loads((d / 'provenance.json').read_text())
    n = int(pv['n_per_task']); vals = {}
    for r in csv.DictReader(open(d / 'layer_stats.csv')):
        if int(float(r['u'])) != n: continue
        vals[(int(r['task']), r['probe'], int(r['layer']))] = {q: float(r[q]) for q in QTY}
    return pv, vals, {str((d / f).relative_to(ROOT)): sha(d / f) for f in ('layer_stats.csv', 'provenance.json')}


def main():
    judge = json.loads((ROOT / 'results/chargpt_seed10_1022/judge.json').read_text())
    rows, k7, summary, inputs = [], [], {}, {}
    for part, (model_arm, arms) in PARTS.items():
        win = windows(part); per = {}
        for seed, (S, E) in sorted(win.items()):
            u = {}
            for arm in arms:
                pv, vals, h = read_unit(part, seed, arm); inputs.update(h)
                ok = bool(pv.get('complete')) and pv['from_task'] == S and pv['to_task'] == E and pv.get('model_arm', 'C3') == model_arm and pv['arm'] == arm
                k7.append(dict(check='K7_complete_and_range', part=part, seed=seed, arm=arm, pass_=ok, from_task=pv['from_task'], to_task=pv['to_task'], S=S, E=E))
                u[arm] = vals
                for (t, probe, layer), q in sorted(vals.items()):
                    rows.append(dict(part=part, model_arm=model_arm, seed=seed, arm=arm, task=t, probe=probe, layer=layer, **q))
            # K7: 開始の課題末（介入の直後、学習前）では、L0・L1 の前活性は介入で変わらないので全腕が control と同じ値
            for arm in arms[1:]:
                same = all(u[arm][(S, p, l)][q] == u['control'][(S, p, l)][q] for p in ('uniform', 'chain') for l in (0, 1) for q in QTY)
                k7.append(dict(check='K7_start_equal', part=part, seed=seed, arm=arm, pass_=same, task=S))
            ctl = {t: u['control'][(t, 'uniform', 1)]['frac_p_lt005'] for t in range(S + 1, E + 1)}
            Rp = next((t for t in range(S + 1, E + 1) if ctl[t] <= 0.5), None) if part != 'C2' else None
            if part in ('C1', 'C3'):
                jp = judge['C'][part]['per_seed'][f'seed{seed}']['lnex1']
                k7.append(dict(check='K7_Rprime_matches_judge', part=part, seed=seed, pass_=(Rp == jp.get('Rp')), Rp=Rp, judge_Rp=jp.get('Rp'), judge_eligible=jp.get('eligible')))
            res = dict(S=S, E=E, Rp=Rp)
            for arm in arms[1:]:
                d = {}
                for probe in ('uniform', 'chain'):
                    for layer, q in [(0, 'frac_p_lt005'), (0, 'r_med'), (1, 'frac_p_lt005'), (0, 'dead')]:
                        diff = lambda ts: float(np.mean([u[arm][(t, probe, layer)][q] - u['control'][(t, probe, layer)][q] for t in ts]))
                        key = f'{probe}_L{layer}_{q}'
                        full = list(range(S + 1, E + 1))
                        d[f'{key}_win'] = diff(full)
                        d[f'{key}_end'] = diff([E])
                        d[f'{key}_ctl_win'] = float(np.mean([u['control'][(t, probe, layer)][q] for t in full]))
                        d[f'{key}_arm_win'] = float(np.mean([u[arm][(t, probe, layer)][q] for t in full]))
                        if Rp is not None:
                            jw = list(range(Rp, E + 1))
                            d[f'{key}_jw'] = diff(jw)
                            d[f'{key}_ctl_jw'] = float(np.mean([u['control'][(t, probe, layer)][q] for t in jw]))
                            d[f'{key}_arm_jw'] = float(np.mean([u[arm][(t, probe, layer)][q] for t in jw]))
                res[arm] = d
            per[seed] = res
        agg = {}
        for arm in arms[1:]:
            for key in [f'{p}_L{l}_{q}_{w}' for p in ('uniform', 'chain') for l, q in [(0, 'frac_p_lt005'), (0, 'r_med'), (1, 'frac_p_lt005'), (0, 'dead')] for w in ('win', 'jw', 'end')]:
                v = [per[s][arm][key] for s in per if key in per[s][arm]]
                if not v: continue
                npos, nneg = sum(x > 0 for x in v), sum(x < 0 for x in v)
                agg[f'{arm}:{key}'] = dict(n=len(v), median=float(np.median(v)), min=float(min(v)), max=float(max(v)), n_pos=npos, n_neg=nneg, n_zero=len(v) - npos - nneg, p_sign=sign_test_p(npos, nneg))
        summary[part] = dict(model_arm=model_arm, arms=arms, per_seed={f'seed{s}': v for s, v in per.items()}, aggregate=agg)
    fields = ['part', 'model_arm', 'seed', 'arm', 'task', 'probe', 'layer'] + QTY
    with open(OUT / 'test3_rows.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator='\n'); w.writeheader(); w.writerows(rows)
    res = dict(K7=dict(all_pass=all(r['pass_'] for r in k7), n=len(k7), n_pass=sum(r['pass_'] for r in k7), checks=k7),
               summary=summary, inputs=inputs, script_sha256=sha(Path(__file__)), rows_sha256=sha(OUT / 'test3_rows.csv'),
               note='差は腕 − control。dead005 = frac_p_lt005（p < 0.005）。win = 課題 S+1〜E、jw = [R′, E]（R′ のある seed だけ）、end = 課題 E。')
    (OUT / 'test3.json').write_text(json.dumps(res, indent=1, ensure_ascii=False) + '\n')
    print(f"K7 {'PASS' if res['K7']['all_pass'] else 'FAIL'} {res['K7']['n_pass']}/{res['K7']['n']}")
    for part in summary:
        for k, v in summary[part]['aggregate'].items():
            if k.split(':')[1].startswith('uniform_L0_frac') or k.split(':')[1].startswith('uniform_L0_r_med') or k.split(':')[1].startswith('uniform_L1_frac'):
                print(part, k, {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()})


if __name__ == '__main__':
    main()
