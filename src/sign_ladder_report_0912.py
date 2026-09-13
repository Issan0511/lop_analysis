"""Verdict for spec_sign_ladder_0912.md.  Gates first, then the registered label."""
from pathlib import Path
import csv, json, re, sys

import numpy as np

from src import width_sink_clamp_0909 as C
from src import transport_common_0910 as T

ROOT = C.ROOT
OUT = ROOT / 'results/sign_ladder_0912'
ARMS = ['GELU', 'GELUF', 'GELUA']
SEEDS = [0, 1, 2]
BASE, LATE = (16, 20), (301, 400)
G0_MIN, G3_MIN_UNITS, G4_MIN_GAIN = 2.0, 20, 0.10
CAUSAL, IRREL = 1.5, 0.7


def rows_of(arm, s):
    with open(OUT / f'{arm}_none_s{s}_rows.csv') as f:
        return [{k: (float(v) if re.match(r'^-?[\d.eE+]+$', v or 'x') else v)
                 for k, v in r.items()} for r in csv.DictReader(f)]


def L_of(arm, s):
    r = rows_of(arm, s)
    b = np.median([x['acc'] for x in r if BASE[0] <= x['task'] <= BASE[1]])
    l = np.median([x['acc'] for x in r if LATE[0] <= x['task'] <= LATE[1]])
    return 100 * (b - l)


def units_of(arm, s):
    z = np.load(OUT / f'{arm}_none_s{s}_units.npz')
    ts = sorted(int(m.group(1)) for k in z.files if (m := re.match(r'ref_zbar_i_t(\d+)$', k)))
    return (np.array([z[f'ref_zbar_i_t{t}'] for t in ts]),
            np.array([z[f'ref_sd_i_t{t}'] for t in ts]))


def mobility(arm, lo=-9., hi=-5., k=0.7, H=50, t0=21, t1=400):
    """scale-free P(zbar_i rises by k*sd within H tasks), pooled over seeds"""
    num = den = 0
    for s in SEEDS:
        Z, S = units_of(arm, s)
        for t in range(t0 - 1, min(t1, len(Z)) - H):
            m = (Z[t] >= lo) & (Z[t] < hi)
            if not m.any():
                continue
            num += (Z[t + 1:t + 1 + H][:, m] > Z[t][m] + k * S[t][m]).any(axis=0).sum()
            den += int(m.sum())
    return (num / den if den else float('nan')), den


def main():
    prov = {(a, s): json.load(open(OUT / f'{a}_none_s{s}_provenance.json')) for a in ARMS for s in SEEDS}
    L = {(a, s): L_of(a, s) for a in ARMS for s in SEEDS}
    last = {(a, s): rows_of(a, s)[-1] for a in ARMS for s in SEEDS}

    print('=== 進捗ゲート ===')
    g0 = [L[('GELU', s)] >= G0_MIN for s in SEEDS]
    print(f'  G0 参照の損失 >= {G0_MIN} pt : ' +
          ' '.join(f'{L[("GELU",s)]:.2f}' for s in SEEDS) + f'  -> {"PASS" if all(g0) else "FAIL"}')
    ck0 = prov[('GELU', 0)]['checks']
    g1 = ck0.get('g1_units_maxabs'), ck0.get('g1_units_compared'), ck0.get('g1_units_expected')
    print(f'  G1 committed 参照と bit 一致 : maxabs={g1[0]} 比較{g1[1]}/{g1[2]} 件 '
          f'-> {"PASS" if g1[0] == 0. and g1[1] == g1[2] else "FAIL"}   ({ck0.get("g1_anchor")})')
    g2 = all(prov[(a, s)]['checks']['gate_neg_frac_max'] == 0. for a in ARMS[1:] for s in SEEDS)
    print('  G2 新腕のゲートが一度も負にならない : ' +
          ' '.join(f'{a}={max(prov[(a,s)]["checks"]["gate_neg_frac_max"] for s in SEEDS):g}' for a in ARMS[1:])
          + f'  -> {"PASS" if g2 else "FAIL"}')
    pz = {a: [int(last[(a, s)]['past_zc_units']) for s in SEEDS] for a in ARMS}
    g3 = all(min(pz[a]) >= G3_MIN_UNITS for a in ARMS)
    print('  G3 谷を越えたユニット >= %d : ' % G3_MIN_UNITS +
          '  '.join(f'{a} {pz[a]}' for a in ARMS) + f'  -> {"PASS" if g3 else "FAIL"}')
    mob = {a: mobility(a) for a in ARMS}
    g4 = mob['GELUA'][0] >= mob['GELU'][0] + G4_MIN_GAIN
    print('  G4 GELUA の可動性が GELU+%.2f 以上 : ' % G4_MIN_GAIN +
          '  '.join(f'{a} {mob[a][0]:.3f}(n{mob[a][1]//1000}k)' for a in ARMS)
          + f'  -> {"PASS" if g4 else "FAIL"}')

    print('\n=== 主判定 ===')
    print(f"{'arm':7s}" + ''.join(f'{f"L(s{s})":>10s}' for s in SEEDS) + f"{'平均':>9s}{'ΔL 対 GELU':>28s}")
    d = {}
    for a in ARMS:
        v = [L[(a, s)] for s in SEEDS]
        dd = [L[('GELU', s)] - L[(a, s)] for s in SEEDS]
        d[a] = dd
        print(f'{a:7s}' + ''.join(f'{x:10.2f}' for x in v) + f'{np.mean(v):9.2f}'
              + ('' if a == 'GELU' else '   ' + ' '.join(f'{x:+.2f}' for x in dd)
                 + f'  (平均 {np.mean(dd):+.2f})'))

    def cls(a):
        if all(x >= CAUSAL for x in d[a]):
            return 'reduces'
        if all(abs(x) < IRREL for x in d[a]):
            return 'flat'
        return 'partial'
    cf, ca = cls('GELUF'), cls('GELUA')
    label = ('VALLEY_CAUSAL' if cf == ca == 'reduces' else
             'VALLEY_IRRELEVANT' if cf == ca == 'flat' else
             'SIGN_ONLY' if ca == 'reduces' and cf != 'reduces' else
             'FLOOR_ONLY' if cf == 'reduces' and ca != 'reduces' else 'PARTIAL')
    gates_ok = all(g0) and g1[0] == 0. and g1[1] == g1[2] and g2 and g3
    print(f'\n  GELUF={cf}  GELUA={ca}  ->  **{label}**'
          + ('' if gates_ok else '   (※ ゲート不通過あり: NOT_TESTABLE 扱い)'))
    print(f'  記名: Claude VALLEY_IRRELEVANT / Issa VALLEY_CAUSAL  -> '
          f'{"Claude 的中" if label=="VALLEY_IRRELEVANT" else "Issa 的中" if label=="VALLEY_CAUSAL" else "両者外れ"}')

    print('\n=== 副次（ラベルにしない）===')
    print(f"{'arm':7s}{'z̄(t400)':>11s}{'sd(t400)':>10s}{'cnorm':>9s}{'past_zc':>9s}{'acc late':>10s}")
    for a in ARMS:
        zb, sd = zip(*[(units_of(a, s)[0][-1].mean(), units_of(a, s)[1][-1].mean()) for s in SEEDS])
        cn = np.mean([last[(a, s)].get('cnorm', float('nan')) for s in SEEDS])
        al = np.mean([np.median([r['acc'] for r in rows_of(a, s) if LATE[0] <= r['task'] <= LATE[1]])
                      for s in SEEDS])
        print(f'{a:7s}{np.mean(zb):11.2f}{np.mean(sd):10.2f}{cn:9.2f}'
              f'{np.mean(pz[a]):9.1f}{100*al:10.2f}')
    print('\n  参考（0910 の committed L_ref 平均）: ReLU 4.78 / leaky 4.36 / ELU1 4.13 / SiLU 6.93')


if __name__ == '__main__':
    main()
