"""Verdicts for spec_clamp_drift_0912 (J0-J4).  Refuses an incomplete run unless --partial."""
from pathlib import Path
import argparse, csv, json
import numpy as np
from src import gate_shape_report_0911 as R

ROOT = R.ROOT
OUT = ROOT / 'results/clamp_drift_0912'
ARMS = ['N0', 'N0w', 'N1', 'N1w']
SEEDS = [0, 1, 2]
BASE = (16, 20)
LATE = (101, 120)
FIRST = (1, 5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=None); ap.add_argument('--partial', action='store_true')
    a = ap.parse_args()
    global OUT
    if a.src:
        OUT = Path(a.src)
    have = {f.name.replace('_rows.csv', '') for f in OUT.glob('*_rows.csv')}
    want = {f'{x}_s{s}' for x in ARMS for s in SEEDS}
    gap = sorted(want - have)
    if gap and not a.partial:
        print(f'REFUSING: {len(gap)}/{len(want)} runs missing {gap[:6]}'); raise SystemExit(2)

    D = {}
    for x in ARMS:
        for s in SEEDS:
            try:
                D[(x, s)] = (R.read_rows(OUT / f'{x}_s{s}_rows.csv'),
                             json.load(open(OUT / f'{x}_s{s}_provenance.json')))
            except FileNotFoundError:
                pass
    arms = [x for x in ARMS if any(k[0] == x for k in D)]
    ok = lambda x: all((x, s) in D for s in SEEDS)

    def w(x, s, key, lo=LATE[0], hi=LATE[1]):
        v = [r[key] for r in D[(x, s)][0] if lo <= r['task'] <= hi and key in r and np.isfinite(r[key])]
        return float(np.median(v)) if v else np.nan

    def L(x, s):
        return (w(x, s, 'acc', *BASE) - w(x, s, 'acc')) * 100

    def med(x, k):
        return float(np.median([w(x, s, k) for s in SEEDS if (x, s) in D]))

    V, lines = {}, ['# clamp_drift_0912 summary\n',
                    'spec: `specs/spec_clamp_drift_0912.md`（事前登録 commit `8f99e94`・実装 `3000839`・判定値は未読で起動）\n',
                    f'scope: {len(arms)} 腕 × 3 seed・t1–120・base t{BASE[0]}–{BASE[1]}・late t{LATE[0]}–{LATE[1]}\n']

    # ---- J0 / checks
    lines.append('\n## 0. 検査\n\n| arm | c | clamp | G1 錨 | G1 maxabs | 配列数 | 射影の恒等式 | 注入 平均 | クランプ検査 |\n|---|---:|---|---|---|---:|---:|---:|---|')
    g1_ok = True
    for x in arms:
        ck = D[(x, 0)][1]['checks']
        g1s = [D[(x, s)][1]['checks']['g1_units_maxabs'] for s in SEEDS if (x, s) in D]
        g1_ok = g1_ok and all(v <= 1e-10 for v in g1s)
        cl = (f"c3 {ck['c3_cnorm_rel']:.1e} / tail {ck['clamp_tail_absdiff']}" if 'c3_cnorm_rel' in ck
              else f"射影の差 {ck.get('g2_clamp_zero')}")
        lines.append(f"| {x} | {ck['noise_c']:.1f} | {ck['clamp']} | {ck['g1_anchor']} | "
                     f"{' / '.join(f'{v:g}' for v in g1s)} | {ck['g1_units_compared']} | "
                     f"{max(D[(x, s)][1]['checks']['g2_ident'] for s in SEEDS if (x, s) in D):.1e} | "
                     f"{ck.get('g4_inj_mean', float('nan')):.4f} | {cl} |")
    V['J0'] = 'PASS' if g1_ok else 'FAIL'
    lines.append(f"\n- **J0（G1）**: `{V['J0']}`")
    if ok('N0') and ok('N0w'):
        dl = [L('N0', s) - L('N0w', s) for s in SEEDS]
        V['testability_L_gap'] = [round(x, 2) for x in dl]
        V['testability_met'] = bool(all(x >= 1.0 for x in dl))
        lines.append(f"- **可検定性（J0 とは独立に記録）**: L(N0) − L(N0w) = {V['testability_L_gap']} pt、`>= 1.0 を 3/3` は {V['testability_met']}")

    # ---- per-arm table
    lines.append('\n## 1. 腕ごとの量（seed 中央値）\n\n| arm | L [pt] | level | acc t1–5 | **drift2_net** | drift2_adam | diff2_net | f_diff_net | ρ | κ2 | graw² | ‖W̃ᵢ‖ | 射影が消した量 | うち勾配方向 |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for x in arms:
        lines.append(f"| {x} | {float(np.median([L(x, s) for s in SEEDS if (x, s) in D])):.2f} | {med(x, 'acc'):.4f} | "
                     f"{float(np.median([w(x, s, 'acc', *FIRST) for s in SEEDS if (x, s) in D])):.4f} | "
                     f"**{med(x, 'drift2_net'):.2f}** | {med(x, 'drift2_adam'):.2f} | {med(x, 'diff2_net'):.2f} | "
                     f"{med(x, 'f_diff_net'):.3f} | {med(x, 'rho_mean'):.1f} | {med(x, 'kap2'):.4f} | "
                     f"{med(x, 'graw2'):.4f} | {med(x, 'cnorm'):.2f} | {med(x, 'clamp_removed'):.3f} | {med(x, 'clamp_drift_share'):.3f} |")

    if V.get('J0') == 'PASS':
        # ---- J1
        if ok('N0') and ok('N0w'):
            rr = [w('N0w', s, 'drift2_net') / w('N0', s, 'drift2_net') for s in SEEDS]
            V['J1'] = ('CLAMP_RAISES_DRIFT' if all(x >= 1.3 for x in rr)
                       else 'CLAMP_LOWERS_DRIFT' if all(x <= 0.9 for x in rr) else 'CLAMP_DRIFT_FLAT')
            V['J1_ratio'] = [round(x, 3) for x in rr]
            lines.append(f"\n- **J1（決定的）**: drift2_net(N0w)/drift2_net(N0) = {V['J1_ratio']} → `{V['J1']}`")
        # ---- J2  the pair rho could not separate
        if ok('N0w') and ok('N1'):
            rr = [w('N0w', s, 'drift2_net') / w('N1', s, 'drift2_net') for s in SEEDS]
            V['J2'] = ('DRIFT_SEPARATES_THE_PAIR' if all(x >= 1.5 for x in rr)
                       else 'DRIFT_FAILS_THE_PAIR' if all(0.8 <= x <= 1.2 for x in rr) else 'J2_PARTIAL')
            V['J2_ratio'] = [round(x, 3) for x in rr]
            V['J2_rho'] = [round(med('N0w', 'rho_mean'), 1), round(med('N1', 'rho_mean'), 1)]
            V['J2_L'] = [round(float(np.median([L('N0w', s) for s in SEEDS])), 2),
                         round(float(np.median([L('N1', s) for s in SEEDS])), 2)]
            lines.append(f"- **J2（ρ が分けられなかった対）**: N0w 対 N1 は ρ {V['J2_rho']}・L {V['J2_L']} pt。"
                         f"drift2_net の比 = {V['J2_ratio']} → `{V['J2']}`")
        # ---- J3
        if ok('N0w'):
            sh = [w('N0w', s, 'clamp_drift_share') for s in SEEDS]
            V['J3'] = ('CLAMP_CUTS_SIGNAL' if all(x >= 0.3 for x in sh)
                       else 'CLAMP_CUTS_ONLY_RADIAL' if all(x <= 0.1 for x in sh) else 'J3_PARTIAL')
            V['J3_share'] = [round(x, 4) for x in sh]
            lines.append(f"- **J3（射影は何を消すか）**: 消した量のうち勾配方向の割合 = {V['J3_share']} → `{V['J3']}`")
        # ---- J4 descriptive ordering
        if len(arms) >= 4:
            Ls = [float(np.median([L(x, s) for s in SEEDS if (x, s) in D])) for x in arms]
            V['J4_spearman_drift_L'] = R.spearman([med(x, 'drift2_net') for x in arms], Ls)
            V['J4_spearman_N_L'] = R.spearman([med(x, 'cnorm') for x in arms], Ls)
            V['J4_spearman_rho_L'] = R.spearman([med(x, 'rho_mean') for x in arms], Ls)
            lines.append(f"- **J4（n=4・記述）**: Spearman vs L は drift2_net {V['J4_spearman_drift_L']:+.2f}、"
                         f"‖W̃ᵢ‖ {V['J4_spearman_N_L']:+.2f}、ρ {V['J4_spearman_rho_L']:+.2f}")
    (OUT / 'summary.md').write_text('\n'.join(lines) + '\n')
    with open(OUT / 'verdict.csv', 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(V.keys()), lineterminator='\n')
        wr.writeheader(); wr.writerow({k: json.dumps(v) if isinstance(v, list) else v for k, v in V.items()})
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        for x in arms:
            ax[0].scatter(med(x, 'drift2_net'), float(np.median([L(x, s) for s in SEEDS])), s=50)
            ax[0].annotate(x, (med(x, 'drift2_net'), float(np.median([L(x, s) for s in SEEDS]))), fontsize=9,
                           xytext=(5, 4), textcoords='offset points')
            ax[1].scatter(med(x, 'rho_mean'), float(np.median([L(x, s) for s in SEEDS])), s=50)
            ax[1].annotate(x, (med(x, 'rho_mean'), float(np.median([L(x, s) for s in SEEDS]))), fontsize=9,
                           xytext=(5, 4), textcoords='offset points')
        ax[0].set_xlabel('drift2_net (一歩のうち勾配方向・lr² 単位)'); ax[0].set_ylabel('目減り L [pt]')
        ax[1].set_xlabel('ρ (タスク内の経路持続)'); ax[1].set_ylabel('目減り L [pt]')
        fig.tight_layout(); fig.savefig(OUT / 'fig_clamp_drift.png', dpi=120)
    except Exception as e:
        print('figure skipped:', e)
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
