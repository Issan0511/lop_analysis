"""Verdicts for spec_grad_floor_0911 (H0-H5).  Refuses an incomplete run unless --partial."""
from pathlib import Path
import argparse, csv, json
import numpy as np
from src import gate_shape_report_0911 as R

ROOT = R.ROOT
OUT = ROOT / 'results/grad_floor_0911'
ARMS = ['N0', 'N2', 'LRq', 'Gm2', 'Gm2c', 'LRx']
SEEDS = [0, 1, 2]
BASE = (16, 20)
LATE = (101, 120)
FIRST = (1, 5)
G3_CE_FRAC = 0.90


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=None); ap.add_argument('--partial', action='store_true')
    a = ap.parse_args()
    global OUT
    if a.src:
        OUT = Path(a.src)
    have = {f.name.replace('_rows.csv', '') for f in OUT.glob('*_rows.csv')}
    want = {f'{arm}_s{s}' for arm in ARMS for s in SEEDS}
    gap = sorted(want - have)
    if gap and not a.partial:
        print(f'REFUSING: {len(gap)}/{len(want)} runs missing {gap[:6]}'); raise SystemExit(2)

    D = {}
    for arm in ARMS:
        for s in SEEDS:
            try:
                D[(arm, s)] = (R.read_rows(OUT / f'{arm}_s{s}_rows.csv'),
                               json.load(open(OUT / f'{arm}_s{s}_provenance.json')))
            except FileNotFoundError:
                pass
    arms = [x for x in ARMS if any(k[0] == x for k in D)]
    ok = lambda x: all((x, s) in D for s in SEEDS)

    def w(arm, s, key, lo=LATE[0], hi=LATE[1]):
        v = [r[key] for r in D[(arm, s)][0] if lo <= r['task'] <= hi and key in r and np.isfinite(r[key])]
        return float(np.median(v)) if v else np.nan

    def L(arm, s):
        return (w(arm, s, 'acc', *BASE) - w(arm, s, 'acc')) * 100

    def dmg(arm, s):
        return L(arm, s) - L('N0', s)

    def med(arm, k):
        return float(np.median([w(arm, s, k) for s in SEEDS if (arm, s) in D]))

    V, lines = {}, ['# grad_floor_0911 summary\n',
                    'spec: `specs/spec_grad_floor_0911.md`（事前登録 commit `273e9dc`・実装 `bb14cb2`・判定値は未読で起動）\n',
                    f'scope: {len(arms)} 腕 × 3 seed・t1–120・base t{BASE[0]}–{BASE[1]}・late t{LATE[0]}–{LATE[1]}\n']

    # ---------------- checks
    lines.append('\n## 0. 検査\n\n| arm | mode | c | lr | G1 錨 | G1 maxabs | 注入の実測 | g=0 座標 | 同エネルギー誤差 | CE 改善率 |\n|---|---|---:|---:|---|---|---|---:|---:|---:|')
    for arm in arms:
        ck = D[(arm, 0)][1]['checks']
        inj = f"{ck['g0_inj_lo']:.3f}–{ck['g0_inj_hi']:.3f}" if 'g0_inj_lo' in ck else '—'
        zt = ck.get('g0_zero_touched', '—')
        cef = min(float(np.mean([r['ce20'] - r['ce_probe'] > 0 for r in D[(arm, s)][0]
                                 if LATE[0] <= r['task'] <= LATE[1]])) for s in SEEDS if (arm, s) in D)
        lines.append(f"| {arm} | {ck['mode']} | {ck['noise_c']:.1f} | {ck['lr']:.6f} | {ck['g1_anchor']} | "
                     f"{' / '.join(str(D[(arm, s)][1]['checks'].get('g1_units_maxabs')) for s in SEEDS if (arm, s) in D)} | "
                     f"{inj} | {zt} | {ck.get('g2_energy_rel', '—')} | {cef:.2f} |")
    broken = [x for x in arms if min(float(np.mean([r['ce20'] - r['ce_probe'] > 0 for r in D[(x, s)][0]
                                                    if LATE[0] <= r['task'] <= LATE[1]])) for s in SEEDS if (x, s) in D) < G3_CE_FRAC]
    if broken:
        lines.append(f"\n**LEARNING_BROKEN**: {broken}（判定から外す）")
    ok = lambda x: all((x, s) in D for s in SEEDS) and x not in broken

    # ---------------- per-arm table
    lines.append('\n## 1. 腕ごとの量（seed 中央値）\n\n| arm | L [pt] | 損傷 D | level(late) | acc t1–5 | drift2 | diff2 | f_diff | ρ | κ2 | N | ‖W2‖ | ce625 |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for arm in arms:
        dm = float(np.median([dmg(arm, s) for s in SEEDS if (arm, s) in D]))
        lines.append(f"| {arm} | {float(np.median([L(arm, s) for s in SEEDS if (arm, s) in D])):.2f} | {dm:+.2f} | "
                     f"{med(arm, 'acc'):.4f} | {float(np.median([w(arm, s, 'acc', *FIRST) for s in SEEDS if (arm, s) in D])):.4f} | "
                     f"{med(arm, 'drift2'):.2f} | {med(arm, 'diff2'):.2f} | {med(arm, 'f_diff'):.3f} | {med(arm, 'rho_mean'):.1f} | "
                     f"{med(arm, 'kap2'):.4f} | {med(arm, 'cnorm'):.2f} | {med(arm, 'w2_fro'):.2f} | {med(arm, 'ce625'):.3f} |")
    lines.append('\n生勾配の形（走中に記録・late 窓・seed 中央値）: ' +
                 '、'.join(f"{arm} p50 {med(arm, 'r_p50'):.4f} / p90 {med(arm, 'r_p90'):.2f} / r<0.2 {med(arm, 'r_lt02'):.3f} / 上位10% {med(arm, 'top10_energy'):.3f}"
                           for arm in arms if ok(arm)))

    # ---------------- H0
    if ok('N2') and ok('Gm2') and ok('N0'):
        c_ok, z_ok, e_ok = True, True, True
        for arm in ('N2', 'Gm2', 'Gm2c'):
            if not ok(arm):
                continue
            for s in SEEDS:
                ck = D[(arm, s)][1]['checks']
                c_ok = c_ok and 0.9 * ck['noise_c'] <= ck['g0_inj_lo'] and ck['g0_inj_hi'] <= 1.1 * ck['noise_c']
                e_ok = e_ok and ck.get('g2_energy_rel', 1.) < 1e-6
                if ck['mode'] == 'coord':
                    z_ok = z_ok and ck.get('g0_zero_touched', 1.) == 0.
        shape_ok = all(med(arm, 'r_p50') < 0.05 and med(arm, 'r_lt02') > 0.5 for arm in arms if ok(arm))
        dN2 = [dmg('N2', s) for s in SEEDS]
        V['H0'] = 'MANIPULATION_OK' if (c_ok and z_ok and e_ok and shape_ok and all(x >= 0.5 for x in dN2)) else 'NOT_TESTABLE'
        V['H0_parts'] = f'inj={c_ok} zero={z_ok} energy={e_ok} shape={shape_ok} D_N2={[round(x, 2) for x in dN2]}'
        lines.append(f"\n- **H0**: 注入 {c_ok}・g=0 不動 {z_ok}・同エネルギー {e_ok}・勾配の形 {shape_ok}・D(N2) = {[round(x, 2) for x in dN2]} pt → `{V['H0']}`")

    if V.get('H0') == 'MANIPULATION_OK':
        # ---------------- H1 decisive
        dN2 = [dmg('N2', s) for s in SEEDS]; dG = [dmg('Gm2', s) for s in SEEDS]
        rr = [g / n for g, n in zip(dG, dN2)]
        V['H1'] = ('FLOOR_EXPLAINS_HARM' if all(x <= 0.3 for x in rr)
                   else 'FLOOR_NOT_THE_CAUSE' if all(x >= 0.7 for x in rr) else 'H1_PARTIAL')
        V['H1_D_Gm2'] = [round(x, 2) for x in dG]; V['H1_D_N2'] = [round(x, 2) for x in dN2]
        lines.append(f"- **H1（決定的）**: D(Gm2) = {V['H1_D_Gm2']}、D(N2) = {V['H1_D_N2']} pt、比 {[round(x, 2) for x in rr]} → `{V['H1']}`")
        # ---------------- H4
        dr_g, dr_n = med('Gm2', 'drift2'), med('N2', 'drift2')
        V['H4_drift_Gm2'] = dr_g; V['H4_drift_N2'] = dr_n; V['H4_drift_N0'] = med('N0', 'drift2')
        if dr_g <= 1.3 * dr_n and V['H1'] == 'FLOOR_EXPLAINS_HARM':
            V['H4'] = 'DRIFT_NOT_SUFFICIENT'
        elif dr_g >= 2 * dr_n:
            V['H4'] = 'DRIFT_TRACKS_HARM'
        else:
            V['H4'] = 'H4_PARTIAL'
        lines.append(f"- **H4**: drift2 = N0 {V['H4_drift_N0']:.2f} / N2 {dr_n:.2f} / Gm2 {dr_g:.2f}（Gm2/N2 = {dr_g / dr_n:.2f}） → `{V['H4']}`")
        # ---------------- H3
        if ok('LRq'):
            d3 = [L('Gm2', s) - L('LRq', s) for s in SEEDS]
            V['H3'] = 'MULT_EQUALS_LOWER_LR' if all(abs(x) < 0.3 for x in d3) else 'MULT_DIFFERS_FROM_LR'
            V['H3_d'] = [round(x, 2) for x in d3]
            lines.append(f"- **H3**: L(Gm2) − L(LRq) = {V['H3_d']} pt → `{V['H3']}`")
    # ---------------- H2 (independent of H0's D_N2 gate)
    if ok('Gm2c') and ok('LRx'):
        d2 = [L('Gm2c', s) - L('LRx', s) for s in SEEDS]
        V['H2'] = ('INCOHERENCE_HARMLESS' if all(abs(x) < 0.3 for x in d2)
                   else 'INCOHERENCE_HURTS' if all(x >= 0.5 for x in d2)
                   else 'INCOHERENCE_HELPS' if all(x <= -0.5 for x in d2) else 'H2_PARTIAL')
        V['H2_d'] = [round(x, 2) for x in d2]
        lines.append(f"- **H2**: L(Gm2c) − L(LRx) = {V['H2_d']} pt（lr = 1e−3·√5 で一律縮小を打ち消した対） → `{V['H2']}`")
    # ---------------- H5 absolute level
    lines.append('\n- **H5** 絶対水準 level(late) − level(N0) [pt]:')
    for arm in arms:
        if arm == 'N0' or not ok(arm):
            continue
        dd = [(w(arm, s, 'acc') - w('N0', s, 'acc')) * 100 for s in SEEDS]
        lab = 'BEATS_REF_ABSOLUTE' if all(x >= 0.2 for x in dd) else 'not'
        V[f'H5_{arm}'] = lab; V[f'H5_{arm}_d'] = [round(x, 2) for x in dd]
        lines.append(f"  - {arm}: {[round(x, 2) for x in dd]} → `{lab}`")
    (OUT / 'summary.md').write_text('\n'.join(lines) + '\n')
    with open(OUT / 'verdict.csv', 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(V.keys()), lineterminator='\n')
        wr.writeheader(); wr.writerow({k: json.dumps(v) if isinstance(v, list) else v for k, v in V.items()})
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
        for arm, c in (('N2', 'C3'), ('Gm2', 'C0'), ('LRq', 'C2'), ('Gm2c', 'C1'), ('LRx', 'C4')):
            if not ok(arm):
                continue
            ts = sorted(r['task'] for r in D[('N0', 0)][0])
            a0 = [{r['task']: r['acc'] for r in D[('N0', s)][0]} for s in SEEDS]
            aa = [{r['task']: r['acc'] for r in D[(arm, s)][0]} for s in SEEDS]
            g = np.median([[(aa[i][t] - a0[i][t]) * 100 for t in ts] for i in range(3)], axis=0)
            k = 5; ax[0].plot(ts[k - 1:], np.convolve(g, np.ones(k) / k, mode='valid'), color=c, label=arm)
        ax[0].axhline(0, color='gray', lw=.8); ax[0].set_xlabel('task'); ax[0].set_ylabel('acc − N0 [pt]'); ax[0].legend(fontsize=8)
        for arm in arms:
            if ok(arm):
                ax[1].scatter(med(arm, 'drift2'), float(np.median([dmg(arm, s) for s in SEEDS])), s=40)
                ax[1].annotate(arm, (med(arm, 'drift2'), float(np.median([dmg(arm, s) for s in SEEDS]))), fontsize=9, xytext=(4, 4), textcoords='offset points')
        ax[1].axhline(0, color='gray', lw=.8); ax[1].set_xlabel('drift2 (一歩の揃った成分)'); ax[1].set_ylabel('損傷 D [pt]')
        fig.tight_layout(); fig.savefig(OUT / 'fig_grad_floor.png', dpi=120)
    except Exception as e:
        print('figure skipped:', e)
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
