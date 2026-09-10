"""Verdicts for clamp_horizon_0910 (spec_clamp_horizon_0910 §2).

rho_C = (acc_C(late) - acc_ref(late)) / (acc_base - acc_ref(late))
      = the share of the reference's accumulated loss that clamp C recovers.
late = t301..400, base = t16..20 (the shared prefix, identical across clamps).
"""
from pathlib import Path
import csv, json
import numpy as np
from src import boundary_gradient_0908 as G

ROOT = G.ROOT
OUT = ROOT / 'results/clamp_horizon_0910'
ARMS = ['LR', 'ELU1']
CLAMPS = ['ref', 'dclamp', 'wclamp', 'wcap2']
SEEDS = [0, 1, 2]
LATE = (301, 400)
BASE = (16, 20)
G0_MIN_LOSS = 0.02          # 2.0 pt
HI, LO = 0.5, 0.2           # rho thresholds
G3_CE_FRAC = 0.90
G4_HOLD = 0.25
G5_DEPTH_TOL = 1.0
D_WIDTH_MIN = 0.75
E_TOL = 0.1


def read(arm, seed):
    rows = list(csv.DictReader(open(OUT / f'{arm}_none_s{seed}_rows.csv')))
    for r in rows:
        for k, v in list(r.items()):
            if k in ('arm', 'iv', 'clamp'):
                continue
            r[k] = float(v) if v not in ('', None) else np.nan
    return rows


def series(rows, clamp):
    return sorted([r for r in rows if r['clamp'] == clamp], key=lambda r: r['task'])


def win(rows, key, lo, hi, f=np.median):
    v = [r[key] for r in rows if lo <= r['task'] <= hi and np.isfinite(r[key])]
    return float(f(v)) if v else np.nan


def per_seed(arm, seed):
    rows = read(arm, seed)
    S = {c: series(rows, c) for c in CLAMPS}
    # the shared prefix is stored under 'ref' for t<=20 and is identical for every clamp
    acc_base = win(S['ref'], 'acc', *BASE)
    z20 = win(S['ref'], 'zbar_inv', 20, 20)
    out = {}
    for c in CLAMPS:
        e = S[c]
        late = [r for r in e if LATE[0] <= r['task'] <= LATE[1]]
        ce_ok = [1 for r in late if np.isfinite(r['ce20']) and r['ce20'] - r['ce_probe'] > 0]
        t = np.array([r['task'] for r in late]); a = np.array([r['acc'] for r in late])
        out[c] = dict(acc_late=float(np.median(a)), acc_base=acc_base,
                      slope_late=float(np.polyfit(t, a, 1)[0]) * 100 * 100,
                      ce_frac=len(ce_ok) / max(1, len(late)),
                      zbar_late=win(e, 'zbar_inv', *LATE), z20=z20,
                      sigma_late=win(e, 'sigma_inv', *LATE), star_sd_late=win(e, 'star_sd', *LATE),
                      cnorm_late=win(e, 'cnorm', *LATE), pos_late=win(e, 'pos_frac', *LATE),
                      w2col_late=win(e, 'w2col', *LATE), dead_hard_late=win(e, 'dead_hard', *LATE),
                      dead_soft_late=win(e, 'dead_soft', *LATE), sat_late=win(e, 'sat', *LATE),
                      pressure=float(e[-1]['pressure']))
    L = acc_base - out['ref']['acc_late']
    for c in CLAMPS:
        out[c]['L_ref'] = L
        out[c]['rho'] = (out[c]['acc_late'] - out['ref']['acc_late']) / L if L > 0 else np.nan
    return out


def rule_rho(x):
    if not np.isfinite(x):
        return None
    return 'REMOVES' if x >= HI else ('NO_EFFECT' if x <= LO else 'PARTIAL')


def label3(vals, rule, flat='NOT_TESTABLE_NO_LOSS'):
    got = [rule(v) for v in vals]
    if any(g is None for g in got):
        return flat
    return got[0] if len(set(got)) == 1 else 'MIXED'


def main():
    per, seedrows, verdicts = {}, [], []
    for arm in ARMS:
        try:
            per[arm] = {s: per_seed(arm, s) for s in SEEDS}
        except FileNotFoundError as e:
            print('missing', e); continue
        for s in SEEDS:
            for c in CLAMPS:
                seedrows.append(dict(arm=arm, seed=s, clamp=c, **per[arm][s][c]))
        R = lambda c, k: [per[arm][s][c][k] for s in SEEDS]
        testable = [per[arm][s]['ref']['L_ref'] >= G0_MIN_LOSS for s in SEEDS]
        g3 = {c: [per[arm][s][c]['ce_frac'] >= G3_CE_FRAC for s in SEEDS] for c in CLAMPS}
        g4 = [abs(per[arm][s]['dclamp']['zbar_late'] - per[arm][s]['dclamp']['z20'])
              < G4_HOLD * abs(per[arm][s]['ref']['zbar_late'] - per[arm][s]['ref']['z20']) for s in SEEDS]
        g5 = [per[arm][s]['wclamp']['zbar_late'] <= per[arm][s]['ref']['zbar_late'] + G5_DEPTH_TOL for s in SEEDS]

        def rhos(c):
            return [per[arm][s][c]['rho'] if (testable[s] and g3[c][s]) else np.nan for s in SEEDS]

        rA, rB, rE = rhos('dclamp'), rhos('wclamp'), rhos('wcap2')
        A = label3(rA, rule_rho); B = label3(rB, rule_rho)
        A = f'DEPTH_{A}' if A in ('REMOVES', 'NO_EFFECT', 'PARTIAL') else A
        A = A.replace('DEPTH_REMOVES', 'DEPTH_REMOVES_LOSS')
        B = f'WIDTH_{B}' if B in ('REMOVES', 'NO_EFFECT', 'PARTIAL') else B
        B = B.replace('WIDTH_REMOVES', 'WIDTH_REMOVES_LOSS')
        if not all(g3['dclamp']):
            A = 'NOT_TESTABLE_LEARNING_BROKEN'
        if not all(g3['wclamp']):
            B = 'NOT_TESTABLE_LEARNING_BROKEN'
        if not all(g4):
            A = 'NOT_TESTABLE_DEPTH_NOT_HELD'
        # C: width removes the loss WHILE sinking continues
        if not all(g5):
            Cl = 'NOT_TESTABLE_DEPTH_ALSO_STOPPED'
        else:
            Cl = 'WIDTH_WITHOUT_DEPTH' if all(np.isfinite(x) and x >= HI for x in rB) else 'NOT_SHOWN'
        # D: depth removes the loss WHILE width grows
        wr = [per[arm][s]['dclamp']['cnorm_late'] / per[arm][s]['ref']['cnorm_late'] for s in SEEDS]
        Dl = 'DEPTH_WITHOUT_WIDTH' if (all(np.isfinite(x) and x >= HI for x in rA)
                                       and all(w >= D_WIDTH_MIN for w in wr)) else 'NOT_SHOWN'
        # E: dose
        pairs = list(zip(rE, rB))
        if any(not (np.isfinite(a) and np.isfinite(b)) for a, b in pairs):
            El = 'NOT_TESTABLE_NO_LOSS'
        elif all(-E_TOL <= a <= b + E_TOL for a, b in pairs):
            El = 'WIDTH_DOSE_MONOTONE'
        elif all(a > b + E_TOL for a, b in pairs):
            El = 'CAP_BEATS_CLAMP'
        else:
            El = 'DOSE_PARTIAL'
        j = lambda v, f='{:+.3f}': ';'.join(f.format(x) if np.isfinite(x) else 'NA' for x in v)
        verdicts.append(dict(arm=arm, A_depth=A, B_width=B, C_dissoc=Cl, D_dissoc=Dl, E_dose=El,
                             rho_dclamp=j(rA), rho_wclamp=j(rB), rho_wcap2=j(rE),
                             L_ref_pt=j([100 * per[arm][s]['ref']['L_ref'] for s in SEEDS], '{:.2f}'),
                             acc_base=j([per[arm][s]['ref']['acc_base'] for s in SEEDS], '{:.4f}'),
                             zbar_late=';'.join(f"{per[arm][s]['ref']['zbar_late']:+.2f}/{per[arm][s]['wclamp']['zbar_late']:+.2f}/{per[arm][s]['dclamp']['zbar_late']:+.2f}" for s in SEEDS),
                             cnorm_ratio_dclamp=j(wr), g3=''.join('1' if all(g3[c]) else '0' for c in CLAMPS),
                             g4=''.join('1' if x else '0' for x in g4), g5=''.join('1' if x else '0' for x in g5)))
    OUT.mkdir(parents=True, exist_ok=True)
    for name, rr in [('seed_verdict.csv', seedrows), ('verdict.csv', verdicts)]:
        if rr:
            keys = []; [keys.append(k) for r in rr for k in r if k not in keys]
            G.B.csvwrite(OUT / name, [{k: r.get(k) for k in keys} for r in rr])
    summary(per, verdicts); figure(per)
    for v in verdicts:
        print(v['arm'], v['A_depth'], '|', v['B_width'], '|', v['C_dissoc'], '|', v['D_dissoc'], '|', v['E_dose'])
        print('   rho d/w/cap:', v['rho_dclamp'], '/', v['rho_wclamp'], '/', v['rho_wcap2'], '  L_ref', v['L_ref_pt'], 'pt')


def md(rows, cols):
    return '\n'.join(['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)] +
                     ['| ' + ' | '.join(str(r.get(c, '')) for c in cols) + ' |' for r in rows])


def summary(per, verdicts):
    S = ['# clamp_horizon_0910 summary', '',
         'spec `specs/spec_clamp_horizon_0910.md`（単独 commit `7ec15b7`）。窓は late = t301–400、base = t16–20（共有前置き）。',
         'ρ = (acc_C − acc_ref)/(acc_base − acc_ref) ＝ 参照の累積損失のうちその腕が取り戻した割合。ラベルは可検定な 3 seed が 3/3 一致したときのみ。', '',
         '## 事前登録の判定', '', md(verdicts, ['arm', 'A_depth', 'B_width', 'C_dissoc', 'D_dissoc', 'E_dose']), '',
         '## 連続量（seed 別）', '', md(verdicts, ['arm', 'rho_dclamp', 'rho_wclamp', 'rho_wcap2', 'L_ref_pt', 'acc_base']), '',
         '`zbar_late`（ref/wclamp/dclamp）と `cnorm_ratio_dclamp`、ゲート G3/G4/G5:', '',
         md(verdicts, ['arm', 'zbar_late', 'cnorm_ratio_dclamp', 'g3', 'g4', 'g5']), '',
         '## 副測定（late 窓・seed 中央値）', '',
         '| arm | clamp | acc | slope (pt/100task) | z̄ | σ_inv | star_sd | ‖W̃‖ | p⁺ | ‖W2col‖ | dead_hard | 飽和対 |', '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for arm in per:
        for c in CLAMPS:
            v = [per[arm][s][c] for s in SEEDS]
            g = lambda k, f='{:.3f}': f.format(float(np.median([x[k] for x in v])))
            S.append(f"| {arm} | {c} | {g('acc_late','{:.4f}')} | {g('slope_late','{:+.2f}')} | {g('zbar_late','{:+.2f}')} | {g('sigma_late')} | "
                     f"{g('star_sd_late')} | {g('cnorm_late','{:.2f}')} | {g('pos_late')} | {g('w2col_late','{:.2f}')} | {g('dead_hard_late','{:.0f}')} | {g('sat_late','{:.2f}')} |")
    S += ['', '検算値は各腕の `*_provenance.json` の `checks`（G1 は t21–100 の committed 再現）。', '図: `fig_clamp_horizon.png`。']
    (OUT / 'summary.md').write_text('\n'.join(S))


def figure(per):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    COL = {'ref': '#0072B2', 'dclamp': '#E69F00', 'wclamp': '#D55E00', 'wcap2': '#009E73'}
    arms = list(per); k = 20
    fig, ax = plt.subplots(len(arms), 4, figsize=(16, 3.6 * len(arms)), squeeze=False)
    for i, arm in enumerate(arms):
        for c in CLAMPS:
            for s in SEEDS:
                e = series(read(arm, s), c); t = np.array([r['task'] for r in e])
                kw = dict(color=COL[c], lw=1.3 if s == 0 else .6, alpha=1 if s == 0 else .35)
                a = np.array([r['acc'] for r in e])
                if len(a) > k:
                    ax[i][0].plot(t[k - 1:], np.convolve(a, np.ones(k) / k, 'valid'), label=c if s == 0 else None, **kw)
                ax[i][1].plot(t, [r['zbar_inv'] for r in e], **kw)
                ax[i][2].plot(t, [r['cnorm'] for r in e], **kw)
                ax[i][3].plot(t, [r['pos_frac'] for r in e], **kw)
        ax[i][0].set_ylabel(f'{arm}\naccuracy (20-task mean)'); ax[i][0].legend(fontsize=8, frameon=False)
        for j, ti in enumerate(['accuracy', 'sinking  z̄', 'width  ‖W̃ᵢ‖', 'occupancy  p⁺']):
            ax[i][j].set_title(ti if i == 0 else '', loc='left', fontsize=10)
            ax[i][j].axvspan(301, 400, color='#eee', zorder=0)
            ax[i][j].spines[['top', 'right']].set_visible(False); ax[i][j].grid(axis='y', color='#eee', lw=.6)
            if i == len(arms) - 1:
                ax[i][j].set_xlabel('task')
    fig.tight_layout(); fig.savefig(OUT / 'fig_clamp_horizon.png', dpi=150)


if __name__ == '__main__':
    main()
