"""Verdicts for long_horizon_0910 (spec_long_horizon_0910 §2)."""
import csv, json
import numpy as np
from src import boundary_gradient_0908 as G

ROOT = G.ROOT; OUT = ROOT / 'results/long_horizon_0910'
ARMS = ['LR', 'ELU1', 'SNA']; SEEDS = [0, 1, 2]
LATE = (301, 400); EARLY = (20, 120)


def read(arm, s):
    r = list(csv.DictReader(open(OUT / f'{arm}_none_s{s}_rows.csv')))
    return {k: np.array([float(x[k]) if x[k] not in ('', None) else np.nan for x in r]) for k in r[0] if k not in ('arm', 'iv', 'clamp')}


def slope(d, key, lo, hi):        # pt per 100 tasks for acc, raw units otherwise
    m = (d['task'] >= lo) & (d['task'] <= hi); return float(np.polyfit(d['task'][m], d[key][m], 1)[0]) * 100


def win(d, key, lo, hi, f=np.median):
    m = (d['task'] >= lo) & (d['task'] <= hi); return float(f(d[key][m]))


def agree(vals, rule):
    got = [rule(v) for v in vals]; return got[0] if len(set(got)) == 1 else 'PARTIAL'


def main():
    D = {a: {s: read(a, s) for s in SEEDS} for a in ARMS}
    rows = []
    for a in ARMS:
        for s in SEEDS:
            d = D[a][s]
            rows.append(dict(arm=a, seed=s, acc_early=win(d, 'acc', *EARLY), acc_late=win(d, 'acc', *LATE),
                             acc_t400=float(d['acc'][d['task'] == 400][0]),
                             slope_early=slope(d, 'acc', *EARLY) * 100, slope_late=slope(d, 'acc', *LATE) * 100,
                             dead_hard_t400=int(d['dead_hard'][d['task'] == 400][0]), dead_hard_t120=int(d['dead_hard'][d['task'] == 120][0]),
                             dead_soft_t400=int(d['dead_soft'][d['task'] == 400][0]), sat_late=win(d, 'sat', *LATE),
                             zbar_late=win(d, 'zbar_inv', *LATE), sigma_late=win(d, 'sigma_inv', *LATE),
                             cnorm_t120=float(d['cnorm'][d['task'] == 120][0]), cnorm_t400=float(d['cnorm'][d['task'] == 400][0]),
                             pos_late=win(d, 'pos_frac', *LATE), w2col_t400=float(d['w2col'][d['task'] == 400][0])))
    keys = list(rows[0]); G.B.csvwrite(OUT / 'seed_verdict.csv', [{k: r[k] for k in keys} for r in rows])
    R = lambda a, k: [next(r[k] for r in rows if r['arm'] == a and r['seed'] == s) for s in SEEDS]
    dA = [100 * (x - y) for x, y in zip(R('LR', 'acc_late'), R('ELU1', 'acc_late'))]
    A = agree(dA, lambda x: 'ELU_DIVERGES' if x >= 1.0 else ('NO_DIVERGENCE' if abs(x) < .5 else 'MID'))
    B = agree(R('LR', 'slope_late'), lambda x: 'LEAKY_PLATEAUS' if x > -.5 else ('LEAKY_KEEPS_FALLING' if x < -1.0 else 'MID'))
    C = agree([l - e for l, e in zip(R('ELU1', 'slope_late'), R('ELU1', 'slope_early'))],
              lambda x: 'ELU_ACCELERATES' if x < 0 else 'ELU_DECELERATES')
    Dd = agree(R('ELU1', 'dead_hard_t400'), lambda x: 'DEATH_ACCUMULATES' if x >= 20 else ('DEATH_SATURATES' if x <= 10 else 'MID'))
    A = A if A not in ('MID',) else 'PARTIAL'; B = 'PARTIAL' if B == 'MID' else B; Dd = 'PARTIAL' if Dd == 'MID' else Dd
    verdict = dict(A_divergence=A, B_leaky_plateau=B, C_elu_acceleration=C, D_death=Dd,
                   dA_pt=';'.join(f'{x:+.2f}' for x in dA), LR_slope_late=';'.join(f'{x:+.2f}' for x in R('LR', 'slope_late')),
                   LR_slope_early=';'.join(f'{x:+.2f}' for x in R('LR', 'slope_early')),
                   ELU_slope_late=';'.join(f'{x:+.2f}' for x in R('ELU1', 'slope_late')), ELU_slope_early=';'.join(f'{x:+.2f}' for x in R('ELU1', 'slope_early')),
                   ELU_dead_hard_t400=';'.join(str(x) for x in R('ELU1', 'dead_hard_t400')), LR_dead_hard_max=max(int(D['LR'][s]['dead_hard'].max()) for s in SEEDS))
    G.B.csvwrite(OUT / 'verdict.csv', [verdict])
    S = ['# long_horizon_0910 summary', '', 'spec `specs/spec_long_horizon_0910.md`（単独 commit 34f155e）。窓: early = t20–120、late = t301–400。精度は窓内タスク終端の中央値、傾きは pt/100 task。', '',
         '## 事前登録の判定', '', '| 判定 | ラベル | seed 別 |', '|---|---|---|',
         f"| A 分岐 (acc LR − ELU, late) | **{A}** | {verdict['dA_pt']} pt |",
         f"| B leaky の late 傾き | **{B}** | {verdict['LR_slope_late']} (early {verdict['LR_slope_early']}) |",
         f"| C ELU の加速 | **{C}** | late {verdict['ELU_slope_late']} vs early {verdict['ELU_slope_early']} |",
         f"| D ELU の dead_hard(t400) | **{Dd}** | {verdict['ELU_dead_hard_t400']} (t120: {';'.join(str(x) for x in R('ELU1','dead_hard_t120'))}) |",
         f"| 検査: leaky の dead_hard | max {verdict['LR_dead_hard_max']} (must be 0) | |", '',
         '## seed 別', '', '| arm | seed | acc early | acc late | acc t400 | slope early | slope late | dead_hard 120→400 | dead_soft 400 | sat late | z̄ late | σ late | ‖W̃‖ 120→400 | p⁺ late |', '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        S.append(f"| {r['arm']} | {r['seed']} | {r['acc_early']:.4f} | {r['acc_late']:.4f} | {r['acc_t400']:.4f} | {r['slope_early']:+.2f} | {r['slope_late']:+.2f} | {r['dead_hard_t120']}→{r['dead_hard_t400']} | {r['dead_soft_t400']} | {r['sat_late']:.2f} | {r['zbar_late']:+.2f} | {r['sigma_late']:.2f} | {r['cnorm_t120']:.2f}→{r['cnorm_t400']:.2f} | {r['pos_late']:.3f} |")
    (OUT / 'summary.md').write_text('\n'.join(S)); print('\n'.join(S))
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    COL = {'LR': '#0072B2', 'ELU1': '#E69F00', 'SNA': '#D55E00'}; LAB = {'LR': 'leaky 0.1', 'ELU1': 'ELU α=1', 'SNA': 'Snake'}
    fig, ax = plt.subplots(1, 4, figsize=(17, 4)); k = 20
    for a in ARMS:
        for s in SEEDS:
            d = D[a][s]; t = d['task']; kw = dict(color=COL[a], lw=1.4 if s == 0 else .7, alpha=1 if s == 0 else .4)
            sm = np.convolve(d['acc'], np.ones(k) / k, mode='valid')
            ax[0].plot(t[k - 1:], sm, label=LAB[a] if s == 0 else None, **kw)
            ax[1].plot(t, d['dead_hard'], **kw); ax[1].plot(t, d['dead_soft'], ls=':', **kw)
            ax[2].plot(t, d['zbar_inv'], **kw); ax[3].plot(t, d['cnorm'], **kw)
    ax[0].set_ylabel('test accuracy (20-task moving mean)'); ax[0].legend(fontsize=8, frameon=False); ax[0].set_title('A/B/C: accuracy to t400', loc='left', fontsize=9.5)
    ax[1].set_ylabel('dead units  (solid φ′<1e−6, dotted φ′<0.05)'); ax[1].set_title('D: death', loc='left', fontsize=9.5)
    ax[2].set_ylabel('z̄ (inv)'); ax[2].set_title('sinking', loc='left', fontsize=9.5); ax[3].set_ylabel('‖W̃ᵢ‖ mean'); ax[3].set_title('width', loc='left', fontsize=9.5)
    for a_ in ax:
        a_.set_xlabel('task'); a_.axvspan(301, 400, color='#eee', zorder=0); a_.spines[['top', 'right']].set_visible(False); a_.grid(axis='y', color='#eee', lw=.6)
    fig.tight_layout(); fig.savefig(OUT / 'fig_long_horizon.png', dpi=150)


if __name__ == '__main__':
    main()
