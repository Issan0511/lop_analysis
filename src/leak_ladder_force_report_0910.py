"""Force curve across the leak ladder: threshold p*(a) and stiffness vs a.

Tests Leak距離補償仮説_0908 §2.2 (self-moment: negative side weighted a^2)
against §2.3 (residual-weighted update: negative side weighted a).
  p*/(1-p*) ∝ a^gamma   with gamma≈1 (§2.3) or ≈2 (§2.2)
Per-unit sinking over the next task, binned by occupancy Phi(zbar_i/sd_i).
"""
from pathlib import Path
import csv
import numpy as np
from math import erf, sqrt
from src import boundary_gradient_0908 as G

ROOT = G.ROOT
OUT = ROOT / 'results/leak_ladder_force_posthoc_0910'
SBAR = 103.74
Phi = np.vectorize(lambda x: .5 * (1 + erf(x / sqrt(2))))
# arm -> (leak, npz dir, key prefix)
LADDER = {'LR03': (.3, OUT, 'ref'), 'LR': (.1, ROOT / 'results/width_sink_clamp_0909', 'ref'),
          'LR001': (.01, OUT, 'ref'), 'R': (0., OUT, 'ref')}
EDGES = np.array([0, .01, .02, .035, .05, .07, .1, .14, .2, .28, .4, .55, 1.0])


def curve(arm):
    a, d, pre = LADDER[arm]
    X, Y = [], []
    for s in range(3):
        f = d / (f'{arm}_none_s{s}_units.npz' if arm == 'LR' else f'{arm}_s{s}_units.npz')
        z = np.load(f)
        for t in range(20, 100):
            zb, sd = z[f'{pre}_zbar_i_t{t}'], z[f'{pre}_sd_i_t{t}']
            X.append(Phi(zb / np.maximum(sd, 1e-12))); Y.append((z[f'{pre}_m_i_t{t + 1}'] - z[f'{pre}_m_i_t{t}']) * SBAR)
    X, Y = np.concatenate(X), np.concatenate(Y)
    xs, ys, ses, ns = [], [], [], []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        m = (X >= lo) & (X < hi)
        if m.sum() < 40:
            continue
        xs.append(float(np.sqrt(lo * hi)) if lo > 0 else float(hi / 2)); ys.append(float(Y[m].mean())); ses.append(float(Y[m].std() / np.sqrt(m.sum()))); ns.append(int(m.sum()))
    xs, ys = np.array(xs), np.array(ys)
    # zero crossing by linear interpolation between the last positive and first negative bin (in log p)
    zc, slope = np.nan, np.nan
    for i in range(len(ys) - 1):
        if ys[i] > 0 and ys[i + 1] < 0:
            lx0, lx1 = np.log(xs[i]), np.log(xs[i + 1])
            zc = float(np.exp(lx0 + (lx1 - lx0) * ys[i] / (ys[i] - ys[i + 1])))
            slope = float((ys[i + 1] - ys[i]) / (lx1 - lx0))   # dF / d ln p at the crossing
            break
    sat = float(np.min(ys)) if len(ys) else np.nan
    return dict(arm=arm, leak=a, n=int(len(X)), zero_crossing=zc, odds=zc / (1 - zc) if np.isfinite(zc) else np.nan,
                slope_dF_dlnp=slope, saturation=sat, frac_units_below=float(np.mean(X < zc)) if np.isfinite(zc) else np.nan,
                pooled_pos=float(X.mean()), bins=list(zip(xs.tolist(), ys.tolist(), ses, ns)))


def main():
    rows = [curve(a) for a in LADDER]
    keys = ['arm', 'leak', 'n', 'zero_crossing', 'odds', 'slope_dF_dlnp', 'saturation', 'frac_units_below', 'pooled_pos']
    G.B.csvwrite(OUT / 'ladder_summary.csv', [{k: r[k] for k in keys} for r in rows])
    binrows = [dict(arm=r['arm'], leak=r['leak'], p_mid=b[0], dz_per_task=b[1], se=b[2], n=b[3]) for r in rows for b in r['bins']]
    G.B.csvwrite(OUT / 'ladder_bins.csv', binrows)
    print(f"{'arm':6s} {'a':>5s} {'p*':>7s} {'p*/(1-p*)':>10s} {'dF/dlnp @p*':>12s} {'saturation':>11s} {'units<p*':>9s} {'pooled p+':>10s}")
    for r in rows:
        print(f"{r['arm']:6s} {r['leak']:5.2f} {r['zero_crossing']:7.4f} {r['odds']:10.4f} {r['slope_dF_dlnp']:12.4f} {r['saturation']:11.3f} {r['frac_units_below']:9.2f} {r['pooled_pos']:10.3f}")
    # scaling of the threshold odds with a (leaky arms only; ReLU is a=0 and cannot enter a log fit)
    lk = [r for r in rows if r['leak'] > 0 and np.isfinite(r['odds'])]
    if len(lk) >= 3:
        la = np.log([r['leak'] for r in lk]); lo = np.log([r['odds'] for r in lk]); ls = np.log([-r['slope_dF_dlnp'] for r in lk if r['slope_dF_dlnp'] < 0])
        g = np.polyfit(la, lo, 1)[0]
        print(f"\np*/(1-p*) ∝ a^gamma:  gamma = {g:.2f}   (§2.3 residual-weighted predicts ≈1, §2.2 self-moment predicts ≈2)")
        if len(ls) == len(la):
            print(f"stiffness |dF/dlnp| at p* ∝ a^beta:  beta = {np.polyfit(la, ls, 1)[0]:.2f}   (井戸の剛性_0907 found k ∝ a^2.77 in the offset system)")
    for r in rows:
        print(f"\n{r['arm']} (a={r['leak']}) bins: " + '  '.join(f"{b[0]:.3f}:{b[1]:+.3f}±{b[2]:.3f}" for b in r['bins']))
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    col = {'LR03': '#009E73', 'LR': '#0072B2', 'LR001': '#E69F00', 'R': '#D55E00'}
    for r in rows:
        b = np.array([[x[0], x[1], x[2]] for x in r['bins']])
        ax[0].errorbar(b[:, 0], b[:, 1], yerr=b[:, 2], color=col[r['arm']], marker='o', ms=4, lw=1.4, capsize=2, label=f"{r['arm']}  a={r['leak']}")
        if np.isfinite(r['zero_crossing']):
            ax[0].axvline(r['zero_crossing'], color=col[r['arm']], lw=.8, ls=':')
    ax[0].axhline(0, color='#888', lw=.7); ax[0].set_xscale('log'); ax[0].set_xlabel('unit occupancy p⁺ᵢ = Φ(z̄ᵢ/σᵢ)  (log)'); ax[0].set_ylabel('Δz̄ᵢ over the next task')
    ax[0].legend(fontsize=8, frameon=False); ax[0].set_title('Force curve across the leak ladder (dotted = zero crossing)', loc='left', fontsize=9.5)
    if len(lk) >= 3:
        ax[1].plot([r['leak'] for r in lk], [r['odds'] for r in lk], 'o', color='#0072B2', ms=6)
        aa = np.array([min(r['leak'] for r in lk), max(r['leak'] for r in lk)])
        c0 = np.exp(np.polyfit(la, lo, 1)[1])
        ax[1].plot(aa, c0 * aa ** g, color='#0072B2', lw=1.2, label=f'fit  ∝ a^{g:.2f}')
        ax[1].plot(aa, lk[1]['odds'] * (aa / lk[1]['leak']) ** 1, color='#009E73', lw=1, ls='--', label='∝ a   (§2.3 residual-weighted)')
        ax[1].plot(aa, lk[1]['odds'] * (aa / lk[1]['leak']) ** 2, color='#D55E00', lw=1, ls='--', label='∝ a²  (§2.2 self-moment)')
        ax[1].set_xscale('log'); ax[1].set_yscale('log'); ax[1].set_xlabel('leak a'); ax[1].set_ylabel('threshold odds p*/(1−p*)'); ax[1].legend(fontsize=8, frameon=False)
        ax[1].set_title('Threshold vs leak', loc='left', fontsize=9.5)
    for a_ in ax:
        a_.spines[['top', 'right']].set_visible(False); a_.grid(axis='y', color='#eee', lw=.6)
    fig.tight_layout(); fig.savefig(OUT / 'fig_leak_ladder.png', dpi=150)


if __name__ == '__main__':
    main()
