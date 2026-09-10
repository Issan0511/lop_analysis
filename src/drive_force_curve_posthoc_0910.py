"""The driving source as a force curve (post-hoc, unregistered, 2026-09-10).

Three views of the same force on a unit's row mean m_i, all from committed data:

  A. F(p+_i): per-unit sinking over the next task, binned by the unit's positive-side
     occupancy (Gaussian proxy Phi(zbar_i/sd_i) under the reference perms).
     width_sink_clamp_0909 ref arm, t20..100, 3 seeds -> 24 000 unit-tasks.
  B. the two side components of the brightness gradient, binned by occupancy
     (growth_engine_posthoc_0910 units: gpos, gneg, pos0 under the current perm),
     and the balance point they predict.
  C. the force blocked by dclamp (removed mbar shift per task) versus the natural
     sinking rate, i.e. how much of the push is normally relieved by sinking.

Writes results/drive_force_curve_posthoc_0910/{force_curve.csv,side_balance.csv,
blocked_force.csv,summary.md,fig_force_curve.png}.
"""
from pathlib import Path
import csv
import numpy as np
from math import erf, sqrt
from src import boundary_gradient_0908 as G
import src.width_sink_clamp_report_0909 as R

ROOT = G.ROOT
OUT = ROOT / 'results/drive_force_curve_posthoc_0910'
GE = ROOT / 'results/growth_engine_posthoc_0910'
SBAR = 103.74
ARMS = ['LR', 'ELU1', 'SNA']
LAB = {'LR': 'leaky 0.1', 'ELU1': 'ELU α=1', 'SNA': 'Snake'}
COL = {'LR': '#0072B2', 'ELU1': '#E69F00', 'SNA': '#D55E00'}
EDGES = [0, .02, .05, .08, .12, .16, .20, .25, .32, .40, .55, 1.0]
Phi = np.vectorize(lambda x: .5 * (1 + erf(x / sqrt(2))))


def csvw(name, rows):
    keys = []
    [keys.append(k) for r in rows for k in r if k not in keys]
    G.B.csvwrite(OUT / name, [{k: r.get(k) for k in keys} for r in rows])


def force_curve():
    rows = []
    for arm in ARMS:
        X, Y = [], []
        for s in range(3):
            d = np.load(ROOT / f'results/width_sink_clamp_0909/{arm}_none_s{s}_units.npz')
            for t in range(20, 100):
                zb, sd = d[f'ref_zbar_i_t{t}'], d[f'ref_sd_i_t{t}']
                X.append(Phi(zb / sd)); Y.append((d[f'ref_m_i_t{t + 1}'] - d[f'ref_m_i_t{t}']) * SBAR)
        X, Y = np.concatenate(X), np.concatenate(Y)
        prev = None; zero = None
        for lo, hi in zip(EDGES[:-1], EDGES[1:]):
            m = (X >= lo) & (X < hi)
            if m.sum() < 30:
                continue
            mu, se = float(Y[m].mean()), float(Y[m].std() / np.sqrt(m.sum()))
            if prev is not None and prev > 0 and mu < 0 and zero is None:
                zero = lo
            rows.append(dict(arm=arm, p_lo=lo, p_hi=hi, n=int(m.sum()), dz_per_task=mu, se=se))
            prev = mu
        rows.append(dict(arm=arm, p_lo='zero_crossing', p_hi='', n=len(X), dz_per_task=zero, se=float(Y.mean())))
    csvw('force_curve.csv', rows)
    return rows


def side_balance():
    rows = []
    for arm in ARMS:
        U = R.load(GE / f'{arm}_s0_units.csv') if hasattr(R, 'load') else None
        r = list(csv.DictReader(open(GE / f'{arm}_s0_units.csv')))
        p0 = np.array([float(x['pos0']) for x in r]); gp = np.array([float(x['gpos']) for x in r])
        gn = np.array([float(x['gneg']) for x in r]); dm = np.array([float(x['dm']) for x in r]) * SBAR
        live = p0 > 0
        # push per unit of mass on each side (task-summed brightness gradient / occupancy)
        Ep = np.median(gp[live] / p0[live]); En = np.median(gn[p0 < 1] / (1 - p0[p0 < 1]))
        pstar = -En / (Ep - En) if Ep > En else float('nan')
        for lo, hi in zip(EDGES[:-1], EDGES[1:]):
            m = (p0 >= lo) & (p0 < hi)
            if m.sum() < 15:
                continue
            rows.append(dict(arm=arm, p_lo=lo, p_hi=hi, n=int(m.sum()), gpos=float(gp[m].mean()), gneg=float(gn[m].mean()),
                             gnet=float((gp + gn)[m].mean()), dz_per_task=float(dm[m].mean()),
                             se_dz=float(dm[m].std() / np.sqrt(m.sum()))))
        rows.append(dict(arm=arm, p_lo='balance', p_hi='', n=int(live.sum()), gpos=float(Ep), gneg=float(En),
                         gnet=float(np.median(np.abs(gn[live]) / np.abs(gp[live]))), dz_per_task=float(pstar), se_dz=0.))
    csvw('side_balance.csv', rows)
    return rows


def blocked_force():
    rows = []
    for arm in ARMS:
        for s in range(3):
            rr = R.read(arm, s); d = R.series(rr, 'dclamp'); rf = R.series(rr, 'ref')
            t = np.array([x['task'] for x in d]); P = np.array([x['pressure'] for x in d]); sg = np.array([x['sigma_inv'] for x in d])
            f = np.diff(P) * SBAR; tt = t[1:]
            z = np.array([x['zbar_inv'] for x in rf]); tr = np.array([x['task'] for x in rf]); dz = np.diff(z)
            late = (tt >= 61)
            rows.append(dict(arm=arm, seed=s, blocked_early=float(f[(tt >= 21) & (tt <= 30)].mean()),
                             blocked_mid=float(f[(tt >= 61) & (tt <= 70)].mean()), blocked_late=float(f[(tt >= 91)].mean()),
                             blocked_t61_100=float(f[late].mean()), natural_t61_100=float(dz[tr[1:] >= 61].mean()),
                             corr_force_sigma=float(np.corrcoef(f, sg[1:])[0, 1]),
                             p_late_dclamp=float(np.mean([x['pos_frac'] for x in d if x['task'] >= 61]))))
    csvw('blocked_force.csv', rows)
    return rows


def figure(fc, sb, bf):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.4))
    for arm in ARMS:
        r = [x for x in fc if x['arm'] == arm and x['p_lo'] != 'zero_crossing']
        xm = [(x['p_lo'] + x['p_hi']) / 2 for x in r]
        ax[0].errorbar(xm, [x['dz_per_task'] for x in r], yerr=[x['se'] for x in r], color=COL[arm], marker='o', ms=4, lw=1.4, capsize=2, label=LAB[arm])
    ax[0].axhline(0, color='#888', lw=.7); ax[0].set_xscale('log'); ax[0].set_xlabel('unit occupancy p⁺ᵢ = Φ(z̄ᵢ/σᵢ)  (log)')
    ax[0].set_ylabel('Δz̄ᵢ over the next task'); ax[0].legend(fontsize=8, frameon=False)
    ax[0].set_title('A. Force curve: rise below a threshold, sink above, saturate', loc='left', fontsize=9.5)
    for arm in ARMS:
        r = [x for x in sb if x['arm'] == arm and x['p_lo'] != 'balance']
        xm = [(x['p_lo'] + x['p_hi']) / 2 for x in r]
        ax[1].plot(xm, [x['gpos'] for x in r], color=COL[arm], lw=1.4, marker='^', ms=4, label=f'{LAB[arm]}  z>0 side')
        ax[1].plot(xm, [x['gneg'] for x in r], color=COL[arm], lw=1.4, ls='--', marker='v', ms=4, label=f'{LAB[arm]}  z≤0 side')
    ax[1].axhline(0, color='#888', lw=.7); ax[1].set_xscale('log'); ax[1].set_xlabel('unit occupancy p⁺ᵢ (current perm, task start)')
    ax[1].set_ylabel('task-summed brightness gradient (>0 pushes m down)'); ax[1].legend(fontsize=7, frameon=False, ncol=1)
    ax[1].set_title('B. The two sides: positive side pushes down, negative side lifts', loc='left', fontsize=9.5)
    k = 0
    for arm in ARMS:
        rr = [x for x in bf if x['arm'] == arm]
        for x in rr:
            ax[2].bar(k, x['blocked_t61_100'], color=COL[arm], width=.8, alpha=.85)
            ax[2].bar(k, x['natural_t61_100'], color='#555', width=.8, alpha=.7)
            k += 1
        k += .6
    ax[2].axhline(0, color='#888', lw=.7); ax[2].set_xticks([1, 4.6, 8.2]); ax[2].set_xticklabels([LAB[a] for a in ARMS])
    ax[2].set_ylabel('z per task (t61–100)'); ax[2].set_title('C. Push blocked when depth is held (colour) vs natural sinking (grey)', loc='left', fontsize=9.5)
    for a in ax:
        a.spines[['top', 'right']].set_visible(False); a.grid(axis='y', color='#eee', lw=.6)
    fig.tight_layout(); fig.savefig(OUT / 'fig_force_curve.png', dpi=150)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fc, sb, bf = force_curve(), side_balance(), blocked_force()
    figure(fc, sb, bf)
    S = ['# drive_force_curve_posthoc_0910 — 駆動源を力の曲線として見る（事後・未登録）', '']
    S += ['## A. 力の曲線 F(p⁺ᵢ)（ref・t20–100・3 seed・24000 unit-task、p⁺ᵢ は参照 perm のガウス近似）', '']
    for arm in ARMS:
        z = [x for x in fc if x['arm'] == arm and x['p_lo'] == 'zero_crossing'][0]
        S.append(f"- {LAB[arm]}: 零交差 p⁺ᵢ ≈ {z['dz_per_task']}（集団平均 Δz̄/task = {z['se']:+.4f}）")
    S += ['', '## B. 両側の成分（growth_engine の計装窓・seed0・現 perm の占有率）', '']
    for arm in ARMS:
        b = [x for x in sb if x['arm'] == arm and x['p_lo'] == 'balance'][0]
        S.append(f"- {LAB[arm]}: 正側の押し/質量 E⁺ = {b['gpos']:+.3e}、負側 E⁻ = {b['gneg']:+.3e}、|E⁻|/|E⁺| 中央値 {b['gnet']:.2f} → 釣り合い p* = −E⁻/(E⁺−E⁻) = {b['dz_per_task']:.3f}")
    S += ['', '## C. 深さを固定したときに塞き止められる押し（dclamp の除去量）対 自然な沈下（t61–100、z/task）', '']
    for x in bf:
        S.append(f"- {LAB[x['arm']]} s{x['seed']}: 塞き止め {x['blocked_t61_100']:+.4f}、自然 {x['natural_t61_100']:+.4f}、比 {x['blocked_t61_100'] / max(1e-9, -x['natural_t61_100']):.1f}、corr(force, σ) {x['corr_force_sigma']:+.2f}、dclamp の p⁺ late {x['p_late_dclamp']:.3f}")
    (OUT / 'summary.md').write_text('\n'.join(S))
    print('\n'.join(S))


if __name__ == '__main__':
    main()
