"""Aggregation for growth_engine_posthoc_0910 (post-hoc, unregistered)."""
from pathlib import Path
import csv
import numpy as np
from src import boundary_gradient_0908 as G

ROOT = G.ROOT
OUT = ROOT / 'results/growth_engine_posthoc_0910'
ARMS = ['LR', 'ELU1', 'SNA']
WINDOWS = [(21, 25), (96, 100)]
SBAR = 103.74          # mean total ink of the probe: z units = 103.74 x row-mean units


def load(f):
    r = list(csv.DictReader(open(f)))
    return {k: np.array([float(x[k]) for x in r]) for k in r[0] if k != 'arm'}


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def per_arm(arm):
    S = load(OUT / f'{arm}_s0_steps.csv'); U = load(OUT / f'{arm}_s0_units.csv')
    rows = []
    for lo, hi in WINDOWS:
        w = (S['task'] >= lo) & (S['task'] <= hi); uw = (U['task'] >= lo) & (U['task'] <= hi)
        n = hi - lo + 1
        tot = lambda k: float(S[k][w].sum() / n)                    # per-task total
        # --- W~ budget, task level, referenced to the task-start weights ---
        dn2 = float(U['dnorm2'][uw].sum() / n / 100)                 # mean over units of d||W~_i||^2 per task
        al = float(U['align_task'][uw].sum() / n / 100)
        dp = float(U['dep_task'][uw].sum() / n / 100)
        dep_diag = tot('dep')                                        # sum_s ||dW~_s||^2 only
        cos_task = float(np.nanmedian(U['cos_task'][uw]))
        # --- m budget, task level, from per-unit task sums ---
        dmbar = float(np.mean([U['dm'][uw & (U['task'] == t)].mean() for t in range(lo, hi + 1)]))
        cov2, vdm = [], []
        for t in range(lo, hi + 1):
            m0 = U['m0'][uw & (U['task'] == t)]; dm = U['dm'][uw & (U['task'] == t)]
            cov2.append(2 * np.mean((m0 - m0.mean()) * (dm - dm.mean()))); vdm.append(dm.var())
        # --- Adam vs gradient ---
        fr_a, fr_g = float(np.nanmean(S['fr_adam'][w])), float(np.nanmean(S['fr_grad'][w]))
        amp_row = float(S['adam_rowstep'][w].mean() / S['grad_rowstep'][w].mean())
        amp_ctr = float(S['adam_ctrstep'][w].mean() / S['grad_ctrstep'][w].mean())
        # net drift vs per-step motion of the row mean
        rect = abs(dmbar / 625) / float(S['adam_rowstep'][w].mean())
        # --- side split of the brightness gradient (per task) ---
        gp, gn = tot('grow_pos'), tot('grow_neg')
        # --- within-task cumulative profile of d mbar (z units) ---
        prof = []
        for t in range(lo, hi + 1):
            c = np.cumsum(S['dmbar'][w & (S['task'] == t)]) * SBAR
            prof.append([c[19], c[99], c[299], c[-1]])
        prof = np.mean(prof, 0)
        # deposition profile (share by step)
        dprof = []
        for t in range(lo, hi + 1):
            c = np.cumsum(S['dep'][w & (S['task'] == t)]); dprof.append([c[19] / c[-1], c[99] / c[-1], c[299] / c[-1]])
        dprof = np.mean(dprof, 0)
        # --- per-unit ---
        dm = U['dm'][uw]; m0 = U['m0'][uw]; cn = U['cnorm0'][uw]; fo = U['fossil'][uw]; z0 = U['zbar0'][uw]; p0 = U['pos0'][uw]
        gpu = U['gpos'][uw]; gnu = U['gneg'][uw]
        q = np.quantile(z0, [.25, .75]); deep = z0 < q[0]; shallow = z0 >= q[1]
        rows.append(dict(arm=arm, window=f't{lo}-{hi}',
                         dnorm2_task=dn2, align_task=al, dep_task=dp, dep_diag_only=dep_diag, cos_task=cos_task,
                         dmbar_task=dmbar, dmbar_task_z=dmbar * SBAR, cov2_task=float(np.mean(cov2)), vdm_task=float(np.mean(vdm)),
                         fr_row_adam=fr_a, fr_row_grad=fr_g, amp_row=amp_row, amp_ctr=amp_ctr,
                         sign_agree=float(S['sign_agree'][w].mean()), rectification=rect,
                         grow_pos=gp, grow_neg=gn, pos_frac=float(S['pos_frac'][w].mean()),
                         zero_grad_units=float(S['n_zero_grad_units'][w].mean()),
                         dmbar_z_by20=prof[0], dmbar_z_by100=prof[1], dmbar_z_by300=prof[2], dmbar_z_by625=prof[3],
                         dep_share_by20=dprof[0], dep_share_by100=dprof[1], dep_share_by300=dprof[2],
                         sp_dm_m0=spearman(dm, m0), sp_dm_cnorm=spearman(dm, cn), sp_dm_fossil=spearman(dm, fo),
                         sp_dm_zbar0=spearman(dm, z0), sp_dm_pos0=spearman(dm, p0),
                         frac_units_sinking=float(np.mean(dm < 0)),
                         deep_zbar0=float(z0[deep].mean()), deep_dm_z=float(dm[deep].mean() * SBAR),
                         deep_gpos=float(gpu[deep].mean()), deep_gneg=float(gnu[deep].mean()),
                         shallow_zbar0=float(z0[shallow].mean()), shallow_dm_z=float(dm[shallow].mean() * SBAR),
                         shallow_gpos=float(gpu[shallow].mean()), shallow_gneg=float(gnu[shallow].mean()),
                         split_identity=float(U['split_identity'][uw].max()),
                         side_identity=float(S['side_identity'][w].max())))
    return rows


def main():
    rows = [r for a in ARMS if (OUT / f'{a}_s0_steps.csv').exists() for r in per_arm(a)]
    keys = list(rows[0])
    G.B.csvwrite(OUT / 'summary_table.csv', [{k: r.get(k) for k in keys} for r in rows])
    for r in rows:
        print(f"\n== {r['arm']} {r['window']}  (identities: split {r['split_identity']:.1e}, side {r['side_identity']:.1e})")
        print(f"  W~ per task: d||W~||^2 {r['dnorm2_task']:+.4f} = align(start) {r['align_task']:+.4f} + deposition {r['dep_task']:+.4f}   "
              f"[diag-only sum_s||dW~_s||^2 {r['dep_diag_only']:.4f}]  cos(W~_start, dW~_task) {r['cos_task']:+.3f}")
        print(f"  m  per task: d mbar {r['dmbar_task']:+.2e} (= {r['dmbar_task_z']:+.4f} in z)   d var(m) = 2cov {r['cov2_task']:+.2e} + var(dm) {r['vdm_task']:+.2e}")
        print(f"  Adam vs grad: row-direction energy share {r['fr_row_adam']:.3f} vs {r['fr_row_grad']:.3f};  |step| amplification row x{r['amp_row']:.0f}, centred x{r['amp_ctr']:.0f};  "
              f"sign agreement {r['sign_agree']:+.2f};  net drift / per-step row motion = {r['rectification']:.3f}")
        print(f"  brightness gradient per task: z>0 side {r['grow_pos']:+.2e}, z<=0 side {r['grow_neg']:+.2e}  (pos_frac {r['pos_frac']:.3f}; zero-grad units {r['zero_grad_units']:.1f})")
        print(f"  d mbar within task (z units, cumulative): step20 {r['dmbar_z_by20']:+.4f}  100 {r['dmbar_z_by100']:+.4f}  300 {r['dmbar_z_by300']:+.4f}  625 {r['dmbar_z_by625']:+.4f};  deposition share by 20/100/300: {r['dep_share_by20']:.2f}/{r['dep_share_by100']:.2f}/{r['dep_share_by300']:.2f}")
        print(f"  per-unit spearman(dm_i, .): m0 {r['sp_dm_m0']:+.2f}  ||W~|| {r['sp_dm_cnorm']:+.2f}  fossil {r['sp_dm_fossil']:+.2f}  zbar0 {r['sp_dm_zbar0']:+.2f}  pos0 {r['sp_dm_pos0']:+.2f};  units sinking {r['frac_units_sinking']:.2f}")
        print(f"  deepest 25% (zbar0 {r['deep_zbar0']:+.2f}): dm {r['deep_dm_z']:+.4f} z, g_row from z>0 {r['deep_gpos']:+.2e} / z<=0 {r['deep_gneg']:+.2e};   "
              f"shallowest 25% (zbar0 {r['shallow_zbar0']:+.2f}): dm {r['shallow_dm_z']:+.4f} z, g_row z>0 {r['shallow_gpos']:+.2e} / z<=0 {r['shallow_gneg']:+.2e}")



def figure():
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    COL = {'LR': '#0072B2', 'ELU1': '#E69F00', 'SNA': '#D55E00'}; LAB = {'LR': 'leaky 0.1', 'ELU1': 'ELU α=1', 'SNA': 'Snake'}
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.4))
    for arm in ARMS:
        S = load(OUT / f'{arm}_s0_steps.csv')
        for (lo, hi), ls in [((21, 25), '-'), ((96, 100), '--')]:
            w = (S['task'] >= lo) & (S['task'] <= hi)
            prof = np.mean([np.cumsum(S['dmbar'][w & (S['task'] == t)]) * SBAR for t in range(lo, hi + 1)], 0)
            ax[0].plot(np.arange(1, 626), prof, color=COL[arm], ls=ls, lw=1.6, label=f'{LAB[arm]} t{lo}–{hi}')
    ax[0].axhline(0, color='#888', lw=.7); ax[0].set_xscale('log'); ax[0].set_xlabel('update within task (log)')
    ax[0].set_ylabel('cumulative Δz̄ within the task'); ax[0].legend(fontsize=7, frameon=False)
    ax[0].set_title('Sinking = switch kick + in-task relaxation', loc='left', fontsize=9.5)
    k = 0; ticks = []
    for arm in ARMS:
        U = load(OUT / f'{arm}_s0_units.csv')
        for (lo, hi), alpha in [((21, 25), 1.), ((96, 100), .55)]:
            w = (U['task'] >= lo) & (U['task'] <= hi); n = hi - lo + 1
            dep = U['dep_task'][w].sum() / n / 100; al = U['align_task'][w].sum() / n / 100
            ax[1].bar(k, dep, color=COL[arm], alpha=alpha, width=.8); ax[1].bar(k, al, color='#555', alpha=alpha, width=.8)
            ax[1].plot([k - .4, k + .4], [dep + al] * 2, color='black', lw=2); ax[1].text(k, dep + .03, f'{dep + al:+.2f}', ha='center', fontsize=7.5)
            ticks.append(f'{LAB[arm]}\nt{lo}–{hi}'); k += 1
    ax[1].axhline(0, color='#888', lw=.7); ax[1].set_xticks(range(k)); ax[1].set_xticklabels(ticks, fontsize=7.5)
    ax[1].set_ylabel('per task, mean over units'); ax[1].set_title('Δ‖W̃‖² = deposition (colour) + erosion (grey); net = black', loc='left', fontsize=9.5)
    for arm in ARMS:
        U = load(OUT / f'{arm}_s0_units.csv'); w = (U['task'] >= 21) & (U['task'] <= 25)
        ax[2].scatter(U['zbar0'][w], U['dm'][w] * SBAR, s=7, color=COL[arm], alpha=.45, label=LAB[arm], linewidths=0)
    ax[2].axhline(0, color='#888', lw=.7); ax[2].axvline(0, color='#888', lw=.7)
    ax[2].set_xlabel('unit start position z̄ᵢ (t21–25)'); ax[2].set_ylabel('unit Δz̄ᵢ over the task'); ax[2].legend(fontsize=7.5, frameon=False)
    ax[2].set_title('Who sinks: units with mass above the kink', loc='left', fontsize=9.5)
    for a in ax:
        a.spines[['top', 'right']].set_visible(False); a.grid(axis='y', color='#eee', lw=.6)
    fig.tight_layout(); fig.savefig(OUT / 'fig_growth_engine.png', dpi=150)

if __name__ == '__main__':
    main(); figure()
