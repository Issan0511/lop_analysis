"""Verdicts for spec_unit_triage_0911: gate x update x contribution per unit (LR / ELU1 / SNA,
ref vs wclamp, t61-100)."""
from pathlib import Path
import csv, json
import numpy as np
from src import gate_shape_report_0911 as R

ROOT = R.ROOT
OUT = ROOT / 'results/unit_triage_0911'
ARMS = ['LR', 'ELU1', 'SNA']
SEEDS = R.SEEDS
LATE = (61, 100)
CORE = (81, 100)
TEST_TASKS = (61, 80, 100)
B3_KEYS = ['Gbar', 'Cov', 'S2', 'rho', 'kap2', 'graw2', 'P', 'Gini', 'w2col', 'hard_dead']


def gini(x):
    x = np.sort(np.abs(np.asarray(x, dtype=float)))
    n = len(x)
    if x.sum() == 0:
        return np.nan
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def participation(x):
    p = np.abs(np.asarray(x, dtype=float)); s = p.sum()
    if s == 0:
        return np.nan
    p = p / s; p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def per_seed(arm, seed):
    rows = R.read_rows(OUT / f'{arm}_none_s{seed}_rows.csv')
    units = np.load(OUT / f'{arm}_none_s{seed}_units.npz')
    prov = json.load(open(OUT / f'{arm}_none_s{seed}_provenance.json'))
    out = dict(arm=arm, seed=seed, g1=prov['checks']['g1_units_maxabs'], g1_n=prov['checks']['g1_units_compared'],
               g3=prov['checks'].get('g3_adam_recon'), g4=prov['checks'].get('g4_tot_ident'), rho_max=prov['checks'].get('g4_rho_max'))
    for c in ('ref', 'wclamp'):
        rS, rg, rk, rD, cores = [], [], [], [], []
        P, GI, dce_med = [], [], []
        for t in range(LATE[0], LATE[1] + 1):
            g = units[f'{c}_gbar_i_t{t}']; hard = units[f'{c}_hard_i_t{t}'].astype(bool)
            m = ~hard
            rS.append(R.spearman(g[m], units[f'{c}_S2_i_t{t}'][m])); rg.append(R.spearman(g[m], units[f'{c}_graw2_i_t{t}'][m]))
            rk.append(R.spearman(g[m], units[f'{c}_kap2_i_t{t}'][m])); rD.append(R.spearman(g[m], units[f'{c}_D2_i_t{t}'][m]))
            d = units[f'{c}_dce_i_t{t}']
            P.append(participation(d)); GI.append(gini(d)); dce_med.append(float(np.median(np.abs(d))))
        out[f'{c}_rho_S'] = float(np.nanmedian(rS)); out[f'{c}_rho_g'] = float(np.nanmedian(rg))
        out[f'{c}_rho_k'] = float(np.nanmedian(rk)); out[f'{c}_rho_D'] = float(np.nanmedian(rD))
        # core: bottom gate decile at every task of CORE
        core = None
        for t in range(CORE[0], CORE[1] + 1):
            b = set(R.bottom_decile(units[f'{c}_gbar_i_t{t}']).tolist())
            core = b if core is None else core & b
        out[f'{c}_n_core'] = len(core)
        if core:
            ratios = []
            for t in range(LATE[0], LATE[1] + 1):
                d = np.abs(units[f'{c}_dce_i_t{t}']); cm = np.zeros(100, dtype=bool); cm[list(core)] = True
                den = float(np.median(d[~cm]))
                ratios.append(float(np.median(d[cm])) / den if den > 0 else np.nan)
            out[f'{c}_core_ratio'] = float(np.nanmedian(ratios))
        else:
            out[f'{c}_core_ratio'] = np.nan
        # B3 population summaries (late medians)
        out[f'{c}_Gbar'] = R.win(rows, 'gbar', *LATE, clamp=c); out[f'{c}_Cov'] = R.win(rows, 'cov', *LATE, clamp=c)
        out[f'{c}_S2'] = R.win(rows, 'S2', *LATE, clamp=c); out[f'{c}_rho'] = R.win(rows, 'rho_mean', *LATE, clamp=c)
        out[f'{c}_kap2'] = R.win(rows, 'kap2', *LATE, clamp=c); out[f'{c}_graw2'] = R.win(rows, 'graw2', *LATE, clamp=c)
        out[f'{c}_P'] = float(np.nanmedian(P)); out[f'{c}_Gini'] = float(np.nanmedian(GI)); out[f'{c}_dce_med'] = float(np.median(dce_med))
        out[f'{c}_w2col'] = R.win(rows, 'w2col', *LATE, clamp=c); out[f'{c}_hard_dead'] = R.win(rows, 'hard_dead', *LATE, clamp=c)
        out[f'{c}_acc'] = R.win(rows, 'acc', *LATE, clamp=c); out[f'{c}_cnorm'] = R.win(rows, 'cnorm', *LATE, clamp=c)
        late_rows = [r for r in rows if LATE[0] <= r['task'] <= LATE[1] and r['clamp'] == c]
        out[f'{c}_g6'] = float(np.mean([r['ce20'] - r['ce_probe'] > 0 for r in late_rows]))
        out[f'{c}_n_frozen'] = R.win(rows, 'n_frozen', *LATE, clamp=c)
        # secondary: test dAcc vs probe dCE, readout proxy vs |dCE|
        cc, rr = [], []
        for t in TEST_TASKS:
            if f'{c}_dacc_i_t{t}' in units:
                cc.append(R.spearman(-units[f'{c}_dacc_i_t{t}'], units[f'{c}_dce_i_t{t}']))
        for t in range(LATE[0], LATE[1] + 1):
            rr.append(R.spearman(units[f'{c}_r_i_t{t}'], np.abs(units[f'{c}_dce_i_t{t}'])))
        out[f'{c}_corr_dacc_dce'] = float(np.nanmedian(cc)) if cc else np.nan; out[f'{c}_corr_r_dce'] = float(np.nanmedian(rr))
    out['L61'] = (R.win(rows, 'acc', *R.BASE, clamp='ref') - out['ref_acc']) * 100
    out['rho_w'] = (out['wclamp_acc'] - out['ref_acc']) * 100 / out['L61'] if out['L61'] > 0 else np.nan
    return out


def b1_label(rg, rS):
    if rg >= 0.5 and abs(rS) <= 0.2:
        return 'ADAM_COMPENSATES'
    if rg >= 0.5 and rS >= 0.5:
        return 'GATE_SETS_STEP'
    if rg >= 0.5 and rS <= -0.5:
        return 'GATE_INVERTS_STEP'
    if rg < 0.5:
        return 'GATE_NOT_EVEN_GRADIENT'
    return 'PARTIAL'


def main():
    seedrows, missing = [], []
    for arm in ARMS:
        for s in SEEDS:
            try:
                seedrows.append(per_seed(arm, s))
            except FileNotFoundError:
                missing.append(f'{arm}_s{s}')
    keys = list(seedrows[0].keys())
    with open(OUT / 'seed_verdict.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys, lineterminator='\n'); w.writeheader(); w.writerows(seedrows)
    arms = [a for a in ARMS if any(r['arm'] == a for r in seedrows)]
    verdict, lines = {}, ['# unit_triage_0911 summary\n', 'spec: `specs/spec_unit_triage_0911.md`（事前登録 commit `1388d10`・追補 1 `8e11d07`・判定値は未読で起動）\n']
    lines.append('## 0. G1 / G3 / G4 / G6\n\n| arm | g1 maxabs (s0/s1/s2) | arrays | G3 Adam recon | G4 ident | ρ max | G6 ref / wclamp |\n|---|---|---|---|---|---|---|')
    for a in arms:
        rs = [r for r in seedrows if r['arm'] == a]
        lines.append(f"| {a} | {' / '.join(str(r['g1']) for r in rs)} | {rs[0]['g1_n']} | {max(r['g3'] for r in rs):.2e} | {max(r['g4'] for r in rs):.1e} | {max(r['rho_max'] for r in rs):.1f} | {min(r['ref_g6'] for r in rs):.2f} / {min(r['wclamp_g6'] for r in rs):.2f} |")
    lines.append('\n## 1. B1 ゲート → 歩幅（ref・late 窓・hard-dead 除外・seed 別）\n\n| arm | ρ_g(ḡ, graw²) | ρ_S(ḡ, S²) | ρ_κ(ḡ, κ2) | ρ_D(ḡ, D²) | frozen | B1 |\n|---|---|---|---|---|---|---|')
    for a in arms:
        rs = [r for r in seedrows if r['arm'] == a]
        labs = [b1_label(r['ref_rho_g'], r['ref_rho_S']) for r in rs]
        lab = labs[0] if len(set(labs)) == 1 else 'MIXED'
        verdict[f'B1_{a}'] = lab
        lines.append(f"| {a} | {' / '.join(f'{r['ref_rho_g']:+.2f}' for r in rs)} | {' / '.join(f'{r['ref_rho_S']:+.2f}' for r in rs)} | {' / '.join(f'{r['ref_rho_k']:+.2f}' for r in rs)} | {' / '.join(f'{r['ref_rho_D']:+.2f}' for r in rs)} | {' / '.join(f'{r['ref_n_frozen']:.0f}' for r in rs)} | `{lab}` |")
    lines.append('\n## 2. B2 慢性の芯（ref・t81–100 のすべてで下位 10%）\n\n| arm | n_core (s0/s1/s2) | 貢献比 median|ΔCE| core / non-core | B2 |\n|---|---|---|---|')
    for a in arms:
        rs = [r for r in seedrows if r['arm'] == a]
        if all(r['ref_n_core'] == 0 for r in rs):
            lab = 'NO_CORE'
        else:
            ratios = [r['ref_core_ratio'] for r in rs if r['ref_n_core'] > 0]
            lab = 'CORE_INERT' if all(x <= 0.2 for x in ratios) else 'CORE_CONTRIBUTES' if all(x >= 0.8 for x in ratios) else 'CORE_PARTIAL'
        verdict[f'B2_{a}'] = lab
        lines.append(f"| {a} | {' / '.join(f'{r['ref_n_core']:.0f}' for r in rs)} | {' / '.join(f'{r['ref_core_ratio']:.2f}' for r in rs)} | `{lab}` |")
    lines.append('\n## 3. B3 wclamp で動く量（late 窓中央値・wclamp − ref・9 対）\n\n| 量 | ref (LR/ELU1/SNA) | wclamp (LR/ELU1/SNA) | 相対変化 中央値 | 同符号 9/9 | ラベル |\n|---|---|---|---:|---|---|')
    for k in B3_KEYS:
        rel, signs = [], []
        for r in seedrows:
            a, b = r[f'ref_{k}'], r[f'wclamp_{k}']
            if np.isfinite(a) and np.isfinite(b):
                rel.append((b - a) / abs(a) if a != 0 else (np.inf if b > 0 else -np.inf if b < 0 else 0.))
                signs.append(np.sign(b - a))
        mrel = float(np.median(rel)) if rel else np.nan
        same = len(set(signs)) == 1 and signs[0] != 0 and len(signs) == 9
        lab = (f'{k}_UP' if mrel > 0 else f'{k}_DOWN') if (same and abs(mrel) >= 0.10) else f'{k}_STATIC'
        verdict[f'B3_{k}'] = lab
        refv = ' / '.join(f"{np.median([r[f'ref_{k}'] for r in seedrows if r['arm'] == a]):.3g}" for a in arms)
        wv = ' / '.join(f"{np.median([r[f'wclamp_{k}'] for r in seedrows if r['arm'] == a]):.3g}" for a in arms)
        lines.append(f"| {k} | {refv} | {wv} | {mrel:+.2f} | {same} | `{lab}` |")
    lines.append('\n## 4. 副測定\n\n| arm | L61 | ρ_w | corr(−Δacc_test, ΔCE) | corr(r_i, |ΔCE|) ref | wclamp の ρ_S / ρ_g |\n|---|---:|---:|---:|---:|---|')
    for a in arms:
        rs = [r for r in seedrows if r['arm'] == a]
        lines.append(f"| {a} | {np.median([r['L61'] for r in rs]):.2f} | {np.median([r['rho_w'] for r in rs]):.2f} | {np.median([r['ref_corr_dacc_dce'] for r in rs]):+.2f} | {np.median([r['ref_corr_r_dce'] for r in rs]):+.2f} | {np.median([r['wclamp_rho_S'] for r in rs]):+.2f} / {np.median([r['wclamp_rho_g'] for r in rs]):+.2f} |")
    if missing:
        lines.append(f'\n**missing**: {missing}')
    (OUT / 'summary.md').write_text('\n'.join(lines) + '\n')
    with open(OUT / 'verdict.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(verdict.keys()), lineterminator='\n'); w.writeheader(); w.writerow(verdict)
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(len(arms), 3, figsize=(12, 3.6 * len(arms)))
        axes = np.atleast_2d(axes)
        for i, a in enumerate(arms):
            u = np.load(OUT / f'{a}_none_s0_units.npz'); t = 100
            g = u[f'ref_gbar_i_t{t}']; hard = u[f'ref_hard_i_t{t}'].astype(bool)
            for j, (k, lab) in enumerate((('S2_i', 'S2 (step budget)'), ('graw2_i', 'raw grad^2'), ('dce_i', '|dCE| (ablation)'))):
                v = np.abs(u[f'ref_{k}_t{t}']); axes[i, j].scatter(g[~hard], v[~hard], s=12); axes[i, j].scatter(g[hard], v[hard], s=12, c='r', label='hard-dead')
                axes[i, j].set_xlabel('gbar_i'); axes[i, j].set_ylabel(lab); axes[i, j].set_yscale('log'); axes[i, j].set_title(f'{a} seed0 t{t}')
        fig.tight_layout(); fig.savefig(OUT / 'fig_triage.png', dpi=110)
    except Exception as e:
        print('figure skipped:', e)
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
