"""Verdicts for spec_gate_shape_0911 (candidates for c_act vs loss L and residual loss R_c).

Shared helpers (rankdata / spearman / row readers) are imported by the sister reports
gate_persistence_report_0911, band_dial_report_0911 and unit_triage_report_0911.
"""
from pathlib import Path
import csv, json
import numpy as np
from src import boundary_gradient_0908 as G

ROOT = G.ROOT
OUT = ROOT / 'results/gate_shape_0911'
ARMS = ['R', 'LR03', 'LR', 'LR001', 'ELU1', 'ELU03', 'SN02', 'SN06', 'SN15', 'SNA', 'LIN', 'GELU', 'SILU']
SEEDS = [0, 1, 2]
BASE = (16, 20)
LATE = (101, 120)
LATE61 = (61, 100)
CANDS = ['Cov', 'Cov_chronic', 'NL_x', 'NL_u', 'gcorr']
HI, LO = 0.6, 0.2
WIN_MARGIN = 0.2
# where each arm's committed / new wclamp continuation lives (ref + wclamp rows, step 625)
WCLAMP_SRC = {'LR': 'width_sink_clamp_0909', 'ELU1': 'width_sink_clamp_0909', 'SNA': 'width_sink_clamp_0909',
              'R': 'clamp_horizon_acts_0910', 'GELU': 'clamp_horizon_acts_0910', 'SILU': 'clamp_horizon_acts_0910',
              'SN02': 'gate_wclamp_0911', 'SN06': 'gate_wclamp_0911', 'SN15': 'gate_wclamp_0911',
              'LR03': 'gate_wclamp_0911', 'LR001': 'gate_wclamp_0911'}


# ------------------------------------------------------------------ helpers
def rankdata(x):
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind='stable')
    ranks = np.empty(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2. + 1.
        i = j + 1
    return ranks


def spearman(x, y):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float('nan')
    rx, ry = rankdata(x[ok]), rankdata(y[ok])
    rx -= rx.mean(); ry -= ry.mean()
    den = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / den) if den > 0 else float('nan')


def band(rho):
    if not np.isfinite(rho):
        return 'NOT_TESTABLE'
    if rho >= HI:
        return 'ORDERS'
    if rho > LO:
        return 'PARTIAL'
    if rho >= -LO:
        return 'FAILS'
    if rho > -HI:
        return 'PARTIAL_INVERTED'
    return 'INVERTED'


def read_rows(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        for k, v in list(r.items()):
            if k in ('arm', 'iv', 'clamp', 'kind'):
                continue
            try:
                r[k] = float(v) if v not in ('', None, 'None') else np.nan
            except ValueError:
                pass
    return rows


def win(rows, key, lo, hi, f=np.median, clamp='ref', step=625):
    v = [r[key] for r in rows if lo <= r['task'] <= hi and r.get('clamp', 'ref') == clamp
         and (('step' not in r) or r['step'] == step) and np.isfinite(r[key])]
    return float(f(v)) if v else np.nan


def bottom_decile(g):
    return np.argsort(g, kind='stable')[:10]


def chronic_units(units, lo, hi, prefix='ref'):
    """Units in the bottom gate decile at EVERY task of [lo, hi]."""
    sets = None
    for t in range(lo, hi + 1):
        key = f'{prefix}_gbar_i_t{t}'
        if key not in units:
            return None
        b = set(bottom_decile(units[key]).tolist())
        sets = b if sets is None else (sets & b)
    return sets


# ------------------------------------------------------------------ per arm / seed
def per_seed(arm, seed):
    rows = read_rows(OUT / f'{arm}_s{seed}_rows.csv')
    units = np.load(OUT / f'{arm}_s{seed}_units.npz')
    prov = json.load(open(OUT / f'{arm}_s{seed}_provenance.json'))
    acc_base = win(rows, 'acc', *BASE)
    acc_late = win(rows, 'acc', *LATE)
    out = dict(arm=arm, seed=seed, acc_base=acc_base, acc_late=acc_late, L=(acc_base - acc_late) * 100,
               Cov=win(rows, 'cov', *LATE), NL_x=win(rows, 'gvar_mean', *LATE), NL_u=win(rows, 'gbar_var', *LATE),
               gcorr=win(rows, 'gcorr', *LATE), Gbar=win(rows, 'gbar', *LATE), N=win(rows, 'cnorm', *LATE),
               Cov10=win(rows, 'cov10', *LATE), Cov50=win(rows, 'cov50', *LATE), Cov_abs=win(rows, 'cov_abs', *LATE),
               zbar=win(rows, 'zbar_inv', *LATE), sigma=win(rows, 'sigma_inv', *LATE), pos=win(rows, 'pos_frac', *LATE),
               hard_dead=win(rows, 'hard_dead', *LATE), g1=prov['checks'].get('g1_units_maxabs'),
               g1_cnorm=prov['checks'].get('g1_cnorm_maxabs'), g1_n=prov['checks'].get('g1_units_compared'))
    ch = chronic_units(units, *LATE)
    if ch is None:
        out['Cov_chronic'] = np.nan; out['n_chronic'] = np.nan
    else:
        offs = [np.median([units[f'ref_off_i_t{t}'][i] for t in range(LATE[0], LATE[1] + 1)]) for i in sorted(ch)]
        out['Cov_chronic'] = float(sum(offs) / 100.) if offs else 0.
        out['n_chronic'] = len(ch)
    # residual loss under wclamp (window t61-100), from the committed / new continuation
    src = WCLAMP_SRC.get(arm)
    out['R_c'] = np.nan; out['L61'] = np.nan; out['rho_w'] = np.nan; out['R_c_src'] = src or ''
    if src is not None:
        p = ROOT / 'results' / src / f'{arm}_none_s{seed}_rows.csv'
        if p.exists():
            wr = read_rows(p)
            b = win(wr, 'acc', *BASE, clamp='ref')
            a_w = win(wr, 'acc', *LATE61, clamp='wclamp')
            a_r = win(wr, 'acc', *LATE61, clamp='ref')
            out['R_c'] = (b - a_w) * 100; out['L61'] = (b - a_r) * 100
            out['rho_w'] = 1 - out['R_c'] / out['L61'] if out['L61'] > 0 else np.nan
            out['acc_base_wsrc'] = b
    return out


def main():
    seedrows, missing = [], []
    for arm in ARMS:
        for s in SEEDS:
            try:
                seedrows.append(per_seed(arm, s))
            except FileNotFoundError as e:
                missing.append(f'{arm}_s{s}')
    if missing:
        print('missing:', missing)
    OUT.mkdir(parents=True, exist_ok=True)
    keys = list(seedrows[0].keys())
    with open(OUT / 'seed_verdict.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys, lineterminator='\n', restval=''); w.writeheader(); w.writerows(seedrows)
    arms = [a for a in ARMS if any(r['arm'] == a for r in seedrows)]
    numeric = [k for k in keys if k not in ('arm', 'seed', 'R_c_src')]
    med = {a: {k: float(np.nanmedian([float(r.get(k, np.nan)) if r.get(k) is not None else np.nan for r in seedrows if r['arm'] == a]))
               for k in numeric}
           for a in arms}
    verdict, lines = {}, []
    lines.append('# gate_shape_0911 summary\n')
    lines.append(f'spec: `specs/spec_gate_shape_0911.md`（事前登録 commit `1388d10`・追補 1 `8e11d07`・判定値は未読で起動）\n')
    lines.append(f'scope: {len(arms)} 腕 × 3 seed・t1–120・base t{BASE[0]}–{BASE[1]}・late t{LATE[0]}–{LATE[1]}・R_c は t{LATE61[0]}–{LATE61[1]}\n')
    # G1 table
    lines.append('\n## 0. G1\n\n| arm | g1 maxabs (seed 0/1/2) | cnorm maxabs | arrays |\n|---|---|---|---|')
    for a in arms:
        rs = [r for r in seedrows if r['arm'] == a]
        lines.append(f"| {a} | {' / '.join(str(r['g1']) for r in rs)} | {' / '.join(str(r['g1_cnorm']) for r in rs)} | {rs[0]['g1_n']} |")
    # candidate table
    lines.append('\n## 1. 腕ごとの量（seed 中央値・late 窓）\n')
    lines.append('| arm | L [pt] | R_c [pt] | L61 | ρ_w | Cov | Cov_chronic (n) | NL_x | NL_u | gcorr | Ḡ | N | z̄ | p⁺ | dead |')
    lines.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for a in arms:
        m = med[a]
        lines.append(f"| {a} | {m['L']:.2f} | {m['R_c']:.2f} | {m['L61']:.2f} | {m['rho_w']:.2f} | {m['Cov']:.3f} | {m['Cov_chronic']:.3f} ({m['n_chronic']:.0f}) | "
                     f"{m['NL_x']:.4f} | {m['NL_u']:.4f} | {m['gcorr']:.3f} | {m['Gbar']:.3f} | {m['N']:.2f} | {m['zbar']:.2f} | {m['pos']:.3f} | {m['hard_dead']:.0f} |")
    # A1-A3: Spearman of each candidate vs L (all arms) and vs R_c (arms with wclamp)
    lines.append('\n## 2. 事前登録の判定\n')
    lines.append('| 候補 Q | ρ(Q, L) n | ラベル | ρ(Q, R_c) n | ラベル | seed 別 ρ(Q, L) |')
    lines.append('|---|---:|---|---:|---|---|')
    Lv = np.array([med[a]['L'] for a in arms]); Rv = np.array([med[a]['R_c'] for a in arms])
    for q in CANDS:
        Qv = np.array([med[a][q] for a in arms])
        rL = spearman(Qv, Lv); rR = spearman(Qv, Rv)
        nL = int((np.isfinite(Qv) & np.isfinite(Lv)).sum()); nR = int((np.isfinite(Qv) & np.isfinite(Rv)).sum())
        per = []
        for s in SEEDS:
            qa = [next((r[q] for r in seedrows if r['arm'] == a and r['seed'] == s), np.nan) for a in arms]
            la = [next((r['L'] for r in seedrows if r['arm'] == a and r['seed'] == s), np.nan) for a in arms]
            per.append(spearman(qa, la))
        verdict[f'{q}_vs_L'] = f'{q}_{band(rL)}_L'; verdict[f'rho_{q}_L'] = rL
        verdict[f'{q}_vs_Rc'] = f'{q}_{band(rR)}_Rc'; verdict[f'rho_{q}_Rc'] = rR
        lines.append(f"| {q} | {rL:+.2f} ({nL}) | `{verdict[f'{q}_vs_L']}` | {rR:+.2f} ({nR}) | `{verdict[f'{q}_vs_Rc']}` | {' / '.join(f'{p:+.2f}' for p in per)} |")
    # A4 winner on R_c
    absr = sorted(((abs(verdict[f'rho_{q}_Rc']), q) for q in CANDS if np.isfinite(verdict[f'rho_{q}_Rc'])), reverse=True)
    if len(absr) >= 2 and absr[0][0] - absr[1][0] >= WIN_MARGIN:
        verdict['A4'] = f'WINNER_{absr[0][1]}'
    else:
        verdict['A4'] = 'NO_WINNER'
    # A5 pair SN02 vs SN06 per seed
    pair = []
    if 'SN02' in arms and 'SN06' in arms:
        for q in CANDS:
            ok = True
            for s in SEEDS:
                r2 = next(r for r in seedrows if r['arm'] == 'SN02' and r['seed'] == s)
                r6 = next(r for r in seedrows if r['arm'] == 'SN06' and r['seed'] == s)
                dL = r6['L'] - r2['L']; dQ = r6[q] - r2[q]
                if not (np.isfinite(dL) and np.isfinite(dQ)) or dL == 0 or np.sign(dQ) != np.sign(dL):
                    ok = False
            if ok:
                pair.append(q)
        verdict['A5'] = 'PAIR_' + ('+'.join(pair) if pair else 'NONE')
        dLs = [next(r for r in seedrows if r['arm'] == 'SN06' and r['seed'] == s)['L'] - next(r for r in seedrows if r['arm'] == 'SN02' and r['seed'] == s)['L'] for s in SEEDS]
        verdict['A5_dL'] = dLs
    else:
        verdict['A5'] = 'NOT_TESTABLE'
    # A6 width vs L
    Nv = np.array([med[a]['N'] for a in arms])
    verdict['rho_N_L'] = spearman(Nv, Lv); verdict['rho_N_Rc'] = spearman(Nv, Rv)
    verdict['A6'] = 'WIDTH_ORDERS_BETWEEN_ARMS' if verdict['rho_N_L'] > 0.3 else 'WIDTH_NOT_BETWEEN_ARMS'
    lines.append(f"\n- **A4**: `{verdict['A4']}`（R_c に対する |ρ| の順: {', '.join(f'{q} {v:.2f}' for v, q in absr)}）")
    lines.append(f"- **A5**（SN02 対 SN06・seed 対）: `{verdict['A5']}`；ΔL(SN06−SN02) = {verdict.get('A5_dL')}")
    lines.append(f"- **A6**: Spearman(N, L) = {verdict['rho_N_L']:+.2f}、Spearman(N, R_c) = {verdict['rho_N_Rc']:+.2f} → `{verdict['A6']}`")
    # secondary thresholds
    lines.append('\n## 3. 副測定\n\n| arm | Cov θ=0.1 | Cov θ=0.5 | Cov |φ′| | σ_inv |\n|---|---:|---:|---:|---:|')
    for a in arms:
        m = med[a]
        lines.append(f"| {a} | {m['Cov10']:.3f} | {m['Cov50']:.3f} | {m['Cov_abs']:.3f} | {m['sigma']:.2f} |")
    for q in ('Cov10', 'Cov50', 'Cov_abs', 'Gbar'):
        verdict[f'rho_{q}_L'] = spearman([med[a][q] for a in arms], Lv)
    lines.append(f"\nSpearman vs L: Cov10 {verdict['rho_Cov10_L']:+.2f}, Cov50 {verdict['rho_Cov50_L']:+.2f}, Cov_abs {verdict['rho_Cov_abs_L']:+.2f}, Ḡ {verdict['rho_Gbar_L']:+.2f}")
    if missing:
        lines.append(f"\n**missing runs**: {missing}")
    (OUT / 'summary.md').write_text('\n'.join(lines) + '\n')
    with open(OUT / 'verdict.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(verdict.keys()), lineterminator='\n'); w.writeheader(); w.writerow(verdict)
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
        for ax, q in zip(axes.flat, CANDS + ['N']):
            for a in arms:
                ax.scatter(med[a][q], med[a]['L'], s=30)
                ax.annotate(a, (med[a][q], med[a]['L']), fontsize=8, xytext=(3, 3), textcoords='offset points')
            r = verdict.get(f'rho_{q}_L', verdict.get('rho_N_L'))
            ax.set_xlabel(q); ax.set_ylabel('L [pt] (t16-20 minus t101-120)'); ax.set_title(f'{q}: Spearman {r:+.2f}')
        fig.tight_layout(); fig.savefig(OUT / 'fig_gate_shape.png', dpi=120)
    except Exception as e:
        print('figure skipped:', e)
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
