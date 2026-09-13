"""Verdicts for spec_band_dial_0911: CELU tau = 0.3 / 1 / 3 and the floored ELU (ELUF)."""
from pathlib import Path
import csv, json
import numpy as np
from src import gate_shape_report_0911 as R

ROOT = R.ROOT
SRC = ROOT / 'results/gate_shape_0911'
OUT = ROOT / 'results/band_dial_0911'
ARMS = ['CELU03', 'CELU1', 'CELU3', 'ELUF']
SEEDS = R.SEEDS
LATE = R.LATE


def per_seed(arm, seed):
    rows = R.read_rows(SRC / f'{arm}_s{seed}_rows.csv')
    prov = json.load(open(SRC / f'{arm}_s{seed}_provenance.json'))
    b = R.win(rows, 'acc', *R.BASE); l = R.win(rows, 'acc', *LATE)
    return dict(arm=arm, seed=seed, L=(b - l) * 100, acc_base=b, acc_late=l, Cov=R.win(rows, 'cov', *LATE),
                Cov10=R.win(rows, 'cov10', *LATE), Cov50=R.win(rows, 'cov50', *LATE), NL_x=R.win(rows, 'gvar_mean', *LATE),
                NL_u=R.win(rows, 'gbar_var', *LATE), N=R.win(rows, 'cnorm', *LATE), zbar=R.win(rows, 'zbar_inv', *LATE),
                sigma=R.win(rows, 'sigma_inv', *LATE), pos=R.win(rows, 'pos_frac', *LATE), hard_dead=R.win(rows, 'hard_dead', *LATE),
                Gbar=R.win(rows, 'gbar', *LATE), g1=prov['checks'].get('g1_units_maxabs'), g1_cnorm=prov['checks'].get('g1_cnorm_maxabs'),
                g2=prov['checks'].get('g2_dphi'), cont=prov['checks'].get('g2_eluf_continuity'))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
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

    def get(arm, s, k):
        return next((r[k] for r in seedrows if r['arm'] == arm and r['seed'] == s), np.nan)
    verdict, lines = {}, ['# band_dial_0911 summary\n', 'spec: `specs/spec_band_dial_0911.md`（事前登録 commit `1388d10`・判定値は未読で起動）\n']
    arms = [a for a in ARMS if any(r['arm'] == a for r in seedrows)]
    lines.append('## 0. G1 / G2\n\n| arm | g1 maxabs (s0/s1/s2) | cnorm | φ′ vs autograd | ELUF continuity |\n|---|---|---|---|---|')
    for a in arms:
        rs = [r for r in seedrows if r['arm'] == a]
        lines.append(f"| {a} | {' / '.join(str(r['g1']) for r in rs)} | {' / '.join(str(r['g1_cnorm']) for r in rs)} | {rs[0]['g2']} | {rs[0]['cont']} |")
    lines.append('\n## 1. 腕ごとの量（seed 別 L と late 窓の中央値）\n\n| arm | L s0/s1/s2 [pt] | L med | Cov | Cov θ=0.5 | NL_x | NL_u | Ḡ | N | z̄ | σ | p⁺ | dead |\n|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    med = {}
    for a in arms:
        rs = [r for r in seedrows if r['arm'] == a]
        m = {k: float(np.median([r[k] for r in rs])) for k in keys if isinstance(rs[0][k], (int, float)) and k != 'seed'}
        med[a] = m
        lines.append(f"| {a} | {' / '.join(f'{r['L']:.2f}' for r in rs)} | {m['L']:.2f} | {m['Cov']:.3f} | {m['Cov50']:.3f} | {m['NL_x']:.4f} | {m['NL_u']:.4f} | {m['Gbar']:.3f} | {m['N']:.2f} | {m['zbar']:.2f} | {m['sigma']:.2f} | {m['pos']:.3f} | {m['hard_dead']:.0f} |")
    have = all(a in arms for a in ('CELU03', 'CELU1', 'CELU3'))
    if have:
        d0 = all(get('CELU3', s, 'Cov') < get('CELU1', s, 'Cov') < get('CELU03', s, 'Cov') for s in SEEDS)
        verdict['D0'] = 'MANIPULATION_OK' if d0 else 'NOT_TESTABLE_NO_COVERAGE_CHANGE'
        less = all(get('CELU03', s, 'L') > get('CELU1', s, 'L') > get('CELU3', s, 'L') for s in SEEDS)
        more = all(get('CELU03', s, 'L') < get('CELU1', s, 'L') < get('CELU3', s, 'L') for s in SEEDS)
        d1 = 'WIDER_BAND_LESS_LOSS' if less else 'WIDER_BAND_MORE_LOSS' if more else 'BAND_NO_MONOTONE'
        eff = float(np.median([abs(get('CELU3', s, 'L') - get('CELU03', s, 'L')) for s in SEEDS]))
        verdict['D1'] = d1 if d0 else 'NOT_TESTABLE_NO_COVERAGE_CHANGE'; verdict['D1_effect_pt'] = eff
        verdict['D1_size'] = 'BAND_WEAK' if eff < 1.0 else 'BAND_STRONG'
        dN = {a: float(np.median([get(a, s, 'N') / get('CELU1', s, 'N') - 1 for s in SEEDS])) for a in arms if a != 'CELU1'}
        verdict['D3_N_rel'] = json.dumps({k: round(v, 3) for k, v in dN.items()})
        verdict['D3'] = 'WIDTH_MOVES_TOO' if any(abs(v) >= 0.10 for v in dN.values()) else 'WIDTH_UNCHANGED'
        lines.append(f"\n- **D0**: Cov(CELU3) < Cov(CELU1) < Cov(CELU03) in 3/3 seeds: {d0} → `{verdict['D0']}`")
        lines.append(f"- **D1**: `{verdict['D1']}`、効果量 |L(CELU3) − L(CELU03)| 中央値 = {eff:.2f} pt → `{verdict['D1_size']}`")
        lines.append(f"- **D3**: N の CELU1 に対する相対差 {verdict['D3_N_rel']} → `{verdict['D3']}`")
    if 'ELUF' in arms and 'CELU1' in arms:
        dd = [get('ELUF', s, 'L') - get('CELU1', s, 'L') for s in SEEDS]
        verdict['D2'] = 'FLOOR_HELPS' if all(x <= -0.5 for x in dd) else 'FLOOR_HURTS' if all(x >= 0.5 for x in dd) else 'FLOOR_NEUTRAL'
        verdict['D2_dL'] = json.dumps([round(x, 2) for x in dd])
        lines.append(f"- **D2**: ΔL(ELUF − CELU1) = {[round(x, 2) for x in dd]} pt → `{verdict['D2']}`")
    if missing:
        lines.append(f'\n**missing**: {missing}')
    (OUT / 'summary.md').write_text('\n'.join(lines) + '\n')
    with open(OUT / 'verdict.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(verdict.keys()), lineterminator='\n'); w.writeheader(); w.writerow(verdict)
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        for a in arms:
            for s in SEEDS:
                rows = R.read_rows(SRC / f'{a}_s{s}_rows.csv')
                t = [r['task'] for r in rows]; acc = [r['acc'] for r in rows]
                ax[0].plot(t, acc, lw=0.8, alpha=0.7, label=a if s == 0 else None)
            ax[1].scatter(med[a]['Cov'], med[a]['L']); ax[1].annotate(a, (med[a]['Cov'], med[a]['L']), fontsize=8, xytext=(3, 3), textcoords='offset points')
        ax[0].set_xlabel('task'); ax[0].set_ylabel('acc'); ax[0].legend(fontsize=7)
        ax[1].set_xlabel('Cov (late)'); ax[1].set_ylabel('L [pt]')
        fig.tight_layout(); fig.savefig(OUT / 'fig_band_dial.png', dpi=120)
    except Exception as e:
        print('figure skipped:', e)
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
