"""Verdicts for spec_tau_clamp_0912 (N0-N3).  Refuses an incomplete run unless --partial."""
from pathlib import Path
import argparse, csv, json
import numpy as np
from src import gate_shape_report_0911 as R

ROOT = R.ROOT
OUT = ROOT / 'results/tau_clamp_0912'
ET = ROOT / 'results/elu_turn_0912'          # tau = 1 is carried in from there
ARMS = ['T03', 'T03w', 'T3', 'T3w']
SEEDS = [0, 1, 2]
BASE = (16, 20)
LATE = (101, 120)
G3_CE_FRAC = 0.90
# (tau, unclamped tag, clamped tag, directory)
FAM = [(0.3, 'T03', 'T03w', 'OUT'), (1.0, 'CELU1', 'CELU1w', 'ET'), (3.0, 'T3', 'T3w', 'OUT')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=None); ap.add_argument('--partial', action='store_true')
    a = ap.parse_args()
    global OUT
    if a.src:
        OUT = Path(a.src)
    have = {f.name.replace('_rows.csv', '') for f in OUT.glob('*_rows.csv')}
    gap = sorted({f'{x}_s{s}' for x in ARMS for s in SEEDS} - have)
    if gap and not a.partial:
        print(f'REFUSING: {len(gap)} runs missing {gap[:6]}'); raise SystemExit(2)

    D = {}
    for d, names in ((OUT, ARMS), (ET, ['CELU1', 'CELU1w'])):
        for x in names:
            for s in SEEDS:
                try:
                    D[(x, s)] = (R.read_rows(d / f'{x}_s{s}_rows.csv'), json.load(open(d / f'{x}_s{s}_provenance.json')))
                except FileNotFoundError:
                    pass
    present = [x for x in ARMS + ['CELU1', 'CELU1w'] if any(k[0] == x for k in D)]

    def w(x, s, key, lo=LATE[0], hi=LATE[1]):
        v = [r[key] for r in D[(x, s)][0] if lo <= r['task'] <= hi and key in r and np.isfinite(r[key])]
        return float(np.median(v)) if v else np.nan

    def L(x, s):
        return (w(x, s, 'acc', *BASE) - w(x, s, 'acc')) * 100

    def med(x, k):
        return float(np.median([w(x, s, k) for s in SEEDS if (x, s) in D]))

    def mL(x):
        return float(np.median([L(x, s) for s in SEEDS if (x, s) in D]))

    broken = [x for x in present if min(float(np.mean([r['ce20'] - r['ce0'] < 0 for r in D[(x, s)][0]
                                                       if LATE[0] <= r['task'] <= LATE[1]])) for s in SEEDS if (x, s) in D) < G3_CE_FRAC]
    ok = lambda x: all((x, s) in D for s in SEEDS) and x not in broken

    V, lines = {}, ['# tau_clamp_0912 summary\n',
                    'spec: `specs/spec_tau_clamp_0912.md`（事前登録 `dad00cd`・実装 `daa7a04`・判定値は未読で起動）\n',
                    f'scope: 新規 4 腕 × 3 seed ＋ τ=1 を `elu_turn_0912` から持ち込み・base t{BASE[0]}–{BASE[1]}・late t{LATE[0]}–{LATE[1]}\n']

    # ---- N0
    lines.append('\n## 0. 検査\n\n| arm | τ | clamp | G1 錨 | G1 maxabs | 配列数 | 幅の釘付け | 角度 | 三角 | クランプ | CE 改善率 |\n|---|---:|---|---|---|---:|---:|---:|---:|---|---:|')
    g1_ok, pin_ok = True, True
    for x in present:
        ck = D[(x, 0)][1]['checks']
        g1s = [D[(x, s)][1]['checks']['g1_units_maxabs'] for s in SEEDS if (x, s) in D]
        g1_ok = g1_ok and all(v is not None and v <= 1e-10 for v in g1s)
        pin = max(D[(x, s)][1]['checks'].get('n0_cnorm_rel', 0.) for s in SEEDS if (x, s) in D)
        if ck['clamp'] != 'ref' and x in ARMS:
            pin_ok = pin_ok and pin <= 1e-6
        cef = min(float(np.mean([r['ce20'] - r['ce0'] < 0 for r in D[(x, s)][0]
                                 if LATE[0] <= r['task'] <= LATE[1]])) for s in SEEDS if (x, s) in D)
        lines.append(f"| {x} | {ck.get('tau', '—')} | {ck['clamp']} | {ck['g1_anchor']} | {' / '.join(f'{v:g}' for v in g1s)} | "
                     f"{ck['g1_units_compared']} | {pin:.1e} | {max(D[(x, s)][1]['checks'].get('g2_angle_rel', 0) for s in SEEDS if (x, s) in D):.1e} | "
                     f"{max(D[(x, s)][1]['checks']['g2_triangle'] for s in SEEDS if (x, s) in D):.1e} | "
                     f"{('c3 ' + format(ck['c3_cnorm_rel'], '.1e')) if 'c3_cnorm_rel' in ck else '—'} | {cef:.2f} |")
    if broken:
        lines.append(f"\n**LEARNING_BROKEN**: {broken}")
    V['N0'] = 'MANIPULATION_OK' if (g1_ok and pin_ok and not broken) else 'NOT_TESTABLE'
    V['N0_parts'] = f'G1={g1_ok} pinned={pin_ok} broken={broken}'
    lines.append(f"\n- **N0**: {V['N0_parts']} → `{V['N0']}`")
    if ok('T03') and ok('T3'):
        dl = [L('T03', s) - L('T3', s) for s in SEEDS]
        V['testability_dL'] = [round(x, 2) for x in dl]; V['testability_met'] = bool(all(x >= 1.0 for x in dl))
        lines.append(f"- **可検定性（N0 と独立）**: L(T03) − L(T3) = {V['testability_dL']} pt、3/3 は {V['testability_met']}")

    # ---- the family table
    lines.append('\n## 1. τ × クランプ（seed 中央値）\n\n| τ | L(無クランプ) | **L(wclamp) = S** | **R_c = 消えた分** | level(wclamp) | ‖W̃ᵢ‖ ref → clamp | ω ref → clamp | 死んだユニット ref / clamp |\n|---:|---:|---:|---:|---:|---|---|---|')
    S, Rc = {}, {}
    for tau, u, c, _ in FAM:
        if not (ok(u) and ok(c)):
            continue
        S[tau] = mL(c); Rc[tau] = mL(u) - mL(c)
        lines.append(f"| {tau} | {mL(u):.2f} | **{S[tau]:.2f}** | **{Rc[tau]:.2f}** | {med(c, 'acc'):.4f} | "
                     f"{med(u, 'cnorm'):.2f} → {med(c, 'cnorm'):.2f} | {med(u, 'omega_step'):.2e} → {med(c, 'omega_step'):.2e} | "
                     f"{med(u, 'hard_dead'):.0f} / {med(c, 'hard_dead'):.0f} |")
    V['S'] = {str(k): round(v, 2) for k, v in S.items()}; V['R_c'] = {str(k): round(v, 2) for k, v in Rc.items()}

    if V['N0'] == 'MANIPULATION_OK' and ok('T03w'):
        s03 = [L('T03w', s) for s in SEEDS]
        V['N1_S03'] = [round(x, 2) for x in s03]
        V['N1'] = ('SATURATION_VIA_WIDTH' if all(x <= .5 for x in s03)
                   else 'SATURATION_INDEPENDENT' if all(x >= 1.5 for x in s03) else 'N1_PARTIAL')
        lines.append(f"\n- **N1（決定的）**: S(0.3) = L(T03w) = {V['N1_S03']} pt → `{V['N1']}`")
        if ok('T3w'):
            ds = [L('T03w', s) - L('T3w', s) for s in SEEDS]
            V['N2_dS'] = [round(x, 2) for x in ds]
            V['N2'] = ('TAU_GONE_UNDER_CLAMP' if all(abs(x) < .3 for x in ds)
                       else 'TAU_SURVIVES_CLAMP' if all(x >= 1.0 for x in ds) else 'N2_PARTIAL')
            lines.append(f"- **N2**: ΔS = S(0.3) − S(3) = {V['N2_dS']} pt（S(1) = {S.get(1.0, float('nan')):.2f} を併記） → `{V['N2']}`")
        if len(Rc) == 3:
            seq = [Rc[t] for t in (0.3, 1.0, 3.0)]
            V['N3_Rc_seq'] = [round(x, 2) for x in seq]
            V['N3'] = ('RC_RISES_WITH_BAND' if seq[0] < seq[1] < seq[2]
                       else 'RC_FALLS_WITH_BAND' if seq[0] > seq[1] > seq[2] else 'RC_FLAT_OR_MIXED')
            lines.append(f"- **N3**: R_c(τ) = {V['N3_Rc_seq']}（τ = 0.3 / 1 / 3） → `{V['N3']}`")
    else:
        lines.append('\n**N1–N3 は N0 のゲートで `NOT_TESTABLE`。**')

    (OUT / 'summary.md').write_text('\n'.join(lines) + '\n')
    with open(OUT / 'verdict.csv', 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(V.keys()), lineterminator='\n')
        wr.writeheader(); wr.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in V.items()})
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        taus = [t for t, u, c, _ in FAM if ok(u) and ok(c)]
        ax.plot(taus, [mL(dict((t, u) for t, u, c, _ in FAM)[t]) for t in taus], 'o-', label='no clamp')
        ax.plot(taus, [S[t] for t in taus], 's-', label='width clamped')
        ax.set_xscale('log'); ax.set_xlabel('tau  (negative-side band)'); ax.set_ylabel('L [pt]')
        ax.axhline(0, color='k', lw=.5); ax.legend()
        fig.tight_layout(); fig.savefig(OUT / 'fig_tau_clamp.png', dpi=120)
    except Exception as e:
        print('figure skipped:', e)
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
