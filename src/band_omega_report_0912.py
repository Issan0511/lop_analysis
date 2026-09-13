"""Verdicts for spec_band_omega_0912 (Q0-Q4).  Refuses an incomplete run unless --partial."""
from pathlib import Path
import argparse, csv, json
import numpy as np
from src import gate_shape_report_0911 as R

ROOT = R.ROOT
OUT = ROOT / 'results/band_omega_0912'
ET = ROOT / 'results/elu_turn_0912'        # E1 = CELU1w, E0 = CELU1 carried in
NEW = ['E1h', 'E2', 'E1d', 'E05']
CLAMPED = ['E1', 'E1h', 'E2', 'E1d', 'E05']
PAIRS = (('E1h', 'E2'), ('E1d', 'E05'))
SEEDS = [0, 1, 2]
BASE = (16, 20)
LATE = (101, 120)
B_LEAKY = -1.06        # pt per octave of omega, from turn_rate_0912 (carried in, fixed)
G3_CE_FRAC = 0.90
SRC = {'E1': (ET, 'CELU1w'), 'E0': (ET, 'CELU1')}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=None); ap.add_argument('--partial', action='store_true')
    a = ap.parse_args()
    global OUT
    if a.src:
        OUT = Path(a.src)
    have = {f.name.replace('_rows.csv', '') for f in OUT.glob('*_rows.csv')}
    gap = sorted({f'{x}_s{s}' for x in NEW for s in SEEDS} - have)
    if gap and not a.partial:
        print(f'REFUSING: {len(gap)} runs missing {gap[:6]}'); raise SystemExit(2)

    D = {}
    for x in NEW + list(SRC):
        d, tag = SRC.get(x, (OUT, x))
        for s in SEEDS:
            try:
                D[(x, s)] = (R.read_rows(d / f'{tag}_s{s}_rows.csv'), json.load(open(d / f'{tag}_s{s}_provenance.json')))
            except FileNotFoundError:
                pass
    present = [x for x in ['E0'] + CLAMPED if any(k[0] == x for k in D)]

    def w(x, s, key, lo=LATE[0], hi=LATE[1]):
        v = [r[key] for r in D[(x, s)][0] if lo <= r['task'] <= hi and key in r and np.isfinite(r[key])]
        return float(np.median(v)) if v else np.nan

    def band(x, s):
        # carried-in arms have no 'band' column; reconstruct from the same two shares
        v = [(1 - r['cov'] - r['pos_frac']) for r in D[(x, s)][0] if LATE[0] <= r['task'] <= LATE[1]]
        return float(np.median(v)) if v else np.nan

    def L(x, s):
        return (w(x, s, 'acc', *BASE) - w(x, s, 'acc')) * 100

    def med(x, k):
        return float(np.median([w(x, s, k) for s in SEEDS if (x, s) in D]))

    def mB(x):
        return float(np.median([band(x, s) for s in SEEDS if (x, s) in D]))

    def mL(x):
        return float(np.median([L(x, s) for s in SEEDS if (x, s) in D]))

    broken = [x for x in present if min(float(np.mean([r['ce20'] - r['ce0'] < 0 for r in D[(x, s)][0]
                                                       if LATE[0] <= r['task'] <= LATE[1]])) for s in SEEDS if (x, s) in D) < G3_CE_FRAC]
    ok = lambda x: all((x, s) in D for s in SEEDS) and x not in broken

    V, lines = {}, ['# band_omega_0912 summary\n',
                    'spec: `specs/spec_band_omega_0912.md`（事前登録 `38bca36`・実装 `798663d`・判定値は未読で起動）\n',
                    f'scope: 新規 4 腕 × 3 seed ＋ E1/E0 を `elu_turn_0912` から持ち込み・base t{BASE[0]}–{BASE[1]}・late t{LATE[0]}–{LATE[1]}・leaky の傾き {B_LEAKY} pt/octave を固定で持ち込む\n']

    # ---- Q0
    lines.append('\n## 0. 検査\n\n| arm | κ | lr | G1 錨 | G1 maxabs | 配列数 | 幅の釘付け | 帯の恒等式 | 角度 | 三角 | CE 改善率 |\n|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|')
    g1_ok = True
    for x in present:
        ck = D[(x, 0)][1]['checks']
        g1s = [D[(x, s)][1]['checks']['g1_units_maxabs'] for s in SEEDS if (x, s) in D]
        g1_ok = g1_ok and all(v is not None and v <= 1e-10 for v in g1s)
        cef = min(float(np.mean([r['ce20'] - r['ce0'] < 0 for r in D[(x, s)][0]
                                 if LATE[0] <= r['task'] <= LATE[1]])) for s in SEEDS if (x, s) in D)
        lines.append(f"| {x} | {ck.get('kappa', '—')} | {ck['lr']:.5f} | {ck['g1_anchor']} | {' / '.join(f'{v:g}' for v in g1s)} | "
                     f"{ck['g1_units_compared']} | {max(D[(x, s)][1]['checks'].get('q0_cnorm_rel', 0.) for s in SEEDS if (x, s) in D):.1e} | "
                     f"{max(D[(x, s)][1]['checks'].get('g5_band_ident', 0.) for s in SEEDS if (x, s) in D):.1e} | "
                     f"{max(D[(x, s)][1]['checks'].get('g2_angle_rel', 0) for s in SEEDS if (x, s) in D):.1e} | "
                     f"{max(D[(x, s)][1]['checks']['g2_triangle'] for s in SEEDS if (x, s) in D):.1e} | {cef:.2f} |")
    if broken:
        lines.append(f"\n**LEARNING_BROKEN**: {broken}")
    nref, oref = med('E1', 'cnorm'), med('E1', 'omega_step')
    n_ok = (1.9 <= med('E2', 'cnorm') / nref <= 2.1) and (0.45 <= med('E05', 'cnorm') / nref <= 0.55)
    o_ok = (all(0.4 <= med(x, 'omega_step') / oref <= 0.6 for x in ('E1h', 'E2')) and
            all(1.7 <= med(x, 'omega_step') / oref <= 2.3 for x in ('E1d', 'E05')))
    pr = {f'{p}/{q}': med(p, 'omega_step') / med(q, 'omega_step') for p, q in PAIRS}
    p_ok = all(abs(v - 1) < 0.15 for v in pr.values())
    V['Q0'] = 'MANIPULATION_OK' if (g1_ok and n_ok and o_ok and not broken) else 'NOT_TESTABLE'
    V['Q0_pairs'] = 'PAIRS_MATCHED' if p_ok else 'PAIRS_NOT_MATCHED'
    V['Q0_parts'] = f'G1={g1_ok} norms={n_ok} omega={o_ok} pairs={p_ok} broken={broken}'
    V['Q0_pair_omega_ratio'] = {k: round(v, 3) for k, v in pr.items()}
    lines.append(f"\n- ω を揃えた対の実測比: {V['Q0_pair_omega_ratio']}（帯は 1 から 0.15 以内）")
    lines.append(f"- **Q0**: {V['Q0_parts']} → `{V['Q0']}`／対の一致 `{V['Q0_pairs']}`（Q0.5 は Q1 だけを塞ぐ）")
    if ok('E0') and ok('E1'):
        dl = [L('E0', s) - L('E1', s) for s in SEEDS]
        V['testability_dL'] = [round(x, 2) for x in dl]; V['testability_met'] = bool(all(x >= 1.0 for x in dl))
        lines.append(f"- **可検定性（Q0 と独立）**: L(E0) − L(E1) = {V['testability_dL']} pt、3/3 は {V['testability_met']}")

    # ---- per-arm
    lines.append('\n## 1. 腕ごとの量（seed 中央値）\n\n| arm | κ | lr | L [pt] | level | **ω** | **‖W̃ᵢ‖** | **浅い帯 B** | 正側 | 深い側 | 一歩 | 死 | ρ |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for x in present:
        ck = D[(x, 0)][1]['checks']
        lines.append(f"| {x} | {ck.get('kappa', '—')} | {ck['lr']:.5f} | {mL(x):.2f} | {med(x, 'acc'):.4f} | {med(x, 'omega_step'):.3e} | "
                     f"{med(x, 'cnorm'):.2f} | **{mB(x):.4f}** | {med(x, 'pos_frac'):.3f} | {med(x, 'cov'):.3f} | "
                     f"{med(x, 'step_norm'):.3e} | {med(x, 'hard_dead'):.0f} | {med(x, 'rho_mean'):.1f} |")

    if V['Q0'] == 'MANIPULATION_OK':
        ds = [L('E1h', s) - L('E2', s) for s in SEEDS]
        df = [L('E1d', s) - L('E05', s) for s in SEEDS]
        V['Q1_d_slow'] = [round(x, 2) for x in ds]; V['Q1_d_fast'] = [round(x, 2) for x in df]
        lab = ('OMEGA_SUFFICES_IN_ELU' if all(abs(x) < .3 for x in ds + df)
               else 'BAND_SPLITS_THE_PAIR' if all(x <= -.5 for x in ds)
               else 'BAND_SPLITS_INVERTED' if all(x >= .5 for x in ds) else 'Q1_PARTIAL')
        V['Q1'] = lab if p_ok else 'NOT_TESTABLE (pairs not omega-matched)'
        V['Q1_posthoc_label'] = lab
        V['Q1_band_ratio'] = {f'{p}/{q}': round(mB(p) / mB(q), 3) for p, q in PAIRS}
        lines.append(f"\n- **Q1（決定的）**: 遅い対 L(E1h) − L(E2) = {V['Q1_d_slow']}、速い対 L(E1d) − L(E05) = {V['Q1_d_fast']} pt → `{V['Q1']}`")
        lines.append(f"  - 同じ対の浅い帯の比 {V['Q1_band_ratio']}、長さの比 {{'E1h/E2': {med('E1h', 'cnorm') / med('E2', 'cnorm'):.2f}, 'E1d/E05': {med('E1d', 'cnorm') / med('E05', 'cnorm'):.2f}}}")
        lines.append(f"  - 参考・leaky の同じ対（`turn_rate_0912`）: 遅い +0.09 / 速い +0.48 pt、浅い帯はどちらも 0.000")
        cl = [x for x in CLAMPED if ok(x)]
        X = np.array([np.log2(w(x, s, 'omega_step')) for x in cl for s in SEEDS if (x, s) in D])
        Y = np.array([L(x, s) for x in cl for s in SEEDS if (x, s) in D])
        Z = np.array([np.log(band(x, s)) for x in cl for s in SEEDS if (x, s) in D])
        b, a0 = np.polyfit(X, Y, 1)
        V['Q2_slope'] = round(float(b), 3); V['Q2_n'] = len(X)
        V['Q2'] = ('SAME_SLOPE_AS_LEAKY' if abs(b - B_LEAKY) < .3
                   else 'ELU_SLOPE_STEEPER' if abs(b - B_LEAKY) >= .6 else 'Q2_PARTIAL')
        lines.append(f"- **Q2**（n={V['Q2_n']}）: ELU の傾き = **{b:+.2f}** pt/octave（leaky {B_LEAKY}、τ ダイヤルでは −1.59） → `{V['Q2']}`")
        rk = {}
        for x in ('E2', 'E05', 'E1h', 'E1d'):
            if ok(x):
                rk[x] = round(mB(x) / mB('E1'), 3)
        V['Q3_band_ratio_vs_E1'] = rk
        kap = max(abs(np.log(rk.get('E2', 1))), abs(np.log(rk.get('E05', 1))))
        lrm = max(abs(np.log(rk.get('E1h', 1))), abs(np.log(rk.get('E1d', 1))))
        V['Q3'] = ('BAND_MOVES_WITH_KAPPA_ONLY' if kap >= np.log(1.2) and lrm < np.log(1.2)
                   else 'BAND_STATIC' if kap < np.log(1.2) and lrm < np.log(1.2) else 'BAND_MOVES_WITH_BOTH')
        lines.append(f"- **Q3**: E1 に対する浅い帯の比 {rk}（κ 側 E2/E05・lr 側 E1h/E1d） → `{V['Q3']}`")
        res = Y - (a0 + b * X)
        V['Q4_spearman'] = R.spearman(list(Z), list(res))
        V['Q4'] = ('BAND_ADDS_NOTHING' if abs(V['Q4_spearman']) <= .3
                   else 'BAND_ADDS_BEYOND_OMEGA' if abs(V['Q4_spearman']) >= .6 else 'Q4_PARTIAL')
        lines.append(f"- **Q4**（n={len(X)}）: ω の当てはめ残差と log(浅い帯) の Spearman = **{V['Q4_spearman']:+.2f}** → `{V['Q4']}`")
    else:
        lines.append('\n**Q1–Q4 は Q0 のゲートで `NOT_TESTABLE`。**')

    (OUT / 'summary.md').write_text('\n'.join(lines) + '\n')
    with open(OUT / 'verdict.csv', 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(V.keys()), lineterminator='\n')
        wr.writeheader(); wr.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in V.items()})
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
        for x in present:
            ax[0].scatter(med(x, 'omega_step'), mL(x), s=55)
            ax[0].annotate(x, (med(x, 'omega_step'), mL(x)), fontsize=9, xytext=(5, 4), textcoords='offset points')
            ax[1].scatter(mB(x), mL(x), s=55)
            ax[1].annotate(x, (mB(x), mL(x)), fontsize=9, xytext=(5, 4), textcoords='offset points')
        ax[0].set_xscale('log'); ax[0].set_xlabel('omega  (turn per update)'); ax[0].set_ylabel('L [pt]')
        ax[1].set_xlabel('shallow band share'); ax[1].set_ylabel('L [pt]')
        fig.tight_layout(); fig.savefig(OUT / 'fig_band_omega.png', dpi=120)
    except Exception as e:
        print('figure skipped:', e)
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
