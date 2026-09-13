"""Verdicts for spec_turn_rate_0912 (K0-K4).  Refuses an incomplete run unless --partial."""
from pathlib import Path
import argparse, csv, json
import numpy as np
from src import gate_shape_report_0911 as R

ROOT = R.ROOT
OUT = ROOT / 'results/turn_rate_0912'
ARMS = ['N0', 'W1', 'W1h', 'W2', 'W1d', 'W05']
CLAMPED = ['W1', 'W1h', 'W2', 'W1d', 'W05']
PAIRS = (('W1h', 'W2'), ('W1d', 'W05'))
SEEDS = [0, 1, 2]
BASE = (16, 20)
LATE = (101, 120)
G3_CE_FRAC = 0.90


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=None); ap.add_argument('--partial', action='store_true')
    a = ap.parse_args()
    global OUT
    if a.src:
        OUT = Path(a.src)
    have = {f.name.replace('_rows.csv', '') for f in OUT.glob('*_rows.csv')}
    want = {f'{x}_s{s}' for x in ARMS for s in SEEDS}
    gap = sorted(want - have)
    if gap and not a.partial:
        print(f'REFUSING: {len(gap)}/{len(want)} runs missing {gap[:6]}'); raise SystemExit(2)

    D = {}
    for x in ARMS:
        for s in SEEDS:
            try:
                D[(x, s)] = (R.read_rows(OUT / f'{x}_s{s}_rows.csv'),
                             json.load(open(OUT / f'{x}_s{s}_provenance.json')))
            except FileNotFoundError:
                pass
    arms = [x for x in ARMS if any(k[0] == x for k in D)]

    def w(x, s, key, lo=LATE[0], hi=LATE[1]):
        v = [r[key] for r in D[(x, s)][0] if lo <= r['task'] <= hi and key in r and np.isfinite(r[key])]
        return float(np.median(v)) if v else np.nan

    def L(x, s):
        return (w(x, s, 'acc', *BASE) - w(x, s, 'acc')) * 100

    def med(x, k):
        return float(np.median([w(x, s, k) for s in SEEDS if (x, s) in D]))

    def mL(x):
        return float(np.median([L(x, s) for s in SEEDS if (x, s) in D]))

    broken = [x for x in arms if min(float(np.mean([r['ce20'] - r['ce0'] < 0 for r in D[(x, s)][0]
                                                    if LATE[0] <= r['task'] <= LATE[1]]))
                                     for s in SEEDS if (x, s) in D) < G3_CE_FRAC]
    ok = lambda x: all((x, s) in D for s in SEEDS) and x not in broken

    V, lines = {}, ['# turn_rate_0912 summary\n',
                    'spec: `specs/spec_turn_rate_0912.md`（事前登録 `8e54b0d`・実装とスモーク修正 `a6a6dc4`・判定値は未読で起動）\n',
                    f'scope: {len(arms)} 腕 × 3 seed・t1–120・base t{BASE[0]}–{BASE[1]}・late t{LATE[0]}–{LATE[1]}\n']

    # ---- K0
    lines.append('\n## 0. 検査\n\n| arm | κ | lr | G1 錨 | G1 maxabs | 角度の突合 | 三角不等式 | クランプ | CE 改善率 |\n|---|---:|---:|---|---|---:|---:|---|---:|')
    g1_ok = True
    for x in arms:
        ck = D[(x, 0)][1]['checks']
        g1s = [D[(x, s)][1]['checks']['g1_units_maxabs'] for s in SEEDS if (x, s) in D]
        g1_ok = g1_ok and all(v is not None and v <= 1e-10 for v in g1s)
        cef = min(float(np.mean([r['ce20'] - r['ce0'] < 0 for r in D[(x, s)][0]
                                 if LATE[0] <= r['task'] <= LATE[1]])) for s in SEEDS if (x, s) in D)
        cl = f"c3 {ck['c3_cnorm_rel']:.1e}" if 'c3_cnorm_rel' in ck else '—'
        lines.append(f"| {x} | {ck['kappa']} | {ck['lr']:.5f} | {ck['g1_anchor']} | "
                     f"{' / '.join(f'{v:g}' for v in g1s)} | "
                     f"{max(D[(x, s)][1]['checks']['g2_angle_rel'] for s in SEEDS if (x, s) in D):.1e} | "
                     f"{max(D[(x, s)][1]['checks']['g2_triangle'] for s in SEEDS if (x, s) in D):.1e} | {cl} | {cef:.2f} |")
    if broken:
        lines.append(f"\n**LEARNING_BROKEN**: {broken}（判定から外す）")

    nref = med('W1', 'cnorm')
    oref = med('W1', 'omega_step')
    n_ok = (1.9 <= med('W2', 'cnorm') / nref <= 2.1) and (0.45 <= med('W05', 'cnorm') / nref <= 0.55)
    o_ok = (all(0.4 <= med(x, 'omega_step') / oref <= 0.6 for x in ('W1h', 'W2')) and
            all(1.7 <= med(x, 'omega_step') / oref <= 2.3 for x in ('W1d', 'W05')))
    pr = {f'{p}/{q}': med(p, 'omega_step') / med(q, 'omega_step') for p, q in PAIRS}
    p_ok = all(abs(v - 1) < 0.15 for v in pr.values())
    core = g1_ok and n_ok and o_ok and not broken
    V['K0'] = 'MANIPULATION_OK' if core else 'NOT_TESTABLE'          # gates K2-K4
    V['K0_pairs'] = 'PAIRS_MATCHED' if p_ok else 'PAIRS_NOT_MATCHED'  # gates K1 only
    V['K0_strict'] = 'MANIPULATION_OK' if (core and p_ok) else 'NOT_TESTABLE'
    V['K0_parts'] = f'G1={g1_ok} norms={n_ok} omega={o_ok} pairs={p_ok} broken={broken}'
    V['K0_pair_omega_ratio'] = {k: round(v, 3) for k, v in pr.items()}
    lines.append('\n### K0 操作の実測（seed 中央値）\n\n| arm | κ | ‖W̃ᵢ‖ | W1 比 | ω [rad/更新] | W1 比 | 設計比 |\n|---|---:|---:|---:|---:|---:|---:|')
    for x in CLAMPED:
        if not any((x, s) in D for s in SEEDS):
            continue
        ck = D[(x, 0)][1]['checks']
        lines.append(f"| {x} | {ck['kappa']} | {med(x, 'cnorm'):.3f} | {med(x, 'cnorm') / nref:.2f} | "
                     f"{med(x, 'omega_step'):.4e} | {med(x, 'omega_step') / oref:.2f} | "
                     f"{(ck['lr'] / 1e-3) / ck['kappa']:.2f} |")
    lines.append(f"\n- ω を揃えた対の実測比: {V['K0_pair_omega_ratio']}（登録帯は 1 から 0.15 以内）")
    lines.append(f"- **K0**: {V['K0_parts']} → `{V['K0']}`")
    if ok('N0') and ok('W1'):
        dl = [L('N0', s) - L('W1', s) for s in SEEDS]
        V['testability_L_gap'] = [round(x, 2) for x in dl]
        V['testability_met'] = bool(all(x >= 1.0 for x in dl))
        lines.append(f"- **可検定性（K0 と独立に記録）**: L(N0) − L(W1) = {V['testability_L_gap']} pt、3/3 は {V['testability_met']}")

    # ---- per-arm
    lines.append('\n## 1. 腕ごとの量（seed 中央値）\n\n| arm | κ | lr | L [pt] | level | ‖W̃ᵢ‖ | ω | Θ_task | turn_eff | drift2_net | ρ | κ2 | ‖W2‖ |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for x in arms:
        ck = D[(x, 0)][1]['checks']
        lines.append(f"| {x} | {ck['kappa']} | {ck['lr']:.5f} | {mL(x):.2f} | {med(x, 'acc'):.4f} | "
                     f"{med(x, 'cnorm'):.2f} | {med(x, 'omega_step'):.4e} | {med(x, 'theta_task'):.3f} | "
                     f"{med(x, 'turn_eff'):.4f} | {med(x, 'drift2_net'):.2f} | {med(x, 'rho_mean'):.1f} | "
                     f"{med(x, 'kap2'):.4f} | {med(x, 'w2_fro'):.2f} |")

    if V['K0'] == 'MANIPULATION_OK':
        ds = [L('W1h', s) - L('W2', s) for s in SEEDS]
        df = [L('W1d', s) - L('W05', s) for s in SEEDS]
        V['K1_d_slow'] = [round(x, 2) for x in ds]; V['K1_d_fast'] = [round(x, 2) for x in df]
        lab = ('OMEGA_EXPLAINS_WIDTH' if all(abs(x) < .3 for x in ds + df)
               else 'OMEGA_FAILS' if any(abs(x) >= .7 for x in ds + df) else 'K1_PARTIAL')
        V['K1'] = lab if p_ok else 'NOT_TESTABLE (pairs not omega-matched)'
        V['K1_posthoc_label'] = lab
        lines.append(f"\n- **K1（決定的）**: 遅い対 L(W1h)−L(W2) = {V['K1_d_slow']}、速い対 L(W1d)−L(W05) = {V['K1_d_fast']} pt → `{V['K1']}`"
                     + ('' if p_ok else f"（対の ω が揃っていないので登録上は判定できない。事後のラベルは `{lab}`）"))
        cl = [x for x in CLAMPED if ok(x)]
        V['K2_spearman'] = R.spearman([med(x, 'omega_step') for x in cl], [mL(x) for x in cl])
        V['K2'] = ('OMEGA_ORDERS' if V['K2_spearman'] <= -.8
                   else 'OMEGA_DOES_NOT_ORDER' if V['K2_spearman'] >= -.2 else 'K2_PARTIAL')
        V['K3_spearman_theta'] = R.spearman([med(x, 'theta_task') for x in cl], [mL(x) for x in cl])
        d3 = abs(V['K3_spearman_theta']) - abs(V['K2_spearman'])
        V['K3'] = 'NET_TURN_BETTER' if d3 >= .2 else 'STEP_RATE_BETTER' if d3 <= -.2 else 'K3_TIE'
        lines.append(f"- **K2**: Spearman(ω, L) over {cl} = {V['K2_spearman']:+.2f} → `{V['K2']}`")
        lines.append(f"- **K3**: Spearman(Θ_task, L) = {V['K3_spearman_theta']:+.2f} → `{V['K3']}`")
        # K4: does length survive after omega is accounted for?
        pts = [(np.log2(w(x, s, 'omega_step')), L(x, s), np.log2(w(x, s, 'cnorm')))
               for x in cl for s in SEEDS if (x, s) in D]
        X = np.array([p[0] for p in pts]); Y = np.array([p[1] for p in pts]); Z = np.array([p[2] for p in pts])
        b, a0 = np.polyfit(X, Y, 1)
        res = Y - (a0 + b * X)
        V['K4_slope_pt_per_octave'] = round(float(b), 3)
        V['K4_n'] = len(pts)
        V['K4_spearman_resid_len'] = R.spearman(list(Z), list(res))
        V['K4'] = ('LENGTH_ADDS_NOTHING' if abs(V['K4_spearman_resid_len']) <= .3
                   else 'LENGTH_ADDS_BEYOND_OMEGA' if abs(V['K4_spearman_resid_len']) >= .6 else 'K4_PARTIAL')
        lines.append(f"- **K4**（n={V['K4_n']}）: L = {a0:.2f} {b:+.2f}·log₂ω に当てはめた残差と log₂‖W̃ᵢ‖ の "
                     f"Spearman = {V['K4_spearman_resid_len']:+.2f} → `{V['K4']}`")
    else:
        lines.append('\n**K1–K4 は K0 のゲートで `NOT_TESTABLE`。** 事後の数値は下表のとおり。')
        for p, q in PAIRS:
            if ok(p) and ok(q):
                lines.append(f"- （事後）L({p}) − L({q}) = {[round(L(p, s) - L(q, s), 2) for s in SEEDS]} pt")

    (OUT / 'summary.md').write_text('\n'.join(lines) + '\n')
    with open(OUT / 'verdict.csv', 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(V.keys()), lineterminator='\n')
        wr.writeheader(); wr.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in V.items()})
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        for x in arms:
            for axi, k in ((ax[0], 'omega_step'), (ax[1], 'cnorm')):
                axi.scatter(med(x, k), mL(x), s=55)
                axi.annotate(x, (med(x, k), mL(x)), fontsize=9, xytext=(5, 4), textcoords='offset points')
        ax[0].set_xscale('log'); ax[0].set_xlabel('ω  回る速さ [rad/更新]'); ax[0].set_ylabel('目減り L [pt]')
        ax[1].set_xlabel('‖W̃ᵢ‖  ユニットの長さ'); ax[1].set_ylabel('目減り L [pt]')
        fig.tight_layout(); fig.savefig(OUT / 'fig_turn_rate.png', dpi=120)
    except Exception as e:
        print('figure skipped:', e)
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
