"""Verdicts for spec_elu_turn_0912 (M0-M4).  Refuses an incomplete run unless --partial."""
from pathlib import Path
import argparse, csv, json
import numpy as np
from src import gate_shape_report_0911 as R

ROOT = R.ROOT
OUT = ROOT / 'results/elu_turn_0912'
TR = ROOT / 'results/turn_rate_0912'
ARMS = ['CELU03', 'CELU1', 'CELU3', 'CELU1w', 'LR']
ELU = ['CELU03', 'CELU1', 'CELU3', 'CELU1w']
TR_ARMS = ['N0', 'W1', 'W1h', 'W2', 'W1d', 'W05']
SEEDS = [0, 1, 2]
BASE = (16, 20)
LATE = (101, 120)
B_LEAKY = 1.05          # pt per octave of omega, from turn_rate_0912 (fixed, carried in)
G3_CE_FRAC = 0.90


def partial_spearman(x, y, z):
    """Spearman of the rank residuals of x~z and y~z."""
    rx, ry, rz = (R.rankdata(np.asarray(v, float)) for v in (x, y, z))
    def resid(a, c):
        A = np.vstack([c, np.ones_like(c)]).T
        beta = np.linalg.lstsq(A, a, rcond=None)[0]
        return a - A @ beta
    return R.spearman(list(resid(rx, rz)), list(resid(ry, rz)))


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

    D, U = {}, {}
    for x in ARMS:
        for s in SEEDS:
            try:
                D[(x, s)] = (R.read_rows(OUT / f'{x}_s{s}_rows.csv'), json.load(open(OUT / f'{x}_s{s}_provenance.json')))
                U[(x, s)] = np.load(OUT / f'{x}_s{s}_units.npz')
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
                                                    if LATE[0] <= r['task'] <= LATE[1]])) for s in SEEDS if (x, s) in D) < G3_CE_FRAC]
    ok = lambda x: all((x, s) in D for s in SEEDS) and x not in broken

    V, lines = {}, ['# elu_turn_0912 summary\n',
                    'spec: `specs/spec_elu_turn_0912.md`（事前登録 `8fa2d2e`・判定値は未読で起動）\n',
                    f'scope: {len(arms)} 腕 × 3 seed・t1–120・base t{BASE[0]}–{BASE[1]}・late t{LATE[0]}–{LATE[1]}・leaky の傾き b = {B_LEAKY} pt/octave を固定で持ち込む\n']

    # ---- M0
    lines.append('\n## 0. 検査\n\n| arm | act | clamp | G1 錨 | G1 maxabs | 配列数 | 角度の突合 | θ≤1e−4 の割合 | 三角不等式 | ω 分解の検算 | クランプ |\n|---|---|---|---|---|---:|---:|---:|---:|---:|---|')
    g1_ok, dec_ok = True, True
    for x in arms:
        ck = D[(x, 0)][1]['checks']
        g1s = [D[(x, s)][1]['checks']['g1_units_maxabs'] for s in SEEDS if (x, s) in D]
        g1_ok = g1_ok and all(v is not None and v <= 1e-10 for v in g1s)
        gap_ = med(x, 'omega_decomp_gap'); dec_ok = dec_ok and gap_ <= 0.15
        cl = f"c3 {ck['c3_cnorm_rel']:.1e}" if 'c3_cnorm_rel' in ck else '—'
        lines.append(f"| {x} | {ck['act']} | {ck['clamp']} | {ck['g1_anchor']} | {' / '.join(f'{v:g}' for v in g1s)} | {ck['g1_units_compared']} | "
                     f"{max(D[(x, s)][1]['checks'].get('g2_angle_rel', 0) for s in SEEDS if (x, s) in D):.1e} | "
                     f"{max(D[(x, s)][1]['checks'].get('g2_tiny_angle_frac', 0) for s in SEEDS if (x, s) in D):.4f} | "
                     f"{max(D[(x, s)][1]['checks']['g2_triangle'] for s in SEEDS if (x, s) in D):.1e} | {gap_:.3f} | {cl} |")
    if broken:
        lines.append(f"\n**LEARNING_BROKEN**: {broken}")
    V['M0'] = 'MANIPULATION_OK' if (g1_ok and dec_ok and not broken) else 'NOT_TESTABLE'
    V['M0_parts'] = f'G1={g1_ok} decomp={dec_ok} broken={broken}'
    lines.append(f"\n- **M0**: {V['M0_parts']} → `{V['M0']}`")
    if ok('CELU03') and ok('CELU3'):
        dl = [L('CELU03', s) - L('CELU3', s) for s in SEEDS]
        V['testability_dL'] = [round(x, 2) for x in dl]; V['testability_met'] = bool(all(x >= 1.0 for x in dl))
        lines.append(f"- **可検定性（M0 と独立）**: L(CELU03) − L(CELU3) = {V['testability_dL']} pt、3/3 は {V['testability_met']}")

    # ---- per-arm
    lines.append('\n## 1. 腕ごとの量（seed 中央値）\n\n| arm | L [pt] | level | **ω** | **step**（分子） | **‖W̃ᵢ‖**（分母） | step/‖W̃‖ | κ2 | graw² | Ḡ | Cov | z̄ | dead | drift2 | ρ |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for x in arms:
        lines.append(f"| {x} | {mL(x):.2f} | {med(x, 'acc'):.4f} | **{med(x, 'omega_step'):.4e}** | **{med(x, 'step_norm'):.4e}** | **{med(x, 'cnorm'):.2f}** | "
                     f"{med(x, 'step_norm') / med(x, 'cnorm'):.4e} | {med(x, 'kap2'):.4f} | {med(x, 'graw2'):.4f} | {med(x, 'gbar'):.3f} | "
                     f"{med(x, 'cov'):.3f} | {med(x, 'zbar_inv'):.2f} | {med(x, 'hard_dead'):.0f} | {med(x, 'drift2_net'):.2f} | {med(x, 'rho_mean'):.1f} |")

    if V['M0'] == 'MANIPULATION_OK':
        # ---- M1
        r_om = med('CELU3', 'omega_step') / med('CELU03', 'omega_step')
        r_N = med('CELU3', 'cnorm') / med('CELU03', 'cnorm')
        r_s = med('CELU3', 'step_norm') / med('CELU03', 'step_norm')
        V.update(M1_r_omega=round(r_om, 3), M1_r_N=round(r_N, 3), M1_r_step=round(r_s, 3))
        V['M1'] = ('TAU_MOVES_OMEGA_NOT_LENGTH' if (r_om >= 2.0 and abs(np.log2(r_N)) < 0.3)
                   else 'TAU_DOES_NOT_MOVE_OMEGA' if r_om <= 1.2 else 'M1_PARTIAL')
        lines.append(f"\n- **M1（決定的）**: ω(CELU3)/ω(CELU03) = **{r_om:.2f}**、長さの比 {r_N:.3f}（log₂ {np.log2(r_N):+.2f}）、一歩の比 {r_s:.2f} → `{V['M1']}`")
        lines.append(f"  - 分解: log₂ω の差 {np.log2(r_om):+.2f} = 一歩 {np.log2(r_s):+.2f} − 長さ {np.log2(r_N):+.2f}")
        # ---- M2
        dL_obs = [L('CELU03', s) - L('CELU3', s) for s in SEEDS]
        dL_pred = B_LEAKY * np.log2(r_om)
        gapM2 = [abs(dL_pred - x) for x in dL_obs]
        V.update(M2_dL_pred=round(float(dL_pred), 2), M2_dL_obs=[round(x, 2) for x in dL_obs])
        V['M2'] = ('LEAKY_SLOPE_ACCOUNTS' if all(g < .5 for g in gapM2) else 'SLOPE_DOES_NOT_ACCOUNT' if any(g >= 1.0 for g in gapM2) else 'M2_PARTIAL')
        lines.append(f"- **M2**: leaky の傾き {B_LEAKY} × log₂({r_om:.2f}) = 予測 ΔL **{dL_pred:.2f}** pt、実測 ΔL = {V['M2_dL_obs']} pt → `{V['M2']}`")
        # ---- M3 per-unit partials
        lines.append('\n### M3 ユニット内の偏 Spearman（late 窓・hard-dead 除外・タスク中央値・seed 別）\n\n| arm | ρ(ω_i, ḡ_i \\| log‖W̃ᵢ‖) | ρ(ω_i, −log‖W̃ᵢ‖ \\| ḡ_i) | ラベル |\n|---|---|---|---|')
        m3 = {}
        for x in arms:
            if not ok(x):
                continue
            pg, pn = [], []
            for s in SEEDS:
                u = U[(x, s)]; vg, vn = [], []
                for t in range(LATE[0], LATE[1] + 1):
                    try:
                        om = u[f'ref_omega_i_t{t}']; g = u[f'ref_gbar_i_t{t}']; cn = u[f'ref_cnorm_i_t{t}']; hd = u[f'ref_hard_i_t{t}'].astype(bool)
                    except KeyError:
                        continue
                    m = ~hd & np.isfinite(om) & (om > 0)
                    if m.sum() < 20:
                        continue
                    vg.append(partial_spearman(om[m], g[m], np.log(cn[m])))
                    vn.append(partial_spearman(om[m], -np.log(cn[m]), g[m]))
                pg.append(float(np.median(vg))); pn.append(float(np.median(vn)))
            lab = ('BOTH_ENTER' if all(v >= .3 for v in pg) and all(v >= .3 for v in pn)
                   else 'GATE_ONLY' if all(v >= .3 for v in pg) else 'LENGTH_ONLY' if all(v >= .3 for v in pn) else 'M3_PARTIAL')
            m3[x] = lab
            V[f'M3_{x}_gate'] = [round(v, 2) for v in pg]; V[f'M3_{x}_len'] = [round(v, 2) for v in pn]
            lines.append(f"| {x} | {' / '.join(f'{v:+.2f}' for v in pg)} | {' / '.join(f'{v:+.2f}' for v in pn)} | `{lab}` |")
        V['M3'] = m3.get('CELU1', 'n/a'); V['M3_all'] = m3
        lines.append(f"\n- **M3**（CELU1 を代表に）: `{V['M3']}`（全腕: {m3}）")
        # ---- M4 one line across activations
        # turn_rate_0912 is read on ITS OWN registered windows, independent of this module's
        TR_BASE, TR_LATE, TR_SEEDS = (16, 20), (101, 120), [0, 1, 2]
        pts = []
        for x in TR_ARMS:
            try:
                rows = [R.read_rows(TR / f'{x}_s{s}_rows.csv') for s in TR_SEEDS]
            except FileNotFoundError:
                continue
            def ww(rs, k, lo, hi):
                v = [r[k] for r in rs if lo <= r['task'] <= hi and k in r and np.isfinite(r[k])]
                return float(np.median(v)) if v else np.nan
            om = float(np.median([ww(rs, 'omega_step', *TR_LATE) for rs in rows]))
            Lx = float(np.median([(ww(rs, 'acc', *TR_BASE) - ww(rs, 'acc', *TR_LATE)) * 100 for rs in rows]))
            if np.isfinite(om) and np.isfinite(Lx) and om > 0:
                pts.append(('leaky:' + x, np.log2(om), Lx))
        assert len(pts) >= 3, ('M4 needs the turn_rate_0912 arms', len(pts))
        X = np.array([p[1] for p in pts]); Y = np.array([p[2] for p in pts]); b, a0 = np.polyfit(X, Y, 1)
        V['M4_leaky_fit'] = f'L = {a0:.2f} {b:+.2f}·log2(omega)'
        res = {x: float(mL(x) - (a0 + b * np.log2(med(x, 'omega_step')))) for x in ELU if ok(x)}
        delta = float(np.median(list(res.values())))
        V['M4_resid'] = {k: round(v, 2) for k, v in res.items()}; V['M4_delta'] = round(delta, 2)
        V['M4'] = 'ONE_LINE' if abs(delta) < .5 else 'OFFSET' if abs(delta) >= 1.0 else 'M4_PARTIAL'
        lines.append(f"- **M4**: leaky 6 腕の直線 `{V['M4_leaky_fit']}` からの ELU 4 腕の残差 {V['M4_resid']}、中央値 δ = **{delta:+.2f}** pt → `{V['M4']}`")
        if ok('LR'):
            V['M4_LR_resid_here'] = round(mL('LR') - (a0 + b * np.log2(med('LR', 'omega_step'))), 2)
            lines.append(f"  - 本走の LR（同じ計装）の残差 {V['M4_LR_resid_here']:+.2f} pt（0 に近いほど 2 走の計装が整合）")
        all_pts = pts + [('elu:' + x, np.log2(med(x, 'omega_step')), mL(x)) for x in ELU if ok(x)]
        V['M4_spearman_all'] = R.spearman([p[1] for p in all_pts], [p[2] for p in all_pts])
        lines.append(f"  - 11 腕まとめての Spearman(log₂ω, L) = {V['M4_spearman_all']:+.2f}")
    else:
        lines.append('\n**M1–M4 は M0 のゲートで `NOT_TESTABLE`。**')

    (OUT / 'summary.md').write_text('\n'.join(lines) + '\n')
    with open(OUT / 'verdict.csv', 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(V.keys()), lineterminator='\n')
        wr.writeheader(); wr.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in V.items()})
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
        if V['M0'] == 'MANIPULATION_OK':
            for name, lx, ly in all_pts:
                c = 'tab:blue' if name.startswith('leaky') else 'tab:red'
                ax[0].scatter(lx, ly, s=55, color=c); ax[0].annotate(name.split(':')[1], (lx, ly), fontsize=8, xytext=(5, 4), textcoords='offset points', color=c)
            xs = np.linspace(min(p[1] for p in all_pts) - .3, max(p[1] for p in all_pts) + .3, 50)
            ax[0].plot(xs, a0 + b * xs, '--', color='tab:blue', lw=1, label='leaky fit')
            ax[0].set_xlabel('log2 omega  (turn per update)'); ax[0].set_ylabel('L [pt]'); ax[0].legend()
        for x in arms:
            ax[1].scatter(med(x, 'cnorm'), med(x, 'step_norm'), s=55); ax[1].annotate(x, (med(x, 'cnorm'), med(x, 'step_norm')), fontsize=8, xytext=(5, 4), textcoords='offset points')
        ax[1].set_xlabel('||W~_i||  (denominator)'); ax[1].set_ylabel('step norm  (numerator)')
        fig.tight_layout(); fig.savefig(OUT / 'fig_elu_turn.png', dpi=120)
    except Exception as e:
        print('figure skipped:', e)
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
