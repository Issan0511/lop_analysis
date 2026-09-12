"""Verdicts for spec_turn_budget_0912 (T0-T5).  Refuses an incomplete run unless --partial."""
from pathlib import Path
import argparse, csv, json
import numpy as np
from src import gate_shape_report_0911 as R

ROOT = R.ROOT
OUT = ROOT / 'results/turn_budget_0912'
TR = ROOT / 'results/turn_rate_0912'      # the width-clamp arm, for T5
ARMS = ['R0', 'TH', 'P1', 'FLAT']
SEEDS = [0, 1, 2]
BASE = (16, 20)
LATE = (101, 120)
BRANCH = 21
G3_CE_FRAC = 0.90


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
    for x in ARMS:
        for s in SEEDS:
            try:
                D[(x, s)] = (R.read_rows(OUT / f'{x}_s{s}_rows.csv'), json.load(open(OUT / f'{x}_s{s}_provenance.json')))
            except FileNotFoundError:
                pass
    present = [x for x in ARMS if any(k[0] == x for k in D)]

    def w(x, s, key, lo=LATE[0], hi=LATE[1]):
        v = [r[key] for r in D[(x, s)][0] if lo <= r['task'] <= hi and key in r and np.isfinite(r[key])]
        return float(np.median(v)) if v else np.nan

    def L(x, s):
        return (w(x, s, 'acc', *BASE) - w(x, s, 'acc')) * 100

    def med(x, k):
        return float(np.median([w(x, s, k) for s in SEEDS if (x, s) in D]))

    def mL(x):
        return float(np.median([L(x, s) for s in SEEDS if (x, s) in D]))

    def ckv(x, k):
        return [D[(x, s)][1]['checks'][k] for s in SEEDS if (x, s) in D]

    broken = [x for x in present if min(float(np.mean([r['ce20'] - r['ce0'] < 0 for r in D[(x, s)][0]
                                                       if LATE[0] <= r['task'] <= LATE[1]])) for s in SEEDS if (x, s) in D) < G3_CE_FRAC]
    ok = lambda x: all((x, s) in D for s in SEEDS) and x not in broken

    V, lines = {}, ['# turn_budget_0912 summary\n',
                    'spec: `specs/spec_turn_budget_0912.md`（事前登録 `e8225fc`・実装と追補 1 `a1d8567`・判定値は未読で起動）\n',
                    f'scope: {len(present)} 腕 × 3 seed・t1–120・分岐 t{BRANCH}・base t{BASE[0]}–{BASE[1]}（分岐前・全腕同一）・late t{LATE[0]}–{LATE[1]}\n']

    # ---- T0
    lines.append('\n## 0. 検査\n\n| arm | 方針 | G1 錨 | G1 maxabs | 配列数 | 角度 | 三角 | 系列非汚染 | 停止規則の違反 | CE 改善率 |\n|---|---|---|---|---:|---:|---:|---|---:|---:|')
    g1_ok = True
    for x in present:
        ck = D[(x, 0)][1]['checks']
        g1s = ckv(x, 'g1_units_maxabs')
        g1_ok = g1_ok and all(v <= 1e-10 for v in g1s)
        cef = min(float(np.mean([r['ce20'] - r['ce0'] < 0 for r in D[(x, s)][0]
                                 if LATE[0] <= r['task'] <= LATE[1]])) for s in SEEDS if (x, s) in D)
        lines.append(f"| {x} | {ck['policy']} | {ck['g1_anchor']} | {' / '.join(f'{v:g}' for v in g1s)} | {ck['g1_units_compared']} | "
                     f"{max(ckv(x, 'g2_angle_rel')):.1e} | {max(ckv(x, 'g2_triangle')):.1e} | {all(ckv(x, 'g4_streams_intact'))} | "
                     f"{ck.get('g3_stop_grid_violations', '—')} | {cef:.2f} |")
    if broken:
        lines.append(f"\n**LEARNING_BROKEN**: {broken}")
    th_ratio = [w('TH', s, 'theta_ratio') for s in SEEDS if ('TH', s) in D]
    hit = [float(np.mean([r['hit_cap'] for r in D[('TH', s)][0] if r['task'] >= BRANCH])) for s in SEEDS if ('TH', s) in D]
    reach = [float(np.mean([r['theta_ratio'] >= 0.95 for r in D[('TH', s)][0] if LATE[0] <= r['task'] <= LATE[1]]))
             for s in SEEDS if ('TH', s) in D]
    V['T0_reach_frac'] = [round(x, 3) for x in reach]; V['T0_cap_frac'] = [round(x, 3) for x in hit]
    V['T0'] = 'MANIPULATION_OK' if (g1_ok and all(x >= .90 for x in reach) and not broken) else 'NOT_TESTABLE'
    lines.append(f"\n- **T0**: G1={g1_ok}／`TH` の late 窓で Θ/Θ* ≥ 0.95 の割合 = {V['T0_reach_frac']}（要 0.90）／上限に当たった割合 = {V['T0_cap_frac']} → `{V['T0']}`")
    lines.append(f"- 目標 Θ* = {[round(x, 4) for x in ckv('TH', 'theta_star')]} rad")
    if ok('R0'):
        V['testability_L_R0'] = [round(L('R0', s), 2) for s in SEEDS]
        V['testability_met'] = bool(all(x >= 1.5 for x in V['testability_L_R0']))
        lines.append(f"- **可検定性（T0 と独立）**: L(R0) = {V['testability_L_R0']} pt、`>= 1.5 を 3/3` は {V['testability_met']}")

    # ---- table
    lines.append('\n## 1. 腕ごとの量（seed 中央値）\n\n| arm | **費用 C** | **L [pt]** | 絶対水準 | 更新数 平均 / 最大 | Θ/Θ* | ‖W̃ᵢ‖ | ω | 周回 | ρ | ‖W2‖ |\n|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|')
    for x in present:
        c = float(np.median(ckv(x, 'cost')))
        lines.append(f"| {x} | **{c:.2f}** | **{mL(x):.2f}** | {med(x, 'acc'):.4f} | "
                     f"{float(np.median(ckv(x, 'n_steps_mean'))):.0f} / {max(ckv(x, 'n_steps_max')):.0f} | "
                     f"{med(x, 'theta_ratio'):.2f} | {med(x, 'cnorm'):.2f} | {med(x, 'omega_step'):.3e} | "
                     f"{med(x, 'epochs'):.1f} | {med(x, 'rho_mean'):.1f} | {med(x, 'w2_fro'):.1f} |")
    if (TR / 'W1_s0_rows.csv').exists():
        wr = [R.read_rows(TR / f'W1_s{s}_rows.csv') for s in SEEDS]
        def ww(rs, k, lo, hi):
            v = [r[k] for r in rs if lo <= r['task'] <= hi and k in r and np.isfinite(r[k])]
            return float(np.median(v)) if v else np.nan
        wl = float(np.median([(ww(rs, 'acc', *BASE) - ww(rs, 'acc', *LATE)) * 100 for rs in wr]))
        wlev = float(np.median([ww(rs, 'acc', *LATE) for rs in wr]))
        V['clamp_L'] = round(wl, 2); V['clamp_level'] = round(wlev, 4)
        lines.append(f"| 幅クランプ（`turn_rate_0912/W1`） | **1.00** | **{wl:.2f}** | {wlev:.4f} | 625 / 625 | — | 3.72 | 1.378e−3 | 1.0 | 38.4 | — |")

    if V['T0'] == 'MANIPULATION_OK':
        lt = [L('TH', s) for s in SEEDS]
        V['T1_L_TH'] = [round(x, 2) for x in lt]
        V['T1'] = ('TURN_BUYS_PLASTICITY' if all(x <= .3 for x in lt)
                   else 'TURN_NOT_ENOUGH' if all(x >= 1.5 for x in lt) else 'T1_PARTIAL')
        lines.append(f"\n- **T1（決定的）**: L(TH) = {V['T1_L_TH']} pt（費用 {float(np.median(ckv('TH', 'cost'))):.2f} 倍） → `{V['T1']}`")
        if ok('FLAT'):
            d = [L('FLAT', s) - L('TH', s) for s in SEEDS]
            V['T2_d'] = [round(x, 2) for x in d]
            V['T2'] = ('TRACKING_BEATS_FLAT' if all(x >= .5 for x in d)
                       else 'COMPUTE_NOT_TRACKING' if all(abs(x) < .3 for x in d) else 'T2_PARTIAL')
            lines.append(f"- **T2（対照）**: L(FLAT) − L(TH) = {V['T2_d']} pt（費用 2.00 対 {float(np.median(ckv('TH', 'cost'))):.2f}） → `{V['T2']}`")
        qs = []
        for s in SEEDS:
            if ('TH', s) not in D:
                continue
            cn20 = D[('TH', s)][1]['checks']['cn20']
            pts = [(np.log(r['cnorm'] / cn20), np.log(r['n_steps'] / 625))
                   for r in D[('TH', s)][0] if LATE[0] <= r['task'] <= LATE[1] and not r['hit_cap']]
            if len(pts) >= 5:
                qs.append(float(np.polyfit([p[0] for p in pts], [p[1] for p in pts], 1)[0]))
        V['T3_q'] = [round(x, 2) for x in qs]
        V['T3'] = ('BALLISTIC' if qs and all(x <= .8 for x in qs)
                   else 'DIFFUSIVE' if qs and all(x >= 1.6 for x in qs) else 'T3_LINEAR_ISH')
        lines.append(f"- **T3（指数）**: log(N/625) を log(‖W̃ᵢ‖/‖W̃ᵢ‖_t20) に当てはめた傾き q = {V['T3_q']}（除外: 上限に当たったタスク） → `{V['T3']}`")
        if ok('R0'):
            rr = [w('TH', s, 'cnorm') / w('R0', s, 'cnorm') for s in SEEDS if ('TH', s) in D and ('R0', s) in D]
            V['T4_ratio'] = [round(x, 3) for x in rr]
            V['T4'] = 'COMPOUNDING' if all(x >= 1.15 for x in rr) else 'NO_COMPOUNDING'
            lines.append(f"- **T4（複利）**: ‖W̃ᵢ‖(TH)/‖W̃ᵢ‖(R0) = {V['T4_ratio']}（費用 {float(np.median(ckv('TH', 'cost'))):.2f} 倍） → `{V['T4']}`")
            V['T5_level_gain'] = [round((w('TH', s, 'acc') - w('R0', s, 'acc')) * 100, 2) for s in SEEDS]
            lines.append(f"- **T5（絶対水準）**: level(TH) − level(R0) = {V['T5_level_gain']} pt。"
                         + (f"幅クランプは費用 1.00 で L {V['clamp_L']}・水準 {V['clamp_level']}。" if 'clamp_L' in V else ''))
    else:
        lines.append('\n**T1–T5 は T0 のゲートで `NOT_TESTABLE`。**')

    (OUT / 'summary.md').write_text('\n'.join(lines) + '\n')
    with open(OUT / 'verdict.csv', 'w', newline='') as f:
        wr2 = csv.DictWriter(f, fieldnames=list(V.keys()), lineterminator='\n')
        wr2.writeheader(); wr2.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in V.items()})
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
        for x in present:
            c = float(np.median(ckv(x, 'cost')))
            ax[0].scatter(c, mL(x), s=60); ax[0].annotate(x, (c, mL(x)), fontsize=9, xytext=(5, 4), textcoords='offset points')
        if 'clamp_L' in V:
            ax[0].scatter(1.0, V['clamp_L'], s=60, marker='*', color='tab:red')
            ax[0].annotate('width clamp', (1.0, V['clamp_L']), fontsize=9, xytext=(5, -12), textcoords='offset points', color='tab:red')
        ax[0].set_xlabel('cost  (updates vs reference)'); ax[0].set_ylabel('L [pt]'); ax[0].axhline(0, color='k', lw=.5)
        for x in present:
            ts = sorted({r['task'] for r in D[(x, 0)][0] if r['task'] >= BRANCH})
            ax[1].plot(ts, [np.median([[r['n_steps'] for r in D[(x, s)][0] if r['task'] == t][0]
                                       for s in SEEDS if (x, s) in D]) for t in ts], label=x)
        ax[1].set_xlabel('task'); ax[1].set_ylabel('updates used'); ax[1].legend()
        fig.tight_layout(); fig.savefig(OUT / 'fig_turn_budget.png', dpi=120)
    except Exception as e:
        print('figure skipped:', e)
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
