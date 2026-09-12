"""Verdicts for spec_budget_ladder_0912 (U0-U5).  Refuses an incomplete run unless --partial."""
from pathlib import Path
import argparse, csv, json
import numpy as np
from src import gate_shape_report_0911 as R

ROOT = R.ROOT
OUT = ROOT / 'results/budget_ladder_0912'
ARMS = ['B1', 'B2', 'B4', 'B8']
BUD = {'B1': 625, 'B2': 1250, 'B4': 2500, 'B8': 5000}
SEEDS = [0, 1, 2]
EARLY = (21, 30)
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

    def wm(x, s, key, lo, hi):
        v = [r[key] for r in D[(x, s)][0] if lo <= r['task'] <= hi and key in r and np.isfinite(r[key])]
        return float(np.mean(v)) if v else np.nan

    def dec(x, s):
        return (wm(x, s, 'acc', *EARLY) - wm(x, s, 'acc', *LATE)) * 100

    def mdec(x):
        return float(np.median([dec(x, s) for s in SEEDS if (x, s) in D]))

    def med(x, k, lo, hi):
        return float(np.median([wm(x, s, k, lo, hi) for s in SEEDS if (x, s) in D]))

    def ckv(x, k):
        return [D[(x, s)][1]['checks'].get(k) for s in SEEDS if (x, s) in D]

    broken = [x for x in present if min(float(np.mean([r['ce20'] - r['ce0'] < 0 for r in D[(x, s)][0]
                                                       if LATE[0] <= r['task'] <= LATE[1]])) for s in SEEDS if (x, s) in D) < G3_CE_FRAC]
    ok = lambda x: all((x, s) in D for s in SEEDS) and x not in broken

    V, lines = {}, ['# budget_ladder_0912 summary\n',
                    'spec: `specs/spec_budget_ladder_0912.md`（事前登録 `9eb211c`・判定値は未読で起動）\n',
                    f'scope: {len(present)} 腕 × 3 seed・t1–120 を同じ予算で・早い窓 t{EARLY[0]}–{EARLY[1]}・遅い窓 t{LATE[0]}–{LATE[1]}（窓内は平均・seed 中央値）\n']

    lines.append('\n## 0. 検査\n\n| arm | 予算 | G1 錨 | G1 maxabs | 更新数厳密 | 系列非汚染 | 角度 | 三角 | CE 改善率 |\n|---|---:|---|---|---|---|---:|---:|---:|')
    g1_ok, ex_ok = True, True
    for x in present:
        ck = D[(x, 0)][1]['checks']
        g1s = ckv(x, 'g1_units_maxabs')
        if x == 'B1':
            g1_ok = g1_ok and all(v is not None and v <= 1e-10 for v in g1s)
        ex_ok = ex_ok and all(ckv(x, 'u0_steps_exact'))
        cef = min(float(np.mean([r['ce20'] - r['ce0'] < 0 for r in D[(x, s)][0]
                                 if LATE[0] <= r['task'] <= LATE[1]])) for s in SEEDS if (x, s) in D)
        lines.append(f"| {x} | {ck['budget']} | {ck['g1_anchor']} | {' / '.join('—' if v is None else f'{v:g}' for v in g1s)} | "
                     f"{all(ckv(x, 'u0_steps_exact'))} | {all(ckv(x, 'g4_streams_intact'))} | "
                     f"{max(ckv(x, 'g2_angle_rel')):.1e} | {max(ckv(x, 'g2_triangle')):.1e} | {cef:.2f} |")
    if broken:
        lines.append(f"\n**LEARNING_BROKEN**: {broken}")
    V['U0'] = 'MANIPULATION_OK' if (g1_ok and ex_ok and not broken) else 'NOT_TESTABLE'
    lines.append(f"\n- **U0**: G1={g1_ok} steps_exact={ex_ok} broken={broken} → `{V['U0']}`")
    if ok('B1'):
        V['testability_D625'] = [round(dec('B1', s), 2) for s in SEEDS]
        V['testability_met'] = bool(all(x >= 1.0 for x in V['testability_D625']))
        lines.append(f"- **可検定性（U0 と独立）**: D(625) = {V['testability_D625']} pt、3/3 は {V['testability_met']}")

    lines.append('\n## 1. 予算ごとの量（seed 中央値）\n\n| arm | 予算 | 費用 | acc 早い窓 | acc 遅い窓 | **減衰 D** | 全タスク平均 | ‖W̃ᵢ‖ 早→遅 | ω 遅 | Θ 遅 | ‖W2‖ 遅 |\n|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|')
    for x in present:
        allm = float(np.median([wm(x, s, 'acc', 21, 120) for s in SEEDS if (x, s) in D]))
        lines.append(f"| {x} | {BUD[x]} | {BUD[x] / 625:.0f} | {med(x, 'acc', *EARLY):.4f} | {med(x, 'acc', *LATE):.4f} | **{mdec(x):.2f}** | {allm:.4f} | "
                     f"{med(x, 'cnorm', *EARLY):.2f} → {med(x, 'cnorm', *LATE):.2f} | {med(x, 'omega_step', *LATE):.2e} | "
                     f"{med(x, 'theta_task', *LATE):.3f} | {med(x, 'w2_fro', *LATE):.1f} |")

    if V['U0'] == 'MANIPULATION_OK' and ok('B1') and ok('B8'):
        rr = [dec('B8', s) / dec('B1', s) for s in SEEDS]
        V['U1_ratio'] = [round(x, 2) for x in rr]
        V['U1'] = ('DECLINE_SHRINKS' if all(x <= .5 for x in rr) else 'DECLINE_GROWS' if all(x >= 1.3 for x in rr) else 'DECLINE_FLAT')
        lines.append(f"\n- **U1（決定的）**: D(5000)/D(625) = {V['U1_ratio']} → `{V['U1']}`")
        d8 = [dec('B8', s) for s in SEEDS]
        V['U2_D5000'] = [round(x, 2) for x in d8]
        V['U2'] = ('COMPUTE_REMOVES_DECLINE' if all(x <= .5 for x in d8) else 'DECLINE_SURVIVES_COMPUTE' if all(x >= 1.5 for x in d8) else 'U2_PARTIAL')
        lines.append(f"- **U2（残る劣化）**: D(5000) = {V['U2_D5000']} pt → `{V['U2']}`")
        g = [(wm('B8', s, 'acc', *LATE) - wm('B1', s, 'acc', *EARLY)) * 100 for s in SEEDS]
        V['U3_gap'] = [round(x, 2) for x in g]
        V['U3'] = 'LEVEL_BUYABLE' if all(x >= 0 for x in g) else 'LEVEL_NOT_BUYABLE'
        lines.append(f"- **U3（水準は買えるか）**: acc(遅い窓, 5000) − acc(早い窓, 625) = {V['U3_gap']} pt → `{V['U3']}`")
        w8 = [wm('B8', s, 'cnorm', *LATE) / wm('B1', s, 'cnorm', *LATE) for s in SEEDS]
        V['U4_ratio'] = [round(x, 2) for x in w8]
        V['U4'] = 'COMPOUNDING' if all(x >= 1.5 for x in w8) else 'NO_COMPOUNDING'
        lines.append(f"- **U4（複利）**: ‖W̃ᵢ‖(遅い窓) 5000/625 = {V['U4_ratio']} → `{V['U4']}`")
        seq = [mdec(x) for x in ARMS if ok(x)]
        V['U5_D_seq'] = [round(x, 2) for x in seq]
        V['U5'] = ('MONOTONE_UP' if all(a <= b for a, b in zip(seq, seq[1:])) else 'MONOTONE_DOWN' if all(a >= b for a, b in zip(seq, seq[1:])) else 'NON_MONOTONE')
        lines.append(f"- **U5（単調性）**: D(625 / 1250 / 2500 / 5000) = {V['U5_D_seq']} → `{V['U5']}`")
    else:
        lines.append('\n**U1–U5 は U0 のゲートで `NOT_TESTABLE`。**')

    (OUT / 'summary.md').write_text('\n'.join(lines) + '\n')
    with open(OUT / 'verdict.csv', 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(V.keys()), lineterminator='\n')
        wr.writeheader(); wr.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in V.items()})
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
        for x in present:
            ts = list(range(1, 121, 5))
            ys = [np.median([np.mean([r['acc'] for r in D[(x, s)][0] if t <= r['task'] < t + 5]) for s in SEEDS if (x, s) in D]) * 100 for t in ts]
            ax[0].plot(ts, ys, label=f'{x} ({BUD[x]})')
        ax[0].set_xlabel('task'); ax[0].set_ylabel('accuracy [%]'); ax[0].legend()
        bs = [BUD[x] for x in present]; ds = [mdec(x) for x in present]
        ax[1].semilogx(bs, ds, 'o-'); ax[1].set_xlabel('updates per task'); ax[1].set_ylabel('decline D [pt]'); ax[1].axhline(0, color='k', lw=.5)
        fig.tight_layout(); fig.savefig(OUT / 'fig_budget_ladder.png', dpi=120)
    except Exception as e:
        print('figure skipped:', e)
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
