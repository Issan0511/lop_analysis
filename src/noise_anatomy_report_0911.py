"""Verdicts for spec_noise_anatomy_0911 (F0-F5).  Refuses an incomplete run unless --partial."""
from pathlib import Path
import argparse, csv, json
import numpy as np
from src import gate_shape_report_0911 as R

ROOT = R.ROOT
OUT = ROOT / 'results/noise_anatomy_0911'
ARMS = ['N0', 'N2', 'N2on', 'N2off', 'N2post', 'N2step']
SEEDS = [0, 1, 2]
BASE = (16, 20)
LATE = (101, 120)
EARLY = (21, 40)
IMM = (61, 65)
SWITCH = 61
CE_KEYS = ['ce0', 'ce20', 'ce100', 'ce300', 'ce625']


def load(arm, seed):
    rows = R.read_rows(OUT / f'{arm}_s{seed}_rows.csv')
    prov = json.load(open(OUT / f'{arm}_s{seed}_provenance.json'))
    return sorted(rows, key=lambda r: r['task']), prov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=None); ap.add_argument('--partial', action='store_true')
    a = ap.parse_args()
    global OUT
    if a.src:
        OUT = Path(a.src)
    have = {f.name.replace('_rows.csv', '') for f in OUT.glob('*_rows.csv')}
    want = {f'{arm}_s{s}' for arm in ARMS for s in SEEDS}
    gap = sorted(want - have)
    if gap and not a.partial:
        print(f'REFUSING: {len(gap)}/{len(want)} runs missing {gap[:6]}'); raise SystemExit(2)
    data = {}
    for arm in ARMS:
        for s in SEEDS:
            try:
                data[(arm, s)] = load(arm, s)
            except FileNotFoundError:
                pass
    arms = [a_ for a_ in ARMS if any(k[0] == a_ for k in data)]

    def acc_series(arm, s):
        return {r['task']: r['acc'] for r in data[(arm, s)][0]}

    def wmed(arm, s, key, lo, hi):
        return float(np.median([r[key] for r in data[(arm, s)][0] if lo <= r['task'] <= hi and np.isfinite(r[key])]))

    def L(arm, s):
        return (wmed(arm, s, 'acc', *BASE) - wmed(arm, s, 'acc', *LATE)) * 100

    def gap_win(arm, s, lo, hi):
        a0, a1 = acc_series('N0', s), acc_series(arm, s)
        return float(np.median([(a1[t] - a0[t]) * 100 for t in range(lo, hi + 1) if t in a0 and t in a1]))

    V, lines = {}, ['# noise_anatomy_0911 summary\n',
                    'spec: `specs/spec_noise_anatomy_0911.md`（事前登録 commit `2dda57a`・判定値は未読で起動）\n']
    # checks
    lines.append('\n## 0. 検査\n\n| arm | G1 錨 | G1 maxabs (s0/s1/s2) | 配列数 | 注入 c / post RMS | f_diff (late, s0/s1/s2) |\n|---|---|---|---|---|---|')
    for arm in arms:
        ps = [data[(arm, s)][1]['checks'] for s in SEEDS if (arm, s) in data]
        inj = (f"{ps[0].get('g3_inj_lo', float('nan')):.2f}–{ps[0].get('g3_inj_hi', float('nan')):.2f}" if 'g3_inj_lo' in ps[0]
               else f"post {ps[0].get('g3_post_rms_lo', float('nan')):.3f}–{ps[0].get('g3_post_rms_hi', float('nan')):.3f}" if 'g3_post_rms_lo' in ps[0] else '—')
        fd = ' / '.join(f"{wmed(arm, s, 'f_diff', *LATE):.2f}" for s in SEEDS if (arm, s) in data)
        lines.append(f"| {arm} | {ps[0]['g1_anchor']} | {' / '.join(str(p.get('g1_units_maxabs')) for p in ps)} | {ps[0].get('g1_units_compared')} | {inj} | {fd} |")
    # per-arm table
    lines.append('\n## 1. 腕ごとの量（seed 中央値）\n\n| arm | L [pt] | acc t1–5 | acc late | f_diff | drift2 | diff2 | ρ | κ2 | N | ce0 late | ce625 late | ‖W2‖ | ‖W3‖ |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    M = {}
    for arm in arms:
        m = {k: float(np.median([wmed(arm, s, k, *LATE) for s in SEEDS if (arm, s) in data])) for k in
             ['acc', 'f_diff', 'drift2', 'diff2', 'rho_mean', 'kap2', 'cnorm', 'ce0', 'ce625', 'w2_fro', 'w3_fro']}
        m['L'] = float(np.median([L(arm, s) for s in SEEDS if (arm, s) in data]))
        m['acc1'] = float(np.median([wmed(arm, s, 'acc', 1, 5) for s in SEEDS if (arm, s) in data]))
        M[arm] = m
        lines.append(f"| {arm} | {m['L']:.2f} | {m['acc1']:.4f} | {m['acc']:.4f} | {m['f_diff']:.2f} | {m['drift2']:.4f} | {m['diff2']:.4f} | "
                     f"{m['rho_mean']:.1f} | {m['kap2']:.4f} | {m['cnorm']:.2f} | {m['ce0']:.3f} | {m['ce625']:.3f} | {m['w2_fro']:.2f} | {m['w3_fro']:.2f} |")

    ok = lambda a_: all((a_, s) in data for s in SEEDS)
    # ---- F0
    if ok('N2') and ok('N0'):
        fd = [wmed('N2', s, 'f_diff', *LATE) for s in SEEDS]
        fd0 = [wmed('N0', s, 'f_diff', *LATE) for s in SEEDS]
        V['F0_fdiff_N2'] = float(np.median(fd)); V['F0_fdiff_N0'] = float(np.median(fd0))
        f0 = all(0.60 <= x <= 0.90 for x in fd)
        ex = {}
        for arm in ('N2post', 'N2step'):
            if ok(arm):
                exc = [(wmed(arm, s, 'diff2', *LATE) - wmed('N0', s, 'diff2', *LATE)) /
                       (wmed('N2', s, 'diff2', *LATE) - wmed('N0', s, 'diff2', *LATE)) for s in SEEDS]
                ex[arm] = float(np.median(exc)); V[f'F0_excess_{arm}'] = ex[arm]
                f0 = f0 and all(0.7 <= x <= 1.3 for x in exc)
        D = [L('N2', s) - L('N0', s) for s in SEEDS]
        V['D_N2'] = float(np.median(D)); V['D_N2_seeds'] = [round(x, 2) for x in D]
        V['F0'] = 'MANIPULATION_OK' if (f0 and all(x >= 0.5 for x in D)) else 'NOT_TESTABLE'
        lines.append(f"\n- **F0**: f_diff(N2) = {' / '.join(f'{x:.2f}' for x in fd)}（N0 は {' / '.join(f'{x:.2f}' for x in fd0)}）、"
                     f"post/step の拡散超過分 ÷ N2 の超過分 = {ex}、D_N2 = {V['D_N2_seeds']} pt → `{V['F0']}`")
        lines.append("  （拡散超過分 = late 窓の diff2 から N0 の diff2 を引いたもの。N0 でも Adam の座標スケーリングで生勾配方向から外れる成分があるので、その分を差し引いて比べる）")
    # ---- F1 onset
    if ok('N2on') and ok('N0'):
        gi = [gap_win('N2on', s, *IMM) for s in SEEDS]; gl = [gap_win('N2on', s, *LATE) for s in SEEDS]
        rr = [l / i if (i < 0 and l < 0) else float('nan') for i, l in zip(gi, gl)]
        lab = ['DAMAGE_IMMEDIATE_ONLY' if r <= 1.3 else 'DAMAGE_ACCUMULATES' if r >= 1.8 else 'DAMAGE_BOTH' if np.isfinite(r) else 'NOT_TESTABLE' for r in rr]
        V['F1'] = lab[0] if len(set(lab)) == 1 else 'MIXED'; V['F1_g_imm'] = [round(x, 2) for x in gi]; V['F1_g_late'] = [round(x, 2) for x in gl]
        lines.append(f"- **F1** onset（N2on − N0）: t61–65 {V['F1_g_imm']} pt、t101–120 {V['F1_g_late']} pt、比 {[round(r, 2) for r in rr]} → `{V['F1']}`（seed 別 {lab}）")
    # ---- F2 offset
    if ok('N2off') and ok('N0'):
        taus, ends = [], []
        for s in SEEDS:
            a0, a1 = acc_series('N0', s), acc_series('N2off', s)
            tau = None
            for t in range(SWITCH, 120 - 4 + 1):
                g = float(np.median([(a1[u] - a0[u]) * 100 for u in range(t, t + 5)]))
                if g >= -0.3:
                    tau = t - (SWITCH - 1); break
            taus.append(tau); ends.append(gap_win('N2off', s, *LATE))
        lab = ['RECOVERS_FAST' if (t is not None and t <= 10) else 'RECOVERS_SLOW' if (t is not None and t <= 40) else 'DAMAGE_PERSISTS' for t in taus]
        V['F2'] = lab[0] if len(set(lab)) == 1 else 'MIXED'
        V['F2_tau'] = taus; V['F2_end'] = [round(x, 2) for x in ends]
        if all(e >= 0.3 for e in ends):
            V['F2'] += '+OVERSHOOTS'
        pre = [gap_win('N2off', s, 41, 60) for s in SEEDS]
        lines.append(f"- **F2** offset（N2off − N0）: 切替前 t41–60 {[round(x, 2) for x in pre]} pt、回復時間 τ = {taus}、末尾 t101–120 {V['F2_end']} pt → `{V['F2']}`（seed 別 {lab}）")
    # ---- F3
    if all(ok(a_) for a_ in ('N2post', 'N2step', 'N2', 'N0')) and V.get('F0') == 'MANIPULATION_OK':
        Dn = [L('N2', s) - L('N0', s) for s in SEEDS]
        Dp = [L('N2post', s) - L('N0', s) for s in SEEDS]; Ds = [L('N2step', s) - L('N0', s) for s in SEEDS]
        rp = [p / n for p, n in zip(Dp, Dn)]; rs = [q / n for q, n in zip(Ds, Dn)]
        if all(x >= 0.7 for x in rp) and all(x <= 0.4 for x in rs):
            V['F3'] = 'QUIET_COORD_KICKS'
        elif all(x >= 0.7 for x in rp) and all(x >= 0.7 for x in rs):
            V['F3'] = 'ANY_DIFFUSION_HURTS'
        elif all(x <= 0.4 for x in rp) and all(x <= 0.4 for x in rs):
            V['F3'] = 'DILUTION_ALSO_NEEDED'
        else:
            V['F3'] = 'F3_PARTIAL'
        V['F3_Dpost'] = [round(x, 2) for x in Dp]; V['F3_Dstep'] = [round(x, 2) for x in Ds]
        lines.append(f"- **F3**: D_post = {V['F3_Dpost']}、D_step = {V['F3_Dstep']}、D_N2 = {[round(x, 2) for x in Dn]} pt；比 post {[round(x, 2) for x in rp]}・step {[round(x, 2) for x in rs]} → `{V['F3']}`")
    # ---- F4 carry-over
    if ok('N2') and ok('N0'):
        dl = [wmed('N2', s, 'ce0', *LATE) - wmed('N0', s, 'ce0', *LATE) for s in SEEDS]
        de = [wmed('N2', s, 'ce0', *EARLY) - wmed('N0', s, 'ce0', *EARLY) for s in SEEDS]
        if all(x >= 0.05 for x in dl) and all(l > e for l, e in zip(dl, de)):
            V['F4'] = 'CARRYOVER_DAMAGED'
        elif all(abs(x) < 0.02 for x in dl):
            V['F4'] = 'CARRYOVER_INTACT'
        else:
            V['F4'] = 'F4_PARTIAL'
        V['F4_dce0_late'] = [round(x, 3) for x in dl]; V['F4_dce0_early'] = [round(x, 3) for x in de]
        lines.append(f"- **F4** 持ち越し CE@0（N2 − N0）: t21–40 {V['F4_dce0_early']}、t101–120 {V['F4_dce0_late']} nat → `{V['F4']}`")
        # ---- F5 within-task shape
        prof = {k: float(np.median([wmed('N2', s, k, *LATE) - wmed('N0', s, k, *LATE) for s in SEEDS])) for k in CE_KEYS}
        V.update({f'F5_d{k}': round(v, 3) for k, v in prof.items()})
        kmax = max(prof, key=lambda k: prof[k])
        if kmax == 'ce0' and prof['ce625'] < prof['ce0']:
            V['F5'] = 'STARTS_WORSE_CATCHES_UP'
        elif kmax in ('ce300', 'ce625'):
            V['F5'] = 'ERODES_LATE'
        else:
            V['F5'] = 'F5_FLAT'
        lines.append(f"- **F5** タスク内の CE 差（N2 − N0・late 窓）: " + '、'.join(f"@{k[2:]} {prof[k]:+.3f}" for k in CE_KEYS) + f" → `{V['F5']}`")
    # secondary
    lines.append('\n## 2. 副測定\n')
    if ok('N2off') and ok('N0'):
        lines.append('N2off の切替後の幅の再成長（‖W̃ᵢ‖・seed 中央値）: ' +
                     '、'.join(f"t{lo}–{hi} N2off {float(np.median([wmed('N2off', s, 'cnorm', lo, hi) for s in SEEDS])):.2f} / N0 {float(np.median([wmed('N0', s, 'cnorm', lo, hi) for s in SEEDS])):.2f}"
                               for lo, hi in ((56, 60), (61, 70), (81, 100), (101, 120))))
    if ok('N2on') and ok('N0'):
        lines.append('N2on の切替直後の CE@0 / CE@20（seed 中央値）: ' +
                     '、'.join(f"t{lo}–{hi} ce0 {float(np.median([wmed('N2on', s, 'ce0', lo, hi) for s in SEEDS])):.3f}/{float(np.median([wmed('N0', s, 'ce0', lo, hi) for s in SEEDS])):.3f} ce20 {float(np.median([wmed('N2on', s, 'ce20', lo, hi) for s in SEEDS])):.3f}/{float(np.median([wmed('N0', s, 'ce20', lo, hi) for s in SEEDS])):.3f}"
                               for lo, hi in ((56, 60), (61, 65), (66, 80), (101, 120))))
    (OUT / 'summary.md').write_text('\n'.join(lines) + '\n')
    with open(OUT / 'verdict.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(V.keys()), lineterminator='\n'); w.writeheader(); w.writerow({k: json.dumps(v) if isinstance(v, list) else v for k, v in V.items()})
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
        for arm, c in (('N2', 'C3'), ('N2on', 'C1'), ('N2off', 'C2'), ('N2post', 'C4'), ('N2step', 'C5')):
            if not ok(arm):
                continue
            ts = sorted(acc_series('N0', 0).keys())
            g = np.median([[(acc_series(arm, s)[t] - acc_series('N0', s)[t]) * 100 for t in ts] for s in SEEDS], axis=0)
            k = 5; gs = np.convolve(g, np.ones(k) / k, mode='valid')
            ax[0].plot(ts[k - 1:], gs, color=c, label=arm)
        ax[0].axhline(0, color='gray', lw=.8); ax[0].axvline(SWITCH, color='gray', ls='--', lw=.8)
        ax[0].set_xlabel('task'); ax[0].set_ylabel('acc − N0 [pt]  (5-task mean, seed median)'); ax[0].legend(fontsize=8)
        for arm, c in (('N0', 'k'), ('N2', 'C3')):
            if ok(arm):
                ax[1].plot([0, 20, 100, 300, 625], [M[arm][k] if k in M[arm] else float(np.median([wmed(arm, s, k, *LATE) for s in SEEDS])) for k in CE_KEYS], marker='o', color=c, label=arm)
        ax[1].set_xlabel('update within task'); ax[1].set_ylabel('probe CE (late window)'); ax[1].set_xscale('symlog'); ax[1].legend()
        fig.tight_layout(); fig.savefig(OUT / 'fig_noise_anatomy.png', dpi=120)
    except Exception as e:
        print('figure skipped:', e)
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
