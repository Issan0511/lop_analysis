"""Verdicts for spec_gate_persistence_0911: are the low-gate units fixed members or a
rotating set?  Analysis of the per-unit gate series of gate_shape_0911 (no new run)."""
from pathlib import Path
import csv, json
import numpy as np
from src import gate_shape_report_0911 as R

ROOT = R.ROOT
SRC = ROOT / 'results/gate_shape_0911'
OUT = ROOT / 'results/gate_persistence_0911'
ARMS = R.ARMS
SEEDS = R.SEEDS
LATE = (101, 120)
LAGS = (1, 5, 20, 40)
T0 = 41
TN = 120
ALPHA = {'SN02': .2, 'SN06': .6, 'SN15': 1.5}
FIXED, ROT = 0.75, 0.55


def series(units, key, prefix='ref'):
    return np.stack([units[f'{prefix}_{key}_t{t}'] for t in range(1, TN + 1)])     # (120, 100), index t-1


def bottoms(G):
    return [set(np.argsort(G[t], kind='stable')[:10].tolist()) for t in range(G.shape[0])]


def overlap(B, k, t0=T0, tn=TN):
    vals = [len(B[t - 1] & B[t + k - 1]) / 10. for t in range(t0, tn - k + 1)]
    return float(np.mean(vals)) if vals else np.nan


def longest_run(B, lo=21, hi=TN):
    best = np.zeros(100, dtype=int); cur = np.zeros(100, dtype=int)
    for t in range(lo, hi + 1):
        inb = np.zeros(100, dtype=bool); inb[list(B[t - 1])] = True
        cur = np.where(inb, cur + 1, 0); best = np.maximum(best, cur)
    return best


def per_seed(arm, seed):
    units = np.load(SRC / f'{arm}_s{seed}_units.npz')
    rows = R.read_rows(SRC / f'{arm}_s{seed}_rows.csv')
    G = series(units, 'gbar_i'); Z = series(units, 'zcur_i'); SD = series(units, 'sdcur_i')
    late = slice(LATE[0] - 1, LATE[1])
    out = dict(arm=arm, seed=seed, L=(R.win(rows, 'acc', *R.BASE) - R.win(rows, 'acc', *LATE)) * 100)
    out['var_gbar_late'] = float(np.max([G[t].var() for t in range(LATE[0] - 1, LATE[1])]))
    out['degenerate'] = out['var_gbar_late'] < 1e-10
    B = bottoms(G)
    for k in LAGS:
        out[f'O_{k}'] = overlap(B, k)
    out['Ndistinct'] = len(set().union(*B[T0 - 1:TN]))
    out['Nrun60'] = int((longest_run(B) >= 60).sum())
    ch = set.intersection(*B[LATE[0] - 1:LATE[1]])
    out['n_chronic'] = len(ch)
    # controls: shuffled membership (null ~ 0.10) and frozen membership (= 1.0)
    rng = np.random.default_rng(12345)
    Bs = [set(rng.permutation(100)[:10].tolist()) for _ in range(TN)]
    out['O_20_null'] = overlap(Bs, 20)
    Bf = [B[T0 - 1] for _ in range(TN)]
    out['O_20_fixed'] = overlap(Bf, 20)
    out['sizes_ok'] = all(len(b) == 10 for b in B)
    # secondary: |gbar| version and depth version, agreement with the gate version
    Ba = bottoms(np.abs(G)); Bz = bottoms(Z)
    out['O_20_abs'] = overlap(Ba, 20); out['O_20_depth'] = overlap(Bz, 20)
    # secondary: the permutation-invariant depth (8 reference perms, committed zbar_i) -- the
    # quantity the 9/10 post-hoc used; the current-perm gate above is re-drawn by each permutation
    Zi = series(units, 'zbar_i'); Bi = bottoms(Zi)
    out['O_20_depth_inv'] = overlap(Bi, 20); out['O_1_depth_inv'] = overlap(Bi, 1)
    out['jaccard_gate_depth'] = float(np.mean([len(B[t] & Bz[t]) / len(B[t] | Bz[t]) for t in range(LATE[0] - 1, LATE[1])]))
    # Snake phase
    if arm in ALPHA:
        a = ALPHA[arm]
        fr, dev = [], []
        for t in range(LATE[0] - 1, LATE[1]):
            mem = sorted(B[t]); th = 2 * a * Z[t][mem]
            fr.append(float(np.mean(np.sin(th) < -0.5)))
            dev.append(float(np.abs(G[t] - 1).max()))
        out['phase_frac'] = float(np.mean(fr)); out['max_dev_from_1'] = float(np.max(dev))
        out['alphaW_med'] = float(np.median(a * SD[late]))
    else:
        out['phase_frac'] = np.nan; out['max_dev_from_1'] = np.nan; out['alphaW_med'] = np.nan
    # secondary: growth of the bottom members vs the others (late), and depth returns
    CN = series(units, 'cnorm_i')
    gb, go = [], []
    for t in range(LATE[0] - 1, LATE[1] - 1):
        d = CN[t + 1] ** 2 - CN[t] ** 2
        m = np.zeros(100, dtype=bool); m[list(B[t])] = True
        gb.append(d[m].mean()); go.append(d[~m].mean())
    out['dcn2_bottom'] = float(np.mean(gb)); out['dcn2_other'] = float(np.mean(go))
    crossed = (Z[:100] < -8).any(0); ret = np.zeros(100, dtype=bool)
    for i in np.where(crossed)[0]:
        t1 = int(np.argmax(Z[:100, i] < -8)); ret[i] = bool((Z[t1:, i] > -6).any())
    out['cross8'] = int(crossed.sum()); out['return8'] = int(ret.sum())
    return out


def label_c1(vals):
    labs = []
    for v in vals:
        labs.append('FIXED_MEMBERS' if v >= FIXED else 'ROTATING' if v <= ROT else 'PARTIAL')
    return labs[0] if len(set(labs)) == 1 else 'MIXED'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    seedrows, missing = [], []
    for arm in ARMS:
        for s in SEEDS:
            try:
                seedrows.append(per_seed(arm, s))
            except FileNotFoundError:
                missing.append(f'{arm}_s{s}')
    arms = [a for a in ARMS if any(r['arm'] == a for r in seedrows)]
    keys = list(seedrows[0].keys())
    with open(OUT / 'seed_verdict.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys, lineterminator='\n'); w.writeheader(); w.writerows(seedrows)
    verdict, lines = {}, ['# gate_persistence_0911 summary\n',
                          'spec: `specs/spec_gate_persistence_0911.md`（事前登録 commit `1388d10`・走なし・`gate_shape_0911` の解析）\n']
    # controls
    nulls = [r['O_20_null'] for r in seedrows]; fixed = [r['O_20_fixed'] for r in seedrows]
    verdict['ctl_null_ok'] = all(0. <= v <= 0.2 for v in nulls); verdict['ctl_fixed_ok'] = all(v == 1. for v in fixed)
    verdict['ctl_null_range'] = f'{min(nulls):.3f}-{max(nulls):.3f}'; verdict['sizes_ok'] = all(r['sizes_ok'] for r in seedrows)
    lines.append(f"## 0. 対照\n\n- 帰無（毎タスク独立に 10 個を引く）の O_20: {verdict['ctl_null_range']}（帯 0–0.2 に {'入る' if verdict['ctl_null_ok'] else '**入らない**'}）\n- 固定対照の O_20 = 1.0: {verdict['ctl_fixed_ok']}\n- |B(t)| = 10 毎タスク: {verdict['sizes_ok']}\n")
    lines.append('## 1. 腕ごとの滞留（seed 別 O_20 と中央値）\n')
    lines.append('| arm | L | O_1 | O_5 | **O_20** (s0/s1/s2) | O_40 | Ndistinct | Nrun60 | chronic | O_20 |g| | O_20 depth(cur) | O_20 depth(inv) | Jaccard gate/depth | C1 |')
    lines.append('|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|')
    med = {}
    for a in arms:
        rs = [r for r in seedrows if r['arm'] == a]
        m = {k: float(np.median([r[k] for r in rs])) for k in keys if isinstance(rs[0][k], (int, float, np.floating, np.integer)) and k not in ('seed',)}
        med[a] = m
        if all(r['degenerate'] for r in rs):
            lab = 'DEGENERATE'
        else:
            lab = label_c1([r['O_20'] for r in rs])
        verdict[f'C1_{a}'] = lab
        lines.append(f"| {a} | {m['L']:.2f} | {m['O_1']:.2f} | {m['O_5']:.2f} | {' / '.join(f'{r['O_20']:.2f}' for r in rs)} | {m['O_40']:.2f} | {m['Ndistinct']:.0f} | {m['Nrun60']:.0f} | {m['n_chronic']:.0f} | {m['O_20_abs']:.2f} | {m['O_20_depth']:.2f} | {m['O_20_depth_inv']:.2f} | {m['jaccard_gate_depth']:.2f} | `{lab}` |")
    # C2
    ok = [a for a in arms if verdict[f'C1_{a}'] != 'DEGENERATE']
    rho = R.spearman([med[a]['O_20'] for a in ok], [med[a]['L'] for a in ok])
    verdict['rho_O20_L'] = rho; verdict['C2'] = 'PERSISTENCE_' + R.band(rho).replace('ORDERS', 'ORDERS_LOSS')
    lines.append(f"\n- **C2**: Spearman(O_20, L) = {rho:+.2f}（n={len(ok)}）→ `{verdict['C2']}`")
    # C3 Snake phase
    for a in ('SN02', 'SN06', 'SN15'):
        if a not in arms:
            continue
        rs = [r for r in seedrows if r['arm'] == a]
        labs = ['LOW_GATE_IS_PHASE' if r['phase_frac'] >= 0.8 else 'LOW_GATE_NOT_PHASE' if r['phase_frac'] <= 0.4 else 'PHASE_PARTIAL' for r in rs]
        lab = labs[0] if len(set(labs)) == 1 else 'MIXED'
        if all(r['max_dev_from_1'] <= 0.15 for r in rs):
            lab += '+NO_GATE_STRUCTURE'
        verdict[f'C3_{a}'] = lab
        lines.append(f"- **C3 {a}**: phase_frac = {' / '.join(f'{r['phase_frac']:.2f}' for r in rs)}、max|ḡ−1| = {' / '.join(f'{r['max_dev_from_1']:.2f}' for r in rs)}、αW 中央値 {med[a]['alphaW_med']:.2f} → `{lab}`")
    # C4 ELU1 vs LR
    if 'ELU1' in arms and 'LR' in arms:
        d = [next(r for r in seedrows if r['arm'] == 'ELU1' and r['seed'] == s)['O_20'] - next(r for r in seedrows if r['arm'] == 'LR' and r['seed'] == s)['O_20'] for s in SEEDS]
        dl = [next(r for r in seedrows if r['arm'] == 'ELU1' and r['seed'] == s)['L'] - next(r for r in seedrows if r['arm'] == 'LR' and r['seed'] == s)['L'] for s in SEEDS]
        if all(x >= 0.2 for x in d):
            verdict['C4'] = 'ELU_FIXED_LEAKY_ROTATES' + ('+FIXED_CORE_WITHOUT_EXTRA_LOSS' if all(x <= 0.5 for x in dl) else '+FIXED_CORE_WITH_EXTRA_LOSS')
        else:
            verdict['C4'] = 'NOT_SHOWN'
        lines.append(f"- **C4**: ΔO_20(ELU1−LR) = {[round(x, 2) for x in d]}、ΔL = {[round(x, 2) for x in dl]} pt → `{verdict['C4']}`")
    lines.append('\n## 2. 副測定\n\n| arm | Δ‖W̃ᵢ‖²/task 下位 10% | 他 | −8 を割った個体 | −6 へ戻った個体 |\n|---|---:|---:|---:|---:|')
    for a in arms:
        m = med[a]
        lines.append(f"| {a} | {m['dcn2_bottom']:.3f} | {m['dcn2_other']:.3f} | {m['cross8']:.0f} | {m['return8']:.0f} |")
    if missing:
        lines.append(f'\n**missing**: {missing}')
    (OUT / 'summary.md').write_text('\n'.join(lines) + '\n')
    with open(OUT / 'verdict.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(verdict.keys()), lineterminator='\n'); w.writeheader(); w.writerow(verdict)
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
        for a in arms:
            ax[0].plot(LAGS, [med[a][f'O_{k}'] for k in LAGS], marker='o', label=a)
            ax[1].scatter(med[a]['O_20'], med[a]['L']); ax[1].annotate(a, (med[a]['O_20'], med[a]['L']), fontsize=8, xytext=(3, 3), textcoords='offset points')
        ax[0].axhline(0.1, ls='--', c='gray'); ax[0].set_xlabel('lag k [tasks]'); ax[0].set_ylabel('O_k'); ax[0].legend(fontsize=7, ncol=2)
        ax[1].set_xlabel('O_20'); ax[1].set_ylabel('L [pt]'); ax[1].set_title(f'Spearman {rho:+.2f}')
        fig.tight_layout(); fig.savefig(OUT / 'fig_persistence.png', dpi=120)
    except Exception as e:
        print('figure skipped:', e)
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
