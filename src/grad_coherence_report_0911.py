"""Verdicts for spec_grad_coherence_0911 (E0-E7)."""
from pathlib import Path
import csv, json, math
import numpy as np
from src import gate_shape_report_0911 as R

ROOT = R.ROOT
OUT = ROOT / 'results/grad_coherence_0911'
SEEDS = [0, 1, 2]
BASE = (16, 20)
LATE = (101, 120)
FIRST = (1, 5)
ARMS = ['N0', 'N05', 'N1', 'N2', 'LRh', 'LRq', 'N1L23', 'N0w', 'N1w', 'SN0', 'SN1']
DOSE = ['N0', 'N05', 'N1', 'N2']
G3_CE_FRAC = 0.90


def per_seed(arm, seed):
    rows = R.read_rows(OUT / f'{arm}_s{seed}_rows.csv')
    prov = json.load(open(OUT / f'{arm}_s{seed}_provenance.json'))
    ck = prov['checks']
    cl = ck['clamp']          # every row of this arm carries its own clamp label
    W = lambda k, lo, hi: R.win(rows, k, lo, hi, clamp=cl)
    b = W('acc', *BASE)
    l = W('acc', *LATE)
    late = [r for r in rows if LATE[0] <= r['task'] <= LATE[1]]
    t = np.array([r['task'] for r in late]); a = np.array([r['acc'] for r in late])
    ce = [r for r in rows if LATE[0] <= r['task'] <= LATE[1] and np.isfinite(r.get('ce20', np.nan))]
    return dict(arm=arm, seed=seed, lr=ck['lr'], c=ck['noise_c'], clamp=ck['clamp'],
                L=(b - l) * 100, acc_base=b, level=l, acc1=W('acc', *FIRST),
                slope=float(np.polyfit(t, a, 1)[0]) * 100 * 100,
                kap2=W('kap2', *LATE), S2=W('S2', *LATE),
                rho=W('rho_mean', *LATE), graw2=W('graw2', *LATE),
                N=W('cnorm', *LATE), Cov=W('cov', *LATE),
                Gbar=W('gbar', *LATE), zbar=W('zbar_inv', *LATE),
                sigma=W('sigma_inv', *LATE), pos=W('pos_frac', *LATE),
                dead=W('hard_dead', *LATE), w2col=W('w2col', *LATE),
                ce_frac=float(np.mean([r['ce20'] - r['ce_probe'] > 0 for r in ce])) if ce else np.nan,
                g1=ck.get('g1_units_maxabs'), g1n=ck.get('g1_units_compared'),
                g2m=ck.get('g2_mean_rel'), g2v=ck.get('g2_var_rel'), inj_lo=ck.get('g3_inj_lo'),
                inj_hi=ck.get('g3_inj_hi'), untouched=ck.get('g3_untouched'))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=None, help='results directory (default: the real run)')
    ap.add_argument('--partial', action='store_true',
                    help='aggregate even though runs are missing. WITHOUT this the module refuses, '
                         'so that a "dry run" of the aggregator cannot silently read judgment '
                         'quantities off a half-finished run (2026-09-11 の開示).')
    a = ap.parse_args()
    global OUT
    if a.src:
        OUT = Path(a.src)
    have = {f.name.replace('_rows.csv', '') for f in OUT.glob('*_rows.csv')}
    want = {f'{arm}_s{s}' for arm in ARMS for s in SEEDS}
    gap = sorted(want - have)
    if gap and not a.partial:
        print(f'REFUSING: {len(gap)}/{len(want)} runs missing ({gap[:6]}{"..." if len(gap) > 6 else ""}).')
        print('Pass --partial only if you accept reading judgment quantities off an incomplete run.')
        raise SystemExit(2)
    seedrows, missing = [], []
    for arm in ARMS:
        for s in SEEDS:
            try:
                seedrows.append(per_seed(arm, s))
            except FileNotFoundError:
                missing.append(f'{arm}_s{s}')
    arms = [a for a in ARMS if any(r['arm'] == a for r in seedrows)]
    keys = list(seedrows[0].keys())
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / 'seed_verdict.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys, lineterminator='\n', restval=''); w.writeheader(); w.writerows(seedrows)

    def get(arm, s, k):
        return next((r[k] for r in seedrows if r['arm'] == arm and r['seed'] == s), np.nan)

    def med(arm, k):
        v = [r[k] for r in seedrows if r['arm'] == arm and r[k] is not None and np.isfinite(r[k])]
        return float(np.median(v)) if v else np.nan

    broken = [a for a in arms if any(get(a, s, 'ce_frac') < G3_CE_FRAC for s in SEEDS)]
    V, lines = {}, ['# grad_coherence_0911 summary\n',
                    'spec: `specs/spec_grad_coherence_0911.md`（事前登録 commit `3e0072a`・実装 `9227884`・判定値は未読で起動）\n',
                    f'scope: {len(arms)} 腕 × 3 seed・t1–120・base t16–20・late t101–120\n']
    lines.append('\n## 0. 検査\n\n| arm | c | lr | clamp | G1 maxabs | G2 mean/var | 注入の実測 c | 非対象 | CE 改善率 |\n|---|---:|---:|---|---|---|---|---:|---:|')
    for a in arms:
        r0 = next(r for r in seedrows if r['arm'] == a and r['seed'] == 0)
        inj = f"{r0['inj_lo']:.3f}–{r0['inj_hi']:.3f}" if r0['inj_lo'] is not None else '—'
        g2 = f"{r0['g2m']:.3f}/{r0['g2v']:.3f}" if r0['g2m'] is not None else '—'
        lines.append(f"| {a} | {r0['c']:.1f} | {r0['lr']:.6f} | {r0['clamp']} | {r0['g1']} ({r0['g1n']}) | {g2} | {inj} | "
                     f"{'0.0' if r0['untouched'] == 0. else r0['untouched']} | {min(get(a, s, 'ce_frac') for s in SEEDS):.2f} |")
    if broken:
        lines.append(f"\n**LEARNING_BROKEN**: {broken}")

    lines.append('\n## 1. 腕ごとの量（seed 中央値）\n\n| arm | L [pt] | acc(t1–5) | level | slope | κ2 | S² | ρ | graw² | N | Cov | mob | z̄ | dead |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for a in arms:
        lines.append(f"| {a} | {med(a,'L'):.2f} | {med(a,'acc1'):.4f} | {med(a,'level'):.4f} | {med(a,'slope'):+.2f} | "
                     f"{med(a,'kap2'):.4f} | {med(a,'S2'):.4f} | {med(a,'rho'):.1f} | {med(a,'graw2'):.4f} | "
                     f"{med(a,'N'):.2f} | {med(a,'Cov'):.3f} | {med(a,'Gbar'):.3f} | {med(a,'zbar'):.2f} | {med(a,'dead'):.0f} |")

    ok = lambda a: a in arms and a not in broken
    # ---- E0 manipulation check
    if all(ok(a) for a in DOSE):
        mono = all(get('N0', s, 'kap2') > get('N05', s, 'kap2') > get('N1', s, 'kap2') > get('N2', s, 'kap2') for s in SEEDS)
        ratio = float(np.median([get('N2', s, 'kap2') / get('N0', s, 'kap2') for s in SEEDS]))
        V['E0'] = 'MANIPULATION_OK' if (mono and ratio <= 0.6) else 'NOT_TESTABLE_NO_COHERENCE_CHANGE'
        V['E0_kap2_ratio'] = ratio
        V['E0_pred_ratio'] = 1 / math.sqrt(5)
        lines.append(f"\n- **E0**: κ2 は c に単調減 {mono}、κ2(N2)/κ2(N0) = **{ratio:.3f}**（機械的な予測 1/√5 = 0.447）→ `{V['E0']}`")
    # ---- E1 dose response
    if V.get('E0') == 'MANIPULATION_OK':
        up = all(get('N0', s, 'L') < get('N05', s, 'L') < get('N1', s, 'L') < get('N2', s, 'L') for s in SEEDS)
        dn = all(get('N0', s, 'L') > get('N05', s, 'L') > get('N1', s, 'L') > get('N2', s, 'L') for s in SEEDS)
        d20 = float(np.median([get('N2', s, 'L') - get('N0', s, 'L') for s in SEEDS]))
        V['E1'] = 'NOISE_COSTS_PLASTICITY' if up else 'NOISE_HELPS' if dn else ('NOISE_NEUTRAL' if abs(d20) < 0.5 else 'DOSE_PARTIAL')
        V['E1_dL'] = d20
        lines.append(f"- **E1**: L(N2) − L(N0) = **{d20:+.2f} pt**（seed 別 {[round(get('N2',s,'L')-get('N0',s,'L'),2) for s in SEEDS]}）、単調増 {up} → `{V['E1']}`")
        # ---- E2 plasticity or optimisation
        ds = float(np.median([get('N2', s, 'slope') - get('N0', s, 'slope') for s in SEEDS]))
        da = float(np.median([get('N2', s, 'acc1') - get('N0', s, 'acc1') for s in SEEDS])) * 100
        allds = all(get('N2', s, 'slope') - get('N0', s, 'slope') <= -0.3 for s in SEEDS)
        V['E2'] = 'PLASTICITY_SPECIFIC' if allds else ('OPTIMIZATION_ONLY' if (abs(ds) < 0.3 and da <= -1.0) else 'E2_PARTIAL')
        V['E2_dslope'] = ds; V['E2_dacc1'] = da
        lines.append(f"- **E2**: Δslope(N2 − N0) = **{ds:+.2f} pt/100task**、Δacc(t1–5) = {da:+.2f} pt → `{V['E2']}`")
        # ---- E3 width route
        r = float(np.median([get('N2', s, 'N') / get('N0', s, 'N') for s in SEEDS]))
        hi = all(get('N2', s, 'N') / get('N0', s, 'N') >= 1.15 for s in SEEDS)
        lo = all(get('N2', s, 'N') / get('N0', s, 'N') <= 0.95 for s in SEEDS)
        V['E3'] = 'NOISE_ROUTES_VIA_WIDTH' if hi else 'NOISE_SHRINKS_WIDTH' if lo else 'WIDTH_UNCHANGED'
        V['E3_ratio'] = r
        note = ''
        if V['E3'] == 'NOISE_SHRINKS_WIDTH' and V['E1'] == 'NOISE_COSTS_PLASTICITY':
            V['E3_extra'] = 'WIDTH_NOT_SUFFICIENT'; note = ' ＋ **`WIDTH_NOT_SUFFICIENT`**（幅が縮むのに損失が増える）'
        lines.append(f"- **E3**: N(N2)/N(N0) = **{r:.3f}** → `{V['E3']}`{note}")
    # ---- E4 wclamp
    if ok('N0w') and ok('N1w') and ok('N0') and ok('N1'):
        D = float(np.median([get('N1', s, 'L') - get('N0', s, 'L') for s in SEEDS]))
        Dw = float(np.median([get('N1w', s, 'L') - get('N0w', s, 'L') for s in SEEDS]))
        V['E4_D'] = D; V['E4_Dw'] = Dw
        if D < 0.5:
            V['E4'] = 'NOT_TESTABLE_NO_DAMAGE'
        else:
            V['E4'] = 'NOISE_DAMAGE_IS_WIDTH' if Dw <= 0.3 * D else 'NOISE_DAMAGE_SURVIVES_CLAMP' if Dw >= 0.7 * D else 'E4_PARTIAL'
        lines.append(f"- **E4**: 損傷 D(無クランプ) = **{D:+.2f} pt**、D_w(幅を止めた上で) = **{Dw:+.2f} pt**（比 {Dw/D:.2f}） → `{V['E4']}`")
    # ---- E5 coherence vs step size
    if all(ok(a) for a in ('N1', 'LRh', 'N2', 'LRq')):
        m1 = float(np.median([(get('N1', s, 'kap2') * get('N1', s, 'lr') ** 2) / (get('LRh', s, 'kap2') * get('LRh', s, 'lr') ** 2) for s in SEEDS]))
        m2 = float(np.median([(get('N2', s, 'kap2') * get('N2', s, 'lr') ** 2) / (get('LRq', s, 'kap2') * get('LRq', s, 'lr') ** 2) for s in SEEDS]))
        V['E5_match1'] = m1; V['E5_match2'] = m2
        d1 = float(np.median([get('N1', s, 'L') - get('LRh', s, 'L') for s in SEEDS]))
        d2 = float(np.median([get('N2', s, 'L') - get('LRq', s, 'L') for s in SEEDS]))
        V['E5_d1'] = d1; V['E5_d2'] = d2
        if not (0.85 <= m1 <= 1.15 and 0.85 <= m2 <= 1.15):
            V['E5'] = 'NOT_TESTABLE_STEP_NOT_MATCHED'
        else:
            both_hi = all(get('N1', s, 'L') - get('LRh', s, 'L') >= 0.5 for s in SEEDS) and all(get('N2', s, 'L') - get('LRq', s, 'L') >= 0.5 for s in SEEDS)
            both_lo = all(abs(get('N1', s, 'L') - get('LRh', s, 'L')) < 0.5 for s in SEEDS) and all(abs(get('N2', s, 'L') - get('LRq', s, 'L')) < 0.5 for s in SEEDS)
            neg = all(get('N1', s, 'L') - get('LRh', s, 'L') <= -0.5 for s in SEEDS)
            V['E5'] = 'COHERENCE_COSTS_BEYOND_STEP' if both_hi else 'STEP_SIZE_EXPLAINS_ALL' if both_lo else 'NOISE_BEATS_SMALL_LR' if neg else 'E5_PARTIAL'
        lines.append(f"- **E5**: 歩幅の照合 κ2·lr² 比 = {m1:.2f} / {m2:.2f}、ΔL(N1 − LRh) = **{d1:+.2f}**、ΔL(N2 − LRq) = **{d2:+.2f}** pt → `{V['E5']}`")
    # ---- E6 layer specificity
    if ok('N1L23') and ok('N1') and ok('N0'):
        D = float(np.median([get('N1', s, 'L') - get('N0', s, 'L') for s in SEEDS]))
        D23 = float(np.median([get('N1L23', s, 'L') - get('N0', s, 'L') for s in SEEDS]))
        V['E6_D23'] = D23
        V['E6'] = ('NOT_TESTABLE_NO_DAMAGE' if abs(D) < 0.5 else
                   'FIRST_LAYER_SPECIFIC' if D23 <= 0.3 * D else 'NOT_LAYER_SPECIFIC' if D23 >= 0.7 * D else 'E6_PARTIAL')
        lines.append(f"- **E6**: W2・W3 だけに注いだときの損傷 = **{D23:+.2f} pt**（W1 は {D:+.2f}） → `{V['E6']}`")
    # ---- E7 Snake
    if ok('SN0') and ok('SN1') and ok('N1') and ok('N0'):
        D = float(np.median([get('N1', s, 'L') - get('N0', s, 'L') for s in SEEDS]))
        Ds = float(np.median([get('SN1', s, 'L') - get('SN0', s, 'L') for s in SEEDS]))
        V['E7_Dsn'] = Ds
        V['E7'] = ('GENERALISES_TO_SNAKE' if (np.sign(Ds) == np.sign(D) and abs(Ds) >= 0.3 * abs(D))
                   else 'LEAKY_SPECIFIC' if abs(Ds) <= 0.3 * abs(D) else 'E7_PARTIAL')
        lines.append(f"- **E7**: Snake の損傷 = **{Ds:+.2f} pt**（leaky は {D:+.2f}） → `{V['E7']}`")
    # ---------------------------------------------------------------- post-hoc
    # E0 gates E1-E3 by registration.  Their quantities are still worth printing, so they
    # go here under an explicit 事後 heading; the registered labels above are untouched.
    lines.append('\n## 2. 事後・未登録（E0 のゲートで登録判定が出なかった量）\n')
    if all(ok(a) for a in DOSE):
        d20 = float(np.median([get('N2', s, 'L') - get('N0', s, 'L') for s in SEEDS]))
        up = all(get('N0', s, 'L') < get('N05', s, 'L') < get('N1', s, 'L') < get('N2', s, 'L') for s in SEEDS)
        ds = float(np.median([get('N2', s, 'slope') - get('N0', s, 'slope') for s in SEEDS]))
        da = float(np.median([get('N2', s, 'acc1') - get('N0', s, 'acc1') for s in SEEDS])) * 100
        rN = float(np.median([get('N2', s, 'N') / get('N0', s, 'N') for s in SEEDS]))
        rR = float(np.median([get('N2', s, 'rho') / get('N0', s, 'rho') for s in SEEDS]))
        V['post_E1_dL'] = d20; V['post_E1_monotone'] = up
        V['post_E2_dslope'] = ds; V['post_E2_dacc1'] = da
        V['post_E3_N_ratio'] = rN; V['post_rho_ratio'] = rR
        lines.append(f"- E1 相当: L(N2) − L(N0) = **{d20:+.2f} pt**、seed 別 {[round(get('N2',s,'L')-get('N0',s,'L'),2) for s in SEEDS]}、4 点単調増 **{up}**")
        lines.append(f"- E2 相当: Δslope = {ds:+.2f} pt/100task、Δacc(t1–5) = {da:+.2f} pt")
        lines.append(f"- E3 相当: N(N2)/N(N0) = **{rN:.3f}**、ρ(N2)/ρ(N0) = **{rR:.3f}**")
    # does the loss follow width or path persistence, across BOTH dials on the W1 gradient?
    w1arms = [a for a in ('N0', 'N05', 'N1', 'N2', 'LRh', 'LRq') if ok(a)]
    if len(w1arms) >= 5:
        Ls = [med(a, 'L') for a in w1arms]
        V['post_spearman_rho_L'] = R.spearman([med(a, 'rho') for a in w1arms], Ls)
        V['post_spearman_N_L'] = R.spearman([med(a, 'N') for a in w1arms], Ls)
        V['post_spearman_kap2_L'] = R.spearman([med(a, 'kap2') for a in w1arms], Ls)
        V['post_spearman_S2_L'] = R.spearman([med(a, 'S2') for a in w1arms], Ls)
        lines.append(f"\n**第 1 層の勾配に対する 2 つのダイヤル（ノイズ・lr）を合わせた {len(w1arms)} 腕で、損失は何に従うか**（事後・seed 中央値の Spearman）\n")
        lines.append('| 量 | Spearman vs L |\n|---|---:|')
        for k, lab in (('rho', 'ρ（タスク内の経路持続）'), ('N', '‖W̃ᵢ‖（幅）'), ('kap2', 'κ2（正規化歩幅）'), ('S2', 'S²（歩幅の予算）')):
            lines.append(f"| {lab} | {V[f'post_spearman_{k}_L']:+.2f} |")
        lines.append('\n| arm | ρ | N | L |\n|---|---:|---:|---:|')
        for a in sorted(w1arms, key=lambda x: -med(x, 'rho')):
            lines.append(f"| {a} | {med(a,'rho'):.1f} | {med(a,'N'):.2f} | {med(a,'L'):.2f} |")
    if missing:
        lines.append(f"\n**missing**: {missing}")
    (OUT / 'summary.md').write_text('\n'.join(lines) + '\n')
    with open(OUT / 'verdict.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(V.keys()), lineterminator='\n'); w.writeheader(); w.writerow(V)
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
        cs = [med(a, 'c') for a in DOSE if a in arms]
        for k, axi, lab in ((('L'), ax[0], 'L [pt] 目減り'), ('N', ax[1], '‖W̃ᵢ‖ (late)'), ('kap2', ax[2], 'κ2 (Adam 正規化歩幅)')):
            axi.plot(cs, [med(a, k) for a in DOSE if a in arms], marker='o', label='noise dial')
            for a, mk in (('LRh', 's'), ('LRq', '^')):
                if a in arms:
                    axi.axhline(med(a, k), ls='--', lw=.8, label=f'{a} (lr matched)')
            axi.set_xlabel('noise c'); axi.set_ylabel(lab); axi.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(OUT / 'fig_grad_coherence.png', dpi=120)
    except Exception as e:
        print('figure skipped:', e)
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
