"""Aggregation, verdicts and figure for `width_sink_clamp_0909`.

Implements 追補 1 A4 (labels), 追補 2 B2 (the G5 testability gate) and
追補 3 C3 (b_B measured on star_sd, not on the 8-perm between_inv).
Windows are fixed: late = t61..100, held = t91..100.  Labels need 3/3 seeds.
"""
from pathlib import Path
import csv, json
import numpy as np
from src import boundary_gradient_0908 as G

ROOT = G.ROOT
OUT = ROOT / 'results/width_sink_clamp_0909'
ARMS = ['LR', 'SNA', 'ELU1']
CLAMPS = ['ref', 'wclamp', 'mclamp', 'dclamp']
SEEDS = [0, 1, 2]
LATE = (61, 100)
HELD = (91, 100)
G5_FLOOR = 0.005          # 追補 2 B2: a ratio needs |b_ref| >= this per task
ACC_DROP = 0.05           # 追補 1 A4 G3
CE_FRAC = 0.90
R_LO, R_HI = 0.25, 0.75   # sink / width ratio thresholds
EDGE_EQ, EDGE_EFF = 0.03, 0.05
COL = {'ref': '#0072B2', 'wclamp': '#D55E00', 'mclamp': '#009E73', 'dclamp': '#E69F00'}


def read(arm, seed):
    f = OUT / f'{arm}_none_s{seed}_rows.csv'
    rows = list(csv.DictReader(open(f)))
    for r in rows:
        for k, v in list(r.items()):
            if k in ('arm', 'iv', 'clamp'):
                continue
            r[k] = float(v) if v not in ('', None) else np.nan
    return rows


def slope(rows, key, lo, hi):
    t = np.array([r['task'] for r in rows if lo <= r['task'] <= hi])
    y = np.array([r[key] for r in rows if lo <= r['task'] <= hi])
    ok = np.isfinite(y)
    return float(np.polyfit(t[ok], y[ok], 1)[0]) if ok.sum() > 2 else np.nan


def series(rows, clamp, step=625):
    return sorted([r for r in rows if r['clamp'] == clamp and r['step'] == step], key=lambda r: r['task'])


def per_seed(arm, seed):
    rows = read(arm, seed)
    ends = {c: series(rows, c) for c in CLAMPS}
    starts = {c: {r['task']: r['ce_probe'] for r in series(rows, c, 20)} for c in CLAMPS}
    z20 = [r['zbar_inv'] for r in ends['ref'] if r['task'] == 20][0]
    out = {}
    for c in CLAMPS:
        e = ends[c]
        late = [r for r in e if LATE[0] <= r['task'] <= LATE[1]]
        held = [r for r in e if HELD[0] <= r['task'] <= HELD[1]]
        ce_ok = [1. for r in late if np.isfinite(starts[c].get(r['task'], np.nan))
                 and starts[c][r['task']] - r['ce_probe'] > 0]
        out[c] = dict(
            b_Z=slope(e, 'zbar_inv', *LATE), b_S=slope(e, 'sigma_inv', *LATE),
            b_B=slope(e, 'star_sd', *LATE),
            b_B_between=slope([dict(task=r['task'], y=np.sqrt(r['between_inv'])) for r in e], 'y', *LATE),
            p_late=float(np.mean([r['pos_frac'] for r in late])),
            acc_med=float(np.median([r['acc'] for r in late])),
            ce_frac=len(ce_ok) / max(1, len(late)),
            sigma_late=float(np.mean([r['sigma_inv'] for r in late])),
            between_late=float(np.mean([np.sqrt(r['between_inv']) for r in late])),
            star_sd_late=float(np.mean([r['star_sd'] for r in late])),
            w2col_late=float(np.mean([r['w2col'] for r in late])),
            pressure=float(e[-1]['pressure']),
            z_held=float(np.mean([r['zbar_inv'] for r in held])), z20=z20,
            eps_max_late=float(np.mean([r['eps_max'] for r in late])))
    return out


def label3(vals, rule):
    """A label only when every seed is testable and they all agree."""
    got = [rule(v) for v in vals]
    if any(g is None for g in got):
        return 'NOT_TESTABLE_REF_FLAT'
    return got[0] if len(set(got)) == 1 else 'MIXED'


def main():
    seedrows, verdicts, validation = [], [], []
    per = {}
    for arm in ARMS:
        try:
            per[arm] = {s: per_seed(arm, s) for s in SEEDS}
        except FileNotFoundError as e:
            print('missing', e)
            continue
        for s in SEEDS:
            for c in CLAMPS:
                seedrows.append(dict(arm=arm, seed=s, clamp=c, **{k: v for k, v in per[arm][s][c].items()}))
        ref = {s: per[arm][s]['ref'] for s in SEEDS}
        # --- G3 / G4 gates -------------------------------------------------
        gate = {}
        for c in CLAMPS:
            g3 = [per[arm][s][c]['acc_med'] >= ref[s]['acc_med'] - ACC_DROP
                  and per[arm][s][c]['ce_frac'] >= CE_FRAC for s in SEEDS]
            g4 = [True] * 3
            if c == 'dclamp':
                g4 = [abs(per[arm][s][c]['z_held'] - per[arm][s][c]['z20'])
                      < R_LO * abs(ref[s]['z_held'] - ref[s]['z20']) for s in SEEDS]
            gate[c] = dict(g3=g3, g4=g4, ok=[a and b for a, b in zip(g3, g4)])
        # --- ratios and labels ---------------------------------------------
        def ratio(c, key, floor=G5_FLOOR):
            out = []
            for s in SEEDS:
                den = ref[s][key]
                out.append(np.nan if abs(den) < floor else per[arm][s][c][key] / den)
            return out

        def sink_rule(x):
            if not np.isfinite(x):
                return None
            return 'STOPS' if x < R_LO else ('SURVIVES' if x > R_HI else 'PARTIAL')

        row = dict(arm=arm)
        for c, tag in [('wclamp', 'W'), ('mclamp', 'M')]:
            R = ratio(c, 'b_Z')
            R = [r if gate[c]['ok'][i] else np.nan for i, r in enumerate(R)]
            lab = label3(R, sink_rule)
            row[f'R_{tag}'] = ';'.join(f'{r:+.3f}' if np.isfinite(r) else 'NA' for r in R)
            row[f'sink_{tag}'] = ('NOT_TESTABLE_LEARNING_BROKEN' if not all(gate[c]['ok'])
                                 else (f'SINK_{lab}_{tag}' if lab not in ('MIXED', 'NOT_TESTABLE_REF_FLAT') else lab))
        gw = ratio('dclamp', 'b_S')
        gb = ratio('dclamp', 'b_B')
        gwb = [(a, b) for a, b in zip(gw, gb)]
        if not all(gate['dclamp']['ok']):
            wl = 'NOT_TESTABLE_LEARNING_BROKEN' if not all(gate['dclamp']['g3']) else 'NOT_TESTABLE_DEPTH_NOT_HELD'
        else:
            def wrule(ab):
                a, b = ab
                if not (np.isfinite(a) and np.isfinite(b)):
                    return None
                if a > R_HI and b > R_HI:
                    return 'WIDTH_GROWS_REGARDLESS'
                if a < R_LO and b < R_LO:
                    return 'WIDTH_NEEDS_SINK'
                if a > R_HI:
                    return 'WIDTH_GROWS_WITHIN_ONLY'
                if b > R_HI:
                    return 'WIDTH_GROWS_BETWEEN_ONLY'
                return 'WIDTH_PARTIAL'
            wl = label3(gwb, wrule)
        row['G_within'] = ';'.join(f'{x:+.3f}' if np.isfinite(x) else 'NA' for x in gw)
        row['G_between'] = ';'.join(f'{x:+.3f}' if np.isfinite(x) else 'NA' for x in gb)
        row['width'] = wl
        for c in CLAMPS[1:]:
            dp = [per[arm][s][c]['p_late'] - ref[s]['p_late'] for s in SEEDS]

            def erule(d):
                if abs(d) < EDGE_EQ:
                    return 'EDGE_HELD'
                if d < -EDGE_EFF:
                    return 'EDGE_DETACHES_DOWN'
                if d > EDGE_EFF:
                    return 'EDGE_LIFTS'
                return 'EDGE_PARTIAL'
            row[f'dp_{c}'] = ';'.join(f'{d:+.3f}' for d in dp)
            row[f'edge_{c}'] = label3(dp, erule)
        row['b_Z_ref'] = ';'.join(f'{ref[s]["b_Z"]:+.5f}' for s in SEEDS)
        row['b_S_ref'] = ';'.join(f'{ref[s]["b_S"]:+.5f}' for s in SEEDS)
        row['b_B_ref'] = ';'.join(f'{ref[s]["b_B"]:+.5f}' for s in SEEDS)
        row['g3_ok'] = ';'.join(''.join('1' if gate[c]['ok'][s] else '0' for s in SEEDS) for c in CLAMPS)
        verdicts.append(row)
        for s in SEEDS:
            p = ROOT / 'results/width_sink_clamp_0909' / f'{arm}_none_s{s}_provenance.json'
            if p.exists():
                ck = json.loads(p.read_text())['checks']
                validation.append(dict(arm=arm, seed=s, **{k: v for k, v in ck.items() if not isinstance(v, (dict, list))}))
    OUT.mkdir(parents=True, exist_ok=True)
    G.B.csvwrite(OUT / 'seed_verdict.csv', seedrows)
    G.B.csvwrite(OUT / 'verdict.csv', verdicts)
    G.B.csvwrite(OUT / 'validation.csv', validation)
    figure(per)
    summary(per, verdicts)
    for v in verdicts:
        print(v['arm'], v.get('sink_W'), v.get('sink_M'), v.get('width'),
              [v.get(f'edge_{c}') for c in CLAMPS[1:]])


def figure(per):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    arms = [a for a in ARMS if a in per]
    fig, ax = plt.subplots(len(arms), 4, figsize=(15, 3.2 * len(arms)), squeeze=False)
    for i, arm in enumerate(arms):
        for c in CLAMPS:
            for s in SEEDS:
                rows = read(arm, s)
                e = series(rows, c)
                t = [r['task'] for r in e]
                kw = dict(color=COL[c], lw=1.3 if s == 0 else .7, alpha=1 if s == 0 else .4)
                ax[i][0].plot(t, [r['zbar_inv'] for r in e], label=c if s == 0 else None, **kw)
                ax[i][1].plot(t, [r['sigma_inv'] for r in e], **kw)
                ax[i][1].plot(t, [r['star_sd'] for r in e], ls='--', **kw)
                ax[i][2].plot(t, [r['pos_frac'] for r in e], **kw)
                ax[i][3].plot(t, [r['acc'] for r in e], **kw)
        ax[i][0].set_ylabel(f'{arm}\nz̄ (inv)')
        for j, ti in enumerate(['sinking  z̄_inv', 'width  σ_inv (solid) / star_sd (dashed)',
                                'occupancy  p⁺', 'test accuracy']):
            ax[i][j].set_title(ti if i == 0 else '', loc='left', fontsize=10)
            ax[i][j].axvline(61, color='#bbb', lw=.7)
            ax[i][j].spines[['top', 'right']].set_visible(False)
            ax[i][j].grid(axis='y', color='#eee', lw=.6)
            if i == len(arms) - 1:
                ax[i][j].set_xlabel('task')
        ax[i][0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / 'fig_clamp.png', dpi=150)


def md(rows, cols):
    out = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    for r in rows:
        out.append('| ' + ' | '.join(str(r.get(c, '')) for c in cols) + ' |')
    return '\n'.join(out)


def summary(per, verdicts):
    S = ['# width_sink_clamp_0909 summary', '',
         'spec: `specs/spec_width_sink_clamp_0909.md`（追補 1/2/3）。窓は late = t61–100 固定、'
         'held = t91–100。ラベルは可検定な 3 seed が 3/3 一致したときのみ。CI・有意差なし。', '',
         '判定量: b_Z = z̄_inv の late OLS 傾き、b_S = σ_inv、b_B = **star_sd**（追補 3）。'
         '比の分母が |b_ref| < 0.005/task の seed は `NOT_TESTABLE_REF_FLAT`（G5）。',
         'G3 = 精度中央値が ref −5pt 以内かつ CE 改善が late の 90% 以上のタスクで正。'
         'G4 = dclamp の残留沈下が ref の 0.25 倍未満。CE の起点は更新 20（更新 0 は測っていない）。', '',
         '## 事前登録の判定', '',
         md(verdicts, ['arm', 'sink_W', 'sink_M', 'width', 'edge_wclamp', 'edge_mclamp', 'edge_dclamp']), '',
         '## 連続量（seed 別・; 区切り）', '',
         md(verdicts, ['arm', 'R_W', 'R_M', 'G_within', 'G_between', 'dp_wclamp', 'dp_mclamp', 'dp_dclamp']), '',
         '## 参照腕の late 傾き（G5 の分母）', '',
         md(verdicts, ['arm', 'b_Z_ref', 'b_S_ref', 'b_B_ref', 'g3_ok']), '',
         '`g3_ok` は ref/wclamp/mclamp/dclamp の順に seed0,1,2 の G3+G4 通過を 1/0 で。', '',
         '## 副測定（ラベル無し）', '']
    for arm in [a for a in ARMS if a in per]:
        for c in CLAMPS:
            v = [per[arm][s][c] for s in SEEDS]
            S.append(f'- `{arm}/{c}`: σ_inv late ' + '/'.join(f'{x["sigma_late"]:.3f}' for x in v)
                     + ' · star_sd late ' + '/'.join(f'{x["star_sd_late"]:.3f}' for x in v)
                     + ' · √between late ' + '/'.join(f'{x["between_late"]:.3f}' for x in v)
                     + ' · ‖W2col‖ ' + '/'.join(f'{x["w2col_late"]:.3f}' for x in v)
                     + ' · 圧力 ' + '/'.join(f'{x["pressure"]:+.3g}' for x in v)
                     + ' · 精度 ' + '/'.join(f'{x["acc_med"]:.3f}' for x in v)
                     + ' · eps_max ' + '/'.join(f'{x["eps_max_late"]:.2f}' for x in v))
    S += ['', '図: `fig_clamp.png`（行 = 活性化、列 = 沈下 / 幅 / 占有率 / 精度、色 = クランプ、'
              '縦線 = late 窓の開始）。', '', '検算値は `validation.csv`、seed 別の連続量は `seed_verdict.csv`。']
    (OUT / 'summary.md').write_text('\n'.join(S))


if __name__ == '__main__':
    main()
