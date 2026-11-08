"""追補 H.5〜H.7（1108）の採点: posthoc_h/ の JSON から量・判定・予測の採点・図を作る。走が欠けていても動く（途中の列）。

    cd <worktree> && PYTHONPATH=. python analysis/vit_long_1024/posthoc_2blk/judge_h.py [--raw DIR] [--out DIR] [--diag diag.json]

量（H.5）: spike = 更新 1〜5 の訓練損失の平均（track.loss）、T90（届かなければ 5,025）、A1000 = 更新 1,000 の訓練精度、
A_end = 最後の 20 評価点（最後の 500 更新）の訓練精度の平均。
判定（H.6）: s_T・s_A は F61(none)（t30 側は F31(none)）の 5 seed の標本 SD（ddof 1）。「4/5 以上」は使える seed 数 n に対し
ceil(0.8 n) 以上と読み、n < 5 のときは判定に「暫定」を付ける。
"""
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median, stdev

GROUPS = ('B60', 'F61', 'B30', 'F31')
CUTS = ('none', 'block1', 'ffn1', 'attn1', 'ffn0', 'attn0', 'ffn0ffn1', 'all')
EXTRA = ()
RECORD = 'B60z'                                        # 記録: B60 の −block1 を零の切除で（H.3、1108 修正）
CUT_JA = {'none': 'なし', 'block1': '−block1', 'ffn1': '−FFN1', 'attn1': '−attn1', 'ffn0': '−FFN0', 'attn0': '−attn0',
          'ffn0ffn1': '−FFN0−FFN1', 'all': '−all'}
SEEDS = (10252010, 10252011, 10252012, 10252013, 10252014)
UNREACHED = 5025
REPO = Path(__file__).resolve().parents[3]
RAW = Path.home() / 'Projects/obsidian-research-data/vit_long_1024_2blk_rlfix/posthoc_h'
OUTDIR = REPO / 'results/vit_long_1024_2blk_rlfix/addendum_h'


def need(n):
    return math.ceil(0.8 * n)


def sd(v):
    return stdev(v) if len(v) >= 2 else float('nan')


def metrics(j):
    acc = j['mem_acc']
    ee = j.get('eval_every', 25)
    k = 1000 // ee
    loss = (j.get('track') or {}).get('loss') or []
    return dict(spike=sum(loss[:5]) / 5 if len(loss) >= 5 else float('nan'),
                T90=float(j['T90']) if j.get('T90') is not None else float('nan'),
                A1000=acc[k] if len(acc) > k else float('nan'),
                A_end=sum(acc[-20:]) / 20 if len(acc) > 20 else float('nan'))


def load(raw):
    runs = {}
    for g in GROUPS + (RECORD,):
        for c in (CUTS if g != RECORD else ('block1',)):
            for s in SEEDS:
                f = raw / f'{g}_{c}_seed{s}.json'
                if not f.exists():
                    continue
                try:
                    j = json.loads(f.read_text())
                except json.JSONDecodeError:
                    continue
                if j.get('status') != 'ok':
                    continue
                runs[(g, c, s)] = metrics(j)
    return runs


def vals(runs, g, c, key):
    return {s: runs[(g, c, s)][key] for s in SEEDS if (g, c, s) in runs}


def band(runs, g, c):
    """中央値 ± 2 SD（T90 と A_end、5 seed）。"""
    T, A = list(vals(runs, g, c, 'T90').values()), list(vals(runs, g, c, 'A_end').values())
    if len(T) < 2:
        return None
    mt, st, ma, sa = median(T), sd(T), median(A), sd(A)
    return dict(n=len(T), T_med=mt, T_sd=st, T_lo=mt - 2 * st, T_hi=mt + 2 * st,
                A_med=ma, A_sd=sa, A_lo=ma - 2 * sa, A_hi=ma + 2 * sa)


def in_band(b, T, A):
    return b['T_lo'] <= T <= b['T_hi'] and b['A_lo'] <= A <= b['A_hi']


def yard(runs, g):
    T, A = list(vals(runs, g, 'none', 'T90').values()), list(vals(runs, g, 'none', 'A_end').values())
    return dict(group=g, n=len(T), s_T=sd(T), s_A=sd(A))


def jh1(runs, B, s):
    """B 群の切除ごとに seed 対の差（切除 − none）で NO_EFFECT / DEGRADES / IMPROVES / MIXED。"""
    out = {}
    for c in CUTS[1:] + EXTRA:
        if math.isnan(s['s_T']):
            out[c] = dict(label='NO_DATA', n=0)
            continue
        d = []
        for sd_ in SEEDS:
            if (B, c, sd_) in runs and (B, 'none', sd_) in runs:
                a, b = runs[(B, c, sd_)], runs[(B, 'none', sd_)]
                d.append(dict(seed=sd_, dT=a['T90'] - b['T90'], dA=a['A_end'] - b['A_end']))
        n = len(d)
        if n == 0:
            if c not in EXTRA:
                out[c] = dict(label='NO_DATA', n=0)
            continue
        tT, tA = 2 * s['s_T'], 2 * s['s_A']
        k_no = sum(abs(x['dT']) <= tT and abs(x['dA']) <= tA for x in d)
        k_dg = sum(x['dT'] > tT or x['dA'] < -tA for x in d)
        k_im = sum(x['dT'] < -tT or x['dA'] > tA for x in d)
        k = need(n)
        if k_no >= k:
            lab = 'NO_EFFECT'
        elif k_dg >= k and k_im < k:
            lab = 'DEGRADES'
        elif k_im >= k and k_dg < k:
            lab = 'IMPROVES'
        else:
            lab = 'MIXED'
        out[c] = dict(label=lab, provisional=n < 5, n=n, need=k, k_no_effect=k_no, k_degrades=k_dg, k_improves=k_im,
                      thr_T=tT, thr_A=tA, dT_median=median(x['dT'] for x in d), dA_median=median(x['dA'] for x in d),
                      n_unreached=sum(runs[(B, c, x['seed'])]['T90'] >= UNREACHED for x in d), diffs=d)
    return out


def jh2(runs):
    """B60(none) の各 seed が F61 のどの切除の帯に入るか。"""
    bands = {c: band(runs, 'F61', c) for c in CUTS + EXTRA}
    b60 = {s: runs[('B60', 'none', s)] for s in SEEDS if ('B60', 'none', s) in runs}
    n = len(b60)
    res = dict(bands=bands, n=n, counts={}, provisional=n < 5)
    if n == 0:
        res['label'] = 'NO_DATA'
        return res
    k = need(n)
    for c, b in bands.items():
        if b:
            res['counts'][c] = sum(in_band(b, r['T90'], r['A_end']) for r in b60.values())
    thin = res['counts'].get('block1')
    res['thin_band_available'] = thin is not None
    if thin is not None and thin >= k:
        res['label'] = 'THIN_MATCH_1BLOCK'
    elif res['counts'].get('none', 0) >= k:
        res['label'] = 'FULL_MATCH'
    else:
        others = [c for c in res['counts'] if res['counts'][c] >= k and c not in ('none', 'block1')]
        b_med = median(r['T90'] for r in b60.values())
        meds = {c: bands[c]['T_med'] for c in bands if bands[c]}
        excl = {c: m for c, m in meds.items() if c != 'all'}
        faster = {c: m for c, m in meds.items() if m < b_med}
        slower = {c: m for c, m in meds.items() if m > b_med}
        tied = [c for c, m in meds.items() if m == b_med]
        res.update(B60_T90_median=b_med, fresh_T90_medians=meds, tied=tied, other_band_matches=others)
        if excl and all(m < b_med for m in excl.values()):
            res['label'] = 'BELOW_ALL'
        elif meds and all(m > b_med for m in meds.values()):
            res['label'] = 'ABOVE_ALL'
        else:
            a = max(faster, key=faster.get) if faster else None
            b = min(slower, key=slower.get) if slower else None
            res['label'] = f'BETWEEN({CUT_JA.get(a, a)}, {CUT_JA.get(b, b)})'
            res['between'] = [a, b]
        if not res['thin_band_available']:
            res['note'] = 'F61(−block1) の走が無く THIN_MATCH_1BLOCK は判定できない'
    return res


def jh3(runs):
    b = band(runs, 'F31', 'none')
    b30 = {s: runs[('B30', 'none', s)] for s in SEEDS if ('B30', 'none', s) in runs}
    n = len(b30)
    res = dict(band=b, n=n, provisional=n < 5)
    if not b or n == 0:
        res['label'] = 'NO_DATA'
        return res
    k = need(n)
    k_above = sum(r['T90'] < b['T_lo'] for r in b30.values())
    k_in = sum(in_band(b, r['T90'], r['A_end']) for r in b30.values())
    k_below = sum(r['T90'] > b['T_hi'] for r in b30.values())
    res.update(k_above=k_above, k_in=k_in, k_below=k_below, need=k)
    res['label'] = 'ABOVE_FULL' if k_above >= k else 'FULL_MATCH' if k_in >= k else 'BELOW_FULL' if k_below >= k else 'UNDECIDED'
    return res


def same_cut_band(runs, B, Fr, c):
    b = band(runs, Fr, c)
    r = {s: runs[(B, c, s)] for s in SEEDS if (B, c, s) in runs}
    if not b or not r:
        return None
    return sum(in_band(b, x['T90'], x['A_end']) for x in r.values()), len(r)


def score(h1, h2, h3, h1_30, runs, diag):
    """H.7 の予測の採点。○ 当たり / × 外れ / — 判定できない。"""
    def mark(ok):
        return '—' if ok is None else ('○' if ok else '×')

    def lab(h, c):
        return h.get(c, {}).get('label')
    rows = []
    # 受け手
    rows.append(('受け手', 'J H1 −FFN1', 'NO_EFFECT', lab(h1, 'ffn1'), mark(None if lab(h1, 'ffn1') in (None, 'NO_DATA') else lab(h1, 'ffn1') == 'NO_EFFECT')))
    rows.append(('受け手', 'J H1 −attn1', 'NO_EFFECT', lab(h1, 'attn1'), mark(None if lab(h1, 'attn1') in (None, 'NO_DATA') else lab(h1, 'attn1') == 'NO_EFFECT')))
    f0 = h1.get('ffn0', {})
    ok = None if f0.get('label') in (None, 'NO_DATA') else (f0['label'] == 'DEGRADES' and f0['n_unreached'] * 2 > f0['n'])
    rows.append(('受け手', 'J H1 −FFN0', 'DEGRADES（T90 届かずが過半）', f"{f0.get('label')}（届かず {f0.get('n_unreached')}/{f0.get('n')}）", mark(ok)))
    a0 = h1.get('attn0', {})
    ok = None
    if a0.get('label') not in (None, 'NO_DATA') and f0.get('label') not in (None, 'NO_DATA'):
        ok = (a0['label'] == 'DEGRADES' and a0['dT_median'] >= f0['dT_median'] - a0['thr_T']
              and a0['dA_median'] <= f0['dA_median'] + a0['thr_A'])
    rows.append(('受け手', 'J H1 −attn0', 'DEGRADES（−FFN0 と同程度以上: ΔT 中央値 ≥ −FFN0 の − 2s_T、ΔA 中央値 ≤ −FFN0 の + 2s_A）',
                 f"{a0.get('label')}（ΔT 中央値 {a0.get('dT_median')}、−FFN0 {f0.get('dT_median')}）", mark(ok)))
    ff = h1.get('ffn0ffn1', {})
    Aff = list(vals(runs, 'B60', 'ffn0ffn1', 'A_end').values())
    ok = None if ff.get('label') in (None, 'NO_DATA') else (ff['label'] == 'DEGRADES' and 0.2 <= median(Aff) <= 0.6)
    rows.append(('受け手', 'J H1 −FFN0−FFN1', 'DEGRADES、A_end 0.2〜0.6（中央値）', f"{ff.get('label')}（A_end 中央値 {median(Aff) if Aff else None}）", mark(ok)))
    al = h1.get('all', {})
    Aal = list(vals(runs, 'B60', 'all', 'A_end').values())
    ok = None if al.get('label') in (None, 'NO_DATA') else (al['label'] == 'DEGRADES' and median(Aal) <= 0.15 and al['n_unreached'] == al['n'])
    rows.append(('受け手', 'J H1 −all', 'DEGRADES、A_end ≈ 0.1（中央値 ≤ 0.15）、T90 届かず 5/5',
                 f"{al.get('label')}（A_end 中央値 {median(Aal) if Aal else None}、届かず {al.get('n_unreached')}/{al.get('n')}）", mark(ok)))
    ok = None
    if h2.get('label') not in (None, 'NO_DATA'):
        bt = h2.get('bands', {})
        b_med = median(vals(runs, 'B60', 'none', 'T90').values())
        slow_ok = bt.get('ffn0ffn1') and b_med < bt['ffn0ffn1']['T_med']
        fast_ok = bt.get('block1') and b_med > bt['block1']['T_med']
        if not h2['label'].startswith('BETWEEN'):
            ok = False
        elif bt.get('block1') is None:
            ok = None if slow_ok else False
        else:
            ok = bool(slow_ok and fast_ok)
    rows.append(('受け手', 'J H2', 'BETWEEN（F61(−block1) より遅く F61(−FFN0−FFN1) より速い）', h2.get('label'), mark(ok)))
    rows.append(('受け手', 'J H3', 'ABOVE_FULL', h3.get('label'), mark(None if h3.get('label') == 'NO_DATA' else h3['label'] == 'ABOVE_FULL')))
    want = dict(ffn1={'NO_EFFECT'}, attn1={'MIXED', 'DEGRADES'}, ffn0={'DEGRADES'}, attn0={'DEGRADES'})
    for c, w in want.items():
        L = lab(h1_30, c)
        rows.append(('受け手', f'B30 J H1 {CUT_JA[c]}', '・'.join(sorted(w)), L, mark(None if L in (None, 'NO_DATA') else L in w)))
    if diag:
        d60 = [r for r in diag['rows'] if r['t_end'] == 60]
        for kind, key, preds in (
                ('零', 'd_acc_zero', (('FFN1', lambda v: v >= -0.05, 'Δ精度 ≥ −0.05'), ('attn1', lambda v: v >= -0.1, 'Δ精度 ≥ −0.1'),
                                      ('FFN0', lambda v: v <= -0.5, 'Δ精度 ≤ −0.5'), ('attn0', lambda v: v <= -0.5, 'Δ精度 ≤ −0.5'))),
                ('平均置換', 'd_acc_mean', (('FFN1', lambda v: v >= -0.1, 'Δ精度 ≥ −0.1'), ('attn1', lambda v: v >= -0.2, 'Δ精度 ≥ −0.2'),
                                           ('FFN0', lambda v: v <= -0.3, 'Δ精度 ≤ −0.3'), ('attn0', lambda v: v <= -0.3, 'Δ精度 ≤ −0.3')))):
            for b, cond, txt in preds:
                vs = [r['branches'][b].get(key, r['branches'][b].get('d_acc') if key == 'd_acc_zero' else None) for r in d60]
                if any(v is None for v in vs):
                    rows.append(('受け手', f'診断 t60 −{b}（{kind}）', f'{txt}（4/5 seed 以上と読む）', '—', '—'))
                    continue
                k = sum(cond(v) for v in vs)
                rows.append(('受け手', f'診断 t60 −{b}（{kind}）', f'{txt}（4/5 seed 以上と読む）', f'{k}/{len(d60)}', mark(k >= need(len(d60)))))
    # 送り手
    for c in ('ffn1', 'attn1'):
        L = lab(h1, c)
        rows.append(('送り手', f't60 {CUT_JA[c]}', '基準から 2SE 以内（J H1 NO_EFFECT と読む）', L, mark(None if L in (None, 'NO_DATA') else L == 'NO_EFFECT')))
    L = lab(h1, 'ffn0')
    rows.append(('送り手', 't60 −FFN0', '大きく落ちる（J H1 DEGRADES と読む）', L, mark(None if L in (None, 'NO_DATA') else L == 'DEGRADES')))
    ok = None if al.get('label') in (None, 'NO_DATA') else (median(Aal) <= 0.15 and al['n_unreached'] == al['n'])
    rows.append(('送り手', 't60 −all', '偶然のまま（A_end 中央値 ≤ 0.15・届かず 5/5 と読む）', f"A_end 中央値 {median(Aal) if Aal else None}", mark(ok)))
    L = h2.get('label')
    ok = None if L in (None, 'NO_DATA') or (not h2.get('thin_band_available') and L != 'FULL_MATCH') else L == 'THIN_MATCH_1BLOCK'
    rows.append(('送り手', 'J H2', 'THIN_MATCH_1BLOCK', L, mark(ok)))
    L = lab(h1_30, 'ffn1')
    others = [lab(h1_30, c) for c in ('block1', 'attn1', 'ffn0', 'attn0', 'ffn0ffn1', 'all')]
    ok = None if L in (None, 'NO_DATA') or any(o in (None, 'NO_DATA') for o in others) else (L == 'NO_EFFECT' and all(o != 'NO_EFFECT' for o in others))
    rows.append(('送り手', 't30 −FFN1 だけ基準と並ぶ', 'B30 J H1: −FFN1 NO_EFFECT、他の切除は NO_EFFECT でない', f"−FFN1 {L}、他 {others}", mark(ok)))
    cnt = {c: same_cut_band(runs, 'B30', 'F31', c) for c in ('block1', 'attn1', 'ffn0', 'attn0', 'ffn0ffn1', 'all')}
    ok = None if any(v is None for v in cnt.values()) else all(k >= need(n) for k, n in cnt.values())
    rows.append(('送り手', 't30 他は fresh と同じ', 'B30(c) が F31(c) の帯に 4/5 以上（c = −FFN1 以外の切除）',
                 '、'.join(f"{CUT_JA[c]} {v[0]}/{v[1]}" if v else f"{CUT_JA[c]} —" for c, v in cnt.items()), mark(ok)))
    return rows


def fmt(v, d=0):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return '—'
    return f'{v:,.{d}f}' if d == 0 else f'{v:.{d}f}'


def figure(runs, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    names = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams['font.family'] = ['Noto Sans CJK JP'] if 'Noto Sans CJK JP' in names else ['DejaVu Sans']
    cuts = list(CUTS)
    style = {'B60': dict(color='#D55E00', filled=True, dx=-0.27, lab='B60（課題 60 の末の体）'),
             'F61': dict(color='#009E73', filled=True, dx=-0.09, lab='F61（新品・課題 61）'),
             'B30': dict(color='#E69F00', filled=False, dx=0.09, lab='B30（課題 30 の末の体）'),
             'F31': dict(color='#0072B2', filled=False, dx=0.27, lab='F31（新品・課題 31）')}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), dpi=150)
    for ax, key, yl in ((axes[0], 'T90', 'T90（更新）'), (axes[1], 'A_end', 'A_end（最後の 500 更新の訓練精度）')):
        for g, st in style.items():
            for xi, c in enumerate(cuts):
                v = list(vals(runs, g, c, key).values())
                if not v:
                    continue
                x = xi + st['dx']
                ax.scatter([x] * len(v), v, s=22, facecolors=st['color'] if st['filled'] else 'none',
                           edgecolors=st['color'], linewidths=1.2, zorder=3, label=st['lab'] if xi == 0 or c == 'none' else None)
                ax.plot([x - 0.07, x + 0.07], [median(v)] * 2, color=st['color'], lw=2.2, zorder=4)
        if key == 'T90':
            ax.axhline(UNREACHED, color='0.4', lw=0.8, ls='--', zorder=1)
            ax.text(len(cuts) - 0.5, UNREACHED, '届かず', va='bottom', ha='right', fontsize=9, color='0.3')
        ax.set_xticks(range(len(cuts)))
        ax.set_xticklabels([CUT_JA[c] for c in cuts], fontsize=9)
        ax.set_xlabel('切除')
        ax.set_ylabel(yl)
        ax.grid(axis='y', color='0.9', lw=0.6)
        ax.spines[['top', 'right']].set_visible(False)
    h, l = axes[0].get_legend_handles_labels()
    seen = {}
    for hh, ll in zip(h, l):
        seen.setdefault(ll, hh)
    fig.legend(seen.values(), seen.keys(), loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=4, frameon=False, fontsize=9)
    fig.suptitle('追補 H 枝の切除（横棒は 5 seed の中央値、点は seed）', fontsize=11)
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def main(args):
    raw, out = Path(args.raw), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = load(raw)
    with open(out / 'rows_h.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['group', 'cut', 'seed', 'spike', 'T90', 'A1000', 'A_end'])
        for (g, c, s), m in sorted(runs.items(), key=lambda kv: ((GROUPS + (RECORD,)).index(kv[0][0]), CUTS.index(kv[0][1]), kv[0][2])):
            w.writerow([g, c, s] + [f"{m[k]:.6g}" for k in ('spike', 'T90', 'A1000', 'A_end')])
    y60, y30 = yard(runs, 'F61'), yard(runs, 'F31')
    h1, h1_30 = jh1(runs, 'B60', y60), jh1(runs, 'B30', y30)
    h2, h3 = jh2(runs), jh3(runs)
    diag = json.loads(Path(args.diag).read_text()) if args.diag and Path(args.diag).exists() else None
    sc = score(h1, h2, h3, h1_30, runs, diag)
    n_total = sum(k[0] != RECORD for k in runs)
    n_rec = sum(k[0] == RECORD for k in runs)
    rec_rows = []
    for s_ in SEEDS:
        a, z = runs.get(('B60', 'block1', s_)), runs.get((RECORD, 'block1', s_))
        if a or z:
            rec_rows.append(dict(seed=s_, T90_mean=a and a['T90'], A_end_mean=a and a['A_end'], T90_zero=z and z['T90'], A_end_zero=z and z['A_end']))
    (out / 'judge_h.json').write_text(json.dumps(dict(n_runs=n_total, yard_t60=y60, yard_t30=y30, JH1_B60=h1, JH2=h2, JH3=h3,
                                                      JH1_B30=h1_30, record_B60_block1_mean_vs_zero=rec_rows, n_record=n_rec, predictions=[dict(zip(('who', 'item', 'pred', 'got', 'mark'), r)) for r in sc]),
                                                 indent=1, ensure_ascii=False, default=float))
    L = ['# 追補 H 判定（枝の切除）', '', f'走の数: {n_total} / 160（H.3 の 8 切除）＋記録 B60z {n_rec} / 5（status ok のみ）。B60・B30 の切除は平均置換、F61・F31 は零。生成: `analysis/vit_long_1024/posthoc_2blk/judge_h.py`。', '',
         '## 量（5 seed の中央値 [最小〜最大]）', '',
         '| 群 | 切除 | n | spike | T90 | A1000 | A_end |', '|---|---|---|---|---|---|---|']
    for g in GROUPS:
        for c in CUTS + EXTRA:
            ms = [runs[(g, c, s)] for s in SEEDS if (g, c, s) in runs]
            if not ms:
                if c not in EXTRA:
                    L.append(f'| {g} | {CUT_JA[c]} | 0 | — | — | — | — |')
                continue
            cell = []
            for k, d in (('spike', 3), ('T90', 0), ('A1000', 3), ('A_end', 3)):
                v = [m[k] for m in ms]
                cell.append(f'{fmt(median(v), d)} [{fmt(min(v), d)}〜{fmt(max(v), d)}]')
            L.append(f'| {g} | {CUT_JA[c]} | {len(ms)} | ' + ' | '.join(cell) + ' |')
    L += ['', '## 物差し（新品 none の 5 seed の SD）', '', '| 側 | 群 | n | s_T | s_A |', '|---|---|---|---|---|',
          f"| t60 | F61 | {y60['n']} | {fmt(y60['s_T'], 1)} | {fmt(y60['s_A'], 4)} |",
          f"| t30 | F31 | {y30['n']} | {fmt(y30['s_T'], 1)} | {fmt(y30['s_A'], 4)} |", '']
    for title, h in (('J H1 枝ごとの効き（B60、seed 対の差 切除 − none）', h1), ('記録: B30 の J H1（物差しは F31(none)）', h1_30)):
        L += [f'## {title}', '', '| 切除 | 判定 | n | NO_EFFECT の数 | DEGRADES の数 | IMPROVES の数 | ΔT90 中央値 | ΔA_end 中央値 | 届かず |',
              '|---|---|---|---|---|---|---|---|---|']
        for c, r in h.items():
            if r.get('n', 0) == 0:
                L.append(f'| {CUT_JA[c]} | {r["label"]} | 0 | — | — | — | — | — | — |')
                continue
            L.append(f"| {CUT_JA[c]} | {r['label']}{'（暫定）' if r['provisional'] else ''} | {r['n']} | {r['k_no_effect']} | {r['k_degrades']} | "
                     f"{r['k_improves']} | {fmt(r['dT_median'])} | {fmt(r['dA_median'], 3)} | {r['n_unreached']} |")
        L.append('')
    L += ['## J H2 薄い新品と並ぶか（B60 none の各 seed が F61 の帯に入る数）', '',
          f"判定: **{h2.get('label')}**{'（暫定）' if h2.get('provisional') else ''}" + (f"。{h2['note']}" if h2.get('note') else ''), '',
          '| F61 の切除 | n | T90 中央値 | T90 帯 | A_end 中央値 | A_end 帯 | B60(none) が入る seed |', '|---|---|---|---|---|---|---|']
    for c, b in h2.get('bands', {}).items():
        if b:
            L.append(f"| {CUT_JA[c]} | {b['n']} | {fmt(b['T_med'])} | {fmt(b['T_lo'])}〜{fmt(b['T_hi'])} | {fmt(b['A_med'], 3)} | "
                     f"{fmt(b['A_lo'], 3)}〜{fmt(b['A_hi'], 3)} | {h2['counts'].get(c, '—')}/{h2['n']} |")
    if 'B60_T90_median' in h2:
        L += ['', f"B60(none) の T90 中央値 {fmt(h2['B60_T90_median'])}。同値の F61: {', '.join(CUT_JA[c] for c in h2['tied']) or 'なし'}。"]
    L += ['', '## 記録: B60 −block1 の平均置換と零の切除', '', '| seed | T90 平均置換 | T90 零 | A_end 平均置換 | A_end 零 |', '|---|---|---|---|---|']
    for r in rec_rows:
        L.append(f"| {r['seed']} | {fmt(r['T90_mean'])} | {fmt(r['T90_zero'])} | {fmt(r['A_end_mean'], 3)} | {fmt(r['A_end_zero'], 3)} |")
    if not rec_rows:
        L.append('| — | — | — | — | — |')
    L += ['', '## J H3 課題 30 の対照（B30 none 対 F31 none の帯）', '']
    if h3.get('band'):
        b = h3['band']
        L += [f"判定: **{h3['label']}**{'（暫定）' if h3.get('provisional') else ''}", '',
              '| F31(none) T90 帯 | A_end 帯 | 帯より速い | 帯の中 | 帯より遅い | n |', '|---|---|---|---|---|---|',
              f"| {fmt(b['T_lo'])}〜{fmt(b['T_hi'])} | {fmt(b['A_lo'], 3)}〜{fmt(b['A_hi'], 3)} | {h3.get('k_above', '—')} | {h3.get('k_in', '—')} | {h3.get('k_below', '—')} | {h3['n']} |"]
    else:
        L.append('判定: NO_DATA')
    L += ['', '## 予測の採点（H.7）', '', '○ 当たり・× 外れ・— 判定できない（走が欠けている）。', '',
          '| 予測者 | 項目 | 予測 | 結果 | 採点 |', '|---|---|---|---|---|']
    for r in sc:
        L.append('| ' + ' | '.join(str(x) for x in r) + ' |')
    (out / 'judge_h.md').write_text('\n'.join(L) + '\n')
    if runs:
        figure(runs, out / 'fig_h.png')
    print(f'{n_total} 走。J H2 {h2.get("label")}、J H3 {h3.get("label")}、J H1 ' + ', '.join(f'{c}:{r["label"]}' for c, r in h1.items()))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--raw', default=str(RAW))
    p.add_argument('--out', default=str(OUTDIR))
    p.add_argument('--diag', default=str(OUTDIR / 'diag.json'))
    main(p.parse_args())
