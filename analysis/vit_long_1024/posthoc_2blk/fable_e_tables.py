"""追補 E の判定（spec E.3）と予測の採点（E.5）の表を、本走の csv から作る（Fable、1111）。

読むもの: runs/<SL|R0L|SRMS|R0RMS>/seed*/per_task.csv、runs/<freshL|freshRMS>/seed*/task{25,50,75,100}/per_task.csv、
記録の J5 は <raw>/posthoc_j5/j5_<腕>_seed<seed>.json（fable_eg_j5.py の出力。無ければ —）。
書くもの: results/vit_long_1024_2blk_rlfix/addendum_e/judge_e.md・judge_e.json・e_t90.png。
規則は spec E.3 を走る前に固定したまま。T90 の未到達は 5,025。後期は課題 81〜100、早期は 2〜11。新品は同じ seed・同じ正規化の
課題 25・50・75・100 の 4 本。seed 10252010 の LN の組は追補 D の走（借りた機械）、ほかはすべて white-san。

    PYTHONPATH=$PWD python -m analysis.vit_long_1024.posthoc_2blk.fable_e_tables
"""
import csv
import json
import statistics as St
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RAW = Path.home() / 'Projects/obsidian-research-data/vit_long_1024_2blk_rlfix'
RUNS = RAW / 'runs'
REPO = Path(__file__).resolve().parents[3]
OUT = REPO / 'results/vit_long_1024_2blk_rlfix/addendum_e'
SEEDS = [10252010 + i for i in range(5)]
UNR = 5025
EARLY, LATE = range(2, 12), range(81, 101)
FRESH_TASKS = (25, 50, 75, 100)
NORMS = {'LN': ('SL', 'R0L', 'freshL'), 'RMS': ('SRMS', 'R0RMS', 'freshRMS')}
NAME = {'LN': 'LayerNorm', 'RMS': 'RMSNorm'}
J5_EARLY = (1, 2, 5)          # 課題末 1〜9 の中で保存した状態（追補 E の保存。seed 10252010 の LN の組は 1〜9 すべて保存）
INK = '#333333'
PALETTE = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#D55E00']   # seed の順に固定（追補 G の図と同じ）
plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'font.size': 9, 'axes.edgecolor': '#c9c8c3',
                     'axes.spines.top': False, 'axes.spines.right': False})


def reached(r):
    return str(r.get('reached90', '')).strip().lower() in ('1', 'true', '1.0')


def t90_run(arm, seed):
    d = RUNS / arm / f'seed{seed}'
    if not (d / 'done.json').exists():
        return None
    rows = list(csv.DictReader(open(d / 'per_task.csv')))
    t = {int(r['task']): (float(r['T90']) if reached(r) else UNR) for r in rows}
    return t if all(k in t for k in range(1, 101)) else None


def t90_fresh(arm, seed):
    out = []
    for task in FRESH_TASKS:
        p = RUNS / arm / f'seed{seed}' / f'task{task}' / 'per_task.csv'
        if not p.exists():
            return None
        r = list(csv.DictReader(open(p)))[-1]
        out.append(float(r['T90']) if reached(r) else UNR)
    return out


def fmt(v):
    if v is None:
        return '—'
    return f'{v:,.0f}' if abs(v - round(v)) < 1e-9 else f'{v:,.1f}'


def j5_early(arm, seed):
    p = RAW / 'posthoc_j5' / f'j5_{arm}_seed{seed}.json'
    if not p.exists():
        return None
    res = {r['task']: r for r in json.load(open(p))['results']}
    ts = [t for t in range(1, 10) if t in res]
    if not ts:
        return None
    return dict(tasks=ts, s_load=St.median(res[t]['s_load'] for t in ts),
                s_load_direct=St.median(res[t]['s_load_direct'] for t in ts),
                s_load_var=St.median(res[t]['s_load_var'] for t in ts),
                kappa=St.median(res[t]['kappa'] for t in ts), cos2=St.median(res[t]['cos2'] for t in ts),
                kappa_pos=sum(res[t]['kappa'] > 0 for t in ts), n=len(ts))


def main():
    OUT.mkdir(exist_ok=True)
    L = []
    md = L.append
    J = {}
    runs = {(a, s): t90_run(a, s) for n in NORMS for a in NORMS[n][:2] for s in SEEDS}
    fresh = {(n, s): t90_fresh(NORMS[n][2], s) for n in NORMS for s in SEEDS}
    md('# 追補 E の判定（spec E.3）と予測の採点（E.5）')
    md('')
    md('規則は spec E.3 を走る前に固定したまま。T90 の未到達は 5,025。後期は課題 81〜100、早期は 2〜11。新品は同じ seed・同じ正規化の課題 25・50・75・100 の 4 本。'
       'Q = 後期の T90 の中央値 ÷ 新品の中央値（記録）。seed 10252010 の LN の組は追補 D の走（借りた機械）、ほかはすべて white-san（E.2 の 1108 の記録）。')
    have = sum(1 for v in runs.values() if v)
    md(f'読んだ走: {have}/20、新品 {sum(len(v) for v in fresh.values() if v)}/40。')
    md('')
    # E.6 止める規則
    stop = [(a, s) for (a, s), t in runs.items() if t and a.startswith('R0') and any(t[k] >= UNR for k in range(2, 6))]
    md(f"止める規則（E.6、戻す腕が課題 2〜5 で 0.9 に届かない）に当たった走: {'なし' if not stop else '、'.join(f'{a} {s}' for a, s in stop)}。")
    md('')
    for n, (sh, re_, fa) in NORMS.items():
        md(f'## {NAME[n]} の組（共有 {sh}・戻す {re_}・新品 {fa}）')
        md('')
        md(f'| seed | 新品の T90 課題 25 / 50 / 75 / 100 | 新品の中央値 | {sh} 早期 | {sh} 後期 | Q | 遅い | {re_} 早期 | {re_} 後期 | 遅い | 共有が遅い |')
        md('|---|---|---|---|---|---|---|---|---|---|---|')
        cnt = {sh: 0, re_: 0, 'pair': 0}
        miss = 0
        rows_j = []
        for s in SEEDS:
            f, a, b = fresh[(n, s)], runs[(sh, s)], runs[(re_, s)]
            if not (f and a and b):
                miss += 1
                md(f'| {s} | — | — | — | — | — | — | — | — | — | — |')
                continue
            fm = St.median(f)
            ae, al = St.median(a[k] for k in EARLY), St.median(a[k] for k in LATE)
            be, bl = St.median(b[k] for k in EARLY), St.median(b[k] for k in LATE)
            sa, sb, pw = al > fm, bl > fm, al > bl
            cnt[sh] += sa; cnt[re_] += sb; cnt['pair'] += pw
            ua = sum(a[k] >= UNR for k in LATE)
            md(f"| {s} | {' / '.join(fmt(v) for v in f)} | {fmt(fm)} | {fmt(ae)} | {fmt(al)}{f'（未到達 {ua}）' if ua else ''} | {al / fm:.2f} | {'○' if sa else '×'} | {fmt(be)} | {fmt(bl)} | {'○' if sb else '×'} | {'○' if pw else '×'} |")
            rows_j.append(dict(seed=s, fresh=f, fresh_med=fm, shared_early=ae, shared_late=al, Q=al / fm, shared_slow=sa,
                               reset_early=be, reset_late=bl, reset_slow=sb, shared_worse=pw))
        n_ok = len(SEEDS) - miss
        lab = lambda k: 'LOSS' if cnt[k] >= 3 else ('NO_LOSS' if cnt[k] + miss < 3 else '保留')
        pair = 'SHARED_WORSE' if cnt['pair'] >= 4 else ('SHARED_BETTER' if cnt['pair'] + miss <= 1 else
                                                         ('MIXED' if cnt['pair'] + miss < 4 and cnt['pair'] > 1 else '保留'))
        ls_, lr_ = lab(sh), lab(re_)
        shield = {('LOSS', 'NO_LOSS'): 'PHASE1_BREAKS_SHIELD', ('NO_LOSS', 'NO_LOSS'): 'SHIELD_HOLDS',
                  ('LOSS', 'LOSS'): 'NO_SHIELD'}.get((ls_, lr_), '（規則に無い組）' if '保留' not in (ls_, lr_) else '保留')
        md('')
        md(f'**喪失 {sh}: 遅い {cnt[sh]}/{n_ok} → {ls_}**  ')
        md(f'**喪失 {re_}: 遅い {cnt[re_]}/{n_ok} → {lr_}**  ')
        md(f"**組: 共有が遅い {cnt['pair']}/{n_ok} → {pair}**  ")
        md(f'**盾（{NAME[n]}）: {shield}**')
        md('')
        J[n] = dict(rows=rows_j, loss={sh: ls_, re_: lr_}, counts=cnt, pair=pair, shield=shield, missing=miss)
    # 記録: J5
    md('## 記録（判定に入れない）: 装填の押しの向き（E.3 の最後の項、追補 D の J5 を RMSNorm にも当てる）')
    md('')
    md('早期 = 課題末 1〜9 の中で保存した状態（追補 E の走は課題 1・2・5、seed 10252010 の LN の組は 1〜9）。s_load は装填成分を通した押しが'
       '沈める側の生きた unit の割合（CLS 位置）、その直通の分と分散微分の分。LN は比較に置く（同じ道具）。')
    md('')
    md('| 腕 | seed | 課題末 | s_load の中央値 | 直通だけ | 分散微分だけ | κ > 0 | cos² の中央値 | ≥ 0.75 |')
    md('|---|---|---|---|---|---|---|---|---|')
    j5c = {}
    for arm in ('SRMS', 'SL'):
        c = 0
        nn = 0
        for s in SEEDS:
            e = j5_early(arm, s)
            if not e:
                md(f'| {arm} | {s} | — | — | — | — | — | — | — |')
                continue
            nn += 1
            c += e['s_load'] >= .75
            md(f"| {arm} | {s} | {'・'.join(map(str, e['tasks']))} | {e['s_load']:.3f} | {e['s_load_direct']:.3f} | {e['s_load_var']:.3f} | {e['kappa_pos']}/{e['n']} | {e['cos2']:.3f} | {'○' if e['s_load'] >= .75 else '×'} |")
        j5c[arm] = dict(count=c, n=nn)
    md('')
    md(f"SRMS で早期の s_load の中央値が 0.75 以上の seed: {j5c['SRMS']['count']}/{j5c['SRMS']['n']}（LN の SL は {j5c['SL']['count']}/{j5c['SL']['n']}）。")
    md('')
    # 予測
    hit = lambda b: '当たり' if b else '外れ'
    pe6 = (hit(j5c['SRMS']['count'] >= 3) if j5c['SRMS']['n'] == 5 else '未')
    P = [('PE1', 'SL は LOSS', 0.70, hit(J['LN']['loss']['SL'] == 'LOSS')),
         ('PE2', 'R0L は NO_LOSS', 0.85, hit(J['LN']['loss']['R0L'] == 'NO_LOSS')),
         ('PE3', 'SRMS は LOSS', 0.60, hit(J['RMS']['loss']['SRMS'] == 'LOSS')),
         ('PE4', 'R0RMS は NO_LOSS', 0.70, hit(J['RMS']['loss']['R0RMS'] == 'NO_LOSS')),
         ('PE5', 'RMSNorm の組は SHARED_WORSE', 0.65, hit(J['RMS']['pair'] == 'SHARED_WORSE')),
         ('PE6', 'SRMS の早期の s_load の中央値が 0.75 以上の seed が 3 以上', 0.30, pe6)]
    md('## 予測の採点（E.5、Claude。Issa の欄は空のまま）')
    md('')
    md('| 記号 | 予測 | 確率 | 結果 |')
    md('|---|---|---|---|')
    for k, txt, pr, r in P:
        md(f'| {k} | {txt} | {pr:.2f} | {r} |')
    md('')
    md('時系列は `e_t90.png`。')
    (OUT / 'judge_e.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
    json.dump(dict(judge=J, j5_count=j5c, predictions=P, j5={f'{a}/{s}': j5_early(a, s) for a in ('SRMS', 'SL') for s in SEEDS}),
              open(OUT / 'judge_e.json', 'w'), ensure_ascii=False, indent=1, default=str)
    # 図: 課題ごとの T90（共有の腕は seed の色、戻す腕は灰色、点線は同じ seed の新品の中央値）
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (n, (sh, re_, fa)) in zip(axes, NORMS.items()):
        for i, s in enumerate(SEEDS):
            a, b, f = runs[(sh, s)], runs[(re_, s)], fresh[(n, s)]
            if b:
                ax.plot(range(1, 101), [b[k] for k in range(1, 101)], color='#b4b2a9', lw=.8, label=f'{re_}（戻す）' if i == 0 else None)
            if a:
                ax.plot(range(1, 101), [a[k] for k in range(1, 101)], color=PALETTE[i], lw=1.0, label=f'{sh} seed {s}')
            if f:
                ax.axhline(St.median(f), color=PALETTE[i], lw=.8, ls=':')
        ax.axvspan(81, 100, color='#f1efe8', zorder=0)
        ax.set_title(f'{NAME[n]}（共有 {sh}・戻す {re_}）', fontsize=10)
        ax.set_xlabel('課題')
        ax.grid(axis='y', color='#eeeeee', lw=.6)
        ax.legend(frameon=False, fontsize=7, loc='upper left')
    axes[0].set_ylabel('T90（暗記率 0.9 までの更新数）')
    fig.text(0.5, 0.01, '点線は同じ seed の新品 4 本（課題 25・50・75・100）の T90 の中央値。網かけは判定の窓（課題 81〜100）。',
             ha='center', fontsize=8, color=INK)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / 'e_t90.png', dpi=130)
    print('\n'.join(L))


if __name__ == '__main__':
    main()
