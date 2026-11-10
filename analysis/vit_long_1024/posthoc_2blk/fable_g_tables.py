"""追補 G の判定（spec G.4）と予測の採点（G.5）の表を、本走の csv から作る（Fable、1108）。

読むもの: runs/<SLP|SLW|SL>/seed*/per_task.csv・layer_stats.csv、runs/freshL/seed*/task{25,50,75,100}/per_task.csv。
書くもの: results/vit_long_1024_2blk_rlfix/addendum_g/judge.md・judge.json・g_timeseries.png。記録として最終 LN の利得 γ（ckpt/model_t01.pt と model_t100.pt）も読む。
規則は spec G.4 を走る前に固定したまま。T90 の未到達は 5,025。窓は早期 2〜11・後期 81〜100。‖W‖ は head の損失の 10 行のノルム
（per_task.csv の head_norm_end）で、比べるのはその腕自身の課題 1 末。Q = 後期の T90 の中央値 ÷ 同 seed の freshL 4 本（課題 25・50・75・100）の中央値。
対照（SL・freshL）が届いていない seed は — で埋め、ある分だけで出す。判定の表示は、成立の数が規則の数に届けば「成立」、欠けた seed を全部
成立と数えても届かなければ「不成立」、どちらでもなければ「保留」。

    PYTHONPATH=$PWD python -m analysis.vit_long_1024.posthoc_2blk.fable_g_tables
"""
import csv
import json
import statistics as St
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

RAW = Path.home() / 'Projects/obsidian-research-data/vit_long_1024_2blk_rlfix'
RUNS = RAW / 'runs'
REPO = Path(__file__).resolve().parents[3]
OUT = REPO / 'results/vit_long_1024_2blk_rlfix/addendum_g'
SEEDS = [10252010 + i for i in range(5)]
ARMS = ['SLP', 'SLW']
UNR = 5025
EARLY, LATE = range(2, 12), range(81, 101)
FRESH_TASKS = [25, 50, 75, 100]
TH = dict(norm=1.3, conf=1.3, q=1.2, dead=.05, dr=.3)
NEED = dict(G1=5, G2=4, G3=4, G4=4)
LABEL = dict(G1=('NORM_HELD', 'NORM_GROWS'), G2=('CONFIDENCE_HELD', 'CONFIDENCE_GROWS'),
             G3=('COST_STOPS', 'COST_REMAINS'), G4=('BODY_UNCHANGED', 'BODY_CHANGED'))
INK = '#333333'
PALETTE = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#D55E00']   # seed の順に固定
plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'font.size': 9, 'axes.edgecolor': '#c9c8c3',
                     'axes.spines.top': False, 'axes.spines.right': False})


def fmt(v, nd=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '—'
    if isinstance(v, bool):
        return '○' if v else '×'
    if isinstance(v, int):
        return f'{v:,}'
    return f'{v:,.{nd}f}'


def reached(r):
    return str(r.get('reached90', '')).strip().lower() in ('1', 'true', '1.0')


def read_run(arm, seed):
    d = RUNS / arm / f'seed{seed}'
    p = d / 'per_task.csv'
    if not p.exists():
        return None
    rows = list(csv.DictReader(open(p)))
    if len(rows) < 100:
        return None
    pt = {int(r['task']): r for r in rows}
    l2 = {}
    for r in csv.DictReader(open(d / 'layer_stats.csv')):
        if r['state'] == 'end' and r['layer'] == '2':
            l2[int(r['task'])] = dict(dead005=float(r['dead005']), r=float(r['r_med']), sigma=float(r['sigma_med']))
    return dict(t90={t: (float(r['T90']) if reached(r) else UNR) for t, r in pt.items()},
                norm={t: float(r['head_norm_end']) for t, r in pt.items()},
                conf={t: float(r['mem_loss_start']) for t, r in pt.items()},
                dead={t: v['dead005'] for t, v in l2.items()}, r={t: v['r'] for t, v in l2.items()},
                sigma={t: v['sigma'] for t, v in l2.items()})


def fresh_t90(seed):
    out = {}
    for t in FRESH_TASKS:
        p = RUNS / 'freshL' / f'seed{seed}' / f'task{t}' / 'per_task.csv'
        if p.exists():
            rows = list(csv.DictReader(open(p)))
            if rows:
                out[t] = float(rows[-1]['T90']) if reached(rows[-1]) else UNR
    return out if len(out) == 4 else None


def final_ln_gain(arm, seed):
    """最終 LN（layers.norm）の利得 γ の RMS と β の RMS を、課題 1 末と課題 100 末の保存状態から（事後の記録。判定に入れない）。"""
    d = RUNS / arm / f'seed{seed}' / 'ckpt'
    out = {}
    for t in ('01', '100'):
        p = d / f'model_t{t}.pt'
        if not p.exists():
            return None
        sd = torch.load(p, map_location='cpu', weights_only=False)
        out[t] = dict(gamma=float(sd['layers.norm.weight'].float().pow(2).mean().sqrt()), beta=float(sd['layers.norm.bias'].float().pow(2).mean().sqrt()))
    out['ratio'] = out['100']['gamma'] / out['01']['gamma']
    return out


def med(d, win):
    v = [d[t] for t in win if t in d]
    return St.median(v) if v else None


def unreached(d, win):
    return sum(1 for t in win if d.get(t) == UNR)


def stats(run, fresh):
    if run is None:
        return None
    n, c, t = run['norm'], run['conf'], run['t90']
    ts = np.arange(10, 101)
    slope = float(np.polyfit(ts, [n[k] ** 2 for k in ts], 1)[0])
    s = dict(norm1=n[1], norm10=n[10], norm100=n[100], norm_ratio=n[100] / n[1], norm2_late=St.median([n[k] ** 2 for k in LATE]),
             slope=slope, conf_early=med(c, EARLY), conf_late=med(c, LATE),
             t90_early=med(t, EARLY), t90_late=med(t, LATE), unr_early=unreached(t, EARLY), unr_late=unreached(t, LATE),
             dead_late=med(run['dead'], LATE), r_late=med(run['r'], LATE), sigma_late=med(run['sigma'], LATE),
             fresh=fresh, fresh_med=(St.median(fresh.values()) if fresh else None))
    s['conf_ratio'] = s['conf_late'] / s['conf_early']
    s['q'] = (s['t90_late'] / s['fresh_med']) if fresh else None
    return s


def decide(flags, need):
    n_true = sum(f is True for f in flags)
    n_miss = sum(f is None for f in flags)
    if n_true >= need:
        return 'pos', f'{n_true}/5（成立）'
    if n_true + n_miss < need:
        return 'neg', f'{n_true}/5（不成立' + (f'、欠け {n_miss}）' if n_miss else '）')
    return 'hold', f'{n_true}/{5 - n_miss}（保留、欠け {n_miss}）'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    runs = {(a, s): read_run(a, s) for a in ARMS + ['SL'] for s in SEEDS}
    fresh = {s: fresh_t90(s) for s in SEEDS}
    S = {(a, s): stats(runs[(a, s)], fresh[s]) for a in ARMS + ['SL'] for s in SEEDS}
    J, L = {}, []
    md = L.append

    md('# 追補 G の判定（spec G.4）と予測の採点（G.5）')
    md('')
    md('規則は spec G.4 を走る前に固定したまま。T90 の未到達は 5,025。窓は早期 2〜11・後期 81〜100。‖W‖ は head の損失の 10 行のノルム（課題末）で、比べるのはその腕自身の課題 1 末。'
       'Q = 後期の T90 の中央値 ÷ 同 seed の新品 freshL 4 本（課題 25・50・75・100）の中央値。対照（SL・freshL）が無い seed は — で、判定はある分だけで出す。')
    have_sl = [s for s in SEEDS if runs[('SL', s)]]
    have_fr = [s for s in SEEDS if fresh[s]]
    md(f'読んだ走: SLP・SLW は {sum(1 for a in ARMS for s in SEEDS if runs[(a, s)])}/10。対照の SL は seed {", ".join(str(s)[-2:] for s in have_sl)}、freshL は seed {", ".join(str(s)[-2:] for s in have_fr)}（下 2 桁）。')
    md('')

    # ---------- J G1 ----------
    md('## J G1 `NORM_HELD`（‖W‖ の課題 100 末 ÷ 課題 1 末 ≤ 1.3。5/5。1/1.3 = 0.77 を下回れば「縮んだ」）')
    md('')
    md('| 腕 | seed | ‖W‖ 課題 1 末 / 10 末 / 100 末 | 比 100/1 | ‖W‖² の傾き（課題 10〜100、/課題） | 判定 |')
    md('|---|---|---|---|---|---|')
    for a in ARMS + ['SL']:
        flags = []
        for s in SEEDS:
            st = S[(a, s)]
            if st is None:
                md(f'| {a} | {s} | — | — | — | — |'); flags.append(None); continue
            ok = st['norm_ratio'] <= TH['norm']
            tag = '○' if ok else '×'
            if st['norm_ratio'] < 1 / TH['norm']:
                tag += '（縮んだ）'
            flags.append(ok)
            md(f"| {a} | {s} | {fmt(st['norm1'])} / {fmt(st['norm10'])} / {fmt(st['norm100'])} | {fmt(st['norm_ratio'])} | {fmt(st['slope'], 3)} | {tag} |")
        if a in ARMS:
            J.setdefault(a, {})['G1'] = dict(flags=flags, decision=decide(flags, NEED['G1']))
    md('')
    for a in ARMS:
        kind, txt = J[a]['G1']['decision']
        md(f"**J G1 {a}: {txt} → {LABEL['G1'][0] if kind == 'pos' else LABEL['G1'][1] if kind == 'neg' else '保留'}**  ")
    md('')

    # ---------- J G2 ----------
    md('## J G2 `CONFIDENCE_HELD`（課題の始めの暗記損失の中央値、後期 ÷ 早期 ≤ 1.3。4/5）')
    md('')
    md('| 腕 | seed | 早期 2〜11 | 後期 81〜100 | 比 | 判定 |')
    md('|---|---|---|---|---|---|')
    for a in ARMS + ['SL']:
        flags = []
        for s in SEEDS:
            st = S[(a, s)]
            if st is None:
                md(f'| {a} | {s} | — | — | — | — |'); flags.append(None); continue
            ok = st['conf_ratio'] <= TH['conf']
            flags.append(ok)
            md(f"| {a} | {s} | {fmt(st['conf_early'], 1)} | {fmt(st['conf_late'], 1)} | {fmt(st['conf_ratio'])} | {fmt(ok)} |")
        if a in ARMS:
            J[a]['G2'] = dict(flags=flags, decision=decide(flags, NEED['G2']))
    md('')
    for a in ARMS:
        kind, txt = J[a]['G2']['decision']
        md(f"**J G2 {a}: {txt} → {LABEL['G2'][0] if kind == 'pos' else LABEL['G2'][1] if kind == 'neg' else '保留'}**  ")
    md('')

    # ---------- J G3 ----------
    md('## J G3 `COST_STOPS`（主。Q = 後期の T90 の中央値 ÷ 新品 freshL の中央値 ≤ 1.2。4/5）')
    md('')
    md('| 腕 | seed | T90 早期 2〜11（未到達） | T90 後期 81〜100（未到達） | freshL T90 課題 25 / 50 / 75 / 100 | 新品の中央値 | Q | 判定 |')
    md('|---|---|---|---|---|---|---|---|')
    for a in ARMS + ['SL']:
        flags = []
        for s in SEEDS:
            st = S[(a, s)]
            if st is None:
                md(f'| {a} | {s} | — | — | — | — | — | — |'); flags.append(None); continue
            fr = st['fresh']
            frs = ' / '.join(fmt(int(fr[t])) for t in FRESH_TASKS) if fr else '—'
            ok = (st['q'] <= TH['q']) if st['q'] is not None else None
            flags.append(ok)
            md(f"| {a} | {s} | {fmt(int(st['t90_early']))} ({st['unr_early']}) | {fmt(int(st['t90_late']))} ({st['unr_late']}) | {frs} | {fmt(int(st['fresh_med'])) if fr else '—'} | {fmt(st['q'])} | {fmt(ok)} |")
        if a in ARMS:
            J[a]['G3'] = dict(flags=flags, decision=decide(flags, NEED['G3']))
    md('')
    for a in ARMS:
        kind, txt = J[a]['G3']['decision']
        md(f"**J G3 {a}: {txt} → {LABEL['G3'][0] if kind == 'pos' else LABEL['G3'][1] if kind == 'neg' else '保留'}**  ")
    md('')

    # ---------- J G4 ----------
    md('## J G4 `BODY_UNCHANGED`（後期の課題末の固定 probe: 2 層目の死 dead005 の中央値 ≤ 0.05、かつ 2 層目の r の中央値が同 seed の SL の ±0.3。4/5）')
    md('')
    md('| 腕 | seed | 死 dead005（後期の中央値） | r（後期の中央値） | SL の r | Δr | 幅 σ（後期の中央値） | 判定 |')
    md('|---|---|---|---|---|---|---|---|')
    for a in ARMS + ['SL']:
        flags = []
        for s in SEEDS:
            st, sl = S[(a, s)], S[('SL', s)]
            if st is None:
                md(f'| {a} | {s} | — | — | — | — | — | — |'); flags.append(None); continue
            dr = (st['r_late'] - sl['r_late']) if sl else None
            ok = None if dr is None else (st['dead_late'] <= TH['dead'] and abs(dr) <= TH['dr'])
            flags.append(ok)
            md(f"| {a} | {s} | {fmt(st['dead_late'], 3)} | {fmt(st['r_late'])} | {fmt(sl['r_late']) if sl else '—'} | {fmt(dr)} | {fmt(st['sigma_late'])} | {fmt(ok) if a in ARMS else '（基準）'} |")
        if a in ARMS:
            J[a]['G4'] = dict(flags=flags, decision=decide(flags, NEED['G4']))
    md('')
    for a in ARMS:
        kind, txt = J[a]['G4']['decision']
        md(f"**J G4 {a}: {txt} → {LABEL['G4'][0] if kind == 'pos' else LABEL['G4'][1] if kind == 'neg' else '保留'}**  ")
    md('')

    # ---------- まとめ ----------
    md('## 判定のまとめ')
    md('')
    md('| 腕 | J G1 ノルム | J G2 確信 | J G3 費用（主） | J G4 体 |')
    md('|---|---|---|---|---|')
    for a in ARMS:
        cells = []
        for k in ('G1', 'G2', 'G3', 'G4'):
            kind, txt = J[a][k]['decision']
            name = LABEL[k][0] if kind == 'pos' else LABEL[k][1] if kind == 'neg' else '保留'
            cells.append(f'{name} {txt}')
        md(f'| {a} | ' + ' | '.join(cells) + ' |')
    md('')

    # ---------- 予測 ----------
    def verdict(a, k):
        kind, _ = J[a][k]['decision']
        return '当たり' if kind == 'pos' else '外れ' if kind == 'neg' else '保留'
    md('## 予測の採点（G.5、Fable）')
    md('')
    md('| 記号 | 予測 | 確率 | 採点の規則 | 結果 |')
    md('|---|---|---|---|---|')
    md(f"| PG1 | WD で head のノルムが止まる | 0.9 | SLW の J G1 | {verdict('SLW', 'G1')} |")
    md(f"| PG2 | ノルムを止めると確信も止まる | 0.8 | J G2（SLP / SLW） | {verdict('SLP', 'G2')} / {verdict('SLW', 'G2')} |")
    md(f"| PG3P | 行ノルムの射影で費用が止まる | 0.7 | SLP の J G3 | {verdict('SLP', 'G3')} |")
    md(f"| PG3W | WD で費用が止まる | 0.6 | SLW の J G3 | {verdict('SLW', 'G3')} |")
    md(f"| PG4 | 体は変わらない | 0.8 | J G4（SLP / SLW） | {verdict('SLP', 'G4')} / {verdict('SLW', 'G4')} |")
    md('')

    # ---------- 記録 ----------
    md('## 記録（判定に入れない）')
    md('')
    md('- SLW の釣り合い: λ = 0.077 は、伸び 0.17/課題と課題 1 末の ‖W‖² 2.2 から置いた。後期の ‖W‖² の中央値は次のとおり（SL は比較）。')
    md('')
    md('| seed | SLW ‖W‖² 後期 | SLP ‖W‖² 後期 | SL ‖W‖² 後期 | SLW の ‖W‖²（課題 1 末） |')
    md('|---|---|---|---|---|')
    for s in SEEDS:
        a, b, c = S[('SLW', s)], S[('SLP', s)], S[('SL', s)]
        md(f"| {s} | {fmt(a['norm2_late']) if a else '—'} | {fmt(b['norm2_late']) if b else '—'} | {fmt(c['norm2_late']) if c else '—'} | {fmt(a['norm1'] ** 2) if a else '—'} |")
    md('')
    md('- ‖W‖² の傾き（/課題、窓ごとの最小二乗。事後の記録、1109）。')
    md('')
    md('| 腕 | seed | 課題 10〜40 | 課題 41〜70 | 課題 71〜100 |')
    md('|---|---|---|---|---|')
    for arm_ in ('SLW', 'SL'):
        for seed_ in SEEDS:
            run_ = runs[(arm_, seed_)]
            if not run_:
                md(f'| {arm_} | {seed_} | — | — | — |')
                continue
            cells = []
            for lo, hi in ((10, 40), (41, 70), (71, 100)):
                tt = np.arange(lo, hi + 1)
                cells.append(f"{np.polyfit(tt, [run_['norm'][int(i)] ** 2 for i in tt], 1)[0]:+.3f}")
            md(f"| {arm_} | {seed_} | {' | '.join(cells)} |")
    md('')
    md('- 最終 LN の利得 γ（head の入力に掛かる尺度）。head のノルムを止めた腕では γ が伸びる（事後の記録、1108。走る前の鎖は γ を一定と置いていた）。')
    md('')
    md('| 腕 | seed | γ の RMS 課題 1 末 → 100 末 | 比 | ‖W‖ の比（J G1） | ‖W‖ × γ の比 | 確信の比（J G2） |')
    md('|---|---|---|---|---|---|---|')
    gains = {}
    for a in ARMS + ['SL']:
        for s in SEEDS:
            g, st = final_ln_gain(a, s), S[(a, s)]
            gains[f'{a}/{s}'] = g
            if g is None or st is None:
                md(f'| {a} | {s} | — | — | — | — | — |'); continue
            md(f"| {a} | {s} | {fmt(g['01']['gamma'], 3)} → {fmt(g['100']['gamma'], 3)} | {fmt(g['ratio'])} | {fmt(st['norm_ratio'])} | {fmt(g['ratio'] * st['norm_ratio'])} | {fmt(st['conf_ratio'])} |")
    md('')
    md('- 課題ごとの ‖W‖・始めの暗記損失・T90 の時系列は `g_timeseries.png`。')
    # 記録: 体の速さ（G.4、1111）
    md('- 体の速さ（G.4 の記録、1111）: 保存した課題 100 末から head を 0 に戻し、保存した Adam のまま課題 101 を回した T90'
       '（`scale_counterfactual.py` の head0、eager、≤ 5,000 更新。腕の WD・射影は掛けない。道具 `run_g_records.sh`、出力 `<raw>/posthoc_g/`）。'
       '15 本とも 1111 に white-san の同じ起動の中で回した。比べるのは同じ表の SL とだけ（G.4）。')
    md('')
    md('| seed | SL | SLW | SLP | SLW − SL | SLP − SL | 新品の中央値（本走、参考） |')
    md('|---|---|---|---|---|---|---|')
    bs = {}
    for a in ['SL'] + ARMS:
        for s in SEEDS:
            p_ = RAW / 'posthoc_g' / f'body_speed_{a}_seed{s}.json'
            bs[(a, s)] = json.load(open(p_)) if p_.exists() else None
    t90b = lambda a, s: (float(bs[(a, s)]['T90']) if bs[(a, s)] and bs[(a, s)]['status'] == 'reached90' else
                         (float(UNR) if bs[(a, s)] else None))
    for s in SEEDS:
        a_, w_, p__ = t90b('SL', s), t90b('SLW', s), t90b('SLP', s)
        fr = fresh[s]
        md(f"| {s} | {fmt(a_, 0)} | {fmt(w_, 0)} | {fmt(p__, 0)} | {fmt(w_ - a_, 0) if None not in (w_, a_) else '—'} | "
           f"{fmt(p__ - a_, 0) if None not in (p__, a_) else '—'} | {fmt(St.median(fr.values()), 0) if fr else '—'} |")
    md('')
    md('  同じ保存状態・同じコードでも、11/05 の再起動の前（10/27・11/02）は SL の seed 10252010 が 575 で、再起動の後（1111 に 4 回）は 500 だった。'
       '設定・更新数の上限・保存状態・パッケージは同じで、違うのは再起動で入れ替わった kernel（7.0.0-38 → 42）。原因は特定していない。'
       '起動をまたいだ回し直しの T90 は比べない。')
    md('')
    # 記録: 行と特徴の cos²（G.4、1111）
    md('- 保存した課題末の行と特徴の cos²（G.4 の記録、1111）: 追補 D の J5 の量（`j5_rlfix.py`）。cos² = ⟨W^c, F_t⟩² / (‖W^c‖² ‖F_t‖²)、'
       'W^c は損失の 10 行を行平均で中心化した head、F_t = Σ_m (e_{y_m} − 1/K) ⊗ h_m（課題 t の 2,500 枚・そのラベル、CLS 位置の最終 LN の出力）。'
       '道具 `fable_eg_j5.py`、出力 `<raw>/posthoc_j5/j5_<腕>_seed<seed>.json`。5 seed の中央値（最小〜最大）。')
    md('')
    CT = (1, 10, 30, 60, 100)
    md('| 腕 | ' + ' | '.join(f'課題 {t} 末' for t in CT) + ' | κ > 0 の状態 |')
    md('|---|' + '---|' * (len(CT) + 1))
    cos_rec = {}
    for a in ['SL'] + ARMS:
        per = {t: [] for t in CT}
        kpos = [0, 0]
        for s in SEEDS:
            p_ = RAW / 'posthoc_j5' / f'j5_{a}_seed{s}.json'
            if not p_.exists():
                continue
            for r in json.load(open(p_))['results']:
                kpos[0] += r['kappa'] > 0; kpos[1] += 1
                if r['task'] in per:
                    per[r['task']].append(r['cos2'])
        cells = [f"{St.median(v):.2f}（{min(v):.2f}〜{max(v):.2f}）" if v else '—' for v in per.values()]
        md(f"| {a} | {' | '.join(cells)} | {kpos[0]}/{kpos[1]} |")
        cos_rec[a] = {t: v for t, v in per.items()}
    md('')
    md('')

    # ---------- figure ----------
    fig, axes = plt.subplots(3, 3, figsize=(11, 8.2), sharex=True, sharey='row')
    cols = ['SLP', 'SLW', 'SL']
    titles = dict(SLP='SLP（head の行ノルムを射影）', SLW='SLW（head の重みに WD、λ 0.077）', SL='SL（そのまま、対照）')
    for j, a in enumerate(cols):
        for i_s, s in enumerate(SEEDS):
            run = runs[(a, s)]
            if run is None:
                continue
            t = sorted(run['norm'])
            kw = dict(color=PALETTE[i_s], lw=1.1, alpha=.9, label=f'seed {s}' if j == 0 else None)
            axes[0, j].plot(t, [run['norm'][k] for k in t], **kw)
            axes[1, j].plot(t, [run['conf'][k] for k in t], **kw)
            axes[2, j].plot(t, [run['t90'][k] for k in t], **kw)
            if fresh[s]:
                axes[2, j].axhline(St.median(fresh[s].values()), color=PALETTE[i_s], lw=.8, ls=':', alpha=.8)
        axes[0, j].set_title(titles[a], fontsize=10)
        axes[2, j].set_xlabel('課題')
    axes[0, 0].set_ylabel('head の ‖W‖（課題末）')
    axes[1, 0].set_ylabel('課題の始めの暗記損失')
    axes[2, 0].set_ylabel('T90（未到達は 5,025）')
    for ax in axes[0]:
        ax.axhline(1.0, color='#c9c8c3', lw=.6)
    for ax in axes.ravel():
        ax.grid(axis='y', color='#eeeeee', lw=.6)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=5, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.0))
    fig.text(0.5, 0.005, '点線は同 seed の新品 freshL 4 本（課題 25・50・75・100）の T90 の中央値。',
             ha='center', fontsize=8, color=INK)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    fig.savefig(OUT / 'g_timeseries.png', dpi=130)

    (OUT / 'judge.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
    json.dump(dict(stats={f'{a}/{s}': S[(a, s)] for a in ARMS + ['SL'] for s in SEEDS}, judge=J, thresholds=TH, need=NEED, final_ln_gain=gains,
                   body_speed={f'{a}/{s}': t90b(a, s) for a in ['SL'] + ARMS for s in SEEDS}, cos2=cos_rec),
              open(OUT / 'judge.json', 'w'), ensure_ascii=False, indent=1, default=str)
    print('\n'.join(L))


if __name__ == '__main__':
    main()
