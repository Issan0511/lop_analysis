"""width_cap_replay.py の結果を表にする（Claude、1027）。

幅を抑えた走（cap）、同じ機械の対照（none）、元の走（per_task.csv）を、課題 31〜40・41〜50・51〜60 の平均で並べ、
続けて課題ごとの表を書く。元の走の層の量は layer_stats.csv の課題末（state = end）。

    python analysis/vit_long_1024/posthoc_2blk/width_cap_table.py <width_cap の置き場> <SL の走の置き場> <表.md>
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

WINDOWS = ((31, 40), (41, 50), (51, 60))


def load(src, run):
    cap = json.loads((src / 'SL_T30_60_cap.json').read_text())['rows']
    none = json.loads((src / 'SL_T30_60_none.json').read_text())['rows']
    pt = {int(r['task']): r for r in csv.DictReader(open(run / 'per_task.csv'))}
    ls = {(int(r['task']), int(r['layer'])): r for r in csv.DictReader(open(run / 'layer_stats.csv')) if r['state'] == 'end'}
    orig = [dict(task=t, T90=float(pt[t]['T90']), mem_loss_start=float(pt[t]['mem_loss_start']),
                 layers=[{k: float(ls[(t, i)][k]) for k in ('mu_med', 'sigma_med', 'r_med', 'dead005')} for i in (1, 2)])
            for t in range(31, 61)]
    return dict(cap=cap, none=none, orig=orig)


def win(rows, f):
    return [np.mean([f(r) for r in rows if lo <= r['task'] <= hi]) for lo, hi in WINDOWS]


def main(src, run, out):
    runs = load(src, run)
    lines = ['# 幅を抑えた SL の回し直し（width_cap_replay.py、課題 30 末から 31〜60）', '',
             'cap は毎更新の後、2 つの FFN の fc1 の行ノルムを課題 30 末の値で上から抑える。none は同じ機械で抑えない対照、'
             '元の走は本走の per_task.csv と layer_stats.csv。前活性の量は固定 probe の全トークンの unit ごとの値の中央値、'
             '死は開いている割合 < 0.5% の unit の割合。', '',
             '## 10 課題ごとの平均', '',
             '| 量 | 走 | 課題 31〜40 | 41〜50 | 51〜60 |', '|---|---|---|---|---|']
    items = [('T90', lambda r: r['T90'], '{:.0f}'), ('始めの暗記損失', lambda r: r['mem_loss_start'], '{:.2f}')]
    for i in (0, 1):
        items += [(f'{i + 1} 層目の mean ‖w‖²', lambda r, i=i: r['w2'][i], '{:.2f}'),
                  (f'{i + 1} 層目の μ', lambda r, i=i: r['layers'][i]['mu_med'], '{:+.2f}'),
                  (f'{i + 1} 層目の σ', lambda r, i=i: r['layers'][i]['sigma_med'], '{:.2f}'),
                  (f'{i + 1} 層目の r', lambda r, i=i: r['layers'][i]['r_med'], '{:+.2f}'),
                  (f'{i + 1} 層目の死', lambda r, i=i: r['layers'][i]['dead005'], '{:.3f}')]
    for name, f, fmt in items:
        for arm in ('cap', 'none', 'orig'):
            if arm == 'orig' and '‖w‖²' in name:
                continue
            lines.append(f"| {name} | {dict(cap='幅を抑える', none='対照', orig='元の走')[arm]} | "
                         + ' | '.join(fmt.format(v) for v in win(runs[arm], f)) + ' |')
    lines += ['', '## T90 の差（10 課題の平均 ± 標準誤差）', '',
              '| 差 | 課題 31〜40 | 41〜50 | 51〜60 |', '|---|---|---|---|']
    for a, b, name in (('cap', 'none', '幅を抑える − 対照'), ('orig', 'none', '元の走 − 対照（抑えない走どうし）')):
        d = np.array([x['T90'] - y['T90'] for x, y in zip(runs[a], runs[b])])
        lines.append(f'| {name} | ' + ' | '.join(f'{d[k:k + 10].mean():+.0f} ± {d[k:k + 10].std(ddof=1) / np.sqrt(10):.0f}'
                                                  for k in (0, 10, 20)) + ' |')
    for arm in ('cap', 'none'):
        T = np.array([r['T90'] for r in runs[arm]])
        l0 = np.array([r['mem_loss_start'] for r in runs[arm]])
        lines.append(f"\n{dict(cap='幅を抑える', none='対照')[arm]}: T90 と始めの暗記損失の相関 {np.corrcoef(T, l0)[0, 1]:.2f}。")
    lines += ['', '## 課題ごと', '',
              '| 課題 | T90（抑える / 対照 / 元） | 始めの損失（抑える / 対照 / 元） | 2 層目の mean ‖w‖²（抑える / 対照） | 2 層目の r（抑える / 対照 / 元） | 2 層目の死（抑える / 対照 / 元） |',
              '|---|---|---|---|---|---|']
    for c, n, o in zip(runs['cap'], runs['none'], runs['orig']):
        lines.append(f"| {c['task']} | {c['T90']} / {n['T90']} / {o['T90']:.0f}"
                     f" | {c['mem_loss_start']:.2f} / {n['mem_loss_start']:.2f} / {o['mem_loss_start']:.2f}"
                     f" | {c['w2'][1]:.3f} / {n['w2'][1]:.3f}"
                     f" | {c['layers'][1]['r_med']:+.2f} / {n['layers'][1]['r_med']:+.2f} / {o['layers'][1]['r_med']:+.2f}"
                     f" | {c['layers'][1]['dead005']:.3f} / {n['layers'][1]['dead005']:.3f} / {o['layers'][1]['dead005']:.3f} |")
    Path(out).write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
