"""tangent_rank.py の結果と体の速さ（head を 0 にした回し直しの T90）を 1 つの表にする（Claude、1027）。

体の速さ: SL は scale_counterfactual.py の head0（Adam の保存が無い課題末は head0+freshadam）、R0L・R0 は本走の T90（head を 0 に
戻す手順そのもの）、S は本走の T90（装填込み、参考）。接核の量は方向 4 本の平均。

    python analysis/vit_long_1024/posthoc_2blk/tangent_rank_table.py <posthoc_tangent> <posthoc_scale> <runs> <表.md>
"""
import csv
import json
import sys
from pathlib import Path


def body_speed(arm, t, scale, runs):
    if arm == 'SL':
        for name in (f'scale_SL_T{t}_head0.json', f'scale_SL_T{t}_head0+freshadam.json'):
            f = scale / name
            if f.exists():
                return str(json.loads(f.read_text())['T90']) + ('' if name.endswith('head0.json') else '†')
        return '—'
    if t == 0:
        return '—'
    rows = {int(r['task']): r for r in csv.DictReader(open(runs / arm / 'seed10252010/per_task.csv'))}
    r = rows.get(t + 1)
    if r is None:                                     # 課題 100 の末: 本走に課題 101 は無いので、そのままの回し直し（base）を使う
        f = scale / f'scale_{arm}_T{t}_base.json'
        return str(json.loads(f.read_text())['T90']) + '§' if f.exists() else '—'
    return r['T90'] + ('‡' if arm == 'S' else '')


def main(src, scale, runs, out):
    avg = lambda ds, k: sum(d[k] for d in ds) / len(ds)
    lines = ['# 体の接核の有効階数と体の速さ（tangent_rank.py、256 枚、方向 4 本の平均）', '',
             '体の速さは head を 0 にして次の課題を回したときの T90（SL は回し直し、† は Adam の記憶を空にして回した、'
             'R0L・R0 は本走そのもの、§ は課題 100 末からのそのままの回し直し、‡ の S は本走の装填込みの値）。「新品」は課題 0 の初期値（本走の課題 1 の T90 は 525）。'
             '参加比 = 固有値の参加比（有効階数）、向きだけ = 画像ごとの勾配を単位ベクトルにした Gram の参加比、'
             '上位 1 本 = 最大固有値の割合、‖g‖ = 画像ごとの勾配ノルムの中央値、旧クラス同/異 = 前の課題で同じクラスだった対と違う対の cos の平均。', '',
             '| 腕 | 課題末 | 体の速さ T90 | 特徴の参加比（2,500 枚） | 乱数の向き: 参加比 | 向きだけ | 上位 1 本 | ‖g‖ | 旧クラス 同/異 '
             '| head が読む向き: 参加比 | 向きだけ | 上位 1 本 | ‖g‖ | 旧クラス 同/異 |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    order = {'SL': 0, 'R0L': 1, 'S': 2, 'R0': 3}
    files = sorted(src.glob('*_t*.json'), key=lambda f: (order.get(f.stem.split('_t')[0], 9), int(f.stem.split('_t')[1])))
    for f in files:
        j = json.loads(f.read_text())
        arm, t = j['arm'], j['task']
        cells = [arm if t > 0 else '新品', str(t), body_speed(arm, t, scale, runs), f"{j['feature_pr_2500']:.1f}"]
        for key in ('rand', 'pc'):
            ds = j[key]
            same = f"{avg(ds, 'cos_same_old'):+.3f} / {avg(ds, 'cos_diff_old'):+.3f}" if 'cos_same_old' in ds[0] else '—'
            cells += [f"{avg(ds, 'pr'):.0f}", f"{avg(ds, 'pr_cos'):.0f}", f"{avg(ds, 'top1_share'):.2f}", f"{avg(ds, 'norm_med'):.0f}", same]
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines.append('')
    if files and 'norm_quantiles' in json.loads(files[0].read_text())['rand'][0]:
        lines += ['## 勾配ノルムの分布と、大きい少数を除いた参加比（乱数の向き）', '',
                  '| 腕 | 課題末 | ノルム 10% / 50% / 90% / 99% | 上位 5% の画像が持つトレースの割合 | 下位 90% だけの参加比 |', '|---|---|---|---|---|']
        for f in files:
            j = json.loads(f.read_text())
            ds = j['rand']
            q = [sum(d['norm_quantiles'][i] for d in ds) / len(ds) for i in (0, 2, 4, 5)]
            lines.append(f"| {j['arm'] if j['task'] > 0 else '新品'} | {j['task']} | {q[0]:.0f} / {q[1]:.0f} / {q[2]:.0f} / {q[3]:.0f}"
                         f" | {avg(ds, 'trace_share_top5pct'):.2f} | {avg(ds, 'pr_bulk90'):.0f} |")
        lines.append('')
    Path(out).write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
