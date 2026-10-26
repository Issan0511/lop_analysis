"""j5_rlfix.py の結果の表と図（Claude、1026）: 課題末ごとの cos²・沈める側の割合（装填・全体・J5r）・実際の 1 歩。

    python analysis/vit_long_1024/posthoc_2blk/j5_plot.py <JSON の置き場> <図と表の置き場>
"""
import json
import statistics as St
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'font.size': 10, 'axes.edgecolor': '#c9c8c3',
                     'axes.labelcolor': '#52514e', 'xtick.color': '#52514e', 'ytick.color': '#52514e',
                     'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb'})
ARMS = [('SL', 'SL（LN あり・共有）', '#eb6834'), ('R0L', 'R0L（LN あり・戻す、反実仮想）', '#2a78d6'),
        ('S', 'S（LN なし・共有）', '#eda100'), ('R0', 'R0（LN なし・戻す、反実仮想）', '#1baf7a')]


def main(src, out):
    data = {arm: json.loads((src / f'j5_{arm}.json').read_text())['results'] for arm, _, _ in ARMS
            if (src / f'j5_{arm}.json').exists()}
    lines = ['| 腕 | 窓 | cos² の中央値 | s_load（装填を通す押しが沈める側の割合）| s_full（head 全体）| s_r（J5r）| 直通だけ s_load | 分散微分だけ s_load |',
             '|---|---|---|---|---|---|---|---|']
    for arm, name, _ in ARMS:
        if arm not in data:
            continue
        for lo, hi, label in [(1, 9, '早期（課題末 1〜9）'), (10, 60, '中期（10〜60）'), (70, 100, '後期（70〜100）')]:
            rr = [r for r in data[arm] if lo <= r['task'] <= hi]
            if not rr:
                continue
            med = lambda k: St.median(r[k] for r in rr)
            lines.append(f"| {name} | {label} | {med('cos2'):.3f} | {med('s_load'):.3f} | {med('s_full'):.3f} | {med('s_r'):.3f} | {med('s_load_direct'):.3f} | {med('s_load_var'):.3f} |")
    lines += ['', '実際の 1 歩（課題 t+1 の最初の batch、Adam の状態つき）。沈める側 = 勾配の射影 G_j > 0、Adam の 1 歩で w_j·ȳ + b_j が下がった割合。一致 = S_load の符号との一致。', '',
              '| 腕 | 課題末 t | 勾配が沈める側 | Adam が沈める側 | 一致（装填・勾配） | 一致（装填・Adam） | 一致（勾配・Adam） |', '|---|---|---|---|---|---|---|']
    for arm, name, _ in ARMS:
        for r in data.get(arm, []):
            fs = r.get('first_step', {})
            if 'frac_grad_sinks' in fs:
                lines.append(f"| {name} | {r['task']} | {fs['frac_grad_sinks']:.3f} | {fs['frac_adam_sinks']:.3f} | {fs['agree_Sload_grad']:.3f} | {fs['agree_Sload_adam']:.3f} | {fs['agree_grad_adam']:.3f} |")
    text = '\n'.join(lines)
    (out / 'j5_table.md').write_text(text + '\n')
    print(text)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    ax = axes[0]
    for arm, name, color in ARMS:
        if arm in data:
            ax.plot([r['task'] for r in data[arm]], [r['cos2'] for r in data[arm]], color=color, lw=1.8, marker='o', ms=3, mec='#fcfcfb', label=name)
    ax.set_ylim(0, 1.02); ax.set_ylabel('装填 cos²（head の行と F_t の整列）'); ax.set_xlabel('課題末')
    ax = axes[1]
    for arm, name, color in ARMS:
        if arm in data:
            t = [r['task'] for r in data[arm]]
            ax.plot(t, [r['s_load'] for r in data[arm]], color=color, lw=1.8, marker='o', ms=3, mec='#fcfcfb', label=f'{name}：装填を通す')
            ax.plot(t, [r['s_r'] for r in data[arm]], color=color, lw=1.4, ls='--', label=f'{name}：J5r（head の学び直し）')
    ax.axhline(0.75, color='#c9c8c3', lw=0.8); ax.axhline(0.5, color='#c9c8c3', lw=0.8, ls=':')
    ax.set_ylim(0, 1.02); ax.set_ylabel('最終 FFN の生きた unit のうち\n押しが沈める側の割合'); ax.set_xlabel('課題末')
    for a in axes:
        a.grid(True, color='#ebeae6', lw=0.6); a.set_axisbelow(True)
        for s in ('top', 'right'):
            a.spines[s].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8, loc='lower left'); axes[1].legend(frameon=False, fontsize=7, loc='lower left')
    fig.suptitle('J5：装填と、最終ブロックの FFN の自分の重みへの切替の押しの向き（CLS 位置、保存状態）', fontsize=11)
    fig.tight_layout()
    fig.savefig(out / 'j5_loading.png', dpi=140)
    print(out / 'j5_loading.png')


if __name__ == '__main__':
    main(Path(sys.argv[1]), Path(sys.argv[2]))
