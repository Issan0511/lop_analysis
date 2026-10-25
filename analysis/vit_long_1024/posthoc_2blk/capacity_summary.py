"""追補 C の要約: 状態 × N の T50・T75・T90 の表と、暗記率の曲線の図（Claude、1025）。

    python analysis/vit_long_1024/posthoc_2blk/capacity_summary.py <図と表の置き場>
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'font.size': 10, 'axes.edgecolor': '#c9c8c3',
                     'axes.labelcolor': '#52514e', 'xtick.color': '#52514e', 'ytick.color': '#52514e',
                     'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb'})
RAW = Path.home() / 'Projects/obsidian-research-data/vit_long_1024_2blk_t100/capacity'
NS = [250, 1000, 2500]
U_MAX = {250: 2000, 1000: 4000, 2500: 8000}
STATES = [('R0L', 'R0L（LN あり・戻す）', '#2a78d6', '-'), ('SL', 'SL（LN あり・共有）', '#eb6834', '-'),
          ('R0', 'R0（LN なし・戻す）', '#1baf7a', '-'), ('S', 'S（LN なし・共有）', '#eda100', '-'),
          ('freshL', '新品（LN あり）', '#52514e', ':'), ('fresh', '新品（LN なし）', '#52514e', '--')]


def load(state, n):
    d = RAW / f'{state}_N{n}'
    if not (d / 'result.json').exists():
        return None, None
    return json.loads((d / 'result.json').read_text()), np.load(d / 'intask.npz')


def table():
    lines = ['| 状態 | ' + ' | '.join(f'N = {n:,}' for n in NS) + ' |', '|---|' + '---|' * len(NS)]
    for state, name, _, _ in STATES:
        cells = []
        for n in NS:
            r, _ = load(state, n)
            if r is None:
                cells.append('—')
                continue
            t90 = f"{r['T90']:,}" if r['status'] == 'reached90' else f"未到達（{r['mem_acc_last']:.2f}）"
            cells.append(f"{r['T50']:,}・{r['T75']:,}・{t90}")
        lines.append(f'| {name} | ' + ' | '.join(cells) + ' |')
    lines.append('')
    lines.append('課題の始めの暗記損失（更新前、N 枚の平均。一様なら 2.30）')
    lines.append('')
    lines.append('| 状態 | ' + ' | '.join(f'N = {n:,}' for n in NS) + ' |')
    lines.append('|---|' + '---|' * len(NS))
    for state, name, _, _ in STATES:
        cells = []
        for n in NS:
            r, _ = load(state, n)
            cells.append('—' if r is None else f"{r['mem_loss_start']:.2f}")
        lines.append(f'| {name} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def figure(out):
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.3), sharey=True)
    for ax, n in zip(axes, NS):
        for state, name, color, ls in STATES:
            r, d = load(state, n)
            if r is None:
                continue
            u, acc = d['u'], d['mem_correct'] / n
            ax.plot(u, acc, color=color, ls=ls, lw=2, label=name)
            if r['status'] == 'reached90':
                ax.plot(u[-1], acc[-1], 'o', color=color, ms=5, mec='#fcfcfb', mew=1)
        ax.axhline(0.9, color='#c9c8c3', lw=0.8, zorder=0)
        ax.set_title(f'N = {n:,} 枚（上限 {U_MAX[n]:,} 更新）', fontsize=10)
        ax.set_xlabel('更新')
        ax.set_ylim(0, 1.02)
        ax.grid(True, color='#ebeae6', lw=0.6)
        ax.set_axisbelow(True)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel('暗記率（N 枚、評価の状態）')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=6, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle('課題 100 末の 4 状態と新品が、未使用の画像 N 枚の乱数ラベルを覚える速さ（2 ブロック ViT・seed 0）',
                 fontsize=11, y=1.07)
    fig.text(0.01, -0.04, '25 更新ごとに N 枚全部で暗記率を測り、0.9 に届いたら止めた（丸）。灰色の線は 0.9。'
             '各状態は自分の腕の規則で学ぶ（戻す腕と新品は head を 0 に、共有の腕は課題 1 の 10 行のまま）。',
             fontsize=8, color='#52514e')
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / 'capacity_curves.png', dpi=140, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    text = table()
    (out / 'capacity_table.md').write_text(text + '\n')
    print(text)
    figure(out)
    print(out / 'capacity_curves.png')
