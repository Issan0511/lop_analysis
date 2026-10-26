"""switch_replay.py の結果の図（Claude、1026）: 課題の中の更新ごとの相対位置 r・平均・幅を、SL と R0L で。

    python analysis/vit_long_1024/posthoc_2blk/switch_replay_plot.py <JSON の置き場> <図の置き場>
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'font.size': 10, 'axes.edgecolor': '#c9c8c3',
                     'axes.labelcolor': '#52514e', 'xtick.color': '#52514e', 'ytick.color': '#52514e',
                     'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb'})
ARMS = [('R0L', 'R0L（LN あり・戻す）', '#2a78d6'), ('SL', 'SL（LN あり・共有）', '#eb6834'),
        ('R0', 'R0（LN なし・戻す）', '#1baf7a'), ('S', 'S（LN なし・共有）', '#eda100')]
TS = (10, 30, 60, 100)
METRICS = [('r_med', '相対位置 r'), ('mu_med', '前活性の平均'), ('sigma_med', '前活性の幅')]


def main(src, out):
    data = {(a, t): json.loads((src / f'{a}_T{t}.json').read_text()) for a, _, _ in ARMS for t in TS
            if (src / f'{a}_T{t}.json').exists()}
    for layer in (1, 2):
        fig, axes = plt.subplots(len(METRICS), len(TS), figsize=(12.0, 7.6), sharex=True, squeeze=False)
        for j, t in enumerate(TS):
            for i, (key, label) in enumerate(METRICS):
                ax = axes[i][j]
                for arm, name, color in ARMS:
                    d = data.get((arm, t))
                    if d is None:
                        continue
                    x = [s['step'] for s in d['snapshots']]
                    y = [s['layers'][layer - 1][key] for s in d['snapshots']]
                    ax.plot(x, y, color=color, lw=1.8, marker='o', ms=3, mec='#fcfcfb', mew=0.6, label=name)
                ax.set_xscale('symlog', linthresh=1)
                if key == 'r_med':
                    ax.axhline(0, color='#c9c8c3', lw=0.8, zorder=0)
                ax.axvline(100, color='#c9c8c3', lw=0.8, ls=':', zorder=0)
                ax.grid(True, color='#ebeae6', lw=0.6)
                ax.set_axisbelow(True)
                for s in ('top', 'right'):
                    ax.spines[s].set_visible(False)
                if i == 0:
                    ax.set_title(f'課題 {t} の末 → 課題 {t + 1}', fontsize=10)
                if j == 0:
                    ax.set_ylabel(label, fontsize=9)
                if i == len(METRICS) - 1:
                    ax.set_xlabel('課題の中の更新（対数、0 を含む）')
        h, l = axes[0][0].get_legend_handles_labels()
        fig.legend(h, l, loc='upper center', ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.995))
        fig.suptitle(f'{layer} 層目：切替の後の 5,000 更新で、前活性がどう動くか（保存状態から回し直し、固定 probe・unit の中央値）',
                     y=1.03, fontsize=11)
        fig.text(0.01, -0.01, '点線は 100 更新目。SL は head を残すので切替の直後に大きな誤差が体へ流れる。R0L は head を 0 に戻す。'
                 '元の走は借りた機械で、回し直しは white-san（課題の始めの暗記損失は元の記録と一致）。', fontsize=8, color='#52514e')
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(out / f'switch_replay_L{layer}.png', dpi=140, bbox_inches='tight')
        plt.close(fig)
        print(out / f'switch_replay_L{layer}.png')


if __name__ == '__main__':
    main(Path(sys.argv[1]), Path(sys.argv[2]))
