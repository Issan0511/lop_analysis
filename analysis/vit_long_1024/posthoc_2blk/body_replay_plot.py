"""body_replay.py の結果の図と表（Claude、1026）: 課題の中の更新ごとの、勾配の揃い（coh）・歩幅（step/lr²d）・
w 方向の成分（rad_g、rad_d）を、SL と R0L、課題 11・61・101 で。100 更新の窓で平均し、横軸は対数。

    python analysis/vit_long_1024/posthoc_2blk/body_replay_plot.py <npz の置き場> <図と表の置き場>
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'font.size': 10, 'axes.edgecolor': '#c9c8c3',
                     'axes.labelcolor': '#52514e', 'xtick.color': '#52514e', 'ytick.color': '#52514e',
                     'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb'})
ARMS = [('R0L', 'R0L（LN あり・戻す）', '#2a78d6'), ('SL', 'SL（LN あり・共有）', '#eb6834')]
TS = (10, 60, 100)
W = 100
ROWS = [('loss', 'batch の損失（対数）', True), ('coh', '勾配の向きの揃い coh\n= rms(m̂/√v̂)', False),
        ('step', '歩幅 ‖Δw‖² ÷ lr²d', True), ('rad_g', '勾配の w 方向の成分 rad_g\n（正 = 縮める向き）', False),
        ('rad_d', '1 歩の相対ノルム変化 rad_d\n（正 = 伸びる）', False)]


def windows(a):
    n = len(a) // W
    return np.arange(1, n + 1) * W - W / 2, a[:n * W].reshape(n, W).mean(1)


def main(src, out):
    data = {(a, t): np.load(src / f'body_{a}_T{t}.npz') for a, _, _ in ARMS for t in TS if (src / f'body_{a}_T{t}.npz').exists()}
    for layer in (1, 0):
        fig, axes = plt.subplots(len(ROWS), len(TS), figsize=(11.5, 11.0), sharex=True, squeeze=False)
        for j, t in enumerate(TS):
            for i, (key, label, logy) in enumerate(ROWS):
                ax = axes[i][j]
                for arm, name, color in ARMS:
                    d = data.get((arm, t))
                    if d is None:
                        continue
                    series = d['loss'] if key == 'loss' else d[key][:, layer]
                    x, y = windows(series)
                    ax.plot(x, y, color=color, lw=1.6, label=name)
                if logy:
                    ax.set_yscale('log')
                else:
                    ax.axhline(0, color='#c9c8c3', lw=0.8, zorder=0)
                ax.set_xscale('log')
                ax.grid(True, color='#ebeae6', lw=0.6)
                ax.set_axisbelow(True)
                for s in ('top', 'right'):
                    ax.spines[s].set_visible(False)
                if i == 0:
                    ax.set_title(f'課題 {t} の末 → 課題 {t + 1}', fontsize=10)
                if j == 0:
                    ax.set_ylabel(label, fontsize=9)
                if i == len(ROWS) - 1:
                    ax.set_xlabel('課題の中の更新（100 更新の窓、対数）')
        h, l = axes[0][0].get_legend_handles_labels()
        fig.legend(h, l, loc='upper center', ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.995))
        fig.suptitle(f'{layer + 1} 層目の fc1：課題の胴の回し直し（保存状態＋Adam から 5,000 更新）', y=1.02, fontsize=11)
        fig.text(0.01, -0.01, 'coh は Adam の比 m̂/√v̂ の rms（勾配の向きが揃うと 1 に近づく）。歩幅は unit ごとの ‖Δw‖² ÷ lr²d の平均。'
                 'rad_g = Σ g·ŵ ÷ Σ‖g‖、rad_d = mean Δw·w ÷ ‖w‖²。white-san で回し直し（元の走は借りた機械）。',
                 fontsize=8, color='#52514e')
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(out / f'body_replay_L{layer + 1}.png', dpi=140, bbox_inches='tight')
        plt.close(fig)
        print(out / f'body_replay_L{layer + 1}.png')
    lines = ['| 腕 | 切替 | 窓（更新） | 損失 | 2 層目 coh | 歩幅 ÷ lr²d | rad_g | rad_d（×1e−5） | frac_d | 1 層目 coh | 歩幅 | rad_g | rad_d（×1e−5） |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for arm, _, _ in ARMS:
        for t in TS:
            d = data.get((arm, t))
            if d is None:
                continue
            for a, b in [(0, 100), (100, 500), (500, 2000), (2000, 5000)]:
                s = slice(a, b)
                f = lambda k, i: float(d[k][s, i].mean())
                lines.append(f"| {arm} | {t}→{t + 1} | {a + 1}〜{b} | {float(d['loss'][s].mean()):.1e} | {f('coh', 1):.3f} | {f('step', 1):.3f} | {f('rad_g', 1):+.4f} | {1e5 * f('rad_d', 1):+.2f} | {f('frac_d', 1):.4f}"
                             f" | {f('coh', 0):.3f} | {f('step', 0):.3f} | {f('rad_g', 0):+.4f} | {1e5 * f('rad_d', 0):+.2f} |")
    (out / 'body_replay_table.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main(Path(sys.argv[1]), Path(sys.argv[2]))
