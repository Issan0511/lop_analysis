"""沈みの図: 層ごとの前活性の相対位置 r・CLS の r・幅・死を課題末で描く（Claude、1025）。
色は dataviz skill の既定の色の組の先頭 4 つ（検証済み、水色と黄は地との対比が弱いので線の端に名前を書く）。"""
import csv, statistics as St, sys
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'font.size': 10, 'axes.edgecolor': '#c9c8c3',
                     'axes.labelcolor': '#52514e', 'xtick.color': '#52514e', 'ytick.color': '#52514e',
                     'axes.titlecolor': '#0b0b0b', 'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb'})
ARMS = [('R0L', 'R0L（LN あり・戻す）', '#2a78d6'), ('SL', 'SL（LN あり・共有）', '#eb6834'),
        ('R0', 'R0（LN なし・戻す）', '#1baf7a'), ('S', 'S（LN なし・共有）', '#eda100')]
METRICS = [('r_med', '相対位置 r（全 token）', False), ('cls_r_med', 'CLS 位置の相対位置 r', False),
           ('sigma_med', '前活性の幅 sd（対数）', True), ('dead005', '死んだ unit の割合', False)]

def load(run_dir):
    d = {}
    f = Path(run_dir) / 'layer_stats.csv'
    if not f.exists(): return d
    for r in csv.DictReader(open(f)):
        t = 0 if r['state'] == 'init' else int(r['task'])
        d[(t, int(r['layer']))] = r
    return d

def series(datas, layer, key):
    tasks = sorted({t for d in datas for (t, L) in d if L == layer})
    xs, ys = [], []
    for t in tasks:
        v = [float(d[(t, layer)][key]) for d in datas if (t, layer) in d and d[(t, layer)][key] not in ('', 'nan')]
        if v: xs.append(t); ys.append(St.median(v))
    return xs, ys

def figure(root, seeds, layers, metrics, title, out, note):
    nrow, ncol = len(metrics), len(layers)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.6 * ncol + 1.2, 2.2 * nrow + 1.0), squeeze=False, sharex=True)
    for i, (key, lab, logy) in enumerate(metrics):
        for j, L in enumerate(layers):
            ax = axes[i][j]
            for arm, name, col in ARMS:
                datas = [load(Path(root) / arm / f'seed{s}') for s in seeds]
                datas = [d for d in datas if d]
                if not datas: continue
                xs, ys = series(datas, L, key)
                if not xs: continue
                ax.plot(xs, ys, color=col, lw=2, label=name, solid_capstyle='round')
                if i == 0 and j == ncol - 1:
                    ax.annotate(arm, (xs[-1], ys[-1]), xytext=(4, 0), textcoords='offset points',
                                va='center', fontsize=8, color='#52514e')
            if key in ('r_med', 'cls_r_med'): ax.axhline(0, color='#c9c8c3', lw=0.8, zorder=0)
            if logy: ax.set_yscale('log')
            if key == 'dead005': ax.set_ylim(-0.03, 1.0)
            ax.grid(True, color='#ebeae6', lw=0.6); ax.set_axisbelow(True)
            for s in ('top', 'right'): ax.spines[s].set_visible(False)
            if i == 0: ax.set_title(f'{L} 層目', fontsize=10)
            if j == 0: ax.set_ylabel(lab, fontsize=9)
            if i == nrow - 1: ax.set_xlabel('課題')
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle(title, y=1.04, fontsize=11)
    fig.text(0.01, -0.01, note, fontsize=8, color='#52514e')
    fig.tight_layout(rect=(0, 0.01, 1, 0.96))
    fig.savefig(out, dpi=140, bbox_inches='tight'); plt.close(fig); print(out)

if __name__ == '__main__':
    D = Path.home() / 'Projects/obsidian-research-data'
    out = Path(sys.argv[1])
    figure(D / 'vit_long_1024_2blk_t100/runs', [0], [1, 2], METRICS,
           '2 ブロック ViT・FFN 3072・seed 0・100 課題（乱数ラベル 10 値 × 250 枚、500 更新/課題）',
           out / 'sinking_2blk_t100.png',
           '課題末の固定 probe の値（unit の中央値）。相対位置 r = 前活性の平均 ÷ 幅、負ほど沈む。死は p < 0.005。0 は初期化。')
    figure(D / 'vit_long_1024/merged/vit_long_1024/runs', [0, 1, 2, 3, 4], [1, 2, 3, 4, 5, 6],
           [METRICS[0], METRICS[1], METRICS[3]],
           '6 ブロック ViT・FFN 1536・5 seed の中央値・60 課題（同じ箱、Vast.ai で実行）',
           out / 'sinking_6blk_t60.png',
           '課題末の固定 probe の値（unit の中央値を 5 seed で中央値）。6 層目が読み出し（head）の隣の FFN。')
