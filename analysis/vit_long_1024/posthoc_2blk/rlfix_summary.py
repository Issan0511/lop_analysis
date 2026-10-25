"""追補 D の要約（Claude、1026）: T90・課題の始めの損失・2 層目の死の図、層ごとの沈みの図、
head に入る CLS の特徴の大きさ（保存状態・固定 probe）。

    PYTHONPATH=$PWD VIT_LONG_PRODUCTION=production_2blk_rlfix.json \
        python analysis/vit_long_1024/posthoc_2blk/rlfix_summary.py results/vit_long_1024_2blk_rlfix
"""
import csv
import json
import statistics as St
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

from analysis.vit_long_1024 import common as C

plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'font.size': 10, 'axes.edgecolor': '#c9c8c3',
                     'axes.labelcolor': '#52514e', 'xtick.color': '#52514e', 'ytick.color': '#52514e',
                     'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb'})
RUNS = Path.home() / 'Projects/obsidian-research-data/vit_long_1024_2blk_rlfix/runs'
SEED = 10252010
U = 5000
ARMS = [('R0L', 'R0L（LN あり・戻す）', '#2a78d6'), ('SL', 'SL（LN あり・共有）', '#eb6834'),
        ('R0', 'R0（LN なし・戻す）', '#1baf7a'), ('S', 'S（LN なし・共有）', '#eda100')]
INK = '#52514e'


def rows(arm):
    return list(csv.DictReader(open(RUNS / arm / f'seed{SEED}' / 'per_task.csv')))


def layers(arm):
    out = {}
    for r in csv.DictReader(open(RUNS / arm / f'seed{SEED}' / 'layer_stats.csv')):
        out[(0 if r['state'] == 'init' else int(r['task']), int(r['layer']))] = r
    return out


def fresh_t90(name):
    return {t: int(list(csv.DictReader(open(RUNS / name / f'seed{SEED}' / f'task{t}' / 'per_task.csv')))[0]['T90'])
            for t in (25, 50, 75, 100)}


def style(ax):
    ax.grid(True, color='#ebeae6', lw=0.6)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


def figure_t90(out):
    fig, axes = plt.subplots(3, 1, figsize=(8.2, 9.6), sharex=True, gridspec_kw=dict(height_ratios=[1.6, 1, 1]))
    ax = axes[0]
    for name, ls, label in [('freshL', ':', '新品（LN あり）'), ('fresh', '--', '新品（LN なし）')]:
        v = list(fresh_t90(name).values())
        ax.axhline(St.median(v), color=INK, ls=ls, lw=1.4, label=f'{label}（課題 25〜100 の中央値 {St.median(v):.0f}）')
    for arm, label, color in ARMS:
        rr = rows(arm)
        t, y = [int(r['task']) for r in rr], [int(r['T90']) for r in rr]
        ax.plot(t, y, color=color, lw=1.6, label=label)
        un = [(a, b) for a, b in zip(t, y) if b > U]
        if un:
            ax.plot([a for a, _ in un], [b for _, b in un], 'x', color=color, ms=6, mew=1.6)
    ax.set_yscale('log')
    ax.set_ylabel('T90（更新、対数）\n× は 5,000 更新で未到達')
    ax.legend(frameon=False, fontsize=8, ncol=2, loc='lower right', bbox_to_anchor=(1.0, 0.15))
    ax = axes[1]
    for arm, label, color in ARMS:
        if arm in ('SL', 'S'):
            rr = rows(arm)
            ax.plot([int(r['task']) for r in rr], [float(r['mem_loss_start']) for r in rr], color=color, lw=1.6, label=label)
    ax.axhline(2.3026, color='#c9c8c3', lw=0.8)
    ax.set_ylabel('課題の始めの暗記損失\n（戻す腕は 2.30）')
    ax.legend(frameon=False, fontsize=8, loc='upper right')
    ax = axes[2]
    for arm, label, color in ARMS:
        ls = layers(arm)
        t = sorted(k[0] for k in ls if k[1] == 2)
        ax.plot(t, [float(ls[(x, 2)]['dead005']) for x in t], color=color, lw=1.6, label=label)
    ax.set_ylim(-0.03, 1)
    ax.set_ylabel('2 層目の\n死んだ unit の割合')
    ax.set_xlabel('課題')
    for a in axes:
        style(a)
    fig.suptitle('追補 D：同じ 2,500 枚にラベルを引き直す・5,000 更新/課題・100 課題・seed 10252010', fontsize=11)
    fig.text(0.01, 0.005, 'SL・R0L は借りた機械、S・R0・新品は white-san。T90 は 25 更新ごとに 2,500 枚で測る。'
             '1 層目の死は 4 腕とも全課題で 0。', fontsize=8, color=INK)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(out / 'rlfix_t90_100tasks.png', dpi=140)
    plt.close(fig)


def figure_sinking(out):
    metrics = [('r_med', '相対位置 r（全 token）', False), ('cls_r_med', 'CLS 位置の相対位置 r', False),
               ('sigma_med', '前活性の幅 sd（対数）', True), ('dead005', '死んだ unit の割合', False)]
    fig, axes = plt.subplots(4, 2, figsize=(7.4, 9.8), sharex=True, squeeze=False)
    for i, (key, label, logy) in enumerate(metrics):
        for j, L in enumerate((1, 2)):
            ax = axes[i][j]
            for arm, name, color in ARMS:
                ls = layers(arm)
                t = sorted(k[0] for k in ls if k[1] == L)
                ax.plot(t, [float(ls[(x, L)][key]) for x in t], color=color, lw=1.6, label=name)
            if key in ('r_med', 'cls_r_med'):
                ax.axhline(0, color='#c9c8c3', lw=0.8, zorder=0)
            if logy:
                ax.set_yscale('log')
            if key == 'dead005':
                ax.set_ylim(-0.03, 1.0)
            style(ax)
            if i == 0:
                ax.set_title(f'{L} 層目', fontsize=10)
            if j == 0:
                ax.set_ylabel(label, fontsize=9)
            if i == len(metrics) - 1:
                ax.set_xlabel('課題')
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle('追補 D の層ごとの沈み（課題末の固定 probe、unit の中央値）', y=1.02, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out / 'rlfix_sinking_100tasks.png', dpi=140, bbox_inches='tight')
    plt.close(fig)


@torch.no_grad()
def readout_scale():
    """head に入る CLS の特徴（最後の norm の後）の ‖h‖ と lr × Σ|h_i| を、保存状態ごとに固定 probe で。"""
    source = C.image_source('real')
    x = C.eval_tensor(source.val(C.probe_names(source, SEED)), 'cpu')
    out = {}
    for arm, _, _ in ARMS:
        ln = arm in C.LN_ARMS
        out[arm] = {}
        for path in sorted((RUNS / arm / f'seed{SEED}' / 'ckpt').glob('model_t*.pt')):
            t = int(path.stem.split('_t')[1])
            m = C.build_model(SEED, 'cpu', 0.0, ln, 2, 8.0)
            m.load_state_dict(torch.load(path, map_location='cpu', weights_only=False))
            m.eval()
            L = m.layers
            h = L['patch_embed'](x)
            h = torch.cat((m.cls_token.expand(h.shape[0], -1, -1), h), 1) + m.pos_embed
            for i in range(m.depth):
                b = L[f'block_{i}'].layers
                h = h + b['attn'](b['norm1'](h))
                h = h + b['mlp'](b['norm2'](h))
            feat = L['norm'](h)[:, 0].double()
            out[arm][t] = dict(norm=float(feat.norm(dim=1).mean()), l1_lr=float(C.LR * feat.abs().sum(1).mean()))
    return out


def figure_scale(scale, out):
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    for arm, name, color in ARMS:
        t = sorted(scale[arm])
        ax.plot(t, [scale[arm][x]['l1_lr'] for x in t], color=color, lw=1.8, marker='o', ms=3.2, mec='#fcfcfb',
                mew=0.8, label=name)
    ax.set_yscale('log')
    ax.set_ylabel('lr × Σ|h_i|\n（Adam の 1 歩で logit が動く目安）')
    ax.set_xlabel('課題（保存した課題末）')
    style(ax)
    ax.legend(frameon=False, fontsize=8, loc='upper left')
    fig.suptitle('head に入る CLS の特徴の大きさ（固定 probe 64 枚の平均）', fontsize=11)
    fig.tight_layout()
    fig.savefig(out / 'rlfix_readout_scale.png', dpi=140)
    plt.close(fig)


if __name__ == '__main__':
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    figure_t90(out)
    figure_sinking(out)
    scale = readout_scale()
    (out / 'readout_scale.json').write_text(json.dumps(scale, indent=1))
    figure_scale(scale, out)
    for arm in scale:
        t = sorted(scale[arm])
        print(arm, ' '.join(f"t{x}:{scale[arm][x]['norm']:.0f}/{scale[arm][x]['l1_lr']:.2f}" for x in t if x in (1, 10, 30, 50, 70, 100)))
