"""R0 後期の持続崩れの図 2 枚（Fable エージェント B、1028）。"""
import json
import os
from pathlib import Path

ROOT = Path('/home/issan/Projects/obsidian-research-data/vit_long_1024_2blk_rlfix')
OUT = ROOT / 'posthoc_r0_late'
RESULT = Path('/home/issan/Projects/claude/wt/vit_long_1024/results/vit_long_1024_2blk_rlfix')
os.environ['MPLCONFIGDIR'] = str(OUT / 'fable_mpl_config')
import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
import matplotlib.pyplot as plt

font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 150})
COL = {'base': '#1f4e79', 'br_body': '#2a9d8f', 'br_head': '#e76f51', 'br_ffn1': '#8e44ad', 'br_attn1': '#f4a261',
       'br_block0': '#264653', 'br_bl01': '#999999', 'bl001': '#06d6a0', 'bl01': '#b5838d'}
LABEL = {'base': 'head lr 1/100（分岐の元）', 'br_body': '体を全部凍結', 'br_head': 'head を凍結', 'br_ffn1': '最終 FFN だけ凍結',
         'br_attn1': '注意 1 だけ凍結', 'br_block0': 'ブロック 0 だけ凍結', 'br_bl01': '体の lr 1/10', 'bl001': '頭から両方 1/100',
         'bl01': '頭から体 1/10'}


def load(name):
    p = OUT / f'{name}.json'
    return json.load(p.open()) if p.exists() else None


def traces():
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharey='row')
    for col, task in enumerate((60, 30)):
        ax = axes[0, col]
        for key in ('base', 'br_body', 'br_head', 'br_ffn1', 'br_attn1', 'br_block0', 'br_bl01', 'bl001', 'bl01'):
            d = load(f'fable_R0_t{task}_hl001_{key}')
            if d is None:
                continue
            s = [r['step'] for r in d['records']]
            a = [r['acc'] for r in d['records']]
            ax.plot(s, a, color=COL[key], lw=1.4 if key == 'base' else 1., label=LABEL[key], alpha=.95)
            if key == 'base' and d['summary']['T90'] is not None:
                ax.axvline(d['summary']['T90'], color='k', ls=':', lw=.8)
        ax.set(title=f'R0 課題 {task}→{task + 1}（届いた時点で分岐）', xlabel='更新', ylabel='暗記率（訓練 2,500 枚）', ylim=(0, 1.02))
        ax.grid(alpha=.2)
        if col == 0:
            ax.legend(fontsize=7, loc='lower right', ncol=2)
        ax = axes[1, col]
        for key, c, lab in (('br_newhead1', '#2a9d8f', '新 head lr 1e-4'), ('br_newhead001', '#e76f51', '新 head lr 1e-6')):
            d = load(f'fable_R0_t{task}_hl001_{key}')
            if d is None:
                continue
            ax.plot([r['step'] for r in d['records']], [r['acc'] for r in d['records']], color=c, label=lab)
        ax.set(title=f'届いた時点の体を凍結し head を 0 から学び直す（課題 {task + 1}）', xlabel='更新（新しい 5,000 更新）',
               ylabel='暗記率', ylim=(0, 1.02))
        ax.grid(alpha=.2)
        ax.legend(fontsize=7, loc='lower right')
    fig.suptitle('R0 後期の持続崩れ：届いた時点からの分岐（eager 回し直し、1 seed）', fontsize=12)
    fig.tight_layout()
    fig.savefig(RESULT / 'fable_r0late_traces.png')


def steps():
    fig, axes = plt.subplots(3, 2, figsize=(11, 8.5), sharex='col')
    for col, task in enumerate((60, 30)):
        d = load(f'fable_track_R0_t{task}_from_t90')
        if d is None:
            continue
        q = d['per_step']
        s = [x['step'] for x in q]
        ax = axes[0, col]
        for key, c, lab in (('body', '#e76f51', '体 W(Δh)'), ('head', '#2a9d8f', 'head (ΔW)h')):
            ax.plot(s, [max(x[key], 1e-6) for x in q], color=c, lw=.7, label=lab, alpha=.9)
        ax.plot(s, [max(x['margin_median_before'], 1e-2) for x in q], color='k', lw=.9, ls='--', label='余裕の中央値（0.01 未満は 0.01）')
        ax.set_yscale('log')
        ax.set(title=f'課題 {task}→{task + 1}：届いた後の一歩の logit 変化（固定 512 枚、rms）', ylabel='logit')
        ax.grid(alpha=.2)
        ax.legend(fontsize=7, loc='lower right')
        ax = axes[1, col]
        for g, c, lab in (('block0', '#264653', 'ブロック 0'), ('attn1', '#f4a261', '注意 1'), ('ffn1', '#8e44ad', '最終 FFN')):
            ax.plot(s, [x['dtheta_l2_over_eta_sqrt_n'][g] for x in q], color=c, lw=.7, label=lab, alpha=.9)
        ax.axhline(1., color='k', lw=.6, ls=':')
        ax.set_yscale('log')
        ax.set(ylabel='|Δθ|_2 / (η √n)', title='Adam の一歩の大きさ（η を 1 とした座標あたりの比）')
        ax.grid(alpha=.2)
        ax.legend(fontsize=7, loc='lower right')
        ax = axes[2, col]
        rec = d['records']
        ax.plot([r['step'] for r in rec], [r['acc'] for r in rec], color='#1f4e79', label='暗記率')
        ax.plot(s, [x['batch_wrong'] / 128 for x in q], color='#999999', lw=.5, label='batch の誤りの割合')
        ax.plot(s, [x['flip_frac'] for x in q], color='#e76f51', lw=.5, label='一歩で予測が入れ替わる割合')
        ax.set(xlabel='更新', ylabel='割合', ylim=(0, 1.02))
        ax.grid(alpha=.2)
        ax.legend(fontsize=7, loc='center right')
    fig.suptitle('急落の更新では Adam の一歩が跳ね、体の項が余裕と同じ桁で logit を動かす', fontsize=12)
    fig.tight_layout()
    fig.savefig(RESULT / 'fable_r0late_steps.png')


if __name__ == '__main__':
    traces()
    steps()
    print('figures written')
