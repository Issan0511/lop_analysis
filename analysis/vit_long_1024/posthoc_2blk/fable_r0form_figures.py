"""A 形成則の図 2 枚（Fable エージェント A、依頼 4、1028）。

fable_r0form_timeseries.png: 課題の中の FFN0 出力の patch rms（log10）と暗記率、R0 と S、課題 1・10・30・60 から。
fable_r0form_phases.png: (a) 相ごとの log 増分、(b) 力の符号 −∂L/∂α と 50 更新先の log 増分、(c) 保存状態の FFN0 の平均出力の分解。

    MPLCONFIGDIR=<出力>/fable_mpl_config PYTHONPATH=$PWD python analysis/vit_long_1024/posthoc_2blk/fable_r0form_figures.py
"""
import json
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from analysis.vit_long_1024.posthoc_2blk.codex_r0_common import RESULT, ROOT

OUT = ROOT / 'posthoc_r0_form'
plt.rcParams['font.family'] = ['Noto Sans CJK JP', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
COLORS = {1: '#9ecae1', 10: '#4292c6', 30: '#08519c', 60: '#e6550d', 40: '#969696'}


def load(name):
    p = OUT / f'fable_replay_{name}.json'
    return json.loads(p.read_text()) if p.exists() else None


def timeseries():
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for col, arm in enumerate(['R0', 'S']):
        for t in [1, 10, 30, 60]:
            d = load(f'{arm}_t{t}_keep')
            if d is None:
                continue
            steps = [r['step'] for r in d['records']]
            ax = axes[0, col]
            ax.plot(steps, [math.log10(r['ffn0_patch_rms']) for r in d['records']], color=COLORS[t], label=f'課題 {t}→{t + 1}')
            if d['T90'] is not None:
                ax.plot([d['T90']], [math.log10([r for r in d['records'] if r['step'] == d['T90']][0]['ffn0_patch_rms'])], 'o', color=COLORS[t], ms=5)
            axes[1, col].plot(steps, [r['acc'] for r in d['records']], color=COLORS[t], label=f'課題 {t}→{t + 1}')
        for name, ls, lab in (('R0_t30_zero_m', '--', 'R0 30→31 体の m を 0'), ('R0_t30_zero_mv', ':', 'R0 30→31 体の m・v を 0')):
            d = load(name)
            if d is not None and arm == 'R0':
                steps = [r['step'] for r in d['records']]
                axes[0, col].plot(steps, [math.log10(r['ffn0_patch_rms']) for r in d['records']], ls=ls, color=COLORS[30], label=lab)
                axes[1, col].plot(steps, [r['acc'] for r in d['records']], ls=ls, color=COLORS[30])
        axes[0, col].set_title(f'{arm}（LN なし、{"head を毎課題 0 に戻す" if arm == "R0" else "共有 head"}）')
        axes[0, col].set_ylabel('FFN0 出力の patch rms（log10）')
        axes[1, col].set_ylabel('暗記率（訓練 2,500 枚）')
        axes[1, col].set_xlabel('課題の中の更新数')
        axes[0, col].legend(fontsize=8)
        axes[0, col].grid(alpha=.3)
        axes[1, col].grid(alpha=.3)
        axes[1, col].axhline(.9, color='k', lw=.5, ls='--')
    fig.suptitle('課題の中の尺度の時系列（保存状態からの回し直し、● は T90）')
    fig.tight_layout()
    fig.savefig(RESULT / 'fable_r0form_timeseries.png', dpi=130)


def phases():
    summ = json.loads((OUT / 'fable_summary.json').read_text())['runs']
    coh = json.loads((OUT / 'fable_coherence.json').read_text())['rows']
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    # (a) 相ごとの log 増分
    ax = axes[0]
    names = [n for n in ['R0_t1_keep', 'S_t1_keep', 'R0_t10_keep', 'S_t10_keep', 'R0_t30_keep', 'S_t30_keep', 'R0_t60_keep', 'S_t60_keep'] if n in summ]
    x = np.arange(len(names))
    w = .27
    for i, (ph, lab, c) in enumerate((('switch', '切替相（暗記率 0.2 まで）', '#e6550d'), ('fit', '当てはめ相（T90 まで）', '#3182bd'), ('post', '当てはめ後', '#969696'))):
        vals = [summ[n]['phases']['ffn0_patch_rms'][ph] or 0. for n in names]
        ax.bar(x + (i - 1) * w, vals, w, label=lab, color=c)
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace('_keep', '').replace('_t', ' ') for n in names], rotation=30, fontsize=8)
    ax.axhline(0, color='k', lw=.5)
    ax.set_ylabel('FFN0 出力 patch rms の log 増分')
    ax.set_title('(a) 相ごとの増分')
    ax.legend(fontsize=8)
    # (b) 窓の暗記率の変化（崩れ）と増分
    ax = axes[1]
    for arm, c in (('R0', '#08519c'), ('S', '#e6550d')):
        xs, ys = [], []
        for n, s in summ.items():
            if not n.startswith(arm) or not n.endswith('keep'):
                continue
            for r in s['force']['rows']:
                xs.append(r['dacc'])
                ys.append(r['dlog'])
        ax.scatter(xs, ys, s=10, alpha=.6, color=c, label=arm)
    ax.axhline(0, color='k', lw=.5)
    ax.axvline(0, color='k', lw=.5)
    ax.axvline(-.1, color='k', lw=.5, ls=':')
    ax.set_xlabel('50 更新窓の暗記率の変化（−0.1 以下 = 崩れ窓）')
    ax.set_ylabel('同じ窓の FFN0 出力 patch rms の log 増分')
    ax.set_title('(b) 崩れ窓と増分（R0/S の keep 走の全窓）')
    ax.legend(fontsize=8)
    # (c) 保存状態の FFN0 の分解
    ax = axes[2]
    ts = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    for arm, ls in (('R0', '-'), ('S', '--')):
        ax.plot(ts, [math.log10(coh[f'{arm}_t{t}']['ffn0_abar_norm']) for t in ts], ls=ls, color='#3182bd', label=f'{arm} ‖ā‖（開いた値の平均）')
        ax.plot(ts, [math.log10(coh[f'{arm}_t{t}']['ffn0_gain']) for t in ts], ls=ls, color='#e6550d', label=f'{arm} 利得 ‖W2 ā‖/‖ā‖')
        ax.plot(ts, [math.log10(coh[f'{arm}_t{t}']['ffn0_fc2_norm']) for t in ts], ls=ls, color='#969696', label=f'{arm} ‖W2‖_F')
    ax.set_xscale('log')
    ax.set_xlabel('課題末')
    ax.set_ylabel('log10')
    ax.set_title('(c) 保存状態の FFN0 の平均出力 W2 ā の分解')
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(RESULT / 'fable_r0form_phases.png', dpi=130)


if __name__ == '__main__':
    timeseries()
    phases()
    print('図を書いた')
