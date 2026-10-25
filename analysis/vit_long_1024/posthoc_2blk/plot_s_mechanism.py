"""S（LN なし・head 共有）2 ブロック ViT・seed 0・100 課題の 2 枚（Claude、1025）。
図 1: 1 層目は CLS でだけ沈み、全 token では沈まない（layer_stats.csv、全課題）。
図 2: 2 層目の死が戻った担い手の鎖（保存状態 38 個、common_mode_S.json と spread_S.json）。"""
import csv, json, sys
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'font.size': 10, 'axes.edgecolor': '#c9c8c3',
                     'axes.labelcolor': '#52514e', 'xtick.color': '#52514e', 'ytick.color': '#52514e',
                     'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb'})
C1, C2, C3, C4, INK, MUTE = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#52514e', '#8a8984'
S, OUT = Path(sys.argv[1]), Path(sys.argv[2])
runs = Path.home() / 'Projects/obsidian-research-data/vit_long_1024_2blk_t100/runs'

def style(ax, ylab):
    ax.grid(True, color='#ebeae6', lw=0.6); ax.set_axisbelow(True)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    ax.set_ylabel(ylab, fontsize=9)

def end_label(ax, x, y, s, dy=0):
    ax.annotate(s, (x, y), xytext=(5, dy), textcoords='offset points', va='center', fontsize=8, color=INK)

# 図 1
rows = list(csv.DictReader(open(runs / 'S/seed0/layer_stats.csv')))
get = lambda L, k: ([0 if r['state'] == 'init' else int(r['task']) for r in rows if int(r['layer']) == L],
                    [float(r[k]) for r in rows if int(r['layer']) == L])
fig, axes = plt.subplots(3, 1, figsize=(7.0, 7.8), sharex=True)
ax = axes[0]
for k, lab, c in [('r_med', '全 token（CLS＋パッチ 64）', C1), ('cls_r_med', 'CLS の位置だけ', C2)]:
    x, y = get(1, k); ax.plot(x, y, color=c, lw=2, label=lab); end_label(ax, x[-1], y[-1], lab.split('（')[0])
ax.axhline(0, color='#c9c8c3', lw=0.8, zorder=0); style(ax, '1 層目の相対位置 r\n（unit の中央値）')
ax.legend(frameon=False, fontsize=8, loc='lower left')
ax = axes[1]
for k, lab, c in [('dead005', '死（全 token で p < 0.005）', C1), ('cls_dead0', 'CLS の位置で一度も開かない', C2)]:
    x, y = get(1, k); ax.plot(x, y, color=c, lw=2, label=lab)
style(ax, '1 層目の unit の割合'); ax.legend(frameon=False, fontsize=8, loc='upper left')
ax = axes[2]
for L, c in [(1, C3), (2, C4)]:
    x, y = get(L, 'in_mean_frac'); ax.plot(x, y, color=c, lw=2, label=f'{L} 層目の FFN の入力'); end_label(ax, x[-1], y[-1], f'{L} 層目')
style(ax, '入力のうち共通成分の割合\n‖m‖² ÷ E‖u‖²'); ax.set_xlabel('課題'); ax.legend(frameon=False, fontsize=8, loc='center right')
fig.suptitle('S（LN なし・head 共有）の 1 層目は CLS の位置でだけ沈み、全 token では沈まない', fontsize=11)
fig.text(0.01, 0.005, '2 ブロック ViT・FFN 3072・seed 0・課題末の固定 probe。m は FFN 入力の token 平均。死は全 token（1 画像 65 個）で数える。', fontsize=8, color=INK)
fig.tight_layout(rect=(0, 0.02, 1, 0.97)); fig.savefig(OUT / 'S_layer1_cls_only_2blk_t100.png', dpi=140); plt.close(fig)

# 図 2
cm = {r['task']: r for r in json.load(open(S / 'common_mode_S.json'))}
sp = {r['task']: r for r in json.load(open(S / 'spread_S.json'))}
t = sorted(set(cm) & set(sp))
mk = dict(marker='o', ms=3.2, mec='#fcfcfb', mew=0.8)
fig, axes = plt.subplots(4, 1, figsize=(7.0, 9.6), sharex=True, gridspec_kw=dict(height_ratios=[1.5, 1, 1.3, 1]))
ax = axes[0]
tot = [cm[i]['L2_norm_m'] for i in t]; ffn = [cm[i]['L2_contrib']['FFN1・重み'] for i in t]
oth = [tot[j] - ffn[j] for j in range(len(t))]
ax.plot(t, tot, color=INK, lw=1.6, ls='--', label='合計 ‖m‖')
ax.plot(t, ffn, color=C1, lw=2, label='1 層目の FFN の出力の寄与', **mk)
ax.plot(t, oth, color=C3, lw=2, label='それ以外（注意・埋め込み・bias）', **mk)
style(ax, '2 層目の FFN 入力の\n共通成分 ‖m‖ と担い手'); ax.legend(frameon=False, fontsize=8, loc='upper left')
ax = axes[1]
ratio = [sp[i]['f_norm'] / sp[i]['rms_u'] for i in t]
ax.plot(t, ratio, color=C1, lw=2, **mk); ax.set_ylim(0, 12)
style(ax, 'FFN1 が書く共通成分 ÷\nFFN1 の入力の広がり')
ax = axes[2]
for k, lab, c in [('a2', '1 層目の注意の出力', C2), ('e2', '埋め込み', C3), ('cross', '交差', MUTE)]:
    y = [sp[i][k] for i in t]; ax.plot(t, y, color=c, lw=2, label=lab, **mk)
style(ax, 'FFN1 の入力の広がり²\n（token ごとの部分）の内訳'); ax.legend(frameon=False, fontsize=8, loc='upper left')
ax = axes[3]
ax.plot(t, [cm[i]['L2_dead'] for i in t], color=C1, lw=2, **mk)
style(ax, '2 層目の\n死んだ unit の割合'); ax.set_xlabel('課題'); ax.set_ylim(0, 1)
fig.suptitle('S の 2 層目が戻った担い手：FFN1 の入力の広がり → FFN1 が書く共通成分 → 2 層目の死', fontsize=11)
fig.text(0.01, 0.005, '保存状態 38 個（課題 0〜10・20〜80 は 10 おき・81〜100）。広がり = 入力から token 平均を引いた残りの rms。\nFFN1 の ReLU は正の斉次（relu(λz) = λ·relu(z)）で bias ≈ 0 なので、書く共通成分は入力の広がりに比例する。', fontsize=8, color=INK)
fig.tight_layout(rect=(0, 0.035, 1, 0.97)); fig.savefig(OUT / 'S_layer2_revival_chain_2blk_t100.png', dpi=140); plt.close(fig)
print('ok')
