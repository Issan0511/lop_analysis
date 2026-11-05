"""Report direct path ratios with their denominators explicit."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/vit_ffn_ablation_1025'
d = pd.read_csv(OUT / 'path_ratios.csv')
v = json.loads((OUT / 'gradient_validation.json').read_text())
assert len(v) == 80 and len(d) == 1920
assert max(r['final_patch_gradient_max_abs'] for r in v) == 0
group = ['arm', 'task', 'state', 'step', 'layer', 'token', 'kind']
cols = [x for x in d if x.endswith('_median')]
agg = d.groupby(group)[cols].agg(['median', 'min', 'max'])
agg.columns = ['_'.join(c) for c in agg.columns]
agg.reset_index().to_csv(OUT / 'path_ratio_aggregate.csv', index=False)
lines = ['# 残差経路と FFN 経路の直接比率（1025、事後）', '',
         '各 FFN を y = r + F(r) と書き、F は pre-LN と FFN を含める。出力比率は ‖F(r)‖ / (‖F(r)‖ + ‖r‖)。入力へ戻る勾配は g = ∂L/∂y と a = J_Fᵀg に分け、FFN 比率を ‖a‖ / (‖a‖ + ‖g‖) とする。**加算前のノルムの合計を 100% にした比率で、加算後の勾配や学習量の分割ではない。** FFN 出力自体には g が届くので、FFN に配られる誤差がこの割合になるわけでもない。', '',
         '各画像の比率を checkpoint 内で中央値にし、さらに 5 seed の中央値・範囲を取る。以下は CLS。全 token の比率も CSV に保存した。task 60 / update 500 の最終 FFN の勾配比率は、共有 head 55.7%、reset head 74.4%。', '',
         '| head | update | FFN6 の出力比率 | FFN6 の入力勾配比率（seed 範囲） | 勾配2経路のcos |',
         '|---|---:|---:|---:|---:|']
for arm in ('R0L', 'SL'):
    for state in ('mid', 'end'):
        s = d[(d.arm == arm) & (d.task == 60) & (d.layer == 6) & (d.state == state) & (d.token == 'cls')]
        o, g = s[s.kind == 'output'], s[s.kind == 'gradient']
        values = 100 * g.ffn_norm_share_median
        lines.append(f'| {arm} | {int(g.step.iloc[0])} | {100 * o.ffn_norm_share_median.median():.1f}% | {values.median():.1f}%（{values.min():.1f}〜{values.max():.1f}%） | {g.cosine_median.median():.3f} |')
lines += ['', '## 6 ブロックの比較', '', 'task 60 / update 500。FFN 経路の入力勾配比率の seed 中央値。', '',
          '| FFN | reset head | shared head |', '|---:|---:|---:|']
for layer in range(1, 7):
    s = d[(d.task == 60) & (d.state == 'end') & (d.layer == layer) & (d.token == 'cls') & (d.kind == 'gradient')]
    lines.append(f'| {layer} | {100 * s[s.arm == "R0L"].ffn_norm_share_median.median():.1f}% | {100 * s[s.arm == "SL"].ffn_norm_share_median.median():.1f}% |')
lines += ['', '![経路ごとの比率](direct_path_ratios.png)', '',
          '中間 FFN では残差経路の比重が大きく、最終 FFN の比率は再び高い。この測定は「入力まで戻る勾配」を扱う。比率が小さくても FFN の parameter gradient が小さいとは限らない。絶対勾配が小さい学習済み状態でも比率は定義できるため、比率だけで沈降の強さを決めない。', '',
          '補助の `parameter_gradients.csv` は課題全 250 画像の平均 CE に対する勾配で、各 FFN の norm2 + mlp の squared norm が全 parameter の squared norm に占める比率と、parameter 当たり RMS を記録する。parameterization に依存し、上の経路比率とは異なる。', '',
          f'検算: 80/80 保存状態。経路勾配の和と autograd の入力勾配の最大絶対差 {max(r["gradient_decomposition_max_abs_error"] for r in v):.3g}。元の記録との CE 最大絶対差 {max(r["baseline_loss_abs_error"] for r in v):.3g}。最終 FFN の patch token 勾配は全件 0。学習更新なし。']
(OUT / 'direct_ratios.md').write_text('\n'.join(lines) + '\n')
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
for ax, kind, title in zip(axes, ('output', 'gradient'), ('Output norm share', 'Input-gradient norm share')):
    for arm, color, label in [('R0L', '#159b8d', 'Reset head'), ('SL', '#3069ad', 'Shared head')]:
        s = d[(d.arm == arm) & (d.task == 60) & (d.state == 'end') & (d.token == 'cls') & (d.kind == kind)]
        g = 100 * s.groupby('layer').ffn_norm_share_median.agg(['median', 'min', 'max'])
        ax.plot(g.index, g['median'], 'o-', color=color, label=label)
        ax.fill_between(g.index, g['min'], g['max'], color=color, alpha=.15)
    ax.set(title=title, xlabel='FFN block (1 = nearest input)', ylabel='FFN / (FFN + skip) norm, %', ylim=(0, 100), xticks=range(1, 7))
    ax.grid(alpha=.2)
axes[0].legend(frameon=False)
fig.suptitle('Task 60, update 500, CLS; seed median and range (n = 5)')
for ext in ('png', 'pdf'):
    fig.savefig(OUT / f'direct_path_ratios.{ext}', dpi=180)
plt.close(fig)
print((OUT / 'direct_ratios.md').read_text())
