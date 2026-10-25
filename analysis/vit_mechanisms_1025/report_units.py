"""Post-hoc matched-unit audit; no new model evaluations or significance tests."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/vit_mechanisms_1025'
ARCHIVE = Path('/home/issan/Projects/obsidian-research-data')
PRIOR = ARCHIVE / 'vit_phases_1025'
CURRENT = ARCHIVE / 'vit_mechanisms_1025/raw'
sources = set()
rows, positions = [], []
for seed in range(5):
    for task in (10, 60):
        old_path = PRIOR / 'adam_raw' / f'SL_seed{seed}_t{task}.npz'
        next_path = PRIOR / 'adam_next_raw' / f'SL_seed{seed}_t{task}.npz'
        sources.update((old_path, next_path))
        with np.load(old_path) as old, np.load(next_path) as nxt:
            end_path = PRIOR / 'raw' / f'SL_seed{seed}_t{task}_end.npz'
            sources.add(end_path)
            with np.load(end_path) as probe:
                for layer in range(1, 7):
                    fit = old[f'L{layer}_fit_full_r'].astype(float)
                    for inputs, data in [('same_inputs', old), ('next_inputs', nxt)]:
                        load = data[f'L{layer}_load_full_r'].astype(float)
                        valid = probe[f'L{layer}_valid'].astype(bool)
                        keep = (valid & (np.abs(fit) > max(1e-12, 1e-6*np.abs(fit).max()))
                                & (np.abs(load) > max(1e-12, 1e-6*np.abs(load).max())))
                        aa, bb = fit[keep], load[keep]
                        fs, ls = aa > 0, bb > 0
                        opposite = float(np.mean(fs != ls))
                        independent = float(fs.mean()*(1-ls.mean()) + (1-fs.mean())*ls.mean())
                        cells = dict(both_sink_fraction=float(np.mean(fs & ls)),
                                     switch_only_sink_fraction=float(np.mean(~fs & ls)),
                                     fit_only_sink_fraction=float(np.mean(fs & ~ls)),
                                     both_float_fraction=float(np.mean(~fs & ~ls)))
                        assert abs(sum(cells.values()) - 1) < 1e-12
                        rows.append(dict(seed=seed, task=task, layer=layer, inputs=inputs,
                                         n=int(keep.sum()), opposite_fraction=opposite, **cells,
                                         independent_opposite_fraction=independent,
                                         excess_opposite_fraction=opposite-independent,
                                         cosine=float(aa@bb/(np.linalg.norm(aa)*np.linalg.norm(bb))),
                                         norm_ratio=float(np.linalg.norm(bb)/np.linalg.norm(aa))))
        for stage, metric in [('mid', 'SGD'), ('end', 'Adam_fixed')]:
            cov_path = CURRENT / f'covariance_seed{seed}_t{task}_{stage}.npz'
            probe_path = PRIOR / 'raw' / f'SL_seed{seed}_t{task}_{stage}.npz'
            sources.update((cov_path, probe_path))
            with np.load(cov_path) as cov, np.load(probe_path) as probe:
                for signal in ('fit', 'mean', 'covariance'):
                    for layer in range(1, 7):
                        mu = probe[f'L{layer}_mu'].astype(float)
                        sd = probe[f'L{layer}_sd'].astype(float)
                        valid = probe[f'L{layer}_valid'].astype(bool)
                        ds = cov[f'{signal}_{metric}'][layer-1].astype(float)
                        mu_term = ds[0]/(sd+1e-12)
                        sd_term = -mu*ds[1]/(sd+1e-12)**2
                        scale = max(np.abs(ds[2][valid]).max(), 1e-12)
                        assert np.abs((mu_term+sd_term-ds[2])[valid]).max()/scale < 1e-5
                        positions.append(dict(seed=seed, task=task, stage=stage, metric=metric,
                                              signal=signal, layer=layer,
                                              mu_term=float(mu_term[valid].mean()),
                                              sd_term=float(sd_term[valid].mean()),
                                              total=float(ds[2][valid].mean())))

df = pd.DataFrame(rows)
df.to_csv(OUT/'switch_fit_unit_overlap.csv', index=False)
pd.DataFrame(positions).to_csv(OUT/'position_channels.csv', index=False)
(OUT/'unit_overlap_sources.json').write_text(json.dumps([
    dict(path=str(p), bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    for p in sorted(sources)], indent=2)+'\n')

lines = ['# 同じunitを追うと、切替整列と通常学習は二分されているか', '',
         'Issaの追加質問を受けた事後の記述的検査。新しい介入の成功基準には使わない。共有head、課題60末、5 seed。通常学習fitの旧画像上の勾配と、切替期待勾配の整列成分loadを、同じ固定probeの同じFFN unitへ運んで対応付けた。保存Adamの分母固定、momentum除外。正のSがμ/σを下げる方向。', '',
         '**最終FFNには反対向きの偏りがあるが、unitが明確に二分されているわけではない。** 前回の各層の「沈むunitの割合」は二つの別の周辺割合であり、同じunitが反転した割合ではなかった。', '',
         '| FFN | 同じ画像で反対向き | 次課題画像で反対向き | 次課題：seed範囲 | 次課題：独立な組合せとの差 | 次課題：押しベクトルcosine |',
         '|---:|---:|---:|---:|---:|---:|']
for layer in range(1, 7):
    same = df.query('task==60 and layer==@layer and inputs=="same_inputs"')
    nxt = df.query('task==60 and layer==@layer and inputs=="next_inputs"')
    lines.append(f'| {layer} | {100*same.opposite_fraction.median():.1f}% | '
                 f'{100*nxt.opposite_fraction.median():.1f}% | '
                 f'{100*nxt.opposite_fraction.min():.1f}–{100*nxt.opposite_fraction.max():.1f}% | '
                 f'{100*nxt.excess_opposite_fraction.median():+.1f}ポイント | {nxt.cosine.median():+.3f} |')
lines += ['',
          '範囲以外はseed中央値。「独立な組合せ」はそのseedでのfit/loadの沈降割合をa,bとしてa(1−b)+(1−a)b。実測との差もseed内で取ってから中央値にした。別々の中央値の差とは一致しない。これは周辺割合だけでも反対向きが増える効果を区別する記述量で、有意性検定ではない。共通の有効unitのうち、双方の絶対圧がmax(10⁻¹², 10⁻⁶×最大絶対圧)を超えるものを使用。4組の全割合とtask10もCSVに保存した。', '',
          '最終FFNの次課題入力でもseed範囲は46.5〜70.3%で、2/5 seedでは反対向きが半数未満だった。層平均の傾向を、全seedに共通するunitの住み分けと解釈しない。', '',
          '## 逆向きになりやすい理由と、その説明の限界', '',
          '同じ画像を考え、予測確率をp、旧正解one-hotをq、一様分布をuとすると、通常学習のlogit誤差はp−q、独立な新ラベルへの切替の期待logit誤差はp−u。通常学習は残った正解への不足を埋め、切替期待は旧課題で作った確信を弱める。', '',
          '10分類で正解確率0.99、他の9クラスが均等なら、通常誤差の正解成分は−0.01、切替期待は+0.89となり、全logit誤差ベクトルが逆向きになる。より一般にも、旧正解が最大確率なら (p−q)·(p−u)=||p||²−p_y≤0。これはlogit空間での関係である。', '',
          'ただし、読み出しVと画像依存のJacobianは角度を保存しない。画像ごとに通常誤差の大きさが違い、十分覚えた画像は通常学習への重みが小さくなる一方、切替期待では強く寄与できる。各unitにはReLUの開閉、attention、残差経路、上流パラメータ更新を通じて異なる重みで和が届く。そのため、同じ画像でlogit誤差が反対でも、平均したunitのμ/σへの押しまで一律に反転するわけではない。', '',
          'さらに今回比べたloadは、全読み出しVではなく旧ラベルとの整列成分Lだけを通す。次課題入力の場合は画像も変わる。従って上の誤差の式は機構の候補を説明するものであり、この表の層差を導いた証明ではない。', '',
          '## 入力ばらつきとの関係', '',
          '画像ごとに経路と誤差が違うため、平均勾配はE[Jᵀe]=E[Jᵀ]E[e]+Cov(J,e)と分けられる。この画像ごとの対応がどの程度効くかは[共分散介入とfull-batch対照](covariance_summary.md)で測った。単なる画素分散の大きさだけでは押しの符号は決まらない。', '',
          'また、入力データ間のばらつきと、測定指標μ/σの分母σは別の概念である。位置の微分をS_r=S_μ/(σ+ε)−μS_σ/(σ+ε)²に分けた数値をposition_channels.csvへ保存した。μの移動とσの変化は同じ方向に効くとは限らない。', '',
          '整列が特別に沈降を増やすかの介入結果は[機構検証本体](summary.md)。今回の単なる符号対応と混同しない。', '',
          '![同じunitでの反転と周辺割合](switch_fit_unit_overlap.png)']
(OUT/'unit_overlap.md').write_text('\n'.join(lines)+'\n')

plt.rcParams.update({'axes.spines.top':False, 'axes.spines.right':False, 'font.size':10})
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
for inputs, color, caption in [('same_inputs', '#286ca5', 'Same images'),
                               ('next_inputs', '#c65436', 'Next-task images')]:
    grouped = df.query('task==60 and inputs==@inputs').groupby('layer')
    sub = grouped.median(numeric_only=True)
    axes[0].plot(sub.index, 100*sub.opposite_fraction, 'o-', color=color, label=caption)
    axes[0].plot(sub.index, 100*sub.independent_opposite_fraction, '--', color=color, alpha=.6)
    axes[0].fill_between(sub.index, 100*grouped.opposite_fraction.min(),
                         100*grouped.opposite_fraction.max(), color=color, alpha=.08)
    axes[1].plot(sub.index, sub.cosine, 'o-', color=color, label=caption)
axes[0].set(ylabel='Units with opposite pressure (%)', ylim=(0, 100),
            title='Solid: paired units; dashed: independent marginals')
axes[1].set(ylabel='Cosine between unit-pressure vectors', ylim=(-1, 1),
            title='-1 = opposite vectors; +1 = same direction')
axes[1].axhline(0, color='gray', ls=':', lw=1)
axes[0].legend(frameon=False, fontsize=8)
for ax in axes:
    ax.set(xticks=range(1, 7), xlabel='FFN block (1 = nearest input)')
    ax.grid(alpha=.18)
fig.suptitle('Task 60 end: ordinary fit vs. switch alignment, matched units\nMedians and shaded ranges of 5 seeds; saved Adam denominator, no momentum')
for ext in ('png', 'pdf'):
    fig.savefig(OUT/f'switch_fit_unit_overlap.{ext}', dpi=180)
plt.close(fig)
print((OUT/'unit_overlap.md').read_text())
