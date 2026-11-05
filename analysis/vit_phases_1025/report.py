"""Report layerwise phase diagnostics without equating time windows with mechanisms."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/vit_phases_1025'
ARCHIVE = Path('/home/issan/Projects/obsidian-research-data/vit_phases_1025')
s = pd.read_csv(OUT / 'states.csv')
p = pd.read_csv(OUT / 'pressures.csv')
c = pd.read_csv(OUT / 'comparisons.csv')
run = json.loads((OUT / 'run.json').read_text())
validation = json.loads((OUT / 'validation.json').read_text())
assert run['status'] == 'complete' and len(validation) == 60 and len(s) == 360
ap = pd.read_csv(OUT / 'adam_pressures.csv')
ac = pd.read_csv(OUT / 'adam_comparisons.csv')
av = json.loads((OUT / 'adam_validation.json').read_text())
assert len(av) == 20
anp = pd.read_csv(OUT / 'adam_next_pressures.csv')
anv = json.loads((OUT / 'adam_next_validation.json').read_text())
assert len(anv) == 10
for prefix, frame in [('adam', ap), ('adam_next', anp)]:
    assert json.loads((OUT / f'{prefix}_run.json').read_text())['status'] == 'complete'
    aagg = frame.groupby(['arm', 'task', 'stage', 'layer', 'signal', 'route', 'moment', 'population'])[
        ['mean', 'median', 'sink_fraction', 'float_fraction', 'zero_fraction']].agg(['median', 'min', 'max'])
    aagg.columns = ['_'.join(col) for col in aagg.columns]
    aagg.reset_index().to_csv(OUT / f'{prefix}_pressure_aggregate.csv', index=False)
assert not s.duplicated(['arm', 'seed', 'task', 'stage', 'layer']).any()
main = p[(p.route == 'full') & (p.moment == 'r') & (p.population == 'all')]
keys = ['arm', 'task', 'stage', 'step', 'layer', 'signal', 'route', 'moment', 'population']
agg = p.groupby(keys)[['mean', 'median', 'sink_fraction', 'float_fraction', 'zero_fraction']].agg(['median', 'min', 'max'])
agg.columns = ['_'.join(col) for col in agg.columns]
agg.reset_index().to_csv(OUT / 'pressure_aggregate.csv', index=False)
ckeys = ['arm', 'task', 'stage', 'step', 'layer', 'prediction', 'actual', 'population']
ca = c.groupby(ckeys)[['n', 'balanced_accuracy', 'sign_agreement', 'spearman']].agg(['median', 'min', 'max'])
ca.columns = ['_'.join(col) for col in ca.columns]
ca.reset_index().to_csv(OUT / 'comparison_aggregate.csv', index=False)

drifts = []
for arm in ('SL', 'R0L'):
    for seed in range(5):
        for task in (10, 60):
            raw = {stage: np.load(ARCHIVE / 'raw' / f'{arm}_seed{seed}_t{task}_{stage}.npz') for stage in ('start', 'mid', 'end')}
            for layer in range(1, 7):
                for left, right, label in (('start', 'mid', '0-100'), ('mid', 'end', '100-500'), ('start', 'end', '0-500')):
                    valid = raw[left][f'L{layer}_valid'] & raw[right][f'L{layer}_valid']
                    for moment in ('mu', 'sd', 'r'):
                        delta = (raw[right][f'L{layer}_{moment}'] - raw[left][f'L{layer}_{moment}'])[valid]
                        drifts.append(dict(arm=arm, seed=seed, task=task, layer=layer, interval=label, moment=moment,
                                           n=len(delta), mean=float(delta.mean()), median=float(np.median(delta)),
                                           decreased_fraction=float((delta < 0).mean())))
d = pd.DataFrame(drifts)
d.to_csv(OUT / 'observed_drift.csv', index=False)

def pick(frame, **conditions):
    keep = np.ones(len(frame), dtype=bool)
    for key, value in conditions.items():
        keep &= (frame[key] == value).to_numpy()
    return frame.loc[keep]

def med(frame, column, **conditions):
    return pick(frame, **conditions)[column].median()

lines = ['# ViT 各ブロックの切替・通常学習・相2模型（1025、探索的事後解析）', '',
         '**「入力側は通常学習中にも沈み、最終FFNは切替を含む初期に沈んで後半に戻る」という時間的な層差は支持された。切替の整列成分も最終FFNを沈める側だった。ただし、全ての後段が一律に相1優勢とは言えず、前段の沈降を既存の相2理論で同定できたわけでもない。** 共有headのtask10/60で、100〜500更新のμ/σ変位の中央値はFFN1〜3が負、FFN4〜6が正。task60では各層4/5 seedがこの向きだった。最終FFNは両課題とも0〜100更新で沈み、100〜500更新で戻った（中央値）。', '',
         '6 ブロック ReLU/LN、共有 head（SL）と reset（R0L）、全 5 seed。task 10 と60の開始前・100更新後・500更新後、計60保存状態を測定した。各対象課題の画像250枚で勾配を計算し、固定64画像probeのCLSにおける各unitの μ/σ の微分へ運んだ。最初の検査はSGD方向の一次診断であり、実際のAdam更新ではない。', '',
         '追加で、Adam状態が保存されたtask10/60末の20状態を、保存二次momentの分母で重み付けして照合した。共有headではさらに10状態で次課題画像への切替を測定した。分母は固定し、運動量は含めない。', '',
         '今回分解した読み出しvは最終分類headであり、その整列成分の信号を各FFNまで逆伝播した。クラス中心化headと旧ラベル×特徴の整列は強く、task10末のcos²はseed中央値0.907、task60末は0.777。head共有で整列そのものが消えているわけではない。各FFN自身のfc2の整列を独立に分解したものではない。', '',
         '## 実軌跡と保存Adam分母での診断', '',
         '共有head・task60末・全有効unit・5 seed中央値。切替の整列成分を、同じ画像の独立再ラベルと、元の手順で生成する次課題画像の両方で測った。後者のtask61は未学習の仮想的な次課題であり、学習を延長した結果ではない。整列を求める旧画像・旧ラベルはどちらもtask60。', '',
         '| FFN | 整列成分：同じ画像 | 整列成分：次課題画像 | 通常学習が沈めるunit | 実際のΔ(μ/σ)、100–500更新 |',
         '|---:|---:|---:|---:|---:|']
adam_table = []
for layer in range(1, 7):
    load = med(ap, 'sink_fraction', arm='SL', task=60, layer=layer, signal='load', route='full', moment='r', population='all')
    load_next = med(anp, 'sink_fraction', arm='SL', task=60, layer=layer, signal='load', route='full', moment='r', population='all')
    fit = med(ap, 'sink_fraction', arm='SL', task=60, layer=layer, signal='fit', route='full', moment='r', population='all')
    drift = med(d, 'median', arm='SL', task=60, layer=layer, interval='100-500', moment='r')
    lines.append(f'| {layer} | {100 * load:.1f}% | {100 * load_next:.1f}% | {100 * fit:.1f}% | {drift:+.4f} |')
    adam_table.append(dict(layer=layer, aligned_switch_same_input_sink_fraction=load,
                           aligned_switch_next_input_sink_fraction=load_next,
                           current_fit_sink_fraction=fit, observed_late_delta_r=drift))
lines += ['',
         '通常学習は入力側の多くを沈め、FFN5〜6の多くを浮かせる。切替の整列成分は、次課題画像でも最終FFNを64.8%のunitで沈め、5/5 seedで過半数だった（seed範囲50.7〜75.9%）。ただしFFN4/5は51.7/53.3%で中立に近く、画像を同じにした場合の51.9/61.3%とも違う。深さに沿った単調な交代ではない。', '',
         'task10末から11への切替期待では、整列成分が沈める割合はFFN4〜6で67.3%、66.4%、62.2%。前半の課題列では後段3層に広がり、task60末では最終層が最も明瞭だった。割合は押しの符号の頻度であり、相1と相2の期間積分された寄与率ではない。', '',
         '![保存Adam分母と実際の区間変位](adam_and_observed_drift.png)', '',
         '## 「通常学習中の沈降」と相2模型の同定', '',
         'Issaの「相2」を通常学習中の沈降という観測上の意味で使えば、前段優勢という見立てに対応する。一方、既存ノートの相2は、読み出しの共同最小二乗fitから切替の押しを予測する理論でもあり、単純な時間区分とは異なる。以下はその既存式を通常学習にも照合した結果。', '',
         '保存Adam分母で比べたtask60末のP2と通常学習のbalanced accuracyのseed中央値は、FFN1〜6で50.8%、51.3%、49.4%、51.6%、47.2%、53.0%。順位相関も約−.05〜.12で、今回の条件付きP2は通常学習のunitごとの押しを説明していない。したがって、観測された前段の沈降を、そのまま既存の相2模型の成立とは結論しない。', '',
         '## 読み方', '',
         '- 正の押し S は勾配降下で μ/σ を下げる方向。表の割合は有効unitのうち沈める方向の割合で、各seedの値を作ってから中央値を取る。平均Sの符号も別表に残す。',
         '- load は前課題のラベル・特徴と整列したhead成分が、切替で作る信号。load と remainder の和が切替の期待勾配になる。remainder を相2とは扱わない。',
         '- 通常学習の fit は実CEの勾配。これを相2だと決めず、独立なP2予測と照合する。P1/P2は係数を実測へfitせず、後段を通して同じ位置微分へ運んだ。', '',
         '## 共有head: SGD方向で沈めるunitの割合', '',
         'この節の切替はtask9→10・59→60の開始前で、上のAdam表のtask10→11・60→61とは時点も重み付けも異なる。数値の差をAdam重み付けだけの効果とは解釈しない。', '']
tables = []
for task in (10, 60):
    lines += ['', f'### task {task}', '', '| FFN | 切替・整列成分 | 切替・全体 | 通常学習100 | 通常学習500 |', '|---:|---:|---:|---:|---:|']
    for layer in range(1, 7):
        values = [med(main, 'sink_fraction', arm='SL', task=task, layer=layer, stage=stage, signal=signal)
                  for stage, signal in (('start', 'load'), ('start', 'switch'), ('mid', 'fit'), ('end', 'fit'))]
        lines.append('| ' + str(layer) + ' | ' + ' | '.join(f'{100 * x:.1f}%' for x in values) + ' |')
        tables.append(dict(task=task, layer=layer, switch_loading_sink_fraction=values[0], switch_total_sink_fraction=values[1],
                           fit100_sink_fraction=values[2], fit500_sink_fraction=values[3]))
lines += ['', '## P2は通常学習の押しを説明するか', '',
          '下表は共有head・100更新後・全有効unitのSpearman相関（seed中央値）。相関1ならunit間の押しの順位が一致、負なら逆方向の関係。符号のbalanced accuracyと共通集合の数はcomparison_aggregate.csv。', '',
          '| FFN | task10 P1 | task10 P2 | task60 P1 | task60 P2 |', '|---:|---:|---:|---:|---:|']
for layer in range(1, 7):
    values = [med(c, 'spearman', arm='SL', task=task, stage='mid', layer=layer, prediction=pred, actual='fit', population='all')
              for task, pred in ((10, 'P1'), (10, 'P2'), (60, 'P1'), (60, 'P2'))]
    lines.append('| ' + str(layer) + ' | ' + ' | '.join(f'{x:+.3f}' for x in values) + ' |')
lines += ['', '![層ごとの押しと予測の対応](layer_phase_comparison.png)', '',
          '## 実際のμ/σの移動', '',
          '同じunit・同じ固定probeの差を取り、unit中央値、その後seed中央値。負なら実際に沈んだ。0–100更新には、切替直後と通常学習の両方が含まれる。', '',
          '| FFN | task10 0–100 | task10 100–500 | task60 0–100 | task60 100–500 |', '|---:|---:|---:|---:|---:|']
for layer in range(1, 7):
    values = [med(d, 'median', arm='SL', task=task, layer=layer, interval=window, moment='r')
              for task, window in ((10, '0-100'), (10, '100-500'), (60, '0-100'), (60, '100-500'))]
    lines.append('| ' + str(layer) + ' | ' + ' | '.join(f'{x:+.4f}' for x in values) + ' |')
lines += ['',
          'task60の0–100更新は、FFN1と6が沈む一方、FFN2〜5は浮く（中央値）。したがって「出力に近い全ブロックが、各切替の最初の100更新で沈む」という形ではない。', '',
          'head reset群の最終FFNの0–100更新は、task10で+0.0387、task60で+0.0040。共有群の−0.1047、−0.1347と向きが異なる。ただし別々に学習したbody同士なので、共有群の同一状態で整列だけを消す因果介入ではない。', '',
          '## 次課題画像の切替期待：seed間の幅', '',
          '保存Adam分母、整列成分、全経路のμ/σ。次課題は11/61。中央値だけで一貫性を過大評価しないため、seed範囲と、沈めるunitが過半数だったseed数を示す。', '',
          '| 旧課題末 | FFN | 中央値 | seed範囲 | 過半数となるseed |', '|---:|---:|---:|---:|---:|']
for task in (10, 60):
    for layer in range(1, 7):
        values = pick(anp, arm='SL', task=task, layer=layer, signal='load', route='full', moment='r', population='all').sink_fraction
        lines.append(f'| {task} | {layer} | {100 * values.median():.1f}% | {100 * values.min():.1f}–{100 * values.max():.1f}% | {int((values > .5).sum())}/5 |')
lines += ['', '## 上流経路の寄与', '',
          '先の結論に使った保存Adam分母・共有head・task60末の通常学習では、下表の方向だった。S>0が沈降。自unitのfc1更新だけでなく、上流が動いてそのunitへの入力が変わる経路も含める必要がある。例えばFFN3はlocalがわずかに浮かせる側でも、上流込みでは沈める側となる。', '',
          '| FFN | fit500 local | fit500 upstream | fit500 full |', '|---:|---:|---:|---:|']
for layer in range(1, 7):
    values = [med(ap, 'mean', arm='SL', task=60, layer=layer, signal='fit', route=route, moment='r', population='all')
              for route in ('local', 'upstream', 'full')]
    lines.append('| ' + str(layer) + ' | ' + ' | '.join(f'{x:+.4f}' for x in values) + ' |')
lines += ['',
          '共有head・task60、SGD方向。各セルは全unit平均Sのseed中央値。localは自unitのfc1の重み・biasだけ、upstreamは残り、fullは合計。各seedでは加算が成り立つが、個別の中央値同士は必ずしも足し合わない。値が正なら沈める向き。', '',
          '| FFN | 切替load local | 切替load upstream | 切替load full | fit100 local | fit100 upstream | fit100 full |',
          '|---:|---:|---:|---:|---:|---:|---:|']
for layer in range(1, 7):
    values = [med(p, 'mean', arm='SL', task=60, layer=layer, stage=stage, signal=signal, route=route, moment='r', population='all')
              for stage, signal in (('start', 'load'), ('mid', 'fit')) for route in ('local', 'upstream', 'full')]
    lines.append('| ' + str(layer) + ' | ' + ' | '.join(f'{x:+.3g}' for x in values) + ' |')
lines += ['', '## 限界と検算', '',
          '- P2は特徴固定の共同最小二乗head模型。実際のheadはCE/Adamで学習しており、同一ではない。250画像に384特徴なので列独立な模型の仮定も満たさず、ここでは最小ノルム・擬似逆の拡張を検査した。',
          f'- rank閾値1e-8でのrankは {int(s["rank"].min())}〜{int(s["rank"].max())}。閾値1e-6/1e-10でrankが変わった状態は追加予測を保存した。',
          '- 相1と相2を加算して100%にした寄与率は、この定義からは出せない。通常学習中に沈むという観測だけで相2の機構を確定しない。',
          '- 整列成分は最終分類headのものを各FFNへ戻している。中間FFN自身の出力行列の整列を、層ごとに独立に分解した検査ではない。',
          '- R0Lの開始時はheadを0へ戻すため、実fit/switch/loadのbody勾配は0。予測器P1/P2だけが非ゼロでも、実headでその模型が成立したことを意味しない。',
          f'- 全60状態のbaseline精度が元記録と一致。平均CEの最大絶対差 {max(r["baseline_loss_abs_error"] for r in validation):.3g}。checkpoint/source SHA照合、load+remainder=switch、local解析微分とJVPの検算を通過。',
          '- 元の登録J5の早期/後期窓の正式判定ではない。今回は多層と通常学習を追加した別の探索的検査。', '',
          '生unit配列は `/home/issan/Projects/obsidian-research-data/vit_phases_1025/` 以下の `raw/`、`adam_raw/`、`adam_next_raw/`。元のcheckpointはcheckpoint_provenance.jsonおよびadam*_provenance.json、保存物はbackup_manifest.json。']
(OUT / 'summary.md').write_text('\n'.join(lines) + '\n')
(OUT / 'verdict.json').write_text(json.dumps(dict(tier='exploratory_posthoc', checkpoints=60, adam_checkpoints=20, adam_next_input_checks=10,
    temporal_layer_pattern='supported_descriptively', phase2_mechanistic_identification='not_established',
    adam_task60=adam_table, layer_sink_fractions=tables), indent=2) + '\n')

plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 10})
fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, constrained_layout=True)
for col, task in enumerate((10, 60)):
    for stage, sig, label, color in [('start', 'load', 'Switch: aligned readout', '#c65536'), ('mid', 'fit', 'Current task: update 100', '#286ca5'), ('end', 'fit', 'Current task: update 500', '#68a3c8')]:
        sub = pick(main, arm='SL', task=task, stage=stage, signal=sig)
        g = 100 * sub.groupby('layer').sink_fraction.median()
        axes[0, col].plot(g.index, g.values, 'o-', color=color, label=label)
    axes[0, col].axhline(50, color='gray', lw=1, ls=':')
    axes[0, col].set(title=f'Task {task}: units pushed downward', ylabel='Units with positive S_r (%)', ylim=(0, 100))
    for pred, color in [('P1', '#c65536'), ('P2', '#7661a8')]:
        sub = pick(c, arm='SL', task=task, stage='mid', prediction=pred, actual='fit', population='all')
        g = sub.groupby('layer').spearman.agg(['median', 'min', 'max'])
        axes[1, col].plot(g.index, g['median'], 'o-', color=color, label=pred + ' vs current-task gradient')
        axes[1, col].fill_between(g.index, g['min'], g['max'], color=color, alpha=.13)
    axes[1, col].axhline(0, color='gray', lw=1, ls=':')
    axes[1, col].set(ylabel='Spearman correlation', ylim=(-1, 1), xlabel='FFN block (1 = nearest input)', xticks=range(1, 7))
    for ax in axes[:, col]:
        ax.grid(alpha=.18)
axes[0, 0].legend(frameon=False, fontsize=8)
axes[1, 0].legend(frameon=False, fontsize=8)
fig.suptitle('Shared-head ViT: full-parameter SGD pressure on CLS mean / SD\nMedians of 5 seeds; P1/P2 are conditional predictors, not additive components')
for ext in ('png', 'pdf'):
    fig.savefig(OUT / f'layer_phase_comparison.{ext}', dpi=180)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
for signal, color, label in [('load', '#c65536', 'Switch alignment: same images'), ('fit', '#286ca5', 'Current-task learning')]:
    sub = pick(ap, arm='SL', task=60, signal=signal, route='full', moment='r', population='all')
    g = 100 * sub.groupby('layer').sink_fraction.median()
    axes[0].plot(g.index, g.values, 'o-', color=color, label=label)
sub = pick(anp, arm='SL', task=60, signal='load', route='full', moment='r', population='all')
g = 100 * sub.groupby('layer').sink_fraction.median()
axes[0].plot(g.index, g.values, 's--', color='#8d3020', label='Switch alignment: next-task images')
axes[0].axhline(50, color='gray', ls=':', lw=1)
axes[0].set(title='Saved Adam denominator, task 60 end', ylabel='Units pushed downward (%)', ylim=(0, 100))
axes[0].legend(frameon=False, fontsize=8)
for task, color in [(10, '#888888'), (60, '#159b8d')]:
    sub = pick(d, arm='SL', task=task, interval='100-500', moment='r')
    g = sub.groupby('layer')['median'].median()
    axes[1].plot(g.index, g.values, 'o-', color=color, label=f'Task {task}')
axes[1].axhline(0, color='gray', ls=':', lw=1)
axes[1].set(title='Observed change, updates 100 to 500', ylabel='Change in mean / SD (negative = sinking)')
axes[1].legend(frameon=False)
for ax in axes:
    ax.set(xlabel='FFN block (1 = nearest input)', xticks=range(1, 7))
    ax.grid(alpha=.18)
fig.suptitle('Shared-head ViT, CLS; medians of 5 seeds')
for ext in ('png', 'pdf'):
    fig.savefig(OUT / f'adam_and_observed_drift.{ext}', dpi=180)
plt.close(fig)

def hashfile(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 ** 2), b''):
            h.update(chunk)
    return h.hexdigest()
files = [dict(source=str(path), backup=str(path), bytes=path.stat().st_size, sha256=hashfile(path))
         for path in sorted(ARCHIVE.rglob('*')) if path.is_file()]
(OUT / 'backup_manifest.json').write_text(json.dumps(dict(archive=str(ARCHIVE), files=files,
    storage='Raw arrays and logs were written directly into the permanent archive; source and backup therefore coincide.',
    source_archive='/home/issan/Projects/obsidian-research-data/vit_ffn_ablation_1025/source.tar'), indent=2) + '\n')
print((OUT / 'summary.md').read_text())
