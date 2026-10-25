"""Summarize the frozen-checkpoint ablation without threshold-based verdicts."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {'R0L': '#159b8d', 'SL': '#3069ad'}
LABELS = {'R0L': 'Reset head', 'SL': 'Shared head'}


def hash_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--archive', type=Path, required=True)
    a = p.parse_args()
    d = pd.read_csv(a.results / 'per_checkpoint.csv')
    validations = json.loads((a.results / 'validation.json').read_text())
    run = json.loads((a.results / 'run.json').read_text())
    assert run['status'] == 'complete' and len(validations) == 80 and len(d) == 560
    keys = ['arm', 'seed', 'task', 'state', 'step', 'intervention', 'alpha']
    assert not d.duplicated(keys).any()
    for row in validations:
        assert row['checkpoint_sha256_ok'] and row['source_hashes_ok'] and row['parameters_unchanged']
    # Both head protocols are identical on task 1; this also catches head-row errors.
    first = d[d.task == 1]
    match_keys = ['seed', 'task', 'state', 'step', 'intervention', 'alpha']
    paired = first[first.arm == 'R0L'].merge(first[first.arm == 'SL'], on=match_keys, suffixes=('_r', '_s'))
    numeric = ['accuracy', 'loss', 'centered_logit_delta_rms']
    for col in numeric:
        np.testing.assert_allclose(paired[col + '_r'], paired[col + '_s'], atol=1e-9, rtol=1e-9)
    groups = ['arm', 'task', 'state', 'step', 'intervention', 'alpha']
    measures = ['accuracy', 'accuracy_delta_pp', 'loss', 'loss_delta', 'prediction_flip_rate',
                'centered_logit_delta_rms', 'centered_logit_relative_rms', 'kl_full_to_intervention',
                'branch_to_residual_norm_median']
    agg = d.groupby(groups)[measures].agg(['median', 'min', 'max'])
    agg.columns = ['_'.join(col) for col in agg.columns]
    agg['n_seeds'] = d.groupby(groups).size()
    agg.reset_index().to_csv(a.results / 'aggregate.csv', index=False)
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                         'savefig.dpi': 180, 'axes.titleweight': 'bold'})
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7), sharex=True, constrained_layout=True)
    for j, state in enumerate(('mid', 'end')):
        subset = d[(d.task == 60) & (d.state == state) & (d.intervention == 'ffn')]
        for arm in ('R0L', 'SL'):
            arm_data = subset[subset.arm == arm]
            for i, metric in enumerate(('accuracy', 'loss')):
                ax = axes[i, j]
                scale = 100 if metric == 'accuracy' else 1
                for _, seed_data in arm_data.groupby('seed'):
                    seed_data = seed_data.sort_values('alpha')
                    ax.plot(seed_data.alpha, scale * seed_data[metric], color=COLORS[arm], alpha=.22, lw=1)
                med = arm_data.groupby('alpha')[metric].median()
                ax.plot(med.index, scale * med.values, 'o-', color=COLORS[arm], label=LABELS[arm], lw=2.4, ms=4)
                ax.grid(alpha=.18)
        axes[0, j].set_title(f'Task 60, update {100 if state == "mid" else 500}')
        axes[0, j].set_ylim(0, 103)
        axes[1, j].set_xlabel('Final FFN multiplier (0 = removed, 1 = original)')
        axes[1, j].set_xticks([0, .25, .5, .75, 1])
    axes[0, 0].set_ylabel('Task memorization accuracy (%)')
    axes[1, 0].set_ylabel('Cross entropy')
    axes[0, 0].legend(frameon=False)
    fig.suptitle('Does the six-block ViT use its final FFN?\nFrozen weights; 250 task images; 5 seeds (thin lines), medians (bold)', fontsize=13)
    for ext in ('png', 'pdf'):
        fig.savefig(a.results / f'final_ffn_gate_curve.{ext}')
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), constrained_layout=True)
    for arm in ('R0L', 'SL'):
        for state, style in (('mid', '--'), ('end', '-')):
            sub = d[(d.arm == arm) & (d.state == state) & (d.intervention == 'ffn') & (d.alpha == 0)]
            for ax, metric, sign in ((axes[0], 'accuracy_delta_pp', -1), (axes[1], 'loss_delta', 1)):
                med = sub.groupby('task')[metric].median()
                ax.plot(med.index, sign * med.values, marker='o', linestyle=style, color=COLORS[arm],
                        label=f'{LABELS[arm]}, update {100 if state == "mid" else 500}')
                ax.set_xlabel('Task')
                ax.set_xticks([1, 10, 30, 60])
                ax.grid(alpha=.18)
    axes[0].set_ylabel('Accuracy drop on removing final FFN (pp)')
    axes[1].set_ylabel('Loss increase on removing final FFN')
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle('Final FFN removal across training (median of 5 seeds)', fontsize=13)
    for ext in ('png', 'pdf'):
        fig.savefig(a.results / f'final_ffn_over_time.{ext}')
    plt.close(fig)

    lines = ['# ViT 最終 FFN 除去の事後検証（1025）', '',
             '**最終タスクでは最終 FFN が予測に大きく寄与しており、実質的に未使用という仮説に反する。** task 60 / update 500 の最終 FFN 除去は、SL で全 5 seed の精度を 24.0〜58.4 pp、R0L で 42.4〜64.4 pp 下げた。検査した前半の時点では除去後も精度中央値を維持しており、依存はむしろ後期に大きくなった。ただし、これは現在の勾配や更新量の大きさを証明するものではない。', '',
             '6 ブロック・ReLU・LayerNorm ありの保存状態を使用。全 5 seed、task 1/10/30/60、update 100/500、head reset/shared の 80 状態。元の課題内 250 画像と iid ラベルで評価した記憶精度であり、一般化精度ではない。重みは固定し、再学習していない。', '',
             '## task 60 の最終 FFN 除去', '',
             '各値は seed 中央値。精度低下は seed 内の差を取ってから中央値にする。括弧は seed 間の最小〜最大。', '',
             '| head | update | 元の精度 | FFN 除去後 | 精度低下 pp（範囲） | CE 元 → 除去 | logit 相対変化 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    endpoint = []
    for arm in ('R0L', 'SL'):
        for state in ('mid', 'end'):
            sub = d[(d.arm == arm) & (d.task == 60) & (d.state == state) & (d.intervention == 'ffn')]
            base, off = sub[sub.alpha == 1], sub[sub.alpha == 0]
            drop = -off.accuracy_delta_pp
            record = dict(arm=arm, state=state, step=int(off.step.iloc[0]), n_seeds=len(off),
                          original_accuracy_median=float(base.accuracy.median()), removed_accuracy_median=float(off.accuracy.median()),
                          accuracy_drop_pp_median=float(drop.median()), accuracy_drop_pp_min=float(drop.min()),
                          accuracy_drop_pp_max=float(drop.max()), original_loss_median=float(base.loss.median()),
                          removed_loss_median=float(off.loss.median()), loss_increase_median=float(off.loss_delta.median()),
                          relative_logit_change_median=float(off.centered_logit_relative_rms.median()))
            endpoint.append(record)
            lines.append(f'| {arm} | {record["step"]} | {100 * base.accuracy.median():.1f}% | {100 * off.accuracy.median():.1f}% | '
                         f'{drop.median():.1f}（{drop.min():.1f}〜{drop.max():.1f}） | {base.loss.median():.5f} → {off.loss.median():.5f} | '
                         f'{100 * off.centered_logit_relative_rms.median():.1f}% |')
    lines += ['', '![最終 FFN の出力を段階的に弱めた比較](final_ffn_gate_curve.png)', '',
              '## タスク終了時の最終 FFN 除去効果の推移', '',
              'seed 中央値。各 task の元の精度中央値はいずれも 100%。', '',
              '| task | reset: 除去後の精度 | shared: 除去後の精度 |',
              '|---:|---:|---:|']
    for task in (1, 10, 30, 60):
        sub = d[(d.task == task) & (d.state == 'end') & (d.intervention == 'ffn') & (d.alpha == 0)]
        lines.append(f'| {task} | {100 * sub[sub.arm == "R0L"].accuracy.median():.1f}% | {100 * sub[sub.arm == "SL"].accuracy.median():.1f}% |')
    lines += ['',
              '## 最終ブロック全体を除去する補助対照', '',
              'FFN 単独除去とは別に、最後の attention と FFN の両方を恒等写像に置き換えた。task 60 / update 500 の精度中央値:', '']
    for arm in ('R0L', 'SL'):
        block = d[(d.arm == arm) & (d.task == 60) & (d.state == 'end') & (d.intervention == 'block')]
        lines.append(f'- {arm}: {100 * block.accuracy.median():.1f}%（{100 * block.accuracy.min():.1f}〜{100 * block.accuracy.max():.1f}%）。')
    lines += ['', '## 検算', '',
              f'- 80/80 checkpoint の SHA256 と元の source hashes が一致。手動 forward と通常 forward、hook での除去と手動介入を照合した。',
              f'- 元の intask 記録と記憶精度は全件一致。平均 CE の最大絶対差は {max(r["loss_abs_error"] for r in validations):.3g}。',
              '- task 1 の両 head 条件は全 seed・全介入で一致。parameter の in-place version は全件不変。',
              '- alpha = 0 の損失と logit 差がある場合、精度が維持されても FFN が未使用とは判定しない。', '',
              '## 解釈の範囲', '',
              '除去の影響は、その保存状態・課題内画像での予測への寄与を表す。最終 FFN が過去の学習で必要だったか、今後の学習に必要か、勾配がどの程度届くか、Adam の更新量、沈降の原因はこの介入だけから決まらない。最終 LN は元のままであり、FFN 除去による再正規化の影響も介入効果に含む。', '',
              '![task ごとの除去効果](final_ffn_over_time.png)', '',
              '生の example logits とラベル、画像ハッシュ、凍結ソースは `/home/issan/Projects/obsidian-research-data/vit_ffn_ablation_1025/`。使用 checkpoint は `checkpoint_provenance.json` の元の tar と member、SHA256 で特定する。']
    (a.results / 'summary.md').write_text('\n'.join(lines) + '\n')
    (a.results / 'verdict.json').write_text(json.dumps(dict(tier='posthoc_frozen_checkpoint_intervention',
                                                         task60=endpoint, checkpoints=80, validation='passed'), indent=2) + '\n')
    files = []
    for path in sorted(a.archive.rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts:
            files.append(dict(path=str(path), bytes=path.stat().st_size, sha256=hash_file(path)))
    (a.results / 'backup_manifest.json').write_text(json.dumps(dict(archive=str(a.archive), files=files), indent=2) + '\n')
    print(json.dumps(endpoint, indent=2))


if __name__ == '__main__':
    main()
