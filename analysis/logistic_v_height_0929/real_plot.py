"""Summarize the registered actual-system intervention matrix."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = Path('results/logistic_v_height_0929')
LABELS = {
    'A': 'A: original moments',
    'B': 'B: rescaled moments',
    'C': 'C: rescaled moments + epsilon',
    'D': 'D: reference second moment',
    'E': 'E: reference residual (null)',
}
COLORS = {'A': '#2465a8', 'B': '#cd6836', 'C': '#9b73b4', 'D': '#258570', 'E': '#777777'}
STYLES = {'A': '-', 'B': '--', 'C': ':', 'D': '-', 'E': '--'}


def read(name):
    with (OUT/name).open(newline='') as f:
        return list(csv.DictReader(f))


def main():
    data = {seed: read(f'real_s{seed}_contrasts.csv') for seed in (0, 1)}
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.7), sharex=True, layout='constrained')
    for seed in (0, 1):
        for policy in LABELS:
            rows = [r for r in data[seed] if r['policy'] == policy]
            x = [(int(r['task'])-51)*4000+int(r['step']) for r in rows]
            for i, metric in enumerate(['paired_median_T_small_minus_large',
                                        'paired_median_raw_top_small_minus_large']):
                axes[i, seed].plot(x, [float(r[metric]) for r in rows], color=COLORS[policy],
                                   linestyle=STYLES[policy], marker='o', markersize=3,
                                   linewidth=2.1 if policy in ('A', 'D') else 1.8,
                                   label=LABELS[policy])
        axes[0, seed].set_title(f'Seed {seed}: same 100 units, same inputs', fontweight='bold')
        axes[0, seed].set_ylabel('Median paired difference in centered tail T')
        axes[1, seed].set_ylabel('Median paired difference in raw maximum')
        axes[1, seed].set_xlabel('Updates after readout intervention')
        for ax in axes[:, seed]:
            ax.axhline(0, color='.55', linewidth=.8)
            ax.set_xscale('symlog', linthresh=10)
            ax.set_xticks([0, 10, 100, 1000, 4000, 8000, 12000],
                          ['0', '10', '100', '1000', '4000', '8000', '12000'])
            ax.tick_params(axis='x', labelrotation=35)
            ax.grid(alpha=.18)
            ax.spines[['top', 'right']].set_visible(False)
            for step in (4000, 8000):
                ax.axvline(step, color='.8', linewidth=.8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=3, frameon=False)
    fig.suptitle('Actual ELU interventions: c = 0.1 minus c = 10\n'
                 'Tasks 51-53 only; two independently reconstructed seeds', fontsize=15, fontweight='bold')
    fig.savefig(OUT/'real_interventions.png', dpi=180, bbox_inches='tight')
    fig.savefig(OUT/'real_interventions.pdf', bbox_inches='tight')
    plt.close(fig)

    endpoint = []
    for seed in (0, 1):
        for task in (51, 52, 53):
            rows = {r['policy']: r for r in data[seed] if int(r['task']) == task and int(r['step']) == 4000}
            row = dict(seed=seed, task=task)
            for mode in LABELS:
                row[f'{mode}_T_contrast'] = float(rows[mode]['paired_median_T_small_minus_large'])
                row[f'{mode}_raw_contrast'] = float(rows[mode]['paired_median_raw_top_small_minus_large'])
            row['B_over_A'] = row['B_T_contrast']/row['A_T_contrast'] if abs(row['A_T_contrast']) > .05 else None
            row['D_over_C'] = row['D_T_contrast']/row['C_T_contrast'] if abs(row['C_T_contrast']) > .05 else None
            endpoint.append(row)
    null = {seed: read(f'real_s{seed}_null_checks.csv') for seed in (0, 1)}
    numerical_null = {
        str(seed): {key: max(float(r[key]) for r in rows)
                    for key in ['max_abs_hidden_parameter_difference', 'max_abs_output_bias_difference',
                                'max_abs_T_difference', 'max_abs_raw_top_difference']}
        for seed, rows in null.items()}
    (OUT/'real_aggregate.json').write_text(json.dumps(dict(endpoint_contrasts=endpoint,
                                                         residual_replay_numerical_null=numerical_null), indent=2)+'\n')
    lines = [
        '# 実ELUの機構対照：集計値',
        '',
        'これは登録したtask51–53の短期介入であり、元のtask151–200の効果の同定ではない。',
        '各値は同じ100unitの T(c=.1)−T(c=10) の中央値。T=(max−lower median)/sample SD。',
        'A=元のmoment保持、B=hidden moment倍率整合、C=Bにepsilon倍率整合、D=Cの二次momentだけ基準からreplay、E=Cの残差を基準からreplay。',
        '',
        '| seed | task | A | B | C | D | E | B/A | D/C |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in endpoint:
        fmt = lambda value: '—' if value is None else f'{value:.6g}'
        values = [row['seed'], row['task']] + [row[f'{mode}_T_contrast'] for mode in LABELS] + [row['B_over_A'], row['D_over_C']]
        lines.append('| '+' | '.join(fmt(v) for v in values)+' |')
    lines += ['', '比は登録通り、分母の絶対値が.05より大きいときだけ表示。比を媒介因果の寄与率とは扱わない。', '',
              '![介入の時間経過](real_interventions.png)', '',
              '元データ・精度監査はreal_s*_provenance.json、real_s*_source_comparison.json、real_s*_null_checks.csv。',
              'Eの一致は理論で決まる検算条件。浮動小数点での一致精度と実系の機構の発見を区別する。', '',
              '## Residual replayの全記録時点での最大数値差', '']
    for seed, row in numerical_null.items():
        lines.append(f'- seed{seed}: hidden parameter {row["max_abs_hidden_parameter_difference"]:.6g}, '
                     f'T {row["max_abs_T_difference"]:.6g}, raw top {row["max_abs_raw_top_difference"]:.6g}, '
                     f'output bias {row["max_abs_output_bias_difference"]:.6g}.')
    (OUT/'real_aggregate.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(endpoint_contrasts=endpoint, numerical_null=numerical_null)))


if __name__ == '__main__':
    main()
