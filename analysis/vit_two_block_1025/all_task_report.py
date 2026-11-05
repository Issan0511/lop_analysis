"""Post-hoc temporal accounting across every completed baseline task.

Windows classify time, not causal sources. Fixed validation probes permit
telescoping across tasks; changing training probes do not.
"""
import json
from pathlib import Path
import common as T
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
np = T.np
WINDOWS = ((0, 20), (20, 100), (100, 500))
OUTPUT = T.ROOT / 'results/vit_two_block_1025'


def main():
    rows, paths, cumulative, inputs, boundaries = [], [], [], [], []
    for seed in T.SEEDS:
        root = T.ARCHIVE / f'runs/seed{seed}'
        state = json.loads((root / 'stage_complete.json').read_text())
        assert state['status'] == 'complete' and state['through_task'] == 60
        provenance = json.loads((root / 'provenance.json').read_text())
        assert provenance['source_hashes'] == T.own_hashes()
        manifest = {r['path']: r for r in json.loads((root / 'manifest.json').read_text())}
        validation = {layer: [] for layer in (1, 2)}
        previous = {}
        for task in range(1, 61):
            path = root / f'moments/t{task:02d}.npz'
            digest = T.sha256(path)
            assert digest == manifest[str(path.relative_to(root))]['sha256']
            inputs.append(dict(path=str(path), sha256=digest))
            with np.load(path) as data:
                for pop in ('train', 'validation'):
                    for layer in (1, 2):
                        snapshots = {step: {field: data[f's{step}_{pop}_L{layer}_{field}']
                                     for field in ('mu', 'sd', 'r', 'valid')}
                                     for step in (0, 20, 100, 500)}
                        if pop == 'validation':
                            if task > 1:
                                assert all(np.array_equal(previous[layer][k], snapshots[0][k])
                                           for k in snapshots[0]), (seed, task, layer)
                                boundaries.append(dict(seed=seed, task=task, layer=layer, bit_equal=True))
                            previous[layer] = snapshots[500]
                            validation[layer].append(snapshots)
                        for start, end in WINDOWS:
                            a, b = snapshots[start], snapshots[end]
                            valid = a['valid'] & b['valid']
                            assert valid.any()
                            q0, q1 = 1 / (a['sd'] + 1e-12), 1 / (b['sd'] + 1e-12)
                            mean = (b['mu'] - a['mu']) * (q0 + q1) / 2
                            width = (b['mu'] + a['mu']) * (q1 - q0) / 2
                            delta = b['r'] - a['r']
                            error = float(np.abs(mean + width - delta)[valid].max())
                            assert error < 1e-11
                            rows.append(dict(seed=seed, task=task, population=pop, layer=layer,
                                start=start, end=end, common_valid_units=int(valid.sum()),
                                delta_r=float(delta[valid].mean()),
                                r_mean_term=float(mean[valid].mean()), r_width_term=float(width[valid].mean()),
                                identity_error=error))
        for layer, tasks in validation.items():
            # One common mask across the entire trajectory prevents changes in
            # the included units from masquerading as a net temporal drift.
            valid = np.logical_and.reduce([x['valid'] for task in tasks for x in task.values()])
            assert valid.any()
            increments = np.array([[task[b]['r'] - task[a]['r'] for a, b in WINDOWS] for task in tasks])[:, :, valid]
            for first in (1, 2):
                per_window = increments[first - 1:].sum(0).mean(1)
                net_per_unit = (tasks[-1][500]['r'] - tasks[first - 1][0]['r'])[valid]
                error = float(np.abs(increments[first - 1:].sum((0, 1)) - net_per_unit).max())
                assert error < 1e-10
                paths.append(dict(seed=seed, layer=layer, first_task=first, last_task=60,
                    common_valid_units=int(valid.sum()), delta_r=float(net_per_unit.mean()),
                    window_0_20=float(per_window[0]), window_20_100=float(per_window[1]),
                    window_100_500=float(per_window[2]), telescoping_max_error=error))
            sums = increments[1:].mean(2).cumsum(0)
            for index, values in enumerate(sums, 2):
                cumulative.append(dict(seed=seed, layer=layer, through_task=index,
                    window_0_20=values[0], window_20_100=values[1], window_100_500=values[2],
                    total=values.sum()))
    frame, path_frame, curves = pd.DataFrame(rows), pd.DataFrame(paths), pd.DataFrame(cumulative)
    frame.to_csv(OUTPUT / 'all_task_windows.csv', index=False)
    path_frame.to_csv(OUTPUT / 'all_task_validation_accounting.csv', index=False)
    curves.to_csv(OUTPUT / 'all_task_validation_cumulative.csv', index=False)
    T.atomic_json(OUTPUT / 'all_task_inputs.json', dict(source_sha256=T.sha256(Path(__file__)), inputs=inputs,
        validation_boundaries=len(boundaries), all_validation_boundaries_bit_equal=all(x['bit_equal'] for x in boundaries),
        endpoint_max_error=float(frame.identity_error.max()), telescoping_max_error=float(path_frame.telescoping_max_error.max()),
        posthoc_reason='Task-60 complete results showed that relearning and whole-task signs differ from the late-learning window.'))
    text = ['# 全60課題の時間区間別の位置変化（事後解析）', '',
        '課題60の全5seedが揃った後に、同じ時間区間を全課題へ広げた。追加の学習は行っていない。',
        '初回学習の課題1を切替として数えず、下表は課題2〜60の59課題。各値は共通valid unitのΔ(μ/σ)の平均の符号。',
        'seedごとの本数を示し、295個のseed×課題やunit数を独立標本数として検定しない。', '',
        '| probe | block | 区間 | 下向きの課題数：seed 0 / 1 / 2 / 3 / 4（各59中） | 全5seedで下向きの課題数 | 全5seedで上向きの課題数 |',
        '|---|---:|---|---|---:|---:|']
    for keys, group in frame[frame.task >= 2].groupby(['population', 'layer', 'start', 'end']):
        pop, layer, start, end = keys
        pivot = group.pivot(index='task', columns='seed', values='delta_r')
        assert pivot.shape == (59, 5) and not pivot.isna().any().any()
        counts = ' / '.join(str(int((pivot[s] < 0).sum())) for s in T.SEEDS)
        text.append(f'| {pop} | {layer} | {start}→{end} | {counts} | {int((pivot < 0).all(axis=1).sum())} | {int((pivot > 0).all(axis=1).sum())} |')
    text += ['', '## 固定validation上で、長期変位を時間区間へ足し戻す', '',
        '全軌道共通のunit集合を使い、前課題500更新と次課題0更新のμ・σ・r・validがbit一致することを全590境界で検算した。',
        '下表は課題2〜60の合計についての5seed平均。初回の課題1は別途CSVへ残した。三つの窓の和は長期の実変位に一致する。',
        'これは時間区間の帳簿であり、整列・T5・上流などの原因別の割合ではない。学習画像は課題ごとに替わるため、この足し戻しには使わない。', '',
        '| block | 0→20の累積 | 20→100の累積 | 100→500の累積 | 課題1末→60末の実Δr |',
        '|---:|---:|---:|---:|---:|']
    for layer, g in path_frame[path_frame.first_task == 2].groupby('layer'):
        assert len(g) == 5
        means = g[['window_0_20', 'window_20_100', 'window_100_500', 'delta_r']].mean()
        text.append(f'| {layer} | ' + ' | '.join(f'{x:+.6f}' for x in means) + ' |')
    text += ['', '![固定probeの時間区間別累積](all_task_validation_cumulative.png)', '']
    (OUTPUT / 'all_task_summary.md').write_text('\n'.join(text))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    labels = {'window_0_20': 'Updates 0–20', 'window_20_100': 'Updates 20–100',
              'window_100_500': 'Updates 100–500', 'total': 'Net drift'}
    colors = ('#b65542', '#37846b', '#526cba', '#20242b')
    for layer, ax in enumerate(axes, 1):
        for (field, label), color in zip(labels.items(), colors):
            pivot = curves[curves.layer == layer].pivot(index='through_task', columns='seed', values=field)
            ax.plot(pivot.index, pivot, color=color, alpha=.15, linewidth=.7)
            ax.plot(pivot.index, pivot.mean(axis=1), color=color, linewidth=2, label=label)
        ax.axhline(0, color='#999999', linewidth=.7)
        ax.set(title=f'Block {layer}', xlabel='Task', ylabel=r'Cumulative $\Delta(\mu/\sigma)$')
        ax.grid(alpha=.18)
        ax.legend(frameon=False)
    fig.suptitle('Fixed validation probe | Tasks 2–60 | Five seeds (thin), seed mean (thick)')
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(OUTPUT / f'all_task_validation_cumulative.{ext}', dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(json.dumps(dict(status='complete', baseline_seeds=5, tasks=60, boundaries=len(boundaries), output=str(OUTPUT))))


if __name__ == '__main__':
    main()
