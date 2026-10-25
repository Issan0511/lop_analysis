"""Five-seed baseline curves and paired unit changes; no partial-seed pooling."""
import argparse
import csv
import json
from pathlib import Path
import common as T
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
np = T.np
POINTS = (0, 1, 5, 20, *range(25, 501, 25))
COLORS = {1: '#3567a4', 2: '#bf642e'}


def stats(data, step, pop, layer):
    prefix = f's{step}_{pop}_L{layer}_'
    mu, sd, r, valid, opened = (data[prefix + key] for key in ('mu', 'sd', 'r', 'valid', 'opened'))
    return dict(mu=float(mu.mean()), sd=float(sd.mean()), r=float(r[valid].mean()),
                closed=float((opened == 0).mean()), n_valid=int(valid.sum()))


def run(through):
    times, changes, fits, reopening = [], [], [], []
    targets = [10] if through == 10 else [10, 60]
    for seed in T.SEEDS:
        root = T.ARCHIVE / f'runs/seed{seed}'
        stage = json.loads((root / 'stage_complete.json').read_text())
        assert stage['status'] == 'complete' and stage['through_task'] >= through, (seed, stage)
        provenance = json.loads((root / 'provenance.json').read_text())
        assert provenance['source_hashes'] == T.own_hashes()
        fitrows = list(csv.DictReader((root / 'states.csv').open()))
        for task in range(1, through + 1):
            with np.load(root / f'moments/t{task:02d}.npz') as data:
                for pop in ('train', 'validation'):
                    for layer in (1, 2):
                        for step in (POINTS if task in targets else (500,)):
                            summary = stats(data, step, pop, layer)
                            times.append(dict(seed=seed, task=task, step=step, population=pop, layer=layer, **summary))
                            if task in targets and pop == 'train':
                                recorded = next(r for r in fitrows if int(r['task']) == task and int(r['step']) == step
                                                and r['population'] == pop and int(r['layer']) == layer)
                                assert abs(float(recorded['r_mean']) - summary['r']) < 1e-12
                                fits.append(dict(seed=seed, task=task, step=step, layer=layer,
                                                 ce=float(recorded['ce']), accuracy=float(recorded['accuracy'])))
                        if task not in targets:
                            continue
                        closed20 = data[f's20_{pop}_L{layer}_opened'] == 0
                        opened500 = data[f's500_{pop}_L{layer}_opened'] > 0
                        count = int(closed20.sum())
                        revived = int((closed20 & opened500).sum())
                        reopening.append(dict(seed=seed, task=task, population=pop, layer=layer,
                            closed_at20=count, reopened_by500=revived,
                            reopened_fraction=revived / count if count else None))
                        # 20->100 and 0->500 were added after seeing task-10
                        # recovery; these are exploratory, not registered windows.
                        for a, b in ((0, 20), (20, 100), (100, 150), (100, 500), (0, 500)):
                            pa, pb = f's{a}_{pop}_L{layer}_', f's{b}_{pop}_L{layer}_'
                            valid = data[pa + 'valid'] & data[pb + 'valid']
                            row = dict(seed=seed, task=task, population=pop, layer=layer,
                                       start=a, end=b, n_valid=int(valid.sum()))
                            for field in ('mu', 'sd', 'r'):
                                delta = data[pb + field] - data[pa + field]
                                row[f'delta_{field}'] = float(delta[valid].mean())
                            ma, mb = data[pa + 'mu'], data[pb + 'mu']
                            ia, ib = 1 / (data[pa + 'sd'] + 1e-12), 1 / (data[pb + 'sd'] + 1e-12)
                            mean_term = (mb - ma) * (ia + ib) / 2
                            width_term = (ma + mb) * (ib - ia) / 2
                            error = np.abs((mean_term + width_term - (data[pb + 'r'] - data[pa + 'r']))[valid]).max()
                            assert error < 1e-12, (seed, task, pop, layer, a, b, error)
                            row.update(r_mean_term=float(mean_term[valid].mean()),
                                       r_width_term=float(width_term[valid].mean()),
                                       r_identity_max_error=float(error),
                                       registered_window=(a, b) in ((0, 20), (100, 150), (100, 500)))
                            row['r_negative_fraction'] = float(((data[pb + 'r'] - data[pa + 'r'])[valid] < 0).mean())
                            row['sd_ratio_median'] = float(np.median((data[pb + 'sd'] / data[pa + 'sd'])[valid]))
                            changes.append(row)
    output = T.ROOT / 'results/vit_two_block_1025'
    output.mkdir(exist_ok=True)
    df, change = pd.DataFrame(times), pd.DataFrame(changes)
    df.to_csv(output / f'baseline_times_through{through}.csv', index=False)
    change.to_csv(output / f'baseline_changes_through{through}.csv', index=False)
    pd.DataFrame(fits).to_csv(output / f'baseline_fit_through{through}.csv', index=False)
    pd.DataFrame(reopening).to_csv(output / f'baseline_reopening_through{through}.csv', index=False)
    metrics = [('mu', 'Mean unit $\\mu$'), ('sd', 'Mean unit $\\sigma$'),
               ('r', 'Mean unit $\\mu/\\sigma$'), ('closed', 'Closed unit fraction')]
    for pop in ('train', 'validation'):
        fig, axes = plt.subplots(1 + len(targets), 4, figsize=(15, 3.3 * (1 + len(targets))), squeeze=False)
        for row, task in enumerate([None] + targets):
            source = df[(df.population == pop) & ((df.step == 500) if task is None else (df.task == task))]
            key = 'task' if task is None else 'step'
            for col, (metric, title) in enumerate(metrics):
                ax = axes[row, col]
                for layer in (1, 2):
                    part = source[source.layer == layer]
                    pivot = part.pivot(index=key, columns='seed', values=metric).sort_index()
                    assert pivot.shape[1] == 5 and not pivot.isna().any().any()
                    for seed in T.SEEDS:
                        ax.plot(pivot.index, pivot[seed], color=COLORS[layer], alpha=.22, linewidth=.8)
                    ax.plot(pivot.index, pivot.median(axis=1), color=COLORS[layer], linewidth=2.2, label=f'Block {layer}')
                ax.set_title(title)
                ax.set_xlabel('Task (after 500 updates)' if task is None else f'Update within task {task}')
                ax.grid(alpha=.2)
                if col == 0:
                    ax.legend(frameon=False)
        fig.suptitle(f'Depth-2 ViT | {pop} images | five seeds (thin), seed median (thick)', fontsize=13)
        fig.tight_layout()
        for ext in ('png', 'pdf'):
            fig.savefig(output / f'baseline_{pop}_through{through}.{ext}', dpi=180, bbox_inches='tight')
        plt.close(fig)
    lines = [f'# 2block基準走・全5seed・課題{through}まで', '',
        '各点でunitごとのμ・σ・μ/σを計算してからunit平均を取り、太線をseed中央値として描いた。平均μ÷平均σではない。',
        '閉じたunitは、その時点の評価画像全てで前活性が0以下のunit。全入力空間で死んだことを意味しない。', '',
        '| 課題 | block | 区間 | Δμ（seed中央値） | Δσ | Δr | rが下がるseed |',
        '|---:|---:|---|---:|---:|---:|---:|']
    for (task, layer, start, end), group in change[change.population == 'train'].groupby(['task', 'layer', 'start', 'end']):
        assert len(group) == 5
        lines.append(f'| {task} | {layer} | {start}→{end} | {group.delta_mu.median():.5g} | '
                     f'{group.delta_sd.median():.5g} | {group.delta_r.median():.5g} | {int((group.delta_r < 0).sum())}/5 |')
    lines += ['', '0→20は切替直後、100→500は登録した通常学習の観測窓。20→100と0→500は課題10の回復を見た後に追加した探索的な窓。起点のCE・精度は数表に残し、100更新を自動的に「適合後」とは呼ばない。', '',
              'CSVのr_mean_termとr_width_termは端点の対称分解: Δr = Δμ×平均[1/(σ+ε)] + 平均[μ]×Δ[1/(σ+ε)]。ここで平均は二つの時点の平均。unitごとに実Δrとの加法性を検算した。有限変位の帳簿であり、μ・σを独立に操作した因果割合ではない。', '',
              '第2blockについて、20更新時に学習250画像で閉じていたunitが、500更新までに同じ画像のいずれかで再び開く本数:', '',
              '| 課題 | seed | 20更新で閉じた本数 | そのうち500更新までに再開口 |',
              '|---:|---:|---:|---:|']
    for r in reopening:
        if r['layer'] == 2 and r['population'] == 'train':
            lines.append(f"| {r['task']} | {r['seed']} | {r['closed_at20']} | {r['reopened_by500']} |")
    lines += ['',
              f'![学習画像](baseline_train_through{through}.png)', '',
              f'![固定検証画像](baseline_validation_through{through}.png)', '']
    (output / f'baseline_summary_through{through}.md').write_text('\n'.join(lines))
    print(json.dumps(dict(status='complete', seeds=5, through=through, output=str(output))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--through-task', type=int, choices=(10, 60), required=True)
    run(parser.parse_args().through_task)
