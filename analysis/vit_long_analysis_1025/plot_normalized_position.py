"""Plot the recorded per-unit mean / standard deviation (not mean / variance).

Uses the frozen 2026-10-25 13:12 JST analysis snapshot. No checkpoint inference,
training, or live-run polling. The source column is median_j(mu_j/(sigma_j+1e-12));
it is not the ratio of separately aggregated means and standard deviations.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {'R0L': '#157a87', 'SL': '#365bc2'}
LABELS = {'R0L': 'Head reset', 'SL': 'Head shared'}
LAST = {'6blk': 6, '2blk': 2}
N = {'6blk': 5, '2blk': 3}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def draw(ax, data, title, legend=False):
    for arm in ['R0L', 'SL']:
        sub = data[data.arm == arm]
        p = sub.pivot(index='task', columns='seed', values='cls_r_med').sort_index()
        assert len(p) == 61 and p.notna().all().all()
        for seed in p:
            ax.plot(p.index, p[seed], color=COLORS[arm], lw=.8, alpha=.23)
        ax.plot(p.index, p.median(axis=1), color=COLORS[arm], lw=2.5, label=LABELS[arm])
    ax.axhline(0, color='#777777', lw=.8, ls='--')
    ax.set_title(title, weight='bold', fontsize=11)
    ax.set_xlim(0, 60)
    ax.set_xticks(range(0, 61, 10))
    ax.set_xlabel('Task')
    ax.set_ylabel(r'Median $\mu_j / \sigma_j$ (CLS)')
    if legend:
        ax.legend(frameon=False, fontsize=10, loc='lower left')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True, help='Existing frozen analysis result directory')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    source = args.source / 'layer_metrics.csv'
    data = pd.read_csv(source)
    data = data[data.arm.isin(['R0L', 'SL'])].copy()
    assert np.isfinite(data[['r_med', 'cls_r_med']].to_numpy()).all()
    # Audit the plotted values against each immutable archived per-run CSV.
    manifest = json.loads((args.source / 'backup_manifest.json').read_text())
    snapshot = Path(manifest['snapshot_root'])
    hashes = {f['snapshot']: f['sha256'] for f in manifest['files']}
    for (variant, arm, seed), group in data.groupby(['variant', 'arm', 'seed']):
        key = f'{variant}/{arm}/seed{seed}/layer_stats.csv'
        p = snapshot / key
        assert sha(p) == hashes[key]
        old = pd.read_csv(p).set_index(['task', 'layer']).sort_index()
        new = group.set_index(['task', 'layer']).sort_index()
        assert old.index.equals(new.index)
        np.testing.assert_allclose(old[['r_med', 'cls_r_med']], new[['r_med', 'cls_r_med']], rtol=0, atol=1e-12)
    for variant, depth in LAST.items():
        for arm in ['R0L', 'SL']:
            g = data[(data.variant == variant) & (data.arm == arm)]
            assert g.seed.nunique() == N[variant]
            assert len(g) == N[variant] * depth * 61
            assert set(g.layer) == set(range(1, depth+1))
        a = data[(data.variant == variant) & (data.arm == 'R0L') & data.task.isin([0, 1])]
        b = data[(data.variant == variant) & (data.arm == 'SL') & data.task.isin([0, 1])]
        key = ['seed', 'task', 'layer']
        np.testing.assert_array_equal(a.sort_values(key).cls_r_med, b.sort_values(key).cls_r_med)
    selected = data[[int(layer) == LAST[variant] for variant, layer in zip(data.variant, data.layer)]]
    # Aggregate units first (already recorded), then seeds. No smoothing.
    rows = []
    for (variant, arm, task, layer), g in data.groupby(['variant', 'arm', 'task', 'layer']):
        rows.append(dict(variant=variant, arm=arm, task=int(task), layer=int(layer), n=len(g),
                         cls_r_seed_median=float(g.cls_r_med.median()),
                         cls_r_seed_min=float(g.cls_r_med.min()), cls_r_seed_max=float(g.cls_r_med.max()),
                         all_token_r_seed_median=float(g.r_med.median())))
    summary = pd.DataFrame(rows)
    summary.to_csv(args.out / 'normalized_position_by_task.csv', index=False)
    data[['variant', 'arm', 'seed', 'task', 'layer', 'cls_r_med', 'r_med']].to_csv(
        args.out / 'normalized_position_per_seed.csv', index=False)
    windows = []
    for (variant, arm), g in selected.groupby(['variant', 'arm']):
        for start, end in [(2, 11), (21, 30), (41, 60)]:
            values = g[g.task.between(start, end)].groupby('seed').cls_r_med.mean()
            windows.append(dict(variant=variant, arm=arm, first_task=start, last_task=end, n=len(values),
                                median_of_seed_window_means=float(values.median()),
                                min_of_seed_window_means=float(values.min()),
                                max_of_seed_window_means=float(values.max())))
    pd.DataFrame(windows).to_csv(args.out / 'normalized_position_windows.csv', index=False)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.grid': True, 'grid.alpha': .18, 'figure.facecolor': 'white'})
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8), sharey=True)
    for ax, variant in zip(axes, LAST):
        title = ('6 blocks / FFN 1536 | 5 seeds' if variant == '6blk' else
                 '2 blocks / FFN 3072 | 3 seeds, interim')
        draw(ax, selected[selected.variant == variant], title, legend=True)
    lo = np.floor(selected.cls_r_med.min() * 5) / 5 - .1
    hi = max(.2, np.ceil(selected.cls_r_med.max() * 5) / 5 + .05)
    axes[0].set_ylim(lo, hi)
    fig.suptitle(r'LN models: mean / standard deviation in the final FFN', fontsize=15, weight='bold')
    fig.text(.5, .025, 'Fixed 64-image probe, CLS position. Thick: seed median of per-unit ratios; thin: individual seeds.\n'
             'Unsmoothed. Frozen input: 2026-10-25 13:12 JST. No new training.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .11, 1, .92))
    fig.savefig(args.out / 'normalized_position.png', dpi=180)
    fig.savefig(args.out / 'normalized_position.pdf')
    plt.close(fig)
    fig, axes = plt.subplots(3, 3, figsize=(13, 10), sharex=True, sharey=True)
    panels = [('6blk', i) for i in range(1, 7)] + [('2blk', 1), ('2blk', 2)]
    for ax, (variant, layer) in zip(axes.flat, panels):
        draw(ax, data[(data.variant == variant) & (data.layer == layer)],
             f'{variant}: FFN {layer} (n={N[variant]})', legend=False)
    axes.flat[-1].axis('off')
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles, labels, loc='center', frameon=False)
    axes.flat[-1].text(.5, .30, 'Thick: seed median\nThin: individual seeds\nCLS position, no smoothing',
                       ha='center', va='center', transform=axes.flat[-1].transAxes)
    axes.flat[0].set_ylim(np.floor(data.cls_r_med.min()*2)/2-.1, max(.2,np.ceil(data.cls_r_med.max()*2)/2+.1))
    fig.suptitle(r'LN models: $\mu/\sigma$ in every FFN (CLS position)', fontsize=15, weight='bold')
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(args.out / 'normalized_position_all_layers.png', dpi=160)
    plt.close(fig)
    check = dict(source_file='layer_metrics.csv', source_sha256=sha(source),
                 snapshot_time_utc=manifest['captured_at_utc'], scope='LN arms, all FFN blocks, CLS position',
                 definition='median_seed(median_unit(mu_unit / (sample_sd_unit + 1e-12)))',
                 denominator='sample standard deviation across fixed 64-image CLS probe, ddof=1; not variance',
                 smoothing='none', complete_runs_checked=16, rows_checked=len(data),
                 archived_csv_values_match=True, paired_init_and_task1_match=True,
                 note='This is a dimensionless position measure, not a direct measurement of a driving force.')
    (args.out / 'normalized_position_checks.json').write_text(json.dumps(check, indent=2)+'\n')
    print(json.dumps(check, indent=2))
    print(selected.groupby(['variant', 'task', 'arm']).cls_r_med.median().unstack('arm').loc[
        (slice(None), [1,10,20,30,40,50,60]), :].round(5).to_string())


if __name__ == '__main__':
    main()
