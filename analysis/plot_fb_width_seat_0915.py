from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
out = root / 'results/fb_width_seat_0915'
arms = [('LRoff0_1216', 'SGD, free width', '#cf5c36', '--'),
        ('LRwf21_1216', 'SGD, fixed width', '#cf5c36', '-'),
        ('FB21LRoff0_1216', 'Full batch, free width', '#20639b', '--'),
        ('FB21LRwf21_1216', 'Full batch, fixed width', '#20639b', '-')]
fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
for arm, label, color, style in arms:
    curves = [[], [], []]
    for seed in range(10):
        with np.load(out / 'logs' / f'{arm}_seed{seed}.npz', allow_pickle=False) as z:
            if bool(z['numeric_divergence']):
                continue
            take = (z['step'] > 0) & (z['step'] % 10000 == 0)
            tasks = z['step'][take] // 10000
            for k, metric in enumerate(['layer1_zbar', 'layer1_zmax']):
                curves[k].append(np.median(z[metric][take].astype(float), axis=1))
            wt = z['layer1_w_free_step'] > 0
            assert np.array_equal(z['layer1_w_free_step'][wt] // 10000, tasks)
            norm = np.linalg.norm(z['layer1_w_free'][wt].astype(float), axis=-1)
            curves[2].append(np.median(norm, axis=1))
    for ax, rows in zip(axes, curves):
        if rows:
            ax.plot(tasks, np.mean(rows, axis=0), style, color=color, label=label, lw=1.8)
for ax, title, ylabel in zip(axes, ['Center', 'Upper edge', 'Free-weight norm'],
                           ['Mean preactivation', 'Maximum preactivation', 'L2 norm']):
    ax.set(title=title, xlabel='Task', ylabel=ylabel)
    ax.axvline(20, color='#666666', lw=.8, alpha=.6)
    ax.grid(alpha=.18)
axes[0].legend(frameon=False, fontsize=8)
fig.suptitle('condA: same SGD history through task 20; unit median per task, then seed mean', fontsize=11)
fig.savefig(out / 'registered_trajectories.png', dpi=180)
plt.close(fig)
