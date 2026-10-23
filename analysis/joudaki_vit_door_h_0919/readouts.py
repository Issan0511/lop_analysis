"""Descriptive secondary readouts [spec 追補 10 §7]; never a decision endpoint.

Both quantities come from the preactivation snapshots the run already saves for
16 fixed validation images, so they cost no GPU time.  They exist to make a null
result on RH-R informative: if plain R never sinks, LN is already doing door H's
job, and that is the finding rather than a miss.
"""
import numpy as np

TASKS = (1, 21, 40)          # early / window start / final, matching the main window


def run_readouts(folder, layers=6, tasks=TASKS):
    """Per layer and task: negative-preactivation mass, and the channel median of z/sd(z)."""
    rows = []
    for task in tasks:
        path = folder / f'preact_t{task:02d}.npz'
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as data:
            for layer in range(1, layers + 1):
                key = f'z{layer}'
                if key not in data:
                    continue
                z = data[key].astype(np.float64).reshape(-1, data[key].shape[-1])
                sd = z.std(0, ddof=0)
                # Channels with no spread carry no ratio; report them separately
                # rather than dividing by zero and hiding them in the median.
                alive = sd > 0
                ratio = np.median(z.mean(0)[alive] / sd[alive]) if alive.any() else None
                rows.append(dict(task=task, layer=layer,
                                 negative_mass=float((z < 0).mean()),
                                 zbar_over_sd_median=None if ratio is None else float(ratio),
                                 dead_channels=int((~alive).sum()),
                                 channels=int(z.shape[1])))
    return rows


def collect(root, arms, seeds=range(10)):
    rows = []
    for arm in arms:
        for seed in seeds:
            folder = root / arm / f'seed{seed}'
            for row in run_readouts(folder):
                rows.append(dict(arm=arm, seed=seed, **row))
    return rows
