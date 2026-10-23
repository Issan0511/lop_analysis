"""Copy the small per-run records from the raw root into results/vit_long_1024/ (spec §10.1)."""
import argparse
import shutil
from pathlib import Path

from . import common as C

SMALL = ('per_task.csv', 'layer_stats.csv', 'intask.npz', 'provenance.json', 'done.json', 'manifest.json',
         'status.json', 'diverged_task.npz')


def collect(root=None, dest=None):
    root = Path(root or C.RAW)
    dest = Path(dest or C.RESULTS)
    copied = []
    for done in sorted((root / 'runs').glob('**/provenance.json')):
        run = done.parent
        target = dest / run.relative_to(root / 'runs')
        target.mkdir(parents=True, exist_ok=True)
        for name in SMALL:
            if (run / name).exists():
                shutil.copy2(run / name, target / name)
                copied.append(str((target / name).relative_to(dest)))
    return copied


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root')
    parser.add_argument('--dest')
    args = parser.parse_args()
    print(f'copied {len(collect(args.root, args.dest))} files')
