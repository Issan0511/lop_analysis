"""sgd_bridge_mnist_0914 stability pilot (spec §5-3): seeds 100-102 x 5 arms x t1-40, no extras.
Any DIVERGED at lr 0.02 -> re-pilot once at lr 0.01; still diverging -> SGD_UNSTABLE.  Writes pilot.json."""
import json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / 'results/_smoke_sgd_bridge_mnist_0914/pilot'
ARMS = ['N06', 'P06', 'V06', 'LIN', 'LR']


def one(args):
    arm, seed, lr = args
    d = OUT / f'lr{lr}' / f'{arm}_s{seed}'
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    r = subprocess.run([sys.executable, '-m', 'src.sgd_bridge_mnist_0914', '--arm', arm, '--seed', str(seed), '--tasks', '40',
                        '--out', str(d), '--lr', str(lr), '--no-extras'], cwd=ROOT, env=env, capture_output=True, text=True)
    st = json.loads((d / 'provenance.json').read_text())['status'] if (d / 'provenance.json').exists() else f'ERROR rc={r.returncode}'
    return f'{arm}_s{seed}', st


def pilot(lr, jobs):
    with ThreadPoolExecutor(jobs) as ex:
        return dict(ex.map(one, [(a, s, lr) for s in (100, 101, 102) for a in ARMS]))


def main():
    jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    OUT.mkdir(parents=True, exist_ok=True)
    r1 = pilot(0.02, jobs)
    res = dict(lr_0_02=r1)
    if all(v == 'COMPLETE' for v in r1.values()):
        res['decision'] = 'GO'
    else:
        r2 = pilot(0.01, jobs)
        res['lr_0_01'] = r2
        res['decision'] = 'GO_LR_0.01' if all(v == 'COMPLETE' for v in r2.values()) else 'SGD_UNSTABLE'
    (OUT / 'pilot.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


if __name__ == '__main__':
    main()
