"""sgd_bridge_mnist_0914 training (spec_sgd_bridge_mnist_0914 §1-§3).

Runs the committed snake_phase_mnist_0914 loop with ONE substitution: the per-task Adam
function is replaced (in this process only; no file is edited) by plain SGD with the
update rule of H.run_one(optimizer='sgd', epochs=4): p -= lr * g over the same 625-batch
order re-walked 4 times.  Fresh probes call the same substituted function, so they are
SGD from theta0 with 2,500 updates.  The Adam state tuple is still allocated by the
loop but never read.

Exit codes: 0 COMPLETE, 3 DIVERGED.
"""
from pathlib import Path
import argparse, hashlib, json, sys

import torch

from src import snake_phase_mnist_0914 as SP
from src import snake_phase_acts_0914 as A

H = SP.H
ROOT = SP.ROOT
SPEC = ROOT / 'specs/spec_sgd_bridge_mnist_0914.md'
CONFIG = ROOT / 'configs/sgd_bridge_mnist_0914.yaml'
ARMS = ['N06', 'P06', 'V06', 'LIN', 'LR']
STEPS = 625


def make_sgd_task(lr=0.02, epochs=4):
    def sgd_task(p, act, adam, xs, ys, hook=None, inject=None, task=None):
        for s in range(STEPS * epochs):
            j = s % STEPS
            xb = xs[j * 16:(j + 1) * 16]
            yb = ys[j * 16:(j + 1) * 16]
            out = H.forward(p, xb, act)
            loss = torch.nn.functional.cross_entropy(out[4], yb)
            if inject is not None and inject == (task, s + 1):
                loss = loss * float('nan')
            if not bool(torch.isfinite(loss)):
                raise SP.Diverged(task, s + 1)
            grads = torch.autograd.grad(loss, p)
            with torch.no_grad():
                for q, g in zip(p, grads):
                    q -= lr * g
                if isinstance(act, H.AdaptiveSnake):
                    act.update(out[0], out[2])
            if hook is not None:
                hook(s + 1)
    return sgd_task


def run(arm, seed, tasks, out, lr=0.02, epochs=4, **kw):
    SP.adam_task = make_sgd_task(lr, epochs)
    status = SP.run(arm, seed, tasks, out, **kw)
    pj = Path(out) / 'provenance.json'
    prov = json.loads(pj.read_text())
    prov.update(run_id='sgd_bridge_mnist_0914', optimizer=dict(kind='sgd', lr=lr, epochs_per_task=epochs,
                                                               updates_per_task=STEPS * epochs),
                substitution='SP.adam_task replaced in-process by sgd_task (src/sgd_bridge_mnist_0914.py)',
                sha256_bridge=dict(script=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                   spec=hashlib.sha256(SPEC.read_bytes()).hexdigest(),
                                   config=hashlib.sha256(CONFIG.read_bytes()).hexdigest()))
    pj.write_text(json.dumps(prov, indent=1))
    return status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True, choices=ARMS)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--tasks', type=int, default=120)
    ap.add_argument('--out', required=True)
    ap.add_argument('--lr', type=float, default=0.02)
    ap.add_argument('--epochs', type=int, default=4)
    ap.add_argument('--force-probe-tasks', default=None)
    ap.add_argument('--no-extras', action='store_true')
    ap.add_argument('--mutate', default=None, choices=[None, 'probe_draw'])
    a = ap.parse_args()
    A.check_config_constants()
    ft = [int(x) for x in a.force_probe_tasks.split(',')] if a.force_probe_tasks else None
    status = run(a.arm, a.seed, a.tasks, a.out, lr=a.lr, epochs=a.epochs, fresh_tasks=ft, snap_tasks=ft,
                 extras=not a.no_extras, mutate=a.mutate)
    sys.exit(3 if status == 'DIVERGED' else 0)


if __name__ == '__main__':
    main()
