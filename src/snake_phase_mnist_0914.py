"""snake_phase_mnist_0914 training (spec_snake_phase_mnist_0914 §2, §5).

Box B = gate_shape_0911 protocol.  The per-step update is character-identical to
GS.train (Adam, then AdaptiveSnake.update); the committed read-outs C.measure and
GS.gate_block are called unchanged, so N06/SNA/LR/LIN reproduce gate_shape_0911
SN06/SNA/LR/LIN bit for bit (G1).  Everything added is a no_grad read-out or runs
on separate tensors (fresh probes), and never draws from the task generators.

Exit codes: 0 COMPLETE, 3 DIVERGED.
"""
from pathlib import Path
import argparse, hashlib, json, math, os, resource, subprocess, sys, time

import numpy as np
import torch

from src import gate_shape_0911 as GS
from src import snake_phase_acts_0914 as A
from src import transport_common_0910 as T

C = GS.C
H = GS.H
ROOT = GS.ROOT
SPEC = ROOT / 'specs/spec_snake_phase_mnist_0914.md'
CONFIG = ROOT / 'configs/snake_phase_mnist_0914.yaml'
FRESH_TASKS = [1] + list(range(16, 21)) + list(range(101, 121))
SNAP_TASKS = [20, 60, 120]
STEPS = 625
L2KEYS = ('cnorm', 'm', 'b', 'zcur', 'sdcur', 'gbar', 'A', 'abar', 'outcol', 'alpha')
GSKEYS = ('zbar_i', 'sd_i', 'cnorm_i', 'm_i', 'bias_i', 'star_i', 'gbar_i', 'gvar_i', 'off_i',
          'offabs_i', 'off10_i', 'off50_i', 'q10_i', 'zcur_i', 'sdcur_i', 'hard_i')


class Diverged(Exception):
    def __init__(self, task, step):
        super().__init__(f'DIVERGED({task},{step})'); self.task, self.step = task, step


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def param_sha(p):
    h = hashlib.sha256()
    for q in p:
        h.update(q.detach().numpy().tobytes())
    return h.hexdigest()


def is_ada(act):
    return isinstance(act, H.AdaptiveSnake)


def theta0(arm, seed, mutate_init=None, map_mutate=None):
    p = H.init_params(seed, torch.device('cpu'))
    if mutate_init == 'init':
        with torch.no_grad():
            p[0].data[0, 0] += 1e-3
    A.apply_init_map(p, A.ARM_TABLE[arm][1], mutate=map_mutate)
    return p


def adam_task(p, act, adam, xs, ys, hook=None, inject=None, task=None):
    """625 steps, character-identical to GS.train's inner loop.  hook(step) is a read-out."""
    for step in range(1, STEPS + 1):
        out = H.forward(p, xs[(step - 1) * 16:step * 16], act)
        loss = torch.nn.functional.cross_entropy(out[4], ys[(step - 1) * 16:step * 16])
        if inject is not None and inject == (task, step):
            loss = loss * float('nan')
        if not bool(torch.isfinite(loss)):
            raise Diverged(task, step)
        gr = torch.autograd.grad(loss, p)
        with torch.no_grad():
            m_, v_, tc = adam
            tc[0] += 1
            c1 = 1 - .9 ** tc[0]; c2 = 1 - .999 ** tc[0]
            for q, g, mi, vi in zip(p, gr, m_, v_):
                mi.mul_(.9).add_(g, alpha=1 - .9)
                vi.mul_(.999).addcmul_(g, g, value=1 - .999)
                q -= .001 * (mi / c1) / ((vi / c2).sqrt() + 1e-8)
            if is_ada(act):
                act.update(out[0], out[2])
        if hook is not None:
            hook(step)


def probe_ce(p, act, probe, perm):
    with torch.no_grad():
        return float(torch.nn.functional.cross_entropy(H.forward(p, probe.px[:, perm], act)[4], probe.py))


def test_correct(p, act, mnist, perm):
    with torch.no_grad():
        return int((H.forward(p, mnist.test_x[:, perm], act)[4].argmax(1) == mnist.test_y).sum())


def unit_block(p, act, probe, perm, mut=None):
    """Layer 1 and 2 per-unit arrays (float64 after the float32 forward)."""
    out = {}
    with torch.no_grad():
        z1, a1, z2, a2, _ = H.forward(p, probe.px[:, perm], act)
        for l, (z, W, b, Wn) in enumerate(((z1, p[0], p[1], p[2]), (z2, p[2], p[3], p[4])), start=1):
            Wd = W.double(); m = Wd.mean(1)
            if mut == 'w2row' and l == 2:
                Wd = Wd.clone(); Wd[0] += 1e-3; m = Wd.mean(1)
            out[f'L{l}_cnorm'] = (Wd - m[:, None]).norm(dim=1)
            out[f'L{l}_m'] = m
            out[f'L{l}_b'] = b.double()
            zd = z.double()
            out[f'L{l}_zcur'] = zd.mean(0)
            out[f'L{l}_sdcur'] = zd.std(0, unbiased=False)
            g = (act.dphi(z, l - 1) if is_ada(act) else act.dphi(z)).double()
            out[f'L{l}_gbar'] = g.mean(0)
            ph = (act.phi(z, l - 1) if is_ada(act) else act.phi(z)).double()
            out[f'L{l}_abar'] = ph.mean(0)
            out[f'L{l}_outcol'] = Wn.double().norm(dim=0)
            if is_ada(act):
                al = act.alpha(l - 1).double()
            elif isinstance(act, A.PhaseSnake):
                al = torch.full((W.shape[0],), act.param, dtype=torch.float64)
            else:
                al = torch.full((W.shape[0],), float('nan'), dtype=torch.float64)
            out[f'L{l}_alpha'] = al
            if torch.isnan(al).any():
                out[f'L{l}_A'] = torch.full_like(al, float('nan'))
            else:
                coef = 2.0 * al if mut != 'A_alpha' else al
                ang = coef[None, :] * zd
                out[f'L{l}_A'] = torch.sqrt(torch.cos(ang).mean(0) ** 2 + torch.sin(ang).mean(0) ** 2)
    return {k: v.numpy().copy() for k, v in out.items()}


def cur_cpu():
    try:
        return int(open('/proc/self/stat').read().rsplit(')', 1)[1].split()[36])
    except Exception:
        return -1


def vmrss_kb():
    for line in open('/proc/self/status'):
        if line.startswith('VmRSS:'):
            return int(line.split()[1])
    return -1


def run(arm, seed, tasks, out, fresh_tasks=None, snap_tasks=None, extras=True, snapshots=True,
        inject=None, mutate=None):
    """mutate (check-only): 'init' (W1[0,0]+=1e-3), 'probe_draw' (fresh probe draws once from the
    batch generator), 'extra_w1' (read-out adds 1e-9 to W1)."""
    torch.set_num_threads(1); H.setup('cpu'); T.data_dir()
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    fresh_tasks = FRESH_TASKS if fresh_tasks is None else fresh_tasks
    snap_tasks = SNAP_TASKS if snap_tasks is None else snap_tasks
    if not extras:
        fresh_tasks, snap_tasks = [], []
    if not snapshots:
        snap_tasks = []
    t_start = time.monotonic()
    mnist = H.Mnist(torch.device('cpu'))
    pidx = torch.randperm(len(mnist.test_x), generator=H.stream('boundary_probe', seed))[:512]
    probe = C.Probe(mnist.test_x[pidx], mnist.test_y[pidx], float(mnist.train_x.mean()))

    p = theta0(arm, seed, mutate_init=mutate)
    p0 = [q.detach().clone() for q in p]
    act = A.make_act(arm)
    adam = ([torch.zeros_like(q) for q in p], [torch.zeros_like(q) for q in p], [0])
    gp, gd, gb = H.stream('perm', seed), H.stream('data', seed), H.stream('batch', seed)

    rows_path = out / 'rows.csv'
    fh = open(rows_path, 'w')
    header = None
    units = {k: [] for k in (f'L{l}_{n}' for l in (1, 2) for n in L2KEYS)} if extras else {}
    gsu = {k: [] for k in GSKEYS}
    status, diverged_at, stream_hashes, snaps = 'COMPLETE', None, [], {}
    maxcpu = []
    try:
        for task in range(1, tasks + 1):
            perm = torch.randperm(784, generator=gp)
            idx = H.stratified_draw(mnist, gd)
            order = torch.randperm(H.TASK_EXAMPLES, generator=gb)
            stream_hashes.append(hashlib.sha256(perm.numpy().tobytes() + idx.numpy().tobytes()
                                                + order.numpy().tobytes()).hexdigest())
            xs = mnist.train_x[idx][:, perm][order]
            ys = mnist.train_y[idx][order]
            if extras and task == 1:
                for k, v in unit_block(p, act, probe, perm).items():
                    units[k].append(v)
            ce0 = probe_ce(p, act, probe, perm) if extras else float('nan')
            ce20 = [float('nan')]
            _, _, g_start = GS.gate_block(p, act, probe, perm)

            def hook(step):
                if step == 20:
                    ce20[0] = probe_ce(p, act, probe, perm)
            adam_task(p, act, adam, xs, ys, hook=hook if extras else None, inject=inject, task=task)
            if not all(bool(torch.isfinite(q).all()) for q in p):
                raise Diverged(task, STEPS)
            # ---- task end: committed read-outs, in GS.train's order
            r, u = C.measure(p, act, probe, perm, True, mnist, None)
            gr_, gu, g_end = GS.gate_block(p, act, probe, perm)
            r.update(gr_)
            r['gcorr'], r['gcorr_degenerate'] = GS.gcorr(g_start, g_end)
            r.update(task=task, step=625, clamp='ref')
            if is_ada(act):
                r['alpha_med'] = float(act.alpha(0).median())
            for k in GSKEYS:
                gsu[k].append((u if k in u else gu)[k])
            if extras:
                ub = unit_block(p, act, probe, perm)
                for k, v in ub.items():
                    units[k].append(v)
            if mutate == 'extra_w1':
                with torch.no_grad():
                    p[0].data.add_(1e-9)
            correct = int(round(r['acc'] * 10000))
            r.update(correct_seq=correct, acc_seq=correct / 10000, ce0=ce0, ce20=ce20[0],
                     correct_fresh='', acc_fresh='', param_sha=param_sha(p))
            if task in fresh_tasks:
                if mutate == 'probe_draw':
                    torch.randperm(2, generator=gb)
                pf = [q.clone().requires_grad_(True) for q in p0]
                actf = A.make_act(arm)
                adamf = ([torch.zeros_like(q) for q in pf], [torch.zeros_like(q) for q in pf], [0])
                adam_task(pf, actf, adamf, xs, ys, task=('fresh', task))
                cf = test_correct(pf, actf, mnist, perm)
                r.update(correct_fresh=cf, acc_fresh=cf / 10000,
                         fresh_param_sha=param_sha(pf) if task == 1 else '')
                del pf, actf, adamf
            if task in snap_tasks:
                snap = dict(p=[q.detach().clone() for q in p],
                            adam=([x.clone() for x in adam[0]], [x.clone() for x in adam[1]], adam[2][0]),
                            V=[v.clone() for v in act.V] if is_ada(act) else None,
                            gens=[g.get_state() for g in (gp, gd, gb)], arm=arm, seed=seed, task=task)
                sp = out / f'snap_t{task:03d}.pt'
                torch.save(snap, sp); snaps[sp.name] = sha_file(sp)
            r.update(secs=time.monotonic() - t_start, vmrss_kb=vmrss_kb(),
                     maxrss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, cpu=cur_cpu(),
                     arm=arm, seed=seed)
            r.setdefault('fresh_param_sha', '')
            if header is None:
                header = list(r.keys())
                fh.write(','.join(header) + '\n')
            fh.write(','.join(str(r.get(k, '')) for k in header) + '\n'); fh.flush()
    except Diverged as e:
        status, diverged_at = 'DIVERGED', [e.task, e.step]
    finally:
        fh.close()
        arr = {}
        for k, v in units.items():
            if v:
                arr[k] = np.stack(v)
        for k, v in gsu.items():
            if v:
                arr['gs_' + k] = np.stack(v)
        np.savez_compressed(out / 'units.npz', **arr)
    try:
        gh = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    except Exception:
        gh = ''
    cpu_model = next((l.split(':', 1)[1].strip() for l in open('/proc/cpuinfo') if l.startswith('model name')), '')
    prov = dict(run_id='snake_phase_mnist_0914', arm=arm, seed=seed, tasks=tasks, status=status,
                diverged_at=diverged_at, extras=extras, fresh_tasks=fresh_tasks, snap_tasks=snap_tasks,
                inject=inject, mutate=mutate, git_hash=gh,
                sha256=dict(host=sha_file(H.__file__), measure=sha_file(C.__file__), gate_shape=sha_file(GS.__file__),
                            acts=sha_file(A.__file__), script=sha_file(__file__), spec=sha_file(SPEC),
                            config=sha_file(CONFIG)),
                data_sha256=mnist.sha256, torch_version=torch.__version__, numpy_version=np.__version__,
                cpu_model=cpu_model, wall_seconds=time.monotonic() - t_start,
                maxrss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                stream_sha256=stream_hashes, snapshot_sha256=snaps)
    (out / 'provenance.json').write_text(json.dumps(prov, indent=1))
    print('FINISHED', arm, seed, status, diverged_at, round(prov['wall_seconds'], 1), 's', flush=True)
    return status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True, choices=A.ARMS)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--tasks', type=int, default=120)
    ap.add_argument('--out', required=True)
    ap.add_argument('--force-probe-tasks', default=None)
    ap.add_argument('--inject-nan', default=None)
    ap.add_argument('--no-extras', action='store_true')
    ap.add_argument('--no-snapshots', action='store_true')
    ap.add_argument('--mutate', default=None, choices=[None, 'init', 'probe_draw', 'extra_w1'])
    a = ap.parse_args()
    A.check_config_constants()
    ft = [int(x) for x in a.force_probe_tasks.split(',')] if a.force_probe_tasks else None
    st = ft if ft is not None else None
    inj = tuple(int(x) for x in a.inject_nan.split(',')) if a.inject_nan else None
    status = run(a.arm, a.seed, a.tasks, a.out, fresh_tasks=ft, snap_tasks=st, extras=not a.no_extras,
                 snapshots=not a.no_snapshots, inject=inj, mutate=a.mutate)
    sys.exit(3 if status == 'DIVERGED' else 0)


if __name__ == '__main__':
    main()
