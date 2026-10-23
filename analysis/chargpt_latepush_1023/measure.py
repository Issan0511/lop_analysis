#!/usr/bin/env python3
"""Fixed-state, fixed-softmax readout decomposition; no optimizer or training.

Definitions/probes are imported from chargpt_seed10_1022/load10.py.
The sole gradient change is to feed (p_original-u) @ v into the unchanged body.
See preregistered spec §8 for exact meanings and the seed-0 gate.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import argparse
import csv
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / 'results/chargpt_latepush_1023'
os.environ['CHARGPT_SRC'] = str(ROOT)
sys.path.insert(0, str(HERE / 'source/src'))
spec = importlib.util.spec_from_file_location('load10', ROOT / 'analysis/chargpt_seed10_1022/load10.py')
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)
np, torch, C = A.np, A.torch, A.C


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n'); w.writeheader(); w.writerows(rows)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative_max(delta, *scales):
    return float(delta.abs().max()) / max(*(float(x.abs().max()) for x in scales), 1e-300)


@torch.no_grad()
def readouts(m, x, y):
    # Exact expression from A.head, retaining F instead of discarding it.
    h = torch.cat([m.body(xx).reshape(-1, C.D) for _, xx in A.chunks(x)])
    vc = m.head.weight.detach() - m.head.weight.detach().mean(0)
    fm = (torch.nn.functional.one_hot(y.reshape(-1), C.V).to(A.DT) - 1/C.V).T @ h
    kap = float((vc*fm).sum()/fm.square().sum())
    load = kap*fm
    res = vc-load
    orth = abs(float((res*fm).sum())) / max(float(vc.norm()*fm.norm()), 1e-300)
    return (vc, load, res), kap, orth, relative_max(vc-load-res, vc)


def fixed_error_gradients(m, x, vs):
    """Identical function for full/load/res; original logits/errors never change."""
    params = [b.mlp.up.weight for b in m.h]
    gs = [[torch.zeros_like(p) for p in params] for _ in vs]
    for _, xx in A.chunks(x):
        h = m.body(xx)
        with torch.no_grad():
            e = m.head(h.detach()).softmax(-1)-1/C.V
        for j, v in enumerate(vs):
            grad = (e @ v)/(len(x)*x.shape[1])
            gg = torch.autograd.grad(h, params, grad_outputs=grad, retain_graph=(j < len(vs)-1))
            for accum, value in zip(gs[j], gg):
                accum.add_(value.detach())
    return gs


def state(arm, seed, t, reference_gradient=False):
    m = A.load(arm, seed, t)
    x, y, _, _ = A.data(seed, max(1, t))
    head = A.head(m, x, y)
    vs, kap, orth, decomp = readouts(m, x, y)
    assert A.rel(kap, head['kappa']) <= 1e-12
    assert orth <= 1e-12 and decomp <= 1e-12
    gs = fixed_error_gradients(m, A.uniform(seed), vs)
    original = None
    if reference_gradient:
        original = A.units(m, x, A.gradient(m, A.uniform(seed)))[1]
    rows, raw = [], {}
    for l, (yy, z) in enumerate(A.moments(m, x)):
        ybar = yy.mean(0)
        alive = (z > 0).any(0)
        pushes = [(g[l]*ybar).sum(1) for g in gs]
        full, load, res = pushes
        err = relative_max(full-load-res, full, load, res)
        assert err <= 1e-10, (arm, seed, t, l, err)
        match = relative_max(full-original[l]['own'], original[l]['own']) if original else 0.0
        if original:
            assert match <= 1e-10, (arm, seed, t, l, match)
            assert torch.equal(alive, original[l]['alive'])
        denom = load[alive].abs()+res[alive].abs()
        shares = load[alive].abs()/denom
        kn = yy @ ybar
        rows.append(dict(seed=seed, arm=arm, task=t, layer=l, n_alive=int(alive.sum()),
                         kappa=head['kappa'], cos2=head['cos2'],
                         s_full=A.sink(full, alive), s_load=A.sink(load, alive), s_res=A.sink(res, alive),
                         share_load=float(np.median(shares.numpy())) if len(shares) else float('nan'),
                         share_zero_denominator=int((denom == 0).sum()),
                         abs_full_median=float(np.median(full[alive].abs().numpy())) if alive.any() else float('nan'),
                         abs_load_median=float(np.median(load[alive].abs().numpy())) if alive.any() else float('nan'),
                         abs_res_median=float(np.median(res[alive].abs().numpy())) if alive.any() else float('nan'),
                         kn_positive=float((kn > 0).double().mean()), kn_negative=float((kn < 0).double().mean()),
                         kn_zero=float((kn == 0).double().mean()), orthogonality_error=orth,
                         readout_additivity_error=decomp, push_additivity_error=err,
                         original_gradient_error=match if original else float('nan')))
        raw[f't{t}_L{l}_alive'] = alive.numpy()
        for name, v in zip(['full', 'load', 'res'], pushes):
            raw[f't{t}_L{l}_{name}'] = v.numpy()
    return rows, raw


def run_part(arm, seed, gate=False):
    dest = OUT / 'parts' / f'{arm}_seed{seed}.csv'
    done = dest.with_suffix('.json')
    if done.exists():
        d = json.loads(done.read_text())
        assert d['sha256'] == sha(dest)
        if not gate or d['original_gradient_checked']:
            return list(csv.DictReader(dest.open()))
    start = time.monotonic(); rows = []; raw = {}; timings = []
    for t in A.STATES:
        ts = time.monotonic()
        rr, qq = state(arm, seed, t, reference_gradient=gate)
        rows.extend(rr); raw.update(qq)
        timings.append(dict(task=t, seconds=time.monotonic()-ts))
        print(f'{arm} seed{seed} task{t}: {time.monotonic()-start:.1f}s', flush=True)
        # Keep the bounded probe cache; no next-task evals or optimizer are used.
    write_csv(dest, rows)
    rawpath = OUT / 'raw' / f'{arm}_seed{seed}.npz'
    rawpath.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(rawpath, **raw)
    dump(done, dict(arm=arm, seed=seed, states=len(A.STATES), seconds=time.monotonic()-start,
                    original_gradient_checked=gate, sha256=sha(dest), raw_sha256=sha(rawpath), timings=timings))
    return rows


def gate_check():
    ref = {(r['arm'], int(r['seed']), int(r['task']), int(r['layer'])): r
           for r in csv.DictReader((ROOT/'results/chargpt_seed10_1022/load/states.csv').open())}
    rows = []
    for arm in A.ARMS:
        rows.extend(run_part(arm, 0, gate=True))
    worst = dict(s_full=0., kappa=0., cos2=0., original_gradient=0.)
    failures = []
    for row in rows:
        key = (row['arm'], 0, int(row['task']), int(row['layer']))
        r = ref[key]
        e = abs(float(row['s_full'])-float(r['own_sink']))
        ek = A.rel(row['kappa'], r['head_kappa']); ec = A.rel(row['cos2'], r['head_cos2'])
        worst['s_full'] = max(worst['s_full'], e)
        worst['kappa'] = max(worst['kappa'], ek); worst['cos2'] = max(worst['cos2'], ec)
        worst['original_gradient'] = max(worst['original_gradient'], float(row['original_gradient_error']))
        if e > 1e-12 or ek > 1e-8 or ec > 1e-8 or int(row['n_alive']) != int(r['n_alive']):
            failures.append(dict(key=key, s_error=e, kappa_error=ek, cos2_error=ec))
    d = dict(passed=not failures, seed=0, arms=A.ARMS, states_per_arm=len(A.STATES), layer_rows=len(rows),
             worst=worst, failures=failures, reference_sha256=sha(ROOT/'results/chargpt_seed10_1022/load/states.csv'))
    dump(OUT/'reproduction_check.json', d)
    print(json.dumps(d, ensure_ascii=False), flush=True)
    assert d['passed'], 'STOP: seed-0 reproduction failed'


def main():
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=['gate', 'run']); args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.mode == 'gate':
        gate_check(); return
    gate = json.loads((OUT/'reproduction_check.json').read_text())
    assert gate['passed']
    start = time.monotonic(); rows = []
    for seed in range(10):
        for arm in A.ARMS:
            rows.extend(run_part(arm, seed))
        A.CACHE.clear(); A.UCACHE.clear()
    write_csv(OUT/'states.csv', rows)
    dump(OUT/'provenance.json', dict(
        finished=datetime.datetime.now(datetime.timezone.utc).isoformat(), python=sys.version,
        torch=torch.__version__, numpy=np.__version__, host=platform.node(), threads=torch.get_num_threads(),
        device='CPU', dtype='float64', training_steps=0,
        git_hash=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        spec_sha256=sha(ROOT/'specs/spec_chargpt_latepush_1023.md'),
        load10_sha256=sha(ROOT/'analysis/chargpt_seed10_1022/load10.py'),
        measure_sha256=sha(Path(__file__)), source_manifest=json.loads((HERE/'source_manifest.json').read_text()),
        rows=len(rows), states=720, task_end_states=700, task_ends=A.TASKS, includes_init=True,
        elapsed_run_seconds=time.monotonic()-start,
        raw_files=[dict(path=str(p.relative_to(ROOT)), bytes=p.stat().st_size, sha256=sha(p)) for p in sorted((OUT/'raw').glob('*.npz'))]))


if __name__ == '__main__':
    main()
