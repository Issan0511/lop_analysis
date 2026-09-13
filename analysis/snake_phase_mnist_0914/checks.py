"""snake_phase_mnist_0914 S checks (spec §10).  Every check has a mutation control; a check
passes only if the check itself passes AND its mutation is detected.

    python3 analysis/snake_phase_mnist_0914/checks.py            -> checks.json
    python3 analysis/snake_phase_mnist_0914/checks.py --only S2,S4
    python3 analysis/snake_phase_mnist_0914/checks.py --g1-full results/snake_phase_mnist_0914/runs
"""
import argparse, csv, hashlib, json, math, os, re, shutil, subprocess, sys, tempfile, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('MKL_NUM_THREADS', '1')

import numpy as np
import torch

from src import snake_phase_acts_0914 as A
from src import snake_phase_mnist_0914 as SP

H, C, GS = SP.H, SP.C, SP.GS
HERE = Path(__file__).resolve().parent
SMOKE = ROOT / 'results/_smoke_snake_phase_mnist_0914/checks'
EPS64 = float(torch.finfo(torch.float64).eps)
EPS32 = float(torch.finfo(torch.float32).eps)
GSK = SP.GSKEYS
torch.set_num_threads(1)


def result(ok, mut, **detail):
    return dict(pass_=bool(ok and mut), check_ok=bool(ok), mutation_detected=bool(mut), detail=detail)


def rows_of(d):
    return list(csv.DictReader(open(Path(d) / 'rows.csv')))


def train(arm, seed, tasks, out, **kw):
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    return SP.run(arm, seed, tasks, out, **kw)


# ------------------------------------------------------------------ S1 G1 anchor
def g1_compare(units_path, rows_path, anchor, seed, tasks):
    u = np.load(units_path)
    ref = np.load(ROOT / f'results/gate_shape_0911/{anchor}_s{seed}_units.npz')
    rr = list(csv.DictReader(open(ROOT / f'results/gate_shape_0911/{anchor}_s{seed}_rows.csv')))
    mr = list(csv.DictReader(open(rows_path)))
    state, derived, first_state, first_derived, n = 0., 0., None, None, 0
    for t in range(1, tasks + 1):
        for k in GSK:
            key = f'ref_{k}_t{t}'
            if key not in ref.files or u['gs_' + k].shape[0] < t:
                continue
            d = float(np.abs(u['gs_' + k][t - 1].astype(np.float64) - ref[key].astype(np.float64)).max())
            n += 1
            if k in ('cnorm_i', 'm_i', 'bias_i'):
                state = max(state, d)
                if d > 0 and first_state is None:
                    first_state = t
            else:
                # derived float64 statistics are produced by the same committed code in the same
                # order, so their rounding bound is 0; any difference is a code difference.
                derived = max(derived, d)
                if d > 0 and first_derived is None:
                    first_derived = t
    acc_mismatch = [i + 1 for i, (a, b) in enumerate(zip(rr[:tasks], mr[:tasks])) if a['acc'] != b['acc']]
    return dict(state_maxabs=state, derived_maxabs=derived, first_state_mismatch=first_state,
                first_derived_mismatch=first_derived, arrays_compared=n, expected=16 * tasks,
                acc_mismatch_tasks=acc_mismatch[:5], rows=len(mr))


def S1():
    det, ok = {}, True
    for arm, anc in A.ANCHOR.items():
        out = SMOKE / f'S1/{arm}'
        train(arm, 0, 3, out, fresh_tasks=[1], snap_tasks=[])
        c = g1_compare(out / 'units.npz', out / 'rows.csv', anc, 0, 3)
        det[arm] = c
        ok &= c['state_maxabs'] == 0.0 and c['derived_maxabs'] == 0.0 and c['arrays_compared'] == c['expected'] \
            and not c['acc_mismatch_tasks']
    out = SMOKE / 'S1/N06_mut_init'
    train('N06', 0, 1, out, extras=False, mutate='init')
    cm = g1_compare(out / 'units.npz', out / 'rows.csv', 'SN06', 0, 1)
    det['mutation_init'] = cm
    return result(ok, cm['state_maxabs'] > 0.0, **det)


def g1_full(runs):
    runs = Path(runs); det = {}; status = 'ANCHORED'; first = None
    for arm, anc in A.ANCHOR.items():
        for seed in (0, 1, 2):
            d = runs / f'{arm}_s{seed}'
            if not (d / 'provenance.json').exists():
                det[f'{arm}_s{seed}'] = 'missing'; status = 'CODE_MISMATCH' if status == 'ANCHORED' else status
                continue
            c = g1_compare(d / 'units.npz', d / 'rows.csv', anc, seed, 120)
            det[f'{arm}_s{seed}'] = c
            if c['first_state_mismatch'] == 1:
                status = 'CODE_MISMATCH'; first = first or (arm, seed, 1)
            elif c['first_state_mismatch'] is not None and status != 'CODE_MISMATCH':
                status = 'UNANCHORED'; first = first or (arm, seed, c['first_state_mismatch'])
            elif c['first_derived_mismatch'] is not None and status == 'ANCHORED':
                status = 'MEASURE_MISMATCH'; first = first or (arm, seed, c['first_derived_mismatch'])
            elif c['arrays_compared'] != c['expected'] and status == 'ANCHORED':
                status = 'CODE_MISMATCH'; first = first or (arm, seed, 'count')
    res = dict(status=status, first_mismatch=first, detail=det)
    (runs.parent / 'g1_full.json').write_text(json.dumps(res, indent=1, default=str))
    return res


# ------------------------------------------------------------------ S2 activation shape
def _grid(a, th):
    base = torch.linspace(-6 / a, 6 / a, 4001, dtype=torch.float64)
    t = A.TH_RAD[th]
    zeros = [(-math.pi / 2 - t + 2 * k * math.pi) / (2 * a) for k in range(-3, 4)]
    return torch.cat([base, torch.tensor(zeros + [0.0], dtype=torch.float64)]).sort().values


def _dphi_err(phi, dphi, z):
    zg = z.clone().requires_grad_(True)
    d1 = torch.autograd.grad(phi(zg).sum(), zg)[0]
    err = (dphi(z) - d1).abs()
    # rounding bound: each of phi' (autograd) and the analytic form uses <= 6 float64 ops on
    # terms of size <= 1 + |2 a z|; gamma_6 = 6 eps / (1 - 6 eps).
    g6 = 6 * EPS64 / (1 - 6 * EPS64)
    bound = 2 * g6 * (2.0 + (2 * A.ALPHA * z).abs())
    return float((err - bound).max()), float(err.max())


def S2():
    det, ok = {}, True
    exp0 = {0: 1.0, 1: 2.0, -1: 0.0}
    for th in (0, 1, -1):
        for dt in (torch.float32, torch.float64):
            f = A.PhaseSnake(th)
            z0 = torch.zeros(1, dtype=dt)
            ok &= float(f.phi(z0)) == 0.0 and float(f.dphi(z0)) == exp0[th]
            ad = A.AdaptivePhaseSnake(th); ad.V = [v.to(dt) for v in ad.V]
            z0a = torch.zeros(100, dtype=dt)
            ok &= bool((ad.phi(z0a, 0) == 0).all()) and bool((ad.dphi(z0a, 0) == exp0[th]).all())
        f = A.PhaseSnake(th)
        excess, err = _dphi_err(f.phi, f.dphi, _grid(A.ALPHA, th))
        det[f'th{th}_autograd_err'] = err; ok &= excess <= 0
    # mutations (analytic mismatch of a wrong phase's phi' with the right phi: 2|sin(dth/2)|)
    z = _grid(A.ALPHA, 1)
    peak = A.PhaseSnake(1); normal = A.PhaseSnake(0); valley = A.PhaseSnake(-1)
    m_a = float(normal.dphi(torch.zeros(1, dtype=torch.float64))) != 2.0           # peak falls back to normal
    zg = z.clone().requires_grad_(True)
    dpk = torch.autograd.grad(peak.phi(zg).sum(), zg)[0]
    m_b = float((valley.dphi(z) - dpk).abs().max())                               # sign flip: analytic 2
    m_c = float((normal.dphi(z) - dpk).abs().max())                               # other phase: analytic sqrt2
    det.update(mut_peak_to_normal_detected=m_a, mut_signflip_maxdiff=m_b, mut_otherphase_maxdiff=m_c)
    mut = m_a and m_b >= 0.5 * 2.0 and m_c >= 0.5 * math.sqrt(2)
    return result(ok, mut, **det)


# ------------------------------------------------------------------ S3 q=0, theta=0 identity
def _fwd_bwd(act, p, x, y):
    pp = [q.clone().requires_grad_(True) for q in p]
    out = H.forward(pp, x, act)
    g = torch.autograd.grad(torch.nn.functional.cross_entropy(out[4], y), pp)
    return [o.detach() for o in out], g


def S3():
    torch.manual_seed(0)
    p = H.init_params(3, torch.device('cpu'))
    with torch.no_grad():
        for q in p:
            q.mul_(4.0)
    x = torch.rand(64, 784); y = torch.randint(0, 10, (64,))
    det, ok = {}, True
    for mine, host in ((A.PhaseSnake(0), H.Activation('SNx', 'snake', .6)), (A.OffsetLeaky(.1), H.ARMS['LR'])):
        o1, g1 = _fwd_bwd(mine, p, x, y); o2, g2 = _fwd_bwd(host, p, x, y)
        e = all(torch.equal(a, b) for a, b in zip(o1, o2)) and all(torch.equal(a, b) for a, b in zip(g1, g2))
        det[mine.name] = e; ok &= e
    mine, host = A.AdaptivePhaseSnake(0), H.AdaptiveSnake(.6, .01, 'cpu')
    Vr = [torch.rand(100) * 5 + 0.1, torch.rand(100) * 5 + 0.1]
    mine.V = [v.clone() for v in Vr]; host.V = [v.clone() for v in Vr]
    o1, g1 = _fwd_bwd(mine, p, x, y); o2, g2 = _fwd_bwd(host, p, x, y)
    mine.update(o1[0], o1[2]); host.update(o2[0], o2[2])
    e = all(torch.equal(a, b) for a, b in zip(o1, o2)) and all(torch.equal(a, b) for a, b in zip(g1, g2)) \
        and all(torch.equal(a, b) for a, b in zip(mine.V, host.V))
    inst = isinstance(mine, H.AdaptiveSnake)
    det.update(adaptive_equal=e, isinstance_host=inst); ok &= e and inst

    class Bump(A.PhaseSnake):
        def phi(self, z):
            return z + torch.sin(self.param * z + 1e-6) ** 2 / self.param
    o3, _ = _fwd_bwd(Bump(0), p, x, y); o4, _ = _fwd_bwd(H.Activation('SNx', 'snake', .6), p, x, y)
    m1 = not all(torch.equal(a, b) for a, b in zip(o3, o4))
    from src import pmnist_0905 as P5
    bad = P5.AdaptiveSnake(.6, .01, 'cpu')
    m2 = not isinstance(bad, H.AdaptiveSnake)
    # consequence: the box-B loop skips the V update for the wrongly based class
    pp = [q.clone().requires_grad_(True) for q in H.init_params(0, torch.device('cpu'))]
    adam = ([torch.zeros_like(q) for q in pp], [torch.zeros_like(q) for q in pp], [0])
    V0 = [v.clone() for v in bad.V]
    try:
        SP.adam_task.__globals__['STEPS'] = 1
        SP.adam_task(pp, bad, adam, x[:16].repeat(1, 1), y[:16], task=0)
        v_unchanged = all(torch.equal(a, b) for a, b in zip(V0, bad.V))
    except TypeError:
        v_unchanged = True
    finally:
        SP.adam_task.__globals__['STEPS'] = 625
    det.update(mut_bump_detected=m1, mut_wrong_base_isinstance_false=m2, mut_wrong_base_V_not_updated=v_unchanged)
    return result(ok, m1 and m2 and v_unchanged, **det)


# ------------------------------------------------------------------ S4 identity (float64)
def _fwd64(p, x, phi1, phi2):
    W1, b1, W2, b2, W3, b3 = p
    return phi2(phi1(x @ W1.T + b1) @ W2.T + b2) @ W3.T + b3


def _tol_fwd(p, x, extra=0.0):
    """float64 rounding-propagation bound: per layer <= gamma_n * (sum of |terms|) amplified by
    ||W||_inf * max|phi'| (=2) of later layers; n = 16 ops per scalar."""
    g = 16 * EPS64 / (1 - 16 * EPS64)
    W1, b1, W2, b2, W3, b3 = [q.double() for q in p]
    s1 = float((x.abs() @ W1.abs().T + b1.abs()).max()) + extra
    s2 = float((s1 * 2 + extra + 1) * W2.abs().sum(1).max() + b2.abs().max()) + extra
    s3 = float((s2 * 2 + extra + 1) * W3.abs().sum(1).max() + b3.abs().max())
    L2 = 2 * float(W2.abs().sum(1).max()); L3 = 2 * float(W3.abs().sum(1).max())
    return g * (s1 * L2 * L3 + s2 * L3 + s3)


def S4():
    x = H.Mnist(torch.device('cpu')).test_x[:256].double() if SP.T.data_dir() else None
    states = {}
    p0 = [q.detach().double() for q in H.init_params(0, torch.device('cpu'))]
    states['theta0'] = p0
    states['x4'] = [q * 4 for q in p0]
    snap = SMOKE / 'S7/N06_extras/snap_t003.pt'
    if snap.exists():
        states['smoke_t3'] = [q.double() for q in torch.load(snap)['p']]
    a = A.ALPHA
    psi0 = lambda z: z + torch.sin(a * z) ** 2 / a
    phase = {1: lambda z: z + torch.sin(2 * a * z) / (2 * a), -1: lambda z: z - torch.sin(2 * a * z) / (2 * a)}
    cases = [(1, A.S_P, A.K_P), (1, A.S_P1, A.K_P1), (-1, A.S_V, A.K_V)]
    det, ok = {}, True
    mut_ok = True
    for name, p in states.items():
        for th, s, K in cases:
            ps = [q.clone().requires_grad_(True) for q in p]
            pn = [q.clone() for q in p]; pn[1] = pn[1] + s; pn[3] = pn[3] + s
            pn = [q.requires_grad_(True) for q in pn]
            y = torch.arange(256) % 10
            lp = _fwd64(ps, x, phase[th], phase[th]); ln = _fwd64(pn, x, lambda z: psi0(z) + K, lambda z: psi0(z) + K)
            gp = torch.autograd.grad(torch.nn.functional.cross_entropy(lp, y), ps)
            gn = torch.autograd.grad(torch.nn.functional.cross_entropy(ln, y), pn)
            tol = _tol_fwd(p, x, extra=abs(s) + abs(K))
            dl = float((lp - ln).abs().max()); dg = max(float((u - v).abs().max()) for u, v in zip(gp, gn))
            # gradient bound: the loss gradient is 1-Lipschitz-scaled in the logits; propagate the
            # logit bound through the same layer amplification once more.
            tolg = tol * 2 * float(p[4].abs().sum(1).max() + 1) * float(p[2].abs().sum(1).max() + 1) * 4
            det[f'{name}_th{th}_s{s:.3f}'] = dict(dlogit=dl, tol=tol, dgrad=dg, tolg=tolg)
            ok &= dl <= tol and dg <= tolg
            # mutation: drop K in layer 2's activation -> logits shift by K * W3 1 exactly
            lm = _fwd64(pn, x, lambda z: psi0(z) + K, psi0)
            dev = abs(K) * float(p[4].sum(1).abs().max())
            dm = float((lp - lm).abs().max())
            pns = [q.clone() for q in p]; pns[1] = pns[1] - s; pns[3] = pns[3] - s
            ls = _fwd64(pns, x, lambda z: psi0(z) + K, lambda z: psi0(z) + K)
            dsf = float((lp - ls).abs().max())
            det[f'{name}_th{th}_s{s:.3f}'].update(mut_dropK=dm, mut_dropK_analytic=dev, mut_sflip=dsf)
            mut_ok &= dm >= 0.5 * dev and dsf > tol
    return result(ok, mut_ok, **det)


# ------------------------------------------------------------------ S5 init map
def _fwd64_act(p, x, act, layer_ada=False):
    W1, b1, W2, b2, W3, b3 = [q.detach().double() for q in p]
    if isinstance(act, H.AdaptiveSnake):
        f1 = lambda z: act.phi(z, 0); f2 = lambda z: act.phi(z, 1)
    else:
        f1 = f2 = act.phi
    return f2(f1(x @ W1.T + b1) @ W2.T + b2) @ W3.T + b3


def _tol_map(pm, pr, x):
    """half-ulp of each cast bias, propagated with ||W||_inf * max|phi'| (<= 2), plus float64 rounding."""
    W2, W3 = pm[2].double(), pm[4].double()
    hu = lambda b: float((torch.nextafter(b.abs(), torch.tensor(float('inf'))) - b.abs()).max()) / 2
    L2 = 2 * float(W2.abs().sum(1).max()); L3 = 2 * float(W3.abs().sum(1).max())
    d1 = hu(pm[1]) + hu(pr[1]); d2 = L2 * d1 + hu(pm[3]) + hu(pr[3]); d3 = L3 * d2 + hu(pm[5]) + hu(pr[5])
    return d3 + 2 * _tol_fwd(pm, x, extra=6.0)


def S5():
    x = H.Mnist(torch.device('cpu')).test_x[:256].double()
    pairs = [('P06c', 'N06'), ('P06c_k1', 'N06'), ('V06c', 'N06'), ('P06i', 'P06'), ('SNAi_P', 'SNAP'),
             ('V06i', 'V06'), ('SNAi_V', 'SNAV'), ('LR_qKp', 'LR'), ('LR_qKpn', 'LR')]
    det, ok, mut_ok = {}, True, True
    for seed in (0, 1):
        for arm, ref in pairs:
            pm = SP.theta0(arm, seed); pr = SP.theta0(ref, seed)
            am, ar = A.make_act(arm), A.make_act(ref)
            lm = _fwd64_act(pm, x, am); lr = _fwd64_act(pr, x, ar)
            tol = _tol_map(pm, pr, x); d = float((lm - lr).abs().max())
            det[f'{arm}_vs_{ref}_s{seed}'] = dict(d=d, tol=tol)
            ok &= d <= tol
            spec = A.ARM_TABLE[arm][1]
            q = spec[1] if spec[0] == 'comp' else spec[2]
            pm_b3 = SP.theta0(arm, seed, map_mutate='no_b3')
            dev = abs(q) * float(pm[4].double().sum(1).abs().max())
            db3 = float((_fwd64_act(pm_b3, x, A.make_act(arm)) - lr).abs().max())
            pm_2 = SP.theta0(arm, seed, map_mutate='double')
            d2 = float((_fwd64_act(pm_2, x, A.make_act(arm)) - lr).abs().max())
            det[f'{arm}_vs_{ref}_s{seed}'].update(mut_no_b3=db3, mut_no_b3_analytic=dev, mut_double=d2)
            mut_ok &= db3 >= 0.5 * dev and d2 > tol
    return result(ok, mut_ok, **det)


# ------------------------------------------------------------------ S6 fresh probe
def _task_tensors(mnist, seed, task):
    gp, gd, gb = H.stream('perm', seed), H.stream('data', seed), H.stream('batch', seed)
    for t in range(1, task + 1):
        perm = torch.randperm(784, generator=gp); idx = H.stratified_draw(mnist, gd)
        order = torch.randperm(H.TASK_EXAMPLES, generator=gb)
    return perm, mnist.train_x[idx][:, perm][order], mnist.train_y[idx][order]


def S6():
    det, ok, muts = {}, True, {}
    mnist = H.Mnist(torch.device('cpu'))
    for arm in ('N06', 'SNAV', 'P06c'):
        out = SMOKE / f'S6/{arm}'
        train(arm, 0, 1, out, fresh_tasks=[1], snap_tasks=[])
        r = rows_of(out)[0]
        e = r['fresh_param_sha'] == r['param_sha'] and r['correct_fresh'] == r['correct_seq']
        det[arm] = dict(fresh_eq_seq=e, correct=r['correct_seq'])
        ok &= e
    # the sequential task-1 end state rebuilt in-process (identical code) and the mutated probes
    for arm in ('N06', 'SNA'):
        seq_sha = rows_of(SMOKE / f'S6/{"N06" if arm == "N06" else "SNAV"}')[0]['param_sha'] if arm == 'N06' else None
        perm1, xs1, ys1 = _task_tensors(mnist, 0, 1)
        _, xs2, ys2 = _task_tensors(mnist, 0, 2)
        p = SP.theta0(arm, 0); act = A.make_act(arm)
        adam = ([torch.zeros_like(q) for q in p], [torch.zeros_like(q) for q in p], [0])
        SP.adam_task(p, act, adam, xs1, ys1, task=1)
        good = SP.param_sha(p)
        if seq_sha is not None:
            ok &= good == seq_sha

        def probe(start, a_state, xs, ys, V=None):
            pf = [q.detach().clone().requires_grad_(True) for q in start]
            af = A.make_act(arm)
            if V is not None:
                af.V = [v.clone() for v in V]
            SP.adam_task(pf, af, a_state, xs, ys, task=1)
            return SP.param_sha(pf)
        z = lambda P: ([torch.zeros_like(q) for q in P], [torch.zeros_like(q) for q in P], [0])
        p0 = SP.theta0(arm, 0)
        muts[f'{arm}_a_start_task1_end'] = probe(p, z(p), xs1, ys1) != good
        muts[f'{arm}_b_task2_perm'] = probe(p0, z(p0), xs2, ys2) != good
        carried = ([x.clone() for x in adam[0]], [x.clone() for x in adam[1]], [adam[2][0]])
        muts[f'{arm}_c_adam_carried'] = probe(p0, carried, xs1, ys1) != good
        if arm == 'SNA':
            muts['SNA_d_V_carried'] = probe(p0, z(p0), xs1, ys1, V=act.V) != good
    det['mutations'] = muts
    return result(ok, all(muts.values()), **det)


# ------------------------------------------------------------------ S7 non-invasive
def S7():
    det, ok = {}, True
    for arm in ('N06', 'SNAV'):
        a = SMOKE / f'S7/{arm}_extras'; b = SMOKE / f'S7/{arm}_plain'
        train(arm, 0, 5, a, fresh_tasks=[3], snap_tasks=[3])
        train(arm, 0, 5, b, extras=False)
        ra, rb = rows_of(a), rows_of(b)
        e = [x['param_sha'] == y['param_sha'] and x['acc'] == y['acc'] for x, y in zip(ra, rb)]
        det[arm] = dict(equal_per_task=e); ok &= all(e) and len(e) == 5
    m1 = SMOKE / 'S7/N06_mut_probe_draw'; m2 = SMOKE / 'S7/N06_mut_extra_w1'
    train('N06', 0, 5, m1, fresh_tasks=[3], snap_tasks=[], mutate='probe_draw')
    train('N06', 0, 2, m2, fresh_tasks=[], snap_tasks=[], mutate='extra_w1')
    rb = rows_of(SMOKE / 'S7/N06_plain')
    r1, r2 = rows_of(m1), rows_of(m2)
    d1 = [x['param_sha'] != y['param_sha'] for x, y in zip(r1, rb)]
    d2 = [x['param_sha'] != y['param_sha'] for x, y in zip(r2, rb)]
    det.update(mut_probe_draw_differs_per_task=d1, mut_extra_w1_differs_per_task=d2)
    # probe_draw is taken at t3 after training -> tasks 1-3 equal, 4-5 differ; extra_w1 from t1
    mut = d1[:3] == [False] * 3 and all(d1[3:]) and all(d2)
    return result(ok, mut, **det)


# ------------------------------------------------------------------ S8 stream independence
def _stream_hashes(mnist, seed, tasks, suffix=''):
    gp, gd, gb = H.stream('perm' + suffix, seed), H.stream('data', seed), H.stream('batch', seed)
    out = []
    for t in range(tasks):
        perm = torch.randperm(784, generator=gp); idx = H.stratified_draw(mnist, gd)
        order = torch.randperm(H.TASK_EXAMPLES, generator=gb)
        out.append(hashlib.sha256(perm.numpy().tobytes() + idx.numpy().tobytes() + order.numpy().tobytes()).hexdigest())
    return out


def S8():
    mnist = H.Mnist(torch.device('cpu'))
    provs = {}
    for d in list((SMOKE / 'S7').glob('*_plain')) + list((SMOKE / 'S7').glob('*_extras')) + list((SMOKE / 'S6').glob('*')):
        pj = d / 'provenance.json'
        if pj.exists():
            j = json.loads(pj.read_text()); provs[d.name] = (j['arm'], j['seed'], j['stream_sha256'])
    recomputed = _stream_hashes(mnist, 0, 5)
    same = all(h[:len(recomputed)] == recomputed[:len(h)] for _, s, h in provs.values() if s == 0)
    arms = {a for a, s, _ in provs.values()}
    s1 = _stream_hashes(mnist, 1, 5)
    distinct = all(a != b for a, b in zip(recomputed, s1))
    ok = same and len(arms) >= 3 and distinct
    m_role = _stream_hashes(mnist, 0, 3, suffix='N06') != recomputed[:3]
    m_seed = not all(a != b for a, b in zip(recomputed, _stream_hashes(mnist, 0, 5)))   # seed s+1 built from s
    return result(ok, m_role and m_seed, arms_compared=sorted(arms), same=same, distinct=distinct,
                  mut_role_detected=m_role, mut_seed_detected=m_seed)


# ------------------------------------------------------------------ S9 extra measurements
def S9():
    sys.path.insert(0, str(HERE))
    import verdict as V
    mnist = H.Mnist(torch.device('cpu'))
    det, ok = {}, True
    d = SMOKE / 'S7/SNAV_extras'
    snap = torch.load(d / 'snap_t003.pt')
    p = [q.clone() for q in snap['p']]
    act = A.make_act('SNAV'); act.V = [v.clone() for v in snap['V']]
    seed = 0
    pidx = torch.randperm(len(mnist.test_x), generator=H.stream('boundary_probe', seed))[:512]
    probe = C.Probe(mnist.test_x[pidx], mnist.test_y[pidx], float(mnist.train_x.mean()))
    perm, _, _ = _task_tensors(mnist, 0, 3)
    u = np.load(d / 'units.npz')
    # independent re-implementation (explicit float32 ops, then float64 statistics)
    with torch.no_grad():
        x = probe.px[:, perm]
        z1 = x @ p[0].T + p[1]
        a1 = act.phi(z1, 0)
        z2 = a1 @ p[2].T + p[3]
        for l, z, W, Wn in ((1, z1, p[0], p[2]), (2, z2, p[2], p[4])):
            al = act.alpha(l - 1).double(); zd = z.double()
            ref = dict(cnorm=(W.double() - W.double().mean(1, keepdim=True)).norm(dim=1), m=W.double().mean(1),
                       zcur=zd.mean(0), gbar=act.dphi(z, l - 1).double().mean(0), abar=act.phi(z, l - 1).double().mean(0),
                       outcol=Wn.double().norm(dim=0),
                       A=torch.abs(torch.exp(1j * (2 * al[None, :] * zd)).mean(0)))
            for k, v in ref.items():
                got = u[f'L{l}_{k}'][3]
                dd = float(np.abs(got - v.numpy()).max())
                # bound: mean of 512 float64 terms each O(1+|v|): 512 eps64 (1 + max|v|)
                bnd = 512 * EPS64 * (1 + float(v.abs().max()))
                det[f'L{l}_{k}'] = dict(d=dd, bound=bnd); ok &= dd <= bnd
    # verdict helpers against inline computations on the same units
    logN = V.logN_window(u['L1_cnorm'], 1, 3)
    inline = 0.5 * math.log(float((u['L1_cnorm'][1:4] ** 2).mean()))
    det['verdict_logN'] = dict(got=logN, inline=inline); ok &= abs(logN - inline) <= 64 * EPS64 * (1 + abs(inline))
    # mutations: W2 row shift in the run's code path (L2_m moves by exactly 1e-5), alpha coefficient
    ub = SP.unit_block(p, act, probe, perm); ubm = SP.unit_block(p, act, probe, perm, mut='w2row')
    ubA = SP.unit_block(p, act, probe, perm, mut='A_alpha')
    m1 = float(np.abs(ubm['L2_m'] - ub['L2_m']).max()) >= 0.5 * 1e-3
    m2 = float(np.abs(ubA['L1_A'] - ub['L1_A']).max()) > 1e-3
    det.update(mut_w2row=float(np.abs(ubm['L2_m'] - ub['L2_m']).max()), mut_A_alpha=float(np.abs(ubA['L1_A'] - ub['L1_A']).max()))
    return result(ok, m1 and m2, **det)


# ------------------------------------------------------------------ S15a divergence
def S15a():
    out = SMOKE / 'S15/N06_nan'
    if out.exists():
        shutil.rmtree(out)
    r = subprocess.run([sys.executable, '-m', 'src.snake_phase_mnist_0914', '--arm', 'N06', '--seed', '0', '--tasks', '3',
                        '--out', str(out), '--inject-nan', '2,10', '--force-probe-tasks', '1'],
                       cwd=ROOT, capture_output=True, text=True)
    prov = json.loads((out / 'provenance.json').read_text())
    rows = rows_of(out); u = np.load(out / 'units.npz')
    ok = r.returncode == 3 and prov['status'] == 'DIVERGED' and prov['diverged_at'] == [2, 10] \
        and len(rows) == 1 and u['L1_cnorm'].shape[0] == 2
    # mutation: a copy that exits immediately on divergence without flushing -> task-1 row missing
    src = (ROOT / 'src/snake_phase_mnist_0914.py').read_text()
    a1 = "fh.write(','.join(str(r.get(k, '')) for k in header) + '\\n'); fh.flush()"
    a2 = "    except Diverged as e:\n"
    assert src.count(a1) == 1 and src.count(a2) == 1
    mut_src = src.replace(a1, "fh.write(','.join(str(r.get(k, '')) for k in header) + '\\n')").replace(
        a2, "    except Diverged as e:\n        os._exit(3)\n")
    tmp = ROOT / 'src/_mut_s15_snake_phase_0914.py'
    tmp.write_text(mut_src)
    outm = SMOKE / 'S15/N06_nan_mut'
    if outm.exists():
        shutil.rmtree(outm)
    try:
        subprocess.run([sys.executable, '-m', 'src._mut_s15_snake_phase_0914', '--arm', 'N06', '--seed', '0', '--tasks', '3',
                        '--out', str(outm), '--inject-nan', '2,10', '--no-extras'], cwd=ROOT, capture_output=True, text=True)
    finally:
        tmp.unlink()
    rp = outm / 'rows.csv'
    mut = (not rp.exists()) or len(rp.read_text().strip().splitlines()) <= 1 or not (outm / 'provenance.json').exists()
    return result(ok, mut, returncode=r.returncode, status=prov['status'], diverged_at=prov['diverged_at'],
                  rows=len(rows), units_t=int(u['L1_cnorm'].shape[0]), mutation_rows_missing=mut)


def ingest(name, path):
    p = HERE / path
    if not p.exists():
        return dict(pass_=False, check_ok=False, mutation_detected=False, detail=f'{path} missing')
    j = json.loads(p.read_text())
    muts = j.get('mutations', [])
    md = bool(muts) and all(m.get('detected') for m in muts)
    return dict(pass_=bool(j.get('all_pass')) and md, check_ok=bool(j.get('all_pass')), mutation_detected=md,
                detail=dict(n_cases=len(j.get('cases', [])), n_mutations=len(muts)))


ORDER = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S15a']
FUN = dict(S1=S1, S2=S2, S3=S3, S4=S4, S5=S5, S6=S6, S7=S7, S8=S8, S9=S9, S15a=S15a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default=None)
    ap.add_argument('--g1-full', default=None)
    a = ap.parse_args()
    H.setup('cpu'); SP.T.data_dir()
    if a.g1_full:
        res = g1_full(a.g1_full)
        print(json.dumps(dict(status=res['status'], first=res['first_mismatch']), default=str))
        return
    A.check_config_constants()
    path = HERE / 'checks.json'
    prev = json.loads(path.read_text())['checks'] if path.exists() else {}
    names = a.only.split(',') if a.only else ORDER
    out = dict(prev)
    for n in names:
        t0 = time.time()
        try:
            out[n] = FUN[n]()
        except Exception as e:
            import traceback
            out[n] = dict(pass_=False, check_ok=False, mutation_detected=False, detail=traceback.format_exc()[-2000:])
        out[n]['secs'] = round(time.time() - t0, 1)
        print(n, 'pass' if out[n]['pass_'] else 'FAIL', out[n]['secs'], 's', flush=True)
    out['S14'] = ingest('S14', 'verdict_selftest.json')
    out['S15b'] = ingest('S15b', 'launch_selftest.json')
    allp = all(out.get(n, {}).get('pass_') for n in ORDER + ['S14', 'S15b'])
    path.write_text(json.dumps(dict(all_pass=allp, checks=out), indent=1, default=str))
    print('all_pass', allp)


if __name__ == '__main__':
    main()
