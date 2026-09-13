"""Width/sinking clamp experiment (spec_width_sink_clamp_0909, 追補 1/2/3).

Three clamps on the orthogonal decomposition of a first-layer row

    W_i = mbar*1 + (m_i - mbar)*1 + Wt_i          (Wt_i = W_i - m_i*1)

applied in place after every Adam update from task CLAMP_FROM onwards:

    wclamp   ||Wt_i|| pinned to its t20 value      (m untouched)
    mclamp   sd_i(m_i) pinned to its t20 value     (mbar and Wt untouched)
    dclamp   mbar pinned to its t20 value          (sd_i(m_i) and Wt untouched)

`ref` is unclamped and must reproduce the committed trajectories (G1).
"""
from pathlib import Path
import argparse, concurrent.futures, hashlib, json, subprocess, sys, time
import numpy as np
import torch
from src import boundary_gradient_0908 as G
from src import elu_growth_0909 as EG

H = G.H
ROOT = G.ROOT
OUT = ROOT / 'results/width_sink_clamp_0909'
SMOKE = ROOT / 'results/_smoke_width_sink_clamp'
ARMS = [('LR', 'none'), ('SNA', 'none'), ('ELU1', 'none')]
CLAMPS = ['ref', 'wclamp', 'mclamp', 'dclamp']
MEAS = [20, 300, 625]
NREF = 8
REF_SEED = 20260909
TASKS = 100
CLAMP_FROM = 21
CAP = 2400.
EPS32 = float(torch.finfo(torch.float32).eps)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def exact(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(exact(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(exact(x, y) for x, y in zip(a, b))
    return a == b


def make_act(arm):
    if arm == 'SNA':
        return H.AdaptiveSnake(.6, .01, 'cpu')
    return EG.ELU(1.0) if arm == 'ELU1' else H.ARMS[arm]


def refperms():
    g = torch.Generator().manual_seed(REF_SEED)
    return [torch.randperm(784, generator=g) for _ in range(NREF)]


# ------------------------------------------------------------------ clamps
def rows_of(W):
    """float64 row split: (Wd, m column (n,1), Wt (n,d))."""
    Wd = W.detach().double()
    m = Wd.mean(1, keepdim=True)
    return Wd, m, Wd - m


def baselines(W):
    Wd, m, Wt = rows_of(W)
    return dict(c=Wt.norm(dim=1, keepdim=True),
                sigma_m=float(m.std(unbiased=False)),
                mbar0=float(m.mean()))


def clamped_W(kind, Wd, m, Wt, base, mut=None):
    """The clamped float64 weight matrix.  `mut` selects a mutation control."""
    if kind == 'wclamp':
        s = base['c'] / Wt.norm(dim=1, keepdim=True)
        if mut == 'rowscale':          # C7 control: scale the whole row, not just Wt
            return (m + Wt) * s
        return m + Wt * s
    if kind == 'mclamp':
        mb = m.mean()
        lam = base['sigma_m'] / m.std(unbiased=False)
        if mut == 'inv':               # C4 control
            lam = 1. / lam
        if mut == 'rowscale':          # C7 control: Wt dragged along
            return (mb + lam * (m - mb)) + Wt * lam
        return (mb + lam * (m - mb)) + Wt
    if kind == 'dclamp':
        sh = base['mbar0'] - m.mean()
        if mut == 'p90':               # C5 control: leave a tenth of the drift in
            sh = sh * .90
        if mut == 'rowscale':          # C7 control: multiplicative instead of additive
            return Wd * (base['mbar0'] / m.mean())
        return m + sh + Wt
    raise ValueError(kind)


def apply_clamp(p, kind, base, ck, mut=None):
    """In place on p[0]; updates running check maxima in `ck`."""
    Wd, m, Wt = rows_of(p[0])
    cn0 = Wt.norm(dim=1)
    sd0 = float(m.std(unbiased=False))
    mb0 = float(m.mean())
    tail = [q.detach().clone() for q in p[1:]]
    # Provable float32 bound on how much a row mean can move purely from rounding
    # the reassembled row: |mean_j round32(x_j) - mean_j x_j| <= eps32 * mean_j|x_j|.
    # Tolerances are floored by this rather than by a guessed constant, so they
    # track the weight scale as ||W|| grows over the run.
    ck['f32_rowmean_bound'] = max(ck.get('f32_rowmean_bound', 0.),
                                  EPS32 * float(Wd.abs().mean(1).max()))
    New = clamped_W(kind, Wd, m, Wt, base, mut)
    p[0].data.copy_(New.float())
    Wd1, m1, Wt1 = rows_of(p[0])
    cn1 = Wt1.norm(dim=1)

    def up(k, v):
        ck[k] = max(ck.get(k, 0.), float(v))

    if kind == 'wclamp':
        up('c3_m_absdiff', (m1 - m).abs().max())                       # C7: m untouched
    elif kind == 'mclamp':
        up('c4_mbar_absdiff', abs(float(m1.mean()) - mb0))             # mbar untouched
        up('c4_cnorm_rel', ((cn1 - cn0).abs() / cn0).max())            # C7: Wt untouched
    else:
        up('c5_sd_rel', abs(float(m1.std(unbiased=False)) - sd0) / sd0)  # sd untouched
        up('c5_cnorm_rel', ((cn1 - cn0).abs() / cn0).max())            # C7: Wt untouched
    up('clamp_tail_absdiff', max(float((a - b).abs().max()) for a, b in zip(p[1:], tail)))
    return dict(cnorm_pre=cn0, sd_pre=sd0, mbar_pre=mb0)


def verify_clamped(p, kind, base, ck):
    """Does the CURRENT state meet the clamp target?  Runs every step, clamp applied
    or not, so a skipped step is caught instead of being repaired by the next one."""
    Wd, m, Wt = rows_of(p[0])

    def up(k, v):
        ck[k] = max(ck.get(k, 0.), float(v))

    if kind == 'wclamp':
        c = base['c'].squeeze(1)
        up('c3_cnorm_rel', ((Wt.norm(dim=1) - c).abs() / c).max())
    elif kind == 'mclamp':
        up('c4_sd_rel', abs(float(m.std(unbiased=False)) - base['sigma_m']) / base['sigma_m'])
    else:
        up('c5_mbar_abs', abs(float(m.mean()) - base['mbar0']))


# ------------------------------------------------------------- measurement
class Probe:
    """Fixed probe plus the permutation-invariant pieces of the decomposition."""

    def __init__(self, px, py, mu):
        self.px, self.py, self.mu = px, py, mu
        self.refs = refperms()
        self.xr = [px[:, r].double() for r in self.refs]
        self.Sx = px.double().sum(1)                 # 1^T x is permutation invariant
        self.Sbar = float(self.Sx.mean())
        self.Sxc = self.Sx - self.Sx.mean()
        self.varSx = float(self.Sx.var(unbiased=False))
        self.xbar_r = [x.mean(0) for x in self.xr]


def measure(p, act, probe, perm, want_acc=False, mnist=None, ck=None, mut_decomp=False):
    """One measurement row + the per-unit arrays.  All maths in float64."""
    with torch.no_grad():
        W = p[0].double()
        b = p[1].double()
        m = W.mean(1)
        Wt = W - m[:, None]
        n = probe.px.shape[0]

        z = probe.px[:, perm].double() @ W.T + b
        zm = z.mean(0)
        zv = z.var(0, unbiased=False)

        zim = torch.zeros_like(m)
        ziv = torch.zeros_like(m)
        off = torch.zeros_like(m)
        cross = torch.zeros_like(m)
        wid = torch.zeros_like(m)
        pos_inv = 0.
        md = m * 1.01 if mut_decomp else m           # C6 mutation control
        for x, xbar in zip(probe.xr, probe.xbar_r):
            zr = x @ W.T + b
            # u is always built from the TRUE m, so scaling md really does break the
            # identity below.  (Deriving u from md would make the control vacuous:
            # md*Sx + u would collapse back to zr - b for any md.)
            u = zr - b - probe.Sx[:, None] * m[None, :]        # = x @ Wt^T, exactly
            zim += zr.mean(0) / NREF
            ziv += zr.var(0, unbiased=False) / NREF
            off += md ** 2 * probe.varSx / NREF
            cross += 2 * md * ((probe.Sxc[:, None] * u).sum(0) / n) / NREF
            wid += u.var(0, unbiased=False) / NREF
            pos_inv += float((zr > 0).double().mean()) / NREF
            if ck is not None:
                # C6a per-perm variance identity, C6b' per-perm mean identity (追補 3)
                lhs = zr.var(0, unbiased=False)
                rhs = md ** 2 * probe.varSx + 2 * md * ((probe.Sxc[:, None] * u).sum(0) / n) + u.var(0, unbiased=False)
                ck['c6_var_rel'] = max(ck.get('c6_var_rel', 0.), float(((lhs - rhs).abs() / lhs).max()))
                pred = md * probe.Sbar + b + Wt @ xbar
                ck['c6_mean_abs'] = max(ck.get('c6_mean_abs', 0.), float((zr.mean(0) - pred).abs().max()))

        star = W.sum(1) * probe.mu + b
        gate = act.dphi(z.float(), 0).double() if isinstance(act, H.AdaptiveSnake) else act.dphi(z.float()).double()
        row = dict(zbar_cur=float(zm.mean()), sigma_cur=float(zv.mean().sqrt()),
                   between_cur=float(zm.var(unbiased=False)), pos_frac=float((z > 0).double().mean()),
                   zbar_inv=float(zim.mean()), sigma_inv=float(ziv.mean().sqrt()),
                   between_inv=float(zim.var(unbiased=False)), pos_frac_inv=pos_inv,
                   var_off=float(off.mean()), var_wid=float(wid.mean()), var_cross=float(cross.mean()),
                   rowmean=float(m.mean()), rowmean_sd=float(m.std(unbiased=False)),
                   cnorm=float(Wt.norm(dim=1).mean()), bias=float(b.mean()),
                   bias_sd=float(b.std(unbiased=False)), star=float(star.mean()),
                   star_sd=float(star.std(unbiased=False)),
                   eps_max=float((zim - (m * probe.Sbar + b)).abs().max()),
                   w2col=float(p[2].double().norm(dim=0).mean()), gate_mean=float(gate.mean()))
        row['ce_probe'] = float(torch.nn.functional.cross_entropy(
            H.forward(p, probe.px[:, perm], act)[4], probe.py))
        if isinstance(act, H.AdaptiveSnake):
            row['alpha_med'] = float(act.alpha(0).median())
        if want_acc:
            row['acc'] = float((H.forward(p, mnist.test_x[:, perm], act)[4].argmax(1) == mnist.test_y).float().mean())
        units = dict(zbar_i=zim.numpy().copy(), sd_i=ziv.sqrt().numpy().copy(),
                     cnorm_i=Wt.norm(dim=1).numpy().copy(), m_i=m.numpy().copy(),
                     bias_i=b.numpy().copy(), star_i=star.numpy().copy())
    return row, units


def decompose_crosscheck(p, probe):
    """Agreement between the fast decomposition and the committed reference one."""
    from src.width_depth_intervention_0909 import decompose as ref_decompose
    W = p[0].detach().double()
    m = W.mean(1)
    Wt = W - m[:, None]
    worst = 0.
    for x in probe.xr:
        o, c, w = ref_decompose(W, x)
        o, c, w = o.detach(), c.detach(), w.detach()
        u = x @ Wt.T
        fast_o = m ** 2 * probe.varSx
        fast_c = 2 * m * ((probe.Sxc[:, None] * u).sum(0) / x.shape[0])
        fast_w = u.var(0, unbiased=False)
        worst = max(worst, float((o - fast_o).abs().max()), float((c - fast_c).abs().max()),
                    float((w - fast_w).abs().max()))
    return worst


# ------------------------------------------------------------------ engine
def train_tasks(p, act, adam, gens, mnist, probe, t_from, t_to, clamp, base, rows, units, ck,
                tag, mut=None, measure_mut=False, skip_one_clamp_at=None):
    gp, gd, gb = gens
    for task in range(t_from, t_to + 1):
        perm = torch.randperm(784, generator=gp)
        idx = H.stratified_draw(mnist, gd)
        order = torch.randperm(H.TASK_EXAMPLES, generator=gb)
        xs = mnist.train_x[idx][:, perm][order]
        ys = mnist.train_y[idx][order]
        for step in range(1, 626):
            xb = xs[(step - 1) * 16:step * 16]
            yb = ys[(step - 1) * 16:step * 16]
            out = H.forward(p, xb, act)
            gr = torch.autograd.grad(torch.nn.functional.cross_entropy(out[4], yb), p)
            with torch.no_grad():
                m_, v_, tc = adam
                tc[0] += 1
                c1 = 1 - .9 ** tc[0]
                c2 = 1 - .999 ** tc[0]
                for q, g, mi, vi in zip(p, gr, m_, v_):
                    mi.mul_(.9).add_(g, alpha=1 - .9)
                    vi.mul_(.999).addcmul_(g, g, value=1 - .999)
                    q -= .001 * (mi / c1) / ((vi / c2).sqrt() + 1e-8)
                if isinstance(act, H.AdaptiveSnake):
                    act.update(out[0], out[2])
                if clamp != 'ref':
                    if skip_one_clamp_at is not None and (task, step) == skip_one_clamp_at:
                        pass                              # C3/C4/C5 control: miss exactly one step
                    else:
                        pre = apply_clamp(p, clamp, base, ck, mut)
                        if clamp == 'wclamp':
                            ck['pressure'] = ck.get('pressure', 0.) + float((pre['cnorm_pre'] / base['c'].squeeze(1)).log().mean())
                        elif clamp == 'mclamp':
                            ck['pressure'] = ck.get('pressure', 0.) + float(np.log(pre['sd_pre'] / base['sigma_m']))
                        else:
                            ck['pressure'] = ck.get('pressure', 0.) + (base['mbar0'] - pre['mbar_pre'])
            if step in MEAS:
                r, u = measure(p, act, probe, perm, step == 625, mnist, ck)
                r.update(task=task, step=step, clamp=clamp, pressure=ck.get('pressure', 0.))
                rows.append(r)
                if step == 625:
                    for k, v in u.items():
                        units[f'{clamp}_{k}_t{task}'] = v
                if measure_mut:
                    with torch.no_grad():
                        p[0].data.add_(1e-9)
    return p, act, adam


def snapshot(p, act, adam, gens):
    return dict(p=[q.detach().clone() for q in p],
                V=[v.clone() for v in act.V] if isinstance(act, H.AdaptiveSnake) else None,
                adam=([x.clone() for x in adam[0]], [x.clone() for x in adam[1]], adam[2][0]),
                gens=[g.get_state().clone() for g in gens])


def restore(snap, arm):
    p = [q.clone().requires_grad_(True) for q in snap['p']]
    act = make_act(arm)
    if snap['V'] is not None:
        act.V = [v.clone() for v in snap['V']]
    adam = ([x.clone() for x in snap['adam'][0]], [x.clone() for x in snap['adam'][1]], [snap['adam'][2]])
    gens = []
    for s in snap['gens']:
        g = torch.Generator()
        g.set_state(s.clone())
        gens.append(g)
    return p, act, adam, gens


def g1_check(arm, seed, tasks, ends, ck, prefix_units):
    """追補 A3: the ref continuation must reproduce the committed trajectories."""
    R = ROOT / 'results'
    worst_exact, worst_npz, worst_units = None, 0., 0.
    if arm in ('LR', 'SNA'):
        ok = True
        for tag, t in [('task1', 1), ('task100', 100)]:
            if t > tasks:
                continue
            cp = torch.load(R / 'boundary_groups_0908' / f'{arm}_none_s{seed}_{tag}.pt',
                            weights_only=False, map_location='cpu')
            ok = ok and exact(ends[t]['state'], cp['state']) and exact(ends[t]['adam'], cp['adam']) \
                and exact(ends[t]['perm'], cp['perm']) and exact(ends[t]['streams'], cp['streams'])
        worst_exact = bool(ok)
        ref = np.load(R / 'task20to100_0908' / f'{arm}_none_s{seed}.npz')
        for i, t in enumerate(ref['task']):
            if int(t) > tasks or int(t) not in ends:
                continue
            e = ends[int(t)]
            worst_npz = max(worst_npz, float(np.abs(e['zmean_cur'] - ref['zmean'][i]).max()),
                            float(np.abs(e['star_i'] - ref['star'][i]).max()),
                            float(np.abs(e['within_cur'] - ref['within'][i]).max()))
    eg = np.load(R / 'elu_growth_0909' / f'{arm}_none_s{seed}_units.npz') if arm != 'ELU1' \
        else np.load(R / 'elu_growth_0909' / f'ELU1_none_s{seed}_units.npz')
    for t in range(1, min(tasks, 100) + 1):
        if f'ref_zbar_i_t{t}' not in prefix_units:
            continue
        for k in ['zbar_i', 'sd_i', 'cnorm_i']:
            worst_units = max(worst_units, float(np.abs(prefix_units[f'ref_{k}_t{t}'] - eg[k][t - 1]).max()))
    ck.update(g1_exact=worst_exact, g1_npz_maxabs=worst_npz, g1_units_maxabs=worst_units)
    return worst_exact, worst_npz, worst_units


# -------------------------------------------------------------------- run
def _endinfo(p, act, adam, gens, perm, probe, capture_state):
    with torch.no_grad():
        W = p[0].double(); b = p[1].double()
        z = probe.px[:, perm].double() @ W.T + b
        info = dict(zmean_cur=z.mean(0).numpy().copy(), within_cur=z.var(0, unbiased=False).numpy().copy(),
                    star_i=(W.sum(1) * probe.mu + b).numpy().copy())
    if capture_state:
        info.update(state=dict(params=[q.detach().clone() for q in p],
                               V=[v.clone() for v in act.V] if isinstance(act, H.AdaptiveSnake) else [],
                               alpha=[act.alpha(i).clone() for i in range(2)] if isinstance(act, H.AdaptiveSnake) else []),
                    adam=([x.clone() for x in adam[0]], [x.clone() for x in adam[1]], [adam[2][0]]),
                    perm=perm.clone(),
                    streams={k: g.get_state() for k, g in zip(['perm', 'data', 'batch'], gens)})
    return info


def loop(p, act, adam, gens, mnist, probe, t_from, t_to, clamp, base, rows, units, ck,
         mut=None, measure_mut=False, skip_at=None, mut_decomp=False, ends=None, capture=()):
    gp, gd, gb = gens
    for task in range(t_from, t_to + 1):
        perm = torch.randperm(784, generator=gp)
        idx = H.stratified_draw(mnist, gd)
        order = torch.randperm(H.TASK_EXAMPLES, generator=gb)
        xs = mnist.train_x[idx][:, perm][order]
        ys = mnist.train_y[idx][order]
        for step in range(1, 626):
            out = H.forward(p, xs[(step - 1) * 16:step * 16], act)
            gr = torch.autograd.grad(torch.nn.functional.cross_entropy(out[4], ys[(step - 1) * 16:step * 16]), p)
            with torch.no_grad():
                m_, v_, tc = adam
                tc[0] += 1
                c1 = 1 - .9 ** tc[0]; c2 = 1 - .999 ** tc[0]
                for q, g, mi, vi in zip(p, gr, m_, v_):
                    mi.mul_(.9).add_(g, alpha=1 - .9)
                    vi.mul_(.999).addcmul_(g, g, value=1 - .999)
                    q -= .001 * (mi / c1) / ((vi / c2).sqrt() + 1e-8)
                if isinstance(act, H.AdaptiveSnake):
                    act.update(out[0], out[2])
                if clamp != 'ref':
                    pre = apply_clamp(p, clamp, base, ck, mut) if (task, step) != skip_at \
                        else dict(cnorm_pre=rows_of(p[0])[2].norm(dim=1),
                                  sd_pre=float(rows_of(p[0])[1].std(unbiased=False)),
                                  mbar_pre=float(rows_of(p[0])[1].mean()))
                    verify_clamped(p, clamp, base, ck)
                    if clamp == 'wclamp':
                        ck['pressure'] = ck.get('pressure', 0.) + float((pre['cnorm_pre'] / base['c'].squeeze(1)).log().mean())
                    elif clamp == 'mclamp':
                        ck['pressure'] = ck.get('pressure', 0.) + float(np.log(pre['sd_pre'] / base['sigma_m']))
                    else:
                        ck['pressure'] = ck.get('pressure', 0.) + (base['mbar0'] - pre['mbar_pre'])
            if step in MEAS and rows is not None:
                r, u = measure(p, act, probe, perm, step == 625, mnist, ck, mut_decomp)
                r.update(task=task, step=step, clamp=clamp, pressure=ck.get('pressure', 0.))
                rows.append(r)
                if step == 625:
                    for k, v in u.items():
                        units[f'{clamp}_{k}_t{task}'] = v
            if measure_mut and step in MEAS:
                with torch.no_grad():
                    p[0].data.add_(1e-9)
        if ends is not None:
            ends[task] = _endinfo(p, act, adam, gens, perm, probe, task in capture)
    return p, act, adam


def run(arm, iv, seed, tasks=TASKS, clamp_from=CLAMP_FROM, out=OUT, controls=True):
    torch.set_num_threads(1); H.setup('cpu')
    mnist = H.Mnist(torch.device('cpu'))
    out.mkdir(parents=True, exist_ok=True)
    prefix = f'{arm}_{iv}_s{seed}'
    pidx = torch.randperm(len(mnist.test_x), generator=H.stream('boundary_probe', seed))[:512]
    probe = Probe(mnist.test_x[pidx], mnist.test_y[pidx], float(mnist.train_x.mean()))
    t0 = time.monotonic()
    rows, units, ck = [], {}, {}

    p = H.init_params(seed, torch.device('cpu'))
    act = make_act(arm)
    adam = ([torch.zeros_like(q) for q in p], [torch.zeros_like(q) for q in p], [0])
    gens = [H.stream('perm', seed), H.stream('data', seed), H.stream('batch', seed)]
    ck['decompose_crosscheck'] = decompose_crosscheck(p, probe)
    ends = {}
    loop(p, act, adam, gens, mnist, probe, 1, clamp_from - 1, 'ref', None, rows, units, ck,
         ends=ends, capture=(1,))   # the shared prefix is labelled 'ref': it IS the ref series
    base = baselines(p[0])
    ck.update(base_sigma_m=base['sigma_m'], base_mbar0=base['mbar0'], base_cnorm_mean=float(base['c'].mean()))
    snap = snapshot(p, act, adam, gens)

    for clamp in CLAMPS:
        p2, act2, adam2, gens2 = restore(snap, arm)
        e2 = {} if clamp == 'ref' else None
        # `ck` is shared across the four branches so that every check is a running
        # maximum over the whole run.  `pressure` is an ACCUMULATOR, not a maximum,
        # so it has to be reset per branch and parked under its own key.
        ck['pressure'] = 0.
        loop(p2, act2, adam2, gens2, mnist, probe, clamp_from, tasks, clamp, base, rows, units, ck,
             ends=e2, capture=(tasks,) if clamp == 'ref' else ())
        ck[f'pressure_{clamp}'] = ck.pop('pressure', 0.)
        if clamp == 'ref':
            ends.update(e2)
        print('DONE', prefix, clamp, round(time.monotonic() - t0, 1), 's', flush=True)

    g1 = g1_check(arm, seed, tasks, ends, ck, units)
    if controls:
        ck.update(_controls(arm, seed, snap, base, mnist, probe, clamp_from, ck))
    _assert(ck)

    keys = []
    [keys.append(k) for r in rows for k in r if k not in keys]
    for r in rows:
        r.update(arm=arm, iv=iv, seed=seed)
    keys += ['arm', 'iv', 'seed']
    G.B.csvwrite(out / f'{prefix}_rows.csv', [{k: r.get(k) for k in keys} for r in rows])
    np.savez_compressed(out / f'{prefix}_units.npz', **units)
    meta = dict(arm=arm, iv=iv, seed=seed, tasks=tasks, clamp_from=clamp_from, clamps=CLAMPS,
                meas=MEAS, nref=NREF, ref_seed=REF_SEED, checks=ck, g1=list(map(str, g1)),
                spec_sha256=sha(ROOT / 'specs/spec_width_sink_clamp_0909.md'),
                code_sha256=sha(Path(__file__)), host_sha256=sha(Path(H.__file__)),
                data_sha256=mnist.sha256, wall_seconds=time.monotonic() - t0,
                scope='CPU training from init; ref reproduces the committed trajectory')
    (out / f'{prefix}_provenance.json').write_text(json.dumps(meta, indent=1, default=str))
    print('FINISHED', prefix, round(time.monotonic() - t0, 1), 's', flush=True)
    assert time.monotonic() - t0 < CAP, 'runtime cap'


# ---------------------------------------------------------------- controls
TOL = dict(c3_cnorm_rel=1e-6, c3_m_absdiff=1e-9, c4_sd_rel=1e-6, c4_mbar_absdiff=1e-9,
           c4_cnorm_rel=1e-6, c5_mbar_abs=1e-8, c5_sd_rel=1e-6, c5_cnorm_rel=1e-6,
           clamp_tail_absdiff=0., c6_var_rel=1e-9, c6_mean_abs=1e-9, decompose_crosscheck=1e-9)
# Absolute checks on a row mean are floored by the provable float32 rounding bound;
# relative checks by 10 x float32 epsilon.  Both track the weight scale.
FLOOR_OF = dict(c3_m_absdiff='f32_rowmean_bound', c4_mbar_absdiff='f32_rowmean_bound',
                c5_mbar_abs='f32_rowmean_bound')
REL_FLOOR = 10 * EPS32
REL_CHECKS = ('c3_cnorm_rel', 'c4_sd_rel', 'c4_cnorm_rel', 'c5_sd_rel', 'c5_cnorm_rel')
# (clamp, mutation, which check the mutation must blow up).  The bar a control has to
# clear is 100x that check's own tolerance: a control that only just exceeds the
# tolerance is not convincing evidence that the check discriminates.
CONTROL_MARGIN = 100.
CONTROLS = [('wclamp', 'skip', 'c3_cnorm_rel'), ('wclamp', 'rowscale', 'c3_m_absdiff'),
            ('mclamp', 'inv', 'c4_sd_rel'), ('mclamp', 'rowscale', 'c4_cnorm_rel'),
            ('dclamp', 'p90', 'c5_mbar_abs'), ('dclamp', 'rowscale', 'c5_cnorm_rel')]


def _controls(arm, seed, snap, base, mnist, probe, clamp_from, ck_main):
    """Every check gets a mutation that must actually break it (追補 A6)."""
    res = {}
    for clamp, mut, key in CONTROLS:
        floor = CONTROL_MARGIN * max(limit(ck_main, key), ck_main.get(key, 0.))
        p2, act2, adam2, gens2 = restore(snap, arm)
        ck2 = {}
        loop(p2, act2, adam2, gens2, mnist, probe, clamp_from, clamp_from, clamp, base, None, {}, ck2,
             mut=None if mut == 'skip' else mut,
             skip_at=(clamp_from, 300) if mut == 'skip' else None)
        got = ck2.get(key, 0.)
        res[f'ctl_{clamp}_{mut}_{key}'] = got
        assert got > floor, (f'vacuous control {clamp}/{mut}: {key} = {got:g} <= {floor:g}')
    # C6: scaling the coefficient (not u) must break both identities
    p2, act2, adam2, gens2 = restore(snap, arm)
    ck2 = {}
    measure(p2, act2, probe, probe.refs[0], False, mnist, ck2, mut_decomp=True)
    res['ctl_c6_var_rel'] = ck2['c6_var_rel']
    res['ctl_c6_mean_abs'] = ck2['c6_mean_abs']
    assert ck2['c6_var_rel'] > 1e-3 and ck2['c6_mean_abs'] > 1e-3, ('vacuous C6 control', ck2)
    # decompose cross-check control: 1.01 * m inside the reference implementation
    from src.width_depth_intervention_0909 import decompose as ref_decompose
    W = snap['p'][0].double()
    x = probe.xr[0]
    o0, _, _ = ref_decompose(W, x)
    o1, _, _ = ref_decompose(W * 1.0, x)
    res['ctl_decompose'] = float((o0 - (W.mean(1) * 1.01) ** 2 * probe.varSx).abs().max())
    assert res['ctl_decompose'] > 1e-6, 'vacuous decompose control'
    # G1 controls: a perturbed init, and a measurement that touches the weights
    for tag, kw in [('init', dict(mut_init=True)), ('measure', dict(measure_mut=True))]:
        p3 = H.init_params(seed, torch.device('cpu'))
        if kw.get('mut_init'):
            with torch.no_grad():
                p3[0].data[0, 0] += 1e-3
        act3 = make_act(arm)
        adam3 = ([torch.zeros_like(q) for q in p3], [torch.zeros_like(q) for q in p3], [0])
        gens3 = [H.stream('perm', seed), H.stream('data', seed), H.stream('batch', seed)]
        u3 = {}
        loop(p3, act3, adam3, gens3, mnist, probe, 1, 1, 'ref', None, [], u3, {},
             measure_mut=kw.get('measure_mut', False))
        eg = np.load(ROOT / 'results/elu_growth_0909' / f'{arm}_none_s{seed}_units.npz')
        d = max(float(np.abs(u3[f'ref_{k}_t1'] - eg[k][0]).max()) for k in ['cnorm_i', 'zbar_i', 'sd_i'])
        res[f'ctl_g1_{tag}'] = d
        assert d > CONTROL_MARGIN * 1e-10, f'vacuous G1 {tag} control: {d:g}'
    return res


def limit(ck, k):
    if k in REL_CHECKS:
        return max(TOL[k], REL_FLOOR)
    return max(TOL[k], ck.get(FLOOR_OF.get(k, ''), 0.))


def _assert(ck):
    bad = {k: (ck[k], limit(ck, k)) for k in TOL if k in ck and not ck[k] <= limit(ck, k)}
    assert not bad, ('check failed (value, limit)', bad)
    assert ck['g1_exact'] in (True, None), ('G1 exact-state reproduction failed', ck['g1_exact'])
    assert ck['g1_npz_maxabs'] <= 1e-10 and ck['g1_units_maxabs'] <= 1e-10, ('G1 reproduction failed', ck)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm'); ap.add_argument('--iv', default='none'); ap.add_argument('--seed', type=int)
    ap.add_argument('--all', action='store_true'); ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    if a.smoke:
        for arm, iv in ([(a.arm, a.iv)] if a.arm else ARMS):
            run(arm, iv, 0 if a.seed is None else a.seed, tasks=4, clamp_from=3, out=SMOKE)
        return
    if a.all:
        def job(j):
            subprocess.run([sys.executable, '-m', 'src.width_sink_clamp_0909',
                            '--arm', j[0], '--iv', j[1], '--seed', str(j[2])], check=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            for _ in pool.map(job, [(arm, iv, s) for arm, iv in ARMS for s in range(3)]):
                pass
        return
    run(a.arm, a.iv, a.seed)


if __name__ == '__main__':
    main()
