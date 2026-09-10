"""gate_shape_0911 / band_dial_0911: the per-(unit, input) gate on the fixed probe at
every task end, for 13 reference arms (spec_gate_shape_0911) and the band-dial arms
(spec_band_dial_0911).  Host module untouched.

The training step below is character-identical to the committed `C.loop` ('ref'
branch): Adam, then the AdaptiveSnake running variance.  Everything this file adds
is a no_grad read-out, and G1 (bit reproduction of the committed per-unit arrays)
is what proves the read-out never touches the trajectory.

Per task end, on the 512-image probe under the CURRENT permutation:
    G = phi'(z_i(x))  (512 x 100)
    gbar_i = mean_x G,  gvar_i = Var_x G,  off_i = P_x[G < THETA]  (+ 0.1 / 0.5 / |G| variants)
    q10_i, zcur_i, sdcur_i, and gcorr = corr(G at task start, G at task end)
plus the committed `C.measure` row / per-unit arrays and the test accuracy.
"""
from pathlib import Path
import argparse, concurrent.futures, json, math, subprocess, sys, time
import numpy as np
import torch
from src import width_sink_clamp_0909 as C
from src import elu_growth_0909 as EG
from src import valley_acts_0910 as VA
from src import transport_common_0910 as T

H = C.H
ROOT = C.ROOT
OUT = ROOT / 'results/gate_shape_0911'
SMOKE = ROOT / 'results/_smoke_gate_shape_0911'
TASKS = 120
THETA = 0.25
TIME_CAP = 900.
SPEC = ROOT / 'specs/spec_gate_shape_0911.md'
SPEC_DIAL = ROOT / 'specs/spec_band_dial_0911.md'


# ------------------------------------------------------------------ activations
class CELU:
    """phi = z (z>0), tau*(e^{z/tau}-1) (z<=0).  tau=1 is ELU1 with the same op order
    (z/1.0 and 1.0*x are exact), so CELU1 must reproduce elu_growth_0909's ELU1 bit for bit."""
    kind = 'celu'

    def __init__(self, tau):
        self.param = float(tau); self.name = f'CELU{tau}'

    def phi(self, z):
        return torch.where(z > 0, z, self.param * torch.expm1(z.clamp(max=0.) / self.param))

    def dphi(self, z):
        return torch.where(z > 0, torch.ones_like(z), torch.exp(z.clamp(max=0.) / self.param))


class ELUFloor:
    """ELU1 with leaky's floor: phi' = max(e^z, a) on z<=0.  phi is C^1 at z = ln a."""
    kind = 'elufloor'

    def __init__(self, a):
        self.param = float(a); self.lna = math.log(a); self.name = f'ELUF{a}'

    def phi(self, z):
        zc = z.clamp(max=0.)
        deep = self.param * (z - self.lna) + (self.param - 1.)
        return torch.where(z > 0, z, torch.where(z >= self.lna, torch.expm1(zc), deep))

    def dphi(self, z):
        return torch.where(z > 0, torch.ones_like(z),
                           torch.clamp(torch.exp(z.clamp(max=0.)), min=self.param))


MAIN_ARMS = ['R', 'LR03', 'LR', 'LR001', 'ELU1', 'ELU03', 'SN02', 'SN06', 'SN15', 'SNA', 'LIN', 'GELU', 'SILU']
DIAL_ARMS = ['CELU03', 'CELU1', 'CELU3', 'ELUF']


def make_act(arm):
    if arm == 'R':
        return H.Activation('R', 'relu')
    if arm == 'LR':
        return H.ARMS['LR']
    if arm == 'LR03':
        return H.Activation('LRx', 'leaky', .3)
    if arm == 'LR001':
        return H.Activation('LRx', 'leaky', .01)
    if arm == 'ELU1':
        return EG.ELU(1.0)
    if arm == 'ELU03':
        return EG.ELU(0.3)
    if arm == 'SN02':
        return H.Activation('SNx', 'snake', .2)
    if arm == 'SN06':
        return H.Activation('SNx', 'snake', .6)
    if arm == 'SN15':
        return H.Activation('SNx', 'snake', 1.5)
    if arm == 'SNA':
        return H.AdaptiveSnake(.6, .01, 'cpu')
    if arm == 'LIN':
        return H.Activation('LINx', 'linear')
    if arm == 'GELU':
        return VA.GELU()
    if arm == 'SILU':
        return VA.SiLU()
    if arm == 'CELU03':
        return CELU(0.3)
    if arm == 'CELU1':
        return CELU(1.0)
    if arm == 'CELU3':
        return CELU(3.0)
    if arm == 'ELUF':
        return ELUFloor(0.1)
    raise KeyError(arm)


def dphi_of(act, z):
    return act.dphi(z, 0) if isinstance(act, H.AdaptiveSnake) else act.dphi(z)


def check_dphi(arm):
    """G2: analytic phi' against autograd, plus a wrong-activation control that must be
    clearly detectable.  ELUF additionally: continuity of phi at ln a."""
    act = make_act(arm)
    z = torch.linspace(-6, 6, 241, dtype=torch.float64)
    if isinstance(act, H.AdaptiveSnake):          # per-unit alpha: evaluate on a (241, 100) grid
        z = z[:, None].expand(241, 100).contiguous()
    zg = z.clone().requires_grad_(True)
    ph = act.phi(zg, 0) if isinstance(act, H.AdaptiveSnake) else act.phi(zg)
    d1 = torch.autograd.grad(ph.sum(), zg)[0]
    e = float((dphi_of(act, z) - d1).abs().max())
    other = H.Activation('m', 'snake', .6) if arm not in ('SN06', 'SNA') else H.Activation('m', 'leaky', .1)
    m = float((other.dphi(z) - d1).abs().max())
    out = dict(g2_dphi=e, g2_dphi_mutctl=m)
    if arm == 'ELUF':
        a = act.param; lna = torch.tensor([act.lna], dtype=torch.float64)
        lo = act.phi(lna - 1e-9); hi = act.phi(lna + 1e-9)
        out['g2_eluf_continuity'] = float((lo - hi).abs().max())
    return out


# ------------------------------------------------------------------ gate block
def gate_block(p, act, probe, perm, mut_var=False):
    """Per-unit gate statistics on the probe under `perm`.  All maths in float64
    after the float32 forward (the same z the gradient sees)."""
    with torch.no_grad():
        z = probe.px[:, perm] @ p[0].detach().T + p[1].detach()
        g = dphi_of(act, z).double()
        gbar = g.mean(0)
        gvar = g.var(0, unbiased=False)
        if mut_var:                                   # G3 mutation control
            gvar = gvar * 1.01
        pooled = float(g.var(unbiased=False))
        ident = abs(pooled - (float(gvar.mean()) + float(gbar.var(unbiased=False))))
        off = (g < THETA).double().mean(0)
        offabs = (g.abs() < THETA).double().mean(0)
        off10 = (g < 0.1).double().mean(0)
        off50 = (g < 0.5).double().mean(0)
        q10 = torch.quantile(g, 0.1, dim=0)
        zd = z.double()
        zcur = zd.mean(0)
        sdcur = zd.std(0, unbiased=False)
        hard = g.abs().max(0).values < 1e-6
        row = dict(gbar=float(gbar.mean()), gvar_mean=float(gvar.mean()),
                   gbar_var=float(gbar.var(unbiased=False)), pooled_var=pooled, g3_ident=ident,
                   cov=float(off.mean()), cov_abs=float(offabs.mean()), cov10=float(off10.mean()),
                   cov50=float(off50.mean()), q10_med=float(q10.median()), hard_dead=int(hard.sum()),
                   gmin=float(g.min()), gmax=float(g.max()))
        units = dict(gbar_i=gbar.numpy().copy(), gvar_i=gvar.numpy().copy(), off_i=off.numpy().copy(),
                     offabs_i=offabs.numpy().copy(), off10_i=off10.numpy().copy(),
                     off50_i=off50.numpy().copy(), q10_i=q10.numpy().copy(),
                     zcur_i=zcur.numpy().copy(), sdcur_i=sdcur.numpy().copy(),
                     hard_i=hard.numpy().copy())
    return row, units, g


def gcorr(g0, g1):
    """Pearson correlation of the two gate matrices.  Degenerate cases (追補 1): both
    constant -> 1.0 (perfectly stable), one constant -> 0.0; the count is recorded."""
    a = g0.flatten() - g0.mean(); b = g1.flatten() - g1.mean()
    na, nb = float(a.norm()), float(b.norm())
    if na == 0. and nb == 0.:
        return 1.0, 1
    if na == 0. or nb == 0.:
        return 0.0, 1
    return float((a * b).sum() / (na * nb)), 0


# ------------------------------------------------------------------ training
def train(arm, seed, mnist, probe, tasks, mutate=None, rows=None, units=None, ck=None):
    """'ref' training, step-identical to C.loop.  Returns per-task-end info."""
    p = H.init_params(seed, torch.device('cpu'))
    if mutate == 'init':
        with torch.no_grad():
            p[0].data[0, 0] += 1e-3
    act = make_act(arm)
    adam = ([torch.zeros_like(q) for q in p], [torch.zeros_like(q) for q in p], [0])
    gp, gd, gb = H.stream('perm', seed), H.stream('data', seed), H.stream('batch', seed)
    for task in range(1, tasks + 1):
        perm = torch.randperm(784, generator=gp)
        idx = H.stratified_draw(mnist, gd)
        order = torch.randperm(H.TASK_EXAMPLES, generator=gb)
        xs = mnist.train_x[idx][:, perm][order]
        ys = mnist.train_y[idx][order]
        _, _, g_start = gate_block(p, act, probe, perm)
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
        # ---- task end: committed measurement + gate block (read-outs only)
        r, u = C.measure(p, act, probe, perm, True, mnist, ck)
        gr_, gu, g_end = gate_block(p, act, probe, perm)
        r.update(gr_)
        r['gcorr'], r['gcorr_degenerate'] = gcorr(g_start, g_end)
        r.update(task=task, step=625, clamp='ref')
        if isinstance(act, H.AdaptiveSnake):
            r['alpha_med'] = float(act.alpha(0).median())
        if mutate == 'measure':
            with torch.no_grad():
                p[0].data.add_(1e-9)
        if rows is not None:
            rows.append(r)
            for k, v in u.items():
                units[f'ref_{k}_t{task}'] = v
            for k, v in gu.items():
                units[f'ref_{k}_t{task}'] = v
    return p, act


# ------------------------------------------------------------------ G1 anchors
def anchor(arm, seed):
    """(loader(units, t_from, t_to) -> (maxabs, n), expected n, first anchored task, note)."""
    R = ROOT / 'results'
    if arm in ('LR', 'ELU1', 'SNA', 'ELU03', 'CELU1'):
        src = 'ELU1' if arm == 'CELU1' else arm
        eg = np.load(R / 'elu_growth_0909' / f'{src}_none_s{seed}_units.npz')

        def cmp(units, t_from, t_to):
            w, n = 0., 0
            for t in range(t_from, min(t_to, 120) + 1):
                for k in ('zbar_i', 'sd_i', 'cnorm_i'):
                    key = f'ref_{k}_t{t}'
                    if key in units:
                        w = max(w, float(np.abs(units[key] - eg[k][t - 1]).max())); n += 1
            return w, n
        return cmp, 3 * 120, 1, f'elu_growth_0909/{src}_none_s{seed}_units.npz (registered)'
    if arm in ('R', 'LR03', 'LR001'):
        ref = np.load(R / 'leak_ladder_force_posthoc_0910' / f'{arm}_s{seed}_units.npz')
        return _cmp_keyed(ref, 100), 6 * 100, 1, f'leak_ladder_force_posthoc_0910/{arm}_s{seed}_units.npz (post-hoc; itself 0.0 vs gate_scale_invariance_0909)'
    if arm in ('GELU', 'SILU'):
        ref = np.load(R / 'long_horizon_acts_0910' / f'{arm}_none_s{seed}_units.npz')
        return _cmp_keyed(ref, 120), 6 * 120, 1, f'long_horizon_acts_0910/{arm}_none_s{seed}_units.npz (sub-run A of spec_transport_holes_0910)'
    if arm in ('SN02', 'SN06', 'SN15'):
        ref = np.load(R / 'gate_scale_invariance_0909' / f'{arm}_s{seed}_units.npz')
        tasks = [int(t) for t in ref['tasks']]

        def cmp(units, t_from, t_to):
            w, n = 0., 0
            for j, t in enumerate(tasks):
                key = f'ref_cnorm_i_t{t}'
                if t_from <= t <= t_to and key in units:
                    w = max(w, float(np.abs(units[key] - ref['cnorm_i'][j]).max())); n += 1
            return w, n
        return cmp, len(tasks), tasks[0], f'gate_scale_invariance_0909/{arm}_s{seed}_units.npz (registered; cnorm_i at 11 tasks)'
    if arm == 'LIN':
        ref = np.load(R / 'linear_growth_0910' / f'LIN_s{seed}_units.npz')
        tasks = [int(t) for t in ref['tasks']]

        def cmp(units, t_from, t_to):
            w, n = 0., 0
            for j, t in enumerate(tasks):
                for k in ('zbar_i', 'sd_i', 'cnorm_i'):
                    key = f'ref_{k}_t{t}'
                    if t_from <= t <= t_to and key in units:
                        w = max(w, float(np.abs(units[key] - ref[k][j]).max())); n += 1
            return w, n
        return cmp, 3 * len(tasks), tasks[0], f'linear_growth_0910/LIN_s{seed}_units.npz (registered; 3 arrays at 11 tasks)'
    return None, 0, None, 'no committed anchor (new activation)'


def _cmp_keyed(ref, last):
    def cmp(units, t_from, t_to):
        w, n = 0., 0
        for t in range(t_from, min(t_to, last) + 1):
            for k in T.UNIT_KEYS:
                key = f'ref_{k}_t{t}'
                if key in units and key in ref:
                    w = max(w, float(np.abs(units[key] - ref[key]).max())); n += 1
        return w, n
    return cmp


# ------------------------------------------------------------------ run
def run(arm, seed, tasks=TASKS, out=OUT, controls=True):
    torch.set_num_threads(1); H.setup('cpu'); T.data_dir()
    mnist = H.Mnist(torch.device('cpu'))
    out.mkdir(parents=True, exist_ok=True)
    tag = f'{arm}_s{seed}'
    pidx = torch.randperm(len(mnist.test_x), generator=H.stream('boundary_probe', seed))[:512]
    probe = C.Probe(mnist.test_x[pidx], mnist.test_y[pidx], float(mnist.train_x.mean()))
    t0 = time.monotonic()
    rows, units, ck = [], {}, {}
    ck.update(check_dphi(arm))
    assert ck['g2_dphi'] < 1e-6 and ck['g2_dphi_mutctl'] > 1e-3, ('G2', ck)
    if arm == 'ELUF':
        assert ck['g2_eluf_continuity'] < 1e-8, ('ELUF not continuous', ck)

    train(arm, seed, mnist, probe, tasks, rows=rows, units=units, ck=ck)
    T.finite_guard(rows, tag)
    # G3: gate bookkeeping identity (law of total variance, equal group sizes)
    ck['g3_ident_max'] = max(r['g3_ident'] for r in rows)
    assert ck['g3_ident_max'] <= 1e-9, ('G3 identity', ck['g3_ident_max'])
    # G1
    cmp, expected, first, note = anchor(arm, seed)
    ck['g1_anchor'] = note
    if cmp is not None and first is not None and first > tasks:
        ck.update(g1_units_maxabs=None, g1_units_compared=0, g1_units_expected=0,
                  g1_note=f'no anchored task within t1-{tasks}')
        cmp = None
    if cmp is not None:
        w, n = cmp(units, 1, tasks)
        # cnorm_i is computed identically in every anchor (Wt.norm), so it must match to the bit;
        # zbar_i / sd_i are accumulated in a different order by elu_growth_0909 (1e-16 rounding).
        cn = {k: v for k, v in units.items() if '_cnorm_i_t' in k}
        ck['g1_cnorm_maxabs'] = cmp(cn, 1, tasks)[0]
        exp_n = expected if tasks >= 120 else None
        ck.update(g1_units_maxabs=w, g1_units_compared=n, g1_units_expected=exp_n)
        assert w <= 1e-10, ('G1 failed', arm, seed, w)
        if exp_n is not None:
            assert n == exp_n, ('G1 count guard', n, exp_n)
        else:
            assert n > 0, 'G1 compared nothing'
    else:
        ck.update(g1_units_maxabs=None, g1_units_compared=0, g1_units_expected=0)
    if controls and seed == 0 and cmp is not None and first <= tasks:
        # G1 mutation control (init +1e-3) and measurement non-invasiveness control,
        # both run only to the first anchored task.
        u_init = {}
        train(arm, seed, mnist, probe, first, mutate='init', rows=[], units=u_init, ck={})
        ck['ctl_g1_init'] = cmp(u_init, first, first)[0]
        assert ck['ctl_g1_init'] > 100 * 1e-10, ('vacuous G1 init control', ck['ctl_g1_init'])
        u_meas = {}
        train(arm, seed, mnist, probe, first, mutate='measure', rows=[], units=u_meas, ck={})
        # the +1e-9 is applied AFTER the measurement of task `first`, so compare at first+1
        # when possible, else re-run one task further.
        u_meas2 = {}
        train(arm, seed, mnist, probe, first + 1, mutate='measure', rows=[], units=u_meas2, ck={})
        ck['ctl_measure_mut'] = cmp(u_meas2, first + 1, first + 1)[0] if cmp(u_meas2, first + 1, first + 1)[1] > 0 else float('nan')
        if np.isfinite(ck['ctl_measure_mut']):
            assert ck['ctl_measure_mut'] > 0., 'vacuous measurement control'
    if controls and seed == 0:
        # G3 control for every arm (追補 1), skipped only for a degenerate gate such as LIN
        pr, _, _ = gate_block([q.detach() for q in H.init_params(seed, torch.device('cpu'))], make_act(arm), probe, probe.refs[0], mut_var=True)
        ck['ctl_g3_ident'] = pr['g3_ident']
        ck['ctl_g3_applicable'] = pr['pooled_var'] > 1e-6
        if ck['ctl_g3_applicable']:
            assert pr['g3_ident'] > 100 * 1e-9, ('vacuous G3 control', pr['g3_ident'])

    for r in rows:
        r.update(arm=arm, seed=seed)
    T.write_rows(out / f'{tag}_rows.csv', rows)
    np.savez_compressed(out / f'{tag}_units.npz', **units)
    wall = time.monotonic() - t0
    T.dump(out / f'{tag}_provenance.json',
           dict(arm=arm, seed=seed, tasks=tasks, theta=THETA, checks=ck,
                spec_sha256=T.sha(SPEC), spec_dial_sha256=T.sha(SPEC_DIAL),
                code_sha256=T.sha(Path(__file__)), base_sha256=T.sha(Path(C.__file__)),
                host_sha256=T.sha(Path(H.__file__)), elu_sha256=T.sha(Path(EG.__file__)),
                valley_sha256=T.sha(Path(VA.__file__)), data_sha256=mnist.sha256,
                torch_version=torch.__version__, numpy_version=np.__version__,
                wall_seconds=wall, scope='CPU training from init; ref reproduces the committed anchor'))
    print('FINISHED', tag, 'G1', ck.get('g1_units_maxabs'), f"({ck.get('g1_units_compared')} arrays)",
          round(wall, 1), 's', flush=True)
    assert wall < TIME_CAP, ('runtime cap', wall)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm'); ap.add_argument('--seed', type=int)
    ap.add_argument('--all', action='store_true'); ap.add_argument('--dial', action='store_true')
    ap.add_argument('--arms', default=None); ap.add_argument('--jobs', type=int, default=6)
    ap.add_argument('--smoke', action='store_true'); ap.add_argument('--tasks', type=int, default=TASKS)
    a = ap.parse_args()
    if a.smoke:
        for arm in (a.arms.split(',') if a.arms else ['LR', 'SN02', 'CELU1', 'ELUF']):
            run(arm, 0, tasks=a.tasks if a.tasks != TASKS else 3, out=SMOKE, controls=True)
        return
    if a.all or a.dial:
        arms = a.arms.split(',') if a.arms else (DIAL_ARMS if a.dial else MAIN_ARMS)

        def job(j):
            subprocess.run([sys.executable, '-m', 'src.gate_shape_0911', '--arm', j[0], '--seed', str(j[1])], check=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as pool:
            for _ in pool.map(job, [(arm, s) for arm in arms for s in range(3)]):
                pass
        return
    run(a.arm, a.seed, tasks=a.tasks)


if __name__ == '__main__':
    main()
