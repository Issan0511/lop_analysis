"""grad_coherence_0911: move gradient coherence directly (spec_grad_coherence_0911).

Unbiased Gaussian noise is injected into the FIRST-LAYER gradient just before Adam:

    g'_i = g_i + c * s_i * xi_i,      s_i = ||g_i|| / sqrt(784),  xi ~ N(0, I)

E[g'] = g exactly, so the expected gradient, the data, the batch order, the number
of updates and the activation are all unchanged.  What changes is Adam's view:
v_hat picks up (1 + c^2), so the normalised step shrinks by 1/sqrt(1+c^2) and its
direction is scrambled.  The noise is scaled by each unit's OWN gradient, so the
relative SNR degradation is the same for every unit (a low-gate unit is not singled
out) -- that is what makes this a clean dial on coherence rather than on the gate.

The step-size confound is separated by the `LRh` / `LRq` arms: no noise, lr scaled
by 1/sqrt(1+c^2), i.e. the same step magnitude with the direction intact.

The noise is drawn from its own sha256-derived stream (`gradnoise`), and when c = 0
the generator is never touched, so the c = 0 arms reproduce the committed
trajectories bit for bit (G1).
"""
from pathlib import Path
import argparse, concurrent.futures, json, math, subprocess, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from src import width_sink_clamp_0909 as C
from src import clamp_horizon_0910 as CH
from src import transport_common_0910 as T
from src import gate_shape_0911 as GS

H = C.H
ROOT = C.ROOT
OUT = ROOT / 'results/grad_coherence_0911'
SMOKE = ROOT / 'results/_smoke_grad_coherence_0911'
TASKS = 120
CLAMP_FROM = 21
LATE = (61, 120)
LR0 = .001
EPS = 1e-8
TIME_CAP = 900.
SPEC = ROOT / 'specs/spec_grad_coherence_0911.md'

# name -> (activation, noise c, which params get noise, lr, clamp)
W1_ONLY = (0,)
L23 = (2, 4)
ARMS = {
    'N0':    ('LR',  0.0, W1_ONLY, LR0,                 'ref'),
    'N05':   ('LR',  0.5, W1_ONLY, LR0,                 'ref'),
    'N1':    ('LR',  1.0, W1_ONLY, LR0,                 'ref'),
    'N2':    ('LR',  2.0, W1_ONLY, LR0,                 'ref'),
    'LRh':   ('LR',  0.0, W1_ONLY, LR0 / math.sqrt(2.), 'ref'),
    'LRq':   ('LR',  0.0, W1_ONLY, LR0 / math.sqrt(5.), 'ref'),
    'N1L23': ('LR',  1.0, L23,     LR0,                 'ref'),
    'N0w':   ('LR',  0.0, W1_ONLY, LR0,                 'wclamp'),
    'N1w':   ('LR',  1.0, W1_ONLY, LR0,                 'wclamp'),
    'SN0':   ('SNA', 0.0, W1_ONLY, LR0,                 'ref'),
    'SN1':   ('SNA', 1.0, W1_ONLY, LR0,                 'ref'),
}
# arms whose c = 0 and lr = LR0 and clamp = ref must reproduce a committed trajectory
ANCHORED = {'N0': 'LR', 'SN0': 'SNA'}


# ------------------------------------------------------------------- the dial
def inject(g, c, gen):
    """Unbiased per-row relative noise.  Never called when c == 0."""
    s = g.norm(dim=1, keepdim=True) / math.sqrt(g.shape[1])
    xi = torch.randn(g.shape, generator=gen, dtype=g.dtype)
    return g + (c * s) * xi


def check_unbiased(c, seed, n=2000):
    """G2: the injection is unbiased and has the intended variance, with a control."""
    gen = H.stream('gradnoise', seed)
    g = torch.randn((8, 784), generator=H.stream('g2probe', seed))
    s = g.norm(dim=1, keepdim=True) / math.sqrt(784)
    acc = torch.zeros_like(g); acc2 = torch.zeros_like(g)
    for _ in range(n):
        d = inject(g, c, gen) - g
        acc += d; acc2 += d * d
    mean = acc / n
    var = acc2 / n
    scale = (c * s).clamp(min=1e-30)
    # Tolerances come from the sampling distribution, not from a guessed constant:
    # mean/scale has sd 1/sqrt(n) and var/scale^2 has sd sqrt(2/n), and the checks take a
    # MAX over 8*784 coordinates, so the bar is 5 sd.  (A flat 0.10 rejects a correct
    # injector: the observed max is ~4 sd = 0.126 at n = 2000.)
    out = dict(g2_n=n, g2_mean_rel=float((mean / scale).abs().max()),
               g2_mean_tol=5. / math.sqrt(n),
               g2_var_rel=float(((var / (scale ** 2)) - 1).abs().max()),
               g2_var_tol=5. * math.sqrt(2. / n),
               g2_var_rel_pooled=float(abs(float((var / (scale ** 2)).mean()) - 1.)))
    # control: a bias of half the injected scale must fail the mean test at any c
    out['g2_mean_rel_mutctl'] = float(((mean + .5 * scale) / scale).abs().max())
    return out


# ------------------------------------------------------------------ training
def loop(p, act, adam, gens, ngen, mnist, probe, t_from, t_to, arm_cfg, base,
         rows, units, ck, instrument=True, mut=None, cfrom=CLAMP_FROM):
    """One update is character-identical to the committed C.loop ref branch, except
    for the injection (skipped entirely when c == 0) and the lr constant."""
    _, c_noise, which, lr, clamp = arm_cfg
    gp, gd, gb = gens
    for task in range(t_from, t_to + 1):
        perm = torch.randperm(784, generator=gp)
        idx = H.stratified_draw(mnist, gd)
        order = torch.randperm(H.TASK_EXAMPLES, generator=gb)
        xs = mnist.train_x[idx][:, perm][order]
        ys = mnist.train_y[idx][order]
        late = instrument and LATE[0] <= task <= LATE[1]
        if late:
            S2 = torch.zeros(100, dtype=torch.float64); tot = torch.zeros(100, 784, dtype=torch.float64)
            graw2 = torch.zeros(100, dtype=torch.float64); kap2 = torch.zeros(100, dtype=torch.float64)
            Wt0 = (lambda W: W - W.mean(1, keepdim=True))(p[0].detach().double()).clone()
        ce20 = None
        for step in range(1, 626):
            out = H.forward(p, xs[(step - 1) * 16:step * 16], act)
            gr = torch.autograd.grad(torch.nn.functional.cross_entropy(
                out[4], ys[(step - 1) * 16:step * 16]), p)
            with torch.no_grad():
                if late:
                    Wb = p[0].detach().double().clone()
                    graw2 += (gr[0].double() ** 2).sum(1)
                if c_noise > 0:
                    raw = gr
                    gr = list(gr)
                    for j in which:
                        gr[j] = inject(gr[j], c_noise, ngen)
                    if ck is not None and step % 100 == 0:
                        # G3 (real, not by construction): the realised relative perturbation is
                        # ~c on the targeted tensors and EXACTLY 0 on every other one.
                        for j in range(len(gr)):
                            d = float((gr[j] - raw[j]).norm()); n0 = float(raw[j].norm()) or 1.
                            if j in which:
                                ck['g3_inj_lo'] = min(ck.get('g3_inj_lo', 9e9), d / n0)
                                ck['g3_inj_hi'] = max(ck.get('g3_inj_hi', 0.), d / n0)
                            else:
                                ck['g3_untouched'] = max(ck.get('g3_untouched', 0.), d)
                m_, v_, tc = adam
                tc[0] += 1
                c1 = 1 - .9 ** tc[0]; c2 = 1 - .999 ** tc[0]
                for q, g, mi, vi in zip(p, gr, m_, v_):
                    mi.mul_(.9).add_(g, alpha=1 - .9)
                    vi.mul_(.999).addcmul_(g, g, value=1 - .999)
                    q -= lr * (mi / c1) / ((vi / c2).sqrt() + 1e-8)
                if late:
                    kap2 += (((p[0].detach().double() - Wb) / lr) ** 2).mean(1)
                if isinstance(act, H.AdaptiveSnake):
                    act.update(out[0], out[2])
                if clamp != 'ref' and task >= cfrom:
                    CH.apply_clamp(p, clamp, base, ck, mut)
                    CH.verify_clamped(p, clamp, base, ck)
                if late:
                    dW = p[0].detach().double() - Wb
                    dWt = dW - dW.mean(1, keepdim=True)
                    S2 += (dWt ** 2).sum(1); tot += dWt
            if step == 20:
                with torch.no_grad():
                    ce20 = float(F.cross_entropy(H.forward(p, probe.px[:, perm], act)[4], probe.py))
        if rows is None:
            continue
        r, u = C.measure(p, act, probe, perm, True, mnist, ck)
        gr_, gu, _ = GS.gate_block(p, act, probe, perm)
        r.update(gr_)
        r.update(task=task, step=625, clamp=clamp, ce20=ce20)
        for k, v in u.items():
            units[f'ref_{k}_t{task}'] = v
        for k, v in gu.items():
            units[f'ref_{k}_t{task}'] = v
        if late:
            D2 = (tot ** 2).sum(1)
            rho = torch.where(S2 > 0, D2 / S2.clamp(min=1e-300), torch.full_like(S2, float('nan')))
            Wt1 = (lambda W: W - W.mean(1, keepdim=True))(p[0].detach().double())
            ck['g6_tot_ident'] = max(ck.get('g6_tot_ident', 0.), float((tot - (Wt1 - Wt0)).abs().max()))
            ck['g6_rho_max'] = max(ck.get('g6_rho_max', 0.), float(torch.nan_to_num(rho, nan=0.).max()))
            r.update(S2=float(S2.mean()), D2=float(D2.mean()), rho_mean=float(torch.nanmean(rho)),
                     kap2=float(kap2.mean() / 625), graw2=float(graw2.mean() / 625),
                     step2=float(((S2 / 625).mean())))
            units[f'ref_kap2_i_t{task}'] = (kap2 / 625).numpy().copy()
            units[f'ref_S2_i_t{task}'] = S2.numpy().copy()
        rows.append(r)
    return p, act, adam


def run(arm, seed, tasks=TASKS, out=OUT, controls=True, clamp_from=CLAMP_FROM):
    torch.set_num_threads(1); H.setup('cpu'); T.data_dir()
    cfg = ARMS[arm]
    actname, c_noise, which, lr, clamp = cfg
    mnist = H.Mnist(torch.device('cpu'))
    out.mkdir(parents=True, exist_ok=True)
    tag = f'{arm}_s{seed}'
    pidx = torch.randperm(len(mnist.test_x), generator=H.stream('boundary_probe', seed))[:512]
    probe = C.Probe(mnist.test_x[pidx], mnist.test_y[pidx], float(mnist.train_x.mean()))
    t0 = time.monotonic()
    rows, units, ck = [], {}, {}
    ck.update(GS.check_dphi(actname))
    assert ck['g2_dphi'] < 1e-6 and ck['g2_dphi_mutctl'] > 1e-3, ('phi-prime', ck)
    ck['lr'] = lr; ck['noise_c'] = c_noise; ck['noise_params'] = list(which); ck['clamp'] = clamp
    want = {'LRh': LR0 / math.sqrt(2.), 'LRq': LR0 / math.sqrt(5.)}.get(arm, LR0)
    ck['g4_lr_rel_err'] = abs(lr / want - 1)
    assert ck['g4_lr_rel_err'] < 1e-12, ('G4 lr', lr, want)
    if c_noise > 0 and controls:
        ck.update(check_unbiased(c_noise, seed))
        assert ck['g2_mean_rel'] < ck['g2_mean_tol'], ('G2 biased injection', ck['g2_mean_rel'], ck['g2_mean_tol'])
        assert ck['g2_var_rel'] < ck['g2_var_tol'], ('G2 variance (max)', ck['g2_var_rel'], ck['g2_var_tol'])
        assert ck['g2_var_rel_pooled'] < 0.02, ('G2 variance (pooled)', ck['g2_var_rel_pooled'])
        assert ck['g2_mean_rel_mutctl'] > 3 * ck['g2_mean_tol'], ('vacuous G2 control', ck['g2_mean_rel_mutctl'])

    p = H.init_params(seed, torch.device('cpu'))
    act = GS.make_act(actname)
    adam = ([torch.zeros_like(q) for q in p], [torch.zeros_like(q) for q in p], [0])
    gens = [H.stream('perm', seed), H.stream('data', seed), H.stream('batch', seed)]
    ngen = H.stream('gradnoise', seed)
    st0 = [g.get_state().clone() for g in gens]
    base = None
    if clamp != 'ref':
        loop(p, act, adam, gens, ngen, mnist, probe, 1, clamp_from - 1, cfg, None, rows, units, ck, cfrom=clamp_from)
        base = C.baselines(p[0])
        ck['f32_rowmean_bound_t20'] = CH.EPS32 * float(p[0].detach().double().abs().mean(1).max())
        ck.update(base_cnorm_mean=float(base['c'].mean()))
        loop(p, act, adam, gens, ngen, mnist, probe, clamp_from, tasks, cfg, base, rows, units, ck, cfrom=clamp_from)
    else:
        loop(p, act, adam, gens, ngen, mnist, probe, 1, tasks, cfg, None, rows, units, ck)
    T.finite_guard(rows, tag)

    # G3: the noise stream must not have disturbed the other streams
    if c_noise > 0:
        g2 = [H.stream('perm', seed), H.stream('data', seed), H.stream('batch', seed)]
        ck['g3_streams_intact'] = all(torch.equal(a, b.get_state()) for a, b in zip(st0, g2))
        assert ck['g3_streams_intact'], 'the gradnoise stream perturbed perm/data/batch'
        assert ck['g3_untouched'] == 0., ('noise leaked into an untargeted tensor', ck['g3_untouched'])
        assert 0.5 * c_noise < ck['g3_inj_lo'] and ck['g3_inj_hi'] < 2.0 * c_noise, \
            ('realised injection is not ~c', c_noise, ck['g3_inj_lo'], ck['g3_inj_hi'])

    # G1: the c = 0, lr = LR0, ref arms reproduce the committed trajectory
    # c = 0 and lr = LR0 arms reproduce the committed trajectory: the whole run for the
    # unclamped ones, and the shared t1..t20 prefix for the clamped one.
    anch = {'N0': ('LR', 120), 'SN0': ('SNA', 120), 'N0w': ('LR', clamp_from - 1)}.get(arm)
    ck['g1_anchor'] = f'elu_growth_0909/{anch[0]} t1-{anch[1]}' if anch else 'none (noise / lr arm)'
    if anch:
        eg = np.load(ROOT / 'results/elu_growth_0909' / f'{anch[0]}_none_s{seed}_units.npz')
        w, wc, n = 0., 0., 0
        for t in range(1, min(tasks, anch[1]) + 1):
            for k in ('zbar_i', 'sd_i', 'cnorm_i'):
                key = f'ref_{k}_t{t}'
                if key in units:
                    d = float(np.abs(units[key] - eg[k][t - 1]).max())
                    w = max(w, d); n += 1
                    if k == 'cnorm_i':
                        wc = max(wc, d)
        ck.update(g1_units_maxabs=w, g1_cnorm_maxabs=wc, g1_units_compared=n,
                  g1_units_expected=3 * min(tasks, anch[1]))
        assert w <= 1e-10, ('G1 failed', arm, seed, w)
        assert n == ck['g1_units_expected'], ('G1 count guard', n, ck['g1_units_expected'])
        if controls and seed == 0:
            p3 = H.init_params(seed, torch.device('cpu'))
            with torch.no_grad():
                p3[0].data[0, 0] += 1e-3
            a3 = GS.make_act(actname)
            ad3 = ([torch.zeros_like(q) for q in p3], [torch.zeros_like(q) for q in p3], [0])
            g3 = [H.stream('perm', seed), H.stream('data', seed), H.stream('batch', seed)]
            u3 = {}
            loop(p3, a3, ad3, g3, H.stream('gradnoise', seed), mnist, probe, 1, 1, cfg, None, [], u3, {}, instrument=False)
            ck['ctl_g1_init'] = max(float(np.abs(u3[f'ref_{k}_t1'] - eg[k][0]).max()) for k in ('zbar_i', 'sd_i', 'cnorm_i'))
            assert ck['ctl_g1_init'] > 1e-8, ('vacuous G1 control', ck['ctl_g1_init'])
    else:
        ck.update(g1_units_maxabs=None, g1_cnorm_maxabs=None, g1_units_compared=0, g1_units_expected=0)

    if clamp != 'ref':
        for k in ('c3_cnorm_rel', 'c3_m_absdiff', 'clamp_tail_absdiff'):
            assert k in ck, ('clamp check never recorded', k)
            assert ck[k] <= CH.limit(ck, k, 'f32_rowmean_bound_t20'), ('G5', k, ck[k])
    assert ck.get('g6_rho_max', 0.) <= 625 * (1 + 1e-9), ('G6 Cauchy-Schwarz', ck['g6_rho_max'])
    assert ck.get('g6_tot_ident', 0.) <= 1e-10, ('G6 telescoping', ck['g6_tot_ident'])

    for r in rows:
        r.update(arm=arm, seed=seed)
    T.write_rows(out / f'{tag}_rows.csv', rows)
    np.savez_compressed(out / f'{tag}_units.npz', **units)
    wall = time.monotonic() - t0
    T.dump(out / f'{tag}_provenance.json',
           dict(arm=arm, seed=seed, tasks=tasks, cfg=dict(act=actname, c=c_noise, params=list(which), lr=lr, clamp=clamp),
                late=LATE, checks=ck, spec_sha256=T.sha(SPEC), code_sha256=T.sha(Path(__file__)),
                base_sha256=T.sha(Path(C.__file__)), host_sha256=T.sha(Path(H.__file__)),
                gate_shape_sha256=T.sha(Path(GS.__file__)), data_sha256=mnist.sha256,
                torch_version=torch.__version__, wall_seconds=wall,
                scope='CPU training from init; unbiased noise on the pre-Adam gradient'))
    print('FINISHED', tag, 'G1', ck.get('g1_units_maxabs'), round(wall, 1), 's', flush=True)
    assert wall < TIME_CAP, ('runtime cap', wall)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm'); ap.add_argument('--seed', type=int)
    ap.add_argument('--all', action='store_true'); ap.add_argument('--arms', default=None)
    ap.add_argument('--jobs', type=int, default=8); ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--tasks', type=int, default=TASKS)
    a = ap.parse_args()
    if a.smoke:
        for arm in (a.arms.split(',') if a.arms else ['N0', 'N1', 'LRh', 'N1w', 'N1L23']):
            run(arm, 0, tasks=4, out=SMOKE, controls=True, clamp_from=3)
        return
    if a.all:
        arms = a.arms.split(',') if a.arms else list(ARMS)

        def job(j):
            subprocess.run([sys.executable, '-m', 'src.grad_coherence_0911', '--arm', j[0], '--seed', str(j[1])], check=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as pool:
            for _ in pool.map(job, [(arm, s) for arm in arms for s in range(3)]):
                pass
        return
    run(a.arm, a.seed, tasks=a.tasks)


if __name__ == '__main__':
    main()
