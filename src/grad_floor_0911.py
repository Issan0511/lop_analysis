"""grad_floor_0911: is the harm the noise FLOOR or the loss of coherence?
(spec_grad_floor_0911)

Two injectors with IDENTICAL total energy per row and different allocation:

  row    g'_ij = g_ij + c * s_i    * xi_ij     s_i = ||g_i|| / sqrt(784)   (a uniform floor)
  coord  g'_ij = g_ij + c * |g_ij| * xi_ij                                 (uniform relative SNR)

  sum_j (c s_i)^2 = c^2 * 784 * ||g_i||^2/784 = c^2 ||g_i||^2 = sum_j (c |g_ij|)^2

Both are unbiased.  'row' puts a floor at c*s_i, which the reference-arm measurement puts
near the 90th percentile of |g_ij|/s_i (median 0.003, 65% below 0.2), so nearly every
coordinate's signal is drowned.  'coord' creates no floor and leaves g_ij = 0 exactly
untouched; it multiplies v_hat by (1+c^2) uniformly, i.e. it acts like lr/sqrt(1+c^2)
plus incoherence.  The lr-compensated arm (lr * sqrt(1+c^2)) isolates the incoherence.

The 'row' arm must reproduce grad_coherence_0911/N2 bit for bit, and 'LRq' its LRq, so
that adding the new mode is provably inert for the existing arms (G1).
"""
from pathlib import Path
import argparse, concurrent.futures, json, math, subprocess, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from src import width_sink_clamp_0909 as C
from src import transport_common_0910 as T
from src import gate_shape_0911 as GS
from src import grad_coherence_0911 as GC

H = C.H
ROOT = C.ROOT
OUT = ROOT / 'results/grad_floor_0911'
SMOKE = ROOT / 'results/_smoke_grad_floor_0911'
TASKS = 120
LATE = (61, 120)
LR0 = .001
CE_STEPS = (20, 100, 300, 625)
TIME_CAP = 900.
SPEC = ROOT / 'specs/spec_grad_floor_0911.md'
SQ5 = math.sqrt(5.)

# name -> (mode, c, lr)
ARMS = {
    'N0':   (None,    0.0, LR0),
    'N2':   ('row',   2.0, LR0),
    'LRq':  (None,    0.0, LR0 / SQ5),
    'Gm2':  ('coord', 2.0, LR0),
    'Gm2c': ('coord', 2.0, LR0 * SQ5),
    'LRx':  (None,    0.0, LR0 * SQ5),
}
ANCHOR = {'N0': ('eg', 'LR'), 'N2': ('gc', 'N2'), 'LRq': ('gc', 'LRq')}


# ------------------------------------------------------------------- injectors
def inject(g, mode, c, gen):
    """Unbiased.  'row' is character-identical to grad_coherence_0911.inject."""
    if mode == 'row':
        return GC.inject(g, c, gen)
    xi = torch.randn(g.shape, generator=gen, dtype=g.dtype)
    return g + (c * g.abs()) * xi


def check_inject(mode, c, seed, n=2000):
    """G2: unbiased with the intended per-coordinate variance, tolerances from the
    sampling distribution (5 sd), plus a biased-injector control.  For 'coord' the
    zero coordinates must stay exactly zero."""
    gen = H.stream('gradnoise', seed)
    g = torch.randn((8, 784), generator=H.stream('g2probe', seed))
    g[:, :300] = 0.                                   # empty pixels, as in the real gradient
    s = g.norm(dim=1, keepdim=True) / math.sqrt(784)
    scale = (c * s).expand_as(g) if mode == 'row' else (c * g.abs())
    acc = torch.zeros_like(g); acc2 = torch.zeros_like(g); zmax = 0.
    for _ in range(n):
        d = inject(g, mode, c, gen) - g
        acc += d; acc2 += d * d
        if mode == 'coord':
            zmax = max(zmax, float(d[g == 0].abs().max()) if (g == 0).any() else 0.)
    nz = scale > 0
    mean, var = acc / n, acc2 / n
    out = dict(g2_n=n, g2_mode=mode,
               g2_mean_rel=float((mean[nz] / scale[nz]).abs().max()), g2_mean_tol=5. / math.sqrt(n),
               g2_var_rel=float(((var[nz] / scale[nz] ** 2) - 1).abs().max()), g2_var_tol=5. * math.sqrt(2. / n),
               g2_var_pooled=float(abs(float((var[nz] / scale[nz] ** 2).mean()) - 1.)),
               g2_zero_coord_max=zmax,
               g2_mean_rel_mutctl=float(((mean[nz] + .5 * scale[nz]) / scale[nz]).abs().max()),
               # the identity the whole design rests on: both modes inject the same energy
               g2_energy_row=float(((c * s) ** 2 * 784).mean()),
               g2_energy_coord=float(((c * g.abs()) ** 2).sum(1).mean()))
    out['g2_energy_rel'] = abs(out['g2_energy_row'] / out['g2_energy_coord'] - 1)
    return out


# ------------------------------------------------------------------ training
def probe_ce(p, act, probe, perm):
    with torch.no_grad():
        return float(F.cross_entropy(H.forward(p, probe.px[:, perm], act)[4], probe.py))


def grad_shape(g, s):
    """Registered read-out of how uneven the gradient is across coordinates in a row."""
    r = (g.abs() / s).flatten().numpy()
    q = np.percentile(r, [10, 50, 90, 99])
    gg = g ** 2
    top = torch.topk(gg, 78, dim=1).values.sum(1)
    return dict(r_p10=float(q[0]), r_p50=float(q[1]), r_p90=float(q[2]), r_p99=float(q[3]),
                r_lt02=float((r < .2).mean()), r_gt2=float((r > 2.).mean()),
                top10_energy=float((top / gg.sum(1).clamp(min=1e-300)).mean()))


def loop(p, act, adam, gens, ngen, mnist, probe, t_from, t_to, cfg, rows, units, ck, instrument=True):
    mode, c_noise, lr = cfg
    gp, gd, gb = gens
    for task in range(t_from, t_to + 1):
        perm = torch.randperm(784, generator=gp)
        idx = H.stratified_draw(mnist, gd)
        order = torch.randperm(H.TASK_EXAMPLES, generator=gb)
        xs = mnist.train_x[idx][:, perm][order]
        ys = mnist.train_y[idx][order]
        ces = {0: probe_ce(p, act, probe, perm)}
        late = instrument and LATE[0] <= task <= LATE[1]
        if late:
            S2 = torch.zeros(100, dtype=torch.float64); tot = torch.zeros(100, 784, dtype=torch.float64)
            drift2 = torch.zeros(100, dtype=torch.float64); diff2 = torch.zeros(100, dtype=torch.float64)
            graw2 = torch.zeros(100, dtype=torch.float64); kap2 = torch.zeros(100, dtype=torch.float64)
            Wt0 = (lambda W: W - W.mean(1, keepdim=True))(p[0].detach().double()).clone()
            shape = None
        for step in range(1, 626):
            out = H.forward(p, xs[(step - 1) * 16:step * 16], act)
            gr = torch.autograd.grad(torch.nn.functional.cross_entropy(
                out[4], ys[(step - 1) * 16:step * 16]), p)
            with torch.no_grad():
                Wb = p[0].detach().double().clone() if late else None
                if late:
                    g_raw = gr[0].double().clone()
                    graw2 += (g_raw ** 2).sum(1)
                    if step == 300:
                        shape = grad_shape(g_raw, g_raw.norm(dim=1, keepdim=True).clamp(min=1e-300) / math.sqrt(784))
                if mode is not None:
                    raw0 = gr[0]
                    gr = list(gr)
                    gr[0] = inject(gr[0], mode, c_noise, ngen)
                    if ck is not None and step % 100 == 0:
                        d = (gr[0] - raw0).double(); r0 = raw0.double()
                        rel = float(d.norm() / r0.norm().clamp(min=1e-300))
                        ck['g0_inj_lo'] = min(ck.get('g0_inj_lo', 9e9), rel)
                        ck['g0_inj_hi'] = max(ck.get('g0_inj_hi', 0.), rel)
                        if mode == 'coord':
                            z = r0 == 0
                            ck['g0_zero_touched'] = max(ck.get('g0_zero_touched', 0.),
                                                        float(d[z].abs().max()) if bool(z.any()) else 0.)
                m_, v_, tc = adam
                tc[0] += 1
                c1 = 1 - .9 ** tc[0]; c2 = 1 - .999 ** tc[0]
                for q, g, mi, vi in zip(p, gr, m_, v_):
                    mi.mul_(.9).add_(g, alpha=1 - .9)
                    vi.mul_(.999).addcmul_(g, g, value=1 - .999)
                    q -= lr * (mi / c1) / ((vi / c2).sqrt() + 1e-8)
                if late:
                    dW = p[0].detach().double() - Wb
                    kap2 += ((dW / lr) ** 2).mean(1)
                    dWt = dW - dW.mean(1, keepdim=True)
                    S2 += (dWt ** 2).sum(1); tot += dWt
                    gn = g_raw.norm(dim=1, keepdim=True).clamp(min=1e-300)
                    ghat = g_raw / gn
                    dr = (dW * ghat).sum(1, keepdim=True) * ghat
                    drift2 += (dr ** 2).sum(1); diff2 += ((dW - dr) ** 2).sum(1)
                    ck['g5_proj_ident'] = max(ck.get('g5_proj_ident', 0.), float(
                        (((dr ** 2).sum(1) + ((dW - dr) ** 2).sum(1)) - (dW ** 2).sum(1)).abs().max()
                        / max(float((dW ** 2).sum(1).max()), 1e-300)))
            if step in CE_STEPS:
                ces[step] = probe_ce(p, act, probe, perm)
        if rows is None:
            continue
        r, u = C.measure(p, act, probe, perm, True, mnist, ck)
        gr_, gu, _ = GS.gate_block(p, act, probe, perm)
        r.update(gr_)
        r.update(task=task, step=625, clamp='ref', ce0=ces[0], ce20=ces[20],
                 ce100=ces[100], ce300=ces[300], ce625=ces[625])
        with torch.no_grad():
            r.update(w2_fro=float(p[2].double().norm()), w3_fro=float(p[4].double().norm()))
        for k, v in u.items():
            units[f'ref_{k}_t{task}'] = v
        for k, v in gu.items():
            units[f'ref_{k}_t{task}'] = v
        if late:
            D2 = (tot ** 2).sum(1)
            rho = torch.where(S2 > 0, D2 / S2.clamp(min=1e-300), torch.full_like(S2, float('nan')))
            Wt1 = (lambda W: W - W.mean(1, keepdim=True))(p[0].detach().double())
            ck['g5_tot_ident'] = max(ck.get('g5_tot_ident', 0.), float((tot - (Wt1 - Wt0)).abs().max()))
            ck['g5_rho_max'] = max(ck.get('g5_rho_max', 0.), float(torch.nan_to_num(rho, nan=0.).max()))
            tote = drift2 + diff2
            r.update(S2=float(S2.mean()), D2=float(D2.mean()), rho_mean=float(torch.nanmean(rho)),
                     kap2=float(kap2.mean() / 625), graw2=float(graw2.mean() / 625),
                     drift2=float(drift2.mean() / 625 / lr ** 2), diff2=float(diff2.mean() / 625 / lr ** 2),
                     f_diff=float(diff2.sum() / tote.sum()), **(shape or {}))
            units[f'ref_kap2_i_t{task}'] = (kap2 / 625).numpy().copy()
        rows.append(r)
    return p, act, adam


def run(arm, seed, tasks=TASKS, out=OUT, controls=True):
    torch.set_num_threads(1); H.setup('cpu'); T.data_dir()
    cfg = ARMS[arm]
    mode, c_noise, lr = cfg
    mnist = H.Mnist(torch.device('cpu'))
    out.mkdir(parents=True, exist_ok=True)
    tag = f'{arm}_s{seed}'
    pidx = torch.randperm(len(mnist.test_x), generator=H.stream('boundary_probe', seed))[:512]
    probe = C.Probe(mnist.test_x[pidx], mnist.test_y[pidx], float(mnist.train_x.mean()))
    t0 = time.monotonic()
    rows, units, ck = [], {}, {}
    ck.update(GS.check_dphi('LR'))
    assert ck['g2_dphi'] < 1e-6, ('phi-prime', ck)
    ck.update(mode=mode, noise_c=c_noise, lr=lr)
    want = {'LRq': LR0 / SQ5, 'Gm2c': LR0 * SQ5, 'LRx': LR0 * SQ5}.get(arm, LR0)
    ck['g4_lr_rel_err'] = abs(lr / want - 1)
    assert ck['g4_lr_rel_err'] < 1e-12, ('G4 lr', lr, want)
    if mode is not None and controls:
        ck.update(check_inject(mode, c_noise, seed))
        assert ck['g2_mean_rel'] < ck['g2_mean_tol'], ('G2 bias', ck['g2_mean_rel'], ck['g2_mean_tol'])
        assert ck['g2_var_rel'] < ck['g2_var_tol'], ('G2 var max', ck['g2_var_rel'], ck['g2_var_tol'])
        assert ck['g2_var_pooled'] < 0.02, ('G2 var pooled', ck['g2_var_pooled'])
        assert ck['g2_mean_rel_mutctl'] > 3 * ck['g2_mean_tol'], ('vacuous G2 control', ck)
        assert ck['g2_energy_rel'] < 1e-6, ('the two modes do not inject equal energy', ck['g2_energy_rel'])
        if mode == 'coord':
            assert ck['g2_zero_coord_max'] == 0., ('coord mode touched a zero coordinate', ck['g2_zero_coord_max'])

    p = H.init_params(seed, torch.device('cpu'))
    act = GS.make_act('LR')
    adam = ([torch.zeros_like(q) for q in p], [torch.zeros_like(q) for q in p], [0])
    gens = [H.stream('perm', seed), H.stream('data', seed), H.stream('batch', seed)]
    ngen = H.stream('gradnoise', seed)
    st0 = [g.get_state().clone() for g in gens]
    n0 = ngen.get_state().clone()
    loop(p, act, adam, gens, ngen, mnist, probe, 1, tasks, cfg, rows, units, ck)
    T.finite_guard(rows, tag)

    g2s = [H.stream('perm', seed), H.stream('data', seed), H.stream('batch', seed)]
    ck['g3_streams_intact'] = all(torch.equal(a, b.get_state()) for a, b in zip(st0, g2s))
    assert ck['g3_streams_intact'], 'the gradnoise stream perturbed perm/data/batch'
    if mode is None:
        ck['g3_ngen_untouched'] = bool(torch.equal(n0, ngen.get_state()))
        assert ck['g3_ngen_untouched']
    else:
        assert 0.9 * c_noise <= ck['g0_inj_lo'] and ck['g0_inj_hi'] <= 1.1 * c_noise, \
            ('H0.1 realised injection not in [0.9c, 1.1c]', c_noise, ck['g0_inj_lo'], ck['g0_inj_hi'])
        if mode == 'coord':
            assert ck.get('g0_zero_touched', 0.) == 0., ('H0.2 zero coordinate moved', ck['g0_zero_touched'])

    anch = ANCHOR.get(arm)
    if anch:
        kind, src = anch
        w, wc, n = 0., 0., 0
        if kind == 'eg':
            eg = np.load(ROOT / 'results/elu_growth_0909' / f'{src}_none_s{seed}_units.npz')
            for t in range(1, min(tasks, 120) + 1):
                for k in ('zbar_i', 'sd_i', 'cnorm_i'):
                    key = f'ref_{k}_t{t}'
                    if key in units:
                        d = float(np.abs(units[key] - eg[k][t - 1]).max()); w = max(w, d); n += 1
                        if k == 'cnorm_i':
                            wc = max(wc, d)
            exp = 3 * min(tasks, 120)
        else:
            ref = np.load(ROOT / 'results/grad_coherence_0911' / f'{src}_s{seed}_units.npz')
            for key in units:
                if key in ref and int(key.rsplit('_t', 1)[1]) <= tasks:
                    d = float(np.abs(units[key].astype(float) - ref[key].astype(float)).max())
                    w = max(w, d); n += 1
                    if '_cnorm_i_' in key:
                        wc = max(wc, d)
            exp = None
        ck.update(g1_anchor=f'{kind}:{src}', g1_units_maxabs=w, g1_cnorm_maxabs=wc,
                  g1_units_compared=n, g1_units_expected=exp)
        assert w <= 1e-10, ('G1 failed', arm, seed, w)
        assert n > 0 and (exp is None or n == exp), ('G1 count guard', n, exp)
        if kind == 'gc':
            assert n >= 6 * min(tasks, 120), ('G1 compared too few arrays', n)
        if controls and seed == 0 and arm == 'N0':
            p3 = H.init_params(seed, torch.device('cpu'))
            with torch.no_grad():
                p3[0].data[0, 0] += 1e-3
            ad3 = ([torch.zeros_like(q) for q in p3], [torch.zeros_like(q) for q in p3], [0])
            g3 = [H.stream('perm', seed), H.stream('data', seed), H.stream('batch', seed)]
            u3 = {}
            loop(p3, GS.make_act('LR'), ad3, g3, H.stream('gradnoise', seed), mnist, probe, 1, 1,
                 ARMS['N0'], [], u3, {}, instrument=False)
            eg = np.load(ROOT / 'results/elu_growth_0909' / f'LR_none_s{seed}_units.npz')
            ck['ctl_g1_init'] = max(float(np.abs(u3[f'ref_{k}_t1'] - eg[k][0]).max()) for k in ('zbar_i', 'sd_i', 'cnorm_i'))
            assert ck['ctl_g1_init'] > 1e-8, ('vacuous G1 control', ck['ctl_g1_init'])
    else:
        ck.update(g1_anchor='none (new mode / lr arm)', g1_units_maxabs=None, g1_units_compared=0)
    assert ck.get('g5_rho_max', 0.) <= 625 * (1 + 1e-9), ('G5 Cauchy-Schwarz', ck['g5_rho_max'])
    assert ck.get('g5_tot_ident', 0.) <= 1e-10, ('G5 telescoping', ck['g5_tot_ident'])
    assert ck.get('g5_proj_ident', 0.) <= 1e-10, ('G5 projection identity', ck['g5_proj_ident'])

    for r in rows:
        r.update(arm=arm, seed=seed)
    T.write_rows(out / f'{tag}_rows.csv', rows)
    np.savez_compressed(out / f'{tag}_units.npz', **units)
    wall = time.monotonic() - t0
    T.dump(out / f'{tag}_provenance.json',
           dict(arm=arm, seed=seed, tasks=tasks, cfg=dict(mode=mode, c=c_noise, lr=lr), late=LATE,
                checks=ck, spec_sha256=T.sha(SPEC), code_sha256=T.sha(Path(__file__)),
                grad_coherence_sha256=T.sha(Path(GC.__file__)), base_sha256=T.sha(Path(C.__file__)),
                host_sha256=T.sha(Path(H.__file__)), data_sha256=mnist.sha256,
                torch_version=torch.__version__, wall_seconds=wall,
                scope='CPU training from init; row-uniform vs per-coordinate gradient noise at equal energy'))
    print('FINISHED', tag, 'G1', ck.get('g1_units_maxabs'), f"({ck.get('g1_units_compared')})", round(wall, 1), 's', flush=True)
    assert wall < TIME_CAP, ('runtime cap', wall)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm'); ap.add_argument('--seed', type=int)
    ap.add_argument('--all', action='store_true'); ap.add_argument('--arms', default=None)
    ap.add_argument('--jobs', type=int, default=8); ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    if a.smoke:
        global LATE
        LATE = (2, 3)
        for arm in (a.arms.split(',') if a.arms else list(ARMS)):
            run(arm, 0, tasks=3, out=SMOKE, controls=True)
        return
    if a.all:
        arms = a.arms.split(',') if a.arms else list(ARMS)

        def job(j):
            subprocess.run([sys.executable, '-m', 'src.grad_floor_0911', '--arm', j[0], '--seed', str(j[1])], check=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as pool:
            for _ in pool.map(job, [(arm, s) for arm in arms for s in range(3)]):
                pass
        return
    run(a.arm, a.seed)


if __name__ == '__main__':
    main()
