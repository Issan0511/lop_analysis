"""unit_triage_0911: gate x update x contribution per unit, on the committed ref / wclamp
branches of width_sink_clamp_0909 (LR / ELU1 / SNA, t21-100).  spec_unit_triage_0911.

The training step is character-identical to the committed C.loop (Adam, the
AdaptiveSnake running variance, then CH.apply_clamp + CH.verify_clamped for wclamp --
the same functions clamp_horizon_0910 proved bit-identical to width_sink_clamp_0909).
Everything else is a no_grad read-out:

  every task end       committed C.measure arrays (G1), gate block (GS.gate_block),
                       mean-ablation dCE_i on the probe, readout proxy r_i
  t61..100 every step  applied centred step of W1 per unit:  S2_i, D2_i, rho_i,
                       Adam normalised step kap2_i, raw gradient graw2_i
  t61 / 80 / 100       mean-ablation dAcc_i on the full test set

G1 = both branches reproduce the committed per-unit arrays bit for bit.
"""
from pathlib import Path
import argparse, concurrent.futures, json, subprocess, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from src import width_sink_clamp_0909 as C
from src import clamp_horizon_0910 as CH
from src import transport_common_0910 as T
from src import gate_shape_0911 as GS

H = C.H
ROOT = C.ROOT
OUT = ROOT / 'results/unit_triage_0911'
SMOKE = ROOT / 'results/_smoke_unit_triage_0911'
REF = ROOT / 'results/width_sink_clamp_0909'
ARMS = ['LR', 'ELU1', 'SNA']
CLAMPS = ['ref', 'wclamp']
TASKS = 100
CLAMP_FROM = 21
LATE = (61, 100)
TEST_TASKS = (61, 80, 100)
LR_, B1, B2, EPS = .001, .9, .999, 1e-8
RECON_STEPS = (1, 200, 625)
TIME_CAP = 1500.
SPEC = ROOT / 'specs/spec_unit_triage_0911.md'
TOL = dict(c3_cnorm_rel=1e-6, c3_m_absdiff=1e-9, clamp_tail_absdiff=0., c6_var_rel=1e-9,
           c6_mean_abs=1e-9, decompose_crosscheck=1e-9)
CONTROL_OF = {'ctl_wclamp_skip_c3_cnorm_rel': 'c3_cnorm_rel',
              'ctl_wclamp_rowscale_c3_m_absdiff': 'c3_m_absdiff',
              'ctl_wclamp_tail_clamp_tail_absdiff': 'clamp_tail_absdiff',
              'ctl_c6_var_rel': 'c6_var_rel', 'ctl_c6_mean_abs': 'c6_mean_abs'}


def phi(act, z, layer):
    return act.phi(z, layer) if isinstance(act, H.AdaptiveSnake) else act.phi(z)


def centered(W):
    Wd = W.detach().double()
    return Wd - Wd.mean(1, keepdim=True)


def ablation(p, act, probe, perm, mnist=None, test=False, zero_col=None):
    """Mean-ablation of each first-layer unit.  Returns dCE (100,) on the probe, the
    all-ablated probe accuracy (control), the readout proxy r_i, and on request the
    test-set dAcc (100,).  `zero_col` zeroes one W2 column first (G5 control: the
    ablation of that unit must then change nothing, exactly)."""
    with torch.no_grad():
        W1, b1, W2, b2, W3, b3 = [q.detach() for q in p]
        if zero_col is not None:
            W2 = W2.clone(); W2[:, zero_col] = 0.

        def head(a1):
            return phi(act, a1 @ W2.T + b2, 1) @ W3.T + b3

        x, y = probe.px[:, perm], probe.py
        a1 = phi(act, x @ W1.T + b1, 0)
        mean1 = a1.mean(0)
        L0 = float(F.cross_entropy(head(a1), y))
        dce = np.zeros(100)
        for i in range(100):
            a = a1.clone(); a[:, i] = mean1[i]
            dce[i] = float(F.cross_entropy(head(a), y)) - L0
        acc_all = float((head(mean1.expand_as(a1).clone()).argmax(1) == y).float().mean())
        out = dict(dce=dce, ce0=L0, acc_all=acc_all,
                   r_i=(W2.double().norm(dim=0) * a1.double().std(0, unbiased=False)).numpy().copy())
        if test:
            xt, yt = mnist.test_x[:, perm], mnist.test_y
            a1t = phi(act, xt @ W1.T + b1, 0)
            m1t = a1t.mean(0)
            acc0 = float((head(a1t).argmax(1) == yt).float().mean())
            dacc = np.zeros(100)
            for i in range(100):
                a = a1t.clone(); a[:, i] = m1t[i]
                dacc[i] = float((head(a).argmax(1) == yt).float().mean()) - acc0
            out.update(dacc=dacc, acc0=acc0)
    return out


def loop(p, act, adam, gens, mnist, probe, t_from, t_to, clamp, base, rows, units, ck, mut=None, instrument=True):
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
            Wt_start = centered(p[0]).clone(); recon, recon_mut = 0., 0.
        ce20 = None
        for step in range(1, 626):
            out = H.forward(p, xs[(step - 1) * 16:step * 16], act)
            gr = torch.autograd.grad(torch.nn.functional.cross_entropy(out[4], ys[(step - 1) * 16:step * 16]), p)
            with torch.no_grad():
                if late:
                    Wb = p[0].detach().double().clone()
                m_, v_, tc = adam
                tc[0] += 1
                c1 = 1 - .9 ** tc[0]; c2 = 1 - .999 ** tc[0]
                for q, g, mi, vi in zip(p, gr, m_, v_):
                    mi.mul_(.9).add_(g, alpha=1 - .9)
                    vi.mul_(.999).addcmul_(g, g, value=1 - .999)
                    q -= .001 * (mi / c1) / ((vi / c2).sqrt() + 1e-8)
                if late:
                    dW_adam = p[0].detach().double() - Wb
                    kap2 += ((dW_adam / LR_) ** 2).mean(1)
                    graw2 += (gr[0].double() ** 2).sum(1)
                    if step in RECON_STEPS:            # G3: the Adam step reconstructed from its state
                        s_rec = (LR_ * (m_[0] / c1) / ((v_[0] / c2).sqrt() + EPS)).double()
                        recon = max(recon, float((dW_adam + s_rec).norm() / s_rec.norm()))
                        # control: a wrong reconstruction (denominator scaled by 1.1) must be caught
                        s_mut = (LR_ * (m_[0] / c1) / ((v_[0] / c2).sqrt() * 1.1 + EPS)).double()
                        recon_mut = max(recon_mut, float((dW_adam + s_mut).norm() / s_mut.norm()))
                if isinstance(act, H.AdaptiveSnake):
                    act.update(out[0], out[2])
                if clamp != 'ref':
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
        abl = ablation(p, act, probe, perm, mnist, test=task in TEST_TASKS)
        r.update(gr_)
        r.update(task=task, step=625, clamp=clamp, ce20=ce20, ce_probe0=abl['ce0'], acc_all_ablated=abl['acc_all'],
                 dce_abs_mean=float(np.abs(abl['dce']).mean()), dce_abs_med=float(np.median(np.abs(abl['dce']))))
        ck['g5_acc_all_max'] = max(ck.get('g5_acc_all_max', 0.), abl['acc_all'])
        if 'dacc' in abl:
            r['acc_test0'] = abl['acc0']
        for k, v in u.items():
            units[f'{clamp}_{k}_t{task}'] = v
        for k, v in gu.items():
            units[f'{clamp}_{k}_t{task}'] = v
        units[f'{clamp}_dce_i_t{task}'] = abl['dce']
        units[f'{clamp}_r_i_t{task}'] = abl['r_i']
        if 'dacc' in abl:
            units[f'{clamp}_dacc_i_t{task}'] = abl['dacc']
        if late:
            D2 = (tot ** 2).sum(1)
            rho = torch.where(S2 > 0, D2 / S2.clamp(min=1e-300), torch.full_like(S2, float('nan')))
            ident = float((tot - (centered(p[0]) - Wt_start)).abs().max())
            ck['g4_tot_ident'] = max(ck.get('g4_tot_ident', 0.), ident)
            ck['g4_rho_max'] = max(ck.get('g4_rho_max', 0.), float(torch.nan_to_num(rho, nan=0.).max()))
            ck['g3_adam_recon'] = max(ck.get('g3_adam_recon', 0.), recon)
            ck['ctl_g3_adam_recon'] = max(ck.get('ctl_g3_adam_recon', 0.), recon_mut)
            units[f'{clamp}_S2_i_t{task}'] = S2.numpy().copy()
            units[f'{clamp}_D2_i_t{task}'] = D2.numpy().copy()
            units[f'{clamp}_rho_i_t{task}'] = rho.numpy().copy()
            units[f'{clamp}_kap2_i_t{task}'] = (kap2 / 625).numpy().copy()
            units[f'{clamp}_graw2_i_t{task}'] = (graw2 / 625).numpy().copy()
            r.update(S2=float(S2.mean()), D2=float(D2.mean()), rho_mean=float(torch.nanmean(rho)),
                     kap2=float(kap2.mean() / 625), graw2=float(graw2.mean() / 625), g4_ident=ident,
                     n_frozen=int((S2 == 0).sum()))
        rows.append(r)
    return p, act, adam


def restore(snap, arm):
    return C.restore(snap, arm)


def _controls(arm, seed, snap, base, mnist, probe, ck_main, clamp_from):
    res = {}

    def floor(key):
        b = 'f32_rowmean_bound_t20' if key in CH.FLOOR_OF else None
        return 100 * CH.limit(ck_main, key, b)

    for mut, keys in [('skip', ['c3_cnorm_rel']), ('rowscale', ['c3_m_absdiff']), ('tail', ['clamp_tail_absdiff'])]:
        p2, act2, adam2, gens2 = restore(snap, arm)
        ck2 = {}
        if mut == 'skip':
            gp, gd, gb = gens2
            perm = torch.randperm(784, generator=gp)
            idx = H.stratified_draw(mnist, gd)
            order = torch.randperm(H.TASK_EXAMPLES, generator=gb)
            xs = mnist.train_x[idx][:, perm][order]
            ys = mnist.train_y[idx][order]
            for step in range(1, 626):
                out = H.forward(p2, xs[(step - 1) * 16:step * 16], act2)
                gr = torch.autograd.grad(torch.nn.functional.cross_entropy(out[4], ys[(step - 1) * 16:step * 16]), p2)
                with torch.no_grad():
                    m_, v_, tc = adam2
                    tc[0] += 1
                    c1, c2 = 1 - .9 ** tc[0], 1 - .999 ** tc[0]
                    for q, g, mi, vi in zip(p2, gr, m_, v_):
                        mi.mul_(.9).add_(g, alpha=1 - .9)
                        vi.mul_(.999).addcmul_(g, g, value=1 - .999)
                        q -= .001 * (mi / c1) / ((vi / c2).sqrt() + 1e-8)
                    if isinstance(act2, H.AdaptiveSnake):
                        act2.update(out[0], out[2])
                    if step != 300:
                        CH.apply_clamp(p2, 'wclamp', base, ck2)
                    CH.verify_clamped(p2, 'wclamp', base, ck2)
        else:
            loop(p2, act2, adam2, gens2, mnist, probe, clamp_from, clamp_from, 'wclamp', base, None, {}, ck2, mut=mut, instrument=False)
        for key in keys:
            got = ck2.get(key, 0.)
            res[f'ctl_wclamp_{mut}_{key}'] = got
            assert got > floor(key), (f'vacuous control wclamp/{mut}: {key} = {got:g} <= {floor(key):g}')

    p4, act4, adam4, gens4 = restore(snap, arm)
    ck4 = {}
    C.measure(p4, act4, probe, probe.refs[0], False, mnist, ck4, mut_decomp=True)
    res['ctl_c6_var_rel'] = ck4['c6_var_rel']; res['ctl_c6_mean_abs'] = ck4['c6_mean_abs']
    assert ck4['c6_var_rel'] > floor('c6_var_rel') and ck4['c6_mean_abs'] > floor('c6_mean_abs'), ('vacuous C6 control', ck4)

    # G5 controls on the t20 state: a zeroed W2 column makes that unit's ablation exactly null;
    # ablating every unit at once collapses the probe accuracy.
    p5, act5, _, _ = restore(snap, arm)
    perm0 = probe.refs[0]
    a0 = ablation(p5, act5, probe, perm0)
    j = int(np.argmax(np.abs(a0['dce'])))
    az = ablation(p5, act5, probe, perm0, zero_col=j)
    res['ctl_g5_zero_col_dce'] = float(abs(az['dce'][j]))
    res['ctl_g5_zero_col_unit'] = j
    res['ctl_g5_zero_col_dce_before'] = float(abs(a0['dce'][j]))
    assert az['dce'][j] == 0.0, ('zeroed-column ablation is not exactly null', az['dce'][j])
    assert abs(a0['dce'][j]) > 0, 'vacuous G5 control: the chosen unit contributed nothing'
    assert a0['acc_all'] < 0.2, ('all-ablated probe accuracy is not at chance', a0['acc_all'])
    res['ctl_g5_acc_all'] = a0['acc_all']

    # G1: a perturbed init must break the reproduction of t1
    p6 = H.init_params(seed, torch.device('cpu'))
    with torch.no_grad():
        p6[0].data[0, 0] += 1e-3
    act6 = C.make_act(arm)
    adam6 = ([torch.zeros_like(q) for q in p6], [torch.zeros_like(q) for q in p6], [0])
    gens6 = [H.stream('perm', seed), H.stream('data', seed), H.stream('batch', seed)]
    u6 = {}
    loop(p6, act6, adam6, gens6, mnist, probe, 1, 1, 'ref', None, [], u6, {}, instrument=False)
    ref_u = np.load(REF / f'{arm}_none_s{seed}_units.npz')
    d = max(float(np.abs(u6[f'ref_{k}_t1'] - ref_u[f'ref_{k}_t1']).max()) for k in T.UNIT_KEYS)
    res['ctl_g1_init'] = d
    assert d > 100 * 1e-10, f'vacuous G1 control: {d:g}'
    return res


def g1_check(arm, seed, units, ck, tasks, clamp_from):
    ref_u = np.load(REF / f'{arm}_none_s{seed}_units.npz')
    worst, n = 0., 0
    for clamp in CLAMPS:
        lo = 1 if clamp == 'ref' else clamp_from
        for t in range(lo, min(tasks, 100) + 1):
            for k in T.UNIT_KEYS:
                key = f'{clamp}_{k}_t{t}'
                if key in units and key in ref_u:
                    worst = max(worst, float(np.abs(units[key] - ref_u[key]).max())); n += 1
    exp = 6 * (min(tasks, 100)) + 6 * (min(tasks, 100) - clamp_from + 1)
    ck.update(g1_units_maxabs=worst, g1_units_compared=n, g1_units_expected=exp,
              g1_anchor=f'width_sink_clamp_0909/{arm}_none_s{seed}_units.npz (registered; ref AND wclamp)')
    return worst, n


def _assert(ck, tasks, controls):
    missing = [k for k in TOL if k not in ck]
    assert not missing, ('checks never recorded', missing)
    bad = {k: (ck[k], CH.limit(ck, k)) for k in TOL if not ck[k] <= CH.limit(ck, k)}
    assert not bad, ('check failed (value, limit)', bad)
    assert ck['g1_units_maxabs'] <= 1e-10, ('G1 failed', ck['g1_units_maxabs'])
    assert ck['g1_units_compared'] == ck['g1_units_expected'], ('G1 count guard', ck['g1_units_compared'], ck['g1_units_expected'])
    if 'g3_adam_recon' in ck:
        assert ck['g3_adam_recon'] <= 1e-4, ('G3 Adam reconstruction', ck['g3_adam_recon'])
        assert ck['ctl_g3_adam_recon'] > 1e-2, ('vacuous G3 control', ck['ctl_g3_adam_recon'])
        assert ck['g4_tot_ident'] <= 1e-10, ('G4 telescoping identity', ck['g4_tot_ident'])
        assert ck['g4_rho_max'] <= 625 * (1 + 1e-9), ('G4 Cauchy-Schwarz', ck['g4_rho_max'])
    assert ck['g5_acc_all_max'] < 0.2, ('G5 all-ablated accuracy', ck['g5_acc_all_max'])
    if controls:
        for cname, k in CONTROL_OF.items():
            assert cname in ck, ('a registered mutation control never ran', cname)
            assert ck[cname] > ck[k], ('control no longer beats the observed value', cname, ck[cname], k, ck[k])
            ck[f'margin_{k}'] = ck[cname] / (100 * ck[k]) if ck[k] > 0 else float('inf')


def run(arm, seed, tasks=TASKS, clamp_from=CLAMP_FROM, out=OUT, controls=True):
    torch.set_num_threads(1); H.setup('cpu'); T.data_dir()
    mnist = H.Mnist(torch.device('cpu'))
    out.mkdir(parents=True, exist_ok=True)
    prefix = f'{arm}_none_s{seed}'
    pidx = torch.randperm(len(mnist.test_x), generator=H.stream('boundary_probe', seed))[:512]
    probe = C.Probe(mnist.test_x[pidx], mnist.test_y[pidx], float(mnist.train_x.mean()))
    t0 = time.monotonic()
    rows, units, ck = [], {}, {}
    ck.update(GS.check_dphi(arm))
    assert ck['g2_dphi'] < 1e-6 and ck['g2_dphi_mutctl'] > 1e-3, ('G2', ck)

    p = H.init_params(seed, torch.device('cpu'))
    act = C.make_act(arm)
    adam = ([torch.zeros_like(q) for q in p], [torch.zeros_like(q) for q in p], [0])
    gens = [H.stream('perm', seed), H.stream('data', seed), H.stream('batch', seed)]
    loop(p, act, adam, gens, mnist, probe, 1, clamp_from - 1, 'ref', None, rows, units, ck)
    base = C.baselines(p[0])
    ck['f32_rowmean_bound_t20'] = CH.EPS32 * float(p[0].detach().double().abs().mean(1).max())
    ck['decompose_crosscheck'] = C.decompose_crosscheck(p, probe)
    ck.update(base_sigma_m=base['sigma_m'], base_mbar0=base['mbar0'], base_cnorm_mean=float(base['c'].mean()))
    snap = C.snapshot(p, act, adam, gens)
    ck['controls_run'] = bool(controls)
    if controls:
        ck.update(_controls(arm, seed, snap, base, mnist, probe, ck, clamp_from))
        print('CONTROLS OK', prefix, round(time.monotonic() - t0, 1), 's', flush=True)

    for clamp in CLAMPS:
        p2, act2, adam2, gens2 = restore(snap, arm)
        loop(p2, act2, adam2, gens2, mnist, probe, clamp_from, tasks, clamp, base, rows, units, ck)
        T.finite_guard([r for r in rows if r['clamp'] == clamp], f'{prefix}/{clamp}')
        print('DONE', prefix, clamp, round(time.monotonic() - t0, 1), 's', flush=True)

    g1_check(arm, seed, units, ck, tasks, clamp_from)
    _assert(ck, tasks, controls)

    for r in rows:
        r.update(arm=arm, iv='none', seed=seed)
    T.write_rows(out / f'{prefix}_rows.csv', rows)
    np.savez_compressed(out / f'{prefix}_units.npz', **units)
    wall = time.monotonic() - t0
    T.dump(out / f'{prefix}_provenance.json',
           dict(arm=arm, iv='none', seed=seed, tasks=tasks, clamp_from=clamp_from, clamps=CLAMPS, late=LATE,
                test_tasks=TEST_TASKS, checks=ck, spec_sha256=T.sha(SPEC), code_sha256=T.sha(Path(__file__)),
                clamp_horizon_sha256=T.sha(Path(CH.__file__)), base_sha256=T.sha(Path(C.__file__)),
                host_sha256=T.sha(Path(H.__file__)), gate_shape_sha256=T.sha(Path(GS.__file__)),
                data_sha256=mnist.sha256, torch_version=torch.__version__, wall_seconds=wall,
                scope='replay of width_sink_clamp_0909 ref/wclamp with read-out instrumentation; G1 both branches'))
    print('FINISHED', prefix, 'G1', ck['g1_units_maxabs'], f"({ck['g1_units_compared']} arrays)", round(wall, 1), 's', flush=True)
    assert wall < TIME_CAP, ('runtime cap', wall)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm'); ap.add_argument('--seed', type=int)
    ap.add_argument('--all', action='store_true'); ap.add_argument('--arms', default=None)
    ap.add_argument('--jobs', type=int, default=6); ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    if a.smoke:
        global LATE, TEST_TASKS
        LATE = (22, 24); TEST_TASKS = (24,)
        run('LR', 0, tasks=24, clamp_from=21, out=SMOKE, controls=True)
        return
    if a.all:
        arms = a.arms.split(',') if a.arms else ARMS

        def job(j):
            subprocess.run([sys.executable, '-m', 'src.unit_triage_0911', '--arm', j[0], '--seed', str(j[1])], check=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as pool:
            for _ in pool.map(job, [(arm, s) for arm in arms for s in range(3)]):
                pass
        return
    run(a.arm, a.seed)


if __name__ == '__main__':
    main()
