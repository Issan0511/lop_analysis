"""noise_anatomy_0911: what does the noise damage?  (spec_noise_anatomy_0911)

Imports grad_coherence_0911 (committed, never edited) for the pre-Adam injector and
its unbiasedness check, and adds:

  schedule      pre-Adam noise on only for tasks [pre_from, pre_to]  (onset / offset arms)
  post modes    noise on the APPLIED W1 step, after Adam:
                  'add'  W1 += r*lr*xi           isotropic, r = 0.2052 (= N2's diffusion energy)
                  'mult' W1 += (c'*xi) * dW      shaped by the step, c' = 0.822 (same energy)
  CE@0          probe CE on the new permutation BEFORE the first update (carry-over state)
  CE@k          probe CE after update 20 / 100 / 300 / 625
  projection    each realised W1 step split into its component along the raw (pre-noise)
                gradient direction (drift) and the orthogonal remainder (diffusion), per row
  layer norms   ||W2||_F, ||W3||_F, ||b2||, ||b3|| at task end

G1: N0 reproduces elu_growth_0909/LR; N2 reproduces grad_coherence_0911/N2 bit for bit
(same generator, same draws -- the extra read-outs must not move the trajectory);
N2off's t1-60 reproduces N2; N2on's t1-60 reproduces N0.
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
OUT = ROOT / 'results/noise_anatomy_0911'
SMOKE = ROOT / 'results/_smoke_noise_anatomy_0911'
TASKS = 120
SWITCH = 61
LATE = (61, 120)
LR0 = .001
CE_STEPS = (20, 100, 300, 625)
TIME_CAP = 900.
SPEC = ROOT / 'specs/spec_noise_anatomy_0911.md'
# derived constants (spec §1): N2's diffusion energy per coordinate, in lr^2 units, is
#   k * c^2 / (1 + c^2),  k = (1 - b1)/(1 + b1)      -> 0.04211 at c = 2
# 'add' reproduces it as an isotropic RMS; 'mult' as a multiple of the clean step energy
# kappa2_0 = 0.0623 (grad_coherence_0911 N0, late window).
K_M = (1 - .9) / (1 + .9)
C_REF = 2.0
DIFF_ENERGY = K_M * C_REF ** 2 / (1 + C_REF ** 2)
KAPPA2_0 = 0.0623
ADD_RMS = math.sqrt(DIFF_ENERGY)                 # 0.2052
MULT_C = math.sqrt(DIFF_ENERGY / KAPPA2_0)       # 0.822

ARMS = {
    'N0':     dict(pre=0.0, pre_from=1,      pre_to=TASKS, post=None),
    'N2':     dict(pre=2.0, pre_from=1,      pre_to=TASKS, post=None),
    'N2on':   dict(pre=2.0, pre_from=SWITCH, pre_to=TASKS, post=None),
    'N2off':  dict(pre=2.0, pre_from=1,      pre_to=SWITCH - 1, post=None),
    'N2post': dict(pre=0.0, pre_from=1,      pre_to=TASKS, post=('add', ADD_RMS)),
    'N2step': dict(pre=0.0, pre_from=1,      pre_to=TASKS, post=('mult', MULT_C)),
}


# ------------------------------------------------------------------ post-Adam noise
def post_noise(dW, mode, amp, lr, gen):
    """Noise to ADD to W1 after the Adam step dW has been applied."""
    xi = torch.randn(dW.shape, generator=gen, dtype=dW.dtype)
    if mode == 'add':
        return (amp * lr) * xi
    return dW * (amp * xi)                      # 'mult'


def check_post(mode, amp, seed, n=2000):
    """G2 for the post modes: realised energy matches the design within 1% (pooled)."""
    gen = H.stream('stepnoise', seed)
    dW = torch.randn((100, 784), generator=H.stream('g2probe', seed)) * (LR0 * math.sqrt(KAPPA2_0))
    acc2 = 0.
    for _ in range(n):
        acc2 += float((post_noise(dW, mode, amp, LR0, gen) ** 2).mean())
    got = acc2 / n / LR0 ** 2                    # energy per coordinate in lr^2 units
    want = amp ** 2 if mode == 'add' else amp ** 2 * float((dW ** 2).mean()) / LR0 ** 2
    return dict(g2_post_energy=got, g2_post_want=want, g2_post_rel=abs(got / want - 1),
                g2_post_target=DIFF_ENERGY, g2_post_vs_target=abs(got / DIFF_ENERGY - 1))


# ------------------------------------------------------------------ training
def probe_ce(p, act, probe, perm):
    with torch.no_grad():
        return float(F.cross_entropy(H.forward(p, probe.px[:, perm], act)[4], probe.py))


def loop(p, act, adam, gens, ngen, sgen, mnist, probe, t_from, t_to, cfg, rows, units, ck, instrument=True):
    pre_c, pre_from, pre_to, post = cfg['pre'], cfg['pre_from'], cfg['pre_to'], cfg['post']
    gp, gd, gb = gens
    lr = LR0
    for task in range(t_from, t_to + 1):
        perm = torch.randperm(784, generator=gp)
        idx = H.stratified_draw(mnist, gd)
        order = torch.randperm(H.TASK_EXAMPLES, generator=gb)
        xs = mnist.train_x[idx][:, perm][order]
        ys = mnist.train_y[idx][order]
        pre_on = pre_c > 0 and pre_from <= task <= pre_to
        ces = {0: probe_ce(p, act, probe, perm)}          # CE@0: before any update on this task
        late = instrument and LATE[0] <= task <= LATE[1]
        if late:
            S2 = torch.zeros(100, dtype=torch.float64); tot = torch.zeros(100, 784, dtype=torch.float64)
            drift2 = torch.zeros(100, dtype=torch.float64); diff2 = torch.zeros(100, dtype=torch.float64)
            graw2 = torch.zeros(100, dtype=torch.float64); kap2 = torch.zeros(100, dtype=torch.float64)
            Wt0 = (lambda W: W - W.mean(1, keepdim=True))(p[0].detach().double()).clone()
        for step in range(1, 626):
            out = H.forward(p, xs[(step - 1) * 16:step * 16], act)
            gr = torch.autograd.grad(torch.nn.functional.cross_entropy(
                out[4], ys[(step - 1) * 16:step * 16]), p)
            with torch.no_grad():
                if late or post is not None:
                    Wb = p[0].detach().double().clone()
                if late:
                    g_raw = gr[0].double().clone()
                    graw2 += (g_raw ** 2).sum(1)
                if pre_on:
                    raw0 = gr[0]
                    gr = list(gr)
                    gr[0] = GC.inject(gr[0], pre_c, ngen)
                    if ck is not None and step % 100 == 0:
                        d = float((gr[0] - raw0).norm()); n0 = float(raw0.norm()) or 1.
                        ck['g3_inj_lo'] = min(ck.get('g3_inj_lo', 9e9), d / n0)
                        ck['g3_inj_hi'] = max(ck.get('g3_inj_hi', 0.), d / n0)
                m_, v_, tc = adam
                tc[0] += 1
                c1 = 1 - .9 ** tc[0]; c2 = 1 - .999 ** tc[0]
                for q, g, mi, vi in zip(p, gr, m_, v_):
                    mi.mul_(.9).add_(g, alpha=1 - .9)
                    vi.mul_(.999).addcmul_(g, g, value=1 - .999)
                    q -= lr * (mi / c1) / ((vi / c2).sqrt() + 1e-8)
                if post is not None:
                    dW = p[0].detach().double() - Wb
                    nz = post_noise(dW, post[0], post[1], lr, sgen)
                    p[0].data.add_(nz.float())
                    if ck is not None and step % 100 == 0:
                        ck['g3_post_rms_lo'] = min(ck.get('g3_post_rms_lo', 9e9), float(nz.pow(2).mean().sqrt()) / lr)
                        ck['g3_post_rms_hi'] = max(ck.get('g3_post_rms_hi', 0.), float(nz.pow(2).mean().sqrt()) / lr)
                if late:
                    dW = p[0].detach().double() - Wb
                    kap2 += ((dW / lr) ** 2).mean(1)
                    dWt = dW - dW.mean(1, keepdim=True)
                    S2 += (dWt ** 2).sum(1); tot += dWt
                    # projection of the realised step onto the raw gradient direction, per row
                    gn = g_raw.norm(dim=1, keepdim=True).clamp(min=1e-300)
                    ghat = g_raw / gn
                    a = (dW * ghat).sum(1, keepdim=True)
                    dr = a * ghat
                    drift2 += (dr ** 2).sum(1); diff2 += ((dW - dr) ** 2).sum(1)
                    ck['g3_proj_ident'] = max(ck.get('g3_proj_ident', 0.),
                                              float((((dr ** 2).sum(1) + ((dW - dr) ** 2).sum(1)) - (dW ** 2).sum(1)).abs().max()
                                                    / max(float((dW ** 2).sum(1).max()), 1e-300)))
            if step in CE_STEPS:
                ces[step] = probe_ce(p, act, probe, perm)
        if rows is None:
            continue
        r, u = C.measure(p, act, probe, perm, True, mnist, ck)
        gr_, gu, _ = GS.gate_block(p, act, probe, perm)
        r.update(gr_)
        r.update(task=task, step=625, clamp='ref', pre_on=int(pre_on), ce0=ces[0], ce20=ces[20],
                 ce100=ces[100], ce300=ces[300], ce625=ces[625])
        with torch.no_grad():
            r.update(w2_fro=float(p[2].double().norm()), w3_fro=float(p[4].double().norm()),
                     b2_norm=float(p[3].double().norm()), b3_norm=float(p[5].double().norm()))
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
            tot_e = drift2 + diff2
            r.update(S2=float(S2.mean()), D2=float(D2.mean()), rho_mean=float(torch.nanmean(rho)),
                     kap2=float(kap2.mean() / 625), graw2=float(graw2.mean() / 625),
                     drift2=float(drift2.mean() / 625 / lr ** 2), diff2=float(diff2.mean() / 625 / lr ** 2),
                     f_diff=float((diff2.sum() / tot_e.sum())))
            units[f'ref_kap2_i_t{task}'] = (kap2 / 625).numpy().copy()
            units[f'ref_S2_i_t{task}'] = S2.numpy().copy()
        rows.append(r)
    return p, act, adam


def run(arm, seed, tasks=TASKS, out=OUT, controls=True, switch=SWITCH):
    torch.set_num_threads(1); H.setup('cpu'); T.data_dir()
    cfg = dict(ARMS[arm])
    if switch != SWITCH:                          # smoke: shift the schedule
        if arm == 'N2on':
            cfg['pre_from'] = switch
        if arm == 'N2off':
            cfg['pre_to'] = switch - 1
    mnist = H.Mnist(torch.device('cpu'))
    out.mkdir(parents=True, exist_ok=True)
    tag = f'{arm}_s{seed}'
    pidx = torch.randperm(len(mnist.test_x), generator=H.stream('boundary_probe', seed))[:512]
    probe = C.Probe(mnist.test_x[pidx], mnist.test_y[pidx], float(mnist.train_x.mean()))
    t0 = time.monotonic()
    rows, units, ck = [], {}, {}
    ck.update(GS.check_dphi('LR'))
    assert ck['g2_dphi'] < 1e-6 and ck['g2_dphi_mutctl'] > 1e-3, ('phi-prime', ck)
    ck.update(cfg=dict(cfg), add_rms=ADD_RMS, mult_c=MULT_C, diff_energy_target=DIFF_ENERGY)
    if cfg['pre'] > 0 and controls:
        ck.update(GC.check_unbiased(cfg['pre'], seed))
        assert ck['g2_mean_rel'] < ck['g2_mean_tol'] and ck['g2_var_rel'] < ck['g2_var_tol'], ('G2 pre', ck)
        assert ck['g2_mean_rel_mutctl'] > 3 * ck['g2_mean_tol'], ('vacuous G2 control', ck)
    if cfg['post'] is not None and controls:
        ck.update(check_post(cfg['post'][0], cfg['post'][1], seed))
        assert ck['g2_post_rel'] < 0.01, ('G2 post energy vs design', ck['g2_post_rel'])
        assert ck['g2_post_vs_target'] < 0.02, ('G2 post energy vs N2 target', ck['g2_post_vs_target'])

    p = H.init_params(seed, torch.device('cpu'))
    act = GS.make_act('LR')
    adam = ([torch.zeros_like(q) for q in p], [torch.zeros_like(q) for q in p], [0])
    gens = [H.stream('perm', seed), H.stream('data', seed), H.stream('batch', seed)]
    ngen = H.stream('gradnoise', seed)
    sgen = H.stream('stepnoise', seed)
    st0 = [g.get_state().clone() for g in gens]
    n_state = {'start': ngen.get_state().clone()}
    # G4: the gradnoise generator must be untouched across every c = 0 interval
    if arm == 'N2on':
        loop(p, act, adam, gens, ngen, sgen, mnist, probe, 1, cfg['pre_from'] - 1, cfg, rows, units, ck)
        n_state['pre_switch'] = ngen.get_state().clone()
        loop(p, act, adam, gens, ngen, sgen, mnist, probe, cfg['pre_from'], tasks, cfg, rows, units, ck)
        ck['g4_ngen_untouched_before_switch'] = bool(torch.equal(n_state['start'], n_state['pre_switch']))
        assert ck['g4_ngen_untouched_before_switch']
    elif arm == 'N2off':
        loop(p, act, adam, gens, ngen, sgen, mnist, probe, 1, cfg['pre_to'], cfg, rows, units, ck)
        n_state['post_switch'] = ngen.get_state().clone()
        loop(p, act, adam, gens, ngen, sgen, mnist, probe, cfg['pre_to'] + 1, tasks, cfg, rows, units, ck)
        ck['g4_ngen_untouched_after_switch'] = bool(torch.equal(n_state['post_switch'], ngen.get_state()))
        assert ck['g4_ngen_untouched_after_switch']
    else:
        loop(p, act, adam, gens, ngen, sgen, mnist, probe, 1, tasks, cfg, rows, units, ck)
        if cfg['pre'] == 0:
            ck['g4_ngen_untouched'] = bool(torch.equal(n_state['start'], ngen.get_state()))
            assert ck['g4_ngen_untouched']
    T.finite_guard(rows, tag)
    g2 = [H.stream('perm', seed), H.stream('data', seed), H.stream('batch', seed)]
    ck['g3_streams_intact'] = all(torch.equal(a, b.get_state()) for a, b in zip(st0, g2))
    assert ck['g3_streams_intact']
    if cfg['pre'] > 0:
        assert 0.5 * cfg['pre'] < ck['g3_inj_lo'] and ck['g3_inj_hi'] < 2.0 * cfg['pre'], ('injection ~c', ck)
    if cfg['post'] is not None:
        # realised per-step RMS (in lr units) of the added noise: 'add' is exactly ADD_RMS in
        # expectation; 'mult' scales with the step so only a loose band is asserted here
        lo, hi = ck['g3_post_rms_lo'], ck['g3_post_rms_hi']
        if cfg['post'][0] == 'add':
            assert 0.9 * ADD_RMS < lo and hi < 1.1 * ADD_RMS, ('post add rms', lo, hi)
        else:
            assert lo > 0.02 and hi < 2.0, ('post mult rms', lo, hi)
    assert ck.get('g3_proj_ident', 0.) <= 1e-10, ('projection identity', ck['g3_proj_ident'])

    # ------------------------------------------------------------- G1 anchors
    def cmp_eg(src, t_to):
        eg = np.load(ROOT / 'results/elu_growth_0909' / f'{src}_none_s{seed}_units.npz')
        w, wc, n = 0., 0., 0
        for t in range(1, min(tasks, t_to) + 1):
            for k in ('zbar_i', 'sd_i', 'cnorm_i'):
                key = f'ref_{k}_t{t}'
                if key in units:
                    d = float(np.abs(units[key] - eg[k][t - 1]).max()); w = max(w, d); n += 1
                    if k == 'cnorm_i':
                        wc = max(wc, d)
        return w, wc, n, 3 * min(tasks, t_to)

    def cmp_gc(t_to):
        ref = np.load(ROOT / 'results/grad_coherence_0911' / f'N2_s{seed}_units.npz')
        w, wc, n = 0., 0., 0
        for key in units:
            t = int(key.rsplit('_t', 1)[1])
            if t <= min(tasks, t_to) and key in ref:
                d = float(np.abs(units[key].astype(float) - ref[key].astype(float)).max()); w = max(w, d); n += 1
                if '_cnorm_i_' in key:
                    wc = max(wc, d)
        return w, wc, n, None

    anchor = {'N0': ('eg', 'LR', tasks), 'N2on': ('eg', 'LR', min(tasks, cfg['pre_from'] - 1)),
              'N2': ('gc', None, tasks), 'N2off': ('gc', None, cfg['pre_to'])}.get(arm)
    if anchor:
        kind, src, t_to = anchor
        w, wc, n, exp = cmp_eg(src, t_to) if kind == 'eg' else cmp_gc(t_to)
        ck.update(g1_anchor=f"{'elu_growth_0909/LR' if kind == 'eg' else 'grad_coherence_0911/N2'} t1-{t_to}",
                  g1_units_maxabs=w, g1_cnorm_maxabs=wc, g1_units_compared=n, g1_units_expected=exp)
        assert w <= 1e-10, ('G1 failed', arm, seed, w)
        assert n > 0 and (exp is None or n == exp), ('G1 count guard', n, exp)
        if kind == 'gc':
            assert n >= 6 * min(tasks, t_to), ('G1 vs grad_coherence compared too few arrays', n)
        if controls and seed == 0 and kind == 'eg':
            p3 = H.init_params(seed, torch.device('cpu'))
            with torch.no_grad():
                p3[0].data[0, 0] += 1e-3
            ad3 = ([torch.zeros_like(q) for q in p3], [torch.zeros_like(q) for q in p3], [0])
            g3 = [H.stream('perm', seed), H.stream('data', seed), H.stream('batch', seed)]
            u3 = {}
            loop(p3, GS.make_act('LR'), ad3, g3, H.stream('gradnoise', seed), H.stream('stepnoise', seed),
                 mnist, probe, 1, 1, dict(ARMS['N0']), [], u3, {}, instrument=False)
            eg = np.load(ROOT / 'results/elu_growth_0909' / f'LR_none_s{seed}_units.npz')
            ck['ctl_g1_init'] = max(float(np.abs(u3[f'ref_{k}_t1'] - eg[k][0]).max()) for k in ('zbar_i', 'sd_i', 'cnorm_i'))
            assert ck['ctl_g1_init'] > 1e-8, ('vacuous G1 control', ck['ctl_g1_init'])
    else:
        ck.update(g1_anchor='none (post-Adam noise arm)', g1_units_maxabs=None, g1_units_compared=0)
    assert ck.get('g5_rho_max', 0.) <= 625 * (1 + 1e-9), ('G5 Cauchy-Schwarz', ck['g5_rho_max'])
    assert ck.get('g5_tot_ident', 0.) <= 1e-10, ('G5 telescoping', ck['g5_tot_ident'])

    for r in rows:
        r.update(arm=arm, seed=seed)
    T.write_rows(out / f'{tag}_rows.csv', rows)
    np.savez_compressed(out / f'{tag}_units.npz', **units)
    wall = time.monotonic() - t0
    T.dump(out / f'{tag}_provenance.json',
           dict(arm=arm, seed=seed, tasks=tasks, cfg=cfg, switch=switch, late=LATE, checks=ck,
                spec_sha256=T.sha(SPEC), code_sha256=T.sha(Path(__file__)),
                grad_coherence_sha256=T.sha(Path(GC.__file__)), base_sha256=T.sha(Path(C.__file__)),
                host_sha256=T.sha(Path(H.__file__)), gate_shape_sha256=T.sha(Path(GS.__file__)),
                data_sha256=mnist.sha256, torch_version=torch.__version__, wall_seconds=wall,
                scope='CPU training from init; noise schedule / post-Adam noise; CE@0 and step projection'))
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
        LATE = (3, 4)
        for arm in (a.arms.split(',') if a.arms else list(ARMS)):
            run(arm, 0, tasks=4, out=SMOKE, controls=True, switch=3)
        return
    if a.all:
        arms = a.arms.split(',') if a.arms else list(ARMS)

        def job(j):
            subprocess.run([sys.executable, '-m', 'src.noise_anatomy_0911', '--arm', j[0], '--seed', str(j[1])], check=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as pool:
            for _ in pool.map(job, [(arm, s) for arm in arms for s in range(3)]):
                pass
        return
    run(a.arm, a.seed)


if __name__ == '__main__':
    main()
