"""Sub-run D of spec_transport_holes_0910: the driver decomposition for four activations.

    dL/dm_i = sum_x phi'(z_i(x)) e_i(x) S(x),    e_i(x) = dL/da_i(x)

e_i > 0 means "the loss would fall if unit i were LESS active on x"; S(x) >= 0 is
the ink.  Gradient descent moves m_i against the sum, so a POSITIVE sum sinks the
row mean and a negative one lifts it.

The registered sign prediction (spec §4.D), on the NEGATIVE side z <= 0:
    leaky / ELU   phi' > 0, e < 0  ->  product < 0  ->  lift      (must reproduce)
    GELU / SiLU   phi' < 0 past z_c ->  product > 0  ->  sink     (VALLEY_SIGN_FLIPS)
    ReLU          phi' = 0          ->  exactly 0                 (check)

`src/why_down_posthoc_0910.py` is not reused directly: it hard-codes the
elu_growth_0909 anchor, which exists for LR/SNA/ELU1 only and would raise
FileNotFoundError for R, GELU and SiLU *after* the whole loop had run.  The
accumulator layout is kept identical so the two runs' cells files are comparable.
"""
from pathlib import Path
import argparse, concurrent.futures, json, subprocess, sys, time

import numpy as np
import torch

from src import width_sink_clamp_0909 as C
from src import why_down_posthoc_0910 as WD
from src import transport_common_0910 as T

H = C.H
ROOT = C.ROOT
OUT = ROOT / 'results/why_down_acts_0910'
ARMS = ['R', 'ELU1', 'GELU', 'SILU']
WINDOWS = WD.WINDOWS                 # [(21, 25), (96, 100)]
PHASES = WD.PHASES                   # [(1, 20), (21, 100), (101, 625)]
TASKS = 100


def anchor_cnorm(arm, seed, ends, ck):
    """Trajectory check.  Three different anchor shapes, one per arm family; the
    comparison COUNT is recorded so a zero-key match cannot pass as 0.0."""
    if arm == 'ELU1':
        eg = np.load(ROOT / 'results/elu_growth_0909' / f'{arm}_none_s{seed}_units.npz')
        pairs = [(ends[t], eg['cnorm_i'][t - 1]) for t in sorted(ends) if t - 1 < len(eg['cnorm_i'])]
        src = f'elu_growth_0909/{arm}_none_s{seed}_units.npz'
    else:
        path, note = T.anchor_path(arm, seed)
        ref = np.load(path)
        pairs = [(ends[t], ref[f'ref_cnorm_i_t{t}']) for t in sorted(ends)
                 if f'ref_cnorm_i_t{t}' in ref]
        src = f'{path.relative_to(ROOT)}  [{note}]'
    worst = max((float(np.abs(a - b).max()) for a, b in pairs), default=np.inf)
    ck.update(trajectory_maxabs=worst, trajectory_compared=len(pairs),
              trajectory_expected=len(ends), trajectory_anchor=src)
    assert len(pairs) == len(ends), ('trajectory anchor missing tasks', len(pairs), len(ends))
    assert worst <= 1e-10, ('trajectory drifted from the anchor', worst)
    return worst


def run(arm, seed=0, tasks=TASKS, out=OUT):
    torch.set_num_threads(1)
    H.setup('cpu')
    T.data_dir()
    mnist = H.Mnist(torch.device('cpu'))
    act = T.make_act(arm)
    S_all = mnist.train_x.double().sum(1)
    qs = torch.quantile(S_all, torch.tensor([.25, .5, .75], dtype=torch.float64))
    pidx = torch.randperm(len(mnist.test_x), generator=H.stream('boundary_probe', seed))[:512]
    px = mnist.test_x[pidx]
    p = H.init_params(seed, torch.device('cpu'))
    adam = ([torch.zeros_like(q) for q in p], [torch.zeros_like(q) for q in p], [0])
    gp, gd, gb = H.stream('perm', seed), H.stream('data', seed), H.stream('batch', seed)
    nW = len(WINDOWS)
    acc = np.zeros((nW, 3, 2, 4, 2, 3))          # window, phase, side, ink q, correct, (e, eS, n)
    # beyond[window, phase, (sum e*dphi*S over z<z_c, count)] -- the valley-only slice
    beyond = np.zeros((nW, 3, 2))
    force = np.zeros((nW, 3, 2, 2))              # window, phase, side, (sum dphi*e*S, n)
    # mutation control for ReLU's "exactly zero on the negative side": the same sum
    # with the gate shifted by 1e-12 must NOT be zero, so the zero is a property of
    # phi' and not of e or S being empty there.
    force_mut = np.zeros((nW, 3, 2))
    unit, ends, ck = {}, {}, {}
    zc = T.VA.ZC.get(arm)
    t0 = time.monotonic()
    for task in range(1, tasks + 1):
        perm = torch.randperm(784, generator=gp)
        idx = H.stratified_draw(mnist, gd)
        order = torch.randperm(H.TASK_EXAMPLES, generator=gb)
        xs = mnist.train_x[idx][:, perm][order]
        ys = mnist.train_y[idx][order]
        wi = next((k for k, (lo, hi) in enumerate(WINDOWS) if lo <= task <= hi), None)
        if wi is not None:
            with torch.no_grad():
                z0 = px[:, perm].double() @ p[0].detach().double().T + p[1].double()
                pos0 = (z0 > 0).double().mean(0).numpy()
            usum = np.zeros((100, 2, 3, 3))
        for step in range(625):
            xb = xs[step * 16:(step + 1) * 16]
            yb = ys[step * 16:(step + 1) * 16]
            out_ = H.forward(p, xb, act)
            loss = torch.nn.functional.cross_entropy(out_[4], yb)
            if wi is not None:
                gg = torch.autograd.grad(loss, list(p) + [out_[1]])
                gr, ea = gg[:6], gg[6]
                with torch.no_grad():
                    e = ea.double()
                    z1 = out_[0].detach().double()
                    Sx = xb.double().sum(1)
                    g1 = (act.dphi(out_[0].detach(), 0) if isinstance(act, H.AdaptiveSnake)
                          else act.dphi(out_[0].detach())).double()
                    side = (z1 > 0).long()
                    inkq = torch.bucketize(Sx, qs).long()
                    corr = (out_[4].detach().argmax(1) == yb).long()
                    ph = next(k for k, (a_, b_) in enumerate(PHASES) if a_ <= step + 1 <= b_)
                    eS = e * Sx[:, None]
                    feS = g1 * eS                      # the actual force on the row mean
                    fmut = (g1 + 1e-12) * eS
                    for s_ in (0, 1):
                        m = side == s_
                        force[wi, ph, s_] += [float(feS[m].sum()), float(m.sum())]
                        force_mut[wi, ph, s_] += float(fmut[m].sum())
                        for q in range(4):
                            mq = m & (inkq == q)[:, None]
                            for c in (0, 1):
                                mc = mq & (corr == c)[:, None]
                                if mc.any():
                                    acc[wi, ph, s_, q, c] += [float(e[mc].sum()),
                                                              float(eS[mc].sum()), float(mc.sum())]
                        usum[:, s_, ph, 0] += (e * m).sum(0).numpy()
                        usum[:, s_, ph, 1] += (eS * m).sum(0).numpy()
                        usum[:, s_, ph, 2] += m.sum(0).numpy()
                    if zc is not None:
                        mb = z1 < zc
                        beyond[wi, ph] += [float(feS[mb].sum()), float(mb.sum())]
            else:
                gr = torch.autograd.grad(loss, p)
            with torch.no_grad():
                m_, v_, tc = adam
                tc[0] += 1
                c1, c2 = 1 - .9 ** tc[0], 1 - .999 ** tc[0]
                for q, g, mi, vi in zip(p, gr, m_, v_):
                    mi.mul_(.9).add_(g, alpha=1 - .9)
                    vi.mul_(.999).addcmul_(g, g, value=1 - .999)
                    q -= .001 * (mi / c1) / ((vi / c2).sqrt() + 1e-8)
                if isinstance(act, H.AdaptiveSnake):
                    act.update(out_[0], out_[2])
        if wi is not None:
            for i in range(100):
                unit[(task, i)] = dict(task=task, unit=i, pos0=float(pos0[i]),
                                       **{f'{k}_{s}_{ph}': float(usum[i, s, ph, j])
                                          for s in (0, 1) for ph in range(3)
                                          for j, k in enumerate(['e', 'eS', 'n'])})
            with torch.no_grad():
                Wd = p[0].detach().double()
                ends[task] = (Wd - Wd.mean(1, keepdim=True)).norm(dim=1).numpy().copy()
            print(arm, 'task', task, round(time.monotonic() - t0, 1), flush=True)
    anchor_cnorm(arm, seed, ends, ck)

    # --- registered sign check + its mutation control (spec §6)
    neg = float(force[:, 1:, 0, 0].sum())            # negative side, phases 21-100 and 101-625
    ck['neg_side_force'] = neg
    ck['neg_side_force_by_window_phase'] = force[:, :, 0, 0].tolist()
    ck['pos_side_force_by_window_phase'] = force[:, :, 1, 0].tolist()
    ck['beyond_force_by_window_phase'] = beyond[:, :, 0].tolist()
    ck['beyond_count_by_window_phase'] = beyond[:, :, 1].tolist()
    ck['neg_side_force_gate_shifted'] = float(force_mut[:, 1:, 0].sum())
    if arm == 'R':
        assert neg == 0., ('ReLU must contribute exactly zero on the negative side', neg)
        # Not vacuous: the SAME sum with phi' shifted by 1e-12 is non-zero, so the
        # zero comes from phi'(z<=0)=0 and not from an empty negative side.
        assert ck['neg_side_force_gate_shifted'] != 0., \
            ('vacuous ReLU zero: the negative side carries no samples either',
             ck['neg_side_force_gate_shifted'], force[:, :, 0, 1].tolist())

    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f'{arm}_s{seed}_cells.npy', acc)
    np.save(out / f"{arm}_s{seed}_force.npy", force)
    np.save(out / f"{arm}_s{seed}_force_mut.npy", force_mut)
    np.save(out / f'{arm}_s{seed}_beyond.npy', beyond)
    T.write_rows(out / f'{arm}_s{seed}_units.csv', list(unit.values()))
    wall = time.monotonic() - t0
    T.dump(out / f'{arm}_s{seed}_provenance.json',
           dict(arm=arm, seed=seed, tasks=tasks, windows=WINDOWS, phases=PHASES,
                zc=zc, ink_quartiles=qs.tolist(), checks=ck,
                code_sha256=T.sha(Path(__file__)),
                why_down_sha256=T.sha(Path(WD.__file__)),
                wall_seconds=wall,
                scope='sub-run D of spec_transport_holes_0910: the driver sign on the '
                      'negative side, for four activations.',
                **T.provenance_base(mnist)))
    print('FINISHED', arm, 'neg-side force', f'{neg:+.4g}',
          'trajectory', ck['trajectory_maxabs'], round(wall, 1), 's', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', default='R')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    if a.smoke:
        # R, not GELU: the valley anchor is produced by sub-run A, which runs first in
        # the real programme but must not be a precondition for smoking this module.
        run('R', 0, tasks=22, out=ROOT / 'results/_smoke_why_down_acts')
        return
    if a.all:
        def job(arm):
            subprocess.run([sys.executable, '-m', 'src.why_down_acts_0910', '--arm', arm], check=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as pool:
            for _ in pool.map(job, ARMS):
                pass
        return
    run(a.arm, a.seed)


if __name__ == '__main__':
    main()
