#!/usr/bin/env python3
"""Reviewer's independent GPU checks of arm B (bwdswap_1111), CUDA-graph path, check seeds 100, 101.
Writes review_gpu.json next to this file.  No B accuracy numbers are printed (pre-registration).

RC   prefix task 1 (2 epochs) from the init net; field d = z(init) - z(t1) at layer 2 (donor = init).
     From the t1 state, task 2 for 2 epochs (150 graph replays) with: N, X (E.1 Shift), B (src BwdShift),
     B wrapped in the src Monitor, I (independent: custom Function, forward F.elu(z), backward
     aten.elu_backward at z + d), B0 (BwdShift with d = 0).
F51  task 1 (780 updates) from init; field = build_field(t1 branch, init donor) on task 2's images.
     From t1, task 2 (780 graph replays) with N, X, B, Bmon, I (backward where(z + d >= 0, g, 0) =
     aten clamp_backward), B0 (zero table).
Expected: B == I == Bmon (core state bit for bit, online acc equal), B0 == N, B != N, B != X.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path('/home/issan/Projects/claude/wt/bwdswap_1111')
sys.path.insert(0, str(ROOT))
import torch
import torch.nn.functional as F

from src import bwdswap_1111 as bw

CI, RE, F51 = bw.CI, bw.RE, bw.F51
OUT = Path(__file__).resolve().parent / 'review_gpu.json'
SEEDS = [100, 101]


class IndepEluSwap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, d):
        ctx.save_for_backward(z, d)
        return F.elu(z)

    @staticmethod
    def backward(ctx, g):
        z, d = ctx.saved_tensors
        return torch.ops.aten.elu_backward(g, 1.0, 1.0, 1.0, False, z + d), None


class IndepReluSwap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, d):
        ctx.save_for_backward(z, d)
        return F51.phi(z)

    @staticmethod
    def backward(ctx, g):
        z, d = ctx.saved_tensors
        return torch.where(z + d >= 0, g, torch.zeros((), dtype=g.dtype, device=g.device)), None


class IndepShiftRC(CI.Shift):
    def __call__(self, P, xb, ids):
        W1, b1, W2, b2, W3, b3 = P
        ar = torch.arange(xb.shape[0], device=xb.device)[:, None]
        d = self.d[ar, ids]
        z1 = torch.baddbmm(b1[:, None, :], xb, W1.transpose(1, 2))
        a1 = F.elu(z1)
        z2 = torch.baddbmm(b2[:, None, :], a1, W2.transpose(1, 2))
        a2 = IndepEluSwap.apply(z2, d)
        return z1, a1, z2, a2, torch.baddbmm(b3[:, None, :], a2, W3.transpose(1, 2))


class IndepFieldF51(F51.Field):
    def __call__(self, P, xb, ids):
        W1, b1, W2, b2, W3, b3 = P
        ar = torch.arange(xb.shape[0], device=xb.device)[:, None]
        d = self.d[ar, ids]
        z1 = torch.baddbmm(b1[:, None, :], xb, W1.transpose(1, 2))
        a1 = F51.phi(z1)
        z2 = torch.baddbmm(b2[:, None, :], a1, W2.transpose(1, 2))
        a2 = IndepReluSwap.apply(z2, d)
        return z1, a1, z2, a2, torch.baddbmm(b3[:, None, :], a2, W3.transpose(1, 2))


def rc_part():
    dev = CI.setup()
    X, _ = CI.make_inputs(SEEDS, dev)
    e0 = CI.Engine(SEEDS, X)
    st0 = e0.state()
    e0.train_task(1, 2)
    st1 = e0.state()
    del e0
    d = RE.field_from_states(st1, st0, 2, X)
    P1 = [p.to(dev) for p in st1['P']]
    makers = dict(N=lambda: None, X=lambda: CI.Shift(P1, 2, d), B=lambda: bw.BwdShift(P1, 2, d),
                  Bmon=lambda: bw.Monitor(bw.BwdShift(P1, 2, d), bw.rc_natural, dev),
                  I=lambda: IndepShiftRC(P1, 2, d), B0=lambda: bw.BwdShift(P1, 2, torch.zeros_like(d)))
    res, mon = {}, None
    for k, mk in makers.items():
        sh = mk()
        e = CI.Engine(SEEDS, X, state=st1, shift=sh, graph=True)
        rows, _ = e.train_task(2, 2)
        res[k] = dict(h=CI.tree_hash(CI.core_state(e.state())), online=[r['online_acc'] for r in rows])
        if k == 'Bmon':
            mon = sh.read()
        del e
        torch.cuda.empty_cache()
    return summarize(res, dict(monitor=mon, monitor_ok=bool(mon['graph_calls'] == 150 and mon['graph_mismatch'] == 0
                                                            and mon['eager_mismatch'] == 0)))


def f51_part():
    dev = F51.setup()
    data = F51.Data(dev)
    e0 = F51.Engine(SEEDS, data)
    st0 = e0.state()
    _, b1 = e0.task_batches(1)
    e0.train_steps(b1)
    e0.task = 1
    st1 = e0.state()
    del e0
    rows2 = data.task_rows(SEEDS, 2)
    table, _, _ = F51.build_field(st1, st0, 2, rows2, data)
    P1 = [p.to(dev) for p in st1['P']]
    makers = dict(N=lambda: None, X=lambda: F51.Field(P1, 2, table), B=lambda: bw.BwdField(P1, 2, table),
                  Bmon=lambda: bw.Monitor(bw.BwdField(P1, 2, table), bw.f51_natural, dev),
                  I=lambda: IndepFieldF51(P1, 2, table), B0=lambda: bw.BwdField(P1, 2, torch.zeros_like(table)))
    res, mon = {}, None
    for k, mk in makers.items():
        fld = mk()
        e = F51.Engine(SEEDS, data, state=st1, field=fld, graph=True)
        _, batches = e.task_batches(2)
        e.train_steps(batches)
        res[k] = dict(h=CI.tree_hash(F51.core_state(e.state())), online=[float(e.acc_sum[r]) / F51.STEPS for r in range(len(SEEDS))])
        if k == 'Bmon':
            mon = fld.read()
        del e
        torch.cuda.empty_cache()
    return summarize(res, dict(monitor=mon, monitor_ok=bool(mon['graph_calls'] == F51.STEPS and mon['graph_mismatch'] == 0
                                                            and mon['eager_mismatch'] == 0)))


def summarize(res, extra):
    return dict(B_eq_I_state=res['B']['h'] == res['I']['h'], B_eq_I_online=res['B']['online'] == res['I']['online'],
                B_eq_Bmon_state=res['B']['h'] == res['Bmon']['h'],
                B0_eq_N_state=res['B0']['h'] == res['N']['h'], B0_eq_N_online=res['B0']['online'] == res['N']['online'],
                B_differs_N=res['B']['h'] != res['N']['h'], B_differs_X=res['B']['h'] != res['X']['h'],
                X_differs_N=res['X']['h'] != res['N']['h'], **extra)


if __name__ == '__main__':
    t0 = time.time()
    out = dict(boot_id=bw.boot_id(), RC=rc_part())
    out['F51'] = f51_part()
    out['seconds'] = time.time() - t0
    out['peak_cuda_allocated_gib'] = torch.cuda.max_memory_allocated() / 2 ** 30
    OUT.write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))
