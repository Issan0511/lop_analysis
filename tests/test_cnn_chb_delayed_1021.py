"""CPU-only independent regressions for natural delayed B.

Run: python -m unittest discover -s tests -p test_cnn_response_chb_1021.py -v
No CIFAR files, archived checkpoints, CUDA contexts, or intervention data are read.
"""
from __future__ import annotations

import copy
import io
import math
import random
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import cnn_chb_delayed_1021 as M
from src import cnn_center_feat_1011 as CF
from src import sna_cnn_cause_1009 as E


def equal(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(
        torch.isnan(a) if a.is_floating_point() else torch.zeros_like(a, dtype=torch.bool),
        torch.isnan(b) if b.is_floating_point() else torch.zeros_like(b, dtype=torch.bool)
    ) and bool(torch.all((a == b) | (torch.isnan(a) & torch.isnan(b))))


def make_network(seed=10, runs=2):
    P = E.stack_init(list(range(seed, seed + runs)), torch.device("cpu"))
    act = M.S.ResponseDoors(runs, "feat", torch.device("cpu"))
    g = torch.Generator().manual_seed(303 + seed)
    with torch.no_grad():
        for m in act.m:
            m.copy_(torch.rand(m.shape, generator=g) * .05)
    return P, act


def fake_data(n=1200):
    """Full production architecture with synthetic images and frozen source states."""
    sources = {}
    for arm, seed in (("CHf", 10), ("CHf_B", 20)):
        P, act = make_network(seed, runs=1)
        sources[arm] = dict(P=[p.detach().clone() for p in P],
                            m=[torch.zeros_like(p) for p in P],
                            v=[torch.zeros_like(p) for p in P], tc=900000.,
                            act=act.state(), bad=torch.zeros(1, dtype=torch.bool),
                            g_lab={10: torch.Generator().manual_seed(1021).get_state()},
                            g_batch={10: torch.Generator().manual_seed(1022).get_state()},
                            kappa=.2, enabled=arm == "CHf_B")
    g = torch.Generator().manual_seed(1023)
    X = torch.randn(1, n, 3, 32, 32, generator=g) * .2
    return SimpleNamespace(seeds=[10], device=torch.device("cpu"), sources=sources,
                           cifar=None, inputs=lambda _: X, Xcenter=X,
                           identity={"synthetic": True},
                           ids=torch.arange(n)[None, :])


@torch.no_grad()
def independent_projection(P, xb, act):
    """Separate sequential implementation; direct ReLU difference is safe here."""
    R, B = xb.shape[:2]
    x = xb.transpose(0, 1).reshape(B, R * 3, 32, 32)
    widths, activities = [], []
    for l in range(4):
        w, b = P[2*l:2*l+2]
        if l < 2:
            q = F.conv2d(x, w.reshape(R * 16, -1, 5, 5), padding=2, groups=R)
            layout = q.reshape(B, R, 16, -1).permute(1, 0, 2, 3).double()
            width = layout.var(1, correction=0).mean(-1).sqrt()
        else:
            q = torch.bmm(x, w.transpose(1, 2))
            width = q.double().var(1, correction=0).sqrt()
        cap = .2 * width
        b.copy_(torch.where(b.double().abs() > cap, b.double().clamp(-cap, cap).to(b.dtype), b))
        z = q + (b.reshape(1, R*16, 1, 1) if l < 2 else b[:, None, :])
        a = z.clamp(min=0)
        widths.append(width)
        activities.append(a)
        if l < 2:
            x = F.max_pool2d(a, 2) - act.sub_map(l, act.m)
            if l == 1:
                x = x.reshape(B, R, 1024).transpose(0, 1)
        else:
            x = a - act.m[l][:, None, :]
    return widths, activities


class DelayedChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_initial_state_and_no_initial_projection(self):
        data = fake_data(n=16)
        with torch.no_grad():
            for l in range(4): data.sources['CHf']['P'][2*l+1].fill_(-10.)
        a, b = [M.Engine(data, arm, False) for arm in M.MAIN_ARMS]
        a.new_labels(); b.new_labels(); a.new_order(); b.new_order()
        self.assertIsNone(a.field); self.assertIsNone(b.field)
        sa, sb = a.full_state(), b.full_state()
        sa.pop('enabled'); sb.pop('enabled')
        self.assertEqual(M.tree_hash(sa), M.tree_hash(sb))
        self.assertEqual(M.tree_hash(a.forward_ids(a.order)), M.tree_hash(b.forward_ids(b.order)))
        a.step(0); b.step(0)
        for key in ('m', 'v'):
            self.assertEqual(M.tree_hash(getattr(a,key)), M.tree_hash(getattr(b,key)))
        self.assertEqual(M.tree_hash(a.act.state()), M.tree_hash(b.act.state()))
        for l in range(5): self.assertTrue(equal(a.P[2*l], b.P[2*l]))
        self.assertFalse(equal(a.P[1], b.P[1]))
        self.assertTrue(equal(a.P[-1], b.P[-1]))
        self.assertTrue(torch.equal(a.acc_steps, b.acc_steps))
        self.assertTrue(torch.equal(a.ce_steps, b.ce_steps))

    def test_disabled_B_original_natural_and_replay(self):
        data = fake_data(n=16)
        a, b = [M.Engine(data, 'CH_continue', False) for _ in range(2)]
        for e in (a,b): e.new_labels(); e.new_order()
        initial = a.snapshot()
        a.step(0); CF.Bundle.step(b,0)
        self.assertEqual(M.tree_hash(a.full_state()), M.tree_hash(b.full_state()))
        end = a.snapshot()
        stream = io.BytesIO(); torch.save(initial, stream); stream.seek(0)
        a.restore(torch.load(stream, weights_only=False)); a.step(0)
        self.assertEqual(M.tree_hash(end), M.tree_hash(a.snapshot()))

    def test_independent_sequential_projection_and_immutable_adam(self):
        data = fake_data(n=16)
        e = M.Engine(data, 'CH_addB', False)
        with torch.no_grad():
            for l in range(4): e.P[2*l+1].fill_(10. if l%2 else -10.)
        state = M.tree_hash(dict(m=e.m,v=e.v,act=e.act.state(),head=e.P[-2:]))
        ref = [p.detach().clone() for p in e.P]
        widths, _ = independent_projection(ref, e.X, e.act)
        rows = M.PARENT.project_bias(e.P, e.X, e.act)
        self.assertEqual(state,M.tree_hash(dict(m=e.m,v=e.v,act=e.act.state(),head=e.P[-2:])))
        for l in range(4):
            torch.testing.assert_close(rows[l]['s'],widths[l],rtol=1e-12,atol=1e-12)
            torch.testing.assert_close(e.P[2*l+1],ref[2*l+1],rtol=1e-6,atol=1e-7)
            self.assertTrue(rows[l]['hit'].any())

    def test_projection_zero_width_and_in_bound_bits(self):
        b=torch.tensor([[0.,-0.,-3.,4.]])
        width=torch.tensor([[0.,0.,0.,100.]],dtype=torch.float64)
        before,hit=M.PARENT.clip_bias(b,width)
        self.assertTrue(torch.equal(hit,torch.tensor([[False,False,True,False]])))
        self.assertTrue(torch.signbit(b[0,1]))
        self.assertEqual(b[0,2],0.)
        self.assertEqual(b[0,3],before[0,3])

    def test_statistics_discordant_means_ties_and_failure(self):
        from src.cnn_chb_delayed_1021 import evaluate_primary
        table={a:{'accuracy':np.zeros(10),'ce':np.ones(10)} for a in M.MAIN_ARMS}
        table['CH_addB']['accuracy'][:]=.2
        table['CH_addB']['ce'][:]=2.
        r=evaluate_primary(table)
        self.assertEqual(r['accuracy']['verdict'],'IMPROVES')
        self.assertEqual(r['ce']['verdict'],'WORSENS')
        self.assertEqual(r['accuracy']['p_holm'],.00390625)
        table['CH_addB']['ce'][:]=np.inf
        table['CH_continue']['ce'][:]=np.inf
        r=evaluate_primary(table)
        self.assertEqual(r['ce']['n_tie'],10)
        self.assertEqual(r['ce']['verdict'],'INCONCLUSIVE_NONFINITE')
        self.assertIsNone(r['ce']['mean_benefit'])

if __name__ == '__main__': unittest.main()
