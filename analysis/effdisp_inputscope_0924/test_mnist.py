"""Minimal preflight checks for the input/label MNIST factorial."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import effdisp_inputscope_mnist_0924 as I
from src import effdisp_mnist_0924 as M
from src import pmnist_0905 as H


class InputscopeMnistChecks(unittest.TestCase):
    def test_first_task_shared_and_streams_paired(self):
        p, y = I.schedules(200, 3)
        self.assertTrue(torch.equal(p[0], torch.arange(784)))
        first = [I.actual_task(p, y, cell, 1) for cell in I.CELLS]
        for pp, yy in first:
            self.assertTrue(torch.equal(pp, first[0][0]))
            self.assertTrue(torch.equal(yy, first[0][1]))
        for cell in I.CELLS:
            pp, yy = I.actual_task(p, y, cell, 2)
            self.assertTrue(torch.equal(pp, p[1] if cell[1] == "1" else p[0]))
            self.assertTrue(torch.equal(yy, y[1] if cell[3] == "1" else y[0]))
        p2, y2 = I.schedules(200, 3)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(p, p2)))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(y, y2)))

    def test_covariance_conjugation_and_paired_input(self):
        # Same latent rows at adjacent tasks; permutation changes columns only.
        g = torch.Generator().manual_seed(19)
        x = torch.rand((32, 20), generator=g)
        p = torch.randperm(20, generator=g)
        mean, cov = M.covariance(x)
        next_mean, next_cov = M.covariance(x[:, p])
        np.testing.assert_allclose(next_mean, mean[p.numpy()], rtol=0, atol=1e-14)
        np.testing.assert_allclose(next_cov, cov[np.ix_(p.numpy(), p.numpy())],
                                   rtol=0, atol=1e-14)

    def test_fixed_snake_gradient_matches_autograd(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        act = I.make_act("SN05", device)
        self.assertEqual(act.beta, 0)
        for alpha in (act.alpha(0), act.alpha(1)):
            torch.testing.assert_close(alpha, torch.full_like(alpha, 0.05), rtol=0, atol=0)
        g = torch.Generator().manual_seed(59)
        x = torch.randn((16, 784), generator=g).to(device)
        y = torch.randint(10, (16,), generator=g).to(device)
        params = H.init_params(200, device)
        manual, *_ = M.manual_grads(params, x, y, act)
        ref = [p.detach().clone().requires_grad_(True) for p in params]
        autograd = torch.autograd.grad(F.cross_entropy(M.forward(ref, x, act)[-1], y), ref)
        for a, b in zip(manual, autograd):
            torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(act.V[0], torch.ones_like(act.V[0]), rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA graph requires CUDA")
    def test_fixed_snake_graph_equals_eager(self):
        device = torch.device("cuda")
        g = torch.Generator().manual_seed(73)
        x = torch.randn((32, 784), generator=g).to(device)
        y = torch.randint(10, (32,), generator=g).to(device)
        a = [p.detach() for p in H.init_params(201, device)]
        b = [p.clone() for p in a]
        ae, ag = I.make_act("SN05", device), I.make_act("SN05", device)
        eager = M.Engine(a, ae, "adam", 0.001, device, 4, 2)
        graph = M.Engine(b, ag, "adam", 0.001, device, 4, 2)
        eager.graph = None
        for _ in range(2):
            eager.run_chunk(x, y)
            graph.run_chunk(x, y)
        for got, want in zip(a + ae.V, b + ag.V):
            torch.testing.assert_close(got, want, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
