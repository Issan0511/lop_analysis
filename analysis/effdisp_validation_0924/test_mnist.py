"""Numerical checks for the fresh MNIST task-boundary runner."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import effdisp_mnist_0924 as M
from src import pmnist_0905 as H


class MnistEngineChecks(unittest.TestCase):
    def _data(self, device):
        g = torch.Generator().manual_seed(2718)
        x = torch.randn(16, 784, generator=g).to(device)
        y = torch.randint(10, (16,), generator=g).to(device)
        return x, y

    def test_manual_gradients_match_torch(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x, y = self._data(device)
        for act in ("LR", "R", "ELU", "GELU", "SiLU", "SN1"):
            params = H.init_params(0, device)
            manual, *_ = M.manual_grads(params, x, y, act)
            refs = [p.detach().clone().requires_grad_(True) for p in params]
            loss = F.cross_entropy(M.forward(refs, x, act)[-1], y)
            trusted = torch.autograd.grad(loss, refs)
            for got, want in zip(manual, trusted):
                torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)

    def test_adaptive_manual_gradients_hold_alpha_fixed(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x, y = self._data(device)
        for name in ("SNA03", "SNA06", "SNA1"):
            act = M.make_act(name, device)
            act.V[0].fill_(0.3)
            act.V[1].fill_(2.0)
            before = [v.clone() for v in act.V]
            params = H.init_params(5, device)
            manual, *_ = M.manual_grads(params, x, y, act)
            ref = [p.detach().clone().requires_grad_(True) for p in params]
            trusted = torch.autograd.grad(F.cross_entropy(M.forward(ref, x, act)[-1], y), ref)
            for got, want in zip(manual, trusted):
                torch.testing.assert_close(got, want, rtol=2e-5, atol=2e-6)
            for got, want in zip(act.V, before):
                torch.testing.assert_close(got, want, rtol=0, atol=0)

    def test_elu_tail_matches_noninplace_autograd(self):
        for device in (torch.device("cpu"), torch.device("cuda")) if torch.cuda.is_available() else (torch.device("cpu"),):
            z = torch.tensor([-25.0, -17.0, -10.0, 0.0, 20.0], device=device,
                             requires_grad=True)
            expected = F.elu(z, alpha=1.0, inplace=False)
            grad = torch.autograd.grad(expected.sum(), z)[0]
            torch.testing.assert_close(M.activation(z, "ELU"), expected, rtol=0, atol=0)
            torch.testing.assert_close(M.derivative(z, "ELU"), grad, rtol=1e-6, atol=1e-12)

    def test_optimizer_and_boundary_alignment(self):
        device = torch.device("cpu")
        x, y = self._data(device)
        for optimizer, lr in (("sgd", 0.01), ("adam", 0.001)):
            params = [p.detach() for p in H.init_params(2, device)]
            ref = [p.detach().clone().requires_grad_(True) for p in params]
            trusted = (torch.optim.SGD(ref, lr=lr) if optimizer == "sgd" else
                       torch.optim.Adam(ref, lr=lr, betas=(0.9, 0.999), eps=1e-8))
            engine = M.Engine(params, "LR", optimizer, lr, device,
                              max_steps=4, graph_steps=1)
            for task in (1, 2):
                engine.reset_acc()
                for _ in range(2):
                    trusted.zero_grad(set_to_none=True)
                    F.cross_entropy(M.forward(ref, x, "LR")[-1], y).backward()
                    trusted.step()
                    engine.run_chunk(x, y)
                self.assertEqual(int(engine.step_t), 2 * task)
                # The state after this task is compared before the next task's update.
                for got, want in zip(params, ref):
                    torch.testing.assert_close(got, want, rtol=2e-5, atol=2e-6)
                self.assertGreater(engine.read_acc(), 0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA graph requires CUDA")
    def test_graph_equals_eager_and_snapshot_is_at_boundary(self):
        device = torch.device("cuda")
        x, y = self._data(device)
        rows_x, rows_y = x.repeat(2, 1), y.repeat(2)
        eager_p = [p.detach() for p in H.init_params(3, device)]
        graph_p = [p.clone() for p in eager_p]
        eager = M.Engine(eager_p, "LR", "adam", 0.001, device, 4, 2)
        graph = M.Engine(graph_p, "LR", "adam", 0.001, device, 4, 2)
        # Force the same Engine's eager path without changing its initial state.
        eager.graph = None
        for task in (1, 2):
            eager.run_chunk(rows_x, rows_y)
            graph.run_chunk(rows_x, rows_y)
            with tempfile.TemporaryDirectory() as tmp:
                M.save_state(Path(tmp), task, graph_p, "adam", graph.moments,
                             int(graph.step_t))
                import numpy as np
                saved = np.load(Path(tmp) / f"state_{task:03d}.npz")
                self.assertEqual(int(saved["step"]), task * 2)
                torch.testing.assert_close(torch.from_numpy(saved["W1"]).to(device),
                                           graph_p[0], rtol=0, atol=0)
            for a, b in zip(eager_p, graph_p):
                torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-6)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA graph requires CUDA")
    def test_adaptive_graph_equals_eager_state(self):
        device = torch.device("cuda")
        x, y = self._data(device)
        rows_x, rows_y = x.repeat(2, 1), y.repeat(2)
        ep = [p.detach() for p in H.init_params(7, device)]
        gp = [p.clone() for p in ep]
        ea, ga = M.make_act("SNA06", device), M.make_act("SNA06", device)
        eager = M.Engine(ep, ea, "adam", 0.001, device, 4, 2)
        graph = M.Engine(gp, ga, "adam", 0.001, device, 4, 2)
        eager.graph = None
        for _ in range(2):
            eager.run_chunk(rows_x, rows_y)
            graph.run_chunk(rows_x, rows_y)
        for got, want in zip(ep + ea.V, gp + ga.V):
            torch.testing.assert_close(got, want, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
