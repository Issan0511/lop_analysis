"""Small semantic checks for the paired CondA trajectory runner."""
import tempfile
import unittest
import json
from pathlib import Path

import numpy as np
import torch

from src import effdisp_conda_0924 as E


class CondATests(unittest.TestCase):
    def test_manual_gradient_and_adam(self):
        arms = (("k1", 1, "adam", 1e-3, 1e-8, 5, "leaky"),)
        P, target, flip = E.initialize(arms, (100,), "cpu")
        x = torch.cat((flip[:, :E.M-5], torch.tensor([[0., 1., 0., 1., 1.]])), dim=1)
        y = E.teacher_output(x, target)
        manual = E.gradients(P, x, y)
        Q = {k: v.clone().detach().requires_grad_(True) for k, v in P.items()}
        pre = torch.bmm(Q["W"], x.unsqueeze(-1)).squeeze(-1) + Q["b"]
        a = torch.where(pre > 0, pre, 0.1 * pre)
        loss = (((a * Q["v"]).sum(-1) + Q["c"] - y) ** 2).sum()
        loss.backward()
        for k in P:
            torch.testing.assert_close(manual[k], Q[k].grad, rtol=1e-5, atol=1e-6)
        torch_opt = torch.optim.Adam(list(Q.values()), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
        our_opt = E.Optimizer(P, arms, 1)
        for _ in range(3):
            torch_opt.zero_grad()
            pre = torch.bmm(Q["W"], x.unsqueeze(-1)).squeeze(-1) + Q["b"]
            a = torch.where(pre > 0, pre, 0.1 * pre)
            (((a * Q["v"]).sum(-1) + Q["c"] - y) ** 2).sum().backward()
            torch_opt.step()
            our_opt.step(E.gradients(P, x, y))
            for k in P:
                torch.testing.assert_close(P[k], Q[k], rtol=1e-4, atol=1e-6)

    def test_distinct_nested_flips(self):
        arms = (("k0", 0, "adam", 1e-3, 1e-8, 5, "leaky"),
                ("k1", 1, "adam", 1e-3, 1e-8, 5, "leaky"),
                ("k3", 3, "adam", 1e-3, 1e-8, 5, "leaky"),
                ("k7", 7, "adam", 1e-3, 1e-8, 5, "leaky"))
        _, _, flip = E.initialize(arms, (100, 101), "cpu")
        before = flip.clone()
        E.apply_flip(flip, arms, (100, 101), 1)
        changed = before != flip
        self.assertEqual(changed.sum(1).tolist(), [0, 0, 1, 1, 3, 3, 7, 7])
        for s in range(2):
            self.assertTrue(set(torch.where(changed[2 + s])[0].tolist()) <=
                            set(torch.where(changed[4 + s])[0].tolist()))
            self.assertTrue(set(torch.where(changed[4 + s])[0].tolist()) <=
                            set(torch.where(changed[6 + s])[0].tolist()))

    def test_support_metrics_and_boundary(self):
        arms = (("k0", 0, "sgd", 0.01, 0, 5, "leaky"),
                ("k1", 1, "sgd", 0.01, 0, 5, "leaky"))
        with tempfile.TemporaryDirectory() as tmp:
            E.run_group(Path(tmp), arms, (100,), 2, 2, "cpu", graph=False)
            with np.load(Path(tmp) / "k1/seed100/t001.npz") as first, \
                 np.load(Path(tmp) / "k1/seed100/t002.npz") as second, \
                 np.load(Path(tmp) / "k1/seed100/t000.npz") as initial, \
                 np.load(Path(tmp) / "k0/seed100/t002.npz") as static:
                self.assertEqual(int(first["step"]), 2)
                self.assertEqual(int(second["step"]), 4)
                self.assertEqual(int(np.count_nonzero(first["flip_state"] != second["flip_state"])), 1)
                self.assertEqual(int(np.count_nonzero(first["flip_state"] != static["flip_state"])), 0)
                self.assertEqual(int(np.count_nonzero(second["flip_state"] != static["flip_state"])), 1)
                expected = 0.25 * np.square(second["W"][:, E.F:]).sum()
                self.assertAlmostEqual(float(second["w_sigma2"]), float(expected), places=4)
                self.assertEqual(float(first["task_start_mse"]), float(initial["mse"]))
                self.assertTrue(np.isfinite(float(second["task_start_mse"])))

    def test_activation_derivatives(self):
        z = torch.tensor([[-2.0, -0.4, 0.3, 1.7]], dtype=torch.float64,
                         requires_grad=True)
        for name in ("leaky", "relu", "elu", "gelu", "silu", "snake"):
            a, derivative = E._activation(z, name)
            actual = torch.autograd.grad(a.sum(), z, retain_graph=True)[0]
            torch.testing.assert_close(derivative, actual, rtol=1e-10, atol=1e-10)

    def test_elu_float32_tail_matches_torch_autograd(self):
        z = torch.tensor([[-25., -17., -10., 0., 20.]], requires_grad=True)
        a, derivative = E._activation(z, "elu")
        expected = torch.nn.functional.elu(z, alpha=1.0, inplace=False)
        actual = torch.autograd.grad(expected.sum(), z)[0]
        torch.testing.assert_close(a, expected, rtol=0, atol=0)
        torch.testing.assert_close(derivative, actual, rtol=0, atol=0)
        self.assertGreater(float(derivative[0, 0].detach()), 0.0)

    def test_adaptive_snake_frozen_alpha_and_preupdate_ema(self):
        z = torch.tensor([[-3., -1., 0., 2.]], requires_grad=True)
        alpha = torch.tensor([[0.6, 0.4, 1.2, 2.0]])
        a, derivative = E._activation(z, "sna06", alpha)
        actual = torch.autograd.grad(a.sum(), z)[0]
        torch.testing.assert_close(derivative, actual, rtol=1e-6, atol=1e-6)
        arms = (("adaptive", 1, "adam", 1e-3, 1e-8, 5, "sna06"),)
        with tempfile.TemporaryDirectory() as tmp:
            E.run_group(Path(tmp), arms, (100,), 1, 1, "cpu", graph=False)
            with np.load(Path(tmp) / "adaptive/seed100/t000.npz") as before, \
                 np.load(Path(tmp) / "adaptive/seed100/t001.npz") as after:
                var = 0.25 * np.square(before["W"][:, 15:]).sum(1)
                expected = 0.99 + 0.01 * var
                np.testing.assert_allclose(after["VEMA"], expected, rtol=1e-6, atol=1e-6)
                np.testing.assert_allclose(after["alpha"],
                                           np.clip(.6 / np.sqrt(expected), .05, 3),
                                           rtol=1e-6, atol=1e-6)

    def test_m40_shapes_and_support(self):
        arms = (("m40_test", 2, "adam", 1e-3, 1e-8, 10, "sna06"),)
        with tempfile.TemporaryDirectory() as tmp:
            E.run_group(Path(tmp), arms, (100,), 2, 2, "cpu", graph=False, m=40)
            with np.load(Path(tmp) / "m40_test/seed100/t002.npz") as z:
                self.assertEqual(z["W"].shape, (100, 40))
                self.assertEqual(z["flip_state"].shape, (30,))
                self.assertEqual(int(z["m"]), 40)
                self.assertEqual(int(z["r"]), 10)
                self.assertEqual(z["VEMA"].shape, (100,))
                self.assertTrue(np.isfinite(float(z["mse"])))

    def test_divergence_is_per_case_and_persistent(self):
        arms = (("bad", 1, "adam", 1e-3, 1e-8, 5, "leaky"),
                ("good", 1, "adam", 1e-3, 1e-8, 5, "leaky"))
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            E.run_group(out, arms, (100,), 3, 3, "cpu", graph=False,
                        inject_nonfinite={("bad", 100, 1)})
            clean = out / "clean"
            E.run_group(clean, arms, (100,), 3, 3, "cpu", graph=False)
            bad = json.loads((out / "bad/seed100/status.json").read_text())
            good = json.loads((out / "good/seed100/status.json").read_text())
            self.assertEqual(bad["status"], "DIVERGED")
            self.assertEqual(bad["first_bad_task"], 1)
            self.assertEqual(good["status"], "COMPLETED")
            self.assertEqual(sorted(p.name for p in (out / "bad/seed100").glob("t*.npz")),
                             ["t000.npz"])
            self.assertEqual(len(list((out / "good/seed100").glob("t*.npz"))), 4)
            with np.load(out / "good/seed100/t003.npz") as actual, \
                 np.load(clean / "good/seed100/t003.npz") as reference:
                for key in ("W", "b", "v", "c"):
                    np.testing.assert_array_equal(actual[key], reference[key])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_graph_matches_eager_across_flip(self):
        arms = (("k0", 0, "adam", 1e-3, 1e-8, 5, "leaky"),
                ("k1", 1, "adam", 1e-3, 1e-8, 5, "leaky"),
                ("k1_elu", 1, "adam", 1e-3, 1e-8, 5, "elu"),
                ("k1_sna06", 1, "adam", 1e-3, 1e-8, 5, "sna06"))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a, b = root / "graph", root / "eager"
            E.run_group(a, arms, (100, 101), 11, 3, "cuda", graph=True)
            E.run_group(b, arms, (100, 101), 11, 3, "cuda", graph=False)
            for pa in a.rglob("t*.npz"):
                with np.load(pa) as za, np.load(b / pa.relative_to(a)) as zb:
                    self.assertEqual(set(za.files), set(zb.files))
                    for key in za.files:
                        np.testing.assert_array_equal(za[key], zb[key],
                                                      err_msg=f"{pa.relative_to(a)}:{key}")
            arm40 = (("m40_test", 2, "adam", 1e-3, 1e-8, 10, "sna06"),)
            E.run_group(a / "m40", arm40, (100,), 7, 3, "cuda", graph=True, m=40)
            E.run_group(b / "m40", arm40, (100,), 7, 3, "cuda", graph=False, m=40)
            for pa in (a / "m40").rglob("t*.npz"):
                with np.load(pa) as za, np.load(b / pa.relative_to(a)) as zb:
                    for key in za.files:
                        np.testing.assert_array_equal(za[key], zb[key],
                                                      err_msg=f"{pa.relative_to(a)}:{key}")


if __name__ == "__main__":
    unittest.main()
