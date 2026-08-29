import copy
import unittest

import numpy as np
import torch

from src.common import load_config
from src.mlp2_phase0 import (bootstrap_t, exact_layer_record, setup_arm,
                             spearman)
from src.nets import VecMLP, VecMLPL


class VecMLPLTests(unittest.TestCase):
    def test_l1_is_bit_identical_to_vecmlp(self):
        g_old = torch.Generator().manual_seed(10100)
        g_new = torch.Generator().manual_seed(10100)
        old = VecMLP(3, 10, 5, g_old, "cpu")
        new = VecMLPL(3, [10], 5, g_new, "cpu")
        self.assertTrue(all(torch.equal(old.state_dict()[k], new.state_dict()[k])
                            for k in old.state_dict()))

        data = torch.Generator().manual_seed(7)
        lr = torch.full((3,), 0.01)
        for _ in range(50):
            x = torch.randint(0, 2, (3, 5), generator=data).float()
            target = torch.randn(3, generator=data)
            po, ao, yo = old.forward(x)
            pn, an, yn = new.forward(x)
            self.assertTrue(torch.equal(po, pn))
            self.assertTrue(torch.equal(ao, an))
            self.assertTrue(torch.equal(yo, yn))
            go = old.grads(x, po, ao, yo - target)
            gn = new.grads(x, pn, an, yn - target)
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(go, gn)))
            old.sgd_step(lr, *go)
            new.sgd_step(lr, *gn)
        self.assertTrue(all(torch.equal(old.state_dict()[k], new.state_dict()[k])
                            for k in old.state_dict()))

    def test_l2_closed_form_gradient_matches_autograd(self):
        gen = torch.Generator().manual_seed(13)
        net = VecMLPL(2, [4, 3], 5, gen, "cpu")
        x = torch.randn(2, 5, generator=gen)
        target = torch.randn(2, generator=gen)
        pre, act, yhat = net.forward_layers(x)
        gW, gb, gv, gc = net.grads_layers(x, pre, act, yhat - target)

        for W, b in zip(net.Ws, net.bs):
            W.requires_grad_()
            b.requires_grad_()
        net.v.requires_grad_()
        net.c.requires_grad_()
        _, _, yhat_ref = net.forward_layers(x)
        ((yhat_ref - target) ** 2).sum().backward()
        for i in range(net.L):
            self.assertTrue(torch.allclose(gW[i], net.Ws[i].grad, atol=2e-6))
            self.assertTrue(torch.allclose(gb[i], net.bs[i].grad, atol=2e-6))
        self.assertTrue(torch.allclose(gv, net.v.grad, atol=2e-6))
        self.assertTrue(torch.allclose(gc, net.c.grad, atol=2e-6))


class ExactMeasurementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cfg = load_config("configs/mlp2_phase0_0829.yaml")
        cls.cfg = copy.deepcopy(cfg)
        cls.cfg["common"]["seeds"] = [0]

    def test_layer_identities_hold_on_full_support(self):
        arm = next(a for a in self.cfg["arms"] if a["name"] == "L2")
        st = setup_arm(self.cfg, arm, "cpu")
        rec, sanity = exact_layer_record(
            st, self.cfg["phase0"]["sigma_degenerate_tol"])
        self.assertEqual(sanity["support"], 32)
        self.assertTrue(sanity["run_finite"])
        for row in sanity["layers"]:
            self.assertLess(row["mean_max_relerr"], 1e-10)
            self.assertLess(row["sd_max_relerr"], 1e-10)
            self.assertLess(row["wall_max_relerr"], 1e-10)
        self.assertEqual(len(rec["layers"]), 2)

    def test_bootstrap_t_and_spearman_are_deterministic(self):
        values = np.arange(10, dtype=np.float64)
        draws = np.random.default_rng(20260829).integers(0, 10, size=(1000, 10))
        a = bootstrap_t(values, draws, "mean")
        b = bootstrap_t(values, draws, "mean")
        self.assertEqual(a, b)
        self.assertEqual(spearman(values, values), 1.0)
        self.assertEqual(spearman(values, values[::-1]), -1.0)


if __name__ == "__main__":
    unittest.main()
