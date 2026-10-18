"""Implementation checks; short trajectories are not performance evidence.

python -m unittest src.test_sna_loss_gate_1018 -v
"""
import copy
import gc
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from src import sna_loss_gate_1018 as G

E = G.E


def equal(test, a, b):
    if isinstance(a, dict):
        test.assertEqual(set(a), set(b))
        for k in a:
            equal(test, a[k], b[k])
    elif isinstance(a, (list, tuple)):
        test.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            equal(test, x, y)
    elif isinstance(a, torch.Tensor):
        test.assertTrue(torch.equal(a.cpu(), b.cpu()))
    else:
        test.assertEqual(a, b)


class RuleChecks(unittest.TestCase):
    def act(self, n=2, device="cpu"):
        return G.LossGateSnake([E.parse_arm("SNA")] * n, torch.device(device))

    def test_gate_formula_and_independent_runs(self):
        a = self.act()
        a.calibrate([[0.2, 0.4]] * 10, [0, 1])
        a.observe(torch.tensor([2.0, 0.1], requires_grad=True))
        torch.testing.assert_close(a.gate[0][:, 0], torch.tensor([.1, 1.]))
        self.assertFalse(a.gate[0].requires_grad)
        # Same inputs, unequal per-run CE: only the first run's adaptation slows down.
        v = [torch.full_like(x, 3.) for x in a.V]
        with patch.object(a, "batch_var", side_effect=lambda z, l: z):
            a.update(v)
        for x in a.V:
            torch.testing.assert_close(x[0], torch.full_like(x[0], 1.002))
            torch.testing.assert_close(x[1], torch.full_like(x[1], 1.02))
        a.observe(torch.tensor([0., float("inf")]))
        equal(self, a.gate[0], torch.tensor([[1.], [0.]]))

    def test_calibration_window_floor_and_missing_state(self):
        a = self.act()
        a.calibrate([[100., 100.]] + [[float(x), 0.] for x in range(10)], [0, 1])
        torch.testing.assert_close(a.loss_ref[0][:, 0], torch.tensor([4.5, G.EPS]))
        with self.assertRaises(ValueError):
            a.calibrate([[float("nan"), 0.]], [0])
        st = a.state()
        del st["enabled"]
        with self.assertRaises(ValueError):
            a.load_state(st)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required for captured CNN checks")
class CNNChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dev = E.H.setup("cuda")
        cls.cifar = E.RC.Cifar10()

    def tearDown(self):
        gc.collect()
        torch.cuda.empty_cache()

    def bundle(self, graph=False):
        b = G.LossGateBundle([(a, 10) for a in G.ARMS], self.cifar, self.dev, graph=graph)
        b.new_labels()
        b.new_order()
        return b

    def test_task1_host_training_bit_identical(self):
        b = self.bundle()
        host = E.Bundle([("SNA", 10), ("SNAfrz1", 10), ("SNA", 10)],
                        self.cifar, self.dev, graph=False)
        host.new_labels()
        host.new_order()  # bundle() has already advanced this stream once as well
        b.run_epoch()
        host.run_epoch()
        for k in ("P", "m", "v"):
            equal(self, getattr(b, k), getattr(host, k))
        for k, vs in host.act.state().items():
            equal(self, b.act.state()[k], vs)
        # All duplicate seed trajectories remain equal, including gate arm.
        for p in b.P:
            equal(self, p[0], p[2])

    def test_graph_activation_and_resume(self):
        b = self.bundle(graph=True)
        b.run_epoch()  # capture BEFORE enabling the rule
        b.act.calibrate([[.02] * 3], [2])
        b.act.freeze([1])
        b.new_labels()
        st = copy.deepcopy(b.full_state())
        eager = self.bundle()
        eager.load_full_state(st)
        b.run_epoch()
        eager.run_epoch()
        equal(self, b.full_state(), eager.full_state())
        self.assertLess(float(b.act.gate_sum[0][2, 0] / E.SPE), 0.1)
        equal(self, b.act.gate_sum[0][0], torch.tensor([float(E.SPE)], device=self.dev))
        # Continue on the SAME captured graph after in-place restoration.
        saved = copy.deepcopy(b.full_state())
        b.run_epoch()
        expected = copy.deepcopy(b.full_state())
        b.load_full_state(saved)
        b.run_epoch()
        equal(self, b.full_state(), expected)
        # Per-run checkpoints carry the actual active controller.
        sliced = b.run_state(2)
        for key in G.LossGateSnake.EXTRA:
            self.assertIn(key, sliced["act"])
        equal(self, sliced["act"]["enabled"][0], torch.tensor([1.]))

    def test_runner_boundary_resume_and_mismatch(self):
        class StopAfterTask1(Exception):
            pass

        def stop(msg, **kwargs):
            if msg.startswith("task 1/"):
                raise StopAfterTask1

        # Two short epochs/tasks verify bookkeeping and calibration, not learning quality.
        with tempfile.TemporaryDirectory() as tmp:
            slots = [(a, 10) for a in G.ARMS]
            a, b = Path(tmp) / "continuous", Path(tmp) / "resumed"
            with patch.object(E.RC, "Cifar10", return_value=self.cifar):
                G.run(slots, a, 2, 2, self.dev, graph=True)
                with self.assertRaises(StopAfterTask1):
                    G.run(slots, b, 2, 2, self.dev, graph=True, log=stop)
                G.run(slots, b, 2, 2, self.dev, graph=True)
                s1 = torch.load(a / "state.pt", weights_only=False)
                s2 = torch.load(b / "state.pt", weights_only=False)
                equal(self, s1["state"], s2["state"])
                self.assertEqual(s1["rows"], s2["rows"])
                self.assertEqual(s1["history"], s2["history"])
                self.assertEqual((a / "per_epoch.csv").read_text(), (b / "per_epoch.csv").read_text())
                with self.assertRaises(ValueError):
                    G.run(slots, b, 2, 3, self.dev, graph=True)


if __name__ == "__main__":
    unittest.main()
