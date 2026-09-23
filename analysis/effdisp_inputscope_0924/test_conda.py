"""Small preregistered-context checks; no production outcome inspection."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src import effdisp_inputscope_conda_0924 as C


class ContextFactorialTests(unittest.TestCase):
    def test_context_transform_and_teacher_split(self):
        flip = torch.tensor([[1.]*15+[0.]*5, [0.]*15+[0.]*5])
        initial = torch.tensor([[0.]*20, [0.]*20])
        random = torch.tensor([[1.,0.,1.,0.,1.],[1.,0.,1.,0.,1.]])
        x, y = C.context_inputs(flip, initial, random,
                                torch.tensor([0, 1], dtype=torch.bool),
                                torch.tensor([1, 0], dtype=torch.bool))
        self.assertTrue(torch.equal(x[0, :15], initial[0, :15]))
        self.assertTrue(torch.equal(y[0, :15], flip[0, :15]))
        self.assertTrue(torch.equal(x[1, :15], flip[1, :15]))
        self.assertTrue(torch.equal(y[1, :15], initial[1, :15]))
        self.assertTrue(torch.equal(x[:, 15:], random))
        self.assertTrue(torch.equal(y[:, 15:], random))

    def test_first_task_and_fixed_fixed_control(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            C.run(out, (200,), tasks=2, period=7, device="cpu", graph=False)
            for family in ("LR", "SNA06"):
                # The first task shares x, y, initialization and optimizer state.
                paths = [out/f"{family}_k1_X{x}Y{y}/seed200/t001.npz"
                         for x in (0, 1) for y in (0, 1)]
                with np.load(paths[0]) as first:
                    for path in paths[1:]:
                        with np.load(path) as other:
                            for key in ("W", "b", "v", "c", "adam_m_W", "adam_v_W",
                                        "mse", "task_start_mse", "teacher32targets"):
                                np.testing.assert_array_equal(first[key], other[key], err_msg=key)
                # X0Y0 ignores later flips, hence k1 and k7 are exact controls.
                with np.load(out/f"{family}_k1_X0Y0/seed200/t002.npz") as k1, \
                     np.load(out/f"{family}_k7_X0Y0/seed200/t002.npz") as k7:
                    for key in ("W", "b", "v", "c", "mse", "teacher32targets"):
                        np.testing.assert_array_equal(k1[key], k7[key], err_msg=key)
                with np.load(out/f"{family}_k7_X0Y1/seed200/t001.npz") as prior, \
                     np.load(out/f"{family}_k7_X0Y1/seed200/t002.npz") as moving, \
                     np.load(out/f"{family}_k7_X0Y0/seed200/t002.npz") as fixed:
                    self.assertEqual(np.count_nonzero(prior["flip_state"] != moving["flip_state"]), 7)
                    np.testing.assert_array_equal(moving["mu_input"], fixed["mu_input"])
                    self.assertTrue(np.any(moving["teacher32targets"] != fixed["teacher32targets"]))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_graph_matches_eager_across_flip(self):
        arms = tuple(a for a in C.ARMS if a[0] in (
            "LR_k1_X0Y0", "LR_k1_X0Y1", "LR_k7_X1Y0", "LR_k7_X1Y1",
            "SNA06_k1_X0Y1", "SNA06_k7_X1Y1"))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a, b = root/"graph", root/"eager"
            C.run(a, (200, 201), tasks=3, period=9, device="cuda", graph=True, arms=arms)
            C.run(b, (200, 201), tasks=3, period=9, device="cuda", graph=False, arms=arms)
            for pa in a.rglob("t*.npz"):
                with np.load(pa) as x, np.load(b/pa.relative_to(a)) as y:
                    self.assertEqual(set(x.files), set(y.files))
                    for key in x.files:
                        np.testing.assert_array_equal(x[key], y[key],
                                                      err_msg=f"{pa.relative_to(a)}:{key}")


if __name__ == "__main__":
    unittest.main()
