import unittest

import numpy as np

from analysis.ceiling_t0.ceiling_t0 import aggregate_mask, root_detail


class CeilingT0UnitTests(unittest.TestCase):
    def setUp(self):
        self.ctx = {
            "centers": np.array([-0.15, -0.05, 0.05, 0.15]),
            "valid": np.ones(4, dtype=bool),
            "edge": np.ones(3, dtype=bool),
        }

    def test_rightmost_descending_root(self):
        y = np.array([1.0, -1.0, 1.0, -2.0])
        detail = root_detail(self.ctx, y)
        self.assertEqual(detail["n_down"], 2)
        self.assertAlmostEqual(detail["root"], 0.05 + 0.1 / 3)

    def test_single_zero_and_zero_interval(self):
        self.assertAlmostEqual(
            root_detail(self.ctx, np.array([1.0, 0.0, -1.0, -2.0]))["root"], -0.05)
        self.assertTrue(np.isnan(
            root_detail(self.ctx, np.array([1.0, 0.0, 0.0, -1.0]))["root"]))

    def test_guard_cut_blocks_interpolation(self):
        ctx = dict(self.ctx)
        ctx["edge"] = np.array([True, False, True])
        self.assertTrue(np.isnan(root_detail(ctx, np.array([1.0, 0.5, -0.5, -1.0]))["root"]))

    def test_band_sufficient_statistics(self):
        band = np.array([[0, 0], [1, 1]])
        mask = np.array([[True, False], [True, True]])
        value = np.array([[2.0, 99.0], [3.0, 5.0]])
        rows = aggregate_mask(band, mask, {"Y": value})
        self.assertEqual(rows, [
            {"band": 0, "count": 1, "sum_Y": 2.0},
            {"band": 1, "count": 2, "sum_Y": 8.0},
        ])


if __name__ == "__main__":
    unittest.main()
