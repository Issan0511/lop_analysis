import unittest

import numpy as np

from analysis.task_rotation_m1.task_rotation_m1 import rank_corr


class TaskRotationM1UnitTests(unittest.TestCase):
    def test_rank_corr_with_ties(self):
        x = np.array([0.0, 0.0, 1.0, 2.0])
        y = np.array([3.0, 3.0, 2.0, 1.0])
        self.assertAlmostEqual(rank_corr(x, y), -1.0)

    def test_rank_corr_constant_is_nan(self):
        self.assertTrue(np.isnan(rank_corr(np.ones(4), np.arange(4.0))))


if __name__ == "__main__":
    unittest.main()
