import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.common import load_config
from src.mlp2_phase0b import (_ci_components, _complete_arm_logs,
                              _window_indices, guarded_ci)


class GuardedBootstrapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config("configs/mlp2_phase0b_0829.yaml")

    def test_constant_input_is_degenerate(self):
        draws = np.random.default_rng(20260829).integers(0, 10, size=(1000, 10))
        result = guarded_ci(np.zeros(10), draws, self.cfg)
        self.assertEqual(result["ci_degenerate"], 1)
        self.assertEqual(result["sign_test_p"], 1.0)

    def test_well_spread_input_is_not_se_degenerate(self):
        draws = np.random.default_rng(7).integers(0, 10, size=(1000, 10))
        result = _ci_components(np.arange(1, 11, dtype=float), draws, "mean",
                                1e-15, .01, 100.0)
        self.assertLessEqual(result["degenerate_se_fraction"], .01)
        self.assertLess(result["percentile_ci_lo"], result["point"])
        self.assertGreater(result["percentile_ci_hi"], result["point"])


class Phase0bHelpersTests(unittest.TestCase):
    def test_task_windows_are_inclusive(self):
        steps = np.arange(10_000, 130_000, 10_000)
        idx = _window_indices(steps, 10_000, [2, 11])
        self.assertEqual(steps[idx].tolist(), list(range(20_000, 120_000, 10_000)))

    def test_incomplete_logs_do_not_resume(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertFalse(_complete_arm_logs(Path(td), "L2", [0], 100))


if __name__ == "__main__":
    unittest.main()
