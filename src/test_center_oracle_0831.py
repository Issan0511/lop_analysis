from __future__ import annotations

import unittest

import numpy as np

from src.center_oracle_0831 import estimate, shared_draws, transition_masks


class OracleAnalysisTests(unittest.TestCase):
    def test_seed_bootstrap_is_deterministic(self) -> None:
        draws1 = shared_draws(200, 123)
        draws2 = shared_draws(200, 123)
        np.testing.assert_array_equal(draws1, draws2)
        result = estimate(np.arange(10, dtype=float), draws1)
        self.assertEqual(result["n_seed"], 10)

    def test_ratio_inputs_are_seed_level(self) -> None:
        draws = shared_draws(100, 7)
        result = estimate(np.full(10, 0.1), draws)
        self.assertEqual(result["point"], 0.1)
        self.assertEqual(result["ci_lo"], 0.1)
        self.assertEqual(result["ci_hi"], 0.1)

    def test_followup_transition_partition(self) -> None:
        step = np.arange(0, 5_000_001, 1_000)
        flip = np.zeros((step.size, 1), dtype=np.int8)
        for task in range(1, 500):
            flip[task * 10 + 1:] ^= 1
        masks = transition_masks(step, flip)
        self.assertEqual(int(masks["changed"].sum()), 499)
        self.assertEqual(int(masks["boundary_499"].sum()), 499)
        self.assertEqual(int(masks["boundary_500_report_only"].sum()), 500)
        self.assertEqual(int(masks["internal_4500"].sum()), 4500)
        self.assertEqual(int(masks["startup_0to1000"].sum()), 1)
        np.testing.assert_array_equal(masks["changed"], masks["boundary_499"])


if __name__ == "__main__":
    unittest.main()
