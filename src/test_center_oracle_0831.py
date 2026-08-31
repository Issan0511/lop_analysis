from __future__ import annotations

import unittest

import numpy as np

from src.center_oracle_0831 import estimate, shared_draws


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


if __name__ == "__main__":
    unittest.main()
