from __future__ import annotations

import unittest

import numpy as np

from analysis.function_blind_m2.function_blind_m2 import (
    classify_baseline,
    classify_dynamics,
    estimate_for_multiplicities,
    pair_score,
)


class FunctionBlindM2Tests(unittest.TestCase):
    def test_pair_score_includes_half_ties(self) -> None:
        result = pair_score(np.asarray([2.0, 1.0]), np.asarray([1.0, 2.0]))
        expected = np.asarray([[1.0, 0.5], [0.5, 0.0]])
        np.testing.assert_array_equal(result, expected)

    def test_registered_classification_order(self) -> None:
        self.assertEqual(classify_dynamics(-0.04, 0.04), "EQUIV_DYNAMICS")
        self.assertEqual(classify_dynamics(0.051, 0.10), "HIGH_LESS_PUSHED")
        self.assertEqual(classify_dynamics(-0.10, -0.051), "HIGH_MORE_PUSHED")
        self.assertEqual(classify_dynamics(0.01, 0.06), "INCONCLUSIVE")
        self.assertEqual(classify_baseline(0.051, 0.10), "HIGH_STARTS_MORE_OPEN")
        self.assertEqual(classify_baseline(-0.10, -0.051), "HIGH_STARTS_LESS_OPEN")
        self.assertEqual(classify_baseline(-0.04, 0.04), "EQUIV_BASELINE")

    def test_cell_weighted_point_estimate(self) -> None:
        cells = []
        for cell_id, high, low, seed in (
            (1, 2.0, 1.0, 0),
            (2, 0.0, 1.0, 1),
        ):
            cell = dict(
                cell_id=cell_id,
                high_seed=np.asarray([seed]),
                low_seed=np.asarray([seed]),
            )
            for metric in ("S0", "S1", "delta_S"):
                hv = np.asarray([high])
                lv = np.asarray([low])
                cell[f"{metric}_high"] = hv
                cell[f"{metric}_low"] = lv
                cell[f"{metric}_score"] = pair_score(hv, lv)
            cells.append(cell)
        result = estimate_for_multiplicities(cells, np.ones(20))
        self.assertEqual(float(result["weight"][0]), 2.0)
        self.assertEqual(float(result["A_delta_S"][0]), 0.0)
        self.assertEqual(float(result["D_delta_S"][0]), 0.0)


if __name__ == "__main__":
    unittest.main()
