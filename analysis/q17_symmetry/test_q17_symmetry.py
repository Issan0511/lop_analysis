import unittest

import numpy as np
import pandas as pd

from analysis.q17_symmetry.q17_symmetry import (
    bootstrap_indices,
    bootstrap_metric,
    camp_metrics,
    component_status,
    find_boundaries,
    paired_metrics,
    region_indices,
    verdict_table,
)


class Q17SymmetryUnitTests(unittest.TestCase):
    def test_magnitude_camp_sign_correction(self):
        v = np.array([1.0, 1.0, -1.0, -1.0])
        f = np.array([-2.0, 1.0, 2.0, -1.0])
        p_hat = np.ones(4)
        plus = camp_metrics(v, f, p_hat, 1)
        minus = camp_metrics(v, f, p_hat, -1)
        pair = paired_metrics(plus, minus)

        self.assertAlmostEqual(plus["N"], -1 / 3)
        self.assertAlmostEqual(minus["N"], 1 / 3)
        self.assertAlmostEqual(plus["M"], 1 / 3)
        self.assertAlmostEqual(minus["M"], 1 / 3)
        self.assertAlmostEqual(pair["B_M"], 0.0)
        self.assertAlmostEqual(pair["delta_N"], -2 / 3)
        self.assertAlmostEqual(pair["delta_N"], -2 * pair["A_M"])

    def test_probability_camp_sign_correction(self):
        v = np.array([1.0, 1.0, -1.0, -1.0])
        f = np.array([-2.0, 1.0, 2.0, -1.0])
        p_hat = np.ones(4)
        plus = camp_metrics(v, f, p_hat, 1)
        minus = camp_metrics(v, f, p_hat, -1)
        pair = paired_metrics(plus, minus)
        self.assertEqual(plus["p"], plus["q"])
        self.assertEqual(minus["p"], 1 - minus["q"])
        self.assertAlmostEqual(pair["B"], plus["q"] + minus["q"] - 1)
        self.assertAlmostEqual(pair["C"], plus["q"] - minus["q"])

    def test_zero_excluded_only_from_sign_denominator(self):
        v = np.array([1.0, 1.0, 1.0])
        f = np.array([-1.0, 0.0, 1.0])
        p_hat = np.array([1.0, 0.0, 1.0])
        result = camp_metrics(v, f, p_hat, 1)
        self.assertEqual(result["n_rows"], 3)
        self.assertEqual(result["n_nonzero"], 2)
        self.assertAlmostEqual(result["p"], 0.5)
        self.assertAlmostEqual(result["z"], 1 / 3)
        self.assertEqual(result["zero_phat0_fraction"], 1.0)

    def test_registered_boundary_regions(self):
        step = np.arange(9_900, 10_101, dtype=np.int64)
        flip = np.zeros((len(step), 1), dtype=np.float32)
        flip[step >= 10_001] = 1
        boundary = find_boundaries(step, flip)
        self.assertEqual(step[boundary].tolist(), [10_000])
        bstep = step[boundary]
        self.assertEqual(len(region_indices(step, bstep, -100, 100)), 201)
        self.assertEqual(len(region_indices(step, bstep, -100, -1)), 100)
        self.assertEqual(len(region_indices(step, bstep, 1, 100)), 100)

    def test_bootstrap_is_deterministic_and_duplicates_are_draws(self):
        first = bootstrap_indices(3, n_boot=20, seed=7)
        second = bootstrap_indices(3, n_boot=20, seed=7)
        np.testing.assert_array_equal(first, second)
        stat = bootstrap_metric(np.array([1.0, 2.0, 3.0]), first)
        self.assertEqual(stat["n_valid_seed"], 3)
        self.assertEqual(stat["n_boot_nonfinite"], 0)

    def test_component_decisions(self):
        base = {"n_valid_seed": 10, "n_boot_nonfinite": 0,
                "half_width": 0.01}
        self.assertEqual(component_status(
            {**base, "ci_lo": 0.47, "ci_hi": 0.53}, 0.5, 0.05), "EQUIV")
        self.assertEqual(component_status(
            {**base, "ci_lo": 0.56, "ci_hi": 0.58}, 0.5, 0.05), "MATERIAL")
        self.assertEqual(component_status(
            {**base, "ci_lo": 0.44, "ci_hi": 0.52}, 0.5, 0.05), "INCONCLUSIVE")

    def test_all_frozen_components_equivalent(self):
        rows = []
        for seed in range(10):
            for region in ("FULL", "PRE", "POST"):
                rows.append({
                    "seed": seed, "region": region,
                    "p_plus": 0.5, "p_minus": 0.5, "B": 0.0,
                    "M_plus": 0.0, "M_minus": 0.0, "B_M": 0.0,
                    "valid_W": True, "valid_N": True,
                })
        verdict = verdict_table(pd.DataFrame(rows), epsilon_m=0.02, n_boot=100)
        overall = verdict.loc[verdict.metric == "OVERALL", "status"].iloc[0]
        self.assertEqual(overall, "SYMMETRIC_REST_DRIVE")


if __name__ == "__main__":
    unittest.main()
