import unittest

import numpy as np

from src.common import load_config
from src.dose_const_5m import (_band_verdict, clopper_pearson, gamma_for_k,
                               validate_config)


class FixedDoseFormulaTests(unittest.TestCase):
    def test_all_registered_targets_are_exact_for_every_k(self):
        k = np.arange(16, dtype=np.float64)
        for target in (2.333, 2.510, 2.687, 2.864, 3.041):
            gamma = gamma_for_k(k, target)
            reconstructed = np.sqrt(
                5 * gamma ** 2 - (k + 2.5) * gamma + (k + 1.25))
            np.testing.assert_allclose(reconstructed, target, rtol=1e-13, atol=1e-13)

    def test_discriminants_are_positive(self):
        k = np.arange(16, dtype=np.float64)
        for target in (2.333, 2.510, 2.687, 2.864, 3.041):
            disc = np.square(k + 2.5) - 20 * (k + 1.25 - target ** 2)
            self.assertGreater(float(disc.min()), 0.0)


class ExactBinomialTests(unittest.TestCase):
    def test_zero_of_ten_interval(self):
        lo, hi = clopper_pearson(0, 10)
        self.assertEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 0.3084971078, places=9)

    def test_ten_of_ten_interval(self):
        lo, hi = clopper_pearson(10, 10)
        self.assertAlmostEqual(lo, 1 - 0.3084971078, places=9)
        self.assertEqual(hi, 1.0)


class VerdictBranchTests(unittest.TestCase):
    def test_registered_band_branches(self):
        self.assertEqual(_band_verdict(None, 9.33), "ANCHOR_FAILED")
        self.assertEqual(_band_verdict(10.04, 9.33), "BAND_MOVES_DOWN_BY_5M")
        self.assertEqual(_band_verdict(10.04, 10.04), "BAND_STABLE_TO_5M")


class ConfigTests(unittest.TestCase):
    def test_preflight_config_is_registered(self):
        cfg = load_config("configs/dose_const_5m_0830.yaml")
        validate_config(cfg, stage="preflight")


if __name__ == "__main__":
    unittest.main()
