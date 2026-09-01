from __future__ import annotations

import copy
import unittest

from src.common import load_config
from src.lr_a1_0901 import (classify_p1, classify_p2, g_leaky,
                            preregistration_missing, validate_config)


class LrA1Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = load_config("configs/lr_a1_0901.yaml")

    def test_leaky_closed_form_ratio_matches_spec_rounding(self) -> None:
        self.assertAlmostEqual(g_leaky(0.1), 0.5854770287516298, places=14)
        self.assertEqual(round(g_leaky(0.1), 3), 0.585)
        self.assertAlmostEqual(10.0 * g_leaky(0.1), 5.854770287516298,
                               places=13)

    def test_draft_allows_preflight_but_blocks_scientific_run(self) -> None:
        validate_config(self.cfg, stage="preflight")
        validate_config(self.cfg, stage="smoke")
        self.assertEqual(preregistration_missing(self.cfg), [
            "preregistration.execution_authorized",
        ])
        with self.assertRaisesRegex(ValueError, "preregistration is not frozen"):
            validate_config(self.cfg, stage="full")
        with self.assertRaisesRegex(ValueError, "preregistration is not frozen"):
            validate_config(self.cfg, stage="analyze")

    def test_confirmed_preregistration_unblocks_full_validation(self) -> None:
        frozen = copy.deepcopy(self.cfg)
        frozen["preregistration"].update(
            frozen=True,
            execution_authorized=True,
            repo_spec_committed=True,
        )
        validate_config(frozen, stage="full")
        self.assertEqual(preregistration_missing(frozen), [])

    def test_p1_uses_absolute_dose_band_not_relu_raw_delta(self) -> None:
        cases = [
            ((5.35, 6.35), "A_CLOSED_FORM_MATCH"),
            ((5.5, 6.0), "A_CLOSED_FORM_MATCH"),
            ((4.8, 5.2), "A_DOSE_OFF_PREDICTION"),
            ((6.4, 6.8), "A_DOSE_OFF_PREDICTION"),
            ((5.2, 5.6), "INCONCLUSIVE_WIDE"),
            ((6.2, 6.5), "INCONCLUSIVE_WIDE"),
            ((5.2, 5.35), "INCONCLUSIVE_WIDE"),
        ]
        for bounds, expected in cases:
            with self.subTest(bounds=bounds):
                self.assertEqual(classify_p1(*bounds), expected)

    def test_p2_is_sign_first_and_accepts_stronger_improvement(self) -> None:
        cases = [
            ((-0.1, -0.9), "A_WITHOUT_B_HARMLESS_MULTILAYER"),
            ((-0.1, -1.2), "A_WITHOUT_B_HARMLESS_MULTILAYER"),
            ((-0.1, -0.5), "PARTIAL_IMPROVEMENT"),
            ((0.0, -0.9), "A_WITHOUT_B_NOT_CONFIRMED"),
            ((0.2, -1.2), "A_WITHOUT_B_NOT_CONFIRMED"),
        ]
        for values, expected in cases:
            with self.subTest(values=values):
                self.assertEqual(classify_p2(*values), expected)


if __name__ == "__main__":
    unittest.main()
