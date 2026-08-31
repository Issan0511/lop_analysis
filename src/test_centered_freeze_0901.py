import unittest

import pandas as pd

from src.centered_freeze_0901 import classify_p1, markdown_table


class ClassifyP1Tests(unittest.TestCase):
    def test_decisive_requires_all_three_registered_conditions(self):
        ci = {"lo": -0.5, "hi": -0.2}
        self.assertEqual(classify_p1(0.04, 0.90, ci, 0.05, 0.80),
                         "BIAS_ROUTE_DECISIVE")
        self.assertEqual(classify_p1(0.06, 0.90, ci, 0.05, 0.80),
                         "BIAS_ROUTE_PARTIAL")
        self.assertEqual(classify_p1(0.04, 0.79, ci, 0.05, 0.80),
                         "BIAS_ROUTE_PARTIAL")

    def test_null_and_adverse_categories(self):
        self.assertEqual(classify_p1(0.01, 0.95, {"lo": -0.2, "hi": 0.01},
                                     0.05, 0.80), "BIAS_ROUTE_NOT_SUPPORTED")
        self.assertEqual(classify_p1(0.4, -0.2, {"lo": 0.01, "hi": 0.3},
                                     0.05, 0.80), "BIAS_FREEZE_INCREASES_DEATH")

    def test_markdown_table_has_no_optional_dependency(self):
        rendered = markdown_table(pd.DataFrame([{"name": "a|b", "value": 0.125}]))
        self.assertIn("a\\|b", rendered)
        self.assertIn("0.125", rendered)


if __name__ == "__main__":
    unittest.main()
