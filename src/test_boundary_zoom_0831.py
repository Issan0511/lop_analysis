from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.boundary_zoom_0831 import classify_gate, stage1_probe_steps


class BoundaryZoomTests(unittest.TestCase):
    def setUp(self) -> None:
        self.zoom = {
            "common": {"task_period": 10_000, "coarse_every": 1_000},
            "stage1": {
                "total_steps": 200_000, "fine_offset_min": -300,
                "fine_offset_max": 300, "fine_every": 20,
                "separable_exclusion_interval": [30, 330],
            },
        }

    def test_probe_grid_is_unique_and_contains_registered_windows(self) -> None:
        steps = stage1_probe_steps(self.zoom)
        self.assertEqual(len(steps), len(np.unique(steps)))
        self.assertIn(9_700, steps)
        self.assertIn(10_300, steps)
        self.assertIn(200_000, steps)

    def test_gate_branches(self) -> None:
        rows = pd.DataFrame({"tau_fit": [100.0] * 19,
                             "censored_after_300": [0] * 19})
        self.assertEqual(classify_gate(rows, self.zoom)["label"],
                         "TIMESCALES_NOT_SEPARABLE")
        rows["tau_fit"] = 20.0
        self.assertEqual(classify_gate(rows, self.zoom)["label"],
                         "TIMESCALES_SEPARABLE")
        rows.loc[:9, "censored_after_300"] = 1
        self.assertEqual(classify_gate(rows, self.zoom)["label"],
                         "TIMESCALE_GATE_UNRESOLVED")


if __name__ == "__main__":
    unittest.main()
