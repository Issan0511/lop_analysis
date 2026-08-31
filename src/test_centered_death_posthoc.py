from __future__ import annotations

import unittest

import numpy as np

from src.centered_death_posthoc import (
    SOURCE,
    boundary_mask,
    classify_e2,
    classify_e3,
    classify_e4,
    final_dead_onsets,
    load_run,
)


class PureFunctionTests(unittest.TestCase):
    def test_boundary_mask_has_registered_counts(self) -> None:
        step = np.arange(0, 5_000_001, 1_000)
        mask = boundary_mask(step)
        self.assertEqual(mask.shape, (5000,))
        self.assertEqual(int(mask.sum()), 500)

    def test_final_dead_onset_uses_last_contiguous_run(self) -> None:
        dead = np.array([
            [True, False, True],
            [True, True, False],
            [False, False, True],
            [True, True, True],
        ])
        np.testing.assert_array_equal(final_dead_onsets(dead), [3, 3, 2])

    def test_registered_classification_branches(self) -> None:
        self.assertEqual(classify_e2(0.01, 0.09), "BIAS_CHANNEL_DOMINANT")
        self.assertEqual(classify_e2(0.31, 0.50), "MU_CHANNEL_ALIVE")
        self.assertEqual(classify_e2(0.05, 0.20), "CHANNEL_MIXED")
        self.assertEqual(
            classify_e3(
                {"point": -2.0, "ci_hi": -1.0},
                {"point": 1.0, "ci_lo": 0.2},
            ),
            "BOUNDARY_CARRIES_DESCENT",
        )
        self.assertEqual(
            classify_e4(
                {"ci_lo": -2.0, "ci_hi": -1.0},
                {"ci_lo": 0.1, "ci_hi": 0.3},
            ),
            "CENTERING_REDUCES_BUT_NOT_REMOVES",
        )


class CommittedInputTests(unittest.TestCase):
    def test_committed_flip_timing_matches_registered_s3(self) -> None:
        run = load_run(SOURCE / "logs" / "L2_A1_seed0.npz", "L2_A1", 0)
        changed = np.any(run.flip_state[1:] != run.flip_state[:-1], axis=1)
        self.assertEqual(int(changed.sum()), 499)
        self.assertTrue(np.all(run.step[:-1][changed] % 10_000 == 0))
        self.assertTrue(np.all(run.step[1:][changed] % 10_000 == 1_000))


if __name__ == "__main__":
    unittest.main()
