from __future__ import annotations

import unittest

import numpy as np

from analysis.mlp2_centering_delay_posthoc.analyze import (
    EPSILON_D,
    SOURCE,
    block_levels,
    build_verdicts,
    classify_p1,
    classify_p2,
    classify_p3,
    classify_p4,
    equivalence_status,
    estimate,
    load_and_validate,
    paired_gaps,
    shared_bootstrap_draws,
)


def ci(point: float, lo: float, hi: float) -> dict[str, float]:
    return {"point": point, "ci_lo": lo, "ci_hi": hi}


class VerdictBranchTests(unittest.TestCase):
    def test_equivalence_requires_entire_interval(self) -> None:
        self.assertEqual(equivalence_status(-0.04, 0.04, 0.0, EPSILON_D), "EQUIVALENT")
        self.assertEqual(
            equivalence_status(-0.05, 0.050000000000000044, 0.05, EPSILON_D),
            "EQUIVALENT",
        )
        self.assertEqual(equivalence_status(-0.06, 0.04, 0.0, EPSILON_D), "POINT_NEAR")
        self.assertEqual(equivalence_status(-0.20, -0.10, -0.15, EPSILON_D), "NOT_EQUIVALENT")

    def test_p1_registered_branches(self) -> None:
        early = ci(-0.20, -0.30, -0.10)
        closure = ci(0.20, 0.10, 0.30)
        self.assertEqual(
            classify_p1(early, closure, ci(0.0, -0.04, 0.04)),
            "MORPHOLOGICAL_DELAY_AND_CATCHUP",
        )
        self.assertEqual(
            classify_p1(early, closure, ci(-0.15, -0.20, -0.10)),
            "DURABLE_MORPHOLOGICAL_PROTECTION",
        )
        self.assertEqual(
            classify_p1(early, closure, ci(0.15, 0.10, 0.20)),
            "LATE_OVERSHOOT",
        )
        self.assertEqual(
            classify_p1(early, ci(0.0, -0.01, 0.01), ci(0.0, -0.04, 0.04)),
            "EARLY_GAP_WITHOUT_CLOSURE",
        )

    def test_p2_separates_morphology_and_function(self) -> None:
        catchup = "MORPHOLOGICAL_DELAY_AND_CATCHUP"
        self.assertEqual(
            classify_p2(catchup, ci(0.0, -0.08, 0.08)),
            "DELAY_ONLY_ACROSS_MORPHOLOGY_AND_FUNCTION",
        )
        self.assertEqual(
            classify_p2(catchup, ci(-0.5, -0.7, -0.2)),
            "MORPHOLOGICAL_CATCHUP_WITH_DURABLE_FUNCTIONAL_BENEFIT",
        )
        self.assertEqual(
            classify_p2("EARLY_GAP_WITHOUT_CLOSURE", ci(-0.5, -0.7, -0.2)),
            "DURABLE_PROTECTION",
        )

    def test_p3_and_p4_branches(self) -> None:
        negative = ci(-0.3, -0.4, -0.2)
        self.assertEqual(
            classify_p3(negative, negative, 0.20),
            "AALL_ABSOLUTE_AND_RELATIVE_LOW",
        )
        self.assertEqual(
            classify_p4("MORPHOLOGICAL_DELAY_AND_CATCHUP", negative),
            "LAYER2_LOCALIZED_CATCHUP",
        )
        self.assertEqual(
            classify_p4(
                "MORPHOLOGICAL_DELAY_AND_CATCHUP", ci(0.0, -0.04, 0.04)
            ),
            "GLOBAL_CATCHUP",
        )
        self.assertEqual(
            classify_p4("EARLY_GAP_WITHOUT_CLOSURE", ci(0.2, 0.1, 0.3)),
            "A1_LAYER1_PARADOX",
        )


class BootstrapTests(unittest.TestCase):
    def test_seed_cluster_bootstrap_is_deterministic(self) -> None:
        draws1 = shared_bootstrap_draws(B=200, seed=123)
        draws2 = shared_bootstrap_draws(B=200, seed=123)
        np.testing.assert_array_equal(draws1, draws2)
        result = estimate(np.arange(10, dtype=float), draws1)
        self.assertEqual(result["n_negative"], 0)
        self.assertEqual(result["n_zero"], 1)
        self.assertEqual(result["n_positive"], 9)


class CommittedInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data, cls.cfg, cls.provenance, cls.checks = load_and_validate(SOURCE)

    def test_registered_input_passes_all_sanity(self) -> None:
        self.assertEqual(len(self.data), 30_000)
        self.assertTrue(all(
            value is True
            for key, value in self.checks.items()
            if key != "strict_dead_fraction_max_error"
        ))

    def test_block_contract(self) -> None:
        levels = block_levels(self.data)
        self.assertEqual(len(levels), 600)
        self.assertEqual(set(levels.n_task), {50})
        self.assertEqual(set(levels.block), set(range(1, 11)))

    def test_committed_result_branches(self) -> None:
        levels = block_levels(self.data)
        gaps = paired_gaps(levels, self.data)
        _, details = build_verdicts(levels, gaps, shared_bootstrap_draws())
        self.assertEqual(details["P1"], "NO_EARLY_MORPHOLOGICAL_ADVANTAGE")
        self.assertEqual(details["catchup_time"], "CATCHUP_BY_5M_SINGLE_BLOCK")
        self.assertEqual(details["P2"], "INCONCLUSIVE_FUNCTIONAL_KINETICS")
        self.assertEqual(details["P3"], "AALL_RELATIVE_ONLY")
        self.assertEqual(details["P4"], "INCONCLUSIVE_LAYER_LOCALIZATION")
        self.assertEqual(
            details["P3F_stability"], "AALL_FUNCTION_DETERIORATED_BY_5M"
        )


if __name__ == "__main__":
    unittest.main()
