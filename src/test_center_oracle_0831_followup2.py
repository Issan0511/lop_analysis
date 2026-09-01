from __future__ import annotations

import csv
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.center_oracle_0831 import OracleSanityError
from src.center_oracle_0831_followup2 import (
    INPUT_SHA256,
    KNOWN,
    MIN_SURVIVING_BOUNDARIES,
    ORACLE_ARM,
    REFERENCE_ARM,
    WINDOW_ALL,
    WINDOW_K,
    check_inputs,
    extinction_task,
    label_ratio,
    per_unit_rates,
    revival_counts,
    run_analysis,
    seed_rate,
    survival_K,
    window_dependence,
)
from src.common import ROOT


def _synthetic(n_task: int = 3, n_unit: int = 2):
    """Records every 1000 steps with a true switch at each task boundary."""
    step = np.arange(0, n_task * 10_000 + 1_000, 1_000)
    flip = np.zeros((step.size, 4), dtype=np.float32)
    for i, s in enumerate(step):
        # flips exactly at the records that follow a task boundary
        flip[i] = (s + 9_000) // 10_000 % 2
    beta = np.zeros((step.size, n_unit), dtype=float)
    p_hat = np.ones((step.size, n_unit), dtype=np.float32)
    return step, flip, beta, p_hat


class RateRuleTests(unittest.TestCase):
    def test_only_boundary_transitions_with_a_live_predecessor_count(self):
        step, flip, beta, p_hat = _synthetic()
        boundaries = np.flatnonzero(
            np.any(flip[1:] != flip[:-1], axis=1))
        self.assertEqual(boundaries.size, 3)
        # unit 0 drops 1.0 at each boundary, unit 1 drops 1.0 everywhere
        for b in boundaries:
            beta[b + 1:, 0] -= 1.0
        beta[1:, 1] -= np.arange(1, beta.shape[0])
        rate, n_surv = per_unit_rates(step, flip, beta, p_hat)
        self.assertEqual(list(n_surv), [3, 3])
        self.assertAlmostEqual(rate[0], -1.0)
        self.assertAlmostEqual(rate[1], -1.0)

        # kill unit 0 before the last boundary: it stops accumulating
        p_hat[boundaries[-1], 0] = 0.0
        rate, n_surv = per_unit_rates(step, flip, beta, p_hat)
        self.assertEqual(list(n_surv), [2, 3])
        self.assertAlmostEqual(rate[0], -1.0)

    def test_k_truncates_to_the_first_k_boundaries(self):
        step, flip, beta, p_hat = _synthetic()
        boundaries = np.flatnonzero(np.any(flip[1:] != flip[:-1], axis=1))
        for i, b in enumerate(boundaries):
            beta[b + 1:, 0] -= float(i + 1)
        rate_all, n_all = per_unit_rates(step, flip, beta, p_hat)
        rate_k, n_k = per_unit_rates(step, flip, beta, p_hat, k=2)
        self.assertEqual((n_all[0], n_k[0]), (3, 2))
        self.assertAlmostEqual(rate_all[0], -(1 + 2 + 3) / 3)
        self.assertAlmostEqual(rate_k[0], -(1 + 2) / 2)

    def test_a_unit_that_survives_nothing_is_nan_not_zero(self):
        step, flip, beta, p_hat = _synthetic()
        p_hat[:, 0] = 0.0
        rate, n_surv = per_unit_rates(step, flip, beta, p_hat)
        self.assertEqual(n_surv[0], 0)
        self.assertTrue(np.isnan(rate[0]))

    def test_non_finite_beta_aborts(self):
        step, flip, beta, p_hat = _synthetic()
        beta[2, 0] = np.nan
        with self.assertRaises(OracleSanityError):
            per_unit_rates(step, flip, beta, p_hat)

    def test_seed_rate_drops_units_below_the_survival_floor(self):
        rate = np.array([-1.0, -5.0, -3.0])
        n_surv = np.array([MIN_SURVIVING_BOUNDARIES,
                           MIN_SURVIVING_BOUNDARIES - 1,
                           MIN_SURVIVING_BOUNDARIES + 4])
        got = seed_rate(rate, n_surv)
        self.assertEqual(got["n_kept"], 2)
        self.assertAlmostEqual(got["value"], -2.0)     # median of -1 and -3
        self.assertAlmostEqual(got["excluded_frac"], 1 / 3)

    def test_seed_rate_raises_when_every_unit_is_excluded(self):
        with self.assertRaises(OracleSanityError):
            seed_rate(np.array([-1.0]), np.array([1]))


class WindowRuleTests(unittest.TestCase):
    def test_K_is_the_rounded_median_of_per_seed_medians(self):
        per_seed = [np.array([10, 10, 12, 12]), np.array([40, 40, 40, 40]),
                    np.array([36, 36, 37, 37])]
        K, medians = survival_K(per_seed)
        self.assertEqual(medians, [11.0, 40.0, 36.5])
        self.assertEqual(K, 37)      # median 36.5 -> 37, ties round half up

    def test_K_uses_every_unit_including_the_ones_later_excluded(self):
        K, _ = survival_K([np.array([0, 0, 100, 100])])
        self.assertEqual(K, 50)

    def test_ratio_labels_key_off_one_not_zero(self):
        self.assertEqual(label_ratio({"ci_lo": 1.2, "ci_hi": 2.0}),
                         "LAG_IS_PROTECTIVE")
        self.assertEqual(label_ratio({"ci_lo": 0.2, "ci_hi": 0.9}),
                         "LAG_IS_HARMFUL")
        self.assertEqual(label_ratio({"ci_lo": 0.9, "ci_hi": 1.1}),
                         "RATE_INCONCLUSIVE")
        self.assertEqual(label_ratio({"ci_lo": 1.0, "ci_hi": 2.0}),
                         "RATE_INCONCLUSIVE")

    def test_window_dependence_only_on_an_outright_split(self):
        self.assertEqual(
            window_dependence("LAG_IS_PROTECTIVE", "LAG_IS_HARMFUL"),
            "WINDOW_DEPENDENT")
        self.assertIsNone(
            window_dependence("LAG_IS_PROTECTIVE", "RATE_INCONCLUSIVE"))
        self.assertIsNone(
            window_dependence("LAG_IS_PROTECTIVE", "LAG_IS_PROTECTIVE"))


class RevivalTests(unittest.TestCase):
    def test_revivals_split_between_boundary_and_within_task(self):
        step, flip, _, p_hat = _synthetic()
        boundaries = np.flatnonzero(np.any(flip[1:] != flip[:-1], axis=1))
        p_hat[boundaries[0], 0] = 0.0          # dies, revives at a boundary
        p_hat[5, 1] = 0.0                      # dies, revives inside a task
        counts = revival_counts(step, flip, p_hat)
        self.assertEqual(counts["total"], 2)
        self.assertEqual(counts["at_boundary"], 1)
        self.assertEqual(counts["within_task"], 1)

    def test_extinction_task_is_none_while_anything_lives(self):
        step, flip, _, p_hat = _synthetic()
        self.assertIsNone(extinction_task(step, p_hat))
        p_hat[12:, :] = 0.0
        self.assertEqual(extinction_task(step, p_hat), int(step[12] // 10_000))


class CommittedInputTests(unittest.TestCase):
    """Runs against the committed logs of both arms, into a temporary outdir."""

    def test_input_hash_mismatch_aborts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in INPUT_SHA256:
                dst = root / name
                dst.parent.mkdir(parents=True, exist_ok=True)
                os.symlink(Path(ROOT) / name, dst)
            tampered = root / f"results/center_oracle_0831/verdict.csv"
            tampered.unlink()
            tampered.write_bytes(
                (Path(ROOT) / "results/center_oracle_0831/verdict.csv"
                 ).read_bytes() + b"\n")
            with self.assertRaises(OracleSanityError):
                check_inputs(root)

    def test_end_to_end_reproduces_the_vault_addendum(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "followup2"
            result = run_analysis(None, out)

            self.assertEqual(result["K"], KNOWN["K"])
            self.assertTrue(result["sanity"]["S_reproduce"]["pass_"])
            self.assertTrue(result["sanity"]["S_known"]["pass_"])
            self.assertTrue(result["sanity"]["S_mask"]["pass_"])

            self.assertEqual(result["labels"]["P1a"], "LAG_IS_PROTECTIVE")
            self.assertEqual(result["labels"]["P1b"], "LAG_IS_PROTECTIVE")
            self.assertEqual(result["labels"]["P1c"],
                             "ABSORPTION_EXACT_UNDER_ORACLE")
            self.assertIsNone(result["labels"]["window"])

            for window, key in ((WINDOW_ALL, "rate_all_499"),
                                (WINDOW_K, "rate_first_K")):
                for arm in (ORACLE_ARM, REFERENCE_ARM):
                    self.assertAlmostEqual(
                        result["rates"][(window, arm)]["point"],
                        KNOWN[key][arm], places=3)
            # the ratio must clear 1 on both windows, not merely point above it
            self.assertGreater(result["ratios"][WINDOW_ALL]["ci_lo"], 1.0)
            self.assertGreater(result["ratios"][WINDOW_K]["ci_lo"], 1.0)

            self.assertEqual(result["revivals"][ORACLE_ARM]["total"], 0)
            self.assertEqual(result["revivals"][REFERENCE_ARM]["total"],
                             KNOWN["revival_total"][REFERENCE_ARM])
            extinct = [t for t in result["extinction"][ORACLE_ARM]
                       if t is not None]
            self.assertEqual(len(extinct), 10)
            self.assertEqual((min(extinct), max(extinct)), (154, 454))
            self.assertEqual(np.median(extinct), 224)
            self.assertTrue(all(t is None
                                for t in result["extinction"][REFERENCE_ARM]))

            for name in ("verdict.csv", "unit_rates.csv",
                         "sanity_reproduce.csv", "sanity_known.csv",
                         "summary.md", "provenance.json"):
                self.assertTrue((out / name).exists(), name)
            with (out / "verdict.csv").open(newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            self.assertTrue(all(row["registered"] == "0" for row in rows))
            self.assertTrue(any(row["endpoint"] == "P1prime_a" for row in rows))
            prov = json.loads((out / "provenance.json").read_text("utf-8"))
            self.assertEqual(prov["analysis_grade"],
                             "registered_posthoc_not_preregistered")
            self.assertEqual(prov["K"], KNOWN["K"])


if __name__ == "__main__":
    unittest.main()
