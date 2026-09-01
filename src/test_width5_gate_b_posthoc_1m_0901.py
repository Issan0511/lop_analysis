from __future__ import annotations

import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from src.common import ROOT
from src.width5_gate_b_posthoc_1m_0901 import (
    COLLAPSE_THRESHOLD,
    KNOWN_1M_K,
    SOURCE,
    SanityError,
    baseline_drift,
    bimodality,
    check_reproduce,
    collapse_counts,
    load_source,
    run_analysis,
    sign_test,
)


def _values(**arms) -> dict:
    """Build the {arm: {window: [per-seed unfit]}} structure the module uses."""
    return {arm: {"1m": list(one), "5m": list(five)}
            for arm, (one, five) in arms.items()}


class PosthocRuleTests(unittest.TestCase):
    def test_sign_test_applies_the_registered_labels(self) -> None:
        n = 20
        cases = {
            # k, expected label suffix
            20: "_ABOVE_LINEAR",
            0: "_BELOW_LINEAR",
            10: "_NOT_SEPARATED_TIGHT",
            13: "_INCONCLUSIVE_WIDE",
        }
        for k, suffix in cases.items():
            arm = [2.0] * k + [0.5] * (n - k)
            values = _values(R5=(arm, arm), LIN5=([1.0] * n, [1.0] * n))
            got = sign_test(values, "R5", "5m")
            self.assertEqual(got["k"], k)
            self.assertTrue(got["status"].endswith(suffix),
                            f"k={k} gave {got['status']}")
            self.assertEqual(got["registered"], 0)

    def test_ties_stay_in_n_but_not_in_k(self) -> None:
        values = _values(R5=([1.0] * 20, [1.0] * 20),
                         LIN5=([1.0] * 20, [1.0] * 20))
        got = sign_test(values, "R5", "5m")
        self.assertEqual((got["k"], got["n"], got["ties"]), (0, 20, 20))

    def test_nonfinite_seeds_are_dropped_from_n(self) -> None:
        arm = [2.0] * 19 + [float("nan")]
        values = _values(R5=(arm, arm), LIN5=([1.0] * 20, [1.0] * 20))
        got = sign_test(values, "R5", "5m")
        self.assertEqual((got["k"], got["n"]), (19, 19))
        self.assertEqual(got["excluded_seed_indices"], [19])

    def test_collapse_and_bimodality_split_on_the_degeneracy_point(self) -> None:
        late = [1.0, 0.9995, 0.5, 0.4]
        values = {arm: {"1m": late, "5m": late}
                  for arm in ("R5", "LR5", "E5", "LIN5", "R100", "LR100",
                              "E100", "LIN100")}
        values["LIN5"] = {"1m": [0.45] * 4, "5m": [0.45] * 4}
        counts = {r["arm"]: r for r in collapse_counts(values)}
        self.assertEqual(counts["R5"]["k"], 2)
        self.assertEqual(counts["R5"]["threshold"], COLLAPSE_THRESHOLD)
        self.assertEqual(counts["LIN5"]["k"], 0)
        rows = {r["group"]: r for r in bimodality(values)}
        self.assertEqual((rows["collapsed"]["n"],
                          rows["collapsed"]["above_LIN5"]), (2, 2))
        self.assertEqual((rows["not_collapsed"]["n"],
                          rows["not_collapsed"]["above_LIN5"]), (2, 1))

    def test_baseline_drift_reports_degraded_only_when_both_agree(self) -> None:
        worse = baseline_drift(_values(LIN5=([0.1] * 4, [0.2] * 4)))
        self.assertTrue(worse["degraded"])
        flat = baseline_drift(_values(LIN5=([0.1, 0.3, 0.2, 0.4],
                                            [0.3, 0.1, 0.4, 0.2])))
        self.assertFalse(flat["degraded"])

    def test_reproduce_check_fails_on_a_label_mismatch(self) -> None:
        values = _values(R5=([2.0] * 20, [2.0] * 20),
                         LR5=([0.5] * 20, [0.5] * 20),
                         E5=([0.5] * 20, [0.5] * 20),
                         LIN5=([1.0] * 20, [1.0] * 20))
        signs = {(arm, w): sign_test(values, arm, w)
                 for w in ("1m", "5m") for arm in ("R5", "LR5", "E5")}
        good = check_reproduce(signs, {
            "R5": dict(k="20", status="R5_ABOVE_LINEAR"),
            "LR5": dict(k="0", status="LR5_BELOW_LINEAR"),
            "E5": dict(k="0", status="E5_BELOW_LINEAR")})
        self.assertTrue(good["pass_"])
        bad = check_reproduce(signs, {
            "R5": dict(k="13", status="R5_INCONCLUSIVE_WIDE"),
            "LR5": dict(k="0", status="LR5_BELOW_LINEAR"),
            "E5": dict(k="0", status="E5_BELOW_LINEAR")})
        self.assertFalse(bad["pass_"])


class PosthocCommittedInputTests(unittest.TestCase):
    """Runs against the committed parent output, into a temporary outdir."""

    def setUp(self) -> None:
        self.source = Path(ROOT) / SOURCE

    def test_input_hash_mismatch_aborts_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tampered = Path(tmp) / "source"
            tampered.mkdir()
            for name in ("verdict.csv", "provenance.json"):
                shutil.copy(self.source / name, tampered / name)
            with (tampered / "verdict.csv").open("a", encoding="utf-8") as fh:
                fh.write("\n")
            out = Path(tmp) / "out"
            with self.assertRaises(SanityError):
                run_analysis(tampered, out)
            self.assertFalse(out.exists())

    def test_end_to_end_reproduces_the_handoff_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "posthoc_1m"
            result = run_analysis(self.source, out)

            # S_reproduce is gating and already ran; assert it explicitly too.
            self.assertTrue(result["sanity"]["S_reproduce"]["pass_"])
            self.assertTrue(result["sanity"]["S_known"]["pass_"])

            for arm, want in KNOWN_1M_K.items():
                self.assertEqual(result["signs"][(arm, "1m")]["k"], want)
            self.assertEqual(
                result["signs"][("R5", "1m")]["status"],
                "R5_NOT_SEPARATED_TIGHT")
            self.assertEqual(
                result["signs"][("R5", "5m")]["status"],
                "R5_INCONCLUSIVE_WIDE")

            rows = {r["arm"]: r for r in result["collapses"]}
            self.assertEqual(rows["R5"]["k"], 8)
            self.assertEqual(rows["LIN5"]["k"], 0)
            self.assertEqual(rows["R100"]["k"], 1)

            groups = {r["group"]: r for r in result["bimodality"]}
            self.assertEqual((groups["collapsed"]["n"],
                              groups["collapsed"]["above_LIN5"]), (8, 8))
            self.assertEqual((groups["not_collapsed"]["n"],
                              groups["not_collapsed"]["above_LIN5"]), (12, 5))
            self.assertFalse(result["drift"]["degraded"])
            self.assertEqual(result["drift"]["worse_seed_count"], 10)

            for name in ("verdict.csv", "bimodality.csv", "summary.md",
                         "provenance.json"):
                self.assertTrue((out / name).exists())
            with (out / "verdict.csv").open(newline="", encoding="utf-8") as fh:
                emitted = list(csv.DictReader(fh))
            self.assertTrue(all(row["registered"] == "0" for row in emitted))
            prov = json.loads((out / "provenance.json").read_text("utf-8"))
            self.assertEqual(prov["analysis_grade"],
                             "registered_posthoc_not_preregistered")

    def test_load_source_hashes_match_the_registered_values(self) -> None:
        _, meta = load_source(self.source)
        self.assertEqual(set(meta["input_sha256"]),
                         {"verdict.csv", "provenance.json"})


if __name__ == "__main__":
    unittest.main()
