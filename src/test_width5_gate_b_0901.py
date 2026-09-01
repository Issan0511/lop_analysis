from __future__ import annotations

import copy
import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.common import load_config
from src.width5_gate_b_0901 import (
    ARM_ORDER,
    LAYER_KEYS,
    RUN_KEYS,
    _arm,
    _lin_ref_preflight,
    _paired_endpoint,
    _run_arm,
    analyze,
    classify_phenomenon,
    classify_seed_sign,
    preregistration_missing,
    validate_config,
)


class Width5GateBTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = load_config("configs/width5_gate_b_0901.yaml")

    def test_frozen_config_unblocks_all_result_stages(self) -> None:
        validate_config(self.cfg, stage="implementation")
        self.assertEqual(preregistration_missing(self.cfg), [])
        for stage in ("preflight", "smoke", "full", "analyze"):
            with self.subTest(stage=stage):
                validate_config(self.cfg, stage=stage)

    def test_each_freeze_flag_blocks_result_stages(self) -> None:
        for key in ("new_predictions_confirmed", "frozen",
                    "repo_spec_committed", "execution_authorized"):
            with self.subTest(key=key):
                cfg = copy.deepcopy(self.cfg)
                cfg["preregistration"][key] = False
                if key == "new_predictions_confirmed":
                    for prediction in cfg["preregistration"]["new_predictions"]:
                        cfg["preregistration"]["new_predictions"][prediction] = None
                    cfg["preregistration"]["prediction_provenance"] = (
                        "pending_Issa_entry")
                with self.assertRaisesRegex(ValueError, "not frozen"):
                    validate_config(cfg, stage="preflight")

    def test_unconfirmed_prediction_cells_must_stay_empty(self) -> None:
        cfg = copy.deepcopy(self.cfg)
        cfg["preregistration"]["new_predictions_confirmed"] = False
        cfg["preregistration"]["prediction_provenance"] = "pending_Issa_entry"
        cfg["preregistration"]["new_predictions"][
            "R5_k_above_LIN5_5m"] = "at_least_15_of_20"
        with self.assertRaisesRegex(ValueError, "must remain empty"):
            validate_config(cfg, stage="implementation")

    def test_frozen_predictions_cannot_drift(self) -> None:
        cfg = copy.deepcopy(self.cfg)
        cfg["preregistration"]["new_predictions"][
            "R5_k_above_LIN5_5m"] = "unknown"
        with self.assertRaisesRegex(ValueError, "predictions changed"):
            validate_config(cfg, stage="implementation")

    def test_predecessor_structure_is_still_locked(self) -> None:
        cfg = copy.deepcopy(self.cfg)
        cfg["arms"][0]["hidden"] = [6]
        with self.assertRaisesRegex(ValueError, "differs from the decided design"):
            validate_config(cfg, stage="implementation")

    def test_exact_binomial_reachable_labels(self) -> None:
        expected = {
            0: "A_BELOW_LINEAR",
            5: "A_BELOW_LINEAR",
            6: "A_INCONCLUSIVE_WIDE",
            8: "A_INCONCLUSIVE_WIDE",
            9: "A_NOT_SEPARATED_TIGHT",
            10: "A_NOT_SEPARATED_TIGHT",
            11: "A_NOT_SEPARATED_TIGHT",
            12: "A_INCONCLUSIVE_WIDE",
            14: "A_INCONCLUSIVE_WIDE",
            15: "A_ABOVE_LINEAR",
            20: "A_ABOVE_LINEAR",
        }
        for k, label in expected.items():
            with self.subTest(k=k):
                self.assertEqual(
                    classify_seed_sign("A", k, 20)["status"], label)
        self.assertAlmostEqual(
            classify_seed_sign("A", 15, 20)["cp95_lo"], 0.508954, places=5)
        self.assertAlmostEqual(
            classify_seed_sign("A", 5, 20)["cp95_hi"], 0.491046, places=5)

    def test_main_phenomenon_combination_table(self) -> None:
        def row(label: str) -> dict:
            return {"status": label}

        self.assertEqual(classify_phenomenon({
            "LR5": row("LR5_ABOVE_LINEAR"),
            "E5": row("E5_INCONCLUSIVE_WIDE"),
        }), "PHENOMENON3_REPRODUCED")
        self.assertEqual(classify_phenomenon({
            "LR5": row("LR5_BELOW_LINEAR"),
            "E5": row("E5_NOT_SEPARATED_TIGHT"),
        }), "PHENOMENON3_NOT_REPRODUCED")
        self.assertEqual(classify_phenomenon({
            "LR5": row("LR5_BELOW_LINEAR"),
            "E5": row("E5_INCONCLUSIVE_WIDE"),
        }), "PHENOMENON3_INCONCLUSIVE")

    def test_pair_endpoint_drops_only_nonfinite_pairs_and_counts_crossing(self) -> None:
        n = 20
        windows = {
            "R5": {
                "early": {"raw_u": np.full(n, 0.4),
                          "raw_min": np.full(n, 0.1)},
                "5M": {"raw_u": np.full(n, 0.5)},
            },
            "LIN5": {
                "early": {"raw_u": np.full(n, 0.3),
                          "raw_min": np.full(n, 0.2)},
                "5M": {"raw_u": np.full(n, 0.2)},
            },
        }
        windows["LIN5"]["5M"]["raw_u"][3] = np.nan
        sign, crossing = _paired_endpoint(self.cfg, "R5", windows)
        self.assertEqual(sign["n"], 19)
        self.assertEqual(sign["k"], 19)
        self.assertEqual(sign["excluded_seed_indices"], [3])
        self.assertEqual(sign["status"], "R5_ABOVE_LINEAR")
        self.assertEqual(crossing["n"], 19)
        self.assertEqual(crossing["c"], 19)
        self.assertEqual(crossing["early_better_count"], 19)
        self.assertEqual(crossing["status"], "R5_CROSSING_DEFINED")

    def test_lin_reference_preflight_rejects_nonfinite_seed(self) -> None:
        cfg = copy.deepcopy(self.cfg)
        cfg["common"]["seeds"] = [0, 1]
        steps = np.arange(2, 12, dtype=np.int64) * 10_000
        with tempfile.TemporaryDirectory() as tmp:
            logs = Path(tmp) / "logs"
            logs.mkdir()
            np.savez_compressed(
                logs / "LIN5_seed0.npz", step=steps,
                unfit=np.full(len(steps), 0.2))
            bad = np.full(len(steps), 0.2)
            bad[4] = np.nan
            np.savez_compressed(
                logs / "LIN5_seed1.npz", step=steps, unfit=bad)
            result = _lin_ref_preflight(cfg, Path(tmp))
        self.assertFalse(result["pass_"])
        self.assertEqual(result["nonfinite_seeds"], [1])
        self.assertEqual(result["n_valid"], 1)

    def test_two_step_unregistered_smoke_covers_reused_activation_paths(self) -> None:
        cfg = copy.deepcopy(self.cfg)
        cfg["common"]["generator_offset"] = 999_999_999
        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp)
            for arm in ("R5", "LR5", "E5", "LIN5"):
                with self.subTest(arm=arm):
                    result = _run_arm(
                        cfg, arm, "cpu", outdir / arm, [0], 2)
                    self.assertEqual(result["status"], "COMPLETE")
                    self.assertTrue(result["sanity"]["pass_"])

    def test_synthetic_logs_exercise_registered_sign_and_crossing_analysis(self) -> None:
        steps = np.array(
            [task * 10_000 for task in
             list(range(2, 12)) + list(range(91, 101))
             + list(range(491, 501))],
            dtype=np.int64,
        )
        early = {
            "R5": 0.10, "LR5": 0.10, "E5": 0.10, "LIN5": 0.30,
            "R100": 0.20, "LR100": 0.20,
            "E100": 0.20, "LIN100": 0.20,
        }
        late = {
            "R5": 0.50, "LR5": 0.45, "E5": 0.10, "LIN5": 0.20,
            "R100": 0.20, "LR100": 0.20,
            "E100": 0.20, "LIN100": 0.20,
        }
        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp)
            (outdir / "logs").mkdir()
            for arm in ARM_ORDER:
                width = int(_arm(self.cfg, arm)["hidden"][0])
                for seed in self.cfg["common"]["seeds"]:
                    unfit = np.full(len(steps), 0.25, dtype=np.float64)
                    unfit[:10] = early[arm]
                    unfit[-10:] = late[arm]
                    payload = {"step": steps, "unfit": unfit}
                    for key in RUN_KEYS[1:]:
                        payload[key] = np.full(
                            len(steps), 0.5, dtype=np.float64)
                    for key in LAYER_KEYS:
                        value = np.full(
                            len(steps), 0.5, dtype=np.float64)
                        if key in ("layer1_strict_dead",
                                   "layer1_submerged"):
                            value[:] = width // 2
                        if ("mobility_mass" in key
                                or "mobility_tilde" in key) \
                                and arm.startswith("LIN"):
                            value[:] = np.nan
                        payload[key] = value
                    np.savez_compressed(
                        outdir / "logs" / f"{arm}_seed{seed}.npz",
                        **payload)
            result = analyze(self.cfg, outdir, {}, {}, {})

            self.assertEqual(
                result["main_verdict"], "PHENOMENON3_REPRODUCED")
            self.assertEqual(
                result["signs"]["R5"]["status"], "R5_ABOVE_LINEAR")
            self.assertEqual(
                result["signs"]["LR5"]["status"], "LR5_ABOVE_LINEAR")
            self.assertEqual(
                result["signs"]["E5"]["status"], "E5_BELOW_LINEAR")
            self.assertEqual(
                result["crossings_registered_secondary"]["R5"]["c"], 20)
            self.assertEqual(
                result["crossings_registered_secondary"]["LR5"]["c"], 20)
            self.assertEqual(
                result["crossings_registered_secondary"]["E5"]["c"], 0)
            for name in ("verdict.csv", "crossing.csv", "levels.csv",
                         "mechanism.csv", "summary.md"):
                self.assertTrue((outdir / name).exists())

            with (outdir / "crossing.csv").open(
                    newline="", encoding="utf-8") as fh:
                crossing_rows = list(csv.DictReader(fh))
            self.assertEqual(len(crossing_rows), 3)
            with (outdir / "levels.csv").open(
                    newline="", encoding="utf-8") as fh:
                level_rows = list(csv.DictReader(fh))
            self.assertEqual(
                sum(row["kind"] == "G1_CONTRAST"
                    for row in level_rows), 2)
            self.assertEqual(
                sum(row["kind"] == "G2_ARM_LEVEL"
                    for row in level_rows), 8)


if __name__ == "__main__":
    unittest.main()
