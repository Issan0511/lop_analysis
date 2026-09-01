from __future__ import annotations

import copy
import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.common import load_config
from src.width5_gate_0901 import (
    ARM_ORDER,
    LAYER_KEYS,
    RUN_KEYS,
    WIDTH5_ARMS,
    WidthGateRecorder,
    _arm,
    _run_arm,
    _s_linear,
    _s_mobility,
    _s_rank,
    analyze,
    classify_g0,
    classify_level,
    exact_layer_record_width,
    preregistration_missing,
    setup_arm_width,
    validate_config,
    write_arm_logs,
)


class Width5GateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = load_config("configs/width5_gate_0901.yaml")

    def test_frozen_config_unblocks_result_stages(self) -> None:
        validate_config(self.cfg, stage="implementation")
        self.assertEqual(preregistration_missing(self.cfg), [])
        for stage in ("preflight", "smoke", "full", "analyze"):
            with self.subTest(stage=stage):
                validate_config(self.cfg, stage=stage)

    def test_each_freeze_flag_blocks_result_stages(self) -> None:
        for key in ("predictions_confirmed", "frozen", "repo_spec_committed",
                    "execution_authorized"):
            with self.subTest(key=key):
                cfg = copy.deepcopy(self.cfg)
                cfg["preregistration"][key] = False
                with self.assertRaisesRegex(ValueError, "not frozen"):
                    validate_config(cfg, stage="preflight")

    def test_frozen_predictions_cannot_drift(self) -> None:
        cfg = copy.deepcopy(self.cfg)
        cfg["preregistration"]["predictions"]["R5_n_onset_5m"] = "unknown"
        with self.assertRaisesRegex(ValueError, "predictions changed"):
            validate_config(cfg, stage="implementation")

    def test_g0_registered_decision_table_and_saturation_guard(self) -> None:
        base = dict.fromkeys(WIDTH5_ARMS, 0)
        cases = [
            ({**base}, "BINARY_SATURATED_NO_VERDICT"),
            (dict.fromkeys(WIDTH5_ARMS, 20), "BINARY_SATURATED_NO_VERDICT"),
            ({**base, "R5": 20}, "PHENOMENON3_NOT_REPRODUCED"),
            ({**base, "R5": 20, "LR5": 1}, "PHENOMENON3_MARGINAL"),
            ({**base, "R5": 20, "LR5": 5}, "PHENOMENON3_REPRODUCED"),
            ({**base, "R5": 20, "E5": 12}, "PHENOMENON3_REPRODUCED"),
        ]
        for onset, expected in cases:
            with self.subTest(onset=onset):
                self.assertEqual(classify_g0(onset), expected)
        self.assertEqual(classify_g0({**base, "R5": 20}, missing={"E5"}),
                         "INCONCLUSIVE_DIVERGENCE")

    def test_g1_report_only_resolution_labels(self) -> None:
        self.assertEqual(classify_level(0.6, 0.1, 1.0, 0.5, "LEAKY"),
                         "LEAKY_ABOVE_LINEAR")
        self.assertEqual(classify_level(-0.7, -1.0, -0.1, 0.5, "ELU"),
                         "ELU_BELOW_LINEAR")
        self.assertEqual(classify_level(0.1, -0.2, 0.3, 0.5, "LEAKY"),
                         "LEAKY_WITHIN_RESOLUTION")
        self.assertEqual(classify_level(0.1, -0.7, 0.8, 0.5, "LEAKY"),
                         "INCONCLUSIVE_WIDE")

    def test_pure_sanity_helpers(self) -> None:
        self.assertTrue(_s_linear(self.cfg)["pass_"])
        self.assertTrue(_s_mobility(self.cfg)["pass_"])
        self.assertTrue(_s_rank(self.cfg)["pass_"])

    def test_linear_exact_record_keeps_raw_mobility_and_marks_tilde_undefined(self) -> None:
        cfg = copy.deepcopy(self.cfg)
        cfg["common"]["seeds"] = [0, 1]
        st = setup_arm_width(cfg, _arm(cfg, "LIN5"), "cpu")
        rec, sanity = exact_layer_record_width(
            st,
            float(cfg["sanity"]["sigma_degenerate_tol"]),
            float(cfg["width5_gate"]["mobility_floor_tolerance"]),
        )
        layer = rec["layers"][0]
        self.assertTrue(sanity["run_finite"])
        self.assertTrue(np.all(layer["mobility"].cpu().numpy() == 1.0))
        self.assertTrue(np.isnan(layer["mobility_tilde"].cpu().numpy()).all())
        self.assertTrue(np.isnan(
            layer["centered_eff_rank_per_mobility_mass"].cpu().numpy()).all())
        rc = layer["centered_eff_rank"].cpu().numpy()
        self.assertTrue(np.all((rc >= 1.0) & (rc <= 5.0)))

    def test_initial_state_recorder_writes_all_new_logger_columns(self) -> None:
        cfg = copy.deepcopy(self.cfg)
        cfg["common"]["seeds"] = [0, 1]
        st = setup_arm_width(cfg, _arm(cfg, "E5"), "cpu")
        recorder = WidthGateRecorder([0], st, cfg)
        recorder(st, 0)
        self.assertTrue(recorder.sanity()["pass_"])
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_arm_logs(Path(tmp), "E5", st, recorder)
            self.assertEqual(len(paths), 2)
            with np.load(paths[0], allow_pickle=False) as z:
                for key in (
                    "layer1_s", "layer1_mobility", "layer1_mobility_tilde",
                    "layer1_median_s", "layer1_median_zbar",
                    "layer1_median_denom", "layer1_centered_eff_rank",
                    "layer1_centered_eff_rank_per_mobility_mass",
                    "layer1_mobility_weighted_wcos_abs",
                    "layer1_mobility_tilde_weighted_wcos_abs",
                ):
                    self.assertIn(key, z.files)

    def test_two_step_unregistered_smoke_covers_all_activation_paths(self) -> None:
        cfg = copy.deepcopy(self.cfg)
        cfg["common"]["generator_offset"] = 999_999_999
        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp)
            for arm in ("R5", "LR5", "E5", "LIN5"):
                with self.subTest(arm=arm):
                    result = _run_arm(cfg, arm, "cpu", outdir / arm, [0], 2)
                    self.assertEqual(result["status"], "COMPLETE")
                    self.assertTrue(result["sanity"]["pass_"])

    def test_synthetic_logs_exercise_analysis_without_scientific_data(self) -> None:
        steps = np.array(
            [task * 10_000 for task in
             list(range(2, 12)) + list(range(91, 101)) + list(range(491, 501))],
            dtype=np.int64,
        )
        late_levels = {
            "R5": 0.10, "LR5": 0.06, "E5": 0.01, "LIN5": 0.01,
            "R100": 0.10, "LR100": 0.01, "E100": 0.01, "LIN100": 0.01,
        }
        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp)
            (outdir / "logs").mkdir()
            for arm in ARM_ORDER:
                width = int(_arm(self.cfg, arm)["hidden"][0])
                for seed in self.cfg["common"]["seeds"]:
                    unfit = np.full(len(steps), 0.01, dtype=np.float64)
                    unfit[-10:] = late_levels[arm]
                    payload = {"step": steps, "unfit": unfit}
                    for key in RUN_KEYS[1:]:
                        payload[key] = np.full(len(steps), 0.5, dtype=np.float64)
                    for key in LAYER_KEYS:
                        value = np.full(len(steps), 0.5, dtype=np.float64)
                        if key in ("layer1_strict_dead", "layer1_submerged"):
                            value[:] = width // 2
                        if ("mobility_mass" in key or "mobility_tilde" in key) \
                                and arm.startswith("LIN"):
                            value[:] = np.nan
                        payload[key] = value
                    np.savez_compressed(
                        outdir / "logs" / f"{arm}_seed{seed}.npz", **payload)
            result = analyze(self.cfg, outdir, {}, {}, {})
            self.assertEqual(result["main_verdict"], "PHENOMENON3_REPRODUCED")
            self.assertTrue((outdir / "verdict.csv").exists())
            self.assertTrue((outdir / "mechanism.csv").exists())
            with (outdir / "levels.csv").open(newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(len(rows), 10)
            self.assertEqual(sum(row["kind"] == "G1_CONTRAST" for row in rows), 2)
            self.assertEqual(sum(row["kind"] == "G2_ARM_LEVEL" for row in rows), 8)


if __name__ == "__main__":
    unittest.main()
