"""bwd_leak_0902 の単体テスト（spec `specs/spec_bwd_leak_0902.md`）。

    OMP_NUM_THREADS=1 .venv/bin/python -m unittest src.test_bwd_leak_0902 -v
"""
from __future__ import annotations

import copy
import csv
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src import bwd_leak_0902 as B
from src.common import ROOT, load_config
from src.mlp2_phase0b import _window_indices
from src.nets import VecMLPL


CFG_PATH = Path(ROOT) / "configs" / "bwd_leak_0902.yaml"
CFG = load_config(str(CFG_PATH))
GRID = torch.linspace(-30, 30, 4001, dtype=torch.float64)
PARENT_LOGS = Path(ROOT) / CFG["controls"]["reference_run"] / "logs"


def _net(act: str, alpha: float) -> VecMLPL:
    net = VecMLPL(1, [2], 2, torch.Generator().manual_seed(0), "cpu")
    return net.set_activation(act, alpha, "alpha_exp")


class ActivationTests(unittest.TestCase):
    """S-cross / S-limit の静的側。"""

    def test_surrogates_are_the_existing_halves(self):
        relu, leaky = _net("relu", 1.0), _net("leaky_relu", 0.1)
        bl, fl = _net("bwd_leaky", 0.1), _net("fwd_leaky", 0.1)
        a_relu, a_leaky = relu.act_fn(GRID), leaky.act_fn(GRID)
        self.assertTrue(torch.equal(bl.act_fn(GRID), a_relu))
        self.assertTrue(torch.equal(fl.act_fn(GRID), a_leaky))
        self.assertTrue(torch.equal(bl.act_grad(GRID, bl.act_fn(GRID)),
                                    leaky.act_grad(GRID, a_leaky)))
        self.assertTrue(torch.equal(fl.act_grad(GRID, fl.act_fn(GRID)),
                                    relu.act_grad(GRID, a_relu)))

    def test_surrogates_are_not_swapped(self):
        leaky = _net("leaky_relu", 0.1)
        bl, fl = _net("bwd_leaky", 0.1), _net("fwd_leaky", 0.1)
        self.assertFalse(torch.equal(bl.act_fn(GRID), leaky.act_fn(GRID)))
        self.assertFalse(torch.equal(fl.act_grad(GRID, fl.act_fn(GRID)),
                                     leaky.act_grad(GRID, leaky.act_fn(GRID))))

    def test_slope_zero_is_relu(self):
        relu = _net("relu", 1.0)
        for act in ("bwd_leaky", "fwd_leaky"):
            net = _net(act, 0.0)
            self.assertTrue(torch.equal(net.act_fn(GRID), relu.act_fn(GRID)), act)
            self.assertTrue(torch.equal(net.act_grad(GRID, net.act_fn(GRID)),
                                        relu.act_grad(GRID, relu.act_fn(GRID))), act)

    def test_existing_paths_are_untouched(self):
        relu = _net("relu", 1.0)
        self.assertTrue(torch.equal(relu.act_fn(GRID), torch.relu(GRID)))
        self.assertTrue(torch.equal(relu.act_grad(GRID, relu.act_fn(GRID)),
                                    (GRID > 0).to(GRID.dtype)))
        leaky = _net("leaky_relu", 0.1)
        self.assertTrue(torch.equal(leaky.act_fn(GRID),
                                    torch.where(GRID > 0, GRID, 0.1 * GRID)))
        elu = _net("elu", 1.0)
        self.assertTrue(torch.equal(elu.act_fn(GRID),
                                    torch.where(GRID > 0, GRID, torch.expm1(GRID))))

    def test_slope_is_validated(self):
        for act in ("bwd_leaky", "fwd_leaky"):
            with self.assertRaises(ValueError):
                _net(act, 1.5)
            with self.assertRaises(ValueError):
                _net(act, -0.1)
        with self.assertRaises(ValueError):
            _net("no_such_activation", 0.1)

    def test_v_freezes_only_for_bwd_leaky(self):
        """§2 の表: `BL` は v が凍結し、`FL` は v が学習し続ける。"""
        x = torch.tensor([[1.0, -1.0]])
        y = torch.tensor([0.3])
        for act, expect_zero in (("bwd_leaky", True), ("fwd_leaky", False)):
            net = _net(act, 0.1)
            net.Ws[0].zero_()
            net.bs[0].copy_(torch.tensor([[-2.0, -3.0]]))
            net.v.copy_(torch.tensor([[0.5, -0.5]]))
            net.c.zero_()
            net.W, net.b = net.Ws[0], net.bs[0]
            pres, acts, yhat = net.forward_layers(x)
            _, _, gv, _ = net.grads_layers(x, pres, acts, yhat - y)
            self.assertEqual(bool(torch.all(gv == 0.0)), expect_zero, act)


class SanityGateTests(unittest.TestCase):
    def test_s_cross(self):
        self.assertTrue(B._s_cross(CFG)["pass_"])

    def test_s_bwd(self):
        got = B._s_bwd(CFG)
        self.assertTrue(got["pass_"], got["failures"])
        bl = next(r for r in got["rows"] if r["arm"] == "BL")
        self.assertTrue(bl["gv_zero_on_negative"])
        self.assertTrue(bl["gb_gW_closed_form"])
        self.assertTrue(bl["slope_proportional"])
        self.assertTrue(bl["applied_step_bit_exact"])
        self.assertTrue(bl["v_unchanged_on_negative"])
        fl = next(r for r in got["rows"] if r["arm"] == "FL")
        self.assertTrue(fl["gb_gW_zero_on_negative"])
        self.assertTrue(fl["gv_closed_form"])
        self.assertTrue(fl["forward_is_leaky"])
        self.assertTrue(fl["gv_nonzero_on_negative"])
        self.assertTrue(fl["w_b_unchanged_on_negative"])

    def test_s_bwd_catches_a_swapped_surrogate(self):
        """代替勾配を取り違えたら S-bwd が落ちること（gate が効いている証明）。"""
        original = VecMLPL.act_grad

        def swapped(self, pre, a):
            if self.act == "bwd_leaky":
                return (pre > 0).to(pre.dtype)      # わざと FL の勾配にする
            return original(self, pre, a)

        VecMLPL.act_grad = swapped
        try:
            self.assertFalse(B._s_bwd(CFG)["pass_"])
        finally:
            VecMLPL.act_grad = original

    def test_s_wd(self):
        got = B._s_wd(CFG)
        self.assertTrue(got["pass_"], got)
        self.assertTrue(got["W_v_c_untouched"])
        self.assertGreater(got["bias_delta_signal"], 0.0)

    def test_s_ci_selftest(self):
        self.assertTrue(B._s_ci_selftest(CFG)["pass_"])


class VerdictTableTests(unittest.TestCase):
    def setUp(self):
        self.G = B._P(CFG)

    def test_v1_labels(self):
        self.assertEqual(B._v1_label(self.G, "zero", "present"), "GRADIENT_CARRIES")
        self.assertEqual(B._v1_label(self.G, "present", "zero"), "OUTPUT_CARRIES")
        self.assertEqual(B._v1_label(self.G, "zero", "zero"), "EITHER_SUFFICES")
        self.assertEqual(B._v1_label(self.G, "present", "present"), "BOTH_REQUIRED")
        for bl, fl in (("mid", "zero"), ("mid", "mid"), ("zero", "mid"),
                       ("present", "mid"), ("mid", "present")):
            self.assertEqual(B._v1_label(self.G, bl, fl), "PARTIAL", (bl, fl))

    def test_v2_is_not_applicable_when_bl_is_zero(self):
        for blw in ("zero", "mid", "present"):
            for rw in ("zero", "mid", "present"):
                self.assertEqual(B._v2_label(self.G, "zero", blw, rw),
                                 "NOT_APPLICABLE", (blw, rw))

    def test_v2_labels(self):
        self.assertEqual(B._v2_label(self.G, "present", "zero", "present"),
                         "RESTORING_FORCE_REQUIRED")
        self.assertEqual(B._v2_label(self.G, "present", "present", "zero"),
                         "WD_B_SUFFICIENT_ALONE")
        self.assertEqual(B._v2_label(self.G, "present", "present", "present"),
                         "COMPROMISE_FAILS")

    def test_mid_cells_do_not_skip_partial(self):
        """追補 7: ワイルドカード行が `mid` セルを吸わないこと。"""
        self.assertEqual(B._v2_label(self.G, "present", "mid", "mid"), "PARTIAL")
        self.assertEqual(B._v2_label(self.G, "present", "mid", "present"), "PARTIAL")
        self.assertEqual(B._v2_label(self.G, "present", "zero", "mid"), "PARTIAL")
        # ワイルドカード行が本当に当たるセルは PARTIAL ではない
        self.assertEqual(B._v2_label(self.G, "present", "mid", "zero"),
                         "WD_B_SUFFICIENT_ALONE")

    def test_co_satisfied_rows_are_all_reported(self):
        self.assertEqual(B._v2_co_satisfied("present", "zero"),
                         ["WD_B_SUFFICIENT_ALONE", "COMPROMISE_FAILS"])
        self.assertEqual(B._v2_co_satisfied("zero", "present"),
                         ["RESTORING_FORCE_REQUIRED"])
        self.assertEqual(B._v2_co_satisfied("mid", "mid"), ["PARTIAL"])

    def test_enumerated_maps_agree_with_the_spec_table(self):
        self.assertFalse(B._verdict_maps_disagree_with_spec_table(self.G))

    def test_onset_state(self):
        self.assertEqual(B._onset_state([0, 0], 0, 5), "zero")
        self.assertEqual(B._onset_state([0, 7], 0, 5), "present")
        self.assertEqual(B._onset_state([1, 3], 0, 5), "mid")
        self.assertEqual(B._onset_state([0, 1], 0, 5), "mid")
        self.assertEqual(B._onset_state([], 0, 5), "missing")

    def test_p5_label_order_and_below_zero_flag(self):
        G = self.G
        self.assertEqual(
            B._p5_label(G, dict(percentile_ci_lo=-0.10, percentile_ci_hi=0.10))[0],
            "EQUIV_FWD")
        self.assertEqual(
            B._p5_label(G, dict(percentile_ci_lo=0.30, percentile_ci_hi=0.90))[0],
            "SHORT_OF_LR")
        # 重なるケースは書かれた順（等価が先）
        self.assertEqual(
            B._p5_label(G, dict(percentile_ci_lo=0.02, percentile_ci_hi=0.10))[0],
            "EQUIV_FWD")
        self.assertEqual(
            B._p5_label(G, dict(percentile_ci_lo=-0.9, percentile_ci_hi=0.9)),
            ("INCONCLUSIVE_WIDE", False))
        # CI が丸ごと 0 の下は登録ラベルが無いのでフラグで残す（追補 1）
        self.assertEqual(
            B._p5_label(G, dict(percentile_ci_lo=-1.2, percentile_ci_hi=-0.5)),
            ("INCONCLUSIVE_WIDE", True))

    def test_sign_test(self):
        self.assertAlmostEqual(
            B._sign_test(np.array([1.0] * 10))["p_two_sided"], 2 / 1024, places=9)
        got = B._sign_test(np.array([1.0, -1.0, 0.0]))
        self.assertEqual((got["n_positive"], got["n_negative"], got["n_ties"]),
                         (1, 1, 1))


class FrozenDesignTests(unittest.TestCase):
    MUTATIONS = [
        ("wd_b on BL", lambda c: c["arms"][0].__setitem__("wd_b", 1e-3)),
        ("FL becomes BL", lambda c: c["arms"][2].__setitem__("activation", "bwd_leaky")),
        ("dose", lambda c: c["arms"][0].__setitem__("target_dose", 9.34)),
        ("stage", lambda c: c["arms"][0].__setitem__("stage", 2)),
        ("bootstrap seed", lambda c: c["phase1"].__setitem__("bootstrap_seed", 20260829)),
        ("floor", lambda c: c["phase1"].__setitem__("unfit_floor", 1e-23)),
        ("generator offset", lambda c: c["common"].__setitem__("generator_offset", 20260905)),
        ("BL derivative", lambda c: c["activation"]["bwd_leaky"].__setitem__("derivative", "relu")),
        ("FL forward", lambda c: c["activation"]["fwd_leaky"].__setitem__("forward", "relu")),
        ("P5 margin", lambda c: c["bwd_leak"].__setitem__("p5_equivalence_margin", 0.30)),
        ("V1 map", lambda c: c["bwd_leak"]["v1_map"].__setitem__("zero_present", "OUTPUT_CARRIES")),
        ("V2 map", lambda c: c["bwd_leak"]["v2_map"].__setitem__("present_zero", "COMPROMISE_FAILS")),
        ("revival def", lambda c: c["bwd_leak"]["revival"].__setitem__("primary_condition", "any_record")),
        ("P6 label", lambda c: c["bwd_leak"].__setitem__("p6_emit_label", True)),
        ("lambda", lambda c: c["bias_wd"].__setitem__("wd_b_lambda", 1e-2)),
        ("P7 aggregation order", lambda c: c["bwd_leak"]["s_distribution"].__setitem__(
            "aggregation_order", ["unit", "median_over_seeds"])),
        ("control expectation", lambda c: c["bwd_leak"]["control_expected_onset_5m"]
            .__setitem__("LR_933", 5)),
    ]

    def test_config_validates(self):
        for stage in ("preflight", "smoke", "run", "analyze"):
            B.validate_config(copy.deepcopy(CFG), stage=stage)

    def test_tampering_is_rejected(self):
        for name, mutate in self.MUTATIONS:
            with self.subTest(name):
                c = copy.deepcopy(CFG)
                mutate(c)
                with self.assertRaises(ValueError):
                    B.validate_config(c, stage="run")

    def test_arm_selection(self):
        self.assertEqual(B._selected_arms(CFG, "1", "both"), list(B.STAGE_ARMS[1]))
        self.assertEqual(B._selected_arms(CFG, "1", "1216"), ["BL_1216", "FL_1216"])
        self.assertEqual(B._selected_arms(CFG, "2", "both"), list(B.STAGE_ARMS[2]))
        self.assertEqual(B._selected_arms(CFG, "all", "both"), list(B.ARM_ORDER))


class WindowAndControlTests(unittest.TestCase):
    def test_window_is_ten_task_end_records(self):
        """追補 3: 5M 窓はタスク終端 10 点（記録点 100 点ではない）。"""
        steps = np.arange(0, 5_000_001, 1000, dtype=np.int64)
        idx = _window_indices(steps, 10_000, [491, 500])
        self.assertEqual(len(idx), 10)
        self.assertEqual(steps[idx].tolist(),
                         list(range(4_910_000, 5_000_001, 10_000)))
        idx1m = _window_indices(steps, 10_000, [91, 100])
        self.assertEqual(len(idx1m), 10)
        self.assertEqual(int(steps[idx1m][-1]), 1_000_000)

    def test_controls_are_transcribed(self):
        controls = B._load_controls(CFG)
        self.assertEqual(set(controls), set(B.CONTROL_ORDER))
        for values in controls.values():
            self.assertEqual(values["u_5m"].shape, (10,))
            self.assertTrue(values["source"].endswith("gate_dose_0830/verdict.csv"))
        self.assertEqual({a: controls[a]["n_onset_5m"] for a in B.CONTROL_ORDER},
                         dict(B._P(CFG)["control_expected_onset_5m"]))


@unittest.skipUnless((PARENT_LOGS / "R_933_seed0.npz").exists(),
                     "parent logs are gitignored and absent")
class ParentLogTests(unittest.TestCase):
    def test_revival_separates_within_task_from_boundary(self):
        """追補 6: 「ReLU 腕は 0」が成り立つのは within-task の定義のみ。"""
        got = B._revival_counts(PARENT_LOGS / "R_933_seed0.npz")
        self.assertEqual(got["events_within_task"], 0)
        self.assertGreater(got["events_across_boundary"], 0)

    def test_s_log_branch_is_A(self):
        got = B._s_log_branch(CFG)
        self.assertTrue(got["pass_"], got["failures"][:3])
        self.assertEqual(got["branch"], "A")
        shapes = got["rows"][0]["shapes"]
        self.assertEqual(shapes["layer1_M"][1], 100)
        self.assertEqual(shapes["layer1_B"][1], 100)

    def test_p7_seed_values_are_finite_for_controls(self):
        got = B._p7_seed_values(PARENT_LOGS / "R_933_seed0.npz", CFG)
        for key in ("median_s__all", "median_M__all", "median_B__all",
                    "median_denom__all", "median_zbar__all"):
            self.assertTrue(np.isfinite(got[key]), key)
        # 縮退 (ii): median_s と median_M + median_B は別物
        self.assertNotAlmostEqual(got["median_s__all"],
                                  got["median_M_plus_median_B__all"], places=6)

    def test_s_ref_hashes_match(self):
        got = B._s_ref(CFG)
        self.assertTrue(got["pass_"], got)


@unittest.skipUnless((PARENT_LOGS / "R_933_seed0.npz").exists(),
                     "parent logs are gitignored and absent")
class AnalysisEndToEndTests(unittest.TestCase):
    """親走の実 5M ログを新規腕の名前で置いて `analyze` を通す。

    軌道の中身は本走のものではない（対照のログを流用しているだけ）ので、**出る
    数値は結果ではない**。ここで見るのは集計経路が最後まで走り、CSV のスキーマが
    崩れず、判定ラベルが登録どおり導かれること。
    """

    SOURCE = {"BL_933": "LR_933", "BL_1216": "LR_1216",
              "FL_933": "R_933", "FL_1216": "R_1216"}

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        out = Path(self.tmp.name) / "stage1"
        (out / "logs").mkdir(parents=True)
        for arm, src in self.SOURCE.items():
            for seed in range(10):
                os.symlink(PARENT_LOGS / f"{src}_seed{seed}.npz",
                           out / "logs" / f"{arm}_seed{seed}.npz")
        self.out = out

    def tearDown(self):
        self.tmp.cleanup()

    def test_analyze_runs_and_labels_follow_the_registered_map(self):
        result = B.analyze(CFG, self.out, list(B.STAGE_ARMS[1]), "1",
                           sanity={}, elapsed={}, divergences={}, stage1_dir=None)
        # BL <- LR (0/10), FL <- R (10/10) なので登録表どおり GRADIENT_CARRIES
        self.assertEqual(result["onset_states"]["BL"], "zero")
        self.assertEqual(result["onset_states"]["FL"], "present")
        self.assertEqual(result["V1"], "GRADIENT_CARRIES")
        self.assertEqual(result["V2"], "NOT_APPLICABLE")
        for name in ("verdict.csv", "layer_stats.csv", "s_distribution.csv",
                     "s_contrasts.csv", "revival.csv", "freeze_rates.csv",
                     "summary.md"):
            self.assertTrue((self.out / name).exists(), name)

    def test_every_csv_row_has_the_full_schema(self):
        B.analyze(CFG, self.out, list(B.STAGE_ARMS[1]), "1", sanity={},
                  elapsed={}, divergences={}, stage1_dir=None)
        for name in ("verdict.csv", "layer_stats.csv", "s_distribution.csv",
                     "s_contrasts.csv", "revival.csv", "freeze_rates.csv"):
            with (self.out / name).open(newline="") as fh:
                rows = list(csv.DictReader(fh))
            self.assertTrue(rows, name)
            fields = set(rows[0])
            for row in rows:
                self.assertEqual(set(row), fields, f"{name}: ragged row {row}")
                self.assertNotIn(None, row.values(), f"{name}: short row")

    def test_p5_is_computed_against_the_committed_controls(self):
        result = B.analyze(CFG, self.out, list(B.STAGE_ARMS[1]), "1", sanity={},
                           elapsed={}, divergences={}, stage1_dir=None)
        p5 = {k: v for k, v in result["contrasts"].items() if v["kind"] == "P5"}
        self.assertEqual(len(p5), 4)
        for key, value in p5.items():
            self.assertEqual(value["status"], "OK", key)
            self.assertTrue(value["cross_run"], key)
            self.assertEqual(value["n_paired"], 10, key)
            self.assertIn(value["label"],
                          {"EQUIV_FWD", "SHORT_OF_LR", "INCONCLUSIVE_WIDE"})
            self.assertIn("p_two_sided", value["sign_test"])
        # BL_933 <- LR_933 なので BL-LR は自分自身との差 = 恒等的に 0
        identity = p5["BL_933_minus_LR_933"]
        self.assertEqual(identity["seed_values"], [0.0] * 10)

    def test_verdict_marks_controls_as_a_different_run(self):
        B.analyze(CFG, self.out, list(B.STAGE_ARMS[1]), "1", sanity={},
                  elapsed={}, divergences={}, stage1_dir=None)
        with (self.out / "verdict.csv").open(newline="") as fh:
            rows = {r["arm"]: r for r in csv.DictReader(fh)}
        for arm in B.CONTROL_ORDER:
            self.assertEqual(rows[arm]["status"], "COMMITTED_OTHER_RUN")
            self.assertEqual(rows[arm]["is_control"], "1")
        for arm in B.STAGE_ARMS[1]:
            self.assertEqual(rows[arm]["status"], "COMPLETE")
            self.assertEqual(rows[arm]["is_control"], "0")
        text = (self.out / "summary.md").read_text(encoding="utf-8")
        self.assertIn("別走", text)
        self.assertIn("within-task", text)

    def test_p7_emits_both_degenerate_forms_and_raw_series(self):
        B.analyze(CFG, self.out, list(B.STAGE_ARMS[1]), "1", sanity={},
                  elapsed={}, divergences={}, stage1_dir=None)
        with (self.out / "s_distribution.csv").open(newline="") as fh:
            rows = list(csv.DictReader(fh))
        fields = set(rows[0])
        for key in ("median_s__all", "median_M_plus_median_B__all",
                    "median_denom__all", "median_zbar__all", "median_wmu__all",
                    "median_b_raw__all", "median_s__p_hat_positive"):
            self.assertIn(key, fields, key)
        arms = {r["arm"] for r in rows}
        self.assertTrue(set(B.CONTROL_ORDER) <= arms)

    def test_control_onset_mismatch_blocks_the_analysis(self):
        c = copy.deepcopy(CFG)
        c["bwd_leak"]["control_expected_onset_5m"]["LR_933"] = 9
        with self.assertRaises(B.SanityError):
            B.analyze(c, self.out, list(B.STAGE_ARMS[1]), "1", sanity={},
                      elapsed={}, divergences={}, stage1_dir=None)


if __name__ == "__main__":
    unittest.main()
