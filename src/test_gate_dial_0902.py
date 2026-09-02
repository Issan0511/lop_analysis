"""gate_dial_0902 の単体テスト（spec `specs/spec_gate_dial_0902.md`）。

    OMP_NUM_THREADS=1 .venv/bin/python -m unittest src.test_gate_dial_0902 -v
"""
from __future__ import annotations

import copy
import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src import gate_dial_0902 as D
from src.common import ROOT, load_config
from src.nets import VecMLPL


CFG_PATH = Path(ROOT) / "configs" / "gate_dial_0902.yaml"
CFG = load_config(str(CFG_PATH))
GRID = torch.linspace(-30, 30, 4001, dtype=torch.float64)
PARENT = Path(ROOT) / CFG["controls"]["reference_run"]


def _net(act: str, alpha: float) -> VecMLPL:
    net = VecMLPL(1, [2], 2, torch.Generator().manual_seed(0), "cpu")
    return net.set_activation(act, alpha, "alpha_exp")


class ActivationTests(unittest.TestCase):
    """新規 2 活性化の閉形式（spec §4.3）。"""

    def test_beta_one_matches_torch_reference(self):
        silu, gelu = _net("silu", 1.0), _net("gelu", 1.0)
        self.assertLess(float((silu.act_fn(GRID) - F.silu(GRID)).abs().max()), 1e-14)
        self.assertLess(float((gelu.act_fn(GRID)
                               - F.gelu(GRID, approximate="none")).abs().max()), 1e-14)

    def test_backward_matches_autograd(self):
        for act in ("silu", "gelu"):
            for beta in (0.3, 1.0, 3.0):
                net = _net(act, beta)
                z = GRID.clone().requires_grad_(True)
                net.act_fn(z).sum().backward()
                got = net.act_grad(GRID, net.act_fn(GRID))
                rel = ((got - z.grad).abs()
                       / z.grad.abs().clamp_min(1e-300)).max()
                self.assertLess(float(rel), 1e-9, f"{act} beta={beta}")

    def test_gelu_is_exact_erf_not_tanh(self):
        """spec §4.3: tanh 近似は使わない。近似形と有意に違うことを見る。"""
        net = _net("gelu", 1.0)
        z = torch.linspace(-3, 3, 601, dtype=torch.float64)
        tanh_form = 0.5 * z * (1 + torch.tanh(math.sqrt(2 / math.pi)
                                              * (z + 0.044715 * z ** 3)))
        self.assertGreater(float((net.act_fn(z) - tanh_form).abs().max()), 1e-5)
        self.assertLess(float((net.act_fn(z)
                               - F.gelu(z, approximate="none")).abs().max()), 1e-14)

    def test_phi_prime_at_zero_is_one_half_for_every_beta(self):
        """不連続点の中点。全 beta で 1/2 になるのが正しい（バグではない）。"""
        zero = torch.zeros(1, dtype=torch.float64)
        for act in ("silu", "gelu"):
            for beta in (0.3, 1.0, 3.0, 10.0, 1e4):
                net = _net(act, beta)
                self.assertEqual(float(net.act_grad(zero, net.act_fn(zero))), 0.5)

    def test_existing_activation_paths_are_untouched(self):
        relu, leaky, elu = _net("relu", 1.0), _net("leaky_relu", 0.1), _net("elu", 1.0)
        self.assertTrue(torch.equal(relu.act_fn(GRID), torch.relu(GRID)))
        self.assertTrue(torch.equal(relu.act_grad(GRID, relu.act_fn(GRID)),
                                    (GRID > 0).to(GRID.dtype)))
        self.assertTrue(torch.equal(leaky.act_fn(GRID),
                                    torch.where(GRID > 0, GRID, 0.1 * GRID)))
        self.assertTrue(torch.equal(elu.act_fn(GRID),
                                    torch.where(GRID > 0, GRID, torch.expm1(GRID))))
        bl, fl = _net("bwd_leaky", 0.1), _net("fwd_leaky", 0.1)
        self.assertTrue(torch.equal(bl.act_fn(GRID), torch.relu(GRID)))
        self.assertTrue(torch.equal(fl.act_grad(GRID, fl.act_fn(GRID)),
                                    (GRID > 0).to(GRID.dtype)))

    def test_beta_is_validated(self):
        for act in ("silu", "gelu"):
            for bad in (0.0, -1.0, float("inf"), float("nan")):
                with self.assertRaises(ValueError):
                    _net(act, bad)

    def test_activation_choice_consumes_no_randomness(self):
        gen_a = torch.Generator().manual_seed(11)
        gen_b = torch.Generator().manual_seed(11)
        a = VecMLPL(3, [5], 4, gen_a, "cpu")
        b = VecMLPL(3, [5], 4, gen_b, "cpu").set_activation("gelu", 3.0)
        for key in ("W", "b", "v", "c"):
            self.assertTrue(torch.equal(a.state_dict()[key], b.state_dict()[key]))
        self.assertEqual(float(torch.rand(1, generator=gen_a)),
                         float(torch.rand(1, generator=gen_b)))


class GeometryTests(unittest.TestCase):
    """§2 の谷底と凍結深さ。閉形式の数値解であって実験出力ではない。"""

    def test_valley_scales_as_one_over_beta(self):
        for act, constant in (("silu", 1.27846), ("gelu", 0.75179)):
            for beta in (0.3, 1.0, 3.0):
                got = D.valley_depth(act, beta)
                self.assertAlmostEqual(got * beta, constant, places=4,
                                       msg=f"{act} beta={beta}")

    def test_leaky_and_elu_have_no_valley(self):
        self.assertTrue(math.isnan(D.valley_depth("leaky_relu", 0.1)))
        self.assertTrue(math.isnan(D.valley_depth("elu", 1.0)))

    def test_elu_freeze_depth_is_log_of_alpha_over_threshold(self):
        threshold = 1e-6
        for alpha in (1.0, 0.1, 0.01):
            self.assertAlmostEqual(D.freeze_depth("elu", alpha, threshold),
                                   math.log(alpha / threshold), places=3)

    def test_leaky_has_no_freeze_depth_in_the_registered_range(self):
        for a in (0.1, 0.01, 0.001, 1e-4, 1e-5):
            self.assertEqual(D.freeze_depth("leaky_relu", a, 1e-6), float("inf"))

    def test_registered_table_matches_the_numeric_roots(self):
        result = D._s_dial(CFG)
        self.assertTrue(result["pass_"], result["failures"])

    def test_freeze_threshold_is_the_displacement_bound(self):
        """凍結閾値は 0.05/(lr*T*K) であって手置きの定数ではない。"""
        G = CFG["gate_dial"]["design"]
        want = (float(CFG["phase1"]["onset_threshold"])
                / (float(CFG["common"]["lr_main"])
                   * int(CFG["common"]["total_steps"])
                   * float(G["displacement_bound_constant_K"])))
        self.assertAlmostEqual(float(G["freeze_depth_phi_prime_threshold"]), want)


class ConfigTests(unittest.TestCase):
    def test_registered_config_validates(self):
        D.validate_config(copy.deepcopy(CFG), stage="run")

    def test_mutations_are_rejected(self):
        mutations = [
            ("dial", lambda c: c["arms"][0].__setitem__("dial", 2.0)),
            ("stage", lambda c: c["arms"][0].__setitem__("stage", 2)),
            ("dose", lambda c: c["arms"][0].__setitem__("target_dose", 9.33)),
            ("bootstrap", lambda c: c["phase1"].__setitem__("bootstrap_seed", 1)),
            ("margin", lambda c: c["gate_dial"].__setitem__("v2_margin", 0.3)),
            ("offset", lambda c: c["common"].__setitem__("generator_offset", 7)),
            ("v1_map", lambda c: c["gate_dial"]["v1_map"].__setitem__(
                "zero_zero", "SOFT_GATES_RELU_SIDE")),
            ("v2_order", lambda c: c["gate_dial"].__setitem__(
                "v2_label_order", ["MONOTONE_TOWARD_RELU", "REVERSAL",
                                   "FLAT_IN_RANGE", "PARTIAL"])),
            ("K", lambda c: c["gate_dial"]["design"].__setitem__(
                "displacement_bound_constant_K", 2.0)),
            ("controls", lambda c: c["gate_dial"]["control_expected_onset_5m"]
                .__setitem__("R_1216", 9)),
            ("window", lambda c: c["phase1"].__setitem__(
                "window_records_per_10task_window", 100)),
        ]
        for name, mutate in mutations:
            cfg = copy.deepcopy(CFG)
            mutate(cfg)
            with self.assertRaises(ValueError, msg=name):
                D.validate_config(cfg, stage="run")

    def test_every_ladder_ends_at_relu_and_lists_known_arms(self):
        known = set(D.ARM_ORDER) | set(D.CONTROL_ORDER)
        for family, ladder in CFG["gate_dial"]["ladders"].items():
            self.assertEqual(ladder[-1], "R_1216", family)
            self.assertTrue(set(ladder) <= known, family)
            dials = [D._dial(CFG, a) for a in ladder[:-1]]
            if family in ("leaky", "elu"):     # 軟らかい -> 硬い = ダイヤル減少
                self.assertEqual(dials, sorted(dials, reverse=True), family)
            else:                              # SiLU/GELU は beta 増加
                self.assertEqual(dials, sorted(dials), family)

    def test_every_registered_arm_appears_in_exactly_one_ladder(self):
        seen = [a for ladder in CFG["gate_dial"]["ladders"].values()
                for a in ladder if a in D.ARM_ORDER]
        self.assertEqual(sorted(seen), sorted(D.ARM_ORDER))


class VerdictLabelTests(unittest.TestCase):
    G = CFG["gate_dial"]

    def test_v1_table(self):
        self.assertEqual(D._v1_label(self.G, "present", "present"),
                         "SOFT_GATES_RELU_SIDE")
        self.assertEqual(D._v1_label(self.G, "zero", "zero"),
                         "SOFT_GATES_SOFT_SIDE")
        self.assertEqual(D._v1_label(self.G, "present", "zero"), "SPLIT_SILU_GELU")
        self.assertEqual(D._v1_label(self.G, "zero", "present"), "SPLIT_SILU_GELU")
        for pair in (("mid", "zero"), ("mid", "mid"), ("mid", "present"),
                     ("zero", "mid"), ("present", "mid")):
            self.assertEqual(D._v1_label(self.G, *pair), "PARTIAL", pair)

    def test_onset_state(self):
        self.assertEqual(D._onset_state([0], 0, 5), "zero")
        self.assertEqual(D._onset_state([5], 0, 5), "present")
        self.assertEqual(D._onset_state([3], 0, 5), "mid")
        self.assertEqual(D._onset_state([], 0, 5), "missing")

    def test_v2_reversal_wins_over_monotone(self):
        contrasts = [dict(ci_lo=-1.2, ci_hi=-0.4)]
        label, hits = D._v2_label(self.G, contrasts, [10, 0], [], [])
        self.assertEqual(label, "REVERSAL")
        self.assertIn("REVERSAL", hits)

    def test_v2_onset_drop_of_three_is_reversal(self):
        contrasts = [dict(ci_lo=0.1, ci_hi=0.5)]
        self.assertEqual(D._v2_label(self.G, contrasts, [5, 2], [], [])[0],
                         "REVERSAL")
        # 2 しか落ちなければ REVERSAL ではないが単調でもない -> PARTIAL
        self.assertEqual(D._v2_label(self.G, contrasts, [5, 3], [], [])[0],
                         "PARTIAL")

    def test_v2_flat_beats_monotone_when_nothing_moved(self):
        flat = [dict(ci_lo=-0.05, ci_hi=0.05)]
        label, hits = D._v2_label(self.G, flat, [0, 0], flat, [0, 0])
        self.assertEqual(label, "FLAT_IN_RANGE")
        self.assertIn("MONOTONE_TOWARD_RELU", hits)   # 当たっていた行も残す

    def test_v2_monotone(self):
        rising = [dict(ci_lo=0.5, ci_hi=1.5)]
        label, _ = D._v2_label(self.G, rising, [0, 10], rising, [0, 10])
        self.assertEqual(label, "MONOTONE_TOWARD_RELU")

    def test_p5_labels(self):
        def ci(lo, hi):
            return dict(percentile_ci_lo=lo, percentile_ci_hi=hi)
        self.assertEqual(D._p5_label(self.G, ci(-0.1, 0.1), "LR_1216")[0],
                         "EQUIV_SOFT_LR_1216")
        self.assertEqual(D._p5_label(self.G, ci(0.5, 2.0), "E_1216")[0],
                         "SHORT_OF_SOFT_E_1216")
        self.assertEqual(D._p5_label(self.G, ci(-2.0, 2.0), "E_1216")[0],
                         "INCONCLUSIVE_WIDE")
        self.assertTrue(D._p5_label(self.G, ci(-2.0, -0.5), "E_1216")[1])


class WindowTests(unittest.TestCase):
    """★ 窓はタスク終端の記録点だけ（spec 字義の 100 点ではない）。"""

    def _series(self, values_by_task):
        step = np.arange(0, 5_000_001, 1000, dtype=np.int64)
        unfit = np.zeros((len(step), 1), dtype=np.float64)
        for i, s in enumerate(step):
            unfit[i, 0] = values_by_task(int(s))
        return step, unfit

    def test_window_uses_ten_task_end_records(self):
        step, unfit = self._series(lambda s: 1.0 if s % 10_000 == 0 else 99.0)
        rolled = D._rolling_window_unfit(step, unfit, CFG)
        self.assertEqual(rolled["records_per_window"], 10)
        self.assertTrue(np.allclose(rolled["u"], 1.0))
        self.assertEqual(int(rolled["k"][0]), 10)
        self.assertEqual(int(rolled["k"][-1]), 500)

    def test_rolling_window_reproduces_the_host_window(self):
        rng = np.random.default_rng(0)
        step = np.arange(0, 5_000_001, 1000, dtype=np.int64)
        unfit = rng.random((len(step), 3))
        rolled = D._rolling_window_unfit(step, unfit, CFG)
        from src.mlp2_phase0b import _window_indices
        for k, tasks in ((100, [91, 100]), (500, [491, 500])):
            idx = _window_indices(step, 10_000, tasks)
            self.assertEqual(len(idx), 10)
            want = unfit[idx].mean(axis=0)
            got = rolled["u"][np.flatnonzero(rolled["k"] == k)[0]]
            self.assertTrue(np.allclose(got, want, atol=1e-15))

    def test_onset_time_and_censoring(self):
        step = np.arange(0, 5_000_001, 1000, dtype=np.int64)
        unfit = np.zeros((len(step), 2), dtype=np.float64)
        unfit[:, 0] = 0.5                      # 常に発症水準
        unfit[:, 1] = 1e-9                     # 一度も超えない
        got = D._onset_times(CFG, step, unfit)
        self.assertEqual(got["rows"][0], dict(k_star=10, censored=0))
        self.assertEqual(got["rows"][1], dict(k_star=500, censored=1))

    def test_kaplan_meier_is_a_rate(self):
        rows = D._kaplan_meier([10, 20, 500, 500], [0, 0, 1, 1], 500)
        self.assertEqual(rows[0]["survival"], 0.75)
        self.assertEqual(rows[1]["survival"], 0.5)
        self.assertEqual(rows[-1]["survival"], 0.5)   # 打ち切りでは下がらない


class RecorderTests(unittest.TestCase):
    """新規ロガーが既知の量と一致し、軌道に触らないこと（S-mob / S-log-b）。"""

    def test_unit_extra_matches_the_exact_record(self):
        from src.elu_swamp import exact_layer_record_elu
        cfg = copy.deepcopy(CFG)
        cfg["common"]["seeds"] = [0, 1]
        for arm in ("S_b1_1216", "G_b1_1216", "E_a0p1_1216", "LR_a0p001_1216"):
            st = D.setup_arm_dial(cfg, D._arm(cfg, arm), "cpu")
            rec, _ = exact_layer_record_elu(st, D.SIGMA_TOL)
            extra = D.unit_extra_record(st)
            self.assertTrue(torch.equal(extra["zmean"], rec["layers"][0]["zbar"]),
                            f"{arm}: zmean must be the recorded zbar")
            submerged = extra["zmax"] <= 0
            self.assertTrue(torch.equal(submerged,
                                        rec["layers"][0]["p_hat"] == 0),
                            f"{arm}: submerged == strict_dead is an identity")

    def test_mobility_is_p_hat_on_relu_and_leaky(self):
        cfg = copy.deepcopy(CFG)
        cfg["common"]["seeds"] = [0]
        from src.elu_swamp import exact_layer_record_elu
        for activation, dial in (("relu", 0.0), ("leaky", 0.05)):
            arm_cfg = copy.deepcopy(D._arm(cfg, "LR_a0p01_1216"))
            arm_cfg["activation"], arm_cfg["dial"] = activation, dial
            st = D.setup_arm_dial(cfg, arm_cfg, "cpu")
            rec, _ = exact_layer_record_elu(st, D.SIGMA_TOL)
            p_hat = rec["layers"][0]["p_hat"]
            want = p_hat if activation == "relu" else dial + (1 - dial) * p_hat
            got = D.unit_extra_record(st)["mob"]
            self.assertLess(float((got - want).abs().max()), 1e-12)

    def test_logger_is_trajectory_neutral_in_a_short_run(self):
        cfg = copy.deepcopy(CFG)
        cfg["common"]["seeds"] = [0]
        from src.gate_dose import train_arm_gate
        from src.mlp2_phase1 import _init_hashes
        states = []
        with tempfile.TemporaryDirectory() as tmp:
            for record_units in (True, False):
                st = D.setup_arm_dial(cfg, D._arm(cfg, "G_b1_1216"), "cpu")
                rec = D.DialRecorder([0, 1000, 2000], st,
                                     record_units=record_units)
                train_arm_gate(st, rec, [0, 1000, 2000], 2000, Path(tmp), [])
                states.append(_init_hashes(st))
        self.assertEqual(states[0], states[1])


class ControlTests(unittest.TestCase):
    def test_controls_are_transcribed_from_the_committed_verdict(self):
        controls = D._load_controls(CFG)
        self.assertEqual(sorted(controls), sorted(D.CONTROL_ORDER))
        self.assertEqual(controls["R_1216"]["n_onset_5m"], 10)
        self.assertEqual(controls["LR_1216"]["n_onset_5m"], 0)
        self.assertEqual(controls["E_1216"]["n_onset_5m"], 0)
        for arm in D.CONTROL_ORDER:
            self.assertEqual(len(controls[arm]["u_5m"]), 10)
            self.assertTrue(str(controls[arm]["source"]).endswith("verdict.csv"))

    def test_control_expectations_are_enforced(self):
        cfg = copy.deepcopy(CFG)
        cfg["gate_dial"]["control_expected_onset_5m"]["LR_1216"] = 10
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(D.SanityError):
                D.analyze(cfg, Path(tmp), [], "all", {}, {}, {})

    @unittest.skipUnless((PARENT / "ckpts" / "E_1216_step5000000.pt").exists(),
                         "parent checkpoints are gitignored")
    def test_m_minus_from_checkpoint_matches_the_algebraic_value(self):
        """ReLU の m⁻ は 0、leaky は a。ELU だけが独立な量になる。"""
        for arm, want in (("R_1216", 0.0), ("LR_1216", 0.1)):
            geo = D._geometry(CFG, arm)
            got = D._m_minus_from_checkpoint(CFG, arm, 0, geo)
            self.assertEqual(got["status"], "OK", arm)
            self.assertAlmostEqual(got["m_minus"], want, places=6, msg=arm)
            self.assertEqual(got["window"], "final_step5000000")
        elu = D._m_minus_from_checkpoint(CFG, "E_1216", 0, D._geometry(CFG, "E_1216"))
        self.assertEqual(elu["status"], "OK")
        self.assertGreater(elu["m_minus"], 0.0)
        self.assertLess(elu["m_minus"], 0.1)


def _fabricate_log(path: Path, arm: str, seed: int, *, unfit_level: float,
                   width: int = 100, rng: np.random.Generator) -> None:
    """集計経路の結合テスト用の合成ログ（記録点はタスク終端のみの粗い版）。"""
    step = np.arange(0, 5_000_001, 10_000, dtype=np.int64)
    n = len(step)
    zbar = rng.normal(-3.0, 1.0, size=(n, width)).astype(np.float32)
    p_hat = np.where(zbar < -1.0, 0.0, 0.5).astype(np.float32)
    payload = dict(
        step=step, run_id=np.array(f"{arm}_s{seed}"), arm=np.array(arm),
        seed=np.int64(seed), activation=np.array("silu"),
        act_alpha=np.float64(1.0), family=np.array("silu"),
        task_period=np.int64(10_000), target_mu_norm=np.float64(3.041),
        target_dose=np.float64(12.16),
        state_hash_final=np.array("{}"), state_hash_1m=np.array("{}"),
        signal_var=np.full(n, 1.0), residual_var=np.full(n, unfit_level),
        unfit=np.full(n, unfit_level), eval_loss_exact=np.full(n, unfit_level),
        flip_state=rng.integers(0, 2, size=(n, 15)).astype(np.float32),
        gamma=np.full(n, 0.5), gamma_negative=np.zeros(n),
        mu_norm_formula=np.full(n, 3.041), dose_formula=np.full(n, 12.16),
        mu_cos_off=np.full(n, 1.0), dose_relative_error=np.zeros(n),
        layer1_M=rng.normal(-2.0, 0.2, size=(n, width)).astype(np.float32),
        layer1_B=rng.normal(-1.0, 0.2, size=(n, width)).astype(np.float32),
        layer1_denom=np.full((n, width), 1.5, dtype=np.float32),
        layer1_p_hat=p_hat, layer1_w_norm=np.full((n, width), 2.0, dtype=np.float32),
        layer1_zbar=zbar, layer1_dzbar=np.zeros((n, width), dtype=np.float32),
        layer1_median_M=np.full(n, -2.0), layer1_q25_M=np.full(n, -2.2),
        layer1_q75_M=np.full(n, -1.8), layer1_median_B=np.full(n, -1.0),
        layer1_n_na=np.zeros(n), layer1_mu_norm=np.full(n, 3.041),
        layer1_sigma_rms=np.full(n, 0.25), layer1_dose=np.full(n, 12.16),
        layer1_w_norm_median=np.full(n, 2.0), layer1_w_norm_q25=np.full(n, 1.9),
        layer1_w_norm_q75=np.full(n, 2.1), layer1_eff_rank=np.full(n, 5.0),
        layer1_eff_rank_W=np.full(n, 5.0),
        layer1_strict_dead=(p_hat == 0).sum(axis=1).astype(np.int64),
        layer1_alive=(p_hat > 0).sum(axis=1).astype(np.int64),
        layer1_eff_rank_per_alive=np.full(n, 0.1),
        layer1_submerged=(p_hat == 0).sum(axis=1).astype(np.int64),
        layer1_preact_sd_median=np.full(n, 1.0),
        layer1_wcos_mean=np.full(n, 0.3), layer1_stable_rank_W=np.full(n, 3.0),
        layer1_top1_frac=np.full(n, 0.2), layer1_sign_match_frac=np.full(n, 0.4),
        layer1_mob=np.where(p_hat > 0, 0.5, 1e-3).astype(np.float32),
        layer1_absmob=np.where(p_hat > 0, 0.5, 1e-3).astype(np.float32),
        layer1_zmax=(zbar + 1.5).astype(np.float32),
        layer1_zmean=zbar,
        layer1_v_unit=rng.normal(0.0, 1.0, size=(n, width)).astype(np.float32))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


class AnalyzeIntegrationTests(unittest.TestCase):
    """合成ログで集計経路を通す（本走 5M を待たずに CSV の形まで見る）。"""

    ARMS = ["S_b1_1216", "G_b1_1216"]

    def test_analyze_writes_every_registered_output(self):
        cfg = copy.deepcopy(CFG)
        rng = np.random.default_rng(7)
        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp)
            for arm in self.ARMS:
                for seed in range(10):
                    _fabricate_log(outdir / "logs" / f"{arm}_seed{seed}.npz",
                                   arm, seed, unfit_level=1e-3, rng=rng)
            result = D.analyze(cfg, outdir, self.ARMS, "1", {}, {}, {})
            # 両腕とも 0/10 -> V1 は軟らかい側
            self.assertEqual(result["V1"], "SOFT_GATES_SOFT_SIDE")
            self.assertEqual(result["onset"]["5M"],
                             {"S_b1_1216": 0, "G_b1_1216": 0})
            for name in ("verdict.csv", "summary.md", "layer_stats.csv",
                         "dial_table.csv", "dial_spearman.csv",
                         "onset_times.csv", "onset_km.csv", "depth_hist.csv",
                         "s_distribution.csv", "revival.csv"):
                self.assertTrue((outdir / name).exists(), name)
            with (outdir / "verdict.csv").open(newline="") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(len(rows), len(self.ARMS) + len(D.CONTROL_ORDER))
            controls = [r for r in rows if r["is_control"] == "1"]
            self.assertEqual(len(controls), 3)
            for row in controls:
                self.assertEqual(row["status"], "COMMITTED_OTHER_RUN")
            # V3 の表は対照も含み、m⁻ の窓ラベルを持つ
            with (outdir / "dial_table.csv").open(newline="") as fh:
                dial = {r["arm"]: r for r in csv.DictReader(fh)}
            self.assertEqual(dial["S_b1_1216"]["m_minus_window"], "late_tasks_5m")
            if (PARENT / "ckpts" / "R_1216_step5000000.pt").exists():
                self.assertEqual(dial["R_1216"]["m_minus_window"],
                                 "final_step5000000")
            # S-mask は「10 点」で通り、字義の 100 との差を記録している
            self.assertEqual(result["s_mask"]["actual_records_per_window"], 10)
            self.assertEqual(result["s_mask"]["spec_literal_records_per_window"],
                             100)
            self.assertTrue(result["s_mask"]["pass_"])

    def test_onset_arms_flip_the_verdict(self):
        cfg = copy.deepcopy(CFG)
        rng = np.random.default_rng(8)
        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp)
            for arm in self.ARMS:
                for seed in range(10):
                    _fabricate_log(outdir / "logs" / f"{arm}_seed{seed}.npz",
                                   arm, seed, unfit_level=0.5, rng=rng)
            result = D.analyze(cfg, outdir, self.ARMS, "1", {}, {}, {})
            self.assertEqual(result["V1"], "SOFT_GATES_RELU_SIDE")
            # 早期窓も 0.05 を超えるので S-cap が立つ（登録どおりの振る舞い）
            self.assertEqual(sorted(result["capacity_undefined"]),
                             sorted(self.ARMS))

    def test_split_verdict(self):
        cfg = copy.deepcopy(CFG)
        rng = np.random.default_rng(9)
        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp)
            for seed in range(10):
                _fabricate_log(outdir / "logs" / f"S_b1_1216_seed{seed}.npz",
                               "S_b1_1216", seed, unfit_level=0.5, rng=rng)
                _fabricate_log(outdir / "logs" / f"G_b1_1216_seed{seed}.npz",
                               "G_b1_1216", seed, unfit_level=1e-3, rng=rng)
            result = D.analyze(cfg, outdir, self.ARMS, "1", {}, {}, {})
            self.assertEqual(result["V1"], "SPLIT_SILU_GELU")
            self.assertEqual(result["V1_developed"], ["S_b1_1216"])

    def test_divergent_v1_arm_makes_the_verdict_inconclusive(self):
        cfg = copy.deepcopy(CFG)
        rng = np.random.default_rng(10)
        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp)
            for seed in range(10):
                _fabricate_log(outdir / "logs" / f"G_b1_1216_seed{seed}.npz",
                               "G_b1_1216", seed, unfit_level=1e-3, rng=rng)
            divergences = {"S_b1_1216": dict(status="NUMERIC_DIVERGENCE",
                                             arm="S_b1_1216", detected_step=1)}
            result = D.analyze(cfg, outdir, self.ARMS, "1", {}, {},
                               divergences)
            self.assertEqual(result["V1"], "INCONCLUSIVE_DIVERGENCE")


if __name__ == "__main__":
    unittest.main()
