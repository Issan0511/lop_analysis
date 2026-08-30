"""elu_swamp_0830 の単体テスト。

重点は 3 つ:
  * ReLU 経路が phase1 と bit 一致すること（S0' の前提を CI 側でも押さえる）
  * ELU の勾配が深い沈下域まで正しいこと（本走の Q2 が測るのはそこ）
  * タスク内増分の選び方と増分の精度（spec §11.2 / §11.5）
"""
import copy
import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
import torch.nn.functional as F

from src import elu_swamp as E
from src.common import load_config
from src.mlp2_phase1 import forward_centered, grads_centered, setup_arm_p1
from src.nets import VecMLPL


def _cfg() -> dict:
    return load_config("configs/elu_swamp_0830.yaml")


class ClosedFormTests(unittest.TestCase):
    def test_g_elu_matches_the_spec_table(self):
        # spec §2.1: g(0.35)=0.08, g(1)=0.20, g(2)=0.33, g(4)=0.45, g(inf)=0.684
        for s, target in ((0.35, 0.08), (1.0, 0.20), (2.0, 0.33), (4.0, 0.45)):
            self.assertAlmostEqual(float(E.g_elu(s)), target, delta=0.005)
        relu_ratio = (1 / math.sqrt(2 * math.pi)) / math.sqrt(0.5 - 1 / (2 * math.pi))
        self.assertAlmostEqual(float(E.g_elu(1e4)), relu_ratio, delta=1e-3)
        self.assertAlmostEqual(relu_ratio, 0.6833, delta=1e-3)

    def test_g_elu_matches_monte_carlo(self):
        gen = torch.Generator().manual_seed(0)
        for s in (0.35, 1.0, 2.0):
            z = torch.randn(2_000_000, generator=gen, dtype=torch.float64) * s
            a = torch.where(z > 0, z, torch.expm1(z))
            mc = float(a.mean() / a.std(unbiased=False))
            self.assertAlmostEqual(mc, float(E.g_elu(s)), delta=0.005)

    def test_g_elu_survives_large_s_without_overflow(self):
        # exp(2 s^2) alone overflows float64 past s ~ 20; erfcx does not.
        values = E.g_elu(torch.tensor([1e-3, 1.0, 25.0, 1e3, 1e6]))
        self.assertTrue(torch.isfinite(values).all())
        self.assertTrue(bool((values[1:] > values[:-1]).all()))

    def test_dose_reference_is_sqrt_width_times_g(self):
        self.assertAlmostEqual(float(E.dose_reference(1.0, 100)),
                               10.0 * float(E.g_elu(1.0)), places=12)


class ActivationTests(unittest.TestCase):
    def _net(self, act, alpha=1.0, form=None):
        gen = torch.Generator().manual_seed(7)
        return VecMLPL(2, [3, 2], 4, gen, "cpu").set_activation(act, alpha, form)

    def test_forward_and_backward_match_torch_elu(self):
        net = self._net("elu")
        z = torch.linspace(-40, 40, 20001, dtype=torch.float64)
        self.assertTrue(torch.equal(net.act_fn(z), F.elu(z, 1.0)))
        ref = z.clone().requires_grad_(True)
        F.elu(ref, 1.0).sum().backward()
        got = net.act_grad(z, net.act_fn(z))
        rel = (got - ref.grad).abs() / ref.grad.abs().clamp_min(1e-300)
        self.assertLess(float(rel.max()), 1e-6)

    def test_registered_derivative_survives_float32_deep_tail(self):
        """本走は float32 で学習する。深い沈下域で勾配が 0 に落ちてはいけない。"""
        net = self._net("elu", form="alpha_exp")
        z = torch.tensor([-10.0, -16.0, -20.0, -30.0], dtype=torch.float32)
        got = net.act_grad(z, net.act_fn(z)).double()
        true = torch.tensor([math.exp(float(v)) for v in z], dtype=torch.float64)
        self.assertLess(float(((got - true).abs() / true).max()), 1e-6)
        self.assertTrue(bool((got > 0).all()))

    def test_rejected_derivative_form_zeroes_the_float32_tail(self):
        """§11.9 の根拠を回帰テストとして固定する（元の書き方に戻さないため）。"""
        net = self._net("elu", form="activation_plus_alpha")
        z = torch.tensor([-10.0, -16.0, -20.0, -30.0], dtype=torch.float32)
        got = net.act_grad(z, net.act_fn(z)).double()
        true = torch.tensor([math.exp(float(v)) for v in z], dtype=torch.float64)
        rel = ((got - true).abs() / true)
        self.assertGreater(float(rel[0]), 1e-4)      # z=-10 で既に 4e-4 級
        self.assertEqual(float(got[-1]), 0.0)        # z=-30 で厳密に 0

    def test_relu_path_is_untouched(self):
        net = self._net("relu")
        z = torch.linspace(-5, 5, 1001, dtype=torch.float32)
        self.assertTrue(torch.equal(net.act_fn(z), torch.relu(z)))
        self.assertTrue(torch.equal(net.act_grad(z, net.act_fn(z)),
                                    (z > 0).float()))

    def test_alpha_zero_collapses_onto_relu(self):
        elu, relu = self._net("elu", 0.0), self._net("relu")
        z = torch.linspace(-30, 30, 4001, dtype=torch.float64)
        self.assertTrue(torch.equal(elu.act_fn(z), relu.act_fn(z)))
        self.assertTrue(torch.equal(elu.act_grad(z, elu.act_fn(z)),
                                    relu.act_grad(z, relu.act_fn(z))))

    def test_set_activation_does_not_touch_the_init_tensors(self):
        a = VecMLPL(2, [3, 2], 4, torch.Generator().manual_seed(11), "cpu")
        b = VecMLPL(2, [3, 2], 4, torch.Generator().manual_seed(11), "cpu",
                    act="elu")
        for key in a.state_dict():
            self.assertTrue(torch.equal(a.state_dict()[key], b.state_dict()[key]))
        gen_a, gen_b = torch.Generator().manual_seed(3), torch.Generator().manual_seed(3)
        VecMLPL(2, [3, 2], 4, gen_a, "cpu")
        VecMLPL(2, [3, 2], 4, gen_b, "cpu", act="elu")
        self.assertEqual(gen_a.get_state().tolist(), gen_b.get_state().tolist())


class ReluPathIdentityTests(unittest.TestCase):
    """S0' の前提: ReLU 腕では phase1 と演算列が一致する。"""

    def setUp(self):
        self.cfg = _cfg()
        self.cfg["common"]["seeds"] = [0, 1]

    def test_forward_and_grads_match_phase1_bit_for_bit(self):
        for arm in ("R_none", "R_A1"):
            st_new = E.setup_arm_elu(self.cfg, E._arm(self.cfg, arm), "cpu")
            st_old = setup_arm_p1(E._p1_cfg(self.cfg), E._arm(self.cfg, arm), "cpu")
            for _ in range(50):
                x = st_new["env"].step()
                st_old["env"].step()
                y = st_new["teacher"](x)
                new = E.forward_centered_elu(st_new, x)
                old = forward_centered(st_old, x)
                for a, b in zip(new, old):
                    if isinstance(a, list):
                        self.assertTrue(all(torch.equal(p, q) for p, q in zip(a, b)))
                    else:
                        self.assertTrue(torch.equal(a, b))
                gn = E.grads_centered_elu(st_new["net"], new[0], new[1], new[2],
                                          new[3] - y)
                go = grads_centered(st_old["net"], old[0], old[1], old[2], old[3] - y)
                for a, b in zip(gn, go):
                    if isinstance(a, list):
                        self.assertTrue(all(torch.equal(p, q) for p, q in zip(a, b)))
                    else:
                        self.assertTrue(torch.equal(a, b))
                st_new["net"].sgd_step_layers(st_new["lr"], *gn)
                st_old["net"].sgd_step_layers(st_old["lr"], *go)

    def test_submergence_matches_strict_dead_on_relu(self):
        st = E.setup_arm_elu(self.cfg, E._arm(self.cfg, "R_none"), "cpu")
        E.train_arm_elu(st, lambda *_: None, [], 3000, Path("."), [])
        rec, sanity = E.exact_layer_record_elu(st, 1e-8)
        for layer, s in zip(rec["layers"], sanity["layers"]):
            self.assertTrue(torch.equal(layer["submerged"], layer["strict_dead"]))
            self.assertEqual(s["submerge_mismatch"], 0)


class S0PrimeSchemaTests(unittest.TestCase):
    """S0' は「既存列の完全一致」。新規列だけを許し、欠落は必ず捕まえること。"""

    REFERENCE = Path("results/mlp2_phase1_0829/logs/L2_A1_seed0.npz")

    def setUp(self):
        if not self.REFERENCE.exists():
            self.skipTest("phase1 reference logs are not present")

    def test_self_comparison_is_clean(self):
        with TemporaryDirectory() as d:
            twin = Path(d) / "twin.npz"
            twin.write_bytes(self.REFERENCE.read_bytes())
            self.assertEqual(E._compare_arm_logs(twin, self.REFERENCE), [])

    def test_a_missing_legacy_column_is_reported(self):
        with TemporaryDirectory() as d:
            with np.load(self.REFERENCE, allow_pickle=False) as a:
                payload = {k: a[k] for k in a.files if k != "layer2_strict_dead"}
            broken = Path(d) / "broken.npz"
            np.savez_compressed(broken, **payload)
            self.assertEqual(
                E._compare_arm_logs(broken, self.REFERENCE),
                [dict(column="layer2_strict_dead", reason="missing in elu_swamp")])

    def test_every_new_column_is_excluded_from_the_comparison(self):
        """新規列を 1 つでも除外し忘れると S0' が偽陽性で落ちる。"""
        with TemporaryDirectory() as d:
            with np.load(self.REFERENCE, allow_pickle=False) as a:
                payload = {k: a[k] for k in a.files}
            payload["activation"] = np.array("relu")
            payload["act_alpha"] = np.float64(1.0)
            payload["v_readout"] = np.zeros((3, 100), dtype=np.float32)
            payload["v_readout_step"] = np.arange(3, dtype=np.int64)
            for li in (1, 2):
                payload[f"layer{li}_zbar"] = np.zeros((3, 100), dtype=np.float32)
                payload[f"layer{li}_dzbar"] = np.zeros((3, 100), dtype=np.float32)
                payload[f"layer{li}_submerged"] = np.zeros(3, dtype=np.int64)
                payload[f"layer{li}_preact_sd_median"] = np.zeros(3)
            extended = Path(d) / "extended.npz"
            np.savez_compressed(extended, **payload)
            self.assertEqual(E._compare_arm_logs(extended, self.REFERENCE), [])


class IncrementSelectionTests(unittest.TestCase):
    """spec §11.2 / §11.5: タスク内 9 区間、境界を跨がない、始点で条件付ける。"""

    def _write(self, tmp: Path, zbar: np.ndarray, dzbar: np.ndarray,
               p_hat: np.ndarray, steps: np.ndarray) -> None:
        (tmp / "logs").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(tmp / "logs" / "E_A1_seed0.npz", step=steps,
                            layer2_zbar=zbar.astype(np.float32),
                            layer2_dzbar=dzbar.astype(np.float32),
                            layer2_p_hat=p_hat.astype(np.float32))

    def test_nine_within_task_intervals_and_no_boundary_crossing(self):
        cfg = _cfg()
        n_tasks = 6
        steps = np.arange(0, n_tasks * 10_000 + 1, 1000, dtype=np.int64)
        n, h = len(steps), 4
        # 記録点ごとに一意な値を入れて、どの区間が選ばれたかを復元できるようにする。
        zbar = -np.ones((n, h)) * np.arange(n)[:, None]
        dzbar = np.arange(n)[:, None] * np.ones((1, h))
        p_hat = np.zeros((n, h))          # 全ユニットが沈下している
        with TemporaryDirectory() as d:
            tmp = Path(d)
            self._write(tmp, zbar, dzbar, p_hat, steps)
            cfg["elu_swamp"]["q2_window_tasks"] = [2, 5]
            cfg["elu_swamp"]["q2_bins"] = 1
            cfg["elu_swamp"]["q2_bin_min_count"] = 1
            out = E._interval_rows(cfg, tmp / "logs", "E_A1", 0, 2)
        # 窓はタスク 2..5 の 4 タスク -> 9*4 = 36 区間
        self.assertEqual(out["late"]["n_intervals"], 36)
        self.assertEqual(out["all"]["n_intervals"], 9 * n_tasks)
        self.assertEqual(out["late"]["n_unit_intervals"], 36 * h)

    def test_boundary_intervals_are_excluded(self):
        cfg = _cfg()
        steps = np.arange(0, 30_001, 1000, dtype=np.int64)
        n, h = len(steps), 1
        # 境界を跨ぐ区間 (start step が 10000/20000) だけ巨大な増分にしておく。
        dzbar = np.zeros((n, h))
        for i, s in enumerate(steps[:-1]):
            if s % 10_000 == 0:
                dzbar[i + 1] = 1e6
        with TemporaryDirectory() as d:
            tmp = Path(d)
            self._write(tmp, -np.ones((n, h)), dzbar, np.zeros((n, h)), steps)
            cfg["elu_swamp"]["q2_window_tasks"] = [1, 3]
            cfg["elu_swamp"]["q2_bins"] = 1
            cfg["elu_swamp"]["q2_bin_min_count"] = 1
            out = E._interval_rows(cfg, tmp / "logs", "E_A1", 0, 2)
        self.assertEqual(out["late"]["n_intervals"], 27)
        self.assertEqual(out["late"]["bins"][0]["dzbar_sd"], 0.0)   # 1e6 は入っていない

    def test_only_units_submerged_at_the_start_are_kept(self):
        cfg = _cfg()
        steps = np.arange(0, 20_001, 1000, dtype=np.int64)
        n, h = len(steps), 3
        p_hat = np.zeros((n, h))
        p_hat[:, 1] = 0.5              # 常に生きている
        p_hat[-1, 2] = 0.0             # 終点でだけ沈下 -> 始点条件では拾わない
        p_hat[:-1, 2] = 0.25
        with TemporaryDirectory() as d:
            tmp = Path(d)
            self._write(tmp, -np.ones((n, h)), np.ones((n, h)), p_hat, steps)
            cfg["elu_swamp"]["q2_window_tasks"] = [1, 2]
            cfg["elu_swamp"]["q2_bins"] = 1
            cfg["elu_swamp"]["q2_bin_min_count"] = 1
            out = E._interval_rows(cfg, tmp / "logs", "E_A1", 0, 2)
        self.assertEqual(out["late"]["n_unit_intervals"], 18)   # 18 区間 x 1 ユニット

    def test_mobility_slope_recovers_a_known_exponent(self):
        """sd[dzbar] = exp(zbar) を仕込むと自然対数回帰の傾きは 1 になる。"""
        cfg = _cfg()
        steps = np.arange(0, 200_001, 1000, dtype=np.int64)
        n, h = len(steps), 120
        rng = np.random.default_rng(0)
        depth = np.linspace(-12.0, -0.5, h)
        zbar = np.repeat(depth[None], n, axis=0)
        dzbar = rng.standard_normal((n, h)) * np.exp(depth)[None]
        with TemporaryDirectory() as d:
            tmp = Path(d)
            self._write(tmp, zbar, dzbar, np.zeros((n, h)), steps)
            cfg["elu_swamp"]["q2_window_tasks"] = [1, 20]
            out = E._interval_rows(cfg, tmp / "logs", "E_A1", 0, 2)
        self.assertAlmostEqual(out["late"]["beta"], 1.0, delta=0.1)
        self.assertLess(abs(out["late"]["rho"]), 0.1)           # 駆動ゼロ
        self.assertEqual(E._p2a_label(cfg, dict(point=out["late"]["beta"],
                                                percentile_ci_lo=0.9,
                                                percentile_ci_hi=1.1,
                                                ci_degenerate=1), "percentile"),
                         "MOBILITY_SCALING")


class IncrementPrecisionTests(unittest.TestCase):
    """dzbar は float64 の zbar 差から作る（float32 の zbar を引くと沈む）。"""

    def test_dzbar_is_the_float64_difference_not_the_float32_one(self):
        cfg = _cfg()
        cfg["common"]["seeds"] = [0]
        st = E.setup_arm_elu(cfg, E._arm(cfg, "E_A1"), "cpu")
        exact: dict[int, np.ndarray] = {}

        def capture(state, step):
            rec_, _ = E.exact_layer_record_elu(state, 1e-8)
            exact[int(step)] = rec_["layers"][1]["zbar"].numpy().copy()

        steps = [0, 1000, 2000]
        rec = E.EluRecorder(steps, st, 1e-8, 1e-10, 1000,
                            zbar_layers=[2], readout_steps=[])

        def both(state, step):
            capture(state, step)
            rec(state, step)

        E.train_arm_elu(st, both, steps, 2000, Path("."), [])
        dzbar = rec.layers[1]["dzbar"]
        self.assertTrue(np.isnan(dzbar[0]).all())      # 先頭は未定義
        self.assertTrue(np.isfinite(dzbar[1:]).all())
        for i, step in enumerate(steps[1:], start=1):
            want = (exact[step] - exact[steps[i - 1]]).astype(np.float32)
            np.testing.assert_array_equal(dzbar[i], want)

    def test_float32_zbar_differences_would_lose_the_deep_tail(self):
        """§11.5 の設計理由。float32 の zbar を引くと 1e-9 級の増分が消える。"""
        z0 = np.float64(-20.0)
        z1 = z0 + 2.0e-9
        self.assertNotEqual(np.float32(z1 - z0), np.float32(0.0))
        self.assertEqual(np.float32(z1) - np.float32(z0), np.float32(0.0))


class ConfigGateTests(unittest.TestCase):
    def test_full_stage_is_blocked_until_g0b_is_confirmed(self):
        """§10-2 のゲート。Issa は 2026-08-30 に 0.30 のままで確認済み。"""
        cfg = _cfg()
        self.assertIs(cfg["elu_swamp"]["g0b_threshold_confirmed"], True)
        for stage in ("preflight", "smoke", "s0prime", "full", "analyze"):
            E.validate_config(cfg, stage=stage)
        unconfirmed = copy.deepcopy(cfg)
        unconfirmed["elu_swamp"]["g0b_threshold_confirmed"] = False
        for stage in ("preflight", "smoke", "s0prime"):
            E.validate_config(unconfirmed, stage=stage)
        for stage in ("full", "analyze"):
            with self.assertRaises(ValueError) as ctx:
                E.validate_config(unconfirmed, stage=stage)
            self.assertIn("10-2", str(ctx.exception))

    def test_frozen_thresholds_cannot_drift(self):
        for key, bad in (("q1_level_regenerated_above", 9.0),
                         ("g0b_submerged_threshold", 0.5),
                         ("q2_drift_ratio_noise_dominated", 0.2),
                         ("gate_unfit_threshold", 0.1)):
            cfg = _cfg()
            cfg["elu_swamp"][key] = bad
            with self.assertRaises(ValueError):
                E.validate_config(cfg, stage="preflight")

    def test_the_unstable_derivative_form_is_rejected(self):
        cfg = _cfg()
        cfg["activation"]["elu"]["derivative_form"] = "activation_plus_alpha"
        with self.assertRaises(ValueError):
            E.validate_config(cfg, stage="preflight")

    def test_labels_use_only_the_frozen_thresholds(self):
        cfg = _cfg()
        self.assertEqual(E._q1_level_label(cfg, 1.99), "MU_SUPPRESSED")
        self.assertEqual(E._q1_level_label(cfg, 2.00), "MU_INTERMEDIATE")
        self.assertEqual(E._q1_level_label(cfg, 9.32), "MU_INTERMEDIATE")
        self.assertEqual(E._q1_level_label(cfg, 9.33), "MU_REGENERATED")
        self.assertEqual(E._p2b_label(cfg, -0.30), "DRIFT_DOMINATED_DOWNWARD")
        self.assertEqual(E._p2b_label(cfg, -0.20), "MIXED")
        self.assertEqual(E._p2b_label(cfg, 0.09), "NOISE_DOMINATED")
        self.assertEqual(E._p2b_label(cfg, 0.30), "DRIFT_DOMINATED_UPWARD")


if __name__ == "__main__":
    unittest.main()
