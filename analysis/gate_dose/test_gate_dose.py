"""Regression tests for gate_dose_0830."""
import copy
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
import torch.nn.functional as F

from src import gate_dose as G
from src.common import load_config
from src.dose_const_5m import forward_const, setup_arm_const
from src.mlp2_phase1 import grads_centered
from src.nets import VecMLPL


def _cfg() -> dict:
    return load_config("configs/gate_dose_0830.yaml")


class ActivationTests(unittest.TestCase):
    def test_leaky_forward_and_backward_match_torch(self):
        net = VecMLPL(2, [3], 4, torch.Generator().manual_seed(7), "cpu")
        net.set_activation("leaky_relu", 0.1)
        z = torch.linspace(-40, 40, 20001, dtype=torch.float64)
        self.assertTrue(torch.equal(net.act_fn(z), F.leaky_relu(z, 0.1)))
        ref = z.clone().requires_grad_(True)
        F.leaky_relu(ref, 0.1).sum().backward()
        self.assertTrue(torch.equal(net.act_grad(z, net.act_fn(z)), ref.grad))

    def test_leaky_zero_collapses_onto_relu(self):
        gen1, gen2 = torch.Generator().manual_seed(11), torch.Generator().manual_seed(11)
        relu = VecMLPL(2, [3], 4, gen1, "cpu")
        leaky = VecMLPL(2, [3], 4, gen2, "cpu").set_activation("leaky_relu", 0.0)
        z = torch.linspace(-30, 30, 4001, dtype=torch.float64)
        self.assertTrue(torch.equal(relu.act_fn(z), leaky.act_fn(z)))
        self.assertTrue(torch.equal(relu.act_grad(z, relu.act_fn(z)),
                                    leaky.act_grad(z, leaky.act_fn(z))))

    def test_relu_path_is_still_literal_relu(self):
        net = VecMLPL(1, [3], 4, torch.Generator().manual_seed(3), "cpu")
        z = torch.linspace(-5, 5, 1001)
        self.assertTrue(torch.equal(net.act_fn(z), torch.relu(z)))
        self.assertTrue(torch.equal(net.act_grad(z, net.act_fn(z)), (z > 0).float()))


class FrozenReluPathTests(unittest.TestCase):
    def setUp(self):
        self.cfg = _cfg()
        self.cfg["common"]["seeds"] = [0, 1]

    def test_gate_relu_matches_fixed_dose_bit_for_bit(self):
        for arm in G.RELU_ARMS:
            gate_cfg = G._arm(self.cfg, arm)
            reference_cfg = copy.deepcopy(gate_cfg)
            new = G.setup_arm_gate(self.cfg, gate_cfg, "cpu")
            old = setup_arm_const(self.cfg, reference_cfg, "cpu")
            for _ in range(50):
                x_new = new["env"].step()
                x_old = old["env"].step()
                self.assertTrue(torch.equal(x_new, x_old))
                y_new, y_old = new["teacher"](x_new), old["teacher"](x_old)
                self.assertTrue(torch.equal(y_new, y_old))
                f_new, f_old = G.forward_gate(new, x_new), forward_const(old, x_old)
                for a, b in zip(f_new, f_old):
                    if isinstance(a, list):
                        self.assertTrue(all(torch.equal(x, y) for x, y in zip(a, b)))
                    else:
                        self.assertTrue(torch.equal(a, b))
                g_new = G.grads_centered_elu(new["net"], f_new[0], f_new[1], f_new[2],
                                             f_new[3] - y_new)
                g_old = grads_centered(old["net"], f_old[0], f_old[1], f_old[2],
                                       f_old[3] - y_old)
                for a, b in zip(g_new, g_old):
                    if isinstance(a, list):
                        self.assertTrue(all(torch.equal(x, y) for x, y in zip(a, b)))
                    else:
                        self.assertTrue(torch.equal(a, b))
                new["net"].sgd_step_layers(new["lr"], *g_new)
                old["net"].sgd_step_layers(old["lr"], *g_old)


class VerdictTests(unittest.TestCase):
    def test_all_registered_branches(self):
        self.assertEqual(G._gate_verdict(
            {"E_933": 0, "E_1216": 0, "LR_933": 0, "LR_1216": 0}),
            "GATE_LOAD_BEARING")
        self.assertEqual(G._gate_verdict(
            {"E_933": 5, "E_1216": 0, "LR_933": 0, "LR_1216": 0}),
            "SOFT_WALL_AT_HIGH_DOSE")
        self.assertEqual(G._gate_verdict(
            {"E_933": 5, "E_1216": 0, "LR_933": 5, "LR_1216": 0}),
            "DOSE_NOT_GATE_MEDIATED")
        self.assertEqual(G._gate_verdict(
            {"E_933": 0, "E_1216": 0, "LR_933": 5, "LR_1216": 0}),
            "SMOOTHNESS_REQUIRED")
        self.assertEqual(G._gate_verdict(
            {"E_933": 1, "E_1216": 0, "LR_933": 0, "LR_1216": 0}),
            "PARTIAL")
        self.assertEqual(G._gate_verdict(
            {"E_933": 0, "E_1216": 0, "LR_933": 0, "LR_1216": 0},
            {"E_1216"}), "INCONCLUSIVE_DIVERGENCE")


class S0SchemaTests(unittest.TestCase):
    reference = Path("results/dose_const_5m_0830/logs/dose933_seed0.npz")

    def setUp(self):
        if not self.reference.exists():
            self.skipTest("fixed-dose reference log is unavailable")

    def test_new_columns_are_ignored_but_old_columns_are_required(self):
        with TemporaryDirectory() as d:
            with np.load(self.reference, allow_pickle=False) as z:
                payload = {k: z[k] for k in z.files}
            payload["activation"] = np.array("relu")
            payload["act_alpha"] = np.float64(1.0)
            n = len(payload["step"])
            payload["layer1_zbar"] = np.zeros((n, 100), dtype=np.float32)
            payload["layer1_dzbar"] = np.zeros((n, 100), dtype=np.float32)
            payload["layer1_submerged"] = np.zeros(n, dtype=np.int64)
            payload["layer1_preact_sd_median"] = np.zeros(n)
            extended = Path(d) / "extended.npz"
            np.savez_compressed(extended, **payload)
            self.assertEqual(G._compare_reference_log(extended, self.reference), [])
            del payload["layer1_strict_dead"]
            broken = Path(d) / "broken.npz"
            np.savez_compressed(broken, **payload)
            self.assertEqual(G._compare_reference_log(broken, self.reference),
                             [dict(column="layer1_strict_dead",
                                   reason="missing in gate_dose")])


class IncrementTests(unittest.TestCase):
    def test_within_task_start_conditioning_and_minimum_units(self):
        cfg = _cfg()
        cfg["gate_dose"]["q2_window_tasks"] = [2, 5]
        cfg["gate_dose"]["q2_bins"] = 2
        cfg["gate_dose"]["q2_bin_min_count"] = 1
        cfg["gate_dose"]["q2_min_submerged_units_per_seed"] = 1
        steps = np.arange(0, 60_001, 1000, dtype=np.int64)
        n, h = len(steps), 4
        zbar = -np.arange(n, dtype=np.float32)[:, None] * np.ones((1, h), np.float32)
        dzbar = np.arange(n, dtype=np.float32)[:, None] * np.ones((1, h), np.float32)
        p_hat = np.zeros((n, h), dtype=np.float32)
        with TemporaryDirectory() as d:
            out = Path(d)
            (out / "logs").mkdir()
            np.savez_compressed(out / "logs/E_933_seed0.npz", step=steps,
                                layer1_zbar=zbar, layer1_dzbar=dzbar,
                                layer1_p_hat=p_hat)
            got = G._interval_rows(cfg, out, "E_933", 0)
        self.assertEqual(got["n_intervals"], 36)
        self.assertEqual(got["n_unit_intervals"], 36 * h)
        self.assertEqual(got["status"], "OK")


class ConfigAndGradientTests(unittest.TestCase):
    def test_registered_config(self):
        cfg = _cfg()
        for stage in ("preflight", "smoke", "s0prime", "full", "analyze"):
            G.validate_config(cfg, stage=stage)

    def test_registered_closed_form_gradients(self):
        result = G._s_grad_check(_cfg())
        self.assertTrue(result["pass_"], result["failures"])


class AnalysisIntegrationTests(unittest.TestCase):
    sources = {
        "off": Path("results/dose_const_5m_0830/logs/dose_off_seed0.npz"),
        "933": Path("results/dose_const_5m_0830/logs/dose933_seed0.npz"),
        "1216": Path("results/dose_const_5m_0830/logs/dose1216_seed0.npz"),
    }

    def setUp(self):
        if not all(path.exists() for path in self.sources.values()):
            self.skipTest("fixed-dose full logs are unavailable")

    def test_full_analysis_schema_on_frozen_log_fixtures(self):
        cfg = _cfg()
        source_arm = {arm: ("off" if arm.endswith("off") else
                            "1216" if arm.endswith("1216") else "933")
                      for arm in G.ARM_ORDER}
        with TemporaryDirectory() as d:
            out = Path(d)
            (out / "logs").mkdir()
            for arm in G.ARM_ORDER:
                source = source_arm[arm]
                for seed in range(10):
                    path = Path(f"results/dose_const_5m_0830/logs/"
                                f"dose{'_' if source == 'off' else ''}{source}_seed{seed}.npz")
                    with np.load(path, allow_pickle=False) as z:
                        payload = {k: z[k] for k in z.files}
                    payload["arm"] = np.array(arm)
                    payload["activation"] = np.array(G.REGISTERED_ARMS[arm][0])
                    payload["act_alpha"] = np.float64(1.0)
                    payload["layer1_zbar"] = payload["layer1_M"].copy()
                    payload["layer1_dzbar"] = np.zeros_like(payload["layer1_M"])
                    payload["layer1_submerged"] = payload["layer1_strict_dead"].copy()
                    payload["layer1_preact_sd_median"] = np.median(
                        payload["layer1_denom"], axis=1)
                    np.savez_compressed(out / "logs" / f"{arm}_seed{seed}.npz",
                                        **payload)
            sanity = {key: {"pass_": True} for key in
                      ("S0prime", "S_pair", "S_pair_final", "S_dose",
                       "S_dose_final", "S_grad", "S_elu_limit", "S_leaky_limit",
                       "S_submerge", "S_tautology", "S6_floor_inherited")}
            result = G.analyze(cfg, out, sanity, {}, {})
            self.assertEqual(result["main_verdict"], "DOSE_NOT_GATE_MEDIATED")
            for name in ("verdict.csv", "summary.md", "dose_response.csv",
                         "increments.csv"):
                self.assertTrue((out / name).exists(), name)


if __name__ == "__main__":
    unittest.main()
