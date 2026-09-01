from __future__ import annotations

import copy
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.common import load_config
from src.envs import kaiming_mlp_params
from src.lin0_base_0902 import (
    ARM_ORDER,
    SMOKE_SEEDS,
    SMOKE_STEPS,
    SanityError,
    VecLinear0,
    _arm,
    classify_g_base,
    load_parent,
    preregistration_missing,
    run_arm,
    setup_lin0,
    sign_test,
    validate_config,
    window_values,
)
from src.train import make_gens
from src.width5_gate_0901 import setup_arm_width
from src.width5_gate_0901 import _arm as base_arm
from src.ratchet_log import full_support_ro, teacher_f64


class Lin0NetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = load_config("configs/lin0_base_0902.yaml")

    def test_net_is_exactly_affine_and_has_no_hidden_tensors(self) -> None:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(7)
        net = VecLinear0(4, 20, gen, "cpu")
        for name in ("Ws", "bs", "v", "W", "b", "hidden"):
            self.assertFalse(hasattr(net, name), name)
        x1 = torch.rand(4, 20)
        x2 = torch.rand(4, 20)
        lam = 0.25
        got = net.forward(lam * x1 + (1 - lam) * x2).double()
        want = lam * net.forward(x1).double() + (1 - lam) * net.forward(x2).double()
        self.assertLess(float((got - want).abs().max()), 1e-6)

    def test_init_matches_the_readout_rule_with_fan_in_d(self) -> None:
        d, R = 20, 8
        g1 = torch.Generator(device="cpu")
        g1.manual_seed(11)
        net = VecLinear0(R, d, g1, "cpu")
        self.assertAlmostEqual(net.bound, math.sqrt(3.0 / d))
        self.assertTrue(bool((net.a.abs() <= net.bound).all()))
        self.assertTrue(bool((net.c == 0).all()))
        # kaiming_mlp_params draws its readout with the same rule at h = d,
        # after consuming the input-layer block first.
        g2 = torch.Generator(device="cpu")
        g2.manual_seed(11)
        expected = ((torch.rand(R, d, generator=g2) * 2 - 1)
                    * math.sqrt(3.0 / d))
        self.assertTrue(torch.equal(net.a, expected))
        _, _, v, _ = kaiming_mlp_params(R, d, d, torch.Generator(), "cpu")
        self.assertEqual(v.shape, (R, d))

    def test_sgd_step_uses_the_factor_two_squared_error_convention(self) -> None:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(3)
        net = VecLinear0(2, 20, gen, "cpu")
        a0, c0 = net.a.clone(), net.c.clone()
        x = torch.rand(2, 20)
        delta = torch.tensor([0.5, -1.25])
        lr = torch.tensor([0.01, 0.01])
        net.sgd_step(lr, x, delta)
        self.assertTrue(torch.allclose(
            net.a, a0 - lr[:, None] * 2.0 * delta[:, None] * x))
        self.assertTrue(torch.allclose(net.c, c0 - lr * 2.0 * delta))


class Lin0SetupTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = load_config("configs/lin0_base_0902.yaml")
        self.seeds = [int(v) for v in self.cfg["common"]["seeds"]]

    def test_shares_the_parent_realization_despite_a_different_init_draw(self):
        """The whole paired comparison rests on this."""
        st = setup_lin0(self.cfg, _arm(self.cfg, "LIN0"), "cpu", self.seeds)
        parent_cfg = load_config("configs/width5_gate_b_0901.yaml")
        ref = setup_arm_width(parent_cfg, base_arm(parent_cfg, "LIN5"), "cpu")
        self.assertTrue(torch.equal(st["env"].flip_state,
                                    ref["env"].flip_state))
        Xa, Xb = full_support_ro(st["env"]), full_support_ro(ref["env"])
        self.assertTrue(torch.equal(Xa, Xb))
        self.assertTrue(torch.equal(teacher_f64(st["teacher"], Xa.double()),
                                    teacher_f64(ref["teacher"], Xb.double())))
        # ...and the init generator really did consume a different amount.
        gens = make_gens("A", 5, "cpu",
                         offset=int(self.cfg["common"]["generator_offset"]))
        self.assertEqual(st["net"].a.numel(), len(self.seeds) * 20)
        self.assertNotEqual(st["net"].a.numel(),
                            ref["net"].W.numel() + ref["net"].v.numel())
        del gens

    def test_both_arms_share_one_realization_and_differ_only_in_lr(self) -> None:
        a = setup_lin0(self.cfg, _arm(self.cfg, "LIN0"), "cpu", self.seeds)
        b = setup_lin0(self.cfg, _arm(self.cfg, "LIN0_lr03"), "cpu", self.seeds)
        self.assertTrue(torch.equal(a["net"].a, b["net"].a))
        self.assertTrue(torch.equal(a["env"].flip_state, b["env"].flip_state))
        self.assertAlmostEqual(float(a["lr"][0]), 0.01, places=6)
        self.assertAlmostEqual(float(b["lr"][0]), 0.03, places=6)


class Lin0ConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = load_config("configs/lin0_base_0902.yaml")

    def test_frozen_design_is_locked(self) -> None:
        validate_config(self.cfg, stage="full")
        for mutate in (
            lambda c: c["common"].__setitem__("generator_offset", 1),
            lambda c: c["arms"][0].__setitem__("lr", 0.02),
            lambda c: c["arms"][0].__setitem__("generator_width_basis", 100),
            lambda c: c["lin0"].__setitem__("baseline_arm", "LIN5"),
            lambda c: c["lin0"].__setitem__("tight_band", [0.1, 0.9]),
            lambda c: c["preregistration"].__setitem__(
                "prediction_B1", "BASELINE_CONSTRUCTION_MATERIAL"),
        ):
            broken = copy.deepcopy(self.cfg)
            mutate(broken)
            with self.assertRaises(ValueError):
                validate_config(broken, stage="full")

    def test_unfrozen_preregistration_blocks_result_bearing_stages(self) -> None:
        broken = copy.deepcopy(self.cfg)
        broken["preregistration"]["frozen"] = False
        self.assertIn("preregistration.frozen", preregistration_missing(broken))
        with self.assertRaises(ValueError):
            validate_config(broken, stage="full")
        validate_config(broken, stage="implementation")

    def test_parent_hash_mismatch_is_fatal(self) -> None:
        broken = copy.deepcopy(self.cfg)
        broken["parent"]["verdict_sha256"] = "0" * 64
        with self.assertRaises(SanityError):
            load_parent(broken)


class Lin0JudgementTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = load_config("configs/lin0_base_0902.yaml")

    def test_sign_test_reproduces_the_registered_labels(self) -> None:
        n = 20
        for k, suffix in ((20, "_ABOVE_LINEAR"), (0, "_BELOW_LINEAR"),
                          (10, "_NOT_SEPARATED_TIGHT"),
                          (13, "_INCONCLUSIVE_WIDE")):
            arm = [2.0] * k + [0.5] * (n - k)
            got = sign_test(self.cfg, arm, [1.0] * n, "R5", "5m")
            self.assertEqual(got["k"], k)
            self.assertTrue(got["status"].endswith(suffix))
            self.assertEqual(got["registered"], 1)

    def test_g_base_is_immaterial_only_when_all_six_labels_match(self) -> None:
        legacy = self.cfg["lin0"]["legacy_labels"]
        same = {(arm, w): dict(status=legacy[w][arm])
                for w in ("5m", "1m")
                for arm in self.cfg["lin0"]["comparison_arms"]}
        self.assertEqual(classify_g_base(self.cfg, same)["verdict"],
                         "BASELINE_CONSTRUCTION_IMMATERIAL")
        moved = dict(same)
        moved[("E5", "1m")] = dict(status="E5_NOT_SEPARATED_TIGHT")
        result = classify_g_base(self.cfg, moved)
        self.assertEqual(result["verdict"], "BASELINE_CONSTRUCTION_MATERIAL")
        self.assertEqual(result["changed"], ["E5@1m"])
        self.assertFalse(result["prediction_hit"])

    def test_window_values_take_the_registered_task_windows(self) -> None:
        period = int(self.cfg["phase1"]["task_period"])
        steps = np.arange(0, 5_000_001, 1000, dtype=np.int64)
        unfit = np.zeros((len(steps), 2), dtype=np.float64)
        late = ((steps > 0) & (steps % period == 0)
                & (steps // period >= 491) & (steps // period <= 500))
        unfit[late] = 4.0
        got = window_values(self.cfg, dict(step=steps, unfit=unfit), "5m")
        self.assertTrue(np.allclose(got, 4.0))
        self.assertEqual(int(late.sum()), 10)
        self.assertTrue(np.allclose(
            window_values(self.cfg, dict(step=steps, unfit=unfit), "1m"), 0.0))


class Lin0SmokeTests(unittest.TestCase):
    def test_two_arms_run_and_write_run_level_logs_only(self) -> None:
        cfg = load_config("configs/lin0_base_0902.yaml")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            for arm in ARM_ORDER:
                result = run_arm(cfg, arm, "cpu", out, SMOKE_SEEDS, SMOKE_STEPS)
                self.assertEqual(result["status"], "COMPLETE")
            with np.load(out / "logs" / "LIN0_seed0.npz",
                         allow_pickle=False) as z:
                self.assertEqual(int(z["width"]), 0)
                self.assertEqual(str(z["activation"]), "linear0")
                for key in ("unfit", "eval_loss_exact", "signal_var",
                            "residual_var"):
                    self.assertTrue(np.isfinite(z[key]).all())
                for key in z.files:
                    self.assertFalse(key.startswith("layer1_"), key)
            with np.load(out / "logs" / "LIN0_lr03_seed0.npz",
                         allow_pickle=False) as z:
                self.assertAlmostEqual(float(z["lr"]), 0.03, places=6)


if __name__ == "__main__":
    unittest.main()
