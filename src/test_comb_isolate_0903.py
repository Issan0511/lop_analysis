"""comb_isolate_0903 / comb_mlp2_0903 の単体テスト（spec `specs/spec_comb_isolate_0903.md`）。

    OMP_NUM_THREADS=1 python3 -m unittest src.test_comb_isolate_0903 -v

pytest ではなく unittest なのは、この checkout に .venv も pytest も無いため（spec §10 追補 8）。
"""
from __future__ import annotations

import copy
import math
import unittest
from pathlib import Path

import numpy as np
import torch

from src import comb_isolate_0903 as A
from src.common import ROOT, load_config
from src.nets import VecMLPL

CFG = load_config(str(A.CONFIG))
GRID = torch.linspace(-30, 30, 24001, dtype=torch.float64)
NEW = ("comb1_flat", "comb1_leaky", "band_leaky_dpi", "snake")
LOBE = math.pi


def _net(act: str, alpha: float) -> VecMLPL:
    net = VecMLPL(1, [2], 2, torch.Generator().manual_seed(0), "cpu")
    return net.set_activation(act, alpha, "alpha_exp")


def _alpha(act: str) -> float:
    return 0.1 if act == "band_leaky_dpi" else 1.0


class ActivationTests(unittest.TestCase):
    def test_names_are_registered(self):
        for act in NEW:
            self.assertIn(act, VecMLPL.ACTIVATIONS)

    def test_no_silent_elu_fall_through(self):
        elu = _net("elu", 1.0)
        for act in NEW:
            with self.subTest(act=act):
                n = _net(act, _alpha(act))
                f = n.act_fn(GRID)
                self.assertFalse(torch.allclose(f, elu.act_fn(GRID)))
                self.assertFalse(torch.allclose(n.act_grad(GRID, f),
                                                elu.act_grad(GRID, elu.act_fn(GRID))))

    def test_finite_difference_matches_the_closed_form(self):
        h, tol = 1e-6, 1e-6
        for act in NEW:
            with self.subTest(act=act):
                alpha = _alpha(act)
                n = _net(act, alpha)
                z = GRID.clone()
                mask = torch.ones_like(z, dtype=torch.bool)
                for k in A._kinks(act, alpha):
                    mask &= (z - k).abs() > 1e-3
                z = z[mask]
                fd = (n.act_fn(z + h) - n.act_fn(z - h)) / (2 * h)
                self.assertLess(float((fd - n.act_grad(z, n.act_fn(z))).abs().max()),
                                tol)

    def test_comb1_matches_comb_binf_on_the_open_lobe(self):
        """spec §10 追補 1: 一致は **開区間** (-pi, inf)。端点は float64 で sin(-pi)≠0。"""
        cb = _net("comb_binf", 1.0)
        z = torch.linspace(-LOBE + 1e-9, 30.0, 20001, dtype=torch.float64)
        for act in ("comb1_flat", "comb1_leaky"):
            with self.subTest(act=act):
                n = _net(act, 1.0)
                self.assertTrue(torch.equal(n.act_fn(z), cb.act_fn(z)))
                self.assertTrue(torch.equal(n.act_grad(z, n.act_fn(z)),
                                            cb.act_grad(z, cb.act_fn(z))))

    def test_the_lobe_endpoint_is_where_they_differ(self):
        t = torch.tensor([-LOBE], dtype=torch.float64)
        cb = _net("comb_binf", 1.0)
        self.assertNotEqual(float(cb.act_fn(t)), 0.0)          # -1.50e-32
        for act in ("comb1_flat", "comb1_leaky"):
            self.assertEqual(float(_net(act, 1.0).act_fn(t)), 0.0)

    def test_comb1_flat_is_exactly_zero_beyond_the_lobe(self):
        n = _net("comb1_flat", 1.0)
        z = torch.linspace(-30.0, -LOBE - 1e-9, 5001, dtype=torch.float64)
        self.assertTrue(torch.equal(n.act_fn(z), torch.zeros_like(z)))
        self.assertTrue(torch.equal(n.act_grad(z, n.act_fn(z)), torch.zeros_like(z)))

    def test_comb1_leaky_has_the_registered_return_path(self):
        a = VecMLPL.COMB1_LEAK["comb1_leaky"]
        self.assertEqual(a, 0.1)
        n = _net("comb1_leaky", 1.0)
        z = torch.tensor([-LOBE - 1.0, -LOBE - 5.0], dtype=torch.float64)
        self.assertTrue(torch.allclose(n.act_fn(z), a * (z + LOBE)))
        self.assertTrue(torch.allclose(n.act_grad(z, n.act_fn(z)),
                                       torch.full_like(z, a)))

    def test_the_well_is_a_double_zero_only_for_the_flat_variant(self):
        """spec §2: CB1f は phi=phi'=0、CB1l は phi=0 で phi'=a（戻り道）。"""
        t = torch.tensor([-LOBE], dtype=torch.float64)
        f, l = _net("comb1_flat", 1.0), _net("comb1_leaky", 1.0)
        self.assertAlmostEqual(float(f.act_fn(t)), 0.0, places=15)
        self.assertAlmostEqual(float(f.act_grad(t, f.act_fn(t))), 0.0, places=15)
        self.assertAlmostEqual(float(l.act_fn(t)), 0.0, places=15)
        self.assertAlmostEqual(float(l.act_grad(t, l.act_fn(t))), 0.1, places=15)

    def test_band_leaky_dpi_needs_no_new_branch(self):
        """spec §10 追補 3: BAND_WIDTH への 1 エントリだけで forward/backward が通る。"""
        self.assertAlmostEqual(VecMLPL.BAND_WIDTH["band_leaky_dpi"], math.pi, places=15)
        n = _net("band_leaky_dpi", 0.1)
        inside = torch.linspace(-LOBE + 1e-9, -1e-9, 2001, dtype=torch.float64)
        self.assertTrue(torch.equal(n.act_fn(inside), torch.zeros_like(inside)))
        beyond = torch.tensor([-4.0], dtype=torch.float64)
        self.assertAlmostEqual(float(n.act_fn(beyond)), 0.1 * (-4.0 + math.pi),
                               places=12)
        self.assertAlmostEqual(float(n.act_grad(beyond, n.act_fn(beyond))), 0.1,
                               places=15)

    def test_snake_is_monotone_and_has_no_gate(self):
        n = _net("snake", 1.0)
        g = n.act_grad(GRID, n.act_fn(GRID))
        self.assertGreaterEqual(float(g.min()), 0.0)
        self.assertLessEqual(float(g.max()), 2.0 + 1e-12)
        # 正側も恒等ではない（ゲートを持たない）
        pos = torch.linspace(0.5, 5.0, 101, dtype=torch.float64)
        self.assertFalse(torch.allclose(n.act_fn(pos), pos))

    def test_snake_identity_limit_needs_the_registered_alpha(self):
        """spec §10 追補 2: |z|<=30 で相対 1e-4 を満たすには alpha <= 3e-6。"""
        tol = float(CFG["sanity"]["s_limit_snake_rel_tol"])
        alpha = float(CFG["sanity"]["s_limit_snake_alpha"])
        self.assertLessEqual(alpha, 3e-6)

        def rel(a):
            n = _net("snake", a)
            return float(((n.act_fn(GRID) - GRID).abs()
                          / torch.clamp(GRID.abs(), min=1.0)).max())
        self.assertLessEqual(rel(alpha), tol)
        self.assertGreater(rel(1e-3), tol)     # vault の「極小」では足りない

    def test_existing_activation_paths_are_untouched(self):
        self.assertTrue(torch.equal(_net("relu", 1.0).act_fn(GRID), torch.relu(GRID)))
        self.assertTrue(torch.equal(_net("leaky_relu", 0.1).act_fn(GRID),
                                    torch.where(GRID > 0, GRID, 0.1 * GRID)))
        cb = _net("comb_binf", 1.0)
        self.assertTrue(torch.equal(
            cb.act_fn(GRID), torch.where(GRID > 0, GRID, 0.0 - torch.sin(GRID) ** 2)))

    def test_activation_choice_consumes_no_randomness(self):
        for act in NEW:
            with self.subTest(act=act):
                ga, gb = (torch.Generator().manual_seed(11),
                          torch.Generator().manual_seed(11))
                a = VecMLPL(3, [5], 4, ga, "cpu")
                b = VecMLPL(3, [5], 4, gb, "cpu").set_activation(act, _alpha(act),
                                                                 "alpha_exp")
                for key in ("W", "b", "v", "c"):
                    self.assertTrue(torch.equal(a.state_dict()[key],
                                                b.state_dict()[key]))
                self.assertEqual(float(torch.rand(1, generator=ga)),
                                 float(torch.rand(1, generator=gb)))

    def test_guard_tuples(self):
        for act in ("comb1_flat", "comb1_leaky", "snake"):
            self.assertIn(act, VecMLPL.WEIRD_FREQ_ACTIVATIONS)
            _net(act, 2.0)                       # 振動数は 1 を超えてよい
            with self.assertRaises(ValueError):
                _net(act, 0.0)
        self.assertIn("band_leaky_dpi", VecMLPL.WEIRD_SLOPE_ACTIVATIONS)
        with self.assertRaises(ValueError):
            _net("band_leaky_dpi", 1.5)


class ConfigTests(unittest.TestCase):
    def test_stage_a_config_validates(self):
        A.validate_config(copy.deepcopy(CFG), stage="run")

    def test_stage_b_config_validates_against_its_own_module(self):
        path = Path(ROOT) / "configs" / "comb_mlp2_0903.yaml"
        cfg = load_config(str(path))
        self.assertEqual(cfg["arms"][0]["name"], "CB_A1")
        self.assertEqual([int(v) for v in cfg["comb_mlp2"]["late_tasks"]], [451, 500])
        self.assertEqual(float(cfg["comb_mlp2"]["unfit_floor"]), 1e-23)
        self.assertEqual(int(cfg["comb_mlp2"]["bootstrap_seed"]), 20260915)

    def test_both_stages_share_the_bootstrap_seed_and_the_spec(self):
        b = load_config(str(Path(ROOT) / "configs" / "comb_mlp2_0903.yaml"))
        self.assertEqual(int(CFG["phase1"]["bootstrap_seed"]),
                         int(b["comb_mlp2"]["bootstrap_seed"]))
        self.assertEqual(CFG["spec"], b["spec"])

    def test_prediction_provenance_is_recorded_in_both_configs(self):
        """§7.1 は §7.2 と同一。独立の予言ではないことを config が持つ。"""
        b = load_config(str(Path(ROOT) / "configs" / "comb_mlp2_0903.yaml"))
        self.assertEqual(b["preregistration"]["prediction_provenance"],
                         "draft_values_proposed_first_then_approved_by_Issa")
        self.assertIn("独立の予言ではない",
                      b["preregistration"]["prediction_provenance_note"])

    def test_exact_fit_guard_is_registered(self):
        E = CFG["comb_isolate"]["exact_fit"]
        self.assertEqual(float(E["threshold"]), 1e-8)
        self.assertTrue(E["blocks_level_labels"])
        self.assertTrue(E["keeps_onset_labels"])

    def test_stage_c_is_not_registered(self):
        self.assertFalse(CFG["staging"]["stageC_registered"])

    def test_mutations_are_rejected(self):
        mutations = {
            "arm order": lambda c: c["arms"].reverse(),
            "dial": lambda c: c["arms"][0].__setitem__("dial", 2.0),
            "bootstrap seed": lambda c: c["phase1"].__setitem__("bootstrap_seed", 1),
            "guard threshold": lambda c: c["comb_isolate"]["exact_fit"].__setitem__(
                "threshold", 1e-6),
            "margin": lambda c: c["comb_isolate"].__setitem__(
                "p5_equivalence_margin", 0.3),
            "generator offset": lambda c: c["common"].__setitem__("generator_offset", 7),
            "stage C": lambda c: c["staging"].__setitem__("stageC_registered", True),
            "snake alpha": lambda c: c["sanity"].__setitem__("s_limit_snake_alpha", 1e-3),
        }
        for name, mutate in mutations.items():
            with self.subTest(mutation=name):
                c = copy.deepcopy(CFG)
                mutate(c)
                with self.assertRaises(ValueError):
                    A.validate_config(c, stage="run")


class SanityUnitTests(unittest.TestCase):
    def test_s_copy_s_const_s_dial_s_fd_s_num_pass(self):
        self.assertTrue(A._s_copy()["pass_"])
        for fn in (A._s_const, A._s_dial, A._s_fd, A._s_num):
            with self.subTest(check=fn.__name__):
                got = fn(copy.deepcopy(CFG))
                self.assertTrue(got["pass_"], got.get("failures"))

    def test_s_guard_separates_the_comb_from_every_control(self):
        got = A._s_guard(copy.deepcopy(CFG))
        self.assertTrue(got["pass_"], got["failures"])
        self.assertGreater(got["separation_decades"], 4.0)

    def test_s_dial_records_the_registered_design_difference(self):
        rows = {r["arm"]: r for r in A._s_dial(copy.deepcopy(CFG))["rows"]}
        self.assertTrue(rows["CB1f_a1_1216"]["wells"][0]["is_double_zero"])
        self.assertFalse(rows["CB1l_a1_1216"]["wells"][0]["is_double_zero"])

    def test_exact_fit_helper(self):
        fired = A._exact_fit(CFG, np.full(10, 1e-13))
        quiet = A._exact_fit(CFG, np.full(10, 1e-3))
        self.assertTrue(fired["fired"])
        self.assertEqual(fired["label"], "EXACT_FIT")
        self.assertFalse(quiet["fired"])
        self.assertEqual(quiet["label"], "")


if __name__ == "__main__":
    unittest.main()
