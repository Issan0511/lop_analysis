"""weird_act_0903 の単体テスト（spec `specs/spec_weird_act_0903.md`）。

    OMP_NUM_THREADS=1 python3 -m unittest src.test_weird_act_0903 -v

pytest ではなく unittest なのは、この checkout に .venv も pytest も無いため
（spec §10 追補 7）。様式は test_gate_dial_0902 / test_bwd_leak_0902 に倣う。
"""
from __future__ import annotations

import copy
import math
import unittest
from pathlib import Path

import torch

from src import weird_act_0903 as W
from src.common import ROOT, load_config
from src.nets import VecMLPL

CFG_PATH = Path(ROOT) / "configs" / "weird_act_0903.yaml"
CFG = load_config(str(CFG_PATH))
GRID = torch.linspace(-30, 30, 24001, dtype=torch.float64)
NEW_ACTS = ("mirror_leaky", "fold_leaky_d1", "fold_leaky_d2", "fold_leaky_dbig",
            "band_leaky_d0", "band_leaky_d0p5", "band_leaky_d1", "band_leaky_d2",
            "band_leaky_d4", "ramp_leaky_d1", "comb_binf", "comb_b5")


def _net(act: str, alpha: float) -> VecMLPL:
    net = VecMLPL(1, [2], 2, torch.Generator().manual_seed(0), "cpu")
    return net.set_activation(act, alpha, "alpha_exp")


def _alpha_for(act: str) -> float:
    return 1.0 if act.startswith("comb") else 0.1


class ActivationTests(unittest.TestCase):
    def test_all_ten_registered_names_plus_two_limits_are_in_ACTIVATIONS(self):
        for act in NEW_ACTS:
            self.assertIn(act, VecMLPL.ACTIVATIONS)

    def test_no_new_activation_silently_falls_through_to_elu(self):
        """act_fn / act_grad の連鎖にはガードが無く、分岐を忘れると黙って ELU になる。"""
        elu = _net("elu", 1.0)
        for act in NEW_ACTS:
            with self.subTest(act=act):
                n = _net(act, _alpha_for(act))
                f = n.act_fn(GRID)
                g = n.act_grad(GRID, f)
                self.assertFalse(torch.allclose(f, elu.act_fn(GRID)))
                self.assertFalse(torch.allclose(
                    g, elu.act_grad(GRID, elu.act_fn(GRID))))

    def test_finite_difference_matches_the_closed_form_backward(self):
        """5 族はすべて自分の forward の真の導関数（折れ目 ±1e-3 は除外）。"""
        h, tol = 1e-6, 1e-6
        for act in NEW_ACTS:
            with self.subTest(act=act):
                alpha = _alpha_for(act)
                n = _net(act, alpha)
                z = GRID.clone()
                mask = torch.ones_like(z, dtype=torch.bool)
                for k in W._kinks(act):
                    mask &= (z - k).abs() > 1e-3
                z = z[mask]
                fd = (n.act_fn(z + h) - n.act_fn(z - h)) / (2 * h)
                g = n.act_grad(z, n.act_fn(z))
                err = (fd - g).abs()
                if act == "comb_b5":     # 包絡が e^4 まで伸びるので相対で見る
                    err = err / torch.clamp(g.abs(), min=1.0)
                self.assertLess(float(err.max()), tol)

    def test_band_leaky_d0_is_bit_identical_to_leaky(self):
        a, b = _net("band_leaky_d0", 0.1), _net("leaky_relu", 0.1)
        self.assertTrue(torch.equal(a.act_fn(GRID), b.act_fn(GRID)))
        self.assertTrue(torch.equal(a.act_grad(GRID, a.act_fn(GRID)),
                                    b.act_grad(GRID, b.act_fn(GRID))))

    def test_fold_leaky_dbig_is_bit_identical_to_leaky(self):
        a, b = _net("fold_leaky_dbig", 0.1), _net("leaky_relu", 0.1)
        self.assertTrue(torch.equal(a.act_fn(GRID), b.act_fn(GRID)))
        self.assertTrue(torch.equal(a.act_grad(GRID, a.act_fn(GRID)),
                                    b.act_grad(GRID, b.act_fn(GRID))))

    def test_mirror_leaky_at_zero_slope_is_bit_identical_to_relu(self):
        a, b = _net("mirror_leaky", 0.0), _net("relu", 1.0)
        fa, fb = a.act_fn(GRID), b.act_fn(GRID)
        ga, gb = a.act_grad(GRID, fa), b.act_grad(GRID, fb)
        self.assertTrue(torch.equal(fa, fb))
        self.assertTrue(torch.equal(ga, gb))
        # torch.equal は符号盲なので、-0.0 の混入を signbit で見る（追補 9）
        self.assertEqual(int(torch.signbit(fa).sum()), int(torch.signbit(fb).sum()))
        self.assertEqual(int(torch.signbit(ga).sum()), int(torch.signbit(gb).sum()))

    def test_mirror_leaky_has_the_same_phi_squared_as_leaky_but_opposite_phi_prime(self):
        """V1 が割る点そのもの: phi^2 は同じで phi' の符号だけ逆。"""
        m, lk = _net("mirror_leaky", 0.1), _net("leaky_relu", 0.1)
        self.assertTrue(torch.allclose(m.act_fn(GRID) ** 2, lk.act_fn(GRID) ** 2))
        neg = GRID < 0
        gm = m.act_grad(GRID, m.act_fn(GRID))[neg]
        gl = lk.act_grad(GRID, lk.act_fn(GRID))[neg]
        self.assertTrue(torch.allclose(gm, -gl))

    def test_fold_leaky_zeros_are_at_zero_and_minus_two_d(self):
        for act, d in (("fold_leaky_d1", 1.0), ("fold_leaky_d2", 2.0)):
            with self.subTest(act=act):
                n = _net(act, 0.1)
                z = torch.tensor([-2.0 * d], dtype=torch.float64)
                self.assertAlmostEqual(float(n.act_fn(z)), 0.0, places=12)
                self.assertLess(float(n.act_fn(torch.tensor([-d],
                                                            dtype=torch.float64))), 0.0)

    def test_band_leaky_forward_is_exactly_zero_inside_the_dead_band(self):
        for act, d in VecMLPL.BAND_WIDTH.items():
            if d == 0.0:
                continue
            with self.subTest(act=act):
                n = _net(act, 0.1)
                z = torch.linspace(-d + 1e-9, -1e-9, 101, dtype=torch.float64)
                self.assertTrue(torch.equal(n.act_fn(z), torch.zeros_like(z)))
                self.assertTrue(torch.equal(n.act_grad(z, n.act_fn(z)),
                                            torch.zeros_like(z)))

    def test_ramp_leaky_is_c1_at_the_join_and_has_a_zero_derivative_wall(self):
        d, a = VecMLPL.RAMP_DEPTH["ramp_leaky_d1"], 0.1
        n = _net("ramp_leaky_d1", a)
        left = torch.tensor([-d - 1e-9], dtype=torch.float64)
        right = torch.tensor([-d + 1e-9], dtype=torch.float64)
        self.assertAlmostEqual(float(n.act_fn(left)), float(n.act_fn(right)), places=9)
        self.assertAlmostEqual(float(n.act_grad(left, n.act_fn(left))),
                               float(n.act_grad(right, n.act_fn(right))), places=7)
        wall = torch.tensor([-1e-12], dtype=torch.float64)
        self.assertAlmostEqual(float(n.act_grad(wall, n.act_fn(wall))), 0.0, places=12)
        self.assertFalse(bool(torch.signbit(n.act_grad(wall, n.act_fn(wall)))))

    def test_comb_has_double_zeros_at_k_pi_over_alpha(self):
        for act, alpha in (("comb_binf", 1.0), ("comb_binf", 2.0), ("comb_b5", 1.0)):
            with self.subTest(act=act, alpha=alpha):
                n = _net(act, alpha)
                z = torch.tensor([-k * math.pi / alpha for k in (1, 2, 3)],
                                 dtype=torch.float64)
                self.assertLess(float(n.act_fn(z).abs().max()), 1e-25)
                self.assertLess(float(n.act_grad(z, n.act_fn(z)).abs().max()), 1e-12)

    def test_comb_envelope_only_deepens_the_deep_side(self):
        inf_net, b5 = _net("comb_binf", 1.0), _net("comb_b5", 1.0)
        z = torch.linspace(-9.0, -0.1, 2001, dtype=torch.float64)
        self.assertTrue(bool((b5.act_fn(z).abs() >= inf_net.act_fn(z).abs() - 1e-15).all()))
        pos = torch.linspace(0.1, 9.0, 101, dtype=torch.float64)
        self.assertTrue(torch.equal(b5.act_fn(pos), pos))
        self.assertTrue(torch.equal(inf_net.act_fn(pos), pos))

    def test_existing_activation_paths_are_untouched(self):
        r = _net("relu", 1.0)
        self.assertTrue(torch.equal(r.act_fn(GRID), torch.relu(GRID)))
        lk = _net("leaky_relu", 0.1)
        self.assertTrue(torch.equal(lk.act_fn(GRID),
                                    torch.where(GRID > 0, GRID, 0.1 * GRID)))
        e = _net("elu", 1.0)
        self.assertTrue(torch.equal(
            e.act_fn(GRID), torch.where(GRID > 0, GRID, torch.expm1(GRID))))
        s = _net("silu", 1.0)
        self.assertTrue(torch.equal(s.act_fn(GRID), GRID * torch.sigmoid(GRID)))
        bl = _net("bwd_leaky", 0.1)
        self.assertTrue(torch.equal(bl.act_fn(GRID), torch.relu(GRID)))

    def test_activation_choice_consumes_no_randomness(self):
        for act in NEW_ACTS:
            with self.subTest(act=act):
                ga = torch.Generator().manual_seed(11)
                gb = torch.Generator().manual_seed(11)
                a = VecMLPL(3, [5], 4, ga, "cpu")
                b = VecMLPL(3, [5], 4, gb, "cpu").set_activation(
                    act, _alpha_for(act), "alpha_exp")
                for key in ("W", "b", "v", "c"):
                    self.assertTrue(torch.equal(a.state_dict()[key],
                                                b.state_dict()[key]))
                self.assertEqual(float(torch.rand(1, generator=ga)),
                                 float(torch.rand(1, generator=gb)))

    def test_set_activation_range_guards(self):
        for act in VecMLPL.WEIRD_SLOPE_ACTIVATIONS:
            with self.subTest(act=act):
                with self.assertRaises(ValueError):
                    _net(act, 1.5)
                with self.assertRaises(ValueError):
                    _net(act, -0.1)
        for act in VecMLPL.WEIRD_FREQ_ACTIVATIONS:
            with self.subTest(act=act):
                _net(act, 2.0)             # 櫛は alpha=2 を取れる（CB_a2）
                with self.assertRaises(ValueError):
                    _net(act, 0.0)
                with self.assertRaises(ValueError):
                    _net(act, float("inf"))

    def test_second_parameter_tables_cover_every_registered_name(self):
        for act in VecMLPL.FOLD_DEPTH:
            self.assertIn(act, VecMLPL.ACTIVATIONS)
        for act in VecMLPL.BAND_WIDTH:
            self.assertIn(act, VecMLPL.ACTIVATIONS)
        self.assertEqual(VecMLPL.COMB_ENVELOPE["comb_binf"], float("inf"))
        self.assertEqual(VecMLPL.COMB_ENVELOPE["comb_b5"], 5.0)


class ConfigTests(unittest.TestCase):
    def test_shipped_config_validates(self):
        W.validate_config(copy.deepcopy(CFG), stage="run")

    def test_stage_split_is_five_plus_six_with_LRv_d1_in_stage_one(self):
        self.assertEqual(len(W.STAGE_ARMS[1]), 5)
        self.assertEqual(len(W.STAGE_ARMS[2]), 6)
        self.assertIn("LRv_d1_1216", W.STAGE_ARMS[1])
        self.assertNotIn(3, W.STAGE_ARMS)

    def test_v3_label_is_only_emitted_for_LRv_d2(self):
        G = CFG["weird_act"]["v3"]
        self.assertEqual(G["arm"], "LRv_d2_1216")
        self.assertEqual(G["anchor_arm"], "LRv_d1_1216")
        self.assertTrue(G["anchor_is_report_only"])
        self.assertTrue(G["do_not_fold_v3_and_v3_prime"])

    def test_v2_requires_stage_two(self):
        self.assertEqual(int(CFG["weird_act"]["v2"]["requires_stage"]), 2)
        self.assertTrue(CFG["staging"]["v2_requires_stage2"])

    def test_bootstrap_seed_is_the_registered_unused_date(self):
        self.assertEqual(int(CFG["phase1"]["bootstrap_seed"]), 20260914)

    def test_mutations_are_rejected(self):
        mutations = {
            "arm order": lambda c: c["arms"].reverse(),
            "dial": lambda c: c["arms"][0].__setitem__("dial", 0.2),
            "stage": lambda c: c["arms"][4].__setitem__("stage", 3),
            "bootstrap seed": lambda c: c["phase1"].__setitem__("bootstrap_seed", 1),
            "margin": lambda c: c["weird_act"].__setitem__("p5_equivalence_margin", 0.3),
            "generator offset": lambda c: c["common"].__setitem__("generator_offset", 7),
            "u_star": lambda c: c["arms"][0].__setitem__("u_star", 1.0),
            "v2 stage": lambda c: c["weird_act"]["v2"].__setitem__("requires_stage", 1),
            "anchor": lambda c: c["weird_act"]["v3"].__setitem__(
                "anchor_is_report_only", False),
        }
        for name, mutate in mutations.items():
            with self.subTest(mutation=name):
                c = copy.deepcopy(CFG)
                mutate(c)
                with self.assertRaises(ValueError):
                    W.validate_config(c, stage="run")


class SanityUnitTests(unittest.TestCase):
    def test_s_copy_shows_only_the_recorder_line(self):
        got = W._s_copy()
        self.assertTrue(got["pass_"], got)
        self.assertEqual(len(got["differences"]), 1)

    def test_s_const_matches_config_and_nets_tables(self):
        self.assertTrue(W._s_const(copy.deepcopy(CFG))["pass_"])

    def test_s_dial_matches_the_registered_geometry(self):
        got = W._s_dial(copy.deepcopy(CFG))
        self.assertTrue(got["pass_"], got["failures"])

    def test_s_fd_and_s_num_pass_on_the_registered_ranges(self):
        self.assertTrue(W._s_fd(copy.deepcopy(CFG))["pass_"])
        self.assertTrue(W._s_num(copy.deepcopy(CFG))["pass_"])

    def test_weird_unit_keys_add_exactly_zmin(self):
        from src.gate_dial_0902 import NEW_UNIT_KEYS
        self.assertEqual(set(W.WEIRD_UNIT_KEYS) - set(NEW_UNIT_KEYS), {"zmin"})
        self.assertEqual(set(NEW_UNIT_KEYS) - set(W.WEIRD_UNIT_KEYS), set())


if __name__ == "__main__":
    unittest.main()
