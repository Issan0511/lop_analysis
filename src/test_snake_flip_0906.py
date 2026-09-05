"""snake_flip_0906 の検査（spec `specs/spec_snake_flip_0906.md` §5）。

    OMP_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m unittest src.test_snake_flip_0906 -v

S-fd / S-limit / S-fallthrough / S-guard / S-cfg。様式は test_edge_law_nets_0905 に倣う。
"""
from __future__ import annotations

import math
import unittest
from pathlib import Path

import torch

from src.common import ROOT, load_config
from src.nets import VecMLPL

GRID = torch.linspace(-30, 30, 24001, dtype=torch.float64)
NEW = ("snake1", "snake_amp0p25", "snake_amp0p5", "snake_amp1")
ALPHAS = (0.5, 1.0, 3.0)
CFG_NEW = Path(ROOT) / "configs" / "snake_flip_0906.yaml"
CFG_EDGE = Path(ROOT) / "configs" / "edge_law_0905.yaml"


def _net(act: str, alpha: float) -> VecMLPL:
    net = VecMLPL(1, [2], 2, torch.Generator().manual_seed(0), "cpu")
    return net.set_activation(act, alpha, "alpha_exp")


def _lobe(alpha: float) -> tuple[float, float]:
    return -3.0 * math.pi / (4.0 * alpha), math.pi / (4.0 * alpha)


def _bytes(t: torch.Tensor) -> bytes:
    return t.contiguous().numpy().tobytes()


class GuardTests(unittest.TestCase):
    def test_new_names_are_registered_in_both_tuples(self):
        for act in NEW:
            self.assertIn(act, VecMLPL.ACTIVATIONS)
            self.assertIn(act, VecMLPL.WEIRD_FREQ_ACTIVATIONS)

    def test_frequency_guard_rejects_nonpositive_alpha(self):
        for act in NEW:
            with self.assertRaises(ValueError):
                _net(act, 0.0)
            with self.assertRaises(ValueError):
                _net(act, -1.0)

    def test_amp_table_matches_config(self):
        cfg = load_config(str(CFG_NEW))
        for name, A in VecMLPL.SNAKE_AMP.items():
            self.assertEqual(float(cfg["activation"][name].get("amp", 1.0)), A, name)


class FallthroughTests(unittest.TestCase):
    """act_fn/act_grad の if 連鎖は最後が ELU。分岐の書き忘れは黙って ELU になる。"""

    def test_none_of_the_three_directions_falls_through_to_elu(self):
        z = GRID
        for alpha in ALPHAS:
            elu = _net("elu", alpha)
            fe, ge = elu.act_fn(z), elu.act_grad(z, elu.act_fn(z))
            ce = elu.act_curv(z)
            for act in NEW:
                with self.subTest(act=act, alpha=alpha):
                    n = _net(act, alpha)
                    f = n.act_fn(z)
                    self.assertFalse(torch.allclose(f, fe))
                    self.assertFalse(torch.allclose(n.act_grad(z, f), ge))
                    self.assertFalse(torch.allclose(n.act_curv(z), ce))

    def test_snake_curvature_is_now_registered(self):
        n = _net("snake", 1.0)
        c = n.act_curv(GRID)
        self.assertTrue(torch.allclose(c, 2.0 * torch.cos(2.0 * GRID)))


class LimitTests(unittest.TestCase):
    def test_snake_amp1_is_byte_identical_to_snake(self):
        for alpha in ALPHAS:
            a, b = _net("snake_amp1", alpha), _net("snake", alpha)
            fa, fb = a.act_fn(GRID), b.act_fn(GRID)
            self.assertEqual(_bytes(fa), _bytes(fb), alpha)
            self.assertEqual(_bytes(a.act_grad(GRID, fa)), _bytes(b.act_grad(GRID, fb)))
            self.assertEqual(_bytes(a.act_curv(GRID)), _bytes(b.act_curv(GRID)))

    def test_snake_amp_half_is_NOT_snake(self):
        """S-limit の生存証明: A=0.5 では同じ比較が落ちること。"""
        a, b = _net("snake_amp0p5", 1.0), _net("snake", 1.0)
        self.assertNotEqual(_bytes(a.act_fn(GRID)), _bytes(b.act_fn(GRID)))

    def test_snake1_is_byte_identical_to_snake_inside_the_lobe(self):
        for alpha in ALPHAS:
            lo, hi = _lobe(alpha)
            z = GRID[(GRID > lo) & (GRID < hi)]
            self.assertGreater(z.numel(), 100)
            a, b = _net("snake1", alpha), _net("snake", alpha)
            fa, fb = a.act_fn(z), b.act_fn(z)
            self.assertEqual(_bytes(fa), _bytes(fb), alpha)
            self.assertEqual(_bytes(a.act_grad(z, fa)), _bytes(b.act_grad(z, fb)))
            self.assertEqual(_bytes(a.act_curv(z)), _bytes(b.act_curv(z)))

    def test_snake1_differs_from_snake_outside_the_lobe(self):
        lo, hi = _lobe(1.0)
        z = GRID[(GRID < lo - 0.5) | (GRID > hi + 0.5)]
        a, b = _net("snake1", 1.0), _net("snake", 1.0)
        self.assertGreater(float((a.act_fn(z) - b.act_fn(z)).abs().max()), 0.1)
        self.assertTrue(torch.all(a.act_grad(z, a.act_fn(z)) == 1.0))
        self.assertTrue(torch.all(a.act_curv(z) == 0.0))

    def test_snake1_is_continuous_at_both_cuts_and_grad_jumps_two_to_one(self):
        for alpha in ALPHAS:
            n = _net("snake1", alpha)
            for cut in _lobe(alpha):
                z = torch.tensor([cut - 1e-9, cut, cut + 1e-9], dtype=torch.float64)
                f = n.act_fn(z)
                self.assertLess(float((f[0] - f[2]).abs()), 1e-7, (alpha, cut))
                g = n.act_grad(z, f)
                inside = g[1] if cut < 0 else g[1]
                self.assertAlmostEqual(float(inside), 2.0, places=6)
                outside = g[0] if cut < 0 else g[2]
                self.assertEqual(float(outside), 1.0)


class FiniteDifferenceTests(unittest.TestCase):
    """3 族は自分の forward の真の導関数（S-fd）。act_curv は act_grad の導関数。"""

    def _points(self, act: str, alpha: float) -> torch.Tensor:
        z = GRID[(GRID.abs() < 8.0)]
        if act == "snake1":
            for cut in _lobe(alpha):
                z = z[(z - cut).abs() > 1e-3]
        return z

    def test_central_difference_matches_act_grad(self):
        h, tol = 1e-6, 1e-6
        for act in NEW:
            for alpha in ALPHAS:
                with self.subTest(act=act, alpha=alpha):
                    n = _net(act, alpha)
                    z = self._points(act, alpha)
                    fd = (n.act_fn(z + h) - n.act_fn(z - h)) / (2 * h)
                    self.assertLess(float((fd - n.act_grad(z, n.act_fn(z))).abs().max()), tol)

    def test_central_difference_of_act_grad_matches_act_curv(self):
        h, tol = 1e-5, 1e-5
        for act in NEW + ("snake",):
            for alpha in ALPHAS:
                with self.subTest(act=act, alpha=alpha):
                    n = _net(act, alpha)
                    z = self._points(act, alpha)
                    g = lambda x: n.act_grad(x, n.act_fn(x))
                    fd = (g(z + h) - g(z - h)) / (2 * h)
                    self.assertLess(float((fd - n.act_curv(z)).abs().max()), tol)

    def test_amp_bounds_the_gate_and_the_flip_stays_at_the_snake_zero(self):
        """φ' ∈ [1−A, 1+A]、変曲点（φ''=0）は A に依らず −π/4α（spec §1）。"""
        for name, A in VecMLPL.SNAKE_AMP.items():
            n = _net(name, 1.0)
            g = n.act_grad(GRID, n.act_fn(GRID))
            self.assertAlmostEqual(float(g.min()), 1.0 - A, places=6)
            self.assertAlmostEqual(float(g.max()), 1.0 + A, places=6)
            z1 = torch.tensor([-math.pi / 4.0], dtype=torch.float64)
            self.assertLess(float(n.act_curv(z1).abs()), 1e-12)


class ConfigTests(unittest.TestCase):
    def test_new_config_has_the_registered_four_arms_and_five_names(self):
        from src import edge_law_0905 as E
        cfg = load_config(str(CFG_NEW))
        self.assertEqual([a["name"] for a in cfg["arms"]],
                         ["SN_a1_1216", "SN1_a1_1216", "SNA05_a1_1216", "SNA025_a1_1216"])
        for a in cfg["arms"]:
            self.assertEqual(a["hook"], {"type": "lr", "value": 0.005})
            self.assertIn(a["activation"], VecMLPL.ACTIVATIONS)
        built = E.build_cfg(CFG_NEW)
        names = {a["name"] for a in built["arms"]}
        self.assertTrue({"SN_a1_1216", "SN1_a1_1216", "SNA05_a1_1216", "SNA025_a1_1216"} <= names)
        for act in ("snake", "snake1", "snake_amp0p5", "snake_amp0p25"):
            self.assertEqual(built["activation"][act]["name"], act)

    def test_default_edge_law_table_is_unchanged_by_the_new_option(self):
        """S-cfg: --config を使わない edge_law_0905 は commit 済みの腕表のまま。"""
        from src import edge_law_0905 as E
        table = E.arm_table(load_config(str(CFG_EDGE)))
        self.assertEqual(len(table), 30)
        self.assertNotIn("SN_a1_1216", table)
        self.assertEqual(E.CONFIG, CFG_EDGE)

    def test_arm_table_from_new_config(self):
        from src import edge_law_0905 as E
        table = E.arm_table(load_config(str(CFG_NEW)))
        self.assertEqual(list(table), ["SN_a1_1216", "SN1_a1_1216", "SNA05_a1_1216", "SNA025_a1_1216"])
        self.assertEqual(table["SN1_a1_1216"]["activation"], "snake1")
        self.assertEqual(table["SN1_a1_1216"]["hook"], {"type": "lr", "value": 0.005})


if __name__ == "__main__":
    unittest.main()
