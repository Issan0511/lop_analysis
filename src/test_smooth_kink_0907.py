"""smooth_kink_0907 の検査（spec `specs/spec_smooth_kink_0907.md` §7）。

    OMP_NUM_THREADS=1 PYTHONPATH=. python3 -m unittest src.test_smooth_kink_0907 -v

S-fd（φ′・φ″ が差分と合う）/ S-limit（s→0 で leaky・隆起の上限）/ S-mono（φ′ は [a,1] で単調）
/ S-num（裾で overflow しない）/ S-guard / S-fallthrough（既存の名前の式が変わっていない）
/ S-cfg（config の `smooth` と dict が一致・腕表が読める）。
**bit 一致・恒等式の検査には必ず「変異体が落ちる」対を置く**（空虚な S 検査は通算 6 回目・
[[proj-004-edge-law-0905]]）。様式は test_act_offset_0906 に倣う。
"""
from __future__ import annotations

import math
import unittest
from pathlib import Path

import torch

from src import edge_law_0905 as E
from src.common import ROOT, load_config
from src.nets import VecMLPL

GRID = torch.linspace(-30, 30, 24001, dtype=torch.float64)
EXTRA = torch.tensor([0.0, -0.0, 5e-324, -5e-324, 1e-38, -1e-38, 1e-3, -1e-3, 1.0, -1.0],
                     dtype=torch.float64)
NEW = tuple(VecMLPL.SMOOTH_LEAKY)
SLOPES = (0.0, 0.1, 0.5, 0.9)
CFG = Path(ROOT) / "configs" / "smooth_kink_0907.yaml"
ARMS = ["SKref_1216", "SKa0p5ref_1216", "SKs0p01_1216", "SKs0p03_1216", "SKs0p1_1216",
        "SKs0p3_1216", "SKs1_1216", "SKs3_1216", "SKa0p5_s0p1_1216",
        "SKa0p5_s0p3_1216", "SKa0p5_s1_1216", "SKa0p5_s2_1216"]


def _net(act: str, alpha: float) -> VecMLPL:
    net = VecMLPL(1, [2], 2, torch.Generator().manual_seed(0), "cpu")
    return net.set_activation(act, alpha, "alpha_exp")


def _leaky(z: torch.Tensor, a: float) -> torch.Tensor:
    return torch.where(z > 0, z, a * z)


class TestNames(unittest.TestCase):
    def test_registered(self):
        for act in NEW:
            self.assertIn(act, VecMLPL.ACTIVATIONS)
            self.assertIn(act, VecMLPL.WEIRD_SLOPE_ACTIVATIONS)   # [0,1] ガード共有
            # φ'' は 0 でないので零曲率の族に入れてはいけない（入れると m_dphiddphi が黙って 0）
            self.assertNotIn(act, VecMLPL.ZERO_CURVATURE_ACTIVATIONS)

    def test_widths_positive_and_sorted(self):
        w = [VecMLPL.SMOOTH_LEAKY[a] for a in NEW]
        self.assertTrue(all(x > 0 for x in w))
        self.assertEqual(w, sorted(w))

    def test_curv_registered(self):
        """未登録名は NotImplementedError のまま（S-guard の対）。"""
        for act in NEW:
            self.assertTrue(torch.isfinite(_net(act, 0.1).act_curv(GRID)).all())
        with self.assertRaises(NotImplementedError):
            _net("bwd_leaky", 0.1).act_curv(GRID)


class TestGuard(unittest.TestCase):
    def test_slope_guard(self):
        for act in NEW:
            for bad in (-0.1, 1.1):
                with self.assertRaises(ValueError):
                    _net(act, bad)
            _net(act, 0.0)
            _net(act, 1.0)

    def test_unknown_name(self):
        with self.assertRaises(ValueError):
            _net("smleaky_s5", 0.1)      # 未登録の幅（s=2 は追補 1 で登録済み）


class TestLimit(unittest.TestCase):
    """S-limit: s→0 で leaky。隆起は (1-a)·s·log2 以下で、これが上限として厳密。"""

    def test_bump_bound(self):
        for act in NEW:
            s = VecMLPL.SMOOTH_LEAKY[act]
            for a in SLOPES:
                d = _net(act, a).act_fn(GRID) - _leaky(GRID, a)
                self.assertTrue((d >= 0).all())
                self.assertLessEqual(float(d.max()), (1.0 - a) * s * math.log(2) + 1e-12)

    def test_bump_at_zero_is_exact(self):
        z = torch.zeros(1, dtype=torch.float64)
        for act in NEW:
            s = VecMLPL.SMOOTH_LEAKY[act]
            for a in SLOPES:
                got = float(_net(act, a).act_fn(z))
                self.assertAlmostEqual(got, (1.0 - a) * s * math.log(2), places=10)

    def test_small_s_is_close_to_leaky(self):
        """s=0.01 は leaky と 0.007 以内。s=3 はそうならない（変異体の対）。"""
        a = 0.1
        near = (_net("smleaky_s0p01", a).act_fn(GRID) - _leaky(GRID, a)).abs().max()
        far = (_net("smleaky_s3", a).act_fn(GRID) - _leaky(GRID, a)).abs().max()
        self.assertLess(float(near), 0.007)
        self.assertGreater(float(far), 1.8)

    def test_far_from_kink_is_bit_identical_to_leaky(self):
        """|z| ≫ s では leaky と **bit 一致**（隆起は局所であることの検査）。"""
        for act in NEW:
            s = VecMLPL.SMOOTH_LEAKY[act]
            z = torch.tensor([-60.0 * s - 40.0, 60.0 * s + 40.0], dtype=torch.float64)
            for a in (0.1, 0.5):
                got = _net(act, a).act_fn(z)
                self.assertEqual(got.numpy().tobytes(), _leaky(z, a).numpy().tobytes())
        # 対（変異体）: 折れ目のすぐそばでは bit 一致しない
        z0 = torch.tensor([0.0, 0.1], dtype=torch.float64)
        got = _net("smleaky_s1", 0.1).act_fn(z0)
        self.assertNotEqual(got.numpy().tobytes(), _leaky(z0, 0.1).numpy().tobytes())


class TestDerivatives(unittest.TestCase):
    """S-fd: φ′ と φ″ が中心差分と合う（変異体は落ちる）。"""

    def _fd(self, f, z, h):
        return (f(z + h) - f(z - h)) / (2.0 * h)

    def test_grad_matches_fd(self):
        z = torch.linspace(-8, 8, 3201, dtype=torch.float64)
        for act in NEW:
            s = VecMLPL.SMOOTH_LEAKY[act]
            h = 1e-4 * min(1.0, s)
            for a in (0.1, 0.5):
                net = _net(act, a)
                num = self._fd(net.act_fn, z, h)
                ana = net.act_grad(z, net.act_fn(z))
                self.assertLess(float((num - ana).abs().max()), 1e-6)
                # 対: 傾きを取り違えた変異体は落ちる
                bad = net.act_grad(z, net.act_fn(z)) + 0.01
                self.assertGreater(float((num - bad).abs().max()), 1e-3)

    def test_curv_matches_fd(self):
        z = torch.linspace(-8, 8, 3201, dtype=torch.float64)
        for act in NEW:
            s = VecMLPL.SMOOTH_LEAKY[act]
            h = 1e-4 * min(1.0, s)
            for a in (0.1, 0.5):
                net = _net(act, a)
                num = self._fd(lambda t: net.act_grad(t, net.act_fn(t)), z, h)
                ana = net.act_curv(z)
                self.assertLess(float((num - ana).abs().max()), 1e-5 / s)

    def test_grad_range_and_monotone(self):
        """φ′ は a から 1 へ単調（ラチェットの向きが名前どおりであること）。"""
        z = torch.linspace(-40, 40, 8001, dtype=torch.float64)
        for act in NEW:
            for a in (0.1, 0.5):
                g = _net(act, a).act_grad(z, None)
                self.assertGreaterEqual(float(g.min()), a - 1e-12)
                self.assertLessEqual(float(g.max()), 1.0 + 1e-12)
                self.assertTrue(bool((g[1:] - g[:-1] >= -1e-15).all()))

    def test_curv_peak_scales_as_inverse_s(self):
        """φ″ の頂点は (1-a)/(4s)。s→0 でデルタに寄ることの数値的な担保。"""
        z = torch.zeros(1, dtype=torch.float64)
        for act in NEW:
            s = VecMLPL.SMOOTH_LEAKY[act]
            for a in (0.1, 0.5):
                self.assertAlmostEqual(float(_net(act, a).act_curv(z)),
                                       (1.0 - a) / (4.0 * s), places=9)


class TestNumerics(unittest.TestCase):
    def test_no_overflow_in_tails(self):
        z = torch.tensor([-1e30, -1e6, 1e6, 1e30], dtype=torch.float64)
        for act in NEW:
            for a in (0.0, 0.1, 1.0):
                net = _net(act, a)
                self.assertTrue(torch.isfinite(net.act_fn(z)).all())
                self.assertTrue(torch.isfinite(net.act_grad(z, None)).all())
                self.assertTrue(torch.isfinite(net.act_curv(z)).all())

    def test_float32_path(self):
        z = torch.linspace(-30, 30, 2001, dtype=torch.float32)
        for act in NEW:
            net = _net(act, 0.1)
            self.assertTrue(torch.isfinite(net.act_fn(z)).all())
            self.assertEqual(net.act_fn(z).dtype, torch.float32)


class TestFallthrough(unittest.TestCase):
    """既存の名前の式が 1 つも変わっていない（足すだけの検査）。"""

    def test_existing_activations_unchanged(self):
        z = torch.linspace(-6, 6, 1201, dtype=torch.float64)
        for act, a in (("relu", 1.0), ("leaky_relu", 0.1), ("elu", 1.0),
                       ("leaky_off_0", 0.1), ("leaky_off_p0p5", 0.1)):
            net = _net(act, a)
            if act == "relu":
                want = torch.relu(z)
            elif act in ("leaky_relu", "leaky_off_0"):
                want = _leaky(z, a)
            elif act == "leaky_off_p0p5":
                want = _leaky(z, a) + 0.5
            else:
                want = torch.where(z > 0, z, a * torch.expm1(z))
            self.assertEqual(net.act_fn(z).numpy().tobytes(), want.numpy().tobytes())

    def test_leaky_off_0_still_bit_identical(self):
        z = torch.cat([GRID, EXTRA])
        self.assertEqual(_net("leaky_off_0", 0.1).act_fn(z).numpy().tobytes(),
                         _net("leaky_relu", 0.1).act_fn(z).numpy().tobytes())


class TestConfig(unittest.TestCase):
    def test_config_smooth_matches_dict(self):
        cfg = load_config(CFG)
        for name, row in cfg["activation"].items():
            self.assertEqual(name, row["name"])
            if name in VecMLPL.SMOOTH_LEAKY:
                self.assertEqual(float(row["smooth"]), VecMLPL.SMOOTH_LEAKY[name])
            else:
                self.assertEqual(float(row["smooth"]), 0.0)
        used = {r["activation"] for r in cfg["arms"]}
        self.assertTrue(used <= set(cfg["activation"]))
        self.assertEqual(set(VecMLPL.SMOOTH_LEAKY) & used, set(VecMLPL.SMOOTH_LEAKY))

    def test_arm_table(self):
        cfg = load_config(CFG)
        self.assertEqual([r["name"] for r in cfg["arms"]], ARMS)
        for r in cfg["arms"]:
            self.assertIsNone(r["hook"])
            self.assertEqual(int(r["total_steps"]), 5_000_000)
            self.assertIn(float(r["dial"]), (0.1, 0.5))

    def test_analysis_block(self):
        an = load_config(CFG)["analysis"]
        self.assertEqual(an["ladder_a0p1"],
                         ["SKs0p01_1216", "SKs0p03_1216", "SKs0p1_1216",
                          "SKs0p3_1216", "SKs1_1216", "SKs3_1216"])
        for arm, s in an["s_values"].items():
            act = [r["activation"] for r in load_config(CFG)["arms"] if r["name"] == arm][0]
            self.assertEqual(float(s), VecMLPL.SMOOTH_LEAKY[act])
        self.assertEqual(an["s_null_pairs"]["SKref_1216"], "LRoff0_1216")

    def test_build_cfg_runs(self):
        """腕表が宿主 runner の形に落ちる（走らせる前の型検査）。"""
        cfg = E.build_cfg(CFG)
        # build_cfg は宿主（edge_law_0905）の腕表の**後ろに**本走の腕を足す
        names = [r["name"] for r in cfg["arms"]]
        self.assertEqual(names[-len(ARMS):], ARMS)
        self.assertEqual(len(set(names)), len(names))
        for r in cfg["arms"][-len(ARMS):]:
            self.assertEqual(int(r["hidden"][0]), 100)


class TestSupportAveraging(unittest.TestCase):
    """spec §2 の要（s ≪ W なら支持平均は s にほぼ依らない）を数値で確かめる。

    ここが成り立たないと梯子の設計そのものが無意味になるので、走らせる前に置く。
    """

    def test_mob_insensitive_for_small_s(self):
        a, W = 0.1, 3.79
        z = torch.linspace(-W, W, 2001, dtype=torch.float64) - 3.6      # 支持（幅 2W）
        ref = float(_net("leaky_off_0", a).act_grad(z, None).mean())
        for act in ("smleaky_s0p01", "smleaky_s0p03", "smleaky_s0p1"):
            got = float(_net(act, a).act_grad(z, None).mean())
            self.assertLess(abs(got - ref), 0.02 * max(ref, 1e-6) + 0.005)
        far = float(_net("smleaky_s3", a).act_grad(z, None).mean())
        self.assertGreater(abs(far - ref), 0.02)


if __name__ == "__main__":
    unittest.main()
