"""act_offset_0906 の検査（spec `specs/spec_act_offset_0906.md` §5）。

    OMP_NUM_THREADS=1 PYTHONPATH=. python3 -m unittest src.test_act_offset_0906 -v

S-fd / S-limit（格子 ＋ 30k 走行）/ S-shift / S-fallthrough / S-guard / S-cfg ＋ 解析のラベル論理。
様式は test_snake_flip_0906 に倣う。**すべての bit 一致検査に「変異体が落ちる」対を置く**
（空虚な S 検査は本プロジェクトで通算 6 回目まで来ている・[[proj-004-edge-law-0905]]）。
"""
from __future__ import annotations

import copy
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src import act_offset_analyze_0906 as AN
from src import act_offset_preflight_0906 as P
from src import edge_law_0905 as E
from src.common import ROOT, load_config
from src.nets import VecMLPL

GRID = torch.linspace(-30, 30, 24001, dtype=torch.float64)
# 格子には 0 も折れ目も入っていない（中央は -6.2e-16）ので明示の追加点を足す
# （edge_law §5 S-limit の流儀・±0.0 を必ず含める）。
EXTRA = torch.tensor([0.0, -0.0, 5e-324, -5e-324, 1e-38, -1e-38, 1e-3, -1e-3, 1.0, -1.0],
                     dtype=torch.float64)
LEAKY_NEW = tuple(VecMLPL.LEAKY_OFFSET)
ELU_NEW = tuple(VecMLPL.ELU_OFFSET)
NEW = LEAKY_NEW + ELU_NEW
SLOPES = (0.1, 0.5, 1.0)
ELU_ALPHAS = (0.5, 1.0, 2.0)
CFG_NEW = Path(ROOT) / "configs" / "act_offset_0906.yaml"
CFG_EDGE = Path(ROOT) / "configs" / "edge_law_0905.yaml"
ARMS = ["LRoff0_1216", "LRoffm2_1216", "LRoffm0p5_1216", "LRoffp0p5_1216",
        "LRoffp2_1216", "Eoffm1_1216", "Eoffp1_1216"]
# 追補 1（specs/spec_act_offset_0906_addendum1.md）の低 lr ラダー
ARMS_B = ["LRoff0_lr0p00125_1216", "LRoffm2_lr0p00125_1216", "LRoffp2_lr0p00125_1216"]


def _net(act: str, alpha: float) -> VecMLPL:
    net = VecMLPL(1, [2], 2, torch.Generator().manual_seed(0), "cpu")
    return net.set_activation(act, alpha, "alpha_exp")


def _bytes(t: torch.Tensor) -> bytes:
    return t.contiguous().numpy().tobytes()


def _dials(act: str):
    return SLOPES if act in VecMLPL.LEAKY_OFFSET else ELU_ALPHAS


def _base(act: str) -> str:
    return "leaky_relu" if act in VecMLPL.LEAKY_OFFSET else "elu"


def _offset(act: str) -> float:
    return VecMLPL.LEAKY_OFFSET.get(act, VecMLPL.ELU_OFFSET.get(act))


def _three(net: VecMLPL, z: torch.Tensor):
    f = net.act_fn(z)
    return f, net.act_grad(z, f), net.act_curv(z)


# ---------------------------------------------------------------------------
# S-guard / S-const
# ---------------------------------------------------------------------------
class GuardTests(unittest.TestCase):
    def test_all_offset_names_are_registered(self):
        # 本編 7 名 ＋ offset_grid_0906 で足した 4 名（±1・±0.25）
        self.assertEqual(len(NEW), 11)
        self.assertEqual(len(LEAKY_NEW), 9)
        for act in NEW:
            self.assertIn(act, VecMLPL.ACTIVATIONS)

    def test_leaky_off_is_zero_curvature_and_slope_guarded(self):
        for act in LEAKY_NEW:
            self.assertIn(act, VecMLPL.ZERO_CURVATURE_ACTIVATIONS)
            self.assertIn(act, VecMLPL.WEIRD_SLOPE_ACTIVATIONS)
            _net(act, 0.1)                       # 登録 dial は通る
            for bad in (1.5, -0.1):
                with self.assertRaises(ValueError):
                    _net(act, bad)

    def test_elu_off_inherits_the_nonnegative_alpha_guard(self):
        for act in ELU_NEW:
            self.assertNotIn(act, VecMLPL.ZERO_CURVATURE_ACTIVATIONS)
            _net(act, 1.0)
            with self.assertRaises(ValueError):
                _net(act, -1.0)

    def test_unknown_name_is_still_rejected(self):
        with self.assertRaises(ValueError):
            _net("leaky_off_p3", 0.1)

    def test_offsets_match_the_config(self):
        """S-const: 名前に埋めた定数 c と config の `offset` が一致。本編の 7 名は act_offset の
        config に、offset_grid_0906 で足した 4 名（±1・±0.25）はその config に居る。"""
        cfg = load_config(str(CFG_NEW))
        grid = load_config(str(Path(ROOT) / "configs" / "offset_grid_0906.yaml"))
        for act in NEW:
            src = cfg if act in cfg["activation"] else grid
            self.assertIn(act, src["activation"], act)
            self.assertEqual(float(src["activation"][act]["offset"]), _offset(act), act)
        for act in ELU_NEW + ("leaky_off_m2", "leaky_off_m0p5", "leaky_off_0", "leaky_off_p0p5", "leaky_off_p2"):
            self.assertIn(act, cfg["activation"], act)       # 本編の 7 名は本編の config に
        for act in ("leaky_off_m1", "leaky_off_p1", "leaky_off_m0p25", "leaky_off_p0p25"):
            self.assertIn(act, grid["activation"], act)
        self.assertEqual(VecMLPL.LEAKY_OFFSET["leaky_off_0"], 0.0)

    def test_activation_choice_consumes_no_randomness(self):
        a = VecMLPL(1, [4], 3, torch.Generator().manual_seed(7), "cpu", act="relu")
        b = VecMLPL(1, [4], 3, torch.Generator().manual_seed(7), "cpu", act="leaky_off_p2")
        self.assertEqual(_bytes(a.Ws[0]), _bytes(b.Ws[0]))
        self.assertEqual(_bytes(a.v), _bytes(b.v))


# ---------------------------------------------------------------------------
# S-fallthrough（if 連鎖の最後は ELU。分岐の書き忘れは黙って ELU になる）
# ---------------------------------------------------------------------------
class FallthroughTests(unittest.TestCase):
    def test_leaky_off_does_not_fall_through_to_elu_in_any_direction(self):
        z = GRID
        for act in LEAKY_NEW:
            for a in SLOPES:
                with self.subTest(act=act, a=a):
                    fe, ge, ce = _three(_net("elu", a), z)
                    f, g, c = _three(_net(act, a), z)
                    self.assertFalse(torch.allclose(f, fe))
                    self.assertFalse(torch.allclose(g, ge))
                    self.assertFalse(torch.allclose(c, ce))

    def test_elu_off_does_not_fall_through_to_leaky_in_any_direction(self):
        """elu_off_* は φ′・φ″ が ELU と同一（設計）なので、対照は leaky 分岐（spec §5）。"""
        z = GRID
        for act in ELU_NEW:
            for a in ELU_ALPHAS:
                with self.subTest(act=act, a=a):
                    fl, gl, cl = _three(_net("leaky_relu", min(a, 1.0)), z)
                    fe, ge, ce = _three(_net("elu", a), z)
                    f, g, c = _three(_net(act, a), z)
                    self.assertFalse(torch.allclose(f, fl))
                    self.assertFalse(torch.allclose(g, gl))
                    self.assertFalse(torch.allclose(c, cl))
                    # 陽性対照: 導関数側は ELU と一致し、forward だけが c ぶんずれる
                    self.assertTrue(torch.allclose(g, ge))
                    self.assertTrue(torch.allclose(c, ce))
                    self.assertFalse(torch.allclose(f, fe))

    def test_the_elu_reference_itself_is_not_degenerate(self):
        f, g, c = _three(_net("elu", 1.0), GRID)
        self.assertGreater(float(f.abs().max()), 1.0)
        self.assertGreater(float((g - 1.0).abs().max()), 0.5)
        self.assertGreater(float(c.abs().max()), 0.5)


# ---------------------------------------------------------------------------
# S-limit（格子）: leaky_off_0 ≡ leaky_relu をバイトで
# ---------------------------------------------------------------------------
class LimitGridTests(unittest.TestCase):
    def test_extra_points_actually_contain_negative_zero(self):
        signs = [bool(torch.signbit(x)) for x in EXTRA]
        self.assertIn(True, signs)
        self.assertTrue(any(x == 0.0 and torch.signbit(x) for x in EXTRA))

    def test_leaky_off_0_is_byte_identical_to_leaky_in_all_three_directions(self):
        for a in SLOPES:
            for dtype in (torch.float64, torch.float32):
                with self.subTest(a=a, dtype=dtype):
                    z = torch.cat([GRID, EXTRA]).to(dtype)
                    fa, ga, ca = _three(_net("leaky_off_0", a), z)
                    fb, gb, cb = _three(_net("leaky_relu", a), z)
                    self.assertEqual(_bytes(fa), _bytes(fb))
                    self.assertEqual(_bytes(ga), _bytes(gb))
                    self.assertEqual(_bytes(ca), _bytes(cb))

    def test_a_nonzero_offset_must_FAIL_the_same_comparison(self):
        """S-limit の生存証明。"""
        z = torch.cat([GRID, EXTRA])
        fa, ga, _ = _three(_net("leaky_off_p0p5", 0.1), z)
        fb, gb, _ = _three(_net("leaky_relu", 0.1), z)
        self.assertNotEqual(_bytes(fa), _bytes(fb))
        self.assertEqual(_bytes(ga), _bytes(gb))       # 導関数側は設計どおり同一

    def test_adding_plus_zero_would_break_byte_identity_at_negative_zero(self):
        """spec §3 の但し書きの実証: `x + 0.0` は -0.0 を +0.0 に変えるので c=0 は加算しない。"""
        f = _net("leaky_relu", 0.1).act_fn(EXTRA)
        self.assertNotEqual(_bytes(f + 0.0), _bytes(f))
        self.assertEqual(_bytes(_net("leaky_off_0", 0.1).act_fn(EXTRA)), _bytes(f))


# ---------------------------------------------------------------------------
# S-shift: leaky_off_c(z) − leaky_relu(z) == c、act_grad はバイト一致
# ---------------------------------------------------------------------------
class ShiftTests(unittest.TestCase):
    def test_forward_is_the_base_expression_plus_c_byte_exact(self):
        """forward は「元の族の式 + c」そのもの（`(x + c) − x == c` は浮動小数で恒真ではないので、
        バイト厳密は「元の式に c を足したテンソル」との一致で書く）。"""
        z = torch.cat([GRID, EXTRA])
        for act in NEW:
            c = _offset(act)
            if c == 0.0:
                continue                             # c=0 は LimitGridTests
            for a in _dials(act):
                with self.subTest(act=act, a=a):
                    f = _net(act, a).act_fn(z)
                    fb = _net(_base(act), a).act_fn(z)
                    self.assertEqual(_bytes(f), _bytes(fb + c))

    def test_difference_equals_c_within_float64_rounding(self):
        z = GRID
        for act in NEW:
            c = _offset(act)
            for a in _dials(act):
                f = _net(act, a).act_fn(z)
                fb = _net(_base(act), a).act_fn(z)
                self.assertLess(float(((f - fb) - c).abs().max()), 1e-13, (act, a))

    def test_grad_and_curv_are_byte_identical_to_the_base_family(self):
        z = torch.cat([GRID, EXTRA])
        for act in NEW:
            for a in _dials(act):
                with self.subTest(act=act, a=a):
                    _, g, cv = _three(_net(act, a), z)
                    _, gb, cb = _three(_net(_base(act), a), z)
                    self.assertEqual(_bytes(g), _bytes(gb))
                    self.assertEqual(_bytes(cv), _bytes(cb))

    def test_a_wrong_offset_fails_the_shift_comparison(self):
        z = torch.cat([GRID, EXTRA])
        for act in ("leaky_off_p2", "elu_off_m1"):
            c = _offset(act)
            f = _net(act, _dials(act)[0]).act_fn(z)
            fb = _net(_base(act), _dials(act)[0]).act_fn(z)
            self.assertNotEqual(_bytes(f), _bytes(fb + (c + 0.1)))
            self.assertNotEqual(_bytes(f), _bytes(fb - c))


# ---------------------------------------------------------------------------
# S-fd: 7 活性化の act_grad は forward の真の導関数、act_curv は act_grad の導関数
# ---------------------------------------------------------------------------
class FiniteDifferenceTests(unittest.TestCase):
    @staticmethod
    def _points() -> torch.Tensor:
        z = GRID[GRID.abs() < 8.0]
        return z[z.abs() > 1e-3]                     # 折れ目 ±1e-3 を除外

    def test_central_difference_matches_act_grad(self):
        h, tol, z = 1e-6, 1e-6, self._points()
        for act in NEW:
            for a in _dials(act):
                with self.subTest(act=act, a=a):
                    n = _net(act, a)
                    fd = (n.act_fn(z + h) - n.act_fn(z - h)) / (2 * h)
                    self.assertLess(float((fd - n.act_grad(z, n.act_fn(z))).abs().max()), tol)

    def test_central_difference_of_act_grad_matches_act_curv(self):
        h, tol, z = 1e-5, 1e-5, self._points()
        for act in NEW:
            for a in _dials(act):
                with self.subTest(act=act, a=a):
                    n = _net(act, a)
                    g = lambda x: n.act_grad(x, n.act_fn(x))
                    fd = (g(z + h) - g(z - h)) / (2 * h)
                    self.assertLess(float((fd - n.act_curv(z)).abs().max()), tol)

    def test_leaky_off_curvature_is_identically_plus_zero(self):
        z = torch.cat([GRID, EXTRA])
        for act in LEAKY_NEW:
            c = _net(act, 0.1).act_curv(z)
            self.assertTrue(torch.all(c == 0.0))
            self.assertFalse(bool(torch.signbit(c).any()))

    def test_the_finite_difference_is_not_vacuous(self):
        """導関数を 1 か所壊すと落ちる（変異対照）。"""
        z = self._points()
        n = _net("leaky_off_p2", 0.1)
        fd = (n.act_fn(z + 1e-6) - n.act_fn(z - 1e-6)) / 2e-6
        wrong = torch.where(z > 0, torch.ones_like(z), torch.full_like(z, 0.2))
        self.assertGreater(float((fd - wrong).abs().max()), 0.05)


# ---------------------------------------------------------------------------
# S-cfg
# ---------------------------------------------------------------------------
class ConfigTests(unittest.TestCase):
    def test_default_edge_law_table_is_unchanged(self):
        table = E.arm_table(load_config(str(CFG_EDGE)))
        self.assertEqual(len(table), 30)
        for arm in ARMS:
            self.assertNotIn(arm, table)
        self.assertEqual(E.CONFIG, CFG_EDGE)

    def test_new_config_gives_the_ten_registered_arms_verbatim(self):
        table = E.arm_table(load_config(str(CFG_NEW)))
        self.assertEqual(list(table), ARMS + ARMS_B)
        for name in ARMS:
            row = table[name]
            self.assertIsNone(row["hook"])
            self.assertEqual(row["total_steps"], 5_000_000)
            self.assertEqual(row["checkpoints"], [0, 1_000_000, 5_000_000])
            if name.startswith("LR"):
                self.assertEqual((row["family"], row["dial"], row["u_fr"]), ("leaky", 0.1, None))
            else:
                self.assertEqual((row["family"], row["dial"], row["u_fr"]), ("elu", 1.0, 13.8155))

    def test_addendum_arms_are_the_low_lr_ladder(self):
        """追補 1 §2: {c=0, −2, +2} を lr 0.00125・40M（eta*step = 50,000 を本編と揃える）で。"""
        table = E.arm_table(load_config(str(CFG_NEW)))
        want_c = {"LRoff0_lr0p00125_1216": "leaky_off_0",
                  "LRoffm2_lr0p00125_1216": "leaky_off_m2",
                  "LRoffp2_lr0p00125_1216": "leaky_off_p2"}
        for name in ARMS_B:
            row = table[name]
            self.assertEqual(row["hook"], {"type": "lr", "value": 0.00125}, name)
            self.assertEqual(row["total_steps"], 40_000_000, name)
            self.assertEqual(row["checkpoints"], [0, 1_000_000, 5_000_000, 40_000_000], name)
            self.assertEqual((row["family"], row["dial"], row["u_fr"]), ("leaky", 0.1, None), name)
            self.assertEqual(row["activation"], want_c[name], name)
        # eta*step が本編（0.01 x 5M）と一致する
        self.assertEqual(0.00125 * 40_000_000, 0.01 * 5_000_000)
        # 活性化は 1 つも増えていない（既存の leaky_off_* をそのまま使う）
        self.assertEqual({table[n]["activation"] for n in ARMS_B},
                         {"leaky_off_0", "leaky_off_m2", "leaky_off_p2"})

    def test_expected_lr_reads_the_hook(self):
        for name in ARMS:
            self.assertEqual(P.expected_lr(name, CFG_NEW), 0.01, name)
        for name in ARMS_B:
            self.assertEqual(P.expected_lr(name, CFG_NEW), 0.00125, name)

    def test_elu_u_fr_equals_the_edge_law_reference_arm(self):
        edge = E.arm_table(load_config(str(CFG_EDGE)))
        new = E.arm_table(load_config(str(CFG_NEW)))
        self.assertEqual(new["Eoffm1_1216"]["u_fr"], edge["Enull_1216"]["u_fr"])

    def test_build_cfg_appends_the_seven_activations_and_arms(self):
        host = load_config(str(E.HOST_CONFIG))
        built = E.build_cfg(CFG_NEW)
        for act in load_config(str(CFG_NEW))["activation"]:      # 本編 config の 7 名
            self.assertEqual(built["activation"][act]["name"], act)
        self.assertEqual([a["name"] for a in built["arms"]],
                         [a["name"] for a in host["arms"]] + ARMS + ARMS_B)
        self.assertEqual(built["common"]["lr_main"], 0.01)
        self.assertEqual(built["common"]["seeds"], list(range(10)))
        for a in built["arms"][-10:]:
            self.assertEqual(a["hidden"], [100])
            self.assertEqual(a["target_dose"], 12.16)

    def test_every_arm_resolves_to_a_registered_activation_and_dial(self):
        cfg = load_config(str(CFG_NEW))
        for a in cfg["arms"]:
            name = cfg["activation"][a["activation"]]["name"]
            self.assertIn(name, VecMLPL.ACTIVATIONS)
            _net(name, float(a["dial"]))
            self.assertEqual(float(cfg["activation"][a["activation"]]["offset"]), _offset(name))

    def test_analysis_block_names_the_reference_and_the_judged_arms(self):
        A = load_config(str(CFG_NEW))["analysis"]
        self.assertEqual(A["reference_arm"], "LRoff0_1216")
        self.assertEqual(A["judged_arms"], ARMS[1:5])
        self.assertEqual(A["tail_window_tasks"], [451, 500])
        self.assertEqual(A["bands"], {"dzmax_irrelevant": 0.3, "dzbar_irrelevant": 0.5})
        self.assertEqual(A["bootstrap"]["seed"], AN.RNG_SEED)

    def test_addendum_analysis_block(self):
        B = load_config(str(CFG_NEW))["analysis"]["addendum1"]
        self.assertEqual(B["reference_arm"], "LRoff0_lr0p00125_1216")
        self.assertEqual(B["judged_arms"], ARMS_B[1:])
        self.assertEqual(B["lr"], 0.00125)
        self.assertEqual(B["tail_window_tasks"], [3951, 4000])
        self.assertEqual(B["settle_windows_tasks"],
                         [[2751, 2800], [3351, 3400], [3951, 4000]])
        self.assertFalse(B["require_monotone"])
        self.assertEqual(B["label_suffix"], "_B")

    def test_the_addendum_windows_keep_the_main_ladder_fractions(self):
        """追補 1 §3: 主窓は末尾 50 タスク、settle は地平線の 70% / 85% / 100%。"""
        cfg = load_config(str(CFG_NEW))
        A, B = cfg["analysis"], cfg["analysis"]["addendum1"]
        n_a, n_b = 500, 4000                                    # 5M / 40M を task_period 10,000 で
        self.assertEqual(A["tail_window_tasks"][1], n_a)
        self.assertEqual(B["tail_window_tasks"][1], n_b)
        for wa, wb in zip(A["settle_windows_tasks"], B["settle_windows_tasks"]):
            self.assertEqual(wa[1] / n_a, wb[1] / n_b)          # 終端の割合が一致
            self.assertEqual(wa[1] - wa[0], wb[1] - wb[0])      # 窓幅は 50 タスクのまま


class LadderTests(unittest.TestCase):
    def test_two_ladders_with_the_registered_shapes(self):
        L = AN.ladders(load_config(str(CFG_NEW)))
        self.assertEqual([d["name"] for d in L], ["A", "B"])
        a, b = L
        self.assertEqual((a["lr"], a["ref"], a["tail"], a["expect_c"], a["require_monotone"],
                          a["suffix"]),
                         (0.01, "LRoff0_1216", (451, 500), AN.JUDGED_C, True, ""))
        self.assertEqual((b["lr"], b["ref"], b["tail"], b["expect_c"], b["require_monotone"],
                          b["suffix"]),
                         (0.00125, "LRoff0_lr0p00125_1216", (3951, 4000), AN.JUDGED_C_B,
                          False, "_B"))
        self.assertEqual(b["settles"], [(2751, 2800), (3351, 3400), (3951, 4000)])

    def test_a_config_without_the_addendum_gives_one_ladder(self):
        cfg = load_config(str(CFG_NEW))
        cfg["analysis"].pop("addendum1")
        self.assertEqual([d["name"] for d in AN.ladders(cfg)], ["A"])


# ---------------------------------------------------------------------------
# 解析のラベル論理（spec §4）— 純関数なので合成入力で網羅する
# ---------------------------------------------------------------------------
def _d(pt: float, lo: float, hi: float):
    return (pt, (lo, hi))


def _deltas(zmax: dict, zbar: dict) -> dict:
    return {c: dict(dzmax=zmax[c], dzbar=zbar[c]) for c in (-2.0, -0.5, 0.5, 2.0)}


class LabelTests(unittest.TestCase):
    BANDS = {"dzmax_irrelevant": 0.3, "dzbar_irrelevant": 0.5}
    FLAT = {c: _d(0.0, -0.1, 0.1) for c in (-2.0, -0.5, 0.5, 2.0)}

    def test_irrelevant(self):
        self.assertEqual(AN.offset_label(_deltas(self.FLAT, self.FLAT), self.BANDS, False),
                         "OFFSET_IRRELEVANT")

    def test_not_determined_wins_over_everything(self):
        self.assertEqual(AN.offset_label(_deltas(self.FLAT, self.FLAT), self.BANDS, True),
                         "NOT_DETERMINED")

    def test_signed_pattern(self):
        zbar = {-2.0: _d(1.0, 0.6, 1.4), -0.5: _d(0.3, 0.1, 0.5),
                0.5: _d(-0.3, -0.5, -0.1), 2.0: _d(-1.0, -1.4, -0.6)}
        self.assertEqual(AN.offset_label(_deltas(self.FLAT, zbar), self.BANDS, False),
                         "OFFSET_SIGNED")

    def test_signed_pattern_needs_monotone_magnitude(self):
        zbar = {-2.0: _d(0.2, 0.1, 0.3), -0.5: _d(0.6, 0.4, 0.8),   # ±0.5 の方が大きい
                0.5: _d(-0.6, -0.8, -0.4), 2.0: _d(-0.2, -0.3, -0.1)}
        self.assertEqual(AN.offset_label(_deltas(self.FLAT, zbar), self.BANDS, False),
                         "OFFSET_OTHER")

    def test_both_signs_deeper_is_other(self):
        zbar = {c: _d(-1.0, -1.4, -0.6) for c in (-2.0, -0.5, 0.5, 2.0)}
        self.assertEqual(AN.offset_label(_deltas(self.FLAT, zbar), self.BANDS, False),
                         "OFFSET_OTHER")

    def test_a_single_ci_outside_the_band_but_containing_zero_is_inconclusive(self):
        zbar = dict(self.FLAT); zbar[2.0] = _d(-0.3, -0.7, 0.1)
        self.assertEqual(AN.offset_label(_deltas(self.FLAT, zbar), self.BANDS, False),
                         "INCONCLUSIVE")

    def test_zmax_alone_can_make_it_other(self):
        zmax = dict(self.FLAT); zmax[-0.5] = _d(0.5, 0.35, 0.65)
        self.assertEqual(AN.offset_label(_deltas(zmax, self.FLAT), self.BANDS, False),
                         "OFFSET_OTHER")

    def test_missing_arm_is_not_determined(self):
        d = _deltas(self.FLAT, self.FLAT); d.pop(2.0)
        self.assertEqual(AN.offset_label(d, self.BANDS, False), "NOT_DETERMINED")

    def test_band_edges_are_inclusive_and_a_hair_outside_breaks_irrelevant(self):
        zmax = dict(self.FLAT); zmax[0.5] = _d(0.2, 0.0, 0.3)
        self.assertEqual(AN.offset_label(_deltas(zmax, self.FLAT), self.BANDS, False),
                         "OFFSET_IRRELEVANT")
        zmax[0.5] = _d(0.2, 0.0, 0.3001)
        self.assertNotEqual(AN.offset_label(_deltas(zmax, self.FLAT), self.BANDS, False),
                            "OFFSET_IRRELEVANT")


class LabelBTests(unittest.TestCase):
    """追補 1 §4 のラダー B — 判定 c は ±2 の 2 本だけ・単調性は課さない・ラベルに _B が付く。"""

    BANDS = {"dzmax_irrelevant": 0.3, "dzbar_irrelevant": 0.5}

    def _d(self, zmax: dict, zbar: dict) -> dict:
        return {c: dict(dzmax=zmax[c], dzbar=zbar[c]) for c in (-2.0, 2.0)}

    def _label(self, zmax, zbar, nd=False):
        return AN.offset_label(self._d(zmax, zbar), self.BANDS, nd,
                               expect_c=AN.JUDGED_C_B, require_monotone=False, suffix="_B")

    FLAT = {c: _d(0.0, -0.1, 0.1) for c in (-2.0, 2.0)}

    def test_irrelevant_b(self):
        self.assertEqual(self._label(self.FLAT, self.FLAT), "OFFSET_IRRELEVANT_B")

    def test_signed_b_without_any_half_arm(self):
        zbar = {-2.0: _d(1.0, 0.6, 1.4), 2.0: _d(-1.0, -1.4, -0.6)}
        self.assertEqual(self._label(self.FLAT, zbar), "OFFSET_SIGNED_B")

    def test_both_signs_deeper_is_other_b(self):
        zbar = {c: _d(-1.0, -1.4, -0.6) for c in (-2.0, 2.0)}
        self.assertEqual(self._label(self.FLAT, zbar), "OFFSET_OTHER_B")

    def test_not_determined_and_inconclusive_carry_no_suffix(self):
        self.assertEqual(self._label(self.FLAT, self.FLAT, nd=True), "NOT_DETERMINED")
        zbar = dict(self.FLAT); zbar[2.0] = _d(-0.3, -0.7, 0.1)
        self.assertEqual(self._label(self.FLAT, zbar), "INCONCLUSIVE")

    def test_a_missing_judged_arm_is_not_determined(self):
        d = self._d(self.FLAT, self.FLAT); d.pop(2.0)
        self.assertEqual(AN.offset_label(d, self.BANDS, False, expect_c=AN.JUDGED_C_B,
                                         require_monotone=False, suffix="_B"),
                         "NOT_DETERMINED")

    def test_the_four_arm_deltas_are_rejected_by_ladder_B_expectations(self):
        """ラダー B に 4 本渡すと c の集合が合わず NOT_DETERMINED（取り違え防止）。"""
        four = {c: dict(dzmax=_d(0.0, -0.1, 0.1), dzbar=_d(0.0, -0.1, 0.1))
                for c in (-2.0, -0.5, 0.5, 2.0)}
        self.assertEqual(AN.offset_label(four, self.BANDS, False, expect_c=AN.JUDGED_C_B,
                                         require_monotone=False, suffix="_B"),
                         "NOT_DETERMINED")

    def test_monotonicity_is_required_in_A_but_not_in_B(self):
        """同じ ±2 の符号反転でも、ラダー A は ±0.5 との単調性で OTHER に落ちうる。"""
        zbar4 = {-2.0: _d(0.6, 0.4, 0.8), -0.5: _d(1.0, 0.8, 1.2),
                 0.5: _d(-1.0, -1.2, -0.8), 2.0: _d(-0.6, -0.8, -0.4)}
        flat4 = {c: _d(0.0, -0.1, 0.1) for c in (-2.0, -0.5, 0.5, 2.0)}
        self.assertEqual(AN.offset_label(_deltas(flat4, zbar4), self.BANDS, False),
                         "OFFSET_OTHER")
        zbar2 = {c: zbar4[c] for c in (-2.0, 2.0)}
        self.assertEqual(self._label(self.FLAT, zbar2), "OFFSET_SIGNED_B")

    def test_inside_the_band_irrelevant_wins_over_signed(self):
        """ラベルの優先順は spec §4 の並び順。**帯に内包されていれば符号が反対でも IRRELEVANT**
        （＝「動いたが無視できる大きさ」を SIGNED と呼ばない）。両ラダーで同じ。"""
        tiny = {-2.0: _d(0.2, 0.1, 0.3), 2.0: _d(-0.2, -0.3, -0.1)}
        self.assertEqual(self._label(self.FLAT, tiny), "OFFSET_IRRELEVANT_B")
        tiny4 = {-2.0: _d(0.2, 0.1, 0.3), -0.5: _d(0.1, 0.05, 0.15),
                 0.5: _d(-0.1, -0.15, -0.05), 2.0: _d(-0.2, -0.3, -0.1)}
        flat4 = {c: _d(0.0, -0.1, 0.1) for c in (-2.0, -0.5, 0.5, 2.0)}
        self.assertEqual(AN.offset_label(_deltas(flat4, tiny4), self.BANDS, False),
                         "OFFSET_IRRELEVANT")


# ---------------------------------------------------------------------------
# S-limit（30k 走行）＋ 短縮走行（spec §5）。3 腕 × 30k step・10 seed。
# ---------------------------------------------------------------------------
class ShortRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="act_offset_0906_test_"))
        cls.arms = cls.tmp / "arms"
        cls.ref = cls.tmp / "slimit"
        for arm in (P.S_LIMIT_ARM, P.S_LIMIT_MUTANT):
            P.run_short(arm, cls.arms, P.SHORT_STEPS, CFG_NEW)
        cls.slimit_cfg = P.augmented_config(cls.tmp, CFG_NEW)
        P.run_short(P.S_LIMIT_REF_ARM, cls.ref, P.SHORT_STEPS, cls.slimit_cfg)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_runner_module_state_is_restored_after_the_short_runs(self):
        self.assertEqual(E.CONFIG, CFG_EDGE)

    def test_s_limit_30k_state_hash_and_all_columns_match(self):
        r = P.compare_logs(self.arms, P.S_LIMIT_ARM, self.ref, P.S_LIMIT_REF_ARM)
        self.assertTrue(r["state_hash_all_equal"], r)
        self.assertEqual(r["n_mismatched_total"], 0, r)
        self.assertTrue(r["pass_"])
        self.assertGreater(min(row["n_columns"] for row in r["rows"]), 40)

    def test_s_limit_mutant_control_fails(self):
        r = P.compare_logs(self.arms, P.S_LIMIT_MUTANT, self.ref, P.S_LIMIT_REF_ARM)
        self.assertFalse(r["state_hash_all_equal"])
        self.assertGreater(r["n_mismatched_total"], 0)
        self.assertFalse(r["pass_"])

    def test_short_runs_are_finite_with_registered_lr_and_new_columns(self):
        for arm in (P.S_LIMIT_ARM, P.S_LIMIT_MUTANT):
            r = P.check_run(self.arms, arm, P.SHORT_STEPS)
            self.assertTrue(r["pass_"], r)

    def test_check_run_is_not_vacuous(self):
        bad = self.tmp / "bad"
        (bad / "logs").mkdir(parents=True)
        src = self.arms / "logs" / f"{P.S_LIMIT_ARM}_seed0.npz"
        with np.load(src, allow_pickle=True) as z:
            payload = {k: z[k] for k in z.files}
        zb = payload["layer1_zbar"].copy(); zb[10, 3] = np.nan; payload["layer1_zbar"] = zb
        np.savez_compressed(bad / "logs" / f"{P.S_LIMIT_ARM}_seed0.npz", **payload)
        r = P.check_run(bad, P.S_LIMIT_ARM, P.SHORT_STEPS, seeds=[0])
        self.assertFalse(r["pass_"])
        self.assertIn("layer1_zbar", r["rows"][0]["nonfinite"])
        payload["layer1_zbar"] = zb.copy(); payload["layer1_zbar"][10, 3] = 0.0
        payload["lr_used"] = np.array(0.005)
        np.savez_compressed(bad / "logs" / f"{P.S_LIMIT_ARM}_seed0.npz", **payload)
        self.assertFalse(P.check_run(bad, P.S_LIMIT_ARM, P.SHORT_STEPS, seeds=[0])["pass_"])

    def test_compare_logs_is_not_vacuous(self):
        bad = self.tmp / "bad2"
        (bad / "logs").mkdir(parents=True)
        src = self.arms / "logs" / f"{P.S_LIMIT_ARM}_seed0.npz"
        with np.load(src, allow_pickle=True) as z:
            payload = {k: z[k] for k in z.files}
        v = payload["layer1_v_unit"].copy(); v[-1, 0] = np.nextafter(v[-1, 0], np.float32(1e9))
        payload["layer1_v_unit"] = v
        np.savez_compressed(bad / "logs" / f"{P.S_LIMIT_ARM}_seed0.npz", **payload)
        r = P.compare_logs(self.arms, P.S_LIMIT_ARM, bad, P.S_LIMIT_ARM, seeds=[0])
        self.assertFalse(r["pass_"])
        self.assertIn("layer1_v_unit", r["rows"][0]["mismatched"])


if __name__ == "__main__":
    unittest.main()
