"""edge_law_0905 の活性化の単体テスト（spec `specs/spec_edge_law_0905.md` §5）。

    cd <repo> && OMP_NUM_THREADS=1 PYTHONPATH=. python3 -m unittest \
        src.test_edge_law_nets_0905 -v

様式は `src/test_weird_act_0903.py` に倣う（unittest・格子は float64 の
`linspace(-30, 30, 24001)`）。この檻で捕まえるのは §5 の 7 検査:

  S-limit        `shelf_leaky_d0` が leaky と**バイト**一致（既知差 `pre=-0.0` を除く）
  S-flip         `flip_leaky(z) == -leaky(-z)` / `flip'(z) == leaky'(-z)` がバイト一致
  S-fd           新族が自分の forward の真の導関数（float64 中心差分・折れ目除外）
  S-curv         `act_curv` が `act_grad` の真の導関数・区分線形族は恒等的に 0
  S-fallthrough  11 名 × `act_fn`/`act_grad`/`act_curv` の三方向が ELU 分岐と違う
  S-const        `SHELF_DEPTH`/`STEEP_DEPTH`/`STEEP_SLOPE`/`SOFTPLUS_BETA`/`TANH_BETA`
                 が config の第 2 母数と一致
  S-guard        11 名が `ACTIVATIONS` と各ガードタプルの**両方**に入っている

**空虚な検査を置かない**（本プロジェクトの過去の失敗）: バイト一致の検査には
必ず「変異させると FAIL する」対を隣に置く。`torch.equal` は符号盲なので
`±0.0` が問題になるところでは `arr.tobytes()` で比べる。
"""
from __future__ import annotations

import math
import unittest
from pathlib import Path

import torch

from src.common import ROOT, load_config
from src.nets import VecMLPL

CFG_PATH = Path(ROOT) / "configs" / "edge_law_0905.yaml"
CFG = load_config(str(CFG_PATH))

GRID = torch.linspace(-30, 30, 24001, dtype=torch.float64)
# 新 11 名（spec §3.2 の順）。
NEW_ACTS = VecMLPL.EDGE_LAW_ACTIVATIONS
SHELF_ACTS = tuple(VecMLPL.SHELF_DEPTH)
STEEP_ACTS = tuple(VecMLPL.STEEP_DEPTH)
SMOOTH_ACTS = ("softplus_b", "tanh_b")
# leaky 側の傾き。腕表の dial（`LR_1216` と同じ 0.1）に合わせる。
A_SLOPE = 0.1


def _net(act: str, alpha: float) -> VecMLPL:
    net = VecMLPL(1, [2], 2, torch.Generator().manual_seed(0), "cpu")
    return net.set_activation(act, alpha, "alpha_exp")


def _alpha_for(act: str) -> float:
    """`UNIT_ALPHA_ACTIVATIONS` は 1.0 固定、傾き族は腕表の 0.1。"""
    return 1.0 if act in VecMLPL.UNIT_ALPHA_ACTIVATIONS else A_SLOPE


def _bytes(t: torch.Tensor) -> bytes:
    """`torch.equal` は符号盲なので ±0.0 が効く比較はバイトで見る。"""
    return t.detach().contiguous().numpy().tobytes()


def _kinks(act: str):
    """折れ目（S-fd / S-curv で ±1e-3 を外す点）。"""
    if act == "flip_leaky":
        return (0.0,)
    if act in VecMLPL.SHELF_DEPTH:
        return (0.0 - VecMLPL.SHELF_DEPTH[act],)
    if act in VecMLPL.STEEP_DEPTH:
        return (0.0 - VecMLPL.STEEP_DEPTH[act],)
    if act == "elu":
        return (0.0,)
    if act == "softplus_b":
        # 数式には折れ目が無いが、`F.softplus` は `beta*x > threshold`(=20) で
        # **実装が x に切り替わる**。log1p(exp(20)) = 20 + 2.06e-9 なので
        # forward に 2.1e-9 の段差があり、h=1e-6 の中心差分では 1e-3 に化ける。
        # 段差の大きさそのものは下の test が登録する（隠さない）。
        return (20.0,)
    return ()


def _extra_points() -> torch.Tensor:
    """S-limit / S-flip の明示追加点（spec §5）。

    `linspace(-30, 30, 24001)` の中央は −6.245e−16 で、**0 も折れ目も格子に
    入っていない**。ここを外すと「棚が leaky に退化していない」を見逃す。
    重複は落とさない（`set` は 0.0 と −0.0 を同一視して −0.0 を消す）。
    """
    pts = [0.0, -0.0]
    depths = list(VecMLPL.SHELF_DEPTH.values()) + list(VecMLPL.STEEP_DEPTH.values())
    for d in depths:
        for v in (d, math.nextafter(d, math.inf), math.nextafter(d, -math.inf)):
            pts.append(v)
            pts.append(0.0 - v)
    for v in (5e-324, 1e-38):
        pts.append(v)
        pts.append(0.0 - v)
    return torch.tensor(pts, dtype=torch.float64)


EXTRA = _extra_points()
ALL_POINTS = torch.cat([GRID, EXTRA])
NEG_ZERO = (ALL_POINTS == 0.0) & torch.signbit(ALL_POINTS)


class SLimitTests(unittest.TestCase):
    """S-limit: `shelf_leaky_d0` は leaky と bit 一致（`pre=-0.0` は既知差）。"""

    def test_extra_points_actually_contain_the_special_values(self):
        """検査が空虚でないこと: 追加点に ±0.0・折れ目・非正規数が入っている。"""
        self.assertGreaterEqual(int(NEG_ZERO.sum()), 1)          # −0.0 が本当にある
        self.assertIn(0.0, EXTRA.tolist())
        for d in set(list(VecMLPL.SHELF_DEPTH.values())
                     + list(VecMLPL.STEEP_DEPTH.values())):
            self.assertIn(0.0 - d, EXTRA.tolist())
        self.assertEqual(float(EXTRA.abs().max()), math.nextafter(30.0, math.inf))
        self.assertGreater(int((EXTRA.abs() == 5e-324).sum()), 0)
        # 格子には 0 も折れ目も入っていない（追加点が要る理由）
        self.assertEqual(int((GRID == 0.0).sum()), 0)

    def test_shelf_d0_is_byte_identical_to_leaky_except_at_negative_zero(self):
        z = ALL_POINTS[~NEG_ZERO]
        sh, lk = _net("shelf_leaky_d0", A_SLOPE), _net("leaky_relu", A_SLOPE)
        fs, fl = sh.act_fn(z), lk.act_fn(z)
        self.assertEqual(_bytes(fs), _bytes(fl))
        self.assertEqual(_bytes(sh.act_grad(z, fs)), _bytes(lk.act_grad(z, fl)))

    def test_negative_zero_is_the_registered_known_difference(self):
        """`pre=-0.0`: 棚は `+0.0`・leaky は `-0.0`（forward だけ・backward は一致）。"""
        z = torch.tensor([-0.0], dtype=torch.float64)
        self.assertTrue(bool(torch.signbit(z)))
        sh, lk = _net("shelf_leaky_d0", A_SLOPE), _net("leaky_relu", A_SLOPE)
        fs, fl = sh.act_fn(z), lk.act_fn(z)
        self.assertEqual(float(fs), 0.0)
        self.assertEqual(float(fl), 0.0)
        self.assertFalse(bool(torch.signbit(fs)))     # 棚は +0.0
        self.assertTrue(bool(torch.signbit(fl)))      # leaky は −0.0
        self.assertNotEqual(_bytes(fs), _bytes(fl))   # 既知差はバイトでは違う
        self.assertTrue(torch.equal(fs, fl))          # torch.equal は符号盲（証拠）
        self.assertEqual(_bytes(sh.act_grad(z, fs)), _bytes(lk.act_grad(z, fl)))

    def test_a_nonzero_shelf_depth_must_FAIL_the_same_comparison(self):
        """変異体（d>0）は同じ比較で落ちる ＝ S-limit が空虚でない証拠。"""
        z = ALL_POINTS[~NEG_ZERO]
        lk = _net("leaky_relu", A_SLOPE)
        fl = lk.act_fn(z)
        for act in SHELF_ACTS:
            if VecMLPL.SHELF_DEPTH[act] == 0.0:
                continue
            with self.subTest(act=act):
                sh = _net(act, A_SLOPE)
                fs = sh.act_fn(z)
                self.assertNotEqual(_bytes(fs), _bytes(fl))
                self.assertNotEqual(_bytes(sh.act_grad(z, fs)),
                                    _bytes(lk.act_grad(z, fl)))

    def test_shelf_is_exactly_continuous_at_the_kink(self):
        """折れ目ちょうどで恒等枝と厳密に連続（`a*pre+(a-1)*d` 形なら落ちる）。"""
        for act, d in VecMLPL.SHELF_DEPTH.items():
            with self.subTest(act=act):
                n = _net(act, A_SLOPE)
                z = torch.tensor([0.0 - d], dtype=torch.float64)
                self.assertEqual(float(n.act_fn(z)), 0.0 - d)
        for act, d in VecMLPL.STEEP_DEPTH.items():
            with self.subTest(act=act):
                n = _net(act, 1.0)
                z = torch.tensor([0.0 - d], dtype=torch.float64)
                self.assertEqual(float(n.act_fn(z)), 0.0 - d)

    def test_steep_shelf_lower_slope_is_two_and_not_act_alpha(self):
        for act, d in VecMLPL.STEEP_DEPTH.items():
            with self.subTest(act=act):
                n = _net(act, 1.0)
                z = torch.tensor([0.0 - d - 1.0], dtype=torch.float64)
                self.assertEqual(float(n.act_fn(z)),
                                 VecMLPL.STEEP_SLOPE * (-1.0) - d)
                self.assertEqual(float(n.act_grad(z, n.act_fn(z))),
                                 VecMLPL.STEEP_SLOPE)
                self.assertNotEqual(float(n.act_grad(z, n.act_fn(z))),
                                    n.act_alpha)


class SFlipTests(unittest.TestCase):
    """S-flip: `flip_leaky` は leaky の奇鏡像（述語が `<0` でないと ±0.0 で落ちる）。"""

    def test_flip_forward_is_the_byte_exact_odd_mirror_of_leaky(self):
        z = ALL_POINTS
        fl = _net("flip_leaky", A_SLOPE)
        lk = _net("leaky_relu", A_SLOPE)
        self.assertEqual(_bytes(fl.act_fn(z)), _bytes(-(lk.act_fn(-z))))

    def test_flip_backward_equals_leaky_backward_at_minus_z(self):
        z = ALL_POINTS
        fl = _net("flip_leaky", A_SLOPE)
        lk = _net("leaky_relu", A_SLOPE)
        got = fl.act_grad(z, fl.act_fn(z))
        want = lk.act_grad(-z, lk.act_fn(-z))
        self.assertEqual(_bytes(got), _bytes(want))

    def test_the_rejected_gt0_predicate_must_FAIL_at_plus_minus_zero(self):
        """`>0` 形（棄却した書き方）は ±0.0 で鏡像要請と食い違う ＝ 検査は空虚でない。"""
        z = torch.tensor([0.0, -0.0], dtype=torch.float64)
        lk = _net("leaky_relu", A_SLOPE)
        want = lk.act_grad(-z, lk.act_fn(-z))                    # 鏡像要請 = a
        # 棄却形（枝を入れ替えて `>0` で書いたもの）: ±0.0 が恒等枝に落ち phi'=1
        bad = torch.where(z > 0, torch.full_like(z, A_SLOPE),
                          torch.ones_like(z))
        good = torch.where(z < 0, torch.ones_like(z),
                           torch.full_like(z, A_SLOPE))          # 実装形
        self.assertNotEqual(_bytes(bad), _bytes(want))
        self.assertEqual(_bytes(good), _bytes(want))

    def test_flip_is_not_the_same_function_as_mirror_leaky(self):
        """既存の `mirror_leaky`（V 字）とは別物。"""
        z = GRID
        a, b = _net("flip_leaky", A_SLOPE), _net("mirror_leaky", A_SLOPE)
        self.assertFalse(torch.allclose(a.act_fn(z), b.act_fn(z)))


class SFdTests(unittest.TestCase):
    """S-fd: 新 11 族は自分の forward の真の導関数（折れ目 ±1e−3 除外・追加点も除外）。"""

    H = 1e-6
    TOL = 1e-6

    def _fd_grid(self, act):
        """格子 ＋ 折れ目まわりの局所格子（`shelf_leaky_d30` の下側枝を測るため）。"""
        pieces = [GRID]
        for k in _kinks(act):
            pieces.append(torch.linspace(k - 5.0, k + 5.0, 2001, dtype=torch.float64))
        z = torch.cat(pieces)
        mask = torch.ones_like(z, dtype=torch.bool)
        for k in _kinks(act):
            mask &= (z - k).abs() > 1e-3
        return z[mask]

    def test_central_difference_matches_the_closed_form_backward(self):
        for act in NEW_ACTS:
            with self.subTest(act=act):
                n = _net(act, _alpha_for(act))
                z = self._fd_grid(act)
                fd = (n.act_fn(z + self.H) - n.act_fn(z - self.H)) / (2 * self.H)
                g = n.act_grad(z, n.act_fn(z))
                self.assertLess(float((fd - g).abs().max()), self.TOL)

    def test_the_local_grid_really_probes_both_branches(self):
        """d=30 の下側枝が本当に測られていること（空虚でない証拠）。"""
        for act in SHELF_ACTS + STEEP_ACTS:
            with self.subTest(act=act):
                z = self._fd_grid(act)
                d = (VecMLPL.SHELF_DEPTH.get(act)
                     if act in VecMLPL.SHELF_DEPTH else VecMLPL.STEEP_DEPTH[act])
                self.assertGreater(int((z < 0.0 - d).sum()), 100)
                self.assertGreater(int((z > 0.0 - d).sum()), 100)

    def test_the_softplus_threshold_step_is_registered_not_hidden(self):
        """`F.softplus` は `beta*x > 20` で実装が x に切り替わる（式の折れ目ではない）。

        除外した 1 点で何が起きているかを数値で残す: forward の段差は
        `log1p(exp(20)) - 20 = 2.06e-9` で、`sigmoid(20)` も 1 から 2.06e-9 しか
        離れていない。学習は float32 なので、この差は表現できる桁より下。
        """
        n = _net("softplus_b", 1.0)
        z = torch.tensor([20.0, math.nextafter(20.0, math.inf)],
                         dtype=torch.float64)
        f = n.act_fn(z)
        step = float(f[0] - 20.0)
        self.assertGreater(step, 0.0)
        self.assertLess(step, 3e-9)
        self.assertEqual(float(f[1]), math.nextafter(20.0, math.inf))
        self.assertLess(abs(float(n.act_grad(z, f)[0]) - 1.0), 3e-9)
        # 折れ目除外がこの 1 点だけであること（他は 1e-6 で通る）
        zz = GRID[(GRID - 20.0).abs() > 1e-3]
        fd = ((n.act_fn(zz + 1e-6) - n.act_fn(zz - 1e-6)) / 2e-6)
        self.assertLess(float((fd - n.act_grad(zz, n.act_fn(zz))).abs().max()), 1e-6)


class SCurvTests(unittest.TestCase):
    """S-curv: `act_curv` が `act_grad` の真の導関数・区分線形族は恒等的に 0。"""

    H = 1e-5
    TOL = 1e-6

    def test_act_curv_is_the_derivative_of_act_grad_for_the_smooth_families(self):
        for act, alpha in (("softplus_b", 1.0), ("tanh_b", 1.0),
                           ("elu", 1.0), ("elu", 0.5)):
            with self.subTest(act=act, alpha=alpha):
                n = _net(act, alpha)
                z = GRID
                mask = torch.ones_like(z, dtype=torch.bool)
                for k in _kinks(act):
                    mask &= (z - k).abs() > 1e-3
                z = z[mask]
                fd = ((n.act_grad(z + self.H, n.act_fn(z + self.H))
                       - n.act_grad(z - self.H, n.act_fn(z - self.H)))
                      / (2 * self.H))
                self.assertLess(float((fd - n.act_curv(z)).abs().max()), self.TOL)

    def test_piecewise_linear_curvature_is_identically_plus_zero(self):
        for act in ("relu", "leaky_relu") + ("flip_leaky",) + SHELF_ACTS + STEEP_ACTS:
            with self.subTest(act=act):
                n = _net(act, _alpha_for(act) if act in NEW_ACTS else A_SLOPE)
                c = n.act_curv(ALL_POINTS)
                self.assertTrue(torch.equal(c, torch.zeros_like(ALL_POINTS)))
                self.assertEqual(int(torch.signbit(c).sum()), 0)   # −0.0 を作らない

    def test_linear_is_leaky_with_alpha_one_and_has_zero_curvature(self):
        """線形は既存流儀どおり leaky の dial=1.0（新規実装は無い・§3.2 末尾）。"""
        lin = _net("leaky_relu", 1.0)
        self.assertTrue(torch.equal(lin.act_fn(GRID), GRID))
        self.assertTrue(torch.equal(lin.act_grad(GRID, lin.act_fn(GRID)),
                                    torch.ones_like(GRID)))
        self.assertTrue(torch.equal(lin.act_curv(GRID), torch.zeros_like(GRID)))

    def test_act_curv_raises_for_every_unregistered_name(self):
        """未登録名は ELU に落ちず `NotImplementedError`（`m_dphiddphi` の保険）。"""
        for act, alpha in (("silu", 1.0), ("gelu", 1.0), ("snake", 1.0),
                           ("bwd_leaky", 0.1), ("fwd_leaky", 0.1),
                           ("band_leaky_d1", 0.1), ("mirror_leaky", 0.1)):
            with self.subTest(act=act):
                n = _net(act, alpha)
                with self.assertRaises(NotImplementedError):
                    n.act_curv(GRID)

    def test_softplus_and_tanh_curvature_signs(self):
        """`softplus_b` は phi''>0 一定符号（命題 2 の前件）・`tanh_b` は奇。"""
        sp, th = _net("softplus_b", 1.0), _net("tanh_b", 1.0)
        z = torch.linspace(-10, 10, 2001, dtype=torch.float64)
        self.assertTrue(bool((sp.act_curv(z) > 0).all()))
        self.assertTrue(torch.allclose(th.act_curv(z), -th.act_curv(-z),
                                       atol=1e-15))


class SFallthroughTests(unittest.TestCase):
    """S-fallthrough: 11 名 × 三方向が、同じ alpha の ELU 分岐と一致しないこと。"""

    def test_no_new_activation_silently_falls_through_to_elu(self):
        for act in NEW_ACTS:
            with self.subTest(act=act):
                alpha = _alpha_for(act)
                n = _net(act, alpha)
                e = _net("elu", alpha)
                fe = e.act_fn(GRID)
                self.assertFalse(torch.allclose(n.act_fn(GRID), fe))
                self.assertFalse(torch.allclose(n.act_grad(GRID, n.act_fn(GRID)),
                                                e.act_grad(GRID, fe)))
                try:
                    curv = n.act_curv(GRID)
                except NotImplementedError:
                    continue                      # 例外も「ELU ではない」の証明
                self.assertFalse(torch.allclose(curv, e.act_curv(GRID)))

    def test_the_elu_reference_itself_is_not_degenerate(self):
        """比較相手の ELU が定数や 0 でないこと（空虚な `assertFalse` 対策）。"""
        for alpha in (A_SLOPE, 1.0):
            e = _net("elu", alpha)
            fe = e.act_fn(GRID)
            self.assertGreater(float(fe.std()), 1.0)
            self.assertGreater(float(e.act_grad(GRID, fe).std()), 0.0)
            self.assertGreater(float(e.act_curv(GRID).abs().max()), 0.0)


class SConstTests(unittest.TestCase):
    """S-const: クラス定数と config の第 2 母数の突き合わせ（`_s_const` の流儀）。"""

    def test_shelf_depths_match_the_config(self):
        cfg_names = {k for k, v in CFG["activation"].items()
                     if isinstance(v, dict) and k.startswith("shelf_leaky_")}
        self.assertEqual(cfg_names, set(VecMLPL.SHELF_DEPTH))
        for name, d in VecMLPL.SHELF_DEPTH.items():
            with self.subTest(act=name):
                self.assertEqual(float(CFG["activation"][name]["depth"]), d)

    def test_steep_depths_and_slope_match_the_config(self):
        cfg_names = {k for k, v in CFG["activation"].items()
                     if isinstance(v, dict) and k.startswith("steep_shelf_")}
        self.assertEqual(cfg_names, set(VecMLPL.STEEP_DEPTH))
        for name, d in VecMLPL.STEEP_DEPTH.items():
            with self.subTest(act=name):
                self.assertEqual(float(CFG["activation"][name]["depth"]), d)
                self.assertEqual(float(CFG["activation"][name]["lower_slope"]),
                                 VecMLPL.STEEP_SLOPE)

    def test_smooth_betas_match_the_config(self):
        self.assertEqual(float(CFG["activation"]["softplus_b"]["beta"]),
                         VecMLPL.SOFTPLUS_BETA)
        self.assertEqual(float(CFG["activation"]["tanh_b"]["beta"]),
                         VecMLPL.TANH_BETA)
        self.assertEqual(VecMLPL.SOFTPLUS_BETA, 1.0)
        self.assertEqual(VecMLPL.TANH_BETA, 1.0)

    def test_every_registered_name_appears_in_the_config_activation_map(self):
        for act in NEW_ACTS:
            with self.subTest(act=act):
                self.assertEqual(CFG["activation"][act]["name"], act)

    def test_config_marks_d0_as_s_limit_only_and_requires_act_curv(self):
        self.assertTrue(CFG["activation"]["shelf_leaky_d0_is_s_limit_only"])
        self.assertTrue(CFG["activation"]["act_curv_required_for_all"])
        self.assertEqual(VecMLPL.SHELF_DEPTH["shelf_leaky_d0"], 0.0)

    def test_dial_must_be_one_exactly_for_the_arms_the_config_says_so(self):
        for act in NEW_ACTS:
            entry = CFG["activation"][act]
            with self.subTest(act=act):
                if "dial_must_be" in entry:
                    self.assertIn(act, VecMLPL.UNIT_ALPHA_ACTIVATIONS)
                    self.assertEqual(float(entry["dial_must_be"]), 1.0)
                else:
                    self.assertNotIn(act, VecMLPL.UNIT_ALPHA_ACTIVATIONS)

    def test_the_config_arm_table_only_uses_registered_names_and_dials(self):
        for arm in CFG["arms"]:
            with self.subTest(arm=arm["name"]):
                self.assertIn(arm["activation"], VecMLPL.ACTIVATIONS)
                _net(arm["activation"], float(arm["dial"]))   # 域ガードを通す


class SGuardTests(unittest.TestCase):
    """S-guard: 名前は `ACTIVATIONS` と**ガードタプルの両方**に入れる。"""

    def test_all_eleven_names_are_in_ACTIVATIONS(self):
        self.assertEqual(len(NEW_ACTS), 11)
        self.assertEqual(len(set(NEW_ACTS)), 11)
        for act in NEW_ACTS:
            self.assertIn(act, VecMLPL.ACTIVATIONS)

    def test_unknown_name_is_still_rejected(self):
        with self.assertRaises(ValueError):
            _net("shelf_leaky_d4", 0.1)

    def test_slope_families_are_in_WEIRD_SLOPE_ACTIVATIONS_and_guarded(self):
        for act in ("flip_leaky",) + SHELF_ACTS:
            with self.subTest(act=act):
                self.assertIn(act, VecMLPL.WEIRD_SLOPE_ACTIVATIONS)
                _net(act, 0.0)
                _net(act, 1.0)
                with self.assertRaises(ValueError):
                    _net(act, 1.5)
                with self.assertRaises(ValueError):
                    _net(act, -0.1)

    def test_unit_alpha_families_reject_every_dial_but_one(self):
        self.assertEqual(set(VecMLPL.UNIT_ALPHA_ACTIVATIONS),
                         set(STEEP_ACTS) | set(SMOOTH_ACTS))
        for act in VecMLPL.UNIT_ALPHA_ACTIVATIONS:
            with self.subTest(act=act):
                _net(act, 1.0)                       # 登録 dial は通る
                for bad in (0.1, 0.5, 2.0, 0.0):
                    with self.assertRaises(ValueError):
                        _net(act, bad)

    def test_slope_and_unit_alpha_families_do_not_overlap(self):
        self.assertEqual(set(VecMLPL.WEIRD_SLOPE_ACTIVATIONS)
                         & set(VecMLPL.UNIT_ALPHA_ACTIVATIONS), set())

    def test_second_parameter_tables_cover_every_registered_name(self):
        for act in VecMLPL.SHELF_DEPTH:
            self.assertIn(act, VecMLPL.ACTIVATIONS)
        for act in VecMLPL.STEEP_DEPTH:
            self.assertIn(act, VecMLPL.ACTIVATIONS)
        self.assertEqual(set(VecMLPL.SHELF_DEPTH) | set(VecMLPL.STEEP_DEPTH)
                         | set(SMOOTH_ACTS) | {"flip_leaky"}, set(NEW_ACTS))

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


class ExistingPathTests(unittest.TestCase):
    """既存の活性化経路が 1 バイトも動いていないこと（`weird_act_0903` の流儀）。"""

    INV_SQRT2 = 1.0 / math.sqrt(2.0)
    INV_SQRT2PI = 1.0 / math.sqrt(2.0 * math.pi)

    def _same(self, got, want):
        self.assertTrue(torch.equal(got, want))
        # torch.equal は符号盲なので ±0.0 の混入を signbit で見る
        self.assertEqual(int(torch.signbit(got).sum()),
                         int(torch.signbit(want).sum()))

    def test_relu_forward_and_backward_unchanged(self):
        n = _net("relu", 1.0)
        self._same(n.act_fn(GRID), torch.relu(GRID))
        self._same(n.act_grad(GRID, n.act_fn(GRID)),
                   (GRID > 0).to(GRID.dtype))

    def test_leaky_forward_and_backward_unchanged(self):
        n = _net("leaky_relu", 0.1)
        self._same(n.act_fn(GRID), torch.where(GRID > 0, GRID, 0.1 * GRID))
        self._same(n.act_grad(GRID, n.act_fn(GRID)),
                   torch.where(GRID > 0, torch.ones_like(GRID),
                               torch.full_like(GRID, 0.1)))

    def test_elu_forward_and_backward_unchanged(self):
        n = _net("elu", 1.0)
        self._same(n.act_fn(GRID), torch.where(GRID > 0, GRID, torch.expm1(GRID)))
        self._same(n.act_grad(GRID, n.act_fn(GRID)),
                   torch.where(GRID > 0, torch.ones_like(GRID),
                               1.0 * torch.exp(GRID)))

    def test_silu_forward_and_backward_unchanged(self):
        n = _net("silu", 1.0)
        s = torch.sigmoid(1.0 * GRID)
        self._same(n.act_fn(GRID), GRID * torch.sigmoid(1.0 * GRID))
        self._same(n.act_grad(GRID, n.act_fn(GRID)),
                   s * (1.0 + 1.0 * GRID * (1.0 - s)))

    def test_gelu_forward_and_backward_unchanged(self):
        n = _net("gelu", 1.0)
        t = 1.0 * GRID
        cdf = 0.5 * (1.0 + torch.erf(t * self.INV_SQRT2))
        pdf = torch.exp(-0.5 * t * t) * self.INV_SQRT2PI
        self._same(n.act_fn(GRID), GRID * cdf)
        self._same(n.act_grad(GRID, n.act_fn(GRID)), cdf + t * pdf)

    def test_snake_forward_and_backward_unchanged(self):
        n = _net("snake", 1.0)
        self._same(n.act_fn(GRID),
                   GRID + torch.sin(1.0 * GRID) ** 2 / 1.0)
        self._same(n.act_grad(GRID, n.act_fn(GRID)),
                   1.0 + torch.sin(2.0 * 1.0 * GRID))

    def test_bwd_leaky_surrogate_pair_unchanged(self):
        bl = _net("bwd_leaky", 0.1)
        self._same(bl.act_fn(GRID), torch.relu(GRID))
        self._same(bl.act_grad(GRID, bl.act_fn(GRID)),
                   torch.where(GRID > 0, torch.ones_like(GRID),
                               torch.full_like(GRID, 0.1)))
        fl = _net("fwd_leaky", 0.1)
        self._same(fl.act_fn(GRID), torch.where(GRID > 0, GRID, 0.1 * GRID))
        self._same(fl.act_grad(GRID, fl.act_fn(GRID)), (GRID > 0).to(GRID.dtype))

    def test_the_reference_formulas_are_not_trivially_equal_to_each_other(self):
        """参照式どうしが違うこと（自分自身と比べる空虚な検査でない証拠）。"""
        r, lk, e = _net("relu", 1.0), _net("leaky_relu", 0.1), _net("elu", 1.0)
        self.assertFalse(torch.equal(r.act_fn(GRID), lk.act_fn(GRID)))
        self.assertFalse(torch.equal(lk.act_fn(GRID), e.act_fn(GRID)))


if __name__ == "__main__":
    unittest.main()
