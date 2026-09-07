"""boundary_dense_0907 の検査（spec `specs/spec_boundary_dense_0907.md` §7）。

    OMP_NUM_THREADS=1 PYTHONPATH=. python3 -m unittest src.test_boundary_dense_0907 -v

S-grid（記録格子）/ S-col（列を足すだけ・既存列を消していない）/ S-id（N = J+B+A の恒等式）
/ S-cfg。実走を要する S-null（`LRoff0_1216` と state_hash 一致）は前検査
`src.boundary_dense_preflight_0907` で実測する。**bit 一致の検査には変異体の対を置く。**
"""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch

from src import boundary_dense_0907 as B
from src import edge_law_0905 as E
from src.common import ROOT, load_config

CFG = Path(ROOT) / "configs" / "boundary_dense_0907.yaml"
ARMS = ["BDref_1216", "BDa0p5_1216", "BDoffm0p5_1216", "BDvf1_1216", "BDFB_1216"]


class TestGrid(unittest.TestCase):
    def test_contains_all_task_ends(self):
        g = B.grid_of(load_config(str(CFG)))
        s = set(B.probe_steps(2_000_000, g["window"], g["offsets"]))
        self.assertTrue(set(range(0, 2_000_001, 10_000)) <= s)
        self.assertIn(0, s)
        self.assertIn(2_000_000, s)

    def test_dense_offsets_inside_window_only(self):
        g = B.grid_of(load_config(str(CFG)))
        s = set(B.probe_steps(2_000_000, g["window"], g["offsets"]))
        for d in g["offsets"]:                       # 窓の中の境界には全部ある
            self.assertIn(100_000 + d, s)
            self.assertIn(1_000_000 + d, s)
            self.assertIn(1_990_000 + d, s)
        for d in g["offsets"]:                       # 窓の外の境界には無い
            self.assertNotIn(90_000 + d, s)
            self.assertNotIn(50_000 + d, s)

    def test_sorted_unique_and_size(self):
        g = B.grid_of(load_config(str(CFG)))
        p = B.probe_steps(2_000_000, g["window"], g["offsets"])
        self.assertEqual(p, sorted(set(p)))
        self.assertEqual(len(p), 201 + 190 * len(g["offsets"]))

    def test_short_run_window_collapses(self):
        """短縮走行では窓を畳んで、境界を 1 つだけ密にする（前検査で格子を通す）。"""
        p = B.probe_steps(30_000, [20_000, 20_000], [1, 2, 5])
        self.assertEqual(p, [0, 10_000, 20_000, 20_001, 20_002, 20_005, 30_000])


class TestRecorderColumns(unittest.TestCase):
    def _st(self):
        cfg = E.build_cfg(CFG)
        c = dict(cfg)
        c["common"] = dict(cfg["common"])
        c["common"]["seeds"] = [0, 1]
        return E.setup_arm_dial(c, E._arm(c, "BDref_1216"), "cpu")

    def test_adds_only_b(self):
        st = self._st()
        steps = [0, 10_000]
        base = E.EdgeRecorder(list(steps), st)
        new = B.BoundaryRecorder(list(steps), st)
        self.assertEqual(set(new.unit) - set(base.unit), {"b"})
        self.assertEqual(set(base.unit) - set(new.unit), set())
        self.assertEqual(new.unit["b"].shape,
                         (len(steps), st["R"], st["hidden"][0]))

    def test_b_matches_net_bias(self):
        st = self._st()
        rec = B.BoundaryRecorder([0], st)
        rec(st, 0)
        want = st["net"].bs[0].detach().cpu().numpy().astype(np.float32)
        self.assertEqual(rec.unit["b"][0].tobytes(), want.tobytes())
        # 対（変異体）: 別の量を入れたら一致しない
        self.assertNotEqual(rec.unit["b"][0].tobytes(),
                            (want + np.float32(1e-3)).tobytes())

    def test_recorder_does_not_touch_env_or_rng(self):
        """probe を余分に打っても env の状態も生成器も動かない（この実験の生命線）。"""
        st = self._st()
        env = st["env"]
        before = (env.flip_state.clone(), int(getattr(env, "t", 0)),
                  env.gen.get_state().clone() if hasattr(env, "gen") else None)
        rec = B.BoundaryRecorder([0, 1, 2, 5], st)
        for s in (0, 1, 2, 5):          # 記録器は同じ step の二度打ちを拒む（S-dup）
            rec(st, s)
        with self.assertRaises(RuntimeError):
            rec(st, 0)
        self.assertTrue(torch.equal(env.flip_state, before[0]))
        self.assertEqual(int(getattr(env, "t", 0)), before[1])
        if before[2] is not None:
            self.assertTrue(torch.equal(env.gen.get_state(), before[2]))


class TestIdentity(unittest.TestCase):
    def test_j_b_a_sums_to_net(self):
        """N = J + B + A は定義上の恒等式。float64 で 1e-12 まで合うこと。"""
        rng = np.random.default_rng(0)
        zb0, zb1, b0, b1 = (rng.normal(size=(4, 7)) for _ in range(4))
        zbm, bm = rng.normal(size=(4, 7)), rng.normal(size=(4, 7))   # t0+1 の値
        J = (zbm - bm) - (zb0 - b0)
        Bt = b1 - b0
        A = (zb1 - b1) - (zbm - bm)
        N = zb1 - zb0
        self.assertLess(float(np.abs(J + Bt + A - N).max()), 1e-12)


class TestConfig(unittest.TestCase):
    def test_arm_table(self):
        cfg = load_config(str(CFG))
        self.assertEqual([r["name"] for r in cfg["arms"]], ARMS)
        for r in cfg["arms"]:
            self.assertEqual(int(r["total_steps"]), 2_000_000)
            self.assertEqual([int(v) for v in r["checkpoints"]], [0, 1_000_000, 2_000_000])
        self.assertEqual(cfg["record"]["new_unit_columns"], ["b"])

    def test_analysis_block(self):
        an = load_config(str(CFG))["analysis"]
        self.assertEqual(an["judged_arm"], "BDref_1216")
        self.assertEqual(an["s_null_pair"]["BDref_1216"], "LRoff0_1216")
        self.assertGreater(float(an["floor"]["net_per_task"]), 0.0)
        self.assertLess(float(an["labels"]["geometry_carried_max"]),
                        float(an["labels"]["bias_carried_min"]))

    def test_build_cfg_and_hooks(self):
        cfg = E.build_cfg(CFG)
        names = [r["name"] for r in cfg["arms"]]
        self.assertEqual(names[-len(ARMS):], ARMS)
        self.assertIsNone(B._hook_of("BDref_1216"))
        self.assertEqual(B._hook_of("BDvf1_1216")["type"], "v_freeze")
        self.assertEqual(B._hook_of("BDFB_1216")["type"], "full_batch")


if __name__ == "__main__":
    unittest.main()
