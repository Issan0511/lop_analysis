"""drift_law_0907 の検査（読み規則 `specs/spec_drift_law_0907.md`）。

    OMP_NUM_THREADS=1 PYTHONPATH=. python3 -m unittest src.test_drift_law_0907 -v

S-loo（ζ が k を含まない＝平均回帰が入らない）/ S-supp（32 点と半幅の整合）/ S-shape（人工データで
傾きを復元する・変異体は落ちる）/ S-win（窓とビンの規約）。**恒等式の検査には変異体の対を置く。**
"""
from __future__ import annotations

import unittest

import numpy as np

from src import drift_law_0907 as D


class TestSupport(unittest.TestCase):
    def test_32_points_and_halfwidth(self):
        """ζ + u の 32 点の最大は ζ + 0.5Σ|w|、最小は ζ − 0.5Σ|w|。"""
        rng = np.random.default_rng(0)
        w = rng.normal(size=(7, 5))
        zeta = rng.normal(size=7)
        pts = zeta[:, None] + np.einsum("kj,hj->hk", D.SIGNS, w)
        half = 0.5 * np.abs(w).sum(axis=1)
        self.assertTrue(np.allclose(pts.max(axis=1), zeta + half))
        self.assertTrue(np.allclose(pts.min(axis=1), zeta - half))
        self.assertEqual(pts.shape[1], 32)

    def test_p_up_quantised(self):
        rng = np.random.default_rng(1)
        w = rng.normal(size=(50, 5))
        zeta = rng.normal(scale=2.0, size=50)
        pts = zeta[:, None] + np.einsum("kj,hj->hk", D.SIGNS, w)
        p = (pts > 0).mean(axis=1)
        self.assertTrue(np.allclose(p * 32, np.round(p * 32)))
        # ζ を十分下げれば全点が下（p=0）、上げれば全点が上（p=1）
        far = -100.0 + np.einsum("kj,hj->hk", D.SIGNS, w)
        self.assertEqual(float((far > 0).mean()), 0.0)


class TestLeaveOneOut(unittest.TestCase):
    def test_zeta_excludes_both_endpoints(self):
        """ζ_k は N の両端（k と k+1）を除く。片方でも入れると汚染が残る。"""
        H = D.HALF
        self.assertEqual(D.EXCLUDE, (0, 1))
        nb = [d for d in range(-H, H + 2) if d not in D.EXCLUDE]
        self.assertEqual(len(nb), 2 * H)
        self.assertNotIn(0, nb)
        self.assertNotIn(1, nb)

    def test_zeta_kills_mean_reversion(self):
        """揺らぎだけの系（真の drift 0）で、両端を除けば傾き 0。片方だけ除くと**正**に、
        両端とも入れると（対称窓なので）偶然打ち消して 0 になる——だから「k だけ除く」は
        安全ではない。実測でこの 3 通りを分ける。"""
        rng = np.random.default_rng(2)
        n_task, n_unit, H = 600, 200, D.HALF
        base = rng.normal(scale=1.0, size=n_unit)
        z = base[None, :] + rng.normal(scale=0.45, size=(n_task, n_unit))
        N, Zb, Zk, Zw = [], [], [], []
        for k in range(H, n_task - H - 2):
            N.append(z[k + 1] - z[k])
            Zb.append(np.mean([z[k + d] for d in range(-H, H + 2) if d not in (0, 1)], axis=0))
            Zk.append(np.mean([z[k + d] for d in range(-H, H + 1) if d != 0], axis=0))
            Zw.append(np.mean([z[k + d] for d in range(-H, H + 1)], axis=0))
        N, Zb, Zk, Zw = (np.asarray(x).ravel() for x in (N, Zb, Zk, Zw))
        s_both = np.polyfit(Zb, N, 1)[0]
        s_konly = np.polyfit(Zk, N, 1)[0]
        # 理論値: k だけ除くと ζ に z̄_{k+1} が 1/(2H) 残るので
        #   slope ≈ Var(揺らぎ)/(2H) / Var(ζ) = (0.45²/8)/(1+0.45²/8) = 0.0247
        # 実測の drift の傾きは 0.004/単位 程度なので、**この汚染は信号の 6 倍**になる。
        self.assertLess(abs(s_both), 0.01)          # 両端を除く: 0 ← 採用する定義
        self.assertAlmostEqual(s_konly, 0.0247, delta=0.008)   # k だけ除く: 正に化ける


class TestShapeRecovery(unittest.TestCase):
    def test_slope_recovered_from_synthetic(self):
        """E[−N] = κ(1−a) p^s の合成データで、傾き s を復元する。"""
        rng = np.random.default_rng(3)
        for s_true in (0.5, 1.0, 1.5):
            rows = []
            for a in (0.1, 0.3, 0.5):
                for p in np.geomspace(0.02, 0.6, 8):
                    rows.append(dict(p_med=float(p), a=a,
                                     mN=float(0.03 * (1 - a) * p ** s_true
                                              * np.exp(rng.normal(scale=0.02))),
                                     lo=0.0, hi=1.0, n=1000))
            s, nb, span = D.slope_from(rows)
            self.assertAlmostEqual(s, s_true, delta=0.05)
            self.assertGreaterEqual(nb, D.MIN_BINS)
            self.assertGreater(span, D.MIN_SPAN)

    def test_slope_is_insensitive_to_the_a_correction(self):
        """(1−a) で割っても割らなくても、腕ごとの x 範囲が同じなら傾きは同じ。

        つまり主判定 `DRIFT_SHAPE` は **形だけ**を測っており、(1−a) の当否は
        第 2 判定 `A_COLLAPSE` の側でしか効かない（設計の性質・空虚な検査よけに明示）。"""
        rows = []
        for a in (0.1, 0.7):
            for p in np.geomspace(0.02, 0.6, 8):
                rows.append(dict(p_med=float(p), a=a, mN=float(0.03 * (1 - a) * p),
                                 lo=0.0, hi=1.0, n=1000))
        good, _, _ = D.slope_from(rows)
        no_corr, _, _ = D.slope_from([dict(r, a=0.0) for r in rows])
        self.assertAlmostEqual(good, 1.0, delta=0.02)
        self.assertAlmostEqual(no_corr, good, delta=0.02)

    def test_scrambling_p_kills_the_slope(self):
        """p と mN の対応を壊すと傾きが 0 に落ちる（検査に歯があることの対）。"""
        rng = np.random.default_rng(5)
        ps = np.geomspace(0.02, 0.6, 10)
        rows = [dict(p_med=float(p), a=0.1, mN=float(0.03 * 0.9 * p), lo=0.0, hi=1.0, n=1000)
                for p in ps]
        good, _, _ = D.slope_from(rows)
        vals = [r["mN"] for r in rows]
        rng.shuffle(vals)
        bad, _, _ = D.slope_from([dict(r, mN=v) for r, v in zip(rows, vals)])
        self.assertAlmostEqual(good, 1.0, delta=0.02)
        self.assertLess(abs(bad), 0.6)


class TestConventions(unittest.TestCase):
    def test_window_and_constants(self):
        self.assertEqual(D.WINDOW, (10, 199))
        self.assertEqual(D.HALF, 4)
        self.assertEqual(D.T, 10_000)
        self.assertEqual(D.RNG_SEED, 20260907)
        self.assertGreater(D.FLOOR_TOP_BIN, 0)

    def test_deciles_are_increasing_and_unique(self):
        rng = np.random.default_rng(4)
        p = np.round(rng.random(5000) * 32) / 32          # 量子化された p̂_up
        e = D.deciles(p)
        self.assertTrue(np.all(np.diff(e) > 0))
        self.assertGreaterEqual(len(e), 2)


if __name__ == "__main__":
    unittest.main()
