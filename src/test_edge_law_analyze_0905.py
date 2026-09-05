"""edge_law_analyze_0905 の単体テスト（spec `specs/spec_edge_law_0905.md` §4・§5）。

    OMP_NUM_THREADS=1 PYTHONPATH=. python3 -m unittest src.test_edge_law_analyze_0905 -v
    OMP_NUM_THREADS=1 PYTHONPATH=. ~/.local/bin/pytest -q src/test_edge_law_analyze_0905.py

様式は `src/test_weird_act_0903.py` に倣う（unittest・活性化や解析の式は逐語で検算）。
中身は 2 本立て:

1. **合成データ**（真のラベルが分かっている作り物）で全判定を通す。判定ごとに
   「当たる作り物」と「外れる作り物」を対にして、空虚に PASS する経路を塞ぐ
   （本プロジェクトで一度やった「テンソルを自分自身と比べる S 検査」の再発防止）。
2. **committed の実ログ**（`results/p3_extend_0902/logs`）を loader に通し、
   spec §2 の参照値（0.112 / −3.598 / B 1.637 [1.506, 1.796] / 0.794 [0.652, 0.988] /
   死亡率 17.4%）を許容誤差内で再現する。

ログが無い環境ではクラス 2 は `skipTest` する（判定用の 30 腕はまだ走っていない）。
"""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from src import edge_law_analyze_0905 as A
from src.common import ROOT, load_config

CFG = load_config(str(A.CONFIG))
LOGS = Path(ROOT) / "results/p3_extend_0902/logs"
HAVE_LOGS = (LOGS / "LR_1216_seed0.npz").exists()


class ConfigTests(unittest.TestCase):
    """config（正本）と本モジュールの定数が一致すること。"""

    def test_config_constants_match_the_module(self):
        A.check_config(CFG)                      # 例外が出なければ一致

    def test_a_changed_config_raises(self):
        bad = {k: (dict(v) if isinstance(v, dict) else v) for k, v in CFG.items()}
        bad["analysis"] = dict(CFG["analysis"])
        bad["analysis"]["C"] = 11.5              # 閉形式定数を 1 桁ずらす
        with self.assertRaises(A.AnalysisError):
            A.check_config(bad)

    def test_windows_are_the_registered_ones(self):
        self.assertEqual(A.TAIL_5M, (451, 500))
        self.assertEqual(A.TAIL_15M, (1451, 1500))
        self.assertEqual(A.LAG_5M, (351, 400))
        self.assertEqual(A.SETTLE, ((301, 350), (376, 425), (451, 500)))
        self.assertEqual(A.C_CONST, 11.497681)
        self.assertEqual((A.BOOT_N, A.BOOT_SEED), (2000, 20260905))

    def test_tail_window_of_a_15M_arm_is_1451_1500(self):
        arm = A.synth_arm("LRbm5_1216", meta={"family": "leaky",
                                              "total_steps": 15_000_000},
                          final={"layer1_zmax": 0.1}, tasks=1500)
        ctx = A.Ctx({"LRbm5_1216": arm})
        self.assertEqual(ctx.tail("LRbm5_1216"), A.TAIL_15M)
        self.assertEqual(ctx.lag("LRbm5_1216"), A.LAG_15M)


class NumericHelperTests(unittest.TestCase):
    """scipy 無しで書いた道具（順位相関・KS・歪度・緩和フィット・Kendall）。"""

    def test_spearman_is_exact_on_a_monotone_map(self):
        x = np.arange(20.0)
        self.assertAlmostEqual(A.spearman(x, np.exp(x / 5)), 1.0, places=12)
        self.assertAlmostEqual(A.spearman(x, -x), -1.0, places=12)

    def test_spearman_handles_ties_with_average_ranks(self):
        x = np.array([1.0, 1.0, 2.0, 3.0])
        y = np.array([5.0, 5.0, 6.0, 7.0])
        self.assertAlmostEqual(A.spearman(x, y), 1.0, places=12)

    def test_ks_d_matches_the_definition(self):
        a = np.array([0.0, 1.0, 2.0, 3.0])
        b = np.array([2.0, 3.0, 4.0, 5.0])
        self.assertAlmostEqual(A.ks_d(a, b), 0.5, places=12)
        self.assertAlmostEqual(A.ks_d(a, a), 0.0, places=12)

    def test_skewness_sign(self):
        self.assertGreater(A.skewness(np.array([0.0, 0, 0, 0, 10.0])), 0.0)
        self.assertLess(A.skewness(np.array([0.0, 0, 0, 0, -10.0])), 0.0)

    def test_ols_recovers_a_known_line(self):
        x = np.linspace(0, 5, 50)
        a, b = A.ols(2.0 - 3.0 * x, x)
        self.assertAlmostEqual(a, 2.0, places=10)
        self.assertAlmostEqual(b, -3.0, places=10)

    def test_relax_fit_recovers_z_inf_and_tau(self):
        t = np.arange(100, 501, dtype=np.float64)
        z = -1.0 + 5.0 * np.exp(-(t - 100) / 60.0)
        zinf, z0, tau = A.relax_fit(t, z)
        self.assertLess(abs(zinf + 1.0), 0.05)
        self.assertLess(abs(z0 - 4.0), 0.1)
        self.assertLess(abs(tau - 60.0), 6.0)

    def test_kendall_tau(self):
        self.assertEqual(A.kendall_tau([1, 2, 3, 4], [0.1, 0.5, 0.9, 2.0]), 1.0)
        self.assertEqual(A.kendall_tau([1, 2, 3, 4], [2.0, 0.9, 0.5, 0.1]), -1.0)

    def test_bootstrap_draw_table_is_reproducible(self):
        d1 = A.boot_draws(10)
        d2 = A.boot_draws(10)
        self.assertTrue(np.array_equal(d1, d2))
        self.assertEqual(d1.shape, (2000, 10))
        self.assertTrue(np.array_equal(
            d1, np.random.default_rng(20260905).integers(0, 10, size=(2000, 10))))


class ActivationNumpyTests(unittest.TestCase):
    """5-b が使う numpy 版の活性化が spec §3.2 の式どおりであること。"""

    def test_leaky_and_shelf_d0_agree_away_from_the_kink(self):
        z = np.linspace(-30, 30, 4001)
        z = z[np.abs(z) > 1e-9]
        lk = A.act_numpy("leaky_relu", 0.1)
        sh = A.act_numpy("shelf_leaky_d0", 0.1, CFG["activation"])
        self.assertTrue(np.allclose(lk[0](z), sh[0](z), rtol=0, atol=0))
        self.assertTrue(np.allclose(lk[1](z), sh[1](z), rtol=0, atol=0))

    def test_flip_leaky_is_the_odd_mirror_of_leaky(self):
        z = np.linspace(-30, 30, 4001)
        z = z[np.abs(z) > 1e-9]
        lk = A.act_numpy("leaky_relu", 0.1)
        fl = A.act_numpy("flip_leaky", 0.1)
        self.assertTrue(np.allclose(fl[0](z), -lk[0](-z), rtol=0, atol=0))
        self.assertTrue(np.allclose(fl[1](z), lk[1](-z), rtol=0, atol=0))

    def test_shelf_depth_comes_from_the_config(self):
        for name, want in (("shelf_leaky_d0p5", 0.5), ("shelf_leaky_d1", 1.0),
                           ("shelf_leaky_d2", 2.0), ("shelf_leaky_d3", 3.0),
                           ("shelf_leaky_d30", 30.0), ("steep_shelf_d1", 1.0),
                           ("steep_shelf_d2", 2.0)):
            self.assertEqual(A.act_depth(name, CFG["activation"]), want)

    def test_derivatives_match_a_central_difference(self):
        z = np.linspace(-8, 8, 801)
        for name, dial in (("leaky_relu", 0.1), ("elu", 1.0), ("elu", 0.5),
                           ("shelf_leaky_d2", 0.1), ("steep_shelf_d1", 1.0),
                           ("softplus_b", 1.0), ("tanh_b", 1.0)):
            phi, dphi = A.act_numpy(name, dial, CFG["activation"])
            d = A.act_depth(name, CFG["activation"])
            keep = np.abs(z) > 1e-2
            if np.isfinite(d):
                keep &= np.abs(z + d) > 1e-2
            zz, h = z[keep], 1e-5
            fd = (phi(zz + h) - phi(zz - h)) / (2 * h)
            self.assertTrue(np.allclose(fd, dphi(zz), rtol=1e-4, atol=1e-6),
                            f"{name}(dial={dial}) derivative mismatch")

    def test_unknown_activation_raises_rather_than_falling_through(self):
        with self.assertRaises(A.AnalysisError):
            A.act_numpy("not_an_activation", 1.0)


class SupportReconstructionTests(unittest.TestCase):
    """z_p = z̄ + Σ_j s_j w_j/2 の復元（spec §3.4・S-support）。"""

    def test_reconstruction_matches_zmax_and_zmin(self):
        rng = np.random.default_rng(0)
        w = rng.normal(size=(4, 5))
        zbar = rng.normal(size=4)
        signs = ((np.arange(32)[:, None] >> np.arange(5)) & 1) * 2.0 - 1.0
        z = zbar[None, :] + signs @ (w.T * 0.5)
        self.assertTrue(np.allclose(z.max(axis=0),
                                    zbar + 0.5 * np.abs(w).sum(axis=1)))
        self.assertTrue(np.allclose(z.min(axis=0),
                                    2 * zbar - z.max(axis=0)))
        self.assertTrue(np.allclose(z.mean(axis=0), zbar))

    def test_equilibrium_solver_reproduces_the_closed_form(self):
        ok, detail = A._selftest_equilibrium()
        self.assertTrue(ok, detail)


class SyntheticJudgmentTests(unittest.TestCase):
    """真のラベルが分かっている作り物で spec §4 の判定を通す（--selftest と同じ）。"""

    @classmethod
    def setUpClass(cls):
        cls.results = A.selftest_synthetic()

    def _one(self, name):
        hits = [r for r in self.results if r[0] == name]
        self.assertEqual(len(hits), 1, f"{name!r} not produced by selftest_synthetic")
        return hits[0]

    def test_every_synthetic_case_passes(self):
        bad = [f"{n}: {d}" for n, ok, d in self.results if not ok]
        self.assertEqual(bad, [], "; ".join(bad))

    def test_the_expected_cases_are_actually_present(self):
        want = {"1-b mirror exact", "1-b mirror broken (mutant must fail)",
                "1-c ensemble symmetric", "1-d linear pinned",
                "1-e odd symmetric pinned", "2-a sign (FL above +2, LR below -2)",
                "2-b Delta_3 at-init", "2-b Delta_3 nonlocal", "2-c smooth down",
                "2-d asymmetry sign", "2-e reach rate = 0.60",
                "3-a retention rho=0", "3-a retention rho=0.6",
                "3-b relax fit recovers z_inf and tau", "4-b locality monotone",
                "4-b d* = 3", "4-c edge at kink", "4-c edge detached up",
                "4.6 divergence NOT_RUN (3 NaN seeds)",
                "5-a alpha contrast consistent",
                "5-b equilibrium solver reproduces the closed form (ELU)",
                "5-c well from readout", "5-d fluctuation sqrt(eta)",
                "5-f scale law invariant",
                "5-g stationarity on synthetic moments",
                "G1 frozen -> 1-d NOT_DETERMINED", "G1 frozen gate says FROZEN",
                "1-b missing parity columns -> NOT_RUN",
                "1-b fewer than ten seeds -> NOT_RUN",
                "1-b fixture really contains both-zero entries",
                "1-b mirror with +0.0 in both arms (zero-sign exception)",
                "1-b zero fixture mutant must fail"}
        got = {r[0] for r in self.results}
        self.assertEqual(want - got, set())

    def test_mirror_check_is_not_vacuous(self):
        """1 ビット反転させた鏡像は必ず落ちること（空虚な PASS を塞ぐ）。"""
        self.assertTrue(self._one("1-b mirror broken (mutant must fail)")[1])


class GateTests(unittest.TestCase):
    """G1–G6 が実際に働くこと（落ちる作り物を必ず添える）。"""

    def _arm(self, **kw):
        base = dict(final={"layer1_zmax": 0.11, "layer1_zbar": -3.6},
                    init={"layer1_zmax": 0.753, "layer1_zbar": -0.6},
                    meta={"family": "leaky", "activation": "leaky_relu",
                          "dial": 0.1}, tau=5.0)
        base.update(kw)
        return A.synth_arm("LRnull_1216", **base)

    def test_G1_passes_a_moving_arm_and_fails_a_frozen_one(self):
        ctx = A.Ctx({"LRnull_1216": self._arm()})
        self.assertEqual(A.g1_progress(ctx, "LRnull_1216")[0], "PASS")
        frozen = self._arm(final={"layer1_zmax": 0.753, "layer1_zbar": -0.6},
                           w_norm=1.4)
        ctx2 = A.Ctx({"LRnull_1216": frozen})
        self.assertEqual(A.g1_progress(ctx2, "LRnull_1216")[0], "FROZEN")

    def test_G2_flags_a_monotone_drift_wider_than_the_CI(self):
        settled, _ = A.g2_settled(lambda w: 1.0, 0.5)
        self.assertEqual(settled, "PASS")
        drifting, vals = A.g2_settled(lambda w: {(301, 350): 0.0, (376, 425): 1.0,
                                                 (451, 500): 2.0}[tuple(w)], 0.5)
        self.assertEqual(drifting, "NOT_SETTLED")
        self.assertEqual(vals, [0.0, 1.0, 2.0])

    def test_G3_requires_the_two_exclusions_to_agree(self):
        self.assertEqual(A.g3_agreement("X", "X"), "PASS")
        self.assertEqual(A.g3_agreement("X", "Y"), "MISMATCH")

    def test_G4_death_rate_gap(self):
        self.assertEqual(A.g4_comparable(0.10, 0.15), "PASS")
        self.assertEqual(A.g4_comparable(0.05, 0.20), "NOT_COMPARABLE")

    def test_G6_drops_at_most_two_nan_seeds(self):
        arm = self._arm()
        arm.data[0]["layer1_zbar"][-2:, :] = np.nan
        ctx = A.Ctx({"LRnull_1216": arm})
        self.assertEqual(A.g6_divergence(ctx, "LRnull_1216")[0], "PASS")
        for si in (1, 2):
            arm.data[si]["layer1_zbar"][-2:, :] = np.nan
        arm._cache.clear()
        self.assertEqual(A.g6_divergence(ctx, "LRnull_1216")[0], "NOT_RUN")

    def test_G6_flags_a_runaway_without_nan(self):
        arm = self._arm(final={"layer1_zmax": -60.0, "layer1_zbar": -80.0})
        ctx = A.Ctx({"LRnull_1216": arm})
        self.assertEqual(A.g6_divergence(ctx, "LRnull_1216")[0], "ARM_RUNAWAY")


class MissingColumnTests(unittest.TestCase):
    """参照腕に無い新列（zmin / w_free / モーメント）を NOT_DETERMINED にすること。"""

    def _shelf(self):
        return A.synth_arm("SH_d3_1216",
                           meta={"family": "shelf",
                                 "activation": "shelf_leaky_d3", "dial": 0.1},
                           final={"layer1_zmax": -0.4, "layer1_zbar": -3.9},
                           init={"layer1_zmax": 0.753, "layer1_zbar": -0.6},
                           tau=6.0)

    def test_reach_without_zmin_is_not_determined(self):
        ctx = A.Ctx({"SH_d3_1216": self._shelf()})
        rows, rates = A.judge_2e_reach(ctx)
        r = [x for x in rows if x["arm"] == "SH_d3_1216"][0]
        self.assertEqual(r["label"], "NOT_DETERMINED")
        self.assertIn("zmin", r["note"])

    def test_equilibrium_without_w_free_is_not_determined(self):
        ctx = A.Ctx({"SH_d3_1216": self._shelf()})
        rows = A.judge_5b_equilibrium(ctx)
        r = [x for x in rows if x["arm"] == "SH_d3_1216"][0]
        self.assertEqual(r["label"], "NOT_DETERMINED")
        self.assertIn("layer1_w_free", r["note"])

    def test_stationarity_without_moment_columns_is_not_determined(self):
        arm = A.synth_arm("LRnull_1216",
                          meta={"family": "leaky", "activation": "leaky_relu",
                                "dial": 0.1},
                          final={"layer1_zmax": 0.11, "layer1_zbar": -3.6},
                          init={"layer1_zmax": 0.753, "layer1_zbar": -0.6})
        rows = A.judge_5g_stationarity(A.Ctx({"LRnull_1216": arm}))
        r = [x for x in rows if x["arm"] == "LRnull_1216"][0]
        self.assertEqual(r["label"], "NOT_DETERMINED")
        self.assertIn("moment", r["note"])
        overall = [x for x in rows if x["judgment"] == "4.5-g overall"][0]
        self.assertEqual(overall["label"], "NOT_DETERMINED")

    def test_p5_decision_tree(self):
        rows = [A.row("x", "4.5-g overall", "confirmatory", "s", None, "", "STATIONARITY_DIRECT_PASS"),
                A.row("x", "4.5-b overall", "primary", "s", None, "", "EQUILIBRIUM_PREDICTED"),
                A.row("x", "4.5-c v-freeze", "confirmatory", "s", None, "", "WELL_FROM_READOUT")]
        self.assertEqual(A.p5_overall(rows)[0]["label"], "R_SUPPORTED+CAUSAL")
        rows[0]["label"] = "STATIONARITY_DIRECT_FAIL"
        rows[1]["label"] = "EQUILIBRIUM_OFF"
        self.assertEqual(A.p5_overall(rows)[0]["label"], "R_REFUTED+CAUSAL")
        rows[1]["label"] = "NOT_DETERMINED"
        self.assertEqual(A.p5_overall(rows)[0]["label"],
                         "R_NOT_DETERMINED+CAUSAL")
        rows[1]["label"] = "EQUILIBRIUM_ORDER_ONLY"
        rows[2]["label"] = "WELL_INDEPENDENT_OF_READOUT"
        self.assertEqual(A.p5_overall(rows)[0]["label"], "R_PARTIAL+NONCAUSAL")


class VerdictOutputTests(unittest.TestCase):
    """verdict.csv / summary.md の形（列名と順序は spec §9-3 で登録済み）。"""

    def test_verdict_columns_are_the_registered_ones(self):
        # spec §9-3 の列 ＋ §4.6 が要求する「落とした seed 数」列
        # （「落とした seed 数はすべての判定について verdict.csv に列で残す」）。
        self.assertEqual(A.VERDICT_FIELDS,
                         ["arm", "judgment", "role", "statistic", "window",
                          "exclusion", "n", "death_rate", "n_seeds_dropped",
                          "point", "ci_lo",
                          "ci_hi", "gate_G1", "gate_G2", "gate_G3", "gate_G4",
                          "gate_G5", "gate_G6", "label", "note"])

    def test_write_verdict_and_summary(self):
        import csv
        import tempfile
        arm = A.synth_arm("LIN_1216",
                          meta={"family": "linear", "activation": "leaky_relu",
                                "dial": 1.0},
                          final={"layer1_zbar": 0.0, "layer1_zmax": 0.2},
                          init={"layer1_zbar": -0.6, "layer1_zmax": 0.753},
                          w_norm=2.5, tau=5.0)
        ctx = A.Ctx({"LIN_1216": arm})
        rows, extra = A.run_all(ctx)
        with tempfile.TemporaryDirectory() as td:
            A.write_verdict(Path(td) / "verdict.csv", rows)
            A.write_summary(Path(td) / "summary.md", rows, extra, ctx)
            with open(Path(td) / "verdict.csv") as fh:
                got = list(csv.DictReader(fh))
            self.assertEqual(list(got[0]), A.VERDICT_FIELDS)
            self.assertEqual(len(got), len(rows))
            text = (Path(td) / "summary.md").read_text(encoding="utf-8")
        for title, _prefix in A.PROP_SECTIONS:
            self.assertIn(title, text)
        self.assertIn("REPORT_ONLY", text)
        self.assertIn("S-KSnull", text)
        self.assertIn("S-taut", text)

    def test_roles_are_from_the_registered_vocabulary(self):
        arm = A.synth_arm("LIN_1216",
                          meta={"family": "linear", "activation": "leaky_relu",
                                "dial": 1.0},
                          final={"layer1_zbar": 0.0, "layer1_zmax": 0.2},
                          init={"layer1_zbar": -0.6, "layer1_zmax": 0.753})
        rows, _ = A.run_all(A.Ctx({"LIN_1216": arm}))
        self.assertLessEqual({r["role"] for r in rows},
                             {"confirmatory", "primary", "secondary", "report"})


def _full_arm_set(h: int = 8, seeds: int = 4) -> A.Ctx:
    """config の 30 腕ぜんぶを（新列つきで）合成して Ctx にする。"""
    arms = {}
    for blk in CFG["arms"]:
        name = blk["name"]
        tasks = int(blk["total_steps"]) // A.PERIOD
        n = tasks + 1
        zt = A._uniform_units(-0.5, 0.6, seeds=seeds, h=h,
                              rng_seed=abs(hash(name)) % 1000)
        mom = np.full((n, h), 0.01, np.float32)
        extra = {"layer1_zmin": np.full((n, h), -6.0, np.float32),
                 "layer1_dzbar": np.full((n, h), -1e-4, np.float32),
                 "layer1_zmean": np.full((n, h), -3.6, np.float32),
                 "layer1_mob": np.full((n, h), 0.5, np.float32),
                 "layer1_absmob": np.full((n, h), 0.5, np.float32),
                 "layer1_M": np.full((n, h), -3.6, np.float32),
                 "layer1_B": np.zeros((n, h), np.float32),
                 "layer1_p_hat": np.full((n, h), 0.5, np.float32),
                 "layer1_w_free": np.full((n, h, 5), 0.8, np.float32),
                 "layer1_w_free_step": np.arange(n) * A.PERIOD,
                 "layer1_m_phi2": mom, "layer1_m_dphi2": mom,
                 "layer1_m_phidphi": mom, "layer1_m_dphiddphi": -mom,
                 "layer1_moment_step": np.arange(n) * A.PERIOD}
        arms[name] = A.synth_arm(
            name, meta=dict(blk), seeds=seeds, h=h, tasks=tasks,
            final={"layer1_zmax": zt, "layer1_zbar": zt - 3.6},
            init={"layer1_zmax": 0.753, "layer1_zbar": -0.6}, tau=6.0, noise=0.02,
            extra=extra,
            payload={"lr_used": float(blk.get("lr", 0.01) or 0.01),
                     "init_hook": "", "init_hook_arg": float("nan"),
                     "freeze_v": False, "batch_mode": "online"})
    return A.Ctx(arms, CFG)


class FullArmSetTests(unittest.TestCase):
    """30 腕そろった状態で spec §4 の全判定が例外なく回ること（列名の契約）。"""

    @classmethod
    def setUpClass(cls):
        cls.ctx = _full_arm_set()
        cls.rows, cls.extra = A.run_all(cls.ctx)

    def test_every_registered_judgment_produces_a_row(self):
        judged = {r["judgment"] for r in self.rows}
        for want in ("4.1-b S-mirror", "4.1-c ensemble mirror", "4.1-d linear",
                     "4.1-e odd nonlinear", "4.2-a sign", "4.2-b Delta_3",
                     "4.2-b Delta_2", "4.2-c smooth", "4.2-d overall",
                     "4.2-e reach", "4.3-a retention", "4.3-b relaxation fit",
                     "4.3-c return path", "4.4-a literal reading",
                     "4.4-b locality radius", "4.4-c edge overall",
                     "4.5-a alpha contrast", "4.5-b overall", "4.5-c v-freeze",
                     "4.5-d fluctuation scaling", "4.5-f overall",
                     "4.5-g overall", "4.5-h sink order", "4.5-i full batch",
                     "4.5 overall", "4.6 divergence"):
            self.assertIn(want, judged, f"{want} produced no row")

    def test_no_judgment_is_left_as_a_missing_arm(self):
        # FLn/FL/LR_1216 は committed 対照が要る（この合成 Ctx には入れていない）
        bad = [r for r in self.rows if "logs missing" in r["note"]
               and r["arm"] not in ("FLn_1216", "FL_1216", "LR_1216")]
        self.assertEqual([r["judgment"] for r in bad], [])

    def test_fifteen_M_arms_also_report_the_451_500_window(self):
        alt = [r for r in self.rows
               if r["judgment"] == "4.3-a retention (alt window)"]
        self.assertEqual({r["arm"] for r in alt}, {"LRbm5_1216", "Ebm4_1216"})
        for r in alt:
            self.assertEqual(r["window"], "tasks 451-500")

    def test_expected_columns_are_all_present_in_the_synthetic_arms(self):
        for name in ("LRnull_1216", "SH_d3_1216", "Evf1_1216"):
            miss = A.check_arm_columns(self.ctx.get(name))
            self.assertEqual(miss["missing_required"], [])
            self.assertEqual(miss["missing_new"], [])
            self.assertEqual(miss["missing_payload"], [])

    def test_expected_columns_contract(self):
        cols = A.expected_columns()
        self.assertIn("layer1_zmin", cols["new_unit"])
        self.assertIn(["layer1_w_free", "layer1_w_free_step"], cols["new_aux"])
        self.assertEqual(cols["payload"],
                         ["init_hook", "init_hook_arg", "lr_used", "freeze_v",
                          "batch_mode"])
        self.assertEqual(sorted(k for k, _s in A.NEW_AUX_COLUMNS)[1:],
                         ["layer1_m_dphiddphi", "layer1_m_phi2",
                          "layer1_m_phidphi", "layer1_w_free"])

    def test_recorder_shaped_columns_are_accepted(self):
        """recorder が持つ (n_rec, R, h) 形でも seed 軸を落として読めること。"""
        n, S, h = 501, 3, 4
        data = {}
        for si in range(S):
            data[si] = {"step": np.arange(n, dtype=np.int64) * A.PERIOD,
                        "layer1_zmax": np.tile(
                            np.arange(S, dtype=np.float32)[None, :, None],
                            (n, 1, h)),
                        "layer1_w_free": np.full((n, S, h, 5), 0.5, np.float32),
                        "layer1_w_free_step": np.arange(n, dtype=np.int64) * A.PERIOD}
        arm = A.ArmLog("X", data=data, meta={})
        got = arm.unit_window("layer1_zmax", A.TAIL_5M)
        self.assertEqual(got.shape, (S, h))
        self.assertTrue(np.allclose(got[2], 2.0))
        self.assertEqual(
            arm.aux_window("layer1_w_free", "layer1_w_free_step",
                           A.TAIL_5M).shape, (S, 50, h, 5))


@unittest.skipUnless(HAVE_LOGS, "committed p3_extend_0902 logs are not present")
class RealLogTests(unittest.TestCase):
    """committed の実ログで spec §2 の参照値を再現する。"""

    @classmethod
    def setUpClass(cls):
        cls.out = A.selftest_real(verbose=False)

    def test_reference_numbers(self):
        for name, ok, detail in A.check_real_reference(self.out):
            self.assertTrue(ok, f"{name}: {detail}")

    def test_LR_1216_tail_medians(self):
        self.assertAlmostEqual(self.out["LR_1216 tail zmax median"], 0.112, places=3)
        self.assertAlmostEqual(self.out["LR_1216 tail zbar median"], -3.598, places=3)
        self.assertEqual(self.out["LR_1216 death rate"], 0.0)

    def test_E_1216_death_rate_and_B(self):
        self.assertAlmostEqual(self.out["E_1216 death rate"], 0.174, places=4)
        b = self.out["E_1216 B (simultaneous v, ALIVE)"]
        self.assertEqual(b[3], 826)                    # ALIVE ユニット数
        self.assertAlmostEqual(b[0], 1.637, places=3)
        sub = self.out["E_1216 B (lag v, fully submerged)"]
        self.assertEqual(sub[3], 349)                  # 完全沈水部分集団
        self.assertAlmostEqual(sub[0], 0.794, places=3)

    def test_C_is_the_closed_form_constant(self):
        lo, hi = self.out["C from logs (min, max)"]
        self.assertLess(abs(lo - A.C_CONST), 1e-12)
        self.assertLess(abs(hi - A.C_CONST), 1e-12)
        self.assertLess(self.out["dose_relative_error max"], 1e-12)

    def test_s_taut_records_both_definitions(self):
        taut = self.out["S-taut (LR_1216)"]
        self.assertEqual(taut["window"], A.TAIL_5M)
        self.assertLess(taut["new"][2.0], 0.05)
        self.assertLess(taut["new"][3.0], 0.05)
        self.assertGreater(taut["old"][0.5], 0.7)      # 旧定義は浅い d で恒真に近い
        self.assertTrue(taut["new_below_0.05_at_d2_d3"])

    def test_s_ksnull_q95_is_a_usable_threshold(self):
        ks = self.out["S-KSnull (LR_1216)"]
        self.assertEqual(ks["n_rep"], 2000)
        self.assertTrue(0.0 < ks["q95"] < 0.5)
        self.assertLessEqual(ks["median"], ks["q95"])
        self.assertLessEqual(ks["q95"], ks["max"])

    def test_loader_tolerates_the_absent_new_columns(self):
        ctx = A.build_ctx(CFG)
        lr = ctx.get("LR_1216")
        for key in ("layer1_zmin", "layer1_w_free", "layer1_m_phi2",
                    "layer1_moment_step"):
            self.assertFalse(lr.has(key), f"{key} unexpectedly present")
        self.assertTrue(lr.has("layer1_zmax"))
        ctx.close()

    def test_window_is_fifty_task_end_records(self):
        ctx = A.build_ctx(CFG)
        lr = ctx.get("LR_1216")
        self.assertEqual(lr.window_count(A.TAIL_5M), 50)
        self.assertEqual(lr.window_count(A.LAG_5M), 50)
        idx = lr.window_index(0, A.TAIL_5M)
        steps = lr.steps(0)[idx]
        self.assertTrue(bool((steps % A.PERIOD == 0).all()))
        self.assertEqual(int(steps[0]), 451 * A.PERIOD)
        self.assertEqual(int(steps[-1]), 500 * A.PERIOD)
        ctx.close()


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# 批評ラウンド 1 で塞いだ穴（各項目に「変異体が必ず落ちる」対を付ける）
# ---------------------------------------------------------------------------
class SeedDropTests(unittest.TestCase):
    """G6: NaN の seed は数えるだけでなく**落とす**（spec §3.6・§4.6）。"""

    def _lin(self, nan_seeds=()):
        arm = A.synth_arm(
            "LIN_1216",
            meta={"family": "linear", "activation": "leaky_relu", "dial": 1.0},
            final={"layer1_zbar": 0.0, "layer1_zmax": 0.05},
            init={"layer1_zbar": -0.6, "layer1_zmax": 0.753},
            tau=5.0, w_norm=2.2)
        for si in arm.data:
            shape = arm.data[si]["layer1_zbar"].shape
            half = np.linspace(0.01, 0.05, shape[1] // 2)
            sym = np.tile(np.concatenate([half, -half]), (shape[0], 1))
            arm.data[si]["layer1_zbar"] = sym.astype(np.float32)
        for si in nan_seeds:
            arm.data[si]["layer1_zbar"][-3:, :] = np.nan
        return arm

    def test_two_nan_seeds_are_dropped_and_the_label_survives(self):
        clean = A.Ctx({"LIN_1216": self._lin()})
        self.assertEqual(A.judge_1d_linear(clean)[0]["label"], "LINEAR_PINNED")
        arm = self._lin((0, 1))
        ctx = A.Ctx({"LIN_1216": arm})
        self.assertEqual(arm.dropped_seeds, [0, 1])
        self.assertEqual(arm.kept_seeds, [2, 3, 4, 5, 6, 7, 8, 9])
        self.assertEqual(ctx.dropped("LIN_1216"), 2)
        got = A.judge_1d_linear(ctx)[0]
        # 落とさないと median が NaN 汚染されて LINEAR_ASYMMETRIC になっていた
        self.assertEqual(got["label"], "LINEAR_PINNED")
        self.assertTrue(np.isfinite(got["point"]))
        rows, _extra = A.run_all(ctx)
        for r in rows:
            if r["arm"] == "LIN_1216":
                self.assertEqual(r["n_seeds_dropped"], 2, r["judgment"])

    def test_more_than_two_nan_seeds_is_NOT_RUN_and_nothing_is_dropped(self):
        arm = self._lin((0, 1, 2))
        ctx = A.Ctx({"LIN_1216": arm})
        self.assertEqual(A.g6_divergence(ctx, "LIN_1216")[0], "NOT_RUN")
        self.assertEqual(arm.kept_seeds, list(range(10)))
        self.assertEqual(A.judge_1d_linear(ctx)[0]["label"], "NOT_DETERMINED")

    def test_the_drop_is_recorded_on_every_row(self):
        self.assertIn("n_seeds_dropped", A.VERDICT_FIELDS)
        arm = self._lin((0,))
        rows, _ = A.run_all(A.Ctx({"LIN_1216": arm}))
        self.assertTrue(all("n_seeds_dropped" in r for r in rows))


class MirrorParityRuleTests(unittest.TestCase):
    """SYM1/SYM2: 鏡像パリティの規則は 1 つだけ（零の符号の扱いを含む）。"""

    def test_negative_preserves_the_sign_of_zero_and_the_rule_accepts_both_zero(self):
        x = np.array([0.0, -0.0, 1.5], dtype=np.float32)
        y = np.array([0.0, 0.0, -1.5], dtype=np.float32)
        got = A.mirror_parity(x, y, flip=True)
        self.assertTrue(got["pass_"], got)
        self.assertEqual(got["n_zero_sign_exceptions"], 1)   # x[0]=+0 vs y[0]=+0
        self.assertEqual(got["n_mismatch"], 0)

    def test_a_genuine_sign_error_at_a_nonzero_entry_fails(self):
        x = np.array([1.5, 2.0], dtype=np.float32)
        y = np.array([-1.5, 2.0], dtype=np.float32)
        got = A.mirror_parity(x, y, flip=True)
        self.assertFalse(got["pass_"])
        self.assertEqual(got["n_mismatch"], 1)

    def test_the_zero_exception_does_not_swallow_a_nonzero_zero_pair(self):
        """片方だけ 0 なら例外にしない（+0.0 と 0.5 は通してはいけない）。"""
        x = np.array([0.0], dtype=np.float32)
        y = np.array([0.5], dtype=np.float32)
        self.assertFalse(A.mirror_parity(x, y, flip=True)["pass_"])

    def test_moving_a_nan_without_changing_its_count_fails(self):
        x = np.array([np.nan, 1.5, 2.0], dtype=np.float32)
        y = np.array([-1.5, np.nan, -2.0], dtype=np.float32)
        got = A.mirror_parity(x, y, flip=True)
        self.assertFalse(got["pass_"])
        self.assertFalse(got["nan_pattern_equal"])

    def test_the_runner_uses_the_same_rule_object(self):
        from src import edge_law_0905 as E
        self.assertIs(E._mirror_rule, A.mirror_parity)


_FULL_RUN: dict = {}


def _full_run_rows() -> list[dict]:
    """30 腕の合成セットに `run_all` を 1 回だけ通し、行を使い回す（重いため）。"""
    if "rows" not in _FULL_RUN:
        ctx = _full_arm_set()
        try:
            rows, _extra = A.run_all(ctx)
        finally:
            ctx.close()
        _FULL_RUN["rows"] = rows
    return _FULL_RUN["rows"]


class GateCoverageTests(unittest.TestCase):
    """SPE4/SPE5/SPE10: G2・G3・G4 が確証的な判定にも前置されていること。"""

    def setUp(self):
        self.rows = _full_run_rows()

    def _gate(self, judgment_prefix, gate):
        return {r[gate] for r in self.rows
                if r["judgment"].startswith(judgment_prefix)}

    def test_the_confirmatory_judgments_all_carry_G2_and_G3(self):
        for judgment in ("4.3-a", "4.4-b locality radius", "4.5-c", "4.5-g"):
            for gate in ("gate_G2", "gate_G3"):
                got = self._gate(judgment, gate)
                self.assertNotEqual(got, {"NA"},
                                    f"{judgment} {gate} is still NA: {got}")

    def test_the_primary_proposition_5_judgments_carry_G2_and_G3(self):
        for judgment in ("4.5-a Delta B", "4.5-b numerical equilibrium",
                         "4.5-d fluctuation", "4.5-f scale"):
            for gate in ("gate_G2", "gate_G3"):
                got = self._gate(judgment, gate)
                self.assertNotEqual(got, {"NA"},
                                    f"{judgment} {gate} is still NA: {got}")

    def test_G4_blocks_an_arm_vs_arm_comparison(self):
        self.assertEqual(A.blocked({"G1": "PASS", "G6": "PASS",
                                    "G4": "NOT_COMPARABLE"}),
                         "G4: NOT_COMPARABLE")
        self.assertIsNone(A.blocked({"G1": "PASS", "G6": "PASS", "G4": "PASS"}))

    def test_a_death_rate_gap_makes_3a_not_comparable(self):
        ref = A.synth_arm(
            "LRnull_1216",
            meta={"family": "leaky", "activation": "leaky_relu", "dial": 0.1},
            final={"layer1_zmax": 0.11, "layer1_zbar": -3.6},
            init={"layer1_zmax": 0.753, "layer1_zbar": -0.6}, tau=5.0)
        arm = A.synth_arm(
            "LRbp5_1216",
            meta={"family": "leaky", "activation": "leaky_relu", "dial": 0.1},
            final={"layer1_zmax": 0.11, "layer1_zbar": -3.6},
            init={"layer1_zmax": 5.753, "layer1_zbar": 4.4}, tau=5.0)
        for si in arm.data:                       # 死亡率 60% にする
            arm.data[si]["layer1_denom"][:, :60] = 0.0
        rows, _labels = A.judge_3a_retention(A.Ctx({"LRnull_1216": ref,
                                                    "LRbp5_1216": arm}))
        got = [r for r in rows if r["arm"] == "LRbp5_1216"][0]
        self.assertEqual(got["gate_G4"], "NOT_COMPARABLE")
        self.assertEqual(got["label"], "NOT_DETERMINED")


class Proposition3LabelTests(unittest.TestCase):
    """SPE6: 3-a の第 3 分岐は経路であってラベルではない／命題 3 の総合行がある。"""

    def test_no_row_carries_the_unregistered_INTERMEDIATE_label(self):
        rows = _full_run_rows()
        self.assertNotIn("INTERMEDIATE", {r["label"] for r in rows})
        overall = [r for r in rows if r["judgment"].startswith("4.3 overall")]
        self.assertEqual(len(overall), 2, [r["judgment"] for r in overall])
        allowed = {"MEAN_INDEPENDENT", "MEAN_INDEPENDENT_SLOW", "MEAN_DEPENDENT",
                   "NOT_DETERMINED"}
        for r in overall:
            self.assertIn(r["label"], allowed)

    def test_the_routing_state_is_reported_in_the_note(self):
        ref = A.synth_arm(
            "LRnull_1216",
            meta={"family": "leaky", "activation": "leaky_relu", "dial": 0.1},
            final={"layer1_zmax": 0.11, "layer1_zbar": -3.6},
            init={"layer1_zmax": 0.753, "layer1_zbar": -0.6}, tau=5.0)
        arm = A.synth_arm(
            "LRbp5_1216",
            meta={"family": "leaky", "activation": "leaky_relu", "dial": 0.1},
            final={"layer1_zmax": 0.11 + 0.25 * 5.0, "layer1_zbar": -3.6},
            init={"layer1_zmax": 5.753, "layer1_zbar": 4.4}, tau=5.0)
        rows, labels = A.judge_3a_retention(A.Ctx({"LRnull_1216": ref,
                                                   "LRbp5_1216": arm}))
        got = [r for r in rows if r["arm"] == "LRbp5_1216"][0]
        self.assertEqual(labels["LRbp5_1216"], "INTERMEDIATE")
        self.assertEqual(got["label"], "NOT_DETERMINED")
        self.assertIn("ROUTED_TO_3B", got["note"])


class StationarityFamilyTests(unittest.TestCase):
    """SPE2: 5-g の NOT_DETERMINED は**族**を数える（腕ではない）。"""

    def _elu_arm(self, name, h=8, seeds=4, tasks=500, moments=True):
        n = tasks + 1
        mom = np.full((n, h), 0.01, np.float32)
        extra = {"layer1_dzbar": np.full((n, h), -1e-4, np.float32)}
        if moments:
            extra.update({"layer1_m_phi2": mom, "layer1_m_dphi2": mom,
                          "layer1_m_phidphi": mom * 50.0,
                          "layer1_m_dphiddphi": -mom,
                          "layer1_moment_step": np.arange(n) * A.PERIOD})
        return A.synth_arm(
            name, meta={"family": "elu", "activation": "elu", "dial": 1.0},
            seeds=seeds, h=h, tasks=tasks,
            final={"layer1_zmax": -1.0, "layer1_zbar": -6.5},
            init={"layer1_zmax": 0.753, "layer1_zbar": -0.6}, tau=6.0,
            extra=extra, payload={"lr_used": 0.01})

    def test_one_family_with_moment_columns_is_NOT_DETERMINED(self):
        arms = {n: self._elu_arm(n) for n in
                ("Enull_1216", "E_a0p5_1216", "E_a2_1216", "E_a4_1216")}
        ctx = A.Ctx(arms)
        overall = [r for r in A.judge_5g_stationarity(ctx)
                   if r["judgment"] == "4.5-g overall"][0]
        self.assertEqual(overall["label"], "NOT_DETERMINED")
        self.assertIn("1/5", overall["note"])

    def test_no_family_at_all_is_NOT_DETERMINED(self):
        arms = {n: self._elu_arm(n, moments=False) for n in
                ("Enull_1216", "E_a0p5_1216")}
        overall = [r for r in A.judge_5g_stationarity(A.Ctx(arms))
                   if r["judgment"] == "4.5-g overall"][0]
        self.assertEqual(overall["label"], "NOT_DETERMINED")


class AlphaLimitedTests(unittest.TestCase):
    """SPE11: NaN でない逸走は ALPHA_LIMITED として立てる（腕は外さない）。"""

    def _elu(self, name, alpha, zbar=-6.5):
        return A.synth_arm(
            name, meta={"family": "elu", "activation": "elu", "dial": alpha},
            final={"layer1_zmax": -1.0, "layer1_zbar": zbar},
            init={"layer1_zmax": 0.753, "layer1_zbar": -0.6}, tau=6.0, w_norm=6.6)

    def test_a_runaway_alpha_produces_an_ALPHA_LIMITED_row(self):
        arms = {"Enull_1216": self._elu("Enull_1216", 1.0),
                "E_a0p5_1216": self._elu("E_a0p5_1216", 0.5),
                "E_a2_1216": self._elu("E_a2_1216", 2.0),
                "E_a4_1216": self._elu("E_a4_1216", 4.0, zbar=-80.0)}
        rows = A.judge_5a_alpha(A.Ctx(arms))
        limited = [r for r in rows if r["label"] == "ALPHA_LIMITED"]
        self.assertEqual(len(limited), 1, [r["judgment"] for r in rows])
        self.assertEqual(limited[0]["arm"], "E_a4_1216")
        overall = [r for r in rows
                   if r["judgment"] == "4.5-a alpha contrast"][0]
        self.assertEqual(overall["label"], "ALPHA_CONTRAST_INCONSISTENT")
        # 腕は掃引に残っている（外していない）
        self.assertTrue(any(r["judgment"].startswith("4.5-a Delta B(alpha=4.0")
                            for r in rows))

    def test_without_a_runaway_no_ALPHA_LIMITED_row(self):
        arms = {"Enull_1216": self._elu("Enull_1216", 1.0),
                "E_a0p5_1216": self._elu("E_a0p5_1216", 0.5),
                "E_a2_1216": self._elu("E_a2_1216", 2.0),
                "E_a4_1216": self._elu("E_a4_1216", 4.0)}
        rows = A.judge_5a_alpha(A.Ctx(arms))
        self.assertEqual([r for r in rows if r["label"] == "ALPHA_LIMITED"], [])


class LocalityBaselineTests(unittest.TestCase):
    """SPE7: 4-b は基準線 `LIN_1216` の G1 を読む。"""

    def _shelf(self, name, zmax):
        return A.synth_arm(
            name, meta={"family": "shelf", "activation": name.split("_")[0],
                        "dial": 0.1},
            final={"layer1_zmax": zmax, "layer1_zbar": zmax - 3.6},
            init={"layer1_zmax": 0.753, "layer1_zbar": -0.6}, tau=6.0, w_norm=4.0)

    def _ctx(self, lin_w_norm):
        arms = {"LIN_1216": A.synth_arm(
            "LIN_1216", meta={"family": "linear", "activation": "leaky_relu",
                              "dial": 1.0},
            final={"layer1_zmax": 0.05, "layer1_zbar": 0.0},
            init={"layer1_zmax": 0.753, "layer1_zbar": -0.6}, tau=5.0,
            w_norm=lin_w_norm, w_norm_init=1.418)}
        for d, name in ((0.5, "SH_d0p5_1216"), (1.0, "SH_d1_1216"),
                        (2.0, "SH_d2_1216"), (3.0, "SH_d3_1216"),
                        (30.0, "SH_d30_1216")):
            arms[name] = self._shelf(name, 0.05 if d >= 3 else 0.05 - d)
        return A.Ctx(arms)

    def test_a_frozen_baseline_demotes_the_confirmatory_label(self):
        ctx = self._ctx(lin_w_norm=0.41)          # spec §2 の実測（1.41 -> 0.41）
        self.assertEqual(A.g1_progress(ctx, "LIN_1216")[0], "FROZEN")
        ladder = [r for r in A.judge_4b_locality(ctx)
                  if r["arm"] == "SH_d ladder"][0]
        self.assertEqual(ladder["gate_G1"], "FROZEN")
        self.assertEqual(ladder["label"], "NOT_DETERMINED")

    def test_a_moving_baseline_leaves_the_label_alone(self):
        ctx = self._ctx(lin_w_norm=2.2)
        self.assertEqual(A.g1_progress(ctx, "LIN_1216")[0], "PASS")
        ladder = [r for r in A.judge_4b_locality(ctx)
                  if r["arm"] == "SH_d ladder"][0]
        self.assertEqual(ladder["gate_G1"], "PASS")
        self.assertNotEqual(ladder["label"], "NOT_DETERMINED")


class KSNullScaleTests(unittest.TestCase):
    """SPE8: 1-c の統計量と S-KSnull の帰無分布が同じ単位（seed 水準）。"""

    def test_the_null_is_built_from_per_seed_D(self):
        arm = A.synth_arm(
            "LR_1216", meta={"family": "leaky", "activation": "leaky_relu",
                             "dial": 0.1}, h=100,
            final={"layer1_zmax": 0.11, "layer1_zbar": A._uniform_units(-3.6, 1.0)},
            init={"layer1_zmax": 0.753, "layer1_zbar": -0.6}, tau=5.0, noise=0.2)
        got = A.s_ksnull(A.Ctx({"LR_1216": arm}))
        self.assertIn("per-seed KS D", got["unit"])
        # per-seed（≈100 対 ≈100）の D はプール版（≈1000 対 ≈1000）より必ず大きい
        self.assertGreater(got["q95"], got["q95_pooled"])

    def test_an_exact_mirror_reaches_ENSEMBLE_SYMMETRIC(self):
        lr = A.synth_arm(
            "LR_1216", meta={"family": "leaky", "activation": "leaky_relu",
                             "dial": 0.1}, h=100,
            final={"layer1_zmax": 0.11, "layer1_zbar": A._uniform_units(-3.6, 1.0)},
            init={"layer1_zmax": 0.753, "layer1_zbar": -0.6}, tau=5.0, noise=0.2)
        fl = A.synth_arm(
            "FL_1216", meta={"family": "flip", "activation": "flip_leaky",
                             "dial": 0.1}, h=100,
            final={"layer1_zmax": 0.11, "layer1_zbar": -A._uniform_units(-3.6, 1.0)},
            init={"layer1_zmax": 0.753, "layer1_zbar": 0.6}, tau=5.0, noise=0.2)
        ctx = A.Ctx({"LR_1216": lr, "FL_1216": fl})
        ks = A.s_ksnull(ctx)
        got = A.judge_1c_ensemble(ctx, ks["q95"])[0]
        self.assertEqual(got["label"], "ENSEMBLE_SYMMETRIC", got["note"])


class SecondaryAliveRuleTests(unittest.TestCase):
    """SPE12: 副次除外規則（半幅 > 0.25）の REPORT 行と config 照合。"""

    def test_a_report_row_carries_both_n(self):
        rows = _full_run_rows()
        sec = [r for r in rows if r["judgment"].startswith("3.5 alive rule")]
        self.assertEqual(len(sec), len(CFG["arms"]))
        for r in sec:
            self.assertEqual(r["role"], "report")
            self.assertIn(r["label"], ("N_AGREES", "N_GAP_RECORDED"))
            self.assertIn("secondary n=", r["note"])

    def test_the_secondary_rule_is_checked_against_the_config(self):
        cfg = load_config(str(A.CONFIG))
        A.check_config(cfg)                        # 素の config は通る
        cfg["analysis"]["alive_rule_secondary"] = "(zmax - zbar) > 0.9"
        with self.assertRaises(A.AnalysisError):
            A.check_config(cfg)

    def test_the_secondary_mask_is_the_half_width_rule(self):
        arm = A.synth_arm(
            "LRnull_1216",
            meta={"family": "leaky", "activation": "leaky_relu", "dial": 0.1},
            final={"layer1_zmax": 0.11, "layer1_zbar": -3.6},
            init={"layer1_zmax": 0.753, "layer1_zbar": -0.6}, tau=5.0)
        win = (451, 500)
        half = (arm.unit_window("layer1_zmax", win)
                - arm.unit_window("layer1_zbar", win))
        np.testing.assert_array_equal(arm.alive_secondary(win),
                                      half > A.ALIVE_HALF_WIDTH)


class OddNonlinearG3Tests(unittest.TestCase):
    """SPE9: 1-e の G3 は**最終ラベル**どうしを比べる（対称性の述語だけではない）。"""

    def _th(self, **kw):
        base = dict(
            meta={"family": "tanh", "activation": "tanh_b", "dial": 1.0},
            final={"layer1_zmax": 0.4, "layer1_zbar": 0.0},
            init={"layer1_zmax": 0.753, "layer1_zbar": -0.6}, tau=5.0,
            w_norm=2.5, w_norm_init=1.418)
        base.update(kw)
        return A.synth_arm("TH_1216", **base)

    def test_the_note_records_a_full_label_under_ALL_not_a_boolean(self):
        got = A.judge_1e_odd(A.Ctx({"TH_1216": self._th()}))[0]
        self.assertIn("ALL label ODD_", got["note"])
        self.assertNotIn("ALL label True", got["note"])
        self.assertNotIn("ALL label False", got["note"])

    def test_a_label_that_flips_under_the_exclusion_rule_is_caught(self):
        """ALIVE と ALL で釘付け／中間が割れる作り物は G3 で止まる。"""
        arm = self._th()
        # 対称性は ALIVE でも ALL でも成り立つ（±対称）が、除外されるユニットの
        # |z̄| が大きいので「釘付け」だけが割れる: ALIVE=PINNED / ALL=INTERMEDIATE。
        # 対称性の述語だけを比べていた旧実装はここを PASS にしていた。
        for si in arm.data:
            zb = arm.data[si]["layer1_zbar"]
            h = zb.shape[1]
            half = h // 2
            arm.data[si]["layer1_denom"][:, :half] = 0.0
            pattern = np.empty(h, dtype=np.float32)
            pattern[:half] = np.where(np.arange(half) % 2 == 0, 5.0, -5.0)
            pattern[half:] = np.where(np.arange(h - half) % 2 == 0, 0.2, -0.2)
            arm.data[si]["layer1_zbar"] = np.tile(pattern, (zb.shape[0], 1))
        ctx = A.Ctx({"TH_1216": arm})
        got = A.judge_1e_odd(ctx)[0]
        self.assertEqual(got["gate_G3"], "MISMATCH", got["note"])
        self.assertEqual(got["label"], "NOT_DETERMINED")
