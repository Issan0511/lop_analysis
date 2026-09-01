from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from unittest import mock

import src.ident_mu_2x2_0901 as ident_module
from src.bias_wd_common import exact_wall_record
from src.common import load_config
from src.ident_mu_2x2_0901 import (
    ANCHOR_ARM,
    ARM_ORDER,
    BOTH_MATTER,
    BWD_PREVENTS_EXTINCTION,
    EFFECT_LEVEL_DEPENDENT,
    EXTINCTION_PERSISTS,
    IDENT_DOMINANT,
    INCONCLUSIVE_WIDE,
    INTERACTION_DOMINATES,
    MU_DOMINANT,
    NEITHER_MATTERS,
    NOWD_ARM,
    PARTIAL_RESCUE,
    _setup,
    _sha_file,
    _validate_offset,
    analyze,
    band_of,
    block_levels,
    classify_2x2,
    classify_r_ext,
    _s_op_identification,
    extinction_table,
    forward_ident,
    grads_ident,
    ident_run_record,
    onset_table,
    onset_task,
    preregistration_missing,
    set_offset,
    sgd_step_ident,
    validate_config,
)


def _train(st: dict, steps: int) -> dict:
    period = int(st["runs"][0]["period"])
    for t in range(steps):
        x = st["env"].step()
        y = st["teacher"](x)
        if t % period == 0:
            set_offset(st, x[:, :st["n_flip"]])
        inputs, pres, acts, yhat, code = forward_ident(st, x)
        sgd_step_ident(st, grads_ident(st, inputs, pres, acts, code, yhat - y))
    return st


class ConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = load_config("configs/ident_mu_2x2_0901.yaml")

    def test_draft_allows_selftests_but_blocks_gates_run_and_analysis(self) -> None:
        """事前登録が成立する前は自己検査しか通らない（現行 config は成立済み）。"""
        self.assertEqual(preregistration_missing(self.cfg), [])
        draft = copy.deepcopy(self.cfg)
        draft["preregistration"].update(
            decisions_complete=False, frozen=False, repo_spec_committed=False,
            execution_authorized=False)
        validate_config(draft, stage="implementation")
        validate_config(draft, stage="selftest")
        validate_config(draft, stage="smoke")
        self.assertEqual(preregistration_missing(draft), [
            "preregistration.decisions_complete",
            "preregistration.frozen",
            "preregistration.repo_spec_committed",
            "preregistration.execution_authorized",
        ])
        # ゲートは登録済みの合否規則を当てる行為なので、事前登録の成立点
        # （repo 側 spec の単独 commit）より前には回さない。
        with self.assertRaisesRegex(ValueError, "repo_spec_committed"):
            validate_config(draft, stage="gates")
        for stage in ("full", "analyze"):
            with self.assertRaisesRegex(ValueError, "preregistration is not frozen"):
                validate_config(draft, stage=stage)

    def test_frozen_preregistration_unblocks_the_run(self) -> None:
        """事前登録が成立したときだけ本走が通る。

        repo 側 spec は**実体**を要求する（フラグだけでは通らない）ので、
        テストは repo を汚さないよう ROOT を差し替えた仮想 repo で確認する。
        """
        frozen = copy.deepcopy(self.cfg)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = root / frozen["spec"]
            spec.parent.mkdir(parents=True)
            spec.write_text("# frozen spec\n", encoding="utf-8")
            frozen["preregistration"]["repo_spec_sha256"] = _sha_file(spec)
            with mock.patch.object(ident_module, "ROOT", str(root)):
                validate_config(frozen, stage="gates")
                validate_config(frozen, stage="full")
                # sha が動いたら（spec を差し替えたら）通らない。
                spec.write_text("# tampered\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "sha256 does not match"):
                    validate_config(frozen, stage="full")
        self.assertEqual(preregistration_missing(frozen), [])

    def test_generator_base_collision_is_caught_not_just_the_offset(self) -> None:
        _validate_offset(self.cfg)
        reused = copy.deepcopy(self.cfg)
        reused["common"]["generator_offset"] = 20_260_905
        with self.assertRaisesRegex(ValueError, "collides with an existing use"):
            _validate_offset(reused)
        # width 100 の本走と width 5 の既存走は offset 差 95 で同じ系列になる。
        shifted = copy.deepcopy(self.cfg)
        shifted["common"]["generator_offset"] = 20_260_810
        with self.assertRaisesRegex(ValueError, "generator base collision"):
            _validate_offset(shifted)

    def test_arm_table_matches_the_registered_purified_2x2(self) -> None:
        got = [(a["name"], a["visible"], a["code"], float(a["wd_b"]))
               for a in self.cfg["arms"]]
        self.assertEqual(got, [
            ("IM", "flip0", "flip_t", 1e-3), ("iM", "flip0", "zero", 1e-3),
            ("Im", "zero_centered", "flip_t", 1e-3),
            ("im", "zero_centered", "zero", 1e-3),
            (NOWD_ARM, "zero_centered", "zero", 0.0),
            (ANCHOR_ARM, "raw", "zero", 0.0)])
        broken = copy.deepcopy(self.cfg)
        broken["arms"][2]["code"] = "zero"        # (I+, M-) セルを潰す
        with self.assertRaisesRegex(ValueError, "registered arms differ"):
            validate_config(broken, stage="implementation")
        # D8: 要因 4 セルは全部 b-WD 下にある。1 本でも外れたら本走を止める。
        unpurified = copy.deepcopy(self.cfg)
        unpurified["arms"][3]["wd_b"] = 0.0
        with self.assertRaisesRegex(ValueError, "registered arms differ"):
            validate_config(unpurified, stage="implementation")
        # アンカーは逆に λ=0 でなければならない。
        decayed_anchor = copy.deepcopy(self.cfg)
        decayed_anchor["arms"][4]["wd_b"] = 1.0e-3
        with self.assertRaisesRegex(ValueError, "registered arms differ"):
            validate_config(decayed_anchor, stage="implementation")

    def test_s_op_probe_runs_in_the_same_regime_as_the_run(self) -> None:
        """Issa 裁定「regime を本走に合わせる」。T が本走と違えば起動しない。"""
        gates = self.cfg["ident_mu_2x2"]["gates"]
        self.assertEqual(gates["s_op_mode"], "switching_probe")
        self.assertEqual(int(gates["s_op_task_period"]),
                         int(self.cfg["phase1"]["task_period"]))
        self.assertGreaterEqual(
            int(gates["s_op_steps"]),
            (int(gates["s_op_n_boundaries"]) + 1) * int(gates["s_op_task_period"]))
        shortened = copy.deepcopy(self.cfg)
        shortened["ident_mu_2x2"]["gates"]["s_op_task_period"] = 1_000
        with self.assertRaisesRegex(ValueError, "registered S-op differs"):
            validate_config(shortened, stage="implementation")
        stale = copy.deepcopy(self.cfg)
        stale["ident_mu_2x2"]["gates"]["s_op_mode"] = "stationary_task"
        with self.assertRaisesRegex(ValueError, "registered S-op differs"):
            validate_config(stale, stage="implementation")

    def test_endpoint_registration_is_v2_drift_primary(self) -> None:
        P = self.cfg["ident_mu_2x2"]
        self.assertEqual(P["primary_endpoint"], "drift")
        self.assertEqual(P["onset"]["role"], "report_only")
        self.assertEqual(float(P["equivalence_margin"]), 0.15)
        self.assertEqual(float(P["interaction_margin"]), 0.50)
        demoted = copy.deepcopy(self.cfg)
        demoted["ident_mu_2x2"]["primary_endpoint"] = "onset"
        with self.assertRaisesRegex(ValueError, "registered endpoints differ"):
            validate_config(demoted, stage="implementation")


class HarnessTests(unittest.TestCase):
    """可視入力の構成とバイパスが spec §3.2 の表どおりであることの実走検査。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = load_config("configs/ident_mu_2x2_0901.yaml")

    def test_visible_input_matches_the_registered_table(self) -> None:
        for arm in ARM_ORDER:
            with self.subTest(arm=arm):
                st = _train(_setup(self.cfg, arm), 400)
                f = st["n_flip"]
                x = torch.cat([st["env"].flip_state,
                               torch.zeros(st["R"], st["d"] - f)], dim=1)
                visible = x - st["layer_means"][0]
                if arm in ("IM", "iM"):
                    self.assertTrue(torch.equal(visible[:, :f], st["flip0"]))
                elif arm in ("Im", "im", NOWD_ARM):
                    self.assertTrue(torch.equal(visible[:, :f],
                                                torch.zeros_like(st["flip0"])))
                else:
                    self.assertTrue(torch.equal(visible[:, :f],
                                                st["env"].flip_state))
                self.assertTrue(ident_run_record(st)["visible_ok"])

    def test_zero_code_arms_leave_the_bypass_exactly_at_zero(self) -> None:
        for arm, moves in (("IM", True), ("iM", False), ("Im", True),
                           ("im", False), (NOWD_ARM, False), (ANCHOR_ARM, False)):
            with self.subTest(arm=arm):
                st = _train(_setup(self.cfg, arm), 400)
                zero = bool(torch.equal(st["u"], torch.zeros_like(st["u"])))
                self.assertEqual(zero, not moves)

    def test_frozen_flip_columns_on_the_mu_minus_arms(self) -> None:
        st = _train(_setup(self.cfg, "im"), 400)
        self.assertTrue(torch.equal(st["net"].Ws[0][:, :, :st["n_flip"]],
                                    st["W_init_flip"]))
        st_plus = _train(_setup(self.cfg, "IM"), 400)
        self.assertFalse(torch.equal(st_plus["net"].Ws[0][:, :, :st_plus["n_flip"]],
                                     st_plus["W_init_flip"]))

    def test_b_weight_decay_only_touches_the_hidden_bias(self) -> None:
        """D8 の純化。`im` と `im_nowd` は λ だけが違う双子である。"""
        decayed = _train(_setup(self.cfg, "im"), 600)
        plain = _train(_setup(self.cfg, NOWD_ARM), 600)
        self.assertEqual(decayed["net"].wd_b, 1e-3)
        self.assertEqual(plain["net"].wd_b, 0.0)
        # b だけが違い、しかも WD 側の方が小さい。
        self.assertFalse(torch.equal(decayed["net"].bs[0], plain["net"].bs[0]))
        self.assertLess(float(decayed["net"].bs[0].abs().mean()),
                        float(plain["net"].bs[0].abs().mean()))
        # u は WD の対象外（どちらも code=0 なので厳密に 0 のまま）。
        self.assertTrue(torch.equal(decayed["u"], plain["u"]))

    def test_bypass_is_a_within_task_dc_so_it_cannot_move_unfit(self) -> None:
        """spec §4 の `bypass_share` を分散の分け前と読めない理由の実測。

        `code` は task 内で定数なので、バイパスは残差に定数を足すだけで
        `unfit`（32 パターン上の分散比）を動かさない。動くのは DC を見る
        `eval_loss_exact` の方である。
        """
        st = _train(_setup(self.cfg, "Im"), 2_000)
        mine = ident_run_record(st)
        _, frozen = exact_wall_record(st, float(self.cfg["phase1"]["sigma_degenerate_tol"]))
        delta = (mine["unfit"] - frozen["unfit"]).abs().max().item()
        self.assertLess(delta / float(frozen["unfit"].max().item()), 1e-10)
        self.assertGreater(float(mine["bypass_value"].abs().max().item()), 0.0)
        self.assertNotAlmostEqual(
            float(mine["eval_loss_exact"].max().item()),
            float(frozen["eval_loss_exact"].max().item()))


class OnsetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tasks = np.arange(1, 501)

    def _series(self, onset: int) -> np.ndarray:
        """初期過渡（閾値より上から始まる）→ 十分な当てはめ → 崩壊。"""
        values = np.full(500, -2.0)
        values[:5] = 0.5
        values[onset - 1:] = -0.5
        return values

    def test_initial_transient_does_not_count_as_onset(self) -> None:
        result = onset_task(self._series(300), self.tasks, threshold=-1.0,
                            window=10, censor_task=501, require_prior_below=True)
        # trailing の 10 task 中央値なので、階段状の崩壊は +5 task 遅れて出る。
        # 遅れは全腕に共通なので対比では相殺する（未来を見ないことの代金）。
        self.assertEqual(result["tau"], 305)
        self.assertEqual(result["first_below_task"], 11)   # 5 task の過渡が窓から抜けるまで
        self.assertFalse(result["censored"])
        self.assertFalse(result["never_below"])
        # 字義どおり「初めて閾値以上」だと初期過渡で tau=1 になる。
        naive = onset_task(self._series(300), self.tasks, threshold=-1.0,
                           window=10, censor_task=501, require_prior_below=False)
        self.assertEqual(naive["tau"], 1)

    def test_no_up_crossing_is_right_censored(self) -> None:
        result = onset_task(np.full(500, -2.0), self.tasks, threshold=-1.0,
                            window=10, censor_task=501, require_prior_below=True)
        self.assertEqual(result["tau"], 501)
        self.assertTrue(result["censored"])
        self.assertFalse(result["never_below"])

    def test_series_that_never_fits_is_flagged_not_scored_as_early(self) -> None:
        result = onset_task(np.full(500, 0.0), self.tasks, threshold=-1.0,
                            window=10, censor_task=501, require_prior_below=True)
        self.assertEqual(result["tau"], 501)
        self.assertTrue(result["never_below"])

    def test_moving_median_absorbs_a_single_spike(self) -> None:
        values = self._series(400)
        clean = onset_task(values, self.tasks, threshold=-1.0, window=10,
                           censor_task=501, require_prior_below=True)
        values[100] = 0.9                      # 1 点だけの跳ね
        spiked = onset_task(values, self.tasks, threshold=-1.0, window=10,
                            censor_task=501, require_prior_below=True)
        self.assertEqual(spiked["tau"], clean["tau"])


class DecisionTreeTests(unittest.TestCase):
    def test_band_classification(self) -> None:
        self.assertEqual(band_of({"ci_lo": -0.1, "ci_hi": 0.1}, 0.15), "IN")
        self.assertEqual(band_of({"ci_lo": 0.2, "ci_hi": 0.5}, 0.15), "OUT_POS")
        self.assertEqual(band_of({"ci_lo": -0.5, "ci_hi": -0.2}, 0.15), "OUT_NEG")
        self.assertEqual(band_of({"ci_lo": -0.2, "ci_hi": 0.5}, 0.15), "STRADDLE")

    def test_registered_decision_tree_branches(self) -> None:
        def bands(m1, m2, i1, i2):
            return {"M_i": m1, "M_ii": m2, "I_i": i1, "I_ii": i2}

        cases = [
            (bands("OUT_POS", "OUT_POS", "IN", "IN"), "IN", MU_DOMINANT),
            (bands("IN", "IN", "OUT_NEG", "OUT_NEG"), "IN", IDENT_DOMINANT),
            (bands("OUT_POS", "OUT_POS", "OUT_POS", "OUT_POS"), "IN", BOTH_MATTER),
            (bands("IN", "IN", "IN", "IN"), "IN", NEITHER_MATTERS),
            (bands("OUT_POS", "IN", "IN", "IN"), "IN", EFFECT_LEVEL_DEPENDENT),
            (bands("STRADDLE", "OUT_POS", "IN", "IN"), "IN", INCONCLUSIVE_WIDE),
            # 交互作用は他のどの枝よりも先に評価される。
            (bands("OUT_POS", "OUT_POS", "IN", "IN"), "OUT_NEG",
             INTERACTION_DOMINATES),
        ]
        for band_map, interaction, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(classify_2x2(band_map, interaction), expected)

    def test_s_op_identification_counts_prior_flips(self) -> None:
        """S-op が FAIL したとき「u が同定されていないだけ」を切り分ける診断。"""
        # 2 seed・境界 1..4。seed0 は bit 3 -> 3（2 回目は同定済み）。
        flipped = {1: np.array([3, 0]), 2: np.array([5, 1]),
                   3: np.array([3, 2]), 4: np.array([7, 0])}
        report = _s_op_identification(flipped, scored=[3, 4])
        self.assertEqual(report["prior_flips_of_the_flipped_bit"],
                         [[1, 0], [0, 1]])
        self.assertEqual(report["mean_prior_flips"], 0.5)
        self.assertEqual(report["frac_scored_boundaries_with_an_identified_bit"],
                         0.5)

    def test_r_ext_labels(self) -> None:
        thresholds = dict(prevents=8, persists=8, residual=1)
        cases = [
            ((0, 10), BWD_PREVENTS_EXTINCTION),
            ((1, 8), BWD_PREVENTS_EXTINCTION),
            ((10, 10), EXTINCTION_PERSISTS),
            ((8, 3), EXTINCTION_PERSISTS),
            ((5, 10), PARTIAL_RESCUE),
            ((0, 4), PARTIAL_RESCUE),       # 参照側が全滅しなければ救済とは言えない
        ]
        for (treat, control), expected in cases:
            with self.subTest(treat=treat, control=control):
                self.assertEqual(classify_r_ext(treat, control, **thresholds),
                                 expected)


def _synthetic_frame(seeds=range(10), tasks=500, ladder=False,
                     im_extinct=False) -> pd.DataFrame:
    """要因 4 セル ＋ アンカー 2 本の task 末表。

    既定では B02 が全腕 −2 dex で揃い、到達水準だけが腕ごとに違うので
    E-drift と E-level は同じ枝（`BOTH_MATTER`）に落ちる。

    * `ladder=True`: B02 側を 3.5 dex にわたって散らし、到達水準は揃える。
      E-level は `NEITHER_MATTERS`、E-drift は `BOTH_MATTER` になり、
      `CEILING_CONTAMINATED` と `LADDER_INVERTS` が同時に立つ。
    * `im_extinct=True`: b-WD 下の `im` も全滅させる（R-ext の反対側の枝）。
    """
    onsets = {"IM": 400, "iM": 300, "Im": 350, "im": 250,
              NOWD_ARM: 250, ANCHOR_ARM: 450}
    rows = []
    for arm, onset in onsets.items():
        # im_nowd は λ=0 の Aexact 双子なので全滅する側に置く。
        dead_max = 1.0 if (arm == NOWD_ARM or (im_extinct and arm == "im")) else 0.9
        for seed in seeds:
            shift = onset + 3 * (seed - 4.5) + 2 * np.sin(seed * 3 + onset)
            jitter = 0.02 * np.sin(seed * 7 + onset)
            if ladder:
                low = -2.0 - (onset - 250) / 150.0 * 3.5 + jitter
                high = 0.0
            else:
                low = -2.0
                high = -(onset - 250) / 150.0 * 0.5 + jitter
            for task in range(1, tasks + 1):
                gate = 1.0 / (1.0 + np.exp(-(task - shift) / 8.0))
                level = low + (high - low) * gate
                dead = float(np.round(dead_max * gate * 100) / 100)
                rows.append(dict(
                    arm=arm, seed=int(seed), step=task * 10_000, task=task,
                    wd_b=0.0 if arm in (NOWD_ARM, ANCHOR_ARM) else 1e-3,
                    unfit=10.0 ** level,
                    eval_loss_exact=10.0 ** level,
                    eval_loss_exact_nobypass=10.0 ** level,
                    residual_mean=0.0,
                    L1_strict_dead_frac=dead,
                    L1_alive=int(round(100 * (1.0 - dead))),
                    L1_submerged_frac=dead / 2,
                    L1_eff_rank=50.0 - 40.0 * dead,
                    L1_B_median_alive=-0.8 - dead,
                    L1_M_median_alive=0.2,
                    L1_b_median_all=-0.3 - dead,
                    L1_sigma_median_alive=0.5,
                    mu_norm_visible=0.0 if arm in ("Im", "im", NOWD_ARM) else 2.7,
                    u_norm=0.4 if arm in ("IM", "Im") else 0.0,
                    bypass_value=0.3 if arm in ("IM", "Im") else 0.0,
                    bypass_share=0.5 if arm in ("IM", "Im") else 0.0,
                    visible_ok=1))
    return pd.DataFrame(rows)


def _all_complete() -> dict:
    return {arm: dict(status="COMPLETE", included_seeds=list(range(10)),
                      excluded_seeds=[], exclusion_events=[], elapsed_sec=1.0,
                      sanity=dict(pass_=True))
            for arm in ARM_ORDER}


class AnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = copy.deepcopy(load_config("configs/ident_mu_2x2_0901.yaml"))
        cls.cfg["ident_mu_2x2"]["bootstrap_B"] = 2_000
        cls.frame = _synthetic_frame()

    def _analyze(self, frame: pd.DataFrame, meta: dict) -> tuple[dict, Path]:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        outdir = Path(tmp.name)
        frame.to_csv(outdir / "task_end_metrics.csv", index=False)
        return analyze(self.cfg, outdir, meta), outdir

    def test_block_levels_enforce_fifty_task_ends(self) -> None:
        levels = block_levels(self.cfg, self.frame)
        self.assertEqual(sorted(levels.block.unique()), list(range(1, 11)))
        self.assertTrue((levels.n_task_ends == 50).all())
        self.assertTrue((levels.floor_frac == 0.0).all())
        short = self.frame[self.frame.task != 7]
        with self.assertRaisesRegex(RuntimeError, "S-count failed"):
            block_levels(self.cfg, short)

    def test_onset_table_carries_all_three_thresholds(self) -> None:
        table = onset_table(self.cfg, self.frame)
        self.assertEqual(sorted(table.threshold.unique()), [-1.5, -1.0, -0.5])
        self.assertEqual(int(table.primary.sum()), len(ARM_ORDER) * 10)
        order = table[table.primary].groupby("arm").tau.median().sort_values()
        self.assertEqual(list(order.index)[-1], ANCHOR_ARM)
        self.assertEqual(set(list(order.index)[:2]), {"im", NOWD_ARM})

    def test_extinction_table_uses_both_rules(self) -> None:
        table = extinction_table(self.cfg, self.frame).set_index(["arm", "seed"])
        self.assertTrue(table.agree.all())
        self.assertTrue(bool(table.loc[(NOWD_ARM, 0), "extinct"]))
        self.assertFalse(bool(table.loc[("im", 0), "extinct"]))
        self.assertLess(int(table.loc[(NOWD_ARM, 0), "extinction_task"]), 501)

    def test_analyze_writes_the_registered_outputs(self) -> None:
        result, outdir = self._analyze(self.frame, _all_complete())
        for name in ("verdict.csv", "summary.md", "onset.csv", "extinction.csv",
                     "block_levels.csv", "paired_endpoints.csv",
                     "exclusions.csv", "fig_ident_mu_2x2.png"):
            self.assertTrue((outdir / name).exists(), name)
        self.assertEqual(result["n_paired"], 10)
        # 設計上 M も I も帯の外、交互作用は 0 なので BOTH_MATTER。
        self.assertEqual(result["main_verdict"], BOTH_MATTER)
        drift = result["details"]["drift"]
        self.assertEqual(drift["bands"], {"M_i": "OUT_NEG", "M_ii": "OUT_NEG",
                                          "I_i": "OUT_NEG", "I_ii": "OUT_NEG"})
        self.assertEqual(drift["interaction_band"], "IN")
        # v2 では E-drift が主で E-level は併記。両者が一致していれば旗は立たない。
        self.assertEqual(result["details"]["level"]["verdict"], BOTH_MATTER)
        self.assertFalse(result["details"]["ladder_inverts"])
        self.assertFalse(result["details"]["ceiling_contaminated"])
        verdict = pd.read_csv(outdir / "verdict.csv")
        self.assertIn("P-main", set(verdict.pred))
        self.assertIn("R-ext", set(verdict.pred))
        self.assertTrue((verdict[verdict.pred == "U"].verdict
                         == "REPORT_ONLY").all())
        self.assertTrue((verdict[verdict.pred == "D"].verdict
                         == "REPORT_ONLY").all())

    def test_r_ext_reads_extinction_not_function(self) -> None:
        result, _ = self._analyze(self.frame, _all_complete())
        r_ext = result["details"]["r_ext"]
        self.assertEqual(r_ext["verdict"], BWD_PREVENTS_EXTINCTION)
        self.assertEqual(r_ext["counts"], {"im": 0, NOWD_ARM: 10})
        self.assertEqual(r_ext["n_rule_disagreements"], 0)
        persists, _ = self._analyze(_synthetic_frame(im_extinct=True),
                                    _all_complete())
        self.assertEqual(persists["details"]["r_ext"]["verdict"],
                         EXTINCTION_PERSISTS)

    def test_r_ext_falls_back_when_the_anchor_is_invalid(self) -> None:
        meta = _all_complete()
        meta[NOWD_ARM].update(status="ARM_INVALID_EXCLUSION_LIMIT",
                              included_seeds=[], excluded_seeds=[0, 1, 2])
        result, _ = self._analyze(self.frame, meta)
        # アンカーが落ちても要因 4 セルの主判定は生きている。
        self.assertEqual(result["main_verdict"], BOTH_MATTER)
        self.assertEqual(result["details"]["r_ext"]["verdict"],
                         "R_EXT_INVALID_TOO_FEW_PAIRED")

    def test_spread_early_window_flags_ceiling_and_ladder(self) -> None:
        """B02 が 3 dex 超に散ると E-drift 単独では読めない（spec §6.1）。"""
        result, _ = self._analyze(_synthetic_frame(ladder=True), _all_complete())
        self.assertTrue(result["details"]["ceiling_contaminated"])
        self.assertGreater(result["details"]["b02_range"], 3.0)
        self.assertEqual(result["details"]["drift"]["verdict"], BOTH_MATTER)
        self.assertEqual(result["details"]["level"]["verdict"], NEITHER_MATTERS)
        self.assertEqual(result["main_verdict"], "LADDER_INVERTS")

    def test_floor_failure_invalidates_the_drift_endpoint(self) -> None:
        frame = self.frame.copy()
        floor = float(self.cfg["ident_mu_2x2"]["unfit_floor"])
        late = (frame.arm == "IM") & (frame.task > 450)
        frame.loc[late, "unfit"] = floor / 10.0
        result, _ = self._analyze(frame, _all_complete())
        self.assertFalse(result["details"]["floor_pass"])
        self.assertEqual(result["main_verdict"], "E_DRIFT_INVALID_FLOOR")

    def test_too_few_paired_seeds_blocks_the_contrast(self) -> None:
        meta = _all_complete()
        meta["im"]["included_seeds"] = [0, 1, 2, 3, 4, 5, 6]
        meta["im"]["status"] = "COMPLETE_WITH_EXCLUSIONS"
        frame = self.frame[~((self.frame.arm == "im")
                             & (self.frame.seed > 6))].copy()
        result, _ = self._analyze(frame, meta)
        self.assertEqual(result["main_verdict"], "CONTRAST_INVALID_TOO_FEW_PAIRED")

    def test_arm_invalid_stops_the_main_verdict(self) -> None:
        meta = _all_complete()
        meta["Im"].update(status="ARM_INVALID_EXCLUSION_LIMIT",
                          included_seeds=[], excluded_seeds=[0, 1, 2])
        result, _ = self._analyze(self.frame, meta)
        self.assertEqual(result["main_verdict"], "ARM_INVALID_EXCLUSION_LIMIT")


class GateArtifactTests(unittest.TestCase):
    """`--selftest` の出力が残っていれば、その中身も検査する。"""

    def setUp(self) -> None:
        cfg = load_config("configs/ident_mu_2x2_0901.yaml")
        self.gate_dir = Path(cfg["ident_mu_2x2"]["gate_dir"])
        if not (self.gate_dir / "s_bypass.json").exists():
            self.skipTest("selftests have not been run yet")

    def test_selftest_reports_pass(self) -> None:
        for name in ("s_pair.json", "s_seq.json", "s0_replay.json",
                     "s1_s2_algebra.json", "s1_s2.json", "s_mu.json", "s3.json",
                     "s_bypass.json", "s_freeze.json", "s_iso.json",
                     "s_cap.json", "s_count.json"):
            with self.subTest(gate=name):
                report = json.loads((self.gate_dir / name).read_text(encoding="utf-8"))
                self.assertTrue(report["pass_"], name)

    def test_s0_reproduces_the_committed_Aexact_trajectory(self) -> None:
        report = json.loads((self.gate_dir / "s0_replay.json").read_text(
            encoding="utf-8"))
        self.assertEqual(report["reference"], "L1w100_Aexact")
        self.assertEqual(report["init_state_differences"], [])
        self.assertEqual(report["differences"], [])
        self.assertEqual(max(report["max_rel_difference"].values()), 0.0)

    def test_s1_s2_keeps_the_bypass_out_of_the_weight_decay(self) -> None:
        report = json.loads((self.gate_dir / "s1_s2.json").read_text(
            encoding="utf-8"))
        self.assertTrue(report["frozen_algebra_pass"])
        arm = report["arm_level"]
        self.assertTrue(arm["W_v_c_untouched"])
        self.assertTrue(arm["bypass_u_untouched"])
        self.assertLessEqual(arm["bias_delta_max_abs_err"],
                             arm["bias_delta_tol_ulp"])
        self.assertGreater(arm["bias_delta_signal"], 0.0)

    def test_isolation_gate_actually_guards_the_bypass(self) -> None:
        report = json.loads((self.gate_dir / "s_iso.json").read_text(encoding="utf-8"))
        self.assertEqual(report["injected"], "bypass.u")
        self.assertEqual(report["nonfinite_tensors"], {"1": ["bypass.u"]})
        self.assertTrue(report["unaffected_state_bitwise_equal"])


if __name__ == "__main__":
    unittest.main()
