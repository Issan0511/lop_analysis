"""edge_law_0905 runner の単体テスト（spec `specs/spec_edge_law_0905.md` §3・§5）。

    OMP_NUM_THREADS=1 PYTHONPATH=. python3 -m unittest src.test_edge_law_runner_0905 -v

様式は ``test_weird_act_0903`` / ``test_gate_dial_0902`` に倣う（unittest・pytest 非依存）。
**すべての bit 一致検査に「変異体が FAIL する」対を必ず置く**（本プロジェクトで一度
「テンソルを自分自身と比べる空虚な S 検査」をやった失敗の再発防止）。
"""
from __future__ import annotations

import copy
import inspect
import json
import shutil
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import torch

from src import edge_law_0905 as E
from src import gate_dial_0902 as HOST_DIAL
from src import gate_dose as HOST_GATE
from src import weird_act_0903 as HOST_WEIRD
from src.common import ROOT, load_config
from src.nets import VecMLPL
from src.ratchet_log import full_support_ro

CFG_EDGE = load_config(str(E.CONFIG))
SHORT_STEPS = 20_000
SHORT_SEEDS = list(range(10))   # seed はベクトル化されているので部分集合にしない
_SHARED: dict = {}


def _cfg() -> dict:
    if "cfg" not in _SHARED:
        _SHARED["cfg"] = E.build_cfg()
    return copy.deepcopy(_SHARED["cfg"])


def _short_run(arm: str = "LRnull_1216") -> Path:
    """腕を 20,000 step（2 seed）だけ走らせた logs を作り、テスト間で使い回す。"""
    key = f"run:{arm}"
    if key not in _SHARED:
        out = Path(_SHARED["tmp"]) / arm
        E.run_single_arm(arm, steps=SHORT_STEPS, outdir=out, seeds=SHORT_SEEDS,
                         cfg=_cfg())
        _SHARED[key] = out
    return _SHARED[key]


def setUpModule() -> None:
    _SHARED["tmp"] = tempfile.mkdtemp(prefix="edge_law_0905_test_")


def tearDownModule() -> None:
    shutil.rmtree(_SHARED["tmp"], ignore_errors=True)


# ---------------------------------------------------------------------------
# config / 腕表（spec §3.1）
# ---------------------------------------------------------------------------
class ConfigTests(unittest.TestCase):
    def test_thirty_arms_are_taken_from_the_config_verbatim(self):
        table = E.table()
        self.assertEqual(len(table), 30)
        self.assertEqual(list(table), [str(a["name"]) for a in CFG_EDGE["arms"]])
        for row in CFG_EDGE["arms"]:
            got = table[str(row["name"])]
            self.assertEqual(got["family"], str(row["family"]))
            self.assertEqual(got["activation"], str(row["activation"]))
            self.assertEqual(got["dial"], float(row["dial"]))
            self.assertEqual(got["total_steps"], int(row["total_steps"]))
            self.assertEqual(got["checkpoints"],
                             [int(v) for v in (row["checkpoints"] or [])])

    def test_build_cfg_appends_the_new_arms_to_the_host_without_touching_it(self):
        host = load_config(str(E.HOST_CONFIG))
        cfg = _cfg()
        host_names = [a["name"] for a in host["arms"]]
        self.assertEqual([a["name"] for a in cfg["arms"]][:len(host_names)],
                         host_names)
        self.assertEqual(len(cfg["arms"]), len(host_names) + 30)
        # 宿主の腕ブロックは 1 つも書き換わっていない
        for a, b in zip(host["arms"], cfg["arms"]):
            self.assertEqual(a, b)

    def test_every_new_activation_label_is_registered_by_setdefault(self):
        cfg = _cfg()
        host = load_config(str(E.HOST_CONFIG))
        for name, block in CFG_EDGE["activation"].items():
            if not isinstance(block, dict):
                continue
            self.assertIn(name, cfg["activation"])
            self.assertEqual(cfg["activation"][name]["name"], block["name"])
        # setdefault なので宿主の elu ブロック（derivative_form 付き）は残る
        self.assertEqual(cfg["activation"]["elu"], host["activation"]["elu"])

    def test_common_overrides_are_applied(self):
        cfg, ov = _cfg(), CFG_EDGE["common_overrides"]
        self.assertEqual(cfg["common"]["lr_main"], float(ov["lr_main"]))
        self.assertEqual(cfg["common"]["seeds"], [int(v) for v in ov["seeds"]])
        self.assertEqual(cfg["common"]["generator_offset"],
                         int(ov["generator_offset"]))
        self.assertEqual(cfg["sanity"]["omp_num_threads"],
                         int(ov["omp_num_threads"]))

    def test_every_arm_resolves_to_a_registered_activation_and_dial(self):
        cfg = _cfg()
        for arm in E.arm_order():
            act, alpha = E._activation(cfg, E._arm(cfg, arm))
            self.assertIn(act, VecMLPL.ACTIVATIONS, arm)
            VecMLPL(1, [2], 2, torch.Generator().manual_seed(0), "cpu") \
                .set_activation(act, alpha, "alpha_exp")

    def test_the_s_null_arm_block_matches_the_reference_arm_geometry(self):
        """S-null は腕ブロックの逐語一致が前提（spec §5 S-null 行）。"""
        cfg = _cfg()
        blk = E._arm(cfg, "LRnull_1216")
        act, alpha = E._activation(cfg, blk)
        self.assertEqual((act, alpha), ("leaky_relu", 0.1))
        self.assertEqual(blk["family"], "leaky")
        self.assertEqual([int(v) for v in blk["hidden"]], [100])
        self.assertEqual([int(v) for v in blk["centered_layers"]], [1])
        self.assertEqual(float(blk["target_mu_norm"]), 3.041)
        self.assertEqual(float(blk["target_dose"]), 12.16)
        self.assertIsNone(blk["u_star"])

    def test_the_host_validate_config_is_deliberately_not_called(self):
        """宿主の validate_config は 14 腕を逐語照合するので通せない（spec §3.1）。"""
        with self.assertRaises(Exception):
            HOST_DIAL.validate_config(_cfg(), stage="run")

    def test_the_analysis_module_and_the_runner_agree_on_column_names(self):
        """リテラルだけの弱い版（実ログを正本にする版は `ColumnAgreementTests`）。"""
        got = E.expected_column_agreement()
        self.assertTrue(got["pass_"], got)
        self.assertNotIn("skipped", got)
        self.assertFalse(got["from_real_logs"])


# ---------------------------------------------------------------------------
# init フック（spec §3.3・S-hook-inplace / S-hook-noop）
# ---------------------------------------------------------------------------
class HookTests(unittest.TestCase):
    def test_s_hook_inplace_passes_for_every_registered_hook(self):
        got = E.s_hook_inplace(_cfg())
        self.assertTrue(got["pass_"], json.dumps(got, ensure_ascii=False))
        self.assertEqual(len(got["rows"]),
                         sum(1 for r in E.table().values() if r["hook"]))

    def test_s_hook_inplace_catches_a_hook_that_rebinds_instead_of_mutating(self):
        """別名を殺す「代入版 negate」は S-hook-inplace の要求を破る。"""
        cfg = _cfg()
        st = HOST_DIAL.setup_arm_dial(cfg, E._arm(cfg, "FLn_1216"), "cpu")
        net = st["net"]
        net.Ws[0] = -net.Ws[0]                     # ← やってはいけない書き方
        self.assertFalse(net.W is net.Ws[0])
        with self.assertRaises(HOST_DIAL.SanityError):
            E._check_aliases(net, "FLn_1216")

    def test_s_hook_noop_is_byte_identical_and_negate_makes_it_fail(self):
        got = E.s_hook_noop(cfg=_cfg())
        self.assertTrue(got["pass_"], got)
        self.assertEqual(got["differing"], [])
        self.assertEqual(got["negate_differing"], ["W", "b", "v"])

    def test_negate_is_an_exact_sign_flip_of_W_b_v_and_leaves_c(self):
        cfg = _cfg()
        blk = E._arm(cfg, "FLn_1216")
        base = HOST_DIAL.setup_arm_dial(cfg, blk, "cpu")
        w0 = base["net"].Ws[0].clone()
        v0 = base["net"].v.clone()
        c0 = base["net"].c.clone()
        st = E._setup_with_hook(cfg, blk, "cpu", {"type": "negate"})
        net = st["net"]
        self.assertEqual((0.0 - w0.numpy()).tobytes(), net.Ws[0].numpy().tobytes())
        self.assertEqual((0.0 - v0.numpy()).tobytes(), net.v.numpy().tobytes())
        self.assertEqual(c0.numpy().tobytes(), net.c.numpy().tobytes())
        self.assertEqual(st["init_hook"], "negate")
        self.assertTrue(np.isnan(st["init_hook_arg"]))

    def test_b_offset_adds_the_registered_constant_only_to_the_hidden_bias(self):
        cfg = _cfg()
        for arm, want in (("LRbp5_1216", 5.0), ("LRbm5_1216", -5.0),
                          ("Ebp4_1216", 4.0), ("Ebm4_1216", -4.0)):
            with self.subTest(arm=arm):
                blk = E._arm(cfg, arm)
                base = HOST_DIAL.setup_arm_dial(cfg, blk, "cpu")
                b0, w0 = base["net"].bs[0].clone(), base["net"].Ws[0].clone()
                st = E._setup_with_hook(cfg, blk, "cpu", E._hook_of(arm))
                self.assertTrue(torch.equal(st["net"].bs[0], b0 + want))
                self.assertTrue(torch.equal(st["net"].Ws[0], w0))
                self.assertEqual(st["init_hook_arg"], want)

    def test_scale_is_function_preserving_for_the_positively_homogeneous_arms(self):
        """leaky は正斉次なので、s 倍した net は同じ x に対し同じ出力を返す。"""
        cfg = _cfg()
        for arm, s in (("LRs0p5_1216", 0.5), ("LRs2_1216", 2.0)):
            with self.subTest(arm=arm):
                blk = E._arm(cfg, arm)
                base = HOST_DIAL.setup_arm_dial(cfg, blk, "cpu")
                st = E._setup_with_hook(cfg, blk, "cpu", E._hook_of(arm))
                x = full_support_ro(base["env"])[0] - base["layer_means"][0].float()
                _, _, y0 = base["net"].forward_layers(x)
                _, _, y1 = st["net"].forward_layers(x)
                self.assertTrue(torch.allclose(y0, y1, atol=1e-5), (arm, y0, y1))
                self.assertTrue(torch.equal(st["net"].Ws[0], base["net"].Ws[0] * s))
                self.assertTrue(torch.equal(st["net"].v, base["net"].v / s))

    def test_lr_hook_rewrites_both_the_tensor_and_every_run_record(self):
        cfg = _cfg()
        for arm, eta in (("LRlr0p005_1216", 0.005), ("Elr0p02_1216", 0.02)):
            with self.subTest(arm=arm):
                st = E._setup_with_hook(cfg, E._arm(cfg, arm), "cpu",
                                        E._hook_of(arm))
                self.assertTrue(torch.equal(st["lr"],
                                            torch.full_like(st["lr"], eta)))
                self.assertTrue(all(float(r["lr"]) == eta for r in st["runs"]))
                self.assertEqual(st["lr_used"], eta)

    def test_v_freeze_multiplies_v_and_sets_the_flag(self):
        cfg = _cfg()
        for arm, m in (("Evf1_1216", 1.0), ("Evf4_1216", 4.0)):
            with self.subTest(arm=arm):
                base = HOST_DIAL.setup_arm_dial(cfg, E._arm(cfg, arm), "cpu")
                v0 = base["net"].v.clone()
                st = E._setup_with_hook(cfg, E._arm(cfg, arm), "cpu",
                                        E._hook_of(arm))
                self.assertTrue(torch.equal(st["net"].v, v0 * m))
                self.assertTrue(st["freeze_v"])
                self.assertEqual(st["batch_mode"], "online")

    def test_full_batch_hook_only_sets_the_mode(self):
        cfg = _cfg()
        base = HOST_DIAL.setup_arm_dial(cfg, E._arm(cfg, "FBLR_1216"), "cpu")
        st = E._setup_with_hook(cfg, E._arm(cfg, "FBLR_1216"), "cpu",
                                E._hook_of("FBLR_1216"))
        self.assertEqual(st["batch_mode"], "full32")
        self.assertFalse(st["freeze_v"])
        self.assertTrue(torch.equal(st["net"].Ws[0], base["net"].Ws[0]))

    def test_payload_defaults_are_written_even_without_a_hook(self):
        cfg = _cfg()
        st = E._setup_with_hook(cfg, E._arm(cfg, "LIN_1216"), "cpu", None)
        self.assertEqual(st["init_hook"], "")
        self.assertTrue(np.isnan(st["init_hook_arg"]))
        self.assertEqual(st["lr_used"], float(cfg["common"]["lr_main"]))
        self.assertFalse(st["freeze_v"])
        self.assertEqual(st["batch_mode"], "online")

    def test_an_unknown_hook_type_raises(self):
        cfg = _cfg()
        st = HOST_DIAL.setup_arm_dial(cfg, E._arm(cfg, "LIN_1216"), "cpu")
        with self.assertRaises(ValueError):
            E._apply_hook(st, {"type": "not_a_hook"}, "LIN_1216")


# ---------------------------------------------------------------------------
# S-copy（spec §5）
# ---------------------------------------------------------------------------
class CopyTests(unittest.TestCase):
    #: 登録した挿入行（spec §5 に逐語で載せる文字列）
    REGISTERED_RUN = (("_apply_hook(st, _hook_of(arm), arm)",),
                      ("WeirdRecorder = EdgeRecorder",
                       "train_arm_gate = _train_fn(st)",
                       "write_arm_logs_dial = write_arm_logs_edge"))
    REGISTERED_TRAIN = ('if st.get("freeze_v"):',
                        "grads = (grads[0], grads[1], "
                        "torch.zeros_like(grads[2]), grads[3])")

    def test_s_copy_passes_and_the_inserted_lines_are_the_registered_ones(self):
        got = E.s_copy()
        self.assertTrue(got["pass_"], json.dumps(got, ensure_ascii=False))
        self.assertEqual([tuple(b) for b in got["run_arm"]["inserted"]],
                         list(self.REGISTERED_RUN))
        self.assertEqual([tuple(b) for b in got["train_arm"]["inserted"]],
                         [self.REGISTERED_TRAIN])
        self.assertEqual(got["run_arm"]["unregistered_opcodes"], [])
        self.assertEqual(got["train_arm"]["unregistered_opcodes"], [])

    def test_the_registered_inserts_match_the_module_constants(self):
        self.assertEqual(tuple(tuple(b) for b in E.RUN_INSERTS),
                         self.REGISTERED_RUN)
        self.assertEqual(tuple(E.TRAIN_INSERTS), self.REGISTERED_TRAIN)

    def test_an_unregistered_edit_makes_s_copy_fail(self):
        """変異体が FAIL することが、この検査が生きている証拠。"""
        def mutant(cfg: dict, arm: str, device: str, outdir, seeds, total) -> dict:
            c = copy.deepcopy(cfg)
            c["common"]["seeds"] = seeds
            st = HOST_DIAL.setup_arm_dial(c, E._arm(c, arm), device)
            every = int(c["common"]["lop_every"]) + 1        # ← 未登録の書き換え
            return dict(st=st, every=every)

        got = E._copy_opcodes(HOST_WEIRD._run_arm_weird, mutant, E.RUN_INSERTS)
        self.assertFalse(got["pass_"])

    def test_an_unregistered_extra_line_makes_s_copy_fail(self):
        def mutant(st: dict, recorder, probe_steps, total: int, outdir,
                   checkpoints, stream_hook=None) -> float:
            probe_set = {int(v) for v in probe_steps}
            checkpoint_set = {int(v) for v in checkpoints}
            net, env, teacher = st["net"], st["env"], st["teacher"]
            started = time.time()                            # ← 追加行（未登録）
            extra = 1
            return float(len(probe_set) + len(checkpoint_set) + extra)

        got = E._copy_opcodes(HOST_GATE.train_arm_gate, mutant,
                              (E.TRAIN_INSERTS,))
        self.assertFalse(got["pass_"])

    def test_train_fn_returns_the_host_function_itself_for_the_plain_arms(self):
        """字下げを見ない S-copy の穴は、ここ（同一オブジェクト）で塞ぐ。"""
        cfg = _cfg()
        plain, special = [], []
        for arm in E.arm_order():
            hook = E._hook_of(arm)
            st = dict(batch_mode="online", freeze_v=False)
            if hook is not None and hook["type"] == "v_freeze":
                st["freeze_v"] = True
            if hook is not None and hook["type"] == "full_batch":
                st["batch_mode"] = "full32"
            fn = E._train_fn(st)
            (special if hook and hook["type"] in ("v_freeze", "full_batch")
             else plain).append((arm, fn))
        self.assertEqual(len(plain), 27)
        self.assertEqual(len(special), 3)
        for arm, fn in plain:
            self.assertIs(fn, HOST_GATE.train_arm_gate, arm)
        for arm, fn in special:
            self.assertIsNot(fn, HOST_GATE.train_arm_gate, arm)
        self.assertIs(E._train_fn({"freeze_v": True}), E.train_arm_edge)
        self.assertIs(E._train_fn({"batch_mode": "full32"}),
                      E.train_arm_full_batch)


# ---------------------------------------------------------------------------
# S-stream（spec §5）
# ---------------------------------------------------------------------------
class StreamTests(unittest.TestCase):
    def test_all_thirty_arms_share_the_init_and_the_first_hundred_batches(self):
        got = E.s_stream(_cfg())
        self.assertTrue(got["pass_"], [r for r in got["rows"] if r.get("differing")])
        self.assertEqual(len(got["rows"]), 30)
        self.assertEqual(got["n_batches"], 100)

    def test_a_seed_subset_is_a_different_input_stream(self):
        """S-par の境界（本走で `--seeds` を分割してはいけない理由）を検査に残す。"""
        got = E.s_seed_split_note(cfg=_cfg())
        self.assertTrue(got["pass_"], got)
        self.assertTrue(got["init_rows_match"])       # init/教師は行の切り出しで一致
        self.assertTrue(got["initial_flip_rows_match"])
        self.assertFalse(any(got["input_batches_match"]))
        # 陽性対照: 同じ 10 seed 構成なら入力列は一致する（差だけでなく等しさも
        # 見えることの証明。これが無いと「常に不一致」の実装と区別できない）。
        self.assertTrue(all(got["identical_config_batches_match"]))
        self.assertEqual(len(got["identical_config_batches_match"]),
                         len(got["input_batches_match"]))

    def test_the_split_note_fails_when_the_subset_is_not_a_subset(self):
        """陰性対照: seeds が 5 本しかない config なら「半分」は全体と同じ入力列に
        なるので、`not any(same)` の節が働いて `pass_` は False でなければならない
        （この節を落としても緑のままだった穴を塞ぐ）。"""
        cfg = _cfg()
        cfg["common"]["seeds"] = [0, 1, 2, 3, 4]
        got = E.s_seed_split_note(cfg=cfg)
        self.assertTrue(all(got["input_batches_match"]), got)
        self.assertFalse(got["pass_"], got)

    def test_the_stream_fingerprint_is_not_vacuous(self):
        """generator_offset を動かせば同じ比較が必ず FAIL する。"""
        cfg = _cfg()
        other = copy.deepcopy(cfg)
        other["common"]["generator_offset"] = 1
        a = E._state_fingerprint(
            HOST_DIAL.setup_arm_dial(cfg, E._arm(cfg, "LRnull_1216"), "cpu"))
        b = E._state_fingerprint(
            HOST_DIAL.setup_arm_dial(other, E._arm(other, "LRnull_1216"), "cpu"))
        self.assertNotEqual(a["W"], b["W"])
        self.assertNotEqual(a["flip_state"], b["flip_state"])


# ---------------------------------------------------------------------------
# recorder（spec §3.4）
# ---------------------------------------------------------------------------
class RecorderTests(unittest.TestCase):
    def test_the_free_bits_are_the_last_five_input_columns(self):
        """`envs.SCREnv` は cat([flip_state (15), rnd (5)]) を返す（w_free の根拠）。"""
        cfg = _cfg()
        st = HOST_DIAL.setup_arm_dial(cfg, E._arm(cfg, "LRnull_1216"), "cpu")
        X = full_support_ro(st["env"])
        self.assertEqual(tuple(X.shape), (32, st["R"], 20))
        flip = st["env"].flip_state
        for p in range(32):
            self.assertTrue(torch.equal(X[p, :, :15], flip))
        varies = (X[:, :, 15:20].amax(dim=0) - X[:, :, 15:20].amin(dim=0))
        self.assertTrue(bool((varies == 1.0).all()))
        self.assertEqual(E.FREE_SLICE, slice(15, 20))

    def test_recorder_grids_follow_the_registered_frequencies(self):
        cfg = _cfg()
        st = HOST_DIAL.setup_arm_dial(cfg, E._arm(cfg, "LRnull_1216"), "cpu")
        E._apply_hook(st, None, "LRnull_1216")
        total = 300_000
        probes = list(range(0, total + 1, 1000))
        rec = E.EdgeRecorder(probes, st)
        self.assertTrue(bool((rec.w_free_steps % E.PERIOD == 0).all()))
        self.assertEqual(len(rec.w_free_steps), total // E.PERIOD + 1)
        dense_from = total - 20 * E.PERIOD
        want = sorted(set(s for s in probes
                          if s % E.PERIOD == 0 or s > dense_from))
        self.assertEqual(list(rec.moment_steps), want)
        self.assertEqual(rec.w_free.shape,
                         (len(rec.w_free_steps), st["R"], 100, 5))
        for key in E.MOMENT_KEYS:
            self.assertEqual(rec.moments[key].shape,
                             (len(rec.moment_steps), st["R"], 100))

    def test_recorded_w_free_is_exactly_the_last_five_weight_columns(self):
        cfg = _cfg()
        st = HOST_DIAL.setup_arm_dial(cfg, E._arm(cfg, "SH_d2_1216"), "cpu")
        E._apply_hook(st, None, "SH_d2_1216")
        rec = E.EdgeRecorder([0, 1000], st)
        rec(st, 0)
        want = st["net"].Ws[0][:, :, 15:20].numpy().astype(np.float32)
        self.assertEqual(rec.w_free[0].tobytes(), want.tobytes())

    def test_moments_match_an_independent_float64_recomputation(self):
        """recorder とは別の道（zbar と w_free からの支持復元）で計算し直す。"""
        cfg = _cfg()
        for arm in ("LRnull_1216", "Enull_1216", "SH_d2_1216", "SP_1216",
                    "TH_1216", "ST_d1_1216", "FL_1216"):
            with self.subTest(arm=arm):
                st = HOST_DIAL.setup_arm_dial(cfg, E._arm(cfg, arm), "cpu")
                E._apply_hook(st, E._hook_of(arm), arm)
                rec = E.EdgeRecorder([0, 1000], st)
                rec(st, 0)
                extra = HOST_DIAL.unit_extra_record(st)
                zbar = extra["zmean"].numpy()
                wf = st["net"].Ws[0][:, :, 15:20].double().numpy()
                z = torch.from_numpy(E._support_from_log(zbar, wf))
                net = st["net"]
                phi = net.act_fn(z)
                dphi = net.act_grad(z, phi)
                ddphi = net.act_curv(z)
                ref = {"m_phi2": (phi * phi).mean(dim=0),
                       "m_dphi2": (dphi * dphi).mean(dim=0),
                       "m_phidphi": (phi * dphi).mean(dim=0),
                       "m_dphiddphi": (dphi * ddphi).mean(dim=0)}
                for key in E.MOMENT_KEYS:
                    a = ref[key].numpy()
                    b = rec.moments[key][0]
                    scale = max(float(np.abs(a).max()), 1e-12)
                    self.assertLess(float(np.abs(a - b).max() / scale), 1e-5,
                                    (arm, key))

    def test_the_moment_recomputation_is_not_vacuous(self):
        """支持を 1 点ずらせば同じ比較が落ちる（恒真でないことの証拠）。"""
        cfg = _cfg()
        st = HOST_DIAL.setup_arm_dial(cfg, E._arm(cfg, "Enull_1216"), "cpu")
        E._apply_hook(st, None, "Enull_1216")
        rec = E.EdgeRecorder([0, 1000], st)
        rec(st, 0)
        extra = HOST_DIAL.unit_extra_record(st)
        zbar = extra["zmean"].numpy() - 0.5                    # ← ずらした支持
        wf = st["net"].Ws[0][:, :, 15:20].double().numpy()
        z = torch.from_numpy(E._support_from_log(zbar, wf))
        phi = st["net"].act_fn(z)
        a = (phi * phi).mean(dim=0).numpy()
        b = rec.moments["m_phi2"][0]
        self.assertGreater(float(np.abs(a - b).max()), 1e-3)

    def test_every_registered_arm_can_record_all_four_moments(self):
        """30 腕すべてで `act_curv` が登録済みであること（未登録なら例外で落ちる）。"""
        cfg = _cfg()
        for arm in E.arm_order():
            with self.subTest(arm=arm):
                st = HOST_DIAL.setup_arm_dial(cfg, E._arm(cfg, arm), "cpu")
                E._apply_hook(st, E._hook_of(arm), arm)
                rec = E.EdgeRecorder([0, 1000], st)
                rec(st, 0)
                for key in E.MOMENT_KEYS:
                    self.assertTrue(np.isfinite(rec.moments[key][0]).all())
                # 区分線形族は phi'' == 0（厳密）。滑らかな族は 0 でない — ただし
                # b_offset +4/+5 の腕は支持が全部 z>0 側に出るので ELU の phi'' は
                # そこでも 0 になる（例外として除く）。
                zero = float(np.abs(rec.moments["m_dphiddphi"][0]).max()) == 0.0
                smooth = st["activation"] in ("elu", "softplus_b", "tanh_b")
                hook = E._hook_of(arm)
                shifted_up = bool(hook and hook["type"] == "b_offset"
                                  and float(hook["value"]) > 0)
                if not smooth:
                    self.assertTrue(zero, st["activation"])
                elif not shifted_up:
                    self.assertFalse(zero, st["activation"])

    def test_an_activation_without_act_curv_would_raise_here(self):
        """`act_curv` の未登録名で落ちること（黙って ELU にならない）。"""
        cfg = _cfg()
        st = HOST_DIAL.setup_arm_dial(cfg, E._arm(cfg, "LRnull_1216"), "cpu")
        E._apply_hook(st, None, "LRnull_1216")
        st["net"].set_activation("silu", 1.0, "alpha_exp")
        with self.assertRaises(NotImplementedError):
            E.unit_moment_record(st)

    def test_recorder_does_not_advance_the_environment(self):
        cfg = _cfg()
        st = HOST_DIAL.setup_arm_dial(cfg, E._arm(cfg, "LRnull_1216"), "cpu")
        E._apply_hook(st, None, "LRnull_1216")
        env = st["env"]
        before = (env.t, env.flip_state.clone(),
                  env.gen.get_state().clone())
        rec = E.EdgeRecorder([0, 1000], st)
        rec(st, 0)
        self.assertEqual(env.t, before[0])
        self.assertTrue(torch.equal(env.flip_state, before[1]))
        self.assertTrue(torch.equal(env.gen.get_state(), before[2]))


# ---------------------------------------------------------------------------
# 書き出し（既存列を 1 列も変えない）
# ---------------------------------------------------------------------------
class WriterTests(unittest.TestCase):
    def test_the_shared_columns_are_byte_identical_to_the_host_writer(self):
        logdir = _short_run()
        tmp = Path(_SHARED["tmp"]) / "writer"
        cfg = _cfg()
        cfg["common"]["seeds"] = SHORT_SEEDS
        st = HOST_DIAL.setup_arm_dial(cfg, E._arm(cfg, "LRnull_1216"), "cpu")
        E._apply_hook(st, None, "LRnull_1216")
        rec = E.EdgeRecorder([0, 1000], st)
        rec(st, 0)
        rec(st, 1000)
        HOST_DIAL.write_arm_logs_dial(tmp / "host", "LRnull_1216", st, rec)
        E.write_arm_logs_edge(tmp / "mine", "LRnull_1216", st, rec)
        for seed in SHORT_SEEDS:
            a = np.load(tmp / "host" / "logs" / f"LRnull_1216_seed{seed}.npz",
                        allow_pickle=False)
            b = np.load(tmp / "mine" / "logs" / f"LRnull_1216_seed{seed}.npz",
                        allow_pickle=False)
            self.assertTrue(set(a.files) < set(b.files))
            for key in a.files:
                self.assertEqual(a[key].dtype, b[key].dtype, key)
                self.assertEqual(a[key].tobytes(), b[key].tobytes(), key)
            new = set(b.files) - set(a.files)
            self.assertEqual(new, {"layer1_w_free", "layer1_w_free_step",
                                   "layer1_moment_step", "init_hook",
                                   "init_hook_arg", "lr_used", "freeze_v",
                                   "batch_mode"}
                             | {f"layer1_{k}" for k in E.MOMENT_KEYS})
        self.assertTrue(logdir.exists())

    def test_payload_scalars_land_in_the_npz_with_the_expected_types(self):
        logdir = _short_run()
        with np.load(logdir / "logs" / "LRnull_1216_seed0.npz",
                     allow_pickle=False) as z:
            self.assertEqual(str(z["init_hook"]), "")
            self.assertTrue(np.isnan(float(z["init_hook_arg"])))
            self.assertEqual(float(z["lr_used"]), 0.01)
            self.assertEqual(bool(z["freeze_v"]), False)
            self.assertEqual(str(z["batch_mode"]), "online")

    def test_every_column_the_analysis_needs_is_present_with_the_right_shape(self):
        from src.edge_law_analyze_0905 import expected_columns
        want = expected_columns()
        logdir = _short_run()
        with np.load(logdir / "logs" / "LRnull_1216_seed0.npz",
                     allow_pickle=False) as z:
            files, n = set(z.files), len(z["step"])
            for key in want["run"] + want["unit"] + want["new_unit"]:
                self.assertIn(key, files)
            for key in want["payload"]:
                self.assertIn(key, files)
            for key, step_key in want["new_aux"]:
                self.assertIn(key, files)
                self.assertIn(step_key, files)
                self.assertEqual(z[key].shape[0], z[step_key].shape[0])
            for key in want["unit"]:
                self.assertEqual(z[key].shape, (n, 100), key)
            self.assertEqual(z["layer1_zmin"].shape, (n, 100))
            self.assertEqual(z["layer1_w_free"].shape[1:], (100, 5))
            for key in E.MOMENT_KEYS:
                self.assertEqual(z[f"layer1_{key}"].shape[1:], (100,))


# ---------------------------------------------------------------------------
# S-support / S-moment / S-C / S-lr（短縮走行の上で）
# ---------------------------------------------------------------------------
class ShortRunSanityTests(unittest.TestCase):
    def test_s_support(self):
        got = E.s_support(_short_run() / "logs", "LRnull_1216", seeds=SHORT_SEEDS)
        self.assertTrue(got["pass_"], got)

    def test_s_support_is_not_vacuous(self):
        """zmin を 1 だけずらした npz を作れば S-support は落ちる。"""
        src = _short_run() / "logs" / "LRnull_1216_seed0.npz"
        dst_dir = Path(_SHARED["tmp"]) / "mutant"
        dst_dir.mkdir(parents=True, exist_ok=True)
        with np.load(src, allow_pickle=True) as z:
            payload = {k: z[k] for k in z.files}
        payload["layer1_zmin"] = payload["layer1_zmin"] - 1.0
        np.savez_compressed(dst_dir / "LRnull_1216_seed0.npz", **payload)
        got = E.s_support(dst_dir, "LRnull_1216", seeds=(0,))
        self.assertFalse(got["pass_"])

    def test_s_moment(self):
        got = E.s_moment(_short_run() / "logs", "LRnull_1216", seeds=SHORT_SEEDS)
        self.assertTrue(got["pass_"], got)

    def test_s_C(self):
        got = E.s_C(_short_run() / "logs", "LRnull_1216", seeds=SHORT_SEEDS)
        self.assertTrue(got["pass_"], got)
        for row in got["rows"]:
            self.assertLess(row["max_abs_dev"], 1e-12)

    def test_s_C_is_not_vacuous(self):
        """C を閉形式から動かせば落ちる（同じ定数を 2 回計算する恒真検査ではない）。"""
        got = E.s_C(_short_run() / "logs", "LRnull_1216", seeds=(0,), tol=0.0)
        self.assertFalse(got["pass_"])

    def test_s_lr(self):
        got = E.s_lr(_short_run() / "logs", "LRnull_1216", seeds=SHORT_SEEDS)
        self.assertTrue(got["pass_"], got)


# ---------------------------------------------------------------------------
# v 凍結（S-vfreeze）と full-batch（S-fb）
# ---------------------------------------------------------------------------
class TrainingPathTests(unittest.TestCase):
    def test_v_is_exactly_constant_when_frozen(self):
        logdir = _short_run("Evf1_1216")
        got = E.s_vfreeze(logdir / "logs", "Evf1_1216",
                          ref_logdir=None, seeds=SHORT_SEEDS)
        self.assertTrue(got["pass_"], got)
        with np.load(logdir / "logs" / "Evf1_1216_seed0.npz",
                     allow_pickle=False) as z:
            v = z["layer1_v_unit"]
            self.assertEqual(float(np.ptp(v, axis=0).max()), 0.0)
            for i in range(1, v.shape[0]):
                self.assertEqual(v[i].tobytes(), v[0].tobytes())

    def test_the_frozen_arm_still_moves_its_other_weights(self):
        """v だけが止まっていること（全部凍っていたら空虚な PASS）。"""
        logdir = _short_run("Evf1_1216")
        with np.load(logdir / "logs" / "Evf1_1216_seed0.npz",
                     allow_pickle=False) as z:
            self.assertGreater(float(np.abs(z["layer1_zbar"][-1]
                                            - z["layer1_zbar"][0]).max()), 0.0)
            self.assertGreater(float(np.abs(z["layer1_w_norm"][-1]
                                            - z["layer1_w_norm"][0]).max()), 0.0)

    def test_the_freeze_branch_is_the_only_difference_from_the_host_loop(self):
        """`freeze_v` が偽なら写した loop は宿主の loop と bit 一致する。"""
        cfg = _cfg()
        cfg["common"]["seeds"] = SHORT_SEEDS
        out = Path(_SHARED["tmp"]) / "loopcmp"
        digests = []
        for fn in (HOST_GATE.train_arm_gate, E.train_arm_edge):
            st = HOST_DIAL.setup_arm_dial(cfg, E._arm(cfg, "Enull_1216"), "cpu")
            E._apply_hook(st, None, "Enull_1216")
            probes = list(range(0, 2001, 1000))
            rec = E.EdgeRecorder(probes, st)
            fn(st, rec, probes, 2000, out, [])
            digests.append(b"".join(
                t.detach().numpy().tobytes()
                for t in (st["net"].Ws[0], st["net"].bs[0], st["net"].v,
                          st["net"].c)))
        self.assertEqual(digests[0], digests[1])

    def test_s_fb_passes_and_records_the_float32_rounding(self):
        got = E.s_fb(cfg=_cfg())
        self.assertTrue(got["pass_"], json.dumps(got, ensure_ascii=False))
        for row in got["rows"]:
            for value in row["relerr"].values():
                self.assertLess(value, 1e-6)
            for value in row["relerr_f32"].values():
                self.assertLess(value, 1e-5)

    def test_the_batch_gradient_reduction_must_actually_be_a_mean(self):
        """`mean` を `sum` に変えた変異体は S-fb の float64 判定を必ず落とす。"""
        cfg = _cfg()
        c = copy.deepcopy(cfg)
        c["common"]["seeds"] = [0]
        st = E._setup_with_hook(c, E._arm(c, "FBLR_1216"), "cpu",
                                E._hook_of("FBLR_1216"))
        net, env, teacher = st["net"], st["env"], st["teacher"]
        env.step()
        X = full_support_ro(env)
        i64, p64, a64, yhat = E._forward_f64(st, X)
        d64 = yhat - teacher(X).double()
        good = E.grads_centered_elu_batch(net, i64, p64, a64, d64)
        d2 = 2.0 * d64
        bad_gc = d2.sum(dim=0)                      # ← 平均でなく総和にした変異体
        self.assertGreater(float((bad_gc - good[3]).abs().max()), 1e-6)

    def test_full_batch_keeps_the_flip_trajectory_of_the_online_arm(self):
        """`env.step()` を毎回呼ぶので flip の時刻と乱数消費がオンラインと一致する。"""
        cfg = _cfg()
        cfg["common"]["seeds"] = SHORT_SEEDS
        states = []
        for arm, hook in (("LRnull_1216", None),
                          ("FBLR_1216", E._hook_of("FBLR_1216"))):
            st = E._setup_with_hook(cfg, E._arm(cfg, arm), "cpu", hook)
            probes = [0, 1000]
            rec = E.EdgeRecorder(probes, st)
            fn = E._train_fn(st)
            fn(st, rec, probes, 1000, Path(_SHARED["tmp"]) / "fbcmp", [])
            states.append((st["env"].t, st["env"].flip_state.clone(),
                           st["env"].gen.get_state().clone()))
        self.assertEqual(states[0][0], states[1][0])
        self.assertTrue(torch.equal(states[0][1], states[1][1]))
        self.assertTrue(torch.equal(states[0][2], states[1][2]))

    def test_full_batch_actually_changes_the_learning_dynamics(self):
        """同じ入力列でも 32 パターン厳密勾配なので重みは online と一致しない。"""
        cfg = _cfg()
        cfg["common"]["seeds"] = SHORT_SEEDS
        w = []
        for arm, hook in (("LRnull_1216", None),
                          ("FBLR_1216", E._hook_of("FBLR_1216"))):
            st = E._setup_with_hook(cfg, E._arm(cfg, arm), "cpu", hook)
            probes = [0, 1000]
            rec = E.EdgeRecorder(probes, st)
            E._train_fn(st)(st, rec, probes, 1000,
                            Path(_SHARED["tmp"]) / "fbcmp2", [])
            w.append(st["net"].Ws[0].clone())
        self.assertFalse(torch.equal(w[0], w[1]))


# ---------------------------------------------------------------------------
# S-null / S-mirror（列別パリティ）
# ---------------------------------------------------------------------------
class ParityTests(unittest.TestCase):
    REF = Path(ROOT) / "results" / "p3_extend_0902" / "logs"

    def test_s_null_matches_the_committed_reference_on_a_short_run(self):
        if not self.REF.exists():
            self.skipTest("committed reference logs are not present")
        got = E.s_null("LRnull_1216", self.REF, _short_run())
        rows = [r for r in got["rows"] if r["status"] == "OK"]
        self.assertEqual(len(rows), len(SHORT_SEEDS))
        for row in rows:
            self.assertEqual(row["n_bad"], 0, row)
            self.assertGreater(row["n_compared"], 40)

    def test_s_null_is_not_vacuous(self):
        """1 列だけ壊した npz は S-null で必ず落ちる。"""
        if not self.REF.exists():
            self.skipTest("committed reference logs are not present")
        mut = Path(_SHARED["tmp"]) / "snull_mutant" / "logs"
        mut.mkdir(parents=True, exist_ok=True)
        src = _short_run() / "logs" / "LRnull_1216_seed0.npz"
        with np.load(src, allow_pickle=True) as z:
            payload = {k: z[k] for k in z.files}
        payload["layer1_zbar"] = payload["layer1_zbar"] + np.float32(1e-6)
        np.savez_compressed(mut / "LRnull_1216_seed0.npz", **payload)
        got = E.s_null("LRnull_1216", self.REF, mut.parent)
        self.assertFalse(got["pass_"])
        self.assertIn("layer1_zbar", got["rows"][0]["bad"])

    def test_s_null_e_maps_Enull_to_the_committed_E_reference(self):
        self.assertEqual(E.S_NULL_REF["Enull_1216"], "E_1216")
        self.assertEqual(E.S_NULL_REF["LRnull_1216"], "LR_1216")
        self.assertEqual(E.S_MIRROR_REF["FLn_1216"], "LR_1216")

    def test_mirror_column_helper_detects_sign_and_bytes(self):
        x = np.array([1.5, -2.25, 0.0, np.nan], dtype=np.float32)
        y = np.array([-1.5, 2.25, 0.0, np.nan], dtype=np.float32)
        self.assertTrue(E._mirror_column(x, y, flip=True)["pass_"])
        self.assertFalse(E._mirror_column(x, y, flip=False)["pass_"])
        self.assertTrue(E._mirror_column(x, x, flip=False)["pass_"])
        # NaN の位置が違えば落ちる
        z = y.copy()
        z[3] = 0.0
        self.assertFalse(E._mirror_column(x, z, flip=True)["pass_"])
        # 1 ulp の差でも落ちる（torch.equal 的な符号盲にはならない）
        w = y.copy()
        w[0] = np.nextafter(w[0], np.float32(0.0))
        self.assertFalse(E._mirror_column(x, w, flip=True)["pass_"])

    def test_s_mirror_on_synthetic_logs(self):
        """合成ログで PASS / FAIL の両方を出す（実走前に判定器を検算する）。"""
        base = Path(_SHARED["tmp"]) / "mirror"
        mine, ref = base / "logs", base / "ref"
        mine.mkdir(parents=True, exist_ok=True)
        ref.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(0)
        for seed in range(10):
            n, h = 5, 4
            cols = {k: rng.normal(size=(n, h)).astype(np.float32)
                    for k in E.MIRROR_SIGN_FLIP + E.MIRROR_INVARIANT}
            # 実走の p_hat は 32 パターン中の個数 k/32 なので、p' + p は
            # float32 で厳密に 1.0 になる（任意の float32 では一般に成り立たない）。
            p = (rng.integers(0, 33, size=(n, h)) / 32.0).astype(np.float32)
            a = dict(step=np.arange(n) * 1000)
            b = dict(step=np.arange(n) * 1000)
            for key, value in cols.items():
                b[key] = value
                a[key] = (0.0 - value) if key in E.MIRROR_SIGN_FLIP else value
            b["layer1_p_hat"] = p
            a["layer1_p_hat"] = (1.0 - p).astype(np.float32)
            np.savez_compressed(mine / f"FLn_1216_seed{seed}.npz", **a)
            np.savez_compressed(ref / f"LR_1216_seed{seed}.npz", **b)
        got = E.s_mirror("FLn_1216", ref, base)
        self.assertTrue(got["pass_"], json.dumps(got, ensure_ascii=False)[:800])
        # 変異体: 1 列だけ符号を戻すと落ちる
        with np.load(mine / "FLn_1216_seed3.npz", allow_pickle=False) as z:
            payload = {k: z[k] for k in z.files}
        payload["layer1_M"] = 0.0 - payload["layer1_M"]
        np.savez_compressed(mine / "FLn_1216_seed3.npz", **payload)
        self.assertFalse(E.s_mirror("FLn_1216", ref, base)["pass_"])

    def test_p_hat_sum_must_be_exactly_one(self):
        base = Path(_SHARED["tmp"]) / "mirror_p"
        mine, ref = base / "logs", base / "ref"
        mine.mkdir(parents=True, exist_ok=True)
        ref.mkdir(parents=True, exist_ok=True)
        for seed in range(10):
            n, h = 3, 2
            p = np.full((n, h), 0.25, dtype=np.float32)
            cols_b = {k: np.ones((n, h), dtype=np.float32)
                      for k in E.MIRROR_SIGN_FLIP + E.MIRROR_INVARIANT}
            cols_a = {k: ((0.0 - v) if k in E.MIRROR_SIGN_FLIP else v)
                      for k, v in cols_b.items()}
            # p_hat = k/32 は float32 で厳密に表せるので、登録された「p'+p == 1.0
            # の厳密等号」の感度は **1 ulp** で試す（1e-3 だと将来の許容誤差版でも
            # 落ちてしまい、厳密等号かどうかを試験できていなかった）。
            p_mine = np.float32(0.75)
            if seed == 0:
                p_mine = np.nextafter(p_mine, np.float32(0.0))
            np.savez_compressed(
                mine / f"FLn_1216_seed{seed}.npz", step=np.arange(n) * 1000,
                layer1_p_hat=np.full((n, h), p_mine, dtype=np.float32),
                **cols_a)
            np.savez_compressed(ref / f"LR_1216_seed{seed}.npz",
                                step=np.arange(n) * 1000, layer1_p_hat=p,
                                **cols_b)
        got = E.s_mirror("FLn_1216", ref, base)
        self.assertFalse(got["pass_"])
        self.assertGreater(got["rows"][0]["p_hat_sum_exceptions"], 0)
        self.assertEqual(got["rows"][1]["p_hat_sum_exceptions"], 0)


# ---------------------------------------------------------------------------
# 縮約ログ・投入計画
# ---------------------------------------------------------------------------
class OutputTests(unittest.TestCase):
    def test_tail_extract_keeps_the_records_the_judgments_read(self):
        logdir = _short_run() / "logs"
        out = Path(_SHARED["tmp"]) / "tail"
        got = E.tail_extract(logdir, out, arms=["LRnull_1216"])
        self.assertTrue(got["pass_"])
        with np.load(logdir / "LRnull_1216_seed0.npz", allow_pickle=True) as z:
            full = z["step"]
        with np.load(out / "LRnull_1216_seed0.npz", allow_pickle=True) as z:
            kept = z["step"]
            self.assertIn("lr_used", z.files)
            self.assertIn("layer1_w_free", z.files)
            self.assertEqual(z["layer1_w_free"].shape[0],
                             z["layer1_w_free_step"].shape[0])
        self.assertTrue(set(kept.tolist()) <= set(full.tolist()))
        for step in full:
            if step == 0 or step % E.PERIOD == 0:
                self.assertIn(int(step), kept.tolist())

    def test_tail_extract_selection_on_a_five_million_grid(self):
        step = np.arange(0, 5_000_001, 1000, dtype=np.int64)
        keep = E._tail_keep_steps(step, 5_000_000)
        kept = step[keep]
        self.assertIn(0, kept.tolist())
        for task in (301, 350, 351, 400, 451, 500, 100, 300):
            self.assertIn(task * E.PERIOD, kept.tolist())
        self.assertTrue(all(s in kept.tolist()
                            for s in range(4_800_000 + 1000, 5_000_001, 1000)))
        self.assertNotIn(1_001_000, kept.tolist())
        self.assertLess(len(kept), len(step) / 5)

    def test_launch_plan_lists_every_arm_and_a_parallelism(self):
        text = E.launch_plan(peak_gib=0.7)
        for arm in E.arm_order():
            self.assertIn(arm, text)
        self.assertIn("OMP_NUM_THREADS=1", text)
        self.assertIn("xargs -P", text)
        self.assertIn("--s-null LRnull_1216", text)
        self.assertIn("--s-mirror FLn_1216", text)

    def test_recommended_parallelism_formula(self):
        text = E.launch_plan(peak_gib=1.0)
        self.assertIn("min(20, floor(", text)
        self.assertIn("PAR=", text)


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# 批評ラウンド 1 で塞いだ穴（それぞれに「変異体が必ず落ちる」対を付ける）
# ---------------------------------------------------------------------------
def _mutant_of(fn, after_line: str, extra_line: str, name: str):
    """`fn` のソースに 1 行だけ足した関数を、``inspect.getsource`` が読める形で作る。

    `_copy_opcodes` は ``inspect.getsource`` を使うので、exec ではなく一時
    モジュールファイルに書いて import する。注釈は評価させない
    （``from __future__ import annotations``）。
    """
    import importlib.util
    import textwrap
    src = textwrap.dedent(inspect.getsource(fn))
    lines = src.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == after_line:
            indent = line[: len(line) - len(line.lstrip())]
            lines.insert(i + 1, indent + extra_line)
            break
    else:                                            # pragma: no cover - 保険
        raise AssertionError(f"anchor line not found: {after_line!r}")
    path = Path(_SHARED["tmp"]) / f"{name}.py"
    path.write_text("from __future__ import annotations\n" + "\n".join(lines)
                    + "\n", encoding="utf-8")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, fn.__name__)


class MinimalInsertCopyTests(unittest.TestCase):
    """MUT4: `inserts == want` の節そのものに陰性対照を付ける。

    既存の 2 つの変異体は宿主と形が違いすぎて difflib が `replace`/`delete` を
    出すため、別の節（`not bad`）で捕まっていた。ここでは**1 行だけ**足して
    `unregistered_opcodes == []`（＝`bad` の節は無傷）であることまで確かめる。
    """

    def test_one_extra_line_in_the_training_loop_is_caught_by_the_insert_clause(self):
        mutant = _mutant_of(E.train_arm_edge, "x = env.step()",
                            "_unregistered = 1", "mut_train_edge")
        got = E._copy_opcodes(HOST_GATE.train_arm_gate, mutant,
                              (E.TRAIN_INSERTS,))
        self.assertFalse(got["pass_"])
        self.assertEqual(got["unregistered_opcodes"], [],
                         "must be caught by `inserts == want`, not by `bad`")
        self.assertIn(["_unregistered = 1"], got["inserted"])

    def test_one_extra_line_in_the_runner_is_caught_by_the_insert_clause(self):
        mutant = _mutant_of(E._run_arm_edge,
                            "_apply_hook(st, _hook_of(arm), arm)",
                            "_unregistered = 1", "mut_run_edge")
        got = E._copy_opcodes(HOST_WEIRD._run_arm_weird, mutant, E.RUN_INSERTS)
        self.assertFalse(got["pass_"])
        self.assertEqual(got["unregistered_opcodes"], [])

    def test_the_unmodified_copy_still_passes_through_the_same_path(self):
        got = E.s_copy()
        self.assertTrue(got["pass_"], json.dumps(got, ensure_ascii=False)[:600])


class HookValueTests(unittest.TestCase):
    """MUT5: S-hook-inplace は「どれが動いたか」でなく「いくつになったか」を見る。"""

    def test_every_hook_matches_its_analytic_expectation(self):
        got = E.s_hook_inplace(_cfg())
        self.assertTrue(got["pass_"], json.dumps(got, ensure_ascii=False))
        for row in got["rows"]:
            self.assertTrue(row["matches_expected_values"], row)

    def _with_patched_hook(self, patch):
        real = E._apply_hook

        def fake(st, hook, arm=""):
            st = real(st, hook, arm)
            patch(st, hook)
            return st
        E._apply_hook = fake
        try:
            return E.s_hook_inplace(_cfg())
        finally:
            E._apply_hook = real

    def test_a_sign_flipped_b_offset_fails(self):
        def patch(st, hook):
            if hook is not None and hook["type"] == "b_offset":
                with torch.no_grad():
                    st["net"].bs[0].add_(-2.0 * float(hook["value"]))
        got = self._with_patched_hook(patch)
        self.assertFalse(got["pass_"])
        bad = [r for r in got["rows"] if r["hook"] == "b_offset"]
        self.assertTrue(bad and not any(r["matches_expected_values"] for r in bad))

    def test_an_effectively_zero_b_offset_fails(self):
        def patch(st, hook):
            if hook is not None and hook["type"] == "b_offset":
                with torch.no_grad():
                    st["net"].bs[0].add_(-float(hook["value"])
                                         * (1.0 - 1e-30))
        got = self._with_patched_hook(patch)
        self.assertFalse(got["pass_"])

    def test_a_scale_that_forgets_to_divide_v_fails(self):
        def patch(st, hook):
            if hook is not None and hook["type"] == "scale":
                with torch.no_grad():
                    st["net"].v.mul_(float(hook["value"]))     # div_ を打ち消す
        got = self._with_patched_hook(patch)
        self.assertFalse(got["pass_"])


class HookNoopLivenessTests(unittest.TestCase):
    """MUT2: noop 側が本当に `_apply_hook` を通ったことを要求する。"""

    def test_the_noop_state_carries_the_payload_keys_only_apply_hook_writes(self):
        got = E.s_hook_noop(cfg=_cfg())
        self.assertTrue(got["pass_"], got)
        self.assertTrue(got["went_through_apply_hook"])
        self.assertEqual(got["host_has_hook_payload"], [])

    def test_comparing_the_host_state_with_itself_would_fail(self):
        """`_setup_with_hook(..., None)` を `setup_arm_dial(...)` に退化させたら落ちる。"""
        cfg = _cfg()
        blk = E._arm(cfg, "LRnull_1216")
        st_host = HOST_DIAL.setup_arm_dial(cfg, blk, "cpu")
        hook_keys = ("init_hook", "init_hook_arg", "lr_used", "freeze_v",
                     "batch_mode")
        for key in hook_keys:
            self.assertNotIn(key, st_host)
        st_hooked = E._setup_with_hook(cfg, blk, "cpu", None)
        for key in hook_keys:
            self.assertIn(key, st_hooked)


class VFreezeReferenceTests(unittest.TestCase):
    """MUT3: 「Evf1 の初期 v が Enull と bit 一致」の節に試験を付ける。"""

    def _both_logs(self) -> Path:
        key = "vfreeze_pair"
        if key not in _SHARED:
            out = Path(_SHARED["tmp"]) / "vfreeze_pair"
            for arm in ("Evf1_1216", "Enull_1216"):
                E.run_single_arm(arm, steps=SHORT_STEPS, outdir=out,
                                 seeds=SHORT_SEEDS, cfg=_cfg())
            _SHARED[key] = out
        return _SHARED[key] / "logs"

    def test_the_frozen_v_equals_the_untouched_initial_v(self):
        logs = self._both_logs()
        got = E.s_vfreeze(logs, "Evf1_1216", ref_logdir=logs,
                          ref_arm="Enull_1216", seeds=SHORT_SEEDS)
        self.assertTrue(got["pass_"], json.dumps(got, ensure_ascii=False))
        for row in got["rows"]:
            self.assertIs(row["initial_v_equals_reference"], True, row)

    def test_a_one_ulp_change_in_the_initial_v_makes_it_fail(self):
        logs = self._both_logs()
        mut = Path(_SHARED["tmp"]) / "vfreeze_mut"
        mut.mkdir(parents=True, exist_ok=True)
        with np.load(logs / "Evf1_1216_seed0.npz", allow_pickle=True) as z:
            payload = {k: z[k] for k in z.files}
        v = payload["layer1_v_unit"].copy()
        v[:, 0] = np.nextafter(v[0, 0], np.float32(np.inf))
        payload["layer1_v_unit"] = v
        np.savez_compressed(mut / "Evf1_1216_seed0.npz", **payload)
        shutil.copy(logs / "Enull_1216_seed0.npz",
                    mut / "Enull_1216_seed0.npz")
        got = E.s_vfreeze(mut, "Evf1_1216", ref_logdir=mut,
                          ref_arm="Enull_1216", seeds=(0,))
        self.assertFalse(got["pass_"])
        self.assertIs(got["rows"][0]["initial_v_equals_reference"], False)

    def test_a_missing_reference_log_fails_instead_of_passing_silently(self):
        logs = self._both_logs()
        empty = Path(_SHARED["tmp"]) / "vfreeze_no_ref"
        empty.mkdir(parents=True, exist_ok=True)
        got = E.s_vfreeze(logs, "Evf1_1216", ref_logdir=empty,
                          ref_arm="Enull_1216", seeds=(0,))
        self.assertFalse(got["pass_"])
        self.assertIs(got["rows"][0]["initial_v_equals_reference"], False)


class MirrorZeroSignTests(unittest.TestCase):
    """SYM1/SYM2: 期待値は `np.negative`・比較はバイト・両腕 0.0 は登録した例外。"""

    def test_the_runner_and_the_analysis_share_one_rule(self):
        from src import edge_law_analyze_0905 as A
        self.assertIs(E._mirror_rule, A.mirror_parity)

    def test_a_negative_zero_that_should_not_be_there_is_caught(self):
        """`0.0 - x` 版が見逃していた変異（`+0.0` を `-0.0` に変える）。"""
        x = np.array([0.0, 1.5], dtype=np.float32)     # 参照側
        good = np.array([0.0, -1.5], dtype=np.float32)
        self.assertTrue(E._mirror_column(x, good, flip=True)["pass_"])
        # 参照が -0.0 のとき鏡像は +0.0 でなければならない（np.negative の要請）
        xn = np.array([-0.0, 1.5], dtype=np.float32)
        bad = np.array([-0.0, -1.5], dtype=np.float32)
        got = E._mirror_column(xn, bad, flip=True)
        self.assertTrue(got["pass_"])                  # 両腕 0.0 → 登録した例外
        self.assertEqual(got["n_zero_sign_exceptions"], 1)
        # 例外は「両腕とも 0.0」に限る: 片方が 0 でないなら通してはいけない
        self.assertFalse(E._mirror_column(
            np.array([0.0], dtype=np.float32),
            np.array([1e-45], dtype=np.float32), flip=True)["pass_"])

    def test_pass_and_n_mismatch_can_never_disagree(self):
        """`!=` で位置を拾っていたので `pass_=False, n_mismatch=0` が出せた。"""
        x = np.array([0.0, 1.5, -2.0], dtype=np.float32)
        y = np.array([0.5, -1.5, 2.0], dtype=np.float32)
        got = E._mirror_column(x, y, flip=True)
        self.assertFalse(got["pass_"])
        self.assertGreater(got["n_mismatch"], 0)

    def test_a_moved_nan_with_the_same_count_fails(self):
        """MUT7: NaN を落とす経路が黙って記録を飛ばさないこと。"""
        x = np.array([np.nan, 1.5, 2.0], dtype=np.float32)
        y = np.array([-1.5, np.nan, -2.0], dtype=np.float32)
        got = E._mirror_column(x, y, flip=True)
        self.assertFalse(got["pass_"])
        self.assertFalse(got["nan_pattern_equal"])     # この節だけが捕まえる
        self.assertEqual(got["n_mismatch"], 0)

    def test_n_records_compared_is_n_minus_the_shared_nan_count(self):
        x = np.array([np.nan, 1.5, 2.0, 3.0], dtype=np.float32)
        y = np.array([np.nan, -1.5, -2.0, -3.0], dtype=np.float32)
        got = E._mirror_column(x, y, flip=True)
        self.assertTrue(got["pass_"])
        self.assertEqual(got["n_records_compared"], 3)

    def test_s_mirror_reports_the_registered_zero_sign_exception_count(self):
        base = Path(_SHARED["tmp"]) / "mirror_zero"
        mine, ref = base / "logs", base / "ref"
        mine.mkdir(parents=True, exist_ok=True)
        ref.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(11)
        for seed in range(10):
            n, h = 6, 4
            cols_b, cols_a = {}, {}
            for key in E.MIRROR_SIGN_FLIP + E.MIRROR_INVARIANT:
                raw = rng.normal(size=(n + 1, h)).astype(np.float32)
                raw[2] = raw[1]                        # 差が +0.0 になる場所
                if key == "layer1_dzbar":              # 実際に差で作る列
                    cols_b[key] = np.diff(raw, axis=0)
                    cols_a[key] = np.diff(-raw, axis=0)
                else:
                    cols_b[key] = raw[1:]
                    cols_a[key] = (np.negative(raw[1:])
                                   if key in E.MIRROR_SIGN_FLIP else raw[1:])
            p = (rng.integers(0, 33, size=(n, h)) / 32.0).astype(np.float32)
            np.savez_compressed(mine / f"FLn_1216_seed{seed}.npz",
                                step=np.arange(n) * 1000,
                                layer1_p_hat=(1.0 - p).astype(np.float32),
                                **cols_a)
            np.savez_compressed(ref / f"LR_1216_seed{seed}.npz",
                                step=np.arange(n) * 1000, layer1_p_hat=p,
                                **cols_b)
        got = E.s_mirror("FLn_1216", ref, base)
        self.assertTrue(got["pass_"], json.dumps(got, ensure_ascii=False)[:900])
        self.assertGreater(sum(r["zero_sign_exceptions"] for r in got["rows"]), 0)
        self.assertGreater(got["rows"][0]["elements_compared"], 0)


class MirrorZmaxSubstituteTests(unittest.TestCase):
    """SYM3: `layer1_zmax` の代替検査に入口が有ること（配線されていること）。"""

    def _pair(self) -> Path:
        key = "zmax_pair"
        if key not in _SHARED:
            out = Path(_SHARED["tmp"]) / "zmax_pair"
            for arm in ("FLn_1216", "LRnull_1216"):
                E.run_single_arm(arm, steps=SHORT_STEPS, outdir=out,
                                 seeds=SHORT_SEEDS, cfg=_cfg())
            _SHARED[key] = out
        return _SHARED[key]

    def test_zmax_of_the_mirror_is_minus_zmin_of_the_reference(self):
        out = self._pair()
        got = E.s_mirror_zmax("FLn_1216", "LRnull_1216", out)
        self.assertTrue(got["pass_"], json.dumps(got, ensure_ascii=False)[:800])
        self.assertEqual(len(got["rows"]), len(SHORT_SEEDS))

    def test_a_perturbed_zmax_makes_it_fail(self):
        out = self._pair()
        mut = Path(_SHARED["tmp"]) / "zmax_mut"
        (mut / "logs").mkdir(parents=True, exist_ok=True)
        for arm in ("FLn_1216", "LRnull_1216"):
            shutil.copy(out / "logs" / f"{arm}_seed0.npz",
                        mut / "logs" / f"{arm}_seed0.npz")
        path = mut / "logs" / "FLn_1216_seed0.npz"
        with np.load(path, allow_pickle=True) as z:
            payload = {k: z[k] for k in z.files}
        zmax = payload["layer1_zmax"].copy()
        zmax[3, 2] = np.nextafter(zmax[3, 2], np.float32(np.inf))
        payload["layer1_zmax"] = zmax
        np.savez_compressed(path, **payload)
        got = E.s_mirror_zmax("FLn_1216", "LRnull_1216", mut)
        self.assertFalse(got["pass_"])

    def test_the_cli_and_the_launch_plan_expose_it(self):
        plan = E.launch_plan(parallel=4)
        self.assertIn("--s-mirror-zmax FLn_1216 LRnull_1216", plan)
        import argparse
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), self.assertRaises(SystemExit):
            import sys
            argv = sys.argv
            sys.argv = ["edge_law_0905", "--help"]
            try:
                E.main()
            finally:
                sys.argv = argv
        self.assertIn("--s-mirror-zmax", buf.getvalue())
        del argparse


class SNullReferenceTests(unittest.TestCase):
    """SYM4: 参照が自分自身に落ちる経路を塞ぐ。"""

    def test_an_unregistered_arm_without_an_explicit_reference_raises(self):
        with self.assertRaises(HOST_DIAL.SanityError):
            E.s_null("Evf1_1216", _short_run() / "logs", _short_run())

    def test_comparing_a_directory_with_itself_raises(self):
        with self.assertRaises(HOST_DIAL.SanityError):
            E.s_null("LRnull_1216", _short_run() / "logs", _short_run(),
                     ref_arm="LRnull_1216")

    def test_an_explicit_distinct_reference_still_works(self):
        src = _short_run() / "logs"
        copy_dir = Path(_SHARED["tmp"]) / "snull_copy" / "logs"
        copy_dir.mkdir(parents=True, exist_ok=True)
        for seed in SHORT_SEEDS:
            shutil.copy(src / f"LRnull_1216_seed{seed}.npz",
                        copy_dir / f"LRnull_1216_seed{seed}.npz")
        got = E.s_null("LRnull_1216", copy_dir, _short_run(),
                       ref_arm="LRnull_1216")
        self.assertTrue(got["pass_"])


class StateHash1mTests(unittest.TestCase):
    """MUT8: `state_hash_1m` の文字列一致の節に陰性対照を付ける。"""

    def _pair(self, mutate_hash: bool):
        """1M step に届いたことにした npz を 2 組作る（step 列を引き伸ばす）。"""
        tag = "hashmut" if mutate_hash else "hashok"
        base = Path(_SHARED["tmp"]) / tag
        mine, ref = base / "logs", base / "ref"
        mine.mkdir(parents=True, exist_ok=True)
        ref.mkdir(parents=True, exist_ok=True)
        src = _short_run() / "logs"
        for seed in SHORT_SEEDS:
            with np.load(src / f"LRnull_1216_seed{seed}.npz",
                         allow_pickle=True) as z:
                payload = {k: z[k] for k in z.files}
            payload["step"] = payload["step"].astype(np.int64) * 100
            payload["state_hash_1m"] = np.array('{"W": "deadbeef"}')
            np.savez_compressed(ref / f"LRnull_1216_seed{seed}.npz", **payload)
            if mutate_hash:
                payload = dict(payload)
                payload["state_hash_1m"] = np.array('{"W": "0000feed"}')
            np.savez_compressed(mine / f"LRnull_1216_seed{seed}.npz", **payload)
        return base, ref

    def test_the_hash_is_actually_compared_once_both_runs_pass_1M(self):
        base, ref = self._pair(mutate_hash=False)
        got = E.s_null("LRnull_1216", ref, base, ref_arm="LRnull_1216")
        self.assertTrue(got["pass_"], json.dumps(got, ensure_ascii=False)[:600])
        for row in got["rows"]:
            self.assertIs(row["state_hash_1m_equal"], True, row)

    def test_a_different_hash_alone_makes_S_null_fail(self):
        base, ref = self._pair(mutate_hash=True)
        got = E.s_null("LRnull_1216", ref, base, ref_arm="LRnull_1216")
        self.assertFalse(got["pass_"])
        self.assertEqual(got["rows"][0]["n_bad"], 0)      # 配列は無傷
        self.assertIs(got["rows"][0]["state_hash_1m_equal"], False)

    def test_a_short_run_leaves_the_hash_uncompared(self):
        got = E.s_null("LRnull_1216",
                       Path(ROOT) / "results" / "p3_extend_0902" / "logs",
                       _short_run())
        self.assertTrue(all(r["state_hash_1m_equal"] is None
                            for r in got["rows"] if r["status"] == "OK"))


class ColumnAgreementTests(unittest.TestCase):
    """MUT1: `columns` 項目は実際に書かれた npz を正本にする。"""

    def test_it_reads_the_real_logs(self):
        got = E.expected_column_agreement(_short_run() / "logs")
        self.assertTrue(got["pass_"], got)
        self.assertTrue(got["from_real_logs"])
        self.assertEqual(got["missing"], [])
        self.assertEqual(got["literal_not_written"], [])

    def test_a_renamed_output_column_is_caught(self):
        mut = Path(_SHARED["tmp"]) / "colmut" / "logs"
        mut.mkdir(parents=True, exist_ok=True)
        with np.load(_short_run() / "logs" / "LRnull_1216_seed0.npz",
                     allow_pickle=True) as z:
            payload = {k: z[k] for k in z.files}
        payload["layer1_zminimum"] = payload.pop("layer1_zmin")
        np.savez_compressed(mut / "LRnull_1216_seed0.npz", **payload)
        got = E.expected_column_agreement(mut)
        self.assertFalse(got["pass_"])
        self.assertIn("layer1_zmin", got["missing"])

    def test_a_broken_analysis_import_is_not_a_silent_pass(self):
        real = E.expected_column_agreement
        self.assertTrue(callable(real))
        src = inspect.getsource(E.expected_column_agreement)
        self.assertNotIn("skipped", src)
        self.assertNotIn("except ImportError", src)


class SeedSubsetGuardTests(unittest.TestCase):
    """MUT9: 登録の地平線で seed 部分集合を回そうとしたら止める。"""

    def test_a_subset_at_the_registered_horizon_raises(self):
        with self.assertRaises(HOST_DIAL.SanityError):
            E.run_single_arm("LRnull_1216", steps=None,
                             outdir=Path(_SHARED["tmp"]) / "subset",
                             seeds=[0, 1], cfg=_cfg())

    def test_a_shortened_check_run_with_a_subset_is_still_allowed(self):
        out = Path(_SHARED["tmp"]) / "subset_ok"
        E.run_single_arm("LRnull_1216", steps=2000, outdir=out, seeds=[0, 1],
                         cfg=_cfg())
        self.assertTrue((out / "logs" / "LRnull_1216_seed0.npz").exists())
