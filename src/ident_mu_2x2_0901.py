"""ident_mu_2x2_0901: µ ダイヤル × タスク可識別性ダイヤルの 2×2（純化版・実装）。

起草 spec は Obsidian `可塑性喪失/spec/可識別性2x2_spec_0901.md` **v2**（**未凍結**）。
`configs/ident_mu_2x2_0901.yaml` の `preregistration.frozen` が false のあいだ、
本モジュールは自己検査（`--selftest` / `--smoke`）しか通さず、ゲート・本走・解析は
`validate_config` が段階ごとに拒否する。

v2（vault `ee831d6`・D8）で入った変更。

* **要因 4 セル全部に b 限定 WD λ=1e-3 を敷く**（純化）。`Aexact` の死が全数 b の
  自走だったため、M ダイヤルが現象1 の壁と現象2 を同時に切り替えていた。b を縛った
  地面の上で I × M を測る。更新式は `nets.VecMLPL.sgd_step_layers` の
  `b <- b - eta*(gb + wd_b*b)` をそのまま使い、$W$・$v$・$c$・$\\mathbf u$ には
  掛けない（S1/S2 が検査する）。
* アンカーは 2 本。`im_nowd`（λ=0 の `Aexact` 双子・登録副判定 R-ext を担う）と
  `std_anchor`（引き直しを持つ唯一の腕）。
* 主 endpoint は **E-drift（B10−B02）＋ E-level（B10）の共主**に戻り、E-onset は
  REPORT_ONLY に降格した。

設計の要点（spec §2 Option B / §3.2）。

* **M ダイヤル**（µ）は「第1層入力から構成された定数 offset を引く」ことで置く。
  offset は task 境界でのみ引き直し、追随（EMA）を一切持たない。
      M+ : offset = [flip_t - flip_0, 0_5]      -> 可視入力 [flip_0,  rnd]
      M- : offset = [flip_t,          0.5_5]    -> 可視入力 [0_15,    rnd-0.5]
      std: offset = 0                           -> 可視入力 [flip_t,  rnd]
  これは凍結済み `mlp2_phase1.forward_centered` の「層入力から `layer_means` を引く」
  経路そのものなので、`center_alpha=0`（EMA 更新が恒等）にすれば厳密レコーダ
  (`exact_wall_record` / `exact_layer_record_p1`) も無改造で可視入力を見る。
* **I ダイヤル**（可識別性）は出力側バイパス `u^T code` の on/off。`code` が
  恒等 0 の腕では `gu = 0` なので u は厳密に 0 のまま動かず、乱数も消費しない。
* 2 因子は実装レベルで直交する: 腕どうしの差は offset テンソルの中身と code だけ。

9/1 の改定（vault `3142500`）で spec 側が実装へ寄せた 4 点。もう乖離ではない。

1. **`unfit` は DC 盲**（`residual.var()/y.var()`）。task 内で定数のバイパスは
   `unfit` を 1 ビットも動かさないので、I ダイヤルは主 endpoint を表現経由では
   動かせず、力学経由（境界での DC 再フィットの churn）でしか動かせない。
   spec §6.1 に明記された（解釈はむしろ鋭くなる）。DC を見る量は
   `eval_loss_exact` の方で、こちらはバイパス込みで記録する。
2. `bypass_share` は**出力パワーの分け前**（分散比だとタスク内で恒等 0）。
3. **S-op は切替プローブへ改定**。旧定義（定常単一タスク・`unfit` 比較）は
   バイパスが $c$ と冗長かつ `unfit` が DC 盲で二重に成立不能だった。
4. R-ext の全滅は `strict_dead_frac == 1.0` で判定する（`alive == 0` は σ 退化の
   飽和常時発火を絶滅と誤分類しうるので判定に使わず、食い違い件数のみ併記）。

**残っている乖離は 1 点だけ。** E-onset（REPORT_ONLY）を「一度閾値を下回った
あとの最初の上抜け」と定義している（spec §6.1 の字義は「初超え」だが、素直に
読むと初期過渡で全 seed が tau=1 になる）。
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from .bias_wd_0901 import paired_ci, s1_s2_algebra
from .dose_const_5m import clopper_pearson
from .bias_wd_common import (
    TaskEndRecorder,
    exact_wall_record,
    markdown_table,
    provenance,
    require_omp,
    wall_closed_form_kappa,
    write_arm_npz,
)
from .bias_wd_std_seediso_0901 import (
    ARM_INVALID_EXCLUSION_LIMIT,
    COMPLETE,
    COMPLETE_WITH_EXCLUSIONS,
    ExclusionLimitExceeded,
    SeedIsolationRecorder,
    _quarantine_seed,
)
from .common import ROOT, load_config
from .mlp2_phase0 import identity_sanity_pass
from .mlp2_phase1 import (
    NUMERIC_DIVERGENCE,
    _base_cfg,
    _numeric_divergence_event,
    exact_layer_record_p1,
    forward_centered,
    grads_centered,
    setup_arm_p1,
    train_arm_p1,
)
from .ratchet_log import full_support_ro, teacher_f64


CONFIG = Path(ROOT) / "configs" / "ident_mu_2x2_0901.yaml"

CELL_ORDER = ("IM", "iM", "Im", "im")
NOWD_ARM = "im_nowd"
MPLUS_NOWD_ARM = "iM_nowd"      # 追補2（2026-09-02）
ANCHOR_ARM = "std_anchor"
ANCHOR_ARMS = (NOWD_ARM, MPLUS_NOWD_ARM, ANCHOR_ARM)
ARM_ORDER = CELL_ORDER + ANCHOR_ARMS
VALID_ARM_STATUSES = {COMPLETE, COMPLETE_WITH_EXCLUSIONS}

VISIBLE_MODES = ("flip0", "zero_centered", "raw")
CODE_MODES = ("flip_t", "zero")

# 主判定（spec §6.3）
MU_DOMINANT = "MU_DOMINANT"
IDENT_DOMINANT = "IDENT_DOMINANT"
BOTH_MATTER = "BOTH_MATTER"
NEITHER_MATTERS = "NEITHER_MATTERS"
INTERACTION_DOMINATES = "INTERACTION_DOMINATES"
EFFECT_LEVEL_DEPENDENT = "EFFECT_LEVEL_DEPENDENT"
INCONCLUSIVE_WIDE = "INCONCLUSIVE_WIDE"
# 登録副判定 R-ext（spec §6.4）
BWD_PREVENTS_EXTINCTION = "BWD_PREVENTS_EXTINCTION"
EXTINCTION_PERSISTS = "EXTINCTION_PERSISTS"
PARTIAL_RESCUE = "PARTIAL_RESCUE"
# 登録副判定 R-ext-M+（追補2 §3）。符号ベースで、タイトな null は主張しない。
BWD_DELAYS_EXTINCTION_UNDER_MU = "BWD_DELAYS_EXTINCTION_UNDER_MU"
BWD_ACCELERATES_EXTINCTION_UNDER_MU = "BWD_ACCELERATES_EXTINCTION_UNDER_MU"
BWD_EFFECT_NOT_DISTINGUISHED_UNDER_MU = "BWD_EFFECT_NOT_DISTINGUISHED_UNDER_MU"
R_EXT_INVALID_TOO_FEW_PAIRED = "R_EXT_INVALID_TOO_FEW_PAIRED"
# フラグ
I_CELLS_INVALID_S_OP = "I_CELLS_INVALID_S_OP"
ONSET_NEVER_BELOW = "ONSET_NEVER_BELOW"
LADDER_INVERTS = "LADDER_INVERTS"
CEILING_CONTAMINATED = "CEILING_CONTAMINATED"
E_DRIFT_INVALID_FLOOR = "E_DRIFT_INVALID_FLOOR"
CONTRAST_INVALID_TOO_FEW_PAIRED = "CONTRAST_INVALID_TOO_FEW_PAIRED"

# 2 因子の対比（spec §6.3）。符号は「左 - 右」。
FACTOR_CONTRASTS = {
    "M_i": ("iM", "im"),
    "M_ii": ("IM", "Im"),
    "I_i": ("IM", "iM"),
    "I_ii": ("Im", "im"),
}
STAGES = ("implementation", "selftest", "smoke", "gates", "full", "analyze")


# --------------------------------------------------------------- config

def _P(cfg: dict) -> dict:
    return cfg["ident_mu_2x2"]


def _compat_cfg(cfg: dict) -> dict:
    """凍結済みの CI helper に section 名だけ適合させる。"""
    out = dict(cfg)
    out["bias_wd"] = _P(cfg)
    return out


def _arm_cfg(cfg: dict, name: str) -> dict:
    return next(arm for arm in cfg["arms"] if arm["name"] == name)


def outdir_of(cfg: dict) -> Path:
    return Path(ROOT) / _P(cfg)["output_dir"]


def gatedir_of(cfg: dict) -> Path:
    return Path(ROOT) / _P(cfg)["gate_dir"]


def preregistration_missing(cfg: dict) -> list[str]:
    """事前登録の未成立点。空リストになって初めて本走が許される。"""
    pre = cfg["preregistration"]
    return [f"preregistration.{key}" for key in
            ("decisions_complete", "predictions_confirmed", "frozen",
             "repo_spec_committed", "execution_authorized")
            if pre.get(key) is not True]


def _validate_offset(cfg: dict) -> None:
    """乱数系列の実体 (SEED_BASE + width + offset) の衝突を潰す。

    `--seeds` を変えるだけでは同一系列になる（T0b の事故）ので新 seed 群は
    generator_offset で作る。幅違いの走があるため、offset だけでなく
    **offset + width** で衝突を見る。
    """
    offset = int(cfg["common"]["generator_offset"])
    widths = {int(v) for v in cfg["sanity"]["used_widths"]}
    used = {int(v) for v in cfg["sanity"]["used_generator_offsets"]}
    if offset in used:
        raise ValueError("generator_offset collides with an existing use")
    ours = {offset + int(arm["hidden"][0]) for arm in cfg["arms"]}
    theirs = {value + width for value in used for width in widths}
    collisions = sorted(ours & theirs)
    if collisions:
        raise ValueError(f"generator base collision at {collisions}")
    if int(_P(cfg)["bootstrap_seed"]) in {
            int(v) for v in cfg["sanity"]["used_bootstrap_seeds"]}:
        raise ValueError("bootstrap_seed collides with an existing use")


REGISTERED_PREDICTIONS = {
    "main_verdict": MU_DOMINANT,
    "e_drift_rank": "mu_strong_type_M_plus_cells_degrade_more_than_M_minus_cells",
    "r_ext": BWD_PREVENTS_EXTINCTION,
    "s_op_passes": True,
    "rewrite_target_if_wrong": "unknown",
}


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_preregistration_record(cfg: dict) -> None:
    """事前登録の**記録**が spec と食い違っていないこと。

    フラグを立てるだけで実体が伴わない状態を作れないようにする。
    `predictions_confirmed` は §8 の 5 項が字句で入っていることを、
    `repo_spec_committed` は repo 側 spec が実在し sha256 が一致することを要求する。
    """
    pre = cfg["preregistration"]
    if pre.get("predictions_confirmed") is True:
        got = {key: pre.get("predictions", {}).get(key)
               for key in REGISTERED_PREDICTIONS}
        if got != REGISTERED_PREDICTIONS:
            raise ValueError(f"recorded predictions differ from spec §8: {got}")
    if pre.get("repo_spec_committed") is True:
        spec_path = Path(ROOT) / str(cfg["spec"])
        if not spec_path.is_file():
            raise ValueError(f"repo spec is missing: {spec_path}")
        if pre.get("repo_spec_sha256") != _sha_file(spec_path):
            raise ValueError("repo spec sha256 does not match the committed file")


def validate_config(cfg: dict, *, stage: str) -> None:
    """設計の検査。`stage` が進むほど事前登録の成立を強く要求する。"""
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}")
    C, A, I, P, S = (cfg["common"], cfg["condA"], cfg["intervention"],
                     _P(cfg), cfg["sanity"])
    if cfg.get("spec") != "specs/spec_ident_mu_2x2_0901.md":
        raise ValueError("registered spec path differs")
    if (str(C["device"]), float(C["lr_main"]), int(C["generator_offset"])) != (
            "cpu", 0.01, 20_260_910):
        raise ValueError("registered device/lr/generator_offset differs")
    if (int(A["m"]), int(A["f"]), int(A["target_hidden"]), float(A["beta"])) != (
            20, 15, 100, 0.7):
        raise ValueError("registered condA differs")
    if list(A["T_values"]) != [10_000] or list(A["encodings"]) != ["std"]:
        raise ValueError("registered T/encoding differs")
    if (str(I["name"]) != "B_visible_input_construction"
            or str(I["tracker"]) != "none"
            or float(I["center_alpha"]) != 0.0
            or I["consumes_rng"] is not False
            or str(I["offset_refresh"]) != "task_boundary_only"):
        raise ValueError("Option B intervention differs from the design")
    lam = float(cfg["bias_weight_decay"]["lam"])
    expected = [
        ("IM", "flip0", "flip_t", lam), ("iM", "flip0", "zero", lam),
        ("Im", "zero_centered", "flip_t", lam), ("im", "zero_centered", "zero", lam),
        (NOWD_ARM, "zero_centered", "zero", 0.0),
        (MPLUS_NOWD_ARM, "flip0", "zero", 0.0),
        (ANCHOR_ARM, "raw", "zero", 0.0),
    ]
    got = [(a["name"], str(a["visible"]), str(a["code"]), float(a["wd_b"]))
           for a in cfg["arms"]]
    if got != expected:
        raise ValueError(f"registered arms differ: {got}")
    for arm in cfg["arms"]:
        if ([int(v) for v in arm["hidden"]] != [100]
                or list(arm.get("centered_layers") or []) != [1]):
            raise ValueError(f"{arm['name']}: depth/offset differ")
    wd = cfg["bias_weight_decay"]
    if (lam != 1e-3 or wd["decoupled"] is not False
            or str(wd["targets"]) != "hidden_bias_only"
            or list(wd["excluded"]) != ["W", "v", "c", "u"]
            or wd["branchless"] is not True):
        raise ValueError("registered b-weight-decay differs")
    if cfg["pairing"]["paired_groups"] != [list(CELL_ORDER)]:
        raise ValueError("registered pairing differs")
    if (list(cfg["pairing"]["anchor_arms"]) != [NOWD_ARM, ANCHOR_ARM]
            or list(cfg["pairing"]["r_ext_group"]) != ["im", NOWD_ARM]
            or list(cfg["pairing"]["r_ext_mplus_group"]) != ["iM", MPLUS_NOWD_ARM]):
        raise ValueError("registered anchors / R-ext groups differ")
    mplus = P["r_ext_mplus"]
    if (list(mplus["arms"]) != ["iM", MPLUS_NOWD_ARM]
            or str(mplus["primary"]) != "extinction_task"
            or str(mplus["rule"]) != "paired_ci_sign_only"
            or mplus["equivalence_margin"] is not None
            or mplus["null_is_not_tight"] is not True
            or int(mplus["bootstrap_seed"]) != 20_260_913):
        raise ValueError("registered R-ext-M+ differs")
    if int(cfg["phase1"]["task_period"]) != 10_000:
        raise ValueError("registered task period differs")
    if (list(P["early_block_tasks"]), list(P["late_block_tasks"]),
            int(P["block_tasks"]), int(P["n_blocks"])) != (
            [51, 100], [451, 500], 50, 10):
        raise ValueError("registered blocks differ")
    onset = P["onset"]
    if (str(P["primary_endpoint"]) != "drift"
            or list(P["drift"]["windows"]) != ["B02", "B10"]
            or str(P["drift"]["statistic"]) != "mean_log10_unfit"
            or str(P["level"]["block"]) != "B10"):
        raise ValueError("registered endpoints differ (v2 is E-drift primary)")
    if (str(onset["role"]) != "report_only"
            or int(onset["window_tasks"]) != 10
            or str(onset["smoother"]) != "moving_median"
            or float(onset["threshold"]) != -1.0
            or [float(v) for v in onset["sensitivity_thresholds"]] != [-0.5, -1.5]
            or int(onset["censor_task"]) != 501
            or onset["require_prior_below"] is not True):
        raise ValueError("registered onset definition differs")
    if (float(P["equivalence_margin"]), float(P["interaction_margin"]),
            float(P["ceiling_flag_dex"]),
            float(P["unfit_floor"])) != (0.15, 0.50, 3.0, 1e-16):
        raise ValueError("registered margins/floor differ")
    r_ext = P["r_ext"]
    if (list(r_ext["arms"]) != ["im", NOWD_ARM]
            or str(r_ext["endpoint"]) != "extinction_reached_by_5M"
            or (int(r_ext["prevents_threshold"]), int(r_ext["persists_threshold"]),
                int(r_ext["residual_threshold"])) != (8, 8, 1)
            or str(r_ext["ci_method"]) != "clopper_pearson"):
        raise ValueError("registered R-ext differs")
    gates = P["gates"]
    if (str(gates["s0_arm"]) != NOWD_ARM
            or str(gates["s0_reference_arm"]) != "L1w100_Aexact"
            or int(gates["s0_generator_offset"]) != 0
            or int(gates["s0_steps"]) != 30_000):
        raise ValueError("registered S0 differs")
    # S-op は 9/1 に切替プローブへ改定された（vault 3142500）。旧定義
    # （定常単一タスク・unfit 比較）へ戻すことは許さない。プローブの T は
    # **本走と一致**していなければならない（Issa 裁定「regime を本走に合わせる」）。
    if (str(gates["s_op_mode"]) != "switching_probe"
            or (str(gates["s_op_arm"]), str(gates["s_op_control"])) != ("Im", "im")
            or (int(gates["s_op_task_period"]), int(gates["s_op_n_boundaries"]),
                int(gates["s_op_window_steps"]),
                int(gates["s_op_score_boundaries"])) != (10_000, 150, 100, 50)
            or int(gates["s_op_task_period"]) != int(cfg["phase1"]["task_period"])
            or int(gates["s_op_steps"]) < (int(gates["s_op_n_boundaries"]) + 1)
            * int(gates["s_op_task_period"])
            or str(gates["s_op_metric"]) != "eval_loss_exact"
            or str(gates["s_op_rule"]) != "paired_ci_hi_below_zero"):
        raise ValueError("registered S-op differs")
    if (int(P["bootstrap_B"]), int(P["bootstrap_seed"]),
            str(P["ci_method"])) != (20_000, 20_260_911, "percentile_paired"):
        raise ValueError("registered bootstrap differs")
    iso = P["seed_isolation"]
    if (int(iso["max_exclusions_per_arm"]), int(iso["min_complete_seeds_per_arm"]),
            int(iso["min_paired_seeds"]), int(iso["keep_rng_rows"])) != (2, 8, 8, 10):
        raise ValueError("registered seed-isolation limits differ")
    if list(iso["guarded_tensors"]) != ["W", "b", "v", "c", "u"]:
        raise ValueError("registered guarded tensors differ")
    if int(S["omp_num_threads"]) != 1:
        raise ValueError("registered thread count differs")
    _validate_offset(cfg)
    _validate_preregistration_record(cfg)
    if stage in ("implementation", "selftest", "smoke"):
        return
    missing = preregistration_missing(cfg)
    if stage == "gates":
        # ゲートは事前登録された合否規則を当てる行為である（S-op の失敗は I+ セルを
        # 無効にする）。したがって (a) §8 の予測が入っていること、(b) 事前登録の
        # 成立点である repo 側 spec の単独 commit が済んでいることを要求する。
        blocking = [key for key in missing
                    if key.endswith(("predictions_confirmed", "repo_spec_committed"))]
        if blocking:
            raise ValueError("gates apply registered pass/fail rules and need: "
                             + ", ".join(blocking))
        return
    if missing:
        raise ValueError("ident_mu_2x2 preregistration is not frozen: "
                         + ", ".join(missing))
    if int(C["total_steps"]) != 5_000_000 or list(C["seeds"]) != list(range(10)):
        raise ValueError("full run must be 5M and seeds 0..9")


# ----------------------------------------------------- 腕の状態と 1 step

def set_offset(st: dict, flip: torch.Tensor) -> None:
    """可視入力を作る定数 offset を task 境界で引き直す（spec §3.2）。

    `flip` は現タスクの flip 状態 [R,f]。offset は第1層の `layer_means[0]` に置く
    ので、凍結済みの forward / 厳密レコーダがそのまま可視入力を見る。
    """
    mean, f = st["layer_means"][0], st["n_flip"]
    with torch.no_grad():
        if st["visible"] == "flip0":
            mean[:, :f] = flip - st["flip0"]
            mean[:, f:] = 0.0
        elif st["visible"] == "zero_centered":
            mean[:, :f] = flip
            mean[:, f:] = 0.5
        else:                                    # raw: 可視入力 = 生入力
            mean.zero_()


def code_of(st: dict, flip: torch.Tensor) -> torch.Tensor:
    """I ダイヤル。`zero` 腕では常に同一の零テンソルを返す（確保も乱数も無し）。"""
    return flip if st["code_mode"] == "flip_t" else st["zero_code"]


def forward_ident(st: dict, x: torch.Tensor):
    """可視入力を通した前向き＋出力側バイパス。

    `code` が零の腕では `+0.0` を足すだけなので、バイパス無し実装と bit 一致する
    （S-bypass がこれを検査する）。
    """
    inputs, pres, acts, yhat = forward_centered(st, x)
    code = code_of(st, x[:, :st["n_flip"]])
    return inputs, pres, acts, yhat + (st["u"] * code).sum(dim=1), code


def grads_ident(st: dict, inputs, pres, acts, code: torch.Tensor,
                delta: torch.Tensor):
    gWs, gbs, gv, gc = grads_centered(st["net"], inputs, pres, acts, delta)
    return gWs, gbs, gv, gc, (2.0 * delta)[:, None] * code


def sgd_step_ident(st: dict, grads) -> None:
    gWs, gbs, gv, gc, gu = grads
    st["net"].sgd_step_layers(st["lr"], gWs, gbs, gv, gc)
    st["u"] -= st["lr"][:, None] * gu


def _setup(cfg: dict, arm_name: str, *, generator_offset: int | None = None,
           task_period: int | None = None, seeds: list[int] | None = None) -> dict:
    work = copy.deepcopy(cfg)
    if generator_offset is not None:
        work["common"]["generator_offset"] = int(generator_offset)
    if task_period is not None:
        work["phase1"]["task_period"] = int(task_period)
    if seeds is not None:
        work["common"]["seeds"] = list(seeds)
    if float(work["intervention"]["center_alpha"]) != 0.0:
        raise ValueError("Option B forbids a tracker: center_alpha must be 0")
    arm_cfg = _arm_cfg(work, arm_name)
    if str(arm_cfg["visible"]) not in VISIBLE_MODES:
        raise ValueError(f"unknown visible mode {arm_cfg['visible']!r}")
    if str(arm_cfg["code"]) not in CODE_MODES:
        raise ValueError(f"unknown code mode {arm_cfg['code']!r}")
    st = setup_arm_p1(_base_cfg(work), arm_cfg, "cpu")
    st["net"].set_weight_decay_b(float(arm_cfg["wd_b"]))
    st["visible"] = str(arm_cfg["visible"])
    st["code_mode"] = str(arm_cfg["code"])
    st["n_flip"] = int(work["condA"]["f"])
    st["flip0"] = st["env"].flip_state.clone()
    st["u"] = torch.zeros(st["R"], st["n_flip"], device=st["device"])
    st["zero_code"] = torch.zeros(st["R"], st["n_flip"], device=st["device"])
    st["W_init_flip"] = st["net"].Ws[0][:, :, :st["n_flip"]].clone()
    set_offset(st, st["env"].flip_state)
    _, sanity = exact_layer_record_p1(st, float(work["phase1"]["sigma_degenerate_tol"]))
    if not identity_sanity_pass(sanity, float(work["sanity"]["s1_identity_tol"])):
        raise RuntimeError(f"{arm_name}: preflight identity failed")
    return st


def save_checkpoint_ident(st: dict, step: int, outdir: Path) -> Path:
    path = Path(outdir) / "ckpts" / f"{st['arm']}_step{step}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(step=int(step), arm=st["arm"], net=st["net"].state_dict(),
                    u=st["u"].clone(), offset=st["layer_means"][0].clone(),
                    flip0=st["flip0"].clone(), visible=st["visible"],
                    code_mode=st["code_mode"], env=st["env"].state_dict(),
                    teacher=st["teacher"].state_dict(), runs=st["runs"]), path)
    return path


def train_arm_ident(st: dict, recorder, probe_steps, total: int, outdir: Path,
                    checkpoints, stream_hook=None) -> float:
    """`mlp2_phase1.train_arm_p1` にバイパスと offset 引き直しを足した学習ループ。"""
    probe_set = {int(v) for v in probe_steps}
    checkpoint_set = {int(v) for v in checkpoints}
    env, teacher = st["env"], st["teacher"]
    period = int(st["runs"][0]["period"])
    f = st["n_flip"]
    started = time.time()
    for t in range(total):
        if t in checkpoint_set:
            save_checkpoint_ident(st, t, outdir)
        if t in probe_set:
            recorder(st, t)
        x = env.step()
        y = teacher(x)
        if t % period == 0:                     # 境界（と t=0）で offset を更新
            set_offset(st, x[:, :f])
        if stream_hook is not None:
            stream_hook(t, x, y)
        inputs, pres, acts, yhat, code = forward_ident(st, x)
        sgd_step_ident(st, grads_ident(st, inputs, pres, acts, code, yhat - y))
    if total in probe_set:
        recorder(st, total)
    if total in checkpoint_set:
        save_checkpoint_ident(st, total, outdir)
    return time.time() - started


# ------------------------------------------------- バイパス込みの厳密記録

def visible_expectation(st: dict) -> tuple[torch.Tensor, float]:
    """32 パターン上での可視入力平均の**構成上の期待値**（flip ブロック, rnd 値）。"""
    if st["visible"] == "flip0":
        return st["flip0"].double(), 0.5
    if st["visible"] == "zero_centered":
        return torch.zeros_like(st["flip0"], dtype=torch.float64), 0.0
    return st["env"].flip_state.double(), 0.5


def ident_run_record(st: dict) -> dict:
    """32 パターン厳密サポート上の**バイパス込み**の run 量（独立実装）。

    凍結済みの 2 実装（`exact_wall_record` / `exact_layer_record_p1`）はバイパスを
    知らないが、`code` は task 内で定数なので残差の分散＝`unfit` は両者で一致する。
    その一致自体を S3 の一部として測る（`unfit_bypass`）。`eval_loss_exact` は DC を
    見る量なので、こちらはバイパス込みの値で上書きする。
    """
    net, f = st["net"], st["n_flip"]
    with torch.no_grad():
        X = full_support_ro(st["env"]).double()                 # [P,R,m]
        y = teacher_f64(st["teacher"], X)                       # [P,R]
        xv = X - st["layer_means"][0].double()[None]
        W, b = net.Ws[0].double(), net.bs[0].double()
        z = torch.einsum("rhd,prd->prh", W, xv) + b
        hidden = (torch.relu(z) * net.v.double()).sum(dim=-1) + net.c.double()
        flip = X[0, :, :f]                                      # task 内で定数
        code = flip if st["code_mode"] == "flip_t" else torch.zeros_like(flip)
        bypass = (st["u"].double() * code).sum(dim=1)           # [R]
        residual = hidden + bypass[None] - y
        signal_var = y.var(dim=0, unbiased=False)
        mu_vis = xv.mean(dim=0)                                 # [R,m]
        want_flip, want_rnd = visible_expectation(st)
        visible_ok = (torch.equal(mu_vis[:, :f], want_flip)
                      and torch.equal(mu_vis[:, f:],
                                      torch.full_like(mu_vis[:, f:], want_rnd)))
        power_hidden = hidden.square().mean(dim=0)
        return dict(
            unfit=residual.var(dim=0, unbiased=False) / signal_var,
            eval_loss_exact=residual.square().mean(dim=0),
            residual_mean=residual.mean(dim=0),
            mu_norm_visible=mu_vis.norm(dim=1),
            u_norm=st["u"].double().norm(dim=1),
            bypass_value=bypass,
            # spec §4 の字義は「出力分散のうちバイパスが説明する割合」だが、
            # task 内でバイパスは定数なので分散の分け前は恒等に 0 である。
            # ここでは出力**パワー**の分け前を記録する。
            bypass_share=bypass.square() / (bypass.square() + power_hidden),
            visible_ok=bool(visible_ok),
        )


def _divergence_event_ident(st: dict, step: int) -> dict | None:
    """凍結済みの非有限ガードに、バイパス u を足したもの（spec §3.5）。"""
    event = _numeric_divergence_event(st, step)
    finite = torch.isfinite(st["u"]).all(dim=1)
    bad = [int(st["runs"][ri]["seed"])
           for ri in torch.nonzero(~finite, as_tuple=False).flatten().tolist()]
    if not bad:
        return event
    if event is None:
        period = int(st["runs"][0]["period"])
        event = dict(status=NUMERIC_DIVERGENCE, arm=str(st["arm"]),
                     detected_step=int(step),
                     detected_task=int((step + period - 1) // period) if step else 0,
                     probe_every=None, bad_seeds=[], nonfinite_tensors={},
                     action="arm_stopped_no_rescue")
    for seed in bad:
        event["nonfinite_tensors"].setdefault(str(seed), []).append("bypass.u")
    event["bad_seeds"] = sorted(set(event["bad_seeds"]) | set(bad))
    return event


class IdentRecorder(SeedIsolationRecorder):
    """seed 隔離レコーダに (a) u のガードと quarantine、(b) 可視入力・バイパスの
    記録列、(c) 可視入力の構成チェックを足したもの。

    隔離の本体は `SeedIsolationRecorder.__call__` と同じ順序・同じイベント形式で、
    違いは非有限判定に u が入ることと quarantine が u も零にすることだけである
    （S-iso / S-cap がこの同値性を検査する）。
    """

    def __init__(self, *args, bypass_dc_tol: float = 1e-8, **kwargs):
        super().__init__(*args, **kwargs)
        self.bypass_dc_tol = float(bypass_dc_tol)
        self.max_bypass_relerr = 0.0
        self.n_visible_violations = 0

    def __call__(self, st: dict, step: int) -> None:
        step = int(step)
        if step in self.isolation_guard_steps:
            event = _divergence_event_ident(st, step)
            if event is not None:
                self._isolate(st, step, event)
        n_before = len(self.rows)
        TaskEndRecorder.__call__(self, st, step)
        if len(self.rows) > n_before:
            self._ident_columns(st, n_before)

    def _isolate(self, st: dict, step: int, event: dict) -> None:
        for seed in event["bad_seeds"]:
            seed = int(seed)
            if seed in self.excluded:
                continue
            seed_event = dict(
                status=NUMERIC_DIVERGENCE, arm=self.arm, seed=seed,
                detected_step=int(step), detected_task=int(event["detected_task"]),
                probe_every=self.guard_every,
                nonfinite_tensors=event["nonfinite_tensors"][str(seed)],
                action="quarantine_seed_and_continue",
                exclude_entire_seed_trajectory=True, rescue="none")
            index = _quarantine_seed(st, seed)
            with torch.no_grad():
                st["u"][index].zero_()
            self.excluded[seed] = seed_event
            self.status_dir.mkdir(parents=True, exist_ok=True)
            (self.status_dir / f"{self.arm}_seed{seed}.json").write_text(
                json.dumps(seed_event, indent=2, ensure_ascii=False),
                encoding="utf-8")
            print(f"[{self.arm}] isolate seed {seed} at step {step:,} "
                  f"({len(self.excluded)}/{self.exclusion_cap})", flush=True)
            if len(self.excluded) > self.exclusion_cap:
                raise ExclusionLimitExceeded(dict(
                    status=ARM_INVALID_EXCLUSION_LIMIT, arm=self.arm,
                    detected_step=int(step), excluded_seeds=sorted(self.excluded),
                    cap=self.exclusion_cap))

    def _ident_columns(self, st: dict, n_before: int) -> None:
        record = ident_run_record(st)
        if not record["visible_ok"]:
            self.n_visible_violations += 1
        for ri in range(st["R"]):
            row = self.rows[n_before + ri]
            frozen_unfit = float(row["unfit"])
            mine = float(record["unfit"][ri].item())
            scale = max(abs(frozen_unfit), float(self.bypass_dc_tol))
            self.max_bypass_relerr = max(self.max_bypass_relerr,
                                         abs(mine - frozen_unfit) / scale)
            row["eval_loss_exact_nobypass"] = row["eval_loss_exact"]
            row["eval_loss_exact"] = float(record["eval_loss_exact"][ri].item())
            row["residual_mean"] = float(record["residual_mean"][ri].item())
            row["mu_norm_visible"] = float(record["mu_norm_visible"][ri].item())
            row["u_norm"] = float(record["u_norm"][ri].item())
            row["bypass_value"] = float(record["bypass_value"][ri].item())
            row["bypass_share"] = float(record["bypass_share"][ri].item())
            row["visible_ok"] = int(record["visible_ok"])

    def sanity(self) -> dict:
        out = super().sanity()
        out["max_relerr"]["unfit_bypass"] = float(self.max_bypass_relerr)
        out["bypass_dc_tol"] = self.bypass_dc_tol
        out["n_visible_violations"] = int(self.n_visible_violations)
        out["pass_"] = bool(out["pass_"] and self.n_visible_violations == 0
                            and self.max_bypass_relerr <= self.bypass_dc_tol)
        return out


# ----------------------------------------------------------- 腕を 1 本走らせる

def _median(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else float("nan")


def _boundary_snapshot(st: dict, step: int, sigma_tol: float) -> list[dict]:
    """境界直後（新しい flip / 新しい offset）のスナップショット（spec §4）。"""
    layers, _ = exact_wall_record(st, sigma_tol)
    record = ident_run_record(st)
    rows = []
    for ri, run in enumerate(st["runs"]):
        layer = layers[0]
        p = layer["p_hat"][ri].detach().cpu().numpy()
        sigma = layer["sigma"][ri].detach().cpu().numpy()
        beta = layer["beta"][ri].detach().cpu().numpy()
        b = layer["b"][ri].detach().cpu().numpy()
        valid = layer["valid"][ri].detach().cpu().numpy() & np.isfinite(beta)
        alive = (p > 0) & valid
        B = b / np.where(sigma > 0, sigma, np.nan)
        rows.append(dict(
            arm=st["arm"], seed=int(run["seed"]), step=int(step),
            task_boundary=int(step // int(run["period"])), side="post",
            L1_strict_dead_frac=float((p == 0).mean()),
            L1_submerged_frac=float(
                (layer["pre_max"][ri].detach().cpu().numpy() <= 0).mean()),
            L1_M_median_alive=_median((beta - B)[alive]),
            L1_B_median_alive=_median(B[alive]),
            L1_sigma_median_alive=_median(sigma[alive]),
            mu_norm_visible=float(record["mu_norm_visible"][ri].item()),
            bypass_value=float(record["bypass_value"][ri].item())))
    return rows


def run_arm_ident(cfg: dict, arm_name: str, outdir: Path, *, total_steps: int,
                  task_period: int, guard_every: int,
                  keep_unit_arrays: bool = True, write_logs: bool = True,
                  record_boundaries: bool = True,
                  generator_offset: int | None = None) -> dict:
    st = _setup(cfg, arm_name, generator_offset=generator_offset,
                task_period=task_period)
    P = _P(cfg)
    cap = int(P["seed_isolation"]["max_exclusions_per_arm"])
    sigma_tol = float(cfg["phase1"]["sigma_degenerate_tol"])
    record_steps = list(range(0, total_steps + 1, task_period))
    guard_steps = list(range(0, total_steps + 1, guard_every))
    recorder = IdentRecorder(
        arm_name, float(_arm_cfg(cfg, arm_name)["wd_b"]), st,
        record_steps=record_steps, guard_steps=guard_steps,
        guard_every=guard_every, exclusion_cap=cap,
        status_dir=outdir / "seed_status", sigma_tol=sigma_tol,
        identity_tol=float(cfg["sanity"]["s1_identity_tol"]),
        keep_unit_arrays=keep_unit_arrays,
        bypass_dc_tol=float(P["gates"]["bypass_dc_tol"]))
    probes = sorted(set(record_steps) | set(guard_steps))
    checkpoints = [int(value) for value in cfg["common"].get("checkpoints", [])
                   if int(value) <= total_steps]
    boundary_rows: list[dict] = []

    def stream_hook(t: int, _x: torch.Tensor, _y: torch.Tensor) -> None:
        if record_boundaries and t > 0 and t % task_period == 0:
            boundary_rows.extend(_boundary_snapshot(st, t, sigma_tol))

    print(f"[{arm_name}] visible={st['visible']} code={st['code_mode']} "
          f"offset={cfg['common']['generator_offset'] if generator_offset is None else generator_offset} "
          f"steps={total_steps:,}", flush=True)
    started = time.time()
    limit_event = None
    try:
        elapsed = train_arm_ident(st, recorder, probes, total_steps, outdir,
                                  checkpoints, stream_hook=stream_hook)
    except ExclusionLimitExceeded as exc:
        elapsed = time.time() - started
        limit_event = exc.event
    excluded = sorted(recorder.excluded)
    included = [int(run["seed"]) for run in st["runs"]
                if int(run["seed"]) not in recorder.excluded]
    status = (ARM_INVALID_EXCLUSION_LIMIT if limit_event is not None else
              COMPLETE_WITH_EXCLUSIONS if excluded else COMPLETE)
    raw = recorder.dataframe()
    frame = raw[~raw.seed.isin(excluded)].copy() if len(raw) else raw
    boundaries = pd.DataFrame(boundary_rows)
    if not boundaries.empty:
        boundaries = boundaries[~boundaries.seed.isin(excluded)].copy()
    sanity = recorder.sanity()
    if write_logs and limit_event is None:
        write_arm_npz(outdir, arm_name, float(_arm_cfg(cfg, arm_name)["wd_b"]),
                      st, recorder)
    result = dict(
        arm=arm_name, visible=st["visible"], code=st["code_mode"],
        wd_b=float(_arm_cfg(cfg, arm_name)["wd_b"]), status=status,
        elapsed_sec=float(elapsed), excluded_seeds=excluded, included_seeds=included,
        exclusion_events=[recorder.excluded[seed] for seed in excluded],
        exclusion_cap=cap, limit_event=limit_event, sanity=sanity,
        frame=frame, boundary_frame=boundaries)
    status_path = outdir / "arm_status" / f"{arm_name}.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(
        {key: value for key, value in result.items()
         if key not in {"frame", "boundary_frame"}}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    print(f"[{arm_name}] {status} in {elapsed:.1f}s; excluded={excluded}; "
          f"sanity={'PASS' if sanity['pass_'] else 'FAIL'}", flush=True)
    return result


# ------------------------------------------------------------------ ゲート

def _sha_tensor(value: torch.Tensor) -> str:
    data = np.ascontiguousarray(value.detach().cpu().numpy()).tobytes()
    return hashlib.sha256(data).hexdigest()


def _initial_hashes(st: dict) -> dict:
    out = {f"W{li}": _sha_tensor(v) for li, v in enumerate(st["net"].Ws, 1)}
    out.update({f"b{li}": _sha_tensor(v) for li, v in enumerate(st["net"].bs, 1)})
    out.update(v=_sha_tensor(st["net"].v), c=_sha_tensor(st["net"].c),
               u=_sha_tensor(st["u"]), flip0=_sha_tensor(st["flip0"]),
               env_flip_state=_sha_tensor(st["env"].flip_state),
               eval_fixed=_sha_tensor(st["eval_fixed"]))
    out.update({f"teacher_{k}": _sha_tensor(v)
                for k, v in st["teacher"].state_dict().items()})
    out.update({f"generator_{k}": _sha_tensor(v.get_state())
                for k, v in st["gens"].items()})
    return out


def _write(gate_dir: Path, name: str, payload: dict) -> dict:
    gate_dir.mkdir(parents=True, exist_ok=True)
    (gate_dir / name).write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                                 encoding="utf-8")
    return payload


def s_pair_seq_gate(cfg: dict, gate_dir: Path) -> dict:
    """S-pair / S-seq / S-mu / S3。5 腕が init・入力・教師出力で bit 一致すること。"""
    P = _P(cfg)["gates"]
    steps, grid = int(P["s_pair_steps"]), int(P["s_pair_grid"])
    sigma_tol = float(cfg["phase1"]["sigma_degenerate_tol"])
    reports: dict[str, dict] = {}
    for arm in ARM_ORDER:
        st = _setup(cfg, arm)
        recorder = IdentRecorder(
            arm, 0.0, st, record_steps=[0, steps], guard_steps=[],
            guard_every=grid, exclusion_cap=2,
            status_dir=gate_dir / "_pair_seed_status", sigma_tol=sigma_tol,
            identity_tol=float(cfg["sanity"]["s1_identity_tol"]),
            keep_unit_arrays=False,
            bypass_dc_tol=float(_P(cfg)["gates"]["bypass_dc_tol"]))
        initial = _initial_hashes(st)
        recorder(st, 0)
        mu_start = ident_run_record(st)
        x_stream, y_stream = [], []
        period = int(st["runs"][0]["period"])
        for t in range(steps):
            x = st["env"].step()
            y = st["teacher"](x)
            if t % period == 0:
                set_offset(st, x[:, :st["n_flip"]])
            if t % grid == 0:
                x_stream.append(_sha_tensor(x))
                y_stream.append(_sha_tensor(y))
            inputs, pres, acts, yhat, code = forward_ident(st, x)
            sgd_step_ident(st, grads_ident(st, inputs, pres, acts, code, yhat - y))
        recorder(st, steps)
        mu_end = ident_run_record(st)
        reports[arm] = dict(
            initial=initial, x_stream=x_stream, y_stream=y_stream,
            env_t=int(st["env"].t),
            env_flip_state=_sha_tensor(st["env"].flip_state),
            generator_after={k: _sha_tensor(v.get_state())
                             for k, v in st["gens"].items()},
            s3=recorder.sanity(),
            mu_norm_visible=[float(mu_start["mu_norm_visible"].max().item()),
                             float(mu_end["mu_norm_visible"].max().item())],
            mu_norm_visible_end=mu_end["mu_norm_visible"].tolist(),
            u_norm_end=mu_end["u_norm"].tolist(),
            visible_ok=[bool(mu_start["visible_ok"]), bool(mu_end["visible_ok"])])
    base = reports[ARM_ORDER[0]]
    pair_diff, seq_diff = {}, {}
    for arm in ARM_ORDER[1:]:
        pair_diff[arm] = [field for field in
                          ("initial", "x_stream", "env_t", "env_flip_state",
                           "generator_after")
                          if reports[arm][field] != base[field]]
        seq_diff[arm] = reports[arm]["y_stream"] != base["y_stream"]
    pair = _write(gate_dir, "s_pair.json", dict(
        pass_=bool(all(not v for v in pair_diff.values())),
        steps=steps, grid=grid, compared=list(ARM_ORDER), differences=pair_diff,
        initial_hashes=base["initial"], x_stream_hashes=base["x_stream"]))
    seq = _write(gate_dir, "s_seq.json", dict(
        pass_=bool(not any(seq_diff.values())), steps=steps,
        teacher_stream_differs=seq_diff, y_stream_hashes=base["y_stream"]))
    zero_mu = all(v == 0.0 for arm in ("Im", "im")
                  for v in reports[arm]["mu_norm_visible_end"])
    mplus_equal = (reports["IM"]["mu_norm_visible_end"]
                   == reports["iM"]["mu_norm_visible_end"])
    mu = _write(gate_dir, "s_mu.json", dict(
        pass_=bool(zero_mu and mplus_equal
                   and all(all(reports[arm]["visible_ok"]) for arm in ARM_ORDER)),
        note="構成上ほぼ恒真であり、独立な発見として引いてはならない（spec §9-6）",
        zero_mu_on_M_minus=bool(zero_mu),
        mu_norm_equal_on_M_plus=bool(mplus_equal),
        mu_norm_visible_end={arm: reports[arm]["mu_norm_visible_end"]
                             for arm in ARM_ORDER},
        visible_ok={arm: reports[arm]["visible_ok"] for arm in ARM_ORDER}))
    s3 = _write(gate_dir, "s3.json", dict(
        pass_=bool(all(reports[arm]["s3"]["pass_"] for arm in ARM_ORDER)),
        arms={arm: reports[arm]["s3"] for arm in ARM_ORDER}))
    u_zero = {arm: reports[arm]["u_norm_end"] for arm in ARM_ORDER}
    for name, report in (("S-pair", pair), ("S-seq", seq), ("S-mu", mu),
                         ("S3", s3)):
        if not report["pass_"]:
            raise RuntimeError(f"{name} failed: {report}")
    print("S-pair/S-seq/S-mu/S3: PASS", flush=True)
    return dict(pair=pair, seq=seq, mu=mu, s3=s3, u_norm_end=u_zero)


def s0_replay(cfg: dict, gate_dir: Path) -> dict:
    """S0。`im_nowd` を `center_oracle_0831` の `Aexact` に対して replay する。

    `Aexact` の厳密支持平均 $[\\text{flip}_t,\\ 0.5]$ の減算は M− の可視入力構成と
    構成的に同一なので、系列（`generator_offset=0`）を揃えれば committed 軌跡と
    bit 一致する。**本走（新系列 `20260910`）の腕の正しさは保証しない。**
    """
    P = _P(cfg)["gates"]
    arm, ref_arm = str(P["s0_arm"]), str(P["s0_reference_arm"])
    steps, grid, tol = int(P["s0_steps"]), int(P["s0_grid"]), float(P["s0_tol"])
    ref_dir = Path(ROOT) / str(P["s0_reference_dir"]) / "logs"
    checkpoint = Path(ROOT) / str(P["s0_reference_checkpoint"])
    st = _setup(cfg, arm, generator_offset=int(P["s0_generator_offset"]))

    # (1) init が committed の step0 checkpoint と bit 一致すること
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    ours = st["net"].state_dict()
    state_differences = [key for key, value in saved["net"].items()
                         if not torch.equal(value, ours[key])]
    state_differences += [f"teacher.{key}" for key, value
                          in saved["teacher"].items()
                          if not torch.equal(value, st["teacher"].state_dict()[key])]
    if not torch.equal(saved["env"]["flip_state"], st["env"].flip_state):
        state_differences.append("env.flip_state")

    # (2) 30k step replay して committed ログと突き合わせ
    record_steps = list(range(0, steps + 1, grid))
    recorder = IdentRecorder(
        arm, float(_arm_cfg(cfg, arm)["wd_b"]), st, record_steps=record_steps,
        guard_steps=record_steps, guard_every=grid, exclusion_cap=2,
        status_dir=gate_dir / "_s0_seed_status",
        sigma_tol=float(cfg["phase1"]["sigma_degenerate_tol"]),
        identity_tol=float(cfg["sanity"]["s1_identity_tol"]),
        keep_unit_arrays=False,
        bypass_dc_tol=float(P["bypass_dc_tol"]))
    train_arm_ident(st, recorder, record_steps, steps, gate_dir, [])
    frame = recorder.dataframe()
    differences, max_abs = [], {key: 0.0 for key in P["s0_metrics"]}
    for seed in cfg["common"]["seeds"]:
        mine = frame[frame.seed == int(seed)].set_index("step")
        with np.load(ref_dir / f"{ref_arm}_seed{int(seed)}.npz",
                     allow_pickle=False) as data:
            for step in mine.index:
                found = np.flatnonzero(data["step"] == int(step))
                if len(found) != 1:
                    differences.append(dict(seed=int(seed), step=int(step),
                                            field="step"))
                    continue
                index = int(found[0])
                theirs = {
                    "unfit": float(data["unfit"][index]),
                    "eval_loss_exact": float(data["eval_loss_exact"][index]),
                    "strict_dead_frac": float(
                        (data["layer1_p_hat"][index] == 0).mean()),
                }
                for key, value in theirs.items():
                    column = key if key != "strict_dead_frac" else "L1_strict_dead_frac"
                    delta = abs(float(mine.loc[step, column]) - value)
                    scale = max(abs(value), 1.0)
                    max_abs[key] = max(max_abs[key], delta / scale)
                    if delta / scale > tol:
                        differences.append(dict(seed=int(seed), step=int(step),
                                                field=key, rel=delta / scale))
    report = _write(gate_dir, "s0_replay.json", dict(
        pass_=bool(not state_differences and not differences
                   and recorder.sanity()["pass_"] and not recorder.excluded),
        arm=arm, reference=ref_arm, steps=steps, grid=grid,
        generator_offset=int(P["s0_generator_offset"]),
        checkpoint=str(checkpoint),
        init_state_differences=state_differences,
        max_rel_difference=max_abs, tol=tol, differences=differences[:50],
        note=("実装同値性のみ。本走は generator_offset=20260910 の新系列であり、"
              "committed 腕との bit 一致は取れない（spec §2.3・§9-7）。")))
    if not report["pass_"]:
        raise RuntimeError(f"S0 failed: {report}")
    print("S0: PASS", flush=True)
    return report


def _step_once(st: dict, x: torch.Tensor, y: torch.Tensor) -> None:
    inputs, pres, acts, yhat, code = forward_ident(st, x)
    sgd_step_ident(st, grads_ident(st, inputs, pres, acts, code, yhat - y))


def s1_s2_gate(cfg: dict, gate_dir: Path) -> dict:
    """S1 / S2。b-WD が触るのは隠れ層 bias だけであることの検査。

    凍結済み `bias_wd_0901.s1_s2_algebra`（λ=0 経路の bit 一致・W/v/c 不変・
    b の差が厳密に $-\\eta\\lambda b$・nets.py の AST 検査）をそのまま使い、
    本走で新しく増えた**バイパス $\\mathbf u$ が WD の対象外である**ことだけを
    腕の状態で 1 step 検査して足す。$b$ は init が 0 なので、WD が恒等にならない
    ところまで暖機してから 1 step を比べる。
    """
    algebra = s1_s2_algebra(_compat_cfg(cfg), gate_dir)
    lam = float(cfg["bias_weight_decay"]["lam"])
    warmup = 500
    st = _setup(cfg, "im")
    period = int(st["runs"][0]["period"])
    for t in range(warmup):
        x = st["env"].step()
        y = st["teacher"](x)
        if t % period == 0:
            set_offset(st, x[:, :st["n_flip"]])
        _step_once(st, x, y)
    x, y = st["env"].step(), None
    y = st["teacher"](x)
    before = {key: value.clone() for key, value in st["net"].state_dict().items()}
    u_before = st["u"].clone()
    out = {}
    for tag, value in (("zero", 0.0), ("decay", lam)):
        st["net"].load_state({key: v.clone() for key, v in before.items()})
        st["u"] = u_before.clone()
        st["net"].set_weight_decay_b(value)
        _step_once(st, x, y)
        out[tag] = (st["net"].state_dict(), st["u"].clone())
    zero_net, zero_u = out["zero"]
    decay_net, decay_u = out["decay"]
    lr = st["lr"]
    expected = -lr[:, None] * lam * before["b"]
    eps = float(torch.finfo(before["b"].dtype).eps)
    err = float((decay_net["b"] - zero_net["b"] - expected).abs().max())
    ulp_tol = 4.0 * eps * float(before["b"].abs().max())
    arm_level = dict(
        warmup_steps=warmup, lam=lam,
        b_before_max_abs=float(before["b"].abs().max()),
        W_v_c_untouched=bool(torch.equal(zero_net["W"], decay_net["W"])
                             and torch.equal(zero_net["v"], decay_net["v"])
                             and torch.equal(zero_net["c"], decay_net["c"])),
        bypass_u_untouched=bool(torch.equal(zero_u, decay_u)),
        bias_delta_max_abs_err=err, bias_delta_tol_ulp=ulp_tol,
        bias_delta_signal=float(expected.abs().max()))
    arm_level["pass_"] = bool(arm_level["W_v_c_untouched"]
                              and arm_level["bypass_u_untouched"]
                              and err <= ulp_tol
                              and arm_level["bias_delta_signal"] > 0.0)
    report = _write(gate_dir, "s1_s2.json", dict(
        pass_=bool(algebra["pass_"] and arm_level["pass_"]),
        frozen_algebra_pass=bool(algebra["pass_"]), arm_level=arm_level))
    if not report["pass_"]:
        raise RuntimeError(f"S1/S2 failed: {report}")
    print("S1/S2 (incl. bypass): PASS", flush=True)
    return report


def _train_reference_no_bypass(cfg: dict, arm: str, steps: int) -> dict:
    """凍結済み `train_arm_p1`（バイパスを一切持たない実装）での対照走。"""
    st = _setup(cfg, arm)
    period = int(st["runs"][0]["period"])
    f = st["n_flip"]

    def hook(t: int, x: torch.Tensor, _y: torch.Tensor) -> None:
        if t % period == 0:
            set_offset(st, x[:, :f])

    class _Null:
        def __call__(self, *_args) -> None:
            return None

    train_arm_p1(st, _Null(), [], steps, Path(ROOT) / "results", [],
                 stream_hook=hook)
    return st


def s_bypass_freeze_gate(cfg: dict, gate_dir: Path) -> dict:
    """S-bypass（code≡0 腕はバイパス無し実装と bit 一致）と S-freeze（M− の凍結列）。"""
    P = _P(cfg)["gates"]
    steps = int(P["s_bypass_steps"])
    bypass_rows, freeze_rows = [], []
    for arm in ARM_ORDER:
        st = _setup(cfg, arm)
        period = int(st["runs"][0]["period"])
        for t in range(steps):
            x = st["env"].step()
            y = st["teacher"](x)
            if t % period == 0:
                set_offset(st, x[:, :st["n_flip"]])
            inputs, pres, acts, yhat, code = forward_ident(st, x)
            sgd_step_ident(st, grads_ident(st, inputs, pres, acts, code, yhat - y))
        u_exact_zero = bool(torch.equal(st["u"], torch.zeros_like(st["u"])))
        row = dict(arm=arm, code=st["code_mode"], u_exact_zero=u_exact_zero,
                   u_max_abs=float(st["u"].abs().max().item()))
        if st["code_mode"] == "zero":
            ref = _train_reference_no_bypass(cfg, arm, steps)
            mine = st["net"].Ws + st["net"].bs + [st["net"].v, st["net"].c]
            theirs = ref["net"].Ws + ref["net"].bs + [ref["net"].v, ref["net"].c]
            row["bitwise_equal_to_no_bypass"] = bool(
                all(torch.equal(a, b) for a, b in zip(mine, theirs)))
            row["pass_"] = bool(u_exact_zero and row["bitwise_equal_to_no_bypass"])
        else:
            row["bitwise_equal_to_no_bypass"] = None
            row["pass_"] = bool(not u_exact_zero)     # I+ 腕は u が動いていること
        bypass_rows.append(row)
        if st["visible"] == "zero_centered":
            frozen = torch.equal(st["net"].Ws[0][:, :, :st["n_flip"]],
                                 st["W_init_flip"])
            layers, _ = exact_wall_record(st, float(cfg["phase1"]["sigma_degenerate_tol"]))
            kappa = layers[0]["kappa"]
            valid = layers[0]["valid"] & torch.isfinite(kappa)
            closed = wall_closed_form_kappa(layers[0]["W"], st["n_flip"])
            in_range = bool(((kappa[valid] >= 1.0 - 1e-12)
                             & (kappa[valid] <= math.sqrt(5.0) + 1e-12)).all().item())
            freeze_rows.append(dict(
                arm=arm, flip_columns_frozen=bool(frozen),
                kappa_in_range=in_range,
                kappa_min=float(kappa[valid].min().item()),
                kappa_max=float(kappa[valid].max().item()),
                kappa_closed_form_max_relerr=float(
                    ((kappa[valid] - closed[valid]).abs()
                     / closed[valid].abs().clamp_min(1e-300)).max().item()),
                pass_=bool(frozen and in_range)))
    bypass = _write(gate_dir, "s_bypass.json", dict(
        pass_=bool(all(row["pass_"] for row in bypass_rows)),
        steps=steps, arms=bypass_rows))
    freeze = _write(gate_dir, "s_freeze.json", dict(
        pass_=bool(freeze_rows and all(row["pass_"] for row in freeze_rows)),
        steps=steps, arms=freeze_rows))
    for name, report in (("S-bypass", bypass), ("S-freeze", freeze)):
        if not report["pass_"]:
            raise RuntimeError(f"{name} failed: {report}")
    print("S-bypass/S-freeze: PASS", flush=True)
    return dict(bypass=bypass, freeze=freeze)


def _train_step_ident(st: dict) -> torch.Tensor:
    x = st["env"].step()
    y = st["teacher"](x)
    period = int(st["runs"][0]["period"])
    if st["env"].t % period == 1:                 # env.step() 後なので 1 ずれる
        set_offset(st, x[:, :st["n_flip"]])
    inputs, pres, acts, yhat, code = forward_ident(st, x)
    sgd_step_ident(st, grads_ident(st, inputs, pres, acts, code, yhat - y))
    return x


def isolation_gates(cfg: dict, gate_dir: Path) -> dict:
    """S-iso（隔離後も他 seed が bit 一致）と S-cap（3 本目で腕停止）。"""
    arm = "Im"
    sigma_tol = float(cfg["phase1"]["sigma_degenerate_tol"])
    identity_tol = float(cfg["sanity"]["s1_identity_tol"])
    control, isolated = _setup(cfg, arm), _setup(cfg, arm)
    bad_index = 1
    bad_seed = int(isolated["runs"][bad_index]["seed"])
    isolated["u"][bad_index, 0] = float("inf")     # u だけを壊す（u もガード対象）
    rec = IdentRecorder(
        arm, 0.0, isolated, record_steps=[], guard_steps=[0], guard_every=1_000,
        exclusion_cap=2, status_dir=gate_dir / "_synthetic_seed_status",
        sigma_tol=sigma_tol, identity_tol=identity_tol, keep_unit_arrays=False)
    rec(isolated, 0)
    keep = torch.tensor([i for i in range(len(cfg["common"]["seeds"]))
                         if i != bad_index])
    streams_equal = True
    for _ in range(100):
        streams_equal &= torch.equal(_train_step_ident(control),
                                     _train_step_ident(isolated))
    tensors_c = (control["net"].Ws + control["net"].bs
                 + [control["net"].v, control["net"].c, control["u"]])
    tensors_i = (isolated["net"].Ws + isolated["net"].bs
                 + [isolated["net"].v, isolated["net"].c, isolated["u"]])
    state_equal = all(torch.equal(a[keep], b[keep])
                      for a, b in zip(tensors_c, tensors_i))
    offset_equal = torch.equal(control["layer_means"][0][keep],
                               isolated["layer_means"][0][keep])
    env_equal = (torch.equal(control["env"].flip_state, isolated["env"].flip_state)
                 and control["env"].t == isolated["env"].t)
    iso = _write(gate_dir, "s_iso.json", dict(
        pass_=bool(set(rec.excluded) == {bad_seed} and streams_equal
                   and state_equal and offset_equal and env_equal),
        injected="bypass.u", isolated_seed=bad_seed,
        unaffected_state_bitwise_equal=bool(state_equal),
        unaffected_offset_bitwise_equal=bool(offset_equal),
        input_stream_bitwise_equal=bool(streams_equal),
        env_state_equal=bool(env_equal),
        excluded=sorted(rec.excluded),
        nonfinite_tensors={str(k): v["nonfinite_tensors"]
                           for k, v in rec.excluded.items()}))

    cap_state = _setup(cfg, arm)
    for index in (0, 1, 2):
        cap_state["net"].Ws[0][index, 0, 0] = float("inf")
    cap_rec = IdentRecorder(
        arm, 0.0, cap_state, record_steps=[], guard_steps=[0], guard_every=1_000,
        exclusion_cap=2, status_dir=gate_dir / "_synthetic_cap_status",
        sigma_tol=sigma_tol, identity_tol=identity_tol, keep_unit_arrays=False)
    event = None
    try:
        cap_rec(cap_state, 0)
    except ExclusionLimitExceeded as exc:
        event = exc.event
    cap = _write(gate_dir, "s_cap.json", dict(
        pass_=bool(event and event["status"] == ARM_INVALID_EXCLUSION_LIMIT
                   and event["excluded_seeds"] == [0, 1, 2]), event=event))
    for name, report in (("S-iso", iso), ("S-cap", cap)):
        if not report["pass_"]:
            raise RuntimeError(f"{name} failed: {report}")
    print("S-iso/S-cap: PASS", flush=True)
    return dict(iso=iso, cap=cap)


def s_count_gate(cfg: dict, gate_dir: Path) -> dict:
    """S-count。ブロック内 task 末点が 50 個ちょうど、境界数と flip 変化数が一致。"""
    P = _P(cfg)
    n_tasks = int(P["gates"]["s_count_tasks"])
    grid = np.arange(1, n_tasks + 1, dtype=int)
    blocks = (grid - 1) // int(P["block_tasks"]) + 1
    counts = {str(block): int((blocks == block).sum())
              for block in range(1, int(P["n_blocks"]) + 1)}
    st = _setup(cfg, "im", seeds=[0])
    period = int(st["runs"][0]["period"])
    states = []
    for _ in range(50):
        st["env"].segment(period)
        states.append(st["env"].flip_state.clone())
    changes = sum(not torch.equal(a, b) for a, b in zip(states[:-1], states[1:]))
    report = _write(gate_dir, "s_count.json", dict(
        pass_=bool(all(v == int(P["block_tasks"]) for v in counts.values())
                   and changes == len(states) - 1
                   and sum(counts.values()) == n_tasks),
        block_task_end_counts=counts, tested_tasks=[1, n_tasks],
        flip_state_changes=int(changes), boundary_comparisons=len(states) - 1))
    if not report["pass_"]:
        raise RuntimeError(f"S-count failed: {report}")
    print("S-count: PASS", flush=True)
    return report


def _s_op_probe(cfg: dict, arm: str, *, steps: int, period: int,
                window: int) -> dict:
    """切替プローブ 1 本。境界直後 `window` step の `eval_loss_exact` を集める。

    測るのは各 step の**更新前**の状態（offset は境界で引き直し済み）なので、
    境界直後の第 1 点はタスクが切り替わった瞬間の DC 誤差そのものになる。
    """
    st = _setup(cfg, arm, task_period=period)
    flip = st["n_flip"]
    boundaries: dict[int, list[np.ndarray]] = {}
    u_norm: dict[int, np.ndarray] = {}
    flipped_bit: dict[int, np.ndarray] = {}
    previous = st["env"].flip_state.clone()
    for t in range(steps):
        x = st["env"].step()
        y = st["teacher"](x)
        index, offset = divmod(t, period)
        if t % period == 0:
            set_offset(st, x[:, :flip])
            if index > 0:
                changed = (st["env"].flip_state != previous)
                flipped_bit[index] = changed.float().argmax(dim=1).numpy().copy()
                previous = st["env"].flip_state.clone()
        if index > 0 and offset < window:
            record = ident_run_record(st)
            boundaries.setdefault(index, []).append(
                record["eval_loss_exact"].numpy().copy())
            if offset == 0:
                u_norm[index] = record["u_norm"].numpy().copy()
        inputs, pres, acts, yhat, code = forward_ident(st, x)
        sgd_step_ident(st, grads_ident(st, inputs, pres, acts, code, yhat - y))
    return dict(
        arm=arm,
        boundary_means={index: np.mean(values, axis=0)
                        for index, values in boundaries.items()},
        window_points={index: len(values) for index, values in boundaries.items()},
        flipped_bit=flipped_bit, u_norm=u_norm)


def _s_op_identification(flipped_bit: dict, scored: list[int]) -> dict:
    """採点対象の境界で「その bit が過去に何回反転済みか」を数える（REPORT_ONLY）。

    バイパスの $u_k$ は bit $k$ が両状態で観測されて初めて同定される。境界での
    自動的な跳ね $\\pm u_k$ は、同定前は符号すら当てにならない（$u_k$ は現タスクの
    DC の分け前を映すだけで、教師 DC の**変化**の符号は映さない）。したがって
    S-op が FAIL したとき「バイパスが情報を運べない」と「プローブが短くて $u$ が
    同定されていない」を切り分けるにはこの数が要る。
    """
    order = sorted(flipped_bit)
    counts, prior = [], {}
    for index in order:
        bits = flipped_bit[index]
        if index in scored:
            counts.append([int(prior.get((seed, int(bit)), 0))
                           for seed, bit in enumerate(bits)])
        for seed, bit in enumerate(bits):
            prior[(seed, int(bit))] = prior.get((seed, int(bit)), 0) + 1
    flat = [value for row in counts for value in row]
    return dict(
        prior_flips_of_the_flipped_bit=counts,
        mean_prior_flips=float(np.mean(flat)) if flat else float("nan"),
        frac_scored_boundaries_with_an_identified_bit=(
            float(np.mean([value > 0 for value in flat])) if flat else float("nan")),
        n_bits=15, n_boundaries=len(order))


def s_op_gate(cfg: dict, gate_dir: Path) -> dict:
    """S-op（操作チェック・9/1 改の切替プローブ）。

    旧定義（定常単一タスク・`unfit` 比較）は二重に成立不能だった —— 定常タスクでは
    `code` が定数なのでバイパスは出力バイアス $c$ と冗長で、しかも `unfit` は
    32 パターン上の分散比なので DC を見ない。改定版は切替を入れ、**境界直後
    100 step 窓の `eval_loss_exact`（DC 感応の素の MSE）**の平均を最後の
    `s_op_score_boundaries` 境界で取り、paired 差（`Im` − code ゼロ化対照）の
    95% CI 上端 < 0 を要求する。失敗時の帰結は不変（I+ セルを無効と宣言し
    主判定を出さない）。

    **regime は本走と一致させる**（$T=10^4$・150 境界・後半 50 で採点）。$u_k$ は
    bit $k$ が両状態で観測されて初めて同定され、それまで境界での跳ね $\\pm u_k$ は
    符号が当てにならない。短い $T$ や少ない境界数では、バイパスの能力ではなく
    同定の薄さを測ってしまう（`identification` ブロックで常時監視する）。
    """
    P = _P(cfg)["gates"]
    if str(P["s_op_mode"]) != "switching_probe":
        raise ValueError("S-op must be the registered switching probe")
    steps, period = int(P["s_op_steps"]), int(P["s_op_task_period"])
    window, scored = int(P["s_op_window_steps"]), int(P["s_op_score_boundaries"])
    expected_boundaries = int(P["s_op_n_boundaries"])
    if window > period or scored > expected_boundaries:
        raise ValueError("S-op probe geometry is inconsistent")
    treat, control = str(P["s_op_arm"]), str(P["s_op_control"])
    probes = {arm: _s_op_probe(cfg, arm, steps=steps, period=period, window=window)
              for arm in (treat, control)}
    boundaries = sorted(probes[treat]["boundary_means"])
    geometry_ok = bool(
        len(boundaries) == expected_boundaries
        and sorted(probes[control]["boundary_means"]) == boundaries
        and all(count == window for probe in probes.values()
                for count in probe["window_points"].values()))
    last = boundaries[-scored:]
    score = {arm: np.mean([probes[arm]["boundary_means"][i] for i in last], axis=0)
             for arm in (treat, control)}
    delta = score[treat] - score[control]
    log_delta = np.log10(score[treat]) - np.log10(score[control])
    rng = np.random.default_rng(int(P["s_op_bootstrap_seed"]))
    n = len(cfg["common"]["seeds"])
    draws = rng.integers(0, n, size=(int(_P(cfg)["bootstrap_B"]), n))
    ci = paired_ci(_compat_cfg(cfg), delta, draws)
    ci_log = paired_ci(_compat_cfg(cfg), log_delta, draws)
    u_final = {arm: [float(v) for v in probes[arm]["u_norm"][boundaries[-1]]]
               for arm in (treat, control)}
    report = _write(gate_dir, "s_op.json", dict(
        pass_=bool(ci["ci_hi"] < 0.0 and geometry_ok),
        # 幾何は config から書き出す（数字を二重に持つと片方が腐る）。
        registered_rule=(f"switching probe at the run's regime (T={period}): mean "
                         f"{P['s_op_metric']} over the {window}-step window after "
                         f"each boundary, averaged over the last {scored} of "
                         f"{expected_boundaries} boundaries; paired "
                         f"({treat} - {control}) 95% CI upper bound must be < 0"),
        mode=str(P["s_op_mode"]), treat=treat, control=control,
        steps=steps, task_period=period, window_steps=window,
        n_boundaries=len(boundaries), scored_boundaries=last,
        geometry_ok=geometry_ok,
        eval_loss_delta={key: float(ci[key])
                         for key in ("point", "ci_lo", "ci_hi")},
        ci_degenerate=int(ci["ci_degenerate"]),
        log10_eval_loss_delta_report_only={
            key: float(ci_log[key]) for key in ("point", "ci_lo", "ci_hi")},
        per_seed_delta=[float(v) for v in delta],
        per_seed_score={arm: [float(v) for v in score[arm]]
                        for arm in (treat, control)},
        u_norm_at_last_boundary=u_final,
        identification=_s_op_identification(probes[treat]["flipped_bit"], last),
        note=("`eval_loss_exact` は DC 感応の素の MSE なので、バイパスが運ぶ"
              "タスク内定数がここには映る（`unfit` は分散比なので映らない）。"
              "FAIL のときは `identification` を先に見ること —— $u_k$ が同定"
              "されていない境界では、境界での自動的な跳ね $\\pm u_k$ の符号は"
              "教師 DC の変化と無相関で、バイパスの能力を測れていない。")))
    if not report["pass_"]:
        raise RuntimeError(f"S-op failed (I+ cells are invalid): {report}")
    print("S-op: PASS", flush=True)
    return report


SELFTEST_FILES = ("s_pair.json", "s_seq.json", "s0_replay.json",
                  "s1_s2_algebra.json", "s1_s2.json", "s_mu.json", "s3.json",
                  "s_bypass.json", "s_freeze.json", "s_iso.json", "s_cap.json",
                  "s_count.json")
GATE_FILES = SELFTEST_FILES + ("s_op.json",)


def run_selftests(cfg: dict, gate_dir: Path) -> dict:
    """測定を含まない実装同一性ゲート。事前予測より前に回してよい。

    S0 は committed 済みの `Aexact` 軌跡との突き合わせであり、本走の新系列に
    ついては何も測らないのでここに含める。
    """
    gate_dir.mkdir(parents=True, exist_ok=True)
    out = dict(s_pair_seq=s_pair_seq_gate(cfg, gate_dir),
               s0=s0_replay(cfg, gate_dir),
               s1_s2=s1_s2_gate(cfg, gate_dir),
               s_bypass_freeze=s_bypass_freeze_gate(cfg, gate_dir),
               isolation=isolation_gates(cfg, gate_dir),
               s_count=s_count_gate(cfg, gate_dir))
    _require_gates(gate_dir, SELFTEST_FILES)
    print(f"ALL SELFTESTS PASS -> {gate_dir}", flush=True)
    return out


def run_gates(cfg: dict, gate_dir: Path) -> dict:
    out = run_selftests(cfg, gate_dir)
    out["s_op"] = s_op_gate(cfg, gate_dir)
    _require_gates(gate_dir, GATE_FILES)
    print(f"ALL GATES PASS -> {gate_dir}", flush=True)
    return out


def _gates_for_run(gate_dir: Path) -> tuple[dict, bool]:
    """本走の起動可否。**S-op だけは失敗しても走を止めない。**

    spec §7 の失敗欄は、S-op 以外のすべての行が「本走禁止」であるのに対し、
    S-op だけ「**I+ セルを無効と宣言し主判定を出さない**」と書かれている。
    より特定的なこの行が、表の直後の一般文（「PASS でなければ本走は起動しない」）
    に優先すると読む。したがって S-op の FAIL は I+ セルを落とすだけで、
    I− 側の登録済み対比（M 主効果 (i)）と R-ext は残る。
    """
    reports, failed = {}, []
    for name in GATE_FILES:
        path = Path(gate_dir) / name
        if not path.exists():
            raise RuntimeError(f"missing gate: {path}")
        reports[name] = json.loads(path.read_text(encoding="utf-8"))
        if not reports[name].get("pass_"):
            failed.append(name)
    if failed and failed != ["s_op.json"]:
        raise RuntimeError(f"failed gates: {failed}")
    return reports, bool(failed)


def _require_gates(gate_dir: Path, names=GATE_FILES) -> dict:
    reports = {}
    for name in names:
        path = Path(gate_dir) / name
        if not path.exists():
            raise RuntimeError(f"missing gate: {path}")
        report = json.loads(path.read_text(encoding="utf-8"))
        if not report.get("pass_"):
            raise RuntimeError(f"failed gate: {path}")
        reports[name] = report
    return reports


# -------------------------------------------------------------- 集計

def block_levels(cfg: dict, frame: pd.DataFrame) -> pd.DataFrame:
    P, floor = _P(cfg), float(_P(cfg)["unfit_floor"])
    excluded = {"arm", "seed", "step", "task", "wd_b", "visible_ok"}
    numeric = [c for c in frame.select_dtypes(include=[np.number]).columns
               if c not in excluded]
    rows = []
    for (arm, seed), group in frame.groupby(["arm", "seed"], sort=True):
        group = group[group.task > 0].copy()
        group["block"] = ((group["task"] - 1) // int(P["block_tasks"]) + 1).astype(int)
        group["log10_unfit"] = np.log10(np.maximum(group["unfit"].to_numpy(), floor))
        group["at_floor"] = group["unfit"].to_numpy() <= floor
        for block, gb in group.groupby("block"):
            row = dict(arm=arm, seed=int(seed), block=int(block),
                       task_lo=int(gb.task.min()), task_hi=int(gb.task.max()),
                       n_task_ends=int(len(gb)),
                       mean_log10_unfit=float(gb.log10_unfit.mean()),
                       log10_mean_unfit=float(np.log10(max(float(gb.unfit.mean()),
                                                           floor))),
                       floor=floor, floor_frac=float(gb.at_floor.mean()))
            row.update({c: float(gb[c].mean()) for c in numeric if c != "unfit"})
            row["unfit"] = float(gb.unfit.mean())
            rows.append(row)
    out = pd.DataFrame(rows).sort_values(["arm", "seed", "block"])
    if not out.empty and not (out.n_task_ends == int(P["block_tasks"])).all():
        bad = out[out.n_task_ends != int(P["block_tasks"])]
        raise RuntimeError(f"S-count failed in realized data: "
                           f"{bad.to_dict('records')[:10]}")
    return out


def onset_task(values: np.ndarray, tasks: np.ndarray, *, threshold: float,
               window: int, censor_task: int, require_prior_below: bool) -> dict:
    """E-onset（spec §6.2 ＋ §実装注 3）。

    10 task 窓の移動中央値（trailing, min_periods=1）で平滑し、**一度閾値を
    下回ったあとの最初の上抜け** task を返す。一度も下回らない seed は
    `never_below` を立て、tau は打ち切り値にする（対比からは外さず旗で報告する）。
    """
    smooth = pd.Series(values).rolling(window, min_periods=1).median().to_numpy()
    below = smooth < threshold
    if require_prior_below:
        if not below.any():
            return dict(tau=int(censor_task), censored=True, never_below=True,
                        first_below_task=None)
        start = int(np.argmax(below))
        above = np.flatnonzero(~below[start:])
        if not above.size:
            return dict(tau=int(censor_task), censored=True, never_below=False,
                        first_below_task=int(tasks[start]))
        return dict(tau=int(tasks[start + int(above[0])]), censored=False,
                    never_below=False, first_below_task=int(tasks[start]))
    above = np.flatnonzero(~below)
    if not above.size:
        return dict(tau=int(censor_task), censored=True, never_below=False,
                    first_below_task=None)
    return dict(tau=int(tasks[int(above[0])]), censored=False, never_below=False,
                first_below_task=None)


def onset_table(cfg: dict, frame: pd.DataFrame) -> pd.DataFrame:
    P = _P(cfg)
    onset, floor = P["onset"], float(P["unfit_floor"])
    thresholds = [float(onset["threshold"])] + [
        float(v) for v in onset["sensitivity_thresholds"]]
    rows = []
    for (arm, seed), group in frame.groupby(["arm", "seed"], sort=True):
        group = group[group.task > 0].sort_values("task")
        values = np.log10(np.maximum(group["unfit"].to_numpy(), floor))
        tasks = group["task"].to_numpy()
        for threshold in thresholds:
            result = onset_task(
                values, tasks, threshold=threshold,
                window=int(onset["window_tasks"]),
                censor_task=int(onset["censor_task"]),
                require_prior_below=bool(onset["require_prior_below"]))
            rows.append(dict(arm=arm, seed=int(seed), threshold=threshold,
                             primary=threshold == float(onset["threshold"]),
                             **result))
    return pd.DataFrame(rows)


def _draws(cfg: dict, n: int) -> np.ndarray:
    rng = np.random.default_rng(int(_P(cfg)["bootstrap_seed"]))
    return rng.integers(0, n, size=(int(_P(cfg)["bootstrap_B"]), n))


def _ci(cfg: dict, values: np.ndarray, draws: np.ndarray) -> dict:
    return paired_ci(_compat_cfg(cfg), np.asarray(values, dtype=np.float64), draws)


def _fmt_ci(ci: dict) -> str:
    return (f"{ci['point']:+.4f} CI [{ci['ci_lo']:+.4f}, {ci['ci_hi']:+.4f}]"
            f"; ci_degenerate={bool(ci['ci_degenerate'])}")


def band_of(ci: dict, margin: float) -> str:
    """帯 ±margin に対する位置。IN / OUT_POS / OUT_NEG / STRADDLE。"""
    if ci["ci_lo"] >= -margin and ci["ci_hi"] <= margin:
        return "IN"
    if ci["ci_lo"] > margin:
        return "OUT_POS"
    if ci["ci_hi"] < -margin:
        return "OUT_NEG"
    return "STRADDLE"


def classify_2x2(bands: dict, interaction_band: str) -> str:
    """spec §6.3 の決定木。`bands` は 4 対比 -> IN/OUT_POS/OUT_NEG/STRADDLE。

    評価順は spec の番号どおり。1 交互作用 -> 2 M 支配 -> 3 I 支配 -> 4 両方 ->
    5 どちらも無し -> 6 同一因子の 2 本が食い違う -> 7 帯をまたぐ。
    """
    if interaction_band in ("OUT_POS", "OUT_NEG"):
        return INTERACTION_DOMINATES

    def outside(values) -> bool:
        return all(value.startswith("OUT") for value in values)

    def inside(values) -> bool:
        return all(value == "IN" for value in values)

    m = [bands["M_i"], bands["M_ii"]]
    i = [bands["I_i"], bands["I_ii"]]
    if outside(m) and inside(i):
        return MU_DOMINANT
    if outside(i) and inside(m):
        return IDENT_DOMINANT
    if outside(m) and outside(i):
        return BOTH_MATTER
    if inside(m) and inside(i):
        return NEITHER_MATTERS
    for pair in (m, i):
        if (any(value == "IN" for value in pair)
                and any(value.startswith("OUT") for value in pair)):
            return EFFECT_LEVEL_DEPENDENT
    return INCONCLUSIVE_WIDE


def _endpoint_verdict(cfg: dict, values: dict[str, np.ndarray], draws: np.ndarray,
                      *, margin: float, interaction_margin: float) -> dict:
    cis = {name: _ci(cfg, values[a] - values[b], draws)
           for name, (a, b) in FACTOR_CONTRASTS.items()}
    interaction = _ci(cfg, (values["IM"] - values["iM"])
                      - (values["Im"] - values["im"]), draws)
    bands = {name: band_of(ci, margin) for name, ci in cis.items()}
    interaction_band = band_of(interaction, interaction_margin)
    return dict(cis=cis, interaction=interaction, bands=bands,
                interaction_band=interaction_band,
                verdict=classify_2x2(bands, interaction_band),
                margin=margin, interaction_margin=interaction_margin)


def extinction_table(cfg: dict, frame: pd.DataFrame) -> pd.DataFrame:
    """seed ごとの全滅到達（第1層の全ユニットが dead）と到達 task。

    登録規則は `strict_dead_frac == 1.0`。`alive == 0`（σ 退化ユニットも
    非 alive に数える別定義）を併記し、食い違いを `agree` で出す。
    """
    rows = []
    for (arm, seed), group in frame.groupby(["arm", "seed"], sort=True):
        group = group[group.task > 0].sort_values("task")
        tasks = group["task"].to_numpy()
        dead = np.flatnonzero(group["L1_strict_dead_frac"].to_numpy() >= 1.0)
        alive_zero = np.flatnonzero(group["L1_alive"].to_numpy() == 0)
        rows.append(dict(
            arm=arm, seed=int(seed), extinct=bool(dead.size),
            extinction_task=int(tasks[dead[0]]) if dead.size else int(
                _P(cfg)["onset"]["censor_task"]),
            alive_zero=bool(alive_zero.size),
            alive_zero_task=int(tasks[alive_zero[0]]) if alive_zero.size else int(
                _P(cfg)["onset"]["censor_task"]),
            agree=bool(bool(dead.size) == bool(alive_zero.size))))
    return pd.DataFrame(rows)


def classify_r_ext(treat_count: int, control_count: int, *, prevents: int,
                   persists: int, residual: int) -> str:
    """spec §6.4 の R-ext ラベル。`treat` = `im`（λ>0）、`control` = `im_nowd`。

    しきい値は spec の字義どおり 10 seed 中の**本数**である。
    """
    if control_count >= prevents and treat_count <= residual:
        return BWD_PREVENTS_EXTINCTION
    if treat_count >= persists:
        return EXTINCTION_PERSISTS
    return PARTIAL_RESCUE


def _r_ext(cfg: dict, frame: pd.DataFrame, onsets: pd.DataFrame,
           levels: pd.DataFrame, meta: dict, valid: dict, included: dict,
           b10: int) -> dict:
    """登録副判定 R-ext（spec §6.4）: b-WD は `Aexact` 型の全滅を止めるか。

    **死についての主張であって機能の主張ではない。** `BWD_PREVENTS_EXTINCTION`
    を `LOP_CURED` と読み替えてはいけない（spec §9-4）。
    """
    R = _P(cfg)["r_ext"]
    treat, control = str(R["arms"][0]), str(R["arms"][1])
    minimum = int(_P(cfg)["seed_isolation"]["min_paired_seeds"])
    if not (valid[treat] and valid[control]):
        return dict(verdict=R_EXT_INVALID_TOO_FEW_PAIRED,
                    evidence=f"{treat}={meta[treat]['status']}; "
                             f"{control}={meta[control]['status']}")
    seeds = sorted(included[treat] & included[control])
    if len(seeds) < minimum:
        return dict(verdict=R_EXT_INVALID_TOO_FEW_PAIRED,
                    evidence=f"common complete seeds={seeds}; n={len(seeds)} "
                             f"< {minimum}")
    table = extinction_table(cfg, frame)
    table = table[table.seed.isin(seeds)]
    counts, cis, tasks = {}, {}, {}
    for arm in (treat, control):
        group = table[table.arm == arm]
        counts[arm] = int(group.extinct.sum())
        cis[arm] = clopper_pearson(counts[arm], len(seeds))
        reached = group[group.extinct]
        tasks[arm] = ([int(reached.extinction_task.min()),
                       int(reached.extinction_task.max())]
                      if len(reached) else None)
    verdict = classify_r_ext(counts[treat], counts[control],
                             prevents=int(R["prevents_threshold"]),
                             persists=int(R["persists_threshold"]),
                             residual=int(R["residual_threshold"]))
    level = {arm: float(levels[(levels.arm == arm) & (levels.block == b10)
                               & levels.seed.isin(seeds)].mean_log10_unfit.mean())
             for arm in (treat, control)}
    dead = {arm: float(levels[(levels.arm == arm) & (levels.block == b10)
                              & levels.seed.isin(seeds)
                              ].L1_strict_dead_frac.mean())
            for arm in (treat, control)}
    tau = {arm: float(onsets[(onsets.arm == arm) & onsets.primary
                             & onsets.seed.isin(seeds)].tau.median())
           for arm in (treat, control)}
    disagreements = table[~table.agree]
    return dict(
        verdict=verdict, ci_basis="Clopper-Pearson",
        seeds=seeds, n=len(seeds), counts=counts,
        ci={arm: [float(lo), float(hi)] for arm, (lo, hi) in cis.items()},
        extinction_task_range=tasks, level_B10=level, dead_B10=dead,
        median_tau=tau,
        n_rule_disagreements=int(len(disagreements)),
        reference=dict(R["reference"]),
        evidence=(f"n={len(seeds)}"
                  + ("" if len(seeds) == 10 else " (registered thresholds are"
                     " literal counts out of 10)")
                  + f"; {control} extinct={counts[control]}/{len(seeds)} "
                    f"CI={cis[control][0]:.3f}-{cis[control][1]:.3f} "
                    f"tasks={tasks[control]}; "
                    f"{treat} extinct={counts[treat]}/{len(seeds)} "
                    f"CI={cis[treat][0]:.3f}-{cis[treat][1]:.3f} "
                    f"tasks={tasks[treat]}; "
                    f"reference Aexact={R['reference']['reached']}/"
                    f"{R['reference']['of']} tasks={list(R['reference']['task_range'])}; "
                    f"E-level {treat}={level[treat]:.4f} / "
                    f"{control}={level[control]:.4f} "
                    f"(diff={level[treat] - level[control]:+.4f} dex, REPORT); "
                    f"strict_dead B10 {treat}={dead[treat]:.4f} / "
                    f"{control}={dead[control]:.4f}; "
                    f"median tau {treat}={tau[treat]:.1f} / "
                    f"{control}={tau[control]:.1f}; "
                    f"alive==0 rule disagreements={len(disagreements)}"))


def active_cells(cfg: dict, *, i_cells_invalid: bool) -> list[str]:
    """S-op が落ちたら I+ セルは無効（spec §7 の S-op 行の失敗時帰結）。

    S-op 行の失敗欄だけが他の行の「本走禁止」と違い、**「I+ セルを無効と宣言し
    主判定を出さない」**である。したがって S-op の FAIL は走そのものを止めず、
    I− 側の登録済み対比（M 主効果 (i) `iM − im`）と R-ext を残す。
    """
    if not i_cells_invalid:
        return list(CELL_ORDER)
    return [name for name in CELL_ORDER if not bool(_arm_cfg(cfg, name)["I"])]


def _single_contrast_verdict(cfg: dict, values: dict[str, np.ndarray],
                             draws: np.ndarray, *, margin: float) -> dict:
    """I+ セルが無効なときに残る、登録済みの M 主効果 (i) `iM − im` 1 本。

    新しい判定語彙は作らない。登録済みの帯（$\\pm\\Delta$）に対する位置
    （`IN` / `OUT_POS` / `OUT_NEG` / `STRADDLE`）をそのまま出すだけで、
    2×2 の決定木は回さない（主判定は出さない）。
    """
    ci = _ci(cfg, values["iM"] - values["im"], draws)
    band = band_of(ci, margin)
    return dict(cis={"M_i": ci}, interaction=None, bands={"M_i": band},
                interaction_band="not computed (I+ cells invalid)",
                verdict=band, margin=margin,
                interaction_margin="not applicable")


def classify_r_ext_mplus(ci: dict) -> str:
    """R-ext-M+ の符号ベース判定（追補2 §3）。

    等価限界を登録していないので、**「効果なし」は主張できない**。CI が 0 を
    跨いだときのラベルは「区別できなかった」であって、タイトな null ではない。
    """
    if ci["ci_lo"] > 0.0:
        return BWD_DELAYS_EXTINCTION_UNDER_MU
    if ci["ci_hi"] < 0.0:
        return BWD_ACCELERATES_EXTINCTION_UNDER_MU
    return BWD_EFFECT_NOT_DISTINGUISHED_UNDER_MU


def _r_ext_mplus(cfg: dict, frame: pd.DataFrame, levels: pd.DataFrame,
                 meta: dict, valid: dict, included: dict, b02: int,
                 b10: int) -> dict:
    """登録副判定 R-ext-M+（追補2 §3）: µ が立っていても b-WD は何かしているか。

    二値の全滅到達は両腕とも飽和する見込みなので（教訓⑪）、主 endpoint は
    **全滅到達 task**（seed 対応・右打ち切り）。
    """
    R = _P(cfg)["r_ext_mplus"]
    treat, control = str(R["arms"][0]), str(R["arms"][1])
    minimum = int(_P(cfg)["seed_isolation"]["min_paired_seeds"])
    if not (valid.get(treat) and valid.get(control)):
        return dict(verdict=R_EXT_INVALID_TOO_FEW_PAIRED,
                    evidence=f"{treat}={meta.get(treat, {}).get('status')}; "
                             f"{control}={meta.get(control, {}).get('status')}")
    seeds = sorted(included[treat] & included[control])
    if len(seeds) < minimum:
        return dict(verdict=R_EXT_INVALID_TOO_FEW_PAIRED,
                    evidence=f"common complete seeds={seeds}; n={len(seeds)} "
                             f"< {minimum}")
    table = extinction_table(cfg, frame)
    table = table[table.seed.isin(seeds)].set_index(["arm", "seed"])
    tasks = {arm: np.array([float(table.loc[(arm, s), "extinction_task"])
                            for s in seeds]) for arm in (treat, control)}
    counts = {arm: int(sum(bool(table.loc[(arm, s), "extinct"]) for s in seeds))
              for arm in (treat, control)}
    censored = {arm: len(seeds) - counts[arm] for arm in (treat, control)}
    rng = np.random.default_rng(int(R["bootstrap_seed"]))
    draws = rng.integers(0, len(seeds),
                         size=(int(_P(cfg)["bootstrap_B"]), len(seeds)))
    delta = tasks[treat] - tasks[control]
    ci = paired_ci(_compat_cfg(cfg), delta, draws)
    heavy = [arm for arm in (treat, control)
             if censored[arm] >= int(R["censor_flag_min_seeds"])]
    verdict = classify_r_ext_mplus(ci)

    def level(arm: str, block: int, column: str) -> float:
        group = levels[(levels.arm == arm) & (levels.block == block)
                       & levels.seed.isin(seeds)]
        return float(group[column].mean())

    detail = {arm: dict(
        extinct=counts[arm], censored=censored[arm],
        clopper_pearson=[float(v) for v in clopper_pearson(counts[arm], len(seeds))],
        median_extinction_task=float(np.median(tasks[arm])),
        E_level_B10=level(arm, b10, "mean_log10_unfit"),
        E_drift=level(arm, b10, "mean_log10_unfit") - level(arm, b02,
                                                            "mean_log10_unfit"),
        b_median_all_B10=level(arm, b10, "L1_b_median_all"))
        for arm in (treat, control)}
    return dict(
        verdict=(f"{verdict} (+{ONSET_CENSORED})" if heavy else verdict),
        ci_basis="paired percentile", seeds=seeds, n=len(seeds),
        delta_extinction_task={key: float(ci[key])
                               for key in ("point", "ci_lo", "ci_hi")},
        ci_degenerate=int(ci["ci_degenerate"]),
        per_seed_delta=[float(v) for v in delta], arms=detail,
        censored_flag=heavy,
        null_is_not_tight=True,
        evidence=(f"n={len(seeds)}; {treat} - {control} on the extinction task = "
                  f"{_fmt_ci(ci)}; median task {treat}="
                  f"{detail[treat]['median_extinction_task']:.1f} / {control}="
                  f"{detail[control]['median_extinction_task']:.1f}; "
                  f"extinct {treat}={counts[treat]}/{len(seeds)} / "
                  f"{control}={counts[control]}/{len(seeds)} (REPORT); "
                  f"E-level B10 {treat}={detail[treat]['E_level_B10']:.4f} / "
                  f"{control}={detail[control]['E_level_B10']:.4f}; "
                  f"b(all units) B10 {treat}={detail[treat]['b_median_all_B10']:+.4f}"
                  f" / {control}={detail[control]['b_median_all_B10']:+.4f}; "
                  f"censored={censored}"
                  + ("; 等価限界を登録していないので『効果なし』は主張しない"
                     if verdict == BWD_EFFECT_NOT_DISTINGUISHED_UNDER_MU else "")))


def analyze(cfg: dict, outdir: Path, meta: dict[str, dict], *,
            i_cells_invalid: bool = False) -> dict:
    P, iso = _P(cfg), _P(cfg)["seed_isolation"]
    cells = active_cells(cfg, i_cells_invalid=i_cells_invalid)
    arms_present = [arm for arm in ARM_ORDER if arm in meta]
    frame = pd.read_csv(outdir / "task_end_metrics.csv")
    levels = block_levels(cfg, frame)
    levels.to_csv(outdir / "block_levels.csv", index=False)
    onsets = onset_table(cfg, frame)
    onsets.to_csv(outdir / "onset.csv", index=False)
    b02 = int(P["early_block_tasks"][1]) // int(P["block_tasks"])
    b10 = int(P["late_block_tasks"][1]) // int(P["block_tasks"])
    valid = {arm: meta[arm]["status"] in VALID_ARM_STATUSES for arm in arms_present}
    included = {arm: set(int(s) for s in meta[arm]["included_seeds"]) if valid[arm]
                else set() for arm in arms_present}
    paired = sorted(set.intersection(*(included[arm] for arm in cells))
                    if all(valid.get(arm) for arm in cells) else set())
    rows: list[dict] = []

    def add(pred: str, scope: str, verdict: str, evidence: str,
            ci_basis: str = "", ci_degenerate: object = "") -> None:
        rows.append(dict(pred=pred, scope=scope, verdict=verdict,
                         evidence=evidence, ci_basis=ci_basis,
                         ci_degenerate=ci_degenerate))

    def series(arm: str, block: int, column: str) -> np.ndarray:
        group = levels[(levels.arm == arm) & (levels.block == block)].set_index("seed")
        missing = [s for s in paired if s not in group.index]
        if missing:
            raise RuntimeError(f"{arm} block {block}: missing seeds {missing}")
        return group.loc[paired, column].to_numpy(dtype=np.float64)

    def tau_series(arm: str, threshold: float) -> np.ndarray:
        group = onsets[(onsets.arm == arm) & (onsets.threshold == threshold)]
        group = group.set_index("seed")
        return group.loc[paired, "tau"].to_numpy(dtype=np.float64)

    details: dict = {}
    if not all(valid.get(arm) for arm in cells):
        main = ARM_INVALID_EXCLUSION_LIMIT
        add("P-main", "2x2", main,
            "; ".join(f"{arm}={meta[arm]['status']}" for arm in arms_present))
    elif len(paired) < int(iso["min_paired_seeds"]):
        main = CONTRAST_INVALID_TOO_FEW_PAIRED
        add("P-main", "2x2", main,
            f"common complete seeds={paired}; n={len(paired)} "
            f"< {iso['min_paired_seeds']}")
    else:
        draws = _draws(cfg, len(paired))
        margin = float(P["equivalence_margin"])
        interaction_margin = float(P["interaction_margin"])
        level02 = {arm: series(arm, b02, "mean_log10_unfit") for arm in cells}
        level10 = {arm: series(arm, b10, "mean_log10_unfit") for arm in cells}
        drift_values = {arm: level10[arm] - level02[arm] for arm in cells}
        if i_cells_invalid:
            drift_result = _single_contrast_verdict(
                cfg, drift_values, draws, margin=margin)
            level_result = _single_contrast_verdict(
                cfg, level10, draws, margin=margin)
        else:
            drift_result = _endpoint_verdict(cfg, drift_values, draws, margin=margin,
                                             interaction_margin=interaction_margin)
            level_result = _endpoint_verdict(cfg, level10, draws, margin=margin,
                                             interaction_margin=interaction_margin)

        floor_values = {(arm, block): float(series(arm, block, "floor_frac").max())
                        for arm in cells for block in (b02, b10)}
        for arm in ANCHOR_ARMS:
            if valid.get(arm):
                for block in (b02, b10):
                    group = levels[(levels.arm == arm) & (levels.block == block)]
                    floor_values[(arm, block)] = float(group.floor_frac.max())
        floor_pass = all(value == 0.0 for value in floor_values.values())
        b02_means = {arm: float(level02[arm].mean()) for arm in cells}
        b02_range = max(b02_means.values()) - min(b02_means.values())
        ceiling_flag = b02_range > float(P["ceiling_flag_dex"])
        ladder = drift_result["verdict"] != level_result["verdict"]

        if i_cells_invalid:
            # spec §7 の S-op 行の失敗時帰結。主判定は出さない。
            main = I_CELLS_INVALID_S_OP
            add("P-main", "2x2 main verdict", main,
                "S-op FAILED → I+ セル（IM・Im）を無効と宣言し、要因計画の主判定は"
                "出さない（spec §7 の S-op 行）。以下は I− 側の登録済み対比と R-ext "
                f"のみ。共通完走 seed={paired}; n={len(paired)}")
        else:
            main = (E_DRIFT_INVALID_FLOOR if not floor_pass else
                    LADDER_INVERTS if ladder else drift_result["verdict"])
            add("P-main", "E-drift (primary) + E-level", main,
                f"common complete seeds={paired}; n={len(paired)}; "
                f"E-drift={drift_result['verdict']}; E-level={level_result['verdict']}; "
                f"S-floor={'PASS' if floor_pass else 'FAIL'}; "
                f"b-WD lambda={float(cfg['bias_weight_decay']['lam']):g} on all four cells",
                "paired percentile")
        for label, result in (("E-drift", drift_result), ("E-level", level_result)):
            for name, ci in result["cis"].items():
                a, b = FACTOR_CONTRASTS[name]
                add(label, f"{name}: {a} - {b} [dex]", result["bands"][name],
                    _fmt_ci(ci), "paired percentile", int(ci["ci_degenerate"]))
            if not i_cells_invalid:
                add(label, "interaction (IM-iM)-(Im-im)", result["interaction_band"],
                    _fmt_ci(result["interaction"]), "paired percentile",
                    int(result["interaction"]["ci_degenerate"]))
            add(label, "decision tree" if not i_cells_invalid
                else "M main effect band (I+ cells invalid)", result["verdict"],
                f"bands={result['bands']}; interaction={result['interaction_band']}; "
                f"margin={result['margin']} dex; "
                f"interaction_margin={result['interaction_margin']} dex")
        add("S-floor", "B02/B10 floor_frac, all running arms",
            "PASS" if floor_pass else "FAIL",
            "; ".join(f"{arm}/B{block:02d}={value:.6g}"
                      for (arm, block), value in sorted(floor_values.items())))
        add("S-ceiling", f"B02 level range over {len(cells)} cell(s)",
            CEILING_CONTAMINATED if ceiling_flag else "PASS",
            f"range={b02_range:.6f} dex; threshold={float(P['ceiling_flag_dex']):.1f}; "
            f"levels={b02_means}")
        add("L", "E-drift vs E-level ladder",
            LADDER_INVERTS if ladder else "CONSISTENT",
            f"E-drift={drift_result['verdict']}; E-level={level_result['verdict']}")

        r_ext = _r_ext(cfg, frame, onsets, levels, meta, valid, included, b10)
        add("R-ext", "extinction by 5M: im vs im_nowd", r_ext["verdict"],
            r_ext["evidence"], r_ext.get("ci_basis", ""))
        r_ext_mplus = None
        if MPLUS_NOWD_ARM in meta:
            r_ext_mplus = _r_ext_mplus(cfg, frame, levels, meta, valid, included,
                                       b02, b10)
            add("R-ext-M+", "extinction task: iM vs iM_nowd",
                r_ext_mplus["verdict"], r_ext_mplus["evidence"],
                r_ext_mplus.get("ci_basis", ""))

        # REPORT_ONLY（spec §6.1 の E-onset 降格 / §6.5）
        primary_threshold = float(P["onset"]["threshold"])
        never_below = {arm: int(onsets[(onsets.arm == arm)
                                       & (onsets.threshold == primary_threshold)
                                       ].never_below.sum()) for arm in arms_present}
        ceiling_arm = str(P["ceiling_count_arm"])
        ceiling_seeds = 0
        if valid[ceiling_arm]:
            group = levels[(levels.arm == ceiling_arm) & (levels.block == b10)]
            ceiling_seeds = int((group.mean_log10_unfit
                                 >= float(P["ceiling_level_dex"])).sum())
        add("S-ceiling", f"{ceiling_arm} seeds at the ceiling "
            f"(B10 >= {P['ceiling_level_dex']} dex)", "REPORT_ONLY",
            f"{ceiling_seeds} seeds")
        if any(never_below.values()):
            add("C", "never below threshold", ONSET_NEVER_BELOW,
                f"counts={never_below}（一度も閾値を下回らない seed。E-onset の"
                f"『上抜け』が定義できない。E-onset は REPORT_ONLY なので主判定には効かない）")
        for arm in arms_present:
            if not valid[arm]:
                continue
            seeds = paired if arm in cells else sorted(included[arm])
            group = levels[(levels.arm == arm) & (levels.block == b10)
                           & levels.seed.isin(seeds)]
            early = levels[(levels.arm == arm) & (levels.block == b02)
                           & levels.seed.isin(seeds)]
            tau = onsets[(onsets.arm == arm) & onsets.primary
                         & onsets.seed.isin(seeds)]
            add("D", f"{arm} L1 strict_dead_frac B02->B10", "REPORT_ONLY",
                f"{early.L1_strict_dead_frac.mean():.6f}->"
                f"{group.L1_strict_dead_frac.mean():.6f}")
            add("R", f"{arm} L1 eff_rank B10", "REPORT_ONLY",
                f"{group.L1_eff_rank.mean():.6f}")
            add("U", f"{arm} u_norm / bypass_share B10", "REPORT_ONLY",
                f"u_norm={group.u_norm.mean():.6g}; "
                f"bypass_share={group.bypass_share.mean():.6g}; "
                f"|bypass|={group.bypass_value.abs().mean():.6g}")
            add("ledger", f"{arm} B / M / b(all units) B02->B10", "REPORT_ONLY",
                f"B={early.L1_B_median_alive.mean():+.6f}->"
                f"{group.L1_B_median_alive.mean():+.6f}; "
                f"M={early.L1_M_median_alive.mean():+.6f}->"
                f"{group.L1_M_median_alive.mean():+.6f}; "
                f"b_all={early.L1_b_median_all.mean():+.6f}->"
                f"{group.L1_b_median_all.mean():+.6f}")
            add("A", f"{arm} E-level B10 / E-drift / E-onset", "REPORT_ONLY",
                f"E-level={group.mean_log10_unfit.mean():.6f}; "
                f"E-drift={float((group.mean_log10_unfit.mean() - early.mean_log10_unfit.mean())):+.6f}; "
                f"median tau={float(tau.tau.median()):.1f}; "
                f"censored={int(tau.censored.sum())}")
        details = dict(
            drift=_plain(drift_result), level=_plain(level_result),
            r_ext=r_ext, r_ext_mplus=r_ext_mplus, never_below=never_below,
            b02_means=b02_means, b02_range=b02_range,
            ceiling_seeds_on_anchor={ceiling_arm: ceiling_seeds},
            floor_values={f"{arm}_B{block:02d}": value
                          for (arm, block), value in floor_values.items()},
            floor_pass=bool(floor_pass), ladder_inverts=bool(ladder),
            ceiling_contaminated=bool(ceiling_flag))

    for arm in arms_present:
        add("exclusion", arm, "ARM_VALID" if valid[arm] else meta[arm]["status"],
            f"status={meta[arm]['status']}; excluded={meta[arm]['excluded_seeds']}; "
            f"included={meta[arm]['included_seeds']}")
    verdict = pd.DataFrame(rows)
    verdict.to_csv(outdir / "verdict.csv", index=False)

    extinction = extinction_table(cfg, frame)
    extinction.to_csv(outdir / "extinction.csv", index=False)

    endpoints = pd.DataFrame({"seed": paired})
    if paired:
        extinct = extinction.set_index(["arm", "seed"])
        for arm in arms_present:
            if not valid[arm] or not set(paired) <= included[arm]:
                continue
            endpoints[f"{arm}_tau"] = tau_series(
                arm, float(P["onset"]["threshold"]))
            for block, tag in ((b02, "B02"), (b10, "B10")):
                endpoints[f"{arm}_{tag}_meanlog10unfit"] = series(
                    arm, block, "mean_log10_unfit")
            endpoints[f"{arm}_drift"] = (
                endpoints[f"{arm}_B10_meanlog10unfit"]
                - endpoints[f"{arm}_B02_meanlog10unfit"])
            endpoints[f"{arm}_B10_u_norm"] = series(arm, b10, "u_norm")
            endpoints[f"{arm}_B10_bypass_share"] = series(arm, b10, "bypass_share")
            endpoints[f"{arm}_extinct"] = [
                int(extinct.loc[(arm, seed), "extinct"]) for seed in paired]
    endpoints.to_csv(outdir / "paired_endpoints.csv", index=False)

    exclusion_rows = []
    for arm in arms_present:
        events = {int(e["seed"]): e for e in meta[arm]["exclusion_events"]}
        for seed in cfg["common"]["seeds"]:
            event = events.get(int(seed))
            exclusion_rows.append(dict(
                arm=arm, seed=int(seed), excluded=int(event is not None),
                detected_step="" if event is None else event["detected_step"],
                detected_task="" if event is None else event["detected_task"],
                nonfinite_tensors="" if event is None
                else ";".join(event["nonfinite_tensors"]),
                arm_status=meta[arm]["status"]))
    pd.DataFrame(exclusion_rows).to_csv(outdir / "exclusions.csv", index=False)

    result = dict(main_verdict=main, common_complete_seeds=paired,
                  n_paired=len(paired), details=details, arms=arms_present,
                  cells=cells, i_cells_invalid=bool(i_cells_invalid),
                  blocks=dict(B02=b02, B10=b10))
    _figure(frame, outdir)
    _summary(cfg, outdir, verdict, levels, onsets, result)
    return result


def _plain(result: dict) -> dict:
    """JSON に落とせる形へ（CI dict から必要な数値だけ抜く）。"""
    keep = ("point", "ci_lo", "ci_hi", "ci_degenerate")

    def flat(ci) -> dict | None:
        if ci is None:
            return None
        return {k: bool(ci[k]) if k == "ci_degenerate" else float(ci[k])
                for k in keep}

    return dict(verdict=result["verdict"], bands=result["bands"],
                interaction_band=result["interaction_band"],
                margin=result["margin"],
                interaction_margin=result["interaction_margin"],
                cis={name: flat(ci) for name, ci in result["cis"].items()},
                interaction=flat(result["interaction"]))


def _figure(frame: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    colors = {"IM": "#e34a33", "iM": "#fdae61", "Im": "#2b8cbe",
              "im": "#31a354", NOWD_ARM: "#984ea3", MPLUS_NOWD_ARM: "#a65628",
              ANCHOR_ARM: "#555555"}
    panels = [("unfit", "exact-support unfit", True),
              ("L1_strict_dead_frac", "strict_dead_frac L1", False),
              ("L1_submerged_frac", "submerged_frac L1", False),
              ("L1_B_median_alive", "alive median B = b/sigma", False),
              ("mu_norm_visible", "||mu|| of the visible input", False),
              ("bypass_share", "bypass power share", False)]
    present = [arm for arm in ARM_ORDER if arm in set(frame.arm)]
    for (metric, label, logy), axis in zip(panels, axes.flat):
        if metric not in frame.columns:
            continue
        for arm in present:
            group = frame[frame.arm == arm].groupby("task")[metric].median()
            if group.empty:
                continue
            axis.plot(group.index, group.values, color=colors[arm], lw=1.1,
                      label=arm)
        axis.set_xlabel("task")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
        if logy:
            axis.set_yscale("log")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("ident_mu_2x2_0901 — µ dial x task-identifiability dial "
                 "(all four cells under b-WD)")
    fig.tight_layout()
    fig.savefig(outdir / "fig_ident_mu_2x2.png", dpi=150)
    plt.close(fig)


def _summary(cfg: dict, outdir: Path, verdict: pd.DataFrame, levels: pd.DataFrame,
             onsets: pd.DataFrame, result: dict) -> None:
    P, paired = _P(cfg), result["common_complete_seeds"]
    b02, b10 = result["blocks"]["B02"], result["blocks"]["B10"]
    details = result.get("details", {})
    table_rows = []
    for arm in result.get("arms", ARM_ORDER):
        tau = onsets[(onsets.arm == arm) & onsets.primary]
        for block, tag in ((b02, "B02"), (b10, "B10")):
            group = levels[(levels.arm == arm) & (levels.block == block)]
            if group.empty:
                continue
            table_rows.append(dict(
                arm=arm, wd_b=float(_arm_cfg(cfg, arm)["wd_b"]), window=tag,
                mean_log10_unfit=float(group.mean_log10_unfit.mean()),
                L1_dead=float(group.L1_strict_dead_frac.mean()),
                L1_eff_rank=float(group.L1_eff_rank.mean()),
                b_median_all=float(group.L1_b_median_all.mean()),
                u_norm=float(group.u_norm.mean()) if "u_norm" in group else np.nan,
                bypass_share=float(group.bypass_share.mean())
                if "bypass_share" in group else np.nan,
                median_tau=float(tau.tau.median()) if len(tau) else np.nan,
                n_censored=int(tau.censored.sum()) if len(tau) else 0,
                floor_frac=float(group.floor_frac.max())))
    r_ext = details.get("r_ext", {})
    warning = []
    if details.get("ceiling_contaminated"):
        warning.append("`CEILING_CONTAMINATED`: 要因 4 セルの B02 水準差が 3 dex を"
                       "超えたため **E-drift 単独では読まない**。")
    if details.get("ladder_inverts"):
        warning.append("`LADDER_INVERTS`: E-drift と E-level の判定が食い違うため、"
                       "どちらも単独では引かない。")
    if details.get("floor_pass") is False:
        warning.append("`S-floor FAIL`: E-drift は無効で、E-level のみ報告する。")
    cells = result.get("cells", list(CELL_ORDER))
    invalid = bool(result.get("i_cells_invalid"))
    lines = [
        "# ident_mu_2x2_0901 — 可識別性 × µ の 2×2（純化版）", "",
        f"事前登録: `{cfg['spec']}`"
        + ("＋`specs/spec_ident_mu_2x2_0901_addendum1.md`（追補1）" if invalid else "")
        + f"。主判定は **{result['main_verdict']}**。"
        + (f"S-op が FAIL したため I+ セル（IM・Im）は無効で、走ったのは "
           f"{', '.join(f'`{a}`' for a in result.get('arms', []))} の "
           f"{len(result.get('arms', []))} 腕。" if invalid else "")
        + f"対比に使った共通完走 seed（{' / '.join(cells)}）= {paired} "
          f"(n={result['n_paired']})。", "",
        f"要因 4 セルはすべて b 限定 WD λ="
        f"{float(cfg['bias_weight_decay']['lam']):g} の下にある（spec D8・§3.5）。"
        f"**主判定の読みは「b 拘束下」限定であり、拘束なしの世界の I/M 主効果は"
        f"測っていない**（橋は `im_nowd` と committed `Aexact` まで）。", "",
        f"窓は B02 = task {P['early_block_tasks'][0]}–{P['early_block_tasks'][1]}、"
        f"B10 = task {P['late_block_tasks'][0]}–{P['late_block_tasks'][1]}。"
        f"共主 endpoint は E-drift（`mean(log10 unfit)` の B10−B02）と "
        + ("E-level（同 B10）。**要因計画の主判定は出さない**（追補1 §3）。"
           if invalid else "E-level（同 B10）で、主判定は E-drift。")
        + f"等価限界は "
        f"Δ={float(P['equivalence_margin'])} dex・"
        f"Δ_int={float(P['interaction_margin'])} dex、床は {P['unfit_floor']:.0e}。", "",
        "## 判定", "", markdown_table(verdict), "",
        "## 水準・死・バイパス", "", markdown_table(pd.DataFrame(table_rows)), "",
        "## 登録副判定 R-ext（死の主張）", "",
        f"- 判定: **{r_ext.get('verdict', 'not computed')}**",
        f"- {r_ext.get('evidence', 'not computed')}",
        "- **`LOP_CURED` と読み替えない。** dead と機能が逆向きに動いた実例が 2 件ある。",
        "", "## フラグ", "",
    ]
    lines.extend(f"- {item}" for item in warning)
    if not warning:
        lines.append("- S-floor / S-ceiling / ladder の追加フラグなし。")
    lines.extend([
        "", "## 引いてはいけない線（spec §9）", "",
        "- **主判定の読みは「b 拘束下（λ=1e-3）」限定。** λ は移送値であり、"
        "結果を見て選び直さない。",
        "- `I+` は「可識別性あり」ではなく**ゲートに触れない経路での可識別性あり**である。"
        "A2 のバイパスが運ぶのは出力側の大域的な加法成分 1 自由度であり、"
        "std がタスク内で使っている per-unit の実効バイアスの分け前ではない。",
        "- per-unit の可識別性は前活性のタスク依存 DC そのものなので µ と分離できない"
        "（補題）。本走はそこを閉じない。",
        "- 要因 4 セルはいずれも「境界ごとの µ の引き直し」を持たない。本走の µ は"
        "**静的な壁**の効果である。",
        "- `im_nowd` が全滅しても発見として引かない（committed `Aexact` の再現）。"
        "`S-mu` の PASS も構成上ほぼ恒真。",
        "- `std_anchor` は 2 因子が縮退した外部アンカーであって要因計画のセルではない。",
        "- `strict_dead` は主判定に使っていない（R-ext の全滅到達だけが明示的な例外で、"
        "死についてのみ語る）。", "",
        "## 集計上の注意", "",
        "- `unfit` は 32 パターン上の**分散比**で DC を見ないので、task 内で定数の"
        "バイパスは `unfit` を動かさない。**I 主効果が E-drift に出たら、それは"
        "表現力のアーティファクトではありえず力学経由である**（spec §6.1）。"
        "DC を見る量は `eval_loss_exact`（バイパス込みで記録）である。",
        "- 同じ理由で `bypass_share` は分散の分け前ではなく**出力パワーの分け前**。",
        "- E-onset（REPORT_ONLY）は「一度閾値を下回ったあとの最初の上抜け」と定義した"
        "（素直に読むと初期過渡で全 seed が tau=1 になるため）。**spec の字義から"
        "離れて残っている唯一の点。**",
        "- R-ext の全滅は `strict_dead_frac == 1.0` で判定し、`alive == 0` の別定義との"
        "食い違い件数を併記する（spec §6.4 の裁定どおり）。",
    ])
    (outdir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


OUTPUTS = ("verdict.csv", "summary.md", "paired_endpoints.csv", "onset.csv",
           "extinction.csv", "exclusions.csv", "task_end_metrics.csv",
           "block_levels.csv", "boundary_snapshots.csv", "run_sanity.json",
           "config_used.yaml", "fig_ident_mu_2x2.png")


def _shard(outdir: Path) -> Path:
    path = Path(outdir) / "shards"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_boundary_snapshots(outdir: Path, frame: pd.DataFrame,
                              post_frames: list[pd.DataFrame], total: int,
                              period: int) -> None:
    columns = ["arm", "seed", "step", "L1_strict_dead_frac", "L1_submerged_frac",
               "L1_M_median_alive", "L1_B_median_alive", "L1_sigma_median_alive",
               "mu_norm_visible", "bypass_value"]
    pre = frame[(frame.step > 0) & (frame.step < total)
                & (frame.step % period == 0)].copy()
    pre = pre[[c for c in columns if c in pre.columns]]
    pre["task_boundary"] = (pre.step // period).astype(int)
    pre["side"] = "pre"
    posts = pd.concat(post_frames, ignore_index=True) if post_frames else pd.DataFrame()
    pd.concat([pre, posts], ignore_index=True).sort_values(
        ["arm", "seed", "step", "side"]).to_csv(
            outdir / "boundary_snapshots.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--selftest", action="store_true",
                        help="測定を含まない実装同一性ゲートだけを回す")
    parser.add_argument("--gates", action="store_true",
                        help="S-op まで含む本走前ゲート（事前予測が要る）")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--arm")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--outdir")
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = load_config(str(cfg_path))
    stage = ("selftest" if args.selftest else "gates" if args.gates
             else "smoke" if args.smoke
             else "analyze" if args.analyze_only else "full")
    validate_config(cfg, stage=stage)
    require_omp(int(_P(cfg)["omp_num_threads"]))
    outdir = Path(args.outdir).resolve() if args.outdir else outdir_of(cfg)
    gate_dir = gatedir_of(cfg)
    started = time.time()

    if args.selftest:
        run_selftests(cfg, gate_dir)
        return
    if args.gates:
        run_gates(cfg, gate_dir)
        return
    if args.smoke:
        smoke_dir = Path(ROOT) / "results" / "_smoke_ident_mu_2x2_0901"
        for arm in ARM_ORDER:
            smoke = copy.deepcopy(cfg)
            smoke["common"]["checkpoints"] = []
            result = run_arm_ident(smoke, arm, smoke_dir, total_steps=30_000,
                                   task_period=10_000, guard_every=1_000,
                                   keep_unit_arrays=False, write_logs=False,
                                   record_boundaries=True)
            if result["status"] != COMPLETE or not result["sanity"]["pass_"]:
                raise RuntimeError(f"smoke failed: {arm}: {result['status']} "
                                   f"{result['sanity']}")
        print("SMOKE PASS", flush=True)
        return

    outdir.mkdir(parents=True, exist_ok=True)
    gates, i_cells_invalid = _gates_for_run(gate_dir)
    run_arms = [arm for arm in ARM_ORDER
                if arm not in CELL_ORDER
                or arm in active_cells(cfg, i_cells_invalid=i_cells_invalid)]
    if i_cells_invalid:
        print("S-op FAILED -> I+ cells (IM, Im) are invalid; running "
              f"{run_arms} and reporting no 2x2 main verdict", flush=True)
    total = int(cfg["common"]["total_steps"])
    period = int(cfg["phase1"]["task_period"])
    guard = int(_P(cfg)["guard_every"])
    if args.arm and args.arm not in run_arms:
        raise SystemExit(f"unknown or invalidated arm {args.arm}")
    todo = [args.arm] if args.arm else list(run_arms)
    if not args.analyze_only:
        for arm in todo:
            result = run_arm_ident(cfg, arm, outdir, total_steps=total,
                                   task_period=period, guard_every=guard)
            result["frame"].to_csv(_shard(outdir) / f"{arm}.csv", index=False)
            result["boundary_frame"].to_csv(
                _shard(outdir) / f"{arm}_boundary_post.csv", index=False)
            (_shard(outdir) / f"{arm}.json").write_text(json.dumps(
                {k: v for k, v in result.items()
                 if k not in {"frame", "boundary_frame"}},
                indent=2, ensure_ascii=False), encoding="utf-8")
        if args.arm:
            return

    frames, post_frames, meta = [], [], {}
    for arm in run_arms:
        meta[arm] = json.loads((_shard(outdir) / f"{arm}.json").read_text(
            encoding="utf-8"))
        if meta[arm]["status"] in VALID_ARM_STATUSES:
            frames.append(pd.read_csv(_shard(outdir) / f"{arm}.csv"))
            post = _shard(outdir) / f"{arm}_boundary_post.csv"
            if post.exists() and post.stat().st_size:
                post_frames.append(pd.read_csv(post))
    if not frames:
        raise RuntimeError("no valid arms to analyze")
    frame = pd.concat(frames, ignore_index=True)
    frame.to_csv(outdir / "task_end_metrics.csv", index=False)
    _write_boundary_snapshots(outdir, frame, post_frames, total, period)

    result = analyze(cfg, outdir, meta, i_cells_invalid=i_cells_invalid)
    run_sanity = dict(
        gates={name: report["pass_"] for name, report in gates.items()},
        i_cells_invalid=bool(i_cells_invalid), arms_run=run_arms,
        S3={arm: dict(pass_=meta[arm]["sanity"]["pass_"],
                      max_relerr=meta[arm]["sanity"]["max_relerr"],
                      n_visible_violations=meta[arm]["sanity"]["n_visible_violations"],
                      quantization_violations=meta[arm]["sanity"]["n_quantization_violations"],
                      wall_violations=meta[arm]["sanity"]["n_wall_identity_violations"])
            for arm in run_arms},
        seed_isolation={arm: dict(status=meta[arm]["status"],
                                  excluded_seeds=meta[arm]["excluded_seeds"],
                                  events=meta[arm]["exclusion_events"])
                        for arm in run_arms},
        details=result["details"],
        training_elapsed_sec={arm: meta[arm]["elapsed_sec"] for arm in run_arms})
    (outdir / "run_sanity.json").write_text(
        json.dumps(run_sanity, indent=2, ensure_ascii=False), encoding="utf-8")
    with (outdir / "config_used.yaml").open("w") as stream:
        yaml.safe_dump(cfg, stream, allow_unicode=True, sort_keys=False)
    (outdir / "provenance.json").write_text(json.dumps(provenance(
        "ident_mu_2x2_0901", cfg_path, cfg, outdir,
        dict(analysis=result, run_sanity=run_sanity), started, sys.argv, OUTPUTS),
        indent=2, ensure_ascii=False), encoding="utf-8")
    print(pd.read_csv(outdir / "verdict.csv")[["pred", "scope", "verdict"]]
          .to_string(index=False), flush=True)
    print(f"ALL DONE -> {outdir}", flush=True)


if __name__ == "__main__":
    main()
