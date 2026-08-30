"""elu_swamp_0830: ELU の沼 — 第2層 µ の再生と、沈下が駆動か拡散か。

``mlp2_phase1_0829`` の ``L2_A1``（第1層だけ走行平均で中心化した隠れ2層×100）は、
第2層の壁が立ったまま dose 7.18 で平坦だった。本モジュールはその第1層の活性化を
ELU に替え、2 点を測る:

  Q1  第2層の入力 dose は抑えられる（Clevert）のか再生される（沈下レバー）のか
  Q2  沈下は µ 方向の駆動か、無方向の拡散 + 可動度勾配（phi' = alpha e^z）か

段階は事前登録ゲートが高コストな本走に先立つよう明示的に分けてある::

    OMP_NUM_THREADS=1 python -m src.elu_swamp --preflight   # S-grad/S-elu-limit/S-copy/S-pair/S-taut/S-submerge/S5/S7
    OMP_NUM_THREADS=1 python -m src.elu_swamp --s0prime     # R_none/R_A1 == phase1 L2_none/L2_A1
    OMP_NUM_THREADS=1 python -m src.elu_swamp

``--s0prime`` は本物の 5M を走らせて結果ディレクトリに置くので、本走はその 2 腕を
再計算せず再開する。

本モジュールは ``mlp2_phase1`` の編集ではなく意図的な fork である（凍結した実験
モジュールは凍結したままにする、というリポジトリの慣行）。ReLU 経路が phase1 と
bit 一致することは S-copy と S0' が守る。
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from .common import ROOT, load_config, pick_device
from .mlp2_phase0 import (LOG_UNIT_KEYS, _effective_rank, _max_relative,
                          _sha_array, _sha_file, _seed_state_hashes,
                          identity_sanity_pass, require_omp, spearman,
                          write_csv)
from .mlp2_phase0b import _ci_components, _complete_arm_logs, _window_indices
from .mlp2_phase1 import (NUMERIC_DIVERGENCE, P1_LOG_LAYER_KEYS,
                          NumericDivergenceError, PhaseRecorderP1, StreamDigest,
                          _alignment_metrics, _arm, _centered_flags, _decide,
                          _env_hashes, _init_hashes, _nanmax,
                          _numeric_divergence_event, _seed_state_hashes_p1,
                          exact_layer_record_p1, setup_arm_p1)
from .ratchet_log import full_support_ro, teacher_f64


EXPERIMENT = "elu_swamp_0830"
ARM_ORDER = ("R_none", "R_A1", "E_none", "E_A1", "E_Aall")
SMOKE_STEPS = 30_000

# (hidden, centered_layers, activation) — 事前登録。validate_config が逐語照合する。
REGISTERED_ARMS = {
    "R_none": ([100, 100], [], "relu"),
    "R_A1": ([100, 100], [1], "relu"),
    "E_none": ([100, 100], [], "elu"),
    "E_A1": ([100, 100], [1], "elu"),
    "E_Aall": ([100, 100], [1, 2], "elu"),
}
RELU_ARMS = tuple(a for a, v in REGISTERED_ARMS.items() if v[2] == "relu")
ELU_ARMS = tuple(a for a, v in REGISTERED_ARMS.items() if v[2] == "elu")

# 新規の記録列。S0' は「既存列の完全一致」なのでこれらは比較から外す（phase1 が
# alignment 列に対して行ったのと同じ扱い）。
ELU_EXTRA_UNIT_KEYS = ("zbar", "dzbar")
ELU_EXTRA_LAYER_KEYS = ("preact_sd_median", "submerged")
ELU_LOG_UNIT_KEYS = LOG_UNIT_KEYS + ELU_EXTRA_UNIT_KEYS
ELU_LOG_LAYER_KEYS = P1_LOG_LAYER_KEYS + ELU_EXTRA_LAYER_KEYS
ELU_EXTRA_RUN_KEYS = ("v_readout", "v_readout_step", "activation", "act_alpha")
S0PRIME_META_KEYS = {"run_id", "arm"}

# 事前登録の対比。全て paired（活性化も中心化も乱数を消費しない）。
CONTRASTS = (
    ("E_A1", "R_A1", "elu_vs_relu_A1"),        # G0 / Q1 の主対比
    ("E_none", "R_none", "elu_vs_relu_none"),
    ("E_Aall", "E_A1", "A2_given_A1_elu"),     # P2c の介入対比
    ("R_A1", "R_none", "relu_A1_vs_none"),     # phase1 の再現（REPORT_ONLY）
)

SQRT2 = math.sqrt(2.0)
SQRT2PI = math.sqrt(2.0 * math.pi)


# --------------------------------------------------------------------------
# ELU の閉形式 g(s)（spec §2.1 / §5.1c）
# --------------------------------------------------------------------------
def g_elu(s, alpha: float = 1.0):
    """``E[phi] / sd(phi)`` for ``phi = ELU_alpha`` and ``z ~ N(0, s^2)``.

    spec §2.1 の式そのもの（alpha=1 で逐語一致）を、溢れない形に書き換えて評価する::

        exp(s^2/2)*Phi(-s)  = 0.5 * erfcx(s/sqrt2)
        exp(2*s^2)*Phi(-2s) = 0.5 * erfcx(sqrt2*s)

    ``exp(2 s^2)`` は s が 20 を越えると単独では float64 を溢れるが、``erfcx``
    は溢れない。s=0 では sd(phi)=0 になるので NaN を返す。
    """
    s = torch.as_tensor(s, dtype=torch.float64)
    e1 = s / SQRT2PI + alpha * (0.5 * torch.special.erfcx(s / SQRT2) - 0.5)
    e2 = (s * s / 2.0
          + alpha * alpha * (0.5 * torch.special.erfcx(SQRT2 * s)
                             - torch.special.erfcx(s / SQRT2) + 0.5))
    var = (e2 - e1 * e1).clamp_min(0.0)
    out = torch.where(var > 0, e1 / var.clamp_min(1e-300).sqrt(),
                      torch.full_like(e1, float("nan")))
    return out


def dose_reference(preact_sd, width: int, alpha: float = 1.0):
    """``sqrt(width) * g_elu(s)`` — 第1層が iid なら第2層 dose はこの値になる。"""
    return math.sqrt(float(width)) * g_elu(preact_sd, alpha)


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------
def _p1_cfg(cfg: dict) -> dict:
    """凍結済み phase1 ヘルパが読む節名へ写す（``elu_swamp`` -> ``phase1``）。"""
    out = copy.deepcopy(cfg)
    out["phase1"] = copy.deepcopy(cfg["elu_swamp"])
    return out


def _activation(cfg: dict, arm: str) -> tuple[str, float]:
    name = str(_arm(cfg, arm)["activation"])
    spec = cfg["activation"][name]
    return str(spec["name"]), float(spec.get("alpha", 1.0))


def validate_config(cfg: dict, *, stage: str) -> None:
    """登録された設計を照合する。full/analyze は §10-2 の確認も要求する。"""
    if stage not in {"preflight", "smoke", "s0prime", "full", "analyze"}:
        raise ValueError(f"unknown validation stage {stage!r}")
    C, A, I, P = cfg["common"], cfg["condA"], cfg["intervention"], cfg["elu_swamp"]
    names = [a["name"] for a in cfg["arms"]]
    if names != list(ARM_ORDER):
        raise ValueError(f"arms must be {ARM_ORDER}, got {names}")
    for arm in cfg["arms"]:
        hidden, centered, act = REGISTERED_ARMS[arm["name"]]
        if [int(v) for v in arm["hidden"]] != hidden:
            raise ValueError(f"{arm['name']} hidden differs from the preregistration")
        if [int(v) for v in (arm.get("centered_layers") or [])] != centered:
            raise ValueError(f"{arm['name']} centering differs from the preregistration")
        if str(arm["activation"]) != act:
            raise ValueError(f"{arm['name']} activation differs from the preregistration")
    if int(A["m"]) != 20 or int(A["f"]) != 15:
        raise ValueError("elu_swamp requires condA m=20, f=15")
    if list(A["T_values"]) != [10_000] or list(A["encodings"]) != ["std"]:
        raise ValueError("elu_swamp requires T=10000 and std encoding")
    elu = cfg["activation"]["elu"]
    if (str(elu["name"]) != "elu" or float(elu["alpha"]) != 1.0
            or elu["autograd"] is not False or elu["consumes_rng"] is not False):
        raise ValueError("ELU is registered as alpha=1.0, closed-form, rng-free")
    if str(elu["derivative_form"]) != "alpha_exp":
        raise ValueError(
            "the registered ELU derivative is alpha*exp(z); phi(z)+alpha cancels to "
            "exactly zero below z ~ -17.3 in float32 and fails spec §6 (1e-6 at z=-30)")
    if str(cfg["activation"]["relu"]["name"]) != "relu":
        raise ValueError("the relu entry must stay relu")
    if (str(I["name"]) != "A_layer_input_centering"
            or float(I["center_alpha"]) != 0.01
            or I["stop_gradient_on_running_mean"] is not True
            or I["consumes_rng"] is not False):
        raise ValueError("elu_swamp requires the existing center_alpha=0.01")
    if int(P["exact_support"]) != 2 ** (int(A["m"]) - int(A["f"])):
        raise ValueError("elu_swamp.exact_support does not match full support")
    if str(P["ci_method"]) != "studentized_paired":
        raise ValueError("CI must be the paired studentized interval")
    if int(P["bootstrap_B"]) != 10_000 or int(P["bootstrap_seed"]) != 20_260_829:
        raise ValueError("B=10000 and rng seed 20260829 are registered")
    if float(P["unfit_floor"]) != 1e-23 or P["recalibrate_floor"] is not False:
        raise ValueError("the floor is the frozen phase1 value and is not recalibrated")
    frozen = {
        "gate_unfit_threshold": 0.05,
        "g0b_submerged_threshold": 0.30,
        "q1_level_suppressed_below": 2.00,
        "q1_level_regenerated_above": 9.33,
        "q2_scaling_expected_slope": 1.0,
        "q2_drift_ratio_drift_dominated": 0.30,
        "q2_drift_ratio_noise_dominated": 0.10,
        "na_frac_max": 0.20,
        "censor_frac_max": 0.20,
    }
    for key, value in frozen.items():
        if float(P[key]) != value:
            raise ValueError(f"{key} differs from the frozen spec value {value}")
    if int(P["q2_bins"]) != 12 or str(P["q2_bin_method"]) != "equal_count_quantile":
        raise ValueError("Q2 registers 12 equal-count quantile bins (spec §11.4)")
    if str(P["q2_log_base"]) != "natural":
        raise ValueError("the mobility regression is registered in natural log (spec §11.5)")
    if str(P["q2_pool_reduction"]) != "median_over_bins":
        raise ValueError("Q2b pools bins by their median (spec §11.6)")
    if int(P["q2_increment_interval_steps"]) != int(C["lop_every"]):
        raise ValueError("increments are adjacent recording points")
    if (str(P["q2_increment_scope"]) != "within_task_adjacent_records"
            or str(P["q2_condition_on"]) != "interval_start"):
        raise ValueError("increments must be within-task and conditioned on the start")
    if int(P["q2_layer"]) != 2 or int(P["q1_layer"]) != 2:
        raise ValueError("Q1 and Q2 are registered on layer 2")
    if int(P["q2_layer"]) not in [int(v) for v in P["record_zbar_layers"]]:
        raise ValueError("record_zbar_layers must contain the Q2 layer")
    if int(P["q1_lever_source_layer"]) != 1:
        raise ValueError("the P1c lever is read from layer 1 (spec §2.2)")
    if list(P["q2_intervention_contrast"]) != ["E_A1", "E_Aall"]:
        raise ValueError("P2c is the E_A1 vs E_Aall contrast")
    if (P["submerged_frac_in_verdict"] is not False
            or P["strict_dead_in_verdict"] is not False
            or P["lop_signature_table"] is not False):
        raise ValueError("submergence/dead counts and the signature table stay out of verdicts")
    expected_divergence = dict(
        status=NUMERIC_DIVERGENCE,
        detection="nonfinite_training_state_at_probe",
        probe_every=int(C["lop_every"]),
        action="mark_arm_failed_and_continue",
        contrast_policy="any_involved_arm_is_numeric_divergence",
        exclude_partial_logs_from_analysis=True,
        rescue="none",
    )
    if P.get("numeric_divergence") != expected_divergence:
        raise ValueError("numeric-divergence policy differs from spec §5.4")
    if list(P["late_tasks"]) != [451, 500] or list(P["early_tasks"]) != [2, 11]:
        raise ValueError("early 2..11 and late 451..500 are registered")
    if list(P["q1_trend_range_tasks"]) != [1, 500]:
        raise ValueError("the dose trend runs over tasks 1..500 (spec §11.1)")
    if list(P["q2_window_tasks"]) != list(P["late_tasks"]):
        raise ValueError("the primary Q2 window is the late window (spec §11.3)")
    groups = cfg["pairing"]["paired_groups"]
    if groups != [list(ARM_ORDER)]:
        raise ValueError("all five arms are registered as one paired group")
    if str(cfg["pairing"]["baseline_relu"]) != "R_A1":
        raise ValueError("the ReLU baseline is R_A1")
    mapping = cfg["sanity"]["s0_prime_arm_map"]
    if mapping != {"R_none": "L2_none", "R_A1": "L2_A1"}:
        raise ValueError("S0' maps R_none/R_A1 onto phase1 L2_none/L2_A1")
    if stage in {"s0prime", "full", "analyze"}:
        if int(C["total_steps"]) != 5_000_000 or list(C["seeds"]) != list(range(10)):
            raise ValueError("the full run requires 5M steps and seeds 0..9")
        if str(C["device"]) != "cpu":
            raise ValueError("elu_swamp is CPU-only")
    if stage in {"full", "analyze"} and P.get("g0b_threshold_confirmed") is not True:
        raise ValueError(
            "spec §10-2: set elu_swamp.g0b_threshold_confirmed: true only after Issa "
            "has confirmed the G0b submergence threshold 0.30; the full run is blocked "
            "until then")


# --------------------------------------------------------------------------
# 学習経路（ReLU 側は phase1 と逐語同一）
# --------------------------------------------------------------------------
def setup_arm_elu(cfg: dict, arm_cfg: dict, device: str) -> dict:
    """phase1 の腕状態 + 活性化の切り替え。

    ``VecMLPL.__init__`` は活性化を一切参照しないので、構築後に
    ``set_activation`` を呼んでも初期化テンソルも generator の消費も変わらない。
    これが「活性化は乱数を消費しない」（spec §4.2）の実装上の根拠であり、腕の
    セットアップを凍結済みの ``setup_arm`` 経路に残せる理由でもある。
    """
    st = setup_arm_p1(_p1_cfg(cfg), arm_cfg, device)
    act, alpha = _activation(cfg, arm_cfg["name"])
    form = str(cfg["activation"]["elu"].get("derivative_form", "alpha_exp"))
    st["net"].set_activation(act, alpha, form)
    st["activation"] = act
    st["act_alpha"] = float(alpha)
    st["act_grad_form"] = form
    return st


def forward_centered_elu(st: dict, x: torch.Tensor):
    """``mlp2_phase1.forward_centered`` の活性化差し替え版。

    ``net.act_fn`` は ``act == "relu"`` のとき文字通り ``torch.relu`` を返すので、
    ReLU 腕ではこの関数は phase1 の元関数と演算列が完全に一致する（S-copy / S0'）。
    """
    net, alpha = st["net"], st["center_alpha"]
    flags, means = st["centered_layers"], st["layer_means"]
    inputs, pres, acts = [], [], []
    cur = x
    for li, (W, b) in enumerate(zip(net.Ws, net.bs)):
        mean = means[li]
        cur_in = cur - mean if flags[li] else cur
        if mean is not None:
            mean.mul_(1.0 - alpha).add_(alpha * cur)
        pre = torch.einsum("rhd,rd->rh", W, cur_in) + b
        cur = net.act_fn(pre)
        inputs.append(cur_in)
        pres.append(pre)
        acts.append(cur)
    yhat = (acts[-1] * net.v).sum(dim=1) + net.c
    return inputs, pres, acts, yhat


def grads_centered_elu(net, inputs: list[torch.Tensor], pres: list[torch.Tensor],
                       acts: list[torch.Tensor], delta: torch.Tensor):
    """``mlp2_phase1.grads_centered`` の活性化差し替え版（autograd 不使用）。

    ``net.act_grad(pre, a)`` は ReLU では ``(pre > 0)``、ELU では前向きの ``a`` を
    再利用した ``a + alpha = alpha*e^z`` を返す。``exp`` を二度評価しないので
    前向きと後ろ向きが数値的に食い違うことがない（spec §4.3）。
    """
    d2 = 2.0 * delta
    gv = d2[:, None] * acts[-1]
    gc = d2
    dz = d2[:, None] * net.v * net.act_grad(pres[-1], acts[-1])
    gWs: list[torch.Tensor | None] = [None] * net.L
    gbs: list[torch.Tensor | None] = [None] * net.L
    for layer in range(net.L - 1, -1, -1):
        gbs[layer] = dz
        gWs[layer] = dz[:, :, None] * inputs[layer][:, None, :]
        if layer:
            dz = (torch.einsum("rhi,rh->ri", net.Ws[layer], dz)
                  * net.act_grad(pres[layer - 1], acts[layer - 1]))
    return gWs, gbs, gv, gc


def save_checkpoint_elu(st: dict, arm: str, step: int, outdir: Path) -> Path:
    path = outdir / "ckpts" / f"{arm}_step{step}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(step=step, arm=arm, net=st["net"].state_dict(),
                    env=st["env"].state_dict(), teacher=st["teacher"].state_dict(),
                    running_mean=st["running_mean"].clone(),
                    layer_means=[None if m is None else m.clone()
                                 for m in st["layer_means"]],
                    centered_layers=list(st["centered_layers"]),
                    activation=st["activation"], act_alpha=st["act_alpha"],
                    runs=st["runs"]), path)
    return path


def train_arm_elu(st: dict, recorder, probe_steps, total: int, outdir: Path,
                  checkpoints, stream_hook=None) -> float:
    probe_set = {int(v) for v in probe_steps}
    checkpoint_set = {int(v) for v in checkpoints}
    net, env, teacher = st["net"], st["env"], st["teacher"]
    started = time.time()
    for step in range(total):
        if step in checkpoint_set:
            save_checkpoint_elu(st, st["arm"], step, outdir)
        if step in probe_set:
            recorder(st, step)
        x = env.step()
        y = teacher(x)
        if stream_hook is not None:
            stream_hook(step, x, y)
        inputs, pres, acts, yhat = forward_centered_elu(st, x)
        grads = grads_centered_elu(net, inputs, pres, acts, yhat - y)
        net.sgd_step_layers(st["lr"], *grads)
    if total in probe_set:
        recorder(st, total)
    if total in checkpoint_set:
        save_checkpoint_elu(st, st["arm"], total, outdir)
    return time.time() - started


# --------------------------------------------------------------------------
# 厳密サポート測定（phase1 の fork + zbar / 沈下 / 事前活性スケール）
# --------------------------------------------------------------------------
def exact_layer_record_elu(st: dict, sigma_tol: float, *,
                           mean_source: str = "ema") -> tuple[dict, dict]:
    """``mlp2_phase1.exact_layer_record_p1`` に活性化と新規列を足した版。

    ReLU 腕では ``net.act_fn`` が ``torch.relu`` そのものなので、追加列を除いた
    全量が phase1 の記録と bit 一致する（S-copy がこれを検査する）。

    追加する量:

    ``zbar``               ``E_support[z_i]``（生の壁座標。正規化量ではない）
    ``dzbar``              直前の記録点からの ``zbar`` の増分（recorder が埋める）
    ``submerged``          ``max_x z_i(x) <= 0`` なユニット数。``strict_dead`` とは
                           **別経路**（amax）で計算する。S-submerge が ReLU 腕で
                           両者の完全一致を要求する
    ``preact_sd_median``   ユニット中央値の ``sd(z_i)``（= §5.1c の ``s``）
    """
    if mean_source not in ("ema", "support"):
        raise ValueError(f"unknown mean_source {mean_source!r}")
    net = st["net"]
    flags = st.get("centered_layers") or [False] * len(net.Ws)
    means = st.get("layer_means") or [None] * len(net.Ws)
    with torch.no_grad():
        X = full_support_ro(st["env"]).double()
        y = teacher_f64(st["teacher"], X)
        cur = X
        layers, sanity_layers, taut = [], [], []

        for layer, (W0, b0) in enumerate(zip(net.Ws, net.bs), start=1):
            W, b = W0.double(), b0.double()
            raw_mu_norm = float(cur.mean(dim=0).norm(dim=1).max().item())
            if flags[layer - 1]:
                offset = (cur.mean(dim=0) if mean_source == "support"
                          else means[layer - 1].double())
                cur = cur - offset[None]
            mu = cur.mean(dim=0)
            centered = cur - mu[None]
            z = torch.einsum("rhd,prd->prh", W, cur) + b
            direct_mean = z.mean(dim=0)
            direct_sd = z.var(dim=0, unbiased=False).clamp_min(0).sqrt()
            wmu = torch.einsum("rhd,rd->rh", W, mu)
            formula_mean = wmu + b
            centered_proj = torch.einsum("rhd,prd->prh", W, centered)
            denom = centered_proj.square().mean(dim=0).clamp_min(0).sqrt()
            valid = denom >= float(sigma_tol)

            M = torch.full_like(denom, float("nan"))
            B = torch.full_like(denom, float("nan"))
            M[valid] = wmu[valid] / denom[valid]
            B[valid] = b[valid] / denom[valid]
            wall_direct = direct_mean[valid] / direct_sd[valid]
            wall_formula = M[valid] + B[valid]

            activation = net.act_fn(z)
            p_hat = (z > 0).double().mean(dim=0)
            # 沈下は amax 経路で独立に出す（S-submerge が strict_dead と照合する）。
            submerged_unit = z.amax(dim=0) <= 0
            w_norm = W.norm(dim=2)
            mu_norm = mu.norm(dim=1)
            sigma_rms = centered.square().mean(dim=0).sum(dim=1)
            sigma_rms = (sigma_rms / cur.shape[2]).clamp_min(0).sqrt()
            dose = mu_norm / sigma_rms.clamp_min(1e-300)
            eff_rank = _effective_rank(activation.permute(1, 0, 2))
            eff_rank_W = _effective_rank(W)
            strict_dead = (p_hat == 0).sum(dim=1)
            alive = torch.full_like(strict_dead, W.shape[1]) - strict_dead
            eff_per_alive = torch.where(
                alive > 0, eff_rank / alive.double(),
                torch.full_like(eff_rank, float("nan")))

            qM = torch.nanquantile(M, torch.tensor([0.25, 0.5, 0.75],
                                                   dtype=M.dtype), dim=1)
            median_B = torch.nanquantile(B, 0.5, dim=1)
            qW = torch.quantile(w_norm, torch.tensor([0.25, 0.5, 0.75],
                                                     dtype=w_norm.dtype), dim=1)
            alignment = _alignment_metrics(
                W, float(st.get("sign_match_tau", 0.95)))
            layers.append(dict(
                M=M, B=B, denom=denom, p_hat=p_hat, w_norm=w_norm,
                zbar=direct_mean,
                median_M=qM[1], q25_M=qM[0], q75_M=qM[2], median_B=median_B,
                n_na=(~valid).sum(dim=1), mu_norm=mu_norm, sigma_rms=sigma_rms,
                dose=dose, w_norm_median=qW[1], w_norm_q25=qW[0],
                w_norm_q75=qW[2], eff_rank=eff_rank, eff_rank_W=eff_rank_W,
                strict_dead=strict_dead, alive=alive,
                eff_rank_per_alive=eff_per_alive,
                submerged=submerged_unit.sum(dim=1),
                preact_sd_median=torch.quantile(direct_sd, 0.5, dim=1),
                **alignment))

            cos_err = 0.0
            if layer == 1:
                mu_u = mu / mu_norm.clamp_min(1e-300)[:, None]
                cos = torch.einsum("rhd,rd->rh", W, mu_u) / w_norm.clamp_min(1e-300)
                cos_err = _max_relative(cos * mu_norm[:, None],
                                        wmu / w_norm.clamp_min(1e-300))
            finite_required = (torch.isfinite(z).all() and torch.isfinite(mu).all()
                               and torch.isfinite(denom).all()
                               and torch.isfinite(eff_rank).all()
                               and torch.isfinite(eff_rank_W).all())
            sanity_layers.append(dict(
                layer=layer,
                mean_max_relerr=_max_relative(direct_mean, formula_mean),
                sd_max_relerr=_max_relative(direct_sd, denom),
                wall_max_relerr=_max_relative(wall_direct, wall_formula),
                l1_cos_mu_max_relerr=cos_err,
                n_degenerate=int((~valid).sum().item()),
                # S-submerge: amax 経路と p_hat 経路が一致すること。
                submerge_mismatch=int((submerged_unit != (p_hat == 0)).sum().item()),
                finite_required=bool(finite_required)))
            if flags[layer - 1]:
                scale = float((w_norm.max() * max(raw_mu_norm, 1e-300)).item())
                taut.append(dict(layer=layer, mean_source=mean_source,
                                 mu_projection_max=float(wmu.abs().max().item()),
                                 projection_scale=scale,
                                 relative=float(wmu.abs().max().item()) / max(scale, 1e-300),
                                 abs_M_max=_nanmax(M),
                                 median_M_max=_nanmax(layers[-1]["median_M"])))
            cur = activation

        yhat = (cur * net.v.double()).sum(dim=-1) + net.c.double()
        residual = yhat - y
        signal_var = y.var(dim=0, unbiased=False)
        residual_var = residual.var(dim=0, unbiased=False)
        unfit = residual_var / signal_var
        run = dict(signal_var=signal_var, residual_var=residual_var, unfit=unfit,
                   eval_loss_exact=residual.square().mean(dim=0))
        run_finite = bool(all(torch.isfinite(v).all() for v in run.values())
                          and (signal_var > 0).all())
        sanity = dict(layers=sanity_layers, run_finite=run_finite,
                      support=int(X.shape[0]), taut=taut)
        return dict(run=run, layers=layers, v_readout=net.v.double(),
                    flip_state=st["env"].flip_state.double()), sanity


class EluRecorder(PhaseRecorderP1):
    """``PhaseRecorderP1`` を ELU 対応の厳密記録で駆動する。

    ``dzbar`` は float64 の ``zbar`` 同士の差として計算してから float32 に落とす。
    ``zbar`` を float32 に落としてから引くと、深い沈下域（``zbar ~ -20`` で増分が
    ``1e-9`` 級）が丸めに沈んで P2a の可動度スケーリングが測れなくなる。
    """

    def __init__(self, steps: list[int], st: dict, sigma_tol: float,
                 identity_tol: float, interval: int, *,
                 zbar_layers: list[int], readout_steps: list[int]):
        super().__init__(steps, st, sigma_tol, identity_tol)
        n, R = len(self.steps), st["R"]
        self.zbar_layers = sorted({int(v) for v in zbar_layers})
        for li, width in enumerate(st["hidden"]):
            if li + 1 in self.zbar_layers:
                for key in ELU_EXTRA_UNIT_KEYS:
                    self.layers[li][key] = np.empty((n, R, width), dtype=np.float32)
            self.layers[li]["preact_sd_median"] = np.empty((n, R), dtype=np.float64)
            self.layers[li]["submerged"] = np.empty((n, R), dtype=np.int64)
        self.readout_steps = np.asarray(sorted(readout_steps), dtype=np.int64)
        self.readout_index = {int(v): i for i, v in enumerate(self.readout_steps)}
        self.readout = np.empty((len(self.readout_steps), R, st["hidden"][-1]),
                                dtype=np.float32)
        self.interval = int(interval)
        self._prev_zbar: list[torch.Tensor] | None = None
        self._prev_step: int | None = None
        self.submerge_mismatch = 0

    def __call__(self, st: dict, step: int) -> None:
        i = self.index.get(int(step))
        if i is None:
            return
        if self.filled[i]:
            raise RuntimeError(f"duplicate elu_swamp probe at step {step}")
        divergence = _numeric_divergence_event(st, int(step))
        if divergence is not None:
            raise NumericDivergenceError(divergence)
        rec, sanity = exact_layer_record_elu(st, self.sigma_tol)
        for key, value in rec["run"].items():
            self.run[key][i] = value.detach().cpu().numpy()
        self.flip_state[i] = rec["flip_state"].detach().cpu().numpy().astype(np.float32)
        ri = self.readout_index.get(int(step))
        if ri is not None:
            self.readout[ri] = rec["v_readout"].detach().cpu().numpy().astype(np.float32)

        adjacent = (self._prev_step is not None
                    and int(step) - int(self._prev_step) == self.interval)
        zbar_now = []
        for li, layer in enumerate(rec["layers"]):
            for key in LOG_UNIT_KEYS:
                self.layers[li][key][i] = layer[key].detach().cpu().numpy().astype(np.float32)
            for key in ELU_LOG_LAYER_KEYS:
                self.layers[li][key][i] = layer[key].detach().cpu().numpy()
            zbar = layer["zbar"].detach()
            zbar_now.append(zbar.clone())
            if li + 1 in self.zbar_layers:
                self.layers[li]["zbar"][i] = zbar.cpu().numpy().astype(np.float32)
                if adjacent:
                    delta = (zbar - self._prev_zbar[li]).cpu().numpy()
                else:
                    delta = np.full(zbar.shape, np.nan, dtype=np.float64)
                self.layers[li]["dzbar"][i] = delta.astype(np.float32)

            s, acc = sanity["layers"][li], self.max_errors[li]
            acc["mean"] = max(acc["mean"], s["mean_max_relerr"])
            acc["sd"] = max(acc["sd"], s["sd_max_relerr"])
            acc["wall"] = max(acc["wall"], s["wall_max_relerr"])
            acc["cos_mu"] = max(acc["cos_mu"], s["l1_cos_mu_max_relerr"])
            acc["n_degenerate_max"] = max(acc["n_degenerate_max"], s["n_degenerate"])
            self.submerge_mismatch += int(s["submerge_mismatch"])
            if not s["finite_required"]:
                self.required_nonfinite += 1
        if not sanity["run_finite"]:
            self.required_nonfinite += 1
        self._prev_zbar, self._prev_step = zbar_now, int(step)
        self.filled[i] = True

    def sanity(self) -> dict:
        out = super().sanity()
        out["submerge_mismatch"] = int(self.submerge_mismatch)
        out["pass_"] = bool(out["pass_"] and self.submerge_mismatch == 0)
        return out


def write_arm_logs_elu(outdir: Path, arm: str, st: dict,
                       rec: EluRecorder) -> list[Path]:
    logdir = outdir / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ri, run in enumerate(st["runs"]):
        seed = int(run["seed"])
        payload = dict(step=rec.steps, run_id=np.array(run["run_id"]),
                       arm=np.array(arm), seed=np.int64(seed),
                       activation=np.array(st["activation"]),
                       act_alpha=np.float64(st["act_alpha"]),
                       task_period=np.int64(run["period"]),
                       state_hash_final=np.array(json.dumps(
                           _seed_state_hashes_p1(st, ri), sort_keys=True)))
        for key, value in rec.run.items():
            payload[key] = value[:, ri]
        payload["flip_state"] = rec.flip_state[:, ri]
        payload["v_readout"] = rec.readout[:, ri]
        payload["v_readout_step"] = rec.readout_steps
        for li, layer in enumerate(rec.layers, start=1):
            for key, value in layer.items():
                payload[f"layer{li}_{key}"] = value[:, ri]
        path = logdir / f"{arm}_seed{seed}.npz"
        np.savez_compressed(path, **payload)
        paths.append(path)
    return paths


# --------------------------------------------------------------------------
# 前段チェック（spec §6）
# --------------------------------------------------------------------------
def _s_grad_check(cfg: dict, device: str) -> dict:
    """S-grad: ELU の閉形式勾配を独立実装・有限差分・網全体の 3 層で照合する。

    (a) 前向き/後ろ向きを PyTorch の ``F.elu``（別実装の C++ カーネル）とその
        autograd に対して照合する。深い負側 ``z=-30`` では真の微分が ``e^-30``
        で関数値 ``~ -1`` の 13 桁下にあり、float64 の差分商では原理的に分離
        できないので、そこはこの独立実装が判定を担う。
    (b) 登録された probe 点で中心差分を取る。差分商自体の丸め床
        ``eps*|phi| / (h*|phi'|)`` を各点で出し、床が 1e-8 未満の点だけを判定に
        使う（残りは数値を出すだけで落とさない）。
    (c) 網全体: float64 の小さな 2 層 ELU 網で、損失を各パラメータについて中心
        差分し ``grads_centered_elu`` と突き合わせる。中心化した層入力を通る
        実際の学習経路そのものを覆う。
    """
    import torch.nn.functional as F

    S = cfg["sanity"]
    tol = float(S["s_grad_finite_difference_tol"])
    h = float(S["s_grad_finite_difference_h"])
    alpha = float(cfg["activation"]["elu"]["alpha"])
    points = torch.tensor([float(v) for v in S["s_grad_probe_points"]],
                          dtype=torch.float64, device=device)

    from .nets import VecMLPL
    form = str(cfg["activation"]["elu"]["derivative_form"])
    probe = VecMLPL(1, [2], 2, torch.Generator(device=device).manual_seed(0),
                    device).set_activation("elu", alpha, form)

    a = probe.act_fn(points)
    ref_a = F.elu(points, alpha)
    forward_relerr = float((a - ref_a).abs().max()
                           / ref_a.abs().max().clamp_min(1e-300))
    z = points.clone().requires_grad_(True)
    F.elu(z, alpha).sum().backward()
    ref_grad = z.grad.detach()
    got_grad = probe.act_grad(points, a)
    per_point = (got_grad - ref_grad).abs() / ref_grad.abs().clamp_min(1e-300)
    backward_relerr = float(per_point.max())

    # 登録形と却下形を float64 / float32 の両方で並べて記録に残す。
    # 学習は float32 なので、そこでの深い負側の壊れ方が判定に効く。
    forms = {}
    for form in probe.GRAD_FORMS:
        probe.act_grad_form = form
        per_dtype = {}
        for dtype in (torch.float64, torch.float32):
            zz = points.to(dtype)
            got = probe.act_grad(zz, probe.act_fn(zz)).double()
            rel = ((got - ref_grad).abs()
                   / ref_grad.abs().clamp_min(1e-300))
            per_dtype[str(dtype).replace("torch.", "")] = dict(
                max_relerr=float(rel.max()),
                per_point={f"{float(v):g}": float(r)
                           for v, r in zip(points, rel)},
                zeroed_points=[float(v) for v, g in zip(points, got)
                               if float(g) == 0.0 and float(v) <= 0.0])
        forms[form] = per_dtype
    probe.act_grad_form = str(cfg["activation"]["elu"]["derivative_form"])
    registered = forms[probe.act_grad_form]
    registered_ok = all(v["max_relerr"] <= tol for v in registered.values())

    eps = float(np.finfo(np.float64).eps)
    fd_rows = []
    fd_fail = []
    fwd, bwd = probe.act_fn(points + h), probe.act_fn(points - h)
    central = (fwd - bwd) / (2.0 * h)
    for idx in range(points.numel()):
        zi = float(points[idx])
        exact = float(ref_grad[idx])
        floor = (eps * float(a[idx].abs()) / (h * abs(exact))
                 if abs(exact) > 0 else float("inf"))
        rel = abs(float(central[idx]) - exact) / max(abs(exact), 1e-300)
        row = dict(z=zi, closed_form=float(got_grad[idx]), autograd=exact,
                   central_difference=float(central[idx]),
                   closed_form_relerr=float(per_point[idx]),
                   central_difference_relerr=rel,
                   roundoff_floor=floor, fd_informative=bool(floor < 1e-8))
        fd_rows.append(row)
        if row["fd_informative"] and not rel <= tol:
            fd_fail.append(row)

    # (c) 網全体の中心差分。
    gen = torch.Generator(device=device).manual_seed(20260830)
    net = VecMLPL(2, [4, 3], 5, gen, device).set_activation("elu", alpha, form)
    for name in ("Ws", "bs"):
        setattr(net, name, [t.double() for t in getattr(net, name)])
    net.v, net.c = net.v.double(), net.c.double()
    net.W, net.b = net.Ws[0], net.bs[0]
    x = torch.rand(2, 5, generator=gen, device=device).double()
    offset = torch.rand(2, 5, generator=gen, device=device).double() * 0.1
    target = torch.rand(2, generator=gen, device=device).double()

    def loss_value() -> torch.Tensor:
        cur = x - offset
        for li, (W, b) in enumerate(zip(net.Ws, net.bs)):
            cur_in = cur if li else cur
            pre = torch.einsum("rhd,rd->rh", W, cur_in) + b
            cur = net.act_fn(pre)
        yhat = (cur * net.v).sum(dim=1) + net.c
        return (yhat - target).square()

    inputs = [x - offset]
    cur, pres, acts = x - offset, [], []
    for W, b in zip(net.Ws, net.bs):
        pre = torch.einsum("rhd,rd->rh", W, cur) + b
        cur = net.act_fn(pre)
        pres.append(pre)
        acts.append(cur)
        inputs.append(cur)
    yhat = (acts[-1] * net.v).sum(dim=1) + net.c
    gWs, gbs, gv, gc = grads_centered_elu(net, inputs[:-1], pres, acts,
                                          yhat - target)

    net_worst = 0.0
    hn = 1e-6
    tensors = [(f"W{i + 1}", W, gWs[i]) for i, W in enumerate(net.Ws)]
    tensors += [(f"b{i + 1}", b, gbs[i]) for i, b in enumerate(net.bs)]
    tensors += [("v", net.v, gv), ("c", net.c.unsqueeze(1), gc.unsqueeze(1))]
    for name, param, analytic in tensors:
        flat = param.reshape(param.shape[0], -1)
        gflat = analytic.reshape(analytic.shape[0], -1)
        for j in range(flat.shape[1]):
            original = flat[:, j].clone()
            flat[:, j] = original + hn
            up = loss_value()
            flat[:, j] = original - hn
            down = loss_value()
            flat[:, j] = original
            fd = (up - down) / (2.0 * hn)
            scale = torch.maximum(fd.abs(), gflat[:, j].abs()).clamp_min(1e-12)
            net_worst = max(net_worst,
                            float(((fd - gflat[:, j]).abs() / scale).max()))

    passed = bool(forward_relerr <= tol and backward_relerr <= tol
                  and registered_ok and not fd_fail and net_worst <= 1e-6)
    return dict(pass_=passed, alpha=alpha, tolerance=tol, step=h,
                derivative_form=probe.act_grad_form,
                forward_vs_torch_elu_relerr=forward_relerr,
                backward_vs_autograd_relerr=backward_relerr,
                registered_form_within_tolerance=bool(registered_ok),
                derivative_forms=forms,
                network_finite_difference_max_relerr=net_worst,
                central_difference=fd_rows, central_difference_failures=fd_fail)


def _s_elu_limit_check(cfg: dict, device: str, steps: int = 2000) -> dict:
    """S-elu-limit: ``alpha -> 0`` で ELU 経路が ReLU 経路に一致すること（教訓⑩）。

    恒真のサニティを避けるため、静的な一致だけでなく ``steps`` ステップ学習させた
    後の全パラメータのハッシュまで一致することを要求する。
    """
    if cfg["sanity"]["s_elu_limit_alpha_to_zero"] is not True:
        raise ValueError("S-elu-limit is registered as required")
    c = _p1_cfg(cfg)
    c["common"]["seeds"] = [0, 1]
    zero = copy.deepcopy(c)
    zero["activation"]["elu"]["alpha"] = 0.0
    relu_arm = copy.deepcopy(_arm(cfg, "R_A1"))
    elu_arm = copy.deepcopy(relu_arm)
    elu_arm["activation"] = "elu"

    st_relu = setup_arm_elu(c, relu_arm, device)
    st_elu = setup_arm_elu(zero, elu_arm, device)
    grid = torch.linspace(-30.0, 30.0, 4001, dtype=torch.float64, device=device)
    static_equal = bool(torch.equal(st_relu["net"].act_fn(grid),
                                    st_elu["net"].act_fn(grid)))
    static_grad_equal = bool(torch.equal(
        st_relu["net"].act_grad(grid, st_relu["net"].act_fn(grid)),
        st_elu["net"].act_grad(grid, st_elu["net"].act_fn(grid))))
    train_arm_elu(st_relu, lambda *_: None, [], steps, Path("."), [])
    train_arm_elu(st_elu, lambda *_: None, [], steps, Path("."), [])
    hashes_relu = _init_hashes(st_relu)
    hashes_elu = _init_hashes(st_elu)
    differences = sorted(k for k, v in hashes_relu.items()
                         if hashes_elu.get(k) != v)
    return dict(pass_=bool(static_equal and static_grad_equal and not differences),
                steps=steps, static_forward_equal=static_equal,
                static_grad_equal=static_grad_equal,
                trained_state_differences=differences,
                grid=[-30.0, 30.0, 4001])


def _s_copy_check(cfg: dict, device: str, outdir: Path) -> dict:
    """S-copy: ReLU 腕では本モジュールの厳密記録が phase1 のそれと一致すること。

    fork が黙って漂うと S0' も壁座標も同時に無効になるので、追加列を除いた全量を
    ハッシュで突き合わせる。中心化あり・なしの両方を掛ける。
    """
    c = _p1_cfg(cfg)
    c["common"]["seeds"] = [0, 1]
    tol = float(cfg["elu_swamp"]["sigma_degenerate_tol"])
    shared_unit = LOG_UNIT_KEYS
    shared_layer = P1_LOG_LAYER_KEYS
    differences = []
    for arm in RELU_ARMS:
        st = setup_arm_elu(c, _arm(cfg, arm), device)
        for step in (0, 2000):
            if step:
                train_arm_elu(st, lambda *_: None, [], step, outdir, [])
            new, _ = exact_layer_record_elu(st, tol)
            old, _ = exact_layer_record_p1(st, tol)
            for key in new["run"]:
                if _sha_array(new["run"][key]) != _sha_array(old["run"][key]):
                    differences.append(dict(arm=arm, step=step, where=f"run.{key}"))
            if _sha_array(new["flip_state"]) != _sha_array(old["flip_state"]):
                differences.append(dict(arm=arm, step=step, where="flip_state"))
            for li, (a, b) in enumerate(zip(new["layers"], old["layers"]), start=1):
                for key in shared_unit + shared_layer:
                    if _sha_array(a[key]) != _sha_array(b[key]):
                        differences.append(dict(arm=arm, step=step,
                                                where=f"layer{li}.{key}"))
        del st
    return dict(pass_=not differences, differences=differences,
                arms=list(RELU_ARMS), steps=[0, 2000], seeds=[0, 1],
                compared_keys=sorted(set(shared_unit + shared_layer)))


def _s_submerge_check(cfg: dict, device: str, outdir: Path) -> dict:
    """S-submerge: ``max_x z_i <= 0``（amax 経路）が ReLU 腕の ``strict_dead`` と
    完全一致すること。片方は amax、もう片方は ``p_hat`` の平均から出しており、
    同じ式を二度書いただけの恒真ではない。"""
    c = _p1_cfg(cfg)
    c["common"]["seeds"] = [0, 1]
    tol = float(cfg["elu_swamp"]["sigma_degenerate_tol"])
    rows, mismatches = [], []
    for arm in ARM_ORDER:
        st = setup_arm_elu(c, _arm(cfg, arm), device)
        train_arm_elu(st, lambda *_: None, [], 5000, outdir, [])
        rec, sanity = exact_layer_record_elu(st, tol)
        for li, layer in enumerate(rec["layers"], start=1):
            same = bool(torch.equal(layer["submerged"], layer["strict_dead"]))
            row = dict(arm=arm, activation=st["activation"], layer=li,
                       submerged=[int(v) for v in layer["submerged"]],
                       strict_dead=[int(v) for v in layer["strict_dead"]],
                       equal=same,
                       elementwise_mismatch=int(
                           sanity["layers"][li - 1]["submerge_mismatch"]))
            rows.append(row)
            # 一致を要求するのは ReLU 腕（spec §6）。ELU 腕は数値を出すだけ。
            if arm in RELU_ARMS and not (same and row["elementwise_mismatch"] == 0):
                mismatches.append(row)
            if row["elementwise_mismatch"]:
                mismatches.append(row)
        del st
    return dict(pass_=not mismatches, steps=5000, rows=rows, failures=mismatches)


def _s_pair_check(cfg: dict, device: str, outdir: Path) -> dict:
    """S-pair / S-taut: 5 腕が対応づくこと、A を入れた層で µ 項が消えること。"""
    S, P = cfg["sanity"], cfg["elu_swamp"]
    steps = int(S["s_pair_steps"])
    arms = [str(a) for a in cfg["pairing"]["paired_groups"][0]]
    tol = float(P["sigma_degenerate_tol"])
    init, final, stream, taut, taut_ema = {}, {}, {}, {}, {}
    for name in arms:
        c = _p1_cfg(cfg)
        st = setup_arm_elu(c, _arm(cfg, name), device)
        init[name] = _init_hashes(st)
        digest = StreamDigest()
        print(f"[S-pair] {name} ({st['activation']}) {steps:,} steps x "
              f"{len(c['common']['seeds'])} seeds", flush=True)
        train_arm_elu(st, lambda *_: None, [], steps, outdir, [], stream_hook=digest)
        final[name] = _env_hashes(st)
        stream[name] = digest.digest()
        _, sanity_support = exact_layer_record_elu(st, tol, mean_source="support")
        _, sanity_ema = exact_layer_record_elu(st, tol, mean_source="ema")
        taut[name] = sanity_support["taut"]
        taut_ema[name] = sanity_ema["taut"]
        del st

    reference = arms[0]
    differences = []
    for name in arms[1:]:
        for key, value in init[reference].items():
            if init[name].get(key) != value:
                differences.append(dict(arm=name, where=f"init.{key}"))
        for key, value in final[reference].items():
            if final[name].get(key) != value:
                differences.append(dict(arm=name, where=f"final.{key}"))
        for key in ("x", "y", "n"):
            if stream[name][key] != stream[reference][key]:
                differences.append(dict(arm=name, where=f"stream.{key}"))

    staut_tol = float(S["s1_identity_tol"])
    taut_rows = [dict(arm=name, **row) for name, rows in taut.items() for row in rows]
    taut_fail = [r for r in taut_rows if not (r["relative"] <= staut_tol)]
    if not taut_rows:
        taut_fail = [dict(arm="", note="no centered layer was checked")]
    return dict(pass_=bool(not differences and not taut_fail),
                spair=dict(pass_=not differences, arms=arms, steps=steps,
                           reference=reference, differences=differences,
                           caveat=str(cfg["pairing"]["pairing_caveat"]),
                           init_hashes=init, final_env_hashes=final,
                           stream_digests=stream),
                staut=dict(pass_=not taut_fail, tolerance=staut_tol,
                           exact_substitution=taut_rows, failures=taut_fail,
                           ema_residual_report_only=[
                               dict(arm=name, **row)
                               for name, rows in taut_ema.items() for row in rows]))


def _ci(cfg: dict, values: np.ndarray, draws: np.ndarray) -> dict:
    P = cfg["elu_swamp"]
    return _ci_components(np.asarray(values, dtype=np.float64), draws, "median",
                          float(P["degenerate_se_tol"]),
                          float(P["degenerate_frac_max"]),
                          float(P["degenerate_width_ratio_max"]))


def _s5_selftest(cfg: dict) -> dict:
    P = cfg["elu_swamp"]
    n = len(cfg["common"]["seeds"])
    rng = np.random.default_rng(int(P["bootstrap_seed"]))
    draws = rng.integers(0, n, size=(int(P["bootstrap_B"]), n))
    result = _ci(cfg, np.zeros(n), draws)
    return dict(pass_=bool(result["ci_degenerate"]), result=result)


def _s7_numeric_divergence_selftest(cfg: dict, device: str) -> dict:
    """NaN を 1 個注入して §5.4 が厳密 SVD の前に捕まえることを確認する。"""
    c = _p1_cfg(cfg)
    c["common"]["seeds"] = [0, 1]
    st = setup_arm_elu(c, _arm(cfg, "E_Aall"), device)
    st["net"].Ws[1][1, 0, 0] = float("nan")
    rec = EluRecorder([0], st, float(cfg["elu_swamp"]["sigma_degenerate_tol"]),
                      float(cfg["sanity"]["s1_identity_tol"]),
                      int(cfg["common"]["lop_every"]),
                      zbar_layers=[int(v) for v in
                                   cfg["elu_swamp"]["record_zbar_layers"]],
                      readout_steps=[])
    event = None
    try:
        rec(st, 0)
    except NumericDivergenceError as exc:
        event = exc.event
    passed = bool(
        event
        and event.get("status") == NUMERIC_DIVERGENCE
        and event.get("bad_seeds") == [1]
        and "net.Ws.2" in event.get("nonfinite_tensors", {}).get("1", [])
        and not rec.filled.any())
    return dict(pass_=passed, injected_arm="E_Aall", injected_seed=1,
                detected_event=event, exact_record_skipped=bool(not rec.filled.any()))


def _s_closed_form_table(cfg: dict) -> dict:
    """spec §2.1 の g(s) 表を実装が再現することを固定する（§11.8）。"""
    expected = {0.35: 0.08, 1.0: 0.20, 2.0: 0.33, 4.0: 0.45}
    rows, failures = [], []
    for s, target in expected.items():
        value = float(g_elu(s))
        row = dict(s=s, g=value, spec_table=target, abs_error=abs(value - target))
        rows.append(row)
        if row["abs_error"] > 0.005:
            failures.append(row)
    limit = float(g_elu(1e4))
    relu_ratio = (1.0 / SQRT2PI) / math.sqrt(0.5 - 1.0 / (2 * math.pi))
    if abs(limit - relu_ratio) > 1e-3:
        failures.append(dict(s="1e4", g=limit, spec_table=relu_ratio,
                             abs_error=abs(limit - relu_ratio)))
    return dict(pass_=not failures, table=rows, large_s_limit=limit,
                relu_ratio=relu_ratio, failures=failures)


def preflight(cfg: dict, device: str, outdir: Path) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    omp = require_omp(cfg)
    checks: dict[str, dict] = {"S3": omp}
    print("[S-gelu] closed-form g(s) vs spec §2.1 table", flush=True)
    checks["S_g_elu_table"] = _s_closed_form_table(cfg)
    print(f"[S-gelu] {'PASS' if checks['S_g_elu_table']['pass_'] else 'FAIL'}", flush=True)
    print("[S-grad] ELU closed-form gradient", flush=True)
    checks["S_grad"] = _s_grad_check(cfg, device)
    print(f"[S-grad] {'PASS' if checks['S_grad']['pass_'] else 'FAIL'}", flush=True)
    print("[S-elu-limit] alpha -> 0 collapses onto the ReLU path", flush=True)
    checks["S_elu_limit"] = _s_elu_limit_check(cfg, device)
    print(f"[S-elu-limit] {'PASS' if checks['S_elu_limit']['pass_'] else 'FAIL'}", flush=True)
    print("[S-copy] elu_swamp exact record vs phase1 reference (ReLU arms)", flush=True)
    checks["S_copy"] = _s_copy_check(cfg, device, outdir / "scopy")
    print(f"[S-copy] {'PASS' if checks['S_copy']['pass_'] else 'FAIL'}", flush=True)
    print("[S-submerge] amax submergence vs strict_dead", flush=True)
    checks["S_submerge"] = _s_submerge_check(cfg, device, outdir / "ssubmerge")
    print(f"[S-submerge] {'PASS' if checks['S_submerge']['pass_'] else 'FAIL'}", flush=True)
    pair = _s_pair_check(cfg, device, outdir / "spair")
    checks["S_pair"], checks["S_taut"] = pair["spair"], pair["staut"]
    print(f"[S-pair] {'PASS' if pair['spair']['pass_'] else 'FAIL'}  "
          f"[S-taut] {'PASS' if pair['staut']['pass_'] else 'FAIL'}", flush=True)
    checks["S5"] = _s5_selftest(cfg)
    print(f"[S5] {'PASS' if checks['S5']['pass_'] else 'FAIL'}", flush=True)
    checks["S7_numeric_divergence"] = _s7_numeric_divergence_selftest(cfg, device)
    print(f"[S7-divergence] "
          f"{'PASS' if checks['S7_numeric_divergence']['pass_'] else 'FAIL'}", flush=True)
    result = dict(pass_=bool(all(v.get("pass_") for v in checks.values())), **checks)
    (outdir / "preflight.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    if not result["pass_"]:
        failed = [k for k, v in checks.items() if not v.get("pass_")]
        raise RuntimeError(f"preflight failed: {failed}")
    return result


# --------------------------------------------------------------------------
# S0'（spec §6）と本走の腕
# --------------------------------------------------------------------------
def _readout_steps(cfg: dict, probe_steps: list[int]) -> list[int]:
    """v_i を残す記録点。§5.3 は末尾窓のタスク末尾しか読まない。"""
    P = cfg["elu_swamp"]
    if not P.get("record_readout_at_task_end_only", True):
        return list(probe_steps)
    period = int(P["task_period"])
    return [int(s) for s in probe_steps if s > 0 and int(s) % period == 0]


def _arm_status_path(outdir: Path, arm: str) -> Path:
    return outdir / "arm_status" / f"{arm}.json"


def _write_divergence_status(outdir: Path, event: dict) -> Path:
    path = _arm_status_path(outdir, str(event["arm"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(event, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _load_divergence_status(outdir: Path, arm: str, seeds: list[int],
                            total: int, probe_every: int) -> dict | None:
    path = _arm_status_path(outdir, arm)
    if not path.exists():
        return None
    try:
        event = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    valid = (event.get("status") == NUMERIC_DIVERGENCE
             and event.get("arm") == arm
             and event.get("registered_seeds") == seeds
             and int(event.get("registered_total_steps", -1)) == total
             and int(event.get("probe_every", -1)) == probe_every
             and event.get("rescue") == "none")
    return event if valid else None


def _run_arm(cfg: dict, arm: str, device: str, outdir: Path, seeds: list[int],
             total: int) -> dict:
    C, P = cfg["common"], cfg["elu_swamp"]
    c = _p1_cfg(cfg)
    c["common"]["seeds"] = seeds
    arm_cfg = _arm(cfg, arm)
    probe_steps = list(range(0, total + 1, int(C["lop_every"])))
    if probe_steps[-1] != total:
        probe_steps.append(total)
    st = setup_arm_elu(c, arm_cfg, device)
    print(f"[{arm}] act={st['activation']}(alpha={st['act_alpha']:g}) "
          f"hidden={arm_cfg['hidden']} centered={arm_cfg['centered_layers']} "
          f"seeds={seeds} steps={total:,}", flush=True)
    _, before = exact_layer_record_elu(st, float(P["sigma_degenerate_tol"]))
    if not identity_sanity_pass(before, float(cfg["sanity"]["s1_identity_tol"])):
        raise RuntimeError(f"{arm} preflight identity failed")
    rec = EluRecorder(probe_steps, st, float(P["sigma_degenerate_tol"]),
                      float(cfg["sanity"]["s1_identity_tol"]), int(C["lop_every"]),
                      zbar_layers=[int(v) for v in P["record_zbar_layers"]],
                      readout_steps=_readout_steps(cfg, probe_steps))
    checkpoints = [int(v) for v in C.get("checkpoints", []) if int(v) <= total]
    started = time.time()
    try:
        elapsed = train_arm_elu(st, rec, probe_steps, total, outdir, checkpoints)
    except NumericDivergenceError as exc:
        elapsed = time.time() - started
        event = dict(exc.event)
        event.update(probe_every=int(C["lop_every"]),
                     registered_total_steps=int(total),
                     registered_seeds=[int(v) for v in seeds],
                     activation=st["activation"], act_alpha=st["act_alpha"],
                     elapsed_sec=float(elapsed),
                     detection="nonfinite_training_state_at_probe",
                     contrast_policy="any_involved_arm_is_numeric_divergence",
                     partial_logs_excluded=True, rescue="none")
        status_path = _write_divergence_status(outdir, event)
        print(f"[{arm}] {NUMERIC_DIVERGENCE} at step {event['detected_step']:,}; "
              f"seeds={event['bad_seeds']} -> {status_path}", flush=True)
        result = dict(status=NUMERIC_DIVERGENCE, elapsed_sec=elapsed,
                      sanity=dict(pass_=False, numeric_divergence=True, event=event),
                      divergence=event, final_env=_env_hashes(st))
        del rec, st
        return result
    sanity = rec.sanity()
    if not sanity["pass_"]:
        raise RuntimeError(f"{arm} S1/S2/S-submerge failed: {sanity}")
    write_arm_logs_elu(outdir, arm, st, rec)
    print(f"[{arm}] complete in {elapsed:.1f}s", flush=True)
    result = dict(status="COMPLETE", elapsed_sec=elapsed, sanity=sanity,
                  final_env=_env_hashes(st))
    del rec, st
    return result


def _compare_arm_logs(ours: Path, theirs: Path) -> list[dict]:
    """既存列の完全一致。本モジュールが足した列だけ「phase1 に無い」を許す。"""
    new_suffixes = tuple("_" + key for key in
                         ELU_EXTRA_UNIT_KEYS + ELU_EXTRA_LAYER_KEYS)
    new_names = set(ELU_EXTRA_RUN_KEYS)
    with np.load(ours, allow_pickle=False) as a, np.load(theirs, allow_pickle=False) as b:
        keys_a = {key for key in set(a.files) - S0PRIME_META_KEYS
                  if key not in new_names and not key.endswith(new_suffixes)}
        keys_b = set(b.files) - S0PRIME_META_KEYS
        differences = [dict(column=k, reason="missing in phase1 reference")
                       for k in sorted(keys_a - keys_b)]
        differences += [dict(column=k, reason="missing in elu_swamp")
                        for k in sorted(keys_b - keys_a)]
        for key in sorted(keys_a & keys_b):
            if _sha_array(a[key]) != _sha_array(b[key]):
                differences.append(dict(column=key, reason="hash mismatch"))
    return differences


def s0prime(cfg: dict, device: str, outdir: Path) -> dict:
    """S0': R_none / R_A1 が phase1 の L2_none / L2_A1 を bit 再現すること。

    本物の 5M を走らせて ``outdir/logs`` に置くので、本走はこの 2 腕を再計算しない。
    """
    outdir.mkdir(parents=True, exist_ok=True)
    require_omp(cfg)
    C, S = cfg["common"], cfg["sanity"]
    total, seeds = int(C["total_steps"]), [int(v) for v in C["seeds"]]
    reference_dir = Path(ROOT) / S["s0_prime_baseline_ref"]
    mapping = dict(S["s0_prime_arm_map"])
    arms_result, elapsed = {}, {}

    for arm, reference_arm in mapping.items():
        if _complete_arm_logs(outdir, arm, seeds, total, int(C["lop_every"])):
            print(f"[S0'] complete {arm} logs found; comparing only", flush=True)
            elapsed[arm] = 0.0
        else:
            elapsed[arm] = _run_arm(cfg, arm, device, outdir, seeds,
                                    total)["elapsed_sec"]
        differences, missing = [], []
        for seed in seeds:
            theirs = reference_dir / "logs" / f"{reference_arm}_seed{seed}.npz"
            ours = outdir / "logs" / f"{arm}_seed{seed}.npz"
            if not theirs.exists():
                missing.append(str(theirs))
                continue
            differences += [dict(seed=seed, **d)
                            for d in _compare_arm_logs(ours, theirs)]

        reference_ckpt = reference_dir / "ckpts" / f"{reference_arm}_step{total}.pt"
        state_differences, expected_state, actual_state = [], {}, {}
        ours_ckpt = outdir / "ckpts" / f"{arm}_step{total}.pt"
        if not reference_ckpt.exists():
            missing.append(str(reference_ckpt))
        elif not ours_ckpt.exists():
            missing.append(str(ours_ckpt))
        else:
            ck = torch.load(reference_ckpt, map_location="cpu", weights_only=False)
            mine = torch.load(ours_ckpt, map_location="cpu", weights_only=False)
            for label, blob, into in (("expected", ck, expected_state),
                                      ("actual", mine, actual_state)):
                into.update({f"net.{k}": _sha_array(v) for k, v in blob["net"].items()})
                into.update(env_flip_state=_sha_array(blob["env"]["flip_state"]),
                            env_t=str(blob["env"]["t"]),
                            running_mean=_sha_array(blob["running_mean"]))
            state_differences = sorted(k for k, v in expected_state.items()
                                       if actual_state.get(k) != v)
        arms_result[arm] = dict(
            pass_=bool(not differences and not missing and not state_differences),
            reference_arm=reference_arm, missing=missing,
            column_differences=differences, state_differences=state_differences,
            expected_state_hash=expected_state, actual_state_hash=actual_state)

    result = dict(pass_=bool(all(v["pass_"] for v in arms_result.values())),
                  arms=arms_result, reference=str(reference_dir),
                  total_steps=total, seeds=seeds, elapsed_sec=elapsed,
                  ignored_columns=sorted(S0PRIME_META_KEYS),
                  new_columns=sorted(set(ELU_EXTRA_UNIT_KEYS)
                                     | set(ELU_EXTRA_LAYER_KEYS)
                                     | set(ELU_EXTRA_RUN_KEYS)))
    (outdir / "s0prime.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"S0' {'PASS' if result['pass_'] else 'FAIL'}", flush=True)
    if not result["pass_"]:
        raise RuntimeError("S0' failed; the full run must not proceed (spec §6)")
    return result


def _pair_check_final(cfg: dict, outdir: Path, seeds: list[int],
                      divergences: dict[str, dict] | None = None) -> dict:
    """本走後も 5 腕が同じ環境を共有していたこと（5M 全域）を確認する。"""
    def env_of(logdir: Path, arm: str, seed: int) -> dict:
        with np.load(logdir / f"{arm}_seed{seed}.npz", allow_pickle=False) as z:
            state = json.loads(str(z["state_hash_final"]))
        return {k: state[k] for k in ("env.flip_state", "env.t")}

    divergences = divergences or {}
    completed = [arm for arm in ARM_ORDER if arm not in divergences]
    reference_arm = next((a for a in completed), None)
    if reference_arm is None:
        return dict(pass_=False, paired_pass=False, paired_arms=[],
                    not_tested_divergent=sorted(divergences),
                    differences=[dict(where="every_arm_diverged")])
    differences = []
    for seed in seeds:
        reference = env_of(outdir / "logs", reference_arm, seed)
        for arm in completed:
            if arm == reference_arm:
                continue
            if env_of(outdir / "logs", arm, seed) != reference:
                differences.append(dict(seed=seed, arm=arm, where="env"))
    return dict(pass_=not differences, paired_pass=not differences,
                reference_arm=reference_arm, paired_arms=completed,
                caveat=str(cfg["pairing"]["pairing_caveat"]),
                not_tested_divergent=sorted(divergences),
                partial_due_to_numeric_divergence=bool(divergences),
                differences=differences)


# --------------------------------------------------------------------------
# 集計（spec §5 / §11）— ゲートが通るまで走らせない
# --------------------------------------------------------------------------
def _arm_arrays(logdir: Path, arm: str, seeds: list[int], depth: int,
                period: int) -> dict:
    """タスク末尾の記録点だけを読む（水準・傾き・dose 用）。"""
    per_seed = []
    for seed in seeds:
        with np.load(logdir / f"{arm}_seed{seed}.npz", allow_pickle=False) as z:
            idx = np.flatnonzero((z["step"] > 0) & (z["step"] % period == 0))
            per_seed.append({
                "steps": z["step"][idx].copy(), "unfit": z["unfit"][idx].copy(),
                "layers": [{k: z[f"layer{li}_{k}"][idx].copy()
                            for k in ELU_LOG_LAYER_KEYS}
                           for li in range(1, depth + 1)]})
    result = {"steps": per_seed[0]["steps"],
              "unfit": np.stack([v["unfit"] for v in per_seed], axis=1),
              "layers": []}
    for li in range(depth):
        result["layers"].append({
            k: np.stack([v["layers"][li][k] for v in per_seed], axis=1)
            for k in ELU_LOG_LAYER_KEYS})
    return result


def _interval_rows(cfg: dict, logdir: Path, arm: str, seed: int,
                   layer: int) -> dict:
    """1 seed 分のタスク内増分をビン集計まで落とす（生の 10^7 行は持たない）。

    区間始点 ``s`` は ``s > 0`` かつ ``s % T != 0``（spec §11.2）。probe はステップ
    ``k*T`` の**前**に走るので、この規則で 1 タスクあたりちょうど 9 区間になり、
    タスク境界を跨ぐ区間と step 0 起点の区間が落ちる。
    """
    P = cfg["elu_swamp"]
    period = int(P["task_period"])
    every = int(P["q2_increment_interval_steps"])
    nbins = int(P["q2_bins"])
    min_count = int(P["q2_bin_min_count"])
    windows = {
        "late": [int(v) for v in P["q2_window_tasks"]],
        "early": [int(v) for v in P["early_tasks"]],
        "all": [1, 10 ** 9],
    }
    with np.load(logdir / f"{arm}_seed{seed}.npz", allow_pickle=False) as z:
        steps = z["step"].astype(np.int64)
        zbar = z[f"layer{layer}_zbar"].astype(np.float64)
        dzbar = z[f"layer{layer}_dzbar"].astype(np.float64)
        p_hat = z[f"layer{layer}_p_hat"].astype(np.float64)

    # 区間 i は記録 i-1 -> i。始点は記録 i-1。
    start_step = steps[:-1]
    adjacent = (steps[1:] - steps[:-1]) == every
    within = (start_step > 0) & (start_step % period != 0) & adjacent
    start_task = start_step // period + 1

    out = {}
    for name, (lo, hi) in windows.items():
        keep = within & (start_task >= lo) & (start_task <= hi)
        idx = np.flatnonzero(keep)
        z0 = zbar[idx]                      # [n_int, h] 始点の zbar
        inc = dzbar[idx + 1]                # [n_int, h] その区間の増分
        sub = p_hat[idx] == 0.0             # 始点で沈下しているユニット
        good = sub & np.isfinite(z0) & np.isfinite(inc)
        x = z0[good]
        y = inc[good]
        rows: list[dict] = []
        beta = float("nan")
        rho = float("nan")
        n_dropped = 0
        if x.size >= nbins * min_count:
            edges = np.quantile(x, np.linspace(0.0, 1.0, nbins + 1))
            edges[0], edges[-1] = -np.inf, np.inf
            which = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, nbins - 1)
            for b in range(nbins):
                sel = which == b
                n = int(sel.sum())
                if not n:
                    continue
                xb, yb = x[sel], y[sel]
                sd = float(yb.std(ddof=0))
                med = float(np.median(yb))
                eligible = bool(n >= min_count and np.isfinite(sd) and sd > 0)
                rows.append(dict(
                    bin=b, n=n, zbar_bin_median=float(np.median(xb)),
                    zbar_bin_lo=float(xb.min()), zbar_bin_hi=float(xb.max()),
                    dzbar_median=med, dzbar_sd=sd,
                    rho=(med / sd if eligible else float("nan")),
                    eligible=int(eligible)))
                if not eligible:
                    n_dropped += n
            fit = [r for r in rows if r["eligible"]]
            if len(fit) >= 2:
                bx = np.array([r["zbar_bin_median"] for r in fit])
                by = np.log(np.array([r["dzbar_sd"] for r in fit]))
                if np.isfinite(by).all() and bx.std() > 0:
                    beta = float(np.polyfit(bx, by, 1)[0])
                neg = [r["rho"] for r in fit if r["zbar_bin_median"] < 0]
                if neg:
                    rho = float(np.median(neg))
        out[name] = dict(bins=rows, beta=beta, rho=rho,
                         n_intervals=int(idx.size),
                         n_unit_intervals=int(good.sum()),
                         n_dropped_unit_intervals=int(n_dropped),
                         submerged_frac_start=(float(sub.mean())
                                               if sub.size else float("nan")))
    return out


def _increment_summary(cfg: dict, outdir: Path, arms: list[str],
                       seeds: list[int]) -> tuple[dict, list[dict]]:
    layer = int(cfg["elu_swamp"]["q2_layer"])
    windows = [str(w) for w in cfg["elu_swamp"]["q2_report_windows"]]
    logdir = outdir / "logs"
    per_arm: dict[str, dict] = {}
    csv_rows: list[dict] = []
    for arm in arms:
        activation = REGISTERED_ARMS[arm][2]
        per_arm[arm] = {w: dict(beta=[], rho=[], submerged_frac_start=[],
                                n_unit_intervals=[], n_dropped=[])
                        for w in windows}
        for seed in seeds:
            result = _interval_rows(cfg, logdir, arm, seed, layer)
            for w in windows:
                r = result[w]
                per_arm[arm][w]["beta"].append(r["beta"])
                per_arm[arm][w]["rho"].append(r["rho"])
                per_arm[arm][w]["submerged_frac_start"].append(r["submerged_frac_start"])
                per_arm[arm][w]["n_unit_intervals"].append(r["n_unit_intervals"])
                per_arm[arm][w]["n_dropped"].append(r["n_dropped_unit_intervals"])
                for row in r["bins"]:
                    csv_rows.append(dict(arm=arm, activation=activation, seed=seed,
                                         window=w, layer=layer,
                                         n_intervals=r["n_intervals"],
                                         n_submerged_start=r["n_unit_intervals"],
                                         beta_seed=r["beta"], rho_seed=r["rho"],
                                         **row))
        for w in windows:
            for key in ("beta", "rho", "submerged_frac_start"):
                per_arm[arm][w][key] = np.asarray(per_arm[arm][w][key],
                                                  dtype=np.float64)
    return per_arm, csv_rows


def _interval(ci: dict, basis: str) -> tuple[float, float]:
    """判定に実際に使った区間を返す（別の区間を見せない）。"""
    if basis == "percentile":
        return float(ci["percentile_ci_lo"]), float(ci["percentile_ci_hi"])
    return float(ci["studentized_ci_lo"]), float(ci["studentized_ci_hi"])


def _excludes(ci: dict, basis: str, value: float, side: str) -> bool:
    lo, hi = _interval(ci, basis)
    return bool(hi < value) if side == "below" else bool(lo > value)


def _contains(ci: dict, basis: str, value: float) -> bool:
    lo, hi = _interval(ci, basis)
    return bool(lo <= value <= hi)


def _basis(ci: dict, censored: bool) -> str:
    if censored:
        return "sign_test"
    return "percentile" if ci["ci_degenerate"] else "studentized"


def _seed_median(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(np.median(values)) if np.isfinite(values).all() else float("nan")


def _q1_level_label(cfg: dict, dose_median: float) -> str:
    P = cfg["elu_swamp"]
    if not np.isfinite(dose_median):
        return "UNDEFINED"
    if dose_median < float(P["q1_level_suppressed_below"]):
        return "MU_SUPPRESSED"
    if dose_median >= float(P["q1_level_regenerated_above"]):
        return "MU_REGENERATED"
    return "MU_INTERMEDIATE"


def _trend_label(ci: dict, basis: str) -> str:
    if _excludes(ci, basis, 0.0, "above"):
        return "INCREASING"
    if _excludes(ci, basis, 0.0, "below"):
        return "DECREASING"
    return "NO_TREND"


def _p2a_label(cfg: dict, ci: dict, basis: str) -> str:
    slope = float(cfg["elu_swamp"]["q2_scaling_expected_slope"])
    if not np.isfinite(ci["point"]):
        return "UNDEFINED"
    if _contains(ci, basis, slope):
        return "MOBILITY_SCALING"
    if _contains(ci, basis, 0.0):
        return "NO_SCALING"
    return "PARTIAL_SCALING"


def _p2b_label(cfg: dict, rho: float) -> str:
    P = cfg["elu_swamp"]
    drift = float(P["q2_drift_ratio_drift_dominated"])
    noise = float(P["q2_drift_ratio_noise_dominated"])
    if not np.isfinite(rho):
        return "UNDEFINED"
    if rho <= -drift:
        return "DRIFT_DOMINATED_DOWNWARD"
    if abs(rho) < noise:
        return "NOISE_DOMINATED"
    if rho >= drift:
        return "DRIFT_DOMINATED_UPWARD"
    return "MIXED"


def _p2c_signature(a1: dict, aall: dict, d_rho: dict, d_rho_basis: str,
                   d_sub: dict, d_sub_basis: str) -> str:
    """spec §5.2c の署名表。登録済みの閾値だけで書く。"""
    if a1["p2b_label"] == "DRIFT_DOMINATED_DOWNWARD" and aall["p2b_label"] == "NOISE_DOMINATED":
        return "MU_DRIVEN"
    if (a1["p2b_label"] == "NOISE_DOMINATED" and aall["p2b_label"] == "NOISE_DOMINATED"
            and a1["p2a_label"] == "MOBILITY_SCALING"
            and aall["p2a_label"] == "MOBILITY_SCALING"):
        return "DIFFUSION_PLUS_MOBILITY"
    if (_excludes(d_sub, d_sub_basis, 0.0, "below")
            and _contains(d_rho, d_rho_basis, 0.0)):
        return "MU_CHANGES_RATE_MECHANISM_DIFFUSION"
    return "MIXED"


def _lever_label(cfg: dict, ratio: float, ci_corr: dict, basis: str) -> str:
    P = cfg["elu_swamp"]
    lo, hi = float(P["q1_lever_ratio_close_lo"]), float(P["q1_lever_ratio_close_hi"])
    if not np.isfinite(ratio):
        return "UNDEFINED"
    if lo <= ratio <= hi:
        return "MU_FROM_PREACT_SCALE"
    if ratio > hi and _excludes(ci_corr, basis, 0.0, "above"):
        return "MU_FROM_SUBMERGENCE"
    return "MU_LEVER_UNRESOLVED"


def _degeneracy(arrays: dict, layer: int, indices: np.ndarray, width: int,
                limit: float) -> dict:
    n_na = np.asarray(arrays["layers"][layer - 1]["n_na"], dtype=np.float64)[indices]
    frac = n_na / float(width)
    return dict(na_frac_late=float(frac.mean()),
                na_frac_late_max_seed=float(frac.mean(axis=0).max()),
                degenerate=int(frac.mean() > limit))


def _floor_fracs(arrays: dict, indices: np.ndarray, floor: float) -> dict:
    unfit = np.asarray(arrays["unfit"], dtype=np.float64)
    return dict(floor_frac_late=float(np.mean(unfit[indices] <= floor)),
                floor_frac_all=float(np.mean(unfit <= floor)))


def _zstar_test(cfg: dict, outdir: Path, arm: str, seeds: list[int]) -> dict:
    """§5.3: 末尾窓の沈下ユニットで Spearman(zbar_i, log v_i^2)。REPORT_ONLY。"""
    P = cfg["elu_swamp"]
    period, layer = int(P["task_period"]), int(P["q2_layer"])
    lo, hi = [int(v) for v in P["late_tasks"]]
    values, n_units, n_zero_v = [], [], 0
    for seed in seeds:
        with np.load(outdir / "logs" / f"{arm}_seed{seed}.npz", allow_pickle=False) as z:
            steps = z["step"].astype(np.int64)
            idx = np.flatnonzero((steps > 0) & (steps % period == 0)
                                 & (steps // period >= lo) & (steps // period <= hi))
            zbar = z[f"layer{layer}_zbar"][idx].astype(np.float64)
            p_hat = z[f"layer{layer}_p_hat"][idx].astype(np.float64)
            where = {int(v): i for i, v in enumerate(z["v_readout_step"])}
            rows = [where[int(steps[i])] for i in idx]
            v = z["v_readout"][rows].astype(np.float64)
        per_record, counts = [], []
        for r in range(zbar.shape[0]):
            sel = (p_hat[r] == 0.0) & (v[r] != 0.0)
            n_zero_v += int(((p_hat[r] == 0.0) & (v[r] == 0.0)).sum())
            if sel.sum() >= 3:
                per_record.append(spearman(zbar[r][sel], np.log(v[r][sel] ** 2)))
                counts.append(int(sel.sum()))
        values.append(float(np.median(per_record)) if per_record else float("nan"))
        n_units.append(float(np.median(counts)) if counts else float("nan"))
    return dict(seed_values=values, median_submerged_units=n_units,
                n_zero_readout_excluded=n_zero_v)


def analyze(cfg: dict, outdir: Path, sanity: dict, elapsed: dict) -> dict:
    P = cfg["elu_swamp"]
    seeds = [int(v) for v in cfg["common"]["seeds"]]
    period, floor = int(P["task_period"]), float(P["unfit_floor"])
    width = int(_arm(cfg, "E_A1")["hidden"][-1])
    q1_layer, q2_layer = int(P["q1_layer"]), int(P["q2_layer"])
    lever_layer = int(P["q1_lever_source_layer"])
    na_limit, censor_limit = float(P["na_frac_max"]), float(P["censor_frac_max"])
    rng = np.random.default_rng(int(P["bootstrap_seed"]))
    draws = rng.integers(0, len(seeds), size=(int(P["bootstrap_B"]), len(seeds)))
    logdir = outdir / "logs"

    divergences: dict[str, dict] = sanity.get("numeric_divergence") or {}
    completed = [arm for arm in ARM_ORDER if arm not in divergences]
    data = {arm: _arm_arrays(logdir, arm, seeds,
                             len(_arm(cfg, arm)["hidden"]), period)
            for arm in completed}
    pair_ok = bool((sanity.get("S_pair") or {}).get("pass_")
                   and (sanity.get("S_pair_final") or {}).get("paired_pass"))

    verdict_rows: list[dict] = []
    details: dict = {"numeric_divergence": divergences, "completed_arms": completed,
                     "levels": [], "contrasts": [], "q1": {}, "q2": {},
                     "g0": {}, "zstar": [], "pair_ok": int(pair_ok)}

    # ---- 腕ごとの水準（末尾窓） ----
    per_arm: dict[str, dict] = {}
    for arm in completed:
        a = data[arm]
        steps = np.asarray(a["steps"])
        early_i = _window_indices(steps, period, list(P["early_tasks"]))
        late_i = _window_indices(steps, period, list(P["late_tasks"]))
        trend_i = _window_indices(steps, period, list(P["q1_trend_range_tasks"]))
        task = steps / period
        unfit_late = np.maximum(np.asarray(a["unfit"])[late_i].mean(axis=0), floor)
        unfit_early = np.maximum(np.asarray(a["unfit"])[early_i].mean(axis=0), floor)
        dose = np.asarray(a["layers"][q1_layer - 1]["dose"], dtype=np.float64)
        dose_late = dose[late_i].mean(axis=0)
        dose_early = dose[early_i].mean(axis=0)
        dose_rho = np.array([spearman(task[trend_i], dose[trend_i][:, s])
                             for s in range(len(seeds))])
        sub = (np.asarray(a["layers"][q2_layer - 1]["submerged"], dtype=np.float64)
               / width)
        sub_late = sub[late_i].mean(axis=0)
        lever_sub = (np.asarray(a["layers"][lever_layer - 1]["submerged"],
                                dtype=np.float64) / width)
        s_pre = np.asarray(a["layers"][lever_layer - 1]["preact_sd_median"],
                           dtype=np.float64)
        reference = dose_reference(torch.as_tensor(s_pre), width).numpy()
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.where(reference > 0, dose / reference, np.nan)
            excess = dose - reference
        ratio_late = np.array([np.nanmedian(ratio[late_i][:, s])
                               for s in range(len(seeds))])
        lever_rho = np.array([spearman(excess[trend_i][:, s],
                                       lever_sub[trend_i][:, s])
                              for s in range(len(seeds))])
        per_arm[arm] = dict(
            activation=REGISTERED_ARMS[arm][2], early_i=early_i, late_i=late_i,
            unfit_late=unfit_late, unfit_early=unfit_early,
            log_unfit_late=np.log10(unfit_late), log_unfit_early=np.log10(unfit_early),
            dose_late=dose_late, dose_early=dose_early, dose_rho=dose_rho,
            submerged_late=sub_late, ratio_late=ratio_late, lever_rho=lever_rho,
            preact_sd_late=s_pre[late_i].mean(axis=0),
            reference_late=reference[late_i].mean(axis=0),
            **_floor_fracs(a, late_i, floor),
            **_degeneracy(a, q1_layer, late_i, width, na_limit))
        details["levels"].append(dict(
            arm=arm, activation=per_arm[arm]["activation"], window="late",
            unfit_seed_values=unfit_late.tolist(),
            unfit_median=float(np.median(unfit_late)),
            unfit_min=float(unfit_late.min()), unfit_max=float(unfit_late.max()),
            dose_layer=q1_layer, dose_median=float(np.median(dose_late)),
            dose_early_median=float(np.median(dose_early)),
            submerged_frac_median=(float(np.median(sub_late))
                                   if per_arm[arm]["activation"] == "elu" else ""),
            strict_dead_median=(float(np.median(sub_late)) * width
                                if per_arm[arm]["activation"] == "relu" else ""),
            preact_sd_median=float(np.median(per_arm[arm]["preact_sd_late"])),
            dose_reference_median=float(np.median(per_arm[arm]["reference_late"])),
            floor_frac_late=per_arm[arm]["floor_frac_late"],
            na_frac_late=per_arm[arm]["na_frac_late"],
            degenerate=per_arm[arm]["degenerate"]))
        for li in range(1, len(_arm(cfg, arm)["hidden"]) + 1):
            layer = a["layers"][li - 1]
            row = dict(arm=arm, activation=per_arm[arm]["activation"], layer=li,
                       centered=int(_centered_flags(_arm(cfg, arm),
                                                    len(_arm(cfg, arm)["hidden"]))[li - 1]),
                       window="late")
            for key in ("eff_rank", "eff_rank_W", "stable_rank_W", "top1_frac",
                        "wcos_mean", "sign_match_mean", "sign_clone_frac",
                        "w_norm_median", "dose", "mu_norm", "sigma_rms",
                        "preact_sd_median"):
                values = np.asarray(layer[key], dtype=np.float64)[late_i].mean(axis=0)
                row[key] = float(np.median(values))
            counted = np.asarray(layer["submerged"], dtype=np.float64)[late_i].mean(axis=0)
            # §4.4: ReLU 腕は strict_dead、ELU 腕は submerged_frac。列を混ぜない。
            row["strict_dead"] = (float(np.median(counted))
                                  if per_arm[arm]["activation"] == "relu" else "")
            row["submerged_frac"] = (float(np.median(counted)) / width
                                     if per_arm[arm]["activation"] == "elu" else "")
            row["alive"] = (float(width - np.median(counted))
                            if per_arm[arm]["activation"] == "relu" else "")
            details["levels"].append(row)

    # ---- 増分（Q2）----
    increments, increment_rows = _increment_summary(cfg, outdir, completed, seeds)
    details["increment_windows"] = [str(w) for w in P["q2_report_windows"]]

    def ci_of(values, censored=False):
        """登録 seed のどれかで統計が未定義なら CI を作らない。

        部分集合で bootstrap を回すと「10 seed の中央値」という登録された統計とは
        別のものを同じ名前で報告することになるので、そこは黙って埋めずに落とす。
        未定義 seed 数は呼び出し側が verdict.csv に出す。
        """
        values = np.asarray(values, dtype=np.float64)
        if not np.isfinite(values).all():
            return None, "insufficient_data"
        ci = _ci(cfg, values, draws)
        return ci, _basis(ci, censored)

    def n_undefined(*arrays) -> int:
        return int(max(int((~np.isfinite(np.asarray(a, dtype=np.float64))).sum())
                       for a in arrays))

    # ---- G0（§5.0）----
    g0_arm, g0_ref = str(P["g0_arm"]), str(P["g0_relu_reference"])
    if g0_arm in per_arm and g0_ref in per_arm:
        censored = bool(max(per_arm[g0_arm]["floor_frac_late"],
                            per_arm[g0_ref]["floor_frac_late"]) > censor_limit)
        delta = per_arm[g0_arm]["log_unfit_late"] - per_arm[g0_ref]["log_unfit_late"]
        ci_g0, basis_g0 = ci_of(delta, censored)
        improved, basis_used = _decide(ci_g0, "down", censored, paired=True)
        g0a = float(np.median(per_arm[g0_arm]["unfit_late"]))
        g0b = float(np.median(per_arm[g0_arm]["submerged_late"]))
        if g0a >= float(P["gate_unfit_threshold"]):
            label = "LOP_PRESENT"
        elif improved and pair_ok:
            label = ("ABSORPTION_ISOLATED"
                     if g0b >= float(P["g0b_submerged_threshold"]) else "LOP_REDUCED")
        else:
            label = "INCONCLUSIVE"
        details["g0"] = dict(
            arm=g0_arm, reference=g0_ref, label=label, g0a_unfit_median=g0a,
            g0a_seed_values=per_arm[g0_arm]["unfit_late"].tolist(),
            g0b_submerged_median=g0b,
            g0b_seed_values=per_arm[g0_arm]["submerged_late"].tolist(),
            g0b_threshold=float(P["g0b_submerged_threshold"]),
            g0b_threshold_confirmed=bool(P.get("g0b_threshold_confirmed")),
            paired_delta_log10_unfit=ci_g0, decision_basis=basis_used,
            improved=int(improved), censored=int(censored), pair_ok=int(pair_ok))
        verdict_rows.append(dict(
            metric="G0_paired_delta_log10_unfit", arm=g0_arm, baseline=g0_ref,
            activation="elu", contrast_type="g0", pairing="paired",
            verdict=label, decision_basis=basis_used, improved=int(improved),
            censored=int(censored), pair_ok=int(pair_ok),
            g0a_unfit_median=g0a, g0b_submerged_median=g0b,
            floor_frac_late_arm=per_arm[g0_arm]["floor_frac_late"],
            floor_frac_late_baseline=per_arm[g0_ref]["floor_frac_late"],
            na_frac_late_arm=per_arm[g0_arm]["na_frac_late"],
            degenerate=per_arm[g0_arm]["degenerate"],
            seed_values=json.dumps(delta.tolist()), **ci_g0))

    # ---- Q1（§5.1）----
    q1_arm = g0_arm
    if q1_arm in per_arm:
        a = per_arm[q1_arm]
        level = float(np.median(a["dose_late"]))
        ci_trend, basis_trend = ci_of(a["dose_rho"])
        ci_ratio, basis_ratio = ci_of(a["ratio_late"])
        ci_lever, basis_lever = ci_of(a["lever_rho"])
        p1a = _q1_level_label(cfg, level)
        p1b = _trend_label(ci_trend, basis_trend)
        lever = _lever_label(cfg, float(np.median(a["ratio_late"])),
                             ci_lever, basis_lever)
        if a["degenerate"]:
            p1a = p1b = "DEGENERATE"
        details["q1"] = dict(
            arm=q1_arm, layer=q1_layer, p1a_label=p1a, p1a_dose_median=level,
            p1a_seed_values=a["dose_late"].tolist(),
            p1a_early_median=float(np.median(a["dose_early"])),
            p1b_label=p1b, p1b_trend=ci_trend, p1b_basis=basis_trend,
            p1c_label=lever, p1c_ratio=ci_ratio, p1c_ratio_basis=basis_ratio,
            p1c_excess_vs_submergence=ci_lever, p1c_basis=basis_lever,
            p1c_preact_sd_median=float(np.median(a["preact_sd_late"])),
            p1c_reference_median=float(np.median(a["reference_late"])),
            degenerate=a["degenerate"], na_frac_late=a["na_frac_late"])
        verdict_rows.append(dict(
            metric="P1a_dose_level_late", arm=q1_arm, baseline="", layer=q1_layer,
            activation="elu", contrast_type="q1", pairing="single_arm",
            verdict=p1a, decision_basis="seed_median_vs_frozen_thresholds",
            point=level, seed_values=json.dumps(a["dose_late"].tolist()),
            na_frac_late_arm=a["na_frac_late"], degenerate=a["degenerate"],
            floor_frac_late_arm=a["floor_frac_late"]))
        verdict_rows.append(dict(
            metric="P1b_dose_trend_spearman", arm=q1_arm, baseline="", layer=q1_layer,
            activation="elu", contrast_type="q1", pairing="single_arm",
            verdict=p1b, decision_basis=basis_trend,
            seed_values=json.dumps(a["dose_rho"].tolist()),
            na_frac_late_arm=a["na_frac_late"], degenerate=a["degenerate"],
            **ci_trend))
        verdict_rows.append(dict(
            metric="P1c_dose_over_closed_form", arm=q1_arm, baseline="",
            layer=q1_layer, activation="elu", contrast_type="q1_lever",
            pairing="single_arm", verdict="REPORT_ONLY", lever_label=lever,
            decision_basis=basis_ratio,
            seed_values=json.dumps(a["ratio_late"].tolist()), **ci_ratio))
        verdict_rows.append(dict(
            metric="P1c_excess_vs_layer1_submergence", arm=q1_arm, baseline="",
            layer=q1_layer, activation="elu", contrast_type="q1_lever",
            pairing="single_arm", verdict="REPORT_ONLY", lever_label=lever,
            decision_basis=basis_lever,
            seed_values=json.dumps(a["lever_rho"].tolist()), **ci_lever))

    # ---- Q2（§5.2）----
    primary_window = "late"
    q2_labels: dict[str, dict] = {}
    for arm in completed:
        for window in details["increment_windows"]:
            stats = increments[arm][window]
            ci_beta, basis_beta = ci_of(stats["beta"])
            ci_rho, basis_rho = ci_of(stats["rho"])
            p2a = _p2a_label(cfg, ci_beta, basis_beta) if ci_beta else "UNDEFINED"
            p2b = _p2b_label(cfg, _seed_median(stats["rho"]))
            registered = bool(arm in ELU_ARMS and window == primary_window)
            undefined = n_undefined(stats["beta"], stats["rho"])
            if undefined:
                p2a = p2a if ci_beta else "INSUFFICIENT_DATA"
                p2b = p2b if np.isfinite(_seed_median(stats["rho"])) else "INSUFFICIENT_DATA"
            if window == primary_window:
                q2_labels[arm] = dict(p2a_label=p2a, p2b_label=p2b,
                                      beta=ci_beta, beta_basis=basis_beta,
                                      rho=ci_rho, rho_basis=basis_rho,
                                      n_undefined_seeds=undefined, stats=stats)
            details["q2"].setdefault(arm, {})[window] = dict(
                p2a_label=p2a, p2b_label=p2b, registered=int(registered),
                beta_seed_values=stats["beta"].tolist(),
                rho_seed_values=stats["rho"].tolist(),
                beta_ci=ci_beta, rho_ci=ci_rho, n_undefined_seeds=undefined,
                beta_basis=basis_beta, rho_basis=basis_rho,
                n_unit_intervals=[int(v) for v in stats["n_unit_intervals"]],
                n_dropped_unit_intervals=[int(v) for v in stats["n_dropped"]],
                submerged_frac_start=stats["submerged_frac_start"].tolist())
            common = dict(arm=arm, baseline="", layer=q2_layer,
                          activation=REGISTERED_ARMS[arm][2],
                          contrast_type="q2", pairing="single_arm", window=window,
                          registered_endpoint=int(registered),
                          n_undefined_seeds=undefined,
                          n_unit_intervals_median=float(
                              np.median(stats["n_unit_intervals"])),
                          n_dropped_unit_intervals_median=float(
                              np.median(stats["n_dropped"])))
            verdict_rows.append(dict(
                metric="P2a_mobility_slope", verdict=(p2a if registered else "REPORT_ONLY"),
                label=p2a, decision_basis=basis_beta,
                seed_values=json.dumps(stats["beta"].tolist()), **common,
                **(ci_beta or {})))
            verdict_rows.append(dict(
                metric="P2b_drift_to_noise", verdict=(p2b if registered else "REPORT_ONLY"),
                label=p2b, decision_basis=basis_rho,
                seed_values=json.dumps(stats["rho"].tolist()), **common,
                **(ci_rho or {})))

    # ---- P2c（§5.2c）----
    a1, aall = [str(v) for v in P["q2_intervention_contrast"]]
    if a1 in q2_labels and aall in q2_labels:
        d_rho = q2_labels[aall]["stats"]["rho"] - q2_labels[a1]["stats"]["rho"]
        d_beta = q2_labels[aall]["stats"]["beta"] - q2_labels[a1]["stats"]["beta"]
        d_sub = (per_arm[aall]["submerged_late"] - per_arm[a1]["submerged_late"])
        ci_drho, basis_drho = ci_of(d_rho)
        ci_dbeta, basis_dbeta = ci_of(d_beta)
        ci_dsub, basis_dsub = ci_of(d_sub)
        signature = (_p2c_signature(q2_labels[a1], q2_labels[aall], ci_drho,
                                    basis_drho, ci_dsub, basis_dsub)
                     if ci_drho and ci_dsub else "UNDEFINED")
        p2c_undefined = n_undefined(d_rho, d_beta, d_sub)
        if p2c_undefined and not (ci_drho and ci_dsub):
            signature = "INSUFFICIENT_DATA"
        details["q2"]["P2c"] = dict(
            arm=aall, baseline=a1, signature=signature,
            n_undefined_seeds=p2c_undefined,
            delta_rho=ci_drho, delta_rho_basis=basis_drho,
            delta_beta=ci_dbeta, delta_beta_basis=basis_dbeta,
            delta_submerged_frac=ci_dsub, delta_submerged_frac_basis=basis_dsub,
            arm_labels={a1: {k: q2_labels[a1][k] for k in ("p2a_label", "p2b_label")},
                        aall: {k: q2_labels[aall][k] for k in ("p2a_label", "p2b_label")}},
            pair_ok=int(pair_ok))
        for metric, ci, basis, values in (
                ("P2c_delta_rho", ci_drho, basis_drho, d_rho),
                ("P2c_delta_beta", ci_dbeta, basis_dbeta, d_beta),
                ("P2c_delta_submerged_frac", ci_dsub, basis_dsub, d_sub)):
            verdict_rows.append(dict(
                metric=metric, arm=aall, baseline=a1, layer=q2_layer,
                activation="elu", contrast_type="p2c_intervention",
                pairing="paired", window=primary_window,
                verdict=signature, decision_basis=basis, pair_ok=int(pair_ok),
                n_undefined_seeds=p2c_undefined,
                seed_values=json.dumps(np.asarray(values).tolist()),
                **(ci or {})))

    # ---- 事前登録の対比（REPORT_ONLY を含む）----
    for arm, base, ctype in CONTRASTS:
        involved = {n: divergences[n] for n in (arm, base) if n in divergences}
        if involved:
            detected = min(int(v["detected_step"]) for v in involved.values())
            details["contrasts"].append(dict(
                arm=arm, baseline=base, contrast_type=ctype,
                verdict=NUMERIC_DIVERGENCE, divergent_arms=sorted(involved),
                divergence_detected_step=detected))
            verdict_rows.append(dict(
                metric="C_delta_log10_unfit_late", arm=arm, baseline=base,
                contrast_type=ctype, pairing="paired", verdict=NUMERIC_DIVERGENCE,
                decision_basis="numeric_divergence", numeric_divergence=1,
                divergent_arms=json.dumps(sorted(involved)),
                divergence_detected_step=detected))
            continue
        censored = bool(max(per_arm[arm]["floor_frac_late"],
                            per_arm[base]["floor_frac_late"]) > censor_limit)
        rows = []
        for metric, values, direction in (
                ("C_delta_log10_unfit_late",
                 per_arm[arm]["log_unfit_late"] - per_arm[base]["log_unfit_late"], "down"),
                ("C_delta_dose_late",
                 per_arm[arm]["dose_late"] - per_arm[base]["dose_late"], "up"),
                ("C_delta_dose_trend",
                 per_arm[arm]["dose_rho"] - per_arm[base]["dose_rho"], "up")):
            ci, basis = ci_of(values, censored if metric.endswith("unfit_late") else False)
            improved, basis_used = _decide(
                ci, direction, censored if metric.endswith("unfit_late") else False,
                paired=True)
            rows.append(dict(metric=metric, ci=ci, basis=basis_used,
                             improved=int(improved), values=values))
            verdict_rows.append(dict(
                metric=metric, arm=arm, baseline=base, layer=q1_layer,
                activation=f"{REGISTERED_ARMS[arm][2]}_vs_{REGISTERED_ARMS[base][2]}",
                contrast_type=ctype, pairing="paired", verdict="REPORT_ONLY",
                decision_basis=basis_used, improved=int(improved),
                censored=int(censored), pair_ok=int(pair_ok),
                floor_frac_late_arm=per_arm[arm]["floor_frac_late"],
                floor_frac_late_baseline=per_arm[base]["floor_frac_late"],
                na_frac_late_arm=per_arm[arm]["na_frac_late"],
                degenerate=int(per_arm[arm]["degenerate"] or per_arm[base]["degenerate"]),
                seed_values=json.dumps(np.asarray(values).tolist()), **ci))
        details["contrasts"].append(dict(
            arm=arm, baseline=base, contrast_type=ctype, censored=int(censored),
            rows=[{k: v for k, v in r.items() if k != "values"} for r in rows]))

    # ---- §5.3 z* ----
    for arm in completed:
        if REGISTERED_ARMS[arm][2] != "elu":
            continue
        zs = _zstar_test(cfg, outdir, arm, seeds)
        ci_z, basis_z = ci_of(np.asarray(zs["seed_values"]))
        details["zstar"].append(dict(arm=arm, **zs, ci=ci_z, basis=basis_z))
        verdict_rows.append(dict(
            metric="S53_spearman_zbar_vs_log_v2", arm=arm, baseline="",
            layer=q2_layer, activation="elu", contrast_type="secondary",
            pairing="single_arm", verdict="REPORT_ONLY", decision_basis=basis_z,
            seed_values=json.dumps(zs["seed_values"]), **(ci_z or {})))

    fields: list[str] = []
    for row in verdict_rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    write_csv(outdir / "verdict.csv",
              [{key: row.get(key, "") for key in fields} for row in verdict_rows])
    write_csv(outdir / "increments.csv", increment_rows)
    write_csv(outdir / "layer_stats.csv",
              _task_rows_from_logs(cfg, outdir, seeds, completed))
    _write_summary(cfg, outdir, details, sanity)
    details["elapsed_sec"] = elapsed
    return details


def _task_rows_from_logs(cfg: dict, outdir: Path, seeds: list[int],
                         arms: list[str]) -> list[dict]:
    period = int(cfg["elu_swamp"]["task_period"])
    rows = []
    for arm in arms:
        arm_cfg = _arm(cfg, arm)
        hidden = [int(v) for v in arm_cfg["hidden"]]
        flags = _centered_flags(arm_cfg, len(hidden))
        activation = REGISTERED_ARMS[arm][2]
        for seed in seeds:
            with np.load(outdir / "logs" / f"{arm}_seed{seed}.npz",
                         allow_pickle=False) as z:
                idx = np.flatnonzero((z["step"] > 0) & (z["step"] % period == 0))
                for i in idx:
                    for li, w in enumerate(hidden, start=1):
                        row = dict(arm=arm, activation=activation,
                                   run_id=str(z["run_id"]), seed=int(seed),
                                   step=int(z["step"][i]),
                                   task=int(z["step"][i] // period), task_end=1,
                                   layer=li, centered=int(flags[li - 1]))
                        for key in ELU_LOG_LAYER_KEYS:
                            value = z[f"layer{li}_{key}"][i]
                            row[key] = (int(value) if key in
                                        ("n_na", "strict_dead", "alive", "submerged")
                                        else float(value))
                        # §4.4: 列を混ぜない。
                        row["strict_dead_frac"] = (row["strict_dead"] / w
                                                   if activation == "relu" else "")
                        row["submerged_frac"] = (row["submerged"] / w
                                                 if activation == "elu" else "")
                        if activation == "elu":
                            row["strict_dead"] = ""
                            row["alive"] = ""
                        else:
                            row["submerged"] = ""
                        for key in ("signal_var", "residual_var", "unfit",
                                    "eval_loss_exact"):
                            row[key] = float(z[key][i])
                        rows.append(row)
    return rows


def _write_summary(cfg: dict, outdir: Path, details: dict, sanity: dict) -> None:
    P = cfg["elu_swamp"]
    def fmt(value, spec=".6g") -> str:
        if value is None or value == "":
            return "—"
        try:
            return format(float(value), spec)
        except (TypeError, ValueError):
            return str(value)

    def band(ci: dict | None, basis: str) -> str:
        if not ci:
            return "—（登録 seed のどれかで統計が定義できなかった）"
        if basis == "sign_test":
            return f"符号検定 p={fmt(ci.get('sign_test_p'), '.4g')}"
        lo, hi = _interval(ci, basis)
        return f"[{fmt(lo)}, {fmt(hi)}]"

    def est(ci: dict | None, basis: str) -> str:
        """点推定 + 判定に使った区間。CI が立たない場合は理由ごと出す。"""
        if not ci:
            return band(ci, basis)
        return f"{fmt(ci['point'])} {band(ci, basis)}"

    lines = [f"# {EXPERIMENT} summary", "",
             "ELU の沼。Q1 = 第2層 dose は抑えられるか再生されるか、",
             "Q2 = 沈下は µ 駆動か拡散＋可動度か。**共主表現型 G0a（LoP）と G0b（沈下）は",
             "どちらもゲートではない**（spec §5.0・2026-08-30 改訂）。", "",
             "**ELU 腕と ReLU 腕のペアリングは init・教師・入力実現までで、同一個体の追跡ではない**",
             "（spec §4.2）。step 1 以降で軌道は分岐している。", ""]

    g0 = details.get("g0") or {}
    if g0:
        lines += ["## G0 — 表現型の 2 本立て（§5.0）", "",
                  "| 腕 | G0a 未フィット率 median | G0b 沈下率 median | "
                  f"paired Δlog10 U vs {g0['reference']} | 判定基底 | ラベル |",
                  "|---|---:|---:|---:|---|---|",
                  f"| {g0['arm']} | {fmt(g0['g0a_unfit_median'])} | "
                  f"{fmt(g0['g0b_submerged_median'])} | "
                  f"{est(g0['paired_delta_log10_unfit'], g0['decision_basis'])} | "
                  f"{g0['decision_basis']} | **{g0['label']}** |", "",
                  f"- G0a 全 seed: {[fmt(v) for v in g0['g0a_seed_values']]}",
                  f"- G0b 全 seed: {[fmt(v) for v in g0['g0b_seed_values']]}",
                  f"- G0b 閾値 {fmt(g0['g0b_threshold'])}（Issa 確認済み: "
                  f"{g0['g0b_threshold_confirmed']}）",
                  "- **`submerged_frac` は verdict に使わない。沈下は死ではない**（§4.4）。"
                  "G0b は表現型の記述であって機能損失の代理ではない", ""]

    q1 = details.get("q1") or {}
    if q1:
        lines += ["## Q1 — 第2層 dose（§5.1・主 endpoint）", "",
                  "| endpoint | 値 | 区間 | 判定基底 | ラベル |",
                  "|---|---:|---:|---|---|",
                  f"| P1a 水準（末尾窓 seed 中央値） | {fmt(q1['p1a_dose_median'])} | — | "
                  f"seed 中央値 vs 凍結閾値 | **{q1['p1a_label']}** |",
                  f"| P1b 傾き Spearman(task, dose) | "
                  f"{fmt((q1['p1b_trend'] or {}).get('point'))} | "
                  f"{band(q1['p1b_trend'], q1['p1b_basis'])} | {q1['p1b_basis']} | "
                  f"**{q1['p1b_label']}** |",
                  f"| P1c 実測/10·g(s) | {fmt((q1['p1c_ratio'] or {}).get('point'))} | "
                  f"{band(q1['p1c_ratio'], q1['p1c_ratio_basis'])} | "
                  f"{q1['p1c_ratio_basis']} | {q1['p1c_label']}（REPORT_ONLY） |",
                  f"| P1c 超過 vs 第1層沈下率 Spearman | "
                  f"{fmt((q1['p1c_excess_vs_submergence'] or {}).get('point'))} | "
                  f"{band(q1['p1c_excess_vs_submergence'], q1['p1c_basis'])} | "
                  f"{q1['p1c_basis']} | REPORT_ONLY |", "",
                  f"- 早期窓 dose median {fmt(q1['p1a_early_median'])}、"
                  f"第1層 sd(z) median {fmt(q1['p1c_preact_sd_median'])}、"
                  f"閉形式参照 10·g(s) = {fmt(q1['p1c_reference_median'])}",
                  f"- 閾値: < {fmt(P['q1_level_suppressed_below'])} = MU_SUPPRESSED、"
                  f">= {fmt(P['q1_level_regenerated_above'])} = MU_REGENERATED（凍結）",
                  f"- `Sigma^(2)` 退化: NA 割合（末尾窓）{fmt(q1['na_frac_late'])}、"
                  f"DEGENERATE={q1['degenerate']}",
                  "- **P1c は P1a/P1b の verdict を上書きしない**（§5.1c）", ""]

    q2 = details.get("q2") or {}
    if q2:
        w_lo, w_hi = [int(v) for v in P["q2_window_tasks"]]
        per_task = int(P["task_period"]) // int(P["q2_increment_interval_steps"]) - 1
        lines += ["## Q2 — 沈下は駆動か拡散か（§5.2・主 endpoint）", "",
                  f"窓 = 末尾窓（タスク {w_lo}–{w_hi}）、区間 = タスク内隣接記録点"
                  f"（1 タスク {per_task} 区間）、始点で条件付け、"
                  f"等頻度 {int(P['q2_bins'])} 分位、自然対数（spec §11.2–11.6）。", "",
                  "| 腕 | 活性化 | P2a 傾き β | 区間 | P2a | P2b ρ | 区間 | P2b |",
                  "|---|---|---:|---:|---|---:|---:|---|"]
        for arm in details.get("completed_arms", []):
            row = (q2.get(arm) or {}).get("late")
            if not row:
                continue
            mark = "" if arm in ELU_ARMS else "（REPORT_ONLY）"
            undefined = int(row.get("n_undefined_seeds", 0))
            note = f"（{undefined} seed 未定義）" if undefined else ""
            lines.append(
                f"| {arm} | {REGISTERED_ARMS[arm][2]} | "
                f"{fmt((row['beta_ci'] or {}).get('point'))} | "
                f"{band(row['beta_ci'], row.get('beta_basis', ''))} | "
                f"{row['p2a_label']}{mark}{note} | "
                f"{fmt((row['rho_ci'] or {}).get('point'))} | "
                f"{band(row['rho_ci'], row.get('rho_basis', ''))} | "
                f"{row['p2b_label']}{mark}{note} |")
        p2c = q2.get("P2c")
        if p2c:
            lines += ["", "### P2c — µ 駆動の介入判別（§5.2c・Q2 の本命）", "",
                      "| 対比 | Δρ | Δβ | Δ沈下率 | 署名 |",
                      "|---|---:|---:|---:|---|",
                      f"| {p2c['arm']} − {p2c['baseline']} | "
                      f"{est(p2c['delta_rho'], p2c['delta_rho_basis'])} | "
                      f"{est(p2c['delta_beta'], p2c['delta_beta_basis'])} | "
                      f"{est(p2c['delta_submerged_frac'], p2c['delta_submerged_frac_basis'])} | "
                      f"**{p2c['signature']}** |", "",
                      "**P2c は観察的な条件付けではなく介入**なので、層別後の関連を因果と呼ぶ"
                      "禁則には抵触しない（§5.2c）。", ""]
        lines += ["", "早期窓・全区間の β / ρ は `verdict.csv`（`window` 列）と "
                  "`increments.csv` に入っている。ビンごとの生値も `increments.csv` に"
                  "全部残してあるので、別の縮約は事後に再計算できる。", ""]

    lines += ["## 事前登録の対比（REPORT_ONLY・§5.3）", "",
              "| 腕 | baseline | Δlog10 U（末尾窓） | Δdose（第2層） | Δdose 傾き | 検閲 |",
              "|---|---|---:|---:|---:|---:|"]
    for c in details.get("contrasts", []):
        if c.get("verdict") == NUMERIC_DIVERGENCE:
            lines.append(f"| {c['arm']} | {c['baseline']} | — | — | — | "
                         f"**{NUMERIC_DIVERGENCE}** |")
            continue
        byname = {r["metric"]: r for r in c["rows"]}
        def cell(metric: str) -> str:
            r = byname.get(metric)
            return est(r["ci"], r["basis"]) if r else "—"
        lines.append(f"| {c['arm']} | {c['baseline']} | "
                     f"{cell('C_delta_log10_unfit_late')} | "
                     f"{cell('C_delta_dose_late')} | "
                     f"{cell('C_delta_dose_trend')} | {c['censored']} |")

    lines += ["", "## 水準（末尾窓・REPORT_ONLY）", "",
              "| 腕 | 活性化 | 未フィット率 median [min, max] | 第2層 dose | "
              "第1層 sd(z) | 10·g(s) | 床割合 | NA 割合 |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in details.get("levels", []):
        if row.get("window") != "late" or "unfit_median" not in row:
            continue
        lines.append(
            f"| {row['arm']} | {row['activation']} | {fmt(row['unfit_median'])} "
            f"[{fmt(row['unfit_min'])}, {fmt(row['unfit_max'])}] | "
            f"{fmt(row['dose_median'])} | {fmt(row['preact_sd_median'])} | "
            f"{fmt(row['dose_reference_median'])} | {fmt(row['floor_frac_late'])} | "
            f"{fmt(row['na_frac_late'])} |")

    lines += ["", "**§4.4: `strict_dead` は ReLU 腕のみ、`submerged_frac` は ELU 腕のみ。"
              "同じ列に混ぜない。**", "",
              "| 腕 | 活性化 | layer | centered | eff_rank | strict_dead | submerged_frac | "
              "dose | ||w|| median |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in details.get("levels", []):
        if "layer" not in row or "eff_rank" not in row:
            continue
        lines.append(
            f"| {row['arm']} | {row['activation']} | {row['layer']} | {row['centered']} | "
            f"{fmt(row['eff_rank'])} | {fmt(row['strict_dead'])} | "
            f"{fmt(row['submerged_frac'])} | {fmt(row['dose'])} | "
            f"{fmt(row['w_norm_median'])} |")

    zstar = details.get("zstar") or []
    if zstar:
        lines += ["", "## 副次: `z* ∝ log v_i^2`（§5.3・REPORT_ONLY）", "",
                  "**`v_i` は沈下しても学習され続けるので観察的関連であり、因果ではない。**", "",
                  "| 腕 | Spearman(zbar, log v²) seed 中央値 | 区間 | 沈下ユニット数 median |",
                  "|---|---:|---:|---:|"]
        for row in zstar:
            units = np.asarray(row["median_submerged_units"], dtype=np.float64)
            lines.append(f"| {row['arm']} | {fmt((row['ci'] or {}).get('point'))} | "
                         f"{band(row['ci'], row['basis'])} | "
                         f"{fmt(np.nanmedian(units) if units.size else float('nan'))} |")

    divergences = details.get("numeric_divergence") or {}
    if divergences:
        lines += ["", "## 数値発散（§5.4）", "", "| 腕 | detected step | task | seeds | 扱い |",
                  "|---|---:|---:|---|---|"]
        for arm, event in divergences.items():
            seeds = ", ".join(str(v) for v in event.get("bad_seeds", []))
            lines.append(f"| {arm} | {event.get('detected_step', '')} | "
                         f"{event.get('detected_task', '')} | {seeds} | "
                         f"**{NUMERIC_DIVERGENCE}**（停止・救済なし） |")

    def mark(node) -> str:
        return "**PASS**" if node and node.get("pass_") else "**FAIL**"

    lines += ["", "## Sanity（§6）", "",
              f"- S0'（R_none/R_A1 == phase1 L2_none/L2_A1）: {mark(sanity.get('S0prime'))}",
              f"- S-grad（ELU 閉形式勾配 vs 独立実装・有限差分・網全体）: {mark(sanity.get('S_grad'))}",
              f"- S-elu-limit（alpha->0 で ReLU 経路に一致）: {mark(sanity.get('S_elu_limit'))}",
              f"- S-copy（厳密記録の fork 検査・ReLU 腕）: {mark(sanity.get('S_copy'))}",
              f"- S-submerge（amax 沈下 == strict_dead）: {mark(sanity.get('S_submerge'))}",
              f"- S-pair（5 腕の対応づけ）: {mark(sanity.get('S_pair'))}",
              f"- S-pair-final（5M 後の env 一致）: {mark(sanity.get('S_pair_final'))}",
              f"- S-taut（A を入れた層の µ 項）: {mark(sanity.get('S_taut'))}",
              f"- S-g(s)（spec §2.1 の表を再現）: {mark(sanity.get('S_g_elu_table'))}",
              f"- S1/S2（完走腕の 32 パターン厳密恒等式）: "
              f"{'**PASS**' if sanity.get('S1_S2_completed_arms_pass') else '**FAIL**'}",
              f"- S3（OMP_NUM_THREADS=1）: {mark(sanity.get('S3'))}",
              f"- S5（退化ガード自己検査）: {mark(sanity.get('S5'))}",
              f"- S7（数値発散検出器）: {mark(sanity.get('S7_numeric_divergence'))}",
              f"- NUMERIC_DIVERGENCE 腕: "
              f"{', '.join(sorted(divergences)) if divergences else 'なし'}",
              f"- 床（phase1 較正値・再較正なし）: {fmt(P['unfit_floor'], '.1e')}", "",
              "## 引用上の注意（§9）", "",
              "- ELU 腕の結果で ReLU 側の柱1・柱2・柱3 を書き換えない。ELU は新しいスコープ",
              "- 柱2 の「片道の壁・不可逆吸収」を ELU に literal に適用しない（§2.4）",
              "- §2.1 の閉形式・§2.3 の滞在密度と平衡深さは導出であり、本走の測定対象。前提として引かない",
              "- 中心化を入れた層の `M` / `D` は恒真。verdict にも機構の主張にも使わない",
              "- rsl_rl / PPO への接続は予測。Adam・幅・教師がすべて外",
              "- スコープ: condA・T=1e4・batch=1・隠れ 2 層 × 100・alpha=1・"
              "Q2 は末尾窓・等頻度 12 分位・自然対数", ""]
    (outdir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
PREFLIGHT_DIR = f"results/_preflight_{EXPERIMENT}"
SMOKE_DIR = f"results/_smoke_{EXPERIMENT}"


def _provenance(cfg_path: Path, cfg: dict, outdir: Path, sanity: dict,
                analysis: dict, elapsed: dict, started: float) -> dict:
    files = [outdir / name for name in ("verdict.csv", "summary.md",
                                        "layer_stats.csv", "increments.csv",
                                        "config_used.yaml", "s0prime.json")]
    files.extend(sorted((outdir / "arm_status").glob("*.json")))
    reference = Path(ROOT) / cfg["sanity"]["s0_prime_baseline_ref"]
    reference_files = {}
    for name in ("verdict.csv", "summary.md", "provenance.json"):
        path = reference / name
        if path.exists():
            reference_files[f"{reference.name}/{name}"] = _sha_file(path)
    for arm in cfg["sanity"]["s0_prime_arm_map"].values():
        for seed in cfg["common"]["seeds"]:
            path = reference / "logs" / f"{arm}_seed{int(seed)}.npz"
            if path.exists():
                reference_files[f"{reference.name}/logs/{path.name}"] = _sha_file(path)
    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                           text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--short"], cwd=ROOT,
                                        text=True).splitlines()
    except (OSError, subprocess.CalledProcessError):
        git_hash, dirty = None, []
    spec = Path(ROOT) / cfg["spec"]
    return dict(experiment=EXPERIMENT,
                created=time.strftime("%Y-%m-%d %H:%M:%S %z"), command=sys.argv,
                elapsed_sec=round(time.time() - started, 3), arm_elapsed_sec=elapsed,
                cwd=os.getcwd(), python=sys.version, platform=platform.platform(),
                torch=torch.__version__, numpy=np.__version__,
                device=cfg["common"]["device"], git_hash=git_hash, git_dirty=dirty,
                config=str(cfg_path), config_sha256=_sha_file(cfg_path),
                spec=str(spec), spec_sha256=_sha_file(spec) if spec.exists() else None,
                baseline_inputs=reference_files, sanity=sanity, analysis=analysis)


def run_full(cfg_path: Path, cfg: dict, device: str, outdir: Path, *,
             smoke: bool) -> dict:
    started = time.time()
    outdir.mkdir(parents=True, exist_ok=True)
    require_omp(cfg)
    C = cfg["common"]
    total = SMOKE_STEPS if smoke else int(C["total_steps"])
    seeds = [0, 1] if smoke else [int(v) for v in C["seeds"]]

    if smoke:
        gates = dict(pass_=True, S3=require_omp(cfg), S5=_s5_selftest(cfg),
                     S_g_elu_table=_s_closed_form_table(cfg))
        s0 = {"pass_": True, "smoke": True}
    else:
        preflight_path = Path(ROOT) / PREFLIGHT_DIR / "preflight.json"
        if not preflight_path.exists():
            raise FileNotFoundError("run --preflight before the full run")
        gates = json.loads(preflight_path.read_text(encoding="utf-8"))
        if not gates.get("pass_"):
            raise RuntimeError("saved preflight did not pass")
        s0_path = outdir / "s0prime.json"
        if not s0_path.exists():
            raise FileNotFoundError("run --s0prime before the full run (spec §6)")
        s0 = json.loads(s0_path.read_text(encoding="utf-8"))
        if not s0.get("pass_"):
            raise RuntimeError("saved S0' did not pass; the full run must not proceed")

    with (outdir / "config_used.yaml").open("w") as fh:
        yaml.safe_dump(cfg, fh, allow_unicode=True, sort_keys=False)

    elapsed, identity, divergences = {}, {}, {}
    for arm in ARM_ORDER:
        saved = _load_divergence_status(outdir, arm, seeds, total, int(C["lop_every"]))
        if saved is not None:
            elapsed[arm] = 0.0
            divergences[arm] = saved
            identity[arm] = dict(pass_=False, numeric_divergence=True,
                                 resumed_from_status=True, event=saved)
            print(f"[{arm}] saved {NUMERIC_DIVERGENCE} found; resuming after arm",
                  flush=True)
            continue
        if _complete_arm_logs(outdir, arm, seeds, total, int(C["lop_every"])):
            elapsed[arm] = 0.0
            identity[arm] = {"pass_": True, "resumed_from_complete_logs": True}
            print(f"[{arm}] complete logs found; resuming after arm", flush=True)
            continue
        result = _run_arm(cfg, arm, device, outdir, seeds, total)
        elapsed[arm] = result["elapsed_sec"]
        identity[arm] = result["sanity"]
        if result.get("status") == NUMERIC_DIVERGENCE:
            divergences[arm] = result["divergence"]

    sanity = dict(S0prime=s0, S3=gates.get("S3"), S5=gates.get("S5"),
                  S_copy=gates.get("S_copy"), S_pair=gates.get("S_pair"),
                  S_taut=gates.get("S_taut"), S_grad=gates.get("S_grad"),
                  S_elu_limit=gates.get("S_elu_limit"),
                  S_submerge=gates.get("S_submerge"),
                  S_g_elu_table=gates.get("S_g_elu_table"),
                  S7_numeric_divergence=gates.get("S7_numeric_divergence"),
                  S1_S2=identity,
                  S1_S2_completed_arms_pass=bool(all(
                      v["pass_"] for arm, v in identity.items()
                      if arm not in divergences)),
                  S1_S2_all_pass=bool(not divergences and all(
                      v["pass_"] for v in identity.values())),
                  numeric_divergence=divergences)
    if smoke:
        (outdir / "smoke_sanity.json").write_text(
            json.dumps(dict(pass_=sanity["S1_S2_all_pass"], sanity=sanity,
                            elapsed_sec=elapsed), indent=2, ensure_ascii=False,
                       default=str), encoding="utf-8")
        print(f"SMOKE DONE -> {outdir}", flush=True)
        return dict(sanity=sanity, analysis=dict(smoke=True, elapsed_sec=elapsed))

    sanity["S_pair_final"] = _pair_check_final(cfg, outdir, seeds, divergences)
    print(f"[S-pair-final] {'PASS' if sanity['S_pair_final']['pass_'] else 'FAIL'}",
          flush=True)
    result = analyze(cfg, outdir, sanity, elapsed)
    (outdir / "provenance.json").write_text(
        json.dumps(_provenance(cfg_path, cfg, outdir, sanity, result, elapsed, started),
                   indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"ALL DONE -> {outdir}", flush=True)
    return dict(sanity=sanity, analysis=result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/elu_swamp_0830.yaml")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--s0prime", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--outdir")
    args = parser.parse_args()
    if sum((args.preflight, args.s0prime, args.smoke, args.analyze_only)) > 1:
        parser.error("stage flags are mutually exclusive")
    cfg_path = Path(args.config).resolve()
    cfg = load_config(str(cfg_path))
    device = pick_device(cfg)
    if device != "cpu":
        raise ValueError("elu_swamp is CPU-only")
    stage = ("preflight" if args.preflight else "s0prime" if args.s0prime else
             "smoke" if args.smoke else "analyze" if args.analyze_only else "full")
    validate_config(cfg, stage=stage)
    main_dir = Path(ROOT) / cfg["output"]["dir"]
    outdir = (Path(args.outdir).resolve() if args.outdir else
              Path(ROOT) / PREFLIGHT_DIR if args.preflight else
              Path(ROOT) / SMOKE_DIR if args.smoke else main_dir)
    if args.preflight:
        preflight(cfg, device, outdir)
    elif args.s0prime:
        s0prime(cfg, device, outdir)
    elif args.analyze_only:
        gates = json.loads((Path(ROOT) / PREFLIGHT_DIR / "preflight.json")
                           .read_text(encoding="utf-8"))
        seeds = [int(v) for v in cfg["common"]["seeds"]]
        total = int(cfg["common"]["total_steps"])
        divergences = {
            arm: event for arm in ARM_ORDER
            if (event := _load_divergence_status(
                outdir, arm, seeds, total, int(cfg["common"]["lop_every"]))) is not None
        }
        sanity = dict(S0prime=json.loads((outdir / "s0prime.json").read_text()),
                      S1_S2={}, S1_S2_completed_arms_pass=True,
                      S1_S2_all_pass=not divergences,
                      numeric_divergence=divergences,
                      **{k: gates.get(k) for k in
                         ("S3", "S5", "S_copy", "S_pair", "S_taut", "S_grad",
                          "S_elu_limit", "S_submerge", "S_g_elu_table",
                          "S7_numeric_divergence")})
        sanity["S_pair_final"] = _pair_check_final(cfg, outdir, seeds, divergences)
        analyze(cfg, outdir, sanity, {})
    else:
        run_full(cfg_path, cfg, device, outdir, smoke=args.smoke)


if __name__ == "__main__":
    main()
