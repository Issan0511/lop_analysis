"""gate_dial_0902: ゲートの硬さダイヤル（ReLU に収束する 4 族で LoP がいつ現れるか）。

事前登録: ``specs/spec_gate_dial_0902.md``（この実装より**先に** config と一緒に単独
commit する）。Obsidian 側の正本は ``可塑性喪失/spec/ゲート硬さダイヤル_spec_0902.md``。

宿主は ``gate_dose_0830``（1 層・オラクル用量固定 12.16・5M）で、学習経路・記録経路・
用量固定はそのまま ``src.gate_dose`` から import する。新規に足すのは

* ``VecMLPL`` の 2 活性化 ``silu`` / ``gelu``（``src/nets.py``。閉形式・真の導関数）
* ユニット別 ``mob`` / ``absmob`` / ``zmax`` / ``zmean`` / ``v`` の記録（spec §4.3）
* 本モジュールの sanity・集計

対照 ``R_1216`` / ``LR_1216`` / ``E_1216`` は再走しない。主 endpoint は
``results/gate_dose_0830/verdict.csv`` から転記し、V3 と §5.5 の対照だけ同走の
``logs/*.npz``（と m⁻ のための 5M ``ckpts/*.pt``）を読む。

Stages::

    OMP_NUM_THREADS=1 .venv/bin/python -m src.gate_dial_0902 --preflight
    OMP_NUM_THREADS=1 .venv/bin/python -m src.gate_dial_0902 --smoke
    OMP_NUM_THREADS=1 .venv/bin/python -m src.gate_dial_0902 --s-par
    OMP_NUM_THREADS=1 .venv/bin/python -m src.gate_dial_0902 --arm S_b1_1216
    OMP_NUM_THREADS=1 .venv/bin/python -m src.gate_dial_0902 --stage all
    OMP_NUM_THREADS=1 .venv/bin/python -m src.gate_dial_0902 --stage all --analyze-only

``--arm`` は 1 腕だけを走らせて logs を置いて終わる（腕プロセス並列の投入単位）。
決定論は seed ループに触らないので壊れないが、投入前に ``--s-par`` を通すこと。
"""
from __future__ import annotations

import argparse
import copy
import csv
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
from .dose_const_5m import (_input_stats, _refresh_fixed_offset, _target,
                            clopper_pearson, setup_arm_const)
from .elu_swamp import exact_layer_record_elu
from .gate_dose import (GateRecorder, IDENTITY_TOL, SIGMA_TOL, _interval_rows,
                        _load_arm, _window, train_arm_gate)
from .mlp2_phase0 import (_sha_array, _sha_file, identity_sanity_pass,
                          require_omp, spearman, write_csv)
from .mlp2_phase0b import _ci_components, _complete_arm_logs, _window_indices
from .mlp2_phase1 import (NUMERIC_DIVERGENCE, NumericDivergenceError,
                          StreamDigest, _env_hashes, _init_hashes,
                          _seed_state_hashes_p1)
from .nets import VecMLPL
from .ratchet_log import full_support_ro


EXPERIMENT = "gate_dial_0902"
CONFIG = Path(ROOT) / "configs" / "gate_dial_0902.yaml"

ARM_ORDER = ("S_b1_1216", "G_b1_1216",
             "LR_a0p01_1216", "LR_a0p001_1216", "LR_a0p0001_1216",
             "E_a0p1_1216", "E_a0p01_1216",
             "S_b0p3_1216", "S_b3_1216", "G_b0p3_1216", "G_b3_1216",
             "LR_a0p00001_1216", "E_a0p001_1216", "S_b10_1216")
CONTROL_ORDER = ("R_1216", "LR_1216", "E_1216")
STAGE_ARMS = {
    1: ("S_b1_1216", "G_b1_1216"),
    2: ("LR_a0p01_1216", "LR_a0p001_1216", "LR_a0p0001_1216", "E_a0p1_1216",
        "E_a0p01_1216", "S_b0p3_1216", "S_b3_1216", "G_b0p3_1216", "G_b3_1216"),
    3: ("LR_a0p00001_1216", "E_a0p001_1216", "S_b10_1216"),
}

# 事前登録の腕定義（stage, family, activation label, dial）。validate_config が逐語照合する。
REGISTERED_ARMS = {
    "S_b1_1216": (1, "silu", "silu", 1.0),
    "G_b1_1216": (1, "gelu", "gelu", 1.0),
    "LR_a0p01_1216": (2, "leaky", "leaky", 0.01),
    "LR_a0p001_1216": (2, "leaky", "leaky", 0.001),
    "LR_a0p0001_1216": (2, "leaky", "leaky", 0.0001),
    "E_a0p1_1216": (2, "elu", "elu", 0.1),
    "E_a0p01_1216": (2, "elu", "elu", 0.01),
    "S_b0p3_1216": (2, "silu", "silu", 0.3),
    "S_b3_1216": (2, "silu", "silu", 3.0),
    "G_b0p3_1216": (2, "gelu", "gelu", 0.3),
    "G_b3_1216": (2, "gelu", "gelu", 3.0),
    "LR_a0p00001_1216": (3, "leaky", "leaky", 0.00001),
    "E_a0p001_1216": (3, "elu", "elu", 0.001),
    "S_b10_1216": (3, "silu", "silu", 10.0),
}
CONTROL_FAMILY = {"R_1216": "relu", "LR_1216": "leaky", "E_1216": "elu"}
CONTROL_DIAL = {"R_1216": 0.0, "LR_1216": 0.1, "E_1216": 1.0}

SMOKE_STEPS = 30_000
# 本モジュールが足すユニット別列。既存列は 1 列も変えない・消さない。
NEW_UNIT_KEYS = ("mob", "absmob", "zmax", "zmean", "v_unit")


class SanityError(RuntimeError):
    """登録済みの前段チェックが落ちたとき。本走・集計を止める。"""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def _arm(cfg: dict, name: str) -> dict:
    return next(a for a in cfg["arms"] if a["name"] == name)


def _P(cfg: dict) -> dict:
    return cfg["gate_dial"]


def _activation(cfg: dict, arm_cfg: dict) -> tuple[str, float]:
    """arm の (family label, dial) を ``VecMLPL`` の (act, act_alpha) に写す。"""
    label = str(arm_cfg["activation"])
    if label == "relu":
        return "relu", 1.0
    return str(cfg["activation"][label]["name"]), float(arm_cfg["dial"])


def validate_config(cfg: dict, *, stage: str) -> None:
    """凍結した設計からのずれをすべて ValueError にする。"""
    if stage not in {"preflight", "smoke", "spar", "run", "analyze"}:
        raise ValueError(f"unknown stage {stage!r}")
    C, A, I, P, G, S = (cfg["common"], cfg["condA"], cfg["intervention"],
                        cfg["phase1"], cfg["gate_dial"], cfg["sanity"])
    if [a["name"] for a in cfg["arms"]] != list(ARM_ORDER):
        raise ValueError(f"arms must be ordered as {ARM_ORDER}")
    for arm in cfg["arms"]:
        want_stage, want_family, want_act, want_dial = REGISTERED_ARMS[arm["name"]]
        if (int(arm["stage"]) != want_stage
                or str(arm["family"]) != want_family
                or str(arm["activation"]) != want_act
                or float(arm["dial"]) != want_dial
                or [int(v) for v in arm["hidden"]] != [100]
                or [int(v) for v in arm.get("centered_layers", [])] != [1]
                or _target(arm) != 3.041
                or float(arm["target_dose"]) != 12.16):
            raise ValueError(f"{arm['name']} differs from the preregistration")
    if [a["name"] for a in cfg["arms"] if int(a["stage"]) == 1] != list(STAGE_ARMS[1]):
        raise ValueError("stage 1 arms changed")
    if (int(A["m"]), int(A["f"]), int(A["target_hidden"])) != (20, 15, 100):
        raise ValueError("gate_dial requires condA m=20, f=15, teacher width=100")
    if list(A["T_values"]) != [10_000] or list(A["encodings"]) != ["std"]:
        raise ValueError("gate_dial requires T=10000 and std encoding")
    if int(C.get("generator_offset", -1)) != 0:
        raise ValueError("generator_offset must be an explicit 0 (spec §3)")
    if (str(I["name"]) != "oracle_fixed_mu_offset" or I["oracle"] is not True
            or I["consumes_rng"] is not False
            or float(I["center_alpha_compat"]) != 0.01):
        raise ValueError("the oracle-dose intervention changed")
    act = cfg["activation"]
    if (str(act["relu"]["name"]) != "relu"
            or str(act["leaky"]["name"]) != "leaky_relu"
            or str(act["elu"]["name"]) != "elu"
            or str(act["elu"]["derivative_form"]) != "alpha_exp"
            or str(act["silu"]["name"]) != "silu"
            or str(act["gelu"]["name"]) != "gelu"
            or act["gelu"]["exact_erf"] is not True
            or act["silu"]["has_valley"] is not True
            or act["gelu"]["has_valley"] is not True
            or act["autograd"] is not False
            or act["consumes_rng"] is not False
            or act["is_true_gradient"] is not True):
        raise ValueError("activation definitions changed")
    expected_phase = {
        "task_period": 10_000, "early_tasks": [2, 11],
        "late_tasks_5m": [491, 500], "window_1m_tasks": [91, 100],
        "window_points_are_task_ends_only": True,
        "window_records_per_10task_window": 10,
        "onset_threshold": 0.05, "onset_present_min": 5,
        "unfit_floor": 1e-16, "recalibrate_floor": False,
        "bootstrap_B": 10_000, "bootstrap_seed": 20_260_907,
        "ci_method": "percentile_primary_studentized_secondary",
    }
    for key, value in expected_phase.items():
        if P[key] != value:
            raise ValueError(f"phase1.{key} differs from the preregistration")
    if (float(P["degenerate_se_tol"]) != 1e-15
            or float(P["degenerate_frac_max"]) != 0.01
            or float(P["degenerate_width_ratio_max"]) != 100.0):
        raise ValueError("CI degeneracy guard changed")
    if (G["design"]["uses_self_term_decomposition"] is not False
            or float(G["design"]["displacement_bound_constant_K"]) != 1.0
            or float(G["design"]["freeze_depth_phi_prime_threshold"]) != 1e-6):
        raise ValueError("the §2 design constants changed")
    want_threshold = (float(P["onset_threshold"])
                      / (float(C["lr_main"]) * int(C["total_steps"])
                         * float(G["design"]["displacement_bound_constant_K"])))
    if not math.isclose(float(G["design"]["freeze_depth_phi_prime_threshold"]),
                        want_threshold, rel_tol=1e-12):
        raise ValueError("the freeze-depth threshold is not 0.05/(lr*T*K)")
    if list(G["v1_arms"]) != ["S_b1_1216", "G_b1_1216"]:
        raise ValueError("V1 arms changed")
    if len(G["v1_map"]) != 9 or _v1_map_disagrees_with_spec_table(G):
        raise ValueError("the enumerated V1 map disagrees with the spec table")
    if (int(G["onset_zero_max"]) != 0 or int(G["onset_present_min"]) != 5):
        raise ValueError("onset state definition changed")
    if dict(G["control_expected_onset_5m"]) != {"R_1216": 10, "LR_1216": 0,
                                                "E_1216": 0}:
        raise ValueError("control expectations changed")
    if str(G["p3prime_baseline"]) != "R_1216":
        raise ValueError("the P3' baseline changed")
    if dict(G["p5_soft_end_by_family"]) != {"leaky": ["LR_1216"], "elu": ["E_1216"],
                                            "silu": ["E_1216", "LR_1216"],
                                            "gelu": ["E_1216", "LR_1216"]}:
        raise ValueError("the P5' soft ends changed")
    if (float(G["p5_equivalence_margin"]) != 0.15
            or G["p5_margin_recalibrated_for_this_system"] is not False
            or list(G["p5_labels"]["order"]) != ["equivalent", "short_of_soft",
                                                 "inconclusive"]
            or G["p5_sign_test_report_only"] is not True):
        raise ValueError("P5' registration changed")
    if float(G["v2_margin"]) != 0.15 or int(G["v2_onset_drop_reversal_min"]) != 3:
        raise ValueError("V2 thresholds changed")
    if list(G["v2_label_order"]) != ["REVERSAL", "FLAT_IN_RANGE",
                                     "MONOTONE_TOWARD_RELU", "PARTIAL"]:
        raise ValueError("the V2 decision order changed")
    for family, ladder in dict(G["ladders"]).items():
        if ladder[-1] != "R_1216":
            raise ValueError(f"ladder {family} must end at R_1216")
        for name in ladder[:-1]:
            if name not in ARM_ORDER and name not in CONTROL_ORDER:
                raise ValueError(f"unknown arm {name} in ladder {family}")
    if (G["dial_table"]["emit_label"] is not False
            or list(G["dial_table"]["predictors"]) != ["m_minus", "u_fr",
                                                       "frozen_plus_valley_frac"]
            or list(G["dial_table"]["m_minus"]["aggregation_order"]) != [
                "submerged_unit_records_within_seed", "median_within_seed",
                "median_over_seeds"]):
        raise ValueError("the V3 registration changed")
    if G["report_only"]["in_verdict"] is not False:
        raise ValueError("REPORT_ONLY quantities must stay out of the verdict")
    if (str(G["report_only"]["revival_primary_condition"])
            != "same_task_and_flip_state_unchanged"):
        raise ValueError("the revival definition changed")
    if (int(S["s_pair_steps"]) != 30_000 or int(S["s_limit_steps"]) != 30_000
            or float(S["s_dose_rel_tol"]) != 1e-10
            or float(S["s_fd_tol"]) != 1e-6
            or float(S["s_mob_tol"]) != 1e-6
            or float(S["s_limit_grad_tol"]) != 1e-3
            or int(S["s_cap_min_seeds"]) != 9
            or S["s_elu_limit_alpha_to_zero"] is not True
            or S["s_taut_check"] is not True
            or S["s_mask_check"] is not True
            or S["s_dial_check"] is not True
            or int(S["omp_num_threads"]) != 1
            or S["s6_floor_calibration"] is not False):
        raise ValueError("sanity gates changed")
    if stage in {"run", "analyze"}:
        if int(C["total_steps"]) != 5_000_000 or list(C["seeds"]) != list(range(10)):
            raise ValueError("the full run requires 5M steps and seeds 0..9")
        if str(C["device"]) != "cpu":
            raise ValueError("gate_dial is CPU-only")


def _v1_map_disagrees_with_spec_table(G: dict) -> bool:
    """列挙した V1 写像が spec §5.1 の表と一致することを独立に検算する。

    bwd_leak_0902 追補 7 と同じ用心。config の 9 セルと、ここに書いた表の側の
    素直な実装を突き合わせる。片方を書き換えたらここで落ちる。
    """
    for s in ("zero", "mid", "present"):
        for g in ("zero", "mid", "present"):
            if s == "present" and g == "present":
                want = "SOFT_GATES_RELU_SIDE"
            elif s == "zero" and g == "zero":
                want = "SOFT_GATES_SOFT_SIDE"
            elif {s, g} == {"zero", "present"}:
                want = "SPLIT_SILU_GELU"
            else:
                want = "PARTIAL"
            if G["v1_map"].get(f"{s}_{g}") != want:
                return True
    return False


def _selected_arms(cfg: dict, stage: str) -> list[str]:
    if stage == "all":
        return list(ARM_ORDER)
    return list(STAGE_ARMS[int(stage)])


def _family(cfg: dict, arm: str) -> str:
    if arm in CONTROL_FAMILY:
        return CONTROL_FAMILY[arm]
    return str(_arm(cfg, arm)["family"])


def _dial(cfg: dict, arm: str) -> float:
    if arm in CONTROL_DIAL:
        return CONTROL_DIAL[arm]
    return float(_arm(cfg, arm)["dial"])


# ---------------------------------------------------------------------------
# §2 の幾何（谷底と凍結深さ）。閉形式を**数値で解き直す**。実験出力ではない。
# ---------------------------------------------------------------------------
def _probe_net(act: str, alpha: float) -> VecMLPL:
    net = VecMLPL(1, [2], 2, torch.Generator().manual_seed(0), "cpu")
    return net.set_activation(act, alpha, "alpha_exp")


def _phi_prime(net: VecMLPL, u: torch.Tensor) -> torch.Tensor:
    """深さ ``u > 0`` での ``phi'(-u)``（float64）。"""
    z = -u
    return net.act_grad(z, net.act_fn(z))


def valley_depth(act: str, dial: float) -> float:
    """負側で ``phi'`` が最初に 0 を切る深さ ``u*``。無ければ NaN。"""
    if act not in ("silu", "gelu"):
        return float("nan")
    net = _probe_net(act, dial)
    hi = 40.0 / dial
    u = torch.linspace(hi / 400_000, hi, 400_000, dtype=torch.float64)
    g = _phi_prime(net, u)
    idx = torch.nonzero(g <= 0)
    if not len(idx):
        return float("nan")
    j = int(idx[0])
    lo_u, hi_u = float(u[j - 1]) if j else 0.0, float(u[j])
    for _ in range(200):
        mid = 0.5 * (lo_u + hi_u)
        if float(_phi_prime(net, torch.tensor([mid], dtype=torch.float64))) <= 0:
            hi_u = mid
        else:
            lo_u = mid
    return hi_u


def freeze_depth(act: str, dial: float, threshold: float) -> float:
    """``u >= u_fr`` の全域で ``|phi'(-u)| < threshold`` になる最小の深さ。

    ``|phi'|`` は SiLU/GELU では単調でない（符号が変わる）ので、単純な二分探索を
    掛けずに「その深さより深い全域の最大値」で判定する。leaky は定数なので
    ``a >= threshold`` のとき ``inf``。
    """
    net = _probe_net(act, dial)
    hi = 400.0 if act != "silu" else max(400.0, 200.0 / dial)
    tail = torch.linspace(hi * 0.999, hi, 1001, dtype=torch.float64)
    if float(_phi_prime(net, tail).abs().max()) > threshold:
        return float("inf")
    lo_u, hi_u = 0.0, hi
    for _ in range(80):
        mid = 0.5 * (lo_u + hi_u)
        us = torch.linspace(mid, hi, 20_000, dtype=torch.float64)
        if float(_phi_prime(net, us).abs().max()) <= threshold:
            hi_u = mid
        else:
            lo_u = mid
    return hi_u


def _geometry(cfg: dict, arm: str) -> dict:
    """腕の (u*, u_fr) を登録値と数値解の両方で返す。"""
    G = _P(cfg)
    threshold = float(G["design"]["freeze_depth_phi_prime_threshold"])
    if arm in CONTROL_ORDER:
        entry = dict(cfg["controls"]["arms"][arm])
        label, dial = str(entry["activation"]), float(entry["dial"])
        act = {"relu": "relu", "leaky_relu": "leaky_relu", "elu": "elu"}[label]
    else:
        arm_cfg = _arm(cfg, arm)
        entry = arm_cfg
        act, dial = _activation(cfg, arm_cfg)
    if act == "relu":
        numeric_star, numeric_fr = 0.0, 0.0
    else:
        numeric_star = valley_depth(act, dial)
        numeric_fr = freeze_depth(act, dial, threshold)
    registered_star = entry.get("u_star")
    registered_fr = entry.get("u_fr")
    return dict(arm=arm, activation=act, dial=dial,
                u_star_registered=(None if registered_star is None
                                   else float(registered_star)),
                u_fr_registered=(None if registered_fr is None
                                 else float(registered_fr)),
                u_star_numeric=numeric_star, u_fr_numeric=numeric_fr,
                has_valley=bool(np.isfinite(numeric_star)),
                threshold=threshold)


# ---------------------------------------------------------------------------
# Learning path — gate_dose の経路をそのまま使い、活性化とダイヤルだけ差し込む
# ---------------------------------------------------------------------------
def setup_arm_dial(cfg: dict, arm_cfg: dict, device: str) -> dict:
    """``setup_arm_const`` の状態に活性化とダイヤルを差し込む。

    ``set_activation`` は乱数を消費せず状態も書き換えないので、腕は
    ``gate_dose_0830`` と init・教師・入力列・flip が bit 一致する（S-pair）。
    """
    st = setup_arm_const(cfg, arm_cfg, device)
    act, alpha = _activation(cfg, arm_cfg)
    st["net"].set_activation(act, alpha, "alpha_exp")
    st["activation"] = act
    st["act_alpha"] = float(alpha)
    st["family"] = str(arm_cfg.get("family", ""))
    return st


def unit_extra_record(st: dict) -> dict:
    """第 1 層の ``mob`` / ``absmob`` / ``zmax`` / ``zmean`` と読み出し ``v``。

    ``exact_layer_record_elu`` の層 1 の ``z`` の作り方を逐語で真似る（32 点の
    厳密支持・float64・中心化フラグと ``layer_means`` の扱いまで同じ）。したがって
    ``zmean`` は既存列 ``layer1_zbar`` と同じ量であり、S-mob が両者と
    ``(M+B)*denom`` の一致を検査する。学習状態は読むだけで書き換えない。
    """
    net = st["net"]
    flags = st.get("centered_layers") or [False] * len(net.Ws)
    means = st.get("layer_means") or [None] * len(net.Ws)
    with torch.no_grad():
        cur = full_support_ro(st["env"]).double()
        if flags[0]:
            cur = cur - means[0].double()[None]
        W, b = net.Ws[0].double(), net.bs[0].double()
        z = torch.einsum("rhd,prd->prh", W, cur) + b
        grad = net.act_grad(z, net.act_fn(z))
        return dict(mob=grad.mean(dim=0), absmob=grad.abs().mean(dim=0),
                    zmax=z.amax(dim=0), zmean=z.mean(dim=0),
                    v_unit=net.v.double())


class DialRecorder(GateRecorder):
    """``GateRecorder`` に spec §4.3 の 5 つのユニット別列を足したもの。

    ``record_units=False`` にすると ``GateRecorder`` と完全に同じ挙動になり、
    S-log-b が 2 走の bit 一致（軌道中立性）を検査できる。
    """

    def __init__(self, steps: list[int], st: dict, *, record_units: bool = True):
        super().__init__(steps, st)
        self.record_units = bool(record_units)
        n, runs, width = len(self.steps), st["R"], st["hidden"][0]
        self.unit = ({key: np.empty((n, runs, width), dtype=np.float32)
                      for key in NEW_UNIT_KEYS} if self.record_units else {})

    def __call__(self, st: dict, step: int) -> None:
        super().__call__(st, step)
        if not self.record_units:
            return
        i = self.index.get(int(step))
        if i is None:
            return
        extra = unit_extra_record(st)
        for key in NEW_UNIT_KEYS:
            self.unit[key][i] = extra[key].detach().cpu().numpy().astype(np.float32)


def write_arm_logs_dial(outdir: Path, arm: str, st: dict,
                        rec: DialRecorder) -> list[Path]:
    """``gate_dose.write_arm_logs_gate`` の列に family / dial と 5 列を足す。"""
    logdir = outdir / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ri, run in enumerate(st["runs"]):
        seed = int(run["seed"])
        payload = dict(
            step=rec.steps, run_id=np.array(run["run_id"]), arm=np.array(arm),
            seed=np.int64(seed), activation=np.array(st["activation"]),
            act_alpha=np.float64(st["act_alpha"]),
            family=np.array(st.get("family", "")),
            task_period=np.int64(run["period"]),
            target_mu_norm=np.float64(np.nan if st.get("target_mu_norm") is None
                                      else st["target_mu_norm"]),
            target_dose=np.float64(np.nan if st.get("target_dose") is None
                                   else st["target_dose"]),
            state_hash_final=np.array(json.dumps(
                _seed_state_hashes_p1(st, ri), sort_keys=True)),
            state_hash_1m=np.array(json.dumps(
                rec.state_hash_1m.get(seed, {}), sort_keys=True)))
        for key, value in rec.run.items():
            payload[key] = value[:, ri]
        payload["flip_state"] = rec.flip_state[:, ri]
        for key, value in rec.extra.items():
            payload[key] = value[:, ri]
        for li, layer in enumerate(rec.layers, start=1):
            for key, value in layer.items():
                payload[f"layer{li}_{key}"] = value[:, ri]
        for key, value in rec.unit.items():
            payload[f"layer1_{key}"] = value[:, ri]
        path = logdir / f"{arm}_seed{seed}.npz"
        np.savez_compressed(path, **payload)
        paths.append(path)
    return paths


def _arm_status_path(outdir: Path, arm: str) -> Path:
    return outdir / "arm_status" / f"{arm}.json"


def _load_divergence_status(outdir: Path, arm: str, seeds: list[int], total: int,
                            probe_every: int) -> dict | None:
    path = _arm_status_path(outdir, arm)
    if not path.exists():
        return None
    try:
        event = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    valid = (event.get("status") == NUMERIC_DIVERGENCE and event.get("arm") == arm
             and event.get("registered_seeds") == seeds
             and int(event.get("registered_total_steps", -1)) == total
             and int(event.get("probe_every", -1)) == probe_every
             and event.get("rescue") == "none")
    return event if valid else None


def _run_arm(cfg: dict, arm: str, device: str, outdir: Path, seeds: list[int],
             total: int) -> dict:
    c = copy.deepcopy(cfg)
    c["common"]["seeds"] = seeds
    st = setup_arm_dial(c, _arm(c, arm), device)
    every = int(c["common"]["lop_every"])
    probes = list(range(0, total + 1, every))
    if probes[-1] != total:
        probes.append(total)
    _, sanity0 = exact_layer_record_elu(st, SIGMA_TOL)
    if not identity_sanity_pass(sanity0, IDENTITY_TOL):
        raise SanityError(f"{arm} initial exact-support identity failed")
    rec = DialRecorder(probes, st)
    checkpoints = [int(v) for v in c["common"].get("checkpoints", []) if int(v) <= total]
    print(f"[{arm}] act={st['activation']} dial={st['act_alpha']:g} "
          f"dose={st.get('target_dose')} seeds={seeds} steps={total:,}", flush=True)
    started = time.time()
    try:
        elapsed = train_arm_gate(st, rec, probes, total, outdir, checkpoints)
    except NumericDivergenceError as exc:
        elapsed = time.time() - started
        event = dict(exc.event)
        event.update(probe_every=every, registered_total_steps=int(total),
                     registered_seeds=[int(v) for v in seeds],
                     activation=st["activation"], act_alpha=st["act_alpha"],
                     family=st.get("family"), elapsed_sec=float(elapsed),
                     detection="nonfinite_training_state_at_probe",
                     partial_logs_excluded=True, rescue="none")
        path = _arm_status_path(outdir, arm)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(event, indent=2, ensure_ascii=False),
                        encoding="utf-8")
        print(f"[{arm}] {NUMERIC_DIVERGENCE} at step {event['detected_step']:,}",
              flush=True)
        return dict(status=NUMERIC_DIVERGENCE, elapsed_sec=elapsed,
                    sanity=dict(pass_=False, numeric_divergence=True, event=event),
                    divergence=event)
    sanity = rec.sanity()
    if not sanity["pass_"]:
        raise SanityError(f"{arm} exact-support sanity failed: {sanity}")
    write_arm_logs_dial(outdir, arm, st, rec)
    print(f"[{arm}] complete in {elapsed:.1f}s", flush=True)
    return dict(status="COMPLETE", elapsed_sec=elapsed, sanity=sanity)


# ---------------------------------------------------------------------------
# 前段チェック（spec §6）
# ---------------------------------------------------------------------------
def _s_dial(cfg: dict) -> dict:
    """S-dial: config に凍結した u* / u_fr が数値解と一致すること。

    spec §2 の表は 2 桁の丸めなので相対許容を置く。**この表は実験出力ではない**
    ことを provenance と summary に書く（spec §8）。
    """
    tol = float(_P(cfg)["design"]["u_fr_spec_rel_tol"])
    rows, failures = [], []
    for arm in list(ARM_ORDER) + list(CONTROL_ORDER):
        geo = _geometry(cfg, arm)
        row = dict(arm=arm, **{k: v for k, v in geo.items() if k != "arm"})
        for key in ("u_star", "u_fr"):
            registered = geo[f"{key}_registered"]
            numeric = geo[f"{key}_numeric"]
            if registered is None:
                # 登録が null なのは「無い」の意味。数値解も無限/NaN でなければ矛盾。
                ok = not np.isfinite(numeric)
            elif not np.isfinite(numeric):
                ok = False
            else:
                ok = abs(numeric - registered) <= tol * max(abs(registered), 1e-12)
            row[f"{key}_agrees"] = bool(ok)
            if not ok:
                failures.append(dict(arm=arm, quantity=key, registered=registered,
                                     numeric=numeric))
        rows.append(row)
    return dict(pass_=not failures, rel_tol=tol, rows=rows, failures=failures,
                note="closed-form + K=1 substitution; not an experimental output")


def _phi_prime_reference(act: str, beta: float, z: torch.Tensor) -> torch.Tensor:
    """桁落ちしない別形での ``phi'``（S-fd の第 2 経路）。

    登録された GELU は ``Phi(t) = 0.5*(1 + erf(t/sqrt2))``（spec §4.3）で、これは
    ``t < -8.3`` 付近から **1 + erf の引き算で桁落ちする**（float64 で Phi が 0 に
    潰れる）。潰れた分の絶対誤差は eps/2 なので力学には効かない（凍結閾値 1e-6 の
    10 桁下）が、**中心差分の照合は成立しなくなる**。そこで ``erfc`` 版
    ``Phi(t) = 0.5*erfc(-t/sqrt2)`` を参照として持ち、
      * 差分が無情報でない領域 → 中心差分で照合（登録どおり）
      * 無情報な深い裾     → この参照形と絶対許容で照合
    の 2 経路にする。**登録された算術は変えない。**
    """
    t = beta * z
    if act == "silu":
        sig = torch.sigmoid(t)
        return sig * (1.0 + t * (1.0 - sig))
    cdf = 0.5 * torch.erfc(-t / math.sqrt(2.0))
    pdf = torch.exp(-0.5 * t * t) / math.sqrt(2.0 * math.pi)
    return cdf + t * pdf


def _s_fd(cfg: dict) -> dict:
    """S-fd: SiLU/GELU の後ろ向き閉形式を float64 中心差分と照合する。

    代替勾配（bwd_leak）と違い**真の導関数**なので有限差分が成立する。ただし
    GELU の登録形は深い負側で桁落ちするので、差分が無情報になった点は
    ``_phi_prime_reference`` との絶対照合に切り替える（どちらの経路を通ったかを
    行ごとに残す）。
    """
    S = cfg["sanity"]
    tol = float(S["s_fd_tol"])
    n = int(S["s_fd_points"])
    eps = float(np.finfo(np.float64).eps)
    abs_tol = 8.0 * eps
    rows, failures = [], []
    for arm in ARM_ORDER:
        act, dial = _activation(cfg, _arm(cfg, arm))
        if act not in ("silu", "gelu"):
            continue
        net = _probe_net(act, dial)
        ranges = [(-20.0, 20.0), (-20.0 / dial, 20.0 / dial)]
        for lo, hi in ranges:
            worst_fd, worst_ref, n_informative = 0.0, 0.0, 0
            shallowest_uninformative = float("-inf")
            for value in torch.linspace(lo, hi, n, dtype=torch.float64).tolist():
                h = max(abs(value), 1.0) * 1e-6
                z = torch.tensor([value - h, value + h], dtype=torch.float64)
                f = net.act_fn(z)
                fd = float((f[1] - f[0]) / (2 * h))
                zz = torch.tensor([value], dtype=torch.float64)
                exact = float(net.act_grad(zz, net.act_fn(zz)))
                reference = float(_phi_prime_reference(act, dial, zz))
                # 参照形との照合は全点で行う（深い裾では絶対許容が効く）
                scale_ref = max(abs(reference), 1e-12)
                err_ref = abs(exact - reference)
                worst_ref = max(worst_ref, err_ref / scale_ref
                                if err_ref > abs_tol else 0.0)
                if err_ref > abs_tol and err_ref / scale_ref > tol:
                    failures.append(dict(arm=arm, where="reference_form", z=value,
                                         exact=exact, reference=reference))
                # 差分が情報を持つか: 登録形の丸め誤差 (eps/2 の桁落ちを含む) が
                # 差分の信号 2h*|phi'| を超えていないか
                noise = max(eps * abs(float(f.abs().max())),
                            eps * 0.5 * abs(value) if act == "gelu" else 0.0)
                signal = 2.0 * h * max(abs(reference), 1e-300)
                informative = bool(noise / signal < tol)
                if informative:
                    n_informative += 1
                    rel = abs(fd - exact) / max(abs(exact), abs(fd), 1e-300)
                    worst_fd = max(worst_fd, rel)
                    if rel > tol:
                        failures.append(dict(arm=arm, where="finite_difference",
                                             z=value, closed_form=exact,
                                             finite_difference=fd, relerr=rel))
                elif value < 0:
                    shallowest_uninformative = max(shallowest_uninformative, value)
            rows.append(dict(arm=arm, activation=act, dial=dial, lo=lo, hi=hi,
                             points=n, informative_points=n_informative,
                             fd_max_relerr=worst_fd,
                             reference_max_relerr=worst_ref,
                             fd_uninformative_at_or_below_z=(
                                 shallowest_uninformative
                                 if np.isfinite(shallowest_uninformative)
                                 else float("nan"))))
    return dict(pass_=not failures, tolerance=tol, absolute_tolerance=abs_tol,
                rows=rows, failures=failures,
                note="the registered Phi = 0.5*(1+erf) cancels in the deep negative "
                     "tail, so the central difference stops being informative there; "
                     "those probes are checked against the erfc form instead. The "
                     "registered arithmetic is unchanged (spec §4.3).")


def _s_num(cfg: dict) -> dict:
    """S-num: NaN/inf が出ないことと、float32 で厳密 0 になる深さ。

    深さは provenance に書く（spec §4.3 の float32 飽和）。**判定には使わない。**
    """
    S = cfg["sanity"]
    lo, hi = [float(v) for v in S["s_num_range"]]
    n = int(S["s_num_points"])
    rows, failures = [], []
    for arm in ARM_ORDER:
        act, dial = _activation(cfg, _arm(cfg, arm))
        net = _probe_net(act, dial)
        for dtype in (torch.float64, torch.float32):
            z = torch.linspace(lo, hi, n, dtype=dtype)
            a = net.act_fn(z)
            g = net.act_grad(z, a)
            finite = bool(torch.isfinite(a).all() and torch.isfinite(g).all())
            if not finite:
                failures.append(dict(arm=arm, dtype=str(dtype), reason="nonfinite"))
            if dtype is torch.float32:
                negative = z < 0
                zero_fwd = (a == 0) & negative
                zero_bwd = (g == 0) & negative
                depth_fwd = (float(-z[zero_fwd].max()) if bool(zero_fwd.any())
                             else float("inf"))
                depth_bwd = (float(-z[zero_bwd].max()) if bool(zero_bwd.any())
                             else float("inf"))
                rows.append(dict(arm=arm, activation=act, dial=dial, finite=finite,
                                 float32_forward_exact_zero_depth=depth_fwd,
                                 float32_backward_exact_zero_depth=depth_bwd))
    return dict(pass_=not failures, range=[lo, hi], points=n, rows=rows,
                failures=failures,
                note="saturation depths are recorded, not judged")


def _s_limit_smooth(cfg: dict) -> dict:
    """S-limit（SiLU/GELU 側）: beta を大きくすると ReLU に寄ることの単体テスト。

    極限は厳密一致しないので bit 一致ではなく許容つきで見る（spec §6）。
    ``z=0`` の ``phi' = 1/2`` は全 beta で正しい（不連続点の中点）。
    """
    S = cfg["sanity"]
    beta = float(S["s_limit_beta"])
    fwd_tol = float(S["s_limit_forward_tol_over_beta"]) / beta
    grad_tol = float(S["s_limit_grad_tol"])
    cut = float(S["s_limit_grid_abs_z_min_over_beta"]) / beta
    relu = _probe_net("relu", 1.0)
    rows, failures = [], []
    z = torch.cat([torch.linspace(-1.0, -cut, 4000, dtype=torch.float64),
                   torch.linspace(cut, 1.0, 4000, dtype=torch.float64)])
    for act in ("silu", "gelu"):
        net = _probe_net(act, beta)
        fwd = float((net.act_fn(z) - relu.act_fn(z)).abs().max())
        grad = float((net.act_grad(z, net.act_fn(z))
                      - relu.act_grad(z, relu.act_fn(z))).abs().max())
        zero = torch.zeros(1, dtype=torch.float64)
        half = float(net.act_grad(zero, net.act_fn(zero)))
        row = dict(activation=act, beta=beta, forward_max_abs=fwd,
                   forward_tol=fwd_tol, grad_max_abs=grad, grad_tol=grad_tol,
                   phi_prime_at_zero=half)
        rows.append(row)
        if fwd > fwd_tol or grad > grad_tol or half != 0.5:
            failures.append(row)
    return dict(pass_=not failures, rows=rows, failures=failures)


def _s_elu_limit(cfg: dict, outdir: Path) -> dict:
    """S-limit（ELU 側）: alpha=0 の ELU が ReLU 経路と 30k 短走で bit 一致すること。"""
    steps = int(cfg["sanity"]["s_limit_steps"])
    c = copy.deepcopy(cfg)
    c["common"]["seeds"] = [0, 1]
    base = copy.deepcopy(_arm(c, "E_a0p1_1216"))
    relu_cfg = copy.deepcopy(base)
    relu_cfg["activation"] = "relu"
    relu = setup_arm_dial(c, relu_cfg, "cpu")
    other = setup_arm_dial(c, base, "cpu")
    other["net"].set_activation("elu", 0.0, "alpha_exp")
    other["activation"], other["act_alpha"] = "elu", 0.0
    grid = torch.linspace(-30, 30, 4001, dtype=torch.float64)
    static_forward = bool(torch.equal(relu["net"].act_fn(grid),
                                      other["net"].act_fn(grid)))
    static_grad = bool(torch.equal(
        relu["net"].act_grad(grid, relu["net"].act_fn(grid)),
        other["net"].act_grad(grid, other["net"].act_fn(grid))))
    train_arm_gate(relu, lambda *_: None, [], steps, outdir, [])
    train_arm_gate(other, lambda *_: None, [], steps, outdir, [])
    a, b = _init_hashes(relu), _init_hashes(other)
    differences = sorted(k for k, v in a.items() if b.get(k) != v)
    return dict(pass_=bool(static_forward and static_grad and not differences),
                steps=steps, static_forward_equal=static_forward,
                static_grad_equal=static_grad,
                trained_state_differences=differences)


def _s_mob(cfg: dict, outdir: Path) -> dict:
    """S-mob: 新規ロガーが既知の量と一致すること（30k 短走）。

    ``mob`` は ReLU 腕で ``p_hat``、leaky 腕で ``a + (1-a) p_hat`` と一致する
    （独立な量になるのは ELU・SiLU・GELU 腕だけ = [[引用禁止]] の可動度の項）。
    ``zmean`` は ``(M+B) * denom`` と一致する。
    """
    S = cfg["sanity"]
    steps, tol = int(S["s_mob_steps"]), float(S["s_mob_tol"])
    every = int(cfg["common"]["lop_every"])
    c = copy.deepcopy(cfg)
    c["common"]["seeds"] = [0, 1]
    probes = list(range(0, steps + 1, every))
    base = copy.deepcopy(_arm(c, "LR_a0p01_1216"))
    rows, failures = [], []
    for label, activation, dial in (("relu", "relu", 0.0),
                                    ("leaky_a0.01", "leaky", 0.01),
                                    ("leaky_a0.1", "leaky", 0.1)):
        arm_cfg = copy.deepcopy(base)
        arm_cfg["activation"] = activation
        arm_cfg["dial"] = dial
        st = setup_arm_dial(c, arm_cfg, "cpu")
        rec = DialRecorder(probes, st)
        train_arm_gate(st, rec, probes, steps, outdir, [])
        p_hat = rec.layers[0]["p_hat"].astype(np.float64)
        mob = rec.unit["mob"].astype(np.float64)
        expected = p_hat if activation == "relu" else dial + (1.0 - dial) * p_hat
        mob_err = float(np.abs(mob - expected).max())
        zmean = rec.unit["zmean"].astype(np.float64)
        formula = ((rec.layers[0]["M"].astype(np.float64)
                    + rec.layers[0]["B"].astype(np.float64))
                   * rec.layers[0]["denom"].astype(np.float64))
        good = np.isfinite(formula)
        scale = np.maximum(np.abs(formula[good]), 1.0)
        zmean_err = float((np.abs(zmean[good] - formula[good]) / scale).max())
        absmob_err = float(np.abs(rec.unit["absmob"].astype(np.float64)
                                  - mob).max())    # 負側の phi' が非負な族では同値
        zmax_ge_zmean = bool((rec.unit["zmax"] >= rec.unit["zmean"] - 1e-6).all())
        row = dict(arm=label, activation=activation, dial=dial, steps=steps,
                   mob_max_abs_err=mob_err, zmean_max_rel_err=zmean_err,
                   absmob_equals_mob_max_abs=absmob_err,
                   zmax_ge_zmean=zmax_ge_zmean,
                   n_na_in_formula=int((~good).sum()))
        rows.append(row)
        if (mob_err > tol or zmean_err > tol or absmob_err > tol
                or not zmax_ge_zmean):
            failures.append(row)
    return dict(pass_=not failures, tolerance=tol, rows=rows, failures=failures,
                note="mob is a linear function of p_hat on ReLU/leaky; it is an "
                     "independent quantity only on ELU/SiLU/GELU arms")


def _s_log_b(cfg: dict, outdir: Path) -> dict:
    """S-log-b: 追加ロガーが軌道中立であること（30k・既存全列が bit 一致）。"""
    steps = int(cfg["sanity"]["s_log_b_steps"])
    every = int(cfg["common"]["lop_every"])
    c = copy.deepcopy(cfg)
    c["common"]["seeds"] = [0, 1]
    probes = list(range(0, steps + 1, every))
    results = {}
    for label, record_units in (("with_logger", True), ("without_logger", False)):
        st = setup_arm_dial(c, _arm(c, "S_b1_1216"), "cpu")
        rec = DialRecorder(probes, st, record_units=record_units)
        train_arm_gate(st, rec, probes, steps, outdir, [])
        results[label] = dict(
            state=_init_hashes(st), env=_env_hashes(st),
            run={k: _sha_array(v) for k, v in rec.run.items()},
            layers=[{k: _sha_array(v) for k, v in layer.items()}
                    for layer in rec.layers],
            extra={k: _sha_array(v) for k, v in rec.extra.items()},
            flip=_sha_array(rec.flip_state), unit_keys=sorted(rec.unit))
    a, b = results["with_logger"], results["without_logger"]
    differences = []
    for section in ("state", "env", "run", "extra"):
        for key, value in a[section].items():
            if b[section].get(key) != value:
                differences.append(dict(where=f"{section}.{key}"))
    for li, (la, lb) in enumerate(zip(a["layers"], b["layers"]), start=1):
        for key, value in la.items():
            if lb.get(key) != value:
                differences.append(dict(where=f"layer{li}.{key}"))
    if a["flip"] != b["flip"]:
        differences.append(dict(where="flip_state"))
    added = sorted(set(a["unit_keys"]) - set(b["unit_keys"]))
    removed = sorted(set(b["unit_keys"]) - set(a["unit_keys"]))
    return dict(pass_=bool(not differences and not removed
                           and added == sorted(NEW_UNIT_KEYS)),
                steps=steps, differences=differences, added_columns=added,
                removed_columns=removed)


def _s_pair_and_dose(cfg: dict, outdir: Path, arms: list[str]) -> dict:
    """S-pair / S-dose: 新規腕どうし・親走との init/教師/入力列/flip の bit 一致。

    対応は **seed ごとのハッシュ**で取る（位置合わせではない）。用量が 1 点なので
    ``running_mean``（オラクルオフセット）も全腕で一致していなければならない。
    """
    S = cfg["sanity"]
    steps = int(S["s_pair_steps"])
    every = int(cfg["common"]["lop_every"])
    seeds = [int(v) for v in cfg["common"]["seeds"]]
    init, final, streams, dose_rows = {}, {}, {}, []
    per_seed: dict[str, dict[int, dict]] = {}
    flip0: dict[str, dict[int, np.ndarray]] = {}
    for arm in arms:
        c = copy.deepcopy(cfg)
        st = setup_arm_dial(c, _arm(c, arm), "cpu")
        init[arm] = _init_hashes(st)
        per_seed[arm] = {int(run["seed"]): _seed_state_hashes_p1(st, ri)
                         for ri, run in enumerate(st["runs"])}
        flip0[arm] = {int(run["seed"]):
                      st["env"].flip_state[ri].detach().cpu().numpy().astype(np.float32)
                      for ri, run in enumerate(st["runs"])}
        stream = StreamDigest()

        def dose_probe(state: dict, step: int, arm_name: str = arm) -> None:
            if state.get("target_mu_norm") is not None:
                _refresh_fixed_offset(state)
            stats = _input_stats(state)
            errors = stats["relative_error"].detach().cpu().numpy()
            dose_rows.append(dict(arm=arm_name, step=int(step),
                                  target_mu_norm=state.get("target_mu_norm"),
                                  max_relative_error=float(errors.max())))

        print(f"[S-pair/S-dose] {arm} {steps:,} steps", flush=True)
        train_arm_gate(st, dose_probe, range(0, steps + 1, every), steps,
                       outdir, [], stream_hook=stream)
        final[arm], streams[arm] = _env_hashes(st), stream.digest()

    reference, differences = arms[0], []
    for arm in arms[1:]:
        for key, value in init[reference].items():
            if init[arm].get(key) != value:
                differences.append(dict(arm=arm, where=f"init.{key}"))
        for key, value in final[reference].items():
            if final[arm].get(key) != value:
                differences.append(dict(arm=arm, where=f"final.{key}"))
        for key in ("x", "y", "n"):
            if streams[arm][key] != streams[reference][key]:
                differences.append(dict(arm=arm, where=f"stream.{key}"))
        for seed in seeds:
            for key, value in per_seed[reference][seed].items():
                if per_seed[arm][seed].get(key) != value:
                    differences.append(dict(arm=arm, seed=seed,
                                            where=f"seed_hash.{key}"))

    parent = Path(ROOT) / S["s_pair_reference"] / "logs"
    ref_arm = str(S["s_pair_reference_arm"])
    parent_rows, parent_missing = [], []
    for arm in arms:
        for seed in seeds:
            path = parent / f"{ref_arm}_seed{seed}.npz"
            if not path.exists():
                parent_missing.append(str(path))
                continue
            with np.load(path, allow_pickle=False) as z:
                ref_flip = z["flip_state"][0].copy()
            same = bool(np.array_equal(flip0[arm][seed], ref_flip))
            parent_rows.append(dict(arm=arm, reference_arm=ref_arm, seed=seed,
                                    flip_state_equal=same))
            if not same:
                differences.append(dict(arm=arm, seed=seed, where="parent.flip_state"))

    tol = float(S["s_dose_rel_tol"])
    dose_fail = [r for r in dose_rows if r["target_mu_norm"] is not None
                 and float(r["max_relative_error"]) > tol]
    return dict(
        spair=dict(pass_=bool(not differences and not parent_missing),
                   reference=reference, arms=list(arms), steps=steps,
                   match_by="seed_init_hash", differences=differences,
                   parent_flip_rows=parent_rows, parent_missing=parent_missing,
                   caveat="init/teacher/input realization only; trajectories "
                          "diverge after step 1"),
        sdose=dict(pass_=not dose_fail, tolerance=tol, n_probes=len(dose_rows),
                   failures=dose_fail))


def _s_taut(cfg: dict, outdir: Path) -> dict:
    """S-taut: 未フィット率が介入で定義上恒真になっていないこと＋判定表の検算。"""
    c = copy.deepcopy(cfg)
    c["common"]["seeds"] = [0, 1]
    values, hashes = {}, {}
    for arm in ("S_b1_1216", "G_b1_1216", "LR_a0p001_1216"):
        st = setup_arm_dial(c, _arm(c, arm), "cpu")
        train_arm_gate(st, lambda *_: None, [], 2000, outdir, [])
        rec, _ = exact_layer_record_elu(st, SIGMA_TOL)
        values[arm] = rec["run"]["unfit"].detach().cpu().numpy().tolist()
        hashes[arm] = {k: _sha_array(v) for k, v in st["net"].state_dict().items()}
    changes_state = (hashes["S_b1_1216"] != hashes["G_b1_1216"]
                     and hashes["S_b1_1216"] != hashes["LR_a0p001_1216"])
    finite = all(np.isfinite(np.asarray(v)).all() and (np.asarray(v) > 0).all()
                 for v in values.values())
    G = _P(cfg)
    mutants = {
        "relu_side": _v1_label(G, "present", "present"),
        "soft_side": _v1_label(G, "zero", "zero"),
        "split_s": _v1_label(G, "present", "zero"),
        "split_g": _v1_label(G, "zero", "present"),
        "partial": _v1_label(G, "mid", "present"),
    }
    expected = dict(relu_side="SOFT_GATES_RELU_SIDE", soft_side="SOFT_GATES_SOFT_SIDE",
                    split_s="SPLIT_SILU_GELU", split_g="SPLIT_SILU_GELU",
                    partial="PARTIAL")
    v2_mutants = {
        "reversal": _v2_label(G, [dict(ci_lo=-1.0, ci_hi=-0.5)], [10, 0], [], []),
        "flat": _v2_label(G, [dict(ci_lo=-0.05, ci_hi=0.05)], [0, 0],
                          [dict(ci_lo=-0.05, ci_hi=0.05)], [0, 0]),
        "monotone": _v2_label(G, [dict(ci_lo=0.5, ci_hi=1.0)], [0, 10],
                              [dict(ci_lo=0.5, ci_hi=1.0)], [0, 10]),
        "partial": _v2_label(G, [dict(ci_lo=-0.05, ci_hi=0.05)], [3, 1],
                             [dict(ci_lo=-0.05, ci_hi=0.05)], [3, 1]),
    }
    v2_expected = dict(reversal="REVERSAL", flat="FLAT_IN_RANGE",
                       monotone="MONOTONE_TOWARD_RELU", partial="PARTIAL")
    return dict(pass_=bool(changes_state and finite and mutants == expected
                           and {k: v[0] for k, v in v2_mutants.items()} == v2_expected),
                activation_changes_state=changes_state,
                unfit_finite_positive=finite, short_run_unfit=values,
                v1_mutants=mutants, v1_expected=expected,
                v2_mutants={k: v[0] for k, v in v2_mutants.items()},
                v2_expected=v2_expected)


def _endpoint_columns_unchanged(cfg: dict, ref_rel: str, want_sha: str) -> dict:
    """転記する列が provenance 記録時の版から 1 バイトも動いていないことの確認。

    親走の ``verdict.csv`` は provenance 記録後に別 commit で再生成されうる
    （``--analyze-only`` は provenance を書き直さない。実際 `gate_dose_0830` は
    N5 の「非 ReLU 腕で `strict_dead` 列を空にする」修正で再生成されている）。
    ファイルのハッシュが合わないこと自体は事故とは限らないが、**本走が転記する列**
    が動いていたら事故である。``provenance.output_sha256`` に一致する版を履歴から
    探し出し、その blob と現行版を列単位で突き合わせる。
    ``bwd_leak_0902`` の同名関数と同じ手続きで、対照の顔ぶれだけが違う。
    """
    import hashlib
    import io

    columns = list(cfg["controls"]["endpoint_columns"])
    try:
        revs = subprocess.check_output(
            ["git", "log", "--format=%H", "--", ref_rel], cwd=ROOT,
            text=True).split()
    except (OSError, subprocess.CalledProcessError) as exc:
        return dict(checked=False, reason=f"git log failed: {exc}")
    blob, found_at = None, None
    for rev in revs:
        try:
            raw = subprocess.check_output(["git", "show", f"{rev}:{ref_rel}"],
                                          cwd=ROOT)
        except (OSError, subprocess.CalledProcessError):
            continue
        if hashlib.sha256(raw).hexdigest() == want_sha:
            blob, found_at = raw.decode("utf-8"), rev
            break
    if blob is None:
        return dict(checked=False,
                    reason="no commit in history matches the recorded sha256",
                    revisions_searched=len(revs))
    then = {r["arm"]: r for r in csv.DictReader(io.StringIO(blob))}
    now = {r["arm"]: r for r in csv.DictReader(
        (Path(ROOT) / ref_rel).read_text(encoding="utf-8").splitlines())}
    differing, missing = [], []
    for arm in CONTROL_ORDER:
        if arm not in then or arm not in now:
            missing.append(arm)
            continue
        for column in columns:
            if then[arm].get(column) != now[arm].get(column):
                differing.append(dict(arm=arm, column=column))
    changed_columns = sorted({
        key for arm in then for key in set(then[arm]) | set(now.get(arm, {}))
        if arm in now and then[arm].get(key) != now[arm].get(key)})
    return dict(checked=True, provenance_era_commit=found_at,
                columns_transcribed=columns, arms=list(CONTROL_ORDER),
                differing=differing, missing=missing,
                columns_that_changed_anywhere=changed_columns,
                unchanged=bool(not differing and not missing))


def _s_ref(cfg: dict) -> dict:
    """S-ref: 対照として読む親走の出力が親 ``provenance.json`` と一致すること。"""
    ref_rel = str(cfg["controls"]["reference_run"])
    ref_dir = (Path(ROOT) / ref_rel).resolve()
    prov_path = ref_dir / "provenance.json"
    if not prov_path.exists():
        return dict(pass_=False, reason="missing provenance", path=str(prov_path))
    prov = json.loads(prov_path.read_text(encoding="utf-8"))
    recorded = dict(prov.get("output_sha256", {}))
    parent_sha = prov.get("git_hash")
    read_files = ["verdict.csv", "floor_calibration.csv"] + [
        f"logs/{arm}_seed{seed}.npz" for arm in CONTROL_ORDER for seed in range(10)]
    checked, mismatches, missing = 0, [], []
    for name, want in recorded.items():
        path = ref_dir / name
        if not path.exists():
            missing.append(name)
            continue
        checked += 1
        if _sha_file(path) != want:
            mismatches.append(name)
    absent = [n for n in read_files if not (ref_dir / n).exists()]
    ckpts = [f"ckpts/{arm}_step{int(cfg['controls']['m_minus_checkpoint_step'])}.pt"
             for arm in CONTROL_ORDER]
    ckpt_absent = [n for n in ckpts if not (ref_dir / n).exists()]
    read_mismatches = [n for n in mismatches if n in read_files]
    column_check = None
    if "verdict.csv" in read_mismatches and recorded.get("verdict.csv"):
        column_check = _endpoint_columns_unchanged(
            cfg, f"{ref_rel}/verdict.csv", recorded["verdict.csv"])
        if column_check.get("unchanged"):
            read_mismatches = [n for n in read_mismatches if n != "verdict.csv"]
    remote_ok = None
    if cfg["sanity"]["s_ref_remote_check"] and parent_sha:
        try:
            out = subprocess.run(["git", "branch", "-r", "--contains", parent_sha],
                                 cwd=ROOT, capture_output=True, text=True, timeout=60)
            remote_ok = bool(out.returncode == 0 and out.stdout.strip())
        except (OSError, subprocess.SubprocessError):
            remote_ok = None
    return dict(pass_=bool(not read_mismatches and not missing and not absent),
                reference=str(ref_dir), parent_git_hash=parent_sha,
                parent_on_remote=remote_ok, files_checked=checked,
                hash_mismatches_on_read_files=read_mismatches,
                hash_mismatches_on_unread_files=[n for n in mismatches
                                                 if n not in read_files],
                verdict_column_check=column_check,
                recorded_but_missing=missing, required_but_absent=absent,
                checkpoints_absent=ckpt_absent,
                note="logs/*.npz and ckpts/*.pt are gitignored; the control side of "
                     "V3 and §5.5 is not reproducible from a fresh clone. m_minus for "
                     "the controls comes from the 5M checkpoint (single record point, "
                     "window label final_step5000000), not from the 491-500 window.")


def _s_floor_inheritance(cfg: dict) -> dict:
    reference = (Path(ROOT) / cfg["controls"]["reference_run"]
                 / "floor_calibration.csv")
    if not reference.exists():
        return dict(pass_=False, reference=str(reference), reason="missing")
    data = np.genfromtxt(reference, delimiter=",", names=True)
    values = np.unique(np.asarray(data["calibrated_floor"], dtype=np.float64))
    configured = float(cfg["phase1"]["unfit_floor"])
    return dict(pass_=bool(values.size == 1 and values[0] == configured
                           and cfg["phase1"]["recalibrate_floor"] is False),
                reference=str(reference), reference_values=values.tolist(),
                configured=configured, recalibrated=False)


def _s_ci_selftest(cfg: dict) -> dict:
    P = cfg["phase1"]
    n = len(cfg["common"]["seeds"])
    draws = np.random.default_rng(int(P["bootstrap_seed"])).integers(
        0, n, size=(int(P["bootstrap_B"]), n))
    result = _ci_components(np.zeros(n), draws, "median",
                            float(P["degenerate_se_tol"]),
                            float(P["degenerate_frac_max"]),
                            float(P["degenerate_width_ratio_max"]))
    return dict(pass_=bool(result["ci_degenerate"]), result=result)


def preflight(cfg: dict, outdir: Path) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    checks: dict[str, dict] = {"S1_omp": require_omp(cfg)}
    print("[S-dial] registered valley/freeze depths vs numeric roots", flush=True)
    checks["S_dial"] = _s_dial(cfg)
    print("[S-fd] SiLU/GELU closed-form backward vs central difference", flush=True)
    checks["S_fd"] = _s_fd(cfg)
    print("[S-num] finiteness and float32 saturation depths", flush=True)
    checks["S_num"] = _s_num(cfg)
    print("[S-limit] beta -> inf (SiLU/GELU unit test)", flush=True)
    checks["S_limit_smooth"] = _s_limit_smooth(cfg)
    print("[S-limit] ELU alpha -> 0 is the ReLU path", flush=True)
    checks["S_elu_limit"] = _s_elu_limit(cfg, outdir / "slimit_elu")
    print("[S-ref] parent output hashes", flush=True)
    checks["S_ref"] = _s_ref(cfg)
    print("[S-mob] new loggers against p_hat and (M+B)*denom", flush=True)
    checks["S_mob"] = _s_mob(cfg, outdir / "smob")
    print("[S-log-b] logger trajectory neutrality", flush=True)
    checks["S_log_b"] = _s_log_b(cfg, outdir / "slogb")
    pair = _s_pair_and_dose(cfg, outdir / "spair", list(ARM_ORDER))
    checks["S_pair"], checks["S_dose"] = pair["spair"], pair["sdose"]
    print("[S-taut] endpoint is not tautological", flush=True)
    checks["S_taut"] = _s_taut(cfg, outdir / "staut")
    checks["S6_floor_inherited"] = _s_floor_inheritance(cfg)
    checks["S_CI_degeneracy"] = _s_ci_selftest(cfg)
    result = dict(pass_=bool(all(v.get("pass_") for v in checks.values())), **checks)
    (outdir / "preflight.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    for name, value in checks.items():
        print(f"[{name}] {'PASS' if value.get('pass_') else 'FAIL'}", flush=True)
    if not result["pass_"]:
        raise SanityError(f"preflight failed: "
                          f"{[k for k, v in checks.items() if not v.get('pass_')]}")
    return result


def s_par(cfg: dict, cfg_path: Path, outdir: Path) -> dict:
    """S-par: 直列投入と腕プロセス並列投入が bit 一致すること（spec §6）。

    同じ腕を (a) この過程で直列に、(b) 別プロセスを 3 本同時に走らせたうちの 1 本
    として回し、``logs/*.npz`` の全列と state hash を突き合わせる。
    """
    S = cfg["sanity"]
    steps = int(S["s_par_steps"])
    target = str(S["s_par_arm"])
    seeds = [int(v) for v in cfg["common"]["seeds"]]
    outdir.mkdir(parents=True, exist_ok=True)
    serial_dir = outdir / "serial"
    parallel_dir = outdir / "parallel"
    got = _run_arm(cfg, target, "cpu", serial_dir, seeds, steps)
    if got["status"] != "COMPLETE":
        return dict(pass_=False, reason="serial run did not complete", detail=got)
    companions = [a for a in ARM_ORDER if a != target][:2]
    procs = []
    env = dict(os.environ, OMP_NUM_THREADS="1")
    for arm in [target] + companions:
        procs.append(subprocess.Popen(
            [sys.executable, "-m", "src.gate_dial_0902", "--config", str(cfg_path),
             "--arm", arm, "--steps", str(steps), "--outdir", str(parallel_dir)],
            cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT))
    codes = [p.wait() for p in procs]
    if any(codes):
        tails = [p.stdout.read().decode("utf-8", "replace")[-2000:] for p in procs]
        return dict(pass_=False, reason="a parallel worker failed",
                    returncodes=codes, output=tails)
    differences, rows = [], []
    for seed in seeds:
        a = serial_dir / "logs" / f"{target}_seed{seed}.npz"
        b = parallel_dir / "logs" / f"{target}_seed{seed}.npz"
        with np.load(a, allow_pickle=False) as za, np.load(b, allow_pickle=False) as zb:
            keys_a, keys_b = set(za.files), set(zb.files)
            if keys_a != keys_b:
                differences.append(dict(seed=seed, where="columns"))
            for key in sorted(keys_a & keys_b):
                if _sha_array(za[key]) != _sha_array(zb[key]):
                    differences.append(dict(seed=seed, column=key))
            rows.append(dict(seed=seed, columns=len(keys_a),
                             state_hash_equal=bool(str(za["state_hash_final"])
                                                   == str(zb["state_hash_final"]))))
    result = dict(pass_=not differences, arm=target, steps=steps,
                  parallel_workers=[target] + companions, rows=rows,
                  differences=differences,
                  note="arm-level process parallelism does not touch the runner's "
                       "seed loop, so determinism is preserved")
    (outdir / "s_par.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")
    print(f"S-par {'PASS' if result['pass_'] else 'FAIL'}", flush=True)
    return result


# ---------------------------------------------------------------------------
# 判定ラベル（spec §5.1・§5.3）
# ---------------------------------------------------------------------------
def _onset_state(onsets: list[int], zero_max: int, present_min: int) -> str:
    if not onsets:
        return "missing"
    if all(int(v) <= zero_max for v in onsets):
        return "zero"
    if any(int(v) >= present_min for v in onsets):
        return "present"
    return "mid"


def _v1_label(G: dict, s_state: str, g_state: str) -> str:
    return str(G["v1_map"][f"{s_state}_{g_state}"])


def _v2_label(G: dict, all_contrasts: list[dict], all_onsets: list[int],
              new_contrasts: list[dict], new_onsets: list[int]) -> tuple[str, list[str]]:
    """族の V2 ラベルと、**条件を満たしていた行すべて**（追補・bwd_leak 追補 7 に倣う）。

    ``all_*`` は対照を含む梯子の全隣接対、``new_*`` は新規腕どうしの対と新規腕の
    発症数。spec §5.3 の表には順序が書かれていないので、config の
    ``v2_label_order``（REVERSAL → FLAT_IN_RANGE → MONOTONE → PARTIAL）で決める。
    """
    margin = float(G["v2_margin"])
    drop = int(G["v2_onset_drop_reversal_min"])
    below = [c for c in all_contrasts if c["ci_hi"] < -margin]
    onset_drop = any(all_onsets[i] - all_onsets[i + 1] >= drop
                     for i in range(len(all_onsets) - 1))
    non_decreasing = all(all_onsets[i] <= all_onsets[i + 1]
                         for i in range(len(all_onsets) - 1))
    hits = []
    if below or onset_drop:
        hits.append("REVERSAL")
    if (new_contrasts
            and all(-margin <= c["ci_lo"] and c["ci_hi"] <= margin
                    for c in new_contrasts)
            and all(int(v) == 0 for v in new_onsets)):
        hits.append("FLAT_IN_RANGE")
    if not below and non_decreasing:
        hits.append("MONOTONE_TOWARD_RELU")
    for label in G["v2_label_order"]:
        if label == "PARTIAL":
            return "PARTIAL", hits or ["PARTIAL"]
        if label in hits:
            return label, hits
    return "PARTIAL", hits or ["PARTIAL"]


def _p5_label(G: dict, ci: dict, soft_arm: str) -> tuple[str, bool]:
    """spec §5.2 の書かれた順に判定する。CI が丸ごと 0 の下ならフラグを立てる。"""
    margin = float(G["p5_equivalence_margin"])
    lo, hi = float(ci["percentile_ci_lo"]), float(ci["percentile_ci_hi"])
    below = bool(hi < 0.0)
    suffix = f"_{soft_arm}" if G["p5_labels"].get("suffix_is_soft_end_arm") else ""
    if lo >= -margin and hi <= margin:
        return str(G["p5_labels"]["equivalent"]) + suffix, below
    if lo > 0.0:
        return str(G["p5_labels"]["short_of_soft"]) + suffix, below
    return str(G["p5_labels"]["inconclusive"]), below


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
def _load_controls(cfg: dict) -> dict:
    """対照の主 endpoint を親走の ``verdict.csv`` から**転記**する（再計算しない）。"""
    path = Path(ROOT) / cfg["controls"]["reference_run"] / "verdict.csv"
    floor = float(cfg["phase1"]["unfit_floor"])
    out = {}
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            if row["arm"] not in CONTROL_ORDER:
                continue
            u5 = np.maximum(np.asarray(json.loads(row["U_5m_seed_values"]),
                                       dtype=np.float64), floor)
            u1 = np.maximum(np.asarray(json.loads(row["U_1m_seed_values"]),
                                       dtype=np.float64), floor)
            out[row["arm"]] = dict(
                u_5m=u5, u_1m=u1, log_u_5m=np.log10(u5), log_u_1m=np.log10(u1),
                n_onset_5m=int(row["n_onset_5m"]), n_onset_1m=int(row["n_onset_1m"]),
                source=str(path))
    missing = [a for a in CONTROL_ORDER if a not in out]
    if missing:
        raise SanityError(f"control arms missing from {path}: {missing}")
    return out


def _load_new_arm(cfg: dict, outdir: Path, arm: str) -> dict:
    data = _load_arm(cfg, outdir, arm)
    P = cfg["phase1"]
    return {"data": data,
            "5M": _window(data, cfg, list(P["late_tasks_5m"])),
            "1M": _window(data, cfg, list(P["window_1m_tasks"])),
            "early": _window(data, cfg, list(P["early_tasks"]))}


def _draws(cfg: dict) -> np.ndarray:
    P = cfg["phase1"]
    n = len(cfg["common"]["seeds"])
    rng = np.random.default_rng(int(P["bootstrap_seed"]))
    return rng.integers(0, n, size=(int(P["bootstrap_B"]), n))


def _ci(cfg: dict, values: np.ndarray, draws: np.ndarray) -> dict:
    P = cfg["phase1"]
    return _ci_components(np.asarray(values, dtype=np.float64), draws, "median",
                          float(P["degenerate_se_tol"]),
                          float(P["degenerate_frac_max"]),
                          float(P["degenerate_width_ratio_max"]))


def _sign_test(values: np.ndarray) -> dict:
    """seed 別符号検定（REPORT_ONLY）。較正定数が要らない。"""
    values = np.asarray(values, dtype=np.float64)
    pos, neg = int((values > 0).sum()), int((values < 0).sum())
    ties = int((values == 0).sum())
    n = pos + neg
    if n == 0:
        return dict(n_positive=pos, n_negative=neg, n_ties=ties,
                    p_two_sided=float("nan"))
    tail = sum(math.comb(n, i) for i in range(min(pos, neg) + 1)) / (2.0 ** n)
    return dict(n_positive=pos, n_negative=neg, n_ties=ties,
                p_two_sided=float(min(2.0 * tail, 1.0)))


# ---------------------------------------------------------------------------
# §5.3 発症時刻 k*
# ---------------------------------------------------------------------------
def _rolling_window_unfit(step: np.ndarray, unfit: np.ndarray, cfg: dict) -> dict:
    """``U^(10)_k`` = タスク k-9..k のタスク終端記録点の unfit 平均。

    ★ spec §5.3 の S-mask は「記録点数が各 k で 100」と書くが、宿主の
    ``_window_indices`` は ``step % task_period == 0`` だけを拾う（**タスク終端
    10 点**）。spec 自身が要求する「``U^(10)_100`` / ``U^(10)_500`` が 1M / 5M の
    ``U_k`` に一致」は終端のみの定義でしか成り立たず、committed 対照の ``U_k`` も
    その定義で作られているので、逐語継承（終端のみ）を採る。
    """
    P = cfg["phase1"]
    period = int(P["task_period"])
    width = int(_P(cfg)["onset_time"]["window_tasks"])
    ends = np.flatnonzero((step > 0) & (step % period == 0))
    tasks = (step[ends] // period).astype(np.int64)
    values = np.asarray(unfit, dtype=np.float64)[ends]
    order = np.argsort(tasks, kind="mergesort")
    tasks, values = tasks[order], values[order]
    csum = np.cumsum(values, axis=0)
    out_k, out_u = [], []
    index = {int(t): i for i, t in enumerate(tasks)}
    for k in range(width, int(tasks.max()) + 1):
        lo, hi = index.get(k - width + 1), index.get(k)
        if lo is None or hi is None or hi - lo + 1 != width:
            continue
        total = csum[hi] - (csum[lo - 1] if lo else 0.0)
        out_k.append(k)
        out_u.append(total / width)
    return dict(k=np.asarray(out_k, dtype=np.int64),
                u=np.asarray(out_u, dtype=np.float64), records_per_window=width)


def _onset_times(cfg: dict, step: np.ndarray, unfit: np.ndarray) -> dict:
    """seed 別 ``k* = min{k >= 10 : U^(10)_k >= 0.05}``。無ければ 500 で打ち切り。"""
    P, G = cfg["phase1"], _P(cfg)["onset_time"]
    threshold = float(P["onset_threshold"])
    censor = int(G["censor_at"])
    rolled = _rolling_window_unfit(step, unfit, cfg)
    ks, us = rolled["k"], rolled["u"]
    out = []
    for j in range(us.shape[1]):
        hit = np.flatnonzero((us[:, j] >= threshold) & (ks >= int(G["k_min"])))
        if hit.size:
            out.append(dict(k_star=int(ks[hit[0]]), censored=0))
        else:
            out.append(dict(k_star=censor, censored=1))
    return dict(rows=out, rolled_k=ks, rolled_u=us,
                records_per_window=rolled["records_per_window"])


def _kaplan_meier(k_star: list[int], censored: list[int], censor_at: int) -> list[dict]:
    """未発症＝生存の KM 曲線（率）。``n_onset``（総和）と並べる（教訓⑰）。"""
    order = sorted(range(len(k_star)), key=lambda i: (k_star[i], censored[i]))
    at_risk, survival, rows = len(k_star), 1.0, []
    i = 0
    while i < len(order):
        t = k_star[order[i]]
        events = deaths = 0
        while i < len(order) and k_star[order[i]] == t:
            if not censored[order[i]]:
                deaths += 1
            events += 1
            i += 1
        if deaths and at_risk:
            survival *= 1.0 - deaths / at_risk
        rows.append(dict(k=int(t), at_risk=int(at_risk), events=int(deaths),
                         censored=int(events - deaths), survival=float(survival)))
        at_risk -= events
    return rows


# ---------------------------------------------------------------------------
# §5.4 V3 の材料（ユニット別）
# ---------------------------------------------------------------------------
def _tail_index(cfg: dict, step: np.ndarray) -> np.ndarray:
    P = cfg["phase1"]
    return _window_indices(step, int(P["task_period"]), list(P["late_tasks_5m"]))


def _unit_summary(cfg: dict, path: Path, geo: dict) -> dict:
    """1 seed 分の末尾窓ユニット量（m⁻・沈下率・谷率・凍結率・深さ十分位）。"""
    with np.load(path, allow_pickle=False) as z:
        step = z["step"].astype(np.int64)
        idx = _tail_index(cfg, step)
        p_hat = z["layer1_p_hat"][idx].astype(np.float64)
        zbar = z["layer1_zbar"][idx].astype(np.float64)
        has_mob = "layer1_mob" in z.files
        mob = z["layer1_mob"][idx].astype(np.float64) if has_mob else None
        zmax = (z["layer1_zmax"][idx].astype(np.float64)
                if "layer1_zmax" in z.files else None)
        v_unit = (z["layer1_v_unit"][idx].astype(np.float64)
                  if "layer1_v_unit" in z.files else None)
    submerged = (zmax <= 0.0) if zmax is not None else (p_hat == 0.0)
    u_star, u_fr = geo["u_star_numeric"], geo["u_fr_numeric"]
    beyond = ((zmax <= -u_star) if (zmax is not None and np.isfinite(u_star))
              else None)
    frozen = zbar <= -u_fr if np.isfinite(u_fr) else np.zeros_like(submerged)
    out = dict(
        n_records=int(len(idx)),
        submerged_frac=float(submerged.mean()),
        beyond_valley_frac=(float(beyond.mean()) if beyond is not None
                            else float("nan")),
        frozen_frac=float(frozen.mean()),
        frozen_or_beyond_frac=float((frozen | (beyond if beyond is not None
                                               else np.zeros_like(frozen))).mean()),
        m_minus=(float(np.median(mob[submerged]))
                 if (mob is not None and submerged.any()) else float("nan")),
        abs_v_median=(float(np.median(np.abs(v_unit))) if v_unit is not None
                      else float("nan")),
        submerged_abs_v_median=(float(np.median(np.abs(v_unit[submerged])))
                                if (v_unit is not None and submerged.any())
                                else float("nan")),
        strict_dead_frac=float((p_hat == 0.0).mean()),
        submerged_equals_strict_dead=bool(np.array_equal(submerged, p_hat == 0.0)))
    depths = -zbar[submerged] if submerged.any() else np.asarray([])
    out["depth_deciles"] = ([float(v) for v in np.quantile(depths,
                                                           np.arange(1, 10) / 10.0)]
                            if depths.size else [float("nan")] * 9)
    out["depth_median"] = float(np.median(depths)) if depths.size else float("nan")
    return out


def _m_minus_from_checkpoint(cfg: dict, arm: str, seed_index: int,
                             geo: dict) -> dict:
    """対照の m⁻ を 5M チェックポイントから 1 点だけ復元する。

    committed logs にはユニット別 ``mob`` が無い（P-3 の指摘そのもの）。窓は
    **``final_step5000000`` の 1 点**であって末尾窓 491-500 ではない。引用時に
    必ず窓を添えること（教訓⑫）。
    """
    step = int(cfg["controls"]["m_minus_checkpoint_step"])
    template = str(cfg["controls"]["m_minus_checkpoint"])
    path = Path(ROOT) / template.format(arm=arm, step=step)
    if not path.exists():
        return dict(status="INSUFFICIENT_DATA", reason="missing checkpoint",
                    path=str(path))
    blob = torch.load(path, map_location="cpu", weights_only=False)
    entry = dict(cfg["controls"]["arms"][arm])
    act = {"relu": "relu", "leaky_relu": "leaky_relu", "elu": "elu"}[
        str(entry["activation"])]
    net = VecMLPL(blob["net"]["v"].shape[0], [blob["net"]["v"].shape[1]],
                  blob["net"]["W"].shape[2], torch.Generator().manual_seed(0), "cpu")
    net.load_state(blob["net"])
    net.set_activation(act, float(entry["dial"]) if act != "relu" else 1.0,
                       "alpha_exp")
    with torch.no_grad():
        W, b = net.Ws[0].double(), net.bs[0].double()
        # full_support_ro と同じ 32 パターンの厳密支持を、チェックポイントの
        # flip_state から組み直す（env インスタンスは checkpoint に無い）。
        flip = blob["env"]["flip_state"]
        runs, free = int(flip.shape[0]), int(W.shape[2] - flip.shape[1])
        patterns = ((torch.arange(2 ** free)[:, None]
                     >> torch.arange(free)) & 1).to(flip.dtype)
        cur = torch.cat([flip[None].expand(patterns.shape[0], -1, -1),
                         patterns[:, None, :].expand(-1, runs, -1)],
                        dim=2).double()
        mean = blob["layer_means"][0]
        if list(blob["centered_layers"])[0]:
            cur = cur - mean.double()[None]
        z = torch.einsum("rhd,prd->prh", W, cur) + b
        mob = net.act_grad(z, net.act_fn(z)).mean(dim=0)
        zmax = z.amax(dim=0)
        submerged = zmax[seed_index] <= 0
        values = mob[seed_index][submerged]
        v_abs = net.v.double()[seed_index][submerged].abs()
    return dict(status="OK", window=str(cfg["controls"]["m_minus_checkpoint_window_label"]),
                m_minus=(float(values.median()) if values.numel() else float("nan")),
                submerged_frac=float(submerged.double().mean()),
                submerged_abs_v_median=(float(v_abs.median()) if v_abs.numel()
                                        else float("nan")),
                source=str(path))


def _revival_counts(path: Path) -> dict:
    """``p_hat`` が 0 -> 正 になった件数。同一タスク内 / 境界越えを分ける。"""
    with np.load(path, allow_pickle=False) as z:
        p = z["layer1_p_hat"]
        step = z["step"].astype(np.int64)
        flip = z["flip_state"]
        period = int(z["task_period"])
    dead = p == 0.0
    revived = dead[:-1] & ~dead[1:]
    same_task = (step[:-1] // period) == (step[1:] // period)
    flip_same = (flip[:-1] == flip[1:]).all(axis=1)
    within = same_task & flip_same
    return dict(
        events_within_task=int(revived[within].sum()),
        events_across_boundary=int(revived[~within].sum()),
        units_within_task=int(revived[within].any(axis=0).sum()),
        units_across_boundary=int(revived[~within].any(axis=0).sum()),
        opportunities_within_task=int(dead[:-1][within].sum()),
        opportunities_across_boundary=int(dead[:-1][~within].sum()),
        n_units=int(p.shape[1]), n_records=int(p.shape[0]))


def _s_series(cfg: dict, path: Path) -> dict:
    """末尾窓のユニット別 ``s = M + B``（縮退 (ii)(iii) を併記して出す）。"""
    with np.load(path, allow_pickle=False) as z:
        step = z["step"].astype(np.int64)
        idx = _tail_index(cfg, step)
        M = z["layer1_M"][idx].astype(np.float64)
        B = z["layer1_B"][idx].astype(np.float64)
        denom = z["layer1_denom"][idx].astype(np.float64)
        zbar = z["layer1_zbar"][idx].astype(np.float64)
    with np.errstate(invalid="ignore"):
        out = dict(
            median_s=float(np.nanmean(np.nanmedian(M + B, axis=1))),
            median_M=float(np.nanmean(np.nanmedian(M, axis=1))),
            median_B=float(np.nanmean(np.nanmedian(B, axis=1))),
            median_denom=float(np.nanmean(np.nanmedian(denom, axis=1))),
            median_zbar=float(np.nanmean(np.nanmedian(zbar, axis=1))),
            n_na_M=int(np.isnan(M).sum()), n_na_B=int(np.isnan(B).sum()),
            n_records=int(len(idx)))
    out["median_M_plus_median_B"] = out["median_M"] + out["median_B"]
    return out


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def analyze(cfg: dict, outdir: Path, arms: list[str], stage: str, sanity: dict,
            elapsed: dict, divergences: dict) -> dict:
    seeds = [int(v) for v in cfg["common"]["seeds"]]
    P, G = cfg["phase1"], _P(cfg)
    draws = _draws(cfg)
    controls = _load_controls(cfg)
    expected = dict(G["control_expected_onset_5m"])
    got = {a: controls[a]["n_onset_5m"] for a in CONTROL_ORDER}
    if got != expected:
        raise SanityError(
            f"committed control onsets differ from the preregistration: "
            f"expected {expected}, got {got}; the result must not be read")

    complete = [a for a in arms if a not in divergences]
    windows = {a: _load_new_arm(cfg, outdir, a) for a in complete}
    threshold = float(P["onset_threshold"])
    onset = {w: {a: int(np.sum(windows[a][w]["raw_u"] >= threshold))
                 for a in complete} for w in ("1M", "5M")}

    # --- S-cap（腕ごと・登録） ---
    cap = {}
    for arm in complete:
        good = int(np.sum(windows[arm]["early"]["raw_u"]
                          < float(cfg["sanity"]["s_cap_threshold"])))
        cap[arm] = dict(seeds_below_threshold=good,
                        required=int(cfg["sanity"]["s_cap_min_seeds"]),
                        status=("OK" if good >= int(cfg["sanity"]["s_cap_min_seeds"])
                                else str(cfg["sanity"]["s_cap_label"])),
                        early_median_log10_U=float(
                            np.median(windows[arm]["early"]["log_u"])))
    capacity_undefined = {a for a, v in cap.items() if v["status"] != "OK"}

    def log_u(arm: str, window: str) -> np.ndarray | None:
        if arm in controls:
            return controls[arm][f"log_u_{window.lower()}"]
        if arm in windows:
            return windows[arm][window]["log_u"]
        return None

    def n_onset(arm: str, window: str) -> int | None:
        if arm in controls:
            return controls[arm][f"n_onset_{window.lower()}"]
        return onset[window].get(arm)

    # --- V1（§5.1） ---
    v1_arms = [str(v) for v in G["v1_arms"]]
    zero_max, present_min = int(G["onset_zero_max"]), int(G["onset_present_min"])
    v1_states = {}
    for arm in v1_arms:
        value = n_onset(arm, "5M")
        if arm in capacity_undefined:
            # S-cap（spec §6）: 満たさない腕は V1・V2 の n_onset から外す。
            # 「発症しなかった」ではなく「発症が定義されない」。水準は報告する。
            v1_states[arm] = str(cfg["sanity"]["s_cap_label"])
        elif value is None:
            v1_states[arm] = "missing"
        else:
            v1_states[arm] = _onset_state([value], zero_max, present_min)
    v1_capacity_blocked = [a for a in v1_arms if a in capacity_undefined]
    if any(a in divergences for a in v1_arms):
        v1 = str(G["numeric_divergence"]["inconclusive_label"])
    elif v1_capacity_blocked:
        # 登録された 4 ラベルのどれでもない。S-cap の除外規則の帰結である。
        v1 = str(cfg["sanity"]["s_cap_label"])
    elif any(v == "missing" for v in v1_states.values()):
        v1 = "NOT_RUN"
    else:
        v1 = _v1_label(G, v1_states[v1_arms[0]], v1_states[v1_arms[1]])
    v1_developed = [a for a in v1_arms if v1_states.get(a) == "present"]

    # --- E2 水準（§5.2） ---
    contrasts: dict[str, dict] = {}

    def add_contrast(kind: str, high: str, low: str, soft_arm: str = "") -> None:
        label = f"{kind}:{high}_minus_{low}"
        hi_v, lo_v = log_u(high, "5M"), log_u(low, "5M")
        if hi_v is None or lo_v is None:
            contrasts[label] = dict(
                kind=kind, high=high, low=low,
                status=(NUMERIC_DIVERGENCE if (high in divergences
                                               or low in divergences) else "NOT_RUN"))
            return
        values = np.asarray(hi_v) - np.asarray(lo_v)
        row = dict(kind=kind, high=high, low=low, status="OK", n_paired=len(values),
                   seed_values=values.tolist(), ci=_ci(cfg, values, draws),
                   sign_test=_sign_test(values),
                   cross_run=bool(low in controls or high in controls))
        if kind == "P5prime":
            row["label"], row["ci_below_zero"] = _p5_label(G, row["ci"], soft_arm)
            row["equivalence_margin"] = float(G["p5_equivalence_margin"])
            row["margin_recalibrated"] = False
            row["soft_end"] = soft_arm
        contrasts[label] = row

    baseline = str(G["p3prime_baseline"])
    for arm in arms:
        add_contrast("P3prime", arm, baseline)
    soft_by_family = dict(G["p5_soft_end_by_family"])
    for arm in arms:
        for soft in soft_by_family[_family(cfg, arm)]:
            add_contrast("P5prime", arm, soft, soft_arm=soft)

    # 位置比 rho（REPORT_ONLY）
    rho = {}
    relu_end = str(G["rho_relu_end"])
    min_denom = float(G["rho_min_abs_denominator_dex"])
    for arm in complete:
        for soft in soft_by_family[_family(cfg, arm)]:
            denom = log_u(relu_end, "5M") - log_u(soft, "5M")
            numer = log_u(arm, "5M") - log_u(soft, "5M")
            with np.errstate(divide="ignore", invalid="ignore"):
                values = np.where(np.abs(denom) >= min_denom, numer / denom, np.nan)
            key = f"{arm}_vs_{soft}"
            if np.isfinite(values).all():
                rho[key] = dict(status="OK", arm=arm, soft_end=soft,
                                seed_values=values.tolist(),
                                ci=_ci(cfg, values, draws))
            else:
                rho[key] = dict(status="INSUFFICIENT_DATA", arm=arm, soft_end=soft,
                                seed_values=values.tolist())

    # --- E3 発症時刻（§5.3） ---
    onset_rows, km_rows, mask_rows = [], [], []
    k_star_by_arm: dict[str, list[int]] = {}
    ref_logs = Path(ROOT) / cfg["controls"]["reference_run"] / "logs"
    for arm in complete + list(CONTROL_ORDER):
        source = outdir / "logs" if arm in complete else ref_logs
        paths = [source / f"{arm}_seed{s}.npz" for s in seeds]
        if not all(p.exists() for p in paths):
            continue
        steps_arr, unfit = None, []
        for path in paths:
            with np.load(path, allow_pickle=False) as z:
                steps_arr = z["step"].astype(np.int64)
                unfit.append(z["unfit"].astype(np.float64))
        got_times = _onset_times(cfg, steps_arr, np.stack(unfit, axis=1))
        k_star = [r["k_star"] for r in got_times["rows"]]
        censored = [r["censored"] for r in got_times["rows"]]
        k_star_by_arm[arm] = k_star
        for i, seed in enumerate(seeds):
            onset_rows.append(dict(arm=arm, seed=seed,
                                   is_control=int(arm in CONTROL_ORDER),
                                   source=("this_run" if arm in complete
                                           else str(source)),
                                   k_star=k_star[i], censored=censored[i]))
        for row in _kaplan_meier(k_star, censored,
                                 int(G["onset_time"]["censor_at"])):
            km_rows.append(dict(arm=arm, **row))
        # S-mask: U^(10)_100 / U^(10)_500 が窓の U_k に一致すること
        ks, us = got_times["rolled_k"], got_times["rolled_u"]
        for label, k in (("1M", 100), ("5M", 500)):
            hit = np.flatnonzero(ks == k)
            if not hit.size:
                continue
            rolled = us[hit[0]]
            reference = (windows[arm][label]["raw_u"] if arm in complete
                         else controls[arm][f"u_{label.lower()}"])
            mask_rows.append(dict(
                arm=arm, window=label, k=k,
                records_per_window=got_times["records_per_window"],
                max_abs_diff=float(np.abs(rolled - np.asarray(reference)).max()),
                is_control=int(arm in CONTROL_ORDER)))
    s_mask = dict(
        pass_=bool(mask_rows
                   and all(r["records_per_window"]
                           == int(P["window_records_per_10task_window"])
                           for r in mask_rows)
                   and all(r["max_abs_diff"] <= 1e-12 for r in mask_rows
                           if not r["is_control"])),
        rows=mask_rows,
        spec_literal_records_per_window=int(P["spec_literal_records_per_window"]),
        actual_records_per_window=int(P["window_records_per_10task_window"]),
        resolution=("the host's _window_indices keeps only step %% task_period == 0, "
                    "so a 10-task window has 10 records, not the 100 the spec's "
                    "S-mask text asserts. The spec's own requirement that U^(10)_100 "
                    "and U^(10)_500 equal the 1M/5M U_k, and the fact that the "
                    "committed controls' U_k were built that way, both force the "
                    "task-ends-only reading. A spec addendum is needed."),
        control_diff_note=("controls are compared against the transcribed "
                           "verdict.csv values, so a nonzero diff there would mean "
                           "the committed endpoint and the committed logs disagree"))

    # --- V2 族ごとの単調性（§5.3） ---
    ladders = dict(G["ladders"])
    v2: dict[str, dict] = {}
    for family, ladder in ladders.items():
        # S-cap 落ちの腕は n_onset が定義されないので梯子から外す（spec §6）。
        present = [a for a in ladder
                   if ((a in controls) or (a in complete))
                   and a not in capacity_undefined]
        dropped = [a for a in ladder if a not in present]
        dropped_reason = {a: ("capacity_undefined" if a in capacity_undefined
                              else NUMERIC_DIVERGENCE if a in divergences
                              else "not_run") for a in dropped}
        if len(present) < 2:
            v2[family] = dict(status="NOT_RUN", ladder=ladder, used=present,
                              dropped=dropped, dropped_reason=dropped_reason)
            continue
        adjacent, onsets = [], []
        for i in range(len(present) - 1):
            soft, hard = present[i], present[i + 1]
            values = np.asarray(log_u(hard, "5M")) - np.asarray(log_u(soft, "5M"))
            ci = _ci(cfg, values, draws)
            adjacent.append(dict(softer=soft, harder=hard, point=ci["point"],
                                 ci_lo=float(ci["percentile_ci_lo"]),
                                 ci_hi=float(ci["percentile_ci_hi"]),
                                 ci=ci, sign_test=_sign_test(values),
                                 seed_values=values.tolist(),
                                 cross_run=bool(soft in controls or hard in controls)))
        onsets = [int(n_onset(a, "5M")) for a in present]
        new_only = [a for a in present if a not in controls]
        new_contrasts = [c for c in adjacent
                         if c["softer"] not in controls and c["harder"] not in controls]
        new_onsets = [int(n_onset(a, "5M")) for a in new_only]
        label, hits = _v2_label(G, adjacent, onsets, new_contrasts, new_onsets)
        v2[family] = dict(status="OK", label=label, co_satisfied=hits, ladder=ladder,
                          used=present, dropped=dropped,
                          dropped_reason=dropped_reason, adjacent=adjacent,
                          onsets=onsets, new_arms=new_only, new_onsets=new_onsets,
                          k_star_median={a: (float(np.median(k_star_by_arm[a]))
                                             if a in k_star_by_arm else float("nan"))
                                         for a in present},
                          relu_end_is_exact_limit=bool(
                              dict(G["ladder_relu_end_is_exact_limit"])[family]))

    # --- V3 硬さ表（§5.4） ---
    dial_rows, depth_rows, s_rows = [], [], []
    unit_by_arm: dict[str, list[dict]] = {}
    for arm in complete + list(CONTROL_ORDER):
        geo = _geometry(cfg, arm)
        source = outdir / "logs" if arm in complete else ref_logs
        per_seed = []
        for i, seed in enumerate(seeds):
            path = source / f"{arm}_seed{seed}.npz"
            if not path.exists():
                continue
            summary = _unit_summary(cfg, path, geo)
            if arm in CONTROL_ORDER and cfg["controls"]["m_minus_from_checkpoint"]:
                ckpt = _m_minus_from_checkpoint(cfg, arm, i, geo)
                summary["m_minus"] = (ckpt.get("m_minus", float("nan"))
                                      if ckpt["status"] == "OK" else float("nan"))
                summary["m_minus_window"] = ckpt.get(
                    "window", "INSUFFICIENT_DATA")
                summary["m_minus_source"] = ckpt.get("source", "")
            else:
                summary["m_minus_window"] = "late_tasks_5m"
                summary["m_minus_source"] = str(path)
            summary.update(arm=arm, seed=seed,
                           is_control=int(arm in CONTROL_ORDER))
            per_seed.append(summary)
            depth_rows.append(dict(
                arm=arm, seed=seed, is_control=int(arm in CONTROL_ORDER),
                depth_median=summary["depth_median"],
                **{f"decile_{i + 1}": v
                   for i, v in enumerate(summary["depth_deciles"])}))
            s_rows.append(dict(arm=arm, seed=seed,
                               is_control=int(arm in CONTROL_ORDER),
                               source=str(path), **_s_series(cfg, path)))
        if not per_seed:
            continue
        unit_by_arm[arm] = per_seed

        def med(key: str) -> float:
            values = np.asarray([row[key] for row in per_seed], dtype=np.float64)
            values = values[np.isfinite(values)]
            return float(np.median(values)) if values.size else float("nan")

        level = log_u(arm, "5M")
        dial_rows.append(dict(
            arm=arm, family=_family(cfg, arm), dial=_dial(cfg, arm),
            is_control=int(arm in CONTROL_ORDER),
            u_star_registered=geo["u_star_registered"],
            u_star_numeric=geo["u_star_numeric"],
            u_fr_registered=geo["u_fr_registered"],
            u_fr_numeric=geo["u_fr_numeric"],
            m_minus=med("m_minus"),
            m_minus_window=per_seed[0]["m_minus_window"],
            m_minus_is_independent=int(_family(cfg, arm) in
                                       list(G["dial_table"]["m_minus"]
                                            ["independent_only_for"])),
            submerged_frac=med("submerged_frac"),
            beyond_valley_frac=med("beyond_valley_frac"),
            frozen_frac=med("frozen_frac"),
            frozen_plus_valley_frac=med("frozen_or_beyond_frac"),
            strict_dead_frac=(med("strict_dead_frac")
                              if _family(cfg, arm) == "relu" else ""),
            submerged_abs_v_median=med("submerged_abs_v_median"),
            abs_v_median=med("abs_v_median"),
            n_onset_1m=n_onset(arm, "1M"), n_onset_5m=n_onset(arm, "5M"),
            k_star_median=(float(np.median(k_star_by_arm[arm]))
                           if arm in k_star_by_arm else float("nan")),
            median_log10_U_5m=(float(np.median(level)) if level is not None
                               else float("nan")),
            residual_level=(float(np.median(windows[arm]["5M"]["metrics"]
                                            ["eval_loss_exact"]))
                            if arm in complete else float("nan")),
            capacity_status=cap.get(arm, {}).get("status", "")))

    # Spearman（プール・族内）。**ラベルは置かない**（spec §5.4）
    def _predictor(row: dict, name: str) -> float:
        if name == "u_fr":
            value = row["u_fr_numeric"]
            return value if np.isfinite(value) else np.inf
        return row[{"m_minus": "m_minus",
                    "frozen_plus_valley_frac": "frozen_plus_valley_frac"}[name]]

    spearman_rows = []
    for scope in list(G["dial_table"]["spearman_scopes"]):
        groups = ({"pool": dial_rows} if scope == "pool" else
                  {f: [r for r in dial_rows if r["family"] == f]
                   for f in {r["family"] for r in dial_rows}})
        for name, rows in groups.items():
            outcome = np.asarray([r["median_log10_U_5m"] for r in rows],
                                 dtype=np.float64)
            for predictor in list(G["dial_table"]["predictors"]):
                x = np.asarray([_predictor(r, predictor) for r in rows],
                               dtype=np.float64)
                good = np.isfinite(outcome) & ~np.isnan(x)
                if predictor == "u_fr":
                    ranked = np.where(np.isinf(x), np.nanmax(x[np.isfinite(x)],
                                                             initial=0.0) + 1.0, x)
                    x = ranked
                    good = np.isfinite(outcome) & np.isfinite(x)
                spearman_rows.append(dict(
                    scope=scope, group=name, predictor=predictor,
                    n=int(good.sum()),
                    rho=(spearman(x[good], outcome[good]) if good.sum() >= 2
                         else float("nan"))))

    # --- §5.5 REPORT_ONLY ---
    revival_rows = []
    for arm in complete + list(CONTROL_ORDER):
        source = outdir / "logs" if arm in complete else ref_logs
        for seed in seeds:
            path = source / f"{arm}_seed{seed}.npz"
            if not path.exists():
                continue
            counts = _revival_counts(path)
            within = counts["opportunities_within_task"]
            across = counts["opportunities_across_boundary"]
            revival_rows.append(dict(
                arm=arm, seed=seed, is_control=int(arm in CONTROL_ORDER),
                **counts,
                rate_within_task=(counts["events_within_task"] / within
                                  if within else float("nan")),
                rate_across_boundary=(counts["events_across_boundary"] / across
                                      if across else float("nan"))))

    increment_rows = []
    for arm in complete:
        for seed in seeds:
            got_rows = _interval_rows(_q2_cfg(cfg), outdir, arm, seed)
            increment_rows.append(dict(row_type="seed_summary", arm=arm, seed=seed,
                                       status=got_rows["status"],
                                       beta_seed=got_rows["beta"],
                                       rho_seed=got_rows["rho"],
                                       n_intervals=got_rows["n_intervals"],
                                       n_submerged_unit_intervals=got_rows["n_unit_intervals"],
                                       median_submerged_units=got_rows["median_submerged_units"],
                                       bin="", n="", zbar_bin_median="",
                                       zbar_bin_lo="", zbar_bin_hi="",
                                       dzbar_median="", dzbar_sd="", rho="",
                                       eligible=""))
            for row in got_rows["bins"]:
                increment_rows.append(dict(row_type="bin", arm=arm, seed=seed,
                                           status=got_rows["status"],
                                           beta_seed=got_rows["beta"],
                                           rho_seed=got_rows["rho"],
                                           n_intervals=got_rows["n_intervals"],
                                           n_submerged_unit_intervals=got_rows["n_unit_intervals"],
                                           median_submerged_units=got_rows["median_submerged_units"],
                                           **row))

    result = dict(
        stage=stage, arms_run=list(arms), complete=complete,
        divergences=sorted(divergences), V1=v1, V1_states=v1_states,
        V1_developed=v1_developed, V1_capacity_blocked=v1_capacity_blocked,
        V2={f: {k: v for k, v in value.items() if k != "adjacent"}
            for f, value in v2.items()},
        V2_adjacent={f: value.get("adjacent", []) for f, value in v2.items()},
        onset=onset, capacity=cap,
        capacity_undefined=sorted(capacity_undefined),
        controls={a: dict(n_onset_5m=controls[a]["n_onset_5m"],
                          n_onset_1m=controls[a]["n_onset_1m"]) for a in CONTROL_ORDER},
        contrasts=contrasts, rho=rho, spearman=spearman_rows,
        s_mask=s_mask, elapsed_sec=elapsed)

    sanity = dict(sanity, S_cap=dict(pass_=not capacity_undefined, rows=cap),
                  S_mask=s_mask)
    _write_outputs(cfg, outdir, arms, complete, divergences, windows, controls,
                   onset, result, dial_rows, spearman_rows, onset_rows, km_rows,
                   depth_rows, s_rows, revival_rows, increment_rows, v2, sanity)
    result["sanity_added"] = ["S_cap", "S_mask"]
    return result


def _q2_cfg(cfg: dict) -> dict:
    """``gate_dose._interval_rows`` が読む ``gate_dose`` セクションを合成する。

    Q2 は REPORT_ONLY で、機構は宿主のものをそのまま回す（[[現在地]] 穴 8 に
    ``SCALING_MISMATCH`` の履歴があるので**主張には使わない**）。
    """
    out = copy.deepcopy(cfg)
    out["gate_dose"] = dict(
        q2_increment_interval_steps=int(cfg["common"]["lop_every"]),
        q2_window_tasks=list(cfg["phase1"]["late_tasks_5m"]),
        q2_bins=12, q2_bin_method="equal_count_quantile", q2_bin_min_count=20,
        q2_min_submerged_units_per_seed=3)
    return out


def _write_outputs(cfg, outdir, arms, complete, divergences, windows, controls,
                   onset, result, dial_rows, spearman_rows, onset_rows, km_rows,
                   depth_rows, s_rows, revival_rows, increment_rows, v2,
                   sanity) -> None:
    seeds = [int(v) for v in cfg["common"]["seeds"]]
    G = _P(cfg)
    verdict_rows = []
    for arm in arms:
        arm_cfg = _arm(cfg, arm)
        family = str(arm_cfg["family"])
        v2_entry = v2.get(family, {})
        base = dict(arm=arm, stage=int(arm_cfg["stage"]), family=family,
                    activation=str(arm_cfg["activation"]),
                    dial=float(arm_cfg["dial"]),
                    target_dose=float(arm_cfg["target_dose"]), is_control=0,
                    V1=result["V1"],
                    V1_states=json.dumps(result["V1_states"]),
                    V1_developed="|".join(result["V1_developed"]),
                    V2_family=v2_entry.get("label", ""),
                    V2_co_satisfied="|".join(v2_entry.get("co_satisfied", [])),
                    capacity_status=result["capacity"].get(arm, {}).get("status", ""))
        if arm in complete:
            w = windows[arm]
            cp1 = clopper_pearson(onset["1M"][arm], len(seeds))
            cp5 = clopper_pearson(onset["5M"][arm], len(seeds))
            base.update(
                status="COMPLETE", NUMERIC_DIVERGENCE=0,
                n_onset_1m=onset["1M"][arm], cp95_1m_lo=cp1[0], cp95_1m_hi=cp1[1],
                U_1m_seed_values=json.dumps(w["1M"]["u"].tolist()),
                median_log10_U_1m=float(np.median(w["1M"]["log_u"])),
                n_onset_5m=onset["5M"][arm], cp95_5m_lo=cp5[0], cp95_5m_hi=cp5[1],
                U_5m_seed_values=json.dumps(w["5M"]["u"].tolist()),
                median_log10_U_5m=float(np.median(w["5M"]["log_u"])),
                median_submerged_frac_5m=float(
                    np.median(w["5M"]["metrics"]["layer1_submerged"] / 100.0)),
                median_w_norm_5m=float(
                    np.median(w["5M"]["metrics"]["layer1_w_norm_median"])),
                median_eval_loss_exact_5m=float(
                    np.median(w["5M"]["metrics"]["eval_loss_exact"])))
        else:
            base.update(status=NUMERIC_DIVERGENCE, NUMERIC_DIVERGENCE=1,
                        n_onset_1m="", cp95_1m_lo="", cp95_1m_hi="",
                        U_1m_seed_values="", median_log10_U_1m="",
                        n_onset_5m="", cp95_5m_lo="", cp95_5m_hi="",
                        U_5m_seed_values="", median_log10_U_5m="",
                        median_submerged_frac_5m="", median_w_norm_5m="",
                        median_eval_loss_exact_5m="")
        for kind in ("P3prime", "P5prime"):
            hits = [v for v in result["contrasts"].values()
                    if v["kind"] == kind and v["high"] == arm]
            base[f"{kind}_contrasts"] = "|".join(
                f"{v['high']}_minus_{v['low']}" for v in hits)
            base[f"{kind}_labels"] = "|".join(str(v.get("label", "")) for v in hits)
            base[f"{kind}_points"] = "|".join(
                (f"{v['ci']['point']:.6g}" if v["status"] == "OK" else v["status"])
                for v in hits)
            base[f"{kind}_ci"] = "|".join(
                (f"[{v['ci']['percentile_ci_lo']:.6g},{v['ci']['percentile_ci_hi']:.6g}]"
                 if v["status"] == "OK" else "") for v in hits)
        verdict_rows.append(base)
    for arm in CONTROL_ORDER:
        c = controls[arm]
        family = CONTROL_FAMILY[arm]
        v2_entry = v2.get(family, {})
        verdict_rows.append(dict(
            arm=arm, stage=0, family=family,
            activation=str(cfg["controls"]["arms"][arm]["activation"]),
            dial=CONTROL_DIAL[arm], target_dose=12.16, is_control=1,
            V1=result["V1"], V1_states=json.dumps(result["V1_states"]),
            V1_developed="|".join(result["V1_developed"]),
            V2_family=v2_entry.get("label", ""),
            V2_co_satisfied="|".join(v2_entry.get("co_satisfied", [])),
            capacity_status="",
            status="COMMITTED_OTHER_RUN", NUMERIC_DIVERGENCE=0,
            n_onset_1m=c["n_onset_1m"], cp95_1m_lo="", cp95_1m_hi="",
            U_1m_seed_values=json.dumps(c["u_1m"].tolist()),
            median_log10_U_1m=float(np.median(c["log_u_1m"])),
            n_onset_5m=c["n_onset_5m"], cp95_5m_lo="", cp95_5m_hi="",
            U_5m_seed_values=json.dumps(c["u_5m"].tolist()),
            median_log10_U_5m=float(np.median(c["log_u_5m"])),
            median_submerged_frac_5m="", median_w_norm_5m="",
            median_eval_loss_exact_5m="",
            P3prime_contrasts="", P3prime_labels="", P3prime_points="",
            P3prime_ci="", P5prime_contrasts="", P5prime_labels="",
            P5prime_points="", P5prime_ci=""))
    write_csv(outdir / "verdict.csv", verdict_rows)

    contrast_rows = []
    for label, value in result["contrasts"].items():
        row = dict(endpoint=value["kind"], contrast=label, high=value["high"],
                   low=value["low"], status=value["status"],
                   cross_run=int(value.get("cross_run", 0)),
                   n_paired=value.get("n_paired", ""),
                   label=value.get("label", ""),
                   soft_end=value.get("soft_end", ""),
                   ci_below_zero=(int(value["ci_below_zero"])
                                  if "ci_below_zero" in value else ""),
                   equivalence_margin=value.get("equivalence_margin", ""))
        ci = value.get("ci")
        for key in ("point", "percentile_ci_lo", "percentile_ci_hi",
                    "studentized_ci_lo", "studentized_ci_hi", "ci_degenerate"):
            row[key] = "" if ci is None else ci[key]
        st = value.get("sign_test") or {}
        row.update(sign_n_positive=st.get("n_positive", ""),
                   sign_n_negative=st.get("n_negative", ""),
                   sign_p_two_sided=st.get("p_two_sided", ""),
                   seed_values=json.dumps(value.get("seed_values", [])))
        contrast_rows.append(row)
    for family, value in v2.items():
        for entry in value.get("adjacent", []):
            ci = entry["ci"]
            contrast_rows.append(dict(
                endpoint="V2_adjacent", contrast=f"{family}:{entry['harder']}_minus_{entry['softer']}",
                high=entry["harder"], low=entry["softer"], status="OK",
                cross_run=int(entry["cross_run"]), n_paired=len(entry["seed_values"]),
                label=value.get("label", ""), soft_end="", ci_below_zero="",
                equivalence_margin=float(_P(cfg)["v2_margin"]),
                point=ci["point"], percentile_ci_lo=ci["percentile_ci_lo"],
                percentile_ci_hi=ci["percentile_ci_hi"],
                studentized_ci_lo=ci["studentized_ci_lo"],
                studentized_ci_hi=ci["studentized_ci_hi"],
                ci_degenerate=ci["ci_degenerate"],
                sign_n_positive=entry["sign_test"]["n_positive"],
                sign_n_negative=entry["sign_test"]["n_negative"],
                sign_p_two_sided=entry["sign_test"]["p_two_sided"],
                seed_values=json.dumps(entry["seed_values"])))
    for key, value in result["rho"].items():
        ci = value.get("ci")
        contrast_rows.append(dict(
            endpoint="rho", contrast=key, high=value["arm"], low=value["soft_end"],
            status=value["status"], cross_run=1, n_paired="", label="",
            soft_end=value["soft_end"], ci_below_zero="", equivalence_margin="",
            point="" if ci is None else ci["point"],
            percentile_ci_lo="" if ci is None else ci["percentile_ci_lo"],
            percentile_ci_hi="" if ci is None else ci["percentile_ci_hi"],
            studentized_ci_lo="" if ci is None else ci["studentized_ci_lo"],
            studentized_ci_hi="" if ci is None else ci["studentized_ci_hi"],
            ci_degenerate="" if ci is None else ci["ci_degenerate"],
            sign_n_positive="", sign_n_negative="", sign_p_two_sided="",
            seed_values=json.dumps(value.get("seed_values", []))))
    write_csv(outdir / "layer_stats.csv", contrast_rows)

    fields = list(dict.fromkeys(k for row in dial_rows for k in row))
    write_csv(outdir / "dial_table.csv",
              [{k: row.get(k, "") for k in fields} for row in dial_rows])
    write_csv(outdir / "dial_spearman.csv", spearman_rows)
    if onset_rows:
        write_csv(outdir / "onset_times.csv", onset_rows)
    if km_rows:
        write_csv(outdir / "onset_km.csv", km_rows)
    if depth_rows:
        write_csv(outdir / "depth_hist.csv", depth_rows)
    if s_rows:
        write_csv(outdir / "s_distribution.csv", s_rows)
    if revival_rows:
        write_csv(outdir / "revival.csv", revival_rows)
    if increment_rows:
        keys = list(dict.fromkeys(k for row in increment_rows for k in row))
        write_csv(outdir / "increments.csv",
                  [{k: row.get(k, "") for k in keys} for row in increment_rows])
    _write_summary(cfg, outdir, result, verdict_rows, dial_rows, spearman_rows,
                   v2, sanity)


def _write_summary(cfg: dict, outdir: Path, result: dict, verdict_rows: list[dict],
                   dial_rows: list[dict], spearman_rows: list[dict], v2: dict,
                   sanity: dict) -> None:
    G = _P(cfg)
    lines = [f"# {EXPERIMENT} summary (stage {result['stage']})", "",
             "## Verdict", "",
             f"- **V1（標準点の位置）: {result['V1']}** — "
             + ", ".join(f"{a}={s}" for a, s in result["V1_states"].items()),
             f"- V1 で発症した腕: {', '.join(result['V1_developed']) or '—'}",
             (f"- **V1 は S-cap 除外の帰結として `{result['V1']}`**: "
              f"{', '.join(result['V1_capacity_blocked'])} は early 窓でフィットして"
              f"おらず、絶対閾値 0.05 に対して発症が**定義されない**（spec §6 の "
              f"S-cap。`width5_gate_0901` と同型）。登録された 4 ラベルのどれでもない。"
              if result["V1_capacity_blocked"] else
              "- V1 は登録どおりの 4 ラベルから出ている（S-cap 落ちの腕は無い）"),
             ""]
    lines += ["| family | V2 | 当たっていた行 | 梯子（軟→硬） | n_onset(5M) | 落とした腕 |",
              "|---|---|---|---|---|---|"]
    for family, value in v2.items():
        if value["status"] != "OK":
            lines.append(f"| {family} | {value['status']} | — | "
                         f"{' → '.join(value['ladder'])} | — | "
                         f"{', '.join(value['dropped']) or '—'} |")
            continue
        lines.append(
            f"| {family} | **{value['label']}** | "
            f"{', '.join(value['co_satisfied'])} | "
            f"{' → '.join(value['used'])} | "
            f"{', '.join(str(v) for v in value['onsets'])} | "
            f"{', '.join(f'{a}({value['dropped_reason'][a]})' for a in value['dropped']) or '—'} |")
    lines += ["", f"- Numeric divergence: {', '.join(result['divergences']) or 'none'}",
              f"- CAPACITY_UNDEFINED: {', '.join(result['capacity_undefined']) or 'none'}",
              "",
              "### 引用上の注意（spec §8）", "",
              "- 0/10 は「5M までに観測しなかった」（片側 95% 上限 p<=0.2589）。「起きない」と書かない。",
              "- **対照 `R_1216` / `LR_1216` / `E_1216` は別走 `gate_dose_0830` の committed 値であり、",
              "  同一走の腕ではない。** ペアリングは init・教師・入力実現までで、軌道は step 1 以降で分岐する。",
              "- **用量 1 点（12.16）の主張である。** 引くときは必ず用量を添える。",
              "- **u_fr・谷底は閉形式 + K=1 の代入で、実験出力ではない。** 「凍結深さは 13.8」と",
              "  測定値のように書かない（dial_table.csv の *_registered / *_numeric とも同じ格）。",
              "- `layer1_mob` は ReLU・leaky では p_hat の一次関数。「可動度を測った」と書けるのは",
              "  ELU・SiLU・GELU 腕だけ。",
              "- 対照の m⁻ は **5M チェックポイントの 1 点**（窓 `final_step5000000`）であって",
              "  末尾窓 491-500 の量ではない。窓を落として引かない。",
              "- `beyond_valley` / `frozen` は**位置**であって病理ではない。",
              "- V3 からラベルを作らない。「硬さはスカラーか」は裁定であって判定ではない。",
              "- SiLU/GELU の beta -> inf は数学的極限であって、`R_1216` は SiLU/GELU 族の腕ではない。",
              "- Q2（increments.csv）は `SCALING_MISMATCH` の履歴があるので主張に使わない。",
              "", "## Endpoints (5M)", "",
              "| arm | family | dial | onset 1M | onset 5M | median log10 U 1M | median log10 U 5M | source |",
              "|---|---|---:|---:|---:|---:|---:|---|"]
    for row in verdict_rows:
        if row["status"] == NUMERIC_DIVERGENCE:
            lines.append(f"| {row['arm']} | {row['family']} | {row['dial']:g} | "
                         f"— | — | — | — | {row['status']} |")
            continue
        source = "gate_dose_0830 (別走)" if row["is_control"] else "this run"
        lines.append(
            f"| {row['arm']} | {row['family']} | {row['dial']:g} | "
            f"{row['n_onset_1m']}/10 | {row['n_onset_5m']}/10 | "
            f"{row['median_log10_U_1m']:.6g} | {row['median_log10_U_5m']:.6g} | {source} |")
    lines += ["", "## §5.4 V3 硬さ表（ラベルを置かない）", "",
              "| arm | family | dial | u* | u_fr | m⁻ | m⁻ 窓 | 沈下率 | 谷率 | 凍結率 | n_onset 5M | k* 中央値 | median log10 U 5M |",
              "|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|"]
    for row in dial_rows:
        def fmt(value):
            if value in ("", None):
                return "—"
            if isinstance(value, float):
                return "—" if not np.isfinite(value) else (
                    "inf" if np.isinf(value) else f"{value:.4g}")
            return str(value)
        lines.append(
            f"| {row['arm']} | {row['family']} | {fmt(row['dial'])} | "
            f"{fmt(row['u_star_numeric'])} | {fmt(row['u_fr_numeric'])} | "
            f"{fmt(row['m_minus'])} | {row['m_minus_window']} | "
            f"{fmt(row['submerged_frac'])} | {fmt(row['beyond_valley_frac'])} | "
            f"{fmt(row['frozen_frac'])} | {row['n_onset_5m']} | "
            f"{fmt(row['k_star_median'])} | {fmt(row['median_log10_U_5m'])} |")
    lines += ["", "Spearman（予測子 対 median log10 U 5M・**ラベル無し**）", "",
              "| scope | group | predictor | n | rho |", "|---|---|---|---:|---:|"]
    for row in spearman_rows:
        lines.append(f"| {row['scope']} | {row['group']} | {row['predictor']} | "
                     f"{row['n']} | {row['rho']:.4g} |")
    lines += ["", "## S-mask（spec 字義との差）", "",
              f"- 実際の窓の記録点数: **{sanity['S_mask']['actual_records_per_window']}**"
              f"（spec §5.3 の字義は {sanity['S_mask']['spec_literal_records_per_window']}）",
              f"- {sanity['S_mask']['resolution']}",
              "", "## Sanity", ""]
    for key in ("S1_omp", "S_dial", "S_fd", "S_num", "S_limit_smooth",
                "S_elu_limit", "S_ref", "S_mob", "S_log_b", "S_pair", "S_dose",
                "S_taut", "S_cap", "S_mask", "S6_floor_inherited",
                "S_CI_degeneracy"):
        value = sanity.get(key, {})
        lines.append(f"- {key}: **{'PASS' if value.get('pass_') else 'FAIL'}**")
    (outdir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Run driver
# ---------------------------------------------------------------------------
def _provenance(cfg_path: Path, cfg: dict, outdir: Path, stage: str,
                arms: list[str], sanity: dict, analysis: dict, elapsed: dict,
                started: float) -> dict:
    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                           text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--short"], cwd=ROOT,
                                        text=True).splitlines()
    except (OSError, subprocess.CalledProcessError):
        git_hash, dirty = None, []
    names = ("verdict.csv", "summary.md", "layer_stats.csv", "dial_table.csv",
             "dial_spearman.csv", "onset_times.csv", "onset_km.csv",
             "depth_hist.csv", "s_distribution.csv", "revival.csv",
             "increments.csv", "config_used.yaml")
    hashes = {n: _sha_file(outdir / n) for n in names if (outdir / n).exists()}
    hashes.update({f"logs/{p.name}": _sha_file(p)
                   for p in sorted((outdir / "logs").glob("*.npz"))})
    hashes.update({f"arm_status/{p.name}": _sha_file(p)
                   for p in sorted((outdir / "arm_status").glob("*.json"))})
    ref_dir = (Path(ROOT) / cfg["controls"]["reference_run"]).resolve()
    parent_prov = ref_dir / "provenance.json"
    parent = (json.loads(parent_prov.read_text(encoding="utf-8"))
              if parent_prov.exists() else {})
    saturation = {row["arm"]: dict(
        forward=row["float32_forward_exact_zero_depth"],
        backward=row["float32_backward_exact_zero_depth"])
        for row in sanity.get("S_num", {}).get("rows", [])}
    return dict(
        experiment=EXPERIMENT, created=time.strftime("%Y-%m-%d %H:%M:%S %z"),
        command=sys.argv, elapsed_sec=round(time.time() - started, 3),
        arm_elapsed_sec=elapsed, cwd=os.getcwd(), python=sys.version,
        platform=platform.platform(), torch=torch.__version__, numpy=np.__version__,
        device=cfg["common"]["device"], git_hash=git_hash, git_dirty=dirty,
        config=str(cfg_path), config_sha256=_sha_file(cfg_path),
        spec=str(Path(ROOT) / cfg["spec"]),
        spec_sha256=_sha_file(Path(ROOT) / cfg["spec"]),
        stage_run=stage, arms_run=list(arms), dose="12.16",
        stages_registered=dict(cfg["staging"]),
        float32_saturation_depth=saturation,
        displacement_bound_constant_K=float(
            _P(cfg)["design"]["displacement_bound_constant_K"]),
        K_posthoc_check=dict(
            note="the K=1 promise is checked after the fact in dial_table.csv "
                 "(abs_v_median and residual_level columns); it is a convention, "
                 "not a guarantee (spec §2)"),
        generator_offset=int(cfg["common"]["generator_offset"]),
        generator_offset_note=(
            "explicit 0: this run deliberately shares the parent run's seed set and "
            "random stream (S-pair)."),
        window_definition=dict(
            task_ends_only=True, records_per_10task_window=10,
            spec_literal=int(cfg["phase1"]["spec_literal_records_per_window"]),
            note=analysis.get("s_mask", {}).get("resolution", "")),
        baseline_reference=str(ref_dir), baseline_git_hash=parent.get("git_hash"),
        baseline_endpoint_source=str(ref_dir / "verdict.csv"),
        baseline_unit_source=str(ref_dir / "logs"),
        baseline_unit_source_is_gitignored=True,
        sanity=sanity, analysis=analysis, output_sha256=hashes)


def run(cfg_path: Path, cfg: dict, device: str, outdir: Path, stage: str, *,
        smoke: bool, analyze_only: bool) -> dict:
    started = time.time()
    outdir.mkdir(parents=True, exist_ok=True)
    require_omp(cfg)
    arms = _selected_arms(cfg, stage)
    total = SMOKE_STEPS if smoke else int(cfg["common"]["total_steps"])
    seeds = [0] if smoke else [int(v) for v in cfg["common"]["seeds"]]
    every = int(cfg["common"]["lop_every"])
    with (outdir / "config_used.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, allow_unicode=True, sort_keys=False)

    preflight_path = Path(ROOT) / f"results/_preflight_{EXPERIMENT}/preflight.json"
    if smoke:
        preflight_result = {"pass_": True, "smoke": True}
    else:
        if not preflight_path.exists():
            raise FileNotFoundError("run --preflight before the full run")
        preflight_result = json.loads(preflight_path.read_text(encoding="utf-8"))
        if not preflight_result.get("pass_"):
            raise SanityError("the saved preflight did not pass")

    elapsed, divergences, identities = {}, {}, {}
    for arm in arms:
        existing = _load_divergence_status(outdir, arm, seeds, total, every)
        if existing is not None and not smoke:
            divergences[arm] = existing
            elapsed[arm] = 0.0
            print(f"[{arm}] saved {NUMERIC_DIVERGENCE}; resume", flush=True)
            continue
        if analyze_only:
            continue
        if _complete_arm_logs(outdir, arm, seeds, total, every):
            elapsed[arm] = 0.0
            identities[arm] = dict(pass_=True, resumed_from_logs=True)
            print(f"[{arm}] complete logs found; resume", flush=True)
            continue
        got = _run_arm(cfg, arm, device, outdir, seeds, total)
        elapsed[arm] = got["elapsed_sec"]
        identities[arm] = got["sanity"]
        if got["status"] == NUMERIC_DIVERGENCE:
            divergences[arm] = got["divergence"]

    if smoke:
        payload = dict(pass_=bool(all(v.get("pass_") for v in identities.values())),
                       identities=identities, divergences=divergences,
                       elapsed_sec=elapsed, arms=arms)
        (outdir / "smoke_sanity.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8")
        print(f"SMOKE DONE -> {outdir}", flush=True)
        return payload

    missing = [a for a in arms if a not in divergences
               and not _complete_arm_logs(outdir, a, seeds, total, every)]
    if missing:
        raise SanityError(f"arms without complete logs: {missing}")

    sanity = dict(preflight_result)
    sanity.pop("pass_", None)
    result = analyze(cfg, outdir, arms, stage, sanity, elapsed, divergences)
    sanity = dict(sanity, S_cap=dict(pass_=not result["capacity_undefined"],
                                     rows=result["capacity"]),
                  S_mask=result["s_mask"])
    provenance = _provenance(cfg_path, cfg, outdir, stage, arms, sanity, result,
                             elapsed, started)
    (outdir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")
    print(f"V1={result['V1']}", flush=True)
    for family, value in result["V2"].items():
        print(f"V2[{family}]={value.get('label', value.get('status'))}", flush=True)
    print(f"ALL DONE -> {outdir}", flush=True)
    return dict(sanity=sanity, analysis=result)


def run_single_arm(cfg: dict, arm: str, device: str, outdir: Path,
                   total: int) -> dict:
    """腕プロセス並列の投入単位。1 腕だけ走らせて logs を置いて終わる。"""
    require_omp(cfg)
    outdir.mkdir(parents=True, exist_ok=True)
    seeds = [int(v) for v in cfg["common"]["seeds"]]
    every = int(cfg["common"]["lop_every"])
    if _load_divergence_status(outdir, arm, seeds, total, every) is not None:
        print(f"[{arm}] saved {NUMERIC_DIVERGENCE}; nothing to do", flush=True)
        return dict(status=NUMERIC_DIVERGENCE)
    if _complete_arm_logs(outdir, arm, seeds, total, every):
        print(f"[{arm}] complete logs found; nothing to do", flush=True)
        return dict(status="COMPLETE", resumed=True)
    return _run_arm(cfg, arm, device, outdir, seeds, total)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--s-par", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--stage", default="all", choices=["1", "2", "3", "all"])
    parser.add_argument("--arm", default=None, choices=list(ARM_ORDER))
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--outdir")
    args = parser.parse_args()
    exclusive = sum((args.preflight, args.smoke, args.s_par,
                     args.arm is not None))
    if exclusive > 1:
        parser.error("--preflight / --smoke / --s-par / --arm are exclusive")
    cfg_path = Path(args.config).resolve()
    cfg = load_config(str(cfg_path))
    device = pick_device(cfg)
    if device != "cpu":
        raise ValueError("gate_dial is CPU-only")
    stage = ("preflight" if args.preflight else "smoke" if args.smoke
             else "spar" if args.s_par
             else "analyze" if args.analyze_only else "run")
    validate_config(cfg, stage=stage)
    main_dir = Path(ROOT) / cfg["output"]["dir"]
    if args.preflight:
        preflight(cfg, Path(ROOT) / f"results/_preflight_{EXPERIMENT}")
        return
    if args.s_par:
        result = s_par(cfg, cfg_path, Path(ROOT) / f"results/_spar_{EXPERIMENT}")
        if not result["pass_"]:
            raise SanityError(f"S-par failed: {result}")
        return
    if args.arm is not None:
        outdir = Path(args.outdir).resolve() if args.outdir else main_dir
        total = int(args.steps) if args.steps else int(cfg["common"]["total_steps"])
        run_single_arm(cfg, args.arm, device, outdir, total)
        return
    outdir = (Path(args.outdir).resolve() if args.outdir
              else Path(ROOT) / f"results/_smoke_{EXPERIMENT}" if args.smoke
              else main_dir)
    run(cfg_path, cfg, device, outdir, args.stage, smoke=args.smoke,
        analyze_only=args.analyze_only)


if __name__ == "__main__":
    main()
