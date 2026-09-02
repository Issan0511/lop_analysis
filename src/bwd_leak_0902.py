"""bwd_leak_0902: forward と backward の片道性を別々に外す（順逆分離 × b-WD）。

事前登録: ``specs/spec_bwd_leak_0902.md``（この実装より**先に** ``9e9caa0`` で
config と一緒に単独 commit されている）。Obsidian 側の正本は
``可塑性喪失/spec/逆伝播漏れ2x2_spec_0902.md`` v2。

宿主は ``gate_dose_0830``（1 層・オラクル用量固定・5M）で、学習経路・記録経路・
用量固定はそのまま ``src.gate_dose`` から import する。新規に足すのは

* ``VecMLPL`` の 2 活性化 ``bwd_leaky`` / ``fwd_leaky``（``src/nets.py``。既存の
  ``relu`` / ``leaky_relu`` の式を組み合わせるだけで新しい算術を書かない）
* ユニット別 ``v`` / ``b`` の記録（§5.3 の凍結率のためだけ。S-log-b が軌道中立性を検査）
* 本モジュールの sanity・集計

対照 ``R_*`` / ``LR_*`` は再走しない。主 endpoint は
``results/gate_dose_0830/verdict.csv`` から転記し、P7b / P7c と §5.3 の対照だけは
同走の ``logs/*.npz`` を読む（spec §6.2 追補 4 の carve-out）。

Stages::

    OMP_NUM_THREADS=1 .venv/bin/python -m src.bwd_leak_0902 --preflight
    OMP_NUM_THREADS=1 .venv/bin/python -m src.bwd_leak_0902 --smoke
    OMP_NUM_THREADS=1 .venv/bin/python -m src.bwd_leak_0902 --stage 1
    OMP_NUM_THREADS=1 .venv/bin/python -m src.bwd_leak_0902 --stage 2
    OMP_NUM_THREADS=1 .venv/bin/python -m src.bwd_leak_0902 --stage 1 --doses 1216
    OMP_NUM_THREADS=1 .venv/bin/python -m src.bwd_leak_0902 --stage 1 --analyze-only
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
from .dose_const_5m import (EXTRA_LOG_KEYS, _input_stats, _refresh_fixed_offset,
                            _target, clopper_pearson, setup_arm_const)
from .elu_swamp import exact_layer_record_elu, grads_centered_elu
from .gate_dose import (GateRecorder, SIGMA_TOL, IDENTITY_TOL, forward_gate,
                        train_arm_gate, _window)
from .mlp2_phase0 import (_sha_array, _sha_file, identity_sanity_pass,
                          require_omp, write_csv)
from .mlp2_phase0b import _ci_components, _complete_arm_logs, _window_indices
from .mlp2_phase1 import (NUMERIC_DIVERGENCE, NumericDivergenceError,
                          StreamDigest, _env_hashes, _init_hashes,
                          _seed_state_hashes_p1)
from .nets import VecMLPL


EXPERIMENT = "bwd_leak_0902"
CONFIG = Path(ROOT) / "configs" / "bwd_leak_0902.yaml"

ARM_ORDER = ("BL_933", "BL_1216", "FL_933", "FL_1216",
             "RW_933", "RW_1216", "BLW_933", "BLW_1216")
CONTROL_ORDER = ("R_933", "R_1216", "LR_933", "LR_1216")
STAGE_ARMS = {1: ("BL_933", "BL_1216", "FL_933", "FL_1216"),
              2: ("RW_933", "RW_1216", "BLW_933", "BLW_1216")}
# 判定に使う腕族（用量をまとめた単位）と、その族に属する腕。
FAMILY_ARMS = {"BL": ("BL_933", "BL_1216"), "FL": ("FL_933", "FL_1216"),
               "RW": ("RW_933", "RW_1216"), "BLW": ("BLW_933", "BLW_1216"),
               "R": ("R_933", "R_1216"), "LR": ("LR_933", "LR_1216")}

# 事前登録の腕定義（stage, activation label, wd_b, target_mu_norm, target_dose）。
REGISTERED_ARMS = {
    "BL_933":   (1, "bwd_leaky", 0.0, 2.333, 9.33),
    "BL_1216":  (1, "bwd_leaky", 0.0, 3.041, 12.16),
    "FL_933":   (1, "fwd_leaky", 0.0, 2.333, 9.33),
    "FL_1216":  (1, "fwd_leaky", 0.0, 3.041, 12.16),
    "RW_933":   (2, "relu", 1e-3, 2.333, 9.33),
    "RW_1216":  (2, "relu", 1e-3, 3.041, 12.16),
    "BLW_933":  (2, "bwd_leaky", 1e-3, 2.333, 9.33),
    "BLW_1216": (2, "bwd_leaky", 1e-3, 3.041, 12.16),
}

SMOKE_STEPS = 30_000
STATE_HASH_STEP = 1_000_000
# 本モジュールが足すユニット別列。既存列は 1 列も変えない・消さない（spec §6.1）。
NEW_UNIT_KEYS = ("v_unit", "b_unit")


class SanityError(RuntimeError):
    """登録済みの前段チェックが落ちたとき。本走・集計を止める。"""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def _arm(cfg: dict, name: str) -> dict:
    return next(a for a in cfg["arms"] if a["name"] == name)


def _P(cfg: dict) -> dict:
    return cfg["bwd_leak"]


def _activation(cfg: dict, arm_cfg: dict) -> tuple[str, float]:
    """arm の activation ラベルを ``VecMLPL`` の (act, alpha) に写す。"""
    label = str(arm_cfg["activation"])
    if label == "relu":
        return "relu", 1.0
    spec = cfg["activation"][label]
    return str(spec["name"]), float(spec["slope"])


def validate_config(cfg: dict, *, stage: str) -> None:
    """凍結した設計からのずれをすべて ValueError にする。"""
    if stage not in {"preflight", "smoke", "run", "analyze"}:
        raise ValueError(f"unknown stage {stage!r}")
    C, A, I, P, G, S = (cfg["common"], cfg["condA"], cfg["intervention"],
                        cfg["phase1"], cfg["bwd_leak"], cfg["sanity"])
    if [a["name"] for a in cfg["arms"]] != list(ARM_ORDER):
        raise ValueError(f"arms must be ordered as {ARM_ORDER}")
    for arm in cfg["arms"]:
        want_stage, want_act, want_wd, want_target, want_dose = REGISTERED_ARMS[arm["name"]]
        if (int(arm["stage"]) != want_stage
                or str(arm["activation"]) != want_act
                or float(arm["wd_b"]) != want_wd
                or [int(v) for v in arm["hidden"]] != [100]
                or [int(v) for v in arm.get("centered_layers", [])] != [1]
                or _target(arm) != want_target
                or float(arm["target_dose"]) != want_dose):
            raise ValueError(f"{arm['name']} differs from the preregistration")
    if (int(A["m"]), int(A["f"]), int(A["target_hidden"])) != (20, 15, 100):
        raise ValueError("bwd_leak requires condA m=20, f=15, teacher width=100")
    if list(A["T_values"]) != [10_000] or list(A["encodings"]) != ["std"]:
        raise ValueError("bwd_leak requires T=10000 and std encoding")
    if int(C.get("generator_offset", -1)) != 0:
        raise ValueError("generator_offset must be an explicit 0 (spec §3)")
    if (str(I["name"]) != "oracle_fixed_mu_offset" or I["oracle"] is not True
            or I["consumes_rng"] is not False
            or float(I["center_alpha_compat"]) != 0.01
            or I["is_ema_centering"] is not False):
        raise ValueError("the oracle-dose intervention changed")
    act = cfg["activation"]
    if (str(act["relu"]["name"]) != "relu"
            or str(act["leaky"]["name"]) != "leaky_relu"
            or float(act["leaky"]["slope"]) != 0.1
            or str(act["bwd_leaky"]["name"]) != "bwd_leaky"
            or float(act["bwd_leaky"]["slope"]) != 0.1
            or str(act["bwd_leaky"]["forward"]) != "relu"
            or str(act["bwd_leaky"]["derivative"]) != "leaky"
            or str(act["fwd_leaky"]["name"]) != "fwd_leaky"
            or float(act["fwd_leaky"]["slope"]) != 0.1
            or str(act["fwd_leaky"]["forward"]) != "leaky"
            or str(act["fwd_leaky"]["derivative"]) != "relu"
            or act["bwd_leaky"]["is_true_gradient"] is not False
            or act["fwd_leaky"]["is_true_gradient"] is not False
            or act["autograd"] is not False
            or act["consumes_rng"] is not False):
        raise ValueError("activation definitions changed")
    if (float(cfg["bias_wd"]["wd_b_lambda"]) != 1e-3
            or str(cfg["bias_wd"]["applies_to"]) != "hidden_bias_only"
            or cfg["bias_wd"]["decoupled"] is not False
            or cfg["bias_wd"]["calibrated_for_this_system"] is not False):
        raise ValueError("bias weight-decay definition changed")
    expected_phase = {
        "task_period": 10_000, "early_tasks": [2, 11],
        "late_tasks_5m": [491, 500], "window_1m_tasks": [91, 100],
        "window_points_are_task_ends_only": True,
        "onset_threshold": 0.05, "onset_present_min": 5,
        "unfit_floor": 1e-16, "recalibrate_floor": False,
        "bootstrap_B": 10_000, "bootstrap_seed": 20_260_905,
        "ci_method": "percentile_primary_studentized_secondary",
    }
    for key, value in expected_phase.items():
        if P[key] != value:
            raise ValueError(f"phase1.{key} differs from the preregistration")
    if (float(P["degenerate_se_tol"]) != 1e-15
            or float(P["degenerate_frac_max"]) != 0.01
            or float(P["degenerate_width_ratio_max"]) != 100.0):
        raise ValueError("CI degeneracy guard changed")
    if (list(G["onset_states"]) != ["zero", "mid", "present"]
            or int(G["onset_zero_max"]) != 0 or int(G["onset_present_min"]) != 5):
        raise ValueError("onset state definition changed")
    if len(G["v1_map"]) != 9 or len(G["v2_map"]) != 9:
        raise ValueError("V1/V2 maps must enumerate all nine 3x3 cells")
    if _verdict_maps_disagree_with_spec_table(G):
        raise ValueError("the enumerated V1/V2 maps disagree with the spec table")
    if str(G["v2_not_applicable_when_bl"]) != "zero":
        raise ValueError("V2 NOT_APPLICABLE condition changed")
    if dict(G["control_expected_onset_5m"]) != {"R_933": 10, "R_1216": 10,
                                                "LR_933": 0, "LR_1216": 0}:
        raise ValueError("control expectations changed")
    if float(G["p5_equivalence_margin"]) != 0.15:
        raise ValueError("the P5 equivalence margin changed")
    if list(G["p5_labels"]["order"]) != ["equivalent", "short_of_lr", "inconclusive"]:
        raise ValueError("the P5 decision order changed")
    if (G["p5_margin_recalibrated_for_this_system"] is not False
            or G["p5_emit_ci_below_zero_flag"] is not True
            or G["p5_sign_test_report_only"] is not True):
        raise ValueError("P5 companion registration changed")
    if G["p6_emit_label"] is not False or G["p6_in_verdict"] is not False:
        raise ValueError("P6 must stay label-free and out of the verdict")
    if str(G["revival"]["primary_condition"]) != "same_task_and_flip_state_unchanged":
        raise ValueError("the revival definition changed (spec §6.2 追補 6)")
    sd = G["s_distribution"]
    if (str(sd["log_branch"]) != "A" or str(sd["window"]) != "late_tasks_5m"
            or list(sd["aggregation_order"]) != [
                "unit", "seed_internal_median_per_record",
                "mean_over_window_records", "median_over_seeds"]
            or list(sd["unit_sets"]) != ["all", "p_hat_positive"]
            or int(sd["min_units_per_seed_for_subset"]) != 3
            or sd["emit_both_median_s_and_median_M_plus_median_B"] is not True
            or sd["emit_denom_median"] is not True
            or sd["recompute_all_arms_from_unit_arrays"] is not True
            or sd["p7c_margin_is_not_dex"] is not True
            or sd["in_verdict"] is not False):
        raise ValueError("the P7 registration changed")
    if (int(S["s_pair_steps"]) != 30_000 or int(S["s_limit_steps"]) != 30_000
            or float(S["s_dose_rel_tol"]) != 1e-10
            or float(S["s_bwd_closed_form_tol"]) != 0.0
            or [float(v) for v in S["s_bwd_slope_probe"]] != [0.0, 0.1, 0.2, 1.0]
            or float(S["s_wd_lambda"]) != 1e-3
            or str(S["s_pair_match_by"]) != "seed_init_hash"
            or int(S["s_log_b_steps"]) != 30_000
            or S["s_taut_check"] is not True
            or S["s_ref_hash_check"] is not True
            or int(S["omp_num_threads"]) != 1
            or S["s6_floor_calibration"] is not False):
        raise ValueError("sanity gates changed")
    if stage in {"run", "analyze"}:
        if int(C["total_steps"]) != 5_000_000 or list(C["seeds"]) != list(range(10)):
            raise ValueError("the full run requires 5M steps and seeds 0..9")
        if str(C["device"]) != "cpu":
            raise ValueError("bwd_leak is CPU-only")


def _verdict_maps_disagree_with_spec_table(G: dict) -> bool:
    """列挙した写像が spec §5.1 の順序付き表と一致することを独立に検算する。

    追補 7: ワイルドカード入りの表を順序規則で実装すると ``mid`` セルが
    ``PARTIAL`` を飛ばしうる。config には全 9 セルを書き、ここでは**表の側**を
    素直に実装して両者を突き合わせる。片方を書き換えたらここで落ちる。
    """
    states = ("zero", "mid", "present")
    for bl in states:
        for fl in states:
            if bl == "zero" and fl == "present":
                want = "GRADIENT_CARRIES"
            elif bl == "present" and fl == "zero":
                want = "OUTPUT_CARRIES"
            elif bl == "zero" and fl == "zero":
                want = "EITHER_SUFFICES"
            elif bl == "present" and fl == "present":
                want = "BOTH_REQUIRED"
            else:
                want = "PARTIAL"
            if G["v1_map"].get(f"{bl}_{fl}") != want:
                return True
    for blw in states:
        for rw in states:
            if blw == "zero" and rw == "present":      # 表の 1 行目
                want = "RESTORING_FORCE_REQUIRED"
            elif rw == "zero":                          # 2 行目（BLW はワイルドカード）
                want = "WD_B_SUFFICIENT_ALONE"
            elif blw == "present":                      # 3 行目（RW はワイルドカード）
                want = "COMPROMISE_FAILS"
            else:
                want = "PARTIAL"
            if G["v2_map"].get(f"{blw}_{rw}") != want:
                return True
    return False


def _selected_arms(cfg: dict, stage: str, doses: str) -> list[str]:
    if stage == "all":
        arms = list(ARM_ORDER)
    else:
        arms = list(STAGE_ARMS[int(stage)])
    if doses != "both":
        arms = [a for a in arms if a.endswith("_" + doses)]
    if not arms:
        raise ValueError(f"no arms selected for stage={stage} doses={doses}")
    return arms


def _stage_outdir(cfg: dict, stage: str) -> Path:
    base = Path(ROOT) / cfg["output"]["dir"]
    if stage == "all" or not cfg["staging"]["outdir_per_stage"]:
        return base
    return base / f"stage{stage}"


# ---------------------------------------------------------------------------
# Learning path — gate_dose の経路をそのまま使い、活性化と wd_b だけ差し込む
# ---------------------------------------------------------------------------
def setup_arm_bwd(cfg: dict, arm_cfg: dict, device: str) -> dict:
    """``setup_arm_const`` の状態に活性化と b 限定 WD を差し込む。

    ``set_activation`` も ``set_weight_decay_b`` も乱数を消費せず状態も書き換え
    ないので、腕は ``gate_dose_0830`` と init・教師・入力列・flip が bit 一致する
    （S-pair がこれを実測する）。
    """
    st = setup_arm_const(cfg, arm_cfg, device)
    act, alpha = _activation(cfg, arm_cfg)
    st["net"].set_activation(act, alpha, "alpha_exp")
    st["net"].set_weight_decay_b(float(arm_cfg["wd_b"]))
    st["activation"] = act
    st["act_alpha"] = float(alpha)
    st["wd_b"] = float(arm_cfg["wd_b"])
    return st


class BwdRecorder(GateRecorder):
    """``GateRecorder`` にユニット別 ``v`` / ``b`` の読み出しを足したもの。

    §5.3 の凍結率（`BL` の $\\Delta v_i$、`FL` の $\\Delta b_i$）のためだけの追加。
    ``net.v`` と ``net.bs[0]`` を**読むだけ**で、乱数も状態も触らない。
    ``record_units=False`` にすると ``GateRecorder`` と完全に同じ挙動になり、
    S-log-b が 2 走の bit 一致を検査できる。
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
        net = st["net"]
        self.unit["v_unit"][i] = net.v.detach().cpu().numpy().astype(np.float32)
        self.unit["b_unit"][i] = net.bs[0].detach().cpu().numpy().astype(np.float32)


def write_arm_logs_bwd(outdir: Path, arm: str, st: dict,
                       rec: BwdRecorder) -> list[Path]:
    """``gate_dose.write_arm_logs_gate`` の列に ``wd_b`` と 2 つのユニット列を足す。"""
    logdir = outdir / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ri, run in enumerate(st["runs"]):
        seed = int(run["seed"])
        payload = dict(
            step=rec.steps, run_id=np.array(run["run_id"]), arm=np.array(arm),
            seed=np.int64(seed), activation=np.array(st["activation"]),
            act_alpha=np.float64(st["act_alpha"]), wd_b=np.float64(st["wd_b"]),
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
            payload[key] = value[:, ri]
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
    st = setup_arm_bwd(c, _arm(c, arm), device)
    every = int(c["common"]["lop_every"])
    probes = list(range(0, total + 1, every))
    if probes[-1] != total:
        probes.append(total)
    _, sanity0 = exact_layer_record_elu(st, SIGMA_TOL)
    if not identity_sanity_pass(sanity0, IDENTITY_TOL):
        raise SanityError(f"{arm} initial exact-support identity failed")
    rec = BwdRecorder(probes, st)
    checkpoints = [int(v) for v in c["common"].get("checkpoints", []) if int(v) <= total]
    print(f"[{arm}] act={st['activation']} alpha={st['act_alpha']:g} "
          f"wd_b={st['wd_b']:g} dose={st.get('target_dose')} seeds={seeds} "
          f"steps={total:,}", flush=True)
    started = time.time()
    try:
        elapsed = train_arm_gate(st, rec, probes, total, outdir, checkpoints)
    except NumericDivergenceError as exc:
        elapsed = time.time() - started
        event = dict(exc.event)
        event.update(probe_every=every, registered_total_steps=int(total),
                     registered_seeds=[int(v) for v in seeds],
                     activation=st["activation"], act_alpha=st["act_alpha"],
                     wd_b=st["wd_b"], elapsed_sec=float(elapsed),
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
    write_arm_logs_bwd(outdir, arm, st, rec)
    print(f"[{arm}] complete in {elapsed:.1f}s", flush=True)
    return dict(status="COMPLETE", elapsed_sec=elapsed, sanity=sanity)


# ---------------------------------------------------------------------------
# Preregistered sanity gates (spec §6)
# ---------------------------------------------------------------------------
def _grid(cfg: dict) -> torch.Tensor:
    lo, hi, n = cfg["sanity"]["s_cross_grid"]
    return torch.linspace(float(lo), float(hi), int(n), dtype=torch.float64)


def _probe_net(act: str, alpha: float) -> VecMLPL:
    net = VecMLPL(1, [2], 2, torch.Generator().manual_seed(0), "cpu")
    return net.set_activation(act, alpha, "alpha_exp")


def _s_cross(cfg: dict) -> dict:
    """S-cross: 新規 2 活性化が既存 relu / leaky の**組合せ**であること（bit 一致）。

    `bwd_leaky` は forward が relu・backward が leaky、`fwd_leaky` はその逆。
    新しい算術が入っていれば float64 の 4001 点グリッドで必ず落ちる。
    """
    grid = _grid(cfg)
    slope = float(cfg["activation"]["bwd_leaky"]["slope"])
    nets = {name: _probe_net(name, a) for name, a in
            (("relu", 1.0), ("leaky_relu", slope),
             ("bwd_leaky", slope), ("fwd_leaky", slope))}
    out = {}
    for name, net in nets.items():
        a = net.act_fn(grid)
        out[name] = (a, net.act_grad(grid, a))
    rows = [
        dict(pair="bwd_leaky.forward == relu.forward",
             equal=bool(torch.equal(out["bwd_leaky"][0], out["relu"][0]))),
        dict(pair="bwd_leaky.backward == leaky.backward",
             equal=bool(torch.equal(out["bwd_leaky"][1], out["leaky_relu"][1]))),
        dict(pair="fwd_leaky.forward == leaky.forward",
             equal=bool(torch.equal(out["fwd_leaky"][0], out["leaky_relu"][0]))),
        dict(pair="fwd_leaky.backward == relu.backward",
             equal=bool(torch.equal(out["fwd_leaky"][1], out["relu"][1]))),
        # 逆側も明示: 組合せが取り違えられていないこと
        dict(pair="bwd_leaky.forward != leaky.forward",
             equal=not bool(torch.equal(out["bwd_leaky"][0], out["leaky_relu"][0]))),
        dict(pair="fwd_leaky.backward != leaky.backward",
             equal=not bool(torch.equal(out["fwd_leaky"][1], out["leaky_relu"][1]))),
    ]
    return dict(pass_=all(r["equal"] for r in rows), grid_points=int(grid.numel()),
                slope=slope, rows=rows)


def _s_limit(cfg: dict, act: str, outdir: Path) -> dict:
    """S-limit: slope=0 の新規活性化が ReLU 経路と 30k 短走で bit 一致すること。"""
    steps = int(cfg["sanity"]["s_limit_steps"])
    c = copy.deepcopy(cfg)
    c["common"]["seeds"] = [0, 1]
    base = copy.deepcopy(_arm(c, "RW_933"))
    base["wd_b"] = 0.0                       # ここでは活性化だけを比べる
    relu = setup_arm_bwd(c, base, "cpu")
    other = setup_arm_bwd(c, base, "cpu")
    other["net"].set_activation(act, 0.0, "alpha_exp")
    other["activation"], other["act_alpha"] = act, 0.0
    grid = _grid(cfg)
    static_forward = bool(torch.equal(relu["net"].act_fn(grid), other["net"].act_fn(grid)))
    static_grad = bool(torch.equal(
        relu["net"].act_grad(grid, relu["net"].act_fn(grid)),
        other["net"].act_grad(grid, other["net"].act_fn(grid))))
    train_arm_gate(relu, lambda *_: None, [], steps, outdir, [])
    train_arm_gate(other, lambda *_: None, [], steps, outdir, [])
    a, b = _init_hashes(relu), _init_hashes(other)
    differences = sorted(k for k, v in a.items() if b.get(k) != v)
    return dict(pass_=bool(static_forward and static_grad and not differences),
                activation=act, steps=steps, static_forward_equal=static_forward,
                static_grad_equal=static_grad, trained_state_differences=differences)


def _s_wd_limit(cfg: dict, outdir: Path) -> dict:
    """S-limit の後半: ``set_weight_decay_b(0)`` が無介入と bit 一致すること。"""
    steps = int(cfg["sanity"]["s_limit_steps"])
    c = copy.deepcopy(cfg)
    c["common"]["seeds"] = [0, 1]
    zero = copy.deepcopy(_arm(c, "RW_933"))
    zero["wd_b"] = 0.0
    a_state = setup_arm_bwd(c, zero, "cpu")
    b_state = setup_arm_bwd(c, zero, "cpu")
    b_state["net"].wd_b = 0.0
    train_arm_gate(a_state, lambda *_: None, [], steps, outdir, [])
    train_arm_gate(b_state, lambda *_: None, [], steps, outdir, [])
    ha, hb = _init_hashes(a_state), _init_hashes(b_state)
    differences = sorted(k for k, v in ha.items() if hb.get(k) != v)
    return dict(pass_=not differences, steps=steps,
                trained_state_differences=differences)


def _s_bwd(cfg: dict) -> dict:
    """S-bwd: 手置きの負側ユニットで §2 の表どおりの更新になること。

    代替勾配は損失の勾配ではないので有限差分照合は成立しない（spec §6）。
    spec §6.2 追補 2 の 5 点を検査する。**照合は勾配の水準で bit 一致を要求する。**
    パラメタの差分 ``after - before`` は float32 の減算 1 回ぶん丸まるので、
    ``after == before - lr*(g + wd_b*b)`` という**適用後の値**の側で bit 一致を取る。
    """
    slope = float(cfg["activation"]["bwd_leaky"]["slope"])
    lr_value = float(cfg["common"]["lr_main"])
    probes = [float(v) for v in cfg["sanity"]["s_bwd_slope_probe"]]
    R, h, d = 1, 3, 4
    g = torch.Generator().manual_seed(90211)
    x = torch.randn(R, d, generator=g)
    y = torch.randn(R, generator=g)
    lr = torch.full((R,), lr_value)

    def fresh(act: str, a: float, wd_b: float = 0.0) -> VecMLPL:
        net = VecMLPL(R, [h], d, torch.Generator().manual_seed(7), "cpu")
        net.set_activation(act, a, "alpha_exp")
        net.set_weight_decay_b(wd_b)
        # ユニット 0・2 を負側、ユニット 1 を正側に手で置く。
        net.Ws[0].zero_()
        net.Ws[0][0, 0, 0] = 1.0
        net.Ws[0][0, 1, 1] = 1.0
        net.bs[0].zero_()
        net.bs[0][0, 0] = -3.0 - float(x[0, 0])
        net.bs[0][0, 1] = 3.0 - float(x[0, 1])
        net.bs[0][0, 2] = -1.0
        net.v.copy_(torch.tensor([[0.7, -0.4, 0.25]]))
        net.c.zero_()
        net.W, net.b = net.Ws[0], net.bs[0]
        return net

    def grads(net: VecMLPL):
        pres, acts, yhat = net.forward_layers(x)
        delta = yhat - y
        gWs, gbs, gv, gc = grads_centered_elu(net, [x], pres, acts, delta)
        return dict(pre=pres[0], act=acts[0], delta=delta, gW=gWs[0], gb=gbs[0],
                    gv=gv, gc=gc)

    rows, failures = [], []

    # --- BL: 出力 0 のまま勾配だけ a 倍で通る ---
    bl = fresh("bwd_leaky", slope)
    gbl = grads(bl)
    neg = (gbl["pre"][0] <= 0).nonzero().flatten().tolist()
    pos = (gbl["pre"][0] > 0).nonzero().flatten().tolist()
    if not neg or not pos:
        failures.append(dict(where="bl_setup", negative=neg, positive=pos))
    d2 = 2.0 * gbl["delta"]
    want_gb = d2[:, None] * bl.v * slope
    want_gW = want_gb[:, :, None] * x[:, None, :]
    a_ok = bool(torch.equal(gbl["gv"][0, neg], torch.zeros(len(neg))))
    b_ok = bool(torch.equal(gbl["gb"][0, neg], want_gb[0, neg])
                and torch.equal(gbl["gW"][0, neg], want_gW[0, neg]))

    # (c) 負側の勾配が a に厳密比例する。forward が ReLU なので delta は a に依らない。
    ref = grads(fresh("bwd_leaky", 1.0))
    c_ok, ratio_rows = True, []
    for a in probes:
        got = grads(fresh("bwd_leaky", a))
        delta_same = bool(torch.equal(got["delta"], ref["delta"]))
        prop = bool(torch.equal(got["gb"][0, neg], ref["gb"][0, neg] * a))
        pos_same = bool(torch.equal(got["gb"][0, pos], ref["gb"][0, pos]))
        ratio_rows.append(dict(slope=a, delta_slope_independent=delta_same,
                               negative_gb_proportional=prop,
                               positive_gb_unchanged=pos_same,
                               gb_negative=got["gb"][0, neg].tolist()))
        if not (delta_same and prop and pos_same):
            c_ok = False
            failures.append(dict(where="bl_proportionality", slope=a,
                                 delta=delta_same, prop=prop, pos=pos_same))

    # 適用後の値の側で bit 一致（sgd_step_layers の配線確認）
    step_net = fresh("bwd_leaky", slope)
    before = step_net.state_dict()
    gs = grads(step_net)
    want_b = before["b"] - lr[:, None] * (gs["gb"] + step_net.wd_b * before["b"])
    want_W = before["W"] - lr[:, None, None] * gs["gW"]
    want_v = before["v"] - lr[:, None] * gs["gv"]
    step_net.sgd_step_layers(lr, [gs["gW"]], [gs["gb"]], gs["gv"], gs["gc"])
    after = step_net.state_dict()
    step_ok = bool(torch.equal(after["b"], want_b) and torch.equal(after["W"], want_W)
                   and torch.equal(after["v"], want_v))
    v_unchanged = bool(torch.equal(after["v"][0, neg], before["v"][0, neg]))
    rows.append(dict(arm="BL", negative_units=neg, positive_units=pos,
                     gv_zero_on_negative=a_ok, gb_gW_closed_form=b_ok,
                     slope_proportional=c_ok, applied_step_bit_exact=step_ok,
                     v_unchanged_on_negative=v_unchanged, by_slope=ratio_rows))
    if not (a_ok and b_ok and c_ok and step_ok and v_unchanged):
        failures.append(dict(where="BL", a=a_ok, b=b_ok, c=c_ok, step=step_ok,
                             v=v_unchanged))

    # --- FL: w,b は吸収され、v だけが a*z で学習を続ける ---
    fl = fresh("fwd_leaky", slope)
    gfl = grads(fl)
    neg_f = (gfl["pre"][0] <= 0).nonzero().flatten().tolist()
    d_ok = bool(torch.equal(gfl["gb"][0, neg_f], torch.zeros(len(neg_f)))
                and torch.equal(gfl["gW"][0, neg_f],
                                torch.zeros(len(neg_f), x.shape[1])))
    want_gv = (2.0 * gfl["delta"])[:, None] * gfl["act"]
    e_ok = bool(torch.equal(gfl["gv"][0, neg_f], want_gv[0, neg_f]))
    act_ok = bool(torch.equal(gfl["act"][0, neg_f], slope * gfl["pre"][0, neg_f]))
    nonzero_v = bool(torch.all(gfl["gv"][0, neg_f] != 0.0))
    step_fl = fresh("fwd_leaky", slope)
    bef_f = step_fl.state_dict()
    gs_f = grads(step_fl)
    step_fl.sgd_step_layers(lr, [gs_f["gW"]], [gs_f["gb"]], gs_f["gv"], gs_f["gc"])
    aft_f = step_fl.state_dict()
    frozen = bool(torch.equal(aft_f["b"][0, neg_f], bef_f["b"][0, neg_f])
                  and torch.equal(aft_f["W"][0, neg_f], bef_f["W"][0, neg_f]))
    rows.append(dict(arm="FL", negative_units=neg_f, gb_gW_zero_on_negative=d_ok,
                     gv_closed_form=e_ok, forward_is_leaky=act_ok,
                     gv_nonzero_on_negative=nonzero_v,
                     w_b_unchanged_on_negative=frozen))
    if not (d_ok and e_ok and act_ok and nonzero_v and frozen):
        failures.append(dict(where="FL", d=d_ok, e=e_ok, act=act_ok,
                             nonzero_v=nonzero_v, frozen=frozen))

    return dict(pass_=not failures,
                tolerance=float(cfg["sanity"]["s_bwd_closed_form_tol"]),
                comparison="bit identity at the gradient level and on the "
                           "post-step parameter values, not on differences",
                slope=slope, rows=rows, failures=failures)


def _s_wd(cfg: dict) -> dict:
    """S-wd: step 1 で ``RW`` と ``R`` の差が b のみ・厳密に ``-lr*lambda*b``。

    ``bias_wd_0901`` の S1/S2 と同じ形。b=0 初期化だと WD が恒等になるので、
    b を乱数で埋めてから 1 step 進める。
    """
    lam = float(cfg["sanity"]["s_wd_lambda"])
    ulp_factor = float(cfg["sanity"]["s_wd_ulp_factor"])
    lr_value = float(cfg["common"]["lr_main"])
    R, hidden, d = 4, [100], int(cfg["condA"]["m"])
    gen = torch.Generator().manual_seed(20260905)
    ref = VecMLPL(R, hidden, d, torch.Generator().manual_seed(11), "cpu")
    ref.bs[0].copy_(torch.randn(ref.bs[0].shape, generator=gen))
    state = ref.state_dict()
    lr = torch.full((R,), lr_value)
    gW = torch.randn(R, hidden[0], d, generator=gen)
    gb = torch.randn(R, hidden[0], generator=gen)
    gv = torch.randn(R, hidden[0], generator=gen)
    gc = torch.randn(R, generator=gen)

    def stepped(wd_b: float) -> VecMLPL:
        net = VecMLPL(R, hidden, d, torch.Generator().manual_seed(11), "cpu")
        net.load_state(state)
        net.set_weight_decay_b(wd_b)
        net.sgd_step_layers(lr, [gW], [gb], gv, gc)
        return net

    zero, decayed = stepped(0.0), stepped(lam)
    untouched = bool(torch.equal(zero.Ws[0], decayed.Ws[0])
                     and torch.equal(zero.v, decayed.v)
                     and torch.equal(zero.c, decayed.c))
    expected = -lr[:, None] * lam * state["b"]
    err = float((decayed.bs[0] - zero.bs[0] - expected).abs().max())
    eps = float(torch.finfo(zero.bs[0].dtype).eps)
    tol = ulp_factor * eps * float(state["b"].abs().max())
    signal = float(expected.abs().max())
    return dict(pass_=bool(untouched and err <= tol and signal > 0),
                lam=lam, W_v_c_untouched=untouched, bias_delta_max_abs_err=err,
                bias_delta_tol_ulp=tol, bias_delta_signal=signal)


def _s_pair_and_dose(cfg: dict, outdir: Path, arms: list[str]) -> dict:
    """S-pair / S-dose: 新規腕どうし・親走との init/教師/入力列/flip の bit 一致。

    対応は **seed ごとのハッシュ**で取る（位置合わせではない）。seed ラベルは
    乱数系列に入らないので、位置で照合すると seed をずらした事故を見逃す。

    比較の粒度に注意:

    * ``net.*`` / ``teacher.*`` / ``env.*`` は用量にも活性化にも WD にも依らない。
      **8 腕すべてで一致**しなければならない
    * ``running_mean`` は用量固定のオラクルオフセットそのものなので**用量ごとに
      違って当然**。同一用量の腕どうしでのみ一致を要求する
    """
    steps = int(cfg["sanity"]["s_pair_steps"])
    every = int(cfg["common"]["lop_every"])
    seeds = [int(v) for v in cfg["common"]["seeds"]]
    init, final, streams, dose_rows = {}, {}, {}, []
    per_seed: dict[str, dict[int, dict]] = {}
    flip0: dict[str, dict[int, np.ndarray]] = {}
    for arm in arms:
        c = copy.deepcopy(cfg)
        st = setup_arm_bwd(c, _arm(c, arm), "cpu")
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
    dose_of = {a: str(_arm(cfg, a)["target_dose"]) for a in arms}
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
        same_dose = dose_of[arm] == dose_of[reference]
        for seed in seeds:
            for key, value in per_seed[reference][seed].items():
                # running_mean は用量固定のオフセット。用量が違えば違って当然。
                if key.startswith("running_mean") and not same_dose:
                    continue
                if per_seed[arm][seed].get(key) != value:
                    differences.append(dict(arm=arm, seed=seed,
                                            where=f"seed_hash.{key}"))
    # 用量ごとの running_mean は「群内で一致し、群間で違う」ことまで見る
    dose_groups: dict[str, list[str]] = {}
    for arm in arms:
        dose_groups.setdefault(dose_of[arm], []).append(arm)
    dose_offsets = {}
    for dose, members in dose_groups.items():
        head = members[0]
        dose_offsets[dose] = per_seed[head][seeds[0]].get("running_mean")
        for arm in members[1:]:
            for seed in seeds:
                if (per_seed[arm][seed].get("running_mean")
                        != per_seed[head][seed].get("running_mean")):
                    differences.append(dict(arm=arm, seed=seed,
                                            where="running_mean_within_dose"))
    distinct_offsets = len(set(dose_offsets.values())) == len(dose_offsets)
    if len(dose_offsets) > 1 and not distinct_offsets:
        differences.append(dict(where="dose_offsets_not_distinct"))

    # 親走との照合: step 0 の flip_state が committed ログの 1 行目と bit 一致。
    # 腕は 10 seed でまとめて構築したものの行 ri を使う（R を変えると SCREnv の
    # 引きが変わるので、seed 1 本ずつ作り直して比べてはいけない）。
    parent = Path(ROOT) / cfg["sanity"]["s_pair_reference"] / "logs"
    arm_map = dict(cfg["sanity"]["s_pair_arm_map"])
    parent_rows, parent_missing = [], []
    for arm in arms:
        ref_arm = arm_map[arm]
        for seed in seeds:
            path = parent / f"{ref_arm}_seed{seed}.npz"
            if not path.exists():
                parent_missing.append(str(path))
                continue
            with np.load(path, allow_pickle=False) as z:
                ref_flip = z["flip_state"][0].copy()
                ref_state = json.loads(str(z["state_hash_final"]))
            same = bool(np.array_equal(flip0[arm][seed], ref_flip))
            parent_rows.append(dict(arm=arm, reference_arm=ref_arm, seed=seed,
                                    flip_state_equal=same,
                                    parent_has_state_hash=bool(ref_state)))
            if not same:
                differences.append(dict(arm=arm, seed=seed, where="parent.flip_state"))

    tol = float(cfg["sanity"]["s_dose_rel_tol"])
    dose_fail = [r for r in dose_rows if r["target_mu_norm"] is not None
                 and float(r["max_relative_error"]) > tol]
    return dict(
        spair=dict(pass_=bool(not differences and not parent_missing),
                   reference=reference, arms=list(arms), steps=steps,
                   match_by="seed_init_hash", differences=differences,
                   dose_groups=dose_groups,
                   dose_offsets_distinct=distinct_offsets,
                   parent_flip_rows=parent_rows, parent_missing=parent_missing,
                   caveat="init/teacher/input realization only; trajectories "
                          "diverge after step 1"),
        sdose=dict(pass_=not dose_fail, tolerance=tol, n_probes=len(dose_rows),
                   failures=dose_fail))


def _s_log_b(cfg: dict, outdir: Path) -> dict:
    """S-log-b: 追加ロガー（``v_unit`` / ``b_unit``）が軌道中立であること。

    30k 短走を ``record_units`` の有無で 2 回回し、最終状態と**既存の全列**が
    bit 一致することを要求する。既存列を 1 列も変えないことも同時に見る。
    """
    steps = int(cfg["sanity"]["s_log_b_steps"])
    every = int(cfg["common"]["lop_every"])
    c = copy.deepcopy(cfg)
    c["common"]["seeds"] = [0, 1]
    probes = list(range(0, steps + 1, every))
    results = {}
    for label, record_units in (("with_logger", True), ("without_logger", False)):
        st = setup_arm_bwd(c, _arm(c, "BL_933"), "cpu")
        rec = BwdRecorder(probes, st, record_units=record_units)
        train_arm_gate(st, rec, probes, steps, outdir, [])
        results[label] = dict(
            state=_init_hashes(st), env=_env_hashes(st),
            run={k: _sha_array(v) for k, v in rec.run.items()},
            layers=[{k: _sha_array(v) for k, v in layer.items()}
                    for layer in rec.layers],
            extra={k: _sha_array(v) for k, v in rec.extra.items()},
            flip=_sha_array(rec.flip_state),
            unit_keys=sorted(rec.unit))
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


def _s_taut(cfg: dict, outdir: Path) -> dict:
    """S-taut: 未フィット率が介入で定義上恒真になっていないこと＋判定表の検算。"""
    c = copy.deepcopy(cfg)
    c["common"]["seeds"] = [0, 1]
    values, hashes = {}, {}
    for arm in ("RW_933", "BL_933", "FL_933"):
        st = setup_arm_bwd(c, _arm(c, arm), "cpu")
        train_arm_gate(st, lambda *_: None, [], 2000, outdir, [])
        rec, _ = exact_layer_record_elu(st, SIGMA_TOL)
        values[arm] = rec["run"]["unfit"].detach().cpu().numpy().tolist()
        hashes[arm] = {k: _sha_array(v) for k, v in st["net"].state_dict().items()}
    changes_state = all(hashes[a] != hashes["RW_933"] for a in ("BL_933", "FL_933"))
    finite = all(np.isfinite(np.asarray(v)).all() and (np.asarray(v) > 0).all()
                 for v in values.values())
    G = _P(cfg)
    mutants = {
        "grad": _v1_label(G, "zero", "present"),
        "out": _v1_label(G, "present", "zero"),
        "either": _v1_label(G, "zero", "zero"),
        "both": _v1_label(G, "present", "present"),
        "partial": _v1_label(G, "mid", "present"),
        "v2_restore": _v2_label(G, "present", "zero", "present"),
        "v2_wd_alone": _v2_label(G, "present", "present", "zero"),
        "v2_fails": _v2_label(G, "present", "present", "present"),
        "v2_na": _v2_label(G, "zero", "present", "present"),
    }
    expected = dict(grad="GRADIENT_CARRIES", out="OUTPUT_CARRIES",
                    either="EITHER_SUFFICES", both="BOTH_REQUIRED",
                    partial="PARTIAL", v2_restore="RESTORING_FORCE_REQUIRED",
                    v2_wd_alone="WD_B_SUFFICIENT_ALONE",
                    v2_fails="COMPROMISE_FAILS", v2_na="NOT_APPLICABLE")
    return dict(pass_=bool(changes_state and finite and mutants == expected),
                activation_changes_state=changes_state, unfit_finite_positive=finite,
                short_run_unfit=values, verdict_mutants=mutants, expected=expected)


def _endpoint_columns_unchanged(cfg: dict, ref_rel: str, want_sha: str) -> dict:
    """転記する列が provenance 記録時の版から 1 バイトも動いていないことの確認。

    親走の ``verdict.csv`` は provenance 記録後に別 commit で再生成されうる
    （``--analyze-only`` は provenance を書き直さない）。ファイルのハッシュが
    合わないこと自体は事故とは限らないが、**本走が転記する列**が動いていたら
    事故である。``provenance.output_sha256`` に一致する版を履歴から探し出し、
    その blob と現行版を列単位で突き合わせる。``git_hash`` は走の**開始時**の
    commit なので、そこに成果物が存在するとは限らない（探索でこれを避ける）。
    """
    import csv
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
                current_head_differs=True, columns_transcribed=columns,
                arms=list(CONTROL_ORDER), differing=differing, missing=missing,
                columns_that_changed_anywhere=changed_columns,
                unchanged=bool(not differing and not missing))


def _s_ref(cfg: dict) -> dict:
    """S-ref: 対照として読む親走の出力が親 ``provenance.json`` と全数一致すること。

    ``logs/*.npz`` は ``.gitignore`` 対象なので、fresh clone では P7b / P7c を
    再現できない。使う前にハッシュで同一性を確かめ、親 commit がリモートに
    在ることも確認する（運用ルール §2 の push 監査）。

    ハッシュが合わないファイルは、**本走が実際に読むかどうか**で扱いを分ける。
    読まないファイル（``summary.md`` など）の再生成は情報として記録するだけ。
    読むファイル（``verdict.csv``）は列単位で provenance 当時の版と照合し、
    転記する列が動いていれば FAIL にする。
    """
    ref_rel = str(cfg["controls"]["reference_run"])
    ref_dir = (Path(ROOT) / ref_rel).resolve()
    prov_path = ref_dir / "provenance.json"
    if not prov_path.exists():
        return dict(pass_=False, reason="missing provenance", path=str(prov_path))
    prov = json.loads(prov_path.read_text(encoding="utf-8"))
    recorded = dict(prov.get("output_sha256", {}))
    parent_sha = prov.get("git_hash")

    # 本走が実際に読むファイル
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

    read_mismatches = [n for n in mismatches if n in read_files]
    other_mismatches = [n for n in mismatches if n not in read_files]
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
                read_files_count=len(read_files),
                hash_mismatches_on_read_files=read_mismatches,
                hash_mismatches_on_unread_files=other_mismatches,
                verdict_column_check=column_check,
                recorded_but_missing=missing, required_but_absent=absent,
                note="logs/*.npz are gitignored; P7b/P7c are not reproducible "
                     "from a fresh clone (spec §6.2 追補 4). A hash mismatch on a "
                     "file this run never reads is recorded, not fatal; verdict.csv "
                     "is additionally checked column-by-column against the "
                     "provenance-era blob.")


def _s_log_branch(cfg: dict) -> dict:
    """S-log: ユニット別 M / B / denom が親走の committed ログに在ること（分岐 A）。"""
    ref = Path(ROOT) / cfg["controls"]["reference_run"] / "logs"
    want = list(cfg["controls"]["unit_arrays"])
    rows, failures = [], []
    for arm in CONTROL_ORDER:
        for seed in [int(v) for v in cfg["common"]["seeds"]]:
            path = ref / f"{arm}_seed{seed}.npz"
            if not path.exists():
                failures.append(dict(arm=arm, seed=seed, reason="missing"))
                continue
            with np.load(path, allow_pickle=False) as z:
                shapes = {k: (list(z[k].shape) if k in z.files else None) for k in want}
                nan = {k: int(np.isnan(z[k]).sum()) for k in ("layer1_M", "layer1_B")
                       if k in z.files}
                dmin = float(z["layer1_denom"].min()) if "layer1_denom" in z.files else float("nan")
            bad = [k for k, v in shapes.items() if v is None or len(v) != 2]
            rows.append(dict(arm=arm, seed=seed, shapes=shapes, nan_counts=nan,
                             denom_min=dmin))
            if bad:
                failures.append(dict(arm=arm, seed=seed, missing_or_not_per_unit=bad))
    return dict(pass_=not failures, branch="A" if not failures else "C",
                registered_branch=str(_P(cfg)["s_distribution"]["log_branch"]),
                reason="per-unit layer1_M/B/denom/p_hat/zbar/w_norm are already "
                       "stored at every record point in the committed parent logs; "
                       "no re-run and no logger change is needed for P7a-P7d",
                sigma_tol=SIGMA_TOL, rows=rows, failures=failures)


def _s_floor_inheritance(cfg: dict) -> dict:
    reference = (Path(ROOT) / cfg["controls"]["reference_run"] / "floor_calibration.csv")
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
    print("[S-cross] surrogate activations are the existing halves", flush=True)
    checks["S_cross"] = _s_cross(cfg)
    print("[S-bwd] hand-placed negative unit", flush=True)
    checks["S_bwd"] = _s_bwd(cfg)
    print("[S-wd] one step of b-only weight decay", flush=True)
    checks["S_wd"] = _s_wd(cfg)
    print("[S-log] per-unit M/B/denom in the committed parent logs", flush=True)
    checks["S_log"] = _s_log_branch(cfg)
    print("[S-ref] parent output hashes", flush=True)
    checks["S_ref"] = _s_ref(cfg)
    print("[S-limit] bwd_leaky slope -> 0", flush=True)
    checks["S_limit_bwd"] = _s_limit(cfg, "bwd_leaky", outdir / "slimit_bwd")
    print("[S-limit] fwd_leaky slope -> 0", flush=True)
    checks["S_limit_fwd"] = _s_limit(cfg, "fwd_leaky", outdir / "slimit_fwd")
    print("[S-limit] wd_b -> 0", flush=True)
    checks["S_limit_wd"] = _s_wd_limit(cfg, outdir / "slimit_wd")
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


# ---------------------------------------------------------------------------
# Verdict labels (spec §5.1)
# ---------------------------------------------------------------------------
def _onset_state(onsets: list[int], zero_max: int, present_min: int) -> str:
    """腕族の発症状態を 3 値に潰す。回した用量すべてを見る。"""
    if not onsets:
        return "missing"
    if all(int(v) <= zero_max for v in onsets):
        return "zero"
    if any(int(v) >= present_min for v in onsets):
        return "present"
    return "mid"


def _v1_label(G: dict, bl: str, fl: str) -> str:
    return str(G["v1_map"][f"{bl}_{fl}"])


def _v2_label(G: dict, bl: str, blw: str, rw: str) -> str:
    if bl == str(G["v2_not_applicable_when_bl"]):
        return "NOT_APPLICABLE"
    return str(G["v2_map"][f"{blw}_{rw}"])


def _v2_co_satisfied(blw: str, rw: str) -> list[str]:
    """spec §5.1 の V2 表で**条件を満たしていた行**をすべて挙げる（追補 7）。"""
    hits = []
    if blw == "zero" and rw == "present":
        hits.append("RESTORING_FORCE_REQUIRED")
    if rw == "zero":
        hits.append("WD_B_SUFFICIENT_ALONE")
    if blw == "present":
        hits.append("COMPROMISE_FAILS")
    return hits or ["PARTIAL"]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
def _load_controls(cfg: dict) -> dict:
    """対照の主 endpoint を親走の ``verdict.csv`` から**転記**する（再計算しない）。"""
    import csv

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
    from .gate_dose import _load_arm

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
    """seed 別符号検定（REPORT_ONLY・追補 1）。較正定数が要らない。"""
    values = np.asarray(values, dtype=np.float64)
    pos = int((values > 0).sum())
    neg = int((values < 0).sum())
    ties = int((values == 0).sum())
    n = pos + neg
    if n == 0:
        return dict(n_positive=pos, n_negative=neg, n_ties=ties, p_two_sided=float("nan"))

    def tail(k: int) -> float:
        return sum(math.comb(n, i) for i in range(k + 1)) / (2.0 ** n)

    p = 2.0 * min(tail(min(pos, neg)), 1.0)
    return dict(n_positive=pos, n_negative=neg, n_ties=ties,
                p_two_sided=float(min(p, 1.0)))


def _p5_label(G: dict, ci: dict) -> tuple[str, bool]:
    """spec §5.2 の書かれた順に判定する。CI が丸ごと 0 の下ならフラグを立てる。"""
    margin = float(G["p5_equivalence_margin"])
    lo, hi = float(ci["percentile_ci_lo"]), float(ci["percentile_ci_hi"])
    below = bool(hi < 0.0)
    if lo >= -margin and hi <= margin:
        return str(G["p5_labels"]["equivalent"]), below
    if lo > 0.0:
        return str(G["p5_labels"]["short_of_lr"]), below
    return str(G["p5_labels"]["inconclusive"]), below


# ---------------------------------------------------------------------------
# §5.3 REPORT_ONLY: 復活数・凍結率・FL の固定特徴
# ---------------------------------------------------------------------------
def _unit_log(outdir: Path, arm: str, seed: int) -> Path:
    return outdir / "logs" / f"{arm}_seed{seed}.npz"


def _revival_counts(path: Path) -> dict:
    """p_hat が 0 -> 正 になった件数。同一タスク内 / 境界越え を分けて数える。

    追補 6: タスク境界では 32 パターンの厳密支持が引き直されるので、1 ステップも
    動いていない dead ユニットが p_hat > 0 になる。「ReLU 腕は 0」が成り立つのは
    同一タスク内かつ flip 不変の定義のみ。
    """
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
    n_units = p.shape[1]
    return dict(
        events_within_task=int(revived[within].sum()),
        events_across_boundary=int(revived[~within].sum()),
        units_within_task=int((revived[within].any(axis=0)).sum()),
        units_across_boundary=int((revived[~within].any(axis=0)).sum()),
        opportunities_within_task=int(dead[:-1][within].sum()),
        opportunities_across_boundary=int(dead[:-1][~within].sum()),
        n_units=n_units, n_records=int(p.shape[0]))


def _freeze_rates(path: Path, cfg: dict) -> dict:
    """末尾窓で ``v_unit`` / ``b_unit`` が 1 度も動かなかったユニットの割合。

    ``gW = gb (x) x`` なので ``gb == 0`` と ``gW == 0`` は同値。したがって
    `FL` の $w,b$ 凍結は ``b_unit`` の不動で言える。
    """
    P = cfg["phase1"]
    with np.load(path, allow_pickle=False) as z:
        if "v_unit" not in z.files:
            return dict(status="NO_UNIT_LOG")
        step = z["step"].astype(np.int64)
        idx = _window_indices(step, int(P["task_period"]), list(P["late_tasks_5m"]))
        lo, hi = int(idx.min()), int(idx.max())
        span = slice(lo, hi + 1)
        v, b, p = z["v_unit"][span], z["b_unit"][span], z["layer1_p_hat"][span]
    v_frozen = (np.diff(v, axis=0) == 0.0).all(axis=0)
    b_frozen = (np.diff(b, axis=0) == 0.0).all(axis=0)
    dead_all = (p == 0.0).all(axis=0)
    n = v.shape[1]
    return dict(status="OK", n_units=n, window_records=int(v.shape[0]),
                v_frozen_frac=float(v_frozen.mean()),
                b_frozen_frac=float(b_frozen.mean()),
                strict_dead_all_window_frac=float(dead_all.mean()),
                v_frozen_matches_dead=bool(np.array_equal(v_frozen, dead_all)),
                b_frozen_matches_dead=bool(np.array_equal(b_frozen, dead_all)),
                v_frozen_xor_dead=int((v_frozen != dead_all).sum()),
                b_frozen_xor_dead=int((b_frozen != dead_all).sum()))


def _submerged_abs_v(path: Path, cfg: dict) -> float:
    """沈んだユニットの |v| の末尾窓中央値（`FL` の固定特徴の傍証・§5.3）。"""
    P = cfg["phase1"]
    with np.load(path, allow_pickle=False) as z:
        if "v_unit" not in z.files:
            return float("nan")
        step = z["step"].astype(np.int64)
        idx = _window_indices(step, int(P["task_period"]), list(P["late_tasks_5m"]))
        v, p = z["v_unit"][idx], z["layer1_p_hat"][idx]
    mask = p == 0.0
    if not mask.any():
        return float("nan")
    return float(np.median(np.abs(v[mask])))


# ---------------------------------------------------------------------------
# §5.4 P7: unit-level s = M + B
# ---------------------------------------------------------------------------
def _p7_seed_values(path: Path, cfg: dict) -> dict:
    """1 seed 分の P7 系列。集約順は追補 5 で凍結したとおり。

    ユニット -> 記録点ごとの seed 内中央値（nan 対応） -> 窓内記録点の平均。
    ``all`` と ``p_hat_positive`` の 2 系列。生の量も併記する。
    """
    P, sd = cfg["phase1"], _P(cfg)["s_distribution"]
    min_units = int(sd["min_units_per_seed_for_subset"])
    with np.load(path, allow_pickle=False) as z:
        step = z["step"].astype(np.int64)
        idx = _window_indices(step, int(P["task_period"]), list(P["late_tasks_5m"]))
        M = z["layer1_M"][idx].astype(np.float64)
        B = z["layer1_B"][idx].astype(np.float64)
        denom = z["layer1_denom"][idx].astype(np.float64)
        zbar = z["layer1_zbar"][idx].astype(np.float64)
        p_hat = z["layer1_p_hat"][idx].astype(np.float64)
    s = M + B
    wmu, b_raw = M * denom, B * denom
    series = {"median_s": s, "median_M": M, "median_B": B, "median_denom": denom,
              "median_zbar": zbar, "median_wmu": wmu, "median_b_raw": b_raw}
    out: dict[str, float] = {}
    with np.errstate(invalid="ignore"):
        for name, arr in series.items():
            out[f"{name}__all"] = float(np.nanmean(np.nanmedian(arr, axis=1)))
        alive = p_hat > 0.0
        counts = alive.sum(axis=1)
        ok = counts >= min_units
        for name, arr in series.items():
            if not ok.any():
                out[f"{name}__p_hat_positive"] = float("nan")
                continue
            masked = np.where(alive, arr, np.nan)
            per_record = np.nanmedian(masked[ok], axis=1)
            out[f"{name}__p_hat_positive"] = float(np.nanmean(per_record))
    out["n_records"] = int(len(idx))
    out["n_na_M"] = int(np.isnan(M).sum())
    out["n_na_B"] = int(np.isnan(B).sum())
    out["alive_records_used"] = int(ok.sum())
    out["alive_units_median"] = float(np.median(counts))
    # 縮退 (ii): median_s と median_M + median_B は別物。両方出す
    out["median_M_plus_median_B__all"] = out["median_M__all"] + out["median_B__all"]
    out["median_M_plus_median_B__p_hat_positive"] = (
        out["median_M__p_hat_positive"] + out["median_B__p_hat_positive"])
    return out


def _p7_arm(cfg: dict, arm: str, log_dir: Path) -> dict:
    seeds = [int(v) for v in cfg["common"]["seeds"]]
    per_seed = [_p7_seed_values(log_dir / f"{arm}_seed{s}.npz", cfg) for s in seeds]
    keys = [k for k in per_seed[0] if k.startswith("median")]
    out = {k: np.asarray([row[k] for row in per_seed], dtype=np.float64) for k in keys}
    out["_meta"] = per_seed
    return out


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def analyze(cfg: dict, outdir: Path, arms: list[str], stage: str, sanity: dict,
            elapsed: dict, divergences: dict, stage1_dir: Path | None) -> dict:
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
    onset = {w: {a: int(np.sum(windows[a][w]["raw_u"] >= threshold)) for a in complete}
             for w in ("1M", "5M")}

    # --- 腕族の 3 値状態 ---
    zero_max, present_min = int(G["onset_zero_max"]), int(G["onset_present_min"])
    states, family_onsets = {}, {}
    for family, members in FAMILY_ARMS.items():
        if family in ("R", "LR"):
            values = [controls[m]["n_onset_5m"] for m in members]
        else:
            values = [onset["5M"][m] for m in members if m in onset["5M"]]
        family_onsets[family] = values
        states[family] = _onset_state(values, zero_max, present_min)

    # 段 2 だけを回した場合、V2 は段 1 の BL 状態を要する
    bl_state = states["BL"]
    bl_source = "this_run"
    if bl_state == "missing" and stage1_dir is not None:
        prior = stage1_dir / "verdict.csv"
        if prior.exists():
            import csv
            with prior.open(newline="") as fh:
                rows = [r for r in csv.DictReader(fh)]
            if rows and rows[0].get("BL_onset_state"):
                bl_state = rows[0]["BL_onset_state"]
                bl_source = str(prior)

    div_families = {f for f, members in FAMILY_ARMS.items()
                    if any(m in divergences for m in members)}
    v1_required = [str(v) for v in G["numeric_divergence"]["v1_required_arms"]]
    v2_required = [str(v) for v in G["numeric_divergence"]["v2_required_arms"]]
    inconclusive = str(G["numeric_divergence"]["inconclusive_label"])

    if div_families & set(v1_required):
        v1 = inconclusive
    elif any(states[f] == "missing" for f in v1_required):
        v1 = "NOT_RUN"
    else:
        v1 = _v1_label(G, states["BL"], states["FL"])

    # 判定の順序: BL の状態が決まらなければ V2 は問えない。BL が zero なら折衷は
    # 不要なので、段 2 を回していなくても NOT_APPLICABLE が確定する（spec §5.1）。
    if "BL" in div_families:
        v2, v2_hits = inconclusive, []
    elif bl_state == "missing":
        v2, v2_hits = "STAGE1_MISSING", []
    elif bl_state == str(G["v2_not_applicable_when_bl"]):
        v2, v2_hits = "NOT_APPLICABLE", []
    elif div_families & set(v2_required):
        v2, v2_hits = inconclusive, []
    elif any(states[f] == "missing" for f in v2_required):
        v2, v2_hits = "NOT_RUN", []
    else:
        v2 = _v2_label(G, bl_state, states["BLW"], states["RW"])
        v2_hits = _v2_co_satisfied(states["BLW"], states["RW"])

    # --- 水準 (§5.2) ---
    def log_u(arm: str, window: str) -> np.ndarray | None:
        if arm in controls:
            return controls[arm][f"log_u_{window.lower()}"]
        if arm in windows:
            return windows[arm][window]["log_u"]
        return None

    contrasts: dict[str, dict] = {}

    def add_contrast(kind: str, high: str, low: str) -> None:
        label = f"{high}_minus_{low}"
        hi_v, lo_v = log_u(high, "5M"), log_u(low, "5M")
        if hi_v is None or lo_v is None:
            contrasts[label] = dict(kind=kind, high=high, low=low,
                                    status="NOT_RUN" if not (high in divergences
                                                             or low in divergences)
                                    else NUMERIC_DIVERGENCE)
            return
        values = np.asarray(hi_v) - np.asarray(lo_v)
        row = dict(kind=kind, high=high, low=low, status="OK", n_paired=len(values),
                   seed_values=values.tolist(), ci=_ci(cfg, values, draws),
                   sign_test=_sign_test(values),
                   cross_run=bool(low in controls or high in controls))
        if kind == "P5":
            row["label"], row["ci_below_zero"] = _p5_label(G, row["ci"])
            row["equivalence_margin"] = float(G["p5_equivalence_margin"])
            row["margin_recalibrated"] = False
        contrasts[label] = row

    for high, low in G["p3prime_contrasts"]:
        add_contrast("P3prime", high, low)
    for high, low in G["p3prime_delta_contrasts"]:
        add_contrast("P3prime_delta", high, low)
    for high, low in G["p5_contrasts"]:
        add_contrast("P5", high, low)

    p6: dict[str, dict] = {}
    for entry in G["p6_contrasts"]:
        a_hi, a_lo = entry["a"]
        b_hi, b_lo = entry["b"]
        parts = [log_u(x, "5M") for x in (a_hi, a_lo, b_hi, b_lo)]
        if any(x is None for x in parts):
            p6[str(entry["dose"])] = dict(status="NOT_RUN",
                                          formula=f"({a_hi}-{a_lo})-({b_hi}-{b_lo})")
            continue
        values = (parts[0] - parts[1]) - (parts[2] - parts[3])
        p6[str(entry["dose"])] = dict(
            status="OK", formula=f"({a_hi}-{a_lo})-({b_hi}-{b_lo})",
            seed_values=values.tolist(), ci=_ci(cfg, values, draws),
            label_emitted=False,
            note="REPORT_ONLY. No threshold is registered and the R baseline sits "
                 "near the ceiling, so this is ceiling-contaminated by construction. "
                 "Orientation is the standard interaction, the opposite of "
                 "channel_2x2_0901; none of its label vocabulary transfers.")

    # 天井フラグ（追補 8）。P6 は `RW`-`R` と `BLW`-`BL` の差なので、**対照を含む**
    # 腕間のベースライン差が効く。対照には early 窓が無い（committed の verdict.csv は
    # 1M と 5M しか持たない）ので、新規腕だけの early 窓と、対照を含む 1M 窓の
    # 両方を測り、大きい方で旗を立てる。
    early_levels = {a: float(np.median(windows[a]["early"]["log_u"])) for a in complete}
    levels_1m = {a: float(np.median(windows[a]["1M"]["log_u"])) for a in complete}
    for a in CONTROL_ORDER:
        levels_1m[a] = float(np.median(controls[a]["log_u_1m"]))

    def _spread(values: dict) -> float:
        finite = [v for v in values.values() if np.isfinite(v)]
        return float(max(finite) - min(finite)) if len(finite) > 1 else float("nan")

    spread_early = _spread(early_levels)
    spread_1m = _spread(levels_1m)
    worst = float(np.nanmax([spread_early, spread_1m]))
    threshold = float(G["ceiling_baseline_spread_flag_dex"])
    ceiling = dict(early_window_spread_dex=spread_early,
                   window_1m_spread_dex_including_controls=spread_1m,
                   spread_dex=worst, threshold=threshold,
                   flagged=bool(np.isfinite(worst) and worst > threshold),
                   early_median_log10_U=early_levels,
                   median_log10_U_1m=levels_1m)

    # --- §5.3 REPORT_ONLY ---
    ref_logs = Path(ROOT) / cfg["controls"]["reference_run"] / "logs"
    revival_rows: list[dict] = []
    for arm in complete + list(G["revival"]["controls"]):
        source = outdir / "logs" if arm in complete else ref_logs
        for seed in seeds:
            path = source / f"{arm}_seed{seed}.npz"
            if not path.exists():
                continue
            counts = _revival_counts(path)
            within_opp = counts["opportunities_within_task"]
            revival_rows.append(dict(
                arm=arm, seed=seed,
                is_control=int(arm in CONTROL_ORDER), source=str(source),
                **counts,
                rate_within_task=(counts["events_within_task"] / within_opp
                                  if within_opp else float("nan"))))

    freeze_fields = ("status", "n_units", "window_records", "v_frozen_frac",
                     "b_frozen_frac", "strict_dead_all_window_frac",
                     "v_frozen_matches_dead", "b_frozen_matches_dead",
                     "v_frozen_xor_dead", "b_frozen_xor_dead")
    freeze_rows = []
    for arm in complete:
        for seed in seeds:
            got_rates = _freeze_rates(_unit_log(outdir, arm, seed), cfg)
            # write_csv は先頭行でスキーマを固定する。腕ごとに列が欠けると
            # 黙って落ちるので、ここで全列そろえる。
            freeze_rows.append(dict(arm=arm, seed=seed,
                                    **{k: got_rates.get(k, "") for k in freeze_fields}))
    fl_feature = {}
    for arm in complete:
        if arm.startswith("FL"):
            fl_feature[arm] = [
                _submerged_abs_v(_unit_log(outdir, arm, s), cfg) for s in seeds]

    # --- §5.4 P7 ---
    p7_arms = [a for a in G["s_distribution"]["p7a_arms"]
               if a in complete or a in CONTROL_ORDER]
    p7 = {}
    for arm in p7_arms:
        log_dir = outdir / "logs" if arm in complete else ref_logs
        if not all((log_dir / f"{arm}_seed{s}.npz").exists() for s in seeds):
            continue
        p7[arm] = _p7_arm(cfg, arm, log_dir)

    p7_contrasts = {}
    for kind, pairs in (("P7b", G["s_distribution"]["p7b_contrasts"]),
                        ("P7c", G["s_distribution"]["p7c_contrasts"]),
                        ("P7d", G["s_distribution"]["p7d_contrasts"])):
        for high, low in pairs:
            if high not in p7 or low not in p7:
                continue
            for channel in list(G["s_distribution"]["channels"]) + ["median_s"]:
                for unit_set in G["s_distribution"]["unit_sets"]:
                    key = f"{kind}:{high}_minus_{low}:{channel}:{unit_set}"
                    a = p7[high][f"{channel}__{unit_set}"]
                    b = p7[low][f"{channel}__{unit_set}"]
                    values = a - b
                    if not np.isfinite(values).all():
                        p7_contrasts[key] = dict(kind=kind, status="INSUFFICIENT_DATA",
                                                 high=high, low=low, channel=channel,
                                                 unit_set=unit_set)
                        continue
                    p7_contrasts[key] = dict(
                        kind=kind, status="OK", high=high, low=low, channel=channel,
                        unit_set=unit_set, seed_values=values.tolist(),
                        ci=_ci(cfg, values, draws))

    result = dict(
        stage=stage, arms_run=list(arms), complete=complete,
        divergences=sorted(divergences), V1=v1, V2=v2,
        V2_co_satisfied=v2_hits, onset_states=states, family_onsets=family_onsets,
        BL_onset_state=bl_state, BL_state_source=bl_source,
        onset=onset, controls={a: dict(n_onset_5m=controls[a]["n_onset_5m"],
                                       n_onset_1m=controls[a]["n_onset_1m"])
                               for a in CONTROL_ORDER},
        contrasts=contrasts, p6=p6, ceiling=ceiling, p7_contrasts=p7_contrasts,
        fl_submerged_abs_v=fl_feature, elapsed_sec=elapsed)

    _write_outputs(cfg, outdir, arms, complete, divergences, windows, controls,
                   onset, result, revival_rows, freeze_rows, p7, sanity)
    return result


def _write_outputs(cfg, outdir, arms, complete, divergences, windows, controls,
                   onset, result, revival_rows, freeze_rows, p7, sanity) -> None:
    seeds = [int(v) for v in cfg["common"]["seeds"]]
    G = _P(cfg)
    verdict_rows = []
    for arm in arms:
        arm_cfg = _arm(cfg, arm)
        base = dict(arm=arm, stage=int(arm_cfg["stage"]),
                    activation=str(arm_cfg["activation"]),
                    wd_b=float(arm_cfg["wd_b"]),
                    target_dose=float(arm_cfg["target_dose"]),
                    is_control=0, V1=result["V1"], V2=result["V2"],
                    V2_co_satisfied="|".join(result["V2_co_satisfied"]),
                    BL_onset_state=result["BL_onset_state"],
                    onset_state=result["onset_states"].get(arm.split("_")[0], ""))
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
                median_strict_dead_frac_5m=float(
                    np.median(w["5M"]["metrics"]["layer1_strict_dead"] / 100.0)),
                median_submerged_frac_5m=float(
                    np.median(w["5M"]["metrics"]["layer1_submerged"] / 100.0)),
                median_w_norm_5m=float(np.median(w["5M"]["metrics"]["layer1_w_norm_median"])),
                median_eval_loss_exact_5m=float(
                    np.median(w["5M"]["metrics"]["eval_loss_exact"])))
        else:
            base.update(status=NUMERIC_DIVERGENCE, NUMERIC_DIVERGENCE=1,
                        n_onset_1m="", cp95_1m_lo="", cp95_1m_hi="",
                        U_1m_seed_values="", median_log10_U_1m="",
                        n_onset_5m="", cp95_5m_lo="", cp95_5m_hi="",
                        U_5m_seed_values="", median_log10_U_5m="",
                        median_strict_dead_frac_5m="", median_submerged_frac_5m="",
                        median_w_norm_5m="", median_eval_loss_exact_5m="")
        verdict_rows.append(base)
    # 対照は同じ表に別走の値として載せる（引用時の注記は summary.md にも出す）
    for arm in CONTROL_ORDER:
        c = controls[arm]
        verdict_rows.append(dict(
            arm=arm, stage=0, activation=("relu" if arm.startswith("R_") else "leaky_relu"),
            wd_b=0.0, target_dose=float(arm.split("_")[1]) / 100.0, is_control=1,
            V1=result["V1"], V2=result["V2"],
            V2_co_satisfied="|".join(result["V2_co_satisfied"]),
            BL_onset_state=result["BL_onset_state"],
            onset_state=result["onset_states"]["R" if arm.startswith("R_") else "LR"],
            status="COMMITTED_OTHER_RUN", NUMERIC_DIVERGENCE=0,
            n_onset_1m=c["n_onset_1m"], cp95_1m_lo="", cp95_1m_hi="",
            U_1m_seed_values=json.dumps(c["u_1m"].tolist()),
            median_log10_U_1m=float(np.median(c["log_u_1m"])),
            n_onset_5m=c["n_onset_5m"], cp95_5m_lo="", cp95_5m_hi="",
            U_5m_seed_values=json.dumps(c["u_5m"].tolist()),
            median_log10_U_5m=float(np.median(c["log_u_5m"])),
            median_strict_dead_frac_5m="", median_submerged_frac_5m="",
            median_w_norm_5m="", median_eval_loss_exact_5m=""))
    write_csv(outdir / "verdict.csv", verdict_rows)

    contrast_rows = []
    for label, value in result["contrasts"].items():
        row = dict(endpoint=value["kind"], contrast=label, high=value["high"],
                   low=value["low"], status=value["status"],
                   cross_run=int(value.get("cross_run", 0)),
                   n_paired=value.get("n_paired", ""),
                   label=value.get("label", ""),
                   ci_below_zero=int(value["ci_below_zero"])
                   if "ci_below_zero" in value else "",
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
    for dose, value in result["p6"].items():
        row = dict(endpoint="P6", contrast=f"dose{dose}", high="", low="",
                   status=value["status"], cross_run=1, n_paired="",
                   label="", ci_below_zero="", equivalence_margin="")
        ci = value.get("ci")
        for key in ("point", "percentile_ci_lo", "percentile_ci_hi",
                    "studentized_ci_lo", "studentized_ci_hi", "ci_degenerate"):
            row[key] = "" if ci is None else ci[key]
        row.update(sign_n_positive="", sign_n_negative="", sign_p_two_sided="",
                   seed_values=json.dumps(value.get("seed_values", [])))
        contrast_rows.append(row)
    write_csv(outdir / "layer_stats.csv", contrast_rows)

    if revival_rows:
        write_csv(outdir / "revival.csv", revival_rows)
    if freeze_rows:
        write_csv(outdir / "freeze_rates.csv", freeze_rows)

    s_rows = []
    for arm, values in p7.items():
        for i, seed in enumerate(seeds):
            row = dict(arm=arm, seed=seed,
                       is_control=int(arm in CONTROL_ORDER),
                       source=("this_run" if arm not in CONTROL_ORDER
                               else str(Path(cfg["controls"]["reference_run"]) / "logs")))
            for key, arr in values.items():
                if key == "_meta":
                    continue
                row[key] = float(arr[i])
            meta = values["_meta"][i]
            row.update(n_records=meta["n_records"], n_na_M=meta["n_na_M"],
                       n_na_B=meta["n_na_B"],
                       alive_units_median=meta["alive_units_median"],
                       alive_records_used=meta["alive_records_used"])
            s_rows.append(row)
    if s_rows:
        fields = list(dict.fromkeys(k for row in s_rows for k in row))
        write_csv(outdir / "s_distribution.csv",
                  [{k: row.get(k, "") for k in fields} for row in s_rows])
    p7_rows = []
    for key, value in result["p7_contrasts"].items():
        ci = value.get("ci")
        row = dict(key=key, kind=value["kind"], status=value["status"],
                   high=value["high"], low=value["low"], channel=value["channel"],
                   unit_set=value["unit_set"])
        for name in ("point", "percentile_ci_lo", "percentile_ci_hi",
                     "studentized_ci_lo", "studentized_ci_hi", "ci_degenerate"):
            row[name] = "" if ci is None else ci[name]
        p7_rows.append(row)
    if p7_rows:
        write_csv(outdir / "s_contrasts.csv", p7_rows)

    _write_summary(cfg, outdir, result, verdict_rows, sanity)


def _write_summary(cfg: dict, outdir: Path, result: dict, verdict_rows: list[dict],
                   sanity: dict) -> None:
    G = _P(cfg)
    lines = [f"# {EXPERIMENT} summary (stage {result['stage']})", "",
             "## Verdict", "",
             f"- **V1 (担い手): {result['V1']}**  — BL={result['onset_states']['BL']}, "
             f"FL={result['onset_states']['FL']}",
             f"- **V2 (折衷): {result['V2']}**  — BLW={result['onset_states']['BLW']}, "
             f"RW={result['onset_states']['RW']} "
             f"(BL={result['BL_onset_state']}, source: {result['BL_state_source']})",
             f"- V2 rows whose condition also held: "
             f"{', '.join(result['V2_co_satisfied']) or '—'}",
             f"- Raw onset triples (5M, n_onset per dose): "
             + "; ".join(f"{f}={v}" for f, v in result["family_onsets"].items()),
             f"- Numeric divergence: {', '.join(result['divergences']) or 'none'}",
             "",
             "### 引用上の注意（spec §8）", "",
             "- 0/10 は「5M までに観測しなかった」（片側 95% 上限 p<=0.2589）。",
             "- **対照 `R_*` / `LR_*` は別走 `gate_dose_0830` の committed 値であり、",
             "  同一走の腕ではない。** ペアリングは init・教師・入力実現までで、",
             "  軌道は step 1 以降で分岐する。",
             "- 復活数は **within-task**（同一タスク内かつ flip 不変）の定義でのみ",
             "  「ReLU 腕は 0」。境界越えは支持の引き直しによる見かけの復活を含む。",
             "- P5 の等価限界 0.15 dex は channel_2x2_0901 D4 からの継承で、",
             "  **この系（1 層・std・床 1e-16・log10(mean U)）で較正し直していない**。",
             "- P7c の 0.15 は s の単位であって dex ではない。P5 の 0.15 とは別の数。",
             "- P6 はラベルを付けない。閾値が登録されておらず、R 系のベースラインが",
             "  天井付近なので構成上すでに天井汚染される。",
             "", "## Endpoints (5M)", "",
             "| arm | act | wd_b | dose | onset 1M | onset 5M | median log10 U 1M | median log10 U 5M | source |",
             "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for row in verdict_rows:
        if row["status"] == NUMERIC_DIVERGENCE:
            lines.append(f"| {row['arm']} | {row['activation']} | {row['wd_b']} | "
                         f"{row['target_dose']} | — | — | — | — | {row['status']} |")
            continue
        source = ("gate_dose_0830 (別走)" if row["is_control"] else "this run")
        lines.append(
            f"| {row['arm']} | {row['activation']} | {row['wd_b']} | "
            f"{row['target_dose']} | {row['n_onset_1m']}/10 | {row['n_onset_5m']}/10 | "
            f"{row['median_log10_U_1m']:.6g} | {row['median_log10_U_5m']:.6g} | {source} |")
    lines += ["", "## Paired level contrasts at 5M (§5.2)", "",
              "| endpoint | contrast | cross-run | n | median delta log10 U | percentile 95% CI | label | CI<0 | sign test p |",
              "|---|---|---:|---:|---:|---|---|---:|---:|"]
    for label, value in result["contrasts"].items():
        if value["status"] != "OK":
            lines.append(f"| {value['kind']} | {label} | — | — | — | {value['status']} | — | — | — |")
            continue
        ci, st = value["ci"], value["sign_test"]
        lines.append(
            f"| {value['kind']} | {label} | {int(value['cross_run'])} | "
            f"{value['n_paired']} | {ci['point']:.6g} | "
            f"[{ci['percentile_ci_lo']:.6g}, {ci['percentile_ci_hi']:.6g}] | "
            f"{value.get('label', '—')} | {int(value.get('ci_below_zero', 0))} | "
            f"{st['p_two_sided']:.4g} |")
    lines += ["", "## P6 (REPORT_ONLY, no label)", ""]
    for dose, value in result["p6"].items():
        if value["status"] != "OK":
            lines.append(f"- dose {dose}: {value['status']}")
            continue
        ci = value["ci"]
        lines.append(f"- dose {dose}: {value['formula']} = {ci['point']:.6g}, "
                     f"95% CI [{ci['percentile_ci_lo']:.6g}, {ci['percentile_ci_hi']:.6g}]")
    ceiling = result["ceiling"]
    lines += ["", f"Between-arm baseline spread: {ceiling['spread_dex']:.6g} dex "
              f"(early window, new arms: {ceiling['early_window_spread_dex']:.6g}; "
              f"1M window incl. committed controls: "
              f"{ceiling['window_1m_spread_dex_including_controls']:.6g}; "
              f"flag threshold {ceiling['threshold']}). "
              f"**{'P6 must not be read alone.' if ceiling['flagged'] else 'Not flagged.'}**",
              "", "## Sanity", ""]
    for key in ("S1_omp", "S_cross", "S_bwd", "S_wd", "S_log", "S_ref",
                "S_limit_bwd", "S_limit_fwd", "S_limit_wd", "S_log_b",
                "S_pair", "S_dose", "S_taut", "S6_floor_inherited",
                "S_CI_degeneracy"):
        value = sanity.get(key, {})
        lines.append(f"- {key}: **{'PASS' if value.get('pass_') else 'FAIL'}**")
    (outdir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Run driver
# ---------------------------------------------------------------------------
def _provenance(cfg_path: Path, cfg: dict, outdir: Path, stage: str, doses: str,
                arms: list[str], sanity: dict, analysis: dict, elapsed: dict,
                started: float) -> dict:
    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                           text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--short"], cwd=ROOT,
                                        text=True).splitlines()
    except (OSError, subprocess.CalledProcessError):
        git_hash, dirty = None, []
    names = ("verdict.csv", "summary.md", "layer_stats.csv", "s_distribution.csv",
             "s_contrasts.csv", "revival.csv", "freeze_rates.csv", "config_used.yaml")
    hashes = {n: _sha_file(outdir / n) for n in names if (outdir / n).exists()}
    hashes.update({f"logs/{p.name}": _sha_file(p)
                   for p in sorted((outdir / "logs").glob("*.npz"))})
    hashes.update({f"arm_status/{p.name}": _sha_file(p)
                   for p in sorted((outdir / "arm_status").glob("*.json"))})
    ref_dir = (Path(ROOT) / cfg["controls"]["reference_run"]).resolve()
    parent_prov = ref_dir / "provenance.json"
    parent = json.loads(parent_prov.read_text(encoding="utf-8")) if parent_prov.exists() else {}
    return dict(
        experiment=EXPERIMENT, created=time.strftime("%Y-%m-%d %H:%M:%S %z"),
        command=sys.argv, elapsed_sec=round(time.time() - started, 3),
        arm_elapsed_sec=elapsed, cwd=os.getcwd(), python=sys.version,
        platform=platform.platform(), torch=torch.__version__, numpy=np.__version__,
        device=cfg["common"]["device"], git_hash=git_hash, git_dirty=dirty,
        config=str(cfg_path), config_sha256=_sha_file(cfg_path),
        spec=str(Path(ROOT) / cfg["spec"]),
        spec_sha256=_sha_file(Path(ROOT) / cfg["spec"]),
        # spec §6: 回した段と用量・S-log の分岐と理由を記録する
        stage_run=stage, doses_run=doses, arms_run=list(arms),
        s_log_branch=str(sanity.get("S_log", {}).get("branch")),
        s_log_reason=str(sanity.get("S_log", {}).get("reason")),
        generator_offset=int(cfg["common"]["generator_offset"]),
        generator_offset_note=(
            "explicit 0: this run deliberately shares the parent run's seed set and "
            "random stream (S-pair). 9月運用_0901 §3-1's 'new seed groups need a "
            "generator_offset' does not apply."),
        baseline_reference=str(ref_dir),
        baseline_git_hash=parent.get("git_hash"),
        baseline_endpoint_source=str(ref_dir / "verdict.csv"),
        baseline_unit_source=str(ref_dir / "logs"),
        baseline_unit_source_is_gitignored=True,
        sanity=sanity, analysis=analysis, output_sha256=hashes)


def run(cfg_path: Path, cfg: dict, device: str, outdir: Path, stage: str,
        doses: str, *, smoke: bool, analyze_only: bool,
        stage1_dir: Path | None) -> dict:
    started = time.time()
    outdir.mkdir(parents=True, exist_ok=True)
    require_omp(cfg)
    arms = _selected_arms(cfg, stage, doses)
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
    if not analyze_only:
        for arm in arms:
            existing = _load_divergence_status(outdir, arm, seeds, total, every)
            if existing is not None and not smoke:
                divergences[arm] = existing
                elapsed[arm] = 0.0
                print(f"[{arm}] saved {NUMERIC_DIVERGENCE}; resume", flush=True)
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
    else:
        for arm in arms:
            existing = _load_divergence_status(outdir, arm, seeds, total, every)
            if existing is not None:
                divergences[arm] = existing

    if smoke:
        payload = dict(pass_=bool(all(v.get("pass_") for v in identities.values())),
                       identities=identities, divergences=divergences,
                       elapsed_sec=elapsed, arms=arms)
        (outdir / "smoke_sanity.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8")
        print(f"SMOKE DONE -> {outdir}", flush=True)
        return payload

    sanity = dict(preflight_result)
    sanity.pop("pass_", None)
    result = analyze(cfg, outdir, arms, stage, sanity, elapsed, divergences,
                     stage1_dir)
    provenance = _provenance(cfg_path, cfg, outdir, stage, doses, arms, sanity,
                             result, elapsed, started)
    (outdir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")
    print(f"V1={result['V1']}  V2={result['V2']}", flush=True)
    print(f"ALL DONE -> {outdir}", flush=True)
    return dict(sanity=sanity, analysis=result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--stage", default="1", choices=["1", "2", "all"])
    parser.add_argument("--doses", default="both", choices=["both", "933", "1216"])
    parser.add_argument("--outdir")
    args = parser.parse_args()
    if args.preflight and (args.smoke or args.analyze_only):
        parser.error("stage flags are mutually exclusive")
    cfg_path = Path(args.config).resolve()
    cfg = load_config(str(cfg_path))
    device = pick_device(cfg)
    if device != "cpu":
        raise ValueError("bwd_leak is CPU-only")
    stage = ("preflight" if args.preflight else "smoke" if args.smoke
             else "analyze" if args.analyze_only else "run")
    validate_config(cfg, stage=stage)
    if args.preflight:
        preflight(cfg, Path(ROOT) / f"results/_preflight_{EXPERIMENT}")
        return
    outdir = (Path(args.outdir).resolve() if args.outdir
              else Path(ROOT) / f"results/_smoke_{EXPERIMENT}" if args.smoke
              else _stage_outdir(cfg, args.stage))
    stage1_dir = _stage_outdir(cfg, "1") if args.stage == "2" else None
    run(cfg_path, cfg, device, outdir, args.stage, args.doses, smoke=args.smoke,
        analyze_only=args.analyze_only, stage1_dir=stage1_dir)


if __name__ == "__main__":
    main()
