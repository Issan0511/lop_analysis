"""phantom_wall_0902: 幻の壁 3 型 × w-WD。

事前登録: ``specs/spec_phantom_wall_0902.md``（この実装より**先に** ``89dfc10`` で
config と一緒に単独 commit されている）。Obsidian 側の正本は
``可塑性喪失/spec/幻の壁3型_spec_0902.md``。

親は ``bwd_leak_0902``（`BL` 4 腕が全発散）。宿主は ``gate_dose_0830`` で、学習・
記録・用量固定はそのまま import する。新規に足すのは

* ``VecMLPL`` の 3 活性化 ``bwd_reflect`` / ``bwd_quad`` / ``bwd_leaky_proj`` と
  ``set_weight_decay_w``（``src/nets.py``）
* `BLP` の µ 射影（**``act_grad`` では書けない**ので勾配を組み立てたあとに掛ける）
* 本モジュールの sanity・集計

対照 `R_1216` / `LR_1216`（``gate_dose_0830``）と `RW_1216`（``bwd_leak_0902``）は
再走しない。主 endpoint は各 ``verdict.csv`` から転記する。

Stages::

    OMP_NUM_THREADS=1 .venv/bin/python -m src.phantom_wall_0902 --preflight
    OMP_NUM_THREADS=1 .venv/bin/python -m src.phantom_wall_0902 --smoke
    OMP_NUM_THREADS=1 .venv/bin/python -m src.phantom_wall_0902
    OMP_NUM_THREADS=1 .venv/bin/python -m src.phantom_wall_0902 --analyze-only
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

from .bwd_leak_0902 import (BwdRecorder, NEW_UNIT_KEYS, SanityError, _ci,
                            _draws, _p7_seed_values, _revival_counts,
                            _sign_test, write_arm_logs_bwd)
from .common import ROOT, load_config, pick_device
from .dose_const_5m import (_input_stats, _refresh_fixed_offset, _target,
                            clopper_pearson, gamma_for_k, setup_arm_const)
from .elu_swamp import exact_layer_record_elu
from .gate_dose import SIGMA_TOL, IDENTITY_TOL, forward_gate, _window
from .mlp2_phase0 import (_sha_array, _sha_file, identity_sanity_pass,
                          require_omp, write_csv)
from .mlp2_phase0b import _ci_components, _complete_arm_logs, _window_indices
from .mlp2_phase1 import (NUMERIC_DIVERGENCE, NumericDivergenceError,
                          StreamDigest, _env_hashes, _init_hashes,
                          _seed_state_hashes_p1)
from .nets import VecMLPL


EXPERIMENT = "phantom_wall_0902"
CONFIG = Path(ROOT) / "configs" / "phantom_wall_0902.yaml"

ARM_ORDER = ("BLR_1216", "BLQ_1216", "BLP_1216",
             "BLRw_1216", "BLQw_1216", "BLPw_1216", "RWw_1216")
CONTROL_ORDER = ("R_1216", "LR_1216", "RW_1216")
PHANTOM_TYPES = ("BLR", "BLQ", "BLP")
SPRING_CONTROL = "RWw_1216"

REGISTERED_ARMS = {
    "BLR_1216":  ("bwd_reflect", 0.0),
    "BLQ_1216":  ("bwd_quad", 0.0),
    "BLP_1216":  ("bwd_leaky_proj", 0.0),
    "BLRw_1216": ("bwd_reflect", 1e-4),
    "BLQw_1216": ("bwd_quad", 1e-4),
    "BLPw_1216": ("bwd_leaky_proj", 1e-4),
    "RWw_1216":  ("relu", 1e-4),
}

SMOKE_STEPS = 30_000
ONSET_STATES = ("zero", "mid", "present", "diverged")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def _arm(cfg: dict, name: str) -> dict:
    return next(a for a in cfg["arms"] if a["name"] == name)


def _P(cfg: dict) -> dict:
    return cfg["phantom_wall"]


def _activation(cfg: dict, arm_cfg: dict) -> tuple[str, float]:
    label = str(arm_cfg["activation"])
    if label == "relu":
        return "relu", 1.0
    return str(cfg["activation"][label]["name"]), float(cfg["activation"][label]["slope"])


def _spec_table_label(x: str, xw: str, rww: str) -> str:
    """spec §5.1 の順序つき表を**表の側から**素直に実装する（追補 5 の検算用）。

    「0」は zero、「≥5」は present、「D」は diverged、「—」はワイルドカード。
    """
    if x == "diverged":
        return "PHANTOM_DIVERGES"                                  # 行 1
    if x == "zero" and rww == "present":
        return "PHANTOM_RESCUES"                                   # 行 2
    if x == "present" and xw == "zero" and rww == "present":
        return "PHANTOM_NEEDS_SPRING"                              # 行 3
    if rww == "zero":
        return "WD_W_SUFFICIENT_ALONE"                             # 行 4
    if x == "present" and xw == "present" and rww == "present":
        return "PHANTOM_DEAF"                                      # 行 5
    return "PARTIAL"                                               # 行 6


def _enumerate_decision_table() -> dict[str, str]:
    """4 値 × 3 腕 = 64 セルの全列挙（追補 5）。解析はこの写像を引く。"""
    return {f"{x}|{xw}|{r}": _spec_table_label(x, xw, r)
            for x in ONSET_STATES for xw in ONSET_STATES for r in ONSET_STATES}


DECISION_TABLE = _enumerate_decision_table()


def _co_satisfied(x: str, xw: str, rww: str) -> list[str]:
    """当たった行以外に**条件を満たしていた行**も残す（追補 5）。"""
    hits = []
    if x == "diverged":
        hits.append("PHANTOM_DIVERGES")
    if x == "zero" and rww == "present":
        hits.append("PHANTOM_RESCUES")
    if x == "present" and xw == "zero" and rww == "present":
        hits.append("PHANTOM_NEEDS_SPRING")
    if rww == "zero":
        hits.append("WD_W_SUFFICIENT_ALONE")
    if x == "present" and xw == "present" and rww == "present":
        hits.append("PHANTOM_DEAF")
    return hits or ["PARTIAL"]


def validate_config(cfg: dict, *, stage: str) -> None:
    if stage not in {"preflight", "smoke", "run", "analyze"}:
        raise ValueError(f"unknown stage {stage!r}")
    C, A, I, P, G, S = (cfg["common"], cfg["condA"], cfg["intervention"],
                        cfg["phase1"], cfg["phantom_wall"], cfg["sanity"])
    if [a["name"] for a in cfg["arms"]] != list(ARM_ORDER):
        raise ValueError(f"arms must be ordered as {ARM_ORDER}")
    for arm in cfg["arms"]:
        want_act, want_wd = REGISTERED_ARMS[arm["name"]]
        if (str(arm["activation"]) != want_act or float(arm["wd_w"]) != want_wd
                or [int(v) for v in arm["hidden"]] != [100]
                or [int(v) for v in arm.get("centered_layers", [])] != [1]
                or _target(arm) != 3.041 or float(arm["target_dose"]) != 12.16):
            raise ValueError(f"{arm['name']} differs from the preregistration")
    if (int(A["m"]), int(A["f"]), int(A["target_hidden"])) != (20, 15, 100):
        raise ValueError("phantom_wall requires condA m=20, f=15, teacher width=100")
    if list(A["T_values"]) != [10_000] or list(A["encodings"]) != ["std"]:
        raise ValueError("phantom_wall requires T=10000 and std encoding")
    if int(C.get("generator_offset", -1)) != 0:
        raise ValueError("generator_offset must be an explicit 0")
    if (str(I["name"]) != "oracle_fixed_mu_offset" or I["oracle"] is not True
            or I["consumes_rng"] is not False
            or I["mu_is_function_of_flip_state_only"] is not True):
        raise ValueError("the oracle-dose intervention changed")
    act = cfg["activation"]
    if (float(act["bwd_reflect"]["slope"]) != 0.1
            or str(act["bwd_reflect"]["derivative"]) != "where(z>0, 1, -slope)"
            or float(act["bwd_quad"]["slope"]) != 0.01
            or act["bwd_quad"]["slope_is_not_a_slope"] is not True
            or float(act["bwd_quad"]["crossover_depth_z"]) != -10.0
            or float(act["bwd_leaky_proj"]["slope"]) != 0.1
            or act["bwd_leaky_proj"]["post_gradient_projection"] is not True
            or str(act["bwd_leaky_proj"]["projection_applies_to"]) != "submerged_units_only"
            or str(act["bwd_leaky_proj"]["b_rule"]) != "zero"
            or str(act["bwd_leaky_proj"]["v_rule"]) != "untouched"
            or act["autograd"] is not False or act["consumes_rng"] is not False):
        raise ValueError("phantom activation definitions changed")
    if (float(cfg["weight_decay_w"]["lambda_w"]) != 1e-4
            or str(cfg["weight_decay_w"]["applies_to"]) != "hidden_W_only"
            or cfg["weight_decay_w"]["decoupled"] is not False
            or cfg["weight_decay_w"]["calibrated_for_this_system"] is not False):
        raise ValueError("the w weight-decay definition changed")
    expected_phase = {
        "task_period": 10_000, "early_tasks": [2, 11],
        "late_tasks_5m": [491, 500], "window_1m_tasks": [91, 100],
        "window_points_are_task_ends_only": True,
        "onset_threshold": 0.05, "onset_present_min": 5,
        "unfit_floor": 1e-16, "recalibrate_floor": False,
        "bootstrap_B": 10_000, "bootstrap_seed": 20_260_906,
        "ci_method": "percentile_primary_studentized_secondary",
    }
    for key, value in expected_phase.items():
        if P[key] != value:
            raise ValueError(f"phase1.{key} differs from the preregistration")
    if list(G["onset_states"]) != list(ONSET_STATES):
        raise ValueError("onset state definition changed")
    if int(G["onset_zero_max"]) != 0 or int(G["onset_present_min"]) != 5:
        raise ValueError("onset thresholds changed")
    rows = G["decision_table_rows"]
    if len(rows) != 6 or [r["verdict"] for r in rows] != [
            "PHANTOM_DIVERGES", "PHANTOM_RESCUES", "PHANTOM_NEEDS_SPRING",
            "WD_W_SUFFICIENT_ALONE", "PHANTOM_DEAF", "PARTIAL"]:
        raise ValueError("the §5.1 decision table changed")
    if _decision_table_disagrees_with_config(G):
        raise ValueError("the enumerated decision table disagrees with the config rows")
    if dict(G["control_expected_onset_5m"]) != {"R_1216": 10, "LR_1216": 0,
                                                "RW_1216": 10}:
        raise ValueError("control expectations changed")
    if float(G["p5_equivalence_margin"]) != 0.15:
        raise ValueError("the P5 equivalence margin changed")
    if (G["p5_margin_recalibrated_for_this_system"] is not False
            or G["p5_emit_ci_below_zero_flag"] is not True
            or G["p5_sign_test_report_only"] is not True
            or G["p8_emit_label"] is not False or G["p8_in_verdict"] is not False
            or G["p8_cross_type_difference_registered"] is not False
            or cfg["blr_blq_contrast_registered"] is not False):
        raise ValueError("contrast registration changed")
    if (str(G["saturation_guard"]["trigger_when_spring_control_not"]) != "present"
            or str(G["saturation_guard"]["action"]) != "promote_levels_to_coequal"):
        raise ValueError("the saturation guard changed")
    inv = G["blp_mu_invariance"]
    if (str(inv["primary_quantity"]) != "zbar" or float(inv["primary_tolerance"]) != 1e-6
            or "delta_s" not in list(inv["also_report"])
            or "denom_growth_ratio" not in list(inv["also_report"])
            or inv["exclude_records_where_unit_fired"] is not True):
        raise ValueError("the BLP mu-invariance registration changed (追補 8)")
    if (float(S["s_proj_tol"]) != 1e-12
            or str(S["s_proj_gate_precision"]) != "float64"
            or S["s_proj_report_float32"] is not True
            or S["s_limit_check_signbit"] is not True
            or float(S["s_bwd_closed_form_tol"]) != 0.0
            or int(S["s_pair_steps"]) != 30_000 or int(S["s_limit_steps"]) != 30_000
            or float(S["s_dose_rel_tol"]) != 1e-10
            or S["s_taut_check"] is not True or S["s_ref_hash_check"] is not True
            or int(S["omp_num_threads"]) != 1
            or S["s6_floor_calibration"] is not False):
        raise ValueError("sanity gates changed")
    if (S.get("s_refl_report_only") is not True
            or S.get("s_refl_is_required_gate") is not False):
        raise ValueError("S-refl の REPORT_ONLY 降格が config に登録されていない（追補 13）")
    if str(_P(cfg)["s_distribution"]["log_branch"]) != "A":
        raise ValueError("the S-log branch changed")
    if stage in {"run", "analyze"}:
        if int(C["total_steps"]) != 5_000_000 or list(C["seeds"]) != list(range(10)):
            raise ValueError("the full run requires 5M steps and seeds 0..9")
        if str(C["device"]) != "cpu":
            raise ValueError("phantom_wall is CPU-only")


def _decision_table_disagrees_with_config(G: dict) -> bool:
    """config の行定義から順序で解いた結果と、全列挙が一致することを検算する。"""
    def matches(spec_value: str, actual: str) -> bool:
        return spec_value == "any" or spec_value == actual

    for x in ONSET_STATES:
        for xw in ONSET_STATES:
            for r in ONSET_STATES:
                got = None
                for row in G["decision_table_rows"]:
                    if (matches(str(row["X"]), x) and matches(str(row["Xw"]), xw)
                            and matches(str(row["RWw"]), r)):
                        got = str(row["verdict"])
                        break
                if got != DECISION_TABLE[f"{x}|{xw}|{r}"]:
                    return True
    return False


# ---------------------------------------------------------------------------
# Learning path — BLP の µ 射影だけが gate_dose の経路から外れる
# ---------------------------------------------------------------------------
def setup_arm_phantom(cfg: dict, arm_cfg: dict, device: str) -> dict:
    st = setup_arm_const(cfg, arm_cfg, device)
    act, alpha = _activation(cfg, arm_cfg)
    st["net"].set_activation(act, alpha, "alpha_exp")
    st["net"].set_weight_decay_w(float(arm_cfg["wd_w"]))
    st["activation"] = act
    st["act_alpha"] = float(alpha)
    st["wd_w"] = float(arm_cfg["wd_w"])
    st["wd_b"] = 0.0
    st["mu_projection"] = act == "bwd_leaky_proj"
    return st


def mu_hat(st: dict) -> torch.Tensor:
    """オラクル用量で実際に加えた µ の単位ベクトル [R, m]（float64）。

    µ は ``env.flip_state`` だけの関数なので厳密支持の列挙は要らない（追補 12）:
    ``mu = [flip, 0.5 * 1_free] - 0.5 * gamma``、``gamma = gamma_for_k(k, target)``。
    前向きが引くオフセットが ``0.5 * gamma`` なので、``E[cur_in] = mu`` である。
    """
    env = st["env"]
    free = env.m - env.f
    tail = torch.full((st["R"], free), 0.5, dtype=torch.float64,
                      device=env.flip_state.device)
    raw = torch.cat([env.flip_state.double(), tail], dim=1)
    k = env.flip_state.double().sum(dim=1)
    gamma = gamma_for_k(k, float(st["target_mu_norm"]))
    mu = raw - 0.5 * gamma[:, None]
    return mu / mu.norm(dim=1, keepdim=True).clamp_min(1e-300)


def grads_phantom(st: dict, inputs, pres, acts, delta):
    """``grads_centered_elu`` に `BLP` の µ 射影を足したもの（1 層専用）。

    順序は spec §4.3 のとおり: (1) 代替勾配 ``where(pre>0, 1, a)`` で ``gW``・``gb``
    を作り、(2) **沈下ユニット由来の寄与だけ**に $w$ の µ 直交射影と $b$ のゼロ化を
    掛ける。$g_W = g_b \\otimes x_{\\rm in}$ なので、$g_W$ を射影することは
    $x_{\\rm in}$ を射影することと厳密に同値（追補 12）。射影は float64 で行って
    から学習器の dtype へ落とす。$v$ には触らない。
    """
    net = st["net"]
    d2 = 2.0 * delta
    gv = d2[:, None] * acts[-1]
    gc = d2
    dz = d2[:, None] * net.v * net.act_grad(pres[-1], acts[-1])
    x_in = inputs[0]
    if not st.get("mu_projection"):
        return [dz[:, :, None] * x_in[:, None, :]], [dz], gv, gc
    # 沈下ユニット（このステップで z_i <= 0）だけを加工する。batch=1 なので
    # 「沈下パターン由来の寄与」はこれと等価（spec §4.3）。
    submerged = pres[-1] <= 0                                  # [R, h]
    mh = mu_hat(st)                                            # [R, m] float64
    x64 = x_in.double()
    x_proj = (x64 - (x64 * mh).sum(dim=1, keepdim=True) * mh).to(x_in.dtype)
    gW_plain = dz[:, :, None] * x_in[:, None, :]
    gW_proj = dz[:, :, None] * x_proj[:, None, :]
    gW = torch.where(submerged[:, :, None], gW_proj, gW_plain)
    gb = torch.where(submerged, torch.zeros_like(dz), dz)
    return [gW], [gb], gv, gc


def train_arm_phantom(st: dict, recorder, probe_steps, total: int, outdir: Path,
                      checkpoints, stream_hook=None) -> float:
    """``gate_dose.train_arm_gate`` と同じループで、勾配だけ ``grads_phantom``。"""
    probe_set = {int(v) for v in probe_steps}
    checkpoint_set = {int(v) for v in checkpoints}
    net, env, teacher = st["net"], st["env"], st["teacher"]
    started = time.time()
    for step in range(total):
        if step in checkpoint_set:
            _save_checkpoint(st, st["arm"], step, outdir)
        if step in probe_set:
            recorder(st, step)
        x = env.step()
        y = teacher(x)
        if stream_hook is not None:
            stream_hook(step, x, y)
        inputs, pres, acts, yhat = forward_gate(st, x)
        grads = grads_phantom(st, inputs, pres, acts, yhat - y)
        net.sgd_step_layers(st["lr"], *grads)
    if total in probe_set:
        recorder(st, total)
    if total in checkpoint_set:
        _save_checkpoint(st, st["arm"], total, outdir)
    return time.time() - started


def _save_checkpoint(st: dict, arm: str, step: int, outdir: Path) -> Path:
    path = outdir / "ckpts" / f"{arm}_step{step}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(step=step, arm=arm, net=st["net"].state_dict(),
                    env=st["env"].state_dict(), teacher=st["teacher"].state_dict(),
                    running_mean=st["running_mean"].clone(),
                    layer_means=[None if m is None else m.clone()
                                 for m in st["layer_means"]],
                    centered_layers=list(st["centered_layers"]),
                    target_mu_norm=st.get("target_mu_norm"),
                    target_dose=st.get("target_dose"),
                    activation=st["activation"], act_alpha=st["act_alpha"],
                    wd_w=st["wd_w"], runs=st["runs"]), path)
    return path


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
    st = setup_arm_phantom(c, _arm(c, arm), device)
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
          f"wd_w={st['wd_w']:g} proj={st['mu_projection']} seeds={seeds} "
          f"steps={total:,}", flush=True)
    started = time.time()
    try:
        elapsed = train_arm_phantom(st, rec, probes, total, outdir, checkpoints)
    except NumericDivergenceError as exc:
        elapsed = time.time() - started
        event = dict(exc.event)
        event.update(probe_every=every, registered_total_steps=int(total),
                     registered_seeds=[int(v) for v in seeds],
                     activation=st["activation"], act_alpha=st["act_alpha"],
                     wd_w=st["wd_w"], elapsed_sec=float(elapsed),
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
    st["wd_b"] = 0.0
    write_arm_logs_phantom(outdir, arm, st, rec)
    print(f"[{arm}] complete in {elapsed:.1f}s", flush=True)
    return dict(status="COMPLETE", elapsed_sec=elapsed, sanity=sanity)


def write_arm_logs_phantom(outdir: Path, arm: str, st: dict, rec) -> list[Path]:
    """``write_arm_logs_bwd`` の列に ``wd_w`` と ``mu_projection`` を足す。"""
    paths = write_arm_logs_bwd(outdir, arm, st, rec)
    for path in paths:
        with np.load(path, allow_pickle=False) as z:
            payload = {k: z[k].copy() for k in z.files}
        payload["wd_w"] = np.float64(st["wd_w"])
        payload["mu_projection"] = np.int64(bool(st["mu_projection"]))
        np.savez_compressed(path, **payload)
    return paths


# ---------------------------------------------------------------------------
# Preregistered sanity gates (spec §6)
# ---------------------------------------------------------------------------
def _grid(cfg: dict) -> torch.Tensor:
    lo, hi, n = cfg["sanity"]["s_cross_grid"]
    return torch.linspace(float(lo), float(hi), int(n), dtype=torch.float64)


def _probe(act: str, alpha: float) -> VecMLPL:
    return VecMLPL(1, [2], 2, torch.Generator().manual_seed(0), "cpu") \
        .set_activation(act, alpha, "alpha_exp")


def _s_cross(cfg: dict) -> dict:
    """S-cross: 幻 3 型の forward が `relu` と bit 一致。`BLP` の勾配は `bwd_leaky` と同一。"""
    grid = _grid(cfg)
    relu = _probe("relu", 1.0)
    a_relu = relu.act_fn(grid)
    bl = _probe("bwd_leaky", float(cfg["activation"]["bwd_leaky_proj"]["slope"]))
    rows = []
    for name in ("bwd_reflect", "bwd_quad", "bwd_leaky_proj"):
        alpha = float(cfg["activation"][name]["slope"])
        net = _probe(name, alpha)
        rows.append(dict(act=name, alpha=alpha,
                         forward_equals_relu=bool(torch.equal(net.act_fn(grid), a_relu))))
    proj = _probe("bwd_leaky_proj", float(cfg["activation"]["bwd_leaky_proj"]["slope"]))
    rows.append(dict(act="bwd_leaky_proj.grad == bwd_leaky.grad", alpha=float("nan"),
                     forward_equals_relu=bool(torch.equal(
                         proj.act_grad(grid, a_relu), bl.act_grad(grid, a_relu)))))
    # 負側ゲートの値（追補 4 の交差深度を記録に残す）
    refl = _probe("bwd_reflect", float(cfg["activation"]["bwd_reflect"]["slope"]))
    quad = _probe("bwd_quad", float(cfg["activation"]["bwd_quad"]["slope"]))
    probes = {}
    for z in (-1.0, -10.0, -30.0, -100.0):
        t = torch.tensor([z], dtype=torch.float64)
        probes[str(z)] = dict(
            bwd_reflect=float(refl.act_grad(t, refl.act_fn(t))[0]),
            bwd_quad=float(quad.act_grad(t, quad.act_fn(t))[0]))
    return dict(pass_=all(r["forward_equals_relu"] for r in rows), rows=rows,
                negative_gate_by_depth=probes,
                crossover_depth_z=float(cfg["activation"]["bwd_quad"]["crossover_depth_z"]))


def _s_limit_static(cfg: dict) -> dict:
    """S-limit の静的側: $a=0$ で ReLU と bit 一致し、**負のゼロを作らない**（追補 9）。"""
    grid = _grid(cfg)
    relu = _probe("relu", 1.0)
    a_relu = relu.act_fn(grid)
    g_relu = relu.act_grad(grid, a_relu)
    rows, failures = [], []
    for name in ("bwd_reflect", "bwd_quad", "bwd_leaky_proj"):
        net = _probe(name, 0.0)
        got_a, got_g = net.act_fn(grid), net.act_grad(grid, net.act_fn(grid))
        row = dict(act=name,
                   forward_equal=bool(torch.equal(got_a, a_relu)),
                   grad_equal=bool(torch.equal(got_g, g_relu)),
                   negative_zeros=int(torch.signbit(got_g).sum()),
                   relu_negative_zeros=int(torch.signbit(g_relu).sum()))
        rows.append(row)
        if not (row["forward_equal"] and row["grad_equal"]
                and row["negative_zeros"] == row["relu_negative_zeros"]):
            failures.append(row)
    return dict(pass_=not failures, rows=rows, failures=failures,
                signbit_checked=bool(cfg["sanity"]["s_limit_check_signbit"]))


def _s_limit_trained(cfg: dict, act: str, outdir: Path) -> dict:
    """S-limit の走側: $a=0$ の幻が ReLU 腕と 30k 短走で bit 一致すること。"""
    steps = int(cfg["sanity"]["s_limit_steps"])
    c = copy.deepcopy(cfg)
    c["common"]["seeds"] = [0, 1]
    base = copy.deepcopy(_arm(c, "RWw_1216"))
    base["wd_w"] = 0.0
    relu = setup_arm_phantom(c, base, "cpu")
    other = setup_arm_phantom(c, base, "cpu")
    other["net"].set_activation(act, 0.0, "alpha_exp")
    other["activation"], other["act_alpha"] = act, 0.0
    other["mu_projection"] = act == "bwd_leaky_proj"
    train_arm_phantom(relu, lambda *_: None, [], steps, outdir, [])
    train_arm_phantom(other, lambda *_: None, [], steps, outdir, [])
    a, b = _init_hashes(relu), _init_hashes(other)
    differences = sorted(k for k, v in a.items() if b.get(k) != v)
    return dict(pass_=not differences, activation=act, steps=steps,
                trained_state_differences=differences,
                note=("BLP は a=0 でも射影経路を通るが、負側の勾配が 0 なので "
                      "射影する対象が無く ReLU と一致する"))


def _s_limit_wd_w(cfg: dict, outdir: Path) -> dict:
    steps = int(cfg["sanity"]["s_limit_steps"])
    c = copy.deepcopy(cfg)
    c["common"]["seeds"] = [0, 1]
    zero = copy.deepcopy(_arm(c, "RWw_1216"))
    zero["wd_w"] = 0.0
    a_state = setup_arm_phantom(c, zero, "cpu")
    b_state = setup_arm_phantom(c, zero, "cpu")
    b_state["net"].wd_w = 0.0
    train_arm_phantom(a_state, lambda *_: None, [], steps, outdir, [])
    train_arm_phantom(b_state, lambda *_: None, [], steps, outdir, [])
    ha, hb = _init_hashes(a_state), _init_hashes(b_state)
    differences = sorted(k for k, v in ha.items() if hb.get(k) != v)
    return dict(pass_=not differences, steps=steps,
                trained_state_differences=differences)


def _hand_net(cfg: dict, act: str, alpha: float, x: torch.Tensor) -> VecMLPL:
    net = VecMLPL(1, [3], x.shape[1], torch.Generator().manual_seed(7), "cpu")
    net.set_activation(act, alpha, "alpha_exp")
    net.Ws[0].zero_()
    net.Ws[0][0, 0, 0] = 1.0
    net.Ws[0][0, 1, 1] = 1.0
    net.bs[0].zero_()
    net.bs[0][0, 0] = -3.0 - float(x[0, 0])          # unit 0: 負側
    net.bs[0][0, 1] = 3.0 - float(x[0, 1])           # unit 1: 正側
    net.bs[0][0, 2] = -1.0                           # unit 2: 負側
    net.v.copy_(torch.tensor([[0.7, -0.4, 0.25]]))
    net.c.zero_()
    net.W, net.b = net.Ws[0], net.bs[0]
    return net


def _s_bwd(cfg: dict) -> dict:
    """S-bwd: 沈下ユニットを手で置き、型ごとに §2 の表どおりの更新になること。

    ``bwd_leak_0902`` §6.2 追補 2 と同じ扱いで、**照合は勾配の水準で bit 一致**を
    要求する（パラメタの差分は float32 の減算 1 回ぶん丸まるので差分に bit 一致を
    要求できない）。
    """
    a = float(cfg["activation"]["bwd_reflect"]["slope"])
    a_q = float(cfg["activation"]["bwd_quad"]["slope"])
    g = torch.Generator().manual_seed(90211)
    x = torch.randn(1, 4, generator=g)
    y = torch.randn(1, generator=g)

    def grads(net: VecMLPL):
        pres, acts, yhat = net.forward_layers(x)
        delta = yhat - y
        d2 = 2.0 * delta
        dz = d2[:, None] * net.v * net.act_grad(pres[0], acts[0])
        return dict(pre=pres[0], delta=delta, v=net.v.clone(),
                    gW=dz[:, :, None] * x[:, None, :], gb=dz,
                    gv=d2[:, None] * acts[0])

    rows, failures = [], []
    leaky_ref = grads(_hand_net(cfg, "bwd_leaky", a, x))
    neg = (leaky_ref["pre"][0] <= 0).nonzero().flatten().tolist()
    pos = (leaky_ref["pre"][0] > 0).nonzero().flatten().tolist()

    # BLR: 負側が leaky 腕の**符号反転**と bit 一致、v は凍結
    r = grads(_hand_net(cfg, "bwd_reflect", a, x))
    blr_ok = bool(torch.equal(r["gb"][0, neg], -leaky_ref["gb"][0, neg])
                  and torch.equal(r["gW"][0, neg], -leaky_ref["gW"][0, neg])
                  and torch.equal(r["gb"][0, pos], leaky_ref["gb"][0, pos])
                  and torch.equal(r["gv"][0, neg], torch.zeros(len(neg))))
    rows.append(dict(type="BLR", negative_units=neg,
                     sign_flip_of_leaky=blr_ok,
                     gv_zero_on_negative=bool(torch.equal(
                         r["gv"][0, neg], torch.zeros(len(neg))))))
    if not blr_ok:
        failures.append(dict(where="BLR"))

    # BLQ: 負側が leaky 腕の (a_Q * z / a) 倍と bit 一致
    q = grads(_hand_net(cfg, "bwd_quad", a_q, x))
    scale = (a_q * q["pre"][0, neg]) / a
    want_gb = leaky_ref["gb"][0, neg] * scale
    blq_ok = bool(torch.allclose(q["gb"][0, neg], want_gb, rtol=1e-6, atol=0.0)
                  and torch.equal(q["gv"][0, neg], torch.zeros(len(neg))))
    rows.append(dict(type="BLQ", negative_units=neg,
                     depth_scaled_match=blq_ok,
                     gate_values=(a_q * q["pre"][0, neg]).tolist(),
                     gv_zero_on_negative=bool(torch.equal(
                         q["gv"][0, neg], torch.zeros(len(neg))))))
    if not blq_ok:
        failures.append(dict(where="BLQ"))

    # BLP: 射影後に gW.mu = 0、gb = 0、mu 直交成分は leaky 腕と一致、v は凍結。
    # mu-hat は乱数ではなく **実際の腕の µ̂** を使う（S-proj が方向も検査する）。
    c = copy.deepcopy(cfg)
    c["common"]["seeds"] = [0]
    st = setup_arm_phantom(c, _arm(c, "BLP_1216"), "cpu")
    mh = mu_hat(st)[:1]                                   # [1, m] float64
    xb = torch.randn(1, st["env"].m, generator=g)
    p_net = _hand_net(cfg, "bwd_leaky_proj", a, xb)
    l_net = _hand_net(cfg, "bwd_leaky", a, xb)
    pres, acts, yhat = p_net.forward_layers(xb)
    dl = yhat - y
    d2 = 2.0 * dl
    dz = d2[:, None] * p_net.v * p_net.act_grad(pres[0], acts[0])
    negp = (pres[0][0] <= 0).nonzero().flatten().tolist()
    posp = (pres[0][0] > 0).nonzero().flatten().tolist()
    x64 = xb.double()
    x_proj64 = x64 - (x64 * mh).sum(dim=1, keepdim=True) * mh
    x_proj = x_proj64.to(xb.dtype)
    gW_plain = dz[:, :, None] * xb[:, None, :]
    gW_proj = dz[:, :, None] * x_proj[:, None, :]
    submerged = pres[0] <= 0
    gW = torch.where(submerged[:, :, None], gW_proj, gW_plain)
    gb = torch.where(submerged, torch.zeros_like(dz), dz)
    # (i) float64 の解析経路での射影残差（gate はこちら・追補 7）
    resid64 = float((dz[0, negp].double()[:, None] * x_proj64 * mh).sum(dim=1).abs().max())
    # (ii) float32 で実際に適用される勾配の残差（併記のみ）
    resid32 = float((gW[0, negp].double() * mh).sum(dim=1).abs().max())
    # (iii) 発火ユニットには射影が掛かっていない（bit 一致）
    untouched = bool(torch.equal(gW[0, posp], gW_plain[0, posp])
                     and torch.equal(gb[0, posp], dz[0, posp]))
    # (iv) 沈下ユニットの b は厳密 0、v は凍結
    b_zero = bool(torch.equal(gb[0, negp], torch.zeros(len(negp))))
    gv_zero = bool(torch.equal((d2[:, None] * acts[0])[0, negp],
                               torch.zeros(len(negp))))
    # (v) 直交成分は射影前と一致（射影が µ 方向だけを抜いていること）
    keep = float((gW[0, negp].double()
                  - (gW_plain[0, negp].double()
                     - (gW_plain[0, negp].double() * mh).sum(dim=1, keepdim=True) * mh)
                  ).abs().max())
    tol = float(cfg["sanity"]["s_proj_tol"])
    blp_ok = bool(resid64 <= tol and untouched and b_zero and gv_zero and keep <= 1e-6)
    rows.append(dict(type="BLP", negative_units=negp, firing_units=posp,
                     projection_residual_float64=resid64,
                     projection_residual_float32=resid32,
                     tolerance=tol,
                     gate_precision=str(cfg["sanity"]["s_proj_gate_precision"]),
                     firing_units_untouched=untouched,
                     gb_zero_on_submerged=b_zero, gv_zero_on_submerged=gv_zero,
                     orthogonal_component_preserved=keep))
    if not blp_ok:
        failures.append(dict(where="BLP", resid64=resid64, untouched=untouched,
                             b_zero=b_zero, gv_zero=gv_zero, keep=keep))

    return dict(pass_=not failures, rows=rows, failures=failures,
                comparison="bit identity at the gradient level (追補 2 を継承)",
                note=("S-proj の float64 gate と float32 実現値の両方を出す。"
                      "float32 では eps*|g| ~ 1e-7 が下限で 1e-12 は到達不能（追補 7）"))


def _s_proj(cfg: dict) -> dict:
    """S-proj: 射影に使う $\\hat\\mu$ が実際に加えた用量ベクトルの方向と一致すること。

    ``mu_hat`` は ``env.flip_state`` から再構成する。これが ``_input_stats`` が
    報告する µ（前向きがオフセットを引いたあとの平均）と方向一致することを、
    相対誤差 ``s_proj_mu_rel_tol`` で検査する。あわせてタスク境界をまたいで
    µ̂ が実際に回ることも記録する（凍結ユニットでも M が動く理由・追補 4 の隣）。
    """
    tol = float(cfg["sanity"]["s_proj_mu_rel_tol"])
    c = copy.deepcopy(cfg)
    c["common"]["seeds"] = [0, 1, 2, 3]
    st = setup_arm_phantom(c, _arm(c, "BLP_1216"), "cpu")
    rows, failures = [], []
    prev = None
    for step in range(0, 30_001, 10_000):
        _refresh_fixed_offset(st)
        stats = _input_stats(st)
        got = mu_hat(st)
        # _input_stats は ||µ|| と cos しか返さないので、µ そのものを同じ式で作り直す
        free = st["env"].m - st["env"].f
        tail = torch.full((st["R"], free), 0.5, dtype=torch.float64)
        raw = torch.cat([st["env"].flip_state.double(), tail], dim=1)
        gamma = gamma_for_k(st["env"].flip_state.double().sum(dim=1),
                            float(st["target_mu_norm"]))
        want = raw - 0.5 * gamma[:, None]
        cos = (got * want).sum(dim=1) / want.norm(dim=1).clamp_min(1e-300)
        norm_err = float((want.norm(dim=1) - float(st["target_mu_norm"])).abs().max()
                         / float(st["target_mu_norm"]))
        unit_err = float((got.norm(dim=1) - 1.0).abs().max())
        cos_err = float((cos - 1.0).abs().max())
        rotation = (float((got * prev).sum(dim=1).min()) if prev is not None
                    else float("nan"))
        rows.append(dict(step=step, mu_norm_rel_err=norm_err,
                         mu_hat_unit_err=unit_err, direction_cos_err=cos_err,
                         min_cos_with_previous_probe=rotation))
        if norm_err > tol or unit_err > tol or cos_err > tol:
            failures.append(rows[-1])
        prev = got.clone()
        for _ in range(10_000):
            x = st["env"].step()
            st["teacher"](x)
    return dict(pass_=not failures, tolerance=tol, rows=rows, failures=failures,
                note="mu is a function of env.flip_state only; no exact-support "
                     "enumeration is needed (追補 12)")


def _s_wd_w(cfg: dict) -> dict:
    """S-wd-w: step 1 で `RWw` と `R` の差が $W$ のみ・厳密に $-\\eta\\lambda_w W$。"""
    lam = float(cfg["weight_decay_w"]["lambda_w"])
    ulp = float(cfg["sanity"]["s_wd_w_ulp_factor"])
    lr_value = float(cfg["common"]["lr_main"])
    R, hidden, d = 4, [100], int(cfg["condA"]["m"])
    gen = torch.Generator().manual_seed(20260906)
    ref = VecMLPL(R, hidden, d, torch.Generator().manual_seed(11), "cpu")
    state = ref.state_dict()
    lr = torch.full((R,), lr_value)
    gW = torch.randn(R, hidden[0], d, generator=gen)
    gb = torch.randn(R, hidden[0], generator=gen)
    gv = torch.randn(R, hidden[0], generator=gen)
    gc = torch.randn(R, generator=gen)

    def stepped(wd_w: float) -> VecMLPL:
        net = VecMLPL(R, hidden, d, torch.Generator().manual_seed(11), "cpu")
        net.load_state(state)
        net.set_weight_decay_w(wd_w)
        net.sgd_step_layers(lr, [gW], [gb], gv, gc)
        return net

    zero, decayed = stepped(0.0), stepped(lam)
    untouched = bool(torch.equal(zero.bs[0], decayed.bs[0])
                     and torch.equal(zero.v, decayed.v)
                     and torch.equal(zero.c, decayed.c))
    expected = -lr[:, None, None] * lam * state["W"]
    err = float((decayed.Ws[0] - zero.Ws[0] - expected).abs().max())
    eps = float(torch.finfo(zero.Ws[0].dtype).eps)
    tol = ulp * eps * float(state["W"].abs().max())
    signal = float(expected.abs().max())
    plain = state["W"] - lr[:, None, None] * gW
    identity = bool(torch.equal(zero.Ws[0], plain))
    return dict(pass_=bool(untouched and err <= tol and signal > 0 and identity),
                lam=lam, b_v_c_untouched=untouched, wd_zero_is_identity=identity,
                W_delta_max_abs_err=err, W_delta_tol_ulp=tol, W_delta_signal=signal)


def _s_refl(cfg: dict, outdir: Path) -> dict:
    """S-refl: 30k 短走で沈下ユニットの $\\bar z$ が `BL` 短走と**逆向き**に動く。

    符号だけを見る（数値は引かない）。`BL` は `bwd_leak_0902` の代替勾配 $+a$、
    `BLR` は $-a$ なので、沈下ユニットの $\\bar z$ の平均変化が符号反転するはず。
    """
    steps = int(cfg["sanity"]["s_refl_steps"])
    every = int(cfg["common"]["lop_every"])
    c = copy.deepcopy(cfg)
    c["common"]["seeds"] = [0, 1]
    probes = list(range(0, steps + 1, every))
    out = {}
    for label, act in (("BL", "bwd_leaky"), ("BLR", "bwd_reflect")):
        arm = copy.deepcopy(_arm(c, "BLR_1216"))
        arm["wd_w"] = 0.0
        st = setup_arm_phantom(c, arm, "cpu")
        st["net"].set_activation(act, float(cfg["activation"]["bwd_reflect"]["slope"]),
                                 "alpha_exp")
        st["activation"], st["mu_projection"] = act, False
        rec = BwdRecorder(probes, st, record_units=False)
        train_arm_phantom(st, rec, probes, steps, outdir, [])
        zbar = rec.layers[0]["zbar"]                      # [n, R, h] float32
        p_hat = rec.layers[0]["p_hat"]
        sub = p_hat[0] == 0.0                             # step 0 で沈下していたユニット
        drift = float(np.mean((zbar[-1] - zbar[0])[sub])) if sub.any() else float("nan")
        out[label] = dict(mean_zbar_drift=drift,
                          n_submerged_at_step0=int(sub.sum()))
    signs_opposite = bool(np.isfinite(out["BL"]["mean_zbar_drift"])
                          and np.isfinite(out["BLR"]["mean_zbar_drift"])
                          and np.sign(out["BL"]["mean_zbar_drift"])
                          != np.sign(out["BLR"]["mean_zbar_drift"]))
    # **REPORT_ONLY へ降格（spec §10.3 追補 13・9/2 Issa 裁定 A）。**
    # 登録時は必須ゲートだったが、これは実装の検査ではなく**本走の主 endpoint と
    # 同じ物理の主張**を検査するゲートで、30k では符号が安定しない（沈下ユニットの
    # 定義 3 通り・窓 4 通りのいずれでも反転しない）。実装側は他の 11 ゲートが
    # 保証している。測定はすべて残し、gate としては使わない。
    return dict(pass_=True, report_only=True, gate_demoted_to_report_only=True,
                registered_expectation_met=signs_opposite,
                steps=steps, arms=out, sign_only=True,
                note=("spec §6 S-refl は登録時 必須。§10.3 追補 13 により "
                      "REPORT_ONLY へ降格した。registered_expectation_met が "
                      "実際の判定結果で、これが False でも本走は止まらない"))


def _s_pair_and_dose(cfg: dict, outdir: Path) -> dict:
    """S-pair / S-dose: 7 腕どうし・親走との init/教師/入力列/flip の bit 一致。

    全腕が同一用量（12.16）なので、``running_mean`` も含めて**全腕で一致**を
    要求できる（`bwd_leak_0902` は 2 用量だったので群内比較が要った）。
    """
    steps = int(cfg["sanity"]["s_pair_steps"])
    every = int(cfg["common"]["lop_every"])
    seeds = [int(v) for v in cfg["common"]["seeds"]]
    init, final, streams, dose_rows = {}, {}, {}, []
    per_seed, flip0 = {}, {}
    for arm in ARM_ORDER:
        c = copy.deepcopy(cfg)
        st = setup_arm_phantom(c, _arm(c, arm), "cpu")
        init[arm] = _init_hashes(st)
        per_seed[arm] = {int(r["seed"]): _seed_state_hashes_p1(st, i)
                         for i, r in enumerate(st["runs"])}
        flip0[arm] = {int(r["seed"]):
                      st["env"].flip_state[i].detach().cpu().numpy().astype(np.float32)
                      for i, r in enumerate(st["runs"])}
        stream = StreamDigest()

        def dose_probe(state: dict, step: int, arm_name: str = arm) -> None:
            _refresh_fixed_offset(state)
            stats = _input_stats(state)
            dose_rows.append(dict(arm=arm_name, step=int(step),
                                  max_relative_error=float(
                                      stats["relative_error"].abs().max())))

        print(f"[S-pair/S-dose] {arm} {steps:,} steps", flush=True)
        train_arm_phantom(st, dose_probe, range(0, steps + 1, every), steps,
                          outdir, [], stream_hook=stream)
        final[arm], streams[arm] = _env_hashes(st), stream.digest()

    reference, differences = ARM_ORDER[0], []
    for arm in ARM_ORDER[1:]:
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
                    differences.append(dict(arm=arm, seed=seed, where=f"seed_hash.{key}"))

    parent = Path(ROOT) / cfg["sanity"]["s_pair_reference"] / "logs"
    ref_arm = str(cfg["sanity"]["s_pair_reference_arm"])
    parent_rows, parent_missing = [], []
    for arm in ARM_ORDER:
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

    tol = float(cfg["sanity"]["s_dose_rel_tol"])
    dose_fail = [r for r in dose_rows if float(r["max_relative_error"]) > tol]
    return dict(
        spair=dict(pass_=bool(not differences and not parent_missing),
                   reference=reference, arms=list(ARM_ORDER), steps=steps,
                   match_by="seed_init_hash", differences=differences,
                   parent_flip_rows=parent_rows, parent_missing=parent_missing,
                   caveat="init/teacher/input realization only; trajectories "
                          "diverge after step 1"),
        sdose=dict(pass_=not dose_fail, tolerance=tol, n_probes=len(dose_rows),
                   failures=dose_fail))


def _s_taut(cfg: dict, outdir: Path) -> dict:
    """S-taut: 未フィット率が介入で恒真になっていないこと＋判定表の検算。"""
    c = copy.deepcopy(cfg)
    c["common"]["seeds"] = [0, 1]
    values, hashes = {}, {}
    for arm in ("RWw_1216", "BLR_1216", "BLQ_1216", "BLP_1216"):
        st = setup_arm_phantom(c, _arm(c, arm), "cpu")
        train_arm_phantom(st, lambda *_: None, [], 2000, outdir, [])
        rec, _ = exact_layer_record_elu(st, SIGMA_TOL)
        values[arm] = rec["run"]["unfit"].detach().cpu().numpy().tolist()
        hashes[arm] = {k: _sha_array(v) for k, v in st["net"].state_dict().items()}
    distinct = len({json.dumps(h, sort_keys=True) for h in hashes.values()}) == len(hashes)
    finite = all(np.isfinite(np.asarray(v)).all() and (np.asarray(v) > 0).all()
                 for v in values.values())
    mutants = {
        "diverge": DECISION_TABLE["diverged|zero|present"],
        "rescue": DECISION_TABLE["zero|present|present"],
        "spring": DECISION_TABLE["present|zero|present"],
        "wd_alone": DECISION_TABLE["present|present|zero"],
        "deaf": DECISION_TABLE["present|present|present"],
        "partial": DECISION_TABLE["mid|mid|present"],
    }
    expected = dict(diverge="PHANTOM_DIVERGES", rescue="PHANTOM_RESCUES",
                    spring="PHANTOM_NEEDS_SPRING", wd_alone="WD_W_SUFFICIENT_ALONE",
                    deaf="PHANTOM_DEAF", partial="PARTIAL")
    return dict(pass_=bool(distinct and finite and mutants == expected),
                all_arms_distinct=distinct, unfit_finite_positive=finite,
                short_run_unfit=values, verdict_mutants=mutants, expected=expected)


def _s_ref(cfg: dict) -> dict:
    """S-ref: 対照 2 走の出力が各 provenance の output_sha256 と一致すること。"""
    out, ok = {}, True
    for run in ("results/gate_dose_0830", "results/bwd_leak_0902"):
        ref = (Path(ROOT) / run).resolve()
        prov = ref / "provenance.json"
        if not prov.exists():
            out[run] = dict(pass_=False, reason="missing provenance")
            ok = False
            continue
        recorded = dict(json.loads(prov.read_text(encoding="utf-8"))
                        .get("output_sha256", {}))
        read_files = ["verdict.csv"] + [
            f"logs/{a}_seed{s}.npz" for a in CONTROL_ORDER for s in range(10)
            if (ref / f"logs/{a}_seed{s}.npz").exists()]
        mismatches, missing = [], []
        for name, want in recorded.items():
            path = ref / name
            if not path.exists():
                missing.append(name)
            elif _sha_file(path) != want:
                mismatches.append(name)
        read_bad = [n for n in mismatches if n in read_files]
        # gate_dose_0830 の verdict.csv は a930b6e で再生成されている（bwd_leak §6.2
        # 追補 4 と同じ事情）。転記する列が動いていないかを列単位で確かめる。
        column_note = None
        if "verdict.csv" in read_bad:
            column_note = _verdict_columns_unchanged(cfg, run, recorded["verdict.csv"])
            if column_note.get("unchanged"):
                read_bad = [n for n in read_bad if n != "verdict.csv"]
        out[run] = dict(pass_=bool(not read_bad and not missing),
                        files_checked=len(recorded),
                        hash_mismatches_on_read_files=read_bad,
                        hash_mismatches_on_unread_files=[n for n in mismatches
                                                         if n not in read_files],
                        recorded_but_missing=missing,
                        verdict_column_check=column_note)
        ok = ok and out[run]["pass_"]
    return dict(pass_=ok, runs=out,
                note="logs/*.npz are gitignored; the unit-level endpoints are not "
                     "reproducible from a fresh clone")


def _verdict_columns_unchanged(cfg: dict, run: str, want_sha: str) -> dict:
    """provenance の sha に一致する版を履歴から探し、転記する列を突き合わせる。"""
    import hashlib
    import io

    rel = f"{run}/verdict.csv"
    columns = list(cfg["controls"]["endpoint_columns"])
    try:
        revs = subprocess.check_output(["git", "log", "--format=%H", "--", rel],
                                       cwd=ROOT, text=True).split()
    except (OSError, subprocess.CalledProcessError) as exc:
        return dict(checked=False, reason=str(exc))
    for rev in revs:
        try:
            raw = subprocess.check_output(["git", "show", f"{rev}:{rel}"], cwd=ROOT)
        except (OSError, subprocess.CalledProcessError):
            continue
        if hashlib.sha256(raw).hexdigest() != want_sha:
            continue
        then = {r["arm"]: r for r in csv.DictReader(io.StringIO(raw.decode("utf-8")))}
        now = {r["arm"]: r for r in csv.DictReader(
            (Path(ROOT) / rel).read_text(encoding="utf-8").splitlines())}
        differing = [dict(arm=a, column=c) for a in CONTROL_ORDER
                     if a in then and a in now for c in columns
                     if then[a].get(c) != now[a].get(c)]
        return dict(checked=True, provenance_era_commit=rev, differing=differing,
                    unchanged=not differing)
    return dict(checked=False, reason="no commit matches the recorded sha256")


def _s_floor_and_ci(cfg: dict) -> tuple[dict, dict]:
    reference = Path(ROOT) / "results/gate_dose_0830/floor_calibration.csv"
    floor = dict(pass_=False, reference=str(reference), reason="missing")
    if reference.exists():
        data = np.genfromtxt(reference, delimiter=",", names=True)
        values = np.unique(np.asarray(data["calibrated_floor"], dtype=np.float64))
        configured = float(cfg["phase1"]["unfit_floor"])
        floor = dict(pass_=bool(values.size == 1 and values[0] == configured
                                and cfg["phase1"]["recalibrate_floor"] is False),
                     reference=str(reference), reference_values=values.tolist(),
                     configured=configured)
    P = cfg["phase1"]
    n = len(cfg["common"]["seeds"])
    draws = np.random.default_rng(int(P["bootstrap_seed"])).integers(0, n, size=(int(P["bootstrap_B"]), n))
    result = _ci_components(np.zeros(n), draws, "median", float(P["degenerate_se_tol"]),
                            float(P["degenerate_frac_max"]),
                            float(P["degenerate_width_ratio_max"]))
    return floor, dict(pass_=bool(result["ci_degenerate"]), result=result)


def preflight(cfg: dict, outdir: Path) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    checks: dict[str, dict] = {"S1_omp": require_omp(cfg)}
    print("[S-cross] phantom forwards are strict ReLU", flush=True)
    checks["S_cross"] = _s_cross(cfg)
    print("[S-limit] static a -> 0 (incl. signbit)", flush=True)
    checks["S_limit_static"] = _s_limit_static(cfg)
    print("[S-bwd] hand-placed submerged unit, per type", flush=True)
    checks["S_bwd"] = _s_bwd(cfg)
    print("[S-proj] mu-hat direction", flush=True)
    checks["S_proj"] = _s_proj(cfg)
    print("[S-wd-w] one step of W-only weight decay", flush=True)
    checks["S_wd_w"] = _s_wd_w(cfg)
    print("[S-ref] parent output hashes", flush=True)
    checks["S_ref"] = _s_ref(cfg)
    for act in ("bwd_reflect", "bwd_quad", "bwd_leaky_proj"):
        print(f"[S-limit] {act} a -> 0 over 30k", flush=True)
        checks[f"S_limit_{act}"] = _s_limit_trained(cfg, act, outdir / f"sl_{act}")
    print("[S-limit] wd_w -> 0", flush=True)
    checks["S_limit_wd_w"] = _s_limit_wd_w(cfg, outdir / "sl_wdw")
    print("[S-refl] BLR drifts opposite to BL (REPORT_ONLY・追補 13)", flush=True)
    checks["S_refl"] = _s_refl(cfg, outdir / "srefl")
    pair = _s_pair_and_dose(cfg, outdir / "spair")
    checks["S_pair"], checks["S_dose"] = pair["spair"], pair["sdose"]
    print("[S-taut] endpoint is not tautological", flush=True)
    checks["S_taut"] = _s_taut(cfg, outdir / "staut")
    checks["S6_floor_inherited"], checks["S_CI_degeneracy"] = _s_floor_and_ci(cfg)
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
# Analysis
# ---------------------------------------------------------------------------
def _onset_state(n_onset: int | None, zero_max: int, present_min: int) -> str:
    if n_onset is None:
        return "diverged"
    if int(n_onset) <= zero_max:
        return "zero"
    if int(n_onset) >= present_min:
        return "present"
    return "mid"


def _load_controls(cfg: dict) -> dict:
    """対照の主 endpoint を各親走の ``verdict.csv`` から**転記**する。"""
    floor = float(cfg["phase1"]["unfit_floor"])
    out = {}
    for arm, meta in cfg["controls"]["arms"].items():
        path = Path(ROOT) / str(meta["source_run"]) / "verdict.csv"
        with path.open(newline="") as fh:
            row = next((r for r in csv.DictReader(fh) if r["arm"] == arm), None)
        if row is None:
            raise SanityError(f"control {arm} missing from {path}")
        u5 = np.maximum(np.asarray(json.loads(row["U_5m_seed_values"]), dtype=np.float64), floor)
        u1 = np.maximum(np.asarray(json.loads(row["U_1m_seed_values"]), dtype=np.float64), floor)
        out[arm] = dict(u_5m=u5, u_1m=u1, log_u_5m=np.log10(u5), log_u_1m=np.log10(u1),
                        n_onset_5m=int(row["n_onset_5m"]), n_onset_1m=int(row["n_onset_1m"]),
                        source=str(path), source_run=str(meta["source_run"]))
    return out


def _load_new_arm(cfg: dict, outdir: Path, arm: str) -> dict:
    from .gate_dose import _load_arm

    data = _load_arm(cfg, outdir, arm)
    P = cfg["phase1"]
    return {"data": data,
            "5M": _window(data, cfg, list(P["late_tasks_5m"])),
            "1M": _window(data, cfg, list(P["window_1m_tasks"])),
            "early": _window(data, cfg, list(P["early_tasks"]))}


def _p5_label(G: dict, ci: dict) -> tuple[str, bool]:
    margin = float(G["p5_equivalence_margin"])
    lo, hi = float(ci["percentile_ci_lo"]), float(ci["percentile_ci_hi"])
    below = bool(hi < 0.0)
    if lo >= -margin and hi <= margin:
        return str(G["p5_labels"]["equivalent"]), below
    if lo > 0.0:
        return str(G["p5_labels"]["short_of_lr"]), below
    return str(G["p5_labels"]["inconclusive"]), below


def _phat_histogram(path: Path, cfg: dict) -> np.ndarray:
    """末尾窓のユニット別 $\\hat p$ のヒストグラム（bin 幅 1/32・33 bin）。"""
    P, G = cfg["phase1"], _P(cfg)["phat_histogram"]
    with np.load(path, allow_pickle=False) as z:
        step = z["step"].astype(np.int64)
        idx = _window_indices(step, int(P["task_period"]), list(P["late_tasks_5m"]))
        p = z["layer1_p_hat"][idx].astype(np.float64).ravel()
    edges = (np.arange(int(G["n_bins"]) + 1) - 0.5) * float(G["bin_width"])
    counts, _ = np.histogram(p, bins=edges)
    return counts


def _w_growth_exponent(path: Path, cfg: dict) -> dict:
    """沈下ユニットの $\\lVert w\\rVert$ を 1M〜5M で log-log 回帰（§5.3）。"""
    P, G = cfg["phase1"], _P(cfg)["w_growth_exponent"]
    lo, hi = [int(v) for v in G["range_steps"]]
    with np.load(path, allow_pickle=False) as z:
        step = z["step"].astype(np.int64)
        tail = _window_indices(step, int(P["task_period"]), list(P["late_tasks_5m"]))
        sub = (z["layer1_p_hat"][tail] == 0.0).all(axis=0)     # 末尾窓で常に沈下
        keep = (step >= lo) & (step <= hi) & (step % int(P["task_period"]) == 0)
        wn = z["layer1_w_norm"][keep].astype(np.float64)
        t = step[keep].astype(np.float64)
    n_units = int(sub.sum())
    if n_units < int(G["min_submerged_units_per_seed"]) or len(t) < int(G["min_record_points"]):
        return dict(status=str(G["insufficient_data_label"]), slope=float("nan"),
                    r_squared=float("nan"), n_points=int(len(t)), n_units=n_units)
    y = np.median(wn[:, sub], axis=1)
    good = np.isfinite(y) & (y > 0)
    if good.sum() < int(G["min_record_points"]):
        return dict(status=str(G["insufficient_data_label"]), slope=float("nan"),
                    r_squared=float("nan"), n_points=int(good.sum()), n_units=n_units)
    lx, ly = np.log(t[good]), np.log(y[good])
    slope, intercept = np.polyfit(lx, ly, 1)
    pred = slope * lx + intercept
    ss_res = float(((ly - pred) ** 2).sum())
    ss_tot = float(((ly - ly.mean()) ** 2).sum())
    return dict(status="OK", slope=float(slope),
                r_squared=float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
                n_points=int(good.sum()), n_units=n_units)


def _blp_mu_invariance(path: Path, cfg: dict) -> dict:
    """`BLP` の µ 不変性。**不変量は生の $\\bar z$**（追補 8）。$\\Delta s$ と denom も出す。"""
    P, G = cfg["phase1"], _P(cfg)["blp_mu_invariance"]
    with np.load(path, allow_pickle=False) as z:
        step = z["step"].astype(np.int64)
        idx = _window_indices(step, int(P["task_period"]), list(P["late_tasks_5m"]))
        lo, hi = int(idx.min()), int(idx.max())
        span = slice(lo, hi + 1)
        zbar = z["layer1_zbar"][span].astype(np.float64)
        denom = z["layer1_denom"][span].astype(np.float64)
        p_hat = z["layer1_p_hat"][span].astype(np.float64)
        idx1 = _window_indices(step, int(P["task_period"]), list(P["window_1m_tasks"]))
        denom_1m = z["layer1_denom"][idx1].astype(np.float64)
    never_fired = (p_hat == 0.0).all(axis=0) if G["exclude_records_where_unit_fired"] \
        else np.ones(zbar.shape[1], dtype=bool)
    if not never_fired.any():
        return dict(status="INSUFFICIENT_DATA", n_units=0)
    dz = np.abs(np.diff(zbar[:, never_fired], axis=0)).max(axis=0)
    s = zbar / np.maximum(denom, 1e-300)
    ds = np.abs(np.diff(s[:, never_fired], axis=0)).max(axis=0)
    tol = float(G["primary_tolerance"])
    return dict(status="OK", n_units=int(never_fired.sum()),
                max_abs_delta_zbar=float(dz.max()),
                frac_units_within_tol=float((dz < tol).mean()),
                tolerance=tol,
                max_abs_delta_s=float(ds.max()),
                denom_growth_ratio=float(np.median(denom[-1, never_fired])
                                         / max(np.median(denom_1m[0, never_fired]), 1e-300)),
                note="不変量は zbar。s は denom 経由で動くので併記のみ（追補 8）")


def _freeze_v(path: Path, cfg: dict) -> dict:
    P = cfg["phase1"]
    with np.load(path, allow_pickle=False) as z:
        if "v_unit" not in z.files:
            return dict(status="NO_UNIT_LOG")
        step = z["step"].astype(np.int64)
        idx = _window_indices(step, int(P["task_period"]), list(P["late_tasks_5m"]))
        span = slice(int(idx.min()), int(idx.max()) + 1)
        v, p = z["v_unit"][span], z["layer1_p_hat"][span]
    frozen = (np.diff(v, axis=0) == 0.0).all(axis=0)
    dead = (p == 0.0).all(axis=0)
    return dict(status="OK", n_units=int(v.shape[1]),
                v_frozen_frac=float(frozen.mean()),
                strict_dead_all_window_frac=float(dead.mean()),
                v_frozen_xor_dead=int((frozen != dead).sum()))


def analyze(cfg: dict, outdir: Path, sanity: dict, elapsed: dict,
            divergences: dict) -> dict:
    seeds = [int(v) for v in cfg["common"]["seeds"]]
    P, G = cfg["phase1"], _P(cfg)
    draws = _draws(cfg)
    controls = _load_controls(cfg)
    expected = dict(G["control_expected_onset_5m"])
    got = {a: controls[a]["n_onset_5m"] for a in CONTROL_ORDER}
    if got != expected:
        raise SanityError(f"committed control onsets differ from the preregistration: "
                          f"expected {expected}, got {got}; the result must not be read")

    complete = [a for a in ARM_ORDER if a not in divergences]
    windows = {a: _load_new_arm(cfg, outdir, a) for a in complete}
    threshold = float(P["onset_threshold"])
    onset = {w: {a: int(np.sum(windows[a][w]["raw_u"] >= threshold)) for a in complete}
             for w in ("1M", "5M")}
    zero_max, present_min = int(G["onset_zero_max"]), int(G["onset_present_min"])
    states = {a: _onset_state(onset["5M"].get(a), zero_max, present_min)
              if a in complete else "diverged" for a in ARM_ORDER}
    spring_state = states[SPRING_CONTROL]

    # --- §5.1 型の判定（64 セル全列挙を引く・追補 5） ---
    types = {}
    for name, spec in cfg["phantom_types"].items():
        x, xw = states[str(spec["plain"])], states[str(spec["spring"])]
        key = f"{x}|{xw}|{spring_state}"
        types[name] = dict(
            state_triple=[x, xw, spring_state], verdict=DECISION_TABLE[key],
            co_satisfied=_co_satisfied(x, xw, spring_state),
            plain=str(spec["plain"]), spring=str(spec["spring"]),
            surrogate_constant=float(spec["surrogate_constant"]),
            surrogate_form=str(spec["surrogate_form"]))

    # 追補 6: 対照 RWw が present でないと 3 型とも潰れる。水準を主読みへ繰り上げる。
    guard = G["saturation_guard"]
    saturated = spring_state != str(guard["trigger_when_spring_control_not"])
    saturation = dict(triggered=bool(saturated), spring_control_state=spring_state,
                      action=str(guard["action"]) if saturated else "none",
                      promoted=list(guard["also_flag"]) if saturated else [])

    # --- §5.2 水準 ---
    def log_u(arm: str) -> np.ndarray | None:
        if arm in controls:
            return controls[arm]["log_u_5m"]
        if arm in windows:
            return windows[arm]["5M"]["log_u"]
        return None

    contrasts = {}

    def add(kind: str, high: str, low: str) -> None:
        label = f"{high}_minus_{low}"
        hv, lv = log_u(high), log_u(low)
        if hv is None or lv is None:
            contrasts[label] = dict(kind=kind, high=high, low=low,
                                    status=NUMERIC_DIVERGENCE)
            return
        values = np.asarray(hv) - np.asarray(lv)
        row = dict(kind=kind, high=high, low=low, status="OK", n_paired=len(values),
                   seed_values=values.tolist(), ci=_ci(cfg, values, draws),
                   sign_test=_sign_test(values),
                   cross_run=bool(low in controls or high in controls))
        if kind == "P5":
            row["label"], row["ci_below_zero"] = _p5_label(G, row["ci"])
            row["equivalence_margin"] = float(G["p5_equivalence_margin"])
        contrasts[label] = row

    for high, low in G["p3prime_contrasts"]:
        add("P3prime", high, low)
    for high, low in G["p5_contrasts"]:
        add("P5", high, low)

    p8 = {}
    for entry in G["p8_contrasts"]:
        a_hi, a_lo = entry["a"]
        b_hi, b_lo = entry["b"]
        parts = [log_u(x) for x in (a_hi, a_lo, b_hi, b_lo)]
        if any(x is None for x in parts):
            p8[str(entry["type"])] = dict(status=NUMERIC_DIVERGENCE,
                                          formula=f"({a_hi}-{a_lo})-({b_hi}-{b_lo})")
            continue
        values = (parts[0] - parts[1]) - (parts[2] - parts[3])
        p8[str(entry["type"])] = dict(
            status="OK", formula=f"({a_hi}-{a_lo})-({b_hi}-{b_lo})",
            seed_values=values.tolist(), ci=_ci(cfg, values, draws), label_emitted=False,
            note=("REPORT_ONLY・ラベルなし。3 型で第 2 項が同一なので P8 どうしを "
                  "引くと未登録の BLR-BLQ 対比になる。差は出さない（追補 4）"))

    levels_1m = {a: float(np.median(windows[a]["1M"]["log_u"])) for a in complete}
    for a in CONTROL_ORDER:
        levels_1m[a] = float(np.median(controls[a]["log_u_1m"]))
    finite = [v for v in levels_1m.values() if np.isfinite(v)]
    spread = float(max(finite) - min(finite)) if len(finite) > 1 else float("nan")
    ceiling = dict(spread_dex=spread, threshold=float(G["ceiling_baseline_spread_flag_dex"]),
                   flagged=bool(np.isfinite(spread)
                                and spread > float(G["ceiling_baseline_spread_flag_dex"])),
                   median_log10_U_1m=levels_1m)

    # --- §5.3 REPORT_ONLY ---
    ref_logs = {a: Path(ROOT) / str(m["source_run"]) / "logs"
                for a, m in cfg["controls"]["arms"].items()}
    hist_rows, growth_rows, invariance, revival_rows, freeze_rows = [], [], {}, [], []
    for arm in list(complete) + list(CONTROL_ORDER):
        src = outdir / "logs" if arm in complete else ref_logs[arm]
        for seed in seeds:
            path = src / f"{arm}_seed{seed}.npz"
            if not path.exists():
                continue
            counts = _phat_histogram(path, cfg)
            hist_rows.append(dict(arm=arm, seed=seed, is_control=int(arm in CONTROL_ORDER),
                                  **{f"bin_{i}": int(c) for i, c in enumerate(counts)}))
            growth_rows.append(dict(arm=arm, seed=seed,
                                    is_control=int(arm in CONTROL_ORDER),
                                    **_w_growth_exponent(path, cfg)))
            counts_r = _revival_counts(path)
            opp = counts_r["opportunities_within_task"]
            revival_rows.append(dict(arm=arm, seed=seed,
                                     is_control=int(arm in CONTROL_ORDER), **counts_r,
                                     rate_within_task=(counts_r["events_within_task"] / opp
                                                       if opp else float("nan"))))
            if arm in complete:
                freeze_rows.append(dict(arm=arm, seed=seed, **_freeze_v(path, cfg)))
    for arm in _P(cfg)["blp_mu_invariance"]["arms"]:
        if arm not in complete:
            invariance[arm] = dict(status=NUMERIC_DIVERGENCE)
            continue
        per_seed = [_blp_mu_invariance(outdir / "logs" / f"{arm}_seed{s}.npz", cfg)
                    for s in seeds]
        ok = [r for r in per_seed if r["status"] == "OK"]
        invariance[arm] = dict(
            status="OK" if ok else "INSUFFICIENT_DATA", n_seeds=len(ok),
            median_max_abs_delta_zbar=(float(np.median([r["max_abs_delta_zbar"] for r in ok]))
                                       if ok else float("nan")),
            median_frac_within_tol=(float(np.median([r["frac_units_within_tol"] for r in ok]))
                                    if ok else float("nan")),
            median_max_abs_delta_s=(float(np.median([r["max_abs_delta_s"] for r in ok]))
                                    if ok else float("nan")),
            median_denom_growth_ratio=(float(np.median([r["denom_growth_ratio"] for r in ok]))
                                       if ok else float("nan")),
            per_seed=per_seed)

    # --- P7 ---
    p7 = {}
    for arm in list(complete) + list(CONTROL_ORDER):
        src = outdir / "logs" if arm in complete else ref_logs[arm]
        if not all((src / f"{arm}_seed{s}.npz").exists() for s in seeds):
            continue
        per_seed = [_p7_seed_values(src / f"{arm}_seed{s}.npz", cfg) for s in seeds]
        p7[arm] = {k: np.asarray([r[k] for r in per_seed], dtype=np.float64)
                   for k in per_seed[0] if k.startswith("median")}
    p7_contrasts = {}
    for high, low in G["s_distribution"]["p7_contrasts"]:
        if high not in p7 or low not in p7:
            continue
        for channel in list(G["s_distribution"]["channels"]) + ["median_s"]:
            for unit_set in G["s_distribution"]["unit_sets"]:
                key = f"{high}_minus_{low}:{channel}:{unit_set}"
                values = p7[high][f"{channel}__{unit_set}"] - p7[low][f"{channel}__{unit_set}"]
                if not np.isfinite(values).all():
                    p7_contrasts[key] = dict(status="INSUFFICIENT_DATA", high=high,
                                             low=low, channel=channel, unit_set=unit_set)
                    continue
                p7_contrasts[key] = dict(status="OK", high=high, low=low, channel=channel,
                                         unit_set=unit_set, seed_values=values.tolist(),
                                         ci=_ci(cfg, values, draws))

    result = dict(arms_run=list(ARM_ORDER), complete=complete,
                  divergences=sorted(divergences), types=types,
                  onset_states=states, onset=onset, saturation_guard=saturation,
                  controls={a: dict(n_onset_5m=controls[a]["n_onset_5m"],
                                    n_onset_1m=controls[a]["n_onset_1m"],
                                    source_run=controls[a]["source_run"])
                            for a in CONTROL_ORDER},
                  contrasts=contrasts, p8=p8, ceiling=ceiling,
                  blp_mu_invariance=invariance, p7_contrasts=p7_contrasts,
                  elapsed_sec=elapsed)
    _write_outputs(cfg, outdir, complete, divergences, windows, controls, onset,
                   result, hist_rows, growth_rows, revival_rows, freeze_rows, p7, sanity)
    return result


def _write_outputs(cfg, outdir, complete, divergences, windows, controls, onset,
                   result, hist_rows, growth_rows, revival_rows, freeze_rows,
                   p7, sanity) -> None:
    seeds = [int(v) for v in cfg["common"]["seeds"]]
    G = _P(cfg)
    type_of = {str(s["plain"]): t for t, s in cfg["phantom_types"].items()}
    type_of.update({str(s["spring"]): t for t, s in cfg["phantom_types"].items()})
    rows = []
    for arm in ARM_ORDER:
        arm_cfg = _arm(cfg, arm)
        t = type_of.get(arm, "")
        base = dict(arm=arm, is_control=0, phantom_type=t,
                    activation=str(arm_cfg["activation"]), wd_w=float(arm_cfg["wd_w"]),
                    target_dose=float(arm_cfg["target_dose"]),
                    onset_state=result["onset_states"][arm],
                    type_verdict=result["types"][t]["verdict"] if t else "",
                    type_state_triple="|".join(result["types"][t]["state_triple"]) if t else "",
                    co_satisfied_labels="|".join(result["types"][t]["co_satisfied"]) if t else "",
                    surrogate_constant=(result["types"][t]["surrogate_constant"] if t else ""),
                    surrogate_form=(result["types"][t]["surrogate_form"] if t else ""),
                    saturation_guard_triggered=int(result["saturation_guard"]["triggered"]),
                    source="this run")
        if arm in complete:
            w = windows[arm]
            cp1 = clopper_pearson(onset["1M"][arm], len(seeds))
            cp5 = clopper_pearson(onset["5M"][arm], len(seeds))
            base.update(status="COMPLETE", NUMERIC_DIVERGENCE=0,
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
        rows.append(base)
    for arm in CONTROL_ORDER:
        c = controls[arm]
        rows.append(dict(
            arm=arm, is_control=1, phantom_type="",
            activation=str(cfg["controls"]["arms"][arm]["activation"]), wd_w=0.0,
            target_dose=12.16, onset_state=_onset_state(
                c["n_onset_5m"], int(G["onset_zero_max"]), int(G["onset_present_min"])),
            type_verdict="", type_state_triple="", co_satisfied_labels="",
            surrogate_constant="", surrogate_form="",
            saturation_guard_triggered=int(result["saturation_guard"]["triggered"]),
            source=f"COMMITTED {c['source_run']}",
            status="COMMITTED_OTHER_RUN", NUMERIC_DIVERGENCE=0,
            n_onset_1m=c["n_onset_1m"], cp95_1m_lo="", cp95_1m_hi="",
            U_1m_seed_values=json.dumps(c["u_1m"].tolist()),
            median_log10_U_1m=float(np.median(c["log_u_1m"])),
            n_onset_5m=c["n_onset_5m"], cp95_5m_lo="", cp95_5m_hi="",
            U_5m_seed_values=json.dumps(c["u_5m"].tolist()),
            median_log10_U_5m=float(np.median(c["log_u_5m"])),
            median_strict_dead_frac_5m="", median_submerged_frac_5m="",
            median_w_norm_5m="", median_eval_loss_exact_5m=""))
    write_csv(outdir / "verdict.csv", rows)

    contrast_rows = []
    for label, v in result["contrasts"].items():
        row = dict(endpoint=v["kind"], contrast=label, high=v["high"], low=v["low"],
                   status=v["status"], cross_run=int(v.get("cross_run", 0)),
                   n_paired=v.get("n_paired", ""), label=v.get("label", ""),
                   ci_below_zero=int(v["ci_below_zero"]) if "ci_below_zero" in v else "",
                   equivalence_margin=v.get("equivalence_margin", ""))
        ci = v.get("ci")
        for k in ("point", "percentile_ci_lo", "percentile_ci_hi",
                  "studentized_ci_lo", "studentized_ci_hi", "ci_degenerate"):
            row[k] = "" if ci is None else ci[k]
        st = v.get("sign_test") or {}
        row.update(sign_n_positive=st.get("n_positive", ""),
                   sign_n_negative=st.get("n_negative", ""),
                   sign_p_two_sided=st.get("p_two_sided", ""),
                   seed_values=json.dumps(v.get("seed_values", [])))
        contrast_rows.append(row)
    for t, v in result["p8"].items():
        row = dict(endpoint="P8", contrast=t, high="", low="", status=v["status"],
                   cross_run=1, n_paired="", label="", ci_below_zero="",
                   equivalence_margin="")
        ci = v.get("ci")
        for k in ("point", "percentile_ci_lo", "percentile_ci_hi",
                  "studentized_ci_lo", "studentized_ci_hi", "ci_degenerate"):
            row[k] = "" if ci is None else ci[k]
        row.update(sign_n_positive="", sign_n_negative="", sign_p_two_sided="",
                   seed_values=json.dumps(v.get("seed_values", [])))
        contrast_rows.append(row)
    write_csv(outdir / "layer_stats.csv", contrast_rows)

    if hist_rows:
        write_csv(outdir / "phat_hist.csv", hist_rows)
    if revival_rows:
        write_csv(outdir / "revival.csv", revival_rows)
    if growth_rows:
        fields = list(dict.fromkeys(k for r in growth_rows for k in r))
        write_csv(outdir / "w_growth.csv",
                  [{k: r.get(k, "") for k in fields} for r in growth_rows])
    if freeze_rows:
        fields = list(dict.fromkeys(k for r in freeze_rows for k in r))
        write_csv(outdir / "freeze_rates.csv",
                  [{k: r.get(k, "") for k in fields} for r in freeze_rows])
    s_rows = []
    for arm, values in p7.items():
        for i, seed in enumerate(seeds):
            row = dict(arm=arm, seed=seed, is_control=int(arm in CONTROL_ORDER))
            row.update({k: float(v[i]) for k, v in values.items()})
            s_rows.append(row)
    if s_rows:
        fields = list(dict.fromkeys(k for r in s_rows for k in r))
        write_csv(outdir / "s_distribution.csv",
                  [{k: r.get(k, "") for k in fields} for r in s_rows])
    p7_rows = []
    for key, v in result["p7_contrasts"].items():
        ci = v.get("ci")
        row = dict(key=key, status=v["status"], high=v["high"], low=v["low"],
                   channel=v["channel"], unit_set=v["unit_set"])
        for k in ("point", "percentile_ci_lo", "percentile_ci_hi",
                  "studentized_ci_lo", "studentized_ci_hi", "ci_degenerate"):
            row[k] = "" if ci is None else ci[k]
        p7_rows.append(row)
    if p7_rows:
        write_csv(outdir / "s_contrasts.csv", p7_rows)
    _write_summary(cfg, outdir, result, rows, sanity)


def _write_summary(cfg: dict, outdir: Path, result: dict, rows: list[dict],
                   sanity: dict) -> None:
    G = _P(cfg)
    xover = float(cfg["activation"]["bwd_quad"]["crossover_depth_z"])
    lines = [f"# {EXPERIMENT} summary", "", "## 型の verdict（§5.1）", "",
             "| 型 | 単独 | +w-WD | RWw | verdict | 条件を満たしていた行 | 代替勾配 |",
             "|---|---|---|---|---|---|---|"]
    for t, v in result["types"].items():
        x, xw, r = v["state_triple"]
        lines.append(f"| {t} | {x} | {xw} | {r} | **{v['verdict']}** | "
                     f"{', '.join(v['co_satisfied'])} | "
                     f"{v['surrogate_form']} ({v['surrogate_constant']}) |")
    sat = result["saturation_guard"]
    lines += ["",
              f"- 対照 `RWw` の状態: **{sat['spring_control_state']}**",
              (f"- **飽和ガード発動**（追補 6）: `RWw` が present でないので "
               f"§5.2 の水準（{', '.join(sat['promoted'])}）を主読みへ繰り上げる。"
               f"3 型の verdict は同じラベルに潰れている可能性がある"
               if sat["triggered"] else "- 飽和ガード: 発動せず"),
              f"- Numeric divergence: {', '.join(result['divergences']) or 'none'}",
              "", "### 引用上の注意", "",
              "- **用量 12.16 の 1 点**の主張。引用時に用量を添える。",
              "- 対照は**別走の committed 値**（`R_1216`/`LR_1216` は gate_dose_0830、",
              "  `RW_1216` は bwd_leak_0902）。ペアリングは init・教師・入力実現までで、",
              "  軌道は step 1 以降で分岐する。",
              f"- **`BLR`（a=0.1）と `BLQ`（a_Q=0.01）は係数が違う。** |gamma| が一致するのは",
              f"  z={xover:g} で、それより深いと `BLQ` の方が強い。両者の対比は登録していない。",
              "- P8 は 3 型で第 2 項が同一なので、P8 どうしを引かない（追補 4）。",
              "- P5 の等価限界 0.15 dex は**この系で較正していない**継承値。",
              "- **`BLP` の µ 不変量は生の zbar**。s は denom 経由で動く（追補 8）。",
              "- §1 の恒等式は**本 spec 初出の未登録の代数**であって既存ノートの定理ではない",
              "  （spec §10.1 追補 1）。その代数は `BLR` について §7.2 と逆を指している",
              "  （追補 3）が、9/2 裁定により予測は据え置いて凍結した。",
              "", "## Endpoints (5M)", "",
              "| arm | act | wd_w | onset 1M | onset 5M | median log10 U 5M | source |",
              "|---|---|---:|---:|---:|---:|---|"]
    for row in rows:
        if row["status"] == NUMERIC_DIVERGENCE:
            lines.append(f"| {row['arm']} | {row['activation']} | {row['wd_w']} | "
                         f"— | — | — | {row['status']} |")
            continue
        lines.append(f"| {row['arm']} | {row['activation']} | {row['wd_w']} | "
                     f"{row['n_onset_1m']}/10 | {row['n_onset_5m']}/10 | "
                     f"{row['median_log10_U_5m']:.6g} | {row['source']} |")
    lines += ["", "## Paired level contrasts at 5M (§5.2)", "",
              "| endpoint | contrast | n | median delta | percentile 95% CI | label | CI<0 | sign p |",
              "|---|---|---:|---:|---|---|---:|---:|"]
    for label, v in result["contrasts"].items():
        if v["status"] != "OK":
            lines.append(f"| {v['kind']} | {label} | — | — | {v['status']} | — | — | — |")
            continue
        ci, st = v["ci"], v["sign_test"]
        lines.append(f"| {v['kind']} | {label} | {v['n_paired']} | {ci['point']:.6g} | "
                     f"[{ci['percentile_ci_lo']:.6g}, {ci['percentile_ci_hi']:.6g}] | "
                     f"{v.get('label', '—')} | {int(v.get('ci_below_zero', 0))} | "
                     f"{st['p_two_sided']:.4g} |")
    lines += ["", "## P8（REPORT_ONLY・ラベルなし）", ""]
    for t, v in result["p8"].items():
        if v["status"] != "OK":
            lines.append(f"- {t}: {v['status']}")
            continue
        ci = v["ci"]
        lines.append(f"- {t}: {v['formula']} = {ci['point']:.6g}, "
                     f"95% CI [{ci['percentile_ci_lo']:.6g}, {ci['percentile_ci_hi']:.6g}]")
    inv = result["blp_mu_invariance"]
    lines += ["", "## BLP の µ 不変性（追補 8: 不変量は zbar）", ""]
    for arm, v in inv.items():
        if v.get("status") != "OK":
            lines.append(f"- {arm}: {v.get('status')}")
            continue
        lines.append(f"- {arm}: max|Δzbar| 中央値 {v['median_max_abs_delta_zbar']:.3e} "
                     f"(許容 1e-6・許容内のユニット割合 {v['median_frac_within_tol']:.3f}) / "
                     f"参考 max|Δs| {v['median_max_abs_delta_s']:.3e} / "
                     f"denom 成長比 {v['median_denom_growth_ratio']:.4f}")
    refl = sanity.get("S_refl", {})
    if refl:
        lines += ["", "## S-refl（**REPORT_ONLY へ降格**・追補 13）", "",
                  "登録時は必須ゲートだったが、実装ではなく本走の主 endpoint と同じ",
                  "物理の主張を検査するゲートだったため 9/2 に REPORT_ONLY へ降格した。",
                  f"- 登録された期待（`BLR` の $\\bar z$ が `BL` と逆向き）を満たしたか: "
                  f"**{refl.get('registered_expectation_met')}**",
                  "- 30k 短走の平均 $\\bar z$ ドリフト（**登録外の診断・転記対象ではない**）: "
                  + ", ".join(f"{k} {v['mean_zbar_drift']:+.5f} (n={v['n_submerged_at_step0']})"
                              for k, v in refl.get("arms", {}).items()),
                  "- 実装側のゲート（S-cross・S-limit ×4・S-bwd・S-proj・S-wd-w・",
                  "  S-pair・S-dose・S-taut・S-ref）はすべて PASS している。"]
    c = result["ceiling"]
    lines += ["", f"腕間ベースライン広がり（1M 窓・対照込み）: {c['spread_dex']:.6g} dex "
              f"(閾値 {c['threshold']}) — flagged={c['flagged']}", "", "## Sanity", ""]
    for key in ("S1_omp", "S_cross", "S_limit_static", "S_bwd", "S_proj", "S_wd_w",
                "S_ref", "S_limit_bwd_reflect", "S_limit_bwd_quad",
                "S_limit_bwd_leaky_proj", "S_limit_wd_w", "S_refl", "S_pair",
                "S_dose", "S_taut", "S6_floor_inherited", "S_CI_degeneracy"):
        v = sanity.get(key, {})
        lines.append(f"- {key}: **{'PASS' if v.get('pass_') else 'FAIL'}**")
    (outdir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Run driver
# ---------------------------------------------------------------------------
def _provenance(cfg_path: Path, cfg: dict, outdir: Path, sanity: dict,
                analysis: dict, elapsed: dict, started: float) -> dict:
    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                           text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--short"], cwd=ROOT,
                                        text=True).splitlines()
    except (OSError, subprocess.CalledProcessError):
        git_hash, dirty = None, []
    names = ("verdict.csv", "summary.md", "layer_stats.csv", "s_distribution.csv",
             "s_contrasts.csv", "phat_hist.csv", "revival.csv", "w_growth.csv",
             "freeze_rates.csv", "config_used.yaml")
    hashes = {n: _sha_file(outdir / n) for n in names if (outdir / n).exists()}
    hashes.update({f"logs/{p.name}": _sha_file(p)
                   for p in sorted((outdir / "logs").glob("*.npz"))})
    hashes.update({f"arm_status/{p.name}": _sha_file(p)
                   for p in sorted((outdir / "arm_status").glob("*.json"))})
    parents = {}
    for run in ("results/gate_dose_0830", "results/bwd_leak_0902"):
        prov = Path(ROOT) / run / "provenance.json"
        parents[run] = (json.loads(prov.read_text(encoding="utf-8")).get("git_hash")
                        if prov.exists() else None)
    return dict(
        experiment=EXPERIMENT, created=time.strftime("%Y-%m-%d %H:%M:%S %z"),
        command=sys.argv, elapsed_sec=round(time.time() - started, 3),
        arm_elapsed_sec=elapsed, cwd=os.getcwd(), python=sys.version,
        platform=platform.platform(), torch=torch.__version__, numpy=np.__version__,
        device=cfg["common"]["device"], git_hash=git_hash, git_dirty=dirty,
        config=str(cfg_path), config_sha256=_sha_file(cfg_path),
        spec=str(Path(ROOT) / cfg["spec"]),
        spec_sha256=_sha_file(Path(ROOT) / cfg["spec"]),
        dose_run="12.16 only", arms_run=list(ARM_ORDER),
        lambda_w=float(cfg["weight_decay_w"]["lambda_w"]),
        blq_scale=dict(a_Q=float(cfg["activation"]["bwd_quad"]["slope"]),
                       raw_z=True,
                       crossover_depth_z=float(cfg["activation"]["bwd_quad"]["crossover_depth_z"])),
        s_log_branch="A", generator_offset=int(cfg["common"]["generator_offset"]),
        baseline_runs=parents, sanity=sanity, analysis=analysis, output_sha256=hashes)


def run(cfg_path: Path, cfg: dict, device: str, outdir: Path, *, smoke: bool,
        analyze_only: bool) -> dict:
    started = time.time()
    outdir.mkdir(parents=True, exist_ok=True)
    require_omp(cfg)
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
    for arm in ARM_ORDER:
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
                       identities=identities, divergences=divergences, elapsed_sec=elapsed)
        (outdir / "smoke_sanity.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        print(f"SMOKE DONE -> {outdir}", flush=True)
        return payload

    sanity = dict(preflight_result)
    sanity.pop("pass_", None)
    result = analyze(cfg, outdir, sanity, elapsed, divergences)
    provenance = _provenance(cfg_path, cfg, outdir, sanity, result, elapsed, started)
    (outdir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    for t, v in result["types"].items():
        print(f"{t}: {v['verdict']}  ({'|'.join(v['state_triple'])})", flush=True)
    print(f"ALL DONE -> {outdir}", flush=True)
    return dict(sanity=sanity, analysis=result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--outdir")
    args = parser.parse_args()
    if sum((args.preflight, args.smoke, args.analyze_only)) > 1:
        parser.error("stage flags are mutually exclusive")
    cfg_path = Path(args.config).resolve()
    cfg = load_config(str(cfg_path))
    device = pick_device(cfg)
    if device != "cpu":
        raise ValueError("phantom_wall is CPU-only")
    stage = ("preflight" if args.preflight else "smoke" if args.smoke
             else "analyze" if args.analyze_only else "run")
    validate_config(cfg, stage=stage)
    if args.preflight:
        preflight(cfg, Path(ROOT) / f"results/_preflight_{EXPERIMENT}")
        return
    outdir = (Path(args.outdir).resolve() if args.outdir
              else Path(ROOT) / f"results/_smoke_{EXPERIMENT}" if args.smoke
              else Path(ROOT) / cfg["output"]["dir"])
    run(cfg_path, cfg, device, outdir, smoke=args.smoke,
        analyze_only=args.analyze_only)


if __name__ == "__main__":
    main()
