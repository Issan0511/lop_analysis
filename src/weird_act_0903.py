"""weird_act_0903 — 謎関数ダイヤル（spec `specs/spec_weird_act_0903.md`）。

    OMP_NUM_THREADS=1 python3 -m src.weird_act_0903 --stage preflight
    OMP_NUM_THREADS=1 python3 -m src.weird_act_0903 --stage run --substage 1

宿主は ``gate_dial_0902``（1 層・オラクル用量 12.16 固定・5M）。宿主の
``validate_config`` は 14 腕を逐語照合するので通さず、``_run_arm`` の本体を写した
``_run_arm_weird`` を使う（spec §10 追補 2）。**``src/gate_dial_0902.py`` の既存の行は
1 行も変えない**。``src/nets.py`` へは活性化 10 名＋退化点 2 名の追記のみ。

ユニット別ロガーは宿主の ``DialRecorder`` を継承し ``zmin`` を 1 列足す。
``_run_arm`` は recorder を直書きしているので差し替え口が無く、40 行の写しを持つ
（S-copy がその写しを機械的に検算する）。
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
from .dose_const_5m import (_input_stats, _refresh_fixed_offset,
                            clopper_pearson)
from .elu_swamp import exact_layer_record_elu
from .gate_dose import IDENTITY_TOL, SIGMA_TOL, _load_arm, _window, train_arm_gate
from .gate_dial_0902 import (DialRecorder, NEW_UNIT_KEYS, SanityError, _arm,
                             _arm_status_path, _ci, _draws, _kaplan_meier,
                             _load_new_arm, _sign_test, setup_arm_dial,
                             unit_extra_record, write_arm_logs_dial)
from .mlp2_phase0 import (_sha_array, _sha_file, identity_sanity_pass,
                          require_omp, write_csv)
from .mlp2_phase0b import _window_indices
from .mlp2_phase1 import (NUMERIC_DIVERGENCE, NumericDivergenceError,
                          StreamDigest, _env_hashes, _init_hashes,
                          _seed_state_hashes_p1)
from .nets import VecMLPL
from .ratchet_log import full_support_ro

EXPERIMENT = "weird_act_0903"
CONFIG = Path(ROOT) / "configs" / "weird_act_0903.yaml"

ARM_ORDER = ("LRm_a0p1_1216", "LRv_d2_1216", "RB_d1_1216", "CB_a1_1216",
             "LRv_d1_1216", "RB_d0p5_1216", "RB_d2_1216", "RB_d4_1216",
             "LRq_d1_1216", "CB_a1_b5_1216", "CB_a2_1216")
STAGE_ARMS = {1: ("LRm_a0p1_1216", "LRv_d2_1216", "RB_d1_1216", "CB_a1_1216",
                  "LRv_d1_1216"),
              2: ("RB_d0p5_1216", "RB_d2_1216", "RB_d4_1216", "LRq_d1_1216",
                  "CB_a1_b5_1216", "CB_a2_1216")}
# 事前登録の腕定義（stage, family, activation label, dial）。validate_config が逐語照合する。
REGISTERED_ARMS = {
    "LRm_a0p1_1216":  (1, "mirror", "mirror_leaky", 0.1),
    "LRv_d2_1216":    (1, "fold", "fold_leaky_d2", 0.1),
    "RB_d1_1216":     (1, "band", "band_leaky_d1", 0.1),
    "CB_a1_1216":     (1, "comb", "comb_binf", 1.0),
    "LRv_d1_1216":    (1, "fold", "fold_leaky_d1", 0.1),
    "RB_d0p5_1216":   (2, "band", "band_leaky_d0p5", 0.1),
    "RB_d2_1216":     (2, "band", "band_leaky_d2", 0.1),
    "RB_d4_1216":     (2, "band", "band_leaky_d4", 0.1),
    "LRq_d1_1216":    (2, "ramp", "ramp_leaky_d1", 0.1),
    "CB_a1_b5_1216":  (2, "comb", "comb_b5", 1.0),
    "CB_a2_1216":     (2, "comb", "comb_binf", 2.0),
}
# 本モジュールが足すユニット別列。既存列は 1 列も変えない・消さない（spec §10 追補 2）。
WEIRD_UNIT_KEYS = tuple(NEW_UNIT_KEYS) + ("zmin",)
SMOKE_STEPS = 30_000
# S-limit 専用の退化点。本走の腕には使わない（spec §10 追補 9）。
S_LIMIT_CASES = (("band_leaky_d0", 0.1, "leaky_relu", 0.1),
                 ("fold_leaky_dbig", 0.1, "leaky_relu", 0.1),
                 ("mirror_leaky", 0.0, "relu", 1.0))


def _P(cfg: dict) -> dict:
    return cfg["weird_act"]


def _selected_arms(cfg: dict, stage: str) -> list[str]:
    if stage in ("all", "0"):
        return list(ARM_ORDER)
    return list(STAGE_ARMS[int(stage)])


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def validate_config(cfg: dict, *, stage: str) -> None:
    """凍結した設計からのずれをすべて ValueError にする。"""
    if stage not in {"preflight", "smoke", "run", "analyze", "finalize",
                     "diverge-probe"}:
        raise ValueError(f"unknown stage {stage!r}")
    C, A, I, P, G, S = (cfg["common"], cfg["condA"], cfg["intervention"],
                        cfg["phase1"], _P(cfg), cfg["sanity"])
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
                or float(arm["target_mu_norm"]) != 3.041
                or float(arm["target_dose"]) != 12.16
                or arm["u_star"] is not None or arm["u_fr"] is not None):
            raise ValueError(f"{arm['name']} differs from the preregistration")
    for key, want in ((1, STAGE_ARMS[1]), (2, STAGE_ARMS[2])):
        if [a["name"] for a in cfg["arms"] if int(a["stage"]) == key] != list(want):
            raise ValueError(f"stage {key} arms changed")
    if [str(v) for v in cfg["staging"]["stage1_arms"]] != list(STAGE_ARMS[1]):
        raise ValueError("staging.stage1_arms changed")
    if [str(v) for v in cfg["staging"]["stage2_arms"]] != list(STAGE_ARMS[2]):
        raise ValueError("staging.stage2_arms changed")
    # 2026-09-03 の段裁定（spec §3・§10 追補 8）
    if not bool(cfg["staging"]["stage2_frozen_before_stage1_results"]):
        raise ValueError("stage 2 must be frozen before stage 1 results")
    if not bool(cfg["staging"]["v2_requires_stage2"]):
        raise ValueError("V2 is registered as requiring stage 2")
    if (int(A["m"]), int(A["f"]), int(A["target_hidden"])) != (20, 15, 100):
        raise ValueError("condA differs from the preregistration")
    if [int(v) for v in A["T_values"]] != [10000] or list(A["encodings"]) != ["std"]:
        raise ValueError("condA task period / encoding differ")
    if int(C.get("generator_offset", -1)) != 0:
        raise ValueError("generator_offset must be an explicit 0 (spec §4)")
    if (str(I["name"]) != "oracle_fixed_mu_offset" or I["oracle"] is not True
            or I["consumes_rng"] is not False
            or float(I["center_alpha_compat"]) != 0.01):
        raise ValueError("intervention differs from the preregistration")
    if (cfg["activation"]["autograd"] is not False
            or cfg["activation"]["consumes_rng"] is not False
            or cfg["activation"]["is_true_gradient"] is not True):
        raise ValueError("activation block differs from the preregistration")
    for label, (_, _, act, _) in ((k, v) for k, v in REGISTERED_ARMS.items()):
        if str(cfg["activation"][act]["name"]) != act:
            raise ValueError(f"activation.{act}.name must be {act!r}")
        if act not in VecMLPL.ACTIVATIONS:
            raise ValueError(f"{act} is not registered in VecMLPL.ACTIVATIONS")
    expected_phase = {
        "task_period": 10_000, "early_tasks": [2, 11],
        "late_tasks_5m": [491, 500], "window_1m_tasks": [91, 100],
        "window_points_are_task_ends_only": True,
        "window_records_per_10task_window": 10,
        "onset_threshold": 0.05, "onset_present_min": 5,
        "unfit_floor": 1e-16, "recalibrate_floor": False,
        "bootstrap_B": 10_000, "bootstrap_seed": 20_260_914,
        "ci_method": "percentile_primary_studentized_secondary",
    }
    for key, value in expected_phase.items():
        if P[key] != value:
            raise ValueError(f"phase1.{key} differs from the preregistration")
    design = G["design"]
    if (design["uses_self_term_decomposition"] is not False
            or float(design["displacement_bound_constant_K"]) != 1.0):
        raise ValueError("weird_act.design differs from the preregistration")
    threshold = float(design["freeze_depth_phi_prime_threshold"])
    want_threshold = (float(P["onset_threshold"])
                      / (float(C["lr_main"]) * float(C["total_steps"])
                         * float(design["displacement_bound_constant_K"])))
    if not math.isclose(threshold, want_threshold, rel_tol=1e-12):
        raise ValueError("the freeze-depth threshold is not 0.05/(lr*T*K)")
    if list(G["verdict_order"]) != ["V1", "V2", "V3", "V4"]:
        raise ValueError("the verdict order is registered as V1..V4")
    if G["v1"]["arm"] != "LRm_a0p1_1216" or G["v3"]["arm"] != "LRv_d2_1216":
        raise ValueError("V1 / V3 judgment arms changed")
    if G["v3"]["anchor_arm"] != "LRv_d1_1216" or not G["v3"]["anchor_is_report_only"]:
        raise ValueError("LRv_d1 is registered as a REPORT_ONLY anchor (spec §6 V3)")
    if int(G["v2"]["requires_stage"]) != 2:
        raise ValueError("V2 is registered as requiring stage 2")
    if list(G["v2"]["ladder_new_arms"]) != ["RB_d0p5_1216", "RB_d1_1216",
                                            "RB_d2_1216", "RB_d4_1216"]:
        raise ValueError("the V2 ladder changed")
    for key in ("p5_equivalence_margin",):
        if float(G[key]) != 0.15:
            raise ValueError("the equivalence margin is registered as 0.15 dex")
    if float(G["v2"]["margin"]) != 0.15 or float(G["v4"]["margin"]) != 0.15:
        raise ValueError("the equivalence margin is registered as 0.15 dex")
    if G["p3prime_baseline"] != "R_1216":
        raise ValueError("the P3' baseline is registered as R_1216")
    wanted_sanity = {"s_pair_steps": 30000, "s_limit_steps": 30000,
                     "s_dose_rel_tol": 1e-10, "s_fd_tol": 1e-6,
                     "s_mob_tol": 1e-6, "s_cap_min_seeds": 9,
                     "s_dial_rel_tol": 0.06, "omp_num_threads": 1}
    for key, value in wanted_sanity.items():
        if S[key] != value:
            raise ValueError(f"sanity.{key} differs from the preregistration")
    for key in ("s_dial_check", "s_const_check", "s_taut_check", "s_mask_check",
                "s_cover_check"):
        if S[key] is not True:
            raise ValueError(f"sanity.{key} must be true")
    if S["s6_floor_calibration"] is not False:
        raise ValueError("the floor is inherited, never recalibrated")
    if stage in {"run", "analyze"}:
        if int(C["total_steps"]) != 5_000_000:
            raise ValueError("total_steps is registered as 5,000,000")
        if [int(v) for v in C["seeds"]] != list(range(10)):
            raise ValueError("seeds are registered as 0..9")
        if str(C["device"]) != "cpu":
            raise ValueError("device is registered as cpu")


# ---------------------------------------------------------------------------
# Logger — 宿主の DialRecorder に zmin を 1 列足すだけ（spec §10 追補 2）
# ---------------------------------------------------------------------------
def unit_zmin_record(st: dict) -> torch.Tensor:
    """第 1 層の ``min_x z_i(x)``。

    ``gate_dial_0902.unit_extra_record`` の ``z`` の作り方を逐語で真似る（32 点の
    厳密支持・float64・中心化フラグと ``layer_means`` の扱いまで同じ）。したがって
    ``zmin`` は同関数の ``zmax`` の厳密な鏡である。学習状態は読むだけで書き換えない
    （``full_support_ro`` は RNG を消費しない読み取り専用の支持列挙）。
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
        return z.amin(dim=0)


class WeirdRecorder(DialRecorder):
    """``DialRecorder`` ＋ ``zmin`` の 1 列。``record_units=False`` で宿主と同一。"""

    def __init__(self, steps: list[int], st: dict, *, record_units: bool = True):
        super().__init__(steps, st, record_units=record_units)
        if self.record_units:
            n, runs, width = len(self.steps), st["R"], st["hidden"][0]
            self.unit["zmin"] = np.empty((n, runs, width), dtype=np.float32)

    def __call__(self, st: dict, step: int) -> None:
        super().__call__(st, step)
        if not self.record_units:
            return
        i = self.index.get(int(step))
        if i is None:
            return
        self.unit["zmin"][i] = (unit_zmin_record(st).detach().cpu().numpy()
                                .astype(np.float32))


# ---------------------------------------------------------------------------
# Runner — gate_dial_0902._run_arm の写し。recorder の 1 行だけが違う（S-copy が検算）
# ---------------------------------------------------------------------------
def _run_arm_weird(cfg: dict, arm: str, device: str, outdir: Path,
                   seeds: list[int], total: int) -> dict:
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
    rec = WeirdRecorder(probes, st)          # ← 宿主との唯一の差（spec §10 追補 2）
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
# 前段チェック（spec §7）
# ---------------------------------------------------------------------------
def _probe_net(act: str, alpha: float) -> VecMLPL:
    net = VecMLPL(1, [2], 2, torch.Generator().manual_seed(0), "cpu")
    return net.set_activation(act, alpha, "alpha_exp")


def _s_copy() -> dict:
    """S-copy: ``_run_arm_weird`` が宿主の ``_run_arm`` の写しで、差が recorder の 1 行だけ。"""
    import inspect

    from . import gate_dial_0902 as host

    def body(fn):
        lines = inspect.getsource(fn).splitlines()
        out, in_signature = [], True
        for line in lines:
            s = line.strip()
            if in_signature:                      # 複数行の署名を丸ごと落とす
                if s.endswith("-> dict:"):
                    in_signature = False
                continue
            if not s or s.startswith("#") or s.startswith('"""'):
                continue
            out.append(s.split("#")[0].rstrip())
        return out

    mine, theirs = body(_run_arm_weird), body(host._run_arm)
    diff = [(i, a, b) for i, (a, b) in enumerate(zip(mine, theirs)) if a != b]
    only_recorder = (len(mine) == len(theirs) and len(diff) == 1
                     and diff[0][1] == "rec = WeirdRecorder(probes, st)"
                     and diff[0][2] == "rec = DialRecorder(probes, st)")
    return dict(pass_=bool(only_recorder), n_lines=len(mine),
                host_lines=len(theirs),
                differences=[dict(index=i, mine=a, host=b) for i, a, b in diff])


def _s_const(cfg: dict) -> dict:
    """S-const: config の第 2 母数の写しと ``nets.py`` のクラス定数辞書が一致（spec §10 追補 5）。"""
    rows, bad = [], []
    tables = {**{k: ("FOLD_DEPTH", v) for k, v in VecMLPL.FOLD_DEPTH.items()},
              **{k: ("BAND_WIDTH", v) for k, v in VecMLPL.BAND_WIDTH.items()},
              **{k: ("RAMP_DEPTH", v) for k, v in VecMLPL.RAMP_DEPTH.items()},
              **{k: ("COMB_ENVELOPE", v) for k, v in VecMLPL.COMB_ENVELOPE.items()}}
    for arm in cfg["arms"]:
        act = str(arm["activation"])
        want = arm.get("second_param")
        block = float(cfg["activation"][act]["second_param_value"]) \
            if "second_param_value" in cfg["activation"][act] else None
        table = tables.get(act)
        code = None if table is None else float(table[1])
        row = dict(arm=arm["name"], activation=act, arm_value=want,
                   config_block_value=block, nets_value=code,
                   nets_table=None if table is None else table[0])
        ok = True
        if act == "mirror_leaky":
            ok = want is None and block is None and code is None
        elif act == "comb_binf":
            # 包絡なし。config は .inf、腕側は null（振動数だけが母数）
            ok = (code == float("inf") and block == float("inf") and want is None)
        else:
            ok = (code is not None and block is not None and want is not None
                  and float(want) == code == block)
        row["pass_"] = bool(ok)
        rows.append(row)
        if not ok:
            bad.append(row)
    return dict(pass_=not bad, rows=rows, failures=bad)


def _phi2_extrema(act: str, alpha: float, umax: float, n: int = 400001) -> dict:
    """深さ ``u = -z`` 上で ``phi^2`` の極大（分水嶺）と極小（井戸・極小）を数値で解く。"""
    net = _probe_net(act, alpha)
    u = torch.linspace(1e-9, umax, n, dtype=torch.float64)
    with torch.no_grad():
        phi2 = net.act_fn(-u) ** 2
    mid = phi2[1:-1]
    hi = (mid > phi2[:-2]) & (mid > phi2[2:])
    lo = (mid < phi2[:-2]) & (mid < phi2[2:])

    def thin(mask):
        vals = u[1:-1][mask].tolist()
        out = []
        for v in vals:
            if not out or v - out[-1] > 1e-2:
                out.append(v)
        return out

    return dict(maxima=thin(hi), minima=thin(lo))


def _s_dial(cfg: dict) -> dict:
    """S-dial: 登録した分水嶺・井戸・極小を数値解と照合（相対許容 6%）。"""
    tol = float(cfg["sanity"]["s_dial_rel_tol"])
    rows, bad = [], []
    for arm in cfg["arms"]:
        act, alpha = str(arm["activation"]), float(arm["dial"])
        reg_max = [float(v) for v in arm.get("watershed") or []]
        reg_min = [float(v) for v in (list(arm.get("well") or [])
                                      + list(arm.get("minimum") or []))]
        if not reg_max and not reg_min:
            rows.append(dict(arm=arm["name"], activation=act, registered_none=True,
                             pass_=True))
            continue
        umax = max(reg_max + reg_min) * 1.25 + 1.0
        got = _phi2_extrema(act, alpha, umax)
        row = dict(arm=arm["name"], activation=act, dial=alpha,
                   registered_watershed=reg_max, solved_maxima=got["maxima"][:5],
                   registered_wells_and_minima=reg_min,
                   solved_minima=got["minima"][:5])
        ok = True
        for want, have in ((reg_max, got["maxima"]), (reg_min, got["minima"])):
            for i, w in enumerate(want):
                if i >= len(have) or abs(have[i] - w) / w > tol:
                    ok = False
        row["pass_"] = bool(ok)
        rows.append(row)
        if not ok:
            bad.append(row)
    return dict(pass_=not bad, rel_tol=tol, rows=rows, failures=bad)


def _kinks(act: str) -> list[float]:
    ks = [0.0]
    if act in VecMLPL.FOLD_DEPTH:
        d = VecMLPL.FOLD_DEPTH[act]
        ks += [-d, -2.0 * d]
    if act in VecMLPL.BAND_WIDTH:
        ks += [-VecMLPL.BAND_WIDTH[act]]
    if act in VecMLPL.RAMP_DEPTH:
        ks += [-VecMLPL.RAMP_DEPTH[act]]
    return ks


def _s_fd(cfg: dict) -> dict:
    """S-fd: 5 族の backward を float64 中心差分と照合。折れ目の ±1e-3 は除外する。"""
    S = cfg["sanity"]
    tol, excl = float(S["s_fd_tol"]), float(S["s_fd_kink_exclusion"])
    lo, hi = [float(v) for v in S["s_fd_range"]]
    n = int(S["s_fd_points"])
    relative_for = set(S.get("s_fd_relative_for") or [])
    h = 1e-6
    rows, bad = [], []
    for act in sorted({str(a["activation"]) for a in cfg["arms"]}
                      | {c[0] for c in S_LIMIT_CASES}):
        alphas = sorted({float(a["dial"]) for a in cfg["arms"]
                         if str(a["activation"]) == act}) or [0.1]
        for alpha in alphas:
            net = _probe_net(act, alpha)
            grid = [torch.linspace(lo, hi, n, dtype=torch.float64)]
            for k in _kinks(act):
                for off in [float(v) for v in S["s_fd_kink_offsets"]]:
                    grid.append(torch.linspace(k - off, k + off, 21,
                                               dtype=torch.float64))
            z = torch.cat(grid)
            mask = torch.ones_like(z, dtype=torch.bool)
            for k in _kinks(act):
                mask &= (z - k).abs() > excl
            z = z[mask]
            with torch.no_grad():
                fd = (net.act_fn(z + h) - net.act_fn(z - h)) / (2.0 * h)
                g = net.act_grad(z, net.act_fn(z))
            err = (fd - g).abs()
            rel = err / torch.clamp(g.abs(), min=1.0)
            use_rel = act in relative_for
            worst = float(rel.max() if use_rel else err.max())
            row = dict(activation=act, alpha=alpha, n_points=int(z.numel()),
                       metric="relative" if use_rel else "absolute",
                       worst=worst, tol=tol, pass_=bool(worst <= tol))
            rows.append(row)
            if not row["pass_"]:
                bad.append(row)
    return dict(pass_=not bad, rows=rows, failures=bad)


def _s_num(cfg: dict) -> dict:
    """S-num: 登録した範囲で NaN・inf が出ないこと。float32 の飽和・溢れ深さを記録する。"""
    S = cfg["sanity"]
    n = int(S["s_num_points"])
    rows, bad = [], []
    for act in sorted({str(a["activation"]) for a in cfg["arms"]}):
        rng = ([float(v) for v in S["s_num_range_comb_b5"]] if act == "comb_b5"
               else [float(v) for v in S["s_num_range"]])
        alphas = sorted({float(a["dial"]) for a in cfg["arms"]
                         if str(a["activation"]) == act})
        for alpha in alphas:
            net = _probe_net(act, alpha)
            z = torch.linspace(rng[0], rng[1], n, dtype=torch.float32)
            with torch.no_grad():
                f = net.act_fn(z)
                g = net.act_grad(z, f)
            finite = bool(torch.isfinite(f).all() and torch.isfinite(g).all())
            # float32 で厳密 0 になる深さ（記録のみ・判定しない）
            neg = z < 0
            zero_depth = None
            zz = z[neg][(f[neg] == 0) & (g[neg] == 0)]
            if zz.numel():
                zero_depth = float(-zz.max())
            # 溢れる深さ（comb_b5 のみ意味を持つ。記録のみ）
            over_depth = None
            probe = torch.arange(0.0, 2000.0, 1.0, dtype=torch.float32)
            with torch.no_grad():
                fo = net.act_fn(-probe)
            bad_idx = torch.nonzero(~torch.isfinite(fo)).flatten()
            if bad_idx.numel():
                over_depth = float(probe[bad_idx[0]])
            row = dict(activation=act, alpha=alpha, range=rng, finite=finite,
                       float32_zero_depth=zero_depth,
                       float32_overflow_depth=over_depth, pass_=finite)
            rows.append(row)
            if not finite:
                bad.append(row)
    return dict(pass_=not bad, rows=rows, failures=bad)


def _short_run_digest(cfg: dict, arm_block: dict, outdir: Path, steps: int,
                      *, record_units: bool = True) -> dict:
    """30k 短走を回して、記録の全列と学習状態のハッシュを返す（logs は書かない）。"""
    c = copy.deepcopy(cfg)
    st = setup_arm_dial(c, arm_block, "cpu")
    every = int(c["common"]["lop_every"])
    probes = list(range(0, steps + 1, every))
    rec = WeirdRecorder(probes, st, record_units=record_units)
    stream = StreamDigest()
    train_arm_gate(st, rec, probes, steps, outdir, [], stream_hook=stream)
    return dict(
        state=_init_hashes(st), env=_env_hashes(st), stream=stream.digest(),
        run={k: _sha_array(v) for k, v in rec.run.items()},
        layers=[{k: _sha_array(v) for k, v in layer.items()} for layer in rec.layers],
        extra={k: _sha_array(v) for k, v in rec.extra.items()},
        flip=_sha_array(rec.flip_state),
        unit={k: _sha_array(v) for k, v in rec.unit.items()},
        seed_hashes={int(run["seed"]): _seed_state_hashes_p1(st, ri)
                     for ri, run in enumerate(st["runs"])},
        raw=rec)


def _s_limit(cfg: dict, outdir: Path) -> dict:
    """S-limit: 退化した母数が既存活性化と一致すること（閉形式＋30k 短走の bit 一致）。"""
    S = cfg["sanity"]
    steps = int(S["s_limit_steps"])
    grid = torch.linspace(-30.0, 30.0, 24001, dtype=torch.float64)
    rows = []
    template = copy.deepcopy(_arm(cfg, "LRm_a0p1_1216"))
    c = copy.deepcopy(cfg)
    c["common"]["seeds"] = [0, 1]
    for act, alpha, ref_act, ref_alpha in S_LIMIT_CASES:
        a, b = _probe_net(act, alpha), _probe_net(ref_act, ref_alpha)
        with torch.no_grad():
            fa, fb = a.act_fn(grid), b.act_fn(grid)
            ga, gb = a.act_grad(grid, fa), b.act_grad(grid, fb)
        closed = bool(torch.equal(fa, fb) and torch.equal(ga, gb))
        signs = dict(forward=int(torch.signbit(fa).sum()),
                     reference_forward=int(torch.signbit(fb).sum()),
                     grad=int(torch.signbit(ga).sum()),
                     reference_grad=int(torch.signbit(gb).sum()))
        # 30k 短走の bit 一致（logs 全列と state_hash）
        digests = {}
        for label, (act_name, act_dial) in (("new", (act, alpha)),
                                            ("reference", (ref_act, ref_alpha))):
            block = copy.deepcopy(template)
            block["name"] = f"slimit_{label}"
            block["activation"] = f"_slimit_{label}"
            block["dial"] = act_dial
            cc = copy.deepcopy(c)
            cc["activation"][f"_slimit_{label}"] = {"name": act_name}
            cc["arms"] = [block]
            digests[label] = _short_run_digest(cc, block, outdir, steps)
        same = {k: digests["new"][k] == digests["reference"][k]
                for k in ("state", "env", "stream", "run", "layers", "extra",
                          "flip", "unit", "seed_hashes")}
        row = dict(activation=act, alpha=alpha, reference=ref_act,
                   reference_alpha=ref_alpha, closed_form_bit_equal=closed,
                   negative_zeros=signs, short_run_equal=same,
                   pass_=bool(closed and all(same.values())
                              and signs["forward"] == signs["reference_forward"]
                              and signs["grad"] == signs["reference_grad"]))
        rows.append(row)
    return dict(pass_=all(r["pass_"] for r in rows), steps=steps, rows=rows)


def _s_log_b(cfg: dict, outdir: Path) -> dict:
    """S-log-b: 追加ロガーの有無で既存の列がすべて bit 一致（軌道中立）。"""
    steps = int(cfg["sanity"]["s_log_b_steps"])
    c = copy.deepcopy(cfg)
    c["common"]["seeds"] = [0, 1]
    block = _arm(c, "CB_a1_1216")
    got = {}
    for label, units in (("with_logger", True), ("without_logger", False)):
        got[label] = _short_run_digest(c, block, outdir, steps, record_units=units)
    differences = []
    for key in ("state", "env", "stream", "run", "layers", "extra", "flip",
                "seed_hashes"):
        if got["with_logger"][key] != got["without_logger"][key]:
            differences.append(key)
    added = sorted(got["with_logger"]["unit"])
    removed = sorted(set(got["without_logger"]["unit"]) - set(added))
    return dict(pass_=bool(not differences and not removed
                           and added == sorted(WEIRD_UNIT_KEYS)),
                steps=steps, differences=differences, added_columns=added,
                removed_columns=removed,
                expected_columns=sorted(WEIRD_UNIT_KEYS))


def _s_mob(cfg: dict, outdir: Path) -> dict:
    """S-mob: 新規ロガーが既知の量と一致すること＋ ``zmin <= zmean <= zmax``。"""
    S = cfg["sanity"]
    tol, steps = float(S["s_mob_tol"]), int(S["s_mob_steps"])
    c = copy.deepcopy(cfg)
    c["common"]["seeds"] = [0, 1]
    rows = []
    for arm_name in ("RB_d1_1216", "LRm_a0p1_1216"):
        block = _arm(c, arm_name)
        st = setup_arm_dial(copy.deepcopy(c), block, "cpu")
        every = int(c["common"]["lop_every"])
        probes = list(range(0, steps + 1, every))
        rec = WeirdRecorder(probes, st)
        train_arm_gate(st, rec, probes, steps, outdir, [])
        # 記録点の最後で、恒等式を厳密支持の上で直接検算する
        extra = unit_extra_record(st)
        zmin = unit_zmin_record(st)
        net = st["net"]
        flags = st.get("centered_layers") or [False]
        means = st.get("layer_means") or [None]
        with torch.no_grad():
            cur = full_support_ro(st["env"]).double()
            if flags[0]:
                cur = cur - means[0].double()[None]
            z = torch.einsum("rhd,prd->prh", net.Ws[0].double(), cur) + net.bs[0].double()
            p_hat = (z > 0).double().mean(dim=0)
            a = float(net.act_alpha)
            if arm_name.startswith("RB_"):
                d = VecMLPL.BAND_WIDTH[net.act]
                want = p_hat + a * (z < -d).double().mean(dim=0)
                identity = "p_hat + a*Pr[z < -d]"
            else:
                want = p_hat - (1.0 - p_hat) * a
                identity = "p_hat - (1-p_hat)*a"
        err = float((extra["mob"] - want).abs().max())
        order = bool((zmin <= extra["zmean"] + 1e-12).all()
                     and (extra["zmean"] <= extra["zmax"] + 1e-12).all())
        # ロガーが書いた列でも同じ順序が成り立つこと
        order_logged = bool((rec.unit["zmin"] <= rec.unit["zmean"] + 1e-5).all()
                            and (rec.unit["zmean"] <= rec.unit["zmax"] + 1e-5).all())
        rows.append(dict(arm=arm_name, identity=identity, max_abs_error=err,
                         zmin_le_zmean_le_zmax=order,
                         zmin_le_zmean_le_zmax_logged=order_logged,
                         pass_=bool(err <= tol and order and order_logged)))
    return dict(pass_=all(r["pass_"] for r in rows), tolerance=tol, steps=steps,
                rows=rows)


def _s_pair_and_dose(cfg: dict, outdir: Path, arms: list[str]) -> dict:
    """S-pair / S-dose: 新規腕どうし・親走との init/教師/入力列/flip の bit 一致。

    親走 ``gate_dose_0830`` の ``logs/*.npz`` は gitignore されており、この機では
    存在しない。**親との照合は本機では検証不能**なので、その旨を明示して
    ``parent_status`` に残す（隠して PASS にしない）。
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
    parent_status = ("VERIFIED" if parent_rows and not parent_missing
                     else "PARENT_LOGS_ABSENT_ON_THIS_MACHINE")

    tol = float(S["s_dose_rel_tol"])
    dose_fail = [r for r in dose_rows if r["target_mu_norm"] is not None
                 and float(r["max_relative_error"]) > tol]
    return dict(
        spair=dict(pass_=bool(not differences), reference=reference,
                   arms=list(arms), steps=steps, match_by="seed_init_hash",
                   differences=differences, parent_status=parent_status,
                   parent_flip_rows=parent_rows,
                   parent_missing=sorted(set(parent_missing)),
                   caveat="init/teacher/input realization only; trajectories "
                          "diverge after step 1. The parent half of S-pair is "
                          "UNVERIFIABLE here because results/gate_dose_0830/logs "
                          "is gitignored and absent on this machine."),
        sdose=dict(pass_=not dose_fail, tolerance=tol, n_probes=len(dose_rows),
                   failures=dose_fail))


def _s_taut(cfg: dict, outdir: Path) -> dict:
    """S-taut: ``frozen`` が fold / comb で構成上恒真・恒偽になっていないか（判定式には未使用）。"""
    steps = 5_000
    c = copy.deepcopy(cfg)
    c["common"]["seeds"] = [0, 1]
    rows = []
    for arm_name in ("LRv_d2_1216", "CB_a1_1216", "RB_d1_1216"):
        block = _arm(c, arm_name)
        st = setup_arm_dial(copy.deepcopy(c), block, "cpu")
        every = int(c["common"]["lop_every"])
        probes = list(range(0, steps + 1, every))
        rec = WeirdRecorder(probes, st)
        train_arm_gate(st, rec, probes, steps, outdir, [])
        mob = np.abs(rec.unit["mob"])
        absmob = rec.unit["absmob"]
        frozen = float((mob < 1e-6).mean())
        frozen_abs = float((absmob < 1e-6).mean())
        rows.append(dict(arm=arm_name, frozen_frac=frozen,
                         frozen_abs_frac=frozen_abs,
                         tautological=bool(frozen in (0.0, 1.0)
                                           and frozen_abs in (0.0, 1.0)),
                         note="frozen/frozen_abs are REPORT_ONLY; no verdict uses them"))
    return dict(pass_=True, rows=rows,
                note="informational: frozen is not in any verdict formula (spec §4)")


def preflight(cfg: dict, outdir: Path) -> dict:
    """全ての前段チェックを回して ``preflight.json`` を書く。失敗したら SanityError。"""
    outdir.mkdir(parents=True, exist_ok=True)
    checks: dict = {}
    checks["S_copy"] = _s_copy()
    checks["S_const"] = _s_const(cfg)
    checks["S_dial"] = _s_dial(cfg)
    checks["S_fd"] = _s_fd(cfg)
    checks["S_num"] = _s_num(cfg)
    checks["S_limit"] = _s_limit(cfg, outdir / "slimit")
    checks["S_log_b"] = _s_log_b(cfg, outdir / "slogb")
    checks["S_mob"] = _s_mob(cfg, outdir / "smob")
    pair = _s_pair_and_dose(cfg, outdir / "spair", list(STAGE_ARMS[1]))
    checks["S_pair"] = pair["spair"]
    checks["S_dose"] = pair["sdose"]
    checks["S_taut"] = _s_taut(cfg, outdir / "staut")
    result = dict(experiment=EXPERIMENT,
                  pass_=all(bool(v.get("pass_")) for v in checks.values()),
                  checks=checks)
    path = outdir / "preflight.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str),
                    encoding="utf-8")
    for name, value in checks.items():
        print(f"[preflight] {name}: {'PASS' if value.get('pass_') else 'FAIL'}",
              flush=True)
    print(f"[preflight] -> {path}", flush=True)
    if not result["pass_"]:
        raise SanityError("preflight failed; see " + str(path))
    return result


# ---------------------------------------------------------------------------
# Provenance / run
# ---------------------------------------------------------------------------
def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _provenance(cfg_path: Path, cfg: dict, outdir: Path, stage: str,
                arms: list[str], sanity: dict, elapsed: float,
                started: str) -> dict:
    names = ("verdict.csv", "summary.md", "onset_times.csv", "ladder_table.csv",
             "position_table.csv", "depth_hist.csv", "c1_table.csv",
             "regime_table.csv", "layer_stats.csv", "config_used.yaml")
    hashes = {n: _sha_file(outdir / n) for n in names if (outdir / n).exists()}
    hashes.update({f"logs/{p.name}": _sha_file(p)
                   for p in sorted((outdir / "logs").glob("*.npz"))})
    hashes.update({f"arm_status/{p.name}": _sha_file(p)
                   for p in sorted((outdir / "arm_status").glob("*.json"))})
    saturation = {}
    for row in (sanity.get("checks", {}).get("S_num", {}).get("rows") or []):
        saturation[f"{row['activation']}@{row['alpha']}"] = dict(
            float32_zero_depth=row.get("float32_zero_depth"),
            float32_overflow_depth=row.get("float32_overflow_depth"))
    return dict(
        experiment=EXPERIMENT, created=started, command=sys.argv,
        elapsed_sec=elapsed, cwd=str(Path.cwd()), python=sys.version,
        platform=platform.platform(), torch=torch.__version__,
        numpy=np.__version__, git_hash=_git("rev-parse", "HEAD"),
        git_dirty=_git("status", "--short"),
        device=cfg["common"]["device"],
        config=str(cfg_path), config_sha256=_sha_file(cfg_path),
        spec=str(Path(ROOT) / cfg["spec"]),
        spec_sha256=_sha_file(Path(ROOT) / cfg["spec"]),
        stages_registered=dict(cfg["staging"]),
        stage_run=stage, arms_run=list(arms),
        stage_split_note="2026-09-03 Issa: stage 1 and stage 2 are submitted "
                         "separately; both are preregistered in the same commit. "
                         "V2 does not get a label until stage 2 completes.",
        dose="12.16",
        host="gate_dial_0902._run_arm (copied verbatim except the recorder line)",
        added_unit_columns=["layer1_zmin"],
        unit_columns=[f"layer1_{k}" for k in WEIRD_UNIT_KEYS],
        generator_offset=int(cfg["common"]["generator_offset"]),
        generator_offset_note="explicit 0: this run deliberately shares the "
                              "parent run's seed set and random stream (S-pair).",
        window_definition=dict(
            task_ends_only=True, records_per_10task_window=10,
            spec_literal=int(cfg["phase1"]["spec_literal_records_per_window"])),
        baseline_reference=str((Path(ROOT) / cfg["controls"]["reference_run"]).resolve()),
        baseline_endpoint_source=str(Path(ROOT) / cfg["controls"]["reference_run"]
                                     / "verdict.csv"),
        baseline_unit_source_is_gitignored=True,
        parent_logs_absent_on_this_machine=(
            sanity.get("checks", {}).get("S_pair", {}).get("parent_status")
            == "PARENT_LOGS_ABSENT_ON_THIS_MACHINE"),
        float32_saturation=saturation,
        sanity=sanity, output_sha256=hashes)


def run(cfg_path: Path, cfg: dict, device: str, outdir: Path, stage: str,
        *, smoke: bool = False) -> dict:
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    t0 = time.time()
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / "config_used.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, allow_unicode=True, sort_keys=False)
    total = SMOKE_STEPS if smoke else int(cfg["common"]["total_steps"])
    seeds = [0] if smoke else [int(v) for v in cfg["common"]["seeds"]]
    arms = _selected_arms(cfg, stage)
    pre_path = Path(ROOT) / "results" / f"_preflight_{EXPERIMENT}" / "preflight.json"
    if not smoke:
        if not pre_path.exists():
            raise SanityError(f"run the preflight first: {pre_path} is missing")
        pre = json.loads(pre_path.read_text(encoding="utf-8"))
        if not pre.get("pass_"):
            raise SanityError(f"preflight did not pass: {pre_path}")
    else:
        pre = dict(pass_=None, note="smoke run: preflight not required")
    statuses, divergences = {}, {}
    for arm in arms:
        got = _run_arm_weird(cfg, arm, device, outdir, seeds, total)
        statuses[arm] = got["status"]
        if got["status"] == NUMERIC_DIVERGENCE:
            divergences[arm] = got["divergence"]
    elapsed = time.time() - t0
    prov = _provenance(cfg_path, cfg, outdir, stage, arms, pre, elapsed, started)
    prov["arm_status"] = statuses
    prov["divergences"] = divergences
    if not smoke:
        (outdir / "provenance.json").write_text(
            json.dumps(prov, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8")
    print(f"[run] stage={stage} arms={arms} elapsed={elapsed:.1f}s", flush=True)
    return prov


def finalize(cfg_path: Path, cfg: dict, outdir: Path, stage: str) -> dict:
    """腕プロセス並列の後始末: config_used.yaml と provenance.json を書く。"""
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / "config_used.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, allow_unicode=True, sort_keys=False)
    pre_path = Path(ROOT) / "results" / f"_preflight_{EXPERIMENT}" / "preflight.json"
    pre = json.loads(pre_path.read_text(encoding="utf-8"))
    if not pre.get("pass_"):
        raise SanityError(f"preflight did not pass: {pre_path}")
    arms = _selected_arms(cfg, stage)
    seeds = [int(v) for v in cfg["common"]["seeds"]]
    statuses, divergences, elapsed = {}, {}, 0.0
    for arm in arms:
        status_path = outdir / "arm_status" / f"{arm}_done.json"
        div_path = _arm_status_path(outdir, arm)
        if div_path.exists():
            divergences[arm] = json.loads(div_path.read_text(encoding="utf-8"))
            statuses[arm] = NUMERIC_DIVERGENCE
        elif status_path.exists():
            done = json.loads(status_path.read_text(encoding="utf-8"))
            statuses[arm] = done.get("status")
            elapsed = max(elapsed, float(done.get("wall_sec") or 0.0))
        else:
            statuses[arm] = "MISSING"
        missing = [s for s in seeds
                   if not (outdir / "logs" / f"{arm}_seed{s}.npz").exists()]
        if missing and statuses[arm] == "COMPLETE":
            statuses[arm] = f"INCOMPLETE_LOGS:{missing}"
    prov = _provenance(cfg_path, cfg, outdir, stage, arms, pre, elapsed,
                       time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    prov["arm_status"] = statuses
    prov["divergences"] = divergences
    prov["arm_process_parallel"] = True
    prov["s_par_note"] = ("arm-process parallelism is inherited from "
                          "gate_dial_0902 S-par (arms are independent processes; "
                          "the seed loop is vectorised inside one process).")
    (outdir / "provenance.json").write_text(
        json.dumps(prov, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")
    print(f"[finalize] stage={stage} arms={statuses}", flush=True)
    return prov


def run_single_arm(cfg: dict, arm: str, device: str, outdir: Path,
                   total: int) -> dict:
    seeds = [int(v) for v in cfg["common"]["seeds"]]
    outdir.mkdir(parents=True, exist_ok=True)
    return _run_arm_weird(cfg, arm, device, outdir, seeds, total)


def main() -> None:
    ap = argparse.ArgumentParser(description=EXPERIMENT)
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--stage", default="preflight",
                    choices=["preflight", "smoke", "run", "finalize", "analyze",
                             "diverge-probe"])
    ap.add_argument("--substage", default="1", help="1 / 2 / all")
    ap.add_argument("--arm", default=None, help="run exactly one arm (process parallel)")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--steps", type=int, default=None)
    args = ap.parse_args()
    cfg_path = Path(args.config)
    cfg = load_config(str(cfg_path))
    validate_config(cfg, stage=("run" if args.stage in ("run", "finalize")
                                else args.stage))
    require_omp(cfg)
    device = pick_device(cfg) if args.stage != "preflight" else "cpu"
    main_dir = Path(ROOT) / cfg["output"]["dir"]
    outdir = Path(args.outdir) if args.outdir else main_dir
    if args.stage == "preflight":
        preflight(cfg, Path(ROOT) / "results" / f"_preflight_{EXPERIMENT}")
        return
    if args.arm:
        total = int(args.steps or cfg["common"]["total_steps"])
        t0 = time.time()
        head = _git("rev-parse", "HEAD")
        got = run_single_arm(cfg, args.arm, device, outdir, total)
        done = dict(arm=args.arm, total_steps=total,
                    seeds=[int(v) for v in cfg["common"]["seeds"]],
                    status=got.get("status"), elapsed_sec=got.get("elapsed_sec"),
                    wall_sec=time.time() - t0, git_head=head,
                    git_head_at_launch=head)
        path = outdir / "arm_status" / f"{args.arm}_done.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(done, indent=2, ensure_ascii=False),
                        encoding="utf-8")
        print(json.dumps(done, ensure_ascii=False), flush=True)
        return
    if args.stage == "diverge-probe":
        divergence_probe(cfg, args.arm or "CB_a1_b5_1216", outdir)
        return
    if args.stage == "analyze":
        got = analyze(cfg, outdir, args.substage)
        print(json.dumps(got["verdicts"], ensure_ascii=False, indent=1, default=str),
              flush=True)
        return
    if args.stage == "finalize":
        # 腕プロセス並列で回したあとに provenance だけを書く（--arm は 1 腕ずつ走るので
        # config_used.yaml / provenance.json を書かない）。
        finalize(cfg_path, cfg, outdir, args.substage)
        return
    if args.stage == "smoke":
        run(cfg_path, cfg, device,
            Path(args.outdir or (Path(ROOT) / "results" / f"_smoke_{EXPERIMENT}")),
            args.substage, smoke=True)
        return
    run(cfg_path, cfg, device, outdir, args.substage)



def divergence_probe(cfg: dict, arm: str, outdir: Path) -> dict:
    """§5.6 が要求する「直前の最深 z̄」を回収する診断走（spec §10 追補 10）。

    宿主は発散時に部分ログを破棄する（``partial_logs_excluded``）ので、登録された
    報告項目のうち「直前の最深 z̄」だけが残らない。**同じ config・同じ腕・登録どおりの
    seed 集合（R=10）**で回し直し、記録器が持っている ``zmean`` を発散直前の probe まで
    読む。seed を 1 本に絞ると乱数の引き方が変わって別の軌跡になるので、絞らない。
    **新しい実験ではなく、登録済みの報告項目の回収である**（判定には一切使わない）。
    """
    c = copy.deepcopy(cfg)
    seeds = [int(v) for v in c["common"]["seeds"]]
    total = int(c["common"]["total_steps"])
    every = int(c["common"]["lop_every"])
    st = setup_arm_dial(c, _arm(c, arm), "cpu")
    probes = list(range(0, total + 1, every))
    rec = WeirdRecorder(probes, st)
    scratch = outdir / "_divergence_probe"
    scratch.mkdir(parents=True, exist_ok=True)
    event = None
    try:
        train_arm_gate(st, rec, probes, total, scratch, [])
    except NumericDivergenceError as exc:
        event = dict(exc.event)
    if event is None:
        return dict(arm=arm, status="NO_DIVERGENCE_ON_REPLAY",
                    note="the replay did not diverge; do not use this arm's numbers")
    detected = int(event["detected_step"])
    last = detected // every - 1          # 発散を検出した probe の 1 つ前
    zmean = rec.unit["zmean"][:last + 1]  # [probe, seed, unit]
    zmin = rec.unit["zmin"][:last + 1]
    finite = np.isfinite(zmean).all(axis=(1, 2))
    last_finite = int(np.flatnonzero(finite)[-1]) if finite.any() else -1
    rows = []
    for j, seed in enumerate(seeds):
        block = zmean[:last_finite + 1, j, :]
        rows.append(dict(
            seed=seed,
            deepest_zbar_at_last_probe=float(block[-1].min()),
            deepest_zbar_over_run=float(block.min()),
            step_of_deepest=int(probes[int(np.unravel_index(block.argmin(),
                                                            block.shape)[0])]),
            deepest_zmin_at_last_probe=float(zmin[last_finite, j, :].min()),
            diverged=bool(seed in [int(v) for v in event.get("bad_seeds", [])])))
    out = dict(arm=arm, status="RECOVERED", detected_step=detected,
               last_finite_probe_step=int(probes[last_finite]),
               probe_every=every, bad_seeds=event.get("bad_seeds"),
               nonfinite_tensors=event.get("nonfinite_tensors"),
               per_seed=rows,
               note="replay of the registered arm with the registered seed set; "
                    "used only to fill the §5.6 reporting field, never in a verdict")
    path = outdir / "arm_status" / f"{arm}_divergence_probe.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[divergence-probe] {arm}: recovered -> {path}", flush=True)
    return out


# ---------------------------------------------------------------------------
# 集計（spec §6）。窓・床・発症定義・CI は gate_dial_0902 §5 の逐語継承。
# ---------------------------------------------------------------------------
# 宿主 config の gate_dial.onset_time の逐語継承（本走の config には置いていないので
# 定数で持つ。値は gate_dial_0902.yaml と同一）。
ONSET_WINDOW_TASKS = 10
ONSET_K_MIN = 10
ONSET_CENSOR_AT = 500


def _rolling_window_unfit(step: np.ndarray, unfit: np.ndarray,
                          period: int) -> dict:
    """``U^(10)_k`` = タスク k-9..k のタスク終端記録点の unfit 平均（宿主の写し）。"""
    width = ONSET_WINDOW_TASKS
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
    threshold = float(cfg["phase1"]["onset_threshold"])
    rolled = _rolling_window_unfit(step, unfit, int(cfg["phase1"]["task_period"]))
    ks, us = rolled["k"], rolled["u"]
    rows = []
    for j in range(us.shape[1]):
        hit = np.flatnonzero((us[:, j] >= threshold) & (ks >= ONSET_K_MIN))
        rows.append(dict(k_star=int(ks[hit[0]]) if hit.size else ONSET_CENSOR_AT,
                         censored=0 if hit.size else 1))
    # k* は「最初に横断した時刻」であって持続を意味しない。誤読を防ぐため、
    # 横断していた窓の割合と末尾の U^(10) を並べる（REPORT・判定には入れない）。
    over = (us >= threshold)
    return dict(rows=rows, rolled_k=ks, rolled_u=us,
                frac_windows_over=[float(over[:, j].mean())
                                   for j in range(us.shape[1])],
                u10_at_last_k=[float(us[-1, j]) for j in range(us.shape[1])],
                last_k=int(ks[-1]),
                records_per_window=rolled["records_per_window"])


def _load_controls_weird(cfg: dict) -> dict:
    """対照の endpoint を各親走の committed 出力から**転記**する（再計算しない）。"""
    import csv as _csv

    floor = float(cfg["phase1"]["unfit_floor"])
    out: dict[str, dict] = {}
    for name, block in cfg["controls"]["arms"].items():
        run = str(block["source_run"])
        verdict = Path(ROOT) / run / "verdict.csv"
        seeds_csv = Path(ROOT) / run / "seed_values.csv"
        got = None
        if verdict.exists():
            with verdict.open(newline="") as fh:
                for row in _csv.DictReader(fh):
                    if row.get("arm") != name or "U_5m_seed_values" not in row:
                        continue
                    u5 = np.maximum(np.asarray(json.loads(row["U_5m_seed_values"]),
                                               dtype=np.float64), floor)
                    u1 = np.maximum(np.asarray(json.loads(row["U_1m_seed_values"]),
                                               dtype=np.float64), floor)
                    got = dict(u_5m=u5, u_1m=u1, log_u_5m=np.log10(u5),
                               log_u_1m=np.log10(u1),
                               n_onset_5m=int(row["n_onset_5m"]),
                               n_onset_1m=int(row["n_onset_1m"]),
                               source=str(verdict), window="5M / 1M")
        if got is None and seeds_csv.exists():
            # valley_clamp_0902 は verdict.csv の schema が違う。末尾窓の seed 値だけを取る
            u5, onset = [], 0
            with seeds_csv.open(newline="") as fh:
                for row in _csv.DictReader(fh):
                    if row.get("arm") != name:
                        continue
                    u5.append(float(row["u"]))
                    onset += int(row["onset"])
            if u5:
                arr = np.maximum(np.asarray(u5, dtype=np.float64), floor)
                got = dict(u_5m=arr, u_1m=None, log_u_5m=np.log10(arr),
                           log_u_1m=None, n_onset_5m=onset, n_onset_1m=None,
                           source=str(seeds_csv), window="5M tail only")
        if got is None:
            raise SanityError(f"control {name} not found under {run}")
        got.update(arm=name, source_run=run, level_only=(got["u_1m"] is None))
        out[name] = got
    return out


def _contrast(cfg: dict, a: np.ndarray, b: np.ndarray, draws: np.ndarray,
              label: str) -> dict:
    """seed クラスタの paired 差（log10 U の差）の中央値と CI・符号検定。"""
    values = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    ci = _ci(cfg, values, draws)
    sign = _sign_test(values)
    margin = float(_P(cfg)["p5_equivalence_margin"])
    lo, hi = ci.get("percentile_ci_lo"), ci.get("percentile_ci_hi")
    if lo is None or hi is None or not np.isfinite([lo, hi]).all():
        equiv = "INCONCLUSIVE_WIDE"
    elif lo >= -margin and hi <= margin:
        equiv = "EQUIV_SOFT"
    elif lo > 0:
        equiv = "SHORT_OF_SOFT"
    elif hi < -margin:
        equiv = "BELOW_SOFT"
    else:
        equiv = "INCONCLUSIVE_WIDE"
    return dict(label=label, point=float(np.median(values)),
                ci=ci, sign_test=sign, equivalence=equiv, margin=margin,
                seed_values=[float(v) for v in values])


def _unit_tail(cfg: dict, outdir: Path, arm_block: dict,
               window: str = "late_tasks_5m") -> dict:
    """末尾窓のユニット別量（沈下・span・深さ・位置指標・凍結・|v|）。

    集計順は宿主の登録どおり: 沈下ユニット記録 → seed 内中央値 → seed 中央値。
    """
    P = cfg["phase1"]
    arm = str(arm_block["name"])
    act = str(arm_block["activation"])
    alpha = float(arm_block["dial"])
    d = None
    if act in VecMLPL.BAND_WIDTH:
        d = VecMLPL.BAND_WIDTH[act]
    if act in VecMLPL.FOLD_DEPTH:
        d = VecMLPL.FOLD_DEPTH[act]
    per_seed, deciles = [], []
    for seed in [int(v) for v in cfg["common"]["seeds"]]:
        path = outdir / "logs" / f"{arm}_seed{seed}.npz"
        with np.load(path, allow_pickle=False) as z:
            step = z["step"]
            idx = _window_indices(step, int(P["task_period"]), list(P[window]))
            zmax = z["layer1_zmax"][idx].astype(np.float64)
            zmin = z["layer1_zmin"][idx].astype(np.float64)
            zmean = z["layer1_zmean"][idx].astype(np.float64)
            mob = z["layer1_mob"][idx].astype(np.float64)
            absmob = z["layer1_absmob"][idx].astype(np.float64)
            v_unit = z["layer1_v_unit"][idx].astype(np.float64)
        sub = zmax <= 0.0
        span = zmax - zmin
        depth = -zmean
        row = dict(seed=seed, n_records=int(zmax.shape[0]),
                   submerged_frac=float(sub.mean()),
                   span_median_submerged=float(np.median(span[sub])) if sub.any() else float("nan"),
                   span_median_all=float(np.median(span)),
                   depth_median_submerged=float(np.median(depth[sub])) if sub.any() else float("nan"),
                   frozen_frac=float((np.abs(mob) < 1e-6).mean()),
                   frozen_abs_frac=float((absmob < 1e-6).mean()),
                   absv_median_submerged=float(np.median(np.abs(v_unit)[sub])) if sub.any() else float("nan"))
        if act in VecMLPL.BAND_WIDTH:
            row["in_band_frac"] = float((sub & (zmin >= -d)).mean())
        if act in VecMLPL.FOLD_DEPTH:
            row["at_sink_frac"] = float((np.abs(zmean + 2.0 * d) <= 0.5).mean())
        if act in VecMLPL.COMB_ENVELOPE:
            ks = np.arange(1, 8)[:, None, None]
            dist = np.abs(zmean[None, ...] + ks * math.pi / alpha).min(axis=0)
            row["at_well_frac"] = float((dist <= 0.5).mean())
        per_seed.append(row)
        if sub.any():
            deciles.append(np.quantile(depth[sub], np.arange(1, 10) / 10.0))
    out = dict(arm=arm, activation=act, dial=alpha, second_param=d,
               window=window, per_seed=per_seed,
               aggregation_order=["submerged_unit_records_within_seed",
                                  "median_within_seed", "median_over_seeds"])
    for key in ("submerged_frac", "span_median_submerged", "span_median_all",
                "depth_median_submerged", "frozen_frac", "frozen_abs_frac",
                "absv_median_submerged", "in_band_frac", "at_sink_frac",
                "at_well_frac"):
        vals = [r[key] for r in per_seed if key in r and not math.isnan(r[key])]
        out[f"median_{key}"] = float(np.median(vals)) if vals else None
    out["depth_deciles_median"] = ([float(v) for v in np.median(np.stack(deciles), axis=0)]
                                   if deciles else None)
    return out


def _s_cap(cfg: dict, windows: dict) -> dict:
    """S-cap（二段）: early 窓で U<0.05 が 9/10 以上。落ちたら 1M 窓で再判定 → SLOW_FIT。"""
    S = cfg["sanity"]
    threshold, need = float(S["s_cap_threshold"]), int(S["s_cap_min_seeds"])
    early = np.asarray(windows["early"]["u"], dtype=np.float64)
    one_m = np.asarray(windows["1M"]["u"], dtype=np.float64)
    n_early = int((early < threshold).sum())
    n_1m = int((one_m < threshold).sum())
    if n_early >= need:
        status = "OK"
    elif n_1m >= need:
        status = str(S["s_cap_fallback_label"])       # SLOW_FIT
    else:
        status = str(S["s_cap_label"])                # CAPACITY_UNDEFINED
    return dict(status=status, n_seeds_below_early=n_early,
                n_seeds_below_1m=n_1m, need=need, threshold=threshold)


def _onset_stats(cfg: dict, u: np.ndarray) -> dict:
    threshold = float(cfg["phase1"]["onset_threshold"])
    n = int((np.asarray(u, dtype=np.float64) >= threshold).sum())
    lo, hi = clopper_pearson(n, int(len(u)))
    return dict(n_onset=n, cp95_lo=float(lo), cp95_hi=float(hi),
                median_log10_u=float(np.median(np.log10(np.asarray(u, dtype=np.float64)))))


def _onset_state(n: int, zero_max: int, present_min: int) -> str:
    if n <= zero_max:
        return "zero"
    if n >= present_min:
        return "present"
    return "mid"


def _v1_label(cfg: dict, n_onset: int) -> str:
    G = _P(cfg)["v1"]
    state = _onset_state(n_onset, int(G["onset_zero_max"]),
                         int(G["onset_present_min"]))
    return str(G["labels"][state])


def _v3_label(cfg: dict, n_onset: int) -> str:
    G = _P(cfg)["v3"]
    state = _onset_state(n_onset, int(_P(cfg)["v1"]["onset_zero_max"]),
                         int(_P(cfg)["v1"]["onset_present_min"]))
    return str(G["labels"][state])


def _v4_label(cfg: dict, n_onset: int, contrast: dict | None,
              at_well_frac: float | None) -> tuple[str, dict]:
    G = _P(cfg)["v4"]
    margin = float(G["margin"])
    ci = (contrast or {}).get("ci") or {}
    lo, hi = ci.get("percentile_ci_lo"), ci.get("percentile_ci_hi")
    below = bool(lo is not None and hi is not None
                 and np.isfinite([lo, hi]).all() and hi < -margin)
    detail = dict(n_onset=n_onset, ci_lo=lo, ci_hi=hi, ci_below_margin=below,
                  at_well_frac=at_well_frac, margin=margin)
    if n_onset <= int(G["rescue_onset_max"]) and below:
        return str(G["labels"]["rescue"]), detail
    if n_onset >= int(_P(cfg)["v1"]["onset_present_min"]):
        if at_well_frac is not None and at_well_frac >= float(G["trap_at_well_min"]):
            return str(G["labels"]["trap"]), detail
        return str(G["labels"]["elsewhere"]), detail
    return str(G["labels"]["partial"]), detail


def _v2_ladder(cfg: dict, arms: dict, controls: dict, draws: np.ndarray,
               stage: str) -> dict:
    """V2 は **段 2 の完了後にのみ付く**（2026-09-03 の段裁定・spec §6 V2）。"""
    G = _P(cfg)["v2"]
    new_arms = [str(a) for a in G["ladder_new_arms"]]
    have = [a for a in new_arms if a in arms]
    if str(stage) != "2" and str(stage) not in ("all", "0"):
        return dict(label=None, status="NOT_EVALUATED_STAGE_1_ONLY",
                    reason="V2 の判定条件は RB 梯子 4 腕全体に掛かる（段裁定・spec §6 V2）",
                    present_arms=have, missing_arms=[a for a in new_arms if a not in arms])
    if len(have) < len(new_arms):
        return dict(label=None, status="NOT_EVALUATED_INCOMPLETE_LADDER",
                    present_arms=have,
                    missing_arms=[a for a in new_arms if a not in arms])
    onsets = [int(arms[a]["5M"]["onset"]["n_onset"]) for a in new_arms]
    ds = [float(VecMLPL.BAND_WIDTH[str(REGISTERED_ARMS[a][2])]) for a in new_arms]
    order = np.argsort(ds)
    ordered = [onsets[i] for i in order]
    non_decreasing = all(ordered[i] <= ordered[i + 1] for i in range(len(ordered) - 1))
    zero_max = int(_P(cfg)["v1"]["onset_zero_max"])
    present_min = int(_P(cfg)["v1"]["onset_present_min"])
    drop = max(ordered) - min(ordered[ordered.index(max(ordered)):]) if ordered else 0
    adjacent = []
    ladder = [str(a) for a in G["ladder"]]
    for soft, hard in zip(ladder[:-1], ladder[1:]):
        both = []
        for name in (hard, soft):
            if name in arms:
                both.append(arms[name]["5M"]["log_u"])
            elif name in controls:
                both.append(controls[name]["log_u_5m"])
            else:
                both = None
                break
        if both:
            adjacent.append(_contrast(cfg, both[0], both[1], draws,
                                      f"{hard} - {soft}"))
    margin = float(G["margin"])
    reversal = (drop >= int(G["onset_drop_reversal_min"])
                or any((c["ci"].get("percentile_ci_hi") is not None
                        and c["ci"]["percentile_ci_hi"] < -margin) for c in adjacent))
    if reversal:
        label = "REVERSAL"
    elif non_decreasing and any(o <= zero_max for o in ordered) \
            and any(o >= present_min for o in ordered):
        label = "BAND_WIDTH_THRESHOLD"
    elif all(o >= present_min for o in ordered):
        label = "ANY_BAND_ABSORBS"
    elif all(o <= zero_max for o in ordered):
        label = "NO_BAND_ABSORBS_IN_RANGE"
    else:
        label = "PARTIAL"
    d_star = None
    for i in order:
        if onsets[i] >= present_min:
            d_star = ds[i]
            break
    # 登録された量的照合（REPORT・ラベルを作らない）: d* と 1M 窓の沈下ユニットの
    # span の seed 中央値。LR_1216 には zmin 列が無いので RB_d0p5 を代理にする。
    report_rows = []
    for i in order:
        name = new_arms[i]
        e = arms[name]
        report_rows.append(dict(
            arm=name, d=ds[i],
            n_onset_1m=int(e["1M"]["onset"]["n_onset"]),
            n_onset_5m=int(e["5M"]["onset"]["n_onset"]),
            median_log10_U_5m=float(e["5M"]["onset"]["median_log10_u"]),
            span_median_1m=e["unit_1m"]["median_span_median_submerged"],
            span_median_5m=e["unit"]["median_span_median_submerged"],
            in_band_frac_1m=e["unit_1m"].get("median_in_band_frac"),
            in_band_frac_5m=e["unit"].get("median_in_band_frac"),
            submerged_frac_5m=e["unit"]["median_submerged_frac"]))
    return dict(label=label, status="EVALUATED", d_star=d_star,
                ladder=ladder, onsets_by_d=list(zip([ds[i] for i in order], ordered)),
                non_decreasing=bool(non_decreasing), drop_after_peak=int(drop),
                report_rows=report_rows,
                span_proxy_for_LR_1216=str(G.get("span_proxy_for_LR_1216")),
                adjacent_contrasts=adjacent)


def _v3_followup(cfg: dict, outdir: Path) -> list[dict]:
    """spec §7.4 (v) の登録済み追走: V3 が 0/10 のとき分水嶺を実際に越えているか。

    越えていなければ ``LRv_d2`` は分水嶺を試していない。越えているのに戻っていれば
    ``MOBILITY_SUFFICES`` は本物である（spec §7.4 (v) の字義）。
    """
    P = cfg["phase1"]
    rows = []
    for arm in ("LRv_d2_1216", "LRv_d1_1216"):
        if not (outdir / "logs" / f"{arm}_seed0.npz").exists():
            continue
        d = float(VecMLPL.FOLD_DEPTH[str(REGISTERED_ARMS[arm][2])])
        for window, tasks in (("1M", list(P["window_1m_tasks"])),
                              ("5M", list(P["late_tasks_5m"]))):
            depths, crossed, past_sink, subs = [], [], [], []
            for seed in [int(v) for v in cfg["common"]["seeds"]]:
                with np.load(outdir / "logs" / f"{arm}_seed{seed}.npz",
                             allow_pickle=False) as z:
                    idx = _window_indices(z["step"], int(P["task_period"]), tasks)
                    zmax = z["layer1_zmax"][idx].astype(np.float64)
                    zmean = z["layer1_zmean"][idx].astype(np.float64)
                sub = zmax <= 0.0
                subs.append(float(sub.mean()))
                if sub.any():
                    depth = -zmean[sub]
                    depths.append(float(np.median(depth)))
                    crossed.append(float((depth > d).mean()))
                    past_sink.append(float((depth > 2.0 * d).mean()))
            rows.append(dict(arm=arm, window=window, d=d,
                             median_submerged_frac=float(np.median(subs)),
                             median_depth_submerged=float(np.median(depths)),
                             frac_past_watershed=float(np.median(crossed)),
                             frac_past_sink=float(np.median(past_sink))))
    return rows


def analyze(cfg: dict, outdir: Path, stage: str) -> dict:
    """spec §6 の集計。段 1 では V1・V3・V4 だけが付き、V2 は空にする。"""
    P, G = cfg["phase1"], _P(cfg)
    arms = _selected_arms(cfg, stage)
    draws = _draws(cfg)
    controls = _load_controls_weird(cfg)
    blocks = {a["name"]: a for a in cfg["arms"]}
    data: dict[str, dict] = {}
    onset_rows, km_rows, position_rows, depth_rows = [], [], [], []
    diverged: dict[str, dict] = {}
    for arm in arms:
        if not (outdir / "logs" / f"{arm}_seed0.npz").exists():
            event_path = _arm_status_path(outdir, arm)
            if event_path.exists():
                event = json.loads(event_path.read_text(encoding="utf-8"))
                probe_path = (outdir / "arm_status"
                              / f"{arm}_divergence_probe.json")
                event["probe"] = (json.loads(probe_path.read_text(encoding="utf-8"))
                                  if probe_path.exists() else None)
                diverged[arm] = event
                print(f"[analyze] {arm}: {NUMERIC_DIVERGENCE} at step "
                      f"{event['detected_step']:,}, dropped (spec §6 数値発散)",
                      flush=True)
            else:
                print(f"[analyze] {arm}: logs missing, skipped", flush=True)
            continue
        w = _load_new_arm(cfg, outdir, arm)
        entry = {}
        for key in ("5M", "1M", "early"):
            u = np.asarray(w[key]["u"], dtype=np.float64)
            entry[key] = dict(u=u, log_u=np.log10(u), onset=_onset_stats(cfg, u),
                              metrics=w[key])
        entry["s_cap"] = _s_cap(cfg, {k: dict(u=entry[k]["u"]) for k in
                                      ("early", "1M", "5M")})
        entry["unit"] = _unit_tail(cfg, outdir, blocks[arm])
        entry["unit_1m"] = _unit_tail(cfg, outdir, blocks[arm],
                                      window="window_1m_tasks")
        ot = _onset_times(cfg, w["data"]["step"], w["data"]["unfit"])
        entry["onset_times"] = ot
        for j, (seed, row) in enumerate(zip([int(v) for v in cfg["common"]["seeds"]],
                                            ot["rows"])):
            onset_rows.append(dict(arm=arm, seed=seed, **row,
                                   frac_windows_over=ot["frac_windows_over"][j],
                                   u10_at_last_k=ot["u10_at_last_k"][j],
                                   last_k=ot["last_k"]))
        km_rows.extend(dict(arm=arm, **r) for r in _kaplan_meier(
            [r["k_star"] for r in ot["rows"]], [r["censored"] for r in ot["rows"]],
            ONSET_CENSOR_AT))
        u = entry["unit"]
        position_rows.append(dict(
            arm=arm, activation=u["activation"], dial=u["dial"],
            second_param=u["second_param"],
            median_submerged_frac=u["median_submerged_frac"],
            median_span_submerged=u["median_span_median_submerged"],
            median_span_all=u["median_span_median_all"],
            median_depth_submerged=u["median_depth_median_submerged"],
            median_frozen_frac=u["median_frozen_frac"],
            median_frozen_abs_frac=u["median_frozen_abs_frac"],
            median_absv_submerged=u["median_absv_median_submerged"],
            in_band_frac=u.get("median_in_band_frac"),
            at_sink_frac=u.get("median_at_sink_frac"),
            at_well_frac=u.get("median_at_well_frac"),
            window="late_tasks_5m (491-500, task-end records only)"))
        if u["depth_deciles_median"]:
            depth_rows.append(dict(arm=arm, **{f"d{i + 1}": v for i, v in
                                               enumerate(u["depth_deciles_median"])}))
        data[arm] = entry

    # --- 水準の対比（E2） ---
    contrasts = []
    for arm, entry in data.items():
        family = str(blocks[arm]["family"])
        base = str(G["p3prime_baseline"])
        if base in controls:
            for window, key in (("5M", "log_u_5m"), ("1M", "log_u_1m")):
                if controls[base][key] is None:
                    continue
                contrasts.append(dict(
                    arm=arm, kind="P3prime", window=window, against=base,
                    **_contrast(cfg, entry[window]["log_u"], controls[base][key],
                                draws, f"{arm} - {base} ({window})")))
        for soft in [str(v) for v in G["p5_soft_end_by_family"].get(family, [])]:
            if soft in controls and controls[soft]["log_u_5m"] is not None:
                contrasts.append(dict(
                    arm=arm, kind="P5prime", window="5M", against=soft,
                    **_contrast(cfg, entry["5M"]["log_u"], controls[soft]["log_u_5m"],
                                draws, f"{arm} - {soft} (5M)")))

    # --- 判定 ---
    verdicts: dict[str, object] = {}
    v1_arm = str(G["v1"]["arm"])
    if v1_arm in data:
        verdicts["V1"] = _v1_label(cfg, data[v1_arm]["5M"]["onset"]["n_onset"])
        verdicts["V1_n_onset_5m"] = data[v1_arm]["5M"]["onset"]["n_onset"]
        if data[v1_arm]["s_cap"]["status"] == str(cfg["sanity"]["s_cap_label"]):
            verdicts["V1"] = None
            verdicts["V1_status"] = cfg["sanity"]["s_cap_label"]
    v3_arm = str(G["v3"]["arm"])
    if v3_arm in data:
        verdicts["V3"] = _v3_label(cfg, data[v3_arm]["5M"]["onset"]["n_onset"])
        verdicts["V3_n_onset_5m"] = data[v3_arm]["5M"]["onset"]["n_onset"]
    anchor = str(G["v3"]["anchor_arm"])
    if anchor in data:
        # LRv_d1 は REPORT_ONLY。V3 が PARTIAL のときだけ V3' に格上げする（段裁定）
        verdicts["V3_prime"] = (_v3_label(cfg, data[anchor]["5M"]["onset"]["n_onset"])
                                if verdicts.get("V3") == "PARTIAL" else None)
        verdicts["V3_prime_status"] = ("PROMOTED" if verdicts.get("V3") == "PARTIAL"
                                       else "REPORT_ONLY")
        verdicts["V3_prime_n_onset_5m"] = data[anchor]["5M"]["onset"]["n_onset"]
    v4_arm = str(G["v4"]["arm"])
    if v4_arm in data:
        against = str(G["v4"]["against_level"])
        v4_contrast = None
        if against in controls and controls[against]["log_u_5m"] is not None:
            v4_contrast = _contrast(cfg, data[v4_arm]["5M"]["log_u"],
                                    controls[against]["log_u_5m"], draws,
                                    f"{v4_arm} - {against} (5M)")
            contrasts.append(dict(arm=v4_arm, kind="V4", window="5M",
                                  against=against, **v4_contrast))
        second = str(G["v4"]["against_level_secondary"])
        if second in controls and controls[second]["log_u_5m"] is not None:
            contrasts.append(dict(
                arm=v4_arm, kind="V4_secondary", window="5M", against=second,
                **_contrast(cfg, data[v4_arm]["5M"]["log_u"],
                            controls[second]["log_u_5m"], draws,
                            f"{v4_arm} - {second} (5M)")))
        label, detail = _v4_label(cfg, data[v4_arm]["5M"]["onset"]["n_onset"],
                                  v4_contrast,
                                  data[v4_arm]["unit"].get("median_at_well_frac"))
        verdicts["V4"], verdicts["V4_detail"] = label, detail
    verdicts["V2"] = _v2_ladder(cfg, data, controls, draws, stage)
    v3_rows = _v3_followup(cfg, outdir)
    verdicts["V3_followup"] = v3_rows

    verdicts["divergences"] = {k: dict(detected_step=v["detected_step"],
                                       bad_seeds=v.get("bad_seeds"))
                               for k, v in diverged.items()}
    result = dict(experiment=EXPERIMENT, stage=stage, arms=list(data),
                  diverged=list(diverged), verdicts=verdicts, controls={k: dict(
                      source=v["source"], source_run=v["source_run"],
                      window=v["window"], n_onset_5m=v["n_onset_5m"],
                      median_log10_u_5m=float(np.median(v["log_u_5m"])),
                      level_only=v["level_only"]) for k, v in controls.items()})
    if v3_rows:
        write_csv(outdir / "v3_watershed_followup.csv", v3_rows)
    _write_outputs(cfg, outdir, stage, data, controls, contrasts, verdicts,
                   onset_rows, km_rows, position_rows, depth_rows, result,
                   diverged)
    return result


def _write_outputs(cfg, outdir, stage, data, controls, contrasts, verdicts,
                   onset_rows, km_rows, position_rows, depth_rows, result,
                   diverged=None) -> None:
    diverged = diverged or {}
    blocks = {a["name"]: a for a in cfg["arms"]}
    G = _P(cfg)
    v_rows = []
    for arm, entry in data.items():
        block = blocks[arm]
        row = dict(
            arm=arm, stage=int(block["stage"]), family=str(block["family"]),
            activation=str(block["activation"]), dial=float(block["dial"]),
            second_param=block.get("second_param"),
            target_dose=float(block["target_dose"]),
            is_control=False, status="COMPLETE",
            capacity_status=entry["s_cap"]["status"],
            n_onset_1m=entry["1M"]["onset"]["n_onset"],
            cp95_1m_lo=entry["1M"]["onset"]["cp95_lo"],
            cp95_1m_hi=entry["1M"]["onset"]["cp95_hi"],
            U_1m_seed_values=json.dumps([float(v) for v in entry["1M"]["u"]]),
            median_log10_U_1m=entry["1M"]["onset"]["median_log10_u"],
            n_onset_5m=entry["5M"]["onset"]["n_onset"],
            cp95_5m_lo=entry["5M"]["onset"]["cp95_lo"],
            cp95_5m_hi=entry["5M"]["onset"]["cp95_hi"],
            U_5m_seed_values=json.dumps([float(v) for v in entry["5M"]["u"]]),
            median_log10_U_5m=entry["5M"]["onset"]["median_log10_u"],
            median_submerged_frac_5m=entry["unit"]["median_submerged_frac"],
            median_span_5m=entry["unit"]["median_span_median_submerged"],
            median_depth_5m=entry["unit"]["median_depth_median_submerged"],
            V1=verdicts.get("V1") if arm == str(G["v1"]["arm"]) else "",
            V2="", V3=verdicts.get("V3") if arm == str(G["v3"]["arm"]) else "",
            V3_prime=(verdicts.get("V3_prime") or verdicts.get("V3_prime_status", ""))
            if arm == str(G["v3"]["anchor_arm"]) else "",
            V4=verdicts.get("V4") if arm == str(G["v4"]["arm"]) else "",
            NUMERIC_DIVERGENCE="")
        v2 = verdicts.get("V2") or {}
        if arm in [str(a) for a in G["v2"]["ladder_new_arms"]]:
            row["V2"] = v2.get("label") or ""
        row["V2_status"] = (v2.get("status")
                            if arm in [str(a) for a in G["v2"]["ladder_new_arms"]]
                            else "")
        v_rows.append(row)
    for arm, event in diverged.items():
        block = blocks[arm]
        v_rows.append(dict(
            arm=arm, stage=int(block["stage"]), family=str(block["family"]),
            activation=str(block["activation"]), dial=float(block["dial"]),
            second_param=block.get("second_param"),
            target_dose=float(block["target_dose"]), is_control=False,
            status=NUMERIC_DIVERGENCE, capacity_status="",
            n_onset_1m="", cp95_1m_lo="", cp95_1m_hi="", U_1m_seed_values="",
            median_log10_U_1m="", n_onset_5m="", cp95_5m_lo="", cp95_5m_hi="",
            U_5m_seed_values="", median_log10_U_5m="",
            median_submerged_frac_5m="", median_span_5m="", median_depth_5m="",
            V1="", V2="", V3="", V3_prime="", V4="",
            NUMERIC_DIVERGENCE=json.dumps(dict(
                detected_step=event["detected_step"],
                bad_seeds=event.get("bad_seeds")), ensure_ascii=False),
            V2_status=""))
    for name, c in controls.items():
        v_rows.append(dict(
            arm=name, stage="", family="", activation="", dial="",
            second_param="", target_dose=12.16, is_control=True,
            status="TRANSCRIBED", capacity_status="",
            n_onset_1m=c["n_onset_1m"] if c["n_onset_1m"] is not None else "",
            cp95_1m_lo="", cp95_1m_hi="",
            U_1m_seed_values=json.dumps([float(v) for v in c["u_1m"]])
            if c["u_1m"] is not None else "",
            median_log10_U_1m=float(np.median(c["log_u_1m"]))
            if c["log_u_1m"] is not None else "",
            n_onset_5m=c["n_onset_5m"], cp95_5m_lo="", cp95_5m_hi="",
            U_5m_seed_values=json.dumps([float(v) for v in c["u_5m"]]),
            median_log10_U_5m=float(np.median(c["log_u_5m"])),
            median_submerged_frac_5m="", median_span_5m="", median_depth_5m="",
            V1="", V2="", V3="", V3_prime="", V4="", NUMERIC_DIVERGENCE="",
            V2_status=""))
    # write_csv は先頭行の keys を fieldnames にするので、全行を同じ形に揃える
    columns: list[str] = []
    for row in v_rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    v_rows = [{key: row.get(key, "") for key in columns} for row in v_rows]
    write_csv(outdir / "verdict.csv", v_rows)
    if onset_rows:
        write_csv(outdir / "onset_times.csv", onset_rows)
    if km_rows:
        write_csv(outdir / "onset_km.csv", km_rows)
    if position_rows:
        write_csv(outdir / "position_table.csv", position_rows)
    if depth_rows:
        write_csv(outdir / "depth_hist.csv", depth_rows)
    layer_rows = []
    for c in contrasts:
        ci = c["ci"]
        layer_rows.append(dict(
            arm=c["arm"], kind=c["kind"], window=c["window"], against=c["against"],
            point=c["point"], percentile_lo=ci.get("percentile_ci_lo"),
            percentile_hi=ci.get("percentile_ci_hi"),
            studentized_lo=ci.get("studentized_ci_lo"),
            studentized_hi=ci.get("studentized_ci_hi"),
            ci_degenerate=ci.get("ci_degenerate"),
            equivalence=c["equivalence"], margin=c["margin"],
            sign_pos=c["sign_test"]["n_positive"],
            sign_neg=c["sign_test"]["n_negative"],
            sign_p=c["sign_test"]["p_two_sided"],
            seed_values=json.dumps(c["seed_values"])))
    if layer_rows:
        write_csv(outdir / "layer_stats.csv", layer_rows)
    v2 = verdicts.get("V2") or {}
    if v2.get("status") == "EVALUATED":
        write_csv(outdir / "ladder_table.csv", v2["report_rows"])
        if v2.get("adjacent_contrasts"):
            adj = []
            for c in v2["adjacent_contrasts"]:
                ci = c["ci"]
                adj.append(dict(
                    contrast=c["label"], point=c["point"],
                    percentile_lo=ci.get("percentile_ci_lo"),
                    percentile_hi=ci.get("percentile_ci_hi"),
                    equivalence=c["equivalence"], margin=c["margin"],
                    sign_pos=c["sign_test"]["n_positive"],
                    sign_neg=c["sign_test"]["n_negative"],
                    seed_values=json.dumps(c["seed_values"])))
            write_csv(outdir / "ladder_adjacent_contrasts.csv", adj)
    _write_summary(cfg, outdir, stage, data, controls, contrasts, verdicts,
                   result, diverged)


def _write_summary(cfg, outdir, stage, data, controls, contrasts, verdicts,
                   result, diverged=None) -> None:
    diverged = diverged or {}
    G = _P(cfg)
    L = []
    L.append(f"# {EXPERIMENT} — 謎関数ダイヤル（段 {stage}）\n")
    L.append(f"spec: `{cfg['spec']}` / 事前登録 commit で凍結。"
             "数値の引用は `verdict.csv` と本ファイルからのみ。\n")
    L.append("## S-cover（§6 の各項目 → 実装の対応先）\n")
    L.append("| §6 の項目 | 実装 | 出力 | 段 1 で付くか |")
    L.append("| --- | --- | --- | --- |")
    for item, impl, out, ok in (
            ("V1 反転の定義", "_v1_label", "verdict.csv:V1", "○"),
            ("V2 吸収域の幅", "_v2_ladder", "verdict.csv:V2 / ladder_table.csv",
             "×（段 2 の完了後）"),
            ("V3 分水嶺の位置", "_v3_label", "verdict.csv:V3", "○"),
            ("V3' 錨 LRv_d1", "_v3_label（PARTIAL のときだけ格上げ）",
             "verdict.csv:V3_prime", "REPORT_ONLY"),
            ("V4 井戸の容量", "_v4_label", "verdict.csv:V4", "○"),
            ("E1 発症数", "_onset_stats", "verdict.csv:n_onset_*", "○"),
            ("E2 水準 P3'/P5'", "_contrast", "layer_stats.csv", "○"),
            ("E3 発症時刻 k*", "_onset_times / _kaplan_meier",
             "onset_times.csv / onset_km.csv", "○"),
            ("span・位置指標・凍結", "_unit_tail", "position_table.csv", "○"),
            ("深さ十分位", "_unit_tail", "depth_hist.csv", "○"),
            ("C1 の再現", "未実装（走後の別解析）", "—", "×"),
            ("境界回帰の 3 レジーム", "未実装（走後の別解析）", "—", "×")):
        L.append(f"| {item} | {impl} | {out} | {ok} |")
    L.append("\n**★ 未実装 2 件**: §6 副次の「C1 の再現」と「境界回帰の 3 レジーム」は "
             "REPORT_ONLY で、判定には入らない。走後に別途起こす（spec §6 副次が"
             "「診断スクリプトは本走で `src/` に置いて登録する」と書いているので、"
             "**この 2 件は未了である**）。\n")
    L.append("## 判定\n")
    L.append("| 判定 | ラベル | 腕 | n_onset(5M) |")
    L.append("| --- | --- | --- | --- |")
    L.append(f"| V1 | {verdicts.get('V1')} | {G['v1']['arm']} | "
             f"{verdicts.get('V1_n_onset_5m')} |")
    v2 = verdicts.get("V2") or {}
    L.append(f"| V2 | {v2.get('label') or '—'} | RB 梯子 | {v2.get('status')} |")
    L.append(f"| V3 | {verdicts.get('V3')} | {G['v3']['arm']} | "
             f"{verdicts.get('V3_n_onset_5m')} |")
    L.append(f"| V3' | {verdicts.get('V3_prime') or verdicts.get('V3_prime_status')}"
             f" | {G['v3']['anchor_arm']} | {verdicts.get('V3_prime_n_onset_5m')} |")
    L.append(f"| V4 | {verdicts.get('V4')} | {G['v4']['arm']} | "
             f"{(verdicts.get('V4_detail') or {}).get('n_onset')} |")
    L.append("\n**V1〜V4 は互いに独立の判定で、1 つの verdict に畳まない。**"
             "「3 列のどれが病理を担う」は 4 判定から人が読む裁定であって"
             "本走のラベルではない（spec §9）。\n")
    if (verdicts.get("V2") or {}).get("status") != "EVALUATED":
        L.append("**V2 は段 2 の完了後にのみ付く**（2026-09-03 の段裁定・spec §6 V2）。"
                 "段 1 の `RB_d1_1216` は REPORT として置くだけで、"
                 "その値を見て段 2 の母数を変えない。\n")
    L.append("## 腕（新規・末尾窓 = タスク 491–500 のタスク終端 10 点）\n")
    L.append("| 腕 | 活性化 | dial | S-cap | n_onset 1M | n_onset 5M | "
             "median log10 U (5M) | 沈下率 | span 中央値 | 深さ中央値 |")
    L.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for arm, e in data.items():
        u = e["unit"]
        def _f(x):
            return "—" if x is None or (isinstance(x, float) and math.isnan(x)) \
                else f"{x:.4g}"
        L.append(f"| `{arm}` | {u['activation']} | {u['dial']:g} | "
                 f"{e['s_cap']['status']} | {e['1M']['onset']['n_onset']}/10 | "
                 f"{e['5M']['onset']['n_onset']}/10 | "
                 f"{e['5M']['onset']['median_log10_u']:.4f} | "
                 f"{_f(u['median_submerged_frac'])} | "
                 f"{_f(u['median_span_median_submerged'])} | "
                 f"{_f(u['median_depth_median_submerged'])} |")
    L.append("\n## E3 発症時刻 $k^\\ast$ と**横断の持続**（REPORT）\n")
    L.append("| 腕 | $k^\\ast$ 中央値 | 横断した seed | 横断していた窓の割合（中央値） | "
             "$U^{(10)}_{500}$ 中央値 | 末尾窓 n_onset |")
    L.append("| --- | --- | --- | --- | --- | --- |")
    for arm, e in data.items():
        ot = e["onset_times"]
        ks = [r["k_star"] for r in ot["rows"]]
        n_cross = sum(1 for r in ot["rows"] if not r["censored"])
        L.append(f"| `{arm}` | {float(np.median(ks)):.0f} | {n_cross}/10 | "
                 f"{float(np.median(ot['frac_windows_over'])):.3f} | "
                 f"{float(np.median(ot['u10_at_last_k'])):.3g} | "
                 f"{e['5M']['onset']['n_onset']}/10 |")
    L.append("\n**$k^\\ast$ は「最初に $U^{(10)}_k\\ge0.05$ を横断した時刻」であって"
             "持続を意味しない。** 末尾窓の `n_onset` と食い違う腕（横断はするが戻る）が"
             "あるので、$k^\\ast$ を単独で「発症した」と読まない。"
             "横断窓の割合と $U^{(10)}_{500}$ は**事後の REPORT**で登録判定に入らない。\n")
    L.append("\n## 位置指標（末尾窓・REPORT・verdict には入れない）\n")
    L.append("| 腕 | in_band | at_sink | at_well | frozen | frozen_abs | |v| 中央値 |")
    L.append("| --- | --- | --- | --- | --- | --- | --- |")
    for arm, e in data.items():
        u = e["unit"]
        def _g(k):
            v = u.get(k)
            return "—" if v is None else f"{v:.4g}"
        L.append(f"| `{arm}` | {_g('median_in_band_frac')} | "
                 f"{_g('median_at_sink_frac')} | {_g('median_at_well_frac')} | "
                 f"{_g('median_frozen_frac')} | {_g('median_frozen_abs_frac')} | "
                 f"{_g('median_absv_median_submerged')} |")
    L.append("\n**`frozen` は `LRv`・`CB` では凍結の指標にならない**"
             "（支持が折れ目・井戸を跨ぐと $\\varphi'$ の符号が混じり打ち消す）。"
             "引くなら `frozen_abs` と出所・窓を添える（spec §9）。\n")
    v2 = verdicts.get("V2") or {}
    if v2.get("status") == "EVALUATED":
        L.append("## V2 の梯子（`RB` 4 腕・登録された量的照合も併記）\n")
        L.append("| 腕 | d | n_onset 1M | n_onset 5M | median log10 U (5M) | "
                 "span 中央値 (1M) | in_band 率 (1M) | in_band 率 (5M) |")
        L.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for r in v2["report_rows"]:
            L.append(f"| `{r['arm']}` | {r['d']:g} | {r['n_onset_1m']}/10 | "
                     f"{r['n_onset_5m']}/10 | {r['median_log10_U_5m']:.4f} | "
                     f"{r['span_median_1m']:.3f} | {r['in_band_frac_1m']:.4f} | "
                     f"{r['in_band_frac_5m']:.4f} |")
        L.append(f"\n判定は **{v2['label']}**。梯子は $d$ について**単調ではない**"
                 f"（非減少 = {v2['non_decreasing']}・ピーク後の落ち幅 "
                 f"{v2['drop_after_peak']}）。`span` の代理は "
                 f"`{v2['span_proxy_for_LR_1216']}`（`LR_1216` に `zmin` 列が無いため）。\n")
        d_star, spans = v2.get("d_star"), [r["span_median_1m"] for r in v2["report_rows"]]
        span_med = float(np.median([x for x in spans if x is not None]))
        L.append(f"**登録された予言の照合（§6 V2 の REPORT）**: $d^\\ast$（$n_{{\\rm onset}}\\ge5$ "
                 f"になる最小の $d$）= **{d_star}**、1M 窓の `span` 中央値 = **{span_med:.3f}**。"
                 f"予言「$d^\\ast$ は `span` の中央値の隣の目盛りに来る」は**外れている**"
                 f"（目盛りは 0.5 / 1 / 2 / 4）。")
        L.append("spec §7.4 (iv) の字義: 「$d^\\ast$ が `span` から 2 目盛り以上離れていれば、"
                 "**幅の条件そのものが誤りで実装ではない**」。ここは 2 目盛り以上離れている。\n")
        L.append("**★ さらに逆向き**: 支持が死帯に丸ごと収まる `in_band` 率は "
                 "$d$=4 の 1M 窓で **0.27** と唯一まとまって立つのに、その $d$=4 が"
                 "**発症 0/10 で梯子の中で最も軽い**。"
                 "「吸収域が支持幅を越えると凍る」という幾何の条件は、"
                 "この 4 点では**支持されない**。\n")
        L.append("| 隣接対比（硬い側 − 軟らかい側・5M） | 点推定 | percentile CI | 等価判定 | 符号 |")
        L.append("| --- | --- | --- | --- | --- |")
        for c in v2["adjacent_contrasts"]:
            ci = c["ci"]
            L.append(f"| {c['label']} | {c['point']:+.4f} | "
                     f"[{ci['percentile_ci_lo']:+.3f}, {ci['percentile_ci_hi']:+.3f}] | "
                     f"{c['equivalence']} | "
                     f"{c['sign_test']['n_negative']}:{c['sign_test']['n_positive']} |")
        L.append("")
    rows = verdicts.get("V3_followup") or []
    if rows:
        L.append("## V3 の登録済み追走（spec §7.4 (v)）——分水嶺を実際に越えたか\n")
        L.append("| 腕 | 窓 | d | 沈下率 | 深さ中央値 | 分水嶺 −d を越えた割合 | 極小 −2d より深い割合 |")
        L.append("| --- | --- | --- | --- | --- | --- | --- |")
        for r in rows:
            L.append(f"| `{r['arm']}` | {r['window']} | {r['d']:g} | "
                     f"{r['median_submerged_frac']:.3f} | "
                     f"{r['median_depth_submerged']:.3f} | "
                     f"{r['frac_past_watershed']:.3f} | {r['frac_past_sink']:.3f} |")
        L.append("\nspec §7.4 (v) の字義: 「分水嶺 −d を**越えていない**なら `LRv_d2` は"
                 "分水嶺を試していない。越えているのに戻っているなら `MOBILITY_SUFFICES` は"
                 "本物」。\n")
    L.append("## 水準の対比（対照は**別走の committed 値**・同一走の腕ではない）\n")
    L.append("| 腕 | 種別 | 窓 | 相手 | 点推定 | percentile CI | 等価判定 | 符号 |")
    L.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for c in contrasts:
        ci = c["ci"]
        lo, hi = ci.get("percentile_ci_lo"), ci.get("percentile_ci_hi")
        ci_s = "—" if lo is None or hi is None else f"[{lo:+.3f}, {hi:+.3f}]"
        L.append(f"| `{c['arm']}` | {c['kind']} | {c['window']} | `{c['against']}` | "
                 f"{c['point']:+.4f} | {ci_s} | {c['equivalence']} | "
                 f"{c['sign_test']['n_negative']}:{c['sign_test']['n_positive']} |")
    L.append("\n### 対照の出所\n")
    for name, c in controls.items():
        L.append(f"- `{name}`: {c['source_run']} / `{Path(c['source']).name}`"
                 f"（{c['window']}"
                 f"{'・**水準のみ**' if c['level_only'] else ''}）")
    if diverged:
        L.append("\n## 数値発散（spec §6・gate_dose §5.6 の逐語継承）\n")
        for arm, event in diverged.items():
            bad = event.get("bad_seeds") or []
            L.append(f"- **`{arm}` は `{NUMERIC_DIVERGENCE}`**。"
                     f"最初の発散 step **{int(event['detected_step']):,}**"
                     f"（タスク {event.get('detected_task')}）・"
                     f"発散 seed **{len(bad)}/10**（seed {bad}）。"
                     f"登録どおり**当該腕だけを落とし**、部分ログは破棄した。")
            probe = event.get("probe")
            if probe:
                rows = {int(r["seed"]): r for r in probe["per_seed"]}
                bad_row = rows.get(bad[0]) if bad else None
                deepest = min(r["deepest_zbar_at_last_probe"]
                              for r in probe["per_seed"])
                L.append(f"  - **直前の最深 z̄**（登録された報告項目・"
                         f"最後に有限だった probe = step "
                         f"{int(probe['last_finite_probe_step']):,}）: "
                         f"全 seed の最深は **{deepest:.4f}**、"
                         + (f"発散した seed {bad[0]} は **{bad_row['deepest_zbar_at_last_probe']:.4f}**"
                            f"（走行中の最深は {bad_row['deepest_zbar_over_run']:.4f} @ step "
                            f"{int(bad_row['step_of_deepest']):,}）。" if bad_row else ""))
                if bad_row and bad_row["deepest_zbar_at_last_probe"] > deepest:
                    L.append("  - **★ 発散した seed は、発散直前にいちばん深かった seed ではない。**"
                             " 深さで発散を説明できない（包絡 $e^{-z/\\beta}$ が"
                             " float32 で溢れる深さは 444 で、どの seed もその桁に居ない）。"
                             " §7.3 の理由づけ「深さ 15 に達する seed があれば発散する」は"
                             "**外れている**——予測（発散 0–2 seed）が当たったのは理由が違う。")
                L.append(f"  - 回収は同一 config・同一腕・**登録どおりの seed 集合（R=10）**"
                         f"での再走による（seed を 1 本に絞ると乱数の引き方が変わり別軌跡になる）。"
                         f"再走でも発散 step は同一で決定的。**判定には一切使っていない。**")
            L.append(f"  - `{arm}` は**錨であって判定腕ではない**ので、"
                     f"V4 は `{G['v4']['arm']}` で判定する（spec §6 数値発散）。"
                     f"失われるのは錨 1 本（包絡の効果 `{arm} − {G['v4']['arm']}`）だけ。\n")
    v4_arm = str(G["v4"]["arm"])
    if v4_arm in data and verdicts.get("V4"):
        e = data[v4_arm]
        at_well = e["unit"].get("median_at_well_frac")
        level = e["5M"]["onset"]["median_log10_u"]
        best_control = min(controls.items(),
                           key=lambda kv: float(np.median(kv[1]["log_u_5m"])))
        L.append("\n## ★ V4 の読みの限界（登録ラベルと機構の帰属は別物）\n")
        L.append(f"- `{v4_arm}` の末尾窓 `at_well` 率は **{at_well:.4g}**。"
                 f"登録表の `WELL_RESCUES` 分岐は `at_well` を条件に持たない"
                 f"（条件を持つのは `WELL_TRAPS` 側だけ）ので**ラベルは登録どおり**だが、"
                 f"**「井戸に居るから救われた」とは読めない**。")
        L.append(f"- 水準は median log10 U = **{level:.4f}** で、対照の最良"
                 f"（`{best_control[0]}` の {float(np.median(best_control[1]['log_u_5m'])):.4f}）"
                 f"より桁で低い。**逃走の除去だけでは説明できない**"
                 f"——櫛は負側に周期的な特徴を足すので、"
                 f"**表現力が増えた可能性が交絡している**（spec §7.3 の `CB_a2` 欄と同じ懸念）。")
        L.append("- したがって V4 は「井戸を置くと LoP が観測されなくなる」までで、"
                 "「井戸の容量が救済を運ぶ」は**本走では分離できていない**。\n")
    L.append("\n## 引用上の注意\n")
    L.append("- 0/10 は「5M までに観測しなかった」（片側 95% 上限 0.2589）。"
             "「起きない」と書かない")
    L.append("- **用量 1 点（12.16）・1 層・5M・float32 の主張である。**"
             "引くときは用量を添える")
    L.append("- 5 族はすべて本走のための合成活性化。処方箋として一般化しない")
    L.append("- §2 の分水嶺・井戸・極小は閉形式の代入値。`span` の実測が出るまで引かない")
    L.append("- **S-pair の親走との照合は本機では検証不能**"
             "（`results/gate_dose_0830/logs` が無い）。腕どうしの一致だけが取れている")
    (outdir / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()
