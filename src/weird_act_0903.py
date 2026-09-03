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
from .dose_const_5m import _input_stats, _refresh_fixed_offset
from .elu_swamp import exact_layer_record_elu
from .gate_dose import IDENTITY_TOL, SIGMA_TOL, train_arm_gate
from .gate_dial_0902 import (DialRecorder, NEW_UNIT_KEYS, SanityError, _arm,
                             _arm_status_path, setup_arm_dial,
                             unit_extra_record, write_arm_logs_dial)
from .mlp2_phase0 import (_sha_array, _sha_file, identity_sanity_pass,
                          require_omp)
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
    if stage not in {"preflight", "smoke", "run", "analyze"}:
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


def run_single_arm(cfg: dict, arm: str, device: str, outdir: Path,
                   total: int) -> dict:
    seeds = [int(v) for v in cfg["common"]["seeds"]]
    outdir.mkdir(parents=True, exist_ok=True)
    return _run_arm_weird(cfg, arm, device, outdir, seeds, total)


def main() -> None:
    ap = argparse.ArgumentParser(description=EXPERIMENT)
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--stage", default="preflight",
                    choices=["preflight", "smoke", "run"])
    ap.add_argument("--substage", default="1", help="1 / 2 / all")
    ap.add_argument("--arm", default=None, help="run exactly one arm (process parallel)")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--steps", type=int, default=None)
    args = ap.parse_args()
    cfg_path = Path(args.config)
    cfg = load_config(str(cfg_path))
    validate_config(cfg, stage=("run" if args.stage == "run" else args.stage))
    require_omp(cfg)
    device = pick_device(cfg) if args.stage != "preflight" else "cpu"
    main_dir = Path(ROOT) / cfg["output"]["dir"]
    outdir = Path(args.outdir) if args.outdir else main_dir
    if args.stage == "preflight":
        preflight(cfg, Path(ROOT) / "results" / f"_preflight_{EXPERIMENT}")
        return
    if args.arm:
        total = int(args.steps or cfg["common"]["total_steps"])
        got = run_single_arm(cfg, args.arm, device, outdir, total)
        print(json.dumps({k: v for k, v in got.items() if k != "sanity"},
                         ensure_ascii=False), flush=True)
        return
    if args.stage == "smoke":
        run(cfg_path, cfg, device,
            Path(args.outdir or (Path(ROOT) / "results" / f"_smoke_{EXPERIMENT}")),
            args.substage, smoke=True)
        return
    run(cfg_path, cfg, device, outdir, args.substage)


if __name__ == "__main__":
    main()
