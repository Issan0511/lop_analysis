"""comb_isolate_0903 — 櫛の分離・段 A（spec `specs/spec_comb_isolate_0903.md`）。

    OMP_NUM_THREADS=1 python3 -m src.comb_isolate_0903 --stage preflight
    OMP_NUM_THREADS=1 python3 -m src.comb_isolate_0903 --stage run --arm CB1f_a1_1216
    OMP_NUM_THREADS=1 python3 -m src.comb_isolate_0903 --stage analyze

宿主は ``gate_dial_0902``（1 層・オラクル用量 12.16 固定・5M）で、腕の走らせ方は
``weird_act_0903._run_arm_weird``（宿主 ``_run_arm`` の写し・S-copy で検算済み）と
``WeirdRecorder``（6 列）を**そのまま import して使う**。宿主も weird_act も 1 行も変えない。

段 B（深さ 2）は ``src/comb_mlp2_0903.py``。**同じ commit で事前登録済み**（spec §3）。
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from .common import ROOT, load_config, pick_device
from .dose_const_5m import clopper_pearson
from .gate_dose import train_arm_gate
from .gate_dial_0902 import (SanityError, _arm, _arm_status_path, _ci, _draws,
                             _kaplan_meier, _load_new_arm, _sign_test,
                             setup_arm_dial, unit_extra_record)
from .mlp2_phase0 import _sha_file, require_omp, write_csv
from .mlp2_phase0b import _window_indices
from .mlp2_phase1 import NUMERIC_DIVERGENCE
from .nets import VecMLPL
from .ratchet_log import full_support_ro
from .weird_act_0903 import (ONSET_CENSOR_AT, WEIRD_UNIT_KEYS, WeirdRecorder,
                             _onset_stats, _onset_times, _phi2_extrema,
                             _run_arm_weird, _s_cap, _s_copy,
                             _s_pair_and_dose, unit_zmin_record)

EXPERIMENT = "comb_isolate_0903"
CONFIG = Path(ROOT) / "configs" / "comb_isolate_0903.yaml"

ARM_ORDER = ("CB1f_a1_1216", "CB1l_a1_1216", "RB_dpi_1216", "SN_a1_1216")
# 事前登録の腕定義（family, activation, dial）。validate_config が逐語照合する。
REGISTERED_ARMS = {
    "CB1f_a1_1216": ("comb1", "comb1_flat", 1.0),
    "CB1l_a1_1216": ("comb1", "comb1_leaky", 1.0),
    "RB_dpi_1216":  ("band", "band_leaky_dpi", 0.1),
    "SN_a1_1216":   ("snake", "snake", 1.0),
}
SMOKE_STEPS = 30_000
# S-limit の退化点（spec §7・§10 追補 1・2）
S_LIMIT_OPEN_CASES = (("comb1_leaky", 1.0, "comb_binf", 1.0),
                      ("comb1_flat", 1.0, "comb_binf", 1.0))
S_LIMIT_BIT_CASES = (("band_leaky_d0", 0.1, "leaky_relu", 0.1),)


def _P(cfg: dict) -> dict:
    return cfg["comb_isolate"]


def _lobe(alpha: float) -> float:
    return math.pi / float(alpha)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def validate_config(cfg: dict, *, stage: str) -> None:
    if stage not in {"preflight", "smoke", "run", "analyze", "finalize",
                     "diverge-probe"}:
        raise ValueError(f"unknown stage {stage!r}")
    C, A, I, P, G, S = (cfg["common"], cfg["condA"], cfg["intervention"],
                        cfg["phase1"], _P(cfg), cfg["sanity"])
    if [a["name"] for a in cfg["arms"]] != list(ARM_ORDER):
        raise ValueError(f"arms must be ordered as {ARM_ORDER}")
    for arm in cfg["arms"]:
        family, act, dial = REGISTERED_ARMS[arm["name"]]
        if (str(arm["family"]) != family or str(arm["activation"]) != act
                or float(arm["dial"]) != dial
                or str(arm["stage"]) != "A"
                or [int(v) for v in arm["hidden"]] != [100]
                or [int(v) for v in arm.get("centered_layers", [])] != [1]
                or float(arm["target_mu_norm"]) != 3.041
                or float(arm["target_dose"]) != 12.16):
            raise ValueError(f"{arm['name']} differs from the preregistration")
        if act not in VecMLPL.ACTIVATIONS:
            raise ValueError(f"{act} is not registered in VecMLPL.ACTIVATIONS")
        if str(cfg["activation"][act]["name"]) != act:
            raise ValueError(f"activation.{act}.name must be {act!r}")
    if [str(v) for v in cfg["staging"]["stageA_arms"]] != list(ARM_ORDER):
        raise ValueError("staging.stageA_arms changed")
    if str(cfg["staging"]["stageB_config"]) != "configs/comb_mlp2_0903.yaml":
        raise ValueError("stage B config path changed")
    if not bool(cfg["staging"]["stageB_frozen_before_stageA_results"]):
        raise ValueError("stage B must be frozen before stage A results")
    if bool(cfg["staging"]["stageC_registered"]):
        raise ValueError("stage C is explicitly NOT registered by this spec")
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
    expected_phase = {
        "task_period": 10_000, "early_tasks": [2, 11],
        "late_tasks_5m": [491, 500], "window_1m_tasks": [91, 100],
        "window_points_are_task_ends_only": True,
        "window_records_per_10task_window": 10,
        "onset_threshold": 0.05, "onset_present_min": 5,
        "unfit_floor": 1e-16, "recalibrate_floor": False,
        "bootstrap_B": 10_000, "bootstrap_seed": 20_260_915,
        "ci_method": "percentile_primary_studentized_secondary",
    }
    for key, value in expected_phase.items():
        if P[key] != value:
            raise ValueError(f"phase1.{key} differs from the preregistration")
    design = G["design"]
    threshold = float(design["freeze_depth_phi_prime_threshold"])
    want = (float(P["onset_threshold"])
            / (float(C["lr_main"]) * float(C["total_steps"])
               * float(design["displacement_bound_constant_K"])))
    if not math.isclose(threshold, want, rel_tol=1e-12):
        raise ValueError("the freeze-depth threshold is not 0.05/(lr*T*K)")
    if design["prescription"] is not False:
        raise ValueError("this spec is explicitly not a prescription")
    if list(G["verdict_order"]) != ["V5", "V6"]:
        raise ValueError("stage A registers V5 and V6 only (V7 is stage B)")
    E = G["exact_fit"]
    if (float(E["threshold"]) != 1e-8 or str(E["window"]) != "window_1m_tasks"
            or str(E["statistic"]) != "seed_median"
            or E["blocks_level_labels"] is not True
            or E["keeps_onset_labels"] is not True):
        raise ValueError("the EXACT_FIT guard differs from the preregistration")
    if list(G["v5"]["arms"]) != ["CB1f_a1_1216", "CB1l_a1_1216"]:
        raise ValueError("the V5 arms changed")
    if str(G["v6"]["committed_arm"]) != "CB_a1_1216":
        raise ValueError("the V6 committed arm changed")
    if float(G["p5_equivalence_margin"]) != 0.15 or float(G["v5"]["margin"]) != 0.15:
        raise ValueError("the equivalence margin is registered as 0.15 dex")
    if G["p3prime_baseline"] != "R_1216":
        raise ValueError("the P3' baseline is registered as R_1216")
    wanted_sanity = {"s_pair_steps": 30000, "s_limit_steps": 30000,
                     "s_dose_rel_tol": 1e-10, "s_fd_tol": 1e-6,
                     "s_mob_tol": 1e-6, "s_cap_min_seeds": 9,
                     "s_dial_rel_tol": 0.06, "omp_num_threads": 1,
                     "s_limit_snake_alpha": 1e-6, "s_limit_snake_rel_tol": 1e-4}
    for key, value in wanted_sanity.items():
        if S[key] != value:
            raise ValueError(f"sanity.{key} differs from the preregistration")
    for key in ("s_dial_check", "s_const_check", "s_guard_check", "s_taut_check",
                "s_mask_check", "s_cover_check", "s_limit_lobe_endpoint_excluded"):
        if S[key] is not True:
            raise ValueError(f"sanity.{key} must be true")
    if list(S["s_guard_must_fire"]) != ["CB_a1_1216"]:
        raise ValueError("S-guard's must-fire list changed")
    if S["s6_floor_calibration"] is not False:
        raise ValueError("the floor is inherited, never recalibrated")
    if stage in {"run", "analyze", "finalize"}:
        if int(C["total_steps"]) != 5_000_000:
            raise ValueError("total_steps is registered as 5,000,000")
        if [int(v) for v in C["seeds"]] != list(range(10)):
            raise ValueError("seeds are registered as 0..9")
        if str(C["device"]) != "cpu":
            raise ValueError("device is registered as cpu")


def _selected_arms(cfg: dict, which: str) -> list[str]:
    return list(ARM_ORDER) if which in ("all", "A") else [which]


# ---------------------------------------------------------------------------
# 前段チェック（spec §7）
# ---------------------------------------------------------------------------
def _probe_net(act: str, alpha: float) -> VecMLPL:
    net = VecMLPL(1, [2], 2, torch.Generator().manual_seed(0), "cpu")
    return net.set_activation(act, alpha, "alpha_exp")


def _kinks(act: str, alpha: float) -> list[float]:
    """折れ目（S-fd で ±1e-3 を除外する点）。`snake` は全域滑らかなので空。"""
    if act in VecMLPL.COMB1_ACTIVATIONS:
        return [0.0, -_lobe(alpha)]
    if act in VecMLPL.BAND_WIDTH:
        return [0.0, -VecMLPL.BAND_WIDTH[act]]
    if act == "snake":
        return []
    return [0.0]


def _s_dial(cfg: dict) -> dict:
    """S-dial: 登録した分水嶺と井戸を数値で解き直して照合（相対許容 6%）。

    * **分水嶺** = $\\varphi^2$ の極大（両族）
    * **井戸** = 葉の端。**$\\varphi$ の負側の零点の位置**で照合する。
      ``comb1_flat`` はそこが二重零点（$\\varphi'=0$）だが、``comb1_leaky`` は
      設計上そこで $\\varphi'=a$（戻り道が引き継ぐ）ので、$\\varphi'=0$ は
      **合否条件にせず REPORT に置く**。登録値 3.142 は $\\pi$ の 3 桁丸めなので、
      位置は相対許容で見る（``weird_act_0903._s_dial`` の極小探索は
      ``comb1_flat`` の平坦域で井戸を検出できないため、そのまま使えない）。
    """
    tol = float(cfg["sanity"]["s_dial_rel_tol"])
    rows, bad = [], []
    for arm in cfg["arms"]:
        act, alpha = str(arm["activation"]), float(arm["dial"])
        reg_max = [float(v) for v in arm.get("watershed") or []]
        reg_well = [float(v) for v in arm.get("well") or []]
        if not reg_max and not reg_well:
            rows.append(dict(arm=arm["name"], activation=act, registered_none=True,
                             pass_=True))
            continue
        net = _probe_net(act, alpha)
        umax = max(reg_max + reg_well) * 1.25 + 1.0
        got = _phi2_extrema(act, alpha, umax)
        ok = True
        for i, w in enumerate(reg_max):
            have = got["maxima"]
            if i >= len(have) or abs(have[i] - w) / w > tol:
                ok = False
        # 井戸 = 葉の端（phi の負側の零点）。密な格子で |phi| の谷を拾って位置を照合する。
        u = torch.linspace(1e-9, umax, 4_000_001, dtype=torch.float64)
        with torch.no_grad():
            phi = net.act_fn(-u)
        zeros = []
        absphi = phi.abs()
        mid = absphi[1:-1]
        local_min = (mid <= absphi[:-2]) & (mid <= absphi[2:]) & (mid < 1e-6)
        for v in u[1:-1][local_min].tolist():
            if not zeros or v - zeros[-1] > 1e-2:
                zeros.append(v)
        well_rows = []
        for w in reg_well:
            near = min(zeros, key=lambda v: abs(v - w)) if zeros else None
            rel = None if near is None else abs(near - w) / w
            t = torch.tensor([-w], dtype=torch.float64)
            with torch.no_grad():
                dphi = float(net.act_grad(t, net.act_fn(t)))
            well_rows.append(dict(registered_u=w, solved_zero_u=near,
                                  rel_error=rel, phi_prime_at_registered_u=dphi,
                                  is_double_zero=bool(abs(dphi) <= 1e-9)))
            if near is None or rel > tol:
                ok = False
        row = dict(arm=arm["name"], activation=act, dial=alpha,
                   registered_watershed=reg_max,
                   solved_maxima=[round(v, 4) for v in got["maxima"][:4]],
                   registered_wells=reg_well, wells=well_rows,
                   well_definition="position of the negative-side zero of phi; "
                                   "phi'=0 there is REPORT only (spec §2)",
                   pass_=bool(ok))
        rows.append(row)
        if not ok:
            bad.append(row)
    return dict(pass_=not bad, rel_tol=tol, rows=rows, failures=bad)


def _s_const(cfg: dict) -> dict:
    """S-const: config の第 2 母数の写しと ``nets.py`` のクラス定数が一致（spec §10 追補 3）。"""
    rows, bad = [], []
    for arm in cfg["arms"]:
        act, alpha = str(arm["activation"]), float(arm["dial"])
        block = cfg["activation"][act]
        want = arm.get("second_param")
        row = dict(arm=arm["name"], activation=act, arm_value=want)
        if act in VecMLPL.COMB1_ACTIVATIONS:
            code = _lobe(alpha)                       # nets は pi/act_alpha で解き直す
            row.update(nets_value=code, nets_source="math.pi / act_alpha",
                       config_block_value=float(block["second_param_value"]))
            ok = (want is not None
                  and math.isclose(float(want), code, rel_tol=1e-12)
                  and math.isclose(row["config_block_value"], code, rel_tol=1e-12))
            if act == "comb1_leaky":                  # 漏れ a も照合する
                leak_code = VecMLPL.COMB1_LEAK["comb1_leaky"]
                row.update(leak_nets=leak_code, leak_config=float(block["leak_a"]))
                ok = ok and math.isclose(leak_code, row["leak_config"], rel_tol=1e-12)
        elif act in VecMLPL.BAND_WIDTH:
            code = float(VecMLPL.BAND_WIDTH[act])
            row.update(nets_value=code, nets_source="BAND_WIDTH",
                       config_block_value=float(block["second_param_value"]))
            ok = (want is not None
                  and math.isclose(float(want), code, rel_tol=1e-12)
                  and math.isclose(row["config_block_value"], code, rel_tol=1e-12))
        elif act == "snake":
            row.update(nets_value=None, nets_source="none",
                       config_block_value=None)
            ok = (want is None and str(block["second_param"]) == "none"
                  and block["gate"] is False and block["monotone"] is True)
        else:
            row.update(nets_value=None, nets_source="unexpected")
            ok = False
        row["pass_"] = bool(ok)
        rows.append(row)
        if not ok:
            bad.append(row)
    return dict(pass_=not bad, rows=rows, failures=bad)


def _s_fd(cfg: dict) -> dict:
    """S-fd: 4 族の backward を float64 中心差分と照合。折れ目 ±1e-3 は除外。"""
    S = cfg["sanity"]
    tol, excl = float(S["s_fd_tol"]), float(S["s_fd_kink_exclusion"])
    lo, hi = [float(v) for v in S["s_fd_range"]]
    n = int(S["s_fd_points"])
    h = 1e-6
    rows, bad = [], []
    cases = [(str(a["activation"]), float(a["dial"])) for a in cfg["arms"]]
    cases += [(c[0], c[1]) for c in S_LIMIT_BIT_CASES]
    for act, alpha in cases:
        net = _probe_net(act, alpha)
        grid = [torch.linspace(lo, hi, n, dtype=torch.float64)]
        for k in _kinks(act, alpha):
            for off in [float(v) for v in S["s_fd_kink_offsets"]]:
                grid.append(torch.linspace(k - off, k + off, 21, dtype=torch.float64))
        z = torch.cat(grid)
        mask = torch.ones_like(z, dtype=torch.bool)
        for k in _kinks(act, alpha):
            mask &= (z - k).abs() > excl
        z = z[mask]
        with torch.no_grad():
            fd = (net.act_fn(z + h) - net.act_fn(z - h)) / (2.0 * h)
            g = net.act_grad(z, net.act_fn(z))
        worst = float((fd - g).abs().max())
        row = dict(activation=act, alpha=alpha, n_points=int(z.numel()),
                   worst_abs=worst, tol=tol, pass_=bool(worst <= tol))
        rows.append(row)
        if not row["pass_"]:
            bad.append(row)
    return dict(pass_=not bad, rows=rows, failures=bad)


def _s_num(cfg: dict) -> dict:
    """S-num: 登録範囲で NaN・inf が出ないこと。float32 の飽和・溢れ深さを記録する。"""
    S = cfg["sanity"]
    lo, hi = [float(v) for v in S["s_num_range"]]
    n = int(S["s_num_points"])
    rows, bad = [], []
    for arm in cfg["arms"]:
        act, alpha = str(arm["activation"]), float(arm["dial"])
        net = _probe_net(act, alpha)
        z = torch.linspace(lo, hi, n, dtype=torch.float32)
        with torch.no_grad():
            f = net.act_fn(z)
            g = net.act_grad(z, f)
        finite = bool(torch.isfinite(f).all() and torch.isfinite(g).all())
        over = None
        probe = torch.arange(0.0, 2000.0, 1.0, dtype=torch.float32)
        with torch.no_grad():
            fo = net.act_fn(-probe)
        idx = torch.nonzero(~torch.isfinite(fo)).flatten()
        if idx.numel():
            over = float(probe[idx[0]])
        rows.append(dict(arm=arm["name"], activation=act, alpha=alpha,
                         range=[lo, hi], finite=finite,
                         float32_overflow_depth=over,
                         max_abs=float(f.abs().max()), pass_=finite))
        if not finite:
            bad.append(rows[-1])
    return dict(pass_=not bad, rows=rows, failures=bad)


def _s_limit(cfg: dict, outdir: Path) -> dict:
    """S-limit: 退化点の一致（spec §7・§10 追補 1・2）。

    ``comb1_*`` は **開区間** $(-\\pi,\\infty)$ で ``comb_binf`` と bit 一致する。
    端点 $z=-\\pi$ は float64 で $\\sin(-\\pi)\\ne0$ のため一致しない（追補 1）。
    """
    S = cfg["sanity"]
    rows = []
    lobe = _lobe(1.0)
    open_grid = torch.linspace(-lobe + 1e-9, 30.0, 20001, dtype=torch.float64)
    for act, alpha, ref, ref_alpha in S_LIMIT_OPEN_CASES:
        a, b = _probe_net(act, alpha), _probe_net(ref, ref_alpha)
        with torch.no_grad():
            fa, fb = a.act_fn(open_grid), b.act_fn(open_grid)
            ga, gb = a.act_grad(open_grid, fa), b.act_grad(open_grid, fb)
        endpoint = torch.tensor([-lobe], dtype=torch.float64)
        with torch.no_grad():
            ea, eb = float(a.act_fn(endpoint)), float(b.act_fn(endpoint))
        rows.append(dict(
            kind="open_interval", activation=act, reference=ref,
            interval=f"(-{lobe:.6f}, 30]",
            forward_bit_equal=bool(torch.equal(fa, fb)),
            grad_bit_equal=bool(torch.equal(ga, gb)),
            endpoint_excluded=True, endpoint_new=ea, endpoint_reference=eb,
            pass_=bool(torch.equal(fa, fb) and torch.equal(ga, gb))))
    grid = torch.linspace(-30.0, 30.0, 24001, dtype=torch.float64)
    for act, alpha, ref, ref_alpha in S_LIMIT_BIT_CASES:
        a, b = _probe_net(act, alpha), _probe_net(ref, ref_alpha)
        with torch.no_grad():
            fa, fb = a.act_fn(grid), b.act_fn(grid)
            ga, gb = a.act_grad(grid, fa), b.act_grad(grid, fb)
        rows.append(dict(kind="closed_bit", activation=act, reference=ref,
                         forward_bit_equal=bool(torch.equal(fa, fb)),
                         grad_bit_equal=bool(torch.equal(ga, gb)),
                         pass_=bool(torch.equal(fa, fb) and torch.equal(ga, gb))))
    alpha = float(S["s_limit_snake_alpha"])
    tol = float(S["s_limit_snake_rel_tol"])
    sn = _probe_net("snake", alpha)
    with torch.no_grad():
        dev = ((sn.act_fn(grid) - grid).abs()
               / torch.clamp(grid.abs(), min=1.0)).max()
    rows.append(dict(kind="snake_identity_limit", activation="snake", alpha=alpha,
                     max_rel_dev=float(dev), tol=tol,
                     pass_=bool(float(dev) <= tol)))
    return dict(pass_=all(r["pass_"] for r in rows), rows=rows)


def _s_guard(cfg: dict) -> dict:
    """S-guard（新規・spec §7）: ``EXACT_FIT`` 閾値が committed 出力で正しく分離するか。

    **本走前に確認する。** 立たなければ閾値を動かさず ``GUARD_MISCALIBRATED``。
    """
    import csv as _csv

    G = _P(cfg)["exact_fit"]
    S = cfg["sanity"]
    threshold = float(G["threshold"])
    floor = float(cfg["phase1"]["unfit_floor"])
    want_fire = [str(v) for v in S["s_guard_must_fire"]]
    want_quiet = [str(v) for v in S["s_guard_must_not_fire"]]
    levels: dict[str, float] = {}
    for name, block in cfg["controls"]["arms"].items():
        path = Path(ROOT) / str(block["source_run"]) / "verdict.csv"
        if not path.exists():
            continue
        with path.open(newline="") as fh:
            for row in _csv.DictReader(fh):
                if row.get("arm") != name or not row.get("U_1m_seed_values"):
                    continue
                u = np.maximum(np.asarray(json.loads(row["U_1m_seed_values"]),
                                          dtype=np.float64), floor)
                levels[name] = float(np.median(u))
    rows, bad = [], []
    for name in want_fire + want_quiet:
        med = levels.get(name)
        fired = None if med is None else bool(med <= threshold)
        expect = name in want_fire
        ok = fired is not None and fired == expect
        rows.append(dict(arm=name, u_1m_seed_median=med, fired=fired,
                         expected=expect, pass_=ok))
        if not ok:
            bad.append(rows[-1])
    fire_max = max([r["u_1m_seed_median"] for r in rows
                    if r["expected"] and r["u_1m_seed_median"] is not None],
                   default=None)
    quiet_min = min([r["u_1m_seed_median"] for r in rows
                     if not r["expected"] and r["u_1m_seed_median"] is not None],
                    default=None)
    gap = (math.log10(quiet_min / fire_max)
           if fire_max and quiet_min and fire_max > 0 else None)
    return dict(pass_=not bad, threshold=threshold, rows=rows, failures=bad,
                separation_decades=gap,
                note="閾値は事後に動かさない。分離しなければ V6 は GUARD_MISCALIBRATED")


def _s_mob(cfg: dict, outdir: Path) -> dict:
    """S-mob: 新規ロガーが既知の量と一致すること＋ ``zmin <= zmean <= zmax``。"""
    S = cfg["sanity"]
    tol, steps = float(S["s_mob_tol"]), int(S["s_mob_steps"])
    c = copy.deepcopy(cfg)
    c["common"]["seeds"] = [0, 1]
    rows = []
    for arm_name in ("RB_dpi_1216", "CB1l_a1_1216"):
        block = _arm(c, arm_name)
        st = setup_arm_dial(copy.deepcopy(c), block, "cpu")
        every = int(c["common"]["lop_every"])
        probes = list(range(0, steps + 1, every))
        rec = WeirdRecorder(probes, st)
        train_arm_gate(st, rec, probes, steps, outdir, [])
        extra = unit_extra_record(st)
        zmin = unit_zmin_record(st)
        net = st["net"]
        flags = st.get("centered_layers") or [False]
        means = st.get("layer_means") or [None]
        with torch.no_grad():
            cur = full_support_ro(st["env"]).double()
            if flags[0]:
                cur = cur - means[0].double()[None]
            z = torch.einsum("rhd,prd->prh", net.Ws[0].double(),
                             cur) + net.bs[0].double()
            p_hat = (z > 0).double().mean(dim=0)
            if arm_name.startswith("RB_"):
                d = VecMLPL.BAND_WIDTH[net.act]
                a = float(net.act_alpha)
                want = p_hat + a * (z < -d).double().mean(dim=0)
                identity = "p_hat + a*Pr[z < -d]"
            else:
                # comb1_leaky: 葉の内側は -a*sin(2az)、葉の先は漏れ a_leak
                alpha = float(net.act_alpha)
                lobe = _lobe(alpha)
                leaf = (0.0 - alpha * torch.sin(2.0 * alpha * z))
                beyond = torch.full_like(z, VecMLPL.COMB1_LEAK["comb1_leaky"])
                grad = torch.where(z > 0, torch.ones_like(z),
                                   torch.where(z > -lobe, leaf, beyond))
                want = grad.mean(dim=0)
                identity = "E_x[phi'] rebuilt from the closed form"
        err = float((extra["mob"] - want).abs().max())
        order = bool((zmin <= extra["zmean"] + 1e-12).all()
                     and (extra["zmean"] <= extra["zmax"] + 1e-12).all())
        order_logged = bool((rec.unit["zmin"] <= rec.unit["zmean"] + 1e-5).all()
                            and (rec.unit["zmean"] <= rec.unit["zmax"] + 1e-5).all())
        rows.append(dict(arm=arm_name, identity=identity, max_abs_error=err,
                         zmin_le_zmean_le_zmax=order,
                         zmin_le_zmean_le_zmax_logged=order_logged,
                         pass_=bool(err <= tol and order and order_logged)))
    return dict(pass_=all(r["pass_"] for r in rows), tolerance=tol, steps=steps,
                rows=rows)


def _s_log_b(cfg: dict, outdir: Path) -> dict:
    """S-log-b: 追加ロガーの有無で既存の列がすべて bit 一致（軌道中立）。"""
    from .weird_act_0903 import _short_run_digest

    steps = int(cfg["sanity"]["s_log_b_steps"])
    c = copy.deepcopy(cfg)
    c["common"]["seeds"] = [0, 1]
    block = _arm(c, "CB1f_a1_1216")
    got = {}
    for label, units in (("with_logger", True), ("without_logger", False)):
        got[label] = _short_run_digest(c, block, outdir, steps, record_units=units)
    differences = [k for k in ("state", "env", "stream", "run", "layers", "extra",
                               "flip", "seed_hashes")
                   if got["with_logger"][k] != got["without_logger"][k]]
    added = sorted(got["with_logger"]["unit"])
    return dict(pass_=bool(not differences and added == sorted(WEIRD_UNIT_KEYS)),
                steps=steps, differences=differences, added_columns=added,
                expected_columns=sorted(WEIRD_UNIT_KEYS))


def _s_taut(cfg: dict, outdir: Path) -> dict:
    """S-taut: ``frozen`` が構成上恒真・恒偽になっていないか（判定式には未使用）。"""
    steps = 5_000
    c = copy.deepcopy(cfg)
    c["common"]["seeds"] = [0, 1]
    rows = []
    for arm_name in ("CB1f_a1_1216", "CB1l_a1_1216", "SN_a1_1216"):
        block = _arm(c, arm_name)
        st = setup_arm_dial(copy.deepcopy(c), block, "cpu")
        every = int(c["common"]["lop_every"])
        probes = list(range(0, steps + 1, every))
        rec = WeirdRecorder(probes, st)
        train_arm_gate(st, rec, probes, steps, outdir, [])
        mob, absmob = np.abs(rec.unit["mob"]), rec.unit["absmob"]
        zmax = rec.unit["zmax"]
        frozen = float((mob < 1e-6).mean())
        frozen_abs = float((absmob < 1e-6).mean())
        rows.append(dict(arm=arm_name, frozen_frac=frozen,
                         frozen_abs_frac=frozen_abs,
                         submerged_frac=float((zmax <= 0).mean()),
                         tautological=bool(frozen in (0.0, 1.0)
                                           and frozen_abs in (0.0, 1.0)),
                         note="frozen / frozen_abs は REPORT_ONLY。判定式に入れていない"))
    return dict(pass_=True, rows=rows,
                note="informational; SN は submerged を定義しない（spec §4）")


def preflight(cfg: dict, outdir: Path) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    checks: dict = {}
    checks["S_copy"] = _s_copy()
    checks["S_const"] = _s_const(cfg)
    checks["S_dial"] = _s_dial(cfg)
    checks["S_fd"] = _s_fd(cfg)
    checks["S_num"] = _s_num(cfg)
    checks["S_limit"] = _s_limit(cfg, outdir / "slimit")
    checks["S_guard"] = _s_guard(cfg)
    checks["S_log_b"] = _s_log_b(cfg, outdir / "slogb")
    checks["S_mob"] = _s_mob(cfg, outdir / "smob")
    pair = _s_pair_and_dose(cfg, outdir / "spair", list(ARM_ORDER))
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
# 集計（spec §6）
# ---------------------------------------------------------------------------
def _unit_tail(cfg: dict, outdir: Path, arm_block: dict,
               window: str = "late_tasks_5m") -> dict:
    """末尾窓のユニット別量。``SN`` は ``submerged`` を定義しない（spec §4）。"""
    P = cfg["phase1"]
    arm = str(arm_block["name"])
    act, alpha = str(arm_block["activation"]), float(arm_block["dial"])
    gated = act != "snake"
    per_seed, deciles = [], []
    for seed in [int(v) for v in cfg["common"]["seeds"]]:
        path = outdir / "logs" / f"{arm}_seed{seed}.npz"
        with np.load(path, allow_pickle=False) as z:
            idx = _window_indices(z["step"], int(P["task_period"]), list(P[window]))
            zmax = z["layer1_zmax"][idx].astype(np.float64)
            zmin = z["layer1_zmin"][idx].astype(np.float64)
            zmean = z["layer1_zmean"][idx].astype(np.float64)
            mob = z["layer1_mob"][idx].astype(np.float64)
            absmob = z["layer1_absmob"][idx].astype(np.float64)
            v_unit = z["layer1_v_unit"][idx].astype(np.float64)
        span, depth = zmax - zmin, -zmean
        sub = zmax <= 0.0
        row = dict(seed=seed, n_records=int(zmax.shape[0]),
                   span_median_all=float(np.median(span)),
                   absv_median_all=float(np.median(np.abs(v_unit))),
                   frozen_frac=float((np.abs(mob) < 1e-6).mean()),
                   frozen_abs_frac=float((absmob < 1e-6).mean()))
        if gated:
            row.update(
                submerged_frac=float(sub.mean()),
                span_median_submerged=float(np.median(span[sub])) if sub.any() else float("nan"),
                depth_median_submerged=float(np.median(depth[sub])) if sub.any() else float("nan"),
                absv_median_submerged=float(np.median(np.abs(v_unit)[sub])) if sub.any() else float("nan"))
        if act in VecMLPL.COMB1_ACTIVATIONS:
            row["at_well_frac"] = float((np.abs(zmean + _lobe(alpha)) <= 0.5).mean())
        if act in VecMLPL.BAND_WIDTH:
            d = VecMLPL.BAND_WIDTH[act]
            row["in_band_frac"] = float((sub & (zmin >= -d)).mean())
        per_seed.append(row)
        if gated and sub.any():
            deciles.append(np.quantile(depth[sub], np.arange(1, 10) / 10.0))
    out = dict(arm=arm, activation=act, dial=alpha, window=window,
               submerged_defined=gated, per_seed=per_seed,
               aggregation_order=["unit_records_within_seed", "median_within_seed",
                                  "median_over_seeds"])
    keys = set()
    for r in per_seed:
        keys.update(r)
    for key in sorted(keys - {"seed", "n_records"}):
        vals = [r[key] for r in per_seed
                if key in r and not (isinstance(r[key], float) and math.isnan(r[key]))]
        out[f"median_{key}"] = float(np.median(vals)) if vals else None
    out["depth_deciles_median"] = ([float(v) for v in np.median(np.stack(deciles), axis=0)]
                                   if deciles else None)
    return out


def _exact_fit(cfg: dict, u_1m: np.ndarray) -> dict:
    G = _P(cfg)["exact_fit"]
    med = float(np.median(np.asarray(u_1m, dtype=np.float64)))
    fired = bool(med <= float(G["threshold"]))
    return dict(fired=fired, u_1m_seed_median=med,
                threshold=float(G["threshold"]),
                label=str(G["label"]) if fired else "")


def _v5_label(cfg: dict, data: dict, contrasts: dict) -> tuple[str, dict]:
    """V5（spec §6）。`EXACT_FIT` の腕には水準条件を使わない。"""
    G = _P(cfg)["v5"]
    zero_max, present_min = 2, int(G["onset_present_min"])
    margin = float(G["margin"])
    f, l = "CB1f_a1_1216", "CB1l_a1_1216"
    if f not in data or l not in data or "RB_dpi_1216" not in data:
        return "", dict(status="INCOMPLETE_STAGE_A", present=sorted(data))
    nf = data[f]["5M"]["onset"]["n_onset"]
    nl = data[l]["5M"]["onset"]["n_onset"]

    def below(key):
        c = contrasts.get(key)
        if not c:
            return None
        lo, hi = c["ci"].get("percentile_ci_lo"), c["ci"].get("percentile_ci_hi")
        if lo is None or hi is None or not np.isfinite([lo, hi]).all():
            return None
        return bool(hi < -margin)

    detail = dict(n_onset_CB1f=nf, n_onset_CB1l=nl,
                  exact_fit=dict(CB1f=data[f]["exact_fit"]["fired"],
                                 CB1l=data[l]["exact_fit"]["fired"],
                                 RB_dpi=data["RB_dpi_1216"]["exact_fit"]["fired"]),
                  ci_CB1f_minus_R=below("V5:CB1f-R_1216"),
                  ci_CB1l_minus_RBdpi=below("V5:CB1l-RB_dpi_1216"),
                  equiv_CB1l_minus_RBdpi=(contrasts.get("V5:CB1l-RB_dpi_1216")
                                          or {}).get("equivalence"),
                  margin=margin)
    blocked = (data[l]["exact_fit"]["fired"]
               or data["RB_dpi_1216"]["exact_fit"]["fired"])
    if nf <= zero_max and below("V5:CB1f-R_1216") and not data[f]["exact_fit"]["fired"]:
        return "SINGLE_WELL_RESCUES", detail
    if nf >= present_min:
        if blocked:
            return "INCONCLUSIVE_EXACT_FIT", detail
        if nl <= zero_max and below("V5:CB1l-RB_dpi_1216"):
            return "WELL_HELPS_ONLY_WITH_RETURN_PATH", detail
        if detail["equiv_CB1l_minus_RBdpi"] == "EQUIV_SOFT":
            return "WELL_IRRELEVANT_RETURN_PATH_CARRIES", detail
        if nl >= present_min:
            return "LOBE_DOES_NOT_RESCUE", detail
    return "PARTIAL", detail


def _v6_label(cfg: dict, data: dict, committed_fired: bool | None) -> tuple[str, dict]:
    """V6（spec §6）。**`EXACT_FIT` の有無だけで付ける**（水準の差は使わない）。"""
    f, l = "CB1f_a1_1216", "CB1l_a1_1216"
    detail = dict(CB_a1_fired=committed_fired,
                  CB1l_fired=data.get(l, {}).get("exact_fit", {}).get("fired"),
                  CB1f_fired=data.get(f, {}).get("exact_fit", {}).get("fired"))
    if committed_fired is None or detail["CB1l_fired"] is None:
        return "", dict(status="INCOMPLETE", **detail)
    if not committed_fired:
        return "GUARD_MISCALIBRATED", detail
    if detail["CB1l_fired"]:
        return "LEVEL_FROM_SINGLE_LOBE", detail
    if detail["CB1f_fired"] is False:
        return "LEVEL_FROM_MULTILOBE", detail
    return "PARTIAL", detail


def divergence_probe(cfg: dict, arm: str, outdir: Path) -> dict:
    """発散腕の性質を回収する診断走（spec §6 の報告項目・**判定には使わない**）。

    宿主は発散時に部分ログを破棄するので、``arm_status/<arm>.json`` には検出 step しか
    残らない。**同じ config・同じ腕・登録どおりの seed 集合（R=10）**で刻んで回し直し、
    重みノルムがどこで非有限になるかを取る（seed を絞ると乱数の引き方が変わって別軌跡）。
    実装の誤りか学習の発散かを切り分けるためのもので、**新しい実験ではない**。
    """
    c = copy.deepcopy(cfg)
    st = setup_arm_dial(c, _arm(c, arm), "cpu")
    rows: list[dict] = []

    def probe(state: dict, step: int) -> None:
        net = state["net"]
        with torch.no_grad():
            rows.append(dict(step=int(step),
                             w_norm=float(net.Ws[0].norm()),
                             v_norm=float(net.v.norm()),
                             b_absmax=float(net.bs[0].abs().max()),
                             finite=bool(torch.isfinite(net.Ws[0]).all()
                                         and torch.isfinite(net.v).all())))

    scratch = outdir / "_divergence_probe"
    scratch.mkdir(parents=True, exist_ok=True)
    steps, every = 400, 25
    probes = list(range(0, steps + 1, every))
    raised = None
    try:
        train_arm_gate(st, probe, probes, steps, scratch, [])
    except Exception as exc:                     # 発散は登録どおり許容する腕なので握る
        raised = f"{type(exc).__name__}: {str(exc)[:200]}"
    first_bad = next((r["step"] for r in rows if not r["finite"]), None)
    out = dict(arm=arm, activation=st["activation"], act_alpha=st["act_alpha"],
               probe_every=every, probed_until=steps, rows=rows,
               first_nonfinite_probe_step=first_bad, raised=raised,
               note="diagnostic replay with the registered seed set; never used in a "
                    "verdict. The registered arm_status event records the detection "
                    "step on the 1000-step probe grid, which is coarser than this.")
    path = outdir / "arm_status" / f"{arm}_divergence_probe.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[divergence-probe] {arm}: first non-finite probe = {first_bad} "
          f"-> {path}", flush=True)
    return out


def _contrast(cfg: dict, a: np.ndarray, b: np.ndarray, draws: np.ndarray,
              label: str) -> dict:
    """seed クラスタの paired 差（log10 U の差）の中央値と CI・符号検定。

    ``weird_act_0903._contrast`` と同じ式だが、等価限界を**本走のブロック**
    （``comb_isolate.p5_equivalence_margin``）から読む。宿主の ``_P`` は
    ``cfg["weird_act"]`` を見るのでそのままでは使えない。
    """
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
    return dict(label=label, point=float(np.median(values)), ci=ci,
                sign_test=sign, equivalence=equiv, margin=margin,
                seed_values=[float(v) for v in values])


def _load_controls(cfg: dict) -> dict:
    """対照の endpoint を各親走の committed ``verdict.csv`` から**転記**する。"""
    import csv as _csv

    floor = float(cfg["phase1"]["unfit_floor"])
    out: dict[str, dict] = {}
    for name, block in cfg["controls"]["arms"].items():
        run = str(block["source_run"])
        path = Path(ROOT) / run / "verdict.csv"
        if not path.exists():
            raise SanityError(f"control {name}: {path} is missing")
        with path.open(newline="") as fh:
            for row in _csv.DictReader(fh):
                if row.get("arm") != name or not row.get("U_5m_seed_values"):
                    continue
                u5 = np.maximum(np.asarray(json.loads(row["U_5m_seed_values"]),
                                           dtype=np.float64), floor)
                u1 = np.maximum(np.asarray(json.loads(row["U_1m_seed_values"]),
                                           dtype=np.float64), floor)
                out[name] = dict(arm=name, source_run=run, source=str(path),
                                 u_5m=u5, u_1m=u1, log_u_5m=np.log10(u5),
                                 log_u_1m=np.log10(u1),
                                 n_onset_5m=int(row["n_onset_5m"]),
                                 n_onset_1m=int(row["n_onset_1m"]))
        if name not in out:
            raise SanityError(f"control {name} not found in {path}")
    return out


def analyze(cfg: dict, outdir: Path) -> dict:
    P, G = cfg["phase1"], _P(cfg)
    draws = _draws(cfg)
    controls = _load_controls(cfg)
    blocks = {a["name"]: a for a in cfg["arms"]}
    data: dict[str, dict] = {}
    diverged: dict[str, dict] = {}
    onset_rows, km_rows, position_rows, depth_rows = [], [], [], []
    for arm in ARM_ORDER:
        if not (outdir / "logs" / f"{arm}_seed0.npz").exists():
            event_path = _arm_status_path(outdir, arm)
            if event_path.exists():
                diverged[arm] = json.loads(event_path.read_text(encoding="utf-8"))
                print(f"[analyze] {arm}: {NUMERIC_DIVERGENCE} at step "
                      f"{diverged[arm]['detected_step']:,}, dropped", flush=True)
            else:
                print(f"[analyze] {arm}: logs missing, skipped", flush=True)
            continue
        w = _load_new_arm(cfg, outdir, arm)
        entry = {}
        for key in ("5M", "1M", "early"):
            u = np.asarray(w[key]["u"], dtype=np.float64)
            entry[key] = dict(u=u, log_u=np.log10(u), onset=_onset_stats(cfg, u),
                              metrics=w[key])
        entry["s_cap"] = _s_cap(cfg, {k: dict(u=entry[k]["u"])
                                      for k in ("early", "1M", "5M")})
        entry["exact_fit"] = _exact_fit(cfg, entry["1M"]["u"])
        entry["unit"] = _unit_tail(cfg, outdir, blocks[arm])
        ot = _onset_times(cfg, w["data"]["step"], w["data"]["unfit"])
        entry["onset_times"] = ot
        for j, (seed, row) in enumerate(zip([int(v) for v in cfg["common"]["seeds"]],
                                            ot["rows"])):
            onset_rows.append(dict(arm=arm, seed=seed, **row,
                                   frac_windows_over=ot["frac_windows_over"][j],
                                   u10_at_last_k=ot["u10_at_last_k"][j]))
        km_rows.extend(dict(arm=arm, **r) for r in _kaplan_meier(
            [r["k_star"] for r in ot["rows"]],
            [r["censored"] for r in ot["rows"]], ONSET_CENSOR_AT))
        u = entry["unit"]
        position_rows.append(dict(
            arm=arm, activation=u["activation"], dial=u["dial"],
            submerged_defined=u["submerged_defined"],
            median_submerged_frac=u.get("median_submerged_frac"),
            median_span_all=u["median_span_median_all"] if "median_span_median_all" in u
            else u.get("median_span_all"),
            median_span_submerged=u.get("median_span_median_submerged"),
            median_depth_submerged=u.get("median_depth_median_submerged"),
            median_absv_all=u.get("median_absv_median_all"),
            median_absv_submerged=u.get("median_absv_median_submerged"),
            at_well_frac=u.get("median_at_well_frac"),
            in_band_frac=u.get("median_in_band_frac"),
            median_frozen_frac=u.get("median_frozen_frac"),
            median_frozen_abs_frac=u.get("median_frozen_abs_frac"),
            window="late_tasks_5m (491-500, task-end records only)"))
        if u["depth_deciles_median"]:
            depth_rows.append(dict(arm=arm, **{f"d{i + 1}": v for i, v in
                                               enumerate(u["depth_deciles_median"])}))
        data[arm] = entry

    # --- 水準の対比（E2 と V5 の材料） ---
    contrasts: dict[str, dict] = {}
    rows = []
    def add(key, arm, against, window, a_log, b_log, kind):
        c = _contrast(cfg, a_log, b_log, draws, f"{arm} - {against} ({window})")
        contrasts[key] = c
        rows.append(dict(key=key, arm=arm, kind=kind, window=window,
                         against=against, **c))
    for arm, entry in data.items():
        family = str(blocks[arm]["family"])
        base = str(G["p3prime_baseline"])
        for window, ckey in (("5M", "log_u_5m"), ("1M", "log_u_1m")):
            add(f"P3':{arm}:{window}", arm, base, window,
                entry[window]["log_u"], controls[base][ckey], "P3prime")
        for soft in [str(v) for v in G["p5_soft_end_by_family"].get(family, [])]:
            add(f"P5':{arm}:{soft}", arm, soft, "5M",
                entry["5M"]["log_u"], controls[soft]["log_u_5m"], "P5prime")
    if "CB1f_a1_1216" in data:
        add("V5:CB1f-R_1216", "CB1f_a1_1216", "R_1216", "5M",
            data["CB1f_a1_1216"]["5M"]["log_u"], controls["R_1216"]["log_u_5m"], "V5")
    if "CB1l_a1_1216" in data and "RB_dpi_1216" in data:
        add("V5:CB1l-RB_dpi_1216", "CB1l_a1_1216", "RB_dpi_1216", "5M",
            data["CB1l_a1_1216"]["5M"]["log_u"], data["RB_dpi_1216"]["5M"]["log_u"], "V5")
    if "CB1l_a1_1216" in data:
        add("CB1l-CB_a1_1216", "CB1l_a1_1216", "CB_a1_1216", "5M",
            data["CB1l_a1_1216"]["5M"]["log_u"], controls["CB_a1_1216"]["log_u_5m"],
            "multilobe")

    committed_fired = _exact_fit(cfg, controls["CB_a1_1216"]["u_1m"])
    verdicts: dict[str, object] = {}
    verdicts["V5"], verdicts["V5_detail"] = _v5_label(cfg, data, contrasts)
    verdicts["V6"], verdicts["V6_detail"] = _v6_label(cfg, data,
                                                      committed_fired["fired"])
    verdicts["V7"] = None
    verdicts["V7_status"] = "STAGE_B (comb_mlp2_0903)"
    verdicts["exact_fit"] = {a: e["exact_fit"] for a, e in data.items()}
    verdicts["exact_fit"]["CB_a1_1216 (committed)"] = committed_fired
    verdicts["divergences"] = {k: dict(detected_step=v["detected_step"],
                                       bad_seeds=v.get("bad_seeds"))
                               for k, v in diverged.items()}
    result = dict(experiment=EXPERIMENT, arms=list(data), diverged=list(diverged),
                  verdicts=verdicts)
    _write_outputs(cfg, outdir, data, controls, rows, verdicts, onset_rows,
                   km_rows, position_rows, depth_rows, diverged, committed_fired)
    return result


def _write_outputs(cfg, outdir, data, controls, contrast_rows, verdicts,
                   onset_rows, km_rows, position_rows, depth_rows, diverged,
                   committed_fired) -> None:
    blocks = {a["name"]: a for a in cfg["arms"]}
    G = _P(cfg)
    v_rows = []
    for arm, e in data.items():
        b = blocks[arm]
        v_rows.append(dict(
            arm=arm, stage="A", family=str(b["family"]),
            activation=str(b["activation"]), dial=float(b["dial"]),
            second_param=b.get("second_param"), target_dose=float(b["target_dose"]),
            is_control=False, status="COMPLETE",
            capacity_status=e["s_cap"]["status"],
            EXACT_FIT=e["exact_fit"]["label"],
            u_1m_seed_median=e["exact_fit"]["u_1m_seed_median"],
            n_onset_1m=e["1M"]["onset"]["n_onset"],
            cp95_1m_lo=e["1M"]["onset"]["cp95_lo"],
            cp95_1m_hi=e["1M"]["onset"]["cp95_hi"],
            U_1m_seed_values=json.dumps([float(v) for v in e["1M"]["u"]]),
            median_log10_U_1m=e["1M"]["onset"]["median_log10_u"],
            n_onset_5m=e["5M"]["onset"]["n_onset"],
            cp95_5m_lo=e["5M"]["onset"]["cp95_lo"],
            cp95_5m_hi=e["5M"]["onset"]["cp95_hi"],
            U_5m_seed_values=json.dumps([float(v) for v in e["5M"]["u"]]),
            median_log10_U_5m=e["5M"]["onset"]["median_log10_u"],
            median_submerged_frac_5m=e["unit"].get("median_submerged_frac"),
            median_span_all_5m=e["unit"].get("median_span_median_all"),
            at_well_frac=e["unit"].get("median_at_well_frac"),
            in_band_frac=e["unit"].get("median_in_band_frac"),
            V5=verdicts.get("V5") if arm in list(G["v5"]["arms"]) else "",
            V6=verdicts.get("V6") if arm in list(G["v6"]["new_arms"]) else "",
            NUMERIC_DIVERGENCE=""))
    for arm, event in diverged.items():
        b = blocks[arm]
        v_rows.append(dict(
            arm=arm, stage="A", family=str(b["family"]),
            activation=str(b["activation"]), dial=float(b["dial"]),
            second_param=b.get("second_param"), target_dose=float(b["target_dose"]),
            is_control=False, status=NUMERIC_DIVERGENCE, capacity_status="",
            EXACT_FIT="", u_1m_seed_median="",
            NUMERIC_DIVERGENCE=json.dumps(dict(
                detected_step=event["detected_step"],
                bad_seeds=event.get("bad_seeds")), ensure_ascii=False)))
    for name, c in controls.items():
        fired = _exact_fit(cfg, c["u_1m"])
        v_rows.append(dict(
            arm=name, stage="", family="", activation="", dial="",
            second_param="", target_dose=12.16, is_control=True,
            status="TRANSCRIBED", capacity_status="",
            EXACT_FIT=fired["label"], u_1m_seed_median=fired["u_1m_seed_median"],
            n_onset_1m=c["n_onset_1m"], cp95_1m_lo="", cp95_1m_hi="",
            U_1m_seed_values=json.dumps([float(v) for v in c["u_1m"]]),
            median_log10_U_1m=float(np.median(c["log_u_1m"])),
            n_onset_5m=c["n_onset_5m"], cp95_5m_lo="", cp95_5m_hi="",
            U_5m_seed_values=json.dumps([float(v) for v in c["u_5m"]]),
            median_log10_U_5m=float(np.median(c["log_u_5m"])),
            V5="", V6="", NUMERIC_DIVERGENCE=""))
    columns: list[str] = []
    for row in v_rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    write_csv(outdir / "verdict.csv",
              [{k: row.get(k, "") for k in columns} for row in v_rows])
    if onset_rows:
        write_csv(outdir / "onset_times.csv", onset_rows)
    if km_rows:
        write_csv(outdir / "onset_km.csv", km_rows)
    if position_rows:
        write_csv(outdir / "position_table.csv", position_rows)
    if depth_rows:
        write_csv(outdir / "depth_hist.csv", depth_rows)
    if contrast_rows:
        write_csv(outdir / "layer_stats.csv", [dict(
            key=c["key"], arm=c["arm"], kind=c["kind"], window=c["window"],
            against=c["against"], point=c["point"],
            percentile_lo=c["ci"].get("percentile_ci_lo"),
            percentile_hi=c["ci"].get("percentile_ci_hi"),
            studentized_lo=c["ci"].get("studentized_ci_lo"),
            studentized_hi=c["ci"].get("studentized_ci_hi"),
            ci_degenerate=c["ci"].get("ci_degenerate"),
            equivalence=c["equivalence"], margin=c["margin"],
            sign_pos=c["sign_test"]["n_positive"],
            sign_neg=c["sign_test"]["n_negative"],
            sign_p=c["sign_test"]["p_two_sided"],
            seed_values=json.dumps(c["seed_values"])) for c in contrast_rows])
    _write_summary(cfg, outdir, data, controls, contrast_rows, verdicts, diverged,
                   committed_fired)


def _write_summary(cfg, outdir, data, controls, contrast_rows, verdicts, diverged,
                   committed_fired) -> None:
    G = _P(cfg)
    L = [f"# {EXPERIMENT} — 櫛の分離・段 A（1 層・用量 12.16・5M）\n",
         f"spec: `{cfg['spec']}` / 事前登録 commit で凍結。"
         "数値の引用は `verdict.csv` と本ファイルからのみ。\n",
         "**★ §5.1 の Issa 事前予測は §7.2（Claude）と逐語で同一で、独立の予言ではない。**"
         "結果を引くときは「起草側の予測（Issa 承認）」と 1 行で書く"
         "（`preregistration.prediction_provenance`・引用禁止 B の `lr_a1_0901` 先例）。\n",
         "## S-cover（§6 の各項目 → 実装の対応先）\n",
         "| §6 の項目 | 実装 | 出力 | 段 A で付くか |", "| --- | --- | --- | --- |"]
    for item, impl, out, ok in (
            ("V5 井戸 1 個は救うか", "_v5_label", "verdict.csv:V5", "○"),
            ("V6 水準の帰属", "_v6_label（EXACT_FIT の有無だけ）", "verdict.csv:V6", "○"),
            ("V7 深さ 2", "src/comb_mlp2_0903.py", "results/comb_mlp2_0903/", "×（段 B）"),
            ("EXACT_FIT ガード", "_exact_fit / _s_guard", "verdict.csv:EXACT_FIT", "○"),
            ("E1 発症数", "_onset_stats", "verdict.csv:n_onset_*", "○"),
            ("E2 水準 P3'/P5'", "_contrast", "layer_stats.csv", "○"),
            ("E3 発症時刻 k*", "_onset_times / _kaplan_meier",
             "onset_times.csv / onset_km.csv", "○"),
            ("at_well・in_band・frozen", "_unit_tail", "position_table.csv", "○"),
            ("全ユニット span と |v|", "_unit_tail", "position_table.csv", "○"),
            ("深さ十分位", "_unit_tail", "depth_hist.csv", "○"),
            ("C1 の再現", "未実装（走後の別解析）", "—", "×")):
        L.append(f"| {item} | {impl} | {out} | {ok} |")
    L.append("\n**★ 未実装 1 件**: §6 副次の「C1 の再現」は REPORT_ONLY で判定には入らないが、"
             "spec が「本走で `src/` に置く」と書いているので**未了である**"
             "（`weird_act_0903` から持ち越し）。\n")
    L.append("## 判定\n")
    L.append("| 判定 | ラベル | 腕 |")
    L.append("| --- | --- | --- |")
    L.append(f"| V5 | {verdicts.get('V5') or '—'} | CB1f / CB1l 対 R_1216 / RB_dpi |")
    L.append(f"| V6 | {verdicts.get('V6') or '—'} | CB_a1（committed）対 CB1l / CB1f |")
    L.append(f"| V7 | — | {verdicts.get('V7_status')} |")
    L.append("\n**V5・V6・V7 は互いに独立の判定で、1 つに畳まない。**"
             "「ゲートで非線形性を買う代償」は §1 の**動機**であってラベルではない（spec §9）。\n")
    L.append("## `EXACT_FIT`（1M 窓 U の seed 中央値 ≤ 1e−8）\n")
    L.append("| 腕 | 1M 窓 U 中央値 | EXACT_FIT |")
    L.append("| --- | --- | --- |")
    for arm, e in data.items():
        L.append(f"| `{arm}` | {e['exact_fit']['u_1m_seed_median']:.4e} | "
                 f"{'**立つ**' if e['exact_fit']['fired'] else '立たない'} |")
    L.append(f"| `CB_a1_1216`（committed） | {committed_fired['u_1m_seed_median']:.4e} | "
             f"{'**立つ**' if committed_fired['fired'] else '立たない'} |")
    for name, c in controls.items():
        if name == "CB_a1_1216":
            continue
        f = _exact_fit(cfg, c["u_1m"])
        L.append(f"| `{name}`（committed） | {f['u_1m_seed_median']:.4e} | "
                 f"{'**立つ**' if f['fired'] else '立たない'} |")
    L.append("\n**`EXACT_FIT` の腕の水準差を機構として引かない**（spec §9）。"
             "$n_{\\rm onset}$ は引ける。\n")
    L.append("## 腕（末尾窓 = タスク 491–500 のタスク終端 10 点）\n")
    L.append("| 腕 | 活性化 | S-cap | n_onset 1M | n_onset 5M | median log10 U (5M) | "
             "沈下率 | span 中央値（全ユニット） | at_well | in_band |")
    L.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")

    def _f(x):
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return "—"
        return f"{x:.4g}"
    for arm, e in data.items():
        u = e["unit"]
        L.append(f"| `{arm}` | {u['activation']} | {e['s_cap']['status']} | "
                 f"{e['1M']['onset']['n_onset']}/10 | {e['5M']['onset']['n_onset']}/10 | "
                 f"{e['5M']['onset']['median_log10_u']:.4f} | "
                 f"{_f(u.get('median_submerged_frac'))} | "
                 f"{_f(u.get('median_span_median_all'))} | "
                 f"{_f(u.get('median_at_well_frac'))} | "
                 f"{_f(u.get('median_in_band_frac'))} |")
    L.append("\n**`SN`（Snake）は負側に壁が無いので `submerged` を定義しない**（spec §4）。\n")
    if diverged:
        L.append("## 数値発散（spec §6）\n")
        for arm, event in diverged.items():
            bad = event.get("bad_seeds") or []
            L.append(f"- **`{arm}` は `{NUMERIC_DIVERGENCE}`**。最初の発散 step "
                     f"**{int(event['detected_step']):,}**・発症 seed {len(bad)}/10"
                     f"（seed {bad}）。登録どおり当該腕だけを落とした。"
                     f"`SN` は錨なので判定腕ではない\n")
    L.append("## 水準の対比（対照は**別走の committed 値**・同一走の腕ではない）\n")
    L.append("| 鍵 | 腕 | 種別 | 窓 | 相手 | 点推定 | percentile CI | 等価判定 | 符号 |")
    L.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for c in contrast_rows:
        ci = c["ci"]
        lo, hi = ci.get("percentile_ci_lo"), ci.get("percentile_ci_hi")
        s = "—" if lo is None or hi is None else f"[{lo:+.3f}, {hi:+.3f}]"
        L.append(f"| `{c['key']}` | `{c['arm']}` | {c['kind']} | {c['window']} | "
                 f"`{c['against']}` | {c['point']:+.4f} | {s} | {c['equivalence']} | "
                 f"{c['sign_test']['n_negative']}:{c['sign_test']['n_positive']} |")
    L.append("\n### 対照の出所\n")
    for name, c in controls.items():
        L.append(f"- `{name}`: {c['source_run']} / `verdict.csv`")
    L.append("\n## 引用上の注意\n")
    L.append("- 0/10 は「5M までに観測しなかった」（片側 95% 上限 0.2589）。「起きない」と書かない")
    L.append("- **用量 1 点（12.16）・1 層・5M・float32 の主張である。** 引くときは用量を添える")
    L.append("- 4 族すべて本走のための合成活性化。`SN`（Snake）は先行があるが**推奨として引かない**")
    L.append("- **`EXACT_FIT` の腕の水準差を機構として引かない**")
    L.append("- **§5.1 の予測は独立の予言ではない**（起草側の値を Issa が承認したもの）")
    L.append("- 段 C（実ベンチ）は本 spec の外。段 A・B の結果から段 C を自動起案しない")
    (outdir / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Provenance / run
# ---------------------------------------------------------------------------
def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _provenance(cfg_path: Path, cfg: dict, outdir: Path, arms: list[str],
                sanity: dict, elapsed: float, started: str) -> dict:
    names = ("verdict.csv", "summary.md", "onset_times.csv", "onset_km.csv",
             "position_table.csv", "depth_hist.csv", "layer_stats.csv",
             "config_used.yaml")
    hashes = {n: _sha_file(outdir / n) for n in names if (outdir / n).exists()}
    hashes.update({f"logs/{p.name}": _sha_file(p)
                   for p in sorted((outdir / "logs").glob("*.npz"))})
    hashes.update({f"arm_status/{p.name}": _sha_file(p)
                   for p in sorted((outdir / "arm_status").glob("*.json"))})
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
        stage="A", stageB_config=str(cfg["staging"]["stageB_config"]),
        stages_registered=dict(cfg["staging"]),
        stage_note="stage A and stage B are preregistered in the same commit; "
                   "submission is sequential (2026-09-03 Issa).",
        prediction_provenance="draft_values_proposed_first_then_approved_by_Issa; "
                              "the §7.1 entry is verbatim identical to §7.2 (Claude) "
                              "and is NOT an independent prediction",
        arms_run=list(arms), dose="12.16",
        host="gate_dial_0902._run_arm via weird_act_0903._run_arm_weird (S-copy checked)",
        unit_columns=[f"layer1_{k}" for k in WEIRD_UNIT_KEYS],
        generator_offset=int(cfg["common"]["generator_offset"]),
        generator_offset_note="explicit 0: shares the parent run's stream (S-pair).",
        exact_fit_guard=dict(cfg["comb_isolate"]["exact_fit"]),
        parent_logs_absent_on_this_machine=(
            sanity.get("checks", {}).get("S_pair", {}).get("parent_status")
            == "PARENT_LOGS_ABSENT_ON_THIS_MACHINE"),
        sanity=sanity, output_sha256=hashes)


def run_single_arm(cfg: dict, arm: str, device: str, outdir: Path,
                   total: int) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    seeds = [int(v) for v in cfg["common"]["seeds"]]
    return _run_arm_weird(cfg, arm, device, outdir, seeds, total)


def finalize(cfg_path: Path, cfg: dict, outdir: Path) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / "config_used.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, allow_unicode=True, sort_keys=False)
    pre_path = Path(ROOT) / "results" / f"_preflight_{EXPERIMENT}" / "preflight.json"
    pre = json.loads(pre_path.read_text(encoding="utf-8"))
    if not pre.get("pass_"):
        raise SanityError(f"preflight did not pass: {pre_path}")
    seeds = [int(v) for v in cfg["common"]["seeds"]]
    statuses, divergences, elapsed = {}, {}, 0.0
    for arm in ARM_ORDER:
        done = outdir / "arm_status" / f"{arm}_done.json"
        div = _arm_status_path(outdir, arm)
        if div.exists():
            divergences[arm] = json.loads(div.read_text(encoding="utf-8"))
            statuses[arm] = NUMERIC_DIVERGENCE
        elif done.exists():
            got = json.loads(done.read_text(encoding="utf-8"))
            statuses[arm] = got.get("status")
            elapsed = max(elapsed, float(got.get("wall_sec") or 0.0))
        else:
            statuses[arm] = "MISSING"
        missing = [s for s in seeds
                   if not (outdir / "logs" / f"{arm}_seed{s}.npz").exists()]
        if missing and statuses[arm] == "COMPLETE":
            statuses[arm] = f"INCOMPLETE_LOGS:{missing}"
    prov = _provenance(cfg_path, cfg, outdir, list(ARM_ORDER), pre, elapsed,
                       time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    prov["arm_status"] = statuses
    prov["divergences"] = divergences
    prov["arm_process_parallel"] = True
    (outdir / "provenance.json").write_text(
        json.dumps(prov, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")
    print(f"[finalize] {statuses}", flush=True)
    return prov


def main() -> None:
    ap = argparse.ArgumentParser(description=EXPERIMENT)
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--stage", default="preflight",
                    choices=["preflight", "run", "analyze", "finalize",
                             "diverge-probe"])
    ap.add_argument("--arm", default=None, help="run exactly one arm (process parallel)")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--steps", type=int, default=None)
    args = ap.parse_args()
    cfg_path = Path(args.config)
    cfg = load_config(str(cfg_path))
    validate_config(cfg, stage=args.stage)
    require_omp(cfg)
    device = pick_device(cfg) if args.stage != "preflight" else "cpu"
    outdir = Path(args.outdir) if args.outdir else Path(ROOT) / cfg["output"]["dir"]
    if args.stage == "preflight":
        preflight(cfg, Path(ROOT) / "results" / f"_preflight_{EXPERIMENT}")
        return
    if args.stage == "diverge-probe":
        divergence_probe(cfg, args.arm or "SN_a1_1216", outdir)
        return
    if args.stage == "analyze":
        got = analyze(cfg, outdir)
        print(json.dumps(got["verdicts"], ensure_ascii=False, indent=1, default=str),
              flush=True)
        return
    if args.stage == "finalize":
        finalize(cfg_path, cfg, outdir)
        return
    if not args.arm:
        raise SystemExit("--stage run requires --arm (arms are submitted in parallel)")
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
    path.write_text(json.dumps(done, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(done, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
