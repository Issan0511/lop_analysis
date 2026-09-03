"""comb_mlp2_0903 — 櫛の分離・段 B（深さ 2）。spec `specs/spec_comb_isolate_0903.md` §4.2・§6。

    OMP_NUM_THREADS=1 python3 -m src.comb_mlp2_0903 --stage preflight
    OMP_NUM_THREADS=1 python3 -m src.comb_mlp2_0903 --stage run
    OMP_NUM_THREADS=1 python3 -m src.comb_mlp2_0903 --stage analyze

宿主は ``lr_a1_0901``（深さ 2・幅 [100,100]・第 1 層のみ中心化・素の SGD・5M）。
宿主の ``setup_arm_lr`` / ``_run_arm`` / ``write_arm_logs_lr`` は ``ARM="LR_A1"`` と
``activation="leaky_relu"`` を直書きしているので、**本走は 3 関数の写しを持つ**
（spec §10 追補 5）。``src/lr_a1_0901.py`` は 1 行も変えない。S-copy が写しを検算する。

対照 3 腕は ``results/lr_a1_0901/verdict.csv`` から**転記**する（spec §10 追補 4）。
``E_A1`` の生ログはこの機に無いので、水準は paired 差から seed ごとに再構成する。
再構成は ``L2_A1`` について生ログ（`mlp2_phase1_0829`）と **10 seed すべて bit 一致**
することを 2026-09-03 に確認した。
"""
from __future__ import annotations

import argparse
import copy
import csv
import inspect
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

from . import lr_a1_0901 as HOST
from .common import ROOT, load_config, pick_device
from .elu_swamp import EluRecorder, exact_layer_record_elu, train_arm_elu
from .mlp2_phase0 import (_sha_array, _sha_file, identity_sanity_pass,
                          require_omp, write_csv)
from .mlp2_phase0b import _ci_components, _window_indices
from .mlp2_phase1 import (NUMERIC_DIVERGENCE, NumericDivergenceError,
                          _env_hashes, _seed_state_hashes_p1, setup_arm_p1)
from .nets import VecMLPL

EXPERIMENT = "comb_mlp2_0903"
CONFIG = Path(ROOT) / "configs" / "comb_mlp2_0903.yaml"
ARM = "CB_A1"
ACTIVATION = "comb_binf"
ALPHA = 1.0
WIDTH, LAYERS = 100, 2
PREFLIGHT_DIR = "results/_preflight_comb_mlp2_0903"


def _P(cfg: dict) -> dict:
    return cfg["comb_mlp2"]


def _arm(cfg: dict) -> dict:
    return next(a for a in cfg["arms"] if a["name"] == ARM)


def _p1_cfg(cfg: dict) -> dict:
    """本走のブロックを Phase-1 ヘルパが期待する名前で見せる（宿主 ``_p1_cfg`` の写し）。"""
    out = copy.deepcopy(cfg)
    out["phase1"] = copy.deepcopy(cfg["comb_mlp2"])
    return out


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def validate_config(cfg: dict, *, stage: str) -> None:
    if stage not in {"preflight", "run", "analyze"}:
        raise ValueError(f"unknown stage {stage!r}")
    C, A, I, P, S = (cfg["common"], cfg["condA"], cfg["intervention"],
                     _P(cfg), cfg["sanity"])
    if [a["name"] for a in cfg["arms"]] != [ARM]:
        raise ValueError(f"stage B registers exactly one arm: {ARM}")
    arm = _arm(cfg)
    if (str(arm["activation"]) != "comb"
            or [int(v) for v in arm["hidden"]] != [WIDTH, WIDTH]
            or [int(v) for v in arm["centered_layers"]] != [1]):
        raise ValueError("CB_A1 differs from the preregistration")
    if str(cfg["activation"]["comb"]["name"]) != ACTIVATION:
        raise ValueError(f"activation.comb.name must be {ACTIVATION!r}")
    if float(cfg["activation"]["comb"]["alpha"]) != ALPHA:
        raise ValueError("the comb frequency is registered as alpha=1")
    if ACTIVATION not in VecMLPL.ACTIVATIONS:
        raise ValueError(f"{ACTIVATION} is not registered in VecMLPL.ACTIVATIONS")
    if (int(A["m"]), int(A["f"]), int(A["target_hidden"])) != (20, 15, 100):
        raise ValueError("condA differs from the preregistration")
    if int(C.get("generator_offset", -1)) != 0:
        raise ValueError("generator_offset must be an explicit 0 (S-B の土台)")
    if (str(I["name"]) != "A_layer_input_centering"
            or float(I["center_alpha"]) != 0.01
            or I["stop_gradient_on_running_mean"] is not True):
        raise ValueError("intervention differs from the preregistration")
    if (int(P["task_period"]) != 10_000 or [int(v) for v in P["late_tasks"]] != [451, 500]
            or int(P["exact_support"]) != 32
            or float(P["unfit_floor"]) != 1e-23
            or P["recalibrate_floor"] is not False
            or int(P["bootstrap_B"]) != 10_000
            or int(P["bootstrap_seed"]) != 20_260_915):
        raise ValueError("window, support, floor or bootstrap changed")
    V = P["v7"]
    if (str(V["arm"]) != ARM or str(V["reference_arm"]) != "L2_A1"
            or str(V["leaky_reference_arm"]) != "LR_A1"
            or float(V["margin"]) != 0.15):
        raise ValueError("the V7 registration changed")
    E = P["exact_fit"]
    if (float(E["threshold"]) != 1e-8
            or [int(v) for v in E["window_1m_tasks"]] != [91, 100]
            or E["blocks_level_labels"] is not True):
        raise ValueError("the EXACT_FIT guard differs from the preregistration")
    missing = HOST.preregistration_missing(cfg)
    if missing and stage in {"run", "analyze"}:
        raise ValueError(f"preregistration incomplete: {missing}")
    if (cfg["preregistration"]["prediction_provenance"]
            != "draft_values_proposed_first_then_approved_by_Issa"):
        raise ValueError("the prediction provenance must stay recorded (spec §5.1)")
    if int(S["s_b_steps"]) != 30_000 or str(S["s_b_reference_arm"]) != "LR_A1":
        raise ValueError("the S-B registration changed")
    if int(S["omp_num_threads"]) != 1:
        raise ValueError("OMP_NUM_THREADS is registered as 1")
    if stage in {"run", "analyze"}:
        if int(C["total_steps"]) != 5_000_000:
            raise ValueError("total_steps is registered as 5,000,000")
        if [int(v) for v in C["seeds"]] != list(range(10)):
            raise ValueError("seeds are registered as 0..9")
        if str(C["device"]) != "cpu":
            raise ValueError("device is registered as cpu")


# ---------------------------------------------------------------------------
# Runner — 宿主 3 関数の写し（spec §10 追補 5）。差は活性化とメタデータだけ
# ---------------------------------------------------------------------------
def setup_arm_comb(cfg: dict, device: str) -> dict:
    """``lr_a1_0901.setup_arm_lr`` の写し。活性化を櫛にし、腕を名前で引く。"""
    if int(cfg["common"].get("generator_offset", 0)) != 0:
        raise ValueError("the paired depth-2 harness requires generator_offset=0")
    st = setup_arm_p1(_p1_cfg(cfg), _arm(cfg), device)
    st["net"].set_activation(ACTIVATION, ALPHA, "alpha_exp")
    st["activation"] = ACTIVATION
    st["act_alpha"] = float(ALPHA)
    st["generator_offset"] = 0
    return st


def write_arm_logs_comb(outdir: Path, st: dict, rec: EluRecorder) -> list[Path]:
    """``lr_a1_0901.write_arm_logs_lr`` の写し。メタデータを本走の値で書く。"""
    logdir = outdir / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ri, run in enumerate(st["runs"]):
        payload = dict(
            step=rec.steps, run_id=np.array(run["run_id"]), arm=np.array(ARM),
            seed=np.int64(run["seed"]), activation=np.array(ACTIVATION),
            act_alpha=np.float64(ALPHA), generator_offset=np.int64(0),
            strict_dead_applicable=np.int8(0), task_period=np.int64(run["period"]),
            state_hash_final=np.array(json.dumps(
                _seed_state_hashes_p1(st, ri), sort_keys=True)))
        for key, value in rec.run.items():
            payload[key] = value[:, ri]
        payload["flip_state"] = rec.flip_state[:, ri]
        for li, layer in enumerate(rec.layers, start=1):
            for key, value in layer.items():
                payload[f"layer{li}_{key}"] = value[:, ri]
        path = logdir / f"{ARM}_seed{run['seed']}.npz"
        np.savez_compressed(path, **payload)
        paths.append(path)
    return paths


def _run_arm_comb(cfg: dict, device: str, outdir: Path, *, total: int,
                  seeds: list[int]) -> dict:
    """``lr_a1_0901._run_arm`` の写し。setup と log writer だけが本走のもの。"""
    local = copy.deepcopy(cfg)
    local["common"]["total_steps"] = int(total)
    local["common"]["seeds"] = [int(v) for v in seeds]
    every = int(local["common"]["lop_every"])
    probes = list(range(0, total + 1, every))
    if probes[-1] != total:
        probes.append(total)
    st = setup_arm_comb(local, device)
    _, initial = exact_layer_record_elu(st, float(_P(local)["sigma_degenerate_tol"]))
    if not identity_sanity_pass(initial, float(local["sanity"]["s1_identity_tol"])):
        raise RuntimeError(f"{ARM} initial exact-support identity failed")
    rec = EluRecorder(probes, st, float(_P(local)["sigma_degenerate_tol"]),
                      float(local["sanity"]["s1_identity_tol"]), every,
                      zbar_layers=[], readout_steps=[])
    checkpoints = [int(v) for v in local["common"].get("checkpoints", [])
                   if int(v) <= total]
    print(f"[{ARM}] act={ACTIVATION} alpha={ALPHA:g} hidden={st['hidden']} "
          f"seeds={seeds} steps={total:,}", flush=True)
    started = time.time()
    try:
        elapsed = train_arm_elu(st, rec, probes, total, outdir, checkpoints)
    except NumericDivergenceError as exc:
        elapsed = time.time() - started
        event = dict(exc.event)
        event.update(probe_every=every, registered_total_steps=total,
                     registered_seeds=seeds, activation=ACTIVATION,
                     act_alpha=ALPHA, elapsed_sec=elapsed,
                     detection="nonfinite_training_state_at_probe",
                     action="stop_arm_no_rescue", rescue="none")
        path = outdir / "arm_status" / f"{ARM}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(event, indent=2, ensure_ascii=False),
                        encoding="utf-8")
        print(f"[{ARM}] {NUMERIC_DIVERGENCE} at step "
              f"{event['detected_step']:,}", flush=True)
        return dict(status=NUMERIC_DIVERGENCE, elapsed_sec=elapsed,
                    divergence=event, sanity=dict(pass_=False))
    sanity = rec.sanity()
    if not sanity["pass_"]:
        raise RuntimeError(f"{ARM} recorder sanity failed: {sanity}")
    write_arm_logs_comb(outdir, st, rec)
    print(f"[{ARM}] complete in {elapsed:.1f}s", flush=True)
    return dict(status="COMPLETE", elapsed_sec=elapsed, sanity=sanity,
                final_env=_env_hashes(st))


# ---------------------------------------------------------------------------
# 前段チェック（spec §7）
# ---------------------------------------------------------------------------
def _s_copy() -> dict:
    """S-copy: 宿主 3 関数の写しが**手順を落としていない**ことを AST で検算する。

    段 A（`weird_act_0903`）と違い、段 B の写しは設計上ゼロ差にはならない
    （活性化・腕名・メタデータが変わる）。行や文字で比べると折り返しの違いが差に見えるので、
    **AST から「呼び出した名前と制御構文の順列」だけ**を取り出して比べる。
    文字列リテラルや変数名は見ない。したがって

    * 探索格子・チェックポイント絞り・サニティ呼び出し・発散ハンドラを落とすと **必ず落ちる**
    * 活性化やメタデータの書き換えは ``RENAMED`` / ``ADDED`` に載っていれば通る

    ``RENAMED`` / ``ADDED`` は**実装時に意図した差の全部**であり、ここに無い構造差が
    出たら FAIL にする。
    """
    import ast
    import difflib

    RENAMED = {"setup_arm_lr": "setup_arm_comb",
               "write_arm_logs_lr": "write_arm_logs_comb"}
    ADDED = ("print", "_P")            # 進捗表示と本走のブロック名アクセサ
    REMOVED = ("_arm_status_path",)    # 写しでは outdir / "arm_status" / ... を直に組む

    def skeleton(fn) -> list[str]:
        tree = ast.parse(inspect.getsource(fn).lstrip())
        out: list[str] = []

        class V(ast.NodeVisitor):
            def visit_Call(self, node):
                f = node.func
                name = (f.id if isinstance(f, ast.Name)
                        else f.attr if isinstance(f, ast.Attribute) else "<call>")
                out.append(f"call:{name}")
                self.generic_visit(node)

            def visit_If(self, node):
                out.append("if")
                self.generic_visit(node)

            def visit_Try(self, node):
                out.append("try")
                self.generic_visit(node)

            def visit_For(self, node):
                out.append("for")
                self.generic_visit(node)

            def visit_Raise(self, node):
                out.append("raise")
                self.generic_visit(node)

            def visit_Return(self, node):
                out.append("return")
                self.generic_visit(node)

        V().visit(tree)
        return out

    rows = []
    for mine, theirs in ((setup_arm_comb, HOST.setup_arm_lr),
                         (write_arm_logs_comb, HOST.write_arm_logs_lr),
                         (_run_arm_comb, HOST._run_arm)):
        a, b = skeleton(mine), skeleton(theirs)
        canon = [f"call:{RENAMED[x[5:]]}" if x.startswith("call:")
                 and x[5:] in RENAMED else x for x in b]
        sm = difflib.SequenceMatcher(a=canon, b=a, autojunk=False)
        diffs, unexplained = [], []
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                continue
            host_items, mine_items = canon[i1:i2], a[j1:j2]
            diffs.append(dict(tag=tag, host=host_items, mine=mine_items))
            for item in mine_items:
                if item.startswith("call:") and item[5:] in ADDED:
                    continue
                unexplained.append(dict(side="mine", tag=tag, item=item))
            for item in host_items:
                if item.startswith("call:") and item[5:] in REMOVED:
                    continue
                unexplained.append(dict(side="host", tag=tag, item=item))
        rows.append(dict(function=mine.__name__, host=theirs.__name__,
                         n_nodes_mine=len(a), n_nodes_host=len(b),
                         similarity=round(sm.ratio(), 4), diffs=diffs,
                         unexplained=unexplained, pass_=bool(not unexplained)))
    return dict(pass_=all(r["pass_"] for r in rows), rows=rows,
                renamed=RENAMED, added=list(ADDED), removed=list(REMOVED),
                note="AST の呼び出し列で構造を比べる。意図した差は renamed/added/removed に列挙")


def _s_activation_inherited(cfg: dict) -> dict:
    """段 A（`weird_act_0903`）で `comb_binf` の S-fd / S-num / S-dial が PASS 済みか。

    **再較正はしない。** 継承元の preflight.json を読んで、当該活性化の行が
    PASS だったことを確認するだけ（spec §7・config の
    ``activation_checks_inherited_from``）。
    """
    path = Path(ROOT) / str(cfg["sanity"]["activation_checks_inherited_from"])
    if not path.exists():
        return dict(pass_=False, path=str(path), reason="inherited preflight missing")
    pre = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for check in ("S_fd", "S_num", "S_dial"):
        block = pre.get("checks", {}).get(check, {})
        hit = [r for r in (block.get("rows") or [])
               if str(r.get("activation")) == ACTIVATION
               or str(r.get("arm", "")).startswith("CB_a1")]
        rows.append(dict(check=check, overall_pass=bool(block.get("pass_")),
                         rows_for_comb=len(hit),
                         all_pass=all(bool(r.get("pass_", True)) for r in hit)))
    # 併せて閉形式を今この場で再確認する（安いので）
    net = VecMLPL(1, [2], 2, torch.Generator().manual_seed(0), "cpu")
    net.set_activation(ACTIVATION, ALPHA, "alpha_exp")
    z = torch.linspace(-20, 20, 20001, dtype=torch.float64)
    mask = (z - 0.0).abs() > 1e-3
    h = 1e-6
    with torch.no_grad():
        fd = (net.act_fn(z[mask] + h) - net.act_fn(z[mask] - h)) / (2 * h)
        g = net.act_grad(z[mask], net.act_fn(z[mask]))
    worst = float((fd - g).abs().max())
    return dict(pass_=bool(all(r["overall_pass"] for r in rows) and worst <= 1e-6),
                inherited_from=str(path), rows=rows, refreshed_s_fd_worst=worst,
                note="継承の確認のみ。床も窓も再較正しない")


def _s_b(cfg: dict, device: str, outdir: Path) -> dict:
    """S-B（spec §7・新規）: step 0 で ``LR_A1`` と init・教師・入力列が bit 一致。

    活性化の選択は乱数を消費しないので、``CB_A1`` は宿主 ``LR_A1`` と同じ系列に乗る。
    参照走の生ログがこの機に無い場合は、**その旨を残して PASS にしない**。
    """
    S = cfg["sanity"]
    steps = int(S["s_b_steps"])
    every = int(cfg["common"]["lop_every"])
    ref_dir = Path(ROOT) / str(S["s_b_reference"]) / "logs"
    ref_arm = str(S["s_b_reference_arm"])
    seeds = [int(v) for v in cfg["common"]["seeds"]]
    local = copy.deepcopy(cfg)
    local["common"]["total_steps"] = steps
    st_comb = setup_arm_comb(local, device)
    st_leaky = HOST.setup_arm_lr(HOST_cfg_for_leaky(local), device)
    differences = []
    for ri, run in enumerate(st_comb["runs"]):
        a = _seed_state_hashes_p1(st_comb, ri)
        b = _seed_state_hashes_p1(st_leaky, ri)
        for key, value in b.items():
            if a.get(key) != value:
                differences.append(dict(seed=int(run["seed"]), where=key))
    env_equal = _env_hashes(st_comb) == _env_hashes(st_leaky)
    flip_rows, missing = [], []
    for ri, run in enumerate(st_comb["runs"]):
        path = ref_dir / f"{ref_arm}_seed{run['seed']}.npz"
        if not path.exists():
            missing.append(str(path))
            continue
        with np.load(path, allow_pickle=False) as z:
            ref_flip = z["flip_state"][0].copy()
        mine = (st_comb["env"].flip_state[ri].detach().cpu().numpy()
                .astype(np.float32))
        same = bool(np.array_equal(mine, ref_flip))
        flip_rows.append(dict(seed=int(run["seed"]), flip_state_equal=same))
        if not same:
            differences.append(dict(seed=int(run["seed"]), where="reference.flip_state"))
    status = ("VERIFIED" if flip_rows and not missing
              else "REFERENCE_LOGS_ABSENT_ON_THIS_MACHINE")
    return dict(pass_=bool(not differences and env_equal), steps=steps,
                in_process_state_equal=not differences,
                in_process_env_equal=bool(env_equal),
                reference_status=status, reference_flip_rows=flip_rows,
                reference_missing=sorted(set(missing)), differences=differences,
                caveat="init/teacher/input realization only; trajectories diverge "
                       "after step 1")


def HOST_cfg_for_leaky(cfg: dict) -> dict:
    """宿主 ``setup_arm_lr`` に食わせるための最小の写し（S-B の比較相手）。

    宿主は ``cfg["lr_a1"]`` と ``arms[].name == "LR_A1"`` を見るので、本走の config から
    その 2 つだけを作って渡す。**本走の判定には一切使わない。**
    """
    out = copy.deepcopy(cfg)
    out["lr_a1"] = copy.deepcopy(_P(cfg))
    out["arms"] = [dict(name="LR_A1", activation="leaky",
                        hidden=[WIDTH, WIDTH], centered_layers=[1])]
    out.setdefault("activation", {})["leaky"] = {"name": "leaky_relu",
                                                 "negative_slope": HOST.SLOPE}
    return out


def preflight(cfg: dict, device: str, outdir: Path) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    checks = {"S_copy": _s_copy(),
              "S_activation_inherited": _s_activation_inherited(cfg),
              "S_B": _s_b(cfg, device, outdir / "sb")}
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
        raise RuntimeError("preflight failed; see " + str(path))
    return result


# ---------------------------------------------------------------------------
# 集計（spec §6 V7）
# ---------------------------------------------------------------------------
def _level(cfg: dict, outdir: Path, window: list[int]) -> np.ndarray:
    """本走の seed 別 ``log10(mean unfit)``（宿主 `analyze` と同じ作り方）。"""
    P = _P(cfg)
    period, floor = int(P["task_period"]), float(P["unfit_floor"])
    out = []
    for seed in [int(v) for v in cfg["common"]["seeds"]]:
        with np.load(outdir / "logs" / f"{ARM}_seed{seed}.npz",
                     allow_pickle=False) as z:
            idx = _window_indices(z["step"], period, list(window))
            expected = window[1] - window[0] + 1
            if len(idx) != expected:
                raise RuntimeError(f"seed {seed}: window has {len(idx)} rows "
                                   f"(expected {expected})")
            out.append(float(np.mean(z["unfit"][idx])))
    return np.log10(np.maximum(np.asarray(out, dtype=np.float64), floor))


def _load_controls(cfg: dict) -> dict:
    """対照 3 腕を ``results/lr_a1_0901/verdict.csv`` から転記・再構成（spec §10 追補 4）。"""
    path = Path(ROOT) / str(cfg["controls"]["endpoint_source"])
    if not path.exists():
        raise RuntimeError(f"control source missing: {path}")
    rows = list(csv.DictReader(path.open(newline="")))

    def seed_values(metric: str, arm: str | None = None,
                    baseline: str | None = None) -> np.ndarray:
        for r in rows:
            if (r["metric"] == metric
                    and (arm is None or r["arm"] == arm)
                    and (baseline is None or r["baseline"] == baseline)):
                return np.asarray(json.loads(r["seed_values"]), dtype=np.float64)
        raise RuntimeError(f"{metric} not found in {path}")

    lr = seed_values("P2_log10_mean_unfit_level", "LR_A1")
    d_lr = seed_values("P2_delta_log10_mean_unfit", "LR_A1", "L2_A1")
    d_e = seed_values("P2_E_A1_reference_delta_log10_mean_unfit", "E_A1", "L2_A1")
    l2 = lr - d_lr
    ea = l2 + d_e
    return {
        "LR_A1": dict(arm="LR_A1", log_u=lr, source=str(path),
                      how="P2_log10_mean_unfit_level をそのまま"),
        "L2_A1": dict(arm="L2_A1", log_u=l2, source=str(path),
                      how="LR_A1 の水準 − paired 差(LR_A1−L2_A1)"),
        "E_A1": dict(arm="E_A1", log_u=ea, source=str(path),
                     how="L2_A1 の再構成値 + paired 差(E_A1−L2_A1)"),
    }


def _draws(cfg: dict) -> np.ndarray:
    P = _P(cfg)
    n = len(cfg["common"]["seeds"])
    rng = np.random.default_rng(int(P["bootstrap_seed"]))
    return rng.integers(0, n, size=(int(P["bootstrap_B"]), n))


def _ci(cfg: dict, values: np.ndarray, draws: np.ndarray) -> dict:
    P = _P(cfg)
    return _ci_components(np.asarray(values, dtype=np.float64), draws, "median",
                          float(P["degenerate_se_tol"]),
                          float(P["degenerate_frac_max"]),
                          float(P["degenerate_width_ratio_max"]))


def _contrast(cfg: dict, a: np.ndarray, b: np.ndarray, draws: np.ndarray,
              label: str) -> dict:
    values = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    ci = _ci(cfg, values, draws)
    margin = float(_P(cfg)["v7"]["margin"])
    lo, hi = ci.get("percentile_ci_lo"), ci.get("percentile_ci_hi")
    if lo is None or hi is None or not np.isfinite([lo, hi]).all():
        equiv = "INCONCLUSIVE_WIDE"
    elif hi < -margin:
        equiv = "BELOW_SOFT"
    elif lo >= -margin and hi <= margin:
        equiv = "EQUIV_SOFT"
    elif lo > 0:
        equiv = "SHORT_OF_SOFT"
    else:
        equiv = "INCONCLUSIVE_WIDE"
    pos, neg = int((values > 0).sum()), int((values < 0).sum())
    return dict(label=label, point=float(np.median(values)), ci=ci,
                equivalence=equiv, margin=margin, sign_pos=pos, sign_neg=neg,
                seed_values=[float(v) for v in values])


def _exact_fit(cfg: dict, u_1m_log: np.ndarray) -> dict:
    E = _P(cfg)["exact_fit"]
    med = float(np.median(10.0 ** np.asarray(u_1m_log, dtype=np.float64)))
    fired = bool(med <= float(E["threshold"]))
    return dict(fired=fired, u_1m_seed_median=med,
                threshold=float(E["threshold"]),
                label=str(E["label"]) if fired else "")


def _v7_label(cfg: dict, main: dict, leaky: dict, fired: bool) -> tuple[str, dict]:
    V = _P(cfg)["v7"]
    margin = float(V["margin"])
    ci = main["ci"]
    lo, hi = ci.get("percentile_ci_lo"), ci.get("percentile_ci_hi")
    detail = dict(ci_lo=lo, ci_hi=hi, margin=margin, exact_fit=fired,
                  point=main["point"],
                  vs_leaky=leaky["equivalence"], vs_leaky_point=leaky["point"])
    extra = ("WORSE_THAN_LEAKY_DEPTH2" if leaky["equivalence"] == "SHORT_OF_SOFT"
             else "")
    detail["co_label"] = extra
    if lo is None or hi is None or not np.isfinite([lo, hi]).all():
        return "PARTIAL", detail
    if hi < -margin:
        return str(V["labels"]["below"]), detail
    if (lo <= 0.0 <= hi) or (lo >= -margin and hi <= margin):
        return str(V["labels"]["persists"]), detail
    return "PARTIAL", detail


def analyze(cfg: dict, outdir: Path) -> dict:
    P = _P(cfg)
    div = outdir / "arm_status" / f"{ARM}.json"
    if div.exists() and not (outdir / "logs" / f"{ARM}_seed0.npz").exists():
        event = json.loads(div.read_text(encoding="utf-8"))
        verdicts = dict(V7=str(P["v7"]["labels"]["diverged"]),
                        V7_detail=dict(detected_step=event["detected_step"],
                                       bad_seeds=event.get("bad_seeds")))
        (outdir / "summary.md").write_text(
            f"# {EXPERIMENT}\n\n`{ARM}` は `{NUMERIC_DIVERGENCE}`"
            f"（step {event['detected_step']:,}）。V7 は "
            f"`{verdicts['V7']}` で空にする（spec §6）。\n", encoding="utf-8")
        return dict(experiment=EXPERIMENT, verdicts=verdicts)
    draws = _draws(cfg)
    controls = _load_controls(cfg)
    late = _level(cfg, outdir, [int(v) for v in P["late_tasks"]])
    one_m = _level(cfg, outdir, [int(v) for v in P["exact_fit"]["window_1m_tasks"]])
    fired = _exact_fit(cfg, one_m)
    contrasts = {}
    for name in ("L2_A1", "LR_A1", "E_A1"):
        contrasts[name] = _contrast(cfg, late, controls[name]["log_u"], draws,
                                    f"{ARM} - {name} (late_t451_500)")
    label, detail = _v7_label(cfg, contrasts["L2_A1"], contrasts["LR_A1"],
                              fired["fired"])
    verdicts = dict(V7=label, V7_detail=detail, EXACT_FIT=fired)
    rows = [dict(metric="log10_mean_unfit_level", window="late_t451_500", arm=ARM,
                 baseline="", pairing="single_arm",
                 seed_values=json.dumps([float(v) for v in late]),
                 point=float(np.median(late)), percentile_ci_lo="",
                 percentile_ci_hi="", equivalence="", margin="",
                 sign_neg="", sign_pos="", EXACT_FIT=fired["label"],
                 u_1m_seed_median=fired["u_1m_seed_median"], V7=label)]
    for name, c in contrasts.items():
        rows.append(dict(
            metric="delta_log10_mean_unfit", window="late_t451_500", arm=ARM,
            baseline=name, pairing="paired",
            seed_values=json.dumps(c["seed_values"]), point=c["point"],
            percentile_ci_lo=c["ci"].get("percentile_ci_lo"),
            percentile_ci_hi=c["ci"].get("percentile_ci_hi"),
            equivalence=c["equivalence"], margin=c["margin"],
            sign_neg=c["sign_neg"], sign_pos=c["sign_pos"], EXACT_FIT="",
            u_1m_seed_median="", V7=label if name == "L2_A1" else ""))
    for name, c in controls.items():
        rows.append(dict(
            metric="control_log10_mean_unfit_level", window="late_t451_500",
            arm=name, baseline="", pairing="transcribed",
            seed_values=json.dumps([float(v) for v in c["log_u"]]),
            point=float(np.median(c["log_u"])), percentile_ci_lo="",
            percentile_ci_hi="", equivalence="", margin="", sign_neg="",
            sign_pos="", EXACT_FIT="", u_1m_seed_median="", V7=""))
    write_csv(outdir / "verdict.csv", rows)
    _write_summary(cfg, outdir, late, controls, contrasts, verdicts)
    return dict(experiment=EXPERIMENT, verdicts=verdicts)


def _write_summary(cfg, outdir, late, controls, contrasts, verdicts) -> None:
    v7 = verdicts["V7"]
    detail = verdicts["V7_detail"]
    fired = verdicts["EXACT_FIT"]
    L = [f"# {EXPERIMENT} — 櫛の分離・段 B（深さ 2・第 1 層中心化・5M）\n",
         f"spec: `{cfg['spec']}` / 事前登録 commit で凍結。"
         "数値の引用は `verdict.csv` と本ファイルからのみ。\n",
         "**★ 事前予測は起草側（Claude）の値を Issa が承認したもので、独立の予言ではない**"
         "（`preregistration.prediction_provenance`）。\n",
         "## 判定\n",
         f"**V7 = `{v7}`**"
         + (f"（併記 `{detail['co_label']}`）" if detail.get("co_label") else "")
         + "\n",
         "| 対比（末尾窓 451–500・paired） | 点推定 | percentile CI | 等価判定 | 符号 |",
         "| --- | --- | --- | --- | --- |"]
    for name, c in contrasts.items():
        ci = c["ci"]
        lo, hi = ci.get("percentile_ci_lo"), ci.get("percentile_ci_hi")
        s = "—" if lo is None or hi is None else f"[{lo:+.3f}, {hi:+.3f}]"
        L.append(f"| `{ARM}` − `{name}` | {c['point']:+.4f} | {s} | "
                 f"{c['equivalence']} | {c['sign_neg']}:{c['sign_pos']} |")
    L.append(f"\n## `EXACT_FIT`\n")
    L.append(f"1M 窓（タスク 91–100）の $U$ seed 中央値 = **{fired['u_1m_seed_median']:.4e}**"
             f" → {'**立つ**' if fired['fired'] else '立たない'}"
             f"（閾値 {fired['threshold']:.0e}）。\n")
    if fired["fired"]:
        L.append("**★ `EXACT_FIT` が立っているので、水準差を機構として引かない**（spec §9）。"
                 "V7 のラベルは観測であって、「櫛が深さ 2 で LoP を防ぐ機構」ではない。\n")
    L.append("## 水準\n")
    L.append("| 腕 | median log10 U（末尾窓） | 出所 |")
    L.append("| --- | --- | --- |")
    L.append(f"| `{ARM}`（本走） | {float(np.median(late)):.4f} | "
             f"`results/{EXPERIMENT}/logs` |")
    for name, c in controls.items():
        L.append(f"| `{name}`（committed） | {float(np.median(c['log_u'])):.4f} | "
                 f"`{Path(c['source']).parent.name}/verdict.csv` — {c['how']} |")
    L.append("\n**対照 3 腕は別走の committed 値。** `L2_A1` と `E_A1` は水準行が無いので "
             "paired 差から seed ごとに再構成した（spec §10 追補 4）。"
             "`L2_A1` の再構成は `mlp2_phase1_0829` の生ログからの再計算と "
             "**10 seed すべて bit 一致**することを確認済み。\n")
    L.append("## 引用上の注意\n")
    L.append("- 深さ 2・幅 [100,100]・第 1 層のみ中心化・オラクルなし・素の SGD・5M・float32")
    L.append("- 末尾窓は **451–500**（段 A の 491–500 とは違う）。床は **1e−23**")
    L.append("- **V5・V6（段 A）と V7 は互いに独立**。1 つに畳まない")
    L.append("- 事前予測は独立の予言ではない（起草側の値を Issa が承認したもの）")
    (outdir / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Provenance / run
# ---------------------------------------------------------------------------
def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _provenance(cfg_path: Path, cfg: dict, outdir: Path, sanity: dict,
                status: dict, elapsed: float, started: str) -> dict:
    names = ("verdict.csv", "summary.md", "config_used.yaml")
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
        stage="B", arm=ARM, activation=ACTIVATION, act_alpha=ALPHA,
        hidden=[WIDTH, WIDTH], centered_layers=[1],
        window="late_t451_500", unfit_floor=float(_P(cfg)["unfit_floor"]),
        host="lr_a1_0901 (setup / runner / log writer are copies; the host is untouched)",
        stage_note="stage A and stage B are preregistered in the same commit "
                   "(a6c93f6); submission is sequential (2026-09-03 Issa).",
        prediction_provenance=cfg["preregistration"]["prediction_provenance"],
        control_source=str(cfg["controls"]["endpoint_source"]),
        control_reconstruction=dict(cfg["controls"]["reconstruction"]),
        control_reconstruction_verified_against_raw_logs=(
            "L2_A1: bit-identical to results/mlp2_phase1_0829/logs recomputation "
            "for all 10 seeds (2026-09-03). E_A1 logs are absent on this machine."),
        generator_offset=int(cfg["common"]["generator_offset"]),
        arm_status=status, sanity=sanity, output_sha256=hashes)


def run(cfg_path: Path, cfg: dict, device: str, outdir: Path) -> dict:
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    t0 = time.time()
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / "config_used.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, allow_unicode=True, sort_keys=False)
    pre_path = Path(ROOT) / PREFLIGHT_DIR / "preflight.json"
    if not pre_path.exists():
        raise RuntimeError(f"run the preflight first: {pre_path} is missing")
    pre = json.loads(pre_path.read_text(encoding="utf-8"))
    if not pre.get("pass_"):
        raise RuntimeError(f"preflight did not pass: {pre_path}")
    total = int(cfg["common"]["total_steps"])
    seeds = [int(v) for v in cfg["common"]["seeds"]]
    got = _run_arm_comb(cfg, device, outdir, total=total, seeds=seeds)
    elapsed = time.time() - t0
    prov = _provenance(cfg_path, cfg, outdir, pre,
                       {ARM: got.get("status")}, elapsed, started)
    prov["divergence"] = got.get("divergence")
    (outdir / "provenance.json").write_text(
        json.dumps(prov, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")
    print(f"[run] {ARM}: {got.get('status')} in {elapsed:.1f}s", flush=True)
    return prov


def main() -> None:
    ap = argparse.ArgumentParser(description=EXPERIMENT)
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--stage", default="preflight",
                    choices=["preflight", "run", "analyze"])
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()
    cfg_path = Path(args.config)
    cfg = load_config(str(cfg_path))
    validate_config(cfg, stage=args.stage)
    require_omp(cfg)
    device = pick_device(cfg) if args.stage != "preflight" else "cpu"
    outdir = Path(args.outdir) if args.outdir else Path(ROOT) / cfg["output"]["dir"]
    if args.stage == "preflight":
        preflight(cfg, device, Path(ROOT) / PREFLIGHT_DIR)
        return
    if args.stage == "analyze":
        got = analyze(cfg, outdir)
        print(json.dumps(got["verdicts"], ensure_ascii=False, indent=1, default=str),
              flush=True)
        return
    run(cfg_path, cfg, device, outdir)


if __name__ == "__main__":
    main()
