"""valley_off_0903: 谷の逃走・走 A（オラクルなしの自然な condA で GELU・SiLU は逃げるか）。

事前登録: ``specs/spec_valley_off_0903.md``（この実装より**先に** config と一緒に単独
commit する）。Obsidian 側の正本は ``可塑性喪失/spec/谷の逃走_走A_spec_0903.md``（v1・
Kubo 起案）。親主張は ``到達と離脱_統合主張_0903`` §4。

宿主は ``gate_dose_0830`` の ``_off`` 腕（オラクルなし・自然な condA・1 層・5M）で、
学習経路・記録経路はそのまま ``src.gate_dose`` から、閉形式 SiLU/GELU と
ユニット別ロガーは ``src.gate_dial_0902`` から import する。**新しい算術は 1 つも
足さない**（本走が新しいのは腕の組み合わせだけ）。

対照 ``R_off`` / ``E_off`` / ``LR_off`` は再走しない。主 endpoint は
``results/gate_dose_0830/verdict.csv`` から転記し、ユニット別量だけ同走の
``logs/*.npz`` を読む（対照には ``mob`` / ``zmax`` 列が無いので ReLU では
``p_hat`` を代用する。ELU/leaky 対照では代用が成立しないので凍結率は空欄）。

Stages::

    OMP_NUM_THREADS=1 .venv/bin/python -m src.valley_off_0903 --preflight
    OMP_NUM_THREADS=1 .venv/bin/python -m src.valley_off_0903 --smoke
    OMP_NUM_THREADS=1 .venv/bin/python -m src.valley_off_0903 --arm G_off
    OMP_NUM_THREADS=1 .venv/bin/python -m src.valley_off_0903
    OMP_NUM_THREADS=1 .venv/bin/python -m src.valley_off_0903 --analyze-only
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
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
from .dose_const_5m import clopper_pearson
from .gate_dose import _load_arm, _window, train_arm_gate
from .gate_dial_0902 import (DialRecorder, SanityError, _load_new_arm,
                             _revival_counts, _run_arm, _tail_index,
                             freeze_depth, setup_arm_dial, valley_depth)
from .gate_dial_0902 import _s_fd as _gd_s_fd
from .gate_dial_0902 import _s_limit_smooth as _gd_s_limit
from .gate_dial_0902 import _s_num as _gd_s_num
from .mlp2_phase0 import _sha_array, _sha_file, require_omp, write_csv
from .mlp2_phase0b import _ci_components, _complete_arm_logs, _window_indices
from .mlp2_phase1 import (NUMERIC_DIVERGENCE, StreamDigest, _env_hashes,
                          _init_hashes, _seed_state_hashes_p1)


EXPERIMENT = "valley_off_0903"
CONFIG = Path(ROOT) / "configs" / "valley_off_0903.yaml"

ARM_ORDER = ("G_off", "S_off")
CONTROL_ORDER = ("R_off", "E_off", "LR_off")
BASELINE = "R_off"
PRIMARY = "G_off"
SMOKE_STEPS = 30_000

# 事前登録の腕定義（family, activation label, dial）。validate_config が逐語照合する。
REGISTERED_ARMS = {"G_off": ("gelu", "gelu", 1.0), "S_off": ("silu", "silu", 1.0)}
LABEL_ORDER = ("FLIGHT_NEEDS_OFFSET", "WORSE_UNTESTABLE_AT_CEILING",
               "FLIGHT_WITHOUT_ORACLE", "FLIGHT_SLOW", "PARTIAL")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def _arm(cfg: dict, name: str) -> dict:
    return next(a for a in cfg["arms"] if a["name"] == name)


def _P(cfg: dict) -> dict:
    return cfg["valley_off"]


def validate_config(cfg: dict, *, stage: str) -> None:
    """凍結した設計からのずれをすべて ValueError にする。"""
    if stage not in {"preflight", "smoke", "run", "analyze"}:
        raise ValueError(f"unknown stage {stage!r}")
    C, P, G, S = cfg["common"], cfg["phase1"], _P(cfg), cfg["sanity"]
    if [a["name"] for a in cfg["arms"]] != list(ARM_ORDER):
        raise ValueError(f"arms must be ordered as {ARM_ORDER}")
    for arm_cfg in cfg["arms"]:
        family, label, dial = REGISTERED_ARMS[arm_cfg["name"]]
        if (str(arm_cfg["family"]), str(arm_cfg["activation"]),
                float(arm_cfg["dial"])) != (family, label, dial):
            raise ValueError(f"{arm_cfg['name']} deviates from the registered arm")
        # 走 A の核心。オラクルを掛けたら別の走になる。
        if arm_cfg["target_dose"] is not None or arm_cfg["target_mu_norm"] is not None:
            raise ValueError(f"{arm_cfg['name']} must be an `_off` arm (no oracle)")
        if list(arm_cfg["centered_layers"]) != []:
            raise ValueError(f"{arm_cfg['name']} must not centre any layer")
        if list(arm_cfg["hidden"]) != [100]:
            raise ValueError(f"{arm_cfg['name']} must be 1 layer of width 100")
    if int(C["total_steps"]) != 5_000_000:
        raise ValueError("total_steps is registered as 5M")
    if [int(v) for v in C["seeds"]] != list(range(10)):
        raise ValueError("seeds are registered as 0-9")
    if int(C["generator_offset"]) != 0:
        raise ValueError("generator_offset must stay 0 (S-pair with the parent run)")
    if float(C["lr_main"]) != 0.01 or str(C["device"]) != "cpu":
        raise ValueError("lr / device deviate from the registered design")
    if cfg["intervention"]["oracle"] is not False:
        raise ValueError("走 A is the no-oracle arm; intervention.oracle must be false")
    if float(P["unfit_floor"]) != 1e-16 or P["recalibrate_floor"] is not False:
        raise ValueError("the floor is inherited from dose_const_5m_0830")
    if float(P["onset_threshold"]) != 0.05:
        raise ValueError("onset threshold is registered as 0.05")
    if int(P["bootstrap_seed"]) != 20260912:
        raise ValueError("bootstrap seed is registered as 20260912")
    if P["window_points_are_task_ends_only"] is not True:
        raise ValueError("windows are task-end records only (parent U_k)")
    if str(G["design"]["freeze_source_column"]) != "layer1_mob":
        raise ValueError("A3's freeze rate is defined on this run's logger")
    if float(G["design"]["freeze_phi_prime_threshold"]) != 1e-6:
        raise ValueError("freeze threshold is registered as 1e-6")
    if list(G["labels"]["order"]) != list(LABEL_ORDER):
        raise ValueError(f"label order must be {LABEL_ORDER}")
    if str(G["predictions"]["primary_arm"]) != PRIMARY:
        raise ValueError("the label is decided on G_off alone")
    if str(cfg["controls"]["baseline"]) != BASELINE:
        raise ValueError("the paired baseline is R_off")
    if list(cfg["staging"]["arms"]) != list(ARM_ORDER):
        raise ValueError("staging.arms deviates from the registered arms")
    if int(S["omp_num_threads"]) != 1:
        raise ValueError("OMP_NUM_THREADS is registered as 1")


# ---------------------------------------------------------------------------
# 幾何（谷底と凍結深さ）
# ---------------------------------------------------------------------------
def _geometry(cfg: dict, arm: str) -> dict:
    """腕の (u*, u_fr) を登録値と数値解の両方で返す。"""
    threshold = float(_P(cfg)["design"]["freeze_phi_prime_threshold"])
    if arm in CONTROL_ORDER:
        entry = dict(cfg["controls"]["arms"][arm])
        act = {"relu": "relu", "leaky_relu": "leaky_relu", "elu": "elu"}[
            str(entry["activation"])]
        dial = float(entry["dial"])
    else:
        entry = _arm(cfg, arm)
        act, dial = str(entry["activation"]), float(entry["dial"])
    if act == "relu":
        numeric_star, numeric_fr = 0.0, 0.0
    else:
        numeric_star = valley_depth(act, dial)
        numeric_fr = freeze_depth(act, dial, threshold)
    reg_star, reg_fr = entry.get("u_star"), entry.get("u_fr")
    return dict(arm=arm, activation=act, dial=dial,
                u_star_registered=None if reg_star is None else float(reg_star),
                u_fr_registered=None if reg_fr is None else float(reg_fr),
                u_star_numeric=numeric_star, u_fr_numeric=numeric_fr,
                has_valley=bool(np.isfinite(numeric_star) and numeric_star > 0),
                threshold=threshold)


# ---------------------------------------------------------------------------
# 前段チェック（spec §7）
# ---------------------------------------------------------------------------
def _activation_cfg(cfg: dict) -> dict:
    """S-fd / S-num / S-limit を回すための ``gate_dial_0902`` config。

    本走の活性化（ダイヤル 1.0 の SiLU/GELU）は ``gate_dial_0902`` の ``S_b1_1216`` /
    ``G_b1_1216`` と**同一の閉形式・同一のダイヤル**なので、活性化だけの検査は
    同モジュールの実装をその config で回して継承する。学習経路は見ない検査である。
    """
    path = Path(ROOT) / str(cfg["sanity"]["activation_checks_via"])
    other = load_config(str(path))
    dials = {a["name"]: float(a["dial"]) for a in other["arms"]}
    if dials.get("S_b1_1216") != 1.0 or dials.get("G_b1_1216") != 1.0:
        raise SanityError("the borrowed activation config no longer has dial 1.0")
    for key in ("s_fd_tol", "s_fd_points", "s_num_range", "s_num_points",
                "s_limit_beta", "s_limit_forward_tol_over_beta",
                "s_limit_grad_tol", "s_limit_grid_abs_z_min_over_beta"):
        if other["sanity"][key] != cfg["sanity"][key]:
            raise SanityError(f"sanity.{key} differs from the borrowed config")
    return other


def _s_dial(cfg: dict) -> dict:
    """S-dial: config に凍結した u* / u_fr が数値解と一致すること。"""
    tol = float(_P(cfg)["design"]["u_fr_spec_rel_tol"])
    rows, failures = [], []
    for arm in list(ARM_ORDER) + list(CONTROL_ORDER):
        geo = _geometry(cfg, arm)
        row = dict(geo)
        for key in ("u_star", "u_fr"):
            registered, numeric = geo[f"{key}_registered"], geo[f"{key}_numeric"]
            if registered is None:
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


def _s_mob_off(cfg: dict, outdir: Path) -> dict:
    """S-mob: **本走の ``_off`` 経路で**新規ロガーが既知の量と一致すること（30k）。

    ``_off`` 腕は中心化しないので、``gate_dial_0902`` の S-mob（用量固定・中心化あり）
    とは ``z`` の座標が違う。ReLU 腕で ``mob == p_hat``、``zmean == (M+B)*denom``、
    ``zmax >= zmean`` を取り直す。
    """
    S = cfg["sanity"]
    steps, tol = int(S["s_mob_steps"]), float(S["s_mob_tol"])
    every = int(cfg["common"]["lop_every"])
    c = copy.deepcopy(cfg)
    c["common"]["seeds"] = [0, 1]
    probes = list(range(0, steps + 1, every))
    base = copy.deepcopy(_arm(c, "G_off"))
    rows, failures = [], []
    for label, activation, dial in (("relu", "relu", 0.0), ("gelu", "gelu", 1.0)):
        arm_cfg = copy.deepcopy(base)
        arm_cfg["activation"], arm_cfg["dial"] = activation, dial
        st = setup_arm_dial(c, arm_cfg, "cpu")
        rec = DialRecorder(probes, st)
        train_arm_gate(st, rec, probes, steps, outdir, [])
        p_hat = rec.layers[0]["p_hat"].astype(np.float64)
        mob = rec.unit["mob"].astype(np.float64)
        mob_err = (float(np.abs(mob - p_hat).max()) if activation == "relu"
                   else float("nan"))
        zmean = rec.unit["zmean"].astype(np.float64)
        formula = ((rec.layers[0]["M"].astype(np.float64)
                    + rec.layers[0]["B"].astype(np.float64))
                   * rec.layers[0]["denom"].astype(np.float64))
        good = np.isfinite(formula)
        scale = np.maximum(np.abs(formula[good]), 1.0)
        zmean_err = float((np.abs(zmean[good] - formula[good]) / scale).max())
        zmax_ge_zmean = bool((rec.unit["zmax"] >= rec.unit["zmean"] - 1e-6).all())
        row = dict(arm=label, activation=activation, dial=dial, steps=steps,
                   mob_max_abs_err_vs_p_hat=mob_err, zmean_max_rel_err=zmean_err,
                   zmax_ge_zmean=zmax_ge_zmean,
                   n_na_in_formula=int((~good).sum()),
                   centered=bool(st["centered_layers"][0]))
        rows.append(row)
        if ((activation == "relu" and mob_err > tol) or zmean_err > tol
                or not zmax_ge_zmean or row["centered"]):
            failures.append(row)
    return dict(pass_=not failures, tolerance=tol, rows=rows, failures=failures,
                note="mob equals p_hat on ReLU only; on GELU it is an independent "
                     "quantity (that is the point of the logger)")


def _s_log_b(cfg: dict, outdir: Path) -> dict:
    """S-log-b: ユニット別ロガーが軌道中立であること（既存全列が bit 一致）。"""
    steps = int(cfg["sanity"]["s_log_b_steps"])
    every = int(cfg["common"]["lop_every"])
    c = copy.deepcopy(cfg)
    c["common"]["seeds"] = [0, 1]
    probes = list(range(0, steps + 1, every))
    results = {}
    for label, record_units in (("with_logger", True), ("without_logger", False)):
        st = setup_arm_dial(c, _arm(c, PRIMARY), "cpu")
        rec = DialRecorder(probes, st, record_units=record_units)
        train_arm_gate(st, rec, probes, steps, outdir, [])
        payload = {f"run.{k}": v for k, v in rec.run.items()}
        payload.update({f"layer1.{k}": v for k, v in rec.layers[0].items()})
        payload.update({f"extra.{k}": v for k, v in rec.extra.items()})
        results[label] = ({k: _sha_array(v) for k, v in payload.items()},
                          _env_hashes(st))
    a, b = results["with_logger"], results["without_logger"]
    differing = sorted(k for k in a[0] if a[0][k] != b[0].get(k))
    env_differing = sorted(k for k in a[1] if a[1][k] != b[1].get(k))
    return dict(pass_=bool(not differing and not env_differing), steps=steps,
                columns=len(a[0]), differing_columns=differing,
                differing_env=env_differing)


def _s_pair(cfg: dict, outdir: Path) -> dict:
    """S-pair: 新規 2 腕と参照 ReLU が互いに、かつ親走 ``R_off`` と bit 一致すること。

    対応は **seed ごとのハッシュ**で取る（位置合わせではない）。オラクルが無いので
    S-dose は非該当（用量は測るだけの量になる）。
    """
    S = cfg["sanity"]
    steps = int(S["s_pair_steps"])
    every = int(cfg["common"]["lop_every"])
    seeds = [int(v) for v in cfg["common"]["seeds"]]
    arms = list(ARM_ORDER) + ["relu_probe"]
    init, final, streams, per_seed, flip0 = {}, {}, {}, {}, {}
    dose_rows = []
    for arm in arms:
        c = copy.deepcopy(cfg)
        if arm == "relu_probe":
            arm_cfg = copy.deepcopy(_arm(c, PRIMARY))
            arm_cfg["activation"], arm_cfg["dial"] = "relu", 0.0
        else:
            arm_cfg = _arm(c, arm)
        st = setup_arm_dial(c, arm_cfg, "cpu")
        init[arm] = _init_hashes(st)
        per_seed[arm] = {int(run["seed"]): _seed_state_hashes_p1(st, ri)
                         for ri, run in enumerate(st["runs"])}
        flip0[arm] = {int(run["seed"]):
                      st["env"].flip_state[ri].detach().cpu().numpy().astype(np.float32)
                      for ri, run in enumerate(st["runs"])}
        stream = StreamDigest()
        print(f"[S-pair] {arm} {steps:,} steps", flush=True)
        train_arm_gate(st, lambda *_: None, range(0, steps + 1, every), steps,
                       outdir, [], stream_hook=stream)
        final[arm], streams[arm] = _env_hashes(st), stream.digest()
        if st.get("target_mu_norm") is not None:      # 走 A では起きない
            dose_rows.append(dict(arm=arm, target_mu_norm=st["target_mu_norm"]))

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
    return dict(pass_=bool(not differences and not parent_missing),
                reference=reference, arms=arms, steps=steps,
                match_by="seed_init_hash", differences=differences,
                parent_flip_rows=parent_rows, parent_missing=parent_missing,
                oracle_arms=dose_rows,
                caveat="init/teacher/input realization only; trajectories diverge "
                       "after step 1")


def _endpoint_columns_unchanged(cfg: dict, ref_rel: str, want_sha: str) -> dict:
    """転記する列が provenance 記録時の版から 1 バイトも動いていないことの確認。

    親走の ``verdict.csv`` は provenance 記録後に別 commit で再生成されうる
    （``--analyze-only`` は provenance を書き直さない）。ファイルのハッシュが合わない
    こと自体は事故とは限らないが、**本走が転記する列**が動いていたら事故である。
    """
    import hashlib
    import io

    columns = list(cfg["controls"]["endpoint_columns"])
    try:
        revs = subprocess.check_output(["git", "log", "--format=%H", "--", ref_rel],
                                       cwd=ROOT, text=True).split()
    except (OSError, subprocess.CalledProcessError) as exc:
        return dict(checked=False, reason=f"git log failed: {exc}")
    blob, found_at = None, None
    for rev in revs:
        try:
            raw = subprocess.check_output(["git", "show", f"{rev}:{ref_rel}"], cwd=ROOT)
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
    return dict(checked=True, provenance_era_commit=found_at,
                columns_transcribed=columns, arms=list(CONTROL_ORDER),
                differing=differing, missing=missing,
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
    read_mismatches = [n for n in mismatches if n in read_files]
    column_check = None
    if "verdict.csv" in read_mismatches and recorded.get("verdict.csv"):
        column_check = _endpoint_columns_unchanged(cfg, f"{ref_rel}/verdict.csv",
                                                   recorded["verdict.csv"])
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
                note="logs/*.npz are gitignored; the control side of the unit "
                     "quantities is not reproducible from a fresh clone. The "
                     "controls carry no mob/zmax column, so submergence uses "
                     "p_hat == 0 and freezing uses p_hat < 1e-6 (exact on ReLU).")


def _s_floor(cfg: dict) -> dict:
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


def _s_ci(cfg: dict) -> dict:
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
    borrowed = _activation_cfg(cfg)
    print("[S-dial] registered valley/freeze depths vs numeric roots", flush=True)
    checks["S_dial"] = _s_dial(cfg)
    print("[S-fd] SiLU/GELU closed-form backward vs central difference", flush=True)
    checks["S_fd"] = _gd_s_fd(borrowed)
    print("[S-num] finiteness and float32 saturation depth", flush=True)
    checks["S_num"] = _gd_s_num(borrowed)
    print("[S-limit] beta -> large approaches ReLU", flush=True)
    checks["S_limit"] = _gd_s_limit(borrowed)
    for name in ("S_fd", "S_num", "S_limit"):
        checks[name]["borrowed_from"] = str(cfg["sanity"]["activation_checks_via"])
        checks[name]["borrowed_note"] = (
            "activation-only check; this run's SiLU/GELU are the same closed forms "
            "at the same dial (1.0) as gate_dial_0902's S_b1_1216 / G_b1_1216")
    print("[S-mob] unit logger identities on the `_off` path", flush=True)
    checks["S_mob"] = _s_mob_off(cfg, outdir)
    print("[S-log-b] logger is trajectory-neutral", flush=True)
    checks["S_log_b"] = _s_log_b(cfg, outdir)
    print("[S-pair] bit-identical init/teacher/stream with the parent run", flush=True)
    checks["S_pair"] = _s_pair(cfg, outdir)
    print("[S-ref] parent outputs match the parent provenance", flush=True)
    checks["S_ref"] = _s_ref(cfg)
    checks["S_floor"] = _s_floor(cfg)
    checks["S_ci"] = _s_ci(cfg)
    result = dict(pass_=all(bool(v.get("pass_", True)) for v in checks.values()),
                  experiment=EXPERIMENT, created=time.strftime("%Y-%m-%d %H:%M:%S %z"),
                  **checks)
    (outdir / "preflight.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")
    print(f"PREFLIGHT {'PASS' if result['pass_'] else 'FAIL'} -> {outdir}", flush=True)
    return result


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
            out[row["arm"]] = dict(u_5m=u5, u_1m=u1, log_u_5m=np.log10(u5),
                                   log_u_1m=np.log10(u1),
                                   n_onset_5m=int(row["n_onset_5m"]),
                                   n_onset_1m=int(row["n_onset_1m"]),
                                   source=str(path))
    missing = [a for a in CONTROL_ORDER if a not in out]
    if missing:
        raise SanityError(f"control arms missing from {path}: {missing}")
    return out


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
    import math
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
# ユニット別量（末尾窓）
# ---------------------------------------------------------------------------
def _unit_summary(cfg: dict, path: Path, geo: dict, *, proxy: bool) -> dict:
    """1 seed 分の末尾窓ユニット量。

    ``proxy=True`` は対照（``mob`` / ``zmax`` 列を持たない committed logs）で、
    沈下は ``p_hat == 0``、凍結は ``p_hat < 1e-6`` で作る。ReLU では
    $\\mathbb E_x\\varphi' = \\hat p$ なのでこの代用は厳密、ELU / leaky では成立
    しないので凍結率を NaN にする（引用禁止）。
    """
    threshold = float(_P(cfg)["design"]["freeze_phi_prime_threshold"])
    with np.load(path, allow_pickle=False) as z:
        step = z["step"].astype(np.int64)
        idx = _tail_index(cfg, step)
        p_hat = z["layer1_p_hat"][idx].astype(np.float64)
        zbar = z["layer1_zbar"][idx].astype(np.float64)
        mob = (z["layer1_mob"][idx].astype(np.float64)
               if "layer1_mob" in z.files else None)
        zmax = (z["layer1_zmax"][idx].astype(np.float64)
                if "layer1_zmax" in z.files else None)
        v_unit = (z["layer1_v_unit"][idx].astype(np.float64)
                  if "layer1_v_unit" in z.files else None)
    submerged = (zmax <= 0.0) if zmax is not None else (p_hat == 0.0)
    exact_relu = geo["activation"] == "relu"
    if mob is not None:
        frozen = np.abs(mob) < threshold
        frozen_source = "layer1_mob"
    elif exact_relu:
        frozen = p_hat < threshold
        frozen_source = "layer1_p_hat (exact on ReLU: E_x phi' = p_hat)"
    else:
        frozen = None
        frozen_source = "unavailable (no mob column, proxy invalid off ReLU)"
    u_star = geo["u_star_numeric"]
    beyond = ((zmax <= -u_star) if (zmax is not None and np.isfinite(u_star)
                                    and u_star > 0) else None)
    depths = -zbar[submerged] if submerged.any() else np.asarray([])
    out = dict(
        arm=geo["arm"], n_records=int(len(idx)), n_units=int(p_hat.shape[1]),
        submerged_frac=float(submerged.mean()),
        submerged_source=("layer1_zmax" if zmax is not None
                          else "layer1_p_hat == 0 (proxy)"),
        frozen_frac=(float(frozen.mean()) if frozen is not None else float("nan")),
        frozen_frac_among_submerged=(
            float(frozen[submerged].mean()) if (frozen is not None
                                                and submerged.any()) else float("nan")),
        frozen_source=frozen_source, frozen_threshold=threshold,
        beyond_valley_frac=(float(beyond.mean()) if beyond is not None
                            else float("nan")),
        m_minus=(float(np.median(mob[submerged]))
                 if (mob is not None and submerged.any()) else float("nan")),
        strict_dead_frac=float((p_hat == 0.0).mean()),
        depth_median=float(np.median(depths)) if depths.size else float("nan"),
        abs_v_median=(float(np.median(np.abs(v_unit))) if v_unit is not None
                      else float("nan")),
        is_proxy=bool(proxy))
    out["depth_deciles"] = ([float(v) for v in np.quantile(depths,
                                                           np.arange(1, 10) / 10.0)]
                            if depths.size else [float("nan")] * 9)
    return out


# ---------------------------------------------------------------------------
# 判定（spec §5・§6）
# ---------------------------------------------------------------------------
def _labels(cfg: dict, a1: bool, a2: bool, a3: bool, onset_5m: int) -> tuple[str, list]:
    L = _P(cfg)["labels"]
    lo, hi = [int(v) for v in L["slow_onset_range"]]
    hits = []
    if onset_5m <= int(L["needs_offset_onset_max"]):
        hits.append("FLIGHT_NEEDS_OFFSET")
    if a1 and a3 and not a2:
        hits.append("WORSE_UNTESTABLE_AT_CEILING")
    if a1 and a3:
        hits.append("FLIGHT_WITHOUT_ORACLE")
    if a3 and lo <= onset_5m <= hi:
        hits.append("FLIGHT_SLOW")
    for label in L["order"]:
        if label == "PARTIAL":
            return "PARTIAL", hits or ["PARTIAL"]
        if label in hits:
            return label, hits
    return "PARTIAL", hits or ["PARTIAL"]


def analyze(cfg: dict, outdir: Path, arms: list[str], sanity: dict,
            elapsed: dict, divergences: dict) -> dict:
    P, G = cfg["phase1"], _P(cfg)
    threshold = float(P["onset_threshold"])
    draws = _draws(cfg)
    controls = _load_controls(cfg)
    seeds = [int(v) for v in cfg["common"]["seeds"]]
    ref_logs = Path(ROOT) / str(cfg["controls"]["unit_source"])

    windows, units, revivals, capacity = {}, {}, {}, []
    for arm in arms:
        if arm in divergences:
            continue
        windows[arm] = _load_new_arm(cfg, outdir, arm)
        geo = _geometry(cfg, arm)
        units[arm] = [_unit_summary(cfg, outdir / "logs" / f"{arm}_seed{s}.npz",
                                    geo, proxy=False) for s in seeds]
        revivals[arm] = [_revival_counts(outdir / "logs" / f"{arm}_seed{s}.npz",
                                         geo["u_star_numeric"]) for s in seeds]
    for arm in CONTROL_ORDER:
        geo = _geometry(cfg, arm)
        paths = [Path(str(ref_logs).format(arm=arm, seed=s)) for s in seeds]
        if all(p.exists() for p in paths):
            units[arm] = [_unit_summary(cfg, p, geo, proxy=True) for p in paths]

    # S-cap: early 窓で U < 0.05 の seed が 9/10 以上
    for arm in windows:
        n_below = int((windows[arm]["early"]["u"] < float(cfg["sanity"]["s_cap_threshold"])).sum())
        capacity.append(dict(arm=arm, n_seeds_below=n_below,
                             required=int(cfg["sanity"]["s_cap_min_seeds"]),
                             pass_=bool(n_below >= int(cfg["sanity"]["s_cap_min_seeds"]))))
    capacity_undefined = [r["arm"] for r in capacity if not r["pass_"]]

    # S-mask: 窓の記録点数がタスク終端 10 点であること
    any_arm = next(iter(windows), None)
    s_mask = dict(checked=False)
    if any_arm is not None:
        step = windows[any_arm]["data"]["step"]
        idx_1m = _window_indices(step, int(P["task_period"]), list(P["window_1m_tasks"]))
        idx_5m = _window_indices(step, int(P["task_period"]), list(P["late_tasks_5m"]))
        s_mask = dict(checked=True, n_records_1m=int(len(idx_1m)),
                      n_records_5m=int(len(idx_5m)),
                      expected=int(P["window_records_per_10task_window"]),
                      spec_literal=int(P["spec_literal_records_per_window"]),
                      steps_1m=[int(step[i]) for i in idx_1m],
                      pass_=bool(len(idx_1m) == len(idx_5m)
                                 == int(P["window_records_per_10task_window"])),
                      resolution="task-end records only, inherited verbatim from "
                                 "the parent run's U_k")

    rows, contrasts = [], []
    for arm in arms:
        if arm in divergences:
            rows.append(dict(arm=arm, status=NUMERIC_DIVERGENCE))
            continue
        w = windows[arm]
        u5, u1 = w["5M"]["u"], w["1M"]["u"]
        n5, n1 = int((u5 >= threshold).sum()), int((u1 >= threshold).sum())
        cp5, cp1 = clopper_pearson(n5, len(u5)), clopper_pearson(n1, len(u1))
        unit = units[arm]
        base_unit = units.get(BASELINE)
        depth = np.asarray([u["depth_median"] for u in unit], dtype=np.float64)
        base_depth = (np.asarray([u["depth_median"] for u in base_unit],
                                 dtype=np.float64) if base_unit else None)
        deeper = (int(np.sum(depth > base_depth)) if base_depth is not None else -1)
        rows.append(dict(
            arm=arm, activation=_arm(cfg, arm)["activation"],
            dial=float(_arm(cfg, arm)["dial"]), target_dose="", status="COMPLETE",
            n_onset_5m=n5, cp95_5m_lo=cp5[0], cp95_5m_hi=cp5[1],
            U_5m_seed_values=json.dumps(u5.tolist()),
            median_log10_U_5m=float(np.median(w["5M"]["log_u"])),
            n_onset_1m=n1, cp95_1m_lo=cp1[0], cp95_1m_hi=cp1[1],
            U_1m_seed_values=json.dumps(u1.tolist()),
            median_log10_U_1m=float(np.median(w["1M"]["log_u"])),
            submerged_frac_5m=float(np.median([u["submerged_frac"] for u in unit])),
            depth_median_5m=float(np.median(depth)),
            frozen_frac_5m=float(np.median([u["frozen_frac"] for u in unit])),
            frozen_source=unit[0]["frozen_source"],
            beyond_valley_frac_5m=float(np.median([u["beyond_valley_frac"]
                                                   for u in unit])),
            m_minus_5m=float(np.median([u["m_minus"] for u in unit])),
            n_seeds_deeper_than_baseline=deeper,
            dose_5m_median=float(np.median(w["5M"]["metrics"]["layer1_dose"])),
            mu_norm_5m_median=float(np.median(w["5M"]["metrics"]["layer1_mu_norm"])),
            window="task_ends_only_10_records"))
        for window, key in (("5M", "log_u"), ("1M", "log_u")):
            delta = w[window][key] - controls[BASELINE][f"{key}_{window.lower()}"]
            ci = _ci(cfg, delta, draws)
            sign = _sign_test(delta)
            contrasts.append(dict(
                contrast=f"{arm}-{BASELINE}", window=window,
                point=float(np.median(delta)),
                percentile_ci_lo=ci["percentile_ci_lo"],
                percentile_ci_hi=ci["percentile_ci_hi"],
                studentized_ci_lo=ci["studentized_ci_lo"],
                studentized_ci_hi=ci["studentized_ci_hi"],
                ci_degenerate=ci["ci_degenerate"],
                n_positive=sign["n_positive"], n_negative=sign["n_negative"],
                p_sign_two_sided=sign["p_two_sided"],
                seed_values=json.dumps(delta.tolist()),
                baseline_source=controls[BASELINE]["source"],
                caveat="controls are committed values from a separate run; pairing "
                       "holds for init/teacher/input realization only"))

    # --- 事前予測の照合 ---
    pred = G["predictions"]
    checks: dict[str, dict] = {}
    label, co_satisfied = "INCONCLUSIVE_DIVERGENCE", []
    if PRIMARY in windows:
        row = next(r for r in rows if r["arm"] == PRIMARY)
        c1 = next(c for c in contrasts
                  if c["contrast"] == f"{PRIMARY}-{BASELINE}" and c["window"] == "1M")
        unit = units[PRIMARY]
        a1 = bool(row["n_onset_5m"] >= int(pred["A1"]["onset_5m_min"])
                  and row["n_onset_1m"] >= int(pred["A1"]["onset_1m_min"]))
        a2 = bool(c1["point"] >= float(pred["A2"]["min_delta_dex"])
                  and c1["n_positive"] >= int(pred["A2"]["min_sign_positive"]))
        a3 = bool(row["submerged_frac_5m"] >= float(pred["A3"]["submerged_frac_min"])
                  and row["depth_median_5m"] >= float(pred["A3"]["depth_median_min"])
                  and row["frozen_frac_5m"] >= float(pred["A3"]["frozen_frac_min"])
                  and row["n_seeds_deeper_than_baseline"]
                  >= int(pred["A3"]["deeper_than_baseline_paired_min"]))
        checks["A1"] = dict(hit=a1, n_onset_5m=row["n_onset_5m"],
                            n_onset_1m=row["n_onset_1m"],
                            note="the 1M side does not imply a difference from ReLU; "
                                 "R_off is itself 9/10 at 1M")
        checks["A2"] = dict(hit=a2, window="1M", point=c1["point"],
                            ci=[c1["percentile_ci_lo"], c1["percentile_ci_hi"]],
                            sign=f"{c1['n_positive']}:{c1['n_negative']}")
        checks["A3"] = dict(hit=a3, submerged_frac=row["submerged_frac_5m"],
                            depth_median=row["depth_median_5m"],
                            frozen_frac=row["frozen_frac_5m"],
                            frozen_source=row["frozen_source"],
                            n_seeds_deeper=row["n_seeds_deeper_than_baseline"])
        label, co_satisfied = _labels(cfg, a1, a2, a3, row["n_onset_5m"])
        rate = [float(r["valley_escapes_within_task"] or 0)
                / max(int(r["n_records"]) * int(r["n_units"]), 1)
                for r in revivals[PRIMARY]]
        checks["A5"] = dict(
            hit=bool(np.median(rate) >= float(pred["A5"]["rate_per_record_min"])),
            median_rate_per_unit_record=float(np.median(rate)),
            total_escapes_within_task=int(sum(int(r["valley_escapes_within_task"] or 0)
                                              for r in revivals[PRIMARY])),
            total_escapes_across_boundary=int(
                sum(int(r["valley_escapes_across_boundary"] or 0)
                    for r in revivals[PRIMARY])),
            report_only=True)
    secondary = str(G["predictions"]["A4"]["arm"])
    if secondary in windows:
        row = next(r for r in rows if r["arm"] == secondary)
        checks["A4"] = dict(
            hit=bool(row["n_onset_5m"] >= int(pred["A4"]["onset_5m_min"])
                     and row["depth_median_5m"] >= float(pred["A4"]["depth_median_min"])
                     and row["submerged_frac_5m"]
                     >= float(pred["A4"]["submerged_frac_min"])),
            n_onset_5m=row["n_onset_5m"], depth_median=row["depth_median_5m"],
            submerged_frac=row["submerged_frac_5m"], report_only=True)

    result = dict(label=label, co_satisfied=co_satisfied, checks=checks,
                  verdict_rows=rows, contrasts=contrasts, capacity=capacity,
                  capacity_undefined=capacity_undefined, s_mask=s_mask,
                  divergences=divergences, elapsed=elapsed,
                  controls={a: dict(n_onset_5m=c["n_onset_5m"],
                                    n_onset_1m=c["n_onset_1m"],
                                    median_log10_U_5m=float(np.median(c["log_u_5m"])),
                                    median_log10_U_1m=float(np.median(c["log_u_1m"])),
                                    source=c["source"])
                            for a, c in controls.items()})
    _write_outputs(cfg, outdir, result, units, revivals, seeds)
    _write_summary(cfg, outdir, result, units, sanity)
    return result


def _write_outputs(cfg: dict, outdir: Path, result: dict, units: dict,
                   revivals: dict, seeds: list[int]) -> None:
    write_csv(outdir / "verdict.csv", result["verdict_rows"])
    write_csv(outdir / "contrasts.csv", result["contrasts"])
    unit_rows = []
    for arm, entries in units.items():
        for seed, entry in zip(seeds, entries):
            row = {k: v for k, v in entry.items() if k != "depth_deciles"}
            row["seed"] = seed
            unit_rows.append(row)
    write_csv(outdir / "unit_summary.csv", unit_rows)
    depth_rows = []
    for arm, entries in units.items():
        for seed, entry in zip(seeds, entries):
            depth_rows.append(dict(arm=arm, seed=seed, is_proxy=entry["is_proxy"],
                                   **{f"d{i}": v for i, v in
                                      enumerate(entry["depth_deciles"], start=1)}))
    write_csv(outdir / "depth_hist.csv", depth_rows)
    revival_rows = []
    for arm, entries in revivals.items():
        for seed, entry in zip(seeds, entries):
            revival_rows.append(dict(arm=arm, seed=seed, **entry))
    write_csv(outdir / "revival.csv", revival_rows)


def _write_summary(cfg: dict, outdir: Path, result: dict, units: dict,
                   sanity: dict) -> None:
    G = _P(cfg)
    lines = [f"# {EXPERIMENT} — 谷の逃走・走 A（オラクルなしの自然な condA）", "",
             f"spec: `{cfg['spec']}` / vault: 可塑性喪失/spec/谷の逃走_走A_spec_0903.md",
             "", f"**登録判定（`{PRIMARY}`）: {result['label']}**",
             f"（満たした行: {', '.join(result['co_satisfied'])}）", "",
             "窓はタスク終端 10 点のみ（親走 `gate_dose_0830` の U_k と同じ作り方）。"
             "沈下は `layer1_zmax <= 0`、凍結は `|layer1_mob| < 1e-6`（**本走のロガー**。"
             "`gate_dial_0902` の `u_fr` 経由の凍結率とは別定義）。対照は別走の committed 値で、"
             "ユニット別量は `p_hat` 代用（ReLU では厳密・ELU/leaky では空欄）。", "",
             "## 主 endpoint", "",
             "| 腕 | 5M 発症 | 5M median log10 U | 1M 発症 | 1M median log10 U |",
             "| --- | --- | --- | --- | --- |"]
    for row in result["verdict_rows"]:
        if row.get("status") != "COMPLETE":
            lines.append(f"| {row['arm']} | {row.get('status')} | | | |")
            continue
        lines.append(f"| {row['arm']} | {row['n_onset_5m']}/10 | "
                     f"{row['median_log10_U_5m']:+.4f} | {row['n_onset_1m']}/10 | "
                     f"{row['median_log10_U_1m']:+.4f} |")
    for arm, c in result["controls"].items():
        lines.append(f"| {arm}（対照・転記） | {c['n_onset_5m']}/10 | "
                     f"{c['median_log10_U_5m']:+.4f} | {c['n_onset_1m']}/10 | "
                     f"{c['median_log10_U_1m']:+.4f} |")
    lines += ["", "## paired 差（対 `R_off`・別走の committed 値）", "",
              "| 対比 | 窓 | 点 | percentile CI | 符号 |", "| --- | --- | --- | --- | --- |"]
    for c in result["contrasts"]:
        lines.append(f"| {c['contrast']} | {c['window']} | {c['point']:+.4f} | "
                     f"[{c['percentile_ci_lo']:+.4f}, {c['percentile_ci_hi']:+.4f}] | "
                     f"{c['n_positive']}:{c['n_negative']} |")
    lines += ["", "**5M では ReLU が天井にいるので「ReLU より悪い」は 5M では検定できない。"
                  "検定は 1M 窓の行だけ。**", "",
              "## 末尾窓のユニット別量（5M・tasks 491-500）", "",
              "| 腕 | 沈下率 | 深さ中央値 | 凍結率 | 谷の向こう率 | 出所 |",
              "| --- | --- | --- | --- | --- | --- |"]
    for arm, entries in units.items():
        med = lambda key: float(np.median([e[key] for e in entries]))
        lines.append(f"| {arm} | {med('submerged_frac'):.4f} | {med('depth_median'):.2f} | "
                     f"{med('frozen_frac'):.4f} | {med('beyond_valley_frac'):.4f} | "
                     f"{entries[0]['frozen_source']} |")
    lines += ["", "## 事前予測の照合（spec §5・走の前に固定）", "",
              "| # | 的中 | 中身 |", "| --- | --- | --- |"]
    for name in ("A1", "A2", "A3", "A4", "A5"):
        check = result["checks"].get(name)
        if check is None:
            continue
        body = ", ".join(f"{k}={v}" for k, v in check.items()
                         if k not in ("hit", "note", "report_only"))
        mark = "✓" if check["hit"] else "✗"
        suffix = "（REPORT_ONLY）" if check.get("report_only") else ""
        lines.append(f"| {name}{suffix} | {mark} | {body} |")
    hit = sum(1 for c in result["checks"].values() if c["hit"])
    lines += ["", f"的中 {hit}/{len(result['checks'])}。", "",
              "## 前段チェック", ""]
    for name, value in sanity.items():
        if isinstance(value, dict) and "pass_" in value:
            lines.append(f"- {name}: {'PASS' if value['pass_'] else 'FAIL'}")
    lines += ["", "## 引用上の注意", "",
              "- 対照は**別走の committed 値**。ペアリングは init・教師・入力実現まで"
              "（step 1 以降は軌道が分岐する）。",
              "- 0/10 は「5M までに観測しなかった」（片側 95% 上限 0.2589）。",
              "- 凍結率・沈下率は出所と窓を添えて引用する。",
              "- `layer1_dose` は**測っただけ**の量（本走はオラクルを掛けていない）。", ""]
    (outdir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Provenance / run
# ---------------------------------------------------------------------------
def _provenance(cfg_path: Path, cfg: dict, outdir: Path, arms: list[str],
                sanity: dict, analysis: dict, elapsed: dict, started: float) -> dict:
    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                           text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--short"], cwd=ROOT,
                                        text=True).splitlines()
    except (OSError, subprocess.CalledProcessError):
        git_hash, dirty = None, []
    names = ("verdict.csv", "contrasts.csv", "unit_summary.csv", "revival.csv",
             "depth_hist.csv", "summary.md", "config_used.yaml")
    hashes = {n: _sha_file(outdir / n) for n in names if (outdir / n).exists()}
    hashes.update({f"logs/{p.name}": _sha_file(p)
                   for p in sorted((outdir / "logs").glob("*.npz"))})
    ref_dir = (Path(ROOT) / cfg["controls"]["reference_run"]).resolve()
    parent_prov = ref_dir / "provenance.json"
    parent = (json.loads(parent_prov.read_text(encoding="utf-8"))
              if parent_prov.exists() else {})
    reference_logs = {f"{arm}_seed{seed}": _sha_file(
        ref_dir / "logs" / f"{arm}_seed{seed}.npz")
        for arm in CONTROL_ORDER for seed in [int(v) for v in cfg["common"]["seeds"]]
        if (ref_dir / "logs" / f"{arm}_seed{seed}.npz").exists()}
    return dict(
        experiment=EXPERIMENT, created=time.strftime("%Y-%m-%d %H:%M:%S %z"),
        command=sys.argv, elapsed_sec=round(time.time() - started, 3),
        arm_elapsed_sec=elapsed, cwd=os.getcwd(), python=sys.version,
        platform=platform.platform(), torch=torch.__version__, numpy=np.__version__,
        device=cfg["common"]["device"], git_hash=git_hash, git_dirty=dirty,
        config=str(cfg_path), config_sha256=_sha_file(cfg_path),
        spec=str(Path(ROOT) / cfg["spec"]),
        spec_sha256=_sha_file(Path(ROOT) / cfg["spec"]),
        arms_run=list(arms), oracle=False, dose="off (natural condA)",
        generator_offset=int(cfg["common"]["generator_offset"]),
        generator_offset_note=("explicit 0: this run deliberately shares the parent "
                               "run's seed set and random stream (S-pair)."),
        freeze_definition=dict(
            column=str(_P(cfg)["design"]["freeze_source_column"]),
            threshold=float(_P(cfg)["design"]["freeze_phi_prime_threshold"]),
            note="this run's logger; NOT the u_fr-based freeze rate of gate_dial_0902"),
        window_definition=dict(task_ends_only=True, records_per_10task_window=10,
                               spec_literal=int(cfg["phase1"]
                                                ["spec_literal_records_per_window"])),
        baseline_reference=str(ref_dir), baseline_git_hash=parent.get("git_hash"),
        baseline_endpoint_source=str(ref_dir / "verdict.csv"),
        baseline_unit_source=str(ref_dir / "logs"),
        baseline_unit_source_is_gitignored=True,
        baseline_unit_proxy=("controls carry no mob/zmax column: submergence uses "
                             "p_hat == 0 and freezing uses p_hat < 1e-6, exact on "
                             "ReLU only"),
        reference_logs=reference_logs,
        sanity=sanity, analysis={k: v for k, v in analysis.items()
                                 if k not in ("verdict_rows", "contrasts")},
        output_sha256=hashes)


def run(cfg_path: Path, cfg: dict, device: str, outdir: Path, *,
        smoke: bool, analyze_only: bool) -> dict:
    started = time.time()
    outdir.mkdir(parents=True, exist_ok=True)
    require_omp(cfg)
    arms = list(ARM_ORDER)
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
    result = analyze(cfg, outdir, arms, sanity, elapsed, divergences)
    sanity = dict(sanity, S_cap=dict(pass_=not result["capacity_undefined"],
                                     rows=result["capacity"]),
                  S_mask=result["s_mask"])
    provenance = _provenance(cfg_path, cfg, outdir, arms, sanity, result, elapsed,
                             started)
    (outdir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")
    print(f"LABEL={result['label']} (co-satisfied: {result['co_satisfied']})",
          flush=True)
    print(f"ALL DONE -> {outdir}", flush=True)
    return dict(sanity=sanity, analysis=result)


def run_single_arm(cfg: dict, arm: str, device: str, outdir: Path,
                   total: int) -> dict:
    require_omp(cfg)
    outdir.mkdir(parents=True, exist_ok=True)
    seeds = [int(v) for v in cfg["common"]["seeds"]]
    every = int(cfg["common"]["lop_every"])
    if _complete_arm_logs(outdir, arm, seeds, total, every):
        print(f"[{arm}] complete logs found; nothing to do", flush=True)
        return dict(status="COMPLETE", resumed=True)
    return _run_arm(cfg, arm, device, outdir, seeds, total)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--arm", default=None, choices=list(ARM_ORDER))
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--outdir")
    args = parser.parse_args()
    if sum((args.preflight, args.smoke, args.arm is not None)) > 1:
        parser.error("--preflight / --smoke / --arm are exclusive")
    cfg_path = Path(args.config).resolve()
    cfg = load_config(str(cfg_path))
    device = pick_device(cfg)
    if device != "cpu":
        raise ValueError("valley_off is CPU-only")
    stage = ("preflight" if args.preflight else "smoke" if args.smoke
             else "analyze" if args.analyze_only else "run")
    validate_config(cfg, stage=stage)
    main_dir = Path(ROOT) / cfg["output"]["dir"]
    if args.preflight:
        preflight(cfg, Path(ROOT) / f"results/_preflight_{EXPERIMENT}")
        return
    if args.arm is not None:
        outdir = Path(args.outdir).resolve() if args.outdir else main_dir
        total = int(args.steps) if args.steps else int(cfg["common"]["total_steps"])
        run_single_arm(cfg, args.arm, device, outdir, total)
        return
    outdir = (Path(args.outdir).resolve() if args.outdir
              else Path(ROOT) / f"results/_smoke_{EXPERIMENT}" if args.smoke
              else main_dir)
    run(cfg_path, cfg, device, outdir, smoke=args.smoke,
        analyze_only=args.analyze_only)


if __name__ == "__main__":
    main()
