"""Oracle exact-support centering experiment registered on 2026-08-31."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import yaml

from .common import ROOT, load_config
from .mlp2_phase0 import _sha_array, identity_sanity_pass
from .mlp2_phase1 import (
    NumericDivergenceError,
    PhaseRecorderP1,
    StreamDigest,
    _arm,
    _base_cfg,
    _env_hashes,
    _init_hashes,
    exact_layer_record_p1,
    grads_centered,
    setup_arm_p1,
    train_arm_p1,
    write_arm_logs_p1,
)


ROOT = Path(ROOT)
CONFIG = ROOT / "configs" / "center_oracle_0831.yaml"


class OracleSanityError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def load_oracle_config(path: Path = CONFIG) -> tuple[dict[str, Any], dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        oracle = yaml.safe_load(handle)
    return oracle, load_config(str(ROOT / oracle["base_config"]))


def require_environment(cfg: dict[str, Any]) -> None:
    expected = str(cfg["run"]["omp_num_threads"])
    if os.environ.get("OMP_NUM_THREADS") != expected:
        raise OracleSanityError(f"OMP_NUM_THREADS must be {expected}")
    if torch.get_num_threads() != int(expected):
        raise OracleSanityError("torch thread count differs from OMP_NUM_THREADS")
    if cfg["analysis"].get("alpha_sweep") is not False:
        raise OracleSanityError("registered execution excludes the optional alpha sweep")


def support_mean(st: dict[str, Any]) -> torch.Tensor:
    env = st["env"]
    free = torch.full(
        (env.R, env.m - env.f), 0.5, device=env.device,
        dtype=env.flip_state.dtype,
    )
    return torch.cat([env.flip_state, free], dim=1)


def set_support_mean(st: dict[str, Any]) -> torch.Tensor:
    mean = support_mean(st)
    st["running_mean"].copy_(mean)
    st["layer_means"][0] = st["running_mean"]
    return mean


def forward_exact(st: dict[str, Any], x: torch.Tensor):
    """One-layer forward with the current task's exact support mean."""
    mean = set_support_mean(st)
    x_in = x - mean
    net = st["net"]
    pre = torch.einsum("rhd,rd->rh", net.Ws[0], x_in) + net.bs[0]
    act = torch.relu(pre)
    yhat = (act * net.v).sum(dim=1) + net.c
    return [x_in], [pre], [act], yhat


def train_exact(
    st: dict[str, Any], recorder: PhaseRecorderP1 | None,
    probe_steps: Iterable[int], total: int, stream_hook=None,
) -> float:
    probe_set = {int(v) for v in probe_steps}
    net, env, teacher = st["net"], st["env"], st["teacher"]
    started = time.time()
    for t in range(total):
        if total >= 1_000_000 and t > 0 and t % 500_000 == 0:
            print(f"[L1w100_Aexact] {t:,}/{total:,} steps "
                  f"({time.time() - started:.1f}s)", flush=True)
        if recorder is not None and t in probe_set:
            set_support_mean(st)
            recorder(st, t)
        x = env.step()
        y = teacher(x)
        if stream_hook is not None:
            stream_hook(t, x, y)
        inputs, pres, acts, yhat = forward_exact(st, x)
        grads = grads_centered(net, inputs, pres, acts, yhat - y)
        net.sgd_step_layers(st["lr"], *grads)
    if recorder is not None and total in probe_set:
        set_support_mean(st)
        recorder(st, total)
    return time.time() - started


def checkpoint_hashes(path: Path) -> dict[str, Any]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    result = {f"net.{key}": _sha_array(value) for key, value in ck["net"].items()}
    result.update({f"teacher.{key}": _sha_array(value)
                   for key, value in ck["teacher"].items()})
    result["env.flip_state"] = _sha_array(ck["env"]["flip_state"])
    result["env.t"] = str(ck["env"]["t"])
    result["running_mean"] = _sha_array(ck["running_mean"])
    return result


def setup_reference_state(base: dict[str, Any], seeds: list[int]) -> dict[str, Any]:
    cfg = _base_cfg(base)
    cfg["common"]["seeds"] = seeds
    return setup_arm_p1(cfg, _arm(base, "L1w100_A1"), "cpu")


def state_hashes(st: dict[str, Any]) -> dict[str, Any]:
    result = {f"net.{key}": _sha_array(value)
              for key, value in st["net"].state_dict().items()}
    result.update({f"teacher.{key}": _sha_array(value)
                   for key, value in st["teacher"].state_dict().items()})
    result["env.flip_state"] = _sha_array(st["env"].flip_state)
    result["env.t"] = str(st["env"].t)
    result["running_mean"] = _sha_array(st["running_mean"])
    return result


def run_preflight(oracle: dict[str, Any], base: dict[str, Any]) -> dict[str, Any]:
    seeds = [int(v) for v in oracle["run"]["seeds"]]
    steps = int(oracle["run"]["preflight_steps"])
    checkpoint = ROOT / oracle["source_checkpoint"]
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    a1 = setup_reference_state(base, seeds)
    initial = state_hashes(a1)
    expected = checkpoint_hashes(checkpoint)
    state_differences = sorted(key for key, value in expected.items()
                               if initial.get(key) != value)
    a1_digest = StreamDigest()
    train_arm_p1(a1, lambda *_: None, [], steps, ROOT / oracle["output_dir"], [],
                 stream_hook=a1_digest)
    a1_final_env = _env_hashes(a1)

    exact = setup_reference_state(base, seeds)
    exact_initial = state_hashes(exact)
    exact_digest = StreamDigest()
    train_exact(exact, None, [], steps, stream_hook=exact_digest)
    exact_final_env = _env_hashes(exact)
    init_pair_differences = sorted(key for key, value in initial.items()
                                   if exact_initial.get(key) != value)
    stream_a1, stream_exact = a1_digest.digest(), exact_digest.digest()
    stream_differences = sorted(key for key in ("x", "y", "n")
                                if stream_a1[key] != stream_exact[key])
    env_differences = sorted(key for key, value in a1_final_env.items()
                             if exact_final_env.get(key) != value)
    result = {
        "pass_": not (state_differences or init_pair_differences
                       or stream_differences or env_differences),
        "S0_state_checkpoint_differences": state_differences,
        "S0_initial_pair_differences": init_pair_differences,
        "S0_stream_differences": stream_differences,
        "S0_final_env_differences": env_differences,
        "steps": steps, "seeds": seeds,
        "a1_stream": stream_a1, "aexact_stream": stream_exact,
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
    }
    if not result["pass_"]:
        raise OracleSanityError(f"preflight failed: {result}")
    return result


def numeric_divergence_selftest(base: dict[str, Any]) -> dict[str, Any]:
    st = setup_reference_state(base, [0, 1])
    set_support_mean(st)
    st["net"].Ws[0][1, 0, 0] = float("nan")
    rec = PhaseRecorderP1(
        [0], st, float(base["phase1"]["sigma_degenerate_tol"]),
        float(base["sanity"]["s1_identity_tol"]),
    )
    caught = False
    try:
        rec(st, 0)
    except NumericDivergenceError:
        caught = True
    return {"pass_": caught, "injected": "net.W1[seed1,unit0,input0]=NaN"}


def shared_draws(B: int, seed: int, n: int = 10) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(B, n))


def estimate(values: Iterable[float], draws: np.ndarray) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=float)
    if array.size != draws.shape[1] or not np.isfinite(array).all():
        raise OracleSanityError("registered estimate requires ten finite seed values")
    boot = np.median(array[draws], axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"point": float(np.median(array)), "ci_lo": float(lo),
            "ci_hi": float(hi), "n_seed": int(array.size)}


def unit_decomposition(logdir: Path, arm: str, seeds: list[int]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        with np.load(logdir / f"{arm}_seed{seed}.npz", allow_pickle=False) as z:
            step = z["step"].astype(int)
            M = z["layer1_M"].astype(float)
            B = z["layer1_B"].astype(float)
            beta = M + B
            p_hat = z["layer1_p_hat"]
            bmask = step[1:] % 10_000 == 1_000
            dbeta = np.diff(beta, axis=0)
            dead = p_hat == 0
            for unit in range(beta.shape[1]):
                rows.append({
                    "arm": arm, "seed": seed, "unit": unit,
                    "delta_beta_bnd": float(np.nansum(dbeta[bmask, unit])),
                    "delta_beta_int": float(np.nansum(dbeta[~bmask, unit])),
                    "strict_dead_final": int(dead[-1, unit]),
                    "continuous_dead_last1000": int(
                        dead[-1, unit] and np.all(dead[-1000:, unit])
                    ),
                })
    return pd.DataFrame(rows)


def final_env_sanity(outdir: Path, source: Path, seeds: list[int]) -> dict[str, Any]:
    differences: dict[str, list[str]] = {}
    for seed in seeds:
        with np.load(outdir / "logs" / f"L1w100_Aexact_seed{seed}.npz") as exact, \
             np.load(source / "logs" / f"L1w100_A1_seed{seed}.npz") as a1:
            exact_state = json.loads(str(exact["state_hash_final"]))
            a1_state = json.loads(str(a1["state_hash_final"]))
            bad = [key for key in ("env.flip_state", "env.t")
                   if exact_state.get(key) != a1_state.get(key)]
            if bad:
                differences[str(seed)] = bad
    return {"pass_": not differences, "differences": differences}


def step0_record_differences(outdir: Path, source: Path) -> list[str]:
    with np.load(outdir / "logs" / "L1w100_Aexact_seed0.npz") as exact, \
         np.load(source / "logs" / "L1w100_A1_seed0.npz") as a1:
        shared = sorted(set(exact.files) & set(a1.files))
        skip = {"arm", "run_id", "state_hash_final", "seed", "task_period", "step"}
        return [key for key in shared if key not in skip
                and not np.array_equal(np.asarray(exact[key])[0], np.asarray(a1[key])[0],
                                       equal_nan=True)]


def analyze(
    oracle: dict[str, Any], outdir: Path, source: Path,
    sanity: dict[str, Any], elapsed: float,
) -> dict[str, Any]:
    seeds = [int(v) for v in oracle["run"]["seeds"]]
    exact_units = unit_decomposition(outdir / "logs", "L1w100_Aexact", seeds)
    a1_units = unit_decomposition(source / "logs", "L1w100_A1", seeds)
    units = pd.concat([exact_units, a1_units], ignore_index=True)
    units.to_csv(outdir / "unit_decomposition.csv", index=False)

    seed_bnd = units.groupby(["arm", "seed"])["delta_beta_bnd"].median().unstack("arm")
    seed_int = units.groupby(["arm", "seed"])["delta_beta_int"].median().unstack("arm")
    ratio = (seed_bnd["L1w100_Aexact"].abs() / seed_bnd["L1w100_A1"].abs()).to_numpy()
    B = int(oracle["analysis"]["bootstrap_B"])
    draws = shared_draws(B, int(oracle["analysis"]["bootstrap_seed"]))
    ratio_est = estimate(ratio, draws)
    exact_bnd_est = estimate(seed_bnd["L1w100_Aexact"].to_numpy(), draws)
    a1_bnd_est = estimate(seed_bnd["L1w100_A1"].to_numpy(), draws)
    exact_int_est = estimate(seed_int["L1w100_Aexact"].to_numpy(), draws)
    a1_int_est = estimate(seed_int["L1w100_A1"].to_numpy(), draws)
    if exact_bnd_est["ci_lo"] <= 0 <= exact_bnd_est["ci_hi"]:
        p1 = "BOUNDARY_DESCENT_ELIMINATED"
    elif ratio_est["ci_hi"] < float(oracle["analysis"]["ratio_ema_upper"]):
        p1 = "EMA_LAG_IS_THE_CAUSE"
    elif ratio_est["ci_lo"] > float(oracle["analysis"]["ratio_shock_lower"]):
        p1 = "SWITCH_SHOCK_IS_THE_CAUSE"
    else:
        p1 = "BOTH_CONTRIBUTE"

    levels = units.groupby(["arm", "seed"]).agg(
        n_unit=("unit", "count"),
        n_final_dead=("strict_dead_final", "sum"),
        n_core_dead=("continuous_dead_last1000", "sum"),
    ).reset_index()
    levels["strict_dead_frac"] = levels["n_final_dead"] / levels["n_unit"]
    levels["core_dead_frac"] = np.where(
        levels["n_final_dead"] > 0,
        levels["n_core_dead"] / levels["n_final_dead"], np.nan,
    )
    wide_dead = levels.pivot(index="seed", columns="arm", values="strict_dead_frac")
    wide_core = levels.pivot(index="seed", columns="arm", values="core_dead_frac")
    dead_gap = estimate((wide_dead["L1w100_Aexact"]
                         - wide_dead["L1w100_A1"]).to_numpy(), draws)
    core_gap = estimate((wide_core["L1w100_Aexact"]
                         - wide_core["L1w100_A1"]).to_numpy(), draws)
    exact_dead_level = estimate(wide_dead["L1w100_Aexact"].to_numpy(), draws)
    a1_dead_level = estimate(wide_dead["L1w100_A1"].to_numpy(), draws)
    exact_core_level = estimate(wide_core["L1w100_Aexact"].to_numpy(), draws)
    a1_core_level = estimate(wide_core["L1w100_A1"].to_numpy(), draws)
    if dead_gap["ci_hi"] < 0:
        p2 = "ORACLE_REDUCES_DEATH"
    elif dead_gap["ci_lo"] <= 0 <= dead_gap["ci_hi"]:
        p2 = "NO_LEVEL_EFFECT"
    else:
        p2 = "ORACLE_INCREASES_DEATH"

    rows = [
        {"endpoint": "P1", "metric": "R_abs_boundary_ratio",
         **ratio_est, "label": p1,
         "basis": "seed-level abs(median exact boundary)/abs(median A1 boundary)"},
        {"endpoint": "P1", "metric": "Aexact_delta_beta_bnd",
         **exact_bnd_est, "label": p1,
         "basis": "seed median unit decomposition"},
        {"endpoint": "P1_REPORT_ONLY", "metric": "A1_delta_beta_bnd",
         **a1_bnd_est, "label": "REFERENCE",
         "basis": "committed mlp2_phase1_0829 log"},
        {"endpoint": "P1_REPORT_ONLY", "metric": "Aexact_delta_beta_int",
         **exact_int_est, "label": "REPORT_ONLY",
         "basis": "seed median unit decomposition"},
        {"endpoint": "P1_REPORT_ONLY", "metric": "A1_delta_beta_int",
         **a1_int_est, "label": "REFERENCE",
         "basis": "committed mlp2_phase1_0829 log"},
        {"endpoint": "P2", "metric": "strict_dead_frac_Aexact_minus_A1",
         **dead_gap, "label": p2, "basis": "paired seed difference at 5M"},
        {"endpoint": "P2_LEVEL", "metric": "strict_dead_frac_Aexact",
         **exact_dead_level, "label": "REPORT_ONLY", "basis": "5M seed level"},
        {"endpoint": "P2_LEVEL", "metric": "strict_dead_frac_A1",
         **a1_dead_level, "label": "REFERENCE", "basis": "5M seed level"},
        {"endpoint": "P2_REPORT_ONLY", "metric": "core_dead_frac_Aexact_minus_A1",
         **core_gap, "label": "REPORT_ONLY",
         "basis": "paired seed difference; continuous dead over final 1000 records"},
        {"endpoint": "P2_REPORT_ONLY", "metric": "core_dead_frac_Aexact",
         **exact_core_level, "label": "REPORT_ONLY",
         "basis": "continuous dead among final-dead units over final 1000 records"},
        {"endpoint": "P2_REPORT_ONLY", "metric": "core_dead_frac_A1",
         **a1_core_level, "label": "REFERENCE",
         "basis": "continuous dead among final-dead units over final 1000 records"},
    ]
    verdict = pd.DataFrame(rows)
    verdict.to_csv(outdir / "verdict.csv", index=False)
    levels.to_csv(outdir / "levels.csv", index=False)

    summary = "\n".join([
        "# center_oracle_0831", "",
        f"- P1: **{p1}**", f"- P2: **{p2}**", "",
        f"R = {ratio_est['point']:.4g} [{ratio_est['ci_lo']:.4g}, {ratio_est['ci_hi']:.4g}]",
        f"Aexact Δβ_boundary = {exact_bnd_est['point']:.4g} [{exact_bnd_est['ci_lo']:.4g}, {exact_bnd_est['ci_hi']:.4g}]",
        f"A1 Δβ_boundary = {a1_bnd_est['point']:.4g} [{a1_bnd_est['ci_lo']:.4g}, {a1_bnd_est['ci_hi']:.4g}]",
        f"Aexact Δβ_internal = {exact_int_est['point']:.4g} [{exact_int_est['ci_lo']:.4g}, {exact_int_est['ci_hi']:.4g}]",
        f"A1 Δβ_internal = {a1_int_est['point']:.4g} [{a1_int_est['ci_lo']:.4g}, {a1_int_est['ci_hi']:.4g}]",
        f"strict_dead_frac: Aexact {exact_dead_level['point']:.4g} / A1 {a1_dead_level['point']:.4g}; gap {dead_gap['point']:.4g} [{dead_gap['ci_lo']:.4g}, {dead_gap['ci_hi']:.4g}]",
        f"continuous-dead fraction among final dead: Aexact {exact_core_level['point']:.4g} / A1 {a1_core_level['point']:.4g}", "",
        "## 必須の交絡", "",
        "**オラクル中心化は µ を消すと同時に、タスク可識別性を完全に消す。** Aexact−A1 の差には「EMA遅れの差」と「可識別性の差」が同居する。書いてよいのは境界降下がEMA遅れ窓に起因する／しないという範囲であり、「µの効果を測った」「centeringを改善すればLoPを防げる」とは書かない。", "",
        "## S0 amendment", "",
        "元specの『step 0全記録量bit一致』はS-tautの `M≡0` と両立しないため、実行前amendmentに従い、介入前state・raw stream・final envを一致対象とした。介入後のstep 0相違列はprovenanceに列挙する。", "",
        f"実行時間: {elapsed:.1f} sec。alpha sweepは未実施。", "",
    ])
    (outdir / "summary.md").write_text(summary, encoding="utf-8")
    return {"P1": p1, "P2": p2, "ratio": ratio_est,
            "exact_boundary": exact_bnd_est, "dead_gap": dead_gap}


def reanalyze_saved(config_path: Path = CONFIG) -> dict[str, Any]:
    oracle, _ = load_oracle_config(config_path)
    outdir = ROOT / oracle["output_dir"]
    provenance_path = outdir / "provenance.json"
    if not provenance_path.exists():
        raise FileNotFoundError(provenance_path)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    source = ROOT / oracle["source_result"]
    result = analyze(
        oracle, outdir, source, provenance["sanity"],
        float(provenance["elapsed_sec"]),
    )
    provenance["analysis"] = result
    provenance["analysis_implementation_commit"] = git("rev-parse", "HEAD")
    provenance_path.write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    return provenance


def run(config_path: Path = CONFIG) -> dict[str, Any]:
    oracle, base = load_oracle_config(config_path)
    require_environment(oracle)
    outdir = ROOT / oracle["output_dir"]
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "config_used.yaml").write_text(
        yaml.safe_dump(oracle, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    preflight = run_preflight(oracle, base)
    s7 = numeric_divergence_selftest(base)
    if not s7["pass_"]:
        raise OracleSanityError("S7 numeric divergence selftest failed")

    seeds = [int(v) for v in oracle["run"]["seeds"]]
    total = int(oracle["run"]["total_steps"])
    every = int(oracle["run"]["probe_every"])
    steps = list(range(0, total + 1, every))
    state = setup_reference_state(base, seeds)
    state["arm"] = "L1w100_Aexact"
    for run_row in state["runs"]:
        run_row["run_id"] = f"L1w100_Aexact_seed{int(run_row['seed'])}"
    set_support_mean(state)
    before, before_sanity = exact_layer_record_p1(
        state, float(base["phase1"]["sigma_degenerate_tol"])
    )
    if not identity_sanity_pass(before_sanity, float(base["sanity"]["s1_identity_tol"])):
        raise OracleSanityError("Aexact pre-run S1/S2 failed")
    recorder = PhaseRecorderP1(
        steps, state, float(base["phase1"]["sigma_degenerate_tol"]),
        float(base["sanity"]["s1_identity_tol"]),
    )
    try:
        elapsed = train_exact(state, recorder, steps, total)
    except NumericDivergenceError as exc:
        event = dict(exc.event, status="NUMERIC_DIVERGENCE", rescue="none")
        (outdir / "numeric_divergence.json").write_text(
            json.dumps(event, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        raise
    recorder_sanity = recorder.sanity()
    if not recorder_sanity["pass_"]:
        raise OracleSanityError(f"Aexact S1/S2 failed: {recorder_sanity}")
    write_arm_logs_p1(outdir, "L1w100_Aexact", state, recorder)

    source = ROOT / oracle["source_result"]
    final_env = final_env_sanity(outdir, source, seeds)
    if not final_env["pass_"]:
        raise OracleSanityError(f"S0-final-env failed: {final_env}")
    M_all = recorder.layers[0]["M"].astype(float)
    B_all = recorder.layers[0]["B"].astype(float)
    p_hat_all = recorder.layers[0]["p_hat"]
    beta_all = M_all + B_all
    max_abs_M = float(np.nanmax(np.abs(M_all)))
    s1_left = int(np.sum((p_hat_all == 0) & (beta_all > -1)))
    s1_right = int(np.sum((beta_all <= -np.sqrt(5.0)) & (p_hat_all != 0)))
    with np.load(outdir / "logs" / "L1w100_Aexact_seed0.npz") as z:
        changed = np.any(z["flip_state"][1:] != z["flip_state"][:-1], axis=1)
        step = z["step"]
        s8 = bool(changed.sum() == 499 and np.all(step[:-1][changed] % 10_000 == 0))
    taut = {"pass_": max_abs_M == 0.0, "max_abs_M": max_abs_M}
    s1 = {"pass_": s1_left == 0 and s1_right == 0,
          "left_violations_all_seeds": s1_left,
          "right_violations_all_seeds": s1_right}
    if not taut["pass_"] or not s1["pass_"] or not s8:
        raise OracleSanityError(f"post-run sanity failed: taut={taut} S1={s1} S8={s8}")
    step0_differences = step0_record_differences(outdir, source)

    sanity = {
        "S0_preflight": preflight, "S0_final_env": final_env,
        "S0_step0_record_differences_expected": step0_differences,
        "S_taut": taut, "S1": s1, "S1_S2_recorder": recorder_sanity,
        "S3_omp": True, "S7": s7, "S8": {"pass_": s8, "count": 499},
    }
    result = analyze(oracle, outdir, source, sanity, elapsed)
    inputs = {
        oracle["spec"]: sha256(ROOT / oracle["spec"]),
        oracle["amendment"]: sha256(ROOT / oracle["amendment"]),
        oracle["base_config"]: sha256(ROOT / oracle["base_config"]),
        oracle["source_checkpoint"]: sha256(ROOT / oracle["source_checkpoint"]),
    }
    for seed in seeds:
        path = source / "logs" / f"L1w100_A1_seed{seed}.npz"
        inputs[str(path.relative_to(ROOT))] = sha256(path)
    provenance = {
        "experiment": "center_oracle_0831", "created": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "elapsed_sec": elapsed, "git_hash": git("rev-parse", "HEAD"),
        "spec_commit": git("log", "-1", "--format=%H", "--", oracle["spec"]),
        "amendment_commit": git("log", "-1", "--format=%H", "--", oracle["amendment"]),
        "source_result_commit": git("log", "-1", "--format=%H", "--", oracle["source_result"]),
        "config": str(config_path), "config_sha256": sha256(config_path),
        "input_sha256": inputs, "sanity": sanity, "analysis": result,
        "alpha_sweep": False, "python": sys.version, "platform": platform.platform(),
        "torch": torch.__version__, "numpy": np.__version__,
    }
    (outdir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    result = (reanalyze_saved(args.config.resolve()) if args.analyze_only
              else run(args.config.resolve()))
    def status(value: Any) -> Any:
        if isinstance(value, bool):
            return value
        if isinstance(value, dict):
            return value.get("pass_", "RECORDED")
        return "RECORDED"
    print(json.dumps({"analysis": result["analysis"], "sanity": {
        key: status(value)
        for key, value in result["sanity"].items()
    }}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
