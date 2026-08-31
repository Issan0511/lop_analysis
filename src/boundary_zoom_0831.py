"""Two-stage boundary-window zoom registered in ``spec_boundary_zoom_0831``.

Stage 1 reruns the first 20 tasks while preserving the original ten-seed
vectorized RNG layout.  Only seed 0 is saved and analyzed.  No stage-2 code is
entered unless the registered timescale gate passes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from .common import ROOT, load_config
from .mlp2_phase1 import (
    P1_LOG_LAYER_KEYS,
    _arm,
    _base_cfg,
    _env_hashes,
    exact_layer_record_p1,
    setup_arm_p1,
    train_arm_p1,
)
from .mlp2_phase0 import LOG_UNIT_KEYS, PhaseRecorder, identity_sanity_pass
from .mlp2_phase1 import PhaseRecorderP1


ROOT = Path(ROOT)
CONFIG = ROOT / "configs" / "boundary_zoom_0831.yaml"
ARM = "L1w100_A1"


class GateUnresolved(RuntimeError):
    """The registered 20-step grid cannot place tau relative to the gate."""


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


def load_zoom_config(path: Path = CONFIG) -> tuple[dict[str, Any], dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        zoom = yaml.safe_load(handle)
    base_path = ROOT / zoom["base_config"]
    base = load_config(str(base_path))
    return zoom, base


def require_registered_environment(zoom: dict[str, Any]) -> None:
    expected = str(zoom["common"]["omp_num_threads"])
    if os.environ.get("OMP_NUM_THREADS") != expected:
        raise RuntimeError(f"OMP_NUM_THREADS must be {expected}")
    if torch.get_num_threads() != int(expected):
        raise RuntimeError("torch thread count differs from OMP_NUM_THREADS")


def stage1_probe_steps(zoom: dict[str, Any]) -> np.ndarray:
    s = zoom["stage1"]
    total = int(s["total_steps"])
    period = int(zoom["common"]["task_period"])
    coarse = int(zoom["common"]["coarse_every"])
    offsets = np.arange(
        int(s["fine_offset_min"]), int(s["fine_offset_max"]) + 1,
        int(s["fine_every"]), dtype=int,
    )
    steps = set(range(0, total + 1, coarse))
    # Twenty tasks contain nineteen fully observable task transitions.
    for boundary in range(period, total, period):
        steps.update(int(boundary + offset) for offset in offsets)
    result = np.array(sorted(step for step in steps if 0 <= step <= total), dtype=int)
    if len(result) != len(np.unique(result)):
        raise RuntimeError("S2 duplicate probe step")
    return result


def recorder_seed_payload(rec: PhaseRecorderP1, seed_index: int, arm: str,
                          seed: int) -> dict[str, np.ndarray]:
    payload: dict[str, Any] = {
        "step": np.asarray(rec.steps, dtype=np.int64),
        "arm": np.array(arm), "seed": np.int64(seed),
    }
    for key, value in rec.run.items():
        payload[key] = value[:, seed_index]
    payload["flip_state"] = rec.flip_state[:, seed_index]
    for layer_index, layer in enumerate(rec.layers, start=1):
        for key, value in layer.items():
            payload[f"layer{layer_index}_{key}"] = value[:, seed_index]
    return payload


def s0prime_compare(payload: dict[str, np.ndarray], reference: Path,
                    coarse_every: int, total: int) -> dict[str, Any]:
    differences: list[str] = []
    with np.load(reference, allow_pickle=False) as old:
        new_step = payload["step"]
        new_idx = np.flatnonzero((new_step % coarse_every == 0) & (new_step <= total))
        old_idx = np.flatnonzero(old["step"] <= total)
        if not np.array_equal(new_step[new_idx], old["step"][old_idx]):
            differences.append("step")
        shared = sorted(
            key for key in old.files
            if key in payload and key not in {"arm", "run_id", "state_hash_final",
                                               "seed", "task_period"}
        )
        for key in shared:
            if not np.array_equal(payload[key][new_idx], old[key][old_idx], equal_nan=True):
                differences.append(key)
        required = sorted(
            ["step", "flip_state", "signal_var", "residual_var", "unfit",
             "eval_loss_exact"]
            + [f"layer1_{key}" for key in LOG_UNIT_KEYS + P1_LOG_LAYER_KEYS]
        )
        missing = sorted(key for key in required if key not in shared and key != "step")
    return {
        "pass_": not differences and not missing,
        "differences": differences,
        "missing": missing,
        "n_common_steps": int(len(new_idx)),
        "reference": str(reference),
    }


def flip_sanity(payload: dict[str, np.ndarray], period: int) -> dict[str, Any]:
    step = payload["step"]
    flip = payload["flip_state"]
    changed = np.any(flip[1:] != flip[:-1], axis=1)
    preceding = step[:-1][changed]
    following = step[1:][changed]
    passed = bool(
        len(preceding) == 19
        and np.all(preceding % period == 0)
        and np.all((following - preceding) == 20)
    )
    return {
        "pass_": passed, "count": int(changed.sum()),
        "preceding_steps": preceding.tolist(), "following_steps": following.tolist(),
    }


def task_recovery_rows(payload: dict[str, np.ndarray], zoom: dict[str, Any]) -> pd.DataFrame:
    s = zoom["stage1"]
    step = payload["step"]
    metric_name = str(s["fit_metric"])
    metric = np.asarray(payload[metric_name], dtype=float)
    period = int(zoom["common"]["task_period"])
    low, high = [int(v) for v in s["baseline_offsets"]]
    fraction = float(s["recovery_fraction"])
    rows: list[dict[str, Any]] = []
    index = {int(value): i for i, value in enumerate(step)}
    for boundary in range(period, int(s["total_steps"]), period):
        offsets = np.arange(int(s["fine_offset_min"]), int(s["fine_offset_max"]) + 1,
                            int(s["fine_every"]), dtype=int)
        values = np.array([metric[index[boundary + int(offset)]] for offset in offsets])
        baseline_mask = (offsets >= low) & (offsets <= high)
        post_mask = offsets > 0
        baseline = float(np.median(values[baseline_mask]))
        post_offsets = offsets[post_mask]
        post_values = values[post_mask]
        excursion = np.abs(post_values - baseline)
        peak_i = int(np.argmax(excursion))
        peak_offset = int(post_offsets[peak_i])
        peak_excursion = float(excursion[peak_i])
        threshold = peak_excursion * fraction
        eligible = np.flatnonzero(
            (np.arange(len(post_offsets)) >= peak_i) & (excursion <= threshold)
        )
        recovery = int(post_offsets[eligible[0]]) if eligible.size else np.nan
        rows.append({
            "boundary_step": boundary, "metric": metric_name,
            "baseline": baseline, "peak_offset": peak_offset,
            "peak_value": float(post_values[peak_i]),
            "peak_excursion": peak_excursion, "one_over_e_excursion": threshold,
            "tau_fit": recovery, "censored_after_300": int(not eligible.size),
        })
    return pd.DataFrame(rows)


def classify_gate(rows: pd.DataFrame, zoom: dict[str, Any]) -> dict[str, Any]:
    valid = rows.loc[rows.censored_after_300 == 0, "tau_fit"].to_numpy(float)
    if len(valid) <= len(rows) // 2:
        return {
            "label": "TIMESCALE_GATE_UNRESOLVED", "tau_fit_median": np.nan,
            "basis": "more than half of task transitions are censored beyond +300",
            "n_task": len(rows), "n_uncensored": len(valid),
        }
    tau = float(np.median(valid))
    lo, hi = [float(v) for v in zoom["stage1"]["separable_exclusion_interval"]]
    label = "TIMESCALES_NOT_SEPARABLE" if lo <= tau <= hi else "TIMESCALES_SEPARABLE"
    return {
        "label": label, "tau_fit_median": tau,
        "basis": f"median uncensored tau_fit compared with [{lo:g},{hi:g}]",
        "n_task": len(rows), "n_uncensored": len(valid),
    }


def render_stage1_summary(gate: dict[str, Any], sanity: dict[str, Any],
                          elapsed: float) -> str:
    return "\n".join([
        "# boundary_zoom_0831 — 段階1", "",
        f"判定: **{gate['label']}**", "",
        f"- tau_fit 中央値: {gate['tau_fit_median']}",
        f"- 回復を観測した境界: {gate['n_uncensored']} / {gate['n_task']}",
        f"- S0'（既存1000-step点のbit一致）: {'PASS' if sanity['S0prime']['pass_'] else 'FAIL'}",
        f"- S1/S2 exact recorder: {'PASS' if sanity['S1_S2']['pass_'] else 'FAIL'}",
        f"- S2 duplicate probe: {'PASS' if sanity['S2_duplicate']['pass_'] else 'FAIL'}",
        f"- S3 OMP_NUM_THREADS=1: {'PASS' if sanity['S3_omp'] else 'FAIL'}",
        f"- S8 flip timing: {'PASS' if sanity['S8']['pass_'] else 'FAIL'}",
        f"- 実行時間: {elapsed:.1f} sec", "",
        "## 実装上の固定", "",
        "tau_fit は residual_var を用い、各境界の offset [-300,0] の中央値を境界前定常水準とした。境界後の最大絶対偏差から 1/e 以内へ初めて戻る20-step格子点を tau_fit とする。これは段階1の実行前に固定した。", "",
        "seed 0 の乱数列を既存runとbit一致させるため、内部では元runと同じ10 seedベクトルを進めた。保存・判定対象はseed 0のみで、他9 seedはRNG paddingである。", "",
        "段階1の tau_fit は seed 0・19境界の記述統計であり、水準として引用しない。", "",
    ])


def run_stage1(config_path: Path = CONFIG) -> dict[str, Any]:
    zoom, base = load_zoom_config(config_path)
    require_registered_environment(zoom)
    outdir = ROOT / zoom["output_dir"] / "stage1"
    outdir.mkdir(parents=True, exist_ok=True)
    steps = stage1_probe_steps(zoom)
    seeds = [int(v) for v in zoom["stage1"]["vectorized_padding_seeds"]]
    analyzed_seed = int(zoom["stage1"]["analyzed_seed"])
    seed_index = seeds.index(analyzed_seed)
    total = int(zoom["stage1"]["total_steps"])

    cfg = _base_cfg(base)
    cfg["common"]["seeds"] = seeds
    state = setup_arm_p1(cfg, _arm(base, ARM), "cpu")
    before, before_sanity = exact_layer_record_p1(
        state, float(base["phase1"]["sigma_degenerate_tol"])
    )
    if not identity_sanity_pass(before_sanity, float(base["sanity"]["s1_identity_tol"])):
        raise RuntimeError("pre-run exact identity failed")
    recorder = PhaseRecorderP1(
        steps.tolist(), state, float(base["phase1"]["sigma_degenerate_tol"]),
        float(base["sanity"]["s1_identity_tol"]),
    )
    started = time.time()
    elapsed = train_arm_p1(state, recorder, steps.tolist(), total, outdir, [])
    payload = recorder_seed_payload(recorder, seed_index, ARM, analyzed_seed)
    log_path = outdir / "L1w100_A1_seed0_fine.npz"
    np.savez_compressed(log_path, **payload)

    reference = ROOT / zoom["source_result"] / "logs" / f"{ARM}_seed{analyzed_seed}.npz"
    sanity = {
        "S0prime": s0prime_compare(
            payload, reference, int(zoom["common"]["coarse_every"]), total
        ),
        "S1_S2": recorder.sanity(),
        "S2_duplicate": {"pass_": len(steps) == len(np.unique(steps)),
                         "count": int(len(steps) - len(np.unique(steps)))},
        "S3_omp": os.environ.get("OMP_NUM_THREADS") == "1",
        "S8": flip_sanity(payload, int(zoom["common"]["task_period"])),
    }
    if not all([
        sanity["S0prime"]["pass_"], sanity["S1_S2"]["pass_"],
        sanity["S2_duplicate"]["pass_"], sanity["S3_omp"], sanity["S8"]["pass_"],
    ]):
        raise RuntimeError(f"stage1 sanity failed: {sanity}")

    tasks = task_recovery_rows(payload, zoom)
    gate = classify_gate(tasks, zoom)
    tasks.to_csv(outdir / "tau_fit_tasks.csv", index=False)
    pd.DataFrame([gate]).to_csv(outdir / "verdict.csv", index=False)
    (outdir / "summary.md").write_text(
        render_stage1_summary(gate, sanity, elapsed), encoding="utf-8"
    )
    provenance = {
        "experiment": "boundary_zoom_0831_stage1",
        "created": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "elapsed_sec": elapsed, "wall_elapsed_sec": time.time() - started,
        "git_hash": git("rev-parse", "HEAD"),
        "spec_commit": git("log", "-1", "--format=%H", "--", zoom["spec"]),
        "config": str(config_path), "config_sha256": sha256(config_path),
        "base_config": zoom["base_config"],
        "base_config_sha256": sha256(ROOT / zoom["base_config"]),
        "reference_log": str(reference.relative_to(ROOT)),
        "reference_log_sha256": sha256(reference),
        "implementation": "src/boundary_zoom_0831.py",
        "python": sys.version, "platform": platform.platform(),
        "torch": torch.__version__, "numpy": np.__version__,
        "analyzed_seed": analyzed_seed, "rng_padding_seeds": seeds,
        "n_probe": int(len(steps)), "sanity": sanity, "gate": gate,
    }
    (outdir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--stage1", action="store_true")
    args = parser.parse_args()
    if not args.stage1:
        parser.error("only --stage1 is implemented before the registered gate")
    result = run_stage1(args.config.resolve())
    print(json.dumps({"gate": result["gate"], "sanity": {
        key: value if isinstance(value, bool) else value.get("pass_")
        for key, value in result["sanity"].items()
    }}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
