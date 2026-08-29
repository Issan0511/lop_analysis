"""dose_const_5m_0830: fixed-input-mean dose response over five million steps.

The five intervention arms keep ``||E[x_in]||`` constant by solving an oracle
offset from the current task's flip-bit count.  ``dose_off`` is an untouched
depth-one control and must reproduce ``mlp2_phase0b_0829/L1w100`` bit for bit.

Stages are deliberately separated so the preregistration gates precede the
expensive run::

    OMP_NUM_THREADS=1 python -m src.dose_const_5m --preflight
    OMP_NUM_THREADS=1 python -m src.dose_const_5m --s0prime
    OMP_NUM_THREADS=1 python -m src.dose_const_5m

The intervention is an oracle measurement device, not a proposed learner.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from .common import ROOT, load_config, pick_device
from .mlp2_phase0 import (_sha_array, _sha_file, identity_sanity_pass,
                         require_omp, write_csv)
from .mlp2_phase0b import (_ci_components, _complete_arm_logs,
                           _window_indices)
from .mlp2_phase1 import (P1_LOG_LAYER_KEYS, PhaseRecorderP1, StreamDigest, _env_hashes,
                          _init_hashes, _seed_state_hashes_p1,
                          _unfit_two_summation_orders,
                          exact_layer_record_p1, forward_centered,
                          grads_centered, setup_arm_p1)


EXPERIMENT = "dose_const_5m_0830"
ARM_ORDER = ("dose_off", "dose933", "dose1004", "dose1075", "dose1146",
             "dose1216")
BASELINE_ARM = "dose_off"
FIXED_ARMS = ARM_ORDER[1:]
SMOKE_STEPS = 30_000
STATE_HASH_STEP = 1_000_000
EXTRA_LOG_KEYS = ("gamma", "gamma_negative", "mu_norm_formula",
                  "dose_formula", "mu_cos_off", "dose_relative_error")


def _arm(cfg: dict, name: str) -> dict:
    return next(a for a in cfg["arms"] if a["name"] == name)


def _target(arm_cfg: dict) -> float | None:
    value = arm_cfg.get("target_mu_norm")
    return None if value is None else float(value)


def _is_power_of_ten(value: float) -> bool:
    if not np.isfinite(value) or value <= 0:
        return False
    exponent = math.log10(value)
    return abs(exponent - round(exponent)) <= 1e-12


def validate_config(cfg: dict, *, stage: str) -> None:
    if stage not in {"preflight", "smoke", "s0prime", "full", "analyze"}:
        raise ValueError(f"unknown stage {stage!r}")
    C, A, I, P = cfg["common"], cfg["condA"], cfg["intervention"], cfg["phase1"]
    if [a["name"] for a in cfg["arms"]] != list(ARM_ORDER):
        raise ValueError(f"arms must be ordered as {ARM_ORDER}")
    expected = {
        "dose_off": (None, None, []),
        "dose933": (2.333, 9.33, [1]),
        "dose1004": (2.510, 10.04, [1]),
        "dose1075": (2.687, 10.75, [1]),
        "dose1146": (2.864, 11.46, [1]),
        "dose1216": (3.041, 12.16, [1]),
    }
    for arm in cfg["arms"]:
        target, dose, layers = expected[arm["name"]]
        if [int(v) for v in arm["hidden"]] != [100]:
            raise ValueError(f"{arm['name']} must be depth-one width 100")
        if [int(v) for v in arm.get("centered_layers", [])] != layers:
            raise ValueError(f"{arm['name']} centered_layers differs from the spec")
        got_target = _target(arm)
        if got_target != target or (dose is not None and float(arm["target_dose"]) != dose):
            raise ValueError(f"{arm['name']} target differs from the preregistration")
    if int(A["m"]) != 20 or int(A["f"]) != 15 or int(A["target_hidden"]) != 100:
        raise ValueError("the experiment requires condA m=20, f=15, teacher width=100")
    if list(A["T_values"]) != [10_000] or list(A["encodings"]) != ["std"]:
        raise ValueError("the experiment requires T=10000 and std input")
    if (I.get("name") != "oracle_fixed_mu_offset" or I.get("oracle") is not True
            or I.get("consumes_rng") is not False
            or float(I.get("center_alpha_compat")) != 0.01):
        raise ValueError("fixed-dose intervention metadata differs from the spec")
    if int(P["task_period"]) != 10_000:
        raise ValueError("task_period must be 10000")
    if list(P["one_m_tasks"]) != [91, 100] or list(P["five_m_tasks"]) != [491, 500]:
        raise ValueError("registered endpoint windows changed")
    if list(P["early_tasks"]) != [2, 11] or float(P["onset_threshold"]) != 0.05:
        raise ValueError("registered early window or onset threshold changed")
    if int(P["bootstrap_B"]) != 10_000 or int(P["bootstrap_seed"]) != 20_260_829:
        raise ValueError("registered bootstrap changed")
    if float(P["dose_relative_tolerance"]) != 1e-10:
        raise ValueError("S-dose tolerance must be 1e-10")
    if int(P["state_hash_step"]) != STATE_HASH_STEP:
        raise ValueError("state_hash_step must be 1000000")
    floor = P["unfit_floor"]
    if isinstance(floor, str):
        if floor != "CALIBRATED" or stage not in {"preflight", "smoke"}:
            raise ValueError("run --preflight to calibrate unfit_floor")
    elif not _is_power_of_ten(float(floor)):
        raise ValueError("unfit_floor must be the calibrated positive power of ten")
    calibration = P["floor_calibration"]
    expected_calibration = dict(steps=200_000, seeds=[0, 1, 2], arm=BASELINE_ARM,
                                n_checkpoints=20, method="two_summation_orders",
                                percentile=99, safety_factor=10,
                                round_to="power_of_10")
    if calibration != expected_calibration:
        raise ValueError("floor calibration differs from the preregistration")
    if cfg["pairing"]["paired_groups"] != [list(ARM_ORDER)]:
        raise ValueError("all six arms must be paired")
    if stage in {"s0prime", "full", "analyze"}:
        if int(C["total_steps"]) != 5_000_000 or list(C["seeds"]) != list(range(10)):
            raise ValueError("full run requires 5M steps and seeds 0..9")
        if str(C["device"]) != "cpu":
            raise ValueError("full run is CPU-only")


def gamma_for_k(k: torch.Tensor | np.ndarray | float, target_mu_norm: float):
    """Return the preregistered small quadratic root in float64."""
    if torch.is_tensor(k):
        kd = k.double()
        disc = (kd + 2.5).square() - 20.0 * (kd + 1.25 - target_mu_norm ** 2)
        if bool((disc < 0).any()):
            raise ValueError("negative gamma discriminant")
        return ((kd + 2.5) - disc.sqrt()) / 10.0
    kd = np.asarray(k, dtype=np.float64)
    disc = np.square(kd + 2.5) - 20.0 * (kd + 1.25 - target_mu_norm ** 2)
    if np.any(disc < 0):
        raise ValueError("negative gamma discriminant")
    return ((kd + 2.5) - np.sqrt(disc)) / 10.0


def _raw_mu(st: dict) -> torch.Tensor:
    free = st["env"].m - st["env"].f
    tail = torch.full((st["R"], free), 0.5, dtype=torch.float64,
                      device=st["env"].flip_state.device)
    return torch.cat([st["env"].flip_state.double(), tail], dim=1)


def _input_stats(st: dict, gamma_override: torch.Tensor | None = None) -> dict:
    raw = _raw_mu(st)
    target = st.get("target_mu_norm")
    if target is None:
        gamma = torch.full((st["R"],), float("nan"), dtype=torch.float64,
                           device=raw.device)
        transformed = raw
    else:
        k = st["env"].flip_state.double().sum(dim=1)
        gamma = (gamma_for_k(k, float(target)) if gamma_override is None
                 else gamma_override.double().to(raw.device))
        transformed = raw - 0.5 * gamma[:, None]
    mu_norm = transformed.norm(dim=1)
    raw_norm = raw.norm(dim=1)
    cosine = ((transformed * raw).sum(dim=1)
              / (mu_norm * raw_norm).clamp_min(1e-300))
    if target is None:
        relative = torch.full_like(mu_norm, float("nan"))
    else:
        relative = (mu_norm - float(target)).abs() / float(target)
    return dict(gamma=gamma, gamma_negative=(gamma < 0), mu_norm=mu_norm,
                dose=4.0 * mu_norm, cosine=cosine, relative_error=relative)


def _refresh_fixed_offset(st: dict) -> None:
    target = st.get("target_mu_norm")
    if target is None:
        return
    stats = _input_stats(st)
    offset = 0.5 * stats["gamma"][:, None].expand(-1, st["env"].m)
    st["running_mean"].copy_(offset.to(st["running_mean"].dtype))


def setup_arm_const(cfg: dict, arm_cfg: dict, device: str) -> dict:
    # setup_arm_p1 expects the compatibility alpha under this name.
    c = copy.deepcopy(cfg)
    c["intervention"]["center_alpha"] = float(
        c["intervention"]["center_alpha_compat"])
    st = setup_arm_p1(c, arm_cfg, device)
    st["target_mu_norm"] = _target(arm_cfg)
    st["target_dose"] = arm_cfg.get("target_dose")
    if st["target_mu_norm"] is not None:
        # The registered S-dose tolerance is 1e-10, tighter than a float32
        # offset can represent.  Keep the oracle parameter in float64 for the
        # exact-support evaluator; the online learner still consumes float32
        # inputs/weights and therefore receives the same value rounded at its
        # arithmetic boundary.
        exact_offset = torch.zeros(st["R"], st["env"].m, dtype=torch.float64,
                                   device=device)
        st["running_mean"] = exact_offset
        st["layer_means"][0] = exact_offset
    _refresh_fixed_offset(st)
    return st


def forward_const(st: dict, x: torch.Tensor):
    if st.get("target_mu_norm") is None:
        return forward_centered(st, x)
    _refresh_fixed_offset(st)
    net = st["net"]
    cur = x
    inputs, pres, acts = [], [], []
    for li, (W, b) in enumerate(zip(net.Ws, net.bs)):
        cur_in = cur - st["layer_means"][li].to(cur.dtype)
        pre = torch.einsum("rhd,rd->rh", W, cur_in) + b
        cur = torch.relu(pre)
        inputs.append(cur_in)
        pres.append(pre)
        acts.append(cur)
    yhat = (acts[-1] * net.v).sum(dim=1) + net.c
    return inputs, pres, acts, yhat


def save_checkpoint_const(st: dict, arm: str, step: int, outdir: Path) -> Path:
    path = outdir / "ckpts" / f"{arm}_step{step}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(step=step, arm=arm, net=st["net"].state_dict(),
                    env=st["env"].state_dict(), teacher=st["teacher"].state_dict(),
                    running_mean=st["running_mean"].clone(),
                    layer_means=[None if m is None else m.clone()
                                 for m in st["layer_means"]],
                    centered_layers=list(st["centered_layers"]),
                    target_mu_norm=st.get("target_mu_norm"), runs=st["runs"]), path)
    return path


def train_arm_const(st: dict, recorder, probe_steps, total: int, outdir: Path,
                    checkpoints, stream_hook=None) -> float:
    probe_set = {int(v) for v in probe_steps}
    checkpoint_set = {int(v) for v in checkpoints}
    net, env, teacher = st["net"], st["env"], st["teacher"]
    started = time.time()
    for step in range(total):
        if step in checkpoint_set:
            save_checkpoint_const(st, st["arm"], step, outdir)
        if step in probe_set:
            recorder(st, step)
        x = env.step()
        y = teacher(x)
        if stream_hook is not None:
            stream_hook(step, x, y)
        inputs, pres, acts, yhat = forward_const(st, x)
        grads = grads_centered(net, inputs, pres, acts, yhat - y)
        net.sgd_step_layers(st["lr"], *grads)
    if total in probe_set:
        recorder(st, total)
    if total in checkpoint_set:
        save_checkpoint_const(st, st["arm"], total, outdir)
    return time.time() - started


class ConstRecorder(PhaseRecorderP1):
    def __init__(self, steps: list[int], st: dict, sigma_tol: float,
                 identity_tol: float):
        super().__init__(steps, st, sigma_tol, identity_tol)
        n, runs = len(self.steps), st["R"]
        self.extra = {key: np.empty((n, runs), dtype=np.float64)
                      for key in EXTRA_LOG_KEYS}
        self.state_hash_1m: dict[int, dict[str, str]] = {}

    def __call__(self, st: dict, step: int) -> None:
        if st.get("target_mu_norm") is not None:
            _refresh_fixed_offset(st)
        super().__call__(st, step)
        index = self.index.get(int(step))
        if index is None:
            return
        stats = _input_stats(st)
        values = {
            "gamma": stats["gamma"],
            "gamma_negative": stats["gamma_negative"].double(),
            "mu_norm_formula": stats["mu_norm"],
            "dose_formula": stats["dose"],
            "mu_cos_off": stats["cosine"],
            "dose_relative_error": stats["relative_error"],
        }
        for key, value in values.items():
            self.extra[key][index] = value.detach().cpu().numpy()
        if int(step) == STATE_HASH_STEP:
            self.state_hash_1m = {
                int(run["seed"]): _seed_state_hashes_p1(st, ri)
                for ri, run in enumerate(st["runs"])
            }


def write_arm_logs_const(outdir: Path, arm: str, st: dict,
                         rec: ConstRecorder) -> list[Path]:
    logdir = outdir / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ri, run in enumerate(st["runs"]):
        seed = int(run["seed"])
        payload = dict(step=rec.steps, run_id=np.array(run["run_id"]),
                       arm=np.array(arm), seed=np.int64(seed),
                       task_period=np.int64(run["period"]),
                       target_mu_norm=np.float64(
                           np.nan if st.get("target_mu_norm") is None
                           else st["target_mu_norm"]),
                       target_dose=np.float64(
                           np.nan if st.get("target_dose") is None
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
        path = logdir / f"{arm}_seed{seed}.npz"
        np.savez_compressed(path, **payload)
        paths.append(path)
    return paths


def _run_arm(cfg: dict, arm: str, device: str, outdir: Path, seeds: list[int],
             total: int) -> dict:
    C, P = cfg["common"], cfg["phase1"]
    c = copy.deepcopy(cfg)
    c["common"]["seeds"] = seeds
    arm_cfg = _arm(cfg, arm)
    st = setup_arm_const(c, arm_cfg, device)
    probe_steps = list(range(0, total + 1, int(C["lop_every"])))
    if probe_steps[-1] != total:
        probe_steps.append(total)
    _, before = exact_layer_record_p1(st, float(P["sigma_degenerate_tol"]))
    if not identity_sanity_pass(before, float(cfg["sanity"]["s1_identity_tol"])):
        raise RuntimeError(f"{arm} initial exact-support identity failed")
    recorder = ConstRecorder(probe_steps, st, float(P["sigma_degenerate_tol"]),
                             float(cfg["sanity"]["s1_identity_tol"]))
    checkpoints = [int(v) for v in C.get("checkpoints", []) if int(v) <= total]
    print(f"[{arm}] seeds={seeds} steps={total:,} target={_target(arm_cfg)}", flush=True)
    elapsed = train_arm_const(st, recorder, probe_steps, total, outdir, checkpoints)
    sanity = recorder.sanity()
    if not sanity["pass_"]:
        raise RuntimeError(f"{arm} exact-support sanity failed: {sanity}")
    write_arm_logs_const(outdir, arm, st, recorder)
    result = dict(elapsed_sec=elapsed, sanity=sanity, final_env=_env_hashes(st))
    print(f"[{arm}] complete in {elapsed:.1f}s", flush=True)
    del recorder, st
    return result


def _compare_hash_groups(groups: dict[str, dict], reference: str,
                         keys: tuple[str, ...] | None = None) -> list[dict]:
    differences = []
    for arm, values in groups.items():
        if arm == reference:
            continue
        compare_keys = keys or tuple(groups[reference])
        for key in compare_keys:
            if values.get(key) != groups[reference].get(key):
                differences.append(dict(arm=arm, where=key))
    return differences


def _s_pair_and_dose(cfg: dict, device: str, outdir: Path) -> dict:
    steps = int(cfg["sanity"]["s_pair_steps"])
    probes = list(range(0, steps + 1, 10_000))
    init, final, stream, dose_rows = {}, {}, {}, []
    for arm in ARM_ORDER:
        st = setup_arm_const(cfg, _arm(cfg, arm), device)
        init[arm] = _init_hashes(st)
        digest = StreamDigest()
        recorder = ConstRecorder(probes, st,
                                 float(cfg["phase1"]["sigma_degenerate_tol"]),
                                 float(cfg["sanity"]["s1_identity_tol"]))
        print(f"[S-pair/S-dose] {arm} {steps:,} steps", flush=True)
        train_arm_const(st, recorder, probes, steps, outdir, [], stream_hook=digest)
        sanity = recorder.sanity()
        if not sanity["pass_"]:
            raise RuntimeError(f"preflight exact-support sanity failed for {arm}")
        final[arm] = _env_hashes(st)
        stream[arm] = digest.digest()
        target = _target(_arm(cfg, arm))
        if target is not None:
            for pi, step in enumerate(recorder.steps):
                for ri, run in enumerate(st["runs"]):
                    dose_rows.append(dict(
                        arm=arm, seed=int(run["seed"]), step=int(step),
                        target_mu_norm=target,
                        measured_mu_norm=float(recorder.layers[0]["mu_norm"][pi, ri]),
                        relative_error=float(
                            abs(recorder.layers[0]["mu_norm"][pi, ri] - target) / target)))
        del recorder, st
    init_diff = _compare_hash_groups(init, BASELINE_ARM)
    final_diff = _compare_hash_groups(final, BASELINE_ARM)
    stream_diff = _compare_hash_groups(stream, BASELINE_ARM, ("x", "y", "n"))
    tolerance = float(cfg["phase1"]["dose_relative_tolerance"])
    dose_fail = [row for row in dose_rows if row["relative_error"] > tolerance]
    return dict(
        pass_=bool(not init_diff and not final_diff and not stream_diff and not dose_fail),
        S_pair=dict(pass_=not (init_diff or final_diff or stream_diff),
                    reference=BASELINE_ARM, arms=list(ARM_ORDER), steps=steps,
                    init_differences=init_diff, final_differences=final_diff,
                    stream_differences=stream_diff, init_hashes=init,
                    final_env_hashes=final, stream_digests=stream),
        S_dose=dict(pass_=not dose_fail, tolerance=tolerance,
                    n_values=len(dose_rows), max_relative_error=float(max(
                        (row["relative_error"] for row in dose_rows), default=0.0)),
                    failures=dose_fail[:20]))


def _s_tautology(cfg: dict, device: str) -> dict:
    c = copy.deepcopy(cfg)
    c["common"]["seeds"] = [0, 1, 2]
    st = setup_arm_const(c, _arm(cfg, "dose1075"), device)
    original, _ = exact_layer_record_p1(st, float(cfg["phase1"]["sigma_degenerate_tol"]))
    original_unfit = original["run"]["unfit"].detach().cpu().numpy()
    saved_v = st["net"].v.clone()
    st["net"].v.zero_()
    mutant, _ = exact_layer_record_p1(st, float(cfg["phase1"]["sigma_degenerate_tol"]))
    mutant_unfit = mutant["run"]["unfit"].detach().cpu().numpy()
    st["net"].v.copy_(saved_v)
    endpoint_changes = bool(np.max(np.abs(original_unfit - mutant_unfit)) > 1e-6)
    stats = _input_stats(st)
    bad = _input_stats(st, stats["gamma"] + 1e-3)
    dose_mutant_rejected = bool(float(bad["relative_error"].max())
                                > float(cfg["phase1"]["dose_relative_tolerance"]))
    before_hash = _sha_array(st["env"].flip_state)
    st["env"].flip_state[0, 0] = 1 - st["env"].flip_state[0, 0]
    pair_mutant_rejected = before_hash != _sha_array(st["env"].flip_state)
    return dict(pass_=bool(endpoint_changes and dose_mutant_rejected
                           and pair_mutant_rejected),
                endpoint_not_defined_by_intervention=endpoint_changes,
                endpoint_original=original_unfit.tolist(),
                endpoint_vzero_mutant=mutant_unfit.tolist(),
                dose_checker_rejects_gamma_mutant=dose_mutant_rejected,
                pair_checker_rejects_flip_mutant=pair_mutant_rejected)


def _ci(cfg: dict, values: np.ndarray, draws: np.ndarray) -> dict:
    P = cfg["phase1"]
    return _ci_components(np.asarray(values, dtype=np.float64), draws, "median",
                          float(P["degenerate_se_tol"]),
                          float(P["degenerate_frac_max"]),
                          float(P["degenerate_width_ratio_max"]))


def _s_degeneracy(cfg: dict) -> dict:
    P = cfg["phase1"]
    n = len(cfg["common"]["seeds"])
    draws = np.random.default_rng(int(P["bootstrap_seed"])).integers(
        0, n, size=(int(P["bootstrap_B"]), n))
    result = _ci(cfg, np.zeros(n), draws)
    return dict(pass_=bool(result["ci_degenerate"]), result=result)


def _write_calibrated_floor(cfg_path: Path, floor: float) -> None:
    text = cfg_path.read_text(encoding="utf-8")
    updated, count = re.subn(
        r"^(\s*unfit_floor:\s*)[^#\r\n]+?(\s*(?:#.*)?)$",
        f"\\g<1>{floor:.1e}\\g<2>", text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise RuntimeError(f"could not replace unfit_floor in {cfg_path}")
    cfg_path.write_text(updated, encoding="utf-8")


def _s_floor_calibration(cfg_path: Path, cfg: dict, device: str,
                         outdir: Path) -> dict:
    K = cfg["phase1"]["floor_calibration"]
    total, n_checkpoints = int(K["steps"]), int(K["n_checkpoints"])
    checkpoints = list(range(total // n_checkpoints, total + 1,
                             total // n_checkpoints))
    c = copy.deepcopy(cfg)
    seeds = [int(v) for v in K["seeds"]]
    c["common"]["seeds"] = seeds
    st = setup_arm_const(c, _arm(cfg, str(K["arm"])), device)
    rows = []

    def record(state: dict, step: int) -> None:
        forward, reverse = _unfit_two_summation_orders(state)
        for ri, seed in enumerate(seeds):
            rows.append(dict(step=int(step), seed=seed,
                             unfit_forward=float(forward[ri]),
                             unfit_reverse=float(reverse[ri]),
                             abs_delta=float(abs(forward[ri] - reverse[ri]))))

    elapsed = train_arm_const(st, record, checkpoints, total, outdir, [])
    deltas = np.asarray([row["abs_delta"] for row in rows], dtype=np.float64)
    percentile = float(np.percentile(deltas, float(K["percentile"])))
    raw_floor = float(K["safety_factor"]) * percentile
    passed = bool(np.isfinite(raw_floor) and raw_floor > 0)
    calibrated = (float(10.0 ** math.ceil(math.log10(raw_floor)))
                  if passed else float("nan"))
    for row in rows:
        row.update(percentile_99=percentile, raw_floor=raw_floor,
                   calibrated_floor=calibrated)
    outdir.mkdir(parents=True, exist_ok=True)
    write_csv(outdir / "floor_calibration.csv", rows)
    result = dict(pass_=bool(passed and _is_power_of_ten(calibrated)),
                  arm=str(K["arm"]), steps=total, seeds=seeds,
                  checkpoints=checkpoints, n_values=len(rows),
                  percentile_99=percentile, raw_floor=raw_floor,
                  calibrated_floor=calibrated, elapsed_sec=elapsed)
    if result["pass_"]:
        _write_calibrated_floor(cfg_path, calibrated)
    return result


def preflight(cfg_path: Path, cfg: dict, device: str, outdir: Path) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    omp = require_omp(cfg)
    paired = _s_pair_and_dose(cfg, device, outdir / "paired")
    print(f"[S-pair] {'PASS' if paired['S_pair']['pass_'] else 'FAIL'}  "
          f"[S-dose] {'PASS' if paired['S_dose']['pass_'] else 'FAIL'}", flush=True)
    tautology = _s_tautology(cfg, device)
    print(f"[S-tautology] {'PASS' if tautology['pass_'] else 'FAIL'}", flush=True)
    degeneracy = _s_degeneracy(cfg)
    print(f"[S-degeneracy] {'PASS' if degeneracy['pass_'] else 'FAIL'}", flush=True)
    print("[S6] dose_off seeds 0..2, 200k, two summation orders", flush=True)
    floor = _s_floor_calibration(cfg_path, cfg, device, outdir)
    print(f"[S6] {'PASS' if floor['pass_'] else 'FAIL'} ({floor['elapsed_sec']:.1f}s)",
          flush=True)
    result = dict(pass_=bool(omp["pass_"] and paired["pass_"]
                             and tautology["pass_"] and degeneracy["pass_"]
                             and floor["pass_"]),
                  S1_omp=omp, S_pair=paired["S_pair"], S_dose=paired["S_dose"],
                  S_tautology=tautology, S_degeneracy=degeneracy, S6=floor)
    (outdir / "preflight.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    if not result["pass_"]:
        raise RuntimeError("preflight failed; full run is blocked")
    return result


def _compare_reference_log(ours: Path, reference: Path) -> list[dict]:
    ignored = {"run_id", "arm"}
    with np.load(ours, allow_pickle=False) as a, np.load(reference, allow_pickle=False) as b:
        differences = []
        for key in sorted(set(b.files) - ignored):
            if key not in a.files:
                differences.append(dict(column=key, reason="missing in dose_off"))
            elif _sha_array(a[key]) != _sha_array(b[key]):
                differences.append(dict(column=key, reason="hash mismatch"))
    return differences


def _checkpoint_hashes(path: Path) -> dict[str, str]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    result = {f"net.{key}": _sha_array(value) for key, value in ck["net"].items()}
    result.update({f"teacher.{key}": _sha_array(value)
                   for key, value in ck["teacher"].items()})
    result.update(env_flip_state=_sha_array(ck["env"]["flip_state"]),
                  env_t=str(ck["env"]["t"]),
                  running_mean=_sha_array(ck["running_mean"]))
    return result


def s0prime(cfg: dict, device: str, outdir: Path) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    require_omp(cfg)
    C = cfg["common"]
    seeds, total = [int(v) for v in C["seeds"]], int(C["total_steps"])
    reference_dir = (Path(ROOT) / cfg["sanity"]["s0_prime_baseline_ref"]).resolve()
    if not _complete_arm_logs(outdir, BASELINE_ARM, seeds, total, int(C["lop_every"])):
        _run_arm(cfg, BASELINE_ARM, device, outdir, seeds, total)
    differences, missing = [], []
    for seed in seeds:
        ours = outdir / "logs" / f"{BASELINE_ARM}_seed{seed}.npz"
        reference = reference_dir / "logs" / f"L1w100_seed{seed}.npz"
        if not reference.exists():
            missing.append(str(reference))
        else:
            differences.extend(dict(seed=seed, **row)
                               for row in _compare_reference_log(ours, reference))
    ours_ckpt = outdir / "ckpts" / f"{BASELINE_ARM}_step{total}.pt"
    reference_ckpt = reference_dir / "ckpts" / f"L1w100_step{total}.pt"
    state_differences = []
    expected_state = actual_state = {}
    if not ours_ckpt.exists() or not reference_ckpt.exists():
        missing.extend(str(p) for p in (ours_ckpt, reference_ckpt) if not p.exists())
    else:
        expected_state = _checkpoint_hashes(reference_ckpt)
        actual_state = _checkpoint_hashes(ours_ckpt)
        state_differences = [key for key in expected_state
                             if expected_state[key] != actual_state.get(key)]
    hash_1m_missing = []
    for seed in seeds:
        with np.load(outdir / "logs" / f"{BASELINE_ARM}_seed{seed}.npz",
                     allow_pickle=False) as z:
            if not json.loads(str(z["state_hash_1m"])):
                hash_1m_missing.append(seed)
    result = dict(pass_=bool(not missing and not differences
                             and not state_differences and not hash_1m_missing),
                  arm=BASELINE_ARM, reference=str(reference_dir),
                  reference_arm="L1w100", seeds=seeds, total_steps=total,
                  missing=missing, column_differences=differences,
                  state_differences=state_differences,
                  state_hash_1m_missing_seeds=hash_1m_missing,
                  expected_state_hash=expected_state, actual_state_hash=actual_state)
    (outdir / "s0prime.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[S0'] {'PASS' if result['pass_'] else 'FAIL'}", flush=True)
    if not result["pass_"]:
        raise RuntimeError("S0' failed; full run is blocked")
    return result


def clopper_pearson(successes: int, trials: int, alpha: float = 0.05) -> tuple[float, float]:
    """Two-sided exact binomial interval without a SciPy dependency."""
    x, n = int(successes), int(trials)
    if not 0 <= x <= n or n <= 0:
        raise ValueError("invalid binomial count")

    def cdf(value: int, p: float) -> float:
        return float(sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i)
                         for i in range(value + 1)))

    def upper_tail(value: int, p: float) -> float:
        return 1.0 - cdf(value - 1, p)

    if x == 0:
        lower = 0.0
    else:
        lo, hi = 0.0, 1.0
        for _ in range(100):
            mid = (lo + hi) / 2
            if upper_tail(x, mid) < alpha / 2:
                lo = mid
            else:
                hi = mid
        lower = (lo + hi) / 2
    if x == n:
        upper = 1.0
    else:
        lo, hi = 0.0, 1.0
        for _ in range(100):
            mid = (lo + hi) / 2
            if cdf(x, mid) > alpha / 2:
                lo = mid
            else:
                hi = mid
        upper = (lo + hi) / 2
    return float(lower), float(upper)


def _load_arm(cfg: dict, outdir: Path, arm: str) -> dict:
    seeds = [int(v) for v in cfg["common"]["seeds"]]
    items = []
    for seed in seeds:
        with np.load(outdir / "logs" / f"{arm}_seed{seed}.npz", allow_pickle=False) as z:
            items.append({key: z[key].copy() for key in (
                "step", "unfit", "eval_loss_exact", "gamma", "gamma_negative",
                "mu_cos_off", "dose_relative_error", "layer1_mu_norm",
                "layer1_dose", "layer1_eff_rank", "layer1_alive",
                "layer1_strict_dead", "layer1_w_norm_median", "layer1_median_M")})
    result = {"step": items[0]["step"]}
    for key in items[0]:
        if key != "step":
            result[key] = np.stack([item[key] for item in items], axis=1)
    return result


def _window_values(data: dict, period: int, tasks: list[int], floor: float) -> dict:
    idx = _window_indices(np.asarray(data["step"]), period, tasks)
    raw_u = np.asarray(data["unfit"], dtype=np.float64)[idx].mean(axis=0)
    u = np.maximum(raw_u, floor)
    return dict(index=idx, raw_u=raw_u, u=u, log_u=np.log10(u),
                eval_loss=np.asarray(data["eval_loss_exact"])[idx].mean(axis=0),
                floor_fraction=float(np.mean(np.asarray(data["unfit"])[idx] <= floor)))


def _band_verdict(g1: float | None, g5: float | None) -> str:
    if g1 is None:
        return "ANCHOR_FAILED"
    if g5 is None:
        return "BAND_NOT_REACHED_BY_5M"
    if g5 < g1:
        return "BAND_MOVES_DOWN_BY_5M"
    if g5 == g1:
        return "BAND_STABLE_TO_5M"
    return "BAND_MOVES_UP_BY_5M"


def analyze(cfg: dict, outdir: Path, sanity: dict, elapsed: dict) -> dict:
    P = cfg["phase1"]
    period, floor = int(P["task_period"]), float(P["unfit_floor"])
    seeds = [int(v) for v in cfg["common"]["seeds"]]
    rng = np.random.default_rng(int(P["bootstrap_seed"]))
    draws = rng.integers(0, len(seeds), size=(int(P["bootstrap_B"]), len(seeds)))
    data = {arm: _load_arm(cfg, outdir, arm) for arm in ARM_ORDER}
    windows = {
        arm: {
            "1M": _window_values(data[arm], period, list(P["one_m_tasks"]), floor),
            "5M": _window_values(data[arm], period, list(P["five_m_tasks"]), floor),
            "early": _window_values(data[arm], period, list(P["early_tasks"]), floor),
        } for arm in ARM_ORDER
    }
    threshold = float(P["onset_threshold"])
    onset = {window: {arm: int(np.sum(windows[arm][window]["raw_u"] >= threshold))
                      for arm in ARM_ORDER} for window in ("1M", "5M")}
    fixed_by_dose = sorted(FIXED_ARMS, key=lambda name: float(_arm(cfg, name)["target_dose"]))

    def transition(window: str) -> float | None:
        for arm in fixed_by_dose:
            if onset[window][arm] >= int(P["onset_present_min"]):
                return float(_arm(cfg, arm)["target_dose"])
        return None

    g1, g5 = transition("1M"), transition("5M")
    band = _band_verdict(g1, g5)
    lower_count = onset["5M"]["dose933"]
    count_verdict = ("ONSET_SPREADS_BY_5M" if lower_count >= 5 else
                     "NO_ONSET_BY_5M" if lower_count == 0 else
                     "PARTIAL_ONSET_BY_5M")
    saturated = {
        window: bool(all(value == 0 for value in onset[window].values())
                     or all(value == len(seeds) for value in onset[window].values()))
        for window in ("1M", "5M")
    }
    main_verdict = ("BINARY_SATURATED_USE_LEVELS" if saturated["5M"]
                    else count_verdict)

    delta_ci = {}
    for arm in ARM_ORDER:
        values = windows[arm]["5M"]["log_u"] - windows[arm]["1M"]["log_u"]
        delta_ci[arm] = dict(values=values, ci=_ci(cfg, values, draws))

    adjacent = []
    jumps = {}
    for window in ("1M", "5M"):
        signed = []
        for low, high in zip(fixed_by_dose[:-1], fixed_by_dose[1:]):
            values = windows[high][window]["log_u"] - windows[low][window]["log_u"]
            ci = _ci(cfg, values, draws)
            signed.append(abs(ci["point"]))
            adjacent.append(dict(window=window, low_arm=low, high_arm=high,
                                 seed_values=json.dumps(values.tolist()), **ci))
        jumps[window] = float(max(signed))

    verdict_rows, response_rows = [], []
    for arm in ARM_ORDER:
        arm_cfg = _arm(cfg, arm)
        delta = delta_ci[arm]
        for window in ("1M", "5M"):
            count = onset[window][arm]
            cp_lo, cp_hi = clopper_pearson(count, len(seeds))
            w = windows[arm][window]
            row = dict(
                arm=arm,
                target_mu_norm="" if _target(arm_cfg) is None else _target(arm_cfg),
                target_dose="" if arm_cfg.get("target_dose") is None
                else float(arm_cfg["target_dose"]),
                window=window, n_onset=count, n_seed=len(seeds),
                onset_threshold=threshold, cp95_lo=cp_lo, cp95_hi=cp_hi,
                U_seed_values=json.dumps(w["u"].tolist()),
                raw_U_seed_values=json.dumps(w["raw_u"].tolist()),
                median_log10_U=float(np.median(w["log_u"])),
                median_eval_loss_exact=float(np.median(w["eval_loss"])),
                floor_fraction=w["floor_fraction"],
                delta_5m_minus_1m_seed_values=json.dumps(delta["values"].tolist()),
                delta_point=delta["ci"]["point"],
                delta_percentile_ci_lo=delta["ci"]["percentile_ci_lo"],
                delta_percentile_ci_hi=delta["ci"]["percentile_ci_hi"],
                delta_studentized_ci_lo=delta["ci"]["studentized_ci_lo"],
                delta_studentized_ci_hi=delta["ci"]["studentized_ci_hi"],
                CI_DEGENERATE=delta["ci"]["ci_degenerate"],
                jump_J=jumps[window], transition_g_star=g1 if window == "1M" else g5,
                binary_saturated=int(saturated[window]),
                count_verdict=count_verdict, main_verdict=main_verdict,
                band_verdict=band)
            verdict_rows.append(row)
            response_rows.append(dict(
                arm=arm, window=window,
                target_mu_norm=row["target_mu_norm"], target_dose=row["target_dose"],
                n_onset=count, median_log10_U=row["median_log10_U"],
                median_U=float(np.median(w["u"])),
                median_eval_loss_exact=row["median_eval_loss_exact"],
                jump_J=jumps[window]))
    write_csv(outdir / "verdict.csv", verdict_rows)
    write_csv(outdir / "dose_response.csv", response_rows)
    write_csv(outdir / "ci.csv", adjacent)

    report_rows = []
    metric_keys = ("layer1_strict_dead", "layer1_alive", "layer1_eff_rank",
                   "layer1_w_norm_median", "layer1_median_M", "layer1_mu_norm",
                   "layer1_dose", "gamma", "gamma_negative", "mu_cos_off")
    for arm in ARM_ORDER:
        for window, task_range in (("early", list(P["early_tasks"])),
                                   ("1M", list(P["one_m_tasks"])),
                                   ("5M", list(P["five_m_tasks"]))):
            idx = _window_indices(data[arm]["step"], period, task_range)
            for metric in metric_keys:
                values = np.asarray(data[arm][metric], dtype=np.float64)[idx]
                finite = values[np.isfinite(values)]
                report_rows.append(dict(
                    arm=arm, window=window, metric=metric,
                    median=float(np.median(finite)) if finite.size else "",
                    q05=float(np.quantile(finite, .05)) if finite.size else "",
                    q95=float(np.quantile(finite, .95)) if finite.size else "",
                    n_finite=int(finite.size)))
    write_csv(outdir / "report_only.csv", report_rows)

    def mark(node: dict | None) -> str:
        return "PASS" if node and node.get("pass_") else "FAIL"

    lines = [f"# {EXPERIMENT} summary", "",
             "## Verdict", "",
             f"- Main: **{main_verdict}**",
             f"- Registered count verdict: **{count_verdict}**",
             f"- Band: **{band}** (g*(1M)={g1}, g*(5M)={g5})",
             f"- Binary saturation: 1M={saturated['1M']}, 5M={saturated['5M']}",
             f"- Jump J: 1M={jumps['1M']:.6g}, 5M={jumps['5M']:.6g}", "",
             "## Dose response", "",
             "| arm | n onset 1M | n onset 5M | median log10 U 1M | median log10 U 5M | Δ median [percentile 95% CI] |",
             "|---|---:|---:|---:|---:|---:|"]
    for arm in ARM_ORDER:
        ci = delta_ci[arm]["ci"]
        lines.append(
            f"| {arm} | {onset['1M'][arm]} | {onset['5M'][arm]} | "
            f"{np.median(windows[arm]['1M']['log_u']):.6g} | "
            f"{np.median(windows[arm]['5M']['log_u']):.6g} | {ci['point']:.6g} "
            f"[{ci['percentile_ci_lo']:.6g}, {ci['percentile_ci_hi']:.6g}] |")
    zero_upper = 1.0 - 0.05 ** (1.0 / len(seeds))
    lines += ["", "0/10 の片側95%上限は " + f"{zero_upper:.4f}。",
              "n_onset が全腕同一端点に飽和した窓は、水準側を判定基底として扱う。", "",
              "## REPORT_ONLY at 5M", "",
              "| arm | strict_dead | alive | eff_rank | ||mu|| | cos(mu, mu_off) |",
              "|---|---:|---:|---:|---:|---:|"]
    keyed = {(r["arm"], r["window"], r["metric"]): r for r in report_rows}
    for arm in ARM_ORDER:
        def value(metric: str) -> str:
            got = keyed[(arm, "5M", metric)]["median"]
            return "—" if got == "" else f"{got:.6g}"
        lines.append(f"| {arm} | {value('layer1_strict_dead')} | "
                     f"{value('layer1_alive')} | {value('layer1_eff_rank')} | "
                     f"{value('layer1_mu_norm')} | {value('mu_cos_off')} |")
    lines += ["", "## Sanity", "",
              f"- S0' bit reproduction: **{mark(sanity.get('S0prime'))}**",
              f"- S-pair preflight: **{mark(sanity.get('S_pair'))}**",
              f"- S-pair final: **{mark(sanity.get('S_pair_final'))}**",
              f"- S-dose preflight: **{mark(sanity.get('S_dose'))}**",
              f"- S-dose full logs: **{mark(sanity.get('S_dose_final'))}**",
              f"- S-tautology mutants: **{mark(sanity.get('S_tautology'))}**",
              f"- S6 floor calibration: **{mark(sanity.get('S6'))}**",
              f"- 1M state hashes: **{mark(sanity.get('S_hash_1m'))}**",
              f"- Exact-support identities: **{'PASS' if sanity.get('S_exact_all') else 'FAIL'}**",
              "", "本介入は k を使うオラクル制御であり、学習手法ではない。"]
    (outdir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return dict(main_verdict=main_verdict, count_verdict=count_verdict,
                band_verdict=band, transition_g_1m=g1, transition_g_5m=g5,
                onset=onset, binary_saturated=saturated, jump_J=jumps,
                elapsed_sec=elapsed)


def _final_sanity(cfg: dict, outdir: Path, preflight_result: dict,
                  s0: dict, identities: dict) -> dict:
    seeds = [int(v) for v in cfg["common"]["seeds"]]
    env_differences, hash_missing, dose_fail = [], [], []
    reference_env = {}
    tolerance = float(cfg["phase1"]["dose_relative_tolerance"])
    n_dose = 0
    state_hashes = {}
    for arm in ARM_ORDER:
        state_hashes[arm] = {}
        for seed in seeds:
            with np.load(outdir / "logs" / f"{arm}_seed{seed}.npz",
                         allow_pickle=False) as z:
                final = json.loads(str(z["state_hash_final"]))
                env = {key: final[key] for key in ("env.flip_state", "env.t")}
                if arm == BASELINE_ARM:
                    reference_env[seed] = env
                elif env != reference_env[seed]:
                    env_differences.append(dict(arm=arm, seed=seed))
                middle = json.loads(str(z["state_hash_1m"]))
                if not middle:
                    hash_missing.append(dict(arm=arm, seed=seed))
                state_hashes[arm][str(seed)] = middle
                if arm in FIXED_ARMS:
                    errors = np.asarray(z["dose_relative_error"], dtype=np.float64)
                    n_dose += errors.size
                    bad = np.flatnonzero(errors > tolerance)
                    if bad.size:
                        dose_fail.append(dict(arm=arm, seed=seed,
                                              max_relative_error=float(errors.max())))
    (outdir / "state_hash_1m.json").write_text(
        json.dumps(state_hashes, indent=2, sort_keys=True), encoding="utf-8")
    return dict(
        S0prime=s0, S1_omp=preflight_result["S1_omp"],
        S_pair=preflight_result["S_pair"], S_dose=preflight_result["S_dose"],
        S_tautology=preflight_result["S_tautology"],
        S_degeneracy=preflight_result["S_degeneracy"], S6=preflight_result["S6"],
        S_pair_final=dict(pass_=not env_differences, differences=env_differences),
        S_dose_final=dict(pass_=not dose_fail, tolerance=tolerance,
                          n_values=n_dose, failures=dose_fail),
        S_hash_1m=dict(pass_=not hash_missing, missing=hash_missing),
        S_exact=identities,
        S_exact_all=bool(all(value["pass_"] for value in identities.values())))


def _provenance(cfg_path: Path, cfg: dict, outdir: Path, sanity: dict,
                analysis: dict, elapsed: dict, started: float) -> dict:
    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                           text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--short"], cwd=ROOT,
                                        text=True).splitlines()
    except (OSError, subprocess.CalledProcessError):
        git_hash, dirty = None, []
    artifact_names = ("verdict.csv", "summary.md", "dose_response.csv", "ci.csv",
                      "report_only.csv", "state_hash_1m.json", "floor_calibration.csv",
                      "config_used.yaml", "s0prime.json")
    output_hashes = {name: _sha_file(outdir / name) for name in artifact_names
                     if (outdir / name).exists()}
    output_hashes.update({f"logs/{path.name}": _sha_file(path)
                          for path in sorted((outdir / "logs").glob("*.npz"))})
    spec = Path(ROOT) / cfg["spec"]
    reference = (Path(ROOT) / cfg["sanity"]["s0_prime_baseline_ref"]).resolve()
    reference_hashes = {}
    for seed in cfg["common"]["seeds"]:
        path = reference / "logs" / f"L1w100_seed{int(seed)}.npz"
        if path.exists():
            reference_hashes[f"logs/{path.name}"] = _sha_file(path)
    return dict(experiment=EXPERIMENT,
                created=time.strftime("%Y-%m-%d %H:%M:%S %z"), command=sys.argv,
                elapsed_sec=round(time.time() - started, 3), arm_elapsed_sec=elapsed,
                cwd=os.getcwd(), python=sys.version, platform=platform.platform(),
                torch=torch.__version__, numpy=np.__version__, device=cfg["common"]["device"],
                git_hash=git_hash, git_dirty=dirty, config=str(cfg_path),
                config_sha256=_sha_file(cfg_path), spec=str(spec),
                spec_sha256=_sha_file(spec), baseline_reference=str(reference),
                baseline_sha256=reference_hashes, sanity=sanity, analysis=analysis,
                output_sha256=output_hashes)


def run_full(cfg_path: Path, cfg: dict, device: str, outdir: Path, *,
             smoke: bool) -> dict:
    started = time.time()
    outdir.mkdir(parents=True, exist_ok=True)
    require_omp(cfg)
    C = cfg["common"]
    total = SMOKE_STEPS if smoke else int(C["total_steps"])
    seeds = [0] if smoke else [int(v) for v in C["seeds"]]
    if smoke:
        preflight_result = dict(S1_omp=require_omp(cfg),
                                S_pair={"pass_": True, "smoke": True},
                                S_dose={"pass_": True, "smoke": True},
                                S_tautology={"pass_": True, "smoke": True},
                                S_degeneracy=_s_degeneracy(cfg),
                                S6={"pass_": True, "smoke": True})
        s0 = {"pass_": True, "smoke": True}
    else:
        preflight_dir = Path(ROOT) / "results/_preflight_dose_const_5m_0830"
        preflight_path = preflight_dir / "preflight.json"
        if not preflight_path.exists():
            raise FileNotFoundError("run --preflight before the full run")
        preflight_result = json.loads(preflight_path.read_text(encoding="utf-8"))
        if not preflight_result.get("pass_"):
            raise RuntimeError("saved preflight did not pass")
        s0_path = outdir / "s0prime.json"
        if not s0_path.exists():
            raise FileNotFoundError("run --s0prime before the full run")
        s0 = json.loads(s0_path.read_text(encoding="utf-8"))
        if not s0.get("pass_"):
            raise RuntimeError("saved S0' did not pass")
        shutil.copy2(preflight_dir / "floor_calibration.csv",
                     outdir / "floor_calibration.csv")
    with (outdir / "config_used.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, allow_unicode=True, sort_keys=False)
    elapsed, identities = {}, {}
    for arm in ARM_ORDER:
        if _complete_arm_logs(outdir, arm, seeds, total, int(C["lop_every"])):
            elapsed[arm] = 0.0
            identities[arm] = {"pass_": True, "resumed_from_logs": True}
            print(f"[{arm}] complete logs found; resume", flush=True)
            continue
        result = _run_arm(cfg, arm, device, outdir, seeds, total)
        elapsed[arm] = result["elapsed_sec"]
        identities[arm] = result["sanity"]
    if smoke:
        payload = dict(pass_=all(v["pass_"] for v in identities.values()),
                       identities=identities, elapsed_sec=elapsed)
        (outdir / "smoke_sanity.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"SMOKE DONE -> {outdir}", flush=True)
        return payload
    sanity = _final_sanity(cfg, outdir, preflight_result, s0, identities)
    if not (sanity["S_pair_final"]["pass_"] and sanity["S_dose_final"]["pass_"]
            and sanity["S_hash_1m"]["pass_"] and sanity["S_exact_all"]):
        raise RuntimeError("final sanity failed; analysis is blocked")
    result = analyze(cfg, outdir, sanity, elapsed)
    provenance = _provenance(cfg_path, cfg, outdir, sanity, result, elapsed, started)
    (outdir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")
    print(f"ALL DONE -> {outdir}", flush=True)
    return dict(sanity=sanity, analysis=result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dose_const_5m_0830.yaml")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--s0prime", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--outdir")
    args = parser.parse_args()
    if sum((args.preflight, args.s0prime, args.smoke, args.analyze_only)) > 1:
        parser.error("stage flags are mutually exclusive")
    cfg_path = Path(args.config).resolve()
    cfg = load_config(str(cfg_path))
    device = pick_device(cfg)
    if device != "cpu":
        raise ValueError("dose_const_5m is CPU-only")
    stage = ("preflight" if args.preflight else "s0prime" if args.s0prime else
             "smoke" if args.smoke else "analyze" if args.analyze_only else "full")
    validate_config(cfg, stage=stage)
    main_dir = Path(ROOT) / cfg["output"]["dir"]
    outdir = (Path(args.outdir).resolve() if args.outdir else
              Path(ROOT) / "results/_preflight_dose_const_5m_0830"
              if args.preflight else
              Path(ROOT) / "results/_smoke_dose_const_5m_0830"
              if args.smoke else main_dir)
    if args.preflight:
        preflight(cfg_path, cfg, device, outdir)
    elif args.s0prime:
        s0prime(cfg, device, outdir)
    elif args.analyze_only:
        preflight_result = json.loads(
            (Path(ROOT) / "results/_preflight_dose_const_5m_0830/preflight.json")
            .read_text(encoding="utf-8"))
        s0 = json.loads((outdir / "s0prime.json").read_text(encoding="utf-8"))
        identities = {arm: {"pass_": True, "analyze_only": True} for arm in ARM_ORDER}
        sanity = _final_sanity(cfg, outdir, preflight_result, s0, identities)
        analyze(cfg, outdir, sanity, {})
    else:
        run_full(cfg_path, cfg, device, outdir, smoke=args.smoke)


if __name__ == "__main__":
    main()
