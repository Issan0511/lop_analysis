"""LR_A1: two-layer leaky-ReLU learner with input-layer centering only.

The experiment adds one new arm and compares it with the committed ``L2_A1``
(ReLU) and ``E_A1`` (ELU) arms.  The numerical freeze conditions are complete;
the expensive run remains blocked until execution is authorized and the repo
spec has been committed on its own.

Stages::

    OMP_NUM_THREADS=1 .venv/bin/python -m src.lr_a1_0901 --preflight
    OMP_NUM_THREADS=1 .venv/bin/python -m src.lr_a1_0901 --smoke
    OMP_NUM_THREADS=1 .venv/bin/python -m src.lr_a1_0901

``--preflight`` performs the registered 30k ReLU replay and the RNG-neutrality
checks.  ``--smoke`` is implementation-only and never produces a scientific
verdict.  The default 5M run and ``--analyze-only`` require a frozen config.
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
from .elu_swamp import (EluRecorder, exact_layer_record_elu, save_checkpoint_elu,
                        train_arm_elu)
from .gate_dose import _network_gradient_error
from .mlp2_phase0 import (_sha_array, _sha_file, identity_sanity_pass,
                          require_omp, write_csv)
from .mlp2_phase0b import _ci_components, _complete_arm_logs, _window_indices
from .mlp2_phase1 import (NUMERIC_DIVERGENCE, NumericDivergenceError,
                          StreamDigest, _env_hashes, _init_hashes,
                          _seed_state_hashes_p1, setup_arm_p1)


EXPERIMENT = "lr_a1_0901"
ARM = "LR_A1"
WIDTH = 100
LAYERS = 2
SLOPE = 0.1
PREFLIGHT_DIR = "results/_preflight_lr_a1_0901"
SMOKE_DIR = "results/_smoke_lr_a1_0901"
SMOKE_STEPS = 5_000
SMOKE_SEEDS = [0, 1]


def _arm(cfg: dict) -> dict:
    return next(a for a in cfg["arms"] if a["name"] == ARM)


def _p1_cfg(cfg: dict) -> dict:
    """Expose the LR_A1 section under the name expected by Phase-1 helpers."""
    out = copy.deepcopy(cfg)
    out["phase1"] = copy.deepcopy(cfg["lr_a1"])
    return out


def _reference(cfg: dict, label: str) -> tuple[Path, str]:
    ref = cfg["references"][label]
    return Path(ROOT) / str(ref["dir"]), str(ref["arm"])


def g_leaky(negative_slope: float = SLOPE) -> float:
    """Return E[phi(Z)] / sd(phi(Z)) for Z~N(0,1), leaky-ReLU phi.

    Positive homogeneity makes the ratio independent of the Gaussian scale.
    """
    a = float(negative_slope)
    mean = (1.0 - a) / math.sqrt(2.0 * math.pi)
    second = 0.5 * (1.0 + a * a)
    variance = second - mean * mean
    return mean / math.sqrt(variance)


def preregistration_missing(cfg: dict) -> list[str]:
    pre = cfg["preregistration"]
    missing = []
    for key in ("frozen", "execution_authorized", "repo_spec_committed",
                "predictions_confirmed", "bootstrap_seed_confirmed"):
        if pre.get(key) is not True:
            missing.append(f"preregistration.{key}")
    margin = pre.get("dose_equivalence_margin")
    if margin is None or not np.isfinite(float(margin)) or float(margin) <= 0:
        missing.append("preregistration.dose_equivalence_margin")
    return missing


def validate_config(cfg: dict, *, stage: str) -> None:
    if stage not in {"preflight", "smoke", "full", "analyze"}:
        raise ValueError(f"unknown stage {stage!r}")
    C, A, I, P, S = (cfg["common"], cfg["condA"], cfg["intervention"],
                     cfg["lr_a1"], cfg["sanity"])
    arms = cfg["arms"]
    if len(arms) != 1 or arms[0]["name"] != ARM:
        raise ValueError("LR_A1 requires exactly one new arm named LR_A1")
    if ([int(v) for v in arms[0]["hidden"]] != [WIDTH, WIDTH]
            or [int(v) for v in arms[0]["centered_layers"]] != [1]
            or str(arms[0]["activation"]) != "leaky"):
        raise ValueError("LR_A1 architecture differs from the draft spec")
    if (int(A["m"]), int(A["f"]), int(A["target_hidden"])) != (20, 15, 100):
        raise ValueError("LR_A1 requires condA m=20, f=15, teacher width=100")
    if list(A["T_values"]) != [10_000] or list(A["encodings"]) != ["std"]:
        raise ValueError("LR_A1 requires T=10000 and std input encoding")
    if (str(I["name"]) != "A_layer_input_centering"
            or float(I["center_alpha"]) != 0.01
            or I["stop_gradient_on_running_mean"] is not True
            or I["consumes_rng"] is not False):
        raise ValueError("the registered first-layer centering changed")
    activation = cfg["activation"]["leaky"]
    if (str(activation["name"]) != "leaky_relu"
            or float(activation["negative_slope"]) != SLOPE
            or activation["positive_homogeneous"] is not True
            or activation["consumes_rng"] is not False
            or cfg["activation"]["autograd"] is not False):
        raise ValueError("leaky-ReLU must be closed-form with negative slope 0.1")
    if (int(C["generator_offset"]) != 0
            or list(C["seeds"]) != list(range(10))
            or int(C["total_steps"]) != 5_000_000
            or float(C["lr_main"]) != 0.01
            or str(C["device"]) != "cpu"):
        raise ValueError("LR_A1 must match the committed reference seed/grid/lr/device")
    if (int(P["task_period"]) != 10_000 or list(P["late_tasks"]) != [451, 500]
            or int(P["exact_support"]) != 32
            or float(P["unfit_floor"]) != 1e-23
            or P["recalibrate_floor"] is not False):
        raise ValueError("window, support, or inherited two-layer floor changed")
    if (int(P["bootstrap_B"]) != 10_000
            or int(P["bootstrap_seed"]) != 202_609_011_252
            or str(P["ci_method"]) != "percentile_primary_studentized_fallback"):
        raise ValueError("the registered bootstrap design changed")
    if int(P["bootstrap_seed"]) in {
            20_260_829, 20_260_901, 20_260_902, 20_260_903, 20_260_904,
            20_260_906}:
        raise ValueError("bootstrap seed collides with an existing registered use")
    if (P["report_count_and_fraction"] is not True
            or P["strict_dead_applicable_to_leaky"] is not False
            or P["strict_dead_in_verdict"] is not False
            or P["submerged_frac_in_verdict"] is not False
            or list(P["p3_report_only"]) != [
                "submerged_count", "submerged_frac", "mobility", "s_i"]
            or str(P["boundary_snapshot"]) != "task_end"
            or P["layer_stats_task_end_only"] is not True):
        raise ValueError("strict-dead/submergence semantics differ from spec section 3")
    output = cfg["output"]
    if (output["layer_stats_csv"] is not True
            or output["s_distribution_csv"] is not True
            or output["logs_npz"] is not True):
        raise ValueError("registered logger outputs must remain enabled")
    if (int(S["s0_prime_steps"]) != 30_000
            or int(S["s0_prime_every"]) != 1_000
            or list(S["s0_prime_metrics"]) != [
                "layer1_p_hat", "layer2_p_hat", "eval_loss_exact", "unfit"]
            or int(S["omp_num_threads"]) != 1):
        raise ValueError("S0' replay contract changed")
    refs = cfg["references"]
    if (refs["relu"]["arm"] != "L2_A1" or refs["elu"]["arm"] != "E_A1"):
        raise ValueError("reference arms must remain L2_A1 and E_A1")
    pre = cfg["preregistration"]
    if (pre["conditions_complete"] is not True
            or pre["prediction_provenance"]
            != "draft_values_proposed_first_then_approved_by_Issa"):
        raise ValueError("the Obsidian preregistration provenance changed")
    if (float(pre["predicted_layer2_dose_late_t451_500"]) != 5.9
            or float(pre["predicted_layer2_submerged_frac_step5m"]) != 0.60
            or float(pre["predicted_unfit_max_late_t451_500"]) != 0.005
            or str(pre["predicted_failure_cause"]) != "全然わからない"
            or float(pre["dose_equivalence_margin"]) != 0.5):
        raise ValueError("the registered predictions or dose margin changed")
    if (float(P["p1_closed_form_center"]) != 5.85
            or [float(v) for v in P["p1_closed_form_band"]] != [5.35, 6.35]
            or float(P["p1_lambda_corrected_center_report_only"]) != 6.15
            or [float(v) for v in
                P["p1_lambda_corrected_band_report_only"]] != [5.65, 6.65]
            or P["p1_relu_raw_delta_in_verdict"] is not False):
        raise ValueError("P1 closed-form dose decision rule changed")
    if (P["p2_rule"]
            != "ci_upper_negative_then_point_at_or_below_registered_E_A1_upper"
            or [float(v) for v in P["p2_elu_registered_delta_interval"]]
            != [-1.098, -0.696]):
        raise ValueError("P2 ordered decision rule changed")
    if stage in {"full", "analyze"}:
        missing = preregistration_missing(cfg)
        if missing:
            raise ValueError("LR_A1 preregistration is not frozen: " + ", ".join(missing))


def setup_arm_lr(cfg: dict, device: str, *, activation: str = "leaky_relu") -> dict:
    """Construct LR_A1 through the frozen Phase-1 initialization path."""
    if int(cfg["common"].get("generator_offset", 0)) != 0:
        # setup_arm_p1's frozen predecessor is the offset-0 reference harness.
        raise ValueError("the paired LR_A1 harness requires generator_offset=0")
    st = setup_arm_p1(_p1_cfg(cfg), _arm(cfg), device)
    alpha = SLOPE if activation == "leaky_relu" else 1.0
    st["net"].set_activation(activation, alpha, "alpha_exp")
    st["activation"] = activation
    st["act_alpha"] = float(alpha)
    st["generator_offset"] = 0
    return st


def _generator_hashes(st: dict) -> dict[str, str]:
    return {name: _sha_array(gen.get_state()) for name, gen in st["gens"].items()}


def _state_snapshot(st: dict) -> dict[str, str]:
    out = _init_hashes(st)
    out.update({f"generator.{k}": v for k, v in _generator_hashes(st).items()})
    out["running_mean"] = _sha_array(st["running_mean"])
    for li, mean in enumerate(st["layer_means"], start=1):
        if mean is not None:
            out[f"running_mean.layer{li}"] = _sha_array(mean)
    return out


def _reference_initial_check(cfg: dict, st: dict) -> dict:
    refdir, refarm = _reference(cfg, "relu")
    path = refdir / "ckpts" / f"{refarm}_step0.pt"
    if not path.exists():
        return dict(pass_=False, path=str(path), missing=True, differences=[])
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    differences = []
    for group, actual in (("net", st["net"].state_dict()),
                          ("teacher", st["teacher"].state_dict())):
        expected = checkpoint[group]
        for key, value in actual.items():
            if key not in expected or _sha_array(value) != _sha_array(expected[key]):
                differences.append(f"{group}.{key}")
    if (_sha_array(st["env"].flip_state)
            != _sha_array(checkpoint["env"]["flip_state"])):
        differences.append("env.flip_state")
    if int(st["env"].t) != int(checkpoint["env"]["t"]):
        differences.append("env.t")
    if _sha_array(st["running_mean"]) != _sha_array(checkpoint["running_mean"]):
        differences.append("running_mean")
    return dict(pass_=not differences, path=str(path), missing=False,
                differences=differences, sha256=_sha_file(path))


def _compare_s0_reference(cfg: dict, rec: EluRecorder) -> dict:
    refdir, refarm = _reference(cfg, "relu")
    differences, missing = [], []
    for ri, seed in enumerate(cfg["common"]["seeds"]):
        path = refdir / "logs" / f"{refarm}_seed{seed}.npz"
        if not path.exists():
            missing.append(str(path))
            continue
        with np.load(path, allow_pickle=False) as z:
            idx = np.flatnonzero(np.isin(z["step"], rec.steps))
            if not np.array_equal(z["step"][idx], rec.steps):
                differences.append(dict(seed=seed, metric="step", reason="grid mismatch"))
                continue
            pairs = (("eval_loss_exact", rec.run["eval_loss_exact"][:, ri]),
                     ("unfit", rec.run["unfit"][:, ri]),
                     ("layer1_p_hat", rec.layers[0]["p_hat"][:, ri]),
                     ("layer2_p_hat", rec.layers[1]["p_hat"][:, ri]))
            for key, ours in pairs:
                theirs = z[key][idx]
                if _sha_array(ours) != _sha_array(theirs):
                    differences.append(dict(seed=int(seed), metric=key,
                                            reason="bit hash mismatch"))
    return dict(pass_=not missing and not differences, missing=missing,
                differences=differences, compared_seeds=list(cfg["common"]["seeds"]),
                steps=rec.steps.tolist(), metrics=list(cfg["sanity"]["s0_prime_metrics"]))


def _s0_and_pair(cfg: dict, device: str, outdir: Path) -> dict:
    steps = int(cfg["sanity"]["s0_prime_steps"])
    every = int(cfg["sanity"]["s0_prime_every"])
    probes = list(range(0, steps + 1, every))
    relu = setup_arm_lr(cfg, device, activation="relu")
    leaky = setup_arm_lr(cfg, device, activation="leaky_relu")
    init_relu, init_leaky = _init_hashes(relu), _init_hashes(leaky)
    checkpoint = _reference_initial_check(cfg, relu)
    rec_relu = EluRecorder(probes, relu, float(cfg["lr_a1"]["sigma_degenerate_tol"]),
                           float(cfg["sanity"]["s1_identity_tol"]), every,
                           zbar_layers=[], readout_steps=[])
    rec_leaky = EluRecorder(probes, leaky, float(cfg["lr_a1"]["sigma_degenerate_tol"]),
                            float(cfg["sanity"]["s1_identity_tol"]), every,
                            zbar_layers=[], readout_steps=[])
    digest_relu, digest_leaky = StreamDigest(), StreamDigest()
    train_arm_elu(relu, rec_relu, probes, steps, outdir, [], digest_relu)
    train_arm_elu(leaky, rec_leaky, probes, steps, outdir, [], digest_leaky)
    s0 = _compare_s0_reference(cfg, rec_relu)
    init_differences = sorted(k for k, value in init_relu.items()
                              if init_leaky.get(k) != value)
    final_env_equal = _env_hashes(relu) == _env_hashes(leaky)
    stream_equal = digest_relu.digest() == digest_leaky.digest()
    grid_ok = rec_relu.steps.tolist() == probes == rec_leaky.steps.tolist()
    pair = dict(pass_=bool(not init_differences and checkpoint["pass_"]
                           and final_env_equal and stream_equal and grid_ok),
                seeds=list(cfg["common"]["seeds"]), generator_offset=0,
                init_differences=init_differences,
                reference_initial_checkpoint=checkpoint,
                stream_equal=stream_equal, final_env_equal=final_env_equal,
                grid_equal=grid_ok, trajectory_caveat="diverges_after_step1")
    sanity = dict(relu=rec_relu.sanity(), leaky=rec_leaky.sanity())
    s0["pass_"] = bool(s0["pass_"] and sanity["relu"]["pass_"])
    pair["pass_"] = bool(pair["pass_"] and sanity["leaky"]["pass_"])
    return dict(S0prime=s0, S1=pair, recorder_sanity=sanity)


def _s3_rng_and_finite(cfg: dict, device: str) -> dict:
    local = copy.deepcopy(cfg)
    local["common"]["seeds"] = [0, 1]
    st = setup_arm_lr(local, device)
    before = _state_snapshot(st)
    record, sanity = exact_layer_record_elu(
        st, float(cfg["lr_a1"]["sigma_degenerate_tol"]))
    after = _state_snapshot(st)
    changed = sorted(k for k, value in before.items() if after.get(k) != value)
    primary = {
        "layer2_dose": record["layers"][1]["dose"],
        "unfit": record["run"]["unfit"],
        "eval_loss_exact": record["run"]["eval_loss_exact"],
    }
    finite = {key: bool(torch.isfinite(value).all()) for key, value in primary.items()}
    return dict(pass_=bool(not changed and all(finite.values())
                           and identity_sanity_pass(
                               sanity, float(cfg["sanity"]["s1_identity_tol"]))),
                exact_support=int(sanity["support"]), state_or_rng_changes=changed,
                finite_primary=finite, identity=sanity)


def _s_grad_and_ratio(cfg: dict) -> dict:
    from .nets import VecMLPL

    tol = float(cfg["sanity"]["s_grad_finite_difference_tol"])
    points = torch.tensor([-30.0, -1.0, -1e-6, 0.0, 1e-6, 1.0, 30.0],
                          dtype=torch.float64)
    net = VecMLPL(1, [2], 2, torch.Generator().manual_seed(0), "cpu")
    net.set_activation("leaky_relu", SLOPE, "alpha_exp")
    activation = net.act_fn(points)
    derivative = net.act_grad(points, activation)
    expected = torch.where(points > 0, torch.ones_like(points),
                           torch.full_like(points, SLOPE))
    static_equal = bool(torch.equal(derivative, expected))
    network_error = float(_network_gradient_error("leaky_relu", SLOPE))
    ratio = g_leaky(SLOPE)
    return dict(pass_=bool(static_equal and network_error <= tol
                           and abs(ratio - 0.585) <= 5e-4),
                slope=SLOPE, static_derivative_equal=static_equal,
                network_finite_difference_max_relerr=network_error,
                tolerance=tol, g_closed_form=ratio,
                layer2_width100_dose_reference=10.0 * ratio,
                spec_rounded_g=0.585)


def _s_floor_inheritance(cfg: dict) -> dict:
    expected = float(cfg["lr_a1"]["unfit_floor"])
    rows = []
    for label in ("relu", "elu"):
        refdir, _ = _reference(cfg, label)
        path = refdir / "config_used.yaml"
        if not path.exists():
            rows.append(dict(reference=label, path=str(path), exists=False, floor=None,
                             matches=False))
            continue
        ref = load_config(str(path))
        section = ref.get("phase1", ref.get("elu_swamp", {}))
        floor = float(section["unfit_floor"])
        rows.append(dict(reference=label, path=str(path), exists=True, floor=floor,
                         matches=floor == expected))
    return dict(pass_=bool(cfg["lr_a1"]["recalibrate_floor"] is False
                           and all(row["matches"] for row in rows)),
                inherited_floor=expected, recalibrated=False, references=rows)


def preflight(cfg: dict, device: str, outdir: Path) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    checks: dict[str, dict] = {"S_OMP": require_omp(cfg)}
    print("[S0'/S1] 30k ReLU replay and paired stream check", flush=True)
    checks.update(_s0_and_pair(cfg, device, outdir / "s0"))
    print("[S3] exact-support RNG neutrality and finite endpoints", flush=True)
    checks["S3"] = _s3_rng_and_finite(cfg, device)
    print("[S-grad/g] leaky gradient and positive-homogeneous ratio", flush=True)
    checks["S_grad_ratio"] = _s_grad_and_ratio(cfg)
    checks["S2_semantics"] = dict(
        pass_=bool(cfg["lr_a1"]["strict_dead_in_verdict"] is False
                   and cfg["lr_a1"]["submerged_frac_in_verdict"] is False
                   and cfg["lr_a1"]["strict_dead_applicable_to_leaky"] is False),
        strict_dead_in_verdict=False, submerged_frac_in_verdict=False)
    checks["S_floor"] = _s_floor_inheritance(cfg)
    required = ("S_OMP", "S0prime", "S1", "S3", "S_grad_ratio",
                "S2_semantics", "S_floor")
    result = dict(pass_=bool(all(checks[key].get("pass_") for key in required)),
                  required=list(required), preregistration_missing=preregistration_missing(cfg),
                  **checks)
    (outdir / "preflight.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    if not result["pass_"]:
        failed = [key for key in required if not result[key].get("pass_")]
        raise RuntimeError(f"LR_A1 preflight failed: {failed}")
    print(f"PREFLIGHT PASS -> {outdir}", flush=True)
    return result


def _arm_status_path(outdir: Path) -> Path:
    return outdir / "arm_status" / f"{ARM}.json"


def _load_divergence(outdir: Path, seeds: list[int], total: int,
                     every: int) -> dict | None:
    path = _arm_status_path(outdir)
    if not path.exists():
        return None
    try:
        event = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    valid = (event.get("status") == NUMERIC_DIVERGENCE
             and event.get("arm") == ARM
             and event.get("registered_seeds") == seeds
             and int(event.get("registered_total_steps", -1)) == total
             and int(event.get("probe_every", -1)) == every
             and event.get("rescue") == "none")
    return event if valid else None


def write_arm_logs_lr(outdir: Path, st: dict, rec: EluRecorder) -> list[Path]:
    logdir = outdir / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ri, run in enumerate(st["runs"]):
        payload = dict(
            step=rec.steps, run_id=np.array(run["run_id"]), arm=np.array(ARM),
            seed=np.int64(run["seed"]), activation=np.array("leaky_relu"),
            negative_slope=np.float64(SLOPE), generator_offset=np.int64(0),
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


def _run_arm(cfg: dict, device: str, outdir: Path, *, total: int,
             seeds: list[int]) -> dict:
    local = copy.deepcopy(cfg)
    local["common"]["total_steps"] = int(total)
    local["common"]["seeds"] = [int(v) for v in seeds]
    every = int(local["common"]["lop_every"])
    probes = list(range(0, total + 1, every))
    if probes[-1] != total:
        probes.append(total)
    st = setup_arm_lr(local, device)
    _, initial = exact_layer_record_elu(st, float(local["lr_a1"]["sigma_degenerate_tol"]))
    if not identity_sanity_pass(initial, float(local["sanity"]["s1_identity_tol"])):
        raise RuntimeError("LR_A1 initial exact-support identity failed")
    rec = EluRecorder(probes, st, float(local["lr_a1"]["sigma_degenerate_tol"]),
                      float(local["sanity"]["s1_identity_tol"]), every,
                      zbar_layers=[], readout_steps=[])
    checkpoints = [int(v) for v in local["common"].get("checkpoints", [])
                   if int(v) <= total]
    started = time.time()
    try:
        elapsed = train_arm_elu(st, rec, probes, total, outdir, checkpoints)
    except NumericDivergenceError as exc:
        elapsed = time.time() - started
        event = dict(exc.event)
        event.update(probe_every=every, registered_total_steps=total,
                     registered_seeds=seeds, activation="leaky_relu",
                     negative_slope=SLOPE, elapsed_sec=elapsed,
                     detection="nonfinite_training_state_at_probe",
                     action="stop_arm_no_rescue", rescue="none")
        path = _arm_status_path(outdir)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(event, indent=2, ensure_ascii=False), encoding="utf-8")
        return dict(status=NUMERIC_DIVERGENCE, elapsed_sec=elapsed,
                    divergence=event, sanity=dict(pass_=False))
    sanity = rec.sanity()
    if not sanity["pass_"]:
        raise RuntimeError(f"LR_A1 recorder sanity failed: {sanity}")
    write_arm_logs_lr(outdir, st, rec)
    return dict(status="COMPLETE", elapsed_sec=elapsed, sanity=sanity,
                final_env=_env_hashes(st))


def _ci(cfg: dict, values: np.ndarray, draws: np.ndarray) -> dict:
    P = cfg["lr_a1"]
    return _ci_components(np.asarray(values, dtype=np.float64), draws, "median",
                          float(P["degenerate_se_tol"]),
                          float(P["degenerate_frac_max"]),
                          float(P["degenerate_width_ratio_max"]))


def classify_p1(ci_lo: float, ci_hi: float, band_lo: float = 5.35,
                band_hi: float = 6.35) -> str:
    """Classify the registered absolute layer-2 dose CI against its band."""
    if ci_lo > ci_hi or band_lo > band_hi:
        raise ValueError("P1 CI and band endpoints must be ordered")
    if ci_lo >= band_lo and ci_hi <= band_hi:
        return "A_CLOSED_FORM_MATCH"
    if ci_hi < band_lo or ci_lo > band_hi:
        return "A_DOSE_OFF_PREDICTION"
    return "INCONCLUSIVE_WIDE"


def classify_p2(ci_hi: float, point: float, interval_lo: float = -1.098,
                interval_hi: float = -0.696) -> str:
    """Apply the registered sign-first, one-sided effect-size rule for P2."""
    if interval_lo > interval_hi:
        raise ValueError("P2 reference interval endpoints must be ordered")
    if ci_hi >= 0.0:
        return "A_WITHOUT_B_NOT_CONFIRMED"
    if point <= interval_hi:
        return "A_WITHOUT_B_HARMLESS_MULTILAYER"
    return "PARTIAL_IMPROVEMENT"


def p2_effect_relation(point: float, interval_lo: float = -1.098,
                       interval_hi: float = -0.696) -> str:
    if interval_lo > interval_hi:
        raise ValueError("P2 reference interval endpoints must be ordered")
    if point < interval_lo:
        return "STRONGER_THAN_E_A1_INTERVAL"
    if point <= interval_hi:
        return "WITHIN_E_A1_INTERVAL"
    return "WEAKER_THAN_E_A1_INTERVAL"


def _endpoint_row(metric: str, window: str, arm: str, baseline: str,
                  values: np.ndarray, ci: dict, *, report_only: bool,
                  decision: str = "", passed: bool | None = None,
                  margin: float | str = "", main_verdict: str = "") -> dict:
    return dict(metric=metric, window=window, arm=arm, baseline=baseline,
                pairing="paired" if baseline else "single_arm",
                report_only=int(report_only), seed_values=json.dumps(values.tolist()),
                point=ci["point"], percentile_ci_lo=ci["percentile_ci_lo"],
                percentile_ci_hi=ci["percentile_ci_hi"],
                studentized_ci_lo=ci["studentized_ci_lo"],
                studentized_ci_hi=ci["studentized_ci_hi"],
                CI_DEGENERATE=ci["ci_degenerate"],
                degenerate_se_fraction=ci["degenerate_se_fraction"],
                sign_test_p=ci["sign_test_p"], n_seed=ci["n_seed"],
                decision_basis="percentile_primary" if decision else "report_only",
                decision=decision, passed="" if passed is None else int(passed),
                equivalence_margin=margin, main_verdict=main_verdict)


def _normalize_rows(rows: list[dict]) -> list[dict]:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    return [{key: row.get(key, "") for key in fields} for row in rows]


def _parse_state_hash(data: np.lib.npyio.NpzFile) -> dict:
    return json.loads(str(data["state_hash_final"]))


def analyze(cfg: dict, outdir: Path, sanity: dict, elapsed: dict) -> dict:
    seeds = [int(v) for v in cfg["common"]["seeds"]]
    total = int(cfg["common"]["total_steps"])
    period = int(cfg["lr_a1"]["task_period"])
    late_tasks = list(cfg["lr_a1"]["late_tasks"])
    floor = float(cfg["lr_a1"]["unfit_floor"])
    ref_relu_dir, ref_relu_arm = _reference(cfg, "relu")
    ref_elu_dir, ref_elu_arm = _reference(cfg, "elu")
    slope = float(cfg["activation"]["leaky"]["negative_slope"])

    lr_dose, relu_dose = [], []
    lr_u, relu_u, elu_u = [], [], []
    submerged_count, submerged_frac, mobility = [], [], []
    boundary_rows, s_rows, pairing_failures, mask_rows = [], [], [], []
    for seed in seeds:
        lr_path = outdir / "logs" / f"{ARM}_seed{seed}.npz"
        relu_path = ref_relu_dir / "logs" / f"{ref_relu_arm}_seed{seed}.npz"
        elu_path = ref_elu_dir / "logs" / f"{ref_elu_arm}_seed{seed}.npz"
        if not (lr_path.exists() and relu_path.exists() and elu_path.exists()):
            raise FileNotFoundError(f"missing paired log for seed {seed}")
        with (np.load(lr_path, allow_pickle=False) as lr,
              np.load(relu_path, allow_pickle=False) as relu,
              np.load(elu_path, allow_pickle=False) as elu):
            # NpzFile.__getitem__ decompresses on every access.  Materialize
            # each required column once before the 500-task boundary loop.
            lr_keys = {"step", "unfit", "eval_loss_exact", "flip_state",
                       "layer2_dose", "layer2_submerged", "layer2_p_hat"}
            for layer in (1, 2):
                lr_keys.update({f"layer{layer}_{key}" for key in (
                    "p_hat", "M", "B", "denom", "dose", "mu_norm",
                    "sigma_rms", "submerged", "eff_rank", "w_norm_median",
                    "preact_sd_median", "median_M", "median_B")})
            lra = {key: lr[key] for key in lr_keys}
            relua = {key: relu[key] for key in
                     ("step", "layer2_dose", "unfit", "flip_state")}
            elua = {key: elu[key] for key in ("step", "unfit", "flip_state")}
            lr_state, relu_state, elu_state = (_parse_state_hash(lr),
                                                _parse_state_hash(relu),
                                                _parse_state_hash(elu))
            steps = lra["step"]
            if not (np.array_equal(steps, relua["step"])
                    and np.array_equal(steps, elua["step"])):
                pairing_failures.append(dict(seed=seed, reason="step_grid"))
            idx = _window_indices(steps, period, late_tasks)
            expected_late = late_tasks[1] - late_tasks[0] + 1
            mask_rows.append(dict(seed=seed, window="late_t451_500",
                                  selected=int(len(idx)), expected=expected_late,
                                  pass_=int(len(idx) == expected_late)))
            if len(idx) != expected_late:
                raise RuntimeError(f"seed {seed}: late window has {len(idx)} rows")
            final = np.flatnonzero(steps == total)
            if len(final) != 1:
                raise RuntimeError(f"seed {seed}: expected one step-5M row")
            fi = int(final[0])
            lr_dose.append(float(np.mean(lra["layer2_dose"][idx])))
            relu_dose.append(float(np.mean(relua["layer2_dose"][idx])))
            lr_u.append(float(np.mean(lra["unfit"][idx])))
            relu_u.append(float(np.mean(relua["unfit"][idx])))
            elu_u.append(float(np.mean(elua["unfit"][idx])))
            count = int(lra["layer2_submerged"][fi])
            frac = count / WIDTH
            submerged_count.append(count)
            submerged_frac.append(frac)
            mobility.append(float(np.mean(slope + (1.0 - slope)
                                                  * lra["layer2_p_hat"][fi])))

            if not (np.array_equal(lra["flip_state"], relua["flip_state"])
                    and np.array_equal(lra["flip_state"], elua["flip_state"])):
                pairing_failures.append(dict(seed=seed, reason="input_realization"))
            for key in ("env.flip_state", "env.t", "running_mean"):
                if not (lr_state.get(key) == relu_state.get(key) == elu_state.get(key)):
                    pairing_failures.append(dict(seed=seed, reason=f"final.{key}"))

            boundary = np.flatnonzero((steps > 0) & (steps % period == 0))
            mask_rows.append(dict(seed=seed, window="task_end_t1_500",
                                  selected=int(len(boundary)), expected=500,
                                  pass_=int(len(boundary) == 500)))
            mask_rows.append(dict(seed=seed, window="step0_excluded_from_primary",
                                  selected=int(np.sum(steps[idx] == 0)), expected=0,
                                  pass_=int(not np.any(steps[idx] == 0))))
            for i in boundary:
                task = int(steps[i] // period)
                for layer in (1, 2):
                    p_hat = lra[f"layer{layer}_p_hat"][i]
                    M = lra[f"layer{layer}_M"][i].astype(np.float64)
                    B = lra[f"layer{layer}_B"][i].astype(np.float64)
                    s = M + B
                    finite_s = np.isfinite(s)
                    boundary_rows.append(dict(
                        arm=ARM, seed=seed, step=int(steps[i]), task=task,
                        window="task_end_t1_500", layer=layer, centered=int(layer == 1),
                        unfit=float(lra["unfit"][i]),
                        eval_loss_exact=float(lra["eval_loss_exact"][i]),
                        dose=float(lra[f"layer{layer}_dose"][i]),
                        mu_norm=float(lra[f"layer{layer}_mu_norm"][i]),
                        sigma_rms=float(lra[f"layer{layer}_sigma_rms"][i]),
                        submerged_count=int(lra[f"layer{layer}_submerged"][i]),
                        submerged_frac=float(lra[f"layer{layer}_submerged"][i] / WIDTH),
                        mobility_mean=float(np.mean(slope + (1.0 - slope) * p_hat)),
                        eff_rank=float(lra[f"layer{layer}_eff_rank"][i]),
                        w_norm_median=float(lra[f"layer{layer}_w_norm_median"][i]),
                        preact_sd_median=float(
                            lra[f"layer{layer}_preact_sd_median"][i]),
                        median_M=float(lra[f"layer{layer}_median_M"][i]),
                        median_B=float(lra[f"layer{layer}_median_B"][i]),
                        median_s=(float(np.median(s[finite_s])) if finite_s.any()
                                  else float("nan")),
                        finite_s_count=int(finite_s.sum()), total_units=WIDTH))
            for layer in (1, 2):
                M = lra[f"layer{layer}_M"][fi].astype(np.float64)
                B = lra[f"layer{layer}_B"][fi].astype(np.float64)
                denom = lra[f"layer{layer}_denom"][fi].astype(np.float64)
                p_hat = lra[f"layer{layer}_p_hat"][fi].astype(np.float64)
                for unit in range(WIDTH):
                    s_rows.append(dict(
                        arm=ARM, seed=seed, step=total, task=500,
                        window="step5m_t500", layer=layer, unit=unit,
                        M=float(M[unit]), B=float(B[unit]), s_i=float(M[unit] + B[unit]),
                        denom=float(denom[unit]), p_hat=float(p_hat[unit]),
                        submerged=int(p_hat[unit] == 0.0),
                        mobility=float(slope + (1.0 - slope) * p_hat[unit]),
                        finite_s=int(np.isfinite(M[unit] + B[unit]))))

    lr_dose = np.asarray(lr_dose)
    relu_dose = np.asarray(relu_dose)
    lr_u, relu_u, elu_u = map(np.asarray, (lr_u, relu_u, elu_u))
    submerged_count = np.asarray(submerged_count, dtype=np.float64)
    submerged_frac = np.asarray(submerged_frac, dtype=np.float64)
    mobility = np.asarray(mobility, dtype=np.float64)
    lr_log = np.log10(np.maximum(lr_u, floor))
    relu_log = np.log10(np.maximum(relu_u, floor))
    elu_log = np.log10(np.maximum(elu_u, floor))
    draws = np.random.default_rng(int(cfg["lr_a1"]["bootstrap_seed"])).integers(
        0, len(seeds), size=(int(cfg["lr_a1"]["bootstrap_B"]), len(seeds)))

    cis = dict(
        lr_dose=_ci(cfg, lr_dose, draws),
        relu_dose=_ci(cfg, relu_dose, draws),
        dose_delta=_ci(cfg, lr_dose - relu_dose, draws),
        lr_log_unfit=_ci(cfg, lr_log, draws),
        lr_delta_unfit=_ci(cfg, lr_log - relu_log, draws),
        elu_delta_unfit=_ci(cfg, elu_log - relu_log, draws),
        submerged_count=_ci(cfg, submerged_count, draws),
        submerged_frac=_ci(cfg, submerged_frac, draws),
        mobility=_ci(cfg, mobility, draws),
    )
    band_lo, band_hi = [float(v) for v in cfg["lr_a1"]["p1_closed_form_band"]]
    p1_verdict = classify_p1(cis["lr_dose"]["percentile_ci_lo"],
                             cis["lr_dose"]["percentile_ci_hi"],
                             band_lo, band_hi)
    corrected_lo, corrected_hi = [float(v) for v in
                                  cfg["lr_a1"][
                                      "p1_lambda_corrected_band_report_only"]]
    p1_lambda_report = classify_p1(cis["lr_dose"]["percentile_ci_lo"],
                                   cis["lr_dose"]["percentile_ci_hi"],
                                   corrected_lo, corrected_hi)
    p2_negative = bool(cis["lr_delta_unfit"]["percentile_ci_hi"] < 0.0)
    e_lo, e_hi = [float(v) for v in
                  cfg["lr_a1"]["p2_elu_registered_delta_interval"]]
    p2_verdict = classify_p2(cis["lr_delta_unfit"]["percentile_ci_hi"],
                             cis["lr_delta_unfit"]["point"], e_lo, e_hi)
    p2_relation = p2_effect_relation(cis["lr_delta_unfit"]["point"],
                                     e_lo, e_hi)
    unfit_bound = float(cfg["preregistration"]["predicted_unfit_max_late_t451_500"])
    unfit_prediction_met = bool(float(np.median(lr_u)) <= unfit_bound)
    rows = [
        _endpoint_row("P1_layer2_dose_level", "late_t451_500", ARM, "", lr_dose,
                      cis["lr_dose"], report_only=False,
                      decision="percentile_CI_vs_closed_form_band_5.35_6.35",
                      passed=p1_verdict == "A_CLOSED_FORM_MATCH",
                      margin="[5.35, 6.35]", main_verdict=p2_verdict),
        _endpoint_row("P1_delta_layer2_dose_vs_L2_A1", "late_t451_500", ARM,
                      ref_relu_arm, lr_dose - relu_dose, cis["dose_delta"],
                      report_only=True, main_verdict=p2_verdict),
        _endpoint_row("P2_log10_mean_unfit_level", "late_t451_500", ARM, "", lr_log,
                      cis["lr_log_unfit"], report_only=True,
                      decision=f"prediction_check_median_raw_unfit_le_{unfit_bound:g}",
                      passed=unfit_prediction_met, main_verdict=p2_verdict),
        _endpoint_row("P2_delta_log10_mean_unfit", "late_t451_500", ARM,
                      ref_relu_arm, lr_log - relu_log, cis["lr_delta_unfit"],
                      report_only=False,
                      decision="CI_upper_lt_0_then_point_le_E_A1_interval_upper",
                      passed=p2_negative, main_verdict=p2_verdict),
        _endpoint_row("P2_E_A1_reference_delta_log10_mean_unfit", "late_t451_500",
                      ref_elu_arm, ref_relu_arm, elu_log - relu_log,
                      cis["elu_delta_unfit"], report_only=True,
                      main_verdict=p2_verdict),
        _endpoint_row("P3_layer2_submerged_count", "step5m_t500", ARM, "",
                      submerged_count, cis["submerged_count"], report_only=True,
                      main_verdict=p2_verdict),
        _endpoint_row("P3_layer2_submerged_frac", "step5m_t500", ARM, "",
                      submerged_frac, cis["submerged_frac"], report_only=True,
                      main_verdict=p2_verdict),
        _endpoint_row("P3_layer2_mean_mobility", "step5m_t500", ARM, "", mobility,
                      cis["mobility"], report_only=True, main_verdict=p2_verdict),
    ]
    for row in rows:
        row["P1_verdict"] = p1_verdict
        row["P2_verdict"] = p2_verdict
    rows[0].update(
        closed_form_center=float(cfg["lr_a1"]["p1_closed_form_center"]),
        closed_form_band=json.dumps([band_lo, band_hi]),
        lambda_corrected_center_report_only=float(
            cfg["lr_a1"]["p1_lambda_corrected_center_report_only"]),
        lambda_corrected_band_report_only=json.dumps(
            [corrected_lo, corrected_hi]),
        lambda_corrected_reading_report_only=p1_lambda_report,
    )
    rows[1]["raw_delta_used_in_verdict"] = 0
    rows[2]["decision_basis"] = "prediction_correspondence_report_only"
    rows[3]["p2_effect_relation_to_registered_E_A1_interval"] = p2_relation
    rows[3]["p2_registered_E_A1_interval"] = json.dumps([e_lo, e_hi])
    rows[0]["preregistered_prediction"] = float(
        cfg["preregistration"]["predicted_layer2_dose_late_t451_500"])
    rows[0]["preregistered_prediction_scale"] = "dose"
    rows[2]["preregistered_prediction"] = float(
        cfg["preregistration"]["predicted_unfit_max_late_t451_500"])
    rows[2]["preregistered_prediction_scale"] = "raw_mean_unfit_upper_bound"
    rows[6]["preregistered_prediction"] = float(
        cfg["preregistration"]["predicted_layer2_submerged_frac_step5m"])
    rows[6]["preregistered_prediction_scale"] = "fraction"
    write_csv(outdir / "verdict.csv", _normalize_rows(rows))
    write_csv(outdir / "layer_stats.csv", boundary_rows)
    write_csv(outdir / "s_distribution.csv", s_rows)
    write_csv(outdir / "mask_counts.csv", mask_rows)
    pair_final = dict(pass_=not pairing_failures, failures=pairing_failures,
                      scope="init_teacher_input_realization",
                      trajectory_caveat="activations_diverge_after_step1")
    if not pair_final["pass_"]:
        raise RuntimeError(f"final pairing check failed: {pairing_failures[:5]}")
    result = dict(
        main_verdict=p2_verdict, p1_verdict=p1_verdict,
        p1_lambda_corrected_report_only=p1_lambda_report,
        p2_verdict=p2_verdict, p2_negative=p2_negative,
        p2_effect_relation=p2_relation,
        unfit_prediction_met=unfit_prediction_met,
        median_unfit_late_t451_500=float(np.median(lr_u)),
        median_layer2_dose_late_t451_500=float(np.median(lr_dose)),
        median_layer2_submerged_frac_step5m=float(np.median(submerged_frac)),
        endpoints=cis, S_pair_final=pair_final, mask_counts=mask_rows,
        elapsed_sec=elapsed)
    _write_summary(cfg, outdir, result, sanity)
    return result


def _write_summary(cfg: dict, outdir: Path, result: dict, sanity: dict) -> None:
    ci_lr_dose = result["endpoints"]["lr_dose"]
    ci_relu_dose = result["endpoints"]["relu_dose"]
    ci_dose = result["endpoints"]["dose_delta"]
    ci_lr = result["endpoints"]["lr_delta_unfit"]
    ci_elu = result["endpoints"]["elu_delta_unfit"]
    lines = [
        f"# {EXPERIMENT} summary", "", "## Verdict", "",
        f"- P1: **{result['p1_verdict']}**",
        f"- P2: **{result['p2_verdict']}**",
        "- Scope: condA, width 100, two hidden layers, 5M steps, leaky-ReLU a=0.1.",
        "- Pairing covers initialization, teacher, and input realization only; trajectories diverge after step 1.",
        "- `strict_dead` and `submerged_frac` were not used in the verdict.",
        "", "## Primary endpoints", "",
        f"- P1 layer-2 dose (late_t451_500, seed median): {ci_lr_dose['point']:.6g} "
        f"[{ci_lr_dose['percentile_ci_lo']:.6g}, {ci_lr_dose['percentile_ci_hi']:.6g}].",
        "- P1 registered decision band: [5.35, 6.35], centered on the closed-form prediction 5.85.",
        f"- P1 lambda-corrected reading (report only): "
        f"**{result['p1_lambda_corrected_report_only']}** against [5.65, 6.65], centered on 6.15.",
        f"- L2_A1 layer-2 dose (late_t451_500, seed median): {ci_relu_dose['point']:.6g} "
        f"[{ci_relu_dose['percentile_ci_lo']:.6g}, {ci_relu_dose['percentile_ci_hi']:.6g}].",
        f"- P1 raw paired delta vs L2_A1 (late_t451_500, report only): {ci_dose['point']:.6g} "
        f"[{ci_dose['percentile_ci_lo']:.6g}, {ci_dose['percentile_ci_hi']:.6g}], "
        "not used in the registered P1 decision. The registered leaky band does not contain the L2_A1 point estimate.",
        f"- P2 raw unfit (late_t451_500, seed median of mean unfit): "
        f"{result['median_unfit_late_t451_500']:.6g}.",
        f"- P2 paired delta log10(mean unfit) vs L2_A1 (late_t451_500): "
        f"{ci_lr['point']:.6g} [{ci_lr['percentile_ci_lo']:.6g}, "
        f"{ci_lr['percentile_ci_hi']:.6g}].",
        f"- E_A1 reference delta (late_t451_500, recomputed): {ci_elu['point']:.6g} "
        f"[{ci_elu['percentile_ci_lo']:.6g}, {ci_elu['percentile_ci_hi']:.6g}].",
        f"- P2 sign condition (CI upper < 0): {'PASS' if result['p2_negative'] else 'FAIL'}.",
        f"- P2 descriptive effect-size reading vs registered E_A1 interval [-1.098, -0.696]: {result['p2_effect_relation']}. Values below the interval retain the harmless label because they indicate stronger improvement.",
        "- The E_A1 interval is descriptive; the registered primary condition is the CI sign.",
        "", "## Prediction registration and correspondence", "",
        "The numerical predictions were proposed in the draft first and then approved by Issa; they are not independent Issa predictions.",
        "| endpoint | preregistered prediction | observed window | observed |",
        "|---|---:|---|---:|",
        f"| layer-2 dose | 5.9 | late_t451_500 | {result['median_layer2_dose_late_t451_500']:.6g} |",
        f"| layer-2 submerged fraction | about 0.60 | step5m_t500 | {result['median_layer2_submerged_frac_step5m']:.6g} |",
        f"| mean unfit | <= 0.005 | late_t451_500 | {result['median_unfit_late_t451_500']:.6g} |",
        f"If a prediction misses, the preregistered suspected cause is: {cfg['preregistration']['predicted_failure_cause']}.",
        "", "## P3 (REPORT_ONLY)", "",
        f"- Layer-2 submerged fraction (step5m_t500, seed median): "
        f"{result['median_layer2_submerged_frac_step5m']:.6g}.",
        "- Both submerged counts and fractions are in verdict.csv; unit-level mobility and s_i=M_i+B_i are in s_distribution.csv.",
        "- Task-end, per-layer boundary snapshots are in layer_stats.csv.",
        "", "## Sanity", "",
        f"- Preflight: **{'PASS' if sanity.get('pass_') else 'FAIL'}**",
        f"- Final pairing: **{'PASS' if result['S_pair_final']['pass_'] else 'FAIL'}**",
        "- Mask check: 50 task-end points per seed in late_t451_500 and 500 task-end points per seed overall.",
        "- Floor: 1e-23 inherited from mlp2_phase1_0829 / elu_swamp_0830; not recalibrated.",
        "", "## Citation limits", "",
        "Do not generalize beyond condA, width 100, two hidden layers, and the 5M horizon. A 0/10 event count is not evidence that an event is impossible; its one-sided 95% upper bound is 0.2589.",
    ]
    (outdir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _provenance(cfg_path: Path, cfg: dict, outdir: Path, sanity: dict,
                analysis: dict, elapsed: dict, started: float) -> dict:
    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                           text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        git_hash = "unknown"
    refs = {}
    for label in ("relu", "elu"):
        refdir, arm = _reference(cfg, label)
        refs[label] = dict(dir=str(refdir), arm=arm,
                           config_sha256=(_sha_file(refdir / "config_used.yaml")
                                          if (refdir / "config_used.yaml").exists()
                                          else None))
    files = [outdir / name for name in ("config_used.yaml", "verdict.csv",
                                        "layer_stats.csv", "s_distribution.csv",
                                        "mask_counts.csv", "summary.md")]
    return dict(experiment=EXPERIMENT, git_commit=git_hash,
                command=sys.argv, cwd=os.getcwd(), python=sys.version,
                platform=platform.platform(), torch=torch.__version__,
                numpy=np.__version__, omp_num_threads=os.environ.get("OMP_NUM_THREADS"),
                config_path=str(cfg_path), config_sha256=_sha_file(cfg_path),
                spec=str(cfg["spec"]), preregistration=cfg["preregistration"],
                generator_offset=int(cfg["common"]["generator_offset"]),
                references=refs, sanity=sanity, analysis=analysis,
                elapsed_sec=elapsed, wall_sec=time.time() - started,
                output_sha256={p.name: _sha_file(p) for p in files if p.exists()})


def run_full(cfg_path: Path, cfg: dict, device: str, outdir: Path, *,
             smoke: bool = False) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    require_omp(cfg)
    if smoke:
        total, seeds = SMOKE_STEPS, list(SMOKE_SEEDS)
        sanity = dict(pass_=True, smoke=True, preregistration_not_evaluated=True)
    else:
        preflight_path = Path(ROOT) / PREFLIGHT_DIR / "preflight.json"
        if not preflight_path.exists():
            raise FileNotFoundError("run --preflight before the frozen 5M run")
        sanity = json.loads(preflight_path.read_text(encoding="utf-8"))
        if not sanity.get("pass_"):
            raise RuntimeError("saved LR_A1 preflight did not pass")
        total, seeds = int(cfg["common"]["total_steps"]), [int(v) for v in cfg["common"]["seeds"]]
    with (outdir / "config_used.yaml").open("w") as fh:
        yaml.safe_dump(cfg, fh, allow_unicode=True, sort_keys=False)
    every = int(cfg["common"]["lop_every"])
    divergence = _load_divergence(outdir, seeds, total, every)
    if divergence is not None:
        result = dict(status=NUMERIC_DIVERGENCE, divergence=divergence,
                      elapsed_sec=0.0, sanity=dict(pass_=False))
    elif _complete_arm_logs(outdir, ARM, seeds, total, every):
        result = dict(status="COMPLETE", resumed_from_complete_logs=True,
                      elapsed_sec=0.0, sanity=dict(pass_=True))
    else:
        result = _run_arm(cfg, device, outdir, total=total, seeds=seeds)
    elapsed = {ARM: float(result["elapsed_sec"])}
    if smoke:
        payload = dict(pass_=result["status"] == "COMPLETE", smoke=True,
                       run=result, elapsed_sec=elapsed)
        (outdir / "smoke_sanity.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        print(f"SMOKE DONE -> {outdir}", flush=True)
        return payload
    if result["status"] == NUMERIC_DIVERGENCE:
        provenance = _provenance(cfg_path, cfg, outdir, sanity,
                                 dict(main_verdict=NUMERIC_DIVERGENCE), elapsed, started)
        (outdir / "provenance.json").write_text(
            json.dumps(provenance, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        return dict(sanity=sanity, analysis=dict(main_verdict=NUMERIC_DIVERGENCE))
    analysis = analyze(cfg, outdir, sanity, elapsed)
    provenance = _provenance(cfg_path, cfg, outdir, sanity, analysis, elapsed, started)
    (outdir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"ALL DONE -> {outdir}", flush=True)
    return dict(sanity=sanity, analysis=analysis)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lr_a1_0901.yaml")
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
        raise ValueError("LR_A1 is CPU-only")
    stage = ("preflight" if args.preflight else "smoke" if args.smoke else
             "analyze" if args.analyze_only else "full")
    validate_config(cfg, stage=stage)
    main_dir = Path(ROOT) / cfg["output"]["dir"]
    outdir = (Path(args.outdir).resolve() if args.outdir else
              Path(ROOT) / PREFLIGHT_DIR if args.preflight else
              Path(ROOT) / SMOKE_DIR if args.smoke else main_dir)
    if args.preflight:
        preflight(cfg, device, outdir)
    elif args.analyze_only:
        sanity = json.loads((Path(ROOT) / PREFLIGHT_DIR / "preflight.json")
                            .read_text(encoding="utf-8"))
        analyze(cfg, outdir, sanity, {})
    else:
        run_full(cfg_path, cfg, device, outdir, smoke=args.smoke)


if __name__ == "__main__":
    main()
