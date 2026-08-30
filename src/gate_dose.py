"""gate_dose_0830: activation-gated fixed-dose response over five million steps.

The experiment crosses the frozen one-layer oracle-dose implementation with
ReLU, ELU, and leaky ReLU.  Frozen predecessors are imported rather than
edited; the only shared change is the leaky activation branch in ``VecMLPL``.

Stages::

    OMP_NUM_THREADS=1 .venv/bin/python -m src.gate_dose --preflight
    OMP_NUM_THREADS=1 .venv/bin/python -m src.gate_dose --s0prime
    OMP_NUM_THREADS=1 .venv/bin/python -m src.gate_dose

``--s0prime`` performs the real 5M ReLU runs and leaves their logs in the main
result directory.  The full stage resumes those arms and runs only ELU/leaky.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from .common import ROOT, load_config, pick_device
from .dose_const_5m import (EXTRA_LOG_KEYS, _input_stats, _refresh_fixed_offset,
                            _target, clopper_pearson, setup_arm_const)
from .elu_swamp import (ELU_EXTRA_LAYER_KEYS, ELU_EXTRA_UNIT_KEYS, EluRecorder,
                        exact_layer_record_elu, grads_centered_elu)
from .mlp2_phase0 import (_sha_array, _sha_file, identity_sanity_pass,
                          require_omp, write_csv)
from .mlp2_phase0b import _ci_components, _complete_arm_logs, _window_indices
from .mlp2_phase1 import (NUMERIC_DIVERGENCE, NumericDivergenceError,
                          StreamDigest, _env_hashes, _init_hashes,
                          _seed_state_hashes_p1)


EXPERIMENT = "gate_dose_0830"
ARM_ORDER = ("R_off", "R_933", "R_1216", "E_off", "E_933", "E_1216",
             "LR_off", "LR_933", "LR_1216")
RELU_ARMS = ("R_off", "R_933", "R_1216")
ELU_ARMS = ("E_off", "E_933", "E_1216")
LEAKY_ARMS = ("LR_off", "LR_933", "LR_1216")
SMOKE_STEPS = 30_000
SIGMA_TOL = 1e-8
IDENTITY_TOL = 1e-10
STATE_HASH_STEP = 1_000_000

REGISTERED_ARMS = {
    "R_off": ("relu", None, None, []),
    "R_933": ("relu", 2.333, 9.33, [1]),
    "R_1216": ("relu", 3.041, 12.16, [1]),
    "E_off": ("elu", None, None, []),
    "E_933": ("elu", 2.333, 9.33, [1]),
    "E_1216": ("elu", 3.041, 12.16, [1]),
    "LR_off": ("leaky", None, None, []),
    "LR_933": ("leaky", 2.333, 9.33, [1]),
    "LR_1216": ("leaky", 3.041, 12.16, [1]),
}

NEW_LOG_NAMES = {"activation", "act_alpha"}
NEW_LOG_SUFFIXES = tuple("_" + key for key in
                         ELU_EXTRA_UNIT_KEYS + ELU_EXTRA_LAYER_KEYS)
S0_META_KEYS = {"run_id", "arm"}


def _arm(cfg: dict, name: str) -> dict:
    return next(a for a in cfg["arms"] if a["name"] == name)


def _activation(cfg: dict, arm_cfg: dict) -> tuple[str, float]:
    label = str(arm_cfg["activation"])
    spec = cfg["activation"][label]
    if label == "elu":
        return "elu", float(spec["alpha"])
    if label == "leaky":
        return "leaky_relu", float(spec["slope"])
    return "relu", 1.0


def validate_config(cfg: dict, *, stage: str) -> None:
    if stage not in {"preflight", "smoke", "s0prime", "full", "analyze"}:
        raise ValueError(f"unknown stage {stage!r}")
    C, A, I, P, G, S = (cfg["common"], cfg["condA"], cfg["intervention"],
                         cfg["phase1"], cfg["gate_dose"], cfg["sanity"])
    if [a["name"] for a in cfg["arms"]] != list(ARM_ORDER):
        raise ValueError(f"arms must be ordered as {ARM_ORDER}")
    for arm in cfg["arms"]:
        act, target, dose, layers = REGISTERED_ARMS[arm["name"]]
        got_target = _target(arm)
        got_dose = arm.get("target_dose")
        if ([int(v) for v in arm["hidden"]] != [100]
                or [int(v) for v in arm.get("centered_layers", [])] != layers
                or str(arm["activation"]) != act
                or got_target != target
                or (None if got_dose is None else float(got_dose)) != dose):
            raise ValueError(f"{arm['name']} differs from the preregistration")
    if (int(A["m"]), int(A["f"]), int(A["target_hidden"])) != (20, 15, 100):
        raise ValueError("gate_dose requires condA m=20, f=15, teacher width=100")
    if list(A["T_values"]) != [10_000] or list(A["encodings"]) != ["std"]:
        raise ValueError("gate_dose requires T=10000 and std encoding")
    if (str(I["name"]) != "oracle_fixed_mu_offset" or I["oracle"] is not True
            or I["consumes_rng"] is not False
            or float(I["center_alpha_compat"]) != 0.01):
        raise ValueError("the oracle-dose intervention changed")
    if (str(cfg["activation"]["relu"]["name"]) != "relu"
            or str(cfg["activation"]["elu"]["name"]) != "elu"
            or float(cfg["activation"]["elu"]["alpha"]) != 1.0
            or str(cfg["activation"]["elu"]["derivative_form"]) != "alpha_exp"
            or str(cfg["activation"]["leaky"]["name"]) != "leaky_relu"
            or float(cfg["activation"]["leaky"]["slope"]) != 0.1
            or cfg["activation"]["autograd"] is not False
            or cfg["activation"]["consumes_rng"] is not False):
        raise ValueError("activation definitions changed")
    expected_phase = {
        "task_period": 10_000, "early_tasks": [2, 11],
        "late_tasks_5m": [491, 500], "window_1m_tasks": [91, 100],
        "onset_threshold": 0.05, "onset_present_min": 5,
        "unfit_floor": 1e-16, "recalibrate_floor": False,
        "bootstrap_B": 10_000, "bootstrap_seed": 20_260_829,
        "ci_method": "percentile_primary_studentized_secondary",
    }
    for key, value in expected_phase.items():
        if P[key] != value:
            raise ValueError(f"phase1.{key} differs from the preregistration")
    if (float(P["degenerate_se_tol"]) != 1e-15
            or float(P["degenerate_frac_max"]) != 0.01
            or float(P["degenerate_width_ratio_max"]) != 100.0):
        raise ValueError("CI degeneracy guard changed")
    if cfg["pairing"]["paired_groups"] != [list(ARM_ORDER)]:
        raise ValueError("all nine arms must be paired")
    baseline_by_dose = dict(cfg["pairing"]["baseline_by_dose"])
    # PyYAML's YAML 1.1 resolver parses the unquoted key ``off`` as False.
    if False in baseline_by_dose and "off" not in baseline_by_dose:
        baseline_by_dose["off"] = baseline_by_dose.pop(False)
    if baseline_by_dose != {"off": "R_off", "933": "R_933", "1216": "R_1216"}:
        raise ValueError("baseline_by_dose changed")
    if list(G["verdict_arms_elu"]) != ["E_933", "E_1216"]:
        raise ValueError("ELU verdict arms changed")
    if list(G["verdict_arms_leaky"]) != ["LR_933", "LR_1216"]:
        raise ValueError("leaky verdict arms changed")
    if int(G["relu_expected_onset_5m"]) != 10:
        raise ValueError("ReLU expected onset count changed")
    if G["p3_contrasts"] != [["E_off", "R_off"], ["E_933", "R_933"],
                              ["E_1216", "R_1216"], ["LR_off", "R_off"],
                              ["LR_933", "R_933"], ["LR_1216", "R_1216"]]:
        raise ValueError("P3 contrasts changed")
    if G["p4_contrasts"] != [["E_1216", "E_933"],
                              ["LR_1216", "LR_933"]]:
        raise ValueError("P4 contrasts changed")
    if (G["submerged_frac_in_verdict"] is not False
            or G["strict_dead_in_verdict"] is not False):
        raise ValueError("submergence/dead counts must stay out of the verdict")
    if (int(G["q2_layer"]) != 1 or list(G["q2_arms"]) != list(ELU_ARMS + LEAKY_ARMS)
            or int(G["q2_increment_interval_steps"]) != int(C["lop_every"])
            or list(G["q2_window_tasks"]) != [491, 500]
            or int(G["q2_bins"]) != 12
            or str(G["q2_bin_method"]) != "equal_count_quantile"
            or int(G["q2_bin_min_count"]) != 20
            or str(G["q2_log_base"]) != "natural"
            or int(G["q2_min_submerged_units_per_seed"]) != 3):
        raise ValueError("Q2 design changed")
    expected_divergence = dict(
        status=NUMERIC_DIVERGENCE,
        detection="nonfinite_training_state_at_probe", probe_every=1000,
        action="mark_arm_failed_and_continue", rescue="none",
        inconclusive_if_cell_missing=["E_1216", "LR_1216"])
    if G["numeric_divergence"] != expected_divergence:
        raise ValueError("numeric-divergence policy changed")
    if (dict(S["s0_prime_arm_map"]) != {
            "R_off": "dose_off", "R_933": "dose933", "R_1216": "dose1216"}
            or list(S["s0_prime_state_hash_steps"]) != [1_000_000, 5_000_000]
            or int(S["s_pair_steps"]) != 30_000
            or float(S["s_dose_rel_tol"]) != 1e-10
            or float(S["s_grad_finite_difference_tol"]) != 1e-6
            or S["s_leaky_limit_slope_to_zero"] is not True
            or S["s_elu_limit_alpha_to_zero"] is not True
            or S["s_submerge_matches_strict_dead_on_relu"] is not True
            or S["s_tautology_check"] is not True
            or int(S["omp_num_threads"]) != 1
            or S["s6_floor_calibration"] is not False):
        raise ValueError("sanity gates changed")
    if stage in {"s0prime", "full", "analyze"}:
        if int(C["total_steps"]) != 5_000_000 or list(C["seeds"]) != list(range(10)):
            raise ValueError("the full run requires 5M steps and seeds 0..9")
        if str(C["device"]) != "cpu":
            raise ValueError("gate_dose is CPU-only")


# ---------------------------------------------------------------------------
# Learning path
# ---------------------------------------------------------------------------
def setup_arm_gate(cfg: dict, arm_cfg: dict, device: str) -> dict:
    st = setup_arm_const(cfg, arm_cfg, device)
    act, alpha = _activation(cfg, arm_cfg)
    form = str(cfg["activation"]["elu"]["derivative_form"])
    st["net"].set_activation(act, alpha, form)
    st["activation"] = act
    st["act_alpha"] = float(alpha)
    return st


def forward_gate(st: dict, x: torch.Tensor):
    """Oracle-dose forward path with a selectable hidden activation.

    The ReLU branches preserve the operation order in ``forward_const``:
    off updates the compatibility running mean and fixed arms refresh the
    float64 oracle offset before subtracting it at the float32 boundary.
    """
    fixed = st.get("target_mu_norm") is not None
    if fixed:
        _refresh_fixed_offset(st)
    net, cur = st["net"], x
    inputs, pres, acts = [], [], []
    for li, (W, b) in enumerate(zip(net.Ws, net.bs)):
        mean = st["layer_means"][li]
        if fixed:
            cur_in = cur - mean.to(cur.dtype)
        else:
            cur_in = cur - mean if st["centered_layers"][li] else cur
            if mean is not None:
                mean.mul_(1.0 - st["center_alpha"]).add_(st["center_alpha"] * cur)
        pre = torch.einsum("rhd,rd->rh", W, cur_in) + b
        cur = net.act_fn(pre)
        inputs.append(cur_in)
        pres.append(pre)
        acts.append(cur)
    yhat = (acts[-1] * net.v).sum(dim=1) + net.c
    return inputs, pres, acts, yhat


def save_checkpoint_gate(st: dict, arm: str, step: int, outdir: Path) -> Path:
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
                    runs=st["runs"]), path)
    return path


def train_arm_gate(st: dict, recorder, probe_steps, total: int, outdir: Path,
                   checkpoints, stream_hook=None) -> float:
    probe_set = {int(v) for v in probe_steps}
    checkpoint_set = {int(v) for v in checkpoints}
    net, env, teacher = st["net"], st["env"], st["teacher"]
    started = time.time()
    for step in range(total):
        if step in checkpoint_set:
            save_checkpoint_gate(st, st["arm"], step, outdir)
        if step in probe_set:
            recorder(st, step)
        x = env.step()
        y = teacher(x)
        if stream_hook is not None:
            stream_hook(step, x, y)
        inputs, pres, acts, yhat = forward_gate(st, x)
        grads = grads_centered_elu(net, inputs, pres, acts, yhat - y)
        net.sgd_step_layers(st["lr"], *grads)
    if total in probe_set:
        recorder(st, total)
    if total in checkpoint_set:
        save_checkpoint_gate(st, st["arm"], total, outdir)
    return time.time() - started


class GateRecorder(EluRecorder):
    def __init__(self, steps: list[int], st: dict):
        super().__init__(steps, st, SIGMA_TOL, IDENTITY_TOL, 1000,
                         zbar_layers=[1], readout_steps=[])
        n, runs = len(self.steps), st["R"]
        self.extra = {key: np.empty((n, runs), dtype=np.float64)
                      for key in EXTRA_LOG_KEYS}
        self.state_hash_1m: dict[int, dict[str, str]] = {}

    def __call__(self, st: dict, step: int) -> None:
        if st.get("target_mu_norm") is not None:
            _refresh_fixed_offset(st)
        super().__call__(st, step)
        i = self.index.get(int(step))
        if i is None:
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
            self.extra[key][i] = value.detach().cpu().numpy()
        if int(step) == STATE_HASH_STEP:
            self.state_hash_1m = {
                int(run["seed"]): _seed_state_hashes_p1(st, ri)
                for ri, run in enumerate(st["runs"])
            }


def write_arm_logs_gate(outdir: Path, arm: str, st: dict,
                        rec: GateRecorder) -> list[Path]:
    logdir = outdir / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ri, run in enumerate(st["runs"]):
        seed = int(run["seed"])
        payload = dict(
            step=rec.steps, run_id=np.array(run["run_id"]), arm=np.array(arm),
            seed=np.int64(seed), activation=np.array(st["activation"]),
            act_alpha=np.float64(st["act_alpha"]),
            task_period=np.int64(run["period"]),
            target_mu_norm=np.float64(np.nan if st.get("target_mu_norm") is None
                                      else st["target_mu_norm"]),
            target_dose=np.float64(np.nan if st.get("target_dose") is None
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


def _arm_status_path(outdir: Path, arm: str) -> Path:
    return outdir / "arm_status" / f"{arm}.json"


def _write_divergence_status(outdir: Path, event: dict) -> Path:
    path = _arm_status_path(outdir, str(event["arm"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(event, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _load_divergence_status(outdir: Path, arm: str, seeds: list[int],
                            total: int, probe_every: int) -> dict | None:
    path = _arm_status_path(outdir, arm)
    if not path.exists():
        return None
    try:
        event = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    valid = (event.get("status") == NUMERIC_DIVERGENCE
             and event.get("arm") == arm
             and event.get("registered_seeds") == seeds
             and int(event.get("registered_total_steps", -1)) == total
             and int(event.get("probe_every", -1)) == probe_every
             and event.get("rescue") == "none")
    return event if valid else None


def _run_arm(cfg: dict, arm: str, device: str, outdir: Path, seeds: list[int],
             total: int) -> dict:
    c = copy.deepcopy(cfg)
    c["common"]["seeds"] = seeds
    st = setup_arm_gate(c, _arm(c, arm), device)
    every = int(c["common"]["lop_every"])
    probes = list(range(0, total + 1, every))
    if probes[-1] != total:
        probes.append(total)
    _, sanity0 = exact_layer_record_elu(st, SIGMA_TOL)
    if not identity_sanity_pass(sanity0, IDENTITY_TOL):
        raise RuntimeError(f"{arm} initial exact-support identity failed")
    rec = GateRecorder(probes, st)
    checkpoints = [int(v) for v in c["common"].get("checkpoints", [])
                   if int(v) <= total]
    print(f"[{arm}] act={st['activation']} alpha={st['act_alpha']:g} "
          f"target={st.get('target_mu_norm')} seeds={seeds} steps={total:,}",
          flush=True)
    started = time.time()
    try:
        elapsed = train_arm_gate(st, rec, probes, total, outdir, checkpoints)
    except NumericDivergenceError as exc:
        elapsed = time.time() - started
        event = dict(exc.event)
        event.update(probe_every=every, registered_total_steps=int(total),
                     registered_seeds=[int(v) for v in seeds],
                     activation=st["activation"], act_alpha=st["act_alpha"],
                     elapsed_sec=float(elapsed),
                     detection="nonfinite_training_state_at_probe",
                     partial_logs_excluded=True, rescue="none")
        status = _write_divergence_status(outdir, event)
        print(f"[{arm}] {NUMERIC_DIVERGENCE} at step "
              f"{event['detected_step']:,} -> {status}", flush=True)
        return dict(status=NUMERIC_DIVERGENCE, elapsed_sec=elapsed,
                    sanity=dict(pass_=False, numeric_divergence=True, event=event),
                    divergence=event, final_env=_env_hashes(st))
    sanity = rec.sanity()
    if not sanity["pass_"]:
        raise RuntimeError(f"{arm} exact-support sanity failed: {sanity}")
    write_arm_logs_gate(outdir, arm, st, rec)
    print(f"[{arm}] complete in {elapsed:.1f}s", flush=True)
    return dict(status="COMPLETE", elapsed_sec=elapsed, sanity=sanity,
                final_env=_env_hashes(st))


# ---------------------------------------------------------------------------
# Preregistered sanity gates
# ---------------------------------------------------------------------------
def _network_gradient_error(act: str, alpha: float) -> float:
    from .nets import VecMLPL

    gen = torch.Generator().manual_seed(7301)
    net = VecMLPL(2, [3], 4, gen, "cpu").set_activation(act, alpha, "alpha_exp")
    net.Ws = [v.double() for v in net.Ws]
    net.bs = [v.double() for v in net.bs]
    net.v, net.c = net.v.double(), net.c.double()
    net.W, net.b = net.Ws[0], net.bs[0]
    x = torch.tensor([[0.3, -0.8, 1.1, -0.2],
                      [-0.4, 0.7, -1.2, 0.6]], dtype=torch.float64)
    target = torch.tensor([0.25, -0.15], dtype=torch.float64)

    def forward_loss() -> torch.Tensor:
        pre = torch.einsum("rhd,rd->rh", net.Ws[0], x) + net.bs[0]
        a = net.act_fn(pre)
        yhat = (a * net.v).sum(dim=1) + net.c
        return (yhat - target).square()

    pre = torch.einsum("rhd,rd->rh", net.Ws[0], x) + net.bs[0]
    a = net.act_fn(pre)
    yhat = (a * net.v).sum(dim=1) + net.c
    analytic = grads_centered_elu(net, [x], [pre], [a], yhat - target)
    tensors = [(net.Ws[0], analytic[0][0]), (net.bs[0], analytic[1][0]),
               (net.v, analytic[2]), (net.c, analytic[3])]
    h, worst = 1e-6, 0.0
    for param, grad in tensors:
        flat, gflat = param.reshape(param.shape[0], -1), grad.reshape(grad.shape[0], -1)
        for j in range(flat.shape[1]):
            original = flat[:, j].clone()
            flat[:, j] = original + h
            up = forward_loss()
            flat[:, j] = original - h
            down = forward_loss()
            flat[:, j] = original
            fd = (up - down) / (2 * h)
            scale = torch.maximum(fd.abs(), gflat[:, j].abs()).clamp_min(1e-8)
            worst = max(worst, float(((fd - gflat[:, j]).abs() / scale).max()))
    return worst


def _s_grad_check(cfg: dict) -> dict:
    import torch.nn.functional as F
    from .nets import VecMLPL

    tol = float(cfg["sanity"]["s_grad_finite_difference_tol"])
    points = torch.tensor([float(v) for v in cfg["sanity"]["s_grad_probe_points"]],
                          dtype=torch.float64)
    rows, failures, network = [], [], {}
    for label, act, alpha in (("elu", "elu", 1.0),
                              ("leaky", "leaky_relu", 0.1)):
        net = VecMLPL(1, [2], 2, torch.Generator().manual_seed(0), "cpu")
        net.set_activation(act, alpha, "alpha_exp")
        got_a = net.act_fn(points)
        ref_a = F.elu(points, 1.0) if act == "elu" else F.leaky_relu(points, 0.1)
        z = points.clone().requires_grad_(True)
        ref_value = F.elu(z, 1.0) if act == "elu" else F.leaky_relu(z, 0.1)
        ref_value.sum().backward()
        got_g = net.act_grad(points, got_a)
        rel_auto = (got_g - z.grad).abs() / z.grad.abs().clamp_min(1e-300)
        if float(rel_auto.max()) > tol or not torch.equal(got_a, ref_a):
            failures.append(dict(activation=label, where="torch_reference",
                                 max_relerr=float(rel_auto.max())))
        eps = np.finfo(np.float64).eps
        for i, value in enumerate(points.tolist()):
            # Stay on one side of the kink.  At z=0 leaky has two registered
            # directional slopes; the backward derivative is the training one.
            h = min(1e-6, max(abs(value) / 10.0, 1e-8)) if value else 1e-7
            if value == 0.0 and act == "leaky_relu":
                fd = float((net.act_fn(points[i:i + 1])
                            - net.act_fn(points[i:i + 1] - h)) / h)
                fd_forward = float((net.act_fn(points[i:i + 1] + h)
                                    - net.act_fn(points[i:i + 1])) / h)
            else:
                fd = float((net.act_fn(points[i:i + 1] + h)
                            - net.act_fn(points[i:i + 1] - h)) / (2 * h))
                fd_forward = float("nan")
            exact = float(got_g[i])
            rel = abs(fd - exact) / max(abs(exact), 1e-300)
            roundoff_floor = eps * max(abs(float(got_a[i])), 1.0) / (h * max(abs(exact), 1e-300))
            informative = roundoff_floor < 1e-6
            row = dict(activation=label, z=value, closed_form=exact,
                       autograd=float(z.grad[i]), finite_difference=fd,
                       finite_difference_forward=fd_forward,
                       finite_difference_relerr=rel,
                       roundoff_floor=roundoff_floor, informative=int(informative))
            rows.append(row)
            if informative and rel > tol:
                failures.append(row)
            if value == 0.0 and act == "leaky_relu" and abs(fd_forward - 1.0) > tol:
                failures.append(dict(activation=label, z=0.0,
                                     where="right_direction", value=fd_forward))
        network[label] = _network_gradient_error(act, alpha)
        if network[label] > tol:
            failures.append(dict(activation=label, where="network_finite_difference",
                                 max_relerr=network[label]))
    return dict(pass_=not failures, tolerance=tol, probes=rows,
                network_finite_difference_max_relerr=network, failures=failures)


def _s_limit_check(cfg: dict, act: str, name: str, steps: int = 2000) -> dict:
    c = copy.deepcopy(cfg)
    c["common"]["seeds"] = [0, 1]
    arm = copy.deepcopy(_arm(c, "R_933"))
    relu = setup_arm_gate(c, arm, "cpu")
    other = setup_arm_gate(c, arm, "cpu")
    other["net"].set_activation(act, 0.0, "alpha_exp")
    other["activation"], other["act_alpha"] = act, 0.0
    grid = torch.linspace(-30, 30, 4001, dtype=torch.float64)
    static_forward = bool(torch.equal(relu["net"].act_fn(grid),
                                      other["net"].act_fn(grid)))
    static_grad = bool(torch.equal(
        relu["net"].act_grad(grid, relu["net"].act_fn(grid)),
        other["net"].act_grad(grid, other["net"].act_fn(grid))))
    train_arm_gate(relu, lambda *_: None, [], steps, Path("."), [])
    train_arm_gate(other, lambda *_: None, [], steps, Path("."), [])
    a, b = _init_hashes(relu), _init_hashes(other)
    differences = sorted(k for k, v in a.items() if b.get(k) != v)
    return dict(pass_=bool(static_forward and static_grad and not differences),
                activation=name, steps=steps, static_forward_equal=static_forward,
                static_grad_equal=static_grad, trained_state_differences=differences)


def _s_pair_and_dose(cfg: dict, outdir: Path) -> dict:
    steps = int(cfg["sanity"]["s_pair_steps"])
    every = int(cfg["common"]["lop_every"])
    init, final, streams, dose_rows = {}, {}, {}, []
    dose_arrays: dict[tuple[str, float], list[np.ndarray]] = {}
    for arm in ARM_ORDER:
        c = copy.deepcopy(cfg)
        st = setup_arm_gate(c, _arm(c, arm), "cpu")
        init[arm] = _init_hashes(st)
        stream = StreamDigest()

        def dose_probe(state: dict, step: int, arm_name: str = arm) -> None:
            if state.get("target_mu_norm") is not None:
                _refresh_fixed_offset(state)
            stats = _input_stats(state)
            target = state.get("target_mu_norm")
            errors = stats["relative_error"].detach().cpu().numpy()
            mu = stats["mu_norm"].detach().cpu().numpy()
            dose_rows.append(dict(arm=arm_name, step=int(step),
                                  target_mu_norm="" if target is None else target,
                                  max_relative_error="" if target is None
                                  else float(errors.max()),
                                  mu_norm_values=mu.tolist()))
            if target is not None:
                dose_arrays.setdefault((arm_name, float(target)), []).append(mu.copy())

        print(f"[S-pair/S-dose] {arm} {steps:,} steps", flush=True)
        train_arm_gate(st, dose_probe, range(0, steps + 1, every), steps,
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
    tol = float(cfg["sanity"]["s_dose_rel_tol"])
    dose_fail = [r for r in dose_rows if r["max_relative_error"] != ""
                 and float(r["max_relative_error"]) > tol]
    by_target = {2.333: ("R_933", "E_933", "LR_933"),
                 3.041: ("R_1216", "E_1216", "LR_1216")}
    activation_mismatch = []
    for target, arms in by_target.items():
        reference_values = dose_arrays[(arms[0], target)]
        for arm in arms[1:]:
            for i, (a, b) in enumerate(zip(reference_values, dose_arrays[(arm, target)])):
                if not np.array_equal(a, b):
                    activation_mismatch.append(dict(target=target, arm=arm, record=i))
    return dict(
        spair=dict(pass_=not differences, reference=reference, arms=list(ARM_ORDER),
                   steps=steps, differences=differences,
                   caveat=str(cfg["pairing"]["pairing_caveat"]),
                   init_hashes=init, final_env_hashes=final, stream_digests=streams),
        sdose=dict(pass_=bool(not dose_fail and not activation_mismatch),
                   tolerance=tol, rows=dose_rows, failures=dose_fail,
                   activation_mismatches=activation_mismatch))


def _s_submerge_check(cfg: dict, outdir: Path) -> dict:
    c = copy.deepcopy(cfg)
    c["common"]["seeds"] = [0, 1]
    rows, failures = [], []
    for arm in RELU_ARMS:
        st = setup_arm_gate(c, _arm(c, arm), "cpu")
        train_arm_gate(st, lambda *_: None, [], 5000, outdir, [])
        rec, sanity = exact_layer_record_elu(st, SIGMA_TOL)
        same = bool(torch.equal(rec["layers"][0]["submerged"],
                                rec["layers"][0]["strict_dead"]))
        row = dict(arm=arm, equal=same,
                   elementwise_mismatch=int(sanity["layers"][0]["submerge_mismatch"]))
        rows.append(row)
        if not same or row["elementwise_mismatch"]:
            failures.append(row)
    return dict(pass_=not failures, rows=rows, failures=failures)


def _gate_verdict(onset: dict[str, int], divergences: set[str] | None = None) -> str:
    divergences = divergences or set()
    if divergences & {"E_1216", "LR_1216"}:
        return "INCONCLUSIVE_DIVERGENCE"
    elu = [onset[a] for a in ("E_933", "E_1216")]
    leaky = [onset[a] for a in ("LR_933", "LR_1216")]
    ez, ep = all(v == 0 for v in elu), any(v >= 5 for v in elu)
    lz, lp = all(v == 0 for v in leaky), any(v >= 5 for v in leaky)
    if ez and lz:
        return "GATE_LOAD_BEARING"
    if ep and lz:
        return "SOFT_WALL_AT_HIGH_DOSE"
    if ep and lp:
        return "DOSE_NOT_GATE_MEDIATED"
    if ez and lp:
        return "SMOOTHNESS_REQUIRED"
    return "PARTIAL"


def _s_tautology(cfg: dict, outdir: Path) -> dict:
    if cfg["sanity"]["s_tautology_check"] is not True:
        raise ValueError("S-taut is required")
    c = copy.deepcopy(cfg)
    c["common"]["seeds"] = [0, 1]
    values, hashes = {}, {}
    for arm in ("R_933", "E_933", "LR_933"):
        st = setup_arm_gate(c, _arm(c, arm), "cpu")
        train_arm_gate(st, lambda *_: None, [], 2000, outdir, [])
        rec, _ = exact_layer_record_elu(st, SIGMA_TOL)
        values[arm] = rec["run"]["unfit"].detach().cpu().numpy().tolist()
        hashes[arm] = {k: _sha_array(v) for k, v in st["net"].state_dict().items()}
    activation_changes_state = any(hashes[a] != hashes["R_933"]
                                   for a in ("E_933", "LR_933"))
    synthetic = {
        "load": _gate_verdict({"E_933": 0, "E_1216": 0,
                                "LR_933": 0, "LR_1216": 0}),
        "soft": _gate_verdict({"E_933": 5, "E_1216": 0,
                                "LR_933": 0, "LR_1216": 0}),
        "dose": _gate_verdict({"E_933": 5, "E_1216": 0,
                                "LR_933": 5, "LR_1216": 0}),
        "smooth": _gate_verdict({"E_933": 0, "E_1216": 0,
                                  "LR_933": 5, "LR_1216": 0}),
        "partial": _gate_verdict({"E_933": 1, "E_1216": 0,
                                   "LR_933": 0, "LR_1216": 0}),
    }
    expected = dict(load="GATE_LOAD_BEARING", soft="SOFT_WALL_AT_HIGH_DOSE",
                    dose="DOSE_NOT_GATE_MEDIATED", smooth="SMOOTHNESS_REQUIRED",
                    partial="PARTIAL")
    return dict(pass_=bool(activation_changes_state and synthetic == expected),
                activation_changes_state=activation_changes_state,
                short_run_unfit=values, trained_state_hashes=hashes,
                verdict_mutants=synthetic, expected=expected)


def _s_floor_inheritance(cfg: dict) -> dict:
    reference = Path(ROOT) / cfg["sanity"]["s0_prime_baseline_ref"] / "floor_calibration.csv"
    if not reference.exists():
        return dict(pass_=False, reference=str(reference), reason="missing")
    data = np.genfromtxt(reference, delimiter=",", names=True)
    values = np.unique(np.asarray(data["calibrated_floor"], dtype=np.float64))
    configured = float(cfg["phase1"]["unfit_floor"])
    return dict(pass_=bool(values.size == 1 and values[0] == configured
                           and cfg["phase1"]["recalibrate_floor"] is False),
                reference=str(reference), reference_values=values.tolist(),
                configured=configured, recalibrated=False)


def _s_ci_selftest(cfg: dict) -> dict:
    P = cfg["phase1"]
    n = len(cfg["common"]["seeds"])
    draws = np.random.default_rng(int(P["bootstrap_seed"])).integers(
        0, n, size=(int(P["bootstrap_B"]), n))
    result = _ci_components(np.zeros(n), draws, "median",
                            float(P["degenerate_se_tol"]),
                            float(P["degenerate_frac_max"]),
                            float(P["degenerate_width_ratio_max"]))
    return dict(pass_=bool(result["ci_degenerate"]), result=result)


def preflight(cfg: dict, device: str, outdir: Path) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    checks: dict[str, dict] = {"S1_omp": require_omp(cfg)}
    print("[S-grad] ELU/leaky closed-form gradients", flush=True)
    checks["S_grad"] = _s_grad_check(cfg)
    print("[S-elu-limit] alpha -> 0", flush=True)
    checks["S_elu_limit"] = _s_limit_check(cfg, "elu", "elu")
    print("[S-leaky-limit] slope -> 0", flush=True)
    checks["S_leaky_limit"] = _s_limit_check(cfg, "leaky_relu", "leaky")
    pair = _s_pair_and_dose(cfg, outdir / "spair")
    checks["S_pair"], checks["S_dose"] = pair["spair"], pair["sdose"]
    checks["S_submerge"] = _s_submerge_check(cfg, outdir / "ssubmerge")
    checks["S_tautology"] = _s_tautology(cfg, outdir / "staut")
    checks["S6_floor_inherited"] = _s_floor_inheritance(cfg)
    checks["S_CI_degeneracy"] = _s_ci_selftest(cfg)
    result = dict(pass_=bool(all(v.get("pass_") for v in checks.values())), **checks)
    (outdir / "preflight.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    for name, value in checks.items():
        print(f"[{name}] {'PASS' if value.get('pass_') else 'FAIL'}", flush=True)
    if not result["pass_"]:
        failed = [k for k, v in checks.items() if not v.get("pass_")]
        raise RuntimeError(f"preflight failed: {failed}")
    return result


# ---------------------------------------------------------------------------
# S0': frozen ReLU dose runs
# ---------------------------------------------------------------------------
def _compare_reference_log(ours: Path, reference: Path) -> list[dict]:
    with np.load(ours, allow_pickle=False) as a, np.load(reference, allow_pickle=False) as b:
        keys_a = {key for key in set(a.files) - S0_META_KEYS
                  if key not in NEW_LOG_NAMES and not key.endswith(NEW_LOG_SUFFIXES)}
        keys_b = set(b.files) - S0_META_KEYS
        differences = [dict(column=k, reason="missing in fixed-dose reference")
                       for k in sorted(keys_a - keys_b)]
        differences += [dict(column=k, reason="missing in gate_dose")
                        for k in sorted(keys_b - keys_a)]
        for key in sorted(keys_a & keys_b):
            if _sha_array(a[key]) != _sha_array(b[key]):
                differences.append(dict(column=key, reason="hash mismatch"))
    return differences


def _checkpoint_hashes(path: Path) -> dict[str, str]:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    out = {f"net.{k}": _sha_array(v) for k, v in blob["net"].items()}
    out.update({f"teacher.{k}": _sha_array(v)
                for k, v in blob["teacher"].items()})
    out.update(env_flip_state=_sha_array(blob["env"]["flip_state"]),
               env_t=str(blob["env"]["t"]),
               running_mean=_sha_array(blob["running_mean"]))
    return out


def s0prime(cfg: dict, device: str, outdir: Path) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    require_omp(cfg)
    C, S = cfg["common"], cfg["sanity"]
    total, every = int(C["total_steps"]), int(C["lop_every"])
    seeds = [int(v) for v in C["seeds"]]
    reference_dir = (Path(ROOT) / S["s0_prime_baseline_ref"]).resolve()
    arms_result, elapsed = {}, {}
    for arm, reference_arm in dict(S["s0_prime_arm_map"]).items():
        if _complete_arm_logs(outdir, arm, seeds, total, every):
            print(f"[S0'] complete {arm} logs found; comparing only", flush=True)
            elapsed[arm] = 0.0
        else:
            run = _run_arm(cfg, arm, device, outdir, seeds, total)
            if run["status"] != "COMPLETE":
                raise RuntimeError(f"S0' ReLU arm diverged unexpectedly: {arm}")
            elapsed[arm] = run["elapsed_sec"]
        differences, missing = [], []
        for seed in seeds:
            ours = outdir / "logs" / f"{arm}_seed{seed}.npz"
            reference = reference_dir / "logs" / f"{reference_arm}_seed{seed}.npz"
            if not reference.exists():
                missing.append(str(reference))
            else:
                differences += [dict(seed=seed, **row)
                                for row in _compare_reference_log(ours, reference)]
        state_differences = []
        state_hashes = {}
        for step in [int(v) for v in S["s0_prime_state_hash_steps"]]:
            ours = outdir / "ckpts" / f"{arm}_step{step}.pt"
            reference = reference_dir / "ckpts" / f"{reference_arm}_step{step}.pt"
            if not ours.exists() or not reference.exists():
                missing.extend(str(p) for p in (ours, reference) if not p.exists())
                continue
            actual, expected = _checkpoint_hashes(ours), _checkpoint_hashes(reference)
            state_hashes[str(step)] = dict(actual=actual, expected=expected)
            state_differences.extend(dict(step=step, key=key)
                                     for key, value in expected.items()
                                     if actual.get(key) != value)
        arms_result[arm] = dict(
            pass_=bool(not differences and not missing and not state_differences),
            reference_arm=reference_arm, missing=missing,
            column_differences=differences, state_differences=state_differences,
            state_hashes=state_hashes)
    result = dict(pass_=bool(all(v["pass_"] for v in arms_result.values())),
                  arms=arms_result, reference=str(reference_dir),
                  total_steps=total, seeds=seeds, elapsed_sec=elapsed,
                  ignored_columns=sorted(S0_META_KEYS),
                  new_columns=sorted(NEW_LOG_NAMES),
                  new_suffixes=sorted(NEW_LOG_SUFFIXES))
    (outdir / "s0prime.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"S0' {'PASS' if result['pass_'] else 'FAIL'}", flush=True)
    if not result["pass_"]:
        raise RuntimeError("S0' failed; the full run must not proceed")
    return result


def _pair_check_final(cfg: dict, outdir: Path, complete: list[str],
                      divergences: dict[str, dict]) -> dict:
    seeds = [int(v) for v in cfg["common"]["seeds"]]
    if not complete:
        return dict(pass_=False, differences=[dict(where="no_complete_arm")])

    def env(arm: str, seed: int) -> dict:
        with np.load(outdir / "logs" / f"{arm}_seed{seed}.npz",
                     allow_pickle=False) as z:
            state = json.loads(str(z["state_hash_final"]))
        return {key: state[key] for key in ("env.flip_state", "env.t")}

    reference, differences = complete[0], []
    for seed in seeds:
        expected = env(reference, seed)
        for arm in complete[1:]:
            if env(arm, seed) != expected:
                differences.append(dict(seed=seed, arm=arm, where="env"))
    return dict(pass_=not differences, reference_arm=reference,
                paired_arms=complete, not_tested_divergent=sorted(divergences),
                partial_due_to_numeric_divergence=bool(divergences),
                caveat=str(cfg["pairing"]["pairing_caveat"]),
                differences=differences)


def _dose_check_final(cfg: dict, outdir: Path, complete: list[str]) -> dict:
    tol = float(cfg["sanity"]["s_dose_rel_tol"])
    seeds = [int(v) for v in cfg["common"]["seeds"]]
    failures, n = [], 0
    for arm in complete:
        if _target(_arm(cfg, arm)) is None:
            continue
        for seed in seeds:
            with np.load(outdir / "logs" / f"{arm}_seed{seed}.npz",
                         allow_pickle=False) as z:
                errors = np.asarray(z["dose_relative_error"], dtype=np.float64)
            n += errors.size
            bad = errors > tol
            if bad.any():
                failures.append(dict(arm=arm, seed=seed,
                                     max_relative_error=float(errors.max())))
    return dict(pass_=not failures, tolerance=tol, n_values=n, failures=failures)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def _load_arm(cfg: dict, outdir: Path, arm: str) -> dict:
    keys = (
        "unfit", "eval_loss_exact", "gamma", "gamma_negative", "mu_cos_off",
        "dose_relative_error", "layer1_mu_norm", "layer1_dose",
        "layer1_eff_rank", "layer1_alive", "layer1_strict_dead",
        "layer1_w_norm_median", "layer1_median_M", "layer1_median_B",
        "layer1_preact_sd_median", "layer1_submerged")
    items = []
    for seed in [int(v) for v in cfg["common"]["seeds"]]:
        with np.load(outdir / "logs" / f"{arm}_seed{seed}.npz",
                     allow_pickle=False) as z:
            items.append({"step": z["step"].copy(),
                          **{key: z[key].copy() for key in keys}})
    result = {"step": items[0]["step"]}
    result.update({key: np.stack([item[key] for item in items], axis=1)
                   for key in keys})
    return result


def _window(data: dict, cfg: dict, tasks: list[int]) -> dict:
    floor = float(cfg["phase1"]["unfit_floor"])
    idx = _window_indices(data["step"], int(cfg["phase1"]["task_period"]), tasks)
    raw_u = np.asarray(data["unfit"], dtype=np.float64)[idx].mean(axis=0)
    u = np.maximum(raw_u, floor)
    metrics = {}
    for key in ("eval_loss_exact", "layer1_mu_norm", "layer1_dose",
                "layer1_eff_rank", "layer1_alive", "layer1_strict_dead",
                "layer1_w_norm_median", "layer1_median_M", "layer1_median_B",
                "layer1_preact_sd_median", "layer1_submerged", "gamma",
                "mu_cos_off"):
        values = np.asarray(data[key], dtype=np.float64)[idx]
        finite = np.isfinite(values)
        count = finite.sum(axis=0)
        total = np.where(finite, values, 0.0).sum(axis=0)
        mean = np.full(count.shape, np.nan, dtype=np.float64)
        np.divide(total, count, out=mean, where=count > 0)
        metrics[key] = mean
    return dict(index=idx, raw_u=raw_u, u=u, log_u=np.log10(u), metrics=metrics,
                floor_fraction=float(np.mean(np.asarray(data["unfit"])[idx] <= floor)))


def _ci(cfg: dict, values: np.ndarray, draws: np.ndarray) -> dict:
    P = cfg["phase1"]
    return _ci_components(np.asarray(values, dtype=np.float64), draws, "median",
                          float(P["degenerate_se_tol"]),
                          float(P["degenerate_frac_max"]),
                          float(P["degenerate_width_ratio_max"]))


def _interval_rows(cfg: dict, outdir: Path, arm: str, seed: int) -> dict:
    G, P = cfg["gate_dose"], cfg["phase1"]
    period = int(P["task_period"])
    every = int(G["q2_increment_interval_steps"])
    lo, hi = [int(v) for v in G["q2_window_tasks"]]
    nbins, min_count = int(G["q2_bins"]), int(G["q2_bin_min_count"])
    min_units = int(G["q2_min_submerged_units_per_seed"])
    with np.load(outdir / "logs" / f"{arm}_seed{seed}.npz",
                 allow_pickle=False) as z:
        steps = z["step"].astype(np.int64)
        zbar = z["layer1_zbar"].astype(np.float64)
        dzbar = z["layer1_dzbar"].astype(np.float64)
        p_hat = z["layer1_p_hat"].astype(np.float64)
    start = steps[:-1]
    adjacent = steps[1:] - steps[:-1] == every
    task = start // period + 1
    keep = ((start > 0) & (start % period != 0) & adjacent
            & (task >= lo) & (task <= hi))
    idx = np.flatnonzero(keep)
    sub = p_hat[idx] == 0.0
    submerged_units = np.sum(sub, axis=1) if sub.size else np.array([])
    median_units = float(np.median(submerged_units)) if submerged_units.size else 0.0
    if median_units < min_units:
        return dict(status="INSUFFICIENT_DATA", bins=[], beta=float("nan"),
                    rho=float("nan"), n_intervals=int(idx.size),
                    n_unit_intervals=int(sub.sum()),
                    median_submerged_units=median_units)
    x0, inc = zbar[idx], dzbar[idx + 1]
    good = sub & np.isfinite(x0) & np.isfinite(inc)
    x, y = x0[good], inc[good]
    rows, beta, rho = [], float("nan"), float("nan")
    if x.size >= nbins * min_count:
        edges = np.quantile(x, np.linspace(0.0, 1.0, nbins + 1))
        edges[0], edges[-1] = -np.inf, np.inf
        which = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, nbins - 1)
        for b in range(nbins):
            sel = which == b
            n = int(sel.sum())
            if not n:
                continue
            xb, yb = x[sel], y[sel]
            sd, med = float(yb.std(ddof=0)), float(np.median(yb))
            eligible = bool(n >= min_count and np.isfinite(sd) and sd > 0)
            rows.append(dict(bin=b, n=n, zbar_bin_median=float(np.median(xb)),
                             zbar_bin_lo=float(xb.min()), zbar_bin_hi=float(xb.max()),
                             dzbar_median=med, dzbar_sd=sd,
                             rho=med / sd if eligible else float("nan"),
                             eligible=int(eligible)))
        fit = [r for r in rows if r["eligible"]]
        if len(fit) >= 2:
            bx = np.asarray([r["zbar_bin_median"] for r in fit])
            by = np.log(np.asarray([r["dzbar_sd"] for r in fit]))
            if np.isfinite(by).all() and bx.std() > 0:
                beta = float(np.polyfit(bx, by, 1)[0])
            negative = [r["rho"] for r in fit if r["zbar_bin_median"] < 0]
            if negative:
                rho = float(np.median(negative))
    status = "OK" if np.isfinite(beta) and np.isfinite(rho) else "INSUFFICIENT_DATA"
    return dict(status=status, bins=rows, beta=beta, rho=rho,
                n_intervals=int(idx.size), n_unit_intervals=int(good.sum()),
                median_submerged_units=median_units)


def _q2_analysis(cfg: dict, outdir: Path, complete: list[str],
                 draws: np.ndarray) -> tuple[dict, list[dict]]:
    seeds = [int(v) for v in cfg["common"]["seeds"]]
    rows, result = [], {}
    for arm in cfg["gate_dose"]["q2_arms"]:
        if arm not in complete:
            result[arm] = dict(status=NUMERIC_DIVERGENCE)
            rows.append(dict(row_type="arm_summary", arm=arm, seed="", bin="",
                             status=NUMERIC_DIVERGENCE))
            continue
        per_seed = []
        for seed in seeds:
            got = _interval_rows(cfg, outdir, arm, seed)
            per_seed.append(got)
            rows.append(dict(row_type="seed_summary", arm=arm, seed=seed, bin="",
                             status=got["status"], beta_seed=got["beta"],
                             rho_seed=got["rho"], n_intervals=got["n_intervals"],
                             n_submerged_unit_intervals=got["n_unit_intervals"],
                             median_submerged_units=got["median_submerged_units"]))
            for row in got["bins"]:
                rows.append(dict(row_type="bin", arm=arm, seed=seed,
                                 status=got["status"], beta_seed=got["beta"],
                                 rho_seed=got["rho"], n_intervals=got["n_intervals"],
                                 n_submerged_unit_intervals=got["n_unit_intervals"],
                                 median_submerged_units=got["median_submerged_units"],
                                 **row))
        beta = np.asarray([v["beta"] for v in per_seed], dtype=np.float64)
        rho = np.asarray([v["rho"] for v in per_seed], dtype=np.float64)
        if not (np.isfinite(beta).all() and np.isfinite(rho).all()):
            result[arm] = dict(status="INSUFFICIENT_DATA",
                               beta_seed_values=beta.tolist(), rho_seed_values=rho.tolist())
            continue
        ci = _ci(cfg, beta, draws)
        expected = (float(cfg["gate_dose"]["q2_scaling_expected_slope_elu"])
                    if arm in ELU_ARMS else
                    float(cfg["gate_dose"]["q2_scaling_expected_slope_leaky"]))
        contains = ci["percentile_ci_lo"] <= expected <= ci["percentile_ci_hi"]
        scaling = ("MOBILITY_SCALING" if arm in ELU_ARMS and contains else
                   "CONSTANT_MOBILITY" if arm in LEAKY_ARMS and contains else
                   "SCALING_MISMATCH")
        rho_point = float(np.median(rho))
        drift = float(cfg["gate_dose"]["q2_drift_ratio_drift_dominated"])
        noise = float(cfg["gate_dose"]["q2_drift_ratio_noise_dominated"])
        drift_label = ("DRIFT_DOMINATED_DOWNWARD" if rho_point <= -drift else
                       "DRIFT_DOMINATED_UPWARD" if rho_point >= drift else
                       "NOISE_DOMINATED" if abs(rho_point) < noise else "MIXED")
        result[arm] = dict(status="OK", beta_seed_values=beta.tolist(),
                           rho_seed_values=rho.tolist(), beta_ci=ci,
                           expected_beta=expected, scaling_label=scaling,
                           rho_median=rho_point, drift_label=drift_label)
    # ``write_csv`` fixes the schema from the first row.  Seed summaries and
    # bin rows intentionally have different payloads, so normalize the union
    # here instead of relying on row order.
    fields = list(dict.fromkeys(key for row in rows for key in row))
    normalized = [{key: row.get(key, "") for key in fields} for row in rows]
    return result, normalized


def analyze(cfg: dict, outdir: Path, sanity: dict, elapsed: dict,
            divergences: dict[str, dict]) -> dict:
    seeds = [int(v) for v in cfg["common"]["seeds"]]
    P, G = cfg["phase1"], cfg["gate_dose"]
    rng = np.random.default_rng(int(P["bootstrap_seed"]))
    draws = rng.integers(0, len(seeds), size=(int(P["bootstrap_B"]), len(seeds)))
    complete = [arm for arm in ARM_ORDER if arm not in divergences]
    data = {arm: _load_arm(cfg, outdir, arm) for arm in complete}
    windows = {arm: {
        "early": _window(data[arm], cfg, list(P["early_tasks"])),
        "1M": _window(data[arm], cfg, list(P["window_1m_tasks"])),
        "5M": _window(data[arm], cfg, list(P["late_tasks_5m"])),
    } for arm in complete}
    threshold = float(P["onset_threshold"])
    onset = {window: {arm: int(np.sum(windows[arm][window]["raw_u"] >= threshold))
                      for arm in complete} for window in ("1M", "5M")}
    expected_relu = int(G["relu_expected_onset_5m"])
    bad_relu = {arm: onset["5M"].get(arm) for arm in RELU_ARMS
                if onset["5M"].get(arm) != expected_relu}
    if bad_relu:
        raise RuntimeError(f"ReLU 5M onset differs from S0' expectation: {bad_relu}")
    main_verdict = _gate_verdict(onset["5M"], set(divergences))

    contrasts = {}
    for kind, pairs in (("P3", G["p3_contrasts"]), ("P4", G["p4_contrasts"])):
        for high, low in pairs:
            label = f"{high}_minus_{low}"
            if high not in complete or low not in complete:
                contrasts[label] = dict(kind=kind, high=high, low=low,
                                        status=NUMERIC_DIVERGENCE)
                continue
            values = windows[high]["5M"]["log_u"] - windows[low]["5M"]["log_u"]
            contrasts[label] = dict(kind=kind, high=high, low=low, status="OK",
                                    seed_values=values.tolist(), ci=_ci(cfg, values, draws))

    jumps = {}
    for activation, low, high in (("relu", "R_933", "R_1216"),
                                  ("elu", "E_933", "E_1216"),
                                  ("leaky", "LR_933", "LR_1216")):
        if low in complete and high in complete:
            delta = windows[high]["5M"]["log_u"] - windows[low]["5M"]["log_u"]
            jumps[activation] = float(abs(np.median(delta)))
        else:
            jumps[activation] = float("nan")

    q2, increment_rows = _q2_analysis(cfg, outdir, complete, draws)
    write_csv(outdir / "increments.csv", increment_rows)

    response_rows, verdict_rows = [], []
    p3_by_high = {v["high"]: v for v in contrasts.values() if v["kind"] == "P3"}
    p4_by_high = {v["high"]: v for v in contrasts.values() if v["kind"] == "P4"}
    for arm in ARM_ORDER:
        arm_cfg = _arm(cfg, arm)
        label = REGISTERED_ARMS[arm][0]
        if arm in complete:
            ws = windows[arm]
            cp1 = clopper_pearson(onset["1M"][arm], len(seeds))
            cp5 = clopper_pearson(onset["5M"][arm], len(seeds))
            submerged_seed = ws["5M"]["metrics"]["layer1_submerged"] / 100.0
            strict_seed = ws["5M"]["metrics"]["layer1_strict_dead"] / 100.0
            row = dict(
                arm=arm, activation=label,
                target_mu_norm="" if _target(arm_cfg) is None else _target(arm_cfg),
                target_dose="" if arm_cfg.get("target_dose") is None
                else float(arm_cfg["target_dose"]), status="COMPLETE",
                n_onset_1m=onset["1M"][arm], cp95_1m_lo=cp1[0], cp95_1m_hi=cp1[1],
                U_1m_seed_values=json.dumps(ws["1M"]["u"].tolist()),
                median_log10_U_1m=float(np.median(ws["1M"]["log_u"])),
                n_onset_5m=onset["5M"][arm], cp95_5m_lo=cp5[0], cp95_5m_hi=cp5[1],
                U_5m_seed_values=json.dumps(ws["5M"]["u"].tolist()),
                median_log10_U_5m=float(np.median(ws["5M"]["log_u"])),
                median_submerged_frac_5m=float(np.median(submerged_seed)),
                median_strict_dead_frac_5m=float(np.median(strict_seed)),
                submerged_near_zero=int(np.median(submerged_seed)
                                        < float(G["submerged_near_zero_threshold"])),
                NUMERIC_DIVERGENCE=0, main_verdict=main_verdict,
                jump_J_5m=jumps[label])
        else:
            row = dict(arm=arm, activation=label,
                       target_mu_norm="" if _target(arm_cfg) is None else _target(arm_cfg),
                       target_dose="" if arm_cfg.get("target_dose") is None
                       else float(arm_cfg["target_dose"]),
                       status=NUMERIC_DIVERGENCE, n_onset_1m="", cp95_1m_lo="",
                       cp95_1m_hi="", U_1m_seed_values="", median_log10_U_1m="",
                       n_onset_5m="", cp95_5m_lo="", cp95_5m_hi="",
                       U_5m_seed_values="", median_log10_U_5m="",
                       median_submerged_frac_5m="", median_strict_dead_frac_5m="",
                       submerged_near_zero="", NUMERIC_DIVERGENCE=1,
                       main_verdict=main_verdict, jump_J_5m="")
        for prefix, contrast in (("P3", p3_by_high.get(arm)),
                                 ("P4", p4_by_high.get(arm))):
            row[f"{prefix}_contrast"] = "" if contrast is None else f"{contrast['high']}_minus_{contrast['low']}"
            row[f"{prefix}_status"] = "" if contrast is None else contrast["status"]
            ci = None if contrast is None else contrast.get("ci")
            for key in ("point", "percentile_ci_lo", "percentile_ci_hi",
                        "studentized_ci_lo", "studentized_ci_hi", "ci_degenerate"):
                row[f"{prefix}_{key}"] = "" if ci is None else ci[key]
        row["CI_DEGENERATE"] = int(any(
            str(row.get(f"{prefix}_ci_degenerate", "")) == "1" for prefix in ("P3", "P4")))
        verdict_rows.append(row)

        if arm in complete:
            for window_name in ("early", "1M", "5M"):
                w = windows[arm][window_name]
                gamma_values = w["metrics"]["gamma"]
                finite_gamma = gamma_values[np.isfinite(gamma_values)]
                response_rows.append(dict(
                    arm=arm, activation=label, target_mu_norm=row["target_mu_norm"],
                    target_dose=row["target_dose"], window=window_name, status="COMPLETE",
                    n_onset="" if window_name == "early" else onset[window_name][arm],
                    median_log10_U=float(np.median(w["log_u"])),
                    median_U=float(np.median(w["u"])),
                    median_eval_loss_exact=float(np.median(w["metrics"]["eval_loss_exact"])),
                    median_strict_dead=float(np.median(w["metrics"]["layer1_strict_dead"])),
                    median_submerged_frac=float(np.median(w["metrics"]["layer1_submerged"] / 100.0)),
                    median_alive=float(np.median(w["metrics"]["layer1_alive"])),
                    median_eff_rank=float(np.median(w["metrics"]["layer1_eff_rank"])),
                    median_w_norm=float(np.median(w["metrics"]["layer1_w_norm_median"])),
                    median_M=float(np.median(w["metrics"]["layer1_median_M"])),
                    median_B=float(np.median(w["metrics"]["layer1_median_B"])),
                    median_preact_sd=float(np.median(w["metrics"]["layer1_preact_sd_median"])),
                    median_gamma=(float(np.median(finite_gamma))
                                  if finite_gamma.size else ""),
                    median_mu_cos_off=float(np.median(w["metrics"]["mu_cos_off"])),
                    jump_J_5m=jumps[label]))
        else:
            for window_name in ("early", "1M", "5M"):
                response_rows.append(dict(arm=arm, activation=label,
                                          target_mu_norm=row["target_mu_norm"],
                                          target_dose=row["target_dose"],
                                          window=window_name,
                                          status=NUMERIC_DIVERGENCE))
    write_csv(outdir / "verdict.csv", verdict_rows)
    write_csv(outdir / "dose_response.csv", response_rows)

    all_near_zero = bool(all(
        row["status"] == "COMPLETE" and row["median_submerged_frac_5m"] <
        float(G["submerged_near_zero_threshold"])
        for row in verdict_rows if row["arm"] in ELU_ARMS + LEAKY_ARMS))
    _write_summary(cfg, outdir, main_verdict, verdict_rows, contrasts, q2,
                   jumps, sanity, divergences, all_near_zero)
    return dict(main_verdict=main_verdict, onset=onset, contrasts=contrasts,
                jump_J=jumps, q2=q2, all_nonrelu_submerged_near_zero=all_near_zero,
                divergences=divergences, elapsed_sec=elapsed)


def _write_summary(cfg: dict, outdir: Path, verdict: str, rows: list[dict],
                   contrasts: dict, q2: dict, jumps: dict, sanity: dict,
                   divergences: dict, all_near_zero: bool) -> None:
    lines = [f"# {EXPERIMENT} summary", "", "## Verdict", "",
             f"- Main: **{verdict}**",
             f"- Numeric divergence: {', '.join(sorted(divergences)) or 'none'}",
             "- Claim strength: observed through 5M steps only; 0/10 one-sided 95% upper bound is 0.2589.",
             "- Pairing removes init/teacher/input-realization variance; activation trajectories diverge after step 1.",
             "", "## Endpoints", "",
             "| arm | act | onset 1M | onset 5M | median log10 U 1M | median log10 U 5M | submerged frac 5M |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        if row["status"] != "COMPLETE":
            lines.append(f"| {row['arm']} | {row['activation']} | — | — | — | — | {row['status']} |")
        else:
            lines.append(f"| {row['arm']} | {row['activation']} | {row['n_onset_1m']}/10 | "
                         f"{row['n_onset_5m']}/10 | {row['median_log10_U_1m']:.6g} | "
                         f"{row['median_log10_U_5m']:.6g} | "
                         f"{row['median_submerged_frac_5m']:.6g} |")
    lines += ["", f"All six non-ReLU arms submerged < 0.05: **{all_near_zero}**.",
              "Submergence and strict-dead counts are REPORT_ONLY and were not used in the verdict.",
              "", "## P3/P4 paired level contrasts at 5M", "",
              "| endpoint | contrast | median delta log10 U | percentile 95% CI | studentized 95% CI | degenerate |",
              "|---|---|---:|---:|---:|---:|"]
    for label, value in contrasts.items():
        if value["status"] != "OK":
            lines.append(f"| {value['kind']} | {label} | — | {value['status']} | — | — |")
            continue
        ci = value["ci"]
        lines.append(f"| {value['kind']} | {label} | {ci['point']:.6g} | "
                     f"[{ci['percentile_ci_lo']:.6g}, {ci['percentile_ci_hi']:.6g}] | "
                     f"[{ci['studentized_ci_lo']:.6g}, {ci['studentized_ci_hi']:.6g}] | "
                     f"{ci['ci_degenerate']} |")
    lines += ["", "Jump J over the two fixed in-band doses (9.33 -> 12.16): "
              + ", ".join(f"{k}={v:.6g}" if np.isfinite(v) else f"{k}=—"
                           for k, v in jumps.items()), "", "## Q2 (REPORT_ONLY)", "",
              "| arm | status | beta | beta percentile 95% CI | scaling | rho | drift/noise |",
              "|---|---|---:|---:|---|---:|---|"]
    for arm in ELU_ARMS + LEAKY_ARMS:
        value = q2[arm]
        if value["status"] != "OK":
            lines.append(f"| {arm} | {value['status']} | — | — | — | — | — |")
        else:
            ci = value["beta_ci"]
            lines.append(f"| {arm} | OK | {ci['point']:.6g} | "
                         f"[{ci['percentile_ci_lo']:.6g}, {ci['percentile_ci_hi']:.6g}] | "
                         f"{value['scaling_label']} | {value['rho_median']:.6g} | "
                         f"{value['drift_label']} |")
    lines += ["", "## Sanity", ""]
    for key in ("S0prime", "S_pair", "S_pair_final", "S_dose", "S_dose_final",
                "S_grad", "S_elu_limit", "S_leaky_limit", "S_submerge",
                "S_tautology", "S6_floor_inherited"):
        value = sanity.get(key, {})
        lines.append(f"- {key}: **{'PASS' if value.get('pass_') else 'FAIL'}**")
    (outdir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _provenance(cfg_path: Path, cfg: dict, outdir: Path, sanity: dict,
                analysis_result: dict, elapsed: dict, started: float) -> dict:
    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                           text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--short"], cwd=ROOT,
                                        text=True).splitlines()
    except (OSError, subprocess.CalledProcessError):
        git_hash, dirty = None, []
    names = ("verdict.csv", "summary.md", "dose_response.csv", "increments.csv",
             "config_used.yaml", "s0prime.json")
    hashes = {name: _sha_file(outdir / name) for name in names
              if (outdir / name).exists()}
    hashes.update({f"logs/{p.name}": _sha_file(p)
                   for p in sorted((outdir / "logs").glob("*.npz"))})
    hashes.update({f"arm_status/{p.name}": _sha_file(p)
                   for p in sorted((outdir / "arm_status").glob("*.json"))})
    reference = (Path(ROOT) / cfg["sanity"]["s0_prime_baseline_ref"]).resolve()
    return dict(experiment=EXPERIMENT,
                created=time.strftime("%Y-%m-%d %H:%M:%S %z"), command=sys.argv,
                elapsed_sec=round(time.time() - started, 3), arm_elapsed_sec=elapsed,
                cwd=os.getcwd(), python=sys.version, platform=platform.platform(),
                torch=torch.__version__, numpy=np.__version__,
                device=cfg["common"]["device"], git_hash=git_hash, git_dirty=dirty,
                config=str(cfg_path), config_sha256=_sha_file(cfg_path),
                spec=str(Path(ROOT) / cfg["spec"]),
                spec_sha256=_sha_file(Path(ROOT) / cfg["spec"]),
                baseline_reference=str(reference), sanity=sanity,
                analysis=analysis_result, output_sha256=hashes)


def _collect_divergences(cfg: dict, outdir: Path) -> dict[str, dict]:
    seeds = [int(v) for v in cfg["common"]["seeds"]]
    total, every = int(cfg["common"]["total_steps"]), int(cfg["common"]["lop_every"])
    return {arm: event for arm in ARM_ORDER
            if (event := _load_divergence_status(outdir, arm, seeds, total, every))
            is not None}


def run_full(cfg_path: Path, cfg: dict, device: str, outdir: Path, *,
             smoke: bool) -> dict:
    started = time.time()
    outdir.mkdir(parents=True, exist_ok=True)
    require_omp(cfg)
    total = SMOKE_STEPS if smoke else int(cfg["common"]["total_steps"])
    seeds = [0] if smoke else [int(v) for v in cfg["common"]["seeds"]]
    with (outdir / "config_used.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, allow_unicode=True, sort_keys=False)
    if smoke:
        preflight_result = {"pass_": True}
        s0 = {"pass_": True, "smoke": True}
    else:
        preflight_path = Path(ROOT) / "results/_preflight_gate_dose_0830/preflight.json"
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
        floor_reference = (Path(ROOT) / cfg["sanity"]["s0_prime_baseline_ref"]
                           / "floor_calibration.csv")
        shutil.copy2(floor_reference, outdir / "floor_calibration.csv")
    elapsed, identities, divergences = {}, {}, {}
    for arm in ARM_ORDER:
        if not smoke:
            existing_divergence = _load_divergence_status(
                outdir, arm, seeds, total, int(cfg["common"]["lop_every"]))
            if existing_divergence is not None:
                divergences[arm] = existing_divergence
                elapsed[arm] = 0.0
                print(f"[{arm}] saved {NUMERIC_DIVERGENCE}; resume", flush=True)
                continue
        if _complete_arm_logs(outdir, arm, seeds, total,
                              int(cfg["common"]["lop_every"])):
            elapsed[arm] = 0.0
            identities[arm] = dict(pass_=True, resumed_from_logs=True)
            print(f"[{arm}] complete logs found; resume", flush=True)
            continue
        result = _run_arm(cfg, arm, device, outdir, seeds, total)
        elapsed[arm] = result["elapsed_sec"]
        identities[arm] = result["sanity"]
        if result["status"] == NUMERIC_DIVERGENCE:
            divergences[arm] = result["divergence"]
    complete = [arm for arm in ARM_ORDER if arm not in divergences]
    if smoke:
        payload = dict(pass_=bool(all(v.get("pass_") for v in identities.values())),
                       identities=identities, divergences=divergences,
                       elapsed_sec=elapsed)
        (outdir / "smoke_sanity.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        print(f"SMOKE DONE -> {outdir}", flush=True)
        return payload
    sanity = dict(
        S0prime=s0, S_pair=preflight_result["S_pair"],
        S_pair_final=_pair_check_final(cfg, outdir, complete, divergences),
        S_dose=preflight_result["S_dose"],
        S_dose_final=_dose_check_final(cfg, outdir, complete),
        S_grad=preflight_result["S_grad"],
        S_elu_limit=preflight_result["S_elu_limit"],
        S_leaky_limit=preflight_result["S_leaky_limit"],
        S_submerge=preflight_result["S_submerge"],
        S_tautology=preflight_result["S_tautology"],
        S6_floor_inherited=preflight_result["S6_floor_inherited"],
        exact_recorders=identities)
    if not sanity["S_pair_final"]["pass_"] or not sanity["S_dose_final"]["pass_"]:
        raise RuntimeError("final pairing/dose sanity failed; analysis is blocked")
    result = analyze(cfg, outdir, sanity, elapsed, divergences)
    provenance = _provenance(cfg_path, cfg, outdir, sanity, result, elapsed, started)
    (outdir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"ALL DONE -> {outdir}", flush=True)
    return dict(sanity=sanity, analysis=result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gate_dose_0830.yaml")
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
        raise ValueError("gate_dose is CPU-only")
    stage = ("preflight" if args.preflight else "s0prime" if args.s0prime else
             "smoke" if args.smoke else "analyze" if args.analyze_only else "full")
    validate_config(cfg, stage=stage)
    main_dir = Path(ROOT) / cfg["output"]["dir"]
    outdir = (Path(args.outdir).resolve() if args.outdir else
              Path(ROOT) / "results/_preflight_gate_dose_0830" if args.preflight else
              Path(ROOT) / "results/_smoke_gate_dose_0830" if args.smoke else main_dir)
    if args.preflight:
        preflight(cfg, device, outdir)
    elif args.s0prime:
        s0prime(cfg, device, outdir)
    elif args.analyze_only:
        preflight_result = json.loads(
            (Path(ROOT) / "results/_preflight_gate_dose_0830/preflight.json")
            .read_text(encoding="utf-8"))
        s0 = json.loads((outdir / "s0prime.json").read_text(encoding="utf-8"))
        divergences = _collect_divergences(cfg, outdir)
        complete = [arm for arm in ARM_ORDER if arm not in divergences]
        sanity = dict(
            S0prime=s0, S_pair=preflight_result["S_pair"],
            S_pair_final=_pair_check_final(cfg, outdir, complete, divergences),
            S_dose=preflight_result["S_dose"],
            S_dose_final=_dose_check_final(cfg, outdir, complete),
            S_grad=preflight_result["S_grad"],
            S_elu_limit=preflight_result["S_elu_limit"],
            S_leaky_limit=preflight_result["S_leaky_limit"],
            S_submerge=preflight_result["S_submerge"],
            S_tautology=preflight_result["S_tautology"],
            S6_floor_inherited=preflight_result["S6_floor_inherited"])
        analyze(cfg, outdir, sanity, {}, divergences)
    else:
        run_full(cfg_path, cfg, device, outdir, smoke=args.smoke)


if __name__ == "__main__":
    main()
