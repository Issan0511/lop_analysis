"""mlp2_phase1_0829: candidate-A (running-mean input centering) intervention.

Phase 0b established that the depth-2 width-100 net is destroyed by 5M steps in
all three arms.  This module adds the first intervention: per-layer running-mean
centering of the layer input,

    a_hat <- (1 - alpha) a_hat + alpha a,   z_i = w_i^T (a - a_hat) + tau_i

with ``a_hat`` detached and ``tau_i`` the ordinary bias.  The intervention
consumes no randomness, so arms of the same depth share init, teacher, input
stream and flip trajectory bit for bit; the contrasts are therefore paired.

Stages are explicit so the preregistered gates run before the expensive full
run::

    OMP_NUM_THREADS=1 python -m src.mlp2_phase1 --preflight   # S-copy/S-pair/S-taut/S5/S6
    OMP_NUM_THREADS=1 python -m src.mlp2_phase1 --s0prime     # L2_none == phase0b L2
    OMP_NUM_THREADS=1 python -m src.mlp2_phase1

``--s0prime`` writes the real 5M ``L2_none`` logs into the run directory, so the
full run resumes that arm instead of recomputing it.

This module is a deliberate fork of ``mlp2_phase0b`` rather than an edit of it:
frozen experiment modules stay frozen (the repository convention behind
``bias_margin``/``bias_margin_p1`` and friends).  ``S-copy`` guards the fork by
checking that the Phase 1 exact recorder is bit-identical to the Phase 0 one
whenever no layer is centered.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
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
from .mlp2_phase0 import (
    LOG_LAYER_KEYS,
    LOG_UNIT_KEYS,
    PhaseRecorder,
    _effective_rank,
    _max_relative,
    _sha_array,
    _sha_file,
    _seed_state_hashes,
    exact_layer_record,
    identity_sanity_pass,
    require_omp,
    setup_arm,
    spearman,
    write_csv,
)
from .mlp2_phase0b import _ci_components, _complete_arm_logs, _window_indices
from .ratchet_log import full_support_ro, teacher_f64


ARM_ORDER = ("L2_none", "L2_A1", "L2_A2", "L2_Aall", "L1w100_A1")
BASELINE_ARM = "L2_none"
SMOKE_STEPS = 30_000
P1_ALIGNMENT_KEYS = ("wcos_mean", "stable_rank_W", "top1_frac",
                     "sign_match_mean", "sign_clone_frac")
P1_LOG_LAYER_KEYS = LOG_LAYER_KEYS + P1_ALIGNMENT_KEYS
# Columns that legitimately differ between L2_none and phase0b's L2: they name
# the arm, not the measurement.  Everything else must match bit for bit (S0').
S0PRIME_META_KEYS = {"run_id", "arm"}
REGISTERED_ARMS = {
    "L2_none": ([100, 100], []),
    "L2_A1": ([100, 100], [1]),
    "L2_A2": ([100, 100], [2]),
    "L2_Aall": ([100, 100], [1, 2]),
    "L1w100_A1": ([100], [1]),
}


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------
def _base_cfg(cfg: dict) -> dict:
    """Adapt Phase 1's section name to the unchanged Phase 0 helpers."""
    out = copy.deepcopy(cfg)
    out["phase0"] = copy.deepcopy(cfg["phase1"])
    out["condA"]["center_alpha"] = float(cfg["intervention"]["center_alpha"])
    return out


def _arm(cfg: dict, name: str) -> dict:
    return next(a for a in cfg["arms"] if a["name"] == name)


def _centered_flags(arm_cfg: dict, depth: int) -> list[bool]:
    flags = [False] * depth
    for layer in arm_cfg.get("centered_layers") or []:
        index = int(layer) - 1
        if not 0 <= index < depth:
            raise ValueError(f"centered layer {layer} outside depth {depth}")
        flags[index] = True
    return flags


def _is_power_of_ten(value: float) -> bool:
    if not np.isfinite(value) or value <= 0:
        return False
    exponent = math.log10(value)
    return abs(exponent - round(exponent)) <= 1e-12


def validate_config(cfg: dict, *, stage: str) -> None:
    """Validate the registered design, allowing an uncalibrated floor only in S6.

    ``stage`` is one of ``preflight``, ``smoke``, ``s0prime``, ``full`` or
    ``analyze``.  The expensive/data-producing stages refuse the
    ``CALIBRATED`` placeholder so the floor cannot silently fall back to the
    Phase 0b value.
    """
    if stage not in {"preflight", "smoke", "s0prime", "full", "analyze"}:
        raise ValueError(f"unknown validation stage {stage!r}")
    C, A, I, P = (cfg["common"], cfg["condA"], cfg["intervention"],
                  cfg["phase1"])
    names = [a["name"] for a in cfg["arms"]]
    if names != list(ARM_ORDER):
        raise ValueError(f"arms must be {ARM_ORDER}, got {names}")
    for arm in cfg["arms"]:
        hidden, centered = REGISTERED_ARMS[arm["name"]]
        if [int(v) for v in arm["hidden"]] != hidden:
            raise ValueError(f"{arm['name']} hidden differs from the preregistration")
        if [int(v) for v in (arm.get("centered_layers") or [])] != centered:
            raise ValueError(f"{arm['name']} centering differs from the preregistration")
    if int(A["m"]) != 20 or int(A["f"]) != 15:
        raise ValueError("Phase 1 requires condA m=20, f=15")
    if list(A["T_values"]) != [10_000] or list(A["encodings"]) != ["std"]:
        raise ValueError("Phase 1 requires T=10000 and std encoding")
    if (str(I["name"]) != "A_layer_input_centering"
            or float(I["center_alpha"]) != 0.01
            or I["stop_gradient_on_running_mean"] is not True
            or I["consumes_rng"] is not False):
        raise ValueError("Phase 1 requires the existing center_alpha=0.01")
    if int(P["exact_support"]) != 2 ** (int(A["m"]) - int(A["f"])):
        raise ValueError("phase1.exact_support does not match full support")
    if str(P["ci_method"]) != "studentized_paired":
        raise ValueError("Phase 1 CI must be the paired studentized interval")
    if int(P["bootstrap_B"]) != 10_000 or int(P["bootstrap_seed"]) != 20_260_829:
        raise ValueError("Phase 1 registers B=10000 and rng seed 20260829")
    floor = P["unfit_floor"]
    if isinstance(floor, str):
        if floor != "CALIBRATED" or stage not in {"preflight", "smoke"}:
            raise ValueError("run S6 to replace unfit_floor: CALIBRATED before this stage")
    elif not _is_power_of_ten(float(floor)):
        raise ValueError("unfit_floor must be the positive power of ten produced by S6")
    if float(P["censor_frac_max"]) != 0.20:
        raise ValueError("Phase 1 registers the 20% floor-censoring threshold")
    if list(P["late_tasks"]) != [451, 500] or list(P["early_tasks"]) != [2, 11]:
        raise ValueError("Phase 1 registers early 2..11 and late 451..500")
    if list(P["trend_range_tasks"]) != [2, 500]:
        raise ValueError("Phase 1 trend range must be tasks 2..500")
    if tuple(P["alignment_metrics"]) != (
            "wcos_mean", "eff_rank_W", "stable_rank_W", "top1_frac",
            "sign_match_mean", "sign_clone_frac"):
        raise ValueError("alignment_metrics differ from the preregistration")
    calibration = P["floor_calibration"]
    expected_calibration = dict(steps=200_000, arm=BASELINE_ARM, n_checkpoints=20,
                                method="two_summation_orders", percentile=99,
                                safety_factor=10, round_to="power_of_10")
    if calibration != expected_calibration:
        raise ValueError("floor_calibration differs from the preregistration")
    paired_groups = cfg["pairing"]["paired_groups"]
    if paired_groups != [["L2_none", "L2_A1", "L2_A2", "L2_Aall"]]:
        raise ValueError("paired_groups differ from the preregistration")
    if list(cfg["pairing"]["unpaired"]) != ["L1w100_A1"]:
        raise ValueError("L1w100_A1 must remain unpaired with the L2 group")
    if str(cfg["pairing"]["baseline"]) != BASELINE_ARM:
        raise ValueError("Phase 1 baseline must be L2_none")
    if stage in {"s0prime", "full", "analyze"}:
        if int(C["total_steps"]) != 5_000_000 or list(C["seeds"]) != list(range(10)):
            raise ValueError("full Phase 1 requires 5M steps and seeds 0..9")
        if str(C["device"]) != "cpu":
            raise ValueError("full Phase 1 is CPU-only")


# --------------------------------------------------------------------------
# candidate A: per-layer running-mean centering
# --------------------------------------------------------------------------
def setup_arm_p1(cfg: dict, arm_cfg: dict, device: str) -> dict:
    """Phase 0 arm state plus the per-layer centering means.

    Layer 1's mean *is* ``st["running_mean"]`` and is updated for every arm,
    exactly as Phase 0b's ``train_arm`` does, so an uncentered arm keeps the
    Phase 0b state hash.  Deeper means are allocated only where A is placed.
    """
    st = setup_arm(_base_cfg(cfg), arm_cfg, device)
    hidden = st["hidden"]
    flags = _centered_flags(arm_cfg, len(hidden))
    means: list[torch.Tensor | None] = [None] * len(hidden)
    means[0] = st["running_mean"]
    for li in range(1, len(hidden)):
        if flags[li]:
            means[li] = torch.zeros(st["R"], hidden[li - 1], device=device)
    st["centered_layers"] = flags
    st["layer_means"] = means
    st["center_alpha"] = float(cfg["intervention"]["center_alpha"])
    st["sign_match_tau"] = float(cfg["common"]["sign_match_tau"])
    return st


def forward_centered(st: dict, x: torch.Tensor):
    """One online forward pass, updating each allocated running mean.

    Mirrors ``train.py``'s ``enc=centered``: the input is centered with the
    pre-update mean, and the mean is then advanced by the *raw* layer input.
    When no layer is centered every operation is the one Phase 0b performs.
    """
    net, alpha = st["net"], st["center_alpha"]
    flags, means = st["centered_layers"], st["layer_means"]
    inputs, pres, acts = [], [], []
    cur = x
    for li, (W, b) in enumerate(zip(net.Ws, net.bs)):
        mean = means[li]
        cur_in = cur - mean if flags[li] else cur
        if mean is not None:
            mean.mul_(1.0 - alpha).add_(alpha * cur)
        pre = torch.einsum("rhd,rd->rh", W, cur_in) + b
        cur = torch.relu(pre)
        inputs.append(cur_in)
        pres.append(pre)
        acts.append(cur)
    yhat = (acts[-1] * net.v).sum(dim=1) + net.c
    return inputs, pres, acts, yhat


def grads_centered(net, inputs: list[torch.Tensor], pres: list[torch.Tensor],
                   acts: list[torch.Tensor], delta: torch.Tensor):
    """Closed-form gradients through the centered inputs (no autograd).

    ``a_hat`` is detached, so ``dz/da`` is unchanged and only the weight
    gradient sees the centered input.  This is ``nets.VecMLPL.grads_layers``
    with ``inputs`` substituted for the raw layer inputs.
    """
    d2 = 2.0 * delta
    gv = d2[:, None] * acts[-1]
    gc = d2
    dz = d2[:, None] * net.v * (pres[-1] > 0).float()
    gWs: list[torch.Tensor | None] = [None] * net.L
    gbs: list[torch.Tensor | None] = [None] * net.L
    for layer in range(net.L - 1, -1, -1):
        gbs[layer] = dz
        gWs[layer] = dz[:, :, None] * inputs[layer][:, None, :]
        if layer:
            dz = (torch.einsum("rhi,rh->ri", net.Ws[layer], dz)
                  * (pres[layer - 1] > 0).float())
    return gWs, gbs, gv, gc


def save_checkpoint_p1(st: dict, arm: str, step: int, outdir: Path) -> Path:
    path = outdir / "ckpts" / f"{arm}_step{step}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(step=step, arm=arm, net=st["net"].state_dict(),
                    env=st["env"].state_dict(), teacher=st["teacher"].state_dict(),
                    running_mean=st["running_mean"].clone(),
                    layer_means=[None if m is None else m.clone()
                                 for m in st["layer_means"]],
                    centered_layers=list(st["centered_layers"]),
                    runs=st["runs"]), path)
    return path


def train_arm_p1(st: dict, recorder, probe_steps, total: int, outdir: Path,
                 checkpoints, stream_hook=None) -> float:
    probe_set = {int(v) for v in probe_steps}
    checkpoint_set = {int(v) for v in checkpoints}
    net, env, teacher = st["net"], st["env"], st["teacher"]
    t0 = time.time()
    for t in range(total):
        if t in checkpoint_set:
            save_checkpoint_p1(st, st["arm"], t, outdir)
        if t in probe_set:
            recorder(st, t)
        x = env.step()
        y = teacher(x)
        if stream_hook is not None:
            stream_hook(t, x, y)
        inputs, pres, acts, yhat = forward_centered(st, x)
        grads = grads_centered(net, inputs, pres, acts, yhat - y)
        net.sgd_step_layers(st["lr"], *grads)
    if total in probe_set:
        recorder(st, total)
    if total in checkpoint_set:
        save_checkpoint_p1(st, st["arm"], total, outdir)
    return time.time() - t0


# --------------------------------------------------------------------------
# exact-support measurement with centering threaded through
# --------------------------------------------------------------------------
def _nanmax(values: torch.Tensor) -> float:
    """Largest finite magnitude, or NaN when every unit is degenerate."""
    array = np.abs(values.detach().cpu().numpy())
    return float(np.nanmax(array)) if np.isfinite(array).any() else float("nan")


def _alignment_metrics(W: torch.Tensor, sign_match_tau: float) -> dict:
    """Registered Sigma-path alignment measures for one layer."""
    width, fan_in = W.shape[1], W.shape[2]
    iu = torch.triu_indices(width, width, offset=1, device=W.device)
    Wn = W / W.norm(dim=2, keepdim=True).clamp_min(1e-300)
    gram = torch.einsum("rid,rjd->rij", Wn, Wn)
    wcos_mean = gram[:, iu[0], iu[1]].abs().mean(dim=1)

    signs = torch.sign(W)
    match = torch.einsum("rid,rjd->rij", signs, signs)
    match = (match + fan_in) / (2 * fan_in)
    pair_match = match[:, iu[0], iu[1]]
    sign_match_mean = pair_match.mean(dim=1)
    sign_clone_frac = (pair_match >= float(sign_match_tau)).double().mean(dim=1)

    singular = torch.linalg.svdvals(W)
    squared = singular.square()
    top1_frac = squared[:, 0] / squared.sum(dim=1).clamp_min(1e-300)
    stable_rank_W = squared.sum(dim=1) / squared[:, 0].clamp_min(1e-300)
    return dict(wcos_mean=wcos_mean, stable_rank_W=stable_rank_W,
                top1_frac=top1_frac, sign_match_mean=sign_match_mean,
                sign_clone_frac=sign_clone_frac)


def exact_layer_record_p1(st: dict, sigma_tol: float, *,
                          mean_source: str = "ema") -> tuple[dict, dict]:
    """``mlp2_phase0.exact_layer_record`` with the layer offsets applied.

    A centered layer really computes ``W (a - a_hat) + b``, so the layer input
    fed to the measurement is ``a - a_hat``: the S1 mean identity then checks
    the centered arithmetic, and ``M_i = w_i (mu - a_hat) / denom`` is the wall
    coordinate of the network that actually ran.

    ``mean_source="support"`` substitutes the exact-support mean for ``a_hat``.
    That is the ``alpha -> 0`` idealization the intervention targets, in which
    the mu projection - and therefore ``M`` - is identically zero.  S-taut uses
    it; nothing else may.
    """
    if mean_source not in ("ema", "support"):
        raise ValueError(f"unknown mean_source {mean_source!r}")
    flags = st.get("centered_layers") or [False] * len(st["net"].Ws)
    means = st.get("layer_means") or [None] * len(st["net"].Ws)
    with torch.no_grad():
        X = full_support_ro(st["env"]).double()
        y = teacher_f64(st["teacher"], X)
        cur = X
        layers = []
        sanity_layers = []
        taut = []

        for layer, (W0, b0) in enumerate(zip(st["net"].Ws, st["net"].bs), start=1):
            W, b = W0.double(), b0.double()
            raw_mu_norm = float(cur.mean(dim=0).norm(dim=1).max().item())
            if flags[layer - 1]:
                offset = (cur.mean(dim=0) if mean_source == "support"
                          else means[layer - 1].double())
                cur = cur - offset[None]
            mu = cur.mean(dim=0)
            centered = cur - mu[None]
            z = torch.einsum("rhd,prd->prh", W, cur) + b
            direct_mean = z.mean(dim=0)
            direct_sd = z.var(dim=0, unbiased=False).clamp_min(0).sqrt()
            wmu = torch.einsum("rhd,rd->rh", W, mu)
            formula_mean = wmu + b
            centered_proj = torch.einsum("rhd,prd->prh", W, centered)
            denom = centered_proj.square().mean(dim=0).clamp_min(0).sqrt()
            valid = denom >= float(sigma_tol)

            M = torch.full_like(denom, float("nan"))
            B = torch.full_like(denom, float("nan"))
            M[valid] = wmu[valid] / denom[valid]
            B[valid] = b[valid] / denom[valid]
            wall_direct = direct_mean[valid] / direct_sd[valid]
            wall_formula = M[valid] + B[valid]

            activation = torch.relu(z)
            p_hat = (z > 0).double().mean(dim=0)
            w_norm = W.norm(dim=2)
            mu_norm = mu.norm(dim=1)
            sigma_rms = centered.square().mean(dim=0).sum(dim=1)
            sigma_rms = (sigma_rms / cur.shape[2]).clamp_min(0).sqrt()
            dose = mu_norm / sigma_rms.clamp_min(1e-300)
            eff_rank = _effective_rank(activation.permute(1, 0, 2))
            eff_rank_W = _effective_rank(W)
            strict_dead = (p_hat == 0).sum(dim=1)
            alive = torch.full_like(strict_dead, W.shape[1]) - strict_dead
            eff_per_alive = torch.where(
                alive > 0, eff_rank / alive.double(),
                torch.full_like(eff_rank, float("nan")))

            qM = torch.nanquantile(M, torch.tensor([0.25, 0.5, 0.75],
                                                   dtype=M.dtype), dim=1)
            median_B = torch.nanquantile(B, 0.5, dim=1)
            qW = torch.quantile(w_norm, torch.tensor([0.25, 0.5, 0.75],
                                                     dtype=w_norm.dtype), dim=1)
            alignment = _alignment_metrics(
                W, float(st.get("sign_match_tau", 0.95)))
            layers.append(dict(
                M=M, B=B, denom=denom, p_hat=p_hat, w_norm=w_norm,
                median_M=qM[1], q25_M=qM[0], q75_M=qM[2], median_B=median_B,
                n_na=(~valid).sum(dim=1), mu_norm=mu_norm, sigma_rms=sigma_rms,
                dose=dose, w_norm_median=qW[1], w_norm_q25=qW[0],
                w_norm_q75=qW[2], eff_rank=eff_rank, eff_rank_W=eff_rank_W,
                strict_dead=strict_dead, alive=alive,
                eff_rank_per_alive=eff_per_alive, **alignment))

            cos_err = 0.0
            if layer == 1:
                mu_u = mu / mu_norm.clamp_min(1e-300)[:, None]
                cos = torch.einsum("rhd,rd->rh", W, mu_u) / w_norm.clamp_min(1e-300)
                cos_err = _max_relative(cos * mu_norm[:, None],
                                        wmu / w_norm.clamp_min(1e-300))
            finite_required = (torch.isfinite(z).all() and torch.isfinite(mu).all()
                               and torch.isfinite(denom).all()
                               and torch.isfinite(eff_rank).all()
                               and torch.isfinite(eff_rank_W).all())
            sanity_layers.append(dict(
                layer=layer,
                mean_max_relerr=_max_relative(direct_mean, formula_mean),
                sd_max_relerr=_max_relative(direct_sd, denom),
                wall_max_relerr=_max_relative(wall_direct, wall_formula),
                l1_cos_mu_max_relerr=cos_err,
                n_degenerate=int((~valid).sum().item()),
                finite_required=bool(finite_required)))
            if flags[layer - 1]:
                # Scale-free tautology residual: the mu projection is measured
                # against the projection scale it would have had uncentered.
                scale = float((w_norm.max() * max(raw_mu_norm, 1e-300)).item())
                taut.append(dict(layer=layer, mean_source=mean_source,
                                 mu_projection_max=float(wmu.abs().max().item()),
                                 projection_scale=scale,
                                 relative=float(wmu.abs().max().item()) / max(scale, 1e-300),
                                 abs_M_max=_nanmax(M),
                                 median_M_max=_nanmax(layers[-1]["median_M"])))
            cur = activation

        yhat = (cur * st["net"].v.double()).sum(dim=-1) + st["net"].c.double()
        residual = yhat - y
        signal_var = y.var(dim=0, unbiased=False)
        residual_var = residual.var(dim=0, unbiased=False)
        unfit = residual_var / signal_var
        run = dict(signal_var=signal_var, residual_var=residual_var, unfit=unfit,
                   eval_loss_exact=residual.square().mean(dim=0))
        run_finite = bool(all(torch.isfinite(v).all() for v in run.values())
                          and (signal_var > 0).all())
        sanity = dict(layers=sanity_layers, run_finite=run_finite,
                      support=int(X.shape[0]), taut=taut)
        return dict(run=run, layers=layers,
                    flip_state=st["env"].flip_state.double()), sanity


class PhaseRecorderP1(PhaseRecorder):
    """Phase 0b's recorder driven by the centering-aware exact record."""

    def __init__(self, steps: list[int], st: dict, sigma_tol: float,
                 identity_tol: float):
        super().__init__(steps, st, sigma_tol, identity_tol)
        n, R = len(self.steps), st["R"]
        for layer in self.layers:
            for key in P1_ALIGNMENT_KEYS:
                layer[key] = np.empty((n, R), dtype=np.float64)

    def __call__(self, st: dict, step: int) -> None:
        i = self.index.get(int(step))
        if i is None:
            return
        if self.filled[i]:
            raise RuntimeError(f"duplicate phase1 probe at step {step}")
        rec, sanity = exact_layer_record_p1(st, self.sigma_tol)
        for key, value in rec["run"].items():
            self.run[key][i] = value.detach().cpu().numpy()
        self.flip_state[i] = rec["flip_state"].detach().cpu().numpy().astype(np.float32)
        for li, layer in enumerate(rec["layers"]):
            for key in LOG_UNIT_KEYS:
                self.layers[li][key][i] = layer[key].detach().cpu().numpy().astype(np.float32)
            for key in P1_LOG_LAYER_KEYS:
                self.layers[li][key][i] = layer[key].detach().cpu().numpy()
            s, acc = sanity["layers"][li], self.max_errors[li]
            acc["mean"] = max(acc["mean"], s["mean_max_relerr"])
            acc["sd"] = max(acc["sd"], s["sd_max_relerr"])
            acc["wall"] = max(acc["wall"], s["wall_max_relerr"])
            acc["cos_mu"] = max(acc["cos_mu"], s["l1_cos_mu_max_relerr"])
            acc["n_degenerate_max"] = max(acc["n_degenerate_max"], s["n_degenerate"])
            if not s["finite_required"]:
                self.required_nonfinite += 1
        if not sanity["run_finite"]:
            self.required_nonfinite += 1
        self.filled[i] = True


def _seed_state_hashes_p1(st: dict, seed_index: int) -> dict[str, str]:
    """Phase 0 seed hashes, plus deeper centering means where A is placed.

    An uncentered arm produces exactly the Phase 0b dictionary, which is what
    S0' compares against.
    """
    out = _seed_state_hashes(st, seed_index)
    for li, mean in enumerate(st["layer_means"][1:], start=2):
        if mean is not None:
            out[f"running_mean_layer{li}"] = _sha_array(mean[seed_index])
    return out


def write_arm_logs_p1(outdir: Path, arm: str, st: dict,
                      rec: PhaseRecorder) -> list[Path]:
    logdir = outdir / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ri, run in enumerate(st["runs"]):
        payload = dict(step=rec.steps, run_id=np.array(run["run_id"]),
                       arm=np.array(arm), seed=np.int64(run["seed"]),
                       task_period=np.int64(run["period"]),
                       state_hash_final=np.array(json.dumps(
                           _seed_state_hashes_p1(st, ri), sort_keys=True)))
        for key, value in rec.run.items():
            payload[key] = value[:, ri]
        payload["flip_state"] = rec.flip_state[:, ri]
        for li, layer in enumerate(rec.layers, start=1):
            for key, value in layer.items():
                payload[f"layer{li}_{key}"] = value[:, ri]
        path = logdir / f"{arm}_seed{run['seed']}.npz"
        np.savez_compressed(path, **payload)
        paths.append(path)
    return paths


# --------------------------------------------------------------------------
# preregistered gates (spec section 4)
# --------------------------------------------------------------------------
class StreamDigest:
    """Rolling digest of the (x, y) stream fed to one arm."""

    def __init__(self) -> None:
        self.x = hashlib.sha256()
        self.y = hashlib.sha256()
        self.n = 0

    def __call__(self, t: int, x: torch.Tensor, y: torch.Tensor) -> None:
        self.x.update(np.ascontiguousarray(x.detach().cpu().numpy()).tobytes())
        self.y.update(np.ascontiguousarray(y.detach().cpu().numpy()).tobytes())
        self.n += 1

    def digest(self) -> dict:
        return dict(x=self.x.hexdigest(), y=self.y.hexdigest(), n=self.n)


def _init_hashes(st: dict) -> dict[str, str]:
    out = {f"net.{k}": _sha_array(v) for k, v in st["net"].state_dict().items()}
    out.update({f"teacher.{k}": _sha_array(v)
                for k, v in st["teacher"].state_dict().items()})
    out["env.flip_state"] = _sha_array(st["env"].flip_state)
    out["env.patterns"] = _sha_array(st["env"].patterns)
    out["env.t"] = str(st["env"].t)
    return out


def _env_hashes(st: dict) -> dict[str, str]:
    return {"env.flip_state": _sha_array(st["env"].flip_state),
            "env.t": str(st["env"].t)}


def _s_copy_check(cfg: dict, device: str, outdir: Path) -> dict:
    """The Phase 1 exact record must be the Phase 0 one when nothing is centered.

    Guards the deliberate fork of ``exact_layer_record``: a copy that silently
    drifts would invalidate S0' and every wall coordinate at once.
    """
    c = _base_cfg(cfg)
    c["common"]["seeds"] = [0, 1]
    st = setup_arm_p1(c, _arm(cfg, BASELINE_ARM), device)
    tol = float(cfg["phase1"]["sigma_degenerate_tol"])
    differences = []
    for step in (0, 2000):
        if step:
            train_arm_p1(st, lambda *_: None, [], step, outdir, [])
        new, _ = exact_layer_record_p1(st, tol)
        old, _ = exact_layer_record(st, tol)
        for key in new["run"]:
            if _sha_array(new["run"][key]) != _sha_array(old["run"][key]):
                differences.append(dict(step=step, where=f"run.{key}"))
        if _sha_array(new["flip_state"]) != _sha_array(old["flip_state"]):
            differences.append(dict(step=step, where="flip_state"))
        for li, (a, b) in enumerate(zip(new["layers"], old["layers"]), start=1):
            for key in LOG_UNIT_KEYS + LOG_LAYER_KEYS:
                if _sha_array(a[key]) != _sha_array(b[key]):
                    differences.append(dict(step=step, where=f"layer{li}.{key}"))
    return dict(pass_=not differences, differences=differences,
                steps=[0, 2000], seeds=[0, 1])


def _s_pair_check(cfg: dict, device: str, outdir: Path) -> dict:
    """S-pair / S-taut: the L2 arms must correspond, and A must kill the mu term."""
    S, P = cfg["sanity"], cfg["phase1"]
    steps = int(S["s_pair_steps"])
    arms = [str(a) for a in cfg["pairing"]["paired_groups"][0]]
    tol = float(P["sigma_degenerate_tol"])
    init, final, stream, taut, taut_ema = {}, {}, {}, {}, {}
    for name in arms:
        c = _base_cfg(cfg)
        st = setup_arm_p1(c, _arm(cfg, name), device)
        init[name] = _init_hashes(st)
        digest = StreamDigest()
        print(f"[S-pair] {name} {steps:,} steps x {len(c['common']['seeds'])} seeds",
              flush=True)
        train_arm_p1(st, lambda *_: None, [], steps, outdir, [], stream_hook=digest)
        final[name] = _env_hashes(st)
        stream[name] = digest.digest()
        _, sanity_support = exact_layer_record_p1(st, tol, mean_source="support")
        _, sanity_ema = exact_layer_record_p1(st, tol, mean_source="ema")
        taut[name] = sanity_support["taut"]
        taut_ema[name] = sanity_ema["taut"]
        del st

    reference = arms[0]
    differences = []
    for name in arms[1:]:
        for key, value in init[reference].items():
            if init[name].get(key) != value:
                differences.append(dict(arm=name, where=f"init.{key}"))
        for key, value in final[reference].items():
            if final[name].get(key) != value:
                differences.append(dict(arm=name, where=f"final.{key}"))
        for key in ("x", "y", "n"):
            if stream[name][key] != stream[reference][key]:
                differences.append(dict(arm=name, where=f"stream.{key}"))

    staut_tol = float(S["s1_identity_tol"])
    taut_rows = [dict(arm=name, **row) for name, rows in taut.items() for row in rows]
    taut_fail = [r for r in taut_rows if not (r["relative"] <= staut_tol)]
    if not taut_rows:
        taut_fail = [dict(arm="", note="no centered layer was checked")]
    return dict(pass_=bool(not differences and not taut_fail),
                spair=dict(pass_=not differences, arms=arms, steps=steps,
                           reference=reference, differences=differences,
                           init_hashes=init, final_env_hashes=final,
                           stream_digests=stream),
                staut=dict(pass_=not taut_fail, tolerance=staut_tol,
                           exact_substitution=taut_rows, failures=taut_fail,
                           ema_residual_report_only=[
                               dict(arm=name, **row)
                               for name, rows in taut_ema.items() for row in rows]))


def _s5_selftest(cfg: dict) -> dict:
    P = cfg["phase1"]
    n = len(cfg["common"]["seeds"])
    rng = np.random.default_rng(int(P["bootstrap_seed"]))
    draws = rng.integers(0, n, size=(int(P["bootstrap_B"]), n))
    paired = _ci_components(np.zeros(n), draws, "median",
                            float(P["degenerate_se_tol"]),
                            float(P["degenerate_frac_max"]),
                            float(P["degenerate_width_ratio_max"]))
    draws_a = rng.integers(0, n, size=(int(P["bootstrap_B"]), n))
    draws_b = rng.integers(0, n, size=(int(P["bootstrap_B"]), n))
    unpaired = _ci_unpaired(cfg, np.zeros(n), np.zeros(n), draws_a, draws_b)
    return dict(pass_=bool(paired["ci_degenerate"] and unpaired["ci_degenerate"]),
                paired=paired, unpaired=unpaired)


def _ordered_mean(values: torch.Tensor, *, reverse: bool) -> torch.Tensor:
    order = range(values.shape[0] - 1, -1, -1) if reverse else range(values.shape[0])
    total = torch.zeros_like(values[0])
    for index in order:
        total = total + values[index]
    return total / values.shape[0]


def _ordered_variance(values: torch.Tensor, *, reverse: bool) -> torch.Tensor:
    mean = _ordered_mean(values, reverse=reverse)
    return _ordered_mean((values - mean).square(), reverse=reverse)


def _unfit_two_summation_orders(st: dict) -> tuple[np.ndarray, np.ndarray]:
    """Exact-support unfitness with explicitly ordered float64 reductions."""
    if any(st["centered_layers"]):
        raise ValueError("floor calibration is registered only for L2_none")
    with torch.no_grad():
        X = full_support_ro(st["env"]).double()
        y = teacher_f64(st["teacher"], X)
        cur = X
        for W, b in zip(st["net"].Ws, st["net"].bs):
            cur = torch.relu(torch.einsum("rhd,prd->prh", W.double(), cur)
                             + b.double())
        yhat = (cur * st["net"].v.double()).sum(dim=-1) + st["net"].c.double()
        residual = yhat - y
        values = []
        for reverse in (False, True):
            signal = _ordered_variance(y, reverse=reverse)
            error = _ordered_variance(residual, reverse=reverse)
            values.append((error / signal).detach().cpu().numpy())
    return values[0], values[1]


def _write_calibrated_floor(cfg_path: Path, floor: float) -> None:
    text = cfg_path.read_text(encoding="utf-8")
    replacement = f"\\g<1>{floor:.1e}\\g<2>"
    updated, count = re.subn(
        r"^(\s*unfit_floor:\s*)[^#\r\n]+?(\s*(?:#.*)?)$",
        replacement, text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise RuntimeError(f"could not replace unfit_floor in {cfg_path}")
    cfg_path.write_text(updated, encoding="utf-8")


def _s6_floor_calibration(cfg_path: Path, cfg: dict, device: str,
                          outdir: Path) -> dict:
    K = cfg["phase1"]["floor_calibration"]
    total, n_checkpoints = int(K["steps"]), int(K["n_checkpoints"])
    if total % n_checkpoints:
        raise ValueError("floor calibration checkpoints must divide the run evenly")
    checkpoints = list(range(total // n_checkpoints, total + 1,
                             total // n_checkpoints))
    c = _base_cfg(cfg)
    seeds = [int(v) for v in cfg["common"]["seeds"]]
    c["common"]["seeds"] = seeds
    st = setup_arm_p1(c, _arm(cfg, str(K["arm"])), device)
    rows: list[dict] = []

    def record(state: dict, step: int) -> None:
        forward, reverse = _unfit_two_summation_orders(state)
        for ri, seed in enumerate(seeds):
            rows.append(dict(step=int(step), seed=seed,
                             unfit_forward=float(forward[ri]),
                             unfit_reverse=float(reverse[ri]),
                             abs_delta=float(abs(forward[ri] - reverse[ri]))))

    elapsed = train_arm_p1(st, record, checkpoints, total, outdir, [])
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
    result = dict(pass_=passed and _is_power_of_ten(calibrated), arm=str(K["arm"]),
                  steps=total, checkpoints=checkpoints, seeds=seeds,
                  n_values=len(rows), percentile_99=percentile,
                  safety_factor=float(K["safety_factor"]), raw_floor=raw_floor,
                  calibrated_floor=calibrated, elapsed_sec=elapsed,
                  csv=str(outdir / "floor_calibration.csv"))
    if result["pass_"]:
        cfg["phase1"]["unfit_floor"] = calibrated
        _write_calibrated_floor(cfg_path, calibrated)
    return result


def preflight(cfg_path: Path, cfg: dict, device: str, outdir: Path) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    omp = require_omp(cfg)
    print("[S-copy] phase1 exact record vs phase0 reference", flush=True)
    scopy = _s_copy_check(cfg, device, outdir / "scopy")
    print(f"[S-copy] {'PASS' if scopy['pass_'] else 'FAIL'}", flush=True)
    pair = _s_pair_check(cfg, device, outdir / "spair")
    print(f"[S-pair] {'PASS' if pair['spair']['pass_'] else 'FAIL'}  "
          f"[S-taut] {'PASS' if pair['staut']['pass_'] else 'FAIL'}", flush=True)
    s5 = _s5_selftest(cfg)
    print(f"[S5] {'PASS' if s5['pass_'] else 'FAIL'}", flush=True)
    print("[S6] L2_none 200k x 20-point two-order floor calibration", flush=True)
    s6 = _s6_floor_calibration(cfg_path, cfg, device, outdir)
    print(f"[S6] {'PASS' if s6['pass_'] else 'FAIL'} ({s6['elapsed_sec']:.1f}s)", flush=True)
    result = dict(pass_=bool(omp["pass_"] and scopy["pass_"] and pair["pass_"]
                             and s5["pass_"] and s6["pass_"]),
                  S3=omp, S_copy=scopy, S_pair=pair["spair"], S_taut=pair["staut"],
                  S5=s5, S6=s6)
    (outdir / "preflight.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    if not result["pass_"]:
        raise RuntimeError(f"preflight failed: {json.dumps({k: v.get('pass_') for k, v in result.items() if isinstance(v, dict)})}")
    return result


def _compare_arm_logs(ours: Path, theirs: Path) -> list[dict]:
    with np.load(ours, allow_pickle=False) as a, np.load(theirs, allow_pickle=False) as b:
        # Phase 1 adds registered alignment columns.  S0' requires every
        # pre-existing Phase 0b column to match, while permitting those new
        # columns to exist only on the Phase 1 side.
        keys_a = {key for key in set(a.files) - S0PRIME_META_KEYS
                  if not any(key.endswith("_" + metric)
                             for metric in P1_ALIGNMENT_KEYS)}
        keys_b = set(b.files) - S0PRIME_META_KEYS
        differences = [dict(column=k, reason="missing in phase0b") for k in sorted(keys_a - keys_b)]
        differences += [dict(column=k, reason="missing in phase1") for k in sorted(keys_b - keys_a)]
        for key in sorted(keys_a & keys_b):
            if _sha_array(a[key]) != _sha_array(b[key]):
                differences.append(dict(column=key, reason="hash mismatch"))
    return differences


def s0prime(cfg: dict, device: str, outdir: Path) -> dict:
    """S0': L2_none must reproduce Phase 0b's L2 bit for bit.

    The arm is the real 5M run and lands in ``outdir/logs``, so the full run
    resumes it rather than paying for it twice.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    require_omp(cfg)
    C, S = cfg["common"], cfg["sanity"]
    total, seeds = int(C["total_steps"]), [int(v) for v in C["seeds"]]
    reference_dir = Path(ROOT) / S["s0_prime_baseline_ref"]
    reference_arm = "L2"
    elapsed = 0.0
    if _complete_arm_logs(outdir, BASELINE_ARM, seeds, total, int(C["lop_every"])):
        print(f"[S0'] complete {BASELINE_ARM} logs found; comparing only", flush=True)
    else:
        elapsed = _run_arm(cfg, BASELINE_ARM, device, outdir, seeds, total)["elapsed_sec"]

    differences, missing = [], []
    for seed in seeds:
        theirs = reference_dir / "logs" / f"{reference_arm}_seed{seed}.npz"
        ours = outdir / "logs" / f"{BASELINE_ARM}_seed{seed}.npz"
        if not theirs.exists():
            missing.append(str(theirs))
            continue
        differences += [dict(seed=seed, **d) for d in _compare_arm_logs(ours, theirs)]

    reference_ckpt = reference_dir / "ckpts" / f"{reference_arm}_step{total}.pt"
    state_differences = []
    expected_state = actual_state = {}
    if not reference_ckpt.exists():
        missing.append(str(reference_ckpt))
    else:
        ck = torch.load(reference_ckpt, map_location="cpu", weights_only=False)
        ours_ckpt = outdir / "ckpts" / f"{BASELINE_ARM}_step{total}.pt"
        if not ours_ckpt.exists():
            missing.append(str(ours_ckpt))
        else:
            mine = torch.load(ours_ckpt, map_location="cpu", weights_only=False)
            expected_state = {f"net.{k}": _sha_array(v) for k, v in ck["net"].items()}
            expected_state.update(
                env_flip_state=_sha_array(ck["env"]["flip_state"]),
                env_t=str(ck["env"]["t"]),
                running_mean=_sha_array(ck["running_mean"]))
            actual_state = {f"net.{k}": _sha_array(v) for k, v in mine["net"].items()}
            actual_state.update(
                env_flip_state=_sha_array(mine["env"]["flip_state"]),
                env_t=str(mine["env"]["t"]),
                running_mean=_sha_array(mine["running_mean"]))
            state_differences = sorted(k for k, v in expected_state.items()
                                       if actual_state.get(k) != v)

    result = dict(pass_=bool(not differences and not missing and not state_differences),
                  arm=BASELINE_ARM, reference=str(reference_dir),
                  reference_arm=reference_arm, total_steps=total, seeds=seeds,
                  elapsed_sec=elapsed, missing=missing,
                  column_differences=differences, state_differences=state_differences,
                  expected_state_hash=expected_state, actual_state_hash=actual_state,
                  ignored_columns=sorted(S0PRIME_META_KEYS))
    (outdir / "s0prime.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"S0' {'PASS' if result['pass_'] else 'FAIL'}", flush=True)
    if not result["pass_"]:
        raise RuntimeError("S0' failed; the full run must not proceed (spec section 4)")
    return result


# --------------------------------------------------------------------------
# full run
# --------------------------------------------------------------------------
def _run_arm(cfg: dict, arm: str, device: str, outdir: Path, seeds: list[int],
             total: int) -> dict:
    C, P = cfg["common"], cfg["phase1"]
    c = _base_cfg(cfg)
    c["common"]["seeds"] = seeds
    arm_cfg = _arm(cfg, arm)
    probe_steps = list(range(0, total + 1, int(C["lop_every"])))
    if probe_steps[-1] != total:
        probe_steps.append(total)
    print(f"[{arm}] hidden={arm_cfg['hidden']} centered={arm_cfg['centered_layers']} "
          f"seeds={seeds} steps={total:,}", flush=True)
    st = setup_arm_p1(c, arm_cfg, device)
    _, before = exact_layer_record_p1(st, float(P["sigma_degenerate_tol"]))
    if not identity_sanity_pass(before, float(cfg["sanity"]["s1_identity_tol"])):
        raise RuntimeError(f"{arm} preflight identity failed")
    rec = PhaseRecorderP1(probe_steps, st, float(P["sigma_degenerate_tol"]),
                          float(cfg["sanity"]["s1_identity_tol"]))
    checkpoints = [int(v) for v in C.get("checkpoints", []) if int(v) <= total]
    elapsed = train_arm_p1(st, rec, probe_steps, total, outdir, checkpoints)
    sanity = rec.sanity()
    if not sanity["pass_"]:
        raise RuntimeError(f"{arm} S1/S2 failed: {sanity}")
    write_arm_logs_p1(outdir, arm, st, rec)
    print(f"[{arm}] complete in {elapsed:.1f}s", flush=True)
    result = dict(elapsed_sec=elapsed, sanity=sanity,
                  final_env=_env_hashes(st))
    del rec, st
    return result


def _pair_check_final(cfg: dict, outdir: Path, seeds: list[int]) -> dict:
    """After the run, the paired arms must still share the environment.

    ``state_hash_final`` carries ``env.flip_state``/``env.t`` per seed, so this
    costs nothing and covers the whole 5M horizon rather than only step 0.
    """
    def env_of(logdir: Path, arm: str, seed: int) -> dict:
        with np.load(logdir / f"{arm}_seed{seed}.npz", allow_pickle=False) as z:
            state = json.loads(str(z["state_hash_final"]))
        return {k: state[k] for k in ("env.flip_state", "env.t", "running_mean")}

    l2 = [str(a) for a in cfg["pairing"]["paired_groups"][0]]
    differences = []
    for seed in seeds:
        reference = env_of(outdir / "logs", BASELINE_ARM, seed)
        for arm in l2[1:]:
            if env_of(outdir / "logs", arm, seed) != reference:
                differences.append(dict(seed=seed, arm=arm, where="env"))
    l2_ok = not differences

    return dict(pass_=l2_ok, paired_pass=l2_ok, paired_arms=l2,
                differences=differences)


# --------------------------------------------------------------------------
# aggregation (spec section 5) - never run before the gates pass
# --------------------------------------------------------------------------
def _arm_arrays(logdir: Path, arm: str, seeds: list[int], depth: int,
                period: int) -> dict:
    per_seed = []
    for seed in seeds:
        with np.load(logdir / f"{arm}_seed{seed}.npz", allow_pickle=False) as z:
            idx = np.flatnonzero((z["step"] > 0) & (z["step"] % period == 0))
            per_seed.append({
                "steps": z["step"][idx].copy(), "unfit": z["unfit"][idx].copy(),
                "layers": [{k: z[f"layer{li}_{k}"][idx].copy()
                            for k in P1_LOG_LAYER_KEYS}
                           for li in range(1, depth + 1)]})
    result = {"steps": per_seed[0]["steps"],
              "unfit": np.stack([v["unfit"] for v in per_seed], axis=1),
              "layers": []}
    for li in range(depth):
        result["layers"].append({
            k: np.stack([v["layers"][li][k] for v in per_seed], axis=1)
            for k in P1_LOG_LAYER_KEYS})
    return result


def _ci(cfg: dict, values: np.ndarray, draws: np.ndarray) -> dict:
    P = cfg["phase1"]
    return _ci_components(np.asarray(values, dtype=np.float64), draws,
                          "median", float(P["degenerate_se_tol"]),
                          float(P["degenerate_frac_max"]),
                          float(P["degenerate_width_ratio_max"]))


def _ci_unpaired(cfg: dict, arm_values: np.ndarray, base_values: np.ndarray,
                 arm_draws: np.ndarray, base_draws: np.ndarray) -> dict:
    """Studentized two-sample bootstrap for a difference of medians."""
    P = cfg["phase1"]
    a = np.asarray(arm_values, dtype=np.float64)
    b = np.asarray(base_values, dtype=np.float64)
    if (a.ndim != 1 or b.ndim != 1 or not np.isfinite(a).all()
            or not np.isfinite(b).all()
            or arm_draws.shape[1] != len(a) or base_draws.shape[1] != len(b)
            or arm_draws.shape[0] != base_draws.shape[0]):
        raise ValueError("unpaired CI requires finite one-dimensional samples")

    def jk_se(matrix: np.ndarray) -> np.ndarray:
        n = matrix.shape[1]
        jk = np.stack([np.median(np.delete(matrix, i, axis=1), axis=1)
                       for i in range(n)], axis=1)
        return np.sqrt((n - 1) / n
                       * np.square(jk - jk.mean(axis=1, keepdims=True)).sum(axis=1))

    point = float(np.median(a) - np.median(b))
    se0 = float(math.hypot(jk_se(a[None])[0], jk_se(b[None])[0]))
    samples_a, samples_b = a[arm_draws], b[base_draws]
    boot = np.median(samples_a, axis=1) - np.median(samples_b, axis=1)
    se = np.hypot(jk_se(samples_a), jk_se(samples_b))
    se_tol = float(P["degenerate_se_tol"])
    good = np.isfinite(boot) & np.isfinite(se) & (se >= se_tol)
    degenerate_fraction = float(np.mean(~np.isfinite(se) | (se < se_tol)))
    if se0 >= se_tol and good.any():
        pivots = (boot[good] - point) / se[good]
        qlo, qhi = np.quantile(pivots, [0.025, 0.975])
        student_lo, student_hi = point - qhi * se0, point - qlo * se0
    else:
        student_lo = student_hi = point
    pct_lo, pct_hi = np.quantile(boot[np.isfinite(boot)], [0.025, 0.975])
    width = float(student_hi - student_lo)
    ratio = width / max(abs(point), se_tol)
    degenerate = bool(
        degenerate_fraction > float(P["degenerate_frac_max"])
        or ratio > float(P["degenerate_width_ratio_max"]))
    return dict(point=point, studentized_ci_lo=float(student_lo),
                studentized_ci_hi=float(student_hi), percentile_ci_lo=float(pct_lo),
                percentile_ci_hi=float(pct_hi), se0=se0,
                degenerate_se_fraction=degenerate_fraction,
                studentized_width_ratio=ratio, ci_degenerate=int(degenerate),
                boot_ok=int(good.sum()), sign_test_p=float("nan"),
                n_seed=f"{len(a)}+{len(b)}", statistic="difference_of_medians")


def _decide(ci: dict, direction: str, censored: bool, *,
            paired: bool, alpha: float = 0.05) -> tuple[bool, str]:
    """Registered decision precedence: censored -> degenerate -> studentized."""
    if censored:
        if not paired:
            return False, "censored_unpaired"
        improved = (ci["sign_test_p"] < alpha
                    and (ci["point"] < 0 if direction == "down" else ci["point"] > 0))
        return bool(improved), "sign_test"
    if ci["ci_degenerate"]:
        improved = (ci["percentile_ci_hi"] < 0 if direction == "down"
                    else ci["percentile_ci_lo"] > 0)
        return bool(improved), "percentile"
    improved = (ci["studentized_ci_hi"] < 0 if direction == "down"
                else ci["studentized_ci_lo"] > 0)
    return bool(improved), "studentized"


def _floor_fracs(arrays: dict, late_i: np.ndarray, floor: float) -> dict:
    unfit = np.asarray(arrays["unfit"], dtype=np.float64)
    return dict(floor_frac_late=float(np.mean(unfit[late_i] <= floor)),
                floor_frac_all=float(np.mean(unfit <= floor)),
                floor_frac_late_seed_level=float(
                    np.mean(unfit[late_i].mean(axis=0) <= floor)))


def _contrast(cfg: dict, arm: str, arm_data: dict, base_label: str,
              base_data: dict, paired_draws: np.ndarray,
              unpaired_draws: tuple[np.ndarray, np.ndarray], *,
              paired: bool, pair_ok: bool, contrast_type: str) -> dict:
    P = cfg["phase1"]
    period, floor = int(P["task_period"]), float(P["unfit_floor"])
    late_a = _window_indices(arm_data["steps"], period, list(P["late_tasks"]))
    late_b = _window_indices(base_data["steps"], period, list(P["late_tasks"]))
    early_a = _window_indices(arm_data["steps"], period, list(P["early_tasks"]))
    early_b = _window_indices(base_data["steps"], period, list(P["early_tasks"]))

    def compare(a: np.ndarray, b: np.ndarray) -> tuple[dict, np.ndarray]:
        if paired:
            delta = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
            return _ci(cfg, delta, paired_draws), delta
        return (_ci_unpaired(cfg, a, b, unpaired_draws[0], unpaired_draws[1]),
                np.empty(0, dtype=np.float64))

    u_a = np.maximum(np.asarray(arm_data["unfit"])[late_a].mean(axis=0), floor)
    u_b = np.maximum(np.asarray(base_data["unfit"])[late_b].mean(axis=0), floor)
    log_u_a, log_u_b = np.log10(u_a), np.log10(u_b)
    ci_p1, d_p1 = compare(log_u_a, log_u_b)

    u_early_a = np.maximum(
        np.asarray(arm_data["unfit"])[early_a].mean(axis=0), floor)
    u_early_b = np.maximum(
        np.asarray(base_data["unfit"])[early_b].mean(axis=0), floor)
    log_early_a, log_early_b = np.log10(u_early_a), np.log10(u_early_b)
    ci_early, d_early = compare(log_early_a, log_early_b)
    ci_change, d_change = compare(log_u_a - log_early_a,
                                  log_u_b - log_early_b)

    er_a = np.asarray(arm_data["layers"][-1]["eff_rank"])[late_a].mean(axis=0)
    er_b = np.asarray(base_data["layers"][-1]["eff_rank"])[late_b].mean(axis=0)
    ci_p2, d_p2 = compare(er_a, er_b)

    trend_a = _window_indices(arm_data["steps"], period,
                              list(P["trend_range_tasks"]))
    trend_b = _window_indices(base_data["steps"], period,
                              list(P["trend_range_tasks"]))
    task_a = np.asarray(arm_data["steps"])[trend_a] / period
    task_b = np.asarray(base_data["steps"])[trend_b] / period
    log_series_a = np.log10(np.maximum(
        np.asarray(arm_data["unfit"], dtype=np.float64)[trend_a], floor))
    log_series_b = np.log10(np.maximum(
        np.asarray(base_data["unfit"], dtype=np.float64)[trend_b], floor))
    rho_a = np.array([spearman(task_a, log_series_a[:, s])
                      for s in range(log_series_a.shape[1])])
    rho_b = np.array([spearman(task_b, log_series_b[:, s])
                      for s in range(log_series_b.shape[1])])
    ci_trend, d_trend = compare(rho_a, rho_b)

    fa, fb = _floor_fracs(arm_data, late_a, floor), _floor_fracs(base_data, late_b, floor)
    censored = bool(max(fa["floor_frac_late"], fb["floor_frac_late"])
                    > float(P["censor_frac_max"]))
    fea = _floor_fracs(arm_data, early_a, floor)
    feb = _floor_fracs(base_data, early_b, floor)
    censored_early = bool(max(fea["floor_frac_late"], feb["floor_frac_late"])
                          > float(P["censor_frac_max"]))
    p1_ok, p1_basis = _decide(ci_p1, "down", censored, paired=paired)
    early_ok, early_basis = _decide(
        ci_early, "down", censored_early, paired=paired)
    change_ok, change_basis = _decide(
        ci_change, "down", censored or censored_early, paired=paired)
    # The floor is a transform of the unfitness only; eff_rank is never censored.
    p2_ok, p2_basis = _decide(ci_p2, "up", False, paired=paired)

    if early_basis == "studentized":
        early_zero = bool(ci_early["studentized_ci_lo"] <= 0
                          <= ci_early["studentized_ci_hi"])
    elif early_basis == "percentile":
        early_zero = bool(ci_early["percentile_ci_lo"] <= 0
                          <= ci_early["percentile_ci_hi"])
    elif early_basis == "sign_test":
        early_zero = bool(ci_early["sign_test_p"] >= 0.05)
    else:
        early_zero = False

    if p1_ok and early_zero:
        signature = "LOP_PREVENTION_SIGNATURE"
    elif p1_ok and early_ok and change_ok:
        signature = "BOTH_LATE_STRONGER"
    elif p1_ok and early_ok:
        signature = "LEVEL_SHIFT_POSSIBLE"
    else:
        signature = "NO_LOP_PREVENTION_SIGNATURE"

    if paired and not pair_ok:
        verdict, split = "PAIR_BROKEN", ""
        p1_ok = p2_ok = early_ok = change_ok = False
        p1_basis = p2_basis = early_basis = change_basis = "pair_broken"
        signature = "PAIR_BROKEN"
    elif censored and not paired:
        verdict, split = "CENSORED_UNPAIRED", ""
    elif p1_ok and p2_ok:
        verdict, split = "A_EFFECTIVE", ""
    elif not p1_ok and not p2_ok:
        verdict, split = "A_NULL", ""
    else:
        verdict, split = "INCONCLUSIVE_SPLIT", ("P1_only" if p1_ok else "P2_only")
    return dict(arm=arm, baseline=base_label, contrast_type=contrast_type,
                pairing="paired" if paired else "unpaired",
                verdict=verdict, split_side=split,
                pair_ok=int(pair_ok), censored=int(censored),
                censored_early=int(censored_early), lop_signature=signature,
                p1=ci_p1, p1_early=ci_early, p1_change=ci_change,
                trend=ci_trend, p2=ci_p2,
                d_p1=d_p1, d_early=d_early, d_change=d_change,
                d_trend=d_trend, d_p2=d_p2,
                u_arm=u_a, u_base=u_b, er_arm=er_a, er_base=er_b,
                u_early_arm=u_early_a, u_early_base=u_early_b,
                rho_arm=rho_a, rho_base=rho_b,
                p1_improved=int(p1_ok), p2_improved=int(p2_ok),
                early_improved=int(early_ok), change_improved=int(change_ok),
                early_consistent_zero=int(early_zero),
                p1_basis=p1_basis, p2_basis=p2_basis,
                early_basis=early_basis, change_basis=change_basis,
                floor_arm=fa, floor_base=fb,
                floor_early_arm=fea, floor_early_base=feb)


def _task_rows_from_logs(cfg: dict, outdir: Path, seeds: list[int]) -> list[dict]:
    period = int(cfg["phase1"]["task_period"])
    rows = []
    for arm in ARM_ORDER:
        arm_cfg = _arm(cfg, arm)
        hidden = [int(v) for v in arm_cfg["hidden"]]
        flags = _centered_flags(arm_cfg, len(hidden))
        for seed in seeds:
            with np.load(outdir / "logs" / f"{arm}_seed{seed}.npz", allow_pickle=False) as z:
                idx = np.flatnonzero((z["step"] > 0) & (z["step"] % period == 0))
                for i in idx:
                    for li, width in enumerate(hidden, start=1):
                        row = dict(arm=arm, run_id=str(z["run_id"]), seed=int(seed),
                                   step=int(z["step"][i]), task=int(z["step"][i] // period),
                                   task_end=1, layer=li, centered=int(flags[li - 1]))
                        for key in P1_LOG_LAYER_KEYS:
                            value = z[f"layer{li}_{key}"][i]
                            row[key] = (int(value) if key in ("n_na", "strict_dead", "alive")
                                        else float(value))
                        row["strict_dead_frac"] = row["strict_dead"] / width
                        for key in ("signal_var", "residual_var", "unfit", "eval_loss_exact"):
                            row[key] = float(z[key][i])
                        rows.append(row)
    return rows


def analyze(cfg: dict, outdir: Path, sanity: dict, elapsed: dict) -> dict:
    P = cfg["phase1"]
    seeds = [int(v) for v in cfg["common"]["seeds"]]
    period, floor = int(P["task_period"]), float(P["unfit_floor"])
    rng = np.random.default_rng(int(P["bootstrap_seed"]))
    B = int(P["bootstrap_B"])
    paired_draws = rng.integers(0, len(seeds), size=(B, len(seeds)))
    unpaired_arm_draws = rng.integers(0, len(seeds), size=(B, len(seeds)))
    unpaired_base_draws = rng.integers(0, len(seeds), size=(B, len(seeds)))
    logdir = outdir / "logs"

    data = {arm: _arm_arrays(logdir, arm, seeds, len(_arm(cfg, arm)["hidden"]), period)
            for arm in ARM_ORDER}

    pair_final = sanity.get("S_pair_final") or {}
    pair_ok = bool((sanity.get("S_pair") or {}).get("pass_")
                   and pair_final.get("paired_pass"))
    contrast_specs = [
        ("L2_A1", BASELINE_ARM, True, "baseline"),
        ("L2_A2", BASELINE_ARM, True, "baseline"),
        ("L2_Aall", BASELINE_ARM, True, "baseline"),
        ("L2_Aall", "L2_A1", True, "interaction_A2_given_A1"),
        ("L1w100_A1", BASELINE_ARM, False, "cross_depth_unpaired"),
    ]

    verdict_rows: list[dict] = []
    details: dict = {"contrasts": [], "levels": [], "wall": [], "dose": []}

    for arm, label, paired, contrast_type in contrast_specs:
        res = _contrast(
            cfg, arm, data[arm], label, data[label], paired_draws,
            (unpaired_arm_draws, unpaired_base_draws), paired=paired,
            pair_ok=(pair_ok if paired else True), contrast_type=contrast_type)
        details["contrasts"].append({
            k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in res.items()})
        common = dict(arm=arm, baseline=label, contrast_type=contrast_type,
                      pairing=res["pairing"], main_verdict=res["verdict"],
                      split_side=res["split_side"], pair_ok=res["pair_ok"],
                      censored=res["censored"], lop_signature=res["lop_signature"],
                      early_consistent_zero=res["early_consistent_zero"],
                      floor_frac_late_arm=res["floor_arm"]["floor_frac_late"],
                      floor_frac_late_baseline=res["floor_base"]["floor_frac_late"],
                      floor_frac_all_arm=res["floor_arm"]["floor_frac_all"],
                      floor_frac_all_baseline=res["floor_base"]["floor_frac_all"])
        verdict_rows.append(dict(
            metric="P1_unfit_late", verdict=res["verdict"],
            improved=res["p1_improved"],
            decision_basis=res["p1_basis"], layer="",
            seed_values=json.dumps(res["d_p1"].tolist()),
            arm_seed_levels=json.dumps(res["u_arm"].tolist()),
            baseline_seed_levels=json.dumps(res["u_base"].tolist()),
            **common, **res["p1"]))
        verdict_rows.append(dict(
            metric="P2_eff_rank_late", verdict=res["verdict"],
            improved=res["p2_improved"],
            decision_basis=res["p2_basis"], layer=len(_arm(cfg, arm)["hidden"]),
            seed_values=json.dumps(res["d_p2"].tolist()),
            arm_seed_levels=json.dumps(res["er_arm"].tolist()),
            baseline_seed_levels=json.dumps(res["er_base"].tolist()),
            **common, **res["p2"]))
        verdict_rows.append(dict(
            metric="P1_unfit_early", verdict="REPORT_ONLY",
            improved=res["early_improved"], decision_basis=res["early_basis"],
            layer="", seed_values=json.dumps(res["d_early"].tolist()),
            arm_seed_levels=json.dumps(res["u_early_arm"].tolist()),
            baseline_seed_levels=json.dumps(res["u_early_base"].tolist()),
            **common, **res["p1_early"]))
        verdict_rows.append(dict(
            metric="P1_late_minus_early", verdict="REPORT_ONLY",
            improved=res["change_improved"], decision_basis=res["change_basis"],
            layer="", seed_values=json.dumps(res["d_change"].tolist()),
            **common, **res["p1_change"]))
        verdict_rows.append(dict(
            metric="P1_unfit_trend_spearman", verdict="REPORT_ONLY",
            improved="", decision_basis="report_only", layer="",
            seed_values=json.dumps(res["d_trend"].tolist()),
            arm_seed_levels=json.dumps(res["rho_arm"].tolist()),
            baseline_seed_levels=json.dumps(res["rho_base"].tolist()),
            **common, **res["trend"]))

    # ---- REPORT_ONLY: levels, wall coordinates, dose ----
    for arm in ARM_ORDER:
        arm_cfg = _arm(cfg, arm)
        hidden = [int(v) for v in arm_cfg["hidden"]]
        flags = _centered_flags(arm_cfg, len(hidden))
        a = data[arm]
        steps = np.asarray(a["steps"])
        early_i = _window_indices(steps, period, list(P["early_tasks"]))
        late_i = _window_indices(steps, period, list(P["late_tasks"]))
        u_late = np.maximum(np.asarray(a["unfit"])[late_i].mean(axis=0), floor)
        details["levels"].append(dict(
            arm=arm, window="late", metric="unfit",
            seed_values=u_late.tolist(), median=float(np.median(u_late)),
            min=float(u_late.min()), max=float(u_late.max()),
            **_floor_fracs(a, late_i, floor)))
        task = steps / period
        for li, layer in enumerate(a["layers"], start=1):
            centered = bool(flags[li - 1])
            for key in ("eff_rank", "eff_rank_W", "stable_rank_W", "top1_frac",
                        "wcos_mean", "sign_match_mean", "sign_clone_frac",
                        "eff_rank_per_alive", "alive", "strict_dead",
                        "w_norm_median", "dose"):
                values = np.asarray(layer[key], dtype=np.float64)[late_i].mean(axis=0)
                details["levels"].append(dict(
                    arm=arm, layer=li, window="late", metric=key,
                    centered=int(centered), seed_values=values.tolist(),
                    median=float(np.median(values)),
                    q25=float(np.quantile(values, .25)),
                    q75=float(np.quantile(values, .75))))
            early_dose = np.asarray(layer["dose"], dtype=np.float64)[early_i].mean(axis=0)
            late_dose = np.asarray(layer["dose"], dtype=np.float64)[late_i].mean(axis=0)
            dose_row = dict(metric="dose_decay", arm=arm, layer=li,
                            centered=int(centered),
                            early_median=float(np.median(early_dose)),
                            late_median=float(np.median(late_dose)),
                            seed_values=json.dumps((late_dose - early_dose).tolist()),
                            verdict="REPORT_ONLY",
                            **_ci(cfg, late_dose - early_dose, paired_draws))
            details["dose"].append(dose_row)
            verdict_rows.append(dose_row)

            D = -np.asarray(layer["median_M"], dtype=np.float64)
            rho = np.array([spearman(task, D[:, s]) for s in range(len(seeds))])
            trend = dict(metric="wall_trend", arm=arm, layer=li,
                         centered=int(centered),
                         verdict="TAUTOLOGICAL" if centered else "REPORT_ONLY",
                         seed_values=json.dumps(rho.tolist()),
                         **_ci(cfg, rho, paired_draws))
            details["wall"].append(trend)
            verdict_rows.append(trend)
            for window_name, indices in (("early", early_i), ("late", late_i)):
                values = D[indices].mean(axis=0)
                details["wall"].append(dict(
                    metric="wall_level", arm=arm, layer=li, window=window_name,
                    centered=int(centered), seed_values=values.tolist(),
                    median=float(np.median(values)),
                    q25=float(np.quantile(values, .25)),
                    q75=float(np.quantile(values, .75))))

    fields: list[str] = []
    for row in verdict_rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    write_csv(outdir / "verdict.csv",
              [{key: row.get(key, "") for key in fields} for row in verdict_rows])
    write_csv(outdir / "layer_stats.csv", _task_rows_from_logs(cfg, outdir, seeds))
    _write_summary(cfg, outdir, details, sanity)
    details["elapsed_sec"] = elapsed
    return details


def _write_summary(cfg: dict, outdir: Path, details: dict, sanity: dict) -> None:
    lines = ["# mlp2_phase1_0829 summary", "",
             "候補 A（層入力の走行平均センタリング）。co-primary は P1（未フィット率の水準）と",
             "P2（読み出し直前層の eff_rank の水準）。**両方が同じ向きを指したときだけ「効いた」**。", "",
             "## 主 endpoint（§5.2）", "",
             "| arm | baseline | pairing | P1 Δlog10 U（判定区間） | P1 改善 | P2 Δeff_rank（判定区間） | P2 改善 | verdict |",
             "|---|---|---|---:|---:|---:|---:|---|"]
    def interval(ci: dict, basis: str) -> str:
        """Show the interval the decision was actually made on, not another one."""
        if basis == "sign_test":
            return f"符号検定 p={ci['sign_test_p']:.4g}"
        if basis == "censored_unpaired":
            return "CENSORED_UNPAIRED"
        if basis == "pair_broken":
            return "PAIR_BROKEN"
        if basis == "report_only":
            return "REPORT_ONLY"
        prefix = "studentized" if basis == "studentized" else "percentile"
        return (f"[{ci[prefix + '_ci_lo']:.6g}, {ci[prefix + '_ci_hi']:.6g}]")

    for c in details["contrasts"]:
        p1, p2 = c["p1"], c["p2"]
        lines.append(
            f"| {c['arm']} | {c['baseline']} | {c['pairing']} | {p1['point']:.6g} "
            f"{interval(p1, c['p1_basis'])} | "
            f"{'yes' if c['p1_improved'] else 'no'} ({c['p1_basis']}) | "
            f"{p2['point']:.6g} {interval(p2, c['p2_basis'])} | "
            f"{'yes' if c['p2_improved'] else 'no'} ({c['p2_basis']}) | "
            f"**{c['verdict']}**{(' (' + c['split_side'] + ')') if c['split_side'] else ''} |")
    lines += ["", "paired 対比の判定基底は censored -> sign_test、CI 退化 -> percentile、"
              "それ以外 -> studentized。unpaired 対比が検閲された場合は未登録の検定を足さない。",
              "表の区間は**その行の判定に使った基底**のもの。studentized・percentile の両方と",
              "符号検定 p、全 seed の Δ と水準は verdict.csv に保存してある。", "",
              "## LoP 防止と水準シフトの判別（§5.2b）", "",
              "| arm | baseline | Δearly | Δlate−early | trend ΔSpearman | signature |",
              "|---|---|---:|---:|---:|---|"]
    for c in details["contrasts"]:
        lines.append(
            f"| {c['arm']} | {c['baseline']} | {c['p1_early']['point']:.6g} | "
            f"{c['p1_change']['point']:.6g} | {c['trend']['point']:.6g} | "
            f"**{c['lop_signature']}** |")
    lines += ["", "`LOP_PREVENTION_SIGNATURE` のときだけ「LoP を防いだ」と記述できる。"
              "この分類は主 verdict を上書きしない。", "",
              "## 床検閲と CI 退化（§5.3 / §5.5）", "",
              "| arm | 床割合(末尾窓) | baseline 床割合 | CENSORED | P1 CI退化 | P2 CI退化 | 符号検定 p (P1) |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for c in details["contrasts"]:
        lines.append(f"| {c['arm']} | {c['floor_arm']['floor_frac_late']:.4g} | "
                     f"{c['floor_base']['floor_frac_late']:.4g} | {c['censored']} | "
                     f"{c['p1']['ci_degenerate']} | {c['p2']['ci_degenerate']} | "
                     f"{c['p1']['sign_test_p']:.4g} |")
    lines += ["", "## 水準（末尾窓・REPORT_ONLY）", "",
              "| arm | 未フィット率 median [min, max] |", "|---|---:|"]
    for row in details["levels"]:
        if row["metric"] == "unfit":
            lines.append(f"| {row['arm']} | {row['median']:.6g} "
                         f"[{row['min']:.6g}, {row['max']:.6g}] |")
    lines += ["", "| arm | layer | centered | eff_rank | alive | strict_dead | dose |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    keyed = {(r["arm"], r.get("layer"), r["metric"]): r for r in details["levels"]
             if r["metric"] != "unfit"}
    for arm in ARM_ORDER:
        for li in range(1, len(_arm(cfg, arm)["hidden"]) + 1):
            def med(metric: str) -> str:
                row = keyed.get((arm, li, metric))
                return f"{row['median']:.6g}" if row else ""
            centered = keyed.get((arm, li, "eff_rank"), {}).get("centered", 0)
            lines.append(f"| {arm} | {li} | {centered} | {med('eff_rank')} | "
                         f"{med('alive')} | {med('strict_dead')} | {med('dose')} |")
    lines += ["", "## 整列（§5.2c・REPORT_ONLY）", "",
              "| arm | layer | wcos_mean | eff_rank_W | stable_rank_W | top1_frac | sign_match_mean | sign_clone_frac |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in ARM_ORDER:
        for li in range(1, len(_arm(cfg, arm)["hidden"]) + 1):
            def alignment_med(metric: str) -> str:
                row = keyed.get((arm, li, metric))
                return f"{row['median']:.6g}" if row else ""
            lines.append(
                f"| {arm} | {li} | {alignment_med('wcos_mean')} | "
                f"{alignment_med('eff_rank_W')} | {alignment_med('stable_rank_W')} | "
                f"{alignment_med('top1_frac')} | {alignment_med('sign_match_mean')} | "
                f"{alignment_med('sign_clone_frac')} |")
    lines += ["", "## dose の減衰（§5.4・REPORT_ONLY）", "",
              "| arm | layer | centered | early median | late median | Δ median [95% CI] |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in details["dose"]:
        lines.append(f"| {row['arm']} | {row['layer']} | {row['centered']} | "
                     f"{row['early_median']:.6g} | {row['late_median']:.6g} | "
                     f"{row['point']:.6g} [{row['studentized_ci_lo']:.6g}, "
                     f"{row['studentized_ci_hi']:.6g}] |")
    lines += ["", "## 壁深さ D = -median(M)（§5.6）", "",
              "**A を入れた層の D は恒真（TAUTOLOGICAL）。verdict にも機構の主張にも使わない。**", "",
              "| arm | layer | centered | median seed Spearman(task,D) | 95% CI | 扱い |",
              "|---|---:|---:|---:|---:|---|"]
    for row in details["wall"]:
        if row["metric"] != "wall_trend":
            continue
        lines.append(f"| {row['arm']} | {row['layer']} | {row['centered']} | "
                     f"{row['point']:.6g} | [{row['studentized_ci_lo']:.6g}, "
                     f"{row['studentized_ci_hi']:.6g}] | {row['verdict']} |")

    def mark(node) -> str:
        return "**PASS**" if node and node.get("pass_") else "**FAIL**"

    lines += ["", "## Sanity（§4）", "",
              f"- S0'（L2_none == phase0b L2）: {mark(sanity.get('S0prime'))}",
              f"- S-pair（L2_* の対応づけ）: {mark(sanity.get('S_pair'))}",
              f"- S-pair-final（5M 後の env 一致）: {mark(sanity.get('S_pair_final'))}",
              f"- S-taut（A を入れた層の µ 項）: {mark(sanity.get('S_taut'))}",
              f"- S-copy（厳密記録の fork 検査）: {mark(sanity.get('S_copy'))}",
              f"- S1/S2（32 パターン厳密恒等式）: "
              f"{'**PASS**' if sanity.get('S1_S2_all_pass') else '**FAIL**'}",
              f"- S3（OMP_NUM_THREADS=1）: {mark(sanity.get('S3'))}",
              f"- S5（退化ガード自己検査）: {mark(sanity.get('S5'))}",
              f"- S6（床較正）: {mark(sanity.get('S6'))}",
              f"- calibrated floor: {(sanity.get('S6') or {}).get('calibrated_floor', '')}", ""]
    (outdir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
def _provenance(cfg_path: Path, cfg: dict, outdir: Path, sanity: dict,
                analysis: dict, elapsed: dict, started: float) -> dict:
    files = [outdir / name for name in ("verdict.csv", "summary.md", "layer_stats.csv",
                                        "floor_calibration.csv", "config_used.yaml",
                                        "s0prime.json")]
    reference = Path(ROOT) / cfg["sanity"]["s0_prime_baseline_ref"]
    reference_files = {}
    for name in ("verdict.csv", "summary.md", "layer_stats.csv", "provenance.json"):
        path = reference / name
        if path.exists():
            reference_files[f"{reference.name}/{name}"] = _sha_file(path)
    for seed in cfg["common"]["seeds"]:
        path = reference / "logs" / f"L2_seed{int(seed)}.npz"
        if path.exists():
            reference_files[f"{reference.name}/logs/{path.name}"] = _sha_file(path)
    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                           text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--short"], cwd=ROOT,
                                        text=True).splitlines()
    except (OSError, subprocess.CalledProcessError):
        git_hash, dirty = None, []
    spec = Path(ROOT) / cfg["spec"]
    return dict(experiment="mlp2_phase1_0829",
                created=time.strftime("%Y-%m-%d %H:%M:%S %z"), command=sys.argv,
                elapsed_sec=round(time.time() - started, 3), arm_elapsed_sec=elapsed,
                cwd=os.getcwd(), python=sys.version, platform=platform.platform(),
                torch=torch.__version__, numpy=np.__version__,
                device=cfg["common"]["device"], git_hash=git_hash, git_dirty=dirty,
                config=str(cfg_path), config_sha256=_sha_file(cfg_path),
                spec=str(spec), spec_sha256=_sha_file(spec) if spec.exists() else None,
                baseline_inputs=reference_files, sanity=sanity, analysis=analysis,
                output_sha256={p.name: _sha_file(p) for p in files if p.exists()})


def run_full(cfg_path: Path, cfg: dict, device: str, outdir: Path, *,
             smoke: bool) -> dict:
    started = time.time()
    outdir.mkdir(parents=True, exist_ok=True)
    require_omp(cfg)
    C = cfg["common"]
    total = SMOKE_STEPS if smoke else int(C["total_steps"])
    seeds = [0] if smoke else [int(v) for v in C["seeds"]]

    if smoke:
        gates = dict(pass_=True, S3=require_omp(cfg), S5=_s5_selftest(cfg))
        s0 = {"pass_": True, "smoke": True}
    else:
        preflight_dir = Path(ROOT) / "results/_preflight_mlp2_phase1_0829"
        preflight_path = preflight_dir / "preflight.json"
        if not preflight_path.exists():
            raise FileNotFoundError("run --preflight before the full run")
        gates = json.loads(preflight_path.read_text(encoding="utf-8"))
        if not gates.get("pass_"):
            raise RuntimeError("saved preflight did not pass")
        s0_path = outdir / "s0prime.json"
        if not s0_path.exists():
            raise FileNotFoundError("run --s0prime before the full run (spec section 4)")
        s0 = json.loads(s0_path.read_text(encoding="utf-8"))
        if not s0.get("pass_"):
            raise RuntimeError("saved S0' did not pass; the full run must not proceed")
        calibration_csv = preflight_dir / "floor_calibration.csv"
        if not calibration_csv.exists():
            raise FileNotFoundError("saved S6 floor_calibration.csv is missing")
        shutil.copy2(calibration_csv, outdir / "floor_calibration.csv")

    with (outdir / "config_used.yaml").open("w") as fh:
        yaml.safe_dump(cfg, fh, allow_unicode=True, sort_keys=False)

    elapsed, identity = {}, {}
    for arm in ARM_ORDER:
        if _complete_arm_logs(outdir, arm, seeds, total, int(C["lop_every"])):
            elapsed[arm] = 0.0
            identity[arm] = {"pass_": True, "resumed_from_complete_logs": True}
            print(f"[{arm}] complete logs found; resuming after arm", flush=True)
            continue
        result = _run_arm(cfg, arm, device, outdir, seeds, total)
        elapsed[arm] = result["elapsed_sec"]
        identity[arm] = result["sanity"]

    sanity = dict(S0prime=s0, S3=gates.get("S3"), S5=gates.get("S5"),
                  S6=gates.get("S6"), S_copy=gates.get("S_copy"),
                  S_pair=gates.get("S_pair"), S_taut=gates.get("S_taut"),
                  S1_S2=identity,
                  S1_S2_all_pass=bool(all(v["pass_"] for v in identity.values())))
    if smoke:
        (outdir / "smoke_sanity.json").write_text(
            json.dumps(dict(pass_=sanity["S1_S2_all_pass"], sanity=sanity,
                            elapsed_sec=elapsed), indent=2, ensure_ascii=False,
                       default=str), encoding="utf-8")
        print(f"SMOKE DONE -> {outdir}", flush=True)
        return dict(sanity=sanity, analysis=dict(smoke=True, elapsed_sec=elapsed))

    sanity["S_pair_final"] = _pair_check_final(cfg, outdir, seeds)
    print(f"[S-pair-final] {'PASS' if sanity['S_pair_final']['pass_'] else 'FAIL'}",
          flush=True)
    result = analyze(cfg, outdir, sanity, elapsed)
    (outdir / "provenance.json").write_text(
        json.dumps(_provenance(cfg_path, cfg, outdir, sanity, result, elapsed, started),
                   indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"ALL DONE -> {outdir}", flush=True)
    return dict(sanity=sanity, analysis=result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mlp2_phase1_0829.yaml")
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
        raise ValueError("Phase 1 is CPU-only")
    stage = ("preflight" if args.preflight else "s0prime" if args.s0prime else
             "smoke" if args.smoke else "analyze" if args.analyze_only else "full")
    validate_config(cfg, stage=stage)
    main_dir = Path(ROOT) / cfg["output"]["dir"]
    outdir = (Path(args.outdir).resolve() if args.outdir else
              Path(ROOT) / "results/_preflight_mlp2_phase1_0829" if args.preflight else
              Path(ROOT) / "results/_smoke_mlp2_phase1_0829" if args.smoke else main_dir)
    if args.preflight:
        preflight(cfg_path, cfg, device, outdir)
    elif args.s0prime:
        s0prime(cfg, device, outdir)
    elif args.analyze_only:
        preflight_result = json.loads(
            (Path(ROOT) / "results/_preflight_mlp2_phase1_0829/preflight.json")
            .read_text(encoding="utf-8"))
        sanity = dict(S0prime=json.loads((outdir / "s0prime.json").read_text()),
                      S3=preflight_result["S3"], S5=preflight_result["S5"],
                      S6=preflight_result["S6"], S_copy=preflight_result["S_copy"],
                      S_pair=preflight_result["S_pair"], S_taut=preflight_result["S_taut"],
                      S1_S2={}, S1_S2_all_pass=True)
        sanity["S_pair_final"] = _pair_check_final(
            cfg, outdir, [int(v) for v in cfg["common"]["seeds"]])
        analyze(cfg, outdir, sanity, {})
    else:
        run_full(cfg_path, cfg, device, outdir, smoke=args.smoke)


if __name__ == "__main__":
    main()
