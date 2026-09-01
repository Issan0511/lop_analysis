"""Frozen width-5 gate experiment.

This module implements the eight-arm design in the frozen repo spec
``specs/spec_width5_gate_0901.md`` while preserving the frozen ``gate_dose``
implementation.  Issa explicitly authorized implementation before filling the
prediction table; the registered predictions and execution authorization are
now frozen in the config.

The new exact-support recorder adds:

* raw and floor-normalized unit mobility ``E_support[phi'(z)]``;
* the median wall coordinate ``median(M + B)`` with raw ``zbar`` and ``denom``;
* centered activation effective rank ``R_c``;
* ``R_c / sum(tilde_m)`` (undefined for the exact-linear arm because its
  structural mobility floor is one); and
* raw- and floor-normalized mobility-weighted pairwise input-weight cosines.

The frozen spec defines mobility-weighted pair cos explicitly.  The logger
emits raw/tilde and signed/absolute variants, with the raw absolute cosine of
input-weight rows weighted by ``m_i m_j`` as the report alias.
"""
from __future__ import annotations

import argparse
import copy
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
from .dose_const_5m import _input_stats, clopper_pearson
from .elu_swamp import ELU_LOG_LAYER_KEYS
from .gate_dose import (STATE_HASH_STEP, GateRecorder, _load_divergence_status,
                        _write_divergence_status, setup_arm_gate,
                        train_arm_gate)
from .mlp2_phase0 import (LOG_UNIT_KEYS, _effective_rank, _max_relative,
                          _sha_array, _sha_file, identity_sanity_pass,
                          require_omp, write_csv)
from .mlp2_phase0b import _ci_components, _complete_arm_logs, _window_indices
from .mlp2_phase1 import (NUMERIC_DIVERGENCE, NumericDivergenceError,
                          _alignment_metrics, _env_hashes, _init_hashes,
                          _numeric_divergence_event, _seed_state_hashes_p1)
from .ratchet_log import full_support_ro, teacher_f64


EXPERIMENT = "width5_gate_0901"
ARM_ORDER = ("R5", "LR5", "E5", "LIN5",
             "R100", "LR100", "E100", "LIN100")
WIDTH5_ARMS = ARM_ORDER[:4]
WIDTH100_ARMS = ARM_ORDER[4:]
RELU_ARMS = ("R5", "R100")
ELU_ARMS = ("E5", "E100")
LEAKY_ARMS = ("LR5", "LR100")
LINEAR_ARMS = ("LIN5", "LIN100")
PREFLIGHT_DIR = "results/_preflight_width5_gate_0901"
SMOKE_DIR = "results/_smoke_width5_gate_0901"
SMOKE_STEPS = 5_000
SMOKE_SEEDS = [0, 1]

REGISTERED_ARMS = {
    "R5": (5, "relu", "relu", 1.0),
    "LR5": (5, "leaky", "leaky_relu", 0.1),
    "E5": (5, "elu", "elu", 1.0),
    "LIN5": (5, "linear", "leaky_relu", 1.0),
    "R100": (100, "relu", "relu", 1.0),
    "LR100": (100, "leaky", "leaky_relu", 0.1),
    "E100": (100, "elu", "elu", 1.0),
    "LIN100": (100, "linear", "leaky_relu", 1.0),
}

UNIT_EXTRA_KEYS = ("s", "mobility", "mobility_tilde")
LAYER_EXTRA_KEYS = (
    "median_s", "median_zbar", "median_denom", "mobility_median",
    "mobility_floor_frac", "mobility_tilde_sum", "centered_eff_rank",
    "centered_eff_rank_per_mobility_mass",
    "mobility_weighted_wcos_abs", "mobility_weighted_wcos_signed",
    "mobility_tilde_weighted_wcos_abs",
    "mobility_tilde_weighted_wcos_signed",
)


def _arm(cfg: dict, name: str) -> dict:
    return next(a for a in cfg["arms"] if a["name"] == name)


def preregistration_missing(cfg: dict) -> list[str]:
    pre = cfg["preregistration"]
    missing = []
    for key in ("predictions_confirmed", "frozen", "repo_spec_committed",
                "execution_authorized"):
        if pre.get(key) is not True:
            missing.append(f"preregistration.{key}")
    return missing


def _validate_offset(cfg: dict) -> None:
    offset = int(cfg["common"]["generator_offset"])
    used = {int(v) for v in cfg["sanity"]["used_generator_offsets"]}
    forbidden = {abs(int(v)) for v in
                 cfg["sanity"]["forbidden_offset_differences"]}
    if offset in used:
        raise ValueError("generator_offset collides with an existing use")
    collisions = sorted(v for v in used if abs(offset - v) in forbidden)
    if collisions:
        raise ValueError(f"generator_offset width collision with {collisions}")


def validate_config(cfg: dict, *, stage: str) -> None:
    """Validate the frozen design before every result-bearing stage."""
    if stage not in {"implementation", "preflight", "smoke", "full", "analyze"}:
        raise ValueError(f"unknown stage {stage!r}")
    C, A, P, G, S = (cfg["common"], cfg["condA"], cfg["phase1"],
                     cfg["width5_gate"], cfg["sanity"])
    if [a["name"] for a in cfg["arms"]] != list(ARM_ORDER):
        raise ValueError(f"arms must be ordered as {ARM_ORDER}")
    for arm in cfg["arms"]:
        width, label, _, _ = REGISTERED_ARMS[arm["name"]]
        if ([int(v) for v in arm["hidden"]] != [width]
                or str(arm["activation"]) != label
                or list(arm.get("centered_layers") or [])
                or arm.get("target_mu_norm") is not None
                or arm.get("target_dose") is not None):
            raise ValueError(f"{arm['name']} differs from the decided design")
    if (int(A["m"]), int(A["f"]), int(A["target_hidden"])) != (20, 15, 100):
        raise ValueError("width5_gate requires condA m=20, f=15, teacher width=100")
    if list(A["T_values"]) != [10_000] or list(A["encodings"]) != ["std"]:
        raise ValueError("width5_gate requires T=10000 and std encoding")
    activation = cfg["activation"]
    if (str(activation["relu"]["name"]) != "relu"
            or str(activation["leaky"]["name"]) != "leaky_relu"
            or float(activation["leaky"]["slope"]) != 0.1
            or str(activation["elu"]["name"]) != "elu"
            or float(activation["elu"]["alpha"]) != 1.0
            or str(activation["elu"]["derivative_form"]) != "alpha_exp"
            or str(activation["linear"]["name"]) != "leaky_relu"
            or float(activation["linear"]["slope"]) != 1.0
            or activation["autograd"] is not False
            or activation["consumes_rng"] is not False):
        raise ValueError("activation definitions differ from the decided design")
    if (str(cfg["intervention"]["name"]) != "none"
            or cfg["intervention"]["oracle"] is not False
            or cfg["intervention"]["consumes_rng"] is not False):
        raise ValueError("all arms must remain intervention-free")
    if (int(C["total_steps"]) != 5_000_000
            or list(C["seeds"]) != list(range(20))
            or int(C["generator_offset"]) != 202_609_011_921
            or float(C["lr_main"]) != 0.01
            or str(C["device"]) != "cpu"):
        raise ValueError("step/seed/offset/lr/device design changed")
    _validate_offset(cfg)
    expected_phase = {
        "task_period": 10_000, "early_tasks": [2, 11],
        "late_tasks_5m": [491, 500], "window_1m_tasks": [91, 100],
        "exact_support": 32, "onset_threshold": 0.05,
        "onset_present_min": 5, "unfit_floor": 1e-16,
        "recalibrate_floor": False, "bootstrap_B": 10_000,
        "bootstrap_seed": 202_609_011_921,
        "ci_method": "percentile_primary_studentized_secondary",
    }
    for key, value in expected_phase.items():
        if P[key] != value:
            raise ValueError(f"phase1.{key} differs from the decided design")
    if (float(P["degenerate_se_tol"]) != 1e-15
            or float(P["degenerate_frac_max"]) != 0.01
            or float(P["degenerate_width_ratio_max"]) != 100.0):
        raise ValueError("CI degeneracy guard changed")
    if cfg["pairing"]["paired_groups"] != [list(WIDTH5_ARMS), list(WIDTH100_ARMS)]:
        raise ValueError("pairing groups changed")
    if (G["primary_verdict"] != "G0_only"
            or list(G["phenomenon_arms"]) != ["LR5", "E5"]
            or list(G["saturation_arms"]) != list(WIDTH5_ARMS)
            or int(G["reproduced_min_onset"]) != 5
            or list(G["level_contrasts"]) != [["LR5", "LIN5"],
                                               ["E5", "LIN5"]]
            or float(G["level_equivalence_resolution_dex"]) != 0.5
            or G["level_verdict_registered"] is not False
            or G["width_contrasts_registered"] is not False
            or G["mechanism_metrics_registered"] is not False
            or G["submerged_in_verdict"] is not False
            or G["strict_dead_in_verdict"] is not False):
        raise ValueError("G0/G1/G2/G3 decision structure changed")
    if (G["mobility_pair_cos_definition"]
            != "weighted_abs_cosine_of_input_weight_rows"
            or G["mobility_pair_weights"] != "product_of_unit_mobilities"
            or G["linear_tilde_mobility_policy"] != "undefined_nan"):
        raise ValueError("report-only pair-cos interpretation changed")
    if (int(S["s0_prime_steps"]) != 30_000
            or list(S["s0_prime_seeds"]) != list(range(10))
            or int(S["s0_prime_generator_offset"]) != 0
            or list(S["s0_prime_metrics"]) != [
                "layer1_p_hat", "eval_loss_exact", "unfit"]
            or list(S["s_cap_arms"]) != ["R5", "LIN5"]
            or list(S["s_cap_tasks"]) != [2, 11]
            or S["s_cap_rule"]
            != "max_seed_min_early_unfit_below_onset_threshold"
            or int(S["omp_num_threads"]) != 1):
        raise ValueError("sanity design changed")
    pre = cfg["preregistration"]
    if (pre["decisions_complete"] is not True
            or pre["implementation_before_predictions_authorized"] is not True
            or pre["level_equivalence_resolution_confirmed"] is not True
            or pre["generator_offset_confirmed"] is not True
            or pre["bootstrap_seed_confirmed"] is not True):
        raise ValueError("Issa's six decisions are not represented in config")
    expected_predictions = {
        "R5_n_onset_5m": "20_of_20",
        "LR5_n_onset_5m": "unknown",
        "E5_n_onset_5m": "unknown",
        "LR5_vs_LIN5_log10U_position": "above",
        "LR5_submerged_fraction_5m": "same_as_w100_0.63_to_0.67",
        "E5_submerged_fraction_5m": "same_as_w100_0.36_to_0.45",
        "Rc_LR5_vs_LIN5": "lower",
        "failure_cause_if_prediction_misses": "unknown",
    }
    if (pre.get("prediction_provenance")
            != "draft_candidates_proposed_first_then_approved_by_Issa"
            or pre.get("predictions") != expected_predictions):
        raise ValueError("Issa's eight frozen predictions changed")
    spec_path = Path(ROOT) / str(cfg["spec"])
    if (not spec_path.is_file()
            or _sha_file(spec_path) != pre.get("repo_spec_sha256")):
        raise ValueError("frozen repo spec is missing or changed")
    if stage != "implementation":
        missing = preregistration_missing(cfg)
        if missing:
            raise ValueError("width5_gate preregistration is not frozen: "
                             + ", ".join(missing))


def _activation(cfg: dict, arm_cfg: dict) -> tuple[str, float]:
    _, _, implementation, alpha = REGISTERED_ARMS[str(arm_cfg["name"])]
    return implementation, float(alpha)


def setup_arm_width(cfg: dict, arm_cfg: dict, device: str) -> dict:
    """Use the frozen gate-dose setup with width and activation only varied."""
    st = setup_arm_gate(cfg, arm_cfg, device)
    implementation, alpha = _activation(cfg, arm_cfg)
    if st["activation"] != implementation or st["act_alpha"] != alpha:
        # gate_dose does not know the `linear` config label, so set it here.
        st["net"].set_activation(implementation, alpha, "alpha_exp")
        st["activation"], st["act_alpha"] = implementation, alpha
    st["activation_label"] = str(arm_cfg["activation"])
    st["generator_offset"] = int(cfg["common"]["generator_offset"])
    return st


def _mobility_floor(st: dict) -> float:
    if st["activation"] == "leaky_relu":
        return float(st["act_alpha"])
    return 0.0


def centered_effective_rank(activation: torch.Tensor) -> torch.Tensor:
    """Entropy effective rank after subtracting each unit's support mean."""
    centered = activation - activation.mean(dim=0, keepdim=True)
    return _effective_rank(centered.permute(1, 0, 2))


def _weighted_pair_cos(W: torch.Tensor, weights: torch.Tensor,
                       *, absolute: bool) -> torch.Tensor:
    """Weighted mean pairwise row cosine for each vectorized seed."""
    width = W.shape[1]
    iu = torch.triu_indices(width, width, offset=1, device=W.device)
    Wn = W / W.norm(dim=2, keepdim=True).clamp_min(1e-300)
    gram = torch.einsum("rhd,rjd->rhj", Wn, Wn)
    pair = gram[:, iu[0], iu[1]]
    if absolute:
        pair = pair.abs()
    pair_weights = weights[:, iu[0]] * weights[:, iu[1]]
    denominator = pair_weights.sum(dim=1)
    result = torch.full_like(denominator, float("nan"))
    valid = denominator > 0
    result[valid] = ((pair[valid] * pair_weights[valid]).sum(dim=1)
                     / denominator[valid])
    return result


def exact_layer_record_width(st: dict, sigma_tol: float,
                             mobility_floor_tol: float) -> tuple[dict, dict]:
    """One-layer exact-support record with the P-3 report-only metrics."""
    if len(st["net"].Ws) != 1:
        raise ValueError("width5_gate exact recorder requires one hidden layer")
    net = st["net"]
    with torch.no_grad():
        X = full_support_ro(st["env"]).double()
        y = teacher_f64(st["teacher"], X)
        W, b = net.Ws[0].double(), net.bs[0].double()
        mu = X.mean(dim=0)
        centered = X - mu[None]
        z = torch.einsum("rhd,prd->prh", W, X) + b
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
        s = M + B

        activation = net.act_fn(z)
        derivative = net.act_grad(z, activation).double()
        mobility = derivative.mean(dim=0)
        floor = _mobility_floor(st)
        if floor < 1.0:
            mobility_tilde = ((mobility - floor) / (1.0 - floor)).clamp(0.0, 1.0)
            tilde_mass = mobility_tilde.sum(dim=1)
        else:
            # For leaky(a=1), (m-a)/(1-a) is 0/0.  Keep this explicit.
            mobility_tilde = torch.full_like(mobility, float("nan"))
            tilde_mass = torch.full((st["R"],), float("nan"), dtype=W.dtype,
                                    device=W.device)
        floor_frac = ((mobility - floor).abs() <= float(mobility_floor_tol))
        floor_frac = floor_frac.double().mean(dim=1)

        rc = centered_effective_rank(activation)
        rc_per_mass = torch.full_like(rc, float("nan"))
        valid_mass = torch.isfinite(tilde_mass) & (tilde_mass > 0)
        rc_per_mass[valid_mass] = rc[valid_mass] / tilde_mass[valid_mass]
        raw_wcos_abs = _weighted_pair_cos(W, mobility, absolute=True)
        raw_wcos_signed = _weighted_pair_cos(W, mobility, absolute=False)
        tilde_wcos_abs = _weighted_pair_cos(W, mobility_tilde, absolute=True)
        tilde_wcos_signed = _weighted_pair_cos(W, mobility_tilde, absolute=False)

        p_hat = (z > 0).double().mean(dim=0)
        submerged_unit = z.amax(dim=0) <= 0
        w_norm = W.norm(dim=2)
        mu_norm = mu.norm(dim=1)
        sigma_rms = centered.square().mean(dim=0).sum(dim=1)
        sigma_rms = (sigma_rms / X.shape[2]).clamp_min(0).sqrt()
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
        qW = torch.quantile(w_norm, torch.tensor([0.25, 0.5, 0.75],
                                                 dtype=w_norm.dtype), dim=1)
        alignment = _alignment_metrics(W, float(st.get("sign_match_tau", 0.95)))
        layer = dict(
            M=M, B=B, denom=denom, p_hat=p_hat, w_norm=w_norm,
            zbar=direct_mean, s=s, mobility=mobility,
            mobility_tilde=mobility_tilde,
            median_M=qM[1], q25_M=qM[0], q75_M=qM[2],
            median_B=torch.nanquantile(B, 0.5, dim=1),
            median_s=torch.nanquantile(s, 0.5, dim=1),
            median_zbar=torch.quantile(direct_mean, 0.5, dim=1),
            median_denom=torch.quantile(denom, 0.5, dim=1),
            mobility_median=torch.quantile(mobility, 0.5, dim=1),
            mobility_floor_frac=floor_frac,
            mobility_tilde_sum=tilde_mass,
            centered_eff_rank=rc,
            centered_eff_rank_per_mobility_mass=rc_per_mass,
            mobility_weighted_wcos_abs=raw_wcos_abs,
            mobility_weighted_wcos_signed=raw_wcos_signed,
            mobility_tilde_weighted_wcos_abs=tilde_wcos_abs,
            mobility_tilde_weighted_wcos_signed=tilde_wcos_signed,
            n_na=(~valid).sum(dim=1), mu_norm=mu_norm, sigma_rms=sigma_rms,
            dose=dose, w_norm_median=qW[1], w_norm_q25=qW[0],
            w_norm_q75=qW[2], eff_rank=eff_rank, eff_rank_W=eff_rank_W,
            strict_dead=strict_dead, alive=alive,
            eff_rank_per_alive=eff_per_alive,
            submerged=submerged_unit.sum(dim=1),
            preact_sd_median=torch.quantile(direct_sd, 0.5, dim=1),
            **alignment)

        mu_u = mu / mu_norm.clamp_min(1e-300)[:, None]
        cos = torch.einsum("rhd,rd->rh", W, mu_u) / w_norm.clamp_min(1e-300)
        cos_err = _max_relative(cos * mu_norm[:, None],
                                wmu / w_norm.clamp_min(1e-300))
        sanity_layer = dict(
            layer=1,
            mean_max_relerr=_max_relative(direct_mean, formula_mean),
            sd_max_relerr=_max_relative(direct_sd, denom),
            wall_max_relerr=_max_relative(wall_direct, wall_formula),
            l1_cos_mu_max_relerr=cos_err,
            n_degenerate=int((~valid).sum().item()),
            submerge_mismatch=int((submerged_unit != (p_hat == 0)).sum().item()),
            mobility_min=float(mobility.min()),
            mobility_max=float(mobility.max()),
            finite_required=bool(
                torch.isfinite(z).all() and torch.isfinite(mu).all()
                and torch.isfinite(denom).all() and torch.isfinite(eff_rank).all()
                and torch.isfinite(eff_rank_W).all() and torch.isfinite(rc).all()
                and torch.isfinite(mobility).all()))

        yhat = (activation * net.v.double()).sum(dim=-1) + net.c.double()
        residual = yhat - y
        signal_var = y.var(dim=0, unbiased=False)
        residual_var = residual.var(dim=0, unbiased=False)
        unfit = residual_var / signal_var
        run = dict(signal_var=signal_var, residual_var=residual_var, unfit=unfit,
                   eval_loss_exact=residual.square().mean(dim=0))
        run_finite = bool(all(torch.isfinite(v).all() for v in run.values())
                          and (signal_var > 0).all())
        sanity = dict(layers=[sanity_layer], run_finite=run_finite,
                      support=int(X.shape[0]), taut=[])
        return dict(run=run, layers=[layer], v_readout=net.v.double(),
                    flip_state=st["env"].flip_state.double()), sanity


class WidthGateRecorder(GateRecorder):
    """Gate recorder driven by :func:`exact_layer_record_width`."""

    def __init__(self, steps: list[int], st: dict, cfg: dict):
        super().__init__(steps, st)
        n, runs, width = len(self.steps), st["R"], st["hidden"][0]
        for key in UNIT_EXTRA_KEYS:
            self.layers[0][key] = np.empty((n, runs, width), dtype=np.float32)
        for key in LAYER_EXTRA_KEYS:
            self.layers[0][key] = np.empty((n, runs), dtype=np.float64)
        self.mobility_floor_tol = float(
            cfg["width5_gate"]["mobility_floor_tolerance"])

    def __call__(self, st: dict, step: int) -> None:
        i = self.index.get(int(step))
        if i is None:
            return
        if self.filled[i]:
            raise RuntimeError(f"duplicate width5_gate probe at step {step}")
        divergence = _numeric_divergence_event(st, int(step))
        if divergence is not None:
            raise NumericDivergenceError(divergence)
        rec, sanity = exact_layer_record_width(
            st, self.sigma_tol, self.mobility_floor_tol)
        for key, value in rec["run"].items():
            self.run[key][i] = value.detach().cpu().numpy()
        self.flip_state[i] = rec["flip_state"].detach().cpu().numpy().astype(np.float32)

        layer = rec["layers"][0]
        for key in LOG_UNIT_KEYS + UNIT_EXTRA_KEYS:
            self.layers[0][key][i] = layer[key].detach().cpu().numpy().astype(np.float32)
        for key in ELU_LOG_LAYER_KEYS + LAYER_EXTRA_KEYS:
            self.layers[0][key][i] = layer[key].detach().cpu().numpy()
        zbar = layer["zbar"].detach()
        self.layers[0]["zbar"][i] = zbar.cpu().numpy().astype(np.float32)
        adjacent = (self._prev_step is not None
                    and int(step) - int(self._prev_step) == self.interval)
        if adjacent:
            delta = (zbar - self._prev_zbar[0]).cpu().numpy()
        else:
            delta = np.full(zbar.shape, np.nan, dtype=np.float64)
        self.layers[0]["dzbar"][i] = delta.astype(np.float32)

        s, acc = sanity["layers"][0], self.max_errors[0]
        acc["mean"] = max(acc["mean"], s["mean_max_relerr"])
        acc["sd"] = max(acc["sd"], s["sd_max_relerr"])
        acc["wall"] = max(acc["wall"], s["wall_max_relerr"])
        acc["cos_mu"] = max(acc["cos_mu"], s["l1_cos_mu_max_relerr"])
        acc["n_degenerate_max"] = max(acc["n_degenerate_max"], s["n_degenerate"])
        self.submerge_mismatch += int(s["submerge_mismatch"])
        if not s["finite_required"] or not sanity["run_finite"]:
            self.required_nonfinite += 1

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
        self._prev_zbar, self._prev_step = [zbar.clone()], int(step)
        self.filled[i] = True


def write_arm_logs(outdir: Path, arm: str, st: dict,
                   rec: WidthGateRecorder) -> list[Path]:
    logdir = outdir / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ri, run in enumerate(st["runs"]):
        seed = int(run["seed"])
        payload = dict(
            step=rec.steps, run_id=np.array(run["run_id"]), arm=np.array(arm),
            seed=np.int64(seed), width=np.int64(st["hidden"][0]),
            activation=np.array(st["activation"]),
            activation_label=np.array(st["activation_label"]),
            act_alpha=np.float64(st["act_alpha"]),
            generator_offset=np.int64(st["generator_offset"]),
            task_period=np.int64(run["period"]),
            state_hash_final=np.array(json.dumps(
                _seed_state_hashes_p1(st, ri), sort_keys=True)),
            state_hash_1m=np.array(json.dumps(
                rec.state_hash_1m.get(seed, {}), sort_keys=True)))
        for key, value in rec.run.items():
            payload[key] = value[:, ri]
        payload["flip_state"] = rec.flip_state[:, ri]
        for key, value in rec.extra.items():
            payload[key] = value[:, ri]
        for key, value in rec.layers[0].items():
            payload[f"layer1_{key}"] = value[:, ri]
        path = logdir / f"{arm}_seed{seed}.npz"
        np.savez_compressed(path, **payload)
        paths.append(path)
    return paths


def _run_arm(cfg: dict, arm: str, device: str, outdir: Path,
             seeds: list[int], total: int) -> dict:
    c = copy.deepcopy(cfg)
    c["common"]["seeds"] = [int(v) for v in seeds]
    st = setup_arm_width(c, _arm(c, arm), device)
    every = int(c["common"]["lop_every"])
    probes = list(range(0, total + 1, every))
    if probes[-1] != total:
        probes.append(total)
    _, sanity0 = exact_layer_record_width(
        st, float(c["sanity"]["sigma_degenerate_tol"]),
        float(c["width5_gate"]["mobility_floor_tolerance"]))
    if not identity_sanity_pass(sanity0, float(c["sanity"]["identity_tol"])):
        raise RuntimeError(f"{arm} initial exact-support identity failed")
    rec = WidthGateRecorder(probes, st, c)
    checkpoints = [int(v) for v in c["common"].get("checkpoints", [])
                   if int(v) <= total]
    print(f"[{arm}] width={st['hidden'][0]} act={st['activation_label']} "
          f"seeds={seeds} steps={total:,}", flush=True)
    started = time.time()
    try:
        elapsed = train_arm_gate(st, rec, probes, total, outdir, checkpoints)
    except NumericDivergenceError as exc:
        elapsed = time.time() - started
        event = dict(exc.event)
        event.update(probe_every=every, registered_total_steps=int(total),
                     registered_seeds=[int(v) for v in seeds],
                     activation=st["activation_label"], act_alpha=st["act_alpha"],
                     width=st["hidden"][0], elapsed_sec=float(elapsed),
                     detection="nonfinite_training_state_at_probe",
                     partial_logs_excluded=True, rescue="none")
        status = _write_divergence_status(outdir, event)
        print(f"[{arm}] {NUMERIC_DIVERGENCE} -> {status}", flush=True)
        return dict(status=NUMERIC_DIVERGENCE, elapsed_sec=elapsed,
                    sanity=dict(pass_=False, numeric_divergence=True, event=event),
                    divergence=event, final_env=_env_hashes(st))
    sanity = rec.sanity()
    if not sanity["pass_"]:
        raise RuntimeError(f"{arm} exact-support sanity failed: {sanity}")
    write_arm_logs(outdir, arm, st, rec)
    print(f"[{arm}] complete in {elapsed:.1f}s", flush=True)
    return dict(status="COMPLETE", elapsed_sec=elapsed, sanity=sanity,
                final_env=_env_hashes(st))


# ---------------------------------------------------------------------------
# Sanity gates (implemented now, execution blocked until predictions freeze)
# ---------------------------------------------------------------------------
def _s_linear(cfg: dict) -> dict:
    from .nets import VecMLPL

    tol = float(cfg["sanity"]["s_linear_tolerance"])
    gen = torch.Generator().manual_seed(90_011)
    net = VecMLPL(3, [5], 4, gen, "cpu").set_activation(
        "leaky_relu", 1.0, "alpha_exp")
    net.Ws = [value.double() for value in net.Ws]
    net.bs = [value.double() for value in net.bs]
    net.v, net.c = net.v.double(), net.c.double()
    net.W, net.b = net.Ws[0], net.bs[0]
    z = torch.linspace(-100.0, 100.0, 10_001, dtype=torch.float64)
    activation_error = float((net.act_fn(z) - z).abs().max())
    x = torch.randn(17, 3, 4, dtype=torch.float64,
                    generator=torch.Generator().manual_seed(90_012))
    pres, acts, yhat = net.forward_layers_batch(x)
    affine_w = torch.einsum("rh,rhd->rd", net.v, net.Ws[0])
    affine_b = (net.v * net.bs[0]).sum(dim=1) + net.c
    expected = torch.einsum("nrd,rd->nr", x, affine_w) + affine_b
    affine_error = float((yhat - expected).abs().max())
    derivative_error = float((net.act_grad(pres[0], acts[0]) - 1).abs().max())
    return dict(pass_=bool(max(activation_error, affine_error, derivative_error) <= tol),
                tolerance=tol, activation_error=activation_error,
                affine_error=affine_error, derivative_error=derivative_error)


def _s_mobility(cfg: dict) -> dict:
    tol = float(cfg["sanity"]["s_mobility_tolerance"])
    z = torch.tensor([[[-2.0, 1.0, 0.0], [3.0, -4.0, 2.0]]],
                     dtype=torch.float64)
    p_hat = (z > 0).double().mean(dim=0)
    rows, failures = [], []
    for label, act, alpha in (("relu", "relu", 1.0),
                              ("leaky", "leaky_relu", 0.1)):
        from .nets import VecMLPL
        net = VecMLPL(2, [3], 2, torch.Generator().manual_seed(4), "cpu")
        net.set_activation(act, alpha, "alpha_exp")
        got = net.act_grad(z, net.act_fn(z)).mean(dim=0)
        expected = p_hat if label == "relu" else alpha + (1 - alpha) * p_hat
        error = float((got - expected).abs().max())
        row = dict(activation=label, max_abs_error=error)
        rows.append(row)
        if error > tol:
            failures.append(row)
    return dict(pass_=not failures, tolerance=tol, rows=rows, failures=failures)


def _s_rank(cfg: dict) -> dict:
    tol = float(cfg["sanity"]["s_rank_tolerance"])
    # Three large independent constant columns plus one varying column: raw
    # rank is >1, while mean removal leaves a rank-one pattern matrix.
    patterns = torch.linspace(-1.0, 1.0, 32, dtype=torch.float64)
    constant = torch.tensor([10.0, -7.0, 4.0], dtype=torch.float64).repeat(32, 1)
    A = torch.cat([constant, patterns[:, None]], dim=1)[:, None, :]
    raw = _effective_rank(A.permute(1, 0, 2))[0]
    centered = centered_effective_rank(A)[0]
    # Five zero-mean orthogonal columns on a 32-point support.
    Q, _ = torch.linalg.qr(torch.randn(
        32, 5, dtype=torch.float64,
        generator=torch.Generator().manual_seed(90_013)))
    Q = Q - Q.mean(dim=0, keepdim=True)
    Q, _ = torch.linalg.qr(Q)
    orth = centered_effective_rank(Q[:, None, :])[0]
    range_value = centered_effective_rank(
        torch.randn(32, 2, 5, dtype=torch.float64,
                    generator=torch.Generator().manual_seed(90_014)))
    passed = (float(centered) < float(raw)
              and abs(float(orth) - 5.0) <= tol
              and bool(((range_value >= 1.0 - tol)
                        & (range_value <= 5.0 + tol)).all()))
    return dict(pass_=bool(passed), tolerance=tol, raw_constant=float(raw),
                centered_constant=float(centered), orthogonal=float(orth),
                width5_range=range_value.tolist())


def _s_offset(cfg: dict) -> dict:
    try:
        _validate_offset(cfg)
        return dict(pass_=True, generator_offset=int(cfg["common"]["generator_offset"]),
                    used=list(cfg["sanity"]["used_generator_offsets"]),
                    forbidden_differences=list(
                        cfg["sanity"]["forbidden_offset_differences"]))
    except ValueError as exc:
        return dict(pass_=False, error=str(exc))


def _s_floor(cfg: dict) -> dict:
    path = Path(ROOT) / str(cfg["sanity"]["floor_inherited_from"])
    if not path.exists():
        return dict(pass_=False, path=str(path), reason="missing")
    data = np.genfromtxt(path, delimiter=",", names=True)
    values = np.unique(np.asarray(data["calibrated_floor"], dtype=np.float64))
    expected = float(cfg["phase1"]["unfit_floor"])
    return dict(pass_=bool(values.size == 1 and values[0] == expected
                           and cfg["phase1"]["recalibrate_floor"] is False),
                path=str(path), values=values.tolist(), configured=expected)


def _s_initial_pairing(cfg: dict) -> dict:
    differences, hashes = [], {}
    for group in (WIDTH5_ARMS, WIDTH100_ARMS):
        reference = group[0]
        for arm in group:
            st = setup_arm_width(copy.deepcopy(cfg), _arm(cfg, arm), "cpu")
            hashes[arm] = _init_hashes(st)
        for arm in group[1:]:
            for key, value in hashes[reference].items():
                if hashes[arm].get(key) != value:
                    differences.append(dict(group=list(group), arm=arm, key=key))
    return dict(pass_=not differences, hashes=hashes, differences=differences,
                caveat=cfg["pairing"]["trajectory_caveat"])


def _compare_replay_log(ours: Path, reference: Path, total: int,
                        metrics: list[str]) -> list[dict]:
    differences = []
    with np.load(ours, allow_pickle=False) as a, np.load(reference,
                                                        allow_pickle=False) as b:
        keep = np.asarray(b["step"], dtype=np.int64) <= int(total)
        if not np.array_equal(a["step"], b["step"][keep]):
            differences.append(dict(metric="step", reason="grid mismatch"))
        for key in metrics:
            if key not in a or key not in b:
                differences.append(dict(metric=key, reason="missing"))
            elif _sha_array(a[key]) != _sha_array(b[key][keep]):
                differences.append(dict(metric=key, reason="hash mismatch"))
    return differences


def _s0_replay(cfg: dict, device: str, outdir: Path) -> dict:
    S = cfg["sanity"]
    c = copy.deepcopy(cfg)
    c["common"]["generator_offset"] = int(S["s0_prime_generator_offset"])
    seeds = [int(v) for v in S["s0_prime_seeds"]]
    total = int(S["s0_prime_steps"])
    reference_dir = Path(ROOT) / str(S["s0_prime_reference"])
    arms, all_differences = {}, []
    for arm, reference_arm in dict(S["s0_prime_arm_map"]).items():
        result = _run_arm(c, arm, device, outdir, seeds, total)
        if result["status"] != "COMPLETE":
            raise RuntimeError(f"S0' replay diverged for {arm}")
        differences = []
        for seed in seeds:
            ours = outdir / "logs" / f"{arm}_seed{seed}.npz"
            reference = reference_dir / "logs" / f"{reference_arm}_seed{seed}.npz"
            if not reference.exists():
                differences.append(dict(seed=seed, reason="reference missing",
                                        path=str(reference)))
            else:
                differences.extend(dict(seed=seed, **row) for row in
                                   _compare_replay_log(
                                       ours, reference, total,
                                       list(S["s0_prime_metrics"])))
        arms[arm] = dict(pass_=not differences, reference_arm=reference_arm,
                         differences=differences)
        all_differences.extend(dict(arm=arm, **row) for row in differences)
    return dict(pass_=not all_differences, total_steps=total, seeds=seeds,
                generator_offset=int(S["s0_prime_generator_offset"]), arms=arms,
                differences=all_differences, reference=str(reference_dir))


def _s_cap(cfg: dict, device: str, outdir: Path) -> dict:
    S, P = cfg["sanity"], cfg["phase1"]
    tasks = [int(v) for v in S["s_cap_tasks"]]
    total = tasks[1] * int(P["task_period"])
    seeds = [int(v) for v in cfg["common"]["seeds"]]
    rows, failures = [], []
    for arm in S["s_cap_arms"]:
        result = _run_arm(cfg, str(arm), device, outdir, seeds, total)
        if result["status"] != "COMPLETE":
            failures.append(dict(arm=arm, reason=NUMERIC_DIVERGENCE))
            continue
        minima = []
        for seed in seeds:
            with np.load(outdir / "logs" / f"{arm}_seed{seed}.npz",
                         allow_pickle=False) as z:
                idx = _window_indices(z["step"], int(P["task_period"]), tasks)
                minima.append(float(np.min(np.asarray(z["unfit"])[idx])))
        max_min = float(np.max(minima))
        passed = max_min < float(P["onset_threshold"])
        row = dict(arm=arm, rule=S["s_cap_rule"], tasks=tasks,
                   per_seed_min=minima, max_seed_min=max_min,
                   onset_threshold=float(P["onset_threshold"]), pass_=passed)
        rows.append(row)
        if not passed:
            failures.append(row)
    return dict(pass_=not failures, rows=rows, failures=failures,
                interpretation_note=S["s_cap_note"])


def preflight(cfg: dict, device: str, outdir: Path) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    checks = {"S_omp": require_omp(cfg), "S_offset": _s_offset(cfg),
              "S_linear": _s_linear(cfg), "S_mobility": _s_mobility(cfg),
              "S_rank": _s_rank(cfg), "S_floor": _s_floor(cfg),
              "S_initial_pairing": _s_initial_pairing(cfg)}
    checks["S0prime"] = _s0_replay(cfg, device, outdir / "s0prime")
    checks["S_cap"] = _s_cap(cfg, device, outdir / "scap")
    result = dict(pass_=bool(all(v.get("pass_") for v in checks.values())),
                  **checks)
    (outdir / "preflight.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")
    for name, value in checks.items():
        print(f"[{name}] {'PASS' if value.get('pass_') else 'FAIL'}", flush=True)
    if not result["pass_"]:
        failed = [k for k, v in checks.items() if not v.get("pass_")]
        raise RuntimeError(f"preflight failed: {failed}")
    return result


# ---------------------------------------------------------------------------
# Registered and report-only decisions
# ---------------------------------------------------------------------------
def classify_g0(onset: dict[str, int], trials: int = 20,
                missing: set[str] | None = None) -> str:
    missing = missing or set()
    if missing & set(WIDTH5_ARMS):
        return "INCONCLUSIVE_DIVERGENCE"
    counts = [int(onset[a]) for a in WIDTH5_ARMS]
    if all(v == 0 for v in counts) or all(v == trials for v in counts):
        return "BINARY_SATURATED_NO_VERDICT"
    lr, elu = int(onset["LR5"]), int(onset["E5"])
    if lr == 0 and elu == 0:
        return "PHENOMENON3_NOT_REPRODUCED"
    if max(lr, elu) >= 5:
        return "PHENOMENON3_REPRODUCED"
    return "PHENOMENON3_MARGINAL"


def classify_level(point: float, ci_lo: float, ci_hi: float,
                   resolution: float, label: str) -> str:
    if point >= resolution:
        return f"{label}_ABOVE_LINEAR"
    if point <= -resolution:
        return f"{label}_BELOW_LINEAR"
    if ci_lo >= -resolution and ci_hi <= resolution:
        return f"{label}_WITHIN_RESOLUTION"
    return "INCONCLUSIVE_WIDE"


def _ci(cfg: dict, values: np.ndarray, draws: np.ndarray) -> dict:
    P = cfg["phase1"]
    return _ci_components(np.asarray(values, dtype=np.float64), draws, "median",
                          float(P["degenerate_se_tol"]),
                          float(P["degenerate_frac_max"]),
                          float(P["degenerate_width_ratio_max"]))


RUN_KEYS = ("unfit", "eval_loss_exact", "signal_var", "residual_var")
LAYER_KEYS = (
    "layer1_strict_dead", "layer1_submerged", "layer1_eff_rank",
    "layer1_median_s", "layer1_median_zbar", "layer1_median_denom",
    "layer1_mobility_median", "layer1_mobility_floor_frac",
    "layer1_mobility_tilde_sum", "layer1_centered_eff_rank",
    "layer1_centered_eff_rank_per_mobility_mass",
    "layer1_mobility_weighted_wcos_abs",
    "layer1_mobility_weighted_wcos_signed",
    "layer1_mobility_tilde_weighted_wcos_abs",
    "layer1_mobility_tilde_weighted_wcos_signed",
)


def _load_arm(cfg: dict, outdir: Path, arm: str) -> dict:
    items = []
    for seed in [int(v) for v in cfg["common"]["seeds"]]:
        with np.load(outdir / "logs" / f"{arm}_seed{seed}.npz",
                     allow_pickle=False) as z:
            items.append({"step": z["step"].copy(),
                          **{key: z[key].copy() for key in RUN_KEYS + LAYER_KEYS}})
    result = {"step": items[0]["step"]}
    result.update({key: np.stack([item[key] for item in items], axis=1)
                   for key in RUN_KEYS + LAYER_KEYS})
    return result


def _finite_mean(values: np.ndarray, axis: int) -> np.ndarray:
    finite = np.isfinite(values)
    count = finite.sum(axis=axis)
    total = np.where(finite, values, 0.0).sum(axis=axis)
    out = np.full(count.shape, np.nan, dtype=np.float64)
    np.divide(total, count, out=out, where=count > 0)
    return out


def _window(data: dict, cfg: dict, tasks: list[int]) -> dict:
    P = cfg["phase1"]
    idx = _window_indices(data["step"], int(P["task_period"]), tasks)
    raw = np.asarray(data["unfit"], dtype=np.float64)[idx]
    raw_u = raw.mean(axis=0)
    u = np.maximum(raw_u, float(P["unfit_floor"]))
    metrics = {key: _finite_mean(np.asarray(data[key], dtype=np.float64)[idx], 0)
               for key in RUN_KEYS[1:] + LAYER_KEYS}
    return dict(index=idx, raw_u=raw_u, u=u, log_u=np.log10(u),
                unfit_sum=raw.sum(axis=0), unfit_rate=raw.mean(axis=0),
                n_records=int(len(idx)), metrics=metrics)


def _collect_divergences(cfg: dict, outdir: Path) -> dict[str, dict]:
    seeds = [int(v) for v in cfg["common"]["seeds"]]
    total, every = int(cfg["common"]["total_steps"]), int(cfg["common"]["lop_every"])
    return {arm: event for arm in ARM_ORDER
            if (event := _load_divergence_status(outdir, arm, seeds, total, every))
            is not None}


def _pair_check_final(cfg: dict, outdir: Path, complete: set[str]) -> dict:
    differences = []
    seeds = [int(v) for v in cfg["common"]["seeds"]]
    for group in (WIDTH5_ARMS, WIDTH100_ARMS):
        available = [a for a in group if a in complete]
        if not available:
            differences.append(dict(group=list(group), reason="no complete arm"))
            continue
        reference = available[0]
        for seed in seeds:
            with np.load(outdir / "logs" / f"{reference}_seed{seed}.npz",
                         allow_pickle=False) as z:
                expected = json.loads(str(z["state_hash_final"]))
            expected = {k: expected[k] for k in ("env.flip_state", "env.t")}
            for arm in available[1:]:
                with np.load(outdir / "logs" / f"{arm}_seed{seed}.npz",
                             allow_pickle=False) as z:
                    actual = json.loads(str(z["state_hash_final"]))
                actual = {k: actual[k] for k in ("env.flip_state", "env.t")}
                if actual != expected:
                    differences.append(dict(group=list(group), seed=seed,
                                            arm=arm, reason="env mismatch"))
    return dict(pass_=not differences, differences=differences,
                caveat=cfg["pairing"]["trajectory_caveat"])


def analyze(cfg: dict, outdir: Path, sanity: dict, elapsed: dict,
            divergences: dict[str, dict]) -> dict:
    seeds = [int(v) for v in cfg["common"]["seeds"]]
    P, G = cfg["phase1"], cfg["width5_gate"]
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
    main_verdict = classify_g0(onset["5M"], len(seeds), set(divergences))

    draws = np.random.default_rng(int(P["bootstrap_seed"])).integers(
        0, len(seeds), size=(int(P["bootstrap_B"]), len(seeds)))
    levels, arm_levels, level_rows = {}, {}, []
    for high, low in G["level_contrasts"]:
        label = "LEAKY" if high == "LR5" else "ELU"
        key = f"{high}_minus_{low}"
        if high not in complete or low not in complete:
            levels[key] = dict(status=NUMERIC_DIVERGENCE)
            level_rows.append(dict(kind="G1_CONTRAST", contrast=key,
                                   status=NUMERIC_DIVERGENCE, registered=0))
            continue
        values = windows[high]["5M"]["log_u"] - windows[low]["5M"]["log_u"]
        ci = _ci(cfg, values, draws)
        status = classify_level(
            float(ci["point"]), float(ci["percentile_ci_lo"]),
            float(ci["percentile_ci_hi"]),
            float(G["level_equivalence_resolution_dex"]), label)
        levels[key] = dict(status=status, seed_values=values.tolist(), ci=ci)
        level_rows.append(dict(
            kind="G1_CONTRAST", contrast=key, status=status, registered=0,
            resolution_dex=float(G["level_equivalence_resolution_dex"]),
            point=ci["point"], percentile_ci_lo=ci["percentile_ci_lo"],
            percentile_ci_hi=ci["percentile_ci_hi"],
            studentized_ci_lo=ci["studentized_ci_lo"],
            studentized_ci_hi=ci["studentized_ci_hi"],
            ci_degenerate=ci["ci_degenerate"],
            seed_values=json.dumps(values.tolist())))

    # G2 is deliberately arm-level and unpaired: width 5 and width 100 use
    # different generator bases.  No delta or paired statistic is emitted.
    for arm in complete:
        values = windows[arm]["5M"]["log_u"]
        ci = _ci(cfg, values, draws)
        arm_levels[arm] = dict(seed_values=values.tolist(), ci=ci)
        level_rows.append(dict(
            kind="G2_ARM_LEVEL", contrast=arm, status="REPORT_ONLY",
            registered=0, resolution_dex="", point=ci["point"],
            percentile_ci_lo=ci["percentile_ci_lo"],
            percentile_ci_hi=ci["percentile_ci_hi"],
            studentized_ci_lo=ci["studentized_ci_lo"],
            studentized_ci_hi=ci["studentized_ci_hi"],
            ci_degenerate=ci["ci_degenerate"],
            seed_values=json.dumps(values.tolist())))

    verdict_rows, mechanism_rows = [], []
    for arm in ARM_ORDER:
        width, label, _, alpha = REGISTERED_ARMS[arm]
        if arm not in complete:
            verdict_rows.append(dict(arm=arm, width=width, activation=label,
                                     act_alpha=alpha, status=NUMERIC_DIVERGENCE,
                                     main_verdict=main_verdict))
            continue
        cp1 = clopper_pearson(onset["1M"][arm], len(seeds))
        cp5 = clopper_pearson(onset["5M"][arm], len(seeds))
        w5 = windows[arm]["5M"]
        strict = w5["metrics"]["layer1_strict_dead"] / width
        submerged = w5["metrics"]["layer1_submerged"] / width
        verdict_rows.append(dict(
            arm=arm, width=width, activation=label, act_alpha=alpha,
            status="COMPLETE", n_onset_1m=onset["1M"][arm],
            cp95_1m_lo=cp1[0], cp95_1m_hi=cp1[1],
            n_onset_5m=onset["5M"][arm], cp95_5m_lo=cp5[0],
            cp95_5m_hi=cp5[1], U_1m_seed_values=json.dumps(
                windows[arm]["1M"]["u"].tolist()),
            U_5m_seed_values=json.dumps(w5["u"].tolist()),
            median_log10_U_1m=float(np.median(windows[arm]["1M"]["log_u"])),
            median_log10_U_5m=float(np.median(w5["log_u"])),
            median_submerged_frac_5m=float(np.median(submerged)),
            median_strict_dead_frac_5m=(float(np.median(strict))
                                        if label == "relu" else ""),
            main_verdict=main_verdict))
        for window_name in G["report_windows"]:
            w = windows[arm][window_name]
            row = dict(arm=arm, width=width, activation=label,
                       act_alpha=alpha, window=window_name, registered=0,
                       n_records=w["n_records"],
                       median_log10_U=float(np.median(w["log_u"])),
                       median_unfit_sum=float(np.median(w["unfit_sum"])),
                       median_unfit_rate=float(np.median(w["unfit_rate"])))
            for key, values in w["metrics"].items():
                row[f"median_{key.removeprefix('layer1_')}"] = (
                    float(np.nanmedian(values)) if np.isfinite(values).any() else "")
            row["median_submerged_frac"] = float(np.median(
                w["metrics"]["layer1_submerged"] / width))
            mechanism_rows.append(row)
    write_csv(outdir / "verdict.csv", verdict_rows)
    write_csv(outdir / "levels.csv", level_rows)
    write_csv(outdir / "mechanism.csv", mechanism_rows)
    _write_summary(outdir, main_verdict, verdict_rows, levels, sanity, divergences)
    return dict(main_verdict=main_verdict, onset=onset, levels_report_only=levels,
                width_arm_levels_report_only=arm_levels,
                divergences=divergences, elapsed_sec=elapsed)


def _write_summary(outdir: Path, verdict: str, rows: list[dict], levels: dict,
                   sanity: dict, divergences: dict) -> None:
    lines = [f"# {EXPERIMENT} summary", "", "## Registered verdict", "",
             f"- G0: **{verdict}**",
             "- G1 levels, G2 width levels, and all G3 mechanism metrics are REPORT_ONLY.",
             "- 0/20 means not observed through 5M; one-sided 95% upper bound 0.1391.",
             f"- Numeric divergence: {', '.join(sorted(divergences)) or 'none'}", "",
             "## Endpoints", "",
             "| arm | width | activation | onset 1M | onset 5M | median log10 U 5M | submerged frac 5M |",
             "|---|---:|---|---:|---:|---:|---:|"]
    for row in rows:
        if row["status"] != "COMPLETE":
            lines.append(f"| {row['arm']} | {row['width']} | {row['activation']} | — | — | — | {row['status']} |")
        else:
            lines.append(
                f"| {row['arm']} | {row['width']} | {row['activation']} | "
                f"{row['n_onset_1m']}/20 | {row['n_onset_5m']}/20 | "
                f"{row['median_log10_U_5m']:.6g} | "
                f"{row['median_submerged_frac_5m']:.6g} |")
    lines += ["", "## G1 level contrasts (REPORT_ONLY)", ""]
    for name, value in levels.items():
        lines.append(f"- {name}: **{value['status']}**")
    lines += ["", "## Logger semantics", "",
              "- `centered_eff_rank` is R_c after subtracting each activation column mean.",
              "- `mobility_weighted_wcos_abs` is the pairwise absolute cosine of input-weight rows weighted by m_i m_j.",
              "- Raw and floor-normalized signed/absolute variants are all retained in logs.",
              "- `mobility_tilde` and R_c/sum(tilde_m) are undefined (NaN) for LIN because a=1 makes (m-a)/(1-a) equal 0/0.",
              "- `submerged` equals p_hat==0 for every arm and is not an independent endpoint.",
              "", "## Sanity", ""]
    for key, value in sanity.items():
        if isinstance(value, dict) and "pass_" in value:
            lines.append(f"- {key}: **{'PASS' if value['pass_'] else 'FAIL'}**")
    (outdir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _provenance(cfg_path: Path, cfg: dict, outdir: Path, sanity: dict,
                analysis_result: dict, elapsed: dict, started: float) -> dict:
    spec_path = Path(ROOT) / str(cfg["spec"])
    if not spec_path.exists():
        raise FileNotFoundError(f"frozen repo spec missing: {spec_path}")
    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                           text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--short"], cwd=ROOT,
                                        text=True).splitlines()
    except (OSError, subprocess.CalledProcessError):
        git_hash, dirty = None, []
    names = ("verdict.csv", "levels.csv", "mechanism.csv", "summary.md",
             "config_used.yaml")
    hashes = {name: _sha_file(outdir / name) for name in names
              if (outdir / name).exists()}
    hashes.update({f"logs/{p.name}": _sha_file(p)
                   for p in sorted((outdir / "logs").glob("*.npz"))})
    return dict(experiment=EXPERIMENT,
                created=time.strftime("%Y-%m-%d %H:%M:%S %z"), command=sys.argv,
                elapsed_sec=round(time.time() - started, 3), arm_elapsed_sec=elapsed,
                cwd=os.getcwd(), python=sys.version, platform=platform.platform(),
                torch=torch.__version__, numpy=np.__version__,
                device=cfg["common"]["device"], git_hash=git_hash, git_dirty=dirty,
                config=str(cfg_path), config_sha256=_sha_file(cfg_path),
                spec=str(spec_path), spec_sha256=_sha_file(spec_path),
                sanity=sanity, analysis=analysis_result, output_sha256=hashes)


def run_full(cfg_path: Path, cfg: dict, device: str, outdir: Path, *,
             smoke: bool) -> dict:
    started = time.time()
    outdir.mkdir(parents=True, exist_ok=True)
    require_omp(cfg)
    total = SMOKE_STEPS if smoke else int(cfg["common"]["total_steps"])
    seeds = SMOKE_SEEDS if smoke else [int(v) for v in cfg["common"]["seeds"]]
    with (outdir / "config_used.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, allow_unicode=True, sort_keys=False)
    if smoke:
        preflight_result = {"pass_": True, "smoke": True}
    else:
        path = Path(ROOT) / PREFLIGHT_DIR / "preflight.json"
        if not path.exists():
            raise FileNotFoundError("run --preflight after freezing predictions")
        preflight_result = json.loads(path.read_text(encoding="utf-8"))
        if not preflight_result.get("pass_"):
            raise RuntimeError("saved preflight did not pass")
    elapsed, identities, divergences = {}, {}, {}
    for arm in ARM_ORDER:
        if _complete_arm_logs(outdir, arm, seeds, total,
                              int(cfg["common"]["lop_every"])):
            elapsed[arm] = 0.0
            identities[arm] = dict(pass_=True, resumed_from_logs=True)
            continue
        result = _run_arm(cfg, arm, device, outdir, seeds, total)
        elapsed[arm], identities[arm] = result["elapsed_sec"], result["sanity"]
        if result["status"] == NUMERIC_DIVERGENCE:
            divergences[arm] = result["divergence"]
    if smoke:
        payload = dict(pass_=bool(all(v.get("pass_") for v in identities.values())),
                       identities=identities, divergences=divergences,
                       elapsed_sec=elapsed)
        (outdir / "smoke_sanity.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8")
        return payload
    complete = set(ARM_ORDER) - set(divergences)
    sanity = dict(preflight=preflight_result,
                  final_pairing=_pair_check_final(cfg, outdir, complete),
                  exact_recorders=identities)
    if not sanity["final_pairing"]["pass_"]:
        raise RuntimeError("final within-width pairing sanity failed")
    result = analyze(cfg, outdir, sanity, elapsed, divergences)
    provenance = _provenance(cfg_path, cfg, outdir, sanity, result, elapsed, started)
    (outdir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")
    print(f"ALL DONE -> {outdir}", flush=True)
    return dict(sanity=sanity, analysis=result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/width5_gate_0901.yaml")
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
        raise ValueError("width5_gate is CPU-only")
    stage = ("preflight" if args.preflight else "smoke" if args.smoke else
             "analyze" if args.analyze_only else "full")
    validate_config(cfg, stage=stage)
    main_dir = Path(ROOT) / str(cfg["output"]["dir"])
    outdir = (Path(args.outdir).resolve() if args.outdir else
              Path(ROOT) / PREFLIGHT_DIR if args.preflight else
              Path(ROOT) / SMOKE_DIR if args.smoke else main_dir)
    if args.preflight:
        preflight(cfg, device, outdir)
    elif args.analyze_only:
        preflight_result = json.loads(
            (Path(ROOT) / PREFLIGHT_DIR / "preflight.json").read_text(
                encoding="utf-8"))
        divergences = _collect_divergences(cfg, outdir)
        complete = set(ARM_ORDER) - set(divergences)
        sanity = dict(preflight=preflight_result,
                      final_pairing=_pair_check_final(cfg, outdir, complete))
        analyze(cfg, outdir, sanity, {}, divergences)
    else:
        run_full(cfg_path, cfg, device, outdir, smoke=args.smoke)


if __name__ == "__main__":
    main()
