"""Dohare's hidden-layer-free linear baseline (``LIN0``) as a comparison arm.

Registered by ``specs/spec_lin0_base_0902.md``.

The published loss-of-plasticity code uses ``nn.Linear(input_size, 1)`` for its
linear baseline, with no hidden layer at all.  ``width5_gate_b_0901``'s ``LIN5``
is leaky(a=1.0) over five hidden units: the same function class, a different
parameterization.  This module runs the original construction so the registered
G0 labels can be re-derived against it.

Only ``LIN0`` and ``LIN0_lr03`` are trained.  The nonlinear arms are read from
the parent run's committed ``verdict.csv``; they are not re-run.
"""
from __future__ import annotations

import argparse
import ast
import copy
import csv
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

from . import width5_gate_0901 as base
from .common import ROOT, load_config, pick_device
from .dose_const_5m import clopper_pearson
from .envs import LTUTarget, SCREnv
from .mlp2_phase0 import _sha_file, require_omp, write_csv
from .mlp2_phase0b import _window_indices
from .ratchet_log import full_support_ro, teacher_f64
from .train import make_gens
from .width5_gate_b_0901 import _rectangular, classify_seed_sign


EXPERIMENT = "lin0_base_0902"
PREFLIGHT_DIR = "results/_preflight_lin0_base_0902"
SMOKE_DIR = "results/_smoke_lin0_base_0902"
ARM_ORDER = ("LIN0", "LIN0_lr03")
RUN_KEYS = ("unfit", "eval_loss_exact", "signal_var", "residual_var")
SMOKE_STEPS = 5_000
SMOKE_SEEDS = [0, 1]
WINDOWS = {"5m": "late_tasks_5m", "1m": "window_1m_tasks"}
WINDOW_LABEL = {"5m": "task 491-500", "1m": "task 91-100"}


class SanityError(RuntimeError):
    """Raised when a registered pre-execution check fails."""


# ---------------------------------------------------------------------------
# Net
# ---------------------------------------------------------------------------
class VecLinear0:
    """``yhat = a . x + c`` with no hidden layer, vectorized over runs.

    Mirrors ``lop/nets/linear.py``'s ``MyLinear``: a single ``nn.Linear(d, 1)``
    initialized with ``kaiming_uniform_(nonlinearity='linear')`` and zero bias.
    PyTorch's rule for that call is ``U(-sqrt(3/fan_in), +sqrt(3/fan_in))``,
    which is exactly ``envs.kaiming_mlp_params``'s readout rule with ``h -> d``.
    """

    def __init__(self, R: int, d: int, gen, device):
        self.R, self.d = int(R), int(d)
        self.bound = math.sqrt(3.0 / self.d)
        self.a = ((torch.rand(self.R, self.d, generator=gen, device=device)
                   * 2 - 1) * self.bound)
        self.c = torch.zeros(self.R, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.a * x).sum(dim=1) + self.c

    def sgd_step(self, lr: torch.Tensor, x: torch.Tensor,
                 delta: torch.Tensor) -> None:
        """Squared-error gradients with the repository's factor-2 convention."""
        d2 = 2.0 * delta
        self.a -= lr[:, None] * (d2[:, None] * x)
        self.c -= lr * d2

    def state_dict(self) -> dict:
        return {"a": self.a.clone(), "c": self.c.clone()}


# ---------------------------------------------------------------------------
# Setup and training
# ---------------------------------------------------------------------------
def _arm(cfg: dict, name: str) -> dict:
    return next(a for a in cfg["arms"] if a["name"] == name)


def preregistration_missing(cfg: dict) -> list[str]:
    pre = cfg["preregistration"]
    return [f"preregistration.{key}" for key in
            ("frozen", "repo_spec_committed", "predictions_confirmed",
             "execution_authorized") if not pre.get(key)]


def validate_config(cfg: dict, *, stage: str) -> None:
    if stage not in {"implementation", "preflight", "smoke", "full", "analyze"}:
        raise ValueError(f"unknown stage {stage!r}")
    A, C, L = cfg["condA"], cfg["common"], cfg["lin0"]
    if [a["name"] for a in cfg["arms"]] != list(ARM_ORDER):
        raise ValueError(f"arms must be ordered as {ARM_ORDER}")
    if (int(A["m"]), int(A["f"]), int(A["target_hidden"])) != (20, 15, 100):
        raise ValueError("condA geometry differs from the frozen design")
    if float(A["beta"]) != 0.7 or [int(v) for v in A["T_values"]] != [10000]:
        raise ValueError("condA teacher/period differ from the frozen design")
    if (int(C["total_steps"]) != 5_000_000
            or [int(v) for v in C["seeds"]] != list(range(20))
            or int(C["generator_offset"]) != 202609011921
            or int(C["lop_every"]) != 1000
            or str(C["device"]) != "cpu"):
        raise ValueError("step/seed/offset/device design changed")
    lrs = {a["name"]: float(a["lr"]) for a in cfg["arms"]}
    if lrs != {"LIN0": 0.01, "LIN0_lr03": 0.03}:
        raise ValueError("registered per-arm learning rates changed")
    if any(int(a["generator_width_basis"]) != 5 for a in cfg["arms"]):
        raise ValueError("generator width basis must stay 5 (spec 2.2)")
    if (list(L["comparison_arms"]) != ["R5", "LR5", "E5"]
            or str(L["baseline_arm"]) != "LIN0"
            or float(L["cp_alpha"]) != 0.05
            or [float(v) for v in L["tight_band"]] != [0.20, 0.80]):
        raise ValueError("registered judgement design changed")
    if stage in {"preflight", "full", "analyze"}:
        missing = preregistration_missing(cfg)
        if missing:
            raise ValueError(f"preregistration incomplete: {missing}")
        if (str(cfg["preregistration"]["prediction_B1"])
                != "BASELINE_CONSTRUCTION_IMMATERIAL"):
            raise ValueError("frozen prediction B1 changed")


def setup_lin0(cfg: dict, arm_cfg: dict, device: str,
               seeds: list[int]) -> dict:
    """Build the LIN0 arm on the parent run's input/teacher realization.

    ``make_gens`` splits ``init`` / ``input`` / ``teacher`` / ``eval`` into
    separate generators keyed on ``SEED_BASE["A"] + width + offset``.  Passing
    the registered basis 5 therefore reproduces the width-5 arms' stream while
    LIN0's smaller ``init`` draw stays confined to its own generator.
    """
    A, C = cfg["condA"], cfg["common"]
    m, f = int(A["m"]), int(A["f"])
    R = len(seeds)
    gens = make_gens("A", int(arm_cfg["generator_width_basis"]), device,
                     offset=int(C["generator_offset"]))
    period = torch.tensor([int(cfg["phase1"]["task_period"])] * R,
                          dtype=torch.long)
    env = SCREnv(R, m, f, period, gens["input"], device)
    teacher = LTUTarget(R, m, int(A["target_hidden"]), float(A["beta"]),
                        gens["teacher"], device)
    net = VecLinear0(R, m, gens["init"], device)
    lr = torch.full((R,), float(arm_cfg["lr"]), device=device)
    return dict(exp="A", arm=str(arm_cfg["name"]), R=R, d=m, env=env,
                teacher=teacher, net=net, lr=lr, period=period,
                seeds=[int(v) for v in seeds], device=device, gens=gens,
                generator_offset=int(C["generator_offset"]),
                generator_width_basis=int(arm_cfg["generator_width_basis"]))


def exact_record_lin0(st: dict) -> tuple[dict, dict]:
    """32-point exact-support record.  Same definitions as the parent's run block."""
    net = st["net"]
    with torch.no_grad():
        X = full_support_ro(st["env"]).double()
        y = teacher_f64(st["teacher"], X)
        yhat = torch.einsum("rd,prd->pr", net.a.double(), X) + net.c.double()
        residual = yhat - y
        signal_var = y.var(dim=0, unbiased=False)
        residual_var = residual.var(dim=0, unbiased=False)
        run = dict(signal_var=signal_var, residual_var=residual_var,
                   unfit=residual_var / signal_var,
                   eval_loss_exact=residual.square().mean(dim=0))
        sanity = dict(
            run_finite=bool(all(torch.isfinite(v).all() for v in run.values())
                            and (signal_var > 0).all()),
            support=int(X.shape[0]))
        return run, sanity


class Lin0Recorder:
    """Probe recorder; only run-level quantities exist for a net with no units."""

    def __init__(self, steps: list[int], st: dict):
        self.steps = np.asarray(sorted(set(int(v) for v in steps)),
                                dtype=np.int64)
        self.index = {int(v): i for i, v in enumerate(self.steps)}
        n, R = len(self.steps), st["R"]
        self.run = {key: np.empty((n, R), dtype=np.float64) for key in RUN_KEYS}
        self.failures: list[dict] = []

    def __call__(self, st: dict, step: int) -> None:
        i = self.index.get(int(step))
        if i is None:
            return
        run, sanity = exact_record_lin0(st)
        if not sanity["run_finite"]:
            self.failures.append(dict(step=int(step), **sanity))
        for key in RUN_KEYS:
            self.run[key][i] = run[key].detach().cpu().numpy()

    def sanity(self) -> dict:
        return dict(pass_=not self.failures, failures=self.failures)


def write_arm_logs(outdir: Path, arm: str, st: dict,
                   rec: Lin0Recorder) -> list[Path]:
    paths = []
    (outdir / "logs").mkdir(parents=True, exist_ok=True)
    for ri, seed in enumerate(st["seeds"]):
        path = outdir / "logs" / f"{arm}_seed{seed}.npz"
        np.savez_compressed(
            path, step=rec.steps, arm=np.array(arm), seed=np.int64(seed),
            width=np.int64(0), activation=np.array("linear0"),
            activation_label=np.array("linear0"),
            lr=np.float64(float(st["lr"][ri])),
            generator_offset=np.int64(st["generator_offset"]),
            generator_width_basis=np.int64(st["generator_width_basis"]),
            task_period=np.int64(int(st["period"][ri])),
            **{key: rec.run[key][:, ri] for key in RUN_KEYS})
        paths.append(path)
    return paths


def run_arm(cfg: dict, arm: str, device: str, outdir: Path,
            seeds: list[int], total: int) -> dict:
    st = setup_lin0(cfg, _arm(cfg, arm), device, seeds)
    every = int(cfg["common"]["lop_every"])
    probes = list(range(0, total + 1, every))
    if probes[-1] != total:
        probes.append(total)
    rec = Lin0Recorder(probes, st)
    probe_set = set(probes)
    print(f"[{arm}] linear0 d={st['d']} lr={float(st['lr'][0])} "
          f"seeds={seeds} steps={total:,}", flush=True)
    started = time.time()
    net, env, teacher = st["net"], st["env"], st["teacher"]
    for step in range(total):
        if step in probe_set:
            rec(st, step)
        x = env.step()
        y = teacher(x)
        delta = net.forward(x) - y
        net.sgd_step(st["lr"], x, delta)
    if total in probe_set:
        rec(st, total)
    elapsed = time.time() - started
    sanity = rec.sanity()
    if not sanity["pass_"]:
        raise SanityError(f"{arm} exact-support record went non-finite: {sanity}")
    write_arm_logs(outdir, arm, st, rec)
    print(f"[{arm}] complete in {elapsed:.1f}s", flush=True)
    return dict(status="COMPLETE", elapsed_sec=elapsed, sanity=sanity)


# ---------------------------------------------------------------------------
# Parent inputs
# ---------------------------------------------------------------------------
def load_parent(cfg: dict) -> dict:
    P = cfg["parent"]
    parent = Path(ROOT) / str(P["dir"])
    for name, want in (("verdict.csv", P["verdict_sha256"]),
                       ("provenance.json", P["provenance_sha256"])):
        path = parent / name
        if not path.exists():
            raise SanityError(f"registered parent input missing: {path}")
        got = _sha_file(path)
        if got != str(want):
            raise SanityError(
                f"parent {name} sha256 {got} != registered {want}")
    with (parent / "verdict.csv").open(newline="", encoding="utf-8") as fh:
        rows = {row["arm"]: row for row in csv.DictReader(fh)}
    values = {
        arm: {w: [float(v) for v in
                  ast.literal_eval(rows[arm][f"U_{w}_seed_values"])]
              for w in WINDOWS}
        for arm in rows
    }
    return dict(dir=parent, values=values,
                provenance=json.loads(
                    (parent / "provenance.json").read_text(encoding="utf-8")))


def parent_config(cfg: dict) -> dict:
    path = Path(ROOT) / str(cfg["parent"]["dir"]) / "config_used.yaml"
    if not path.exists():
        raise SanityError(f"parent config missing: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Sanity gates
# ---------------------------------------------------------------------------
def s_share(cfg: dict, device: str) -> dict:
    """The paired comparison only stands if LIN0 sees the same realization."""
    seeds = [int(v) for v in cfg["common"]["seeds"]]
    st = setup_lin0(cfg, _arm(cfg, "LIN0"), device, seeds)
    pcfg = copy.deepcopy(parent_config(cfg))
    pcfg["common"]["seeds"] = seeds
    ref = base.setup_arm_width(pcfg, base._arm(pcfg, str(cfg["sanity"]["s_share_arm"])),
                               device)
    with torch.no_grad():
        Xa = full_support_ro(st["env"]).double()
        Xb = full_support_ro(ref["env"]).double()
        ya = teacher_f64(st["teacher"], Xa)
        yb = teacher_f64(ref["teacher"], Xb)
        flip_equal = bool(torch.equal(st["env"].flip_state,
                                      ref["env"].flip_state))
        x_equal = bool(torch.equal(Xa, Xb))
        y_equal = bool(torch.equal(ya, yb))
    return dict(name="S_share", gating=True,
                pass_=bool(x_equal and y_equal and flip_equal),
                reference_arm=str(cfg["sanity"]["s_share_arm"]),
                support_shape=list(Xa.shape), exact_X_equal=x_equal,
                teacher_y_equal=y_equal, flip_state_equal=flip_equal,
                generator_width_basis=st["generator_width_basis"],
                generator_offset=st["generator_offset"])


def s_lin0(cfg: dict, device: str) -> dict:
    """LIN0 must be exactly affine and must own no hidden tensors."""
    seeds = [int(v) for v in cfg["common"]["seeds"]]
    st = setup_lin0(cfg, _arm(cfg, "LIN0"), device, seeds)
    net, tol = st["net"], float(cfg["sanity"]["s_lin0_tol"])
    gen = torch.Generator(device=device)
    gen.manual_seed(0)
    def _affine_error(dtype: torch.dtype) -> float:
        a, c = net.a.to(dtype), net.c.to(dtype)
        y1, y2 = x1.to(dtype), x2.to(dtype)
        forward = lambda z: (a * z).sum(dim=1) + c  # noqa: E731
        mixed = forward(lam * y1 + (1 - lam) * y2).double()
        expected = (lam * forward(y1).double()
                    + (1 - lam) * forward(y2).double())
        scale = torch.maximum(mixed.abs().max(),
                              expected.abs().max()).clamp_min(1e-300)
        return float(((mixed - expected).abs().max() / scale).item())

    with torch.no_grad():
        x1 = torch.rand(st["R"], st["d"], generator=gen, device=device)
        x2 = torch.rand(st["R"], st["d"], generator=gen, device=device)
        lam = 0.37
        # The registered 1e-12 tolerance is below float32 resolution, so it can
        # only be a statement about the *definition* of the net, not about the
        # online learner's rounding.  Gate on the float64 path -- the same one
        # exact_record_lin0 evaluates unfit on -- and report the float32 value
        # beside it so the rounding scale stays visible.
        affine_error = _affine_error(torch.float64)
        affine_error_float32 = _affine_error(torch.float32)
    hidden_attrs = [name for name in ("Ws", "bs", "v", "W", "b", "hidden")
                    if hasattr(net, name)]
    return dict(name="S_lin0", gating=True,
                pass_=bool(affine_error <= tol and not hidden_attrs),
                tolerance=tol, affine_error=affine_error,
                affine_error_float32=affine_error_float32,
                gate_precision="float64",
                gate_precision_note=(
                    "1e-12 is unreachable in float32 (eps ~1.2e-7); the check "
                    "targets the net definition, evaluated where unfit is"),
                hidden_tensors_present=hidden_attrs,
                parameter_count=int(net.a.numel() // net.R + 1))


def s_init(cfg: dict, device: str) -> dict:
    """kaiming_uniform(nonlinearity='linear') on fan_in = d, zero bias."""
    S = cfg["sanity"]
    seeds = [int(v) for v in cfg["common"]["seeds"]]
    st = setup_lin0(cfg, _arm(cfg, "LIN0"), device, seeds)
    net = st["net"]
    bound = math.sqrt(3.0 / st["d"])
    a = net.a.double().flatten()
    qs = [float(v) for v in S["s_init_quantiles"]]
    got = torch.quantile(a, torch.tensor(qs, dtype=a.dtype)).tolist()
    want = [(2 * q - 1) * bound for q in qs]
    tol = float(S["s_init_quantile_tol"])
    errors = [abs(g - w) for g, w in zip(got, want)]
    in_range = bool(a.min() >= -bound and a.max() <= bound)
    return dict(name="S_init", gating=True,
                pass_=bool(in_range and max(errors) <= tol
                           and bool((net.c == 0).all())),
                bound=bound, formula=str(S["s_init_bound_formula"]),
                n_draws=int(a.numel()), min=float(a.min()), max=float(a.max()),
                in_range=in_range, quantiles=qs, observed=got, expected=want,
                max_abs_error=max(errors), tolerance=tol,
                bias_all_zero=bool((net.c == 0).all()))


def s_floor(cfg: dict) -> dict:
    want = float(cfg["sanity"]["s_floor_expected"])
    got = float(cfg["phase1"]["unfit_floor"])
    parent_floor = float(parent_config(cfg)["phase1"]["unfit_floor"])
    return dict(name="S_floor", gating=True,
                pass_=bool(got == want == parent_floor),
                configured=got, expected=want, parent=parent_floor,
                inherited_from="results/gate_dose_0830")


def _verify_parent_logs(cfg: dict, parent: dict, arms: list[str],
                        seeds: list[int]) -> dict:
    """Pin the uncommitted reference logs to the committed provenance hashes."""
    recorded = parent["provenance"].get("output_sha256", {})
    checked, bad, missing = 0, [], []
    for arm in arms:
        for seed in seeds:
            name = f"logs/{arm}_seed{seed}.npz"
            path = parent["dir"] / name
            if name not in recorded:
                missing.append(dict(name=name, reason="not in provenance"))
            elif not path.exists():
                missing.append(dict(name=name, reason="file absent locally"))
            elif _sha_file(path) != recorded[name]:
                bad.append(name)
            else:
                checked += 1
    return dict(pass_=bool(checked and not bad and not missing),
                verified=checked, mismatched=bad, missing=missing,
                note="reference npz are not committed; hashes come from the "
                     "parent provenance.json, which is")


def s0_prime_w5(cfg: dict, device: str, outdir: Path) -> dict:
    """Replay the parent's width-5 arms and demand a bit-identical prefix."""
    S = cfg["sanity"]
    parent = load_parent(cfg)
    seeds = [int(v) for v in cfg["common"]["seeds"]]
    arms = [str(v) for v in S["s0_prime_w5_arms"]]
    metrics = [str(v) for v in S["s0_prime_w5_metrics"]]
    total = int(S["s0_prime_steps"])
    reference_check = _verify_parent_logs(cfg, parent, arms, seeds)

    pcfg = copy.deepcopy(parent_config(cfg))
    replay = {}
    differences = list(reference_check["mismatched"]) + [
        dict(**row, reason="reference log unusable")
        for row in reference_check["missing"]]
    for arm in arms:
        result = base._run_arm(pcfg, arm, device, outdir, seeds, total)
        if result["status"] != "COMPLETE":
            raise SanityError(f"S0prime-w5 replay diverged for {arm}")
        arm_diff = []
        with np.load(outdir / "logs" / f"{arm}_seed{seeds[0]}.npz",
                     allow_pickle=False) as z:
            steps = np.asarray(z["step"], dtype=np.int64).tolist()
        for seed in seeds:
            ours = outdir / "logs" / f"{arm}_seed{seed}.npz"
            ref = parent["dir"] / "logs" / f"{arm}_seed{seed}.npz"
            if not ref.exists():
                arm_diff.append(dict(seed=seed, reason="reference missing"))
                continue
            arm_diff.extend(
                dict(seed=seed, **row) for row in
                base._compare_replay_log(ours, ref, total, metrics))
        with np.load(outdir / "logs" / f"{arm}_seed{seeds[0]}.npz",
                     allow_pickle=False) as z:
            replay[arm] = dict(
                steps=steps,
                unfit_seed0=np.asarray(z["unfit"], dtype=np.float64).tolist(),
                eval_loss_exact_seed0=np.asarray(
                    z["eval_loss_exact"], dtype=np.float64).tolist())
        differences.extend(dict(arm=arm, **row) for row in arm_diff)
    return dict(name="S0prime_w5", gating=True, pass_=not differences,
                total_steps=total, seeds=seeds, arms=arms, metrics=metrics,
                generator_offset=int(cfg["common"]["generator_offset"]),
                reference=str(parent["dir"]),
                reference_hash_check=reference_check,
                replay_reference_values=replay, differences=differences,
                fresh_clone_reproducible=False,
                note="parent logs/*.npz are gitignored; the replay values "
                     "recorded here are the committed reference")


def s0_prime_w100(cfg: dict, device: str, outdir: Path) -> dict:
    """The parent's own S0' (width-100 arms against gate_dose_0830)."""
    S = cfg["sanity"]
    pcfg = copy.deepcopy(parent_config(cfg))
    pcfg["sanity"]["s0_prime_steps"] = int(S["s0_prime_steps"])
    pcfg["sanity"]["s0_prime_seeds"] = [int(v) for v in
                                        S["s0_prime_w100_seeds"]]
    pcfg["sanity"]["s0_prime_generator_offset"] = int(
        S["s0_prime_w100_generator_offset"])
    pcfg["sanity"]["s0_prime_reference"] = str(S["s0_prime_w100_reference"])
    pcfg["sanity"]["s0_prime_arm_map"] = dict(S["s0_prime_w100_arm_map"])
    result = base._s0_replay(pcfg, device, outdir)
    result.update(name="S0prime_w100", gating=True,
                  fresh_clone_reproducible=False,
                  note="gate_dose_0830 logs/*.npz are gitignored too")
    return result


def preflight(cfg: dict, device: str, outdir: Path) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    require_omp(cfg)
    checks = {}
    checks["S_share"] = s_share(cfg, device)
    checks["S_lin0"] = s_lin0(cfg, device)
    checks["S_init"] = s_init(cfg, device)
    checks["S_floor"] = s_floor(cfg)
    checks["S0prime_w5"] = s0_prime_w5(cfg, device, outdir / "s0prime_w5")
    checks["S0prime_w100"] = s0_prime_w100(cfg, device,
                                           outdir / "s0prime_w100")
    checks["S_mech_na"] = dict(
        name="S_mech_na", gating=False, pass_=True,
        rule="leave the three P-3 axes empty; LIN0 has no hidden units",
        axes=["median_s", "mobility", "centered_eff_rank"])
    result = dict(pass_=all(v["pass_"] for v in checks.values()
                            if v.get("gating")),
                  **checks)
    for name, record in checks.items():
        gate = "gate" if record.get("gating") else "report"
        print(f"[{name}] {'PASS' if record['pass_'] else 'FAIL'} ({gate})",
              flush=True)
    (outdir / "preflight.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")
    if not result["pass_"]:
        failed = [k for k, v in checks.items()
                  if v.get("gating") and not v["pass_"]]
        raise SanityError(f"preflight failed: {failed}")
    return result


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def _load_arm(cfg: dict, outdir: Path, arm: str) -> dict:
    seeds = [int(v) for v in cfg["common"]["seeds"]]
    stacked, steps = {key: [] for key in RUN_KEYS}, None
    for seed in seeds:
        with np.load(outdir / "logs" / f"{arm}_seed{seed}.npz",
                     allow_pickle=False) as z:
            if steps is None:
                steps = np.asarray(z["step"], dtype=np.int64)
            for key in RUN_KEYS:
                stacked[key].append(np.asarray(z[key], dtype=np.float64))
    return dict(step=steps,
                **{key: np.stack(stacked[key], axis=1) for key in RUN_KEYS})


def window_values(cfg: dict, data: dict, window: str) -> np.ndarray:
    """Per-seed window mean of raw unfit, matching the parent's `_window`."""
    P = cfg["phase1"]
    idx = _window_indices(data["step"], int(P["task_period"]),
                          [int(v) for v in P[WINDOWS[window]]])
    return np.asarray(data["unfit"], dtype=np.float64)[idx].mean(axis=0)


def _finite(value: float) -> bool:
    return bool(np.isfinite(value))


def sign_test(cfg: dict, arm_values: list[float], base_values: list[float],
              arm: str, window: str) -> dict:
    L = cfg["lin0"]
    valid = [i for i in range(len(arm_values))
             if _finite(arm_values[i]) and _finite(base_values[i])]
    k = sum(1 for i in valid if arm_values[i] > base_values[i])
    record = classify_seed_sign(
        arm, k, len(valid), alpha=float(L["cp_alpha"]),
        tight_band=tuple(float(v) for v in L["tight_band"]))
    record.update(
        registered=1, window=window, window_tasks=WINDOW_LABEL[window],
        baseline=str(L["baseline_arm"]),
        ties=sum(1 for i in valid if arm_values[i] == base_values[i]),
        excluded_seed_indices=[i for i in range(len(arm_values))
                               if i not in valid],
        above_seed_indices=[i for i in valid
                            if arm_values[i] > base_values[i]])
    return record


def classify_g_base(cfg: dict, signs: dict) -> dict:
    legacy = cfg["lin0"]["legacy_labels"]
    rows, changed = [], []
    for window in WINDOWS:
        for arm in cfg["lin0"]["comparison_arms"]:
            was = str(legacy[window][arm])
            now = signs[(arm, window)]["status"]
            rows.append(dict(arm=arm, window=window,
                             window_tasks=WINDOW_LABEL[window],
                             legacy_label=was, lin0_label=now,
                             match=bool(was == now)))
            if was != now:
                changed.append(f"{arm}@{window}")
    verdict = ("BASELINE_CONSTRUCTION_IMMATERIAL" if not changed
               else "BASELINE_CONSTRUCTION_MATERIAL")
    return dict(verdict=verdict, changed=changed, rows=rows,
                legacy_baseline=str(cfg["lin0"]["legacy_baseline_arm"]),
                prediction_B1=str(cfg["preregistration"]["prediction_B1"]),
                prediction_hit=bool(
                    verdict == str(cfg["preregistration"]["prediction_B1"])))


def analyze(cfg: dict, outdir: Path, sanity: dict, elapsed: dict) -> dict:
    L = cfg["lin0"]
    parent = load_parent(cfg)
    ours = {arm: _load_arm(cfg, outdir, arm) for arm in ARM_ORDER}
    windows = {arm: {w: window_values(cfg, ours[arm], w).tolist()
                     for w in WINDOWS} for arm in ARM_ORDER}
    for arm, values in parent["values"].items():
        windows.setdefault(arm, values)

    baseline = str(L["baseline_arm"])
    signs = {(arm, w): sign_test(cfg, windows[arm][w], windows[baseline][w],
                                 arm, w)
             for w in WINDOWS for arm in L["comparison_arms"]}
    g_base = classify_g_base(cfg, signs)

    threshold = float(L["collapse_threshold"])
    report_arms = [baseline, "LIN0_lr03", str(L["legacy_baseline_arm"])]
    levels = []
    for arm in report_arms:
        for w in WINDOWS:
            values = np.asarray(windows[arm][w], dtype=np.float64)
            floor = float(cfg["phase1"]["unfit_floor"])
            paired = (values - np.asarray(windows[baseline][w],
                                          dtype=np.float64))
            k = int(np.sum(values >= threshold))
            lo, hi = clopper_pearson(k, int(values.size),
                                     float(L["cp_alpha"]))
            levels.append(dict(
                metric="level", registered=0, arm=arm, window=w,
                window_tasks=WINDOW_LABEL[w],
                median_log10_U=float(np.median(
                    np.log10(np.maximum(values, floor)))),
                median_unfit=float(np.median(values)),
                median_paired_diff_vs_LIN0=float(np.median(paired)),
                collapse_k=k, collapse_n=int(values.size),
                collapse_cp95_lo=lo, collapse_cp95_hi=hi,
                seed_values=values.tolist()))

    verdict_rows = [dict(
        metric="G0prime", registered=1, arm=r["arm"], window=r["window"],
        window_tasks=r["window_tasks"], baseline=r["baseline"], k=r["k"],
        n=r["n"], rate=r["rate"], cp95_lo=r["cp95_lo"], cp95_hi=r["cp95_hi"],
        status=r["status"], ties=r["ties"],
        excluded_seed_indices=r["excluded_seed_indices"],
        above_seed_indices=r["above_seed_indices"],
        g_base=g_base["verdict"])
        for r in (signs[(a, w)] for w in WINDOWS
                  for a in L["comparison_arms"])]
    write_csv(outdir / "verdict.csv", _rectangular(verdict_rows))
    write_csv(outdir / "g_base.csv", _rectangular(g_base["rows"]))
    write_csv(outdir / "levels.csv", _rectangular(levels))
    write_csv(outdir / "mechanism.csv", _rectangular([
        dict(arm=arm, window=w, registered=0, hidden_units=0,
             median_s="", mobility="", centered_eff_rank="",
             note="undefined: LIN0 has no hidden units (S_mech_na)")
        for arm in (baseline, "LIN0_lr03") for w in WINDOWS]))
    summary = render_summary(cfg, signs, g_base, levels, sanity, elapsed)
    (outdir / "summary.md").write_text(summary, encoding="utf-8")
    return dict(g_base=g_base, signs={f"{a}@{w}": v for (a, w), v
                                      in signs.items()},
                levels=levels, windows=windows)


def _fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def render_summary(cfg: dict, signs: dict, g_base: dict, levels: list[dict],
                   sanity: dict, elapsed: dict) -> str:
    L = cfg["lin0"]
    lines = [
        f"# {EXPERIMENT} summary", "",
        "## Registered verdict", "",
        f"- G-base: **{g_base['verdict']}**",
        f"- 事前予測 B1 = `{g_base['prediction_B1']}` → "
        f"**{'的中' if g_base['prediction_hit'] else '外れ'}**。"
        "**ただし盲の予言ではない**（`LIN5` 版の結果を見たあとに立てた条件付き予測。"
        "spec §1・§6）",
        "- **`PHENOMENON3_NOT_REPRODUCED` は上書きしていない。** それは `LIN5` を"
        "相手にした登録判定であり、本走はその外部妥当性を測るもの（spec §5）",
        "- `LIN0` は隠れ層ゼロの単層線形回帰（原典 `MyLinear` 対応）、"
        "`LIN5` は leaky($a$=1.0)・隠れ 5 ユニット。**別物である**",
        "- **1M 窓の格は非対称**: `LIN0` 相手は事前登録、`LIN5` 相手は事後登録"
        "（`7d77a90`）",
        "",
        "## G0' 対 `LIN0`（Clopper–Pearson 95%）", "",
        "| 腕 | 窓 | k | n | 除外 | 同値 | CP95 | ラベル |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for window in WINDOWS:
        for arm in L["comparison_arms"]:
            r = signs[(arm, window)]
            lines.append(
                f"| `{arm}` | {r['window_tasks']} | **{r['k']}** | {r['n']} | "
                f"{len(r['excluded_seed_indices'])} | {r['ties']} | "
                f"[{_fmt(r['cp95_lo'])}, {_fmt(r['cp95_hi'])}] | "
                f"**{r['status']}** |")
    lines += ["", "## G-base（6 ラベルの一致）", "",
              "| 腕 | 窓 | `LIN5` 版 | `LIN0` 版 | 一致 |",
              "|---|---|---|---|---|"]
    for row in g_base["rows"]:
        lines.append(f"| `{row['arm']}` | {row['window_tasks']} | "
                     f"{row['legacy_label']} | {row['lin0_label']} | "
                     f"{'YES' if row['match'] else '**NO**'} |")
    if g_base["changed"]:
        lines += ["", f"動いたラベル: **{', '.join(g_base['changed'])}**"]
    lines += ["", "## 水準（報告のみ）", "",
              "| 腕 | 窓 | median log10 U | 対 `LIN0` 対応差の中央値 | "
              "完全崩壊 k/n | CP95 |",
              "|---|---|---:|---:|---:|---|"]
    for row in levels:
        lines.append(
            f"| `{row['arm']}` | {row['window_tasks']} | "
            f"{_fmt(row['median_log10_U'])} | "
            f"{row['median_paired_diff_vs_LIN0']:+.4f} | "
            f"{row['collapse_k']}/{row['collapse_n']} | "
            f"[{_fmt(row['collapse_cp95_lo'])}, "
            f"{_fmt(row['collapse_cp95_hi'])}] |")
    lines += [
        "",
        "`LIN0_lr03`（lr 0.03・原典 step_size 先頭値）は**報告のみで判定に入れない**。",
        "", "## Sanity", ""]
    for key, record in sanity.items():
        if not isinstance(record, dict) or "pass_" not in record:
            continue
        gate = "gate" if record.get("gating") else "report"
        lines.append(f"- **{key}**: {'PASS' if record['pass_'] else 'FAIL'} "
                     f"({gate})")
    lines += [
        "",
        "**S0′ は fresh clone では回せない。** 参照 npz（親走・`gate_dose_0830` とも）"
        "が `.gitignore` でローカルのみだからである。replay 側の値は "
        "`preflight.json` に記録して commit した（spec §4）。",
        "", "## 引用上の注意", "",
        "- **`LIN0` を「原典の Linear ベースライン」と呼ぶときは、step size と環境"
        "（`flip_one` の有無）が原典の図と一致している保証はないことを併記する**",
        "- **`LIN` 系を「線形ネットワーク」と呼ぶときは実装を併記する**"
        "（`LIN0` = 単層／`LIN5` = leaky($a$=1.0) の隠れ 5）",
        "- **完全崩壊カウントと $k'$ を「LoP の発症率」と呼ばない**",
        "- **`LIN0_lr03` の数値を判定に使わない**",
        "- スコープは condA・$T=10^4$・batch 1・seed 20・5M。"
        "`LIN0` は lr 0.01、`LIN0_lr03` は lr 0.03",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Provenance and CLI
# ---------------------------------------------------------------------------
def _provenance(cfg_path: Path, cfg: dict, outdir: Path, sanity: dict,
                analysis: dict, elapsed: dict, started: float) -> dict:
    spec_path = Path(ROOT) / str(cfg["spec"])
    if not spec_path.exists():
        raise SanityError(f"frozen repo spec missing: {spec_path}")
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--short"], cwd=ROOT, text=True).splitlines()
    except (OSError, subprocess.CalledProcessError):
        git_hash, dirty = None, []
    names = ("verdict.csv", "g_base.csv", "levels.csv", "mechanism.csv",
             "summary.md", "config_used.yaml")
    hashes = {name: _sha_file(outdir / name) for name in names
              if (outdir / name).exists()}
    hashes.update({f"logs/{p.name}": _sha_file(p)
                   for p in sorted((outdir / "logs").glob("*.npz"))})
    return dict(
        experiment=EXPERIMENT, created=time.strftime("%Y-%m-%d %H:%M:%S %z"),
        command=sys.argv, elapsed_sec=round(time.time() - started, 3),
        arm_elapsed_sec=elapsed, cwd=os.getcwd(), python=sys.version,
        platform=platform.platform(), device=str(cfg["common"]["device"]),
        git_hash=git_hash, git_dirty=dirty, config=str(cfg_path),
        config_sha256=_sha_file(cfg_path), spec=str(spec_path),
        spec_sha256=_sha_file(spec_path),
        arm_learning_rates={a["name"]: float(a["lr"]) for a in cfg["arms"]},
        preregistration=dict(cfg["preregistration"]), sanity=sanity,
        analysis=dict(g_base=analysis["g_base"]), output_sha256=hashes)


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
            raise SanityError("run --preflight first")
        preflight_result = json.loads(path.read_text(encoding="utf-8"))
        if not preflight_result.get("pass_"):
            raise SanityError("saved preflight did not pass")

    elapsed = {}
    for arm in ARM_ORDER:
        result = run_arm(cfg, arm, device, outdir, seeds, total)
        elapsed[arm] = result["elapsed_sec"]
    if smoke:
        (outdir / "smoke_sanity.json").write_text(
            json.dumps(dict(pass_=True, elapsed_sec=elapsed), indent=2,
                       default=str), encoding="utf-8")
        print(f"SMOKE DONE -> {outdir}", flush=True)
        return dict(elapsed=elapsed)
    sanity = dict(preflight=preflight_result)
    analysis = analyze(cfg, outdir, sanity, elapsed)
    provenance = _provenance(cfg_path, cfg, outdir, sanity, analysis,
                             elapsed, started)
    (outdir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")
    print(f"ALL DONE -> {outdir}", flush=True)
    return dict(sanity=sanity, analysis=analysis)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lin0_base_0902.yaml")
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
        raise ValueError("lin0_base is CPU-only")
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
        path = Path(ROOT) / PREFLIGHT_DIR / "preflight.json"
        sanity = dict(preflight=json.loads(path.read_text(encoding="utf-8")))
        analyze(cfg, outdir, sanity, {})
    else:
        run_full(cfg_path, cfg, device, outdir, smoke=args.smoke)


if __name__ == "__main__":
    main()
