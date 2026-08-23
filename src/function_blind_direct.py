"""Direct functional-utility instrumentation for work item 6 (pilot only).

Run the registered pilot with::

    OMP_NUM_THREADS=1 .venv/bin/python -m src.function_blind_direct \
      --config configs/function_blind_direct_0823_pilot.yaml \
      --reference-logs results/ratchet_log_0819/logs

The probe is read-only: at every registered landmark it enumerates condA's 32
inputs in float64 and records the loss change caused by silencing each hidden
unit.  It never calls ``SCREnv.full_support()`` because that method advances the
environment.  ``--smoke`` changes only the instrumentation grid (B=0) and the
run length/seed count; it is not a scientific pilot result.
"""

from __future__ import annotations

import argparse
import copy
import glob
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Iterable

import numpy as np
import torch
import yaml

from .common import (ROOT, build_runs, group_runs, load_config, pick_device,
                     resolve_outdir)
from .train import setup_group, train_group


UNIT_KEYS = (
    "p_hat", "x", "r", "w_norm", "pre_max", "b", "v",
    "utility_raw", "utility_nmse",
)
RUN_KEYS = ("eval_nmse", "y_var")
SUPPORT_SIZE = 32


# ---------------------------------------------------------------------------
# Read-only exact condA probe

def full_support_ro(env) -> torch.Tensor:
    """Return the current condA support without changing env or a generator."""
    n = int(env.patterns.shape[0])
    flip = env.flip_state.unsqueeze(0).expand(n, -1, -1)
    random_bits = env.patterns[:, None, :].expand(-1, env.R, -1)
    return torch.cat([flip, random_bits], dim=2)


def teacher_f64(teacher, raw_x: torch.Tensor) -> torch.Tensor:
    """Float64 transcription of :class:`LTUTarget`'s forward function."""
    pre = torch.einsum("rhm,prm->prh", teacher.W.double(), raw_x)
    pre = pre + teacher.b.double()
    hidden = (pre >= teacher.tau.double()).double()
    y = (hidden * teacher.v.double()).sum(dim=-1) + teacher.cout.double()
    return y * float(getattr(teacher, "out_scale", 1.0))


def _forward_context(st) -> dict[str, torch.Tensor]:
    """Construct all float64 tensors needed by the exact record and sanity."""
    if st["exp"] != "A":
        raise ValueError("function_blind_direct is restricted to condA")
    if int(st["env"].patterns.shape[0]) != SUPPORT_SIZE:
        raise ValueError("the registered pilot requires exactly 32 support points")

    raw_x = full_support_ro(st["env"]).double()
    y = teacher_f64(st["teacher"], raw_x)
    centered = st["centered"][:, None].double()
    x_in = raw_x - centered[None] * st["running_mean"].double()[None]

    W = st["net"].W.double()
    b = st["net"].b.double()
    v = st["net"].v.double()
    c = st["net"].c.double()
    pre = torch.einsum("rhd,prd->prh", W, x_in) + b
    activation = torch.relu(pre)
    q = activation * v[None]
    yhat = q.sum(dim=-1) + c
    residual = yhat - y
    return dict(raw_x=raw_x, x_in=x_in, y=y, W=W, b=b, v=v, c=c,
                pre=pre, activation=activation, q=q, yhat=yhat,
                residual=residual)


def exact_record(st, *, with_context: bool = False):
    """Compute one registered record in float64 using the exact 32-point support.

    ``utility_raw`` is ``E[q_i**2 - 2*residual*q_i]``.  This equals the
    increase in MSE after setting only output contribution ``q_i`` to zero.
    ``utility_nmse`` divides it by the population variance of the teacher.
    """
    with torch.no_grad():
        ctx = _forward_context(st)
        W, b, v = ctx["W"], ctx["b"], ctx["v"]
        pre, q, residual = ctx["pre"], ctx["q"], ctx["residual"]
        y = ctx["y"]

        gate = pre > 0
        p_hat = gate.double().mean(dim=0)
        w_norm = W.norm(dim=2)
        mu = ctx["x_in"].mean(dim=0)
        mu_norm = mu.norm(dim=1)
        if bool((mu_norm <= 0).any()):
            raise RuntimeError("mu norm is zero; x/r geometry is undefined")
        mu_hat = mu / mu_norm[:, None]
        x_coord = torch.einsum("rhd,rd->rh", W, mu_hat)
        r_sq = torch.clamp(w_norm.square() - x_coord.square(), min=0.0)
        r_coord = torch.sqrt(r_sq)
        pre_max = pre.max(dim=0).values

        utility_raw = (q.square() - 2.0 * residual[:, :, None] * q).mean(dim=0)
        y_var = y.var(dim=0, unbiased=False)
        if bool((y_var <= 0).any()):
            raise RuntimeError("teacher variance is non-positive")
        utility_nmse = utility_raw / y_var[:, None]
        eval_nmse = residual.square().mean(dim=0) / y_var

        geom_num = (x_coord.square() + r_coord.square() - w_norm.square()).abs()
        geom_den = w_norm.square()
        geom_rel = torch.where(geom_den > 0, geom_num / geom_den, geom_num)
        strict_dead = p_hat == 0
        wall_dead = pre_max <= 0

        tensors = dict(
            p_hat=p_hat, x=x_coord, r=r_coord, w_norm=w_norm,
            pre_max=pre_max, b=b, v=v, utility_raw=utility_raw,
            utility_nmse=utility_nmse, eval_nmse=eval_nmse, y_var=y_var,
        )
        n_nonfinite = int(sum((~torch.isfinite(z)).sum().item()
                              for z in tensors.values()))
        sanity = dict(
            support_size=int(ctx["x_in"].shape[0]),
            max_p_hat_quantization_abs_err=float(
                (p_hat * SUPPORT_SIZE - torch.round(p_hat * SUPPORT_SIZE))
                .abs().max().item()),
            max_geometry_relative_error=float(geom_rel.max().item()),
            strict_dead_pre_max_mismatches=int((strict_dead != wall_dead).sum().item()),
            n_nonfinite=n_nonfinite,
        )
        out = {key: value.detach().cpu().numpy().astype(np.float64, copy=False)
               for key, value in tensors.items()}
        if with_context:
            return out, sanity, ctx
        return out, sanity


def brute_force_delta_mse(ctx: dict[str, torch.Tensor], run: int, unit: int) -> float:
    """Re-forward one run after silencing one unit on a cloned output vector."""
    # Recompute the hidden layer, rather than obtaining the answer by subtracting q.
    x = ctx["x_in"][:, run]
    W = ctx["W"][run]
    b = ctx["b"][run]
    v = ctx["v"][run]
    c = ctx["c"][run]
    y = ctx["y"][:, run]
    pre = torch.einsum("hd,pd->ph", W, x) + b
    activation = torch.relu(pre)
    yhat = (activation * v).sum(dim=1) + c
    v_silenced = v.clone()
    v_silenced[unit] = 0.0
    yhat_silenced = (activation * v_silenced).sum(dim=1) + c
    mse = (yhat - y).square().mean()
    mse_silenced = (yhat_silenced - y).square().mean()
    return float((mse_silenced - mse).item())


def check_delta_formula(st, examples: Iterable[tuple[int, int]],
                        formula: np.ndarray,
                        ctx: dict[str, torch.Tensor] | None = None) -> dict[str, Any]:
    """Check vectorized delta-MSE against explicit silenced re-forwards."""
    pairs = [(int(run), int(unit)) for run, unit in examples]
    ctx = _forward_context(st) if ctx is None else ctx
    rows = []
    for run, unit in pairs:
        brute = brute_force_delta_mse(ctx, run, unit)
        vector = float(formula[run, unit])
        rows.append(dict(run=run, unit=unit, vector=vector, brute=brute,
                         abs_error=abs(vector - brute)))
    return dict(n_examples=len(rows),
                max_abs_error=max((r["abs_error"] for r in rows), default=0.0),
                examples=rows)


# ---------------------------------------------------------------------------
# Recorder and deterministic sanity sample

def landmark_grid(total_steps: int, pilot_cfg: dict[str, Any]):
    """Return the complete registered B+1/B+10000 landmark pairs."""
    start = int(pilot_cfg["boundary_start"])
    stop = int(pilot_cfg["boundary_stop"])
    every = int(pilot_cfg["boundary_every"])
    if every <= 0 or stop < start:
        raise ValueError("invalid function_blind_direct boundary grid")
    boundaries = [b for b in range(start, stop + 1, every)
                  if b + every <= int(total_steps)]
    if not boundaries:
        raise ValueError("total_steps does not contain a complete B+1/B+period pair")
    steps, landmark, phase = [], [], []
    for boundary in boundaries:
        steps.extend((boundary + 1, boundary + every))
        landmark.extend((boundary, boundary))
        phase.extend(("t0", "t1"))
    return (np.asarray(steps, dtype=np.int64),
            np.asarray(landmark, dtype=np.int64),
            np.asarray(phase, dtype="U2"))


def deterministic_examples(steps: np.ndarray, R: int, h: int, n: int):
    """Select evenly spread, deterministic (step, run, unit) triples."""
    total = int(len(steps) * R * h)
    n = min(int(n), total)
    if n <= 0:
        return {}
    flat = np.linspace(0, total - 1, num=n, dtype=np.int64)
    chosen: dict[int, list[tuple[int, int]]] = {}
    for value in flat:
        step_index, rem = divmod(int(value), R * h)
        run, unit = divmod(rem, h)
        chosen.setdefault(int(steps[step_index]), []).append((run, unit))
    return chosen


class DirectRecorder:
    """Preallocated read-only probe used by :func:`train_group`."""

    def __init__(self, steps: np.ndarray, landmark: np.ndarray, phase: np.ndarray,
                 R: int, h: int, n_brute: int = 20):
        self.steps = np.asarray(steps, dtype=np.int64)
        self.landmark = np.asarray(landmark, dtype=np.int64)
        self.phase = np.asarray(phase, dtype="U2")
        if not (len(self.steps) == len(self.landmark) == len(self.phase)):
            raise ValueError("landmark metadata length mismatch")
        if len(np.unique(self.steps)) != len(self.steps):
            raise ValueError("probe steps must be unique")
        self.index = {int(step): i for i, step in enumerate(self.steps)}
        self.unit = {key: np.empty((len(self.steps), R, h), dtype=np.float64)
                     for key in UNIT_KEYS}
        self.run = {key: np.empty((len(self.steps), R), dtype=np.float64)
                    for key in RUN_KEYS}
        self.filled = np.zeros(len(self.steps), dtype=bool)
        self.quant_error = np.zeros(len(self.steps), dtype=np.float64)
        self.geometry_error = np.zeros(len(self.steps), dtype=np.float64)
        self.strict_mismatch = np.zeros(len(self.steps), dtype=np.int64)
        self.nonfinite = np.zeros(len(self.steps), dtype=np.int64)
        self.support_size = np.zeros(len(self.steps), dtype=np.int64)
        self.examples = deterministic_examples(self.steps, R, h, n_brute)
        self.brute_rows: list[dict[str, Any]] = []
        self.n_calls = 0

    def __call__(self, st, step: int):
        index = self.index.get(int(step))
        if index is None:
            return
        record, sanity, ctx = exact_record(st, with_context=True)
        for key in UNIT_KEYS:
            self.unit[key][index] = record[key]
        for key in RUN_KEYS:
            self.run[key][index] = record[key]
        self.quant_error[index] = sanity["max_p_hat_quantization_abs_err"]
        self.geometry_error[index] = sanity["max_geometry_relative_error"]
        self.strict_mismatch[index] = sanity["strict_dead_pre_max_mismatches"]
        self.nonfinite[index] = sanity["n_nonfinite"]
        self.support_size[index] = sanity["support_size"]
        if int(step) in self.examples:
            result = check_delta_formula(st, self.examples[int(step)],
                                         record["utility_raw"], ctx=ctx)
            for row in result["examples"]:
                row["step"] = int(step)
                self.brute_rows.append(row)
        self.filled[index] = True
        self.n_calls += 1

    def check_complete(self):
        missing = np.flatnonzero(~self.filled)
        if len(missing):
            raise RuntimeError(f"missing {len(missing)} direct-probe steps: "
                               f"{self.steps[missing][:10].tolist()}")

    def sanity(self, cfg: dict[str, Any]) -> dict[str, Any]:
        brute_atol = float(cfg.get("brute_force_atol", 1e-12))
        geometry_rtol = float(cfg.get("geometry_rtol", 1e-10))
        n_expected = min(int(cfg.get("brute_force_examples", 20)),
                         int(np.prod(next(iter(self.unit.values())).shape)))
        max_brute = max((row["abs_error"] for row in self.brute_rows), default=np.inf)
        max_quant = float(self.quant_error.max(initial=0.0))
        max_geometry = float(self.geometry_error.max(initial=0.0))
        n_mismatch = int(self.strict_mismatch.sum())
        n_nonfinite = int(self.nonfinite.sum())
        support_ok = bool((self.support_size == SUPPORT_SIZE).all())
        return dict(
            delta_formula_pass=bool(len(self.brute_rows) >= n_expected and
                                    max_brute < brute_atol),
            delta_formula_n=int(len(self.brute_rows)),
            delta_formula_expected_min=int(n_expected),
            delta_formula_max_abs_error=float(max_brute),
            delta_formula_atol=brute_atol,
            p_hat_quantization_pass=bool(max_quant == 0.0),
            p_hat_quantization_max_abs_error=max_quant,
            geometry_pass=bool(max_geometry < geometry_rtol),
            geometry_max_relative_error=max_geometry,
            geometry_rtol=geometry_rtol,
            strict_dead_pre_max_identity_pass=bool(n_mismatch == 0),
            strict_dead_pre_max_mismatches=n_mismatch,
            finite_pass=bool(n_nonfinite == 0),
            n_nonfinite=n_nonfinite,
            support_size_pass=support_ok,
            support_sizes=sorted(set(int(x) for x in self.support_size)),
            brute_force_examples=self.brute_rows,
        )


# ---------------------------------------------------------------------------
# Complete mutable-state hashes and reference comparison

def _hash_value(value: Any) -> str:
    h = hashlib.sha256()
    if torch.is_tensor(value):
        array = np.ascontiguousarray(value.detach().cpu().numpy())
        h.update(str(array.dtype).encode())
        h.update(str(array.shape).encode())
        h.update(array.tobytes())
    else:
        h.update(json.dumps(value, sort_keys=True, default=str).encode())
    return h.hexdigest()


def complete_state_hashes(st) -> dict[str, str]:
    """Hash every mutable state named by the registered no-perturbation check."""
    hashes: dict[str, str] = {}
    for prefix, state in (("net", st["net"].state_dict()),
                          ("env", st["env"].state_dict()),
                          ("teacher", st["teacher"].state_dict())):
        for key, value in state.items():
            hashes[f"{prefix}.{key}"] = _hash_value(value)
    hashes["teacher.out_scale"] = _hash_value(
        float(getattr(st["teacher"], "out_scale", 1.0)))
    hashes["running_mean"] = _hash_value(st["running_mean"])
    hashes["eval_fixed"] = _hash_value(st["eval_fixed"])
    for key, generator in sorted(st["gens"].items()):
        hashes[f"generator.{key}"] = _hash_value(generator.get_state())
    if st.get("cbp") is not None:
        for key, value in sorted(st["cbp"].items()):
            hashes[f"cbp.{key}"] = _hash_value(value)
    return hashes


def check_default_zero_offset(cfg, gkey, runs, device) -> dict[str, Any]:
    """Verify omitted generator_offset and explicit zero initialize identically."""
    implicit = copy.deepcopy(cfg)
    implicit["common"].pop("generator_offset", None)
    explicit = copy.deepcopy(cfg)
    explicit["common"]["generator_offset"] = 0
    a = complete_state_hashes(setup_group(gkey, runs, implicit, device))
    b = complete_state_hashes(setup_group(gkey, runs, explicit, device))
    differences = sorted(key for key in a if a[key] != b[key])
    return dict(pass_=not differences, differences=differences,
                implicit_hashes=a, explicit_zero_hashes=b)


def _reference_files(reference: str) -> list[str]:
    path = Path(reference)
    if path.is_dir():
        files = sorted(str(p) for p in path.glob("seed*.npz"))
    elif path.is_file():
        files = [str(path)]
    else:
        files = sorted(glob.glob(reference))
    if not files:
        raise FileNotFoundError(f"no reference NPZ files: {reference}")
    return files


def compare_reference_logs(rec: DirectRecorder, runs: list[dict[str, Any]],
                           reference: str, cfg: dict[str, Any]) -> dict[str, Any]:
    """Compare p_hat/x/r against ratchet_log's saved float32 records."""
    atol = float(cfg.get("reference_atol", 5e-6))
    rtol = float(cfg.get("reference_rtol", 5e-6))
    ref_by_seed = {}
    for filename in _reference_files(reference):
        with np.load(filename, allow_pickle=False) as z:
            seed = int(np.asarray(z["seed"]).item())
        ref_by_seed[seed] = filename

    rows, all_pass = [], True
    for run_index, run in enumerate(runs):
        seed = int(run["seed"])
        filename = ref_by_seed.get(seed)
        if filename is None:
            rows.append(dict(seed=seed, pass_=False, reason="missing seed"))
            all_pass = False
            continue
        with np.load(filename, allow_pickle=False) as z:
            ref_steps = np.asarray(z["step"], dtype=np.int64)
            lookup = {int(step): i for i, step in enumerate(ref_steps)}
            missing = [int(step) for step in rec.steps if int(step) not in lookup]
            if missing:
                rows.append(dict(seed=seed, pass_=False,
                                 reason=f"missing {len(missing)} steps",
                                 missing_steps=missing[:10]))
                all_pass = False
                continue
            idx = np.asarray([lookup[int(step)] for step in rec.steps], dtype=np.int64)
            old_p = np.asarray(z["p_hat"][idx], dtype=np.float64)
            old_w = np.asarray(z["w_norm"][idx], dtype=np.float64)
            old_cos = np.asarray(z["cos_u_mu"][idx], dtype=np.float64)
            old_x = old_w * old_cos
            old_r = old_w * np.sqrt(np.maximum(0.0, 1.0 - np.square(old_cos)))

        current = dict(p_hat=rec.unit["p_hat"][:, run_index],
                       x=rec.unit["x"][:, run_index],
                       r=rec.unit["r"][:, run_index])
        old = dict(p_hat=old_p, x=old_x, r=old_r)
        errors = {}
        passed = True
        for key in ("p_hat", "x", "r"):
            diff = np.abs(current[key] - old[key])
            limit = atol + rtol * np.abs(old[key])
            errors[f"{key}_max_abs_error"] = float(diff.max(initial=0.0))
            errors[f"{key}_max_scaled_error"] = float(
                np.divide(diff, limit, out=np.zeros_like(diff), where=limit > 0)
                .max(initial=0.0))
            passed = passed and bool((diff <= limit).all())
        rows.append(dict(seed=seed, pass_=passed, n_steps=len(rec.steps),
                         file=filename, **errors))
        all_pass = all_pass and passed
    return dict(pass_=bool(all_pass), atol=atol, rtol=rtol, per_seed=rows)


# ---------------------------------------------------------------------------
# Output and execution

def write_npz_logs(rec: DirectRecorder, runs: list[dict[str, Any]], outdir: str,
                   cfg: dict[str, Any]) -> list[str]:
    logdir = Path(outdir) / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    common = cfg["common"]
    spec = str(cfg.get("spec", "specs/spec_function_blind_direct_0823_pilot.md"))
    paths = []
    for run_index, run in enumerate(runs):
        values: dict[str, Any] = dict(
            step=rec.steps,
            landmark_B=rec.landmark,
            phase=rec.phase,
            seed=np.int64(run["seed"]),
            run_id=np.asarray(run["run_id"]),
            condition=np.asarray("condA"),
            encoding=np.asarray(run["enc"]),
            batch=np.asarray(str(run["batch"])),
            width=np.int64(run["width"]),
            period=np.int64(run["period"]),
            lr=np.float64(run["lr"]),
            generator_offset=np.int64(common.get("generator_offset", 0)),
            total_steps=np.int64(common["total_steps"]),
            support_size=np.int64(SUPPORT_SIZE),
            spec=np.asarray(spec),
        )
        for key in UNIT_KEYS:
            values[key] = rec.unit[key][:, run_index]
        for key in RUN_KEYS:
            values[key] = rec.run[key][:, run_index]
        path = logdir / f"seed{run['seed']}.npz"
        np.savez_compressed(path, **values)
        paths.append(str(path))
    return paths


def run_s2(cfg, gkey, runs, device, outdir: str, steps: int,
           pilot_cfg: dict[str, Any]) -> dict[str, Any] | None:
    """Run a short probe/no-probe trajectory and compare complete state hashes."""
    if int(steps) <= 0:
        return None
    short_cfg = copy.deepcopy(cfg)
    short_cfg["common"]["total_steps"] = int(steps)
    short_cfg["common"]["checkpoints"] = []
    probe_steps = np.asarray(sorted({0, int(steps)}), dtype=np.int64)
    landmark = np.full(len(probe_steps), -1, dtype=np.int64)
    phase = np.asarray(["s2"] * len(probe_steps), dtype="U2")
    R, h = len(runs), int(gkey[1])
    recorder = DirectRecorder(probe_steps, landmark, phase, R, h,
                              n_brute=int(pilot_cfg.get("brute_force_examples", 20)))
    s2_out = os.path.join(outdir, "_s2_training_logs")
    with_probe, _ = train_group(
        gkey, runs, short_cfg, device, s2_out, total_steps=int(steps), ckpts=[],
        gname="function_blind_direct_S2_with_probe", probe=recorder,
        probe_steps=probe_steps,
    )
    recorder.check_complete()
    without_probe, _ = train_group(
        gkey, runs, short_cfg, device, s2_out, total_steps=int(steps), ckpts=[],
        gname="function_blind_direct_S2_no_probe",
    )
    hash_probe = complete_state_hashes(with_probe)
    hash_no_probe = complete_state_hashes(without_probe)
    differences = sorted(key for key in hash_probe
                         if hash_probe[key] != hash_no_probe[key])
    return dict(pass_=not differences, steps=int(steps), differences=differences,
                with_probe=hash_probe, without_probe=hash_no_probe,
                probe_sanity=recorder.sanity(pilot_cfg))


def _single_registered_group(cfg):
    runs = build_runs(cfg)
    groups = group_runs(runs)
    if len(groups) != 1:
        raise ValueError(f"expected one condA/w100 group, got {sorted(groups)}")
    gkey, grouped_runs = next(iter(groups.items()))
    if gkey != ("A", 100, 1, "none"):
        raise ValueError(f"registered pilot requires ('A',100,1,'none'), got {gkey}")
    periods = {int(run["period"]) for run in grouped_runs}
    encodings = {run["enc"] for run in grouped_runs}
    if periods != {10000} or encodings != {"std"}:
        raise ValueError("registered pilot requires T=10000 and std encoding")
    return gkey, grouped_runs


def run(cfg, device: str, outdir: str, *, s2_steps: int = 0,
        reference_logs: str | None = None) -> dict[str, Any]:
    """Execute instrumentation and write per-seed float64 NPZ files."""
    if os.environ.get("OMP_NUM_THREADS") != "1" or torch.get_num_threads() != 1:
        raise RuntimeError("OMP_NUM_THREADS=1 is required by the registered pilot")
    pilot_cfg = cfg["function_blind_direct"]
    gkey, runs = _single_registered_group(cfg)
    steps, landmark, phase = landmark_grid(cfg["common"]["total_steps"], pilot_cfg)
    recorder = DirectRecorder(
        steps, landmark, phase, len(runs), int(gkey[1]),
        n_brute=int(pilot_cfg.get("brute_force_examples", 20)),
    )

    os.makedirs(outdir, exist_ok=True)
    started = time.time()
    print(f"group={gkey} R={len(runs)} total={cfg['common']['total_steps']} "
          f"records={len(steps)} offset={cfg['common'].get('generator_offset', 0)}",
          flush=True)
    state, train_seconds = train_group(
        gkey, runs, cfg, device, os.path.join(outdir, "_training_logs"),
        total_steps=int(cfg["common"]["total_steps"]), ckpts=[],
        probe=recorder, probe_steps=steps,
    )
    recorder.check_complete()
    instrumentation_sanity = recorder.sanity(pilot_cfg)
    zero_offset = check_default_zero_offset(cfg, gkey, runs, device)
    s2 = run_s2(cfg, gkey, runs, device, outdir, int(s2_steps), pilot_cfg)
    reference = (compare_reference_logs(recorder, runs, reference_logs, pilot_cfg)
                 if reference_logs else None)
    paths = write_npz_logs(recorder, runs, outdir, cfg)

    required = [
        instrumentation_sanity["delta_formula_pass"],
        instrumentation_sanity["p_hat_quantization_pass"],
        instrumentation_sanity["geometry_pass"],
        instrumentation_sanity["strict_dead_pre_max_identity_pass"],
        instrumentation_sanity["finite_pass"],
        instrumentation_sanity["support_size_pass"],
        zero_offset["pass_"],
    ]
    if s2 is not None:
        required.append(s2["pass_"])
    if reference is not None:
        required.append(reference["pass_"])
    all_pass = bool(all(required))
    meta = dict(
        spec=cfg.get("spec"), pilot_only=True, device=device,
        total_steps=int(cfg["common"]["total_steps"]),
        seeds=[int(run["seed"]) for run in runs], R=len(runs), width=int(gkey[1]),
        generator_offset=int(cfg["common"].get("generator_offset", 0)),
        n_records=int(len(steps)), n_npz=len(paths), npz_paths=paths,
        train_seconds=round(float(train_seconds), 3),
        elapsed_seconds=round(time.time() - started, 3),
        final_state_hashes=complete_state_hashes(state),
        sanity=dict(instrumentation=instrumentation_sanity,
                    default_zero_offset=zero_offset, S2=s2,
                    reference_ratchet_log=reference,
                    all_required_pass=all_pass),
    )
    meta_path = Path(outdir) / "instrumentation_meta.json"
    with meta_path.open("w") as handle:
        json.dump(meta, handle, indent=2, ensure_ascii=False)
    print(f"wrote {len(paths)} NPZ files; sanity={'PASS' if all_pass else 'FAIL'}",
          flush=True)
    if not all_pass:
        raise RuntimeError(f"instrumentation sanity failed; inspect {meta_path}")
    return meta


def apply_smoke(cfg):
    """Return a non-scientific 10k-step/one-run instrumentation smoke config."""
    cfg = copy.deepcopy(cfg)
    cfg["common"].update(total_steps=10000, seeds=[0], checkpoints=[])
    cfg["function_blind_direct"].update(boundary_start=0, boundary_stop=0,
                                          boundary_every=10000)
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/function_blind_direct_0823_pilot.yaml")
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--generator-offset", type=int, default=None)
    parser.add_argument("--s2-steps", type=int, default=0,
                        help="probe/no-probe complete-state comparison length")
    parser.add_argument("--reference-logs", default=None,
                        help="ratchet_log_0819 logs directory, NPZ, or glob")
    parser.add_argument("--device", default=None)
    parser.add_argument("--smoke", action="store_true",
                        help="one run, 10k steps, B=0 instrumentation grid")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.smoke:
        cfg = apply_smoke(cfg)
    if args.total_steps is not None:
        cfg["common"]["total_steps"] = int(args.total_steps)
    if args.seeds is not None:
        cfg["common"]["seeds"] = list(args.seeds)
    if args.generator_offset is not None:
        cfg["common"]["generator_offset"] = int(args.generator_offset)
    if args.device is not None:
        cfg["common"]["device"] = args.device
    device = pick_device(cfg)
    outdir = resolve_outdir(args.config, smoke=args.smoke, outdir=args.outdir)
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "config_used.yaml"), "w") as handle:
        yaml.safe_dump(cfg, handle, allow_unicode=True, sort_keys=False)
    run(cfg, device, outdir, s2_steps=args.s2_steps,
        reference_logs=args.reference_logs)


if __name__ == "__main__":
    main()
