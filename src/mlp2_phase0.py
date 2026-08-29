"""mlp2_phase0_0829: exact-support measurements for one- and two-layer MLPs.

Run the preregistered experiment with::

    OMP_NUM_THREADS=1 .venv/bin/python -m src.mlp2_phase0 \
        --config configs/mlp2_phase0_0829.yaml

The smoke run executes seed 0 for 30k steps and emits only logs plus structural
sanity (no G0 or trend aggregation)::

    OMP_NUM_THREADS=1 .venv/bin/python -m src.mlp2_phase0 --smoke

The implementation is deliberately separate from ``train_group``: the latter
has one-hidden-layer LoP instrumentation baked into it.  Environment, teacher,
generator construction, online-SGD arithmetic, and the L=1 state schema are
kept identical so S0 can compare against ``ratchet_log_0819`` bit for bit.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
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
from .envs import LTUTarget, SCREnv
from .nets import VecMLPL
from .ratchet_log import (exact_record as legacy_exact_record,
                          full_support_ro, record_steps as legacy_record_steps,
                          state_hash, teacher_f64)
from .train import make_gens


SMOKE_STEPS = 30_000
LEGACY_META_KEYS = {"run_id", "seed", "lr", "period", "width"}
LOG_UNIT_KEYS = ("M", "B", "denom", "p_hat", "w_norm")
LOG_LAYER_KEYS = ("median_M", "q25_M", "q75_M", "median_B", "n_na",
                  "mu_norm", "sigma_rms", "dose", "w_norm_median",
                  "w_norm_q25", "w_norm_q75", "eff_rank", "eff_rank_W",
                  "strict_dead", "alive", "eff_rank_per_alive")


def _sha_array(value) -> str:
    arr = np.ascontiguousarray(value.detach().cpu().numpy()
                               if torch.is_tensor(value) else np.asarray(value))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _jsonable_hashes(st) -> dict[str, str]:
    return state_hash(st)


def _seed_state_hashes(st, seed_index: int) -> dict[str, str]:
    out = {f"net.{name}": _sha_array(value[seed_index])
           for name, value in st["net"].state_dict().items()}
    out["env.flip_state"] = _sha_array(st["env"].flip_state[seed_index])
    out["env.t"] = str(st["env"].t)
    out["running_mean"] = _sha_array(st["running_mean"][seed_index])
    return out


def validate_config(cfg: dict, *, smoke: bool) -> None:
    C, A, P = cfg["common"], cfg["condA"], cfg["phase0"]
    arms = {a["name"]: list(a["hidden"]) for a in cfg["arms"]}
    required = {"L1": [100], "L2": [100, 100]}
    if arms != required:
        raise ValueError(f"arms must be exactly {required}, got {arms}")
    if int(A["m"]) != 20 or int(A["f"]) != 15:
        raise ValueError("mlp2 phase0 is preregistered only for m=20, f=15")
    if list(A["T_values"]) != [10_000] or list(A["encodings"]) != ["std"]:
        raise ValueError("mlp2 phase0 is preregistered only for T=10000, enc=std")
    if int(P["exact_support"]) != 2 ** (int(A["m"]) - int(A["f"])):
        raise ValueError("phase0.exact_support does not match the full condA support")
    if str(P["ci_method"]) != "studentized":
        raise ValueError("only the preregistered studentized interval is allowed")
    if str(P["trend_statistic"]) != "spearman" or not P["trend_at_task_end_only"]:
        raise ValueError("trend must be task-end-only Spearman")
    if not smoke:
        if int(C["total_steps"]) != 1_000_000 or list(C["seeds"]) != list(range(10)):
            raise ValueError("the full run requires 1M steps and seeds 0..9")
        if str(C.get("device")) != "cpu":
            raise ValueError("the preregistered full run requires device=cpu")


def require_omp(cfg: dict) -> dict:
    expected = str(int(cfg["sanity"]["omp_num_threads"]))
    actual = os.environ.get("OMP_NUM_THREADS")
    torch_threads = int(torch.get_num_threads())
    result = dict(expected=expected, env=actual, torch_num_threads=torch_threads,
                  pass_=actual == expected and torch_threads == int(expected))
    if not result["pass_"]:
        raise RuntimeError(
            f"S3 requires OMP_NUM_THREADS={expected}; env={actual!r}, "
            f"torch threads={torch_threads}"
        )
    return result


def arm_runs(cfg: dict, arm: str) -> list[dict]:
    C = cfg["common"]
    period = int(cfg["phase0"]["task_period"])
    return [dict(arm=arm, seed=int(seed), period=period, lr=float(C["lr_main"]),
                 run_id=f"{arm}_seed{int(seed)}") for seed in C["seeds"]]


def setup_arm(cfg: dict, arm_cfg: dict, device: str) -> dict:
    runs = arm_runs(cfg, arm_cfg["name"])
    R = len(runs)
    A, C = cfg["condA"], cfg["common"]
    m, f = int(A["m"]), int(A["f"])
    hidden = [int(v) for v in arm_cfg["hidden"]]
    # The legacy base uses the learner width (100).  Keeping this exact is part
    # of S0; the streams themselves remain separated as in train.make_gens.
    gens = make_gens("A", hidden[0], device)
    period = torch.tensor([r["period"] for r in runs], dtype=torch.long)
    env = SCREnv(R, m, f, period, gens["input"], device)
    teacher = LTUTarget(R, m, int(A["target_hidden"]), float(A["beta"]),
                        gens["teacher"], device)
    net = VecMLPL(R, hidden, m, gens["init"], device)
    eval_fixed = torch.randint(0, 2, (int(C["eval_batch"]), m - f),
                               generator=gens["eval"], device=device).float()
    return dict(exp="A", arm=arm_cfg["name"], R=R, d=m, width=hidden[0],
                hidden=hidden, env=env, teacher=teacher, net=net,
                running_mean=torch.zeros(R, m, device=device),
                lr=torch.tensor([r["lr"] for r in runs], device=device),
                centered=torch.zeros(R, dtype=torch.bool, device=device),
                period=period, eval_fixed=eval_fixed, runs=runs, device=device,
                gens=gens, center_alpha=float(A.get("center_alpha", 0.01)))


def _max_relative(a: torch.Tensor, b: torch.Tensor,
                  mask: torch.Tensor | None = None) -> float:
    if mask is not None:
        a, b = a[mask], b[mask]
    if not a.numel():
        return 0.0
    # Match the repository's existing identity checks: one max-norm relative
    # error per layer/quantity.  Per-element division is ill-posed when a true
    # wall coordinate crosses zero.
    scale = torch.maximum(a.abs().max(), b.abs().max()).clamp_min(1e-300)
    return float(((a - b).abs().max() / scale).item())


def _effective_rank(matrices: torch.Tensor) -> torch.Tensor:
    s = torch.linalg.svdvals(matrices)
    p = s / s.sum(dim=1, keepdim=True).clamp_min(1e-300)
    return torch.exp(-(p * p.clamp_min(1e-300).log()).sum(dim=1))


def exact_layer_record(st: dict, sigma_tol: float) -> tuple[dict, dict]:
    """Compute all preregistered quantities on the 32-pattern support.

    The implementation forms each layer input distribution explicitly.  S1
    compares direct preactivation means/SDs with ``w.mu+b`` and
    ``sqrt(w^T Sigma w)``; S2 checks the normalized wall-coordinate identity and
    the legacy first-layer cosine identity independently.
    """
    with torch.no_grad():
        X = full_support_ro(st["env"]).double()
        y = teacher_f64(st["teacher"], X)
        cur = X
        layers = []
        sanity_layers = []

        for layer, (W0, b0) in enumerate(zip(st["net"].Ws, st["net"].bs), start=1):
            W, b = W0.double(), b0.double()
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
            layers.append(dict(
                M=M, B=B, denom=denom, p_hat=p_hat, w_norm=w_norm,
                median_M=qM[1], q25_M=qM[0], q75_M=qM[2], median_B=median_B,
                n_na=(~valid).sum(dim=1), mu_norm=mu_norm, sigma_rms=sigma_rms,
                dose=dose, w_norm_median=qW[1], w_norm_q25=qW[0],
                w_norm_q75=qW[2], eff_rank=eff_rank, eff_rank_W=eff_rank_W,
                strict_dead=strict_dead, alive=alive,
                eff_rank_per_alive=eff_per_alive))

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
                      support=int(X.shape[0]))
        return dict(run=run, layers=layers,
                    flip_state=st["env"].flip_state.double()), sanity


def identity_sanity_pass(sanity: dict, tolerance: float) -> bool:
    return bool(sanity["run_finite"] and all(
        row["finite_required"]
        and row["mean_max_relerr"] <= tolerance
        and row["sd_max_relerr"] <= tolerance
        and row["wall_max_relerr"] <= tolerance
        and row["l1_cos_mu_max_relerr"] <= tolerance
        for row in sanity["layers"]
    ))


class PhaseRecorder:
    """Preallocated exact-support recorder for one arm."""

    def __init__(self, steps: list[int], st: dict, sigma_tol: float, identity_tol: float):
        self.steps = np.asarray(steps, dtype=np.int64)
        self.index = {int(step): i for i, step in enumerate(self.steps)}
        self.sigma_tol = float(sigma_tol)
        self.identity_tol = float(identity_tol)
        n, R, f = len(steps), st["R"], st["env"].f
        self.run = {k: np.empty((n, R), dtype=np.float64)
                    for k in ("signal_var", "residual_var", "unfit", "eval_loss_exact")}
        self.flip_state = np.empty((n, R, f), dtype=np.float32)
        self.layers = []
        for h in st["hidden"]:
            unit = {k: np.empty((n, R, h), dtype=np.float32) for k in LOG_UNIT_KEYS}
            scalars = {k: np.empty((n, R), dtype=(np.int64 if k in
                       ("n_na", "strict_dead", "alive") else np.float64))
                       for k in LOG_LAYER_KEYS}
            self.layers.append({**unit, **scalars})
        self.filled = np.zeros(n, dtype=bool)
        self.max_errors = [dict(mean=0.0, sd=0.0, wall=0.0, cos_mu=0.0,
                                n_degenerate_max=0) for _ in st["hidden"]]
        self.required_nonfinite = 0

    def __call__(self, st: dict, step: int) -> None:
        i = self.index.get(int(step))
        if i is None:
            return
        if self.filled[i]:
            raise RuntimeError(f"duplicate phase0 probe at step {step}")
        rec, sanity = exact_layer_record(st, self.sigma_tol)
        for key, value in rec["run"].items():
            self.run[key][i] = value.detach().cpu().numpy()
        self.flip_state[i] = rec["flip_state"].detach().cpu().numpy().astype(np.float32)
        for li, layer in enumerate(rec["layers"]):
            for key in LOG_UNIT_KEYS:
                self.layers[li][key][i] = layer[key].detach().cpu().numpy().astype(np.float32)
            for key in LOG_LAYER_KEYS:
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

    def check_complete(self) -> None:
        missing = self.steps[~self.filled]
        if missing.size:
            raise RuntimeError(f"missing {missing.size} phase0 probe steps: {missing[:10]}")

    def sanity(self) -> dict:
        self.check_complete()
        layer_rows = []
        for li, e in enumerate(self.max_errors, start=1):
            ok = (e["mean"] <= self.identity_tol and e["sd"] <= self.identity_tol
                  and e["wall"] <= self.identity_tol and e["cos_mu"] <= self.identity_tol)
            layer_rows.append(dict(layer=li, pass_=bool(ok), **e))
        return dict(pass_=bool(all(r["pass_"] for r in layer_rows)
                               and self.required_nonfinite == 0),
                    layers=layer_rows, required_nonfinite=self.required_nonfinite,
                    n_record_steps=int(len(self.steps)))


class S0Replay:
    """Streaming all-column equality check against ratchet_log_0819.

    Expected arrays are hashed one seed at a time.  Replay rows update matching
    digests, avoiding the roughly 600 MB in-memory legacy Recorder.
    """

    def __init__(self, reference_seed0: Path, steps: list[int], seeds: list[int],
                 phase_recorder: PhaseRecorder | None):
        self.steps = np.asarray(steps, dtype=np.int64)
        self.index = {int(v): i for i, v in enumerate(self.steps)}
        self.phase = phase_recorder
        self.seeds = list(seeds)
        self.expected: dict[tuple[int, str], str] = {}
        self.actual: dict[tuple[int, str], hashlib._Hash] = {}
        self.array_keys: list[str] | None = None
        self.metadata_ok = True
        self.reference_dir = reference_seed0.parent
        for ri, seed in enumerate(self.seeds):
            path = self.reference_dir / f"seed{seed}.npz"
            if not path.exists():
                raise FileNotFoundError(path)
            with np.load(path, allow_pickle=False) as z:
                if not np.array_equal(z["step"], self.steps):
                    raise ValueError(f"S0 step grid differs for {path}")
                keys = [k for k in z.files if k not in LEGACY_META_KEYS | {"step", "center_alpha"}]
                if self.array_keys is None:
                    self.array_keys = keys
                elif keys != self.array_keys:
                    raise ValueError(f"S0 column schema differs for {path}")
                expected_id = f"A_w100_T10000_std_lr0.01_s{seed}"
                self.metadata_ok &= (int(z["seed"]) == seed and int(z["period"]) == 10_000
                                     and int(z["width"]) == 100
                                     and float(z["lr"]) == np.float32(0.01)
                                     and str(z["run_id"]) == expected_id)
                if "center_alpha" in z.files:
                    self.metadata_ok &= float(z["center_alpha"]) == np.float32(0.01)
                for key in keys:
                    self.expected[(ri, key)] = _sha_array(z[key])
                    self.actual[(ri, key)] = hashlib.sha256()
        self.filled = np.zeros(len(steps), dtype=bool)

    def __call__(self, st: dict, step: int) -> None:
        i = self.index.get(int(step))
        if i is None:
            return
        rec = legacy_exact_record(st)
        for ri in range(len(self.seeds)):
            for key in self.array_keys or ():
                self.actual[(ri, key)].update(
                    np.ascontiguousarray(rec[key][ri]).tobytes())
        self.filled[i] = True
        if self.phase is not None:
            self.phase(st, step)

    def finish(self, st: dict, expected_checkpoint: Path) -> dict:
        missing = self.steps[~self.filled]
        column_differences = []
        for ident, expected in self.expected.items():
            if self.actual[ident].hexdigest() != expected:
                column_differences.append(
                    dict(seed=self.seeds[ident[0]], column=ident[1]))

        if not expected_checkpoint.exists():
            raise FileNotFoundError(expected_checkpoint)
        ck = torch.load(expected_checkpoint, map_location="cpu", weights_only=False)
        expected_state = {f"net.{k}": _sha_array(v) for k, v in ck["net"].items()}
        expected_state.update(env_flip_state=_sha_array(ck["env"]["flip_state"]),
                              env_t=str(ck["env"]["t"]),
                              running_mean=_sha_array(ck["running_mean"]))
        actual_state_raw = _jsonable_hashes(st)
        actual_state = {f"net.{k}": actual_state_raw[f"net.{k}"]
                        for k in ("W", "b", "v", "c")}
        actual_state.update(env_flip_state=actual_state_raw["env.flip_state"],
                            env_t=actual_state_raw["env.t"],
                            running_mean=actual_state_raw["running_mean"])
        state_differences = sorted(k for k in expected_state
                                   if expected_state[k] != actual_state.get(k))
        result = dict(pass_=bool(not missing.size and self.metadata_ok
                                 and not column_differences and not state_differences),
                      metadata_pass=bool(self.metadata_ok),
                      n_steps=int(len(self.steps)),
                      n_columns=int(len(self.array_keys or ())),
                      n_seed_columns=int(len(self.expected)),
                      missing_steps=[int(v) for v in missing[:10]],
                      column_differences=column_differences,
                      state_differences=state_differences,
                      reference_logs=str(self.reference_dir),
                      reference_checkpoint=str(expected_checkpoint),
                      expected_state_hash=expected_state,
                      actual_state_hash=actual_state)
        return result


def save_checkpoint(st: dict, arm: str, step: int, outdir: Path) -> Path:
    path = outdir / "ckpts" / f"{arm}_step{step}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(step=step, arm=arm, net=st["net"].state_dict(),
                    env=st["env"].state_dict(), teacher=st["teacher"].state_dict(),
                    running_mean=st["running_mean"].clone(), runs=st["runs"]), path)
    return path


def train_arm(st: dict, recorder, probe_steps: list[int], total: int,
              outdir: Path, checkpoints: list[int]) -> float:
    probe_set = set(int(v) for v in probe_steps)
    checkpoint_set = set(int(v) for v in checkpoints)
    net, env, teacher = st["net"], st["env"], st["teacher"]
    alpha = st["center_alpha"]
    t0 = time.time()
    for t in range(total):
        if t in checkpoint_set:
            save_checkpoint(st, st["arm"], t, outdir)
        if t in probe_set:
            recorder(st, t)
        x = env.step()
        y = teacher(x)
        st["running_mean"].mul_(1 - alpha).add_(alpha * x)
        pres, acts, yhat = net.forward_layers(x)
        grads = net.grads_layers(x, pres, acts, yhat - y)
        net.sgd_step_layers(st["lr"], *grads)
    if total in probe_set:
        recorder(st, total)
    if total in checkpoint_set:
        save_checkpoint(st, st["arm"], total, outdir)
    return time.time() - t0


def write_arm_logs(outdir: Path, arm: str, st: dict,
                   rec: PhaseRecorder) -> list[Path]:
    logdir = outdir / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ri, run in enumerate(st["runs"]):
        payload = dict(step=rec.steps, run_id=np.array(run["run_id"]),
                       arm=np.array(arm), seed=np.int64(run["seed"]),
                       task_period=np.int64(run["period"]),
                       state_hash_final=np.array(json.dumps(
                           _seed_state_hashes(st, ri), sort_keys=True)))
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


def layer_rows(arm: str, st: dict, rec: PhaseRecorder,
               period: int) -> list[dict]:
    rows = []
    for si, step in enumerate(rec.steps):
        task_end = bool(step > 0 and step % period == 0)
        task = int(step // period) if task_end else ""
        for ri, run in enumerate(st["runs"]):
            for li, layer in enumerate(rec.layers, start=1):
                row = dict(arm=arm, run_id=run["run_id"], seed=run["seed"],
                           step=int(step), task=task, task_end=int(task_end), layer=li)
                for key in LOG_LAYER_KEYS:
                    value = layer[key][si, ri]
                    row[key] = int(value) if key in ("n_na", "strict_dead", "alive") else float(value)
                row["strict_dead_frac"] = row["strict_dead"] / st["hidden"][li - 1]
                for key in rec.run:
                    row[key] = float(rec.run[key][si, ri])
                rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if not (np.isfinite(x).all() and np.isfinite(y).all()) or len(x) < 2:
        return float("nan")
    rx, ry = _rankdata(x), _rankdata(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    denom = np.sqrt(np.square(rx).sum() * np.square(ry).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


def _jackknife_se(values: np.ndarray, statistic) -> float:
    n = len(values)
    if n < 2:
        return float("nan")
    jk = np.array([statistic(np.delete(values, i)) for i in range(n)], dtype=np.float64)
    return float(np.sqrt((n - 1) / n * np.square(jk - jk.mean()).sum()))


def bootstrap_t(values: np.ndarray, draws: np.ndarray, statistic: str) -> dict:
    """Studentized seed bootstrap for a mean or a median."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) != draws.shape[1]:
        raise ValueError("bootstrap requires one finite value for every preregistered seed")
    stat = np.mean if statistic == "mean" else np.median
    point = float(stat(values))
    se0 = (float(values.std(ddof=1) / math.sqrt(len(values)))
           if statistic == "mean" else _jackknife_se(values, stat))
    samples = values[draws]
    boot = stat(samples, axis=1)
    if statistic == "mean":
        se = samples.std(axis=1, ddof=1) / math.sqrt(len(values))
    else:
        n = samples.shape[1]
        jk = np.stack([np.median(np.delete(samples, i, axis=1), axis=1)
                       for i in range(n)], axis=1)
        se = np.sqrt((n - 1) / n * np.square(jk - jk.mean(axis=1, keepdims=True)).sum(axis=1))
    if se0 == 0:
        return dict(point=point, ci_lo=point, ci_hi=point, boot_ok=int(len(boot)),
                    n_seed=int(len(values)), statistic=statistic,
                    ci_method="studentized")
    good = np.isfinite(boot) & np.isfinite(se) & (se > 0)
    if not good.any():
        return dict(point=point, ci_lo=float("nan"), ci_hi=float("nan"), boot_ok=0,
                    n_seed=int(len(values)), statistic=statistic,
                    ci_method="studentized")
    t = (boot[good] - point) / se[good]
    qlo, qhi = np.quantile(t, [0.025, 0.975])
    return dict(point=point, ci_lo=float(point - qhi * se0),
                ci_hi=float(point - qlo * se0), boot_ok=int(good.sum()),
                n_seed=int(len(values)), statistic=statistic,
                ci_method="studentized")


def analyze(cfg: dict, states: dict[str, dict], recorders: dict[str, PhaseRecorder],
            outdir: Path, sanity: dict, elapsed: dict[str, float]) -> dict:
    P = cfg["phase0"]
    period = int(P["task_period"])
    nseed = len(cfg["common"]["seeds"])
    rng = np.random.default_rng(int(P["bootstrap_seed"]))
    draws = rng.integers(0, nseed, size=(int(P["bootstrap_B"]), nseed))

    l2 = recorders["L2"]
    endpoint_idx = np.flatnonzero((l2.steps > 0) & (l2.steps % period == 0))
    tasks = (l2.steps[endpoint_idx] // period).astype(int)
    early_lo, early_hi = map(int, P["early_tasks"])
    late_lo, late_hi = map(int, P["late_tasks"])
    early_idx = endpoint_idx[(tasks >= early_lo) & (tasks <= early_hi)]
    late_idx = endpoint_idx[(tasks >= late_lo) & (tasks <= late_hi)]
    early_seed = l2.run["unfit"][early_idx].mean(axis=0)
    late_seed = l2.run["unfit"][late_idx].mean(axis=0)
    dU_seed = late_seed - early_seed
    g0 = bootstrap_t(dU_seed, draws, "mean")
    late_mean, early_mean = float(late_seed.mean()), float(early_seed.mean())
    threshold = float(P["unfit_threshold"])
    contains_zero = bool(g0["ci_lo"] <= 0 <= g0["ci_hi"])
    if g0["ci_lo"] > 0 and late_mean >= threshold:
        verdict = "LOP_PRESENT"
    elif contains_zero and late_mean < threshold:
        verdict = "LOP_ABSENT"
    else:
        verdict = "INCONCLUSIVE"
    verdict_row = dict(metric="G0", arm="L2", verdict=verdict,
                       dU=g0["point"], ci_lo=g0["ci_lo"], ci_hi=g0["ci_hi"],
                       early_unfit=early_mean, late_unfit=late_mean,
                       unfit_threshold=threshold, n_seed=g0["n_seed"],
                       bootstrap_B=int(P["bootstrap_B"]), boot_ok=g0["boot_ok"],
                       bootstrap_seed=int(P["bootstrap_seed"]),
                       ci_method=g0["ci_method"])
    write_csv(outdir / "verdict.csv", [verdict_row])

    all_rows = []
    for arm in ("L1", "L2"):
        all_rows.extend(layer_rows(arm, states[arm], recorders[arm], period))
    write_csv(outdir / "layer_stats.csv", all_rows)

    trends = []
    for arm in ("L1", "L2"):
        rec = recorders[arm]
        end = np.flatnonzero((rec.steps > 0) & (rec.steps % period == 0))
        task_no = rec.steps[end].astype(np.float64) / period
        for li, layer in enumerate(rec.layers, start=1):
            seed_rho = np.array([spearman(task_no, layer["median_M"][end, ri])
                                 for ri in range(nseed)])
            interval = bootstrap_t(seed_rho, draws, "median")
            trends.append(dict(arm=arm, layer=li, seed_spearman=seed_rho.tolist(),
                               increase=bool(interval["ci_lo"] > 0), **interval))

    final_levels = []
    for arm in ("L1", "L2"):
        rec = recorders[arm]
        final_i = int(np.where(rec.steps == int(cfg["common"]["total_steps"]))[0][0])
        for li, layer in enumerate(rec.layers, start=1):
            values = layer["median_M"][final_i]
            final_levels.append(dict(arm=arm, layer=li,
                                     median_across_seed=float(np.median(values)),
                                     q25_across_seed=float(np.quantile(values, 0.25)),
                                     q75_across_seed=float(np.quantile(values, 0.75)),
                                     max_n_na=int(layer["n_na"].max())))

    lines = ["# mlp2_phase0_0829 summary", "",
             "## G0", "",
             f"**{verdict}** — dU={g0['point']:.6g}, 95% bootstrap-t CI "
             f"[{g0['ci_lo']:.6g}, {g0['ci_hi']:.6g}], late unfit={late_mean:.6g} "
             f"(threshold={threshold:g}).", "",
             "Depths are not treated as paired; G0 is an L2 within-run time comparison.", "",
             "## Final task-end wall coordinate", "",
             "| arm | layer | median seed median(M) | seed IQR | max NA units |",
             "|---|---:|---:|---:|---:|"]
    for row in final_levels:
        lines.append(f"| {row['arm']} | {row['layer']} | {row['median_across_seed']:.6g} | "
                     f"[{row['q25_across_seed']:.6g}, {row['q75_across_seed']:.6g}] | "
                     f"{row['max_n_na']} |")
    lines += ["", "## Task-end trend", "",
              "Each seed contributes Spearman(task, median M); the reported point is the "
              "median over seeds with a studentized bootstrap interval.", "",
              "| arm | layer | median rho | 95% bootstrap-t CI | increase |",
              "|---|---:|---:|---:|---:|"]
    for row in trends:
        lines.append(f"| {row['arm']} | {row['layer']} | {row['point']:.6g} | "
                     f"[{row['ci_lo']:.6g}, {row['ci_hi']:.6g}] | "
                     f"{'YES' if row['increase'] else 'NO'} |")
    lines += ["", "## Sanity", "", f"- S0 legacy identity: **{'PASS' if sanity['S0']['pass_'] else 'FAIL'}**",
              f"- S1/S2 exact identities: **{'PASS' if sanity['S1_S2_all_pass'] else 'FAIL'}**",
              f"- S3 OMP threads: **{'PASS' if sanity['S3']['pass_'] else 'FAIL'}**",
              "", "All layer statistics use the exact 32-pattern support. M and B are "
              "static wall-condition coordinates, not normalized dynamical variables.", ""]
    (outdir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return dict(g0=verdict_row, trends=trends, final_levels=final_levels,
                elapsed_sec=elapsed)


def provenance(cfg_path: Path, cfg: dict, outdir: Path, sanity: dict,
               analysis: dict | None, logs: list[Path], started: float,
               smoke: bool) -> dict:
    spec_path = Path(ROOT) / cfg["spec"]
    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                           text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--short"], cwd=ROOT,
                                        text=True).splitlines()
    except (OSError, subprocess.CalledProcessError):
        git_hash, dirty = None, []
    files = [p for p in logs + [outdir / "config_used.yaml"] if p.exists()]
    if not smoke:
        files += [p for p in (outdir / "verdict.csv", outdir / "layer_stats.csv",
                              outdir / "summary.md") if p.exists()]
    return dict(experiment="mlp2_phase0_0829", smoke=smoke,
                created=time.strftime("%Y-%m-%d %H:%M:%S %z"),
                elapsed_sec=round(time.time() - started, 3), command=sys.argv,
                cwd=os.getcwd(), python=sys.version, platform=platform.platform(),
                torch=torch.__version__, numpy=np.__version__, device=cfg["common"]["device"],
                git_hash=git_hash, git_dirty=dirty,
                config=str(cfg_path), config_sha256=_sha_file(cfg_path),
                spec=str(spec_path), spec_sha256=_sha_file(spec_path) if spec_path.exists() else None,
                sanity=sanity, analysis=analysis,
                output_sha256={str(p.relative_to(outdir)): _sha_file(p) for p in files})


def apply_smoke(cfg: dict) -> dict:
    import copy
    out = copy.deepcopy(cfg)
    out["common"]["total_steps"] = SMOKE_STEPS
    out["common"]["seeds"] = [0]
    out["common"]["checkpoints"] = []
    return out


def run(cfg_path: Path, cfg: dict, device: str, outdir: Path, *, smoke: bool) -> dict:
    started = time.time()
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / "config_used.yaml").open("w") as fh:
        yaml.safe_dump(cfg, fh, allow_unicode=True, sort_keys=False)
    omp = require_omp(cfg)
    P, C = cfg["phase0"], cfg["common"]
    total, period = int(C["total_steps"]), int(P["task_period"])
    phase_steps = list(range(0, total + 1, int(C["lop_every"])))
    if phase_steps[-1] != total:
        phase_steps.append(total)

    states, recorders, logs, elapsed = {}, {}, [], {}
    sanity = dict(S3=omp, S0={"pass_": None}, preflight_identity={})
    arm_map = {a["name"]: a for a in cfg["arms"]}
    for arm in ("L1", "L2"):
        print(f"[{arm}] setup: hidden={arm_map[arm]['hidden']} seeds={C['seeds']} "
              f"steps={total:,}", flush=True)
        st = setup_arm(cfg, arm_map[arm], device)
        _, preflight = exact_layer_record(st, float(P["sigma_degenerate_tol"]))
        preflight["pass_"] = identity_sanity_pass(
            preflight, float(cfg["sanity"]["s1_identity_tol"]))
        sanity["preflight_identity"][arm] = preflight
        if not preflight["pass_"]:
            raise RuntimeError(f"{arm} preflight S1/S2 failed: {preflight}")
        rec = PhaseRecorder(phase_steps, st, float(P["sigma_degenerate_tol"]),
                            float(cfg["sanity"]["s1_identity_tol"]))
        if arm == "L1" and not smoke:
            reference = Path(ROOT) / cfg["sanity"]["s0_bit_equality_ref"]
            dense_steps = legacy_record_steps(total, period, 100, 1000)
            runner = S0Replay(reference, dense_steps, list(C["seeds"]), rec)
            probe_steps = dense_steps
        else:
            runner, probe_steps = rec, phase_steps
        elapsed[arm] = train_arm(st, runner, probe_steps, total, outdir,
                                 list(C.get("checkpoints", [])))
        rec.check_complete()
        if arm == "L1" and not smoke:
            ref_ckpt = reference.parent.parent / "ckpts" / f"A_w100_step{total}.pt"
            sanity["S0"] = runner.finish(st, ref_ckpt)
            print(f"[L1] S0 {'PASS' if sanity['S0']['pass_'] else 'FAIL'}", flush=True)
            if not sanity["S0"]["pass_"]:
                raise RuntimeError(f"S0 failed; refusing to start L2: {sanity['S0']}")
        states[arm], recorders[arm] = st, rec
        logs.extend(write_arm_logs(outdir, arm, st, rec))
        print(f"[{arm}] complete in {elapsed[arm]:.1f}s", flush=True)

    sanity["S1_S2"] = {arm: recorders[arm].sanity() for arm in ("L1", "L2")}
    sanity["S1_S2_all_pass"] = bool(all(v["pass_"] for v in sanity["S1_S2"].values()))
    if not sanity["S1_S2_all_pass"]:
        raise RuntimeError(f"S1/S2 failed: {sanity['S1_S2']}")

    if smoke:
        schemas, nonfinite_counts, shape_errors = {}, {}, []
        for path in logs:
            with np.load(path, allow_pickle=False) as z:
                schemas[path.name] = {k: list(z[k].shape) for k in z.files}
                nonfinite_counts[path.name] = {
                    k: int((~np.isfinite(z[k])).sum()) for k in z.files
                    if z[k].dtype.kind in "fc"
                }
                expected_n = len(phase_steps)
                if z["step"].shape != (expected_n,):
                    shape_errors.append(f"{path.name}: step={z['step'].shape}")
                for key in ("unfit", "signal_var", "residual_var", "eval_loss_exact"):
                    if z[key].shape != (expected_n,):
                        shape_errors.append(f"{path.name}: {key}={z[key].shape}")
                if z["flip_state"].shape != (expected_n, int(cfg["condA"]["f"])):
                    shape_errors.append(f"{path.name}: flip_state={z['flip_state'].shape}")
                arm_name = str(z["arm"])
                hidden = arm_map[arm_name]["hidden"]
                for li, width in enumerate(hidden, start=1):
                    for key in LOG_UNIT_KEYS:
                        name = f"layer{li}_{key}"
                        if z[name].shape != (expected_n, int(width)):
                            shape_errors.append(f"{path.name}: {name}={z[name].shape}")
                    for key in LOG_LAYER_KEYS:
                        name = f"layer{li}_{key}"
                        if z[name].shape != (expected_n,):
                            shape_errors.append(f"{path.name}: {name}={z[name].shape}")
        all_finite = all(count == 0 for per_file in nonfinite_counts.values()
                         for count in per_file.values())
        smoke_pass = bool(all_finite and not shape_errors
                          and sanity["S1_S2_all_pass"] and sanity["S3"]["pass_"])
        smoke_sanity = dict(pass_=smoke_pass, total_steps=total, seeds=list(C["seeds"]),
                            run_ids=[r["run_id"] for st in states.values() for r in st["runs"]],
                            schemas=schemas, nonfinite_counts=nonfinite_counts,
                            all_columns_finite=all_finite, shape_errors=shape_errors,
                            sanity=sanity)
        with (outdir / "smoke_sanity.json").open("w") as fh:
            json.dump(smoke_sanity, fh, indent=2, ensure_ascii=False, default=str)
        if not smoke_pass:
            raise RuntimeError(f"S4 smoke failed: {smoke_sanity}")
        result = None
    else:
        result = analyze(cfg, states, recorders, outdir, sanity, elapsed)

    prov = provenance(cfg_path, cfg, outdir, sanity, result, logs, started, smoke)
    with (outdir / "provenance.json").open("w") as fh:
        json.dump(prov, fh, indent=2, ensure_ascii=False, default=str)
    print(f"ALL DONE -> {outdir}", flush=True)
    return dict(sanity=sanity, analysis=result, logs=[str(p) for p in logs])


def run_s0_only(cfg_path: Path, cfg: dict, device: str, outdir: Path) -> dict:
    """Execute the mandatory full L=1 legacy replay without exposing §5 values."""
    started = time.time()
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / "config_used.yaml").open("w") as fh:
        yaml.safe_dump(cfg, fh, allow_unicode=True, sort_keys=False)
    omp = require_omp(cfg)
    total = int(cfg["common"]["total_steps"])
    period = int(cfg["phase0"]["task_period"])
    arm = next(a for a in cfg["arms"] if a["name"] == "L1")
    st = setup_arm(cfg, arm, device)
    reference = Path(ROOT) / cfg["sanity"]["s0_bit_equality_ref"]
    steps = legacy_record_steps(total, period, 100, 1000)
    replay = S0Replay(reference, steps, list(cfg["common"]["seeds"]), None)
    elapsed = train_arm(st, replay, steps, total, outdir, [])
    ref_ckpt = reference.parent.parent / "ckpts" / f"A_w100_step{total}.pt"
    s0 = replay.finish(st, ref_ckpt)
    result = dict(S0=s0, S3=omp, elapsed_sec=elapsed,
                  pass_=bool(s0["pass_"] and omp["pass_"]))
    with (outdir / "s0_sanity.json").open("w") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False, default=str)
    prov = provenance(cfg_path, cfg, outdir, result, None, [], started, smoke=True)
    with (outdir / "provenance.json").open("w") as fh:
        json.dump(prov, fh, indent=2, ensure_ascii=False, default=str)
    print(f"S0 {'PASS' if result['pass_'] else 'FAIL'} -> {outdir}", flush=True)
    if not result["pass_"]:
        raise RuntimeError(f"S0 failed: {s0}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mlp2_phase0_0829.yaml")
    parser.add_argument("--smoke", action="store_true",
                        help="seed 0 x 30k; structural checks only, no §5 aggregation")
    parser.add_argument("--s0-only", action="store_true",
                        help="full L1 legacy replay only; no §5 aggregation")
    parser.add_argument("--outdir")
    parser.add_argument("--device")
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = load_config(str(cfg_path))
    if args.smoke and args.s0_only:
        parser.error("--smoke and --s0-only are mutually exclusive")
    validate_config(cfg, smoke=args.smoke)
    if args.smoke:
        cfg = apply_smoke(cfg)
    if args.device:
        cfg["common"]["device"] = args.device
    device = pick_device(cfg)
    if device != "cpu":
        raise ValueError("mlp2_phase0_0829 is CPU-only")
    default_name = ("_smoke_mlp2_phase0_0829" if args.smoke else
                    "_s0_mlp2_phase0_0829" if args.s0_only else cfg_path.stem)
    outdir = (Path(args.outdir).resolve() if args.outdir else
              Path(ROOT) / "results" / default_name)
    if args.s0_only:
        run_s0_only(cfg_path, cfg, device, outdir)
    else:
        run(cfg_path, cfg, device, outdir, smoke=args.smoke)


if __name__ == "__main__":
    main()
