"""Permuted MNIST harness for spec/PermutedMNIST_spec_0905.

New box. Deliberately shares no code with the condA harness (src/common.py,
src/envs.py, ...): the spec (§0) forbids inheriting it, so none of the existing
checks (S-pair / S-limit / exact-support identities) carry over. The checks in
§7 are implemented fresh in this file.

Stages
  checks : §7 S-data / S-perm / S-init / S-act / S-mob
  cost   : §7 S-cost  (1 arm x 5 tasks wall clock -> extrapolation)
  probe  : §4.1 stage 0  (pmnist_probe_0905, arms R/LR/LIN, 50 tasks)
  lrcal  : §4.2 stage 1  (pmnist_lrcal_0905)
  main   : §4.3 stage 2  (pmnist_main_0905)
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

# Determinism must be requested before the first CUDA workspace is allocated.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "data" / "mnist"
DIMS = (784, 100, 100, 10)
BATCH = 16
TASK_EXAMPLES = 10_000
STEPS_PER_TASK = TASK_EXAMPLES // BATCH          # 625
N_CLASSES = 10
DEAD_TOL = 1e-6                                   # §4.4 |phi'| < 1e-6


# --------------------------------------------------------------------------
# activations (§2.2). phi' is written analytically; S-act checks it against
# autograd.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Activation:
    name: str
    kind: str
    param: float = 0.0

    def phi(self, z: torch.Tensor) -> torch.Tensor:
        if self.kind == "relu":
            return torch.clamp(z, min=0.0)
        if self.kind == "leaky":
            return torch.where(z > 0, z, self.param * z)
        if self.kind == "snake":
            a = self.param
            return z + torch.sin(a * z) ** 2 / a
        if self.kind == "linear":
            return z
        raise ValueError(self.kind)

    def dphi(self, z: torch.Tensor) -> torch.Tensor:
        if self.kind == "relu":
            return (z > 0).to(z.dtype)
        if self.kind == "leaky":
            return torch.where(z > 0, torch.ones_like(z), torch.full_like(z, self.param))
        if self.kind == "snake":
            return 1.0 + torch.sin(2.0 * self.param * z)
        if self.kind == "linear":
            return torch.ones_like(z)
        raise ValueError(self.kind)


ARMS = {
    "LIN0": Activation("LIN0", "linear"),      # depth-0 control: 784->10, convex
    "R":   Activation("R",   "relu"),
    "LR":  Activation("LR",  "leaky",  0.1),
    "SN3": Activation("SN3", "snake",  3.0),
    "SN1": Activation("SN1", "snake",  1.0),
    "SN05": Activation("SN05", "snake", 0.5),  # Ziyin Prop.1 optimum ~0.56
    "SN03": Activation("SN03", "snake", 0.3),  # gate e-fold at Adam's W=2.43
    "SN02": Activation("SN02", "snake", 0.2),
    "SN06": Activation("SN06", "snake", 0.6),  # S-ema-off reference (SNA with beta=0)
    "SNA":  Activation("SNA",  "adaptive_snake", 0.6),   # marker; state built per run
    "SN01": Activation("SN01", "snake", 0.1),  # nearly linear: sin^2(az)/a ~ a z^2
    "LIN": Activation("LIN", "linear"),
}


class AdaptiveSnake:
    """Snake whose alpha follows the preactivation spread, per hidden unit.

    alpha_i = clip(c / W_i, lo, hi), W_i = sqrt(EMA_beta[var_batch(z_i)]).
    V holds no grad, so alpha is detached -- the same standing as BatchNorm's
    running statistics (spec pmnist_adapt_0905 §2.2, class B).  beta=0 freezes
    V at its init of 1, which makes this bit-identical to fixed alpha=c (S-ema-off).
    """
    name, kind = "SNA", "adaptive_snake"

    def __init__(self, c: float, beta: float, device, lo: float = 0.05, hi: float = 3.0,
                 widths=(DIMS[1], DIMS[2])):
        self.c, self.beta, self.lo, self.hi = c, beta, lo, hi
        self.V = [torch.ones(w, device=device) for w in widths]

    def alpha(self, layer: int) -> torch.Tensor:
        return (self.c / self.V[layer].sqrt()).clamp(self.lo, self.hi)

    def phi(self, z, layer=0):
        a = self.alpha(layer)
        # `* a.reciprocal()` rather than `/ a`: the fixed-alpha Snake divides by a
        # Python float, which the CUDA kernel folds into a reciprocal-multiply.
        # Matching that op order makes SNA(beta=0) bit-identical to fixed alpha=c
        # (S-ema-off); a true tensor division differs by 1 ulp and the trajectory
        # drifts to ~1e-6 over a few thousand steps.
        return z + torch.sin(a * z) ** 2 * a.reciprocal()

    def dphi(self, z, layer=0):
        return 1.0 + torch.sin(2.0 * self.alpha(layer) * z)

    @torch.no_grad()
    def update(self, z1, z2):
        if self.beta == 0:
            return
        for l, z in ((0, z1), (1, z2)):
            self.V[l].mul_(1 - self.beta).add_(self.beta * z.var(0, unbiased=False))

    @torch.no_grad()
    def stats(self) -> dict:
        out = {}
        for l in (0, 1):
            a = self.alpha(l); W = self.V[l].sqrt()
            raw = self.c / W
            out[f"alpha_med_l{l+1}"] = float(a.median()); out[f"alpha_min_l{l+1}"] = float(a.min())
            out[f"alpha_max_l{l+1}"] = float(a.max())
            out[f"alpha_clip_frac_l{l+1}"] = float(((raw < self.lo) | (raw > self.hi)).float().mean())
            out[f"two_alpha_W_med_l{l+1}"] = float((2 * a * W).median())
        return out


# --------------------------------------------------------------------------
# rng: one independent stream per role, derived from sha256 so that neither the
# seed number nor the role name can collide into the same sequence.
# --------------------------------------------------------------------------

def stream(role: str, seed: int) -> torch.Generator:
    h = hashlib.sha256(f"pmnist_0905|{role}|{seed}".encode()).digest()
    g = torch.Generator(device="cpu")
    g.manual_seed(int.from_bytes(h[:8], "little") & ((1 << 63) - 1))
    return g


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

def _read_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as fh:
        raw = fh.read()
    magic = int.from_bytes(raw[:4], "big")
    ndim = magic & 0xFF
    shape = [int.from_bytes(raw[4 + 4 * i: 8 + 4 * i], "big") for i in range(ndim)]
    return np.frombuffer(raw[4 + 4 * ndim:], dtype=np.uint8).reshape(shape)


class Mnist:
    FILES = {
        "train_x": "train-images-idx3-ubyte.gz",
        "train_y": "train-labels-idx1-ubyte.gz",
        "test_x":  "t10k-images-idx3-ubyte.gz",
        "test_y":  "t10k-labels-idx1-ubyte.gz",
    }

    def __init__(self, device: torch.device):
        self.sha256 = {}
        arr = {}
        for key, fname in self.FILES.items():
            p = DATA_DIR / fname
            self.sha256[fname] = hashlib.sha256(p.read_bytes()).hexdigest()
            arr[key] = _read_idx(p)
        self.train_x = torch.from_numpy(
            arr["train_x"].reshape(-1, 784).astype(np.float32) / 255.0).to(device)
        self.train_y = torch.from_numpy(arr["train_y"].astype(np.int64)).to(device)
        self.test_x = torch.from_numpy(
            arr["test_x"].reshape(-1, 784).astype(np.float32) / 255.0).to(device)
        self.test_y = torch.from_numpy(arr["test_y"].astype(np.int64)).to(device)
        # per-class index lists on cpu, used by the stratified draw
        ytr = arr["train_y"].astype(np.int64)
        self.class_idx = [np.nonzero(ytr == c)[0] for c in range(N_CLASSES)]
        self.class_counts = np.array([len(ix) for ix in self.class_idx])
        self.quota = hamilton(self.class_counts, TASK_EXAMPLES)


def hamilton(counts: np.ndarray, total: int) -> np.ndarray:
    """Largest-remainder apportionment: keeps class ratios and sums to `total`."""
    exact = counts / counts.sum() * total
    base = np.floor(exact).astype(np.int64)
    rem = total - base.sum()
    if rem:
        order = np.argsort(-(exact - base), kind="stable")
        base[order[:rem]] += 1
    return base


def stratified_draw(mnist: Mnist, g: torch.Generator) -> torch.Tensor:
    """Indices of TASK_EXAMPLES training images, class ratios preserved."""
    picks = []
    for c in range(N_CLASSES):
        pool = mnist.class_idx[c]
        sel = torch.randperm(len(pool), generator=g)[: mnist.quota[c]].numpy()
        picks.append(pool[sel])
    return torch.from_numpy(np.concatenate(picks))


# --------------------------------------------------------------------------
# net
# --------------------------------------------------------------------------

def init_params(seed: int, device: torch.device, dims=None):
    """PyTorch nn.Linear default init: U(-1/sqrt(fan_in), +1/sqrt(fan_in)).

    Activation-independent on purpose (S-init): the arm must not enter the RNG.
    Drawn on cpu so the bit-comparison in S-init is device-independent.
    """
    g = stream("init", seed)
    dims = dims or DIMS
    params = []
    for i in range(len(dims) - 1):
        bound = 1.0 / math.sqrt(dims[i])
        W = (torch.rand((dims[i + 1], dims[i]), generator=g, dtype=torch.float32) * 2 - 1) * bound
        b = (torch.rand((dims[i + 1],), generator=g, dtype=torch.float32) * 2 - 1) * bound
        params += [W.to(device).requires_grad_(True), b.to(device).requires_grad_(True)]
    return params


def forward(params, x, act: Activation):
    if len(params) == 2:                       # LIN0: single affine map, no hidden layer
        W1, b1 = params
        z = x @ W1.T + b1
        return z, z, z, z, z
    W1, b1, W2, b2, W3, b3 = params
    ada = isinstance(act, AdaptiveSnake)
    z1 = x @ W1.T + b1
    a1 = act.phi(z1, 0) if ada else act.phi(z1)
    z2 = a1 @ W2.T + b2
    a2 = act.phi(z2, 1) if ada else act.phi(z2)
    return z1, a1, z2, a2, a2 @ W3.T + b3


# --------------------------------------------------------------------------
# metrics (§4.4), evaluated on the full permuted test set
# --------------------------------------------------------------------------

def eff_rank(a: torch.Tensor) -> float:
    """exp(entropy of sigma_i / sum sigma_i) of the activation matrix.

    Same definition as src/lop_metrics.py:160. Computed from the Gram matrix in
    float64 (deterministic, and 100x100 instead of 10000x100).
    """
    g = (a.double().T @ a.double())
    ev = torch.linalg.eigvalsh(g.cpu()).clamp_min(0.0)
    s = ev.sqrt()
    tot = s.sum()
    if tot <= 0:
        return 0.0
    p = s / tot
    return float(torch.exp(-(p * p.clamp_min(1e-300).log()).sum()))


@torch.no_grad()
def evaluate(params, mnist: Mnist, perm: torch.Tensor, act: Activation) -> dict:
    x = mnist.test_x[:, perm]
    z1, a1, z2, a2, logits = forward(params, x, act)
    out = {"acc": float((logits.argmax(1) == mnist.test_y).float().mean())}
    ada = isinstance(act, AdaptiveSnake)
    for li, (tag, z, a) in enumerate((("l1", z1, a1), ("l2", z2, a2))):
        d = act.dphi(z, li) if ada else act.dphi(z)
        out[f"dead_frac_{tag}"] = float((d.abs().amax(0) < DEAD_TOL).float().mean())
        out[f"zeroout_{tag}"] = float((a.abs().amax(0) == 0).float().mean())
        out[f"zbar_{tag}"] = float(z.mean(0).median())
        # per-unit spread of the preactivation over the test set. For Snake the
        # relevant scale is alpha*W: phi' = 1+sin(2*alpha*z) is averaged over the
        # unit's input range, so once alpha*W spans a period the mean pins to 1
        # and no gating structure survives.
        out[f"zsd_{tag}"] = float(z.std(0).median())
        out[f"zbar_min_{tag}"] = float(z.mean(0).amin())
        out[f"mob_{tag}"] = float(d.mean(0).median())
        out[f"eff_rank_{tag}"] = eff_rank(a)
    for i, tag in enumerate(("l1", "l2", "l3")):
        if 2 * i >= len(params):
            break
        W = params[2 * i]
        out[f"w_norm_{tag}"] = float(W.norm(dim=1).median())
    if ada:
        out.update(act.stats())
    return out


# --------------------------------------------------------------------------
# one (arm, seed, lr) run
# --------------------------------------------------------------------------

def run_one(arm: str, seed: int, lr: float, n_tasks: int, mnist: Mnist,
            device: torch.device, progress=None, optimizer: str = "sgd",
            epochs: int = 1, c: float = 0.6, beta: float = 0.01,
            iv: str = "none", bwt: bool = False, relearn: bool = False,
            relearn_rows: list | None = None) -> tuple[list[dict], dict]:
    act = ARMS[arm]
    past_perms = []          # bwt: permutations of earlier tasks (deterministic, replayable)
    PROBE_TASKS = (1, 100, 180, 199)
    CKPTS = (0, 25, 50, 100, 200, 400, 625)
    first_curve = {}         # relearn: acc at CKPTS during the first exposure of each probe task
    if act.kind == "adaptive_snake":
        act = AdaptiveSnake(c, beta, device)          # fresh statistics per run
    params = init_params(seed, device, (784, 10) if arm == "LIN0" else None)
    # weight-space interventions (spec class C) reuse the other session's
    # mechanism code without editing its file; only the loop here differs
    # (Adam / epochs).  cbp under Adam: replaced rows keep their stale moments --
    # recorded as a limitation in provenance.
    ivo = None
    if iv != "none":
        import pmnist_lopcmp_0905 as L
        ivo = L.Intervention.parse(iv)
        p0 = [q.detach().clone() for q in params] if ivo.kind == "l2init" else None
        cbp = L.CbpState(seed, device) if ivo.kind == "cbp" else None
    # Adam moments live for the whole run: resetting them between tasks would
    # break the continual definition in spec §3 the same way resetting weights does.
    adam = ([torch.zeros_like(q) for q in params],
            [torch.zeros_like(q) for q in params], [0]) if optimizer == "adam" else None
    g_perm, g_data, g_batch = stream("perm", seed), stream("data", seed), stream("batch", seed)
    rows, diverged = [], {"diverged": False, "task": None, "step": None, "seed": seed, "arm": arm}

    for t in range(1, n_tasks + 1):
        perm = torch.randperm(784, generator=g_perm).to(device)
        idx = stratified_draw(mnist, g_data).to(device)
        order = torch.randperm(TASK_EXAMPLES, generator=g_batch).to(device)
        xs = mnist.train_x[idx][:, perm][order]
        ys = mnist.train_y[idx][order]

        bad_step = torch.full((), -1, dtype=torch.long, device=device)
        probe_this = relearn and t in PROBE_TASKS
        if probe_this:
            with torch.no_grad():
                first_curve[(t, 0)] = float((forward(params, mnist.test_x[:, perm], act)[4].argmax(1) == mnist.test_y).float().mean())
        for s in range(STEPS_PER_TASK * epochs):
            # epochs>1 re-walks the same 10,000 examples in the same order, so
            # the extra steps buy optimisation time, not fresh data.
            j = s % STEPS_PER_TASK
            xb = xs[j * BATCH:(j + 1) * BATCH]
            yb = ys[j * BATCH:(j + 1) * BATCH]
            out = forward(params, xb, act)
            loss = torch.nn.functional.cross_entropy(out[4], yb)
            grads = torch.autograd.grad(loss, params)
            with torch.no_grad():
                bad = ~torch.isfinite(loss)
                bad_step = torch.where((bad_step < 0) & bad,
                                       torch.tensor(s, device=device), bad_step)
                if ivo is not None and ivo.kind in ("l2", "l2init"):
                    grads = [gr + 2.0 * ivo.lam * (q - (p0[i] if p0 is not None else 0.0))
                             for i, (q, gr) in enumerate(zip(params, grads))]
                if adam is None:
                    for p, gr in zip(params, grads):
                        p -= lr * gr
                else:
                    m, v, tc = adam
                    tc[0] += 1
                    b1, b2, eps = 0.9, 0.999, 1e-8
                    c1 = 1 - b1 ** tc[0]
                    c2 = 1 - b2 ** tc[0]
                    for p, gr, mi, vi in zip(params, grads, m, v):
                        mi.mul_(b1).add_(gr, alpha=1 - b1)
                        vi.mul_(b2).addcmul_(gr, gr, value=1 - b2)
                        p -= lr * (mi / c1) / ((vi / c2).sqrt() + eps)
                if ivo is not None and ivo.kind == "cbp":
                    cbp.step(params, [out[1], out[3]], ivo.rho)
                    # CbpState re-draws the input rows and zeroes the output columns of
                    # replaced units but knows nothing about Adam.  It marks them with
                    # age == 0 (every other unit has age >= 1 after its own increment),
                    # so reset their first/second moments here; otherwise a fresh unit
                    # inherits the stale m/v of the unit it replaced.
                    if adam is not None:
                        m, v, _ = adam
                        for li in (0, 1):
                            idx = cbp.age[li] == 0
                            if bool(idx.any()):
                                m[2 * li][idx] = 0; v[2 * li][idx] = 0          # W_in rows
                                m[2 * li + 1][idx] = 0; v[2 * li + 1][idx] = 0  # b_in
                                m[2 * li + 2][:, idx] = 0; v[2 * li + 2][:, idx] = 0  # W_out cols
                if isinstance(act, AdaptiveSnake):
                    act.update(out[0], out[2])          # running var of this batch's preacts
                if probe_this and (s + 1) in CKPTS:
                    first_curve[(t, s + 1)] = float((forward(params, mnist.test_x[:, perm], act)[4].argmax(1) == mnist.test_y).float().mean())

        bs = int(bad_step)
        if bs >= 0 or not all(torch.isfinite(p).all() for p in params):
            diverged.update(diverged=True, task=t,
                            step=(t - 1) * STEPS_PER_TASK * epochs + max(bs, 0))
            rows.append({"arm": arm, "seed": seed, "lr": lr, "task": t, "acc": float("nan")})
            break                                        # S-div: drop, never rescue

        m = evaluate(params, mnist, perm, act)
        if bwt:
            # backward transfer: accuracy now on the permutations of tasks 1, t-1, t-5, t-20.
            # Compared with that task's own acc (already in rows) this gives one-step /
            # 5-step / 20-step BWT and forgetting of the very first task.
            with torch.no_grad():
                for lag, key in ((1, "acc_prev1"), (5, "acc_prev5"), (20, "acc_prev20")):
                    if t - lag >= 1:
                        pp = past_perms[t - lag - 1]
                        m[key] = float((forward(params, mnist.test_x[:, pp], act)[4].argmax(1)
                                        == mnist.test_y).float().mean())
                if t >= 2:
                    m["acc_task1"] = float((forward(params, mnist.test_x[:, past_perms[0]], act)[4].argmax(1)
                                            == mnist.test_y).float().mean())
        past_perms.append(perm)
        rows.append({"arm": arm, "seed": seed, "lr": lr, "task": t, **m})
        if progress:
            progress(t, rows[-1])
    if relearn and not diverged["diverged"] and relearn_rows is not None:
        _relearn_probes(arm, seed, lr, act, params, adam, ivo, p0 if (ivo is not None and ivo.kind == "l2init") else None,
                        past_perms, PROBE_TASKS, CKPTS, first_curve, mnist, device, relearn_rows)
    return rows, diverged


def _train_steps(params, act, adam_state, ivo, p0, xs, ys, lr, n_steps, ckpts, test_x, test_y, out_rows, tag):
    """625-step relearning loop with the same update rule as run_one (Adam or SGD, l2/l2init)."""
    def acc():
        with torch.no_grad():
            return float((forward(params, test_x, act)[4].argmax(1) == test_y).float().mean())
    if 0 in ckpts: out_rows.append({**tag, "step": 0, "acc": acc()})
    for s in range(n_steps):
        j = s % STEPS_PER_TASK
        out = forward(params, xs[j * BATCH:(j + 1) * BATCH], act)
        loss = torch.nn.functional.cross_entropy(out[4], ys[j * BATCH:(j + 1) * BATCH])
        grads = torch.autograd.grad(loss, params)
        with torch.no_grad():
            if ivo is not None and ivo.kind in ("l2", "l2init"):
                grads = [gr + 2.0 * ivo.lam * (q - (p0[i] if p0 is not None else 0.0)) for i, (q, gr) in enumerate(zip(params, grads))]
            if adam_state is None:
                for q, gr in zip(params, grads): q -= lr * gr
            else:
                m, v, tc = adam_state; tc[0] += 1
                b1, b2, eps = 0.9, 0.999, 1e-8; c1, c2 = 1 - b1 ** tc[0], 1 - b2 ** tc[0]
                for q, gr, mi, vi in zip(params, grads, m, v):
                    mi.mul_(b1).add_(gr, alpha=1 - b1); vi.mul_(b2).addcmul_(gr, gr, value=1 - b2)
                    q -= lr * (mi / c1) / ((vi / c2).sqrt() + eps)
            if isinstance(act, AdaptiveSnake): act.update(out[0], out[2])
        if (s + 1) in ckpts: out_rows.append({**tag, "step": s + 1, "acc": acc()})


def _relearn_probes(arm, seed, lr, act, params, adam, ivo, p0, past_perms, probe_tasks, ckpts, first_curve,
                    mnist, device, out_rows):
    """After the last task: re-fit old tasks from the final state ('relearn') and from a fresh
    init ('fresh'); also emit the stored first-exposure curve ('first'). Fresh data subset per
    probe from an independent stream so the probe tests the task, not the exact examples."""
    g_data, g_batch = stream("relearn_data", seed), stream("relearn_batch", seed)
    for t in probe_tasks:
        if t > len(past_perms): continue
        perm = past_perms[t - 1]
        idx = stratified_draw(mnist, g_data).to(device); order = torch.randperm(TASK_EXAMPLES, generator=g_batch).to(device)
        xs, ys = mnist.train_x[idx][:, perm][order], mnist.train_y[idx][order]
        test_x = mnist.test_x[:, perm]
        for k, v in first_curve.items():
            if k[0] == t: out_rows.append({"arm": arm, "seed": seed, "probe_task": t, "kind": "first", "step": k[1], "acc": v})
        # relearn from the end state (clone everything so probes are independent)
        P = [q.detach().clone().requires_grad_(True) for q in params]
        A = ([x.clone() for x in adam[0]], [x.clone() for x in adam[1]], [adam[2][0]]) if adam is not None else None
        act_r = act
        if isinstance(act, AdaptiveSnake):
            act_r = AdaptiveSnake(act.c, act.beta, device); act_r.V = [x.clone() for x in act.V]
        _train_steps(P, act_r, A, ivo, p0, xs, ys, lr, STEPS_PER_TASK, ckpts, test_x, mnist.test_y, out_rows,
                     {"arm": arm, "seed": seed, "probe_task": t, "kind": "relearn"})
        # fresh init reference (no memory): same task, same data, from theta_0
        F = init_params(seed, device)
        A0 = ([torch.zeros_like(q) for q in F], [torch.zeros_like(q) for q in F], [0]) if adam is not None else None
        act_f = AdaptiveSnake(act.c, act.beta, device) if isinstance(act, AdaptiveSnake) else act
        p0f = [q.detach().clone() for q in F] if (ivo is not None and ivo.kind == "l2init") else None
        _train_steps(F, act_f, A0, ivo, p0f, xs, ys, lr, STEPS_PER_TASK, ckpts, test_x, mnist.test_y, out_rows,
                     {"arm": arm, "seed": seed, "probe_task": t, "kind": "fresh"})


# --------------------------------------------------------------------------
# §7 checks
# --------------------------------------------------------------------------

def check_data(mnist: Mnist) -> dict:
    """S-data: sha256 + class ratios within +-1% across all tasks."""
    ref = mnist.class_counts / mnist.class_counts.sum()
    worst, worst_t = 0.0, None
    for seed in range(10):
        g = stream("data", seed)
        for t in range(1, 201):
            idx = stratified_draw(mnist, g)
            y = mnist.train_y.cpu()[idx].numpy()
            frac = np.bincount(y, minlength=N_CLASSES) / len(y)
            dev = float(np.abs(frac - ref).max())
            if dev > worst:
                worst, worst_t = dev, (seed, t)
            if len(set(idx.tolist())) != TASK_EXAMPLES:
                return {"pass": False, "why": f"duplicate draw seed{seed} task{t}"}
    return {"pass": worst <= 0.01, "max_class_ratio_dev": worst,
            "argmax": worst_t, "sha256": mnist.sha256,
            "quota": mnist.quota.tolist()}


def check_perm() -> dict:
    """S-perm: identical across arms (bit), a valid permutation, not identity."""
    ref = None
    for seed in range(10):
        g = stream("perm", seed)
        seq = [torch.randperm(784, generator=g) for _ in range(200)]
        for t, p in enumerate(seq, 1):
            if torch.equal(p, torch.arange(784)):
                return {"pass": False, "why": f"identity perm seed{seed} task{t}"}
            if len(torch.unique(p)) != 784:
                return {"pass": False, "why": f"not a permutation seed{seed} task{t}"}
        if seed == 0:
            ref = seq
    # arm independence: the stream does not take the arm as input, so redrawing
    # for every arm must give the same bits.
    for arm in ARMS:
        g = stream("perm", 0)
        for t in range(200):
            if not torch.equal(torch.randperm(784, generator=g), ref[t]):
                return {"pass": False, "why": f"arm {arm} diverged at task {t+1}"}
    return {"pass": True, "n_seed": 10, "n_task": 200,
            "sample_hash": hashlib.sha256(ref[0].numpy().tobytes()).hexdigest()[:16]}


def check_init(device) -> dict:
    """S-init: same seed -> bit-identical initial weights for every arm."""
    for seed in range(10):
        ref = [p.detach().cpu() for p in init_params(seed, torch.device("cpu"))]
        for arm in ARMS:
            _ = ARMS[arm]                                  # arm must not consume rng
            got = [p.detach().cpu() for p in init_params(seed, torch.device("cpu"))]
            for a, b in zip(ref, got):
                if not torch.equal(a, b):
                    return {"pass": False, "why": f"seed{seed} arm{arm}"}
    h = hashlib.sha256(b"".join(p.detach().numpy().tobytes()
                                for p in init_params(0, torch.device("cpu")))).hexdigest()
    return {"pass": True, "seed0_param_sha256": h[:16]}


def check_act(device) -> dict:
    """S-act: analytic phi' vs autograd.

    Two stated criteria, because `np.allclose(atol=0)` alone is not attainable
    for Snake.  phi'(z) = 1 + sin(2*alpha*z) has exact zeros; at a grid point
    that lands 1e-6 away from one, |phi'| ~ 7e-12 while the two routes round
    differently in the last bit (~1e-16), so a pure relative test demands
    ~7e-17 and fails.  It is a property of the criterion, not of the code:
    autograd differentiates z + sin(az)^2/a as 1 + 2 sin(az) cos(az), the
    analytic form is 1 + sin(2az), and the double-angle identity is not exact
    in binary floating point.  So:
      rel : np.allclose(atol=0) where relative accuracy is meaningful,
            |phi'| > 1e-9
      abs : max|autograd - analytic| <= 4*eps over the WHOLE grid, which now
            explicitly contains every zero of phi' in range (the earlier
            version only hit them by grid luck -- alpha=1 did, alpha=3 did not)
    """
    eps = float(np.finfo(np.float64).eps)
    res, ok = {}, True
    for name, act in ARMS.items():
        grid = [torch.linspace(-12, 12, 200_001, dtype=torch.float64)]
        if act.kind == "snake":                      # 1+sin(2az)=0 <=> 2az = -pi/2 + 2k*pi
            a = act.param
            k = torch.arange(-200, 201, dtype=torch.float64)
            zeros = (-math.pi / 2 + 2 * math.pi * k) / (2 * a)
            grid.append(zeros[zeros.abs() <= 12])
        elif act.kind in ("relu", "leaky"):
            grid.append(torch.zeros(1, dtype=torch.float64))   # the kink
        z = torch.cat(grid).to(device)

        zz = z.clone().requires_grad_(True)
        (g,) = torch.autograd.grad(act.phi(zz).sum(), zz)
        ana = act.dphi(z)

        # relu/leaky are not differentiable at 0; exclude that single point.
        kink = (z == 0) & (act.kind in ("relu", "leaky"))
        dev = (g - ana).abs()
        rel_mask = (ana.abs() > 1e-9) & ~kink
        rel = np.allclose(g[rel_mask].cpu().numpy(), ana[rel_mask].cpu().numpy(), atol=0.0)
        abs_dev = float(dev[~kink].max())
        abs_ok = abs_dev <= 4 * eps

        zn = z.cpu().numpy()
        if act.kind == "relu":
            phi_ref = np.maximum(zn, 0.0)
        elif act.kind == "leaky":
            phi_ref = np.where(zn > 0, zn, act.param * zn)
        elif act.kind == "snake":
            phi_ref = zn + np.sin(act.param * zn) ** 2 / act.param
        else:
            phi_ref = zn
        phi_ok = bool(np.allclose(act.phi(z).cpu().numpy(), phi_ref, atol=0.0))

        res[name] = {"rel_allclose_atol0": bool(rel), "abs_within_4eps": bool(abs_ok),
                     "phi_vs_closed_form": phi_ok, "max_abs_dev": abs_dev,
                     "max_abs_dev_ulp_of_1": abs_dev / eps,
                     "n_grid": int(z.numel()),
                     "n_near_zero_dphi": int((ana.abs() <= 1e-9).sum())}
        ok &= rel and abs_ok and phi_ok
    return {"pass": bool(ok), "eps": eps, "per_arm": res}


def check_mob(device) -> dict:
    """S-mob: mobility is E[phi'(z)], not phi'(zbar). Show cases that differ by
    orders of magnitude -- including one where both are non-zero, so the gap is
    not an artefact of phi'(zbar) happening to be exactly 0."""
    out = {}

    def case(name, zbar, half):
        act = ARMS[name]
        z = zbar + torch.linspace(-half, half, 200_001, dtype=torch.float64, device=device)
        mean = float(act.dphi(z).mean())
        bar = float(act.dphi(torch.tensor(zbar, dtype=torch.float64, device=device)))
        return {"zbar": zbar, "spread": half, "E[phi'(z)]": mean, "phi'(zbar)": bar,
                "ratio": (mean / bar) if bar != 0 else float("inf")}

    # SN3 centred on a zero of phi' (zbar = -pi/12): phi'(zbar)=0 but E[phi']~1
    out["SN3_at_zero"] = case("SN3", -math.pi / 12, 1.0)
    # SN3 centred where phi'(zbar) is small but non-zero -> finite ratio, still 2 orders
    out["SN3_near_zero"] = case("SN3", -math.pi / 12 + 1e-3, 1.0)
    # ReLU deep in the negative half: phi'(zbar)=0 yet a wide unit still moves
    out["R_submerged"] = case("R", -2.0, 4.0)
    ok = any(c["ratio"] > 10 or c["ratio"] == float("inf") for c in out.values()) and \
        out["SN3_near_zero"]["ratio"] > 10 and out["SN3_near_zero"]["phi'(zbar)"] != 0
    return {"pass": bool(ok), "cases": out}


# --------------------------------------------------------------------------
# io
# --------------------------------------------------------------------------

def git_hash() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    cols, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k); cols.append(k)
    with path.open("w") as fh:
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join(
                "" if r.get(c) is None else
                (f"{r[c]:.10g}" if isinstance(r[c], float) else str(r[c]))
                for c in cols) + "\n")


def setup(device_str: str) -> torch.device:
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device_str)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["checks", "cost", "probe", "lrcal", "main"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--arms", default=None)
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--lrs", default=None)
    ap.add_argument("--tasks", type=int, default=None)
    ap.add_argument("--optimizer", default="sgd", choices=["sgd", "adam"])
    ap.add_argument("--epochs", type=int, default=1,
                    help="passes over each task's 10,000 examples (1 = spec §3)")
    ap.add_argument("--c", type=float, default=0.6, help="SNA: alpha_i = c / W_i")
    ap.add_argument("--beta", type=float, default=0.01, help="SNA: EMA rate of var(z_i); 0 = frozen")
    ap.add_argument("--relearn", action="store_true",
                    help="after the last task, re-fit tasks 1/100/180/199 from the end state and from a fresh init; writes relearn.csv")
    ap.add_argument("--bwt", action="store_true",
                    help="also evaluate on the permutations of tasks 1, t-1, t-5, t-20 (backward transfer)")
    ap.add_argument("--iv", default="none",
                    help="class-C intervention, e.g. l2init:1e-3 or cbp:1e-4 (mechanism from pmnist_lopcmp_0905)")
    args = ap.parse_args()

    device = setup(args.device)
    t_start = time.time()

    if args.stage == "checks":
        mnist = Mnist(device)
        res = {"S-data": check_data(mnist), "S-perm": check_perm(),
               "S-init": check_init(device), "S-act": check_act(device),
               "S-mob": check_mob(device)}
        res["all_pass"] = all(v["pass"] for k, v in res.items() if k.startswith("S-"))
        out = Path(args.out or REPO / "results" / "_checks_pmnist_0905")
        out.mkdir(parents=True, exist_ok=True)
        (out / "checks.json").write_text(json.dumps(res, indent=2, default=str))
        print(json.dumps(res, indent=2, default=str))
        return

    mnist = Mnist(device)
    defaults = {
        "cost":  (["R"], [0], [0.01], 5),
        "probe": (["R", "LR", "LIN"], list(range(10)), [0.01], 50),
        "lrcal": (list(ARMS), [0, 1, 2], [0.2, 0.1, 0.05, 0.02, 0.01, 0.005], 20),
        "main":  (list(ARMS), list(range(10)), [0.01], 200),
    }
    arms, seeds, lrs, tasks = defaults[args.stage]
    if args.arms:  arms = args.arms.split(",")
    if args.seeds: seeds = [int(s) for s in args.seeds.split(",")]
    if args.lrs:   lrs = [float(s) for s in args.lrs.split(",")]
    if args.tasks: tasks = args.tasks

    run_id = {"cost": "_cost_pmnist_0905", "probe": "pmnist_probe_0905",
              "lrcal": "pmnist_lrcal_0905", "main": "pmnist_main_0905"}[args.stage]
    out = Path(args.out or REPO / "results" / run_id)
    out.mkdir(parents=True, exist_ok=True)

    rows, divs, relearn_rows = [], [], []
    for lr in lrs:
        for arm in arms:
            for seed in seeds:
                t0 = time.time()
                r, d = run_one(arm, seed, lr, tasks, mnist, device,
                               optimizer=args.optimizer, epochs=args.epochs,
                               c=args.c, beta=args.beta, iv=args.iv, bwt=args.bwt,
                               relearn=args.relearn, relearn_rows=relearn_rows)
                rows += r
                if d["diverged"]:
                    divs.append(d)
                print(f"[{time.time()-t_start:7.1f}s] lr={lr:<6g} {arm:<4} seed={seed} "
                      f"tasks={len([x for x in r if x.get('acc')==x.get('acc')])} "
                      f"{(time.time()-t0):.1f}s"
                      f"{'  DIVERGED@step ' + str(d['step']) if d['diverged'] else ''}",
                      flush=True)
                write_csv(out / "per_task.csv", rows)
                if relearn_rows: write_csv(out / "relearn.csv", relearn_rows)

    prov = {"run_id": run_id, "stage": args.stage, "optimizer": args.optimizer,
            "git_hash": git_hash(),
            "data_sha256": mnist.sha256, "arms": arms, "seeds": seeds, "lrs": lrs,
            "n_tasks": tasks, "dims": list(DIMS), "batch": BATCH,
            "epochs_per_task": args.epochs, "sna_c": args.c, "sna_beta": args.beta,
            "intervention": args.iv, "bwt_logged": args.bwt, "relearn_probes": args.relearn,
            "cbp_adam_note": "Adam moments of replaced units are reset to 0 at replacement" if (args.iv.startswith("cbp") and args.optimizer=="adam") else None,
            "steps_per_task": STEPS_PER_TASK * args.epochs, "task_examples": TASK_EXAMPLES,
            "class_quota": mnist.quota.tolist(), "device": str(device),
            "torch": torch.__version__, "wall_clock_s": time.time() - t_start,
            "divergences": divs}
    (out / "provenance.json").write_text(json.dumps(prov, indent=2))
    print(f"\nwrote {out}/per_task.csv  ({len(rows)} rows, {time.time()-t_start:.1f}s)")


if __name__ == "__main__":
    main()
