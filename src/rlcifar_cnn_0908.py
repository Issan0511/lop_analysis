#!/usr/bin/env python3
"""Random Label CIFAR on Kumar et al.'s CNN (spec RandomLabelCIFAR_CNN_spec_0908).

    python3 src/rlcifar_cnn_0908.py --arms SNA --seeds 0 --tasks 50

The MLP version of this box (`pmnist_rlcifar_0907`) was stopped at stage 0: a
3072-wide first layer killed ReLU at all four lambda, and Kumar A.1.3 in fact
uses "a CNN on Random Label CIFAR".  This module keeps that box's protocol --
1200 CIFAR-10 images drawn once per seed, relabelled uniformly every task,
fitted to memorisation for 400 epochs x 75 steps of batch 16 -- and replaces the
net with their CNN:

    conv(5x5, 3->16, pad 2) -> phi -> maxpool(2) -> conv(5x5, 16->16, pad 2)
      -> phi -> maxpool(2) -> flatten(1024) -> fc(100) -> phi -> fc(100) -> phi -> fc(10)

Four activation sites, tagged `c1 c2 f1 f2`.  The order is conv -> phi -> pool
(registered in provenance); Snake is monotone (phi' = 1 + sin >= 0) so pool
commutes with phi and the order is a bookkeeping choice, checked by S-pool-commute.

The new piece is `ChannelSnake`: the adaptive alpha of the host's `AdaptiveSnake`
extended to conv, where one alpha is shared by every spatial position of a
channel -- the granularity BatchNorm2d uses, and the only one that keeps the
layer a convolution (S-equivariance).

Data loading, subsetting, labels, the intervention mechanism and the divergence
policy are imported from `pmnist_rlcifar_0907` / `pmnist_rlmnist_0906` / the host
`pmnist_0905`; none of those three files is modified, since other runs' bit
reproducibility depends on them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

# Determinism must be requested before the first CUDA workspace is allocated.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import pmnist_0905 as H              # host; not touched
from src import pmnist_rlcifar_0907 as RC     # CIFAR data layer; not touched

EXPERIMENT = "rlcifar_cnn_0908"
TRAIN_N = RC.TRAIN_N                           # 50,000
N_IMAGES = RC.N_IMAGES                         # 1200 images per seed, shared by every task
BATCH = H.BATCH                                # 16
STEPS_PER_EPOCH = RC.STEPS_PER_EPOCH           # 75
N_CLASSES = H.N_CLASSES                        # 10
DEFAULT_ARMS = "SNA,LR,R"                      # the +l2/+l2init runs pass --iv

# net (spec §2.1). conv fan_in = k^2 * C_in, fc fan_in = in_features.
IMG = (3, 32, 32)
KERNEL, PAD, POOL = 5, 2, 2
CHANNELS = 16
FLAT = CHANNELS * 8 * 8                        # 32 -> pool 16 -> pool 8
HIDDEN = 100
SHAPES = (((CHANNELS, IMG[0], KERNEL, KERNEL), (CHANNELS,)),
          ((CHANNELS, CHANNELS, KERNEL, KERNEL), (CHANNELS,)),
          ((HIDDEN, FLAT), (HIDDEN,)),
          ((HIDDEN, HIDDEN), (HIDDEN,)),
          ((N_CLASSES, HIDDEN), (N_CLASSES,)))
FAN_IN = (KERNEL * KERNEL * IMG[0], KERNEL * KERNEL * CHANNELS, FLAT, HIDDEN, HIDDEN)

# activation sites, in forward order
SITES = ("c1", "c2", "f1", "f2")
IS_CONV = (True, True, False, False)
WIDTHS = (CHANNELS, CHANNELS, HIDDEN, HIDDEN)
WEIGHT_TAGS = ("c1", "c2", "f1", "f2", "f3")   # five weight tensors, four of them gated

# Per-task preactivation histogram, bin width 0.1 (registered).
#
# The range is from measurement, not a guess -- on the MLP box a guessed range came out
# 6x too narrow.  Seed-0 probes at the real 400 epochs (5 arms x 3 tasks; SNA/R/LR at
# --iv none x 10 tasks; R/LR x 10 tasks recording quantiles rather than extremes) say:
#   * the arms that stay alive are small and saturate by task ~5.  Over all four sites
#     at task 10, SNA+none spans [-139.8, 78.5] and LR+none [-169.2, 111.4]; l2 at
#     1e-3 holds everything inside +-10 (and kills R and LR outright).
#   * R+none collapses (memo 1.00 -> 0.15 over 10 tasks) and its conv2 preactivation
#     runs away negative without bound -- and not merely in the tail: at task 10 the
#     MEDIAN of z_c2 is -201 and its 0.1% quantile -854, drifting about -30 and -120
#     per task.  A symmetric range would put most of that arm's mass out of bounds.
# Hence the lopsided limits: -8192 holds R's conv2 bulk past task 40 on that trend, and
# +2048 is 9x the largest positive value seen anywhere (+227.8).  Empty bins are almost
# free after npz compression (3.4 MB per (arm, seed) against 3.0 MB for a range four
# times narrower), so the headroom costs nothing worth saving.  `oob` counts everything
# outside, so a range that is nonetheless too narrow reports itself instead of clipping
# silently -- and the drift itself is recorded exactly regardless, by `m_*`, `zbar_min`
# and the raw samples, none of which are bounded by this range.
HIST_LO, HIST_HI, HIST_NB = -8192.0, 2048.0, 102400
HIST_EDGES = np.linspace(HIST_LO, HIST_HI, HIST_NB + 1)

# Raw preactivation samples on a fixed coordinate grid, so the same units and the
# same image/position pairs are comparable across tasks, arms and seeds.  The
# histogram is the primary record; this is the joint structure it throws away
# (per-channel shape, unit-to-unit correlation).  9 x 512 x 232 x 2 B = 2.1 MB per
# (arm, seed), ~150 MB over the 70-run grid.
SAMPLE_N = 512
SAMPLE_CKPT = (1, 2, 3, 5, 10, 20, 30, 40, 50)


# --------------------------------------------------------------------------
# channel-wise adaptive Snake (spec §2.2)
# --------------------------------------------------------------------------

class ChannelSnake:
    """Adaptive Snake whose alpha is per output channel (conv) or per unit (fc).

    alpha_j = clip(c / W_j, lo, hi), W_j = sqrt(EMA_beta[var(z_j)]), the variance
    taken over (batch, H, W) for conv and over the batch for fc.  One alpha per
    channel, SHARED by every spatial position: a per-position alpha would stop the
    layer being a convolution (S-equivariance).  This is the granularity
    BatchNorm2d keeps its running statistics at.

    V holds no grad, so alpha is detached -- the same standing as BatchNorm's
    running statistics.  beta=0 freezes V at its init of 1, which makes this
    bit-identical to the host's fixed-alpha Snake at alpha=c (S-ema-off).
    """
    name, kind = "SNA", "adaptive_snake"

    def __init__(self, c: float, beta: float, device, lo: float = 0.005, hi: float = 3.0,
                 widths=WIDTHS, is_conv=IS_CONV):
        self.c, self.beta, self.lo, self.hi = c, beta, lo, hi
        self.is_conv = tuple(is_conv)
        self.V = [torch.ones(w, device=device) for w in widths]

    def alpha(self, layer: int) -> torch.Tensor:
        return (self.c / self.V[layer].sqrt()).clamp(self.lo, self.hi)

    def _bcast(self, layer: int, a: torch.Tensor) -> torch.Tensor:
        """alpha shaped to broadcast over (N,C,H,W) or (N,C)."""
        return a[None, :, None, None] if self.is_conv[layer] else a[None, :]

    def phi(self, z, layer=0):
        a = self._bcast(layer, self.alpha(layer))
        # `* a.reciprocal()` rather than `/ a`: the fixed-alpha Snake divides by a
        # Python float, which the CUDA kernel folds into a reciprocal-multiply.
        # Matching that op order makes ChannelSnake(beta=0) bit-identical to fixed
        # alpha=c; a true tensor division differs by 1 ulp and the trajectory drifts.
        return z + torch.sin(a * z) ** 2 * a.reciprocal()

    def dphi(self, z, layer=0):
        return 1.0 + torch.sin(2.0 * self._bcast(layer, self.alpha(layer)) * z)

    @torch.no_grad()
    def update(self, zs) -> None:
        """One EMA step of the per-channel / per-unit preactivation variance."""
        if self.beta == 0:
            return
        for l, z in enumerate(zs):
            v = z.var((0, 2, 3), unbiased=False) if self.is_conv[l] else z.var(0, unbiased=False)
            self.V[l].mul_(1 - self.beta).add_(self.beta * v)

    @torch.no_grad()
    def stats(self) -> dict:
        out = {}
        for l, tag in enumerate(SITES):
            a = self.alpha(l); W = self.V[l].sqrt()
            raw = self.c / W
            out[f"alpha_med_{tag}"] = float(a.median()); out[f"alpha_min_{tag}"] = float(a.min())
            out[f"alpha_max_{tag}"] = float(a.max())
            out[f"alpha_clip_frac_{tag}"] = float(((raw < self.lo) | (raw > self.hi)).float().mean())
            out[f"two_alpha_W_med_{tag}"] = float((2 * a * W).median())
        return out


# --------------------------------------------------------------------------
# net
# --------------------------------------------------------------------------

def init_params(seed: int, device: torch.device):
    """Host init rule, U(-1/sqrt(fan_in), +1/sqrt(fan_in)), one bound per tensor.

    Weight then bias, layers in forward order, all from the host's `init` stream on
    cpu: activation-independent on purpose (S-init), and device-independent so the
    bit-comparison in S-init means something.
    """
    g = H.stream("init", seed)
    params = []
    for (ws, bs), fan in zip(SHAPES, FAN_IN):
        bound = 1.0 / math.sqrt(fan)
        W = (torch.rand(ws, generator=g, dtype=torch.float32) * 2 - 1) * bound
        b = (torch.rand(bs, generator=g, dtype=torch.float32) * 2 - 1) * bound
        params += [W.to(device).requires_grad_(True), b.to(device).requires_grad_(True)]
    return params


def forward_cnn(params, x, act):
    """(z,a) of the four activation sites plus the logits.  conv -> phi -> pool."""
    Wc1, bc1, Wc2, bc2, Wf1, bf1, Wf2, bf2, Wf3, bf3 = params
    ada = isinstance(act, ChannelSnake)
    z1 = F.conv2d(x, Wc1, bc1, padding=PAD)
    a1 = act.phi(z1, 0) if ada else act.phi(z1)
    z2 = F.conv2d(F.max_pool2d(a1, POOL, POOL), Wc2, bc2, padding=PAD)
    a2 = act.phi(z2, 1) if ada else act.phi(z2)
    h = F.max_pool2d(a2, POOL, POOL).flatten(1)
    z3 = h @ Wf1.T + bf1
    a3 = act.phi(z3, 2) if ada else act.phi(z3)
    z4 = a3 @ Wf2.T + bf2
    a4 = act.phi(z4, 3) if ada else act.phi(z4)
    return z1, a1, z2, a2, z3, a3, z4, a4, a4 @ Wf3.T + bf3


# --------------------------------------------------------------------------
# metrics (spec §2.4)
# --------------------------------------------------------------------------

def eff_rank_ch(a: torch.Tensor) -> float:
    """Host `eff_rank` on the CHANNEL feature matrix: (N,C,H,W) -> (N*H*W, C).

    A conv channel is one feature seen at N*H*W places, so the matrix whose rank
    is at issue has C columns (<= 16 here), not C*H*W.  The formula, the float64
    Gram and the cpu eigendecomposition are the host's, unchanged.
    """
    if a.dim() == 4:
        a = a.permute(0, 2, 3, 1).reshape(-1, a.shape[1])
    return H.eff_rank(a)


@torch.no_grad()
def evaluate_cnn(params, x: torch.Tensor, y: torch.Tensor, act) -> dict:
    """The host's `evaluate`, per activation site, on the 1200 images of this task."""
    z1, a1, z2, a2, z3, a3, z4, a4, logits = forward_cnn(params, x, act)
    out = {"acc": float((logits.argmax(1) == y).float().mean())}
    ada = isinstance(act, ChannelSnake)
    for li, (tag, z, a) in enumerate((("c1", z1, a1), ("c2", z2, a2),
                                      ("f1", z3, a3), ("f2", z4, a4))):
        d = act.dphi(z, li) if ada else act.dphi(z)
        conv = IS_CONV[li]
        dims = (0, 2, 3) if conv else (0,)
        out[f"dead_frac_{tag}"] = float((d.abs().amax(dims) < H.DEAD_TOL).float().mean())
        out[f"zeroout_{tag}"] = float((a.abs().amax(dims) == 0).float().mean())
        out[f"zbar_{tag}"] = float(z.mean(dims).median())
        # spread of the preactivation over everything the channel sees.  For Snake the
        # relevant scale is alpha*W: phi' = 1+sin(2*alpha*z) is averaged over that range,
        # so once alpha*W spans a period the mean pins to 1 and no gating survives.
        out[f"zsd_{tag}"] = float(z.std(dims).median())
        out[f"zbar_min_{tag}"] = float(z.mean(dims).amin())
        out[f"mob_{tag}"] = float(d.mean(dims).median())
        if conv:
            # spec §1.1-1: the pool passes one position per window, so the gradient
            # that reaches the weights only ever sees phi' AT THE ARGMAX.  `mob` over
            # all positions can therefore overstate the gate that is actually in play.
            _, idx = F.max_pool2d(a, POOL, POOL, return_indices=True)
            dp = d.flatten(2).gather(2, idx.flatten(2))
            out[f"mob_pool_{tag}"] = float(dp.mean((0, 2)).median())
        out[f"eff_rank_{tag}"] = eff_rank_ch(a)
    for i, tag in enumerate(WEIGHT_TAGS):
        W = params[2 * i]
        out[f"w_norm_{tag}"] = float(W.reshape(W.shape[0], -1).norm(dim=1).median())
    if ada:
        out.update(act.stats())
    return out


# --------------------------------------------------------------------------
# per-task preactivation distribution and fixed-grid samples
# --------------------------------------------------------------------------

def sample_grid(seed: int) -> dict:
    """The fixed (image, position) coordinates the raw samples are read at.

    Own stream `rlcc_probe`, drawn once per seed, so it disturbs no training stream
    and the same coordinates are compared across tasks, arms and seeds.  One position
    per sampled image; the conv2 position is the conv1 position halved, so the two
    conv sites are read at the same place in the image.
    """
    g = H.stream("rlcc_probe", seed)
    img = torch.randint(N_IMAGES, (SAMPLE_N,), generator=g, dtype=torch.int64)
    pos = torch.randint(IMG[1], (SAMPLE_N, 2), generator=g, dtype=torch.int64)   # 32 x 32
    return {"img_idx": img, "pos_idx": pos, "pos_idx2": pos // POOL}             # 16 x 16


@torch.no_grad()
def preact_hist(params, x: torch.Tensor, act, grid: dict | None = None) -> dict:
    """Histogram + per-channel mean of every site's preactivation, and (optionally)
    the raw values at the fixed grid.

    `evaluate_cnn` is a separate pass on purpose: against 30,000 training steps per
    task a second forward over 1200 images is under 0.1% of the task's work, and
    keeping the two apart means neither has to carry the other's tensors alive.
    """
    zs = forward_cnn(params, x, act)[0:8:2]
    out = {}
    for li, (tag, z) in enumerate(zip(SITES, zs)):
        v = z.reshape(-1).cpu().numpy()
        # bins as (count, range) rather than an edge array: numpy's uniform-bin fast
        # path, ~10x quicker on the 19.7M values of c1, and identical binning.  The
        # last bin is closed, so `oob` below is the exact complement of h.sum().
        out[f"h_{tag}"] = np.histogram(v, bins=HIST_NB, range=(HIST_LO, HIST_HI))[0].astype(np.int32)
        out[f"m_{tag}"] = z.mean((0, 2, 3) if IS_CONV[li] else 0).cpu().numpy().astype(np.float32)
        out[f"oob_{tag}"] = np.int64(((v < HIST_LO) | (v > HIST_HI)).sum())
        if grid is not None:
            im = grid["img_idx"].to(z.device)
            if IS_CONV[li]:
                p = (grid["pos_idx"] if li == 0 else grid["pos_idx2"]).to(z.device)
                # advanced indices split by the channel slice, so the sampled axis
                # comes first: (SAMPLE_N, C), one (image, position) pair per row.
                s = z[im, :, p[:, 0], p[:, 1]]
            else:
                s = z[im, :]                                       # (SAMPLE_N, HIDDEN)
            out[f"s_{tag}"] = s.cpu().numpy().astype(np.float16)
    return out


def write_hist(path: Path, hists: list[dict], accs: list[float],
               grid: dict, ckpts: list[int]) -> None:
    """One self-contained npz per (arm, seed); row index = task - 1."""
    if len(hists) != len(accs):                # a diverged task contributes neither
        raise AssertionError(f"{len(hists)} histograms vs {len(accs)} accuracies")
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = [f"{p}_{t}" for p in ("h", "m", "oob") for t in SITES]
    smp = {f"s_{t}": np.stack([h[f"s_{t}"] for h in hists if f"s_{t}" in h]) for t in SITES}
    np.savez_compressed(
        path, edges=HIST_EDGES, acc=np.asarray(accs, dtype=np.float64),
        ckpt_tasks=np.asarray(ckpts, dtype=np.int64),
        img_idx=grid["img_idx"].numpy(), pos_idx=grid["pos_idx"].numpy(),
        pos_idx2=grid["pos_idx2"].numpy(),
        **{k: np.stack([h[k] for h in hists]) for k in keys}, **smp)


# --------------------------------------------------------------------------
# one (arm, seed, lr) run -- pmnist_rlcifar_0907.run_one with the CNN
# --------------------------------------------------------------------------

def run_one(arm: str, seed: int, lr: float, n_tasks: int, cifar: RC.Cifar10,
            device: torch.device, optimizer: str = "adam", epochs: int = 400,
            c: float = 0.6, alpha_lo: float = 0.005, alpha_hi: float = 3.0,
            beta: float = 0.01, iv: str = "none", debug: dict | None = None,
            hist_dir: Path | None = None,
            zlog: list | None = None) -> tuple[list[dict], dict]:
    act = H.ARMS[arm]
    if act.kind == "adaptive_snake":
        act = ChannelSnake(c, beta, device, lo=alpha_lo, hi=alpha_hi)   # fresh stats per run
    params = init_params(seed, device)
    if debug is not None:                             # S-init / S-online hook, unused in real runs
        debug["init"] = [q.detach().cpu().clone() for q in params]
    ivo = RC.Iv.parse(iv)
    p0 = [q.detach().clone() for q in params] if ivo.kind == "l2init" else None
    # Adam moments live for the whole run: resetting them between tasks would break
    # the continual definition the same way resetting weights does.
    adam = ([torch.zeros_like(q) for q in params],
            [torch.zeros_like(q) for q in params], [0]) if optimizer == "adam" else None

    idx = subset_idx(seed)
    x = images(cifar, idx, device)                    # the task's inputs, fixed forever
    grid = sample_grid(seed)
    g_lab, g_batch = H.stream("rlcc_labels", seed), H.stream("rlcc_batch", seed)
    spt = STEPS_PER_EPOCH * epochs
    rows, hists, ckpts = [], [], []
    diverged = {"diverged": False, "task": None, "step": None, "seed": seed, "arm": arm}

    for t in range(1, n_tasks + 1):
        y = task_labels(g_lab).to(device)             # new labelling, same images
        if debug is not None:
            debug.setdefault("subset", []).append(idx.clone())
            debug.setdefault("labels", []).append(y.cpu().clone())
        acc_sum = torch.zeros((), device=device)
        bad_step = torch.full((), -1, dtype=torch.long, device=device)

        for e in range(epochs):
            order = torch.randperm(N_IMAGES, generator=g_batch).to(device)
            xs, ys = x[order], y[order]               # reshuffled every epoch
            for j in range(STEPS_PER_EPOCH):
                s = e * STEPS_PER_EPOCH + j
                xb, yb = xs[j * BATCH:(j + 1) * BATCH], ys[j * BATCH:(j + 1) * BATCH]
                out = forward_cnn(params, xb, act)
                loss = torch.nn.functional.cross_entropy(out[8], yb)
                # pre-update accuracy: the argmax of the very forward pass the loss
                # came from, i.e. before this batch has been learned from.
                hit = (out[8].detach().argmax(1) == yb).float().mean()
                acc_sum += hit
                grads = torch.autograd.grad(loss, params)
                with torch.no_grad():
                    bad = ~torch.isfinite(loss)
                    bad_step = torch.where((bad_step < 0) & bad,
                                           torch.tensor(s, device=device), bad_step)
                    if ivo.kind in ("l2", "l2init"):
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
                    if isinstance(act, ChannelSnake):
                        act.update(out[0:8:2])        # running var of this batch's preacts
                if debug is not None:
                    debug.setdefault("online", []).append(float(hit))

        bs = int(bad_step)
        if bs >= 0 or not all(torch.isfinite(p).all() for p in params):
            diverged.update(diverged=True, task=t, step=(t - 1) * spt + max(bs, 0))
            rows.append({"arm": arm, "seed": seed, "lr": lr, "task": t, "iv": iv,
                         "acc": float("nan")})
            break                                     # drop, never rescue
        m = evaluate_cnn(params, x, y, act)
        if hist_dir is not None:
            # raw samples only at the checkpoints, and at the last completed task of a
            # run that is about to be truncated -- decided after the loop, so the final
            # task always carries one.
            hists.append(preact_hist(params, x, act, grid))
            ckpts.append(t)
        if zlog is not None:
            # the envelope the histogram range has to contain; separate from `debug`
            # because that one syncs every step and cannot be used at 400 epochs
            with torch.no_grad():
                zs = forward_cnn(params, x, act)[0:8:2]
                zlog.append({"task": t, **{tag: (float(z.min()), float(z.max()))
                                           for tag, z in zip(SITES, zs)}})
        rows.append({"arm": arm, "seed": seed, "lr": lr, "task": t, "iv": iv,
                     "online_acc": float(acc_sum) / spt, "memo_acc": m["acc"], **m})
    # written per (arm, seed), truncated to the tasks that completed: a diverged run
    # keeps the distributions it did produce.
    if debug is not None:                             # S-hist coordinate check hook
        debug["params_end"] = [q.detach().clone() for q in params]
        debug["grid"] = grid
    if hist_dir is not None and hists:
        keep = [i for i, t in enumerate(ckpts) if t in SAMPLE_CKPT or t == ckpts[-1]]
        for i, h in enumerate(hists):
            if i not in keep:
                for tag in SITES:
                    h.pop(f"s_{tag}", None)
        write_hist(Path(hist_dir) / f"{arm}_seed{seed}.npz", hists,
                   [r["memo_acc"] for r in rows if "memo_acc" in r],
                   grid, [ckpts[i] for i in keep])
    return rows, diverged


# imported so the CNN box and the MLP box share one definition of each, not two
# copies that can drift apart
task_labels = RC.task_labels                   # iid uniform {0..9}, advances g once
out_tag = RC.out_tag


def subset_idx(seed: int) -> torch.Tensor:
    """The seed's 1200 training images: uniform, no replacement, no stratification.

    Own stream (`rlcc_subset`, distinct from the MLP box's `rlc_subset`), drawn once
    per seed, so the image set is a property of the seed and is bit-identical across
    arms, tasks and interventions.
    """
    g = H.stream("rlcc_subset", seed)
    return torch.randperm(TRAIN_N, generator=g)[:N_IMAGES]


def images(cifar: RC.Cifar10, idx: torch.Tensor, device: torch.device) -> torch.Tensor:
    """`idx` rows as (N,3,32,32) float32 in [0,1].

    `data_batch_*` stores each image as 1024 R then 1024 G then 1024 B, each plane
    row-major, so a plain reshape to (3,32,32) is already (C,H,W) -- no transpose.
    /255 only; spec §2.1 puts channel normalisation out of scope.
    """
    return cifar.images(idx, device).reshape(-1, *IMG)


# --------------------------------------------------------------------------
# io
# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=DEFAULT_ARMS)
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--lrs", default="0.001")
    ap.add_argument("--tasks", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=400,
                    help="passes over the 1200 examples per task (75 steps each)")
    ap.add_argument("--alpha-lo", type=float, default=0.005, help="SNA: lower clip on alpha_j")
    ap.add_argument("--alpha-hi", type=float, default=3.0, help="SNA: upper clip on alpha_j")
    ap.add_argument("--c", type=float, default=0.6, help="SNA: alpha_j = c / W_j")
    ap.add_argument("--beta", type=float, default=0.01, help="SNA: EMA rate of var(z_j)")
    ap.add_argument("--iv", default="none", help="none | l2:<lam> | l2init:<lam>")
    ap.add_argument("--optimizer", default="adam", choices=["sgd", "adam"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    arms = args.arms.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    lrs = [float(s) for s in args.lrs.split(",")]
    RC.Iv.parse(args.iv)                               # fail fast on a bad spec
    for a in arms:
        if a not in H.ARMS:
            raise SystemExit(f"unknown arm {a!r}; known: {','.join(H.ARMS)}")
        if a == "LIN0":
            raise SystemExit("LIN0 is the host's depth-0 control and has no CNN form")

    device = H.setup(args.device)
    t_start = time.time()
    cifar = RC.Cifar10()

    out = Path(args.out or H.REPO / "results" / EXPERIMENT / out_tag(arms, args.iv))
    out.mkdir(parents=True, exist_ok=True)

    rows, divs = [], []
    for lr in lrs:
        for arm in arms:
            for seed in seeds:
                t0 = time.time()
                r, d = run_one(arm, seed, lr, args.tasks, cifar, device,
                               optimizer=args.optimizer, epochs=args.epochs,
                               c=args.c, alpha_lo=args.alpha_lo, alpha_hi=args.alpha_hi,
                               beta=args.beta, iv=args.iv, hist_dir=out / "hist")
                rows += r
                if d["diverged"]:
                    divs.append(d)
                print(f"[{time.time()-t_start:7.1f}s] lr={lr:<6g} {arm:<4} seed={seed} "
                      f"tasks={len([q for q in r if q.get('acc') == q.get('acc')])} "
                      f"{(time.time()-t0):.1f}s"
                      f"{'  DIVERGED@step ' + str(d['step']) if d['diverged'] else ''}",
                      flush=True)
                H.write_csv(out / "per_task.csv", rows)

    prov = {"run_id": EXPERIMENT, "git_hash": H.git_hash(),
            "arms": arms, "seeds": seeds, "lrs": lrs, "n_tasks": args.tasks,
            "epochs_per_task": args.epochs, "batch": BATCH,
            "steps_per_task": STEPS_PER_EPOCH * args.epochs,
            "n_images": N_IMAGES, "train_n": TRAIN_N, "n_classes": N_CLASSES,
            "net": {"conv": [[list(SHAPES[0][0]), FAN_IN[0]], [list(SHAPES[1][0]), FAN_IN[1]]],
                    "fc": [[list(s[0]), f] for s, f in zip(SHAPES[2:], FAN_IN[2:])],
                    "padding": PAD, "pool": POOL, "flat": FLAT,
                    "act_order": "conv -> phi -> pool", "sites": list(SITES)},
            "sna_c": args.c, "sna_beta": args.beta, "alpha_lo": args.alpha_lo,
            "alpha_hi": args.alpha_hi, "alpha_granularity": "per output channel (conv) / per unit (fc)",
            "intervention": args.iv,
            "optimizer": args.optimizer, "data_sha256": cifar.sha256,
            "subset_sha256": {str(s): hashlib.sha256(
                np.sort(subset_idx(s).numpy()).tobytes()).hexdigest() for s in seeds},
            "rng_roles": ["rlcc_subset", "rlcc_labels", "rlcc_batch", "rlcc_probe", "init"],
            "hist": {"lo": HIST_LO, "hi": HIST_HI, "bins": HIST_NB, "width": (HIST_HI - HIST_LO) / HIST_NB,
                     "dtype": "int32", "path": "hist/<arm>_seed<seed>.npz",
                     "range_from": "seed-0 probes at 400 epochs (5 arms x 3 tasks; "
                                   "SNA/R/LR none x 10 tasks; R/LR quantiles x 10 tasks). "
                                   "Lopsided because R+none's z_c2 median reaches -201 by "
                                   "task 10 (0.1% quantile -854) while the largest positive "
                                   "value anywhere was +227.8"},
            "samples": {"n": SAMPLE_N, "ckpt_tasks": list(SAMPLE_CKPT), "dtype": "float16",
                        "stream": "rlcc_probe",
                        "note": "one position per sampled image; conv2 position = conv1 position // 2"},
            "device": str(device), "torch": torch.__version__,
            # peak of ONE process, so the 7-way launch can be budgeted (the eval pass
            # over all 1200 images at c1 is the high-water mark, not the training step)
            "cuda_max_mem_mb": (torch.cuda.max_memory_allocated() / 2 ** 20
                                if device.type == "cuda" else None),
            "wall_clock_s": time.time() - t_start, "divergences": divs}
    (out / "provenance.json").write_text(json.dumps(prov, indent=2))
    print(f"\nwrote {out}/per_task.csv  ({len(rows)} rows, {time.time()-t_start:.1f}s)")


if __name__ == "__main__":
    main()
