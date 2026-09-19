#!/usr/bin/env python3
"""rlcifar_mlp_battle_0918 -- RL-CIFAR x MLP, R runs stacked in one process (spec_rlcifar_mlp_battle_0918.md).

    python3 src/rlcifar_mlp_battle_0918.py run --arm KKA                       # raw+std x seeds 0-9 = 20 runs
    python3 src/rlcifar_mlp_battle_0918.py run --arm SNA --seeds 100 --conds raw --tasks 3 --out DIR

The box is pmnist_rlcifar_0907's (3072-100-100-10, 1200 CIFAR images per seed, random labels
per task, 400 epochs x 75 steps of batch 16, Adam 1e-3, no weight decay) and every stream --
image subset, labels, batch order, init -- is the host's, drawn from the host's generators in
the host's order, so slot (seed s, cond) sees exactly what host run seed s sees.  What differs
is only the shape of the computation: the R runs' parameters are stacked on a leading axis and
the three affine maps are `baddbmm` instead of `@` (spec §2.2).  Forward values are bit-identical
to the host's; gradients differ by the summation order inside bmm's backward (~1e-7 relative,
check S-stack), so trajectories are not bit-comparable with the host over 30,000 steps.

Nothing in pmnist_0905 / pmnist_rlmnist_0906 / pmnist_rlcifar_0907 is modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import pmnist_0905 as H              # host: streams, init, metrics helpers, csv
from src import pmnist_rlcifar_0907 as RC     # host: CIFAR data layer, task labels, hist range

EXPERIMENT = "relu_doors_0919"
OUT_ROOT = H.REPO / "results" / EXPERIMENT
DIMS = RC.DIMS                                 # (3072, 100, 100, 10)
N_IMAGES, BATCH, STEPS_PER_EPOCH = RC.N_IMAGES, RC.BATCH, RC.STEPS_PER_EPOCH
N_CLASSES = RC.N_CLASSES
DEAD_TOL = H.DEAD_TOL
LR = 1e-3
CONDS = ("raw", "std")
ARM_ORDER = ("C", "CH", "CHB", "CHB0", "CS", "LN")              # spec §8 / 追補 1
# spec §2.1/§3: which doors each arm closes.  C = centre the input, H = centre the hidden
# activations with an EMA, B = the bias policy ("none" | "wd" on b2 | "zero" on b1 and b2).
# Every arm removes b1 when B is closed, so with C the identity zbar_1i = w_i.xbar + b_1i = 0
# holds exactly for every unit and every task.
DOORS = {"ref":  dict(c=False, h=False, b="none"),
         "C":    dict(c=True,  h=False, b="none"),
         "CH":   dict(c=True,  h=True,  b="none"),
         "CHB":  dict(c=True,  h=True,  b="wd"),
         "CHB0": dict(c=True,  h=True,  b="zero"),
         # 追補 1 (spec §10): the two controls the literature check called for.
         # CS is the exact complement of H -- the SCALING half of the same per-feature running
         # statistic, no centring, no affine (RMSNorm-like).  LN is published LayerNorm:
         # per-sample across features, with learnable gamma/beta, after the linear map and
         # before the nonlinearity (the placement Lewandowski et al. state).
         "CS":   dict(c=True,  h=False, b="none", norm="scale"),
         "LN":   dict(c=True,  h=False, b="none", norm="layernorm")}

# spec §2.3: Lillo & Cheney's CIFAR-10 Normalize, per channel plane (R, G, B: 1024 each)
STD_MEAN = (0.4914, 0.4822, 0.4465)
STD_STD = (0.2470, 0.2435, 0.2616)

# kept from the parent engine so the adaptive branches stay byte-identical to
# rlcifar_mlp_battle_0918 (they never execute here: ReLUDoors.adaptive is False)
TH_LO, TH_HI, TH_NB = -6.0 * math.pi, 6.0 * math.pi, 1885
TH_EDGES = np.linspace(TH_LO, TH_HI, TH_NB + 1)
PI = math.pi


# --------------------------------------------------------------------------
# activations (spec §3).  z has shape (R, B, n); adaptive arms carry per-run,
# per-unit alpha of shape (R, n).  phi'/dphi is the eval-mode gate the metrics read.
# --------------------------------------------------------------------------

class Act:
    name = "?"
    adaptive = False          # Snake family: alpha_i = clip(c / W_i)
    stochastic = False        # RSL: a fresh r per element per training step

    def phi(self, z, layer=0, train=False):
        raise NotImplementedError

    def dphi(self, z, layer=0):
        raise NotImplementedError

    def init_state(self, R: int, device, key: str) -> None:
        pass

    def begin_step(self, R: int, B: int, device) -> None:
        """Called once before every training step (outside a captured CUDA graph)."""
        pass

    def state(self) -> dict:
        return {}

    def load_state(self, st: dict) -> None:
        pass

    def update(self, z1, z2) -> None:
        pass

    def post_update(self, P, lr: float) -> None:
        """Called inside the training step, after Adam has written the parameters."""
        pass

    def stats(self, r: int) -> dict:
        return {}


class ReLU(Act):
    name = "R"

    def phi(self, z, layer=0, train=False):
        return torch.clamp(z, min=0.0)

    def dphi(self, z, layer=0):
        return (z > 0).to(z.dtype)


class ReLUDoors(ReLU):
    """ReLU with the doors of spec §2.1.

    H: the layer's output is phi(z) - m_l, where m_l is a per-unit EMA of phi(z)'s batch mean,
       m_l <- (1-beta) m_l + beta * mean_batch(phi(z)).  `m` is a constant (no gradient flows
       through it) and the SAME value is used in training and in evaluation, so there is no
       train/eval gap.  The EMA tracks the UNCENTRED activation; tracking the centred one would
       decay to zero and do nothing.
    B: "wd"   -> b1 is held at exactly 0 and b2 is decayed by (1 - lr*lam) after every Adam step
       "zero" -> b1 and b2 are both held at exactly 0
       b3 is never touched.
    C lives in `slot_inputs`, not here.
    """
    adaptive = False

    def __init__(self, arm: str, lam: float = 0.0, beta: float = 0.01,
                 widths=(DIMS[1], DIMS[2])):
        d = DOORS[arm]
        self.name, self.arm = arm, arm
        self.door_c, self.door_h, self.door_b = d["c"], d["h"], d["b"]
        self.norm = d.get("norm", "none")          # "none" | "scale" | "layernorm"
        self.gb = None                             # LayerNorm's (gamma, beta) per layer
        self.lam, self.beta, self.widths = lam, beta, widths
        if self.door_b == "wd" and not lam > 0:
            raise SystemExit("arm CHB needs --lam > 0 (spec §2.2 derives it from a probe)")
        self.m = None

    def init_state(self, R, device, key):
        self.m = [torch.zeros(R, w, device=device) for w in self.widths] if self.door_h else None
        # the scaling statistic starts at 1 (EMA of the mean square of phi(z))
        self.v = [torch.ones(R, w, device=device) for w in self.widths] \
            if self.norm == "scale" else None

    def extra_params(self, R, device):
        """LayerNorm's gamma/beta, appended to P so the run's own Adam optimises them."""
        if self.norm != "layernorm":
            return []
        out = []
        for w in self.widths:
            out += [torch.ones(R, w, device=device), torch.zeros(R, w, device=device)]
        return out

    def bind(self, P) -> None:
        """Keep references to the LayerNorm parameters that live in P[6:]."""
        if self.norm == "layernorm":
            self.gb = [(P[6 + 2 * i], P[7 + 2 * i]) for i in range(len(self.widths))]

    def state(self):
        st = {}
        if self.door_h:
            st["m"] = [v.clone() for v in self.m]
        if self.norm == "scale":
            st["v"] = [v.clone() for v in self.v]
        return st

    def load_state(self, st):
        if self.door_h and st.get("m") is not None:
            for dst, src in zip(self.m, st["m"]):
                dst.copy_(src)
        if self.norm == "scale" and st.get("v") is not None:
            for dst, src in zip(self.v, st["v"]):
                dst.copy_(src)

    def phi(self, z, layer=0, train=False):
        if self.norm == "layernorm":                       # before the nonlinearity
            mu = z.mean(-1, keepdim=True)
            sd = (z.var(-1, unbiased=False, keepdim=True) + 1e-5).sqrt()
            g, b = self.gb[layer]
            z = g[:, None, :] * ((z - mu) / sd) + b[:, None, :]
            return torch.clamp(z, min=0.0)
        a = torch.clamp(z, min=0.0)
        if self.door_h:
            return a - self.m[layer][:, None, :]
        if self.norm == "scale":                           # the complement of door H
            return a / self.v[layer][:, None, :].sqrt().clamp(min=1e-12)
        return a

    def update(self, z1, z2) -> None:
        if self.door_h:
            for mi, z in zip(self.m, (z1, z2)):
                mi.mul_(1.0 - self.beta).add_(torch.clamp(z, min=0.0).mean(1), alpha=self.beta)
        if self.norm == "scale":
            for vi, z in zip(self.v, (z1, z2)):
                vi.mul_(1.0 - self.beta).add_(
                    torch.clamp(z, min=0.0).pow(2).mean(1), alpha=self.beta)

    def post_update(self, P, lr: float) -> None:
        if self.door_b == "none":
            return
        P[1].zero_()                                   # b1: the layer has no bias at all
        if self.door_b == "zero":
            P[3].zero_()                               # b2 too
        else:
            P[3].mul_(1.0 - lr * self.lam)             # decoupled weight decay on b2

    def stats(self, r: int) -> dict:
        out = {}
        if self.door_h:
            out |= {"m_l1": float(self.m[0][r].mean()), "m_l2": float(self.m[1][r].mean())}
        if self.norm == "scale":
            out |= {"s_l1": float(self.v[0][r].mean().sqrt()),
                    "s_l2": float(self.v[1][r].mean().sqrt())}
        if self.norm == "layernorm":
            out |= {"gamma_l1": float(self.gb[0][0][r].mean()),
                    "beta_l1": float(self.gb[0][1][r].mean())}
        return out


def make_act(arm: str, lam: float = 0.0, beta: float = 0.01) -> Act:
    if arm not in DOORS:
        raise SystemExit(f"unknown arm {arm!r}; known: {','.join(DOORS)}")
    return ReLUDoors(arm, lam, beta)


# --------------------------------------------------------------------------
# data (spec §2.3)
# --------------------------------------------------------------------------

def standardize(x_raw: torch.Tensor) -> torch.Tensor:
    """(x/255 - mean_c) / std_c per channel plane; x_raw is the host's [0,1] tensor (..., 3072)."""
    mean = torch.tensor(STD_MEAN, dtype=x_raw.dtype, device=x_raw.device).repeat_interleave(1024)
    std = torch.tensor(STD_STD, dtype=x_raw.dtype, device=x_raw.device).repeat_interleave(1024)
    return (x_raw - mean) / std


def slot_inputs(cifar: RC.Cifar10, seed: int, cond: str, device, center: bool = False
                ) -> torch.Tensor:
    """The seed's 1200 images.  Door C subtracts their own mean image and does NOT divide,
    so the scale stays raw and only r = |xbar| / rms|x - xbar| changes (1.911 -> 0.121)."""
    x = cifar.images(RC.subset_idx(seed), device)              # host: (1200, 3072) in [0,1]
    if cond != "raw":
        x = standardize(x)
    return x - x.mean(0) if center else x


# --------------------------------------------------------------------------
# stacked forward / evaluation
# --------------------------------------------------------------------------

def forward(P, X, act: Act, train: bool = False):
    """P: six stacked tensors (R, ...); X: (R, B, 3072).  Same maps as H.forward, batched."""
    W1, b1, W2, b2, W3, b3 = P[:6]   # P[6:] is LayerNorm's gamma/beta when present
    z1 = torch.baddbmm(b1[:, None, :], X, W1.transpose(1, 2))
    a1 = act.phi(z1, 0, train)
    z2 = torch.baddbmm(b2[:, None, :], a1, W2.transpose(1, 2))
    a2 = act.phi(z2, 1, train)
    return z1, a1, z2, a2, torch.baddbmm(b3[:, None, :], a2, W3.transpose(1, 2))


def _hist_rows(v: np.ndarray, edges: np.ndarray, lo: float, hi: float):
    """Per-run histogram over the flattened (B, n) values; `oob` is the exact complement."""
    h = [np.histogram(row.reshape(-1), bins=edges)[0].astype(np.int32) for row in v]
    oob = [np.int64(((row < lo) | (row > hi)).sum()) for row in v]
    return h, oob


@torch.no_grad()
def evaluate(P, X, Y, act: Act, live: list[int] | None = None
             ) -> tuple[list[dict], list[dict], list[np.ndarray]]:
    """Per-run metrics (the host's evaluate_rl, column for column) and per-run histograms
    (the host's preact_hist + alpha and the theta histogram for the adaptive arms).
    Only slots in `live` are read (default: all); the others get empty entries."""
    R = X.shape[0]
    live = list(range(R)) if live is None else live
    z1, a1, z2, a2, logits = forward(P, X, act, train=False)
    acc = (logits.argmax(-1) == Y).float().mean(1)
    rows = [{"acc": float(acc[r])} if r in live else {} for r in range(R)]
    hists = [{} for _ in range(R)]
    zs = []
    for li, (tag, z, a) in enumerate((("l1", z1, a1), ("l2", z2, a2))):
        d = act.dphi(z, li)
        for r in live:
            # the host's evaluate_rl expressions on the slot's own (1200, n) tensors; median
            # without dim is CUDA-deterministic, median(dim) is not
            zr, ar_, dr = z[r], a[r], d[r]
            rows[r].update({
                f"dead_frac_{tag}": float((dr.abs().amax(0) < DEAD_TOL).float().mean()),
                f"zeroout_{tag}": float((ar_.abs().amax(0) == 0).float().mean()),
                f"zbar_{tag}": float(zr.mean(0).median()),
                f"zsd_{tag}": float(zr.std(0).median()),
                f"zbar_min_{tag}": float(zr.mean(0).amin()),
                f"mob_{tag}": float(dr.mean(0).median()),
                f"eff_rank_{tag}": H.eff_rank(ar_),
                # spec §4: the collapse is "the unit stops crossing 0", which the mean alone
                # does not capture once the doors pin zbar -- so take P(z>0) directly.
                f"gate_zero_frac_{tag}": float((dr == 0).float().mean()),
                f"sink_ratio_{tag}": float((zr.mean(0) / zr.std(0).clamp(min=1e-12)).median()),
                f"p_pos_{tag}": float((zr > 0).float().mean(0).median()),
                f"fold_mean_{tag}": float(torch.minimum((zr > 0).float().mean(0),
                                                        (zr <= 0).float().mean(0)).mean()),
                f"onesided_frac_{tag}": float((torch.minimum((zr > 0).float().mean(0),
                                                             (zr <= 0).float().mean(0)) < 0.01
                                               ).float().mean()),
                f"skew_{tag}": float((((zr - zr.mean(0)) / zr.std(0).clamp(min=1e-12)) ** 3
                                      ).mean(0).median()),
                f"bias_over_sd_{tag}": float(P[2 * li + 1][r].abs().mean()
                                             / float(zr.std(0).median().clamp(min=1e-12))),
                f"bias_share_{tag}": float(P[2 * li + 1][r].mean()
                                           / zr.mean(0).mean().clamp(min=1e-12).abs()
                                           if float(zr.mean(0).mean().abs()) > 1e-12 else 0.0),
                f"abs_zbar_{tag}": float(zr.mean(0).abs().max())})
        k = li + 1
        v = z[live].cpu().numpy()
        z16 = [None] * R
        h, oob = _hist_rows(v, RC.HIST_EDGES, RC.HIST_LO, RC.HIST_HI)
        m = torch.stack([z[r].mean(0) for r in live]).cpu().numpy().astype(np.float32) if live else None
        if act.adaptive:
            al = act.alpha(li)                                             # (R, n)
            th = (2.0 * al[live][:, None, :] * z[live]).cpu().numpy()
            th_h, th_oob = _hist_rows(th, TH_EDGES, TH_LO, TH_HI)
            aln = al[live].cpu().numpy().astype(np.float32)
        for j, r in enumerate(live):
            z16[r] = v[j].astype(np.float16)
            hists[r][f"h{k}"], hists[r][f"oob{k}"], hists[r][f"m{k}"] = h[j], oob[j], m[j]
            if act.adaptive:
                hists[r][f"th{k}"], hists[r][f"thoob{k}"], hists[r][f"alpha{k}"] = th_h[j], th_oob[j], aln[j]
        zs.append(z16)
    for r in live:                                   # r of layer 2's input (spec §4)
        ub = a1[r].mean(0)
        dev = (a1[r] - ub).pow(2).sum(1).mean().sqrt()
        rows[r]["r_a1"] = float(ub.norm() / dev) if float(dev) > 0 else float("nan")
    for i, tag in enumerate(("l1", "l2", "l3")):
        for r in live:
            rows[r][f"w_norm_{tag}"] = float(P[2 * i][r].norm(dim=1).median())
    for r in live:
        rows[r].update(act.stats(r))
    return rows, hists, zs


def write_snapshot(path: Path, P, act: Act, r: int, z=None) -> None:
    """The slot's weights, biases and (adaptive arms) alpha state, float32, plus -- at a
    task's end -- z1/z2 on the 1200 images as float16 (the float32 values are rebuilt
    bit for bit from the weights by `replay`)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    d = {k: P[i][r].detach().cpu().numpy() for i, k in enumerate(("W1", "b1", "W2", "b2", "W3", "b3"))}
    for i, k in enumerate(("g1", "bn1", "g2", "bn2")):        # LayerNorm's gamma/beta, when present
        if len(P) > 6 + i:
            d[k] = P[6 + i][r].detach().cpu().numpy()
    if act.adaptive:
        d["V1"], d["V2"] = act.V[0][r].cpu().numpy(), act.V[1][r].cpu().numpy()
    if getattr(act, "door_h", False):
        d["m1"], d["m2"] = act.m[0][r].cpu().numpy(), act.m[1][r].cpu().numpy()
    if z is not None:
        d["z1"], d["z2"] = z[0][r], z[1][r]
    np.savez(path, **d)


def snapshot_path(out: Path, arm: str, cond: str, seed: int, task: int) -> Path:
    return out / "snap" / f"{arm}_{cond}_seed{seed}" / f"t{task:02d}.npz"


@torch.no_grad()
def replay_stack(out: Path, arm: str, slots: list[tuple[int, str]], task: int, device=None,
                 cifar: RC.Cifar10 | None = None, lam=0.0, beta=0.01):
    """Rebuild stacked (z1, a1, z2, a2, logits, Y) on each slot's 1200 images from the saved
    snapshots.  With the run's own slot list (provenance "slots", same order) this is the
    forward `evaluate` ran, bit for bit on the same device (S-snap); other layouts differ by
    batched-BLAS round-off (up to ~1e-4 absolute at |z| ~ 50)."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cifar = cifar or RC.Cifar10()
    ds = [np.load(snapshot_path(out, arm, cd, s, task)) for s, cd in slots]
    P = [torch.stack([torch.from_numpy(d[k]) for d in ds]).to(device)
         for k in ("W1", "b1", "W2", "b2", "W3", "b3")]
    act = make_act(arm, lam, beta)
    act.init_state(len(slots), device, key="replay")
    if act.adaptive:
        act.V = [torch.stack([torch.from_numpy(d[k]) for d in ds]).to(device) for k in ("V1", "V2")]
    if act.door_h:
        act.m = [torch.stack([torch.from_numpy(d[k]) for d in ds]).to(device) for k in ("m1", "m2")]
    X = torch.stack([slot_inputs(cifar, s, cd, device, center=act.door_c)
                     for s, cd in slots])
    Y = []
    for s, cd in slots:
        g = H.stream("rlc_labels", s)
        for _ in range(task):
            y = RC.task_labels(g)
        Y.append(y)
    z1, a1, z2, a2, logits = forward(P, X, act, train=False)
    return z1, a1, z2, a2, logits, torch.stack(Y).to(device)


def replay(out: Path, arm: str, cond: str, seed: int, task: int, device=None,
           cifar: RC.Cifar10 | None = None, **kw):
    """One slot on its own (R=1): the snapshot's network on its 1200 images."""
    return tuple(v[0] for v in replay_stack(out, arm, [(seed, cond)], task, device, cifar, **kw))


def write_hist(path: Path, hs: list[dict], accs: list[float]) -> None:
    if len(hs) != len(accs):
        raise AssertionError(f"{len(hs)} histograms vs {len(accs)} accuracies")
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(hs[0].keys())
    extra = {"th_edges": TH_EDGES} if "th1" in keys else {}
    np.savez_compressed(path, edges=RC.HIST_EDGES, acc=np.asarray(accs, dtype=np.float64),
                        **{k: np.stack([h[k] for h in hs]) for k in keys}, **extra)


# --------------------------------------------------------------------------
# the stacked run (host run_one, R slots at once)
# --------------------------------------------------------------------------

def git_state() -> dict:
    try:
        st = subprocess.run(["git", "-C", str(H.REPO), "status", "--porcelain", "src", "analysis"],
                            capture_output=True, text=True, timeout=30).stdout
        return {"git_hash": H.git_hash(), "dirty_src_analysis": [l for l in st.splitlines() if l]}
    except Exception:
        return {"git_hash": "unknown", "dirty_src_analysis": ["unknown"]}


def run(arm: str, seeds: list[int], conds: list[str], n_tasks: int, epochs: int, device,
        out: Path, lr: float = LR, lam: float = 0.0, beta: float = 0.01,
        progress=None, cifar: RC.Cifar10 | None = None,
        debug: dict | None = None, perturb: list[float] | None = None,
        nan_slot: int | None = None, snapshots: bool = True, checkpoint: bool = False,
        resume: bool = False, graph: bool = True, lifecycle=None) -> dict:
    """Train R = len(seeds) * len(conds) runs in lockstep and write their rows/hists/snapshots.

    Check-only hooks (never used by the main run): `perturb[r]` multiplies slot r's W1 by
    (1 + perturb[r]) at init, so repeated (seed, cond) slots become separate realizations of
    one run (S-reuse); `nan_slot` poisons that slot's W1 at init (S-diverge).

    `checkpoint` writes out/ckpt.pt after every task (weights, Adam moments and step, alpha
    statistics, every generator's state, the rows); `resume` continues from it, bit for bit
    (S-resume).  A checkpoint of a finished run can also extend it to more tasks.

    `graph` (cuda only) captures one training step as a CUDA graph and replays it: the same
    kernels on the same static tensors, so the rows are the eager engine's bit for bit (S-graph),
    without a host round trip per step."""
    t_start = time.time()
    progress = progress or (lambda m: print(m, flush=True))   # a redirected stdout is block-buffered
    act = make_act(arm, lam, beta)
    slots = [(s, cd) for s in seeds for cd in conds]
    R = len(slots)
    useeds = list(dict.fromkeys(seeds))       # each seed's streams advance once per draw
    cifar = cifar or RC.Cifar10()
    X = torch.stack([slot_inputs(cifar, s, cd, device, center=act.door_c)
                     for s, cd in slots])                                        # (R, 1200, 3072)
    init = {s: [q.detach() for q in H.init_params(s, device, DIMS)] for s in useeds}
    P = [torch.stack([init[s][i] for s, cd in slots]) for i in range(6)]
    if perturb is not None:
        P[0] = P[0] * torch.tensor(perturb, dtype=P[0].dtype, device=device)[:, None, None].add(1.0)
    if nan_slot is not None:
        P[0][nan_slot] = float("nan")
    act.init_state(R, device, key=f"{arm}|{seeds}|{conds}")
    P = P + act.extra_params(R, device)            # LayerNorm's gamma/beta, optimised by the run's Adam
    P = [q.contiguous().requires_grad_(True) for q in P]
    act.bind(P)
    adam_m = [torch.zeros_like(q) for q in P]
    adam_v = [torch.zeros_like(q) for q in P]
    tc = 0
    g_lab = {s: H.stream("rlc_labels", s) for s in useeds}
    g_batch = {s: H.stream("rlc_batch", s) for s in useeds}
    ar = torch.arange(R, device=device)[:, None]
    spt = STEPS_PER_EPOCH * epochs
    alive = torch.ones(R, dtype=torch.bool, device=device)
    rows, hists, diverged = [], [[] for _ in range(R)], []
    out.mkdir(parents=True, exist_ok=True)
    if debug is not None:
        debug["init"] = [q.detach().cpu().clone() for q in P]
    step_ms = float("nan")
    meta = {"arm": arm, "seeds": seeds, "conds": conds, "epochs": epochs, "lr": lr,
            "lam": lam, "beta": beta, "doors": DOORS[arm],
            "perturb": perturb, "nan_slot": nan_slot}
    ck = out / "ckpt.pt"
    git_states, resumed, t_first = [git_state()], [], 1
    if resume and ck.exists():
        st = torch.load(ck, map_location="cpu", weights_only=False)   # generator states must stay on the cpu
        if st["meta"] != meta:
            raise SystemExit(f"{ck} belongs to another configuration: {st['meta']}")
        with torch.no_grad():
            for dst, src in ((P, st["P"]), (adam_m, st["m"]), (adam_v, st["v"])):
                for q, v in zip(dst, src):
                    q.copy_(v)
        tc = st["tc"]
        act.load_state(st["act_state"])       # door H's EMA lives here (S-resume)
        if act.stochastic:
            act.gen.set_state(st["rsl_state"])
        for s in useeds:
            g_lab[s].set_state(st["g_lab"][s])
            g_batch[s].set_state(st["g_batch"][s])
        alive = st["alive"].to(device)
        rows, diverged, step_ms = st["rows"], st["diverged"], st["step_ms"]
        git_states = st["git_states"] + git_states
        resumed = st["resumed"] + [st["t"] + 1]
        t_first = st["t"] + 1
        for r, (s, cd) in enumerate(slots):             # histories up to the checkpointed task
            f = out / "hist" / f"{arm}_{cd}_seed{s}.npz"
            n_done = sum(1 for q in rows if q["slot"] == r and "memo_acc" in q)
            if f.exists() and n_done:
                d = np.load(f)
                keys = [k for k in d.files if k not in ("edges", "acc", "th_edges")]
                hists[r] = [{k: d[k][i] for k in keys} for i in range(n_done)]
        progress(f"[{time.strftime('%T')}] {arm} resumed from {ck} at task {t_first}")
    elif snapshots:
        for r, (s, cd) in enumerate(slots):
            write_snapshot(snapshot_path(out, arm, cd, s, 0), P, act, r)      # t00 = init

    # ---- one training step on static tensors (eager, or captured once and replayed)
    b1, b2, eps = 0.9, 0.999, 1e-8
    static_idx = torch.zeros(R, BATCH, dtype=torch.long, device=device)
    Ydev = torch.zeros(R, N_IMAGES, dtype=torch.long, device=device)
    inv_c1 = torch.zeros((), device=device)          # x / c (python float) == x * float32(1/c), bit for bit
    inv_c2 = torch.zeros((), device=device)
    step_t = torch.zeros((), dtype=torch.long, device=device)
    acc_sum = torch.zeros(R, device=device)
    bad_step = torch.full((R,), -1, dtype=torch.long, device=device)
    last_hit = torch.zeros(R, device=device)

    def step():
        xb, yb = X[ar, static_idx], Ydev[ar, static_idx]                # gathers, no arithmetic
        z1, a1, z2, a2, z3 = forward(P, xb, act, train=True)
        lossv = F.cross_entropy(z3.reshape(-1, N_CLASSES), yb.reshape(-1),
                                reduction="none").view(R, BATCH).mean(1)
        hit = (z3.detach().argmax(-1) == yb).float().mean(1)
        acc_sum.add_(hit)
        last_hit.copy_(hit)
        grads = torch.autograd.grad(lossv.sum(), P)
        with torch.no_grad():
            bad = ~torch.isfinite(lossv)
            bad_step.copy_(torch.where((bad_step < 0) & bad, step_t, bad_step))
            step_t.add_(1)
            for p, gr, mi, vi in zip(P, grads, adam_m, adam_v):
                mi.mul_(b1).add_(gr, alpha=1 - b1)
                vi.mul_(b2).addcmul_(gr, gr, value=1 - b2)
                p.sub_(lr * (mi * inv_c1) / ((vi * inv_c2).sqrt() + eps))
            act.post_update(P, lr)                 # door B: b1 held at 0, b2 decayed or zeroed
            act.update(z1.detach(), z2.detach())   # door H: advance the EMA of phi(z)

    use_graph = graph and device.type == "cuda" and debug is None
    cg = None
    if use_graph:
        # warm-up and capture both execute the step, so every tensor they touch is saved and
        # put back in place afterwards (the graph keeps pointers to these very tensors)
        keep = [q.detach().clone() for q in (*P, *adam_m, *adam_v, acc_sum, bad_step, step_t, last_hit)]
        keep_act = act.state()
        inv_c1.fill_(1.0); inv_c2.fill_(1.0)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                act.begin_step(R, BATCH, device)
                step()
        torch.cuda.current_stream().wait_stream(side)
        cg = torch.cuda.CUDAGraph()
        act.begin_step(R, BATCH, device)
        with torch.cuda.graph(cg):
            step()
        with torch.no_grad():
            for q, v in zip((*P, *adam_m, *adam_v, acc_sum, bad_step, step_t, last_hit), keep):
                q.copy_(v)
        act.load_state(keep_act)
        del keep

    # Optional read-only observer / task-boundary STOP for registered extensions.
    if lifecycle is not None:
        lifecycle("ready", locals())
    for t in range(t_first, n_tasks + 1):
        lab = {s: RC.task_labels(g_lab[s]) for s in useeds}               # once per seed per task
        Y = torch.stack([lab[s] for s, cd in slots]).to(device)           # (R, 1200)
        Ydev.copy_(Y)
        if lifecycle is not None and t == t_first:
            lifecycle("labels", locals())
        if debug is not None:
            debug.setdefault("labels", []).append(Y.cpu().clone())
        acc_sum.zero_()
        bad_step.fill_(-1)
        step_t.zero_()
        t0 = time.time()
        for e in range(epochs):
            order = {s: torch.randperm(N_IMAGES, generator=g_batch[s]) for s in useeds}
            ORD = torch.stack([order[s] for s, cd in slots]).to(device)  # (R, 1200)
            if lifecycle is not None and t == t_first and e == 0:
                lifecycle("order", locals())
            if debug is not None:
                debug.setdefault("orders", []).append(ORD.cpu().clone())
            for j in range(STEPS_PER_EPOCH):
                static_idx.copy_(ORD[:, j * BATCH:(j + 1) * BATCH])
                tc += 1
                inv_c1.fill_(1.0 / (1 - b1 ** tc))
                inv_c2.fill_(1.0 / (1 - b2 ** tc))
                act.begin_step(R, BATCH, device)
                if cg is not None:
                    cg.replay()
                else:
                    step()
                if debug is not None:
                    debug.setdefault("online", []).append(last_hit.cpu().clone())
        if device.type == "cuda":
            torch.cuda.synchronize()
        step_ms = 1e3 * (time.time() - t0) / spt

        with torch.no_grad():
            finite = torch.stack([torch.isfinite(q).flatten(1).all(1) for q in P]).all(0)
            newly = alive & ((bad_step >= 0) | ~finite)
            for r in torch.nonzero(newly).flatten().tolist():
                s, cd = slots[r]
                diverged.append({"diverged": True, "task": t,
                                 "step": (t - 1) * spt + max(int(bad_step[r]), 0),
                                 "seed": s, "cond": cd, "arm": arm})
                rows.append({"arm": arm, "cond": cd, "seed": s, "slot": r, "lr": lr, "task": t,
                             "iv": "none", "acc": float("nan")})
            alive &= ~newly
            m, hs, zs = evaluate(P, X, Y, act, torch.nonzero(alive).flatten().tolist())
        for r in torch.nonzero(alive).flatten().tolist():
            s, cd = slots[r]
            rows.append({"arm": arm, "cond": cd, "seed": s, "slot": r, "lr": lr, "task": t,
                         "iv": "none", "online_acc": float(acc_sum[r]) / spt,
                         "memo_acc": m[r]["acc"], **m[r]})
            hists[r].append(hs[r])
            if snapshots:
                write_snapshot(snapshot_path(out, arm, cd, s, t), P, act, r, zs)
        rows.sort(key=lambda q: (q["slot"], q["task"]))
        H.write_csv(out / "per_task.csv", rows)
        for r in range(R):
            if hists[r] and snapshots:
                s, cd = slots[r]
                write_hist(out / "hist" / f"{arm}_{cd}_seed{s}.npz", hists[r],
                           [q["memo_acc"] for q in rows if q["slot"] == r and "memo_acc" in q])
        if checkpoint:
            tmp = out / "ckpt.pt.tmp"
            torch.save({"t": t, "meta": meta, "P": [q.detach() for q in P], "m": adam_m, "v": adam_v,
                        "tc": tc, "act_state": act.state(),
                        "rsl_state": act.gen.get_state() if act.stochastic else None,
                        "g_lab": {s: g_lab[s].get_state() for s in useeds},
                        "g_batch": {s: g_batch[s].get_state() for s in useeds},
                        "alive": alive.cpu(), "rows": rows, "diverged": diverged, "step_ms": step_ms,
                        "git_states": git_states, "resumed": resumed}, tmp)
            os.replace(tmp, ck)
        if lifecycle is not None and lifecycle("task_end", locals()):
            break
        on = acc_sum[alive] / spt
        memo = torch.tensor([m[r]["acc"] for r in range(R) if alive[r]])
        el = time.time() - t_start
        progress(f"[{time.strftime('%T')}] {arm} task {t:2d}/{n_tasks} alive {int(alive.sum())}/{R} "
                 f"online {float(on.mean()) if len(on) else float('nan'):.3f} "
                 f"(min {float(on.min()) if len(on) else float('nan'):.3f}) "
                 f"memo min {float(memo.min()) if len(memo) else float('nan'):.3f} "
                 f"{step_ms:.2f} ms/step  {el/60:.0f} min, "
                 f"ETA {el/(t-t_first+1)*(n_tasks-t)/60:.0f} min")

    prov = {"run_id": EXPERIMENT, **git_states[0], "git_states": git_states, "resumed_at_task": resumed,
            "arm": arm, "conds": conds, "seeds": seeds,
            "slots": [{"seed": s, "cond": cd} for s, cd in slots], "R": R, "lr": lr,
            "n_tasks": n_tasks, "epochs_per_task": epochs, "batch": BATCH, "steps_per_task": spt,
            "n_images": N_IMAGES, "train_n": RC.TRAIN_N, "dims": list(DIMS), "n_classes": N_CLASSES,
            "doors": DOORS[arm], "door_lam": lam, "door_beta": beta, "intervention": "none",
            "optimizer": "adam", "weight_decay": 0.0, "data_sha256": cifar.sha256,
            "subset_sha256": {str(s): hashlib.sha256(
                np.sort(RC.subset_idx(s).numpy()).tobytes()).hexdigest() for s in seeds},
            "std": {"mean": STD_MEAN, "std": STD_STD, "planes": "R,G,B x 1024"},
            "rng_roles": ["rlc_subset", "rlc_labels", "rlc_batch", "init"]
                         + (["rsl_noise(cuda, per process)"] if act.stochastic else []),
            "rsl_noise_seed": getattr(act, "noise_seed", None),
            "hist": {"lo": RC.HIST_LO, "hi": RC.HIST_HI, "bins": RC.HIST_NB, "dtype": "int32",
                     "theta": {"lo": TH_LO, "hi": TH_HI, "bins": TH_NB} if act.adaptive else None,
                     "path": "hist/<arm>_<cond>_seed<seed>.npz"},
            "engine": "stacked baddbmm, autograd, elementwise Adam (spec §2.2)"
                      + (", one step captured as a CUDA graph" if use_graph else ", eager"),
            "snapshots": "snap/<arm>_<cond>_seed<seed>/t<task>.npz: W1,b1,W2,b2,W3,b3 float32 "
                         "(+m1,m2 for door H) at init (t00) and every task's end, plus z1,z2 "
                         "(1200x100, float16, images in subset order) at every task's end; exact "
                         "float32 z via replay()",
            "device": str(device), "torch": torch.__version__,
            "cublas_workspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "step_ms_last_task": step_ms, "wall_clock_s": time.time() - t_start,
            "divergences": diverged,
            "check_hooks": {"perturb": perturb, "nan_slot": nan_slot, "snapshots": snapshots}}
    (out / "provenance.json").write_text(json.dumps(prov, indent=2))
    progress(f"wrote {out}/per_task.csv ({len(rows)} rows, {(time.time()-t_start)/60:.1f} min)")
    return prov


def parse_ints(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run"])
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--conds", default="raw")
    ap.add_argument("--tasks", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lam", type=float, default=0.0, help="door B: decoupled WD on b2 (spec §2.2)")
    ap.add_argument("--beta", type=float, default=0.01, help="door H: EMA rate")
    ap.add_argument("--out", default=None, help="default results/<experiment>/<arm>")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--threads", type=int, default=2, help="torch cpu threads (eval: eff_rank, histograms)")
    ap.add_argument("--no-resume", action="store_true", help="ignore an existing out/ckpt.pt")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    device = H.setup(a.device)
    out = Path(a.out) if a.out else OUT_ROOT / a.arm
    run(a.arm, parse_ints(a.seeds), a.conds.split(","), a.tasks, a.epochs, device, out,
        lam=a.lam, beta=a.beta, checkpoint=True, resume=not a.no_resume)


if __name__ == "__main__":
    main()
