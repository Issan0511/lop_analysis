#!/usr/bin/env python3
"""layer_chimera_cifar_0921 -- layerwise activation chimera on RL-CIFAR (spec_layer_chimera_cifar_0921.md).

    python3 src/layer_chimera_cifar_0921.py run --cell EL                      # raw x seeds 0-9 = 10 runs
    python3 src/layer_chimera_cifar_0921.py run --cell LL --seeds 100 --tasks 3 --out DIR

A fork of src/rlcifar_mlp_battle_0918.py (spec 2.2).  The ONLY addition is `Chimera`, which
sends layer 1 to one activation and layer 2 to another; `forward` already passes the layer
index, so with act1 == act2 every kernel, every op order and every stream is the parent's and
the run is bit-identical to the parent engine in the same configuration (check S1, S-off).
The cells are LL, EL, GL, LE, EE, GE and the reference GG (spec 3); L = leaky .1 (the parent's
`LR`), E = ELU(1), G = GELU.  Registered condition: raw only, R = 10 (spec 2.3).

The box is pmnist_rlcifar_0907's (3072-100-100-10, 1200 CIFAR images per seed, random labels
per task, 400 epochs x 75 steps of batch 16, Adam 1e-3, no weight decay) and every stream --
image subset, labels, batch order, init -- is the host's, drawn from the host's generators in
the host's order, so slot (seed s, cond) sees exactly what host run seed s sees.  What differs
is only the shape of the computation: the R runs' parameters are stacked on a leading axis and
the three affine maps are `baddbmm` instead of `@` (spec §2.2).  Forward values are bit-identical
to the host's; gradients differ by the summation order inside bmm's backward (~1e-7 relative,
check S-stack), so trajectories are not bit-comparable with the host over 30,000 steps.

Nothing in pmnist_0905 / pmnist_rlmnist_0906 / pmnist_rlcifar_0907 / rlcifar_mlp_battle_0918
is modified.
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

EXPERIMENT = "layer_chimera_cifar_0921"
OUT_ROOT = H.REPO / "results" / EXPERIMENT
DIMS = RC.DIMS                                 # (3072, 100, 100, 10)
N_IMAGES, BATCH, STEPS_PER_EPOCH = RC.N_IMAGES, RC.BATCH, RC.STEPS_PER_EPOCH
N_CLASSES = RC.N_CLASSES
DEAD_TOL = H.DEAD_TOL
LR = 1e-3
CONDS = ("raw",)                                       # spec §2.3: raw only
# spec §3: cell -> (layer-1 arm, layer-2 arm) in the parent engine's names.  The name reads
# "first layer, second layer"; L = leaky .1 (parent `LR`), E = ELU(1), G = GELU.
CELLS = {"LL": ("LR", "LR"), "EL": ("ELU", "LR"), "GL": ("GELU", "LR"),
         "LE": ("LR", "ELU"), "EE": ("ELU", "ELU"), "GE": ("GELU", "ELU"),
         "GG": ("GELU", "GELU")}
CELL_ORDER = ("LL", "EL", "GL", "LE", "EE", "GE", "GG")          # spec §8 launch order

# spec §2.3: Lillo & Cheney's CIFAR-10 Normalize, per channel plane (R, G, B: 1024 each)
STD_MEAN = (0.4914, 0.4822, 0.4465)
STD_STD = (0.2470, 0.2435, 0.2616)

# phase histogram of theta = 2 alpha_i z for the adaptive arms (spec §5, descriptive)
TH_LO, TH_HI, TH_NB = -6.0 * math.pi, 6.0 * math.pi, 1885           # 0.02 rad bins
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

    def stats(self, r: int) -> dict:
        return {}


class ReLU(Act):
    name = "R"

    def phi(self, z, layer=0, train=False):
        return torch.clamp(z, min=0.0)

    def dphi(self, z, layer=0):
        return (z > 0).to(z.dtype)


class Leaky(Act):
    def __init__(self, name, slope):
        self.name, self.slope = name, slope

    def phi(self, z, layer=0, train=False):
        return torch.where(z > 0, z, self.slope * z)          # the host's form

    def dphi(self, z, layer=0):
        return torch.where(z > 0, torch.ones_like(z), torch.full_like(z, self.slope))


class SmoothLeaky(Act):
    """Lillo & Cheney (utils/custom_activations.py, verbatim form):
    alpha * x + (1 - alpha) * x * sigmoid(c * (x / p)), defaults alpha=0.1, p=3, c=5."""
    name = "SL"

    def __init__(self, a=0.1, c=5.0, p=3.0):
        self.a, self.c, self.p = a, c, p

    def _phi(self, z, a):
        return a * z + (1.0 - a) * z * torch.sigmoid(self.c * (z / self.p))

    def _dphi(self, z, a):
        s = torch.sigmoid(self.c * (z / self.p))
        return a + (1.0 - a) * (s + (self.c / self.p) * z * s * (1.0 - s))

    def phi(self, z, layer=0, train=False):
        return self._phi(z, self.a)

    def dphi(self, z, layer=0):
        return self._dphi(z, self.a)


class RandSmoothLeaky(SmoothLeaky):
    """Lillo & Cheney: in training r = empty_like(x).uniform_(l, u) (per element, per call);
    in eval r = (l + u) / 2."""
    name = "RSL"
    stochastic = True

    def __init__(self, l=0.125, u=0.333, c=5.0, p=3.0):
        super().__init__(a=(l + u) / 2, c=c, p=p)
        self.l, self.u = l, u
        self.gen = None

    def init_state(self, R, device, key):
        # one generator per process; the draw covers every slot at once, so the
        # sequence is a property of (arm, slots), not of the seed alone (spec §3)
        self.gen = torch.Generator(device=device)
        h = hashlib.sha256(f"{EXPERIMENT}|rsl_noise|{key}".encode()).digest()
        self.gen.manual_seed(int.from_bytes(h[:8], "little") & ((1 << 63) - 1))
        self.noise_seed = int(self.gen.initial_seed())
        self.noise = None

    def begin_step(self, R, B, device):
        # U(0,1) for layer 1 then layer 2, into fixed buffers (a captured graph reads them)
        if self.noise is None:
            self.noise = [torch.empty(R, B, w, device=device) for w in (DIMS[1], DIMS[2])]
        for buf in self.noise:
            torch.rand(buf.shape, generator=self.gen, out=buf)

    def state(self):
        return {"gen": self.gen.get_state()}

    def load_state(self, st):
        self.gen.set_state(st["gen"])

    def draw(self, z):
        return self.l + (self.u - self.l) * torch.rand(z.shape, generator=self.gen,
                                                       device=z.device, dtype=z.dtype)

    def phi(self, z, layer=0, train=False):
        if not train:
            return self._phi(z, self.a)
        if self.noise is None:                       # standalone use (S-rsl-mode)
            return self._phi(z, self.draw(z))
        return self._phi(z, self.l + (self.u - self.l) * self.noise[layer])


class ELU(Act):
    """F.elu, alpha=1 (nn.ELU as in L&C).  Training differentiates it through autograd, whose
    non-inplace backward in torch 2.13 is exp(z) on z <= 0 -- no float32 floor (the output+1
    form, elu(z)+1, is exactly 0 below -16.64 on cpu / -16.98 on cuda; spec addendum 1).
    The gate below is that exp(z) form, so the metrics read the derivative training used."""
    name = "ELU"

    def phi(self, z, layer=0, train=False):
        return F.elu(z, 1.0)

    def dphi(self, z, layer=0):
        return torch.where(z > 0, torch.ones_like(z), torch.exp(z))


class SiLU(Act):
    name = "SILU"

    def phi(self, z, layer=0, train=False):
        return z * torch.sigmoid(z)

    def dphi(self, z, layer=0):
        s = torch.sigmoid(z)
        return s * (1.0 + z * (1.0 - s))


class GELU(Act):
    name = "GELU"

    def phi(self, z, layer=0, train=False):
        return F.gelu(z)                                       # exact erf form

    def dphi(self, z, layer=0):
        cdf = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
        pdf = torch.exp(-0.5 * z * z) / math.sqrt(2.0 * PI)
        return cdf + z * pdf


class SnakeFamily(Act):
    """Adaptive-alpha Snake and the three kunekune cuts (spec §3).

    alpha_i = clip(c / W_i, lo, hi), W_i = sqrt(EMA_beta var_batch(z_i)), per run and unit,
    the host's AdaptiveSnake rule with the same op order (`* a.reciprocal()`).  theta = 2 alpha z.
      snake : z + sin^2(alpha z)/alpha everywhere
      kk    : Snake on theta in [-3pi/2, pi/2], slope-2 lines outside (C^2 at the joins)
      kk23  : (2/3) * kk
      kkt1  : Snake on theta in [-2pi, pi], identity outside (C^1 at the joins)
    """
    adaptive = True

    def __init__(self, name, kind, c=0.6, beta=0.01, lo=0.005, hi=3.0, widths=(DIMS[1], DIMS[2])):
        assert kind in ("snake", "kk", "kk23", "kkt1")
        self.name, self.kind, self.c, self.beta, self.lo, self.hi = name, kind, c, beta, lo, hi
        self.widths = widths
        self.V = None

    def init_state(self, R, device, key):
        self.V = [torch.ones(R, w, device=device) for w in self.widths]

    def state(self):
        return {"V": [v.clone() for v in self.V]}

    def load_state(self, st):
        for v, src in zip(self.V, st["V"]):
            v.copy_(src)                             # in place: a captured graph holds these

    def alpha(self, layer):
        return (self.c / self.V[layer].sqrt()).clamp(self.lo, self.hi)       # (R, n)

    def phi(self, z, layer=0, train=False):
        a = self.alpha(layer)[:, None, :]
        ia = a.reciprocal()
        snake = z + torch.sin(a * z) ** 2 * ia
        if self.kind == "snake":
            return snake
        th = 2.0 * a * z
        if self.kind in ("kk", "kk23"):
            left = 2.0 * z + (0.75 * PI) * ia + 0.5 * ia
            right = 2.0 * z - (0.25 * PI) * ia + 0.5 * ia
            out = torch.where(th < -1.5 * PI, left, torch.where(th > 0.5 * PI, right, snake))
            return out if self.kind == "kk" else out * (2.0 / 3.0)
        right = z + ia                                                        # kkt1
        return torch.where(th < -2.0 * PI, z, torch.where(th > PI, right, snake))

    def dphi(self, z, layer=0):
        a = self.alpha(layer)[:, None, :]
        th = 2.0 * a * z
        s = 1.0 + torch.sin(th)
        if self.kind == "snake":
            return s
        if self.kind in ("kk", "kk23"):
            inside = (th >= -1.5 * PI) & (th <= 0.5 * PI)
            out = torch.where(inside, s, torch.full_like(z, 2.0))
            return out if self.kind == "kk" else out * (2.0 / 3.0)
        inside = (th >= -2.0 * PI) & (th <= PI)
        return torch.where(inside, s, torch.ones_like(z))

    @torch.no_grad()
    def update(self, z1, z2):
        if self.beta == 0:
            return
        for l, z in ((0, z1), (1, z2)):
            self.V[l].mul_(1 - self.beta).add_(self.beta * z.var(1, unbiased=False))

    @torch.no_grad()
    def stats(self, r):
        out = {}
        for l in (0, 1):
            a = self.alpha(l)[r]; W = self.V[l][r].sqrt(); raw = self.c / W
            out[f"alpha_med_l{l+1}"] = float(a.median()); out[f"alpha_min_l{l+1}"] = float(a.min())
            out[f"alpha_max_l{l+1}"] = float(a.max())
            out[f"alpha_clip_frac_l{l+1}"] = float(((raw < self.lo) | (raw > self.hi)).float().mean())
            out[f"two_alpha_W_med_l{l+1}"] = float((2 * a * W).median())
        return out


class Chimera(Act):
    """One activation per hidden layer (spec §2.2).

    `forward` calls phi(z1, 0, train) and phi(z2, 1, train), so the only thing this class does
    is pick which of the two wrapped activations answers.  The wrapped calls keep the layer
    index, so an adaptive arm would still see its own per-layer alpha; with act1 == act2 the
    object is a transparent wrapper and the run reproduces the parent engine bit for bit."""
    adaptive = False
    stochastic = False

    def __init__(self, cell: str, a1: Act, a2: Act):
        self.name, self.a1, self.a2 = cell, a1, a2
        self.adaptive = a1.adaptive or a2.adaptive
        self.stochastic = a1.stochastic or a2.stochastic
        if self.adaptive or self.stochastic:                       # spec §3: the six cells are
            raise SystemExit(f"{cell}: only fixed, stateless activations are registered")

    def phi(self, z, layer=0, train=False):
        return (self.a1 if layer == 0 else self.a2).phi(z, layer, train)

    def dphi(self, z, layer=0):
        return (self.a1 if layer == 0 else self.a2).dphi(z, layer)

    def parts(self) -> tuple[str, str]:
        return self.a1.name, self.a2.name


def make_arm(arm: str, c=0.6, beta=0.01, lo=0.005, hi=3.0) -> Act:
    """The parent engine's activation table, verbatim."""
    if arm == "SNA":
        return SnakeFamily("SNA", "snake", c, beta, lo, hi)
    if arm == "KKA":
        return SnakeFamily("KKA", "kk", c, beta, lo, hi)
    if arm == "KKA23":
        return SnakeFamily("KKA23", "kk23", c, beta, lo, hi)
    if arm == "KKT1":
        return SnakeFamily("KKT1", "kkt1", c, beta, lo, hi)
    fixed = {"R": ReLU, "LK001": lambda: Leaky("LK001", 0.01), "LR": lambda: Leaky("LR", 0.1),
             "LK03": lambda: Leaky("LK03", 0.3), "SL": SmoothLeaky, "RSL": RandSmoothLeaky,
             "ELU": ELU, "SILU": SiLU, "GELU": GELU}
    if arm not in fixed:
        raise SystemExit(f"unknown arm {arm!r}")
    return fixed[arm]()


def make_act(cell: str, c=0.6, beta=0.01, lo=0.005, hi=3.0) -> Act:
    if cell not in CELLS:
        raise SystemExit(f"unknown cell {cell!r}; known: {','.join(CELL_ORDER)}")
    a1, a2 = (make_arm(a, c, beta, lo, hi) for a in CELLS[cell])
    return Chimera(cell, a1, a2)


# --------------------------------------------------------------------------
# data (spec §2.3)
# --------------------------------------------------------------------------

def standardize(x_raw: torch.Tensor) -> torch.Tensor:
    """(x/255 - mean_c) / std_c per channel plane; x_raw is the host's [0,1] tensor (..., 3072)."""
    mean = torch.tensor(STD_MEAN, dtype=x_raw.dtype, device=x_raw.device).repeat_interleave(1024)
    std = torch.tensor(STD_STD, dtype=x_raw.dtype, device=x_raw.device).repeat_interleave(1024)
    return (x_raw - mean) / std


def slot_inputs(cifar: RC.Cifar10, seed: int, cond: str, device) -> torch.Tensor:
    x = cifar.images(RC.subset_idx(seed), device)              # host: (1200, 3072) in [0,1]
    return x if cond == "raw" else standardize(x)


# --------------------------------------------------------------------------
# stacked forward / evaluation
# --------------------------------------------------------------------------

def forward(P, X, act: Act, train: bool = False):
    """P: six stacked tensors (R, ...); X: (R, B, 3072).  Same maps as H.forward, batched."""
    W1, b1, W2, b2, W3, b3 = P
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
                f"eff_rank_{tag}": H.eff_rank(ar_)})
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
    if act.adaptive:
        d["V1"], d["V2"] = act.V[0][r].cpu().numpy(), act.V[1][r].cpu().numpy()
    if z is not None:
        d["z1"], d["z2"] = z[0][r], z[1][r]
    np.savez(path, **d)


def snapshot_path(out: Path, arm: str, cond: str, seed: int, task: int) -> Path:
    return out / "snap" / f"{arm}_{cond}_seed{seed}" / f"t{task:02d}.npz"


@torch.no_grad()
def replay_stack(out: Path, arm: str, slots: list[tuple[int, str]], task: int, device=None,
                 cifar: RC.Cifar10 | None = None, c=0.6, beta=0.01, lo=0.005, hi=3.0):
    """Rebuild stacked (z1, a1, z2, a2, logits, Y) on each slot's 1200 images from the saved
    snapshots.  With the run's own slot list (provenance "slots", same order) this is the
    forward `evaluate` ran, bit for bit on the same device (S-snap); other layouts differ by
    batched-BLAS round-off (up to ~1e-4 absolute at |z| ~ 50)."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cifar = cifar or RC.Cifar10()
    ds = [np.load(snapshot_path(out, arm, cd, s, task)) for s, cd in slots]
    P = [torch.stack([torch.from_numpy(d[k]) for d in ds]).to(device)
         for k in ("W1", "b1", "W2", "b2", "W3", "b3")]
    act = make_act(arm, c, beta, lo, hi)
    act.init_state(len(slots), device, key="replay")
    if act.adaptive:
        act.V = [torch.stack([torch.from_numpy(d[k]) for d in ds]).to(device) for k in ("V1", "V2")]
    X = torch.stack([slot_inputs(cifar, s, cd, device) for s, cd in slots])
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
        out: Path, lr: float = LR, c: float = 0.6, beta: float = 0.01, lo: float = 0.005,
        hi: float = 3.0, progress=None, cifar: RC.Cifar10 | None = None,
        debug: dict | None = None, perturb: list[float] | None = None,
        nan_slot: int | None = None, snapshots: bool = True, checkpoint: bool = False,
        resume: bool = False, graph: bool = True) -> dict:
    """Train R = len(seeds) * len(conds) runs in lockstep and write their rows/hists/snapshots.

    `arm` is a cell name from CELLS; the per_task rows keep the parent engine's column name
    `arm` so that a cell's csv differs from the parent arm's only in that column (check S1).

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
    act = make_act(arm, c, beta, lo, hi)
    slots = [(s, cd) for s in seeds for cd in conds]
    R = len(slots)
    useeds = list(dict.fromkeys(seeds))       # each seed's streams advance once per draw
    cifar = cifar or RC.Cifar10()
    X = torch.stack([slot_inputs(cifar, s, cd, device) for s, cd in slots])      # (R, 1200, 3072)
    init = {s: [q.detach() for q in H.init_params(s, device, DIMS)] for s in useeds}
    P = [torch.stack([init[s][i] for s, cd in slots]) for i in range(6)]
    if perturb is not None:
        P[0] = P[0] * torch.tensor(perturb, dtype=P[0].dtype, device=device)[:, None, None].add(1.0)
    if nan_slot is not None:
        P[0][nan_slot] = float("nan")
    P = [q.contiguous().requires_grad_(True) for q in P]
    act.init_state(R, device, key=f"{arm}|{seeds}|{conds}")
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
    meta = {"arm": arm, "seeds": seeds, "conds": conds, "epochs": epochs, "lr": lr, "c": c,
            "beta": beta, "lo": lo, "hi": hi, "perturb": perturb, "nan_slot": nan_slot}
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
        if act.adaptive:
            act.load_state({"V": st["V"]})
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
            act.update(z1.detach(), z2.detach())

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

    for t in range(t_first, n_tasks + 1):
        lab = {s: RC.task_labels(g_lab[s]) for s in useeds}               # once per seed per task
        Y = torch.stack([lab[s] for s, cd in slots]).to(device)           # (R, 1200)
        Ydev.copy_(Y)
        if debug is not None:
            debug.setdefault("labels", []).append(Y.cpu().clone())
        acc_sum.zero_()
        bad_step.fill_(-1)
        step_t.zero_()
        t0 = time.time()
        for e in range(epochs):
            order = {s: torch.randperm(N_IMAGES, generator=g_batch[s]) for s in useeds}
            ORD = torch.stack([order[s] for s, cd in slots]).to(device)  # (R, 1200)
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
                        "tc": tc, "V": act.V if act.adaptive else None,
                        "rsl_state": act.gen.get_state() if act.stochastic else None,
                        "g_lab": {s: g_lab[s].get_state() for s in useeds},
                        "g_batch": {s: g_batch[s].get_state() for s in useeds},
                        "alive": alive.cpu(), "rows": rows, "diverged": diverged, "step_ms": step_ms,
                        "git_states": git_states, "resumed": resumed}, tmp)
            os.replace(tmp, ck)
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
            "cell": arm, "act1": CELLS[arm][0], "act2": CELLS[arm][1],
            "arm": arm, "conds": conds, "seeds": seeds,
            "slots": [{"seed": s, "cond": cd} for s, cd in slots], "R": R, "lr": lr,
            "n_tasks": n_tasks, "epochs_per_task": epochs, "batch": BATCH, "steps_per_task": spt,
            "n_images": N_IMAGES, "train_n": RC.TRAIN_N, "dims": list(DIMS), "n_classes": N_CLASSES,
            "sna_c": c, "sna_beta": beta, "alpha_lo": lo, "alpha_hi": hi, "intervention": "none",
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
                         "(+V1,V2 for adaptive arms) at init (t00) and every task's end, plus z1,z2 "
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
    ap.add_argument("--cell", required=True, help=f"one of {','.join(CELL_ORDER)}")
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--conds", default="raw", help="spec 2.3 registers raw only")
    ap.add_argument("--tasks", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--c", type=float, default=0.6)
    ap.add_argument("--beta", type=float, default=0.01)
    ap.add_argument("--alpha-lo", type=float, default=0.005)
    ap.add_argument("--alpha-hi", type=float, default=3.0)
    ap.add_argument("--out", default=None, help="default results/<experiment>/<cell>")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--threads", type=int, default=2, help="torch cpu threads (eval: eff_rank, histograms)")
    ap.add_argument("--no-resume", action="store_true", help="ignore an existing out/ckpt.pt")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    device = H.setup(a.device)
    out = Path(a.out) if a.out else OUT_ROOT / a.cell
    run(a.cell, parse_ints(a.seeds), a.conds.split(","), a.tasks, a.epochs, device, out,
        c=a.c, beta=a.beta, lo=a.alpha_lo, hi=a.alpha_hi, checkpoint=True, resume=not a.no_resume)


if __name__ == "__main__":
    main()
