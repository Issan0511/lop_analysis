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
import fcntl
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

EXPERIMENT = "rlcifar_mlp_battle_0918"
OUT_ROOT = H.REPO / "results" / EXPERIMENT
DIMS = RC.DIMS                                 # (3072, 100, 100, 10)
N_IMAGES, BATCH, STEPS_PER_EPOCH = RC.N_IMAGES, RC.BATCH, RC.STEPS_PER_EPOCH
N_CLASSES = RC.N_CLASSES
DEAD_TOL = H.DEAD_TOL
LR = 1e-3
CONDS = ("raw", "std")
ARM_ORDER = ("SNA", "KKA", "R", "LR", "KKT1", "KKA23", "SL", "RSL",
             "LK001", "LK03", "ELU", "SILU", "GELU")            # spec §7 launch order

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


def make_act(arm: str, c=0.6, beta=0.01, lo=0.005, hi=3.0) -> Act:
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
        raise SystemExit(f"unknown arm {arm!r}; known: {','.join(ARM_ORDER)}")
    return fixed[arm]()


# --------------------------------------------------------------------------
# label schedules (altlabels_cifar_0923 §1.1).  "iid" is the parent's: one fresh
# draw from the seed's rlc_labels stream per task.  The others draw A (1st), B
# (2nd), C (3rd) once up front -- the same first draws the iid run makes -- and
# replay them, so abab's tasks 1 and 2 are the iid run's tasks 1 and 2.
# --------------------------------------------------------------------------

SCHEDULES = {"iid": 0, "abab": 2, "aaaa": 1, "abc": 3}          # name -> fixed labelings needed


def schedule_index(schedule: str, t: int) -> int:
    """0-based index into the fixed labelings [A, B, C] for task t (1-based).

    abab: t odd -> A, t even -> B;  aaaa: always A;  abc: t mod 3 = 1/2/0 -> A/B/C.
    """
    n = SCHEDULES[schedule]
    if n == 0:
        raise ValueError("the iid schedule has no fixed labelings")
    return (t - 1) % n


def labels_sha256(fixed: dict[int, list[torch.Tensor]]) -> str:
    """One digest over every fixed labeling, seed by seed, in draw order."""
    h = hashlib.sha256()
    for s in sorted(fixed):
        for y in fixed[s]:
            h.update(np.ascontiguousarray(y.cpu().numpy().astype(np.int64)).tobytes())
    return h.hexdigest()


def labels_stats(fixed: dict[int, list[torch.Tensor]]) -> dict:
    """Per seed: the class histogram of each labeling and the pairwise agreement fractions.

    Two independent uniform labelings over 10 classes agree on 1/10 of the images, so these are
    the descriptive numbers the analysis quotes when it says A and B are "different labelings".
    """
    seeds = sorted(fixed)
    names = ("A", "B", "C")
    out: dict[str, list] = {"seeds": [int(s) for s in seeds]}
    y = {s: [q.cpu().numpy().astype(np.int64) for q in fixed[s]] for s in seeds}
    for i, nm in enumerate(names):
        if all(len(y[s]) > i for s in seeds):
            out[f"{nm}_counts"] = [np.bincount(y[s][i], minlength=N_CLASSES).tolist() for s in seeds]
    for i, j in ((0, 1), (0, 2), (1, 2)):
        if all(len(y[s]) > max(i, j) for s in seeds):
            out[f"agree_{names[i]}{names[j]}"] = [float((y[s][i] == y[s][j]).mean()) for s in seeds]
    return out


def write_labels(path: Path, fixed: dict[int, list[torch.Tensor]]) -> None:
    """labels.npz: A/B/C as (n_seeds, 1200) int64 where the schedule has them, plus seeds,
    each labeling's class counts and the pairwise agreement fractions.  Written through a
    temporary file so an interrupted write cannot leave a half-written labelling behind."""
    seeds = sorted(fixed)
    d = {"seeds": np.asarray(seeds, dtype=np.int64)}
    for i, name in enumerate(("A", "B", "C")):
        if all(len(fixed[s]) > i for s in seeds):
            d[name] = np.stack([fixed[s][i].cpu().numpy().astype(np.int64) for s in seeds])
    for k, v in labels_stats(fixed).items():
        if k != "seeds":
            d[k] = np.asarray(v)
    savez_atomic(path, **d)


def read_labels(path: Path) -> dict[int, list[torch.Tensor]]:
    """The fixed labelings back out of a labels.npz, in the draw order write_labels used."""
    d = np.load(path)
    seeds = [int(x) for x in d["seeds"]]
    return {s: [torch.from_numpy(d[nm][i].astype(np.int64))
                for nm in ("A", "B", "C") if nm in d.files] for i, s in enumerate(seeds)}


def publish_labels(path: Path, fixed: dict[int, list[torch.Tensor]], sha: str,
                   strict: bool) -> None:
    """Write labels.npz, or -- when one is already there -- leave it alone after checking it.

    `strict` (a run that is resuming from a checkpoint) turns a digest mismatch into a refusal:
    the labelling an existing run was trained on is an artifact, never something to overwrite.
    """
    if path.exists():
        try:
            have = labels_sha256(read_labels(path))
        except Exception as e:                                   # unreadable / truncated
            if strict:
                raise SystemExit(f"{path} cannot be read ({e}); refusing to overwrite it")
            have = None
        if have == sha:
            return                                               # already correct, untouched
        if strict:
            raise SystemExit(f"{path} holds another labelling ({have} != {sha}); refusing to "
                             f"overwrite the labels this run was trained on")
    write_labels(path, fixed)


def parse_stop(s: str | None) -> tuple[float, int] | None:
    """'0.999,500' -> (0.999, 500); None/'' -> None."""
    if not s:
        return None
    a, b = s.split(",")
    return float(a), int(b)


def need_correct(acc: float) -> int:
    """How many of the 1200 images an accuracy threshold asks for: 0.99 -> 1188, 0.999 -> 1199
    (0.999 * 1200 = 1198.8).  The hit logic counts images, never float32 accuracies."""
    return int(math.ceil(acc * N_IMAGES - 1e-9))


TRACE_COLS = ("correct", "ce", "margin_med", "n1", "n2", "n3", "sig_med")
HIT_PLUS = 500                     # the post-hit window the rows report (spec §1.2)
POSTFIT_MODES = ("adam", "adam_restore", "freeze", "sgd", "adam_ce",
                 "sgd_all")                    # sgd_postfit_cifar_0923 §1, 追補 1


def postfit_update(P, grads, adam_m, adam_v, post, m_pin, v_pin, sgd_eta, lr, b1, b2, eps,
                   inv_c1, inv_c2) -> None:
    """The step of sgd_postfit_cifar_0923's masked modes, in place.

    Every slot first gets the engine's Adam arithmetic, op for op (so a slot with post False ends
    bit for bit where the plain step puts it).  A slot with post True then keeps, per tensor:
    p -= sgd_eta * g (sgd) or p unchanged (sgd_eta None: freeze), and m, v pinned to m_pin, v_pin.
    torch.where only selects, so it adds no rounding to either side."""
    R = post.shape[0]
    for i, (p, gr, mi, vi) in enumerate(zip(P, grads, adam_m, adam_v)):
        mi.mul_(b1).add_(gr, alpha=1 - b1)
        vi.mul_(b2).addcmul_(gr, gr, value=1 - b2)
        ua = lr * (mi * inv_c1) / ((vi * inv_c2).sqrt() + eps)
        pm = post.view((R,) + (1,) * (p.dim() - 1))
        alt = gr * sgd_eta if sgd_eta is not None else torch.zeros_like(ua)
        p.sub_(torch.where(pm, alt, ua))
        mi.copy_(torch.where(pm, m_pin[i], mi))
        vi.copy_(torch.where(pm, v_pin[i], vi))
# traces written before this commit put the task's FINAL tc on every row; repair them with
#   tc_row = tc_stored - task_steps + step        (task_steps = 30,000, or the row's `steps`)
TRACE_TC_BUG_BEFORE = "253b8386393ae4d6a436c6ebc303536b8bc6e1b8"
PERM_RULE = ("rlc_batch: one torch.randperm(1200) per epoch, drawn at the epoch's start; a task "
             "that ends mid-epoch has already drawn that epoch's permutation, and one ending "
             "exactly on an epoch boundary draws and discards one more (width_replay.py's "
             "convention, lines 314-321).  Every task starts a fresh epoch.")


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
                 cifar: RC.Cifar10 | None = None, c=0.6, beta=0.01, lo=0.005, hi=3.0,
                 schedule: str = "iid", labels: Path | None = None):
    """Rebuild stacked (z1, a1, z2, a2, logits, Y) on each slot's 1200 images from the saved
    snapshots.  With the run's own slot list (provenance "slots", same order) this is the
    forward `evaluate` ran, bit for bit on the same device (S-snap); other layouts differ by
    batched-BLAS round-off (up to ~1e-4 absolute at |z| ~ 50).

    WARNING: the default regenerates the labels by TASK NUMBER off the iid stream.  A run under
    schedule abab/aaaa/abc repeats A/B/C instead, so pass that run's `schedule` and its
    `labels.npz` (provenance "schedule" / <out>/labels.npz) or the labels are simply wrong."""
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
    if schedule != "iid":
        lab = np.load(labels if labels is not None else out / "labels.npz")
        lseeds = list(lab["seeds"])
        name = ("A", "B", "C")[schedule_index(schedule, task)]
        for s, cd in slots:
            Y.append(torch.from_numpy(lab[name][lseeds.index(s)].astype(np.int64)))
        return (*forward(P, X, act, train=False), torch.stack(Y).to(device))
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
    # these files are cumulative and rewritten every task, so an interrupted write would
    # destroy the earlier tasks: build the new one beside it and swap it in
    savez_atomic(path, compressed=True, edges=RC.HIST_EDGES,
                 acc=np.asarray(accs, dtype=np.float64),
                 **{k: np.stack([h[k] for h in hs]) for k in keys}, **extra)


def save_atomic(obj, path: Path) -> None:
    """torch.save through a temporary file in the same directory, then os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def savez_atomic(path: Path, compressed: bool = False, **arrays) -> None:
    """np.savez[_compressed] into a temporary file beside `path`, then os.replace.

    The temporary file is opened as a handle: handed a *name* without a .npz suffix numpy
    appends one, and the replace would then miss the file it just wrote.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        (np.savez_compressed if compressed else np.savez)(fh, **arrays)
    os.replace(tmp, path)


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
        resume: bool = False, graph: bool = True,
        schedule: str = "iid", hit_every: int = 0, keep_ckpts: bool = False,
        stop: tuple[float, int] | None = None, stop_snap: str | None = None,
        labels_fixed: dict[int, list[torch.Tensor]] | None = None,
        restore: dict | None = None, run_id: str = EXPERIMENT,
        extra_prov: dict | None = None, postfit: dict | None = None) -> dict:
    """Train R = len(seeds) * len(conds) runs in lockstep and write their rows/hists/snapshots.

    Check-only hooks (never used by the main run): `perturb[r]` multiplies slot r's W1 by
    (1 + perturb[r]) at init, so repeated (seed, cond) slots become separate realizations of
    one run (S-reuse); `nan_slot` poisons that slot's W1 at init (S-diverge).

    `checkpoint` writes out/ckpt.pt after every task (weights, Adam moments and step, alpha
    statistics, every generator's state, the rows); `resume` continues from it, bit for bit
    (S-resume).  A checkpoint of a finished run can also extend it to more tasks.

    `graph` (cuda only) captures one training step as a CUDA graph and replays it: the same
    kernels on the same static tensors, so the rows are the eager engine's bit for bit (S-graph),
    without a host round trip per step.

    altlabels_cifar_0923 §1 (every default leaves the parent's path bit for bit):
      `schedule`    iid (parent) / abab / aaaa / abc -- see SCHEDULES.
      `hit_every`   >0: a read-only trace eval every that many steps (and once at step 0 of every
                    task, recorded but not eligible for a hit): per slot the correct count out of
                    1200, mean CE, median margin, n1/n2/n3 and sig_med (median over the 100 units
                    of z1's std over images), into trace/<arm>_<cond>_seed<s>.npz.  hit99/hit999
                    (the first step >= hit_every with >= 1188 / >= 1199 correct, -1 = never),
                    acc_at_hit_plus_500 and min_correct_after_hit go into the rows.  It runs
                    outside the CUDA graph, touches no static tensor and mutates nothing (S3).
      `keep_ckpts`  also write ckpts/t<NN>.pt (everything a fork needs) after every task.
      `stop`        (acc, extra): every slot gets its own stop step, `extra` after the first trace
                    eval at or above `acc` (capped at the task's steps); the task ends when every
                    slot has passed its own -- for R = 1 that is width_replay.py's rule exactly.
                    Slots are never frozen.  `steps`/`stop_step`/`tc` go into the rows and `tc`
                    keeps counting across tasks.
      `stop_snap`   a file name template ("fork_t02_A_seed{seed}_stop.npz") written under
                    out/snap/ at each slot's own stop step.
      `labels_fixed`  {seed: [A, B, C]} supplied from outside instead of drawn (fork).
      `restore`     {"path": ckpts/t<NN>.pt} (optionally "slots": [j, ...], one per slot here):
                    start from that checkpoint's weights, Adam moments, tc, per-seed streams and
                    alpha state instead of from the initialization.
      `run_id` / `extra_prov`  what provenance.json records the run as.

    sgd_postfit_cifar_0923 §1 (None leaves every path above bit for bit):
      `postfit`     {"mode", "acc", "extra", "eta"}: every slot gets its own switch step
                    s_sw = `extra` after the first trace eval at or above `acc` (never capped: past
                    the task's steps there is no switch).  Steps 1..s_sw are the engine's Adam in
                    every mode; after step s_sw the slot is post-fit until the task ends:
                      adam          Adam unchanged (the control; only the bookkeeping is added)
                      adam_restore  Adam unchanged, and at the task's end the slot's m and v are
                                    put back to their values right after step s_sw
                      freeze        no update of the slot's parameters; m and v frozen at s_sw
                      sgd           p -= eta * g (the same minibatch gradient, all six tensors,
                                    no momentum, no weight decay); m and v frozen at s_sw
                      adam_ce       Adam unchanged until the first trace eval after s_sw whose
                                    ce_st <= ce_st(s_sw) * exp(-x), then frozen there as in freeze
                                    (never reached: Adam to the task's end)
                      sgd_all       (追補 1) no Adam at all: every step of every task is
                                    p -= eta * g, and m, v stay pinned at zero
                    tc stays one scalar per run and counts every step (1/(1 - b2^tc) is 1.0 in
                    float32 from tc = 16,628 on).  The trace gets one more column, ce_st: the
                    mean over images of log1p(sum_{k != y} exp(z_k - z_y)) in float64 from the
                    float32 logits (no cancellation, never 0 at the float32 CE floor).  The rows
                    get switch_step, pin_step, the post-fit displacement of W1/W2/W3 and the
                    freeze flags."""
    t_start = time.time()
    progress = progress or (lambda m: print(m, flush=True))   # a redirected stdout is block-buffered
    act = make_act(arm, c, beta, lo, hi)
    slots = [(s, cd) for s in seeds for cd in conds]
    R = len(slots)
    if schedule not in SCHEDULES:
        raise SystemExit(f"unknown schedule {schedule!r}; known: {','.join(SCHEDULES)}")
    if hit_every < 0:
        raise SystemExit(f"--hit-every must be >= 0; got {hit_every}")
    stop_need = None
    if stop is not None:
        if not (0.0 < stop[0] <= 1.0) or stop[1] < 0:
            raise SystemExit(f"--stop wants (0 < acc <= 1, extra >= 0); got {stop}")
        if not hit_every:
            raise SystemExit("--stop needs --hit-every (the stop rule reads that eval grid)")
        stop_need = need_correct(stop[0])
    pf_mode = pf_need = pf_extra = pf_eta = pf_x = None
    if postfit is not None:
        pf_mode = postfit["mode"]
        if pf_mode not in POSTFIT_MODES:
            raise SystemExit(f"unknown postfit mode {pf_mode!r}; known: {','.join(POSTFIT_MODES)}")
        if stop is not None or not hit_every:
            raise SystemExit("postfit needs --hit-every and no --stop (every task runs its steps)")
        if postfit["extra"] < 0 or postfit["extra"] % hit_every:
            raise SystemExit(f"postfit extra must be a multiple of hit_every: {postfit['extra']}")
        pf_need, pf_extra = need_correct(postfit["acc"]), int(postfit["extra"])
        if pf_mode in ("sgd", "sgd_all"):
            pf_eta = float(postfit["eta"])
            if not pf_eta > 0:
                raise SystemExit(f"postfit sgd wants eta > 0; got {pf_eta}")
        if pf_mode == "adam_ce":
            pf_x = float(postfit["x"])
            if not pf_x > 0:
                raise SystemExit(f"postfit adam_ce wants x > 0 (e-folds); got {pf_x}")
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
    # the fixed labelings: the schedule's first SCHEDULES[schedule] draws of the seed's own
    # rlc_labels stream, taken up front and never drawn again (iid keeps drawing per task)
    fixed_lab = labels_fixed
    if fixed_lab is None and SCHEDULES[schedule]:
        fixed_lab = {s: [RC.task_labels(g_lab[s]) for _ in range(SCHEDULES[schedule])]
                     for s in useeds}
    # drawn, not published: labels.npz is written only after the checkpoint has been validated,
    # so a refused resume leaves the artifacts of the run it refused untouched
    lab_sha = labels_sha256(fixed_lab) if fixed_lab is not None else None
    ar = torch.arange(R, device=device)[:, None]
    spt = STEPS_PER_EPOCH * epochs
    alive = torch.ones(R, dtype=torch.bool, device=device)
    rows, hists, diverged = [], [[] for _ in range(R)], []
    traces: list[dict[str, np.ndarray]] = [{} for _ in range(R)]
    out.mkdir(parents=True, exist_ok=True)
    if debug is not None:
        debug["init"] = [q.detach().cpu().clone() for q in P]
    step_ms = float("nan")
    meta = {"arm": arm, "seeds": seeds, "conds": conds, "epochs": epochs, "lr": lr, "c": c,
            "beta": beta, "lo": lo, "hi": hi, "perturb": perturb, "nan_slot": nan_slot}
    # every setting that changes the trajectory, so a resume refuses a checkpoint from another
    # one -- device and the execution mode included, since S5b shows cuda and cpu diverge
    meta = {**meta, "schedule": schedule, "stop": stop, "hit_every": hit_every,
            "labels_sha256": lab_sha, "device": device.type,
            "graph": bool(graph and device.type == "cuda" and debug is None)}
    if postfit is not None:           # only then: the checkpoints of earlier runs keep their meta
        meta["postfit"] = {"mode": pf_mode, "acc": postfit["acc"], "extra": pf_extra,
                           "eta": pf_eta, "x": pf_x}
    tc_end: dict[str, int] = {}
    ck = out / "ckpt.pt"
    # one writer per output directory (the launcher can adopt, retry and be started twice)
    lock_fh = open(out / ".lock", "w")
    try:
        fcntl.flock(lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise SystemExit(f"{out} is already being written by another process (.lock is held)")
    lock_fh.write(f"{os.getpid()} {time.strftime('%F %T')}\n")
    lock_fh.flush()
    git_states, resumed, t_first = [git_state()], [], 1
    if restore is not None:
        st = torch.load(restore["path"], map_location="cpu", weights_only=False)
        js = list(restore.get("slots") or range(R))       # ckpt slot for each slot here
        if len(js) != R:
            raise SystemExit(f"restore wants one checkpoint slot per slot: {len(js)} vs R = {R}")
        cks = [(q["seed"], q["cond"]) for q in (st.get("slots") or [])]
        if cks and [cks[j] for j in js] != slots:
            raise SystemExit(f"{restore['path']} slots {[cks[j] for j in js]} are not this run's "
                             f"{slots}")
        idx = torch.as_tensor(js, dtype=torch.long)
        with torch.no_grad():
            for dst, src in ((P, st["P"]), (adam_m, st["m"]), (adam_v, st["v"])):
                for q, v in zip(dst, src):
                    q.copy_(v[idx].to(q.device))
        tc = st["tc"]
        if act.adaptive:
            act.load_state({"V": [v[idx].to(device) for v in st["V"]]})
        if act.stochastic:
            act.gen.set_state(st["rsl_state"])
        for s in useeds:
            g_lab[s].set_state(st["g_lab"][s])
            g_batch[s].set_state(st["g_batch"][s])
        alive = st["alive"][idx].to(device)
        progress(f"[{time.strftime('%T')}] {arm} restored slots {js} of {restore['path']} "
                 f"(task {st['t']}, tc {tc})")
    resuming = bool(resume and ck.exists())
    if resuming:
        st = torch.load(ck, map_location="cpu", weights_only=False)   # generator states must stay on the cpu
        # validate FIRST: nothing in `out` has been written yet, so a refusal is inert
        if st["meta"] != meta:
            raise SystemExit(f"{ck} belongs to another configuration: {st['meta']}")
        if st.get("fixed_lab") is not None:              # the labelling the run was trained on
            ck_lab = {int(s): [y.cpu() for y in st["fixed_lab"][s]] for s in st["fixed_lab"]}
            ck_sha = labels_sha256(ck_lab)
            if ck_sha != lab_sha:
                raise SystemExit(f"{ck} was trained on another labelling ({ck_sha} != {lab_sha})")
            fixed_lab = ck_lab                           # restored, not redrawn
        if keep_ckpts and not (out / "ckpts" / f"t{st['t']:02d}.pt").exists():
            raise SystemExit(f"{ck} is at task {st['t']} but {out}/ckpts/t{st['t']:02d}.pt is "
                             f"missing; the fork points of that task cannot be rebuilt")
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
        tc_end = dict(st.get("tc_end") or {})
        for r, (s, cd) in enumerate(slots):             # histories up to the checkpointed task
            f = out / "hist" / f"{arm}_{cd}_seed{s}.npz"
            n_done = sum(1 for q in rows if q["slot"] == r and "memo_acc" in q)
            if f.exists() and n_done:
                d = np.load(f)
                keys = [k for k in d.files if k not in ("edges", "acc", "th_edges")]
                hists[r] = [{k: d[k][i] for k in keys} for i in range(n_done)]
            f = out / "trace" / f"{arm}_{cd}_seed{s}.npz"
            if f.exists() and hit_every:                # trace rows up to the checkpointed task
                d = np.load(f)
                keep = d["task"] <= st["t"]
                traces[r] = {k: d[k][keep] for k in d.files}
        progress(f"[{time.strftime('%T')}] {arm} resumed from {ck} at task {t_first}")
    # the checkpoint (if any) has been accepted: now the labelling may be published
    if fixed_lab is not None and labels_fixed is None:
        publish_labels(out / "labels.npz", fixed_lab, lab_sha, strict=resuming)
    if not resuming and snapshots:
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

    # ---- sgd_postfit_cifar_0923 §1: which slots are past their switch step, and every slot's
    # parameters and Adam moments right after it (written outside the graph at the switch eval;
    # the graph only reads them).  Only freeze/sgd change the step itself.
    post = torch.zeros(R, dtype=torch.bool, device=device)
    pf_mask = pf_mode in ("freeze", "sgd", "adam_ce", "sgd_all")
    if postfit is not None:
        # m_pin/v_pin/P_pin: the state a masked slot is held at (s_sw, or adam_ce's pin step);
        # P_sw: the parameters at s_sw (the post-fit displacement is measured from there)
        m_pin = [torch.zeros_like(q) for q in P]
        v_pin = [torch.zeros_like(q) for q in P]
        P_pin = [torch.zeros_like(q) for q in P]
        P_sw = [torch.zeros_like(q) for q in P]

    # ---- the trace eval (§1.2): its own tensors, none of them the graph's
    tcols = TRACE_COLS + (("ce_st",) if postfit is not None else ())
    n_ev = (spt // hit_every + 2) if hit_every else 1
    TR = torch.zeros(n_ev, R, len(tcols), dtype=torch.float64, device=device)
    TR_step: list[int] = []

    @torch.no_grad()
    def trace_eval(ts: int) -> torch.Tensor:
        """Read only.  An eval-mode forward over the slot's own 1200 images; it allocates its own
        activations and never writes P, the Adam moments, tc, the activation's state (no
        begin_step, no update), the generators or any tensor the captured graph owns (S3)."""
        z1, a1, z2, a2, logits = forward(P, X, act, train=False)
        correct = (logits.argmax(-1) == Ydev).sum(1)                     # (R,) counts, not floats
        i = len(TR_step)
        TR[i, :, 0] = correct
        TR[i, :, 1] = F.cross_entropy(logits.reshape(-1, N_CLASSES), Ydev.reshape(-1),
                                      reduction="none").view(R, N_IMAGES).mean(1)
        cor = logits.gather(2, Ydev[:, :, None]).squeeze(2)
        oth = logits.clone()
        oth.scatter_(2, Ydev[:, :, None], float("-inf"))
        marg = cor - oth.amax(2)
        # median without a dim is CUDA-deterministic, median(dim) is not (see evaluate())
        TR[i, :, 2] = torch.stack([marg[r].median() for r in range(R)])
        for k, q in enumerate((P[0], P[2], P[4])):
            TR[i, :, 3 + k] = (q.double() ** 2).flatten(1).sum(1)        # n1, n2, n3
        TR[i, :, 6] = torch.stack([z1[r].std(0, unbiased=True).median() for r in range(R)])
        if postfit is not None:
            # ce_st: log1p(sum_{k != y} exp(z_k - z_y)) in float64 from the float32 logits -- the
            # float32 CE above cancels to exactly 0 once 1 - p_y < ~6e-8; this does not
            d = logits.double() - logits.double().gather(2, Ydev[:, :, None])
            d.scatter_(2, Ydev[:, :, None], float("-inf"))
            TR[i, :, 7] = torch.log1p(torch.exp(d).sum(2)).mean(1)
        TR_step.append(ts)
        return correct

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
            if not pf_mask:
                for p, gr, mi, vi in zip(P, grads, adam_m, adam_v):
                    mi.mul_(b1).add_(gr, alpha=1 - b1)
                    vi.mul_(b2).addcmul_(gr, gr, value=1 - b2)
                    p.sub_(lr * (mi * inv_c1) / ((vi * inv_c2).sqrt() + eps))
            else:          # freeze / sgd / adam_ce: the slots past their pin leave Adam
                postfit_update(P, grads, adam_m, adam_v, post, m_pin, v_pin, pf_eta, lr, b1, b2,
                               eps, inv_c1, inv_c2)
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
        if fixed_lab is None:
            lab = {s: RC.task_labels(g_lab[s]) for s in useeds}           # once per seed per task
        else:
            k = schedule_index(schedule, t)
            lab = {s: fixed_lab[s][k] for s in useeds}
        Y = torch.stack([lab[s] for s, cd in slots]).to(device)           # (R, 1200)
        Ydev.copy_(Y)
        if debug is not None:
            debug.setdefault("labels", []).append(Y.cpu().clone())
        acc_sum.zero_()
        bad_step.fill_(-1)
        step_t.zero_()
        TR_step.clear()
        ts, done = 0, False
        stop_at: list[int | None] = [None] * R          # per slot; the task ends when all passed
        snapped = [False] * R
        sw_at: list[int | None] = [None] * R            # postfit: each slot's own switch step
        sw_done = [False] * R
        pinned, pin_at, ce0 = [False] * R, [-1] * R, [float("nan")] * R
        post.zero_()
        if pf_mode == "sgd_all":          # SGD from the task's first step, moments held at 0
            post.fill_(True)
            pinned, pin_at = [True] * R, [0] * R
        live = alive.cpu().tolist()                     # a diverged slot never reaches a hit
        t0 = time.time()
        if hit_every:
            trace_eval(0)          # the state each task starts from; never eligible for a hit
        for e in range(epochs):
            if done:
                break
            order = {s: torch.randperm(N_IMAGES, generator=g_batch[s]) for s in useeds}
            ORD = torch.stack([order[s] for s, cd in slots]).to(device)  # (R, 1200)
            if debug is not None:
                debug.setdefault("orders", []).append(ORD.cpu().clone())
            for j in range(STEPS_PER_EPOCH):
                # the stop check sits at the top of the minibatch loop, after the epoch's
                # permutation has been drawn (PERM_RULE; width_replay.py lines 314-321)
                if stop is not None and any(live) and all(
                        stop_at[r] is not None and ts >= stop_at[r]
                        for r in range(R) if live[r]):
                    done = True
                    break
                static_idx.copy_(ORD[:, j * BATCH:(j + 1) * BATCH])
                tc += 1
                ts += 1
                inv_c1.fill_(1.0 / (1 - b1 ** tc))
                inv_c2.fill_(1.0 / (1 - b2 ** tc))
                act.begin_step(R, BATCH, device)
                if cg is not None:
                    cg.replay()
                else:
                    step()
                if debug is not None:
                    debug.setdefault("online", []).append(last_hit.cpu().clone())
                if hit_every and ts % hit_every == 0:
                    correct = trace_eval(ts)
                    if stop is not None:
                        cc = correct.cpu().tolist()
                        for r in range(R):
                            if stop_at[r] is None and cc[r] >= stop_need:
                                stop_at[r] = min(ts + stop[1], spt)
                            if stop_at[r] == ts and not snapped[r] and stop_snap:
                                # the slot's own stop point; it keeps training with the bundle
                                s_, cd_ = slots[r]
                                write_snapshot(out / "snap" / stop_snap.format(seed=s_, cond=cd_,
                                                                              slot=r), P, act, r)
                                snapped[r] = True
                    if postfit is not None and any(
                            live[r] and not pinned[r] and (sw_at[r] is None or sw_at[r] <= spt)
                            for r in range(R)):
                        cc = correct.cpu().tolist()
                        ce_now = TR[len(TR_step) - 1, :, 7].cpu().tolist()
                        for r in range(R):
                            if sw_at[r] is None and cc[r] >= pf_need:
                                sw_at[r] = ts + pf_extra
                            if sw_at[r] == ts and not sw_done[r]:
                                # step s_sw is done: every later step of this slot is post-fit
                                # (the stop chain's last step is this same s_sw)
                                with torch.no_grad():
                                    for i in range(6):
                                        P_sw[i][r].copy_(P[i][r])
                                sw_done[r] = True
                                ce0[r] = ce_now[r]
                            if sw_done[r] and not pinned[r] and (
                                    pf_mode != "adam_ce" or ce_now[r] <= ce0[r] * math.exp(-pf_x)):
                                # the pin: the state right after this step, which the masked
                                # modes hold from the next step on (at s_sw except adam_ce)
                                with torch.no_grad():
                                    for i in range(6):
                                        m_pin[i][r].copy_(adam_m[i][r])
                                        v_pin[i][r].copy_(adam_v[i][r])
                                        P_pin[i][r].copy_(P[i][r])
                                if pf_mask:
                                    post[r] = True
                                pinned[r], pin_at[r] = True, ts
        if device.type == "cuda":
            torch.cuda.synchronize()
        step_ms = 1e3 * (time.time() - t0) / max(ts, 1)

        # ---- postfit bookkeeping (sgd_postfit_cifar_0923 §3, §5 C4): measured equalities, so
        # each flag can fail (a mode that moves the slot after s_sw must show False)
        pf_rows: list[dict] = [{} for _ in range(R)]
        if postfit is not None:
            with torch.no_grad():
                nan = float("nan")
                for r in range(R):
                    if pf_mode == "sgd_all":
                        mv0 = int(all(torch.equal(adam_m[i][r], m_pin[i][r]) and
                                      torch.equal(adam_v[i][r], v_pin[i][r]) for i in range(6)))
                        pf_rows[r] = {"switch_step": -1, "pin_step": 0, "pf_disp_l1": nan,
                                      "pf_disp_l2": nan, "pf_disp_l3": nan, "pf_P_same": -1,
                                      "pf_mv_same": mv0, "pf_mv_restored": mv0}
                        continue
                    if not sw_done[r]:
                        pf_rows[r] = {"switch_step": -1, "pin_step": -1, "pf_disp_l1": nan,
                                      "pf_disp_l2": nan, "pf_disp_l3": nan,
                                      "pf_P_same": -1, "pf_mv_same": -1, "pf_mv_restored": -1}
                        continue
                    q = {"switch_step": sw_at[r], "pin_step": pin_at[r]}
                    for k, i in ((1, 0), (2, 2), (3, 4)):
                        q[f"pf_disp_l{k}"] = float(((P[i][r].double() - P_sw[i][r].double()) ** 2)
                                                   .sum())
                    if not pinned[r]:                 # adam_ce that never reached its target
                        q.update({"pf_P_same": -1, "pf_mv_same": -1, "pf_mv_restored": -1})
                        pf_rows[r] = q
                        continue
                    q["pf_P_same"] = int(all(torch.equal(P[i][r], P_pin[i][r]) for i in range(6)))
                    q["pf_mv_same"] = int(all(torch.equal(adam_m[i][r], m_pin[i][r]) and
                                              torch.equal(adam_v[i][r], v_pin[i][r])
                                              for i in range(6)))
                    if pf_mode == "adam_restore":
                        for i in range(6):
                            adam_m[i][r].copy_(m_pin[i][r])
                            adam_v[i][r].copy_(v_pin[i][r])
                    q["pf_mv_restored"] = int(all(torch.equal(adam_m[i][r], m_pin[i][r]) and
                                                  torch.equal(adam_v[i][r], v_pin[i][r])
                                                  for i in range(6)))
                    pf_rows[r] = q
            post.zero_()

        # ---- the trace: counts in, hit99/hit999/stop bookkeeping out (§1.2)
        # .copy() is load-bearing: on cpu, .cpu() is a no-op and .numpy() SHARES TR's memory, so
        # the rows this task hands to `traces` would be overwritten by the next task's evals
        tr = TR[:len(TR_step)].cpu().numpy().copy()              # (E, R, len(tcols))
        ev = np.asarray(TR_step, dtype=np.int64)                 # (E,)
        h99, h999 = [-1] * R, [-1] * R
        hplus, hseen, hmin = [-1] * R, [0] * R, [-1] * R
        if hit_every:
            elig = ev > 0                       # step 0 is recorded but never a hit
            cnt = tr[:, :, 0].astype(np.int64)
            for r in range(R):
                for need, dst in ((need_correct(0.99), h99), (need_correct(0.999), h999)):
                    w = np.nonzero(elig & (cnt[:, r] >= need))[0]
                    dst[r] = int(ev[w[0]]) if len(w) else -1
                if h999[r] >= 0:
                    # the count at exactly hit999 + HIT_PLUS, and -1 with hseen = 0 when the
                    # task ended (stop rule or the 30,000 cap) before that step was evaluated
                    w = np.nonzero(ev == h999[r] + HIT_PLUS)[0]
                    if len(w):
                        hplus[r], hseen[r] = int(cnt[w[0], r]), 1
                    after = cnt[ev > h999[r], r]        # window: after the hit, to the task's end
                    hmin[r] = int(after.min()) if len(after) else int(cnt[ev == h999[r], r][0])
            for r in range(R):
                s_, cd_ = slots[r]
                old = traces[r]
                # tc is the global clock AT THAT ROW: the task's starting clock plus the step.
                # For a fork that starting clock is the parent checkpoint's tc (restore set it).
                new = {"task": np.full(len(ev), t, dtype=np.int64), "tc": (tc - ts) + ev,
                       "step": ev, **{k: tr[:, r, i] for i, k in enumerate(tcols)}}
                new["correct"] = new["correct"].astype(np.int64)
                traces[r] = {k: (np.concatenate([old[k], new[k]]) if old else new[k]) for k in new}
        with torch.no_grad():
            finite = torch.stack([torch.isfinite(q).flatten(1).all(1) for q in P]).all(0)
            newly = alive & ((bad_step >= 0) | ~finite)
            for r in torch.nonzero(newly).flatten().tolist():
                s, cd = slots[r]
                diverged.append({"diverged": True, "task": t,
                                 "step": tc - ts + max(int(bad_step[r]), 0),
                                 "seed": s, "cond": cd, "arm": arm})
                rows.append({"arm": arm, "cond": cd, "seed": s, "slot": r, "lr": lr, "task": t,
                             "iv": "none", "acc": float("nan")})
            alive &= ~newly
            m, hs, zs = evaluate(P, X, Y, act, torch.nonzero(alive).flatten().tolist())
        for r in torch.nonzero(alive).flatten().tolist():
            s, cd = slots[r]
            extra = {}                       # only when the feature is on: the default csv is the parent's
            if stop is not None:
                extra["steps"] = ts
                extra["stop_step"] = stop_at[r] if stop_at[r] is not None else -1
                extra["tc"] = tc
            if hit_every:
                extra["hit99"], extra["hit999"] = h99[r], h999[r]
                extra["correct_at_hit_plus_500"] = hplus[r]        # -1 unless it was observed
                extra["hit_plus_500_observed"] = hseen[r]
                extra["min_correct_after_hit"] = hmin[r]           # window: (hit999, task end]
            extra.update(pf_rows[r])                               # postfit only, else empty
            rows.append({"arm": arm, "cond": cd, "seed": s, "slot": r, "lr": lr, "task": t,
                         "iv": "none", "online_acc": float(acc_sum[r]) / ts,
                         "memo_acc": m[r]["acc"], **m[r], **extra})
            hists[r].append(hs[r])
            if snapshots:
                write_snapshot(snapshot_path(out, arm, cd, s, t), P, act, r, zs)
        rows.sort(key=lambda q: (q["slot"], q["task"]))
        H.write_csv(out / "per_task.csv", rows)
        tc_end[str(t)] = tc
        for r in range(R):
            if hists[r] and snapshots:
                s, cd = slots[r]
                write_hist(out / "hist" / f"{arm}_{cd}_seed{s}.npz", hists[r],
                           [q["memo_acc"] for q in rows if q["slot"] == r and "memo_acc" in q])
            if traces[r]:
                s, cd = slots[r]
                (out / "trace").mkdir(parents=True, exist_ok=True)
                # cumulative: build beside it and swap, never overwrite in place
                savez_atomic(out / "trace" / f"{arm}_{cd}_seed{s}.npz", compressed=True,
                             **traces[r])
        if checkpoint or keep_ckpts:
            # an independent payload: detached cpu clones, so nothing the next task mutates
            # can reach into a file that is being written (S8)
            cpu = lambda qs: [q.detach().cpu().clone() for q in qs]
            st = {"t": t, "meta": meta, "P": cpu(P), "m": cpu(adam_m), "v": cpu(adam_v),
                  "tc": tc, "V": cpu(act.V) if act.adaptive else None,
                  "rsl_state": act.gen.get_state().clone() if act.stochastic else None,
                  "g_lab": {s: g_lab[s].get_state().clone() for s in useeds},
                  "g_batch": {s: g_batch[s].get_state().clone() for s in useeds},
                  "alive": alive.detach().cpu().clone(), "rows": rows, "diverged": diverged,
                  "step_ms": step_ms,
                  "git_states": git_states, "resumed": resumed, "tc_end": tc_end,
                  "slots": [{"seed": s, "cond": cd} for s, cd in slots], "schedule": schedule,
                  "fixed_lab": ({s: [y.detach().cpu().clone() for y in fixed_lab[s]]
                                 for s in fixed_lab} if fixed_lab is not None else None)}
            # the per-task checkpoint is published FIRST: ckpt.pt is what a resume trusts, and
            # it must never point past a task whose fork checkpoint is missing
            if keep_ckpts:                        # §1.3: one per task, everything a fork needs
                save_atomic(st, out / "ckpts" / f"t{t:02d}.pt")
            if checkpoint:
                save_atomic(st, ck)
        on = acc_sum[alive] / ts
        memo = torch.tensor([m[r]["acc"] for r in range(R) if alive[r]])
        el = time.time() - t_start
        progress(f"[{time.strftime('%T')}] {arm} task {t:2d}/{n_tasks} alive {int(alive.sum())}/{R} "
                 f"online {float(on.mean()) if len(on) else float('nan'):.3f} "
                 f"(min {float(on.min()) if len(on) else float('nan'):.3f}) "
                 f"memo min {float(memo.min()) if len(memo) else float('nan'):.3f} "
                 f"{step_ms:.2f} ms/step  {el/60:.0f} min, "
                 f"ETA {el/(t-t_first+1)*(n_tasks-t)/60:.0f} min"
                 + (f"  steps {ts}" if stop is not None else "")
                 + (f"  hit999 med {int(np.median([q for q in h999 if q >= 0]))}"
                    if hit_every and any(q >= 0 for q in h999) else "")
                 + (f"  switched {sum(sw_done)}/{R} pinned {sum(pinned)}/{R}"
                    if postfit is not None else ""))

    prov = {"run_id": run_id, **git_states[0], "git_states": git_states, "resumed_at_task": resumed,
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
            "check_hooks": {"perturb": perturb, "nan_slot": nan_slot, "snapshots": snapshots},
            # altlabels_cifar_0923 §1 (the parent's path is schedule iid, no hit/ckpts/stop)
            "parent": EXPERIMENT if run_id != EXPERIMENT else None,
            "schedule": schedule, "hit_every": hit_every, "keep_ckpts": keep_ckpts,
            "stop": list(stop) if stop is not None else None,
            "stop_need_correct": stop_need, "stop_snap": stop_snap,
            "hit_need_correct": {"hit99": need_correct(0.99), "hit999": need_correct(0.999),
                                 "of": N_IMAGES},
            "labels_sha256": lab_sha,
            "labels_source": ("external" if labels_fixed is not None else
                              ("labels.npz" if fixed_lab is not None else "rlc_labels per task")),
            "labels_stats": labels_stats(fixed_lab) if fixed_lab is not None else None,
            "perm_consumption_rule": PERM_RULE,
            "tc_at_task_end": tc_end,
            "trace": ({"path": "trace/<arm>_<cond>_seed<seed>.npz", "every": hit_every,
                       "cols": ["task", "tc", "step"] + list(tcols),
                       "step0_row": "one row at step 0 of every task, not eligible for a hit",
                       "tc": "the global clock at that row = the task's starting tc + step",
                       "hit_plus": HIT_PLUS,
                       "post_hit_fields": {
                           "correct_at_hit_plus_500": "the count at exactly hit999 + 500, or -1",
                           "hit_plus_500_observed": "0 when the task ended before that step",
                           "min_correct_after_hit": "min over the evals in (hit999, task end]"},
                       "median": "torch.median (lower of the two middle values) over the 100 "
                                 "units / 1200 images, as the parent engine's evaluate() uses",
                       "trace_tc_bug_before_commit": TRACE_TC_BUG_BEFORE,
                       "trace_tc_repair": "tc_row = tc_stored - task_steps + step "
                                          "(task_steps = 30000, or the row's `steps`)"}
                      if hit_every else None),
            "restore": ({"path": str(restore["path"]),
                         "slots": list(restore.get("slots") or range(R)),
                         "sha256": hashlib.sha256(Path(restore["path"]).read_bytes()).hexdigest()}
                        if restore is not None else None),
            # sgd_postfit_cifar_0923 §1 (absent -> None: the parent's path)
            "postfit": ({**meta["postfit"],
                         "switch": "s_sw = the first trace eval (step >= hit_every) with >= "
                                   f"{pf_need} correct, + extra; steps 1..s_sw are Adam in every "
                                   "mode, later steps follow the mode; no switch when s_sw > "
                                   "the task's steps",
                         "tc": "one scalar per run, counts every step in every mode",
                         "rows": {"switch_step": "s_sw, or -1 when the slot did not switch",
                                  "pin_step": "the eval step whose state the masked modes hold "
                                              "(s_sw, or adam_ce's CE target; -1 = never)",
                                  "pf_disp_l<k>": "||W_k(task end) - W_k(s_sw)||^2 (float64)",
                                  "pf_P_same": "all six tensors at the task end == at the pin",
                                  "pf_mv_same": "m and v at the task end == right after the pin "
                                                "(before any restore)",
                                  "pf_mv_restored": "the same after adam_restore's reset "
                                                    "(== pf_mv_same in the other modes)"},
                         "ce_st": "trace column: mean_i log1p(sum_{k != y} exp(z_k - z_y)), "
                                  "float64 from the float32 logits"}
                        if postfit is not None else None),
            **(extra_prov or {})}
    (out / "provenance.json").write_text(json.dumps(prov, indent=2))
    fcntl.flock(lock_fh, fcntl.LOCK_UN)
    lock_fh.close()
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
    ap.add_argument("--conds", default="raw,std")
    ap.add_argument("--tasks", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--c", type=float, default=0.6)
    ap.add_argument("--beta", type=float, default=0.01)
    ap.add_argument("--alpha-lo", type=float, default=0.005)
    ap.add_argument("--alpha-hi", type=float, default=3.0)
    ap.add_argument("--out", default=None, help="default results/<experiment>/<arm>")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--threads", type=int, default=2, help="torch cpu threads (eval: eff_rank, histograms)")
    ap.add_argument("--no-resume", action="store_true", help="ignore an existing out/ckpt.pt")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    device = H.setup(a.device)
    out = Path(a.out) if a.out else OUT_ROOT / a.arm
    run(a.arm, parse_ints(a.seeds), a.conds.split(","), a.tasks, a.epochs, device, out,
        c=a.c, beta=a.beta, lo=a.alpha_lo, hi=a.alpha_hi, checkpoint=True, resume=not a.no_resume)


if __name__ == "__main__":
    main()
