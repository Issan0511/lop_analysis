#!/usr/bin/env python3
"""Why adaptive Snake loses to fixed Snake on the RL-CIFAR CNN (sna_cnn_cause_1009).

    python3 src/sna_cnn_cause_1009.py --arms SNA,SNAc3 --seeds 10-19 --out results/sna_cnn_cause_1009/b1

The protocol, the net, the init rule, the image subset and the label / batch streams
are the host's (`rlcifar_cnn_0908`), imported unchanged.  What is new is the engine:
R runs (arm x seed) are trained side by side in one process -- grouped conv for the two
conv layers, bmm for the fc layers, one Adam per run -- and one epoch of training steps
is captured as a CUDA graph.  Runs never exchange information: every run's loss depends
only on its own parameters and inputs (S-indep), so summing the R losses hands every run
exactly its own gradient.

An arm sets, per activation site (c1 c2 f1 f2), either an adaptive alpha
alpha_j = clip(c / W_j, lo, hi) with its own c (W_j = sqrt of the EMA of var z_j, the
host's rule) or a fixed alpha, plus two options:
  frz=k : at the end of task k the current alpha_j of every channel / unit is frozen
  pp    : conv W_j from the WITHIN-position variance (variance over the batch at each
          position, averaged over the positions) instead of the host's pooled variance
          over (batch, H, W), which also contains the spread of the position means
  fixconv: the two conv layers never move from their init (their Adam step is masked to 0),
          so the fc layers learn on fixed random conv features
  fixfc : the three fc layers never move (their Adam step is masked to 0); used by the forks
          to measure what the conv layers alone can fit
  fcamp<a>: at the two fc sites phi = z + a sin^2(alpha z)/alpha, phi' = 1 + a sin(2 alpha z)
          >= 1 - a: the same gate pattern with its flat point (phi' = 0) lifted to 1 - a
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import pmnist_0905 as H                 # host of the host; not touched
from src import pmnist_rlcifar_0907 as RC        # CIFAR data layer; not touched
from src import rlcifar_cnn_0908 as CN           # the CNN box; not touched

EXPERIMENT = "sna_cnn_cause_1009"
N_IMAGES, BATCH, SPE, N_CLASSES = CN.N_IMAGES, CN.BATCH, CN.STEPS_PER_EPOCH, CN.N_CLASSES
CH, HID, KER, PAD, POOL = CN.CHANNELS, CN.HIDDEN, CN.KERNEL, CN.PAD, CN.POOL
SITES = CN.SITES                                  # c1 c2 f1 f2
WIDTHS = CN.WIDTHS                                # 16 16 100 100
IS_CONV = CN.IS_CONV
WTAGS = CN.WEIGHT_TAGS                            # c1 c2 f1 f2 f3
CKPT_TASKS = (1, 2, 3, 5, 10, 20, 30, 40, 50)
HALF_PI = 0.5 * math.pi

# --------------------------------------------------------------------------
# arms
# --------------------------------------------------------------------------
# c = adaptive c per site, a = fixed alpha per site; frz = task after which alpha freezes;
# pp = within-position variance for the conv sites.
ARMS = {
    "SNA":     dict(c=(0.6, 0.6, 0.6, 0.6)),            # host SNA
    "SNAc3":   dict(c=(3.0, 3.0, 3.0, 3.0)),            # swish_battle_0917 SNAc3
    "SN3":     dict(a=(3.0, 3.0, 3.0, 3.0)),            # host SN3
    "SN06":    dict(a=(0.6, 0.6, 0.6, 0.6)),            # host SN06
    "CV06FC3": dict(c=(0.6, 0.6, 3.0, 3.0)),            # c = 0.6 only at the conv sites
    "CV3FC06": dict(c=(3.0, 3.0, 0.6, 0.6)),            # c = 0.6 only at the fc sites
    "SNAfrz1": dict(c=(0.6, 0.6, 0.6, 0.6), frz=1),     # SNA, alpha frozen after task 1
    "SNApp":   dict(c=(0.6, 0.6, 0.6, 0.6), pp=True),   # SNA, within-position conv W
    # batch B (spec addendum 1): conv at c = 0.6 throughout, fc split by site
    "F1ONLY":  dict(c=(0.6, 0.6, 0.6, 3.0)),            # c = 0.6 at f1, 3 at f2
    "F2ONLY":  dict(c=(0.6, 0.6, 3.0, 0.6)),            # c = 3 at f1, 0.6 at f2
}


def parse_arm(name: str) -> dict:
    """Registry name, or a site-wise spec `S:<x>-<x>-<x>-<x>` with x = c<val> | a<val>,
    e.g. `S:c0.6-c3-c3-c3`.  Options are appended with `+frz<k>` / `+pp` / `+fixconv` / `+fixfc` /
    `+fcamp<a>`."""
    base, *opts = name.split("+")
    if base in ARMS:
        d = dict(ARMS[base])
    elif base.startswith("S:"):
        parts = base[2:].split("-")
        if len(parts) != 4 or any(p[0] not in "ca" for p in parts):
            raise SystemExit(f"bad site spec {name!r}")
        d = {"site": tuple((p[0], float(p[1:])) for p in parts)}
    else:
        raise SystemExit(f"unknown arm {name!r}; known: {','.join(ARMS)} or S:...")
    for o in opts:
        if o.startswith("frz"):
            d["frz"] = int(o[3:])
        elif o == "pp":
            d["pp"] = True
        elif o == "fixconv":
            d["fixconv"] = True
        elif o == "fixfc":
            d["fixfc"] = True
        elif o.startswith("fcamp"):
            d["fcamp"] = float(o[5:])
        else:
            raise SystemExit(f"unknown option {o!r} in {name!r}")
    if "site" not in d:
        d["site"] = (tuple(("c", v) for v in d["c"]) if "c" in d
                     else tuple(("a", v) for v in d["a"]))
    d.setdefault("frz", None)
    d.setdefault("pp", False)
    d.setdefault("fixconv", False)
    d.setdefault("fixfc", False)
    d.setdefault("fcamp", 1.0)
    return d


# --------------------------------------------------------------------------
# the bundled activation
# --------------------------------------------------------------------------

class BundleSnake:
    """Snake with a per-(run, channel/unit) alpha, for R runs at once.

    alpha = where(ada, clip(c / sqrt(V), lo, hi), fixA); phi = z + sin(alpha z)^2 / alpha
    (the host's op order, `* a.reciprocal()`).  V is the host's EMA of the batch variance
    (beta = 0.01), kept for every run -- fixed-alpha runs keep it only as a record.
    """

    def __init__(self, specs: list[dict], device, lo=0.005, hi=3.0, beta=0.01):
        R = len(specs)
        self.R, self.lo, self.hi, self.beta = R, lo, hi, beta
        self.V = [torch.ones(R, w, device=device) for w in WIDTHS]
        self.ada = [torch.zeros(R, w, dtype=torch.bool, device=device) for w in WIDTHS]
        self.cval = [torch.ones(R, 1, device=device) for _ in WIDTHS]
        self.fixA = [torch.ones(R, w, device=device) for w in WIDTHS]
        self.pp = torch.zeros(R, 1, dtype=torch.bool, device=device)
        for r, s in enumerate(specs):
            for l, (kind, val) in enumerate(s["site"]):
                if kind == "c":
                    self.ada[l][r] = True
                    self.cval[l][r] = val
                else:
                    self.fixA[l][r] = val
            self.pp[r] = bool(s["pp"])
        self.any_pp = bool(self.pp.any())
        # Snake amplitude per run at the fc sites; only built when some run uses one, so that
        # every other bundle computes exactly the plain Snake
        self.amp = None
        if any(s["fcamp"] != 1.0 for s in specs):
            self.amp = torch.tensor([[s["fcamp"]] for s in specs], device=device)   # (R, 1)

    def alpha(self, l: int) -> torch.Tensor:                          # (R, n)
        ada = (self.cval[l] / self.V[l].sqrt()).clamp(self.lo, self.hi)
        return torch.where(self.ada[l], ada, self.fixA[l])

    def _shape(self, l: int, a: torch.Tensor) -> torch.Tensor:
        return a.reshape(1, -1, 1, 1) if IS_CONV[l] else a[:, None, :]

    def phi(self, z: torch.Tensor, l: int) -> torch.Tensor:
        a = self._shape(l, self.alpha(l))
        if self.amp is not None and not IS_CONV[l]:
            return z + self.amp[:, None, :] * (torch.sin(a * z) ** 2 * a.reciprocal())
        return z + torch.sin(a * z) ** 2 * a.reciprocal()

    def dphi(self, z: torch.Tensor, l: int) -> torch.Tensor:
        if self.amp is not None and not IS_CONV[l]:
            return 1.0 + self.amp[:, None, :] * torch.sin(2.0 * self._shape(l, self.alpha(l)) * z)
        return 1.0 + torch.sin(2.0 * self._shape(l, self.alpha(l)) * z)

    @torch.no_grad()
    def batch_var(self, z: torch.Tensor, l: int) -> torch.Tensor:
        """(R, n) variance of this batch's preactivation, the quantity V tracks."""
        R = self.R
        if not IS_CONV[l]:
            return z.var(1, unbiased=False)                          # (R, B, n) -> (R, n)
        pooled = z.var((0, 2, 3), unbiased=False).view(R, -1)       # host: over (N, H, W)
        if not self.any_pp:
            return pooled
        within = z.var(0, unbiased=False).mean((1, 2)).view(R, -1)  # over N, then mean over (H, W)
        return torch.where(self.pp, within, pooled)

    @torch.no_grad()
    def update(self, zs) -> None:
        for l, z in enumerate(zs):
            self.V[l].mul_(1 - self.beta).add_(self.beta * self.batch_var(z, l))

    @torch.no_grad()
    def freeze(self, runs: list[int]) -> None:
        """alpha of these runs fixed at its current value, channel by channel."""
        for l in range(4):
            a = self.alpha(l)
            for r in runs:
                self.fixA[l][r] = a[r]
                self.ada[l][r] = False

    def state(self) -> dict:
        return {k: [t.clone() for t in getattr(self, k)] for k in ("V", "ada", "cval", "fixA")}

    def load_state(self, st: dict) -> None:
        for k in ("V", "ada", "cval", "fixA"):
            for dst, src in zip(getattr(self, k), st[k]):
                dst.copy_(src)                       # in place: a captured graph holds these


# --------------------------------------------------------------------------
# net
# --------------------------------------------------------------------------

def stack_init(seeds: list[int], device) -> list[torch.Tensor]:
    """The host init (`CN.init_params`, cpu stream) per run, stacked on a leading R axis."""
    per = [CN.init_params(s, torch.device("cpu")) for s in seeds]
    return [torch.stack([p[i].detach() for p in per]).to(device).requires_grad_(True)
            for i in range(10)]


def forward(P, xb: torch.Tensor, act: BundleSnake):
    """xb (R, B, 3, 32, 32) -> (z1, a1, z2, a2, z3, a3, z4, a4, logits).

    Conv sites are (B, R*C, H, W) with run r on channels [r*C, (r+1)*C); fc sites are
    (R, B, n).  conv -> phi -> pool, as the host.
    """
    Wc1, bc1, Wc2, bc2, Wf1, bf1, Wf2, bf2, Wf3, bf3 = P
    R, B = xb.shape[:2]
    x = xb.transpose(0, 1).reshape(B, R * 3, 32, 32)
    z1 = F.conv2d(x, Wc1.reshape(R * CH, 3, KER, KER), bc1.reshape(-1), padding=PAD, groups=R)
    a1 = act.phi(z1, 0)
    z2 = F.conv2d(F.max_pool2d(a1, POOL, POOL), Wc2.reshape(R * CH, CH, KER, KER),
                  bc2.reshape(-1), padding=PAD, groups=R)
    a2 = act.phi(z2, 1)
    h = F.max_pool2d(a2, POOL, POOL).reshape(B, R, CN.FLAT).transpose(0, 1)   # (R, B, 1024)
    z3 = torch.baddbmm(bf1[:, None, :], h, Wf1.transpose(1, 2))
    a3 = act.phi(z3, 2)
    z4 = torch.baddbmm(bf2[:, None, :], a3, Wf2.transpose(1, 2))
    a4 = act.phi(z4, 3)
    logits = torch.baddbmm(bf3[:, None, :], a4, Wf3.transpose(1, 2))
    return z1, a1, z2, a2, z3, a3, z4, a4, logits


# --------------------------------------------------------------------------
# per-task evaluation
# --------------------------------------------------------------------------

def _median_rows(x: torch.Tensor) -> torch.Tensor:
    # median without dim is deterministic on CUDA; per-row loop keeps it so
    return torch.stack([x[r].median() for r in range(x.shape[0])])


@torch.no_grad()
def evaluate(P, X, Y, act: BundleSnake, chunk: int = 20) -> list[dict]:
    """The host's `evaluate_cnn` quantities for every run, on its own 1200 images, plus the
    seat 2*alpha*zbar and the readout scale.  Channel statistics are accumulated over
    chunks in float64 (conv: over N, H, W)."""
    R = X.shape[0]
    dev = X.device
    n = {l: 0 for l in range(4)}
    S1 = [torch.zeros(R, w, dtype=torch.float64, device=dev) for w in WIDTHS]
    S2 = [torch.zeros(R, w, dtype=torch.float64, device=dev) for w in WIDTHS]
    SD = [torch.zeros(R, w, dtype=torch.float64, device=dev) for w in WIDTHS]
    SP = [torch.zeros(R, CH, dtype=torch.float64, device=dev) for _ in range(2)]
    G = [torch.zeros(R, w, w, dtype=torch.float64, device=dev) for w in WIDTHS]
    correct = torch.zeros(R, dtype=torch.float64, device=dev)
    ce = torch.zeros(R, dtype=torch.float64, device=dev)
    conf = torch.zeros(R, dtype=torch.float64, device=dev)
    lsd = torch.zeros(R, dtype=torch.float64, device=dev)
    for i0 in range(0, N_IMAGES, chunk):
        xb = X[:, i0:i0 + chunk]
        yb = Y[:, i0:i0 + chunk]
        b = xb.shape[1]
        out = forward(P, xb, act)
        logits = out[8]
        correct += (logits.argmax(-1) == yb).sum(1)
        ce += F.cross_entropy(logits.reshape(-1, N_CLASSES), yb.reshape(-1),
                              reduction="none").view(R, b).double().sum(1)
        conf += logits.softmax(-1).amax(-1).double().sum(1)
        lsd += logits.double().std(-1).sum(1)
        for l in range(4):
            z, a = out[2 * l], out[2 * l + 1]
            d = act.dphi(z, l)
            if IS_CONV[l]:
                zr = z.reshape(b, R, CH, -1).double()             # (b, R, C, HW)
                S1[l] += zr.sum((0, 3)); S2[l] += (zr * zr).sum((0, 3))
                SD[l] += d.reshape(b, R, CH, -1).double().sum((0, 3))
                n[l] += b * zr.shape[3]
                _, idx = F.max_pool2d(a, POOL, POOL, return_indices=True)
                dp = d.flatten(2).gather(2, idx.flatten(2))       # (b, R*C, HW/4)
                SP[l] += dp.reshape(b, R, CH, -1).double().sum((0, 3))
                ar = a.reshape(b, R, CH, -1).permute(1, 0, 3, 2).reshape(R, -1, CH).double()
            else:
                zr = z.double()                                   # (R, b, n)
                S1[l] += zr.sum(1); S2[l] += (zr * zr).sum(1)
                SD[l] += d.double().sum(1)
                n[l] += b
                ar = a.double()
            G[l] += ar.transpose(1, 2) @ ar
    rows = [dict() for _ in range(R)]
    for r in range(R):
        rows[r]["memo_acc"] = float(correct[r]) / N_IMAGES
        rows[r]["fit_ce"] = float(ce[r]) / N_IMAGES
        rows[r]["conf"] = float(conf[r]) / N_IMAGES
        rows[r]["logit_sd"] = float(lsd[r]) / N_IMAGES
    for l, tag in enumerate(SITES):
        m = S1[l] / n[l]
        var = (S2[l] / n[l] - m * m).clamp_min(0) * n[l] / (n[l] - 1)
        sd = var.sqrt()
        mob = SD[l] / n[l]
        alpha = act.alpha(l).double()
        W = act.V[l].double().sqrt()
        seat = 2 * alpha * m
        ev = torch.linalg.eigvalsh(G[l].cpu()).clamp_min(0.0)
        for r in range(R):
            q = rows[r]
            q[f"zbar_{tag}"] = float(m[r].median())
            q[f"zsd_{tag}"] = float(sd[r].median())
            q[f"mob_{tag}"] = float(mob[r].median())
            q[f"seat_{tag}"] = float(seat[r].median())
            q[f"past_valley_{tag}"] = float((seat[r] < -HALF_PI).double().mean())
            q[f"alpha_med_{tag}"] = float(alpha[r].median())
            q[f"two_alpha_W_med_{tag}"] = float((2 * alpha[r] * W[r]).median())
            q[f"two_alpha_sd_med_{tag}"] = float((2 * alpha[r] * sd[r]).median())
            s = ev[r].sqrt(); p = s / s.sum()
            q[f"eff_rank_{tag}"] = float(torch.exp(-(p * p.clamp_min(1e-300).log()).sum()))
            if IS_CONV[l]:
                q[f"mob_pool_{tag}"] = float((SP[l][r] / (n[l] / 4)).median())
    for i, tag in enumerate(WTAGS):
        Wt = P[2 * i].detach()
        nr = Wt.reshape(R, Wt.shape[1], -1).norm(dim=2)                  # (R, out) row norms
        for r in range(R):
            rows[r][f"w_norm_{tag}"] = float(nr[r].median())
    fro = P[8].detach().reshape(R, -1).norm(dim=1)
    for r in range(R):
        rows[r]["w_fro_f3"] = float(fro[r])
    return rows


@torch.no_grad()
def switch_eval(P, X, Y, act: BundleSnake, chunk: int = 20) -> tuple[torch.Tensor, torch.Tensor]:
    """(R,) CE and accuracy of the network on labels it has not trained on yet."""
    R = X.shape[0]
    ce = torch.zeros(R, dtype=torch.float64, device=X.device)
    hit = torch.zeros(R, dtype=torch.float64, device=X.device)
    for i0 in range(0, N_IMAGES, chunk):
        logits = forward(P, X[:, i0:i0 + chunk], act)[8]
        yb = Y[:, i0:i0 + chunk]
        b = yb.shape[1]
        ce += F.cross_entropy(logits.reshape(-1, N_CLASSES), yb.reshape(-1),
                              reduction="none").view(R, b).double().sum(1)
        hit += (logits.argmax(-1) == yb).sum(1)
    return ce / N_IMAGES, hit / N_IMAGES


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------

class Bundle:
    def __init__(self, slots: list[tuple[str, int]], cifar, device, lr=1e-3, epochs=400,
                 lo=0.005, hi=3.0, beta=0.01, graph=True):
        self.slots, self.device, self.lr, self.epochs = slots, device, lr, epochs
        self.R = R = len(slots)
        self.specs = [parse_arm(a) for a, s in slots]
        seeds = [s for a, s in slots]
        self.useeds = sorted(set(seeds))
        self.P = stack_init(seeds, device)
        self.act = BundleSnake(self.specs, device, lo=lo, hi=hi, beta=beta)
        self.m = [torch.zeros_like(p) for p in self.P]
        self.v = [torch.zeros_like(p) for p in self.P]
        self.tc = torch.zeros((), dtype=torch.float64, device=device)
        self.X = torch.stack([CN.images(cifar, CN.subset_idx(s), device) for s in seeds])
        self.Y = torch.zeros(R, N_IMAGES, dtype=torch.long, device=device)
        self.g_lab = {s: H.stream("rlcc_labels", s) for s in self.useeds}
        self.g_batch = {s: H.stream("rlcc_batch", s) for s in self.useeds}
        self.order = torch.zeros(R, N_IMAGES, dtype=torch.long, device=device)
        self.acc_ep = torch.zeros(R, device=device)        # sum of batch hits in this epoch
        self.bad = torch.zeros(R, dtype=torch.bool, device=device)
        self.ar = torch.arange(R, device=device)[:, None]
        self.b1 = torch.tensor(0.9, dtype=torch.float64, device=device)
        self.b2 = torch.tensor(0.999, dtype=torch.float64, device=device)
        self.graph = graph and device.type == "cuda"
        self.cg = None
        # per-parameter update masks, only when some run keeps its conv layers fixed (otherwise
        # the step is exactly the unmasked one)
        self.upd_mask = None
        if any(sp["fixconv"] or sp["fixfc"] for sp in self.specs):
            kc = torch.tensor([0.0 if sp["fixconv"] else 1.0 for sp in self.specs], device=device)
            kf = torch.tensor([0.0 if sp["fixfc"] else 1.0 for sp in self.specs], device=device)
            use_c = any(sp["fixconv"] for sp in self.specs)
            use_f = any(sp["fixfc"] for sp in self.specs)
            self.upd_mask = [(kc if i < 4 else kf).view(-1, *([1] * (p.dim() - 1)))
                             if ((i < 4 and use_c) or (i >= 4 and use_f)) else None
                             for i, p in enumerate(self.P)]

    # one step on the static tensors; j = position of the batch in the epoch
    def step(self, j: int) -> None:
        idx = self.order[:, j * BATCH:(j + 1) * BATCH]
        xb = self.X[self.ar, idx]                       # (R, B, 3, 32, 32)
        yb = self.Y[self.ar, idx]
        out = forward(self.P, xb, self.act)
        logits = out[8]
        lossv = F.cross_entropy(logits.reshape(-1, N_CLASSES), yb.reshape(-1),
                                reduction="none").view(self.R, BATCH).mean(1)
        self.acc_ep.add_((logits.detach().argmax(-1) == yb).float().mean(1))
        grads = torch.autograd.grad(lossv.sum(), self.P)
        with torch.no_grad():
            self.bad.logical_or_(~torch.isfinite(lossv))
            self.tc.add_(1)
            c1 = (1 - torch.pow(self.b1, self.tc)).float()
            c2 = (1 - torch.pow(self.b2, self.tc)).float()
            for i, (p, gr, mi, vi) in enumerate(zip(self.P, grads, self.m, self.v)):
                mi.mul_(0.9).add_(gr, alpha=1 - 0.9)
                vi.mul_(0.999).addcmul_(gr, gr, value=1 - 0.999)
                if self.upd_mask is not None and self.upd_mask[i] is not None:
                    p.sub_(self.upd_mask[i] * (self.lr * (mi / c1) / ((vi / c2).sqrt() + 1e-8)))
                else:
                    p.sub_(self.lr * (mi / c1) / ((vi / c2).sqrt() + 1e-8))
            self.act.update([out[0].detach(), out[2].detach(), out[4].detach(), out[6].detach()])

    def epoch_body(self) -> None:
        self.acc_ep.zero_()
        for j in range(SPE):
            self.step(j)

    def _state_tensors(self):
        return (*self.P, *self.m, *self.v, self.tc, self.acc_ep, self.bad)

    def capture(self) -> None:
        """Warm up on a side stream, capture one epoch (75 steps), then put back every tensor
        the warm-up and the capture touched (both execute the steps)."""
        keep = [q.detach().clone() for q in self._state_tensors()]
        keep_act = self.act.state()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                self.step(0)
        torch.cuda.current_stream().wait_stream(side)
        self.cg = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.cg):
            self.epoch_body()
        with torch.no_grad():
            for q, v in zip(self._state_tensors(), keep):
                q.copy_(v)
        self.act.load_state(keep_act)
        torch.cuda.synchronize()

    def new_order(self) -> None:
        perm = {s: torch.randperm(N_IMAGES, generator=self.g_batch[s]) for s in self.useeds}
        self.order.copy_(torch.stack([perm[s] for a, s in self.slots]))

    def run_epoch(self) -> None:
        self.new_order()
        if self.graph:
            if self.cg is None:
                self.capture()
            self.cg.replay()
        else:
            self.epoch_body()

    def new_labels(self) -> None:
        lab = {s: CN.task_labels(self.g_lab[s]) for s in self.useeds}
        self.Y.copy_(torch.stack([lab[s] for a, s in self.slots]))

    # ---- checkpoints (resume and post-hoc)
    def full_state(self) -> dict:
        return {"P": [p.detach().cpu() for p in self.P], "m": [q.cpu() for q in self.m],
                "v": [q.cpu() for q in self.v], "tc": float(self.tc),
                "act": {k: [t.cpu() for t in vs] for k, vs in self.act.state().items()},
                "g_lab": {s: g.get_state() for s, g in self.g_lab.items()},
                "g_batch": {s: g.get_state() for s, g in self.g_batch.items()},
                "bad": self.bad.cpu()}

    def load_full_state(self, st: dict) -> None:
        with torch.no_grad():
            for dst, src in ((self.P, st["P"]), (self.m, st["m"]), (self.v, st["v"])):
                for q, s in zip(dst, src):
                    q.copy_(s)
            self.tc.fill_(st["tc"])
            self.bad.copy_(st["bad"])
        self.act.load_state({k: [t.to(self.device) for t in vs] for k, vs in st["act"].items()})
        for s in self.useeds:
            self.g_lab[s].set_state(st["g_lab"][s])
            self.g_batch[s].set_state(st["g_batch"][s])

    def run_state(self, r: int) -> dict:
        """One run's slice, for post-hoc analysis (`load_run`)."""
        return {"arm": self.slots[r][0], "seed": self.slots[r][1],
                "P": [p.detach()[r].cpu().clone() for p in self.P],
                "m": [q[r].cpu().clone() for q in self.m], "v": [q[r].cpu().clone() for q in self.v],
                "tc": float(self.tc),
                "act": {k: [t[r].cpu().clone() for t in vs] for k, vs in self.act.state().items()
                        if k != "cval"} | {"cval": [t[r].cpu().clone() for t in self.act.cval]}}


def save_atomic(obj, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def git_state() -> dict:
    import subprocess
    root = H.REPO
    head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--", "src"],
                           capture_output=True, text=True).stdout.strip()
    return {"git_hash": head, "git_dirty_src": bool(dirty)}


def run(slots, out: Path, n_tasks: int, epochs: int, device, lr=1e-3, ckpt_tasks=CKPT_TASKS,
        graph=True, resume=True, log=None) -> None:
    if log is None:
        def log(msg):
            print(msg, flush=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / "ckpt").mkdir(exist_ok=True)
    cifar = RC.Cifar10()
    B = Bundle(slots, cifar, device, lr=lr, epochs=epochs, graph=graph)
    R = B.R
    meta = {"slots": [list(s) for s in slots], "epochs": epochs, "lr": lr, "n_tasks": n_tasks}
    st_path = out / "state.pt"
    rows: list[dict] = []
    curves = np.zeros((R, n_tasks, epochs), dtype=np.float32)
    t0, gits, resumed = 1, [git_state()], []
    if resume and st_path.exists():
        st = torch.load(st_path, map_location="cpu", weights_only=False)
        if st["meta"] != meta:
            raise SystemExit(f"{st_path} belongs to another configuration")
        B.load_full_state(st["state"])
        rows, t0 = st["rows"], st["t"] + 1
        curves[:, :st["t"]] = st["curves"][:, :st["t"]]
        gits = st["gits"] + gits
        resumed = st["resumed"] + [t0]
        log(f"[{time.strftime('%T')}] resumed at task {t0}")
    frz = {}
    for r, sp in enumerate(B.specs):
        if sp["frz"] is not None:
            frz.setdefault(sp["frz"], []).append(r)
    # Reserve the evaluation's peak memory now: the caching allocator keeps it, so a later
    # neighbour on the GPU (ollama loads a 15 GB model on demand) cannot starve the
    # end-of-task evaluation.  Read only: no parameter, moment, V or generator is touched.
    evaluate(B.P, B.X, B.Y, B.act)
    switch_eval(B.P, B.X, B.Y, B.act)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_start = time.time()
    for t in range(t0, n_tasks + 1):
        tt = time.time()
        B.new_labels()
        sce, sacc = switch_eval(B.P, B.X, B.Y, B.act)
        acc_sum = torch.zeros(R, dtype=torch.float64, device=device)
        ep_acc = torch.zeros(R, epochs, device=device)
        for e in range(epochs):
            B.run_epoch()
            ep_acc[:, e].copy_(B.acc_ep)
        ep_acc /= SPE
        curves[:, t - 1] = ep_acc.cpu().numpy()
        online = ep_acc.double().mean(1)
        ev = evaluate(B.P, B.X, B.Y, B.act)
        bad = B.bad.cpu()
        for r, (arm, seed) in enumerate(slots):
            cur = curves[r, t - 1]
            hit99 = np.nonzero(cur >= 0.99)[0]
            rows.append({"arm": arm, "seed": seed, "task": t, "online_acc": float(online[r]),
                         "switch_ce": float(sce[r]), "switch_acc": float(sacc[r]),
                         "ep_first99": int(hit99[0]) + 1 if hit99.size else -1,
                         "acc_ep1": float(cur[0]), "acc_ep10": float(cur[min(9, epochs - 1)]),
                         "diverged": bool(bad[r]), **ev[r]})
        if t in frz:
            B.act.freeze(frz[t])
        if t in ckpt_tasks or t == n_tasks:
            for r in range(R):
                arm, seed = slots[r]
                save_atomic(B.run_state(r), out / "ckpt" / f"{arm.replace(':', '_')}_seed{seed}_t{t:02d}.pt")
        H.write_csv(out / "per_task.csv", rows)
        np.save(out / "curves.npy", curves)
        save_atomic({"meta": meta, "state": B.full_state(), "rows": rows, "t": t,
                     "curves": curves, "gits": gits, "resumed": resumed}, st_path)
        m_on = {}
        for r, (arm, seed) in enumerate(slots):
            m_on.setdefault(arm, []).append(float(online[r]))
        log(f"[{time.strftime('%T')}] task {t:2d} {time.time() - tt:6.1f}s  "
            + "  ".join(f"{a} {np.mean(v):.4f}" for a, v in m_on.items()))
    prov = {"run_id": EXPERIMENT, **meta, "arms": {a: parse_arm(a) for a in sorted({a for a, s in slots})},
            "host": "src/rlcifar_cnn_0908.py (init, subset, streams, net)",
            "engine": "grouped conv / bmm bundle, one epoch per CUDA graph" if B.graph else "eager",
            "gits": gits, "resumed": resumed, "device": str(device),
            "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
            "torch": torch.__version__, "data_sha256": cifar.sha256,
            "wall_clock_s_this_process": time.time() - t_start,
            "ckpt_tasks": list(ckpt_tasks),
            "cuda_max_mem_mb": torch.cuda.max_memory_allocated() / 2 ** 20 if device.type == "cuda" else None}
    (out / "provenance.json").write_text(json.dumps(prov, indent=2, default=str))
    log(f"done {out}")


def parse_seeds(s: str) -> list[int]:
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
    ap.add_argument("--arms", required=True)
    ap.add_argument("--seeds", default="10-19")
    ap.add_argument("--tasks", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = H.setup(args.device)
    arms = args.arms.split(",")
    for a in arms:
        parse_arm(a)
    slots = [(a, s) for a in arms for s in parse_seeds(args.seeds)]
    run(slots, Path(args.out), args.tasks, args.epochs, device, lr=args.lr, graph=not args.no_graph)


if __name__ == "__main__":
    main()
