#!/usr/bin/env python3
"""Centering doors C / H / CH on the RL-CIFAR CNN with ReLU (cnn_center_relu_1011).

    python src/cnn_center_relu_1011.py run --arm CH --seeds 10-19 --out results/cnn_center_relu_1011/CH
    python src/cnn_center_relu_1011.py run --arm Cr --seeds 10    --out results/cnn_center_relu_1011/Cr

Spec: specs/spec_cnn_center_relu_1011.md (registered before this file was written).

The box, the init, the image subset, the label / batch streams and the bundled training engine
(grouped conv / bmm / one epoch per CUDA graph, one Adam per run) are sna_cnn_cause_1009's,
imported unchanged.  What this file adds:

  * the activation is ReLU at all four sites (host "R": clamp(z, min=0); diagnostic gate z > 0),
    the interface of cnn_drive_verify_1009_lr's BundleLeaky with slope 0, plus the H door state;
  * door C: x <- x - mu, mu = the CIFAR-10 training-set channel mean (cnn_center_1010's LRc,
    subtracted once when the bundle is built); control Cr: x <- x - r, r = mu - |mu| v with v a
    random direction orthogonal to mu-hat (spec §3.3);
  * door H: the next layer receives (uncentred output) - m_l.  conv sites: m_l per channel, the
    batch mean over (images, positions) of maxpool(ReLU(z)) -- what the next layer reads; fc sites:
    per unit, the batch mean of ReLU(z).  After every Adam step m <- (1 - beta) m + beta * bm,
    beta = 0.01, from the forward pass that produced the step's gradient (pre-update weights),
    init 0, never reset, never differentiated; training, online, memo, diagnostics and the switch
    CE all use the current m;
  * a forward pass with the doors (`forward`); with door H off it is the engine's forward, op by op;
  * per-task diagnostics (spec §4): death rate, open fraction, the input's mean patch / vector of
    every layer and the identity zbar_j = <W_j, m_in> + b_j (mu path and b path), the H state, and
    for the H arms an evaluation with m = 0 (read only);
  * the task loop (`run`): the engine's `run` with t0 / units.npz / diagnostics / h_upd added.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np                                   # noqa: E402
import torch                                         # noqa: E402
import torch.nn.functional as F                      # noqa: E402

from src import pmnist_0905 as H                     # noqa: E402  (host of the host; not touched)
from src import pmnist_rlcifar_0907 as RC            # noqa: E402  (CIFAR data layer; not touched)
from src import rlcifar_cnn_0908 as CN               # noqa: E402  (the CNN box; not touched)
from src import sna_cnn_cause_1009 as E              # noqa: E402  (the bundled engine; not touched)

EXPERIMENT = "cnn_center_relu_1011"
SPEC = "specs/spec_cnn_center_relu_1011.md"
N_IMAGES, BATCH, SPE, N_CLASSES = CN.N_IMAGES, CN.BATCH, CN.STEPS_PER_EPOCH, CN.N_CLASSES
CH, KER, PAD, POOL = CN.CHANNELS, CN.KERNEL, CN.PAD, CN.POOL
SITES, WIDTHS, IS_CONV = CN.SITES, CN.WIDTHS, CN.IS_CONV
D_IN = (3 * KER * KER, CH * KER * KER, CN.FLAT, CN.HIDDEN)      # length of each layer's input patch
CKPT_TASKS = (1, 2, 3, 5, 10, 20, 30)
BETA_H = 0.01
CR_STREAM = ("cnn_center_relu_1011_cr", 0)

# arm -> (input shift, door H)
ARMS = {
    "ref": dict(shift=None, h=False),
    "C":   dict(shift="mu", h=False),
    "H":   dict(shift=None, h=True),
    "CH":  dict(shift="mu", h=True),
    "Cr":  dict(shift="cr", h=False),
}


# --------------------------------------------------------------------------
# input shifts (door C and the control Cr)
# --------------------------------------------------------------------------

def train_channel_mean(cifar) -> torch.Tensor:
    """(3,) float32 mean of each colour channel over the 50,000 training images, /255 scale
    (the same expression as cnn_center_1010.train_channel_mean)."""
    x = cifar.train_u8.reshape(-1, 3, 1024).double() / 255.0
    return x.mean((0, 2)).float()


def cr_constant(mu: torch.Tensor) -> torch.Tensor:
    """r = mu - |mu| v, v = the part of u ~ N(0, I3) orthogonal to mu-hat, normalised (spec §3.3).
    The shifted input keeps a mean of norm |mu| (as the raw input) along v, orthogonal to the raw
    mean direction."""
    g = H.stream(*CR_STREAM)
    u = torch.randn(3, generator=g, dtype=torch.float64)
    m = mu.double()
    mh = m / m.norm()
    v = u - (u @ mh) * mh
    v = v / v.norm()
    return (m - m.norm() * v).float()


def input_shift(kind, cifar) -> torch.Tensor | None:
    if kind is None:
        return None
    mu = train_channel_mean(cifar)
    if kind == "mu":
        return mu
    if kind == "cr":
        return cr_constant(mu)
    raise ValueError(kind)


# --------------------------------------------------------------------------
# activation: ReLU + the H door state
# --------------------------------------------------------------------------

class ReLUDoors:
    """ReLU for R bundled runs with BundleLeaky's interface (phi / dphi / update / state /
    load_state / alpha / freeze / V) and the door-H running means m (one (R, width) tensor per
    site).  `h_on` is per bundle: every run of a bundle shares its arm."""

    def __init__(self, R: int, h: bool, device, beta: float = 0.01, beta_h: float = BETA_H):
        self.R, self.beta, self.beta_h, self.h_on = R, beta, beta_h, bool(h)
        self.V = [torch.ones(R, w, device=device) for w in WIDTHS]          # record only
        self.ada = [torch.zeros(R, w, dtype=torch.bool, device=device) for w in WIDTHS]
        self.cval = [torch.ones(R, 1, device=device) for _ in WIDTHS]
        self.fixA = [torch.full((R, w), float("nan"), device=device) for w in WIDTHS]
        self.pp = torch.zeros(R, 1, dtype=torch.bool, device=device)
        self.any_pp = False
        self.amp = None
        self.m = [torch.zeros(R, w, device=device) for w in WIDTHS]          # door H
        self.hupd = torch.zeros(R, 4, device=device)                         # updates that moved m

    def alpha(self, l: int) -> torch.Tensor:                                # no alpha: NaN
        return self.fixA[l]

    def phi(self, z: torch.Tensor, l: int) -> torch.Tensor:
        return torch.clamp(z, min=0.0)                                      # host "R"

    def dphi(self, z: torch.Tensor, l: int) -> torch.Tensor:
        return (z > 0).to(z.dtype)

    @torch.no_grad()
    def batch_var(self, z: torch.Tensor, l: int) -> torch.Tensor:
        if not IS_CONV[l]:
            return z.var(1, unbiased=False)
        return z.var((0, 2, 3), unbiased=False).view(self.R, -1)

    @torch.no_grad()
    def out_mean(self, z: torch.Tensor, l: int) -> torch.Tensor:
        """(R, width) batch mean of the uncentred output the next layer reads: conv sites
        maxpool(ReLU(z)) averaged over (images, positions), fc sites ReLU(z) over images."""
        a = self.phi(z, l)
        if IS_CONV[l]:
            return F.max_pool2d(a, POOL, POOL).mean((0, 2, 3)).view(self.R, -1)
        return a.mean(1)

    @torch.no_grad()
    def update(self, zs) -> None:
        for l, z in enumerate(zs):
            self.V[l].mul_(1 - self.beta).add_(self.beta * self.batch_var(z, l))
        if self.h_on:
            for l, z in enumerate(zs):
                bm = self.out_mean(z, l)
                old = self.m[l].clone()
                self.m[l].mul_(1 - self.beta_h).add_(bm, alpha=self.beta_h)
                self.hupd[:, l].add_((self.m[l] != old).any(1).to(self.hupd.dtype))

    @torch.no_grad()
    def freeze(self, runs) -> None:                                         # nothing to freeze
        return None

    def state(self) -> dict:
        st = {k: [t.clone() for t in getattr(self, k)] for k in ("V", "ada", "cval", "fixA", "m")}
        st["hupd"] = [self.hupd.clone()]
        return st

    def load_state(self, st: dict) -> None:
        for k in ("V", "ada", "cval", "fixA", "m"):
            for dst, src in zip(getattr(self, k), st[k]):
                dst.copy_(src)                       # in place: a captured graph holds these
        self.hupd.copy_(st["hupd"][0])


# --------------------------------------------------------------------------
# forward with the doors
# --------------------------------------------------------------------------

def forward(P, xb: torch.Tensor, act: ReLUDoors, m=None, with_inputs: bool = False):
    """xb (R, B, 3, 32, 32) -> (z1, a1, z2, a2, z3, a3, z4, a4, logits), a = the uncentred ReLU.

    conv -> ReLU -> pool (-> - m) -> next layer, as the engine's forward; door H subtracts the
    running mean from what the next layer reads.  `m` overrides act.m (the read-only m = 0
    evaluation); `with_inputs` also returns the inputs of the four gated layers (x, P1, h, A3)."""
    Wc1, bc1, Wc2, bc2, Wf1, bf1, Wf2, bf2, Wf3, bf3 = P
    R, B = xb.shape[:2]
    if m is None:
        m = act.m
    hon = act.h_on
    x = xb.transpose(0, 1).reshape(B, R * 3, 32, 32)
    z1 = F.conv2d(x, Wc1.reshape(R * CH, 3, KER, KER), bc1.reshape(-1), padding=PAD, groups=R)
    a1 = act.phi(z1, 0)
    p1 = F.max_pool2d(a1, POOL, POOL)
    if hon:
        p1 = p1 - m[0].reshape(1, R * CH, 1, 1)
    z2 = F.conv2d(p1, Wc2.reshape(R * CH, CH, KER, KER), bc2.reshape(-1), padding=PAD, groups=R)
    a2 = act.phi(z2, 1)
    p2 = F.max_pool2d(a2, POOL, POOL)
    if hon:
        p2 = p2 - m[1].reshape(1, R * CH, 1, 1)
    h = p2.reshape(B, R, CN.FLAT).transpose(0, 1)                       # (R, B, 1024)
    z3 = torch.baddbmm(bf1[:, None, :], h, Wf1.transpose(1, 2))
    a3 = act.phi(z3, 2)
    i4 = a3 - m[2][:, None, :] if hon else a3
    z4 = torch.baddbmm(bf2[:, None, :], i4, Wf2.transpose(1, 2))
    a4 = act.phi(z4, 3)
    i5 = a4 - m[3][:, None, :] if hon else a4
    logits = torch.baddbmm(bf3[:, None, :], i5, Wf3.transpose(1, 2))
    out = (z1, a1, z2, a2, z3, a3, z4, a4, logits)
    if with_inputs:
        return out, (x, p1, h, i4)
    return out


@contextlib.contextmanager
def engine_forward():
    """Run the engine's `evaluate` / `switch_eval` (which call their module's `forward`) through
    this file's forward, and put the engine's own forward back afterwards."""
    saved = E.forward
    E.forward = forward
    try:
        yield
    finally:
        E.forward = saved


def evaluate(P, X, Y, act) -> list[dict]:
    with engine_forward():
        return E.evaluate(P, X, Y, act)


def switch_eval(P, X, Y, act):
    with engine_forward():
        return E.switch_eval(P, X, Y, act)


# --------------------------------------------------------------------------
# bundle
# --------------------------------------------------------------------------

class Bundle(E.Bundle):
    """The engine's Bundle with ReLU, the doors and this file's forward (constructor and step
    copied from the engine, every other method inherited).  All slots share one arm."""

    def __init__(self, slots, cifar, device, lr=1e-3, epochs=400, beta=0.01, graph=True,
                 beta_h=BETA_H):
        arms = {a for a, s in slots}
        assert len(arms) == 1 and arms <= set(ARMS), arms
        (arm,) = arms
        self.arm, self.door = arm, ARMS[arm]
        self.slots, self.device, self.lr, self.epochs = slots, device, lr, epochs
        self.R = R = len(slots)
        self.specs = [dict(site=(("relu", 0.0),) * 4, frz=None, pp=False, fixconv=False,
                           fixfc=False, fcamp=1.0) for _ in slots]
        seeds = [s for a, s in slots]
        self.useeds = sorted(set(seeds))
        self.P = E.stack_init(seeds, device)
        self.act = ReLUDoors(R, self.door["h"], device, beta=beta, beta_h=beta_h)
        self.m = [torch.zeros_like(p) for p in self.P]
        self.v = [torch.zeros_like(p) for p in self.P]
        self.tc = torch.zeros((), dtype=torch.float64, device=device)
        self.X = torch.stack([CN.images(cifar, CN.subset_idx(s), device) for s in seeds])
        self.shift = input_shift(self.door["shift"], cifar)
        if self.shift is not None:
            with torch.no_grad():
                self.X.sub_(self.shift.to(device).view(1, 1, 3, 1, 1))
        self.Y = torch.zeros(R, N_IMAGES, dtype=torch.long, device=device)
        self.g_lab = {s: H.stream("rlcc_labels", s) for s in self.useeds}
        self.g_batch = {s: H.stream("rlcc_batch", s) for s in self.useeds}
        self.order = torch.zeros(R, N_IMAGES, dtype=torch.long, device=device)
        self.acc_ep = torch.zeros(R, device=device)
        self.bad = torch.zeros(R, dtype=torch.bool, device=device)
        self.ar = torch.arange(R, device=device)[:, None]
        self.b1 = torch.tensor(0.9, dtype=torch.float64, device=device)
        self.b2 = torch.tensor(0.999, dtype=torch.float64, device=device)
        self.graph = graph and device.type == "cuda"
        self.cg = None
        self.upd_mask = None

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
            for p, gr, mi, vi in zip(self.P, grads, self.m, self.v):
                mi.mul_(0.9).add_(gr, alpha=1 - 0.9)
                vi.mul_(0.999).addcmul_(gr, gr, value=1 - 0.999)
                p.sub_(self.lr * (mi / c1) / ((vi / c2).sqrt() + 1e-8))
            self.act.update([out[0].detach(), out[2].detach(), out[4].detach(), out[6].detach()])


# --------------------------------------------------------------------------
# diagnostics (spec §4)
# --------------------------------------------------------------------------

GAMMA = lambda k: k * 2.0 ** -24 / (1 - k * 2.0 ** -24)                 # noqa: E731
TF32_PROD = 2.0 ** -10                                                  # two operands, 2^-11 each


@torch.no_grad()
def diagnostics(P, X, Y, act: ReLUDoors, chunk: int = 10, m0_eval: bool = True):
    """Per run: scalar columns (list of dicts) and per-unit arrays (dict of (R, n) numpy arrays).

    For every gated layer: zbar_j, sd_j (as the engine's evaluate), open fraction P(z > 0),
    death (z <= 0 on every image and position), the mean input patch / vector m_in (conv: the
    unfolded 5x5 patch with zero padding, as the conv sees it), <W_j, m_in>, b_j, the residual of
    the identity zbar_j = <W_j, m_in> + b_j and its rounding bound, |m_in| / rms deviation.
    With door H on (and m0_eval): memo / CE with m = 0 (read only)."""
    R = X.shape[0]
    dev = X.device
    f64 = dict(dtype=torch.float64, device=dev)
    S1 = [torch.zeros(R, w, **f64) for w in WIDTHS]
    S2 = [torch.zeros(R, w, **f64) for w in WIDTHS]
    SO = [torch.zeros(R, w, **f64) for w in WIDTHS]
    ZMX = [torch.full((R, w), -math.inf, device=dev) for w in WIDTHS]
    MI = [torch.zeros(R, d, **f64) for d in D_IN]
    MA = [torch.zeros(R, d, **f64) for d in D_IN]
    MQ = [torch.zeros(R, **f64) for _ in D_IN]
    nz = [0] * 4
    ni = [0] * 4
    for i0 in range(0, N_IMAGES, chunk):
        xb = X[:, i0:i0 + chunk]
        b = xb.shape[1]
        out, ins = forward(P, xb, act, with_inputs=True)
        for l in range(4):
            z = out[2 * l]
            if IS_CONV[l]:
                zr = z.reshape(b, R, WIDTHS[l], -1)                     # (b, R, C, HW)
                zd = zr.double()
                S1[l] += zd.sum((0, 3)); S2[l] += (zd * zd).sum((0, 3))
                SO[l] += (zr > 0).double().sum((0, 3))
                ZMX[l] = torch.maximum(ZMX[l], zr.amax((0, 3)))
                nz[l] += b * zr.shape[3]
                U = F.unfold(ins[l], KER, padding=PAD).view(b, R, D_IN[l], -1).double()
                MI[l] += U.sum((0, 3)); MA[l] += U.abs().sum((0, 3)); MQ[l] += (U * U).sum((0, 2, 3))
                ni[l] += b * U.shape[3]
            else:
                zd = z.double()                                        # (R, b, n)
                S1[l] += zd.sum(1); S2[l] += (zd * zd).sum(1)
                SO[l] += (z > 0).double().sum(1)
                ZMX[l] = torch.maximum(ZMX[l], z.amax(1))
                nz[l] += b
                U = ins[l].double()                                    # (R, b, d)
                MI[l] += U.sum(1); MA[l] += U.abs().sum(1); MQ[l] += (U * U).sum((1, 2))
                ni[l] += b
    rows = [dict() for _ in range(R)]
    units: dict[str, np.ndarray] = {}
    Wt = (P[0], P[2], P[4], P[6])
    bt = (P[1], P[3], P[5], P[7])
    for l, tag in enumerate(SITES):
        zbar = S1[l] / nz[l]
        var = (S2[l] / nz[l] - zbar * zbar).clamp_min(0) * nz[l] / (nz[l] - 1)
        sd = var.sqrt()
        opn = SO[l] / nz[l]
        dead = ZMX[l] <= 0
        m_in = MI[l] / ni[l]
        m_abs = MA[l] / ni[l]
        sq = MQ[l] / ni[l]
        W = Wt[l].detach().reshape(R, WIDTHS[l], -1).double()            # (R, n, d)
        bias = bt[l].detach().double()
        wm = torch.einsum("rnd,rd->rn", W, m_in)
        res = (zbar - wm - bias).abs()
        bound = (TF32_PROD + GAMMA(D_IN[l] + 1)) * (torch.einsum("rnd,rd->rn", W.abs(), m_abs) + bias.abs())
        mn2 = (m_in * m_in).sum(1)
        dev2 = (sq - mn2).clamp_min(0)
        r_in = mn2.sqrt() / dev2.sqrt()
        units |= {f"zbar_{tag}": zbar.float().cpu().numpy(), f"sd_{tag}": sd.float().cpu().numpy(),
                  f"open_{tag}": opn.float().cpu().numpy(), f"dead_{tag}": dead.cpu().numpy(),
                  f"b_{tag}": bias.float().cpu().numpy(), f"wm_{tag}": wm.float().cpu().numpy(),
                  f"m_in_{tag}": m_in.float().cpu().numpy(),
                  f"mdoor_{tag}": act.m[l].detach().float().cpu().numpy()}
        for r in range(R):
            q = rows[r]
            q[f"zbar_mean_{tag}"] = float(zbar[r].mean())
            q[f"open_mean_{tag}"] = float(opn[r].mean())
            q[f"dead_frac_{tag}"] = float(dead[r].double().mean())
            q[f"r_in_{tag}"] = float(r_in[r])
            q[f"wm_med_{tag}"] = float(wm[r].median())
            q[f"b_med_{tag}"] = float(bias[r].median())
            q[f"ident_res_{tag}"] = float(res[r].max())
            q[f"ident_ratio_{tag}"] = float((res[r] / bound[r]).max())
            if act.h_on:
                q[f"m_norm_{tag}"] = float(act.m[l][r].norm())
                q[f"m_mean_{tag}"] = float(act.m[l][r].mean())
                q[f"h_upd_{tag}"] = int(act.hupd[r, l])
    if act.h_on and m0_eval:
        zeros = [torch.zeros_like(t) for t in act.m]
        hit = torch.zeros(R, **f64)
        ce = torch.zeros(R, **f64)
        for i0 in range(0, N_IMAGES, chunk):
            logits = forward(P, X[:, i0:i0 + chunk], act, m=zeros)[8]
            yb = Y[:, i0:i0 + chunk]
            b = yb.shape[1]
            hit += (logits.argmax(-1) == yb).sum(1)
            ce += F.cross_entropy(logits.reshape(-1, N_CLASSES), yb.reshape(-1),
                                  reduction="none").view(R, b).double().sum(1)
        for r in range(R):
            rows[r]["memo_acc_m0"] = float(hit[r]) / N_IMAGES
            rows[r]["fit_ce_m0"] = float(ce[r]) / N_IMAGES
    return rows, units


def major_frac(Y: torch.Tensor) -> list[float]:
    return [float(torch.bincount(Y[r], minlength=N_CLASSES).max()) / Y.shape[1] for r in range(Y.shape[0])]


# --------------------------------------------------------------------------
# task loop (the engine's `run`, with t0, diagnostics, units.npz and h_upd)
# --------------------------------------------------------------------------

def save_units(path: Path, units: dict, n_tasks: int, R: int) -> None:
    """units[t] = dict of (R, n) arrays for t = 0..n_tasks done so far -> (T+1, R, n) arrays."""
    keys = sorted(next(iter(units.values())).keys())
    arr = {}
    for k in keys:
        shape = next(iter(units.values()))[k].shape
        dt = np.bool_ if next(iter(units.values()))[k].dtype == np.bool_ else np.float32
        a = np.zeros((n_tasks + 1, *shape), dtype=dt) if dt == np.bool_ else \
            np.full((n_tasks + 1, *shape), np.nan, dtype=dt)
        for t, u in units.items():
            a[t] = u[k]
        arr[k] = a
    arr["tasks_done"] = np.array(sorted(units), dtype=np.int64)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, **arr)
    os.replace(tmp, path)


def git_state() -> dict:
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", "src", "analysis"],
                           capture_output=True, text=True).stdout.strip()
    return {"git_hash": head, "git_dirty_code": bool(dirty)}


def run(arm: str, seeds: list[int], out: Path, n_tasks: int, epochs: int, device, lr=1e-3,
        ckpt_tasks=CKPT_TASKS, graph=True, resume=True, log=None, m0_eval=True) -> None:
    if log is None:
        def log(msg):
            print(msg, flush=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / "ckpt").mkdir(exist_ok=True)
    slots = [(arm, s) for s in seeds]
    cifar = RC.Cifar10()
    B = Bundle(slots, cifar, device, lr=lr, epochs=epochs, graph=graph)
    R = B.R
    meta = {"arm": arm, "slots": [list(s) for s in slots], "epochs": epochs, "lr": lr,
            "n_tasks": n_tasks, "beta_h": BETA_H}
    st_path = out / "state.pt"
    rows: list[dict] = []
    rows0: list[dict] = []
    units: dict[int, dict] = {}
    curves = np.zeros((R, n_tasks, epochs), dtype=np.float32)
    t0, gits, resumed = 1, [git_state()], []
    if resume and st_path.exists():
        st = torch.load(st_path, map_location="cpu", weights_only=False)
        if st["meta"] != meta:
            raise SystemExit(f"{st_path} belongs to another configuration")
        B.load_full_state(st["state"])
        rows, rows0, units, t0 = st["rows"], st["rows0"], st["units"], st["t"] + 1
        curves[:, :st["t"]] = st["curves"][:, :st["t"]]
        gits = st["gits"] + gits
        resumed = st["resumed"] + [t0]
        log(f"[{time.strftime('%T')}] resumed at task {t0}")
    # the initial network (task 0); read only.  Like the engine's first `evaluate`, it also
    # reserves the evaluation's peak memory before training starts.
    ev0 = evaluate(B.P, B.X, B.Y, B.act)
    switch_eval(B.P, B.X, B.Y, B.act)
    dg0, un0 = diagnostics(B.P, B.X, B.Y, B.act, m0_eval=m0_eval)
    if t0 == 1:
        rows0 = [{"arm": a, "seed": s, "task": 0, **ev0[r], **dg0[r]} for r, (a, s) in enumerate(slots)]
        units[0] = un0
        H.write_csv(out / "t0.csv", rows0)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_start = time.time()
    for t in range(t0, n_tasks + 1):
        tt = time.time()
        B.new_labels()
        B.act.hupd.zero_()
        sce, sacc = switch_eval(B.P, B.X, B.Y, B.act)
        ep_acc = torch.zeros(R, epochs, device=device)
        for e in range(epochs):
            B.run_epoch()
            ep_acc[:, e].copy_(B.acc_ep)
        ep_acc /= SPE
        curves[:, t - 1] = ep_acc.cpu().numpy()
        online = ep_acc.double().mean(1)
        ev = evaluate(B.P, B.X, B.Y, B.act)
        dg, un = diagnostics(B.P, B.X, B.Y, B.act, m0_eval=m0_eval)
        units[t] = un
        mf = major_frac(B.Y)
        bad = B.bad.cpu()
        for r, (a, seed) in enumerate(slots):
            cur = curves[r, t - 1]
            hit99 = np.nonzero(cur >= 0.99)[0]
            rows.append({"arm": a, "seed": seed, "task": t, "online_acc": float(online[r]),
                         "switch_ce": float(sce[r]), "switch_acc": float(sacc[r]),
                         "ep_first99": int(hit99[0]) + 1 if hit99.size else -1,
                         "acc_ep1": float(cur[0]), "acc_ep10": float(cur[min(9, epochs - 1)]),
                         "diverged": bool(bad[r]), **ev[r], "major_frac": mf[r], **dg[r]})
        if t in ckpt_tasks or t == n_tasks:
            for r in range(R):
                a, seed = slots[r]
                E.save_atomic(B.run_state(r), out / "ckpt" / f"{a}_seed{seed}_t{t:02d}.pt")
        H.write_csv(out / "per_task.csv", rows)
        np.save(out / "curves.npy", curves)
        save_units(out / "units.npz", units, n_tasks, R)
        E.save_atomic({"meta": meta, "state": B.full_state(), "rows": rows, "rows0": rows0,
                       "units": units, "t": t, "curves": curves, "gits": gits,
                       "resumed": resumed}, st_path)
        # no accuracy in the log: monitoring reads the task number and the seconds only (spec §8)
        log(f"[{time.strftime('%T')}] task {t:2d} {time.time() - tt:6.1f}s  {arm}")
    prov = {"run_id": EXPERIMENT, "spec": SPEC, **meta,
            "door": ARMS[arm], "input_shift": None if B.shift is None else B.shift.tolist(),
            "activation": "relu (clamp(z, min=0)); diagnostic gate z > 0",
            "host": "src/rlcifar_cnn_0908.py (init, subset, streams, net)",
            "engine": "src/sna_cnn_cause_1009.py Bundle (grouped conv / bmm / one epoch per CUDA graph)"
                      + ("" if B.graph else " -- eager"),
            "gits": gits, "resumed": resumed, "device": str(device),
            "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
            "torch": torch.__version__, "tf32_cudnn": torch.backends.cudnn.allow_tf32,
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "data_sha256": cifar.sha256, "ckpt_tasks": list(ckpt_tasks),
            "wall_clock_s_this_process": time.time() - t_start,
            "cuda_max_mem_mb": torch.cuda.max_memory_allocated() / 2 ** 20 if device.type == "cuda" else None}
    (out / "provenance.json").write_text(json.dumps(prov, indent=2, default=str))
    log(f"done {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run"])
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--seeds", default="10-19")
    ap.add_argument("--tasks", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mem-gb", type=float, default=2.4)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()
    device = H.setup("cuda")
    di = torch.cuda.current_device()
    total = torch.cuda.get_device_properties(di).total_memory / 2 ** 30
    torch.cuda.set_per_process_memory_fraction(min(1.0, args.mem_gb / total), di)
    run(args.arm, E.parse_seeds(args.seeds), Path(args.out), args.tasks, args.epochs, device,
        lr=args.lr, resume=not args.no_resume)


if __name__ == "__main__":
    main()
