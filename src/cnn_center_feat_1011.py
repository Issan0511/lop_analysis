#!/usr/bin/env python3
"""Per-feature centering door H' on the RL-CIFAR CNN with ReLU (cnn_center_feat_1011).

    python src/cnn_center_feat_1011.py run --arm CHf --seeds 10-19 --out results/cnn_center_feat_1011/CHf
    python src/cnn_center_feat_1011.py run --arm Hs  --seeds 10    --out results/cnn_center_feat_1011/Hs

Spec: specs/spec_cnn_center_feat_1011.md (registered at b1b28854, before this file was written).

The follow-up of cnn_center_relu_1011 (P181).  The box, the bundled engine, ReLU, door C and the
fc part of door H are that file's (src/cnn_center_relu_1011.py, imported unchanged).  What
changes is door H's running mean at the two conv sites (what c2 and f1 read):

  * "chan" (P181's H; arms H / CH):  maxpool(ReLU(z)) - m,  m per channel, (R, 16);
  * "feat" (H'; arms Hf / CHf):       maxpool(ReLU(z)) - m,  m per (channel x position) of the pool
                                       output: (R, 16*16*16) after c1 and (R, 16*8*8) after c2, the
                                       batch mean over images only -- every scalar feature the next
                                       layer reads, as the MLP's H;
  * "shuf" (control; arms Hs / CHs):  the same m as "feat" (same EMA), but the next layer receives
                                       maxpool(ReLU(z)) - m[:, pi(p)], pi a fixed derangement of the
                                       pool positions drawn from a named stream (same for every
                                       channel and run).
fc sites (f1, f2) are per unit in every mode.  m <- (1 - beta) m + beta * bm after every Adam step,
beta = 0.01, bm from the step's own forward (pre-update weights), init 0, never reset, never
differentiated; training, online, memo, diagnostics and the switch CE use the current m.
Arms ref / C / H / CH run P181's code path op by op (checks K-same181, K-reuse).

Added diagnostics (spec §4): the per-feature image mean of the pool outputs read by c2 and f1,
before (mu_pre) and after (mu_post) the door, split into the channel part and the per-position
part; norms over the rms deviation.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
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
from src import cnn_center_relu_1011 as CR           # noqa: E402  (P181's engine; not touched)

EXPERIMENT = "cnn_center_feat_1011"
SPEC = "specs/spec_cnn_center_feat_1011.md"
N_IMAGES, BATCH, SPE, N_CLASSES = CN.N_IMAGES, CN.BATCH, CN.STEPS_PER_EPOCH, CN.N_CLASSES
CH, KER, PAD, POOL = CN.CHANNELS, CN.KERNEL, CN.PAD, CN.POOL
SITES, WIDTHS, IS_CONV = CN.SITES, CN.WIDTHS, CN.IS_CONV
D_IN = CR.D_IN
CKPT_TASKS = (1, 2, 3, 5, 10, 20, 30)
BETA_H = 0.01
SIDE = (16, 8)                                       # pool-output side after c1, c2
NFEAT = tuple(CH * s * s for s in SIDE)              # 4096, 1024 features read by c2, f1
SHUF_ROLE = "cnn_center_feat_1011_shuffle"
PARENT_ENGINE = ROOT / "src" / "cnn_center_relu_1011.py"

# arm -> (input shift, door H mode)
ARMS = {
    "ref": dict(shift=None, h=None),
    "C":   dict(shift="mu", h=None),
    "H":   dict(shift=None, h="chan"),
    "CH":  dict(shift="mu", h="chan"),
    "Hf":  dict(shift=None, h="feat"),
    "CHf": dict(shift="mu", h="feat"),
    "Hs":  dict(shift=None, h="shuf"),
    "CHs": dict(shift="mu", h="shuf"),
}


# --------------------------------------------------------------------------
# the control's position permutation
# --------------------------------------------------------------------------

def shuffle_perm(side: int) -> tuple[torch.Tensor, int]:
    """A derangement of the side*side pool positions (spec §3.1): randperm from the stream
    (SHUF_ROLE, side), drawn again from the same stream while it has a fixed point.
    Returns (perm, number of draws)."""
    g = H.stream(SHUF_ROLE, side)
    n = side * side
    ar = torch.arange(n)
    draws = 0
    while True:
        draws += 1
        p = torch.randperm(n, generator=g)
        if not bool((p == ar).any()):
            return p, draws


# --------------------------------------------------------------------------
# activation: ReLU + door H with a per-channel, per-feature or shuffled conv mean
# --------------------------------------------------------------------------

class FeatDoors(CR.ReLUDoors):
    """P181's ReLUDoors (ReLU, V, m, h_upd) with the conv sites' m per (channel x position) in
    the "feat" and "shuf" modes.  m[l] is (R, n_l): n = 16 ("chan"), 4096 / 1024 ("feat", "shuf")
    at c1 / c2, 100 at f1 / f2; the conv layout of m is (channel, y, x), C order."""

    def __init__(self, R: int, mode, device, beta: float = 0.01, beta_h: float = BETA_H):
        assert mode in (None, "chan", "feat", "shuf"), mode
        super().__init__(R, mode is not None, device, beta=beta, beta_h=beta_h)
        self.mode = mode
        self.perm = None
        self.perm_draws = None
        if mode in ("feat", "shuf"):
            for l in (0, 1):
                self.m[l] = torch.zeros(R, NFEAT[l], device=device)
        if mode == "shuf":
            got = [shuffle_perm(s) for s in SIDE]
            self.perm = [p.to(device) for p, _ in got]
            self.perm_draws = [d for _, d in got]

    @torch.no_grad()
    def update(self, zs) -> None:
        """P181's update, op by op (V record; door H's EMA from the step's own forward)."""
        for l, z in enumerate(zs):
            self.V[l].mul_(1 - self.beta).add_(self.beta * self.batch_var(z, l))
        if self.h_on:
            for l, z in enumerate(zs):
                bm = self.out_mean(z, l)
                old = self.m[l].clone()
                self.m[l].mul_(1 - self.beta_h).add_(bm, alpha=self.beta_h)
                self.hupd[:, l].add_((self.m[l] != old).any(1).to(self.hupd.dtype))

    @torch.no_grad()
    def out_mean(self, z: torch.Tensor, l: int) -> torch.Tensor:
        """(R, n_l) batch mean of the uncentred output the next layer reads.  "feat" / "shuf" conv
        sites: maxpool(ReLU(z)) averaged over the images only (one value per channel and position)."""
        if IS_CONV[l] and self.mode in ("feat", "shuf"):
            return F.max_pool2d(self.phi(z, l), POOL, POOL).mean(0).view(self.R, -1)
        return super().out_mean(z, l)

    def sub_map(self, l: int, m) -> torch.Tensor:
        """What conv site l's next layer loses, in the bundled (1, R*C, s, s) layout."""
        R, s = self.R, SIDE[l]
        if self.mode == "chan":
            return m[l].reshape(1, R * CH, 1, 1)
        if self.mode == "feat":
            return m[l].reshape(1, R * CH, s, s)
        if self.mode == "shuf":
            return m[l].view(R, CH, s * s)[:, :, self.perm[l]].reshape(1, R * CH, s, s)
        raise ValueError(self.mode)


# --------------------------------------------------------------------------
# forward with the doors
# --------------------------------------------------------------------------

def forward(P, xb: torch.Tensor, act: FeatDoors, m=None, with_inputs: bool = False):
    """xb (R, B, 3, 32, 32) -> (z1, a1, z2, a2, z3, a3, z4, a4, logits), a = the uncentred ReLU.

    P181's forward with the conv sites' subtraction taken from act.sub_map (per channel, per
    feature or shuffled).  `m` overrides act.m (the read-only m = 0 evaluation); `with_inputs`
    also returns the inputs of the four gated layers (x, P1, h, A3) as the next layers read them."""
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
        p1 = p1 - act.sub_map(0, m)
    z2 = F.conv2d(p1, Wc2.reshape(R * CH, CH, KER, KER), bc2.reshape(-1), padding=PAD, groups=R)
    a2 = act.phi(z2, 1)
    p2 = F.max_pool2d(a2, POOL, POOL)
    if hon:
        p2 = p2 - act.sub_map(1, m)
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
    """The engine's Bundle with ReLU, the doors and this file's forward (P181's constructor and
    step with FeatDoors; every other method inherited).  All slots share one arm."""

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
        self.act = FeatDoors(R, self.door["h"], device, beta=beta, beta_h=beta_h)
        self.m = [torch.zeros_like(p) for p in self.P]
        self.v = [torch.zeros_like(p) for p in self.P]
        self.tc = torch.zeros((), dtype=torch.float64, device=device)
        self.X = torch.stack([CN.images(cifar, CN.subset_idx(s), device) for s in seeds])
        self.shift = CR.input_shift(self.door["shift"], cifar)
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
# diagnostics (P181's spec §4 + the pool outputs' per-position mean, this spec §4)
# --------------------------------------------------------------------------

GAMMA = lambda k: k * 2.0 ** -24 / (1 - k * 2.0 ** -24)                 # noqa: E731
TF32_PROD = 2.0 ** -10                                                  # two operands, 2^-11 each


def dc_parts(mu: torch.Tensor, side: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """mu (R, C*side*side), (c, y, x) order -> (|delta|, |channel part spread over positions|, |mu|),
    mu[c, p] = mubar_c + delta[c, p] with mubar_c the mean over positions."""
    R = mu.shape[0]
    mc = mu.view(R, CH, side * side)
    cbar = mc.mean(2, keepdim=True)
    delta = (mc - cbar).reshape(R, -1)
    return delta.norm(dim=1), cbar.squeeze(2).norm(dim=1) * side, mu.norm(dim=1)


@torch.no_grad()
def diagnostics(P, X, Y, act: FeatDoors, chunk: int = 10, m0_eval: bool = True):
    """Per run: scalar columns (list of dicts) and per-unit arrays (dict of (R, n) numpy arrays).

    P181's diagnostics, op by op (zbar_j, sd_j, open fraction, death, the mean input patch / vector
    m_in, <W_j, m_in>, b_j, the identity residual and its rounding bound, |m_in| / rms deviation,
    the door state, m = 0 evaluation), plus for the inputs of c2 and f1 (the pool outputs) the
    per-feature image mean before (mu_pre) and after (mu_post) the door: pdc = |per-position part|,
    cdc = |channel part spread over the positions|, dc = |mu|, each over the rms deviation prms."""
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
    PS = [torch.zeros(R, n, **f64) for n in NFEAT]                    # sum of pre pool outputs
    QS = [torch.zeros(R, n, **f64) for n in NFEAT]                    # sum of what the next layer reads
    PQ = [torch.zeros(R, **f64) for _ in NFEAT]                       # sum of |pre|^2
    nz = [0] * 4
    ni = [0] * 4
    npo = 0
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
        for k in (0, 1):
            pre = F.max_pool2d(out[2 * k + 1], POOL, POOL).reshape(b, R, -1).double()   # (b, R, n), (c, y, x)
            PS[k] += pre.sum(0)
            PQ[k] += (pre * pre).sum((0, 2))
        QS[0] += ins[1].reshape(b, R, -1).double().sum(0)
        QS[1] += ins[2].double().sum(1)
        npo += b
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
    # the pool outputs read by c2 (k = 0) and f1 (k = 1): per-position mean before / after the door
    for k, tag in ((0, "c2"), (1, "f1")):
        mu_pre, mu_post = PS[k] / npo, QS[k] / npo
        sig = (PQ[k] / npo - (mu_pre * mu_pre).sum(1)).clamp_min(0).sqrt()
        pd_a, cd_a, dc_a = dc_parts(mu_pre, SIDE[k])
        pd_b, cd_b, dc_b = dc_parts(mu_post, SIDE[k])
        units[f"pmean_pre_{tag}"] = mu_pre.float().cpu().numpy()
        units[f"pmean_post_{tag}"] = mu_post.float().cpu().numpy()
        for r in range(R):
            s = float(sig[r])
            nrm = (lambda v: float(v) / s) if s > 0 else (lambda v: float("nan"))
            q = rows[r]
            q[f"prms_{tag}"] = s
            q[f"pdc_pre_{tag}"], q[f"pdc_post_{tag}"] = nrm(pd_a[r]), nrm(pd_b[r])
            q[f"cdc_pre_{tag}"], q[f"cdc_post_{tag}"] = nrm(cd_a[r]), nrm(cd_b[r])
            q[f"dc_pre_{tag}"], q[f"dc_post_{tag}"] = nrm(dc_a[r]), nrm(dc_b[r])
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
# task loop (P181's `run` with this file's bundle, evaluation and diagnostics)
# --------------------------------------------------------------------------

def save_units(path: Path, units: dict, n_tasks: int, R: int) -> None:
    CR.save_units(path, units, n_tasks, R)


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


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
            "n_tasks": n_tasks, "beta_h": BETA_H, "h_mode": ARMS[arm]["h"]}
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
    act = B.act
    prov = {"run_id": EXPERIMENT, "spec": SPEC, **meta,
            "door": ARMS[arm], "input_shift": None if B.shift is None else B.shift.tolist(),
            "conv_m_shape": [list(act.m[l].shape) for l in (0, 1)],
            "shuffle_perm": None if act.perm is None else [p.tolist() for p in act.perm],
            "shuffle_draws": act.perm_draws,
            "activation": "relu (clamp(z, min=0)); diagnostic gate z > 0",
            "host": "src/rlcifar_cnn_0908.py (init, subset, streams, net)",
            "engine": "src/cnn_center_feat_1011.py on src/cnn_center_relu_1011.py on src/sna_cnn_cause_1009.py Bundle"
                      + ("" if B.graph else " -- eager"),
            "engine_sha256": sha256_file(Path(__file__).resolve()),
            "parent_engine_sha256": sha256_file(PARENT_ENGINE),
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
