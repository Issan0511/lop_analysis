#!/usr/bin/env python3
"""cnn_drive_verify_1009: Codex's CNN drive-source theory, checked on real RL-CIFAR CNN checkpoints.

    python src/cnn_drive_verify_1009.py checks  --out results/cnn_drive_verify_1009/checks
    python src/cnn_drive_verify_1009.py measure --arms SNA,SNAc3,CV06FC3,CV3FC06 --seeds 10-19 \
        --tasks 1,5,10,20 --out results/cnn_drive_verify_1009/measure
    python src/cnn_drive_verify_1009.py frozen  ... (b2: frozen-parameter Adam reference)
    python src/cnn_drive_verify_1009.py replay  --task 1 ... (c: re-run task t+1 with the engine)

Read-only use of the sna_cnn_cause_1009 worktree (engine, host, checkpoints).  Every quantity of
(a)/(b) is computed in float64 on a re-implementation of the engine's forward pass (checked against
the engine in `checks`); the replays of (c) run the engine itself (float32, as trained).

Sign convention (Codex, proof.md sec. 1): G = E_new < grad m , grad L_new >.  G > 0 is the sinking
side: an SGD step moves the mean by -eta G in expectation (exactly for a first-conv mean).
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
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # shared GPU: keep the cache tight

SNA_ROOT = Path("/home/issan/Projects/claude/wt/sna_cnn_cause_1009")
sys.path.insert(0, str(SNA_ROOT))
sys.path.insert(1, str(SNA_ROOT / "analysis" / "sna_cnn_cause_1009"))

import numpy as np                                   # noqa: E402
import torch                                         # noqa: E402
import torch.nn.functional as F                      # noqa: E402

from src import pmnist_0905 as H                     # noqa: E402  (sna worktree, read only)
from src import rlcifar_cnn_0908 as CN               # noqa: E402
from src import sna_cnn_cause_1009 as E              # noqa: E402
import posthoc_lib as PL                             # noqa: E402

CKPT_DIR = SNA_ROOT / "results" / "sna_cnn_cause_1009" / "A" / "ckpt"
N, C, B16 = CN.N_IMAGES, CN.N_CLASSES, CN.BATCH
CH = CN.CHANNELS
LR, BETA1, BETA2, EPS = 1e-3, 0.9, 0.999, 1e-8
EPOCHS, SPE = 400, CN.STEPS_PER_EPOCH
LO, HI = 0.005, 3.0
# conv coordinates, packed in this order: W1 (1200) b1 (16) W2 (6400) b2 (16)
CONV_SIZES = (16 * 3 * 25, 16, 16 * 16 * 25, 16)
CONV_OFF = np.cumsum((0,) + CONV_SIZES)
NCONV = int(CONV_OFF[-1])                            # 7632
DT = torch.float64
# activation of the runs being measured, set in main() from the arms: "snake" | "leaky" | "relu"
ACT = {"kind": "snake", "slope": 0.0}
PIECEWISE = {"LR": 0.1, "R": 0.0}              # phase 2 arms (src/cnn_drive_verify_1009_lr.py)
OWN_RESULTS = Path(__file__).resolve().parents[1] / "results" / "cnn_drive_verify_1009"


def parse_list(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def ckpt_path(arm: str, seed: int, task: int) -> Path:
    if arm in PIECEWISE:
        return OWN_RESULTS / arm / "ckpt" / f"{arm}_seed{seed}_t{task:02d}.pt"
    return CKPT_DIR / f"{arm}_seed{seed}_t{task:02d}.pt"


def set_activation(arms) -> None:
    kinds = {a in PIECEWISE for a in arms}
    assert len(kinds) == 1, "one process measures either Snake arms or one piecewise arm"
    if arms[0] in PIECEWISE:
        assert len(set(arms)) == 1, arms
        ACT["kind"] = "relu" if PIECEWISE[arms[0]] == 0.0 else "leaky"
        ACT["slope"] = PIECEWISE[arms[0]]
    else:
        ACT["kind"], ACT["slope"] = "snake", 0.0


# --------------------------------------------------------------------------
# state
# --------------------------------------------------------------------------

class State:
    """One run at the end of a task, in float64 on `device`."""

    def __init__(self, arm: str, seed: int, task: int, device, dtype=DT):
        self.arm, self.seed, self.task, self.device = arm, seed, task, device
        if arm in PIECEWISE:
            assert ACT["kind"] != "snake", "set_activation() first"
        if task == 0:                                   # the host's init, V = 1
            P = CN.init_params(seed, torch.device("cpu"))
            sp = E.parse_arm(arm) if arm not in PIECEWISE else {"site": (("a", float("nan")),) * 4}
            self.P = [p.detach().to(device, dtype) for p in P]
            self.m = [torch.zeros_like(p) for p in self.P]
            self.v = [torch.zeros_like(p) for p in self.P]
            self.tc = 0.0
            self.alpha = []
            for l, (kind, val) in enumerate(sp["site"]):
                w = CN.WIDTHS[l]
                a = min(max(val, LO), HI) if kind == "c" else val       # NaN for LR / R
                self.alpha.append(torch.full((w,), float(a), dtype=dtype, device=device))
        else:
            st = torch.load(ckpt_path(arm, seed, task), map_location="cpu", weights_only=False)
            assert st["arm"] == arm and st["seed"] == seed
            self.P = [p.to(device, dtype) for p in st["P"]]
            self.m = [q.to(device, dtype) for q in st["m"]]
            self.v = [q.to(device, dtype) for q in st["v"]]
            self.tc = float(st["tc"])
            self.alpha = []
            for l in range(4):
                if arm in PIECEWISE:                       # no alpha
                    self.alpha.append(torch.full((CN.WIDTHS[l],), float("nan"), dtype=dtype, device=device))
                    continue
                V = st["act"]["V"][l].to(dtype)
                ada = st["act"]["ada"][l]
                cval = st["act"]["cval"][l].to(dtype)
                fixA = st["act"]["fixA"][l].to(dtype)
                a = torch.where(ada, (cval / V.sqrt()).clamp(LO, HI), fixA)
                self.alpha.append(a.to(device))
        self.X = images(seed, device, dtype)

    def conv_vec(self, tensors) -> torch.Tensor:
        return torch.cat([tensors[i].reshape(-1) for i in range(4)])


_IMG = {}


def images(seed: int, device, dtype) -> torch.Tensor:
    key = (seed, str(device), dtype)
    if key not in _IMG:
        _IMG[key] = CN.images(PL.cifar(), CN.subset_idx(seed), device).to(dtype)
    return _IMG[key]


def snake(z: torch.Tensor, a: torch.Tensor, conv: bool) -> torch.Tensor:
    """phi of the measured runs (name kept from phase 1): Snake with per-channel alpha, or the
    host's leaky (where(z > 0, z, 0.1 z)) / ReLU (clamp) -- alpha is ignored for those."""
    if ACT["kind"] == "relu":
        return torch.clamp(z, min=0.0)
    if ACT["kind"] == "leaky":
        return torch.where(z > 0, z, ACT["slope"] * z)
    a = a.view(1, -1, 1, 1) if conv else a.view(1, -1)
    return z + torch.sin(a * z) ** 2 * a.reciprocal()


def dphi_b(z: torch.Tensor, a_b: torch.Tensor) -> torch.Tensor:
    """phi'(z) with alpha already shaped to broadcast against z (alpha ignored for leaky / ReLU)."""
    if ACT["kind"] == "relu":
        return (z > 0).to(z.dtype)
    if ACT["kind"] == "leaky":
        return torch.where(z > 0, torch.ones_like(z), torch.full_like(z, ACT["slope"]))
    return 1.0 + torch.sin(2.0 * a_b * z)


def phi_gated(z: torch.Tensor, a: torch.Tensor, conv: bool, gate) -> torch.Tensor:
    """phi with the gate (z > 0 pattern) of a reference state held fixed: smooth in the
    parameters for leaky / ReLU (piecewise linear), identical to `snake` for Snake."""
    if gate is None or ACT["kind"] == "snake":
        return snake(z, a, conv)
    return torch.where(gate, z, ACT["slope"] * z)


def forward(P, x, al):
    """(z1, h1, z2, f): the engine's net for one run (conv -> phi -> pool)."""
    W1, b1, W2, b2, W3, b3, W4, b4, W5, b5 = P
    z1 = F.conv2d(x, W1, b1, padding=CN.PAD)
    h1 = F.max_pool2d(snake(z1, al[0], True), CN.POOL, CN.POOL)
    z2 = F.conv2d(h1, W2, b2, padding=CN.PAD)
    h2 = F.max_pool2d(snake(z2, al[1], True), CN.POOL, CN.POOL).flatten(1)
    z3 = h2 @ W3.T + b3
    z4 = snake(z3, al[2], False) @ W4.T + b4
    f = snake(z4, al[3], False) @ W5.T + b5
    return z1, h1, z2, f


def forward_routed(P, x, al, idx=None):
    """The same net with the two max-pools replaced by gathers at fixed winner indices and, for
    leaky / ReLU, the four gate patterns (z > 0) held fixed: `idx` = (i1, i2, g1, g2, g3, g4) from a
    reference state; None -> use and return the current ones.  Smooth in P in a neighbourhood:
    finite differences of this net check autograd without the kinks of winner switches or gate
    flips (for Snake the gates are not used)."""
    W1, b1, W2, b2, W3, b3, W4, b4, W5, b5 = P
    pw = ACT["kind"] != "snake"
    ref = idx is not None
    z1 = F.conv2d(x, W1, b1, padding=CN.PAD)
    g1 = idx[2] if ref else ((z1 > 0) if pw else None)
    a1 = phi_gated(z1, al[0], True, g1)
    if not ref:
        h1, i1 = F.max_pool2d(a1, CN.POOL, CN.POOL, return_indices=True)
    else:
        i1 = idx[0]
        h1 = a1.flatten(2).gather(2, i1.flatten(2)).view_as(i1)
    z2 = F.conv2d(h1, W2, b2, padding=CN.PAD)
    g2 = idx[3] if ref else ((z2 > 0) if pw else None)
    a2 = phi_gated(z2, al[1], True, g2)
    if not ref:
        h2, i2 = F.max_pool2d(a2, CN.POOL, CN.POOL, return_indices=True)
    else:
        i2 = idx[1]
        h2 = a2.flatten(2).gather(2, i2.flatten(2)).view_as(i2)
    h2 = h2.flatten(1)
    z3 = h2 @ W3.T + b3
    g3 = idx[4] if ref else ((z3 > 0) if pw else None)
    z4 = phi_gated(z3, al[2], False, g3) @ W4.T + b4
    g4 = idx[5] if ref else ((z4 > 0) if pw else None)
    f = phi_gated(z4, al[3], False, g4) @ W5.T + b5
    return f, (i1, i2, g1, g2, g3, g4)


def uniform_value_routed(P, al, X, Pref, chunk=300) -> float:
    """L_u at P with the pool winners of Pref."""
    tot = 0.0
    with torch.no_grad():
        for i0, i1 in chunks(X.shape[0], chunk):
            _, idx = forward_routed(Pref, X[i0:i1], al)
            f, _ = forward_routed(P, X[i0:i1], al, idx)
            tot += float(uniform_loss(f))
    return tot / X.shape[0]


def pool_gap_stats(P, al, X, chunk=300):
    """Smallest gap between the winner and the runner-up of each 2x2 window (in phi(z)), c1 / c2."""
    out = []
    with torch.no_grad():
        mins = [float("inf"), float("inf")]
        ties = [0, 0]
        for i0, i1 in chunks(X.shape[0], chunk):
            z1 = F.conv2d(X[i0:i1], P[0], P[1], padding=CN.PAD)
            a1 = snake(z1, al[0], True)
            h1 = F.max_pool2d(a1, CN.POOL, CN.POOL)
            z2 = F.conv2d(h1, P[2], P[3], padding=CN.PAD)
            a2 = snake(z2, al[1], True)
            for l, a in enumerate((a1, a2)):
                v = F.unfold(a.reshape(-1, 1, *a.shape[2:]), 2, stride=2).sort(1, descending=True).values
                gap = v[:, 0] - v[:, 1]
                mins[l] = min(mins[l], float(gap.min()))
                ties[l] += int((gap == 0).sum())
        out = {"min_gap_c1": mins[0], "min_gap_c2": mins[1], "exact_ties_c1": ties[0], "exact_ties_c2": ties[1]}
    return out


def self_params(P, al, layer: int, j: int):
    """Codex's literal self: delete every other channel of conv `layer` (and the matching
    downstream input columns), keep every other value, no refit."""
    W1, b1, W2, b2, W3, b3, W4, b4, W5, b5 = P
    if layer == 0:
        Q = [W1[j:j + 1], b1[j:j + 1], W2[:, j:j + 1], b2, W3, b3, W4, b4, W5, b5]
        bl = [al[0][j:j + 1], al[1], al[2], al[3]]
    else:
        Q = [W1, b1, W2[j:j + 1], b2[j:j + 1], W3[:, 64 * j:64 * (j + 1)], b3, W4, b4, W5, b5]
        bl = [al[0], al[1][j:j + 1], al[2], al[3]]
    return Q, bl


def uniform_loss(f: torch.Tensor) -> torch.Tensor:
    """E over iid uniform labels of the summed CE of these rows (exact: CE is linear in y)."""
    return (torch.logsumexp(f, 1) - f.mean(1)).sum()


def chunks(n: int, size: int):
    for i0 in range(0, n, size):
        yield i0, min(n, i0 + size)


# --------------------------------------------------------------------------
# (a) building blocks
# --------------------------------------------------------------------------

def channel_stats(S: State, chunk=300):
    """z-bar, sd of z per channel (c1, c2) over (images, positions); logits; p."""
    s1 = [torch.zeros(CH, dtype=DT, device=S.device) for _ in range(2)]
    s2 = [torch.zeros(CH, dtype=DT, device=S.device) for _ in range(2)]
    fs = []
    with torch.no_grad():
        for i0, i1 in chunks(N, chunk):
            z1, h1, z2, f = forward(S.P, S.X[i0:i1], S.alpha)
            for l, z in enumerate((z1, z2)):
                s1[l] += z.sum((0, 2, 3)); s2[l] += (z * z).sum((0, 2, 3))
            fs.append(f)
    f = torch.cat(fs)
    out = {}
    for l, (tag, hw) in enumerate((("c1", 32 * 32), ("c2", 16 * 16))):
        n = N * hw
        m = s1[l] / n
        out[f"zbar_{tag}"] = m
        out[f"zsd_{tag}"] = ((s2[l] / n - m * m).clamp_min(0) * n / (n - 1)).sqrt()
    out["f"] = f
    out["p"] = f.softmax(1)
    return out


def mean_grads(S: State, chunk=100, batched=True) -> torch.Tensor:
    """U (32, NCONV): gradient of every c1 / c2 channel mean w.r.t. the conv parameters.
    Rows 0..15 c1, 16..31 c2.  (fc components are exactly zero.)"""
    U = torch.zeros(2 * CH, NCONV, dtype=DT, device=S.device)
    Pc = [p.detach().clone().requires_grad_(True) for p in S.P[:4]]
    for i0, i1 in chunks(N, chunk):
        x = S.X[i0:i1]
        W1, b1, W2, b2 = Pc
        z1 = F.conv2d(x, W1, b1, padding=CN.PAD)
        h1 = F.max_pool2d(snake(z1, S.alpha[0], True), CN.POOL, CN.POOL)
        z2 = F.conv2d(h1, W2, b2, padding=CN.PAD)
        s1 = z1.sum((0, 2, 3)) / (N * 32 * 32)
        s2 = z2.sum((0, 2, 3)) / (N * 16 * 16)
        # c1: mean_j depends on W1[j], b1[j] only, so one backward of the sum gives every row
        g = torch.autograd.grad(s1.sum(), Pc[:2], retain_graph=True)
        for j in range(CH):
            U[j, CONV_OFF[0]:CONV_OFF[1]].view(CH, -1)[j] += g[0][j].reshape(-1)
            U[j, CONV_OFF[1] + j] += g[1][j]
        if batched:
            gb = torch.autograd.grad(s2, Pc, grad_outputs=torch.eye(CH, dtype=DT, device=S.device),
                                     is_grads_batched=True)
            for i in range(4):
                U[CH:, CONV_OFF[i]:CONV_OFF[i + 1]] += gb[i].reshape(CH, -1)
        else:
            for j in range(CH):
                g = torch.autograd.grad(s2[j], Pc, retain_graph=j < CH - 1)
                for i in range(4):
                    U[CH + j, CONV_OFF[i]:CONV_OFF[i + 1]] += g[i].reshape(-1)
    return U


def conv_jacobian(S: State, chunk=150) -> torch.Tensor:
    """J (N, C, NCONV): d logit_{n,c} / d conv parameters, per image and class.

    One backward per class with all images at once (images are independent: no batch-coupled
    layer), giving d f_{n,c}/d z1[n] and d f_{n,c}/d z2[n]; the weight Jacobians are the
    unfold products (exact).  Checked against single-image autograd in `checks` (K3)."""
    J = torch.empty(N, C, NCONV, dtype=DT, device=S.device)
    W1, b1, W2, b2 = S.P[:4]
    for i0, i1 in chunks(N, chunk):
        x = S.X[i0:i1]
        z1 = F.conv2d(x, W1, b1, padding=CN.PAD).requires_grad_(True)
        h1 = F.max_pool2d(snake(z1, S.alpha[0], True), CN.POOL, CN.POOL)
        z2 = F.conv2d(h1, W2, b2, padding=CN.PAD)
        P = S.P
        h2 = F.max_pool2d(snake(z2, S.alpha[1], True), CN.POOL, CN.POOL).flatten(1)
        z4 = snake(h2 @ P[4].T + P[5], S.alpha[2], False) @ P[6].T + P[7]
        f = snake(z4, S.alpha[3], False) @ P[8].T + P[9]
        ux = F.unfold(x, CN.KERNEL, padding=CN.PAD)                 # (b, 75, 1024)
        uh = F.unfold(h1.detach(), CN.KERNEL, padding=CN.PAD)       # (b, 400, 256)
        for c in range(C):
            d1, d2 = torch.autograd.grad(f[:, c].sum(), (z1, z2), retain_graph=c < C - 1)
            d1 = d1.flatten(2); d2 = d2.flatten(2)
            o = J[i0:i1, c]
            o[:, CONV_OFF[0]:CONV_OFF[1]] = torch.einsum("nos,nks->nok", d1, ux).reshape(i1 - i0, -1)
            o[:, CONV_OFF[1]:CONV_OFF[2]] = d1.sum(2)
            o[:, CONV_OFF[2]:CONV_OFF[3]] = torch.einsum("nos,nks->nok", d2, uh).reshape(i1 - i0, -1)
            o[:, CONV_OFF[3]:CONV_OFF[4]] = d2.sum(2)
    return J


def uniform_grad(P, al, X, wrt: list[int], chunk=300):
    """grad of L_u = (1/N) sum_n E_y CE_n w.r.t. P[i] for i in wrt; also L_u."""
    Pr = [p.detach().clone().requires_grad_(i in wrt) for i, p in enumerate(P)]
    gs = [torch.zeros_like(P[i]) for i in wrt]
    tot = 0.0
    for i0, i1 in chunks(X.shape[0], chunk):
        f = forward(Pr, X[i0:i1], al)[3]
        L = uniform_loss(f) / X.shape[0]
        g = torch.autograd.grad(L, [Pr[i] for i in wrt])
        for a, b in zip(gs, g):
            a += b
        tot += float(L)
    return gs, tot


def uniform_value(P, al, X, chunk=300) -> float:
    tot = 0.0
    with torch.no_grad():
        for i0, i1 in chunks(X.shape[0], chunk):
            tot += float(uniform_loss(forward(P, X[i0:i1], al)[3]))
    return tot / X.shape[0]


def g_self(S: State, U: torch.Tensor, layer: int, j: int) -> float:
    Q, bl = self_params(S.P, S.alpha, layer, j)
    u = U[layer * CH + j]
    if layer == 0:
        gs, _ = uniform_grad(Q, bl, S.X, [0, 1])
        return float((gs[0].reshape(-1) * u[CONV_OFF[0]:CONV_OFF[1]].view(CH, -1)[j]).sum()
                     + gs[1][0] * u[CONV_OFF[1]:CONV_OFF[2]][j])
    gs, _ = uniform_grad(Q, bl, S.X, [0, 1, 2, 3])
    return float((gs[0].reshape(-1) * u[CONV_OFF[0]:CONV_OFF[1]]).sum()
                 + (gs[1] * u[CONV_OFF[1]:CONV_OFF[2]]).sum()
                 + (gs[2].reshape(-1) * u[CONV_OFF[2]:CONV_OFF[3]].view(CH, -1)[j]).sum()
                 + gs[3][0] * u[CONV_OFF[3]:CONV_OFF[4]][j])


def g_self_bundled(S: State, U: torch.Tensor, chunk=100) -> np.ndarray:
    """All 32 literal-self G values at once (checked against `g_self`, which K1/K4 verified).

    c1: the 16 self networks share conv1 values; network r keeps channel r and feeds it to conv2
    through W2[:, r] -- a groups=16 conv.  Network r is the only user of W1[r], b1[r], so the
    gradient of the summed losses w.r.t. W1 / b1 separates by row.
    c2: network r keeps conv2 channel r (W2[r], b2[r], fc1 columns 64r:64r+64) on the full conv1.
    The conv1 parameters are shared, so their part of u_r . grad L_r is taken as
    < dL_r/dh1 , D h1[u_r] > with the input replicated per network (groups=16) and the tangent
    D h1[u_r] built explicitly (phi'(z1) * conv(x, u_r) routed through the pool winners)."""
    P, al = S.P, S.alpha
    W1, b1, W2, b2, W3, b3, W4, b4, W5, b5 = P
    out = np.zeros(2 * CH)
    # ---------------- c1
    gW1 = torch.zeros_like(W1); gb1 = torch.zeros_like(b1)
    W2g = W2.permute(1, 0, 2, 3).reshape(CH * CH, 1, CN.KERNEL, CN.KERNEL)   # group r = W2[:, r]
    b2g = b2.repeat(CH)
    a2g = al[1].repeat(CH)
    for i0, i1 in chunks(N, chunk):
        x = S.X[i0:i1]; B = i1 - i0
        W1r = W1.detach().clone().requires_grad_(True); b1r = b1.detach().clone().requires_grad_(True)
        z1 = F.conv2d(x, W1r, b1r, padding=CN.PAD)
        h1 = F.max_pool2d(snake(z1, al[0], True), CN.POOL, CN.POOL)
        z2 = F.conv2d(h1, W2g, b2g, padding=CN.PAD, groups=CH)                   # (B, 256, 16, 16)
        h2 = F.max_pool2d(snake(z2, a2g, True), CN.POOL, CN.POOL)               # (B, 256, 8, 8)
        h2 = h2.reshape(B, CH, CH * 64).transpose(0, 1)                         # (16 nets, B, 1024)
        z3 = h2 @ W3.T + b3
        z4 = snake(z3, al[2], False) @ W4.T + b4
        f = snake(z4, al[3], False) @ W5.T + b5                                  # (16, B, 10)
        L = (torch.logsumexp(f, 2) - f.mean(2)).sum() / N
        g = torch.autograd.grad(L, (W1r, b1r))
        gW1 += g[0]; gb1 += g[1]
    U1W = U[:CH, CONV_OFF[0]:CONV_OFF[1]]
    for r in range(CH):
        out[r] = float((gW1[r].reshape(-1) * U1W[r].view(CH, -1)[r]).sum() + gb1[r] * U[r, CONV_OFF[1] + r])
    # ---------------- c2
    gW2 = torch.zeros_like(W2); gb2 = torch.zeros_like(b2)
    gt = torch.zeros(CH, dtype=DT, device=S.device)                             # < dL_r/dh1 , Dh1[u_r] >
    UW1 = U[CH:, CONV_OFF[0]:CONV_OFF[1]].reshape(CH * CH, 3, CN.KERNEL, CN.KERNEL)  # (r, c1-ch) filters
    Ub1 = U[CH:, CONV_OFF[1]:CONV_OFF[2]].reshape(-1)                             # (r, c1-ch)
    W3g = torch.stack([W3[:, 64 * r:64 * (r + 1)] for r in range(CH)])          # (16, 100, 64)
    for i0, i1 in chunks(N, chunk):
        x = S.X[i0:i1]; B = i1 - i0
        with torch.no_grad():
            z1 = F.conv2d(x, W1, b1, padding=CN.PAD)
            a1 = snake(z1, al[0], True)
            h1, idx1 = F.max_pool2d(a1, CN.POOL, CN.POOL, return_indices=True)
            # tangent of h1 along u_r (only its W1 / b1 part moves conv1): (B, 16 nets, 16, 16, 16)
            dz1 = F.conv2d(x, UW1, Ub1, padding=CN.PAD).view(B, CH, CH, 32, 32)
            dphi = dphi_b(z1[:, None], al[0].view(1, 1, -1, 1, 1))
            T = (dphi * dz1).flatten(3).gather(3, idx1[:, None].flatten(3).expand(-1, CH, -1, -1))
            T = T.view(B, CH, CH, 16, 16)
        h1rep = h1.repeat(1, CH, 1, 1).requires_grad_(True)                    # (B, 256, 16, 16)
        W2r = W2.detach().clone().requires_grad_(True); b2r = b2.detach().clone().requires_grad_(True)
        z2 = F.conv2d(h1rep, W2r, b2r, padding=CN.PAD, groups=CH)               # (B, 16, 16, 16)
        h2 = F.max_pool2d(snake(z2, al[1], True), CN.POOL, CN.POOL)             # (B, 16, 8, 8)
        h2 = h2.reshape(B, CH, 64).transpose(0, 1)                               # (16, B, 64)
        z3 = torch.bmm(h2, W3g.transpose(1, 2)) + b3
        z4 = snake(z3, al[2], False) @ W4.T + b4
        f = snake(z4, al[3], False) @ W5.T + b5
        L = (torch.logsumexp(f, 2) - f.mean(2)).sum() / N
        gh, gw, gb = torch.autograd.grad(L, (h1rep, W2r, b2r))
        gW2 += gw; gb2 += gb
        gt += (gh.view(B, CH, CH, 16, 16) * T).sum((0, 2, 3, 4))
    U2W = U[CH:, CONV_OFF[2]:CONV_OFF[3]]
    for r in range(CH):
        out[CH + r] = float(gt[r] + (gW2[r].reshape(-1) * U2W[r].view(CH, -1)[r]).sum()
                            + gb2[r] * U[CH + r, CONV_OFF[3] + r])
    return out


def image_conditions(f: torch.Tensor, p: torch.Tensor, R: torch.Tensor) -> dict:
    """Codex's per-image sufficient conditions (ce_trained_state.md (3), (5), (8)) for every
    channel.  f, p: (N, C); R: (N, C, J) logit sensitivities.  Returns per-channel fractions."""
    Nn, Cc, Jn = R.shape
    gn = ((p - 1.0 / Cc)[:, :, None] * R).sum(1)                        # (N, J)
    order = f.argsort(1)                                                # ascending logits
    Rs = R.gather(1, order[:, :, None].expand(-1, -1, Jn))
    fs = f.gather(1, order)
    # (3): R co-sorted with f (non-decreasing along increasing f); ties in f count as fine
    dR = Rs[:, 1:] - Rs[:, :-1]
    df = (fs[:, 1:] - fs[:, :-1])[:, :, None]
    cond3 = ((dR >= 0) | (df == 0)).all(1)                              # (N, J)
    t = f.argmax(1)
    ft = f.gather(1, t[:, None])
    mask = torch.ones_like(f, dtype=torch.bool).scatter_(1, t[:, None], False)
    Delta = (ft - f.masked_fill(~mask, -float("inf"))).masked_fill(~mask, float("inf")).min(1).values
    Rt = R.gather(1, t[:, None, None].expand(-1, 1, Jn))                # (N, 1, J)
    dRt = (Rt - R)                                                      # (N, C, J)
    a = dRt.masked_fill(~mask[:, :, None], float("inf")).min(1).values  # (N, J)
    b = dRt.masked_fill(~mask[:, :, None], -float("inf")).max(1).values
    pt = p.gather(1, t[:, None])[:, 0]
    Bm = ((p - 1.0 / Cc).clamp_min(0) * mask).sum(1)
    cond5 = (Delta[:, None] > math.log(Cc - 1)) & (a > 0)
    cond8 = (a > 0) & (a * (pt[:, None] - 1.0 / Cc) > (b - a) * Bm[:, None])
    return {"g_pos": (gn > 0).double().mean(0), "cond3": cond3.double().mean(0),
            "cond5": cond5.double().mean(0), "cond8": cond8.double().mean(0),
            "cond5_all": cond5.all(0), "cond8_all": cond8.all(0), "cond3_all": cond3.all(0),
            "cond58_all": (cond5 | cond8).all(0), "gn_mean": gn.mean(0)}


# --------------------------------------------------------------------------
# (b) Adam
# --------------------------------------------------------------------------

def adam_dir(m0, v0, g, t):
    """Adam's descent direction A (step = -lr * A) for gradient g at global step t (host formula)."""
    m = BETA1 * m0 + (1 - BETA1) * g
    v = BETA2 * v0 + (1 - BETA2) * g * g
    c1 = 1 - BETA1 ** t
    c2 = 1 - BETA2 ** t
    return (m / c1) / ((v / c2).sqrt() + EPS), m, v


def next_batch_and_labels(seed: int, task: int):
    """The real first batch (16 indices) and labels of task `task`+1 (the engine's streams)."""
    g = H.stream("rlcc_batch", seed)
    for _ in range(task * EPOCHS):
        torch.randperm(N, generator=g)
    perm = torch.randperm(N, generator=g)
    return perm, PL.labels_at(seed, task + 1)


def first_step_mc(S, J, Jp, U, m0, v0, K: int, gen: torch.Generator, kchunk=512):
    """MC over (16-image batch, iid uniform labels) of the first Adam step's channel-mean change
    (first order, -lr * U @ A) and of SGD's (-U @ g).  Returns per-channel mean / SE, and the
    same with old momentum zeroed."""
    t = S.tc + 1
    Jf = J.reshape(N * C, NCONV)
    acc = {k: torch.zeros(2 * CH, dtype=DT, device=S.device) for k in
           ("a1", "a2", "s1", "s2", "z1", "z2")}
    for k0 in range(0, K, kchunk):
        k = min(kchunk, K - k0)
        idx = torch.rand(k, N, generator=gen).argsort(1)[:, :B16]            # uniform 16-subsets
        lab = torch.randint(C, (k, B16), generator=gen)
        idx_d, lab_d = idx.to(S.device), lab.to(S.device)
        flat = idx_d * C + lab_d
        g = torch.zeros(k, NCONV, dtype=DT, device=S.device)
        for i in range(B16):                                                  # no (k,16,P) temp
            g += Jp[idx_d[:, i]] - Jf[flat[:, i]]
        g /= B16
        A, _, _ = adam_dir(m0, v0, g, t)
        A0, _, _ = adam_dir(torch.zeros_like(m0), v0, g, t)
        da = -LR * (A @ U.T)                                                 # (k, 32)
        d0 = -LR * (A0 @ U.T)
        ds = -(g @ U.T)
        acc["a1"] += da.sum(0); acc["a2"] += (da * da).sum(0)
        acc["z1"] += d0.sum(0); acc["z2"] += (d0 * d0).sum(0)
        acc["s1"] += ds.sum(0); acc["s2"] += (ds * ds).sum(0)
    out = {}
    for key, (s1, s2) in {"adam": ("a1", "a2"), "adam_m0": ("z1", "z2"), "sgd": ("s1", "s2")}.items():
        mu = acc[s1] / K
        var = (acc[s2] / K - mu * mu).clamp_min(0) * K / (K - 1)
        out[key] = mu
        out[key + "_se"] = (var / K).sqrt()
    return out


# --------------------------------------------------------------------------
# measure: (a) + (b1) for one checkpoint
# --------------------------------------------------------------------------

def measure_one(arm, seed, task, device, K=65536, self_net=True, verbose=False):
    t0 = time.time()
    S = State(arm, seed, task, device)
    cs = channel_stats(S)
    U = mean_grads(S)
    gsf = g_self_bundled(S, U) if self_net else None                     # before J (memory)
    J = conv_jacobian(S)
    f, p = cs["f"], cs["p"]
    Jp = torch.einsum("nc,ncp->np", p, J)                                  # sum_c p_nc J_nc
    Jbar = J.mean(1)
    Eg = (Jp - Jbar).mean(0)                                                # exact E[g] (conv)
    G_J = U @ Eg                                                            # (32,) from J
    gs, Lu = uniform_grad(S.P, S.alpha, S.X, [0, 1, 2, 3])
    gconv = torch.cat([q.reshape(-1) for q in gs])
    G_ag = U @ gconv                                                        # (32,) autograd
    R = torch.einsum("ncp,jp->ncj", J, U)                                   # (N, C, 32)
    G_R = ((p - 1.0 / C)[:, :, None] * R).sum(1).mean(0)                    # identity (1)
    cond = image_conditions(f, p, R)
    res = {"arm": arm, "seed": seed, "task": task, "tc": S.tc, "L_u": Lu,
           "fit_conf": float(p.max(1).values.mean()),
           "zbar": torch.cat([cs["zbar_c1"], cs["zbar_c2"]]).cpu().numpy(),
           "zsd": torch.cat([cs["zsd_c1"], cs["zsd_c2"]]).cpu().numpy(),
           "alpha": torch.cat([S.alpha[0], S.alpha[1]]).cpu().numpy(),
           "G_full": G_ag.cpu().numpy(), "G_full_J": G_J.cpu().numpy(), "G_full_R": G_R.cpu().numpy(),
           "u_norm": U.norm(dim=1).cpu().numpy()}
    for k, v in cond.items():
        res["img_" + k] = v.cpu().numpy()
    # literal self (bundled; K7 checks it against the per-network computation)
    if self_net:
        res["G_self"] = gsf
    # (b1) first Adam step
    m0 = S.conv_vec(S.m); v0 = S.conv_vec(S.v)
    gen = torch.Generator().manual_seed(1009_000 + 1000 * seed + task)
    mc = first_step_mc(S, J, Jp, U, m0, v0, K, gen)
    for k, v in mc.items():
        res["fs_" + k] = v.cpu().numpy()
    res["fs_K"] = K
    # the realized first step of task+1 (actual batch, actual labels)
    perm, ylab = next_batch_and_labels(seed, task)
    bi = perm[:B16].to(device); yl = ylab.to(device)[bi]
    g_real = (Jp[bi] - J[bi, yl]).mean(0)
    A_real, _, _ = adam_dir(m0, v0, g_real, S.tc + 1)
    res["real_adam"] = (-LR * (U @ A_real)).cpu().numpy()
    res["real_sgd"] = (-(U @ g_real)).cpu().numpy()
    # B2: is the first step sign-like?  |A_k| relative to (1-b1)/sqrt(1-b2), per coordinate
    ref = (1 - BETA1) / math.sqrt(1 - BETA2)
    absA = A_real.abs() / ref
    res["real_absA_q"] = torch.quantile(absA, torch.tensor([.05, .25, .5, .75, .95], dtype=DT,
                                                          device=device)).cpu().numpy()
    res["real_absA_within20"] = float(((absA > 0.8) & (absA < 1.2)).double().mean())
    # how small is the old second moment against one new-task gradient?  per coordinate ratio
    Es2 = torch.zeros_like(Eg)                                              # E_{n,y}[g_n(y)^2]
    for c in range(C):
        Es2 += ((Jp - J[:, c]) ** 2).mean(0) / C
    Eg2 = Eg ** 2 + (Es2 - Eg ** 2) / B16                                   # ~E[g_k^2], 16-batch
    ratio = (BETA2 * v0 / ((1 - BETA2) * Eg2 + 1e-300)).sqrt()
    res["vold_ratio_q"] = torch.quantile(ratio, torch.tensor([.05, .5, .95], dtype=DT,
                                                             device=device)).cpu().numpy()
    # c2: exact (non-linear) change of the realized Adam step vs its first-order value
    with torch.no_grad():
        dconv = -LR * A_real
        P2 = [p.clone() for p in S.P]
        for i in range(4):
            P2[i] += dconv[CONV_OFF[i]:CONV_OFF[i + 1]].view_as(P2[i])
        S2 = State.__new__(State)
        S2.__dict__.update(S.__dict__)
        S2.P = P2
        cs2 = channel_stats(S2)
        res["real_adam_exact"] = (torch.cat([cs2["zbar_c1"], cs2["zbar_c2"]])
                                  - torch.cat([cs["zbar_c1"], cs["zbar_c2"]])).cpu().numpy()
    res["seconds"] = time.time() - t0
    if verbose:
        print(f"{arm} s{seed} t{task}: {res['seconds']:.1f}s", flush=True)
    del J, Jp, Jbar
    return res


# --------------------------------------------------------------------------
# checks (K1-K6)
# --------------------------------------------------------------------------

def engine_bundle(slots, device, graph=True):
    """The sna engine's Bundle for Snake arms, the phase-2 BundleLR (same engine) for LR / R."""
    if slots[0][0] in PIECEWISE:
        import cnn_drive_verify_1009_lr as LRM
        return LRM.BundleLR(slots, PL.cifar(), device, graph=graph)
    return E.Bundle(slots, PL.cifar(), device, graph=graph)


def load_engine_run(arm, seed, task, device):
    """One run in the engine at the end of `task` (posthoc_lib.load_run, for any arm)."""
    if arm not in PIECEWISE:
        return PL.load_run(ckpt_path(arm, seed, task), device=str(device), graph=False)[0]
    st = torch.load(ckpt_path(arm, seed, task), map_location="cpu", weights_only=False)
    B = engine_bundle([(arm, seed)], device, graph=False)
    with torch.no_grad():
        for dst, src in ((B.P, st["P"]), (B.m, st["m"]), (B.v, st["v"])):
            for q, x in zip(dst, src):
                q[0].copy_(x)
        B.tc.fill_(st["tc"])
        for k in ("V", "ada", "fixA", "cval"):
            for dst, src in zip(getattr(B.act, k), st["act"][k]):
                dst[0].copy_(src)
    B.Y.copy_(PL.labels_at(seed, task)[None])
    for _ in range(task * B.epochs):
        torch.randperm(CN.N_IMAGES, generator=B.g_batch[seed])
    for _ in range(task):
        CN.task_labels(B.g_lab[seed])
    return B


CHECK_STATES = {"checks": (("SNA", 10, 1), ("SNAc3", 11, 20)),
                "checks2": (("SNA", 10, 1), ("CV06FC3", 12, 10), ("SNAc3", 11, 20)),
                "checks3": (("SNA", 10, 1), ("SNAc3", 11, 20))}


def run_checks(out: Path, device):
    out.mkdir(parents=True, exist_ok=True)
    rep = {"checks": []}

    def add(name, ok, **kw):
        rep["checks"].append({"name": name, "pass": bool(ok), **kw})
        print(f"[{'PASS' if ok else 'FAIL'}] {name} {json.dumps(kw, default=float)[:300]}", flush=True)

    torch.backends.cudnn.allow_tf32 = False
    for arm, seed, task in CHECK_STATES["checks"]:
        S = State(arm, seed, task, device)
        # K0: our float64 forward vs the engine's forward (float32, eager, TF32 off) on 40 images
        Bnd = load_engine_run(arm, seed, task, device)
        with torch.no_grad():
            eng = E.forward(Bnd.P, Bnd.X[:, :40], Bnd.act)
            torch.backends.cudnn.enabled = False                 # native conv, plain fp32 sums
            eng_nc = E.forward(Bnd.P, Bnd.X[:, :40], Bnd.act)
            torch.backends.cudnn.enabled = True
            ours = forward(S.P, S.X[:40], S.alpha)
            P32 = [p.float() for p in S.P]; a32 = [a.float() for a in S.alpha]
            ours32 = forward(P32, S.X[:40].float(), a32)
        def rel(a, b):
            return float((a.double() - b.double()).abs().max() / b.double().abs().max())
        rec = dict(arm=arm, seed=seed, task=task,
                   engine_cudnn_vs_f64_logit=rel(eng[8][0], ours[3]), engine_cudnn_vs_f64_z2=rel(eng[2], ours[2]),
                   engine_cudnn_vs_f64_z1=rel(eng[0], ours[0]),
                   engine_native_vs_f64_logit=rel(eng_nc[8][0], ours[3]), engine_native_vs_f64_z2=rel(eng_nc[2], ours[2]),
                   engine_native_vs_f64_z1=rel(eng_nc[0], ours[0]),
                   ours_f32_vs_f64_logit=rel(ours32[3], ours[3]))
        rec.update(pool_gap_stats(S.P, S.alpha, S.X))
        # pass: the engine's fp32 forward with plain (non-cuDNN) convolutions equals our float64
        # net to float32 accuracy; the cuDNN-algorithm difference is reported, not judged
        add("K0_forward_matches_engine", rec["engine_native_vs_f64_logit"] < 1e-5
            and rec["engine_native_vs_f64_z2"] < 1e-5, **rec)
        U = mean_grads(S)
        gs, Lu = uniform_grad(S.P, S.alpha, S.X, [0, 1, 2, 3])
        gconv = torch.cat([q.reshape(-1) for q in gs])
        # K1: finite differences of L_u along u_j (full) and along u_j in the self network
        for row in (0, 5, 16, 21):
            u = U[row]
            G = float(u @ gconv)
            fds, fdr = [], []
            for hu in (1e-3, 1e-4, 1e-5):
                h = hu / float(u.norm())
                Pp = [p.clone() for p in S.P]; Pm = [p.clone() for p in S.P]
                for i in range(4):
                    d = u[CONV_OFF[i]:CONV_OFF[i + 1]].view_as(Pp[i]) * h
                    Pp[i] += d; Pm[i] -= d
                fds.append((uniform_value(Pp, S.alpha, S.X) - uniform_value(Pm, S.alpha, S.X)) / (2 * h))
                fdr.append((uniform_value_routed(Pp, S.alpha, S.X, S.P)
                            - uniform_value_routed(Pm, S.alpha, S.X, S.P)) / (2 * h))
            rel = min(abs(fd - G) / max(abs(G), 1e-300) for fd in fds)
            relr = min(abs(fd - G) / max(abs(G), 1e-300) for fd in fdr)
            # pass: autograd = finite differences of the same function on its smooth piece (fixed
            # pool winners); the free-routing value (kinks of winner switches) is reported
            add("K1_fd_full", relr < 1e-4, arm=arm, task=task, row=row, G=G, fd_fixed_routing=fdr,
                rel_fixed_routing=relr, fd_free_routing=fds, rel_free_routing=rel)
            layer, j = divmod(row, CH)
            Gs = g_self(S, U, layer, j)
            Q, bl = self_params(S.P, S.alpha, layer, j)
            fds, fdr = [], []
            for hu in (1e-3, 1e-4, 1e-5):
                h = hu / float(u.norm())
                Qp = [q.clone() for q in Q]; Qm = [q.clone() for q in Q]
                if layer == 0:
                    for i in range(2):
                        d = u[CONV_OFF[i]:CONV_OFF[i + 1]].view(CH, -1)[j] if i == 0 else u[CONV_OFF[1] + j:CONV_OFF[1] + j + 1]
                        Qp[i] += d.view_as(Qp[i]) * h; Qm[i] -= d.view_as(Qm[i]) * h
                else:
                    for i in range(2):
                        d = u[CONV_OFF[i]:CONV_OFF[i + 1]].view_as(Qp[i]) * h
                        Qp[i] += d; Qm[i] -= d
                    d2 = u[CONV_OFF[2]:CONV_OFF[3]].view(CH, -1)[j].view_as(Qp[2]) * h
                    Qp[2] += d2; Qm[2] -= d2
                    d3 = u[CONV_OFF[3] + j:CONV_OFF[3] + j + 1] * h
                    Qp[3] += d3; Qm[3] -= d3
                fds.append((uniform_value(Qp, bl, S.X) - uniform_value(Qm, bl, S.X)) / (2 * h))
                fdr.append((uniform_value_routed(Qp, bl, S.X, Q)
                            - uniform_value_routed(Qm, bl, S.X, Q)) / (2 * h))
            rel = min(abs(fd - Gs) / max(abs(Gs), 1e-300) for fd in fds)
            relr = min(abs(fd - Gs) / max(abs(Gs), 1e-300) for fd in fdr)
            add("K1_fd_self", relr < 1e-4, arm=arm, task=task, row=row, G_self=Gs,
                fd_fixed_routing=fdr, rel_fixed_routing=relr, fd_free_routing=fds, rel_free_routing=rel)
            # K4: self by zeroing the other channels instead of slicing
            Pz = [p.clone() for p in S.P]
            keep = torch.zeros(CH, dtype=torch.bool, device=device); keep[j] = True
            if layer == 0:
                Pz[0][~keep] = 0; Pz[1][~keep] = 0
            else:
                Pz[2][~keep] = 0; Pz[3][~keep] = 0
            gz, _ = uniform_grad(Pz, S.alpha, S.X, [0, 1, 2, 3])
            gzc = torch.cat([q.reshape(-1) for q in gz])
            Gz = float(u @ gzc)
            add("K4_self_zero_equals_slice", abs(Gz - Gs) <= 1e-9 * max(1.0, abs(Gs)), arm=arm,
                task=task, row=row, slice=Gs, zero=Gz)
        # K2: autograd vs J-identity
        J = conv_jacobian(S)
        cs = channel_stats(S)
        p = cs["p"]
        Jp = torch.einsum("nc,ncp->np", p, J)
        Eg = (Jp - J.mean(1)).mean(0)
        gdiff = float((Eg - gconv).abs().max() / gconv.abs().max())
        add("K2_J_expected_grad_equals_autograd", gdiff < 1e-10, arm=arm, task=task, rel=gdiff)
        # K3: J rows vs single-image autograd (20 random (n, c))
        gen = torch.Generator().manual_seed(7)
        worst = 0.0
        Pr = [q.detach().clone().requires_grad_(True) for q in S.P]
        for _ in range(20):
            n = int(torch.randint(N, (1,), generator=gen)); c = int(torch.randint(C, (1,), generator=gen))
            fn = forward(Pr, S.X[n:n + 1], S.alpha)[3][0, c]
            gg = torch.autograd.grad(fn, Pr[:4])
            ref = torch.cat([q.reshape(-1) for q in gg])
            worst = max(worst, float((ref - J[n, c]).abs().max() / ref.abs().max()))
        add("K3_per_image_jacobian", worst < 1e-10, arm=arm, task=task, worst_rel=worst)
        # K5: our first Adam step vs the engine's step on the same batch / labels (TF32 off)
        perm, ylab = next_batch_and_labels(seed, task)
        Bnd.Y.copy_(ylab[None].to(device))
        Bnd.order.copy_(perm[None].to(device))
        before = [q.detach().clone() for q in Bnd.P[:4]]
        Bnd.step(0)
        eng_d = torch.cat([(q.detach()[0] - b[0]).double().reshape(-1) for q, b in zip(Bnd.P[:4], before)])
        bi = perm[:B16].to(device); yl = ylab.to(device)[bi]
        g_real = (Jp[bi] - J[bi, yl]).mean(0)
        A, _, _ = adam_dir(S.conv_vec(S.m), S.conv_vec(S.v), g_real, S.tc + 1)
        ours_d = -LR * A
        num = (eng_d - ours_d).abs()
        rel = float(num.max() / ours_d.abs().max())
        agree = float((torch.sign(eng_d) == torch.sign(ours_d)).double().mean())
        dm_eng = U @ eng_d
        dm_ours = U @ ours_d
        rel_dm = float((dm_eng - dm_ours).abs().max() / dm_ours.abs().max())
        add("K5_adam_step_equals_engine", rel_dm < 1e-4, arm=arm, task=task, rel_channel_dm=rel_dm,
            rel_max_coord=rel, coord_sign_agree=agree, note="engine float32 (TF32 off) vs ours float64")
        # K6: MC SGD mean vs exact
        U2 = U
        gen2 = torch.Generator().manual_seed(11)
        mc = first_step_mc(S, J, Jp, U2, S.conv_vec(S.m), S.conv_vec(S.v), 16384, gen2)
        z = ((mc["sgd"] - (-(U2 @ Eg))) / mc["sgd_se"]).abs().max()
        add("K6_mc_sgd_unbiased", float(z) < 4.5, arm=arm, task=task, max_abs_z=float(z))
        del J, Jp, Bnd
        torch.cuda.empty_cache()
    rep["all_pass"] = all(c["pass"] for c in rep["checks"])
    rep["cuda_max_mem_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20
    (out / "checks.json").write_text(json.dumps(rep, indent=1, default=float))
    print("all_pass", rep["all_pass"], "max mem MB", rep["cuda_max_mem_mb"])


def c2_exact_one(arm, seed, task, device, K=512, kb=8, chunk=200):
    """Exact (non-linear) expected change of every c1 / c2 channel mean under the first Adam step,
    MC over (16-image batch, iid labels).  Each sampled step's conv parameters are applied and the
    channel means recomputed; kb samples share one grouped forward.  float64 (TF32 off).  The
    linear prediction -lr u.A is computed on the same samples, so the paired difference
    exact - linear (the non-linear part) has a small SE even for modest K.
    alpha is held at the checkpoint (the parameter step only)."""
    S = State(arm, seed, task, device)
    cs = channel_stats(S)
    U = mean_grads(S)
    J = conv_jacobian(S)
    p = cs["p"]
    Jp = torch.einsum("nc,ncp->np", p, J)
    Jf = J.reshape(N * C, NCONV)
    m0 = S.conv_vec(S.m); v0 = S.conv_vec(S.v)
    t = S.tc + 1
    gen = torch.Generator().manual_seed(3009_000 + 1000 * seed + task)
    W1, b1, W2, b2 = S.P[:4]
    a0 = S.alpha[0]
    X = S.X
    cud = torch.backends.cudnn.enabled

    def means(W1s, b1s, W2s, b2s, k):
        """channel means (k, 32) for k parameter sets stacked on the channel axis."""
        s = torch.zeros(k, 2 * CH, dtype=DT, device=device)
        for i0, i1 in chunks(N, chunk):
            z1 = F.conv2d(X[i0:i1], W1s, b1s, padding=CN.PAD)                     # (b, k*16, 32, 32)
            h1 = F.max_pool2d(snake(z1, a0.repeat(k), True), CN.POOL, CN.POOL)
            z2 = F.conv2d(h1, W2s, b2s, padding=CN.PAD, groups=k)                 # (b, k*16, 16, 16)
            s[:, :CH] += z1.double().sum((0, 2, 3)).view(k, CH)
            s[:, CH:] += z2.double().sum((0, 2, 3)).view(k, CH)
        s[:, :CH] /= N * 32 * 32
        s[:, CH:] /= N * 16 * 16
        return s

    with torch.no_grad():
        base = means(W1, b1, W2, b2, 1)[0]
        acc = {k: torch.zeros(2 * CH, dtype=DT, device=device) for k in ("e1", "e2", "l1", "l2", "d1", "d2")}
        for k0 in range(0, K, kb):
            k = min(kb, K - k0)
            idx = torch.rand(k, N, generator=gen).argsort(1)[:, :B16].to(device)
            lab = torch.randint(C, (k, B16), generator=gen).to(device)
            g = torch.zeros(k, NCONV, dtype=DT, device=device)
            for i in range(B16):
                g += Jp[idx[:, i]] - Jf[idx[:, i] * C + lab[:, i]]
            g /= B16
            A, _, _ = adam_dir(m0, v0, g, t)
            d = (-LR * A)                                                          # (k, NCONV)
            lin = d @ U.T                                                          # (k, 32)
            dW1 = d[:, CONV_OFF[0]:CONV_OFF[1]].reshape(k * CH, 3, CN.KERNEL, CN.KERNEL)
            db1 = d[:, CONV_OFF[1]:CONV_OFF[2]].reshape(-1)
            dW2 = d[:, CONV_OFF[2]:CONV_OFF[3]].reshape(k * CH, CH, CN.KERNEL, CN.KERNEL)
            db2 = d[:, CONV_OFF[3]:CONV_OFF[4]].reshape(-1)
            mm = means(W1.repeat(k, 1, 1, 1) + dW1, b1.repeat(k) + db1,
                       W2.repeat(k, 1, 1, 1) + dW2, b2.repeat(k) + db2, k) - base
            dd = mm - lin
            for key, v in (("e", mm), ("l", lin), ("d", dd)):
                acc[key + "1"] += v.sum(0); acc[key + "2"] += (v * v).sum(0)
    torch.backends.cudnn.enabled = cud

    def ms(key):
        mu = acc[key + "1"] / K
        se = ((acc[key + "2"] / K - mu * mu).clamp_min(0) * K / (K - 1) / K).sqrt()
        return mu.cpu().numpy(), se.cpu().numpy()
    em, es = ms("e"); lm, ls = ms("l"); dm, ds = ms("d")
    return {"arm": arm, "seed": seed, "task": task, "K": K, "exact_mean": em, "exact_se": es,
            "linear_mean": lm, "linear_se": ls, "nonlin_mean": dm, "nonlin_se": ds}


def self_shape(S: State, U: torch.Tensor, chunk=100) -> dict:
    """Codex's self shape S_c = <R_c, H_c>_F (proof.md sec. 0-1) for every c1 / c2 channel:
    H = the channel's pooled features (after phi and max-pool), R = their exact derivative along the
    channel-mean direction u (forward-mode tangent through the current pool winners).  Also the
    image-centred version (the constant per feature coordinate removed, as an intercept would).
    For ReLU, H >= 0 and R >= 0, so S > 0 whenever the channel is alive; Snake allows H < 0."""
    P, al = S.P, S.alpha
    W1, b1, W2, b2 = P[:4]
    acc = {k: torch.zeros(2 * CH, dtype=DT, device=S.device) for k in ("RH", "R", "H", "RR", "HH")}
    sumR = torch.zeros(CH, 16 * 16, dtype=DT, device=S.device)
    sumH = torch.zeros(CH, 16 * 16, dtype=DT, device=S.device)
    sumR2 = torch.zeros(CH, 64, dtype=DT, device=S.device)
    sumH2 = torch.zeros(CH, 64, dtype=DT, device=S.device)
    U1W = U[:CH, CONV_OFF[0]:CONV_OFF[1]].view(CH, CH, 3, CN.KERNEL, CN.KERNEL)
    U1b = U[:CH, CONV_OFF[1]:CONV_OFF[2]]
    UW1 = U[CH:, CONV_OFF[0]:CONV_OFF[1]].reshape(CH * CH, 3, CN.KERNEL, CN.KERNEL)
    Ub1 = U[CH:, CONV_OFF[1]:CONV_OFF[2]].reshape(-1)
    UW2 = U[CH:, CONV_OFF[2]:CONV_OFF[3]].view(CH, CH, CH, CN.KERNEL, CN.KERNEL)
    Ub2 = U[CH:, CONV_OFF[3]:CONV_OFF[4]]
    with torch.no_grad():
        for i0, i1 in chunks(N, chunk):
            x = S.X[i0:i1]; B = i1 - i0
            z1 = F.conv2d(x, W1, b1, padding=CN.PAD)
            a1 = snake(z1, al[0], True)
            h1, idx1 = F.max_pool2d(a1, CN.POOL, CN.POOL, return_indices=True)
            dphi1 = dphi_b(z1, al[0].view(1, -1, 1, 1))
            # c1 channel j: u_j moves only W1[j], b1[j]
            Wd = torch.stack([U1W[j, j] for j in range(CH)])                      # (16, 3, 5, 5)
            bd = torch.stack([U1b[j, j] for j in range(CH)])
            dz1 = F.conv2d(x, Wd, bd, padding=CN.PAD)                             # (B, 16, 32, 32)
            R1 = (dphi1 * dz1).flatten(2).gather(2, idx1.flatten(2))              # (B, 16, 256)
            H1 = h1.flatten(2)
            acc["RH"][:CH] += (R1 * H1).sum((0, 2))
            sumR += R1.sum(0); sumH += H1.sum(0)
            # c2 channel j: tangent of h1 along u_j's W1 / b1 part, then conv2 row j + its own part
            z2 = F.conv2d(h1, W2, b2, padding=CN.PAD)
            h2, idx2 = F.max_pool2d(snake(z2, al[1], True), CN.POOL, CN.POOL, return_indices=True)
            dphi2 = dphi_b(z2, al[1].view(1, -1, 1, 1))
            dz1r = F.conv2d(x, UW1, Ub1, padding=CN.PAD).view(B, CH, CH, 32, 32)  # (B, j, c1ch, ...)
            Th1 = (dphi1[:, None] * dz1r).flatten(3).gather(
                3, idx1[:, None].flatten(3).expand(-1, CH, -1, -1)).view(B * CH, CH, 16, 16)
            for j in range(CH):
                dz2 = (F.conv2d(Th1.view(B, CH, CH, 16, 16)[:, j], W2[j:j + 1], None, padding=CN.PAD)
                       + F.conv2d(h1, UW2[j][j:j + 1], None, padding=CN.PAD)
                       + Ub2[j, j])                                                  # (B, 1, 16, 16)
                R2 = (dphi2[:, j:j + 1] * dz2).flatten(2).gather(2, idx2[:, j:j + 1].flatten(2))[:, 0]
                H2 = h2[:, j].flatten(1)
                acc["RH"][CH + j] += (R2 * H2).sum()
                sumR2[j] += R2.sum(0); sumH2[j] += H2.sum(0)
    S0 = acc["RH"].clone()
    # centred: sum_n (R - Rbar)(H - Hbar) = sum RH - N Rbar.Hbar per coordinate
    S1 = S0.clone()
    S1[:CH] -= (sumR * sumH).sum(1) / N
    S1[CH:] -= (sumR2 * sumH2).sum(1) / N
    return {"S0": S0.cpu().numpy(), "S1": S1.cpu().numpy()}


def adam_decomposition(S: State, J, Jp, U, m0, v0, K: int, gen, kchunk=512) -> dict:
    """Codex adam_state.md sec. 2, per channel u:
        E[u.A] = kappa sum_j u_j [ a_j E h_j + b mu_j E h_j + b Cov(g_j, h_j) ]
    with a_j = beta1 m_j-, b = 1 - beta1, h_j(g) = 1/(sqrt(beta2 v_j- + (1-beta2) g^2)/sqrt(c2) ... )
    written here in the host's bias-corrected form A_j = (m_j/c1)/(sqrt(v_j/c2) + eps), so
    E[A_j] = (1/c1)[ a_j E h_j + b E(g_j h_j) ], h_j = 1/(sqrt(v_j/c2) + eps), and
    E(g h) = mu E h + Cov(g, h).  Returns the three projected parts (momentum, mean-scaling, cov)
    and their sum (= the Adam expectation, first order), MC over (batch, labels)."""
    t = S.tc + 1
    c1 = 1 - BETA1 ** t; c2 = 1 - BETA2 ** t
    Jf = J.reshape(N * C, NCONV)
    sh = torch.zeros(NCONV, dtype=DT, device=S.device)
    sgh = torch.zeros(NCONV, dtype=DT, device=S.device)
    for k0 in range(0, K, kchunk):
        k = min(kchunk, K - k0)
        idx = torch.rand(k, N, generator=gen).argsort(1)[:, :B16].to(S.device)
        lab = torch.randint(C, (k, B16), generator=gen).to(S.device)
        g = torch.zeros(k, NCONV, dtype=DT, device=S.device)
        for i in range(B16):
            g += Jp[idx[:, i]] - Jf[idx[:, i] * C + lab[:, i]]
        g /= B16
        v = BETA2 * v0 + (1 - BETA2) * g * g
        h = 1.0 / ((v / c2).sqrt() + EPS)
        sh += h.sum(0); sgh += (g * h).sum(0)
    Eh = sh / K; Egh = sgh / K
    mu = (Jp - J.mean(1)).mean(0)
    a = BETA1 * m0; b = 1 - BETA1
    mom = (U @ (a * Eh)) / c1
    meanscale = (U @ (b * mu * Eh)) / c1
    cov = (U @ (b * (Egh - mu * Eh))) / c1
    # meanscale = diagonal-preconditioned SGD (coordinate scaling E h_j only, no noise correlation)
    return {"dec_mom": (-LR * mom).cpu().numpy(), "dec_meanscale": (-LR * meanscale).cpu().numpy(),
            "dec_cov": (-LR * cov).cpu().numpy(),
            "dec_total": (-LR * (mom + meanscale + cov)).cpu().numpy()}


def extra_one(arm, seed, task, device, K=16384):
    S = State(arm, seed, task, device)
    U = mean_grads(S)
    ss = self_shape(S, U)
    cs = channel_stats(S)
    J = conv_jacobian(S)
    Jp = torch.einsum("nc,ncp->np", cs["p"], J)
    gen = torch.Generator().manual_seed(5009_000 + 1000 * seed + task)
    dec = adam_decomposition(S, J, Jp, U, S.conv_vec(S.m), S.conv_vec(S.v), K, gen)
    return {"arm": arm, "seed": seed, "task": task, **ss, **dec}


def pooled_features(P, al, X, idx=None, chunk=200):
    """(N, 16, 256) c1 and (N, 16, 64) c2 pooled features; fixed winners (and, for leaky / ReLU,
    fixed c1 / c2 gates) if idx is given."""
    H1s, H2s, I1, I2, G1, G2 = [], [], [], [], [], []
    pw = ACT["kind"] != "snake"
    with torch.no_grad():
        for i0, i1 in chunks(N, chunk):
            x = X[i0:i1]
            z1 = F.conv2d(x, P[0], P[1], padding=CN.PAD)
            g1 = idx[2][i0:i1] if (idx is not None and pw) else ((z1 > 0) if pw else None)
            a1 = phi_gated(z1, al[0], True, g1)
            if idx is None:
                h1, j1 = F.max_pool2d(a1, CN.POOL, CN.POOL, return_indices=True)
            else:
                j1 = idx[0][i0:i1]; h1 = a1.flatten(2).gather(2, j1.flatten(2)).view_as(j1)
            z2 = F.conv2d(h1, P[2], P[3], padding=CN.PAD)
            g2 = idx[3][i0:i1] if (idx is not None and pw) else ((z2 > 0) if pw else None)
            a2 = phi_gated(z2, al[1], True, g2)
            if idx is None:
                h2, j2 = F.max_pool2d(a2, CN.POOL, CN.POOL, return_indices=True)
            else:
                j2 = idx[1][i0:i1]; h2 = a2.flatten(2).gather(2, j2.flatten(2)).view_as(j2)
            H1s.append(h1.flatten(2)); H2s.append(h2.flatten(2)); I1.append(j1); I2.append(j2)
            G1.append(g1); G2.append(g2)
    gates = (torch.cat(G1), torch.cat(G2)) if pw else (None, None)
    return torch.cat(H1s), torch.cat(H2s), (torch.cat(I1), torch.cat(I2), *gates)


def run_checks_selfshape(out: Path, device):
    """K8: self_shape's forward-mode tangent equals central differences of the pooled features
    with the winners held fixed."""
    torch.backends.cudnn.allow_tf32 = False
    rep = {"checks": []}
    for arm, seed, task in CHECK_STATES["checks3"]:
        S = State(arm, seed, task, device)
        U = mean_grads(S)
        ss = self_shape(S, U)
        H1, H2, idx = pooled_features(S.P, S.alpha, S.X)
        worst = 0.0
        S0fd = np.zeros(2 * CH); S1fd = np.zeros(2 * CH)
        for row in range(2 * CH):
            u = U[row]
            h = 1e-5 / float(u.norm())
            Pp = [p.clone() for p in S.P]; Pm = [p.clone() for p in S.P]
            for i in range(4):
                d = u[CONV_OFF[i]:CONV_OFF[i + 1]].view_as(Pp[i]) * h
                Pp[i] += d; Pm[i] -= d
            A1, A2, _ = pooled_features(Pp, S.alpha, S.X, idx)
            B1, B2, _ = pooled_features(Pm, S.alpha, S.X, idx)
            layer, j = divmod(row, CH)
            if layer == 0:
                R = (A1[:, j] - B1[:, j]) / (2 * h); Hh = H1[:, j]
            else:
                R = (A2[:, j] - B2[:, j]) / (2 * h); Hh = H2[:, j]
            S0fd[row] = float((R * Hh).sum())
            S1fd[row] = float(((R - R.mean(0)) * (Hh - Hh.mean(0))).sum())
        r0 = float(np.abs(S0fd - ss["S0"]).max() / np.abs(S0fd).max())
        r1 = float(np.abs(S1fd - ss["S1"]).max() / np.abs(S1fd).max())
        sg = float((np.sign(S0fd) == np.sign(ss["S0"])).mean())
        rec = dict(name="K8_selfshape_tangent_equals_fd", arm=arm, seed=seed, task=task,
                   pass_=r0 < 1e-6 and r1 < 1e-6, rel_S0=r0, rel_S1=r1, sign_agree=sg)
        rep["checks"].append(rec)
        print(json.dumps(rec, default=float), flush=True)
    rep["all_pass"] = all(c["pass_"] for c in rep["checks"])
    (out / "checks_selfshape.json").write_text(json.dumps(rep, indent=1, default=float))
    print("all_pass", rep["all_pass"], flush=True)


def run_checks_bundle(out: Path, device):
    """K7: the vectorised mean gradients and the bundled literal-self G equal the per-channel
    computations (which K1 / K4 verified against finite differences)."""
    torch.backends.cudnn.allow_tf32 = False
    rep = {"checks": []}
    for arm, seed, task in CHECK_STATES["checks2"]:
        S = State(arm, seed, task, device)
        t0 = time.time(); Ub = mean_grads(S, batched=True); tb = time.time() - t0
        t0 = time.time(); Ul = mean_grads(S, batched=False); tl = time.time() - t0
        du = float((Ub - Ul).abs().max() / Ul.abs().max())
        t0 = time.time(); gb = g_self_bundled(S, Ul); tsb = time.time() - t0
        t0 = time.time()
        gl = np.array([g_self(S, Ul, layer, j) for layer in range(2) for j in range(CH)])
        tsl = time.time() - t0
        dg = float(np.abs(gb - gl).max() / np.abs(gl).max())
        signs = float((np.sign(gb) == np.sign(gl)).mean())
        ok = du < 1e-12 and dg < 1e-10 and signs == 1.0
        rec = dict(name="K7_bundled_equals_loop", arm=arm, seed=seed, task=task, pass_=ok,
                   mean_grad_rel=du, gself_rel=dg, gself_sign_agree=signs,
                   sec_batched_U=tb, sec_loop_U=tl, sec_bundled_self=tsb, sec_loop_self=tsl)
        rep["checks"].append(rec)
        print(json.dumps(rec, default=float), flush=True)
    rep["all_pass"] = all(c["pass_"] for c in rep["checks"])
    (out / "checks_bundle.json").write_text(json.dumps(rep, indent=1, default=float))
    print("all_pass", rep["all_pass"], flush=True)


# --------------------------------------------------------------------------
# frozen-parameter Adam reference (b2)
# --------------------------------------------------------------------------

def frozen_one(arm, seed, task, device, L=32, S_steps=3000, rec=(1, 10, 75, 750, 3000)):
    """Parameters frozen at the checkpoint; the task's labels drawn once and reused; a fresh
    randperm every epoch; Adam's m / v advance.  Draw 0 is the real next task (real labels and
    the engine's batch orders).  Returns the cumulative -lr * sum_s U @ A_s at `rec` steps and
    the SGD counterpart -sum_s U @ g_s, per draw."""
    S = State(arm, seed, task, device)
    cs = channel_stats(S)
    U = mean_grads(S)
    J = conv_jacobian(S)
    p = cs["p"]
    Jp = torch.einsum("nc,ncp->np", p, J)
    Jf = J.reshape(N * C, NCONV)
    del J
    Eg = (Jp - Jf.view(N, C, NCONV).mean(1)).mean(0)
    G = U @ Eg
    m = S.conv_vec(S.m)[None].repeat(L + 1, 1)
    v = S.conv_vec(S.v)[None].repeat(L + 1, 1)
    gen = torch.Generator().manual_seed(2009_000 + 1000 * seed + task)
    ylab = torch.randint(C, (L + 1, N), generator=gen)
    perm0, yreal = next_batch_and_labels(seed, task)
    ylab[0] = yreal
    gb = H.stream("rlcc_batch", seed)
    for _ in range(task * EPOCHS):
        torch.randperm(N, generator=gb)
    ylab_d = ylab.to(device)
    cumA = torch.zeros(L + 1, 2 * CH, dtype=DT, device=device)
    cumS = torch.zeros(L + 1, 2 * CH, dtype=DT, device=device)
    tail = torch.zeros(L + 1, 2 * CH, dtype=DT, device=device)
    rec_out = {}
    ar = torch.arange(L + 1, device=device)[:, None]
    s = 0
    n_epochs = math.ceil(S_steps / SPE)
    for e in range(n_epochs):
        orders = torch.empty(L + 1, N, dtype=torch.long)
        orders[0] = torch.randperm(N, generator=gb)
        orders[1:] = torch.rand(L, N, generator=gen).argsort(1)
        orders = orders.to(device)
        for jb in range(SPE):
            s += 1
            if s > S_steps:
                break
            bi = orders[:, jb * B16:(jb + 1) * B16]                      # (L+1, 16)
            yl = ylab_d[ar, bi]
            g = (Jp[bi] - Jf[bi * C + yl]).mean(1)                       # (L+1, NCONV)
            A, m, v = adam_dir(m, v, g, S.tc + s)
            dA = -LR * (A @ U.T)
            cumA += dA
            cumS += -(g @ U.T)
            if s > S_steps - 750:
                tail += dA
            if s in rec:
                rec_out[s] = (cumA.cpu().numpy().copy(), cumS.cpu().numpy().copy())
    return {"arm": arm, "seed": seed, "task": task, "G_full": G.cpu().numpy(), "L": L,
            "rec": {str(k): {"adam": v[0], "sgd": v[1]} for k, v in rec_out.items()},
            "tail750_adam": tail.cpu().numpy(),
            "zbar": torch.cat([cs["zbar_c1"], cs["zbar_c2"]]).cpu().numpy()}


# --------------------------------------------------------------------------
# replay (c): the engine itself, from checkpoint t through task t+1
# --------------------------------------------------------------------------

@torch.no_grad()
def bundle_zbar(Bd) -> torch.Tensor:
    """(R, 32) channel means of z1 / z2 over the run's 1200 images (float64 accumulation)."""
    R = Bd.R
    s = torch.zeros(R, 2 * CH, dtype=DT, device=Bd.device)
    tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cudnn.allow_tf32 = False          # the record only; training steps keep TF32
    for i0, i1 in chunks(N, 50):
        o = E.forward(Bd.P, Bd.X[:, i0:i1], Bd.act)
        b = i1 - i0
        s[:, :CH] += o[0].reshape(b, R, CH, -1).double().sum((0, 3))
        s[:, CH:] += o[2].reshape(b, R, CH, -1).double().sum((0, 3))
    torch.backends.cudnn.allow_tf32 = tf32
    s[:, :CH] /= N * 32 * 32
    s[:, CH:] /= N * 16 * 16
    return s


def replay(task: int, arms, seeds, out: Path, device, rec_steps=(1, 10, 75, 750, 7500, 30000), tag=""):
    """Load every (arm, seed) checkpoint of `task` into one engine bundle and train task+1 exactly
    as the run did (labels, batch orders, Adam, alpha updates), recording channel means."""
    out.mkdir(parents=True, exist_ok=True)
    slots = [(a, s) for a in arms for s in seeds]
    Bd = engine_bundle(slots, device, graph=True)
    tcs = set()
    with torch.no_grad():
        for r, (a, s) in enumerate(slots):
            st = torch.load(ckpt_path(a, s, task), map_location="cpu", weights_only=False)
            for dst, src in ((Bd.P, st["P"]), (Bd.m, st["m"]), (Bd.v, st["v"])):
                for q, x in zip(dst, src):
                    q[r].copy_(x)
            tcs.add(float(st["tc"]))
            for k in ("V", "ada", "fixA", "cval"):
                for dst, src in zip(getattr(Bd.act, k), st["act"][k]):
                    dst[r].copy_(src)
    assert len(tcs) == 1
    Bd.tc.fill_(tcs.pop())
    for s in Bd.useeds:
        for _ in range(task * EPOCHS):
            torch.randperm(N, generator=Bd.g_batch[s])
        for _ in range(task):
            CN.task_labels(Bd.g_lab[s])
    def alphas():
        with torch.no_grad():
            return torch.cat([Bd.act.alpha(0), Bd.act.alpha(1)], 1).double().cpu().numpy()

    rec = {0: bundle_zbar(Bd).cpu().numpy()}
    rec_a = {0: alphas()}
    t0 = time.time()
    Bd.new_labels()
    step = 0
    # epoch 1 eagerly, step by step
    Bd.new_order()
    Bd.acc_ep.zero_()
    for j in range(SPE):
        Bd.step(j)
        step += 1
        if step in rec_steps:
            rec[step] = bundle_zbar(Bd).cpu().numpy(); rec_a[step] = alphas()
    for e in range(1, EPOCHS):
        Bd.run_epoch()
        step += SPE
        if step in rec_steps:
            rec[step] = bundle_zbar(Bd).cpu().numpy(); rec_a[step] = alphas()
    wall = time.time() - t0
    res = {"task": task, "slots": slots, "rec": {str(k): v for k, v in rec.items()},
           "rec_alpha": {str(k): v for k, v in rec_a.items()},
           "wall_s": wall, "cuda_max_mem_mb": torch.cuda.max_memory_allocated() / 2 ** 20,
           "bad": Bd.bad.cpu().numpy()}
    # fidelity: compare with the saved checkpoint of task+1 if it exists
    fid = {}
    for r, (a, s) in enumerate(slots):
        pth = ckpt_path(a, s, task + 1)
        if pth.exists():
            st = torch.load(pth, map_location="cpu", weights_only=False)
            num = sum(float(((Bd.P[i][r].cpu() - st["P"][i]) ** 2).sum()) for i in range(10))
            den = sum(float((st["P"][i] ** 2).sum()) for i in range(10))
            fid[f"{a}_{s}"] = math.sqrt(num / den)
    res["param_rel_diff_vs_saved"] = fid
    np.save(out / f"replay_t{task:02d}{tag}.npy", res, allow_pickle=True)
    print(f"replay t{task} -> t{task + 1}: {wall:.0f}s, max mem {res['cuda_max_mem_mb']:.0f} MB", flush=True)
    return res


# --------------------------------------------------------------------------
# (d) capacity direction: full raw-parameter NTK and the literal self (Codex full_ntk_positive)
# --------------------------------------------------------------------------

def logit_jacobian(P, al, X, idx) -> torch.Tensor:
    """(n*C, nparams) Jacobian of the logits w.r.t. every raw parameter of P, with the pool
    winners fixed to `idx` (per image) -- equal to the true Jacobian at the reference state."""
    Pr = [p.detach().clone().requires_grad_(True) for p in P]
    rows = []
    for n in range(X.shape[0]):
        f, _ = forward_routed(Pr, X[n:n + 1], al,
                              tuple(None if t is None else t[n:n + 1] for t in idx))
        for c in range(C):
            g = torch.autograd.grad(f[0, c], Pr, retain_graph=c < C - 1)
            rows.append(torch.cat([q.reshape(-1) for q in g]))
    return torch.stack(rows)


def capacity_one(arm, seed, task, device, n_sub=32, chans=(0, 4, 8, 12), lams=(1e-2, 1.0, 1e2), hu=1e-4):
    """d/dq (1/2) log det(lam I + K) along the channel-mean direction u, K = J J^T over n_sub
    images x 10 logits, for the full net and the literal self; K' by central differences of J
    along u with the reference pool winners held fixed (Codex's derivative within the fixed
    gate / winner neighbourhood)."""
    S = State(arm, seed, task, device)
    U = mean_grads(S)
    gen = torch.Generator().manual_seed(4242 + seed)
    sub = torch.randperm(N, generator=gen)[:n_sub].to(device)
    X = S.X[sub]
    res = {"arm": arm, "seed": seed, "task": task, "n_sub": n_sub, "chans": list(chans), "lams": list(lams),
           "rows": []}

    def cap_slopes(Pfull, al, u_full_list):
        """u_full_list: per-parameter-tensor direction (list aligned with Pfull)."""
        with torch.no_grad():
            _, idx = forward_routed(Pfull, X, al)
        J = logit_jacobian(Pfull, al, X, idx)
        u_norm = math.sqrt(sum(float((q * q).sum()) for q in u_full_list))
        h = hu / u_norm
        Pp = [p + h * q for p, q in zip(Pfull, u_full_list)]
        Pm = [p - h * q for p, q in zip(Pfull, u_full_list)]
        Jd = (logit_jacobian(Pp, al, X, idx) - logit_jacobian(Pm, al, X, idx)) / (2 * h)
        K = J @ J.T
        Kd = Jd @ J.T + J @ Jd.T
        ev = torch.linalg.eigvalsh(K)
        evd = torch.linalg.eigvalsh(Kd)
        out = {"K_eig_min": float(ev.min()), "K_eig_max": float(ev.max()),
               "Kd_eig_min": float(evd.min()), "Kd_eig_max": float(evd.max()),
               "Kd_trace": float(torch.trace(Kd))}
        I = torch.eye(K.shape[0], dtype=DT, device=device)
        for lam in lams:
            out[f"cap_slope_lam{lam:g}"] = float(0.5 * torch.trace(torch.linalg.solve(lam * I + K, Kd)))
        # CE direction on the same images (exact uniform-label expectation, same u)
        f, _ = forward_routed(Pfull, X, al, idx)
        Rr = (J @ torch.cat([q.reshape(-1) for q in u_full_list])).view(X.shape[0], C)
        out["G_sub"] = float(((f.softmax(1) - 1.0 / C) * Rr).sum(1).mean())
        return out

    for layer in range(2):
        for j in chans:
            u = U[layer * CH + j]
            uf = [u[CONV_OFF[i]:CONV_OFF[i + 1]].view_as(S.P[i]) for i in range(4)] + \
                 [torch.zeros_like(S.P[i]) for i in range(4, 10)]
            full = cap_slopes(S.P, S.alpha, uf)
            Q, bl = self_params(S.P, S.alpha, layer, j)
            if layer == 0:
                us = [uf[0][j:j + 1], uf[1][j:j + 1], torch.zeros_like(Q[2]), torch.zeros_like(Q[3])]
            else:
                us = [uf[0], uf[1], uf[2][j:j + 1], uf[3][j:j + 1]]
            us += [torch.zeros_like(Q[i]) for i in range(4, 10)]
            selfr = cap_slopes(Q, bl, us)
            res["rows"].append({"layer": f"c{layer + 1}", "ch": j,
                                **{f"full_{k}": v for k, v in full.items()},
                                **{f"self_{k}": v for k, v in selfr.items()}})
    return res


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["checks", "checks2", "checks3", "measure", "frozen", "replay",
                                     "timing", "zbar", "capacity", "c2exact", "extra"])
    ap.add_argument("--nsub", type=int, default=32)
    ap.add_argument("--tag", default="")
    ap.add_argument("--states", default="",
                    help="checks: arm:seed:task,... (default: the phase-1 Snake states)")
    ap.add_argument("--arms", default="SNA,SNAc3,CV06FC3,CV3FC06")
    ap.add_argument("--seeds", default="10-19")
    ap.add_argument("--tasks", default="1,5,10,20")
    ap.add_argument("--task", type=int, default=1)
    ap.add_argument("--K", type=int, default=65536)
    ap.add_argument("--L", type=int, default=32)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--no-self", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mem-gb", type=float, default=2.3,
                    help="cap of this process's CUDA caching allocator (the GPU is shared; the "
                         "CUDA context adds ~0.5 GB on top)")
    args = ap.parse_args()
    device = H.setup(args.device)
    if device.type == "cuda":
        di = torch.cuda.current_device()
        total = torch.cuda.get_device_properties(di).total_memory / 2 ** 30
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.mem_gb / total), di)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    arms = args.arms.split(",")
    seeds = parse_list(args.seeds)
    if args.states:
        st = tuple((a, int(s), int(t)) for a, s, t in (x.split(":") for x in args.states.split(",")))
        CHECK_STATES[args.mode] = st
        set_activation([a for a, s, t in st])
    else:
        set_activation(arms)
    if args.mode == "checks":
        run_checks(out, device)
        return
    if args.mode == "checks2":
        run_checks_bundle(out, device)
        return
    if args.mode == "checks3":
        run_checks_selfshape(out, device)
        return
    if args.mode == "replay":
        replay(args.task, arms, seeds, out, device, tag=args.tag)
        return
    torch.backends.cudnn.allow_tf32 = False             # float64 anyway; keep conv exact
    tasks = parse_list(args.tasks)
    if args.mode == "zbar":
        rows = []
        for task in tasks:
            for arm in arms:
                for seed in seeds:
                    if task > 0 and not ckpt_path(arm, seed, task).exists():
                        print(f"missing {arm} s{seed} t{task}", flush=True)
                        continue
                    S = State(arm, seed, task, device)
                    cs = channel_stats(S)
                    for l, tag in enumerate(("c1", "c2")):
                        for j in range(CH):
                            rows.append({"arm": arm, "seed": seed, "task": task, "layer": tag, "ch": j,
                                         "zbar": float(cs[f"zbar_{tag}"][j]), "zsd": float(cs[f"zsd_{tag}"][j]),
                                         "alpha": float(S.alpha[l][j])})
        import pandas as pd
        pd.DataFrame(rows).to_csv(out / "zbar_checkpoints.csv", index=False)
        print("done", len(rows), flush=True)
        return
    if args.mode == "timing":
        r = measure_one(arms[0], seeds[0], tasks[0], device, K=args.K, self_net=not args.no_self,
                        verbose=True)
        print("max mem MB", torch.cuda.max_memory_allocated() / 2 ** 20)
        return
    for task in tasks:
        for arm in arms:
            for seed in seeds:
                fn = out / f"{args.mode}_{arm}_s{seed}_t{task:02d}.npy"
                if fn.exists():
                    continue
                if not ckpt_path(arm, seed, task).exists():
                    print(f"missing {ckpt_path(arm, seed, task)}", flush=True)
                    continue
                if args.mode == "measure":
                    r = measure_one(arm, seed, task, device, K=args.K, self_net=not args.no_self,
                                    verbose=True)
                elif args.mode == "capacity":
                    t0 = time.time()
                    r = capacity_one(arm, seed, task, device, n_sub=args.nsub)
                    print(f"capacity {arm} s{seed} t{task} {time.time() - t0:.0f}s", flush=True)
                elif args.mode == "c2exact":
                    t0 = time.time()
                    r = c2_exact_one(arm, seed, task, device, K=args.K)
                    print(f"c2exact {arm} s{seed} t{task} {time.time() - t0:.0f}s", flush=True)
                elif args.mode == "extra":
                    t0 = time.time()
                    r = extra_one(arm, seed, task, device, K=args.K)
                    print(f"extra {arm} s{seed} t{task} {time.time() - t0:.0f}s", flush=True)
                else:
                    r = frozen_one(arm, seed, task, device, L=args.L, S_steps=args.steps)
                    print(f"frozen {arm} s{seed} t{task}", flush=True)
                np.save(fn, r, allow_pickle=True)
                torch.cuda.empty_cache()
    print("done", flush=True)


if __name__ == "__main__":
    main()
