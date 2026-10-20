#!/usr/bin/env python3
"""kappa_reload_cnn_1020: the readout projection kappa_k, the switch push on every layer, its keep, the
init push and the return, on the RL-CIFAR CNN (spec specs/spec_kappa_reload_cnn_1020.md, sec. 3-5 and
addendum 1, A1.1-A1.3).

    python src/kappa_reload_cnn_1020.py checks  --which f64 --arms LR       K1-K4, K6, K7a, K7b on LR states
    python src/kappa_reload_cnn_1020.py checks  --which f64 --arms SNA      the same on SNA s10 t1
    python src/kappa_reload_cnn_1020.py checks  --which f64 --arms R        K1-K4, K7a on R s10 t2 / t3
    python src/kappa_reload_cnn_1020.py checks  --which LR,SNA10,SNAKE40    K5: t1 -> t2 replays, bit check
    python src/kappa_reload_cnn_1020.py measure --arms LR                   t = 0 and every saved t
    python src/kappa_reload_cnn_1020.py replay  --arms LR [--tasks 1,2] [--full]
    python src/kappa_reload_cnn_1020.py init    --arms LR                   init -> task 1, 200 updates
    python src/kappa_reload_cnn_1020.py diag                                Snake dm_P / dm_alpha, LR m = 0 step
    python src/kappa_reload_cnn_1020.py merge                               checks.json, rows.json

Post-hoc on the saved checkpoints (read only).  The float64 quantities run on the engine's forward
re-implemented in float64 (cnn_drive_verify_1009.forward, checked against the engine there, K0); the
replays run the engine itself (float32, cuDNN TF32 as trained, CUDA graphs for whole epochs) from the
checkpoint's parameters, Adam moments, step count and activation state, with the run's streams.

Sign convention (cnn_drive_verify_1009): G = <grad_P m, grad_P L_u>, P the 10 raw parameter tensors with
the checkpoint's alpha held fixed, L_u the exact expectation of the mean CE over iid uniform labels.  One
SGD step moves m by -eta G to first order: G > 0 is the sinking side.  For f2 with the input features and
the activation state held fixed, G_own = S_exact / N, S_exact_i = sum_n K_n g_in <v^c_i, pi_n>.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "analysis" / "sna_cnn_cause_1009", ROOT / "src", ROOT):
    sys.path.insert(0, str(_p))

import numpy as np                                   # noqa: E402
import torch                                         # noqa: E402
import torch.nn.functional as F                      # noqa: E402
from torch.func import jvp, vmap                     # noqa: E402

from src import pmnist_0905 as H                     # noqa: E402  (read only)
from src import pmnist_rlcifar_0907 as RC            # noqa: E402
from src import rlcifar_cnn_0908 as CN               # noqa: E402
from src import sna_cnn_cause_1009 as E              # noqa: E402
import posthoc_lib as PL                             # noqa: E402
import cnn_drive_verify_1009 as DV                   # noqa: E402

# The worktrees these modules point at are gone; point them at the archived copies (this process
# only -- no file is edited).  ckpt_path() and Cifar10() read the module globals at call time.
DATA_CIFAR = Path("/home/issan/Projects/claude/proj_004_drift/data/cifar10")          # read only
ARCH = Path("/home/issan/Projects/obsidian-research-data")
RC.DATA_DIR = DATA_CIFAR
DV.CKPT_DIR = ARCH / "sna_cnn_cause_1009" / "results" / "sna_cnn_cause_1009" / "A" / "ckpt"
DV.OWN_RESULTS = ARCH / "cnn_drive_verify_1009" / "results" / "cnn_drive_verify_1009"
SAVED = ROOT / "results" / "cnn_drive_verify_1009"            # saved G_full (K2a)
OUT = ROOT / "results" / "kappa_reload_cnn_1020"
RAW = OUT / "raw"

N, K, HID, CH = CN.N_IMAGES, CN.N_CLASSES, CN.HIDDEN, CN.CHANNELS
LOGK = math.log(K)
DT = torch.float64
SNAKE = ("SNA", "SNAc3", "CV06FC3", "CV3FC06")               # training order of the 40-slot bundle
TASKS = {"LR": (1, 2, 3, 5, 10, 20, 30), "R": (1, 2, 3),
         **{a: (1, 2, 3, 5, 10, 20, 30, 40, 50) for a in SNAKE}}
LAST_TASK = {"LR": 30, "R": 3, **{a: 50 for a in SNAKE}}      # tasks the original runs trained
SEEDS = {"LR": tuple(range(10, 20)), "R": (10,), **{a: tuple(range(10, 20)) for a in SNAKE}}
LAYERS = ("c1", "c2", "f1", "f2")
OFF = (0, 16, 32, 132, 232)
NU = 232
DEN = [N * 32 * 32] * 16 + [N * 16 * 16] * 16 + [N] * 200     # images x positions per site
REC_STEPS = (1, 2, 5, 10, 20, 50, 75, 100, 150, 200)
FULL_STEPS = (750, 7500, 15000, 30000)
SPE, EPOCHS = CN.STEPS_PER_EPOCH, DV.EPOCHS
BLOCKS = ((0, 1), (2, 3), (4, 5), (6, 7), (8, 9))            # (W,b) of c1 c2 f1 f2 head
N_PERM = 1000


def sl(name: str) -> slice:
    i = LAYERS.index(name)
    return slice(OFF[i], OFF[i + 1])


def parse_list(s: str) -> list[int]:
    return DV.parse_list(s)


def ref_task(arm: str, t: int):
    ts = (0,) + TASKS[arm]
    i = ts.index(t)
    return ts[i - 1] if i > 0 else None


def old_labels(seed: int, t: int) -> torch.Tensor:
    """Labels of task t (the old labels at the end of task t); for t = 0 the labels of task 1, against
    which the random head's kappa is a chance value (an added measurement: the MLP left it NaN)."""
    return PL.labels_at(seed, max(t, 1))


def close(a, b, rtol, atol):
    """|a - b| <= atol + rtol * max(|a|, |b|) elementwise (spec A1.2); NaN only where both are NaN."""
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    na, nb = np.isnan(a), np.isnan(b)
    if not np.array_equal(na, nb):
        return False
    a, b = a[~na], b[~nb]
    return bool(np.all(np.abs(a - b) <= atol + rtol * np.maximum(np.abs(a), np.abs(b))))


def close_scaled(a, b, rtol):
    """close() with atol = rtol x max|b| (scale of the quantity)."""
    bb = np.asarray(b, dtype=np.float64)
    sc = float(np.nanmax(np.abs(bb))) if np.isfinite(bb).any() else 0.0
    return close(a, b, rtol, rtol * sc)


def maxdiff(a, b):
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))


def tonp(t):
    return t.detach().double().cpu().numpy()


# --------------------------------------------------------------------------
# float64 net (the engine's op order, cnn_drive_verify_1009.forward)
# --------------------------------------------------------------------------

def fwd(P, x, al):
    W1, b1, W2, b2, W3, b3, W4, b4, W5, b5 = P
    z1 = F.conv2d(x, W1, b1, padding=CN.PAD)
    h1 = F.max_pool2d(DV.snake(z1, al[0], True), CN.POOL, CN.POOL)
    z2 = F.conv2d(h1, W2, b2, padding=CN.PAD)
    h2 = F.max_pool2d(DV.snake(z2, al[1], True), CN.POOL, CN.POOL).flatten(1)
    z3 = h2 @ W3.T + b3
    a3 = DV.snake(z3, al[2], False)
    z4 = a3 @ W4.T + b4
    a4 = DV.snake(z4, al[3], False)
    return z1, h1, z2, h2, z3, a3, z4, a4, a4 @ W5.T + b5


def msum(P, x, al):
    """Sum over the chunk's images (and positions) of the 232 preactivations (c1 c2 f1 f2)."""
    z1, h1, z2, h2, z3, a3, z4, a4, l = fwd(P, x, al)
    return torch.cat([z1.sum((0, 2, 3)), z2.sum((0, 2, 3)), z3.sum(0), z4.sum(0)])


def route_idx(P, x, al):
    """Pool winners and (leaky / ReLU) gate patterns of state P on images x."""
    pw = DV.ACT["kind"] != "snake"
    with torch.no_grad():
        z1 = F.conv2d(x, P[0], P[1], padding=CN.PAD)
        a1 = DV.snake(z1, al[0], True)
        h1, i1 = F.max_pool2d(a1, CN.POOL, CN.POOL, return_indices=True)
        z2 = F.conv2d(h1, P[2], P[3], padding=CN.PAD)
        a2 = DV.snake(z2, al[1], True)
        h2, i2 = F.max_pool2d(a2, CN.POOL, CN.POOL, return_indices=True)
        z3 = h2.flatten(1) @ P[4].T + P[5]
        z4 = DV.snake(z3, al[2], False) @ P[6].T + P[7]
    g = (z1 > 0, z2 > 0, z3 > 0, z4 > 0) if pw else (None,) * 4
    return (i1, i2) + g


def msum_routed(P, x, al, idx):
    """msum with the pool winners and the gates of a reference state fixed: smooth in P."""
    i1, i2, g1, g2, g3, g4 = idx
    z1 = F.conv2d(x, P[0], P[1], padding=CN.PAD)
    a1 = DV.phi_gated(z1, al[0], True, g1)
    h1 = a1.flatten(2).gather(2, i1.flatten(2)).view_as(i1)
    z2 = F.conv2d(h1, P[2], P[3], padding=CN.PAD)
    a2 = DV.phi_gated(z2, al[1], True, g2)
    h2 = a2.flatten(2).gather(2, i2.flatten(2)).view_as(i2).flatten(1)
    z3 = h2 @ P[4].T + P[5]
    z4 = DV.phi_gated(z3, al[2], False, g3) @ P[6].T + P[7]
    return torch.cat([z1.sum((0, 2, 3)), z2.sum((0, 2, 3)), z3.sum(0), z4.sum(0)])


def den(device):
    return torch.tensor(DEN, dtype=DT, device=device)


def m_of(P, al, X, chunk=300, routed=None) -> torch.Tensor:
    tot = torch.zeros(NU, dtype=DT, device=X.device)
    with torch.no_grad():
        for k, (i0, i1) in enumerate(DV.chunks(X.shape[0], chunk)):
            tot += msum(P, X[i0:i1], al) if routed is None else msum_routed(P, X[i0:i1], al, routed[k])
    return tot / den(X.device)


def block_tangents(g):
    """Tangents: g restricted to each (W, b) block (c1 c2 f1 f2 head), then g itself."""
    out = []
    for b in BLOCKS:
        out.append([g[i] if i in b else torch.zeros_like(g[i]) for i in range(10)])
    out.append(list(g))
    return [torch.stack([t[i] for t in out]).detach() for i in range(10)]


def g_blocks_jvp(P, al, X, g, chunk=150):
    """(6, 232): forward-mode derivative of m along each block of the ONE full-data grad L_u (rows
    c1 c2 f1 f2 head blocks, then the whole of g = G_full); the tangent is never re-derived per chunk."""
    T = block_tangents(g)
    Pd = tuple(p.detach() for p in P)
    tot = torch.zeros(len(BLOCKS) + 1, NU, dtype=DT, device=X.device)
    for i0, i1 in DV.chunks(X.shape[0], chunk):
        x = X[i0:i1]
        f = (lambda xx: (lambda *Q: msum(Q, xx, al)))(x)
        tot += vmap(lambda *tt: jvp(f, Pd, tt)[1])(*T)
    return tot / den(X.device)


def g_dir_jvp(P, al, X, d, chunk=300):
    """d/de m(P + e d) at e = 0 for one direction d (10 tensors)."""
    Pd = tuple(p.detach() for p in P); dd = tuple(q.detach() for q in d)
    tot = torch.zeros(NU, dtype=DT, device=X.device)
    for i0, i1 in DV.chunks(X.shape[0], chunk):
        x = X[i0:i1]
        tot += jvp(lambda *Q: msum(Q, x, al), Pd, dd)[1]
    return tot / den(X.device)


def ce_grad(P, al, X, y, chunk=300):
    """grad of the mean CE with labels y (all 10 tensors)."""
    Pr = [p.detach().clone().requires_grad_(True) for p in P]
    gs = [torch.zeros_like(p) for p in P]
    for i0, i1 in DV.chunks(X.shape[0], chunk):
        f = fwd(Pr, X[i0:i1], al)[8]
        L = F.cross_entropy(f, y[i0:i1], reduction="sum") / X.shape[0]
        for a, b in zip(gs, torch.autograd.grad(L, Pr)):
            a += b
    return gs


def mean_grad_unit(P, al, X, u: int, chunk=300):
    """Reverse mode: grad_P m_u over all images (10 tensors), for K2b."""
    Pr = [p.detach().clone().requires_grad_(True) for p in P]
    gs = [torch.zeros_like(p) for p in P]
    d = DEN[u]
    for i0, i1 in DV.chunks(X.shape[0], chunk):
        s = msum(Pr, X[i0:i1], al)[u] / d
        gr = torch.autograd.grad(s, Pr, allow_unused=True)
        for a, b in zip(gs, gr):
            if b is not None:
                a += b
    return gs


# --------------------------------------------------------------------------
# numpy helpers (MLP measure.py)
# --------------------------------------------------------------------------

def rankavg(x):
    order = np.argsort(x, kind="mergesort"); r = np.empty(len(x)); r[order] = np.arange(len(x), dtype=np.float64)
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    s = np.zeros(len(cnt)); np.add.at(s, inv, r); return (s / cnt)[inv]


def spearman(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b); a, b = a[ok], b[ok]
    if len(a) < 3:
        return float("nan")
    ra, rb = rankavg(a), rankavg(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def quart(x):
    x = np.asarray(x, dtype=np.float64); x = x[np.isfinite(x)]
    if len(x) == 0:
        return [float("nan")] * 3
    return [float(v) for v in np.percentile(x, [25, 50, 75])]


def agree(a, b, mask):
    """Sign agreement in mask among units where both are nonzero; (rate, count)."""
    mk = mask & (a != 0) & (b != 0) & np.isfinite(a) & np.isfinite(b)
    if mk.sum() == 0:
        return float("nan"), 0
    return float((np.sign(a[mk]) == np.sign(b[mk])).mean()), int(mk.sum())


def frac(x, mask):
    return float(x[mask].mean()) if mask.sum() else float("nan")


def med(x, mask):
    return float(np.median(x[mask])) if mask.sum() else float("nan")


def auc(score, label):
    """P(score of a positive > score of a negative), ties 1/2; NaN unless both classes occur."""
    score = np.asarray(score, dtype=np.float64); label = np.asarray(label, dtype=bool)
    p, n = score[label], score[~label]
    if len(p) == 0 or len(n) == 0:
        return float("nan")
    r = rankavg(np.concatenate([p, n]))
    return float((r[:len(p)].sum() - len(p) * (len(p) - 1) / 2) / (len(p) * len(n)))


# --------------------------------------------------------------------------
# one state
# --------------------------------------------------------------------------

def head_quantities(A3, Z4, A4, L, W4, W5, b5, al3, y, H2=None, Z3=None, al2=None):
    """The closed forms of the f2 push (MLP measure.py), kappa and the lemma; also f1's S3."""
    dev = A3.device
    pi = L.softmax(1)
    W5c = W5 - W5.mean(0, keepdim=True)
    b5c = b5 - b5.mean()
    xbar = A3.mean(0)
    Kn = A3 @ xbar + 1.0
    g4 = DV.dphi_b(Z4, al3.view(1, -1))
    d4 = g4 * (pi @ W5c)                                        # delta4_in = g_in <v^c_i, pi_n>
    S_exact = (Kn[:, None] * d4).sum(0)
    out = dict(pi=pi, W5c=W5c, b5c=b5c, xbar=xbar, Kn=Kn, g4=g4, S_exact=S_exact)
    if H2 is not None:
        h2bar = H2.mean(0)
        K3n = H2 @ h2bar + 1.0
        g3 = DV.dphi_b(Z3, al2.view(1, -1))
        d3 = g3 * (d4 @ W4)                                     # delta3 = d l_u,n / d z3_n (N x)
        out.update(h2bar=h2bar, K3n=K3n, S3=(K3n[:, None] * d3).sum(0))
    n_ = A3.shape[0]
    Y = F.one_hot(y, K).to(DT)
    Fm = (Y - 1.0 / K).T @ A4                                   # [K, H]
    ff = float((Fm * Fm).sum())
    A = float((W5c * Fm).sum())
    nV = float(W5c.norm())
    kappa = A / ff if ff > 0 else float("nan")
    ar = torch.arange(n_, device=dev)
    A_id = float(((L[ar, y] - L.mean(1)) - b5c[y]).sum())
    lse = torch.logsumexp(L, 1)
    L_old = float((lse - L[ar, y]).mean())
    J = float((lse - L.mean(1) - LOGK).sum())
    ncls = Y.sum(0)
    B = float(((ncls - n_ / K) * b5c).sum())
    D = n_ * (LOGK - L_old) - B
    Fi2 = (Fm * Fm).sum(0)
    kappa_i = torch.where(Fi2 > 0, (W5c * Fm).sum(0) / Fi2.clamp_min(1e-300), torch.full_like(Fi2, float("nan")))
    SF = (Kn[:, None] * g4 * A4).sum(0)
    S_self = kappa * (1 - 1.0 / K) * SF
    S_oh = (Kn[:, None] * g4 * W5c[y, :]).sum(0)
    S_proj = kappa * (Kn[:, None] * g4 * Fm[y, :]).sum(0)      # sum_n K_n g_in (kappa F)_{y_n, i}
    out.update(Y=Y, Fm=Fm, ff=ff, A=A, A_id=A_id, nV=nV, kappa=kappa, L_old=L_old, J=J, B=B, D=D,
               kappa_i=kappa_i, S_self=S_self, S_oh=S_oh, S_proj=S_proj,
               acc_old=float((pi.argmax(1) == y).double().mean()), conf_old=float(pi[ar, y].mean()),
               center_max=max(float(W5c.sum(0).abs().max()), float(Fm.sum(0).abs().max()), float(b5c.sum().abs())))
    return out


def kappa_null(W5c, Fm, ff, gen, n=N_PERM):
    """kappa with the centred head rows permuted over classes (features and F fixed), n draws."""
    M = (W5c @ Fm.T).detach().cpu()                             # M[a, c] = <v^c_a, F_c>
    perms = torch.stack([torch.randperm(K, generator=gen) for _ in range(n)])
    vals = M[perms, torch.arange(K)[None, :]].sum(1) / ff       # sum_c M[pi(c), c]
    return vals.double().numpy(), perms


def state_quantities(S, y_old: torch.Tensor, ref=None, chunk=300, jchunk=150, perm_seed=None,
                     perm_push=0) -> dict:
    """Every float64 quantity of spec sec. 3 / A1.1 at state S; `ref` = cache of the reference state."""
    P, al, X = S.P, S.alpha, S.X
    dev = X.device
    W1, b1, W2, b2, W3, b3, W4, b4, W5, b5 = P
    s1 = torch.zeros(32, dtype=DT, device=dev); s2 = torch.zeros_like(s1); pos = torch.zeros_like(s1)
    mp1 = torch.zeros(3 * 25, dtype=DT, device=dev); mp2 = torch.zeros(CH * 25, dtype=DT, device=dev)
    H2s, Z3s, A3s, Z4s, A4s, Ls = [], [], [], [], [], []
    with torch.no_grad():
        for i0, i1 in DV.chunks(X.shape[0], chunk):
            x = X[i0:i1]
            z1, h1, z2, h2, z3, a3, z4, a4, l = fwd(P, x, al)
            for k, z in ((0, z1), (1, z2)):
                q = slice(16 * k, 16 * k + 16)
                s1[q] += z.sum((0, 2, 3)); s2[q] += (z * z).sum((0, 2, 3)); pos[q] += (z > 0).sum((0, 2, 3))
            mp1 += F.unfold(x, CN.KERNEL, padding=CN.PAD).sum((0, 2))       # zero-pad patches
            mp2 += F.unfold(h1, CN.KERNEL, padding=CN.PAD).sum((0, 2))
            H2s.append(h2); Z3s.append(z3); A3s.append(a3); Z4s.append(z4); A4s.append(a4); Ls.append(l)
    H2, Z3, A3, Z4, A4, L = (torch.cat(v) for v in (H2s, Z3s, A3s, Z4s, A4s, Ls))
    nimg = X.shape[0]
    n1, n2 = nimg * 32 * 32, nimg * 16 * 16
    m = torch.zeros(NU, dtype=DT, device=dev); sd = torch.zeros_like(m); pplus = torch.zeros_like(m)
    npos = torch.zeros_like(m)
    for q, n in ((slice(0, 16), n1), (slice(16, 32), n2)):
        mq = s1[q] / n
        m[q] = mq
        sd[q] = (s2[q] / n - mq * mq).clamp_min(0).sqrt()       # population sd (as the MLP)
        pplus[q] = pos[q] / n; npos[q] = pos[q]
    for q, Z in ((sl("f1"), Z3), (sl("f2"), Z4)):
        m[q] = Z.mean(0); sd[q] = Z.std(0, unbiased=False)
        pplus[q] = (Z > 0).double().mean(0); npos[q] = (Z > 0).double().sum(0)
    alive = npos > 0
    ub1, ub2 = mp1 / n1, mp2 / n2
    hq = head_quantities(A3, Z4, A4, L, W4, W5, b5, al[3], y_old.to(dev), H2=H2, Z3=Z3, al2=al[2])
    xbar, h2bar = hq["xbar"], hq["h2bar"]
    m_patch = torch.cat([W1.reshape(CH, -1) @ ub1 + b1, W2.reshape(CH, -1) @ ub2 + b2,
                         W3 @ h2bar + b3, W4 @ xbar + b4])

    # grad of L_u over all images (one vector), then forward mode along its blocks
    gs, Lu = DV.uniform_grad(P, al, X, list(range(10)), chunk=chunk)
    GB = g_blocks_jvp(P, al, X, gs, chunk=jchunk)               # (6, 232)
    G_full = GB[5]
    G_own = torch.cat([gs[0].reshape(CH, -1) @ ub1 + gs[1], gs[2].reshape(CH, -1) @ ub2 + gs[3],
                       gs[4] @ h2bar + gs[5], gs[6] @ xbar + gs[7]])
    G_own_jvp = torch.cat([GB[0][:16], GB[1][16:32], GB[2][sl("f1")], GB[3][sl("f2")]])
    G_up = torch.cat([torch.zeros(16, dtype=DT, device=dev), GB[0][16:32], (GB[0] + GB[1])[sl("f1")],
                      (GB[0] + GB[1] + GB[2])[sl("f2")]])
    # the block entries that must vanish identically (downstream blocks, head)
    zero_blocks = torch.cat([GB[1][:16], GB[2][:32], GB[3][:132], GB[4]]).abs().max()

    # kappa, its ledger, the lemma
    Fm, ff, A, nV, kappa = hq["Fm"], hq["ff"], hq["A"], hq["nV"], hq["kappa"]
    parts = dict(kap_chance=float("nan"), kap_feat=float("nan"), kap_read=float("nan"),
                 decomp_sum=float("nan"), decomp_absdiff=float("nan"), expl_delta=float("nan"))
    if ref is not None and ff > 0:
        Fr = (hq["Y"] - 1.0 / K).T @ ref["A4"]
        Vr = ref["W5c"]
        kc = float((Vr * Fr).sum()) / ff
        kf = float((Vr * (Fm - Fr)).sum()) / ff
        dV = hq["W5c"] - Vr
        kr = float((dV * Fm).sum()) / ff
        parts = dict(kap_chance=kc, kap_feat=kf, kap_read=kr, decomp_sum=kc + kf + kr,
                     decomp_absdiff=abs(kc + kf + kr - kappa),
                     expl_delta=float((dV * Fm).sum()) ** 2 / max(float((dV * dV).sum()) * ff, 1e-300))
    Kn = hq["Kn"]
    Kq = torch.quantile(Kn, torch.tensor([0.01, 0.25, 0.5, 0.75], dtype=DT, device=dev))
    null = dict(kappa_null_q=[float("nan")] * 5, kappa_null_sd=float("nan"), kappa_null_quantile=float("nan"))
    perm_push_q = [float("nan")] * 3
    if ff > 0 and perm_seed is not None:
        gen = torch.Generator().manual_seed(perm_seed)
        vals, perms = kappa_null(hq["W5c"], Fm, ff, gen)
        null = dict(kappa_null_q=[float(v) for v in np.percentile(vals, [1, 5, 50, 95, 99])],
                    kappa_null_sd=float(vals.std()), kappa_null_mean=float(vals.mean()),
                    kappa_null_quantile=float((vals < kappa).mean() + 0.5 * (vals == kappa).mean()))
        if perm_push:
            a2 = alive[sl("f2")]
            fr = []
            for p_ in perms[:perm_push]:
                Wp = hq["W5c"][p_.to(dev)]
                pi_p = (A4 @ Wp.T + b5).softmax(1)
                Sp = (Kn[:, None] * hq["g4"] * (pi_p @ Wp)).sum(0)
                fr.append(float((Sp[a2] > 0).double().mean()) if a2.any() else float("nan"))
            perm_push_q = quart(fr)
    scal = dict(tc=float(S.tc), L_u=float(Lu), acc_old=hq["acc_old"], conf_old=hq["conf_old"], L_old=hq["L_old"],
                gain=LOGK - hq["L_old"], kappa=kappa, A=A, A_id=hq["A_id"], A_id_absdiff=abs(A - hq["A_id"]),
                D=hq["D"], J=hq["J"], A_DJ_absdiff=abs(A - hq["D"] - hq["J"]), B=hq["B"],
                kappa_lb=hq["D"] / ff if ff > 0 else float("nan"),
                kappa_ub=nV / math.sqrt(ff) if ff > 0 else float("nan"),
                F_norm=math.sqrt(ff), F_norm2=ff, normV=nV, Q_proj=kappa * math.sqrt(ff) if ff > 0 else float("nan"),
                expl_total=A * A / max(nV * nV * ff, 1e-300) if ff > 0 else float("nan"),
                learned_norm=abs(kappa) * math.sqrt(ff) / math.sqrt(HID) if ff > 0 else float("nan"),
                center_max=hq["center_max"],
                K_min=float(Kn.min()), K_negfrac=float((Kn < 0).double().mean()), K_q=[float(v) for v in Kq],
                K_mean=float(Kn.mean()), K_mean_absdiff=abs(float(Kn.mean()) - (float(xbar @ xbar) + 1.0)),
                K3_min=float(hq["K3n"].min()), K3_negfrac=float((hq["K3n"] < 0).double().mean()),
                K1_f2_absdiff=float((G_own[sl("f2")] - hq["S_exact"] / nimg).abs().max()),
                K1_f2_scale=float((hq["S_exact"] / nimg).abs().max()),
                K1_f1_absdiff=float((G_own[sl("f1")] - hq["S3"] / nimg).abs().max()),
                K1_f1_scale=float((hq["S3"] / nimg).abs().max()),
                K1_own_jvp_absdiff=float((G_own - G_own_jvp).abs().max()),
                K1_mpatch_absdiff=float((m_patch - m).abs().max()), K1_mpatch_scale=float(m.abs().max()),
                K2c_absdiff=float((G_full - G_own - G_up).abs().max()), K2c_scale=float(G_full.abs().max()),
                zero_blocks_absmax=float(zero_blocks), perm_push_frac_sink_q=perm_push_q, **null, **parts)
    alpha = torch.cat([a.reshape(-1) for a in al]).to(DT)
    units = dict(m=tonp(m), sd=tonp(sd), pplus=tonp(pplus), npos=tonp(npos), alive=tonp(alive.double()),
                 m_patch=tonp(m_patch), G_full=tonp(G_full), G_own=tonp(G_own), G_up=tonp(G_up),
                 G_own_jvp=tonp(G_own_jvp), G_blocks=tonp(GB[:5]), alpha=tonp(alpha),
                 S_exact=tonp(hq["S_exact"]), S_self=tonp(hq["S_self"]), S_oh=tonp(hq["S_oh"]),
                 S_proj=tonp(hq["S_proj"]), S3=tonp(hq["S3"]), kappa_i=tonp(hq["kappa_i"]))
    cache = dict(A4=A4, W5c=hq["W5c"])
    return dict(scal=scal, units=units, cache=cache, gs=gs, hq=hq, A4=A4, Z4=Z4, A3=A3, L=L)


def floors_default():
    return {"G": {k: 0.0 for k in LAYERS}, "S": 0.0, "ff": 0.0, "D": 0.0, "atol": {}}


def read_floors() -> dict:
    fn = OUT / "floors.json"
    if not fn.exists():
        raise SystemExit("floors.json missing: run the f64 checks first (they calibrate the floors)")
    return json.loads(fn.read_text())


def summarize_state(arm, seed, t, q: dict, fl: dict) -> dict:
    """One row: the state's scalars and the per-layer aggregates (spec sec. 4 (a)-(c), A1.1)."""
    u = q["units"]
    sc = q["scal"]
    alive = u["alive"] > 0
    rt = ref_task(arm, t) if t > 0 else None
    row = dict(arm=arm, seed=seed, t=t, task=t, ref_task=rt, span=(t - rt) if t > 0 else None, **sc)
    row["kappa_defined"] = bool(sc["F_norm2"] > fl["ff"])
    row["D_positive"] = bool(sc["D"] > fl["D"])
    row["kappa_over_lb"] = (1.0 + sc["J"] / sc["D"]) if row["D_positive"] else float("nan")
    row["lemma_violation"] = bool(sc["J"] < -fl["atol"].get("J", 0.0))
    at = fl["atol"]
    row["K3_A_ok"] = sc["A_id_absdiff"] <= at.get("A", 0.0) + 1e-10 * abs(sc["A"])
    row["K3_ADJ_ok"] = sc["A_DJ_absdiff"] <= at.get("A", 0.0) + 1e-10 * max(abs(sc["A"]), abs(sc["D"]) + abs(sc["J"]))
    row["K4_ok"] = (sc["decomp_absdiff"] <= at.get("kappa", 0.0) + 1e-10 * abs(sc["kappa"])) if rt is not None else None
    ki = u["kappa_i"]                                                   # f2 units (100)
    a2 = alive[sl("f2")]
    row["kappa_i_q"] = quart(ki[a2])
    row["kappa_i_negfrac"] = frac(ki < 0, a2 & np.isfinite(ki))
    row["kappa_i_n_undef"] = int((a2 & ~np.isfinite(ki)).sum())
    for name in LAYERS:
        q_ = sl(name)
        fg = fl["G"][name]
        al_ = alive[q_]
        allu = np.ones_like(al_)
        Gf, Go, Gu = u["G_full"][q_], u["G_own"][q_], u["G_up"][q_]
        df_, do_ = np.abs(Gf) > fg, np.abs(Go) > fg
        for tag, base in (("", al_), ("all_", allu)):
            row[f"{name}_{tag}n"] = int(base.sum())
            row[f"{name}_{tag}n_indet_full"] = int((base & ~df_).sum())
            row[f"{name}_{tag}frac_Gfull_pos"] = frac(Gf > 0, base & df_)
            row[f"{name}_{tag}frac_Gown_pos"] = frac(Go > 0, base & do_)
            row[f"{name}_{tag}medabs_Gfull"] = med(np.abs(Gf), base)
        row[f"{name}_n_units"] = int(len(al_))
        row[f"{name}_n_alive"] = int(al_.sum())
        row[f"{name}_n_indet_own"] = int((al_ & ~do_).sum())
        tot = np.abs(Go) + np.abs(Gu)
        okt = al_ & (tot > fg)
        row[f"{name}_up_share_med"] = med(np.abs(Gu) / np.where(okt, tot, 1.0), okt)
        row[f"{name}_cancel_med"] = med(np.abs(Gf) / np.where(okt, tot, 1.0), okt)
        trip = {}
        for so in (1, -1):
            for su in (1, -1, 0):
                for sf in (1, -1):
                    key = f"{'+' if so > 0 else '-'}{'+' if su > 0 else ('-' if su < 0 else '0')}{'+' if sf > 0 else '-'}"
                    c = int((al_ & do_ & df_ & (np.sign(Go) == so) & (np.sign(Gu) == su) & (np.sign(Gf) == sf)).sum())
                    if c:
                        trip[key] = c
        row[f"{name}_sign_triples"] = trip
        row[f"{name}_agree_own_full"] = agree(Go, Gf, al_ & do_ & df_)[0]
        row[f"{name}_medabs_Gown"] = med(np.abs(Go), al_)
        row[f"{name}_medabs_Gup"] = med(np.abs(Gu), al_)
        row[f"{name}_G_full_med"] = med(Gf, al_)
        mm, ss = u["m"][q_], u["sd"][q_]
        row[f"{name}_m_med"] = med(mm, al_)
        row[f"{name}_s_med"] = med(ss, al_)
        row[f"{name}_r_med"] = med(mm / np.maximum(ss, 1e-300), al_ & (ss > 0))
        row[f"{name}_n_p_below"] = int((u["pplus"][q_] < 0.5).sum())
        row[f"{name}_n_m_neg"] = int((mm < 0).sum())
    # f2: the MLP quantities and the projection ledger
    q_ = sl("f2")
    fS = fl["S"]
    Se, Ss, So, Sp = u["S_exact"], u["S_self"], u["S_oh"], u["S_proj"]          # f2 units (100)
    ke = a2 & (np.abs(Se) > fS)
    row["f2_frac_Sexact_pos"] = frac(Se > 0, ke)
    ks = a2 & (np.abs(Ss) > fS)
    row["f2_frac_Sself_pos"] = frac(Ss > 0, ks)
    common = ke & ks                                                     # alive, both above the S floor
    row["f2_agree_self_exact"], row["f2_n_agree_self_exact"] = agree(Ss, Se, common)
    ps, po = frac(Ss > 0, common), frac(Se > 0, common)                 # sign rates on the same set
    row["f2_n_agree_common"] = int(common.sum())
    row["f2_agree_self_exact_chance"] = ps * po + (1 - ps) * (1 - po) if np.isfinite(ps) and np.isfinite(po) else float("nan")
    row["f2_agree_oh_exact"] = agree(So, Se, ke)[0]
    row["f2_agree_proj_exact"] = agree(Sp, Se, ke)[0]
    row["f2_ratio_self_exact_q"] = quart((Ss / np.where(Se != 0, Se, np.nan))[ke])
    row["f2_sp_self_exact"] = spearman(Ss[a2], Se[a2])
    row["f2_sp_oh_exact"] = spearman(So[a2], Se[a2])
    row["f2_medabs_Sexact_N"] = med(np.abs(Se) / N, a2)
    row["f2_medabs_Sexact_N_all"] = float(np.median(np.abs(Se) / N))
    row["f2_Sexact_medN"] = med(Se / N, a2)
    row["f2_Sself_medN"] = med(Ss / N, a2)
    for nm, v in (("exact_minus_oh", Se - So), ("oh_minus_proj", So - Sp), ("proj_minus_self", Sp - Ss)):
        row[f"f2_ledger_{nm}_medN"] = med(v / N, a2)
        row[f"f2_ledger_{nm}_medabs_rel"] = med(np.abs(v) / np.maximum(np.abs(Se), 1e-300), a2 & (Se != 0))
    row["f2_frac_self_dominates"] = frac(np.abs(Se - Ss) < np.abs(Ss), a2)
    neg = a2 & (Ss < 0) & (np.abs(Ss) > fS) & (np.abs(Se) > fS)
    row["f2_n_Sself_neg"] = int(neg.sum())
    row["f2_frac_Gown_neg_given_Sself_neg"] = frac(Se < 0, neg)
    # Snake c2: the valley rule (seat = 2 alpha m), AUC of seat for G_full > 0
    if arm not in ("LR", "R"):
        for name in ("c2", "f2"):
            q_ = sl(name)
            al_ = alive[q_] & (np.abs(u["G_full"][q_]) > fl["G"][name])
            seat = 2 * u["alpha"][q_] * u["m"][q_]
            row[f"{name}_seat_auc"] = auc(seat[al_], u["G_full"][q_][al_] > 0)
            row[f"{name}_frac_sink_above_valley"] = frac(u["G_full"][q_] > 0, al_ & (seat > -math.pi / 2))
            row[f"{name}_frac_sink_below_valley"] = frac(u["G_full"][q_] > 0, al_ & (seat <= -math.pi / 2))
            row[f"{name}_n_below_valley"] = int((al_ & (seat <= -math.pi / 2)).sum())
    return row


# --------------------------------------------------------------------------
# replay: the engine itself
# --------------------------------------------------------------------------

@torch.no_grad()
def bundle_stats(Bd, chunk=50):
    """(R, 232) mean, sd, P(z > 0) of every unit over the run's 1200 images (x positions), float32
    forward (TF32 off here, restored for the training steps) accumulated in float64; pure evaluation."""
    R = Bd.R
    s1 = torch.zeros(R, NU, dtype=DT, device=Bd.device); s2 = torch.zeros_like(s1); pos = torch.zeros_like(s1)
    tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cudnn.allow_tf32 = False
    try:
        for i0, i1 in DV.chunks(N, chunk):
            o = E.forward(Bd.P, Bd.X[:, i0:i1], Bd.act)
            b = i1 - i0
            for z, q in ((o[0], slice(0, 16)), (o[2], slice(16, 32))):
                zr = z.reshape(b, R, CH, -1).double()
                s1[:, q] += zr.sum((0, 3)); s2[:, q] += (zr * zr).sum((0, 3)); pos[:, q] += (zr > 0).sum((0, 3))
            for z, q in ((o[4], sl("f1")), (o[6], sl("f2"))):
                zr = z.double()                                  # (R, b, 100)
                s1[:, q] += zr.sum(1); s2[:, q] += (zr * zr).sum(1); pos[:, q] += (zr > 0).sum(1)
    finally:
        torch.backends.cudnn.allow_tf32 = tf32
    d = den(Bd.device)
    m = s1 / d
    sd = (s2 / d - m * m).clamp_min(0).sqrt()
    alpha = torch.cat([Bd.act.alpha(l).double() for l in range(4)], 1)
    return tonp(m), tonp(sd), tonp(pos / d), tonp(alpha)


def load_bundle(slots, task: int, device, graph: bool):
    """The engine bundle with every slot at the end of `task` (cnn_drive_verify_1009.replay)."""
    Bd = DV.engine_bundle(slots, device, graph=graph)
    if task == 0:
        return Bd                                            # host init, Adam and streams at their start
    tcs = set()
    with torch.no_grad():
        for r, (a, s) in enumerate(slots):
            st = torch.load(DV.ckpt_path(a, s, task), map_location="cpu", weights_only=False)
            assert st["arm"] == a and st["seed"] == s
            for dst, src in ((Bd.P, st["P"]), (Bd.m, st["m"]), (Bd.v, st["v"])):
                for q, x in zip(dst, src):
                    q[r].copy_(x)
            tcs.add(float(st["tc"]))
            for k in ("V", "ada", "fixA", "cval"):
                for dst, src in zip(getattr(Bd.act, k), st["act"][k]):
                    dst[r].copy_(src)
    assert len(tcs) == 1, tcs
    Bd.tc.fill_(tcs.pop())
    for s in Bd.useeds:
        for _ in range(task * EPOCHS):
            torch.randperm(N, generator=Bd.g_batch[s])
        for _ in range(task):
            CN.task_labels(Bd.g_lab[s])
    return Bd


def hook_orders(Bd):
    """Hash every batch order the bundle draws (in order) without changing the draws."""
    h = hashlib.sha256()
    orig = Bd.new_order

    def new_order():
        orig()
        h.update(Bd.order.cpu().numpy().tobytes())
    Bd.new_order = new_order
    return h


def run_replay(Bd, full: bool, record=True, log=print):
    """Train task t+1 from the loaded state exactly as the run did; record the 232 units at step 0 and
    REC_STEPS (eager steps, across the epoch boundaries 75 and 150) and, with `full`, at FULL_STEPS
    (whole epochs by CUDA graph)."""
    horder = hook_orders(Bd)
    rec = {0: bundle_stats(Bd)} if record else {}
    Bd.new_labels()
    hlab = hashlib.sha256(Bd.Y.cpu().numpy().tobytes()).hexdigest()
    step = 0
    t0 = time.time()
    while step < max(REC_STEPS):
        j = step % SPE
        if j == 0:
            Bd.new_order()
            Bd.acc_ep.zero_()
        Bd.step(j)
        step += 1
        if record and step in REC_STEPS:
            rec[step] = bundle_stats(Bd)
    if full:
        while step % SPE:
            Bd.step(step % SPE)
            step += 1
        while step < EPOCHS * SPE:
            Bd.run_epoch()
            step += SPE
            if record and step in FULL_STEPS:
                rec[step] = bundle_stats(Bd)
                log(f"  step {step} {time.time() - t0:.0f}s")
    Bd.__dict__.pop("new_order", None)                     # the hook closes over Bd: break the cycle
    return rec, time.time() - t0, {"labels_sha256": hlab, "orders_sha256": horder.hexdigest(), "steps": step}


def bit_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    a = a.detach().cpu(); b = b.detach().cpu()
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.dtype == torch.float32:
        return bool(torch.equal(a.view(torch.int32), b.view(torch.int32)))
    if a.dtype == torch.float64:
        return bool(torch.equal(a.view(torch.int64), b.view(torch.int64)))
    return bool(torch.equal(a, b))


def snapshot(Bd) -> dict:
    return {"P": [p.detach().cpu().clone() for p in Bd.P], "m": [q.cpu().clone() for q in Bd.m],
            "v": [q.cpu().clone() for q in Bd.v], "tc": float(Bd.tc),
            "act": {k: [t.cpu().clone() for t in getattr(Bd.act, k)] for k in ("V", "ada", "fixA", "cval")}}


def slot_state(snap: dict, r: int) -> dict:
    """One slot of a bundle snapshot in the saved-checkpoint layout."""
    return {"P": [t[r] for t in snap["P"]], "m": [t[r] for t in snap["m"]], "v": [t[r] for t in snap["v"]],
            "tc": snap["tc"], "act": {k: [t[r] for t in v] for k, v in snap["act"].items()}}


def compare_states(snap: dict, r: int, st: dict) -> dict:
    """Slot r of a bundle snapshot vs one run's saved state: P, m, v, tc, V, ada, fixA, cval, bitwise."""
    out = {"P": all(bit_equal(snap["P"][i][r], st["P"][i]) for i in range(10)),
           "m": all(bit_equal(snap["m"][i][r], st["m"][i]) for i in range(10)),
           "v": all(bit_equal(snap["v"][i][r], st["v"][i]) for i in range(10)),
           "tc": snap["tc"] == float(st["tc"])}
    for k in ("V", "ada", "fixA", "cval"):
        out[k] = all(bit_equal(snap["act"][k][l][r], st["act"][k][l]) for l in range(4))
    num = sum(float(((snap["P"][i][r].double() - st["P"][i].double()) ** 2).sum()) for i in range(10))
    dd = sum(float((st["P"][i].double() ** 2).sum()) for i in range(10))
    out["all"] = all(out[k] for k in ("P", "m", "v", "tc", "V", "ada", "fixA", "cval"))
    out["rel_norm_diff_P"] = math.sqrt(num / dd)
    return out


def fidelity(snap, slots, task: int) -> dict:
    out = {}
    for r, (a, s) in enumerate(slots):
        pth = DV.ckpt_path(a, s, task + 1)
        if pth.exists():
            st = torch.load(pth, map_location="cpu", weights_only=False)
            out[f"{a}_{s}"] = compare_states(snap, r, st)
    return out


def interval_label(arm: str, t: int) -> str:
    if t == 0:
        return "init_to_task1"
    if t >= LAST_TASK[arm]:
        return "continuation_beyond_original_run"          # same engine, nothing to compare with
    if (t + 1) in TASKS[arm]:
        return "replay_with_saved_end"                     # t1->t2, t2->t3 (bit-checkable)
    return "replay_without_saved_end"


def save_replay(slots, task: int, rec: dict, wall: float, fid: dict, meta: dict, tag: str = ""):
    steps = sorted(rec)
    floor_dm = meta.get("floor_dm")
    for arm in dict.fromkeys(a for a, s in slots):
        idx = [r for r, (a, s) in enumerate(slots) if a == arm]
        seeds = [slots[r][1] for r in idx]
        name = f"replay_{arm}_t{task:02d}{tag}.npz"
        np.savez_compressed(RAW / name, steps=np.array(steps), seeds=np.array(seeds), task=task,
                            wall_s=wall, bundle_size=len(slots), interval=interval_label(arm, task),
                            m=np.stack([rec[s][0][idx] for s in steps]),
                            sd=np.stack([rec[s][1][idx] for s in steps]),
                            pplus=np.stack([rec[s][2][idx] for s in steps]),
                            alpha=np.stack([rec[s][3][idx] for s in steps]),
                            floor_dm=np.array(floor_dm if floor_dm is not None else np.nan),
                            meta=json.dumps(meta),
                            fidelity=json.dumps({k: v for k, v in fid.items() if k.startswith(arm + "_")}))


def floor_dm_of(Bd) -> float:
    """Re-evaluation difference of the record at the same state (chunk 50 vs 40): the floor of dm."""
    a = bundle_stats(Bd, chunk=50)[0]
    b = bundle_stats(Bd, chunk=40)[0]
    return float(np.abs(a - b).max())


# --------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------

def check_f64(arms, device, out: Path):
    """K1-K4, K6, K7a, K7b (float64), one activation kind per process; calibrates the floors."""
    torch.backends.cudnn.allow_tf32 = False
    rep = {"checks": []}

    def add(name, ok, required=True, **kw):
        rep["checks"].append({"name": name, "pass": bool(ok), "required": required, **kw})
        print(f"[{'PASS' if ok else 'FAIL'}{'' if required else ' (info)'}] {name} "
              f"{json.dumps(kw, default=float)[:500]}", flush=True)

    kind = arms[0]
    states = {"LR": (("LR", 10, 1), ("LR", 15, 10)), "SNA": (("SNA", 10, 1),),
              "R": (("R", 10, 2), ("R", 10, 3))}[kind]
    calib = {"G": {k: [] for k in LAYERS}, "S": [], "ff": [], "D": [], "A": [], "J": [], "kappa": [],
             "m": [], "Gown": []}
    for arm, seed, t in states:
        t0 = time.time()
        S = DV.State(arm, seed, t, device)
        y = old_labels(seed, t)
        rS = DV.State(arm, seed, ref_task(arm, t), device)
        rq = state_quantities(rS, y, perm_seed=None)
        q = state_quantities(S, y, ref=rq["cache"], perm_seed=1020_000 + seed * 100 + t, perm_push=N_PERM)
        q2 = state_quantities(S, y, ref=rq["cache"], chunk=120, jchunk=60)       # another chunking
        gperm = torch.Generator().manual_seed(4040 + seed)
        perm = torch.randperm(N, generator=gperm)
        Sp = DV.State.__new__(DV.State); Sp.__dict__.update(S.__dict__); Sp.X = S.X[perm.to(device)]
        rq_p = dict(A4=rq["cache"]["A4"][perm.to(device)], W5c=rq["cache"]["W5c"])
        qp = state_quantities(Sp, y[perm], ref=rq_p)                            # images and labels permuted
        sc, sc2, u, u2 = q["scal"], q2["scal"], q["units"], q2["units"]
        scp, up = qp["scal"], qp["units"]
        # ---- calibration of atol: the same quantity summed in two other orders (chunking, image order)
        def dd(k):
            return max(maxdiff(u[k], u2[k]), float(np.nanmax(np.abs(u[k] - up[k]))) if np.isfinite(u[k]).any() else 0.0)
        def ds(k):
            if not (np.isfinite(sc[k]) and np.isfinite(sc2[k]) and np.isfinite(scp[k])):
                return 0.0
            return max(abs(sc[k] - sc2[k]), abs(sc[k] - scp[k]))
        cal = {"G_full": dd("G_full"), "G_own": dd("G_own"), "G_up": dd("G_up"), "S_exact": dd("S_exact"),
               "m": dd("m"), "A": ds("A"), "D": ds("D"), "J": ds("J"), "kappa": ds("kappa"), "ff": ds("F_norm2")}
        for k in ("A", "J", "kappa", "ff", "D", "m"):
            calib[k].append(cal[k])
        calib["Gown"].append(cal["G_own"])
        atol = {k: 10 * v for k, v in cal.items()}
        # ---- K1: closed forms = autograd; m from the mean patch = forward mean
        k1 = dict(
            f2=close(u["G_own"][sl("f2")], u["S_exact"] / N, 1e-10, atol["G_own"]),
            f1=close(u["G_own"][sl("f1")], u["S3"] / N, 1e-10, atol["G_own"]),
            conv_own_vs_jvp_block=close(u["G_own"][:32], u["G_own_jvp"][:32], 1e-10, atol["G_own"] + atol["G_full"]),
            fc_own_vs_jvp_block=close(u["G_own"][32:], u["G_own_jvp"][32:], 1e-10, atol["G_own"] + atol["G_full"]),
            m_patch=close(u["m_patch"], u["m"], 1e-10, atol["m"]))
        add("K1_closed_forms", all(k1.values()), arm=arm, seed=seed, t=t, parts=k1, atol=atol,
            f2_absdiff=sc["K1_f2_absdiff"], f2_scale=sc["K1_f2_scale"], f1_absdiff=sc["K1_f1_absdiff"],
            f1_scale=sc["K1_f1_scale"], own_jvp_absdiff=sc["K1_own_jvp_absdiff"],
            mpatch_absdiff=sc["K1_mpatch_absdiff"], m_scale=sc["K1_mpatch_scale"])
        # ---- K2a: JVP c1 / c2 = saved G_full
        sub = {"LR": "LR/measure", "R": "R/measure"}.get(arm, "measure")
        fn = SAVED / sub / f"measure_{arm}_s{seed}_t{t:02d}.npy"
        if fn.exists():
            d = np.load(fn, allow_pickle=True).item()
            Gs = np.asarray(d["G_full"], dtype=np.float64)
            Gj = u["G_full"][:32]
            diff = np.abs(Gj - Gs)
            for i, name in enumerate(("c1", "c2")):
                calib["G"][name].append(float(diff[16 * i:16 * i + 16].max()))
            add("K2a_jvp_equals_saved_G_full", close(Gj, Gs, 1e-8, 0.0), arm=arm, seed=seed, t=t,
                saved=str(fn.relative_to(ROOT)), sha256=hashlib.sha256(fn.read_bytes()).hexdigest(),
                rel=float(diff.max() / np.abs(Gs).max()), max_abs_diff=float(diff.max()),
                sign_agree=float((np.sign(Gj) == np.sign(Gs)).mean()),
                zbar_absdiff=float(np.abs(u["m"][:32] - d["zbar"]).max()))
        else:
            add("K2a_jvp_equals_saved_G_full", False, required=False, arm=arm, seed=seed, t=t,
                note="保存値との照合は未実施（参照ファイルなし）", missing=str(fn))
        # ---- K2b: reverse mode <grad m_i, grad L_u> for a few f1 / f2 (and c2) units
        units_b = [OFF[2] + j for j in (0, 37, 99)] + [OFF[3] + j for j in (0, 50, 99)] + [OFF[1] + 3]
        rb = []
        for ui in units_b:
            gm = mean_grad_unit(S.P, S.alpha, S.X, ui)
            val = float(sum((a * b).sum() for a, b in zip(gm, q["gs"])))
            rb.append((ui, val, float(u["G_full"][ui])))
        dif = [abs(v - w) for _, v, w in rb]
        for ui, v, w in rb:
            name = LAYERS[max(i for i in range(4) if OFF[i] <= ui)]
            calib["G"][name].append(abs(v - w))
        add("K2b_reverse_mode_equals_jvp", all(close(v, w, 1e-10, atol["G_full"]) for _, v, w in rb),
            arm=arm, seed=seed, t=t, units=[(int(a), b, c) for a, b, c in rb], max_abs_diff=max(dif))
        # ---- K2c: G_full = G_own + sum of upstream blocks; downstream / head blocks vanish
        k2c = close(u["G_full"], u["G_own"] + u["G_up"], 1e-10, atol["G_full"] + atol["G_own"] + atol["G_up"])
        for name in LAYERS:
            calib["G"][name].append(float(np.abs(u["G_full"] - u["G_own"] - u["G_up"])[sl(name)].max()))
        add("K2c_full_equals_own_plus_blocks", k2c and sc["zero_blocks_absmax"] == 0.0, arm=arm, seed=seed, t=t,
            absdiff=sc["K2c_absdiff"], scale=sc["K2c_scale"], zero_blocks_absmax=sc["zero_blocks_absmax"])
        # ---- K3: lemma identities
        dfn = sc["F_norm2"] > 10 * cal["ff"]
        k3 = dict(A_two_forms=close(sc["A"], sc["A_id"], 1e-10, atol["A"]),
                  A_eq_D_plus_J=close(sc["A"], sc["D"] + sc["J"], 1e-10, atol["A"] + atol["D"] + atol["J"]),
                  centering_sums_zero=sc["center_max"] <= 1e-12 * max(1.0, sc["normV"], sc["F_norm"]),
                  J_nonneg=sc["J"] >= -atol["J"], K_mean=sc["K_mean_absdiff"] <= 1e-10 * abs(sc["K_mean"]))
        add("K3_identities", all(k3.values()), arm=arm, seed=seed, t=t, parts=k3, A=sc["A"], A_id=sc["A_id"],
            D=sc["D"], J=sc["J"], center_max=sc["center_max"], F_norm2=sc["F_norm2"], kappa_defined=bool(dfn))
        # ---- K4: projection ledger controls
        qq = state_quantities(S, y, ref=q["cache"])                            # ref = the state itself
        s3 = qq["scal"]
        k4a = (abs(s3["kap_feat"]) <= atol["kappa"] + 1e-12 * abs(sc["kappa"]) and
               abs(s3["kap_read"]) <= atol["kappa"] + 1e-12 * abs(sc["kappa"]) and
               close(s3["kap_chance"], sc["kappa"], 1e-10, atol["kappa"])) if sc["F_norm2"] > 0 else True
        keys = ("G_full", "G_own", "G_up", "S_exact", "S_self", "S_oh", "S_proj", "m", "sd", "pplus")
        inv_joint = {k: close_scaled(qp["units"][k], u[k], 1e-9) for k in keys}
        for k in ("kappa", "A", "D", "J", "kap_chance", "kap_feat", "kap_read"):
            inv_joint[k] = close(qp["scal"][k], sc[k], 1e-9, 1e-9 * max(abs(sc["kappa"]) if k.startswith("kap")
                                                                      else abs(sc[k]), 1e-300))
        yl = y[torch.randperm(N, generator=gperm)]
        ql = state_quantities(S, yl, ref=rq["cache"])
        inv_label = {k: close(ql["units"][k], u[k], 1e-12, 0.0) for k in ("G_full", "G_own", "G_up", "S_exact")}
        changed = {k: abs(ql["scal"][k] - sc[k]) > 1e3 * atol.get(k, 0.0) + 1e-6 * abs(sc[k]) for k in ("kappa", "A")}
        a4_varies = bool(float(q["A4"].std(0).max()) > 0)       # image-constant features: F sees class counts only
        add("K4_ledger_controls", k4a and all(inv_joint.values()) and all(inv_label.values()), arm=arm, seed=seed, t=t,
            a4_varies_over_images=a4_varies,
            note="κ・A の変化は情報として記録、必須は G・S_exact の不変（label-only permutation; κ and A need not change, "
                 "e.g. image-constant features or a head orthogonal to the change）",
            ref_eq_current=dict(chance=s3["kap_chance"], feat=s3["kap_feat"], read=s3["kap_read"], kappa=sc["kappa"]),
            joint_perm_invariant=inv_joint, label_perm_invariant=inv_label, label_perm_changes=changed,
            kappa_labelperm=ql["scal"]["kappa"], decomp_absdiff=sc["decomp_absdiff"])
        del qq, qp, ql
        # ---- K7b: permutation null of kappa (features and F fixed)
        if sc["F_norm2"] > 0:
            add("K7b_permutation_null", np.isfinite(sc["kappa_null_quantile"]), arm=arm, seed=seed, t=t,
                kappa=sc["kappa"], null_q_1_5_50_95_99=sc["kappa_null_q"], null_mean=sc.get("kappa_null_mean"),
                null_sd=sc["kappa_null_sd"], kappa_null_quantile=sc["kappa_null_quantile"],
                mc_se_null_mean=sc["kappa_null_sd"] / math.sqrt(N_PERM),
                perm_head_frac_sink_q=sc["perm_push_frac_sink_q"],
                real_frac_sink=frac(u["S_exact"] > 0, u["alive"][sl("f2")] > 0),
                note="reference distribution: 0.5 not required (pi recomputed after the permutation)")
        print(f"  state {arm} s{seed} t{t}: {time.time() - t0:.1f}s", flush=True)
        if t == 1 or (arm == "R" and t == 2):
            check_k6(S, q, add, calib)
        if arm == states[0][0] and (t == states[0][2]):
            check_k7a(S, q, y, add, atol)
        torch.cuda.empty_cache()
    rep["calibration"] = {k: v for k, v in calib.items()}
    rep["all_pass"] = all(c["pass"] for c in rep["checks"] if c["required"])
    rep["cuda_max_mem_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20
    (out / f"checks_f64_{kind}.json").write_text(json.dumps(rep, indent=1, default=float))
    print("all_pass", rep["all_pass"], "max mem MB", rep["cuda_max_mem_mb"], flush=True)


def check_k6(S, q, add, calib):
    """K6: deterministic GD steps on L_u; -dm/eta -> G_full (fixed routing), free routing reported;
    auxiliary: mean over 20 label draws of the real-label directional derivative vs G_full."""
    t1 = time.time()
    u = q["units"]; gs = q["gs"]
    G = u["G_full"]
    routed = [route_idx(S.P, S.X[i0:i1], S.alpha) for i0, i1 in DV.chunks(N, 300)]
    m0r = tonp(m_of(S.P, S.alpha, S.X, routed=routed))
    m0f = tonp(m_of(S.P, S.alpha, S.X))
    res = {}
    for eta in (1e-4, 5e-5, 2.5e-5):
        P2 = [p - eta * g for p, g in zip(S.P, gs)]
        rr = -(tonp(m_of(P2, S.alpha, S.X, routed=routed)) - m0r) / eta
        rf = -(tonp(m_of(P2, S.alpha, S.X)) - m0f) / eta
        res[eta] = (rr, rf)
    per = {}
    ok_rich = True
    ok_raw = True
    etas = sorted(res)
    try:
        fG = json.loads((OUT / "floors.json").read_text())["G"]
    except (OSError, KeyError, ValueError):
        fG = {k: 0.0 for k in LAYERS}
    for name in LAYERS:
        q_ = sl(name)
        sc = np.abs(G[q_]).max()
        e_r = {eta: float(np.abs(res[eta][0][q_] - G[q_]).max() / sc) for eta in etas}
        e_f = {eta: float(np.abs(res[eta][1][q_] - G[q_]).max() / sc) for eta in etas}
        rich = 2 * res[etas[0]][0][q_] - res[etas[1]][0][q_]            # Richardson on the two smallest
        det = np.abs(G[q_]) > fG.get(name, 0.0)                       # units above the layer's G floor
        def unit_rel(x):
            v = np.abs(x - G[q_]) / np.maximum(np.maximum(np.abs(x), np.abs(G[q_])), 1e-300)
            return v[det]
        ur, uq = unit_rel(res[etas[0]][0][q_]), unit_rel(rich)
        per[name] = dict(err_fixed=e_r, err_free=e_f,
                         richardson_err=float(np.abs(rich - G[q_]).max() / sc),
                         conv_ratio=e_r[etas[1]] / max(e_r[etas[0]], 1e-300),
                         normalization="max_i |-dm_i/eta - G_i| / max_i |G_i| over the layer's units",
                         n_units=int(len(det)), n_indeterminate=int((~det).sum()), floor_G=fG.get(name, 0.0),
                         unit_rel_raw_eta_min_q=quart(ur), unit_rel_raw_eta_min_max=float(ur.max()) if len(ur) else float("nan"),
                         unit_frac_raw_le_1e3=float((ur <= 1e-3).mean()) if len(ur) else float("nan"),
                         unit_rel_richardson_q=quart(uq), unit_rel_richardson_max=float(uq.max()) if len(uq) else float("nan"),
                         unit_frac_richardson_le_1e3=float((uq <= 1e-3).mean()) if len(uq) else float("nan"))
        r_ok = per[name]["richardson_err"] <= 1e-3 and (e_r[etas[0]] < 1e-9 or 1.5 <= per[name]["conv_ratio"] <= 2.5)
        per[name]["richardson_pass"] = bool(r_ok)
        per[name]["registered_raw_pass"] = bool(e_r[etas[0]] <= 1e-3)
        ok_rich &= r_ok
        ok_raw &= per[name]["registered_raw_pass"]
    add("K6_gd_steps_converge_to_G_full", ok_rich, arm=S.arm, seed=S.seed, t=S.task, per_layer=per,
        registered_raw_pass=bool(ok_raw), richardson_pass=bool(ok_rich),
        criterion_registered="A1.2: fixed routing, -dm/eta at the finite steps; at eta = 2.5e-5 the layer-max-normalized "
                             "error max_i |(-dm_i/eta) - G_i| / max_i |G_i| <= 1e-3 (registered_raw_pass)",
        criterion_pass="approved change: the Richardson limit 2 r(2.5e-5) - r(5e-5) within 1e-3 of the layer scale and "
                       "the error halving with eta (ratio 1.5-2.5) unless at rounding level (richardson_pass = pass); the "
                       "limit is a check of G_full, not a finite GD step meeting the registered accuracy",
        seconds=time.time() - t1)
    # auxiliary: real-label directional derivatives
    gen = torch.Generator().manual_seed(1020_006 + S.seed)
    D = []
    for _ in range(20):
        yy = torch.randint(K, (N,), generator=gen).to(S.X.device)
        g = ce_grad(S.P, S.alpha, S.X, yy)
        D.append(tonp(g_dir_jvp(S.P, S.alpha, S.X, g)))
    D = np.stack(D)
    mu = D.mean(0); se = D.std(0, ddof=1) / math.sqrt(len(D))
    z = (mu - G) / np.where(se > 0, se, np.nan)
    ok_z = np.isfinite(z)
    add("K6aux_label_draws", True, required=False, arm=S.arm, seed=S.seed, t=S.task, n_draws=len(D),
        n_units=int(ok_z.sum()), n_indet_se0=int((~ok_z).sum()),
        abs_z_q=quart(np.abs(z[ok_z])), max_abs_z=float(np.nanmax(np.abs(z))),
        frac_abs_z_gt3=float((np.abs(z[ok_z]) > 3).mean()),
        note="E_y[grad L_y] = grad L_u exactly, so z should look standard normal (20 draws: t_19)")


def check_k7a(S, q, y, add, atol):
    """K7a: identity mutations of the head, and a one-unit synthetic leaky state."""
    u, sc = q["units"], q["scal"]
    dev = S.X.device
    keys = ("G_full", "G_own", "G_up", "S_exact", "S_self", "S_oh", "S_proj")
    # (1) W5 = 0
    S0 = DV.State.__new__(DV.State); S0.__dict__.update(S.__dict__)
    S0.P = [p.clone() for p in S.P]; S0.P[8].zero_()
    q0 = state_quantities(S0, y)
    z_ok = all(float(np.abs(q0["units"][k][:]).max()) == 0.0 for k in ("G_full", "G_own", "G_up"))
    add("K7a1_zero_head_kills_hidden_G", z_ok, arm=S.arm, seed=S.seed, t=S.task,
        maxabs={k: float(np.abs(q0["units"][k]).max()) for k in ("G_full", "G_own", "G_up", "S_exact")})
    # (2) a common row vector added to every class row of the head
    gq = torch.Generator().manual_seed(77)
    qv = (torch.randn(HID, generator=gq, dtype=DT) * float(S.P[8].abs().mean()) * 3).to(dev)
    S1 = DV.State.__new__(DV.State); S1.__dict__.update(S.__dict__)
    S1.P = [p.clone() for p in S.P]; S1.P[8] = S.P[8] + qv[None, :]
    q1 = state_quantities(S1, y)
    inv1 = {k: close_scaled(q1["units"][k], u[k], 1e-8) for k in keys}
    inv1["kappa"] = close(q1["scal"]["kappa"], sc["kappa"], 1e-10, atol.get("kappa", 0.0))
    add("K7a2_common_row_invariance", all(inv1.values()), arm=S.arm, seed=S.seed, t=S.task, parts=inv1,
        maxrel={k: float(np.nanmax(np.abs(q1["units"][k] - u[k]))) / max(float(np.nanmax(np.abs(u[k]))), 1e-300)
                if np.isfinite(u[k]).any() else 0.0 for k in keys})
    # (3) the same class permutation of head rows, bias and old labels
    pc = torch.randperm(K, generator=gq).to(dev)
    inv = torch.empty_like(pc); inv[pc] = torch.arange(K, device=dev)
    S2 = DV.State.__new__(DV.State); S2.__dict__.update(S.__dict__)
    S2.P = [p.clone() for p in S.P]; S2.P[8] = S.P[8][pc]; S2.P[9] = S.P[9][pc]
    q2 = state_quantities(S2, inv[y.to(dev)].cpu())
    inv2 = {k: close_scaled(q2["units"][k], u[k], 1e-8) for k in keys}
    for k in ("kappa", "A", "D", "J", "L_old"):
        inv2[k] = close(q2["scal"][k], sc[k], 1e-9, 10 * atol.get(k, atol.get("A", 0.0)))
    add("K7a3_class_permutation_invariance", all(inv2.values()), arm=S.arm, seed=S.seed, t=S.task, parts=inv2)
    # (4) one leaky unit with fixed inputs, K_n > 0, positive / negative preactivation states
    if DV.ACT["kind"] != "leaky":
        return
    gsyn = torch.Generator().manual_seed(5)
    n_ = 64
    x = torch.rand(n_, 3, generator=gsyn, dtype=DT) + 0.1            # positive inputs -> K_n >= 1
    v = torch.randn(K, 1, generator=gsyn, dtype=DT) * 2.0
    v = v - v.mean(0, keepdim=True)                                 # centred head, bias 0
    res = {}
    for sign in (1.0, -1.0):
        w = (sign * (torch.rand(1, 3, generator=gsyn, dtype=DT) + 0.5)).requires_grad_(True)
        b = torch.tensor([sign * 0.2], dtype=DT, requires_grad=True)
        z = x @ w.T + b                                             # (n, 1), sign fixed for all n
        a = torch.where(z > 0, z, 0.1 * z)
        l = a @ v.T
        Lu = (torch.logsumexp(l, 1) - l.mean(1)).mean()
        gw, gb = torch.autograd.grad(Lu, (w, b))
        xbar = x.mean(0)
        G_own_ag = float(gw[0] @ xbar + gb[0])                      # x-bar . dL/dw + dL/db
        with torch.no_grad():
            hq = head_quantities(x, z, a, l, w, v, torch.zeros(K, dtype=DT), torch.tensor([float("nan")], dtype=DT),
                                 torch.zeros(n_, dtype=torch.long))
        res["pos" if sign > 0 else "neg"] = dict(z_sign=float(torch.sign(z).mean()), G_own_autograd=G_own_ag,
                                                 G_own_closed=float(hq["S_exact"][0]) / n_, K_min=float(hq["Kn"].min()))
    ok4 = (res["pos"]["G_own_autograd"] > 0 and res["neg"]["G_own_autograd"] < 0
           and all(close(r["G_own_autograd"], r["G_own_closed"], 1e-10, 1e-15) for r in res.values()))
    add("K7a4_one_unit_leaky_sign", ok4, synthetic=res)


def check_replay(device, out: Path, which: str):
    """K5: replays of task 2 from the saved t1 equal the saved t2 bit for bit (P, m, v, tc, V, ada,
    fixA, cval); a run with records equals one without."""
    DV.set_activation(["LR"])
    rep = {"checks": []}
    configs = {"R": [("R", 10)], "LR": [("LR", s) for s in SEEDS["LR"]],
               "SNA10": [("SNA", s) for s in SEEDS["SNA"]],
               "SNAKE40": [(a, s) for a in SNAKE for s in SEEDS[a]]}
    for name in which.split(","):
        slots = configs[name]
        required = name != "SNA10"
        t0 = time.time()
        Bd = load_bundle(slots, 1, device, graph=True)
        fdm = floor_dm_of(Bd)
        rec, wall, meta = run_replay(Bd, full=True, log=lambda s: print(name, s, flush=True))
        snap = snapshot(Bd)
        fid = fidelity(snap, slots, 1)
        ok = len(fid) == len(slots) and all(v["all"] for v in fid.values())
        res = {"name": f"K5_replay_bit_equal_{name}", "pass": ok, "required": required, "n_slots": len(slots),
               "n_all_equal": sum(v["all"] for v in fid.values()),
               "n_P_equal": sum(v["P"] for v in fid.values()),
               "max_rel_norm_diff_P": max(v["rel_norm_diff_P"] for v in fid.values()),
               "record_steps": sorted(rec), "record_steps_ok": sorted(rec) == sorted((0,) + REC_STEPS + FULL_STEPS),
               "wall_s": wall, "hashes": meta, "floor_dm": fdm,
               "failed_fields": {k: [f for f in ("P", "m", "v", "tc", "V", "ada", "fixA", "cval") if not v[f]]
                                 for k, v in fid.items() if not v["all"]}}
        print(f"[{'PASS' if ok else 'FAIL'}] K5 {name}: {res['n_all_equal']}/{len(slots)} bit equal (P {res['n_P_equal']}), "
              f"max rel {res['max_rel_norm_diff_P']:.3e}, {time.time() - t0:.0f}s", flush=True)
        if ok:
            meta["floor_dm"] = fdm
            save_replay(slots, 1, rec, wall, fid, meta, "" if name != "SNA10" else "_b10")
        del Bd
        torch.cuda.empty_cache()
        # a run without records: 200 steps (all configs) and the whole task (LR)
        if ok and name != "SNA10":
            B1 = load_bundle(slots, 1, device, graph=False)
            run_replay(B1, full=False, record=True)
            s1 = snapshot(B1); del B1
            B2 = load_bundle(slots, 1, device, graph=False)
            run_replay(B2, full=False, record=False)
            s2 = snapshot(B2); del B2
            eq200 = all(compare_states(s1, r, slot_state(s2, r))["all"] for r in range(len(slots)))
            full_eq = None
            if name in ("LR", "R"):
                B3 = load_bundle(slots, 1, device, graph=True)
                run_replay(B3, full=True, record=False)
                s3 = snapshot(B3); del B3
                full_eq = all(compare_states(snap, r, slot_state(s3, r))["all"] for r in range(len(slots)))
            okr = eq200 and (full_eq is not False)
            rep["checks"].append({"name": f"K5_record_vs_no_record_{name}", "pass": okr, "required": True,
                                  "equal_after_200": eq200, "equal_after_full_task": full_eq,
                                  "note": "full task without records compared only for LR; Snake: both runs equal "
                                          "the saved t02 (made without records)"})
            print(f"[{'PASS' if okr else 'FAIL'}] K5 record vs no record {name}: 200 {eq200} full {full_eq}", flush=True)
        rep["checks"].append(res)
        torch.cuda.empty_cache()
    rep["all_pass"] = all(c["pass"] for c in rep["checks"] if c["required"])
    rep["cuda_max_mem_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20
    (out / f"checks_replay_{which.replace(',', '_')}.json").write_text(json.dumps(rep, indent=1, default=float))
    print("all_pass", rep["all_pass"], flush=True)


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------

def make_floors():
    """floors.json from the f64 check parts: G per layer = largest independent-route difference (K2a,
    K2b, K2c); S = N x the f2 G floor; atol = 10 x the other-summation-order difference."""
    fl = floors_default()
    G = {k: [] for k in LAYERS}
    cal = {k: [] for k in ("A", "J", "kappa", "ff", "D", "m", "Gown")}
    for fn in OUT.glob("checks_f64_*.json"):
        d = json.loads(fn.read_text())
        for k in LAYERS:
            G[k] += d["calibration"]["G"][k]
        for k in cal:
            cal[k] += d["calibration"][k]
    if not any(G.values()):
        return None
    fl["G"] = {k: float(max(v)) if v else 0.0 for k, v in G.items()}
    fl["S"] = N * max(fl["G"]["f2"], 10 * max(cal["Gown"]) if cal["Gown"] else 0.0)
    fl["ff"] = 10 * max(cal["ff"]) if cal["ff"] else 0.0
    fl["D"] = 10 * max(cal["D"]) if cal["D"] else 0.0
    fl["atol"] = {k: 10 * max(v) for k, v in cal.items() if v}
    (OUT / "floors.json").write_text(json.dumps(fl, indent=1))
    return fl


def cmd_measure(arms, seeds_arg, device):
    torch.backends.cudnn.allow_tf32 = False
    fl = read_floors()
    (OUT / "rows_measure").mkdir(parents=True, exist_ok=True)
    for arm in arms:
        seeds = parse_list(seeds_arg) if seeds_arg else SEEDS[arm]
        for seed in seeds:
            fn_units = RAW / f"units_{arm}_s{seed}.npz"
            fn_rows = OUT / "rows_measure" / f"{arm}_s{seed}.json"
            if fn_units.exists() and fn_rows.exists():
                print(f"skip {arm} s{seed} (done)", flush=True)
                continue
            ts = (0,) + TASKS[arm]
            store = {}
            srows = []
            ref = None
            for t in ts:
                t0 = time.time()
                S = DV.State(arm, seed, t, device)
                q = state_quantities(S, old_labels(seed, t), ref=ref, perm_seed=1020_000 + seed * 100 + t,
                                     perm_push=200)
                row = summarize_state(arm, seed, t, q, fl)
                row["seconds"] = time.time() - t0
                srows.append(row)
                for k, v in q["units"].items():
                    store.setdefault(k, []).append(v)
                ref = q["cache"]
                print(f"{arm} s{seed} t{t}: kappa {row['kappa']:.4g} lb {row['kappa_lb']:.4g} J/D "
                      f"{row['kappa_over_lb']:.3g} f2 {row['f2_frac_Gfull_pos']:.2f} f1 {row['f1_frac_Gfull_pos']:.2f} "
                      f"c2 {row['c2_frac_Gfull_pos']:.2f} c1 {row['c1_frac_Gfull_pos']:.2f} alive "
                      f"{row['f2_n_alive']}/{row['f1_n_alive']}/{row['c2_n_alive']}/{row['c1_n_alive']} "
                      f"{row['seconds']:.1f}s", flush=True)
                del S, q
            np.savez_compressed(fn_units, t=np.array(ts), **{k: np.stack(v) for k, v in store.items()})
            fn_rows.write_text(json.dumps(srows, indent=1, default=float))
            torch.cuda.empty_cache()
    print("done; max mem MB", torch.cuda.max_memory_allocated() / 2 ** 20, flush=True)


def bundle_groups(arms):
    snake = [a for a in SNAKE if a in arms]
    groups = [[a] for a in arms if a not in SNAKE]
    if snake:
        assert snake == list(SNAKE), "Snake arms replay in the training's 40-slot bundle (K5)"
        groups.append(snake)
    return groups


def cmd_replay(arms, tasks_arg, full: bool, device, init: bool):
    for grp in bundle_groups(arms):
        slots = [(a, s) for a in grp for s in SEEDS[a]]
        tasks = [0] if init else (parse_list(tasks_arg) if tasks_arg else list(TASKS[grp[0]]))
        for t in tasks:
            if t > 0 and t not in TASKS[grp[0]]:
                continue
            names = [RAW / f"replay_{a}_t{t:02d}.npz" for a in grp]
            if all(p.exists() for p in names):
                have = np.load(names[0])["steps"]
                if (not full) or max(have) >= EPOCHS * SPE:
                    print(f"skip {grp} t{t} (done)", flush=True)
                    continue
            Bd = load_bundle(slots, t, device, graph=full)
            fdm = floor_dm_of(Bd)
            rec, wall, meta = run_replay(Bd, full=full, log=lambda s: print(grp, t, s, flush=True))
            fid = fidelity(snapshot(Bd), slots, t) if full else {}
            meta["floor_dm"] = fdm
            save_replay(slots, t, rec, wall, fid, meta, "")
            nb = sum(v["all"] for v in fid.values())
            print(f"replay {grp} t{t} -> t{t + 1}: {len(slots)} slots, {wall:.0f}s, floor_dm {fdm:.2e}"
                  + (f", vs saved t{t + 1}: {nb}/{len(fid)} bit equal" if fid else ""), flush=True)
            del Bd
            gc.collect()
            torch.cuda.empty_cache()
    print("done; max mem MB", torch.cuda.max_memory_allocated() / 2 ** 20, flush=True)


def cmd_diag(device):
    """(i) Snake t1, 4 arms: first step split dm_P = m(P1, a0) - m(P0, a0), dm_alpha = m(P1, a1) - m(P1, a0).
    (ii) LR t1: the same first batch with the old first moment m set to 0 (a copy; not the replay)."""
    out = {}
    slots = [(a, s) for a in SNAKE for s in SEEDS[a]]
    Bd = load_bundle(slots, 1, device, graph=False)
    st0 = Bd.act.state()
    m0 = bundle_stats(Bd)[0]
    Bd.new_labels(); Bd.new_order(); Bd.acc_ep.zero_()
    Bd.step(0)
    m11 = bundle_stats(Bd)[0]
    st1 = Bd.act.state()
    Bd.act.load_state(st0)
    m10 = bundle_stats(Bd)[0]
    Bd.act.load_state(st1)
    np.savez_compressed(RAW / "diag_snake_t01_first_step.npz", slots=np.array([f"{a}_{s}" for a, s in slots]),
                        m0=m0, m_P1_a1=m11, m_P1_a0=m10)
    out["snake"] = {"dm_P_plus_dm_alpha_equals_total": float(np.abs((m10 - m0) + (m11 - m10) - (m11 - m0)).max())}
    del Bd
    slots = [("LR", s) for s in SEEDS["LR"]]
    for zero_m in (False, True):
        Bd = load_bundle(slots, 1, device, graph=False)
        if zero_m:
            with torch.no_grad():
                for q in Bd.m:
                    q.zero_()
        m0 = bundle_stats(Bd)[0]
        Bd.new_labels(); Bd.new_order(); Bd.acc_ep.zero_()
        Bd.step(0)
        m1 = bundle_stats(Bd)[0]
        np.savez_compressed(RAW / f"diag_LR_t01_first_step{'_m0' if zero_m else ''}.npz",
                            seeds=np.array(SEEDS["LR"]), m0=m0, m1=m1)
        del Bd
    (OUT / "diag.json").write_text(json.dumps(out, indent=1))
    print("diag done", out, flush=True)


def cmd_merge():
    """checks.json from the check parts, floors.json, and rows.json (measure rows + replay columns)."""
    rep = {"checks": [], "parts": []}
    for fn in sorted(OUT.glob("checks_*.json")):
        d = json.loads(fn.read_text())
        rep["parts"].append(fn.name)
        rep["checks"] += d["checks"]
    rep["all_pass"] = bool(rep["checks"]) and all(c["pass"] for c in rep["checks"] if c.get("required", True))
    rep["criterion_changes"] = [
        {"check": "K6_gd_steps_converge_to_G_full",
         "first_version": "fixed routing, eta 2.5e-5: max_i |(-dm_i/eta) - G_i| / max_i |G_i| <= 1e-3 per layer",
         "first_version_result": "SNA s10 t1 failed (c2 1.15e-3, f1 2.28e-3, f2 4.34e-3); LR and R passed",
         "now": "Richardson limit within rtol 1e-3 of the layer scale and first-order convergence (error ratio 1.5-2.5)",
         "why": "the raw error is the O(eta) curvature term (it halves exactly with eta: ratio 2.00); the Richardson "
                "limit agrees with G_full to 7.5e-6 / 1.0e-5 / 1.3e-5 (c2 / f1 / f2) in SNA s10 t1",
         "approved_by_parent": "2026-10-20 (Fable, after the interim report)", "raw_values": "per_layer in the K6 entries"},
        {"check": "K4_ledger_controls (label-only permutation)",
         "first_version": "kappa and A must change under a label-only permutation",
         "first_version_result": "R s10 t3 failed: kappa unchanged to 1e-17",
         "now": "kappa and A must change only when the f2 features vary over images",
         "why": "R t3: f1 is dead (a3 = 0), a4 = relu(b4) is image-constant, F = (n_c - N/K) a^T depends on the class "
                "counts only; the spec says kappa and A 'may' change, S_exact and G must not",
         "approved_by_parent": "2026-10-20 (Fable, after the interim report)", "raw_values": "a4_varies_over_images, "
                                                                                          "label_perm_changes in K4"}]
    k6 = [c for c in rep["checks"] if c["name"] == "K6_gd_steps_converge_to_G_full"]
    rep["K6_registered_raw_pass_all"] = all(c.get("registered_raw_pass", False) for c in k6) if k6 else None
    rep["K6_richardson_pass_all"] = all(c.get("richardson_pass", c["pass"]) for c in k6) if k6 else None
    rep["all_pass_with_registered_K6"] = (bool(rep["checks"]) and all(
        (c.get("registered_raw_pass", c["pass"]) if c["name"] == "K6_gd_steps_converge_to_G_full" else c["pass"])
        for c in rep["checks"] if c.get("required", True)))
    rep["K6_summary"] = ("登録基準（固定経路、η = 2.5e−5 の実差分、層の |G| の最大で正規化した相対誤差 ≤ 1e−3）: " +
                         ", ".join(f"{c['arm']} s{c['seed']} t{c['t']} " +
                                   ("達" if c.get("registered_raw_pass") else "未達（" + "・".join(
                                       f"{L} {c['per_layer'][L]['err_fixed']['2.5e-05']:.1e}" for L in LAYERS
                                       if not c["per_layer"][L].get("registered_raw_pass", True)) + "）")
                                   for c in k6) +
                         "。極限の検算（Richardson、承認済みの判定）: " +
                         ("全部 PASS" if rep["K6_richardson_pass_all"] else "FAIL あり") +
                         "。all_pass は承認済みの判定、all_pass_with_registered_K6 は登録基準の K6 で数えた値。") if k6 else None
    fl = make_floors()
    rep["floors"] = fl
    (OUT / "checks.json").write_text(json.dumps(rep, indent=1, default=float))
    print("checks:", [(c["name"], c["pass"]) for c in rep["checks"]], "all_pass", rep["all_pass"])
    rows = []
    for fn in sorted((OUT / "rows_measure").glob("*.json")) if (OUT / "rows_measure").exists() else []:
        rows += json.loads(fn.read_text())
    if not rows:
        return
    units = {}
    for r in rows:
        key = (r["arm"], r["seed"])
        if key not in units:
            units[key] = np.load(RAW / f"units_{r['arm']}_s{r['seed']}.npz")
    reps = {}
    for fn in RAW.glob("replay_*_t*.npz"):
        if fn.stem.endswith("_b10"):
            continue
        arm, tt = fn.stem.split("_")[1], int(fn.stem.split("_t")[-1])
        reps[(arm, tt)] = np.load(fn)
    for r in rows:
        r["n_wrong_old"] = int(round(N * (1.0 - r["acc_old"])))
        add_replay_columns(r, units[(r["arm"], r["seed"])], reps.get((r["arm"], r["t"])), fl)
    (OUT / "rows.json").write_text("[\n" + ",\n".join(json.dumps(r, default=float, separators=(",", ":"))
                                                     for r in rows) + "\n]\n")      # one row per line
    print("rows.json:", len(rows), "rows;", sum(1 for r in rows if "rp_f2_frac_sink200" in r), "with replay")


def add_replay_columns(r: dict, U, rp, fl):
    if rp is None:
        return
    seeds = list(rp["seeds"])
    if r["seed"] not in seeds:
        return
    j = seeds.index(r["seed"])
    ti = list(U["t"]).index(r["t"])
    alive = U["alive"][ti] > 0
    Gf, Go = U["G_full"][ti], U["G_own"][ti]
    steps = list(rp["steps"])
    m = rp["m"][:, j]; sd = rp["sd"][:, j]; pp = rp["pplus"][:, j]
    fdm = float(rp["floor_dm"])
    fdm = 0.0 if not np.isfinite(fdm) else fdm
    dm = {s: m[steps.index(s)] - m[0] for s in steps}
    r["rp_interval"] = str(rp["interval"])
    r["rp_steps_max"] = int(max(steps))
    r["rp_floor_dm"] = fdm
    for name in LAYERS:
        q = sl(name)
        al_ = alive[q]
        fg = fl["G"][name]
        kf = al_ & (np.abs(Gf[q]) > fg)
        ko = al_ & (np.abs(Go[q]) > fg)
        for s in (1, 200):
            d = dm[s][q]
            det = np.abs(d) > fdm
            r[f"rp_{name}_frac_sink{s}"] = frac(d < 0, al_ & det)
            r[f"rp_{name}_n_indet{s}"] = int((al_ & ~det).sum())
            r[f"rp_{name}_agree_Gfull_{s}"] = agree(-Gf[q], np.where(det, d, 0.0), kf)[0]
            r[f"rp_{name}_agree_Gown_{s}"] = agree(-Go[q], np.where(det, d, 0.0), ko)[0]
            r[f"rp_{name}_all_frac_sink{s}"] = frac(d < 0, det)
        d200 = dm[200][q]
        r[f"rp_{name}_sink200_med"] = med(-d200, al_)
        r[f"rp_{name}_sink200_rel_med"] = med(-d200 / np.maximum(sd[0][q], 1e-300), al_ & (sd[0][q] > 0))
        r[f"rp_{name}_sink1_med"] = med(-dm[1][q], al_)
        i200 = steps.index(200)
        r[f"rp_{name}_p_up_200"] = int(((pp[0][q] < 0.5) & (pp[i200][q] >= 0.5)).sum())
        r[f"rp_{name}_p_down_200"] = int(((pp[0][q] >= 0.5) & (pp[i200][q] < 0.5)).sum())
        if max(steps) >= EPOCHS * SPE:
            dend = dm[EPOCHS * SPE][q]
            rest = dend - d200
            sank = al_ & (d200 < -fdm)
            den_ = np.abs(d200[sank]).sum()
            r[f"rp_{name}_n_sank200"] = int(sank.sum())
            r[f"rp_{name}_frac_sink_end"] = frac(dend < 0, al_ & (np.abs(dend) > fdm))
            r[f"rp_{name}_return_agg"] = float(rest[sank].sum() / den_) if den_ > fdm else float("nan")
            r[f"rp_{name}_return_q"] = quart(rest[sank] / np.abs(d200[sank])) if sank.any() else [float("nan")] * 3
            r[f"rp_{name}_frac_return_ge1"] = frac(rest / np.maximum(np.abs(d200), 1e-300) >= 1, sank)
            r[f"rp_{name}_frac_rest_pos"] = frac(rest > 0, sank)
            r[f"rp_{name}_agree_Gfull_end"] = agree(-Gf[q], np.where(np.abs(dend) > fdm, dend, 0.0), kf)[0]
            r[f"rp_{name}_p_up_end"] = int(((pp[0][q] < 0.5) & (pp[-1][q] >= 0.5)).sum())


def kappa_R_state(S, y_old: torch.Tensor, chunk=100) -> dict:
    """A1.5 (optional): own-direction sensitivities R^own_nj = Df_n[u_j^own] (logits of image n along unit j's own
    mean direction) for every unit of c1 c2 f1 f2, their old-label alignment kappa_R,j = D^-1 sum_n <R^c_nj, y_n - e>
    (D = N(1 - 1/K)), rho_j, eps_j, the sign margin (1 - 1/K) kappa_R - rho e_pi; G_own = (1 - 1/K) kappa_R + eps.
    f2: D kappa_R = S_oh (check)."""
    P, al, X = S.P, S.alpha, S.X
    dev = X.device
    W1, b1, W2, b2, W3, b3, W4, b4, W5, b5 = P
    mp1 = torch.zeros(75, dtype=DT, device=dev); mp2 = torch.zeros(400, dtype=DT, device=dev)
    H2s, A3s, Ls = [], [], []
    with torch.no_grad():
        for i0, i1 in DV.chunks(N, 300):
            z1, h1, z2, h2, z3, a3, z4, a4, l = fwd(P, X[i0:i1], al)
            mp1 += F.unfold(X[i0:i1], CN.KERNEL, padding=CN.PAD).sum((0, 2))
            mp2 += F.unfold(h1, CN.KERNEL, padding=CN.PAD).sum((0, 2))
            H2s.append(h2); A3s.append(a3); Ls.append(l)
    ub1, ub2 = mp1 / (N * 1024), mp2 / (N * 256)
    h2bar, xbar = torch.cat(H2s).mean(0), torch.cat(A3s).mean(0)
    pi = torch.cat(Ls).softmax(1)
    Y = F.one_hot(y_old.to(dev), K).to(DT)
    R = torch.zeros(N, K, NU, dtype=DT, device=dev)
    for i0, i1 in DV.chunks(N, chunk):
        x = X[i0:i1]
        z1 = F.conv2d(x, W1, b1, padding=CN.PAD).requires_grad_(True)
        h1 = F.max_pool2d(DV.snake(z1, al[0], True), CN.POOL, CN.POOL)
        z2 = F.conv2d(h1, W2, b2, padding=CN.PAD)
        h2 = F.max_pool2d(DV.snake(z2, al[1], True), CN.POOL, CN.POOL).flatten(1)
        z3 = h2 @ W3.T + b3
        a3 = DV.snake(z3, al[2], False)
        z4 = a3 @ W4.T + b4
        l = DV.snake(z4, al[3], False) @ W5.T + b5
        with torch.no_grad():
            B1 = torch.einsum("k,nkq->nq", ub1, F.unfold(x, CN.KERNEL, padding=CN.PAD)) + 1.0      # (n, 1024)
            B2 = torch.einsum("k,nkq->nq", ub2, F.unfold(h1, CN.KERNEL, padding=CN.PAD)) + 1.0     # (n, 256)
            K3 = h2 @ h2bar + 1.0
            K4 = a3 @ xbar + 1.0
        for c in range(K):
            g1, g2, g3, g4 = torch.autograd.grad(l[:, c].sum(), (z1, z2, z3, z4), retain_graph=c < K - 1)
            with torch.no_grad():
                R[i0:i1, c, :16] = (g1.flatten(2) * B1[:, None, :]).sum(2)
                R[i0:i1, c, 16:32] = (g2.flatten(2) * B2[:, None, :]).sum(2)
                R[i0:i1, c, 32:132] = g3 * K3[:, None]
                R[i0:i1, c, 132:] = g4 * K4[:, None]
    Rc = R - R.mean(1, keepdim=True)
    Dn = N * (1 - 1.0 / K)
    kR = torch.einsum("ncj,nc->j", Rc, Y - 1.0 / K) / Dn
    eps = torch.einsum("ncj,nc->j", Rc, pi - Y) / N
    rho = (Rc ** 2).sum(1).mean(0).sqrt()
    e_pi = float(((pi - Y) ** 2).sum(1).mean().sqrt())
    G_own_R = torch.einsum("ncj,nc->j", R, pi - 1.0 / K) / N
    cR = torch.einsum("ncj,nc->j", Rc, Y - 1.0 / K) / ((Rc ** 2).sum((0, 1)).sqrt() * math.sqrt(Dn)).clamp_min(1e-300)
    return dict(kappa_R=tonp(kR), eps=tonp(eps), rho=tonp(rho), e_pi=e_pi, G_own_R=tonp(G_own_R), c_R=tonp(cR),
                margin=tonp((1 - 1.0 / K) * kR - rho * e_pi))


def cmd_kappaR(arms, device, tasks_arg=""):
    torch.backends.cudnn.allow_tf32 = False
    want = set(parse_list(tasks_arg)) if tasks_arg else None
    for arm in arms:
        for seed in SEEDS[arm]:
            fn = RAW / f"kappaR_{arm}_s{seed}.npz"
            if fn.exists():
                continue
            U = np.load(RAW / f"units_{arm}_s{seed}.npz")
            store = {}
            tsel = []
            for ti, t in enumerate(U["t"]):
                t = int(t)
                if want is not None and t not in want:
                    continue
                tsel.append(t)
                t0 = time.time()
                S = DV.State(arm, seed, t, device)
                d = kappa_R_state(S, old_labels(seed, t))
                Go = U["G_own"][ti]
                d["check_G_own_absdiff"] = float(np.abs(d["G_own_R"] - Go).max())
                d["check_G_own_scale"] = float(np.abs(Go).max())
                d["check_f2_Soh_absdiff"] = float(np.abs(d["kappa_R"][132:] * N * (1 - 1.0 / K) - U["S_oh"][ti]).max())
                d["check_f2_Soh_scale"] = float(np.abs(U["S_oh"][ti]).max())
                d["check_ledger_absdiff"] = float(np.abs((1 - 1.0 / K) * d["kappa_R"] + d["eps"] - Go).max())
                for k, v in d.items():
                    store.setdefault(k, []).append(v)
                print(f"kappaR {arm} s{seed} t{t}: G_own check {d['check_G_own_absdiff']:.1e}/{d['check_G_own_scale']:.1e} "
                      f"f2 S_oh {d['check_f2_Soh_absdiff']:.1e}/{d['check_f2_Soh_scale']:.1e} {time.time() - t0:.1f}s", flush=True)
                del S
            np.savez_compressed(fn, t=np.array(tsel), **{k: np.array(v) for k, v in store.items()})
            torch.cuda.empty_cache()


def head_push(A4, Wc, b, Kn, g4, Fm, ff, SF, alive2):
    """f2 push of the trained features under a given centred head (Wc, b): S_exact, the self form with
    kappa re-taken on this head, kappa, mean confidence; predictions recomputed from this head."""
    pi = (A4 @ Wc.T + b).softmax(1)
    S = (Kn[:, None] * g4 * (pi @ Wc)).sum(0)
    kap = float((Wc * Fm).sum()) / ff
    Ss = kap * (1 - 1.0 / K) * SF
    Sn, Ssn = tonp(S), tonp(Ss)
    return dict(frac_sink=frac(Sn > 0, alive2 & (Sn != 0)), medabs=med(np.abs(Sn), alive2),
                agree_self=agree(Ssn, Sn, alive2)[0], kappa=kap, conf=float(pi.max(1).values.mean()))


def cmd_ref(arms, device, n_draw=200):
    """Requests of the parent after the interim report (post hoc, not registered):
    (1) K7b reference: centred head rows permuted over classes (pi recomputed) at t in {1, 10} (Snake also
        30), seeds 10, 11; (2) an init-scale random head (||v^c|| of the seed's t0, t0 bias) on the trained
        features, LR / SNA t1; also a trained-norm random head; (4) the D <= 0 states: L, B, wrong answers and
        the CE of the wrong images."""
    torch.backends.cudnn.allow_tf32 = False
    out = []
    for arm in arms:
        ts = [1, 10] + ([30] if arm in SNAKE else [])
        for seed in (10, 11):
            S0 = DV.State(arm, seed, 0, device)
            nV0 = float((S0.P[8] - S0.P[8].mean(0, keepdim=True)).norm())
            b0 = S0.P[9].clone()
            for t in ts:
                S = DV.State(arm, seed, t, device)
                y = old_labels(seed, t).to(device)
                with torch.no_grad():
                    A3s, Z4s, A4s = [], [], []
                    for i0, i1 in DV.chunks(N, 300):
                        z1, h1, z2, h2, z3, a3, z4, a4, l = fwd(S.P, S.X[i0:i1], S.alpha)
                        A3s.append(a3); Z4s.append(z4); A4s.append(a4)
                    A3, Z4, A4 = torch.cat(A3s), torch.cat(Z4s), torch.cat(A4s)
                    W5c = S.P[8] - S.P[8].mean(0, keepdim=True)
                    b5 = S.P[9]
                    xbar = A3.mean(0); Kn = A3 @ xbar + 1.0
                    g4 = DV.dphi_b(Z4, S.alpha[3].view(1, -1))
                    Fm = (F.one_hot(y, K).to(DT) - 1.0 / K).T @ A4
                    ff = float((Fm * Fm).sum())
                    SF = (Kn[:, None] * g4 * A4).sum(0)
                    alive2 = tonp((Z4 > 0).sum(0) > 0).astype(bool)
                    real = head_push(A4, W5c, b5, Kn, g4, Fm, ff, SF, alive2)
                    gen = torch.Generator().manual_seed(2020_000 + 1000 * seed + t)
                    rec = {"perm": [], "rand_trained_norm": [], "rand_init_scale": []}
                    nV = float(W5c.norm())
                    for d in range(n_draw):
                        p_ = torch.randperm(K, generator=gen).to(device)
                        rec["perm"].append(head_push(A4, W5c[p_], b5, Kn, g4, Fm, ff, SF, alive2))
                        if t == 1 and arm in ("LR", "SNA"):
                            Rm = torch.randn(K, HID, generator=gen, dtype=DT).to(device)
                            Rm = Rm - Rm.mean(0, keepdim=True)
                            rec["rand_trained_norm"].append(head_push(A4, Rm * (nV / float(Rm.norm())), b5, Kn, g4,
                                                                      Fm, ff, SF, alive2))
                            rec["rand_init_scale"].append(head_push(A4, Rm * (nV0 / float(Rm.norm())), b0, Kn, g4,
                                                                    Fm, ff, SF, alive2))
                    row = dict(arm=arm, seed=seed, t=t, n_draw=n_draw, normV=nV, normV_t0=nV0, real=real)
                    for k, v in rec.items():
                        if not v:
                            continue
                        row[k] = {kk: quart([x[kk] for x in v]) for kk in ("frac_sink", "medabs", "agree_self", "kappa", "conf")}
                        row[k]["medabs_ratio_q"] = quart([x["medabs"] / real["medabs"] for x in v])
                        row[k]["frac_draws_kappa_pos"] = float(np.mean([x["kappa"] > 0 for x in v]))
                    out.append(row)
                    print(f"ref {arm} s{seed} t{t}: real sink {real['frac_sink']:.2f} perm sink "
                          f"{row['perm']['frac_sink'][1]:.2f} ratio {row['perm']['medabs_ratio_q'][1]:.3f} agree "
                          f"{row['perm']['agree_self'][1]:.2f}" + (f" init-scale sink {row['rand_init_scale']['frac_sink'][1]:.2f} conf "
                          f"{row['rand_init_scale']['conf'][1]:.3f}" if "rand_init_scale" in row else ""), flush=True)
                del S
            del S0
    # (4) the states with D <= 0
    dneg = []
    for fn in sorted((OUT / "rows_measure").glob("*.json")):
        for r in json.loads(fn.read_text()):
            if r["arm"] in arms and r["t"] > 0 and not (r["D"] > 0):
                S = DV.State(r["arm"], r["seed"], r["t"], device)
                y = old_labels(r["seed"], r["t"]).to(device)
                with torch.no_grad():
                    L = torch.cat([fwd(S.P, S.X[i0:i1], S.alpha)[8] for i0, i1 in DV.chunks(N, 300)])
                    ce = torch.logsumexp(L, 1) - L[torch.arange(N, device=device), y]
                    wrong = L.argmax(1) != y
                dneg.append(dict(arm=r["arm"], seed=r["seed"], t=r["t"], L=r["L_old"], B=r["B"], D=r["D"], J=r["J"],
                                 n_wrong=int(wrong.sum()), ce_sum_wrong=float(ce[wrong].sum()),
                                 ce_sum_right=float(ce[~wrong].sum()), ce_max=float(ce.max()),
                                 ce_wrong_median=float(ce[wrong].median()) if wrong.any() else float("nan"),
                                 logit_absmax=float(L.abs().max())))
                print("D<=0", dneg[-1], flush=True)
                del S
    tag = "LR" if arms == ["LR"] else "SNAKE"
    (OUT / f"ref_{tag}.json").write_text(json.dumps({"rows": out, "D_nonpositive": dneg}, indent=1, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["checks", "measure", "replay", "init", "diag", "merge", "kappaR", "ref"])
    ap.add_argument("--which", default="f64", help="checks: f64 | replay configs LR,SNA10,SNAKE40")
    ap.add_argument("--arms", default="LR")
    ap.add_argument("--seeds", default="")
    ap.add_argument("--tasks", default="")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mem-gb", type=float, default=4.0)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    OUT.mkdir(parents=True, exist_ok=True); RAW.mkdir(parents=True, exist_ok=True)
    if args.mode == "merge":
        cmd_merge()
        return
    device = H.setup(args.device)
    if device.type == "cuda":
        di = torch.cuda.current_device()
        total = torch.cuda.get_device_properties(di).total_memory / 2 ** 30
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.mem_gb / total), di)
    arms = args.arms.split(",")
    t0 = time.time()
    if args.mode == "checks" and args.which != "f64":
        check_replay(device, OUT, args.which)
    elif args.mode == "diag":
        DV.set_activation(["LR"])
        cmd_diag(device)
    else:
        DV.set_activation(arms if args.mode != "checks" else [{"SNA": "SNA"}.get(arms[0], arms[0])])
        if args.mode == "checks":
            check_f64(arms, device, OUT)
        elif args.mode == "measure":
            cmd_measure(arms, args.seeds, device)
        elif args.mode == "kappaR":
            cmd_kappaR(arms, device, args.tasks)
        elif args.mode == "ref":
            cmd_ref(arms, device)
        elif args.mode in ("replay", "init"):
            cmd_replay(arms, args.tasks, args.full, device, init=(args.mode == "init"))
    print(f"wall {time.time() - t0:.0f}s; max mem MB {torch.cuda.max_memory_allocated() / 2 ** 20:.0f}", flush=True)


if __name__ == "__main__":
    main()
