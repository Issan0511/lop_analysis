#!/usr/bin/env python3
"""c1_direction_1010 sec. 2.1 (CNN): the switch push of every c1 / c2 channel mean, split by the
downstream weight that carries it -- init + drift (the mu-path component) + rest.

    python src/c1_direction_1010.py checks  --states LR:10:10  --out results/c1_direction_1010/cnn
    python src/c1_direction_1010.py checks  --states SNA:10:10 --out results/c1_direction_1010/cnn
    python src/c1_direction_1010.py measure --arms LR --seeds 10-19 --tasks 1,2,5,10,20,30 \
        --out results/c1_direction_1010/cnn
    python src/c1_direction_1010.py measure --arms SNA,SNAc3,CV06FC3,CV3FC06 ... (own process)
    python src/c1_direction_1010.py measure --arms LRc --tasks 5,10,20,30 ...      (own process)
    python src/c1_direction_1010.py collect --out results/c1_direction_1010/cnn   (checks.json)

One process measures either Snake arms or one piecewise arm (the ACT global of
cnn_drive_verify_1009, whose State / activation / mean_grads / self_shape are reused unchanged).

Definitions (analysis/c1_direction_1010/derivation.md sec. 1).  The state is fixed (float64, its own
pool winners and gates).  L = L_u = (1/N) sum_n [logsumexp f_n - mean_c f_nc] (the exact expectation
of the CE over iid uniform labels) unless `labels` is given: then the ordinary CE on those labels,
mean over the N images (for the Q2 replay; everything else identical).
    delta3 = dL/dz3,  H2 = dL/dP2 = (W3^T delta3) reshaped,  delta2 = dL/dz2,
    H1 = dL/dP1 = conv_transpose(delta2, W2),  delta1 = dL/dz1.
For a variant W2v of the conv2 weight, H1(W2v) = conv_transpose(delta2, W2v) with delta2 held at
the state's value; back-propagating H1(W2v) from P1 to (W1, b1) on the state's graph gives
G_j(W2v) = <u_j, grad>.  G_j is linear in W2v, so W2 = W2^0 + dW2^drift + dW2^rest gives
G_j = G_j^0 + G_j^drift + G_j^rest exactly (eq. (2)), with
    dW2^drift[k] = <dW2[k], mu2^> mu2^,  mu2 = mean over images and the 16x16 positions of
    unfold(P1, 5, pad 2) (zero padding included, so zbar2_k = <W2[k], mu2> + b2_k exactly).
One layer up the same with W3 = W3^0 + dW3^drift + dW3^rest, mu3 = mean_n vec(P2), H2(W3v) back-
propagated from P2 to (W1, b1, W2, b2), and G_k = <u_k, that gradient> (u_k has W1 / b1 parts).

Sign convention (cnn_drive_verify_1009): G = <u, grad L>, u = gradient of the channel mean.
G > 0 is the SINKING side: an SGD step moves the mean by -eta G.
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
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "analysis" / "sna_cnn_cause_1009"))   # posthoc_lib, imported by V

import numpy as np                                   # noqa: E402
import torch                                         # noqa: E402
import torch.nn.functional as F                      # noqa: E402

# cnn_drive_verify_1009 prepends the (removed) sna worktree to sys.path; that path no longer exists,
# so `src` and `posthoc_lib` resolve to this worktree's copies (same files, merged to main).
from src import cnn_drive_verify_1009 as V           # noqa: E402
from src import rlcifar_cnn_0908 as CN               # noqa: E402
from src.cnn_center_1010 import train_channel_mean   # noqa: E402
import posthoc_lib as PL                             # noqa: E402

N, C, CH = V.N, V.C, V.CH
DT = V.DT
PAD, KER, POOL = CN.PAD, CN.KERNEL, CN.POOL
OFF = V.CONV_OFF

DATA = Path("/home/issan/Projects/obsidian-research-data")
CKPT_DIRS = {"LR": DATA / "cnn_drive_verify_1009" / "results" / "cnn_drive_verify_1009" / "LR" / "ckpt",
             "LRc": DATA / "cnn_center_1010" / "results" / "cnn_center_1010" / "LRc" / "ckpt"}
SNAKE_CKPT = DATA / "sna_cnn_cause_1009" / "results" / "sna_cnn_cause_1009" / "A" / "ckpt"
SNAKE_ARMS = ("SNA", "SNAc3", "CV06FC3", "CV3FC06")
REF = Path("/home/issan/Projects/claude/proj_004_drift/results/cnn_drive_verify_1009")   # read only

V.PIECEWISE["LRc"] = 0.1               # the centred leaky arm: leaky 0.1, same init as LR
CENTRED = {"LRc"}
MIX = (0.7, -1.3, 0.4)                 # C5 linearity: G(a W^0 + b drift + c rest) = a G^0 + b G^d + c G^r
VARIANTS = ("full", "init", "drift", "rest")


def ckpt_path(arm: str, seed: int, task: int) -> Path:
    return CKPT_DIRS.get(arm, SNAKE_CKPT) / f"{arm}_seed{seed}_t{task:02d}.pt"


V.ckpt_path = ckpt_path                # verify's State reads checkpoints through this module name


def saved_ref(kind: str, arm: str, seed: int, task: int):
    """cnn_drive_verify_1009's saved per-state dict (kind 'measure' or 'extra'), or None."""
    if arm == "LR":
        p = REF / "LR" / kind / f"{kind}_LR_s{seed}_t{task:02d}.npy"
    elif arm in SNAKE_ARMS:
        p = REF / kind / f"{kind}_{arm}_s{seed}_t{task:02d}.npy"
    else:
        return None
    return np.load(p, allow_pickle=True).item() if p.exists() else None


# --------------------------------------------------------------------------
# states
# --------------------------------------------------------------------------

_XC = {}


def arm_images(arm: str, seed: int, device, dtype=DT) -> torch.Tensor:
    """The run's 1200 inputs as trained: raw /255, or for LRc minus the CIFAR-10 training-set channel
    mean, subtracted in float32 exactly as BundleLRc does, then cast."""
    if arm not in CENTRED:
        return V.images(seed, device, dtype)
    key = (seed, str(device), dtype)
    if key not in _XC:
        cifar = PL.cifar()
        x32 = CN.images(cifar, CN.subset_idx(seed), device)                    # float32
        mu = train_channel_mean(cifar).to(device).view(1, 3, 1, 1)             # float32
        _XC[key] = (x32 - mu).to(dtype)
    return _XC[key]


def alpha_from_act(arm: str, act: dict, device, dtype=DT) -> list:
    """Per-channel alpha of a checkpoint's activation state (verify's State rule); NaN if piecewise."""
    out = []
    for l in range(4):
        if arm in V.PIECEWISE:
            out.append(torch.full((CN.WIDTHS[l],), float("nan"), dtype=dtype, device=device))
            continue
        Vv = act["V"][l].to(dtype)
        ada = act["ada"][l]
        cval = act["cval"][l].to(dtype)
        fixA = act["fixA"][l].to(dtype)
        out.append(torch.where(ada, (cval / Vv.sqrt()).clamp(V.LO, V.HI), fixA).to(device))
    return out


def state_from_dict(st: dict, device, X: torch.Tensor | None = None, task=None, dtype=DT) -> V.State:
    """A verify State from an in-memory dict in the checkpoint format (keys arm, seed, P, m, v, tc,
    act) and the images X (already centred or not; default: the arm's own inputs).  Lets replayed
    intermediate states be analysed without writing checkpoint files."""
    S = V.State.__new__(V.State)
    S.arm, S.seed, S.task, S.device = st["arm"], int(st["seed"]), task, device
    if S.arm in V.PIECEWISE:
        assert V.ACT["kind"] != "snake", "set_activation() for this arm first"
    else:
        assert V.ACT["kind"] == "snake", "set_activation() for this arm first"
    S.P = [p.detach().to(device, dtype) for p in st["P"]]
    S.m = [q.detach().to(device, dtype) for q in st["m"]] if st.get("m") is not None else None
    S.v = [q.detach().to(device, dtype) for q in st["v"]] if st.get("v") is not None else None
    S.tc = float(st.get("tc", float("nan")))
    S.alpha = alpha_from_act(S.arm, st["act"], device, dtype)
    S.X = X.to(device, dtype) if X is not None else arm_images(S.arm, S.seed, device, dtype)
    return S


def load_state(arm: str, seed: int, task: int, device) -> V.State:
    """task 0 = the host init (verify's State(arm, seed, 0): CN.init_params(seed), init alpha)."""
    if task == 0:
        S = V.State(arm, seed, 0, device)
        S.X = arm_images(arm, seed, device)
        return S
    st = torch.load(ckpt_path(arm, seed, task), map_location="cpu", weights_only=False)
    assert st["arm"] == arm and int(st["seed"]) == seed, (st["arm"], st["seed"])
    return state_from_dict(st, device, task=task)


# --------------------------------------------------------------------------
# passes
# --------------------------------------------------------------------------

@torch.no_grad()
def means_pass(P, al, X, chunk=200) -> dict:
    """Means over the images of a fixed net: mu2 (400, zero-pad incl.), mu3 (1024), pooled-feature
    means Pbar1 / Pbar2 (per channel, over images and positions), zbar of z1 / z2 / z3 and the mean
    input patch M (75, zero-pad incl.)."""
    dev = X.device
    acc = {k: torch.zeros(n, dtype=DT, device=dev) for k, n in
           (("mu2", CH * KER * KER), ("mu3", CN.FLAT), ("Pbar1", CH), ("Pbar2", CH),
            ("zbar1", CH), ("zbar2", CH), ("zbar3", CN.HIDDEN), ("M", 3 * KER * KER))}
    for i0, i1 in V.chunks(X.shape[0], chunk):
        x = X[i0:i1]
        z1 = F.conv2d(x, P[0], P[1], padding=PAD)
        P1 = F.max_pool2d(V.snake(z1, al[0], True), POOL, POOL)
        z2 = F.conv2d(P1, P[2], P[3], padding=PAD)
        P2 = F.max_pool2d(V.snake(z2, al[1], True), POOL, POOL)
        h2 = P2.flatten(1)
        z3 = h2 @ P[4].T + P[5]
        acc["mu2"] += F.unfold(P1, KER, padding=PAD).sum((0, 2))
        acc["mu3"] += h2.sum(0)
        acc["Pbar1"] += P1.sum((0, 2, 3)); acc["Pbar2"] += P2.sum((0, 2, 3))
        acc["zbar1"] += z1.sum((0, 2, 3)); acc["zbar2"] += z2.sum((0, 2, 3)); acc["zbar3"] += z3.sum(0)
        acc["M"] += F.unfold(x, KER, padding=PAD).sum((0, 2))
    n = X.shape[0]
    div = {"mu2": n * 256, "mu3": n, "Pbar1": n * 256, "Pbar2": n * 64, "zbar1": n * 1024,
           "zbar2": n * 256, "zbar3": n, "M": n * 1024}
    return {k: v / div[k] for k, v in acc.items()}


def grad_pass(S, W2var: dict, W3var: dict, M: torch.Tensor, mu2: torch.Tensor, labels=None,
              chunk=200) -> dict:
    """One pass over the images on the state's graph: the conv-parameter gradient of L, the
    back-propagated gradients of every W2 / W3 variant, and the side sums (D-bars, gates, K1).
    Post hoc sums: A-tilde_kj = (1/25) sum_{n,q} R1_j (ones *^T delta2_k), R1_j = gamma_j K1 at the
    c1 winners; the regression sums of H1_j on P1_j and of H2_k on P2_k; the own tangent of P2_k
    along u_k's (W2[k], b2[k]) part, R2own_k = gamma2_k K2 at the c2 winners, K2 = <mu2, P1patch> + 1."""
    P, al, X = S.P, S.alpha, S.X
    dev = S.device
    n_img = X.shape[0]
    W1r, b1r, W2r, b2r = (P[i].detach().clone().requires_grad_(True) for i in range(4))
    W3, b3, W4, b4, W5, b5 = P[4:]
    z = lambda *s: torch.zeros(*s, dtype=DT, device=dev)          # noqa: E731
    acc1 = {k: [torch.zeros_like(P[0]), torch.zeros_like(P[1])] for k in W2var}
    acc2 = {k: [torch.zeros_like(P[i]) for i in range(4)] for k in W3var}
    gconv = [torch.zeros_like(P[i]) for i in range(4)]
    H1sum = {k: z(CH) for k in W2var}
    H2sum = {k: z(CH) for k in W3var}
    side = {k: z(n) for k, n in (("D2", CH), ("D3", CN.HIDDEN), ("G_eq1", CH), ("gam1", CH),
                                  ("K1w", CH), ("gK1w", CH), ("gam2", CH), ("H1all", CH), ("H2all", CH))}
    # post hoc sums
    side["Atil"] = z(CH, CH)                                          # (k, j)
    for k in ("RP1", "RH1", "HP1", "PP1", "HH1", "Hs1", "Ps1",
              "R2P2own", "R2H2own", "HP2", "PP2", "HH2", "Hs2", "Ps2"):
        side[k] = z(CH)
    errH1 = errH2 = 0.0
    Lval = 0.0
    Mk = M.view(1, 3, KER, KER)
    one = torch.ones(1, dtype=DT, device=dev)
    ones_k = torch.ones(CH, 1, KER, KER, dtype=DT, device=dev)      # per-channel 5x5 all-ones kernel
    mu2k = mu2.view(1, CH, KER, KER)
    if labels is not None:
        labels = labels.to(dev).long()
        assert labels.shape == (n_img,)
    for i0, i1 in V.chunks(n_img, chunk):
        x = X[i0:i1]
        z1 = F.conv2d(x, W1r, b1r, padding=PAD)
        P1, idx1 = F.max_pool2d(V.snake(z1, al[0], True), POOL, POOL, return_indices=True)
        z2 = F.conv2d(P1, W2r, b2r, padding=PAD)
        P2, idx2 = F.max_pool2d(V.snake(z2, al[1], True), POOL, POOL, return_indices=True)
        z3 = P2.flatten(1) @ W3.T + b3
        z4 = V.snake(z3, al[2], False) @ W4.T + b4
        f = V.snake(z4, al[3], False) @ W5.T + b5
        if labels is None:
            L = V.uniform_loss(f) / n_img                              # L_u (registered default)
        else:
            L = F.cross_entropy(f, labels[i0:i1], reduction="sum") / n_img
        g = torch.autograd.grad(L, (W1r, b1r, W2r, b2r, z1, z2, z3, P1, P2), retain_graph=True)
        for a, b in zip(gconv, g[:4]):
            a += b
        d1, d2, d3, H1, H2 = g[4:]
        Lval += float(L.detach())
        # c1: H1 rebuilt from every W2 variant, back-propagated P1 -> (W1, b1)
        for k, W2v in W2var.items():
            H1v = F.conv_transpose2d(d2, W2v, padding=PAD)
            if k == "full":
                errH1 = max(errH1, float((H1v - H1).abs().max() / H1.abs().max()))
            gv = torch.autograd.grad(P1, (W1r, b1r), grad_outputs=H1v, retain_graph=True)
            acc1[k][0] += gv[0]; acc1[k][1] += gv[1]
            H1sum[k] += H1v.sum((0, 2, 3)).detach()
        # c2: H2 rebuilt from every W3 variant, back-propagated P2 -> (W1, b1, W2, b2)
        for k, W3v in W3var.items():
            H2v = (d3 @ W3v).view_as(P2)
            if k == "full":
                errH2 = max(errH2, float((H2v - H2).abs().max() / H2.abs().max()))
            gv = torch.autograd.grad(P2, (W1r, b1r, W2r, b2r), grad_outputs=H2v, retain_graph=True)
            for a, b in zip(acc2[k], gv):
                a += b
            H2sum[k] += H2v.sum((0, 2, 3)).detach()
        with torch.no_grad():
            side["D2"] += d2.sum((0, 2, 3))
            side["D3"] += d3.sum(0)
            side["H1all"] += H1.sum((0, 2, 3)); side["H2all"] += H2.sum((0, 2, 3))
            K1 = F.conv2d(x, Mk, one, padding=PAD)                          # <M, x(n,p)> + 1
            side["G_eq1"] += (d1 * K1).sum((0, 2, 3))                       # eq. (1): sum delta1 K1
            z1w = z1.detach().flatten(2).gather(2, idx1.flatten(2))          # z1 at the pool winners
            gam = V.dphi_b(z1w, al[0].view(1, -1, 1))
            K1w = K1.flatten(2).expand(-1, CH, -1).gather(2, idx1.flatten(2))
            side["gam1"] += gam.sum((0, 2)); side["K1w"] += K1w.sum((0, 2))
            side["gK1w"] += (gam * K1w).sum((0, 2))
            z2w = z2.detach().flatten(2).gather(2, idx2.flatten(2))
            gam2 = V.dphi_b(z2w, al[1].view(1, -1, 1))
            side["gam2"] += gam2.sum((0, 2))
            # ---- post hoc: c1 tangent R1 = gamma K1 at the winners, A-tilde, regression sums
            R1 = gam * K1w                                                     # (B, 16 j, 256 q)
            P1f = P1.detach().flatten(2); H1f = H1.flatten(2)
            side["RP1"] += (R1 * P1f).sum((0, 2)); side["RH1"] += (R1 * H1f).sum((0, 2))
            side["HP1"] += (H1f * P1f).sum((0, 2)); side["PP1"] += (P1f * P1f).sum((0, 2))
            side["HH1"] += (H1f * H1f).sum((0, 2))
            side["Hs1"] += H1f.sum((0, 2)); side["Ps1"] += P1f.sum((0, 2))
            Bk = F.conv_transpose2d(d2, ones_k, padding=PAD, groups=CH).flatten(2)   # ones *^T delta2_k
            side["Atil"] += torch.einsum("njq,nkq->kj", R1, Bk) / (KER * KER)
            # ---- post hoc: c2 own tangent R2own = gamma2 K2 at the c2 winners, regression sums
            K2 = F.conv2d(P1.detach(), mu2k, one, padding=PAD)                 # <mu2, P1patch> + 1
            K2w = K2.flatten(2).expand(-1, CH, -1).gather(2, idx2.flatten(2))
            R2o = gam2 * K2w                                                   # (B, 16 k, 64 r)
            P2f = P2.detach().flatten(2); H2f = H2.flatten(2)
            side["R2P2own"] += (R2o * P2f).sum((0, 2)); side["R2H2own"] += (R2o * H2f).sum((0, 2))
            side["HP2"] += (H2f * P2f).sum((0, 2)); side["PP2"] += (P2f * P2f).sum((0, 2))
            side["HH2"] += (H2f * H2f).sum((0, 2))
            side["Hs2"] += H2f.sum((0, 2)); side["Ps2"] += P2f.sum((0, 2))
        del z1, P1, z2, P2, z3, z4, f, L, g, d1, d2, d3, H1, H2
    side["gam1"] /= n_img * 256; side["K1w"] /= n_img * 256; side["gK1w"] /= n_img * 256
    side["gam2"] /= n_img * 64
    return {"acc1": acc1, "acc2": acc2, "gconv": gconv, "H1sum": H1sum, "H2sum": H2sum,
            "side": side, "errH1": errH1, "errH2": errH2, "L": Lval}


def project(dW: torch.Tensor, mu: torch.Tensor):
    """Row-wise split of dW (rows, d) into the component along mu^ and the rest."""
    muh = mu / mu.norm()
    proj = dW @ muh
    drift = proj[:, None] * muh[None, :]
    return proj, drift, dW - drift


# --------------------------------------------------------------------------
# one state
# --------------------------------------------------------------------------

def measure_state(S, S0, labels=None, U=None, chunk=200, saved=None, saved_extra=None,
                  mix=False, self_form=True) -> dict:
    """Everything of spec sec. 2.1 for one state S (S0 = the init of the same run, for W2^0 / W3^0).
    Returns a flat dict of numpy arrays / scalars; checks are the `chk_*` keys."""
    t0 = time.time()
    dev = S.device
    if U is None:
        U = V.mean_grads(S)
    A = means_pass(S.P, S.alpha, S.X, chunk)
    A0 = means_pass(S0.P, S0.alpha, S.X, chunk)             # the init net on the same inputs
    M = A["M"]
    W1, b1, W2, b2, W3, b3 = S.P[:6]
    W10, b10, W20, b20, W30, b30 = S0.P[:6]
    # W2 split (c1's carrier)
    dW2 = (W2 - W20).reshape(CH, -1)
    proj2, drift2, rest2 = project(dW2, A["mu2"])
    W2var = {"full": W2, "init": W20, "drift": drift2.view_as(W2), "rest": rest2.view_as(W2)}
    # W3 split (c2's carrier)
    dW3 = W3 - W30
    proj3, drift3, rest3 = project(dW3, A["mu3"])
    W3var = {"full": W3, "init": W30, "drift": drift3, "rest": rest3}
    if mix:
        W2var["mix"] = MIX[0] * W20 + MIX[1] * W2var["drift"] + MIX[2] * W2var["rest"]
        W3var["mix"] = MIX[0] * W30 + MIX[1] * drift3 + MIX[2] * rest3
    # ---- post hoc variants (coordinator, after the registered LR numbers): DC / AC buckets.
    # c1: per (k, j) block of 5x5 taps, DC = (block sum / 25) * ones, AC = the remainder; applied to
    # dW2, to W2^0 and to dW2^drift (the drift's block DC is -(c_k / |mu2|) * mean of mu2's j-block).
    dW2t = W2 - W20
    W2var["DC"] = dW2t.mean((2, 3), keepdim=True).expand_as(dW2t).contiguous()
    W2var["AC"] = dW2t - W2var["DC"]
    W2var["init_DC"] = W20.mean((2, 3), keepdim=True).expand_as(W20).contiguous()
    W2var["init_AC"] = W20 - W2var["init_DC"]
    dr = W2var["drift"]
    W2var["drift_DC"] = dr.mean((2, 3), keepdim=True).expand_as(dr).contiguous()
    W2var["drift_AC"] = dr - W2var["drift_DC"]
    # c2: per (u, k) the 64 positions of channel k in fc1's input, DC = (sum / 64) * ones, AC = rest
    def bucket3(Wm):
        dc = Wm.view(CN.HIDDEN, CH, 64).mean(2, keepdim=True).expand(-1, -1, 64).reshape(CN.HIDDEN, CN.FLAT)
        return dc.contiguous(), (Wm - dc)
    W3var["DC"], W3var["AC"] = bucket3(dW3)
    W3var["init_DC"], W3var["init_AC"] = bucket3(W30)
    B = grad_pass(S, W2var, W3var, M, A["mu2"], labels, chunk)
    # ---- G per variant
    ar = torch.arange(CH, device=dev)
    u1W = U[:CH, OFF[0]:OFF[1]].reshape(CH, CH, 3, KER, KER)[ar, ar]          # (16, 3, 5, 5)
    u1b = U[ar, OFF[1] + ar]                                                 # (16,)
    out = {"arm": S.arm, "seed": S.seed, "task": -1 if S.task is None else int(S.task),
           "loss": "L_u" if labels is None else "CE_labels", "L": B["L"]}
    G = {}
    for k, (gW1, gb1) in B["acc1"].items():
        G[f"G_c1_{k}"] = (gW1 * u1W).sum((1, 2, 3)) + gb1 * u1b
    for k, gs in B["acc2"].items():
        gvec = torch.cat([q.reshape(-1) for q in gs])
        G[f"G_c2_{k}"] = U[CH:] @ gvec
        # post hoc F: own = u_k's (W2[k], b2[k]) part, up = its (W1, b1) part (through P1)
        G[f"G_c2_own_{k}"] = U[CH:, OFF[2]:OFF[4]] @ gvec[OFF[2]:OFF[4]]
        G[f"G_c2_up_{k}"] = U[CH:, OFF[0]:OFF[2]] @ gvec[OFF[0]:OFF[2]]
    G_ag = U @ torch.cat([q.reshape(-1) for q in B["gconv"]])
    sd = B["side"]
    # ---- side quantities
    s = W2.sum((2, 3)); s0 = W20.sum((2, 3))                                  # (k, j) DC sums
    s3 = W3.view(CN.HIDDEN, CH, 64).sum(2)                                    # (u, k) DC sums
    mu2, mu3 = A["mu2"], A["mu3"]
    Mh = M / M.norm()
    vals = {
        **G, "G_ag": G_ag, "G_eq1": sd["G_eq1"],
        "Pbar1": A["Pbar1"], "Pbar2": A["Pbar2"], "Pbar1_init": A0["Pbar1"], "Pbar2_init": A0["Pbar2"],
        "c_k": -proj2, "c_u": -proj3,                                         # c := -<dW[row], mu^>
        "D2bar": sd["D2"], "D3bar": sd["D3"],
        "s_kj": s, "s0_kj": s0, "ds_kj": s - s0,
        "That_c1": s.T @ sd["D2"], "That0_c1": s0.T @ sd["D2"], "ThatD_c1": (s - s0).T @ sd["D2"],
        "That_c2": s3.T @ sd["D3"],
        "gamma_bar_c1": sd["gam1"], "gamma_bar_c2": sd["gam2"],
        "K1_mean_win": sd["K1w"], "gK1_mean_win": sd["gK1w"],
        "H1sum_all": sd["H1all"], "H2sum_all": sd["H2all"],
        **{f"H1sum_{k}": v for k, v in B["H1sum"].items()},
        **{f"H2sum_{k}": v for k, v in B["H2sum"].items()},
        "zbar1_direct": A["zbar1"], "zbar2_direct": A["zbar2"], "zbar3_direct": A["zbar3"],
        "zbar1_init": A0["zbar1"], "zbar2_init": A0["zbar2"], "zbar3_init": A0["zbar3"],
        # zbar2_k(t) - zbar2_k(0) = <dW2[k], mu2> + db2_k + <W2^0[k], mu2 - mu2(0)>  (exact)
        "dz2_W": dW2 @ mu2, "dz2_b": b2 - b20, "dz2_up": W20.reshape(CH, -1) @ (mu2 - A0["mu2"]),
        "dz3_W": dW3 @ mu3, "dz3_b": b3 - b30, "dz3_up": W30 @ (mu3 - A0["mu3"]),
        # zbar1_j(t) - zbar1_j(0) = <dW1[j], M> + db1_j  (M is fixed: the input does not move)
        "dz1_W": (W1 - W10).reshape(CH, -1) @ M, "dz1_b": b1 - b10,
        "a_dc": W1.reshape(CH, -1) @ Mh, "a_dc_init": W10.reshape(CH, -1) @ Mh,
        "mu2": mu2, "mu3": mu3, "M": M,
    }
    out.update({k: v.detach().cpu().numpy() for k, v in vals.items()})
    out.update({"mu2_norm": float(mu2.norm()), "mu3_norm": float(mu3.norm()), "M_norm": float(M.norm()),
                "K1_mean_all": float(M @ M) + 1.0})
    cs = V.channel_stats(S)
    out["zbar"] = torch.cat([cs["zbar_c1"], cs["zbar_c2"]]).cpu().numpy()
    out["zsd"] = torch.cat([cs["zsd_c1"], cs["zsd_c2"]]).cpu().numpy()
    out["fit_conf"] = float(cs["p"].max(1).values.mean())
    if self_form:
        ss = V.self_shape(S, U)
        out["S0"], out["S1"] = ss["S0"], ss["S1"]
    # ---- checks (relative errors; G > 0 sinking throughout)
    def rel(a, b):
        a = torch.as_tensor(a, dtype=DT); b = torch.as_tensor(b, dtype=DT)
        return float(((a - b).abs() / b.abs().clamp_min(1e-300)).max())
    g1 = {k: G[f"G_c1_{k}"].cpu() for k in VARIANTS}
    g2 = {k: G[f"G_c2_{k}"].cpu() for k in VARIANTS}
    gag = G_ag.cpu()
    chk = {"H1_full_vs_autograd": B["errH1"], "H2_full_vs_autograd": B["errH2"],
           "C1_vs_autograd": rel(g1["full"], gag[:CH]),
           "C1_eq1_vs_split": rel(sd["G_eq1"].cpu(), g1["full"]),
           "C3_vs_autograd": rel(g2["full"], gag[CH:])}
    for tag, g in (("C2", g1), ("C2_c2split", g2)):
        ssum = g["init"] + g["drift"] + g["rest"]
        scale = g["init"].abs() + g["drift"].abs() + g["rest"].abs()
        chk[f"{tag}_vs_terms"] = float(((ssum - g["full"]).abs() / scale.clamp_min(1e-300)).max())
        chk[f"{tag}_vs_G"] = rel(ssum, g["full"])
    if saved is not None:
        chk["C1_vs_saved"] = rel(g1["full"], saved["G_full"][:CH])
        chk["C3_vs_saved"] = rel(g2["full"], saved["G_full"][CH:])
        chk["zbar_vs_saved"] = float(np.abs(out["zbar"] - saved["zbar"]).max() / np.abs(saved["zbar"]).max())
    if saved_extra is not None and self_form:
        chk["S0_vs_saved_extra"] = rel(out["S0"], saved_extra["S0"])
        chk["S1_vs_saved_extra"] = rel(out["S1"], saved_extra["S1"])
    # C4: zbar2 = <W2[k], mu2> + b2 exactly; the drift alone carries <dW2[k], mu2>; the same for fc1
    zb2 = W2.reshape(CH, -1) @ mu2 + b2
    chk["C4_zbar2_identity"] = float((zb2 - A["zbar2"]).abs().max() / A["zbar2"].abs().max())
    chk["C4_zbar2_vs_channel_stats"] = float((zb2 - cs["zbar_c2"]).abs().max() / cs["zbar_c2"].abs().max())
    zb20 = W20.reshape(CH, -1) @ A0["mu2"] + b20
    chk["C4_zbar2_identity_init"] = float((zb20 - A0["zbar2"]).abs().max() / A0["zbar2"].abs().max())
    dmu = W2var["drift"].reshape(CH, -1) @ mu2
    chk["C4_drift_carries_mu2_path"] = float((dmu - dW2 @ mu2).abs().max() / (dW2 @ mu2).abs().max())
    chk["C4_rest_off_mu2"] = float((W2var["rest"].reshape(CH, -1) @ mu2).abs().max() / (dW2 @ mu2).abs().max())
    zb3 = W3 @ mu3 + b3
    chk["C4_zbar3_identity"] = float((zb3 - A["zbar3"]).abs().max() / A["zbar3"].abs().max())
    chk["C4_W3_drift_carries_mu3_path"] = float((drift3 @ mu3 - dW3 @ mu3).abs().max() / (dW3 @ mu3).abs().max())
    zb1 = W1.reshape(CH, -1) @ M + b1
    chk["C4_zbar1_identity"] = float((zb1 - A["zbar1"]).abs().max() / A["zbar1"].abs().max())
    # u_j's W1 part is the mean patch M for every c1 channel (and its b1 part is 1)
    chk["U_c1_rows_equal_M"] = float((u1W.reshape(CH, -1) - M[None]).abs().max() / M.abs().max())
    chk["U_c1_b_equal_1"] = float((u1b - 1).abs().max())
    if mix:
        for tag, g, key in (("c1", G, "G_c1_"), ("c2", G, "G_c2_")):
            lin = MIX[0] * g[key + "init"] + MIX[1] * g[key + "drift"] + MIX[2] * g[key + "rest"]
            scale = (MIX[0] * g[key + "init"]).abs() + (MIX[1] * g[key + "drift"]).abs() + (MIX[2] * g[key + "rest"]).abs()
            chk[f"C5_linearity_{tag}"] = float(((g[key + "mix"] - lin).abs() / scale.clamp_min(1e-300)).max())
    # ==== post hoc quantities A-F (coordinator, 1010, after the registered LR numbers) ====
    # Every registered key above is unchanged.  New keys: G_c1_{DC,AC,init_DC,init_AC,drift_DC,
    # drift_AC}, G_c2_{DC,AC,init_DC,init_AC}, G_c2_own_* / G_c2_up_* (every c2 variant), the
    # H1sum_* / H2sum_* of the new variants, the arrays in `ph` below, posthoc_version, chk_PH_*.
    n_img = S.X.shape[0]
    At = sd["Atil"]                                                     # A-tilde (k, j)
    ds_kj = (W2 - W20).sum((2, 3)); s0_kj = W20.sum((2, 3))
    c_k, c_u = -proj2, -proj3
    mu2n = mu2.norm()
    # P-bar used in C: the mean over the 25 taps of mu2's j-block (zero padding included), so that
    # G_c1_drift_j = -(Pbar1_blk_j / |mu2|) Z_j - E_ac_j holds exactly with |mu2| the 400-vector norm
    Pbar1_blk = mu2.view(CH, KER * KER).mean(1)
    Z = (KER * KER) * (c_k[:, None] * At).sum(0)                       # Z_j = 25 sum_k c_k A_kj
    lead = Pbar1_blk / mu2n * Z
    E_ac = -G["G_c1_drift"] - lead
    trans_DC = (ds_kj * At).sum(0)                                      # sum_k ds_kj A_kj = G_c1_DC
    trans_init = (s0_kj * At).sum(0)                                    # sum_k s0_kj A_kj = G_c1_init_DC
    cD2 = c_k * sd["D2"]; cD3 = c_u * sd["D3"]

    def regress(HP, PP, HH, Hs, Ps, n):
        """H on P over all (image, position): no-intercept beta, with-intercept beta, R2 (with
        intercept = corr^2), uncentred R2 of the no-intercept fit."""
        cov = HP - Hs * Ps / n; vp = PP - Ps * Ps / n; vh = HH - Hs * Hs / n
        return HP / PP, cov / vp, cov * cov / (vp * vh), HP * HP / (PP * HH)
    b1n, b1i, r1i, r1n = regress(sd["HP1"], sd["PP1"], sd["HH1"], sd["Hs1"], sd["Ps1"], n_img * 256)
    b2n, b2i, r2i, r2n = regress(sd["HP2"], sd["PP2"], sd["HH2"], sd["Hs2"], sd["Ps2"], n_img * 64)
    G1sa = b1n * sd["RP1"]                                # beta_j sum_{n,q} gamma K1 P1 (self-aligned part)
    G2own_sa = b2n * sd["R2P2own"]
    nanv = torch.full((CH,), float("nan"), dtype=DT, device=dev)
    if self_form:
        G2sa = b2n * torch.as_tensor(out["S0"][CH:], dtype=DT, device=dev)   # full tangent of P2_k (self_shape)
    else:
        G2sa = nanv
    ph = {"Atil": At, "Pbar1_blk": Pbar1_blk, "Z_c1": Z, "lead_c1": lead, "E_ac_c1": E_ac,
          "trans_DC_c1": trans_DC, "trans_init_DC_c1": trans_init,
          "cD2_sum": cD2.sum(), "cD2_abs": cD2.abs().sum(), "cD3_sum": cD3.sum(), "cD3_abs": cD3.abs().sum(),
          "beta1_noint": b1n, "beta1_int": b1i, "R2_c1_int": r1i, "R2_c1_noint": r1n,
          "G_c1_selfal": G1sa, "G_c1_perp": G["G_c1_full"] - G1sa, "S0own_c1": sd["RP1"], "RH1_c1": sd["RH1"],
          "beta2_noint": b2n, "beta2_int": b2i, "R2_c2_int": r2i, "R2_c2_noint": r2n,
          "G_c2_selfal": G2sa, "G_c2_perp": G["G_c2_full"] - G2sa,
          "G_c2_own_selfal": G2own_sa, "G_c2_own_perp": G["G_c2_own_full"] - G2own_sa,
          "S0own_c2": sd["R2P2own"], "R2H2own_c2": sd["R2H2own"],
          "ds3_uk": dW3.view(CN.HIDDEN, CH, 64).sum(2), "s03_uk": W30.view(CN.HIDDEN, CH, 64).sum(2)}
    out.update({k: v.detach().cpu().numpy() for k, v in ph.items()})
    for k in G:                                          # the post hoc variants' G (also in `vals`)
        out[k] = G[k].detach().cpu().numpy()
    out["posthoc_version"] = 1

    def rsc(a, b, scale):
        return float(((a - b).abs() / scale.clamp_min(1e-300)).max())
    g = {k: v for k, v in G.items()}
    chk["PH_A_DC_plus_AC_eq_dW2"] = float(((W2var["DC"] + W2var["AC"]) - dW2t).abs().max() / dW2t.abs().max())
    chk["PH_A_G_DC_plus_AC"] = rsc(g["G_c1_DC"] + g["G_c1_AC"], g["G_c1_drift"] + g["G_c1_rest"],
                                   g["G_c1_DC"].abs() + g["G_c1_AC"].abs())
    chk["PH_A_G_initDC_plus_initAC"] = rsc(g["G_c1_init_DC"] + g["G_c1_init_AC"], g["G_c1_init"],
                                           g["G_c1_init_DC"].abs() + g["G_c1_init_AC"].abs())
    chk["PH_A_G_driftDC_plus_driftAC"] = rsc(g["G_c1_drift_DC"] + g["G_c1_drift_AC"], g["G_c1_drift"],
                                             g["G_c1_drift_DC"].abs() + g["G_c1_drift_AC"].abs())
    chk["PH_A_transmission_identity"] = rsc(g["G_c1_DC"], trans_DC, (ds_kj * At).abs().sum(0))
    chk["PH_A_init_transmission_identity"] = rsc(g["G_c1_init_DC"], trans_init, (s0_kj * At).abs().sum(0))
    chk["PH_C_driftDC_identity"] = rsc(g["G_c1_drift_DC"], -lead,
                                       (KER * KER) * (Pbar1_blk / mu2n).abs() * (c_k[:, None] * At).abs().sum(0))
    chk["PH_C_Eac_identity"] = rsc(E_ac, -g["G_c1_drift_AC"], g["G_c1_drift_DC"].abs() + g["G_c1_drift_AC"].abs())
    chk["PH_B_DC_plus_AC_eq_dW3"] = float(((W3var["DC"] + W3var["AC"]) - dW3).abs().max() / dW3.abs().max())
    chk["PH_B_G_DC_plus_AC"] = rsc(g["G_c2_DC"] + g["G_c2_AC"], g["G_c2_drift"] + g["G_c2_rest"],
                                   g["G_c2_DC"].abs() + g["G_c2_AC"].abs())
    chk["PH_B_G_initDC_plus_initAC"] = rsc(g["G_c2_init_DC"] + g["G_c2_init_AC"], g["G_c2_init"],
                                           g["G_c2_init_DC"].abs() + g["G_c2_init_AC"].abs())
    chk["PH_E_RH1_eq_G_c1"] = rsc(sd["RH1"], g["G_c1_full"], g["G_c1_full"].abs())
    chk["PH_E_R2H2own_eq_G_c2_own"] = rsc(sd["R2H2own"], g["G_c2_own_full"], g["G_c2_own_full"].abs())
    chk["PH_F_own_plus_up_eq_G_c2"] = rsc(g["G_c2_own_full"] + g["G_c2_up_full"], g["G_c2_full"],
                                          g["G_c2_own_full"].abs() + g["G_c2_up_full"].abs())
    if self_form:
        chk["PH_E_S0own_c1_eq_self_shape"] = rsc(sd["RP1"], torch.as_tensor(out["S0"][:CH], dtype=DT, device=dev),
                                                 sd["RP1"].abs())
    # u_k's own part: W2 block of row k = mu2 (other rows 0), b2 part = e_k
    U2W = U[CH:, OFF[2]:OFF[3]].reshape(CH, CH, -1)
    own_blk = U2W[ar, ar]
    off_blk = U2W.clone(); off_blk[ar, ar] = 0
    chk["PH_U_c2_own_rows_equal_mu2"] = float((own_blk - mu2[None]).abs().max() / mu2.abs().max())
    chk["PH_U_c2_other_rows_zero"] = float(off_blk.abs().max() / mu2.abs().max())
    chk["PH_U_c2_b_equal_eye"] = float((U[CH:, OFF[3]:OFF[4]] - torch.eye(CH, dtype=DT, device=dev)).abs().max())
    for k, v in chk.items():
        out["chk_" + k] = v
    out["seconds"] = time.time() - t0
    return out


# --------------------------------------------------------------------------
# checks (C1-C5 on named states)
# --------------------------------------------------------------------------

THRESH = {"H1_full_vs_autograd": 1e-12, "H2_full_vs_autograd": 1e-12,
          "C1_vs_saved": 1e-8, "C1_vs_autograd": 1e-8, "C1_eq1_vs_split": 1e-8,
          "C2_vs_terms": 1e-10, "C2_c2split_vs_terms": 1e-10,
          "C3_vs_saved": 1e-8, "C3_vs_autograd": 1e-8,
          "C4_zbar2_identity": 1e-10, "C4_zbar2_vs_channel_stats": 1e-10, "C4_zbar2_identity_init": 1e-10,
          "C4_drift_carries_mu2_path": 1e-10, "C4_rest_off_mu2": 1e-10,
          "C4_zbar3_identity": 1e-10, "C4_W3_drift_carries_mu3_path": 1e-10, "C4_zbar1_identity": 1e-10,
          "U_c1_rows_equal_M": 1e-12, "U_c1_b_equal_1": 1e-12,
          "C5_linearity_c1": 1e-10, "C5_linearity_c2": 1e-10,
          "C5_formula_vs_split": 1e-10, "C5_formula_vs_saved": 1e-8, "C5_M_direct_vs_U": 1e-12,
          "C5_fd_fixed_routing": 1e-4,
          # post hoc identities (A-F)
          "PH_A_DC_plus_AC_eq_dW2": 1e-12, "PH_A_G_DC_plus_AC": 1e-10, "PH_A_G_initDC_plus_initAC": 1e-10,
          "PH_A_G_driftDC_plus_driftAC": 1e-10, "PH_A_transmission_identity": 1e-10,
          "PH_A_init_transmission_identity": 1e-10, "PH_C_driftDC_identity": 1e-10, "PH_C_Eac_identity": 1e-10,
          "PH_B_DC_plus_AC_eq_dW3": 1e-12, "PH_B_G_DC_plus_AC": 1e-10, "PH_B_G_initDC_plus_initAC": 1e-10,
          "PH_E_RH1_eq_G_c1": 1e-10, "PH_E_R2H2own_eq_G_c2_own": 1e-10, "PH_F_own_plus_up_eq_G_c2": 1e-10,
          "PH_E_S0own_c1_eq_self_shape": 1e-10, "PH_U_c2_own_rows_equal_mu2": 1e-12,
          "PH_U_c2_other_rows_zero": 1e-12, "PH_U_c2_b_equal_eye": 1e-12}
REPORTED_ONLY = {"C2_vs_G", "C2_c2split_vs_G", "zbar_vs_saved", "S0_vs_saved_extra", "S1_vs_saved_extra"}


def run_checks(states, out: Path, device, chunk=200):
    """C1-C4 (the per-state checks of measure_state, plus the H1 / H2 rebuild) and C5 on each state:
    G_j of c1 by direct autograd of L_u w.r.t. (W1, b1) and the formula <M, grad_W1[j]> + grad_b1[j]
    with M the mean patch from unfold (zero padding included); M equals u_j's W1 part; central
    differences of L_u along u (pool winners and gates of the state held fixed); linearity in W2 / W3."""
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    for arm, seed, task in states:
        t0 = time.time()
        S = load_state(arm, seed, task, device)
        S0 = load_state(arm, seed, 0, device)
        U = V.mean_grads(S)
        r = measure_state(S, S0, U=U, chunk=chunk, saved=saved_ref("measure", arm, seed, task),
                          saved_extra=saved_ref("extra", arm, seed, task), mix=True)
        rep = {"arm": arm, "seed": seed, "task": task, "loss": r["loss"], "checks": {}}
        for k, v in r.items():
            if k.startswith("chk_"):
                rep["checks"][k[4:]] = v
        # C5 (a): direct autograd of L_u w.r.t. (W1, b1), contracted with the unfold mean patch
        gs, Lu = V.uniform_grad(S.P, S.alpha, S.X, [0, 1])
        Md = means_pass(S.P, S.alpha, S.X, chunk)["M"]
        Gf = (gs[0].reshape(CH, -1) @ Md + gs[1]).cpu().numpy()
        ar = torch.arange(CH, device=device)
        u1W = U[:CH, OFF[0]:OFF[1]].reshape(CH, CH, -1)[ar, ar]
        rep["checks"]["C5_M_direct_vs_U"] = float((u1W - Md[None]).abs().max() / Md.abs().max())
        rep["checks"]["C5_formula_vs_split"] = float(np.max(np.abs(Gf - r["G_c1_full"]) / np.abs(r["G_c1_full"])))
        sv = saved_ref("measure", arm, seed, task)
        if sv is not None:
            rep["checks"]["C5_formula_vs_saved"] = float(np.max(np.abs(Gf - sv["G_full"][:CH]) / np.abs(sv["G_full"][:CH])))
        rep["L_u_autograd"] = Lu
        rep["L_u_split"] = r["L"]
        # C5 (b): central differences of L_u along u_j with the state's pool winners / gates fixed
        Gall = np.concatenate([r["G_c1_full"], r["G_c2_full"]])
        fd_rows = []
        worst = 0.0
        for row in (0, 5, CH + 0, CH + 5):
            u = U[row]
            fds = []
            for hu in (1e-3, 1e-4, 1e-5):
                h = hu / float(u.norm())
                Pp = [p.clone() for p in S.P]; Pm = [p.clone() for p in S.P]
                for i in range(4):
                    d = u[OFF[i]:OFF[i + 1]].view_as(Pp[i]) * h
                    Pp[i] += d; Pm[i] -= d
                fds.append((V.uniform_value_routed(Pp, S.alpha, S.X, S.P)
                            - V.uniform_value_routed(Pm, S.alpha, S.X, S.P)) / (2 * h))
            e = min(abs(fd - Gall[row]) / abs(Gall[row]) for fd in fds)
            worst = max(worst, e)
            fd_rows.append({"row": row, "G_split": float(Gall[row]), "fd": fds, "rel": e})
        rep["checks"]["C5_fd_fixed_routing"] = worst
        rep["C5_fd_rows"] = fd_rows
        # state_from_dict reproduces verify's State (P, alpha) for this checkpoint
        if arm not in CENTRED:
            Sv = V.State(arm, seed, task, device)
            same = all(bool(torch.equal(a, b)) for a, b in zip(Sv.P, S.P)) and all(
                bool(torch.equal(a, b)) or bool(torch.isnan(a).all() and torch.isnan(b).all())
                for a, b in zip(Sv.alpha, S.alpha))
            rep["aux_state_from_dict_equals_verify_State"] = same
        rep["pass"] = {k: (v < THRESH[k]) for k, v in rep["checks"].items() if k in THRESH}
        rep["all_pass"] = all(rep["pass"].values()) and rep.get("aux_state_from_dict_equals_verify_State", True)
        rep["reported_only"] = {k: v for k, v in rep["checks"].items() if k in REPORTED_ONLY}
        rep["seconds"] = time.time() - t0
        rep["cuda_max_mem_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20
        fn = out / f"checks_{arm}_s{seed}_t{task:02d}.json"
        fn.write_text(json.dumps(rep, indent=1, default=float))
        for k, v in rep["checks"].items():
            flag = ("PASS" if rep["pass"][k] else "FAIL") if k in rep["pass"] else "info"
            print(f"[{flag}] {arm} s{seed} t{task} {k} = {v:.3e}", flush=True)
        print(f"{arm} s{seed} t{task}: all_pass={rep['all_pass']} ({rep['seconds']:.0f}s, "
              f"max mem {rep['cuda_max_mem_mb']:.0f} MB)", flush=True)
        del S, S0, U
        torch.cuda.empty_cache()


def collect(out: Path):
    """checks.json: C1-C4 maxima over every measured state, plus the C5 runs."""
    rows = {}
    nstates = 0
    for fn in sorted(out.glob("*_s*_t*.npz")):
        if ".tmp" in fn.name:
            continue
        d = np.load(fn, allow_pickle=True)
        nstates += 1
        for k in d.files:
            if k.startswith("chk_"):
                v = float(d[k])
                cur = rows.get(k[4:])
                if cur is None or v > cur["max"]:
                    rows[k[4:]] = {"max": v, "worst_state": fn.stem, "n": (cur["n"] if cur else 0) + 1}
                else:
                    cur["n"] += 1
    per_state = {}
    for k, r in rows.items():
        r["threshold"] = THRESH.get(k)
        r["pass"] = (r["max"] < THRESH[k]) if k in THRESH else None
        per_state[k] = r
    c5 = {}
    for fn in sorted(out.glob("checks_*.json")):
        c5[fn.stem] = json.loads(fn.read_text())
    rep = {"n_states": nstates, "per_state_checks_max": per_state,
           "check_runs": {k: {"all_pass": v["all_pass"], "checks": v["checks"], "pass": v["pass"],
                              "C5_fd_rows": v.get("C5_fd_rows"),
                              "aux_state_from_dict_equals_verify_State": v.get("aux_state_from_dict_equals_verify_State")}
                          for k, v in c5.items()},
           "legend": {"C1": "G_c1 (split path, W2v = W2) vs saved G_full[:16] and vs fresh autograd U @ grad L_u",
                      "C2": "G0 + Gdrift + Grest = G (c1, W2 split); relative to |G0|+|Gd|+|Gr|",
                      "C2_c2split": "the same for c2 (W3 split)",
                      "C3": "G_c2 (split path, W3v = W3) vs saved G_full[16:] and vs fresh autograd",
                      "C4": "zbar2_k = <W2[k], mu2> + b2_k exactly; <drift[k], mu2> = <dW2[k], mu2>; rest _|_ mu2; "
                            "same for zbar3 / W3 and zbar1 = <W1[j], M> + b1_j",
                      "C5": "direct autograd of L_u w.r.t. (W1, b1) contracted with the unfold mean patch M; "
                            "M = u_j's W1 part; central differences (fixed winners / gates); linearity",
                      "H1/H2": "conv_transpose(delta2, W2) = dL/dP1 and (W3^T delta3) = dL/dP2 (autograd)"}}
    ok = [r["pass"] for r in per_state.values() if r["pass"] is not None]
    ok += [v["all_pass"] for v in c5.values()]
    rep["all_pass"] = bool(all(ok)) if ok else False
    (out / "checks.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps({k: (v["max"], v["pass"]) for k, v in per_state.items()}, indent=1, default=float))
    print("states", nstates, "check runs", list(c5), "all_pass", rep["all_pass"], flush=True)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["checks", "measure", "collect"])
    ap.add_argument("--states", default="LR:10:10", help="checks: arm:seed:task,...")
    ap.add_argument("--arms", default="LR")
    ap.add_argument("--seeds", default="10-19")
    ap.add_argument("--tasks", default="1,2,5,10,20,30")
    ap.add_argument("--chunk", type=int, default=200)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mem-gb", type=float, default=3.0,
                    help="cap of this process's CUDA caching allocator (shared GPU)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.mode == "collect":
        collect(out)
        return
    device = V.H.setup(args.device)                       # deterministic algorithms, as verify
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if device.type == "cuda":
        di = torch.cuda.current_device()
        total = torch.cuda.get_device_properties(di).total_memory / 2 ** 30
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.mem_gb / total), di)
    if args.mode == "checks":
        st = [(a, int(s), int(t)) for a, s, t in (x.split(":") for x in args.states.split(","))]
        V.set_activation([a for a, _, _ in st])
        run_checks(st, out, device, chunk=args.chunk)
        return
    arms = args.arms.split(",")
    V.set_activation(arms)
    seeds = V.parse_list(args.seeds)
    tasks = V.parse_list(args.tasks)
    n_oom = 0
    t_all = time.time()
    for task in tasks:
        for arm in arms:
            for seed in seeds:
                fn = out / f"{arm}_s{seed}_t{task:02d}.npz"
                if fn.exists():
                    try:
                        with np.load(fn) as d:
                            done = "posthoc_version" in d.files      # files before the post hoc keys are redone
                    except Exception:
                        done = False
                    if done:
                        continue
                if not ckpt_path(arm, seed, task).exists():
                    print(f"missing {ckpt_path(arm, seed, task)}", flush=True)
                    continue
                chunk = args.chunk
                while True:
                    try:
                        S = load_state(arm, seed, task, device)
                        S0 = load_state(arm, seed, 0, device)
                        r = measure_state(S, S0, chunk=chunk, saved=saved_ref("measure", arm, seed, task),
                                          saved_extra=saved_ref("extra", arm, seed, task))
                        break
                    except torch.cuda.OutOfMemoryError:
                        n_oom += 1
                        S = S0 = None
                        torch.cuda.empty_cache()
                        if n_oom >= 2:
                            print(f"STOP: CUDA out of memory twice (last at {arm} s{seed} t{task}, "
                                  f"chunk {chunk}); the GPU is shared -- rerun when memory is free "
                                  f"(finished states are kept and skipped)", flush=True)
                            sys.exit(3)
                        chunk = max(25, chunk // 2)
                        print(f"OOM at {arm} s{seed} t{task}; retry with chunk {chunk}", flush=True)
                tmp = fn.with_name(fn.stem + ".tmp.npz")
                np.savez(tmp, **{k: np.asarray(v) for k, v in r.items()})
                tmp.replace(fn)
                bad = [k for k, v in r.items() if k.startswith("chk_") and k[4:] in THRESH and not v < THRESH[k[4:]]]
                print(f"{arm} s{seed} t{task:02d}: {r['seconds']:.1f}s  "
                      f"c1 G>0 {np.mean(r['G_c1_full'] > 0):.2f}  c2 G>0 {np.mean(r['G_c2_full'] > 0):.2f}  "
                      f"C1 {r.get('chk_C1_vs_saved', float('nan')):.1e}  C2 {r['chk_C2_vs_terms']:.1e}  "
                      f"C3 {r.get('chk_C3_vs_saved', float('nan')):.1e}"
                      f"{'  CHECK FAIL ' + ','.join(bad) if bad else ''}", flush=True)
                del S, S0, r
                torch.cuda.empty_cache()
    print(f"done ({time.time() - t_all:.0f}s, max mem {torch.cuda.max_memory_allocated() / 2 ** 20:.0f} MB)",
          flush=True)


if __name__ == "__main__":
    main()
