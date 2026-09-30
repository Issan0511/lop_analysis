#!/usr/bin/env python3
"""sink_roots_0930 -- RL-MNIST / PM-1200 engine for studies R3, R5, R9, R11a
(spec: specs/spec_sink_roots_0930.md).

Box (1 hidden layer, the default) = pplus_sign_1layer_0925 / drive_recon_0930 exactly:
784-100-K, raw MNIST/255, a fixed subset of N = 1,200 images (hash "rl_subset"), K = 10 random
labels redrawn every task (stream "env_labels_0913"), batch 16 (stream "env_batch_0913"),
init U(+-1/sqrt(fan_in)) (stream "init"), Adam 1e-3 (0.9/0.999, 1e-8) carried across tasks,
float32 on CPU, one thread.  ELU = where(z > 0, z, expm1(min(z, 0))) with autograd, so the
training derivative is expm1(z) + 1 and is exactly 0 below -16.6355.

`--layers 2` gives the 2-hidden-layer box of drive_recon_0930/align_probe3 (784-100-100-K, same
streams, same activation in both layers).  `--env pm` replaces the random labels by the true
labels and draws a new pixel permutation every task (stream "env_perm_0913"; the label stream is
still drawn, so an RL run and a PM run of the same seed share every other draw).

Adam is written out here (the arithmetic of torch.optim.Adam's single-tensor path, which is the
CPU default) so that the v-arms can reset / restore the second moment and its bias-correction
clock without touching the first moment; check `--selftest adam` compares it with torch.

Everything that is not the plain arm is opt-in; with no option the run is the parent's run bit for
bit (check B1 in the spec compares the base arm with drive_recon_0930/runs_ref).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/home/issan/Projects/claude/proj_004_drift/data/mnist")
EXPERIMENT = "sink_roots_0930"
N, BATCH, H = 1200, 16, 100
SQRT2 = math.sqrt(2.0)
INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


# ----------------------------------------------------------------------------- streams, data
def stream(role: str, seed: int) -> torch.Generator:
    h = hashlib.sha256(f"pmnist_0905|{role}|{seed}".encode()).digest()
    return torch.Generator().manual_seed(int.from_bytes(h[:8], "little") & ((1 << 63) - 1))


def read_idx(p: Path) -> np.ndarray:
    b = gzip.open(p, "rb").read()
    dims = int(b[3])
    shape = [int.from_bytes(b[4 + 4 * i:8 + 4 * i], "big") for i in range(dims)]
    return np.frombuffer(b, dtype=np.uint8, offset=4 + 4 * dims).reshape(shape)


def initial(gen_seed_role: str, seed: int, dims) -> list[torch.Tensor]:
    g = stream(gen_seed_role, seed)
    p = []
    for din, dout in zip(dims[:-1], dims[1:]):
        p += [(torch.rand(dout, din, generator=g) * 2 - 1) * (1.0 / math.sqrt(din)),
              (torch.rand(dout, generator=g) * 2 - 1) * (1.0 / math.sqrt(din))]
    return p


# ----------------------------------------------------------------------------- activations
def activ(z, act):
    if act == "LR":
        return torch.where(z > 0, z, z * 0.1)
    if act == "ELU":
        return torch.where(z > 0, z, torch.expm1(z.clamp_max(0)))
    if act == "R":
        return torch.relu(z)
    if act == "GELU":
        return z * 0.5 * (1.0 + torch.erf(z / SQRT2))
    if act == "SILU":
        return z * torch.sigmoid(z)
    if act == "ELUT":                 # round 4 G3 / RR3(a): ELU above -3, a slope-0.1 tail below (continuous at -3)
        return torch.where(z > 0, z, torch.where(z > -3.0, torch.expm1(z.clamp(-3.0, 0.0)),
                                                 (math.exp(-3.0) - 1.0) + 0.1 * (z + 3.0)))
    if act == "LRF":                  # round 4 RR3(b): leaky 0.1 above -3, the constant -0.3 below
        return torch.where(z > 0, z, torch.where(z > -3.0, 0.1 * z, torch.full_like(z, -0.3)))
    raise ValueError(act)


def dphi(z, act):
    """The training derivative (what autograd of `activ` gives, up to the last bit for GELU/SiLU)."""
    if act == "LR":
        return torch.where(z > 0, torch.ones_like(z), torch.full_like(z, 0.1))
    if act == "ELU":
        return torch.where(z > 0, torch.ones_like(z), torch.expm1(z.clamp_max(0)) + 1.0)
    if act == "R":
        return (z > 0).to(z.dtype)
    if act == "GELU":
        return 0.5 * (1.0 + torch.erf(z / SQRT2)) + z * torch.exp(-0.5 * z * z) * INV_SQRT_2PI
    if act == "SILU":
        s = torch.sigmoid(z)
        return s * (1.0 + z * (1.0 - s))
    if act == "ELUT":
        return torch.where(z > 0, torch.ones_like(z), torch.where(z > -3.0, torch.expm1(z.clamp(-3.0, 0.0)) + 1.0,
                                                                  torch.full_like(z, 0.1)))
    if act == "LRF":
        return torch.where(z > 0, torch.ones_like(z), torch.where(z > -3.0, torch.full_like(z, 0.1), torch.zeros_like(z)))
    raise ValueError(act)


def valley_bottom(act: str) -> float:
    """argmin of phi (phi' = 0) for GELU / SiLU, by Newton in float64."""
    z = torch.tensor(-1.0, dtype=torch.float64)
    for _ in range(60):
        z = z.detach().requires_grad_(True)
        d = dphi(z, act)
        (dd,) = torch.autograd.grad(d, z)
        z = z - d / dd
    return float(z)


ZC = {a: valley_bottom(a) for a in ("GELU", "SILU")}
PHI_MIN = {a: float(activ(torch.tensor(ZC[a], dtype=torch.float64), a)) for a in ZC}
FLOOR_OUT = {"ELU": -1.0, "R": 0.0, **PHI_MIN}           # lower bound of phi (none for LR)


class ModAct(torch.autograd.Function):
    """Forward = activ; backward = a modified derivative (used only inside the R3 windows)."""

    @staticmethod
    def forward(ctx, z, act, mode):
        ctx.save_for_backward(z)
        ctx.act, ctx.mode = act, mode
        return activ(z, act)

    @staticmethod
    def backward(ctx, g):
        (z,) = ctx.saved_tensors
        d = dphi(z, ctx.act)
        if ctx.mode == "floor":
            d = torch.where(z > 0, d, torch.clamp_min(d, 0.1))
        elif ctx.mode == "abs":
            d = d.abs()
        elif ctx.mode == "relu":                   # round 2 R3prime: max(phi', 0) -- the valley's negative sign removed, its smallness kept
            d = torch.clamp_min(d, 0.0)
        else:
            raise ValueError(ctx.mode)
        return g * d, None, None


# ----------------------------------------------------------------------------- Adam
class Adam:
    """torch.optim.Adam's single-tensor arithmetic, with separate bias-correction clocks for the
    first (tm) and second (tv) moments so that v can be reset or restored alone."""

    def __init__(self, params, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.P, self.lr, self.b1, self.b2, self.eps = params, lr, b1, b2, eps
        self.m = [torch.zeros_like(p) for p in params]
        self.v = [torch.zeros_like(p) for p in params]
        self.tm = [0] * len(params)
        self.tv = [0] * len(params)

    @torch.no_grad()
    def step(self, grads):
        for i, (p, g) in enumerate(zip(self.P, grads)):
            self.tm[i] += 1
            self.tv[i] += 1
            m, v = self.m[i], self.v[i]
            m.lerp_(g, 1 - self.b1)
            v.mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
            bc1 = 1 - self.b1 ** float(self.tm[i])
            bc2 = 1 - self.b2 ** float(self.tv[i])
            step_size = self.lr / bc1
            denom = (v.sqrt() / (bc2 ** 0.5)).add_(self.eps)
            p.addcdiv_(m, denom, value=-step_size)

    @torch.no_grad()
    def direction(self, i):
        """u = m_hat / (sqrt(v_hat) + eps): the step per unit lr (the update is -lr * u)."""
        if self.tm[i] == 0 or self.tv[i] == 0:
            return torch.full_like(self.P[i], float("nan"))
        mh = self.m[i] / (1 - self.b1 ** float(self.tm[i]))
        vh = self.v[i] / (1 - self.b2 ** float(self.tv[i]))
        return mh / (vh.sqrt() + self.eps)

    def state(self):
        return {"m": [q.clone() for q in self.m], "v": [q.clone() for q in self.v],
                "tm": list(self.tm), "tv": list(self.tv)}

    def load(self, st):
        for q, s in zip(self.m, st["m"]):
            q.copy_(s)
        for q, s in zip(self.v, st["v"]):
            q.copy_(s)
        self.tm, self.tv = list(st["tm"]), list(st["tv"])


class SGD:
    def __init__(self, params, lr):
        self.P, self.lr = params, lr
        self.m = self.v = None

    @torch.no_grad()
    def step(self, grads):
        for p, g in zip(self.P, grads):
            p.add_(g, alpha=-self.lr)

    def direction(self, i):
        return torch.full_like(self.P[i], float("nan"))

    def state(self):
        return {}

    def load(self, st):
        pass


# ----------------------------------------------------------------------------- model
def forward(P, x, act, n_layers, mod=None):
    zs, as_ = [], []
    h = x
    for l in range(n_layers):
        z = h @ P[2 * l].T + P[2 * l + 1]
        a = ModAct.apply(z, act, mod) if mod is not None else activ(z, act)
        zs.append(z)
        as_.append(a)
        h = a
    logits = h @ P[2 * n_layers].T + P[2 * n_layers + 1]
    return zs, as_, logits


@torch.no_grad()
def wcap_frac(W, wcap, mode, mhat):
    """round 4: share of units whose capped quantity sits at its cap (row norm, |w . mu_hat| or the perpendicular norm)."""
    if mode == "row":
        q = W.norm(dim=1)
    else:
        par = W @ mhat
        q = par.abs() if mode == "par" else (W - par[:, None] * mhat[None, :]).norm(dim=1)
    return float((q >= wcap * (1.0 - 1e-5)).double().mean())


@torch.no_grad()
def stop_value(P, X64, act, n_layers, y, kind):
    """round 3 R3p: the task's own fit on all images, float64 as in probe(): 'pold' = mean p of the current label
    (p_old at the next switch), 'lsd' = mean over images of the class-centred logit RMS (probe's logit_sd)."""
    h = X64
    for l in range(n_layers):
        h = activ(h @ P[2 * l].double().T + P[2 * l + 1].double(), act)
    f = h @ P[2 * n_layers].double().T + P[2 * n_layers + 1].double()
    if kind == "pold":
        return float(f.softmax(-1)[torch.arange(len(y)), y].mean())
    return float((f - f.mean(-1, keepdim=True)).pow(2).mean(-1).sqrt().mean())


def effective_logits(logits, cap):
    return logits if cap is None else cap * torch.tanh(logits / cap)


def loss_of(logits, yb, cap, ls, K, sq=0.0):
    f = effective_logits(logits, cap)
    if ls:
        yb = (1.0 - ls) * yb + ls / K
    loss = (f.logsumexp(-1) - (f * yb).sum(-1)).mean()
    if sq:
        fc = f - f.mean(-1, keepdim=True)
        loss = loss + sq * (fc * fc).sum(-1).mean()
    return loss


def probe_grid(T: int) -> list[int]:
    g = list(range(0, min(500, T) + 1, 25))
    g += list(range(750, T + 1, 250))
    if g[-1] != T:
        g.append(T)
    return sorted(set(g))


# ----------------------------------------------------------------------------- probes
@torch.no_grad()
def probe(P, X64, act, n_layers, K, y_new, y_old, cap, ls, opt, mu_in, prev_z, sq=0.0):
    """Full-data diagnostics in float64.  Returns (per-unit arrays, scalars, z list)."""
    P64 = [q.double() for q in P]
    zs, as_, logits = [], [], None
    h = X64
    for l in range(n_layers):
        z = h @ P64[2 * l].T + P64[2 * l + 1]
        a = activ(z, act)
        zs.append(z)
        as_.append(a)
        h = a
    raw = h @ P64[2 * n_layers].T + P64[2 * n_layers + 1]
    f = effective_logits(raw, cap)
    p = f.softmax(-1)
    unit, sc = {}, {}
    for l, z in enumerate(zs):
        unit[f"m{l+1}"] = z.mean(0)
        unit[f"k{l+1}"] = (z > 0).sum(0).double()
        unit[f"top{l+1}"] = z.max(0).values
        unit[f"s{l+1}"] = z.std(0)
    sc["cov1_zero"] = float(((zs[-1] > 0).sum(1) == 0).double().mean())   # inputs no unit opens (last layer)
    ar = torch.arange(len(y_new))
    Y = torch.nn.functional.one_hot(y_new, K).double()
    Yt = (1.0 - ls) * Y + ls / K if ls else Y
    sc["ce"] = float((f.logsumexp(-1) - (f * Yt).sum(-1)).mean())
    sc["logit_sd"] = float((f - f.mean(-1, keepdim=True)).pow(2).mean(-1).sqrt().mean())
    sc["acc"] = float((f.argmax(-1) == y_new).double().mean())
    sc["l1"] = float((p - Y).abs().sum(-1).mean())
    sc["one_minus_pnew"] = float((1 - p[ar, y_new]).mean())
    sc["pmax"] = float(p.max(-1).values.mean())
    fy = f[ar, y_new]
    other = f.clone()
    other[ar, y_new] = -float("inf")
    sc["margin_new_med"] = float((fy - other.max(-1).values).median())
    if y_old is not None:
        sc["p_old"] = float(p[ar, y_old].mean())
        sc["acc_old"] = float((f.argmax(-1) == y_old).double().mean())
        fo = f[ar, y_old]
        other = f.clone()
        other[ar, y_old] = -float("inf")
        sc["margin_old_med"] = float((fo - other.max(-1).values).median())
        diff = y_old != y_new
        sc["p_rest"] = float((1 - p[ar, y_new] - torch.where(diff, p[ar, y_old], torch.zeros_like(p[:, 0]))).mean())
    # Adam direction statistics (per unit), layer l weights W (H, din), bias b (H), readout columns
    mus = [mu_in] + [a.mean(0) for a in as_[:-1]]
    for l in range(n_layers):
        uW = opt.direction(2 * l).double()
        ub = opt.direction(2 * l + 1).double()
        mu = mus[l]
        unit[f"u_mean{l+1}"] = uW.mean(1)
        unit[f"u_abs{l+1}"] = uW.abs().mean(1)
        unit[f"u_mu{l+1}"] = uW @ mu + ub
        unit[f"u_coh{l+1}"] = (torch.sign(uW) * mu[None, :]).sum(1) / mu.abs().sum()
    uR = opt.direction(2 * n_layers).double()             # (K, H_last)
    unit["ro_abs"] = uR.abs().mean(0)
    unit["ro_common"] = uR.sum(0).abs() / uR.abs().sum(0)
    # demand on each (input, unit) of each hidden layer, split into cancel / label parts
    Wout = P64[2 * n_layers]
    if cap is None:
        Wc = Wout - Wout.mean(0, keepdim=True)
        canc = p @ Wc
        lab = -(Yt @ Wc)
    else:
        sech2 = 1.0 - torch.tanh(raw / cap) ** 2
        canc = (p * sech2) @ Wout
        lab = -((Yt * sech2) @ Wout)
    dem = [None] * n_layers
    dem_c = [None] * n_layers
    dem_l = [None] * n_layers
    if sq:                                   # logit squeeze: its demand is put in the label part
        fc = f - f.mean(-1, keepdim=True)
        lab = lab + 2.0 * sq * (fc @ Wout)
    dem[-1], dem_c[-1], dem_l[-1] = canc + lab, canc, lab
    for l in range(n_layers - 1, 0, -1):
        g = dphi(zs[l], act)
        W = P64[2 * l]
        dem_c[l - 1] = (dem_c[l] * g) @ W
        dem_l[l - 1] = (dem_l[l] * g) @ W
        dem[l - 1] = dem_c[l - 1] + dem_l[l - 1]
    # expected-push fields of the last hidden layer (SGD form, per input averaged): K_n phi'(z_in) x ...
    zl = zs[-1]
    hin = X64 if n_layers == 1 else as_[-2]
    Kn = hin @ mus[n_layers - 1] + 1.0
    gl = dphi(zl, act) * Kn[:, None]
    Wc_last = Wout - Wout.mean(0, keepdim=True)                  # (K, H): column i = v_i^c
    unit["S_cancel"] = (gl * (p @ Wc_last)).mean(0)                 # E-push field for iid uniform labels
    unit["A_new"] = (gl * Wc_last.T[:, y_new].T).mean(0)            # alignment with the current labels
    if y_old is not None:
        unit["A_old"] = (gl * Wc_last.T[:, y_old].T).mean(0)        # alignment with the previous labels
    if n_layers == 2:
        a1 = as_[0]
        kern = a1 @ a1.T + 1.0
        sc["pair_kernel_min"] = float(kern.min())
        sc["pair_kernel_neg"] = float((kern < 0).double().mean())
        sc["K2_min"] = float(Kn.min())
    if act in ZC:
        zc = ZC[act]
        for l, z in enumerate(zs):
            below = z < zc
            band = (z >= zc) & (z < 0)
            nb = int(below.sum())
            sc[f"v{l+1}_n_below"] = nb
            sc[f"v{l+1}_n_band"] = int(band.sum())
            if nb:
                sc[f"v{l+1}_below_u_neg"] = float((dem[l][below] < 0).double().mean())
                sc[f"v{l+1}_below_u_mean"] = float(dem[l][below].mean())
                sc[f"v{l+1}_below_canc_neg"] = float((dem_c[l][below] < 0).double().mean())
                sc[f"v{l+1}_below_lab_neg"] = float((dem_l[l][below] < 0).double().mean())
                # dL/dz summed in the SGD sense: -phi' * u; positive = z moves up under its own demand
                sc[f"v{l+1}_below_selfup"] = float(((-dphi(z, act) * dem[l])[below] > 0).double().mean())
            if int(band.sum()):
                sc[f"v{l+1}_band_u_neg"] = float((dem[l][band] < 0).double().mean())
            if prev_z is not None:
                pb = prev_z[l] < zc
                npb = int(pb.sum())
                sc[f"v{l+1}_prev_below"] = npb
                if npb:
                    dz = (z - prev_z[l])[pb]
                    sc[f"v{l+1}_moved_up"] = float((dz > 0).double().mean())
                    sc[f"v{l+1}_dz_mean"] = float(dz.mean())
                    sc[f"v{l+1}_crossed_up"] = float((z[pb] > zc).double().mean())
    # demand sign on closed pairs for every activation (z <= 0) and open pairs, last layer
    zl = zs[-1]
    closed, opened = zl <= 0, zl > 0
    if int(closed.sum()):
        sc["closed_u_neg"] = float((dem[-1][closed] < 0).double().mean())
    if int(opened.sum()):
        sc["open_u_neg"] = float((dem[-1][opened] < 0).double().mean())
    return {k: v.float().numpy() for k, v in unit.items()}, sc, zs, (zs[-1] > 0).sum(1).numpy().astype(np.int16), (f.argmax(-1) == y_new).numpy()


# ----------------------------------------------------------------------------- the run
def git_state():
    try:
        h = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        d = subprocess.check_output(["git", "status", "--porcelain", "--", "src", "analysis", "specs"],
                                    cwd=ROOT, text=True).strip()
        return {"git_hash": h, "dirty": bool(d)}
    except Exception as e:      # pragma: no cover
        return {"git_hash": None, "error": str(e)}


def load_mnist(seed: int):
    ax = read_idx(DATA / "train-images-idx3-ubyte.gz").reshape(-1, 784).astype(np.float32) / 255
    ay = read_idx(DATA / "train-labels-idx1-ubyte.gz").astype(np.int64)
    subset = torch.randperm(len(ax), generator=stream("rl_subset", seed))[:N].numpy()
    return torch.tensor(ax[subset]), torch.tensor(ay[subset]), subset


def run(cfg: dict, out: Path, fork: dict | None = None) -> dict:
    """One seed.  cfg keys: act, seed, layers, K, env, tasks, T, lr, opt, b2, tau, ls, cap,
    vreset, vrestore, bwmode, bw_from, probes ('full' | 'ends'), snap_before, ckpt_after, fork."""
    t_start = time.time()
    act, seed, L, K = cfg["act"], cfg["seed"], cfg["layers"], cfg["K"]
    T = cfg["T"]
    X, ytrue, subset = load_mnist(seed)
    ink_mean = None
    if cfg.get("ink"):                          # round 1 R6: every image scaled to the subset's mean ink (sum of pixels)
        ink = X.sum(1, keepdim=True)
        ink_mean = float(ink.mean())
        X = X * (ink.mean() / ink)
    X64 = X.double()
    dims = (784,) + (H,) * L + (K,)
    glabel, gbatch = stream("env_labels_0913", seed), stream("env_batch_0913", seed)
    gperm, gtau = stream("env_perm_0913", seed), stream("tau_keep_0930", seed)
    P = [q.requires_grad_(True) for q in initial("init", seed, dims)]
    opt = Adam(P, cfg["lr"], 0.9, cfg["b2"], 1e-8) if cfg["opt"] == "adam" else SGD(P, cfg["lr"])
    task0, y_cur = 1, None
    wcap = None
    perm = torch.arange(784)
    if fork is not None:                                   # R5: continue from a checkpoint
        ck = torch.load(fork["ckpt"], weights_only=False)
        with torch.no_grad():
            for q, s in zip(P, ck["P"]):
                q.copy_(s)
        opt.load(ck["opt"])
        for g, s in ((glabel, "glabel"), (gbatch, "gbatch"), (gperm, "gperm"), (gtau, "gtau")):
            g.set_state(ck[s])
        task0, y_cur, perm = ck["task"] + 1, ck["y_cur"], ck["perm"]
        apply_fork(fork["mode"], P, opt, X64 if not cfg["env"] == "pm" else X64[:, perm], act, L, seed, dims)
    grid = probe_grid(T) if cfg["probes"] == "full" else [0, 200, T]
    grid = [s for s in grid if s <= T]
    mu_in = None
    rows, unit_store, task_sc = [], {}, []
    cover, correct = [], []
    snaps = set(cfg.get("snap_before", []))
    ckpts = set(cfg.get("ckpt_after", []))
    out.mkdir(parents=True, exist_ok=True)
    last_task = task0 + cfg["tasks"] - 1
    y_first, r_ref, rescale_lambda, rr1_m5, rr1_s5 = None, None, None, None, None      # round 4
    mhat_run = X64.mean(0).float(); mhat_run = mhat_run / mhat_run.norm()
    gnoise = stream("noise_0930", seed) if cfg.get("noise_sigma") else None
    lit = (X.sum(0) > 0).float()                                   # pixels lit in some image of the subset
    for task in range(task0, last_task + 1):
        y_draw = torch.randint(K, (N,), generator=glabel)
        steps = T
        epochs = -(-(steps * BATCH) // N)
        order = torch.stack([torch.randperm(N, generator=gbatch) for _ in range(epochs)]).reshape(-1)[
            :steps * BATCH].reshape(steps, BATCH)
        if cfg["env"] == "pm":
            perm = torch.randperm(784, generator=gperm)
            y_new = ytrue.clone()
        else:
            if cfg["tau"] > 0 and y_cur is not None:
                keep = torch.rand(N, generator=gtau) < cfg["tau"]
                y_new = torch.where(keep, y_cur, y_draw)
            else:
                y_new = y_draw
            if cfg.get("same_labels"):                       # round 4 G2: every task keeps the first task's labels
                if y_first is None:
                    y_first = y_new.clone()
                y_new = y_first
        Xt = X[:, perm] if cfg["env"] == "pm" else X
        Xt64 = Xt.double()
        mu_in = Xt64.mean(0)
        Y = torch.nn.functional.one_hot(y_new, K).to(torch.float32)
        y_old = y_cur
        if task in snaps:
            np.savez_compressed(out / f"state_before_t{task:03d}.npz",
                                **{f"P{i}": q.detach().numpy() for i, q in enumerate(P)},
                                **({f"m{i}": q.numpy() for i, q in enumerate(opt.m)} if opt.m else {}),
                                **({f"v{i}": q.numpy() for i, q in enumerate(opt.v)} if opt.v else {}),
                                tm=np.array(getattr(opt, "tm", [])), tv=np.array(getattr(opt, "tv", [])),
                                y_old=(y_old.numpy() if y_old is not None else np.array([])),
                                y_new=y_new.numpy(), subset=subset, perm=perm.numpy(), task=task,
                                seed=seed, act=act, layers=L, K=K)
        # ---- switch-time interventions (task >= 2 of this run's own sequence)
        if cfg["vreset"] and y_old is not None and cfg["opt"] == "adam":
            with torch.no_grad():
                for i in range(len(P)):
                    opt.v[i].zero_()
                    opt.tv[i] = 0
        if cfg["adamreset"] and y_old is not None and cfg["opt"] == "adam":
            with torch.no_grad():
                for i in range(len(P)):
                    opt.m[i].zero_(); opt.v[i].zero_()
                    opt.tm[i] = 0; opt.tv[i] = 0
        v_sw = [q.clone() for q in opt.v] if (cfg["vrestore"] and y_old is not None) else None
        if cfg.get("rescale_at"):                             # round 4 G1(c): (w_i, b_i) *= r_i(ref) / r_i(now) at rescale_at
            if task == cfg["rescale_ref"]:
                r_ref = P[0].detach().norm(dim=1).clone()
            if task == cfg["rescale_at"]:
                with torch.no_grad():
                    lam = r_ref / P[0].norm(dim=1)
                    P[0].mul_(lam[:, None])
                    P[1].mul_(lam)
                rescale_lambda = lam.numpy().copy()
        if cfg.get("rr1") and task >= cfg["rr1_from"]:        # round 4 RR1: move only the bias at each switch
            with torch.no_grad():
                z = Xt64 @ P[0].double().T + P[1].double()
                m_i, s_i = z.mean(0), z.std(0)
                if task == cfg["rr1_from"]:
                    rr1_m5, rr1_s5 = m_i.clone(), s_i.clone()
                else:
                    target = rr1_m5 if cfg["rr1"] == "A" else rr1_m5 / rr1_s5 * s_i
                    P[1].add_((target - m_i).float())
        if cfg.get("wcap_from") and task == cfg["wcap_from"]:     # round 1 R1: cap W1 row norms at this task's start
            W0 = P[0].detach()
            mhat = X64.mean(0).float(); mhat = mhat / mhat.norm()
            par0 = W0 @ mhat
            if cfg.get("wcap_mode", "row") == "row":
                wcap = W0.norm(dim=1).clone() * cfg.get("wcap_scale", 1.0)
            elif cfg["wcap_mode"] == "par":                        # R1b-3a: cap only |w . mu_hat|
                wcap = par0.abs().clone() * cfg.get("wcap_scale", 1.0)
            elif cfg["wcap_mode"] == "perp":                       # R1b-3b: cap only |w - (w . mu_hat) mu_hat|
                wcap = (W0 - par0[:, None] * mhat[None, :]).norm(dim=1).clone() * cfg.get("wcap_scale", 1.0)
            else:
                raise ValueError(cfg["wcap_mode"])
        prev_z = None
        U = {}
        S_list = []
        online = 0
        diverged = False
        gi = 0
        cov_sw = None
        n_upd = steps
        for s in range(steps + 1):
            if (cfg.get("stop") and s >= cfg["stop_min"] and s % cfg["stop_every"] == 0 and s < steps
                    and stop_value(P, Xt64, act, L, y_new, cfg["stop"]) >= cfg["stop_at"]):
                # round 3 R3p: the task ends here; the final probe goes to the last grid slot, the slots between are NaN
                u, sc, zs, cov, corr = probe(P, Xt64, act, L, K, y_new, y_old, cfg["cap"], cfg["ls"], opt,
                                            mu_in, prev_z, cfg["sq"])
                n_left = len(grid) - gi
                for k2, v2 in u.items():
                    U.setdefault(k2, []).extend([np.full_like(v2, np.nan)] * (n_left - 1) + [v2])
                S_list.extend([{k2: float("nan") for k2 in sc}] * (n_left - 1) + [sc])
                if cov_sw is None:
                    cov_sw = cov
                cover.append(np.stack([cov_sw, cov]))
                correct.append(corr)
                n_upd = s
                break
            if gi < len(grid) and grid[gi] == s:
                u, sc, zs, cov, corr = probe(P, Xt64, act, L, K, y_new, y_old, cfg["cap"], cfg["ls"], opt,
                                            mu_in, prev_z, cfg["sq"])
                prev_z = zs
                for k2, v2 in u.items():
                    U.setdefault(k2, []).append(v2)
                if wcap is not None:
                    sc["wcap_at"] = wcap_frac(P[0].detach(), wcap, cfg.get("wcap_mode", "row"), mhat_run)
                S_list.append(sc)
                if s == 0:
                    cov_sw = cov
                if s == steps:
                    cover.append(np.stack([cov_sw, cov]))
                    correct.append(corr)
                gi += 1
            if s == steps:
                break
            idx = order[s]
            xb, yb = Xt[idx], Y[idx]
            mod = None
            if cfg["bwmode"] and y_old is not None and s >= cfg["bw_from"]:
                mod = cfg["bwmode"]
            _, _, logits = forward(P, xb, act, L, mod)
            if cfg["cap"] is None and not cfg["ls"] and not cfg["sq"]:
                loss = (logits.logsumexp(-1) - (logits * yb).sum(-1)).mean()
            else:
                loss = loss_of(logits, yb, cfg["cap"], cfg["ls"], K, cfg["sq"])
            online += int((logits.detach().argmax(-1) == y_new[idx]).sum())
            grads = torch.autograd.grad(loss, P)
            opt.step(grads)
            if cfg.get("ro_clamp") and task >= cfg["ro_clamp_from"]:
                with torch.no_grad():                      # round 2 R3_readout_scale_clamp: class-centred readout norm per unit -> r
                    Wo = P[2 * L]
                    mean_k = Wo.mean(0, keepdim=True)
                    Vc = Wo - mean_k
                    Wo.copy_(mean_k + Vc * (cfg["ro_clamp"] / Vc.norm(dim=0, keepdim=True).clamp_min(1e-30)))
            if wcap is not None:
                with torch.no_grad():
                    mode = cfg.get("wcap_mode", "row")
                    if mode == "row":
                        nr = P[0].norm(dim=1)
                        P[0].mul_(torch.clamp(wcap / nr, max=1.0)[:, None])
                    else:
                        par = P[0] @ mhat
                        perp = P[0] - par[:, None] * mhat[None, :]
                        if mode == "par":
                            newpar = torch.sign(par) * torch.minimum(par.abs(), wcap)
                            P[0].copy_(perp + newpar[:, None] * mhat[None, :])
                        else:
                            npp = perp.norm(dim=1)
                            P[0].copy_(par[:, None] * mhat[None, :] + perp * torch.clamp(wcap / npp, max=1.0)[:, None])
            if gnoise is not None and task >= cfg["noise_from_task"] and s + 1 > cfg["noise_from_step"]:
                with torch.no_grad():                         # round 4 RR2: isotropic noise on the lit pixels' weights
                    P[0].add_(torch.randn(P[0].shape, generator=gnoise) * cfg["noise_sigma"] * lit[None, :])
            if v_sw is not None and s + 1 == cfg["vrestore_at"]:
                with torch.no_grad():
                    for q, v0 in zip(opt.v, v_sw):
                        q.copy_(v0)
            if not torch.isfinite(loss):
                diverged = True
                break
        if diverged:
            rows.append({"task": task, "diverged": True, "step": s})
            break
        for k2, v2 in U.items():
            unit_store.setdefault(k2, []).append(np.stack(v2))
        keys = sorted(set().union(*[set(d) for d in S_list]))
        S = {k2: [d.get(k2, float("nan")) for d in S_list] for k2 in keys}
        task_sc.append(S)
        row = {"task": task, "online_acc": online / (n_upd * BATCH), "acc_end": S["acc"][-1], "steps_used": n_upd,
               "ce_end": S["ce"][-1], "pplus_end": float(np.mean(U[f"k{L}"][-1]) / N),
               "allclosed_end": float(np.mean(U[f"k{L}"][-1] == 0))}
        if y_old is not None:
            row.update({"p_old_sw": S["p_old"][0], "margin_old_sw": S["margin_old_med"][0],
                        "acc_old_sw": S["acc_old"][0]})
        rows.append(row)
        y_cur = y_new
        if task in ckpts:
            torch.save({"P": [q.detach().clone() for q in P], "opt": opt.state(), "task": task,
                        "y_cur": y_cur.clone(), "perm": perm.clone(),
                        "glabel": glabel.get_state(), "gbatch": gbatch.get_state(),
                        "gperm": gperm.get_state(), "gtau": gtau.get_state(), "cfg": cfg},
                       out / f"ckpt_after_t{task:03d}.pt")
        if cfg.get("verbose", True):
            print(f"{cfg['name']} task {task} acc {row['acc_end']:.3f} online {row['online_acc']:.3f} "
                  f"p+ {row['pplus_end']:.4f} ({time.time() - t_start:.0f}s)", flush=True)
    arrays = {f"u_{k}": np.stack(v) for k, v in unit_store.items()}
    allk = sorted(set().union(*[set(d) for d in task_sc])) if task_sc else []
    arrays.update({f"s_{k}": np.array([d.get(k, [float("nan")] * len(grid)) for d in task_sc], dtype=np.float64)
                   for k in allk})
    arrays["grid"] = np.array(grid)
    if rescale_lambda is not None:
        arrays["rescale_lambda"] = rescale_lambda
    arrays["cover"] = np.stack(cover) if cover else np.zeros(0)
    arrays["correct_end"] = np.stack(correct) if correct else np.zeros(0)
    arrays["tasks"] = np.arange(task0, task0 + len(unit_store.get(f"m{L}", [])))
    np.savez_compressed(out / "arrays.npz", **arrays)
    (out / "rows.json").write_text(json.dumps(rows))
    prov = {"experiment": EXPERIMENT, **git_state(), "cfg": {k: v for k, v in cfg.items()},
            "fork": fork, "torch": torch.__version__, "threads": torch.get_num_threads(),
            "wall_s": time.time() - t_start, "zc": ZC, "phi_min": PHI_MIN, "ink_mean": ink_mean}
    (out / "provenance.json").write_text(json.dumps(prov, indent=1, default=str))
    return {"rows": rows}


def apply_fork(mode, P, opt, X64, act, L, seed, dims):
    """R5 fork modifications, applied to the state at the end of the checkpoint task."""
    if mode in ("nat", "long"):
        return
    fresh = initial("fork_init_0930", seed, dims)
    with torch.no_grad():
        if mode == "adam0":
            for i in range(len(P)):
                opt.m[i].zero_(); opt.v[i].zero_(); opt.tm[i] = 0; opt.tv[i] = 0
        elif mode in ("ro_reinit", "l1_reinit", "fresh"):
            idx = {"ro_reinit": [2 * L, 2 * L + 1], "l1_reinit": [0, 1],
                   "fresh": list(range(len(P)))}[mode]
            for i in idx:
                P[i].copy_(fresh[i])
                opt.m[i].zero_(); opt.v[i].zero_(); opt.tm[i] = 0; opt.tv[i] = 0
        elif mode == "recenter":
            z = X64 @ P[0].double().T + P[1].double()
            P[1].sub_(z.median(0).values.float())
        else:
            raise ValueError(mode)


# ----------------------------------------------------------------------------- self-tests
def selftest_adam():
    """Custom Adam vs torch.optim.Adam on the base box, 300 updates: bit for bit."""
    X, _, _ = load_mnist(0)
    for act in ("ELU", "GELU"):
        res = []
        for use_torch in (True, False):
            P = [q.requires_grad_(True) for q in initial("init", 0, (784, H, 10))]
            opt = torch.optim.Adam(P, lr=1e-3, betas=(0.9, 0.999), eps=1e-8) if use_torch else Adam(P, 1e-3)
            g = torch.Generator().manual_seed(1)
            y = torch.randint(10, (N,), generator=g)
            Y = torch.nn.functional.one_hot(y, 10).float()
            for s in range(300):
                idx = torch.randint(N, (BATCH,), generator=g)
                z = X[idx] @ P[0].T + P[1]
                logits = activ(z, act) @ P[2].T + P[3]
                loss = (logits.logsumexp(-1) - (logits * Y[idx]).sum(-1)).mean()
                if use_torch:
                    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                else:
                    opt.step(torch.autograd.grad(loss, P))
            res.append([q.detach().clone() for q in P])
        same = all(torch.equal(a, b) for a, b in zip(*res))
        print(f"selftest adam {act}: bit-identical = {same}")
        if not same:
            raise SystemExit("custom Adam differs from torch.optim.Adam")
    # mutation control: a different eps must break the identity
    P1 = [q.requires_grad_(True) for q in initial("init", 0, (784, H, 10))]
    P2 = [q.requires_grad_(True) for q in initial("init", 0, (784, H, 10))]
    o1, o2 = torch.optim.Adam(P1, lr=1e-3), Adam(P2, 1e-3, eps=1e-7)
    z = X[:16] @ P1[0].T + P1[1]
    l1 = activ(z, "ELU") @ P1[2].T + P1[3]
    l1 = l1.logsumexp(-1).mean(); o1.zero_grad(); l1.backward(); o1.step()
    z = X[:16] @ P2[0].T + P2[1]
    l2 = activ(z, "ELU") @ P2[2].T + P2[3]
    o2.step(torch.autograd.grad(l2.logsumexp(-1).mean(), P2))
    assert not all(torch.equal(a, b) for a, b in zip(P1, P2)), "mutation control failed"
    print("selftest adam: mutation control OK")


def selftest_modact():
    z = torch.linspace(-20, 5, 2001, dtype=torch.float32, requires_grad=True)
    for act in ("ELU", "GELU", "SILU", "LR"):
        (ga,) = torch.autograd.grad(activ(z, act).sum(), z)
        d = dphi(z.detach(), act)
        err = float((ga - d).abs().max())
        print(f"selftest dphi {act}: max |autograd - dphi| = {err:.2e}")
        assert err < 1e-6
        (gf,) = torch.autograd.grad(ModAct.apply(z, act, "floor").sum(), z)
        want = torch.where(z.detach() > 0, d, d.clamp_min(0.1))
        assert torch.equal(gf, want)
        (gb,) = torch.autograd.grad(ModAct.apply(z, act, "abs").sum(), z)
        assert torch.equal(gb, d.abs())
    assert abs(ZC["GELU"] + 0.7517915) < 1e-6 and abs(ZC["SILU"] + 1.2784645) < 1e-6, ZC
    print("selftest modact OK; ZC", ZC, "PHI_MIN", PHI_MIN)


# ----------------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", choices=["adam", "modact"])
    ap.add_argument("--name", default="run")
    ap.add_argument("--act", default="ELU")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--env", default="rl", choices=["rl", "pm"])
    ap.add_argument("--tasks", type=int, default=30)
    ap.add_argument("--T", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--opt", default="adam", choices=["adam", "sgd"])
    ap.add_argument("--b2", type=float, default=0.999)
    ap.add_argument("--tau", type=float, default=0.0)
    ap.add_argument("--ls", type=float, default=0.0)
    ap.add_argument("--cap", type=float, default=None)
    ap.add_argument("--vreset", action="store_true", help="v and its clock to 0 at each switch (m kept)")
    ap.add_argument("--adamreset", action="store_true", help="m, v and both clocks to 0 at each switch")
    ap.add_argument("--sq", type=float, default=0.0, help="logit squeeze: + sq * mean sum_c (f_c - fbar)^2")
    ap.add_argument("--vrestore", type=int, default=0, help="restore v at this update of each task")
    ap.add_argument("--bwmode", default=None, choices=[None, "floor", "abs", "relu"])
    ap.add_argument("--bw-from", type=int, default=200)
    ap.add_argument("--probes", default="full", choices=["full", "ends"])
    ap.add_argument("--ink-normalize", action="store_true", help="round 1 R6: scale every image to the mean ink")
    ap.add_argument("--wcap-from", type=int, default=0, help="round 1 R1: cap W1 row norms from this task's start")
    ap.add_argument("--wcap-mode", default="row", choices=["row", "par", "perp"], help="R1b: which W1 component to cap")
    ap.add_argument("--wcap-scale", type=float, default=1.0, help="R1b-4: cap = scale x the value at the start of --wcap-from")
    ap.add_argument("--ro-clamp", type=float, default=0.0, help="round 2: after every update from --ro-clamp-from, set each unit's class-centred readout norm to this value")
    ap.add_argument("--ro-clamp-from", type=int, default=51)
    ap.add_argument("--stop", default=None, choices=[None, "pold", "lsd"], help="round 3 R3p: end each task when the fit reaches --stop-at")
    ap.add_argument("--stop-at", type=float, default=0.0)
    ap.add_argument("--stop-min", type=int, default=500)
    ap.add_argument("--stop-every", type=int, default=25)
    ap.add_argument("--same-labels", action="store_true", help="round 4 G2: every task keeps the first task's labels")
    ap.add_argument("--rescale-at", type=int, default=0, help="round 4 G1(c): scale (w_i, b_i) by r_i(ref)/r_i(now) at this task's start")
    ap.add_argument("--rescale-ref", type=int, default=5)
    ap.add_argument("--rr1", default=None, choices=[None, "A", "B"], help="round 4 RR1: at each switch move b so that m (A) or m/s (B) returns to its value at --rr1-from")
    ap.add_argument("--rr1-from", type=int, default=5)
    ap.add_argument("--noise-sigma", type=float, default=0.0, help="round 4 RR2: N(0, sigma^2) on the lit pixels' W1 after every update past --noise-from-step")
    ap.add_argument("--noise-from-task", type=int, default=5)
    ap.add_argument("--noise-from-step", type=int, default=200)
    ap.add_argument("--snap-before", type=int, nargs="*", default=[])
    ap.add_argument("--ckpt-after", type=int, nargs="*", default=[])
    ap.add_argument("--fork-ckpt", default=None)
    ap.add_argument("--fork-mode", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.selftest == "adam":
        return selftest_adam()
    if a.selftest == "modact":
        return selftest_modact()
    lr = a.lr if a.lr is not None else (1e-3 if a.opt == "adam" else 0.1)
    cfg = {"name": a.name, "act": a.act, "seed": a.seed, "layers": a.layers, "K": a.K, "env": a.env,
           "tasks": a.tasks, "T": a.T, "lr": lr, "opt": a.opt, "b2": a.b2, "tau": a.tau, "ls": a.ls,
           "cap": a.cap, "sq": a.sq, "vreset": a.vreset, "adamreset": a.adamreset, "vrestore": bool(a.vrestore), "vrestore_at": a.vrestore,
           "bwmode": a.bwmode, "bw_from": a.bw_from, "probes": a.probes,
           "snap_before": a.snap_before, "ckpt_after": a.ckpt_after,
           "ink": a.ink_normalize, "wcap_from": a.wcap_from, "wcap_mode": a.wcap_mode, "wcap_scale": a.wcap_scale,
           "ro_clamp": a.ro_clamp, "ro_clamp_from": a.ro_clamp_from,
           "stop": a.stop, "stop_at": a.stop_at, "stop_min": a.stop_min, "stop_every": a.stop_every,
           "same_labels": a.same_labels, "rescale_at": a.rescale_at, "rescale_ref": a.rescale_ref,
           "rr1": a.rr1, "rr1_from": a.rr1_from, "noise_sigma": a.noise_sigma, "noise_from_task": a.noise_from_task,
           "noise_from_step": a.noise_from_step}
    fork = {"ckpt": a.fork_ckpt, "mode": a.fork_mode} if a.fork_ckpt else None
    out = Path(a.out) if a.out else ROOT / "results" / EXPERIMENT / "runs" / a.name
    run(cfg, out, fork)


if __name__ == "__main__":
    main()
