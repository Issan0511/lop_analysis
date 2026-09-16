#!/usr/bin/env python3
"""l2_wall_0916 (specs/spec_l2_wall_0916.md): the second layer's mu wall along trajectories.

Part A re-runs two committed boxes bit for bit and only READS their state between CUDA-graph blocks:
  --part chimera : layer_chimera_rl_0914 (24 models, 50 tasks)
  --part ext150  : relu_gelu_silu_rl_ext150_0914 (30 models, 150 tasks)
Parts B and C run one variant engine in the chimera box (same streams, same loop):
  --part variant : 132 models = {none, inLN, inLN_a, preLN, preLN_a} x {EE, EL, LE, LL} x {RL, PM} x seeds 0-2
                   + Adam eps {1e-6, 1e-30} x {EE, LE} x RL x seeds 0-2  (eps 1e-8 control = the `none` models)
  --selfcheck    : S1-S4 and S7 with mutation controls on short runs (writes checks_selfcheck.json)
All diagnostics are float64 recomputations from the float32 state; they never write to the engine.
Run from the repo root:  python3 -m src.l2_wall_0916 --part chimera
"""
from __future__ import annotations
import argparse, csv, gzip, hashlib, json, math, os, resource, subprocess, sys, time
from pathlib import Path
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from src import layer_chimera_rl_0914 as C
from src import relu_gelu_silu_rl_0914 as B

ROOT = Path(__file__).resolve().parents[1]
NAME = "l2_wall_0916"
OUT = ROOT / f"results/{NAME}"
SPEC = ROOT / f"specs/spec_{NAME}.md"
PREREG_COMMIT = "b24a34c1f4e12d879e8a216fb856a424d04737a6"
EPS64 = float(np.finfo(np.float64).eps)
EPS32 = float(np.finfo(np.float32).eps)
LEAK = float(np.float32(0.1))          # the hosts' leak slope as it is actually applied (float32 0.1)
LN_EPS = 1e-5
SQ2 = math.sqrt(2.0)
ISQ2PI = 1.0 / math.sqrt(2.0 * math.pi)
DENSE = (75, 375, 1500, 3000)
PAIRS = (("ELU1", "ELU1"), ("ELU1", "LR"), ("LR", "ELU1"), ("LR", "LR"))
LN_ARMS = ("none", "inLN", "inLN_a", "preLN", "preLN_a")
EPS_ARMS = (("eps1e-6", 1e-6), ("eps1e-30", 1e-30))
VARIANT_MODELS = (
    [dict(seed=s, env=e, act1=a1, act2=a2, arm=arm, eps=1e-8)
     for s in range(3) for e in ("RL", "PM") for (a1, a2) in PAIRS for arm in LN_ARMS]
    + [dict(seed=s, env="RL", act1=a1, act2=a2, arm=arm, eps=ep)
       for s in range(3) for (a1, a2) in (("ELU1", "ELU1"), ("LR", "ELU1")) for arm, ep in EPS_ARMS])
PARTS = {"chimera": dict(mod=C, tasks=50, dense_upto=20),
         "ext150": dict(mod=B, tasks=150, dense_upto=0),
         "variant": dict(mod=C, tasks=50, dense_upto=20)}
UNIT_SUMMARY = ("d", "rho", "zbar", "sig", "U", "pplus", "K", "hd", "hbar", "hU", "hpplus", "gate_mean", "low",
                "tiny", "q", "vnorm", "m", "wt", "b", "var_q", "var_v", "cov",
                "adam_epsfrac", "adam_step", "adam_unb", "adam_mh", "adam_den", "adam_svmin")
QUANT_FIELDS = ("d", "rho", "U", "hU", "zbar", "hbar", "gate_mean", "adam_epsfrac")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def model_key(part, m):
    if part == "ext150":
        return (m["seed"], m["env"], m["act"])
    if part == "chimera":
        return (m["seed"], m["env"], m["act1"], m["act2"])
    return (m["seed"], m["env"], m["act1"], m["act2"], m["arm"], m["eps"])


# ----------------------------------------------------------------------------------------------
# float32 variant model (LayerNorm arms and per-model Adam eps) with hand-written backprop
# ----------------------------------------------------------------------------------------------

def ln32(t, eps=LN_EPS):
    c = t - t.mean(-1, keepdim=True)
    s = (c.square().mean(-1, keepdim=True) + eps).sqrt()
    return c / s, s


def ln_back(g, n, s, mut=""):
    """d/dt of LN(t) = (t - mean)/sqrt(var + eps), applied to the upstream gradient g."""
    if mut == "ln_no_nterm":
        return (g - g.mean(-1, keepdim=True)) / s
    return (g - g.mean(-1, keepdim=True) - n * (g * n).mean(-1, keepdim=True)) / s


def vforward(p, x, E):
    xn, _ = ln32(x)
    xin = torch.where(E.fin, xn, x)
    z1 = torch.bmm(xin, p[0].transpose(1, 2)) + p[1][:, None, :]
    n1, s1 = ln32(z1)
    h1 = torch.where(E.fpre, p[8][:, None, :] * n1 + p[9][:, None, :], z1)
    a1 = C.activ(h1, E.elu1)
    an, sa = ln32(a1)
    u1 = torch.where(E.fin, p[6][:, None, :] * an + p[7][:, None, :], a1)
    z2 = torch.bmm(u1, p[2].transpose(1, 2)) + p[3][:, None, :]
    n2, s2 = ln32(z2)
    h2 = torch.where(E.fpre, p[10][:, None, :] * n2 + p[11][:, None, :], z2)
    a2 = C.activ(h2, E.elu2)
    z3 = torch.bmm(a2, p[4].transpose(1, 2)) + p[5][:, None, :]
    return dict(xin=xin, z1=z1, n1=n1, s1=s1, h1=h1, a1=a1, an=an, sa=sa, u1=u1,
                z2=z2, n2=n2, s2=s2, h2=h2, a2=a2, z3=z3)


def vgradients(p, x, y, E, mut=""):
    f = vforward(p, x, E)
    g3 = (f["z3"].softmax(-1) - y) / x.shape[1]
    gh2 = torch.bmm(g3, p[4]) * C.gate(f["h2"], E.elu2)
    gam2 = gh2 if mut == "ln_no_gamma" else p[10][:, None, :] * gh2
    pre_m, in_m = (E.fin, E.fpre) if mut == "swap_masks" else (E.fpre, E.fin)
    gz2 = torch.where(pre_m, ln_back(gam2, f["n2"], f["s2"], mut), gh2)
    gu1 = torch.bmm(gz2, p[2])
    gami = gu1 if mut == "ln_no_gamma" else p[6][:, None, :] * gu1
    ga1 = torch.where(in_m, ln_back(gami, f["an"], f["sa"], mut), gu1)
    gh1 = ga1 * C.gate(f["h1"], E.elu1)
    gam1 = gh1 if mut == "ln_no_gamma" else p[8][:, None, :] * gh1
    gz1 = torch.where(pre_m, ln_back(gam1, f["n1"], f["s1"], mut), gh1)
    zero = torch.zeros_like(p[6])
    gr = [torch.bmm(gz1.transpose(1, 2), f["xin"]), gz1.sum(1),
          torch.bmm(gz2.transpose(1, 2), f["u1"]), gz2.sum(1),
          torch.bmm(g3.transpose(1, 2), f["a2"]), g3.sum(1),
          torch.where(E.aff_in, (gu1 * f["an"]).sum(1), zero), torch.where(E.aff_in, gu1.sum(1), zero),
          torch.where(E.aff_pre, (gh1 * f["n1"]).sum(1), zero), torch.where(E.aff_pre, gh1.sum(1), zero),
          torch.where(E.aff_pre, (gh2 * f["n2"]).sum(1), zero), torch.where(E.aff_pre, gh2.sum(1), zero)]
    ce = (f["z3"].logsumexp(-1) - (f["z3"] * y).sum(-1)).mean(-1)
    acc = (f["z3"].argmax(-1) == y.argmax(-1)).float().mean(-1)
    return gr, ce, acc


class Flags:
    """Per-model masks; usable with float32 (engine) and float64 (self-test) tensors."""
    def __init__(self, models, device):
        t3 = lambda f: torch.tensor([bool(f(m)) for m in models], device=device)[:, None, None]
        self.elu1 = t3(lambda m: m["act1"] == "ELU1")
        self.elu2 = t3(lambda m: m["act2"] == "ELU1")
        self.fin = t3(lambda m: m["arm"] in ("inLN", "inLN_a"))
        self.fpre = t3(lambda m: m["arm"] in ("preLN", "preLN_a"))
        self.aff_in = t3(lambda m: m["arm"] == "inLN_a")[:, :, 0]
        self.aff_pre = t3(lambda m: m["arm"] == "preLN_a")[:, :, 0]


class VariantEngine(C.Engine):
    def __init__(self, models, device="cuda"):
        super().__init__(models=models, device=device)
        M = self.M
        F = Flags(models, self.device)
        self.fin, self.fpre, self.aff_in, self.aff_pre = F.fin, F.fpre, F.aff_in, F.aff_pre
        # host masks elu1/elu2 were built by C.Engine from the same act1/act2 fields
        for q in (torch.ones(M, 100), torch.zeros(M, 100), torch.ones(M, 100), torch.zeros(M, 100),
                  torch.ones(M, 100), torch.zeros(M, 100)):
            q = q.to(self.device)
            self.p.append(q); self.m.append(torch.zeros_like(q)); self.v.append(torch.zeros_like(q))
        ev = torch.tensor([float(m["eps"]) for m in models], dtype=torch.float32, device=self.device)
        self.eps_param = [ev.view(-1, *([1] * (q.dim() - 1))) for q in self.p]
        self.eps64 = torch.tensor([float(np.float32(m["eps"])) for m in models], dtype=torch.float64, device=self.device)

    @torch.no_grad()
    def step(self, x, y):
        gr, ce, acc = vgradients(self.p, x, y, self)
        self.ce.add_(ce); self.acc.add_(acc); self.t.add_(1)
        c1 = 1 - torch.pow(.9, self.t); c2 = 1 - torch.pow(.999, self.t)
        for p, g, m, v, ep in zip(self.p, gr, self.m, self.v, self.eps_param):
            m.mul_(.9).add_(g, alpha=.1); v.mul_(.999).addcmul_(g, g, value=.001)
            p.addcdiv_(m / c1, (v / c2).sqrt() + ep, value=-.001)

    @torch.no_grad()
    def evaluate(self):
        f = vforward(self.p, self.cx, self)
        logits = f["z3"]
        ce = (logits.logsumexp(-1) - (logits * self.cy).sum(-1)).mean(-1)
        acc = (logits.argmax(-1) == self.cy.argmax(-1)).float().mean(-1)
        out = dict(train_acc=acc.cpu().numpy(), train_ce=ce.cpu().numpy())
        out["a1_rms"] = f["a1"].square().mean(dim=(1, 2)).sqrt().cpu().numpy()
        units = {}
        for l, z, elu in [(1, f["h1"], self.elu1), (2, f["h2"], self.elu2)]:
            g = C.gate(z, elu)
            for key, val in {"zmean": z.mean(1), "zstd": z.std(1, unbiased=False), "gate_mean": g.mean(1),
                             "gate_rms": g.square().mean(1).sqrt(), "lowgate": (g < .05).float().mean(1),
                             "nearzero": (z.abs() < 1).float().mean(1)}.items():
                units[f"{key}_l{l}"] = val.cpu().numpy()
        units["cnorm_l1"] = (self.p[0] - self.p[0].mean(-1, keepdim=True)).norm(dim=-1).cpu().numpy()
        units["rowmean_l1"] = self.p[0].mean(-1).cpu().numpy()
        for l in range(1, 4):
            out[f"wnorm_l{l}"] = self.p[2 * (l - 1)].norm(dim=-1).mean(-1).cpu().numpy()
        out["cnorm_l1"] = units["cnorm_l1"].mean(-1)
        out["lowgate_l1"] = units["lowgate_l1"].mean(-1)
        out["nearzero_l1"] = units["nearzero_l1"].mean(-1)
        return out, units


# ----------------------------------------------------------------------------------------------
# float64 diagnostics (read-only)
# ----------------------------------------------------------------------------------------------

def act64(z, K):
    base = torch.where(z > 0, z, torch.where(K["elu"], torch.expm1(z.clamp_max(0)),
                       torch.where(K["r"], torch.zeros_like(z), z * LEAK)))
    gelu = z * 0.5 * (1.0 + torch.erf(z / SQ2))
    silu = z * torch.sigmoid(z)
    return torch.where(K["gelu"], gelu, torch.where(K["silu"], silu, base))


def gate64(z, K):
    base = torch.where(z > 0, torch.ones_like(z), torch.where(K["elu"], z.clamp_max(0).exp(),
                       torch.where(K["r"], torch.zeros_like(z), torch.full_like(z, LEAK))))
    gelu = 0.5 * (1.0 + torch.erf(z / SQ2)) + z * torch.exp(-0.5 * z * z) * ISQ2PI
    s = torch.sigmoid(z)
    silu = s * (1.0 + z * (1.0 - s))
    return torch.where(K["gelu"], gelu, torch.where(K["silu"], silu, base))


def ln64(t):
    c = t - t.mean(-1, keepdim=True)
    return c / (c.square().mean(-1, keepdim=True) + LN_EPS).sqrt()


def input_stats(u, pre):
    mu = u.mean(1)
    r = mu.norm(dim=-1)
    e = mu / r[:, None]
    p = torch.bmm(u, e[:, :, None])[..., 0]
    sp = p.std(1, unbiased=False)
    pmin = p.amin(1)
    S = u.sum(-1)
    st = {f"{pre}_r": r, f"{pre}_sp": sp, f"{pre}_pmin": pmin, f"{pre}_pmax": p.amax(1),
          f"{pre}_c": r / sp, f"{pre}_k": (p.mean(1) - pmin) / sp, f"{pre}_pmean": p.mean(1),
          f"{pre}_onesided": (pmin > 0).double(),
          f"{pre}_cos1": mu.sum(-1) / (r * math.sqrt(u.shape[-1])),
          f"{pre}_Smean": S.mean(1), f"{pre}_Ssd": S.std(1, unbiased=False)}
    return mu, r, e, p, st


def unit_stats(z, h, g, q, vnorm, zv, p, W, b, r, umax, din, N, mut=""):
    nanv = float("nan")
    zbar = z.mean(1)
    sig = z.std(1, unbiased=False)
    safe = torch.where(sig > 0, sig, torch.full_like(sig, nanv))
    U = z.amax(1)
    hbar = h.mean(1)
    hsd = h.std(1, unbiased=False)
    hsafe = torch.where(hsd > 0, hsd, torch.full_like(hsd, nanv))
    pc = p - p.mean(1, keepdim=True)
    zvc = zv - zv.mean(1, keepdim=True)
    var_q = q * q * pc.square().mean(1)[:, None]
    var_v = zvc.square().mean(1)
    cov = 2 * q * (pc[..., None] * zvc).mean(1)
    F = dict(zbar=zbar, sig=sig, U=U, pplus=(z > 0).double().mean(1), d=zbar / safe, K=(U - zbar) / safe,
             hbar=hbar, hsd=hsd, hU=h.amax(1), hpplus=(h > 0).double().mean(1), hd=hbar / hsafe,
             gate_mean=g.mean(1), gate_rms=g.square().mean(1).sqrt(), low=(g.abs() < 0.05).double().mean(1),
             tiny=(g.abs() < 1e-8).double().mean(1), neg=(g < 0).double().mean(1),
             q=q, vnorm=vnorm, m=W.mean(-1), wt=(W - W.mean(-1, keepdim=True)).norm(dim=-1), wn=W.norm(dim=-1),
             b=b, var_q=var_q, var_v=var_v, cov=cov, rho=var_q / safe.square())
    # every float64 error below is a sum of at most (din + N + 16) rounding errors, each bounded by
    # eps64 times the absolute scale Wu = sum_j |W_j| * max|u| (+|b|); 4 covers the chained operations
    wabs = W.abs().sum(-1)
    Wu = wabs * umax[:, None] + b.abs()
    F["wu"] = Wu
    tz = 4 * (din + N + 16) * EPS64 * Wu + 1e-300
    tot = var_q + var_v + (0.0 if mut == "cov" else cov)
    R = dict(zbar_id=(zbar - (q * r[:, None] + b)).abs() / tz,
             var_split=(tot - sig.square()).abs() / (4 * (din + N + 16) * EPS64 * Wu.square() + 1e-300))
    return F, R, Wu, wabs, tz


def adam_stats(mW, vW, mb, vb, t, eps):
    mm, U = mW.shape[0], mW.shape[1]
    keys = ("mh", "den", "svmin", "epsfrac", "step", "unb")
    if t <= 0:
        z = torch.full((mm, U), float("nan"), dtype=torch.float64, device=mW.device)
        return {k: z for k in keys}
    c1 = 1.0 - 0.9 ** t
    c2 = 1.0 - 0.999 ** t
    mh = torch.cat([mW.double(), mb.double()[..., None]], -1) / c1
    sv = (torch.cat([vW.double(), vb.double()[..., None]], -1) / c2).sqrt()
    ep = eps[:, None, None]
    den = sv + ep
    unb = torch.where(sv > 0, mh / torch.where(sv > 0, sv, torch.ones_like(sv)), torch.zeros_like(sv))
    rms = lambda a: a.square().mean(-1).sqrt()
    return dict(mh=rms(mh), den=rms(den), svmin=sv.amin(-1), epsfrac=(sv < ep).double().mean(-1),
                step=rms(mh / den), unb=rms(unb))


@torch.no_grad()
def diag_chunk(P, x32, K1, K2, fin, fpre, ain, eps, t, AD, mut=""):
    W1, b1, W2, b2, gi, bi, gp1, bp1, gp2, bp2 = (a.double() for a in P)
    x = x32.double()
    mm, N, _ = x.shape
    fin3, fpre3 = fin[:, None, None], fpre[:, None, None]
    Uf, Mo, R = {}, {}, {}
    # layer 1
    xin = torch.where(fin3, ln64(x), x)
    mu1, r1, e1, p1, st = input_stats(xin, "in1")
    Mo.update(st)
    q1 = torch.bmm(W1, e1[:, :, None])[..., 0]
    v1 = W1 - q1[..., None] * e1[:, None, :]
    z1 = torch.bmm(xin, W1.transpose(1, 2)) + b1[:, None, :]
    zv1 = torch.bmm(xin, v1.transpose(1, 2))
    h1 = torch.where(fpre3, gp1[:, None, :] * ln64(z1) + bp1[:, None, :], z1)
    g1 = gate64(h1, K1)
    a1 = act64(h1, K1)
    umax1 = xin.abs().amax((1, 2))
    F1, R1, Wu1, wabs1, _ = unit_stats(z1, h1, g1, q1, v1.norm(dim=-1), zv1, p1, W1, b1, r1, umax1, 784, N)
    # layer-2 input
    an = ln64(a1)
    u1 = torch.where(fin3, gi[:, None, :] * an + bi[:, None, :], a1)
    u_mu = u1.roll(1, 0) if mut == "roll" else (u1.float().double() if mut == "f32mu" else u1)
    mu2, r2, e2, p2, st = input_stats(u_mu, "in2")
    Mo.update(st)
    e_q = torch.full_like(e2, 1.0 / math.sqrt(e2.shape[-1])) if mut == "axis" else e2
    q2 = torch.bmm(W2, e_q[:, :, None])[..., 0]
    v2 = W2 - q2[..., None] * e_q[:, None, :]
    z2 = torch.bmm(u1, W2.transpose(1, 2)) + b2[:, None, :]
    zv2 = torch.bmm(u1, v2.transpose(1, 2))
    h2 = torch.where(fpre3, gp2[:, None, :] * ln64(z2) + bp2[:, None, :], z2)
    g2 = gate64(h2, K2)
    umax2 = u1.abs().amax((1, 2))
    F2, R2, Wu2, wabs2, tz2 = unit_stats(z2, h2, g2, q2, v2.norm(dim=-1), zv2, p2, W2, b2, r2, umax2, 100, N, mut)
    wb = torch.bmm(W2, bi[:, :, None])[..., 0]
    gam = torch.bmm(W2, (gi * an.mean(1))[:, :, None])[..., 0]
    beta = torch.where(ain[:, None], wb, torch.zeros_like(wb))
    R2["beta_split"] = torch.where(fin[:, None], (F2["zbar"] - (gam + wb + b2)).abs() / tz2, torch.zeros_like(wb))
    F2["beta"] = beta
    uc = u1 - u1.mean(1, keepdim=True)
    S2 = torch.bmm(uc.transpose(1, 2), uc) / N
    ev, evec = torch.linalg.eigh(S2.cpu())
    tr = ev.sum(-1)
    Mo["in2_tr"] = tr
    Mo["in2_lam1"], Mo["in2_lam2"], Mo["in2_lam3"] = ev[:, -1], ev[:, -2], ev[:, -3]
    Mo["in2_cos_top"] = (evec[..., -1] * e2.cpu()).sum(-1).abs()
    Mo["in2_eshare"] = st["in2_sp"].cpu() ** 2 / tr
    pos2 = (h2 > 0).double().mean(-1)
    pos1 = (h1 > 0).double().mean(-1)
    mu_a1 = a1.mean(1)
    Mo.update(h2pos_min=pos2.amin(1), h2pos_mean=pos2.mean(1), h2pos_zero=(pos2 == 0).double().mean(1),
              h1pos_min=pos1.amin(1), h1pos_zero=(pos1 == 0).double().mean(1),
              a1_max=a1.amax((1, 2)), a1_min=a1.amin((1, 2)),
              a1_floor_frac=(mu_a1 < -0.9).double().mean(-1), a1_neg_frac=(mu_a1 < 0).double().mean(-1),
              umax1=umax1, umax2=umax2)
    A1 = adam_stats(AD[0], AD[1], AD[2], AD[3], t, eps)
    A2 = adam_stats(AD[4], AD[5], AD[6], AD[7], t, eps)
    for k, v in F1.items():
        Uf[f"{k}_l1"] = v
    for k, v in F2.items():
        Uf[f"{k}_l2"] = v
    for k, v in A1.items():
        Uf[f"adam_{k}_l1"] = v
    for k, v in A2.items():
        Uf[f"adam_{k}_l2"] = v
    for k, v in R1.items():
        R[f"{k}_l1"] = v
    for k, v in R2.items():
        R[f"{k}_l2"] = v
    S = dict(W1=W1, b1=b1, mu1=mu1, W2=W2, b2=b2, mu2=mu2, e2=e2, r2=r2, S2=S2,
             zbar1=F1["zbar"], zbar2=F2["zbar"], Wu1=Wu1, Wu2=Wu2, wabs2=wabs2, beta=beta,
             d2=F2["d"], rho2=F2["rho"], sig2=F2["sig"])
    return Uf, Mo, R, S


def lnparams(e):
    p = e.p
    if len(p) == 12:
        return [p[0], p[1], p[2], p[3], p[6], p[7], p[8], p[9], p[10], p[11]]
    M = p[0].shape[0]
    one = torch.ones(M, 100, device=p[0].device)
    zero = torch.zeros(M, 100, device=p[0].device)
    return [p[0], p[1], p[2], p[3], one, zero, one, zero, one, zero]


@torch.no_grad()
def diag(e, K1, K2, kind, mut="", chunk=24):
    P = lnparams(e)
    M = P[0].shape[0]
    t = float(e.t.item())
    parts = []
    for lo in range(0, M, chunk):
        sl = slice(lo, min(M, lo + chunk))
        k1 = {k: v[sl] for k, v in K1.items()}
        k2 = {k: v[sl] for k, v in K2.items()}
        AD = [e.m[0][sl], e.v[0][sl], e.m[1][sl], e.v[1][sl], e.m[2][sl], e.v[2][sl], e.m[3][sl], e.v[3][sl]]
        parts.append(diag_chunk([q[sl] for q in P], e.cx[sl], k1, k2, kind["fin"][sl], kind["fpre"][sl],
                                kind["ain"][sl], kind["eps"][sl], t, AD, mut))
    cat = lambda xs: torch.cat([x.to(xs[0].device) for x in xs], 0)
    Uf = {k: cat([p[0][k] for p in parts]) for k in parts[0][0]}
    Mo = {k: cat([p[1][k].to(parts[0][1][k].device) for p in parts]) for k in parts[0][1]}
    R = {k: cat([p[2][k] for p in parts]) for k in parts[0][2]}
    S = {k: cat([p[3][k] for p in parts]) for k in parts[0][3]}
    return Uf, Mo, R, S


def dstat(W, b, mu, S):
    num = torch.bmm(W, mu[:, :, None])[..., 0] + b
    var = (torch.bmm(W, S) * W).sum(-1)
    sig = var.clamp_min(0).sqrt()
    r = mu.norm(dim=-1, keepdim=True)
    e = mu / r
    q = torch.bmm(W, e[:, :, None])[..., 0]
    se = (torch.bmm(S, e[:, :, None])[..., 0] * e).sum(-1, keepdim=True)
    return num / sig, q * q * se / var, sig


@torch.no_grad()
def ledger(o, n, N, mut=""):
    bmv = lambda A, v: torch.bmm(A, v[:, :, None])[..., 0]
    L, R = {}, {}
    dW2 = n["W2"] - o["W2"]
    mu_n = n["mu2"].float().double() if mut == "f32mu" else n["mu2"]
    dmu = mu_n - o["mu2"]
    self_ = bmv(dW2, o["mu2"])
    up = bmv(o["W2"], dmu)
    cross = bmv(dW2, dmu)
    bias = n["b2"] - o["b2"]
    meas = n["zbar2"] - o["zbar2"]
    clos = meas - (self_ + up + (0.0 if mut == "cross" else cross) + bias)
    ebar = 0.5 * (n["e2"] + o["e2"])
    r_hi = o["r2"] if mut == "rold" else n["r2"]
    stretch = (r_hi - o["r2"])[:, None] * bmv(o["W2"], ebar)
    rot = (0.5 * (n["r2"] + o["r2"]))[:, None] * bmv(o["W2"], n["e2"] - o["e2"])
    L.update(self=self_, up=up, cross=cross, bias=bias, meas=meas, stretch=stretch, rot=rot,
             beta=n["beta"] - o["beta"], dw2=dW2.norm(dim=-1))
    tl = 4 * (100 + N + 16) * EPS64 * (o["Wu2"] + n["Wu2"]) + 1e-300
    R["ledger_closure"] = clos.abs() / tl
    tsr = 16 * (100 + 16) * EPS64 * o["wabs2"] * (o["r2"] + n["r2"])[:, None] + 1e-300
    R["stretch_rot"] = (stretch + rot - up).abs() / tsr
    dW1 = n["W1"] - o["W1"]
    dmu1 = n["mu1"] - o["mu1"]
    s1, u1_, x1 = bmv(dW1, o["mu1"]), bmv(o["W1"], dmu1), bmv(dW1, dmu1)
    b1 = n["b1"] - o["b1"]
    m1 = n["zbar1"] - o["zbar1"]
    L.update(self1=s1, up1=u1_, cross1=x1, bias1=b1, meas1=m1)
    tl1 = 4 * (784 + N + 16) * EPS64 * (o["Wu1"] + n["Wu1"]) + 1e-300
    R["ledger1_closure"] = (m1 - (s1 + u1_ + x1 + b1)).abs() / tl1
    WB = [(o["W2"], o["b2"]), (n["W2"], n["b2"])]
    MU = [o["mu2"], n["mu2"]]
    SS = [o["S2"], n["S2"]]
    cache = {}
    for i in (0, 1):
        for j in (0, 1):
            for k in (0, 1):
                kk = 0 if (mut == "Sold" and (i, j, k) == (1, 1, 1)) else k
                cache[i, j, k] = dstat(WB[i][0], WB[i][1], MU[j], SS[kk])
    kq = 8 * (N + 100 * 100 + 16) * EPS64
    worst = {}
    for corner, st in (((0, 0, 0), o), ((1, 1, 1), n)):
        sig = st["sig2"]
        tvar = kq * st["Wu2"].square()                              # var <= (sum|W| max|u|)^2
        tsig = tvar / (2 * sig)
        td = 8 * (100 + N + 16) * EPS64 * st["Wu2"] / sig + st["d2"].abs() * tsig / sig
        trho = 3 * st["rho2"].abs() * tvar / sig.square() + kq * (st["rho2"].abs() + 1)
        for idx, key, tol in ((0, "d2", td), (1, "rho2", trho), (2, "sig2", tsig)):
            rr = (cache[corner][idx] - st[key]).abs() / (tol + 1e-300)
            worst[key] = rr if key not in worst else torch.fmax(worst[key], rr)
    for key, v in worst.items():
        R[f"corner_{key}"] = v
    for idx, key in ((0, "d"), (1, "rho")):
        f = lambda i, j, k: cache[i, j, k][idx]
        L[f"sh_{key}_W"] = ((f(1, 0, 0) - f(0, 0, 0)) / 3 + (f(1, 1, 0) - f(0, 1, 0)) / 6
                            + (f(1, 0, 1) - f(0, 0, 1)) / 6 + (f(1, 1, 1) - f(0, 1, 1)) / 3)
        L[f"sh_{key}_mu"] = ((f(0, 1, 0) - f(0, 0, 0)) / 3 + (f(1, 1, 0) - f(1, 0, 0)) / 6
                             + (f(0, 1, 1) - f(0, 0, 1)) / 6 + (f(1, 1, 1) - f(1, 0, 1)) / 3)
        L[f"sh_{key}_S"] = ((f(0, 0, 1) - f(0, 0, 0)) / 3 + (f(1, 0, 1) - f(1, 0, 0)) / 6
                            + (f(0, 1, 1) - f(0, 1, 0)) / 6 + (f(1, 1, 1) - f(1, 1, 0)) / 3)
        L[f"d_{key}"] = f(1, 1, 1) - f(0, 0, 0)
    g = lambda i, k: cache[i, 0, k][2]
    L["sh_sig_W"] = 0.5 * ((g(1, 0) - g(0, 0)) + (g(1, 1) - g(0, 1)))
    L["sh_sig_S"] = 0.5 * ((g(0, 1) - g(0, 0)) + (g(1, 1) - g(1, 0)))
    L["d_sig"] = g(1, 1) - g(0, 0)
    return L, R


# ----------------------------------------------------------------------------------------------
# engines, committed references, the loop
# ----------------------------------------------------------------------------------------------

def build(part, device="cuda", models=None):
    if part == "chimera":
        models = models or C.MODELS
        e = C.Engine(models=models, device=device)
        F3 = torch.zeros_like(e.elu1)
        K1 = dict(elu=e.elu1, r=F3, gelu=F3, silu=F3)
        K2 = dict(elu=e.elu2, r=F3, gelu=F3, silu=F3)
    elif part == "ext150":
        models = models or B.MODELS
        e = B.Engine(models=models, device=device)
        K1 = K2 = dict(elu=e.A.elu, r=e.A.r, gelu=e.A.gelu, silu=e.A.silu)
    else:
        models = models or VARIANT_MODELS
        e = VariantEngine(models, device=device)
        F3 = torch.zeros_like(e.elu1)
        K1 = dict(elu=e.elu1, r=F3, gelu=F3, silu=F3)
        K2 = dict(elu=e.elu2, r=F3, gelu=F3, silu=F3)
    M = len(models)
    if part == "variant":
        kind = dict(fin=e.fin[:, 0, 0], fpre=e.fpre[:, 0, 0], ain=e.aff_in[:, 0], eps=e.eps64)
    else:
        F1 = torch.zeros(M, dtype=torch.bool, device=e.device)
        kind = dict(fin=F1, fpre=F1, ain=F1,
                    eps=torch.full((M,), float(np.float32(1e-8)), dtype=torch.float64, device=e.device))
    return e, models, K1, K2, kind


def committed(part):
    if part == "chimera":
        d = ROOT / "results/layer_chimera_rl_0914"
        keyf = lambda r: (int(r["seed"]), r["env"], r["act1"], r["act2"], int(r["task"]))
    elif part == "ext150":
        d = ROOT / "results/relu_gelu_silu_rl_ext150_0914"
        keyf = lambda r: (int(r["seed"]), r["env"], r["act"], int(r["task"]))
    else:
        return None
    with open(d / "rows.csv", newline="") as f:
        rows = {keyf(r): r for r in csv.DictReader(f)}
    units = dict(np.load(d / "units.npz"))
    return dict(rows=rows, units=units, sha={"rows.csv": sha(d / "rows.csv"), "units.npz": sha(d / "units.npz")})


def bitcheck(ref, part, models, task, met, units, online_acc, online_ce, floors):
    worst_u, exact_u, n_u = 0.0, True, 0
    for k, v in units.items():
        r = ref["units"][f"{k}_t{task}"]
        same = v.shape == r.shape and np.array_equal(v, r, equal_nan=True)
        exact_u &= bool(same)
        n_u += 1
        if v.shape == r.shape:
            dd = np.abs(v.astype(np.float64) - r.astype(np.float64))
            if np.isfinite(dd).any():
                worst_u = max(worst_u, float(np.nanmax(dd)))
    worst_r, n_r, exact_r = 0.0, 0, True
    for j, m in enumerate(models):
        rr = ref["rows"][model_key(part, m) + (task,)]
        vals = dict(online_acc=float(online_acc[j]), online_ce=float(online_ce[j]),
                    floor_acc=floors[m["seed"], m["env"]], **{k: float(v[j]) for k, v in met.items()})
        for k, v in vals.items():
            if k in rr:
                n_r += 1
                same = (v == float(rr[k])) or (math.isnan(v) and math.isnan(float(rr[k])))
                exact_r &= same
                if not same and np.isfinite(v):
                    worst_r = max(worst_r, abs(v - float(rr[k])))
    mut = None
    if task >= 2:
        mut = max(float(np.nanmax(np.abs(units[k].astype(np.float64) - ref["units"][f"{k}_t{task - 1}"].astype(np.float64))))
                  for k in units)
    return dict(task=task, units_exact=exact_u, units_maxabs=worst_u, units_keys=n_u,
                rows_exact=exact_r, rows_maxabs=worst_r, rows_values=n_r, mutation_prev_task_maxabs=mut)


class Recorder:
    def __init__(self):
        self.pts, self.units, self.model, self.resid = [], {}, {}, {}
        self.led, self.led_resid, self.led_idx = {}, {}, []
        self.w2, self.w2_task, self.mu2 = {}, [], []

    @staticmethod
    def _add(store, d, dtype):
        for k, v in d.items():
            store.setdefault(k, []).append(v.detach().cpu().numpy().astype(dtype))

    def point(self, task, step, Uf, Mo, R, S):
        self.pts.append((task, step))
        self._add(self.units, Uf, np.float32)
        self._add(self.model, Mo, np.float64)
        self._add(self.resid, {k: torch.nan_to_num(v, nan=0.0).amax(-1) for k, v in R.items()}, np.float64)
        self.mu2.append(S["mu2"].cpu().numpy())

    def interval(self, i0, i1, L, R):
        self.led_idx.append((i0, i1))
        self._add(self.led, L, np.float64)
        self._add(self.led_resid, {k: torch.nan_to_num(v, nan=0.0).amax(-1) for k, v in R.items()}, np.float64)

    def weights(self, task, P):
        self.w2_task.append(task)
        names = ("W2", "b2", "gi", "bi", "gp2", "bp2")
        for nm, q in zip(names, (P[2], P[3], P[4], P[5], P[8], P[9])):
            self.w2.setdefault(nm, []).append(q.detach().cpu().numpy().astype(np.float32))


def summarize(rec, part, models):
    rows = []
    tasks = [t for t, _ in rec.pts]
    steps = [s for _, s in rec.pts]
    for pi in range(len(rec.pts)):
        for j, m in enumerate(models):
            row = dict(part=part, **{k: m[k] for k in m}, task=tasks[pi], step=steps[pi])
            for L in ("l1", "l2"):
                for f in UNIT_SUMMARY:
                    key = f"{f}_{L}"
                    if key not in rec.units:
                        continue
                    a = rec.units[key][pi][j].astype(np.float64)
                    fin = a[np.isfinite(a)]
                    row[f"med_{key}"] = float(np.median(fin)) if fin.size else float("nan")
                    if f in QUANT_FIELDS:
                        row[f"p10_{key}"] = float(np.quantile(fin, .1)) if fin.size else float("nan")
                        row[f"p90_{key}"] = float(np.quantile(fin, .9)) if fin.size else float("nan")
                row[f"frac_Uneg_{L}"] = float(np.mean(rec.units[f"U_{L}"][pi][j] < 0))
                row[f"frac_hUneg_{L}"] = float(np.mean(rec.units[f"hU_{L}"][pi][j] < 0))
                row[f"mean_gate_{L}"] = float(np.mean(rec.units[f"gate_mean_{L}"][pi][j]))
            if "beta_l2" in rec.units:
                row["med_beta_l2"] = float(np.median(rec.units["beta_l2"][pi][j]))
            for k, v in rec.model.items():
                row[k] = float(v[pi][j])
            rows.append(row)
    return rows


def write_csv(path, rows, gz=False):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    opener = (lambda: gzip.open(path, "wt", newline="")) if gz else (lambda: open(path, "w", newline=""))
    with opener() as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.7g}" if isinstance(v, float) else v) for k, v in r.items()})


def run(part, tasks=None, out_dir=OUT, device="cuda", models=None, tag=None, dense_upto=None):
    cfg = PARTS[part]
    mod = cfg["mod"]
    tasks = tasks or cfg["tasks"]
    dense_upto = cfg["dense_upto"] if dense_upto is None else dense_upto
    tag = tag or part
    out = Path(out_dir)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    wall0 = time.monotonic()
    if part == "ext150":
        B.selftest()                     # the parent runs it before building its engine
    e, models, K1, K2, kind = build(part, device, models)
    ref = committed(part) if models in (C.MODELS, B.MODELS) else None
    gperm = {s: mod.stream("env_perm_0913", s) for s in range(3)}
    glabel = {s: mod.stream("env_labels_0913", s) for s in range(3)}
    gbatch = {s: mod.stream("env_batch_0913", s) for s in range(3)}
    ax = mod.read_idx(mod.DATA / "train-images-idx3-ubyte.gz").reshape(-1, 784).astype(np.float32) / 255
    ay = mod.read_idx(mod.DATA / "train-labels-idx1-ubyte.gz").astype(np.int64)
    subset = {s: torch.randperm(len(ax), generator=mod.stream("rl_subset", s))[:mod.N].numpy() for s in range(3)}
    xs = {s: torch.tensor(ax[subset[s]], device=e.device) for s in range(3)}
    ys_cpu = {s: torch.tensor(ay[subset[s]]) for s in range(3)}
    ys = {s: ys_cpu[s].to(e.device) for s in range(3)}
    rec = Recorder()
    taskend, bits, s6, nonfinite_first = [], [], [], {}
    state = dict(prev=None)
    diag_seconds = [0.0]

    def take(task, step):
        t_ = time.monotonic()
        Uf, Mo, R, S = diag(e, K1, K2, kind)
        idx = len(rec.pts)
        rec.point(task, step, Uf, Mo, R, S)
        if state["prev"] is not None:
            L, RL_ = ledger(state["prev"], S, mod.N)
            rec.interval(idx - 1, idx, L, RL_)
        state["prev"] = S
        diag_seconds[0] += time.monotonic() - t_
        return Uf

    e.capture()
    t0 = time.monotonic()
    for task in range(1, tasks + 1):
        steps = mod.STEPS
        epochs_needed = -(-(steps * mod.BATCH) // mod.N)
        orders = {}
        for s in range(3):
            flat = torch.stack([torch.randperm(mod.N, generator=gbatch[s]) for _ in range(epochs_needed)]).reshape(-1)[:steps * mod.BATCH]
            orders[s] = flat.reshape(steps, mod.BATCH)
        perm_cpu = {s: torch.randperm(784, generator=gperm[s]) for s in range(3)}
        perm = {s: perm_cpu[s].to(e.device) for s in range(3)}
        labels_cpu = {s: torch.randint(10, (mod.N,), generator=glabel[s]) for s in range(3)}
        labels = {s: labels_cpu[s].to(e.device) for s in range(3)}
        orderstack = torch.stack([orders[m["seed"]] for m in models], 1).to(e.device)
        floors = {}
        for s in range(3):
            floors[s, "PM"] = float(torch.bincount(ys_cpu[s], minlength=10).max()) / mod.N
            floors[s, "RL"] = float(torch.bincount(labels_cpu[s], minlength=10).max()) / mod.N
        for j, m in enumerate(models):
            s = m["seed"]
            e.cx[j].copy_(xs[s][:, perm[s]] if m["env"] == "PM" else xs[s])
            yy = ys[s] if m["env"] == "PM" else labels[s]
            e.cy[j].copy_(torch.nn.functional.one_hot(yy, 10))
        e.acc.zero_(); e.ce.zero_()
        take(task, 0)
        for step in range(0, steps, mod.BLOCK):
            e.indices.copy_(orderstack[step:step + mod.BLOCK]); e.replay_block()
            end = step + mod.BLOCK
            if task <= dense_upto and end in DENSE:
                take(task, end)
        met, units = e.evaluate()
        online_acc = (e.acc / steps).cpu().numpy()
        online_ce = (e.ce / steps).cpu().numpy()
        fin_model = torch.stack([torch.isfinite(q).flatten(1).all(1) for q in e.p]).all(0).cpu().numpy()
        for j in np.where(~fin_model)[0]:
            nonfinite_first.setdefault(int(j), task)
        Uf = take(task, steps)
        rec.weights(task, lnparams(e))
        for j, m in enumerate(models):
            taskend.append(dict(**m, task=task, floor_acc=floors[m["seed"], m["env"]],
                                online_acc=float(online_acc[j]), online_ce=float(online_ce[j]),
                                finite=bool(fin_model[j]), **{k: float(v[j]) for k, v in met.items()}))
        if ref is not None:
            bits.append(bitcheck(ref, part, models, task, met, units, online_acc, online_ce, floors))
        # S6: float64 recomputation against the engine's own float32 means (h is what evaluate() reports)
        rat = {}
        plain = ~(kind["fin"] | kind["fpre"]).cpu()
        for L, din in (("l1", 784), ("l2", 100)):
            f32 = torch.tensor(units[f"zmean_{L}"], dtype=torch.float64)
            f64 = Uf[f"hbar_{L}"].cpu()
            bound = 4 * (din + mod.N + 16) * EPS32 * Uf[f"wu_{L}"].cpu() + 1e-30
            per = torch.nan_to_num((f32 - f64).abs() / bound, nan=0.0).amax(1)
            rat[L] = float(per[plain].max()) if bool(plain.any()) else 0.0
            rat[f"{L}_ln_report_only"] = float(per[~plain].max()) if bool((~plain).any()) else 0.0
        s6.append(dict(task=task, **{f"ratio_{k}": v for k, v in rat.items()}))
        if task == 1 or task % 5 == 0 or task == tasks:
            msg = f"[{time.monotonic() - t0:7.1f}s] {tag} task {task}/{tasks} diag {diag_seconds[0]:.1f}s"
            if bits:
                msg += f" bit units={bits[-1]['units_exact']} rows={bits[-1]['rows_exact']}"
            msg += f" s6 {max(rat['l1'], rat['l2']):.3g}"
            print(msg, flush=True)
    wall = time.monotonic() - t0
    # ---------------- write
    logs = out / "logs"
    pt = np.array(rec.pts)
    np.savez_compressed(logs / f"diag_{tag}.npz", task=pt[:, 0], step=pt[:, 1],
                        **{f"u_{k}": np.stack(v) for k, v in rec.units.items()},
                        **{f"m_{k}": np.stack(v) for k, v in rec.model.items()},
                        **{f"r_{k}": np.stack(v) for k, v in rec.resid.items()})
    li = np.array(rec.led_idx)
    np.savez_compressed(logs / f"ledger_{tag}.npz", i0=li[:, 0], i1=li[:, 1],
                        **{f"l_{k}": np.stack(v) for k, v in rec.led.items()},
                        **{f"r_{k}": np.stack(v) for k, v in rec.led_resid.items()})
    np.savez_compressed(logs / f"w2_{tag}.npz", task=np.array(rec.w2_task), mu2=np.stack(rec.mu2),
                        **{k: np.stack(v) for k, v in rec.w2.items()})
    write_csv(out / f"taskend_{tag}.csv", taskend)
    write_csv(out / f"summary_{tag}.csv.gz", summarize(rec, part, models), gz=True)
    resid_max = {k: float(np.max(np.stack(v))) for k, v in rec.resid.items()}
    resid_max.update({k: float(np.max(np.stack(v))) for k, v in rec.led_resid.items()})
    checks = dict(part=part, tag=tag, tasks=tasks, points=len(rec.pts), intervals=len(rec.led_idx),
                  identity_max_ratio_to_tol=resid_max, identity_pass=all(v <= 1.0 for v in resid_max.values()),
                  s6_max_ratio={k: max(r[k] for r in s6) for k in s6[0] if k != "task"},
                  s6_pass=all(r[k] <= 1.0 for r in s6 for k in ("ratio_l1", "ratio_l2")),
                  nonfinite_models={str(k): dict(task=v, model=models[k]) for k, v in nonfinite_first.items()})
    if bits:
        checks["bit"] = dict(all_units_exact=all(b["units_exact"] for b in bits),
                             all_rows_exact=all(b["rows_exact"] for b in bits),
                             max_units_abs=max(b["units_maxabs"] for b in bits),
                             max_rows_abs=max(b["rows_maxabs"] for b in bits),
                             mutation_min_prev_task_abs=min(b["mutation_prev_task_maxabs"] for b in bits
                                                            if b["mutation_prev_task_maxabs"] is not None) if tasks > 1 else None,
                             per_task=bits)
    (out / f"checks_{tag}.json").write_text(json.dumps(checks, indent=1, default=float))
    git = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True).strip()
    prov = dict(name=NAME, part=part, tag=tag, prereg_commit=PREREG_COMMIT, git_hash=git, git_dirty=bool(dirty),
                spec_sha256=sha(SPEC), code_sha256=sha(Path(__file__)),
                parent_code_sha256={"layer_chimera_rl_0914.py": sha(C.__file__), "relu_gelu_silu_rl_0914.py": sha(B.__file__)},
                committed_reference=(ref or {}).get("sha"), data_sha256={n: sha(mod.DATA / n) for n in mod.DATA_FILES},
                torch_version=torch.__version__, device=torch.cuda.get_device_name(e.device) if e.device.type == "cuda" else "cpu",
                models=models, tasks=tasks, dense_upto=dense_upto, dense_steps=DENSE, steps_per_task=mod.STEPS,
                wall_seconds=wall, diag_seconds=diag_seconds[0], setup_seconds=t0 - wall0,
                peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20,
                registered_parts=["B", "C"] if part == "variant" else [], analysis_grade="observational_rerun" if part != "variant" else "preregistered_intervention")
    (out / f"provenance_{tag}.json").write_text(json.dumps(prov, indent=1, default=str))
    print(json.dumps({k: checks[k] for k in ("identity_pass", "s6_pass", "nonfinite_models")}, default=str), flush=True)
    if bits:
        print("BIT", json.dumps({k: v for k, v in checks["bit"].items() if k != "per_task"}), flush=True)
    return checks


# ----------------------------------------------------------------------------------------------
# self-checks with mutation controls (S1-S4, S7)
# ----------------------------------------------------------------------------------------------

def selfcheck(out_dir, device="cuda"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    res = {}
    # S1: hand-written backprop vs float64 autograd, every arm x every act pair, affine paths exercised
    g = torch.Generator().manual_seed(0)
    models = [dict(seed=0, env="RL", act1=a1, act2=a2, arm=arm, eps=1e-8) for (a1, a2) in PAIRS for arm in LN_ARMS]
    M = len(models)
    E = Flags(models, "cpu")
    base = C.initial(0)
    p = [torch.stack([base[i].double() * 3 for _ in range(M)]) for i in range(6)]
    p += [1 + 0.3 * torch.randn(M, 100, generator=g, dtype=torch.float64), 0.2 * torch.randn(M, 100, generator=g, dtype=torch.float64),
          1 + 0.3 * torch.randn(M, 100, generator=g, dtype=torch.float64), 0.2 * torch.randn(M, 100, generator=g, dtype=torch.float64),
          1 + 0.3 * torch.randn(M, 100, generator=g, dtype=torch.float64), 0.2 * torch.randn(M, 100, generator=g, dtype=torch.float64)]
    x = torch.rand(M, 16, 784, generator=g, dtype=torch.float64)
    y = torch.nn.functional.one_hot(torch.randint(10, (M, 16), generator=g), 10).double()
    pg = [q.clone().requires_grad_(True) for q in p]
    z3 = vforward(pg, x, E)["z3"]
    auto = torch.autograd.grad((z3.logsumexp(-1) - (z3 * y).sum(-1)).mean(-1).sum(), pg)
    # autograd gives the gamma/beta gradient for every model; the engine masks non-affine ones to zero
    affmask = [None] * 6 + [E.aff_in, E.aff_in, E.aff_pre, E.aff_pre, E.aff_pre, E.aff_pre]
    auto = [a if mk is None else torch.where(mk, a, torch.zeros_like(a)) for a, mk in zip(auto, affmask)]
    gmax = max(float(a.abs().max()) for a in auto)
    tol = 1e4 * EPS64 * gmax

    def gerr(mut):
        man, _, _ = vgradients(p, x, y, E, mut)
        return max(float((m_ - a_).abs().max()) for m_, a_ in zip(man, auto))
    s1 = dict(err=gerr(""), tol=tol, mutations={m: gerr(m) for m in ("ln_no_nterm", "ln_no_gamma", "swap_masks")})
    s1["pass"] = s1["err"] <= tol and all(v > 10 * tol for v in s1["mutations"].values())
    res["S1_backprop"] = s1
    print("S1", json.dumps(s1), flush=True)

    # shared task-1 data for S2/S3
    mod = C
    ax = mod.read_idx(mod.DATA / "train-images-idx3-ubyte.gz").reshape(-1, 784).astype(np.float32) / 255
    subset = {s: torch.randperm(len(ax), generator=mod.stream("rl_subset", s))[:mod.N].numpy() for s in range(3)}
    glabel = {s: mod.stream("env_labels_0913", s) for s in range(3)}
    gbatch = {s: mod.stream("env_batch_0913", s) for s in range(3)}
    labels = {s: torch.randint(10, (mod.N,), generator=glabel[s]) for s in range(3)}
    order = {s: torch.randperm(mod.N, generator=gbatch[s])[:mod.BLOCK * mod.BATCH].reshape(mod.BLOCK, mod.BATCH) for s in range(3)}

    def load(eng, mods):
        for j, m in enumerate(mods):
            s = m["seed"]
            eng.cx[j].copy_(torch.tensor(ax[subset[s]], device=eng.device))   # RL-style inputs for both envs in this check
            eng.cy[j].copy_(torch.nn.functional.one_hot(labels[s].to(eng.device), 10))
        eng.indices.copy_(torch.stack([order[m["seed"]] for m in mods], 1).to(eng.device))

    # S2: the variant engine with only `none` models is the host engine, bit for bit (same batch size)
    host = C.Engine(models=C.MODELS, device=device)
    same = [dict(**m, arm="none", eps=1e-8) for m in C.MODELS]
    var = VariantEngine(same, device=device)
    load(host, C.MODELS); load(var, same)
    with torch.no_grad():
        fh = C.forward(host.p, host.cx, host.elu1, host.elu2)
        fv = vforward(var.p, var.cx, var)
    fwd_exact = all(torch.equal(a, b) for a, b in zip(fh, (fv["z1"], fv["a1"], fv["z2"], fv["a2"], fv["z3"])))
    host.block(); var.block()
    upd_exact = all(torch.equal(a, b) for a, b in zip(host.p + host.m + host.v, var.p[:6] + var.m[:6] + var.v[:6]))
    ext_zero = all(torch.equal(q, q0) for q, q0 in zip(var.p[6:], [torch.ones_like(var.p[6]), torch.zeros_like(var.p[6])] * 3))
    ev_h = host.evaluate(); ev_v = var.evaluate()
    eval_exact = all(np.array_equal(ev_h[1][k], ev_v[1][k]) for k in ev_h[1]) and all(np.array_equal(ev_h[0][k], ev_v[0][k]) for k in ev_h[0])
    # mutation: one ulp on one weight of the variant engine before the block must be seen
    var2 = VariantEngine(same, device=device)
    load(var2, same)
    with torch.no_grad():
        var2.p[0][0, 0, 400] = torch.nextafter(var2.p[0][0, 0, 400], torch.tensor(1.0, device=var2.device))
    var2.block()
    mut_seen = not torch.equal(var2.p[0], host.p[0])
    # the full 132-model engine: none-models against the host (batch-size effect, report only)
    full = VariantEngine(VARIANT_MODELS, device=device)
    load(full, VARIANT_MODELS)
    full.block()
    idx = [next(j for j, v in enumerate(VARIANT_MODELS) if v["arm"] == "none" and all(v[k] == m[k] for k in ("seed", "env", "act1", "act2"))) for m in C.MODELS]
    full_dev = max(float((full.p[i][idx] - host.p[i]).abs().max()) for i in range(6))
    res["S2_host_equivalence"] = dict(forward_exact=fwd_exact, update_exact=upd_exact, eval_exact=eval_exact,
                                      ln_params_untouched=ext_zero, mutation_one_ulp_seen=mut_seen,
                                      full_engine_none_vs_host_maxabs_after_25_steps=full_dev,
                                      **{"pass": fwd_exact and upd_exact and eval_exact and ext_zero and mut_seen})
    print("S2", json.dumps(res["S2_host_equivalence"]), flush=True)

    # S3: per-model eps is what the engine applies (one eager step from the task-1 state)
    e3 = VariantEngine(VARIANT_MODELS, device=device)
    load(e3, VARIANT_MODELS)
    x3 = e3.cx[e3.mid, e3.indices[0]]
    y3 = e3.cy[e3.mid, e3.indices[0]]
    p0 = [q.clone() for q in e3.p]
    with torch.no_grad():
        gr, _, _ = vgradients(p0, x3, y3, e3)
    e3.step(x3, y3)
    t1 = torch.ones((), device=e3.device)
    c1 = 1 - torch.pow(.9, t1); c2 = 1 - torch.pow(.999, t1)
    exact, affected = True, 0
    for q0, g_, q1, ep in zip(p0, gr, e3.p, e3.eps_param):
        m_ = torch.zeros_like(g_).mul_(.9).add_(g_, alpha=.1)
        v_ = torch.zeros_like(g_).mul_(.999).addcmul_(g_, g_, value=.001)
        want = q0.clone().addcdiv_(m_ / c1, (v_ / c2).sqrt() + ep, value=-.001)
        wrong = q0.clone().addcdiv_(m_ / c1, (v_ / c2).sqrt() + 1e-8, value=-.001)
        exact &= torch.equal(want, q1)
        affected += int((wrong != q1).sum())
    res["S3_eps"] = dict(update_exact=exact, coords_changed_by_eps_mutation=affected, **{"pass": exact and affected > 0})
    print("S3", json.dumps(res["S3_eps"]), flush=True)

    del host, var, var2, full, e3
    torch.cuda.empty_cache()
    # S4 + S5 + S7: two chimera tasks with the real loop, then mutated diagnostics on its last state
    t_ = time.monotonic()
    ck = run("chimera", tasks=2, out_dir=out / "smoke_chimera", device=device, tag="smoke_chimera")
    res["S5_bit_two_tasks"] = ck.get("bit", {})
    res["S4_identities_unmutated"] = dict(max_ratio=ck["identity_max_ratio_to_tol"], **{"pass": ck["identity_pass"]})
    res["S6"] = dict(max_ratio=ck["s6_max_ratio"], **{"pass": ck["s6_pass"]})
    res["S7_cost_chimera_2tasks_seconds"] = time.monotonic() - t_
    e4, models4, K1, K2, kind = build("chimera", device)
    load(e4, models4)
    for _ in range(3):
        e4.block()
    Ua, Ma, Ra, Sa = diag(e4, K1, K2, kind)
    for _ in range(6):
        e4.block()
    Ub, Mb, Rb, Sb = diag(e4, K1, K2, kind)
    muts = {}
    for mname in ("axis", "roll", "cov", "f32mu"):
        _, _, Rm, _ = diag(e4, K1, K2, kind, mut=mname)
        muts[f"diag_{mname}"] = {k: float(torch.nan_to_num(v, nan=0.0).max()) for k, v in Rm.items()}
    L0, R0 = ledger(Sa, Sb, mod.N)
    muts["ledger_unmutated"] = {k: float(torch.nan_to_num(v, nan=0.0).max()) for k, v in R0.items()}
    for mname in ("cross", "rold", "Sold", "f32mu"):
        _, Rm = ledger(Sa, Sb, mod.N, mut=mname)
        muts[f"ledger_{mname}"] = {k: float(torch.nan_to_num(v, nan=0.0).max()) for k, v in Rm.items()}
    detect = {"diag_axis": ("zbar_id_l2",), "diag_roll": ("zbar_id_l2",), "diag_cov": ("var_split_l2",),
              "diag_f32mu": ("zbar_id_l2",), "ledger_cross": ("ledger_closure",), "ledger_rold": ("stretch_rot",),
              "ledger_Sold": ("corner_d2", "corner_rho2", "corner_sig2"), "ledger_f32mu": ("ledger_closure",)}
    seen = {k: max(muts[k][c] for c in cs) > 10 for k, cs in detect.items()}
    unm = max(max(v.values()) for k, v in muts.items() if k == "ledger_unmutated")
    unm_diag = max(float(torch.nan_to_num(v, nan=0.0).max()) for v in Rb.values())
    res["S4_mutations"] = dict(values=muts, detected=seen, unmutated_ledger_max=unm, unmutated_diag_max=unm_diag,
                               **{"pass": all(seen.values()) and unm <= 1 and unm_diag <= 1})
    print("S4", json.dumps({k: v for k, v in res["S4_mutations"].items() if k != "values"}), flush=True)
    # S7: one variant task for cost (all 132 models, dense diagnostics on)
    t_ = time.monotonic()
    ckv = run("variant", tasks=1, out_dir=out / "smoke_variant", device=device, tag="smoke_variant")
    res["S7_cost_variant_1task_seconds"] = time.monotonic() - t_
    res["S7_variant_identity_pass"] = ckv["identity_pass"]
    res["S7_variant_identity_max"] = ckv["identity_max_ratio_to_tol"]
    res["S7_peak_rss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20
    res["all_pass"] = all(v.get("pass", True) for v in res.values() if isinstance(v, dict)) and bool(res["S5_bit_two_tasks"].get("all_units_exact")) and bool(res["S5_bit_two_tasks"].get("all_rows_exact"))
    (out / "checks_selfcheck.json").write_text(json.dumps(res, indent=1, default=float))
    print("ALL_PASS", res["all_pass"], flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=sorted(PARTS))
    ap.add_argument("--tasks", type=int, default=None)
    ap.add_argument("--out-dir", default=str(OUT))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    if a.selfcheck:
        selfcheck(a.out_dir, a.device)
    else:
        run(a.part, a.tasks, a.out_dir, a.device)
