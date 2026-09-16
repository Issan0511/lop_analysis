#!/usr/bin/env python3
"""H4: does the second layer's response field decide whether the RL ELU->ELU net can still learn?

    OMP_NUM_THREADS=1 python3 src/resp_ee_0917.py --seed 0 --out results/resp_ee_0917/runs/s0

specs/spec_resp_ee_0917.md (registered at PREREG_COMMIT).

The box is mucap_ee_0917's ref arm (Random Label MNIST, 1200 fixed images, 784-100-100-10, ELU on both
hidden layers, Adam 1e-3, 6000 updates per task, flush_denormal on).  train_task() is the host's
update loop written out (src/mucap_el_run_0916.py run_one): same draws, same arithmetic, so the prefix is
the ref trajectory bit for bit (checks S0, S1).

One process per seed:
  1. the natural prefix, tasks 1-22, with the full state and both layers' preactivation fields on the
     1200 images saved at the end of the branch tasks 2, 5, 7, 10, 15, 20;
  2. 26 arms, each continuing one branch state for 2 new tasks with the prefix's own label and order
     streams (so N<t> is the natural trajectory again, check S2).

The intervention changes a layer's response and nothing else at the branch point.  For layer l with a
fixed shift field d_i(x) (a unit x image table), frozen branch parameters p0 and z0 = the frozen net's
preactivation,

    a_i(x) = phi(z0_i(x)) + [ phi(z_i(x) + d_i(x)) - phi(z0_i(x) + d_i(x)) ],

so at the branch point the bracket is exactly 0 (same ops on the same bits) and the features, logits
and loss are the natural net's bit for bit (check S3), while d a / d z = phi'(z + d) from the first
update on (check S4).  z0 is recomputed from p0 on each minibatch rather than cached, which is what
makes the equality exact.  Two kinds of field:

    field    d = z^(src)(x) - z^(branch)(x): the unit responds as it did at task src, plus whatever it
             learns from here (anchor 0913's shift made per image and per unit)
    uniform  d = -delta for every unit and image

Adam: "r" arms start from zero moments and step count (a fresh optimizer, first step = lr * sign);
the others keep the branch state's moments.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import resource
import socket
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import pmnist_0905 as H                 # host; not touched
from src import pmnist_rlmnist_0906 as RL        # subset, labels; not touched
from src import shell_l2_rlmnist_0913 as SH      # state_sha256, file_sha256, git_dirty; not touched
from src import elu_growth_0909 as EG            # ELU1; not touched
from src import mucap_el_run_0916 as EL          # forward2 (mucap_ee_0917's version, byte-identical)

EXPERIMENT = "resp_ee_0917"
SPEC = "specs/spec_resp_ee_0917.md"
PREREG_COMMIT = None                             # written after the spec is pushed, before any arm runs
ACT = EG.ELU(1.0)
LR = 1e-3
BETA1, BETA2, EPS = 0.9, 0.999, 1e-8             # the host's Adam
N_IMG = RL.N_IMAGES                              # 1200
BATCH = RL.BATCH                                 # 16
SPE = RL.STEPS_PER_EPOCH                         # 75
EPOCHS = 80                                      # 6000 updates per task
BRANCH_T = (2, 5, 7, 10, 15, 20)
CONT = 2                                         # new tasks per arm
PREFIX_T = max(BRANCH_T) + CONT                  # 22: N<t> must meet the prefix at t+1 and t+2
LOWGATE = 0.05
WEIGHTS = (0, 2, 4)                              # W1, W2, W3 in [W1, b1, W2, b2, W3, b3]
LADDER = (5, 10, 15, 20, 30)
# The host ELU is where(z > 0, z, expm1(min(z, 0))) and autograd differentiates expm1 as (result + 1).
# Near -1 float32 steps by 2**-24, and this machine's float32 expm1 kernel returns exactly -1 once
# exp(z) < 2**-24 (measured 2026-09-17: the largest float32 z with a zero factor is -16.635534, the next
# one up gives 2**-24 -- a truncation, not the round-to-nearest 2**-25), so the derivative the optimizer
# sees is exactly 0 for z < ln(2**-24) = -16.64 and moves in steps of 2**-24 above it -- not exp(z).
# dphi_train is that factor, bit for bit (check S4b); ACT.dphi (exp) is the analytic one the older
# diagnostics report.  The threshold is a property of the kernel, so of the machine.
ZERO_Z32 = math.log(2.0 ** -24)


def dphi_train(z: torch.Tensor) -> torch.Tensor:
    return torch.where(z > 0, torch.ones_like(z), torch.expm1(z.clamp(max=0.0)) + 1)


def _arm(branch, layer=None, kind=None, src=None, delta=None, reset=False):
    return {"branch": branch, "layer": layer, "kind": kind, "src": src, "delta": delta, "reset": reset}


ARMS: dict[str, dict] = {}
for _s in (5, 7, 10, 15, 20):                    # sink: the healthy t2 net answers with a later field
    ARMS[f"S2_{_s}r"] = _arm(2, 2, "field", src=_s, reset=True)
ARMS["S2_10"] = _arm(2, 2, "field", src=10)
for _d in (5, 10, 15, 20, 30):                   # the uniform ladder
    ARMS[f"S2u{_d}r"] = _arm(2, 2, "uniform", delta=float(_d), reset=True)
ARMS["S1u20r"] = _arm(2, 1, "uniform", delta=20.0, reset=True)
for _t in (5, 7, 10, 15, 20):                    # restore: a collapsed net answers with the t2 field
    ARMS[f"R2_{_t}"] = _arm(_t, 2, "field", src=2)
ARMS["R2_10r"] = _arm(10, 2, "field", src=2, reset=True)
ARMS["N2r"] = _arm(2, reset=True)
ARMS["N10r"] = _arm(10, reset=True)
for _t in BRANCH_T:                              # natural continuations last (an aliasing bug in an
    ARMS[f"N{_t}"] = _arm(_t)                    # earlier arm shows up as a hash mismatch here)


def eps_points(spt: int) -> tuple[int, ...]:
    """Updates into a task at which the eps-dominated share of each weight is read: 1, 75, 1/4, end."""
    return tuple(sorted({1, min(SPE, spt), max(1, spt // 4), spt}))


def gen_from(state: torch.Tensor) -> torch.Generator:
    g = torch.Generator(device="cpu")
    g.set_state(state.clone())
    return g


# --------------------------------------------------------------------------
# the response-shifted forward
# --------------------------------------------------------------------------

def make_forward(sh: dict | None):
    """None -> the host's forward2.  Otherwise sh = {"p0": frozen params, 1: d1 or None, 2: d2 or None}."""
    if sh is None:
        return None
    p0 = sh["p0"]
    d1, d2 = sh.get(1), sh.get(2)

    def fwd(params, xb, ob):
        W1, b1, W2, b2, W3, b3 = params
        z1 = xb @ W1.T + b1
        with torch.no_grad():
            z01 = xb @ p0[0].T + p0[1]
            a01 = ACT.phi(z01)
        if d1 is None:
            a1 = ACT.phi(z1)
        else:
            e1 = d1[ob]
            with torch.no_grad():
                A1 = ACT.phi(z01 + e1)
            a1 = a01 + (ACT.phi(z1 + e1) - A1)
        z2 = a1 @ W2.T + b2
        if d2 is None:
            a2 = ACT.phi(z2)
        else:
            e2 = d2[ob]
            with torch.no_grad():
                z02 = a01 @ p0[2].T + p0[3]
                a02 = ACT.phi(z02)
                A2 = ACT.phi(z02 + e2)
            a2 = a02 + (ACT.phi(z2 + e2) - A2)
        return z1, a1, z2, a2, a2 @ W3.T + b3

    return fwd


def build_shift(arm: dict, cks: dict) -> dict | None:
    if arm["kind"] is None:
        return None
    ck = cks[arm["branch"]]
    L = arm["layer"]
    if arm["kind"] == "field":
        d = (cks[arm["src"]][f"z{L}"].double() - ck[f"z{L}"].double()).float()
    elif arm["kind"] == "uniform":
        d = torch.full_like(ck[f"z{L}"], -float(arm["delta"]))
    else:
        raise ValueError(arm["kind"])
    return {"p0": [p.detach().clone() for p in ck["params"]], L: d}


def forward(params, x, ob, fwd):
    return EL.forward2(params, x, ACT, ACT) if fwd is None else fwd(params, x, ob)


# --------------------------------------------------------------------------
# per-unit state on the 1200 images (float64)
# --------------------------------------------------------------------------

@torch.no_grad()
def unit_point(params, x, fwd, sh) -> tuple[dict, torch.Tensor]:
    """Both hidden layers: the unshifted preactivation (zbar, sigma, U) and the response the net
    actually trains with, phi'(z + d): its mean, the low-gate share, the positive share of z + d and
    neff = (mean g)^2 / mean g^2 in (0, 1] -- how many images carry the unit's gradient, independent of
    its size (log-space: exp(z) underflows float64 near z = -745).  ELU: log g = min(z + d, 0)."""
    z1, a1, z2, _, logits = forward(params, x, torch.arange(x.shape[0]), fwd)
    out = {}
    for li, z in ((1, z1), (2, z2)):
        d = sh.get(li) if sh is not None else None
        z64 = z.double()
        ze = z64 + d.double() if d is not None else z64
        lg = ze.clamp(max=0.0)                      # log phi'(z + d) for ELU(1)
        lse1 = torch.logsumexp(lg, 0)
        lse2 = torch.logsumexp(2 * lg, 0)
        n = z64.shape[0]
        out[f"zbar_l{li}"] = z64.mean(0).numpy()
        out[f"sigma_l{li}"] = z64.std(0, unbiased=False).numpy()
        out[f"U_l{li}"] = z64.amax(0).numpy()
        out[f"effbar_l{li}"] = ze.mean(0).numpy()
        out[f"Ueff_l{li}"] = ze.amax(0).numpy()
        out[f"pplus_eff_l{li}"] = (ze > 0).double().mean(0).numpy()
        out[f"loggmean_l{li}"] = (lse1 - math.log(n)).numpy()
        out[f"gmean_l{li}"] = torch.exp(lse1 - math.log(n)).numpy()
        out[f"lowgate_l{li}"] = (lg < math.log(LOWGATE)).double().mean(0).numpy()
        out[f"neff_l{li}"] = torch.exp(2 * lse1 - lse2 - math.log(n)).numpy()
        ze32 = z + d if d is not None else z                 # the float32 argument training uses
        gt = dphi_train(ze32).double()
        out[f"gtrain_l{li}"] = gt.mean(0).numpy()
        out[f"gzero_l{li}"] = (gt == 0).double().mean(0).numpy()
        s1 = gt.sum(0)
        out[f"neffT_l{li}"] = torch.where(s1 > 0, s1 * s1 / (n * gt.square().sum(0)).clamp(min=1e-300),
                                          torch.zeros_like(s1)).numpy()
    out["mu2_norm"] = np.array([float(a1.double().mean(0).norm())])
    return out, logits


def unit_summary(u: dict, tag: str) -> dict:
    """Unit means of the per-unit arrays (the logs are averaged in log space: a geometric mean)."""
    s = {}
    for li in (1, 2):
        s[f"g{li}_{tag}"] = float(np.mean(u[f"gmean_l{li}"]))
        s[f"logg{li}_{tag}"] = float(np.mean(u[f"loggmean_l{li}"]))
        s[f"neff{li}_{tag}"] = float(np.mean(u[f"neff_l{li}"]))
        s[f"lowgate{li}_{tag}"] = float(np.mean(u[f"lowgate_l{li}"]))
        s[f"zbar{li}_{tag}"] = float(np.mean(u[f"zbar_l{li}"]))
        s[f"sigma{li}_{tag}"] = float(np.mean(u[f"sigma_l{li}"]))
        s[f"effbar{li}_{tag}"] = float(np.mean(u[f"effbar_l{li}"]))
        s[f"pplus{li}_{tag}"] = float(np.mean(u[f"pplus_eff_l{li}"]))
        s[f"gtr{li}_{tag}"] = float(np.mean(u[f"gtrain_l{li}"]))
        s[f"zero{li}_{tag}"] = float(np.mean(u[f"gzero_l{li}"]))
        s[f"deadunits{li}_{tag}"] = float(np.mean(u[f"gzero_l{li}"] == 1.0))
        s[f"neffT{li}_{tag}"] = float(np.mean(u[f"neffT_l{li}"]))
    s[f"mu2_{tag}"] = float(u["mu2_norm"][0])
    return s


# --------------------------------------------------------------------------
# one task of updates: the host's loop (src/mucap_el_run_0916.py run_one, ref arm)
# --------------------------------------------------------------------------

def train_task(params, adam, x, y, g_batch, epochs, fwd=None, probe_mb=False):
    spt = SPE * epochs
    pts = set(eps_points(spt))
    ce = torch.empty(spt)
    acc = torch.empty(spt)
    stp = torch.empty(spt, len(WEIGHTS))
    epsf = {}
    mb_equal = None
    m, v, tc = adam
    for ep in range(epochs):
        order = torch.randperm(N_IMG, generator=g_batch).to(x.device)
        xs, ys = x[order], y[order]
        for j in range(SPE):
            s = ep * SPE + j
            xb, yb = xs[j * BATCH:(j + 1) * BATCH], ys[j * BATCH:(j + 1) * BATCH]
            if fwd is None:
                out = EL.forward2(params, xb, ACT, ACT)
            else:
                out = fwd(params, xb, order[j * BATCH:(j + 1) * BATCH])
                if probe_mb and s == 0:
                    with torch.no_grad():
                        mb_equal = bool(torch.equal(EL.forward2(params, xb, ACT, ACT)[4], out[4].detach()))
            loss = F.cross_entropy(out[4], yb)
            ce[s] = loss.detach()
            acc[s] = (out[4].detach().argmax(1) == yb).float().mean()
            grads = torch.autograd.grad(loss, params)
            with torch.no_grad():
                tc[0] += 1
                c1, c2 = 1 - BETA1 ** tc[0], 1 - BETA2 ** tc[0]
                for k, (p, gr, mi, vi) in enumerate(zip(params, grads, m, v)):
                    mi.mul_(BETA1).add_(gr, alpha=1 - BETA1)
                    vi.mul_(BETA2).addcmul_(gr, gr, value=1 - BETA2)
                    upd = LR * (mi / c1) / ((vi / c2).sqrt() + EPS)
                    p -= upd
                    if k % 2 == 0:
                        stp[s, k // 2] = upd.square().mean().sqrt()
                if s + 1 in pts:
                    epsf[s + 1] = [float(((v[k] / c2).sqrt() < EPS).double().mean()) for k in WEIGHTS]
    return ce, acc, stp, epsf, mb_equal


def task_row(ce, acc, stp, epsf, y, spt) -> dict:
    """Online means are float64 sums of the per-update float32 values (acc: exact, k/16 each)."""
    ce64, acc64, st64 = ce.double(), acc.double(), stp.double()
    first = min(SPE, spt)
    mf = float(torch.bincount(y, minlength=10).max()) / N_IMG
    row = {"online_acc": float(acc64.sum()) / spt, "online_ce": float(ce64.sum()) / spt,
           "first75_acc": float(acc64[:first].sum()) / first, "first75_ce": float(ce64[:first].sum()) / first,
           "last750_acc": float(acc64[-max(1, spt // 8):].mean()),
           "major_frac": mf, "finite": bool(torch.isfinite(ce).all())}
    for i, w in enumerate(("w1", "w2", "w3")):
        row[f"step_{w}"] = float(st64[:, i].mean()) / LR
        row[f"step_{w}_first75"] = float(st64[:first, i].mean()) / LR
    for p, fr in sorted(epsf.items()):
        for i, w in enumerate(("w1", "w2", "w3")):
            row[f"eps_{w}_at{p}"] = fr[i]
    return row


# --------------------------------------------------------------------------
# the prefix
# --------------------------------------------------------------------------

def run_prefix(seed: int, mnist: H.Mnist, epochs: int = EPOCHS, n_tasks: int = PREFIX_T,
               save_at=BRANCH_T, stop_at=None, progress=False):
    """stop_at(task, row) -> True ends the prefix early (checks: first mismatch against a record)."""
    t0 = time.time()
    params = H.init_params(seed, H.setup("cpu"))
    adam = ([torch.zeros_like(q) for q in params], [torch.zeros_like(q) for q in params], [0])
    idx = RL.subset_idx(seed)
    x = mnist.train_x[idx]
    g_lab, g_batch = H.stream("rl_labels", seed), H.stream("rl_batch", seed)
    spt = SPE * epochs
    rows, units, cks = [], {}, {}
    info = {"init_sha256": hashlib.sha256(b"".join(q.detach().numpy().tobytes() for q in params)).hexdigest(),
            "subset_idx_sha256": hashlib.sha256(idx.numpy().tobytes()).hexdigest()}
    for t in range(1, n_tasks + 1):
        t_task = time.time()
        y = RL.task_labels(g_lab)
        ce, acc, stp, epsf, _ = train_task(params, adam, x, y, g_batch, epochs)
        with torch.no_grad():
            z1, _, z2, _, logits = EL.forward2(params, x, ACT, ACT)
            memo = float((logits.argmax(1) == y).float().mean())
        u, _ = unit_point(params, x, None, None)
        row = {"seed": seed, "task": t, **task_row(ce, acc, stp, epsf, y, spt), "memo_acc": memo,
               "state_sha256": SH.state_sha256(params, adam, ACT), **unit_summary(u, "end"),
               "sec": time.time() - t_task}
        rows.append(row)
        for k, a in u.items():
            units.setdefault(f"prefix_{k}", []).append(a)
        if t in save_at:
            cks[t] = {"params": [p.detach().clone() for p in params],
                      "m": [q.clone() for q in adam[0]], "v": [q.clone() for q in adam[1]],
                      "tc": adam[2][0], "g_lab": g_lab.get_state().clone(),
                      "g_batch": g_batch.get_state().clone(), "state_sha256": row["state_sha256"],
                      "z1": z1.detach().clone(), "z2": z2.detach().clone(), "task": t}
        if progress:
            print(f"[{time.time() - t0:8.1f}s] seed={seed} prefix task {t}/{n_tasks} "
                  f"online {row['online_acc']:.4f}", flush=True)
        if stop_at is not None and stop_at(t, row):
            break
    units = {k: np.stack(v) for k, v in units.items()}
    return rows, units, cks, x, info


# --------------------------------------------------------------------------
# one arm
# --------------------------------------------------------------------------

def preflight(params, fwd, x, y, g_batch_state, epochs) -> dict:
    """At the branch point, before any update: the per-coordinate RMS over the first epoch's 75
    minibatches of each weight's gradient under the arm's forward (a clone of the order stream, so the
    run is untouched).  A fresh Adam's sqrt(v-hat) is this RMS, so 'rms < eps' is the eps-dominated
    share the arm starts with."""
    gb = gen_from(g_batch_state)
    order = torch.randperm(N_IMG, generator=gb).to(x.device)
    ws = [params[k] for k in WEIGHTS]
    sq = [torch.zeros_like(w, dtype=torch.float64) for w in ws]
    nb = min(SPE, SPE * epochs)
    for j in range(nb):
        ob = order[j * BATCH:(j + 1) * BATCH]
        out = forward(params, x[ob], ob, fwd)
        loss = F.cross_entropy(out[4], y[ob])
        for acc_, g in zip(sq, torch.autograd.grad(loss, ws)):
            acc_ += g.double().square()
    res = {}
    for w, acc_ in zip(("w1", "w2", "w3"), sq):
        rms = (acc_ / nb).sqrt()
        pos = rms[rms > 0]
        res[f"pf_eps_{w}"] = float((rms < EPS).double().mean())
        res[f"pf_zero_{w}"] = float((rms == 0).double().mean())
        res[f"pf_logrms_med_{w}"] = float(pos.log().median()) if pos.numel() else float("-inf")
    return res


def dstar(cks: dict, x: torch.Tensor, epochs: int) -> dict:
    """The uniform ladder's arithmetic (spec 2.4): at the t2 state, the W2 gradient RMS of the same net
    with the second layer's derivative replaced by exp(z) (the ELU's negative branch, extended), over
    the first epoch of task 3.  Shifting every z by -delta multiplies it by exp(-delta) while every
    shifted z is negative, so the share of W2 coordinates below eps at the start of S2u<delta>r is the
    share with ln(rms/eps) < delta.  Returns the quantiles of ln(rms/eps) over W2's coordinates."""
    ck = cks[2]
    params = [p.detach().clone().requires_grad_(True) for p in ck["params"]]
    p0 = [p.detach().clone() for p in ck["params"]]
    y = RL.task_labels(gen_from(ck["g_lab"]))
    order = torch.randperm(N_IMG, generator=gen_from(ck["g_batch"]))
    sq = torch.zeros_like(params[2], dtype=torch.float64)
    nb = min(SPE, SPE * epochs)
    for j in range(nb):
        ob = order[j * BATCH:(j + 1) * BATCH]
        xb = x[ob]
        W1, b1, W2, b2, W3, b3 = params
        a1 = ACT.phi(xb @ W1.T + b1)
        z2 = a1 @ W2.T + b2
        with torch.no_grad():
            z02 = ACT.phi(xb @ p0[0].T + p0[1]) @ p0[2].T + p0[3]
            base = ACT.phi(z02) - torch.expm1(z02)
        a2 = base + torch.expm1(z2)
        loss = F.cross_entropy(a2 @ W3.T + b3, y[ob])
        sq += torch.autograd.grad(loss, W2)[0].double().square()
    rms = (sq / nb).sqrt()
    lr_ = (rms[rms > 0] / EPS).log()
    qs = (0.1, 0.25, 0.5, 0.75, 0.9)
    # the float32 zero: a unit's training derivative is 0 on every image once delta > U_i - ZERO_Z32
    U = ck["z2"].double().amax(0)
    z0 = (U - ZERO_Z32)
    rungs = {}
    for d in LADDER:
        gt = dphi_train(ck["z2"] - float(d))
        rungs[str(d)] = {"pair_zero_share": float((gt == 0).double().mean()),
                         "unit_all_zero_share": float(((gt == 0).all(0)).double().mean()),
                         "eps_share_exp_model": float((lr_ < d).double().mean())}
    return {"dstar_q": {str(q): float(torch.quantile(lr_, q)) for q in qs},
            "dzero_q": {str(q): float(torch.quantile(z0, q)) for q in qs},
            "zero_rms_share": float((rms == 0).double().mean()), "zero_z32": ZERO_Z32, "rungs": rungs,
            "U2_max_t2": float(ck["z2"].amax()), "U2_unit_median_t2": float(ck["z2"].amax(0).median())}


def run_arm(name: str, arm: dict, cks: dict, x: torch.Tensor, prefix_rows: list, seed: int,
            epochs: int = EPOCHS, units: dict | None = None, progress=False) -> list[dict]:
    t0 = time.time()
    ck = cks[arm["branch"]]
    params = [p.detach().clone().requires_grad_(True) for p in ck["params"]]
    if arm["reset"]:
        adam = ([torch.zeros_like(q) for q in params], [torch.zeros_like(q) for q in params], [0])
    else:
        adam = ([q.clone() for q in ck["m"]], [q.clone() for q in ck["v"]], [ck["tc"]])
    g_lab, g_batch = gen_from(ck["g_lab"]), gen_from(ck["g_batch"])
    sh = build_shift(arm, cks)
    fwd = make_forward(sh)
    spt = SPE * epochs
    by_task = {r["task"]: r for r in prefix_rows}
    with torch.no_grad():
        nat = EL.forward2(params, x, ACT, ACT)[4]
    rows = []
    for k in range(1, CONT + 1):
        t_task = time.time()
        y = RL.task_labels(g_lab)
        head = {}
        if k == 1:
            u0, lg0 = unit_point(params, x, fwd, sh)
            head = {"logits_equal_full": bool(torch.equal(lg0, nat)),
                    "logit_maxdiff_full": float((lg0.double() - nat.double()).abs().max()),
                    **unit_summary(u0, "start"),
                    **preflight(params, fwd, x, y, g_batch.get_state(), epochs)}
            if units is not None:
                for kk, a in u0.items():
                    units[f"{name}_p0_{kk}"] = a
        ce, acc, stp, epsf, mbeq = train_task(params, adam, x, y, g_batch, epochs, fwd, probe_mb=(k == 1))
        with torch.no_grad():
            u1, lg1 = unit_point(params, x, fwd, sh)
            memo = float((lg1.argmax(1) == y).float().mean())
            full_ce = float(F.cross_entropy(lg1.double(), y))
        if units is not None:
            for kk, a in u1.items():
                units[f"{name}_p{k}_{kk}"] = a
        h = SH.state_sha256(params, adam, ACT)
        task = arm["branch"] + k
        pref = by_task.get(task)
        tr = task_row(ce, acc, stp, epsf, y, spt)
        if k == 1 and fwd is None:
            mbeq = True                               # the natural forward is the reference itself
        row = {"seed": seed, "arm": name, **{f"arm_{a}": arm[a] for a in arm}, "k": k, "task": task,
               **tr, "memo_acc_end": memo, "full_ce_end": full_ce, "state_sha256": h,
               "hash_match_prefix": (None if pref is None else h == pref["state_sha256"]),
               "acc_match_prefix": (None if pref is None else tr["online_acc"] == pref["online_acc"]),
               "logits_equal_mb": (mbeq if k == 1 else None),
               **head, **unit_summary(u1, "end"), "sec": time.time() - t_task}
        if units is not None:
            units[f"{name}_k{k}_ce"] = ce.numpy().astype(np.float32)
            units[f"{name}_k{k}_acc"] = acc.numpy().astype(np.float32)
            units[f"{name}_k{k}_step"] = stp.numpy().astype(np.float32)
        rows.append(row)
    if progress:
        print(f"[{time.time() - t0:8.1f}s] seed={seed} arm {name}: "
              + " ".join(f"t{r['task']} {r['online_acc']:.4f}" for r in rows), flush=True)
    return rows


# --------------------------------------------------------------------------

def write_csv(path: Path, rows: list[dict]) -> None:
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in keys})


def run_seed(seed: int, out: Path, mnist: H.Mnist, epochs: int = EPOCHS, arms=None, prefix_tasks=PREFIX_T,
             save_ckpt: bool = True, progress: bool = True) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    names = [a for a in ARMS if arms is None or a in arms]
    need = {ARMS[a]["branch"] for a in names} | {ARMS[a]["src"] for a in names if ARMS[a]["src"]}
    need |= {2}
    prows, units, cks, x, info = run_prefix(seed, mnist, epochs, prefix_tasks,
                                            save_at=tuple(sorted(need)), progress=progress)
    t_prefix = time.time() - t0
    ds = dstar(cks, x, epochs)
    rows = []
    for a in names:
        rows += run_arm(a, ARMS[a], cks, x, prows, seed, epochs, units=units, progress=progress)
    write_csv(out / "prefix.csv", prows)
    write_csv(out / "arms.csv", rows)
    (out / "dstar.json").write_text(json.dumps(ds, indent=2))
    np.savez_compressed(out / "units.npz", **units)
    if save_ckpt:
        torch.save({t: {k: v for k, v in c.items()} for t, c in cks.items()}, out / "branch_states.pt")
    root = Path(__file__).resolve().parents[1]
    prov = {
        "experiment": EXPERIMENT, "spec": SPEC, "prereg_commit": PREREG_COMMIT,
        "spec_sha256": SH.file_sha256(root / SPEC) if (root / SPEC).exists() else None,
        "git_hash": H.git_hash(), "git_dirty_code": SH.git_dirty(["src", "analysis/resp_ee_0917"]),
        "hostname": socket.gethostname(), "platform": platform.platform(),
        "cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "torch": torch.__version__, "python": sys.version.split()[0],
        "threads": torch.get_num_threads(), "flush_denormal": _flush_is_on(),
        "seed": seed, "epochs_per_task": epochs, "steps_per_task": SPE * epochs, "prefix_tasks": prefix_tasks,
        "branch_tasks": sorted(cks), "cont_tasks": CONT, "arms": {a: ARMS[a] for a in names},
        "lr": LR, "adam": [BETA1, BETA2, EPS], "batch": BATCH, "n_images": N_IMG,
        "code_sha256": {f"src/{n}": SH.file_sha256(root / "src" / n) for n in
                        ("resp_ee_0917.py", "mucap_el_run_0916.py", "pmnist_0905.py",
                         "pmnist_rlmnist_0906.py", "elu_growth_0909.py", "shell_l2_rlmnist_0913.py")},
        "data_sha256": mnist.sha256, "prefix_info": info,
        "seconds_prefix": t_prefix, "seconds_total": time.time() - t0,
        "peak_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    (out / "provenance.json").write_text(json.dumps(prov, indent=2, default=str))
    if progress:
        print(f"wrote {out}  ({len(prows)} prefix rows, {len(rows)} arm rows, {time.time() - t0:.1f}s)",
              flush=True)
    return prov


def _flush_is_on() -> bool:
    return float(torch.tensor([1e-30], dtype=torch.float32) * torch.tensor([1e-10], dtype=torch.float32)) == 0.0


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--arms", default=None, help="comma list (checks and S-cost only)")
    ap.add_argument("--prefix-tasks", type=int, default=PREFIX_T)
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)
    torch.set_flush_denormal(True)
    H.setup("cpu")
    mnist = H.Mnist(torch.device("cpu"))
    arms = args.arms.split(",") if args.arms else None
    run_seed(args.seed, Path(args.out), mnist, args.epochs, arms, args.prefix_tasks)


if __name__ == "__main__":
    main()
