#!/usr/bin/env python3
"""baselines_cifar5p1_1008 -- CBP, Shrink & Perturb and ReDo on the 5+1 CIFAR x MLP box.

    python -m src.baselines_cifar5p1_1008 run --method none --seeds 0-9
    python -m src.baselines_cifar5p1_1008 run --method snp --eps 1e-4 --sigma 1e-2 --seeds 100-109
    python -m src.baselines_cifar5p1_1008 run --method cbp --rho 1e-4 --seeds 100-109
    python -m src.baselines_cifar5p1_1008 run --method redo --tau 0 --period 1560 --seeds 100-109
    python -m src.baselines_cifar5p1_1008 run --method cbp --rho 1e-3 --seeds 100-102 --tasks 3 --out /tmp/x

Spec: specs/spec_baselines_cifar5p1_1008.md.  The box is the host's (src/cifar5p1_mlp_0920.py:
arm R, cond std, Adam lr 1e-4, 30 tasks of 780 updates, R = 10 seeds stacked, one step captured
as a CUDA graph, fresh-network control) and nothing in it is modified: data, task plan, batches,
init, the stacked forward, the evaluation and the CSV writer are imported.  Only the host's
`run()` is copied (as in cap_cifar5p1_1007), because CBP needs two in-graph additions, and the
copy keeps the host's arithmetic for R/std line for line; S-nochange compares the `none` arm with
the host's own `run()` byte for byte.

The three methods (spec 2.3-2.5), all on both hidden layers:

  snp   after every Adam update, every tensor p <- (1-eps) p + sigma*zeta, zeta ~ U(+-1/sqrt(fan_in))
        (the init distribution; Kumar et al. 2024 App. A.3), one CUDA generator per process.
  cbp   Dohare et al. 2024 (GnT accumulate=True + AdamGnT): age += 1 and the contribution utility
        u <- eta u + (1-eta) mean|W_next[:, i]| mean_b|h_i| inside the graph; when the exact
        rational accumulator rho * n_eligible crosses an integer, the eligible (age > m) units of
        smallest u / (1 - eta^age) get a fresh U(+-1/sqrt(fan_in)) input row, a zero bias, zero
        output weights, zero Adam moments and a per-element Adam step count of 0.
  redo  Sokar et al. 2023 (Dopamine NeuronRecycler): every F updates, units whose layer-normalised
        mean |h| over the last 64 training images is <= tau get a fresh input row and a zero bias,
        then zero output weights; the Adam moments of those weights are zeroed (bias moments and
        the global step count are kept).  A check on a task's last update runs after that task's
        evaluation.

`none` contains no method arithmetic at all: it is the host's R.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import pmnist_0905 as H                    # streams, setup, csv writer
from src import rlcifar_mlp_battle_0918 as B        # stacked forward, git_state, parse_ints
from src import cifar5p1_mlp_0920 as C              # the box (not modified)

EXPERIMENT = "baselines_cifar5p1_1008"
OUT_ROOT = H.REPO / "results" / EXPERIMENT
CALIB_ROOT = OUT_ROOT / "calib"
MAIN_ROOT = OUT_ROOT / "main"
SPEC = "specs/spec_baselines_cifar5p1_1008.md"
CHECKS_JSON = OUT_ROOT / "checks.json"
SELECTED_JSON = CALIB_ROOT / "selected.json"
REQUIRED_CHECKS = ("S-nochange", "S-host-repro", "S-snp", "S-cbp", "S-redo", "S-graph", "S-fresh",
                   "S-select", "S-verdict", "S-CLI")
SOURCES = ("src/baselines_cifar5p1_1008.py", "analysis/baselines_cifar5p1_1008/checks.py",
           "analysis/baselines_cifar5p1_1008/verdict.py")

ACT_ARM, COND, LR = "R", "std", C.LR                 # the host's arm, condition and lr
METHODS = ("none", "snp", "cbp", "redo")
CBP_MATURITY = 100                                   # Kumar A.1.3 / Dohare Alg. 1
CBP_DECAY = 0.99
REDO_PROBE_BATCHES = 2                               # 2 x 32 = Sokar Table 1's 64 images
REDO_SCORE_EPS = 1e-9                                # Dopamine estimate_neuron_score
CALIB_SEEDS = list(range(100, 110))
EVAL_GROUPS = (list(range(0, 10)), list(range(10, 20)))
THREADS = 2

# spec 3.1 / 3.2: the registered grids, in the tie-break order (literature default first)
GRID = {
    "snp": [{"eps": e, "sigma": s} for e, s in (("1e-4", "1e-2"), ("1e-4", "1e-3"), ("1e-4", "1e-4"),
                                                ("1e-5", "1e-2"), ("1e-5", "1e-3"), ("1e-5", "1e-4"),
                                                ("1e-3", "1e-2"), ("1e-3", "1e-3"), ("1e-3", "1e-4"))],
    "cbp": [{"rho": r} for r in ("1e-4", "1e-5", "1e-3")],
    "redo": [{"tau": t, "period": f} for t, f in (("0", 1560), ("0", 780), ("0", 100),
                                                  ("0.025", 1560), ("0.025", 780), ("0.025", 100),
                                                  ("0.1", 1560), ("0.1", 780), ("0.1", 100))],
}
REGISTERED = dict(tasks=C.N_TASKS, lr=LR, steps_hard=C.STEPS_PER_TASK, steps_easy=C.STEPS_PER_TASK,
                  fresh=True, graph=True, threads=THREADS)


# --------------------------------------------------------------------------
# method configuration
# --------------------------------------------------------------------------

def make_cfg(method: str, eps: str | None = None, sigma: str | None = None, rho: str | None = None,
             tau: str | None = None, period: int | None = None) -> dict:
    """The method and its hyperparameters as registered decimal strings (exact for rho's Fraction)."""
    if method not in METHODS:
        raise SystemExit(f"unknown method {method!r}; known: {','.join(METHODS)}")
    need = {"none": (), "snp": ("eps", "sigma"), "cbp": ("rho",), "redo": ("tau", "period")}[method]
    given = {"eps": eps, "sigma": sigma, "rho": rho, "tau": tau, "period": period}
    cfg = {"method": method}
    for k in need:
        if given[k] is None:
            raise SystemExit(f"method {method} needs --{k}")
        cfg[k] = int(given[k]) if k == "period" else str(given[k])
    extra = [k for k, v in given.items() if v is not None and k not in need]
    if extra:
        raise SystemExit(f"method {method} takes no {extra}")
    if method == "cbp":
        cfg.update(maturity=CBP_MATURITY, decay=CBP_DECAY)
    return cfg


def cfg_tag(cfg: dict) -> str:
    m = cfg["method"]
    if m == "snp":
        return f"snp_e{cfg['eps']}_s{cfg['sigma']}"
    if m == "cbp":
        return f"cbp_r{cfg['rho']}"
    if m == "redo":
        return f"redo_t{cfg['tau']}_f{cfg['period']}"
    return "none"


def hyper(cfg: dict) -> dict:
    """The registered hyperparameters only (what GRID lists)."""
    return {k: v for k, v in cfg.items() if k in ("eps", "sigma", "rho", "tau", "period")}


def in_grid(cfg: dict) -> bool:
    return cfg["method"] == "none" or hyper(cfg) in GRID[cfg["method"]]


def seeds_tag(seeds: list[int]) -> str:
    return f"s{seeds[0]}-{seeds[-1]}" if seeds == list(range(seeds[0], seeds[-1] + 1)) else \
        "s" + "_".join(map(str, seeds))


def snp_seed(seeds: list[int]) -> int:
    h = hashlib.sha256(f"{EXPERIMENT}|snp_noise|{','.join(map(str, seeds))}".encode()).digest()
    return int.from_bytes(h[:8], "little") & ((1 << 63) - 1)


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# --------------------------------------------------------------------------
# CBP's exact replacement-count ledger (spec 2.4; GnT accumulate=True in rationals)
# --------------------------------------------------------------------------

class CbpLedger:
    """Per layer: n_eligible(t) and the integer number of replacements, identical for every slot.

    A unit replaced after update s has age t - s at update t and is eligible iff t - s > m; a unit
    never replaced has age t.  Every replacement takes an eligible unit, so
    n_elig(t) = 0 if t <= m else width - (replacements made after updates t-m .. t-1)."""

    def __init__(self, rho: str, width: int, maturity: int, n_layers: int = 2):
        self.rho = Fraction(rho)
        self.width, self.m = width, maturity
        self.reset(n_layers)

    def reset(self, n_layers: int = 2) -> None:
        self.acc = [Fraction(0)] * n_layers
        self.hist = [dict() for _ in range(n_layers)]     # update -> count replaced after it

    def n_elig(self, l: int, t: int) -> int:
        if t <= self.m:
            return 0
        recent = sum(c for s, c in self.hist[l].items() if t - self.m <= s <= t - 1)
        return self.width - recent

    def count(self, l: int, t: int) -> tuple[int, int]:
        """(k, n_elig) for update t: acc += rho * n_elig; k = floor(acc); acc -= k."""
        ne = self.n_elig(l, t)
        self.acc[l] += self.rho * ne
        k = math.floor(self.acc[l])
        self.acc[l] -= k
        if k > ne:
            raise RuntimeError(f"CBP wants {k} replacements but only {ne} units are eligible")
        if k:
            self.hist[l][t] = k
            for s in [s for s in self.hist[l] if s < t - self.m]:     # keep the window short
                del self.hist[l][s]
        return k, ne


# --------------------------------------------------------------------------
# task-end extras (spec 2.7)
# --------------------------------------------------------------------------

@torch.no_grad()
def extra_stats(P, Xt: torch.Tensor, act) -> dict:
    """Relative position z_bar / sd per unit (median over units with sd > 0), norms."""
    z1 = torch.baddbmm(P[1][:, None, :], Xt, P[0].transpose(1, 2))
    a1 = act.phi(z1, 0, False)
    z2 = torch.baddbmm(P[3][:, None, :], a1, P[2].transpose(1, 2))
    out = {}
    for li, z in ((1, z1), (2, z2)):
        zbar = z.mean(1).double().cpu()
        zsd = z.std(1).double().cpu()
        rel, excl = [], []
        for r in range(z.shape[0]):
            ok = zsd[r] > 0
            excl.append(int((~ok).sum()))
            v = zbar[r][ok] / zsd[r][ok]
            rel.append(float(v.median()) if len(v) else float("nan"))
        out[f"relpos_l{li}"], out[f"relpos_excl_l{li}"] = rel, excl
    for li in range(3):
        out[f"w_fro_l{li + 1}"] = torch.linalg.vector_norm(P[2 * li].double(), dim=(1, 2)).cpu().tolist()
    for li in range(2):
        out[f"b_absmean_l{li + 1}"] = P[2 * li + 1].double().abs().mean(1).cpu().tolist()
    return out


@torch.no_grad()
def dead_frac_l2(P, Xt: torch.Tensor, act) -> list[float]:
    """The host's dead_frac_l2 (max over the task's images of |phi'(z2)| < DEAD_TOL), per slot."""
    z1, a1, z2, a2, _ = B.forward(P, Xt, act, train=False)
    d = act.dphi(z2, 1)
    return [float((d[r].abs().amax(0) < C.DEAD_TOL).float().mean()) for r in range(Xt.shape[0])]


# --------------------------------------------------------------------------
# the stacked run: the host's run() for R/std, plus the method
# --------------------------------------------------------------------------

def run(cfg: dict, seeds: list[int], n_tasks: int, device, out: Path, lr: float = LR,
        cifar: C.Cifar100 | None = None, fresh: bool = True, graph: bool = True,
        progress=None, debug: dict | None = None, observer=None,
        steps_hard: int = C.STEPS_PER_TASK, steps_easy: int = C.STEPS_PER_TASK,
        final: dict | None = None) -> dict:
    """Train one method arm's R = len(seeds) runs in lockstep over the 30-task sequence.

    `debug` (checks only) makes the steps eager; `debug["hook"]`, if given, is called as
    hook(event, **state) at "post_adam" (inside the step, after the Adam update), "post_step"
    (after the step), "post_method" (after the eager method write), "task_end" (after the
    evaluation, before a boundary ReDo recycle), "boundary_post", "fresh_start" and
    "fresh_self_start".  `observer(t, state)` is called at every task end (diagnostics; the graph
    is kept).  Neither writes to the training state.
    """
    t_start = time.time()
    progress = progress or (lambda m: print(m, flush=True))
    method = cfg["method"]
    if method not in METHODS:
        raise SystemExit(f"unknown method {method!r}")
    is_snp, is_cbp, is_redo = method == "snp", method == "cbp", method == "redo"
    tag = cfg_tag(cfg)
    arm, cond = ACT_ARM, COND
    act = C.make_act(arm)
    R = len(seeds)
    cifar = cifar or C.Cifar100()
    X_all = cifar.inputs("train", cond, device)
    Y_all = cifar.train_y.to(device)
    X_test = cifar.inputs("test", cond, device)
    Y_test = cifar.test_y.to(device)
    tr_rows, te_rows = C.class_rows(cifar.train_y), C.class_rows(cifar.test_y)
    plans = [C.task_plan(s) for s in seeds]
    g_batch = {s: H.stream("c51_batch", s) for s in seeds}

    init = [q.detach() for s in seeds for q in C.init_params(arm, s, device)]
    P = [torch.stack(init[i::6]).contiguous() for i in range(6)]
    P = [q.requires_grad_(True) for q in P]
    P0 = [q.detach().clone() for q in P]
    act.init_state(R, device, key=f"{arm}|{seeds}|{cond}")
    act0 = act.state()
    adam_m = [torch.zeros_like(q) for q in P]
    adam_v = [torch.zeros_like(q) for q in P]
    hook = None if debug is None else debug.get("hook")

    fan_in = [C.layer_shapes(arm)[i // 2][1] for i in range(6)]           # 3072,3072,100,100,100,100
    bound = [1.0 / math.sqrt(f) for f in fan_in]                           # init_params' bound
    width = C.HIDDEN

    # ---- method state (static tensors the captured step reads by address)
    snp_eps = float(cfg["eps"]) if is_snp else 0.0
    snp_sigma = float(cfg["sigma"]) if is_snp else 0.0
    g_snp = None
    if is_snp:
        g_snp = torch.Generator(device=device)
        g_snp.manual_seed(snp_seed(seeds))
    cbp_eta = CBP_DECAY
    cbp_age = [torch.zeros(R, width, device=device) for _ in range(2)]
    cbp_u = [torch.zeros(R, width, device=device) for _ in range(2)]
    cbp_L = [torch.zeros(R, width, dtype=torch.long, device=device) for _ in range(2)]
    tc_dev = torch.zeros((), dtype=torch.long, device=device)
    ledger = CbpLedger(cfg["rho"], width, CBP_MATURITY) if is_cbp else None
    g_reinit = {s: H.stream(f"b5_{method}_reinit", s) for s in seeds} if (is_cbp or is_redo) else {}
    redo_tau = float(cfg["tau"]) if is_redo else 0.0
    redo_F = int(cfg["period"]) if is_redo else 0
    n_reset = torch.zeros(R, 2, dtype=torch.long)                         # per task, cpu
    totals = {"reset_l1": 0, "reset_l2": 0, "redo_checks": 0}
    redo_rows: list[dict] = []

    # ---- one training step on static tensors (the host's, plus CBP's in-graph part)
    b1, b2, eps = 0.9, 0.999, 1e-8
    static_idx = torch.zeros(R, C.BATCH, dtype=torch.long, device=device)
    inv_c1 = torch.zeros((), device=device)
    inv_c2 = torch.zeros((), device=device)
    step_t = torch.zeros((), dtype=torch.long, device=device)
    acc_sum = torch.zeros(R, device=device)
    bad_step = torch.full((R,), -1, dtype=torch.long, device=device)
    last_hit = torch.zeros(R, device=device)
    b1_t = torch.tensor(b1, dtype=torch.float64, device=device)
    b2_t = torch.tensor(b2, dtype=torch.float64, device=device)
    cur = {"tc": 0, "t": 0, "j": 0, "phase": ""}                     # hook context (eager only)

    post = getattr(act, "post_update", None)

    def cbp_factors():
        """Per-tensor bias corrections: the host's scalar where an element was never reset,
        1/(1 - beta^s) with s = t - L (the element's own Adam step count, AdamGnT) where it was."""
        L1, L2 = cbp_L

        def fac(L):
            s = (tc_dev - L).to(torch.float64)                  # float64 like the host's scalar,
            f1 = (1.0 / (1.0 - torch.pow(b1_t, s))).to(torch.float32)   # then one rounding
            f2 = (1.0 / (1.0 - torch.pow(b2_t, s))).to(torch.float32)
            return torch.where(L == 0, inv_c1, f1), torch.where(L == 0, inv_c2, f2)

        Ls = [L1[:, :, None], L1, torch.maximum(L2[:, :, None], L1[:, None, :]), L2, L2[:, None, :]]
        fs = [fac(L) for L in Ls] + [(inv_c1, inv_c2)]
        return [f[0] for f in fs], [f[1] for f in fs]

    def step():
        xb, yb = X_all[static_idx], Y_all[static_idx]            # gathers, no arithmetic
        z1, a1, z2, a2, z3 = B.forward(P, xb, act, train=True)
        lossv = F.cross_entropy(z3.reshape(-1, C.N_CLASSES), yb.reshape(-1),
                                reduction="none").view(R, C.BATCH).mean(1)
        hit = (z3.detach().argmax(-1) == yb).float().mean(1)     # pre-update: the paper's a_j
        acc_sum.add_(hit)
        last_hit.copy_(hit)
        grads = torch.autograd.grad(lossv.sum(), P)
        with torch.no_grad():
            bad = ~torch.isfinite(lossv)
            bad_step.copy_(torch.where((bad_step < 0) & bad, step_t, bad_step))
            step_t.add_(1)
            if is_cbp:
                c1s, c2s = cbp_factors()
                for p, gr, mi, vi, f1, f2 in zip(P, grads, adam_m, adam_v, c1s, c2s):
                    mi.mul_(b1).add_(gr, alpha=1 - b1)
                    vi.mul_(b2).addcmul_(gr, gr, value=1 - b2)
                    p.sub_(lr * (mi * f1) / ((vi * f2).sqrt() + eps))
            else:
                for p, gr, mi, vi in zip(P, grads, adam_m, adam_v):
                    mi.mul_(b1).add_(gr, alpha=1 - b1)
                    vi.mul_(b2).addcmul_(gr, gr, value=1 - b2)
                    p.sub_(lr * (mi * inv_c1) / ((vi * inv_c2).sqrt() + eps))
            act.update(z1.detach(), z2.detach())
            if post is not None:
                post(P, lr)
            if hook is not None:
                hook("post_adam", P=P, m=adam_m, v=adam_v, a1=a1.detach(), a2=a2.detach(),
                     age=cbp_age, u=cbp_u, L=cbp_L, inv_c1=inv_c1, inv_c2=inv_c2, **cur)
            if is_cbp:                                           # spec 2.4: GnT order
                for l, (h, Wn) in enumerate(((a1, P[2]), (a2, P[4]))):
                    cbp_age[l].add_(1)
                    c = Wn.abs().mean(dim=1) * h.detach().abs().mean(dim=1)
                    cbp_u[l].mul_(cbp_eta).add_(c, alpha=1 - cbp_eta)

    # ---- the eager method writes (outside the graph)
    @torch.no_grad()
    def snp_apply():
        for p, bd in zip(P, bound):
            z = (torch.rand(p.shape, generator=g_snp, device=device) * 2 - 1) * bd
            p.mul_(1 - snp_eps).add_(z, alpha=snp_sigma)

    def fresh_row(r: int, l: int, n: int) -> torch.Tensor:
        """n fresh input rows for layer l of slot r from the slot's own cpu stream (init_params' form)."""
        fi = fan_in[2 * l]
        return ((torch.rand((n, fi), generator=g_reinit[seeds[r]], dtype=torch.float32) * 2 - 1)
                * bound[2 * l]).to(device)

    @torch.no_grad()
    def cbp_apply(t: int) -> list[int]:
        ks = [ledger.count(l, t)[0] for l in range(2)]
        if not any(ks):
            return ks
        chosen = []
        for l in range(2):                                       # select from one state (GnT)
            if not ks[l]:
                chosen.append(None)
                continue
            age = cbp_age[l]
            uhat = (cbp_u[l] / (1.0 - torch.pow(torch.tensor(cbp_eta, device=device), age))).cpu().numpy()
            elig = (age > CBP_MATURITY).cpu().numpy()
            sel = []
            for r in range(R):
                cand = np.flatnonzero(elig[r])
                if len(cand) < ks[l]:
                    raise RuntimeError(f"slot {r} layer {l + 1}: {len(cand)} eligible < {ks[l]}")
                order = cand[np.argsort(uhat[r, cand], kind="stable")]
                sel.append(np.sort(order[:ks[l]]))
            chosen.append(sel)
        for l in range(2):                                       # replace: layer 1, then layer 2
            if chosen[l] is None:
                continue
            W, b, Wn = P[2 * l], P[2 * l + 1], P[2 * l + 2]
            for r in range(R):
                idx = torch.as_tensor(chosen[l][r], dtype=torch.long, device=device)
                W[r, idx, :] = fresh_row(r, l, len(idx))
                b[r, idx] = 0.0
                Wn[r, :, idx] = 0.0
                cbp_u[l][r, idx] = 0.0
                cbp_age[l][r, idx] = 0.0
                for q in (adam_m, adam_v):
                    q[2 * l][r, idx, :] = 0.0
                    q[2 * l + 1][r, idx] = 0.0
                    q[2 * l + 2][r, :, idx] = 0.0
                cbp_L[l][r, idx] = t
                n_reset[r, l] += len(idx)
        return ks

    @torch.no_grad()
    def redo_apply(idx_probe: torch.Tensor, n: int, t: int, j: int, phase: str) -> None:
        """Sokar Alg. 1 on the last 64 training images (idx_probe: (R, 64) global rows)."""
        _, a1, _, a2, _ = B.forward(P, X_all[idx_probe], act, train=False)
        masks = []
        for h in (a1, a2):
            s = h.abs().mean(dim=1)
            s = s / (s.mean(dim=1, keepdim=True) + REDO_SCORE_EPS)
            masks.append((s <= redo_tau).cpu())
        units = [[torch.nonzero(masks[l][r]).flatten() for r in range(R)] for l in range(2)]
        for l in range(2):                                       # incoming side, both layers
            W, b = P[2 * l], P[2 * l + 1]
            for r in range(R):
                idx = units[l][r].to(device)
                if len(idx):
                    W[r, idx, :] = fresh_row(r, l, len(idx))
                    b[r, idx] = 0.0
                    adam_m[2 * l][r, idx, :] = 0.0
                    adam_v[2 * l][r, idx, :] = 0.0
        for l in range(2):                                       # then the outgoing side
            Wn = P[2 * l + 2]
            for r in range(R):
                idx = units[l][r].to(device)
                if len(idx):
                    Wn[r, :, idx] = 0.0
                    adam_m[2 * l + 2][r, :, idx] = 0.0
                    adam_v[2 * l + 2][r, :, idx] = 0.0
        for r in range(R):
            k1, k2 = len(units[0][r]), len(units[1][r])
            n_reset[r, 0] += k1
            n_reset[r, 1] += k2
            redo_rows.append({"phase": phase, "seed": seeds[r], "slot": r, "n": n, "task": t,
                              "step_in_task": j + 1, "n_dormant_l1": k1, "n_dormant_l2": k2})

    use_graph = graph and device.type == "cuda" and debug is None
    cg = None
    mutable = (*P, *adam_m, *adam_v, acc_sum, bad_step, step_t, last_hit, *cbp_age, *cbp_u)
    if use_graph:
        keep = [q.detach().clone() for q in mutable]
        keep_act = act.state()
        inv_c1.fill_(1.0); inv_c2.fill_(1.0)
        tc_dev.fill_(1)
        static_idx.zero_()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                act.begin_step(R, C.BATCH, device)
                step()
        torch.cuda.current_stream().wait_stream(side)
        cg = torch.cuda.CUDAGraph()
        act.begin_step(R, C.BATCH, device)
        with torch.cuda.graph(cg):
            step()
        with torch.no_grad():
            for q, v in zip(mutable, keep):
                q.copy_(v)
        act.load_state(keep_act)
        del keep

    def state(**kw):
        return dict(P=P, m=adam_m, v=adam_v, age=cbp_age, u=cbp_u, L=cbp_L, ledger=ledger,
                    n_reset=n_reset, g_snp=g_snp, **kw)

    def train_task(batches, tc0: int, method_on: bool, phase: str, t: int) -> tuple[int, bool]:
        """`batches.shape[1]` steps on (R, steps, 32) global row indices.  Returns (tc, pending):
        pending = a ReDo check falls on the last update and waits for the task-end evaluation."""
        tc = tc0
        acc_sum.zero_()
        bad_step.fill_(-1)
        step_t.zero_()
        steps = batches.shape[1]
        pending = False
        for j in range(steps):
            static_idx.copy_(batches[:, j])
            tc += 1
            inv_c1.fill_(1.0 / (1 - b1 ** tc))
            inv_c2.fill_(1.0 / (1 - b2 ** tc))
            if is_cbp:
                tc_dev.fill_(tc)
            if hook is not None:
                cur.update(tc=tc, t=t, j=j, phase=phase)
            act.begin_step(R, C.BATCH, device)
            if cg is not None:
                cg.replay()
            else:
                step()
            if hook is not None:
                hook("post_step", **state(t=t, j=j, tc=tc, phase=phase, idx=batches[:, j]))
            if not method_on:
                continue
            wrote = False
            if is_snp:
                snp_apply()
                wrote = True
            elif is_cbp:
                wrote = any(cbp_apply(tc))
            elif is_redo and tc % redo_F == 0:
                if j == steps - 1:
                    pending = phase == "train"                   # fresh_self: nothing follows
                else:
                    if j < 1:
                        raise RuntimeError("a ReDo check on a task's first update has no 64-image probe")
                    redo_apply(batches[:, j - 1:j + 1].reshape(R, -1), tc, t, j, phase)
                    wrote = True
            if hook is not None and wrote:
                hook("post_method", **state(t=t, j=j, tc=tc, phase=phase, idx=batches[:, j]))
        if device.type == "cuda":
            torch.cuda.synchronize()
        return tc, pending

    alive = torch.ones(R, dtype=torch.bool, device=device)
    rows, ex_rows, diverged, tc = [], [], [], 0
    last_hard = max(t for t in range(1, n_tasks + 1) if t % 2 == 1)
    saved = {}
    out.mkdir(parents=True, exist_ok=True)
    step_ms = float("nan")

    for t in range(1, n_tasks + 1):
        hard = t % 2 == 1
        steps = steps_hard if hard else steps_easy
        rows_t = [torch.cat([tr_rows[q] for q in plans[r][t - 1][1]]) for r in range(R)]
        batches = torch.stack([C.batch_indices(g_batch[s], rows_t[r], steps)
                               for r, s in enumerate(seeds)]).to(device)
        if t == last_hard and fresh:
            saved[t] = batches.clone()
        n_reset.zero_()
        t0 = time.time()
        tc, pending = train_task(batches, tc, method != "none", "train", t)
        step_ms = 1e3 * (time.time() - t0) / steps

        with torch.no_grad():
            finite = torch.stack([torch.isfinite(q).flatten(1).all(1) for q in P]).all(0)
            newly = alive & ((bad_step >= 0) | ~finite)
            for r in torch.nonzero(newly).flatten().tolist():
                diverged.append({"diverged": True, "task": t, "seed": seeds[r], "arm": arm,
                                 "method": tag, "cond": cond,
                                 "step": (t - 1) * C.STEPS_PER_TASK + max(int(bad_step[r]), 0)})
                rows.append({"arm": arm, "cond": cond, "seed": seeds[r], "slot": r, "lr": lr,
                             "task": t, "hard": int(hard), "acc": float("nan")})
            alive &= ~newly
            live = torch.nonzero(alive).flatten().tolist()
            Xt = torch.stack([X_all[rows_t[r]] for r in range(R)])
            Yt = torch.stack([Y_all[rows_t[r]] for r in range(R)])
            m = C.evaluate(P, Xt, Yt, act, live)
            ex = extra_stats(P, Xt, act)
            if observer is not None:
                observer(t, dict(P=P, m=adam_m, v=adam_v, Xt=Xt, act=act, live=live))
            te = [torch.cat([te_rows[q] for q in plans[r][t - 1][1]]) for r in range(R)]
            test_acc = C.accuracy(P, torch.stack([X_test[q] for q in te]),
                                  torch.stack([Y_test[q] for q in te]), act)
            if hook is not None:
                hook("task_end", **state(t=t, Xt=Xt, rows=m, pending=pending))
            dead_post = [float("nan")] * R
            if pending:                                          # spec 2.5: after the evaluation
                redo_apply(batches[:, steps - 2:].reshape(R, -1), tc, t, steps - 1, "train")
                dead_post = dead_frac_l2(P, Xt, act)
                if hook is not None:
                    hook("boundary_post", **state(t=t, Xt=Xt))
            del Xt
        for r in live:
            rows.append({"arm": arm, "cond": cond, "seed": seeds[r], "slot": r, "lr": lr,
                         "task": t, "hard": int(hard), "n_classes": len(plans[r][t - 1][1]),
                         "online_acc": float(acc_sum[r]) / steps,
                         "train_acc": m[r]["acc"], "test_acc": float(test_acc[r]), **m[r]})
            er = {"method": method, "config": tag, "seed": seeds[r], "slot": r, "task": t,
                  "hard": int(hard)}
            for k in ("relpos_l1", "relpos_l2", "relpos_excl_l1", "relpos_excl_l2"):
                er[k] = ex[k][r]
            er["zbar_over_zsd_l2"] = (m[r]["zbar_l2"] / m[r]["zsd_l2"]) if m[r]["zsd_l2"] != 0 else float("nan")
            er["n_reset_l1"], er["n_reset_l2"] = int(n_reset[r, 0]), int(n_reset[r, 1])
            er["dead_post_l2"] = dead_post[r]
            for k in ("w_fro_l1", "w_fro_l2", "w_fro_l3", "b_absmean_l1", "b_absmean_l2"):
                er[k] = ex[k][r]
            ex_rows.append(er)
        totals["reset_l1"] += int(n_reset[:, 0].sum())
        totals["reset_l2"] += int(n_reset[:, 1].sum())
        rows.sort(key=lambda q: (q["slot"], q["task"]))
        ex_rows.sort(key=lambda q: (q["slot"], q["task"]))
        H.write_csv(out / "per_task.csv", rows)
        H.write_csv(out / "extra_task.csv", ex_rows)
        if is_redo:
            H.write_csv(out / "redo_checks.csv", redo_rows)
        on = acc_sum[alive] / steps
        el = time.time() - t_start
        progress(f"[{time.strftime('%T')}] {tag} task {t:2d}/{n_tasks} "
                 f"{'hard' if hard else 'easy'} alive {int(alive.sum())}/{R} "
                 f"online {float(on.mean()) if len(on) else float('nan'):.3f} "
                 f"reset {int(n_reset[:, 0].sum())}/{int(n_reset[:, 1].sum())}  "
                 f"{step_ms:.2f} ms/step  {el / 60:.1f} min")

    # ---- fresh controls on the last hard task (spec 2.6)
    fresh_rows, self_rows = [], []
    if fresh and last_hard in saved:
        continual = {int(q["seed"]): q["online_acc"] for q in rows
                     if q["task"] == last_hard and "online_acc" in q}

        def reset_net() -> None:
            with torch.no_grad():
                for q, v in zip(P, P0):
                    q.copy_(v)
                for q in (*adam_m, *adam_v):
                    q.zero_()
                for l in range(2):
                    cbp_age[l].zero_()
                    cbp_u[l].zero_()
                    cbp_L[l].zero_()
            if ledger is not None:
                ledger.reset()
            act.load_state(act0)

        # (a) the host's procedure, no method: the same network in every arm
        reset_net()
        if hook is not None:
            hook("fresh_start", **state())
        alive_f = torch.ones(R, dtype=torch.bool, device=device)
        train_task(saved[last_hard], 0, False, "fresh", last_hard)
        with torch.no_grad():
            finite = torch.stack([torch.isfinite(q).flatten(1).all(1) for q in P]).all(0)
            alive_f &= (bad_step < 0) & finite
        for r in range(R):
            if not bool(alive_f[r]) or seeds[r] not in continual:
                continue
            value = float(acc_sum[r]) / steps_hard
            fresh_rows.append({"arm": arm, "cond": cond, "seed": seeds[r], "lr": lr,
                               "task": last_hard, "fresh_online_acc": value,
                               "continual_online_acc": continual[seeds[r]],
                               "fresh_gap": value - continual[seeds[r]]})
        H.write_csv(out / "fresh_control.csv", fresh_rows)
        gaps = [q["fresh_gap"] for q in fresh_rows]
        progress(f"[{time.strftime('%T')}] {tag} fresh control (no method) on task {last_hard}: "
                 f"gap median {float(np.median(gaps)) if gaps else float('nan'):+.3f} "
                 f"({sum(q > 0 for q in gaps)}/{len(gaps)} positive)")
        # (b) the method kept on (REPORT_ONLY)
        if method != "none":
            reset_net()
            if hook is not None:
                hook("fresh_self_start", **state())
            alive_s = torch.ones(R, dtype=torch.bool, device=device)
            n_reset.zero_()
            train_task(saved[last_hard], 0, True, "fresh_self", last_hard)
            with torch.no_grad():
                finite = torch.stack([torch.isfinite(q).flatten(1).all(1) for q in P]).all(0)
                alive_s &= (bad_step < 0) & finite
            for r in range(R):
                if not bool(alive_s[r]) or seeds[r] not in continual:
                    continue
                value = float(acc_sum[r]) / steps_hard
                self_rows.append({"method": method, "config": tag, "seed": seeds[r],
                                  "task": last_hard, "fresh_self_online_acc": value,
                                  "continual_online_acc": continual[seeds[r]],
                                  "gap_self": value - continual[seeds[r]],
                                  "n_reset_l1": int(n_reset[r, 0]), "n_reset_l2": int(n_reset[r, 1])})
            H.write_csv(out / "fresh_self.csv", self_rows)
            if is_redo:
                H.write_csv(out / "redo_checks.csv", redo_rows)

    prov = {"run_id": EXPERIMENT, **B.git_state(), "method": method, "config": tag,
            "method_cfg": cfg,
            "method_rule": {
                "none": "the host's R step, no method arithmetic",
                "snp": "after every Adam update, every tensor: p.mul_(1-eps).add_(zeta, alpha=sigma), "
                       "zeta = (rand*2-1)/sqrt(fan_in), one CUDA generator per process",
                "cbp": "GnT accumulate=True with contribution utility, maturity 100, decay 0.99, "
                       "U(+-1/sqrt(fan_in)) input rows, zero bias and output weights, AdamGnT moments "
                       "and per-element step reset",
                "redo": "every F updates, layer-normalised mean|h| over the last 64 training images "
                        "<= tau: fresh input row, zero bias, zero output weights, Adam m/v of those "
                        "weights zeroed (bias moments and step count kept); task-end checks after "
                        "the evaluation"}[method],
            "rng": {"snp_noise_cuda_seed": snp_seed(seeds) if is_snp else None,
                    "reinit_stream": f"H.stream('b5_{method}_reinit', seed)" if (is_cbp or is_redo) else None},
            "resets_total": totals, "redo_checks_rows": len(redo_rows) if is_redo else None,
            "arm": arm, "cond": cond, "seeds": seeds, "R": R, "lr": lr, "n_tasks": n_tasks,
            "steps_per_task": C.STEPS_PER_TASK, "steps_hard": steps_hard,
            "steps_easy": steps_easy, "batch": C.BATCH, "hidden": C.HIDDEN,
            "layer_shapes": C.layer_shapes(arm), "n_params": C.n_params(arm),
            "dims": list(C.DIMS), "n_classes": C.N_CLASSES, "hard_classes": C.HARD_CLASSES,
            "per_class_train": C.PER_CLASS_TRAIN, "classes_used": C.CLASSES_USED,
            "task_parity": "task 1 hard, alternating (the paper's task 0 hard)",
            "optimizer": "adam", "weight_decay": 0.0, "data_sha256": cifar.sha256,
            "std": {"mean": C.STD_MEAN, "std": C.STD_STD, "planes": "R,G,B x 1024"},
            "class_sha256": {str(s): hashlib.sha256(
                np.asarray(C.seed_classes(s), dtype=np.int64).tobytes()).hexdigest() for s in seeds},
            "rng_roles": ["c51_classes", "c51_batch", "init"]
                         + ([f"b5_{method}_reinit"] if (is_cbp or is_redo) else [])
                         + (["snp_noise(cuda)"] if is_snp else []),
            "fresh_control": {"task": last_hard, "n": len(fresh_rows),
                              "rule": "host procedure, method off (spec 2.6)"} if fresh else None,
            "fresh_self": {"n": len(self_rows), "rule": "method on, method state reset"}
                          if (fresh and method != "none") else None,
            "engine": "host run() of cifar5p1_mlp_0920 copied for R/std + method; stacked baddbmm, "
                      "autograd, elementwise Adam"
                      + (", one step captured as a CUDA graph" if use_graph else ", eager"),
            "host_module_sha256": sha256_file(Path(C.__file__)),
            "engine_sha256": sha256_file(Path(__file__)),
            "device": str(device), "torch": torch.__version__, "threads": torch.get_num_threads(),
            "cublas_workspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "step_ms_last_task": step_ms, "wall_clock_s": time.time() - t_start,
            "divergences": diverged, "spec": SPEC}
    (out / "provenance.json").write_text(json.dumps(prov, indent=2))
    if final is not None:                                       # checks: the end state
        final.update(P=[q.detach().clone() for q in P], m=[q.clone() for q in adam_m],
                     v=[q.clone() for q in adam_v])
    progress(f"wrote {out}/per_task.csv ({len(rows)} rows, {(time.time() - t_start) / 60:.1f} min)")
    return prov


# --------------------------------------------------------------------------
# admission to the registered output (spec 9)
# --------------------------------------------------------------------------

def admit(cfg: dict, run_cfg: dict, out: Path, checks_json: Path = CHECKS_JSON,
          repo: Path = H.REPO) -> dict:
    """Refuse the registered output unless the checks passed on these very sources, the tree is
    clean, the configuration is registered and (main) the selection is committed and matches."""
    bad = [k for k, v in REGISTERED.items() if run_cfg.get(k) != v]
    if bad:
        raise SystemExit(f"ABORT: not the registered configuration: {bad}")
    if not in_grid(cfg):
        raise SystemExit(f"ABORT: {cfg} is not in the registered grid")
    if not checks_json.exists():
        raise SystemExit(f"ABORT: {checks_json} missing")
    ck = json.loads(checks_json.read_text())
    missing = [k for k in REQUIRED_CHECKS if k not in ck]
    if missing or not ck.get("all_pass"):
        raise SystemExit(f"ABORT: checks not all_pass (missing {missing})")
    now = {s: sha256_file(repo / s) for s in SOURCES}
    if ck.get("source_sha256") != now:
        raise SystemExit("ABORT: sources changed since the checks ran")
    paths = ["src", "analysis", "specs"]
    seeds = run_cfg["seeds"]
    if out.resolve().is_relative_to((repo / "results" / EXPERIMENT / "calib").resolve()):
        if seeds != CALIB_SEEDS:
            raise SystemExit(f"ABORT: calibration runs use seeds 100-109 only, not {seeds}")
        sel = None
    elif out.resolve().is_relative_to((repo / "results" / EXPERIMENT / "main").resolve()):
        if seeds not in EVAL_GROUPS:
            raise SystemExit(f"ABORT: main runs use seeds 0-9 or 10-19, not {seeds}")
        sel_path = repo / "results" / EXPERIMENT / "calib" / "selected.json"
        if not sel_path.exists():
            raise SystemExit("ABORT: no selected.json (spec 3.2: select and commit first)")
        paths.append(str(sel_path.relative_to(repo)))
        tracked = subprocess.run(["git", "-C", str(repo), "ls-files", "--error-unmatch",
                                  str(sel_path.relative_to(repo))], capture_output=True).returncode == 0
        if not tracked:
            raise SystemExit("ABORT: selected.json is not committed")
        sel = json.loads(sel_path.read_text())
        if cfg["method"] != "none" and hyper(sel["selected"][cfg["method"]]) != hyper(cfg):
            raise SystemExit(f"ABORT: {cfg} is not the selected {sel['selected'][cfg['method']]}")
    else:
        raise SystemExit(f"ABORT: {out} is neither calib/ nor main/")
    dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--", *paths],
                           capture_output=True, text=True).stdout.strip()
    if dirty:
        raise SystemExit(f"ABORT: uncommitted code, spec or selection:\n{dirty}")
    return {"checks_sha256": sha256_file(checks_json), "source_sha256": now,
            "selected_sha256": (sha256_file(repo / "results" / EXPERIMENT / "calib" / "selected.json")
                                if sel is not None else None)}


def default_out(cfg: dict, seeds: list[int]) -> Path | None:
    if seeds == CALIB_SEEDS:
        return CALIB_ROOT / cfg_tag(cfg)
    if seeds in EVAL_GROUPS:
        return MAIN_ROOT / f"{cfg['method']}_{seeds_tag(seeds)}"
    return None


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run"])
    ap.add_argument("--method", required=True, choices=list(METHODS))
    ap.add_argument("--eps", default=None, help="S&P shrink per update (decimal string)")
    ap.add_argument("--sigma", default=None, help="S&P noise scale (decimal string)")
    ap.add_argument("--rho", default=None, help="CBP replacement rate (decimal string)")
    ap.add_argument("--tau", default=None, help="ReDo dormancy threshold (decimal string)")
    ap.add_argument("--period", type=int, default=None, help="ReDo period F in updates")
    ap.add_argument("--seeds", default="100-109")
    ap.add_argument("--tasks", type=int, default=C.N_TASKS)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--steps-hard", type=int, default=C.STEPS_PER_TASK)
    ap.add_argument("--steps-easy", type=int, default=C.STEPS_PER_TASK)
    ap.add_argument("--no-fresh", action="store_true")
    ap.add_argument("--no-graph", action="store_true", help="eager steps (checks)")
    ap.add_argument("--out", default=None, help="default: results/<run>/calib/<tag> or main/<method>_s<a>-<b>")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--threads", type=int, default=THREADS)
    return ap


def main(argv: list[str] | None = None) -> None:
    a = build_parser().parse_args(argv)
    torch.set_num_threads(a.threads)
    cfg = make_cfg(a.method, a.eps, a.sigma, a.rho, a.tau, a.period)
    seeds = B.parse_ints(a.seeds)
    out = Path(a.out) if a.out else default_out(cfg, seeds)
    if out is None:
        raise SystemExit("--out is required for seeds outside the registered groups")
    run_cfg = dict(seeds=seeds, tasks=a.tasks, lr=a.lr, steps_hard=a.steps_hard,
                   steps_easy=a.steps_easy, fresh=not a.no_fresh, graph=not a.no_graph,
                   threads=torch.get_num_threads())
    admission = None
    if out.resolve().is_relative_to(OUT_ROOT.resolve()):
        admission = admit(cfg, run_cfg, out)
    device = H.setup(a.device)
    prov = run(cfg, seeds, run_cfg["tasks"], device, out, lr=run_cfg["lr"], fresh=run_cfg["fresh"],
               graph=run_cfg["graph"], steps_hard=run_cfg["steps_hard"],
               steps_easy=run_cfg["steps_easy"])
    if admission is not None:
        prov["admission"] = admission
        (out / "provenance.json").write_text(json.dumps(prov, indent=2))


if __name__ == "__main__":
    main()
