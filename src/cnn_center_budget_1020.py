#!/usr/bin/env python3
"""cnn_center_budget_1020 -- CH' (per-feature centering) + B_width (bias projection to the current
width) on the real-label 5+1 CIFAR CNN, ReLU and ELU (alpha 3.6), budgets 780 / 7800 per task.

    python3 src/cnn_center_budget_1020.py smoke --out results/cnn_center_budget_1020/smoke
    python3 src/cnn_center_budget_1020.py main --budget 780 --out results/cnn_center_budget_1020/b780

Registered spec: specs/spec_cnn_center_budget_1020.md.

The box, batch lists, probe, engine and per-unit statistics are cifar5p1_cnn_budget_1019's
(imported, not modified).  New here, per run inside the bundle:

  H'        the next layer reads phi(z) - m: at c1 / c2 m is per (channel x position) of the pool
            output (16x16x16, 16x8x8), at f1 / f2 per unit (f2's output feeds the read-out f3);
            m <- (1 - .01) m + .01 * (image mean of this step's pre-update output), after Adam,
            init 0, kept across task boundaries, no gradient (cnn_center_feat_1011's "feat" door).
  B_width   after Adam and the EMA, one extra no_grad forward of the same minibatch; c1 -> c2 ->
            f1 -> f2 in turn, the bias-free q = W * h_in (h_in with the corrected upstream biases),
            s = sqrt(mean_position Var_images(q)) (fc: SD over images), b <- clip(b, -.2 s, +.2 s),
            rounded toward zero when the float32 value would exceed the float64 cap; f3 untouched
            (Codex cnn_chprime_bias_1018 project_bias, spec section 6.3).

Runs without CH'+B keep m = 0 and are never projected; subtracting m = 0 leaves every bit, so the
uncentred runs of a mixed bundle follow cifar5p1_cnn_budget_1019's trajectories (check S-reuse).
A bundle without any centred run runs the previous operation sequence exactly (check S-off).
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import math
import os
import sys
import time
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import cifar5p1_cnn_budget_1019 as B19      # the budget engine; not modified

M, NK, E, C51 = B19.M, B19.NK, B19.E, B19.C51
EXPERIMENT = "cnn_center_budget_1020"
SPEC = ROOT / "specs" / f"spec_{EXPERIMENT}.md"
SPEC_COMMIT = "74e7cc94"
OUT_ROOT = ROOT / "results" / EXPERIMENT
PREV_ROOT = ROOT / "results" / B19.EXPERIMENT

LR = 3e-4
CENTER_BASE = {"RCB": "R", "ECB": "ELU"}                 # centred arm -> its activation's arm
ARM_CFG = {"RCB": "RCB_lr0.0003", "ECB": "ECB_a3.6_lr0.0003",
           "R": "R_lr0.0003", "ELU": "ELU_a3.6_lr0.0003"}
ARMS = ("RCB", "ECB", "R", "ELU")                        # bundle order (spec section 3)
PAIRS = {"RCB": "R", "ECB": "ELU"}                       # centred arm -> its baseline
BUDGETS = (780, 7800)
SEEDS = NK.SEEDS
SMOKE_SEED = 100
FRESH_TASKS = B19.FRESH_TASKS
CKPT_BOUNDARIES = B19.CKPT_BOUNDARIES
BASE_STEPS = B19.BASE_STEPS
KAPPA = 0.2
BETA_H = 0.01
SIDE = (16, 8)                                           # pool-output side after c1, c2
CH, IS_CONV, WIDTHS, SITES = NK.CH, NK.IS_CONV, NK.WIDTHS, NK.SITES
KER, PAD, POOL, FLAT = NK.KER, NK.PAD, NK.POOL, NK.FLAT
NFEAT = (CH * SIDE[0] ** 2, CH * SIDE[1] ** 2, WIDTHS[2], WIDTHS[3])    # what c2, f1, f2, f3 read
BATCH, N_CLASSES = NK.BATCH, NK.N_CLASSES
LAG_POINTS = (0, 1, 3, 10, 30, 100, 300, 780, 2340, 7800)
# open-fraction classes (spec section 6.2; integer comparisons on the positive count k of n samples)
PPOS_BINS = ("p0", "lt05", "sparse", "mid", "hi", "ge95")
NEW_REC = tuple(f"fire_{t}" for t in SITES) + tuple(f"blag_{t}" for t in SITES)


# --------------------------------------------------------------------------
# slots
# --------------------------------------------------------------------------

_PARSE_CFG = M.parse_cfg                                 # the 12-arm engine's parser (patched below)


def parse_slot(name: str) -> dict:
    """The 12-arm engine's configuration of the slot's activation, plus label and centre flag.
    RCB_lr0.0003 -> R_lr0.0003 with CH'+B; ECB_a3.6_lr0.0003 -> ELU_a3.6_lr0.0003 with CH'+B."""
    head = name.split("_")[0]
    if head in CENTER_BASE:
        cfg = dict(_PARSE_CFG(CENTER_BASE[head] + name[len(head):]))
        cfg.update(name=name, label=head, center=True)
    else:
        cfg = dict(_PARSE_CFG(name))
        cfg.update(label=cfg["arm"], center=False)
    return cfg


def main_slots() -> list[tuple[str, int]]:
    return [(ARM_CFG[a], s) for a in ARMS for s in SEEDS]


# --------------------------------------------------------------------------
# activation with the H' state
# --------------------------------------------------------------------------

class CenterAct(M.MethodsAct):
    """MethodsAct plus door H' (cnn_center_feat_1011.FeatDoors, mode "feat") on the centred runs.
    hm[l] is (R, NFEAT[l]); conv layout (channel, y, x), C order.  hm is part of the activation
    state (saved, restored, rewound after the graph capture)."""

    EXTRA = M.MethodsAct.EXTRA + ("hm",)

    def __init__(self, cfgs: list[dict], device, dtype=torch.float32):
        super().__init__(cfgs, device, dtype=dtype)
        R = self.R
        self.center = torch.tensor([[bool(c.get("center", False))] for c in cfgs], device=device)
        self.any_center = bool(self.center.any())
        self.beta_h = BETA_H
        self.hm = [torch.zeros(R, n, dtype=dtype, device=device) for n in NFEAT]
        # per-step batch lag |bm - m_old| / |bm| (a record, not state; NaN on uncentred runs)
        self.blag = [torch.zeros(R, dtype=dtype, device=device) for _ in NFEAT]

    def sub_map(self, l: int) -> torch.Tensor:
        """What conv site l's next layer loses, in the bundled (1, R*C, s, s) layout."""
        return self.hm[l].reshape(1, self.R * CH, SIDE[l], SIDE[l])

    @torch.no_grad()
    def out_mean(self, z: torch.Tensor, l: int) -> torch.Tensor:
        """(R, NFEAT[l]) batch mean of the uncentred output the next layer reads: conv sites
        maxpool(phi(z)) averaged over the images only, fc sites phi(z) over the images."""
        a = self.phi(z, l)
        if IS_CONV[l]:
            return F.max_pool2d(a, POOL, POOL).mean(0).view(self.R, -1)
        return a.mean(1)

    @torch.no_grad()
    def update(self, zs) -> None:
        super().update(zs)                                   # V / Vrec, the host's order
        if not self.any_center:
            return
        cm = self.center
        for l, z in enumerate(zs):
            bm = self.out_mean(z, l)
            old = self.hm[l]
            d = (bm - old).double().norm(dim=1)
            nb = bm.double().norm(dim=1)
            lag = torch.where(nb > 0, d / nb, torch.full_like(d, math.nan))
            self.blag[l].copy_(torch.where(cm[:, 0], lag, torch.full_like(lag, math.nan)).to(self.blag[l].dtype))
            new = old.mul(1 - self.beta_h).add(bm, alpha=self.beta_h)
            self.hm[l].copy_(torch.where(cm, new, old))


# --------------------------------------------------------------------------
# forward with H'
# --------------------------------------------------------------------------

def forward_c(P, xb: torch.Tensor, act: CenterAct, with_inputs: bool = False):
    """xb (R, B, 3, 32, 32) -> (z1, a1, z2, a2, z3, a3, z4, a4, logits), a = the uncentred phi(z).

    sna_cnn_cause_1009.forward with phi(z) - m before c2, f1, f2 and f3 when the bundle holds a
    centred run (m = 0 on the others); without one, exactly that forward's operations.
    `with_inputs` also returns what the four sites read (x, p1, h, i4), bundled layouts."""
    Wc1, bc1, Wc2, bc2, Wf1, bf1, Wf2, bf2, Wf3, bf3 = P
    R, B = xb.shape[:2]
    on = act.any_center
    x = xb.transpose(0, 1).reshape(B, R * 3, 32, 32)
    z1 = F.conv2d(x, Wc1.reshape(R * CH, 3, KER, KER), bc1.reshape(-1), padding=PAD, groups=R)
    a1 = act.phi(z1, 0)
    p1 = F.max_pool2d(a1, POOL, POOL)
    if on:
        p1 = p1 - act.sub_map(0)
    z2 = F.conv2d(p1, Wc2.reshape(R * CH, CH, KER, KER), bc2.reshape(-1), padding=PAD, groups=R)
    a2 = act.phi(z2, 1)
    p2 = F.max_pool2d(a2, POOL, POOL)
    if on:
        p2 = p2 - act.sub_map(1)
    h = p2.reshape(B, R, FLAT).transpose(0, 1)                           # (R, B, 1024)
    z3 = torch.baddbmm(bf1[:, None, :], h, Wf1.transpose(1, 2))
    a3 = act.phi(z3, 2)
    i4 = a3 - act.hm[2][:, None, :] if on else a3
    z4 = torch.baddbmm(bf2[:, None, :], i4, Wf2.transpose(1, 2))
    a4 = act.phi(z4, 3)
    i5 = a4 - act.hm[3][:, None, :] if on else a4
    logits = torch.baddbmm(bf3[:, None, :], i5, Wf3.transpose(1, 2))
    out = (z1, a1, z2, a2, z3, a3, z4, a4, logits)
    if with_inputs:
        return out, (x, p1, h, i4)
    return out


@contextlib.contextmanager
def centered_forward():
    """Route the previous modules' evaluation (unit_stats2, ce_acc), which call
    sna_cnn_cause_1009.forward, through forward_c; restored afterwards."""
    saved = E.forward
    E.forward = forward_c
    try:
        yield
    finally:
        E.forward = saved


# --------------------------------------------------------------------------
# B_width: Codex's width and clip, op by op (cnn_chprime_bias_1018.image_layout / image_width)
# --------------------------------------------------------------------------

def image_layout(q, conv, R):
    return q.reshape(q.shape[0], R, CH, -1).permute(1, 0, 2, 3) if conv else q


def image_width(q, conv, R):
    """(R, n) float64: conv sqrt(mean over positions of the image-axis population variance),
    fc the image-axis population SD."""
    x = image_layout(q, conv, R).double()
    var = x.var(dim=1, correction=0)
    return var.mean(-1).sqrt() if conv else var.sqrt()


@torch.no_grad()
def clip_toward_zero(b: torch.Tensor, cap: torch.Tensor) -> torch.Tensor:
    """float32 clip of b to [-cap, cap] (cap float64); a value rounded beyond the cap is moved one
    ulp toward zero, so |result| <= cap holds exactly."""
    cl = b.double().clamp(-cap, cap).to(b.dtype)
    return torch.where(cl.double().abs() > cap, torch.nextafter(cl, torch.zeros_like(cl)), cl)


# --------------------------------------------------------------------------
# the engine
# --------------------------------------------------------------------------

class CenterEngine(B19.BudgetEngine):
    def __init__(self, slots, data, device, budget: int, graph: bool = True, dtype=torch.float32,
                 stage: str = "train"):
        with patch.object(M, "parse_cfg", parse_slot), patch.object(M, "MethodsAct", CenterAct):
            super().__init__(slots, data, device, budget, graph=graph, dtype=dtype, stage=stage)
        assert isinstance(self.act, CenterAct)
        if any(c["kind"] not in ("relu", "elu") for c in self.cfgs):
            raise ValueError("this experiment has ReLU and ELU runs only")
        R = self.R
        self.labels = [c["label"] for c in self.cfgs]
        self.proj_on = self.act.any_center
        f64 = dict(dtype=torch.float64, device=device)
        self.fire_cnt = [torch.zeros(R, w, **f64) for w in WIDTHS]      # per task: updates clipped
        self.rmax = [torch.zeros(R, w, **f64) for w in WIDTHS]          # per task: max |b_before| / s
        self.viol = torch.zeros(R, 4, **f64)                            # per task: updates with |b| > cap
        self.fire_now = [torch.zeros(R, dtype=dtype, device=device) for _ in WIDTHS]
        for k in NEW_REC:
            self.rec[k] = torch.zeros(R, budget, dtype=dtype, device=device)

    def _acc_tensors(self):
        return (*self.fire_cnt, *self.rmax, self.viol, *self.fire_now, *self.act.blag)

    def _state_tensors(self):
        # the graph capture rewinds these after its warm-up and capture steps
        return (*super()._state_tensors(), *self._acc_tensors())

    @torch.no_grad()
    def project(self, xb: torch.Tensor) -> torch.Tensor:
        """B_width on the centred runs (spec section 4.3).  Returns (R,) finite flags."""
        act, R = self.act, self.R
        B = xb.shape[1]
        cm = act.center
        value = xb.transpose(0, 1).reshape(B, R * 3, 32, 32)
        fin = torch.ones(R, dtype=torch.bool, device=self.device)
        for l in range(4):
            W, b = self.P[2 * l], self.P[2 * l + 1]
            conv = IS_CONV[l]
            if conv:
                q = F.conv2d(value, W.reshape(R * CH, -1, KER, KER), None, padding=PAD, groups=R)
                qf = torch.isfinite(q).reshape(B, R, -1).all(2).all(0)
            else:
                q = torch.bmm(value, W.transpose(1, 2))
                qf = torch.isfinite(q).flatten(1).all(1)
            s = image_width(q, conv, R)
            cap = KAPPA * s
            bd = b.detach().double()
            hit = (bd.abs() > cap) & cm
            rb = torch.where(s > 0, bd.abs() / s,
                             torch.where(bd == 0, torch.zeros_like(bd), torch.full_like(bd, math.inf)))
            b.copy_(torch.where(hit, clip_toward_zero(b.detach(), cap), b.detach()))
            viol = (b.detach().double().abs() > cap) & cm
            self.fire_cnt[l].add_(hit.double())
            self.rmax[l].copy_(torch.maximum(self.rmax[l], rb))
            self.viol[:, l].add_(viol.any(1).double())
            self.fire_now[l].copy_(hit.to(self.dtype).mean(1))
            fin = fin & qf & torch.isfinite(s).all(1) & torch.isfinite(bd).all(1)
            if conv:
                z = q + b.reshape(1, R * CH, 1, 1)
                value = F.max_pool2d(act.phi(z, l), POOL, POOL) - act.sub_map(l)
                if l == 1:
                    value = value.reshape(B, R, FLAT).transpose(0, 1)
            else:
                z = q + b[:, None, :]
                value = act.phi(z, l) - act.hm[l][:, None, :]
        return fin

    # one update: the 12-arm engine's step through forward_c, then the EMA and B_width
    def step(self) -> None:
        R, j = self.R, self.j
        self.act.train_mode = True
        try:
            idx = self.idx.index_select(1, j.view(1)).squeeze(1)
            xb, yb = self.data.X[idx], self.data.Y[idx]
            out = forward_c(self.P, xb, self.act)
            logits = out[8]
            lossv = F.cross_entropy(logits.reshape(-1, N_CLASSES), yb.reshape(-1),
                                    reduction="none").view(R, BATCH).mean(1)
            hit = (logits.detach().argmax(-1) == yb).to(self.dtype).mean(1)
            self.act.observe(lossv)
            grads = torch.autograd.grad(lossv.sum(), self.P)
        finally:
            self.act.train_mode = False
        with torch.no_grad():
            ok = torch.isfinite(lossv)
            for gr in grads:
                ok = ok & torch.isfinite(gr).flatten(1).all(1)
            self.tc.add_(1)
            c1 = (1 - torch.pow(self.b1, self.tc)).to(self.dtype)
            c2 = (1 - torch.pow(self.b2, self.tc)).to(self.dtype)
            for i, (p, gr, mi, vi) in enumerate(zip(self.P, grads, self.m, self.v)):
                if self.l2_any:
                    gr = gr + self.l2_b[i] * (p.detach() - self.P0[i])
                mi.mul_(NK.ADAM_B1).add_(gr, alpha=1 - NK.ADAM_B1)
                vi.mul_(NK.ADAM_B2).addcmul_(gr, gr, value=1 - NK.ADAM_B2)
                p.sub_(self.lr_b[i] * (mi / c1) / ((vi / c2).sqrt() + NK.ADAM_EPS))
            self.act.update([out[0].detach(), out[2].detach(), out[4].detach(), out[6].detach()])
            for V in self.act.V:
                ok = ok & torch.isfinite(V).all(1)
            if self.proj_on:
                fin = self.project(xb)
                ok = ok & (fin | ~self.act.center[:, 0])
            self.bad_step.copy_(torch.where((self.bad_step < 0) & ~ok, j, self.bad_step))
            pos = self.iota == j
            recs = [("ce", lossv.detach()), ("hit", hit),
                    ("gate", self.act.gate[:, 0]), ("mult", self.act.mult[:, 0])]
            if self.proj_on:
                recs += [(f"fire_{t}", self.fire_now[l]) for l, t in enumerate(SITES)]
                recs += [(f"blag_{t}", self.act.blag[l]) for l, t in enumerate(SITES)]
            for k, val in recs:
                self.rec[k].copy_(torch.where(pos, val[:, None], self.rec[k]))
            j.add_(1)

    def train_task(self, batches: torch.Tensor, n_steps: int | None = None, hook=None, hook_at=()) -> list[int]:
        """`n_steps` (default: the budget) updates; hook(k) after update k for k in hook_at
        (k = 0: before the first update).  The hook must not write any state."""
        n = self.budget if n_steps is None else n_steps
        if tuple(batches.shape) != (self.R, self.budget, BATCH):
            raise ValueError(f"batches must be (R, {self.budget}, 32)")
        self.idx.copy_(batches.to(self.device))
        self.j.zero_()
        self.bad_step.fill_(-1)
        for v in self.rec.values():
            v.zero_()
        for v in self._acc_tensors():
            v.zero_()
        if self.graph and self.cg is None:
            self.capture()
        marks = set(hook_at) if hook is not None else set()
        if 0 in marks:
            hook(0)
        for k in range(1, n + 1):
            if self.cg is not None:
                self.cg.replay()
            else:
                self.step()
            self.n_done += 1
            if k in marks:
                hook(k)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        return []


# --------------------------------------------------------------------------
# diagnostics (read only)
# --------------------------------------------------------------------------

@torch.no_grad()
def unit_stats_c(P, act, X, rows, Y=None, chunk: int = 50):
    """cifar5p1_cnn_budget_1019.unit_stats2 through forward_c, plus the integer positive counts
    npos and the sample count n per site."""
    with centered_forward():
        st, acc = B19.unit_stats2(P, act, X, rows, Y, chunk=chunk)
    N = rows.shape[1]
    for l, tag in enumerate(SITES):
        nd = N * NK.CONV_HW[l] ** 2 if IS_CONV[l] else N
        st[tag]["npos"] = np.rint(st[tag]["ppos"] * nd).astype(np.int64)
        st[tag]["n_samples"] = nd
    return st, acc


def ce_acc_c(P, act, X, Y, rows):
    with centered_forward():
        return NK.ce_acc(P, act, X, Y, rows)


def ppos_classes(npos: np.ndarray, n: int) -> dict:
    """Six exclusive classes of the positive count k out of n (spec section 6.2), integer tests:
    p0 k = 0; lt05 0 < k, 20k < n; sparse 20k >= n, 10k < 3n; mid 10k >= 3n, 10k <= 7n;
    hi 10k > 7n, 20k < 19n; ge95 20k >= 19n."""
    k = np.asarray(npos, dtype=np.int64)
    n = int(n)
    return {"p0": k == 0, "lt05": (k > 0) & (20 * k < n), "sparse": (20 * k >= n) & (10 * k < 3 * n),
            "mid": (10 * k >= 3 * n) & (10 * k <= 7 * n), "hi": (10 * k > 7 * n) & (20 * k < 19 * n),
            "ge95": 20 * k >= 19 * n}


@torch.no_grad()
def bias_stats(P, act, X, rows, chunk: int = 50) -> dict:
    """Per site and unit on X[rows[r]] through forward_c: the bias-free q = W * h_in, its mean a
    (conv: over images x positions), the width s_q (spec 4.3's definition over all images; conv
    sqrt(mean_p Var_images q)), the mixed images x positions SD s_mix, the bias b, and b / s_q,
    a / s_q, (a + b) / s_q (NaN where s_q = 0).  float64 sums shifted by the first chunk's mean."""
    R, N = rows.shape
    dev = X.device
    f64 = torch.float64
    S1, S2, shift = [None] * 4, [None] * 4, [None] * 4
    for i0 in range(0, N, chunk):
        idx = rows[:, i0:i0 + chunk]
        b_ = idx.shape[1]
        _, ins = forward_c(P, X[idx], act, with_inputs=True)
        for l in range(4):
            W = P[2 * l]
            if IS_CONV[l]:
                q = F.conv2d(ins[l], W.reshape(R * CH, -1, KER, KER), None, padding=PAD, groups=R)
                x = q.reshape(b_, R, CH, -1).permute(1, 0, 2, 3).double()       # (R, b, C, P)
            else:
                x = torch.bmm(ins[l], W.transpose(1, 2)).double()               # (R, b, n)
            if shift[l] is None:
                shift[l] = x.mean(1, keepdim=True)
                S1[l] = torch.zeros_like(shift[l][:, 0])
                S2[l] = torch.zeros_like(shift[l][:, 0])
            xc = x - shift[l]
            S1[l] += xc.sum(1)
            S2[l] += (xc * xc).sum(1)
    out = {}
    for l, tag in enumerate(SITES):
        mc = S1[l] / N
        var = (S2[l] / N - mc * mc).clamp_min(0.0)
        mean = mc + shift[l][:, 0]
        if IS_CONV[l]:
            a = mean.mean(-1)
            sq = var.mean(-1).sqrt()
            smix = (var.mean(-1) + mean.var(-1, correction=0)).sqrt()
        else:
            a, sq, smix = mean, var.sqrt(), var.sqrt()
        b = P[2 * l + 1].detach().double()
        pos = sq > 0
        nan = torch.full_like(sq, math.nan)
        res = {"a": a, "s_q": sq, "s_mix": smix, "b": b, "b_over_s": torch.where(pos, b / sq, nan),
               "a_over_s": torch.where(pos, a / sq, nan), "m_over_s": torch.where(pos, (a + b) / sq, nan)}
        out[tag] = {k: v.cpu().numpy() for k, v in res.items()}
    return out


@torch.no_grad()
def lag_eval(P, act, X, rows, chunk: int = 50) -> dict:
    """Per site and run: |xbar - m| / |xbar|, |xbar|, |m|, xbar the mean over X[rows[r]] of the
    uncentred output the next layer reads (pool(phi(z)) per feature; fc phi(z) per unit)."""
    R, N = rows.shape
    S = [torch.zeros(R, n, dtype=torch.float64, device=X.device) for n in NFEAT]
    for i0 in range(0, N, chunk):
        idx = rows[:, i0:i0 + chunk]
        out = forward_c(P, X[idx], act)
        for l in range(4):
            a = out[2 * l + 1]
            if IS_CONV[l]:
                S[l] += F.max_pool2d(a, POOL, POOL).double().sum(0).view(R, -1)
            else:
                S[l] += a.double().sum(1)
    res = {}
    for l, tag in enumerate(SITES):
        xb = S[l] / N
        m = act.hm[l].double()
        d, nx = (xb - m).norm(dim=1), xb.norm(dim=1)
        res[tag] = {"lag": torch.where(nx > 0, d / nx, torch.full_like(d, math.nan)).cpu().numpy(),
                    "xbar_norm": nx.cpu().numpy(), "m_norm": m.norm(dim=1).cpu().numpy()}
    return res


def site_rows_c(stats, bstats, ctl, eng, base: dict) -> list[dict]:
    """cifar5p1_cnn_budget_1019.site_rows (columns unchanged; 'arm' = this run's arm label) plus
    the open-fraction classes and the bias / width medians."""
    rows = B19.site_rows(stats, ctl, eng, base)
    i = 0
    for r in range(eng.R):
        for l, tag in enumerate(SITES):
            q = rows[i]
            i += 1
            assert q["seed"] == eng.slots[r][1] and q["site"] == tag
            q["arm"] = eng.labels[r]
            q["center"] = bool(eng.cfgs[r]["center"])
            cls = ppos_classes(stats[tag]["npos"][r], stats[tag]["n_samples"])
            for k in PPOS_BINS:
                q[f"ppos_{k}_frac"] = float(cls[k].mean())
            if bstats is not None:
                bs = bstats[tag]
                for k in ("s_q", "b_over_s", "a_over_s", "m_over_s"):
                    q.update(NK.agg(bs[k][r], k))
                q.update(NK.agg(np.abs(bs["b_over_s"][r]), "abs_b_over_s"))
    return rows


def unit_arrays_c(stats, prefix: str) -> dict:
    out = B19.unit_arrays(stats, prefix)
    for tag in SITES:
        out[f"{prefix}{tag}_npos"] = stats[tag]["npos"]
        out[f"{prefix}{tag}_n_samples"] = np.int64(stats[tag]["n_samples"])
    return out


# --------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------

def run(slots, out: Path, n_tasks: int, budget: int, device, graph: bool = True, resume: bool = True,
        data: NK.Data | None = None, log=None, allow_dirty: bool = False, chunk: int = 50,
        fresh_tasks=FRESH_TASKS, ckpt_boundaries=CKPT_BOUNDARIES, stop_after: int | None = None,
        lag_points=LAG_POINTS) -> dict:
    log = log or (lambda m: print(m, flush=True))
    out = Path(out)
    t_wall = time.time()
    gstart = NK.git_state()
    if gstart["dirty_src_analysis_specs"] and not allow_dirty:
        raise SystemExit("src/analysis/specs have uncommitted changes; commit before a registered run")
    data = data or NK.Data(device)
    seeds = sorted({s for a, s in slots})
    prep = B19.prepare(seeds, data, budget)
    plans, probe = prep["plans"], prep["probe"].to(device)
    eng = CenterEngine(slots, data, device, budget, graph=graph)
    R = eng.R
    fresh_tasks = tuple(t for t in fresh_tasks if t <= n_tasks)
    lag_at = tuple(k for k in lag_points if k <= budget)
    meta = {"schema": 1, "run_id": EXPERIMENT, "slots": [list(s) for s in eng.slots], "budget": budget,
            "n_tasks": n_tasks, "batch": BATCH, "graph": eng.graph, "fresh_tasks": list(fresh_tasks),
            "ckpt_boundaries": [b for b in ckpt_boundaries if b <= n_tasks], "kappa": KAPPA, "beta_h": BETA_H,
            "lag_points": list(lag_at), "spec_sha256": hashlib.sha256(SPEC.read_bytes()).hexdigest(),
            "data_sha256": data.sha256, "probe_sha256": prep["probe_sha256"],
            "batch_sha256": {str(s): v for s, v in prep["batch_sha256"].items()}}
    for d in ("checkpoints", "units", "steps"):
        (out / d).mkdir(parents=True, exist_ok=True)
    st_path = out / "state.pt"
    book = {"rows": [], "probe": [], "train": [], "lag": [], "timing": [], "gits": [gstart], "resumed": [],
            "diverged": {}}
    t0 = 1
    if st_path.exists():
        if not resume:
            raise SystemExit(f"{st_path} exists; choose a fresh output")
        st = torch.load(st_path, map_location="cpu", weights_only=False)
        if st["meta"] != meta:
            raise SystemExit(f"{st_path} belongs to another configuration")
        eng.load_full_state(st["state"])
        book = st["book"]
        book["gits"].append(gstart)
        t0 = st["task"] + 1
        book["resumed"].append({"at_task": t0, **gstart})
        log(f"[{time.strftime('%T')}] resumed after task {st['task']}")
    elif any(p.is_file() for p in out.iterdir()):
        raise SystemExit(f"{out} holds files but no resumable state; choose a fresh output")
    else:
        NK.write_json(out / "task_plan.json", {str(s): [{"task": t, "hard": bool(h), "classes": c}
                                                        for t, (h, c) in enumerate(plans[s], 1)] for s in seeds})
        NK.write_json(out / "batch_hashes.json",
                      {"rule": "sha256 of the (budget, 32) int64 global train rows of each task; first 780 "
                               "batches = the previous c51_batch stream, the rest = c51_budget_ext|t<t>",
                       "budget": budget, "per_seed": {str(s): v for s, v in prep["batch_sha256"].items()},
                       "first780_sha256": {str(s): [NK.sha(prep["batches"][s][t][:BASE_STEPS])
                                                    for t in range(NK.N_TASKS)] for s in seeds}})
        NK.write_json(out / "probe_ids.json", {"role": NK.PROBE_ROLE, "seed": NK.PROBE_SEED,
                                               "rows": prep["probe"].tolist(), "sha256": prep["probe_sha256"]})
        pr0 = probe[None].expand(R, -1)
        st0, _ = unit_stats_c(eng.P, eng.act, data.X, pr0, chunk=chunk)
        bs0 = bias_stats(eng.P, eng.act, data.X, pr0, chunk=chunk)
        ctl = NK.control_arrays(eng.act)
        book["probe"] += site_rows_c(st0, bs0, ctl, eng, {"boundary": 0})
        np.savez(out / "units" / "b00.npz", **unit_arrays_c(st0, "probe_"),
                 **{f"bias_{t}_{k}": v.astype(np.float32) for t in SITES for k, v in bs0[t].items()},
                 **{f"hm_{t}": eng.act.hm[l].detach().cpu().numpy() for l, t in enumerate(SITES)})
        if 0 in ckpt_boundaries:
            NK.save_atomic({"meta": meta, "state": eng.full_state(), "boundary": 0}, out / "checkpoints" / "b00.pt")
        NK.save_atomic({"meta": meta, "state": eng.full_state(), "task": 0, "book": book}, st_path)
    prov = {"run_id": EXPERIMENT, "status": "RUNNING", "meta": meta, "env": NK.env_info(device),
            "spec": str(SPEC.relative_to(ROOT)), "spec_commit_registered": SPEC_COMMIT,
            "gits": book["gits"], "resumed": book["resumed"], "sources_sha256": NK.source_hashes(),
            "configs": {a: parse_slot(a) for a in sorted({a for a, s in eng.slots})},
            "execution": "CUDA graph of one update (forward_c, Adam, V and H' EMA, B_width)" if eng.graph else "eager",
            "scientific_verdict": None}
    NK.write_json(out / "provenance.json", prov)
    for t in range(t0, n_tasks + 1):
        tt = time.time()
        hard = t % 2 == 1
        batches = torch.stack([prep["batches"][s][t - 1] for a, s in eng.slots]).to(device)
        tr = NK.slot_rows(eng.slots, plans, t, data.tr_rows).to(device)
        te = NK.slot_rows(eng.slots, plans, t, data.te_rows).to(device)
        sce, sacc = ce_acc_c(eng.P, eng.act, data.X, data.Y, tr)
        ctl0 = NK.control_arrays(eng.act)
        trs0, _ = unit_stats_c(eng.P, eng.act, data.X, tr, chunk=chunk)
        lag_task = {}

        def hook(k):
            lag_task[k] = lag_eval(eng.P, eng.act, data.X, tr, chunk=chunk)
        t_train = time.time()
        eng.train_task(batches, hook=hook if eng.act.any_center else None, hook_at=lag_at)
        t_train = time.time() - t_train
        rec = eng.records()
        bad = eng.bad_step.cpu().numpy()
        viol = eng.viol.cpu().numpy()
        if (viol > 0).any():
            NK.write_json(out / "failure.json", {"status": "CHECK_FAILED", "task": t, "reason":
                          "S-proj: |b| > kappa s after the projection", "viol": viol.tolist()})
            prov.update(status="CHECK_FAILED")
            NK.write_json(out / "provenance.json", prov)
            raise RuntimeError(f"CHECK_FAILED: projection exceeded the cap in task {t}")
        finite = np.array([all(bool(torch.isfinite(p[r]).all()) for p in eng.P) for r in range(R)])
        for r in range(R):
            key = f"{eng.slots[r][0]}|{eng.slots[r][1]}"
            if (bad[r] >= 0 or not finite[r]) and key not in book["diverged"]:
                book["diverged"][key] = {"task": t, "bad_step_in_task": int(bad[r]), "params_finite": bool(finite[r])}
        np.savez_compressed(out / "steps" / f"t{t:02d}.npz", ce=rec["ce"].astype(np.float32),
                            hit=rec["hit"].astype(np.float32),
                            **{k: rec[k].astype(np.float32) for k in NEW_REC})
        t_diag = time.time()
        trs1, tr_acc = unit_stats_c(eng.P, eng.act, data.X, tr, data.Y, chunk=chunk)
        pr = probe[None].expand(R, -1)
        prs, _ = unit_stats_c(eng.P, eng.act, data.X, pr, chunk=chunk)
        bs = bias_stats(eng.P, eng.act, data.X, pr, chunk=chunk)
        _, te_acc = ce_acc_c(eng.P, eng.act, data.Xte, data.Yte, te)
        ctl1 = NK.control_arrays(eng.act)
        wn = NK.weight_row_norms(eng.P)
        t_diag = time.time() - t_diag
        units = {**unit_arrays_c(prs, "probe_"), **unit_arrays_c(trs0, "train_start_"),
                 **unit_arrays_c(trs1, "train_end_")}
        for l, tag in enumerate(SITES):
            units[f"hm_{tag}"] = eng.act.hm[l].detach().cpu().numpy()
            units[f"fire_cnt_{tag}"] = eng.fire_cnt[l].cpu().numpy().astype(np.int32)
            units[f"rmax_{tag}"] = eng.rmax[l].cpu().numpy().astype(np.float32)
            for k, v in bs[tag].items():
                units[f"bias_{tag}_{k}"] = v.astype(np.float32)
        for tag, v in wn.items():
            units[f"wnorm_{tag}"] = v.astype(np.float32)
        book["probe"] += site_rows_c(prs, bs, ctl1, eng, {"boundary": t})
        book["train"] += site_rows_c(trs0, None, ctl0, eng, {"task": t, "phase": "start"})
        book["train"] += site_rows_c(trs1, None, ctl1, eng, {"task": t, "phase": "end"})
        for k, lg in sorted(lag_task.items()):
            for r, (name, seed) in enumerate(eng.slots):
                if not eng.cfgs[r]["center"]:
                    continue
                book["lag"].append({"arm": eng.labels[r], "cfg": name, "budget": budget, "seed": seed, "task": t,
                                    "update": k, **{f"{x}_{tag}": float(lg[tag][x][r]) for tag in SITES
                                                    for x in ("lag", "xbar_norm", "m_norm")}})
        for r, (name, seed) in enumerate(eng.slots):
            cfg = eng.cfgs[r]
            ce, hit = rec["ce"][r], rec["hit"][r]
            cls = plans[seed][t - 1][1]
            key = f"{name}|{seed}"
            # the previous run's columns (same names, same order, same computations)
            row = {"arm": cfg["label"], "cfg": name, "lr": cfg["lr"], "budget": budget, "seed": seed, "slot": r,
                   "task": t, "hard": int(hard), "n_classes": len(cls), "classes": " ".join(map(str, cls)),
                   "online_acc": float(hit.mean()), "first100_online_acc": float(hit[:NK.FIRST_N].mean()),
                   "first780_online_acc": float(hit[:BASE_STEPS].mean()),
                   "train_acc": float(tr_acc[r]), "test_acc": float(te_acc[r]),
                   "switch_ce": float(sce[r]), "switch_acc": float(sacc[r]),
                   "ce_step1": float(ce[0]), "ce_first10": float(ce[:10].mean()),
                   "ce_last10": float(ce[-10:].mean()), "ce_mean": float(ce.mean()),
                   "ce_first780_mean": float(ce[:BASE_STEPS].mean()),
                   "alpha_frozen": bool(eng.act.frozen[r, 0]),
                   "diverged": key in book["diverged"], "bad_step": int(bad[r])}
            for l, tag in enumerate(SITES):
                row[f"alpha_med_{tag}"] = None
                row[f"two_alpha_sqrtV_med_{tag}"] = None
                row[f"zsd_med_{tag}"] = float(np.median(trs1[tag]["s"][r]))
                row[f"zbar_med_{tag}"] = float(np.median(trs1[tag]["m"][r]))
                row[f"dmean_med_{tag}"] = float(np.median(trs1[tag]["dmean"][r]))
            # new columns
            row["act"] = cfg["arm"]
            row["center"] = bool(cfg["center"])
            for l, tag in enumerate(SITES):
                if cfg["center"]:
                    row[f"fire_rate_{tag}"] = float(eng.fire_cnt[l][r].mean().item() / budget)
                    row[f"fire_first100_{tag}"] = float(rec[f"fire_{tag}"][r][:NK.FIRST_N].mean())
                    row[f"rmax_med_{tag}"] = float(np.median(eng.rmax[l][r].cpu().numpy()))
                    row[f"blag_first10_{tag}"] = float(rec[f"blag_{tag}"][r][:10].mean())
                    row[f"blag_last100_{tag}"] = float(rec[f"blag_{tag}"][r][-100:].mean())
                    row[f"lag_u0_{tag}"] = float(lag_task[0][tag]["lag"][r])
                    row[f"lag_end_{tag}"] = float(lag_task[lag_at[-1]][tag]["lag"][r])
                else:
                    for k in ("fire_rate", "fire_first100", "rmax_med", "blag_first10", "blag_last100",
                              "lag_u0", "lag_end"):
                        row[f"{k}_{tag}"] = None
                row[f"probe_abs_b_over_s_med_{tag}"] = float(np.nanmedian(np.abs(bs[tag]["b_over_s"][r])))
            book["rows"].append(row)
        np.savez(out / "units" / f"b{t:02d}.npz", **units)
        book["timing"].append({"task": t, "train_s": t_train, "diag_s": t_diag, "total_s": time.time() - tt})
        if t in ckpt_boundaries:
            NK.save_atomic({"meta": meta, "state": eng.full_state(), "boundary": t}, out / "checkpoints" / f"b{t:02d}.pt")
        NK.write_csv(out / "per_task.csv", book["rows"])
        if book["lag"]:
            NK.write_csv(out / "lag.csv", book["lag"])
        NK.save_atomic({"meta": meta, "state": eng.full_state(), "task": t, "book": book}, st_path)
        log(f"[{time.strftime('%T')}] budget {budget} task {t:2d}/{n_tasks} {'hard' if hard else 'easy'} "
            f"train {t_train:6.1f}s diag {t_diag:4.1f}s diverged {len(book['diverged'])}")
        if stop_after is not None and t == stop_after:
            raise KeyboardInterrupt(f"stop_after={t} (check hook)")
    # ---- fresh controls (spec section 6.1)
    fresh_rows = []
    cont = {(q["cfg"], q["seed"], q["task"]): q for q in book["rows"]}
    for tf in fresh_tasks:
        tt = time.time()
        fe = CenterEngine(eng.slots, data, device, budget, graph=graph, stage="fresh")
        batches = torch.stack([prep["batches"][s][tf - 1] for a, s in eng.slots]).to(device)
        fe.train_task(batches)
        if (fe.viol.cpu().numpy() > 0).any():
            raise RuntimeError(f"CHECK_FAILED: projection exceeded the cap in fresh task {tf}")
        frec = fe.records()
        tr = NK.slot_rows(eng.slots, plans, tf, data.tr_rows).to(device)
        te = NK.slot_rows(eng.slots, plans, tf, data.te_rows).to(device)
        _, ftr = ce_acc_c(fe.P, fe.act, data.X, data.Y, tr)
        _, fte = ce_acc_c(fe.P, fe.act, data.Xte, data.Yte, te)
        for r, (name, seed) in enumerate(eng.slots):
            c = cont[(name, seed, tf)]
            h = frec["hit"][r]
            q = {"arm": eng.labels[r], "cfg": name, "budget": budget, "seed": seed, "task": tf,
                 "batch_sha256": prep["batch_sha256"][seed][tf - 1],
                 "fresh_online_acc": float(h.mean()), "fresh_first780_online_acc": float(h[:BASE_STEPS].mean()),
                 "fresh_first100_online_acc": float(h[:NK.FIRST_N].mean()),
                 "fresh_train_acc": float(ftr[r]), "fresh_test_acc": float(fte[r]),
                 "continual_online_acc": c["online_acc"], "continual_first780_online_acc": c["first780_online_acc"],
                 "continual_first100_online_acc": c["first100_online_acc"], "continual_test_acc": c["test_acc"],
                 "fresh_bad_step": int(fe.bad_step[r])}
            for k in ("online_acc", "first780_online_acc", "first100_online_acc", "test_acc"):
                q[f"gap_{k}"] = q[f"fresh_{k}"] - q[f"continual_{k}"]
            fresh_rows.append(q)
        book["timing"].append({"task": f"fresh_t{tf}", "total_s": time.time() - tt})
        del fe
        log(f"[{time.strftime('%T')}] budget {budget} fresh t{tf} done")
    NK.write_csv(out / "fresh_control.csv", fresh_rows)
    NK.write_csv(out / "probe_stats.csv", book["probe"])
    NK.write_csv(out / "train_stats.csv", book["train"])
    prov.update(status="COMPLETED", gits=book["gits"], resumed=book["resumed"], timing=book["timing"],
                diverged=book["diverged"], wall_clock_s_this_process=time.time() - t_wall,
                cuda_max_mem_mb=(torch.cuda.max_memory_allocated() / 2 ** 20 if device.type == "cuda" else None),
                updates={"continual": R * n_tasks * budget, "fresh": R * budget * len(fresh_tasks)},
                sources_sha256=NK.source_hashes())
    NK.write_json(out / "provenance.json", prov)
    log(f"[{time.strftime('%T')}] done {out} ({(time.time() - t_wall) / 60:.1f} min, diverged {len(book['diverged'])})")
    return prov


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["smoke", "main"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    a = ap.parse_args()
    device = NK.setup(a.device)
    data = NK.Data(device)
    out = Path(a.out)
    if a.stage == "smoke":
        # spec section 8: seed 100, 3 tasks, one arm per budget
        for b, arm in ((780, "RCB"), (7800, "ECB")):
            run([(ARM_CFG[arm], SMOKE_SEED)], out / f"b{b}", 3, b, device, graph=not a.no_graph, data=data,
                allow_dirty=a.allow_dirty, fresh_tasks=(1,), ckpt_boundaries=(0, 1, 3))
    else:
        if a.budget not in BUDGETS:
            raise SystemExit(f"--budget from {BUDGETS}")
        run(main_slots(), out, NK.N_TASKS, a.budget, device, graph=not a.no_graph, data=data,
            allow_dirty=a.allow_dirty)


if __name__ == "__main__":
    main()
