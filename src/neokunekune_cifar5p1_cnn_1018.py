#!/usr/bin/env python3
"""neokunekune_cifar5p1_cnn_1018 -- loss-gated Snake (NEO) on the real-label 5+1 CIFAR CNN.

    python3 src/neokunekune_cifar5p1_cnn_1018.py run --seeds 2000-2009 \
        --out results/neokunekune_cifar5p1_cnn_1018/main
    python3 src/neokunekune_cifar5p1_cnn_1018.py run --seeds 9000,9001 --tasks 2 \
        --out results/neokunekune_cifar5p1_cnn_1018/_smoke --allow-dirty

Registered spec: specs/spec_neokunekune_cifar5p1_cnn_1018.md (commit 0250b361).

The box is 5+1 CIFAR as in `cifar5p1_mlp_0920` (CIFAR-100 fine labels, 30 tasks, odd tasks
hard = 5 classes x 500 images, even tasks easy = 1 class x 500, 90 classes consumed without
revisit, 100 outputs, 780 Adam updates of batch 32 per task, lr 1e-4, fixed CIFAR-100 channel
standardisation) on the CNN of `rlcifar_cnn_0908` (four activation sites c1 c2 f1 f2) with a
100-wide head.  The four arms are trained side by side in ONE bundle (grouped conv + bmm, the
engine of `sna_cnn_cause_1009`), so every arm runs the same kernels at the same shapes:

    R        ReLU
    SNA      Snake, alpha = clip(0.6 / sqrt(V), .005, 3), V the EMA (beta .01) of the batch
             variance (conv: per channel over batch x positions; fc: per unit over the batch)
    SNAfrz1  SNA in task 1; alpha and V fixed per channel/unit after the last update of task 1
    NEO      SNA in task 1; from task 2 on V moves at beta_eff = .01 g,
             g = min(1, Lref / max(L, 1e-4)), L the pre-update training CE of the batch and
             Lref the median CE of the last 78 updates of task 1 (spec §3.1-3.2)

One update (spec §3.2): alpha from the pre-update V -> forward, CE, pre-update hit -> g from
the detached CE -> backward and Adam with that alpha -> V from this forward's variance.

Nothing in the imported host modules is modified, and none of their module constants are
patched: the data, task plan and batch lists come from `cifar5p1_mlp_0920`'s functions, the
bundled forward and Snake from `sna_cnn_cause_1009`, the net's shapes from `rlcifar_cnn_0908`.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import pmnist_0905 as H                 # streams, deterministic setup, csv
from src import cifar5p1_mlp_0920 as C51         # CIFAR-100 reader, task plan, batch lists
from src import rlcifar_cnn_0908 as CN           # the CNN's shapes and sites
from src import sna_cnn_cause_1009 as E          # bundled Snake and bundled forward

EXPERIMENT = "neokunekune_cifar5p1_cnn_1018"
SPEC = ROOT / "specs" / f"spec_{EXPERIMENT}.md"
SPEC_COMMIT = "0250b361d0519b1f0e697e142788c90b5f6e2052"
OUT_ROOT = ROOT / "results" / EXPERIMENT

ARMS = ("R", "SNA", "SNAfrz1", "NEO")
SNAKE_ARMS = ("SNA", "SNAfrz1", "NEO")
SEEDS = tuple(range(2000, 2010))
SMOKE_SEEDS = (9000, 9001)

N_TASKS = C51.N_TASKS            # 30
STEPS = C51.STEPS_PER_TASK       # 780
BATCH = C51.BATCH                # 32
N_CLASSES = C51.N_CLASSES        # 100
LR = 1e-4
ADAM_B1, ADAM_B2, ADAM_EPS = 0.9, 0.999, 1e-8
C_SNAKE, BETA, ALPHA_LO, ALPHA_HI = 0.6, 0.01, 0.005, 3.0
LOSS_FLOOR = 1e-4
REF_LAST = 78                    # task-1 updates j = 703..780
BLOCK = 10                       # 78 blocks per task
FIRST_N = 100
LATE = (21, 23, 25, 27, 29)
EARLY = (1, 3, 5, 7, 9)
CKPT_TASKS = (1, 15, 28, 29, 30)
PROBE_ROLE, PROBE_SEED, PROBE_PER_CLASS = "neo_c51_probe", 1018, 20

CH, HID, KER, PAD, POOL, FLAT = CN.CHANNELS, CN.HIDDEN, CN.KERNEL, CN.PAD, CN.POOL, CN.FLAT
SHAPES = (((CH, 3, KER, KER), (CH,)), ((CH, CH, KER, KER), (CH,)),
          ((HID, FLAT), (HID,)), ((HID, HID), (HID,)), ((N_CLASSES, HID), (N_CLASSES,)))
FAN_IN = (KER * KER * 3, KER * KER * CH, FLAT, HID, HID)
SITES, IS_CONV, WIDTHS, WTAGS = CN.SITES, CN.IS_CONV, CN.WIDTHS, CN.WEIGHT_TAGS
CONV_HW = (32, 16)               # spatial size of the c1 / c2 preactivation
assert E.WIDTHS == WIDTHS and E.IS_CONV == IS_CONV and E.CN.FLAT == FLAT


# --------------------------------------------------------------------------
# precision and determinism (spec §2: float32, no AMP, no TF32, benchmark off, deterministic)
# --------------------------------------------------------------------------

def setup(device_str: str = "cuda") -> torch.device:
    dev = H.setup(device_str)          # use_deterministic_algorithms(True), cudnn.benchmark False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.conv.fp32_precision = "ieee"
    return dev


def precision_state() -> dict:
    return {"cuda_matmul_fp32_precision": torch.backends.cuda.matmul.fp32_precision,
            "cudnn_conv_fp32_precision": torch.backends.cudnn.conv.fp32_precision,
            "cudnn_enabled": torch.backends.cudnn.enabled,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "amp": False, "dtype": "float32"}


# --------------------------------------------------------------------------
# data, task plan, batch lists, probe
# --------------------------------------------------------------------------

def sha(a) -> str:
    if isinstance(a, torch.Tensor):
        a = a.detach().cpu().contiguous().numpy()
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


class Data:
    """CIFAR-100 standardised once on the device (u8/255, fixed channel mean/sd), shared by
    every run and every engine.  Rows are the archive's file order."""

    def __init__(self, device, cifar: C51.Cifar100 | None = None, dtype=torch.float32):
        cifar = cifar or C51.Cifar100()
        self.cifar, self.sha256, self.dtype = cifar, cifar.sha256, dtype
        self.X = cifar.inputs("train", "std", device).view(-1, *CN.IMG).to(dtype)
        self.Y = cifar.train_y.to(device)
        self.Xte = cifar.inputs("test", "std", device).view(-1, *CN.IMG).to(dtype)
        self.Yte = cifar.test_y.to(device)
        self.tr_rows = C51.class_rows(cifar.train_y)
        self.te_rows = C51.class_rows(cifar.test_y)


def task_plan(seed: int) -> list[tuple[bool, list[int]]]:
    """[(hard?, class ids)] for tasks 1..30 -- the host's `task_plan` (stream c51_classes)."""
    return C51.task_plan(seed)


def make_batches(seed: int, plan, tr_rows) -> torch.Tensor:
    """(30, 780, 32) global train rows for every task, drawn BEFORE training from a new
    `c51_batch` stream of the seed, consumed task 1..30 in the host's order (spec §4)."""
    g = H.stream("c51_batch", seed)
    out = []
    for hard, cls in plan:
        rows = torch.cat([tr_rows[c] for c in cls])
        out.append(C51.batch_indices(g, rows, STEPS))
    return torch.stack(out)


def probe_ids(tr_rows) -> torch.Tensor:
    """2000 fixed train rows, 20 per class for all 100 classes (CPU stream neo_c51_probe,
    seed 1018), class-major.  A geometry probe: these are training images, not a test set,
    and they are never used for an update or for calibration."""
    g = H.stream(PROBE_ROLE, PROBE_SEED)
    return torch.cat([tr_rows[c][torch.randperm(len(tr_rows[c]), generator=g)[:PROBE_PER_CLASS]]
                      for c in range(N_CLASSES)])


def init_params(seed: int) -> list[torch.Tensor]:
    """U(-1/sqrt(fan_in), +1/sqrt(fan_in)) per tensor, W then b, layers in forward order, all
    from the host's CPU `init` stream (the rule of `rlcifar_cnn_0908.init_params`; only the
    head is 100-wide).  Activation-independent: the four arms of a seed start identical."""
    g = H.stream("init", seed)
    params = []
    for (ws, bs), fan in zip(SHAPES, FAN_IN):
        bound = 1.0 / math.sqrt(fan)
        W = (torch.rand(ws, generator=g, dtype=torch.float32) * 2 - 1) * bound
        b = (torch.rand(bs, generator=g, dtype=torch.float32) * 2 - 1) * bound
        params += [W, b]
    return params


# --------------------------------------------------------------------------
# the bundled activation: ReLU runs, Snake runs, the loss gate, the freeze
# --------------------------------------------------------------------------

class NeoAct(E.BundleSnake):
    """`sna_cnn_cause_1009.BundleSnake` (c = .6 everywhere) for R runs, plus

      relu        runs whose activation is ReLU (their Snake state is kept but never used)
      calibrated  Lref set (all Snake runs after task 1; only a record for SNA / SNAfrz1)
      enabled     the gate drives V (NEO after task 1)
      frozen      alpha and V fixed (SNAfrz1 after task 1)
      gate        g of the last step, min(1, Lref / max(L, 1e-4)); 1 while uncalibrated,
                  0 for a non-finite L
      mult        beta_eff / beta of the last step: g if enabled, 0 if frozen, else 1
      Vrec        the plain beta = .01 EMA of the same batch variance, kept for every run and
                  never gated or frozen (the diagnostic record for SNAfrz1 / NEO)
    """

    EXTRA = ("relu", "calibrated", "enabled", "frozen", "loss_ref", "gate", "mult", "Vrec")

    def __init__(self, arms: list[str], device, dtype=torch.float32):
        super().__init__([E.parse_arm("SNA")] * len(arms), device,
                         lo=ALPHA_LO, hi=ALPHA_HI, beta=BETA)
        R = self.R
        if any(a not in ARMS for a in arms):
            raise ValueError(f"arms must be in {ARMS}")
        self.dtype = dtype
        self.V = [v.to(dtype) for v in self.V]
        self.cval = [v.to(dtype) for v in self.cval]
        self.fixA = [v.to(dtype) for v in self.fixA]
        self.relu = torch.tensor([[a == "R"] for a in arms], device=device)
        self.calibrated = torch.zeros(R, 1, dtype=torch.bool, device=device)
        self.enabled = torch.zeros(R, 1, dtype=torch.bool, device=device)
        self.frozen = torch.zeros(R, 1, dtype=torch.bool, device=device)
        self.loss_ref = torch.ones(R, 1, dtype=dtype, device=device)
        self.gate = torch.ones(R, 1, dtype=dtype, device=device)
        self.mult = torch.ones(R, 1, dtype=dtype, device=device)
        self.Vrec = [torch.ones(R, w, dtype=dtype, device=device) for w in WIDTHS]
        self.any_relu = bool(self.relu.any())
        self.relu_conv = self.relu.repeat_interleave(CH, dim=0).view(1, R * CH, 1, 1)
        self.relu_fc = self.relu.view(R, 1, 1)

    def _mask(self, l: int) -> torch.Tensor:
        return self.relu_conv if IS_CONV[l] else self.relu_fc

    def phi(self, z, l):
        s = super().phi(z, l)
        return torch.where(self._mask(l), torch.relu(z), s) if self.any_relu else s

    def dphi(self, z, l):
        d = super().dphi(z, l)
        return torch.where(self._mask(l), (z > 0).to(z.dtype), d) if self.any_relu else d

    @torch.no_grad()
    def observe(self, loss: torch.Tensor) -> None:
        """g and beta_eff/beta from this step's pre-update CE (R,), before the V update."""
        L = loss.detach().reshape(self.R, 1)
        finite = torch.isfinite(L)
        safe = torch.where(finite, L, torch.ones_like(L)).clamp_min(LOSS_FLOOR)
        g = (self.loss_ref / safe).clamp(max=1.0)
        g = torch.where(self.calibrated, g, torch.ones_like(g))
        g = torch.where(finite, g, torch.zeros_like(g))
        self.gate.copy_(g)
        mult = torch.where(self.enabled, g, torch.ones_like(g))
        self.mult.copy_(torch.where(self.frozen, torch.zeros_like(mult), mult))

    @torch.no_grad()
    def update(self, zs) -> None:
        """V <- (1 - beta_eff) V + beta_eff Var(z), Var of THIS step's pre-update forward.
        Runs whose gate is off take the host's exact operation order."""
        rate = self.beta * self.mult
        for l, z in enumerate(zs):
            var = self.batch_var(z, l)
            self.Vrec[l].mul_(1 - self.beta).add_(self.beta * var)
            normal = self.V[l].clone().mul_(1 - self.beta).add_(self.beta * var)
            gated = self.V[l] * (1 - rate) + rate * var
            new = torch.where(self.enabled, gated, normal)
            self.V[l].copy_(torch.where(self.frozen, self.V[l], new))

    @torch.no_grad()
    def set_reference(self, refs: dict[int, float], control: list[int]) -> None:
        """refs: run -> Lref (already floored); control: the runs whose V the gate drives."""
        for r, ref in refs.items():
            self.loss_ref[r, 0] = float(ref)
            self.calibrated[r, 0] = True
        for r in control:
            if r not in refs:
                raise ValueError("a controlled run needs its own reference")
            self.enabled[r, 0] = True

    @torch.no_grad()
    def freeze_runs(self, runs: list[int]) -> None:
        """alpha of these runs fixed at its current value channel by channel, V fixed too."""
        super().freeze(runs)
        for r in runs:
            self.frozen[r, 0] = True

    def alpha_all(self) -> list[torch.Tensor]:
        return [self.alpha(l) for l in range(4)]

    def state(self) -> dict:
        st = super().state()
        for k in self.EXTRA:
            v = getattr(self, k)
            st[k] = [t.clone() for t in v] if isinstance(v, list) else [v.clone()]
        return st

    @torch.no_grad()
    def load_state(self, st: dict) -> None:
        if set(st) != set(super().state()) | set(self.EXTRA):
            raise ValueError("activation state keys do not match")
        for k in st:
            dst = getattr(self, k)
            dst = dst if isinstance(dst, list) else [dst]
            if len(dst) != len(st[k]) or any(d.shape != s.shape or d.dtype != s.dtype
                                             for d, s in zip(dst, st[k])):
                raise ValueError(f"activation state mismatch: {k}")
            for d, s in zip(dst, st[k]):
                d.copy_(s)                    # in place: a captured graph holds these


# --------------------------------------------------------------------------
# the engine: R runs in lockstep, one update captured as a CUDA graph
# --------------------------------------------------------------------------

REC_KEYS = ("ce", "hit", "gate", "mult")


class Engine:
    def __init__(self, slots, data: Data, device, graph: bool = True, lr: float = LR,
                 dtype=torch.float32):
        self.slots = [(str(a), int(s)) for a, s in slots]
        self.R = R = len(self.slots)
        if R == 0 or len(set(self.slots)) != R:
            raise ValueError("distinct (arm, seed) slots required")
        self.arms = [a for a, s in self.slots]
        self.seeds = [s for a, s in self.slots]
        self.data, self.device, self.lr, self.dtype = data, device, lr, dtype
        if data.dtype != dtype:
            raise ValueError("data and engine dtypes differ")
        per = {s: init_params(s) for s in sorted(set(self.seeds))}
        self.P = [torch.stack([per[s][i] for s in self.seeds]).to(device=device, dtype=dtype)
                  .requires_grad_(True) for i in range(10)]
        self.m = [torch.zeros_like(p) for p in self.P]
        self.v = [torch.zeros_like(p) for p in self.P]
        self.tc = torch.zeros((), dtype=torch.float64, device=device)
        self.b1 = torch.tensor(ADAM_B1, dtype=torch.float64, device=device)
        self.b2 = torch.tensor(ADAM_B2, dtype=torch.float64, device=device)
        self.act = NeoAct(self.arms, device, dtype=dtype)
        self.idx = torch.zeros(R, STEPS, BATCH, dtype=torch.long, device=device)
        self.j = torch.zeros((), dtype=torch.long, device=device)
        self.iota = torch.arange(STEPS, device=device)[None, :]
        self.rec = {k: torch.zeros(R, STEPS, dtype=dtype, device=device) for k in REC_KEYS}
        self.bad_step = torch.full((R,), -1, dtype=torch.long, device=device)
        self.graph = graph and device.type == "cuda"
        self.cg = None

    # one update on static tensors; the step index lives on the device
    def step(self) -> None:
        R, j = self.R, self.j
        idx = self.idx.index_select(1, j.view(1)).squeeze(1)            # (R, B)
        xb, yb = self.data.X[idx], self.data.Y[idx]
        out = E.forward(self.P, xb, self.act)
        logits = out[8]
        lossv = F.cross_entropy(logits.reshape(-1, N_CLASSES), yb.reshape(-1),
                                reduction="none").view(R, BATCH).mean(1)
        hit = (logits.detach().argmax(-1) == yb).to(self.dtype).mean(1)   # pre-update
        self.act.observe(lossv)
        grads = torch.autograd.grad(lossv.sum(), self.P)
        with torch.no_grad():
            ok = torch.isfinite(lossv)
            for gr in grads:
                ok = ok & torch.isfinite(gr).flatten(1).all(1)
            self.tc.add_(1)
            c1 = (1 - torch.pow(self.b1, self.tc)).to(self.dtype)
            c2 = (1 - torch.pow(self.b2, self.tc)).to(self.dtype)
            for p, gr, mi, vi in zip(self.P, grads, self.m, self.v):
                mi.mul_(ADAM_B1).add_(gr, alpha=1 - ADAM_B1)
                vi.mul_(ADAM_B2).addcmul_(gr, gr, value=1 - ADAM_B2)
                p.sub_(self.lr * (mi / c1) / ((vi / c2).sqrt() + ADAM_EPS))
            self.act.update([out[0].detach(), out[2].detach(), out[4].detach(), out[6].detach()])
            for V in self.act.V:
                ok = ok & torch.isfinite(V).all(1)
            self.bad_step.copy_(torch.where((self.bad_step < 0) & ~ok, j, self.bad_step))
            pos = self.iota == j
            for k, val in (("ce", lossv.detach()), ("hit", hit),
                           ("gate", self.act.gate[:, 0]), ("mult", self.act.mult[:, 0])):
                self.rec[k].copy_(torch.where(pos, val[:, None], self.rec[k]))
            j.add_(1)

    def _state_tensors(self):
        return (*self.P, *self.m, *self.v, self.tc, self.j, self.bad_step, *self.rec.values())

    def capture(self) -> None:
        """Warm up on a side stream, capture one update, then restore every tensor the warm-up
        and the capture touched (both execute the update)."""
        keep = [q.detach().clone() for q in self._state_tensors()]
        keep_act = self.act.state()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                self.step()
        torch.cuda.current_stream().wait_stream(side)
        self.cg = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.cg):
            self.step()
        with torch.no_grad():
            for q, v in zip(self._state_tensors(), keep):
                q.copy_(v)
        self.act.load_state(keep_act)
        torch.cuda.synchronize()

    def train_task(self, batches: torch.Tensor, n_steps: int = STEPS) -> None:
        """`n_steps` updates on (R, 780, 32) global rows (n_steps < 780 only in checks)."""
        if tuple(batches.shape) != (self.R, STEPS, BATCH):
            raise ValueError("batches must be (R, 780, 32)")
        self.idx.copy_(batches.to(self.device))
        self.j.zero_()
        self.bad_step.fill_(-1)
        for v in self.rec.values():
            v.zero_()
        if self.graph and self.cg is None:
            self.capture()
        for _ in range(n_steps):
            if self.cg is not None:
                self.cg.replay()
            else:
                self.step()
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def records(self) -> dict[str, np.ndarray]:
        return {k: v.detach().cpu().numpy().astype(np.float64) for k, v in self.rec.items()}

    def full_state(self) -> dict:
        return {"P": [p.detach().cpu().clone() for p in self.P],
                "m": [q.detach().cpu().clone() for q in self.m],
                "v": [q.detach().cpu().clone() for q in self.v],
                "tc": float(self.tc),
                "act": {k: [t.cpu().clone() for t in vs] for k, vs in self.act.state().items()},
                "slots": list(self.slots)}

    @torch.no_grad()
    def load_full_state(self, st: dict) -> None:
        if [tuple(s) for s in st["slots"]] != self.slots:
            raise ValueError("state belongs to other slots")
        for dst, src in ((self.P, st["P"]), (self.m, st["m"]), (self.v, st["v"])):
            for q, s in zip(dst, src):
                if q.shape != s.shape or q.dtype != s.dtype:
                    raise ValueError("parameter state mismatch")
                q.copy_(s)
        self.tc.fill_(st["tc"])
        self.act.load_state({k: [t.to(self.device) for t in vs] for k, vs in st["act"].items()})


# --------------------------------------------------------------------------
# evaluation and diagnostics (read only: no parameter, moment, V, Lref or gate is touched)
# --------------------------------------------------------------------------

@torch.no_grad()
def ce_acc(P, act, X, Y, rows: torch.Tensor, chunk: int = 100):
    """(R,) mean CE and accuracy (argmax over all 100 outputs) on X[rows[r]] for each run."""
    R, N = rows.shape
    ce = torch.zeros(R, dtype=torch.float64, device=X.device)
    hit = torch.zeros(R, dtype=torch.float64, device=X.device)
    for i0 in range(0, N, chunk):
        idx = rows[:, i0:i0 + chunk]
        logits = E.forward(P, X[idx], act)[8]
        yb = Y[idx]
        ce += F.cross_entropy(logits.reshape(-1, N_CLASSES), yb.reshape(-1),
                              reduction="none").view(R, -1).double().sum(1)
        hit += (logits.argmax(-1) == yb).sum(1)
    return (ce / N).cpu().numpy(), (hit / N).cpu().numpy()


@torch.no_grad()
def unit_stats(P, act, X, rows: torch.Tensor, Y=None, chunk: int = 50):
    """Per-unit statistics of z and phi'(z) at the four sites, for each run r on X[rows[r]].

    Sums are float64 and shifted by a per-unit reference (the first chunk's mean), so the
    variance does not cancel.  conv sites keep per-position sums, from which
      s          SD over images x positions        (the adaptation rule's granularity)
      s_within   sqrt(mean over positions of the across-image variance)
      s_between  SD over positions of the position means      (s^2 = within^2 + between^2)
    All SDs are population SDs.  Returns ({site: {name: (R, w) float64}}, acc (R,) | None).
    """
    R, N = rows.shape
    dev = X.device
    f64 = torch.float64
    S1, S2, D1, D2, DA, shift = [], [], [], [], [], [None] * 4
    for l in range(4):
        shp = (R, CH, CONV_HW[l], CONV_HW[l]) if IS_CONV[l] else (R, WIDTHS[l])
        S1.append(torch.zeros(shp, dtype=f64, device=dev))
        S2.append(torch.zeros(shp, dtype=f64, device=dev))
        for lst in (D1, D2, DA):
            lst.append(torch.zeros(R, WIDTHS[l], dtype=f64, device=dev))
    hits = torch.zeros(R, dtype=f64, device=dev)
    for i0 in range(0, N, chunk):
        idx = rows[:, i0:i0 + chunk]
        b = idx.shape[1]
        out = E.forward(P, X[idx], act)
        if Y is not None:
            hits += (out[8].argmax(-1) == Y[idx]).sum(1)
        for l in range(4):
            z = out[2 * l]
            d = act.dphi(z, l)
            if IS_CONV[l]:
                hw = z.shape[-1]
                zr = z.view(b, R, CH, hw, hw).double()
                if shift[l] is None:
                    shift[l] = zr.mean((0, 3, 4))[None, :, :, None, None]
                zr = zr - shift[l]
                S1[l] += zr.sum(0)
                S2[l] += (zr * zr).sum(0)
                dr = d.view(b, R, CH, hw, hw).double()
                D1[l] += dr.sum((0, 3, 4))
                D2[l] += (dr * dr).sum((0, 3, 4))
                DA[l] += dr.abs().sum((0, 3, 4))
            else:
                zr = z.double()                                  # (R, b, n)
                if shift[l] is None:
                    shift[l] = zr.mean(1, keepdim=True)
                zr = zr - shift[l]
                S1[l] += zr.sum(1)
                S2[l] += (zr * zr).sum(1)
                dr = d.double()
                D1[l] += dr.sum(1)
                D2[l] += (dr * dr).sum(1)
                DA[l] += dr.abs().sum(1)
    res = {}
    for l, tag in enumerate(SITES):
        s1, s2 = S1[l].cpu().numpy(), S2[l].cpu().numpy()
        if IS_CONV[l]:
            sh = shift[l].reshape(R, CH).cpu().numpy()
            n_all = N * s1.shape[2] * s1.shape[3]
            mc = s1.sum((2, 3)) / n_all                          # shifted mean
            var_all = np.maximum(s2.sum((2, 3)) / n_all - mc * mc, 0.0)
            pm = s1 / N
            var_within = np.maximum(s2 / N - pm * pm, 0.0).mean((2, 3))
            var_between = ((pm - mc[:, :, None, None]) ** 2).mean((2, 3))
            st = {"m": mc + sh, "s": np.sqrt(var_all), "s_within": np.sqrt(var_within),
                  "s_between": np.sqrt(var_between)}
            nd = n_all
        else:
            sh = shift[l].reshape(R, -1).cpu().numpy()
            mc = s1 / N
            st = {"m": mc + sh, "s": np.sqrt(np.maximum(s2 / N - mc * mc, 0.0))}
            nd = N
        dm = D1[l].cpu().numpy() / nd
        st["dmean"] = dm
        st["dsd"] = np.sqrt(np.maximum(D2[l].cpu().numpy() / nd - dm * dm, 0.0))
        st["dabs"] = DA[l].cpu().numpy() / nd
        with np.errstate(divide="ignore", invalid="ignore"):
            st["r"] = np.where(st["s"] > 0, st["m"] / st["s"], np.nan)
            if IS_CONV[l]:
                st["r_within"] = np.where(st["s_within"] > 0, st["m"] / st["s_within"], np.nan)
        res[tag] = st
    acc = (hits / N).cpu().numpy() if Y is not None else None
    return res, acc


def control_arrays(act: NeoAct) -> dict[str, list[np.ndarray]]:
    return {"alpha": [a.detach().double().cpu().numpy() for a in act.alpha_all()],
            "V": [v.detach().double().cpu().numpy() for v in act.V],
            "Vrec": [v.detach().double().cpu().numpy() for v in act.Vrec]}


def weight_row_norms(P) -> dict[str, np.ndarray]:
    out = {}
    for i, tag in enumerate(WTAGS):
        W = P[2 * i].detach().double()
        out[tag] = W.reshape(W.shape[0], W.shape[1], -1).norm(dim=2).cpu().numpy()   # (R, out)
    return out


def agg(x: np.ndarray, prefix: str) -> dict:
    """median (numpy: mean of the middle two for an even count), quartiles, of one run's units."""
    x = np.asarray(x, dtype=np.float64)
    if not np.isfinite(x).any():
        return {f"{prefix}_med": None, f"{prefix}_q25": None, f"{prefix}_q75": None}
    q25, q75 = np.nanquantile(x, [0.25, 0.75])
    return {f"{prefix}_med": float(np.nanmedian(x)), f"{prefix}_q25": float(q25),
            f"{prefix}_q75": float(q75)}


# --------------------------------------------------------------------------
# calibration (spec §3.1)
# --------------------------------------------------------------------------

def reference_loss(ce_task1: np.ndarray) -> dict:
    """Lref from one run's 780 pre-update CEs of task 1: the median of updates 703..780
    (78 values; the mean of the middle two after sorting), floored at 1e-4.  float64."""
    ce = np.asarray(ce_task1, dtype=np.float64)
    if ce.shape != (STEPS,) or not np.isfinite(ce).all() or (ce < 0).any():
        raise ValueError("calibration needs 780 finite non-negative CEs")
    w = np.sort(ce[STEPS - REF_LAST:])
    raw = 0.5 * (w[REF_LAST // 2 - 1] + w[REF_LAST // 2])
    return {"window_steps": [STEPS - REF_LAST + 1, STEPS], "ce_window": ce[STEPS - REF_LAST:].tolist(),
            "lref_raw": float(raw), "lref": float(max(raw, LOSS_FLOOR)),
            "floor_hit": bool(raw < LOSS_FLOOR)}


# --------------------------------------------------------------------------
# provenance helpers
# --------------------------------------------------------------------------

def git(*args) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True,
                          timeout=60).stdout.strip()


def git_state() -> dict:
    return {"git_hash": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty_src_analysis_specs": [l for l in git("status", "--porcelain", "--", "src",
                                                          "analysis", "specs").splitlines() if l],
            "time": time.strftime("%Y-%m-%d %H:%M:%S %z")}


def source_hashes() -> dict:
    out = {}
    for name, mod in sorted(sys.modules.items()):
        f = getattr(mod, "__file__", None)
        if f and Path(f).resolve().is_relative_to(ROOT / "src"):
            p = Path(f).resolve()
            out[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def env_info(device) -> dict:
    return {"python": sys.version.split()[0], "platform": platform.platform(),
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "numpy": np.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
            "torch_threads": torch.get_num_threads(), **precision_state()}


def save_atomic(obj, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    os.replace(tmp, path)


def write_csv(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    H.write_csv(tmp, rows)
    os.replace(tmp, path)


# --------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------

def last_hard(n_tasks: int) -> int:
    return max(t for t in range(1, n_tasks + 1) if t % 2 == 1)


def prepare(seeds, data: Data) -> dict:
    """Everything random about the protocol, drawn before training (spec §4)."""
    plans = {s: task_plan(s) for s in seeds}
    batches = {s: make_batches(s, plans[s], data.tr_rows) for s in seeds}
    probe = probe_ids(data.tr_rows)
    return {"plans": plans, "batches": batches, "probe": probe,
            "batch_sha256": {s: [sha(batches[s][t]) for t in range(N_TASKS)] for s in seeds},
            "probe_sha256": sha(probe)}


def slot_rows(slots, plans, t: int, rows_by_class) -> torch.Tensor:
    """(R, N) rows of task t's classes (train or test) in class then file order."""
    return torch.stack([torch.cat([rows_by_class[c] for c in plans[s][t - 1][1]])
                        for a, s in slots])


def run(slots, out: Path, n_tasks: int, device, graph: bool = True, resume: bool = True,
        data: Data | None = None, log=None, allow_dirty: bool = False, chunk: int = 50,
        stop_after: int | None = None) -> dict:
    """Train the bundle over tasks 1..n_tasks, then the fresh control on the last hard task.

    `stop_after` (checks only) raises after that task's state is saved, to test resume."""
    log = log or (lambda m: print(m, flush=True))
    out = Path(out)
    t_wall = time.time()
    gstart = git_state()
    if gstart["dirty_src_analysis_specs"] and not allow_dirty:
        raise SystemExit("src/analysis/specs have uncommitted changes; commit before a registered run")
    data = data or Data(device)
    seeds = sorted({s for a, s in slots})
    prep = prepare(seeds, data)
    plans, probe = prep["plans"], prep["probe"].to(device)
    eng = Engine(slots, data, device, graph=graph)
    R = eng.R
    meta = {"schema": 1, "run_id": EXPERIMENT, "slots": [list(s) for s in eng.slots],
            "n_tasks": n_tasks, "steps": STEPS, "batch": BATCH, "lr": LR, "graph": eng.graph,
            "c": C_SNAKE, "beta": BETA, "alpha_lo": ALPHA_LO, "alpha_hi": ALPHA_HI,
            "loss_floor": LOSS_FLOOR, "ref_last": REF_LAST,
            "spec_sha256": hashlib.sha256(SPEC.read_bytes()).hexdigest(),
            "data_sha256": data.sha256, "probe_sha256": prep["probe_sha256"],
            "batch_sha256": {str(s): v for s, v in prep["batch_sha256"].items()}}
    out.mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir(exist_ok=True)
    (out / "units").mkdir(exist_ok=True)
    st_path = out / "state.pt"
    T = n_tasks
    book = {"rows": [], "blocks": [], "excerpt": [], "site": [], "alpha": [], "calibration": {},
            "steps": {k: np.zeros((R, T, STEPS), dtype=np.float32) for k in REC_KEYS},
            "timing": [], "gits": [gstart], "resumed": []}
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
        save_atomic({"meta": meta, "state": eng.full_state(), "task": 0},
                    out / "checkpoints" / "t00.pt")
        save_atomic({"meta": meta, "state": eng.full_state(), "task": 0, "book": book}, st_path)
        write_json(out / "task_plan.json",
                   {str(s): [{"task": t, "hard": bool(h), "classes": c}
                             for t, (h, c) in enumerate(plans[s], 1)] for s in seeds})
        write_json(out / "batch_hashes.json",
                   {"rule": "sha256 of the (780, 32) int64 global train rows of each task; the "
                            "same tensor feeds every arm of a seed",
                    "per_seed": {str(s): v for s, v in prep["batch_sha256"].items()},
                    "per_slot": {f"{a}_seed{s}": prep["batch_sha256"][s] for a, s in eng.slots}})
        write_json(out / "probe_ids.json",
                   {"role": PROBE_ROLE, "seed": PROBE_SEED, "per_class": PROBE_PER_CLASS,
                    "rows": prep["probe"].tolist(),
                    "classes": data.cifar.train_y[prep["probe"]].tolist(),
                    "sha256": prep["probe_sha256"],
                    "note": "training images used only for read-only geometry; never for an "
                            "update or for calibration"})
    prov = {"run_id": EXPERIMENT, "status": "RUNNING", "meta": meta, "env": env_info(device),
            "spec": str(SPEC.relative_to(ROOT)), "spec_commit_registered": SPEC_COMMIT,
            "gits": book["gits"], "resumed": book["resumed"], "sources_sha256": source_hashes(),
            "engine": ("grouped conv / bmm bundle of all arms, one update captured as a CUDA "
                       "graph" if eng.graph else "grouped conv / bmm bundle of all arms, eager"),
            "scientific_verdict": None}
    write_json(out / "provenance.json", prov)

    # reserve the diagnostics' peak memory before the first task (read only)
    unit_stats(eng.P, eng.act, data.X, probe[None].expand(R, -1)[:, :chunk], chunk=chunk)
    for t in range(t0, n_tasks + 1):
        tt = time.time()
        hard = t % 2 == 1
        batches = torch.stack([prep["batches"][s][t - 1] for a, s in eng.slots]).to(device)
        tr = slot_rows(eng.slots, plans, t, data.tr_rows).to(device)
        te = slot_rows(eng.slots, plans, t, data.te_rows).to(device)
        sce, sacc = ce_acc(eng.P, eng.act, data.X, data.Y, tr)
        ctl0 = control_arrays(eng.act)
        start_state = eng.full_state()
        t_train = time.time()
        eng.train_task(batches)
        t_train = time.time() - t_train
        rec = eng.records()
        bad = eng.bad_step.cpu().numpy()
        finite = np.array([all(bool(torch.isfinite(p[r]).all()) for p in eng.P) for r in range(R)])
        if (bad >= 0).any() or not finite.all():
            save_atomic({"meta": meta, "task_start_state": start_state, "end_state": eng.full_state(),
                         "task": t, "bad_step": bad.tolist(), "params_finite": finite.tolist()},
                        out / "failed_state.pt")
            prov["status"] = "FAILED_NONFINITE"
            prov["failure"] = {"task": t, "bad_step_in_task": bad.tolist(),
                               "params_finite": finite.tolist(),
                               "slots": [list(s) for s in eng.slots]}
            write_json(out / "provenance.json", prov)
            raise FloatingPointError(f"non-finite state in task {t}; saved failed_state.pt")
        for k in REC_KEYS:
            book["steps"][k][:, t - 1] = rec[k]
        # ---- read-only evaluation
        t_diag = time.time()
        tr_stats, tr_acc = unit_stats(eng.P, eng.act, data.X, tr, data.Y, chunk=chunk)
        pr_stats, _ = unit_stats(eng.P, eng.act, data.X, probe[None].expand(R, -1), chunk=chunk)
        _, te_acc = ce_acc(eng.P, eng.act, data.Xte, data.Yte, te)
        ctl1 = control_arrays(eng.act)
        wn = weight_row_norms(eng.P)
        t_diag = time.time() - t_diag
        units = {}
        for inp, stats in (("train", tr_stats), ("probe", pr_stats)):
            for tag in SITES:
                for k, v in stats[tag].items():
                    units[f"{inp}_{tag}_{k}"] = v.astype(np.float32)
        for l, tag in enumerate(SITES):
            for k in ("alpha", "V", "Vrec"):
                units[f"ctl_{tag}_{k}_start"] = ctl0[k][l].astype(np.float32)
                units[f"ctl_{tag}_{k}_end"] = ctl1[k][l].astype(np.float32)
        for tag, v in wn.items():
            units[f"wnorm_{tag}"] = v.astype(np.float32)
        np.savez(out / "units" / f"t{t:02d}.npz", **units)
        # ---- rows
        for r, (arm, seed) in enumerate(eng.slots):
            snake = arm != "R"
            ce, hit, g, mu = rec["ce"][r], rec["hit"][r], rec["gate"][r], rec["mult"][r]
            cls = plans[seed][t - 1][1]
            row = {"arm": arm, "seed": seed, "slot": r, "task": t, "hard": int(hard),
                   "n_classes": len(cls), "classes": " ".join(map(str, cls)),
                   "online_acc": float(hit.mean()), "first100_online_acc": float(hit[:FIRST_N].mean()),
                   "train_acc": float(tr_acc[r]), "test_acc": float(te_acc[r]),
                   "switch_ce": float(sce[r]), "switch_acc": float(sacc[r]),
                   "ce_step1": float(ce[0]), "ce_first10": float(ce[:10].mean()),
                   "ce_last10": float(ce[-10:].mean()), "ce_mean": float(ce.mean()),
                   "gate_mean": float(g.mean()), "gate_min": float(g.min()), "gate_max": float(g.max()),
                   "gate_first10": float(g[:10].mean()), "gate_last10": float(g[-10:].mean()),
                   "mult_mean": float(mu.mean()), "sum_beta_eff": float(BETA * mu.sum()),
                   "gate_enabled": bool(eng.act.enabled[r, 0]),
                   "alpha_frozen": bool(eng.act.frozen[r, 0]),
                   "loss_ref": float(eng.act.loss_ref[r, 0]) if bool(eng.act.calibrated[r, 0]) else None,
                   "diverged": False, "bad_step": int(bad[r])}
            for l, tag in enumerate(SITES):
                a1 = ctl1["alpha"][l][r]
                row[f"alpha_med_{tag}"] = float(np.median(a1)) if snake else None
                row[f"two_alpha_sqrtV_med_{tag}"] = (float(np.median(2 * a1 * np.sqrt(ctl1["V"][l][r])))
                                                    if snake else None)
                row[f"zsd_med_{tag}"] = float(np.median(tr_stats[tag]["s"][r]))
                row[f"zbar_med_{tag}"] = float(np.median(tr_stats[tag]["m"][r]))
                row[f"dmean_med_{tag}"] = float(np.median(tr_stats[tag]["dmean"][r]))
            book["rows"].append(row)
            for b in range(STEPS // BLOCK):
                sl = slice(b * BLOCK, (b + 1) * BLOCK)
                book["blocks"].append({
                    "arm": arm, "seed": seed, "task": t, "block": b + 1,
                    "step_lo": b * BLOCK + 1, "step_hi": (b + 1) * BLOCK,
                    "ce_mean": float(ce[sl].mean()), "hit_mean": float(hit[sl].mean()),
                    "g_mean": float(g[sl].mean()), "g_min": float(g[sl].min()),
                    "g_max": float(g[sl].max()), "mult_mean": float(mu[sl].mean()),
                    "sum_beta_eff": float(BETA * mu[sl].sum()),
                    "gate_enabled": bool(eng.act.enabled[r, 0])})
            for j in list(range(10)) + list(range(STEPS - 10, STEPS)):
                book["excerpt"].append({"arm": arm, "seed": seed, "task": t, "step": j + 1,
                                        "ce": float(ce[j]), "hit": float(hit[j]),
                                        "g": float(g[j]), "mult": float(mu[j])})
            for inp, stats, n_img in (("train", tr_stats, tr.shape[1]),
                                      ("probe", pr_stats, probe.shape[0])):
                for l, tag in enumerate(SITES):
                    s = stats[tag]
                    q = {"arm": arm, "seed": seed, "task": t, "input": inp, "site": tag,
                         "n_units": WIDTHS[l], "n_images": int(n_img),
                         "n_positions": CONV_HW[l] ** 2 if IS_CONV[l] else 1}
                    names = ["m", "s", "r", "dmean", "dsd", "dabs"] + (
                        ["s_within", "s_between", "r_within"] if IS_CONV[l] else [])
                    for k in names:
                        q.update(agg(s[k][r], k))
                    q["r_undefined"] = int((~(s["s"][r] > 0)).sum())
                    a1 = ctl1["alpha"][l][r]
                    if snake:
                        q.update(agg(a1, "alpha"))
                        q.update(agg(2 * a1 * s["s"][r], "two_alpha_s"))
                        if IS_CONV[l]:
                            q.update(agg(2 * a1 * s["s_within"][r], "two_alpha_s_within"))
                        q.update(agg(2 * a1 * np.sqrt(ctl1["V"][l][r]), "two_alpha_sqrtV"))
                        q.update(agg(np.sqrt(ctl1["Vrec"][l][r]), "sqrtVrec"))
                    q.update(agg(wn[WTAGS[l]][r], "wnorm"))
                    book["site"].append(q)
            if snake:
                for l, tag in enumerate(SITES):
                    a0, a1 = ctl0["alpha"][l][r], ctl1["alpha"][l][r]
                    book["alpha"].append({
                        "arm": arm, "seed": seed, "task": t, "site": tag,
                        **agg(a0, "alpha_start"), **agg(a1, "alpha_end"),
                        **agg(np.abs(a1 - a0) / a0, "alpha_relchange"),
                        "alpha_frac_changed": float((a1 != a0).mean()),
                        "alpha_at_lo_start": int((a0 <= ALPHA_LO).sum()),
                        "alpha_at_hi_start": int((a0 >= ALPHA_HI).sum()),
                        "alpha_at_lo_end": int((a1 <= ALPHA_LO).sum()),
                        "alpha_at_hi_end": int((a1 >= ALPHA_HI).sum()),
                        "n_units": WIDTHS[l],
                        **agg(np.sqrt(ctl1["V"][l][r]), "sqrtV_end"),
                        **agg(np.sqrt(ctl1["Vrec"][l][r]), "sqrtVrec_end"),
                        "V_frozen": bool(eng.act.frozen[r, 0]),
                        "V_changed_frac": float((ctl1["V"][l][r] != ctl0["V"][l][r]).mean())})
        # ---- task-1 boundary: calibration (all Snake runs; control only NEO), freeze SNAfrz1
        if t == 1:
            refs, control, frz = {}, [], []
            for r, (arm, seed) in enumerate(eng.slots):
                if arm in SNAKE_ARMS:
                    c = reference_loss(rec["ce"][r])
                    c.update({"arm": arm, "seed": seed, "used_for_control": arm == "NEO",
                              "lref_float_used": float(np.float32(c["lref"]))})
                    book["calibration"][f"{arm}_seed{seed}"] = c
                    refs[r] = c["lref"]
                    if arm == "NEO":
                        control.append(r)
                if arm == "SNAfrz1":
                    frz.append(r)
            eng.act.set_reference(refs, control)
            eng.act.freeze_runs(frz)
            for r, (arm, seed) in enumerate(eng.slots):
                if arm in SNAKE_ARMS:
                    c = book["calibration"][f"{arm}_seed{seed}"]
                    c["lref_device"] = float(eng.act.loss_ref[r, 0])
        book["timing"].append({"task": t, "train_s": t_train, "diag_s": t_diag,
                               "total_s": time.time() - tt})
        if t in CKPT_TASKS or t == n_tasks:
            save_atomic({"meta": meta, "state": eng.full_state(), "task": t},
                        out / "checkpoints" / f"t{t:02d}.pt")
        write_csv(out / "per_task.csv", book["rows"])
        save_atomic({"meta": meta, "state": eng.full_state(), "task": t, "book": book}, st_path)
        on = {}
        for q in book["rows"][-R:]:
            on.setdefault(q["arm"], []).append(q["online_acc"])
        log(f"[{time.strftime('%T')}] task {t:2d}/{n_tasks} {'hard' if hard else 'easy'} "
            f"train {t_train:5.1f}s diag {t_diag:4.1f}s  "
            + "  ".join(f"{a} {np.mean(v):.4f}" for a, v in on.items()))
        if stop_after is not None and t == stop_after:
            raise KeyboardInterrupt(f"stop_after={t} (check hook)")

    # ---- fresh control on the last hard task (spec §4): initial state, the same 780 batches
    tf = last_hard(n_tasks)
    tt = time.time()
    fresh = Engine(eng.slots, data, device, graph=graph)
    batches = torch.stack([prep["batches"][s][tf - 1] for a, s in eng.slots]).to(device)
    fresh.train_task(batches)
    frec = fresh.records()
    fbad = fresh.bad_step.cpu().numpy()
    tr = slot_rows(eng.slots, plans, tf, data.tr_rows).to(device)
    te = slot_rows(eng.slots, plans, tf, data.te_rows).to(device)
    _, ftr = ce_acc(fresh.P, fresh.act, data.X, data.Y, tr)
    _, fte = ce_acc(fresh.P, fresh.act, data.Xte, data.Yte, te)
    fstate = fresh.full_state()
    cont = {(q["arm"], q["seed"]): q for q in book["rows"] if q["task"] == tf}
    fresh_rows = []
    for r, (arm, seed) in enumerate(eng.slots):
        c = cont[(arm, seed)]
        f_on = float(frec["hit"][r].mean())
        f_first = float(frec["hit"][r][:FIRST_N].mean())
        # the Snake fresh runs of a seed are the same computation (no calibration or freeze
        # inside a first task); record whether they came out bit-identical
        twin = [q for q, (a2, s2) in enumerate(eng.slots) if s2 == seed and a2 in SNAKE_ARMS]
        same = (all(all(torch.equal(fstate["P"][i][r], fstate["P"][i][q]) for i in range(10))
                    and np.array_equal(frec["hit"][r], frec["hit"][q])
                    and np.array_equal(frec["ce"][r], frec["ce"][q]) for q in twin)
                if arm in SNAKE_ARMS else None)
        fresh_rows.append({"arm": arm, "seed": seed, "task": tf,
                           "batch_sha256": prep["batch_sha256"][seed][tf - 1],
                           "fresh_online_acc": f_on, "fresh_first100_online_acc": f_first,
                           "fresh_train_acc": float(ftr[r]), "fresh_test_acc": float(fte[r]),
                           "continual_online_acc": c["online_acc"],
                           "continual_first100_online_acc": c["first100_online_acc"],
                           "continual_train_acc": c["train_acc"], "continual_test_acc": c["test_acc"],
                           "fresh_gap_online": f_on - c["online_acc"],
                           "fresh_gap_test": float(fte[r]) - c["test_acc"],
                           "fresh_bad_step": int(fbad[r]),
                           "snake_fresh_identical_across_arms": same})
    write_csv(out / "fresh_control.csv", fresh_rows)
    book["timing"].append({"task": f"fresh_t{tf}", "total_s": time.time() - tt})
    del fresh
    # ---- outputs
    write_csv(out / "per_block.csv", book["blocks"])
    write_csv(out / "step_excerpt.csv", book["excerpt"])
    write_csv(out / "site_stats.csv", book["site"])
    write_csv(out / "alpha_stats.csv", book["alpha"])
    write_json(out / "calibration.json",
               {"rule": "median of the pre-update CE of task-1 updates 703..780 (78 values, mean "
                        "of the middle two), floored at 1e-4; float64, used as float32",
                "runs": book["calibration"]})
    np.savez_compressed(out / "per_step.npz", **{k: v for k, v in book["steps"].items()},
                        **{f"fresh_{k}": v.astype(np.float32) for k, v in frec.items()},
                        slots=np.asarray([f"{a}_seed{s}" for a, s in eng.slots]))
    units = {}
    for t in range(1, n_tasks + 1):
        with np.load(out / "units" / f"t{t:02d}.npz") as z:
            for k in z.files:
                units.setdefault(k, []).append(z[k])
    np.savez_compressed(out / "unit_stats.npz", **{k: np.stack(v, axis=1) for k, v in units.items()},
                        slots=np.asarray([f"{a}_seed{s}" for a, s in eng.slots]))
    prov.update(status="COMPLETED", gits=book["gits"], resumed=book["resumed"],
                timing=book["timing"], wall_clock_s_this_process=time.time() - t_wall,
                cuda_max_mem_mb=(torch.cuda.max_memory_allocated() / 2 ** 20
                                 if device.type == "cuda" else None),
                updates={"continual": R * n_tasks * STEPS, "fresh": R * STEPS,
                         "fresh_note": "fresh run for every arm (Snake arms not shared)"},
                fresh_control={"task": tf, "rows": len(fresh_rows)},
                sources_sha256=source_hashes())
    write_json(out / "provenance.json", prov)
    write_json(out / "protocol.json", {"spec": str(SPEC.relative_to(ROOT)),
                                       "spec_sha256": meta["spec_sha256"],
                                       "spec_commit_registered": SPEC_COMMIT,
                                       "arms": list(ARMS), "seeds": seeds, "n_tasks": n_tasks,
                                       "late_window": list(LATE), "early_window": list(EARLY),
                                       "fresh_task": tf, "meta": meta})
    log(f"[{time.strftime('%T')}] done {out} ({(time.time() - t_wall) / 60:.1f} min)")
    return prov


def parse_seeds(s: str) -> list[int]:
    return E.parse_seeds(s)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["run"])
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--seeds", default="2000-2009")
    ap.add_argument("--tasks", type=int, default=N_TASKS)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true", help="smoke/check runs only")
    a = ap.parse_args()
    if not 1 <= a.tasks <= N_TASKS:
        raise SystemExit("1 <= tasks <= 30")
    arms = a.arms.split(",")
    if any(x not in ARMS for x in arms):
        raise SystemExit(f"arms from {ARMS}")
    device = setup(a.device)
    slots = [(arm, s) for arm in arms for s in parse_seeds(a.seeds)]
    run(slots, Path(a.out), a.tasks, device, graph=not a.no_graph, allow_dirty=a.allow_dirty)


if __name__ == "__main__":
    main()
