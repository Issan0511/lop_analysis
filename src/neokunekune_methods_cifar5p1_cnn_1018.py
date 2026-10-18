#!/usr/bin/env python3
"""neokunekune_methods_cifar5p1_cnn_1018 -- 12-arm method comparison on the real-label 5+1 CIFAR CNN.

    python3 src/neokunekune_methods_cifar5p1_cnn_1018.py smoke --out results/neokunekune_methods_cifar5p1_cnn_1018/smoke
    python3 src/neokunekune_methods_cifar5p1_cnn_1018.py calib --out results/neokunekune_methods_cifar5p1_cnn_1018/calib
    python3 src/neokunekune_methods_cifar5p1_cnn_1018.py main --out results/neokunekune_methods_cifar5p1_cnn_1018/main

Registered spec: specs/spec_neokunekune_methods_cifar5p1_cnn_1018.md (commit 170beea5).

The box, data, task plan, batch lists, init, probe, bundled forward, Snake / loss gate / freeze
and the CUDA-graph engine are the previous run's (`neokunekune_cifar5p1_cnn_1018`, imported and
not modified).  New here: a per-run lr in the Adam step, the arm kinds of the spec §3 (fixed
Snake, KKT1, ELU, leaky, RSL, ReLU + ReDo, ReLU + L2 Init) chosen per run inside the bundle,
per-run divergence (a non-finite run is recorded and the others go on), and the calibration /
main drivers.  A bundle holding only the previous four arms runs the previous operation
sequence (checked bit for bit against the previous smoke).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import neokunekune_cifar5p1_cnn_1018 as NK      # the previous run; not modified
from src import pmnist_0905 as H

E = NK.E
EXPERIMENT = "neokunekune_methods_cifar5p1_cnn_1018"
SPEC = ROOT / "specs" / f"spec_{EXPERIMENT}.md"
SPEC_COMMIT = "170beea5"
OUT_ROOT = ROOT / "results" / EXPERIMENT
PREV_RESULTS = ROOT / "results" / NK.EXPERIMENT

ARMS12 = ("R", "SNA", "SNAfrz1", "NEO", "SN06", "SN3", "KKT1", "ELU", "LK", "RSL", "ReDo", "L2I")
SNAKE_FAMILY = ("SNA", "SNAfrz1", "NEO", "SN06", "SN3", "KKT1")
KIND = {"R": "relu", "ReDo": "relu", "L2I": "relu", "SNA": "snake", "SNAfrz1": "snake", "NEO": "snake",
        "SN06": "snake", "SN3": "snake", "KKT1": "kkt1", "ELU": "elu", "LK": "leaky", "RSL": "rsl"}
BASE = {a: (a if a in NK.SNAKE_ARMS else "SNA" if KIND[a] in ("snake", "kkt1") else "R") for a in ARMS12}
FIXED_ALPHA = {"SN06": 0.6, "SN3": 3.0}
# spec §3/§4: grids in tie-break order (literature / previous value first)
SHAPES = {"ELU": [(3.6,), (1.0,)], "LK": [(0.4,), (0.1,), (0.8,)],
          "RSL": [(0.5, 0.5, 0.673, 2.673), (5.0, 3.0, 0.125, 0.333)],
          "ReDo": [(0.0, 1560), (0.025, 1000), (0.1, 1560)], "L2I": [(1e-2,), (1e-3,), (1e-1,)]}
SHAPE_KEYS = {"ELU": ("a",), "LK": ("a",), "RSL": ("c", "p", "l", "u"), "ReDo": ("t", "F"), "L2I": ("lam",)}
LR_UP, LR_DOWN = (1e-4, 3e-4, 1e-3), (1e-3, 3e-4, 1e-4)
LR_ORDER = {a: (LR_UP if (a == "R" or a in SNAKE_FAMILY) else LR_DOWN) for a in ARMS12}
CALIB_SEEDS = (100, 101, 102)
MAIN_SEEDS = NK.SEEDS
BUNDLE_MAX = 40
MAIN_BUNDLES = {"A": ("R", "SNA", "SNAfrz1", "NEO"), "B": ("SN06", "SN3", "KKT1", "ELU"),
                "C": ("LK", "RSL", "ReDo", "L2I")}
REDO_EPS = 1e-9
PI = math.pi
CH, IS_CONV, WIDTHS, SITES = NK.CH, NK.IS_CONV, NK.WIDTHS, NK.SITES
STEPS, BATCH, N_CLASSES = NK.STEPS, NK.BATCH, NK.N_CLASSES


# --------------------------------------------------------------------------
# arm configurations
# --------------------------------------------------------------------------

def cfg_name(arm: str, shape: tuple, lr: float) -> str:
    keys = SHAPE_KEYS.get(arm, ())
    if len(keys) != len(shape):
        raise ValueError(f"{arm}: shape {shape} does not match {keys}")
    return arm + "".join(f"_{k}{v:g}" for k, v in zip(keys, shape)) + f"_lr{lr:g}"


def parse_cfg(name: str) -> dict:
    parts = name.split("_")
    arm = parts[0]
    if arm not in ARMS12 or not parts[-1].startswith("lr"):
        raise ValueError(f"bad configuration name {name!r}")
    keys = SHAPE_KEYS.get(arm, ())
    vals = parts[1:-1]
    if len(vals) != len(keys) or any(not v.startswith(k) for v, k in zip(vals, keys)):
        raise ValueError(f"bad shape in {name!r}")
    shape = tuple(int(v[len(k):]) if k == "F" else float(v[len(k):]) for v, k in zip(vals, keys))
    lr = float(parts[-1][2:])
    cfg = {"name": name, "arm": arm, "kind": KIND[arm], "base": BASE[arm], "lr": lr, "shape": shape}
    if cfg_name(arm, shape, lr) != name:
        raise ValueError(f"non-canonical configuration name {name!r}")
    return cfg


def grid(arm: str) -> list[str]:
    """The arm's calibration grid in tie-break order: shape priority outside, lr inside."""
    return [cfg_name(arm, s, lr) for s in SHAPES.get(arm, [()]) for lr in LR_ORDER[arm]]


def first_cfg(arm: str) -> str:
    return grid(arm)[0]


# --------------------------------------------------------------------------
# the bundled activation of all twelve arms
# --------------------------------------------------------------------------

class MethodsAct(NK.NeoAct):
    """NeoAct (Snake / ReLU / loss gate / freeze) plus fixed Snake, KKT1, ELU, leaky and RSL,
    chosen per run with torch.where.  Kinds absent from the bundle build no branch, so a bundle
    of the previous four arms evaluates exactly NeoAct's operations."""

    def __init__(self, cfgs: list[dict], device, dtype=torch.float32):
        super().__init__([c["base"] for c in cfgs], device, dtype=dtype)
        R = self.R
        self.cfgs = cfgs
        kinds = [c["kind"] for c in cfgs]
        self.present = set(kinds)
        for r, c in enumerate(cfgs):
            if c["arm"] in FIXED_ALPHA:
                for l in range(4):
                    self.ada[l][r] = False
                    self.fixA[l][r] = FIXED_ALPHA[c["arm"]]

        def col(vals):
            return torch.tensor([[float(v)] for v in vals], dtype=dtype, device=device)

        def per_site(t, conv_dtype=None):
            return [t.repeat_interleave(CH, dim=0).view(1, R * CH, 1, 1) if IS_CONV[l] else t.view(R, 1, 1)
                    for l in range(4)]

        mask = {k: torch.tensor([[x == k] for x in kinds], device=device) for k in
                ("relu", "kkt1", "elu", "leaky", "rsl")}
        self.mask = {k: per_site(v) for k, v in mask.items()}
        sh = lambda arm, i, default: col([c["shape"][i] if c["arm"] == arm else default for c in cfgs])
        self.elu_a = per_site(sh("ELU", 0, 1.0))
        self.lk_a = per_site(sh("LK", 0, 0.0))
        rc, rp, rl, ru = (sh("RSL", i, d) for i, d in enumerate((1.0, 1.0, 0.0, 0.0)))
        self.rsl_c, self.rsl_p = per_site(rc), per_site(rp)
        self.rsl_l, self.rsl_w = per_site(rl), per_site(ru - rl)
        self.rsl_mid = per_site((rl + ru) / 2)
        self.train_mode = False
        self.noise = None                         # [4] buffers of the training batch shapes (engine)
        self.segments = []                        # (kind, r0, r1): maximal runs of one kind
        for r, k in enumerate(kinds):
            if self.segments and self.segments[-1][0] == k:
                self.segments[-1] = (k, self.segments[-1][1], r + 1)
            else:
                self.segments.append((k, r, r + 1))

    def _rsl(self, z, l, r):
        sig = torch.sigmoid(self.rsl_c[l] * (z / self.rsl_p[l]))
        return r * z + (1.0 - r) * z * sig

    def _seg(self, t, l, r0, r1):
        """A per-site (conv (1, R*C, 1, 1) / fc (R, 1, 1)) parameter tensor cut to runs r0..r1."""
        return t[l][:, r0 * CH:r1 * CH] if IS_CONV[l] else t[l][r0:r1]

    def _phi_kind(self, k, z, l, r0, r1):
        if k == "relu":
            return torch.relu(z)
        if k in ("snake", "kkt1"):
            a = self.alpha(l)[r0:r1]
            a = a.reshape(1, -1, 1, 1) if IS_CONV[l] else a[:, None, :]
            s = z + torch.sin(a * z) ** 2 * a.reciprocal()          # the host's op order
            if k == "snake":
                return s
            th = 2.0 * a * z
            return torch.where(th < -2.0 * PI, z, torch.where(th > PI, z + a.reciprocal(), s))
        if k == "elu":
            return torch.where(z > 0, z, self._seg(self.elu_a, l, r0, r1) * F.elu(z, 1.0))
        if k == "leaky":
            return torch.where(z > 0, z, self._seg(self.lk_a, l, r0, r1) * z)
        if k == "rsl":
            if self.train_mode:
                if self.noise is None:
                    raise RuntimeError("RSL training forward needs the engine's noise buffers")
                nz = self.noise[l][:, r0 * CH:r1 * CH] if IS_CONV[l] else self.noise[l][r0:r1]
                if nz.shape != z.shape:
                    raise RuntimeError("RSL noise buffer does not match the training batch")
                r = self._seg(self.rsl_l, l, r0, r1) + self._seg(self.rsl_w, l, r0, r1) * nz
            else:
                r = self._seg(self.rsl_mid, l, r0, r1)
            sig = torch.sigmoid(self._seg(self.rsl_c, l, r0, r1) * (z / self._seg(self.rsl_p, l, r0, r1)))
            return r * z + (1.0 - r) * z * sig
        raise ValueError(k)

    def phi(self, z, l):
        if self.present <= {"snake", "relu"}:
            # the previous NeoAct's operations, so a bundle of the previous arms is bit-identical
            s = E.BundleSnake.phi(self, z, l)
            return torch.where(self.mask["relu"][l], torch.relu(z), s) if "relu" in self.present else s
        # every kind only on its own runs: split the run axis into same-kind segments
        dim = 1 if IS_CONV[l] else 0
        sizes = [(r1 - r0) * (CH if IS_CONV[l] else 1) for k, r0, r1 in self.segments]
        parts = torch.split(z, sizes, dim=dim)
        return torch.cat([self._phi_kind(k, zp, l, r0, r1)
                          for (k, r0, r1), zp in zip(self.segments, parts)], dim=dim)

    def dphi(self, z, l):
        """phi' in evaluation mode (RSL at the midpoint): the diagnostics' derivative."""
        d = E.BundleSnake.dphi(self, z, l)
        if "kkt1" in self.present:
            th = 2.0 * self._shape(l, self.alpha(l)) * z
            inside = (th >= -2.0 * PI) & (th <= PI)
            d = torch.where(self.mask["kkt1"][l], torch.where(inside, d, torch.ones_like(d)), d)
        if "relu" in self.present:
            d = torch.where(self.mask["relu"][l], (z > 0).to(z.dtype), d)
        if "elu" in self.present:
            de = torch.where(z > 0, torch.ones_like(z), self.elu_a[l] * torch.exp(torch.clamp(z, max=0.0)))
            d = torch.where(self.mask["elu"][l], de, d)
        if "leaky" in self.present:
            dl = torch.where(z > 0, torch.ones_like(z), self.lk_a[l].expand_as(z))
            d = torch.where(self.mask["leaky"][l], dl, d)
        if "rsl" in self.present:
            r = self.rsl_mid[l]
            sg = torch.sigmoid(self.rsl_c[l] * (z / self.rsl_p[l]))
            dr = r + (1.0 - r) * (sg + (self.rsl_c[l] / self.rsl_p[l]) * z * sg * (1.0 - sg))
            d = torch.where(self.mask["rsl"][l], dr, d)
        return d


# --------------------------------------------------------------------------
# the engine: per-run lr, L2 Init, RSL noise, ReDo, per-run divergence
# --------------------------------------------------------------------------

def stream_seed(*parts) -> int:
    h = hashlib.sha256("|".join(map(str, (EXPERIMENT,) + parts)).encode()).digest()
    return int.from_bytes(h[:8], "little") & ((1 << 63) - 1)


class MethodsEngine(NK.Engine):
    def __init__(self, slots, data: NK.Data, device, graph: bool = True, dtype=torch.float32,
                 stage: str = "train"):
        self.slots = [(str(a), int(s)) for a, s in slots]
        self.R = R = len(self.slots)
        if R == 0 or len(set(self.slots)) != R:
            raise ValueError("distinct (configuration, seed) slots required")
        self.cfgs = [parse_cfg(a) for a, s in self.slots]
        self.arms = [c["base"] for c in self.cfgs]
        self.seeds = [s for a, s in self.slots]
        self.data, self.device, self.dtype, self.stage = data, device, dtype, stage
        if data.dtype != dtype:
            raise ValueError("data and engine dtypes differ")
        per = {s: NK.init_params(s) for s in sorted(set(self.seeds))}
        self.P = [torch.stack([per[s][i] for s in self.seeds]).to(device=device, dtype=dtype)
                  .requires_grad_(True) for i in range(10)]
        self.m = [torch.zeros_like(p) for p in self.P]
        self.v = [torch.zeros_like(p) for p in self.P]
        self.tc = torch.zeros((), dtype=torch.float64, device=device)
        self.b1 = torch.tensor(NK.ADAM_B1, dtype=torch.float64, device=device)
        self.b2 = torch.tensor(NK.ADAM_B2, dtype=torch.float64, device=device)
        self.act = MethodsAct(self.cfgs, device, dtype=dtype)
        self.idx = torch.zeros(R, STEPS, BATCH, dtype=torch.long, device=device)
        self.j = torch.zeros((), dtype=torch.long, device=device)
        self.iota = torch.arange(STEPS, device=device)[None, :]
        self.rec = {k: torch.zeros(R, STEPS, dtype=dtype, device=device) for k in NK.REC_KEYS}
        self.bad_step = torch.full((R,), -1, dtype=torch.long, device=device)
        self.graph = graph and device.type == "cuda"
        self.cg = None
        self.lr = None                                         # per run, below
        lr_t = torch.tensor([c["lr"] for c in self.cfgs], dtype=dtype, device=device)
        self.lr_b = [lr_t.view(R, *([1] * (p.dim() - 1))) for p in self.P]
        lam = [c["shape"][0] if c["arm"] == "L2I" else 0.0 for c in self.cfgs]
        self.l2_any = any(x > 0 for x in lam)
        if self.l2_any:
            l2 = torch.tensor([2.0 * x for x in lam], dtype=dtype, device=device)
            self.l2_b = [l2.view(R, *([1] * (p.dim() - 1))) for p in self.P]
            self.P0 = [p.detach().clone() for p in self.P]
        # RSL: per-run CUDA generators, buffers of the training batch shapes
        self.rsl_runs = [r for r, c in enumerate(self.cfgs) if c["kind"] == "rsl"]
        self.rsl_gen = {}
        if self.rsl_runs:
            shapes = [(BATCH, R * CH, 32, 32), (BATCH, R * CH, 16, 16), (R, BATCH, WIDTHS[2]), (R, BATCH, WIDTHS[3])]
            self.act.noise = [torch.zeros(s, dtype=dtype, device=device) for s in shapes]
            self._tmp = [torch.zeros((BATCH, CH, 32, 32), dtype=dtype, device=device),
                         torch.zeros((BATCH, CH, 16, 16), dtype=dtype, device=device)]
            for r in self.rsl_runs:
                g = torch.Generator(device=device)
                g.manual_seed(stream_seed("rsl_noise", stage, self.slots[r][0], self.slots[r][1]))
                self.rsl_gen[r] = g
        # ReDo: per-run tau / period and cpu reinit streams
        self.redo = {r: (float(c["shape"][0]), int(c["shape"][1])) for r, c in enumerate(self.cfgs)
                     if c["arm"] == "ReDo"}
        self.redo_gen = {r: H.stream(f"neo_m_redo_reinit|{stage}|{self.slots[r][0]}", self.slots[r][1])
                         for r in self.redo}
        self.n_done = 0                                        # updates done by this engine
        self.n_reset = np.zeros((R, 4), dtype=np.int64)        # per task, reset by the caller
        self.redo_log: list[dict] = []

    # one update: the previous step with a per-run lr and the L2 Init gradient
    def step(self) -> None:
        R, j = self.R, self.j
        self.act.train_mode = True
        try:
            idx = self.idx.index_select(1, j.view(1)).squeeze(1)
            xb, yb = self.data.X[idx], self.data.Y[idx]
            out = E.forward(self.P, xb, self.act)
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
            self.bad_step.copy_(torch.where((self.bad_step < 0) & ~ok, j, self.bad_step))
            pos = self.iota == j
            for k, val in (("ce", lossv.detach()), ("hit", hit),
                           ("gate", self.act.gate[:, 0]), ("mult", self.act.mult[:, 0])):
                self.rec[k].copy_(torch.where(pos, val[:, None], self.rec[k]))
            j.add_(1)

    @torch.no_grad()
    def refresh_noise(self) -> None:
        """U(0,1) for every RSL run's slice of the four training-shape buffers (outside the graph)."""
        for r in self.rsl_runs:
            g = self.rsl_gen[r]
            for l in range(4):
                buf = self.act.noise[l]
                if IS_CONV[l]:
                    tmp = self._tmp[l]
                    torch.rand(tmp.shape, generator=g, out=tmp)
                    buf[:, r * CH:(r + 1) * CH].copy_(tmp)
                else:
                    torch.rand(buf[r].shape, generator=g, out=buf[r])

    def train_task(self, batches: torch.Tensor, n_steps: int = STEPS) -> list[int]:
        """n_steps updates; ReDo checks run after the update they are due on.  Returns the runs
        whose check fell on the task's last update (applied by the caller after evaluation)."""
        if tuple(batches.shape) != (self.R, STEPS, BATCH):
            raise ValueError("batches must be (R, 780, 32)")
        self.idx.copy_(batches.to(self.device))
        self.j.zero_()
        self.bad_step.fill_(-1)
        for v in self.rec.values():
            v.zero_()
        if self.graph and self.cg is None:
            self.capture()
        pending = []
        for jj in range(n_steps):
            if self.rsl_runs:
                self.refresh_noise()
            if self.cg is not None:
                self.cg.replay()
            else:
                self.step()
            self.n_done += 1
            due = [r for r, (tau, F_) in self.redo.items() if self.n_done % F_ == 0]
            if due:
                if jj == STEPS - 1:
                    pending = due
                else:
                    if jj < 1:
                        raise RuntimeError("a ReDo check on a task's first update has no 64-image probe")
                    self.redo_apply(due, batches[:, jj - 1:jj + 1].reshape(self.R, -1).to(self.device))
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        return pending

    @torch.no_grad()
    def redo_scores(self, rows: torch.Tensor) -> list[torch.Tensor]:
        """[4] (R, width) layer-normalised mean |phi(z)| on X[rows] (rows: (R, n))."""
        out = E.forward(self.P, self.data.X[rows], self.act)
        res = []
        for l, a in enumerate((out[1], out[3], out[5], out[7])):
            if IS_CONV[l]:
                s = a.abs().mean((0, 2, 3)).view(self.R, CH)
            else:
                s = a.abs().mean(1)
            res.append(s / (s.mean(1, keepdim=True) + REDO_EPS))
        return res

    @torch.no_grad()
    def redo_apply(self, runs: list[int], rows: torch.Tensor, phase: str = "train") -> None:
        """Sokar Alg. 1 for these runs on the 64 images X[rows] (spec §3.1)."""
        scores = self.redo_scores(rows)
        for r in runs:
            tau, F_ = self.redo[r]
            dormant = [torch.nonzero(scores[l][r] <= tau).flatten().cpu() for l in range(4)]
            g = self.redo_gen[r]
            for l in range(4):                                  # incoming side, site order
                idx = dormant[l]
                if len(idx):
                    W, b = self.P[2 * l], self.P[2 * l + 1]
                    fan = NK.FAN_IN[l]
                    fresh = ((torch.rand((len(idx), fan), generator=g, dtype=torch.float32) * 2 - 1)
                             * (1.0 / math.sqrt(fan))).view(len(idx), *W.shape[2:])
                    ii = idx.to(self.device)
                    W[r, ii] = fresh.to(device=self.device, dtype=self.dtype)
                    b[r, ii] = 0.0
                    self.m[2 * l][r, ii] = 0.0
                    self.v[2 * l][r, ii] = 0.0
            for l in range(4):                                  # then the outgoing side
                idx = dormant[l]
                if len(idx):
                    i_next = 2 * l + 2
                    if l == 1:                                  # c2 channel i -> f1 columns 64i..64i+63
                        k = NK.FLAT // CH
                        cols = (idx[:, None] * k + torch.arange(k)[None, :]).flatten().to(self.device)
                    else:
                        cols = idx.to(self.device)
                    for T in (self.P[i_next], self.m[i_next], self.v[i_next]):
                        T[r][:, cols] = 0.0
            for l in range(4):
                self.n_reset[r, l] += len(dormant[l])
            self.redo_log.append({"slot": r, "cfg": self.slots[r][0], "seed": self.slots[r][1],
                                  "phase": phase, "update": self.n_done,
                                  **{f"n_dormant_{SITES[l]}": int(len(dormant[l])) for l in range(4)}})

    def full_state(self) -> dict:
        st = super().full_state()
        st["rsl_gen"] = {str(r): g.get_state() for r, g in self.rsl_gen.items()}
        st["redo_gen"] = {str(r): g.get_state() for r, g in self.redo_gen.items()}
        st["n_done"] = self.n_done
        if self.l2_any:
            st["P0"] = [p.cpu().clone() for p in self.P0]
        return st

    def load_full_state(self, st: dict) -> None:
        super().load_full_state(st)
        if set(st.get("rsl_gen", {})) != {str(r) for r in self.rsl_gen} or \
                set(st.get("redo_gen", {})) != {str(r) for r in self.redo_gen}:
            raise ValueError("RSL / ReDo stream state does not match the slots")
        for r, g in self.rsl_gen.items():
            g.set_state(st["rsl_gen"][str(r)])
        for r, g in self.redo_gen.items():
            g.set_state(st["redo_gen"][str(r)])
        self.n_done = int(st["n_done"])
        if self.l2_any:
            with torch.no_grad():
                for q, s in zip(self.P0, st["P0"]):
                    q.copy_(s)


# --------------------------------------------------------------------------
# the run (adapted from the previous run; divergence is per run here)
# --------------------------------------------------------------------------

def run(slots, out: Path, n_tasks: int, device, graph: bool = True, resume: bool = True,
        data: NK.Data | None = None, log=None, allow_dirty: bool = False, chunk: int = 50,
        diag: bool = True, fresh: bool = True, checkpoints: bool = True, stage: str = "train",
        stop_after: int | None = None) -> dict:
    log = log or (lambda m: print(m, flush=True))
    out = Path(out)
    t_wall = time.time()
    gstart = NK.git_state()
    if gstart["dirty_src_analysis_specs"] and not allow_dirty:
        raise SystemExit("src/analysis/specs have uncommitted changes; commit before a registered run")
    data = data or NK.Data(device)
    seeds = sorted({s for a, s in slots})
    prep = NK.prepare(seeds, data)
    plans, probe = prep["plans"], prep["probe"].to(device)
    eng = MethodsEngine(slots, data, device, graph=graph, stage=stage)
    R = eng.R
    meta = {"schema": 1, "run_id": EXPERIMENT, "slots": [list(s) for s in eng.slots], "stage": stage,
            "n_tasks": n_tasks, "steps": STEPS, "batch": BATCH, "graph": eng.graph, "diag": diag,
            "fresh": fresh, "spec_sha256": hashlib.sha256(SPEC.read_bytes()).hexdigest(),
            "data_sha256": data.sha256, "probe_sha256": prep["probe_sha256"],
            "batch_sha256": {str(s): v for s, v in prep["batch_sha256"].items()}}
    out.mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir(exist_ok=True)
    if diag:
        (out / "units").mkdir(exist_ok=True)
    st_path = out / "state.pt"
    T = n_tasks
    book = {"rows": [], "blocks": [], "excerpt": [], "site": [], "alpha": [], "calibration": {},
            "steps": {k: np.zeros((R, T, STEPS), dtype=np.float32) for k in NK.REC_KEYS},
            "timing": [], "gits": [gstart], "resumed": [], "diverged": {}, "redo": []}
    t0 = 1
    if st_path.exists():
        if not resume:
            raise SystemExit(f"{st_path} exists; choose a fresh output")
        st = torch.load(st_path, map_location="cpu", weights_only=False)
        if st["meta"] != meta:
            raise SystemExit(f"{st_path} belongs to another configuration")
        eng.load_full_state(st["state"])
        book = st["book"]
        eng.redo_log = list(book.get("redo", []))
        book["gits"].append(gstart)
        t0 = st["task"] + 1
        book["resumed"].append({"at_task": t0, **gstart})
        log(f"[{time.strftime('%T')}] resumed after task {st['task']}")
    elif any(p.is_file() for p in out.iterdir()):
        raise SystemExit(f"{out} holds files but no resumable state; choose a fresh output")
    else:
        if checkpoints:
            NK.save_atomic({"meta": meta, "state": eng.full_state(), "task": 0}, out / "checkpoints" / "t00.pt")
        NK.save_atomic({"meta": meta, "state": eng.full_state(), "task": 0, "book": book}, st_path)
        NK.write_json(out / "task_plan.json", {str(s): [{"task": t, "hard": bool(h), "classes": c}
                                                        for t, (h, c) in enumerate(plans[s], 1)] for s in seeds})
        NK.write_json(out / "batch_hashes.json",
                      {"rule": "sha256 of the (780, 32) int64 global train rows of each task; the same "
                               "tensor feeds every configuration of a seed",
                       "per_seed": {str(s): v for s, v in prep["batch_sha256"].items()}})
        NK.write_json(out / "probe_ids.json", {"role": NK.PROBE_ROLE, "seed": NK.PROBE_SEED,
                                               "rows": prep["probe"].tolist(), "sha256": prep["probe_sha256"]})
    prov = {"run_id": EXPERIMENT, "status": "RUNNING", "meta": meta, "env": NK.env_info(device),
            "spec": str(SPEC.relative_to(ROOT)), "spec_commit_registered": SPEC_COMMIT,
            "gits": book["gits"], "resumed": book["resumed"], "sources_sha256": NK.source_hashes(),
            "configs": {a: parse_cfg(a) for a in sorted({a for a, s in eng.slots})},
            "scientific_verdict": None}
    NK.write_json(out / "provenance.json", prov)
    if diag:
        NK.unit_stats(eng.P, eng.act, data.X, probe[None].expand(R, -1)[:, :chunk], chunk=chunk)
    for t in range(t0, n_tasks + 1):
        tt = time.time()
        hard = t % 2 == 1
        batches = torch.stack([prep["batches"][s][t - 1] for a, s in eng.slots]).to(device)
        tr = NK.slot_rows(eng.slots, plans, t, data.tr_rows).to(device)
        te = NK.slot_rows(eng.slots, plans, t, data.te_rows).to(device)
        sce, sacc = NK.ce_acc(eng.P, eng.act, data.X, data.Y, tr)
        ctl0 = NK.control_arrays(eng.act)
        eng.n_reset[:] = 0
        t_train = time.time()
        pending = eng.train_task(batches)
        t_train = time.time() - t_train
        rec = eng.records()
        bad = eng.bad_step.cpu().numpy()
        finite = np.array([all(bool(torch.isfinite(p[r]).all()) for p in eng.P) for r in range(R)])
        for r in range(R):
            key = f"{eng.slots[r][0]}|{eng.slots[r][1]}"
            if (bad[r] >= 0 or not finite[r]) and key not in book["diverged"]:
                book["diverged"][key] = {"task": t, "bad_step_in_task": int(bad[r]),
                                         "params_finite": bool(finite[r])}
        for k in NK.REC_KEYS:
            book["steps"][k][:, t - 1] = rec[k]
        t_diag = time.time()
        if diag:
            tr_stats, tr_acc = NK.unit_stats(eng.P, eng.act, data.X, tr, data.Y, chunk=chunk)
            pr_stats, _ = NK.unit_stats(eng.P, eng.act, data.X, probe[None].expand(R, -1), chunk=chunk)
        else:
            _, tr_acc = NK.ce_acc(eng.P, eng.act, data.X, data.Y, tr)
        _, te_acc = NK.ce_acc(eng.P, eng.act, data.Xte, data.Yte, te)
        ctl1 = NK.control_arrays(eng.act)
        wn = NK.weight_row_norms(eng.P)
        t_diag = time.time() - t_diag
        if diag:
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
        for r, (name, seed) in enumerate(eng.slots):
            cfg = eng.cfgs[r]
            snake = cfg["kind"] in ("snake", "kkt1")
            ce, hit, g, mu = rec["ce"][r], rec["hit"][r], rec["gate"][r], rec["mult"][r]
            cls = plans[seed][t - 1][1]
            key = f"{name}|{seed}"
            row = {"arm": cfg["arm"], "cfg": name, "lr": cfg["lr"],
                   "shape": " ".join(f"{x:g}" for x in cfg["shape"]), "seed": seed, "slot": r, "task": t,
                   "hard": int(hard), "n_classes": len(cls), "classes": " ".join(map(str, cls)),
                   "online_acc": float(hit.mean()), "first100_online_acc": float(hit[:NK.FIRST_N].mean()),
                   "train_acc": float(tr_acc[r]), "test_acc": float(te_acc[r]),
                   "switch_ce": float(sce[r]), "switch_acc": float(sacc[r]),
                   "ce_step1": float(ce[0]), "ce_first10": float(ce[:10].mean()),
                   "ce_last10": float(ce[-10:].mean()), "ce_mean": float(ce.mean()),
                   "gate_mean": float(g.mean()), "gate_min": float(g.min()), "gate_max": float(g.max()),
                   "gate_first10": float(g[:10].mean()), "gate_last10": float(g[-10:].mean()),
                   "mult_mean": float(mu.mean()), "sum_beta_eff": float(NK.BETA * mu.sum()),
                   "gate_enabled": bool(eng.act.enabled[r, 0]), "alpha_frozen": bool(eng.act.frozen[r, 0]),
                   "loss_ref": float(eng.act.loss_ref[r, 0]) if bool(eng.act.calibrated[r, 0]) else None,
                   "diverged": key in book["diverged"], "bad_step": int(bad[r])}
            for l, tag in enumerate(SITES):
                a1 = ctl1["alpha"][l][r]
                row[f"alpha_med_{tag}"] = float(np.median(a1)) if snake else None
                row[f"two_alpha_sqrtV_med_{tag}"] = (float(np.median(2 * a1 * np.sqrt(ctl1["V"][l][r])))
                                                    if snake else None)
                if diag:
                    row[f"zsd_med_{tag}"] = float(np.median(tr_stats[tag]["s"][r]))
                    row[f"zbar_med_{tag}"] = float(np.median(tr_stats[tag]["m"][r]))
                    row[f"dmean_med_{tag}"] = float(np.median(tr_stats[tag]["dmean"][r]))
                row[f"n_redo_{tag}"] = int(eng.n_reset[r, l])
            book["rows"].append(row)
            for b in range(STEPS // NK.BLOCK):
                sl = slice(b * NK.BLOCK, (b + 1) * NK.BLOCK)
                book["blocks"].append({"cfg": name, "seed": seed, "task": t, "block": b + 1,
                                       "ce_mean": float(ce[sl].mean()), "hit_mean": float(hit[sl].mean()),
                                       "g_mean": float(g[sl].mean()), "g_min": float(g[sl].min()),
                                       "g_max": float(g[sl].max()), "mult_mean": float(mu[sl].mean()),
                                       "sum_beta_eff": float(NK.BETA * mu[sl].sum())})
            for jj in list(range(10)) + list(range(STEPS - 10, STEPS)):
                book["excerpt"].append({"cfg": name, "seed": seed, "task": t, "step": jj + 1,
                                        "ce": float(ce[jj]), "hit": float(hit[jj]), "g": float(g[jj]),
                                        "mult": float(mu[jj])})
            if diag:
                for inp, stats, n_img in (("train", tr_stats, tr.shape[1]), ("probe", pr_stats, probe.shape[0])):
                    for l, tag in enumerate(SITES):
                        s_ = stats[tag]
                        q = {"cfg": name, "arm": cfg["arm"], "seed": seed, "task": t, "input": inp, "site": tag}
                        for k in ["m", "s", "r", "dmean", "dsd", "dabs"] + (
                                ["s_within", "s_between", "r_within"] if IS_CONV[l] else []):
                            q.update(NK.agg(s_[k][r], k))
                        q["r_undefined"] = int((~(s_["s"][r] > 0)).sum())
                        if snake:
                            a1 = ctl1["alpha"][l][r]
                            q.update(NK.agg(a1, "alpha"))
                            q.update(NK.agg(2 * a1 * s_["s"][r], "two_alpha_s"))
                            q.update(NK.agg(2 * a1 * np.sqrt(ctl1["V"][l][r]), "two_alpha_sqrtV"))
                        q.update(NK.agg(wn[NK.WTAGS[l]][r], "wnorm"))
                        book["site"].append(q)
            if snake:
                for l, tag in enumerate(SITES):
                    a0, a1 = ctl0["alpha"][l][r], ctl1["alpha"][l][r]
                    book["alpha"].append({
                        "cfg": name, "arm": cfg["arm"], "seed": seed, "task": t, "site": tag,
                        **NK.agg(a0, "alpha_start"), **NK.agg(a1, "alpha_end"),
                        "alpha_frac_changed": float((a1 != a0).mean()),
                        "alpha_at_lo_end": int((a1 <= NK.ALPHA_LO).sum()),
                        "alpha_at_hi_end": int((a1 >= NK.ALPHA_HI).sum()), "n_units": WIDTHS[l],
                        **NK.agg(np.sqrt(ctl1["V"][l][r]), "sqrtV_end"),
                        **NK.agg(np.sqrt(ctl1["Vrec"][l][r]), "sqrtVrec_end")})
        # ---- after the evaluation: a ReDo check that fell on the task's last update
        if pending:
            eng.redo_apply(pending, batches[:, STEPS - 2:STEPS].reshape(R, -1), phase="boundary")
            for r in pending:
                for q in book["rows"][-R:]:
                    if q["slot"] == r:
                        for l, tag in enumerate(SITES):
                            q[f"n_redo_boundary_{tag}"] = int(eng.n_reset[r, l]) - q[f"n_redo_{tag}"]
        book["redo"] = list(eng.redo_log)
        # ---- task-1 boundary: calibration (previous Snake arms; control only NEO), freeze SNAfrz1
        if t == 1:
            refs, control, frz = {}, [], []
            for r, (name, seed) in enumerate(eng.slots):
                base = eng.cfgs[r]["base"]
                if eng.cfgs[r]["arm"] in NK.SNAKE_ARMS:
                    c = NK.reference_loss(rec["ce"][r]) if np.isfinite(rec["ce"][r]).all() else None
                    if c is None:
                        continue
                    c.update({"cfg": name, "seed": seed, "used_for_control": base == "NEO"})
                    book["calibration"][f"{name}|{seed}"] = c
                    refs[r] = c["lref"]
                    if base == "NEO":
                        control.append(r)
                if base == "SNAfrz1":
                    frz.append(r)
            eng.act.set_reference(refs, control)
            eng.act.freeze_runs(frz)
        book["timing"].append({"task": t, "train_s": t_train, "diag_s": t_diag, "total_s": time.time() - tt})
        if checkpoints and (t in NK.CKPT_TASKS or t == n_tasks):
            NK.save_atomic({"meta": meta, "state": eng.full_state(), "task": t}, out / "checkpoints" / f"t{t:02d}.pt")
        NK.write_csv(out / "per_task.csv", book["rows"])
        NK.save_atomic({"meta": meta, "state": eng.full_state(), "task": t, "book": book}, st_path)
        on = {}
        for q in book["rows"][-R:]:
            on.setdefault(q["arm"], []).append(q["online_acc"])
        log(f"[{time.strftime('%T')}] task {t:2d}/{n_tasks} {'hard' if hard else 'easy'} train {t_train:5.1f}s "
            f"diag {t_diag:4.1f}s diverged {len(book['diverged'])}")
        if stop_after is not None and t == stop_after:
            raise KeyboardInterrupt(f"stop_after={t} (check hook)")
    fresh_rows = []
    if fresh:
        tf = NK.last_hard(n_tasks)
        tt = time.time()
        fe = MethodsEngine(eng.slots, data, device, graph=graph, stage="fresh")
        batches = torch.stack([prep["batches"][s][tf - 1] for a, s in eng.slots]).to(device)
        fpend = fe.train_task(batches)
        if fpend:
            fe.redo_apply(fpend, batches[:, STEPS - 2:STEPS].reshape(R, -1), phase="fresh_boundary")
        frec = fe.records()
        tr = NK.slot_rows(eng.slots, plans, tf, data.tr_rows).to(device)
        te = NK.slot_rows(eng.slots, plans, tf, data.te_rows).to(device)
        _, ftr = NK.ce_acc(fe.P, fe.act, data.X, data.Y, tr)
        _, fte = NK.ce_acc(fe.P, fe.act, data.Xte, data.Yte, te)
        cont = {(q["cfg"], q["seed"]): q for q in book["rows"] if q["task"] == tf}
        for r, (name, seed) in enumerate(eng.slots):
            c = cont[(name, seed)]
            f_on = float(frec["hit"][r].mean())
            fresh_rows.append({"arm": eng.cfgs[r]["arm"], "cfg": name, "seed": seed, "task": tf,
                               "batch_sha256": prep["batch_sha256"][seed][tf - 1],
                               "fresh_online_acc": f_on,
                               "fresh_first100_online_acc": float(frec["hit"][r][:NK.FIRST_N].mean()),
                               "fresh_train_acc": float(ftr[r]), "fresh_test_acc": float(fte[r]),
                               "continual_online_acc": c["online_acc"], "continual_test_acc": c["test_acc"],
                               "fresh_gap_online": f_on - c["online_acc"],
                               "fresh_gap_test": float(fte[r]) - c["test_acc"],
                               "fresh_bad_step": int(fe.bad_step[r])})
        NK.write_csv(out / "fresh_control.csv", fresh_rows)
        book["timing"].append({"task": f"fresh_t{tf}", "total_s": time.time() - tt})
        np.savez_compressed(out / "fresh_per_step.npz", **{k: v.astype(np.float32) for k, v in frec.items()})
        del fe
    NK.write_csv(out / "per_block.csv", book["blocks"])
    NK.write_csv(out / "step_excerpt.csv", book["excerpt"])
    NK.write_csv(out / "redo.csv", book["redo"])
    NK.write_csv(out / "alpha_stats.csv", book["alpha"])
    if diag:
        NK.write_csv(out / "site_stats.csv", book["site"])
        units = {}
        for t in range(1, n_tasks + 1):
            with np.load(out / "units" / f"t{t:02d}.npz") as z:
                for k in z.files:
                    units.setdefault(k, []).append(z[k])
        np.savez_compressed(out / "unit_stats.npz", **{k: np.stack(v, axis=1) for k, v in units.items()},
                            slots=np.asarray([f"{a}|{s}" for a, s in eng.slots]))
    NK.write_json(out / "calibration.json", {"runs": book["calibration"]})
    np.savez_compressed(out / "per_step.npz", **{k: v for k, v in book["steps"].items()},
                        slots=np.asarray([f"{a}|{s}" for a, s in eng.slots]))
    prov.update(status="COMPLETED", gits=book["gits"], resumed=book["resumed"], timing=book["timing"],
                diverged=book["diverged"], wall_clock_s_this_process=time.time() - t_wall,
                cuda_max_mem_mb=(torch.cuda.max_memory_allocated() / 2 ** 20 if device.type == "cuda" else None),
                updates={"continual": R * n_tasks * STEPS, "fresh": R * STEPS if fresh else 0},
                sources_sha256=NK.source_hashes())
    NK.write_json(out / "provenance.json", prov)
    log(f"[{time.strftime('%T')}] done {out} ({(time.time() - t_wall) / 60:.1f} min, diverged {len(book['diverged'])})")
    return prov


# --------------------------------------------------------------------------
# drivers
# --------------------------------------------------------------------------

def calib_slots() -> list[tuple[str, int]]:
    return [(c, s) for arm in ARMS12 for c in grid(arm) for s in CALIB_SEEDS]


def calib_bundles() -> list[list[tuple[str, int]]]:
    sl = calib_slots()
    return [sl[i:i + BUNDLE_MAX] for i in range(0, len(sl), BUNDLE_MAX)]


def completed(d: Path) -> bool:
    p = d / "provenance.json"
    return p.exists() and json.loads(p.read_text()).get("status") == "COMPLETED"


def late_A(rows, cfg, seed) -> float:
    xs = [float(q["online_acc"]) for q in rows if q["cfg"] == cfg and int(q["seed"]) == seed
          and int(q["task"]) in NK.LATE]
    return float(np.mean(xs)) if len(xs) == len(NK.LATE) else float("nan")


def select(calib_dir: Path) -> dict:
    """spec §4: max mean A over the calibration seeds; ties -> grid order; drop diverged points."""
    rows, diverged = [], set()
    for k in range(len(calib_bundles())):
        d = calib_dir / f"b{k + 1}"
        if not completed(d):
            raise SystemExit(f"calibration bundle {d} is not COMPLETED")
        rows += list(csv.DictReader(open(d / "per_task.csv")))
        diverged |= {k_.split("|")[0] for k_ in json.loads((d / "provenance.json").read_text())["diverged"]}
    sel = {}
    for arm in ARMS12:
        table = []
        for i, c in enumerate(grid(arm)):
            vals = [late_A(rows, c, s) for s in CALIB_SEEDS]
            ok = c not in diverged and all(np.isfinite(vals))
            table.append({"cfg": c, "order": i, "A_seeds": vals, "A_mean": float(np.mean(vals)) if ok else None,
                          "dropped": not ok})
        valid = [q for q in table if not q["dropped"]]
        if not valid:
            sel[arm] = {"cfg": None, "label": "NO_VALID_CONFIG", "grid": table}
            continue
        best = max(valid, key=lambda q: (q["A_mean"], -q["order"]))
        cfg = parse_cfg(best["cfg"])
        edge = []
        if cfg["lr"] in (1e-4, 1e-3):
            edge.append(f"lr={cfg['lr']:g}")
        if arm in ("ELU",):
            edge.append(f"a={cfg['shape'][0]:g} (two-value grid)")
        if arm == "LK" and cfg["shape"][0] in (0.1, 0.8):
            edge.append(f"a={cfg['shape'][0]:g}")
        if arm == "L2I" and cfg["shape"][0] in (1e-3, 1e-1):
            edge.append(f"lam={cfg['shape'][0]:g}")
        sel[arm] = {"cfg": best["cfg"], "A_mean": best["A_mean"], "EDGE": edge, "grid": table}
    res = {"rule": "spec §4: max mean late-window online accuracy over seeds 100-102; diverged points "
                   "dropped; exact ties -> registered grid order", "selected": sel,
           "dropped": sorted(diverged)}
    NK.write_json(calib_dir / "selected.json", res)
    return res


def main_bundles(selected: dict) -> dict[str, list[tuple[str, int]]]:
    out = {}
    for k, arms in MAIN_BUNDLES.items():
        sl = []
        for arm in arms:
            c = selected["selected"][arm]["cfg"]
            if c is None:
                continue
            sl += [(c, s) for s in MAIN_SEEDS]
        out[k] = sl
    return out


def smoke_slots() -> list[tuple[str, int]]:
    return [(first_cfg(arm), s) for s in NK.SMOKE_SEEDS for arm in ARMS12]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["smoke", "calib", "select", "main", "run"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--slots", default=None, help="run: JSON list of [cfg, seed]")
    ap.add_argument("--tasks", type=int, default=NK.N_TASKS)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    if a.stage == "select":
        print(json.dumps({k: v.get("cfg") for k, v in select(out)["selected"].items()}, indent=2))
        return
    device = NK.setup(a.device)
    data = NK.Data(device)
    g = not a.no_graph
    if a.stage == "smoke":
        run(smoke_slots(), out, 2, device, graph=g, data=data, allow_dirty=a.allow_dirty)
    elif a.stage == "run":
        slots = [tuple(x) for x in json.loads(a.slots)]
        run(slots, out, a.tasks, device, graph=g, data=data, allow_dirty=a.allow_dirty)
    elif a.stage == "calib":
        for k, sl in enumerate(calib_bundles()):
            d = out / f"b{k + 1}"
            if completed(d):
                print(f"skip {d} (COMPLETED)", flush=True)
                continue
            print(f"[{time.strftime('%T')}] calibration bundle {k + 1} ({len(sl)} runs)", flush=True)
            run(sl, d, NK.N_TASKS, device, graph=g, data=data, allow_dirty=a.allow_dirty,
                diag=False, fresh=False, checkpoints=False, stage="calib")
        res = select(out)
        print(json.dumps({k: v.get("cfg") for k, v in res["selected"].items()}, indent=2), flush=True)
    elif a.stage == "main":
        selected = json.loads((OUT_ROOT / "calib" / "selected.json").read_text())
        for k, sl in main_bundles(selected).items():
            d = out / k
            if completed(d):
                print(f"skip {d} (COMPLETED)", flush=True)
                continue
            print(f"[{time.strftime('%T')}] main bundle {k} ({len(sl)} runs)", flush=True)
            run(sl, d, NK.N_TASKS, device, graph=g, data=data, allow_dirty=a.allow_dirty, stage="main")


if __name__ == "__main__":
    main()
