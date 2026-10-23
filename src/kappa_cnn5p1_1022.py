#!/usr/bin/env python3
"""kappa_cnn5p1_1022: the readout projection kappa, cos^2, Q and the lemma on the saved checkpoints of the
real-label 5+1 CIFAR CNN (spec specs/spec_kappa_cnn5p1_1022.md, sec. 3-4).

    python src/kappa_cnn5p1_1022.py checks                    C1-C4 (C1 on every own-end state of every budget)
    python src/kappa_cnn5p1_1022.py measure [--budgets 780]   every (boundary, task) pair of sec. 3 -> rows.csv, raw/

Post-hoc on the saved checkpoints of cifar5p1_cnn_budget_1019 (read only).  The forward is the training
engine's (BudgetEngine with the checkpoint's full state loaded, float32, TF32 off, deterministic); every
statistic is float64 on the f2 features a4 and the logits of a task's 500 x K training images.
"""

from __future__ import annotations

import argparse
import csv
import itertools
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
from src import cifar5p1_cnn_budget_1019 as BG          # the engine of the saved runs; not modified
from src import neokunekune_cifar5p1_cnn_1018 as NK
from src import cifar5p1_mlp_0920 as C51

E = NK.E
EXPERIMENT = "kappa_cnn5p1_1022"
ARCH = Path("/home/issan/Projects/obsidian-research-data/cifar5p1_cnn_budget_1019/results/cifar5p1_cnn_budget_1019")
DATA_TGZ = Path("/home/issan/Projects/claude/proj_004_drift/data/cifar100/cifar-100-python.tar.gz")
PREV = ROOT / "results" / "cifar5p1_cnn_budget_1019"
OUT = ROOT / "results" / EXPERIMENT
RAW = OUT / "raw"
BUDGETS = (780, 2340, 7800)
K = C51.HARD_CLASSES                                    # 5
LOGK = math.log(K)
HID = NK.HID
N_CLASSES = NK.N_CLASSES
DT = torch.float64
# (boundary, target task) pairs of spec sec. 3, with their role
PAIRS = [(0, 1, "init"), (1, 1, "own"),
         (8, 9, "pre"), (8, 7, "last"), (8, 1, "prev"),
         (9, 9, "own"), (9, 7, "last"), (9, 1, "prev"),
         (18, 19, "pre"), (18, 17, "last"), (18, 9, "prev"), (18, 1, "prev"),
         (19, 19, "own"), (19, 17, "last"), (19, 9, "prev"), (19, 1, "prev"),
         (28, 29, "pre"), (28, 27, "last"), (28, 19, "prev"), (28, 9, "prev"), (28, 1, "prev"),
         (29, 29, "own"), (29, 27, "last"), (29, 19, "prev"), (29, 9, "prev"), (29, 1, "prev"),
         (30, 29, "last"), (30, 1, "prev")]
OWN = [(b, t) for b, t, role in PAIRS if role == "own"]
FF_FLOOR_PER = 1e-12                                    # x N x HID (spec sec. 3)
COLLAPSE_ACC = 0.3
PERMS = torch.tensor(list(itertools.permutations(range(K))), dtype=torch.long)   # (120, 5)


def tonp(t):
    return t.detach().double().cpu().numpy()


# --------------------------------------------------------------------------
# states and features
# --------------------------------------------------------------------------

def load_data(device):
    return NK.Data(device, cifar=C51.Cifar100(DATA_TGZ))


def ckpt_path(budget: int, boundary: int) -> Path:
    return ARCH / f"b{budget}" / "checkpoints" / f"b{boundary:02d}.pt"


class Bundle:
    """One budget's 40-slot engine; load_state() swaps the checkpoint in place (read only afterwards)."""

    def __init__(self, budget: int, data, device):
        ck = torch.load(ckpt_path(budget, 0), map_location="cpu", weights_only=False)
        self.meta = ck["meta"]
        if self.meta["data_sha256"] != data.sha256:
            raise SystemExit("CIFAR-100 archive differs from the one the runs used")
        self.slots = [(str(a), int(s)) for a, s in self.meta["slots"]]
        self.budget, self.data, self.device = budget, data, device
        self.eng = BG.BudgetEngine(self.slots, data, device, budget, graph=False)
        self.plans = {s: NK.task_plan(s) for s in sorted({s for a, s in self.slots})}
        self.arms = [BG.M.parse_cfg(a)["arm"] for a, s in self.slots]
        self.boundary = None
        self.state_ok = None

    def load_state(self, boundary: int) -> None:
        ck = torch.load(ckpt_path(self.budget, boundary), map_location="cpu", weights_only=False)
        if ck["meta"] != self.meta or ck["boundary"] != boundary:
            raise SystemExit(f"checkpoint b{boundary:02d} of budget {self.budget} does not match b00's meta")
        self.eng.load_full_state(ck["state"])
        self.boundary = boundary

    def rows(self, t: int) -> torch.Tensor:
        return NK.slot_rows(self.slots, self.plans, t, self.data.tr_rows).to(self.device)

    def classes(self, r: int, t: int) -> list[int]:
        return list(self.plans[self.slots[r][1]][t - 1][1])

    @torch.no_grad()
    def features(self, rows: torch.Tensor, chunk: int = 50):
        """(R, N, HID) a4 and (R, N, 100) logits in float64, and the f2 phi' mean per unit (R, HID)."""
        R, N = rows.shape
        A4 = torch.zeros(R, N, HID, dtype=DT, device=self.device)
        L = torch.zeros(R, N, N_CLASSES, dtype=DT, device=self.device)
        D1 = torch.zeros(R, HID, dtype=DT, device=self.device)
        for i0 in range(0, N, chunk):
            idx = rows[:, i0:i0 + chunk]
            out = E.forward(self.eng.P, self.data.X[idx], self.eng.act)
            A4[:, i0:i0 + chunk] = out[7].double()
            L[:, i0:i0 + chunk] = out[8].double()
            D1 += self.eng.act.dphi(out[6], 3).double().sum(1)
        return A4, L, D1 / N


# --------------------------------------------------------------------------
# kappa and the lemma for one run (spec sec. 3)
# --------------------------------------------------------------------------

def quantities(A4, L, W5, b5, cls: list[int], y_local: torch.Tensor, perms: torch.Tensor) -> dict:
    """A4 (N, HID), L (N, 100) float64; W5 (100, HID), b5 (100,) float64; cls the task's K class ids
    (class-major rows: y_local = class index within the task)."""
    dev = A4.device
    N = A4.shape[0]
    Kt = len(cls)
    cls_t = torch.tensor(cls, device=dev)
    L32 = L
    L = A4 @ W5.T + b5                                       # float64 logits of the float64 features and head
    L32_relmax = float(((L32 - L).abs().max()) / L.abs().max().clamp_min(1e-300))
    W = W5[cls_t]                                            # (K, HID)
    b = b5[cls_t]
    Vc = W - W.mean(0, keepdim=True)
    bc = b - b.mean()
    Y = F.one_hot(y_local, Kt).to(DT)                        # (N, K)
    Fm = (Y - 1.0 / Kt).T @ A4                               # (K, HID)
    ff = float((Fm * Fm).sum())
    A = float((Vc * Fm).sum())
    nV = float(Vc.norm())
    ff_floor = FF_FLOOR_PER * N * HID
    defined = ff > ff_floor
    kappa = A / ff if defined else float("nan")
    cos2 = A * A / (nV * nV * ff) if (defined and nV > 0) else float("nan")
    Q = A / math.sqrt(ff) if defined else float("nan")
    ar = torch.arange(N, device=dev)
    Lk = L[:, cls_t]                                         # (N, K) the task's logits
    lk_y = Lk[ar, y_local]
    lbar = Lk.mean(1)
    A_id = float((lk_y - lbar - bc[y_local]).sum())
    lse5 = torch.logsumexp(Lk, 1)
    L5 = float((lse5 - lk_y).mean())
    J = float((lse5 - lbar - math.log(Kt)).sum())
    ncls = Y.sum(0)
    B = float(((ncls - N / Kt) * bc).sum())
    D5 = N * (math.log(Kt) - L5) - B
    atol_J = 1e-9 * float((lse5.abs() + lbar.abs()).sum())
    y_glob = cls_t[y_local]
    lse100 = torch.logsumexp(L, 1)
    L100 = float((lse100 - L[ar, y_glob]).mean())
    D100 = N * (math.log(Kt) - L100) - B
    acc100 = float((L32.argmax(1) == y_glob).double().mean())      # the training forward's logits (C1)
    acc5 = float((L32[:, cls_t].argmax(1) == y_local).double().mean())
    acc100_f64 = float((L.argmax(1) == y_glob).double().mean())
    Fi2 = (Fm * Fm).sum(0)
    kappa_i = torch.where(Fi2 > 0, (Vc * Fm).sum(0) / Fi2.clamp_min(1e-300), torch.full_like(Fi2, float("nan")))
    alive = (A4 > 0).any(0)                                  # f2 units with a positive output on some image
    # exact class-permutation null (features and F fixed, the K centred head rows permuted)
    Mx = Vc @ Fm.T                                           # M[a, c] = <v^c_a, F_c>
    perms = perms.to(dev)
    vals = Mx[perms, torch.arange(Kt, device=dev)[None, :]].sum(1) / ff if defined else None
    if defined:
        n_gt = int((vals > A / ff + 1e-12 * abs(A / ff)).sum())
        n_eq = int(((vals - A / ff).abs() <= 1e-12 * abs(A / ff)).sum())
        perm_q = float(((vals < kappa).double().mean() + 0.5 * (vals == kappa).double().mean()))
        perm_max = n_gt == 0
    else:
        n_gt, n_eq, perm_q, perm_max = -1, -1, float("nan"), False
    # all 100 head rows (supplementary)
    Vc100 = W5 - W5.mean(0, keepdim=True)
    Y100 = F.one_hot(y_glob, N_CLASSES).to(DT)
    F100 = (Y100 - 1.0 / N_CLASSES).T @ A4
    ff100 = float((F100 * F100).sum())
    A100 = float((Vc100 * F100).sum())
    nV100 = float(Vc100.norm())
    kappa100 = A100 / ff100 if ff100 > ff_floor else float("nan")
    cos2_100 = A100 * A100 / (nV100 * nV100 * ff100) if (ff100 > ff_floor and nV100 > 0) else float("nan")
    scal = dict(N=N, K=Kt, defined=defined, F_norm2=ff, F_norm=math.sqrt(ff), A=A, A_id=A_id,
                A_id_absdiff=abs(A - A_id), A_id_scale=abs(A) + float(lk_y.abs().sum()),
                normV=nV, normW=float(W.norm()), kappa=kappa, cos2=cos2, Q=Q,
                L5=L5, L100=L100, J=J, B=B, D5=D5, D100=D100, atol_J=atol_J,
                lemma_violation=bool(J < -atol_J), D5_positive=bool(D5 > 0),
                kappa_lb5=(D5 / ff if defined else float("nan")), kappa_lb100=(D100 / ff if defined else float("nan")),
                kappa_over_lb5=((A / D5) if D5 > 0 else float("nan")),
                acc100=acc100, acc5=acc5, acc100_f64=acc100_f64, L32_relmax=L32_relmax, perm_q=perm_q, perm_max=perm_max, perm_n_gt=n_gt, perm_n_eq=n_eq,
                Vc_rowsum_absmax=float(Vc.sum(0).abs().max()), F_colsum_absmax=float(Fm.sum(0).abs().max()),
                n_alive_f2=int(alive.sum()), kappa_i_n_undef=int((~torch.isfinite(kappa_i)).sum()),
                kappa_i_negfrac=(float((kappa_i[torch.isfinite(kappa_i)] < 0).double().mean())
                                 if torch.isfinite(kappa_i).any() else float("nan")),
                kappa100=kappa100, cos2_100=cos2_100, F100_norm=math.sqrt(ff100), A100=A100)
    arrays = dict(kappa_i=tonp(kappa_i), Fm=tonp(Fm).astype(np.float32), Vc=tonp(Vc).astype(np.float32),
                  perm_vals=(tonp(vals) if defined else np.full(len(perms), np.nan)))
    return scal, arrays


def y_local_for(rows: torch.Tensor, Kt: int) -> torch.Tensor:
    N = rows.shape[1]
    per = N // Kt
    if per * Kt != N:
        raise ValueError("rows are not class-balanced")
    return torch.repeat_interleave(torch.arange(Kt, device=rows.device), per)


def latest_hard(b: int) -> int:
    return b if b % 2 == 1 else b - 1


def read_per_task() -> dict:
    """(budget, cfg, seed, task) -> row of the saved per_task.csv."""
    out = {}
    for bud in BUDGETS:
        with open(PREV / f"b{bud}" / "per_task.csv") as fh:
            for r in csv.DictReader(fh):
                out[(bud, r["cfg"], int(r["seed"]), int(r["task"]))] = r
    return out


# --------------------------------------------------------------------------
# measure
# --------------------------------------------------------------------------

def measure_pair(Bd: Bundle, b: int, t: int, role: str, per_task: dict, chunk: int = 50):
    if Bd.boundary != b:
        Bd.load_state(b)
    rows = Bd.rows(t)
    Kt = len(Bd.classes(0, t))
    yl = y_local_for(rows, Kt)
    A4, L, dmean = Bd.features(rows, chunk=chunk)
    W5 = Bd.eng.P[8].detach().double()
    b5 = Bd.eng.P[9].detach().double()
    recs, arrs = [], {}
    for r, (cfg, seed) in enumerate(Bd.slots):
        sc, ar = quantities(A4[r], L[r], W5[r], b5[r], Bd.classes(r, t), yl, PERMS)
        lh = latest_hard(b) if b > 0 else None
        acc_state = float(per_task[(Bd.budget, cfg, seed, lh)]["train_acc"]) if lh else float("nan")
        rec = dict(budget=Bd.budget, boundary=b, task=t, role=role, arm=Bd.arms[r], cfg=cfg, seed=seed, slot=r,
                   classes=" ".join(map(str, Bd.classes(r, t))),
                   state_latest_hard=lh, state_train_acc=acc_state,
                   collapsed_state=(bool(acc_state < COLLAPSE_ACC) if lh else False),
                   f2_dmean_med=float(np.median(tonp(dmean[r]))), **sc)
        recs.append(rec)
        for k, v in ar.items():
            arrs.setdefault(k, []).append(v)
    arrs = {k: np.stack(v) for k, v in arrs.items()}
    return recs, arrs


def cmd_measure(budgets, device, chunk: int):
    OUT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    data = load_data(device)
    per_task = read_per_task()
    rows_all = []
    fn = OUT / "rows.csv"
    if fn.exists():
        with open(fn) as fh:
            rows_all = [r for r in csv.DictReader(fh) if int(r["budget"]) not in budgets]
    t0 = time.time()
    for bud in budgets:
        Bd = Bundle(bud, data, device)
        for b, t, role in PAIRS:                            # PAIRS is sorted by boundary: one load per boundary
            tt = time.time()
            recs, arrs = measure_pair(Bd, b, t, role, per_task, chunk=chunk)
            rows_all += recs
            np.savez(RAW / f"b{bud}_s{b:02d}_t{t:02d}.npz", slots=np.array([f"{c}|{s}" for c, s in Bd.slots]),
                     arms=np.array(Bd.arms), **arrs)
            own = [r for r in recs if r["defined"]]
            kp = [r["kappa"] for r in own]
            print(f"[{time.strftime('%T')}] b{bud} state {b:2d} task {t:2d} ({role:4s}) {time.time() - tt:5.1f}s  "
                  f"defined {len(own)}/40  kappa>0 {sum(k > 0 for k in kp)}  lemma viol {sum(r['lemma_violation'] for r in recs)}  "
                  f"cos2 med {np.median([r['cos2'] for r in own]) if own else float('nan'):.3f}", flush=True)
        del Bd
        torch.cuda.empty_cache() if device.type == "cuda" else None
    keys = list(rows_all[0].keys())
    for r in rows_all:
        for k in r:
            if k not in keys:
                keys.append(k)
    rows_all.sort(key=lambda r: (int(r["budget"]), int(r["boundary"]), int(r["task"]), int(r["slot"])))
    with open(fn, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows_all:
            w.writerow(r)
    print(f"wrote {fn} ({len(rows_all)} rows, {(time.time() - t0) / 60:.1f} min)")


# --------------------------------------------------------------------------
# checks (spec sec. 4): C1 reproduction, C2 identities, C3 degeneracy, C4 mutation controls
# --------------------------------------------------------------------------

def cmd_checks(device, chunk: int, budgets=BUDGETS):
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_data(device)
    per_task = read_per_task()
    checks = []

    def add(name, ok, required=True, **kw):
        checks.append(dict(name=name, ok=bool(ok), required=required, **kw))
        print(f"  {'PASS' if ok else 'FAIL'} {name} {json.dumps(kw, default=float)[:300]}", flush=True)

    gen = torch.Generator().manual_seed(1022)
    for bud in budgets:
        Bd = Bundle(bud, data, device)
        # ---- C1 reproduction on every own-end state, C2 identities
        for b, t in OWN:
            recs, arrs = measure_pair(Bd, b, t, "own", per_task, chunk=chunk)
            acc_diff, dm_diff = [], []
            for r in recs:
                pt = per_task[(bud, r["cfg"], r["seed"], t)]
                acc_diff.append(abs(r["acc100"] - float(pt["train_acc"])))
                dm_diff.append(abs(r["f2_dmean_med"] - float(pt["dmean_med_f2"])))
            add("C1_train_acc", max(acc_diff) <= 1e-9, budget=bud, boundary=b, maxdiff=max(acc_diff),
                n_exact=int(sum(d == 0.0 for d in acc_diff)), note="per_task.csv holds 10 significant digits")
            add("C1_f2_dmean_med", max(dm_diff) <= 1e-5, budget=bud, boundary=b, maxdiff=max(dm_diff))
            rel = [r["A_id_absdiff"] / max(r["A_id_scale"], 1e-300) for r in recs]
            add("C2_A_identity", max(rel) <= 1e-10, budget=bud, boundary=b, max_rel=max(rel),
                f32_logit_relmax=max(r["L32_relmax"] for r in recs))
            add("C2_B_zero", max(abs(r["B"]) for r in recs) <= 1e-9 * max(1.0, max(abs(r["A"]) for r in recs)),
                budget=bud, boundary=b, B_absmax=max(abs(r["B"]) for r in recs))
            add("C2_centering", max(max(r["Vc_rowsum_absmax"], r["F_colsum_absmax"]) for r in recs)
                <= 1e-8 * max(1.0, max(r["normV"] for r in recs), max(r["F_norm"] for r in recs)),
                budget=bud, boundary=b, absmax=max(max(r["Vc_rowsum_absmax"], r["F_colsum_absmax"]) for r in recs))
            add("C2_lemma_J_nonneg", not any(r["lemma_violation"] for r in recs), budget=bud, boundary=b,
                J_min=min(r["J"] for r in recs), n_viol=int(sum(r["lemma_violation"] for r in recs)))
            # ---- C4 mutation controls on this state (boundary 1 and 29 only, to keep the log short)
            if b in (1, 29):
                rows = Bd.rows(t)
                yl = y_local_for(rows, K)
                A4, L, _ = Bd.features(rows, chunk=chunk)
                W5 = Bd.eng.P[8].detach().double()
                b5 = Bd.eng.P[9].detach().double()
                sgn, c2r, pq, lp_ratio, lp_cos2, lp_Q, cos2_unc = [], [], [], [], [], [], []
                for r, (cfg, seed) in enumerate(Bd.slots):
                    cls = Bd.classes(r, t)
                    base, _ = quantities(A4[r], L[r], W5[r], b5[r], cls, yl, PERMS)
                    if not base["defined"]:
                        continue
                    cls_t = torch.tensor(cls, device=device)
                    for _ in range(5):
                        # (i) random head rows of the same Frobenius norm (logits recomputed from them)
                        W5m = W5[r].clone()
                        Wr = torch.randn(K, HID, generator=gen, dtype=DT).to(device)
                        Wr *= W5[r][cls_t].norm() / Wr.norm()
                        W5m[cls_t] = Wr
                        Lm = A4[r] @ W5m.T + b5[r]
                        q, _ = quantities(A4[r], Lm, W5m, b5[r], cls, yl, PERMS)
                        sgn.append(q["kappa"] > 0); c2r.append(q["cos2"]); pq.append(q["perm_q"])
                        # (ii) labels permuted over images (the class-membership of the images randomised)
                        yp = yl[torch.randperm(len(yl), generator=gen).to(device)]
                        Lp = L[r]                                  # logits of the images unchanged
                        q2, _ = quantities(A4[r], Lp, W5[r], b5[r], cls, yp, PERMS)
                        lp_ratio.append(abs(q2["kappa"]) / abs(base["kappa"]))
                        lp_cos2.append(q2["cos2"]); lp_Q.append(abs(q2["Q"]) / abs(base["Q"]))
                    # (iii') cos^2 with the uncentred head norm (A is invariant to a class-constant row; the norm is not)
                    cos2_unc.append(base["A"] ** 2 / (base["normW"] ** 2 * base["F_norm2"]))
                add("C4i_random_head_sign_half", 0.3 <= float(np.mean(sgn)) <= 0.7, budget=bud, boundary=b,
                    frac_pos=float(np.mean(sgn)), n=len(sgn))
                add("C4i_random_head_cos2_small", float(np.median(c2r)) <= 10.0 / (K * HID), budget=bud, boundary=b,
                    cos2_med=float(np.median(c2r)), cos2_max=float(np.max(c2r)), ref=1.0 / (K * HID))
                add("C4i_random_head_perm_q_uniform", 0.25 <= float(np.mean(pq)) <= 0.75, budget=bud, boundary=b,
                    perm_q_mean=float(np.mean(pq)), perm_q_sd=float(np.std(pq)))
                add("C4ii_label_perm_cos2_small", float(np.median(lp_cos2)) <= 10.0 / (K * HID), budget=bud, boundary=b,
                    cos2_med=float(np.median(lp_cos2)), cos2_max=float(np.max(lp_cos2)), ref=1.0 / (K * HID))
                add("C4ii_label_perm_Q_ratio_recorded", True, required=False, budget=bud, boundary=b,
                    Q_ratio_med=float(np.median(lp_Q)), Q_ratio_max=float(np.max(lp_Q)),
                    note="|Q_perm|/|Q| = sqrt(cos2_perm/cos2); scored on cos2 above (addendum 1)")
                add("C4ii_label_perm_kappa_ratio_recorded", True, required=False, budget=bud, boundary=b,
                    kappa_ratio_med=float(np.median(lp_ratio)), kappa_ratio_max=float(np.max(lp_ratio)),
                    note="kappa's denominator ||F||^2 collapses under the permutation; kappa is not scale-free")
                add("C4iii_uncentred_cos2_recorded", True, required=False, budget=bud, boundary=b,
                    cos2_uncentred_med=float(np.median(cos2_unc)))
        # ---- C3 degeneracy: an easy task (K = 1) gives F = 0 exactly
        Bd.load_state(8)
        rows = Bd.rows(8)
        A4, L, _ = Bd.features(rows, chunk=chunk)
        Y = torch.ones(rows.shape[1], 1, dtype=DT, device=device)
        Fm = (Y - 1.0).T @ A4[0]
        add("C3_easy_task_F_zero", float(Fm.abs().max()) == 0.0, budget=bud, F_absmax=float(Fm.abs().max()))
        del Bd
    fn = OUT / "checks.json"
    fn.write_text(json.dumps(dict(experiment=EXPERIMENT, time=time.strftime("%F %T"),
                                  n=len(checks), n_required=sum(c["required"] for c in checks),
                                  n_fail_required=sum((not c["ok"]) and c["required"] for c in checks),
                                  checks=checks), indent=1, default=float))
    nf = sum((not c["ok"]) and c["required"] for c in checks)
    print(f"checks: {len(checks)} ({nf} required FAIL) -> {fn}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["checks", "measure"])
    ap.add_argument("--budgets", default="780,2340,7800")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--chunk", type=int, default=50)
    a = ap.parse_args()
    device = NK.setup(a.device)
    budgets = tuple(int(x) for x in a.budgets.split(","))
    if a.cmd == "checks":
        cmd_checks(device, a.chunk, budgets)
    else:
        cmd_measure(budgets, device, a.chunk)


if __name__ == "__main__":
    main()
