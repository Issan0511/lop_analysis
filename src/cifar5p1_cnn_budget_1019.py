#!/usr/bin/env python3
"""cifar5p1_cnn_budget_1019 -- longer per-task budgets on the real-label 5+1 CIFAR CNN.

    python3 src/cifar5p1_cnn_budget_1019.py smoke --out results/cifar5p1_cnn_budget_1019/smoke
    python3 src/cifar5p1_cnn_budget_1019.py main --budget 780 --out results/cifar5p1_cnn_budget_1019/b780

Registered spec: specs/spec_cifar5p1_cnn_budget_1019.md (commit c2597606).

R, ELU (alpha 3.6), KKT1 and SNAfrz1 at lr 3e-4, budgets 780 / 2340 / 7800 updates per task, the
box, data, init, probe and engine of the previous runs (imported, not modified).  New here: batch
and record buffers of the budget's length, batch lists whose first 780 batches per task are the
previous c51_batch stream and whose extension comes from a per-task stream (shorter budgets are
prefixes; later tasks never shift), fixed-probe per-unit statistics at every task boundary with
positive occupancy, death, low response, zero output and effective rank, fresh controls on tasks
9 / 19 / 29, and complete checkpoints for later interventions.
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
from src import neokunekune_methods_cifar5p1_cnn_1018 as M     # the 12-arm engine; not modified
from src import pmnist_0905 as H

NK, E = M.NK, M.E
C51 = NK.C51
EXPERIMENT = "cifar5p1_cnn_budget_1019"
SPEC = ROOT / "specs" / f"spec_{EXPERIMENT}.md"
SPEC_COMMIT = "c2597606"
OUT_ROOT = ROOT / "results" / EXPERIMENT
ARMS4 = ("R_lr0.0003", "ELU_a3.6_lr0.0003", "KKT1_lr0.0003", "SNAfrz1_lr0.0003")
BUDGETS = (780, 2340, 7800)
SEEDS = NK.SEEDS
FRESH_TASKS = (9, 19, 29)
CKPT_BOUNDARIES = (0, 1, 8, 9, 18, 19, 28, 29, 30)
BASE_STEPS = NK.STEPS                                   # 780: the previous stream's part of each task
DEAD_TOL = 1e-6                                          # the host's DEAD_TOL
LOW_RESP = 0.1                                           # spec §5.1
BATCH, N_CLASSES = NK.BATCH, NK.N_CLASSES
CH, IS_CONV, WIDTHS, SITES = NK.CH, NK.IS_CONV, NK.WIDTHS, NK.SITES


# --------------------------------------------------------------------------
# batch lists (spec §4)
# --------------------------------------------------------------------------

def make_budget_batches(seed: int, plan, tr_rows, budget: int) -> torch.Tensor:
    """(30, budget, 32) rows: per task the previous c51_batch stream's 780 batches (the stream is
    consumed task by task exactly as before, whatever the budget), then budget - 780 batches from
    the task's own extension stream."""
    if budget < BASE_STEPS:
        raise ValueError("budget >= 780")
    g = H.stream("c51_batch", seed)
    out = []
    for t, (hard, cls) in enumerate(plan, 1):
        rows = torch.cat([tr_rows[c] for c in cls])
        b = C51.batch_indices(g, rows, BASE_STEPS)
        if budget > BASE_STEPS:
            ge = H.stream(f"c51_budget_ext|t{t}", seed)
            b = torch.cat([b, C51.batch_indices(ge, rows, budget - BASE_STEPS)])
        out.append(b)
    return torch.stack(out)


def prepare(seeds, data: NK.Data, budget: int) -> dict:
    plans = {s: NK.task_plan(s) for s in seeds}
    batches = {s: make_budget_batches(s, plans[s], data.tr_rows, budget) for s in seeds}
    probe = NK.probe_ids(data.tr_rows)
    return {"plans": plans, "batches": batches, "probe": probe,
            "batch_sha256": {s: [NK.sha(batches[s][t]) for t in range(NK.N_TASKS)] for s in seeds},
            "probe_sha256": NK.sha(probe)}


# --------------------------------------------------------------------------
# the engine with budget-long buffers
# --------------------------------------------------------------------------

class BudgetEngine(M.MethodsEngine):
    def __init__(self, slots, data, device, budget: int, graph: bool = True, dtype=torch.float32,
                 stage: str = "train"):
        super().__init__(slots, data, device, graph=graph, dtype=dtype, stage=stage)
        if self.rsl_runs or self.redo:
            raise ValueError("this experiment has no RSL / ReDo arms")
        R = self.R
        self.budget = budget
        self.idx = torch.zeros(R, budget, BATCH, dtype=torch.long, device=device)
        self.iota = torch.arange(budget, device=device)[None, :]
        self.rec = {k: torch.zeros(R, budget, dtype=dtype, device=device) for k in NK.REC_KEYS}

    def train_task(self, batches: torch.Tensor, n_steps: int | None = None) -> list[int]:
        n = self.budget if n_steps is None else n_steps
        if tuple(batches.shape) != (self.R, self.budget, BATCH):
            raise ValueError(f"batches must be (R, {self.budget}, 32)")
        self.idx.copy_(batches.to(self.device))
        self.j.zero_()
        self.bad_step.fill_(-1)
        for v in self.rec.values():
            v.zero_()
        if self.graph and self.cg is None:
            self.capture()
        for _ in range(n):
            if self.cg is not None:
                self.cg.replay()
            else:
                self.step()
            self.n_done += 1
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        return []


# --------------------------------------------------------------------------
# per-unit statistics (NK.unit_stats's accumulations, same order, plus the new ones)
# --------------------------------------------------------------------------

@torch.no_grad()
def unit_stats2(P, act, X, rows: torch.Tensor, Y=None, chunk: int = 50):
    """NK.unit_stats (m, s, s_within, s_between, dmean, dsd, dabs, r) computed with the same
    operations in the same order, plus per unit: ppos (fraction of samples with z > 0), dmax (max
    phi'), amax (max |phi(z)|), dead (dmax < 1e-6), lowresp (dmean < 0.1), zeroout (amax == 0),
    and per site eff_rank (the host's definition on the activation Gram; conv: channels as columns)."""
    R, N = rows.shape
    dev = X.device
    f64 = torch.float64
    S1, S2, D1, D2, DA, shift = [], [], [], [], [], [None] * 4
    PP, DM, AM, G = [], [], [], []
    for l in range(4):
        shp = (R, CH, NK.CONV_HW[l], NK.CONV_HW[l]) if IS_CONV[l] else (R, WIDTHS[l])
        S1.append(torch.zeros(shp, dtype=f64, device=dev))
        S2.append(torch.zeros(shp, dtype=f64, device=dev))
        for lst in (D1, D2, DA, PP):
            lst.append(torch.zeros(R, WIDTHS[l], dtype=f64, device=dev))
        DM.append(torch.full((R, WIDTHS[l]), -math.inf, dtype=f64, device=dev))
        AM.append(torch.zeros(R, WIDTHS[l], dtype=f64, device=dev))
        G.append(torch.zeros(R, WIDTHS[l], WIDTHS[l], dtype=f64, device=dev))
    hits = torch.zeros(R, dtype=f64, device=dev)
    for i0 in range(0, N, chunk):
        idx = rows[:, i0:i0 + chunk]
        b = idx.shape[1]
        out = E.forward(P, X[idx], act)
        if Y is not None:
            hits += (out[8].argmax(-1) == Y[idx]).sum(1)
        for l in range(4):
            z = out[2 * l]
            a = out[2 * l + 1]
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
                zz = z.view(b, R, CH, hw, hw)
                PP[l] += (zz > 0).sum((0, 3, 4)).double()
                DM[l] = torch.maximum(DM[l], dr.amax((0, 3, 4)))
                ar = a.view(b, R, CH, hw, hw).double()
                AM[l] = torch.maximum(AM[l], ar.abs().amax((0, 3, 4)))
                feat = ar.permute(1, 0, 3, 4, 2).reshape(R, -1, CH)          # (R, b*H*W, C)
                G[l] += feat.transpose(1, 2) @ feat
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
                PP[l] += (z > 0).sum(1).double()
                DM[l] = torch.maximum(DM[l], dr.amax(1))
                ar = a.double()
                AM[l] = torch.maximum(AM[l], ar.abs().amax(1))
                G[l] += ar.transpose(1, 2) @ ar
    res = {}
    for l, tag in enumerate(SITES):
        s1, s2 = S1[l].cpu().numpy(), S2[l].cpu().numpy()
        if IS_CONV[l]:
            sh = shift[l].reshape(R, CH).cpu().numpy()
            n_all = N * s1.shape[2] * s1.shape[3]
            mc = s1.sum((2, 3)) / n_all
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
        st["ppos"] = PP[l].cpu().numpy() / nd
        st["dmax"] = DM[l].cpu().numpy()
        st["amax"] = AM[l].cpu().numpy()
        st["dead"] = (st["dmax"] < DEAD_TOL).astype(np.float64)
        st["lowresp"] = (st["dmean"] < LOW_RESP).astype(np.float64)
        st["zeroout"] = (st["amax"] == 0).astype(np.float64)
        ev = torch.linalg.eigvalsh(G[l].cpu()).clamp_min(0.0)          # (R, w)
        sv = ev.sqrt()
        tot = sv.sum(1, keepdim=True)
        p = torch.where(tot > 0, sv / tot.clamp_min(1e-300), torch.zeros_like(sv))
        er = torch.exp(-(p * p.clamp_min(1e-300).log()).sum(1))
        st["eff_rank"] = torch.where(tot[:, 0] > 0, er, torch.zeros_like(er)).numpy()
        res[tag] = st
    acc = (hits / N).cpu().numpy() if Y is not None else None
    return res, acc


UNIT_KEYS = ("m", "s", "r", "dmean", "dsd", "dabs", "ppos", "dmax", "amax", "dead", "lowresp", "zeroout")
CONV_KEYS = ("s_within", "s_between", "r_within")


def site_rows(stats, ctl, eng, base: dict) -> list[dict]:
    """One row per (run, site): unit medians / quartiles and the unit fractions."""
    rows = []
    for r, (name, seed) in enumerate(eng.slots):
        cfg = eng.cfgs[r]
        snake = cfg["kind"] in ("snake", "kkt1")
        for l, tag in enumerate(SITES):
            st = stats[tag]
            q = {"cfg": name, "arm": cfg["arm"], "seed": seed, **base, "site": tag, "n_units": WIDTHS[l]}
            for k in ("m", "s", "r", "dmean", "ppos") + (CONV_KEYS if IS_CONV[l] else ()):
                q.update(NK.agg(st[k][r], k))
            for k in ("dead", "lowresp", "zeroout"):
                q[f"{k}_frac"] = float(st[k][r].mean())
            q["r_undefined"] = int((~(st["s"][r] > 0)).sum())
            q["eff_rank"] = float(st["eff_rank"][r])
            if snake:
                a = ctl["alpha"][l][r]
                q.update(NK.agg(a, "alpha"))
                q.update(NK.agg(2 * a * st["s"][r], "two_alpha_s"))
            rows.append(q)
    return rows


def unit_arrays(stats, prefix: str) -> dict:
    out = {}
    for l, tag in enumerate(SITES):
        for k in UNIT_KEYS + (CONV_KEYS if IS_CONV[l] else ()):
            out[f"{prefix}{tag}_{k}"] = stats[tag][k].astype(np.float32)
        out[f"{prefix}{tag}_eff_rank"] = stats[tag]["eff_rank"].astype(np.float32)
    return out


# --------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------

def run(slots, out: Path, n_tasks: int, budget: int, device, graph: bool = True, resume: bool = True,
        data: NK.Data | None = None, log=None, allow_dirty: bool = False, chunk: int = 50,
        fresh_tasks=FRESH_TASKS, ckpt_boundaries=CKPT_BOUNDARIES, stop_after: int | None = None) -> dict:
    log = log or (lambda m: print(m, flush=True))
    out = Path(out)
    t_wall = time.time()
    gstart = NK.git_state()
    if gstart["dirty_src_analysis_specs"] and not allow_dirty:
        raise SystemExit("src/analysis/specs have uncommitted changes; commit before a registered run")
    data = data or NK.Data(device)
    seeds = sorted({s for a, s in slots})
    prep = prepare(seeds, data, budget)
    plans, probe = prep["plans"], prep["probe"].to(device)
    eng = BudgetEngine(slots, data, device, budget, graph=graph)
    R = eng.R
    fresh_tasks = tuple(t for t in fresh_tasks if t <= n_tasks)
    meta = {"schema": 1, "run_id": EXPERIMENT, "slots": [list(s) for s in eng.slots], "budget": budget,
            "n_tasks": n_tasks, "batch": BATCH, "graph": eng.graph, "fresh_tasks": list(fresh_tasks),
            "ckpt_boundaries": [b for b in ckpt_boundaries if b <= n_tasks],
            "spec_sha256": hashlib.sha256(SPEC.read_bytes()).hexdigest(), "data_sha256": data.sha256,
            "probe_sha256": prep["probe_sha256"], "batch_sha256": {str(s): v for s, v in prep["batch_sha256"].items()}}
    for d in ("checkpoints", "units", "steps"):
        (out / d).mkdir(parents=True, exist_ok=True)
    st_path = out / "state.pt"
    book = {"rows": [], "probe": [], "train": [], "calibration": {}, "frozen_alpha": {}, "timing": [],
            "gits": [gstart], "resumed": [], "diverged": {}}
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
        # boundary 0: the initial state on the probe, and its checkpoint
        st0, _ = unit_stats2(eng.P, eng.act, data.X, probe[None].expand(R, -1), chunk=chunk)
        ctl = NK.control_arrays(eng.act)
        book["probe"] += site_rows(st0, ctl, eng, {"boundary": 0})
        np.savez(out / "units" / "b00.npz", **unit_arrays(st0, "probe_"),
                 **{f"ctl_{t}_{k}": ctl[k][l].astype(np.float32) for l, t in enumerate(SITES) for k in ("alpha", "V")})
        if 0 in ckpt_boundaries:
            NK.save_atomic({"meta": meta, "state": eng.full_state(), "boundary": 0}, out / "checkpoints" / "b00.pt")
        NK.save_atomic({"meta": meta, "state": eng.full_state(), "task": 0, "book": book}, st_path)
    prov = {"run_id": EXPERIMENT, "status": "RUNNING", "meta": meta, "env": NK.env_info(device),
            "spec": str(SPEC.relative_to(ROOT)), "spec_commit_registered": SPEC_COMMIT,
            "gits": book["gits"], "resumed": book["resumed"], "sources_sha256": NK.source_hashes(),
            "configs": {a: M.parse_cfg(a) for a in sorted({a for a, s in eng.slots})}, "scientific_verdict": None}
    NK.write_json(out / "provenance.json", prov)
    for t in range(t0, n_tasks + 1):
        tt = time.time()
        hard = t % 2 == 1
        batches = torch.stack([prep["batches"][s][t - 1] for a, s in eng.slots]).to(device)
        tr = NK.slot_rows(eng.slots, plans, t, data.tr_rows).to(device)
        te = NK.slot_rows(eng.slots, plans, t, data.te_rows).to(device)
        sce, sacc = NK.ce_acc(eng.P, eng.act, data.X, data.Y, tr)
        ctl0 = NK.control_arrays(eng.act)
        trs0, _ = unit_stats2(eng.P, eng.act, data.X, tr, chunk=chunk)
        t_train = time.time()
        eng.train_task(batches)
        t_train = time.time() - t_train
        rec = eng.records()
        bad = eng.bad_step.cpu().numpy()
        finite = np.array([all(bool(torch.isfinite(p[r]).all()) for p in eng.P) for r in range(R)])
        for r in range(R):
            key = f"{eng.slots[r][0]}|{eng.slots[r][1]}"
            if (bad[r] >= 0 or not finite[r]) and key not in book["diverged"]:
                book["diverged"][key] = {"task": t, "bad_step_in_task": int(bad[r]), "params_finite": bool(finite[r])}
        np.savez_compressed(out / "steps" / f"t{t:02d}.npz", ce=rec["ce"].astype(np.float32),
                            hit=rec["hit"].astype(np.float32))
        t_diag = time.time()
        trs1, tr_acc = unit_stats2(eng.P, eng.act, data.X, tr, data.Y, chunk=chunk)
        prs, _ = unit_stats2(eng.P, eng.act, data.X, probe[None].expand(R, -1), chunk=chunk)
        _, te_acc = NK.ce_acc(eng.P, eng.act, data.Xte, data.Yte, te)
        ctl1 = NK.control_arrays(eng.act)
        wn = NK.weight_row_norms(eng.P)
        t_diag = time.time() - t_diag
        units = {**unit_arrays(prs, "probe_"), **unit_arrays(trs0, "train_start_"), **unit_arrays(trs1, "train_end_")}
        for l, tag in enumerate(SITES):
            for k in ("alpha", "V"):
                units[f"ctl_{tag}_{k}"] = ctl1[k][l].astype(np.float32)
        for tag, v in wn.items():
            units[f"wnorm_{tag}"] = v.astype(np.float32)
        book["probe"] += site_rows(prs, ctl1, eng, {"boundary": t})
        book["train"] += site_rows(trs0, ctl0, eng, {"task": t, "phase": "start"})
        book["train"] += site_rows(trs1, ctl1, eng, {"task": t, "phase": "end"})
        for r, (name, seed) in enumerate(eng.slots):
            cfg = eng.cfgs[r]
            snake = cfg["kind"] in ("snake", "kkt1")
            ce, hit = rec["ce"][r], rec["hit"][r]
            cls = plans[seed][t - 1][1]
            key = f"{name}|{seed}"
            row = {"arm": cfg["arm"], "cfg": name, "lr": cfg["lr"], "budget": budget, "seed": seed, "slot": r,
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
                a1 = ctl1["alpha"][l][r]
                row[f"alpha_med_{tag}"] = float(np.median(a1)) if snake else None
                row[f"two_alpha_sqrtV_med_{tag}"] = (float(np.median(2 * a1 * np.sqrt(ctl1["V"][l][r])))
                                                    if snake else None)
                row[f"zsd_med_{tag}"] = float(np.median(trs1[tag]["s"][r]))
                row[f"zbar_med_{tag}"] = float(np.median(trs1[tag]["m"][r]))
                row[f"dmean_med_{tag}"] = float(np.median(trs1[tag]["dmean"][r]))
            book["rows"].append(row)
        # task-1 boundary: SNAfrz1 is frozen after its last update of task 1 (Lref kept as a record)
        if t == 1:
            refs, frz = {}, []
            for r, (name, seed) in enumerate(eng.slots):
                c = eng.cfgs[r]
                if c["arm"] in NK.SNAKE_ARMS and np.isfinite(rec["ce"][r][-NK.REF_LAST:]).all() and budget == BASE_STEPS:
                    cal = NK.reference_loss(rec["ce"][r])
                    book["calibration"][f"{name}|{seed}"] = {**cal, "used_for_control": False}
                if c["base"] == "SNAfrz1":
                    frz.append(r)
            eng.act.freeze_runs(frz)
            alpha_f = NK.control_arrays(eng.act)["alpha"]
            for r in frz:
                book["frozen_alpha"][f"{eng.slots[r][0]}|{eng.slots[r][1]}"] = {
                    tag: [float(x) for x in alpha_f[l][r]] for l, tag in enumerate(SITES)}
            units.update({f"frozen_alpha_{tag}": alpha_f[l].astype(np.float32) for l, tag in enumerate(SITES)})
        np.savez(out / "units" / f"b{t:02d}.npz", **units)
        book["timing"].append({"task": t, "train_s": t_train, "diag_s": t_diag, "total_s": time.time() - tt})
        if t in ckpt_boundaries:
            NK.save_atomic({"meta": meta, "state": eng.full_state(), "boundary": t}, out / "checkpoints" / f"b{t:02d}.pt")
        NK.write_csv(out / "per_task.csv", book["rows"])
        NK.save_atomic({"meta": meta, "state": eng.full_state(), "task": t, "book": book}, st_path)
        log(f"[{time.strftime('%T')}] budget {budget} task {t:2d}/{n_tasks} {'hard' if hard else 'easy'} "
            f"train {t_train:6.1f}s diag {t_diag:4.1f}s diverged {len(book['diverged'])}")
        if stop_after is not None and t == stop_after:
            raise KeyboardInterrupt(f"stop_after={t} (check hook)")
    # ---- fresh controls (spec §5.2)
    fresh_rows = []
    cont = {(q["cfg"], q["seed"], q["task"]): q for q in book["rows"]}
    for tf in fresh_tasks:
        tt = time.time()
        fe = BudgetEngine(eng.slots, data, device, budget, graph=graph, stage="fresh")
        batches = torch.stack([prep["batches"][s][tf - 1] for a, s in eng.slots]).to(device)
        fe.train_task(batches)
        frec = fe.records()
        tr = NK.slot_rows(eng.slots, plans, tf, data.tr_rows).to(device)
        te = NK.slot_rows(eng.slots, plans, tf, data.te_rows).to(device)
        _, ftr = NK.ce_acc(fe.P, fe.act, data.X, data.Y, tr)
        _, fte = NK.ce_acc(fe.P, fe.act, data.Xte, data.Yte, te)
        for r, (name, seed) in enumerate(eng.slots):
            c = cont[(name, seed, tf)]
            h = frec["hit"][r]
            q = {"arm": eng.cfgs[r]["arm"], "cfg": name, "budget": budget, "seed": seed, "task": tf,
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
    NK.write_json(out / "frozen_alpha.json", {"rule": "SNAfrz1's alpha after the last update of task 1 "
                                                      "of this budget, per site and unit", "runs": book["frozen_alpha"]})
    NK.write_json(out / "calibration.json", {"runs": book["calibration"]})
    prov.update(status="COMPLETED", gits=book["gits"], resumed=book["resumed"], timing=book["timing"],
                diverged=book["diverged"], wall_clock_s_this_process=time.time() - t_wall,
                cuda_max_mem_mb=(torch.cuda.max_memory_allocated() / 2 ** 20 if device.type == "cuda" else None),
                updates={"continual": R * n_tasks * budget, "fresh": R * budget * len(fresh_tasks)},
                sources_sha256=NK.source_hashes())
    NK.write_json(out / "provenance.json", prov)
    log(f"[{time.strftime('%T')}] done {out} ({(time.time() - t_wall) / 60:.1f} min, diverged {len(book['diverged'])})")
    return prov


def main_slots() -> list[tuple[str, int]]:
    return [(c, s) for c in ARMS4 for s in SEEDS]


def smoke_slots() -> list[tuple[str, int]]:
    return [(c, s) for c in ARMS4 for s in NK.SMOKE_SEEDS]


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
        for b in BUDGETS:
            run(smoke_slots(), out / f"b{b}", 2, b, device, graph=not a.no_graph, data=data,
                allow_dirty=a.allow_dirty, fresh_tasks=(1,), ckpt_boundaries=(0, 1, 2))
    else:
        if a.budget not in BUDGETS:
            raise SystemExit(f"--budget from {BUDGETS}")
        run(main_slots(), out, NK.N_TASKS, a.budget, device, graph=not a.no_graph, data=data,
            allow_dirty=a.allow_dirty)


if __name__ == "__main__":
    main()
