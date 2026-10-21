#!/usr/bin/env python3
"""Registered CNN width-EMA cap, branching from the b7800 task-1 checkpoint.

No experiment is started by importing this module. See specs/spec_cnn_width_cap_1021.md.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import fcntl
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import cifar5p1_cnn_budget_1019 as B

NK, E, M = B.NK, B.E, B.M
EXPERIMENT = "cnn_width_cap_1021"
SPEC = ROOT / "specs" / f"spec_{EXPERIMENT}.md"
ARMS = ("R_lr0.0003", "ELU_a3.6_lr0.0003")
SEEDS = tuple(range(2000, 2010))
BUDGET = 7800
CAP_BETA = .01
STATS = ("count", "applied", "ratio_sum", "ratio_min", "batch_pre_s_sum",
         "batch_post_s_sum", "pre_exceed", "post_exceed", "post_ratio_max", "ema_exceed")
CAPTURE_STREAM_OVERRIDE = False


def setup(device_str):
    """Enable PyTorch's documented stale-stream compatibility in this process."""
    global CAPTURE_STREAM_OVERRIDE
    device = NK.setup(device_str)
    setter = getattr(torch.autograd.graph, "set_override_stale_capture_stream", None)
    CAPTURE_STREAM_OVERRIDE = device.type == "cuda" and setter is not None
    if setter is not None:
        setter(CAPTURE_STREAM_OVERRIDE)
    return device


def env_info(device):
    return {**NK.env_info(device), "override_stale_capture_stream": CAPTURE_STREAM_OVERRIDE}


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def main_slots(seeds=SEEDS):
    return [(a, int(s)) for a in ARMS for s in seeds]


def subset_state(state, slots):
    """Only select the leading run axis, never reinterpret activation/channel layouts."""
    if "cap" in state:
        raise ValueError("subset_state accepts the original budget checkpoint, not a cap checkpoint")
    old = [tuple(x) for x in state["slots"]]
    idx = [old.index(tuple(s)) for s in slots]
    if len(set(slots)) != len(slots):
        raise ValueError("duplicate slots")
    out = copy.deepcopy(state)
    for key in ("P", "m", "v"):
        out[key] = [x[idx].clone() for x in state[key]]
    for key, vals in state["act"].items():
        if any(x.shape[0] != len(old) for x in vals):
            raise ValueError(f"unrecognised activation run axis: {key}")
        out["act"][key] = [x[idx].clone() for x in vals]
    if state.get("rsl_gen") or state.get("redo_gen"):
        raise ValueError("unexpected stochastic methods in source")
    out["slots"] = list(slots)
    return out, idx


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def load_rng(st):
    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch"])
    if st["cuda"]:
        torch.cuda.set_rng_state_all(st["cuda"])


def variance(q, site, runs):
    """Population variance pooled over images and space, not variance of image means."""
    if B.IS_CONV[site]:
        return q.reshape(q.shape[0], runs, B.WIDTHS[site], *q.shape[-2:]).var(
            dim=(0, 3, 4), correction=0)
    return q.var(dim=1, correction=0)


def cap_ratio(ref, vtmp):
    # No width floor: zero references and zero variance have explicit definitions.
    positive = vtmp > 0
    denom = torch.where(positive, vtmp, torch.ones_like(vtmp)).sqrt()
    return torch.where(positive, (ref / denom).clamp(max=1), torch.ones_like(ref))


def affine(P, h, site, runs, bias=True):
    w, b = P[2 * site:2 * site + 2]
    if B.IS_CONV[site]:
        cin = 3 if site == 0 else NK.CH
        return F.conv2d(h, w.reshape(runs * NK.CH, cin, NK.KER, NK.KER),
                        b.reshape(-1) if bias else None, padding=NK.PAD, groups=runs)
    if bias:
        return torch.baddbmm(b[:, None, :], h, w.transpose(1, 2))
    return torch.bmm(h, w.transpose(1, 2))


class CapEngine(B.BudgetEngine):
    def __init__(self, slots, data, device, budget=BUDGET, graph=True, dtype=torch.float32,
                 enabled=False, ema_start_first_batch=False):
        super().__init__(slots, data, device, budget, graph=graph, dtype=dtype)
        self.cap_enabled = bool(enabled)
        self.ema_start_first_batch = bool(ema_start_first_batch)
        self.s_ref = [torch.zeros(self.R, w, dtype=dtype, device=device) for w in B.WIDTHS]
        self.cap_V = [torch.zeros_like(x) for x in self.s_ref]
        self.cap_stats = {k: [torch.zeros_like(x) for x in self.s_ref] for k in STATS}
        self.reset_cap_stats()

    @torch.no_grad()
    def set_reference(self, refs, variances=None):
        for i, (ref, v, x) in enumerate(zip(self.s_ref, self.cap_V, refs)):
            x = torch.as_tensor(x, dtype=ref.dtype, device=ref.device)
            if x.shape != ref.shape or not bool(torch.isfinite(x).all()) or bool((x < 0).any()):
                raise ValueError("invalid reference widths")
            ref.copy_(x)
            v.copy_(x.square() if variances is None else torch.as_tensor(variances[i], device=v.device, dtype=v.dtype))
            if not bool(torch.isfinite(v).all()) or bool((v < 0).any()):
                raise ValueError("invalid reference variances")

    @torch.no_grad()
    def reset_cap_stats(self):
        for key, vals in self.cap_stats.items():
            for x in vals:
                x.fill_(1 if key == "ratio_min" else 0)

    @torch.no_grad()
    def project(self, xb):
        r, batch = xb.shape[:2]
        h = xb.transpose(0, 1).reshape(batch, r * 3, 32, 32)
        for site in range(4):
            # This h already includes all preceding post-cap affines/activations.
            q = affine(self.P, h, site, r, bias=False)
            v = variance(q, site, r)
            vt = self.cap_V[site] * (1 - CAP_BETA) + CAP_BETA * v
            if self.ema_start_first_batch:
                vt = torch.where(self.cap_stats["count"][site] == 0, v, vt)
            ratio = cap_ratio(self.s_ref[site], vt) if self.cap_enabled else torch.ones_like(vt)
            w = self.P[2 * site]
            w.mul_(ratio.view(r, B.WIDTHS[site], *([1] * (w.ndim - 2))))
            self.P[2 * site + 1].mul_(ratio)
            self.cap_V[site].copy_(ratio.square() * vt)
            z = affine(self.P, h, site, r, bias=True)
            vp = variance(z, site, r)
            pre, post = v.sqrt(), vp.sqrt()
            ref = self.s_ref[site]
            post_ratio = torch.where(ref > 0, post / torch.where(ref > 0, ref, torch.ones_like(ref)),
                                     torch.where(post > 0, torch.full_like(post, math.inf), torch.zeros_like(post)))
            st = {k: self.cap_stats[k][site] for k in STATS}
            st["count"].add_(1)
            st["applied"].add_((ratio < 1).to(v.dtype))
            st["ratio_sum"].add_(ratio)
            st["ratio_min"].copy_(torch.minimum(st["ratio_min"], ratio))
            st["batch_pre_s_sum"].add_(pre)
            st["batch_post_s_sum"].add_(post)
            st["pre_exceed"].add_((pre > ref).to(v.dtype))
            st["post_exceed"].add_((post > ref).to(v.dtype))
            st["post_ratio_max"].copy_(torch.maximum(st["post_ratio_max"], post_ratio))
            # Relative 8 eps only for reporting floating-point roundoff, not for projection.
            st["ema_exceed"].add_((self.cap_V[site] > ref.square() * (1 + 8 * torch.finfo(v.dtype).eps)).to(v.dtype))
            h = self.act.phi(z, site)
            if B.IS_CONV[site]:
                h = F.max_pool2d(h, NK.POOL, NK.POOL)
                if site == 1:
                    h = h.reshape(batch, r, NK.FLAT).transpose(0, 1)
        return h

    def step(self):
        super().step()
        with torch.no_grad():
            idx = self.idx.index_select(1, (self.j - 1).view(1)).squeeze(1)
            self.project(self.data.X[idx])
            finite = torch.ones(self.R, dtype=torch.bool, device=self.device)
            for x in (*self.P, *self.cap_V):
                finite &= torch.isfinite(x).flatten(1).all(1)
            self.bad_step.copy_(torch.where((self.bad_step < 0) & ~finite, self.j - 1, self.bad_step))

    def _state_tensors(self):
        return (*super()._state_tensors(), *self.s_ref, *self.cap_V,
                *(x for vals in self.cap_stats.values() for x in vals))

    def capture(self):
        """Warm up and capture on the same stream, then restore every state.

        PyTorch 2.13 rejects stale AccumulateGrad stream references when the
        warm-up and capture streams differ. This changes scheduling only; the
        eager/graph identity check verifies the numerical update separately.
        """
        keep = [q.detach().clone() for q in self._state_tensors()]
        keep_act = self.act.state()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                self.step()
        self.cg = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.cg, stream=side):
            self.step()
        torch.cuda.current_stream().wait_stream(side)
        with torch.no_grad():
            for q, saved in zip(self._state_tensors(), keep):
                q.copy_(saved)
        self.act.load_state(keep_act)
        torch.cuda.synchronize()

    @torch.no_grad()
    def begin_task(self, batches):
        if tuple(batches.shape) != tuple(self.idx.shape):
            raise ValueError("batch shape differs")
        self.idx.copy_(batches.to(self.device))
        self.j.zero_()
        self.bad_step.fill_(-1)
        for x in self.rec.values():
            x.zero_()
        self.reset_cap_stats()

    def advance(self, n_steps):
        if int(self.j) + n_steps > self.budget:
            raise ValueError("advance past task end")
        if self.graph and self.cg is None:
            self.capture()
        for _ in range(n_steps):
            self.cg.replay() if self.cg is not None else self.step()
            self.n_done += 1
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def full_state(self):
        st = super().full_state()
        st["cap"] = {"enabled": self.cap_enabled, "ema_start_first_batch": self.ema_start_first_batch,
                     "s_ref": [x.cpu().clone() for x in self.s_ref],
                     "V": [x.cpu().clone() for x in self.cap_V],
                     "stats": {k: [x.cpu().clone() for x in vals] for k, vals in self.cap_stats.items()}}
        st["progress"] = {"idx": self.idx.cpu().clone(), "j": int(self.j),
                          "bad_step": self.bad_step.cpu().clone(),
                          "rec": {k: x.cpu().clone() for k, x in self.rec.items()}}
        return st

    @torch.no_grad()
    def load_full_state(self, st):
        super().load_full_state(st)
        if "cap" not in st:
            return
        cs = st["cap"]
        if cs["enabled"] != self.cap_enabled or cs["ema_start_first_batch"] != self.ema_start_first_batch:
            raise ValueError("cap enabled mismatch")
        for dst, src in ((self.s_ref, cs["s_ref"]), (self.cap_V, cs["V"])):
            for d, s in zip(dst, src):
                d.copy_(s)
        for k, vals in self.cap_stats.items():
            for d, s in zip(vals, cs["stats"][k]):
                d.copy_(s)
        p = st["progress"]
        self.idx.copy_(p["idx"])
        self.j.fill_(p["j"])
        self.bad_step.copy_(p["bad_step"])
        for k, d in self.rec.items():
            d.copy_(p["rec"][k])


def cap_arrays(eng):
    out = {}
    for i, site in enumerate(B.SITES):
        out[f"cap_{site}_s_ref"] = eng.s_ref[i].cpu().numpy().copy()
        out[f"cap_{site}_V"] = eng.cap_V[i].cpu().numpy().copy()
        for k, vals in eng.cap_stats.items():
            out[f"cap_{site}_{k}"] = vals[i].cpu().numpy().copy()
    return out


def cap_rows(eng, task):
    rows = []
    for r, (cfg, seed) in enumerate(eng.slots):
        for i, site in enumerate(B.SITES):
            st = {k: vals[i][r].cpu().numpy() for k, vals in eng.cap_stats.items()}
            n = st["count"]
            rows.append({"cfg": cfg, "seed": seed, "task": task, "site": site,
                         "task_type": "hard" if task % 2 else "easy",
                         "condition": "cap" if eng.cap_enabled else "natural",
                         "reference_width_kind": "pooled_images_space" if i < 2 else "images",
                         "reference_zero_units": int((eng.s_ref[i][r] == 0).sum().cpu()),
                         "applied_fraction": float((st["applied"] / np.maximum(n, 1)).mean()),
                         "ratio_mean": float((st["ratio_sum"] / np.maximum(n, 1)).mean()),
                         "ratio_min": float(st["ratio_min"].min()),
                         "batch_pre_s_mean": float((st["batch_pre_s_sum"] / np.maximum(n, 1)).mean()),
                         "batch_post_s_mean": float((st["batch_post_s_sum"] / np.maximum(n, 1)).mean()),
                         "post_exceed_fraction": float((st["post_exceed"] / np.maximum(n, 1)).mean()),
                         "post_ratio_max": float(st["post_ratio_max"].max()),
                         "ema_exceed_count": float(st["ema_exceed"].sum()),
                         "ema_s_median": float(eng.cap_V[i][r].sqrt().median().cpu())})
    return rows


def no_touch_stats(eng, X, rows, Y=None, chunk=50):
    # Diagnostics cannot alter parameters/optimizer/activation or intervention state.
    before = [x.detach().clone() for x in eng._state_tensors()]
    act = eng.act.state()
    result = B.unit_stats2(eng.P, eng.act, X, rows, Y, chunk=chunk)
    if any(not torch.equal(x, y) for x, y in zip(before, eng._state_tensors())):
        raise RuntimeError("diagnostics modified engine state")
    after = eng.act.state()
    if any(not torch.equal(x, y) for k in act for x, y in zip(act[k], after[k])):
        raise RuntimeError("diagnostics modified activation state")
    return result


def load_source(path, slots):
    source = torch.load(path, map_location="cpu", weights_only=False)
    if source.get("boundary") != 1 or source["meta"].get("budget") != BUDGET:
        raise ValueError("requires b7800 boundary-1 checkpoint")
    state, idx = subset_state(source["state"], slots)
    return source, state, idx


@contextlib.contextmanager
def resource_locks(device, output, directory=False):
    """Cooperating jobs share one GPU lock; independent processes cannot reuse an output."""
    path = Path(output)
    if directory:
        path.mkdir(parents=True, exist_ok=True)
        output_lock = path / ".run.lock"
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        output_lock = path.with_name(f".{path.name}.lock")
    paths = ([Path("/tmp/lop_analysis_gpu.lock")] if str(device).startswith("cuda") else []) + [output_lock]
    handles = []
    try:
        for p in paths:
            handle = open(p, "a+")
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                handle.close()
                raise SystemExit(f"resource busy: {p}; no run started")
            handles.append(handle)
            handle.seek(0)
            handle.truncate()
            handle.write(json.dumps({"pid": os.getpid(), "run_id": EXPERIMENT, "output": str(path)}) + "\n")
            handle.flush()
        yield
    finally:
        for handle in reversed(handles):
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()


def compare_tree(a, b, path=""):
    if isinstance(a, torch.Tensor):
        return [] if isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a.cpu(), b.cpu()) else [path]
    if isinstance(a, dict):
        if not isinstance(b, dict) or set(a) != set(b):
            return [path + ".keys"]
        return [q for k in a for q in compare_tree(a[k], b[k], f"{path}.{k}")]
    if isinstance(a, (list, tuple)):
        if not isinstance(b, (list, tuple)) or len(a) != len(b):
            return [path + ".length"]
        return [q for i, (x, y) in enumerate(zip(a, b)) for q in compare_tree(x, y, f"{path}[{i}]")]
    return [] if a == b else [path]


def calibrate(args):
    """Measure the task-1 post-Adam EMA on the original full 40-slot trajectory."""
    if NK.git_state()["dirty_src_analysis_specs"] or "DRAFT" in SPEC.read_text().split("## 1.")[0]:
        raise SystemExit("commit frozen spec/source before real-data calibration")
    out = Path(args.out)
    if out.exists() or out.with_suffix(".json").exists():
        raise SystemExit(f"calibration already exists: {out}")
    device = setup(args.device)
    data = NK.Data(device, cifar=B.C51.Cifar100(Path(args.data)))
    initial = torch.load(args.source_b00, map_location="cpu", weights_only=False)
    target = torch.load(args.source_b01, map_location="cpu", weights_only=False)
    if initial.get("boundary") != 0 or target.get("boundary") != 1:
        raise ValueError("calibration requires b00 and b01")
    slots = [tuple(s) for s in initial["state"]["slots"]]
    if slots != B.main_slots() or slots != [tuple(s) for s in target["state"]["slots"]]:
        raise ValueError("calibration requires the original full 40-slot order")
    prep = B.prepare(SEEDS, data, BUDGET)
    for source in (initial, target):
        meta = source["meta"]
        if meta["budget"] != BUDGET or meta["data_sha256"] != data.sha256 or meta["probe_sha256"] != prep["probe_sha256"]:
            raise ValueError("calibration data/meta mismatch")
        if any(meta["batch_sha256"][str(s)] != prep["batch_sha256"][s] for s in SEEDS):
            raise ValueError("calibration batch mismatch")
    e = CapEngine(slots, data, device, enabled=False, graph=not args.no_graph, ema_start_first_batch=True)
    e.load_full_state(initial["state"])
    e.begin_task(torch.stack([prep["batches"][s][0] for _, s in slots]))
    t0 = time.time()
    for _ in range(BUDGET // 780):
        e.advance(780)
        if bool((e.bad_step >= 0).any()):
            raise RuntimeError("nonfinite calibration; no reference accepted")
    # The historical runner freezes SNAfrz1 exactly at the task-1 boundary.
    e.act.freeze_runs([r for r, c in enumerate(e.cfgs) if c["base"] == "SNAfrz1"])
    replay_state = B.BudgetEngine.full_state(e)
    differences = compare_tree(replay_state, target["state"])
    report = {"run_id": EXPERIMENT, "role": "post_adam_task1_reference_ema",
              "status": "PASS" if not differences else "FAIL_B01_STATE_MISMATCH",
              "state_bit_equal": not differences, "state_difference_paths": differences,
              "source_b00_sha256": sha_file(args.source_b00), "source_b01_sha256": sha_file(args.source_b01),
              "data_sha256": data.sha256, "probe_sha256": prep["probe_sha256"],
              "batch_sha256_task1": {str(s): prep["batch_sha256"][s][0] for s in SEEDS},
              "slots": slots, "beta": CAP_BETA, "initialization": "first_post_adam_batch_variance",
              "spec_sha256": sha_file(SPEC), "git": NK.git_state(), "env": env_info(device),
              "sources_sha256": NK.source_hashes(),
              "seconds": time.time() - t0}
    NK.write_json(out.with_suffix(".json"), report)
    if differences:
        raise RuntimeError(f"calibration must match b01 exactly; differences: {differences[:8]}")
    NK.save_atomic({"meta": report, "V_ref": [x.cpu().clone() for x in e.cap_V],
                    "s_ref": [x.cpu().sqrt().clone() for x in e.cap_V]}, out)
    print(json.dumps(report, indent=2))


def run(args):
    start = time.time()
    git = NK.git_state()
    if git["dirty_src_analysis_specs"]:
        raise SystemExit("commit src/analysis/specs before a registered run")
    if "DRAFT" in SPEC.read_text().split("## 1.")[0]:
        raise SystemExit("spec is DRAFT")
    device = setup(args.device)
    data = NK.Data(device, cifar=B.C51.Cifar100(Path(args.data)))
    slots = main_slots(tuple(int(s) for s in args.seeds.split(",")))
    source, selected, indices = load_source(Path(args.source), slots)
    calibration = torch.load(args.calibration, map_location="cpu", weights_only=False)
    cm = calibration["meta"]
    if cm["status"] != "PASS" or not cm["state_bit_equal"] or cm["source_b01_sha256"] != sha_file(args.source):
        raise ValueError("calibration not validated for this b01")
    if cm["spec_sha256"] != sha_file(SPEC) or [tuple(s) for s in cm["slots"]] != [tuple(s) for s in source["state"]["slots"]]:
        raise ValueError("calibration spec/slots mismatch")
    if cm["sources_sha256"] != NK.source_hashes():
        raise ValueError("calibration used different source code")
    prep = B.prepare(sorted({s for _, s in slots}), data, BUDGET)
    meta0 = source["meta"]
    if meta0["data_sha256"] != data.sha256 or meta0["probe_sha256"] != prep["probe_sha256"]:
        raise ValueError("source data/probe hash mismatch")
    for seed in {s for _, s in slots}:
        if meta0["batch_sha256"][str(seed)] != prep["batch_sha256"][seed]:
            raise ValueError("source batch hash mismatch")
    eng = CapEngine(slots, data, device, enabled=args.condition == "cap", graph=not args.no_graph)
    eng.load_full_state(selected)
    probe = prep["probe"].to(device)[None].expand(eng.R, -1)
    out = Path(args.out)
    for directory in ("checkpoints", "units", "steps"):
        (out / directory).mkdir(parents=True, exist_ok=True)
    meta = {"run_id": EXPERIMENT, "condition": args.condition, "slots": slots,
            "budget": BUDGET, "source_path": str(Path(args.source).resolve()),
            "source_sha256": sha_file(args.source), "source_meta": meta0, "source_indices": indices,
            "calibration_path": str(Path(args.calibration).resolve()), "calibration_sha256": sha_file(args.calibration),
            "data_path": str(Path(args.data).resolve()), "data_sha256": data.sha256,
            "batch_sha256": {str(s): v for s, v in prep["batch_sha256"].items()},
            "probe_sha256": prep["probe_sha256"], "spec_sha256": sha_file(SPEC), "graph": eng.graph,
            "sources_sha256": NK.source_hashes(), "initial_runtime_precision": env_info(device)}
    statepath = out / "state.pt"
    book = {"rows": [], "probe": [], "train": [], "cap": [], "timing": [], "gits": [git]}
    active_task, task_started = 2, False
    if statepath.exists():
        old = torch.load(statepath, map_location="cpu", weights_only=False)
        if old["meta"] != meta:
            raise ValueError("resume meta mismatch")
        eng.load_full_state(old["state"])
        if bool((eng.bad_step >= 0).any()):
            raise RuntimeError("saved run is nonfinite; do not resume it as a normal continuation")
        load_rng(old["rng"])
        book, active_task, task_started = old["book"], old["active_task"], old["task_started"]
        book["gits"].append(git)
    else:
        if any(p.is_file() and p.name != ".run.lock" for p in out.iterdir()):
            raise ValueError("nonempty output without resumable state")
        stats, _ = no_touch_stats(eng, data.X, probe)
        eng.set_reference([x[indices] for x in calibration["s_ref"]], [x[indices] for x in calibration["V_ref"]])
        NK.write_json(out / "reference_widths.json", {
            "source": "task1_post_adam_batch_ema", "probe_sha256": prep["probe_sha256"],
            "sites": {tag: {"kind": "pooled_images_space" if i < 2 else "images",
                             "s_ref": eng.s_ref[i].cpu().tolist(),
                             "V_ref": eng.cap_V[i].cpu().tolist(),
                             "boundary1_fixed_probe_s": stats[tag]["s"].tolist(),
                             "zero_units_per_slot": (eng.s_ref[i] == 0).sum(1).cpu().tolist(),
                             **({k: stats[tag][k].tolist() for k in ("s_within", "s_between")} if i < 2 else {})}
                      for i, tag in enumerate(B.SITES)}})
        book["probe"] += B.site_rows(stats, NK.control_arrays(eng.act), eng, {"boundary": 1})
        np.savez(out / "units" / "b01.npz", **B.unit_arrays(stats, "probe_"), **cap_arrays(eng))
        NK.save_atomic({"meta": meta, "state": eng.full_state(), "boundary": 1}, out / "checkpoints" / "b01.pt")
    prov = {"meta": meta, "status": "RUNNING", "gits": book["gits"], "env": meta["initial_runtime_precision"],
            "source_hashes": NK.source_hashes(), "scientific_verdict": None}
    NK.write_json(out / "provenance.json", prov)

    def save(task, started):
        NK.save_atomic({"meta": meta, "state": eng.full_state(), "rng": rng_state(),
                        "book": book, "active_task": task, "task_started": started}, statepath)

    save(active_task, task_started)
    for task in range(active_task, args.stop_task + 1):
        tt = time.time()
        train_rows = NK.slot_rows(slots, prep["plans"], task, data.tr_rows).to(device)
        test_rows = NK.slot_rows(slots, prep["plans"], task, data.te_rows).to(device)
        if not (task == active_task and task_started):
            train_start, _ = no_touch_stats(eng, data.X, train_rows)
            book["train"] += B.site_rows(train_start, NK.control_arrays(eng.act), eng,
                                        {"task": task, "phase": "start"})
            np.savez(out / "units" / f"t{task:02d}_start.npz", **B.unit_arrays(train_start, "train_start_"))
            batches = torch.stack([prep["batches"][s][task - 1] for _, s in slots])
            eng.begin_task(batches)
            save(task, True)
        while int(eng.j) < BUDGET:
            n = min(args.chunk_steps, BUDGET - int(eng.j))
            eng.advance(n)
            save(task, True)
            if bool((eng.bad_step >= 0).any()):
                prov.update(status="FAILED_NONFINITE", task=task, bad_step=eng.bad_step.cpu().tolist())
                NK.write_json(out / "provenance.json", prov)
                raise RuntimeError("nonfinite run; saved all slots without seed exclusion")
        rec = eng.records()
        np.savez_compressed(out / "steps" / f"t{task:02d}.npz", ce=rec["ce"].astype(np.float32),
                            hit=rec["hit"].astype(np.float32))
        stats, _ = no_touch_stats(eng, data.X, probe)
        train_end, tracc = no_touch_stats(eng, data.X, train_rows, data.Y)
        _, teacc = NK.ce_acc(eng.P, eng.act, data.Xte, data.Yte, test_rows)
        ctl = NK.control_arrays(eng.act)
        book["probe"] += B.site_rows(stats, ctl, eng, {"boundary": task})
        book["train"] += B.site_rows(train_end, ctl, eng, {"task": task, "phase": "end"})
        book["cap"] += cap_rows(eng, task)
        for r, (cfg, seed) in enumerate(slots):
            hit, ce = rec["hit"][r], rec["ce"][r]
            book["rows"].append({"cfg": cfg, "arm": eng.cfgs[r]["arm"], "seed": seed,
                                 "condition": args.condition, "task": task, "hard": int(task % 2 == 1),
                                 "task_type": "hard" if task % 2 else "easy",
                                 "first100_online_acc": float(hit[:100].mean()),
                                 "first780_online_acc": float(hit[:780].mean()),
                                 "online_acc": float(hit.mean()), "train_acc": float(tracc[r]),
                                 "test_acc": float(teacc[r]), "ce_mean": float(ce.mean()),
                                 "ce_first780_mean": float(ce[:780].mean()),
                                 "bad_step": int(eng.bad_step[r])})
        np.savez(out / "units" / f"b{task:02d}.npz", **B.unit_arrays(stats, "probe_"),
                 **B.unit_arrays(train_end, "train_end_"), **cap_arrays(eng))
        book["timing"].append({"task": task, "seconds_this_process": time.time() - tt})
        if task in (8, 9, 18, 19, 28, 29, 30):
            NK.save_atomic({"meta": meta, "state": eng.full_state(), "boundary": task},
                           out / "checkpoints" / f"b{task:02d}.pt")
        for name, key in (("per_task", "rows"), ("probe_stats", "probe"),
                          ("train_stats", "train"), ("cap_stats", "cap")):
            NK.write_csv(out / f"{name}.csv", book[key])
        save(task + 1, False)
        print(f"{args.condition} task {task}/30 done, {time.time()-tt:.1f}s", flush=True)
    prov.update(status="COMPLETED" if args.stop_task == 30 else "STOPPED_AT_REQUESTED_BOUNDARY",
                gits=book["gits"], timing=book["timing"], seconds_this_process=time.time() - start)
    NK.write_json(out / "provenance.json", prov)


def exact_sign(deltas):
    pos, neg = sum(x > 0 for x in deltas), sum(x < 0 for x in deltas)
    n = pos + neg
    p = min(1., 2 * sum(math.comb(n, k) for k in range(min(pos, neg) + 1)) / 2**n) if n else 1.
    return {"positive": pos, "negative": neg, "zero": len(deltas) - n, "p": p}


def reproduce(args):
    """Registered numerical qualification; no intervention outcomes are selected here."""
    import gc
    if NK.git_state()["dirty_src_analysis_specs"] or "DRAFT" in SPEC.read_text().split("## 1.")[0]:
        raise SystemExit("commit a frozen spec and source before real-data reproduction")
    device = setup(args.device)
    data = NK.Data(device, cifar=B.C51.Cifar100(Path(args.data)))
    source = torch.load(args.source, map_location="cpu", weights_only=False)
    if source.get("boundary") != args.task - 1 or source["meta"]["budget"] != BUDGET:
        raise ValueError("checkpoint boundary must precede replay task")
    source_slots = [tuple(s) for s in source["state"]["slots"]]
    selected_slots = main_slots()
    prep = B.prepare(sorted({s for _, s in source_slots}), data, BUDGET)
    if source["meta"]["data_sha256"] != data.sha256 or source["meta"]["probe_sha256"] != prep["probe_sha256"]:
        raise ValueError("source data/probe mismatch")
    for seed in prep["batch_sha256"]:
        if source["meta"]["batch_sha256"][str(seed)] != prep["batch_sha256"][seed]:
            raise ValueError("batch mismatch")
    saved = np.load(args.saved_steps)

    def replay(slots, sham):
        state, indices = subset_state(source["state"], slots)
        e = (CapEngine(slots, data, device, enabled=False, graph=not args.no_graph) if sham else
             B.BudgetEngine(slots, data, device, BUDGET, graph=not args.no_graph))
        e.load_full_state(state)
        batches = torch.stack([prep["batches"][s][args.task - 1] for _, s in slots])
        if sham:
            e.set_reference([torch.ones_like(x) for x in e.s_ref])
            e.begin_task(batches)
            e.advance(args.steps)
        else:
            e.train_task(batches, n_steps=args.steps)
        rec = {k: v[:, :args.steps].copy().astype(np.float32) for k, v in e.records().items() if k in ("ce", "hit")}
        comp = {k: {"bit_equal": bool(np.array_equal(rec[k], saved[k][indices, :args.steps])),
                    "max_abs_error": float(np.max(np.abs(rec[k] - saved[k][indices, :args.steps])))} for k in rec}
        del e
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return rec, comp

    full, full_comp = replay(source_slots, False)
    selected, selected_comp = replay(selected_slots, False)
    sham, sham_comp = replay(selected_slots, True)
    exact_sham = {k: bool(np.array_equal(selected[k], sham[k])) for k in selected}
    report = {"run_id": EXPERIMENT, "role": "numerical_reproduction_before_intervention",
              "task": args.task, "steps": args.steps, "source_sha256": sha_file(args.source),
              "saved_steps_sha256": sha_file(args.saved_steps), "spec_sha256": sha_file(SPEC),
              "git": NK.git_state(), "env": env_info(device), "full40_vs_saved": full_comp,
              "selected20_vs_saved": selected_comp, "natural_sham20_vs_saved": sham_comp,
              "natural20_vs_sham20_bit_equal": exact_sham,
              "source_slots": source_slots, "selected_slots": selected_slots}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    NK.write_json(out, report)
    np.savez_compressed(out.with_suffix(".npz"), **{f"full_{k}": v for k, v in full.items()},
                        **{f"selected_{k}": v for k, v in selected.items()},
                        **{f"sham_{k}": v for k, v in sham.items()})
    print(json.dumps(report, indent=2))
    if not all(exact_sham.values()):
        raise RuntimeError("new engine natural sham differs from matched same-size natural")


def summarize(args):
    import csv
    def read(folder):
        with open(Path(folder) / "per_task.csv") as f:
            return list(csv.DictReader(f))
    folders = {"natural": Path(args.natural), "cap": Path(args.cap)}
    rows, issues = {}, []
    expected_keys = {(cfg, seed, task) for cfg, seed in main_slots() for task in range(2, 31)}
    for cond, folder in folders.items():
        try:
            prov = json.loads((folder / "provenance.json").read_text())
            rows[cond] = read(folder)
        except (OSError, ValueError) as exc:
            issues.append(f"{cond}: missing/unreadable output: {exc}")
            continue
        if prov.get("status") != "COMPLETED":
            issues.append(f"{cond}: status is {prov.get('status')}")
        meta = prov.get("meta", {})
        if meta.get("condition") != cond or meta.get("budget") != BUDGET or [tuple(s) for s in meta.get("slots", [])] != main_slots():
            issues.append(f"{cond}: registered configuration differs")
        try:
            keys = [(r["cfg"], int(r["seed"]), int(r["task"])) for r in rows[cond]]
            if set(keys) != expected_keys or len(keys) != len(expected_keys):
                issues.append(f"{cond}: missing/extra/duplicate task2–30 records")
            if any(int(r["bad_step"]) != -1 for r in rows[cond]):
                issues.append(f"{cond}: nonfinite step recorded")
            metric_keys = ("first100_online_acc", "first780_online_acc", "online_acc", "train_acc", "test_acc", "ce_mean", "ce_first780_mean")
            if any(not math.isfinite(float(r[k])) for r in rows[cond] for k in metric_keys):
                issues.append(f"{cond}: nonfinite evaluation")
        except (KeyError, ValueError) as exc:
            issues.append(f"{cond}: malformed records: {exc}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if issues:
        NK.write_json(out / "primary.json", {"status": "NOT_ESTABLISHED", "issues": issues})
        (out / "summary.md").write_text("# CNN width cap: primary result not established\n\n" +
                                        "\n".join(f"- {issue}" for issue in issues) + "\n")
        raise SystemExit("primary result not established; see recorded issues")
    results = []
    for arm in ARMS:
        delta = []
        pairs = []
        for seed in SEEDS:
            vals = {}
            early = {}
            for cond in rows:
                picked = [r for r in rows[cond] if r["cfg"] == arm and int(r["seed"]) == seed
                          and int(r["task"]) in NK.LATE]
                if len(picked) != len(NK.LATE) or {int(r["task"]) for r in picked} != set(NK.LATE):
                    raise ValueError("incomplete/duplicate primary data")
                vals[cond] = np.mean([float(r["first780_online_acc"]) for r in picked])
                first = [r for r in rows[cond] if r["cfg"] == arm and int(r["seed"]) == seed
                         and int(r["task"]) in (3, 5, 7, 9)]
                if len(first) != 4 or {int(r["task"]) for r in first} != {3, 5, 7, 9}:
                    raise ValueError("incomplete/duplicate early secondary data")
                early[cond] = float(np.mean([float(r["first780_online_acc"]) for r in first]))
            d = float(vals["cap"] - vals["natural"])
            if not math.isfinite(d):
                raise ValueError("nonfinite primary data; cannot exclude seed")
            delta.append(d)
            pairs.append({"seed": seed, **vals, "difference": d, "early": early,
                          "late_minus_early": {cond: float(vals[cond] - early[cond]) for cond in rows}})
        results.append({"cfg": arm, "pairs": pairs, "mean_difference": float(np.mean(delta)),
                        "median_difference": float(np.median(delta)), **exact_sign(delta)})
    order = sorted(range(2), key=lambda i: results[i]["p"])
    prev = 0.
    for rank, i in enumerate(order):
        prev = max(prev, min(1., (2 - rank) * results[i]["p"]))
        results[i]["p_holm"] = prev
        sign_direction = results[i]["positive"] - results[i]["negative"]
        if prev >= .05:
            verdict = "INCONCLUSIVE"
        elif sign_direction * results[i]["mean_difference"] <= 0:
            verdict = "HETEROGENEOUS_SIGN_VS_MEAN"
        else:
            verdict = "IMPROVES" if sign_direction > 0 else "WORSENS"
        results[i]["verdict"] = verdict
    NK.write_json(out / "primary.json", {"status": "ESTABLISHED", "metric": "latehard_first780_online_acc", "results": results})
    lines = ["# CNN width cap: primary result", "", "Paired seeds; five late hard tasks averaged per seed.", ""]
    for r in results:
        lines += [f"- {r['cfg']}: {r['verdict']}; mean cap−natural {100*r['mean_difference']:.3f} pp, "
                  f"signs {r['positive']}/{r['negative']}/{r['zero']}, Holm p={r['p_holm']:.6g}."]
    (out / "summary.md").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sp = ap.add_subparsers(dest="command", required=True)
    r = sp.add_parser("run")
    r.add_argument("--source", required=True)
    r.add_argument("--calibration", required=True)
    r.add_argument("--data", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--condition", choices=("natural", "cap"), required=True)
    r.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    r.add_argument("--device", default="cuda")
    r.add_argument("--no-graph", action="store_true")
    r.add_argument("--chunk-steps", type=int, default=780)
    r.add_argument("--stop-task", type=int, default=30)
    q = sp.add_parser("reproduce")
    q.add_argument("--source", required=True)
    q.add_argument("--saved-steps", required=True)
    q.add_argument("--data", required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--task", type=int, default=2)
    q.add_argument("--steps", type=int, default=32)
    q.add_argument("--device", default="cuda")
    q.add_argument("--no-graph", action="store_true")
    c = sp.add_parser("calibrate")
    c.add_argument("--source-b00", required=True)
    c.add_argument("--source-b01", required=True)
    c.add_argument("--data", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--device", default="cuda")
    c.add_argument("--no-graph", action="store_true")
    s = sp.add_parser("summarize")
    s.add_argument("--natural", required=True)
    s.add_argument("--cap", required=True)
    s.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.command == "run":
        if not 2 <= args.stop_task <= 30 or args.chunk_steps < 1:
            ap.error("stop-task in 2..30 and chunk-steps >=1")
        with resource_locks(args.device, args.out, directory=True):
            run(args)
    elif args.command == "reproduce":
        if not 2 <= args.task <= 30 or not 1 <= args.steps <= BUDGET:
            ap.error("reproduce task in 2..30 and steps in 1..7800")
        with resource_locks(args.device, args.out):
            reproduce(args)
    elif args.command == "calibrate":
        with resource_locks(args.device, args.out):
            calibrate(args)
    else:
        with resource_locks("cpu", args.out, directory=True):
            summarize(args)


if __name__ == "__main__":
    main()
