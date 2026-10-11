#!/usr/bin/env python3
"""adamw_dose2_1011, A1: AdamW (decoupled weight decay, all six tensors) on RL-CIFAR x ELU->ELU / std.

Spec: specs/spec_adamw_dose2_1011.md (sections 2.1-2.3, 3.1).  The box is cap_cifar_ee_0920's ref and nothing in it
is modified: src/cifar_interventions_0920.Engine (stacked R = 10, CUDA graph, the host's Adam and streams) is
subclassed, and its step() is copied line for line with one addition, before the host's Adam update of each tensor:

    p.mul_(wd_c)      wd_c = a float32 0-dim device tensor holding fl32(1 - 1e-3 * lam), written in place after the
                      graph is captured (the capture runs with wd_c = 1); a device counter per tensor counts the decays

lam = 0 puts no decay op and no counter in the step at all: the host's step, op for op (checked by S0 against the
cap_cifar_ee_0920 ref record).  Nothing is added to the gradient.  The arms branch from cap_cifar_ee_0920's common
task-1 prefix (prefix/t01.pt, archived) and run tasks 2-50, as the caps did.

    run_arm(arm, out)        one arm, seeds 0-9 stacked, tasks 2..50 (resumable at epoch boundaries; STOP file)
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import torch.nn.functional as F

from src import cifar_interventions_0920 as CI                    # the host engine; not modified
from src.cifar_interventions_0920 import (E, Engine, core_state, cpu, gtrain, load_pt, make_inputs, save_pt,
                                          tree_hash)
from analysis.cifar_ledger_0920.replay import environment, put, save_npz, setup, sha

EXPERIMENT = "adamw_dose2_1011"
SPEC = "specs/spec_adamw_dose2_1011.md"
PREREG_COMMIT = "7525f9747a6a2872c30eb380fe3da4809bc6fa95"
ROOT = Path(__file__).resolve().parents[1]
LR = 1e-3                                       # the host's Adam lr (Engine.step: p.sub_(1e-3 * ...))
TENSORS = ("W1", "b1", "W2", "b2", "W3", "b3")
# registered arms (spec 2.2) and the lambda = 0 path S0 runs; 'ro1' (W3, b3 only) is the S-ctrl mutation control
ARM_LAMBDA = {"ref0": 0.0, "wd3e-2": 0.03, "wd1e-1": 0.1, "wd1e0": 1.0, "ro1": 1.0}
ARM_SCOPE = {"ref0": "all", "wd3e-2": "all", "wd1e-1": "all", "wd1e0": "all", "ro1": "readout"}
SCOPES = {"all": (0, 1, 2, 3, 4, 5), "readout": (4, 5)}
REGISTERED_ARMS = ("wd3e-2", "wd1e-1", "wd1e0")
SEEDS = tuple(range(10))
TASKS, EPOCHS = 50, 400
STEPS_PER_TASK = EPOCHS * E.STEPS_PER_EPOCH      # 30,000
ZF = -16.982107162475586                        # F.elu(z) == -1 in float32 on CUDA for z <= ZF (spec 1.1 (a))
CAP_RUN = ROOT / "results" / "cap_cifar_ee_0920"  # the record (archived parts resolve through its manifest)
CAP_PREFIX = CAP_RUN / "prefix" / "t01.pt"
SAVE_STATES = (10, 30, 50)


def decay_factor(lam: float) -> float:
    """1 - lr * lam as a python double; filling the float32 wd_c with it rounds once, to fl32(1 - lr * lam)."""
    return 1 - LR * lam


def c32(lam: float) -> np.float32:
    return np.float32(decay_factor(lam))


def manifest_sha(path: Path) -> str | None:
    """sha256 the cap_cifar_ee_0920 backup manifest records for an archived file of that run."""
    m = json.loads((CAP_RUN / "backup_manifest.json").read_text())
    rel = str(Path(path).resolve().relative_to(ROOT))
    hit = [x for x in m["files"] if x.get("relative") == rel]
    return hit[0]["sha256"] if len(hit) == 1 else None


@torch.no_grad()
def adamw_update_(P, grads, m, v, inv1, inv2, wd_c, decay_idx, wd_count) -> None:
    """Engine.step's Adam loop, op for op, with the decoupled decay first on the tensors in decay_idx (spec 2.1)."""
    for i, (p, gr, mi, vi) in enumerate(zip(P, grads, m, v)):
        if i in decay_idx:
            p.mul_(wd_c)
            wd_count[i].add_(1)
        mi.mul_(0.9).add_(gr, alpha=1 - 0.9)
        vi.mul_(0.999).addcmul_(gr, gr, value=1 - 0.999)
        p.sub_(1e-3 * (mi * inv1) / ((vi * inv2).sqrt() + 1e-8))


class WDEngine(Engine):
    """cifar_interventions_0920.Engine with AdamW's decoupled decay (spec 2.1)."""

    def __init__(self, seeds, X, state=None, lam: float = 0.0, scope: str = "all", graph: bool = True):
        self.lam = float(lam)
        self.scope = scope
        self.decay = self.lam != 0.0
        self.decay_idx = SCOPES[scope]
        self.wd_c = torch.ones((), device=X.device)                    # 1 while the graph is captured
        self.wd_count = torch.zeros(6, dtype=torch.long, device=X.device)
        self.hook = None                                                # checks only (eager steps)
        super().__init__(seeds, X, state=state, graph=graph)
        if self.decay:
            self.wd_c.fill_(decay_factor(self.lam))                    # in place: the captured step reads this address

    def step(self):
        # Engine.step, line for line, with the decay before the Adam update (spec 2.1)
        xb, yb = self.X[self.ar, self.ids], self.Y[self.ar, self.ids]
        vals = self.forward(xb, self.ids)
        z3 = vals[-1]
        lossv = F.cross_entropy(z3.reshape(-1, E.N_CLASSES), yb.reshape(-1),
                                reduction='none').view(self.R, E.BATCH).mean(1)
        hit = (z3.detach().argmax(-1) == yb).float().mean(1)
        self.acc_sum.add_(hit); self.last_hit.copy_(hit)
        grads = torch.autograd.grad(lossv.sum(), self.P)
        with torch.no_grad():
            self.ce_sum.add_(lossv.detach())
            bad = ~torch.isfinite(lossv)
            self.bad_step.copy_(torch.where((self.bad_step < 0) & bad, self.step_t, self.bad_step))
            self.step_t.add_(1)
            if self.hook is not None:
                self.hook("grads", grads=grads, engine=self)
            adamw_update_(self.P, grads, self.m, self.v, self.inv1, self.inv2, self.wd_c,
                          self.decay_idx if self.decay else (), self.wd_count)
            if self.hook is not None:
                self.hook("post", engine=self)
        self.last_values = tuple(q.detach() for q in vals)

    def mutable_tensors(self):
        return super().mutable_tensors() + [self.wd_count]

    def state(self):
        st = super().state()
        st["wd"] = cpu(dict(lam=self.lam, scope=self.scope, c_hex=float(c32(self.lam)).hex(), wd_count=self.wd_count))
        return st

    def restore(self, st):
        super().restore(st)
        if "wd" in st:
            assert st["wd"]["lam"] == self.lam and st["wd"]["scope"] == self.scope, "decay configuration"
            self.wd_count.copy_(st["wd"]["wd_count"])

    def train_task(self, task, epochs=EPOCHS, **kwargs):
        if not self.in_task:
            self.wd_count.zero_()
        return super().train_task(task, epochs, **kwargs)

    def diagnostic(self):
        rows, unit = super().diagnostic()
        unit["Wnorm_l3"] = self.P[4].detach().double().norm(dim=2).cpu().numpy()
        return rows, unit

    @torch.no_grad()
    def wd_rows(self, arm: str, epochs: int = EPOCHS) -> list[dict]:
        """spec 3.1 wd_task: the decay record and the layer-2 pair fractions on the 1200 images (no RNG used)."""
        ids = torch.arange(E.N_IMAGES, device=self.device)[None, :].expand(self.R, -1)
        z1, a1, z2, a2, logits = self.forward(self.X, ids)
        counts = [int(v) for v in self.wd_count.cpu().tolist()]
        c_exp = float(c32(self.lam)) if self.decay else 1.0
        c_now = float(self.wd_c.item())
        out = []
        for r, s in enumerate(self.seeds):
            row = {"arm": arm, "seed": s, "task": self.task, "lam": self.lam, "scope": self.scope,
                   "c": float(c32(self.lam)).hex(), "wd_c_ok": int(c_now == c_exp),
                   **{f"wd_steps_{nm}": counts[i] for i, nm in enumerate(TENSORS)}}
            for l, z in ((1, z1), (2, z2)):
                zr = z[r]
                g = gtrain(zr)
                zc = zr.double().cpu()
                row[f"zline{l}"] = float((g == 0).double().mean())
                row[f"fline{l}"] = float((zc <= ZF).double().mean())
                row[f"p{l}"] = float((zc > 0).double().mean())
                row[f"zpair_median{l}"] = float(zc.flatten().median())
            for i in range(3):
                W = self.P[2 * i][r].detach().double().cpu()
                row[f"w{i + 1}_row_med"] = float(W.norm(dim=1).median())
                row[f"w{i + 1}_fro"] = float(W.norm())
                row[f"b{i + 1}_norm"] = float(self.P[2 * i + 1][r].detach().double().cpu().norm())
            out.append(row)
        return out


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------

def source_hashes() -> dict:
    files = ["src/adamw_dose2_1011.py", "src/adamw_dose2_1011_run.py", "src/cifar_interventions_0920.py",
             "src/rlcifar_mlp_battle_0918.py", "src/pmnist_0905.py", "src/pmnist_rlcifar_0907.py",
             "analysis/cifar_ledger_0920/replay.py", "analysis/adamw_dose2_1011/rule.py", SPEC]
    return {f: sha(ROOT / f) for f in files if (ROOT / f).exists()}


def git_state() -> dict:
    def g(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return {"git_hash": g("rev-parse", "HEAD"),
            "git_dirty": g("status", "--porcelain", "--", "src", "analysis", "specs")}


def identity(arm: str, seeds, X, cifar, epochs: int, tasks: int, prefix_sha: str, prefix_core: str) -> dict:
    return dict(run_id=EXPERIMENT, arm=arm, lam=ARM_LAMBDA[arm], scope=ARM_SCOPE[arm],
                c_hex=float(c32(ARM_LAMBDA[arm])).hex(), seeds=list(seeds), epochs=epochs, tasks=tasks,
                prereg_commit=PREREG_COMMIT, source_sha256=source_hashes(), environment=environment(),
                data_sha256=cifar.sha256, input_hash=tree_hash(X), prefix_file_sha256=prefix_sha,
                prefix_core_sha256=prefix_core)


# --------------------------------------------------------------------------
# one arm: tasks 2..tasks from a task-1 prefix
# --------------------------------------------------------------------------

def write_csv(path: Path, rows: list[dict]) -> None:
    import csv
    keys = list(dict.fromkeys(k for r in rows for k in r))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def run_arm(arm: str, out: Path, seeds=SEEDS, tasks: int = TASKS, epochs: int = EPOCHS, prefix=None,
            graph: bool = True, progress=None, admission: dict | None = None, save_states=SAVE_STATES,
            keep_units: bool = True, final: dict | None = None, epoch_limit: int | None = None) -> bool:
    """Run one arm from a task-1 prefix (default: cap_cifar_ee_0920's prefix/t01.pt) over tasks 2..tasks.

    `prefix` may be a state dict (checks) or None.  Writes per_task.csv, wd_task.csv, units_tXX.npz, tXX.pt and
    provenance_{start,end}.json under `out`; resumes from out/active.pt; returns False if stopped by out/STOP."""
    progress = progress or (lambda m: print(m, flush=True))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "done.json").exists():
        progress(f"{arm}: {out} is complete (done.json); nothing to do")
        return True
    lam, scope = ARM_LAMBDA[arm], ARM_SCOPE[arm]
    dev = setup()
    X, cifar = make_inputs(list(seeds), dev)
    if prefix is None:
        assert list(seeds) == list(SEEDS), "the archived prefix holds seeds 0-9"
        prefix_sha = sha(CI.artifact(CAP_PREFIX))
        assert prefix_sha == manifest_sha(CAP_PREFIX), "prefix file differs from the cap_cifar_ee_0920 manifest"
        anchor = load_pt(CAP_PREFIX)
    else:
        anchor = prefix
        prefix_sha = "in-memory"
    assert anchor["seeds"] == list(seeds) and anchor["task"] == 1 and not anchor["in_task"], "not a task-1 prefix"
    prefix_core = tree_hash(core_state(anchor))
    ident = identity(arm, seeds, X, cifar, epochs, tasks, prefix_sha, prefix_core)
    start = out / "provenance_start.json"
    if start.exists():
        old = json.loads(start.read_text())
        assert old["identity"] == ident, "resume identity mismatch"
    else:
        put(start, dict(identity=ident, **git_state(), admission=admission, pid=os.getpid(), argv=sys.argv,
                        started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        started_local=time.strftime("%Y-%m-%d %H:%M:%S %Z")))
    stop = out / "STOP"
    active = out / "active.pt"
    tt = time.time()
    if active.exists():
        saved = load_pt(active)
        assert saved["identity"] == ident
        e = WDEngine(list(seeds), X, state=saved["state"], lam=lam, scope=scope, graph=graph)
        rows, wdr = saved["rows"], saved["wd_rows"]
    else:
        e = WDEngine(list(seeds), X, state=anchor, lam=lam, scope=scope, graph=graph)
        rows, wdr = [], []
        put(out / "branch.json", dict(prefix_core_sha256=prefix_core,
                                      branch_core_sha256=tree_hash(core_state(e.state()))))
    first = e.task if e.in_task else e.task + 1
    for task in range(first, tasks + 1):
        if stop.exists():
            return False
        res = e.train_task(task, epochs, checkpoint=lambda st: save_pt(active, dict(identity=ident, state=st, rows=rows,
                                                                                     wd_rows=wdr)), stop=stop,
                           epoch_limit=epoch_limit if task == first else None)
        if res is None:
            return False
        rr, unit = res
        for r in rr:
            r.update(arm=arm, lam=lam)
        rows.extend(rr)
        wdr.extend(e.wd_rows(arm, epochs))
        write_csv(out / "per_task.csv", rows)
        write_csv(out / "wd_task.csv", wdr)
        if keep_units:
            save_npz(out / f"units_t{task:02d}.npz", unit)
        if task in save_states:
            save_pt(out / f"t{task:02d}.pt", e.state())
        save_pt(active, dict(identity=ident, state=e.state(), rows=rows, wd_rows=wdr))
        on = np.mean([r["online_acc"] for r in rr])
        progress(f"[{time.strftime('%T')}] {arm} task {task}/{tasks} online {on:.3f} "
                 f"wd_steps {int(e.wd_count[0])} finite {all(r['finite'] for r in rr)} "
                 f"elapsed {(time.time() - tt) / 60:.1f} min")
    if final is not None:
        final.update(state=e.state())
    put(out / "provenance_end.json", dict(identity=ident, finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                          elapsed_s=time.time() - tt,
                                          peak_cuda_allocated=torch.cuda.max_memory_allocated(),
                                          peak_cuda_reserved=torch.cuda.max_memory_reserved(),
                                          files={f.name: sha(f) for f in sorted(out.glob("*.csv"))}))
    put(out / "done.json", dict(complete=True, tasks=tasks, rows=len(rows)))
    if active.exists():
        active.unlink()
    return True


def train_prefix(seeds, epochs: int = EPOCHS, graph: bool = True, lam: float = 0.0):
    """Task 1 from the initial values with the host's Engine path (lambda = 0): S0 (a) and the S-ctrl prefix."""
    dev = setup()
    X, cifar = make_inputs(list(seeds), dev)
    e = WDEngine(list(seeds), X, lam=lam, graph=graph)
    rows, unit = e.train_task(1, epochs)
    st = e.state()
    st.pop("wd")                       # a branch point carries no decay record (the host's prefix format)
    return st, rows, unit
