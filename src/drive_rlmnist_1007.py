#!/usr/bin/env python3
"""drive_rlmnist_1007: A2 (cross-task second-layer Adam drive) on the Random-Label MNIST ELU->ELU ref arm.

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m src.drive_rlmnist_1007 --mode production --seed 0 \
        --out results/drive_rlmnist_1007/run/s0 --checks results/drive_rlmnist_1007/checks.json --production-go

specs/spec_drive_rlmnist_1007.md (registered at PREREG_COMMIT).  The host is src/mucap_el_run_0916.py's
run_one("ref", act2_name="ELU1", ledger=False) -- the l2cap_ee_0917 ref arm.  Run.task()/Run.epoch() are
that update loop written out (same draws, same arithmetic, same diagnostics; the ref arm has no caps, no
bias fix and no ledger), so the trajectory is the ref shard's bit for bit (checks A2-host).

The observer only reads.  Before the Adam update it copies the second layer, its first moment and the
batch's native h, z2, logits, W3 and training derivative to float64; after the update it measures with
A2's measure (analysis/drive_cifar_c_0920/numerics.py, imported unchanged) on the fixed 1200 images and
reduces each epoch to per-unit sums and maxima (spec 4.5).  It draws no random numbers and writes to no
training tensor.  The full-image features after update n are reused as the features before update n+1
(no parameter changes in between in the ref arm; checks A2-self-total).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import socket
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
from src import pmnist_0905 as H                 # host; not touched
from src import pmnist_rlmnist_0906 as RL        # subset, labels; not touched
from src import shell_l2_rlmnist_0913 as SH      # state_sha256; not touched
from src import elu_growth_0909 as EG            # ELU1; not touched
from src import mucap_el_0916 as MU              # mu_basis, caps (task-1 info only); not touched
from src import l2cap_ee_0917 as W2C             # row_norms (task-1 info only); not touched
from src import mucap_el_run_0916 as EL          # forward2, unit_arrays, adam_arrays, DENSE; not touched
from analysis.drive_rlmnist_1007 import numerics as N

RUN = "drive_rlmnist_1007"
SPEC = "specs/spec_drive_rlmnist_1007.md"
PREREG_COMMIT = "1c7b2927bac8cd839315dfacb0b3105ba5fc8b34"   # specs/spec_drive_rlmnist_1007.md, pushed before any scientific seed ran
MAIN_SEEDS = (0, 1, 2, 3, 4)
CAL_SEEDS = (5, 6, 7, 8, 9)
CHECK_SEEDS = (100, 101)
TASKS = 10                                       # 2-5 main window, 6-10 REPORT_ONLY, 1 state formation
EPOCHS = EL.EPOCHS                               # 80
SPE = EL.STEPS_PER_EPOCH                         # 75
BATCH = EL.BATCH                                 # 16
N_IMG = EL.N_IMAGES                              # 1200
LR, BETA1, BETA2, EPS = 1e-3, 0.9, 0.999, 1e-8   # the host's Adam (run_one's lr argument and locals)
ABORT_TASKS = frozenset(range(1, 6))             # spec 4.3: a numerical failure here stops the seed
AUDIT = frozenset({(2, 1), (2, 80), (5, 1), (5, 80)})   # spec 8: fixed audit epochs (task, epoch)
RUN_SOURCES = ("src/drive_rlmnist_1007.py", "src/mucap_el_run_0916.py", "src/mucap_el_0916.py",
               "src/l2cap_ee_0917.py", "src/pmnist_0905.py", "src/pmnist_rlmnist_0906.py",
               "src/shell_l2_rlmnist_0913.py", "src/elu_growth_0909.py", "src/boundary_gradient_0908.py",
               "analysis/drive_cifar_c_0920/numerics.py", "analysis/drive_rlmnist_1007/numerics.py", SPEC)


class CheckFailed(RuntimeError):
    pass


# --------------------------------------------------------------------------
# io helpers
# --------------------------------------------------------------------------

def sha(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def run_sources() -> dict:
    return {p: sha(ROOT / p) for p in RUN_SOURCES}


def _bytes(t: torch.Tensor) -> bytes:
    return t.detach().cpu().contiguous().numpy().tobytes()


def tsha(t: torch.Tensor) -> str:
    return hashlib.sha256(_bytes(t)).hexdigest()


def _clean(x):
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    return x


def put(path, data) -> None:
    """Atomic JSON (floats as repr, so they round-trip exactly; NaN allowed only where the host has it)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(_clean(data), ensure_ascii=False, indent=1) + "\n")
    tmp.replace(path)


def save_npz(path, data) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **data)
    tmp.replace(path)


def save_pt(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


# --------------------------------------------------------------------------
# the observer (read-only)
# --------------------------------------------------------------------------

class Observer:
    """A2's per-update measurement, reduced per epoch.  capture: set of (task, update) whose complete
    measure inputs/outputs are kept in .captured (checks only)."""

    def __init__(self, run, capture=None):
        self.r = run
        self.xbar = run.x.double().mean(0)
        self.cache = None
        self.capture = capture or set()
        self.captured = {}
        self.fail_total = {}
        self.first_failure = None
        self.epoch_start = None
        self.order = None

    def begin_task(self, t):
        m = self.r.adam[0]
        self.conf = torch.zeros(1, 100, 101, dtype=torch.float64)
        self.hist = N.augmented(m[2], m[3])[None]
        E = self.r.epochs
        self.sum = torch.zeros(E, 100, len(N.KEYS), dtype=torch.float64)
        self.max = torch.zeros_like(self.sum)
        self.l1 = torch.zeros(E, 100, len(N.L1_KEYS), dtype=torch.float64)
        self.aux = torch.zeros(E, 100, len(N.AUX_KEYS), dtype=torch.float64)

    def pre(self, out, old_b, new_b, t, s):
        r = self.r
        P, m = r.params, r.adam[0]
        self.before = N.augmented(P[2], P[3])
        self.W1b, self.b1b = P[0].detach().double(), P[1].detach().double()
        self.mprev = N.augmented(m[2], m[3])
        if self.cache is None:
            self.cache = N.full_features(P, r.x, r.act1, r.act2, EL.forward2)
        self.f0 = self.cache
        gate = N.train_gate(r.act2, out[2])
        self.dec = N.decompose(out[1], out[2], out[4], P[4], old_b, new_b, gate)
        if (t, s) in self.capture:
            m_, v_, tc_ = r.adam
            j = s % SPE
            self.raw = dict(h=out[1].detach().clone(), z=out[2].detach().clone(), logits=out[4].detach().clone(),
                            old=old_b.clone(), new=new_b.clone(), gate=gate.clone(),
                            ids=self.order[j * BATCH:(j + 1) * BATCH].clone(),
                            P=[q.detach().clone() for q in P], m=[q.clone() for q in m_],
                            v=[q.clone() for q in v_], tc=tc_[0])

    def post(self, grads, c1, c2, t, s):
        r = self.r
        P, (m, v, tc) = r.params, r.adam
        f1 = N.full_features(P, r.x, r.act1, r.act2, EL.forward2)
        inv1 = torch.tensor(1.0 / c1, dtype=torch.float64)
        inv2 = torch.tensor(1.0 / c2, dtype=torch.float64)
        args = (self.before[None], N.augmented(P[2], P[3])[None], self.mprev[None], N.augmented(m[2], m[3])[None],
                N.augmented(v[2], v[3])[None], N.augmented(grads[2], grads[3])[None], self.f0, f1, self.dec,
                inv1, inv2, self.conf, self.hist)
        d, cm, hm = N.measure(*args)
        self.conf, self.hist = cm, hm
        self.cache = f1
        ep = s // SPE
        a = N.pack(d)[0]
        self.sum[ep] += a
        self.max[ep] = torch.maximum(self.max[ep], a.abs())
        S1 = N.layer1_S(self.W1b, self.b1b, P[0], P[1], self.xbar)
        self.l1[ep] += N.pack_l1(S1)
        self.aux[ep, :, 0] += N.bracket(d, inv1)[0]
        if (t, s) in self.capture:
            self.captured[(t, s)] = dict(args=args, out=d, raw=self.raw,
                                         grads=[g.detach().clone() for g in grads],
                                         P_after=[q.detach().clone() for q in P],
                                         m_after=[q.clone() for q in m], v_after=[q.clone() for q in v], tc=tc[0])
        nfail = int(d["fail"].sum()) + int(d["nonfinite"].sum())
        if nfail:
            self.fail_total[t] = self.fail_total.get(t, 0) + nfail
            if self.first_failure is None:
                self.first_failure = dict(task=t, update=s, args=args, out=d, epoch_start=self.epoch_start,
                                          order=self.order)
                if r.out is not None:
                    save_pt(Path(r.out) / "first_failure.pt", self.first_failure)
            if t in ABORT_TASKS:
                raise CheckFailed(f"numerical check failed at task {t} update {s} (seed {r.seed})")

    def state(self):
        return dict(conf=self.conf.clone(), hist=self.hist.clone(),
                    cache=tuple(c.clone() for c in self.cache) if self.cache is not None else None)

    def restore(self, st):
        self.conf, self.hist = st["conf"].clone(), st["hist"].clone()
        self.cache = tuple(c.clone() for c in st["cache"]) if st["cache"] is not None else None


# --------------------------------------------------------------------------
# the host loop (src/mucap_el_run_0916.py run_one, ref arm, act2 = ELU1, ledger off)
# --------------------------------------------------------------------------

class Run:
    def __init__(self, seed, mnist, device, epochs=EPOCHS, observe=True, out=None, capture=None):
        self.t_start = time.time()
        self.seed, self.device, self.epochs, self.out = seed, device, epochs, out
        self.act1, self.act2 = EG.ELU(1.0), EL.ACT2["ELU1"]
        self.params = H.init_params(seed, device)    # host init: bit-identical per seed
        self.adam = ([torch.zeros_like(q) for q in self.params],
                     [torch.zeros_like(q) for q in self.params], [0])
        self.idx = RL.subset_idx(seed).to(device)
        self.x = mnist.train_x[self.idx]              # the task's inputs, fixed forever
        self.xt = mnist.test_x.to(device)
        self.e1, self.e1_64, self.mu1 = MU.mu_basis(self.x)
        if self.e1 is None:
            raise RuntimeError("mu = 0 on this subset")
        self.g_lab, self.g_batch = H.stream("rl_labels", seed), H.stream("rl_batch", seed)
        self.h_lab, self.h_batch = hashlib.sha256(), hashlib.sha256()
        self.spt = SPE * epochs
        self.q_cap = self.v_cap = None
        self.rows, self.diag = [], {"task": [], "step": []}
        self.info = {"init_sha256": hashlib.sha256(b"".join(_bytes(q) for q in self.params)).hexdigest(),
                     "subset_idx_sha256": hashlib.sha256(_bytes(self.idx)).hexdigest(),
                     "mu1_norm": self.mu1, "cos_mu1_to_uniform": float(self.e1_64.sum() / np.sqrt(self.e1_64.numel()))}
        self.diverged = {"diverged": False, "task": None, "step": None, "seed": seed, "arm": "ref"}
        self.y = self.oldY = None
        self.task_end_sha = {}
        self.task_seconds = {}
        self.obs = Observer(self, capture) if observe else None

    # ---- state for audits and fixtures (everything the next epoch depends on) ----
    def state(self):
        m, v, tc = self.adam
        st = dict(params=[q.detach().clone() for q in self.params], m=[q.clone() for q in m],
                  v=[q.clone() for q in v], tc=tc[0], g_lab=self.g_lab.get_state().clone(),
                  g_batch=self.g_batch.get_state().clone(), y=self.y.clone(), oldY=self.oldY.clone(),
                  acc_sum=self.acc_sum.clone(), bad_step=self.bad_step.clone())
        if self.obs is not None:
            st["obs"] = self.obs.state()
        return st

    def restore(self, st):
        with torch.no_grad():
            for q, w in zip(self.params, st["params"]):
                q.copy_(w)
        self.adam = ([q.clone() for q in st["m"]], [q.clone() for q in st["v"]], [st["tc"]])
        self.g_lab.set_state(st["g_lab"].clone())
        self.g_batch.set_state(st["g_batch"].clone())
        self.y, self.oldY = st["y"].clone(), st["oldY"].clone()
        self.acc_sum, self.bad_step = st["acc_sum"].clone(), st["bad_step"].clone()
        if self.obs is not None:
            self.obs.restore(st["obs"])

    def snap(self, t, s):
        u = EL.unit_arrays(self.params, self.x, self.act1, self.act2, self.e1_64) | EL.adam_arrays(self.params, self.adam)
        if t in EL.TEST_TASKS and s == self.spt:
            ut = EL.unit_arrays(self.params, self.xt, self.act1, self.act2, self.e1_64)
            u |= {f"test_{k}": v for k, v in ut.items()}
        self.diag["task"].append(t)
        self.diag["step"].append(s)
        for k, v in u.items():
            self.diag.setdefault(k, []).append(v)

    def epoch(self, t, ep, dense):
        dev = self.device
        if self.obs is not None:
            self.obs.epoch_start = self.state()
        order = torch.randperm(N_IMG, generator=self.g_batch).to(dev)
        self.h_batch.update(_bytes(order))
        xs, ys = self.x[order], self.y[order]
        olds = self.oldY[order]                       # spec 3: o = the previous task's actual label
        if self.obs is not None:
            self.obs.order = order
        for j in range(SPE):
            s = ep * SPE + j
            xb, yb = xs[j * BATCH:(j + 1) * BATCH], ys[j * BATCH:(j + 1) * BATCH]
            out = EL.forward2(self.params, xb, self.act1, self.act2)
            loss = F.cross_entropy(out[4], yb)
            # pre-update accuracy: the argmax of the very forward pass the loss came from
            self.acc_sum += (out[4].detach().argmax(1) == yb).float().mean()
            grads = torch.autograd.grad(loss, self.params)
            with torch.no_grad():
                bad = ~torch.isfinite(loss)
                self.bad_step = torch.where((self.bad_step < 0) & bad, torch.tensor(s, device=dev), self.bad_step)
                if self.obs is not None:
                    self.obs.pre(out, olds[j * BATCH:(j + 1) * BATCH], yb, t, s)
                m, v, tc = self.adam
                tc[0] += 1
                c1, c2 = 1 - BETA1 ** tc[0], 1 - BETA2 ** tc[0]
                for p, gr, mi, vi in zip(self.params, grads, m, v):
                    mi.mul_(BETA1).add_(gr, alpha=1 - BETA1)
                    vi.mul_(BETA2).addcmul_(gr, gr, value=1 - BETA2)
                    p -= LR * (mi / c1) / ((vi / c2).sqrt() + EPS)
                if self.obs is not None:
                    self.obs.post(grads, c1, c2, t, s)
            if s + 1 in dense:
                self.snap(t, s + 1)
        return order

    def task(self, t):
        t0 = time.time()
        dev = self.device
        y = RL.task_labels(self.g_lab).to(dev)        # new labelling, same images
        self.h_lab.update(_bytes(y))
        self.oldY = y.clone() if t == 1 else self.y   # task 1 has no old task: o = n, d = 0
        self.y = y
        self.acc_sum = torch.zeros((), device=dev)
        self.bad_step = torch.full((), -1, dtype=torch.long, device=dev)
        proj = {"rows_par": 0, "rows_perp": 0, "rem_par": 0.0, "rem_perp": 0.0,
                "rows_w2": 0, "rem_w2": 0.0, "rows_bfix": 0}
        dense = set(EL.DENSE) if 2 <= t <= 10 else {0}
        if 0 in dense:
            self.snap(t, 0)
        if self.obs is not None:
            self.obs.begin_task(t)
        chain = hashlib.sha256()
        for ep in range(self.epochs):
            audit = self.obs is not None and self.out is not None and (t, ep + 1) in AUDIT
            start = self.state() if audit else None
            order = self.epoch(t, ep, dense)
            chain.update(_bytes(order))
            if audit:
                save_pt(Path(self.out) / "audit" / f"t{t:02d}_e{ep + 1:03d}.pt",
                        dict(seed=self.seed, task=t, epoch=ep + 1, start=start, order=order, end=self.state(),
                             sum=self.obs.sum[ep].clone(), max=self.obs.max[ep].clone(),
                             l1=self.obs.l1[ep].clone(), aux=self.obs.aux[ep].clone()))
        bs = int(self.bad_step)
        if bs >= 0 or not all(torch.isfinite(p).all() for p in self.params):
            self.diverged.update(diverged=True, task=t, step=(t - 1) * self.spt + max(bs, 0))
            self.rows.append({"arm": "ref", "seed": self.seed, "task": t, "online_acc": float("nan")})
            return False                               # drop, never rescue
        if t == 1:
            self.info["task1_end_state_sha256"] = SH.state_sha256(self.params, self.adam, self.act1)
            self.q_cap = MU.parallel_cap(self.params[0], self.e1)
            self.v_cap = MU.perp_cap(self.params[0], self.e1)
            self.info["zero_q_cap_rows"] = int((self.q_cap == 0).sum())
            self.info["zero_v_cap_rows"] = int((self.v_cap == 0).sum())
            self.info["zero_r2_rows"] = int((W2C.row_norms(self.params[2]) == 0).sum())
        if self.spt not in dense:
            self.snap(t, self.spt)
        with torch.no_grad():
            memo = float((EL.forward2(self.params, self.x, self.act1, self.act2)[4].argmax(1) == self.y).float().mean())
        u = EL.unit_arrays(self.params, self.x, self.act1, self.act2, self.e1_64)
        self.rows.append({"arm": "ref", "seed": self.seed, "task": t, "online_acc": float(self.acc_sum) / self.spt,
                          "memo_acc": memo, "cap_on": 0, **proj,
                          "major_frac": float(torch.bincount(self.y, minlength=10).max()) / N_IMG,
                          **{f"med_{k}": float(np.nanmedian(v)) for k, v in u.items() if v.size == 100}})
        self.task_end_sha[t] = SH.state_sha256(self.params, self.adam, self.act1)
        self.task_seconds[t] = time.time() - t0
        if self.obs is not None and self.out is not None:
            o = self.obs
            ff = o.first_failure
            save_npz(Path(self.out) / f"t{t:02d}.npz", dict(
                sum=o.sum.numpy(), max=o.max.numpy(), l1=o.l1.numpy(), aux=o.aux.numpy(),
                keys=np.array(N.KEYS), l1_keys=np.array(N.L1_KEYS), aux_keys=np.array(N.AUX_KEYS),
                seed=self.seed, task=t, epochs=self.epochs, steps_per_epoch=SPE,
                label_sha256=tsha(self.y), old_label_sha256=tsha(self.oldY), order_chain=chain.hexdigest(),
                fail_count=o.fail_total.get(t, 0),
                first_failure=np.array([ff["task"], ff["update"]] if ff is not None else [-1, -1])))
        return True

    def run(self, n_tasks):
        for t in range(1, n_tasks + 1):
            if not self.task(t):
                break
        arrays = {k: np.asarray(v, dtype=np.float64) for k, v in self.diag.items()}
        arrays["q_cap"] = self.q_cap[:, 0].double().numpy() if self.q_cap is not None else np.full(100, np.nan)
        arrays["v_cap"] = self.v_cap[:, 0].double().numpy() if self.v_cap is not None else np.full(100, np.nan)
        self.info |= {"labels_sha256": self.h_lab.hexdigest(), "batch_sha256": self.h_batch.hexdigest(),
                      "final_state_sha256": SH.state_sha256(self.params, self.adam, self.act1),
                      "tasks_completed": len([r for r in self.rows if r["online_acc"] == r["online_acc"]]),
                      "wall_clock_s": time.time() - self.t_start, "divergence": self.diverged}
        self.arrays = arrays
        return self.rows, arrays, self.info


def replay_epoch(start, seed, t, ep, mnist, device, epochs=EPOCHS):
    """Recompute one saved epoch (audit): restore the full start state, run the same loop (no
    diagnostics), return the end state and the epoch's reductions."""
    r = Run(seed, mnist, device, epochs=epochs, observe=True)
    r.restore(start)
    r.obs.begin_task(t)
    r.obs.restore(start["obs"])
    order = r.epoch(t, ep - 1, dense=set())
    return dict(order=order, end=r.state(), sum=r.obs.sum[ep - 1].clone(), max=r.obs.max[ep - 1].clone(),
                l1=r.obs.l1[ep - 1].clone(), aux=r.obs.aux[ep - 1].clone())


# --------------------------------------------------------------------------
# one seed, written to disk
# --------------------------------------------------------------------------

def role(seed):
    return "main" if seed in MAIN_SEEDS else "calibration" if seed in CAL_SEEDS else "check"


def identity(seed, n_tasks, epochs, mode, x, mnist):
    return dict(run_id=RUN, mode=mode, seed=seed, role=role(seed), tasks=n_tasks, epochs=epochs,
                prereg_commit=PREREG_COMMIT, source_sha256=run_sources(), x_sha256=tsha(x),
                subset_sha256=tsha(RL.subset_idx(seed)), data_sha256=mnist.sha256,
                torch=torch.__version__, threads=torch.get_num_threads(), flush_denormal=True)


def run_seed(seed, out, n_tasks=TASKS, epochs=EPOCHS, mode="check", capture=None):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"{out} is not empty: move a stopped seed aside, never overwrite")
    out.mkdir(parents=True, exist_ok=True)
    device = H.setup("cpu")
    mnist = H.Mnist(device)
    t0 = time.time()
    r = Run(seed, mnist, device, epochs=epochs, observe=True, out=out, capture=capture)
    ident = identity(seed, n_tasks, epochs, mode, r.x, mnist)
    put(out / "input_manifest.json", ident)
    put(out / "provenance_start.json", dict(
        identity=ident, git_hash=git("rev-parse", "HEAD"), dirty=git("status", "--porcelain", "--", "src", "analysis", "specs"),
        utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), pid=os.getpid(), hostname=socket.gethostname(),
        platform=platform.platform(), python=sys.version.split()[0], cpu=torch.backends.cpu.get_cpu_capability()))
    try:
        rows, arrays, info = r.run(n_tasks)
    except CheckFailed as exc:
        put(out / "CHECK_FAILED.json", dict(reason=str(exc), task=r.obs.first_failure["task"],
                                            update=r.obs.first_failure["update"]))
        raise
    put(out / "host_rows.json", rows)
    save_npz(out / "units.npz", arrays)
    put(out / "info.json", info)
    put(out / "task_end_sha256.json", {str(k): v for k, v in r.task_end_sha.items()})
    put(out / "cost.json", dict(seconds=time.time() - t0, task_seconds={str(k): v for k, v in r.task_seconds.items()},
                                full_images=N_IMG, every_update=True, updates_per_task=SPE * epochs,
                                peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                                fail_total={str(k): v for k, v in r.obs.fail_total.items()}))
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name not in ("complete.json", "provenance_end.json"))
    put(out / "complete.json", dict(identity=ident, tasks_completed=info["tasks_completed"],
                                    files=[dict(path=str(p.relative_to(out)), sha256=sha(p)) for p in files]))
    put(out / "provenance_end.json", dict(status="COMPLETE", utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                          seconds=time.time() - t0))
    return info, r


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["check", "production"], default="check")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", type=int, default=TASKS)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--checks")
    ap.add_argument("--production-go", action="store_true")
    a = ap.parse_args(argv)
    torch.set_num_threads(1)
    torch.set_flush_denormal(True)
    if a.mode == "production":
        assert a.production_go and a.checks, "production needs --production-go and a validated checks.json"
        assert a.seed in MAIN_SEEDS + CAL_SEEDS and a.tasks == TASKS and a.epochs == EPOCHS
        from analysis.drive_rlmnist_1007.checks import validate_checks
        validate_checks(json.loads(Path(a.checks).read_text()))
    else:
        assert a.seed not in MAIN_SEEDS + CAL_SEEDS, "check mode never touches the scientific seeds"
    info, _ = run_seed(a.seed, a.out, a.tasks, a.epochs, a.mode)
    print(f"seed {a.seed}: {info['tasks_completed']} tasks, {info['wall_clock_s']:.1f}s; no scientific summary opened",
          flush=True)


if __name__ == "__main__":
    main()
