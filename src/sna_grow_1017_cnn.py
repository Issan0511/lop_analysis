#!/usr/bin/env python3
"""sna_grow_1017, CNN part (spec specs/spec_sna_grow_1017.md §2, §2.1, §3 K-cnn-*).

    python src/sna_grow_1017_cnn.py checks                      # -> results/sna_grow_1017/cnn_checks.json
    python src/sna_grow_1017_cnn.py checks --mutant phi_host    # the suite on a planted bug (must fail)
    python src/sna_grow_1017_cnn.py run                         # SNA, SNA+grow1 x seeds 10-19, 30 x 400
    python src/sna_grow_1017_cnn.py run --seeds 10 --tasks 2 --epochs 20 --out results/_smoke/sg

The rule `grow` (spec §2): task 1 is the host SNA (alpha = clip(c / sqrt(V), lo, hi), V the EMA of
the batch variance, beta = 0.01, c = 0.6, lo 0.005, hi 3.0).  At the end of task 1, at every
adaptive site, V_max <- V (per run, per channel / unit); from then on, right after every EMA
update of V, V_max <- max(V_max, V), and alpha = clip(c / sqrt(V_max), lo, hi).  The EMA of V
itself goes on as in the host.

The engine (`src/sna_cnn_cause_1009.py`, called E here) is imported unchanged.  `install()`
points three of its module globals, in this process only, at this file: `BundleSnake` (the
activation bundle, replaced by the subclass `BundleSnakeGrow`), `parse_arm` (which learns the
option `+grow<k>`) and `EXPERIMENT` (the run id written to provenance.json).  `E.Bundle` and
`E.run` look these names up at call time, so the run loop, the CUDA graph, the checkpoints and
the resume are E's own.  `+grow<k>` sets E's `frz = k`, so E.run's existing end-of-task hook
(`if t in frz: B.act.freeze(frz[t])`) fires at task k; `BundleSnakeGrow.freeze` switches grow on
for those runs instead of freezing their alpha.

Post-hoc reruns from a checkpoint of this run (spec §4.1, the task-21 mechanism rerun) must use
`load_runs` / `load_run` below.  `analysis/sna_cnn_cause_1009/posthoc_lib.load_run` copies only
V / ada / fixA / cval: with install() it would silently rebuild a grow run with grow off and
V_max = 1 (SNA dynamics).  `load_runs` copies every key of `BundleSnakeGrow.KEYS` (a checkpoint
without V_max fails) and asserts, after loading, that grow is on exactly where the rule says and
that alpha = clip(c / sqrt(V_max)) from the checkpoint, bit for bit.

Known engine behaviour, not changed here (spec: the engine is imported unchanged): E.evaluate runs
eigvalsh on all runs' Gram matrices together, so one non-finite run would stop the whole bundle at
a task end; E.run then resumes from state.pt at that task boundary.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np                                    # noqa: E402
import torch                                          # noqa: E402

from src import pmnist_0905 as H                      # noqa: E402  (not touched)
from src import pmnist_rlcifar_0907 as RC             # noqa: E402  (not touched)
from src import rlcifar_cnn_0908 as CN                # noqa: E402  (not touched)
from src import sna_cnn_cause_1009 as E               # noqa: E402  (engine, not touched)

EXPERIMENT = "sna_grow_1017"
WIDTHS, IS_CONV = E.WIDTHS, E.IS_CONV
OUT_DEFAULT = ROOT / "results" / EXPERIMENT / "cnn"
CHECKS_JSON = ROOT / "results" / EXPERIMENT / "cnn_checks.json"
MUTANT_DIR = ROOT / "results" / EXPERIMENT / "cnn_checks_mutants"
ARMS_DEFAULT = ("SNA", "SNA+grow1")

# the engine's originals, captured before any install()
_ORIG = {"BundleSnake": E.BundleSnake, "parse_arm": E.parse_arm, "EXPERIMENT": E.EXPERIMENT}


# --------------------------------------------------------------------------
# arm names
# --------------------------------------------------------------------------

def parse_arm(name: str) -> dict:
    """E.parse_arm plus the option `+grow<k>` (k >= 1): frz = k and grow = True.  Every spec
    carries `grow` (False unless `+grow<k>` is given).  `+grow<k>` with `+frz<j>` is refused
    (both would use the one end-of-task hook)."""
    base, *opts = name.split("+")
    grow = [o for o in opts if o.startswith("grow")]
    rest = [o for o in opts if not o.startswith("grow")]
    if len(grow) > 1:
        raise SystemExit(f"two grow options in {name!r}")
    if grow and any(o.startswith("frz") for o in rest):
        raise SystemExit(f"+grow and +frz together in {name!r}")
    d = _ORIG["parse_arm"]("+".join([base, *rest]))
    d["grow"] = False
    if grow:
        k = grow[0][4:]
        if not k.isdigit() or int(k) < 1:
            raise SystemExit(f"bad grow option {grow[0]!r} in {name!r}")
        d["frz"] = int(k)
        d["grow"] = True
    return d


# --------------------------------------------------------------------------
# the bundled activation with the grow rule
# --------------------------------------------------------------------------

class BundleSnakeGrow(_ORIG["BundleSnake"]):
    """E.BundleSnake plus V_max and a per-(run, channel/unit) grow flag.

    alpha = where(ada, clip(c / sqrt(where(grow_on, V_max, V)), lo, hi), fixA).  Before grow is
    switched on, where(False, V_max, V) returns V itself, so alpha is the host's bit for bit.
    phi / dphi are the host's, which read alpha() (so training uses this alpha).
    V_max and grow_on are allocated once and only ever updated in place (a captured CUDA graph
    holds them).  They are in state()/load_state(), so E's capture (save / restore around the
    warm-up), full_state, run_state and resume carry them.
    """

    KEYS = ("V", "ada", "cval", "fixA", "Vmax", "grow_on")

    def __init__(self, specs: list[dict], device, lo=0.005, hi=3.0, beta=0.01):
        super().__init__(specs, device, lo=lo, hi=hi, beta=beta)
        self.grow_run = [bool(s.get("grow", False)) for s in specs]
        self.Vmax = [v.clone() for v in self.V]            # ones; read only where grow_on
        self.grow_on = [torch.zeros_like(a) for a in self.ada]

    def alpha(self, l: int) -> torch.Tensor:                           # (R, n)
        Vg = torch.where(self.grow_on[l], self.Vmax[l], self.V[l])
        ada = (self.cval[l] / Vg.sqrt()).clamp(self.lo, self.hi)
        return torch.where(self.ada[l], ada, self.fixA[l])

    @torch.no_grad()
    def update(self, zs) -> None:
        super().update(zs)                                  # the host EMA of V, unchanged
        for l in range(4):
            self.Vmax[l].copy_(torch.maximum(self.Vmax[l], self.V[l]))

    @torch.no_grad()
    def freeze(self, runs: list[int]) -> None:
        """Runs whose spec has grow: V_max <- V and grow on at their adaptive sites (alpha is
        not frozen and does not change at this moment).  Other runs: the host's freeze."""
        g = [r for r in runs if self.grow_run[r]]
        rest = [r for r in runs if not self.grow_run[r]]
        if rest:
            super().freeze(rest)
        for l in range(4):
            for r in g:
                self.Vmax[l][r] = self.V[l][r]
                self.grow_on[l][r] = self.ada[l][r]

    def state(self) -> dict:
        return {k: [t.clone() for t in getattr(self, k)] for k in self.KEYS}

    def load_state(self, st: dict) -> None:
        for k in self.KEYS:                                 # strict: a state without V_max fails
            for dst, src in zip(getattr(self, k), st[k]):
                dst.copy_(src)                              # in place: a captured graph holds these


# The class install() hands to the engine.  Only `checks --mutant <name>` ever changes it (to one
# of MUTANTS below), so that the whole check suite can be run on a planted bug and must fail.
GROW_CLS: type = BundleSnakeGrow


def install() -> None:
    """Point the engine's run loop at the grow bundle (this process only; no file is touched)."""
    E.BundleSnake = GROW_CLS
    E.parse_arm = parse_arm
    E.EXPERIMENT = EXPERIMENT


def uninstall() -> None:
    for k, v in _ORIG.items():
        setattr(E, k, v)


# --------------------------------------------------------------------------
# planted bugs (mutation controls; never used by `run`)
# --------------------------------------------------------------------------

def _host_alpha_of(act, l: int) -> torch.Tensor:
    """The host rule clip(c / sqrt(V)) (fixA at fixed sites), whatever V_max / grow_on say."""
    return _ORIG["BundleSnake"].alpha(act, l)


class _MutPhiHost(BundleSnakeGrow):
    """Bug: phi / dphi use the host alpha (from V), while alpha() -- read by the checks and by
    evaluate -- uses V_max.  The grow arm then trains as plain SNA but reports grow alphas."""

    def phi(self, z, l):
        a = self._shape(l, _host_alpha_of(self, l))
        return z + torch.sin(a * z) ** 2 * a.reciprocal()

    def dphi(self, z, l):
        return 1.0 + torch.sin(2.0 * self._shape(l, _host_alpha_of(self, l)) * z)


class _MutAlphaV(BundleSnakeGrow):
    """Bug: grow is a no-op everywhere (alpha from V; V_max and grow_on are kept but unused)."""

    def alpha(self, l):
        return _host_alpha_of(self, l)


class _MutFreezeRealloc(BundleSnakeGrow):
    """Bug: freeze() re-allocates V_max / grow_on before writing them, so a CUDA graph captured
    before the hook keeps reading and writing the old tensors (kept alive here, so the stale
    graph touches valid memory and the failure is a clean mismatch, not a crash)."""

    @torch.no_grad()
    def freeze(self, runs):
        self._stale = getattr(self, "_stale", []) + [(self.Vmax, self.grow_on)]
        self.Vmax = [v.clone() for v in self.Vmax]
        self.grow_on = [g.clone() for g in self.grow_on]
        super().freeze(runs)


MUTANTS = {"phi_host": _MutPhiHost, "alphaV": _MutAlphaV, "freeze_realloc": _MutFreezeRealloc}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

_CIFAR = None


def cifar10():
    global _CIFAR
    if _CIFAR is None:
        _CIFAR = RC.Cifar10()
    return _CIFAR


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head() -> dict:
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", "src"],
                           capture_output=True, text=True).stdout.strip()
    return {"git_hash": head, "git_dirty_src": bool(dirty), "git_dirty_src_lines": dirty.splitlines()}


def cmp_lists(a, b) -> tuple[bool, float]:
    """(bit-identical, max |a - b|) over two lists of tensors (bool tensors compared as 0/1)."""
    eq, mx = True, 0.0
    for x, y in zip(a, b, strict=True):
        x, y = torch.as_tensor(x).detach().cpu(), torch.as_tensor(y).detach().cpu()
        if x.shape != y.shape:
            return False, float("inf")
        eq &= bool((x == y).all())
        if x.numel():
            mx = max(mx, float((x.double() - y.double()).abs().max()))
    return eq, mx


def frz_map(B) -> dict[int, list[int]]:
    """E.run's end-of-task hook table (task -> runs)."""
    frz: dict[int, list[int]] = {}
    for r, sp in enumerate(B.specs):
        if sp["frz"] is not None:
            frz.setdefault(sp["frz"], []).append(r)
    return frz


def train_epochs(B, n: int) -> list[torch.Tensor]:
    accs = []
    for _ in range(n):
        B.run_epoch()
        accs.append(B.acc_ep.clone())
    return accs


def slot_state(st: dict, r: int) -> dict:
    """One run's tensors from a full_state() dict (P, m, v, and the act lists)."""
    return {"P": [p[r] for p in st["P"]], "m": [q[r] for q in st["m"]], "v": [q[r] for q in st["v"]],
            **{k: [t[r] for t in vs] for k, vs in st["act"].items()}}


def read_rows(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def empty_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def _bundle(slots, cifar, device, graph: bool, act_cls=None, epochs: int = 400):
    install()
    E.BundleSnake = act_cls or GROW_CLS
    try:
        return E.Bundle(slots, cifar, device, epochs=epochs, graph=graph)
    finally:
        uninstall()


def _all_tensors(B) -> list[torch.Tensor]:
    return [*B._state_tensors(), *[t for k in BundleSnakeGrow.KEYS for t in getattr(B.act, k)]]


def _core_tensors(B) -> list[torch.Tensor]:
    """P, m, v and the EMA V: the trajectory itself (not V_max / grow_on, which a mutation of
    the state would change by construction)."""
    return [*B.P, *B.m, *B.v, *B.act.V]


def _P_slot_differs(B, r0: int, r1: int) -> bool:
    return not cmp_lists([p[r0] for p in B.P], [p[r1] for p in B.P])[0]


def _grow_rule_alpha(act, l: int, r: int, src: str = "Vmax") -> torch.Tensor:
    """clip(c / sqrt(<src>)) at the adaptive sites of run r, fixA elsewhere (src = Vmax | V)."""
    Vs = getattr(act, src)[l][r]
    return torch.where(act.ada[l][r], (act.cval[l][r] / Vs.sqrt()).clamp(act.lo, act.hi), act.fixA[l][r])


# --------------------------------------------------------------------------
# checkpoint reload for post-hoc reruns (spec §4.1, mechanism rerun)
# --------------------------------------------------------------------------

def labels_at(seed: int, task: int) -> torch.Tensor:
    g = H.stream("rlcc_labels", seed)
    y = None
    for _ in range(task):
        y = CN.task_labels(g)
    return y


def ckpt_task(path) -> int:
    return int(Path(path).stem.split("_t")[-1])


def _grow_active(spec: dict, task: int) -> bool:
    return bool(spec.get("grow")) and spec["frz"] is not None and task >= spec["frz"]


def verify_loaded(B, sts: list[dict], task: int) -> dict:
    """After a reload: V_max / grow_on are the checkpoint's; grow is on exactly at the adaptive
    sites of runs whose grow hook has fired (task >= k for `+grow<k>`), off elsewhere; and every
    run's alpha is clip(c / sqrt(V_max)) (grow on) or clip(c / sqrt(V)) (otherwise) computed from
    the checkpoint's own tensors, bit for bit."""
    act, dev = B.act, B.act.V[0].device
    vmax_eq = grow_rule = alpha_eq = True
    grow_all_true = []
    for r, st in enumerate(sts):
        active = _grow_active(B.specs[r], task)
        a = st["act"]
        for l in range(4):
            ada = a["ada"][l].to(dev)
            vmax_eq &= bool((act.Vmax[l][r] == a["Vmax"][l].to(dev)).all())
            want = ada if active else torch.zeros_like(ada)
            grow_rule &= bool((act.grow_on[l][r] == want).all())
            Vsrc = (a["Vmax"] if active else a["V"])[l].to(dev)
            ref = torch.where(ada, (a["cval"][l].to(dev) / Vsrc.sqrt()).clamp(act.lo, act.hi),
                              a["fixA"][l].to(dev))
            alpha_eq &= bool((act.alpha(l)[r] == ref).all())
        if active:
            grow_all_true.append(all(bool(act.grow_on[l][r][act.ada[l][r]].all()) for l in range(4)))
    return {"Vmax_eq_ckpt": vmax_eq, "grow_on_eq_rule": grow_rule,
            "alpha_eq_ckpt_rule": alpha_eq, "grow_runs_grow_on_all_true_at_adaptive_sites": all(grow_all_true)}


def load_runs(paths, device="cuda", graph: bool = False, epochs: int = 400, cifar=None,
              keys=None, check: bool = True):
    """Rebuild a bundle (R = len(paths)) from per-run checkpoints of one task (E.run_state files of
    this run): parameters, Adam moments, step count and every activation key (V, ada, cval, fixA,
    V_max, grow_on; strict).  Y = the labels of the checkpoint's task; the label and batch streams
    are advanced to the end of that task (E's `epochs` per task), so `B.new_labels()` and
    `run_epoch()` continue the run's own streams.  The end-of-task hook is not run here (for a
    checkpoint at task >= k it has already fired).  check=True asserts verify_loaded().
    Continuing with the run's own slots in its own order (same R) reproduces the run bit for bit
    (K-cnn-resume (iii)); so does an R = 1 reload of an R = 1 run.  Another R is another kernel
    shape and is not bit-identical (the engine's R = 1 and R = 2 runs of one arm and seed already
    differ after task 1).
    `keys` is for the mutation control only (posthoc_lib's subset).  Returns (B, task, verify)."""
    device = torch.device(device)
    paths = [Path(p) for p in paths]
    tasks = {ckpt_task(p) for p in paths}
    if len(tasks) != 1:
        raise ValueError(f"checkpoints of different tasks: {sorted(tasks)}")
    task = tasks.pop()
    sts = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
    if len({st["tc"] for st in sts}) != 1:
        raise ValueError("checkpoints with different step counts")
    slots = [(st["arm"], st["seed"]) for st in sts]
    B = _bundle(slots, cifar or cifar10(), device, graph, epochs=epochs)
    keys = B.act.KEYS if keys is None else keys
    with torch.no_grad():
        for r, st in enumerate(sts):
            for dst, src in ((B.P, st["P"]), (B.m, st["m"]), (B.v, st["v"])):
                for q, s in zip(dst, src, strict=True):
                    q[r].copy_(s)
            for k in keys:                                  # KeyError if the file lacks one
                for dst, src in zip(getattr(B.act, k), st["act"][k], strict=True):
                    dst[r].copy_(src)
            B.Y[r].copy_(labels_at(st["seed"], task))
        B.tc.fill_(sts[0]["tc"])
    for s in B.useeds:
        for _ in range(task * epochs):
            torch.randperm(CN.N_IMAGES, generator=B.g_batch[s])
        for _ in range(task):
            CN.task_labels(B.g_lab[s])
    ver = verify_loaded(B, sts, task)
    if check and not all(ver.values()):
        raise AssertionError(f"reload of {[str(p) for p in paths]} breaks the grow state: {ver}")
    return B, task, ver


def load_run(path, device="cuda", graph: bool = False, epochs: int = 400, cifar=None):
    """One run (R = 1), the drop-in for posthoc_lib.load_run on this run's checkpoints."""
    B, task, _ = load_runs([path], device, graph, epochs, cifar)
    return B, task


# --------------------------------------------------------------------------
# checks (spec §3)
# --------------------------------------------------------------------------

def _run_engine(slots, out: Path, n_tasks: int, epochs: int, device, patched: bool, log_prefix: str,
                log=None):
    if out.exists():
        shutil.rmtree(out)
    if patched:
        install()
    else:
        uninstall()
    try:
        E.run(slots, out, n_tasks, epochs, device, graph=True, resume=True,
              log=log or (lambda m: print(f"  [{log_prefix}] {m}", flush=True)))
    finally:
        uninstall()
    return torch.load(out / "state.pt", map_location="cpu", weights_only=False)


def check_off(device, work: Path, epochs: int) -> dict:
    """K-cnn-off.  Task 1 through E.run: {SNA, SNA+grow1} x seed 10 with the grow bundle, and
    {SNA, SNA+frz1} x seed 10 with the untouched engine (same R, same kernels)."""
    slots_g = [("SNA", 10), ("SNA+grow1", 10)]
    slots_h = [("SNA", 10), ("SNA+frz1", 10)]
    t0 = time.time()
    st_h = _run_engine(slots_h, work / "off_host", 1, epochs, device, False, "off host")
    t1 = time.time()
    st_g = _run_engine(slots_g, work / "off_grow", 1, epochs, device, True, "off grow")
    t2 = time.time()
    sh, sg = st_h["state"], st_g["state"]
    g0, g1, h0, h1 = slot_state(sg, 0), slot_state(sg, 1), slot_state(sh, 0), slot_state(sh, 1)
    det: dict = {"epochs_task1": epochs, "seconds_host": t1 - t0, "seconds_grow": t2 - t1}
    errs = []
    ok = True
    for name, (a, b) in {"grow_SNA_vs_grow_SNA+grow1": (g0, g1),
                         "grow_SNA_vs_host_SNA": (g0, h0),
                         "host_SNA_vs_host_SNA+frz1": (h0, h1)}.items():
        sub = {}
        for k in ("P", "m", "v", "V"):
            eq, mx = cmp_lists(a[k], b[k])
            sub[k] = {"bit_identical": eq, "max_abs_diff": mx}
            ok &= eq
            errs.append(mx)
        det[name] = sub
    det["tc_grow"], det["tc_host"] = sg["tc"], sh["tc"]
    ok &= sg["tc"] == sh["tc"] == float(epochs * E.SPE)
    # every epoch's online accuracy, both slots, both engines
    cg = np.load(work / "off_grow" / "curves.npy")
    chh = np.load(work / "off_host" / "curves.npy")
    curves_eq = bool((cg[0] == cg[1]).all() and (cg[0] == chh[0]).all())
    det["curves_all_epochs_identical"] = curves_eq
    ok &= curves_eq
    # the task-1 per_task rows (evaluation before the hook) identical apart from the arm name
    rg, rh = read_rows(work / "off_grow" / "per_task.csv"), read_rows(work / "off_host" / "per_task.csv")
    strip = [{k: v for k, v in r.items() if k != "arm"} for r in rg + rh]
    rows_eq = all(s == strip[0] for s in strip)
    det["per_task_rows_identical_except_arm"] = rows_eq
    ok &= rows_eq
    # the hook at the end of task 1
    specs = [parse_arm(a) for a, s in slots_g]
    act = GROW_CLS(specs, device)
    act.load_state({k: [t.to(device) for t in vs] for k, vs in sg["act"].items()})
    al = [act.alpha(l) for l in range(4)]
    hook = {
        "grow_on_slot1_all_true": all(bool(act.grow_on[l][1].all()) for l in range(4)),
        "grow_on_slot0_all_false": not any(bool(act.grow_on[l][0].any()) for l in range(4)),
        "Vmax_slot1_eq_V_slot1": all(bool((act.Vmax[l][1] == act.V[l][1]).all()) for l in range(4)),
        "ada_slot1_still_true": all(bool(act.ada[l][1].all()) for l in range(4)),
        "alpha_slot1_eq_alpha_slot0_after_hook": all(bool((al[l][1] == al[l][0]).all()) for l in range(4)),
    }
    # the untouched engine's hook froze its slot 1 at the same alpha
    acth = _ORIG["BundleSnake"]([_ORIG["parse_arm"](a) for a, s in slots_h], device)
    acth.load_state({k: [t.to(device) for t in vs] for k, vs in sh["act"].items()})
    hook["host_frz_slot1_fixA_eq_grow_alpha"] = all(
        bool((acth.fixA[l][1] == al[l][1]).all()) and not bool(acth.ada[l][1].any()) for l in range(4))
    ck = torch.load(work / "off_grow" / "ckpt" / "SNA+grow1_seed10_t01.pt", map_location="cpu",
                    weights_only=False)
    hook["ckpt_run_state_has_Vmax_grow_on"] = ("Vmax" in ck["act"] and "grow_on" in ck["act"]
                                               and all(bool(t.all()) for t in ck["act"]["grow_on"]))
    det["hook_end_of_task1"] = hook
    ok &= all(hook.values())
    det["alpha_med_task1_end"] = {E.SITES[l]: float(al[l][0].median()) for l in range(4)}
    return {"name": "K-cnn-off", "pass": bool(ok), "max_err": max(errs), "detail": det}


def _synthetic_z(noise, fac, s: float):
    """z with per-(run, unit) batch variance about s * fac (fac = exp(U(-6, 2)) per unit, so that at
    the activation some units sit at the hi clip and late in the sequence some at the lo clip)."""
    out = []
    for l in range(4):
        out.append(noise[l] * (s * fac[l]).sqrt())
    return out


def _grow_synthetic(device) -> tuple[bool, float, dict]:
    """K-cnn-grow (i).  A synthetic V sequence (growth and shrinkage, into both clips) through the
    real update() / freeze() / alpha() of the grow bundle, beside the host BundleSnake fed the same z."""
    arms = ["SNA", "SNA+grow1", "S:c0.6-a3-c0.6-a0.6+grow1"]
    specs = [parse_arm(a) for a in arms]
    R = len(arms)
    act = GROW_CLS(specs, device)
    host = _ORIG["BundleSnake"](specs, device)
    g = torch.Generator().manual_seed(1017)
    Bz, Hs = 8, 4
    noise, fac = [], []
    for l, w in enumerate(WIDTHS):
        if IS_CONV[l]:
            x = torch.randn(Bz, R * w, Hs, Hs, generator=g, dtype=torch.float64)
            x = (x - x.mean((0, 2, 3), keepdim=True)) / x.std((0, 2, 3), unbiased=False, keepdim=True)
            f = torch.exp(torch.empty(1, R * w, 1, 1, dtype=torch.float64).uniform_(-6, 2, generator=g))
        else:
            x = torch.randn(R, Bz, w, generator=g, dtype=torch.float64)
            x = (x - x.mean(1, keepdim=True)) / x.std(1, unbiased=False, keepdim=True)
            f = torch.exp(torch.empty(R, 1, w, dtype=torch.float64).uniform_(-6, 2, generator=g))
        noise.append(x.float().to(device))
        fac.append(f.float().to(device))
    # log-variance knots (step, s); activation at T_ACT
    # task 1 is 600 updates so that the init V = 1 has decayed (0.99^600 = 0.0024) and the units
    # with small fac sit at the hi clip when grow switches on
    T_ACT = 600
    knots = [(0, 1.0), (300, 9.0), (600, 4.0),                     # task 1: up then down
             (800, 0.5), (1100, 30.0), (1400, 0.005), (1600, 0.005),  # shrink, grow, shrink to hi-clip
             (1800, 1e5), (2000, 1e5), (2300, 1.0)]                    # grow to lo-clip, shrink
    T_END = knots[-1][0]
    ks = np.array([k for k, _ in knots], dtype=float)
    ls = np.log([s for _, s in knots])
    sched = np.exp(np.interp(np.arange(1, T_END + 1), ks, ls))

    def alphas(a):
        return [a.alpha(l).clone() for l in range(4)]

    pre_ok = True
    V_hist = [[] for _ in range(4)]             # V after each update (float32), for the reference
    a_hist = []
    a_host_hist = []
    Vmax_hist = []
    V_eq_host = True
    act_snap = {}
    for t in range(1, T_END + 1):
        zs = _synthetic_z(noise, fac, float(sched[t - 1]))
        act.update(zs)
        host.update(zs)
        for l in range(4):
            V_eq_host &= bool((act.V[l] == host.V[l]).all())
            V_hist[l].append(act.V[l].clone())
        if t == T_ACT:
            before = alphas(act)
            pre_ok &= all(bool((before[l] == host.alpha(l)).all()) for l in range(4))
            act.freeze([1, 2])                   # E.run's hook: frz = 1 for both grow runs
            after = alphas(act)
            act_snap = {
                "alpha_unchanged_at_activation": all(bool((before[l] == after[l]).all()) for l in range(4)),
                "Vmax_eq_V_at_activation": all(bool((act.Vmax[l][r] == act.V[l][r]).all())
                                               for l in range(4) for r in (1, 2)),
                "grow_on_eq_ada_for_grow_runs": all(bool((act.grow_on[l][r] == act.ada[l][r]).all())
                                                    for l in range(4) for r in (1, 2)),
                "grow_on_run2_sites": [bool(act.grow_on[l][2].any()) for l in range(4)],
                "grow_on_run0_none": not any(bool(act.grow_on[l][0].any()) for l in range(4)),
                "ada_unchanged": all(bool(act.ada[l][r].all()) for l in range(4) for r in (0, 1))
                and [bool(act.ada[l][2].all()) for l in range(4)] == [True, False, True, False],
            }
        if t < T_ACT:
            pre_ok &= all(bool((act.alpha(l) == host.alpha(l)).all()) for l in range(4))
        if t >= T_ACT:
            a_hist.append(alphas(act))
            a_host_hist.append(alphas(host))
            Vmax_hist.append([v.clone() for v in act.Vmax])
    # ---- post-activation checks, run by run
    lo, hi = act.lo, act.hi
    mono_viol = 0
    rel32, rel64 = 0.0, 0.0
    fixed_ok = True
    run0_is_host = True
    host_increases = 0
    mut_fromstart_mismatch = 0
    clip_hi = clip_lo = held = total = 0
    for l in range(4):
        Vs = torch.stack(V_hist[l])                           # (T_END, R, n)
        Vact = Vs[T_ACT - 1:]                                 # from the activation step on
        ref_max = torch.cummax(Vact, 0).values                # independent running max (exact)
        ref_max_start = torch.cummax(Vs, 0).values[T_ACT - 1:]   # mutation: max from step 1
        c32 = act.cval[l]                                     # (R, 1) float32 0.6
        ref32 = (c32 / ref_max.sqrt()).clamp(lo, hi)
        ref64 = (c32.double() / ref_max.double().sqrt()).clamp(lo, hi)
        mut = (c32 / ref_max_start.sqrt()).clamp(lo, hi)
        A = torch.stack([a[l] for a in a_hist])               # (n_steps, R, n)
        Ah = torch.stack([a[l] for a in a_host_hist])
        Vm = torch.stack([v[l] for v in Vmax_hist])
        for r in (1, 2):
            ada = act.ada[l][r]
            if not bool(ada.any()):
                fixed_ok &= bool((A[:, r] == act.fixA[l][r]).all())
                continue
            a_r = A[:, r][:, ada]
            mono_viol += int((a_r[1:] > a_r[:-1]).sum())
            rel32 = max(rel32, float(((a_r - ref32[:, r][:, ada]).abs() / ref32[:, r][:, ada]).max()))
            rel64 = max(rel64, float(((a_r.double() - ref64[:, r][:, ada]).abs() / ref64[:, r][:, ada]).max()))
            mut_fromstart_mismatch += int((a_r != mut[:, r][:, ada]).sum())
            clip_hi += int((a_r == hi).sum()); clip_lo += int((a_r == lo).sum())
            held += int((Vm[:, r][:, ada] > Vact[:, r][:, ada]).sum()); total += a_r.numel()
        run0_is_host &= bool((A[:, 0] == Ah[:, 0]).all())
        host_increases += int((Ah[1:, 0] > Ah[:-1, 0]).sum())
    ok = (pre_ok and V_eq_host and all(v for k, v in act_snap.items() if k != "grow_on_run2_sites")
          and act_snap["grow_on_run2_sites"] == [True, False, True, False]
          and mono_viol == 0 and rel32 <= 1e-7 and fixed_ok and run0_is_host
          and host_increases > 0 and mut_fromstart_mismatch > 0 and clip_hi > 0 and clip_lo > 0)
    det = {
        "pass": bool(ok),
        "arms": arms, "steps": T_END, "activation_step": T_ACT, "knots_step_s": knots,
        "alpha_eq_host_before_activation": pre_ok,
        "V_ema_bit_identical_to_host_all_steps": V_eq_host,
        "activation": act_snap,
        "monotone_violations_grow_runs": mono_viol,
        "max_rel_err_vs_float32_ref": rel32,
        "max_rel_err_vs_float64_ref": rel64,
        "float64_note": "ref64 = clip(float64(c_f32)/sqrt(float64(maxV_f32))); the float32 alpha carries two "
                        "roundings (sqrt, divide), bound 2^-23 = 1.19e-7; reported, the pass uses the float32 ref",
        "fixed_sites_of_grow_run_unchanged": fixed_ok,
        "run0_SNA_alpha_eq_host_after_activation": run0_is_host,
        "control_host_alpha_increases_count": host_increases,
        "control_mutation_Vmax_from_step1_mismatch_count": mut_fromstart_mismatch,
        "grow_entries_total": total, "grow_entries_at_hi_clip": clip_hi, "grow_entries_at_lo_clip": clip_lo,
        "grow_entries_Vmax_gt_V": held,
    }
    return bool(ok), rel32, det


def _per_run(mask: torch.Tensor, l: int, R: int) -> torch.Tensor:
    """(R,) count of True entries per run of a site tensor (conv (B, R*C, H, W), fc (R, B, n))."""
    if IS_CONV[l]:
        return mask.reshape(mask.shape[0], R, -1).sum((0, 2))
    return mask.reshape(R, -1).sum(1)


def _recorder(base: type) -> type:
    """A subclass of `base` that, once start_reference(runs) has been called, checks on the
    training path, at every call, that the phi / dphi of `base` compute the grow rule:
    out == z + sin(a z)^2 / a and dphi == 1 + sin(2 a z), bit for bit, with
    a = clip(c / sqrt(M)) and M = the running max of V since the activation, kept here
    independently of V_max (other runs / sites: the host rule from V).  It also counts where the
    host rule would give a different output (the comparison's power) and checks alpha() and V_max
    against the same reference.  It never changes what `base` computes."""

    class Recorder(base):
        ref_max = None

        def start_reference(self, runs: list[int]) -> None:
            assert self.amp is None, "the reference phi is the plain Snake"
            R, dev = self.R, self.V[0].device
            self.ref_mask = [torch.zeros_like(a) for a in self.ada]
            for l in range(4):
                for r in runs:
                    self.ref_mask[l][r] = self.ada[l][r]
            self.ref_max = [v.clone() for v in self.V]      # the activation: M <- V
            z = lambda: torch.zeros(R, dtype=torch.long, device=dev)       # noqa: E731
            self.mis_phi, self.mis_dphi, self.mis_alpha, self.mis_vmax, self.diff_host = z(), z(), z(), z(), z()
            self.n_entries = z()
            self.n_phi = self.n_upd = 0

        def _ref_alpha(self, l):
            Vg = torch.where(self.ref_mask[l], self.ref_max[l], self.V[l])
            return torch.where(self.ada[l], (self.cval[l] / Vg.sqrt()).clamp(self.lo, self.hi), self.fixA[l])

        def phi(self, z, l):
            out = super().phi(z, l)
            if self.ref_max is not None:
                with torch.no_grad():
                    zz = z.detach()
                    ar = self._ref_alpha(l)
                    a = self._shape(l, ar)
                    ah = self._shape(l, _host_alpha_of(self, l))
                    ref = zz + torch.sin(a * zz) ** 2 * a.reciprocal()
                    ref_h = zz + torch.sin(ah * zz) ** 2 * ah.reciprocal()
                    self.mis_phi += _per_run(out.detach() != ref, l, self.R)
                    self.mis_dphi += _per_run(self.dphi(zz, l) != 1.0 + torch.sin(2.0 * a * zz), l, self.R)
                    self.diff_host += _per_run(ref != ref_h, l, self.R)
                    self.n_entries += _per_run(torch.ones_like(ref, dtype=torch.bool), l, self.R)
                    self.mis_alpha += (self.alpha(l) != ar).sum(1)
                    self.n_phi += 1
            return out

        @torch.no_grad()
        def update(self, zs) -> None:
            super().update(zs)
            if self.ref_max is not None:
                for l in range(4):
                    self.ref_max[l] = torch.maximum(self.ref_max[l], self.V[l])
                    self.mis_vmax += (self.ref_mask[l] & (self.Vmax[l] != self.ref_max[l])).sum(1)
                self.n_upd += 1

    Recorder.__name__ = f"Recorder[{base.__name__}]"
    return Recorder


def _grow_train_path(cls: type, device, cifar, st_task1: dict, epochs: int = 2) -> dict:
    """K-cnn-grow (ii), one class.  From the K-cnn-off state (end of task 1, hook done), task-2
    labels, `epochs` epochs eager with the recorder around `cls`."""
    slots = [("SNA", 10), ("SNA+grow1", 10)]
    B = _bundle(slots, cifar, device, graph=False, act_cls=_recorder(cls))
    B.load_full_state(st_task1["state"])
    B.new_labels()                                    # task-2 labels (streams are after task 1)
    P_eq_start = not _P_slot_differs(B, 0, 1)
    act = B.act
    act.start_reference([r for r, (a, s) in enumerate(slots) if "+grow" in a])
    grow_on_start = all(bool((act.grow_on[l] == act.ref_mask[l]).all()) for l in range(4))
    train_epochs(B, epochs)
    grow_on_end = all(bool((act.grow_on[l] == act.ref_mask[l]).all()) for l in range(4))
    n_steps = epochs * E.SPE
    c = {k: getattr(act, k).cpu().tolist() for k in ("mis_phi", "mis_dphi", "mis_alpha", "mis_vmax",
                                                      "diff_host", "n_entries")}
    asserts = {
        "P_slot0_eq_slot1_at_start": P_eq_start,
        "phi_calls_eq_4_per_step": act.n_phi == 4 * n_steps,
        "updates_eq_steps": act.n_upd == n_steps,
        "grow_on_eq_rule_start_and_end": grow_on_start and grow_on_end,
        "phi_out_eq_grow_rule_every_call": sum(c["mis_phi"]) == 0,
        "dphi_eq_grow_rule_every_call": sum(c["mis_dphi"]) == 0,
        "alpha_fn_eq_grow_rule_every_call": sum(c["mis_alpha"]) == 0,
        "Vmax_eq_running_max_every_update": sum(c["mis_vmax"]) == 0,
        "power_slot1_grow_rule_output_ne_host_rule_somewhere": c["diff_host"][1] > 0,
        "slot0_grow_rule_is_host_rule": c["diff_host"][0] == 0,
        "P_slot1_ne_slot0_after_training": _P_slot_differs(B, 0, 1),
    }
    out = {"class": cls.__name__, "epochs": epochs, "steps": n_steps, "pass": all(asserts.values()),
           "asserts": asserts, "counts_per_slot_[SNA,SNA+grow1]": c,
           "frac_slot1_phi_entries_where_host_rule_differs": c["diff_host"][1] / max(c["n_entries"][1], 1)}
    del B, act
    empty_cache()
    return out


def check_grow(device, cifar, st_task1: dict) -> dict:
    """K-cnn-grow.  (i) the synthetic V sequence through update() / freeze() / alpha();
    (ii) the training path: phi / dphi as used in training compute clip(c / sqrt(running max of V
    since the activation)) at every step, and the grow slot's weights leave the SNA slot's;
    with two planted bugs (phi from the host alpha; grow a no-op) that (ii) must catch."""
    ok_s, rel32, det_s = _grow_synthetic(device)
    tp = _grow_train_path(GROW_CLS, device, cifar, st_task1)
    controls = {}
    for name in ("phi_host", "alphaV"):
        m = _grow_train_path(MUTANTS[name], device, cifar, st_task1)
        controls[name] = {"control_detects": not m["pass"],
                          "failed_asserts": [k for k, v in m["asserts"].items() if not v],
                          "counts_per_slot_[SNA,SNA+grow1]": m["counts_per_slot_[SNA,SNA+grow1]"]}
    ok = ok_s and tp["pass"] and all(v["control_detects"] for v in controls.values())
    det = {"synthetic": det_s, "train_path": tp, "train_path_mutation_controls": controls}
    return {"name": "K-cnn-grow", "pass": bool(ok), "max_err": rel32, "detail": det}


def _no_vmax_in_state(base: type) -> type:
    """Mutation control for K-cnn-graph: V_max left out of state(), so E's capture does not put
    V_max back after its warm-up steps."""

    class NoVmaxInState(base):
        def state(self) -> dict:
            return {k: [t.clone() for t in getattr(self, k)] for k in self.KEYS if k != "Vmax"}

        def load_state(self, st: dict) -> None:
            for k in self.KEYS:
                if k in st:
                    for dst, src in zip(getattr(self, k), st[k]):
                        dst.copy_(src)

    return NoVmaxInState


def _capture_in_task1(cls: type, device, cifar, graph: bool, n1: int):
    """A fresh bundle: task 1 for n1 epochs (the graph, if any, is captured at its first epoch,
    before the hook), E.run's hook, then 2 epochs of task 2."""
    slots = [("SNA", 10), ("SNA+grow1", 10)]
    B = _bundle(slots, cifar, device, graph, act_cls=cls)
    B.new_labels()
    accs = train_epochs(B, n1)
    cg_before = B.cg
    B.act.freeze(frz_map(B)[1])
    B.new_labels()
    accs += train_epochs(B, 2)
    return B, accs, cg_before


def check_graph(device, cifar, st_task1: dict, n1: int) -> dict:
    """K-cnn-graph.  From the grow-active state at the end of task 1 (K-cnn-off's E.run, 400
    epochs), 2 epochs with the CUDA graph and 2 eager.  Variant A keeps the task-1 labels (V_max
    keeps being raised), variant B switches to the task-2 labels (V shrinks, V_max holds alpha).
    Variant C is the main run's situation: the graph is captured in task 1, before the hook, and
    must see the hook's in-place writes (fresh bundle, n1 epochs of task 1, hook, 2 of task 2)."""
    slots = [("SNA", 10), ("SNA+grow1", 10)]
    st = st_task1["state"]
    det, ok, errs = {"C_task1_epochs": n1}, True, []
    for variant in ("A_same_labels", "B_task2_labels", "mutation_A_no_Vmax_in_state"):
        cls = _no_vmax_in_state(GROW_CLS) if variant.startswith("mutation") else GROW_CLS
        res = {}
        for graph in (True, False):
            B = _bundle(slots, cifar, device, graph, act_cls=cls if graph else GROW_CLS)
            if variant.startswith(("A", "mutation")):
                B.new_labels()                 # fresh label stream -> task-1 labels
                B.load_full_state(st)          # generators now after task 1; Y stays task 1
            else:
                B.load_full_state(st)
                B.new_labels()                 # task-2 labels
            Vmax0 = [t.clone() for t in B.act.Vmax]
            accs = train_epochs(B, 2)
            res[graph] = (B, accs, Vmax0)
        Bg, ag, Vmax0 = res[True]
        Be, ae, _ = res[False]
        eq, mx = cmp_lists(_all_tensors(Bg), _all_tensors(Be))
        eq_acc, _ = cmp_lists(ag, ae)
        a1 = [Be.act.alpha(l)[1] for l in range(4)]
        sub = {"bit_identical_P_m_v_tc_acc_bad_V_Vmax_ada_cval_fixA_grow_on": eq, "max_abs_diff": mx,
               "epoch_acc_identical": eq_acc,
               "graph_used": Bg.cg is not None and Be.cg is None,
               "Vmax_entries_raised_in_2_epochs_slot1": int(sum(int((Be.act.Vmax[l][1] != Vmax0[l][1]).sum())
                                                               for l in range(4))),
               "entries_Vmax_gt_V_slot1_end": int(sum(int((Be.act.Vmax[l][1] > Be.act.V[l][1]).sum())
                                                      for l in range(4))),
               "entries_alpha_slot1_ne_host_rule_end": int(sum(int((a1[l] != _grow_rule_alpha(Be.act, l, 1, "V"))
                                                                   .sum()) for l in range(4))),
               "alpha_slot1_eq_grow_rule_end": all(bool((a1[l] == _grow_rule_alpha(Be.act, l, 1, "Vmax")).all())
                                                   for l in range(4)),
               "P_slot1_ne_slot0_graph": _P_slot_differs(Bg, 0, 1),
               "P_slot1_ne_slot0_eager": _P_slot_differs(Be, 0, 1),
               "n_entries_slot1": int(sum(WIDTHS))}
        if variant.startswith("mutation"):
            sub["expected"] = "a difference (shows the check sees a V_max missing from the capture's restore)"
            sub["control_detects"] = not eq
            ok &= not eq
        else:
            ok &= (eq and eq_acc and sub["graph_used"] and sub["entries_alpha_slot1_ne_host_rule_end"] > 0
                   and sub["alpha_slot1_eq_grow_rule_end"] and sub["P_slot1_ne_slot0_graph"]
                   and sub["P_slot1_ne_slot0_eager"])
            errs.append(mx)
        det[variant] = sub
        del res, Bg, Be
        empty_cache()
    for variant in ("C_capture_in_task1", "mutation_C_freeze_reallocates"):
        cls = MUTANTS["freeze_realloc"] if variant.startswith("mutation") else GROW_CLS
        Bg, ag, cg_before = _capture_in_task1(cls, device, cifar, True, n1)
        Be, ae, _ = _capture_in_task1(cls, device, cifar, False, n1)
        eq, mx = cmp_lists(_all_tensors(Bg), _all_tensors(Be))
        eq_acc, _ = cmp_lists(ag, ae)
        a1 = [Be.act.alpha(l)[1] for l in range(4)]
        sub = {"bit_identical_P_m_v_tc_acc_bad_V_Vmax_ada_cval_fixA_grow_on": eq, "max_abs_diff": mx,
               "epoch_acc_identical_task1_and_task2": eq_acc,
               "graph_captured_in_task1_and_kept": cg_before is not None and Bg.cg is cg_before and Be.cg is None,
               "grow_on_slot1_all_true_end": all(bool(Bg.act.grow_on[l][1].all()) for l in range(4)),
               "entries_alpha_slot1_ne_host_rule_end": int(sum(int((a1[l] != _grow_rule_alpha(Be.act, l, 1, "V"))
                                                                   .sum()) for l in range(4))),
               "P_slot1_ne_slot0_graph": _P_slot_differs(Bg, 0, 1),
               "P_slot1_ne_slot0_eager": _P_slot_differs(Be, 0, 1)}
        if variant.startswith("mutation"):
            sub["expected"] = "a difference (shows the check sees a graph that misses the hook's writes)"
            sub["control_detects"] = not eq
            ok &= not eq
        else:
            ok &= (eq and eq_acc and sub["graph_captured_in_task1_and_kept"] and sub["grow_on_slot1_all_true_end"]
                   and sub["entries_alpha_slot1_ne_host_rule_end"] > 0
                   and sub["P_slot1_ne_slot0_graph"] and sub["P_slot1_ne_slot0_eager"])
            errs.append(mx)
        det[variant] = sub
        del Bg, Be
        empty_cache()
    return {"name": "K-cnn-graph", "pass": bool(ok), "max_err": max(errs), "detail": det}


def _ckpt_paths(out: Path, slots, t: int) -> list[Path]:
    return [out / "ckpt" / f"{a.replace(':', '_')}_seed{s}_t{t:02d}.pt" for a, s in slots]


def _cmp_to_ckpts(B, sts: list[dict]) -> dict:
    """Each run of B against its checkpoint, bit for bit: everything (P, m, v and every act key)
    and the trajectory alone (P, m, v, V; a reset of V_max / grow_on differs there only if it
    changed the training); tc."""
    per = []
    for r, st in enumerate(sts):
        core_a = [*[p[r] for p in B.P], *[q[r] for q in B.m], *[q[r] for q in B.v], *[t[r] for t in B.act.V]]
        core_b = [*st["P"], *st["m"], *st["v"], *st["act"]["V"]]
        a = core_a + [t[r] for k in BundleSnakeGrow.KEYS if k != "V" for t in getattr(B.act, k)]
        b = core_b + [t.reshape(-1) if k == "cval" else t
                      for k in BundleSnakeGrow.KEYS if k != "V" for t in st["act"][k]]
        eq, mx = cmp_lists(a, b)
        eq_c, mx_c = cmp_lists(core_a, core_b)
        per.append({"arm": st["arm"], "bit_identical": eq, "max_abs_diff": mx,
                    "bit_identical_P_m_v_V": eq_c, "max_abs_diff_P_m_v_V": mx_c})
    return {"runs": per, "tc_equal": all(float(B.tc) == st["tc"] for st in sts)}


def _reload_part(device, cifar, outU: Path, slots, epochs_run: int) -> tuple[bool, dict]:
    """K-cnn-resume (iii): the per-run checkpoints (E.run_state) reload with V_max / grow_on.
    From E.run's t02 files: load_runs (R = 2, the run's slot order, graph as in E.run), task 3,
    against E.run's t03 files bit for bit; the R = 1 reload a post-hoc rerun would use (its
    invariants asserted, its continuation reported: another R is another kernel shape); and the
    posthoc_lib-style copy (V, ada, fixA, cval only), which must be caught."""
    p2, p3 = _ckpt_paths(outU, slots, 2), _ckpt_paths(outU, slots, 3)
    st3 = [torch.load(p, map_location="cpu", weights_only=False) for p in p3]
    det: dict = {"from": [p.name for p in p2], "against": [p.name for p in p3]}
    B, task, ver = load_runs(p2, device, graph=True, epochs=epochs_run, cifar=cifar)
    B.new_labels()
    train_epochs(B, epochs_run)
    c = _cmp_to_ckpts(B, st3)
    det["R2_load_runs"] = {"task": task, "verify_after_load": ver, "task3_vs_ckpt_t03": c}
    ok = all(ver.values()) and all(x["bit_identical"] for x in c["runs"]) and c["tc_equal"]
    del B
    empty_cache()
    B1, _, ver1 = load_runs([p2[1]], device, graph=True, epochs=epochs_run, cifar=cifar)
    B1.new_labels()
    train_epochs(B1, epochs_run)
    c1 = _cmp_to_ckpts(B1, [st3[1]])
    det["R1_load_run_grow"] = {"verify_after_load": ver1, "task3_vs_ckpt_t03_reported": c1}
    ok &= all(ver1.values())
    del B1
    empty_cache()
    Bc, _, verc = load_runs(p2, device, graph=True, epochs=epochs_run, cifar=cifar,
                            keys=("V", "ada", "fixA", "cval"), check=False)
    Bc.new_labels()
    train_epochs(Bc, epochs_run)
    cc = _cmp_to_ckpts(Bc, st3)
    detects = (not all(verc.values())) and not cc["runs"][1]["bit_identical_P_m_v_V"]
    det["mutation_posthoc_lib_keys"] = {"verify_after_load": verc, "task3_vs_ckpt_t03": cc,
                                        "control_detects": detects,
                                        "control_rule": "detects only if verify_after_load flags the load and "
                                                        "the grow run's task-3 P/m/v/V differ from t03"}
    ok &= detects
    del Bc
    empty_cache()
    return bool(ok), det


def check_resume(device, cifar, work: Path, epochs: int, k_save: int, epochs_run: int) -> dict:
    """K-cnn-resume.  (i) mid-task 2: an uninterrupted bundle (task 1, hook, task 2) is snapshot
    (full_state + the task-2 labels, which E's state does not hold because E resumes at task
    boundaries) to disk at epoch k_save of task 2 and keeps going; a fresh bundle loads the file
    and runs the rest.  The mutation control (V_max / grow_on reset after loading) must change the
    trajectory itself: P, m, v, V and the post-resume epoch accuracies.  (ii) E.run's own resume
    at a task boundary: a run killed after task 2's state.pt, restarted, against an uninterrupted
    E.run (3 tasks of `epochs_run` epochs).  (iii) the per-run checkpoints reload (_reload_part)."""
    slots = [("SNA", 10), ("SNA+grow1", 10)]
    det, ok, errs = {"epochs_per_task": epochs, "save_at_task2_epoch": k_save}, True, []
    U = _bundle(slots, cifar, device, True)
    U.new_labels()
    train_epochs(U, epochs)
    fz = frz_map(U)
    U.act.freeze(fz[1])
    U.new_labels()
    accU = train_epochs(U, k_save)
    snap = U.full_state()
    snap["Y"] = U.Y.cpu()
    f = work / "resume_mid_task2.pt"
    E.save_atomic(snap, f)
    det["state_act_keys"] = sorted(snap["act"].keys())
    det["state_has_Vmax_grow_on"] = "Vmax" in snap["act"] and "grow_on" in snap["act"]
    s1 = slot_state(snap, 1)
    det["at_save_slot1_entries_Vmax_gt_V"] = int(sum(int((a > b).sum()) for a, b in zip(s1["Vmax"], s1["V"])))
    det["at_save_slot1_P_ne_slot0"] = not cmp_lists(s1["P"], slot_state(snap, 0)["P"])[0]
    accU += train_epochs(U, epochs - k_save)
    endU = [t.detach().clone() for t in _all_tensors(U)]
    coreU = [t.detach().clone() for t in _core_tensors(U)]
    gensU = {s: (U.g_lab[s].get_state(), U.g_batch[s].get_state()) for s in U.useeds}
    del U
    empty_cache()
    for variant in ("resumed", "mutation_Vmax_grow_on_not_loaded"):
        J = _bundle(slots, cifar, device, True)
        st = torch.load(f, map_location="cpu", weights_only=False)
        J.load_full_state(st)
        J.Y.copy_(st["Y"])
        if variant.startswith("mutation"):
            with torch.no_grad():
                for l in range(4):
                    J.act.Vmax[l].fill_(1.0)
                    J.act.grow_on[l].zero_()
        accJ = train_epochs(J, epochs - k_save)
        eq, mx = cmp_lists(_all_tensors(J), endU)
        eq_core, mx_core = cmp_lists(_core_tensors(J), coreU)
        eq_acc, _ = cmp_lists(accJ, accU[k_save:])
        gens_eq = all(bool((J.g_lab[s].get_state() == gensU[s][0]).all())
                      and bool((J.g_batch[s].get_state() == gensU[s][1]).all()) for s in J.useeds)
        sub = {"bit_identical_P_m_v_tc_acc_bad_V_Vmax_ada_cval_fixA_grow_on": eq, "max_abs_diff": mx,
               "bit_identical_P_m_v_V": eq_core, "max_abs_diff_P_m_v_V": mx_core,
               "post_resume_epoch_acc_identical": eq_acc, "generators_identical": gens_eq}
        if variant.startswith("mutation"):
            sub["control_detects"] = (not eq_core) and (not eq_acc)
            sub["control_rule"] = "detects only if P/m/v/V and the post-resume epoch accuracies both change"
            ok &= sub["control_detects"]
        else:
            ok &= eq and eq_core and eq_acc and gens_eq
            errs.append(mx)
        det[variant] = sub
        del J
        empty_cache()
    ok &= det["state_has_Vmax_grow_on"] and det["at_save_slot1_entries_Vmax_gt_V"] > 0
    # (ii) E.run's resume at a task boundary
    outU, outI = work / "erun_uninterrupted", work / "erun_interrupted"

    class _Kill(Exception):
        pass

    def killer(m):
        print(f"  [erun interrupted] {m}", flush=True)
        if "task  2" in m:
            raise _Kill()

    erun_ok = False
    try:
        stU = _run_engine(slots, outU, 3, epochs_run, device, True, "erun uninterrupted")
        killed = False
        try:
            _run_engine(slots, outI, 3, epochs_run, device, True, "erun interrupted", log=killer)
        except _Kill:
            killed = True
        install()
        try:
            E.run(slots, outI, 3, epochs_run, device, graph=True, resume=True,
                  log=lambda m: print(f"  [erun resumed] {m}", flush=True))
        finally:
            uninstall()
        stI = torch.load(outI / "state.pt", map_location="cpu", weights_only=False)
        a, b = stI["state"], stU["state"]
        tens_a = [*a["P"], *a["m"], *a["v"], *[t for k in BundleSnakeGrow.KEYS for t in a["act"][k]], a["bad"]]
        tens_b = [*b["P"], *b["m"], *b["v"], *[t for k in BundleSnakeGrow.KEYS for t in b["act"][k]], b["bad"]]
        eq, mx = cmp_lists(tens_a, tens_b)
        rows_eq = read_rows(outI / "per_task.csv") == read_rows(outU / "per_task.csv")
        curves_eq = bool((np.load(outI / "curves.npy") == np.load(outU / "curves.npy")).all())
        ck_keys = all({"Vmax", "grow_on"} <= set(torch.load(outI / "ckpt" / f"SNA+grow1_seed10_t{t:02d}.pt",
                                                             map_location="cpu", weights_only=False)["act"])
                      for t in (1, 2, 3))
        det["erun_task_boundary"] = {"epochs_per_task": epochs_run, "killed_after_task2": killed,
                                     "resumed_at": stI["resumed"], "bit_identical_state": eq, "max_abs_diff": mx,
                                     "tc_equal": a["tc"] == b["tc"], "per_task_rows_identical": rows_eq,
                                     "curves_identical": curves_eq, "ckpt_t01_t02_t03_have_Vmax_grow_on": ck_keys}
        erun_ok = bool(killed and stI["resumed"] == [3] and eq and a["tc"] == b["tc"] and rows_eq
                       and curves_eq and ck_keys)
        errs.append(mx)
    except Exception:
        uninstall()
        det["erun_task_boundary"] = {"error": traceback.format_exc()}
    det["erun_task_boundary"]["pass"] = erun_ok
    ok &= erun_ok
    # (iii) the per-run checkpoints reload with V_max / grow_on (for the post-hoc mechanism rerun)
    reload_ok = False
    try:
        reload_ok, det["ckpt_reload"] = _reload_part(device, cifar, outU, slots, epochs_run)
    except Exception:
        uninstall()
        det["ckpt_reload"] = {"error": traceback.format_exc()}
    det["ckpt_reload"]["pass"] = reload_ok
    ok &= reload_ok
    return {"name": "K-cnn-resume", "pass": bool(ok), "max_err": max(errs) if errs else None, "detail": det}


def run_checks(args) -> None:
    global GROW_CLS
    if args.mutant:
        GROW_CLS = MUTANTS[args.mutant]
        print(f"MUTANT SUITE: the grow class is {GROW_CLS.__name__} (this suite must fail)", flush=True)
    json_path = Path(args.json or (MUTANT_DIR / f"{args.mutant}.json" if args.mutant else CHECKS_JSON))
    json_path.parent.mkdir(parents=True, exist_ok=True)
    device = H.setup("cuda")
    work = ROOT / "results" / EXPERIMENT / ("_cnn_checks_work" + (f"_mut_{args.mutant}" if args.mutant else ""))
    work.mkdir(parents=True, exist_ok=True)
    cifar = cifar10()
    only = set(args.only.split(",")) if args.only else None
    out = {"spec": "specs/spec_sna_grow_1017.md §3", "engine": "src/sna_cnn_cause_1009.py (unchanged)",
           "engine_sha256": sha256(ROOT / "src" / "sna_cnn_cause_1009.py"),
           "this_file_sha256": sha256(Path(__file__)), **git_head(), "torch": torch.__version__,
           "gpu": torch.cuda.get_device_name(), "grow_class": GROW_CLS.__name__,
           "mutant": args.mutant, "time": time.strftime("%F %T"), "settings": {
               "K-cnn-off": f"full task 1 ({args.epochs_off} epochs) through E.run, R = 2",
               "K-cnn-grow": "(i) synthetic, 2300 updates, activation at update 600; (ii) training path: "
                             "2 epochs eager of task 2 from the K-cnn-off state, every phi / dphi call checked "
                             "against the grow rule; planted bugs phi_host and alphaV must fail (ii)",
               "K-cnn-graph": "A/B: from the K-cnn-off grow state (end of task 1), 2 epochs graph vs eager; "
                              f"C: graph captured in task 1 ({args.epochs_capture1} epochs), hook, 2 epochs "
                              "of task 2, graph vs eager; mutation controls for A and C",
               "K-cnn-resume": f"(i) {args.epochs_resume} epochs per task, snapshot at task-2 epoch "
                               f"{args.k_save}; (ii) E.run 3 tasks x {args.epochs_run} epochs, killed after "
                               "task 2; (iii) per-run t02 checkpoints reloaded (load_runs), task 3 vs t03"},
           "checks": []}
    oom = 0

    def guarded(fn, *a):
        nonlocal oom
        for attempt in range(2):
            try:
                return fn(*a)
            except torch.cuda.OutOfMemoryError:
                oom += 1
                empty_cache()
                print(f"CUDA OOM in {fn.__name__} (attempt {attempt + 1})", flush=True)
                if oom >= 2:
                    raise SystemExit("CUDA OOM twice: stopping")
        raise SystemExit("unreachable")

    def record(res):
        print(f"{res['name']}: pass={res['pass']} max_err={res['max_err']}", flush=True)
        print(json.dumps(res["detail"], indent=1, default=str), flush=True)
        if "error" in res:
            print(res["error"], flush=True)
        out["checks"].append(res)
        out["all_pass"] = all(c["pass"] for c in out["checks"])
        json_path.write_text(json.dumps(out, indent=1, default=str))

    def run_one(name, fn, *a):
        """An exception inside a check is that check failing (recorded), not the suite crashing."""
        t0 = time.time()
        try:
            res = guarded(fn, *a)
        except SystemExit:
            raise
        except Exception:
            uninstall()
            res = {"name": name, "pass": False, "max_err": None, "detail": {}, "error": traceback.format_exc()}
        res["seconds"] = time.time() - t0
        record(res)
        empty_cache()
        return res

    st_task1 = None
    if only is None or only & {"off", "grow", "graph"}:
        res = run_one("K-cnn-off", check_off, device, work, args.epochs_off)
        if (work / "off_grow" / "state.pt").exists():
            st_task1 = torch.load(work / "off_grow" / "state.pt", map_location="cpu", weights_only=False)
    if only is None or "grow" in only:
        run_one("K-cnn-grow", check_grow, device, cifar, st_task1)
    if only is None or "graph" in only:
        run_one("K-cnn-graph", check_graph, device, cifar, st_task1, args.epochs_capture1)
    if only is None or "resume" in only:
        run_one("K-cnn-resume", check_resume, device, cifar, work, args.epochs_resume, args.k_save,
                args.epochs_run)
    print("all_pass", out.get("all_pass"), "->", json_path, flush=True)
    if not args.keep_work:
        shutil.rmtree(work, ignore_errors=True)


def run_main(args) -> None:
    device = H.setup(args.device)
    arms = args.arms.split(",")
    for a in arms:
        parse_arm(a)
    seeds = E.parse_seeds(args.seeds)
    slots = [(a, s) for a in arms for s in seeds]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    assert GROW_CLS is BundleSnakeGrow
    prov = {"run_id": EXPERIMENT, "spec": "specs/spec_sna_grow_1017.md §2.1",
            "engine": "src/sna_cnn_cause_1009.py run() with BundleSnakeGrow / parse_arm from this file "
                      "(install(): E.BundleSnake, E.parse_arm, E.EXPERIMENT patched in process)",
            "engine_sha256": sha256(ROOT / "src" / "sna_cnn_cause_1009.py"),
            "this_file_sha256": sha256(Path(__file__)), **git_head(),
            "arms": {a: parse_arm(a) for a in arms}, "seeds": seeds, "tasks": args.tasks,
            "epochs": args.epochs, "lr": args.lr, "graph": not args.no_graph, "torch": torch.__version__}
    pf = out / "provenance_sna_grow.json"
    starts = json.loads(pf.read_text()).get("starts", []) if pf.exists() else []
    prov["starts"] = starts + [{"time": time.strftime("%F %T"), **git_head(),
                                "this_file_sha256": prov["this_file_sha256"]}]
    pf.write_text(json.dumps(prov, indent=1, default=str))
    install()
    try:
        E.run(slots, out, args.tasks, args.epochs, device, lr=args.lr, graph=not args.no_graph)
    finally:
        uninstall()
    prov["finished"] = time.strftime("%F %T")
    pf.write_text(json.dumps(prov, indent=1, default=str))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    r = sub.add_parser("run")
    r.add_argument("--arms", default=",".join(ARMS_DEFAULT))
    r.add_argument("--seeds", default="10-19")
    r.add_argument("--tasks", type=int, default=30)
    r.add_argument("--epochs", type=int, default=400)
    r.add_argument("--lr", type=float, default=1e-3)
    r.add_argument("--out", default=str(OUT_DEFAULT))
    r.add_argument("--no-graph", action="store_true")
    r.add_argument("--device", default="cuda")
    c = sub.add_parser("checks")
    c.add_argument("--only", default=None, help="comma list of off,grow,graph,resume")
    c.add_argument("--epochs-off", type=int, default=400)
    c.add_argument("--epochs-resume", type=int, default=400)
    c.add_argument("--k-save", type=int, default=173)
    c.add_argument("--epochs-run", type=int, default=50)
    c.add_argument("--epochs-capture1", type=int, default=20)
    c.add_argument("--keep-work", action="store_true")
    c.add_argument("--mutant", default=None, choices=sorted(MUTANTS),
                   help="run the suite with a planted bug in place of BundleSnakeGrow (must fail)")
    c.add_argument("--json", default=None, help=f"default {CHECKS_JSON} ({MUTANT_DIR}/<mutant>.json with --mutant)")
    args = ap.parse_args()
    if args.mode == "run":
        run_main(args)
    else:
        run_checks(args)


if __name__ == "__main__":
    main()
