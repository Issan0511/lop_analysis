#!/usr/bin/env python3
"""The moving-field transplant: does the RL ELU->ELU net lose the ability to learn because its second
layer's response is lost (a state), or because that loss keeps going (continuing transport)?

    OMP_NUM_THREADS=1 python3 src/respdyn_ee_0917.py --seed 0 --out results/respdyn_ee_0917/runs/s0

specs/spec_respdyn_ee_0917.md (registered at PREREG_COMMIT).  Design: vault
可塑性喪失/spec/動く場の移植_応答低下からLoPへ_設計案_0917.md.

The box, the natural prefix, the branch states, the anchored forward and the per-unit diagnostics are
resp_ee_0917's, imported and not touched (RE.run_prefix, RE.make_forward, RE.build_shift,
RE.unit_point, RE.preflight, RE.task_row, RE.dstar).  resp_ee fixed the shift field d at the branch
point.  Here d may move during the continuation:

    dyn       d(x, s) = z2^(shadow)(x, s) - z2^(branch)(x).  The shadow is the source branch's own
              natural continuation N<src> (its parameters, Adam moments and label/order streams),
              stepped in lockstep with the arm; z2^(shadow)(x, s) is its full-batch preactivation after
              s updates of the continuation, computed by the same op that stored the branch fields.
              At s = 0 the shadow is the source state, so d(0) is resp_ee's fixed field bit for bit
              (recorded per arm; check S-t0) and the shadow reproduces N<src> (check S2, recorded).
    ramp      d = -(RAMP[0] + (RAMP[1] - RAMP[0]) g / G) for every unit and image, g = update index over
              the G = 12000 continuation updates (10 -> 30).  d(0) is S2u10r's field.
    noanchor  a2 = phi(z2 + d) with resp_ee's fixed field: the forward value moves to the source's too
              (the branch-point logits change, so the logit identity is not required of this arm).

The field tensor is refreshed in place before every update, so the anchored forward is RE.make_forward's
own code.  Every arm -- resp_ee's 26 and the 6 new ones -- runs through Stepper below, which is
RE.train_task's update written out one step at a time (check S1b: bit-equal for all 26 resp_ee arms),
with a trajectory probe on the 1200 images every PROBE_EVERY updates: the unit-mean effective number of
images carrying the second layer's training derivative (neffT2, resp_ee's definition), its exact-zero
share, the mean trained-with argument z2 + d and its two parts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import resource
import socket
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import pmnist_0905 as H                 # host; not touched
from src import pmnist_rlmnist_0906 as RL        # subset, labels; not touched
from src import shell_l2_rlmnist_0913 as SH      # state_sha256, file_sha256, git_dirty; not touched
from src import mucap_el_run_0916 as EL          # forward2; not touched
from src import resp_ee_0917 as RE               # the box, prefix, anchored forward; not touched

EXPERIMENT = "respdyn_ee_0917"
SPEC = "specs/spec_respdyn_ee_0917.md"
PREREG_COMMIT = "7f15cad5bc7456250998e4db009ace2055c8adc6"                               # set to the registration commit before the run
ACT = RE.ACT
LR, BETA1, BETA2, EPS = RE.LR, RE.BETA1, RE.BETA2, RE.EPS
N_IMG, BATCH, SPE = RE.N_IMG, RE.BATCH, RE.SPE
EPOCHS = RE.EPOCHS                               # 80 -> 6000 updates per task
CONT = RE.CONT                                   # 2 continuation tasks per arm
PREFIX_T = RE.PREFIX_T                           # 22
WEIGHTS = RE.WEIGHTS
RAMP = (10.0, 30.0)                              # the uniform ramp over all CONT tasks
PROBE_EVERY = 750                                # trajectory probe at s = 0, 750, ..., 6000 of each task
RESP_EE_RUNS = Path(__file__).resolve().parents[1] / "results" / "resp_ee_0917" / "runs"


def _arm(branch, kind=None, src=None, delta=None, reset=False, layer=2):
    return {"branch": branch, "layer": layer if kind is not None else None, "kind": kind, "src": src,
            "delta": delta, "reset": reset}


NEW_ARMS: dict[str, dict] = {
    "S2dyn_10r": _arm(2, "dyn", src=10, reset=True),         # primary: the t2 net follows N10's field
    "R2dyn_10": _arm(10, "dyn", src=2),                      # the t10 net follows N2's field
    "S2ramp_r": _arm(2, "ramp", reset=True),                 # uniform, deepening 10 -> 30
    "S2dyn_5r": _arm(2, "dyn", src=5, reset=True),
    "S2dyn_20r": _arm(2, "dyn", src=20, reset=True),
    "R2_10_noanchor": _arm(10, "noanchor", src=2),           # R2_10 without the fixed anchor
}
FIXED_TWIN = {"S2dyn_10r": "S2_10r", "R2dyn_10": "R2_10", "S2dyn_5r": "S2_5r", "S2dyn_20r": "S2_20r",
              "S2ramp_r": "S2u10r", "R2_10_noanchor": "R2_10"}
# New arms first, then resp_ee's arms in resp_ee's order (its natural continuations last: an aliasing
# bug in any earlier arm shows up as a hash mismatch there).
ARMS: dict[str, dict] = {**NEW_ARMS, **RE.ARMS}
ARANGE = torch.arange(N_IMG)


def gen_from(state: torch.Tensor) -> torch.Generator:
    return RE.gen_from(state)


# --------------------------------------------------------------------------
# one update at a time: RE.train_task's loop body
# --------------------------------------------------------------------------

class Stepper:
    """RE.train_task written as one call per update, so two nets can advance in lockstep.
    The arithmetic, the order of the draws and the slicing are the host's (check S1b)."""

    def __init__(self, params, adam, x, g_batch, fwd):
        self.params, self.adam, self.x, self.g_batch, self.fwd = params, adam, x, g_batch, fwd
        self.hook = None                         # checks only: hook(s) just before update s's forward

    def begin_task(self, y, spt, probe_mb=False):
        self.y, self.spt, self.probe_mb = y, spt, probe_mb
        self.pts = set(RE.eps_points(spt))
        self.ce = torch.empty(spt)
        self.acc = torch.empty(spt)
        self.stp = torch.empty(spt, len(WEIGHTS))
        self.epsf = {}
        self.mb_equal = None

    def step(self, s):
        params, x = self.params, self.x
        if self.hook is not None:
            self.hook(s)
        if s % SPE == 0:
            self.order = torch.randperm(N_IMG, generator=self.g_batch).to(x.device)
            self.xs, self.ys = x[self.order], self.y[self.order]
        j = s % SPE
        xb, yb = self.xs[j * BATCH:(j + 1) * BATCH], self.ys[j * BATCH:(j + 1) * BATCH]
        if self.fwd is None:
            out = EL.forward2(params, xb, ACT, ACT)
        else:
            out = self.fwd(params, xb, self.order[j * BATCH:(j + 1) * BATCH])
            if self.probe_mb and s == 0:
                with torch.no_grad():
                    self.mb_equal = bool(torch.equal(EL.forward2(params, xb, ACT, ACT)[4], out[4].detach()))
        loss = F.cross_entropy(out[4], yb)
        self.ce[s] = loss.detach()
        self.acc[s] = (out[4].detach().argmax(1) == yb).float().mean()
        grads = torch.autograd.grad(loss, params)
        m, v, tc = self.adam
        with torch.no_grad():
            tc[0] += 1
            c1, c2 = 1 - BETA1 ** tc[0], 1 - BETA2 ** tc[0]
            for k, (p, gr, mi, vi) in enumerate(zip(params, grads, m, v)):
                mi.mul_(BETA1).add_(gr, alpha=1 - BETA1)
                vi.mul_(BETA2).addcmul_(gr, gr, value=1 - BETA2)
                upd = LR * (mi / c1) / ((vi / c2).sqrt() + EPS)
                p -= upd
                if k % 2 == 0:
                    self.stp[s, k // 2] = upd.square().mean().sqrt()
            if s + 1 in self.pts:
                self.epsf[s + 1] = [float(((v[k] / c2).sqrt() < EPS).double().mean()) for k in WEIGHTS]

    def row(self):
        return RE.task_row(self.ce, self.acc, self.stp, self.epsf, self.y, self.spt)


# --------------------------------------------------------------------------
# moving fields
# --------------------------------------------------------------------------

def z2_full(params, x):
    """EL.forward2's first three lines: the op RE.run_prefix stored the branch fields with."""
    W1, b1, W2, b2 = params[:4]
    z1 = x @ W1.T + b1
    a1 = ACT.phi(z1)
    return a1 @ W2.T + b2


class DynField:
    """The shadow: N<src> itself, advanced one update after each arm update.  refresh() writes
    float32(float64(z2_shadow) - float64(z2_branch)) into the arm's field tensor -- RE.build_shift's
    arithmetic on the shadow's current full-batch field."""

    def __init__(self, arm, cks, x, d, freeze=False):
        src, br = cks[arm["src"]], cks[arm["branch"]]
        self.params = [p.detach().clone().requires_grad_(True) for p in src["params"]]
        self.adam = ([q.clone() for q in src["m"]], [q.clone() for q in src["v"]], [src["tc"]])
        self.g_lab, self.g_batch = gen_from(src["g_lab"]), gen_from(src["g_batch"])
        self.zbr = br["z2"].double()
        self.x, self.d, self.freeze = x, d, freeze
        self.src_task = src["task"]
        self.stepper = Stepper(self.params, self.adam, x, self.g_batch, None)

    def begin_task(self, spt):
        self.stepper.begin_task(RL.task_labels(self.g_lab), spt)

    def refresh(self, g):
        with torch.no_grad():
            z2 = z2_full(self.params, self.x)
            self.d.copy_((z2.double() - self.zbr).float())

    def step(self, s):
        if not self.freeze:
            self.stepper.step(s)

    def end_task(self) -> dict:
        r = self.stepper.row() if not self.freeze else {"online_acc": float("nan")}
        return {"shadow_online_acc": r["online_acc"],
                "shadow_state_sha256": SH.state_sha256(self.params, self.adam, ACT)}


class RampField:
    def __init__(self, d, total, lo, hi):
        self.d, self.total, self.lo, self.hi = d, total, lo, hi

    def begin_task(self, spt):
        pass

    def refresh(self, g):
        self.d.fill_(-(self.lo + (self.hi - self.lo) * g / self.total))

    def step(self, s):
        pass

    def end_task(self) -> dict:
        return {}


def make_forward_noanchor(sh: dict):
    """a2 = phi(z2 + d): value and derivative both at the shifted argument (no fixed anchor)."""
    d2 = sh[2]

    def fwd(params, xb, ob):
        W1, b1, W2, b2, W3, b3 = params
        z1 = xb @ W1.T + b1
        a1 = ACT.phi(z1)
        z2 = a1 @ W2.T + b2
        a2 = ACT.phi(z2 + d2[ob])
        return z1, a1, z2, a2, a2 @ W3.T + b3

    return fwd


def build(arm: dict, cks: dict, x: torch.Tensor, spt: int, freeze_shadow: bool = False):
    """-> (sh, fwd, mover).  sh[2] is the tensor the forward reads; a mover rewrites it in place."""
    kind = arm["kind"]
    if kind in (None, "field", "uniform"):
        sh = RE.build_shift(arm, cks)
        return sh, RE.make_forward(sh), None
    if kind in ("dyn", "noanchor"):
        sh = RE.build_shift({**arm, "kind": "field"}, cks)
    elif kind == "ramp":
        sh = RE.build_shift({**arm, "kind": "uniform", "delta": RAMP[0]}, cks)
    else:
        raise ValueError(kind)
    if kind == "noanchor":
        return sh, make_forward_noanchor(sh), None
    fwd = RE.make_forward(sh)
    if kind == "dyn":
        return sh, fwd, DynField(arm, cks, x, sh[2], freeze=freeze_shadow)
    return sh, fwd, RampField(sh[2], spt * CONT, RAMP[0], RAMP[1])


def fixed_field(arm: dict, cks: dict) -> torch.Tensor:
    """The field the arm starts from, built the way resp_ee built its fixed arms."""
    if arm["kind"] == "ramp":
        return RE.build_shift({**arm, "kind": "uniform", "delta": RAMP[0]}, cks)[2]
    return RE.build_shift({**arm, "kind": "field"}, cks)[2]


# --------------------------------------------------------------------------
# the trajectory probe (all 1200 images)
# --------------------------------------------------------------------------

@torch.no_grad()
def probe(params, x, fwd, d) -> dict:
    """neffT2: per unit (sum g)^2 / (N sum g^2) of the float32 training derivative g = phi'_train(z2 + d)
    (0 for a unit whose g is 0 on every image), averaged over units -- RE.unit_point's neffT_l2."""
    _, _, z2, _, _ = RE.forward(params, x, ARANGE, fwd)
    z64 = z2.double()
    ze32 = z2 if d is None else z2 + d
    gt = RE.dphi_train(ze32).double()
    n = gt.shape[0]
    s1 = gt.sum(0)
    neff = torch.where(s1 > 0, s1 * s1 / (n * gt.square().sum(0)).clamp(min=1e-300), torch.zeros_like(s1))
    dm = 0.0 if d is None else float(d.double().mean())
    return {"neffT2": float(neff.mean()), "zero2": float((gt == 0).double().mean()),
            "gtr2": float(gt.mean()), "zarm2": float(z64.mean()), "dmean2": dm,
            "effbar2": float(z64.mean()) + dm}


# --------------------------------------------------------------------------
# one arm
# --------------------------------------------------------------------------

def _resp_ee_rows(seed: int, name: str) -> dict:
    f = RESP_EE_RUNS / f"s{seed}" / "arms.csv"
    if not f.exists():
        return {}
    out = {}
    with f.open() as fh:
        for r in csv.DictReader(fh):
            if r["arm"] == name:
                out[int(r["k"])] = r
    return out


def run_arm(name: str, arm: dict, cks: dict, x: torch.Tensor, prefix_rows: list, seed: int,
            epochs: int = EPOCHS, units: dict | None = None, traj: list | None = None,
            progress: bool = False, freeze_shadow: bool = False, hook=None) -> list[dict]:
    t0 = time.time()
    ck = cks[arm["branch"]]
    params = [p.detach().clone().requires_grad_(True) for p in ck["params"]]
    if arm["reset"]:
        adam = ([torch.zeros_like(q) for q in params], [torch.zeros_like(q) for q in params], [0])
    else:
        adam = ([q.clone() for q in ck["m"]], [q.clone() for q in ck["v"]], [ck["tc"]])
    g_lab, g_batch = gen_from(ck["g_lab"]), gen_from(ck["g_batch"])
    spt = SPE * epochs
    sh, fwd, mover = build(arm, cks, x, spt, freeze_shadow)
    d = None if sh is None else sh.get(2)            # the probe reads layer 2 only
    by_task = {r["task"]: r for r in prefix_rows}
    rec = _resp_ee_rows(seed, name) if name in RE.ARMS else {}
    with torch.no_grad():
        nat = EL.forward2(params, x, ACT, ACT)[4]
    st = Stepper(params, adam, x, g_batch, fwd)
    rows = []
    for k in range(1, CONT + 1):
        t_task = time.time()
        y = RL.task_labels(g_lab)
        g0 = (k - 1) * spt
        head = {}
        if mover is not None:
            mover.begin_task(spt)
            mover.refresh(g0)
        if k == 1:
            u0, lg0 = RE.unit_point(params, x, fwd, sh)
            head = {"logits_equal_full": bool(torch.equal(lg0, nat)),
                    "logit_maxdiff_full": float((lg0.double() - nat.double()).abs().max()),
                    **RE.unit_summary(u0, "start"),
                    **RE.preflight(params, fwd, x, y, g_batch.get_state(), epochs)}
            if arm["kind"] in ("dyn", "ramp"):
                head["field0_equal_fixed"] = bool(torch.equal(sh[2], fixed_field(arm, cks)))
            if units is not None:
                for kk, a in u0.items():
                    units[f"{name}_p0_{kk}"] = a
        st.begin_task(y, spt, probe_mb=(k == 1))
        st.hook = None if hook is None else (lambda s, k=k: hook(k, s, sh))
        pr = []
        for s in range(spt):
            if mover is not None and s > 0:
                mover.refresh(g0 + s)
            if s % PROBE_EVERY == 0:
                pr.append({"s": s, **probe(params, x, fwd, d)})
            st.step(s)
            if mover is not None:
                mover.step(s)
        if mover is not None:
            mover.refresh(g0 + spt)
        pr.append({"s": spt, **probe(params, x, fwd, d)})
        with torch.no_grad():
            u1, lg1 = RE.unit_point(params, x, fwd, sh)
            memo = float((lg1.argmax(1) == y).float().mean())
            full_ce = float(F.cross_entropy(lg1.double(), y))
        if units is not None:
            for kk, a in u1.items():
                units[f"{name}_p{k}_{kk}"] = a
            units[f"{name}_k{k}_ce"] = st.ce.numpy().astype(np.float32)
            units[f"{name}_k{k}_acc"] = st.acc.numpy().astype(np.float32)
            units[f"{name}_k{k}_step"] = st.stp.numpy().astype(np.float32)
        h = SH.state_sha256(params, adam, ACT)
        task = arm["branch"] + k
        pref = by_task.get(task)
        tr = st.row()
        mbeq = st.mb_equal if fwd is not None else True     # the natural forward is the reference itself
        extra = {}
        if isinstance(mover, DynField):
            m = mover.end_task()
            sp = by_task.get(mover.src_task + k)
            extra = {**m, "shadow_hash_match_prefix": (None if sp is None else m["shadow_state_sha256"] == sp["state_sha256"]),
                     "shadow_acc_match_prefix": (None if sp is None else m["shadow_online_acc"] == sp["online_acc"])}
        pr_dm = [p["neffT2"] for p in pr]
        r_old = rec.get(k)
        row = {"seed": seed, "arm": name, **{f"arm_{a}": arm[a] for a in arm}, "k": k, "task": task,
               **tr, "memo_acc_end": memo, "full_ce_end": full_ce, "state_sha256": h,
               "hash_match_prefix": (None if pref is None else h == pref["state_sha256"]),
               "acc_match_prefix": (None if pref is None else tr["online_acc"] == pref["online_acc"]),
               "hash_match_resp_ee": (None if r_old is None else h == r_old["state_sha256"]),
               "acc_match_resp_ee": (None if r_old is None else tr["online_acc"] == float(r_old["online_acc"])),
               "logits_equal_mb": (mbeq if k == 1 else None),
               "nbar_neffT2": float(np.mean(pr_dm)), "n_probe": len(pr),
               "probe_zero2_mean": float(np.mean([p["zero2"] for p in pr])),
               "probe_effbar2_first": pr[0]["effbar2"], "probe_effbar2_last": pr[-1]["effbar2"],
               "probe_dmean2_first": pr[0]["dmean2"], "probe_dmean2_last": pr[-1]["dmean2"],
               "probe_zarm2_first": pr[0]["zarm2"], "probe_zarm2_last": pr[-1]["zarm2"],
               **extra, **head, **RE.unit_summary(u1, "end"), "sec": time.time() - t_task}
        rows.append(row)
        if traj is not None:
            traj += [{"seed": seed, "arm": name, "k": k, **p} for p in pr]
    if progress:
        print(f"[{time.time() - t0:8.1f}s] seed={seed} arm {name}: "
              + " ".join(f"t{r['task']} {r['online_acc']:.4f}" for r in rows), flush=True)
    return rows


# --------------------------------------------------------------------------

def write_csv(path: Path, rows: list[dict]) -> None:
    RE.write_csv(path, rows)


def link_prefix(seed: int, prows: list[dict]) -> None:
    """resp_ee_0917's prefix for this seed (committed), state sha256 per task (report; machine-bound)."""
    f = RESP_EE_RUNS / f"s{seed}" / "prefix.csv"
    rec = {}
    if f.exists():
        with f.open() as fh:
            rec = {int(r["task"]): r["state_sha256"] for r in csv.DictReader(fh)}
    for r in prows:
        r["hash_match_resp_ee"] = (None if r["task"] not in rec else rec[r["task"]] == r["state_sha256"])


def run_seed(seed: int, out: Path, mnist: H.Mnist, epochs: int = EPOCHS, arms=None, prefix_tasks=PREFIX_T,
             save_ckpt: bool = True, progress: bool = True) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    git0 = dict(GIT_AT_START) or _git_state()
    names = [a for a in ARMS if arms is None or a in arms]
    need = {ARMS[a]["branch"] for a in names} | {ARMS[a]["src"] for a in names if ARMS[a]["src"]}
    need |= {2}
    prows, units, cks, x, info = RE.run_prefix(seed, mnist, epochs, prefix_tasks,
                                               save_at=tuple(sorted(need)), progress=progress)
    link_prefix(seed, prows)
    t_prefix = time.time() - t0
    ds = RE.dstar(cks, x, epochs)
    rows, traj = [], []
    for a in names:
        rows += run_arm(a, ARMS[a], cks, x, prows, seed, epochs, units=units, traj=traj, progress=progress)
    write_csv(out / "prefix.csv", prows)
    write_csv(out / "arms.csv", rows)
    write_csv(out / "traj.csv", traj)
    (out / "dstar.json").write_text(json.dumps(ds, indent=2))
    np.savez_compressed(out / "units.npz", **units)
    if save_ckpt:
        torch.save({t: {k: v for k, v in c.items()} for t, c in cks.items()}, out / "branch_states.pt")
    root = Path(__file__).resolve().parents[1]
    prov = {
        "experiment": EXPERIMENT, "spec": SPEC, "prereg_commit": PREREG_COMMIT,
        "spec_sha256": SH.file_sha256(root / SPEC) if (root / SPEC).exists() else None,
        "git_hash": git0["hash"], "git_dirty_code": git0["dirty"],
        "hostname": socket.gethostname(), "platform": platform.platform(),
        "cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "torch": torch.__version__, "python": sys.version.split()[0],
        "threads": torch.get_num_threads(), "flush_denormal": RE._flush_is_on(),
        "seed": seed, "epochs_per_task": epochs, "steps_per_task": SPE * epochs, "prefix_tasks": prefix_tasks,
        "branch_tasks": sorted(cks), "cont_tasks": CONT, "arms": {a: ARMS[a] for a in names},
        "ramp": RAMP, "probe_every": PROBE_EVERY,
        "lr": LR, "adam": [BETA1, BETA2, EPS], "batch": BATCH, "n_images": N_IMG,
        "code_sha256": {f"src/{n}": SH.file_sha256(root / "src" / n) for n in
                        ("respdyn_ee_0917.py", "resp_ee_0917.py", "mucap_el_run_0916.py", "pmnist_0905.py",
                         "pmnist_rlmnist_0906.py", "elu_growth_0909.py", "shell_l2_rlmnist_0913.py")},
        "data_sha256": mnist.sha256, "prefix_info": info,
        "prefix_hash_match_resp_ee": sum(bool(r["hash_match_resp_ee"]) for r in prows),
        "seconds_prefix": t_prefix, "seconds_total": time.time() - t0,
        "peak_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    (out / "provenance.json").write_text(json.dumps(prov, indent=2, default=str))
    if progress:
        print(f"wrote {out}  ({len(prows)} prefix rows, {len(rows)} arm rows, {time.time() - t0:.1f}s)",
              flush=True)
    return prov


GIT_AT_START: dict = {}


def _git_state() -> dict:
    """Read once, when the process starts (a commit made during a long run must not relabel it)."""
    return {"hash": H.git_hash(), "dirty": SH.git_dirty(["src", "analysis/respdyn_ee_0917"])}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--arms", default=None, help="comma list (checks and S-cost only)")
    ap.add_argument("--prefix-tasks", type=int, default=PREFIX_T)
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args(argv)
    GIT_AT_START.update(_git_state())
    torch.set_num_threads(args.threads)
    torch.set_flush_denormal(True)
    H.setup("cpu")
    mnist = H.Mnist(torch.device("cpu"))
    arms = args.arms.split(",") if args.arms else None
    run_seed(args.seed, Path(args.out), mnist, args.epochs, arms, args.prefix_tasks)


if __name__ == "__main__":
    main()
