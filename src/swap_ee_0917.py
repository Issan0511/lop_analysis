#!/usr/bin/env python3
"""Swap the second layer's response field between the cap12 net and the ref net at task 10 (RL ELU->ELU).

    OMP_NUM_THREADS=1 python3 src/swap_ee_0917.py --seed 0 --out results/swap_ee_0917/runs/s0

specs/spec_swap_ee_0917.md (registered at PREREG_COMMIT).  The question is respdyn_ee_0917's remainder:
a healthy t2 net given the collapsing t10 net's moving field still learns 0.112 above the response floor
(S2dyn_10r - S2u30r), and that run read it as "the natural net's own weights push, the transplant host's
do not" (unverified).  l2cap_ee_0917's cap12 net (first-layer mu-component caps + second-layer row-norm
cap) is a host whose own weights cannot push: its lever ||w2|| ||mu2|| stays near 9 where ref's is 272.

One process per seed:
  1. ref's natural prefix, tasks 1-12 (resp_ee_0917's run_prefix, branch states at t2 and t10);
  2. cap12's natural prefix, tasks 1-12: the same update loop (respdyn_ee_0917's Stepper) with the caps
     applied right after every Adam update from task 2, in src/mucap_el_run_0916.run_one's order, so the
     trajectory is l2cap_ee_0917's cap12 bit for bit (recorded per task against the archived units);
  3. 14 arms, each continuing a t10 branch (or ref's t2 branch) for 2 tasks with a fresh Adam, the
     branch's own label and order streams, resp_ee's anchored forward
         a2 = phi(z0_2) + [phi(z2 + d) - phi(z0_2 + d)]
     and respdyn's fields: fixed d = z2^(src) - z2^(branch), moving d(s) = z2^(shadow)(s) - z2^(branch)
     with the shadow = the source network's own natural continuation (its caps included), or uniform
     d = -30.  An arm's caps are None, "own" (cap12's task-1-end radii) or "here" (radii taken at the
     branch state: further growth stopped, nothing moved at the branch point).

Host C = cap12 at t10, host R = ref at t10, host T2 = ref at t2 (two respdyn/resp_ee arms reproduced).
"""

from __future__ import annotations

import argparse
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
from src import mucap_el_0916 as MU              # first-layer caps; not touched
from src import l2cap_ee_0917 as W2C             # second-layer row-norm cap; not touched
from src import mucap_el_run_0916 as EL          # forward2, unit_arrays; not touched
from src import resp_ee_0917 as RE               # prefix, anchored forward, unit summaries; not touched
from src import respdyn_ee_0917 as RD            # Stepper, DynField, probe, run_arm; not touched

EXPERIMENT = "swap_ee_0917"
SPEC = "specs/spec_swap_ee_0917.md"
PREREG_COMMIT = "unregistered"                   # set to the spec's registration commit before the run
ACT = RE.ACT
EPOCHS = RE.EPOCHS                               # 80 -> 6000 updates per task
SPE = RE.SPE
CONT = RE.CONT                                   # 2 continuation tasks per arm
BRANCH = 10
PREFIX_T = BRANCH + CONT                         # 12
PROBE_EVERY = RD.PROBE_EVERY
ROOT = Path(__file__).resolve().parents[1]
RESP_EE_RUNS = ROOT / "results" / "resp_ee_0917" / "runs"
RESPDYN_RUNS = ROOT / "results" / "respdyn_ee_0917" / "runs"
L2CAP_MANIFEST = ROOT / "results" / "l2cap_ee_0917" / "backup_manifest.json"


def _arm(branch, kind=None, src=None, delta=None, caps=None):
    return {"branch": branch, "layer": 2 if kind is not None else None, "kind": kind, "src": src,
            "delta": delta, "caps": caps, "reset": True}


ARMS: dict[str, dict] = {
    # host C: the cap12 net at t10 (its own caps on unless "free")
    "NC_r": _arm("C10", caps="own"),
    "NC_free_r": _arm("C10"),
    "SC_fix_r": _arm("C10", "field", src="R10", caps="own"),
    "SC_dyn_r": _arm("C10", "dyn", src="R10", caps="own"),
    "SC_dyn_free_r": _arm("C10", "dyn", src="R10"),
    "SC_u30_r": _arm("C10", "uniform", delta=30.0, caps="own"),
    # host R: the ref net at t10 (free unless "here")
    "NR_r": _arm("R10"),
    "NR_cap_r": _arm("R10", caps="here"),
    "RR_fix_r": _arm("R10", "field", src="C10"),
    "RR_dyn_r": _arm("R10", "dyn", src="C10"),
    "RR_dyn_cap_r": _arm("R10", "dyn", src="C10", caps="here"),
    "RR_u30_r": _arm("R10", "uniform", delta=30.0),
}
T2_ARMS = ("S2dyn_10r", "S2u30r")                # host T2: respdyn's primary and resp_ee's floor, re-run
ALL_ARMS = (*ARMS, *T2_ARMS)
RESP_EE_TWIN = {"NR_r": "N10r", "S2u30r": "S2u30r"}   # same state, same streams: the hash must match


# --------------------------------------------------------------------------
# caps inside the per-update loop
# --------------------------------------------------------------------------

class Caps:
    """src/mucap_el_run_0916.run_one's caps, in its order, right after an Adam update: the first
    layer's parallel then perpendicular mu-component cap, then the second layer's row-norm cap."""

    def __init__(self, e1, q_cap, v_cap, r2):
        self.e1, self.q_cap, self.v_cap, self.r2 = e1, q_cap, v_cap, r2

    @classmethod
    def at(cls, params, e1):
        """Radii read off this state (so applying them at this state writes nothing)."""
        return cls(e1, MU.parallel_cap(params[0], e1), MU.perp_cap(params[0], e1), W2C.row_norms(params[2]))

    @torch.no_grad()
    def apply(self, params) -> tuple[int, int, int]:
        a = int(MU.cap_parallel_(params[0], self.e1, self.q_cap)[0])
        b = int(MU.cap_perp_(params[0], self.e1, self.v_cap)[0])
        c = int(W2C.cap_row_norm_(params[2], self.r2)[0])
        return a, b, c

    @torch.no_grad()
    def excess(self, params) -> dict:
        """How far above each radius the state is (float64; <= rounding when the caps are on)."""
        q, v = MU.split_mu(params[0].double(), self.e1.double())
        n2 = torch.linalg.vector_norm(params[2].double(), dim=1, keepdim=True)
        return {"exc_q": float((q.abs() - self.q_cap.double()).max()),
                "exc_v": float((torch.linalg.vector_norm(v, dim=1, keepdim=True) - self.v_cap.double()).max()),
                "exc_w2": float((n2 - self.r2.double()).max())}


class CapStepper(RD.Stepper):
    """RD.Stepper with the caps applied after each update (None: RD.Stepper itself)."""

    def __init__(self, params, adam, x, g_batch, fwd, caps: Caps | None = None):
        super().__init__(params, adam, x, g_batch, fwd)
        self.caps = caps
        self.rows_written = [0, 0, 0]

    def step(self, s):
        super().step(s)
        if self.caps is not None:
            for i, n in enumerate(self.caps.apply(self.params)):
                self.rows_written[i] += n


# --------------------------------------------------------------------------
# the cap12 prefix
# --------------------------------------------------------------------------

def _l2cap_units(seed: int) -> dict | None:
    """l2cap_ee_0917's archived cap12 units for this seed (only if the file's sha256 is the manifest's)."""
    man = json.loads(L2CAP_MANIFEST.read_text())
    rel = f"results/l2cap_ee_0917/runs/cap12_s{seed}/units.npz"
    for f in man["files"]:
        if f["source_rel"] == rel:
            p = Path(f["backup"])
            if not p.exists() or RE_sha256(p) != f["sha256"]:
                return None
            with np.load(p) as z:
                return dict(z)
    return None


def RE_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def run_cap_prefix(seed: int, mnist: H.Mnist, epochs: int = EPOCHS, n_tasks: int = PREFIX_T,
                   save_at=(BRANCH,), progress: bool = False, archive: dict | None = None):
    t0 = time.time()
    params = H.init_params(seed, H.setup("cpu"))
    adam = ([torch.zeros_like(q) for q in params], [torch.zeros_like(q) for q in params], [0])
    idx = RL.subset_idx(seed)
    x = mnist.train_x[idx]
    e1, e1_64, _ = MU.mu_basis(x)
    g_lab, g_batch = H.stream("rl_labels", seed), H.stream("rl_batch", seed)
    spt = SPE * epochs
    st = CapStepper(params, adam, x, g_batch, None, None)
    caps = None
    rows, cks = [], {}
    pre = f"s{seed}_"
    if archive is not None:
        a_task, a_step = archive[pre + "task"].astype(int), archive[pre + "step"]
        a_end = {int(t): i for i, (t, s) in enumerate(zip(a_task, a_step)) if s == spt}
    for t in range(1, n_tasks + 1):
        t_task = time.time()
        y = RL.task_labels(g_lab)
        st.caps = caps
        st.rows_written = [0, 0, 0]
        st.begin_task(y, spt)
        for s in range(spt):
            st.step(s)
        if t == 1:
            caps = Caps.at(params, e1)
        with torch.no_grad():
            z1, _, z2, _, logits = EL.forward2(params, x, ACT, ACT)
            memo = float((logits.argmax(1) == y).float().mean())
        u, _ = RE.unit_point(params, x, None, None)
        row = {"seed": seed, "net": "C", "task": t, **st.row(), "memo_acc": memo,
               "rows_par": st.rows_written[0], "rows_perp": st.rows_written[1], "rows_w2": st.rows_written[2],
               "state_sha256": SH.state_sha256(params, adam, ACT), **RE.unit_summary(u, "end"),
               **caps.excess(params), "sec": time.time() - t_task}
        if archive is not None and t in a_end:
            ua = EL.unit_arrays(params, x, ACT, ACT, e1_64)
            i = a_end[t]
            bad = [k for k, v in ua.items() if (pre + k) in archive
                   and not np.array_equal(archive[pre + k][i], v, equal_nan=True)]
            row["match_l2cap_units"] = not bad
            row["n_l2cap_keys"] = sum((pre + k) in archive for k in ua)
            row["l2cap_bad_keys"] = ";".join(bad[:5])
        else:
            row["match_l2cap_units"] = None
        rows.append(row)
        if t in save_at:
            cks[f"C{t}"] = {"params": [p.detach().clone() for p in params],
                            "m": [q.clone() for q in adam[0]], "v": [q.clone() for q in adam[1]],
                            "tc": adam[2][0], "g_lab": g_lab.get_state().clone(),
                            "g_batch": g_batch.get_state().clone(), "state_sha256": row["state_sha256"],
                            "z1": z1.detach().clone(), "z2": z2.detach().clone(), "task": t, "net": "C",
                            "caps": caps}
        if progress:
            print(f"[{time.time() - t0:8.1f}s] seed={seed} cap12 prefix task {t}/{n_tasks} "
                  f"online {row['online_acc']:.4f}", flush=True)
    return rows, cks, x, e1


# --------------------------------------------------------------------------
# moving field with the source's own caps
# --------------------------------------------------------------------------

class CapDynField(RD.DynField):
    """RD.DynField whose shadow applies the source network's own caps (none for ref)."""

    def __init__(self, arm, cks, x, d):
        super().__init__(arm, cks, x, d)
        src = cks[arm["src"]]
        self.stepper = CapStepper(self.params, self.adam, x, self.g_batch, None, src.get("caps"))
        self.src_net = src.get("net", "R")


def build(arm: dict, cks: dict, x: torch.Tensor):
    kind = arm["kind"]
    if kind in (None, "field", "uniform"):
        sh = RE.build_shift(arm, cks)
        return sh, RE.make_forward(sh), None
    if kind == "dyn":
        sh = RE.build_shift({**arm, "kind": "field"}, cks)
        return sh, RE.make_forward(sh), CapDynField(arm, cks, x, sh[2])
    raise ValueError(kind)


def arm_caps(arm: dict, ck: dict, e1) -> Caps | None:
    if arm["caps"] is None:
        return None
    if arm["caps"] == "own":
        return ck["caps"]
    if arm["caps"] == "here":
        return Caps.at(ck["params"], e1)
    raise ValueError(arm["caps"])


def run_arm(name: str, arm: dict, cks: dict, x: torch.Tensor, e1, nat_rows: dict, seed: int,
            epochs: int = EPOCHS, units: dict | None = None, traj: list | None = None,
            progress: bool = False) -> list[dict]:
    """RD.run_arm for the swap arms: string branch keys, caps in the arm's own loop, shadows that keep
    their network's caps, and the natural continuation (Adam continued) of either network as the
    shadow's reference.  nat_rows[(net, task)] -> prefix row."""
    t0 = time.time()
    ck = cks[arm["branch"]]
    params = [p.detach().clone().requires_grad_(True) for p in ck["params"]]
    adam = ([torch.zeros_like(q) for q in params], [torch.zeros_like(q) for q in params], [0])
    g_lab, g_batch = RD.gen_from(ck["g_lab"]), RD.gen_from(ck["g_batch"])
    spt = SPE * epochs
    sh, fwd, mover = build(arm, cks, x)
    d = None if sh is None else sh.get(2)
    caps = arm_caps(arm, ck, e1)
    own = ck.get("caps") or Caps.at(ck["params"], e1)      # the reference radii for the excess record
    with torch.no_grad():
        nat = EL.forward2(params, x, ACT, ACT)[4]
    st = CapStepper(params, adam, x, g_batch, fwd, caps)
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
            if arm["kind"] == "dyn":
                head["field0_equal_fixed"] = bool(torch.equal(sh[2], RE.build_shift({**arm, "kind": "field"}, cks)[2]))
            if units is not None:
                for kk, a in u0.items():
                    units[f"{name}_p0_{kk}"] = a
        st.begin_task(y, spt, probe_mb=(k == 1))
        st.rows_written = [0, 0, 0]
        pr = []
        for s in range(spt):
            if mover is not None and s > 0:
                mover.refresh(g0 + s)
            if s % PROBE_EVERY == 0:
                pr.append({"s": s, **RD.probe(params, x, fwd, d)})
            st.step(s)
            if mover is not None:
                mover.step(s)
        if mover is not None:
            mover.refresh(g0 + spt)
        pr.append({"s": spt, **RD.probe(params, x, fwd, d)})
        with torch.no_grad():
            u1, lg1 = RE.unit_point(params, x, fwd, sh)
            memo = float((lg1.argmax(1) == y).float().mean())
        if units is not None:
            for kk, a in u1.items():
                units[f"{name}_p{k}_{kk}"] = a
            units[f"{name}_k{k}_acc"] = st.acc.numpy().astype(np.float32)
        h = SH.state_sha256(params, adam, ACT)
        tr = st.row()
        extra = {}
        if mover is not None:
            m = mover.end_task()
            ref_row = nat_rows.get((mover.src_net, BRANCH + k))
            extra = {**m, "shadow_hash_match_natural": (None if ref_row is None else
                                                        m["shadow_state_sha256"] == ref_row["state_sha256"])}
        exc = {f"{kk}_own": vv for kk, vv in own.excess(params).items()}
        row = {"seed": seed, "arm": name, **{f"arm_{a}": arm[a] for a in arm}, "k": k, "task": BRANCH + k,
               **tr, "memo_acc_end": memo, "state_sha256": h,
               "rows_par": st.rows_written[0], "rows_perp": st.rows_written[1], "rows_w2": st.rows_written[2],
               "logits_equal_mb": (st.mb_equal if fwd is not None else True) if k == 1 else None,
               "nbar_neffT2": float(np.mean([p["neffT2"] for p in pr])), "n_probe": len(pr),
               "probe_zero2_mean": float(np.mean([p["zero2"] for p in pr])),
               "probe_effbar2_first": pr[0]["effbar2"], "probe_effbar2_750": pr[1]["effbar2"],
               "probe_effbar2_last": pr[-1]["effbar2"],
               "probe_dmean2_first": pr[0]["dmean2"], "probe_dmean2_750": pr[1]["dmean2"],
               "probe_dmean2_last": pr[-1]["dmean2"],
               "probe_zarm2_first": pr[0]["zarm2"], "probe_zarm2_750": pr[1]["zarm2"],
               "probe_zarm2_last": pr[-1]["zarm2"],
               "w2norm_end": float(torch.linalg.vector_norm(params[2].detach().double(), dim=1).mean()),
               **exc, **(caps.excess(params) if caps is not None else {}),
               **extra, **head, **RE.unit_summary(u1, "end"), "sec": time.time() - t_task}
        rows.append(row)
        if traj is not None:
            traj += [{"seed": seed, "arm": name, "k": k, **p} for p in pr]
    if progress:
        print(f"[{time.time() - t0:8.1f}s] seed={seed} arm {name}: "
              + " ".join(f"t{r['task']} {r['online_acc']:.4f}" for r in rows), flush=True)
    return rows


# --------------------------------------------------------------------------

def _csv_rows(path: Path) -> list[dict]:
    import csv
    if not path.exists():
        return []
    with path.open() as fh:
        return list(csv.DictReader(fh))


def prefixes(seed: int, mnist: H.Mnist, epochs: int = EPOCHS, progress: bool = True, archive: bool = True):
    """Both natural prefixes and the branch states (checks reuse one call across mutated arm code)."""
    rrows, units, rcks, x, info = RE.run_prefix(seed, mnist, epochs, PREFIX_T, save_at=(2, BRANCH),
                                                progress=progress)
    rec = {int(r["task"]): r["state_sha256"] for r in _csv_rows(RESP_EE_RUNS / f"s{seed}" / "prefix.csv")}
    for r in rrows:
        r["net"] = "R"
        r["hash_match_resp_ee"] = (None if r["task"] not in rec else rec[r["task"]] == r["state_sha256"])
    arch = _l2cap_units(seed) if (archive and epochs == EPOCHS) else None
    crows, ccks, x2, e1 = run_cap_prefix(seed, mnist, epochs, PREFIX_T, progress=progress, archive=arch)
    assert torch.equal(x, x2)
    rcks[BRANCH]["net"] = "R"
    cks = {"R10": rcks[BRANCH], "C10": ccks[f"C{BRANCH}"], 2: rcks[2], 10: rcks[BRANCH]}
    return {"rrows": rrows, "crows": crows, "units": units, "cks": cks, "x": x, "e1": e1, "info": info,
            "archive_used": arch is not None}


def run_seed(seed: int, out: Path, mnist: H.Mnist, epochs: int = EPOCHS, arms=None, progress: bool = True,
             archive: bool = True, pre: dict | None = None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    git0 = dict(GIT_AT_START) or _git_state()
    names = [a for a in ALL_ARMS if arms is None or a in arms]
    # 1-2. ref and cap12
    if pre is None:
        pre = prefixes(seed, mnist, epochs, progress, archive)
    rrows, crows, cks, x, e1, info = pre["rrows"], pre["crows"], pre["cks"], pre["x"], pre["e1"], pre["info"]
    units = dict(pre["units"])
    nat_rows = {("R", r["task"]): r for r in rrows} | {("C", r["task"]): r for r in crows}
    t_prefix = time.time() - t0
    # 3. arms
    rows, traj = [], []
    resp_rec = {(r["arm"], int(r["k"])): r for r in _csv_rows(RESP_EE_RUNS / f"s{seed}" / "arms.csv")}
    dyn_rec = {(r["arm"], int(r["k"])): r for r in _csv_rows(RESPDYN_RUNS / f"s{seed}" / "arms.csv")}
    for a in names:
        if a in ARMS:
            got = run_arm(a, ARMS[a], cks, x, e1, nat_rows, seed, epochs, units=units, traj=traj,
                          progress=progress)
        else:
            got = RD.run_arm(a, RD.ARMS[a], cks, x, rrows, seed, epochs, units=units, traj=traj,
                             progress=progress)
            for r in got:
                r["seed"] = seed
        for r in got:
            twin = RESP_EE_TWIN.get(a)
            r["hash_match_resp_ee"] = (None if twin is None or (twin, r["k"]) not in resp_rec else
                                       r["state_sha256"] == resp_rec[(twin, r["k"])]["state_sha256"])
            r["hash_match_respdyn"] = (None if (a, r["k"]) not in dyn_rec or a not in T2_ARMS else
                                       r["state_sha256"] == dyn_rec[(a, r["k"])]["state_sha256"])
        rows += got
    RE.write_csv(out / "prefix.csv", rrows + crows)
    RE.write_csv(out / "arms.csv", rows)
    RE.write_csv(out / "traj.csv", traj)
    np.savez_compressed(out / "units.npz", **units)
    prov = {
        "experiment": EXPERIMENT, "spec": SPEC, "prereg_commit": PREREG_COMMIT,
        "spec_sha256": SH.file_sha256(ROOT / SPEC) if (ROOT / SPEC).exists() else None,
        "git_hash": git0["hash"], "git_dirty_code": git0["dirty"],
        "hostname": socket.gethostname(), "platform": platform.platform(),
        "cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "torch": torch.__version__, "python": sys.version.split()[0],
        "threads": torch.get_num_threads(), "flush_denormal": RE._flush_is_on(),
        "seed": seed, "epochs_per_task": epochs, "steps_per_task": SPE * epochs, "prefix_tasks": PREFIX_T,
        "branch": BRANCH, "cont_tasks": CONT, "arms": {a: (ARMS.get(a) or RD.ARMS[a]) for a in names},
        "l2cap_archive_used": pre["archive_used"],
        "code_sha256": {f"src/{n}": SH.file_sha256(ROOT / "src" / n) for n in
                        ("swap_ee_0917.py", "respdyn_ee_0917.py", "resp_ee_0917.py", "mucap_el_run_0916.py",
                         "mucap_el_0916.py", "l2cap_ee_0917.py", "pmnist_0905.py", "pmnist_rlmnist_0906.py",
                         "elu_growth_0909.py", "shell_l2_rlmnist_0913.py")},
        "data_sha256": mnist.sha256, "prefix_info": info,
        "ref_prefix_hash_match_resp_ee": sum(bool(r["hash_match_resp_ee"]) for r in rrows),
        "cap_prefix_match_l2cap": sum(bool(r["match_l2cap_units"]) for r in crows),
        "seconds_prefix": t_prefix, "seconds_total": time.time() - t0,
        "peak_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    (out / "provenance.json").write_text(json.dumps(prov, indent=2, default=str))
    if progress:
        print(f"wrote {out}  ({len(rrows) + len(crows)} prefix rows, {len(rows)} arm rows, "
              f"{time.time() - t0:.1f}s)", flush=True)
    return prov


GIT_AT_START: dict = {}


def _git_state() -> dict:
    """Read once, when the process starts (a commit made during a long run must not relabel it)."""
    return {"hash": H.git_hash(), "dirty": SH.git_dirty(["src", "analysis/swap_ee_0917"])}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--arms", default=None, help="comma list (checks and S-cost only)")
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args(argv)
    GIT_AT_START.update(_git_state())
    torch.set_num_threads(args.threads)
    torch.set_flush_denormal(True)
    H.setup("cpu")
    mnist = H.Mnist(torch.device("cpu"))
    arms = args.arms.split(",") if args.arms else None
    run_seed(args.seed, Path(args.out), mnist, args.epochs, arms)


if __name__ == "__main__":
    main()
