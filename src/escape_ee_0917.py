#!/usr/bin/env python3
"""Does respdyn_ee_0917's remainder measure how far the transplant host can escape on its own?

    OMP_NUM_THREADS=1 python3 src/escape_ee_0917.py --seed 0 --out results/escape_ee_0917/runs/s0

specs/spec_escape_ee_0917.md (registered at PREREG_COMMIT).  respdyn's primary arm S2dyn_10r (the healthy
t2 net answering with the collapsing t10 net's moving second-layer field, fresh Adam) learns 0.112 above
the response floor S2u30r.  swap_ee_0917 found that remainder much smaller in a host that cannot grow
(cap12: +0.037) and, post hoc, ordered by how far each host's own second layer climbed under the imposed
field (t2 +3.5, cap12 free +1.6, cap12 capped +1.0).  Here the host is respdyn's own t2 net, and only its
room to move is changed: after every Adam update a "hold" is applied with radii / values read off the t2
branch state (so nothing moves at the branch point):

    none   nothing (respdyn's arm, reproduced bit for bit)
    c1     first layer: |q_i| and ||v_i|| capped (mucap_el_0916's caps, the seed's mean-image axis)
    c2     second layer: row norms capped (l2cap_ee_0917's cap)
    c12    both
    c12b   both, and b1, b2 put back to their t2 values
    frz    W1, b1, W2, b2 put back to their t2 values (no escape at all: the host's features are the
           branch's, exactly as on the floor, so this arm equals the floor bit for bit -- a calibration)

The same holds on the fixed field (S2_10r), the floor (S2u30r) and the natural continuation (N2r, the
impairment control).  Everything else is resp_ee / respdyn / swap_ee's machinery, imported: the prefix,
the anchored forward, the shadow that follows N10, the per-update loop, the probes.
"""

from __future__ import annotations

import argparse
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import pmnist_0905 as H                 # host; not touched
from src import pmnist_rlmnist_0906 as RL        # subset, labels; not touched
from src import shell_l2_rlmnist_0913 as SH      # state_sha256, file_sha256, git_dirty; not touched
from src import mucap_el_0916 as MU              # first-layer caps; not touched
from src import l2cap_ee_0917 as W2C             # second-layer row-norm cap; not touched
from src import mucap_el_run_0916 as EL          # forward2; not touched
from src import resp_ee_0917 as RE               # prefix, anchored forward, unit summaries; not touched
from src import respdyn_ee_0917 as RD            # probe, gen_from; not touched
from src import swap_ee_0917 as SW               # CapStepper, CapDynField, build, _csv_rows; not touched

EXPERIMENT = "escape_ee_0917"
SPEC = "specs/spec_escape_ee_0917.md"
PREREG_COMMIT = "106b28fe43993a7e76fae75959ca230ddd323715"   # specs/spec_escape_ee_0917.md, pushed before any held arm ran for the record
ACT = RE.ACT
EPOCHS = RE.EPOCHS
SPE = RE.SPE
CONT = RE.CONT
BRANCH, SRC = 2, 10
PREFIX_T = SRC + CONT                            # 12: the shadow N10 must meet the prefix at t11 and t12
PROBE_EVERY = RD.PROBE_EVERY
ROOT = Path(__file__).resolve().parents[1]
RESP_EE_RUNS = ROOT / "results" / "resp_ee_0917" / "runs"
RESPDYN_RUNS = ROOT / "results" / "respdyn_ee_0917" / "runs"
HOLDS = ("none", "c1", "c2", "c12", "c12b", "frz")


def _arm(kind=None, delta=None, hold="none"):
    return {"branch": BRANCH, "layer": 2 if kind is not None else None, "kind": kind,
            "src": SRC if kind in ("field", "dyn") else None, "delta": delta, "hold": hold, "reset": True}


ARMS: dict[str, dict] = {
    "S2dyn_10r": _arm("dyn"),
    "S2dyn_c1": _arm("dyn", hold="c1"),
    "S2dyn_c2": _arm("dyn", hold="c2"),
    "S2dyn_c12": _arm("dyn", hold="c12"),
    "S2dyn_c12b": _arm("dyn", hold="c12b"),
    "S2dyn_frz": _arm("dyn", hold="frz"),
    "S2u30r": _arm("uniform", delta=30.0),
    "S2u30r_c12": _arm("uniform", delta=30.0, hold="c12"),
    "S2_10r": _arm("field"),
    "S2_10r_c12": _arm("field", hold="c12"),
    "N2r": _arm(),
    "N2r_c12": _arm(hold="c12"),
}
# the respdyn / resp_ee arms with the same name are the same computation: their state hash must match
REPRO = {"S2dyn_10r": ("respdyn",), "S2u30r": ("respdyn", "resp_ee"), "S2_10r": ("respdyn", "resp_ee"),
         "N2r": ("respdyn", "resp_ee")}


class Hold:
    """Applied right after every Adam update (CapStepper's slot).  Radii and held values are the t2
    branch state's, so applying the hold at the branch state writes nothing."""

    def __init__(self, kind: str, params, e1):
        if kind not in HOLDS or kind == "none":
            raise ValueError(kind)
        self.kind = kind
        self.l1 = kind in ("c1", "c12", "c12b")
        self.l2 = kind in ("c2", "c12", "c12b")
        self.bias = kind == "c12b"
        self.e1 = e1
        with torch.no_grad():
            self.q_cap = MU.parallel_cap(params[0], e1)
            self.v_cap = MU.perp_cap(params[0], e1)
            self.r2 = W2C.row_norms(params[2])
            self.p0 = [p.detach().clone() for p in params[:4]]
        self.restored = 0

    @torch.no_grad()
    def apply(self, params) -> tuple[int, int, int]:
        a = b = c = 0
        if self.kind == "frz":
            for k in range(4):
                self.restored += int((params[k] != self.p0[k]).sum())
                params[k].copy_(self.p0[k])
            return 0, 0, 0
        if self.l1:
            a = int(MU.cap_parallel_(params[0], self.e1, self.q_cap)[0])
            b = int(MU.cap_perp_(params[0], self.e1, self.v_cap)[0])
        if self.l2:
            c = int(W2C.cap_row_norm_(params[2], self.r2)[0])
        if self.bias:
            for k in (1, 3):
                self.restored += int((params[k] != self.p0[k]).sum())
                params[k].copy_(self.p0[k])
        return a, b, c

    @torch.no_grad()
    def excess(self, params) -> dict:
        """How far the state is outside the hold (float64): caps above radius, held tensors off value."""
        out = {}
        if self.kind == "frz":
            out["exc_frz"] = max(float((params[k].double() - self.p0[k].double()).abs().max()) for k in range(4))
            return out
        if self.l1:
            q, v = MU.split_mu(params[0].double(), self.e1.double())
            out["exc_q"] = float((q.abs() - self.q_cap.double()).max())
            out["exc_v"] = float((torch.linalg.vector_norm(v, dim=1, keepdim=True) - self.v_cap.double()).max())
        if self.l2:
            n2 = torch.linalg.vector_norm(params[2].double(), dim=1, keepdim=True)
            out["exc_w2"] = float((n2 - self.r2.double()).max())
        if self.bias:
            out["exc_b"] = max(float((params[k].double() - self.p0[k].double()).abs().max()) for k in (1, 3))
        return out


def run_arm(name: str, arm: dict, cks: dict, x: torch.Tensor, e1, nat_rows: dict, seed: int,
            epochs: int = EPOCHS, units: dict | None = None, traj: list | None = None,
            progress: bool = False) -> list[dict]:
    """swap_ee_0917.run_arm with a Hold in place of the caps and the t2 host's task labels."""
    t0 = time.time()
    ck = cks[arm["branch"]]
    params = [p.detach().clone().requires_grad_(True) for p in ck["params"]]
    adam = ([torch.zeros_like(q) for q in params], [torch.zeros_like(q) for q in params], [0])
    g_lab, g_batch = RD.gen_from(ck["g_lab"]), RD.gen_from(ck["g_batch"])
    spt = SPE * epochs
    sh, fwd, mover = SW.build(arm, cks, x)
    d = None if sh is None else sh.get(2)
    hold = None if arm["hold"] == "none" else Hold(arm["hold"], ck["params"], e1)
    with torch.no_grad():
        nat = EL.forward2(params, x, ACT, ACT)[4]
    st = SW.CapStepper(params, adam, x, g_batch, fwd, hold)
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
                    **RE.unit_summary(u0, "start"),
                    **RE.preflight(params, fwd, x, y, g_batch.get_state(), epochs)}
            if arm["kind"] == "dyn":
                head["field0_equal_fixed"] = bool(torch.equal(sh[2], RE.build_shift({**arm, "kind": "field"}, cks)[2]))
            if units is not None:
                for kk, a in u0.items():
                    units[f"{name}_p0_{kk}"] = a
        st.begin_task(y, spt, probe_mb=(k == 1))
        st.rows_written = [0, 0, 0]
        if hold is not None:
            hold.restored = 0
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
        extra = {}
        if mover is not None:
            m = mover.end_task()
            ref_row = nat_rows.get(SRC + k)
            extra = {**m, "shadow_hash_match_natural": (None if ref_row is None else
                                                        m["shadow_state_sha256"] == ref_row["state_sha256"])}
        with torch.no_grad():
            w1n = torch.linalg.vector_norm(params[0].detach().double(), dim=1).mean()
            w2n = torch.linalg.vector_norm(params[2].detach().double(), dim=1).mean()
        row = {"seed": seed, "arm": name, **{f"arm_{a}": arm[a] for a in arm}, "k": k, "task": BRANCH + k,
               **st.row(), "memo_acc_end": memo, "state_sha256": h,
               "rows_par": st.rows_written[0], "rows_perp": st.rows_written[1], "rows_w2": st.rows_written[2],
               "restored": (hold.restored if hold is not None else 0),
               "logits_equal_mb": (st.mb_equal if fwd is not None else True) if k == 1 else None,
               "nbar_neffT2": float(np.mean([p["neffT2"] for p in pr])),
               "probe_zero2_mean": float(np.mean([p["zero2"] for p in pr])),
               "probe_zarm2_first": pr[0]["zarm2"], "probe_zarm2_750": pr[1]["zarm2"],
               "probe_zarm2_last": pr[-1]["zarm2"], "climb2": pr[-1]["zarm2"] - pr[0]["zarm2"],
               "probe_effbar2_first": pr[0]["effbar2"], "probe_effbar2_750": pr[1]["effbar2"],
               "probe_effbar2_last": pr[-1]["effbar2"],
               "w1norm_end": float(w1n), "w2norm_end": float(w2n),
               "b2_mean_end": float(params[3].detach().double().mean()),
               **(hold.excess(params) if hold is not None else {}),
               **extra, **head, **RE.unit_summary(u1, "end"), "sec": time.time() - t_task}
        rows.append(row)
        if traj is not None:
            traj += [{"seed": seed, "arm": name, "k": k, **p} for p in pr]
    if progress:
        print(f"[{time.time() - t0:8.1f}s] seed={seed} arm {name}: "
              + " ".join(f"t{r['task']} {r['online_acc']:.4f}" for r in rows), flush=True)
    return rows


def prefix(seed: int, mnist: H.Mnist, epochs: int = EPOCHS, progress: bool = True) -> dict:
    rows, units, cks, x, info = RE.run_prefix(seed, mnist, epochs, PREFIX_T, save_at=(BRANCH, SRC),
                                              progress=progress)
    rec = {int(r["task"]): r["state_sha256"] for r in SW._csv_rows(RESP_EE_RUNS / f"s{seed}" / "prefix.csv")}
    for r in rows:
        r["hash_match_resp_ee"] = (None if r["task"] not in rec else rec[r["task"]] == r["state_sha256"])
    e1 = MU.mu_basis(x)[0]
    return {"rows": rows, "units": units, "cks": cks, "x": x, "e1": e1, "info": info}


def run_seed(seed: int, out: Path, mnist: H.Mnist, epochs: int = EPOCHS, arms=None, progress: bool = True,
             pre: dict | None = None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    git0 = dict(GIT_AT_START) or _git_state()
    names = [a for a in ARMS if arms is None or a in arms]
    if pre is None:
        pre = prefix(seed, mnist, epochs, progress)
    prows, cks, x, e1 = pre["rows"], pre["cks"], pre["x"], pre["e1"]
    units = dict(pre["units"])
    nat_rows = {r["task"]: r for r in prows}
    t_prefix = time.time() - t0
    recs = {"resp_ee": {(r["arm"], int(r["k"])): r for r in SW._csv_rows(RESP_EE_RUNS / f"s{seed}" / "arms.csv")},
            "respdyn": {(r["arm"], int(r["k"])): r for r in SW._csv_rows(RESPDYN_RUNS / f"s{seed}" / "arms.csv")}}
    rows, traj = [], []
    for a in names:
        got = run_arm(a, ARMS[a], cks, x, e1, nat_rows, seed, epochs, units=units, traj=traj, progress=progress)
        for r in got:
            for src in ("resp_ee", "respdyn"):
                rec = recs[src].get((a, r["k"])) if src in REPRO.get(a, ()) else None
                r[f"hash_match_{src}"] = None if rec is None else r["state_sha256"] == rec["state_sha256"]
        rows += got
    by = {(r["arm"], r["k"]): r for r in rows}
    for k in (1, 2):                              # the capped floor is the floor (no gradient reaches W1, W2)
        if ("S2u30r_c12", k) in by and ("S2u30r", k) in by:
            by[("S2u30r_c12", k)]["hash_equal_floor"] = by[("S2u30r_c12", k)]["state_sha256"] == by[("S2u30r", k)]["state_sha256"]
        if ("S2dyn_frz", k) in by and ("S2u30r", k) in by:
            by[("S2dyn_frz", k)]["acc_equal_floor"] = by[("S2dyn_frz", k)]["online_acc"] == by[("S2u30r", k)]["online_acc"]
    RE.write_csv(out / "prefix.csv", prows)
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
        "branch": BRANCH, "src": SRC, "cont_tasks": CONT, "arms": {a: ARMS[a] for a in names},
        "code_sha256": {f"src/{n}": SH.file_sha256(ROOT / "src" / n) for n in
                        ("escape_ee_0917.py", "swap_ee_0917.py", "respdyn_ee_0917.py", "resp_ee_0917.py",
                         "mucap_el_run_0916.py", "mucap_el_0916.py", "l2cap_ee_0917.py", "pmnist_0905.py",
                         "pmnist_rlmnist_0906.py", "elu_growth_0909.py", "shell_l2_rlmnist_0913.py")},
        "data_sha256": mnist.sha256, "prefix_info": pre["info"],
        "prefix_hash_match_resp_ee": sum(bool(r["hash_match_resp_ee"]) for r in prows),
        "seconds_prefix": t_prefix, "seconds_total": time.time() - t0,
        "peak_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    (out / "provenance.json").write_text(json.dumps(prov, indent=2, default=str))
    if progress:
        print(f"wrote {out}  ({len(prows)} prefix rows, {len(rows)} arm rows, {time.time() - t0:.1f}s)", flush=True)
    return prov


GIT_AT_START: dict = {}


def _git_state() -> dict:
    return {"hash": H.git_hash(), "dirty": SH.git_dirty(["src", "analysis/escape_ee_0917"])}


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
