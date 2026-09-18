#!/usr/bin/env python3
"""H7: move the transplant host's second-layer derivative carriers directly (RL ELU->ELU).

    OMP_NUM_THREADS=1 python3 src/neffdir_ee_0918.py --seed 30 --out results/neffdir_ee_0918/runs/s30

specs/spec_neffdir_ee_0918.md (registered at PREREG_COMMIT).  escneff_ee_0917 held the healthy t2 host's
growth, and respdyn's remainder (S2dyn_10r above the response floor S2u30r) fell in proportion to nbar_neff,
the in-task effective number of images that carry the host's second-layer training derivative.  What moved
there was growth, and nbar_neff also carries the result of learning.  Here the carriers are moved directly:

    fixed q   drop: a fixed random share 1 - q of the (image, unit) pairs never passes the second layer's
              derivative; the kept pairs pass it times 1/q (the expected gradient is unchanged)
    step q    the same share dropped afresh at every update (the same noise, no pair lost for good)
    top r     add: a fixed random share r of the pairs answers with the moving field lifted so that the
              pair sits at the shadow unit's current top preactivation (plus the host's own movement)
    felu      the second layer's bracket computed with F.elu: the same forward values (to one float32 step
              2^-24 at the host kernel's truncation point), but the derivative is exp(z) where the host's
              expm1 kernel gives exactly 0 below -16.64 (the exact zeros go, the effective number barely
              moves)

calibration (seed 44, unregistered) showed that a fixed mask is partly undone within the task: the host
grows more of its own carriers (compensation), most of it after the first 1500 updates.  So the primary
comparison adds carriers with growth held (S2dyn_c12_a02 vs S2dyn_c12); the drops are read over the whole
task and over the first EARLY updates, and the compensation itself is a registered quantity.

The branch-point logits are the natural t2 net's bit for bit in every arm: fixed / step change the
backward only (u.detach() + s * (u - u.detach()) is u exactly), top changes the field and the anchor with
it, felu changes the bracket's function (the bracket is zero at the branch point).  The prefix,
the shadow that follows N10, the per-update loop, the c12 hold and the probes are imported unchanged
(escape_ee_0917, swap_ee_0917, respdyn_ee_0917, resp_ee_0917).
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
from src import pmnist_rlmnist_0906 as RL        # labels; not touched
from src import shell_l2_rlmnist_0913 as SH      # state_sha256, file_sha256, git_dirty; not touched
from src import mucap_el_run_0916 as EL          # forward2; not touched
from src import resp_ee_0917 as RE               # prefix, anchored forward, unit summaries; not touched
from src import respdyn_ee_0917 as RD            # probe, z2_full, gen_from; not touched
from src import swap_ee_0917 as SW               # CapStepper, CapDynField, build, _csv_rows; not touched
from src import escape_ee_0917 as ES             # Hold, prefix; not touched

EXPERIMENT = "neffdir_ee_0918"
SPEC = "specs/spec_neffdir_ee_0918.md"
PREREG_COMMIT = "b52bb8930881700d969590a45285d4b78bd5ad46"   # pushed registration before any registered arm ran
ACT = RE.ACT
EPOCHS = RE.EPOCHS
SPE = RE.SPE
CONT = RE.CONT
N_IMG = RE.N_IMG
BRANCH, SRC = ES.BRANCH, ES.SRC                  # 2, 10
PREFIX_T = ES.PREFIX_T                           # 12
PROBE_EVERY = RD.PROBE_EVERY
EARLY = 1500                                     # the early window: updates 0..1499, probes s = 0, 750, 1500
SEEDS = tuple(range(30, 40))
MASK_SALT = 918_001                              # the per-seed uniform tables U (drop) and V (add)
STEP_SALT = 918_002                              # the per-update draws of the step mask
ROOT = Path(__file__).resolve().parents[1]
RECORDS = {"resp_ee": ROOT / "results" / "resp_ee_0917" / "runs",
           "respdyn": ROOT / "results" / "respdyn_ee_0917" / "runs",
           "escape": ROOT / "results" / "escape_ee_0917" / "runs"}
ARANGE = torch.arange(N_IMG)


def _arm(kind=None, delta=None, hold="none", carrier=None, q=None, r=None):
    return {"branch": BRANCH, "layer": 2 if kind is not None else None, "kind": kind,
            "src": SRC if kind in ("field", "dyn") else None, "delta": delta, "hold": hold, "reset": True,
            "carrier": carrier, "q": q, "r": r}


ARMS: dict[str, dict] = {
    "S2dyn_10r": _arm("dyn"),                                   # respdyn's primary (none)
    "S2dyn_m50": _arm("dyn", carrier="fixed", q=0.5),           # drop
    "S2dyn_m25": _arm("dyn", carrier="fixed", q=0.25),          # drop, deeper (nested in m50)
    "S2dyn_s50": _arm("dyn", carrier="step", q=0.5),            # the same noise, no pair lost
    "S2dyn_a02": _arm("dyn", carrier="top", r=0.02),            # add
    "S2dyn_felu": _arm("dyn", carrier="felu"),                  # exact zeros removed
    "S2dyn_c12": _arm("dyn", hold="c12"),                       # escape's held host
    "S2dyn_c12_m50": _arm("dyn", hold="c12", carrier="fixed", q=0.5),   # drop with growth held
    "S2dyn_c12_a02": _arm("dyn", hold="c12", carrier="top", r=0.02),    # add with growth held (primary)
    "S2u30r": _arm("uniform", delta=30.0),                      # the response floor
    "N2r": _arm(),                                              # the natural continuation
    "N2r_m50": _arm(carrier="fixed", q=0.5),                    # drop in the healthy net (report)
}
# checks only: each must be S2dyn_10r bit for bit (the carrier path adds nothing but the carrier)
CHECK_ARMS: dict[str, dict] = {
    "S2dyn_m100": _arm("dyn", carrier="fixed", q=1.0),
    "S2dyn_s100": _arm("dyn", carrier="step", q=1.0),
    "S2dyn_a00": _arm("dyn", carrier="top", r=0.0),
}
# the arms with a record of the same computation (seed 0-9 only; the registered seeds have none)
REPRO = {"S2dyn_10r": ("respdyn", "escape"), "S2dyn_c12": ("escape",), "S2u30r": ("resp_ee", "respdyn", "escape"),
         "N2r": ("resp_ee", "respdyn", "escape")}


# --------------------------------------------------------------------------
# the carriers
# --------------------------------------------------------------------------

def tables(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """U (drop) and V (add): per-seed uniform (image x unit) tables from their own generator, so the
    fixed masks are nested in q (U < q) and the lifted pairs are independent of them (V < r)."""
    g = torch.Generator(device="cpu")
    g.manual_seed(MASK_SALT + seed)
    U = torch.rand(N_IMG, 100, generator=g)
    V = torch.rand(N_IMG, 100, generator=g)
    return U, V


def _felu(z: torch.Tensor) -> torch.Tensor:
    return F.elu(z, alpha=1.0)


def dphi_felu(z: torch.Tensor) -> torch.Tensor:
    """F.elu's own backward factor at z: exp(z) for z <= 0 and 1 above, as its kernel computes it (the
    kernel's exp and torch.exp differ by one ulp at some z, so the factor is taken from autograd itself).
    F.elu's forward value is the host ELU's to within 2^-24 (check S1c), so the felu arm changes the
    backward and, at the truncation point only, a forward value by one float32 step."""
    with torch.enable_grad():
        zz = z.detach().clone().requires_grad_(True)
        return torch.autograd.grad(F.elu(zz, alpha=1.0).sum(), zz)[0]


class Carrier:
    """The identity: phi2 is the host's ELU, gate passes u and its gradient."""
    kind = "none"
    phi2 = staticmethod(ACT.phi)
    dphi2 = staticmethod(RE.dphi_train)
    keep = None                                  # (N_IMG, 100) bool: the pairs the probe counts, or None

    def gate(self, u, ob):
        return u

    def facts(self) -> dict:
        return {}


class FixedMask(Carrier):
    kind = "fixed"

    def __init__(self, U: torch.Tensor, q: float):
        self.q = float(q)
        self.keep = U < self.q
        self.s = torch.where(self.keep, torch.tensor(1.0 / self.q), torch.tensor(0.0))

    def gate(self, u, ob):
        ud = u.detach()
        return ud + self.s[ob] * (u - ud)

    def facts(self) -> dict:
        return {"keep_share": float(self.keep.double().mean()),
                "dead_units_by_mask": int((~self.keep).all(0).sum())}


class StepMask(Carrier):
    kind = "step"

    def __init__(self, seed: int, q: float):
        self.q = float(q)
        self.g = torch.Generator(device="cpu")
        self.g.manual_seed(STEP_SALT + seed)
        self.draws = 0
        self.kept = 0
        self.total = 0

    def gate(self, u, ob):
        if not torch.is_grad_enabled():          # probes and summaries: value only, no draw
            return u
        keep = torch.rand(u.shape, generator=self.g) < self.q
        self.draws += 1
        self.kept += int(keep.sum())
        self.total += keep.numel()
        s = torch.where(keep, torch.tensor(1.0 / self.q), torch.tensor(0.0))
        ud = u.detach()
        return ud + s * (u - ud)

    def facts(self) -> dict:
        return {"step_draws": self.draws, "keep_share": self.kept / max(1, self.total)}


class FeluCarrier(Carrier):
    kind = "felu"
    phi2 = staticmethod(_felu)
    dphi2 = staticmethod(dphi_felu)


class TopDynField(SW.CapDynField):
    """The shadow N10 as in respdyn; after each refresh the pairs in S get the field that puts them at the
    shadow unit's current top preactivation: d = top_u - z2_branch (float64, then float32).  r = 0 is the
    parent's field bit for bit (torch.where with no True is its third argument)."""

    def __init__(self, arm, cks, x, d, S: torch.Tensor):
        super().__init__(arm, cks, x, d)
        self.S = S

    def refresh(self, g):
        with torch.no_grad():
            z2 = RD.z2_full(self.params, self.x)
            top = z2.amax(0).double()
            dd = torch.where(self.S, top.unsqueeze(0) - self.zbr, z2.double() - self.zbr)
            self.d.copy_(dd.float())


def make_carrier(arm: dict, seed: int, U: torch.Tensor) -> Carrier:
    c = arm["carrier"]
    if c in (None, "top"):
        return Carrier()
    if c == "fixed":
        return FixedMask(U, arm["q"])
    if c == "step":
        return StepMask(seed, arm["q"])
    if c == "felu":
        return FeluCarrier()
    raise ValueError(c)


def build(arm: dict, cks: dict, x: torch.Tensor, V: torch.Tensor):
    """-> (sh, mover): swap_ee_0917.build, with the lifted shadow for "top"."""
    if arm["carrier"] == "top":
        sh = RE.build_shift({**arm, "kind": "field"}, cks)
        return sh, TopDynField(arm, cks, x, sh[2], V < float(arm["r"]))
    sh, _, mover = SW.build(arm, cks, x)
    return sh, mover


def make_forward(sh: dict | None, car: Carrier):
    """resp_ee_0917.make_forward's layer-2 anchored forward with the carrier's phi2 and gate
    (None: the host's forward2 itself)."""
    if sh is None and car.kind == "none":
        return None
    phi2, gate = car.phi2, car.gate
    if sh is None:
        def fwd_nat(params, xb, ob):
            W1, b1, W2, b2, W3, b3 = params
            z1 = xb @ W1.T + b1
            a1 = ACT.phi(z1)
            z2 = a1 @ W2.T + b2
            a2 = gate(ACT.phi(z2), ob)
            return z1, a1, z2, a2, a2 @ W3.T + b3
        return fwd_nat
    p0, d2 = sh["p0"], sh[2]

    def fwd(params, xb, ob):
        W1, b1, W2, b2, W3, b3 = params
        z1 = xb @ W1.T + b1
        with torch.no_grad():
            z01 = xb @ p0[0].T + p0[1]
            a01 = ACT.phi(z01)
        a1 = ACT.phi(z1)
        z2 = a1 @ W2.T + b2
        e2 = d2[ob]
        with torch.no_grad():
            z02 = a01 @ p0[2].T + p0[3]
            a02 = ACT.phi(z02)
            A2 = phi2(z02 + e2)
        a2 = a02 + (gate(phi2(z2 + e2), ob) - A2)
        return z1, a1, z2, a2, a2 @ W3.T + b3

    return fwd


# --------------------------------------------------------------------------
# the probe: the registered raw n_eff (respdyn) and the arm's own ("structural") one
# --------------------------------------------------------------------------

def _neff(gt: torch.Tensor) -> torch.Tensor:
    n = gt.shape[0]
    s1 = gt.sum(0)
    return torch.where(s1 > 0, s1 * s1 / (n * gt.square().sum(0)).clamp(min=1e-300), torch.zeros_like(s1))


@torch.no_grad()
def probe(params, x, fwd, d, car: Carrier) -> dict:
    """RD.probe (neffT2 on the host's expm1 derivative of z2 + d, all images) plus neffS2: the same formula
    on the derivative the arm trains with -- the carrier's dphi2, times the fixed mask where there is one
    (a step mask drops every pair equally often, so its structural derivative is the unmasked one)."""
    base = RD.probe(params, x, fwd, d)
    _, _, z2, _, _ = RE.forward(params, x, ARANGE, fwd)
    ze32 = z2 if d is None else z2 + d
    gt = car.dphi2(ze32).double()
    if car.keep is not None:
        gt = gt * car.keep.double()
    return {**base, "neffS2": float(_neff(gt).mean()), "zeroS2": float((gt == 0).double().mean())}


# --------------------------------------------------------------------------
# one arm
# --------------------------------------------------------------------------

def run_arm(name: str, arm: dict, cks: dict, x: torch.Tensor, e1, nat_rows: dict, seed: int, U, V,
            epochs: int = EPOCHS, units: dict | None = None, traj: list | None = None,
            progress: bool = False) -> list[dict]:
    """escape_ee_0917.run_arm with a carrier (and the lifted shadow for "top")."""
    t0 = time.time()
    ck = cks[arm["branch"]]
    params = [p.detach().clone().requires_grad_(True) for p in ck["params"]]
    adam = ([torch.zeros_like(q) for q in params], [torch.zeros_like(q) for q in params], [0])
    g_lab, g_batch = RD.gen_from(ck["g_lab"]), RD.gen_from(ck["g_batch"])
    spt = SPE * epochs
    sh, mover = build(arm, cks, x, V)
    car = make_carrier(arm, seed, U)
    fwd = make_forward(sh, car)
    d = None if sh is None else sh.get(2)
    hold = None if arm["hold"] == "none" else ES.Hold(arm["hold"], ck["params"], e1)
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
                fixed = RE.build_shift({**arm, "kind": "field"}, cks)[2]
                if arm["carrier"] == "top":
                    S = mover.S
                    top = RD.z2_full(cks[SRC]["params"], x).amax(0).double()
                    want = (top.unsqueeze(0) - cks[BRANCH]["z2"].double()).float()
                    head["field0_equal_fixed"] = bool(torch.equal(sh[2][~S], fixed[~S]))
                    head["field0_top_ok"] = bool(torch.equal(sh[2][S], want[S]))
                    head["lift_share"] = float(S.double().mean())
                    head["lift_units_alive"] = float((top > RE.ZERO_Z32).double().mean())
                else:
                    head["field0_equal_fixed"] = bool(torch.equal(sh[2], fixed))
            p0 = probe(params, x, fwd, d, car)
            head["neffS2_start"], head["neffT2_start"] = p0["neffS2"], p0["neffT2"]
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
                pr.append({"s": s, **probe(params, x, fwd, d, car)})
            st.step(s)
            if mover is not None:
                mover.step(s)
        if mover is not None:
            mover.refresh(g0 + spt)
        pr.append({"s": spt, **probe(params, x, fwd, d, car)})
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
               "logits_equal_mb": (st.mb_equal if fwd is not None else True) if k == 1 else None,
               "nbar_neffS2": float(np.mean([p["neffS2"] for p in pr])),
               "early_acc": float(st.acc[:min(EARLY, spt)].double().mean()),
               "nbar_neffS2_early": float(np.mean([p["neffS2"] for p in pr if p["s"] <= EARLY])),
               "nbar_neffT2_early": float(np.mean([p["neffT2"] for p in pr if p["s"] <= EARLY])),
               "nbar_neffT2": float(np.mean([p["neffT2"] for p in pr])),
               "probe_zeroS2_mean": float(np.mean([p["zeroS2"] for p in pr])),
               "probe_zero2_mean": float(np.mean([p["zero2"] for p in pr])),
               "probe_zarm2_first": pr[0]["zarm2"], "probe_zarm2_last": pr[-1]["zarm2"],
               "climb2": pr[-1]["zarm2"] - pr[0]["zarm2"],
               "probe_effbar2_first": pr[0]["effbar2"], "probe_effbar2_last": pr[-1]["effbar2"],
               "w1norm_end": float(w1n), "w2norm_end": float(w2n),
               "b2_mean_end": float(params[3].detach().double().mean()),
               **car.facts(),
               **(hold.excess(params) if hold is not None else {}),
               **extra, **head, **RE.unit_summary(u1, "end"), "sec": time.time() - t_task}
        rows.append(row)
        if traj is not None:
            traj += [{"seed": seed, "arm": name, "k": k, **p} for p in pr]
    if progress:
        print(f"[{time.time() - t0:8.1f}s] seed={seed} arm {name}: done", flush=True)
    return rows


# --------------------------------------------------------------------------

def run_seed(seed: int, out: Path, mnist: H.Mnist, epochs: int = EPOCHS, arms=None, progress: bool = True,
             pre: dict | None = None, arm_table: dict | None = None) -> dict:
    """arm_table: ARMS (default) or {**ARMS, **CHECK_ARMS} (checks only)."""
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    git0 = dict(GIT_AT_START) or _git_state()
    table = ARMS if arm_table is None else arm_table
    names = [a for a in table if arms is None or a in arms]
    if pre is None:
        pre = ES.prefix(seed, mnist, epochs, progress)
    prows, cks, x, e1 = pre["rows"], pre["cks"], pre["x"], pre["e1"]
    units = dict(pre["units"])
    nat_rows = {r["task"]: r for r in prows}
    U, V = tables(seed)
    t_prefix = time.time() - t0
    recs = {src: {(r["arm"], int(r["k"])): r for r in SW._csv_rows(p / f"s{seed}" / "arms.csv")}
            for src, p in RECORDS.items()}
    rows, traj = [], []
    for a in names:
        got = run_arm(a, table[a], cks, x, e1, nat_rows, seed, U, V, epochs, units=units, traj=traj,
                      progress=progress)
        for r in got:
            for src in RECORDS:
                rec = recs[src].get((a, r["k"])) if src in REPRO.get(a, ()) else None
                r[f"hash_match_{src}"] = None if rec is None else r["state_sha256"] == rec["state_sha256"]
        rows += got
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
        "branch": BRANCH, "src": SRC, "cont_tasks": CONT, "arms": {a: table[a] for a in names},
        "mask_salt": MASK_SALT, "step_salt": STEP_SALT,
        "tables_sha256": {"U": hashlib.sha256(U.numpy().tobytes()).hexdigest(),
                          "V": hashlib.sha256(V.numpy().tobytes()).hexdigest()},
        "code_sha256": {f"src/{n}": SH.file_sha256(ROOT / "src" / n) for n in
                        ("neffdir_ee_0918.py", "escape_ee_0917.py", "swap_ee_0917.py", "respdyn_ee_0917.py",
                         "resp_ee_0917.py", "mucap_el_run_0916.py", "mucap_el_0916.py", "l2cap_ee_0917.py",
                         "pmnist_0905.py", "pmnist_rlmnist_0906.py", "elu_growth_0909.py",
                         "shell_l2_rlmnist_0913.py")},
        "data_sha256": mnist.sha256, "prefix_info": pre["info"],
        "prefix_hash_match_resp_ee": sum(bool(r["hash_match_resp_ee"]) for r in prows),
        "prefix_record_tasks_resp_ee": sum(r["hash_match_resp_ee"] is not None for r in prows),
        "seconds_prefix": t_prefix, "seconds_total": time.time() - t0,
        "peak_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    (out / "provenance.json").write_text(json.dumps(prov, indent=2, default=str))
    if progress:
        print(f"wrote {out}  ({len(prows)} prefix rows, {len(rows)} arm rows, {time.time() - t0:.1f}s)", flush=True)
    return prov


GIT_AT_START: dict = {}


def _git_state() -> dict:
    """Read once, when the process starts (a commit made during a long run must not relabel it)."""
    return {"hash": H.git_hash(), "dirty": SH.git_dirty(["src", "analysis/neffdir_ee_0918"])}


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
