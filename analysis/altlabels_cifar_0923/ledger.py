#!/usr/bin/env python3
"""ledger.py -- altlabels_cifar_0923 §3.  POST-HOC, zero new training, CPU only.

Reads the arms' W1 snapshots, per_task.csv and trace/*.npz and writes, under
results/altlabels_cifar_0923/:

    ledger.csv        arm, seed, t, band : N_b, d_b, e_b, identity residual, V_b,
                      sigma_med, H, hit99, hit999, tc
    cycle.csv         arm, seed, j, band : the cycle's centre M_j, spread D_j, mean
                      energy E_{b,j} and the identity, the centre's displacement, and
                      the same-label return at lag 2 (lag 3 for the ABC arm)
    lags.csv          arm, seed, band, window : the increments' lag 1..6 correlations in
                      two conventions, the cumulative ratio r_b, and q_b(t0)
    growth.csv        arm, seed, band : the registered slope of E_{b,j} over j = 14..25,
                      its ratio to LR_iid (rho_b), the per-task slope, the log-log
                      exponent, and the levels N_b(25) / N_b(50) / sigma_med / V_top
    stop.csv          the stop chains on both axes (tasks and cumulative updates tc)
    forks_scored.csv  per (fork set, t, seed): hit99 / hit999 / hit_full per branch and
                      the A/C comparison with the spec's censoring rules
    forks_bands.csv   the band ledger of every fork bundle's stop and end snapshot
    trace_phases.csv  per (arm, seed, task): delta n1 over shock / fit / hit999+500 /
                      remainder, and hit99 / hit999 / hit_full recomputed from the counts

The band basis is band_ledger.build_basis (a copy of the 0922 ledger's, sha256 in that
file's header): per (seed, cond) the 1199 PCA directions of the seed's own 1200 images
plus mu_perp, with bands top 0-10 / mid1 10-100 / mid2 100-439 / low 439-1199 /
mu 1199-1200 / comp = the 1872-dim orthogonal complement / all = the whole 3072.

Nothing is written inside an arm's directory.

Usage:
  ledger.py [--arms a,b,...] [--seeds 0-9] [--out DIR] [--threads 8] [--no-cache]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

NT = str(min(8, int(os.environ.get("ALTLAB_THREADS", "8"))))
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, NT)
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True

import numpy as np                                                       # noqa: E402
import pandas as pd                                                      # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import band_ledger as BL                                                 # noqa: E402

REPO = HERE.parents[1]
EXPERIMENT = "altlabels_cifar_0923"
OUT_DEFAULT = REPO / "results" / EXPERIMENT
EXT_ROOT = Path("/home/issan/Projects/obsidian-research-data/rlcifar_mlp_battle_0918"
                "/results/rlcifar_mlp_battle_0918")
CACHE = Path(os.environ.get(
    "ALTLAB_CACHE",
    "/tmp/claude-1000/-home-issan-Projects-claude/cc0afbce-b1e9-456c-95f4-43f5a640c220"
    "/scratchpad/opus_altlabels_analysis/basis_cache"))

NDIM, NPC, NSPAN = BL.NDIM, BL.NPC, BL.NSPAN
EIGBANDS = BL.EIGBANDS                       # (name, lo, hi) over the 1199 PCA directions
BANDS = BL.BANDS                             # + mu
BANDNAMES = BL.BANDNAMES                     # + comp, all
BAND_RANGE = {n: (a, b) for n, a, b in BANDS}
EIGNAMES = tuple(n for n, _, _ in EIGBANDS)

IDENTITY_RTOL = 1e-6                         # S7: |ΔN - d - e| <= rtol * max(|ΔN|, |N|)
CENSOR = 30000                               # a fork bundle that never hits is right-censored here


# ==========================================================================
# arms
# ==========================================================================

@dataclass
class Arm:
    name: str
    act: str                     # the engine's arm name: LR / SNA
    cond: str
    tasks: int                   # the schedule's task count (50, or 12 for ABC)
    cycle: int                   # complete-cycle length: 2 (ABAB / IID / AAAA), 3 (ABC)
    layout: str                  # "run" (one dir, R slots) | "chain" (seed<s>/) | "ext"
    root: Path = field(default=None)
    seeds: tuple = tuple(range(10))
    report_only: bool = False    # external reference: different layout, no bit match assumed
    schedule: str = ""

    # --- the windows this arm's statistics use (inclusive task bounds)
    @property
    def windows(self) -> dict:
        if self.tasks >= 50:
            return {"early": (5, 14), "late": (26, 50), "all": (2, 50)}
        return {"early": (2, self.tasks // 2), "late": (4, self.tasks), "all": (2, self.tasks)}

    @property
    def cum_window(self) -> tuple:
        return (6, 50) if self.tasks >= 50 else (4, self.tasks)

    @property
    def growth_tasks(self) -> tuple:
        """The registered primary window: complete cycles inside these task bounds."""
        return (27, 50) if self.tasks >= 50 else (None, None)

    @property
    def q_points(self) -> dict:
        if self.tasks < 50:
            return {}
        return {"q20": (20, 50), "qA": (19, 49), "qB": (20, 50)}

    def slot_dir(self, seed: int) -> Path:
        return self.root / f"seed{seed}" if self.layout == "chain" else self.root

    def snap(self, seed: int, t: int) -> Path:
        return self.slot_dir(seed) / "snap" / f"{self.act}_{self.cond}_seed{seed}" / f"t{t:02d}.npz"

    def per_task(self, seed: int) -> Path:
        return self.slot_dir(seed) / "per_task.csv"

    def trace(self, seed: int) -> Path:
        return self.slot_dir(seed) / "trace" / f"{self.act}_{self.cond}_seed{seed}.npz"

    def present(self, seed: int) -> bool:
        return self.snap(seed, 0).exists()


def arm_specs(out: Path) -> dict:
    A = {}

    def add(**kw):
        a = Arm(**kw)
        A[a.name] = a

    add(name="LR_iid", act="LR", cond="std", tasks=50, cycle=2, layout="run",
        root=out / "LR_iid", schedule="iid")
    add(name="LR_abab", act="LR", cond="std", tasks=50, cycle=2, layout="run",
        root=out / "LR_abab", schedule="abab")
    add(name="LR_aaaa", act="LR", cond="std", tasks=50, cycle=2, layout="run",
        root=out / "LR_aaaa", schedule="aaaa")
    add(name="LR_abc", act="LR", cond="std", tasks=12, cycle=3, layout="run",
        root=out / "LR_abc", schedule="abc")
    add(name="SNA_abab", act="SNA", cond="std", tasks=50, cycle=2, layout="run",
        root=out / "SNA_abab", schedule="abab")
    add(name="LR_abab_stop", act="LR", cond="std", tasks=50, cycle=2, layout="chain",
        root=out / "LR_abab_stop", seeds=tuple(range(5)), schedule="abab")
    add(name="LR_iid_stop", act="LR", cond="std", tasks=50, cycle=2, layout="chain",
        root=out / "LR_iid_stop", seeds=tuple(range(5)), schedule="iid")
    add(name="LR_iid_ext", act="LR", cond="std", tasks=50, cycle=2, layout="ext",
        root=EXT_ROOT / "LR", report_only=True, schedule="iid")
    add(name="SNA_iid_ext", act="SNA", cond="std", tasks=50, cycle=2, layout="ext",
        root=EXT_ROOT / "SNA", report_only=True, schedule="iid")
    return A


FORK_SETS = {                       # name -> (fork dir, the main run it forked from)
    "LR_abab_fork": ("LR_abab_fork", "LR_abab"),
    "LR_iid_fork": ("LR_iid_fork", "LR_iid"),
}


# ==========================================================================
# the band algebra.  A "dec" is (C, Rc): C = W Q (100, 1200), Rc = W - C Q^T.
# Q is orthonormal, so <W, W'> = <C, C'> + <Rc, Rc'> and every band is a
# coordinate block of C (or the whole of Rc).
# ==========================================================================

def decompose(W: np.ndarray, Q: np.ndarray):
    C = W @ Q
    return C, W - C @ Q.T


def bdot(name: str, A, B) -> float:
    if name == "comp":
        return float((A[1] * B[1]).sum())
    if name == "all":
        return float((A[0] * B[0]).sum()) + float((A[1] * B[1]).sum())
    a, b = BAND_RANGE[name]
    return float((A[0][:, a:b] * B[0][:, a:b]).sum())


def bnorm2(name: str, A) -> float:
    return bdot(name, A, A)


def lc(coefs, decs):
    """A linear combination of decompositions, kept in the same (C, Rc) form."""
    C = sum(c * d[0] for c, d in zip(coefs, decs))
    R = sum(c * d[1] for c, d in zip(coefs, decs))
    return C, R


def sigmas(C: np.ndarray, lam: np.ndarray, lo: int = 0, hi: int = NPC) -> np.ndarray:
    """Per unit sqrt(sum_j lam_j C_ij^2) over the PCA directions [lo, hi).

    This is exactly sd(z1_i) over the 1200 images (ddof = 1, the run's own convention),
    because the input's covariance is sum_j lam_j u_j u_j^T and W1 sees nothing else.
    """
    return np.sqrt((lam[lo:hi] * C[:, lo:hi] ** 2).sum(1))


def lower_median(v: np.ndarray) -> float:
    """torch.median's convention (the lower of the two central values).

    The REGISTERED sigma_med is numpy's (the mean of the two central values); this one
    exists only to compare against the run's trace `sig_med`, which torch produced.
    """
    return float(np.sort(np.asarray(v, float))[(len(v) - 1) // 2])


def band_variance(name: str, C: np.ndarray, lam: np.ndarray) -> float:
    """V_b.  mu_perp and comp carry no input variance, so they are exactly 0."""
    if name in ("mu", "comp"):
        return 0.0
    if name == "all":
        return float((lam * C[:, :NPC] ** 2).sum())
    a, b = BAND_RANGE[name]
    return float((lam[a:b] * C[:, a:b] ** 2).sum())


# ==========================================================================
# small statistics
# ==========================================================================

def ols(x, y):
    """slope, intercept, n, residual sd (ddof = 2).  NaN when fewer than 3 points."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 2 or np.ptp(x) == 0:
        return np.nan, np.nan, len(x), np.nan
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ coef
    sd = float(np.sqrt((r ** 2).sum() / (len(x) - 2))) if len(x) > 2 else np.nan
    return float(coef[0]), float(coef[1]), len(x), sd


def loglog(t, N):
    t, N = np.asarray(t, float), np.asarray(N, float)
    ok = np.isfinite(t) & np.isfinite(N) & (t > 0) & (N > 0)
    if ok.sum() < 3:
        return np.nan, np.nan, int(ok.sum()), np.nan, int((~ok).sum())
    s, b, n, sd = ols(np.log(t[ok]), np.log(N[ok]))
    return s, b, n, sd, int((~ok).sum())


def med(v):
    v = np.asarray([x for x in v if np.isfinite(x)], float)
    return float(np.median(v)) if v.size else np.nan


# ==========================================================================
# one slot (arm x seed)
# ==========================================================================

def load_W1(p: Path) -> np.ndarray:
    with np.load(p) as d:
        w = d["W1"]
    if w.shape != (100, NDIM):
        raise SystemExit(f"{p}: W1 shape {w.shape}")
    return w.astype(np.float64)                      # float32 -> float64 is exact


def load_W1_safe(p: Path):
    """None when the file is absent OR still being written.

    The main runs append snapshots task by task while this script reads them, and np.savez
    is not atomic, so a half-written t<NN>.npz must count as "not there yet" rather than
    crash the pass or enter the ledger as garbage.
    """
    if not p.exists():
        return None
    try:
        return load_W1(p)
    except SystemExit:
        raise
    except Exception as exc:                          # zip/EOF/pickle errors: still landing
        print(f"    (skipping {p.name}: {type(exc).__name__}, still being written?)",
              flush=True)
        return None


def read_per_task(arm: Arm, seed: int) -> dict:
    """task -> the row's hit / tc / steps columns (they only exist when the run had them)."""
    p = arm.per_task(seed)
    if not p.exists():
        return {}
    try:
        df = pd.read_csv(p)
    except Exception as exc:                          # rewritten after every task
        print(f"    (per_task.csv unreadable: {type(exc).__name__})", flush=True)
        return {}
    df = df[df["seed"] == seed] if "seed" in df.columns else df
    out = {}
    for _, r in df.iterrows():
        d = {}
        for k in ("hit99", "hit999", "tc", "steps", "stop_step",
                  "acc_at_hit_plus_500", "min_correct_after_hit", "memo_acc", "online_acc"):
            if k in df.columns and pd.notna(r[k]):
                d[k] = float(r[k])
        d["diverged"] = bool(pd.isna(r.get("acc", 0.0)))
        out[int(r["task"])] = d
    return out


def slot_arrays(arm: Arm, seed: int, Q: np.ndarray, lam: np.ndarray, diag: dict = None,
                W_of=None):
    """Everything one (arm, seed) needs, over the snapshots that are on disk.

    dec[t] = (C_t, Rc_t) and ddec[t] = the INDEPENDENT decomposition of W_t - W_{t-1},
    so the identity residual tests the projection and not just the float sum.

    `W_of(t) -> W or None` replaces the snapshot loader (synthetic_checks.py drives the
    very same code path with made-up weights).
    """
    nb = len(BANDNAMES)
    dec, ddec = {}, {}
    get = W_of or (lambda t: load_W1_safe(arm.snap(seed, t)))
    Wc = {}
    T = -1
    for t in range(arm.tasks + 1):
        w = get(t)
        if w is None:
            break
        Wc[t] = w
        T = t
    if T < 0:
        return None
    nan = lambda: np.full((T + 1, nb), np.nan)                            # noqa: E731
    N, dd, ee, res = nan(), nan(), nan(), nan()
    V = nan()
    sig_med = np.full(T + 1, np.nan)
    sig_med_lo = np.full(T + 1, np.nan)
    H = np.full(T + 1, np.nan)
    sig_band = np.full((T + 1, len(EIGNAMES)), np.nan)
    nonfinite = []
    W_p = None
    for t in range(T + 1):
        W = Wc.pop(t)
        if not np.isfinite(W).all():
            nonfinite.append(t)
        C, Rc = decompose(W, Q)
        dec[t] = (C, Rc)
        for k, nm in enumerate(BANDNAMES):
            N[t, k] = bnorm2(nm, dec[t])
            V[t, k] = band_variance(nm, C, lam)
        s = sigmas(C, lam)
        sig_med[t] = float(np.median(s))              # REGISTERED: numpy (midpoint) median
        sig_med_lo[t] = lower_median(s)               # torch's convention, for the trace check
        H[t] = float((s ** 2).mean() / np.median(s) ** 2) if np.median(s) > 0 else np.nan
        for k, (nm, a, b) in enumerate(EIGBANDS):
            sig_band[t, k] = float(np.median(sigmas(C, lam, a, b)))
        if t >= 1:
            dW = W - W_p                            # exact in float64
            ddec[t] = decompose(dW, Q)
            for k, nm in enumerate(BANDNAMES):
                dd[t, k] = bnorm2(nm, ddec[t])
                ee[t, k] = 2.0 * bdot(nm, dec[t - 1], ddec[t])
                res[t, k] = N[t, k] - N[t - 1, k] - dd[t, k] - ee[t, k]
        W_p = W
    return dict(T=T, dec=dec, ddec=ddec, N=N, d=dd, e=ee, res=res, V=V,
                sig_med=sig_med, sig_med_lo=sig_med_lo, H=H, sig_band=sig_band,
                nonfinite=nonfinite)


# --------------------------------------------------------------------------
# ledger.csv
# --------------------------------------------------------------------------

def rows_ledger(arm: Arm, seed: int, S: dict, pt: dict, tph: dict) -> list:
    """tph: task -> the trace-derived row (its hit99/hit999 are the ones that go in, because
    the run's per_task.csv copies are the engine's; both are kept so they can be compared)."""
    out = []
    for t in range(S["T"] + 1):
        info = pt.get(t, {})
        tr = tph.get(t, {})
        h99 = tr.get("hit99", info.get("hit99", np.nan))
        h999 = tr.get("hit999", info.get("hit999", np.nan))
        for k, nm in enumerate(BANDNAMES):
            rel = abs(S["res"][t, k]) / max(abs(S["N"][t, k]), abs(S["N"][t - 1, k]) if t else 0.0,
                                            1e-300) if t >= 1 else np.nan
            out.append({
                "arm": arm.name, "seed": seed, "t": t, "band": nm,
                "N": S["N"][t, k], "d": S["d"][t, k], "e": S["e"][t, k],
                "dN": (S["N"][t, k] - S["N"][t - 1, k]) if t >= 1 else np.nan,
                "identity_residual": S["res"][t, k], "identity_residual_rel": rel,
                "V": S["V"][t, k],
                "sigma_med": S["sig_med"][t],
                "sigma_med_lower": S["sig_med_lo"][t],
                "sigma_med_band": (S["sig_band"][t, EIGNAMES.index(nm)]
                                   if nm in EIGNAMES else (0.0 if nm in ("mu", "comp") else np.nan)),
                "H": S["H"][t],
                "hit99": h99, "hit999": h999, "hit_full": tr.get("hit_full", np.nan),
                "hit99_per_task": info.get("hit99", np.nan),
                "hit999_per_task": info.get("hit999", np.nan),
                "hit_agrees_with_per_task": (
                    bool(info.get("hit99", h99) == h99 and info.get("hit999", h999) == h999)
                    if tr and ("hit99" in info) else None),
                "tc": tr.get("tc_end", info.get("tc", np.nan)),
                "tc_per_task": info.get("tc", np.nan),
                "steps": tr.get("task_steps", info.get("steps", np.nan)),
                "stop_step": info.get("stop_step", np.nan),
                "status": ("nonfinite_W1" if t in S["nonfinite"] else
                           "diverged" if info.get("diverged") else "ok"),
            })
    return out


# --------------------------------------------------------------------------
# cycle.csv
# --------------------------------------------------------------------------

def cycles(arm: Arm, T: int) -> list:
    """Complete cycles [(j, [tasks...]), ...] inside 1..T."""
    out, j = [], 1
    while True:
        ts = [(j - 1) * arm.cycle + i + 1 for i in range(arm.cycle)]
        if ts[-1] > min(T, arm.tasks):
            break
        out.append((j, ts))
        j += 1
    return out


def rows_cycle(arm: Arm, seed: int, S: dict) -> list:
    dec, N, T, L = S["dec"], S["N"], S["T"], arm.cycle
    cyc = cycles(arm, T)
    prevM = {}
    out = []
    for j, ts in cyc:
        M = lc([1.0 / L] * L, [dec[t] for t in ts])
        for k, nm in enumerate(BANDNAMES):
            E = float(np.mean([N[t, k] for t in ts]))
            m2 = bnorm2(nm, M)
            dsq = float(np.mean([bnorm2(nm, lc([1.0, -1.0], [dec[t], M])) for t in ts]))
            diam = (bnorm2(nm, lc([0.5, -0.5], [dec[ts[1]], dec[ts[0]]])) if L == 2 else np.nan)
            r = {"arm": arm.name, "seed": seed, "j": j, "band": nm,
                 "cycle_len": L, "tasks": "+".join(str(t) for t in ts),
                 "x_j": float(np.mean(ts)),
                 "E": E, "M_sq": m2, "D_sq": dsq, "D_sq_diameter": diam,
                 "E_identity_residual": E - m2 - dsq,
                 "center_disp": (bnorm2(nm, lc([1.0, -1.0], [M, prevM[k]]))
                                 if k in prevM else np.nan)}
            # the same-label return, one column group per position in the cycle
            for i, t in enumerate(ts):
                tag = ("odd", "even", "third")[i] if L <= 3 else f"p{i}"
                tp = t - L
                if tp >= 0:
                    a = bdot(nm, dec[t], dec[tp])
                    nprev, ncur = N[tp, k], N[t, k]
                    coef = a / nprev if nprev > 0 else np.nan
                    r[f"ret_{tag}_t"] = t
                    r[f"ret_{tag}"] = bnorm2(nm, lc([1.0, -1.0], [dec[t], dec[tp]]))
                    r[f"a_{tag}"] = coef
                    r[f"resid_{tag}"] = (bnorm2(nm, lc([1.0, -coef], [dec[t], dec[tp]]))
                                         if np.isfinite(coef) else np.nan)
                    r[f"cos_{tag}"] = (a / math.sqrt(ncur * nprev)
                                       if ncur > 0 and nprev > 0 else np.nan)
                else:
                    for s in ("_t", "", "a_", "resid_", "cos_"):
                        pass
                    r[f"ret_{tag}_t"] = t
                    r[f"ret_{tag}"] = r[f"a_{tag}"] = r[f"resid_{tag}"] = r[f"cos_{tag}"] = np.nan
            r["status"] = "ok"
            out.append(r)
        prevM = {k: M for k in range(len(BANDNAMES))}
    return out


# --------------------------------------------------------------------------
# lags.csv
# --------------------------------------------------------------------------

KMAX = 6


def rows_lags(arm: Arm, seed: int, S: dict) -> list:
    ddec, dec, N, T = S["ddec"], S["dec"], S["N"], S["T"]
    out = []
    clo, chi = arm.cum_window
    chi = min(chi, T)
    for k, nm in enumerate(BANDNAMES):
        # ---- the cumulative ratio r_b over the fixed window
        ts = [t for t in range(clo, chi + 1) if t in ddec]
        if ts:
            tot = lc([1.0] * len(ts), [ddec[t] for t in ts])
            den = float(sum(bnorm2(nm, ddec[t]) for t in ts))
            r_b = bnorm2(nm, tot) / den if den > 0 else np.nan
            r_status = "ok" if den > 0 else "denom_nonpositive"
            if chi < arm.cum_window[1]:
                r_status = "incomplete_window"
        else:
            r_b, r_status = np.nan, "incomplete_window"
        # ---- q_b(t0)
        qs = {}
        for tag, (t0, t1) in arm.q_points.items():
            if t0 <= T and t1 <= T and N[t0, k] > 0:
                qs[tag] = bdot(nm, lc([1.0, -1.0], [dec[t1], dec[t0]]), dec[t0]) / N[t0, k]
            else:
                qs[tag] = np.nan
        for wname, (lo, hi) in arm.windows.items():
            hi_eff = min(hi, T)
            r = {"arm": arm.name, "seed": seed, "band": nm, "window": wname,
                 "window_lo": lo, "window_hi": hi, "window_hi_used": hi_eff,
                 "complete": bool(hi_eff >= hi)}
            idx = [t for t in range(lo, hi_eff + 1) if t in ddec]
            r["n_increments"] = len(idx)
            for kk in range(1, KMAX + 1):
                pairs = [(t, t - kk) for t in idx if (t - kk) in idx]
                cos_v, num, den_t, den_p = [], 0.0, 0.0, 0.0
                for t, tp in pairs:
                    a = bdot(nm, ddec[t], ddec[tp])
                    na, nb_ = bnorm2(nm, ddec[t]), bnorm2(nm, ddec[tp])
                    if na > 0 and nb_ > 0:
                        cos_v.append(a / math.sqrt(na * nb_))
                    num += a
                    den_t += na
                    den_p += nb_
                r[f"C{kk}_cos"] = float(np.mean(cos_v)) if cos_v else np.nan
                r[f"C{kk}_energy"] = num / den_t if den_t > 0 else np.nan
                r[f"C{kk}_energy_sym"] = (num / math.sqrt(den_t * den_p)
                                          if den_t > 0 and den_p > 0 else np.nan)
                r[f"n_pairs_{kk}"] = len(pairs)
                r[f"n_cos_{kk}"] = len(cos_v)
            for tag in ("cos", "energy", "energy_sym"):
                c1, c2, c3 = (r[f"C{i}_{tag}"] for i in (1, 2, 3))
                r[f"C2_minus_C13_{tag}"] = c2 - 0.5 * (c1 + c3)
                r[f"C3_minus_C2_{tag}"] = c3 - c2
            r["even_odd_pattern_cos"] = ";".join(
                "nan" if not np.isfinite(r[f"C{i}_cos"]) else f"{r[f'C{i}_cos']:+.4f}"
                for i in range(1, KMAX + 1))
            r["r_b"] = r_b
            r["r_window"] = f"t{clo}-{arm.cum_window[1]}"
            r["r_status"] = r_status
            for tag in ("q20", "qA", "qB"):
                r[tag] = qs.get(tag, np.nan)
            r["q_points"] = json.dumps({k2: list(v) for k2, v in arm.q_points.items()})
            r["status"] = "ok" if r["complete"] else "incomplete_window"
            out.append(r)
    return out


# --------------------------------------------------------------------------
# growth.csv
# --------------------------------------------------------------------------

def rows_growth(arm: Arm, seed: int, S: dict, cyc_rows: list) -> list:
    N, T = S["N"], S["T"]
    lo_g, hi_g = arm.growth_tasks
    by_band = {}
    for r in cyc_rows:
        by_band.setdefault(r["band"], []).append(r)
    clo, chi = arm.cum_window
    out = []
    for k, nm in enumerate(BANDNAMES):
        rows = sorted(by_band.get(nm, []), key=lambda q: q["j"])
        if lo_g is None:
            sel, complete = [], False
        else:
            sel = [q for q in rows
                   if min(int(x) for x in q["tasks"].split("+")) >= lo_g
                   and max(int(x) for x in q["tasks"].split("+")) <= hi_g]
            n_want = (hi_g - lo_g + 1) // arm.cycle
            complete = len(sel) >= n_want
        slope, icept, n, sd = ols([q["x_j"] for q in sel], [q["E"] for q in sel]) if sel \
            else (np.nan, np.nan, 0, np.nan)
        # secondary: the per-task OLS over t26-50 (t4-12 for the ABC arm)
        s_lo, s_hi = (26, 50) if arm.tasks >= 50 else (arm.windows["late"])
        tt = [t for t in range(s_lo, min(s_hi, T) + 1)]
        slope2, _, n2, sd2 = ols(tt, [N[t, k] for t in tt]) if tt else (np.nan, np.nan, 0, np.nan)
        # log-log exponent over the cumulative window
        bt = [t for t in range(clo, min(chi, T) + 1)]
        beta, _, nb_, bsd, nbad = loglog(bt, [N[t, k] for t in bt]) if bt \
            else (np.nan, np.nan, 0, np.nan, 0)
        lv = {}
        for t in (25, 49, 50):
            lv[f"N_{t}"] = N[t, k] if t <= T else np.nan
        out.append({
            "arm": arm.name, "seed": seed, "band": nm,
            "slope_cycle": slope, "slope_cycle_intercept": icept,
            "slope_cycle_n": n, "slope_cycle_resid_sd": sd,
            "growth_window": (f"j{sel[0]['j']}-{sel[-1]['j']} (t{lo_g}-{hi_g})" if sel
                              else f"t{lo_g}-{hi_g}" if lo_g else "n/a"),
            "growth_complete": complete,
            "slope_task": slope2, "slope_task_n": n2, "slope_task_resid_sd": sd2,
            "slope_task_window": f"t{s_lo}-{s_hi}",
            "beta_loglog": beta, "beta_n": nb_, "beta_resid_sd": bsd,
            "beta_nonpositive_points": nbad, "beta_window": f"t{clo}-{chi}",
            "N_25": lv["N_25"], "N_49": lv["N_49"], "N_50": lv["N_50"],
            "sigma_med_25": S["sig_med"][25] if 25 <= T else np.nan,
            "sigma_med_50": S["sig_med"][50] if 50 <= T else np.nan,
            "H_50": S["H"][50] if 50 <= T else np.nan,
            "V_top_50": S["V"][50, BANDNAMES.index("top")] if 50 <= T else np.nan,
            "V_50": S["V"][50, k] if 50 <= T else np.nan,
            "T_present": T,
            "status": "ok" if complete else "incomplete_window",
        })
    return out


def add_rho(growth: pd.DataFrame, ref_arm: str = "LR_iid") -> pd.DataFrame:
    """rho_b = slope_cycle(arm) / slope_cycle(LR_iid), per seed, per band."""
    ref = growth[growth["arm"] == ref_arm].set_index(["seed", "band"])
    rho, st, rho2 = [], [], []
    for _, r in growth.iterrows():
        key = (r["seed"], r["band"])
        if ref_arm not in set(growth["arm"]) or key not in ref.index:
            rho.append(np.nan)
            rho2.append(np.nan)
            st.append("ref_missing")
            continue
        den = float(ref.loc[key, "slope_cycle"])
        den2 = float(ref.loc[key, "slope_task"])
        if not np.isfinite(den) or den <= 0:
            rho.append(np.nan)
            st.append("denom_nonpositive" if np.isfinite(den) else "ref_incomplete")
        else:
            rho.append(float(r["slope_cycle"]) / den)
            st.append("ok")
        rho2.append(float(r["slope_task"]) / den2 if np.isfinite(den2) and den2 > 0 else np.nan)
    growth = growth.copy()
    growth["rho"] = rho
    growth["rho_status"] = st
    growth["rho_task"] = rho2
    growth["rho_ref_arm"] = ref_arm
    return growth


# --------------------------------------------------------------------------
# stop.csv
# --------------------------------------------------------------------------

def rows_stop(arm: Arm, seed: int, S: dict, pt: dict) -> list:
    out = []
    cum = 0.0
    for t in range(1, S["T"] + 1):
        info = pt.get(t, {})
        cum = info.get("tc", cum + info.get("steps", np.nan))
        for k, nm in enumerate(BANDNAMES):
            out.append({
                "arm": arm.name, "seed": seed, "t": t, "band": nm,
                "steps": info.get("steps", np.nan), "hit99": info.get("hit99", np.nan),
                "hit999": info.get("hit999", np.nan), "stop_step": info.get("stop_step", np.nan),
                "tc": info.get("tc", np.nan), "tc_cum_check": cum,
                "N": S["N"][t, k], "V": S["V"][t, k],
                "sigma_med": S["sig_med"][t], "H": S["H"][t],
                "sigma_med_band": (S["sig_band"][t, EIGNAMES.index(nm)]
                                   if nm in EIGNAMES else (0.0 if nm in ("mu", "comp") else np.nan)),
                "memo_acc": info.get("memo_acc", np.nan),
                "online_acc": info.get("online_acc", np.nan),
                "status": "diverged" if info.get("diverged") else "ok",
            })
    return out


# ==========================================================================
# trace_phases.csv
# ==========================================================================

SHOCK_END = 200
HIT_EXTRA = 500
NEED99, NEED999, NEED_FULL = 1188, 1199, 1200


def read_prov(d: Path) -> dict:
    p = d / "provenance.json"
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def trace_tables(tr: Path, prov: dict, spt_default: int = 30000) -> dict:
    """task -> the per-task trace, with the `tc` column REPAIRED.

    The engine writes one `tc` per row but fills every row of a task with the task's FINAL
    tc, so the column is wrong for every non-terminal row.  The repair is

        tc(t, step) = tc_at_task_end[t] - steps(t) + step,
        steps(t)    = tc_at_task_end[t] - tc_at_task_end[t-1]   (the trace's own last step
                                                                 when t-1 is not recorded),

    and for an unstopped full run this must give tc(t, 0) = steps_per_task * (t - 1).
    """
    if not tr.exists():
        return {}
    try:
        with np.load(tr) as D:
            d = {k: D[k] for k in D.files}
    except Exception as exc:                          # rewritten after every task
        print(f"    (trace unreadable: {type(exc).__name__})", flush=True)
        return {}
    tce = {int(k): int(v) for k, v in (prov.get("tc_at_task_end") or {}).items()}
    spt = int(prov.get("steps_per_task", spt_default))
    stopped = prov.get("stop") is not None
    out = {}
    for t in sorted(set(int(x) for x in d["task"].tolist())):
        m = d["task"] == t
        o = np.argsort(d["step"][m].astype(np.int64), kind="stable")
        rec = {k: d[k][m][o] for k in d}
        step = rec["step"].astype(np.int64)
        last = int(step[-1])
        tc_end = tce.get(t, int(rec["tc"][-1]))
        # the task's own step count: the tc difference when the previous task is recorded,
        # otherwise the trace's last step (the eval grid always reaches the task's last step,
        # because a task ends either at steps_per_task or at a stop step, both multiples of 100)
        steps = (tc_end - tce[t - 1]) if (t - 1) in tce else last
        tc0 = tc_end - steps
        rec["tc_repaired"] = tc0 + step
        rec["_meta"] = {
            "last_step": last, "steps": int(steps), "tc_start": int(tc0), "tc_end": int(tc_end),
            "steps_vs_last_step_ok": bool(steps == last),
            "tc0_expected": (spt * (t - 1)) if not stopped else None,
            "tc0_ok": (bool(tc0 == spt * (t - 1)) if not stopped and prov.get("restore") is None
                       else None),
        }
        out[t] = rec
    return out


def trace_phase_rows(arm: Arm, seed: int, tt: dict | None = None) -> list:
    tt = trace_tables(arm.trace(seed), read_prov(arm.slot_dir(seed))) if tt is None else tt
    out = []
    for t, rec in sorted(tt.items()):
        step = rec["step"].astype(np.int64)
        cnt = rec["correct"].astype(np.int64)
        n1 = rec["n1"].astype(float)
        sig = rec["sig_med"].astype(float)
        tcr = rec["tc_repaired"].astype(np.int64)
        meta = rec["_meta"]
        elig = step > 0

        def first(need):
            w = np.nonzero(elig & (cnt >= need))[0]
            return int(step[w[0]]) if len(w) else -1
        h99, h999, hfull = first(NEED99), first(NEED999), first(NEED_FULL)
        at = {int(s): i for i, s in enumerate(step)}
        last = int(step[-1])

        def n1_at(s):
            return float(n1[at[int(s)]]) if int(s) in at else np.nan

        def tc_at(s):
            return int(tcr[at[int(s)]]) if int(s) in at else -1
        s0 = 0
        s1 = min(SHOCK_END, last)
        s2 = min(max(h999, s1), last) if h999 >= 0 else last
        s3 = min(max(h999 + HIT_EXTRA, s2), last) if h999 >= 0 else np.nan
        s4 = last
        # --- the counts the engine's per_task.csv gets wrong (Codex note 3)
        want = h999 + HIT_EXTRA if h999 >= 0 else -1
        at500_obs = want >= 0 and want in at
        c_at500 = int(cnt[at[want]]) if at500_obs else np.nan
        after_hit = cnt[step > h999] if h999 >= 0 else np.array([], dtype=np.int64)
        after_500 = cnt[step > want] if want >= 0 else np.array([], dtype=np.int64)
        out.append({
            "arm": arm.name, "seed": seed, "task": int(t),
            "n_evals": int(len(step)), "step_last": last,
            "task_steps": meta["steps"], "tc_start": meta["tc_start"], "tc_end": meta["tc_end"],
            "tc_start_expected_ok": meta["tc0_ok"],
            "steps_vs_last_step_ok": meta["steps_vs_last_step_ok"],
            "hit99": h99, "hit999": h999, "hit_full": hfull,
            "hit999_tc": tc_at(h999) if h999 >= 0 else -1,
            "horizon_step": last,
            "hit_full_censored": bool(hfull < 0),
            "correct_start": int(cnt[0]), "correct_end": int(cnt[-1]),
            "correct_at_hit999_plus_500": c_at500,
            "correct_at_hit999_plus_500_observed": bool(at500_obs),
            "min_correct_after_hit999": int(after_hit.min()) if after_hit.size else np.nan,
            "min_correct_after_hit999_plus_500": int(after_500.min()) if after_500.size else np.nan,
            "n1_start": n1_at(s0), "n1_end": n1_at(s4),
            "sigma_med_start_trace": float(sig[at[0]]) if 0 in at else np.nan,
            "sigma_med_end_trace": float(sig[at[last]]),
            "d_n1_shock": n1_at(s1) - n1_at(s0),
            "d_n1_fit": n1_at(s2) - n1_at(s1),
            "d_n1_hit500": (n1_at(s3) - n1_at(s2)) if np.isfinite(s3) else np.nan,
            "d_n1_rest": (n1_at(s4) - n1_at(s3)) if np.isfinite(s3) else np.nan,
            "phase_steps": f"{s0}/{s1}/{s2}/{s3 if np.isfinite(s3) else 'nan'}/{s4}",
            "phase_tc": f"{tc_at(s0)}/{tc_at(s1)}/{tc_at(s2)}/"
                        f"{tc_at(s3) if np.isfinite(s3) else 'nan'}/{tc_at(s4)}",
            "status": "ok" if h999 >= 0 else "hit999_not_reached",
        })
    return out


# ==========================================================================
# forks
# ==========================================================================

def fork_trace_hits(bundle: Path, act: str, cond: str, seed: int) -> dict:
    """hit99 / hit999 / hit_full and the OBSERVATION HORIZON, from the bundle's own trace.

    A fork bundle stops when every slot is 500 steps past its own hit999 (or at 30,000), so a
    slot that never reaches 1200/1200 is censored at this bundle's last step, which can be far
    short of 30,000 (Codex note 4).  hit999 itself can only be censored at 30,000, because a
    slot without a stop step keeps the whole bundle running to the end.
    """
    p = bundle / "trace" / f"{act}_{cond}_seed{seed}.npz"
    if not p.exists():
        return {"trace": False}
    try:
        D = np.load(p)
    except Exception:
        return {"trace": False}
    with D:
        step, cnt = D["step"].astype(np.int64), D["correct"].astype(np.int64)
        n1 = D["n1"].astype(float)
    o = np.argsort(step, kind="stable")
    step, cnt, n1 = step[o], cnt[o], n1[o]
    elig = step > 0

    def first(need):
        w = np.nonzero(elig & (cnt >= need))[0]
        return int(step[w[0]]) if len(w) else -1
    return {"trace": True, "hit99": first(NEED99), "hit999": first(NEED999),
            "hit_full": first(NEED_FULL), "horizon": int(step[-1]),
            "last_correct": int(cnt[-1]), "n1_last": float(n1[-1])}


def censored_ratio(a: float, c: float, a_cens: bool, c_cens: bool,
                   a_h: float = CENSOR, c_h: float = CENSOR):
    """The A/C ratio interval and the sign of A - C under right censoring.

    A censored branch is only known to exceed its own observation horizon (a_h / c_h), so the
    ratio becomes an interval and the sign is decided only when the intervals do not overlap.
    sign: -1 = A earlier than C, +1 = A later, 0 = tie, nan = not determined.
    """
    if not a_cens and not c_cens:
        r = a / c if c > 0 else np.nan
        return r, r, float(np.sign(a - c)), "observed"
    if not a_cens and c_cens:
        # C > c_h; A is known.  A < C iff a <= c_h (it is, or C would have hit first)
        return (0.0, (a / c_h if c_h > 0 else np.nan),
                (-1.0 if a <= c_h else np.nan), "C_censored")
    if a_cens and not c_cens:
        return ((a_h / c if c > 0 else np.nan), np.inf,
                (+1.0 if a_h >= c else np.nan), "A_censored")
    return np.nan, np.nan, np.nan, "both_censored"


def fork_rows(name: str, fdir: Path, main: Arm, Qlam: dict, seeds) -> tuple:
    """(wide rows for forks_scored.csv, band rows for forks_bands.csv)."""
    fcsv = fdir / "forks.csv"
    if not fcsv.exists():
        return [], []
    df = pd.read_csv(fcsv)
    wide, bands = [], []
    for (t, seed), g in df.groupby(["t", "seed"]):
        seed = int(seed)
        t = int(t)
        if seed not in seeds:
            continue
        r = {"fork_set": name, "src_arm": main.name, "t": t, "seed": seed}
        per = {}
        for br in ("A", "B", "C", "next"):
            gb = g[g["branch"] == br]
            if gb.empty:
                continue
            row = gb.iloc[0]
            bundle = fdir / "_bundles" / f"t{t:02d}_{br}"
            tr = fork_trace_hits(bundle, main.act, main.cond, seed)
            h999 = int(tr["hit999"]) if tr.get("trace") else int(row.get("hit999", -1))
            h99 = int(tr["hit99"]) if tr.get("trace") else int(row.get("hit99", -1))
            hfull = int(tr["hit_full"]) if tr.get("trace") else -1
            horizon = int(tr["horizon"]) if tr.get("trace") else int(
                row.get("bundle_steps", CENSOR) or CENSOR)
            # hit999 is censored at 30,000 (a slot that never hits keeps the bundle running);
            # hit_full / hit99 are censored at the bundle's own last step
            per[br] = {"hit999": (h999, CENSOR), "hit99": (h99, horizon),
                       "hit_full": (hfull, horizon)}
            r[f"hit999_{br}"] = h999
            r[f"hit99_{br}"] = h99
            r[f"hit_full_{br}"] = hfull if tr.get("trace") else np.nan
            r[f"hit_full_source_{br}"] = "bundle_trace" if tr.get("trace") else "unavailable"
            r[f"horizon_{br}"] = horizon
            r[f"last_correct_{br}"] = tr.get("last_correct", np.nan)
            r[f"stop_step_{br}"] = row.get("stop_step", np.nan)
            r[f"bundle_steps_{br}"] = row.get("bundle_steps", np.nan)
            r[f"correct_at_stop_{br}"] = row.get("correct_at_stop", np.nan)
            r[f"min_correct_after_hit_{br}"] = row.get("min_correct_after_hit", np.nan)
            r[f"memo_acc_{br}"] = row.get("memo_acc", np.nan)
            r[f"tc_{br}"] = row.get("tc", np.nan)
            r[f"parent_sha256_{br}"] = row.get("parent_sha256", "")
            # --- the band ledger at this branch's stop and end snapshots
            Q, lam = Qlam[seed]
            for slot in ("stop", "end"):
                col = f"{slot}_snap"
                rel = row.get(col, "")
                p = fdir / rel if isinstance(rel, str) and rel else None
                if p is None or not p.exists():
                    continue
                W = load_W1(p)
                C, Rc = decompose(W, Q)
                s = sigmas(C, lam)
                for nm in BANDNAMES:
                    bands.append({"fork_set": name, "t": t, "seed": seed, "branch": br,
                                  "slot": slot, "band": nm,
                                  "N": bnorm2(nm, (C, Rc)),
                                  "V": band_variance(nm, C, lam),
                                  "sigma_med": float(np.median(s)),
                                  "H": float((s ** 2).mean() / np.median(s) ** 2),
                                  "snap": str(rel), "status": "ok"})
                if slot == "stop":
                    r[f"N_all_stop_{br}"] = bnorm2("all", (C, Rc))
                    r[f"sigma_med_stop_{br}"] = float(np.median(s))
        pairs = [("A", "C", "AC"), ("next", "C", "nextC")]
        for key in ("hit999", "hit99", "hit_full"):
          for lhs, rhs, tag in pairs:
            if lhs in per and rhs in per:
                (a, ah), (c, ch) = per[lhs][key], per[rhs][key]
                ac, cc = a < 0, c < 0
                a = ah if ac else a
                c = ch if cc else c
                lo, hi, sgn, how = censored_ratio(float(a), float(c), ac, cc, float(ah), float(ch))
                r[f"ratio_{tag}_{key}_lo"] = lo
                r[f"ratio_{tag}_{key}_hi"] = hi
                r[f"ratio_{tag}_{key}"] = lo if how == "observed" else np.nan
                r[f"sign_{tag}_{key}"] = sgn
                r[f"censoring_{tag}_{key}"] = how
            else:
                for suf in ("_lo", "_hi", ""):
                    r[f"ratio_{tag}_{key}{suf}"] = np.nan
                r[f"sign_{tag}_{key}"] = np.nan
                r[f"censoring_{tag}_{key}"] = "branch_missing"
        r["status"] = ("ok" if ("A" in per and "C" in per) else
                       "iid_fork_next_vs_C" if ("next" in per and "C" in per) else
                       "incomplete_branches")
        wide.append(r)
    return wide, bands


# ==========================================================================
# main
# ==========================================================================

def git_state() -> dict:
    def q(*c):
        try:
            return subprocess.run(c, cwd=REPO, capture_output=True, text=True,
                                  timeout=20).stdout.strip()
        except Exception:
            return ""
    return {"git_hash": q("git", "rev-parse", "HEAD"),
            "git_branch": q("git", "rev-parse", "--abbrev-ref", "HEAD"),
            "git_dirty": bool(q("git", "status", "--porcelain"))}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def basis(seed: int, cond: str, cache: bool):
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"basis_{cond}_s{seed}.npz"
    if cache and f.exists():
        with np.load(f) as d:
            return d["Q"], d["lam"], json.loads(str(d["diag"]))
    Q, lam, diag = BL.build_basis(seed, cond)
    if cache:
        np.savez(f, Q=Q, lam=lam, diag=np.array(json.dumps(diag)))
    return Q, lam, diag


def parse_fork_token(tok: str):
    """NAME | NAME=SUBDIR | NAME=SUBDIR@MAINARM  ->  (name, subdir, main arm)."""
    tok = tok.strip()
    main = None
    if "@" in tok:
        tok, main = tok.split("@", 1)
    if "=" in tok:
        name, sub = tok.split("=", 1)
    elif tok in FORK_SETS:
        name, sub = tok, FORK_SETS[tok][0]
    else:
        raise SystemExit(f"unknown fork set {tok!r}; known: {','.join(FORK_SETS)} "
                         f"(or give NAME=SUBDIR@MAINARM)")
    main = main or (FORK_SETS[name][1] if name in FORK_SETS else None)
    if main is None:
        raise SystemExit(f"fork set {name!r}: say which main arm with NAME=SUBDIR@MAINARM")
    return name, sub, main


def parse_ints(s):
    out = []
    for tok in str(s).split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            a, b = tok.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(tok))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="all")
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--results", default=None, help="where the arm dirs live (default = --out)")
    ap.add_argument("--forks", default="all", help="fork sets to score, or 'none'")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--suffix", default="", help="append to every output file's stem")
    a = ap.parse_args()

    out = Path(a.out)
    res = Path(a.results) if a.results else out
    out.mkdir(parents=True, exist_ok=True)
    seeds_want = parse_ints(a.seeds)
    ARMS = arm_specs(res)
    names = list(ARMS) if a.arms == "all" else [x.strip() for x in a.arms.split(",")]
    t0 = time.time()

    present, skipped = {}, {}
    for n in names:
        if n not in ARMS:
            raise SystemExit(f"unknown arm {n!r}; known: {','.join(ARMS)}")
        arm = ARMS[n]
        sd = [s for s in arm.seeds if s in seeds_want and arm.present(s)]
        if sd:
            present[n] = (arm, sd)
        else:
            skipped[n] = "no snapshots on disk"
    print(f"arms present: {', '.join(f'{k}({len(v[1])} seeds)' for k, v in present.items())}",
          flush=True)
    if skipped:
        print(f"arms absent : {', '.join(skipped)}", flush=True)

    L, CY, LG, GR, ST, TP = [], [], [], [], [], []
    diag_rows, basis_diag = [], {}
    conds = sorted({arm.cond for arm, _ in present.values()})
    all_seeds = sorted({s for _, sd in present.values() for s in sd})
    Qlam = {}
    for cond in conds:
        for seed in all_seeds:
            tb = time.time()
            Q, lam, bdiag = basis(seed, cond, not a.no_cache)
            basis_diag[f"{cond}_s{seed}"] = bdiag
            Qlam[seed] = (Q, lam)
            print(f"[{time.time()-t0:7.1f}s] basis {cond} seed{seed} "
                  f"orth {bdiag['orthonormality_max_err']:.1e} ({time.time()-tb:.1f}s)", flush=True)
            for n, (arm, sd) in present.items():
                if arm.cond != cond or seed not in sd:
                    continue
                ts = time.time()
                S = slot_arrays(arm, seed, Q, lam, bdiag)
                if S is None:
                    continue
                pt = read_per_task(arm, seed)
                prov_slot = read_prov(arm.slot_dir(seed))
                tt = trace_tables(arm.trace(seed), prov_slot)
                tpr = trace_phase_rows(arm, seed, tt)
                for q in tpr:                       # the REGISTERED sigma_med, from the snapshots
                    t_ = q["task"]
                    q["sigma_med_start_snap"] = S["sig_med"][t_ - 1] if t_ - 1 <= S["T"] else np.nan
                    q["sigma_med_end_snap"] = S["sig_med"][t_] if t_ <= S["T"] else np.nan
                    q["sigma_med_end_lower_snap"] = (S["sig_med_lo"][t_] if t_ <= S["T"]
                                                     else np.nan)
                tph = {q["task"]: q for q in tpr}
                L += rows_ledger(arm, seed, S, pt, tph)
                cy = rows_cycle(arm, seed, S)
                CY += cy
                LG += rows_lags(arm, seed, S)
                GR += rows_growth(arm, seed, S, cy)
                if arm.layout == "chain":
                    ST += rows_stop(arm, seed, S, pt)
                TP += tpr
                # --- S7 diagnostics for this slot
                ka = BANDNAMES.index("all")
                relres = np.abs(S["res"][1:]) / np.maximum(np.abs(S["N"][1:]), 1e-300)
                bandsum = np.nansum(S["N"][:, :len(BANDNAMES) - 1], 1) - S["N"][:, ka]
                vsum = (np.nansum(S["V"][:, [BANDNAMES.index(x) for x in EIGNAMES]], 1)
                        - S["V"][:, ka])
                # sigma against the run's own trace: an independent check that this basis is the
                # run's input.  The trace's sig_med is torch's LOWER-middle median, so the
                # comparison uses sig_med_lower; the registered sigma_med stays numpy's.
                errs = []
                for t, rec in tt.items():
                    if t > S["T"]:
                        continue
                    j = int(np.argmax(rec["step"]))
                    errs.append(abs(float(rec["sig_med"][j]) - S["sig_med_lo"][t]))
                sig_err = float(max(errs)) if errs else np.nan
                tc_bad = [q["task"] for q in tpr if q["tc_start_expected_ok"] is False]
                st_bad = [q["task"] for q in tpr if not q["steps_vs_last_step_ok"]]
                diag_rows.append({
                    "arm": n, "seed": seed, "T_present": S["T"],
                    "max_identity_abs": float(np.nanmax(np.abs(S["res"]))),
                    "max_identity_rel": float(np.nanmax(relres)) if relres.size else np.nan,
                    "max_bandsum_err": float(np.nanmax(np.abs(bandsum))),
                    "max_bandsum_rel": float(np.nanmax(np.abs(bandsum)
                                                       / np.maximum(S["N"][:, ka], 1e-300))),
                    "max_V_sum_err": float(np.nanmax(np.abs(vsum))),
                    "max_V_sum_rel": float(np.nanmax(np.abs(vsum)
                                                     / np.maximum(S["V"][:, ka], 1e-300))),
                    "V_mu_max": float(np.nanmax(np.abs(S["V"][:, BANDNAMES.index("mu")]))),
                    "V_comp_max": float(np.nanmax(np.abs(S["V"][:, BANDNAMES.index("comp")]))),
                    "sigma_med_vs_trace_max_abs": sig_err,
                    "tc_repair_bad_tasks": ";".join(map(str, tc_bad)),
                    "steps_vs_last_step_bad_tasks": ";".join(map(str, st_bad)),
                    "nonfinite_snapshots": ";".join(map(str, S["nonfinite"])),
                })
                print(f"[{time.time()-t0:7.1f}s]   {n:14s} seed{seed} T={S['T']:2d} "
                      f"res {diag_rows[-1]['max_identity_rel']:.1e} "
                      f"sig-vs-trace {sig_err:.2e} ({time.time()-ts:.1f}s)", flush=True)
                del S
            del Q, lam
            Qlam.pop(seed, None)

    # ---- forks (they need the basis again; done in one pass at the end)
    FW, FB = [], []
    fsets = ([] if a.forks == "none" else
             [(k, *v) for k, v in FORK_SETS.items()] if a.forks == "all" else
             [parse_fork_token(x) for x in a.forks.split(",") if x.strip()])
    for fname, sub, mainname in fsets:
        fdir = res / sub
        if not (fdir / "forks.csv").exists():
            skipped[fname] = "no forks.csv"
            continue
        mainarm = ARMS[mainname]
        need = sorted(set(pd.read_csv(fdir / "forks.csv")["seed"].astype(int)) & set(seeds_want))
        QL = {}
        for s in need:
            QL[s] = basis(s, mainarm.cond, not a.no_cache)[:2]
        w, b = fork_rows(fname, fdir, mainarm, QL, set(need))
        FW += w
        FB += b
        print(f"[{time.time()-t0:7.1f}s] forks {fname}: {len(w)} (t, seed) rows", flush=True)
        del QL

    # ---- write
    sfx = a.suffix
    def W(rows, name, cols_first=()):
        p = out / f"{name}{sfx}.csv"
        df = pd.DataFrame(rows)
        if len(df):
            first = [c for c in cols_first if c in df.columns]
            df = df[first + [c for c in df.columns if c not in first]]
        df.to_csv(p, index=False, float_format="%.10g")
        print(f"wrote {p} ({len(df)} rows)")
        return df

    W(L, "ledger", ("arm", "seed", "t", "band"))
    W(CY, "cycle", ("arm", "seed", "j", "band"))
    W(LG, "lags", ("arm", "seed", "band", "window"))
    gdf = pd.DataFrame(GR)
    if len(gdf):
        gdf = add_rho(gdf)
    W(gdf.to_dict("records") if len(gdf) else [], "growth", ("arm", "seed", "band"))
    W(ST, "stop", ("arm", "seed", "t", "band"))
    W(FW, "forks_scored", ("fork_set", "t", "seed"))
    W(FB, "forks_bands", ("fork_set", "t", "seed", "branch", "slot", "band"))
    W(TP, "trace_phases", ("arm", "seed", "task"))

    d = pd.DataFrame(diag_rows)
    prov = {
        "run_id": EXPERIMENT, "stage": "ledger", "written": time.strftime("%FT%T%z"),
        **git_state(),
        "band_ledger_copy": str(HERE / "band_ledger.py"),
        "band_ledger_copy_sha256": sha256_file(HERE / "band_ledger.py"),
        "band_ledger_source": "/home/issan/Projects/obsidian-research-data/"
                              "rl_ledger_posthoc_0922/band_ledger.py",
        "band_ledger_source_sha256":
            "3a3978b607a95222b1067a00bc53e1ce3cfab84e04219a393e052f31d06165c5",
        "bands": {n: list(BAND_RANGE[n]) for n in BAND_RANGE} | {"comp": "3072 - 1200 dims",
                                                                 "all": "3072"},
        "arms": {n: {"act": arm.act, "cond": arm.cond, "tasks": arm.tasks,
                     "cycle": arm.cycle, "schedule": arm.schedule, "root": str(arm.root),
                     "seeds": sd, "report_only": arm.report_only,
                     "windows": arm.windows, "cum_window": list(arm.cum_window),
                     "growth_tasks": list(arm.growth_tasks)}
                 for n, (arm, sd) in present.items()},
        "arms_absent": skipped,
        "fork_sets": {k: str(res / v[0]) for k, v in FORK_SETS.items()},
        "censor_at": CENSOR, "identity_rtol": IDENTITY_RTOL,
        "checks": {
            "max_identity_rel": float(d["max_identity_rel"].max()) if len(d) else None,
            "max_identity_abs": float(d["max_identity_abs"].max()) if len(d) else None,
            "identity_ok": bool(len(d) and d["max_identity_rel"].max() <= IDENTITY_RTOL),
            "max_bandsum_rel": float(d["max_bandsum_rel"].max()) if len(d) else None,
            "max_V_sum_rel": float(d["max_V_sum_rel"].max()) if len(d) else None,
            "V_mu_max": float(d["V_mu_max"].max()) if len(d) else None,
            "V_comp_max": float(d["V_comp_max"].max()) if len(d) else None,
            "sigma_med_vs_trace_max_abs":
                float(d["sigma_med_vs_trace_max_abs"].max()) if len(d) else None,
            "tc_repair_ok": bool(len(d) and (d["tc_repair_bad_tasks"] == "").all()),
            "steps_vs_last_step_ok": bool(len(d) and (d["steps_vs_last_step_bad_tasks"] == "").all()),
        },
        "conventions": {
            "sigma_med": "numpy median (midpoint) of sd(z1_i) over the 100 units, from W1",
            "sigma_med_lower": "torch's lower-middle median, only to compare with trace sig_med",
            "trace_tc": "REPAIRED: tc_at_task_end[t] - task_steps + step (the engine writes the "
                        "task's final tc on every row)",
            "hit99/hit999/hit_full": "recomputed from the trace counts (>=1188 / >=1199 / =1200), "
                                     "step > 0 only; per_task.csv's copies kept alongside",
            "correct_at_hit999_plus_500": "recomputed and flagged when that eval point was never "
                                          "reached (the engine's acc_at_hit_plus_500 falls back "
                                          "to the final count)",
            "fork_censoring": "hit999 at 30,000; hit99 / hit_full at the bundle's own last step",
        },
        "basis_diag": basis_diag,
        "threads": NT, "numpy": np.__version__, "pandas": pd.__version__,
        "wall_clock_s": time.time() - t0,
    }
    (out / f"ledger_provenance{sfx}.json").write_text(json.dumps(prov, indent=1, default=str))
    d.to_csv(out / f"ledger_diagnostics{sfx}.csv", index=False, float_format="%.10g")
    print(json.dumps(prov["checks"], indent=1))
    print(f"total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
