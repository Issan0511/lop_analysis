"""pm_ledger_add_1108 ledger: the main run's first-layer band ledger for the add-on arms.

Same definitions as analysis/pm_ledger_1108/ledger.py.  The only difference: for arms with a
switch operation (W1s.npy exists), D_t = W_t - W^start_t and S_{t-1} = W^start_t - W_0
use the state after the switch operation (spec_pm_ledger_add §4).

(main-run docstring follows)
pm_ledger_1108 ledger: the first-layer band ledger of spec §4, task by task.

Rebuilds each task's inputs from the box's random streams, checks them against the
saved perm.npy and idx_sha.json (S-sha), computes the eigenbasis of each task's
centred covariance, and books per arm x seed x task x band:

  d, e, N (0922 ledger in the task's own basis), the deposit seen in the next task's
  bands, the weighted (visible) norms, and the erosion measured from the deposit side.

float64 numpy, CPU, one BLAS thread per process.

Usage:
  .venv/bin/python analysis/pm_ledger_add_1108/ledger.py --seed 0 --runs RUNS_DIR --out LEDGER_DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
import pmnist_0905 as B                                  # noqa: E402

B.DATA_DIR = Path(os.environ.get("PM_MNIST_DIR",
                                 "/home/issan/Projects/claude/proj_004_drift/data/mnist"))

N_PIX = 784
N_UNITS = 100
DIM = N_PIX * N_UNITS                                   # spec §6: 78,400
TMAX = 500
BAND_NAMES = ("top", "mid1", "mid2", "low", "comp")
NB = len(BAND_NAMES)
ARMS = ("CHB", "KKT1", "CHB_INV0", "CHB_ISO", "L2I_ADD", "KKT1_INV0", "none_INV0")
# deposit windows of spec §4(b): (t1, t2)
WINDOWS = ((6, 50), (101, 150), (251, 300), (451, 500))
LAGS = (1, 50)
EPS = float(np.finfo(np.float64).eps)
U = EPS / 2


def bands_of(rank: int):
    """Same rank cuts as the 0922 ledger: 1-10, 11-100, 101-439, 440-rank, rank+1-784."""
    if rank <= 439:
        raise SystemExit(f"rank {rank} <= 439: the band cut does not apply")
    return ((0, 10), (10, 100), (100, 439), (439, rank), (rank, N_PIX))


class Tasks:
    """Task t's covariance eigenbasis, rebuilt from the box's streams (S-sha)."""

    def __init__(self, seed: int, mnist: B.Mnist, run_dir: Path):
        self.g_perm = B.stream("perm", seed)
        self.g_data = B.stream("data", seed)
        self.mnist = mnist
        self.perms = np.load(run_dir / "perm.npy", mmap_mode="r")
        self.idx_sha = json.loads((run_dir / "idx_sha.json").read_text())
        self.x = mnist.train_x.numpy()                  # float32 (the values the net saw)
        self.t = 0

    def next(self):
        self.t += 1
        t = self.t
        perm = torch.randperm(N_PIX, generator=self.g_perm)
        idx = B.stratified_draw(self.mnist, self.g_data)
        if not np.array_equal(perm.numpy().astype(np.int16), np.asarray(self.perms[t])):
            raise SystemExit(f"S-sha: perm mismatch at task {t}")
        if hashlib.sha256(idx.numpy().tobytes()).hexdigest() != self.idx_sha[str(t)]:
            raise SystemExit(f"S-sha: idx mismatch at task {t}")
        X = self.x[idx.numpy()].astype(np.float64)[:, perm.numpy()]   # float32 -> float64 is exact
        Xc = X - X.mean(0)
        S = (Xc.T @ Xc) / (X.shape[0] - 1)
        tr = float(np.trace(S))
        alpha = tr / N_PIX
        beta = (float(S.sum()) - tr) / (N_PIX * (N_PIX - 1))
        lam, Q = np.linalg.eigh(S)
        lam, Q = lam[::-1], Q[:, ::-1]
        lam = np.maximum(lam, 0.0)
        rank = int((lam > N_PIX * EPS * lam[0]).sum())
        lam[rank:] = 0.0                                 # the null space carries no variance
        return {"t": t, "lam": lam, "Q": Q, "rank": rank, "alpha": alpha, "beta": beta,
                "bands": bands_of(rank), "lam_sum": float(lam.sum())}


def allocate():
    T = TMAX + 1
    keys_tb = ("d", "e", "V", "Nprev", "Nnow", "res", "resbound", "Sprev",
               "dnext", "Vnext", "a1", "a50", "awin", "awhole")
    out = {k: np.full((T, NB), np.nan) for k in keys_tb}
    out["T"] = np.full((T, NB, NB), np.nan)
    for k in ("Vnext_all", "Viso", "D2", "S2", "L2win", "L2whole", "N_all"):
        out[k] = np.full(T, np.nan)
    return out


def ledger_step(store, t, W, W0, W_prev, W_now, cur, nxt, Ov):
    """One arm, one task. W(s) gives the end-of-task state s; W_prev is the state at the start of
    learning task t (after any switch operation, spec_pm_ledger_add §4)."""
    Q = cur["Q"]
    lam = cur["lam"]
    D = W_now - W_prev
    C = D @ Q                                            # coefficients of the deposit
    Cp = W_prev @ Q                                      # independent projections (identity check)
    Cn = W_now @ Q
    S_prev = W_prev - W0
    Cs = S_prev @ Q
    store["D2"][t] = float((D * D).sum())
    store["S2"][t] = float((S_prev * S_prev).sum())
    store["N_all"][t] = float((W_now * W_now).sum())
    # later writes, lag 1 / lag 50 / to the end of the window / to the end of the run
    L = {}
    for S, key in zip(LAGS, ("a1", "a50")):
        if t + S <= TMAX:
            L[key] = W(t + S) - W_now
    L["awhole"] = W(TMAX) - W_now
    win_end = next((t2 for t1, t2 in WINDOWS if t1 <= t <= t2), None)
    if win_end is not None:
        L["awin"] = W(win_end) - W_now
    CL = {k: v @ Q for k, v in L.items()}
    if "awin" in L:
        store["L2win"][t] = float((L["awin"] * L["awin"]).sum())
    store["L2whole"][t] = float((L["awhole"] * L["awhole"]).sum())

    for k, (a, b) in enumerate(cur["bands"]):
        c = C[:, a:b]
        cc = c * c
        cp, cn, cs = Cp[:, a:b], Cn[:, a:b], Cs[:, a:b]
        d = float(cc.sum())
        e = 2.0 * float((cp * c).sum())
        Np, Nn = float((cp * cp).sum()), float((cn * cn).sum())
        store["d"][t, k] = d
        store["e"][t, k] = e
        store["V"][t, k] = float((lam[a:b] * cc).sum())
        store["Nprev"][t, k] = Np
        store["Nnow"][t, k] = Nn
        store["res"][t, k] = Nn - Np - d - e
        store["resbound"][t, k] = DIM * 2 * U * (Nn + Np + d + abs(e))
        store["Sprev"][t, k] = float((cs * cs).sum())
        for key, cl in CL.items():
            store[key][t, k] = float((cl[:, a:b] * c).sum())

    if nxt is not None:
        Qn, lamn = nxt["Q"], nxt["lam"]
        Cnext = C @ Ov                                    # D in the next task's basis
        cc2 = Cnext * Cnext
        store["Vnext_all"][t] = float((lamn * cc2).sum())
        # the isotropic reference of spec §4(a)
        row_sum = D.sum(1)
        store["Viso"][t] = float((nxt["alpha"] - nxt["beta"]) * store["D2"][t]
                                 + nxt["beta"] * (row_sum * row_sum).sum())
        for k, (a, b) in enumerate(cur["bands"]):
            G = C[:, a:b] @ Ov[a:b, :]                    # the band-b part of D in the next basis
            g2 = G * G
            store["Vnext"][t, k] = float((lamn * g2).sum())
            for kp, (ap, bp) in enumerate(nxt["bands"]):
                store["T"][t, k, kp] = float(g2[:, ap:bp].sum())
            ap, bp = nxt["bands"][k]
            store["dnext"][t, k] = float(cc2[:, ap:bp].sum())
    return store


def run_seed(seed, runs_dir, out_dir):
    t_start = time.time()
    torch.set_num_threads(1)
    device = torch.device("cpu")
    mnist = B.Mnist(device)
    arms_here = tuple(a for a in ARMS if (runs_dir / a / f"seed{seed}" / "provenance.json").exists())
    if not arms_here:
        raise SystemExit(f"no finished runs for seed {seed}")
    W_mm = {a: np.load(runs_dir / a / f"seed{seed}" / "W1.npy", mmap_mode="r") for a in arms_here}
    Ws_mm = {a: (np.load(runs_dir / a / f"seed{seed}" / "W1s.npy", mmap_mode="r")
                 if (runs_dir / a / f"seed{seed}" / "W1s.npy").exists() else None) for a in arms_here}
    W0 = {a: W_mm[a][0].astype(np.float64) for a in arms_here}
    for a in arms_here:
        if W_mm[a].shape != (TMAX + 1, N_UNITS, N_PIX):
            raise SystemExit(f"{a}: W1 shape {W_mm[a].shape}")

    tasks = Tasks(seed, mnist, runs_dir / arms_here[0] / f"seed{seed}")
    # S-sha for the other arms: they must have drawn the same tasks as the first arm
    rd = [runs_dir / a / f"seed{seed}" for a in arms_here]
    for other in rd[1:]:
        if not np.array_equal(np.load(rd[0] / "perm.npy"), np.load(other / "perm.npy")):
            raise SystemExit(f"S-sha: perm.npy differs: {rd[0]} vs {other}")
        if json.loads((rd[0] / "idx_sha.json").read_text()) != json.loads((other / "idx_sha.json").read_text()):
            raise SystemExit(f"S-sha: idx_sha.json differs: {rd[0]} vs {other}")
    store = {a: allocate() for a in arms_here}
    meta = {"rank": np.full(TMAX + 1, -1), "lam_share": np.full((TMAX + 1, NB), np.nan),
            "dims": np.full((TMAX + 1, NB), -1), "lam_rank": np.full(TMAX + 1, np.nan),
            "lam_rank_next": np.full(TMAX + 1, np.nan), "tr": np.full(TMAX + 1, np.nan),
            "alpha": np.full(TMAX + 1, np.nan), "beta": np.full(TMAX + 1, np.nan)}

    cur = tasks.next()
    for t in range(1, TMAX + 1):
        nxt = None
        if t < TMAX:
            nxt = tasks.next()
        Ov = (cur["Q"].T @ nxt["Q"]) if nxt is not None else None
        for a in arms_here:
            W = lambda s: W_mm[a][s].astype(np.float64)   # noqa: E731
            W_prev = Ws_mm[a][t].astype(np.float64) if Ws_mm[a] is not None else W(t - 1)
            ledger_step(store[a], t, W, W0[a], W_prev, W(t), cur, nxt, Ov)
        meta["rank"][t] = cur["rank"]
        meta["tr"][t] = float(cur["lam"].sum())
        meta["alpha"][t], meta["beta"][t] = cur["alpha"], cur["beta"]
        for k, (a_, b_) in enumerate(cur["bands"]):
            meta["lam_share"][t, k] = float(cur["lam"][a_:b_].sum()) / cur["lam_sum"]
            meta["dims"][t, k] = b_ - a_
        meta["lam_rank"][t] = float(cur["lam"][cur["rank"] - 1])
        cur = nxt
        if t % 100 == 0:
            print(f"[{time.time() - t_start:7.1f}s] seed {seed} task {t}", flush=True)

    # r_all numerator of spec §4(b): ||W_{t2} - W_{t1-1}||^2 per window (identity check)
    for a in arms_here:
        rnum = {}
        for t1, t2 in WINDOWS + ((6, TMAX),):
            dW = W_mm[a][t2].astype(np.float64) - W_mm[a][t1 - 1].astype(np.float64)
            rnum[f"{t1}_{t2}"] = float((dW * dW).sum())
        store[a]["rnum_keys"] = np.array(list(rnum.keys()))
        store[a]["rnum_vals"] = np.array(list(rnum.values()))
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {f"{a}__{k}": v for a in arms_here for k, v in store[a].items()}
    payload.update({f"meta__{k}": v for k, v in meta.items()})
    np.savez(out_dir / f"seed{seed}.npz", **payload)
    print(f"seed {seed} done in {time.time() - t_start:.1f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    run_seed(a.seed, a.runs, a.out)


if __name__ == "__main__":
    main()
