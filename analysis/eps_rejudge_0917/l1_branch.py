#!/usr/bin/env python3
"""eps_rejudge_0917 C2 (spec 2.3) with checks S1 and S2: what holds the second layer still at the
branch states of resp_ee_0917 (lineage L1: white-san CPU, autograd expm1, flush_denormal on).

    OMP_NUM_THREADS=1 python3 analysis/eps_rejudge_0917/l1_branch.py --seeds 0,1,2,3

For every saved state (task t = 2, 5, 7, 10, 15, 20 of seed s):
  natural  task t+1, 6000 updates, the host's arithmetic (src/resp_ee_0917.py train_task) with the
           gradients read on the side: per coordinate, how many updates had a gradient that is not
           exactly 0 (first epoch and whole task) and the sum of squares; the parameters after the
           first epoch (75 updates).  The end state must hash to prefix.csv's task t+1 (S1).
  eps30    the same state, labels and minibatch order, 75 updates with Adam's eps = 1e-30.
  replay   the same 75 updates with eps = 1e-8 again: must equal the natural first epoch bit for bit
           (S2a -- the class E is empty when nothing is changed).
  mutant   seed 0, t = 15 and 20 only: 75 updates with the second layer's derivative replaced by the
           analytic exp(z) (the GPU lineage's gate).  Its second layer must have fewer class-Z
           coordinates than the natural run (S2b).
  eps30full  t = 10, 15, 20: the whole task t+1 with eps = 1e-30 (report only).

Classes per coordinate after the first epoch (spec 2.3): M moved; Z still, every first-epoch gradient
exactly 0 and m exactly 0 at the branch; E still, not Z, moves under eps = 1e-30; Q the rest.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import resource
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src import pmnist_0905 as H                 # noqa: E402  host; not touched
from src import pmnist_rlmnist_0906 as RL        # noqa: E402
from src import shell_l2_rlmnist_0913 as SH      # noqa: E402
from src import mucap_el_run_0916 as EL          # noqa: E402
from src import resp_ee_0917 as R                # noqa: E402

RAW = Path.home() / "Projects" / "obsidian-research-data" / "resp_ee_0917" / "results" / "resp_ee_0917" / "runs"
PREFIX = ROOT / "results" / "resp_ee_0917" / "runs"
OUT = ROOT / "results" / "eps_rejudge_0917" / "l1_branch"
NAMES = ("W1", "b1", "W2", "b2", "W3", "b3")
BRANCH_T = R.BRANCH_T
FULL_EPS_T = (10, 15, 20)
MUTANT = {0: (15, 20)}
EPS8 = R.EPS
EPS30 = 1e-30
SPE, BATCH, N_IMG = R.SPE, R.BATCH, R.N_IMG
SPT = SPE * R.EPOCHS
ADAM_BOUND = 1e-2       # |update| above this is not an Adam step (|m^/sqrt(v^)| <= 3.2 so |upd| <= 3.2e-3)


class ELUExpGrad(torch.autograd.Function):
    """ELU(1) forward (the host's values); backward with exp(min(z, 0)) -- the GPU lineage's gate."""

    @staticmethod
    def forward(ctx, z):
        ctx.save_for_backward(z)
        return R.ACT.phi(z)

    @staticmethod
    def backward(ctx, g):
        z, = ctx.saved_tensors
        return g * torch.where(z > 0, torch.ones_like(z), z.clamp(max=0.0).exp())


def forward_nat(params, xb):
    return EL.forward2(params, xb, R.ACT, R.ACT)[4]


def forward_mut(params, xb):
    W1, b1, W2, b2, W3, b3 = params
    a1 = R.ACT.phi(xb @ W1.T + b1)
    a2 = ELUExpGrad.apply(a1 @ W2.T + b2)
    return a2 @ W3.T + b3


def load_state(ck):
    params = [p.detach().clone().requires_grad_(True) for p in ck["params"]]
    adam = ([q.clone() for q in ck["m"]], [q.clone() for q in ck["v"]], [int(ck["tc"])])
    return params, adam


def run_updates(params, adam, x, y, g_batch, n_updates, eps, fwd=forward_nat, rec=None):
    """src/resp_ee_0917.py train_task's loop with Adam's eps as an argument (same ops, same order).
    rec: accumulators (read-only side channel).  Returns (sum of the per-update float32 accuracies in
    float64, the first update index whose |update| exceeded ADAM_BOUND or None, first non-finite or None)."""
    m, v, tc = adam
    acc_sum = 0.0
    big = None
    s = 0
    for ep in range(math.ceil(n_updates / SPE)):
        order = torch.randperm(N_IMG, generator=g_batch).to(x.device)
        xs, ys = x[order], y[order]
        for j in range(SPE):
            if s >= n_updates:
                break
            xb, yb = xs[j * BATCH:(j + 1) * BATCH], ys[j * BATCH:(j + 1) * BATCH]
            out = fwd(params, xb)
            loss = F.cross_entropy(out, yb)
            acc_sum += float((out.detach().argmax(1) == yb).float().mean())
            grads = torch.autograd.grad(loss, params)
            with torch.no_grad():
                if rec is not None:
                    for k, gr in enumerate(grads):
                        nz = gr != 0
                        rec["nz_all"][k] += nz
                        rec["sq_all"][k] += gr.double().square()
                        if s < SPE:
                            rec["nz_ep1"][k] += nz
                            rec["sq_ep1"][k] += gr.double().square()
                tc[0] += 1
                c1, c2 = 1 - R.BETA1 ** tc[0], 1 - R.BETA2 ** tc[0]
                for p, gr, mi, vi in zip(params, grads, m, v):
                    mi.mul_(R.BETA1).add_(gr, alpha=1 - R.BETA1)
                    vi.mul_(R.BETA2).addcmul_(gr, gr, value=1 - R.BETA2)
                    upd = R.LR * (mi / c1) / ((vi / c2).sqrt() + eps)
                    p -= upd
                    if big is None and bool((upd.abs() > ADAM_BOUND).any()):
                        big = s
                if rec is not None and s + 1 == SPE:
                    rec["after_ep1"] = [p.detach().clone() for p in params]
            s += 1
    finite = all(bool(torch.isfinite(p).all()) for p in params)
    return acc_sum, big, finite


def new_rec(params):
    return {"nz_all": [torch.zeros_like(p, dtype=torch.int32) for p in params],
            "sq_all": [torch.zeros_like(p, dtype=torch.float64) for p in params],
            "nz_ep1": [torch.zeros_like(p, dtype=torch.int32) for p in params],
            "sq_ep1": [torch.zeros_like(p, dtype=torch.float64) for p in params]}


@torch.no_grad()
def layer_state(params, x):
    """Per unit, both hidden layers, on the 1200 images (the batch-1200 float32 z): U, dead (training
    factor 0 on every image), share of (unit, image) pairs with factor 0."""
    z1, _, z2, _, _ = EL.forward2(params, x, R.ACT, R.ACT)
    out = {}
    for li, z in ((1, z1), (2, z2)):
        zero = R.dphi_train(z) == 0
        out[f"U{li}"] = z.amax(0).double().numpy()
        out[f"dead{li}"] = int(zero.all(0).sum())
        out[f"pairzero{li}"] = float(zero.double().mean())
    return out


def classify(start, after8, after30, nz_ep1, m0, v0, tc0):
    """Counts of M/Z/E/Q per parameter (spec 2.3) plus the report columns."""
    c2 = 1 - R.BETA2 ** tc0
    res = {}
    for k, name in enumerate(NAMES):
        p0 = start[k]
        moved8 = after8[k] != p0
        moved30 = (after30[k] != p0) | ~torch.isfinite(after30[k])
        still = ~moved8
        Z = still & (nz_ep1[k] == 0) & (m0[k] == 0)
        E = still & ~Z & moved30
        Q = still & ~Z & ~moved30
        epsdom0 = (v0[k] / c2).sqrt() < EPS8
        res[name] = {"n": int(p0.numel()), "M": int(moved8.sum()), "Z": int(Z.sum()), "E": int(E.sum()),
                     "Q": int(Q.sum()), "Z_moved30": int((Z & moved30).sum()),
                     "M_epsdom0": int((moved8 & epsdom0).sum()), "epsdom0": int(epsdom0.sum()),
                     "still_had_grad": int((still & ~Z & (nz_ep1[k] > 0)).sum()),
                     "still_m_only": int((still & ~Z & (nz_ep1[k] == 0)).sum()),
                     "m0_zero": int((m0[k] == 0).sum())}
    return res


def rms_bins(sq, nsteps):
    out = {}
    for k, name in enumerate(NAMES):
        rms = (sq[k] / nsteps).sqrt()
        out[name] = {"zero": int((rms == 0).sum()), "below_eps": int(((rms > 0) & (rms < EPS8)).sum()),
                     "at_or_above_eps": int((rms >= EPS8).sum())}
    return out


def read_prefix(seed):
    import csv
    with (PREFIX / f"s{seed}" / "prefix.csv").open() as fh:
        return {int(r["task"]): r for r in csv.DictReader(fh)}


def run_seed(seed, mnist):
    t_seed = time.time()
    cks = torch.load(RAW / f"s{seed}" / "branch_states.pt", weights_only=False)
    prefix = read_prefix(seed)
    x = mnist.train_x[RL.subset_idx(seed)]
    rows = []
    for t in BRANCH_T:
        t0 = time.time()
        ck = cks[t]
        assert ck["state_sha256"] == prefix[t]["state_sha256"], (seed, t, "saved state is not the prefix")
        y = RL.task_labels(R.gen_from(ck["g_lab"]))
        start = [p.detach().clone() for p in ck["params"]]
        m0 = [q.clone() for q in ck["m"]]
        v0 = [q.clone() for q in ck["v"]]
        tc0 = int(ck["tc"])
        row = {"seed": seed, "t": t, "tc0": tc0}
        with torch.no_grad():
            st = layer_state(ck["params"], x)
        row["start"] = {k: v for k, v in st.items() if not k.startswith("U")}
        row["start"]["U2_max"] = float(st["U2"].max())
        row["start"]["U2_margin_units"] = int((np.abs(st["U2"] - (-16.635534286499023)) < 1e-4).sum())
        # natural task t+1
        params, adam = load_state(ck)
        rec = new_rec(params)
        acc_nat, big_nat, fin_nat = run_updates(params, adam, x, y, R.gen_from(ck["g_batch"]), SPT, EPS8, rec=rec)
        h = SH.state_sha256(params, adam, R.ACT)
        pref = prefix[t + 1]
        row["S1_hash_match"] = h == pref["state_sha256"]
        row["S1_online_match"] = (acc_nat / SPT) == float(pref["online_acc"])
        row["nat_online"] = acc_nat / SPT
        with torch.no_grad():
            en = layer_state(params, x)
        row["nat_end"] = {k: v for k, v in en.items() if not k.startswith("U")}
        after8 = rec["after_ep1"]
        # eps = 1e-30, 75 updates
        p30, a30 = load_state(ck)
        _, big30, fin30 = run_updates(p30, a30, x, y, R.gen_from(ck["g_batch"]), SPE, EPS30)
        after30 = [p.detach().clone() for p in p30]
        row["eps30_ep1_big_update_at"] = big30
        row["eps30_ep1_finite"] = fin30
        # S2a replay with eps = 1e-8
        pr, ar = load_state(ck)
        run_updates(pr, ar, x, y, R.gen_from(ck["g_batch"]), SPE, EPS8)
        row["S2a_replay_equal"] = all(torch.equal(a, b.detach()) for a, b in zip(after8, pr))
        row["classes"] = classify(start, after8, after30, rec["nz_ep1"], m0, v0, tc0)
        row["classes_replay_E"] = {n: c["E"] for n, c in
                                   classify(start, after8, [p.detach() for p in pr], rec["nz_ep1"], m0, v0, tc0).items()}
        row["rms_ep1"] = rms_bins(rec["sq_ep1"], SPE)
        row["rms_task"] = rms_bins(rec["sq_all"], SPT)
        row["nz_task_zero"] = {n: int((rec["nz_all"][k] == 0).sum()) for k, n in enumerate(NAMES)}
        # S2b mutant (exp derivative in layer 2)
        if t in MUTANT.get(seed, ()):
            pm, am = load_state(ck)
            recm = new_rec(pm)
            run_updates(pm, am, x, y, R.gen_from(ck["g_batch"]), SPE, EPS8, fwd=forward_mut, rec=recm)
            cm = classify(start, [p.detach() for p in pm], after30, recm["nz_ep1"], m0, v0, tc0)
            row["S2b_mutant_Z"] = {n: cm[n]["Z"] for n in ("W2", "b2")}
            row["S2b_natural_Z"] = {n: row["classes"][n]["Z"] for n in ("W2", "b2")}
        # report only: whole task t+1 with eps = 1e-30
        if t in FULL_EPS_T:
            pf, af = load_state(ck)
            acc30, bigf, finf = run_updates(pf, af, x, y, R.gen_from(ck["g_batch"]), SPT, EPS30)
            row["eps30full_online"] = acc30 / SPT
            row["eps30full_big_update_at"] = bigf
            row["eps30full_finite"] = finf
            if finf:
                with torch.no_grad():
                    ef = layer_state(pf, x)
                row["eps30full_end"] = {k: v for k, v in ef.items() if not k.startswith("U")}
        row["sec"] = time.time() - t0
        rows.append(row)
        print(f"[{time.time() - t_seed:7.1f}s] seed {seed} t{t}: S1 {row['S1_hash_match']} "
              f"S2a {row['S2a_replay_equal']} W2 {row['classes']['W2']} ({row['sec']:.1f}s)", flush=True)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--smoke", default=None, help="comma list of branch tasks; writes to results/_smoke_eps_rejudge_0917")
    args = ap.parse_args(argv)
    global OUT, BRANCH_T
    if args.smoke:
        OUT = ROOT / "results" / "_smoke_eps_rejudge_0917" / "l1_branch"
        BRANCH_T = tuple(int(t) for t in args.smoke.split(","))
    git0 = {"git_hash": H.git_hash(), "git_dirty_code": SH.git_dirty(["src", "analysis/eps_rejudge_0917"])}
    torch.set_num_threads(args.threads)
    torch.set_flush_denormal(True)                 # resp_ee_0917's main sets it before anything
    H.setup("cpu")
    mnist = H.Mnist(torch.device("cpu"))
    OUT.mkdir(parents=True, exist_ok=True)
    for seed in (int(s) for s in args.seeds.split(",")):
        t0 = time.time()
        rows = run_seed(seed, mnist)
        prov = {**git0, "torch": torch.__version__, "threads": torch.get_num_threads(),
                "flush_denormal": R._flush_is_on(), "seconds": time.time() - t0,
                "peak_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        (OUT / f"s{seed}.json").write_text(json.dumps({"seed": seed, "rows": rows, "provenance": prov}, indent=1))
        print(f"wrote {OUT / f's{seed}.json'} ({prov['seconds']:.0f}s, rss {prov['peak_rss_kb'] / 1e6:.2f} GB)",
              flush=True)


if __name__ == "__main__":
    main()
