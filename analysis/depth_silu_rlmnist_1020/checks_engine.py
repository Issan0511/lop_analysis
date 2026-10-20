#!/usr/bin/env python3
"""Engine checks of depth_silu_rlmnist_1020 (spec section 6): S-repro1, S-repro2, S-online, S-doorC, each with its
mutant.  Writes results/depth_silu_rlmnist_1020/checks/<name>.json.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python analysis/depth_silu_rlmnist_1020/checks_engine.py --only repro1_SILU
"""
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src import depth_silu_rlmnist_1020 as E

OUT = ROOT / "results" / E.EXPERIMENT / "checks"
STATES = Path("/home/issan/Projects/obsidian-research-data/push_direction_proof_0930/states")
SIGN = Path("/home/issan/Projects/obsidian-research-data/sink_roots_0930/round1/sign_states")


def maxdiff(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return float("inf")
    return float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max()) if a.size else 0.0


def write(name, rec):
    OUT.mkdir(parents=True, exist_ok=True)
    rec["check"] = name
    (OUT / f"{name}.json").write_text(json.dumps(rec, indent=1))
    print(name, "PASS" if rec["pass"] else "FAIL", {k: v for k, v in rec.items() if k not in ("check",)}, flush=True)


def repro1(act, seed, tasks=30):
    """L = 1 against the saved gen_states run: end*, mid*, y, acc over tasks 1..tasks, exact."""
    t0 = time.time()
    d, _, info = E.run(act, 1, seed, tasks, 4000, 1e-3, progress=False)
    s = np.load(STATES / f"{act}_K10_s{seed}.npz")
    diffs = {}
    for j in range(4):
        diffs[f"end{j}"] = maxdiff(d[f"end{j}"], s[f"end{j}"][:tasks])
        diffs[f"mid{j}"] = maxdiff(d[f"mid{j}"], s[f"mid{j}"][:tasks])
        diffs[f"init{j}"] = maxdiff(d[f"init{j}"], s[f"init{j}"])
    diffs["y"] = maxdiff(d["y"], s["y"][:tasks]); diffs["acc"] = maxdiff(d["acc"], s["acc"][:tasks])
    diffs["subset"] = maxdiff(d["subset"], s["subset"])
    ok = all(v == 0.0 for v in diffs.values())
    # mutant: Adam eps 1e-7, one task -> must differ from the saved task-1 end
    m, _, _ = E.run(act, 1, seed, 1, 4000, 1e-3, eps=1e-7, progress=False)
    mut = max(maxdiff(m[f"end{j}"][0], s[f"end{j}"][0]) for j in range(4))
    write(f"S-repro1_{act}_s{seed}", {"pass": ok and mut > 0, "max_abs_diff": diffs, "mutant_eps1e-7_task1_max_abs_diff": mut,
                                      "mutant_detected": mut > 0, "tasks": tasks, "seconds": time.time() - t0})


def repro2(seed, tasks=5):
    """L = 2, 6,000 updates per task, against sink_roots_0930/round1/sign_states (params and Adam moments) at the
    starts of tasks 2, 3, 4, 6 (= ends of tasks 1, 2, 3, 5), exact."""
    t0 = time.time()
    d, adam, _ = E.run("SILU", 2, seed, tasks, 6000, 1e-3, progress=False, capture_adam=(1, 2, 3, 5))
    res, ok = {}, True
    for t in (2, 3, 4, 6):
        f = np.load(SIGN / f"L2_SILU_K10_s{seed}_t00{t}.npz")
        k = t - 1
        dp = max(maxdiff(d[f"end{j}"][k - 1], f[f"p{j}"]) for j in range(6))
        dm = max(maxdiff(adam["captured"][k]["m"][j], f[f"m{j}"]) for j in range(6))
        dv = max(maxdiff(adam["captured"][k]["v"][j], f[f"v{j}"]) for j in range(6))
        dy = maxdiff(d["y"][k - 1], f["y_old"])
        res[f"t00{t}"] = {"params": dp, "adam_m": dm, "adam_v": dv, "labels": dy}
        ok &= (dp == 0 and dm == 0 and dv == 0 and dy == 0)
    m, _, _ = E.run("SILU", 2, seed, 1, 6000, 1e-3, init_order="reverse", progress=False)
    f = np.load(SIGN / f"L2_SILU_K10_s{seed}_t002.npz")
    mut = max(maxdiff(m[f"end{j}"][0], f[f"p{j}"]) for j in range(6))
    write(f"S-repro2_SILU_s{seed}", {"pass": bool(ok and mut > 0), "max_abs_diff": res, "mutant_reverse_init_max_abs_diff": mut,
                                     "mutant_detected": mut > 0, "seconds": time.time() - t0})


def online(act, L, door, seed=0):
    """The first recorded batch: hits and loss equal a recomputation from the init parameters on the first batch
    (index stream and label stream redrawn independently here)."""
    d, _, _ = E.run(act, L, seed, 1, 4000, 1e-3, door=door, progress=False)
    _, _, X = E.load_inputs(seed, door)
    y = torch.randint(10, (E.N,), generator=E.stream("env_labels_0913", seed))
    g = E.stream("env_batch_0913", seed)
    idx = torch.randperm(E.N, generator=g)[:E.BATCH]
    P = [torch.tensor(d[f"init{j}"]) for j in range(2 * L + 2)]
    with torch.no_grad():
        h = X[idx]
        for l in range(L):
            h = E.activ(h @ P[2 * l].T + P[2 * l + 1], act)
        lg = h @ P[2 * L].T + P[2 * L + 1]
        hits = int((lg.argmax(-1) == y[idx]).sum())
        yb = torch.nn.functional.one_hot(y[idx], 10).to(torch.float32)
        loss = float((lg.logsumexp(-1) - (lg * yb).sum(-1)).mean())
    # mutant: the post-update parameters of the first batch would not give the recorded loss
    ok = hits == int(d["hits"][0, 0]) and loss == float(d["loss"][0, 0])
    write(f"S-online_{act}_L{L}_{door}", {"pass": bool(ok), "hits_recorded": int(d["hits"][0, 0]), "hits_recomputed": hits,
                                          "loss_recorded": float(d["loss"][0, 0]), "loss_recomputed": loss})


def doorc(seed=0):
    """Door C: the centred x's per-pixel mean is within the float32 rounding bound; the uncentred x (mutant) is not;
    the x every batch sees has the recorded sha256."""
    subset, x_raw, x = E.load_inputs(seed, "C")
    bound = E.c_bound(x_raw, x, E.N)
    m64 = x.double().mean(0)
    ok_c = bool((m64.abs() <= bound).all())
    m64_raw = x_raw.double().mean(0)
    mut = bool((m64_raw.abs() <= bound).all())
    d, _, info = E.run("SILU", 4, seed, 1, 4000, 1e-3, door="C", progress=False)
    sha_ok = d["x_sha256"] == E.sha(x) and info["c_within_bound"]
    # layer-1 K_n = xbar_c . x_n + 1: its spread is bounded by |xbar_c| * max_n |x_n|
    x64 = x.double(); xb = x64.mean(0); K = x64 @ xb + 1.0
    kb = float(xb.norm() * x64.norm(dim=1).max())
    write(f"S-doorC_s{seed}", {"pass": bool(ok_c and (not mut) and sha_ok and float(K.std()) <= kb),
                               "centred_absmax": float(m64.abs().max()), "bound_max": float(bound.max()),
                               "centred_within_bound": ok_c, "mutant_uncentred_within_bound": mut,
                               "x_sha256_match": bool(d["x_sha256"] == E.sha(x)), "K_sd": float(K.std()), "K_sd_bound": kb,
                               "xbar_c_norm": float(xb.norm())})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--only", required=True)
    a = ap.parse_args()
    {"repro1_SILU": lambda: repro1("SILU", 0), "repro1_R": lambda: repro1("R", 0),
     "repro2_s0": lambda: repro2(0), "repro2_s1": lambda: repro2(1),
     "online": lambda: [online("SILU", 1, "none"), online("SILU", 4, "none"), online("SILU", 4, "C"), online("R", 4, "none")],
     "doorc": lambda: doorc(0)}[a.only]()
