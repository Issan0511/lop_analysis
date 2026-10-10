#!/usr/bin/env python3
"""c1_direction_1010 Q3 (spec §2.3, derivation §1-§3): the first-layer push in the 3-layer RL-CIFAR MLP
(rlcifar_mlp_battle_0918, arm LR = leaky 0.1 in both hidden layers, conds raw / std), split along W2.

SIGN CONVENTION: G_i > 0 <=> the expected SGD step on the new-task loss LOWERS the unit's mean
m_i = mean_n z1_ni (sinking side); G_i < 0 raises it (floating side).  Same for every part (G0, Gd, Gr).

usage:
  mlp_drift.py                                  # raw, std x seeds 0-9 x t in {1,2,5,10,20,50}, then the tables
  mlp_drift.py --tables-only                    # rebuild the tables from results/c1_direction_1010/mlp/*.npz
  mlp_drift.py --conds raw --seeds 0 --tasks 5  # a subset of states (the tables use whatever npz exist)

Per state (cond, seed, t), float64 on CPU, x = the seed's 1200 images in [0,1] (std: rlcifar_mlp_battle_0918.standardize):
  z1 = x W1^T + b1, h1 = phi(z1), z2 = h1 W2^T + b2, h2 = phi(z2), f = h2 W3^T + b3, p = softmax(f)
  df = p - 1/10            (= N dL_u/df with L_u = mean_n[logsumexp f_n - mean_c f_nc], the uniform-label expected CE)
  d2 = phi'(z2) * (df W3),  d1 = phi'(z1) * (d2 W2)
  K_n = xbar . x_n + 1     (xbar = mean image over the 1200; the +1 is the bias component, as in p169)
  G_i = sum_n K_n d1_ni    (= N <grad m_i, grad L_u>, grad m_i = (xbar, 1) on (W1[i], b1[i]); p169's S_push)
Split (exact: G is linear in W2 at fixed d2 and phi'(z1)):
  A_ki = sum_n d2_nk K_n phi'(z1_ni),   G_i(V) = sum_k V_ki A_ki
  W2 = W2^0 (snapshot t00) + dW2^drift + dW2^rest,  dW2^drift[k] = <dW2[k], mu2hat> mu2hat,  mu2 = mean_n h1 (current)
  G = G0 + Gd + Gr,   Gd_i = -mu2hat_i * Cpl_i  with  Cpl_i = sum_k c_k A_ki,  c_k = -<dW2[k], mu2hat>
Side quantities: hbar1_i = mu2_i;  p_i = P(z1_i > 0), alive <=> p_i >= 0.01;  Dbar_k = sum_n d2_nk;
  zbar2 change, exact 3-way:  zbar2_k(t) - zbar2_k(0) = <dW2[k], mu2(t)> + db2_k + <W2^0[k], mu2(t) - mu2(0)>
  (the first two terms are exactly zbar2_k(t) - [<W2^0[k], mu2(t)> + b2^0_k], the parameter change at the current mean);
  S0_i = sum_n K_n phi'(z1_ni) phi(z1_ni);  That_i = sum_k W2_ki Dbar_k  (the transmission proxy of derivation (3)).
Checks per state: M1 = autograd of L_u w.r.t. (W1, b1): <xbar, dL_u/dW1[i]> + dL_u/db1[i] against G_i / N (rel 1e-10);
  split sum G0+Gd+Gr = G (1e-12); zbar2 identity; zbar2 change split; projection <dW2^drift[k], mu2> = <dW2[k], mu2>;
  recomputed z1/z2 against the float16 z1/z2 stored in the snapshot.  M2 (sign rate of G over alive units against
  p169's S_push, seeds 0-2) is computed in the tables step.
"""
from __future__ import annotations

import os
import sys

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "4"

import argparse
import glob
import json
import math
import subprocess
import time

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(4)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
import pmnist_rlcifar_0907 as RC                       # noqa: E402
from rlcifar_mlp_battle_0918 import standardize       # noqa: E402

OBS = "/home/issan/Projects/obsidian-research-data"
SNAP = os.path.join(OBS, "rlcifar_mlp_battle_0918/results/rlcifar_mlp_battle_0918/LR/snap")
P169 = os.path.join(OBS, "push_direction_proof_0930/review/stage4_1002/block_1003/stage12_1007")
OUT = os.path.join(REPO, "results/c1_direction_1010/mlp")
TAB = os.path.join(REPO, "results/c1_direction_1010/mlp_tables")
SLOPE = 0.1          # leaky slope of the LR arm (rlcifar_mlp_battle_0918: Leaky("LR", 0.1)), both hidden layers
NCLS = 10
ALIVE_P = 0.01       # alive <=> P(z1 > 0) >= 1% (p169's live1)
TASKS = (1, 2, 5, 10, 20, 50)
SEEDS = tuple(range(10))
CONDS = ("raw", "std")


def phi(z):
    return np.where(z > 0, z, SLOPE * z)


def dphi(z):
    return np.where(z > 0, 1.0, SLOPE)


def load_state(cond, seed, t):
    d = np.load(os.path.join(SNAP, f"LR_{cond}_seed{seed}", f"t{t:02d}.npz"))
    P = {k: d[k].astype(np.float64) for k in ("W1", "b1", "W2", "b2", "W3", "b3")}
    Z = {k: d[k].astype(np.float64) for k in ("z1", "z2") if k in d.files}
    return P, Z


def forward(X, P):
    z1 = X @ P["W1"].T + P["b1"]
    h1 = phi(z1)
    z2 = h1 @ P["W2"].T + P["b2"]
    h2 = phi(z2)
    f = h2 @ P["W3"].T + P["b3"]
    e = np.exp(f - f.max(1, keepdims=True))
    return dict(z1=z1, h1=h1, g1=dphi(z1), z2=z2, h2=h2, g2=dphi(z2), f=f, p=e / e.sum(1, keepdims=True))


def autograd_push(X, P):
    """M1: <xbar, dL_u/dW1[i]> + dL_u/db1[i] by autograd (independent code path from the manual backprop)."""
    Xt = torch.from_numpy(X)
    W1 = torch.tensor(P["W1"], requires_grad=True)
    b1 = torch.tensor(P["b1"], requires_grad=True)
    W2, b2, W3, b3 = (torch.from_numpy(P[k]) for k in ("W2", "b2", "W3", "b3"))
    lk = lambda z: torch.where(z > 0, z, SLOPE * z)                                   # noqa: E731
    f = lk(lk(Xt @ W1.T + b1) @ W2.T + b2) @ W3.T + b3
    Lu = (torch.logsumexp(f, 1) - f.mean(1)).mean()
    gW1, gb1 = torch.autograd.grad(Lu, (W1, b1))
    return (gW1 @ Xt.mean(0) + gb1).numpy(), float(Lu.detach())


def relmax(err, scale):
    s = float(np.max(np.abs(scale)))
    return float(np.max(np.abs(err)) / s) if s > 0 else float("nan")


def zcheck(z, zs):
    """p169's alignment check: median |z - z_stored| / (|z_stored| + 1e-3) against the float16 copy."""
    return float(np.median(np.abs(z - zs) / (np.abs(zs) + 1e-3)))


def measure(X, K, P, P0, F0, Z):
    N = X.shape[0]
    F = forward(X, P)
    df = F["p"] - 1.0 / NCLS                         # (N, 10)   N * dL_u/df
    d2 = F["g2"] * (df @ P["W3"])                     # (N, H2)   delta2
    d1 = F["g1"] * (d2 @ P["W2"])                     # (N, H1)   delta1
    G = K @ d1                                        # (H1,)     G_i = sum_n K_n d1_ni
    Kg1 = K[:, None] * F["g1"]                        # (N, H1)
    A = d2.T @ Kg1                                    # (H2, H1)  A_ki = sum_n d2_nk K_n phi'(z1_ni)
    h1 = F["h1"]
    mu2 = h1.mean(0)
    mu2_norm = float(np.linalg.norm(mu2))
    mu2hat = mu2 / mu2_norm
    W2_0 = P0["W2"]
    dW2 = P["W2"] - W2_0
    a = dW2 @ mu2hat                                  # <dW2[k], mu2hat>;  c_k = -a_k
    dW2d = np.outer(a, mu2hat)
    dW2r = dW2 - dW2d
    G0 = (W2_0 * A).sum(0)
    Gd = (dW2d * A).sum(0)
    Gr = (dW2r * A).sum(0)
    c = -a
    Cpl = c @ A                                       # Gd_i = -mu2hat_i * Cpl_i
    # side quantities
    p1 = (F["z1"] > 0).mean(0)
    alive = p1 >= ALIVE_P
    sd_h1 = h1.std(0, ddof=1)
    S0 = (Kg1 * h1).sum(0)
    Dbar = d2.sum(0)
    That = P["W2"].T @ Dbar
    zbar2 = F["z2"].mean(0)
    zbar2_0 = F0["z2"].mean(0)
    mu2_0 = F0["h1"].mean(0)
    dzA = dW2 @ mu2                                   # mu2 path: <dW2[k], mu2(t)>
    dzB = P["b2"] - P0["b2"]                          # bias
    dzC = W2_0 @ (mu2 - mu2_0)                        # upstream mean change through the init weights
    dz = zbar2 - zbar2_0
    # checks
    Ga, Lu = autograd_push(X, P)
    chk = dict(
        m1_rel=relmax(G / N - Ga, Ga),
        split_rel_parts=relmax(G0 + Gd + Gr - G, np.abs(G0) + np.abs(Gd) + np.abs(Gr)),
        split_rel_G=relmax(G0 + Gd + Gr - G, G),
        cpl_rel=relmax(Gd + mu2hat * Cpl, Gd),
        zbar2_id_rel=relmax(zbar2 - (P["W2"] @ mu2 + P["b2"]), zbar2),
        dz_split_rel=relmax(dzA + dzB + dzC - dz, np.abs(dzA) + np.abs(dzB) + np.abs(dzC)),
        proj_rel=relmax(dW2d @ mu2 - dzA, dzA),
        z1_chk=zcheck(F["z1"], Z["z1"]) if "z1" in Z else float("nan"),
        z2_chk=zcheck(F["z2"], Z["z2"]) if "z2" in Z else float("nan"),
    )
    nrm = float(np.linalg.norm(dW2))
    arrays = dict(G=G, G0=G0, Gd=Gd, Gr=Gr, Cpl=Cpl, hbar1=mu2, sd_h1=sd_h1, p1=p1, alive=alive, S0=S0,
                  That=That, c=c, Dbar=Dbar, zbar2=zbar2, dz=dz, dzA=dzA, dzB=dzB, dzC=dzC,
                  p2=(F["z2"] > 0).mean(0), G_autograd_times_N=Ga * N)
    scal = dict(N=N, Lu=Lu, mu2_norm=mu2_norm, dW2_fro=nrm,
                dW2_drift_frac=float(np.linalg.norm(dW2d) / nrm) if nrm > 0 else float("nan"),
                K_neg_frac=float((K < 0).mean()), K_mean=float(K.mean()), K_sd=float(K.std()),
                pmax_mean=float(F["p"].max(1).mean()), **chk)
    return arrays, scal


def compute(conds, seeds, tasks):
    os.makedirs(OUT, exist_ok=True)
    cifar = RC.Cifar10()
    for cond in conds:
        for seed in seeds:
            x = cifar.images(RC.subset_idx(seed), torch.device("cpu"))
            if cond == "std":
                x = standardize(x)
            X = x.numpy().astype(np.float64)
            xbar = X.mean(0)
            K = X @ xbar + 1.0
            P0, _ = load_state(cond, seed, 0)
            F0 = forward(X, P0)
            for t in tasks:
                t0 = time.time()
                P, Z = load_state(cond, seed, t)
                arrays, scal = measure(X, K, P, P0, F0, Z)
                np.savez(os.path.join(OUT, f"LR_{cond}_s{seed}_t{t:02d}.npz"), cond=cond, seed=seed, t=t,
                         scalars=json.dumps(scal), **arrays)
                al = arrays["alive"]
                print(f"{cond} s{seed} t{t:02d} alive {int(al.sum()):3d} G>0 {np.mean(arrays['G'][al] > 0):.3f} "
                      f"| M1 {scal['m1_rel']:.1e} split {scal['split_rel_parts']:.1e} dz {scal['dz_split_rel']:.1e} "
                      f"z1chk {scal['z1_chk']:.1e} z2chk {scal['z2_chk']:.1e} | K<0 {scal['K_neg_frac']:.3f} "
                      f"({time.time() - t0:.2f}s)", flush=True)
    return cifar.sha256


# ---------------------------------------------------------------------------------------------- tables

_T975 = {1: 12.7062, 2: 4.3027, 3: 3.1824, 4: 2.7764, 5: 2.5706, 6: 2.4469, 7: 2.3646, 8: 2.3060, 9: 2.2622,
         10: 2.2281, 11: 2.2010, 12: 2.1788, 13: 2.1604, 14: 2.1448, 15: 2.1314, 16: 2.1199, 17: 2.1098,
         18: 2.1009, 19: 2.0930, 20: 2.0860}


def mean_ci(v):
    v = np.asarray([x for x in v if np.isfinite(x)], float)
    n = v.size
    if n == 0:
        return dict(n_seeds=0, mean=np.nan, sd=np.nan, lo=np.nan, hi=np.nan)
    m = float(v.mean())
    if n == 1:
        return dict(n_seeds=1, mean=m, sd=np.nan, lo=np.nan, hi=np.nan)
    sd = float(v.std(ddof=1))
    h = _T975[n - 1] * sd / math.sqrt(n)
    return dict(n_seeds=n, mean=m, sd=sd, lo=m - h, hi=m + h)


def spearman(a, b):
    if len(a) < 3:
        return np.nan
    ra = pd.Series(np.asarray(a, float)).rank().to_numpy()
    rb = pd.Series(np.asarray(b, float)).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def frac(m):
    return float(np.mean(m)) if np.size(m) else np.nan


def state_metrics(d, scal):
    alive = d["alive"].astype(bool)
    h = d["hbar1"]
    N = scal["N"]
    G, G0, Gd, Gr, Cpl, S0 = (d[k] for k in ("G", "G0", "Gd", "Gr", "Cpl", "S0"))
    out = {}
    subsets = {"all": alive, "hneg": alive & (h < 0), "hpos": alive & (h > 0)}
    for name, m in subsets.items():
        out[f"n_{name}"] = int(m.sum())
        tot = np.abs(G0[m]) + np.abs(Gd[m]) + np.abs(Gr[m])
        out[f"G_pos_{name}"] = frac(G[m] > 0)
        out[f"Gd_pos_{name}"] = frac(Gd[m] > 0)
        out[f"Gd_neg_{name}"] = frac(Gd[m] < 0)
        out[f"Gr_pos_{name}"] = frac(Gr[m] > 0)
        out[f"G0_pos_{name}"] = frac(G0[m] > 0)
        out[f"G0_agree_{name}"] = frac(np.sign(G0[m]) == np.sign(G[m]))
        out[f"d_gt_r_{name}"] = frac(np.abs(Gd[m]) > np.abs(Gr[m]))
        out[f"share_d_med_{name}"] = float(np.median(np.abs(Gd[m]) / tot)) if m.any() else np.nan
        out[f"Cpl_pos_{name}"] = frac(Cpl[m] > 0)
        out[f"S0_pos_{name}"] = frac(S0[m] > 0)
    out["frac_hneg"] = out["n_hneg"] / out["n_all"] if out["n_all"] else np.nan
    out["spearman_That_G"] = spearman(d["That"][alive], G[alive])
    out["c_pos"] = frac(d["c"] > 0)
    out["Dbar_pos"] = frac(d["Dbar"] > 0)
    A, B, C = np.abs(d["dzA"]), np.abs(d["dzB"]), np.abs(d["dzC"])
    out["mu2path_share_med"] = float(np.median(A / (A + B)))
    out["mu2path_share3_med"] = float(np.median(A / (A + B + C)))
    out["dzA_neg"] = frac(d["dzA"] < 0)
    out["dz_neg"] = frac(d["dz"] < 0)
    out["dzA_mean"] = float(d["dzA"].mean())
    out["dzB_mean"] = float(d["dzB"].mean())
    out["dzC_mean"] = float(d["dzC"].mean())
    out["dz_mean"] = float(d["dz"].mean())
    out["n_h0"] = int((alive & (np.abs(h) < 2.0 * d["sd_h1"] / math.sqrt(N))).sum())
    out["hbar1_med_alive"] = float(np.median(h[alive])) if alive.any() else np.nan
    for k in ("K_neg_frac", "K_mean", "K_sd", "mu2_norm", "dW2_fro", "dW2_drift_frac", "pmax_mean", "Lu"):
        out[k] = scal[k]
    return out


CHECKS = ("m1_rel", "split_rel_parts", "split_rel_G", "cpl_rel", "zbar2_id_rel", "dz_split_rel", "proj_rel",
          "z1_chk", "z2_chk")

# (key, label) -- the rows of the summary tables, in order
MAIN = [
    ("n_all", "alive units (p >= 0.01), mean per seed"),
    ("frac_hneg", "fraction of alive units with hbar1 < 0"),
    ("G_pos_all", "G > 0 (sinking), alive"),
    ("G_pos_hneg", "G > 0, alive with hbar1 below 0"),
    ("G_pos_hpos", "G > 0, alive with hbar1 above 0"),
    ("Gd_pos_hneg", "G^drift > 0, alive with hbar1 below 0"),
    ("Gd_neg_hpos", "G^drift < 0, alive with hbar1 above 0"),
    ("Gr_pos_all", "G^rest > 0, alive"),
    ("Gr_pos_hneg", "G^rest > 0, alive with hbar1 below 0"),
    ("Gr_pos_hpos", "G^rest > 0, alive with hbar1 above 0"),
    ("G0_agree_all", "sign G0 = sign G, alive"),
    ("d_gt_r_all", "abs G^drift > abs G^rest, alive"),
    ("share_d_med_all", "median abs Gd / (abs G0 + abs Gd + abs Gr), alive"),
    ("c_pos", "c_k > 0 (100 L2 units)"),
    ("Dbar_pos", "Dbar_k > 0 (100 L2 units)"),
    ("mu2path_share_med", "mu2-path share of dzbar2: median_k A/(A+B)"),
    ("spearman_That_G", "Spearman(That, G), alive"),
]
EXTRA = [
    ("Cpl_pos_all", "coupling Cpl_i = sum_k c_k A_ki > 0, alive"),
    ("Cpl_pos_hneg", "Cpl > 0, alive with hbar1 below 0"),
    ("Cpl_pos_hpos", "Cpl > 0, alive with hbar1 above 0"),
    ("G0_pos_all", "G0 > 0, alive"),
    ("G0_agree_hneg", "sign G0 = sign G, alive with hbar1 below 0"),
    ("G0_agree_hpos", "sign G0 = sign G, alive with hbar1 above 0"),
    ("d_gt_r_hneg", "abs Gd > abs Gr, alive with hbar1 below 0"),
    ("d_gt_r_hpos", "abs Gd > abs Gr, alive with hbar1 above 0"),
    ("share_d_med_hneg", "median drift share, alive with hbar1 below 0"),
    ("share_d_med_hpos", "median drift share, alive with hbar1 above 0"),
    ("S0_pos_all", "self-form S0 > 0, alive"),
    ("mu2path_share3_med", "median_k A/(A+B+C) (C = W2^0 . dmu2)"),
    ("dzA_neg", "A_k = <dW2[k], mu2> < 0 (100 L2 units)"),
    ("dz_neg", "dzbar2_k < 0 since init (100 L2 units)"),
    ("dzA_mean", "mean_k A_k (mu2 path)"),
    ("dzB_mean", "mean_k B_k (bias)"),
    ("dzC_mean", "mean_k C_k (upstream, init W2)"),
    ("dz_mean", "mean_k dzbar2_k (total since init)"),
    ("dW2_drift_frac", "Frobenius norm(dW2^drift) / norm(dW2)"),
    ("hbar1_med_alive", "median hbar1 over alive units"),
    ("K_neg_frac", "fraction of inputs with K_n < 0"),
]


def fmt(st, n_total, key):
    if st["n_seeds"] == 0 or not np.isfinite(st["mean"]):
        return "--"
    if key == "n_all":
        return f"{st['mean']:.1f}"
    big = key in ("dzA_mean", "dzB_mean", "dzC_mean", "dz_mean")
    f = (lambda v: f"{v:+.2f}") if big else (lambda v: f"{v:.2f}")
    s = f(st["mean"]) if not np.isfinite(st["lo"]) else f"{f(st['mean'])} [{f(st['lo'])}, {f(st['hi'])}]"
    if st["n_seeds"] < n_total:
        s += f" ({st['n_seeds']}s)"
    return s


def git_hash():
    try:
        return subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def tables(cifar_sha=None, argv=None):
    os.makedirs(TAB, exist_ok=True)
    files = sorted(glob.glob(os.path.join(OUT, "LR_*_s*_t*.npz")))
    rows, chk_rows = [], []
    for fpath in files:
        d = np.load(fpath)
        scal = json.loads(str(d["scalars"]))
        key = dict(cond=str(d["cond"]), seed=int(d["seed"]), t=int(d["t"]))
        rows.append({**key, **state_metrics(d, scal)})
        chk_rows.append({**key, **{k: scal[k] for k in CHECKS}})
    per = pd.DataFrame(rows).sort_values(["cond", "t", "seed"]).reset_index(drop=True)
    chk = pd.DataFrame(chk_rows).sort_values(["cond", "t", "seed"]).reset_index(drop=True)
    per.to_csv(os.path.join(TAB, "mlp_per_state.csv"), index=False, float_format="%.6g")
    chk.to_csv(os.path.join(TAB, "mlp_checks.csv"), index=False, float_format="%.3e")

    metrics = [c for c in per.columns if c not in ("cond", "seed", "t")]
    agg = []
    for (cond, t), g in per.groupby(["cond", "t"]):
        for mkey in metrics:
            st = mean_ci(g[mkey].to_numpy(float))
            agg.append(dict(cond=cond, t=t, metric=mkey, **st, seeds_total=len(g)))
    agg = pd.DataFrame(agg)
    agg.to_csv(os.path.join(TAB, "mlp_seed_mean.csv"), index=False, float_format="%.6g")

    # M2: sign rate of G over alive units against p169's S_push (live1 = p >= 0.01), seeds 0-2
    m2 = []
    for (cond, seed, t), r in per.set_index(["cond", "seed", "t"]).iterrows():
        jf = os.path.join(P169, f"p169_LR_{cond}_s{seed}.json")
        if not os.path.exists(jf):
            continue
        js = json.load(open(jf))
        tk = js["tasks"].get(str(t))
        if tk is None:
            continue
        sp = tk["l1"]["S_push"]
        m2.append(dict(cond=cond, seed=seed, t=t, ours_rate=r["G_pos_all"], ours_n=int(r["n_all"]),
                       p169_rate=sp["pos"], p169_n=sp["n"], diff=r["G_pos_all"] - sp["pos"],
                       ours_Kneg=r["K_neg_frac"], p169_Kneg=js["K_neg_frac"]))
    m2 = pd.DataFrame(m2)
    m2.to_csv(os.path.join(TAB, "mlp_m2_vs_p169.csv"), index=False, float_format="%.6g")

    # ------------------------------------------------------------------ summary_mlp.md
    L = []
    L.append("# c1_direction_1010 Q3 — 3-layer RL-CIFAR MLP, first-layer push split along W2 (LR raw / std)\n")
    L.append(f"Generated by `analysis/c1_direction_1010/mlp_drift.py` at {time.strftime('%Y-%m-%d %H:%M')} "
             f"(git HEAD {git_hash()[:8]}). Snapshots `{SNAP}/LR_{{cond}}_seed{{s}}/t{{tt}}.npz` (t00 = init).\n")
    L.append("**Sign convention: G > 0 = the expected SGD step on the uniform-label loss lowers the unit's mean "
             "(sinking side); G < 0 raises it (floating side).** Same for G0, G^drift, G^rest.\n")
    L.append("Definitions: G_i = sum_n K_n d1_ni (K_n = xbar.x_n + 1, d1 = phi'(z1) * (d2 W2), d2 = phi'(z2) * ((p - 1/10) W3)); "
             "G = G0 + G^drift + G^rest with W2 = W2^0 + dW2^drift + dW2^rest, dW2^drift[k] = <dW2[k], mu2hat> mu2hat, "
             "mu2 = mean_n h1 = hbar1 (current state). G^drift_i = -mu2hat_i * Cpl_i exactly, Cpl_i = sum_k c_k A_ki, "
             "A_ki = sum_n d2_nk K_n phi'(z1_ni), c_k = -<dW2[k], mu2hat>. Dbar_k = sum_n d2_nk. "
             "That_i = sum_k W2_ki Dbar_k. alive = P(z1 > 0) >= 0.01. "
             "zbar2 change since init (exact): dzbar2_k = A_k + B_k + C_k with A_k = <dW2[k], mu2(t)> (mu2 path), "
             "B_k = db2_k, C_k = <W2^0[k], mu2(t) - mu2(0)>; A_k + B_k is exactly zbar2_k(t) - [<W2^0[k], mu2(t)> + b2^0_k]. "
             "'mu2-path share' = median_k |A_k| / (|A_k| + |B_k|).\n")
    L.append("Cells: seed mean [t-based 95% interval] over the seeds; the rate in each seed is over that seed's units "
             "in the stated subset. '(ks)' = only k seeds had at least one unit in the subset.\n")

    seeds_total = per.groupby(["cond", "t"]).size()
    tlist = sorted(per["t"].unique())
    for cond in sorted(per["cond"].unique()):
        for title, rowsdef in (("main", MAIN), ("extra", EXTRA)):
            L.append(f"\n## {cond} — {title}\n")
            L.append("| quantity | " + " | ".join(f"t{t}" for t in tlist) + " |")
            L.append("|---|" + "---|" * len(tlist))
            for key, label in rowsdef:
                cells = []
                for t in tlist:
                    sel = agg[(agg.cond == cond) & (agg.t == t) & (agg.metric == key)]
                    if sel.empty:
                        cells.append("--")
                        continue
                    st = sel.iloc[0].to_dict()
                    cells.append(fmt(st, int(seeds_total[(cond, t)]), key))
                L.append(f"| {label} | " + " | ".join(cells) + " |")

    L.append("\n## Unit counts (alive = p >= 0.01), per (cond, t)\n")
    L.append("| cond | t | alive per seed (s0..s9) | total alive | hbar1<0 | hbar1>0 | abs hbar1 < 2 SE | K_n<0 frac (seed range) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for (cond, t), g in per.groupby(["cond", "t"]):
        g = g.sort_values("seed")
        L.append(f"| {cond} | {t} | {' '.join(str(int(v)) for v in g['n_all'])} | {int(g['n_all'].sum())} | "
                 f"{int(g['n_hneg'].sum())} | {int(g['n_hpos'].sum())} | {int(g['n_h0'].sum())} | "
                 f"{g['K_neg_frac'].min():.3f}-{g['K_neg_frac'].max():.3f} |")

    L.append("\n## Per-seed rates (key quantities)\n")
    for key, label in (("G_pos_all", "G > 0, alive"), ("Gd_pos_hneg", "G^drift > 0, alive with hbar1 below 0"),
                       ("Gd_neg_hpos", "G^drift < 0, alive with hbar1 above 0"), ("Gr_pos_all", "G^rest > 0, alive"),
                       ("frac_hneg", "fraction hbar1 < 0")):
        L.append(f"\n**{label}** (`{key}`)\n")
        L.append("| cond | t | " + " | ".join(f"s{s}" for s in sorted(per['seed'].unique())) + " |")
        L.append("|---|---|" + "---|" * per["seed"].nunique())
        for (cond, t), g in per.groupby(["cond", "t"]):
            g = g.set_index("seed")
            L.append(f"| {cond} | {t} | " + " | ".join(
                ("--" if not np.isfinite(g.loc[s, key]) else f"{g.loc[s, key]:.2f}") if s in g.index else "--"
                for s in sorted(per['seed'].unique())) + " |")

    L.append("\n## Checks (max over all states)\n")
    L.append("| check | max | threshold |")
    L.append("|---|---|---|")
    thr = dict(m1_rel="1e-10 (M1: autograd of L_u vs G/N, rel. to max abs)", split_rel_parts="1e-12 (rel. to max of abs G0 + abs Gd + abs Gr)",
               split_rel_G="(same error rel. to max abs G)", cpl_rel="(Gd = -mu2hat * Cpl, identity)",
               zbar2_id_rel="(zbar2 = W2 mu2 + b2)", dz_split_rel="(dzbar2 = A + B + C)",
               proj_rel="(<dW2^drift[k], mu2> = <dW2[k], mu2>)",
               z1_chk="(median rel. diff. to the float16 z1 in the snapshot; float16 eps 4.9e-4)",
               z2_chk="(same for z2)")
    for k in CHECKS:
        L.append(f"| {k} | {chk[k].max():.2e} | {thr[k]} |")
    if not m2.empty:
        L.append("\n## M2: sign rate of G over alive units vs p169 S_push (live1 = p >= 0.01), seeds 0-2\n")
        L.append(f"max |diff| = {m2['diff'].abs().max():.3g}; unit counts equal in {int((m2.ours_n == m2.p169_n).sum())}/{len(m2)} states; "
                 f"K<0 fraction max |diff| = {(m2.ours_Kneg - m2.p169_Kneg).abs().max():.3g}.\n")
        L.append("| cond | seed | t | ours (n) | p169 (n) | diff |")
        L.append("|---|---|---|---|---|---|")
        for _, r in m2.sort_values(["cond", "seed", "t"]).iterrows():
            L.append(f"| {r.cond} | {int(r.seed)} | {int(r.t)} | {r.ours_rate:.4f} ({int(r.ours_n)}) | "
                     f"{r.p169_rate:.4f} ({int(r.p169_n)}) | {r['diff']:+.4f} |")
    with open(os.path.join(TAB, "summary_mlp.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    info = dict(script="analysis/c1_direction_1010/mlp_drift.py", argv=argv, git_head=git_hash(),
                generated=time.strftime("%Y-%m-%d %H:%M:%S"), snapshots=SNAP, p169=P169, n_states=len(per),
                cifar_sha256=cifar_sha, torch=torch.__version__, numpy=np.__version__, dtype="float64", device="cpu",
                threads=torch.get_num_threads())
    with open(os.path.join(TAB, "run_info.json"), "w") as fh:
        json.dump(info, fh, indent=1)
    print(f"tables: {len(per)} states -> {TAB}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conds", nargs="+", default=list(CONDS))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--tasks", nargs="+", type=int, default=list(TASKS))
    ap.add_argument("--tables-only", action="store_true")
    args = ap.parse_args()
    sha = None
    if not args.tables_only:
        sha = compute(args.conds, args.seeds, args.tasks)
    tables(sha, sys.argv[1:])


if __name__ == "__main__":
    main()
