#!/usr/bin/env python3
"""R7 on the GPU: the registered checks K2-K5 of spec_postfit_elu_cifar_0924 §5 (K1 is r7_k1_full.py; K6 needs arm S,
which is not run because the pilot gave no eta_S).  Each check has its mutation control.
K2 intervention flags (per_task.csv), K3 branching point (task-1 traces of F and A, and F99 earlier than F),
K4 labels (major_frac from the replayed labels against S5 ref), K5 post-hoc quantities of A's snapshots against S5 ref
(G2, Q2, |mu2|, Wnorm1; relative 1e-5, Q2 within 1e-4)."""
import csv, json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src import rlcifar_mlp_battle_0918 as B          # noqa: E402
from src import pmnist_rlcifar_0907 as RC             # noqa: E402

RAW = Path("/home/issan/Projects/obsidian-research-data/sink_roots_0930/r7gpu")
REF = ROOT / "results" / "cap_cifar_ee_0920" / "per_task.csv"
SEEDS = range(10)
torch.set_num_threads(4)


def rows(arm):
    f = RAW / arm / "per_task.csv"
    if not f.exists() or not (RAW / arm / "provenance.json").exists():
        return None
    return {(int(r["seed"]), int(r["task"])): r for r in csv.DictReader(open(f))}


def switched(r):
    return 0 < int(float(r["switch_step"])) <= 30000


def k2(P):
    out = {}
    for arm, rs in P.items():
        if rs is None:
            out[arm] = "INCOMPLETE"; continue
        bad, n_sw = [], 0
        for k, r in rs.items():
            f = {c: int(float(r[c])) for c in ("pf_P_same", "pf_mv_same", "pf_mv_restored")}
            if arm == "A":
                if switched(r) and (f["pf_mv_same"] == 1 or f["pf_P_same"] == 1):
                    bad.append(k)                                  # A must not look frozen
                if switched(r):
                    n_sw += 1
            elif switched(r):
                n_sw += 1
                if f["pf_mv_same"] != 1 or f["pf_P_same"] != 1:
                    bad.append(k)
            elif any(v != -1 for v in f.values()):
                bad.append(k)                                      # no switch -> every flag "no switch"
        out[arm] = {"pass": not bad, "switched_rows": n_sw, "bad": bad[:10]}
    return out


def k3(P):
    res = {"F_vs_A": {}, "F99_before_F": None}
    earlier = []
    for s in SEEDS:
        a = np.load(RAW / "A" / "trace" / f"ELU_std_seed{s}.npz"); f = np.load(RAW / "F" / "trace" / f"ELU_std_seed{s}.npz")
        sw = int(float(P["F"][(s, 1)]["switch_step"]))
        ma, mf = a["task"] == 1, f["task"] == 1
        cols = ("correct", "ce", "ce_st", "margin_med", "n1", "n2", "n3", "sig_med")
        A_ = np.stack([a[c][ma].astype(float) for c in cols], 1); F_ = np.stack([f[c][mf].astype(float) for c in cols], 1)
        st = a["step"][ma]; assert (st == f["step"][mf]).all()
        before = st <= sw; after = st > sw
        same_before = bool(np.array_equal(A_[before], F_[before]))
        diff_after = bool(after.any() and not np.array_equal(A_[after], F_[after]))
        res["F_vs_A"][s] = {"s_sw": sw, "same_up_to_s_sw": same_before, "differ_after": diff_after}
        if P.get("F99") is not None:
            earlier.append(int(float(P["F99"][(s, 1)]["switch_step"])) < sw)
    res["pass"] = all(v["same_up_to_s_sw"] and v["differ_after"] for v in res["F_vs_A"].values())
    if earlier:
        res["F99_before_F"] = {"n_earlier": int(sum(earlier)), "n": len(earlier)}
    return res


def k4_k5():
    ref = {(int(r["seed"]), int(r["task"])): r for r in csv.DictReader(open(REF)) if r["arm"] == "ref"}
    cifar = RC.Cifar10()
    out = RAW / "A"
    # the run's own slot layout (provenance "slots", R = 10) on the run's device: the forward `evaluate` ran, bit for bit
    slots = [(d["seed"], d["cond"]) for d in json.load(open(out / "provenance.json"))["slots"]]
    dev = torch.device("cuda")
    worst = {"major_frac": 0.0, "G2": 0.0, "mu2_norm": 0.0, "Wnorm1": 0.0, "Q2_abs": 0.0}
    g2_abs = 0.0
    layer_swap_match = 0
    mut_mismatch = 0
    Ys = {}
    for t in range(1, 51):
        Z1, A1, Z2, A2, LG, YY = B.replay_stack(out, "ELU", slots, t, device=dev, cifar=cifar)
        with torch.enable_grad():
            q = Z2.detach().clone().requires_grad_(True)
            G = torch.autograd.grad(F.elu(q, 1.0).sum(), q)[0].detach()      # the training derivative, as S5's gtrain
        for r_, (s, cd) in enumerate(slots):
            Y = YY[r_].cpu(); Ys[(s, t)] = Y
            r = ref[(s, t)]
            mf = float(torch.bincount(Y, minlength=10).max()) / 1200
            worst["major_frac"] = max(worst["major_frac"], abs(mf - float(r["major_frac"])))
            g = G[r_]
            G2 = float(g.double().mean()); Q2 = int((g.abs() < 1e-6).sum()) / g.numel()
            mu2 = float(A1[r_].double().mean(0).norm())
            W1 = torch.tensor(np.load(B.snapshot_path(out, "ELU", "std", s, t))["W1"]).double()
            wn1 = float(W1.norm(dim=1).mean())
            for key, mine in (("G2", G2), ("mu2_norm", mu2), ("Wnorm1", wn1)):
                worst[key] = max(worst[key], abs(mine / float(r[key]) - 1))
            g2_abs = max(g2_abs, abs(G2 - float(r["G2"])))
            with torch.enable_grad():                                             # mutation control: layer 1 in place of layer 2
                q1 = Z1[r_].detach().clone().requires_grad_(True)
                g1 = torch.autograd.grad(F.elu(q1, 1.0).sum(), q1)[0].detach()
            layer_swap_match += int(float(g1.double().mean()) == float(r["G2"]))
            worst["Q2_abs"] = max(worst["Q2_abs"], abs(Q2 - float(r["Q2"])))
    # mutation control for K4: the labels of the next seed must not reproduce this seed's major_frac everywhere
    for s in SEEDS:
        s2 = (s + 1) % 10
        if any(abs(float(torch.bincount(Ys[(s2, t)], minlength=10).max()) / 1200 - float(ref[(s, t)]["major_frac"])) > 0
               for t in range(1, 51)):
            mut_mismatch += 1
    k4 = {"max_abs_diff": worst["major_frac"], "pass": worst["major_frac"] == 0.0, "mutation_seeds_mismatching": mut_mismatch}
    k5 = {"max_rel_diff": {k: worst[k] for k in ("G2", "mu2_norm", "Wnorm1")}, "G2_max_abs_diff": g2_abs, "Q2_max_abs_diff": worst["Q2_abs"], "mutation_layer_swap_G2_matches": layer_swap_match,
          "pass": max(worst["G2"], worst["mu2_norm"], worst["Wnorm1"]) < 1e-5 and worst["Q2_abs"] <= 1e-4}
    return k4, k5


def main():
    P = {arm: rows(arm) for arm in ("A", "F", "F99")}
    res = {"K2": k2(P), "K3": k3(P)}
    res["K4"], res["K5"] = k4_k5()
    res["K6"] = "not applicable (arm S not run: the pilot gave no eta_S)"
    print(json.dumps(res, indent=1, default=str))
    (ROOT / "results" / "sink_roots_0930" / "R7gpu_checks.json").write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
