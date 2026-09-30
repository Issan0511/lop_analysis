#!/usr/bin/env python3
"""R7 = spec_postfit_elu_cifar_0924 §3-§5 on the CPU runs (sink_roots_0930 §4).

Reads RAW/r7/<arm>/s<seed>/{per_task.csv, snap/, trace/}; rebuilds G2, Q2, |mu2|, V_Sigma(W1), |W1|^2 from the
end-of-task snapshots on each seed's 1200 images (float64 means), the floor F_s(t) from the regenerated labels.
"""
import csv, json, math, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src import rlcifar_mlp_battle_0918 as B          # noqa: E402
from src import pmnist_rlcifar_0907 as RC             # noqa: E402

GPU = "--gpu" in sys.argv          # after the reboot: one R = 10 run per arm (r7gpu/<arm>/, all seeds in one per_task.csv)
RAW = Path("/home/issan/Projects/obsidian-research-data/sink_roots_0930/" + ("r7gpu" if GPU else "r7"))
TAG = "R7gpu" if GPU else "R7"
RES = ROOT / "results" / "sink_roots_0930"
SEEDS = list(range(10))
ARMS = ["A", "F", "F99", "S"]
T19 = 1.7291                                    # t_(19, 0.95)
T9_975 = 2.2622                                 # t_(9, 0.975)
T9_9875 = 2.6850                                # t_(9, 1 - 0.05/4)  (Bonferroni over 2)
W = range(31, 51)
torch.set_num_threads(4)


def per_task(arm, s):
    f = RAW / arm / "per_task.csv" if GPU else RAW / arm / f"s{s}" / "per_task.csv"
    if not f.exists() or (GPU and not (RAW / arm / "provenance.json").exists()):
        return None
    return {int(r["task"]): r for r in csv.DictReader(open(f)) if not GPU or int(r["seed"]) == s}


_cifar = None


def snap_stats(arm, s, t):
    """G2, Q2, |mu2|, V_Sigma(W1), |W1|^2 at the end of task t (float64 means)."""
    global _cifar
    _cifar = _cifar or RC.Cifar10()
    out = RAW / arm if GPU else RAW / arm / f"s{s}"
    if not B.snapshot_path(out, "ELU", "std", s, t).exists():
        return None
    z1, a1, z2, a2, logits, Y = B.replay(out, "ELU", "std", s, t, device=torch.device("cpu"), cifar=_cifar)
    g2 = torch.where(z2 > 0, torch.ones_like(z2), torch.exp(z2)).double()
    z1d = z1.double()
    return {"G2": float(g2.mean()), "Q2": float((g2.abs() < 1e-6).double().mean()),
            "mu2": float(a1.double().mean(0).norm()), "VS": float(z1d.var(0, unbiased=False).sum()),
            "W1sq": float(np.square(np.load(B.snapshot_path(out, "ELU", "std", s, t))["W1"].astype(np.float64)).sum()),
            "acc_replay": float((logits.argmax(-1) == Y).double().mean()),
            "floor": float(torch.bincount(Y, minlength=10).max()) / 1200}


def main():
    rows, stats = {}, {}
    for arm in ARMS:
        for s in SEEDS:
            pt = per_task(arm, s)
            if pt is None or len(pt) < 50:
                continue
            rows[(arm, s)] = pt
    have = sorted({a for a, s in rows})
    print("complete arms:", {a: sum(1 for x, s in rows if x == a) for a in have})
    # snapshot statistics: tasks 1, 2 for the Q6 ratios, every task for G2 (the E2 endpoint)
    for (arm, s), pt in rows.items():
        for t in range(1, 51):
            st = snap_stats(arm, s, t)
            if st:
                stats[(arm, s, t)] = st
    # floors from arm A's replays (labels do not depend on the arm)
    U = {}
    for s in SEEDS:
        F = [stats[("A", s, t)]["floor"] for t in W if ("A", s, t) in stats]
        if len(F) == 20:
            U[s] = float(np.mean(F) + T19 * np.std(F, ddof=1) / math.sqrt(20))
    res = {"U_s": U}
    A1 = {s: float(rows[("A", s)][1]["online_acc"]) for s in SEEDS if ("A", s) in rows}
    G1 = {s: stats[("A", s, 1)]["G2"] for s in SEEDS if ("A", s, 1) in stats}

    def arm_state(arm):
        fl = []
        for s in SEEDS:
            if (arm, s) not in rows or s not in U:
                return None
            mA = np.mean([float(rows[(arm, s)][t]["online_acc"]) for t in W])
            fl.append(mA <= U[s])
        return "FLOORED" if all(fl) else ("ALIVE" if not any(fl) else "SPLIT"), int(sum(fl))

    def E(arm, s):
        mA = np.mean([float(rows[(arm, s)][t]["online_acc"]) for t in W])
        mG = np.mean([stats[(arm, s, t)]["G2"] for t in W])
        return mA - A1[s], mG - G1[s]

    def sgn(lo, hi):
        return "+" if lo > 0 else ("-" if hi < 0 else "0")

    for arm, tq in (("F", T9_9875), ("F99", T9_975), ("S", T9_975)):
        stt = arm_state(arm)
        if stt is None or any((arm, s, t) not in stats for s in SEEDS for t in W):
            res[arm] = "INCOMPLETE"
            continue
        d1 = [E(arm, s)[0] - E("A", s)[0] for s in SEEDS]
        d2 = [E(arm, s)[1] - E("A", s)[1] for s in SEEDS]
        ci = []
        for d in (d1, d2):
            m, sd = float(np.mean(d)), float(np.std(d, ddof=1))
            ci.append((m, m - tq * sd / math.sqrt(10), m + tq * sd / math.sqrt(10), sd == 0))
        state, nfl = stt
        s1, s2 = sgn(ci[0][1], ci[0][2]), sgn(ci[1][1], ci[1][2])
        if state == "FLOORED":
            lab = "COLLAPSED"
        elif state == "SPLIT":
            lab = "SPLIT"
        elif s1 == "+":
            lab = "RESCUED" if s2 == "+" else "RESCUED_FUNCTION_ONLY"
        else:
            lab = "ALIVE_UNRESOLVED"
        # timing Q2
        def thalf(a, s):
            A1a = float(rows[(a, s)][1]["online_acc"])
            for t in range(2, 51):
                if (float(rows[(a, s)][t]["online_acc"]) - 0.1) / (A1a - 0.1) < 0.5:
                    return t
            return None
        later = earlier = 0
        for s in SEEDS:
            ta, tA = thalf(arm, s), thalf("A", s)
            if ta == tA:
                continue
            if ta is None or (tA is not None and ta > tA):
                later += 1
            else:
                earlier += 1
        n = later + earlier
        from math import comb
        p = min(1.0, 2 * sum(comb(n, k) for k in range(max(later, earlier), n + 1)) / 2 ** n) if n else 1.0
        q2 = "TIMING_UNDETERMINED" if n == 0 else ("LATER" if (p < 0.05 and later > earlier) else
                                                   ("EARLIER" if (p < 0.05 and earlier > later) else "NO_TIMING_DIFF"))
        res[arm] = {"state": state, "n_floor": nfl, "E1_delta_ci": ci[0], "E2_delta_ci": ci[1], "Q1": lab,
                    "Q2": q2, "later": later, "earlier": earlier, "p": p,
                    "T_half": {s: thalf(arm, s) for s in SEEDS}, "T_half_A": {s: thalf("A", s) for s in SEEDS}}
        if arm == "F":
            q5 = {"RESCUED": "BRIDGE_RESCUED", "RESCUED_FUNCTION_ONLY": "BRIDGE_RESCUED",
                  "ALIVE_UNRESOLVED": "BRIDGE_RESCUED", "SPLIT": "BRIDGE_PARTIAL"}.get(lab)
            if q5 is None:
                q5 = "BRIDGE_DELAYED" if q2 == "LATER" else "BRIDGE_NONE"
            res["Q5"] = q5
    # Q6 M1-M4
    M = {}
    for t in (1, 2):
        for key, nm in (("VS", "s_V"), ("mu2", "s_mu"), ("W1sq", "s_N")):
            v = [1 - stats[("F", s, t)][key] / stats[("A", s, t)][key] for s in SEEDS
                 if ("F", s, t) in stats and ("A", s, t) in stats]
            if v:
                M[f"{nm}_t{t}"] = {"median": float(np.median(v)), "min": float(min(v)), "max": float(max(v)),
                                   "n_F_lt_A": int(sum(x > 0 for x in v)), "n": len(v)}
    g = [stats[("F", s, 2)]["G2"] - stats[("A", s, 2)]["G2"] for s in SEEDS if ("F", s, 2) in stats and ("A", s, 2) in stats]
    if g:
        M["M3_G2_F_minus_A_t2"] = {"n_pos": int(sum(x > 0 for x in g)), "n": len(g), "median": float(np.median(g))}
    for arm in ("F", "F99", "S"):
        sw = []
        for t in range(1, 51):
            vals = [float(rows[(arm, s)][t]["switch_step"]) for s in SEEDS if (arm, s) in rows]
            if vals:
                sw.append(float(np.mean([0 < v <= 30000 for v in vals])))
        if sw:
            M[f"M4_switch_rate_{arm}"] = {"t1_10": sw[:10], "all_mean": float(np.mean(sw))}
    res["Q6"] = M
    # trajectories (M5) for the table
    traj = []
    for (arm, s, t), st in sorted(stats.items()):
        traj.append({"arm": arm, "seed": s, "task": t, "online": float(rows[(arm, s)][t]["online_acc"]), **st})
    if traj:
        with open(RES / f"{TAG}_trajectories.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(traj[0])); w.writeheader(); w.writerows(traj)
    (RES / f"{TAG}_verdict.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: v for k, v in res.items() if k != "U_s"}, indent=1, default=str)[:4000])


if __name__ == "__main__":
    main()
