#!/usr/bin/env python3
"""sink_roots_r11b_0930 (spec_sink_roots_r11b_0930.md §4, registered in spec_sink_roots_0930.md §6.2 and §9.1).
Push = m(78) - m(0) on the task's own training images; units open at the switch = top(0) > 0 on those images; per task
the median over those units.  Tasks 2-30.  Q11b = fraction of tasks with a negative median push, per layer, arm, seed.
Predictions: (1) layer 2's fraction pooled over the 3 seeds > 0.5 for all 6 arms; (2) median |S2_new| > median |S2_cancel|
(layer-2 units open at the switch) in a majority of the pooled tasks, for all 6 arms.
Descriptions: the input kick m(new images) - m(old images) at the switch weights, the push on the fixed probe set, the signs
of the cancel and new-class parts, hard (odd) and easy (even) tasks apart, and C2 (online_acc against 0920's run)."""
import csv, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RAW = Path("/home/issan/Projects/obsidian-research-data/sink_roots_r11b_0930")
RES = ROOT / "results" / "sink_roots_r11b_0930"
ARMS = ("ELU", "GELU", "SILU", "LR", "R", "KKT1")
SEEDS = (0, 1, 2)


def one(arm, seed):
    d = RAW / f"{arm}_s{seed}"
    if not (d / ".done").exists():
        return None
    a = np.load(d / "arrays.npz", allow_pickle=True)
    grid = list(a["grid"]); g78 = grid.index(78)
    tasks = list(a["tasks_m1"]); assert tasks == list(range(1, len(tasks) + 1))
    tS = list(a["tasks_S2_new"])
    out = {"tasks": {}}
    for t in range(2, len(tasks) + 1):
        ti, si = t - 1, tS.index(t)
        r = {"hard": t % 2 == 1}
        for L in ("1", "2"):
            top0 = a["u_top" + L][ti, 0]; op = top0 > 0
            push = a["u_m" + L][ti, g78] - a["u_m" + L][ti, 0]
            pprobe = a["u_m" + L + "_probe"][ti, g78] - a["u_m" + L + "_probe"][ti, 0]
            kick = a["u_m" + L][ti, 0] - a["u_m" + L + "_old"][si, 0]
            r[f"n_open{L}"] = int(op.sum())
            r[f"push{L}"] = float(np.median(push[op])) if op.any() else float("nan")
            r[f"push_probe{L}"] = float(np.median(pprobe[op])) if op.any() else float("nan")
            r[f"kick{L}"] = float(np.median(kick))
        op2 = a["u_top2"][ti, 0] > 0
        if op2.any():
            sn, sc = a["u_S2_new"][si, 0][op2], a["u_S2_cancel"][si, 0][op2]
            r.update(S2_new_med=float(np.median(sn)), S2_cancel_med=float(np.median(sc)),
                     new_gt_cancel=bool(np.median(np.abs(sn)) > np.median(np.abs(sc))))
        out["tasks"][t] = r
    rows = json.loads((d / "rows.json").read_text())
    out["online"] = {r["task"]: r["online_acc"] for r in rows if "online_acc" in r}
    return out


def c2(arm, res):
    f = ROOT / "results" / "cifar5p1_mlp_0920" / f"{arm}_std_lr0.0001" / "per_task.csv"
    if not f.exists():
        return None
    ref = {(int(r["seed"]), int(r["task"])): float(r["online_acc"]) for r in csv.DictReader(open(f)) if r.get("online_acc")}
    diffs = [abs(res[s]["online"][t] - ref[(s, t)]) for s in res for t in res[s]["online"] if (s, t) in ref]
    return {"pairs": len(diffs), "max_abs_diff": max(diffs) if diffs else None, "median_abs_diff": float(np.median(diffs)) if diffs else None}


def main():
    L, J = [], {}
    verdict1, verdict2 = {}, {}
    for arm in ARMS:
        res = {s: one(arm, s) for s in SEEDS}
        res = {s: v for s, v in res.items() if v}
        if len(res) < 3:
            L.append(f"{arm}: {len(res)}/3 runs done"); continue
        J[arm] = {}
        pooled = {"1": [], "2": []}; ngc = []
        for s, v in res.items():
            T = v["tasks"]
            fr = {}
            for Ly in ("1", "2"):
                vals = [T[t][f"push{Ly}"] for t in T if np.isfinite(T[t][f"push{Ly}"])]
                pooled[Ly] += [x < 0 for x in vals]
                fr[Ly] = float(np.mean([x < 0 for x in vals])) if vals else float("nan")
            g = [T[t]["new_gt_cancel"] for t in T if "new_gt_cancel" in T[t]]; ngc += g
            hard = [T[t]["push2"] for t in T if T[t]["hard"] and np.isfinite(T[t]["push2"])]
            easy = [T[t]["push2"] for t in T if not T[t]["hard"] and np.isfinite(T[t]["push2"])]
            med = lambda k: float(np.nanmedian([T[t].get(k, np.nan) for t in T]))
            J[arm][s] = {"frac_neg_push1": fr["1"], "frac_neg_push2": fr["2"], "frac_new_gt_cancel": float(np.mean(g)) if g else float("nan"),
                         "push2_med_hard": float(np.median(hard)) if hard else float("nan"), "push2_med_easy": float(np.median(easy)) if easy else float("nan"),
                         **{k + "_med": med(k) for k in ("push1", "push2", "push_probe1", "push_probe2", "kick1", "kick2", "S2_new_med", "S2_cancel_med")},
                         "n_open2_med": float(np.median([T[t]["n_open2"] for t in T])),
                         "online_mean": float(np.mean(list(v["online"].values())))}
            q = J[arm][s]
            L.append(f"{arm:4s} s{s}: Q11b frac(push<0) layer 1 {q['frac_neg_push1']:.2f} layer 2 {q['frac_neg_push2']:.2f} | push median "
                     f"L1 {q['push1_med']:+.3f} L2 {q['push2_med']:+.3f} (hard {q['push2_med_hard']:+.3f}, easy {q['push2_med_easy']:+.3f}) | "
                     f"probe-set push L1 {q['push_probe1_med']:+.3f} L2 {q['push_probe2_med']:+.3f} | kick L1 {q['kick1_med']:+.3f} L2 {q['kick2_med']:+.3f} | "
                     f"S2 new {q['S2_new_med_med']:+.2e} cancel {q['S2_cancel_med_med']:+.2e} |new|>|cancel| {q['frac_new_gt_cancel']:.2f} | "
                     f"open L2 {q['n_open2_med']:.0f} | online {q['online_mean']:.3f}")
        verdict1[arm] = float(np.mean(pooled["2"])); verdict2[arm] = float(np.mean(ngc))
        J[arm]["pooled"] = {"frac_neg_push1": float(np.mean(pooled["1"])), "frac_neg_push2": verdict1[arm], "frac_new_gt_cancel": verdict2[arm],
                            "C2_vs_0920": c2(arm, res)}
        L.append(f"   {arm} pooled: frac(push<0) L1 {np.mean(pooled['1']):.2f} L2 {verdict1[arm]:.2f} | |new|>|cancel| {verdict2[arm]:.2f} | C2 {J[arm]['pooled']['C2_vs_0920']}")
    if len(verdict1) == len(ARMS):
        p1 = all(v > 0.5 for v in verdict1.values()); p2 = all(v > 0.5 for v in verdict2.values())
        J["predictions"] = {"layer2_push_negative_in_majority_all_arms": p1, "new_gt_cancel_in_majority_all_arms": p2}
        L.append(f"\nprediction 1 (layer 2 push < 0 in a majority of tasks, all 6 arms): {'HIT' if p1 else 'MISS'} {verdict1}")
        L.append(f"prediction 2 (|new-class part| > |cancel part| in a majority, all 6 arms): {'HIT' if p2 else 'MISS'} {verdict2}")
    txt = "\n".join(L); print(txt)
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "R11b_report.txt").write_text(txt + "\n")
    (RES / "R11b_report.json").write_text(json.dumps(J, indent=1, default=str))


if __name__ == "__main__":
    main()
