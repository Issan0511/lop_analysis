#!/usr/bin/env python3
"""Endpoints and verdicts of spec_sgd_postfit_cifar_0923 §3-§4 (v2).

    python3 analysis/sgd_postfit_cifar_0923/analyze.py [--root results/sgd_postfit_cifar_0923]

Reads <root>/<arm>/per_task.csv and trace/LR_std_seed<s>.npz for the arms A, AR, F, S_lo, S_hi,
A_ce (whichever exist) and <root>/xfork/xfork.csv, and writes verdict.json, per_seed.csv,
curves.csv and xfork_effects.csv into <root>.

Primary rule for a diverged slot (§4): every later task counts as hit 30,000 and G = +inf.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
ARMS = ("A", "AR", "F", "S_lo", "S_hi", "A_ce")
SEEDS = tuple(range(10))
STEPS = 30000
LATE = range(41, 51)            # E1 window
EARLY = range(1, 6)
MID = range(26, 51)             # manipulation checks, e-folds, phase ledger
STOP_H_SEEDS04 = 2100.0         # LR_iid_stop, seeds 0-4: median of the t41-50 mean hit999
CURVE_T = (1, 2, 3, 5, 10, 20, 30, 40, 45, 50)
N_BOOT = 10000


def load(root: Path, arm: str) -> dict:
    """Per seed, per task records; a diverged slot's later tasks are {"dead": True}."""
    d = root / arm
    rows = list(csv.DictReader(open(d / "per_task.csv")))
    out = {}
    for s in SEEDS:
        z = np.load(d / "trace" / f"LR_std_seed{s}.npz")
        tr = {k: z[k] for k in z.files}
        mine = {int(q["task"]): q for q in rows if int(q["seed"]) == s}
        per = {}
        for t in range(1, 51):
            q = mine.get(t)
            if q is None or q.get("memo_acc") in (None, ""):
                per[t] = {"dead": True}
                continue
            m = tr["task"] == t
            step = tr["step"][m]
            col = {k: tr[k][m] for k in ("correct", "ce", "ce_st", "n1", "n2", "n3") if k in tr}

            def at(k, st):
                w = np.nonzero(step == st)[0]
                return float(col[k][w[0]]) if len(w) else float("nan")

            h = int(q["hit999"])
            h99 = int(q["hit99"])
            full = np.nonzero((col["correct"] >= 1200) & (step > 0))[0]
            sw, pin = int(q["switch_step"]), int(q.get("pin_step", -1) or -1)
            rec = {"dead": False, "hit999": h, "hit_c": (h if h >= 0 else STEPS),
                   "hit99_c": (h99 if h99 >= 0 else STEPS),
                   "hitfull_c": (int(step[full[0]]) if len(full) else STEPS),
                   "censored": h < 0, "switch": sw, "pin": pin,
                   "correct_end": at("correct", STEPS)}
            for k in ("n1", "n2", "n3"):
                rec[f"{k}_0"], rec[f"{k}_end"] = at(k, 0), at(k, STEPS)
                rec[f"{k}_200"] = at(k, min(200, h)) if h >= 0 else at(k, 200)
                rec[f"{k}_hit"] = at(k, h) if h >= 0 else float("nan")
                rec[f"{k}_sw"] = at(k, sw) if sw >= 0 else float("nan")
            if sw >= 0:
                i = int(np.nonzero(step == sw)[0][0])
                ce_sw, ce_end = at("ce_st", sw), at("ce_st", STEPS)
                ce_min = float(col["ce_st"][i:].min())
                rec.update({"ce_sw": ce_sw, "ce_end": ce_end, "correct_sw": at("correct", sw),
                            "efolds": (math.log(ce_sw / ce_end) if ce_end > 0 else math.inf),
                            # to the lowest CE of the segment: Adam's float32-floor burst lifts
                            # the task-end CE back up, so `efolds` undercounts Adam's descent
                            "efolds_max": (math.log(ce_sw / ce_min) if ce_min > 0 else math.inf),
                            "post_updates": STEPS - sw,
                            "min_correct_after_sw": (int(col["correct"][i + 1:].min())
                                                     if i + 1 < len(step) else -1)})
                for k in (1, 2, 3):
                    rec[f"disp_l{k}"] = float(q[f"pf_disp_l{k}"])
            per[t] = rec
        out[s] = per
    return out


def nanmed(v):
    v = [x for x in v if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.median(v)) if v else float("nan")


def seed_stats(per: dict) -> dict:
    """Failure-inclusive H and G (dead task -> 30,000; dead at t50 -> G = inf), plus the
    survivor-only versions and the descriptive numbers over t26-50."""
    dead = any(per[t]["dead"] for t in range(1, 51))
    hit = {t: (STEPS if per[t]["dead"] else per[t]["hit_c"]) for t in range(1, 51)}
    q = {"dead": dead,
         "first_dead_task": min((t for t in range(1, 51) if per[t]["dead"]), default=-1),
         "H": float(np.mean([hit[t] for t in LATE])),
         "H_early": float(np.mean([hit[t] for t in EARLY])),
         "H_censored": sum(1 for t in LATE if not per[t]["dead"] and per[t]["censored"]),
         "G": (math.inf if per[50]["dead"] else per[50]["n1_end"])}
    q["H_minus_early"] = q["H"] - q["H_early"]
    q["H99"] = float(np.mean([STEPS if per[t]["dead"] else per[t]["hit99_c"] for t in LATE]))
    q["Hfull"] = float(np.mean([STEPS if per[t]["dead"] else per[t]["hitfull_c"] for t in LATE]))
    live = [per[t] for t in MID if not per[t]["dead"]]
    sw = [r for r in live if r["switch"] >= 0]
    q.update({
        "G2": (math.inf if per[50]["dead"] else per[50]["n2_end"]),
        "G3": (math.inf if per[50]["dead"] else per[50]["n3_end"]),
        "efolds_mid": nanmed([r["efolds"] for r in sw]),
        "efolds_max_mid": nanmed([r["efolds_max"] for r in sw]),
        "dn1_post_mid": nanmed([r["n1_end"] - r["n1_sw"] for r in sw]),
        "disp_l1_mid": nanmed([r["disp_l1"] for r in sw]),
        "min_correct_after_sw_mid": nanmed([r["min_correct_after_sw"] for r in sw]),
        "correct_sw_mid": nanmed([r["correct_sw"] for r in sw]),
        "post_updates_mid": nanmed([r["post_updates"] for r in sw]),
        "n_noswitch": sum(1 for t in range(1, 51) if not per[t]["dead"] and per[t]["switch"] < 0),
        "n_pinned_mid": sum(1 for r in sw if r["pin"] >= 0),
        "n_switched_mid": len(sw),
        "slope_n1_mid": (float(np.polyfit([t for t in MID if not per[t]["dead"]],
                                          [per[t]["n1_end"] for t in MID if not per[t]["dead"]],
                                          1)[0]) if len(live) >= 5 else float("nan"))})
    led = {k: [] for k in ("shock", "fit", "hold", "post", "total")}
    for r in sw:
        if r["hit999"] < 0:
            continue
        led["shock"].append(r["n1_200"] - r["n1_0"])
        led["fit"].append(r["n1_hit"] - r["n1_200"])
        led["hold"].append(r["n1_sw"] - r["n1_hit"])
        led["post"].append(r["n1_end"] - r["n1_sw"])
        led["total"].append(r["n1_end"] - r["n1_0"])
    for k, v in led.items():
        q[f"led_{k}"] = float(np.mean(v)) if v else float("nan")
    return q


def ratio(a, x, f):
    den = a - f
    if not (den > 0) or math.isinf(a) or math.isinf(f):
        return float("nan")
    return (a - x) / den if not math.isinf(x) else -math.inf


def classify(r, names):
    if math.isnan(r):
        return "UNDEFINED"
    return names[0] if r >= 0.75 else names[2] if r <= 0.25 else names[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(REPO / "results/sgd_postfit_cifar_0923"))
    a = ap.parse_args()
    root = Path(a.root)
    arms = [x for x in ARMS if (root / x / "provenance.json").exists()]
    data = {x: load(root, x) for x in arms}
    st = {x: {s: seed_stats(data[x][s]) for s in SEEDS} for x in arms}

    def med(x, k, seeds=SEEDS, alive_only=False):
        return float(np.median([st[x][s][k] for s in seeds
                                if not (alive_only and st[x][s]["dead"])] or [float("nan")]))

    v = {"arms": arms,
         "n_dead": {x: sum(st[x][s]["dead"] for s in SEEDS) for x in arms},
         "first_dead_task": {x: [st[x][s]["first_dead_task"] for s in SEEDS] for x in arms}}
    for k in ("H", "G", "H_early", "H_minus_early", "H99", "Hfull", "G2", "G3", "efolds_mid",
              "efolds_max_mid",
              "dn1_post_mid", "disp_l1_mid", "min_correct_after_sw_mid", "correct_sw_mid",
              "post_updates_mid", "slope_n1_mid", "H_censored", "n_noswitch"):
        v[k] = {x: med(x, k) for x in arms}
    v["H_survivors"] = {x: med(x, "H", alive_only=True) for x in arms}
    v["G_survivors"] = {x: med(x, "G", alive_only=True) for x in arms}
    v["ledger_mid"] = {x: {k: med(x, f"led_{k}") for k in ("shock", "fit", "hold", "post",
                                                            "total")} for x in arms}
    verdict = {}
    if "A" in arms and "F" in arms:
        H, G = v["H"], v["G"]
        v["ref_gap"] = {"H": H["A"] - H["F"], "H_ok": H["A"] - H["F"] >= 600,
                        "G": G["A"] - G["F"], "G_ok": G["A"] - G["F"] >= 10000}
        v["rho"] = {x: ratio(H["A"], H[x], H["F"]) for x in arms}
        v["rhoW"] = {x: ratio(G["A"], G[x], G["F"]) for x in arms}
        v["rho_H99"] = {x: ratio(v["H99"]["A"], v["H99"][x], v["H99"]["F"]) for x in arms}
        v["rho_Hfull"] = {x: ratio(v["Hfull"]["A"], v["Hfull"][x], v["Hfull"]["F"]) for x in arms}
        v["rho_H_minus_early"] = {x: ratio(v["H_minus_early"]["A"], v["H_minus_early"][x],
                                           v["H_minus_early"]["F"]) for x in arms}
        # per seed, seed-paired
        v["rho_per_seed"] = {x: [ratio(st["A"][s]["H"], st[x][s]["H"], st["F"][s]["H"])
                                 for s in SEEDS] for x in arms}
        v["rho_per_seed_median"] = {x: nanmed(v["rho_per_seed"][x]) for x in arms}
        v["rho_per_seed_counts"] = {
            x: {"ge_0.75": sum(1 for r in v["rho_per_seed"][x] if r >= 0.75),
                "mid": sum(1 for r in v["rho_per_seed"][x] if 0.25 < r < 0.75),
                "le_0.25": sum(1 for r in v["rho_per_seed"][x] if r <= 0.25),
                "undefined": sum(1 for r in v["rho_per_seed"][x] if math.isnan(r))}
            for x in arms}
        # paired seed bootstrap of the ratio of medians
        rng = np.random.default_rng(20260923)
        Hm = {x: np.array([st[x][s]["H"] for s in SEEDS]) for x in arms}
        Gm = {x: np.array([st[x][s]["G"] for s in SEEDS]) for x in arms}
        boot = {x: {"rho": [], "rhoW": []} for x in arms}
        n_undef = {x: {"rho": 0, "rhoW": 0} for x in arms}
        for _ in range(N_BOOT):
            idx = rng.integers(0, 10, 10)
            hA, hF = np.median(Hm["A"][idx]), np.median(Hm["F"][idx])
            gA, gF = np.median(Gm["A"][idx]), np.median(Gm["F"][idx])
            for x in arms:
                for key, (a_, f_, arr) in (("rho", (hA, hF, Hm[x])), ("rhoW", (gA, gF, Gm[x]))):
                    r = ratio(a_, float(np.median(arr[idx])), f_)
                    if math.isnan(r):
                        n_undef[x][key] += 1
                    else:
                        boot[x][key].append(r)
        v["boot90"] = {x: {key: ([float(np.quantile(b, 0.05)), float(np.quantile(b, 0.95))]
                                 if b else None) for key, b in boot[x].items()} for x in arms}
        v["boot_undefined_frac"] = {x: {k: n / N_BOOT for k, n in n_undef[x].items()}
                                    for x in arms}

        def q_arm(x, key, names):
            if x not in arms:
                return "MISSING"
            if not v["ref_gap"]["H_ok" if key == "rho" else "G_ok"]:
                return "REF_GAP_SMALL"
            return classify(v[key][x], names)

        verdict["Q1_S_hi"] = q_arm("S_hi", "rho", ("SGD_RESCUE", "PARTIAL", "NO_RESCUE"))
        verdict["Q2_S_lo"] = q_arm("S_lo", "rho", ("SGD_RESCUE", "PARTIAL", "NO_RESCUE"))
        verdict["Q3_S_hi_W"] = q_arm("S_hi", "rhoW", ("SGD_RESCUE_W", "PARTIAL_W", "NO_RESCUE_W"))
        verdict["Q4_AR"] = q_arm("AR", "rho", ("RESET_RESCUE", "PARTIAL", "NO_RESCUE"))
        verdict["Q5_AR_W"] = q_arm("AR", "rhoW", ("RESET_RESCUE_W", "PARTIAL_W", "NO_RESCUE_W"))
        v["H_F_seeds04"] = med("F", "H", seeds=range(5))
        verdict["Q6_F_minus_stop_seeds04"] = v["H_F_seeds04"] - STOP_H_SEEDS04
        if "A_ce" in arms and "S_hi" in arms and v["ref_gap"]["H_ok"]:
            rs, ra = v["rho"]["S_hi"], v["rho"]["A_ce"]
            if rs >= 0.75 and ra >= 0.75:
                q7 = "MATCHED_CE_HARMLESS"
            elif rs <= 0.25 and ra <= 0.25:
                q7 = "MATCHED_CE_HARMFUL"
            elif rs - ra >= 0.5:
                q7 = "ADAM_WORSE_AT_MATCHED_CE"
            elif ra - rs >= 0.5:
                q7 = "SGD_WORSE_AT_MATCHED_CE"
            else:
                q7 = "MIXED"
            n_sw = sum(r["switch"] >= 0 for s in SEEDS for t in MID
                       for r in [data["A_ce"][s][t]] if not r["dead"])
            n_pin = sum(r["pin"] >= 0 for s in SEEDS for t in MID
                        for r in [data["A_ce"][s][t]] if not r["dead"])
            v["A_ce_match"] = {"efolds_mid_A_ce": v["efolds_mid"]["A_ce"],
                               "efolds_mid_S_hi": v["efolds_mid"]["S_hi"],
                               "pinned_frac_mid": n_pin / max(n_sw, 1),
                               "match_ok": n_pin / max(n_sw, 1) >= 0.8}
            verdict["Q7_matched_ce"] = q7 + ("" if v["A_ce_match"]["match_ok"] else
                                             "+MATCH_INSUFFICIENT")
    xf = root / "xfork" / "xfork.csv"
    if xf.exists():
        v["xfork"] = xfork_effects(xf, root)
        verdict["Q8_xfork"] = {k: v["xfork"][k]["present"] for k in
                               ("state_at_W_end", "state_at_W_sw", "weights_at_S_sw",
                                "weights_at_S_end")}
    v["verdict"] = verdict
    (root / "verdict.json").write_text(json.dumps(v, indent=2, default=float) + "\n")
    keys = sorted({k for x in arms for s in SEEDS for k in st[x][s]})
    with open(root / "per_seed.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["arm", "seed"] + keys)
        w.writeheader()
        for x in arms:
            for s in SEEDS:
                w.writerow({"arm": x, "seed": s, **st[x][s]})
    with open(root / "curves.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "task", "hit999_med", "n1_start_med", "n1_end_med", "n1_sw_med",
                    "efolds_med", "dn1_post_med", "min_correct_after_sw_min", "n_dead",
                    "n_noswitch"])
        for x in arms:
            for t in range(1, 51):
                recs = [data[x][s][t] for s in SEEDS]
                live = [r for r in recs if not r["dead"]]
                w.writerow([x, t, nanmed([STEPS if r["dead"] else r["hit_c"] for r in recs]),
                            nanmed([r["n1_0"] for r in live]), nanmed([r["n1_end"] for r in live]),
                            nanmed([r["n1_sw"] for r in live]),
                            nanmed([r.get("efolds", float("nan")) for r in live]),
                            nanmed([r["n1_end"] - r["n1_sw"] for r in live if r["switch"] >= 0]),
                            min((r["min_correct_after_sw"] for r in live
                                 if "min_correct_after_sw" in r), default=-1),
                            len(recs) - len(live), sum(1 for r in live if r["switch"] < 0)])
    show = ("n_dead", "H", "G", "H_survivors", "H_early", "efolds_mid", "efolds_max_mid",
            "dn1_post_mid",
            "ref_gap", "rho", "rhoW", "rho_per_seed_median", "rho_per_seed_counts", "boot90",
            "ledger_mid", "A_ce_match", "verdict")
    for k in show:
        if k in v:
            print(k, json.dumps(v[k], default=float))
    for t in CURVE_T:
        print(t, {x: nanmed([STEPS if data[x][s][t]["dead"] else data[x][s][t]["hit_c"]
                             for s in SEEDS]) for x in arms})


def xfork_effects(path: Path, root: Path) -> dict:
    """§2b: paired differences over (t, seed); seed-resampled 90% intervals (t kept whole)."""
    rows = list(csv.DictReader(open(path)))
    hit = {}
    for q in rows:
        h = int(q["hit999"]) if q["hit999"] not in ("", None) else -1
        hit[(int(q["t"]), int(q["seed"]), q["weights"], q["state"])] = h if h >= 0 else STEPS
    ts = sorted({int(q["t"]) for q in rows})
    defs = {"state_at_W_end": (("end", "end"), ("end", "sw")),
            "state_at_W_sw": (("sw", "end"), ("sw", "sw")),
            "weights_at_S_sw": (("end", "sw"), ("sw", "sw")),
            "weights_at_S_end": (("end", "end"), ("sw", "end"))}
    out = {}
    rng = np.random.default_rng(923)
    with open(root / "xfork_effects.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["effect", "t", "seed", "diff"])
        for name, (hi, lo) in defs.items():
            d = np.array([[hit[(t, s) + hi] - hit[(t, s) + lo] for s in SEEDS] for t in ts],
                         dtype=float)
            for i, t in enumerate(ts):
                for s in SEEDS:
                    w.writerow([name, t, s, d[i, s]])
            means = [float(d[:, rng.integers(0, 10, 10)].mean()) for _ in range(N_BOOT)]
            lo90, hi90 = float(np.quantile(means, 0.05)), float(np.quantile(means, 0.95))
            out[name] = {"mean": float(d.mean()), "median": float(np.median(d)),
                         "n_pos": int((d > 0).sum()), "n_neg": int((d < 0).sum()),
                         "n": int(d.size), "boot90_mean": [lo90, hi90],
                         "present": not (lo90 <= 0 <= hi90),
                         "by_t_mean": {int(t): float(d[i].mean()) for i, t in enumerate(ts)}}
    return out


if __name__ == "__main__":
    main()
