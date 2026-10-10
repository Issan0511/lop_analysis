"""c1_direction_1010 -- apply the registered bands (spec section 3-4) to the per-state files.

Reads results/c1_direction_1010/cnn/*.npz (written by src/c1_direction_1010.py) and
results/c1_direction_1010/mlp/*.npz (analysis/c1_direction_1010/mlp_drift.py), recomputes every
registered rate directly from the channel / unit arrays (independent of the agents' own tables),
and writes verdict.json + verdict_tables.md.  Sign convention: G > 0 = sinking side.

usage: .venv/bin/python analysis/c1_direction_1010/verdict.py
"""
from __future__ import annotations
import glob, json, math, os, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CNN = ROOT / "results" / "c1_direction_1010" / "cnn"
MLP = ROOT / "results" / "c1_direction_1010" / "mlp"
OUT = ROOT / "results" / "c1_direction_1010"
STRONG, MAJ, CH_LO, CH_HI = 0.85, 0.70, 0.35, 0.65        # bands (spec section 3)
T_MAIN = (5, 10, 20)
SNAKE = ("SNA", "SNAc3", "CV06FC3", "CV3FC06")
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def rate(mask_num, mask_den=None):
    """fraction (nan if the denominator is empty)"""
    den = np.ones_like(mask_num, dtype=bool) if mask_den is None else mask_den
    n = int(den.sum())
    return (float((mask_num & den).sum()) / n if n else float("nan")), n


def spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    ra = ra - ra.mean(); rb = rb - rb.mean()
    d = math.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def seed_mean(vals):
    v = np.array([x for x in vals if x == x], float)
    if v.size == 0:
        return dict(mean=float("nan"), lo=float("nan"), hi=float("nan"), n_seed=0)
    if v.size == 1:
        return dict(mean=float(v[0]), lo=float("nan"), hi=float("nan"), n_seed=1)
    se = v.std(ddof=1) / math.sqrt(v.size)
    tq = T975.get(v.size - 1, 1.96)
    return dict(mean=float(v.mean()), lo=float(v.mean() - tq * se), hi=float(v.mean() + tq * se), n_seed=int(v.size))


def per_state_rates(d, layer):
    """registered quantities for one (arm, seed, t) and one layer ('c1' or 'c2')"""
    G = d[f"G_{layer}_full"]; G0 = d[f"G_{layer}_init"]; Gd = d[f"G_{layer}_drift"]; Gr = d[f"G_{layer}_rest"]
    Pbar = d["Pbar1"] if layer == "c1" else d["Pbar2"]
    pos, neg = Pbar > 0, Pbar <= 0
    share = np.abs(Gd) / (np.abs(G0) + np.abs(Gd) + np.abs(Gr))
    r = {}
    r["sink"], _ = rate(G > 0)
    r["drift_float_pos"], r["n_pos"] = rate(Gd < 0, pos)
    r["drift_sink_nonpos"], r["n_nonpos"] = rate(Gd > 0, neg)
    r["rest_pos"], _ = rate(Gr > 0)
    r["rest_pos_posP"], _ = rate(Gr > 0, pos)
    r["init_agree"], _ = rate(np.sign(G0) == np.sign(G))
    r["dom_drift_rest"], _ = rate(np.abs(Gd) > np.abs(Gr))
    r["dom_drift_init"], _ = rate(np.abs(Gd) > np.abs(G0))
    r["share_drift_med"] = float(np.median(share))
    r["drift_sign_rule"], _ = rate(np.sign(Gd) == -np.sign(Pbar), Pbar != 0)      # -sign(Pbar) rule, all channels
    r["sink_posP"], _ = rate(G > 0, pos)
    r["sink_nonposP"], _ = rate(G > 0, neg)
    That = d["That_c1"] if layer == "c1" else d["That_c2"]
    r["spearman_That_G"] = spearman(That, G)
    r["agree_That_G"], _ = rate(np.sign(That) == np.sign(G))
    if layer == "c1":
        r["c_pos"], _ = rate(d["c_k"] > 0)
        r["D_pos"], _ = rate(d["D2bar"] > 0)
        W, b = d["dz2_W"], d["dz2_b"]
        r["mu_path_share_med"] = float(np.median(np.abs(W) / (np.abs(W) + np.abs(b))))
    else:
        r["c_pos"], _ = rate(d["c_u"] > 0)
        r["D_pos"], _ = rate(d["D3bar"] > 0)
        W, b = d["dz3_W"], d["dz3_b"]
        r["mu_path_share_med"] = float(np.median(np.abs(W) / (np.abs(W) + np.abs(b))))
    return r


def load_cnn():
    rows = defaultdict(dict)       # (arm, layer, t) -> seed -> rates
    checks = []
    for f in sorted(glob.glob(str(CNN / "*.npz"))):
        d = np.load(f, allow_pickle=True)
        arm, seed, t = str(d["arm"]), int(d["seed"]), int(d["task"])
        if str(d["loss"]) != "L_u":
            continue
        for layer in ("c1", "c2"):
            rows[(arm, layer, t)][seed] = per_state_rates(d, layer)
        checks.append(max(float(d[k]) for k in d.files if k.startswith("chk_")))
    return rows, checks


def agg(rows, keys):
    """seed means for every rate at every (arm, layer, t)"""
    out = {}
    for (arm, layer, t), per_seed in sorted(rows.items()):
        o = {"n_seed": len(per_seed)}
        for k in keys:
            o[k] = seed_mean([v[k] for v in per_seed.values()])
        o["n_nonpos_total"] = int(sum(v["n_nonpos"] for v in per_seed.values()))
        out[f"{arm}|{layer}|{t}"] = o
    return out


def pooled(aggd, arm, layer, ts, key):
    """mean over the listed t of the seed-mean rate (equal weights); nan if any t missing"""
    vals = [aggd.get(f"{arm}|{layer}|{t}", {}).get(key, {}).get("mean", float("nan")) for t in ts]
    return float(np.mean(vals)) if all(v == v for v in vals) else float("nan")


def band(x, lo=None, hi=None):
    if x != x:
        return "pending"
    ok = True
    if lo is not None:
        ok &= x >= lo
    if hi is not None:
        ok &= x <= hi
    return "ok" if ok else "fail"


def judge_cnn(aggd, rows):
    P = {}
    # P1: LR c1, t in 5/10/20
    p1 = {"drift_float_pos": pooled(aggd, "LR", "c1", T_MAIN, "drift_float_pos"),
          "init_agree": pooled(aggd, "LR", "c1", T_MAIN, "init_agree"),
          "dom_drift_rest": pooled(aggd, "LR", "c1", T_MAIN, "dom_drift_rest"),
          "rest_pos": pooled(aggd, "LR", "c1", T_MAIN, "rest_pos")}
    p1["score"] = {"drift_float_pos>=0.85": band(p1["drift_float_pos"], STRONG),
                   "init_agree in chance band": band(p1["init_agree"], CH_LO, CH_HI),
                   "dom_drift_rest>=0.70": band(p1["dom_drift_rest"], MAJ),
                   "rest_pos>=0.60": band(p1["rest_pos"], 0.60)}
    P["P1"] = p1
    p2 = {"rest_pos": pooled(aggd, "LR", "c2", T_MAIN, "rest_pos"),
          "dom_rest_drift": 1 - pooled(aggd, "LR", "c2", T_MAIN, "dom_drift_rest"),
          "drift_float_pos": pooled(aggd, "LR", "c2", T_MAIN, "drift_float_pos")}
    p2["score"] = {"rest_pos>=0.85": band(p2["rest_pos"], STRONG),
                   "dom_rest_drift>=0.70": band(p2["dom_rest_drift"], MAJ),
                   "drift_float_pos>=0.60": band(p2["drift_float_pos"], 0.60)}
    P["P2"] = p2
    p3 = {"c_k_pos": pooled(aggd, "LR", "c1", T_MAIN, "c_pos"),
          "c_u_pos": pooled(aggd, "LR", "c2", T_MAIN, "c_pos"),
          "D2bar_pos": pooled(aggd, "LR", "c1", T_MAIN, "D_pos")}
    p3["score"] = {k + ">=0.85": band(v, STRONG) for k, v in list(p3.items())}
    P["P3"] = p3
    # P4: per seed, c1 drift share t20 > t1
    s1 = rows.get(("LR", "c1", 1), {}); s20 = rows.get(("LR", "c1", 20), {})
    common = sorted(set(s1) & set(s20))
    up = sum(s20[s]["share_drift_med"] > s1[s]["share_drift_med"] for s in common)
    P["P4"] = {"seeds_up": up, "n_seed": len(common), "score": {"t20>t1 in >=8/10": ("ok" if (len(common) >= 10 and up >= 8) else ("pending" if len(common) < 10 else "fail"))},
               "share_by_t": {t: aggd.get(f"LR|c1|{t}", {}).get("share_drift_med", {}).get("mean", float("nan")) for t in (1, 2, 5, 10, 20, 30)}}
    # P5: LR t30, channels with Pbar1 <= 0
    a30 = aggd.get("LR|c1|30", {})
    n_np = a30.get("n_nonpos_total", 0)
    v = a30.get("drift_sink_nonpos", {}).get("mean", float("nan"))
    P["P5"] = {"drift_sink_nonpos": v, "n_channels_nonpos": n_np,
               "score": {"drift_sink_nonpos>=0.70": (band(v, MAJ) if n_np >= 5 else "not judged (<5 channels)")},
               "sink_posP_t30": a30.get("sink_posP", {}).get("mean", float("nan")),
               "sink_nonposP_t30": a30.get("sink_nonposP", {}).get("mean", float("nan"))}
    # P6: Snake arms pooled
    p6 = {}
    vals_d, vals_dom = [], []
    for arm in SNAKE:
        vd = pooled(aggd, arm, "c1", T_MAIN, "drift_float_pos"); vm = pooled(aggd, arm, "c1", T_MAIN, "dom_drift_rest")
        p6[arm] = {"drift_float_pos": vd, "dom_drift_rest": vm, "sink": pooled(aggd, arm, "c1", T_MAIN, "sink")}
        vals_d.append(vd); vals_dom.append(vm)
    p6["pooled_drift_float_pos"] = float(np.mean(vals_d)) if all(v == v for v in vals_d) else float("nan")
    p6["pooled_dom_drift_rest"] = float(np.mean(vals_dom)) if all(v == v for v in vals_dom) else float("nan")
    p6["score"] = {"drift_float_pos>=0.70": band(p6["pooled_drift_float_pos"], MAJ),
                   "dom_drift_rest in chance band": band(p6["pooled_dom_drift_rest"], CH_LO, CH_HI)}
    P["P6"] = p6
    # P7: LRc
    tl = (5, 10, 20, 30)
    p7 = {"c1_sink": pooled(aggd, "LRc", "c1", tl, "sink"), "c1_drift_float_pos": pooled(aggd, "LRc", "c1", tl, "drift_float_pos"),
          "c2_sink": pooled(aggd, "LRc", "c2", tl, "sink")}
    p7["score"] = {"c1_sink<=0.40": band(p7["c1_sink"], None, 0.40), "c1_drift_float_pos>=0.85": band(p7["c1_drift_float_pos"], STRONG),
                   "c2_sink>=0.85": band(p7["c2_sink"], STRONG)}
    P["P7"] = p7
    # Q1 label
    s = P["P1"]["score"]; s2 = P["P2"]["score"]
    if any(v == "pending" for v in list(s.values()) + list(s2.values())):
        q1 = "pending"
    elif s["drift_float_pos>=0.85"] == "ok" and s["dom_drift_rest>=0.70"] == "ok" and s["init_agree in chance band"] == "ok" \
            and s2["rest_pos>=0.85"] == "ok" and s2["dom_rest_drift>=0.70"] == "ok":
        q1 = "DRIFT_EXPLAINS"
    elif P["P1"]["drift_float_pos"] < MAJ or P["P1"]["init_agree"] > MAJ:
        q1 = "NOT_DRIFT"
    else:
        q1 = "DRIFT_SIGN_ONLY"
    return P, q1


def judge_mlp():
    rows = defaultdict(dict)
    for f in sorted(glob.glob(str(MLP / "*.npz"))):
        d = np.load(f, allow_pickle=True)
        cond, seed, t = str(d["cond"]), int(d["seed"]), int(d["t"])
        alive = d["alive"].astype(bool); h = d["hbar1"]; G, Gd, Gr, G0 = d["G"], d["Gd"], d["Gr"], d["G0"]
        r = {}
        r["sink"], _ = rate(G > 0, alive)
        r["drift_sink_hneg"], r["n_hneg"] = rate(Gd > 0, alive & (h < 0))
        r["drift_float_hpos"], r["n_hpos"] = rate(Gd < 0, alive & (h > 0))
        r["rest_pos"], _ = rate(Gr > 0, alive)
        r["dom_drift_rest"], _ = rate(np.abs(Gd) > np.abs(Gr), alive)
        r["init_agree"], _ = rate(np.sign(G0) == np.sign(G), alive)
        r["c_pos"], _ = rate(d["c"] > 0)
        r["D_pos"], _ = rate(d["Dbar"] > 0)
        rows[(cond, t)][seed] = r
    keys = ["sink", "drift_sink_hneg", "drift_float_hpos", "rest_pos", "dom_drift_rest", "init_agree", "c_pos", "D_pos"]
    aggd = {}
    for (cond, t), per_seed in sorted(rows.items()):
        aggd[f"{cond}|{t}"] = {k: seed_mean([v[k] for v in per_seed.values()]) for k in keys}
    ts = (5, 10, 20, 50)
    def pool(cond, key):
        vals = [aggd.get(f"{cond}|{t}", {}).get(key, {}).get("mean", float("nan")) for t in ts]
        return float(np.mean(vals)) if all(v == v for v in vals) else float("nan")
    p8 = {"raw_drift_sink_hneg": pool("raw", "drift_sink_hneg"), "raw_sink": pool("raw", "sink"),
          "std_drift_float_hpos": pool("std", "drift_float_hpos"), "std_rest_pos": pool("std", "rest_pos"), "std_sink": pool("std", "sink")}
    p8["score"] = {"raw_drift_sink_hneg>=0.85": band(p8["raw_drift_sink_hneg"], STRONG),
                   "raw_sink in 0.84-0.97": band(p8["raw_sink"], 0.84, 0.97),
                   "std_drift_float_hpos>=0.70": band(p8["std_drift_float_hpos"], MAJ),
                   "std_rest_pos>=0.70": band(p8["std_rest_pos"], MAJ),
                   "std_sink in chance band": band(p8["std_sink"], CH_LO, CH_HI)}
    ok_raw = p8["score"]["raw_drift_sink_hneg>=0.85"] == "ok"; ok_std = p8["score"]["std_drift_float_hpos>=0.70"] == "ok"
    q3 = "pending" if "pending" in p8["score"].values() else ("SIGN_RULE_HOLDS" if ok_raw and ok_std else ("PARTIAL" if ok_raw or ok_std else "NOT"))
    return aggd, p8, q3


def fmt(x):
    return "—" if x != x else f"{x:.2f}"


def main():
    rows, checks = load_cnn()
    keys = ["sink", "drift_float_pos", "drift_sink_nonpos", "rest_pos", "init_agree", "dom_drift_rest", "dom_drift_init",
            "share_drift_med", "drift_sign_rule", "sink_posP", "sink_nonposP", "spearman_That_G", "agree_That_G", "c_pos", "D_pos", "mu_path_share_med"]
    aggd = agg(rows, keys)
    P, q1 = judge_cnn(aggd, rows)
    mlp_agg, p8, q3 = judge_mlp()
    out = {"generated": __import__("datetime").datetime.now().isoformat(timespec="minutes"),
           "n_cnn_states": len(checks), "max_check_error": (max(checks) if checks else None),
           "Q1": q1, "Q3": q3, "P": {**P, "P8": p8}, "cnn": aggd, "mlp": mlp_agg,
           "bands": {"strong": STRONG, "majority": MAJ, "chance": [CH_LO, CH_HI]}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "verdict.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float))
    # markdown tables
    L = [f"# c1_direction_1010 verdict tables ({out['generated']}; {len(checks)} CNN states, max check error {out['max_check_error']})", "",
         f"Q1 = **{q1}**, Q3 = **{q3}**. Sign convention: G > 0 = sinking side. Cells: seed mean [95% t-interval].", ""]
    for arm in ["LR", "LRc"] + list(SNAKE):
        for layer in ("c1", "c2"):
            ts = sorted({t for (a, l, t) in rows if a == arm and l == layer})
            if not ts:
                continue
            L += [f"## {arm} {layer}", "", "| quantity | " + " | ".join(f"t{t}" for t in ts) + " |", "|---|" + "---|" * len(ts)]
            for k in keys:
                cells = []
                for t in ts:
                    o = aggd[f"{arm}|{layer}|{t}"][k]
                    cells.append(f"{fmt(o['mean'])} [{fmt(o['lo'])}, {fmt(o['hi'])}]" if o["n_seed"] > 1 else fmt(o["mean"]))
                L.append(f"| {k} | " + " | ".join(cells) + " |")
            L.append("| n channels with Pbar<=0 | " + " | ".join(str(aggd[f'{arm}|{layer}|{t}']['n_nonpos_total']) for t in ts) + " |")
            L.append("")
    L += ["## predictions", "", "```", json.dumps({k: v for k, v in out["P"].items()}, indent=1, ensure_ascii=False, default=float), "```"]
    (OUT / "verdict_tables.md").write_text("\n".join(L))
    print(f"Q1 = {q1}   Q3 = {q3}   states {len(checks)}  max check {out['max_check_error']}")
    for k, v in out["P"].items():
        print(k, json.dumps(v.get("score", {}), ensure_ascii=False), {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk != "score" and not isinstance(vv, dict)})


if __name__ == "__main__":
    main()
