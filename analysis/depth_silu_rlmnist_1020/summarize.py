#!/usr/bin/env python3
"""Tables, prediction scores and the registered judgments of depth_silu_rlmnist_1020 (spec sections 7-8).

    python analysis/depth_silu_rlmnist_1020/summarize.py --rows raw/depth_silu_rlmnist_1020/units --out results/depth_silu_rlmnist_1020

Writes tables.md (every table, generated), scores.json, and the CSVs table_layers.csv (arm x seed x layer x task),
table_decomp.csv (arm x seed x layer x task x window), table_returns.csv, table_learning.csv.
Registered numbers use seeds 0-2 at task 30; the 10-seed columns are the extension (report only).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ARMS = {("SILU", 1, "none"): "A1", ("SILU", 2, "none"): "A2", ("SILU", 4, "none"): "A4",
        ("R", 1, "none"): "R1", ("R", 4, "none"): "R4", ("SILU", 4, "C"): "A4C"}
ORDER = ["A1", "A2", "A4", "R1", "R4", "A4C"]
TASKS = [1, 5, 10, 20, 30, 50]
REG = (0, 1, 2)
BLOCKS = {"課題 1": range(1, 2), "2〜5": range(2, 6), "6〜10": range(6, 11), "11〜30": range(11, 31), "31〜50": range(31, 51)}


def load(rows_dir: Path):
    runs = {}
    for f in sorted(rows_dir.glob("rows_*.json")):
        r = json.loads(f.read_text())
        arm = ARMS[(r["act"], r["L"], r["door"])]
        runs[(arm, r["seed"])] = r
    return runs


def fmt(x, nd=2):
    if x is None:
        return "—"
    if isinstance(x, str):
        return x
    if not np.isfinite(x):
        return "∞" if x == float("inf") else "—"
    if abs(x) >= 1000:
        return f"{x:.3g}"
    return f"{x:.{nd}f}"


def per_index(r):
    return {(row["task"], row["layer"]): row for row in r["per"]}


def seedvals(runs, arm, seeds, f):
    out = []
    for s in seeds:
        if (arm, s) in runs:
            v = f(runs[(arm, s)])
            if v is not None:
                out.append(v)
    return out


def mean_rng(vals, nd=2):
    v = [x for x in vals if x is not None and np.isfinite(x)]
    if not v:
        return "—"
    if len(v) == 1:
        return fmt(v[0], nd)
    return f"{fmt(float(np.mean(v)), nd)} [{fmt(min(v), nd)}, {fmt(max(v), nd)}]"


def mval(vals):
    v = [x for x in vals if x is not None and np.isfinite(x)]
    return float(np.mean(v)) if v else float("nan")


def layer_val(runs, arm, seeds, task, layer, key):
    return seedvals(runs, arm, seeds, lambda r: per_index(r).get((task, layer), {}).get(key))


def n_layers(arm):
    return {"A1": 1, "A2": 2, "A4": 4, "R1": 1, "R4": 4, "A4C": 4}[arm]


def returns_pooled(runs, arm, seeds, layer, upto, key):
    v = seedvals(runs, arm, seeds, lambda r: r["returns"].get(f"l{layer}_t{upto}", {}).get(key))
    return int(sum(v)) if v else None, 100 * len(v)


def block_mean(runs, arm, seeds, layer, w, key, tasks):
    vals = []
    for s in seeds:
        if (arm, s) not in runs:
            continue
        idx = per_index(runs[(arm, s)])
        xs = [idx[(t, layer)].get(f"{w}_{key}") for t in tasks if (t, layer) in idx]
        xs = [x for x in xs if x is not None and np.isfinite(x)]
        if xs:
            vals.append(float(np.mean(xs)))
    return vals


def score(runs, seeds=REG):
    """Every registered prediction and judgment (spec 7.2 operationalisation, 7.3, 8)."""
    S = {}
    rv = lambda arm, l, t=30: mval(layer_val(runs, arm, seeds, t, l, "r_med"))
    sv = lambda arm, l, t=30: mval(layer_val(runs, arm, seeds, t, l, "s_med"))
    cv = lambda arm, l, t: mval(layer_val(runs, arm, seeds, t, l, "c"))
    G = lambda arm: mval(seedvals(runs, arm, seeds, lambda r: r["learn"]["G30"]))
    def ret(arm, l, key="plain_units", upto=30):
        v = returns_pooled(runs, arm, seeds, l, upto, key)[0]
        return float("nan") if v is None else v
    r1A4, r1A1 = rv("A4", 1), rv("A1", 1)
    S["values"] = {"r_t30": {arm: {l: rv(arm, l) for l in range(1, n_layers(arm) + 1)} for arm in ORDER},
                   "s_t30": {arm: {l: sv(arm, l) for l in range(1, n_layers(arm) + 1)} for arm in ORDER},
                   "G30": {arm: G(arm) for arm in ORDER},
                   "c2_A4": {"t1": cv("A4", 2, 1), "t30": cv("A4", 2, 30)}}
    # --- Fable P1-P6
    S["P1a"] = bool(r1A4 <= -2.5 and all(rv("A4", l) >= -1.5 for l in (2, 3, 4)))
    S["P1b"] = bool(cv("A4", 2, 30) <= 0.5 * cv("A4", 2, 1))
    S["P2a"] = bool(abs(r1A4 - r1A1) <= 0.5)
    nU = 100 * len([x for x in seeds if ("A4", x) in runs])          # units per layer over the seeds present
    S["P2b"] = bool(ret("A4", 1) <= 0.01 * nU)
    up_plain = sum(ret("A4", l) for l in (2, 3, 4)); up_deep = sum(ret("A4", l, "deep_units") for l in (2, 3, 4))
    S["P3"] = bool(up_plain >= 0.05 * 3 * nU)
    U = {l: mval(block_mean(runs, "A4", seeds, l, "sw", "U_abs", range(2, 31))) for l in (1, 2, 3, 4)}
    S["P4_layer1_own_identity"] = bool(U[1] == 0.0)
    S["P4"] = bool(all(U[l] >= 0.5 for l in (2, 3, 4)))
    S["P5a"] = bool(G("A4") >= 0.7 * G("A1"))
    r4_floor = seedvals(runs, "R4", seeds, lambda r: (r["learn"]["floor_21_30"] or [None])[0])
    r4_ret = sum(ret("R4", l) for l in (1, 2, 3, 4))
    S["P5b"] = bool(len(r4_floor) == len(seeds) and all(r4_floor) and r4_ret == 0)
    rA4C = {l: rv("A4C", l) for l in (1, 2, 3, 4)}
    deepest = min(rA4C, key=lambda l: rA4C[l] if np.isfinite(rA4C[l]) else np.inf)
    S["P6a"] = bool(rA4C[1] >= -1.0)
    S["P6b"] = bool(deepest in (2, 3, 4) and rA4C[2] <= rA4C[1] - 0.5)
    S["P6c"] = bool(ret("A4C", 2) >= 0.05 * nU)
    S["P6d"] = bool(G("A4C") <= 0.5 * G("A4"))
    # --- Opus O1-O8
    cls = []
    for s in seeds:
        if ("A4", s) not in runs:
            continue
        idx = per_index(runs[("A4", s)])
        for l in (2, 3, 4):
            row = idx[(30, l)]
            deep, thin = row["r_med"] <= -2.0, row["s_med"] < 1.0
            cls.append({"seed": s, "layer": l, "r": row["r_med"], "s": row["s_med"], "deep": bool(deep), "thin": bool(thin)})
    S["O1_classes"] = cls
    S["O1"] = bool(sum(1 for c in cls if not (c["deep"] or c["thin"])) <= 3)
    l4 = [c for c in cls if c["layer"] == 4]
    S["O1b"] = bool(sum(1 for c in l4 if c["r"] <= -2.0) >= 2)
    S["O1c"] = not S["P1b"]
    S["O2a"], S["O2b"] = S["P2a"], S["P2b"]
    S["O3a"] = bool(up_plain < 0.05 * 3 * nU); S["O3b"] = bool(up_deep < 0.05 * 3 * nU)
    S["O4a"] = S["P4"]
    neg = {}
    for l in (2, 3, 4):
        fr = []
        for t in range(2, 31):
            v = mval([x + y for x, y in zip(layer_val(runs, "A4", seeds, t, l, "sw_up_mean"), layer_val(runs, "A4", seeds, t, l, "sw_cross_mean"))])
            fr.append(v < 0)
        neg[l] = float(np.mean(fr))
    S["O4b_frac_switches_upstream_down"] = neg
    S["O4b"] = bool(all(neg[l] >= 2 / 3 for l in (2, 3, 4)))
    S["O5a"] = S["P5a"]; S["O5b"] = bool(G("A4") >= G("A1")); S["O5c"] = S["P5b"]
    S["O6a"] = S["P6a"]; S["O6b"] = S["P6b"]; S["O6c"] = not S["P6c"]; S["O6d"] = not S["P6d"]
    S["counts"] = {"A4_l1_plain_t30": ret("A4", 1), "A4_l234_plain_t30": up_plain, "A4_l234_deep_t30": up_deep,
                   "A4C_l2_plain_t30": ret("A4C", 2), "R4_all_plain_t30": r4_ret, "R4_floor_21_30": r4_floor,
                   "U_abs_sw_2_30": U, "A4C_deepest_layer": deepest}
    # --- Issa Q1
    rnet = {arm: seedvals(runs, arm, seeds, lambda r: r["r_net"].get("30")) for arm in ("A1", "A2", "A4")}
    m = {arm: mval(v) for arm, v in rnet.items()}
    diffs = [a4 - a1 for a4, a1 in zip(rnet["A4"], rnet["A1"])]
    if m["A1"] < m["A2"] < m["A4"] and len(diffs) == len(seeds) and all(d >= 0.5 for d in diffs):
        q1 = "DEEPER_SINKS_LESS"
    elif m["A1"] > m["A2"] > m["A4"] and len(diffs) == len(seeds) and all(d <= -0.5 for d in diffs):
        q1 = "DEEPER_SINKS_MORE"
    else:
        q1 = "NO_DEPTH_TREND"
    S["Q1"] = {"label": q1, "r_net_t30": rnet, "mean": m, "A4_minus_A1_per_seed": diffs,
               "G30_by_depth": {arm: G(arm) for arm in ("A1", "A2", "A4")}}
    S["Issa"] = q1 == "DEEPER_SINKS_LESS"
    # --- judgments J1-J3 and the sentence
    J1 = r1A4 - r1A1
    J2 = {}
    for l in (2, 3, 4):
        d_l = rv("A4", l) - r1A4; thin = sv("A4", l) < 1.0
        J2[l] = {"delta": d_l, "thin": bool(thin), "shallow_thick": bool(d_l >= 0.5 and not thin), "deeper_thick": bool(d_l <= -0.5 and not thin)}
    J3 = bool(rA4C[1] - r1A4 >= 0.5 and G("A4C") <= 0.5 * G("A4"))
    sentence = []
    if abs(J1) < 0.5 and any(v["shallow_thick"] for v in J2.values()) and up_deep >= 0.05 * 3 * nU and J3:
        sentence.append("担い手")
    if J1 >= 0.5:
        sentence.append("深さ")
    if any(v["deeper_thick"] for v in J2.values()) or q1 == "DEEPER_SINKS_MORE":
        sentence.append("どちらでもない")
    S["J"] = {"J1": J1, "J2": J2, "J3": J3, "J3_parts": {"r1_A4C_minus_r1_A4": rA4C[1] - r1A4, "G30_A4C": G("A4C"), "half_G30_A4": 0.5 * G("A4")},
              "sentence": sentence or ["該当なし"]}
    pred = {"担い手": "carrier", "深さ": "depth", "どちらでもない": "neither"}
    S["O7"] = q1 == "NO_DEPTH_TREND"
    S["O8"] = "どちらでもない" in sentence and len(sentence) == 1
    return S


def tables(runs):
    md = []
    seeds_all = sorted({s for (_, s) in runs})
    md.append(f"走: {len(runs)}（seed {seeds_all[0]}〜{seeds_all[-1]}）。値は 3 seed（0〜2）の平均 [最小, 最大]。延長の 10 seed は別の表。\n")
    # T1 position
    md.append("## 表 1 位置（課題末）\n")
    md.append("| 腕 | 層 | 課題 | r の中央値 | μ の中央値 | 幅 s の中央値 | 開く割合 p | 死（p = 0） | p < 1/2 の unit |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for arm in ORDER:
        for l in range(1, n_layers(arm) + 1):
            for t in TASKS:
                g = lambda key: layer_val(runs, arm, REG, t, l, key)
                md.append(f"| {arm} | {l} | {t} | {mean_rng(g('r_med'))} | {mean_rng(g('mu_med'), 1)} | {mean_rng(g('s_med'))} | "
                          f"{mean_rng(g('p_mean'), 3)} | {mean_rng(g('dead'), 2)} | {mean_rng(g('below_half'), 2)} |")
    md.append("\n## 表 2 応答と担い手（課題末）\n")
    md.append("| 腕 | 層 | 課題 | φ′ の平均 | \\|φ′\\| の平均 | n_eff | ‖h̄_{ℓ−1}‖ | 壁 c_ℓ | 入力の総分散 v_ℓ |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for arm in ORDER:
        for l in range(1, n_layers(arm) + 1):
            for t in TASKS:
                g = lambda key: layer_val(runs, arm, REG, t, l, key)
                md.append(f"| {arm} | {l} | {t} | {mean_rng(g('g_mean'), 3)} | {mean_rng(g('gabs_mean'), 3)} | {mean_rng(g('neff_mean'), 3)} | "
                          f"{mean_rng(g('hbar_norm'))} | {mean_rng(g('c'))} | {mean_rng(g('vin'))} |")
    # T3 decomposition
    md.append("\n## 表 3 Δμ の分解（窓の始まりの幅 s_a で割った unit 平均、課題の塊ごとの平均）\n")
    md.append("| 腕 | 層 | 窓 | 課題 | Δμ/s 平均 | 自分の重み | 上流 | 交差 | 上流の割合（絶対） | 上流の割合（符号つき） | 上流が勝つ unit | Δμ < 0 の生きた unit | 自己項 |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for arm in ORDER:
        for l in range(1, n_layers(arm) + 1):
            for w, wn in (("sw", "切替"), ("rest", "残り")):
                for bn, bt in BLOCKS.items():
                    b = lambda key: block_mean(runs, arm, REG, l, w, key, bt)
                    md.append(f"| {arm} | {l} | {wn} | {bn} | {mean_rng(b('dmu_mean'))} | {mean_rng(b('own_mean'))} | {mean_rng(b('up_mean'))} | "
                              f"{mean_rng(b('cross_mean'))} | {mean_rng(b('U_abs'))} | {mean_rng(b('U_signed'))} | {mean_rng(b('up_dominant'))} | "
                              f"{mean_rng(b('down_live'))} | {mean_rng(b('self_mean')) if w == 'sw' else '—'} |")
    # T4 returns
    md.append("\n## 表 4 戻り（3 seed の合計 / 300 unit。延長は 10 seed の合計 / 1,000 unit）\n")
    md.append("| 腕 | 層 | p < 1/2 に入った | p ≤ 0.1 に入った | 素の戻り（課題 30 まで） | 深い戻り（30 まで） | 戻りの回数（30 まで） | 素の戻り（50 まで） | 深い戻り（50 まで） | 10 seed: 素（50 まで） | 10 seed: 深い（50 まで） |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for arm in ORDER:
        for l in range(1, n_layers(arm) + 1):
            c = lambda key, upto, seeds=REG: returns_pooled(runs, arm, seeds, l, upto, key)
            e1, _ = c("ever_below", 50); e2, _ = c("ever_deep", 50)
            p30, _ = c("plain_units", 30); d30, _ = c("deep_units", 30); ev30, _ = c("events", 30)
            p50, _ = c("plain_units", 50); d50, _ = c("deep_units", 50)
            p50x, nx = c("plain_units", 50, seeds_all); d50x, _ = c("deep_units", 50, seeds_all)
            md.append(f"| {arm} | {l} | {e1} | {e2} | {p30} | {d30} | {ev30} | {p50} | {d50} | {p50x} / {nx} | {d50x} / {nx} |")
    # T5 learning
    md.append("\n## 表 5 学び直し（オンライン精度 A、頭 200 更新 H、床 F）\n")
    md.append("| 腕 | " + " | ".join(f"A({t})" for t in TASKS) + " | " + " | ".join(f"H({t})" for t in (1, 5, 10, 30)) + " | G30 | G50 | 課題 21〜30 で床 | 課題 41〜50 で床 |")
    md.append("|---|" + "---|" * (len(TASKS) + 4 + 4))
    for arm in ORDER:
        A = lambda t: seedvals(runs, arm, REG, lambda r: r["learn"]["A"][t - 1] if len(r["learn"]["A"]) >= t else None)
        Hh = lambda t: seedvals(runs, arm, REG, lambda r: r["learn"]["H200"][t - 1] if len(r["learn"]["H200"]) >= t else None)
        G30 = seedvals(runs, arm, REG, lambda r: r["learn"]["G30"]); G50 = seedvals(runs, arm, REG, lambda r: r["learn"]["G50"])
        f30 = seedvals(runs, arm, REG, lambda r: (r["learn"]["floor_21_30"] or [None])[0])
        f50 = seedvals(runs, arm, REG, lambda r: (r["learn"]["floor_41_50"] or [None])[0])
        md.append(f"| {arm} | " + " | ".join(mean_rng(A(t), 3) for t in TASKS) + " | " + " | ".join(mean_rng(Hh(t), 3) for t in (1, 5, 10, 30))
                  + f" | {mean_rng(G30, 3)} | {mean_rng(G50, 3)} | {sum(map(bool, f30))}/{len(f30)} | {sum(map(bool, f50))}/{len(f50)} |")
    # T6 r_net
    md.append("\n## 表 6 網全体の位置 r_net（全隠れ unit をまとめた r の中央値）\n")
    md.append("| 腕 | " + " | ".join(f"課題 {t}" for t in TASKS) + " |")
    md.append("|---|" + "---|" * len(TASKS))
    for arm in ORDER:
        md.append(f"| {arm} | " + " | ".join(mean_rng(seedvals(runs, arm, REG, lambda r: r["r_net"].get(str(t)))) for t in TASKS) + " |")
    # T7 extension: 10 seeds, per-seed layer r at t30 / t50 for the deep arms
    md.append("\n## 表 7 延長: 10 seed の層ごとの位置（課題 30 / 50 の r の中央値と幅 s の中央値）\n")
    md.append("| 腕 | seed | " + " | ".join(f"層 {l}: r（30） | 層 {l}: s（30） | 層 {l}: r（50）" for l in (1, 2, 3, 4)) + " | G30 | G50 |")
    md.append("|---|---|" + "---|" * 14)
    for arm in ("A1", "A2", "A4", "A4C", "R4"):
        for s in seeds_all:
            if (arm, s) not in runs:
                continue
            idx = per_index(runs[(arm, s)])
            cells = []
            for l in (1, 2, 3, 4):
                if l > n_layers(arm):
                    cells += ["", "", ""]
                    continue
                cells += [fmt(idx[(30, l)]["r_med"]), fmt(idx[(30, l)]["s_med"]), fmt(idx.get((50, l), {}).get("r_med"))]
            ln = runs[(arm, s)]["learn"]
            md.append(f"| {arm} | {s} | " + " | ".join(cells) + f" | {fmt(ln['G30'], 3)} | {fmt(ln['G50'], 3)} |")
    return "\n".join(md) + "\n"


def csvs(runs, out: Path):
    with open(out / "table_layers.csv", "w", newline="") as f:
        w = None
        for (arm, s), r in sorted(runs.items()):
            for row in r["per"]:
                rec = {"arm": arm, "seed": s, **row}
                if w is None:
                    keys = list(rec.keys())
                    w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore"); w.writeheader()
                w.writerow({k: rec.get(k, "") for k in w.fieldnames})
    with open(out / "table_returns.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["arm", "seed", "layer", "upto", "ever_below", "ever_deep", "plain_units", "deep_units", "events"])
        for (arm, s), r in sorted(runs.items()):
            for key, c in r["returns"].items():
                l, t = key.split("_")
                w.writerow([arm, s, l[1:], t[1:], c["ever_below"], c["ever_deep"], c["plain_units"], c["deep_units"], c["events"]])
    with open(out / "table_learning.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["arm", "seed", "task", "A", "H200", "A_last1000", "F"])
        for (arm, s), r in sorted(runs.items()):
            ln = r["learn"]
            for t in range(len(ln["A"])):
                w.writerow([arm, s, t + 1, ln["A"][t], ln["H200"][t], ln["A_last1000"][t], ln["F"][t]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    runs = load(Path(a.rows)); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "tables.md").write_text(tables(runs))
    csvs(runs, out)
    S = score(runs)
    S10 = score(runs, seeds=tuple(sorted({s for (_, s) in runs})))
    (out / "scores.json").write_text(json.dumps({"registered_seeds_0_2": S, "extension_all_seeds": S10}, indent=1, ensure_ascii=False, default=float))
    keys = ["Issa", "P1a", "P1b", "P2a", "P2b", "P3", "P4", "P5a", "P5b", "P6a", "P6b", "P6c", "P6d",
            "O1", "O1b", "O1c", "O2a", "O2b", "O3a", "O3b", "O4a", "O4b", "O5a", "O5b", "O5c", "O6a", "O6b", "O6c", "O6d", "O7", "O8"]
    print({k: S[k] for k in keys})
    print("Q1", S["Q1"]["label"], "J", S["J"]["sentence"], "J1", round(S["J"]["J1"], 3))


if __name__ == "__main__":
    main()
