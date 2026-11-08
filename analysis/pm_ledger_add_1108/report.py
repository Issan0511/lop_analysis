"""pm_ledger_add_1108 report: tables, pair comparisons, predictions P5-P8, the one figure.

Reads the add-on runs and ledger (obsidian-research-data/pm_ledger_add_1108) and, for the
none / L2 Init references, the main run (obsidian-research-data/pm_ledger_1108).
Numbers are rounded once, when written.

Usage:
  .venv/bin/python analysis/pm_ledger_add_1108/report.py --box-check   # S-box2 only
  .venv/bin/python analysis/pm_ledger_add_1108/report.py
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DATA = Path("/home/issan/Projects/obsidian-research-data/pm_ledger_add_1108")
RUNS = DATA / "runs"
LEDGER = DATA / "ledger"
OUT = REPO / "results" / "pm_ledger_add_1108"

_spec = importlib.util.spec_from_file_location("main_report", REPO / "analysis" / "pm_ledger_1108" / "report.py")
MR = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(MR)

ARMS_ADD = ("CHB", "KKT1", "CHB_INV0", "CHB_ISO", "L2I_ADD", "KKT1_INV0", "none_INV0")
NAME = {"none": "none（本走）", "l2init": "L2 Init（本走）", "CHB": "CHB", "KKT1": "KKT1", "CHB_INV0": "CHB+INV0",
        "CHB_ISO": "CHB+ISO", "L2I_ADD": "L2I+ADD", "KKT1_INV0": "KKT1+INV0", "none_INV0": "none+INV0"}
TEACHER = {"CHB": {"G": 1.7, "half": 0.25, "avg": 84.7, "late": 83.4, "t1": 84.8, "G_pts": (-1.1, 0.2, 0.9, 1.7, 1.6), "dead500": (0.0, 0.0)},
           "KKT1": {"G": 2.0, "half": 0.25, "avg": 85.0, "late": 84.4, "t1": 86.2, "G_pts": (0.9, 1.4, 1.5, 2.1, 2.0), "dead500": (0.0, 0.0)}}
FRESH_SRC = {"L2I_ADD": ("l2init", True), "none_INV0": ("none", True)}   # arm -> (main-run arm, reuse)
FRESH_TASKS = MR.FRESH
TMAX = MR.TMAX


def arm_dir(arm, seed):
    return RUNS / arm / f"seed{seed}"


def run_values_add(arm, seed):
    rows = MR.read_csv(arm_dir(arm, seed) / "per_task.csv")
    online = np.full(TMAX + 1, np.nan)
    dead1, dead2 = np.full(TMAX + 1, np.nan), np.full(TMAX + 1, np.nan)
    for r in rows:
        t = int(r["task"])
        online[t] = float(r["online"]) * 100
        dead1[t] = float(r["dead_frac_l1"]) * 100
        dead2[t] = float(r["dead_frac_l2"]) * 100
    # the fresh net of this arm's rule (spec §2): own fresh.csv, or the main run's for the arms whose rule it ran
    if arm in FRESH_SRC:
        main_arm = FRESH_SRC[arm][0]
        fr = MR.read_csv(MR.RUNS / main_arm / f"seed{seed}" / "fresh.csv")
        fresh = {int(r["task"]): float(r["online_fresh"]) * 100 for r in fr}
    else:
        fr = MR.read_csv(arm_dir(arm, seed) / "fresh.csv")
        fresh = {int(r["task"]): float(r["online_fresh"]) * 100 for r in fr}
    gap = {t: fresh[t] - online[t] for t in FRESH_TASKS}
    prov = json.loads((arm_dir(arm, seed) / "provenance.json").read_text())
    return {"online": online, "dead1": dead1, "dead2": dead2, "gap": gap, "prov": prov}


def G_of(v):
    return float(np.mean([v["gap"][400], v["gap"][500]]))


def seeds_of(arm):
    return [s for s in range(10) if (arm_dir(arm, s) / "provenance.json").exists()]


def tol_of(n, half):
    """S-box2 line: t x sqrt(SD²/n + SE²) with the teacher's SD and SE (spec §7)."""
    se_t = half / 2.262
    sd = se_t * np.sqrt(10)
    tcrit = {3: 4.30, 10: 2.10}.get(n, 2.23)
    return float(tcrit * np.sqrt(sd ** 2 / n + se_t ** 2))


def box_check():
    ok = {}
    for arm in ("CHB", "KKT1"):
        seeds = seeds_of(arm)
        if not seeds:
            print(f"[{arm}] no finished runs yet")
            continue
        V = {s: run_values_add(arm, s) for s in seeds}
        G = np.array([G_of(V[s]) for s in seeds])
        avg = np.array([np.nanmean(V[s]["online"][1:]) for s in seeds])
        late = np.array([np.nanmean(V[s]["online"][451:501]) for s in seeds])
        t1 = np.array([V[s]["online"][1] for s in seeds])
        pts = {t: np.array([V[s]["gap"][t] for s in seeds]) for t in FRESH_TASKS}
        dead = (np.array([V[s]["dead1"][500] for s in seeds]), np.array([V[s]["dead2"][500] for s in seeds]))
        T = TEACHER[arm]
        tol = tol_of(len(seeds), T["half"])
        good = abs(float(G.mean()) - T["G"]) <= tol
        ok[arm] = (good, float(G.mean()), tol, len(seeds))
        print(f"[{arm}] seeds {seeds}")
        print(f"  差 G（課題 400・500 の平均）: {G.mean():+.2f}（先生 {T['G']:+.1f}）、seed ごと {[round(x,2) for x in G]}  線 ±{tol:.2f} → {'OK' if good else 'NG（箱の不一致の疑い）'}")
        print(f"  全課題の平均 online: {avg.mean():.2f}%（先生 {T['avg']}%）  後期窓: {late.mean():.2f}%（先生 {T['late']}%）  課題 1: {t1.mean():.2f}%（先生 {T['t1']}%）")
        print(f"  差 G の 5 点: " + " ".join(f"{pts[t].mean():+.2f}" for t in FRESH_TASKS) + f"（先生 {T['G_pts']}）")
        print(f"  課題 500 の死 %（第 1/第 2 層）: {dead[0].mean():.1f}/{dead[1].mean():.1f}（先生 {T['dead500']}）")
    return ok


def load_main_ledger(s):
    L, meta = MR.load_ledger(s)
    return L, meta


def load_add_ledger(s):
    z = np.load(LEDGER / f"seed{s}.npz")
    out = {a: {k.split("__", 1)[1]: z[k] for k in z.files if k.startswith(a + "__")}
           for a in ARMS_ADD if any(k.startswith(a + "__") for k in z.files)}
    meta = {k.split("__", 1)[1]: z[k] for k in z.files if k.startswith("meta__")}
    return out, meta


def main_values(arm, s):
    return MR.run_values(arm, s)


def write_csv(path, rows):
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        for r in rows:
            wr.writerow({k: (f"{v:.10g}" if isinstance(v, float) else v) for k, v in r.items()})


def paired(a, b):
    """mean difference a-b, 95% t CI (df = n-1), count of a>b."""
    d = np.asarray(a) - np.asarray(b)
    n = len(d)
    tc = {3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}[n]
    m = float(d.mean())
    se = float(d.std(ddof=1)) / np.sqrt(n)
    return m, m - tc * se, m + tc * se, int((d > 0).sum()), n


def j2_label_G(m, npos, n):
    """J.2 line for G (lower = better) with diff = G_A - G_B, m < 0 means A keeps plasticity better."""
    if m <= -0.5 and npos == 0:
        return "A が保つ（線 10/10・差 ≥ 0.5）"
    if m >= 0.5 and npos >= 0.8 * n:
        return "B が保つ（線 8/10・差 ≥ 0.5）"
    return "差が定まらない"


def j2_label_online(m, npos, n):
    """J.2 line for mean online (higher = better) with diff = online_A - online_B."""
    if m >= 0.5 and npos == n:
        return "A が上（線 10/10・差 ≥ 0.5）"
    if m <= -0.5 and n - npos >= 0.8 * n:
        return "B が上（線 8/10・差 ≥ 0.5）"
    return "差が定まらない"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--box-check", action="store_true")
    args = ap.parse_args()
    if args.box_check:
        box_check()
        return
    # seeds where every required add-on arm has a finished run and a ledger entry
    required = ("CHB", "KKT1", "CHB_INV0", "CHB_ISO", "L2I_ADD")
    optional = ("KKT1_INV0", "none_INV0")
    seeds = [s for s in range(10) if all((arm_dir(a, s) / "provenance.json").exists() for a in required)
             and (LEDGER / f"seed{s}.npz").exists()]
    print("seeds with all required arms and ledger:", seeds)
    n = len(seeds)
    OUT.mkdir(parents=True, exist_ok=True)
    box = box_check()

    # ---- loads
    main_V = {arm: {s: main_values(arm, s) for s in seeds} for arm in ("none", "l2init")}
    add_V = {}
    for arm in ARMS_ADD:
        ss = [s for s in seeds if (arm_dir(arm, s) / "provenance.json").exists()]
        add_V[arm] = {s: run_values_add(arm, s) for s in ss}
    arms_seen = [a for a in ARMS_ADD if len(add_V[a]) == n]
    print("arms with all seeds:", arms_seen)
    main_L, add_L, metas = {}, {}, {}
    for s in seeds:
        L_main, meta_main = load_main_ledger(s)
        L_add, meta_add = load_add_ledger(s)
        main_L[s], add_L[s] = L_main, L_add
        metas[s] = meta_add if meta_add else meta_main

    # ---- window statistics per seed/arm
    stats = {s: {} for s in seeds}
    for s in seeds:
        for arm in ("none", "l2init"):
            stats[s][arm] = {w: MR.window_stats(main_L[s][arm], metas[s], t1, t2, w) for w, t1, t2 in MR.WINDOWS}
        for arm in arms_seen:
            stats[s][arm] = {w: MR.window_stats(add_L[s][arm], metas[s], t1, t2, w) for w, t1, t2 in MR.WINDOWS}
    ALL_ARMS = ["none", "l2init"] + arms_seen

    # ---- S-ledger on the add-on arms (identity within each task; the r_all identity only without switch ops)
    res_ratio, dcheck, rcheck = [], [], []
    for s in seeds:
        for arm in arms_seen:
            L = add_L[s][arm]
            r = np.abs(L["res"][1:]); b = L["resbound"][1:]
            res_ratio.append(float(np.nanmax(r / b)))
            T = L["T"][1:TMAX]; d = L["d"][1:TMAX]
            dcheck.append(float(np.nanmax(np.abs(T.sum(2) - d) / np.maximum(d, 1e-300))))
            if arm in ("CHB", "KKT1"):
                for w, _, _ in MR.WINDOWS:
                    W = stats[s][arm][w]["whole"]
                    rcheck.append(abs(W["ero_all"] - W["ero_all_r_minus_1"]))
    s_ledger = {"max_res_over_bound": max(res_ratio) if res_ratio else None,
                "max_rel_T_row_sum_vs_d": max(dcheck) if dcheck else None,
                "max_abs_ero_all_vs_r_minus_1_noSwitchArms": max(rcheck) if rcheck else None}
    print("S-ledger:", s_ledger)
    if s_ledger["max_res_over_bound"] is not None and s_ledger["max_res_over_bound"] > 1.0:
        raise SystemExit("S-ledger: identity residual over the bound")

    # ---- tables (same layout as the main run)
    metrics_band = ["dim_frac", "lam_share", "d_share", "V_share", "S_share", "ero_in", "ero_F", "floor",
                    "f_b", "q1", "q50", "rho_b", "dnext_share"]
    metrics_win = ["rho", "rho_iso", "rho_over_iso", "V_all_per_task", "D2_per_task", "N_all_mean", "r_all",
                   "ero_all_r_minus_1", "ero_all", "floor_all", "ero_in_all", "q1_all", "q50_all", "S2_mean",
                   "Minv_mean", "Mvis_mean", "Minv_frac"]
    band_rows, win_rows = [], []
    for arm in ALL_ARMS:
        for w, t1, t2 in MR.WINDOWS:
            for b in MR.BANDS:
                row = {"arm": arm, "window": w, "t1": t1, "t2": t2, "band": b, "n_seeds": n}
                for m in metrics_band:
                    med, lo, hi = MR.agg([stats[s][arm][w]["bands"][b][m] for s in seeds])
                    row[m], row[m + "_min"], row[m + "_max"] = med, lo, hi
                band_rows.append(row)
            row = {"arm": arm, "window": w, "t1": t1, "t2": t2, "n_seeds": n}
            for m in metrics_win:
                med, lo, hi = MR.agg([stats[s][arm][w]["whole"][m] for s in seeds])
                row[m], row[m + "_min"], row[m + "_max"] = med, lo, hi
            win_rows.append(row)
    write_csv(OUT / "table_band_window_arm.csv", band_rows)
    write_csv(OUT / "table_window_arm.csv", win_rows)

    # ---- per-seed G/online/dead and M_inv
    def vals(arm, s):
        return main_V[arm][s] if arm in ("none", "l2init") else add_V[arm][s]

    def Minv_w4(arm, s):
        L = main_L[s][arm] if arm in ("none", "l2init") else add_L[s][arm]
        return float(np.mean(L["Sprev"][451:501, 3] + L["Sprev"][451:501, 4]))

    def Wnorm2_w4(arm, s):
        L = main_L[s][arm] if arm in ("none", "l2init") else add_L[s][arm]
        return float(np.nanmean(L["N_all"][451:501]))

    per_seed = []
    G_by = {}
    OA_by = {}
    for arm in ALL_ARMS:
        G_by[arm] = np.array([G_of(vals(arm, s)) for s in seeds])
        OA_by[arm] = np.array([float(np.nanmean(vals(arm, s)["online"][1:])) for s in seeds])
    for i, s in enumerate(seeds):
        row = {"seed": s}
        for arm in ALL_ARMS:
            V = vals(arm, s)
            row[f"G_{arm}"] = float(G_by[arm][i])
            row[f"online_avg_{arm}"] = float(OA_by[arm][i])
            row[f"dead500_L1_{arm}"] = float(V["dead1"][500])
            row[f"dead500_L2_{arm}"] = float(V["dead2"][500])
            row[f"Minv_w4_{arm}"] = Minv_w4(arm, s)
            row[f"N_w4_{arm}"] = Wnorm2_w4(arm, s)
        per_seed.append(row)
    write_csv(OUT / "per_seed.csv", per_seed)

    # ---- pair comparisons (spec §6)
    pairs = [("CHB_INV0", "CHB"), ("CHB_ISO", "CHB"), ("CHB_INV0", "CHB_ISO"), ("L2I_ADD", "l2init"),
             ("CHB", "none"), ("KKT1", "none")]
    for extra, other in (("KKT1_INV0", "KKT1"), ("none_INV0", "none")):
        if extra in arms_seen:
            pairs.append((extra, other))
    pair_rows = []
    for a, b in pairs:
        m, lo, hi, npos, nn = paired(G_by[a], G_by[b])
        m2, lo2, hi2, npos2, nn2 = paired(OA_by[a], OA_by[b])
        pair_rows.append({"A": NAME[a], "B": NAME[b], "dG_mean": m, "dG_lo": lo, "dG_hi": hi, "n_pos_G": npos, "n": nn,
                          "label_G": j2_label_G(m, npos, nn), "dOnline_mean": m2, "dOnline_lo": lo2, "dOnline_hi": hi2,
                          "n_pos_online": npos2, "label_online": j2_label_online(m2, npos2, nn2)})
    write_csv(OUT / "compare_pairs.csv", pair_rows)

    # ---- predictions P5-P8
    def win(arm, w, k):
        return next(r[k] for r in win_rows if r["arm"] == arm and r["window"] == w)
    P = {}
    # P5
    res_p5 = {}
    for arm in ("CHB", "KKT1"):
        ratio = float(np.median([Wnorm2_w4(arm, s) / Wnorm2_w4("none", s) for s in seeds]))
        frac = float(np.median([stats[s][arm]["w4"]["whole"]["Minv_frac"] for s in seeds]))
        mratio = float(np.median([Minv_w4(arm, s) / Minv_w4("none", s) for s in seeds]))
        ok = (0.5 <= ratio <= 2.0) and (0.40 <= frac <= 0.47)
        res_p5[arm] = {"W_ratio_to_none": ratio, "Minv_frac": frac, "Minv_ratio_to_none": mratio, "ok": bool(ok)}
    p5_oks = [res_p5[a]["ok"] for a in ("CHB", "KKT1")]
    P["P5"] = {"text": "CHB と KKT1 の ‖W1‖²（課題 451〜500）は none の 0.5〜2 倍で M_inv/‖S‖² は 0.40〜0.47",
               "detail": res_p5, "grade": "○" if all(p5_oks) else ("△" if any(p5_oks) else "×")}
    # P6
    G_c = float(G_by["CHB"].mean())
    G_i = float(G_by["CHB_INV0"].mean())
    if G_c <= 0:
        g6 = "判定不能（G(CHB) ≤ 0）"
    else:
        g6 = "○" if G_i <= G_c / 2 else "×"
    P["P6"] = {"text": "CHB+INV0 の G は CHB の半分以下", "G_CHB": G_c, "G_CHB_INV0": G_i, "grade": g6}
    # P7
    diff_iso = G_by["CHB_ISO"] - G_by["CHB_INV0"]
    n_pos = int((diff_iso > 0).sum()); md = float(diff_iso.mean())
    P["P7"] = {"text": "CHB+INV0 の G は CHB+ISO より小さい（8/10 以上で平均差 ≥ 0.5）",
               "n_pos": n_pos, "n": n, "mean_diff_ISO_minus_INV0": md, "G_CHB_ISO": float(G_by["CHB_ISO"].mean()),
               "grade": "○" if (n_pos >= 8 and md >= 0.5) else "×"}
    # P8
    Gn = float(G_by["none"].mean()); Gl = float(G_by["l2init"].mean())
    line = (Gn + Gl) / 2
    G_add = float(G_by["L2I_ADD"].mean())
    P["P8"] = {"text": "L2I+ADD の G は L2 Init から none との差の半分以上まで上がる", "G_L2I_ADD": G_add, "line": line,
               "G_none": Gn, "G_l2init": Gl, "frac_of_gap": (G_add - Gl) / (Gn - Gl), "grade": "○" if G_add >= line else "×"}
    # verdicts
    A = "PATH_DOES_NOT_TOUCH" if P["P5"]["grade"] == "○" else ("PARTIAL" if P["P5"]["grade"] == "△" else "PATH_CHANGES_INVISIBLE_MASS")
    if P["P6"]["grade"] != "○":
        Bv = "NOT_SUPPORTED"
    elif P["P7"]["grade"] != "○":
        Bv = "NORM_OR_NONSPECIFIC"
    elif P["P8"]["grade"] != "○":
        Bv = "INVISIBLE_MASS_CAUSE_NOT_SUFFICIENT"
    else:
        Bv = "LOSS_FOLLOWS_INVISIBLE_MASS"
    judge = {"seeds": seeds, "predictions": P, "verdict_A": A, "verdict_B": Bv, "S_ledger": s_ledger,
             "S_box2": {k: {"ok": bool(v[0]), "G_mean": v[1], "tol": v[2], "n": v[3]} for k, v in box.items()},
             "arms_seen": arms_seen}
    (OUT / "judge.json").write_text(json.dumps(judge, indent=1, ensure_ascii=False, default=float))

    # ---- figure
    draw_figure(ALL_ARMS, seeds, G_by, main_L, add_L, OUT / "fig_pm_ledger_add_1108.png")

    # ---- summary.md
    L = []; w_ = L.append
    w_("# pm_ledger_add_1108 結果の表（自動生成・report.py）"); w_("")
    w_(f"seed {seeds}（{n} 本）、腕 {', '.join(NAME[a] for a in ALL_ARMS)}。数は seed の中央値（[最小, 最大] は CSV）か、G は seed の平均（対の比べ）。表の丸めは最後の 1 回だけ。")
    w_(""); w_("## 1. 予測の採点"); w_("")
    w_("| 予測 | 文 | 採点 |"); w_("|---|---|---|")
    pp = P["P5"]["detail"]
    w_(f"| P5 | {P['P5']['text']} | **{P['P5']['grade']}**（CHB: 比 {pp['CHB']['W_ratio_to_none']:.3f}・{pp['CHB']['Minv_frac']:.3f}　KKT1: 比 {pp['KKT1']['W_ratio_to_none']:.3f}・{pp['KKT1']['Minv_frac']:.3f}） |")
    w_(f"| P6 | {P['P6']['text']} | **{P['P6']['grade']}**（G(CHB+INV0) = {P['P6']['G_CHB_INV0']:+.2f}、G(CHB) = {P['P6']['G_CHB']:+.2f}） |")
    w_(f"| P7 | {P['P7']['text']} | **{P['P7']['grade']}**（正の seed {P['P7']['n_pos']}/{P['P7']['n']}、平均差 {P['P7']['mean_diff_ISO_minus_INV0']:+.2f}、G(CHB+ISO) = {P['P7']['G_CHB_ISO']:+.2f}） |")
    w_(f"| P8 | {P['P8']['text']} | **{P['P8']['grade']}**（G(L2I+ADD) = {P['P8']['G_L2I_ADD']:+.2f}、線 {P['P8']['line']:+.2f}） |")
    w_(f"| 判定 A | P5 | **{A}** |"); w_(f"| 判定 B | P6〜P8 | **{Bv}** |"); w_("")
    w_("## 2. 腕の表（10 seed の平均 G・全課題の平均 online・課題 500 の死・M_inv）"); w_("")
    w_("| 腕 | G（平均・ポイント） | 全課題の平均 online | 課題 500 の死 %（層 1/2） | ‖W1‖²（課題 451〜500） | M_inv（同） | M_inv/‖S‖²（同） |")
    w_("|---|---|---|---|---|---|---|")
    for arm in ALL_ARMS:
        d1 = np.mean([vals(arm, s)["dead1"][500] for s in seeds]); d2 = np.mean([vals(arm, s)["dead2"][500] for s in seeds])
        Nw = np.median([Wnorm2_w4(arm, s) for s in seeds]); Mw = np.median([Minv_w4(arm, s) for s in seeds])
        frac = np.median([stats[s][arm]["w4"]["whole"]["Minv_frac"] for s in seeds])
        w_(f"| {NAME[arm]} | {G_by[arm].mean():+.2f} | {OA_by[arm].mean():.2f}% | {d1:.1f}/{d2:.1f} | {Nw:.0f} | {Mw:.1f} | {frac:.3f} |")
    w_("")
    w_("## 3. 対の比べ（同じ seed の対、差 = A − B。G は小さいほど保つ、線は J.2）"); w_("")
    w_("| A | B | ΔG 平均 [95% t] | n(A>B) | J.2 の線（G） | Δonline 平均 [95% t] | n(A>B) | J.2 の線（online） |")
    w_("|---|---|---|---|---|---|---|---|")
    for r in pair_rows:
        w_(f"| {r['A']} | {r['B']} | {r['dG_mean']:+.2f} [{r['dG_lo']:+.2f}, {r['dG_hi']:+.2f}] | {r['n_pos_G']}/{r['n']} | {r['label_G']} | {r['dOnline_mean']:+.2f} [{r['dOnline_lo']:+.2f}, {r['dOnline_hi']:+.2f}] | {r['n_pos_online']}/{r['n']} | {r['label_online']} |")
    w_("")
    w_("## 4. 窓 × 腕の帳簿（本走と同じ列）"); w_("")
    w_("| 腕 | 窓 | ρ_w | ρ_iso | ero_all | ‖W‖²平均 | ‖S‖²平均 | M_inv | M_vis | M_inv/‖S‖² | D²/課題 | V/課題 |")
    w_("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for arm in ALL_ARMS:
        for w, t1, t2 in MR.WINDOWS:
            r = next(x for x in win_rows if x["arm"] == arm and x["window"] == w)
            w_(f"| {NAME[arm]} | {w} {t1}–{t2} | {r['rho']:.4f} | {r['rho_iso']:.4f} | {r['ero_all']:+.3f} | {r['N_all_mean']:.1f} | {r['S2_mean']:.1f} | {r['Minv_mean']:.1f} | {r['Mvis_mean']:.1f} | {r['Minv_frac']:.3f} | {r['D2_per_task']:.2f} | {r['V_all_per_task']:.2f} |")
    w_("")
    w_("## 4b. 差 G の 5 点（課題 100〜500、10 seed の平均）と online の窓（10 seed の平均）"); w_("")
    w_("| 腕 | G(100) | G(200) | G(300) | G(400) | G(500) | online 1〜10 | 11〜100 | 101〜300 | 301〜450 | 451〜500 |")
    w_("|---|---|---|---|---|---|---|---|---|---|---|")
    for arm in ALL_ARMS:
        g5 = [float(np.mean([vals(arm, s)["gap"][t] for s in seeds])) for t in FRESH_TASKS]
        o5 = [float(np.mean([np.nanmean(vals(arm, s)["online"][i:j]) for s in seeds])) for i, j in ((1, 11), (11, 101), (101, 301), (301, 451), (451, 501))]
        w_(f"| {NAME[arm]} | " + " | ".join(f"{x:+.2f}" for x in g5) + " | " + " | ".join(f"{x:.2f}" for x in o5) + " |")
    w_("")
    w_("## 4c. 切替の操作の検査（per_task.csv、課題 2〜500、10 seed 全部）"); w_("")
    w_("| 腕 | 操作後の見えない質量 ÷ 操作前の ‖W1‖²（最大） | 消す割合 inv_pre/‖W1‖²（平均） | c_iso の範囲 | 足した質量 ÷ none の M_inv(t)（最大の相対誤差） |")
    w_("|---|---|---|---|---|")
    for arm in [a for a in ("CHB_INV0", "KKT1_INV0", "none_INV0", "CHB_ISO", "L2I_ADD") if a in arms_seen]:
        rr = {s: [r for r in MR.read_csv(arm_dir(arm, s) / "per_task.csv") if int(r["task"]) >= 2] for s in seeds}
        col = lambda s, k: np.array([float(r[k]) if r[k] not in ("", "nan") else np.nan for r in rr[s]])
        post = max(float(np.nanmax(col(s, "inv_post") / col(s, "F2_pre"))) for s in seeds) if arm.endswith("INV0") else float("nan")
        frac = float(np.mean([np.nanmean(col(s, "inv_pre") / col(s, "F2_pre")) for s in seeds]))
        if arm == "CHB_ISO":
            ci = (min(float(np.nanmin(col(s, "c_iso"))) for s in seeds), max(float(np.nanmax(col(s, "c_iso"))) for s in seeds))
            cs = f"{ci[0]:.3f}〜{ci[1]:.3f}"
        else:
            cs = "—"
        if arm == "L2I_ADD":
            rel = 0.0
            for s in seeds:
                target = main_L[s]["none"]["Sprev"][:, 3] + main_L[s]["none"]["Sprev"][:, 4]
                ts = np.array([int(r["task"]) for r in rr[s]])
                rel = max(rel, float(np.nanmax(np.abs(col(s, "add_R2") - target[ts]) / target[ts])))
            ms = f"{rel:.1e}"
        else:
            ms = "—"
        ps = f"{post:.1e}" if post == post else "—"
        w_(f"| {NAME[arm]} | {ps} | {frac:.4f} | {cs} | {ms} |")
    w_("")
    w_("## 5. S-box2・検査"); w_("")
    for k, v in box.items():
        w_(f"- {k}: G 平均 {v[1]:+.2f}（線 ±{v[2]:.2f}、n = {v[3]}）→ {'OK' if v[0] else 'NG'}")
    w_(f"- S-ledger: {s_ledger}")
    (OUT / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote", OUT)


def draw_figure(all_arms, seeds, G_by, main_L, add_L, png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    order = [a for a in ("none", "l2init", "CHB", "KKT1", "CHB_INV0", "CHB_ISO", "L2I_ADD") if a in all_arms]
    # the G panel also shows the optional arms; their identity is carried by the row label, so one neutral colour
    order_g = order + [a for a in ("KKT1_INV0", "none_INV0") if a in all_arms]
    COL = {"none": "#2a78d6", "l2init": "#eb6834", "CHB": "#1baf7a", "KKT1": "#eda100",
           "CHB_INV0": "#e87ba4", "CHB_ISO": "#6250d6", "L2I_ADD": "#008300", "KKT1_INV0": "#8a8a8a", "none_INV0": "#8a8a8a"}
    NAMES = {"none": "none", "l2init": "L2 Init", "CHB": "CHB", "KKT1": "KKT1", "CHB_INV0": "CHB+INV0", "CHB_ISO": "CHB+ISO", "L2I_ADD": "L2I+ADD",
             "KKT1_INV0": "KKT1+INV0", "none_INV0": "none+INV0"}
    # line style as secondary encoding: all-pairs CVD separation fails for 7 hues (dataviz skill)
    LS = {"none": "-", "l2init": "-", "CHB": "-.", "KKT1": "-.", "CHB_INV0": "--", "CHB_ISO": ":", "L2I_ADD": (0, (5, 1, 1, 1))}
    n = len(seeds)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=160, gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axes[0]
    tc = 2.262 if n == 10 else 2.5
    ys = np.arange(len(order_g))[::-1]
    for y, arm in zip(ys, order_g):
        g = G_by[arm]
        m = g.mean(); se = g.std(ddof=1) / np.sqrt(n)
        ax.errorbar(m, y, xerr=tc * se, fmt="o", ms=7, color=COL[arm], capsize=3, lw=1.8)
        ax.annotate(f"{m:+.2f}", xy=(m, y), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9, color="#222222")
    ax.axvline(0, color="#999999", lw=1)
    ax.set_yticks(ys); ax.set_yticklabels([NAMES[a] for a in order_g])
    ax.set_xlabel("初期化から学ばせた網との差 G（ポイント、10 seed の平均と 95% t 区間）")
    ax.set_title("腕ごとの可塑性の喪失", fontsize=11, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#eeeeee", lw=0.8)
    ax = axes[1]
    ts = np.arange(2, TMAX + 1)     # M_inv(1) = 0 by definition (no past increment yet), so start at task 2
    for arm in order:
        arr = np.array([(main_L[s][arm]["Sprev"][:, 3] + main_L[s][arm]["Sprev"][:, 4]) if arm in ("none", "l2init")
                        else (add_L[s][arm]["Sprev"][:, 3] + add_L[s][arm]["Sprev"][:, 4]) for s in seeds])[:, 2:]
        med = np.median(arr, axis=0)
        ax.plot(ts, med, color=COL[arm], lw=1.6, ls=LS[arm], label=NAMES[arm])
    ax.set_yscale("log")
    ax.set_xlabel("課題 t")
    ax.set_ylabel("見えない質量 M_inv(t)（対数・seed の中央値）")
    ax.set_title("学習の前の見えない質量（切替の操作の後）", fontsize=11, loc="left")
    ax.legend(frameon=False, ncols=2, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#eeeeee", lw=0.8)
    fig.suptitle(f"pm_ledger_add_1108（{n} seed、事後）", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(png)
    plt.close(fig)


if __name__ == "__main__":
    main()
