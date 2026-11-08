"""pm_ledger_1108 report: tables, predictions, judgment and the one figure.

Reads the per-seed ledger npz (ledger.py) and the run files (run.py), computes
per-seed window statistics, aggregates over seeds (median [min, max]) and writes
results/pm_ledger_1108/.  Numbers are rounded once, when written.

Usage:
  .venv/bin/python analysis/pm_ledger_1108/report.py --box-check          # S-box only
  .venv/bin/python analysis/pm_ledger_1108/report.py
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DATA = Path("/home/issan/Projects/obsidian-research-data/pm_ledger_1108")
RUNS = DATA / "runs"
LEDGER = DATA / "ledger"
OUT = REPO / "results" / "pm_ledger_1108"

ARMS = ("none", "l2init")
BANDS = ("top", "mid1", "mid2", "low", "comp")
NB = len(BANDS)
DIM = 784 * 100
TMAX = 500
WINDOWS = (("w1", 6, 50), ("w2", 101, 150), ("w3", 251, 300), ("w4", 451, 500), ("all", 6, 500))
FOUR = ("w1", "w2", "w3", "w4")
FRESH = (100, 200, 300, 400, 500)
# Spearman, n = 10: exact one-sided p <= 0.05 critical value (spec §6)
RHO_CRIT_10 = 62 / 110
RL_REF = {"top": -0.925, "mid1": -0.683, "mid2": -0.403, "low": -0.194, "comp": -0.053, "all": -0.607}
TEACHER = {"none": {"t1": 83.8, "avg": 78.8, "G": 8.1, "G_pts": (2.1, 4.7, 6.6, 7.9, 8.3), "late": 75.3},
           "l2init": {"t1": 83.6, "avg": 88.4, "G": -4.6, "G_pts": (-4.9, -4.7, -4.6, -4.6, -4.6), "late": 88.4}}
TEACHER_DEAD = {"none": {1: (6.2, 11.1), 50: (35.9, 18.9), 500: (15.7, 80.0)},
                "l2init": {1: (0.3, 6.4), 50: (7.3, 0.8), 500: (7.2, 1.5)}}


def read_csv(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def seeds_available(need=ARMS):
    found = []
    for s in range(10):
        if all((RUNS / a / f"seed{s}" / "provenance.json").exists() for a in need):
            found.append(s)
    return found


def run_values(arm, seed):
    """Online accuracy per task (points), fresh online at FRESH, dead fractions."""
    rows = read_csv(RUNS / arm / f"seed{seed}" / "per_task.csv")
    fresh = read_csv(RUNS / arm / f"seed{seed}" / "fresh.csv")
    online = np.full(TMAX + 1, np.nan)
    dead1, dead2 = np.full(TMAX + 1, np.nan), np.full(TMAX + 1, np.nan)
    acc = np.full(TMAX + 1, np.nan)
    for r in rows:
        t = int(r["task"])
        online[t] = float(r["online"]) * 100
        acc[t] = float(r["acc"]) * 100
        dead1[t] = float(r["dead_frac_l1"]) * 100
        dead2[t] = float(r["dead_frac_l2"]) * 100
    gap = {}
    for r in fresh:
        gap[int(r["task"])] = (float(r["online_fresh"]) - float(r["online_cont"])) * 100
    prov = json.loads((RUNS / arm / f"seed{seed}" / "provenance.json").read_text())
    return {"online": online, "acc": acc, "dead1": dead1, "dead2": dead2, "gap": gap, "prov": prov}


def G_of(run):
    return float(np.mean([run["gap"][400], run["gap"][500]]))


def box_table(seeds):
    out = {}
    for arm in ARMS:
        vals = {s: run_values(arm, s) for s in seeds}
        G = np.array([G_of(vals[s]) for s in seeds])
        t1 = np.array([vals[s]["online"][1] for s in seeds])
        avg = np.array([np.nanmean(vals[s]["online"][1:]) for s in seeds])
        late = np.array([np.nanmean(vals[s]["online"][451:501]) for s in seeds])
        pts = {t: np.array([vals[s]["gap"][t] for s in seeds]) for t in FRESH}
        dead = {t: (np.array([vals[s]["dead1"][t] for s in seeds]),
                    np.array([vals[s]["dead2"][t] for s in seeds])) for t in (1, 50, 500)}
        out[arm] = {"seeds": seeds, "G_seed": G.tolist(), "G_mean": float(G.mean()), "t1_mean": float(t1.mean()),
                    "avg_mean": float(avg.mean()), "late_mean": float(late.mean()),
                    "G_pts_mean": {t: float(pts[t].mean()) for t in FRESH},
                    "dead_mean": {t: (float(dead[t][0].mean()), float(dead[t][1].mean())) for t in dead},
                    "diverged": [bool(vals[s]["prov"]["diverged"]["diverged"]) for s in seeds]}
    return out


def tol_of(n):
    """Tolerance of spec §6/S-box on the seed-mean gap: t x SE of (ours - teacher)."""
    sd_seed = 0.70                   # teacher's seed SD, from +8.1 [+7.6, +8.6], t(9) = 2.262
    se_t = 0.22
    tcrit = {3: 4.30, 10: 2.10}.get(n)
    if tcrit is None:
        tcrit = 2.23                 # n between 3 and 10: df of the smaller sample ~ n-1, conservative
    return float(tcrit * np.sqrt(sd_seed ** 2 / n + se_t ** 2))


def print_box(box):
    n = len(box["none"]["seeds"])
    tol = tol_of(n)
    verdict = {}
    for arm in ARMS:
        B = box[arm]
        T = TEACHER[arm]
        ok = abs(B["G_mean"] - T["G"]) <= tol
        verdict[arm] = ok
        print(f"[{arm}] seeds {B['seeds']}")
        print(f"  課題 1 の online: 実測 {B['t1_mean']:.2f}%（先生 {T['t1']}%）  全課題の平均 online: {B['avg_mean']:.2f}%（先生 {T['avg']}%）"
              f"  後期窓(451-500): {B['late_mean']:.2f}%（先生 {T['late']}%）")
        print(f"  差 G（課題 400・500 の平均・ポイント）: {B['G_mean']:+.2f}（先生 {T['G']:+.1f}）、seed ごと {[round(x,2) for x in B['G_seed']]}"
              f"  線 ±{tol:.2f} → {'OK' if ok else 'NG（箱の不一致の疑い）'}")
        print(f"  差 G の 5 点(100..500): " + " ".join(f"{B['G_pts_mean'][t]:+.2f}" for t in FRESH) + f"（先生 {T['G_pts']}）")
        print("  死の割合 %（第 1 層/第 2 層、試験集合）: " + "  ".join(
            f"t{t}: {B['dead_mean'][t][0]:.1f}/{B['dead_mean'][t][1]:.1f}（先生 {TEACHER_DEAD[arm][t][0]}/{TEACHER_DEAD[arm][t][1]}）" for t in (1, 50, 500)))
        if any(B["diverged"]):
            print("  発散あり:", B["diverged"])
    return verdict


# ------------------------------------------------------------------------- ledger

def load_ledger(seed):
    z = np.load(LEDGER / f"seed{seed}.npz")
    out = {a: {k.split("__", 1)[1]: z[k] for k in z.files if k.startswith(a + "__")} for a in ARMS}
    meta = {k.split("__", 1)[1]: z[k] for k in z.files if k.startswith("meta__")}
    return out, meta


def window_stats(L, meta, t1, t2, name):
    """Per-seed statistics for one window of one arm."""
    idx = np.arange(t1, t2 + 1)
    idx_next = idx[idx <= TMAX - 1]
    d = L["d"][idx]
    D2 = L["D2"][idx]
    dsum = d.sum(0)
    dall = float(D2.sum())
    a_key = "awhole" if name == "all" else "awin"
    L2_key = "L2whole" if name == "all" else "L2win"
    a = L[a_key][idx]
    L2 = L[L2_key][idx]
    out = {}
    # per band
    b = {}
    for k, nm in enumerate(BANDS):
        floor = 2 * np.sqrt(float(np.nansum(L2 * d[:, k])) / DIM) / dsum[k] if dsum[k] > 0 else np.nan
        def lag_q(key):
            den = float(np.nansum(np.where(np.isfinite(L[key][idx, k]), d[:, k], np.nan)))
            return float(np.nansum(L[key][idx, k])) / den if den > 0 else float("nan")
        q1, q50 = lag_q("a1"), lag_q("a50")
        b[nm] = {
            "dim_frac": float(np.mean(meta["dims"][idx, k]) / 784),
            "lam_share": float(np.mean(meta["lam_share"][idx, k])),
            "d_share": float(dsum[k] / dall),
            "V_share": float(L["V"][idx, k].sum() / L["V"][idx].sum()),
            "S_share": float(L["Sprev"][idx, k].sum() / L["S2"][idx].sum()),
            "ero_in": float(L["e"][idx, k].sum() / dsum[k]) if dsum[k] > 0 else np.nan,
            "ero_F": float(2 * np.nansum(a[:, k]) / dsum[k]) if dsum[k] > 0 else np.nan,
            "floor": float(floor),
            "f_b": float(-L["e"][idx, k].sum() / (2 * L["Nprev"][idx, k].sum())),   # post hoc (spec 外)
            "q1": q1, "q50": q50,
            "rho_b": float(L["Vnext"][idx_next, k].sum() / L["V"][idx_next, k].sum()) if L["V"][idx_next, k].sum() > 0 else np.nan,
            "dnext_share": float(L["dnext"][idx_next, k].sum() / L["D2"][idx_next].sum()),
        }
        # the band's own share of where its deposit lands in the next task's bands
        Tm = L["T"][idx_next, k, :].sum(0)
        b[nm]["T_row"] = (Tm / Tm.sum()).tolist() if Tm.sum() > 0 else [np.nan] * NB
    out["bands"] = b
    # whole-layer
    V_all = L["V"][idx_next].sum()
    rho = float(L["Vnext_all"][idx_next].sum() / V_all)
    rho_iso = float(L["Viso"][idx_next].sum() / V_all)
    rkey = f"{t1}_{t2}"
    rvals = dict(zip([str(x) for x in L["rnum_keys"]], L["rnum_vals"]))
    r_all = float(rvals[rkey]) / dall
    ero_all = float(2 * np.nansum(a) / dall)
    floor_all = 2 * np.sqrt(float(np.nansum(L2 * D2)) / DIM) / dall
    Minv = L["Sprev"][idx, 3] + L["Sprev"][idx, 4]
    Mvis = L["Sprev"][idx, 0] + L["Sprev"][idx, 1] + L["Sprev"][idx, 2]
    a1_ok = np.isfinite(L["a1"][idx]).all(1)
    a50_ok = np.isfinite(L["a50"][idx]).all(1)
    out["whole"] = {
        "rho": rho, "rho_iso": rho_iso, "rho_over_iso": rho / rho_iso,
        "V_all_per_task": float(L["V"][idx].sum() / len(idx)),
        "D2_per_task": float(D2.mean()),
        "N_all_mean": float(np.nanmean(L["N_all"][idx])),
        "r_all": r_all, "ero_all_r_minus_1": r_all - 1, "ero_all": ero_all, "floor_all": float(floor_all),
        "ero_in_all": float(L["e"][idx].sum() / dall),
        "q1_all": float(np.nansum(L["a1"][idx][a1_ok]) / d[a1_ok].sum()) if a1_ok.any() else np.nan,
        "q50_all": float(np.nansum(L["a50"][idx][a50_ok]) / d[a50_ok].sum()) if a50_ok.any() else np.nan,
        "S2_mean": float(L["S2"][idx].mean()),
        "Minv_mean": float(Minv.mean()), "Mvis_mean": float(Mvis.mean()),
        "Minv_frac": float(Minv.sum() / L["S2"][idx].sum()),
    }
    return out


def agg(vals):
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], dtype=float)
    if v.size == 0:
        return (np.nan, np.nan, np.nan)
    return (float(np.median(v)), float(v.min()), float(v.max()))


def spearman(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    return float((rx * ry).sum() / np.sqrt((rx * rx).sum() * (ry * ry).sum()))


def fmt(x, p=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:.{p}f}"


def fmts(x, p=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:+.{p}f}"


def fig_series(L_by_seed, key_fn, seeds):
    arr = np.full((len(seeds), TMAX + 1), np.nan)
    for i, s in enumerate(seeds):
        arr[i] = key_fn(L_by_seed[s])
    return arr


def draw_figure(all_L, seeds, out_png, rho_iso_med):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    COL = {"none": "#2a78d6", "l2init": "#eb6834"}
    NAME = {"none": "介入なし", "l2init": "L2 Init"}
    ts = np.arange(1, TMAX)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), dpi=160)
    ax = axes[0]
    for arm in ARMS:
        rho = fig_series(all_L, lambda L: (L[arm]["Vnext_all"][:TMAX + 1] / np.nansum(L[arm]["V"], axis=1)), seeds)[:, 1:TMAX]
        med = np.nanmedian(rho, axis=0); lo = np.nanmin(rho, axis=0); hi = np.nanmax(rho, axis=0)
        ax.fill_between(ts, lo, hi, color=COL[arm], alpha=0.18, linewidth=0)
        ax.plot(ts, med, color=COL[arm], lw=1.6, label=NAME[arm])
        ax.annotate(NAME[arm], xy=(ts[-1], med[-1]), xytext=(6, 0), textcoords="offset points",
                    color="#222222", fontsize=10, va="center")
    ax.axhline(0.2, color="#888888", lw=1, ls=(0, (4, 3)))
    ax.text(3, 0.2, "親の予測の線 1/5", color="#555555", fontsize=9, va="bottom")
    ax.axhline(1.0, color="#bbbbbb", lw=1)
    ax.set_yscale("log")
    ax.set_xlim(1, TMAX + 45)
    ax.set_xlabel("増分 D_t の課題 t（次の課題 t+1 の入力で測る）")
    ax.set_ylabel("可視度 ρ(t)（対数）")
    ax.set_title("前の課題の増分は次の課題の入力にどれだけ見えるか", fontsize=11, loc="left")
    ax.legend(frameon=False, loc="upper right")
    ax = axes[1]
    for arm in ARMS:
        mi = fig_series(all_L, lambda L: L[arm]["Sprev"][:, 3] + L[arm]["Sprev"][:, 4], seeds)[:, 1:]
        ts2 = np.arange(1, TMAX + 1)
        med = np.nanmedian(mi, axis=0); lo = np.nanmin(mi, axis=0); hi = np.nanmax(mi, axis=0)
        ax.fill_between(ts2, lo, hi, color=COL[arm], alpha=0.18, linewidth=0)
        ax.plot(ts2, med, color=COL[arm], lw=1.6, label=NAME[arm])
        ax.annotate(NAME[arm], xy=(ts2[-1], med[-1]), xytext=(6, 0), textcoords="offset points",
                    color="#222222", fontsize=10, va="center")
    ax.set_xlim(1, TMAX + 45)
    ax.set_xlabel("課題 t")
    ax.set_ylabel("見えない質量 M_inv(t)")
    ax.set_title("過去の堆積のうち課題 t の low 以下の帯にある分", fontsize=11, loc="left")
    ax.legend(frameon=False, loc="upper left")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#eeeeee", lw=0.8)
    fig.suptitle(f"PM 第 1 層の帳簿（{len(seeds)} seed の中央値、帯は seed の最小〜最大、事後）", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_png)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--box-check", action="store_true")
    a = ap.parse_args()
    if a.box_check:
        seeds = seeds_available()
        print("seeds available:", seeds)
        box = box_table(seeds)
        print_box(box)
        return

    seeds = [s for s in seeds_available() if (LEDGER / f"seed{s}.npz").exists()]
    print("seeds with ledger:", seeds)
    OUT.mkdir(parents=True, exist_ok=True)
    box = box_table(seeds)
    verdict_box = print_box(box)
    all_L, stats = {}, {}
    for s in seeds:
        L, meta = load_ledger(s)
        all_L[s] = L
        stats[s] = {arm: {w: window_stats(L[arm], meta, t1, t2, w) for w, t1, t2 in WINDOWS} for arm in ARMS}
    n = len(seeds)

    # S-ledger: identity and transfer-matrix checks
    res_ratio, max_res, tcheck, dcheck = [], [], [], []
    for s in seeds:
        L, _ = load_ledger(s)
        for arm in ARMS:
            r = np.abs(L[arm]["res"][1:]); b = L[arm]["resbound"][1:]
            res_ratio.append(float(np.nanmax(r / b)))
            max_res.append(float(np.nanmax(r)))
            T = L[arm]["T"][1:TMAX]
            d = L[arm]["d"][1:TMAX]
            dcheck.append(float(np.nanmax(np.abs(T.sum(2) - d) / np.maximum(d, 1e-300))))
            for w, _, _ in WINDOWS:
                W = stats[s][arm][w]["whole"]
                tcheck.append(abs(W["ero_all"] - W["ero_all_r_minus_1"]))
    s_ledger = {"max_res_over_bound": max(res_ratio), "max_res_abs": max(max_res),
                "max_abs_ero_all_vs_r_minus_1": max(tcheck), "max_rel_T_row_sum_vs_d": max(dcheck)}
    print("S-ledger:", s_ledger)
    if s_ledger["max_res_over_bound"] > 1.0:
        raise SystemExit("S-ledger: identity residual over the bound of spec §6")

    # tables
    band_rows, win_rows = [], []
    metrics_band = ["dim_frac", "lam_share", "d_share", "V_share", "S_share", "ero_in", "ero_F", "floor",
                    "f_b", "q1", "q50", "rho_b", "dnext_share"]
    metrics_win = ["rho", "rho_iso", "rho_over_iso", "V_all_per_task", "D2_per_task", "N_all_mean", "r_all",
                   "ero_all_r_minus_1", "ero_all", "floor_all", "ero_in_all", "q1_all", "q50_all", "S2_mean",
                   "Minv_mean", "Mvis_mean", "Minv_frac"]
    for arm in ARMS:
        for w, t1, t2 in WINDOWS:
            for b in BANDS:
                row = {"arm": arm, "window": w, "t1": t1, "t2": t2, "band": b, "n_seeds": n}
                for m in metrics_band:
                    med, lo, hi = agg([stats[s][arm][w]["bands"][b][m] for s in seeds])
                    row[m] = med; row[m + "_min"] = lo; row[m + "_max"] = hi
                band_rows.append(row)
            row = {"arm": arm, "window": w, "t1": t1, "t2": t2, "n_seeds": n}
            for m in metrics_win:
                med, lo, hi = agg([stats[s][arm][w]["whole"][m] for s in seeds])
                row[m] = med; row[m + "_min"] = lo; row[m + "_max"] = hi
            win_rows.append(row)

    def write(path, rows):
        with open(path, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            for r in rows:
                wr.writerow({k: (f"{v:.10g}" if isinstance(v, float) else v) for k, v in r.items()})
    write(OUT / "table_band_window_arm.csv", band_rows)
    write(OUT / "table_window_arm.csv", win_rows)

    # transfer matrix (row-normalized, seed-median), window w4 only for the md
    def T_med(arm, w):
        M = np.array([[stats[s][arm][w]["bands"][b]["T_row"] for b in BANDS] for s in seeds])
        return np.nanmedian(M, axis=0)

    # per-seed table and (d)
    per_seed = []
    dG, dM, GN, GL, MN, ML, RM = [], [], [], [], [], [], []
    for s in seeds:
        rn = run_values("none", s); rl = run_values("l2init", s)
        G_n, G_l = G_of(rn), G_of(rl)
        L = all_L[s]
        M_at = lambda arm, t: float(L[arm]["Sprev"][t, 3] + L[arm]["Sprev"][t, 4])  # noqa: E731
        M_n = float(np.mean([M_at("none", 400), M_at("none", 500)]))
        M_l = float(np.mean([M_at("l2init", 400), M_at("l2init", 500)]))
        Mw_n = float(np.mean(L["none"]["Sprev"][451:501, 3] + L["none"]["Sprev"][451:501, 4]))
        Mw_l = float(np.mean(L["l2init"]["Sprev"][451:501, 3] + L["l2init"]["Sprev"][451:501, 4]))
        R_M = Mw_l / Mw_n
        dG.append(G_n - G_l); dM.append(M_n - M_l)
        GN.append(G_n); GL.append(G_l); MN.append(M_n); ML.append(M_l); RM.append(R_M)
        per_seed.append({"seed": s, "G_none": G_n, "G_l2init": G_l, "dG": G_n - G_l, "Minv400_500_none": M_n,
                         "Minv400_500_l2init": M_l, "dM": M_n - M_l, "R_M_w4": R_M,
                         "online_avg_none": float(np.nanmean(rn["online"][1:])),
                         "online_avg_l2init": float(np.nanmean(rl["online"][1:]))})
    write(OUT / "per_seed.csv", per_seed)

    # ------------------------------------------------------------- predictions
    def per_window_ok(arm, key, fn):
        return {w: fn(next(r[key] for r in win_rows if r["arm"] == arm and r["window"] == w)) for w in FOUR}

    def grade(okmap):
        v = list(okmap.values())
        return "○" if all(v) else ("△" if any(v) else "×")

    P = {}
    rho_ok = {arm: per_window_ok(arm, "rho", lambda x: x <= 0.2) for arm in ARMS}
    P["P1"] = {"text": "可視度は 1/5 以下に落ちる（none の腕・全 4 窓の seed 中央値 ρ_w ≤ 1/5）",
               "windows_none": rho_ok["none"], "windows_l2init": rho_ok["l2init"],
               "grade": grade(rho_ok["none"]), "grade_l2init": grade(rho_ok["l2init"])}
    ero_ok = per_window_ok("none", "ero_all", lambda x: abs(x) <= 0.1)
    P["P2"] = {"text": "D_t の侵食率は |e| ≤ 0.1（none の腕・全 4 窓の seed 中央値 |ero_all(w)| ≤ 0.1）",
               "windows_none": ero_ok, "grade": grade(ero_ok)}
    P["P3"] = {"text": "L2 Init の見えない質量は介入なしの 1/5 以下（w4 の比 R_M の seed 中央値）",
               "median_R_M": float(np.median(RM)), "n_seeds_ok": int(np.sum(np.array(RM) <= 0.2)), "n": n,
               "grade": "○" if np.median(RM) <= 0.2 else "×"}
    if n == 10:
        rho_s = spearman(dM, dG)
        P4_grade = "○" if rho_s >= RHO_CRIT_10 else ("△" if rho_s > 0 else "×")
    else:
        rho_s = spearman(dM, dG) if n >= 3 else float("nan")
        P4_grade = "未採点（10 seed が揃っていない）"
    P["P4"] = {"text": "(d) の順位は一致する（ΔM と ΔG の Spearman、線 0.5636、n = 10）",
               "rho_s": rho_s, "n": n, "line": RHO_CRIT_10, "grade": P4_grade}
    # S-read checks (reading of the parent; not predictions)
    rd = {}
    w = "w4"
    ero_in = {arm: {b: next(r["ero_in"] for r in band_rows if r["arm"] == arm and r["window"] == w and r["band"] == b) for b in BANDS} for arm in ARMS}
    order_none = ero_in["none"]["top"] < ero_in["none"]["mid1"] < ero_in["none"]["mid2"] < ero_in["none"]["low"]
    rd["削りは見えやすさの順（none・w4・課題内の侵食率で top<mid1<mid2<low）"] = bool(order_none)
    rd["L2 Init の low の課題内の侵食率は none より強い（w4）"] = bool(ero_in["l2init"]["low"] < ero_in["none"]["low"])
    rd["L2 Init の comp の課題内の侵食率は none より強い（w4）"] = bool(ero_in["l2init"]["comp"] < ero_in["none"]["comp"])
    # post hoc (not in the spec): erosion as a fraction of the standing stock, f_b = -Σe_b / (2 ΣN_b(t-1))
    f_w4 = {arm: {b: next(r["f_b"] for r in band_rows if r["arm"] == arm and r["window"] == "w4" and r["band"] == b)
                  for b in BANDS} for arm in ARMS}
    rd_post = {
        "事後: none・w4 の在庫に対する消す割合 f は mid1 > mid2 > low ≥ comp（top は別）":
            bool(f_w4["none"]["mid1"] > f_w4["none"]["mid2"] > f_w4["none"]["low"] >= f_w4["none"]["comp"]),
        "事後: none・w4 の f_top は 0.14（RL-CIFAR の top）より小さい":
            bool(f_w4["none"]["top"] < 0.14),
        "事後: L2 Init の low の f は none の low の 100 倍より大きい（w4）":
            bool(f_w4["l2init"]["low"] > 100 * max(f_w4["none"]["low"], 1e-12)),
    }

    # overall
    if all(P[p]["grade"] == "○" for p in ("P1", "P2", "P3", "P4")):
        verdict = "READING_SUPPORTED"
    elif P["P1"]["grade"] == "○" and P["P3"]["grade"] == "○":
        verdict = "TRANSPORT_AND_L2_ONLY"
    else:
        verdict = "NOT_SUPPORTED"
    if n < 10:
        verdict = f"(未確定: seed {n}/10) {verdict}"
    judge = {"seeds": seeds, "predictions": P, "reading_checks": rd, "posthoc_checks": rd_post, "verdict": verdict,
             "S_ledger": s_ledger, "S_box": {"lines_ok": verdict_box, "tolerance": tol_of(n),
                                             "box": box}}
    (OUT / "judge.json").write_text(json.dumps(judge, indent=1, ensure_ascii=False, default=float))

    # ------------------------------------------------------------- figure
    draw_figure(all_L, seeds, OUT / "fig_pm_ledger_1108.png", None)

    # ------------------------------------------------------------- summary.md
    L = []
    w_ = L.append
    w_("# pm_ledger_1108 結果の表（自動生成・report.py）")
    w_("")
    w_(f"seed {seeds}（{n} 本）、腕 none / l2init、数は seed の中央値（[最小, 最大] は CSV）。表の丸めは最後の 1 回だけ。")
    w_("")
    w_("## 1. 予測の採点")
    w_("")
    w_("| 予測 | 文 | 採点 |")
    w_("|---|---|---|")
    w_(f"| P1 | {P['P1']['text']} | none: **{P['P1']['grade']}** / l2init: {P['P1']['grade_l2init']} |")
    w_(f"| P2 | {P['P2']['text']} | **{P['P2']['grade']}** |")
    w_(f"| P3 | {P['P3']['text']} | **{P['P3']['grade']}**（R_M 中央値 {P['P3']['median_R_M']:.3f}、{P['P3']['n_seeds_ok']}/{P['P3']['n']} seed が ≤ 1/5） |")
    rs = P["P4"]["rho_s"]
    w_(f"| P4 | {P['P4']['text']} | **{P['P4']['grade']}**（ρ_s = {rs:+.3f}、n = {n}） |")
    w_(f"| 合わせた判定 | §5 の規則 | **{verdict}** |")
    w_("")
    w_("窓ごとの P1 の ρ_w と P2 の ero_all（none、seed 中央値）:")
    w_("")
    w_("| 窓 | ρ_w [最小, 最大]（≤0.2?） | ero_all [最小, 最大]（|·|≤0.1?） |")
    w_("|---|---|---|")
    for w in FOUR:
        r = next(x for x in win_rows if x["arm"] == "none" and x["window"] == w)
        w_(f"| {w} ({r['t1']}–{r['t2']}) | {r['rho']:.4f} [{r['rho_min']:.4f}, {r['rho_max']:.4f}]（{'○' if rho_ok['none'][w] else '×'}） | {r['ero_all']:+.3f} [{r['ero_all_min']:+.3f}, {r['ero_all_max']:+.3f}]（{'○' if ero_ok[w] else '×'}） |")
    w_("")
    w_("読みの検査（予測ではない。spec の定義で採点。ero_in は堆積で割った量）:")
    w_("")
    for k, v in rd.items():
        w_(f"- {k}: {'成立' if v else '不成立'}")
    w_("")
    w_("事後（spec の外。在庫 N で割った消す割合 f_b = −Σe_b / 2ΣN_b(t−1)。0922 深夜 追記と同じ定義。参照 RL-CIFAR LR std t41–50: top 0.140・mid1 0.032・mid2 0.010・low 0.0036・comp 0.0007）:")
    w_("")
    for k, v in rd_post.items():
        w_(f"- {k}: {'成立' if v else '不成立'}")
    w_("")
    w_("## 2. 窓 × 腕の表")
    w_("")
    w_("| 腕 | 窓 | ρ_w | ρ_iso | ρ/ρ_iso | r_all | ero_all(=r−1) | 雑音の床 | 課題内 Σe/Σd(全帯) | q(1) | q(50) | ‖W‖²平均 | ‖S‖²平均 | M_inv | M_vis | M_inv/‖S‖² | D²/課題 | V/課題 |")
    w_("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for arm in ARMS:
        for w, t1, t2 in WINDOWS:
            r = next(x for x in win_rows if x["arm"] == arm and x["window"] == w)
            w_(f"| {arm} | {w} {t1}–{t2} | {r['rho']:.4f} | {r['rho_iso']:.4f} | {r['rho_over_iso']:.3f} | {r['r_all']:.3f} | {r['ero_all_r_minus_1']:+.3f} | {r['floor_all']:.4f} | {r['ero_in_all']:+.3f} | {r['q1_all']:+.4f} | {fmts(r['q50_all'])} | {r['N_all_mean']:.1f} | {r['S2_mean']:.1f} | {r['Minv_mean']:.1f} | {r['Mvis_mean']:.1f} | {r['Minv_frac']:.3f} | {r['D2_per_task']:.2f} | {r['V_all_per_task']:.2f} |")
    w_("")
    w_("RL-CIFAR の参照（0922 LR std・t6–50）: top −0.925・mid1 −0.683・mid2 −0.403・low −0.194・comp −0.053・全帯 −0.607。")
    w_("")
    w_("## 3. 帯 × 窓 × 腕の表")
    w_("")
    w_("| 腕 | 窓 | 帯 | 次元比 | λ比 | 堆積比 d | 重み比 V | 在庫比 S | ero_in | ero_F | 床 | f_b（事後） | q(1) | q(50) | ρ_b | 次の課題での質量比 |")
    w_("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for arm in ARMS:
        for w, t1, t2 in WINDOWS:
            for b in BANDS:
                r = next(x for x in band_rows if x["arm"] == arm and x["window"] == w and x["band"] == b)
                w_(f"| {arm} | {w} {t1}–{t2} | {b} | {r['dim_frac']:.3f} | {r['lam_share']:.4f} | {r['d_share']:.3f} | {r['V_share']:.3f} | {r['S_share']:.3f} | {r['ero_in']:+.3f} | {r['ero_F']:+.3f} | {r['floor']:.3f} | {r['f_b']:+.5f} | {r['q1']:+.4f} | {fmts(r['q50'])} | {fmt(r['rho_b'])} | {r['dnext_share']:.3f} |")
    w_("")
    w_("## 4. 移送の行列（課題 t の帯 b の増分のうち、次の課題 t+1 の帯 b′ に在る割合、行で正規化、seed 中央値、none と l2init、w4）")
    w_("")
    for arm in ARMS:
        M = T_med(arm, "w4")
        w_(f"**{arm}**")
        w_("")
        w_("| 帯 b \\ b′ | " + " | ".join(BANDS) + " |")
        w_("|---|" + "---|" * NB)
        for i, b in enumerate(BANDS):
            w_(f"| {b} | " + " | ".join(f"{M[i, j]:.3f}" for j in range(NB)) + " |")
        w_("")
    w_("## 5. seed ごと（G は課題 400・500 の平均の差・ポイント、M は課題 400・500 の平均の見えない質量）")
    w_("")
    w_("| seed | G(none) | G(l2init) | ΔG | M(none) | M(l2init) | ΔM | R_M(w4) |")
    w_("|---|---|---|---|---|---|---|---|")
    for r in per_seed:
        w_(f"| {r['seed']} | {r['G_none']:+.2f} | {r['G_l2init']:+.2f} | {r['dG']:+.2f} | {r['Minv400_500_none']:.1f} | {r['Minv400_500_l2init']:.1f} | {r['dM']:.1f} | {r['R_M_w4']:.3f} |")
    w_("")
    w_(f"Spearman ρ_s(ΔM, ΔG) = {rs:+.3f}（n = {n}）。")
    w_("")
    w_("## 6. 検査")
    w_("")
    w_(f"- S-ledger: 恒等式の残差の最大 / §6 の線 = {s_ledger['max_res_over_bound']:.3g}（線の中）、絶対値 {s_ledger['max_res_abs']:.3g}。")
    w_(f"- ero_all と r_all − 1 の差の最大 {s_ledger['max_abs_ero_all_vs_r_minus_1']:.3g}。移送行列の行和と d の相対差の最大 {s_ledger['max_rel_T_row_sum_vs_d']:.3g}。")
    w_(f"- S-box: 線 ±{tol_of(n):.2f}。none {'OK' if verdict_box['none'] else 'NG'}・l2init {'OK' if verdict_box['l2init'] else 'NG'}（詳細は judge.json の S_box）。")
    (OUT / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
