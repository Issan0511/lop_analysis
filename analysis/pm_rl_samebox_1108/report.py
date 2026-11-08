"""pm_rl_samebox_1108 report: tables, predictions P9-P12, verdicts, the one figure.

Reads the runs (obsidian-research-data/pm_rl_samebox_1108/runs/<seq>_<act>/seed*/) and, for the
perm10000 reference rows, the pm_ledger_1108 (none) and pm_ledger_add_1108 (CHB) runs.
Numbers are rounded once, when written.

Usage:
  .venv/bin/python analysis/pm_rl_samebox_1108/report.py [--allow-partial]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DATA = Path("/home/issan/Projects/obsidian-research-data")
RUNS = DATA / "pm_rl_samebox_1108" / "runs"
REF_DIR = {"none": DATA / "pm_ledger_1108" / "runs" / "none", "CHB": DATA / "pm_ledger_add_1108" / "runs" / "CHB"}
OUT = REPO / "results" / "pm_rl_samebox_1108"
CHECKS = OUT / "checks.json"

ARMS = [("perm1200", "none"), ("perm1200", "CHB"), ("rl1200", "none"), ("rl1200", "CHB")]
REFS = [("perm10000", "none"), ("perm10000", "CHB")]
WINDOWS = {"w1": (6, 50), "w2": (101, 150), "w3": (251, 300), "w4": (451, 500)}
METRICS = ["zsd_l1", "zsd_l1_tr", "zbar_l1", "zbar_l1_tr", "w_norm_l1", "dead_frac_l1", "dead_frac_l2", "dead_frac_l1_tr",
           "dead_frac_l2_tr", "online", "acc_tr", "eff_rank_l1", "zsd_l2", "zbar_l2", "w_norm_l2"]
TMAX = 500
NAME = {("perm1200", "none"): "perm1200 none", ("perm1200", "CHB"): "perm1200 CHB", ("rl1200", "none"): "rl1200 none",
        ("rl1200", "CHB"): "rl1200 CHB", ("perm10000", "none"): "perm10000 none（参照）", ("perm10000", "CHB"): "perm10000 CHB（参照）"}


def arm_dir(seq, act, seed):
    return (REF_DIR[act] if seq == "perm10000" else RUNS / f"{seq}_{act}") / f"seed{seed}"


def load(seq, act, seed):
    d = arm_dir(seq, act, seed)
    return pd.read_csv(d / "per_task.csv"), pd.read_csv(d / "fresh.csv")


def window_means(df):
    out = {}
    for wn, (a, b) in WINDOWS.items():
        g = df[(df.task >= a) & (df.task <= b)]
        out[wn] = {m: float(g[m].mean()) if m in g.columns else float("nan") for m in METRICS}
    return out


def safe_ratio(b, a):
    return b / a if (a == a and b == b and a != 0.0) else float("nan")


def seed_stats(seq, act, seed):
    df, fr = load(seq, act, seed)
    w = window_means(df)
    onl = df.set_index("task")["online"] * 100
    frs = fr.set_index("task")["online_fresh"] * 100
    gaps = {int(t): float(frs[t] - onl[t]) for t in frs.index}
    G = float(np.mean([gaps[400], gaps[500]]))
    dead = df[df.task <= 50]
    over = dead[(dead.dead_frac_l1 > 0.5) | (dead.dead_frac_l2 > 0.5)]
    first = int(over.task.iloc[0]) if len(over) else 0
    row = {"seq": seq, "act": act, "seed": seed, "G": G, "online_avg": float(onl.mean()),
           "dead500_l1": float(df.set_index("task").loc[500, "dead_frac_l1"] * 100),
           "dead500_l2": float(df.set_index("task").loc[500, "dead_frac_l2"] * 100),
           "dead50_over": bool(len(over)), "dead50_first_task": first,
           "width_ratio": safe_ratio(w["w4"]["zsd_l1"], w["w1"]["zsd_l1"]),
           "norm_ratio": safe_ratio(w["w4"]["w_norm_l1"], w["w1"]["w_norm_l1"]),
           "width_ratio_tr": safe_ratio(w["w4"]["zsd_l1_tr"], w["w1"]["zsd_l1_tr"])}
    for t, v in gaps.items():
        row[f"gap{t}"] = v
    # reference only (not scored): the windows of the RL-CIFAR numbers in width_posthoc.md
    def ww(m, a, b):
        return float(df[(df.task >= a) & (df.task <= b)][m].mean())
    for a, b, c, d, tag in ((11, 30, 181, 200, "t11_30"), (1, 10, 181, 200, "t1_10")):
        row[f"width_ratio_{tag}"] = safe_ratio(ww("zsd_l1", c, d), ww("zsd_l1", a, b))
        row[f"norm_ratio_{tag}"] = safe_ratio(ww("w_norm_l1", c, d), ww("w_norm_l1", a, b))
    row["width_over_norm"] = safe_ratio(row["width_ratio"], row["norm_ratio"])
    for wn in WINDOWS:
        for m in METRICS:
            row[f"{m}_{wn}"] = w[wn][m]
    return row


def agg(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    return float(np.median(x)), float(x.min()), float(x.max())


def fmt(x, nd=2):
    return "—" if x != x else f"{x:.{nd}f}"


def fmt_range(x, nd=2):
    m, lo, hi = agg(x)
    return "—" if m != m else f"{m:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args()
    seeds = [s for s in range(10) if all((arm_dir(q, a, s) / "provenance.json").exists() for q, a in ARMS)]
    print("seeds with all 4 arms:", seeds)
    if len(seeds) < 10 and not args.allow_partial:
        raise SystemExit("need all 10 seeds (or --allow-partial)")
    n = len(seeds)
    OUT.mkdir(parents=True, exist_ok=True)
    S = {}
    for q, a in ARMS + REFS:
        S[(q, a)] = [seed_stats(q, a, s) for s in seeds]
    pd.DataFrame([r for k in S for r in S[k]]).to_csv(OUT / "per_seed.csv", index=False, float_format="%.10g")

    # ---- arm x window table
    rows = []
    for key in ARMS + REFS:
        for wn, (a, b) in WINDOWS.items():
            row = {"seq": key[0], "act": key[1], "window": wn, "t1": a, "t2": b, "n_seeds": n}
            for m in METRICS:
                med, lo, hi = agg([r[f"{m}_{wn}"] for r in S[key]])
                row[m] = med; row[m + "_min"] = lo; row[m + "_max"] = hi
            rows.append(row)
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "table_window_arm.csv", index=False, float_format="%.10g")

    def col(key, name):
        return np.array([r[name] for r in S[key]], dtype=float)

    # ---- predictions
    P = {}
    rp, rr = col(("perm1200", "CHB"), "width_ratio"), col(("rl1200", "CHB"), "width_ratio")
    rpt, rrt = col(("perm1200", "CHB"), "width_ratio_tr"), col(("rl1200", "CHB"), "width_ratio_tr")
    c1 = int((rp >= 1.8).sum()); c2 = int((rr <= 1.25).sum()); c3 = int((rp / rr >= 1.5).sum())
    P9_ok = c1 == n and c2 >= 0.8 * n and c3 == n
    P["P9"] = {"text": "CHB の幅の比は perm1200 で ≥ 1.8（10/10）、rl1200 で ≤ 1.25（≥ 8/10）、比の比 perm/rl ≥ 1.5（10/10）",
               "perm_ge_1_8": c1, "rl_le_1_25": c2, "ratio_of_ratios_ge_1_5": c3, "n": n,
               "perm_ratio_median": float(np.median(rp)), "rl_ratio_median": float(np.median(rr)),
               "ror_median": float(np.median(rp / rr)), "grade": "○" if P9_ok else "×",
               "tr_columns_reference": {"perm_ge_1_8": int((rpt >= 1.8).sum()), "rl_le_1_25": int((rrt <= 1.25).sum()),
                                        "ratio_of_ratios_ge_1_5": int((rpt / rrt >= 1.5).sum()),
                                        "perm_ratio_median": float(np.median(rpt)), "rl_ratio_median": float(np.median(rrt))}}
    np_, nr_ = col(("perm1200", "CHB"), "norm_ratio"), col(("rl1200", "CHB"), "norm_ratio")
    c1 = int((np_ >= 2.0).sum()); c2 = int((nr_ >= 1.3).sum())
    P10_ok = c1 >= 0.8 * n and c2 >= 0.8 * n
    P["P10"] = {"text": "CHB のノルムの比は perm1200 で ≥ 2.0、rl1200 で ≥ 1.3（各 ≥ 8/10）", "perm_ge_2": c1, "rl_ge_1_3": c2, "n": n,
                "perm_ratio_median": float(np.median(np_)), "rl_ratio_median": float(np.median(nr_)), "grade": "○" if P10_ok else "×"}
    Gp, Gr = col(("perm1200", "CHB"), "G"), col(("rl1200", "CHB"), "G")
    c1 = int((Gp >= 0.5).sum()); c2 = int((Gr < 0.5).sum())
    P11_ok = c1 >= 0.9 * n and c2 >= 0.8 * n
    P["P11"] = {"text": "CHB の G は perm1200 で ≥ +0.5（≥ 9/10）、rl1200 で < +0.5（≥ 8/10）", "perm_ge_0_5": c1, "rl_lt_0_5": c2, "n": n,
                "G_perm_mean": float(Gp.mean()), "G_rl_mean": float(Gr.mean()), "grade": "○" if P11_ok else "×"}
    dead_rl = np.array([r["dead50_over"] for r in S[("rl1200", "none")]], dtype=bool)
    wn_p = col(("perm1200", "none"), "width_ratio")
    c1 = int(dead_rl.sum()); c2 = int((wn_p >= 2.0).sum())
    P12_ok = c1 >= 0.8 * n and c2 >= 0.8 * n
    P["P12"] = {"text": "rl1200 の none は第 1 層か第 2 層の死が課題 50 までに 50% を超える（≥ 8/10）、perm1200 の none は幅の比 ≥ 2.0（≥ 8/10）",
                "rl_none_dead_over_50": c1, "perm_none_width_ge_2": c2, "n": n,
                "first_tasks": [r["dead50_first_task"] for r in S[("rl1200", "none")]],
                "perm_none_width_ratio_median": float(np.median(wn_p)), "grade": "○" if P12_ok else "×"}
    # ---- verdicts
    if int((rp >= 1.8).sum()) < 0.8 * n:
        V = "PERM1200_DIFFERS"
    elif int((rr > 1.25).sum()) >= 0.8 * n:
        V = "BOTH_GROW"
    elif P["P9"]["grade"] == "○" and P["P10"]["grade"] == "○":
        V = "WIDTH_CONTRAST"
    else:
        V = "MIXED"
    if P["P11"]["grade"] == "○":
        VL = "LOSS_PERM_ONLY"
    elif int((Gp >= 0.5).sum()) >= 0.9 * n and int((Gr >= 0.5).sum()) >= 0.8 * n:
        VL = "LOSS_BOTH"
    elif int((Gp < 0.5).sum()) >= 0.8 * n:
        VL = "NO_LOSS_PERM1200"
    else:
        VL = "LOSS_MIXED"
    checks = json.loads(CHECKS.read_text()) if CHECKS.exists() else {}
    judge = {"seeds": seeds, "predictions": P, "verdict_width": V, "verdict_loss": VL,
             "checks_pass": {k: bool(v.get("pass")) for k, v in checks.items()}}
    (OUT / "judge.json").write_text(json.dumps(judge, indent=1, ensure_ascii=False, default=float))

    # ---- summary.md
    L = []; w_ = L.append
    w_("# pm_rl_samebox_1108 結果の表（自動生成・report.py）"); w_("")
    w_(f"seed {seeds}（{n} 本）。perm10000 は pm_ledger_1108（none）・pm_ledger_add_1108（CHB）の走を参照の行として載せた（走り直していない）。")
    w_("幅 = `zsd_l1`（テスト 10,000 枚で測る、一次の値）、ノルム = `w_norm_l1`、平均 = `zbar_l1`。窓は w1 = 課題 6〜50、w2 = 101〜150、w3 = 251〜300、w4 = 451〜500。丸めは最後の 1 回だけ。"); w_("")
    w_("## 1. 予測の採点"); w_("")
    w_("| 予測 | 文 | 実測 | 採点 |"); w_("|---|---|---|---|")
    p = P["P9"]
    w_(f"| P9 | {p['text']} | perm1200 の比 ≥ 1.8: {p['perm_ge_1_8']}/{n}（中央値 {p['perm_ratio_median']:.2f}）、rl1200 の比 ≤ 1.25: {p['rl_le_1_25']}/{n}（中央値 {p['rl_ratio_median']:.2f}）、比の比 ≥ 1.5: {p['ratio_of_ratios_ge_1_5']}/{n}（中央値 {p['ror_median']:.2f}） | **{p['grade']}** |")
    p = P["P10"]
    w_(f"| P10 | {p['text']} | perm1200 の比 ≥ 2.0: {p['perm_ge_2']}/{n}（中央値 {p['perm_ratio_median']:.2f}）、rl1200 の比 ≥ 1.3: {p['rl_ge_1_3']}/{n}（中央値 {p['rl_ratio_median']:.2f}） | **{p['grade']}** |")
    p = P["P11"]
    w_(f"| P11 | {p['text']} | perm1200 の G ≥ +0.5: {p['perm_ge_0_5']}/{n}（平均 {p['G_perm_mean']:+.2f}）、rl1200 の G < +0.5: {p['rl_lt_0_5']}/{n}（平均 {p['G_rl_mean']:+.2f}） | **{p['grade']}** |")
    p = P["P12"]
    w_(f"| P12 | {p['text']} | rl1200 none の死 > 50%（課題 50 まで）: {p['rl_none_dead_over_50']}/{n}、perm1200 none の幅の比 ≥ 2.0: {p['perm_none_width_ge_2']}/{n}（中央値 {p['perm_none_width_ratio_median']:.2f}） | **{p['grade']}** |")
    w_(f"| 判定（幅） | `verdict_width` | | **{V}** |")
    w_(f"| 判定（喪失） | `verdict_loss` | | **{VL}** |"); w_("")
    w_("## 2. 腕 × 窓（seed の中央値）"); w_("")
    w_("| 腕 | 窓 | 幅 l1 | 平均 l1 | ノルム l1 | 死 l1 | 死 l2 | online | 課題末の訓練正答率 | 幅 l1（課題の 1,200 枚） |")
    w_("|---|---|---|---|---|---|---|---|---|---|")
    for key in ARMS + REFS:
        for wn, (a, b) in WINDOWS.items():
            r = tab[(tab.seq == key[0]) & (tab.act == key[1]) & (tab.window == wn)].iloc[0]
            w_(f"| {NAME[key]} | {wn} {a}–{b} | {fmt(r.zsd_l1)} | {fmt(r.zbar_l1)} | {fmt(r.w_norm_l1)} | {fmt(r.dead_frac_l1)} | {fmt(r.dead_frac_l2)} | {fmt(r.online*100,2)}% | {fmt(r.acc_tr,3)} | {fmt(r.zsd_l1_tr)} |")
    w_(""); w_("## 3. 比と喪失（seed ごとの比の中央値 [最小, 最大]、G は平均 [最小, 最大]）"); w_("")
    w_("| 腕 | 幅の比 w1→w4 | ノルムの比 w1→w4 | 幅の比（課題の 1,200 枚） | G | 全課題の平均 online | 課題 500 の死 %（層 1/2、平均） | 死 > 50%（課題 50 まで） |")
    w_("|---|---|---|---|---|---|---|---|")
    for key in ARMS + REFS:
        G = col(key, "G")
        d1 = np.mean([r["dead500_l1"] for r in S[key]]); d2 = np.mean([r["dead500_l2"] for r in S[key]])
        dn = int(sum(r["dead50_over"] for r in S[key]))
        w_(f"| {NAME[key]} | {fmt_range(col(key,'width_ratio'))} | {fmt_range(col(key,'norm_ratio'))} | {fmt_range(col(key,'width_ratio_tr'))} | {G.mean():+.2f} [{G.min():+.2f}, {G.max():+.2f}] | {np.mean([r['online_avg'] for r in S[key]]):.2f}% | {d1:.1f}/{d2:.1f} | {dn}/{n} |")
    w_(""); w_(f"CHB の比の比 perm1200 / rl1200（同じ seed）: 幅 {fmt_range(rp/rr)}、ノルム {fmt_range(np_/nr_)}。"); w_("")
    w_("## 3b. RL-CIFAR の数字と同じ窓の比と、幅の比 ÷ ノルムの比（CHB、参考・採点に使わない）"); w_("")
    w_("| 腕 | 幅の比 11–30→181–200 | ノルムの比 11–30→181–200 | 幅の比 1–10→181–200 | ノルムの比 1–10→181–200 | 幅の比 ÷ ノルムの比（w1→w4） |")
    w_("|---|---|---|---|---|---|")
    for key in [("perm1200", "CHB"), ("rl1200", "CHB"), ("perm10000", "CHB")]:
        w_(f"| {NAME[key]} | {fmt_range(col(key,'width_ratio_t11_30'))} | {fmt_range(col(key,'norm_ratio_t11_30'))} | {fmt_range(col(key,'width_ratio_t1_10'))} | {fmt_range(col(key,'norm_ratio_t1_10'))} | {fmt_range(col(key,'width_over_norm'))} |")
    wp = {}
    for line in (REPO / "results" / "pm_ledger_add_1108" / "width_posthoc.md").read_text(encoding="utf-8").splitlines():
        if line.startswith("| RL-CIFAR | CHB |"):
            c = [x.strip() for x in line.split("|")]
            wp[c[3]] = (c[4], c[5])
    w_(f"| RL-CIFAR CHB（別の箱、width_posthoc.md より） | {wp['t11–30→t181–200'][0]} | {wp['t11–30→t181–200'][1]} | {wp['t1–10→t181–200'][0]} | {wp['t1–10→t181–200'][1]} | — |")
    w_("")
    fst = P["P12"]["first_tasks"]
    w_(f"rl1200 none の死が 50% を超えた最初の課題（seed 0〜9）: {fst}。"); w_("")
    w_("## 4. 差 G の 5 点（課題 100〜500、10 seed の平均）"); w_("")
    w_("| 腕 | G(100) | G(200) | G(300) | G(400) | G(500) |"); w_("|---|---|---|---|---|---|")
    for key in ARMS + REFS:
        w_(f"| {NAME[key]} | " + " | ".join(f"{np.mean([r[f'gap{t}'] for r in S[key]]):+.2f}" for t in (100, 200, 300, 400, 500)) + " |")
    w_(""); w_("## 5. 課題の 1,200 枚で測った幅での P9 の参考（採点には使わない）"); w_("")
    t = P["P9"]["tr_columns_reference"]
    w_(f"perm1200 の比 ≥ 1.8: {t['perm_ge_1_8']}/{n}（中央値 {t['perm_ratio_median']:.2f}）、rl1200 の比 ≤ 1.25: {t['rl_le_1_25']}/{n}（中央値 {t['rl_ratio_median']:.2f}）、比の比 ≥ 1.5: {t['ratio_of_ratios_ge_1_5']}/{n}。"); w_("")
    w_("## 6. 検査（走る前の S1〜S4）"); w_("")
    for k, v in checks.items():
        w_(f"- {k}: {'PASS' if v.get('pass') else 'FAIL'}" + (f"（線 {v['line']:.4f}、rl1200 CHB の online {[round(x,4) for x in v['rl1200_CHB_online']]}）" if k == "S4" else "")
           + (f"（一致率の最大の外れ {v['agree_max_dev_task_pairs']:.4f}/{v['agree_max_dev_seed_pairs']:.4f}、線 {v['agree_line']:.4f}、{v['n_pairs']} 対）" if k == "S2_labels" else ""))
    (OUT / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    # ---- figure
    draw(seeds)
    print("wrote", OUT)


def draw(seeds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    COL = {"perm1200": "#2a78d6", "rl1200": "#eb6834", "perm10000": "#1baf7a"}
    LS = {"CHB": "-", "none": (0, (4, 2))}
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=160)
    ts = np.arange(1, TMAX + 1)
    for ax, m, lab in zip(axes, ("zsd_l1", "w_norm_l1"), ("第 1 層の実効的な幅（前活性の標準偏差）", "第 1 層のノルム（行ノルムの中央値）")):
        for key in REFS + ARMS:
            arr = []
            for s in seeds:
                df, _ = load(key[0], key[1], s)
                arr.append(df.set_index("task")[m].reindex(ts).to_numpy(dtype=float))
            med = np.nanmedian(np.array(arr), axis=0)
            ax.plot(ts, med, color=COL[key[0]], ls=LS[key[1]], lw=1.8 if key[1] == "CHB" else 1.3,
                    label=NAME[key].replace("（参照）", " 参照"))
        ax.set_xlabel("課題 t"); ax.set_title(lab, fontsize=11, loc="left")
        ax.set_ylim(bottom=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#eeeeee", lw=0.8)
    axes[0].legend(frameon=False, fontsize=9, ncols=1)
    fig.suptitle(f"pm_rl_samebox_1108（同じ箱の順列と乱数ラベル、seed の中央値、{len(seeds)} seed）", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "fig_pm_rl_samebox_1108.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
