#!/usr/bin/env python3
"""kappa_cnn5p1_1022: tables (sec. 4) and the scoring of P1-P7 (sec. 5) from results/kappa_cnn5p1_1022/rows.csv.

    .venv/bin/python analysis/kappa_cnn5p1_1022/summarize.py      -> tables.md, verdict.json
"""
import csv, json, math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "kappa_cnn5p1_1022"
ARMS = ("R", "ELU", "KKT1", "SNAfrz1")
LIVE = ("ELU", "KKT1", "SNAfrz1")
BUDGETS = (780, 2340, 7800)
OWN_B = (1, 9, 19, 29)
K, HID = 5, 100


def fl(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def tf(x):
    return str(x).strip().lower() == "true"


def load():
    rows = []
    with open(OUT / "rows.csv") as fh:
        for r in csv.DictReader(fh):
            q = dict(r)
            for k in ("budget", "boundary", "task", "seed", "slot", "N", "K", "n_alive_f2", "kappa_i_n_undef", "perm_n_gt", "perm_n_eq"):
                q[k] = int(float(r[k]))
            for k in ("kappa", "cos2", "Q", "F_norm", "normV", "kappa_lb5", "kappa_over_lb5", "J", "D5", "D100", "L5", "L100",
                      "acc100", "acc5", "perm_q", "kappa100", "cos2_100", "kappa_i_negfrac", "state_train_acc", "A"):
                q[k] = fl(r[k])
            for k in ("defined", "lemma_violation", "D5_positive", "perm_max", "collapsed_state"):
                q[k] = tf(r[k])
            rows.append(q)
    return rows


def med(x):
    x = [v for v in x if v is not None and np.isfinite(v)]
    return float(np.median(x)) if x else float("nan")


def mean_sd(x):
    x = [v for v in x if v is not None and np.isfinite(v)]
    return (float(np.mean(x)), float(np.std(x, ddof=1)) if len(x) > 1 else float("nan")) if x else (float("nan"), float("nan"))


def f(x, d=3):
    return "—" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


def g(x):
    return "—" if x is None or not np.isfinite(x) else f"{x:.3g}"


def sel(rows, **kw):
    out = rows
    for k, v in kw.items():
        out = [r for r in out if r[k] == v]
    return out


def usable(r):
    return r["defined"] and not r["collapsed_state"]


def main():
    rows = load()
    own = {(r["budget"], r["arm"], r["seed"], r["boundary"]): r for r in rows if r["role"] == "own"}
    init = {(r["budget"], r["arm"], r["seed"]): r for r in rows if r["role"] == "init"}
    lines = ["# kappa_cnn5p1_1022 — 表（spec §4）", "",
             "集約: 10 seed。「数」は 10 seed のうちの個数（分母は定義されかつ崩壊でない seed、括弧に n）。比は seed ごとの比の seed 間中央値。cos² は seed 平均（SD）。κ > 0・置換最大・補題の破れは個数。崩壊 = その境界の直近の hard 課題の train 精度 < 0.3（ReLU）。", ""]
    verdict = {}

    # ---------------- T1 own-end
    lines += ["## T1 自分の課題末（境界 1・9・19・29）", "",
              "| 腕 | 予算 | 境界 | n | κ>0 | κ_t/κ₁ 中央値 | ≥0.8 | Q_t/Q₁ 中央値 | ≥0.8 | cos² 平均 (SD) | ‖F‖ 中央値 | κ 中央値 | κ/κ_lb 中央値 | D>0 | 破れ | 置換最大 | 崩壊/未定義 | cos²₁₀₀ 平均 | κ_i<0 割合 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    T1 = {}
    for arm in ARMS:
        for bud in BUDGETS:
            for b in OWN_B:
                rs = [own[(bud, arm, s, b)] for s in range(2000, 2010) if (bud, arm, s, b) in own]
                us = [r for r in rs if usable(r)]
                kr, qr = [], []
                for r in us:
                    r1 = own.get((bud, arm, r["seed"], 1))
                    if r1 is not None and usable(r1) and r1["kappa"] > 0:
                        kr.append(r["kappa"] / r1["kappa"]); qr.append(r["Q"] / r1["Q"])
                c2 = mean_sd([r["cos2"] for r in us])
                rec = dict(n=len(us), n_collapsed=sum(r["collapsed_state"] for r in rs), n_undef=sum(not r["defined"] for r in rs),
                           kappa_pos=sum(r["kappa"] > 0 for r in us), kappa_ratio_med=med(kr), kappa_ratio_ge08=sum(v >= 0.8 for v in kr),
                           n_ratio=len(kr), Q_ratio_med=med(qr), Q_ratio_ge08=sum(v >= 0.8 for v in qr),
                           cos2_mean=c2[0], cos2_sd=c2[1], F_norm_med=med([r["F_norm"] for r in us]), kappa_med=med([r["kappa"] for r in us]),
                           kappa_over_lb_med=med([r["kappa_over_lb5"] for r in us]), D_pos=sum(r["D5_positive"] for r in us),
                           lemma_viol=sum(r["lemma_violation"] for r in rs), perm_max=sum(r["perm_max"] for r in us),
                           cos2_100_mean=mean_sd([r["cos2_100"] for r in us])[0], kappa_i_negfrac_med=med([r["kappa_i_negfrac"] for r in us]),
                           collapsed_values=[dict(seed=r["seed"], kappa=r["kappa"], cos2=r["cos2"], acc=r["acc100"], defined=r["defined"]) for r in rs if not usable(r)])
                T1[(arm, bud, b)] = rec
                lines.append(f"| {arm} | {bud} | {b} | {rec['n']} | {rec['kappa_pos']} | {f(rec['kappa_ratio_med'],2) if b > 1 else '1'} | "
                             f"{rec['kappa_ratio_ge08'] if b > 1 else '—'}/{rec['n_ratio'] if b > 1 else '—'} | {f(rec['Q_ratio_med'],2) if b > 1 else '1'} | "
                             f"{rec['Q_ratio_ge08'] if b > 1 else '—'}/{rec['n_ratio'] if b > 1 else '—'} | {f(rec['cos2_mean'])} ({f(rec['cos2_sd'])}) | "
                             f"{g(rec['F_norm_med'])} | {g(rec['kappa_med'])} | {f(rec['kappa_over_lb_med'],2)} | {rec['D_pos']} | {rec['lemma_viol']} | {rec['perm_max']} | "
                             f"{rec['n_collapsed']}/{rec['n_undef']} | {f(rec['cos2_100_mean'])} | {f(rec['kappa_i_negfrac_med'],2)} |")
    lines.append("")
    coll = [(k, v["collapsed_values"]) for k, v in T1.items() if v["collapsed_values"]]
    if coll:
        lines += ["崩壊・未定義の状態の値（集計から外したもの）:", ""]
        for (arm, bud, b), vals in coll:
            lines.append(f"- {arm} {bud} 境界 {b}: " + "; ".join(f"seed {v['seed']} acc {f(v['acc'],2)} κ {g(v['kappa'])} cos² {f(v['cos2'])}{'' if v['defined'] else ' (未定義)'}" for v in vals))
        lines.append("")

    # ---------------- T2 pre vs post
    lines += ["## T2 課題の前と後（同じ課題の画像: 境界 b−1 の状態と境界 b の状態。課題 1 は初期値 → 境界 1）", "",
              "| 腕 | 予算 | 課題 | n | κ_pre 中央値 | κ_pre>0 | cos²_pre 平均 | \\|Q_pre\\|/Q_post 中央値 | κ_post 中央値 | cos²_post 平均 | 置換最大 pre | 置換最大 post |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    T2 = {}
    pre = {(r["budget"], r["arm"], r["seed"], r["task"]): r for r in rows if r["role"] in ("pre", "init")}
    for arm in ARMS:
        for bud in BUDGETS:
            for t in (1, 9, 19, 29):
                pairs = []
                for s in range(2000, 2010):
                    rp, ro = pre.get((bud, arm, s, t)), own.get((bud, arm, s, t))
                    if rp and ro and usable(ro) and rp["defined"] and not rp["collapsed_state"]:
                        pairs.append((rp, ro))
                rec = dict(n=len(pairs), kappa_pre_med=med([p["kappa"] for p, o in pairs]), kappa_pre_pos=sum(p["kappa"] > 0 for p, o in pairs),
                           cos2_pre_mean=mean_sd([p["cos2"] for p, o in pairs])[0],
                           Qratio_med=med([abs(p["Q"]) / o["Q"] for p, o in pairs if o["Q"] > 0]),
                           kappa_post_med=med([o["kappa"] for p, o in pairs]), cos2_post_mean=mean_sd([o["cos2"] for p, o in pairs])[0],
                           perm_max_pre=sum(p["perm_max"] for p, o in pairs), perm_max_post=sum(o["perm_max"] for p, o in pairs),
                           neg_frac=(sum(p["kappa"] < 0 for p, o in pairs) / len(pairs)) if pairs else float("nan"))
                T2[(arm, bud, t)] = rec
                lines.append(f"| {arm} | {bud} | {t} | {rec['n']} | {g(rec['kappa_pre_med'])} | {rec['kappa_pre_pos']} | {f(rec['cos2_pre_mean'])} | "
                             f"{f(rec['Qratio_med'])} | {g(rec['kappa_post_med'])} | {f(rec['cos2_post_mean'])} | {rec['perm_max_pre']} | {rec['perm_max_post']} |")
    lines.append("")

    # ---------------- T3 retention
    lines += ["## T3 前の hard 課題への整列が残るか（課題 s の画像とクラス行を、後の境界 b の状態で）", "",
              "| 腕 | 予算 | s → b | n | κ_s(b)/κ_s(s) 中央値 | >0 | Q 比 中央値 | cos²_s(b) 平均 | cos²_s(s) 平均 | 置換最大 |",
              "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    T3 = {}
    prev = {(r["budget"], r["arm"], r["seed"], r["boundary"], r["task"]): r for r in rows if r["role"] == "prev"}
    SPANS = [(1, 8), (1, 9), (9, 18), (9, 19), (19, 28), (19, 29), (1, 18), (1, 19), (9, 28), (9, 29), (1, 28), (1, 29), (1, 30)]
    for arm in ARMS:
        for bud in BUDGETS:
            for s_, b in SPANS:
                pairs = []
                for s in range(2000, 2010):
                    rp, ro = prev.get((bud, arm, s, b, s_)), own.get((bud, arm, s, s_))
                    if rp and ro and usable(ro) and rp["defined"] and not rp["collapsed_state"] and ro["kappa"] > 0:
                        pairs.append((rp, ro))
                rec = dict(n=len(pairs), ratio_med=med([p["kappa"] / o["kappa"] for p, o in pairs]), pos=sum(p["kappa"] > 0 for p, o in pairs),
                           Qratio_med=med([p["Q"] / o["Q"] for p, o in pairs]), cos2_b_mean=mean_sd([p["cos2"] for p, o in pairs])[0],
                           cos2_s_mean=mean_sd([o["cos2"] for p, o in pairs])[0], perm_max=sum(p["perm_max"] for p, o in pairs),
                           ratios=[p["kappa"] / o["kappa"] for p, o in pairs], cos2_lower=sum(p["cos2"] < o["cos2"] for p, o in pairs))
                T3[(arm, bud, s_, b)] = rec
                lines.append(f"| {arm} | {bud} | {s_} → {b} | {rec['n']} | {f(rec['ratio_med'],2)} | {rec['pos']} | {f(rec['Qratio_med'],2)} | "
                             f"{f(rec['cos2_b_mean'])} | {f(rec['cos2_s_mean'])} | {rec['perm_max']} |")
    lines.append("")

    # ---------------- T4 last hard task (not saved at its own end)
    lines += ["## T4 直前の hard 課題（自分の末は保存されていない: 課題 7・17・27 を境界 8/9・18/19・28/29 で、課題 29 を境界 30 で）", "",
              "| 腕 | 予算 | (境界, 課題) | n | κ 中央値 | κ>0 | cos² 平均 | 置換最大 |", "|---|---:|---|---:|---:|---:|---:|---:|"]
    last = {(r["budget"], r["arm"], r["seed"], r["boundary"], r["task"]): r for r in rows if r["role"] == "last"}
    T4 = {}
    for arm in ARMS:
        for bud in BUDGETS:
            for b, t in ((8, 7), (9, 7), (18, 17), (19, 17), (28, 27), (29, 27), (30, 29)):
                rs = [last[(bud, arm, s, b, t)] for s in range(2000, 2010) if (bud, arm, s, b, t) in last]
                us = [r for r in rs if r["defined"] and not r["collapsed_state"]]
                rec = dict(n=len(us), kappa_med=med([r["kappa"] for r in us]), pos=sum(r["kappa"] > 0 for r in us),
                           cos2_mean=mean_sd([r["cos2"] for r in us])[0], perm_max=sum(r["perm_max"] for r in us))
                T4[(arm, bud, b, t)] = rec
                lines.append(f"| {arm} | {bud} | ({b}, {t}) | {rec['n']} | {g(rec['kappa_med'])} | {rec['pos']} | {f(rec['cos2_mean'])} | {rec['perm_max']} |")
    lines.append("")

    # ---------------- scoring P1-P7 (spec sec. 5)
    own_us = [r for r in rows if r["role"] == "own" and usable(r)]
    own_all = [r for r in rows if r["role"] == "own"]
    p1 = {arm: dict(n=sum(r["arm"] == arm for r in own_us), pos=sum(r["arm"] == arm and r["kappa"] > 0 for r in own_us)) for arm in ARMS}
    verdict["P1"] = dict(per_arm=p1, n_collapsed=sum(r["collapsed_state"] for r in own_all), n_undefined=sum(not r["defined"] for r in own_all),
                         ok=all(v["pos"] == v["n"] for v in p1.values()) and all(p1[a]["n"] == 120 for a in LIVE))
    lb = {}
    for (arm, bud, b), rec in T1.items():
        if arm in LIVE or rec["n"] > 0:
            lb[f"{arm}|{bud}|{b}"] = rec["kappa_over_lb_med"]
    lb_ok_780 = all(1.2 <= v <= 4 for k, v in lb.items() if "|780|" in k and np.isfinite(v))
    lb_ok_long = all(v >= 2 for k, v in lb.items() if "|780|" not in k and np.isfinite(v))
    verdict["P2"] = dict(lemma_violations=sum(r["lemma_violation"] for r in rows), D5_pos_all=all(r["D5_positive"] for r in own_us),
                         n_D5_nonpos_usable=sum(not r["D5_positive"] for r in own_us), kappa_over_lb_med=lb,
                         lb_band_780_ok=lb_ok_780, lb_band_long_ok=lb_ok_long,
                         ok=(sum(r["lemma_violation"] for r in rows) == 0) and all(r["D5_positive"] for r in own_us) and lb_ok_780 and lb_ok_long)
    cells_k = {f"{arm}|{bud}": T1[(arm, bud, 29)]["kappa_ratio_ge08"] for arm in ARMS for bud in BUDGETS}
    cells_q = {f"{arm}|{bud}": T1[(arm, bud, 29)]["Q_ratio_ge08"] for arm in ARMS for bud in BUDGETS}
    nk = sum(v >= 8 for v in cells_k.values()); nq = sum(v >= 8 for v in cells_q.values())
    verdict["P3"] = dict(kappa_ratio_ge08_count_at29=cells_k, Q_ratio_ge08_count_at29=cells_q, cells_kappa_pass=nk, cells_Q_pass=nq,
                         ok=(nk <= 2) and (nq >= 9))
    c2 = {f"{arm}|{bud}|{b}": T1[(arm, bud, b)]["cos2_mean"] for arm in LIVE for bud in BUDGETS for b in OWN_B}
    verdict["P4"] = dict(cos2_mean=c2, ok=all(v >= 0.4 for v in c2.values()) and all(T1[(arm, bud, 29)]["cos2_mean"] >= 0.2 for arm in LIVE for bud in BUDGETS))
    pm = sum(r["perm_max"] for r in own_us)
    verdict["P5"] = dict(n=len(own_us), perm_max=pm, frac=pm / len(own_us) if own_us else float("nan"), ok=(pm / len(own_us) >= 0.95) if own_us else False)
    qr = {f"{arm}|{bud}|{t}": T2[(arm, bud, t)]["Qratio_med"] for arm in ARMS for bud in BUDGETS for t in (9, 19, 29)}
    negcells = {}
    for arm in ARMS:
        for bud in BUDGETS:
            fr = [T2[(arm, bud, t)]["neg_frac"] for t in (9, 19, 29)]
            ns = [T2[(arm, bud, t)]["n"] for t in (9, 19, 29)]
            pooled = sum(a * n for a, n in zip(fr, ns) if np.isfinite(a)) / max(sum(n for a, n in zip(fr, ns) if np.isfinite(a)), 1)
            negcells[f"{arm}|{bud}"] = pooled
    verdict["P6"] = dict(Qratio_med=qr, size_ok=all(v <= 0.1 for v in qr.values() if np.isfinite(v)), neg_frac_pooled=negcells,
                         cells_neg_majority=sum(v >= 0.5 for v in negcells.values()), sign_ok=sum(v >= 0.5 for v in negcells.values()) >= 6)
    verdict["P6"]["ok"] = verdict["P6"]["size_ok"] and verdict["P6"]["sign_ok"]
    rat, posn, nn, lower = [], 0, 0, 0
    for (arm, bud, s_, b), rec in T3.items():
        if (s_, b) in ((1, 9), (9, 19), (19, 29)):
            rat += rec["ratios"]; posn += rec["pos"]; nn += rec["n"]; lower += rec["cos2_lower"]
    verdict["P7"] = dict(n=nn, pos=posn, frac_pos=posn / nn if nn else float("nan"), ratio_med=med(rat), cos2_lower_frac=lower / nn if nn else float("nan"),
                         ok=(nn > 0) and (posn / nn >= 0.8) and (0.2 <= med(rat) <= 0.8) and (lower / nn > 0.5))
    lines += ["## 採点（spec §5）", "", "| 予測 | 判定 | 要点 |", "|---|---|---|"]
    lines.append(f"| P1 κ>0 | {'支持' if verdict['P1']['ok'] else '不支持'} | " + ", ".join(f"{a} {v['pos']}/{v['n']}" for a, v in p1.items()) + f"; 崩壊 {verdict['P1']['n_collapsed']}・未定義 {verdict['P1']['n_undefined']} |")
    lines.append(f"| P2 補題 | {'支持' if verdict['P2']['ok'] else '不支持'} | 破れ {verdict['P2']['lemma_violations']}、D≤0 の使用状態 {verdict['P2']['n_D5_nonpos_usable']}、κ/κ_lb の帯 780 {'○' if lb_ok_780 else '×'}・2340/7800 {'○' if lb_ok_long else '×'} |")
    lines.append(f"| P3 κ 比と Q 比 | {'支持' if verdict['P3']['ok'] else '不支持'} | 境界 29 で κ_29/κ₁≥0.8 が 8/10 以上のマス {nk}/12（予測 ≤2）、Q_29/Q₁≥0.8 が 8/10 以上のマス {nq}/12（予測 ≥9） |")
    lines.append(f"| P4 cos² | {'支持' if verdict['P4']['ok'] else '不支持'} | 生きた 3 腕の cos² 平均の最小 {f(min(c2.values()))}（予測 ≥0.4） |")
    lines.append(f"| P5 置換の最大 | {'支持' if verdict['P5']['ok'] else '不支持'} | {pm}/{len(own_us)} = {f(verdict['P5']['frac'])}（予測 ≥0.95） |")
    lines.append(f"| P6 課題の前 | 大きさ {'支持' if verdict['P6']['size_ok'] else '不支持'}・符号 {'支持' if verdict['P6']['sign_ok'] else '不支持'} | \\|Q_pre\\|/Q_post の中央値の最大 {f(max(v for v in qr.values() if np.isfinite(v)))}（予測 ≤0.1）、負が半数以上のマス {verdict['P6']['cells_neg_majority']}/12（予測 ≥6） |")
    lines.append(f"| P7 残存 | {'支持' if verdict['P7']['ok'] else '不支持'} | 正 {posn}/{nn} = {f(verdict['P7']['frac_pos'])}（予測 ≥0.8）、比の中央値 {f(verdict['P7']['ratio_med'],2)}（予測 0.2〜0.8）、cos² が自分の末より低い {f(verdict['P7']['cos2_lower_frac'])} |")
    lines.append("")
    (OUT / "tables.md").write_text("\n".join(lines))
    (OUT / "verdict.json").write_text(json.dumps(verdict, indent=1, default=float))
    print("\n".join(lines[-10:]))


if __name__ == "__main__":
    main()
