#!/usr/bin/env python3
"""avgpool_cnn_0917 — tables, the registered calls (spec §5) and the prediction score (§6).

    python3 analysis/avgpool_cnn_0917/report.py              # after all 20 runs are complete

Refuses an incomplete box.  `--src` points the reader at another directory laid out like
results/avgpool_cnn_0917 (a smoke or a replay of the reference); `--allow-partial` is
accepted only together with such a `--src`.

Window = tasks 31-50 of online_acc.  Early = 1-10, late = 41-50, drop = early - late.
Pairs: seed-paired exact two-sided sign test (zeros dropped); a win = at most one seed
against and p < .05.  Seat = 2 * alpha * zbar (alpha_med for the adaptive arms, the fixed
alpha otherwise); the valley of phi' is at -pi/2.  Far-side fraction = share of a site's
channels/units whose 2 * alpha_med * m (hist channel mean) is below -pi/2.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "avgpool_cnn_0917"
REF = Path("/home/issan/Projects/claude/proj_004_drift/results/rlcifar_cnn_0908")
SNAC3 = Path("/home/issan/Projects/claude/wt/swish_battle_0917/results/swish_battle_0917/cnn/SNAc3")
NEW = ("SNA_avg", "SN3_avg")
REF_ARMS = ("SNA", "SN3", "SN06", "LR", "R")
HOST = {"SNA_avg": "SNA", "SN3_avg": "SN3", "SNA": "SNA", "SN3": "SN3"}
FIXED_ALPHA = {"SN3": 3.0, "SN3_avg": 3.0, "SN06": 0.6}
ADAPTIVE = {"SNA", "SNA_avg", "SNAc3"}
SITES = ("c1", "c2", "f1", "f2")
WIN, EARLY, LATE = (31, 50), (1, 10), (41, 50)
SEEDS = list(range(10))
N_TASKS = 50
VALLEY = -math.pi / 2
SNA_REF_DROP = 0.0284          # spec §5 D (the reference SNA's drop, summary_cnn_A of swish_battle_0917)
A_CUT = 0.20
PREDICTIONS = {                # spec §6, Claude
    "P_avg1": ("A == NEAR_SIDE", 0.55),
    "P_avg2": ("B == MLP_ORDER", 0.40),
    "P_avg3": ("D == DROP_HALVED", 0.50),
    "P_avg4": ("C_SN3 == AVG_WORSE", 0.45),
    "P_avg5": ("seed-mean memo_acc >= 0.99 at every task, both avg arms", 0.75),
    "P_avg6": ("SNA_avg window seat of f1 and f2 in (-pi/2, 0)", 0.70),
    "P_cond": ("if A == NEAR_SIDE then C_SNA == AVG_BETTER", 0.65),
}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def sign(diff):
    """Exact two-sided sign test, zeros dropped (the host verdicts' rule)."""
    d = np.asarray(diff, float)
    nz = d[d != 0]
    n = len(nz)
    pos = int((nz > 0).sum())
    k = min(pos, n - pos)
    p = min(sum(comb(n, i) for i in range(k + 1)) / 2 ** n * 2, 1.0) if n else float("nan")
    return pos, n, p


def load(src: Path, allow_partial: bool):
    parts, prov, missing, hists = [], {}, [], {}
    for arm in NEW:
        for s in SEEDS:
            d = src / arm / f"seed{s}"
            if not (d / "provenance.json").exists():
                missing.append(f"{arm}/seed{s}")
                continue
            x = pd.read_csv(d / "per_task.csv")
            x["arm"] = arm
            parts.append(x)
            prov[f"{arm}/seed{s}"] = json.loads((d / "provenance.json").read_text())
            hists[(arm, s)] = d / "hist" / f"{HOST[arm]}_seed{s}.npz"
    if missing and not allow_partial:
        raise SystemExit(f"{len(missing)} runs incomplete ({', '.join(missing[:6])} ...); "
                         "the report reads a complete box only")
    refs = {}
    for arm in REF_ARMS:
        f = REF / arm / "per_task.csv"
        x = pd.read_csv(f)
        x = x[x.seed.isin(SEEDS)].copy()
        x["arm"] = arm
        parts.append(x)
        refs[arm] = {"path": str(f), "sha256": sha(f)}
        if arm in HOST:
            for s in SEEDS:
                hists[(arm, s)] = REF / arm / "hist" / f"{arm}_seed{s}.npz"
    snac3 = sorted(SNAC3.glob("seed*/per_task.csv"))
    if len(snac3) == 10:
        x = pd.concat([pd.read_csv(f) for f in snac3], ignore_index=True)
        x["arm"] = "SNAc3"
        parts.append(x)
        refs["SNAc3"] = {"path": str(SNAC3), "sha256": {f.parent.name: sha(f) for f in snac3}}
    d = pd.concat(parts, ignore_index=True)
    return d, prov, missing, refs, hists


def done_rows(d: pd.DataFrame) -> pd.DataFrame:
    return d[d.online_acc.notna()] if "online_acc" in d else d


def per_seed(d, arm, col, lo, hi):
    g = d[(d.arm == arm) & (d.task >= lo) & (d.task <= hi) & d.online_acc.notna()]
    full = [s for s, gg in g.groupby("seed") if len(gg) == hi - lo + 1]
    return g[g.seed.isin(full)].groupby("seed")[col].mean()


def pair(d, a, b):
    x, y = per_seed(d, a, "online_acc", *WIN), per_seed(d, b, "online_acc", *WIN)
    s = sorted(set(x.index) & set(y.index))
    if not s:
        return None
    v = (x[s] - y[s]).values
    pos, n, p = sign(v)
    return {"a": a, "b": b, "mean": float(v.mean()),
            "se": float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else float("nan"),
            "wins": pos, "n": n, "n_seeds": len(s), "p": float(p),
            "sig_win": bool(n > 0 and n - pos <= 1 and p < 0.05),
            "sig_loss": bool(n > 0 and pos <= 1 and p < 0.05)}


def alpha_of(arm, row_or_frame, site):
    if arm in FIXED_ALPHA:
        return FIXED_ALPHA[arm]
    if arm in ADAPTIVE:
        return row_or_frame[f"alpha_med_{site}"]
    return None


def seat(d, arm, site, lo, hi):
    g = d[(d.arm == arm) & (d.task >= lo) & (d.task <= hi) & d.online_acc.notna()]
    if arm not in FIXED_ALPHA and arm not in ADAPTIVE:
        return float("nan")
    a = alpha_of(arm, g, site)
    return float((2 * a * g[f"zbar_{site}"]).mean())


def far_side(hists, d, arm, site, task):
    """Per-seed far-side fraction at `task`, from the hist channel means."""
    out = {}
    for s in SEEDS:
        p = hists.get((arm, s))
        if p is None or not p.exists():
            continue
        z = np.load(p)
        if z[f"m_{site}"].shape[0] < task:
            continue
        m = z[f"m_{site}"][task - 1].astype(np.float64)
        r = d[(d.arm == arm) & (d.seed == s) & (d.task == task)]
        if r.empty:
            continue
        a = FIXED_ALPHA.get(arm) or float(r.iloc[0][f"alpha_med_{site}"])
        out[s] = float((2 * a * m < VALLEY).mean())
    return out


def arm_row(d, hists, arm):
    g = done_rows(d[d.arm == arm])
    w = per_seed(d, arm, "online_acc", *WIN)
    e = per_seed(d, arm, "online_acc", *EARLY)
    l = per_seed(d, arm, "online_acc", *LATE)
    s = sorted(set(e.index) & set(l.index))
    memo_curve = g.groupby("task").memo_acc.mean()
    col = g.groupby("seed").online_acc.apply(lambda v: bool((v < 0.5).any()))
    r = {"arm": arm, "n": int(len(w)), "win": float(w.mean()), "win_sd": float(w.std(ddof=1)),
         "early": float(e.mean()), "drop": float((e[s] - l[s]).mean()),
         "memo_min_seedmean": float(memo_curve.min()), "collapsed": int(col.sum())}
    ww = g[(g.task >= WIN[0]) & (g.task <= WIN[1])]
    for site in SITES:
        r[f"mob_{site}"] = float(ww[f"mob_{site}"].mean())
        r[f"W_{site}"] = float(ww[f"zsd_{site}"].mean())
        r[f"seat_{site}"] = seat(d, arm, site, *WIN)
        if site in ("c1", "c2"):
            r[f"mobpool_{site}"] = float(ww[f"mob_pool_{site}"].mean())
        if arm in HOST:
            fs = far_side(hists, d, arm, site, N_TASKS)
            r[f"far50_{site}"] = float(np.mean(list(fs.values()))) if fs else float("nan")
    return r


def course(d, hists, arm, tasks=(1, 3, 5, 10, 20, 30, 40, 50)):
    g = done_rows(d[d.arm == arm]).groupby("task").mean(numeric_only=True)
    rows = []
    for t in tasks:
        if t not in g.index:
            continue
        r = g.loc[t]
        a = FIXED_ALPHA.get(arm, r.get("alpha_med_c2", float("nan")))
        fs = far_side(hists, d, arm, "c2", t)
        rows.append({"task": t, "online": float(r.online_acc), "seat_c2": float(2 * a * r.zbar_c2),
                     "far_c2": float(np.mean(list(fs.values()))) if fs else float("nan"),
                     "mob_c2": float(r.mob_c2), "mobpool_c2": float(r.mob_pool_c2),
                     "W_c2": float(r.zsd_c2)})
    return rows


def labels(d, hists, P, R):
    L = {}
    fa = far_side(hists, d, "SNA_avg", "c2", N_TASKS)
    L["A_far_c2_t50"] = float(np.mean(list(fa.values()))) if fa else float("nan")
    L["A_far_c2_t50_seeds"] = fa
    L["A"] = "NEAR_SIDE" if L["A_far_c2_t50"] < A_CUT else "CROSSES"
    fr = far_side(hists, d, "SNA", "c2", N_TASKS)
    L["A_reference_SNA"] = float(np.mean(list(fr.values()))) if fr else float("nan")
    b = P["SNA_avg-SN3_avg"]
    L["B"] = "MLP_ORDER" if b["sig_win"] else "CNN_ORDER" if b["sig_loss"] else "TIE"
    for key, (a_, b_) in (("C_SNA", ("SNA_avg", "SNA")), ("C_SN3", ("SN3_avg", "SN3"))):
        p = P[f"{a_}-{b_}"]
        L[key] = "AVG_BETTER" if p["sig_win"] else "AVG_WORSE" if p["sig_loss"] else "TIE"
    L["D_drop_SNA_avg"] = R["SNA_avg"]["drop"]
    L["D_drop_SNA_reference_recomputed"] = R["SNA"]["drop"]
    L["D"] = "DROP_HALVED" if R["SNA_avg"]["drop"] < SNA_REF_DROP / 2 else "DROP_KEPT"
    L["P5_memo_min"] = {a: R[a]["memo_min_seedmean"] for a in NEW}
    L["P6_seat_f"] = {s: R["SNA_avg"][f"seat_{s}"] for s in ("f1", "f2")}
    return L


def score(L):
    hit = {
        "P_avg1": L["A"] == "NEAR_SIDE",
        "P_avg2": L["B"] == "MLP_ORDER",
        "P_avg3": L["D"] == "DROP_HALVED",
        "P_avg4": L["C_SN3"] == "AVG_WORSE",
        "P_avg5": all(v >= 0.99 for v in L["P5_memo_min"].values()),
        "P_avg6": all(VALLEY < v < 0 for v in L["P6_seat_f"].values()),
        "P_cond": (L["C_SNA"] == "AVG_BETTER") if L["A"] == "NEAR_SIDE" else None,
    }
    return {k: {"claim": PREDICTIONS[k][0], "p": PREDICTIONS[k][1], "hit": v} for k, v in hit.items()}


def f(v, spec=".4f"):
    return "—" if v is None or (isinstance(v, float) and math.isnan(v)) else format(v, spec)


def write(src, d, prov, missing, refs, hists):
    arms = list(NEW) + [a for a in ("SNA", "SN3", "SNAc3", "SN06", "LR", "R") if a in set(d.arm)]
    R = {a: arm_row(d, hists, a) for a in arms}
    pairs = [("SNA_avg", "SN3_avg"), ("SNA_avg", "SNA"), ("SN3_avg", "SN3"), ("SN3", "SNA"),
             ("SNA_avg", "LR"), ("SN3_avg", "LR")]
    P = {f"{a}-{b}": pair(d, a, b) for a, b in pairs}
    L = labels(d, hists, P, R)
    S = score(L)
    lines = ["# avgpool_cnn_0917 — max-pool を avg-pool に替えた RL-CIFAR CNN", "",
             "spec: `specs/spec_avgpool_cnn_0917.md`。窓 = タスク 31–50 の online_acc（seed 平均 ± seed 間 SD）。"
             "早期 = t1–10、低下 = 早期 − t41–50。座席 = 2α·z̄（谷 −π/2 = −1.57、窓の平均）。"
             "越え t50 = t50 でチャネル平均の座席が谷より深いチャネルの割合（hist）。"
             "mob_pool は max-pool の勝ち位置のゲート（avg 腕では勾配に効くゲートではない）。", "",
             "| 腕 | n | 窓 | 早期 | 低下 | memo 最小 | 崩壊 | 座席 c1 | c2 | f1 | f2 | mob c1 | c2 | f1 | f2 | mob_pool c1 | c2 | 越え t50 c1 | c2 | f1 | f2 |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for a in arms:
        r = R[a]
        lines.append(" | ".join([f"| {a}", str(r["n"]), f"{f(r['win'])} ± {f(r['win_sd'])}", f(r["early"]),
                                 f(r["drop"], "+.4f"), f(r["memo_min_seedmean"]), f"{r['collapsed']}/{r['n']}"]
                                + [f(r[f"seat_{s}"], "+.2f") for s in SITES]
                                + [f(r[f"mob_{s}"], ".2f") for s in SITES]
                                + [f(r.get(f"mobpool_{s}"), ".2f") for s in ("c1", "c2")]
                                + [f(r.get(f"far50_{s}"), ".2f") for s in SITES]) + " |")
    lines += ["", "W（z の標準偏差、窓の平均）:", "", "| 腕 | c1 | c2 | f1 | f2 |", "|---|---|---|---|---|"]
    for a in arms:
        lines.append(f"| {a} | " + " | ".join(f(R[a][f'W_{s}'], '.3f') for s in SITES) + " |")
    lines += ["", "対（窓、seed 対応）:", "", "| a − b | 平均 ± SE | a の勝ち | p |", "|---|---|---|---|"]
    for k, p in P.items():
        if p:
            lines.append(f"| {p['a']} − {p['b']} | {p['mean']:+.4f} ± {p['se']:.4f} | {p['wins']}/{p['n']} | {p['p']:.4f} |")
    lines += ["", "c2 の時間経過（seed 平均）:", ""]
    for a in ("SNA_avg", "SNA", "SN3_avg", "SN3"):
        lines += [f"**{a}**", "", "| t | online | 座席 c2 | 越え c2 | mob c2 | mob_pool c2 | W c2 |",
                  "|---|---|---|---|---|---|---|"]
        for c in course(d, hists, a):
            lines.append(f"| {c['task']} | {c['online']:.3f} | {c['seat_c2']:+.2f} | {f(c['far_c2'], '.2f')} | "
                         f"{c['mob_c2']:.2f} | {c['mobpool_c2']:.2f} | {c['W_c2']:.3f} |")
        lines.append("")
    divs = {k: v["divergence"] for k, v in prov.items() if v.get("divergence", {}).get("diverged")}
    hashes = sorted({v["git_hash"] for v in prov.values()})
    dirty = sum(bool(v.get("git_dirty_code")) for v in prov.values())
    lines += [f"走の記録: {len(prov)} 本・commit {len(hashes)} 種（{', '.join(h[:7] for h in hashes)}）・"
              f"未 commit の変更つきの走 {dirty} 本・発散 {len(divs)} 本・"
              f"宿主 sha256 {sorted({v['host_sha256'][:12] for v in prov.values()})}。", "",
              "## 判定（spec §5）", "",
              f"- **A** = `{L['A']}`（SNA_avg の c2 越え t50 = {f(L['A_far_c2_t50'], '.3f')}、閾値 {A_CUT}。"
              f"参照 SNA の同じ量 = {f(L['A_reference_SNA'], '.3f')}）",
              f"- **B** = `{L['B']}`（SNA_avg − SN3_avg）",
              f"- **C_SNA** = `{L['C_SNA']}`",
              f"- **C_SN3** = `{L['C_SN3']}`",
              f"- **D** = `{L['D']}`（SNA_avg の低下 {f(L['D_drop_SNA_avg'], '+.4f')}、閾値 {SNA_REF_DROP / 2:.4f}。"
              f"参照 SNA の低下の再計算 = {f(L['D_drop_SNA_reference_recomputed'], '+.4f')}）", "",
              "## 予測の採点（spec §6、Claude）", "", "| 予測 | 内容 | 確率 | 結果 |", "|---|---|---|---|"]
    for k, v in S.items():
        res = "—（条件不成立）" if v["hit"] is None else ("的中" if v["hit"] else "外れ")
        lines.append(f"| {k} | {v['claim']} | {v['p']:.2f} | {res} |")
    if missing:
        lines += ["", f"**未完了 {len(missing)} 本（部分集計）**: {', '.join(missing)}"]
    tag = "" if src == OUT else "_" + src.name
    (OUT / f"summary{tag}.md").write_text("\n".join(lines) + "\n")
    verdict = {"labels": L, "score": S, "pairs": P, "arms": R, "refs": refs,
               "runs": {"n": len(prov), "git_hashes": hashes, "dirty": dirty, "divergences": divs},
               "missing": missing, "src": str(src)}
    (OUT / f"verdict{tag}.json").write_text(json.dumps(verdict, indent=1, default=str))
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(OUT))
    ap.add_argument("--allow-partial", action="store_true")
    a = ap.parse_args()
    src = Path(a.src).resolve()
    if a.allow_partial and src == OUT.resolve():
        raise SystemExit("--allow-partial is for a --src other than the main results")
    OUT.mkdir(parents=True, exist_ok=True)
    write(src, *load(src, a.allow_partial))


if __name__ == "__main__":
    main()
