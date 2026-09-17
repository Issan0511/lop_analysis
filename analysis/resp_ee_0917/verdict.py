#!/usr/bin/env python3
"""Verdict for resp_ee_0917 (specs/spec_resp_ee_0917.md section 5).

    python3 analysis/resp_ee_0917/verdict.py                      # the registered run, all 10 seeds
    python3 analysis/resp_ee_0917/verdict.py --src results/_smoke_resp_ee_0917 --out /tmp/x --partial

Reads runs/s<seed>/{arms.csv, prefix.csv, dstar.json, provenance.json} for the listed seeds only (no
glob), builds every quantity per seed and only then summarises across seeds.  Refuses to read the
registered run directory before all seeds have a provenance.json unless --partial is given (and then
says so in the summary).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RUN_ID = "resp_ee_0917"
MAIN_SRC = REPO / "results" / RUN_ID
SEEDS = tuple(range(10))
MIN_SEEDS = 8                                    # spec 5.2 (A)
MAIN_LEVEL = 0.975                               # two primary contrasts, Bonferroni
COMP_LEVEL = 0.95
CHANCE = 0.1                                     # 10 uniform labels
N_IMG = 1200
FLOOR_Z = 3.0                                    # spec 4.3: major_frac + 3 binomial sd over the 1200 labels
LADDER = (5, 10, 15, 20, 30)
PREFIX_T = 22
EPOCHS = 80
BRANCH_T = (2, 5, 7, 10, 15, 20)
ARM_NAMES = (["S2_5r", "S2_7r", "S2_10r", "S2_15r", "S2_20r", "S2_10"]
             + [f"S2u{d}r" for d in LADDER] + ["S1u20r"]
             + ["R2_5", "R2_7", "R2_10", "R2_15", "R2_20", "R2_10r", "N2r", "N10r"]
             + [f"N{t}" for t in BRANCH_T])
N_ARMS = [f"N{t}" for t in BRANCH_T]
P1 = ("R2_10", "N10")                            # d = E(first) - E(second), k = 1
P2 = ("N2r", "S2_10r")
NATURAL_GAP = ("N2", "N10")                      # spec 5.2 (B)
SECONDARY = (
    [(f"R2_{t}", f"N{t}", f"restore at t{t}") for t in (5, 7, 15, 20)]
    + [("N2r", f"S2_{s}r", f"sink to the t{s} field") for s in (5, 7, 15, 20)]
    + [("R2_10r", "N10r", "restore at t10, both fresh Adam"),
       ("N10r", "N10", "fresh Adam alone at t10"),
       ("N2r", "N2", "fresh Adam alone at t2"),
       ("S2_10", "S2_10r", "sink to t10 with the t2 moments kept"),
       ("S1u20r", "S2u20r", "layer 1 vs layer 2, both shifted by 20")]
    + [("N2r", f"S2u{d}r", f"uniform shift {d}") for d in LADDER])
PRED = {"P1_mean": 0.35, "P2_mean": 0.30, "label": "RESPONSE_BOTH_WAYS", "label_prob": 0.75,
        "ladder_intact": 10, "ladder_intact_F": 0.9, "ladder_floor": 20, "ladder_floor_share": 0.8}


# --------------------------------------------------------------------------
# t distribution (the incomplete beta by continued fraction; checks S6 pins it to the published table)
# --------------------------------------------------------------------------

def _betacf(a: float, b: float, x: float) -> float:
    tiny, eps = 1e-300, 1e-15
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 1000):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            return h
    raise RuntimeError("betacf did not converge")


def betainc(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lbt) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbt) * _betacf(b, a, 1.0 - x) / b


def t_cdf(t: float, df: int) -> float:
    tail = 0.5 * betainc(df / 2.0, 0.5, df / (df + t * t))
    return 1.0 - tail if t > 0 else tail


def t_quantile(p: float, df: int) -> float:
    lo, hi = 0.0, 1e3
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-12:
            break
    return 0.5 * (lo + hi)


def paired(x, level: float) -> dict:
    x = np.asarray(x, dtype=np.float64)
    n = int(x.size)
    out = {"n": n, "mean": float(x.mean()) if n else float("nan"),
           "sd": float(x.std(ddof=1)) if n > 1 else float("nan"), "level": level, "degenerate": False,
           "n_pos": int((x > 0).sum()), "n_neg": int((x < 0).sum())}
    if n < 2 or not np.isfinite(out["sd"]):
        out |= {"lo": float("nan"), "hi": float("nan"), "sign": "0"}
        return out
    if out["sd"] == 0.0:
        out |= {"lo": out["mean"], "hi": out["mean"], "degenerate": True}
    else:
        half = t_quantile(0.5 + level / 2.0, n - 1) * out["sd"] / math.sqrt(n)
        out |= {"lo": out["mean"] - half, "hi": out["mean"] + half}
    out["sign"] = "+" if out["lo"] > 0 else "-" if out["hi"] < 0 else "0"
    return out


def spearman(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    ra = pd.Series(a[ok]).rank().to_numpy()
    rb = pd.Series(b[ok]).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


# --------------------------------------------------------------------------
# per seed
# --------------------------------------------------------------------------

def floor_thr(mf: float) -> float:
    """spec 4.3: a net that learns nothing about this task's labels scores at most the majority share
    in expectation; its excess is the binomial noise of 1200 labels at that share."""
    return mf + FLOOR_Z * math.sqrt(mf * (1.0 - mf) / N_IMG)


def load(src: Path, seeds=SEEDS) -> tuple[dict, list]:
    shards, missing = {}, []
    for s in seeds:
        d = src / "runs" / f"s{s}"
        need = [d / n for n in ("arms.csv", "prefix.csv", "dstar.json", "provenance.json")]
        if not all(p.exists() for p in need):
            missing.append(s)
            continue
        shards[s] = {"arms": pd.read_csv(d / "arms.csv", float_precision="round_trip"),
                     "prefix": pd.read_csv(d / "prefix.csv", float_precision="round_trip"),
                     "dstar": json.loads((d / "dstar.json").read_text()),
                     "prov": json.loads((d / "provenance.json").read_text())}
    return shards, missing


def _truth(v) -> bool:
    if isinstance(v, str):
        return v.strip().lower() == "true"
    return bool(v) and not (isinstance(v, float) and math.isnan(v))


def seed_validity(sh: dict, prereg: str | None, epochs: int) -> str | None:
    a, p, prov = sh["arms"], sh["prefix"], sh["prov"]
    if prereg is not None and prov.get("prereg_commit") != prereg:
        return f"prereg_commit {prov.get('prereg_commit')} != {prereg}"
    if not prov.get("flush_denormal"):
        return "flush_denormal off"
    if int(prov.get("epochs_per_task", -1)) != epochs:
        return f"epochs {prov.get('epochs_per_task')}"
    if len(p) != PREFIX_T or not np.isfinite(p["online_acc"]).all():
        return "prefix incomplete or non-finite"
    for arm in ARM_NAMES:
        r = a[a["arm"] == arm]
        if sorted(r["k"].tolist()) != [1, 2]:
            return f"{arm}: continuation tasks {sorted(r['k'].tolist())}"
        if not (np.isfinite(r["online_acc"]).all() and np.isfinite(r["online_ce"]).all()):
            return f"{arm}: non-finite"
        r1 = r[r["k"] == 1].iloc[0]
        if not (_truth(r1["logits_equal_full"]) and _truth(r1["logits_equal_mb"])):
            return f"{arm}: branch-point logits differ from the natural net"
        if arm in N_ARMS and not all(_truth(v) for v in r["hash_match_prefix"]):
            return f"{arm}: natural continuation does not reproduce the prefix"
    return None


def E(a: pd.DataFrame, arm: str, k: int = 1, col: str = "online_acc") -> float:
    r = a[(a["arm"] == arm) & (a["k"] == k)]
    return float(r[col].iloc[0])


def at_floor(a: pd.DataFrame, arm: str, k: int = 1) -> bool:
    r = a[(a["arm"] == arm) & (a["k"] == k)].iloc[0]
    return float(r["online_acc"]) <= floor_thr(float(r["major_frac"]))


# --------------------------------------------------------------------------

def analyze(shards: dict, prereg: str | None = None, epochs: int = EPOCHS) -> dict:
    invalid = {s: why for s in sorted(shards) if (why := seed_validity(shards[s], prereg, epochs))}
    valid = [s for s in sorted(shards) if s not in invalid]
    res = {"valid_seeds": valid, "invalid_seeds": invalid, "n_valid": len(valid)}

    def diffs(x: str, y: str, k: int = 1) -> np.ndarray:
        return np.array([E(shards[s]["arms"], x, k) - E(shards[s]["arms"], y, k) for s in valid])

    if len(valid) < MIN_SEEDS:
        res["label"] = "INAPPLICABLE"
        res["why"] = f"{len(valid)} valid seeds < {MIN_SEEDS}"
        return res

    # (B) the natural loss of learning exists, and the fields differ
    gap = paired(diffs(*NATURAL_GAP), COMP_LEVEL)
    field_drop = {s: (float(shards[s]["prefix"].set_index("task").loc[2, "g2_end"]),
                      float(shards[s]["prefix"].set_index("task").loc[10, "g2_end"])) for s in valid}
    b_gap = gap["lo"] > 0
    b_field = all(g2 > g10 for g2, g10 in field_drop.values())
    res["B"] = {"natural_gap": gap, "field_t2_t10": field_drop, "gap_ok": b_gap, "field_ok": b_field}

    p1 = paired(diffs(*P1), MAIN_LEVEL)
    p2 = paired(diffs(*P2), MAIN_LEVEL)
    res["P1"], res["P2"] = p1, p2
    res["P1_95"], res["P2_95"] = paired(diffs(*P1), COMP_LEVEL), paired(diffs(*P2), COMP_LEVEL)
    if not (b_gap and b_field):
        res["label"] = "NOT_REPRODUCED"
    elif "-" in (p1["sign"], p2["sign"]):
        res["label"] = "RESPONSE_REVERSED"
    elif p1["sign"] == "+" and p2["sign"] == "+":
        res["label"] = "RESPONSE_BOTH_WAYS"
    elif p1["sign"] == "+":
        res["label"] = "RESTORE_ONLY"
    elif p2["sign"] == "+":
        res["label"] = "SINK_ONLY"
    else:
        res["label"] = "RESPONSE_NOT_SHOWN"

    # recovery fractions (report only)
    g_nat = diffs("N2", "N10")
    g_fresh = diffs("N2r", "N10r")
    res["recovery"] = {"rho1": float(np.mean(diffs(*P1)) / np.mean(g_nat)) if np.mean(g_nat) else float("nan"),
                       "rho2": float(np.mean(diffs(*P2)) / np.mean(g_fresh)) if np.mean(g_fresh) else float("nan"),
                       "gap_natural": float(np.mean(g_nat)), "gap_fresh": float(np.mean(g_fresh))}

    # secondary contrasts (95%, report only), both continuation tasks
    sec = []
    for x, y, what in [(P1[0], P1[1], "P1"), (P2[0], P2[1], "P2")] + SECONDARY:
        for k in (1, 2):
            sec.append({"contrast": f"{x} - {y}", "what": what, "k": k, **paired(diffs(x, y, k), COMP_LEVEL)})
    res["secondary"] = sec

    # the uniform ladder against its eps arithmetic (registered descriptive label, spec 5.4)
    ds = {q: float(np.median([shards[s]["dstar"]["dstar_q"][q] for s in valid])) for q in ("0.1", "0.5", "0.9")}
    Fbar = {}
    for d in LADDER:
        f = [(E(shards[s]["arms"], f"S2u{d}r") - CHANCE) / (E(shards[s]["arms"], "N2r") - CHANCE) for s in valid]
        Fbar[d] = float(np.mean(f))
    below = [d for d in LADDER if d < ds["0.1"]]
    above = [d for d in LADDER if d > ds["0.9"]]
    part_b = "NO_RUNG" if not below else ("HELD" if all(Fbar[d] >= 0.5 for d in below) else "DROPPED")
    part_a = "NO_RUNG" if not above else ("FLOORED" if all(Fbar[d] < 0.5 for d in above) else "SURVIVED")
    if part_b == "HELD" and part_a == "FLOORED":
        lad = "EPS_CONSISTENT"
    elif part_b == "DROPPED" and part_a == "SURVIVED":
        lad = "NONMONOTONE"
    elif part_b == "DROPPED":
        lad = "DROPS_BEFORE_EPS"
    elif part_a == "SURVIVED":
        lad = "SURVIVES_PAST_EPS"
    else:
        lad = f"PARTIAL({part_b},{part_a})"
    floors = {d: sum(at_floor(shards[s]["arms"], f"S2u{d}r") for s in valid) for d in LADDER}
    res["ladder"] = {"dstar_median": ds, "F": Fbar, "below": below, "above": above,
                     "part_below": part_b, "part_above": part_a, "label": lad, "at_floor": floors}

    # prediction scoring
    need_floor = math.ceil(PRED["ladder_floor_share"] * len(valid))
    p1c, p2c = res["P1_95"], res["P2_95"]
    res["predictions"] = {
        "P1_mean": {"pred": PRED["P1_mean"], "obs": p1["mean"],
                    "hit": bool(p1c["lo"] <= PRED["P1_mean"] <= p1c["hi"])},
        "P2_mean": {"pred": PRED["P2_mean"], "obs": p2["mean"],
                    "hit": bool(p2c["lo"] <= PRED["P2_mean"] <= p2c["hi"])},
        "label": {"pred": PRED["label"], "prob": PRED["label_prob"], "obs": res["label"],
                  "hit": res["label"] == PRED["label"]},
        "ladder_intact": {"pred": f"F(S2u{PRED['ladder_intact']}r) >= {PRED['ladder_intact_F']}",
                          "obs": Fbar[PRED["ladder_intact"]],
                          "hit": Fbar[PRED["ladder_intact"]] >= PRED["ladder_intact_F"]},
        "ladder_floor": {"pred": f"S2u{PRED['ladder_floor']}r at floor in >= {need_floor} seeds",
                         "obs": floors[PRED["ladder_floor"]], "hit": floors[PRED["ladder_floor"]] >= need_floor},
    }

    # per-arm table (seed means) and the functional-response correlations (report only)
    rows = []
    for arm in ARM_NAMES:
        r1 = [shards[s]["arms"].query("arm == @arm and k == 1").iloc[0] for s in valid]
        r2 = [shards[s]["arms"].query("arm == @arm and k == 2").iloc[0] for s in valid]
        rows.append({
            "arm": arm, "E_k1": float(np.mean([r["online_acc"] for r in r1])),
            "E_k2": float(np.mean([r["online_acc"] for r in r2])),
            "floor_k1": sum(at_floor(shards[s]["arms"], arm) for s in valid),
            "memo_k1": float(np.mean([r["memo_acc_end"] for r in r1])),
            "first75_k1": float(np.mean([r["first75_acc"] for r in r1])),
            "logg2_start": float(np.mean([r["logg2_start"] for r in r1])),
            "neff2_start": float(np.mean([r["neff2_start"] for r in r1])),
            "logg1_start": float(np.mean([r["logg1_start"] for r in r1])),
            "pf_eps_w2": float(np.mean([r["pf_eps_w2"] for r in r1])),
            "pf_eps_w1": float(np.mean([r["pf_eps_w1"] for r in r1])),
            "pf_zero_w2": float(np.mean([r["pf_zero_w2"] for r in r1])),
            "gtr2_start": float(np.mean([r["gtr2_start"] for r in r1])),
            "zero2_start": float(np.mean([r["zero2_start"] for r in r1])),
            "deadunits2_start": float(np.mean([r["deadunits2_start"] for r in r1])),
            "neffT2_start": float(np.mean([r["neffT2_start"] for r in r1])),
            "gtr2_end_k1": float(np.mean([r["gtr2_end"] for r in r1])),
            "zero2_end_k1": float(np.mean([r["zero2_end"] for r in r1])),
            "step_w2_k1": float(np.mean([r["step_w2"] for r in r1])),
            "step_w1_k1": float(np.mean([r["step_w1"] for r in r1])),
            "effbar2_move_k1": float(np.mean([r["effbar2_end"] - r["effbar2_start"] for r in r1])),
            "logg2_end_k1": float(np.mean([r["logg2_end"] for r in r1])),
            "neff2_end_k1": float(np.mean([r["neff2_end"] for r in r1])),
            "lowgate2_end_k1": float(np.mean([r["lowgate2_end"] for r in r1])),
            "reset": bool(r1[0]["arm_reset"]) if not isinstance(r1[0]["arm_reset"], str)
            else r1[0]["arm_reset"] == "True"})
    table = pd.DataFrame(rows)
    res["arm_table"] = table
    corr = {}
    for sub, mask in (("all", np.ones(len(table), bool)), ("fresh_adam", table["reset"].to_numpy())):
        t = table[mask]
        corr[sub] = {c: spearman(t["E_k1"], t[c]) for c in ("logg2_start", "neff2_start", "pf_eps_w2",
                                                            "gtr2_start", "zero2_start", "neffT2_start",
                                                            "pf_zero_w2", "step_w2_k1", "logg2_end_k1",
                                                            "neff2_end_k1", "zero2_end_k1")}
        corr[sub]["n_arms"] = int(mask.sum())
    res["corr"] = corr

    # the natural trajectory (the prefix), seed means
    pref = pd.concat([shards[s]["prefix"] for s in valid])
    res["prefix"] = pref.groupby("task")[["online_acc", "memo_acc", "g1_end", "g2_end", "logg2_end",
                                          "neff2_end", "zbar2_end", "sigma2_end", "mu2_end"]].mean()
    return res


# --------------------------------------------------------------------------

def _f(x, nd=4):
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return "nan" if x is None or not np.isfinite(x) else f"{x:+.{nd}f}"


def _ci(p: dict) -> str:
    return f"{_f(p['mean'])} [{_f(p['lo'])}, {_f(p['hi'])}] ({p['n_pos']}/{p['n']} +)"


def summary_md(res: dict, missing: list, partial: bool, link: dict | None) -> str:
    L = [f"# {RUN_ID} — 判定", ""]
    if partial:
        L += [f"**--partial で集計（欠けている seed: {missing}）。登録判定ではない。**", ""]
    L += [f"有効 seed {res['n_valid']}（無効: {res['invalid_seeds'] or 'なし'}）", ""]
    if res["label"] == "INAPPLICABLE":
        return "\n".join(L + [f"**INAPPLICABLE** — {res['why']}"])
    B = res["B"]
    L += ["## 適用条件 (B)", "",
          f"- 自然な学習能力の差 N2 − N10（95%）: {_ci(B['natural_gap'])} → {'OK' if B['gap_ok'] else 'NG'}",
          f"- 第2層の平均 φ′（t2 > t10）: 全 seed で {'成立' if B['field_ok'] else '不成立'}", "",
          "## 主判定", "",
          "| 比較 | 97.5% | 95% |", "|---|---|---|",
          f"| P1 = R2_10 − N10 | {_ci(res['P1'])} | {_ci(res['P1_95'])} |",
          f"| P2 = N2r − S2_10r | {_ci(res['P2'])} | {_ci(res['P2_95'])} |", "",
          f"**ラベル: {res['label']}**", ""]
    if "recovery" in res:
        rc = res["recovery"]
        L += [f"回復率（報告のみ）: ρ1 = P1 / (N2 − N10) = {_f(rc['rho1'], 3)}（自然差 {_f(rc['gap_natural'])}）、"
              f"ρ2 = P2 / (N2r − N10r) = {_f(rc['rho2'], 3)}（{_f(rc['gap_fresh'])}）", ""]
    lad = res["ladder"]
    L += ["## 一様シフトの階段（登録した記述ラベル）", "",
          f"Δ* の seed 中央値（W2 座標の ln(rms/ε) の分位）: q0.1 {lad['dstar_median']['0.1']:.2f}・"
          f"q0.5 {lad['dstar_median']['0.5']:.2f}・q0.9 {lad['dstar_median']['0.9']:.2f}", "",
          "| Δ | F̄（N2r 比の適合率） | 床の seed 数 |", "|---|---|---|"]
    L += [f"| {d} | {lad['F'][d]:.3f} | {lad['at_floor'][d]} |" for d in LADDER]
    L += ["", f"**{lad['label']}**（Δ* より下: {lad['part_below']}・上: {lad['part_above']}）", "",
          "## 予測との照合", "", "| 項目 | 予測 | 結果 | |", "|---|---|---|---|"]
    pr = res["predictions"]
    L += [f"| P1 の平均（95% 区間に入れば ○） | {pr['P1_mean']['pred']:+.2f} | {_f(pr['P1_mean']['obs'], 3)} | "
          f"{'○' if pr['P1_mean']['hit'] else '×'} |",
          f"| P2 の平均（同） | {pr['P2_mean']['pred']:+.2f} | {_f(pr['P2_mean']['obs'], 3)} | "
          f"{'○' if pr['P2_mean']['hit'] else '×'} |",
          f"| 主ラベル | {pr['label']['pred']} {pr['label']['prob']:.0%} | {pr['label']['obs']} | "
          f"{'○' if pr['label']['hit'] else '×'} |",
          f"| Δ=10 はほぼ無傷 | {pr['ladder_intact']['pred']} | {pr['ladder_intact']['obs']:.3f} | "
          f"{'○' if pr['ladder_intact']['hit'] else '×'} |",
          f"| Δ=20 は床 | {pr['ladder_floor']['pred']} | {pr['ladder_floor']['obs']} | "
          f"{'○' if pr['ladder_floor']['hit'] else '×'} |", "",
          "## 副比較（95%・報告のみ）", "", "| 比較 | 内容 | k | 平均 [95%] |", "|---|---|---|---|"]
    L += [f"| {r['contrast']} | {r['what']} | {r['k']} | {_ci(r)} |" for r in res["secondary"]]
    t = res["arm_table"]
    L += ["", "## 腕ごと（seed 平均・k=1 は分岐後 1 タスク目）", "",
          "| 腕 | E k1 | E k2 | 床 | 訓練の φ′₂ 開始 | 0 の割合 開始 | 全画像 0 のユニット | log exp φ̄′₂ 開始 | "
          "neff₂(exp) 開始 | ε 域 W2 | 勾配 0 の W2 | 歩幅 W2 | 実効引数の移動 | 0 の割合 終了 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in t.iterrows():
        L.append(f"| {r['arm']} | {r['E_k1']:.4f} | {r['E_k2']:.4f} | {r['floor_k1']} | {r['gtr2_start']:.3g} | "
                 f"{r['zero2_start']:.3f} | {r['deadunits2_start']:.2f} | {r['logg2_start']:.2f} | "
                 f"{r['neff2_start']:.3f} | {r['pf_eps_w2']:.3f} | {r['pf_zero_w2']:.3f} | {r['step_w2_k1']:.3f} | "
                 f"{r['effbar2_move_k1']:+.2f} | {r['zero2_end_k1']:.3f} |")
    L += ["", "機能的な応答との順位相関（腕をまたぐ・報告のみ）", ""]
    for sub, c in res["corr"].items():
        L.append(f"- {sub}（{c['n_arms']} 腕）: " + "・".join(f"{k} {v:+.2f}" for k, v in c.items() if k != "n_arms"))
    if link is not None:
        L += ["", f"mucap_ee_0917 の ref との接頭部の一致（マシン依存・報告のみ）: {link}"]
    return "\n".join(L) + "\n"


def link_check(shards: dict, valid: list) -> dict:
    """The prefix against mucap_ee_0917's ref shards (same box, same machine): online_acc per task."""
    root = REPO / "results" / "mucap_ee_0917" / "runs"
    if not root.exists():
        root = Path("/home/issan/Projects/claude/wt/mucap_ee_0917/results/mucap_ee_0917/runs")
    out = {}
    for s in valid:
        f = root / f"ref_s{s}" / "per_task.csv"
        if not f.exists():
            out[s] = "no record"
            continue
        rec = pd.read_csv(f, float_precision="round_trip").set_index("task")["online_acc"]
        mine = shards[s]["prefix"].set_index("task")["online_acc"]
        out[s] = int(sum(float(rec.loc[t]) == float(mine.loc[t]) for t in mine.index if t in rec.index))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(MAIN_SRC))
    ap.add_argument("--out", default=None)
    ap.add_argument("--partial", action="store_true")
    ap.add_argument("--prereg", default=None, help="required prereg_commit (default: the runner's)")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args(argv)
    src = Path(args.src)
    out = Path(args.out) if args.out else src
    shards, missing = load(src)
    if missing and not args.partial:
        raise SystemExit(f"seeds without provenance: {missing} (use --partial deliberately)")
    prereg = args.prereg
    if prereg is None:
        txt = (REPO / "src" / "resp_ee_0917.py").read_text()
        prereg = next((ln.split('"')[1] for ln in txt.splitlines() if ln.startswith("PREREG_COMMIT = \"")), None)
    res = analyze(shards, prereg, args.epochs)
    link = link_check(shards, res["valid_seeds"]) if res["label"] != "INAPPLICABLE" else None
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.md").write_text(summary_md(res, missing, bool(missing), link))
    if res["label"] != "INAPPLICABLE":
        res["arm_table"].to_csv(out / "arm_table.csv", index=False)
        pd.DataFrame(res["secondary"]).to_csv(out / "secondary.csv", index=False)
        res["prefix"].to_csv(out / "prefix_mean.csv")
        rows = []
        for key in ("P1", "P2", "P1_95", "P2_95"):
            rows.append({"quantity": key, **res[key]})
        rows.append({"quantity": "natural_gap", **res["B"]["natural_gap"]})
        pd.DataFrame(rows).to_csv(out / "verdict.csv", index=False)
    slim = {k: v for k, v in res.items() if k not in ("arm_table", "prefix")}
    slim["link"] = link
    slim["missing"] = missing
    (out / "verdict.json").write_text(json.dumps(slim, indent=2, default=str))
    print((out / "summary.md").read_text())


if __name__ == "__main__":
    main()
