#!/usr/bin/env python3
"""swish_battle_0917 — battle tables and the three registered calls (spec §5).

    python3 analysis/swish_battle_0917/report.py --box mlp   # after every mlp run is complete
    python3 analysis/swish_battle_0917/report.py --box cnn

Refuses to read a box that is not complete (spec, harness lesson 8).  `--src` points the
reader at a smoke directory; `--allow-partial` exists only for such smoke directories.

Window = tasks 31-50 of online_acc (host verdicts).  Same-device pairs: seed-paired sign
test with the host verdict's `sign` and win rule (at most one seed against and p < .05).
Cross-device pairs (mlp vs the GPU reference R): unpaired mean difference only.
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
OUT = REPO / "results" / "swish_battle_0917"
WIN, EARLY, LATE = (31, 50), (1, 10), (41, 50)
SEEDS = list(range(10))
N_TASKS = 50
SITES = {"mlp": ("l1", "l2"), "cnn": ("c1", "c2", "f1", "f2")}
NEW = {"cnn": ("SNAc3", "SWA1", "SW1", "SWA3", "SW3"),
       "mlp": ("LR", "SNA", "SN06", "SN3", "SNAc3", "SW1", "SW3", "SWA1", "SWA3")}
REF = {"cnn": (Path("/home/issan/Projects/claude/proj_004_drift/results/rlcifar_cnn_0908"),
               {"SNA": "SNA", "SN06": "SN06", "SN3": "SN3", "LR": "LR", "R": "R"}),
       "mlp": (REPO / "results" / "pmnist_rlmnist_0906",
               {"R@gpu": "R", "LR@gpu": "LR", "SNA@gpu": "SNA"})}
# alpha of the fixed arms, for the gate argument at the mean
FIXED_ALPHA = {"SW1": 1.0, "SW3": 3.0, "SN06": 0.6, "SN3": 3.0}
SNAKE = {"SNA", "SNAc3", "SN06", "SN3"}
SWISH = {"SW1", "SW3", "SWA1", "SWA3"}
ADAPT_C = {"SWA1": 1.0, "SWA3": 3.0, "SNA": 0.6, "SNAc3": 3.0}
PAIRS = {
    "cnn": [("SNAc3", "SNA"), ("SNAc3", "SN3"), ("SN3", "SNA"), ("SN06", "SNA"),
            ("SWA1", "SW1"), ("SWA3", "SW3"), ("SWA1", "SWA3"),
            ("SWA1", "SNA"), ("SWA3", "SNA"), ("SWA1", "LR"), ("SWA3", "LR"),
            ("SW1", "LR"), ("SW3", "LR"), ("SW1", "R")],
    "mlp": [("SWA1", "SW1"), ("SWA3", "SW3"), ("SWA1", "SWA3"),
            ("SWA1", "SNA"), ("SWA3", "SNA"), ("SWA1", "LR"), ("SWA3", "LR"),
            ("SNAc3", "SNA"), ("SN3", "SNA"), ("SN06", "SNA"), ("SW1", "LR"), ("SW3", "LR"),
            ("LR", "LR@gpu"), ("SNA", "SNA@gpu")],
}
CROSS = {"mlp": [("SWA1", "R@gpu"), ("SWA3", "R@gpu"), ("SW1", "R@gpu"), ("SW3", "R@gpu"),
                 ("LR", "R@gpu")], "cnn": []}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def sign(diff):
    """The host verdict's exact two-sided sign test (zeros dropped)."""
    d = np.asarray(diff, float)
    nz = d[d != 0]
    n = len(nz)
    pos = int((nz > 0).sum())
    k = min(pos, n - pos)
    p = min(sum(comb(n, i) for i in range(k + 1)) / 2 ** n * 2, 1.0) if n else float("nan")
    return pos, n, p


def load(box: str, src: Path, allow_partial: bool):
    parts, files, missing = [], {}, []
    for arm in NEW[box]:
        for s in SEEDS:
            d = src / box / arm / f"seed{s}"
            f, pv = d / "per_task.csv", d / "provenance.json"
            if not pv.exists():
                missing.append(f"{arm}/seed{s}")
                continue
            x = pd.read_csv(f)
            x["arm"] = arm
            parts.append(x)
            files[f"{box}/{arm}/seed{s}"] = json.loads(pv.read_text())
    if missing and not allow_partial:
        raise SystemExit(f"{box}: {len(missing)} runs incomplete ({', '.join(missing[:6])} ...); "
                         "the report reads complete boxes only")
    root, arms = REF[box]
    refs = {}
    for label, folder in arms.items():
        f = root / folder / "per_task.csv"
        x = pd.read_csv(f)
        x = x[x.seed.isin(SEEDS)].copy()
        x["arm"] = label
        parts.append(x)
        refs[label] = {"path": str(f), "sha256": sha(f)}
    d = pd.concat(parts, ignore_index=True)
    return d[d.online_acc.notna()], files, refs, missing


def per_seed(d: pd.DataFrame, arm: str, col: str, lo: int, hi: int) -> pd.Series:
    g = d[(d.arm == arm) & (d.task >= lo) & (d.task <= hi)]
    full = [s for s, gg in g.groupby("seed") if len(gg) == hi - lo + 1 and col in gg]
    return g[g.seed.isin(full)].groupby("seed")[col].mean()


def arm_row(d: pd.DataFrame, box: str, arm: str) -> dict:
    x = d[d.arm == arm]
    r = {"arm": arm, "n": int(x.seed.nunique())}
    late = per_seed(d, arm, "online_acc", *WIN)
    r["n_full"] = int(len(late))
    r["onl"] = float(late.mean()) if len(late) else float("nan")
    r["onl_sd"] = float(late.std(ddof=1)) if len(late) > 1 else float("nan")
    e = per_seed(d, arm, "online_acc", *EARLY)
    l2 = per_seed(d, arm, "online_acc", *LATE)
    r["early"] = float(e.mean()) if len(e) else float("nan")
    r["drop"] = float((e - l2).dropna().mean()) if len(l2) else float("nan")
    r["memo"] = float(per_seed(d, arm, "memo_acc", *WIN).mean())
    col = []
    for s, g in x.groupby("seed"):
        g = g.sort_values("task")
        bad = g[g.online_acc < 0.5]
        col.append(int(bad.task.min()) if len(bad) else 0)
    hit = [c for c in col if c]
    r["collapsed"] = len(hit)
    r["collapse_task_med"] = float(np.median(hit)) if hit else float("nan")
    r["diverged"] = int(x.groupby("seed").task.max().lt(N_TASKS).sum())
    for s in SITES[box]:
        r[f"mob_{s}"] = float(per_seed(d, arm, f"mob_{s}", *WIN).mean())
        zb = per_seed(d, arm, f"zbar_{s}", *WIN)
        zs = per_seed(d, arm, f"zsd_{s}", *WIN)
        r[f"zbar_{s}"] = float(zb.mean())
        r[f"zsd_{s}"] = float(zs.mean())
        if f"alpha_med_{s}" in x and x[f"alpha_med_{s}"].notna().any():   # concat unions columns
            a = per_seed(d, arm, f"alpha_med_{s}", *WIN)
            r[f"alpha_{s}"] = float(a.mean())
            r[f"clip_{s}"] = float(per_seed(d, arm, f"alpha_clip_frac_{s}", *WIN).mean())
            aa = a
        else:
            aa = pd.Series(FIXED_ALPHA.get(arm, float("nan")), index=zb.index)
        # gate argument at the mean: Snake 2*alpha*zbar (valley at -pi/2),
        # Swish alpha*zbar (phi' = 0 at -1.278, minimum at -2.40)
        k = 2.0 if arm.split("@")[0] in SNAKE else 1.0
        r[f"garg_{s}"] = float((k * aa * zb).mean()) if arm.split("@")[0] in SNAKE | SWISH else float("nan")
        # standardized depth zbar/zsd; a dead layer has zsd = 0 and no depth to report
        r[f"u_{s}"] = float((zb / zs.where(zs > 0)).mean())
    s0 = SITES[box][0]
    wr = [g.sort_values("task")[f"w_norm_{s0}"] for _, g in x.groupby("seed")]
    r["w_growth"] = float(np.median([w.tail(5).mean() / w.iloc[0] for w in wr]))
    return r


def pair(d, a, b):
    x, y = per_seed(d, a, "online_acc", *WIN), per_seed(d, b, "online_acc", *WIN)
    s = sorted(set(x.index) & set(y.index))
    if not s:
        return None
    v = (x[s] - y[s]).values
    pos, n, p = sign(v)
    return {"a": a, "b": b, "mean": float(v.mean()),
            "se": float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else float("nan"),
            "sd": float(v.std(ddof=1)) if len(v) > 1 else float("nan"),
            "wins": pos, "n": n, "p": float(p),
            "sig_win": bool(n - pos <= 1 and p < 0.05), "sig_loss": bool(pos <= 1 and p < 0.05)}


def cross(d, a, b):
    x, y = per_seed(d, a, "online_acc", *WIN), per_seed(d, b, "online_acc", *WIN)
    if not len(x) or not len(y):
        return None
    return {"a": a, "b": b, "mean": float(x.mean() - y.mean()),
            "se": float(math.sqrt(x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y))),
            "n_a": int(len(x)), "n_b": int(len(y))}


def verdict_ab(p):
    if p is None:
        return "MISSING"
    return "WIN" if p["sig_win"] else "LOSS" if p["sig_loss"] else "TIE"


def labels(box, d, P, V):
    L = {}
    if box == "cnn":
        a, g = P.get(("SNAc3", "SNA")), P.get(("SN3", "SNA"))
        if a and g:
            q = a["mean"] / g["mean"] if g["mean"] else float("nan")
            L["A_SNAc3"] = ("WORSE" if a["sig_loss"] else
                            "ADAPTATION_COST" if not a["sig_win"] else
                            "C_VALUE" if q >= 0.5 else "PARTIAL")
            L["A_q"] = q
    for c, (ad, fx) in (("c1", ("SWA1", "SW1")), ("c3", ("SWA3", "SW3"))):
        v = verdict_ab(P.get((ad, fx)))
        L[f"B_{c}"] = {"WIN": "ADAPT_WINS", "LOSS": "ADAPT_LOSES", "TIE": "TIE"}.get(v, v)
    best = max(("SWA1", "SWA3"), key=lambda k: V[k]["onl"])
    L["C_swa_best"] = best
    for opp in ("SNA", "LR"):
        v = verdict_ab(P.get((best, opp)))
        L[f"C_vs_{opp}"] = {"WIN": "SWA_ABOVE", "LOSS": "SWA_BELOW", "TIE": "TIE"}.get(v, v)
    L["C_ranking"] = [k for k, _ in sorted(((k, v["onl"]) for k, v in V.items()),
                                           key=lambda kv: -kv[1])]
    if box == "mlp":
        for arm in ("LR", "SNA"):
            p = P.get((arm, f"{arm}@gpu"))
            if p:
                sd = math.sqrt((V[arm]["onl_sd"] ** 2 + V[f"{arm}@gpu"]["onl_sd"] ** 2) / 2)
                L[f"D_{arm}"] = {"cpu_minus_gpu": p["mean"], "paired_sd": p["sd"], "seed_sd": sd,
                                 "within_2sd": bool(abs(p["mean"]) <= 2 * sd)}
    return L


def fmt(v, f=".4f"):
    return "—" if v is None or (isinstance(v, float) and v != v) else format(v, f)


def write(box, d, files, refs, missing, src):
    arms = list(dict.fromkeys(list(REF[box][1]) + list(NEW[box])))
    V = {a: arm_row(d, box, a) for a in arms}
    P = {k: pair(d, *k) for k in PAIRS[box]}
    X = {k: cross(d, *k) for k in CROSS[box]}
    L = labels(box, d, P, V)
    S = SITES[box]
    o = [f"# swish_battle_0917 — {box}\n",
         f"spec: `specs/spec_swish_battle_0917.md`。窓 = タスク {WIN[0]}–{WIN[1]} の online_acc（seed 平均 ± seed 間 SD）。"
         "早期 = t1–10、低下 = 早期 − t41–50。崩壊 = online<0.5 になった seed 数（その最初のタスクの中央値）。"
         "mob・α・ゲート引数は窓の平均。ゲート引数は Snake が 2α·z̄（谷 −π/2 = −1.57）、Swish が α·z̄（φ′=0 は −1.28、最小は −2.40）。"
         "u = z̄/zsd（ユニット別の中央値どうしの比、目安）。\n"]
    if missing:
        o.append(f"**未完了 {len(missing)} 本（スモーク専用の --allow-partial で読んだ）**\n")
    hdr = ("| 腕 | n | 窓 | 早期 | 低下 | memo | 崩壊 |" + "".join(f" mob {s} |" for s in S)
           + "".join(f" 引数 {s} |" for s in S) + "".join(f" u {s} |" for s in S) + " ‖w‖比 |")
    o += [hdr, "|" + "---|" * (hdr.count("|") - 1)]
    for a in sorted(arms, key=lambda k: -V[k]["onl"] if V[k]["onl"] == V[k]["onl"] else 1):
        v = V[a]
        colp = f"{v['collapsed']}/{v['n']}" + (f" (t{v['collapse_task_med']:.0f})" if v["collapsed"] else "")
        o.append(f"| {a} | {v['n_full']} | {fmt(v['onl'])} ± {fmt(v['onl_sd'])} | {fmt(v['early'])} | "
                 f"{fmt(v['drop'], '+.4f')} | {fmt(v['memo'])} | {colp} |"
                 + "".join(f" {fmt(v[f'mob_{s}'], '.2f')} |" for s in S)
                 + "".join(f" {fmt(v[f'garg_{s}'], '.2f')} |" for s in S)
                 + "".join(f" {fmt(v[f'u_{s}'], '.2f')} |" for s in S)
                 + f" {fmt(v['w_growth'], '.2f')} |")
    ada = [a for a in arms if f"alpha_{S[0]}" in V[a]]
    if ada:
        o += ["\n適応の腕の α（窓の平均、ユニット／チャネルの中央値）と clip 率:\n",
              "| 腕 |" + "".join(f" α {s} |" for s in S) + " clip 最大 |",
              "|---|" + "---|" * (len(S) + 1)]
        for a in ada:
            v = V[a]
            o.append(f"| {a} |" + "".join(f" {fmt(v[f'alpha_{s}'], '.3f')} |" for s in S)
                     + f" {fmt(100 * max(v[f'clip_{s}'] for s in S), '.1f')}% |")
    o += ["\n対の差（窓の online_acc、seed 対応）:\n", "| a − b | 平均 ± SE | a の勝ち | p |", "|---|---|---|---|"]
    for k, p in P.items():
        if p:
            o.append(f"| {k[0]} − {k[1]} | {p['mean']:+.4f} ± {p['se']:.4f} | {p['wins']}/{p['n']} | {p['p']:.4f} |")
    if any(X.values()):
        o += ["\n装置をまたぐ差（対応なし、平均差 ± SE）:\n", "| a − b | 平均 ± SE |", "|---|---|"]
        for k, p in X.items():
            if p:
                o.append(f"| {k[0]} − {k[1]} | {p['mean']:+.4f} ± {p['se']:.4f} |")
    o += ["\n## 判定（spec §5）\n"]
    for k, v in L.items():
        o.append(f"- **{k}** = `{json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v}`")
    txt = "\n".join(o) + "\n"
    (src / f"summary_{box}.md").write_text(txt)
    (src / f"verdict_{box}.json").write_text(json.dumps(
        {"labels": L, "pairs": {f"{a}-{b}": p for (a, b), p in P.items() if p},
         "cross": {f"{a}-{b}": p for (a, b), p in X.items() if p},
         "arms": V, "refs": refs,
         "runs": {k: {kk: f[kk] for kk in ("git_hash", "git_dirty_code", "device", "threads",
                                          "tasks_completed", "wall_clock_s")}
                  for k, f in files.items()},
         "missing": missing}, indent=1, default=float))
    print(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--box", required=True, choices=("mlp", "cnn"))
    ap.add_argument("--src", default=str(OUT))
    ap.add_argument("--allow-partial", action="store_true")
    a = ap.parse_args()
    src = Path(a.src)
    if a.allow_partial and src.resolve() == OUT.resolve():
        raise SystemExit("--allow-partial is for smoke directories only")
    d, files, refs, missing = load(a.box, src, a.allow_partial)
    write(a.box, d, files, refs, missing, src)


if __name__ == "__main__":
    main()
