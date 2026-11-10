#!/usr/bin/env python3
"""spec §2・§3 の判定（JB1・JB2）と比 β = (B−N)/(X−N) の seed の bootstrap（10,000 回・百分位の 95%）。
judge.py の出力 per_seed.csv と paired.csv を読む。出力: results/bwdswap_1111/judge_beta.md・judge_beta.json"""
import csv, json
from pathlib import Path
import numpy as np
OUT = Path(__file__).resolve().parents[2] / "results/bwdswap_1111"
rows = list(csv.DictReader(open(OUT / "per_seed.csv")))
paired = {(r["setting"], r["direction"], r["comparison"]): r for r in csv.DictReader(open(OUT / "paired.csv"))}
rng = np.random.default_rng(20261111)
res, L = {}, ["| 設定 | 向き | X−N（97.5%） | B−N（97.5%） | JB1 | β（95%） | JB2 |", "|---" * 7 + "|"]
for st in ("RC", "RM", "F51"):
    for dr in ("restore", "reverse"):
        rr = [r for r in rows if r["setting"] == st and r["direction"] == dr]
        xn = np.array([float(r["X-N"]) for r in rr]); bn = np.array([float(r["B-N"]) for r in rr])
        n = len(rr); idx = rng.integers(0, n, size=(10000, n))
        beta = bn.mean() / xn.mean()
        bs = bn[idx].mean(1) / xn[idx].mean(1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        px, pb = paired[(st, dr, "X-N")], paired[(st, dr, "B-N")]
        xs, bsg = px["sign975"], pb["sign975"]
        jb1 = "B_NO_EFFECT" if bsg == "0" else ("B_SAME_DIRECTION" if bsg == xs else "B_OPPOSITE")
        jb2 = ("（X−N が 0 を含むので判定しない）" if xs == "0" else
               "BACKWARD_CARRIES_MOST" if lo > 0.5 else "FORWARD_NEEDED" if hi < 0.5 else "MIXED")
        res[f"{st}|{dr}"] = dict(n=n, XN=float(xn.mean()), BN=float(bn.mean()), beta=float(beta), beta_ci=[float(lo), float(hi)], JB1=jb1, JB2=jb2)
        f = lambda r: f"{float(r['mean']):+.3f} [{float(r['low975']):+.3f}, {float(r['high975']):+.3f}]"
        L.append(f"| {st} | {'戻す' if dr == 'restore' else '渡す'} | {f(px)} | {f(pb)} | {jb1} | {beta:.2f} [{lo:.2f}, {hi:.2f}] | {jb2} |")
(OUT / "judge_beta.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
(OUT / "judge_beta.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))
