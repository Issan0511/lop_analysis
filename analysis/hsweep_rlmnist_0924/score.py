#!/usr/bin/env python3
"""Score the registered predictions (Claude PREDICTIONS_claude.md, Codex PREDICTIONS_codex_raw.txt) against per_seed.csv.

Group values are seed medians.  "Fitting" arms: median late train accuracy >= .99 (spec §4).
Item 6 is scored in the spec's CAPACITY direction (f10 non-decreasing in h among fitting arms); Codex
read the wording the other way and gave 1%, but stated 55% for this direction, which is what we use.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[2] / "results/hsweep_rlmnist_0924"

CLAUDE = {1: .40, 2: .85, 3: .97, 4: .15, 5: .20, 6: .60, 7: .75, 8: .70, 9: .55, 10: .55, 11: .65, 12: .70}
CODEX = {1: .25, 2: .80, 4: .13, 5: .23, 6: .55, 7: .58, 8: .70, 9: .59, 10: .70, 11: .82}
ITEMS = {1: "h=8 fits", 2: "h=16 fits", 3: "all h>=32 fit", 4: "group P2 in a fitting arm",
         5: "f10<=.3 at min fitting h", 6: "f10 non-decreasing in h (fitting arms)",
         7: "read_f1 < unread_f1-.1 in all fitting arms", 8: "loglog>=.5 in all fitting arms",
         9: "V/h in [30,90] in all fitting arms", 10: "f10(300)>f10(100)", 11: "B(min fitting h)>B(100)",
         12: "if h=8 unfit: V50(h8)/V50(h16)<.5"}
POINT = {  # h: (f10, B) point predictions
    "claude": {300: (.62, .93), 100: (.58, .925), 64: (.55, .93), 32: (.50, .94), 16: (.42, .95), 8: (.35, .96)},
    "codex": {300: (.61, .912), 100: (.58, .925), 64: (.56, .934), 32: (.51, .947), 16: (.44, .966), 8: (.32, .985)},
}


def main() -> None:
    d = pd.read_csv(OUT / "per_seed.csv")
    g = d.groupby("hidden").median(numeric_only=True).sort_index()
    n = d.groupby("hidden").size()
    p2 = d.groupby("hidden").P2.sum()
    fit = g.late_train_acc >= 0.99
    fits = g[fit]
    o = {}
    o[1] = bool(fit.get(8, False)); o[2] = bool(fit.get(16, False)); o[3] = bool(all(fit[h] for h in g.index if h >= 32))
    o[4] = bool(any(p2[h] >= 4 for h in fits.index))
    hmin = int(fits.index.min()); o[5] = bool(fits.loc[hmin, "f10_s20_39"] <= 0.3)
    f10 = fits["f10_s20_39"].to_numpy(); o[6] = bool(np.all(np.diff(f10) >= 0))
    o[7] = bool(all(fits.read_f1 < fits.unread_f1 - 0.1)); o[8] = bool(all(fits.late_loglog >= 0.5))
    o[9] = bool(all((fits.V_t50_per_h >= 30) & (fits.V_t50_per_h <= 90)))
    o[10] = bool(g.loc[300, "f10_s20_39"] > g.loc[100, "f10_s20_39"]) if 300 in g.index else None
    o[11] = bool(fits.loc[hmin, "late_B"] > g.loc[100, "late_B"])
    o[12] = (bool(g.loc[8, "V_t50"] / g.loc[16, "V_t50"] < 0.5) if (8 in g.index and not fit.get(8, False)) else None)
    print("arms (seeds):", dict(n)); print("fitting:", list(fits.index), " min fitting h:", hmin)
    rows = []
    for k, name in ITEMS.items():
        if o[k] is None:
            rows.append((k, name, "n/a", CLAUDE.get(k), CODEX.get(k), None, None)); continue
        y = 1.0 if o[k] else 0.0
        bc = (CLAUDE[k] - y) ** 2 if k in CLAUDE else None
        bx = (CODEX[k] - y) ** 2 if k in CODEX else None
        rows.append((k, name, o[k], CLAUDE.get(k), CODEX.get(k), bc, bx))
    tab = pd.DataFrame(rows, columns=["item", "statement", "outcome", "p_claude", "p_codex", "brier_claude", "brier_codex"])
    print(tab.to_string(index=False))
    shared = tab[tab.item.isin(CODEX.keys()) & tab.outcome.isin([True, False])]
    print(f"\nBrier (shared {len(shared)} items): Claude {shared.brier_claude.mean():.3f}  Codex {shared.brier_codex.mean():.3f}")
    allc = tab[tab.outcome.isin([True, False]) & tab.p_claude.notna()]
    print(f"Brier Claude all {len(allc)} items: {allc.brier_claude.mean():.3f}")
    print("\npoint predictions |pred-obs| (f10, B):")
    for who, pp in POINT.items():
        err = {h: (round(abs(pp[h][0] - g.loc[h, 'f10_s20_39']), 3), round(abs(pp[h][1] - g.loc[h, 'late_B']), 3)) for h in pp if h in g.index}
        mean_f = np.mean([e[0] for e in err.values()]); mean_b = np.mean([e[1] for e in err.values()])
        print(f"  {who:7s} {err}  mean f10 err {mean_f:.3f}  mean B err {mean_b:.3f}")
    json.dump({"outcomes": {str(k): v for k, v in o.items()}, "fitting": [int(h) for h in fits.index]},
              open(OUT / "prediction_scores.json", "w"), indent=2)
    tab.to_csv(OUT / "prediction_scores.csv", index=False)


if __name__ == "__main__":
    main()
