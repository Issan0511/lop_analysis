#!/usr/bin/env python3
"""Round 1 valley R_eps_wall (spec_sink_roots_0930_round1 §1): the four judgement quantities of the request, for
eps1 in {1e-6, 1e-8 (existing v4), 1e-12}, GELU and SiLU, seeds 0 and 1, from the v4 task-end unit statistics.
(1) the depth (closed-unit top) where the median |Delta top| per task of closed units falls to 0.01 (GELU) / 0.1 (SiLU),
    interpolated over 0.25-wide bands of the top; (2) the minimum closed top over the trajectory;
(3) all-closed fraction in tasks 151-200 and mu_minus (the fraction of all-closed units that are open at the next task end);
(4) the medians of k (= pplus * 1200) and of the top over live units, tasks 151-200."""
import json
from pathlib import Path
import numpy as np

NEW = Path("/home/issan/Projects/obsidian-research-data/sink_roots_0930/round1/R_eps_wall")
OLD = Path("/home/issan/Projects/obsidian-research-data/elu_alpha_ladder_1layer_0925/runs")
RES = Path(__file__).resolve().parents[2] / "results" / "sink_roots_0930"
THR = {"GELU": 0.01, "SILU": 0.1}


def load(act, eps, s):
    f = (OLD / f"{act}_s{s}_v4.npz") if eps == "1e-8" else (NEW / f"{act}_eps{eps}_s{s}_v4.npz")
    return np.load(f, allow_pickle=True)


def quantities(d, act):
    top = d["task_zmax"]; pp = d["task_pplus"]
    closed = top <= 0                                   # all inputs closed at the task end
    # (1) |Delta top| per task for units closed at both ends, binned by the top at the earlier end
    dt = np.abs(np.diff(top, axis=0)); cc = closed[:-1] & closed[1:]
    x, y = top[:-1][cc], dt[cc]
    lo = np.floor(x.min() / 0.25) * 0.25 if len(x) else 0
    bands = np.arange(lo, 0, 0.25)
    med = []
    for b in bands:
        sel = (x >= b) & (x < b + 0.25)
        med.append((b + 0.125, float(np.median(y[sel])) if sel.sum() >= 20 else np.nan))
    thr = THR[act]; depth = np.nan
    pts = [(c, m) for c, m in med if np.isfinite(m)]
    for (c1, m1), (c2, m2) in zip(pts[:-1], pts[1:]):   # ascending band centers: the deepest band where the median crosses thr
        if (m1 - thr) * (m2 - thr) <= 0 and m1 != m2:
            depth = c1 + (thr - m1) * (c2 - c1) / (m2 - m1)
            break
    ctop = top[closed]
    w = slice(150, 200)
    allc = closed[w].mean()
    trans = closed[150:199] & ~closed[151:200]
    mu_minus = float(trans.sum() / max(closed[150:199].sum(), 1))
    live = ~closed[w]
    k = (pp[w] * 1200)[live]; tl = top[w][live]
    return {"depth_at_thr": float(depth), "min_closed_top": float(ctop.min()) if len(ctop) else np.nan,
            "allclosed_151_200": float(allc), "allclosed_t200": float(closed[199].mean()), "mu_minus_151_200": mu_minus,
            "k_live_median": float(np.median(k)) if len(k) else np.nan,
            "top_live_median": float(np.median(tl)) if len(tl) else np.nan,
            "band_medians": [(round(c, 3), m) for c, m in med]}


def main():
    out = {}
    for act in ("GELU", "SILU"):
        for eps in ("1e-6", "1e-8", "1e-12"):
            for s in (0, 1):
                q = quantities(load(act, eps, s), act)
                out[f"{act}|{eps}|s{s}"] = q
                print(f"{act:4s} eps {eps:5s} s{s}: depth(|dtop|={THR[act]}) {q['depth_at_thr']:+.2f}  min closed top {q['min_closed_top']:+.2f}"
                      f"  all-closed 151-200 {q['allclosed_151_200']:.3f} (t200 {q['allclosed_t200']:.2f})  mu- {q['mu_minus_151_200']:.3f}"
                      f"  live k med {q['k_live_median']:.1f} top med {q['top_live_median']:.2f}")
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "round1_R_eps_wall.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
