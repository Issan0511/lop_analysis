#!/usr/bin/env python3
"""The n_eff -> F curve and its prediction band, fitted inside resp_ee_0917 only (spec section 3).

    python3 analysis/neff_pred_0917/calibrate.py            # writes results/neff_pred_0917/calibration.json

Reads nothing but resp_ee_0917's committed per-seed prefix tables and arm table (their sha256 go into the
output); never reads a neff_pred_0917 shard.

Points (primary, "natural"): every seed s = 0-9 and branch task t = 2-21 of the natural trajectory,
    x = the layer's training-derivative n_eff at the end of task t (the net that starts task t+1; in RL the
        images do not change, so this is the branch point's value),
    E = online accuracy of task t+1,   E0 = the same seed's E at t = 2,
    F = clip((E - 0.1) / (E0 - 0.1), 0, 1).
Curve: F = 1 / (1 + exp(-(log10 x - c) / w)), least squares over (c, w) on a grid refined twice
(x is floored at 1e-12, so a net whose derivative is 0 on every image gets F -> 0).
Band: Mondrian split-conformal on |F - f|, three bins of the fitted value (f < 0.2, 0.2 <= f < 0.8,
f >= 0.8); a bin's half-width is its ceil((n + 1) 0.95)-th smallest residual.  LOSO: refit on 9 seeds,
score the held-out seed's t in {2, 5, 10, 20}; the coverage is reported and gated in checks (S5).

Predictors (all fitted the same way, each on its own x):
    L2   layer 2 (the primary layer in this box: its deepest hidden layer in the fixed-scale class)
    min  min(layer 1, layer 2)
Secondary, design-literal: the 26 arm means (E_k1, neffT2_start), F against E(N2), one global bin.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "results" / "resp_ee_0917"
OUT = REPO / "results" / "neff_pred_0917" / "calibration.json"
SEEDS = tuple(range(10))
T_FIT = tuple(range(2, 22))          # E(t) needs task t+1 <= 22
T_LOSO = (2, 5, 10, 20)
CHANCE = 0.1
X_FLOOR = 1e-12
BIN_EDGES = (0.2, 0.8)
LEVEL = 0.95


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def logistic(u: np.ndarray, c: float, w: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(u - c) / w))


def u_of(x) -> np.ndarray:
    return np.log10(np.maximum(np.asarray(x, dtype=np.float64), X_FLOOR))


def fit(u: np.ndarray, F: np.ndarray) -> tuple[float, float, float]:
    """Least squares over (c, w): a grid, then two local refinements (deterministic, no optimizer)."""
    c_lo, c_hi, w_lo, w_hi = -6.0, 1.0, 0.02, 3.0
    best = None
    for step_c, step_w in ((0.01, 0.01), (0.0005, 0.0005), (0.00002, 0.00002)):
        cs = np.arange(c_lo, c_hi + step_c / 2, step_c)
        ws = np.arange(w_lo, w_hi + step_w / 2, step_w)
        ws = ws[ws > 0]
        for w in ws:
            pred = 1.0 / (1.0 + np.exp(-(u[None, :] - cs[:, None]) / w))
            sse = ((pred - F[None, :]) ** 2).sum(1)
            i = int(np.argmin(sse))
            if best is None or sse[i] < best[0]:
                best = (float(sse[i]), float(cs[i]), float(w))
        _, c0, w0 = best
        c_lo, c_hi = c0 - 20 * step_c, c0 + 20 * step_c
        w_lo, w_hi = max(step_w / 10, w0 - 20 * step_w), w0 + 20 * step_w
    sse, c, w = best
    return c, w, sse


def bin_of(f: np.ndarray) -> np.ndarray:
    return np.digitize(f, BIN_EDGES)          # 0 bottom, 1 middle, 2 top


def conformal_q(scores: np.ndarray, level: float = LEVEL) -> float:
    n = len(scores)
    k = math.ceil((n + 1) * level)
    if n == 0 or k > n:
        return float("inf")
    return float(np.sort(scores)[k - 1])


def band(u: np.ndarray, F: np.ndarray, c: float, w: float, mondrian: bool = True) -> dict:
    f = logistic(u, c, w)
    r = np.abs(F - f)
    if not mondrian:
        return {"q": [conformal_q(r)] * 3, "n": [len(r)] * 3}
    b = bin_of(f)
    return {"q": [conformal_q(r[b == k]) for k in range(3)], "n": [int((b == k).sum()) for k in range(3)]}


def predict(x, cal: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(f, lower, upper) for x under one calibration entry."""
    u = u_of(x)
    f = logistic(u, cal["c"], cal["w"])
    q = np.asarray(cal["q"])[bin_of(f)]
    return f, f - q, f + q


def natural_points(layer_key: str) -> pd.DataFrame:
    rows = []
    for s in SEEDS:
        p = pd.read_csv(SRC / "runs" / f"s{s}" / "prefix.csv", float_precision="round_trip").set_index("task")
        e0 = float(p.loc[3, "online_acc"])
        for t in T_FIT:
            if layer_key == "min":
                x = min(float(p.loc[t, "neffT1_end"]), float(p.loc[t, "neffT2_end"]))
            else:
                x = float(p.loc[t, layer_key])
            e = float(p.loc[t + 1, "online_acc"])
            F = min(max((e - CHANCE) / (e0 - CHANCE), 0.0), 1.0)
            rows.append({"seed": s, "t": t, "x": x, "E": e, "E0": e0, "F": F,
                         "g": float(p.loc[t, "gtr2_end"]), "zero": float(p.loc[t, "zero2_end"])})
    return pd.DataFrame(rows)


def calibrate(d: pd.DataFrame) -> dict:
    u, F = u_of(d["x"]), d["F"].to_numpy()
    c, w, sse = fit(u, F)
    b = band(u, F, c, w)
    # LOSO
    hits, n = 0, 0
    per_seed = {}
    for s in SEEDS:
        tr, te = d[d.seed != s], d[(d.seed == s) & d.t.isin(T_LOSO)]
        cs, ws, _ = fit(u_of(tr["x"]), tr["F"].to_numpy())
        bs = band(u_of(tr["x"]), tr["F"].to_numpy(), cs, ws)
        f, lo, hi = predict(te["x"], {"c": cs, "w": ws, "q": bs["q"]})
        inside = (te["F"].to_numpy() >= lo) & (te["F"].to_numpy() <= hi)
        hits += int(inside.sum())
        n += len(inside)
        per_seed[str(s)] = {"c": cs, "w": ws, "q": bs["q"], "inside": inside.astype(int).tolist()}
    f = logistic(u, c, w)
    return {"c": c, "w": w, "sse": sse, "n_points": int(len(d)), "q": b["q"], "n_bin": b["n"],
            "x_half": float(10 ** c), "rmse": float(np.sqrt(sse / len(d))),
            "loso": {"inside": hits, "n": n, "coverage": hits / n, "per_seed": per_seed},
            "residual_max_by_bin": [float(np.abs(F - f)[bin_of(f) == k].max()) if (bin_of(f) == k).any()
                                    else None for k in range(3)]}


def arms26() -> dict:
    t = pd.read_csv(SRC / "arm_table.csv", float_precision="round_trip")
    e0 = float(t.loc[t.arm == "N2", "E_k1"].iloc[0])
    F = np.clip((t["E_k1"].to_numpy() - CHANCE) / (e0 - CHANCE), 0, 1)
    u = u_of(t["neffT2_start"])
    c, w, sse = fit(u, F)
    b = band(u, F, c, w, mondrian=False)
    return {"c": c, "w": w, "sse": sse, "n_points": int(len(t)), "q": b["q"], "n_bin": b["n"],
            "x_half": float(10 ** c), "E0": e0}


def main() -> None:
    inputs = [SRC / "runs" / f"s{s}" / "prefix.csv" for s in SEEDS] + [SRC / "arm_table.csv"]
    out = {"experiment": "neff_pred_0917", "source": "resp_ee_0917 (natural prefix, seeds 0-9)",
           "inputs_sha256": {str(p.relative_to(REPO)): sha256(p) for p in inputs},
           "t_fit": list(T_FIT), "t_loso": list(T_LOSO), "chance": CHANCE, "x_floor": X_FLOOR,
           "bin_edges": list(BIN_EDGES), "level": LEVEL,
           "form": "F = 1/(1+exp(-(log10 x - c)/w)); band f +/- q[bin(f)]"}
    pts = {}
    for name, key in (("L2", "neffT2_end"), ("min", "min")):
        d = natural_points(key)
        pts[name] = d
        out[name] = calibrate(d)
    out["arms26"] = arms26()
    # the calibration points themselves, for the figure and for checks
    out["points"] = {k: v.to_dict(orient="list") for k, v in pts.items()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    for k in ("L2", "min", "arms26"):
        e = out[k]
        print(f"{k:6s} c={e['c']:+.5f} (x_half {e['x_half']:.4g})  w={e['w']:.5f}  rmse={math.sqrt(e['sse'] / e['n_points']):.4f}"
              f"  q={['%.4f' % q for q in e['q']]}  n_bin={e['n_bin']}"
              + (f"  LOSO {e['loso']['inside']}/{e['loso']['n']}" if "loso" in e else ""))


if __name__ == "__main__":
    main()
