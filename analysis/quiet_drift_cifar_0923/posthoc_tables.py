#!/usr/bin/env python3
"""quiet_drift_cifar_0923 -- the post hoc tables behind summary.md (not registered verdicts).

Reads the pass-1 probes, pass-2/2b diags, pass-4 lag bursts and pass-5 float32 checks and writes
posthoc_tables.json with, per bundle (seed medians):
  lock        tau of sqrt(v) (median over W1 coordinates), of the loss, the active-set margin speed
  cost        loss e-folds in the pre-burst window and ||W1||^2 growth per e-fold
  per_step    growth per step and its factors ||W1||, ||u1||, cos(-u1, W1); the step distribution
  aim         cos(G, u), cos(G, m), cos(G, G/sqrt(v)); batch NSR and the iid-EMA prediction of
              cos(G, m); ||m||^2 over its iid-EMA prediction; the beta1-weighted staleness factor
  turnover    cos(G_t, G_t+k), relative gradient change per unit step
  float32     cos(G32, G64), cos(G32, m) at 8k/12k/16k/20k
Windows: pass 1's pre-burst quiet window (per_seed.csv), from hit999 + 2,500 where noted.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "quiet_drift_cifar_0923"
ALL = ("I_next", "I_C", "I_rev", "A_C", "I_same", "A_own", "A_same")
DIAG2B = {"I_C": "I_C/p2b", "I_rev": "I_rev/p2b", "A_C": "A_C/p2b", "I_same": "I_same/p2b",
          "A_own": "A_own/p2b", "A_same": "A_same"}
LAG = ("I_C", "A_C", "A_own", "I_same")


def slope_tau(y, x):
    ok = np.isfinite(y) & (y > 0)
    if ok.sum() < 20:                  # too short a window for a slope (as in the pass-1 reading)
        return np.nan
    return -1.0 / np.polyfit(x[ok], np.log(y[ok]), 1)[0]


def med(rows, key):
    return float(np.nanmedian([r[key] for r in rows]))


def main() -> None:
    ps = pd.read_csv(OUT / "per_seed.csv")
    T: dict[str, dict] = {}
    for b in ALL:
        p = dict(np.load(OUT / b / "probe.npz"))
        rows = []
        for s in range(10):
            r0 = ps[(ps.bundle == b) & (ps.seed == s)].iloc[0]
            H = int(r0.hit999)
            kh, ke = (H + 500) // 100, int(r0.pre_end) // 100
            k0 = kh + 20
            if ke - k0 < 20:                   # the pre-burst window is too short to read
                continue
            x = p["step"].astype(float)
            L = p["resid_sum"][:, s] / 1200.0
            mg = p["margins"][:, s, :].astype(float)
            w = np.exp(-mg)
            act = (w * mg).sum(1) / w.sum(1)
            a, e = H + 2500, int(r0.pre_end)
            g = -2 * p["ps_dot_Wu"][a:e, s] + p["ps_uu"][a:e, s]
            if len(g) < 100:
                g = np.full(100, np.nan)
            srt = np.sort(g)[::-1]
            r = {"tau_sqrtv": slope_tau(p["sqrt_v_q"][k0:ke + 1, s, 2], x[k0:ke + 1]),
                 "tau_loss": slope_tau(L[k0:ke + 1], x[k0:ke + 1]),
                 "active_margin_per_1k": 1000 * np.polyfit(x[k0:ke + 1], act[k0:ke + 1], 1)[0],
                 "active_margin_start": act[kh], "median_margin_start": float(np.median(mg[kh])),
                 "L_start": L[kh], "L_end": L[ke], "efolds": float(np.log(L[kh] / L[ke])),
                 "pre_growth": float(r0.pre_growth), "per_step": float(g.mean()),
                 "per_step_median": float(np.median(g)), "frac_negative_steps": float((g < 0).mean()),
                 "top1pct_share": float(srt[:len(g) // 100].sum() / g.sum()),
                 "rms_u": float(r0.rms_u), "wnorm": float(r0.wnorm), "cos_1k": float(r0.cos_1k),
                 "sqrt_kappa_1k": float(r0.sqrt_kappa)}
            r["cost_per_efold"] = r["pre_growth"] / r["efolds"]
            rows.append(r)
        T[b] = {k: med(rows, k) for k in rows[0]}
    # the aim (pass 2 / 2b) and the batch noise
    for b, sub in DIAG2B.items():
        d = dict(np.load(OUT / sub / "diag.npz"))
        p = dict(np.load(OUT / b / "probe.npz"))
        rows = []
        for s in range(10):
            r0 = ps[(ps.bundle == b) & (ps.seed == s)].iloc[0]
            k0, k1 = (int(r0.hit999) + 2500) // 100, int(r0.pre_end) // 100
            if k1 - k0 < 20:
                continue
            sel = slice(k0, k1 + 1)
            nsr, pred, mratio = [], [], []
            for k in range(k0, k1 + 1):
                t = int(p["step"][k])
                G2 = d["W1_G"][k, s] ** 2
                n_after = p["ps_gg"][t:t + 50, s].mean() / G2 - 1
                sig2 = p["ps_gg"][max(t - 40, 0):t, s].mean() - G2
                nsr.append(n_after)
                pred.append(1 / np.sqrt(1 + n_after / 19))
                mratio.append(d["W1_m"][k, s] ** 2 / (G2 + sig2 * 0.1 / 1.9))
            rows.append({"cos_G_u": np.median(d["W1_Gu"][sel, s] / (d["W1_G"][sel, s] * d["W1_u"][sel, s])),
                         "cos_G_m": np.median(d["W1_cosGm"][sel, s]),
                         "cos_G_Gp": np.median(d["W1_cosGGp"][sel, s]),
                         "G_per_L": np.median(d["W1_G"][sel, s] / d["L"][sel, s]),
                         "C_img": np.median(d["W1_Cimg"][sel, s]),
                         "k_eff": np.median(d["k_eff"][sel, s]),
                         "batch_NSR": np.median(nsr), "cos_G_m_iid_pred": np.median(pred),
                         "m2_over_iid": np.median(mratio)})
        T[b].update({k: med(rows, k) for k in rows[0]})
    # turnover (pass 4)
    for b in LAG:
        z = dict(np.load(OUT / b / "lag" / "lag.npz"))
        p = dict(np.load(OUT / b / "probe.npz"))
        cs = {k: [] for k in range(1, 21)}
        rho = []
        for s in range(10):
            r0 = ps[(ps.bundle == b) & (ps.seed == s)].iloc[0]
            lo, hi = int(r0.hit999) + 2500, int(r0.pre_end)
            for i, t in enumerate(z["t0"]):
                if t < lo or t + 20 > hi:
                    continue
                for k in cs:
                    cs[k].append(z["cosG"][i, k - 1, s])
                c3, n3 = z["cosG"][i, 2, s], z["normratio"][i, 2, s]
                u = np.sqrt(p["ps_uu"][t:t + 3, s]).mean()
                rho.append(np.sqrt(max(1 + n3 ** 2 - 2 * n3 * c3, 0)) / 3 / u)
        lagc = np.array([1.0] + [np.median(cs[k]) for k in range(1, 21)])
        wts = 0.1 * 0.9 ** np.arange(21)
        T[b].update({"cos_G_lag1": lagc[1], "cos_G_lag10": lagc[10], "cos_G_lag20": lagc[20],
                     "staleness_factor": float(np.dot(wts / wts.sum(), lagc)),
                     "rel_grad_change_per_unit_step": float(np.median(rho))})
    # float32 (pass 5)
    for b in LAG:
        z = dict(np.load(OUT / b / "g32" / "g32.npz"))
        for i, t in enumerate(z["step"][:, 0]):
            T[b][f"cos_G32_G64@{int(t)}"] = float(np.median(z["cos_32_64"][i]))
            T[b][f"cos_G32_m@{int(t)}"] = float(np.median(z["cos32_m"][i]))
    (OUT / "posthoc_tables.json").write_text(json.dumps(T, indent=1, default=float))
    df = pd.DataFrame(T).T
    df.to_csv(OUT / "posthoc_tables.csv", float_format="%.6g")
    print(df.T.to_string(float_format=lambda v: f"{v:.4g}"))


if __name__ == "__main__":
    main()
