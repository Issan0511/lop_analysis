#!/usr/bin/env python3
"""quiet_drift_cifar_0923 pass 3 -- score E1-E3 of PREDICTIONS_pass3.md (β2 switched at 6,000).

Window per slot: step 6,000 -> the first burst episode's first interval after 6,000 (a run of
probes with correct < 1199 whose minimum is < 1190) or 30,000.  L = mean residual (float64 from
the probe's resid_sum).  Baselines are pass 1's I_C and A_own over the same window.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "quiet_drift_cifar_0923"
SW = 6000
RUNS = {("I_C", 0.999): "I_C", ("I_C", 0.99): "I_C_b99", ("I_C", 0.9999): "I_C_b9999",
        ("A_own", 0.999): "A_own", ("A_own", 0.99): "A_own_b99", ("A_own", 0.9999): "A_own_b9999"}
NEED, EP_MIN = 1199, 1190


def window(corr: np.ndarray, k0: int) -> int:
    """Last probe index of the window: the probe before the first burst episode's first interval."""
    k, K = k0 + 1, len(corr)
    while k < K:
        if corr[k] < NEED:
            j = k
            while j + 1 < K and corr[j + 1] < NEED:
                j += 1
            if corr[k:j + 1].min() < EP_MIN:
                return k - 1
            k = j + 1
        else:
            k += 1
    return K - 1


def main() -> None:
    rows = []
    for (base, b2), name in RUNS.items():
        f = OUT / name / "probe.npz"
        if not f.exists():
            raise SystemExit(f"missing {f}")
        with np.load(f) as z:
            p = {k: z[k] for k in ("step", "correct", "n1", "resid_sum", "sqrt_v_q", "ratio_mean",
                                   "ps_uu", "cos_SuW", "kappa_u1k", "margin_med32")}
        k0 = SW // 100
        for s in range(10):
            ke = window(p["correct"][:, s], k0)
            L = p["resid_sum"][:, s] / 1200.0
            n1 = p["n1"][:, s]
            ln = int(p["step"][ke] - SW)
            ef = float(np.log(L[k0] / L[ke])) if ke > k0 else np.nan
            gr = float(n1[ke] - n1[k0])
            sv = p["sqrt_v_q"][k0:ke + 1, s, 2]
            x = p["step"][k0:ke + 1].astype(float)
            ok = sv > 0
            tau_v = -1 / np.polyfit(x[ok], np.log(sv[ok]), 1)[0] if ok.sum() > 3 else np.nan
            rows.append({"base": base, "beta2": b2, "run": name, "seed": s,
                         "end": int(p["step"][ke]), "len": ln, "efolds": ef,
                         "tau_L": ln / ef if ef and ef > 0 else np.nan,
                         "growth": gr, "per_efold": gr / ef if ef and ef > 0 else np.nan,
                         "per_step": gr / ln if ln > 0 else np.nan, "tau_sqrtv": tau_v,
                         "rms_u": float(np.sqrt(p["ps_uu"][SW:SW + max(ln, 1), s].mean())),
                         "ratio_mean": float(np.median(p["ratio_mean"][k0:ke + 1, s])),
                         "margin_med_end": float(p["margin_med32"][ke, s])})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "pass3_per_seed.csv", index=False, float_format="%.8g")
    med = df.groupby(["base", "beta2"]).median(numeric_only=True)
    print(med.drop(columns="seed").to_string(float_format=lambda x: f"{x:.4g}"))
    r10 = np.sqrt(10)
    tau = {(b, q): float(med.loc[(b, q), "tau_L"]) for b in ("I_C", "A_own") for q in (0.99, 0.999, 0.9999)}
    pe = {(b, q): float(med.loc[(b, q), "per_efold"]) for b in ("I_C", "A_own") for q in (0.99, 0.999, 0.9999)}
    e1 = all(tau[(b, 0.9999)] / tau[(b, 0.999)] > r10 and tau[(b, 0.99)] / tau[(b, 0.999)] < 1 / r10
             for b in ("I_C", "A_own"))
    e2 = all(1 / r10 < pe[(b, q)] / pe[(b, 0.999)] < r10 for b in ("I_C", "A_own") for q in (0.99, 0.9999))
    base_ratio = pe[("I_C", 0.999)] / pe[("A_own", 0.999)]
    e3 = all(1 / r10 < (pe[("I_C", q)] / pe[("A_own", q)]) / base_ratio < r10 for q in (0.99, 0.9999))
    v = {"tau_L": {f"{b}@{q}": x for (b, q), x in tau.items()},
         "per_efold": {f"{b}@{q}": x for (b, q), x in pe.items()},
         "tau_ratio_vs_0.999": {f"{b}@{q}": tau[(b, q)] / tau[(b, 0.999)] for (b, q) in tau},
         "per_efold_ratio_vs_0.999": {f"{b}@{q}": pe[(b, q)] / pe[(b, 0.999)] for (b, q) in pe},
         "I_C_over_A_own_per_efold": {str(q): pe[("I_C", q)] / pe[("A_own", q)]
                                      for q in (0.99, 0.999, 0.9999)},
         "E1": e1, "E2": e2, "E3": e3}
    pr = {"E1": 0.65, "E2": 0.50, "E3": 0.60}
    v["scores"] = {k: {"p": pr[k], "outcome": v[k], "brier": (pr[k] - float(v[k])) ** 2} for k in pr}
    (OUT / "pass3_verdict.json").write_text(json.dumps(v, indent=1, default=float))
    print(json.dumps({k: v[k] for k in ("tau_ratio_vs_0.999", "per_efold_ratio_vs_0.999",
                                        "I_C_over_A_own_per_efold", "E1", "E2", "E3")}, indent=1))


if __name__ == "__main__":
    main()
