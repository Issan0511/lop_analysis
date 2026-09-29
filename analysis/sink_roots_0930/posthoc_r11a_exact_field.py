#!/usr/bin/env python3
"""R11a post hoc (not registered): the exact SGD-form expected push field under label retention,
S_i(tau) = S_cancel_i - tau * A_old_i at the switch (E[y_new] = tau*onehot(y_old) + (1-tau)*uniform),
against the realized push P_i = m_i(200) - m_i(0).  S > 0 predicts a downward push."""
import csv
from pathlib import Path
import numpy as np

RAW = Path("/home/issan/Projects/obsidian-research-data/sink_roots_0930/mnist")
RES = Path(__file__).resolve().parents[2] / "results" / "sink_roots_0930"
rows = []
for act in ["ELU", "GELU", "SILU", "LR"]:
    for tau in [0.0, 0.25, 0.5, 0.75, 0.9]:
        agree, Sall, down, pdn, gam = [], [], [], [], []
        for s in [0, 1, 2]:
            n = f"R3_base_{act}_s{s}" if tau == 0 else f"R11a_tau{tau}_{act}_s{s}"
            a = np.load(RAW / n / "arrays.npz"); g = list(a["grid"]); i0, i200 = g.index(0), g.index(200)
            m, k = a["u_m1"], a["u_k1"]
            off = m.shape[0] - a["u_A_old"].shape[0]
            for t in range(5, 31):
                ti = t - 1
                sel = k[ti, i0] > 0
                S = a["u_S_cancel"][ti, i0] - tau * a["u_A_old"][ti - off, i0]
                P = m[ti, i200] - m[ti, i0]
                agree += list((np.sign(P) == -np.sign(S))[sel]); Sall += list(S[sel])
                down += list((S > 0)[sel]); pdn += list((P < 0)[sel])
                gam.append((float(a["s_p_old"][ti, i0]) - 0.1) / 0.9)
        rows.append({"act": act, "tau": tau, "sign_agree": float(np.mean(agree)), "S_median": float(np.median(Sall)),
                     "frac_S_pos": float(np.mean(down)), "frac_P_neg": float(np.mean(pdn)),
                     "gamma_hat_median": float(np.median(gam))})
with open(RES / "R11a_posthoc_exact_field.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for r in rows:
    print(r)
