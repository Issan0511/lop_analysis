#!/usr/bin/env python3
"""Survivor quantities of the parent's a1 (return/final/checks/a1_width_across_arms.py) for the round-1 / 1b engine arms
(R3 base, R6 ink, R1 cap, 1b caps): tasks 5-30, units open at the switch; survivors = still open at the task end.
ratio = -mean R / mean P (push 0 -> 200 updates, return 200 -> end); kds/|P| = mean(kappa0 * ds_task) / |mean P| with
kappa0 = (top0 - m0) / s0; ds/|P| = mean ds_task / |mean P|; closing share = closed at the end / open at the switch."""
import json
from pathlib import Path
import numpy as np

RAW = Path("/home/issan/Projects/obsidian-research-data/sink_roots_0930/mnist")
RES = Path(__file__).resolve().parents[2] / "results" / "sink_roots_0930"


def agg(name):
    a = np.load(RAW / name / "arrays.npz"); g = list(a["grid"]); i0, i2, iT = g.index(0), g.index(200), len(g) - 1
    m, top, s, k = a["u_m1"], a["u_top1"], a["u_s1"], a["u_k1"]
    sl = slice(4, 30)
    m0, m2, mT = m[sl, i0], m[sl, i2], m[sl, iT]; t0 = top[sl, i0]; s0, sT = s[sl, i0], s[sl, iT]
    k0, kT = k[sl, i0], k[sl, iT]
    op = k0 > 0; sv = op & (kT > 0)
    kap0 = (t0 - m0) / s0; P = (m2 - m0)[sv].mean(); R = (mT - m2)[sv].mean()
    return dict(ratio_surv=float(-R / P), kds_over_P=float((kap0 * (sT - s0))[sv].mean() / abs(P)),
                ds_over_P=float((sT - s0)[sv].mean() / abs(P)), close_share=float((op & (kT == 0)).sum() / op.sum()))


def main():
    out = {}; lines = ["run group | survivors ratio | kappa0 ds_task / |P| | ds_task / |P| | closing share  (seed 0/1/2)"]
    for tag, acts in (("R3_base", ("ELU", "LR", "SILU", "GELU")), ("R6ink", ("ELU", "LR", "SILU")), ("R1wcap", ("ELU", "SILU")),
                      ("R1b1_wcap", ("GELU", "LR")), ("R1b3_par", ("ELU", "SILU")), ("R1b3_perp", ("ELU", "SILU")),
                      ("R1b4_scale0.5", ("ELU",)), ("R1b4_scale2", ("ELU",))):
        for act in acts:
            rows = []
            for s in (0, 1, 2):
                n = f"{tag}_{act}_s{s}"
                if (RAW / n / "provenance.json").exists():
                    out[n] = agg(n); rows.append(out[n])
            if rows:
                f = lambda k_: "/".join(f"{r[k_]:.3f}" for r in rows)
                lines.append(f"{tag:13s} {act:4s} | {f('ratio_surv')} | {f('kds_over_P')} | {f('ds_over_P')} | {f('close_share')}")
    txt = "\n".join(lines); print(txt)
    (RES / "round1_survivors.txt").write_text(txt + "\n")
    (RES / "round1_survivors.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
