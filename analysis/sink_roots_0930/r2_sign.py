#!/usr/bin/env python3
"""Round 2 §3 leaky_C2_intervention and §4 df_test (spec_sink_roots_0930_round2.md): summaries of the saved outputs."""
import glob, json
from pathlib import Path
import numpy as np

R2 = Path("/home/issan/Projects/obsidian-research-data/sink_roots_0930/round2")
RES = Path(__file__).resolve().parents[2] / "results" / "sink_roots_0930"


def leaky():
    L = ["## leaky_C2_intervention: alive units with S < 0 (up), per a'  [16 random sets: mean (min..max) over sets; actual y_old; norefit]",
         "state a' | up 16 sets | up actual | up norefit | Mc<0 16 sets | up units with Cov_n(K,phi)<0 | pi+ median of up units | pi+ median of down units | |C2|/|C1| median | refit acc"]
    per_ap = {}
    for f in sorted(glob.glob(str(R2 / "leakyC2" / "c2int_*.json"))):
        rec = json.loads(Path(f).read_text())
        for ap in sorted({r["aprime"] for r in rec["rows"]}):
            rs = [r for r in rec["rows"] if r["aprime"] == ap]
            rr = [r for r in rs if r["set"].startswith("r")]; ra = [r for r in rs if r["set"] == "actual"][0]; rn = [r for r in rs if r["set"] == "norefit"][0]
            up = [r["up"] for r in rr]
            per_ap.setdefault(ap, []).append(dict(up=np.mean(up), cov=np.nanmean([r["up_cov_neg"] for r in rr]),
                                                  pip=np.nanmedian([r["up_pip_med"] for r in rr]), up_actual=ra["up"]))
            L.append(f"{rec['state']} {ap:<5g} | {np.mean(up):.3f} ({min(up):.3f}..{max(up):.3f}) | {ra['up']:.3f} | {rn['up']:.3f} | "
                     f"{np.mean([r['up_Mc'] for r in rr]):.3f} | {np.nanmean([r['up_cov_neg'] for r in rr]):.3f} | {np.nanmedian([r['up_pip_med'] for r in rr]):.4f} | "
                     f"{np.nanmedian([r['down_pip_med'] for r in rr]):.4f} | {np.median([r['C2_over_C1_med'] for r in rr]):.3f} | {np.mean([r['acc'] for r in rr]):.2f}")
    if per_ap:
        aps = sorted(per_ap)
        m = {ap: float(np.mean([x["up"] for x in per_ap[ap]])) for ap in aps}
        pip = {ap: float(np.median([x["pip"] for x in per_ap[ap]])) for ap in aps}
        cov = {ap: float(np.nanmean([x["cov"] for x in per_ap[ap]])) for ap in aps}
        L.append("\nover the 4 states: up share " + ", ".join(f"a'={ap:g}: {m[ap]:.3f}" for ap in aps)
                 + " | monotone " + str(all(m[aps[i]] < m[aps[i + 1]] for i in range(len(aps) - 1))))
        L.append("  up units with Cov<0: " + ", ".join(f"{ap:g}: {cov[ap]:.3f}" for ap in aps))
        L.append("  pi+ median of up units: " + ", ".join(f"{ap:g}: {pip[ap]:.4f}" for ap in aps)
                 + " | ratios " + ", ".join(f"{aps[i+1]:g}/{aps[i]:g}: {pip[aps[i+1]] / pip[aps[i]]:.2f} (a'^2 ratio {(aps[i+1] / aps[i]) ** 2:.0f})" for i in range(len(aps) - 1)))
    return L


def df():
    L = ["\n## df_test: R 12 draws; units alive in all draws; E[S] > 0 share; |S_closed|/|S_open| median over unit-draws (S and S^oh)",
         "state arm | alive | acc | E[S]>0 | frac S>0 (unit-draws) | S^oh>0 | cl/op S | cl/op S^oh"]
    for f in sorted(glob.glob(str(R2 / "df" / "df_*.npz"))):
        d = np.load(f); S, Soh, So, Soho, al = d["S"], d["Soh"], d["S_open"], d["Soh_open"], d["alive"].all(0)
        name = Path(f).stem[3:]
        clop = np.median(np.abs(S[:, al] - So[:, al]) / (np.abs(So[:, al]) + 1e-30))
        clop_oh = np.median(np.abs(Soh[:, al] - Soho[:, al]) / (np.abs(Soho[:, al]) + 1e-30))
        L.append(f"{name:28s} | {al.sum():3d} | {d['acc'].mean():.2f} | {np.mean(S[:, al].mean(0) > 0):.2f} | {np.mean(S[:, al] > 0):.3f} | "
                 f"{np.mean(Soh[:, al] > 0):.3f} | {clop:.3f} | {clop_oh:.3f}")
    return L


if __name__ == "__main__":
    txt = "\n".join(leaky() + df())
    print(txt)
    (RES / "round2_sign.txt").write_text(txt + "\n")
