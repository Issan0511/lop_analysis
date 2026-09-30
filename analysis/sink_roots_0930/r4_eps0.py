#!/usr/bin/env python3
"""Round 4 R_eps_zero_v2 (spec_sink_roots_0930_round4.md §2): v4 box, eps1 = 1e-14 (float32), GELU / SiLU x seeds 0, 1 x 200 tasks.
Closed = top at the task end <= 0; mu+ / mu- = closing / reopening rates between consecutive task ends in a window.
Adam diagnostics per task (the first layer): the most coordinates with v = 0 and m != 0 at any update, and the largest
|delta theta| of one update.  Round 1's eps1 1e-6 / 1e-8 / 1e-12 runs are listed alongside for the same box."""
import glob, json, re
from pathlib import Path
import numpy as np

RAW = Path("/home/issan/Projects/obsidian-research-data")
RES = Path(__file__).resolve().parents[2] / "results" / "sink_roots_0930"


def rates(top, lo, hi):
    oc = oo = co = cc = 0
    for t in range(lo - 1, min(hi, top.shape[0] - 1)):
        x, y = top[t] > 0, top[t + 1] > 0
        oc += (x & ~y).sum(); oo += x.sum(); co += (~x & y).sum(); cc += (~x).sum()
    return oc / max(oo, 1), co / max(cc, 1)


def main():
    files = {}
    for f in glob.glob(str(RAW / "sink_roots_0930" / "round4" / "R_eps_zero_v2" / "*_v4.npz")):
        act, seed = re.match(r"(\w+?)_eps1e-14_s(\d)_v4", Path(f).stem).groups(); files[(act, "1e-14", int(seed))] = f
    for f in glob.glob(str(RAW / "sink_roots_0930" / "round1" / "R_eps_wall" / "*_v4.npz")):
        m = re.match(r"(\w+?)_eps([\d.e-]+)_s(\d)_v4", Path(f).stem)
        if m:
            files[(m.group(1), m.group(2), int(m.group(3)))] = f
    for act in ("GELU", "SILU"):
        for s in (0, 1):
            f = RAW / "elu_alpha_ladder_1layer_0925" / "runs" / f"{act}_s{s}_v4.npz"
            if f.exists():
                files[(act, "1e-8", s)] = str(f)
    L, J = [], {}
    for key in sorted(files, key=lambda k: (k[0], float(k[1]), k[2])):
        d = np.load(files[key]); top = d["task_zmax"].astype(float)
        ac = lambda lo, hi: float((top[lo - 1:hi] <= 0).mean())
        mp1, mm1 = rates(top, 51, 100); mp2, mm2 = rates(top, 151, 200)
        cl = top <= 0
        row = dict(min_closed_top=float(top[cl].min()) if cl.any() else float("nan"), allclosed_51_100=ac(51, 100), allclosed_151_200=ac(151, 200),
                   mu_minus_51_100=float(mm1), mu_minus_151_200=float(mm2), mu_plus_151_200=float(mp2))
        if "diag_max_abs_dtheta" in d.files:
            row.update(max_abs_dtheta=float(d["diag_max_abs_dtheta"].max()), v0_m_nonzero_max=int(d["diag_v0_m_nonzero_max"].max()))
        J["|".join(map(str, key))] = row
        L.append(f"{key[0]:4s} eps1 {key[1]:6s} s{key[2]}: lowest closed top {row['min_closed_top']:.2f} | all-closed 51-100 {row['allclosed_51_100']:.3f} "
                 f"151-200 {row['allclosed_151_200']:.3f} (diff {row['allclosed_151_200'] - row['allclosed_51_100']:+.3f}) | mu- 51-100 {mm1:.3f} "
                 f"151-200 {mm2:.3f} (change {mm2 - mm1:+.3f}) | mu+ 151-200 {mp2:.3f}"
                 + (f" | max |dtheta| {row['max_abs_dtheta']:.2e}, coords with v=0, m!=0 (max) {row['v0_m_nonzero_max']}" if "max_abs_dtheta" in row else ""))
    txt = "\n".join(L); print(txt)
    (RES / "round4_R_eps_zero_v2.txt").write_text(txt + "\n")
    (RES / "round4_R_eps_zero_v2.json").write_text(json.dumps(J, indent=1))


if __name__ == "__main__":
    main()
