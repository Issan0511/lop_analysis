"""Constant-mode split of the c1 push for every CNN arm (post hoc; Codex round 2 eq. (25), round 3 request).

G_j = sum_{n,q} gamma_j K1_j H1_j  =  kappa_j * Hbar1_j  +  sum (gamma K1 - kappa_j/M)(H1_j - Hbar1_j)
     =: G_const_j + G_cen_j,   kappa_j = sum gamma K1 = M * gK1_mean_win_j,  Hbar1_j = H1sum_all_j / M,
so G_const_j = gK1_mean_win_j * H1sum_all_j exactly (both saved per state by src/c1_direction_1010.py; the full W2
including its initial value enters H1).  Also the same split restricted to the learned DC bucket and to W2^0.
Sign convention: G > 0 = sinking side.  Writes results/c1_direction_1010/cnn_tables/constmode_cnn.{csv,md}.

usage: .venv/bin/python analysis/c1_direction_1010/constmode_cnn.py
"""
import glob, math, os, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "c1_direction_1010" / "cnn_tables"
BK = Path(os.environ.get("C1_DATA_ROOT", str(Path.home() / "Projects" / "obsidian-research-data" / "c1_direction_1010")))
CNN = ROOT / "results" / "c1_direction_1010" / "cnn"
if not any(CNN.glob("*.npz")):
    CNN = BK / "results" / "c1_direction_1010" / "cnn"
ARMS = ["LR", "LRc", "SNA", "SNAc3", "CV06FC3", "CV3FC06"]
T975 = {9: 2.262}


def ci(v):
    v = np.asarray(v, float)
    if v.size < 2:
        return (float(v.mean()) if v.size else float("nan")), float("nan"), float("nan")
    se = v.std(ddof=1) / math.sqrt(v.size)
    return float(v.mean()), float(v.mean() - 2.262 * se), float(v.mean() + 2.262 * se)


def main():
    rows = defaultdict(list)
    for f in sorted(glob.glob(str(CNN / "*.npz"))):
        d = np.load(f, allow_pickle=True)
        if str(d["loss"]) != "L_u":
            continue
        arm, seed, t = str(d["arm"]), int(d["seed"]), int(d["task"])
        G = d["G_c1_full"]; gk = d["gK1_mean_win"]
        Gc = gk * d["H1sum_all"]; Gcen = G - Gc
        GcDC = gk * d["H1sum_DC"]; GcInit = gk * d["H1sum_init"]; GcAC = gk * d["H1sum_AC"]
        r = dict(
            const_float=float((Gc < 0).mean()), cen_float=float((Gcen < 0).mean()),
            agree_const_G=float((np.sign(Gc) == np.sign(G)).mean()), agree_cen_G=float((np.sign(Gcen) == np.sign(G)).mean()),
            dom_const=float((np.abs(Gc) > np.abs(Gcen)).mean()),
            share_const_med=float(np.median(np.abs(Gc) / (np.abs(Gc) + np.abs(Gcen)))),
            constDC_float=float((GcDC < 0).mean()), agree_constDC_G=float((np.sign(GcDC) == np.sign(G)).mean()),
            agree_constInit_G=float((np.sign(GcInit) == np.sign(G)).mean()),
            share_constAC_in_const_med=float(np.median(np.abs(GcAC) / (np.abs(GcDC) + np.abs(GcAC) + np.abs(GcInit)))),
            sink=float((G > 0).mean()))
        rows[(arm, t)].append((seed, r))
    keys = ["sink", "const_float", "agree_const_G", "dom_const", "share_const_med", "cen_float", "agree_cen_G",
            "constDC_float", "agree_constDC_G", "agree_constInit_G", "share_constAC_in_const_med"]
    lines = ["arm,t,n_seed," + ",".join(f"{k}_mean,{k}_lo,{k}_hi" for k in keys)]
    md = ["# c1 constant-mode split, every CNN arm (post hoc; G > 0 = sinking; seed mean [95% t-CI])", "",
          "G_const_j = (mean over winners of gamma*K1) * (sum over all (n,q) of H1_j) with the FULL W2 (init included); "
          "G_cen = G - G_const.  'constDC' / 'constInit' = the same constant mode restricted to the learned DC bucket / to W2^0.", ""]
    for arm in ARMS:
        ts = sorted(t for (a, t) in rows if a == arm)
        if not ts:
            continue
        md += [f"## {arm}", "", "| quantity | " + " | ".join(f"t{t}" for t in ts) + " |", "|---|" + "---|" * len(ts)]
        cells = {k: [] for k in keys}
        for t in ts:
            per = [r for _, r in rows[(arm, t)]]
            vals = []
            for k in keys:
                m, lo, hi = ci([r[k] for r in per])
                cells[k].append(f"{m:.2f} [{lo:.2f}, {hi:.2f}]")
                vals += [f"{m:.4f}", f"{lo:.4f}", f"{hi:.4f}"]
            lines.append(f"{arm},{t},{len(per)}," + ",".join(vals))
        for k in keys:
            md.append(f"| {k} | " + " | ".join(cells[k]) + " |")
        md.append("")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "constmode_cnn.csv").write_text("\n".join(lines) + "\n")
    (OUT / "constmode_cnn.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
