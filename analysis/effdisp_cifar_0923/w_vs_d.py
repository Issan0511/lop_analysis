#!/usr/bin/env python3
"""POST HOC (0923 night).  How does the effective displacement ||D||_S depend on the effective weight ||W||_S?
(1) within each arm over tasks (t >= 6): per-seed log-log slope of ||D||_S on ||W||_S (and on the plain norm);
(2) across arms at the late window: ||W||_S vs ||D||_S and 1/(2|c_S|);
(3) crossover fork (same Adam state, weights before vs after the post-fit push): elasticity of the next refit's ||D||_S.
Inputs: sgd_postfit posthoc_c_eff.csv and eps_width per_task_eff.csv from origin/main, xfork snapshots from the archive."""
import os, subprocess, io
from pathlib import Path
import numpy as np, pandas as pd
REPO = Path(__file__).resolve().parents[2]
def show(p): return pd.read_csv(io.StringIO(subprocess.run(["git", "-C", str(REPO), "show", f"HEAD:{p}"], capture_output=True, text=True, check=True).stdout))
a = show("results/sgd_postfit_cifar_0923/posthoc_c_eff.csv"); a["W_S"] = np.sqrt(a.S_V); a["D_S"] = a.S_rho * a.W_S; a["W_N"] = np.sqrt(a.N_V); a["D_N"] = a.N_rho * a.W_N
e = show("results/eps_width_cifar_0923/per_task_eff.csv"); e["W_N"] = np.sqrt(e.n1)
L = []
def P(x): L.append(str(x)); print(x)
P("(1) within-arm log-log slopes, t>=6, median over seeds")
for arm in ("A", "F", "S_hi", "SA_iid"):
    d = a[(a.arm == arm) & (a.t >= 6)]
    f = lambda xc, yc: np.median([np.polyfit(np.log(x[xc]), np.log(x[yc]), 1)[0] for _, x in d.groupby("seed")])
    P(f"  {arm:7s} D_S vs W_S {f('W_S','D_S'):.2f} | plain D vs plain W {f('W_N','D_N'):.2f} | D_S vs plain W {f('W_N','D_S'):.2f}")
for arm in ("LL_e8", "LL_e6", "LL_e4", "LL_e3"):
    d = e[(e.arm == arm) & (e.t >= 6)]
    f = lambda xc, yc: np.median([np.polyfit(np.log(x[xc]), np.log(x[yc]), 1)[0] for _, x in d.groupby("seed")])
    P(f"  {arm:7s} D_S vs W_S {f('W_S','D_S'):.2f} | D_S vs plain W {f('W_N','D_S'):.2f}")
rows = []
for arm in ("SA_iid", "S_hi", "F", "A"):
    x = a[(a.arm == arm) & (a.t >= 41)].groupby("seed")[["W_S", "D_S", "W_N", "S_c", "S_rho"]].mean().median(); rows.append((arm, "t41-50", *x.values))
for arm in ("LL_e3", "LL_e4", "LL_e6", "LL_e8"):
    x = e[(e.arm == arm) & (e.t >= 11)].groupby("seed")[["W_S", "D_S", "W_N", "c_S"]].mean().median(); rows.append((arm, "t11-20", *x.values, np.nan))
t = pd.DataFrame(rows, columns=["arm", "window", "W_S", "D_S", "W_plain", "c_S", "rho_S"]); t["W_S/D_S"] = t.W_S / t.D_S; t["1/(2|c|)"] = 1 / (2 * t.c_S.abs()); t["2|c|"] = 2 * t.c_S.abs()
P("\n(2) across arms"); P(t.round(3).to_string(index=False))
for w in ("t41-50", "t11-20"):
    x = t[t.window == w]; P(f"  {w}: across-arm log-log slope of W_S on D_S {np.polyfit(np.log(x.D_S), np.log(x.W_S), 1)[0]:.2f}")
SP = os.path.expanduser("~/Projects/obsidian-research-data/sgd_postfit_cifar_0923/results/sgd_postfit_cifar_0923")
B = os.path.expanduser("~/Projects/obsidian-research-data/sgd_postfit_cifar_0923/basis_cache")
w1 = lambda p: np.load(p)["W1"].astype(np.float64); rows = []
for s in range(10):
    z = np.load(f"{B}/basis_std_s{s}.npz"); Q = z["Q"][:, :1199].astype(np.float64); lam = z["lam"].astype(np.float64)
    nS = lambda M: np.sqrt(float((lam * (M @ Q) ** 2).sum()))
    for tt in (10, 20, 30, 40, 48):
        sd = f"LR_std_seed{s}"; Wsw = w1(f"{SP}/xfork/t{tt}_sw/snap/{sd}/t01.npz"); Wend = w1(f"{SP}/A/snap/{sd}/t{tt:02d}.npz")
        Rp = w1(f"{SP}/xfork/t{tt}_Wend_Ssw/snap/{sd}/t01.npz") - Wend; R0 = w1(f"{SP}/xfork/t{tt}_Wsw_Ssw/snap/{sd}/t01.npz") - Wsw
        rows.append(dict(WS_ratio=nS(Wend) / nS(Wsw), DS_ratio=nS(Rp) / nS(R0)))
x = pd.DataFrame(rows); x["elast"] = np.log(x.DS_ratio) / np.log(x.WS_ratio)
P(f"\n(3) crossover fork (50 cases): W_S ratio {x.WS_ratio.median():.3f}, next-refit D_S ratio {x.DS_ratio.median():.3f} (>1 in {(x.DS_ratio>1).sum()}/50), elasticity {x.elast.median():.2f}")
(REPO / "results/effdisp_cifar_0923/w_vs_d.txt").write_text("\n".join(L) + "\n")
