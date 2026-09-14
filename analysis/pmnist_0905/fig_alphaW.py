import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, pandas as pd, numpy as np, math, sys

d = pd.read_csv(sys.argv[1])
A = {"SN3": 3.0, "SN1": 1.0}
C = {"R": "#d1495b", "LR": "#2e86ab", "SN3": "#2a9d5c", "SN1": "#8b5cf6", "LIN": "#6b7280"}
N = {"R": "R (ReLU)", "LR": "LR (leaky .1)", "SN3": "SN3 (α=3)", "SN1": "SN1 (α=1)", "LIN": "LIN"}
fig, ax = plt.subplots(1, 4, figsize=(19, 4.3))
for arm in ("R", "LR", "SN3", "SN1", "LIN"):
    g = d[d.arm == arm].sort_values("task")
    if g.empty: continue
    c, s = C[arm], g.zsd_l1.rolling(5, min_periods=1).mean()
    ax[0].plot(g.task, g.acc.rolling(10, min_periods=1).mean(), color=c, lw=1.6, label=N[arm])
    ax[1].plot(g.task, s, color=c, lw=1.6)
    if arm in A:
        a = A[arm]
        ax[2].plot(g.task, 2 * a * s, color=c, lw=1.8, label=N[arm])
        ax[3].plot(g.task, np.exp(-2 * a * a * s ** 2), color=c, lw=1.8, label=f"{N[arm]} predicted")
        ax[3].plot(g.task, (g.mob_l1 - 1).abs().rolling(10, min_periods=1).mean(),
                   color=c, lw=1.2, ls="--", label=f"{N[arm]} |mob−1| measured")
ax[2].axhline(2 * math.pi, color="k", lw=1.2, ls="--")
ax[2].text(6, 2 * math.pi * 1.04, "2αW = 2π — one full period: the gate is fully averaged out",
           fontsize=8.5)
ax[0].set(xlabel="task", ylabel="test accuracy", title="accuracy")
ax[1].set(xlabel="task", ylabel="W = median$_i$ sd$_x[z_i]$",
          title="preactivation spread, layer 1\n(grows ~2.4× in every arm)")
ax[2].set(xlabel="task", ylabel="2αW  (radians per 1 sd)",
          title="how far the Snake phase travels\nacross a unit's own input range")
ax[3].set(xlabel="task", yscale="log", ylabel="surviving gate",
          title="effective gate\nsolid: exp(−2α²W²)   dashed: |mobility − 1|")
for a_, l in ((ax[0], "lower left"), (ax[2], "center right"), (ax[3], "lower left")):
    a_.legend(fontsize=8, loc=l)
for a_ in ax: a_.grid(alpha=.25)
fig.suptitle("What happens to Snake α=3 — lr 0.05, seed 0, layer 1. "
             "α=3 crosses its own averaging threshold during the run; α=1 does not. "
             "(1 seed, REPORT_ONLY)", y=1.04)
fig.tight_layout(); fig.savefig(sys.argv[2], dpi=130, bbox_inches="tight"); print("wrote", sys.argv[2])
