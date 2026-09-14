import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, pandas as pd, numpy as np, glob, sys

df = pd.concat([pd.read_csv(f) for f in glob.glob(f"{sys.argv[1]}/**/per_task.csv", recursive=True)])
df = df[df.acc.notna()]
ARMS = ["R", "LR", "SN3", "SN1", "LIN"]
C = {"R": "#d1495b", "LR": "#2e86ab", "SN3": "#2a9d5c", "SN1": "#8b5cf6", "LIN": "#6b7280"}
NAME = {"R": "R (ReLU)", "LR": "LR (leaky .1)", "SN3": "SN3 (Snake α=3)",
        "SN1": "SN1 (Snake α=1)", "LIN": "LIN (identity)"}
lrs = sorted(df.lr.unique())

fig = plt.figure(figsize=(19, 4.6 * len(lrs) / 2))
gs = fig.add_gridspec(len(lrs), 4, hspace=.42, wspace=.26)
for i, lr in enumerate(lrs):
    ax = [fig.add_subplot(gs[i, j]) for j in range(4)]
    for arm in ARMS:
        g = df[(df.lr == lr) & (df.arm == arm)]
        if g.empty: continue
        g = g.groupby("task").median(numeric_only=True).sort_index()
        c = C[arm]
        ax[0].plot(g.index, g.acc.rolling(10, min_periods=1).mean(), color=c, lw=1.6, label=NAME[arm])
        ax[1].plot(g.index, g.mob_l1.clip(lower=1e-4), color=c, lw=1.6)
        ax[2].plot(g.index, g.zbar_l1, color=c, lw=1.6)
        ax[3].plot(g.index, g.w_norm_l1 / g.w_norm_l1.iloc[0], color=c, lw=1.6)
    ax[0].axvspan(151, 200, color="k", alpha=.06)
    ax[0].set(xlabel="task", ylabel="test acc", title=f"lr = {lr:g} — accuracy (shaded = window 151-200)")
    ax[1].set(xlabel="task", ylabel=r"median$_i$E$[\varphi']$", yscale="log", title="mobility, layer 1")
    ax[2].set(xlabel="task", ylabel=r"median$_i$E$[z_i]$", title="preactivation mean, layer 1")
    ax[3].set(xlabel="task", ylabel="ratio to task 1", title="weight norm, layer 1")
    ax[2].axhline(0, color="k", lw=.7, ls=":")
    if i == 0: ax[0].legend(fontsize=8, loc="lower left", ncol=2)
    for x in ax: x.grid(alpha=.25)
fig.suptitle("pmnist_main_0905 — 5 arms x lr surface, 10 seeds, median. "
             "784-100-100-10, plain SGD, batch 16, 200 tasks", y=.995)
fig.savefig(sys.argv[2], dpi=115, bbox_inches="tight"); print("wrote", sys.argv[2])
