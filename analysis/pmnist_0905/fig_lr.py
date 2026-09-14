import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd, numpy as np, sys

df = pd.concat([pd.read_csv(p) for p in sys.argv[1:-1]])
lrs = sorted(df.lr.unique())
cmap = plt.get_cmap("viridis")
cols = {lr: cmap(i / (len(lrs) - 1)) for i, lr in enumerate(lrs)}

fig, ax = plt.subplots(1, 4, figsize=(19, 4.2))
for lr in lrs:
    g = df[df.lr == lr].groupby("task").median(numeric_only=True).sort_index()
    r = g.acc.rolling(10, min_periods=1).mean()
    c = cols[lr]
    ax[0].plot(g.index, r, color=c, lw=1.6, label=f"lr={lr:g}")
    ax[1].plot(g.index, g.mob_l1, color=c, lw=1.6)
    ax[2].plot(g.index, g.w_norm_l1 / g.w_norm_l1.iloc[0], color=c, lw=1.6)
    ax[3].plot(g.index, g.zbar_l1, color=c, lw=1.6)
    j = int(np.argmax(r.values))
    ax[0].plot(g.index[j], r.values[j], "o", color=c, ms=5, mec="k", mew=.6, zorder=5)

ax[0].set(xlabel="task", ylabel="test accuracy (10-task mean)",
          title="accuracy — dot marks the peak")
ax[1].set(xlabel="task", ylabel=r"median$_i$ E$_{test}[\varphi'(z_i)]$", yscale="log",
          title="mobility, layer 1 (the gate)")
ax[2].set(xlabel="task", ylabel="w_norm(t) / w_norm(1)", title="weight norm growth, layer 1")
ax[3].set(xlabel="task", ylabel=r"median$_i$ E$_{test}[z_i]$", title="preactivation mean, layer 1")
ax[3].axhline(0, color="k", lw=.7, ls=":")
ax[0].legend(fontsize=8, ncol=2, loc="lower right")
for a in ax:
    a.grid(alpha=.25)
fig.suptitle("Permuted MNIST, ReLU (arm R), 784-100-100-10, plain SGD, batch 16, "
             "625 steps/task — 3 seeds, median", y=1.02)
fig.tight_layout()
fig.savefig(sys.argv[-1], dpi=130, bbox_inches="tight")
print("wrote", sys.argv[-1])
