import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, pandas as pd, numpy as np, glob, sys

df = pd.concat([pd.read_csv(f) for f in glob.glob(f"{sys.argv[1]}/**/per_task.csv", recursive=True)])
df = df[df.acc.notna()]
C = {"R": "#d1495b", "LR": "#2e86ab", "LIN": "#6b7280"}
NAME = {"R": "R  (ReLU)", "LR": "LR  (leaky 0.1)", "LIN": "LIN  (identity)"}
lrs = sorted(df.lr.unique())
fig, ax = plt.subplots(2, 4, figsize=(19, 8))
for row, lr in enumerate(lrs):
    for arm in ("R", "LR", "LIN"):
        g = df[(df.lr == lr) & (df.arm == arm)].groupby("task").median(numeric_only=True).sort_index()
        r = g.acc.rolling(10, min_periods=1).mean()
        a, c = ax[row], C[arm]
        a[0].plot(g.index, r, color=c, lw=1.7, label=NAME[arm])
        j = int(np.argmax(r.values)); a[0].plot(g.index[j], r.values[j], "o", color=c, ms=5, mec="k", mew=.6, zorder=5)
        a[1].plot(g.index, g.mob_l1, color=c, lw=1.7)
        a[2].plot(g.index, g.zbar_l1, color=c, lw=1.7)
        a[3].plot(g.index, g.dead_frac_l1, color=c, lw=1.7)
    a = ax[row]
    a[0].set(xlabel="task", ylabel="test acc (10-task mean)", title=f"lr = {lr:g} — accuracy")
    a[1].set(xlabel="task", ylabel=r"median$_i$ E$[\varphi'(z_i)]$", yscale="log", title="mobility, layer 1")
    a[2].set(xlabel="task", ylabel=r"median$_i$ E$[z_i]$", title="preactivation mean, layer 1")
    a[3].set(xlabel="task", ylabel="dead fraction", title=r"dead units, layer 1  ($|\varphi'|<10^{-6}$)")
    a[2].axhline(0, color="k", lw=.7, ls=":")
    a[0].legend(fontsize=9, loc="lower left")
    for x in a: x.grid(alpha=.25)
fig.suptitle("pmnist_probe_0905 — stage 0, 10 seeds, median. 784-100-100-10, plain SGD, batch 16, 200 tasks", y=1.0)
fig.tight_layout(); fig.savefig(sys.argv[2], dpi=125, bbox_inches="tight"); print("wrote", sys.argv[2])
