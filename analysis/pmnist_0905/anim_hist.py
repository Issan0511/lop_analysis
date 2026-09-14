import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np, sys
from pathlib import Path

SRC = Path("results/_diag_hist_0905")
ARMS = ["R", "LR", "SN3", "SN1", "LIN"]
C = {"R": "#d1495b", "LR": "#2e86ab", "SN3": "#2a9d5c", "SN1": "#8b5cf6", "LIN": "#6b7280"}
N = {"R": "R  ReLU", "LR": "LR  leaky 0.1", "SN3": "SN3  Snake α=3",
     "SN1": "SN1  Snake α=1", "LIN": "LIN  identity"}
D = {a: np.load(SRC / f"{a}.npz") for a in ARMS}
E = D["R"]["edges"]; ctr = (E[:-1] + E[1:]) / 2
T = D["R"]["h1"].shape[0]
zz = np.linspace(E[0], E[-1], 1200)


def dphi(a, z):
    if a == "R":   return (z > 0).astype(float)
    if a == "LR":  return np.where(z > 0, 1.0, 0.1)
    if a == "SN3": return 1 + np.sin(6 * z)
    if a == "SN1": return 1 + np.sin(2 * z)
    return np.ones_like(z)


fig, ax = plt.subplots(2, 5, figsize=(19.5, 7.4), sharex=True)
fig.subplots_adjust(top=.84, hspace=.13, wspace=.13, left=.04, right=.985, bottom=.09)
art = {}
for j, a in enumerate(ARMS):
    for i, (hk, mk, lab) in enumerate(((("h1"), "m1", "layer 1"), (("h2"), "m2", "layer 2"))):
        A = ax[i, j]
        g = A.twinx(); g.plot(zz, dphi(a, zz), color="#c9a227", lw=1.1, alpha=.85)
        g.set_ylim(-0.15, 2.3); g.set_yticks([])
        if j == 4: g.set_ylabel("φ'(z)", color="#a8861f", fontsize=8.5)
        pooled, = A.plot([], [], color=C[a], lw=1.0, alpha=.45)
        fill = A.fill_between(ctr, 0, 0, color=C[a], alpha=.16)
        units = A.stem([0], [0])[0] if False else None
        bars = A.bar(ctr, np.zeros_like(ctr), width=(E[1]-E[0]), color=C[a], alpha=.85)
        A.axvline(0, color="k", lw=.7, ls=":")
        A.set_xlim(E[0], E[-1]); A.set_ylim(0, 1.05)
        A.set_yticks([])
        if i == 0: A.set_title(N[a], fontsize=11, color=C[a], pad=7)
        if i == 1: A.set_xlabel("preactivation  z", fontsize=9)
        if j == 0: A.set_ylabel(lab, fontsize=10)
        art[(a, i)] = (pooled, bars)
txt = fig.text(.5, .935, "", ha="center", fontsize=13)
fig.text(.5, .895, "filled bars = distribution of the 100 per-unit MEANS  z̄ᵢ    "
         "faint line = pooled z over test set × units    gold = φ'(z), the gate",
         ha="center", fontsize=9, color="#444")
fig.text(.5, .022, "Permuted MNIST, Adam lr=0.001, seed 0, 784-100-100-10 — "
         "UNREGISTERED diagnostic (results/_diag_hist_0905)", ha="center", fontsize=8.5, color="#666")


def frame(t):
    for a in ARMS:
        for i, (hk, mk) in enumerate((("h1", "m1"), ("h2", "m2"))):
            pooled, bars = art[(a, i)]
            h = D[a][hk][t].astype(float); h = h / max(h.max(), 1)
            pooled.set_data(ctr, h)
            mh, _ = np.histogram(D[a][mk][t], bins=E)
            mh = mh.astype(float) / max(mh.max(), 1)
            for b, v in zip(bars, mh): b.set_height(v)
    txt.set_text(f"task {t+1} / {T}      "
                 + "      ".join(f"{a} {D[a]['acc'][t]:.3f}" for a in ARMS))
    return []


anim = FuncAnimation(fig, frame, frames=range(0, T, 2), interval=110, blit=False)
out = sys.argv[1] if len(sys.argv) > 1 else "results/_diag_hist_0905/preact.gif"
anim.save(out, writer=PillowWriter(fps=9), dpi=72)
print("wrote", out)
