import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np, sys
from pathlib import Path

SRC = Path("results/_diag_rlhist_0907")
ARMS = ["R", "LR", "SNA", "R_l2", "LR_l2", "SNA_l2", "R_l2init"]
C = {"R": "#d1495b", "LR": "#2e86ab", "SNA": "#2a9d5c",
     "R_l2": "#e07a5f", "LR_l2": "#4895ef", "SNA_l2": "#52b788", "R_l2init": "#9d4edd"}
N = {"R": "R  ReLU", "LR": "LR  leaky .1", "SNA": "SNA  adaptive α",
     "R_l2": "R + l2", "LR_l2": "LR + l2", "SNA_l2": "SNA + l2", "R_l2init": "R + l2init"}
FINAL = {"R": 0.114, "LR": 0.808, "SNA": 0.986, "R_l2": 0.941,
         "LR_l2": 0.950, "SNA_l2": 0.964, "R_l2init": 0.958}
D = {a: np.load(SRC / f"{a}.npz") for a in ARMS if (SRC / f"{a}.npz").exists()}
ARMS = [a for a in ARMS if a in D]
E = D[ARMS[0]]["edges"]; ctr = (E[:-1] + E[1:]) / 2
T = min(D[a]["h1"].shape[0] for a in ARMS)
zz = np.linspace(E[0], E[-1], 1400)


def dphi(a, z, alpha=None):
    if a.startswith("R"):  return (z > 0).astype(float)
    if a.startswith("LR"): return np.where(z > 0, 1.0, 0.1)
    return 1 + np.sin(2 * (alpha if alpha else 0.6) * z)


fig, ax = plt.subplots(2, len(ARMS), figsize=(3.05 * len(ARMS), 7.6), sharex=True)
fig.subplots_adjust(top=.80, hspace=.13, wspace=.10, left=.035, right=.985, bottom=.10)
art = {}
for j, a in enumerate(ARMS):
    for i, mk in enumerate(("m1", "m2")):
        A = ax[i, j]
        g = A.twinx(); g.plot(zz, dphi(a, zz), color="#c9a227", lw=1.0, alpha=.8)
        g.set_ylim(-.15, 2.3); g.set_yticks([])
        pooled, = A.plot([], [], color=C[a], lw=1.0, alpha=.45)
        bars = A.bar(ctr, np.zeros_like(ctr), width=(E[1] - E[0]), color=C[a], alpha=.85)
        A.axvline(0, color="k", lw=.7, ls=":")
        A.set_xlim(-14, 6); A.set_ylim(0, 1.05); A.set_yticks([])
        if i == 0: A.set_title(f"{N[a]}\nfinal {FINAL[a]:.3f}", fontsize=10.5, color=C[a], pad=6)
        if i == 1: A.set_xlabel("preactivation z", fontsize=9)
        if j == 0: A.set_ylabel(f"layer {i+1}", fontsize=10)
        art[(a, i)] = (pooled, bars)
txt = fig.text(.5, .935, "", ha="center", fontsize=13)
fig.text(.5, .895, "bars = the 100 per-unit means z̄ᵢ    faint = pooled z over 1200 images × units    gold = φ'(z)",
         ha="center", fontsize=9, color="#444")
fig.text(.5, .022, "Random Label MNIST — 1200 images, fresh random labels each task, 400 epochs/task, "
         "Adam 1e-3, seed 0. UNREGISTERED diagnostic (results/_diag_rlhist_0907)",
         ha="center", fontsize=8.5, color="#666")


def frame(t):
    for a in ARMS:
        for i, (hk, mk) in enumerate((("h1", "m1"), ("h2", "m2"))):
            pooled, bars = art[(a, i)]
            h = D[a][hk][t].astype(float); h = h / max(h.max(), 1)
            pooled.set_data(ctr, h)
            mh, _ = np.histogram(D[a][mk][t], bins=E)
            mh = mh.astype(float) / max(mh.max(), 1)
            for b, v in zip(bars, mh): b.set_height(v)
    txt.set_text(f"task {t+1} / {T}      memorised: "
                 + "   ".join(f"{a} {D[a]['acc'][t]:.2f}" for a in ARMS))
    return []


anim = FuncAnimation(fig, frame, frames=range(T), interval=260, blit=False)
out = sys.argv[1] if len(sys.argv) > 1 else "results/_diag_rlhist_0907/preact_rl.gif"
anim.save(out, writer=PillowWriter(fps=4), dpi=70)
print("wrote", out)
