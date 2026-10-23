import numpy as np, matplotlib, sys
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
U = "raw/depth_silu_rlmnist_1020/units"
arms = [("1 層 SiLU", "SILU_L1_none", 1), ("2 層 SiLU", "SILU_L2_none", 2), ("4 層 SiLU", "SILU_L4_none", 4),
        ("4 層 SiLU + 入力中心化", "SILU_L4_C", 4), ("4 層 ReLU", "R_L4_none", 4)]
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
D = {n: np.load(f"{U}/units_{n}_s{seed}.npz") for _, n, _ in arms}
rng = np.random.default_rng(0); jit = rng.uniform(-0.3, 0.3, 100)
fig = plt.figure(figsize=(17, 8.5))
gs = fig.add_gridspec(2, 5, height_ratios=[3, 1.1], hspace=0.35, wspace=0.35)
axs = [fig.add_subplot(gs[0, j]) for j in range(5)]; axa = fig.add_subplot(gs[1, :])
cols = plt.cm.tab10.colors
for j, (t, n, L) in enumerate(arms):
    axa.plot(np.arange(1, 51), D[n]["A"], color=cols[j], lw=2, label=t)
axa.set_xlim(1, 50); axa.set_ylim(0, 1); axa.set_xlabel("課題"); axa.set_ylabel("オンライン精度"); axa.legend(ncol=5, fontsize=9, loc="upper right")
cur = axa.axvline(1, color="k", lw=1)
def draw(k):
    for j, (t, n, L) in enumerate(arms):
        ax = axs[j]; ax.clear(); d = D[n]
        for l in range(1, L + 1):
            r = d[f"end_r_l{l}"][k]; g = d[f"end_gabs_l{l}"][k]
            ok = np.isfinite(r)
            ax.scatter(np.clip(r[ok], -5, 3.5), l + jit[ok], c=np.clip(g[ok], 0, 1), cmap="viridis", vmin=0, vmax=1, s=10)
            dead = (d[f"end_p_l{l}"][k] == 0).mean(); hb = float(d[f"end_hbar_norm_l{l}"][k])
            ax.text(4.0, l, f"p=0: {dead:.0%}\n担い手 {hb:.3g}", fontsize=7, va="center")
        ax.axvline(0, color="k", lw=0.6); ax.set_xlim(-5, 7); ax.set_ylim(0.4, 4.6)
        ax.set_yticks(range(1, 5)); ax.set_yticklabels([f"第{l}層" for l in range(1, 5)])
        ax.set_xlabel("位置 r = μ/s（3.5 以上は右端）"); ax.set_title(f"{t}\nA = {d['A'][k]:.2f}", fontsize=10)
    cur.set_xdata([k + 1, k + 1])
    fig.suptitle(f"RL-MNIST・課題 {k + 1} の末（seed {seed}）  点 = unit、色 = |φ′| の平均（黄 = 応答あり、紫 = 応答なし）", fontsize=12)
ani = FuncAnimation(fig, draw, frames=50, interval=350)
ani.save(f"results/depth_silu_rlmnist_1020/depth_layers_anim_s{seed}.gif", writer=PillowWriter(fps=3), dpi=80)
draw(29); fig.savefig(f"{sys.argv[2] if len(sys.argv)>2 else '/tmp'}/frame30.png", dpi=80)
print("ok")
