"""aniso_0812 (スパイク型異方 Sigma) の成果物図。

  python -m src.figures_aniso results/aniso_0812

予測: E[g] の入力層成分は Sigma mu = kappa mu に比例し、
cos_inter / cos_intra と snr_W が kappa とともに単調増加する。
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

K_COLORS = {1: "tab:blue", 4: "tab:orange", 16: "tab:red"}
LR_STYLES = {0.003: "-", 0.001: ":"}     # lr 別の線種 (config の lr_values に対応)


def load_all(resdir):
    runs = pd.read_csv(resdir + "/runs.csv").set_index("run_id")

    def cat(prefix):
        fs = sorted(glob.glob(os.path.join(resdir, f"{prefix}_*.csv")))
        if not fs:
            return pd.DataFrame()
        df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
        # inner join: runs.csv に無い run_id (別実験の残骸 CSV 等) は落とす
        return df.join(runs, on="run_id", how="inner")

    return dict(runs=runs, g=cat("freeze_global"), n=cat("freeze_neurons"),
                lop=cat("lop_metrics"))


def fig_snr_vs_kappa(d, figdir):
    g = d["g"]
    last = g.ckpt.max()
    g = g[g.ckpt == last]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, col in zip(axes, ["snr_all", "snr_W"]):
        for width in sorted(g.width.unique()):
            for lr in sorted(g.lr.unique()):
                s = g[(g.width == width) & (g.lr == lr)]
                if s.empty:
                    continue
                m = s.groupby("kappa")[col].mean()
                sd = s.groupby("kappa")[col].std()
                ax.errorbar(m.index, m.values, yerr=sd.values, marker="o",
                            ls=LR_STYLES.get(lr, "-"), label=f"w={width} lr={lr:g}")
        fl = g.noise_floor.mean()
        ax.axhline(fl, ls="--", color="gray", lw=1, label="noise floor")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("kappa")
        ax.set_title(col)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("SNR")
    fig.suptitle(f"freeze SNR vs kappa (ckpt={last:g}; Sigma = I + (kappa-1)uu^T, mu || u)")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "fig_an_snr_kappa.png"), dpi=150)
    plt.close(fig)


def fig_cos_by_kappa(d, figdir):
    n = d["n"]
    last = n.ckpt.max()
    n = n[n.ckpt == last]
    widths = sorted(n.width.unique())
    fig, axes = plt.subplots(2, len(widths), figsize=(5.5 * len(widths), 7),
                             sharex=True, squeeze=False)
    bins = np.linspace(-1, 1, 41)
    for j, width in enumerate(widths):
        for i, col in enumerate(["cos_inter", "cos_intra"]):
            ax = axes[i][j]
            for k in sorted(n.kappa.unique()):
                for lr in sorted(n.lr.unique()):
                    s = n[(n.width == width) & (n.kappa == k) & (n.lr == lr)][col].dropna()
                    if s.empty:
                        continue
                    ax.hist(s, bins=bins, histtype="step", density=True, lw=1.5,
                            color=K_COLORS.get(k), ls=LR_STYLES.get(lr, "-"),
                            label=f"k={k} lr={lr:g} |cos|={s.abs().mean():.2f}")
            ax.set_title(f"width={width}, {col}")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
            if j == 0:
                ax.set_ylabel("density")
    for ax in axes[-1]:
        ax.set_xlabel("cos(E[g_w_i], mu_hat)")
    fig.suptitle(f"cos(E[g_w_i], mu_hat) distribution vs kappa (ckpt={last:g})")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "fig_an_cos_kappa.png"), dpi=150)
    plt.close(fig)


def fig_lop_vs_step(d, figdir):
    lop = d["lop"]
    widths = sorted(lop.width.unique())
    metrics = ["dead_frac", "eff_rank", "wcos_mean", "eval_loss"]
    fig, axes = plt.subplots(len(widths), len(metrics),
                             figsize=(4.2 * len(metrics), 3.6 * len(widths)),
                             squeeze=False)
    for i, width in enumerate(widths):
        for j, metric in enumerate(metrics):
            ax = axes[i][j]
            sub = lop[lop.width == width]
            for k in sorted(sub.kappa.unique()):
                for lr in sorted(sub.lr.unique()):
                    s = sub[(sub.kappa == k) & (sub.lr == lr)]
                    if s.empty:
                        continue
                    m = s.groupby("step")[metric].mean()
                    sd = s.groupby("step")[metric].std()
                    ax.errorbar(m.index, m.values, yerr=sd.values, marker="o", ms=3,
                                lw=1.2, color=K_COLORS.get(k), ls=LR_STYLES.get(lr, "-"),
                                label=f"k={k} lr={lr:g}")
            if metric == "eval_loss":
                ax.set_yscale("log")
            ax.set_title(f"w={width}, {metric}", fontsize=10)
            ax.set_xlabel("step")
            ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=8)
    fig.suptitle("LoP metrics vs step (by kappa)")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "fig_an_lop_step.png"), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    args = ap.parse_args()
    figdir = os.path.join(args.results, "figures")
    os.makedirs(figdir, exist_ok=True)
    d = load_all(args.results)
    fig_snr_vs_kappa(d, figdir)
    fig_cos_by_kappa(d, figdir)
    fig_lop_vs_step(d, figdir)
    print(f"figures -> {figdir}")


if __name__ == "__main__":
    main()
