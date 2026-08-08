"""成果物図 1-7 の生成 (仕様書 §6)。

  python -m src.make_figures [--results results] [--figdir figures]
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import ROOT

T_COLORS = {100: "tab:blue", 1000: "tab:orange", 10000: "tab:green", 100000: "tab:red"}


def load_all(resdir):
    runs = pd.read_csv(os.path.join(resdir, "runs.csv")).set_index("run_id")

    def cat(prefix):
        fs = sorted(glob.glob(os.path.join(resdir, f"{prefix}_*.csv")))
        if not fs:
            return pd.DataFrame()
        df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
        return df.join(runs, on="run_id")

    return dict(runs=runs, g=cat("freeze_global"), n=cat("freeze_neurons"),
                lop=cat("lop_metrics"), loss=cat("online_loss"))


def _seed_mean(df, by, col):
    grp = df.groupby(by)[col]
    return grp.mean().reset_index(), grp.std().reset_index()


def fig1_snr_vs_step(d, figdir):
    """SNR vs ckpt ステップ (条件A, T 別 × 幅×enc パネル、ノイズ床入り)。"""
    g = d["g"]
    A = g[(g.exp == "A") & (g.lr == 0.01)]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey=True)
    for i, width in enumerate(sorted(A.width.unique())):
        for j, enc in enumerate(["std", "centered"]):
            ax = axes[i][j]
            sub = A[(A.width == width) & (A.enc == enc)]
            for T in sorted(sub.period.unique()):
                s = sub[sub.period == T]
                m, sd = _seed_mean(s, "ckpt", "snr_all")
                ax.errorbar(m.ckpt.clip(lower=1), m.snr_all, yerr=sd.snr_all,
                            marker="o", ms=4, label=f"T={T:g}", color=T_COLORS.get(T))
                fl = s.groupby("ckpt").noise_floor.mean()
                ax.plot(fl.index.to_numpy().clip(min=1), fl.values, "--", lw=1,
                        color=T_COLORS.get(T), alpha=0.5)
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_title(f"cond A, width={width}, {enc}")
            ax.set_xlabel("checkpoint step"); ax.grid(alpha=0.3)
            if j == 0:
                ax.set_ylabel("SNR = ||E[g]|| / sqrt(tr C)")
    axes[0][0].legend(fontsize=8, title="solid: SNR / dashed: 1/sqrt(M)")
    fig.suptitle("Fig.1  SNR vs step (cond A; dashed = noise floor)")
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "fig1_snr_vs_step.png"), dpi=150)
    plt.close(fig)


def fig2_snr_vs_T(d, figdir):
    """SNR vs T (最終 ckpt、幅・enc 別) + 条件B の K 依存。"""
    g = d["g"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    A = g[(g.exp == "A") & (g.lr == 0.01)]
    last = A.ckpt.max()
    ax = axes[0]
    for width in sorted(A.width.unique()):
        for enc, ls in [("std", "-"), ("centered", ":")]:
            s = A[(A.width == width) & (A.enc == enc) & (A.ckpt == last)]
            m, sd = _seed_mean(s, "period", "snr_all")
            ax.errorbar(m.period, m.snr_all, yerr=sd.snr_all, marker="o", ls=ls,
                        label=f"w={width} {enc}")
            fl = s.groupby("period").noise_floor.mean()
            ax.plot(fl.index, fl.values, "--", lw=1, color="gray", alpha=0.6)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("T (flip period)"); ax.set_ylabel("SNR at final ckpt")
    ax.set_title(f"cond A (ckpt={last:g}; gray dashed = floor)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    B = g[g.exp == "B"]
    ax = axes[1]
    if len(B):
        lastB = B.ckpt.max()
        for width in sorted(B.width.unique()):
            for c, ls in [(0.0, ":"), (2.0, "-")]:
                s = B[(B.width == width) & (B.c == c) & (B.ckpt == lastB)]
                m, sd = _seed_mean(s, "period", "snr_all")
                ax.errorbar(m.period, m.snr_all, yerr=sd.snr_all, marker="s", ls=ls,
                            label=f"w={width} c={c:g}")
        fl = B[B.ckpt == lastB].groupby("period").noise_floor.mean()
        ax.plot(fl.index, fl.values, "--", lw=1, color="gray", alpha=0.6)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("K (teacher resample period)"); ax.set_title(f"cond B (ckpt={lastB:g})")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle("Fig.2  SNR vs nonstationarity period")
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "fig2_snr_vs_T.png"), dpi=150)
    plt.close(fig)


def fig3_cos_dist(d, figdir):
    """cos(E[g_wi], mu_hat) 分布 (最終 ckpt)。"""
    n = d["n"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    A = n[(n.exp == "A") & (n.lr == 0.01) & (n.period == 10000)]
    last = A.ckpt.max() if len(A) else 0
    for j, (enc, col) in enumerate([("std", "cos_intra"), ("std", "cos_inter")]):
        for i, width in enumerate(sorted(A.width.unique())):
            ax = axes[i][j]
            for enc2, color in [("std", "tab:blue"), ("centered", "tab:red")]:
                s = A[(A.width == width) & (A.enc == enc2) & (A.ckpt == last)]
                ax.hist(s[col].dropna(), bins=40, range=(-1, 1), alpha=0.5,
                        label=enc2, color=color, density=True)
            ax.set_title(f"A w={width}, T=1e4: {col}")
            ax.set_xlabel(col); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle(f"Fig.3  cos(E[g_w_i], mu_hat) distribution (ckpt={last:g})")
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "fig3_cos_dist.png"), dpi=150)
    plt.close(fig)


def fig4_adelta_hist(d, figdir):
    """E[a_i delta] 符号ヒストグラム (sign(v_i) クロス集計付き)。"""
    n = d["n"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    last = n.ckpt.max()
    panels = [("A", 5, "std"), ("A", 100, "std"), ("A", 100, "centered"),
              ("B", 5, "std"), ("B", 100, "std"), (None, None, None)]
    for ax, (exp, width, enc) in zip(axes.flat, panels):
        if exp is None:
            ax.axis("off"); continue
        s = n[(n.exp == exp) & (n.width == width) & (n.enc == enc) &
              (n.lr == 0.01) & (n.ckpt == last)]
        if exp == "A":
            s = s[s.period == 10000]
        for sv, color in [(1.0, "tab:green"), (-1.0, "tab:purple")]:
            vals = s[s.sign_v == sv].E_adelta.dropna()
            if len(vals):
                lim = np.nanpercentile(np.abs(s.E_adelta.dropna()), 99) or 1e-6
                ax.hist(vals.clip(-lim, lim), bins=41, alpha=0.55, density=True,
                        label=f"sign(v)={sv:+.0f}", color=color)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title(f"{exp} w={width} {enc}")
        ax.set_xlabel("E[a_i delta]"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle(f"Fig.4  E[a_i delta] histogram by sign(v_i) (ckpt={last:g}, T=K=1e4/A, all K/B)")
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "fig4_adelta_hist.png"), dpi=150)
    plt.close(fig)


def fig5_m4_scatter(d, figdir):
    """M4: ||w_i|| 依存性の散布図 4 種 (A, w=100, T=1e4, std, 最終 ckpt)。"""
    n = d["n"]
    s = n[(n.exp == "A") & (n.width == 100) & (n.enc == "std") &
          (n.lr == 0.01) & (n.period == 10000)]
    last = s.ckpt.max() if len(s) else 0
    s = s[s.ckpt == last]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    specs = [("snr_i", "log"), ("cos_intra", "linear"),
             ("E_adelta", "symlog"), ("cov_proj", "symlog")]
    for ax, (col, ys) in zip(axes.flat, specs):
        c = np.where(s.sign_v > 0, "tab:green", "tab:purple")
        ax.scatter(s.w_norm, s[col], s=8, c=c, alpha=0.5)
        if ys == "log":
            ax.set_yscale("log")
        elif ys == "symlog":
            ax.set_yscale("symlog", linthresh=max(1e-6, float(np.nanmedian(np.abs(s[col])) or 1e-6)))
        ax.set_xlabel("||w_i||"); ax.set_ylabel(col); ax.grid(alpha=0.3)
    fig.suptitle(f"Fig.5  M4 scatter vs ||w_i|| (A w=100 T=1e4 std, ckpt={last:g}; "
                 "green sign(v)>0 / purple <0)")
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "fig5_m4_scatter.png"), dpi=150)
    plt.close(fig)


def fig6_signclone_vs_dup(d, figdir):
    """符号一致率 vs [J]-duplicate 率の時間発展 (標準 vs 中心化)。"""
    lop = d["lop"]
    A = lop[(lop.exp == "A") & (lop.lr == 0.01) & (lop.period == 10000)]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for i, width in enumerate(sorted(A.width.unique())):
        for j, col in enumerate(["sign_clone_frac", "dup_frac"]):
            ax = axes[i][j]
            for enc, color in [("std", "tab:blue"), ("centered", "tab:red")]:
                s = A[(A.width == width) & (A.enc == enc)]
                m, sd = _seed_mean(s, "step", col)
                ax.plot(m.step, m[col], color=color, label=enc)
                ax.fill_between(m.step, m[col] - sd[col], m[col] + sd[col],
                                color=color, alpha=0.2)
            ax.set_title(f"A w={width}: {col}")
            ax.set_xlabel("step"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle("Fig.6  sign-clone rate vs [J]-duplicate rate over time (T=1e4)")
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "fig6_signclone_vs_dup.png"), dpi=150)
    plt.close(fig)


def fig7_online_loss(d, figdir):
    """オンライン損失 ([D] Fig. 再現サニティチェック)。40k 相当の移動平均。"""
    loss = d["loss"]
    A = loss[(loss.exp == "A") & (loss.lr == 0.01)]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for i, width in enumerate(sorted(A.width.unique())):
        for j, enc in enumerate(["std", "centered"]):
            ax = axes[i][j]
            for T in sorted(A.period.unique()):
                s = A[(A.width == width) & (A.enc == enc) & (A.period == T)]
                m = s.groupby("step").loss.mean()
                m = m.rolling(40, min_periods=1).mean()
                ax.plot(m.index, m.values, label=f"T={T:g}", color=T_COLORS.get(T), lw=1)
            ax.set_title(f"A w={width} {enc}")
            ax.set_xlabel("step"); ax.set_ylabel("online squared error")
            ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle("Fig.7  online loss (40k-step moving average; LoP sanity check)")
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "fig7_online_loss.png"), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(ROOT, "results"))
    ap.add_argument("--figdir", default=os.path.join(ROOT, "figures"))
    args = ap.parse_args()
    os.makedirs(args.figdir, exist_ok=True)
    d = load_all(args.results)
    for f in [fig1_snr_vs_step, fig2_snr_vs_T, fig3_cos_dist, fig4_adelta_hist,
              fig5_m4_scatter, fig6_signclone_vs_dup, fig7_online_loss]:
        try:
            f(d, args.figdir)
            print(f"{f.__name__}: ok")
        except Exception as e:
            print(f"{f.__name__}: FAILED ({e})")


if __name__ == "__main__":
    main()
