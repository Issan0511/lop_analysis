"""fullbatch_0812 (full-batch GD vs mini-batch SGD) の成果物図。

  python -m src.figures_fullbatch results/fullbatch_0812

Path A (dead unit 化) はノイズ起因なら batch 増で消失、
Path B (低ランク整列: eff_rank / wcos_mean / sign_match) はドリフト起因なら full でも残存、
という予測を batch 別の時系列で見る。
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BATCH_ORDER = ["1", "32", "128", "full"]
BATCH_COLORS = {"1": "tab:blue", "32": "tab:orange", "128": "tab:green", "full": "tab:red"}
C_STYLES = {0.0: ":", 2.0: "-"}          # 条件B の潮流強度 (A は c=NaN で "-")


def load_all(resdir):
    runs = pd.read_csv(resdir + "/runs.csv").set_index("run_id")
    runs["batch"] = runs["batch"].astype(str)

    def cat(prefix):
        fs = sorted(glob.glob(os.path.join(resdir, f"{prefix}_*.csv")))
        if not fs:
            return pd.DataFrame()
        df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
        # inner join: runs.csv に無い run_id (別実験の残骸 CSV 等) は落とす
        return df.join(runs, on="run_id", how="inner")

    return dict(runs=runs, g=cat("freeze_global"), lop=cat("lop_metrics"))


def _panels(df):
    """(exp, width) の 2x2 パネル軸を返す。"""
    exps = sorted(df.exp.unique())
    widths = sorted(df.width.unique())
    fig, axes = plt.subplots(len(exps), len(widths),
                             figsize=(5.5 * len(widths), 4 * len(exps)),
                             squeeze=False)
    return fig, axes, exps, widths


def fig_metric_vs_step(d, metric, figdir, logy=False):
    lop = d["lop"]
    fig, axes, exps, widths = _panels(lop)
    for i, exp in enumerate(exps):
        for j, width in enumerate(widths):
            ax = axes[i][j]
            sub = lop[(lop.exp == exp) & (lop.width == width)]
            cvals = [np.nan] if exp == "A" else sorted(sub.c.dropna().unique())
            for b in BATCH_ORDER:
                for c in cvals:
                    s = sub[sub.batch == b] if exp == "A" else \
                        sub[(sub.batch == b) & (sub.c == c)]
                    if s.empty:
                        continue
                    m = s.groupby("step")[metric].mean()
                    sd = s.groupby("step")[metric].std()
                    ls = "-" if exp == "A" else C_STYLES.get(c, "-")
                    lbl = f"B={b}" + ("" if exp == "A" else f" c={c:g}")
                    ax.errorbar(m.index, m.values, yerr=sd.values, ls=ls,
                                marker="o", ms=3, lw=1.2, color=BATCH_COLORS[b],
                                label=lbl, alpha=0.9)
            if logy:
                ax.set_yscale("log")
            ax.set_title(f"cond {exp}, width={width}")
            ax.set_xlabel("step")
            ax.grid(alpha=0.3)
            if j == 0:
                ax.set_ylabel(metric)
    axes[0][0].legend(fontsize=7, ncol=2)
    fig.suptitle(f"{metric} vs step (by batch; full = full-batch GD)")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, f"fig_fb_{metric}.png"), dpi=150)
    plt.close(fig)


def fig_snr_by_batch(d, figdir):
    """凍結測定 SNR (最終 ckpt) の batch 依存。ドリフトは学習バッチによらず残存する予測。"""
    g = d["g"]
    if g.empty:
        return
    last = g.ckpt.max()
    g = g[g.ckpt == last]
    fig, axes, exps, widths = _panels(g)
    x = np.arange(len(BATCH_ORDER))
    for i, exp in enumerate(exps):
        for j, width in enumerate(widths):
            ax = axes[i][j]
            sub = g[(g.exp == exp) & (g.width == width)]
            cvals = [np.nan] if exp == "A" else sorted(sub.c.dropna().unique())
            for c in cvals:
                s = sub if exp == "A" else sub[sub.c == c]
                m = [s[s.batch == b].snr_all.mean() for b in BATCH_ORDER]
                sd = [s[s.batch == b].snr_all.std() for b in BATCH_ORDER]
                lbl = "" if exp == "A" else f"c={c:g}"
                ax.errorbar(x, m, yerr=sd, marker="o",
                            ls="-" if exp == "A" else C_STYLES.get(c, "-"), label=lbl)
            fl = sub.noise_floor.mean()
            ax.axhline(fl, ls="--", color="gray", lw=1, label="noise floor")
            ax.set_xticks(x, [f"B={b}" for b in BATCH_ORDER])
            ax.set_yscale("log")
            ax.set_title(f"cond {exp}, width={width} (ckpt={last:g})")
            ax.grid(alpha=0.3)
            if j == 0:
                ax.set_ylabel("SNR = ||E[g]|| / sqrt(tr C)")
            ax.legend(fontsize=7)
    fig.suptitle("freeze SNR vs training batch (last ckpt)")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "fig_fb_snr_batch.png"), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    args = ap.parse_args()
    figdir = os.path.join(args.results, "figures")
    os.makedirs(figdir, exist_ok=True)
    d = load_all(args.results)
    for metric, logy in [("dead_frac", False), ("eff_rank", False),
                         ("wcos_mean", False), ("sign_match_mean", False),
                         ("dup_frac", False), ("eval_loss", True)]:
        fig_metric_vs_step(d, metric, figdir, logy=logy)
    fig_snr_by_batch(d, figdir)
    print(f"figures -> {figdir}")


if __name__ == "__main__":
    main()
