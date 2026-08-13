"""methods_sde_0813 (Leaky ReLU / S&P / CBP の SDE 分解) の成果物図。

  python -m src.figures_methods results/methods_sde_0813

各介入が SDE dw = E[g]dt + sqrt(C(w))dW のどの項を抑制するかを、
method 別の時系列 (dead_frac / neg_gate_frac / wcos_mean / eff_rank_W /
snr_drift / eval_loss ほか) と最終値の method 横断バーで見る。
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 時系列・バーの両方で描く指標 (logy フラグ付き)
METRICS = [("dead_frac", False), ("neg_gate_frac", False), ("wcos_mean", False),
           ("eff_rank_W", False), ("top1_frac", False), ("snr_drift", True),
           ("trC_W", True), ("eff_rank", False), ("eval_loss", True)]
C_STYLES = {0.0: ":", 2.0: "-"}          # 条件B の潮流強度 (A は c=NaN で "-")


def load_all(resdir):
    runs = pd.read_csv(resdir + "/runs.csv").set_index("run_id")

    def cat(prefix):
        fs = sorted(glob.glob(os.path.join(resdir, f"{prefix}_*.csv")))
        if not fs:
            return pd.DataFrame()
        df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
        return df.join(runs, on="run_id", how="inner")

    # method の並び: runs.csv の出現順 = config の methods 順
    order = list(runs["method"].drop_duplicates())
    cmap = plt.get_cmap("tab10")
    colors = {m: cmap(i % 10) for i, m in enumerate(order)}
    return dict(runs=runs, lop=cat("lop_metrics"), order=order, colors=colors)


def _panels(df, extra_cols=1):
    exps = sorted(df.exp.unique())
    widths = sorted(df.width.unique())
    fig, axes = plt.subplots(len(exps), len(widths),
                             figsize=(6.5 * len(widths), 4.2 * len(exps)),
                             squeeze=False)
    return fig, axes, exps, widths


def fig_metric_vs_step(d, metric, figdir, logy=False):
    lop = d["lop"]
    if metric not in lop.columns:
        return
    fig, axes, exps, widths = _panels(lop)
    for i, exp in enumerate(exps):
        for j, width in enumerate(widths):
            ax = axes[i][j]
            sub = lop[(lop.exp == exp) & (lop.width == width)]
            cvals = [np.nan] if exp == "A" else sorted(sub.c.dropna().unique())
            for meth in d["order"]:
                for c in cvals:
                    s = sub[sub.method == meth] if exp == "A" else \
                        sub[(sub.method == meth) & (sub.c == c)]
                    if s.empty:
                        continue
                    m = s.groupby("step")[metric].mean()
                    ls = "-" if exp == "A" else C_STYLES.get(c, "-")
                    lbl = meth + ("" if exp == "A" else f" c={c:g}")
                    ax.plot(m.index, m.values, ls=ls, lw=1.3,
                            color=d["colors"][meth], label=lbl, alpha=0.9)
            if logy:
                ax.set_yscale("log")
            ax.set_title(f"cond {exp}, width={width}")
            ax.set_xlabel("step")
            ax.grid(alpha=0.3)
            if j == 0:
                ax.set_ylabel(metric)
    axes[0][0].legend(fontsize=6, ncol=2)
    fig.suptitle(f"{metric} vs step (by method)")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, f"fig_ms_{metric}.png"), dpi=150)
    plt.close(fig)


def fig_final_bars(d, metric, figdir, logy=False):
    """最終記録ステップの seed 平均 ± SD を method 横断バーで比較。"""
    lop = d["lop"]
    if metric not in lop.columns:
        return
    last = lop.groupby("run_id")["step"].transform("max")
    fin = lop[lop.step == last]
    fig, axes, exps, widths = _panels(fin)
    x = np.arange(len(d["order"]))
    for i, exp in enumerate(exps):
        for j, width in enumerate(widths):
            ax = axes[i][j]
            sub = fin[(fin.exp == exp) & (fin.width == width)]
            cvals = [np.nan] if exp == "A" else sorted(sub.c.dropna().unique())
            nb = len(cvals)
            for k, c in enumerate(cvals):
                s = sub if exp == "A" else sub[sub.c == c]
                m = [s[s.method == meth][metric].mean() for meth in d["order"]]
                sd = [s[s.method == meth][metric].std() for meth in d["order"]]
                off = (k - (nb - 1) / 2) * 0.8 / nb
                ax.bar(x + off, m, width=0.8 / nb, yerr=sd, capsize=2,
                       color=[d["colors"][meth] for meth in d["order"]],
                       alpha=1.0 if k == nb - 1 else 0.45,
                       label="" if exp == "A" else f"c={c:g}")
            if logy:
                ax.set_yscale("log")
            ax.set_xticks(x, d["order"], rotation=45, ha="right", fontsize=7)
            ax.set_title(f"cond {exp}, width={width}")
            ax.grid(alpha=0.3, axis="y")
            if j == 0:
                ax.set_ylabel(metric)
            if exp == "B":
                ax.legend(fontsize=7)
    fig.suptitle(f"{metric} final value by method (mean±SD over seeds; "
                 "B: light=c0, solid=c2)")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, f"fig_ms_bar_{metric}.png"), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="?", default=None)
    ap.add_argument("--outdir", default=None, help="results ディレクトリ (位置引数と同義)")
    args = ap.parse_args()
    resdir = args.results or args.outdir
    if not resdir:
        ap.error("results ディレクトリを指定してください")
    figdir = os.path.join(resdir, "figures")
    os.makedirs(figdir, exist_ok=True)
    d = load_all(resdir)
    for metric, logy in METRICS:
        fig_metric_vs_step(d, metric, figdir, logy=logy)
        fig_final_bars(d, metric, figdir, logy=logy)
    print(f"figures -> {figdir}")


if __name__ == "__main__":
    main()
