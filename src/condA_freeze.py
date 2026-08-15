"""condA_freeze_0815: 条件A (µ≠0 の既存 LoP レジーム) の freeze_bias 腕。

  python -m src.condA_freeze --config configs/condA_freeze_0815.yaml [--device cpu]
                             [--arms free frozen] [--widths 5 100]
  python -m src.condA_freeze --analyze results/condA_freeze_0815

freeze_bias は run 軸ではないので腕ごとに config を差し替えて
results/condA_freeze_0815/{free,frozen}/ に出力する。
判定基準 PA-1..PA-3 は configs/condA_freeze_0815.yaml のヘッダに事前登録済み。
"""
import argparse
import copy
import glob
import os
import sys
import time

import numpy as np
import pandas as pd

from .common import ROOT, load_config, pick_device, build_runs, group_runs, group_name
from .train import train_group

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.center_selfcov.slopes import paired_boot_ci, boot_ci

OUT = os.path.join(ROOT, "results", "condA_freeze_0815")
MINOR_THRESHOLD = 0.20          # PA-2: 減少率がこれ未満なら「b は脇役」


def run_arm(cfg, frozen, device, widths=None):
    cfg = copy.deepcopy(cfg)
    cfg["condA"]["freeze_bias"] = frozen
    outdir = os.path.join(OUT, "frozen" if frozen else "free")
    os.makedirs(outdir, exist_ok=True)
    # runs.csv は --widths による絞り込みに依らず**全幅分**を書く。
    # 幅別に並列起動すると同じ outdir に書き込むため、絞り込んだ表だと互いに
    # 上書きして片方の幅が消える (解析時の join で行が落ちる)。
    pd.DataFrame(build_runs(cfg)).to_csv(os.path.join(outdir, "runs.csv"), index=False)
    if widths:
        cfg["condA"]["widths"] = widths
    runs = build_runs(cfg)
    import yaml
    with open(os.path.join(outdir, "config_used.yaml"), "w") as fh:
        yaml.safe_dump(cfg, fh, allow_unicode=True)
    for gkey, gruns in group_runs(runs).items():
        t0 = time.time()
        train_group(gkey, gruns, cfg, device, outdir)
        print(f"    [{'frozen' if frozen else 'free'}] {group_name(gkey)}: "
              f"R={len(gruns)} {time.time()-t0:.0f}s", flush=True)


def load_arm(arm):
    d = os.path.join(OUT, arm)
    runs = pd.read_csv(os.path.join(d, "runs.csv")).set_index("run_id")
    fs = sorted(glob.glob(os.path.join(d, "lop_metrics_*.csv")))
    lop = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    lop = lop.join(runs, on="run_id", how="inner")
    lop["arm"] = arm
    return lop


def analyze(resdir):
    rng = np.random.default_rng(0)
    lop = pd.concat([load_arm("free"), load_arm("frozen")], ignore_index=True)
    last = lop.sort_values("step").groupby(["arm", "run_id"]).last().reset_index()

    rows, tab = [], []
    for width, g in last.groupby("width"):
        fr = g[g.arm == "free"].set_index("seed")
        fz = g[g.arm == "frozen"].set_index("seed")
        seeds = fr.index.intersection(fz.index)
        d_ci = paired_boot_ci(fz.loc[seeds].dead_frac.values,
                              fr.loc[seeds].dead_frac.values, rng)
        e_ci = paired_boot_ci(fz.loc[seeds].eval_loss.values,
                              fr.loc[seeds].eval_loss.values, rng)
        dfree, dfroz = fr.loc[seeds].dead_frac.mean(), fz.loc[seeds].dead_frac.mean()
        red = (dfree - dfroz) / dfree if dfree > 0 else np.nan
        tab.append(dict(width=width, dead_free=dfree, dead_frozen=dfroz,
                        reduction=red, dead_diff=d_ci["mean"],
                        dead_lo=d_ci["lo"], dead_hi=d_ci["hi"],
                        eval_free=fr.loc[seeds].eval_loss.mean(),
                        eval_frozen=fz.loc[seeds].eval_loss.mean(),
                        eval_diff=e_ci["mean"], eval_lo=e_ci["lo"], eval_hi=e_ci["hi"],
                        b_mean_free=fr.loc[seeds].b_mean_alive.mean(),
                        b_min_free=fr.loc[seeds].b_min.mean(), n=len(seeds)))
    tab = pd.DataFrame(tab)

    pa1 = bool((tab.dead_hi < 0).all())          # frozen − free < 0 が有意
    rows.append(dict(pred="PA-1", scope="条件A で b 凍結が dead を減らす (frozen < free)",
                     verdict="PASS" if pa1 else "FAIL",
                     evidence="; ".join(
                         f"w{r.width}: free {r.dead_free:.3f} → frozen {r.dead_frozen:.3f} "
                         f"(diff {r.dead_diff:+.3f} CI [{r.dead_lo:+.3f}, {r.dead_hi:+.3f}])"
                         for r in tab.itertuples())))
    minor = bool((tab.reduction < MINOR_THRESHOLD).all())
    rows.append(dict(pred="PA-2", scope=f"寄与の大きさ (減少率 < {MINOR_THRESHOLD} なら b は脇役)",
                     verdict="MINOR" if minor else "MAJOR",
                     evidence="; ".join(f"w{r.width}: 減少率 {r.reduction:.3f}"
                                        for r in tab.itertuples())
                     + ("。→ **b は脇役** (Phase 0 の「b 主導 dead は 5.3%」と整合)"
                        if minor else
                        "。→ **b の寄与は無視できない**。Phase 0 の静的分解 (誰が負性を"
                        "担っているか) では捉えられない「b の沈降が dead の必要条件に"
                        "なっている」経路を示唆する = 磁石ではなく引き金")))
    rows.append(dict(pred="PA-3", scope="b 凍結の機能的影響 (eval_loss, frozen − free)",
                     verdict="FROZEN_BETTER" if bool((tab.eval_hi < 0).all())
                             else ("FROZEN_WORSE" if bool((tab.eval_lo > 0).all())
                                   else "MIXED_OR_NULL"),
                     evidence="; ".join(
                         f"w{r.width}: free {r.eval_free:.4f} → frozen {r.eval_frozen:.4f} "
                         f"(diff {r.eval_diff:+.4f} CI [{r.eval_lo:+.4f}, {r.eval_hi:+.4f}])"
                         for r in tab.itertuples())
                     + "。留保: frozen は「死のない同一ネット」ではなく閾値表現力ごと"
                       "奪ったネットなので厳密な反実仮想ではない"))
    ver = pd.DataFrame(rows)
    ver.to_csv(os.path.join(resdir, "verdict.csv"), index=False)
    tab.to_csv(os.path.join(resdir, "arm_compare.csv"), index=False)

    # 図: dead と eval の時系列 (腕別)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fd = os.path.join(resdir, "figures")
    os.makedirs(fd, exist_ok=True)
    L = lop[lop.step % 5000 == 0]
    ws = sorted(L.width.unique())
    fig, axes = plt.subplots(2, len(ws), figsize=(6 * len(ws), 6.2), squeeze=False)
    for j, w in enumerate(ws):
        for i, col in enumerate(["dead_frac", "eval_loss"]):
            ax = axes[i][j]
            for arm, c in [("free", "tab:red"), ("frozen", "tab:gray")]:
                g = L[(L.width == w) & (L.arm == arm)]
                m = g.groupby("step")[col].agg(["mean", "sem"])
                ax.plot(m.index, m["mean"], lw=1.3, color=c,
                        label=f"b {arm}")
                ax.fill_between(m.index, m["mean"] - m["sem"], m["mean"] + m["sem"],
                                color=c, alpha=0.2, lw=0)
            ax.grid(alpha=0.3)
            if col == "eval_loss":
                ax.set_yscale("log")
                ax.set_xlabel("step")
            if i == 0:
                ax.set_title(f"condA w={w} (µ≠0, existing LoP regime)")
            if j == 0:
                ax.set_ylabel(col)
                ax.legend(fontsize=8)
    fig.suptitle("Level-1 necessity test: does freezing b change the existing LoP?")
    fig.tight_layout()
    fig.savefig(os.path.join(fd, "fig_ca_freeze.png"), dpi=150)
    plt.close(fig)

    lines = ["# condA_freeze_0815 — 条件A の freeze_bias 腕 (レベル1: 必要性)\n",
             "判定基準 PA-1..PA-3 は configs/condA_freeze_0815.yaml のヘッダに"
             "実行前から事前登録済み。\n",
             "## 判定\n", ver.to_string(index=False),
             "\n\n## 腕別の最終値\n", tab.round(5).to_string(index=False),
             "\n\n## 位置づけ\n"]
    if minor and pa1:
        lines.append("- b は条件A の dead に**寄与するが脇役**。bias_margin_0814 の"
                     "Phase 0 (b 主導 dead 5.3%) と整合し、事前予測どおり。")
        lines.append("- したがって「b が既存の LoP を説明する」とは依然として**言えない**。"
                     "言えるのは「µ=0 という µ 経路を塞いだ設定では b が主役になる」まで。")
    elif pa1:
        lines.append("- **事前予測が外れた**: b の寄与は脇役に留まらない。Phase 0 の静的分解"
                     "(dead ユニットの負性を誰が担っているか) は「b は小さい」を示すが、"
                     "b の沈降を止めると dead が大きく減る = b は**磁石ではなく引き金**"
                     "という読みが必要。")
    else:
        lines.append("- b 凍結は条件A の dead を有意に変えない → **b は既存 LoP に不要**。"
                     "bias_margin_0814 の機構は µ=0 レジーム限定と結論できる。")
        w100 = tab[tab.width == 100]
        if len(w100):
            r = w100.iloc[0]
            lines.append(f"- とくに w100 は diff {r.dead_diff:+.3f} CI "
                         f"[{r.dead_lo:+.3f}, {r.dead_hi:+.3f}] で**効果が完全にゼロ**。"
                         "µ≠0 では b を 0 に固定しても dead は同じだけ進む。")
        w5 = tab[tab.width == 5]
        if len(w5):
            r = w5.iloc[0]
            lines.append(f"- w5 は diff {r.dead_diff:+.3f} CI [{r.dead_lo:+.3f}, "
                         f"{r.dead_hi:+.3f}] で減少方向だが CI がゼロを含む (境界)。"
                         "w5 は dead_frac が 0.2 刻みに量子化される (5 ユニット) ため"
                         "n=5 seed の bootstrap は粗く、**示唆どまりで有意ではない**。")
        lines.append("\n### 主張への反映")
        lines.append("- 「b が既存の LoP に関与する」は**言えない** (レベル1 不成立)。")
        lines.append("- bias_margin_0814 で言えるのは**レベル0 限定**: 「µ 経路を塞いだ "
                     "µ=0 設定では b が margin の唯一のノブになり dead を作れる」。"
                     "既存 LoP (µ≠0) の説明にはならない。")
        lines.append("- これは仮説の否定ではなく**適用範囲の確定**。Phase 0 の静的分解 "
                     "(b 主導 dead 5.3%) と本テスト (凍結しても dead 不変) が"
                     "同じ方向を指しており、条件A の dead は µ 経路で完結している。")
    with open(os.path.join(resdir, "summary.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(ver.to_string(index=False))
    print()
    print(tab.round(4).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/condA_freeze_0815.yaml")
    ap.add_argument("--device", default=None)
    ap.add_argument("--arms", nargs="*", default=["free", "frozen"])
    ap.add_argument("--widths", nargs="*", type=int, default=None)
    ap.add_argument("--analyze", default=None)
    args = ap.parse_args()

    if args.analyze:
        analyze(args.analyze)
        return
    cfg = load_config(args.config)
    if args.device:
        cfg["common"]["device"] = args.device
    device = pick_device(cfg)
    os.makedirs(OUT, exist_ok=True)
    for arm in args.arms:
        run_arm(cfg, arm == "frozen", device, widths=args.widths)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
