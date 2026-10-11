"""Performance is the primary endpoint; depth diagnostics never determine its verdict."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from analysis.ch_chb_performance_1011 import common as C

T95_DF9 = 2.2621571628540993


def describe(values):
    x = np.asarray(values, dtype=np.float64)
    return dict(mean=float(x.mean()), median=float(np.median(x)),
                min=float(x.min()), max=float(x.max()))


def paired_ci(values, contrast="late_CHB_minus_CH"):
    x = np.asarray(values, dtype=np.float64)
    if x.shape != (10,) or not np.isfinite(x).all():
        raise ValueError("paired inference requires exactly ten finite seed differences")
    mean, sd = float(x.mean()), float(x.std(ddof=1))
    half = T95_DF9 * sd / math.sqrt(10)
    lo, hi = mean - half, mean + half
    labels = {
        "late_CHB_minus_CH": ("CHB_higher", "CHB_lower"),
        "drop_CH_minus_CHB": ("CH_drops_more", "CHB_drops_more"),
    }
    if contrast not in labels:
        raise ValueError(f"unknown paired contrast: {contrast}")
    positive_label, negative_label = labels[contrast]
    return dict(n=10, mean=mean, sample_sd=sd, standard_error=sd / math.sqrt(10),
                t_critical=T95_DF9, df=9, ci95=[lo, hi], mean_pp=100 * mean,
                ci95_pp=[100 * lo, 100 * hi], differences=x.tolist(),
                positive=int((x > 0).sum()), zero=int((x == 0).sum()), negative=int((x < 0).sum()),
                contrast=contrast,
                direction=positive_label if lo > 0 else negative_label if hi < 0 else "uncertain")


def aggregate(byarm):
    if set(byarm) != set(C.ARMS):
        raise ValueError("both CH and CHB are required")
    valid = {a: C.validate_rows(byarm[a], a) for a in C.ARMS}
    per_seed = []
    for s in C.SEEDS:
        row = dict(seed=s)
        for a in C.ARMS:
            for label, window in (("early", range(31, 51)), ("late", range(181, 201))):
                row[f"{a}_{label}"] = float(np.mean(
                    [valid[a][s, t]["online_acc"] for t in window], dtype=np.float64))
            row[f"{a}_drop"] = row[f"{a}_early"] - row[f"{a}_late"]
        row["late_difference_CHB_minus_CH"] = row["CHB_late"] - row["CH_late"]
        row["drop_difference_CH_minus_CHB"] = row["CH_drop"] - row["CHB_drop"]
        per_seed.append(row)
    diagnostics = {a: {
        str(s): {k: float(np.mean([valid[a][s, t][k] for t in range(181, 201)], dtype=np.float64))
                 for k in C.DIAGNOSTICS} for s in C.SEEDS} for a in C.ARMS}
    result = dict(run_id=C.RUN, primary_window=[181, 200], early_window=[31, 50], seeds=C.SEEDS,
                  primary=paired_ci([r["late_difference_CHB_minus_CH"] for r in per_seed]),
                  arms={a: {window: describe([r[f"{a}_{window}"] for r in per_seed])
                            for window in ("early", "late", "drop")} for a in C.ARMS},
                  secondary_drop_difference=paired_ci(
                      [r["drop_difference_CH_minus_CHB"] for r in per_seed], contrast="drop_CH_minus_CHB"),
                  per_seed=per_seed, diagnostics_last20=diagnostics,
                  method="Paired seed mean; Student t interval, df=9; approximate inference",
                  limitations=["Combined b1/b2 intervention; not b2-only identification",
                               "Prevention from start; not rescue after collapse",
                               "No claim beyond 200 tasks; no equivalence margin specified"])
    return result, valid


def draw(out, valid, result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = dict(CH="#d66b32", CHB="#2176ae")
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
    x = np.arange(1, 201)
    for ax, metric in zip(axes, ("online_acc", "memo_acc")):
        for a in C.ARMS:
            y = np.asarray([[valid[a][s, t][metric] for t in x] for s in C.SEEDS]) * 100
            ax.fill_between(x, y.min(0), y.max(0), color=colors[a], alpha=.15)
            ax.plot(x, y.mean(0), color=colors[a], label=a, linewidth=1.8)
        ax.axvspan(181, 200, color="black", alpha=.05)
        ax.set_ylabel(("Online" if metric == "online_acc" else "End-of-task") + " accuracy (%)")
        ax.set_ylim(0, 101)
        ax.grid(alpha=.15)
    axes[0].legend(title="Line: seed mean; band: seed min–max")
    axes[-1].set_xlabel("Task")
    fig.tight_layout()
    fig.savefig(out / "performance.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for row in result["per_seed"]:
        axes[0].plot([0, 1], [100 * row["CH_late"], 100 * row["CHB_late"]],
                     color="#777777", alpha=.65, marker="o", markersize=3)
    axes[0].set_xticks([0, 1], C.ARMS)
    axes[0].set_ylabel("Tasks 181–200 online accuracy (%)")
    p = result["primary"]
    axes[1].scatter(C.SEEDS, 100 * np.asarray(p["differences"]), color="#4b5360", label="Paired seed differences")
    axes[1].axhline(0, color="#999999", linewidth=1)
    axes[1].axhline(p["mean_pp"], color="#2176ae", label="Mean difference")
    axes[1].axhspan(*p["ci95_pp"], color="#2176ae", alpha=.15, label="95% t CI of mean")
    axes[1].set(xlabel="Seed", ylabel="CHB − CH (percentage points)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "paired_performance.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(10, 9), sharex=True)
    for layer in (1, 2):
        for row_index, metric in enumerate(("bias_over_sd", "dead_frac", "gate_zero_frac")):
            ax = axes[row_index, layer - 1]
            for a in C.ARMS:
                y = np.asarray([[valid[a][s, t][f"{metric}_l{layer}"] for t in x] for s in C.SEEDS])
                ax.fill_between(x, y.min(0), y.max(0), color=colors[a], alpha=.15)
                ax.plot(x, y.mean(0), color=colors[a], label=a)
            ax.set_ylabel(f"{metric} (layer {layer})")
            if row_index == 0:
                ax.legend()
            if row_index == 2:
                ax.set_xlabel("Task")
    fig.suptitle("Diagnostics only; line = seed mean, band = seed min–max")
    fig.tight_layout()
    fig.savefig(out / "diagnostics.png", dpi=180)
    plt.close(fig)


def report(out=C.OUT):
    if Path(out).resolve() != C.OUT.resolve():
        raise ValueError("report is restricted to this experiment's output")
    out = Path(out)
    manifest = json.loads((out / "launch_manifest.json").read_text())
    if manifest["config"] != C.config():
        raise ValueError("launch configuration differs from registered settings")
    byarm = {}
    for a in C.ARMS:
        completion = json.loads((out / a / "completion.json").read_text())
        if completion["completed_task"] != 200 or completion["status"] != "completed":
            raise ValueError(f"{a} is incomplete")
        if completion["csv_sha256"] != C.sha(out / a / "per_task.csv"):
            raise ValueError(f"{a}: result CSV changed since completion")
        byarm[a] = C.read_rows(out / a / "per_task.csv")
    result, valid = aggregate(byarm)
    result["analysis_provenance"] = dict(git_hash=C.git("rev-parse", "HEAD"),
                                        source_sha256=C.sources(), created_at=C.utc(),
                                        launch_manifest_sha256=C.sha(out / "launch_manifest.json"),
                                        input_sha256={a: C.sha(out / a / "per_task.csv") for a in C.ARMS})
    C.put(out / "paired_ci.json", result)
    with (out / "per_seed.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(result["per_seed"][0]))
        writer.writeheader()
        writer.writerows(result["per_seed"])
    p, arms = result["primary"], result["arms"]
    sentence = {"CHB_higher": "CHBの後期成績がCHより高いことを支持する。",
                "CHB_lower": "CHBの後期成績がCHより低いことを支持する。",
                "uncertain": "この反復数では差の方向を確定できない。効果がない、または同等という判定ではない。"}[p["direction"]]
    lines = [f"# CH と CHB の200課題での性能比較", "",
             f"最後の20課題の成績は、CHが **{100 * arms['CH']['late']['mean']:.2f}%**、"
             f"CHBが **{100 * arms['CHB']['late']['mean']:.2f}%**。"
             f"同じseedどうしの差（CHB−CH）は平均 **{p['mean_pp']:.2f} pt**"
             f"［95%信頼区間 {p['ci95_pp'][0]:.2f}, {p['ci95_pp'][1]:.2f} pt］だった。", "", sentence, "",
             "成績は各課題の全30,000更新における、更新前のミニバッチ正解率の平均（online accuracy）。"
             "各seedのt181–200平均を計算し、独立反復の単位をseed（10本）として対応差のt区間を求めた。"
             "区間はseed間差に対する近似推論である。", "",
             f"CHB−CHが正のseedは{p['positive']}/10、ゼロは{p['zero']}/10、負は{p['negative']}/10。", "",
             "|条件|平均 (%)|中央値 (%)|最小 (%)|最大 (%)|", "|---|---:|---:|---:|---:|"]
    for a in C.ARMS:
        v = arms[a]["late"]
        lines.append(f"|{a}|" + "|".join(f"{100*v[k]:.3f}" for k in ("mean", "median", "min", "max")) + "|")
    lines += ["", "|seed|CH後期 (%)|CHB後期 (%)|差 (pt)|CH早期→後期低下 (pt)|CHB早期→後期低下 (pt)|",
              "|---|---:|---:|---:|---:|---:|"]
    for r in result["per_seed"]:
        lines.append(f"|{r['seed']}|" + "|".join(f"{100*r[k]:.4f}" for k in
                     ("CH_late", "CHB_late", "late_difference_CHB_minus_CH", "CH_drop", "CHB_drop")) + "|")
    lines += ["", "早期窓はt31–50。全桁の値、早期成績と低下量の対応差、両層の診断値は"
              " `paired_ci.json` と `per_seed.csv` に保存した。", "",
              "CHBはb1の更新後ゼロ化とb2の減衰を組み合わせた開始時からの介入であり、"
              "b2減衰だけの効果や、崩壊後の回復を切り分けた比較ではない。200課題を超える保持も検証していない。", "",
              "![性能の推移。線はseed平均、帯は全seedの最小〜最大。](performance.png)", "",
              "![各seedの対応差と平均差の95%信頼区間。](paired_performance.png)", "",
              "![主性能評価とは別に示す機構診断。](diagnostics.png)", ""]
    (out / "summary.md").write_text("\n".join(lines))
    draw(out, valid, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=C.OUT)
    args = parser.parse_args()
    try:
        report(args.out)
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit(f"集計を中止: {exc}") from exc
