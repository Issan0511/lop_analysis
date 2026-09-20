#!/usr/bin/env python3
"""Post-hoc layer interpretation of erosion_race_0919; no registered score edits."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics as st

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SEEDS = (1001, 1002, 1003, 1004, 1005)
BASE = ("R", "GELU", "ELU", "LR", "LK001", "SNA")
JOBS = tuple(f"{a}_base" for a in BASE) + ("GELU_early", "GELU_late", "ELU_early")
TIMES = ((1, 5000), (1, 30000), (2, 30000))


def csv_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build(root, raw_root=None):
    raw_root = raw_root or root
    data, source_paths, trajectories = {}, [], {}
    for job in JOBS:
        path = root / job / "diagnostics.csv"
        rows = csv_rows(path)
        by_key = {(int(r["seed"]), int(r["task"]), int(r["step"])): r for r in rows}
        if len(by_key) != len(rows):
            raise ValueError(f"Duplicate observations: {job}")
        unit_path = raw_root / job / "unit_metrics.npz"
        with np.load(unit_path) as archive:
            unit = {key: archive[key].copy() for key in archive.files}
        if tuple(unit["seeds"].tolist()) != SEEDS:
            raise ValueError(f"Wrong seed order: {job}")
        unit_index = {(int(t), int(s)): i for i, (t, s) in enumerate(zip(unit["task"], unit["step"]))}
        records, medians = [], []
        for task, step in TIMES:
            group = []
            for slot, seed in enumerate(SEEDS):
                raw = by_key[(seed, task, step)]
                record = {"seed": seed, "task": task, "step": step, "acc": float(raw["acc"])}
                for layer in (1, 2):
                    dead = unit[f"dead_l{layer}"][unit_index[(task, step)], slot]
                    if not np.all((dead == 0) | (dead == 1)):
                        raise ValueError("Unit dead array must contain binary threshold flags")
                    record[f"layer{layer}"] = {
                        "dead_units": int(dead.sum()), "units": int(dead.size),
                        "dead_fraction": float(raw[f"dead_l{layer}"]),
                        "ppos": float(raw[f"ppos_l{layer}"]),
                        "zmean_unit_lower_median": float(raw[f"zmean_l{layer}"]),
                        "V_unit_lower_median": float(raw[f"output_sd_ratio_l{layer}"]),
                    }
                records.append(record)
                group.append(record)
            summary = {"task": task, "step": step, "accuracy_seed_median": st.median(r["acc"] for r in group)}
            for layer in (1, 2):
                keys = ("dead_units", "dead_fraction", "ppos", "zmean_unit_lower_median", "V_unit_lower_median")
                summary[f"layer{layer}"] = {k + "_seed_median": st.median(r[f"layer{layer}"][k] for r in group) for k in keys}
                summary[f"layer{layer}"]["all_units_threshold_dead_seeds"] = sum(
                    r[f"layer{layer}"]["dead_units"] == r[f"layer{layer}"]["units"] for r in group)
            medians.append(summary)
        data[job] = {"checkpoints": medians, "seed_records": records}
        trajectories[job] = by_key
        source_paths.extend(((path, Path(job) / "diagnostics.csv"), (unit_path, Path(job) / "unit_metrics.npz")))
        if job == "ELU_base":
            tail = []
            i0, i1 = unit_index[(1, 0)], unit_index[(1, 30000)]
            for slot, seed in enumerate(SEEDS):
                initial = unit["outputsd_l1"][i0, slot]
                end = unit["outputsd_l1"][i1, slot]
                eligible = initial > 1e-6
                ratios = (end / np.maximum(initial, np.float32(1e-6)))[eligible]
                lower_median = float(np.sort(ratios)[(len(ratios) - 1) // 2])
                official = float(by_key[(seed, 1, 30000)]["output_sd_ratio_l1"])
                if not np.isclose(lower_median, official, rtol=1e-5, atol=1e-8):
                    raise ValueError(f"ELU ratio median disagrees with recorded V: seed={seed}")
                tail.append({"seed": seed, "task1_end_acc": float(by_key[(seed, 1, 30000)]["acc"]),
                             "eligible_units": int(eligible.sum()), "q50_official_torch_lower_median": official,
                             "q50_recomputed_lower_median": lower_median,
                             "q90_numpy_linear": float(np.quantile(ratios, .9, method="linear"))})
    flux = []
    for arm in BASE:
        job = f"{arm}_base"
        pairs = []
        for seed in SEEDS:
            chosen = [r for (s, task, step), r in trajectories[job].items() if s == seed and task == 1 and 0 < step <= 750]
            chosen.sort(key=lambda r: int(r["step"]))
            if int(chosen[0]["window_start"]) != 0 or int(chosen[-1]["step"]) != 750:
                raise ValueError(f"Incomplete flux coverage: {job}/{seed}")
            down = sum(float(r["down"]) for r in chosen)
            up = sum(float(r["up"]) for r in chosen)
            v = trajectories[job][(seed, 1, 750)]
            pairs.append({"seed": seed, "E_down": down, "R_up": up, "N_down_minus_up": down - up,
                          "V_l1_at750": float(v["output_sd_ratio_l1"]), "V_l2_at750": float(v["output_sd_ratio_l2"])})
        flux.append({"arm": arm, "mean_E": st.mean(r["E_down"] for r in pairs),
                     "mean_R": st.mean(r["R_up"] for r in pairs),
                     "mean_N": st.mean(r["N_down_minus_up"] for r in pairs),
                     "median_V_l1_at750": st.median(r["V_l1_at750"] for r in pairs),
                     "median_V_l2_at750": st.median(r["V_l2_at750"] for r in pairs), "seed_records": pairs})
    result = {"analysis_type": "posthoc_after_observing_registered_results", "experiment": "erosion_race_0919",
              "registered_judgments_unchanged": True, "seeds": list(SEEDS),
              "dead_definition": "max over 1200 images abs(native phi derivative) < 1e-6; NOT exact zero autograd",
              "V_definition": "per-unit output sd divided by initial output sd; exclude initial sd <= 1e-6; Torch lower median",
              "checkpoints": data, "natural_early_flux": flux, "ELU_base_task1_response_quantiles": tail,
              "sources": [{"path": str(relative), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p, relative in source_paths]}
    return result, trajectories


def fmt(x):
    return f"{x:.6g}"


def markdown(result):
    lines = ["# erosion_race_0919 — 層ごとの応答と失敗経路（事後解析）", "",
             "**登録結果を見た後の補助解析。C1–C4・M1–M3・P1–P7の登録判定と採点は変更しない。**", "",
             "C1の救済基準は不成立だった。ただし、介入が第1層の応答を残したまま第2層が閾値死亡する走があり、失敗を単に『5000更新の猶予が短かった』と説明するのは不十分である。", "",
             "ここで死亡は、1200画像での実活性化の微分についてmax|φ′|<10⁻⁶という閾値。厳密なゼロ勾配ではない。ELUにはnative exp(z)を使用しており、実験用の微分床は加えていない。", "",
             "## 課題1末：機能と両層", "", "| 腕 | acc中央値 | L1死亡unit中央値 | L2死亡unit中央値 | 全100unitが閾値死亡 L1/L2 | V1中央値 | V2中央値 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for job, data in result["checkpoints"].items():
        r = next(r for r in data["checkpoints"] if (r["task"], r["step"]) == (1, 30000))
        a, b = r["layer1"], r["layer2"]
        lines.append(f"| {job} | {fmt(r['accuracy_seed_median'])} | {fmt(a['dead_units_seed_median'])}/100 | {fmt(b['dead_units_seed_median'])}/100 | {a['all_units_threshold_dead_seeds']}/5・{b['all_units_threshold_dead_seeds']}/5 | {fmt(a['V_unit_lower_median_seed_median'])} | {fmt(b['V_unit_lower_median_seed_median'])} |")
    lines += ["", "![両層の閾値死亡率](figure_layers.png)", "",
              "## 介入の解除時・課題1末・課題2末", "",
              "全9腕・全5seedを集約。pposは画像×unitの正側割合、zmeanはunitごとの平均前活性の下側中央値、Vはunitごとの活性sd比の下側中央値。下表ではさらにseed中央値を取る。全seed値はinterpretation.jsonに保存。", "",
              "| 腕 | 課題/step | 層 | 死亡unit中央値 | 全死亡seed | ppos | zmean | V |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for job, data in result["checkpoints"].items():
        for r in data["checkpoints"]:
            for layer in (1, 2):
                q = r[f"layer{layer}"]
                lines.append(f"| {job} | {r['task']}/{r['step']} | {layer} | {fmt(q['dead_units_seed_median'])}/100 | {q['all_units_threshold_dead_seeds']}/5 | {fmt(q['ppos_seed_median'])} | {fmt(q['zmean_unit_lower_median_seed_median'])} | {fmt(q['V_unit_lower_median_seed_median'])} |")
    lines += ["", "## 自然腕の最初750更新：下向きと上向きの両方", "",
              "全6自然腕を省略せず表示。E/Rは固定最初64画像×100unitの**初期正側だった同じ対**で、実総更新の点ごとのΔzをrectifyしてから平均・積算した量。現在の正側だけを測った量でも、confidence/label成分の単純な別名でもない。σはunitの初期標準偏差。E/R/Nはseed算術平均、Vはseed中央値。", "",
              "| 腕 | 平均E（下） | 平均R（上） | 平均N=E−R | V1@750 | V2@750 |", "|---|---:|---:|---:|---:|---:|"]
    for r in result["natural_early_flux"]:
        lines.append(f"| {r['arm']} | {fmt(r['mean_E'])} | {fmt(r['mean_R'])} | {fmt(r['mean_N'])} | {fmt(r['median_V_l1_at750'])} | {fmt(r['median_V_l2_at750'])} |")
    lines += ["", "## ELU：中央値の応答低下は機能喪失と同義ではない", "",
              "ELU/baseの課題1末で、全5seedの第1層sd比を提示する。q50は登録Vと同じTorchの下側中央値、q90はNumPyのlinear補間。初期sd≤10⁻⁶のunitを除外する。", "",
              "| seed | 課題1末acc | 適格unit | q50（登録V） | q90（linear） |", "|---|---:|---:|---:|---:|"]
    for r in result["ELU_base_task1_response_quantiles"]:
        lines.append(f"| {r['seed']} | {fmt(r['task1_end_acc'])} | {r['eligible_units']} | {fmt(r['q50_official_torch_lower_median'])} | {fmt(r['q90_numpy_linear'])} |")
    counter = next(r for r in result["ELU_base_task1_response_quantiles"] if r["seed"] == 1003)
    lines += ["", f"seed1003はacc={fmt(counter['task1_end_acc'])}でもV={fmt(counter['q50_official_torch_lower_median'])}、q90={fmt(counter['q90_numpy_linear'])}。典型的unitの応答低下と、少数の大きな応答を持つunitの存在・高い適合が両立している。各unitの因果的な貢献は測っていない。M3の閾値到達はそのまま記録するが、『V≤.1なら学べない』とは解釈しない。", "",
              "## 限定した解釈", "",
              "- 初期confidence由来の負方向平均更新を抑える介入は、第1層の応答保持とネットワーク全体の適合改善を同一にはしなかった。confidenceを弱めれば常に有益、とは支持しない。",
              "- 第1層が残り第2層が閾値死亡する経路は、層ごとの制約を含む説明を支持する。介入による特徴分散や次層入力平均の変化が候補だが、本集計はその唯一の媒介機構を証明していない。",
              "- 第一層の共通入力平均方向の介入は画像間差も変え得る。『平均だけを動かした』純粋な操作とは読まない。",
              "- ELUの反例は中央値だけでは機能容量を代表しきれないことを示す。少数の大きな応答、両層の状態、誤差への整合を併せて扱う必要がある。",
              "- 2課題・5seedの事後解析であり、50課題後の維持、一般の活性化、一般の最適化に拡張しない。", ""]
    return "\n".join(lines)


def plot(trajectories, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axs = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True, sharex=True, sharey=True)
    for col, arm in enumerate(("GELU", "ELU")):
        for row, layer in enumerate((1, 2)):
            ax = axs[row, col]
            for mode, color in (("base", "#64748b"), ("early", "#2563eb")):
                records = trajectories[f"{arm}_{mode}"]
                steps = sorted(step for (seed, task, step) in records if seed == SEEDS[0] and task == 1)
                ys = np.asarray([[float(records[(s, 1, t)][f"dead_l{layer}"]) for t in steps] for s in SEEDS])
                for y in ys:
                    ax.plot(steps, y, color=color, alpha=.22, lw=.9)
                ax.plot(steps, np.median(ys, axis=0), color=color, lw=2.3, label=mode)
            ax.axvspan(1, 5000, color="#2563eb", alpha=.07)
            ax.axvline(5000, color="#2563eb", lw=.8, ls=":")
            ax.set_xscale("symlog", linthresh=10)
            ax.set(ylim=(-.025, 1.025), title=f"{arm} · layer {layer}", ylabel="Fraction of threshold-dead units")
            ax.grid(alpha=.15)
            ax.legend(loc="best")
    for ax in axs[1]:
        ax.set_xlabel("Task 1 update (intervention released after 5000)")
    fig.suptitle("Post-hoc layer trajectories · 5 paired seeds\nThreshold-dead: max over images |native derivative| < 1e−6; thin = seed, thick = median", fontsize=12)
    fig.savefig(path, dpi=190)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_root", type=Path, nargs="?", default=REPO / "results/erosion_race_0919")
    parser.add_argument("--raw-root", type=Path, help="Raw result mirror in the data archive after cleanup")
    args = parser.parse_args()
    result, trajectories = build(args.result_root, args.raw_root)
    for name, content in (("interpretation.json", json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"),
                          ("interpretation.md", markdown(result))):
        (args.result_root / name).write_text(content, encoding="utf-8")
    plot(trajectories, args.result_root / "figure_layers.png")
    print(json.dumps({"posthoc": True, "registered_scores_changed": False,
                      "outputs": [str(args.result_root / p) for p in ("interpretation.json", "interpretation.md", "figure_layers.png")]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
