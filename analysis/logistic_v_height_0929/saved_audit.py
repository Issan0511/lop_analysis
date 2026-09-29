"""Read-only audit of the existing RL-MNIST frozen-readout records.

Post-hoc validation, no training. Original statistics use ddof=1 and a lower
middle order statistic for the body. Cohort controls preserve these definitions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path

import numpy as np

ARMS = ("EEQ", "VF0.1", "VF1", "VF10")
FROZEN = ARMS[1:]
N = 1200
SEEDS = (0, 1)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def median(x, mask):
    return float(np.median(x[mask])) if np.any(mask) else float("nan")


def metrics(d, t, mask):
    top, body, s = (d[f"end_{k}"][t].astype(float) for k in ("top", "body", "s"))
    out = {
        "units": int(mask.sum()), "alive_switch_fraction": float((d["sw_k"][t] > 0).mean()),
        "raw_top": median(top, mask), "body": median(body, mask), "sample_sd": median(s, mask),
        "top_over_s": median(top / s, mask), "body_over_s": median(body / s, mask),
        "tail": median((top - body) / s, mask), "k_end": median(d["end_k"][t], mask),
        "positive_fraction_end": float(np.mean(d["end_k"][t][mask] / N)) if np.any(mask) else float("nan"),
        "ce_end": float(d["ce_end"][t]), "ce_mid": float(d["ce_mid"][t]),
        "acc_end": float(d["acc_end"][t]),
        "net_open_count_loss": median(1 - d["mid_k"][t] / np.maximum(d["sw_k"][t], 1), mask),
        "top_push": median(d["mid_top"][t] - d["sw_top"][t], mask),
        "top_lift": median(d["end_top"][t] - d["mid_top"][t], mask),
        "body_push": median(d["mid_body"][t] - d["sw_body"][t], mask),
    }
    for q in ("q99", "q95", "q90"):
        x = d[f"end_{q}"][t].astype(float)
        out[f"{q}_above_body_over_s"] = median((x - body) / s, mask)
        out[f"{q}_positive_fraction"] = float((x[mask] > 0).mean()) if np.any(mask) else float("nan")
    return out


def table(rows, keys):
    lines = ["| " + " | ".join(keys) + " |", "|" + "|".join(["---"] * len(keys)) + "|"]
    for row in rows:
        cells = [f"{row[k]:.5g}" if isinstance(row[k], (float, np.floating)) else str(row[k]) for k in keys]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, default=Path("/home/issan/Projects/obsidian-research-data/push_lift_ladder_1layer_0928/vfreeze"))
    p.add_argument("--out", type=Path, default=Path("results/logistic_v_height_0929"))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    records, sources = {}, []
    for arm in ARMS:
        for seed in SEEDS:
            path = args.source / "runs_vf" / f"{arm}_s{seed}.npz"
            with np.load(path, allow_pickle=False) as z:
                records[arm, seed] = {k: z[k] for k in z.files}
            sources.append(path)
    source_code = [args.source / "all_probe4.py", args.source / "analyze_vf.py", args.source / "fluct_vs_v.py", args.source.parent / "alpha_ladder_1layer.py"]
    sources.extend(source_code)
    sources.append(Path("/home/issan/Projects/obsidian-research/可塑性喪失/理論/ELUを縮める_α梯子_1層200課題と理論_0925.md"))
    checks = {"prefix_equal": [], "freeze_norm": [], "finite_positive_sd": [], "switch_equals_previous_end": []}
    inventory = []
    for (arm, seed), d in records.items():
        for k, a in d.items():
            if (arm, seed) == (ARMS[0], 0):
                inventory.append({"key": k, "shape": list(a.shape), "dtype": str(a.dtype), "value": a.item() if a.ndim == 0 else None})
        ref = records["EEQ", seed]
        for ph in ("sw", "mid", "end"):
            for k in ("top", "body", "q99", "q95", "q90", "s", "k"):
                key = f"{ph}_{k}"
                ok = bool(np.array_equal(d[key][:50], ref[key][:50]))
                checks["prefix_equal"].append({"arm": arm, "seed": seed, "key": key, "equal_through_task50": ok})
                assert ok
                if ph == "sw":
                    ok2 = bool(np.array_equal(d[key][1:], d[f"end_{k}"][:-1]))
                    checks["switch_equals_previous_end"].append({"arm": arm, "seed": seed, "key": key, "equal": ok2})
                    assert ok2
        for ph in ("sw", "mid", "end"):
            a = d[f"{ph}_s"]
            ok = bool(np.all(np.isfinite(a)) and np.all(a > 0))
            checks["finite_positive_sd"].append({"arm": arm, "seed": seed, "phase": ph, "ok": ok})
            assert ok
        if arm in FROZEN:
            scale = float(d["vscale"])
            target = ref["V_t50_ck4000"].astype(float) * scale
            for task in (100, 200):
                for ck in (0, 200, 500, 2000, 4000):
                    obs = d[f"V_t{task}_ck{ck}"].astype(float)
                    rel = float(np.max(abs(obs - target) / target))
                    const = bool(np.array_equal(obs, d["V_t100_ck0"]))
                    checks["freeze_norm"].append({"arm": arm, "seed": seed, "task": task, "ck": ck, "max_relative_scale_error": rel, "constant": const})
                    assert rel < 2e-7 and const

    # Exactly reproduce original selection, then audit selection sensitivity.
    daily, summaries = [], []
    cohort_names = ("original_switch_alive", "all_units", "common_frozen_switch_alive", "fixed_task50_alive", "always_alive_all_frozen_tasks151_200")
    for seed in SEEDS:
        fixed50 = records["EEQ", seed]["end_k"][49] > 0
        always = np.logical_and.reduce([np.all(records[a, seed]["sw_k"][150:200] > 0, axis=0) for a in FROZEN])
        for arm in ARMS:
            d = records[arm, seed]
            for cohort in cohort_names:
                subset = []
                for t in range(150, 200):
                    masks = {
                        "original_switch_alive": d["sw_k"][t] > 0,
                        "all_units": np.ones(100, dtype=bool),
                        "common_frozen_switch_alive": np.logical_and.reduce([records[a, seed]["sw_k"][t] > 0 for a in FROZEN]),
                        "fixed_task50_alive": fixed50,
                        "always_alive_all_frozen_tasks151_200": always,
                    }
                    row = {"arm": arm, "seed": seed, "cohort": cohort, "task": t + 1, **metrics(d, t, masks[cohort])}
                    daily.append(row)
                    subset.append(row)
                means = {k: float(np.mean([r[k] for r in subset])) for k in subset[0] if k not in ("arm", "seed", "cohort", "task")}
                o1, o2 = d["end_top"][150:199] > 0, d["end_top"][151:200] > 0
                summaries.append({"arm": arm, "seed": str(seed), "cohort": cohort, **means,
                                  "v_norm_t200": float(np.median(d["V_t200_ck4000"])),
                                  "mu_plus_unconditional_units": float((o1 & ~o2).sum() / o1.sum())})
    for arm in ARMS:
        for cohort in cohort_names:
            rows = [r for r in summaries if r["arm"] == arm and r["cohort"] == cohort]
            means = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k not in ("arm", "seed", "cohort")}
            # Original mu+ uses pooled event counts rather than a mean of rates.
            oo = oc = 0
            for seed in SEEDS:
                d = records[arm, seed]
                o1, o2 = d["end_top"][150:199] > 0, d["end_top"][151:200] > 0
                oo += int(o1.sum()); oc += int((o1 & ~o2).sum())
            means["mu_plus_unconditional_units"] = oc / oo
            summaries.append({"arm": arm, "seed": "pooled", "cohort": cohort, **means})
    write_csv(args.out / "saved_daily.csv", daily)
    write_csv(args.out / "saved_summary.csv", summaries)

    snapshots, timecourse, paired = [], [], []
    for (arm, seed), d in records.items():
        for task in (100, 200):
            st = d[f"D_t{task}_ck4000"].astype(float)
            counts, errors = st[:, 0], st[:, 4]
            alive = counts[0] > 0
            snapshots.append({"arm": arm, "seed": str(seed), "task": task, "v_norm": float(np.median(d[f"V_t{task}_ck4000"])),
                              "all_mean_error_norm": float(np.median((counts * errors).sum(0) / counts.sum(0))),
                              "deep_mean_error_norm": median(errors[-1], alive & (counts[-1] > 0)),
                              "open_mean_error_norm": median(errors[0], alive),
                              "alive_units": int(alive.sum()), "deep_threshold": -8})
        for task in (50, 51, 52, 53, 55, 60, 100, 200):
            t = task - 1; mask = d["sw_k"][t] > 0
            timecourse.append({"arm": arm, "seed": str(seed), "task": task,
                               "switch_top_over_s": median(d["sw_top"][t] / d["sw_s"][t], mask),
                               "switch_raw_top": median(d["sw_top"][t], mask),
                               "end_ce": float(d["ce_end"][t]),
                               "end_top_over_s": median(d["end_top"][t] / d["end_s"][t], mask)})
    for rows in (snapshots, timecourse):
        for arm in ARMS:
            for task in sorted({r["task"] for r in rows}):
                relevant = [r for r in rows if r["arm"] == arm and r["task"] == task]
                rows.append({"arm": arm, "seed": "pooled", "task": task,
                             **{k: float(np.mean([r[k] for r in relevant])) for k in relevant[0] if k not in ("arm", "seed", "task")}})
    # Same unit, same task, same subset under each intervention: prevent cohort switching.
    for seed in SEEDS:
        for pair in (("VF0.1", "VF1"), ("VF1", "VF10"), ("VF0.1", "VF10")):
            a, b = (records[arm, seed] for arm in pair)
            for cohort in ("all_units", "common_frozen_switch_alive"):
                diffs = {k: [] for k in ("top_over_s", "tail", "q99_above_body_over_s", "q95_above_body_over_s", "q90_above_body_over_s")}
                for t in range(150, 200):
                    mask = np.ones(100, bool) if cohort == "all_units" else np.logical_and.reduce([records[arm, seed]["sw_k"][t] > 0 for arm in FROZEN])
                    for metric in diffs:
                        def value(d):
                            s, body = d["end_s"][t].astype(float), d["end_body"][t].astype(float)
                            if metric == "top_over_s": return d["end_top"][t] / s
                            if metric == "tail": return (d["end_top"][t] - body) / s
                            q = metric.split("_")[0]
                            return (d[f"end_{q}"][t] - body) / s
                        diffs[metric].extend((value(a) - value(b))[mask].tolist())
                for metric, diff in diffs.items():
                    x = np.asarray(diff)
                    paired.append({"seed": seed, "small_arm": pair[0], "large_arm": pair[1], "cohort": cohort, "metric": metric,
                                   "paired_unit_tasks": len(x), "mean_difference": float(x.mean()), "median_difference": float(np.median(x)),
                                   "positive_difference_fraction": float((x > 0).mean())})
    write_csv(args.out / "saved_snapshots.csv", snapshots)
    write_csv(args.out / "saved_timecourse.csv", timecourse)
    write_csv(args.out / "saved_paired_differences.csv", paired)
    (args.out / "saved_checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    (args.out / "saved_inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
    provenance = {"analysis": "post-hoc existing-data audit; no training", "python": platform.python_version(), "numpy": np.__version__,
                  "analysis_sha256": sha256(__file__), "source_files": [{"path": str(s), "bytes": s.stat().st_size, "sha256": sha256(s)} for s in sources],
                  "primary_aggregation": "per task median across switch-alive units, arithmetic mean over tasks151..200 and two seeds; v norm median all units at task200",
                  "sd": "sample SD ddof1, N1200; multiply by sqrt(1199/1200) to obtain population SD", "body": "lower middle order statistic (torch.median)",
                  "availability": {"W1": False, "b1": False, "W2_vector": False, "W2_column_norm": True, "b2": False, "per_input_z": False,
                                   "per_input_error": False, "Adam_moments": False, "unit_order_statistics": True, "grouped_error_norms": True,
                                   "input_identity_of_max_or_open": False}}
    (args.out / "saved_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    original = [r for r in summaries if r["cohort"] == "original_switch_alive"]
    pooled = [r for r in original if r["seed"] == "pooled"]
    cohorts = [r for r in summaries if r["seed"] == "pooled" and r["arm"] in FROZEN]
    errorrows = [r for r in snapshots if r["seed"] == "pooled"]
    report = """# 保存済み vfreeze の独立再集計

新しい学習は行っていない。既存の8 NPZを読み、§8.36–8.43のうち保存値で確かめられる数値を独立に再集計した。2 seedの記述統計で、unitやtaskを独立標本とした有意差検定はしていない。

## 介入と窓の確定

- 入力は各seedで固定の1200枚のMNIST（raw/255）、784–100–10。taskごとにラベルだけを引き直す。バッチ16、4000更新/task、学習float32、診断はfloat64計算後にunit統計をfloat32保存。
- vはunitごとの10クラス読み出しベクトル、表の |v| はW2の列のEuclidean norm。符号付き平均ではない。class共通方向を除いたnormは保存されていない。
- task50終了まで4腕の保存された全21種の形状統計がbit一致。task51最初、ラベル生成前にW2全体をc倍し、そのparam groupのlrを0にする。W1,b1,b2は引き続き学ぶ。各unitの読み出し方向は固定。
- Adam lr=.001, betas=(.9,.999), eps=1e-8。task間にも凍結時にもmomentをリセットしない。W2のgrad計算とmoment更新はlr=0後も続くが、W2値は動かない。保存されたtask100/200の全checkpointでnorm一定、task50 normのc倍にfloat32保存誤差内で一致した。
- sw=task開始、mid=+500、end=+4000。すべてのtaskでsw統計は前task endとbit一致する。swの変化は入力切替によるものではない。
- 原表はtask151–200の各taskで「sw時点にk>0のunitの中央値」を取り、その50task×2seedを算術平均。全unit-taskの一括中央値ではない。vはtask200全unit中央値を2seed平均。μ+だけはendのopen→closedイベント数をopen unit機会数で割る（task151→152から199→200）。
- sはtorch.std既定の標本標準偏差(ddof=1)。本体はtorch.medianの小さい方の中央値（1200標本の600番目）。q99/q95/q90は降順12/60/120番目で、補間分位点ではない。population SDへの変換差は約0.042%であり結論に影響しない。

## 原表の再現（生存unitを選んだ各task中央値の平均）

"""
    report += table(original, ["arm", "seed", "raw_top", "top_over_s", "tail", "sample_sd", "k_end", "ce_end", "v_norm_t200"]) + "\n\n"
    report += table(pooled, ["arm", "q99_above_body_over_s", "q95_above_body_over_s", "q90_above_body_over_s", "q99_positive_fraction", "net_open_count_loss", "mu_plus_unconditional_units"]) + "\n\n"
    report += """§8.36の主要な丸め値を再現した。raw topはvが100倍違う両端で約16倍違い、正確な逆比例ではない。終端損失も一致していないので、同じ損失/同じlogitでの逆比例恒等式をこの介入の説明として当てはめられない。

「上位1–10%の分位は同じ」は近似的な表現であり完全一致ではない。q99の本体からの距離/sにもc=.1から10で変化があるが、maxの変化より小さい。またq99が常に折れ目・常に閉じた入力とも限らない（上表q99_positive_fraction）。保存値だけでは「変化するのは開いた入力だけ」と厳密には断定できない。

## unit選択を揃えた感度解析

all_unitsは生死によらず100unit、common_frozen_switch_aliveは各taskで3凍結腕すべて生存、fixed_task50_aliveは介入直前の固定集合、always_aliveはtask151–200の3凍結腕すべてで生存していた固定集合。最後の集合は将来情報で選ぶ記述的感度解析で、因果効果の推定対象にはしない。

"""
    report += table(cohorts, ["arm", "cohort", "units", "top_over_s", "tail", "q99_above_body_over_s", "q95_above_body_over_s", "q90_above_body_over_s"]) + "\n\n"
    report += """同じunitを比べてもtail=(max−body)/sの差は残った。全unitの対応比較では、c=.1のtailがc=10より大きいunit-task割合はseed0で97.08%、seed1で95.70%、差の中央値は1.030/1.042 s。共通生存unitでも98.02%/96.95%で、差の中央値は1.068/1.099 s。生存unitの入替えだけで原表の差を説明することはできない（詳細はsaved_paired_differences.csv）。

tailは正のaffine変換で不変な統計なので、前活性分布の一様な拡大・平行移動だけではこの差を説明できない。固定2点のスカラー模型ではtailは変わらず、本実験のこの部分を再現できない。なお50taskすべて・3腕すべて生存の固定集合はseed0で1unit、seed1で0unitしかなく、その感度解析は情報不足（pooled値nan）。

## 誤差の比較

deepはこの保存定義ではz≤−8（−3ではない）。各unitでその群の入力について||p−y||を平均し、課題末にopen inputを持つunitの中央値、その後2seed平均。入力集合は介入間・unit間で異なる。同じ入力IDに条件づけた比較でも、勾配雑音の分散を直接測ったものでもない。

"""
    report += table(errorrows, ["arm", "task", "v_norm", "all_mean_error_norm", "deep_mean_error_norm", "open_mean_error_norm", "alive_units"]) + "\n\n"
    report += """## 機構の推論に関する制限

1. 保存されたものはorder統計、k、CE、accuracy、群別需要/誤差norm、W2列norm。W1/W2の全ベクトル、各入力のz/e、Adam moments、開いた入力や最大入力のIDは無い。従って各入力がどの高さからどこへ行ったか、閉じた/開いた入力の入替え、残差履歴とAdam分母の遅れをこのデータから復元できない。
2. 「押しで失う開いた入力の割合」は実装上1−k_mid/k_swという**個数の正味減少率**。最初に開いていた集合のうち何割が閉じたかではない。別の入力が開けば相殺される。midは500更新後で、その間にも当てはめがある。「純粋な押し」に外科的に分離されていない。
3. vをスカラー倍しても、各クラス残差は変わる。またtask51では既存のAdam momentsをc/c²倍せず残すので、「vがAdamの分子分母で厳密に消える」はこの介入過渡の記述として成り立たない。相殺が成り立つのは同じ残差/活性化/勾配履歴を全期間同じ倍率にした対応系（εの処理も必要）である。
4. 凍結3腕同士は大きさを操作した介入である。一方EEQとVF1の差はvの方向変化だけでなくnorm変化も止める介入なので、その差だけで「向きの学び直し」の寄与を単独で同定できない。
5. スカラーlogisticは絶対高さと残差減衰の制御例として有用だが、この保存実験は多クラス・複数unit・同じwを共有する多数入力・出力bias学習・持ち越しmoments・課題切替がある。損失が1unitの目標logitを指定するという説明はできない。必要な追加機構は、入力間の相対配置と誤差が作る更新方向の変化である。

## 再実行

`python analysis/logistic_v_height_0929/saved_audit.py --out results/logistic_v_height_0929`

source SHA256、解析SHA256、利用可能データはsaved_provenance.json。すべて読み取りのみでsourceを変更しない。
"""
    (args.out / "saved_report.md").write_text(report)
    print(table(pooled, ["arm", "raw_top", "top_over_s", "tail", "sample_sd", "ce_end", "v_norm_t200"]))


if __name__ == "__main__":
    main()
