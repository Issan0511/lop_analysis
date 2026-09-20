#!/usr/bin/env python3
"""Score the registered erosion_race_0919 experiment without changing its thresholds.

Input: one completed result root; output: verdict.json, summary.md, figure.png.
The engine's CSVs are observations, never independent per-unit replications.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics as st

SEEDS = (1001, 1002, 1003, 1004, 1005)
BASE = ("R", "GELU", "ELU", "LR", "LK001", "SNA")
JOBS = tuple(f"{a}/base" for a in BASE) + ("GELU/early", "GELU/late", "ELU/early")
DIAG_STEPS = (0, 1, 10, 30, 75, 150, 300, 750, 1500, 3000, 5000, 7500,
              10000, 15000, 22500, 30000)
REPO = Path(__file__).resolve().parents[2]


def finite(value, context="number"):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite {context}: {value!r}")
    return result


def first(row, names, default=None):
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]
    if default is not None:
        return default
    raise ValueError(f"Missing one of {names}; available={tuple(row)}")


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_runs(root):
    """Normalize the engine schema; refuse to silently score partial main runs."""
    runs, sources = {}, []
    for path in sorted(root.glob("*/diagnostics.csv")):
        rows = read_csv(path)
        if not rows:
            continue
        job = f"{rows[0]['arm']}/{rows[0]['mode']}"
        if job not in JOBS:
            continue
        if job in runs:
            raise ValueError(f"Duplicate job {job}: {path}")
        diagnostic, flux, per_task = {}, {}, {}
        seen = set()
        for raw in rows:
            if f"{raw['arm']}/{raw['mode']}" != job:
                raise ValueError(f"Mixed jobs in {path}")
            seed, task, step = (int(raw[k]) for k in ("seed", "task", "step"))
            if task == 0:  # Optional initial snapshot is not a new-label task diagnosis.
                continue
            if seed not in SEEDS or task not in (1, 2):
                raise ValueError(f"Unexpected main-run seed/task in {path}: {(seed, task)}")
            identity = (seed, task, step)
            if identity in seen:
                raise ValueError(f"Duplicate diagnosis {job} {identity}")
            seen.add(identity)
            row = {"seed": seed, "task": task, "step": step,
                   "acc": finite(raw["acc"], f"{job} acc"),
                   "V_l1": finite(raw["output_sd_ratio_l1"], f"{job} V_l1"),
                   "V_l2": finite(raw["output_sd_ratio_l2"], f"{job} V_l2")}
            diagnostic.setdefault((seed, task), []).append(row)
            start = int(raw["window_start"])
            if int(raw["window_steps"]) != step - start:
                raise ValueError(f"Inconsistent diagnostic window in {path}: {identity}")
            f = {"start_step": start, "end_step": step}
            for key in ("down", "up", "conf", "label", "hist", "correction"):
                f[key] = finite(raw[key], f"{job} {key}")
            f["residual"] = finite(raw["roundoff"], f"{job} roundoff")
            if f["down"] < 0 or f["up"] < 0:
                raise ValueError(f"Negative rectified flux in {path}: {identity}")
            if step > 0:
                flux.setdefault((seed, task), []).append(f)
        task_path = path.parent / "per_task.csv"
        for raw in read_csv(task_path):
            seed, task = int(raw["seed"]), int(raw["task"])
            if (seed, task) in per_task:
                raise ValueError(f"Duplicate task endpoint in {task_path}: {(seed, task)}")
            if str(raw["complete_task"]).lower() not in ("true", "1") or int(raw["steps"]) != 30000:
                raise ValueError(f"Incomplete task in {task_path}: {(seed, task)}")
            per_task[(seed, task)] = {"online_acc": finite(raw["online_acc"])}
        expected = {(s, t) for s in SEEDS for t in (1, 2)}
        if set(diagnostic) != expected or set(per_task) != expected:
            raise ValueError(f"Incomplete seed/task coverage: {job}")
        for identity, group in diagnostic.items():
            group.sort(key=lambda r: r["step"])
            if tuple(r["step"] for r in group) != DIAG_STEPS:
                raise ValueError(f"Incomplete/changed diagnostic schedule: {job} {identity}")
            flux[identity].sort(key=lambda r: r["end_step"])
        runs[job] = {"diagnostics": diagnostic, "flux": flux, "per_task": per_task}
        sources.extend((path, task_path))
        provenance = path.parent / "provenance.json"
        if provenance.exists():
            sources.append(provenance)
    if set(runs) != set(JOBS):
        raise ValueError(f"Expected all 9 completed jobs; missing={sorted(set(JOBS) - set(runs))}")
    return runs, sources


def event_time(rows, key, predicate):
    """Return interval-observed onset, requiring two *within-task* diagnoses.

    A final isolated threshold crossing is reported as pending, not as an event.
    Initial (step zero) onset is left-censored. No missing event becomes infinity.
    """
    rows = sorted(rows, key=lambda r: r["step"])
    if not rows:
        raise ValueError("Cannot score time order without diagnostic observations")
    for index in range(len(rows) - 1):
        if predicate(rows[index][key]) and predicate(rows[index + 1][key]):
            current = rows[index]["step"]
            return {
                "status": "left_censored" if index == 0 else "interval_observed",
                "first_crossing_step": current,
                "confirmed_at_step": rows[index + 1]["step"],
                "onset_interval": {"lower_exclusive": None if index == 0 else rows[index - 1]["step"],
                                   "upper_inclusive": current},
                "observed_through_step": rows[-1]["step"],
            }
    return {"status": "right_censored", "first_crossing_step": None,
            "confirmed_at_step": None, "onset_interval": None,
            "observed_through_step": rows[-1]["step"],
            "pending_single_crossing_step": rows[-1]["step"] if predicate(rows[-1][key]) else None}


def event_reached(event):
    return event["status"] != "right_censored"


def causal_comparison(left, right, success_label, failure_label):
    records = [{"seed": seed, "left": left[seed], "right": right[seed],
                "difference": left[seed] - right[seed]} for seed in SEEDS]
    differences = [r["difference"] for r in records]
    median = st.median(differences)
    positive = sum(d > 0 for d in differences)
    return {"label": success_label if median >= .10 and positive >= 4 else failure_label,
            "median_difference": median, "positive_seeds": positive, "n": len(records),
            "criterion": "median difference >= 0.10 and >= 4 of 5 differences > 0",
            "pairs": records}


def endpoint(runs, job, task, metric):
    result = {}
    for seed in SEEDS:
        if metric == "online_acc":
            result[seed] = runs[job]["per_task"][(seed, task)][metric]
        else:
            rows = runs[job]["diagnostics"][(seed, task)]
            result[seed] = next(r[metric] for r in rows if r["step"] == 30000)
    return result


def early_flux(runs, job, seed):
    rows = runs[job]["flux"][(seed, 1)]
    chosen = [r for r in rows if r["end_step"] <= 750 and r["end_step"] > 0]
    if not chosen or chosen[0]["start_step"] != 0 or chosen[-1]["end_step"] != 750:
        raise ValueError(f"Missing full flux interval (0,750]: {job}, seed={seed}")
    for left, right in zip(chosen, chosen[1:]):
        if left["end_step"] != right["start_step"]:
            raise ValueError(f"Gap or overlap in early flux: {job}, seed={seed}")
    keys = ("down", "up", "conf", "label", "hist", "correction", "residual")
    out = {k: sum(r[k] for r in chosen) for k in keys}
    out["net_downward"] = out["down"] - out["up"]
    out["signed_actual"] = out["up"] - out["down"]
    out["signed_components"] = sum(out[k] for k in ("conf", "label", "hist", "correction"))
    out["closure_error_after_residual"] = out["signed_actual"] - out["signed_components"] - out["residual"]
    return out


def score(runs):
    labels = {}
    definitions = (
        ("C1", "GELU/early", "GELU/base", 1, "acc", "EARLY_HELPS", "NO_CLEAR_RESCUE"),
        ("C2", "GELU/early", "GELU/late", 1, "acc", "EARLY_ADVANTAGE", "NO_EARLY_ADVANTAGE"),
        ("C3", "GELU/early", "GELU/base", 2, "online_acc", "PERSISTENT_HELP", "NO_PERSISTENT_HELP"),
        ("C4", "ELU/early", "ELU/base", 1, "acc", "ELU_EARLY_HELPS", "NO_CLEAR_ELU_RESCUE"),
    )
    for key, left, right, task, metric, yes, no in definitions:
        labels[key] = causal_comparison(endpoint(runs, left, task, metric),
                                        endpoint(runs, right, task, metric), yes, no)
        labels[key].update(left_job=left, right_job=right, task=task, metric=metric)
    flux = {job: {str(seed): early_flux(runs, job, seed) for seed in SEEDS} for job in JOBS}
    origins = {}
    for arm in ("R", "GELU"):
        records = []
        for seed in SEEDS:
            f = flux[f"{arm}/base"][str(seed)]
            negatives = sum(max(-f[k], 0.0) for k in ("conf", "label", "hist"))
            yes = f["conf"] < 0 and -f["conf"] > .5 * negatives
            records.append({"seed": seed, **f, "negative_component_magnitude": negatives,
                            "conf_negative_fraction": max(-f["conf"], 0.0) / negatives if negatives else None,
                            "criterion_met": yes})
        origins[arm] = {"meeting_seeds": sum(r["criterion_met"] for r in records), "pairs": records}
    labels["M1"] = {"label": "CONF_NEGATIVE_DOMINANT" if all(v["meeting_seeds"] >= 4 for v in origins.values())
                      else "NOT_ESTABLISHED", "task": 1, "step_interval": [1, 750], "arms": origins}
    comparisons = {}
    for other in ("LR", "SNA"):
        pairs = []
        for seed in SEEDS:
            gelu, ref = flux["GELU/base"][str(seed)], flux[f"{other}/base"][str(seed)]
            less_down = gelu["down"] - ref["down"]
            more_up = ref["up"] - gelu["up"]
            delta_net = gelu["net_downward"] - ref["net_downward"]
            pairs.append({"seed": seed, "gelu_E": gelu["down"], "other_E": ref["down"],
                          "gelu_R": gelu["up"], "other_R": ref["up"],
                          "delta_net_erosion": delta_net, "less_downward_component": less_down,
                          "more_upward_component": more_up,
                          "decomposition_residual": delta_net - less_down - more_up})
        dn = st.mean(r["delta_net_erosion"] for r in pairs)
        down = st.mean(r["less_downward_component"] for r in pairs)
        up = st.mean(r["more_upward_component"] for r in pairs)
        label = ("NO_LOWER_NET_EROSION" if dn <= 0 else "LESS_DOWNWARD" if down >= 2 * dn / 3
                 else "MORE_UPWARD" if up >= 2 * dn / 3 else "MIXED")
        comparisons[other] = {"label": label, "mean_delta_net_erosion": dn,
                              "mean_less_downward_component": down, "mean_more_upward_component": up,
                              "mean_decomposition_residual": dn - down - up, "pairs": pairs}
    labels["M2"] = {"task": 1, "step_interval": [1, 750], "comparisons": comparisons}
    order = {}
    for job in JOBS:
        records = []
        for seed in SEEDS:
            for task in (1, 2):
                rows = runs[job]["diagnostics"][(seed, task)]
                response = event_time(rows, "V_l1", lambda x: x <= .1)
                fit = event_time(rows, "acc", lambda x: x >= .5)
                observed_order = "CENSORED"
                if event_reached(response) and event_reached(fit):
                    a, b = response["first_crossing_step"], fit["first_crossing_step"]
                    observed_order = "RESPONSE_FIRST" if a < b else "FIT_FIRST" if b < a else "SAME_DIAGNOSTIC"
                record = {"seed": seed, "task": task, "response_l1": response, "fit": fit,
                          "first_observed_order": observed_order}
                if all("V_l2" in r for r in rows):
                    record["response_l2_auxiliary"] = event_time(rows, "V_l2", lambda x: x <= .1)
                records.append(record)
        order[job] = records
    labels["M3"] = {"primary_layer": 1, "V_reference": "initial task-1 weights; exclude initial sd <= 1e-6",
                    "consecutive_diagnostics": 2, "records": order}
    fit_counts = {arm: sum(r["task"] == 1 and event_reached(r["fit"]) for r in order[f"{arm}/base"])
                  for arm in ("ELU", "GELU")}
    predictions = (
        ("P1", "C1 = EARLY_HELPS", .60, labels["C1"]["label"] == "EARLY_HELPS"),
        ("P2", "C2 = EARLY_ADVANTAGE", .65, labels["C2"]["label"] == "EARLY_ADVANTAGE"),
        ("P3", "C3 does not meet persistent-improvement criterion", .65,
         labels["C3"]["label"] == "NO_PERSISTENT_HELP"),
        ("P4", "M1 = CONF_NEGATIVE_DOMINANT", .65, labels["M1"]["label"] == "CONF_NEGATIVE_DOMINANT"),
        ("P5", "M2 GELU vs LR = LESS_DOWNWARD", .55, comparisons["LR"]["label"] == "LESS_DOWNWARD"),
        ("P6", "M2 GELU vs SNA = LESS_DOWNWARD", .55, comparisons["SNA"]["label"] == "LESS_DOWNWARD"),
        ("P7", "task1 fit reached: ELU >= 4/5, GELU <= 1/5", .80,
         fit_counts["ELU"] >= 4 and fit_counts["GELU"] <= 1),
    )
    pred = {key: {"claim": claim, "probability": p, "hit": bool(hit), "brier": (p - int(hit)) ** 2}
            for key, claim, p, hit in predictions}
    pred["_summary"] = {"n": 7, "hits": sum(v["hit"] for v in pred.values()),
                        "brier": st.mean(v["brier"] for v in pred.values()), "P7_fit_counts": fit_counts}
    primary, origin = labels["C1"]["label"] == "EARLY_HELPS", labels["M1"]["label"] == "CONF_NEGATIVE_DOMINANT"
    overall = ("NARROW_INTERVENTION_VERSION_SUPPORTED" if primary and origin else
               "EARLY_INTERVENTION_HELPS_SOURCE_NOT_ESTABLISHED" if primary else "5000_STEP_GRACE_NOT_SUPPORTED")
    outcomes = {job: {"task1_end_accuracy": endpoint(runs, job, 1, "acc"),
                      "task2_end_accuracy": endpoint(runs, job, 2, "acc"),
                      "task1_online_accuracy": endpoint(runs, job, 1, "online_acc"),
                      "task2_online_accuracy": endpoint(runs, job, 2, "online_acc")}
                for job in JOBS}
    return {"experiment": "erosion_race_0919", "seeds": list(SEEDS), "labels": labels,
            "overall": overall, "predictions": pred, "early_flux": flux, "outcomes": outcomes,
            "limitations": ["2 tasks only; fixed random-label fitting, not generalization",
                            "5 paired seeds; registered descriptive thresholds, not a significance test",
                            "flux probes the fixed initially-positive pairs of 64 training images",
                            "signed Adam contributions are attribution on one realized trajectory",
                            "M3 ordering alone is not causal evidence",
                            "ELU task-1 intervention can be ceiling-limited"]}


def fmt(value, digits=5):
    if value is None:
        return "—"
    return f"{value:.{digits}g}" if isinstance(value, (int, float)) else str(value)


def event_text(event):
    if event["status"] == "right_censored":
        pending = event.get("pending_single_crossing_step")
        return f"未到達・右打切り({event['observed_through_step']})" + (f"; {pending}単発・未確認" if pending is not None else "")
    interval = event["onset_interval"]
    left = interval["lower_exclusive"]
    onset = f"≤{interval['upper_inclusive']}・左打切り" if left is None else f"({left}, {interval['upper_inclusive']}]"
    return f"{onset}; 確認{event['confirmed_at_step']}"


def markdown(verdict):
    labels = verdict["labels"]
    lines = ["# erosion_race_0919 — 沈降と新ラベル学習の競争", "",
             f"登録総合判定: **{verdict['overall']}**。主判定C1: **{labels['C1']['label']}**。", "",
             "raw RL-CIFAR・2課題・各30,000更新・seed 1001–1005。第1層の初期正側probeの輸送と、新ラベルへの適合を追跡した。", "",
             "判定は事前登録の中央値差≥0.10かつ4/5以上が正という記述基準。unitを独立標本にした検定ではない。", "",
             "## 機能の終点", "", "| 腕 | 課題1末acc | 課題1 online | 課題2末acc | 課題2 online |",
             "|---|---:|---:|---:|---:|"]
    for job, row in verdict["outcomes"].items():
        keys = ("task1_end_accuracy", "task1_online_accuracy", "task2_end_accuracy", "task2_online_accuracy")
        lines.append("| " + job + " | " + " | ".join(fmt(st.median(row[k].values())) for k in keys) + " |")
    lines += ["", "## 登録因果判定 C1–C4", "",
              "early介入は課題1の1–5000更新、lateは5001–10000更新。課題末の評価は介入解除後。課題2は全腕介入なし。", "",
              "| 判定 | 比較 | 読み出し | 中央値差 | 正のseed | 結果 |", "|---|---|---|---:|---:|---|"]
    for key in ("C1", "C2", "C3", "C4"):
        r = labels[key]
        lines.append(f"| {key} | {r['left_job']} − {r['right_job']} | task{r['task']} {r['metric']} | {fmt(r['median_difference'])} | {r['positive_seeds']}/5 | {r['label']} |")
    lines += ["", "全seedの対応差（省略なし）:", "", "| 判定 | seed | 左 | 右 | 左−右 |", "|---|---:|---:|---:|---:|"]
    for key in ("C1", "C2", "C3", "C4"):
        for r in labels[key]["pairs"]:
            lines.append(f"| {key} | {r['seed']} | {fmt(r['left'])} | {fmt(r['right'])} | {fmt(r['difference'])} |")
    lines += ["", "ELUのC4は対照が課題1で既に高精度なら天井効果を受ける。C1不成立でも競争仮説全般の棄却とはしない。", "",
              "## M1 — 符号付き寄与の起源", "", f"**{labels['M1']['label']}**。課題1の更新1–750。", "",
              "C/L/Hはconfidence/label/過去課題履歴の符号付き寄与。Cを一律に下向き、Lを一律に回復とは命名しない。実net=R−E。寄与単独のAdam反実仮想ではない。", "",
              "| 腕 | seed | C | L | H | 実net | closure残差 | 負成分中C割合 | 基準 |", "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for arm, group in labels["M1"]["arms"].items():
        for r in group["pairs"]:
            lines.append(f"| {arm} | {r['seed']} | {fmt(r['conf'])} | {fmt(r['label'])} | {fmt(r['hist'])} | {fmt(r['signed_actual'])} | {fmt(r['residual'])} | {fmt(r['conf_negative_fraction'])} | {'達成' if r['criterion_met'] else '未達'} |")
    lines += ["", "## M2 — 実際の下向き量と上向き量", "",
              "E=Σ mean max(−Δz/σ初期,0)、R=Σ mean max(Δz/σ初期,0)。maxは実総更新へ適用する。初期正側だった同じ対を固定追跡し、現在の正側集合への出入りで母集団を変えない。", "",
              "ΔN=(E_GELU−E_other)+(R_other−R_GELU)。下表の集約は**算術平均**で、二項分解の加法性を保つ。", "",
              "| GELU対 | 平均ΔN | 下向き差 | 上向き差 | 判定 |", "|---|---:|---:|---:|---|"]
    for arm, r in labels["M2"]["comparisons"].items():
        lines.append(f"| {arm} | {fmt(r['mean_delta_net_erosion'])} | {fmt(r['mean_less_downward_component'])} | {fmt(r['mean_more_upward_component'])} | {r['label']} |")
    lines += ["", "| 比較 | seed | E_G | E_other | R_G | R_other | ΔN | 下向き差 | 上向き差 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for arm, group in labels["M2"]["comparisons"].items():
        for r in group["pairs"]:
            keys = ("gelu_E", "other_E", "gelu_R", "other_R", "delta_net_erosion", "less_downward_component", "more_upward_component")
            lines.append(f"| {arm} | {r['seed']} | " + " | ".join(fmt(r[k]) for k in keys) + " |")
    lines += ["", "## M3 — 応答低下と適合の時間順", "",
              "主読み出しVは第1層: unitごとの活性sd/初期sdの中央値（初期sd≤1e−6を除く）。課題2も同じ初期基準。V≤.1とacc≥.5をそれぞれ連続2診断で確認する。", "",
              "区間は最初の該当診断と直前診断の間。step0から成立は左打切り。未到達は右打切りとして観測末を記し、∞とは置かない。課題をまたいで連続2点をつなげない。第2層の副指標はverdict.jsonに保存。", "",
              "| 腕 | seed | 課題 | 第1層V低下の到達区間 | 適合の到達区間 | 最初の観測順 |", "|---|---:|---:|---|---|---|"]
    for job, records in labels["M3"]["records"].items():
        for r in records:
            lines.append(f"| {job} | {r['seed']} | {r['task']} | {event_text(r['response_l1'])} | {event_text(r['fit'])} | {r['first_observed_order']} |")
    lines += ["", "## 事前予測の採点", "", "| key | 事前予測 | p | 的中 |", "|---|---|---:|---|"]
    for key, r in verdict["predictions"].items():
        if key != "_summary":
            lines.append(f"| {key} | {r['claim']} | {r['probability']:.2f} | {'yes' if r['hit'] else 'no'} |")
    total = verdict["predictions"]["_summary"]
    lines += ["", f"Codex: **{total['hits']}/7**, Brier **{total['brier']:.4f}**。P7の課題1適合到達数: {total['P7_fit_counts']}。", "",
              "## 解釈の限界", "", "- 2課題・5 seedでの固定ランダムラベル適合であり、50課題の帰結や未見画像への汎化は判定していない。",
              "- 介入は実軌道のconfidence由来の負方向平均更新を取り除く。画像間差にも作用し得るため、平均だけの純粋な媒介操作とは主張しない。",
              "- M1/M2のprobeは64枚の初期正側対に限定。全1200画像の厳密輸送ではない。",
              "- M3は時間順の観測であり、単独では因果を示さない。応答閾値は微分の非ゼロ性そのものではない。", "",
              "![Registered experiment diagnostics](figure.png)", ""]
    return "\n".join(lines)


def plot_results(runs, verdict, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    colors = {"R": "#71717a", "GELU": "#dc2626", "ELU": "#d97706", "LR": "#2563eb",
              "LK001": "#7c3aed", "SNA": "#059669"}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 180, "axes.titleweight": "bold"})
    fig, axs = plt.subplots(4, 2, figsize=(15, 15), constrained_layout=True)

    def trajectories(ax, jobs, task, metric, ylabel, title):
        for job in jobs:
            arm, iv = job.split("/")
            data = [[r[metric] for r in runs[job]["diagnostics"][(seed, task)]] for seed in SEEDS]
            steps = [r["step"] for r in runs[job]["diagnostics"][(SEEDS[0], task)]]
            array = np.asarray(data)
            color = colors[arm] if iv == "base" else ("#0891b2" if iv == "early" else "#7c3aed")
            for values in array:
                ax.plot(steps, values, color=color, alpha=.13, lw=.7)
            ax.plot(steps, np.median(array, axis=0), color=color, label=job, lw=2)
        ax.set_xscale("symlog", linthresh=10)
        if metric.startswith("V"):
            ax.set_yscale("symlog", linthresh=.01)
            ax.axhline(.1, color="#777", ls=":", lw=1)
        else:
            ax.set_ylim(0, 1.025)
            ax.axhline(.5, color="#777", ls=":", lw=1)
        ax.set(xlabel=f"Task {task} update", ylabel=ylabel, title=title)
        ax.grid(alpha=.15)

    base_jobs = [f"{a}/base" for a in BASE]
    trajectories(axs[0, 0], base_jobs, 1, "acc", "Full-set accuracy", "A  First-task label fitting")
    axs[0, 0].legend(ncol=3, fontsize=8)
    trajectories(axs[0, 1], base_jobs, 1, "V_l1", "Activation-sd ratio V (initial reference)", "B  First-layer response")
    trajectories(axs[1, 0], ("GELU/base", "GELU/early", "GELU/late"), 1, "acc", "Full-set accuracy", "C  Temporary intervention, then release")
    axs[1, 0].axvspan(1, 5000, alpha=.05, color="#0891b2")
    axs[1, 0].axvspan(5001, 10000, alpha=.05, color="#7c3aed")
    axs[1, 0].legend(fontsize=8)
    trajectories(axs[1, 1], ("GELU/base", "GELU/early", "GELU/late", "ELU/base", "ELU/early"), 2, "acc", "Full-set accuracy", "D  New task; all interventions off")
    axs[1, 1].legend(fontsize=8, ncol=2)
    ax = axs[2, 0]
    groups = verdict["labels"]["M1"]["arms"]
    for j, key in enumerate(("conf", "label", "hist", "signed_actual")):
        vals = [st.mean(r[key] for r in groups[a]["pairs"]) for a in ("R", "GELU")]
        x = np.arange(2) + (j - 1.5) * .18
        ax.bar(x, vals, width=.17, label=key, alpha=.8)
        for i, arm in enumerate(("R", "GELU")):
            ax.scatter(np.repeat(x[i], 5), [r[key] for r in groups[arm]["pairs"]], s=9, color="#222", alpha=.6)
    ax.axhline(0, color="#777", lw=.8)
    ax.set_xticks(range(2), ("R", "GELU"))
    ax.set(title="E  Signed attribution: task 1, updates 1–750", ylabel="Initial-sd units (fixed positive probe)")
    ax.legend(fontsize=8, ncol=2)
    ax = axs[2, 1]
    comps = verdict["labels"]["M2"]["comparisons"]
    for j, (key, label) in enumerate((("mean_less_downward_component", "E_G − E_other"),
                                      ("mean_more_upward_component", "R_other − R_G"),
                                      ("mean_delta_net_erosion", "Delta net = sum"))):
        ax.bar(np.arange(2) + (j - 1) * .23, [comps[a][key] for a in ("LR", "SNA")], width=.22, label=label)
    ax.axhline(0, color="#777", lw=.8)
    ax.set_xticks(range(2), ("GELU vs LR", "GELU vs SNA"))
    ax.set(title="F  Actual down/up flux difference", ylabel="Paired arithmetic mean (initial-sd units)")
    ax.legend(fontsize=8)
    ax = axs[3, 0]
    for j, key in enumerate(("C1", "C2", "C3", "C4")):
        vals = [r["difference"] for r in verdict["labels"][key]["pairs"]]
        ax.scatter(j + np.linspace(-.07, .07, len(vals)), vals, s=30, color="#2563eb", alpha=.75)
        ax.plot([j - .18, j + .18], [st.median(vals)] * 2, color="#111", lw=2)
    ax.axhline(0, color="#777", lw=.8)
    ax.axhline(.1, color="#777", ls=":", lw=1)
    ax.set_xticks(range(4), ("C1: G early-base", "C2: G early-late", "C3: task2 online", "C4: E early-base"), rotation=13)
    ax.set(title="G  Every registered paired effect", ylabel="Accuracy difference (5 paired seeds)")
    ax = axs[3, 1]
    for j, arm in enumerate(BASE):
        f = verdict["early_flux"][f"{arm}/base"]
        ax.bar(j - .17, st.mean(f[str(s)]["down"] for s in SEEDS), width=.32, color="#dc2626", label="Down E" if j == 0 else None)
        ax.bar(j + .17, st.mean(f[str(s)]["up"] for s in SEEDS), width=.32, color="#2563eb", label="Up R" if j == 0 else None)
    ax.set_xticks(range(len(BASE)), BASE)
    ax.set(title="H  Both directions are measured", ylabel="Actual pointwise flux, updates 1–750")
    ax.legend(fontsize=8)
    fig.suptitle("erosion_race_0919  |  5 paired seeds, 2 tasks\nThin lines / dots: seeds; trajectories: seed median; flux bars: arithmetic mean", fontsize=15)
    fig.savefig(output)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_root", type=Path, nargs="?", default=REPO / "results/erosion_race_0919")
    parser.add_argument("--out", type=Path, help="Output directory (defaults to result root)")
    args = parser.parse_args()
    runs, sources = load_runs(args.result_root)
    verdict = score(runs)
    verdict["source_files"] = [{"path": str(p.relative_to(args.result_root)),
                                 "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in sources]
    out = args.out or args.result_root
    out.mkdir(parents=True, exist_ok=True)
    plot_results(runs, verdict, out / "figure.png")
    (out / "verdict.json").write_text(json.dumps(verdict, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (out / "summary.md").write_text(markdown(verdict), encoding="utf-8")
    print(json.dumps({"overall": verdict["overall"], "C1": verdict["labels"]["C1"]["label"],
                      "M1": verdict["labels"]["M1"]["label"], "prediction_score": verdict["predictions"]["_summary"],
                      "outputs": [str(out / name) for name in ("verdict.json", "summary.md", "figure.png")]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
