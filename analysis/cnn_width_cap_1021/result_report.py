#!/usr/bin/env python3
"""Read completed saved outputs only; independently audit and report CNN width-cap results.

No torch import, model forward, checkpoint execution, training, or GPU use. Incomplete or
inconsistent arms are refused before any scientific result is written.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ARMS = ("R_lr0.0003", "ELU_a3.6_lr0.0003")
LABELS = {ARMS[0]: "ReLU", ARMS[1]: "ELU α=3.6"}
SEEDS = tuple(range(2000, 2010))
SLOTS = [(arm, seed) for arm in ARMS for seed in SEEDS]
SITES = ("c1", "c2", "f1", "f2")
WIDTHS = (16, 16, 100, 100)
EARLY, LATE = (3, 5, 7, 9), (21, 23, 25, 27, 29)
METRICS = ("first100_online_acc", "first780_online_acc", "online_acc", "train_acc", "test_acc", "ce_mean", "ce_first780_mean")
SPEC = ROOT / "specs/spec_cnn_width_cap_1021.md"


def resolve_input(path):
    path = Path(path)
    if path.exists():
        return path
    manifest = ROOT / "results/cnn_width_cap_1021/backup_manifest.json"
    relative = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else None
    entries = json.loads(manifest.read_text())["files"]
    entry = next(x for x in entries if x["source"] == str(path) or
                 (relative is not None and x["source_relative"] == relative))
    archived = Path(entry["backup"])
    require(archived.stat().st_size == entry["bytes"] and
            digest(archived) == entry["sha256"], f"archive hash differs: {path}")
    return archived


def digest(path):
    h = hashlib.sha256()
    with resolve_input(path).open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(resolve_input(path).read_text())


def read_csv(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_number(value):
    x = float(value)
    return x if math.isfinite(x) else None


def safe_median(values):
    a = np.asarray(values, dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(np.median(a)) if a.size else None


def mean_valid(values):
    return float(np.mean(values)) if values and all(x is not None and math.isfinite(x) for x in values) else None


def ratio_or_none(numerator, denominator):
    return float(numerator / denominator) if denominator > 0 else None


def show(x, places=4):
    return "未定義" if x is None or not math.isfinite(float(x)) else f"{x:.{places}f}"


def table(headers, rows):
    return ["| " + " | ".join(map(str, headers)) + " |", "| " + " | ".join("---" for _ in headers) + " |"] + [
        "| " + " | ".join(map(str, row)) + " |" for row in rows]


def exact_sign(values):
    positive, negative = sum(x > 0 for x in values), sum(x < 0 for x in values)
    n = positive + negative
    lower_tail = sum(math.comb(n, i) for i in range(min(positive, negative) + 1))
    return {"positive": positive, "negative": negative, "zero": len(values) - n,
            "p": min(1.0, 2 * lower_tail / (2**n)) if n else 1.0}


def close(a, b, name):
    require(math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-14), f"primary mismatch: {name}: {a} vs {b}")


def load_and_audit(base, primary_path):
    """Require completion before reading outcomes; reject config/provenance inconsistencies."""
    provenance = {cond: read_json(base / cond / "provenance.json") for cond in ("natural", "cap")}
    for cond, prov in provenance.items():
        require(prov.get("status") == "COMPLETED", f"{cond} is {prov.get('status')}; both completed arms are required")
    natural_meta, cap_meta = (provenance[c]["meta"] for c in ("natural", "cap"))
    # These are the only intentional run-meta differences. Paths, selected slot order,
    # source/calibration hashes, dataset, batches, probe, spec, code, runtime all must match.
    require({k: v for k, v in natural_meta.items() if k != "condition"} ==
            {k: v for k, v in cap_meta.items() if k != "condition"}, "cross-arm run metadata differ")
    required_meta = ("source_sha256", "source_path", "source_meta", "source_indices", "calibration_sha256",
                     "calibration_path", "data_sha256", "data_path", "batch_sha256", "probe_sha256", "spec_sha256",
                     "sources_sha256", "initial_runtime_precision", "slots", "budget", "graph")
    for key in required_meta:
        require(key in natural_meta, f"missing provenance field: {key}")
    for cond, prov in provenance.items():
        meta = prov["meta"]
        require(meta["condition"] == cond and meta["budget"] == 7800, f"{cond}: condition/budget differ")
        require([tuple(s) for s in meta["slots"]] == SLOTS, f"{cond}: wrong slot order")
        require(prov.get("env") == meta["initial_runtime_precision"], f"{cond}: initial environment was overwritten")
    require(digest(SPEC) == natural_meta["spec_sha256"], "current frozen spec hash differs from run")
    for cond, prov in provenance.items():
        run_commit = prov["gits"][0]["git_hash"]
        for path, expected in natural_meta["sources_sha256"].items():
            data = subprocess.run(["git", "-C", str(ROOT), "show", run_commit + ":" + path],
                                  capture_output=True, check=True).stdout
            require(hashlib.sha256(data).hexdigest() == expected,
                    f"{cond} recorded code hash differs: {path}")
    source_hashes_verified = {}
    for name in ("source", "calibration", "data"):
        path = Path(natural_meta[f"{name}_path"])
        expected = natural_meta[f"{name}_sha256"]
        require(digest(path) == expected, f"{name} file hash differs")
        source_hashes_verified[name] = {"path": str(path), "sha256": expected}
    calibration = read_json(Path(natural_meta["calibration_path"]).with_suffix(".json"))
    require(calibration["status"] == "PASS" and calibration["state_bit_equal"], "calibration b01 replay did not pass")
    require(calibration["source_b01_sha256"] == natural_meta["source_sha256"], "calibration source differs")
    require(calibration["spec_sha256"] == natural_meta["spec_sha256"], "calibration spec differs")
    require(calibration["sources_sha256"] == natural_meta["sources_sha256"], "calibration code differs")
    require(calibration["data_sha256"] == natural_meta["data_sha256"] and
            calibration["probe_sha256"] == natural_meta["probe_sha256"], "calibration data/probe differ")
    rows, keyed = {}, {}
    expected_keys = {(a, s, t) for a, s in SLOTS for t in range(2, 31)}
    for cond in provenance:
        rows[cond] = read_csv(base / cond / "per_task.csv")
        keys = [(r["cfg"], int(r["seed"]), int(r["task"])) for r in rows[cond]]
        require(len(keys) == 580 and set(keys) == expected_keys, f"{cond}: missing/extra/duplicate task records")
        require(all(int(r["bad_step"]) == -1 for r in rows[cond]), f"{cond}: nonfinite update, no seed may be excluded")
        require(all(math.isfinite(float(r[k])) for r in rows[cond] for k in METRICS), f"{cond}: nonfinite performance")
        require(all(r["condition"] == cond and int(r["hard"]) == int(int(r["task"]) % 2 == 1)
                    and r["task_type"] == ("hard" if int(r["task"]) % 2 else "easy") for r in rows[cond]),
                f"{cond}: row condition/task type differs")
        keyed[cond] = {key: row for key, row in zip(keys, rows[cond])}
        expected_cap_keys = {(a, s, t, site) for a, s in SLOTS for t in range(2, 31) for site in SITES}
        cap_rows = read_csv(base / cond / "cap_stats.csv")
        cap_keys = [(r["cfg"], int(r["seed"]), int(r["task"]), r["site"]) for r in cap_rows]
        require(len(cap_keys) == len(expected_cap_keys) and set(cap_keys) == expected_cap_keys,
                f"{cond}: missing/duplicate cap_stats")
        require(all(r["task_type"] == ("hard" if int(r["task"]) % 2 else "easy") for r in cap_rows),
                f"{cond}: cap_stats task type differs")
    registered = read_json(primary_path)
    require(registered.get("status") == "ESTABLISHED", "registered summary not established")
    require(registered.get("metric") == "latehard_first780_online_acc", "registered summary uses another metric")
    audit = {"status": "PASS", "complete_records_per_condition": 580, "cross_arm_meta_equal_except_condition": True,
             "registered_spec_sha256": natural_meta["spec_sha256"], "source_hashes_verified": source_hashes_verified,
             "calibration_b01_state_bit_equal": True, "primary_file_sha256": digest(primary_path),
             "provenance_sha256": {c: digest(base / c / "provenance.json") for c in provenance},
             "recorded_runtime_precision": natural_meta["initial_runtime_precision"],
             "code_sha256": natural_meta["sources_sha256"], "nonfinite_performance_values": 0}
    return keyed, registered, audit


def primary_and_secondary(keyed, registered):
    def average(cond, arm, seed, tasks, metric):
        return float(np.mean([float(keyed[cond][arm, seed, task][metric]) for task in tasks]))
    primary, secondary = [], []
    for arm in ARMS:
        pairs = []
        for seed in SEEDS:
            late = {c: average(c, arm, seed, LATE, "first780_online_acc") for c in keyed}
            early = {c: average(c, arm, seed, EARLY, "first780_online_acc") for c in keyed}
            pairs.append({"seed": seed, **late, "difference": late["cap"] - late["natural"], "early": early,
                          "late_minus_early": {c: late[c] - early[c] for c in keyed}})
            for metric in METRICS:
                values = {c: average(c, arm, seed, LATE, metric) for c in keyed}
                secondary.append({"cfg": arm, "seed": seed, "metric": metric, **values,
                                  "difference": values["cap"] - values["natural"]})
        delta = [p["difference"] for p in pairs]
        primary.append({"cfg": arm, "pairs": pairs, "mean_difference": float(np.mean(delta)),
                        "median_difference": float(np.median(delta)), **exact_sign(delta)})
    previous = 0.
    for rank, index in enumerate(sorted(range(2), key=lambda i: primary[i]["p"])):
        row = primary[index]
        row["p_holm"] = previous = max(previous, min(1., (2 - rank) * row["p"]))
        direction = row["positive"] - row["negative"]
        row["verdict"] = ("INCONCLUSIVE" if previous >= .05 else "HETEROGENEOUS_SIGN_VS_MEAN"
                          if direction * row["mean_difference"] <= 0 else "IMPROVES" if direction > 0 else "WORSENS")
    require(len(registered.get("results", [])) == 2, "registered result count differs")
    by_arm = {r["cfg"]: r for r in registered["results"]}
    require(set(by_arm) == set(ARMS), "registered arm set differs")
    for row in primary:
        other = by_arm[row["cfg"]]
        for key in ("mean_difference", "median_difference", "p", "p_holm"):
            close(row[key], other[key], f"{row['cfg']}/{key}")
        for key in ("positive", "negative", "zero", "verdict"):
            require(row[key] == other[key], f"registered primary mismatch: {key}")
        require([p["seed"] for p in other["pairs"]] == list(SEEDS), "registered seed ordering differs")
        for p, q in zip(row["pairs"], other["pairs"]):
            for key in ("natural", "cap", "difference"):
                close(p[key], q[key], f"{row['cfg']}/{p['seed']}/{key}")
            for cond in keyed:
                close(p["early"][cond], q["early"][cond], "early")
                close(p["late_minus_early"][cond], q["late_minus_early"][cond], "late_minus_early")
    return primary, secondary


def unit_evidence(base):
    snapshots, growth, endpoint, control, nonfinite = {}, [], [], [], []
    # These are ordinary NumPy saved diagnostic arrays, never torch checkpoints.
    for cond in ("natural", "cap"):
        snapshots[cond] = {}
        for task in range(1, 31):
            path = base / cond / "units" / f"b{task:02d}.npz"
            with np.load(resolve_input(path), allow_pickle=False) as archive:
                arrays = {key: archive[key] for key in archive.files}
            snapshots[cond][task] = arrays
            for site, width in zip(SITES, WIDTHS):
                for key in ("m", "s", "r", "dmean", "dsd", "ppos", "dead", "lowresp", "zeroout"):
                    a = arrays[f"probe_{site}_{key}"]
                    require(a.shape == (20, width), f"{cond}/b{task:02d}/{site}/{key}: wrong shape")
                    count = int((~np.isfinite(a)).sum())
                    if count:
                        nonfinite.append({"condition": cond, "boundary": task, "site": site, "metric": key, "count": count})
                    if key != "r":
                        require(count == 0, f"{cond}/b{task:02d}/{site}/{key}: nonfinite diagnostic")
                ref, v = arrays[f"cap_{site}_s_ref"], arrays[f"cap_{site}_V"]
                require(ref.shape == v.shape == (20, width), "EMA width shape differs")
                require(np.isfinite(ref).all() and (ref >= 0).all() and np.isfinite(v).all() and (v >= 0).all(),
                        f"{cond}/b{task:02d}/{site}: invalid EMA/reference")
                require(np.array_equal(ref, snapshots[cond][1][f"cap_{site}_s_ref"]), "reference changed over time")
                if task == 1:
                    continue
                count = arrays[f"cap_{site}_count"]
                require(np.all(count == 7800), f"{cond}/task{task}/{site}: cap record count not 7800")
                for r, (arm, seed) in enumerate(SLOTS):
                    n = float(count[r].sum())
                    get = lambda key: arrays[f"cap_{site}_{key}"][r].astype(np.float64)
                    for key in ("applied", "ratio_sum", "ratio_min", "batch_pre_s_sum", "batch_post_s_sum",
                                "pre_exceed", "post_exceed", "ema_exceed"):
                        require(np.isfinite(get(key)).all(), f"{cond}/task{task}/{site}: nonfinite {key}")
                    maximum = get("post_ratio_max")
                    invalid = ~np.isfinite(maximum)
                    require(not np.isnan(maximum).any() and np.all(~invalid | (ref[r] == 0)),
                            "nonfinite batch/reference ratio without zero reference")
                    control.append({"condition": cond, "cfg": arm, "seed": seed, "task": task, "site": site,
                                    "task_type": "hard" if task % 2 else "easy", "count_unit_steps": n,
                                    "cap_frequency": float(get("applied").sum() / n),
                                    "mean_ratio": float(get("ratio_sum").sum() / n),
                                    "minimum_ratio": float(get("ratio_min").min()),
                                    "mean_pre_batch_width": float(get("batch_pre_s_sum").sum() / n),
                                    "mean_post_batch_width": float(get("batch_post_s_sum").sum() / n),
                                    "pre_exceed_fraction": float(get("pre_exceed").sum() / n),
                                    "post_exceed_fraction": float(get("post_exceed").sum() / n),
                                    "maximum_post_batch_reference_ratio": finite_number(maximum.max()),
                                    "infinite_post_batch_ratio_units": int(invalid.sum()),
                                    "ema_exceed_unit_steps": int(get("ema_exceed").sum()),
                                    "boundary_ema_exceed_units": int((v[r] > ref[r] ** 2 * (1 + 8*np.finfo(v.dtype).eps)).sum()),
                                    "reference_zero_units": int((ref[r] == 0).sum())})
        for r, (arm, seed) in enumerate(SLOTS):
            for site in SITES:
                start, end = snapshots[cond][1], snapshots[cond][30]
                for kind, x, y in (("fixed_probe_pooled" if site in SITES[:2] else "fixed_probe_images",
                                   start[f"probe_{site}_s"][r], end[f"probe_{site}_s"][r]),
                                  ("ema_post_adam_batch", np.sqrt(start[f"cap_{site}_V"][r]), np.sqrt(end[f"cap_{site}_V"][r]))):
                    a, b = float(np.median(x)), float(np.median(y))
                    growth.append({"condition": cond, "cfg": arm, "seed": seed, "site": site, "width_kind": kind,
                                   "start_boundary": 1, "end_boundary": 30, "start": a, "end": b, "difference": b-a,
                                   "ratio": ratio_or_none(b, a), "increased": b > a,
                                   "zero_start_units": int((x == 0).sum()), "zero_end_units": int((y == 0).sum())})
                row = {"condition": cond, "cfg": arm, "seed": seed, "site": site}
                for key in ("m", "s", "r", "dmean", "dsd", "ppos"):
                    row[key] = safe_median(end[f"probe_{site}_{key}"][r])
                for key in ("dead", "lowresp", "zeroout"):
                    row[f"{key}_fraction"] = float(np.mean(end[f"probe_{site}_{key}"][r]))
                row["undefined_r_units"] = int((~np.isfinite(end[f"probe_{site}_r"][r])).sum())
                row["eff_rank"] = finite_number(end[f"probe_{site}_eff_rank"][r])
                if site in SITES[:2]:
                    for key in ("s_within", "s_between"):
                        row[key] = safe_median(end[f"probe_{site}_{key}"][r])
                endpoint.append(row)
    # Both arms must begin with the identical saved network diagnostics and EMA reference.
    for key in snapshots["natural"][1]:
        if key.startswith("probe_") or key.endswith("_V") or key.endswith("_s_ref"):
            require(np.array_equal(snapshots["natural"][1][key], snapshots["cap"][1][key], equal_nan=True),
                    f"arms differ before intervention: {key}")
    return growth, endpoint, control, nonfinite


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_report(primary, secondary, growth, endpoint, control, nonfinite):
    lines = ["# 実ラベル5+1 CNN：幅capの結果", "",
             "ReLU / ELU（α=3.6）、各10seed、7,800更新/課題。両腕とも新しい20slotで課題1末から30末まで再生した。"
             "元40slotの古いnatural成績を対照に用いていない。主判定は登録済み集計と独立計算が一致した。", "",
             "## 主評価と時間低下", "", "late hard = 21,23,25,27,29。first780は最初780更新のpre-update online accuracy。"
             "各task平均→seed内5task平均→10seed平均。単位は%／対応差はポイント。", ""]
    lines += table(["活性化", "natural", "cap", "cap−natural", "正/負/0", "Holm p", "判定"],
                   [[LABELS[p["cfg"]], show(100*np.mean([x["natural"] for x in p["pairs"]]), 3),
                     show(100*np.mean([x["cap"] for x in p["pairs"]]), 3), show(100*p["mean_difference"], 3),
                     f"{p['positive']}/{p['negative']}/{p['zero']}", show(p["p_holm"], 6), p["verdict"]] for p in primary])
    for p in primary:
        lines += ["", f"### {LABELS[p['cfg']]}：全seed", ""]
        lines += table(["seed", "natural late", "cap late", "対応差", "natural early→late", "cap early→late"],
                       [[x["seed"], show(100*x["natural"], 3), show(100*x["cap"], 3), show(100*x["difference"], 3),
                         show(100*x["late_minus_early"]["natural"], 3), show(100*x["late_minus_early"]["cap"], 3)] for x in p["pairs"]])
    lines += ["", "early = hard 3,5,7,9。時間差はlate−earlyであり、cap−naturalとは別の量。", "",
              "## 副評価", "", "late hardのseed平均。副評価に追加の確認的有意差判定は付けない。", ""]
    rows = []
    for arm in ARMS:
        for metric in ("first100_online_acc", "online_acc", "test_acc", "ce_mean", "ce_first780_mean"):
            selected = [r for r in secondary if r["cfg"] == arm and r["metric"] == metric]
            scale = 100 if "acc" in metric else 1
            rows.append([LABELS[arm], metric + (" (%)" if scale == 100 else ""),
                         show(scale*np.mean([r["natural"] for r in selected])),
                         show(scale*np.mean([r["cap"] for r in selected])), show(scale*np.mean([r["difference"] for r in selected]))])
    lines += table(["活性化", "指標", "natural", "cap", "対応差"], rows)
    lines += ["", "online_accは全7,800更新。既存新品との比較はこの報告には混ぜていない。", "",
              "## 新natural自体で幅は増大したか", "", "まず今回再生したnaturalの始点・終点を直接比較する。"
              "各seedでunit中央値を取り、その10seed平均を示す。増加数はseed内unit中央値が厳密に増えたseed数。"
              "固定probeの画像集合は同じ。EMAはその時点の学習batch分布に依存し、課題1はhard、課題30はeasy。", ""]
    for cond in ("natural", "cap"):
        lines += [f"### {cond}：課題1末→30末", ""]
        rows = []
        for arm in ARMS:
            for site in SITES:
                for kind in sorted({r["width_kind"] for r in growth if r["site"] == site}):
                    selected = [r for r in growth if r["condition"] == cond and r["cfg"] == arm and r["site"] == site and r["width_kind"] == kind]
                    rows.append([LABELS[arm], site, kind, show(np.mean([r["start"] for r in selected])),
                                 show(np.mean([r["end"] for r in selected])), f"{sum(r['increased'] for r in selected)}/10",
                                 sum(r["zero_start_units"] for r in selected)])
        lines += table(["活性化", "site", "幅の量", "始点", "終点", "増加seed", "始点0幅unit"], rows) + [""]
    lines += ["全seedの始点・終点・差・倍率は `width_growth.csv`。EMAと固定probeを同じ幅として混ぜない。", "",
              "## 課題30末の応答・位置とcapの幅効果", "", "s,m,r,φ′平均/標準偏差,p+はunit中央値→seed平均。"
              "dead/low-responseはunit割合→seed平均。φ′は活性化関数の局所微分であり、網全体の勾配ではない。", ""]
    rows = []
    for arm in ARMS:
        for site in SITES:
            for cond in ("natural", "cap"):
                selected = [r for r in endpoint if r["cfg"] == arm and r["site"] == site and r["condition"] == cond]
                rows.append([LABELS[arm], site, cond] + [show(mean_valid([r[k] for r in selected])) for k in
                            ("s", "r", "dmean", "dsd", "ppos", "dead_fraction", "lowresp_fraction")])
    lines += table(["活性化", "site", "腕", "幅s", "位置r", "φ′平均", "φ′sd", "p+", "dead", "low-response"], rows)
    lines += ["", "幅への対応差を性能の主判定と分けて示す。終点のunit中央値について、capで小さいseed数とcap−natural平均差：", ""]
    paired_widths = []
    for arm in ARMS:
        for site in SITES:
            for kind in sorted({r["width_kind"] for r in growth if r["site"] == site}):
                pairs = []
                for seed in SEEDS:
                    a = next(r for r in growth if r["condition"] == "natural" and r["cfg"] == arm and r["seed"] == seed and r["site"] == site and r["width_kind"] == kind)
                    b = next(r for r in growth if r["condition"] == "cap" and r["cfg"] == arm and r["seed"] == seed and r["site"] == site and r["width_kind"] == kind)
                    pairs.append(b["end"] - a["end"])
                paired_widths.append([LABELS[arm], site, kind, show(np.mean(pairs)), f"{sum(d < 0 for d in pairs)}/10"])
    lines += table(["活性化", "site", "幅の量", "cap−natural", "capで小さいseed"], paired_widths)
    lines += ["", "固定probeでのseedごとのcap−natural差、convのwithin/between幅、m、rank、未定義r数は `endpoint_effects.csv` と `endpoint_units_summary.csv`。", "",
              "## capの作動とEMA制御", "", "同じ各7,800更新なので、unit×stepを分母に集計する。"
              "hardとeasyを分ける。参照は課題1hardのpost-Adam幅EMAで、現在batchや固定probeに厳密な上限を課した操作ではない。", ""]
    rows = []
    for arm in ARMS:
        for site in SITES:
            for kind in ("hard", "easy"):
                selected = [r for r in control if r["condition"] == "cap" and r["cfg"] == arm and r["site"] == site and r["task_type"] == kind]
                rows.append([LABELS[arm], site, kind, show(np.mean([r["cap_frequency"] for r in selected])),
                             show(np.mean([r["mean_ratio"] for r in selected])), show(min(r["minimum_ratio"] for r in selected)),
                             show(np.mean([r["post_exceed_fraction"] for r in selected])),
                             sum(r["ema_exceed_unit_steps"] for r in selected), sum(r["boundary_ema_exceed_units"] for r in selected)])
    lines += table(["活性化", "site", "課題型", "発火率", "平均倍率", "最小倍率", "batch幅超過率", "EMA超過unit-step", "境界EMA超過unit"], rows)
    zero_rows = [r for r in control if r["condition"] == "cap" and r["task"] == 2]
    lines += ["", f"参照幅0のunitは全20slot・全4siteで {sum(r['reference_zero_units'] for r in zero_rows)}。"
              f" 保存probeの非有限rの記録は {len(nonfinite)} 件（各boundary/siteを1件、unit数はJSON参照）。"
              "非有限の成績・更新を除外した走はない（存在すれば報告生成自体を拒否する）。", "",
              "## 読みの限界", "", "Wとbiasを同率に縮めるため、固定した局所入力で投影する瞬間のrは保つ。"
              "全4siteの介入では上流出力・入力分布・後続の学習が変わるので、全網の純粋な幅だけの媒介効果とは呼ばない。"
              "ELUは元から新品より高い場合があり、改善を自動的に『崩壊救済』としない。"
              "非有意・悪化・無効から『幅は原因ではない』とも結論しない。"
              "操作が幅を抑えたか、応答や性能を変えたか、同じ層・時期かを分けて読む。", ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, default=ROOT / "results/cnn_width_cap_1021")
    ap.add_argument("--primary", type=Path, help="defaults to BASE/summary/primary.json")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    primary_path = args.primary or args.base / "summary/primary.json"
    try:
        keyed, registered, audit = load_and_audit(args.base, primary_path)
        primary, secondary = primary_and_secondary(keyed, registered)
        growth, endpoint, control, nonfinite = unit_evidence(args.base)
        effects = []
        index = {(r["condition"], r["cfg"], r["seed"], r["site"]): r for r in endpoint}
        for arm, seed in SLOTS:
            for site in SITES:
                a, b = index["natural", arm, seed, site], index["cap", arm, seed, site]
                for metric in sorted(a.keys() - {"condition", "cfg", "seed", "site"}):
                    effects.append({"cfg": arm, "seed": seed, "site": site, "metric": metric,
                                    "natural": a[metric], "cap": b[metric],
                                    "difference": None if a[metric] is None or b[metric] is None else b[metric]-a[metric]})
    except (OSError, KeyError, ValueError, TypeError, StopIteration, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"REFUSED: {exc}. No scientific report was written.") from exc
    args.out.mkdir(parents=True, exist_ok=True)
    report = {"status": "COMPLETE_AUDITED", "audit": audit, "primary_independent": primary,
              "secondary": secondary, "width_growth": growth, "endpoint": endpoint,
              "endpoint_effects": effects, "cap_control": control, "nonfinite_r_records": nonfinite,
              "primary_matches_registered": True, "report_script_sha256": digest(__file__)}
    (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    (args.out / "report.md").write_text(make_report(primary, secondary, growth, endpoint, control, nonfinite))
    for name, rows in (("secondary", secondary), ("width_growth", growth), ("endpoint_units_summary", endpoint),
                       ("endpoint_effects", effects), ("cap_control", control)):
        write_csv(args.out / f"{name}.csv", rows)
    print(f"PASS: both completed arms audited; independent primary matches. Report: {args.out / 'report.md'}")


if __name__ == "__main__":
    main()
