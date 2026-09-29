"""Read-only fixed-group compatibility audit of saved ELU states.

No training, fitted coefficients, or causal claims. All rows preserve unit IDs.
Run from repository root with the project's Python environment.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/two_group_logistic_0929"
SOURCE = ROOT / "results/logistic_v_height_0929"
NA = float("nan")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def lower_median(x):
    return float(np.partition(x, (len(x) - 1) // 2)[(len(x) - 1) // 2]) if len(x) else NA


def safe_ratio(a, b):
    return float(a / b) if np.isfinite(a) and np.isfinite(b) and b > 0 else NA


def positive_ratio(a, b):
    return safe_ratio(a, b) if a > 0 else NA


def z_from(X, state, prefix=""):
    # Match original saved-state diagnostic arithmetic: cast before matmul.
    return (X @ torch.from_numpy(state[prefix + "W1"]).double().T
            + torch.from_numpy(state[prefix + "b1"]).double()).numpy()


def unit_metrics(z0, z):
    rows = []
    for u in range(z.shape[1]):
        old, cur = z0[:, u], z[:, u]
        bulk, opened = old <= -8, old > 0
        a0, a = old[bulk], cur[bulk]
        o = cur[opened]
        n, no = int(bulk.sum()), int(opened.sum())
        sd0 = float(a0.std()) if n >= 2 else NA
        sd = float(a.std()) if n >= 2 else NA
        shift = float(np.mean(a - a0)) if n else NA
        centered_rms = float(np.sqrt(np.mean(((a - a0) - shift) ** 2))) if n >= 2 else NA
        corr = float(np.mean((a0 - a0.mean()) * (a - a.mean())) / (sd0 * sd)) if n >= 2 and sd0 > 0 and sd > 0 else NA
        rows.append({
            "unit": u, "n_inputs": len(old), "n_bulk_fixed": n, "n_open_fixed": no,
            "n_neither_fixed": int((~(bulk | opened)).sum()),
            "p_open_fixed": no / len(old), "p_open_current": float(np.mean(cur > 0)),
            "bulk_initial_mean": float(a0.mean()) if n else NA,
            "bulk_initial_lower_median": lower_median(a0), "bulk_initial_sd_population": sd0,
            "bulk_mean": float(a.mean()) if n else NA, "bulk_lower_median": lower_median(a),
            "bulk_sd_population": sd, "bulk_sd_sample": float(a.std(ddof=1)) if n >= 2 else NA,
            "bulk_sd_over_initial": safe_ratio(sd, sd0), "bulk_mean_displacement": shift,
            "bulk_centered_displacement_rms": centered_rms,
            "bulk_centered_displacement_rms_over_initial_sd": safe_ratio(centered_rms, sd0),
            "bulk_initial_correlation": corr,
            "bulk_retained_deep_fraction": float(np.mean(a <= -8)) if n else NA,
            "bulk_retained_closed_fraction": float(np.mean(a <= 0)) if n else NA,
            "open_mean": float(o.mean()) if no else NA, "open_max": float(o.max()) if no else NA,
            "open_retained_fraction": float(np.mean(o > 0)) if no else NA,
            "current_max": float(cur.max()), "current_max_was_initially_open": int(opened[cur.argmax()]),
            "current_open_from_initial_open_fraction": float(np.sum((cur > 0) & opened) / np.sum(cur > 0)) if np.any(cur > 0) else NA,
        })
    return rows


def write_csv(name, rows):
    path = OUT / name
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: ("NA" if isinstance(v, (float, np.floating)) and not np.isfinite(v) else v) for k, v in r.items()})


def summarize(rows, keys):
    groups = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in keys), []).append(r)
    result = []
    ignored = set(keys) | {"unit", "arm"}
    for identity, group in groups.items():
        row = dict(zip(keys, identity))
        row["n_units"] = len(group)
        for metric in group[0]:
            if metric in ignored:
                continue
            vals = np.asarray([r[metric] for r in group], dtype=float)
            valid = vals[np.isfinite(vals)]
            row["median_" + metric] = float(np.median(valid)) if len(valid) else NA
            row["n_valid_" + metric] = len(valid)
        result.append(row)
    return result


def fmt(x):
    return f"{x:.4g}" if isinstance(x, (float, np.floating)) else str(x)


def main():
    torch.set_num_threads(1)
    torch.set_flush_denormal(True)
    OUT.mkdir(parents=True, exist_ok=True)
    script_sha_at_start = sha(Path(__file__))
    git_at_start = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    sources, inputs, units, pairs, bases = [], [], [], [], []
    for seed in (0, 1):
        path = SOURCE / f"real_s{seed}_t50.npz"
        sources.append({"path": str(path.relative_to(ROOT)), "sha256": sha(path)})
        with np.load(path) as initial:
            X = torch.from_numpy(initial["X"]).double()
            z0 = z_from(X, initial)
            inputs.append({"seed": seed, "X_shape": list(initial["X"].shape), "X_dtype": str(initial["X"].dtype),
                           "X_bytes_sha256": hashlib.sha256(initial["X"].tobytes()).hexdigest(),
                           "subset_bytes_sha256": hashlib.sha256(initial["subset"].tobytes()).hexdigest()})
        bases.extend({"seed": seed, **r} for r in unit_metrics(z0, z0))
        for task in (51, 53):
            path = SOURCE / f"real_s{seed}_t{task}_states.npz"
            sources.append({"path": str(path.relative_to(ROOT)), "sha256": sha(path)})
            with np.load(path) as state:
                for policy in ("A", "C"):
                    arms = {}
                    for gain in (.1, 1., 10.):
                        arm = "REF_c1p0" if gain == 1 else f"{policy}_c{str(gain).replace('.', 'p')}"
                        arms[gain] = unit_metrics(z0, z_from(X, state, arm + "__"))
                        units.extend({"seed": seed, "task": task, "policy": policy, "gain": gain,
                                      "arm": arm, **r} for r in arms[gain])
                    # Contrasts are paired on unit ID before aggregation, not ratios of aggregate medians.
                    for small_gain, large_gain in ((.1, 1.), (1., 10.), (.1, 10.)):
                        for small, large in zip(arms[small_gain], arms[large_gain]):
                            r = {"seed": seed, "task": task, "policy": policy,
                                 "small_gain": small_gain, "large_gain": large_gain, "unit": small["unit"],
                                 "bulk_sd_small_over_large": safe_ratio(small["bulk_sd_population"], large["bulk_sd_population"])}
                            for m in ("current_max", "open_max", "open_mean"):
                                r[m + "_small_minus_large"] = small[m] - large[m]
                                r[m + "_small_over_large_if_positive"] = positive_ratio(small[m], large[m]) if large[m] > 0 else NA
                            for m in ("bulk_mean", "bulk_lower_median", "p_open_current", "open_retained_fraction", "bulk_retained_deep_fraction"):
                                r[m + "_small_minus_large"] = small[m] - large[m]
                            pairs.append(r)
    summary = summarize(units, ["seed", "task", "policy", "gain"])
    contrast = summarize(pairs, ["seed", "task", "policy", "small_gain", "large_gain"])
    baseline = summarize(bases, ["seed"])
    primary = [r for r in contrast if (r["small_gain"], r["large_gain"]) == (.1, 10.)]
    small53 = [r for r in summary if r["task"] == 53 and r["gain"] == .1]
    large53 = [r for r in summary if r["task"] == 53 and r["gain"] == 10.]
    def span(rows, key):
        vals = [r[key] for r in rows]
        return f"{min(vals):.4g}–{max(vals):.4g}"

    # Exact structural checks, not a fitted confirmation of the scientific claim.
    assert len(units) == 2400 and len(pairs) == 2400
    assert all(r["bulk_centered_displacement_rms"] == 0 for r in bases)
    assert all(abs(r["bulk_initial_correlation"] - 1) < 1e-12 for r in bases)
    for name, rows in (("actual_unit_metrics.csv", units), ("actual_summary.csv", summary),
                       ("actual_paired_units.csv", pairs), ("actual_paired_summary.csv", contrast),
                       ("actual_initial_units.csv", bases), ("actual_initial_summary.csv", baseline)):
        write_csv(name, rows)
    lines = ["# 保存実系における固定深部・固定開集合の互換性監査", "",
             f"上端と深部幅の非対称性は保存実系にある。gain .1と10を比較すると、同一unitの深部SD比の中央値は全8条件で{span(primary, 'median_bulk_sd_small_over_large')}、現在maxの比は{span(primary, 'median_current_max_small_over_large_if_positive')}。後者は両腕でmax>0の{span(primary, 'n_valid_current_max_small_over_large_if_positive')}unitに限定されるが、全100unitのsigned max差の中央値も{span(primary, 'median_current_max_small_minus_large')}で同じ向き。従って一律1/v拡縮ではない。",
             f"一方、深部が凍結し、同じ稀な開集合だけが動くという厳密な仮定は満たさない。task53・small-vでは深部の平均移動除去RMS/初期SDは{span(small53, 'median_bulk_centered_displacement_rms_over_initial_sd')}、初期との相関は{span(small53, 'median_bulk_initial_correlation')}。現在開いている入力のうち初期open由来の割合はunit中央値で{span(small53, 'median_current_open_from_initial_open_fraction')}しかない。固定open保持率はsmall-vで{span(small53, 'median_open_retained_fraction')}、large-vで{span(large53, 'median_open_retained_fraction')}。非対称性への互換性はあるが、固定群模型の実系での成立や因果機構は未確認。", "",
             "再学習なし。task50のzでunitごとにB={z≤−8}とA={z>0}を固定し、同じ入力をtask51/53末まで追う。",
             "Aは元のmomentを保持したv倍率介入、Cはhidden Adam一次/二次momentを倍率/倍率²に整合しhidden epsilonも倍率に整合した介入。gain=1は共通REFを再利用。",
             "これは二群理論が要求する上端と深部幅の非対称性の**互換性監査**であり、機構の因果同定、一定の目標logit差、固定群近似の証明ではない。", "",
             "## 初期群", "", "| seed | median n bulk | median n open | median p open | median bulk SD | bulk n<2 | open n=0 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for b in baseline:
        group = [r for r in bases if r["seed"] == b["seed"]]
        values = [b["seed"], b["median_n_bulk_fixed"], b["median_n_open_fixed"], b["median_p_open_fixed"], b["median_bulk_sd_population"],
                  sum(r["n_bulk_fixed"] < 2 for r in group), sum(r["n_open_fixed"] == 0 for r in group)]
        lines.append("| " + " | ".join(map(fmt, values)) + " |")
    lines += ["", "## gain .1 / 10：同一unit内の比・差を取ってから中央値", "",
              "完全な一律1/v拡縮なら幅・正の高さの比はいずれも100。高さが負または0の比はNA、signed差は保持。", "",
              "| seed | task | policy | bulk SD比 | current max比 | max比の有効unit | current max差 | 固定open max比 | 固定open max比有効unit | 固定open max差 |", "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in contrast:
        if (r["small_gain"], r["large_gain"]) != (.1, 10.):
            continue
        vals = [r[k] for k in ("seed", "task", "policy", "median_bulk_sd_small_over_large", "median_current_max_small_over_large_if_positive",
                "n_valid_current_max_small_over_large_if_positive", "median_current_max_small_minus_large", "median_open_max_small_over_large_if_positive", "n_valid_open_max_small_over_large_if_positive", "median_open_max_small_minus_large")]
        lines.append("| " + " | ".join(map(fmt, vals)) + " |")
    lines += ["", "## 深部の変形と群の入れ替わり", "",
              "RMSは固定B内の平均移動を除いた(z_t−z_0)のRMSを初期Bのpopulation SDで割る。corrは同じB内での初期zと現在z。retained openは初期Aのうち現在もz>0の割合。", "",
              "| seed | task | policy | gain | bulk幅/初期幅 | bulk平均移動 | centered RMS/初期幅 | corr | bulk deep保持 | 初期open保持 | 現在openの初期open由来率 | 現在p open | 固定open平均 | 現在max |", "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in summary:
        keys = ("seed", "task", "policy", "gain", "median_bulk_sd_over_initial", "median_bulk_mean_displacement", "median_bulk_centered_displacement_rms_over_initial_sd",
                "median_bulk_initial_correlation", "median_bulk_retained_deep_fraction", "median_open_retained_fraction", "median_current_open_from_initial_open_fraction", "median_p_open_current", "median_open_mean", "median_current_max")
        lines.append("| " + " | ".join(fmt(r[k]) for k in keys) + " |")
    lines += ["", "## 定義と限界", "",
              "各表はseedごとに100unitを同じ重みで集計し、unit統計の通常の中央値（偶数では中央2値の平均）を取る。seed、unit、taskを独立反復とした検定や信頼区間は行わない。gain=1のREFはA/C表に重複表示するが独立腕ではない。",
              "zは保存float32パラメータと入力をfloat64へ変換してからtorch単一threadで再構成し、既存診断の算術に合わせる。bulk内medianは元診断と同じlower median。population SDを主列、sample SDを補助列としてCSVに保存。SD・centered RMS・corrはn<2でNA、corrはどちらかの幅が0でもNA。群平均・max・割合は空群でNA。各集計列に有効unit数が付く。",
              "固定open集合は現在の上端を担う集合と一致するとは限らない。保持率の低下やbulkの変形は、固定群・固定bulk幅という最小理論の仮定からの離脱である。上端差と幅差の非対称性だけでは、目標margin一定やbulkが勾配的に凍結していることは示せない。",
              "固定初期pと現在pをともに保存。pの生成機構は本監査の対象外。gain操作は全W2ベクトルの倍率であり単一の符号付きスカラーreadoutではない。多class・多unit・課題切替と最小二群模型の対応にはこの差が残る。", "",
              "出力: actual_unit_metrics.csv（全unit）、actual_paired_units.csv（同一unit対比）、各summary、初期群表。入力とソースのSHAはactual_provenance.json。"]
    (OUT / "actual_report.md").write_text("\n".join(lines) + "\n")
    provenance = {"kind": "read_only_compatibility_audit_no_training", "script": str(Path(__file__).relative_to(ROOT)),
                  "script_sha256_at_start": script_sha_at_start,
                  "git_commit_at_start": git_at_start,
                  "spec_sha256": sha(ROOT / "specs/spec_two_group_logistic_0929.md"),
                  "sources": sources, "inputs": inputs, "numpy": np.__version__, "torch": torch.__version__,
                  "z_arithmetic": "torch float64 CPU single thread; flush_denormal true",
                  "unit_rows": len(units), "paired_rows": len(pairs)}
    assert all(sha(ROOT / s["path"]) == s["sha256"] for s in sources), "Source changed while auditing"
    (OUT / "actual_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"unit_rows": len(units), "paired_rows": len(pairs), "report": str(OUT / "actual_report.md")}, indent=2))


if __name__ == "__main__":
    main()
