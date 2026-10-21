#!/usr/bin/env python3
"""Audit saved source-run width statistics; no model loading, forward, or training.

Uses only the Python standard library. Rebuilds the adjacent Markdown and JSON.
The audit concerns the original 40-slot b7800 run, not the new 20-slot cap run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


NAME = "source_width_growth_audit"
SEEDS = tuple(range(2000, 2010))
CONFIGS = {"R_lr0.0003": "ReLU", "ELU_a3.6_lr0.0003": "ELU α=3.6"}
SITES = ("c1", "c2", "f1", "f2")
UNITS = {"c1": 16, "c2": 16, "f1": 100, "f2": 100}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(row: dict, key: str) -> float:
    value = float(row[key])
    require(math.isfinite(value), f"Non-finite {key}: {row}")
    return value


def read_rows(path: Path, time_key: str, phases: tuple[str, ...]) -> dict:
    index = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["cfg"] not in CONFIGS:
                continue
            seed, t, site = int(row["seed"]), int(row[time_key]), row["site"]
            phase = row.get("phase", "probe")
            key = (row["cfg"], seed, t, phase, site)
            require(key not in index, f"Duplicate row in {path}: {key}")
            require(seed in SEEDS and site in SITES and phase in phases,
                    f"Unexpected row in {path}: {key}")
            require(int(row["n_units"]) == UNITS[site], f"Wrong unit count: {key}")
            require(finite(row, "s_med") >= 0, f"Negative width: {key}")
            if site.startswith("c"):
                for field in ("s_within_med", "s_between_med"):
                    require(finite(row, field) >= 0, f"Negative {field}: {key}")
            index[key] = row
    times = range(31) if time_key == "boundary" else range(1, 31)
    expected = {(cfg, seed, t, phase, site) for cfg in CONFIGS for seed in SEEDS
                for t in times for phase in phases for site in SITES}
    require(set(index) == expected,
            f"Incomplete/unexpected rows in {path}: missing={len(expected - set(index))}, "
            f"extra={len(set(index) - expected)}")
    return index


def comparison(index: dict, cfg: str, site: str, field: str,
               start: int, end: int, phase: str) -> dict:
    per_seed = []
    for seed in SEEDS:
        a = finite(index[cfg, seed, start, phase, site], field)
        b = finite(index[cfg, seed, end, phase, site], field)
        require(a > 0, f"Undefined ratio for {cfg}/{seed}/{site}/{field}")
        per_seed.append({"seed": seed, "start": a, "end": b, "ratio": b / a})
    ratios = [r["ratio"] for r in per_seed]
    return {
        "cfg": cfg, "arm": CONFIGS[cfg], "site": site, "field": field,
        "start": start, "end": end, "phase": phase,
        "start_seed_mean": statistics.mean(r["start"] for r in per_seed),
        "end_seed_mean": statistics.mean(r["end"] for r in per_seed),
        "paired_ratio_mean": statistics.mean(ratios),
        "paired_ratio_median": statistics.median(ratios),
        "paired_ratio_min": min(ratios), "paired_ratio_max": max(ratios),
        "increased_seeds": sum(r["end"] > r["start"] for r in per_seed),
        "equal_seeds": sum(r["end"] == r["start"] for r in per_seed),
        "n_seeds": len(per_seed), "per_seed": per_seed,
    }


def growth_table(rows: list[dict]) -> list[str]:
    lines = ["| 腕 | 層 | 課題1末 | 比較時点末 | 対応倍率の中央値［最小–最大］ | 増大seed |",
             "|---|---|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(
            f"| {row['arm']} | {row['site']} | {row['start_seed_mean']:.3f} | "
            f"{row['end_seed_mean']:.3f} | {row['paired_ratio_median']:.2f}"
            f"［{row['paired_ratio_min']:.2f}–{row['paired_ratio_max']:.2f}］ | "
            f"{row['increased_seeds']}/{row['n_seeds']} |")
    return lines


def markdown(audit: dict) -> str:
    all_grew = all(row["increased_seeds"] == 10 for row in audit["fixed_probe_growth"])
    conclusion = ("元の b7800 走では、ReLU と ELU α=3.6 の全4層で、"
                  "各seedの層内unit幅中央値が課題1末→30末に10/10すべて増大した。"
                  if all_grew else
                  "全腕・全層・全seedの増大は成立しなかった。以下に全結果を示す。")
    lines = [
        "# CNN cap の元走が幅の伸びる箱か：保存結果の監査", "",
        "対象：`cifar5p1_cnn_budget_1019/b7800`、ReLU / ELU α=3.6、seed 2000–2009。",
        "保存CSVの読み取りと集計だけを行う。checkpointのロード、forward、GPU計算、再学習は行わない。", "",
        f"**{conclusion}** 全unitの増大や、全時点での単調増大を意味しない。", "",
        "## 1. 固定probeでの課題1末→30末", "",
        "評価画像は全時点で同じ2,000枚（100クラス各20枚）。幅は前活性の母標準偏差。"
        "convは画像×空間位置、FCは画像方向で求める。表の幅は「seed内unit中央値→seed平均」。"
        "倍率は各seedの30末/1末を計算した後の中央値と範囲であり、表の平均幅どうしの比ではない。", "",
        *growth_table(audit["fixed_probe_growth"]), "",
        "したがって、評価画像の種類を後期だけ替えたことによる見かけの増大ではない。"
        "全seedの元値・倍率は隣接するJSONに保存した。", "",
        "## 2. 畳み込み幅の成分", "",
        "`s_within` は、各空間位置で画像間の母分散を求め、位置平均して平方根を取る。"
        "`s_between` は、各位置の画像平均の、位置間での母標準偏差。"
        "「画像内」「クラス内」の分散ではない。unit単位では pooled分散=within分散+between分散だが、"
        "unit中央値を取った値どうしに同じ恒等式は課さない。", "",
        "| 腕 | 層 | 成分 | 課題1末 | 課題30末 | 増大seed |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in audit["conv_components"]:
        component = row["field"].removeprefix("s_").removesuffix("_med")
        lines.append(f"| {row['arm']} | {row['site']} | {component} | "
                     f"{row['start_seed_mean']:.3f} | {row['end_seed_mean']:.3f} | "
                     f"{row['increased_seeds']}/{row['n_seeds']} |")
    lines += [
        "", "位置内幅も両腕・両conv層で10/10増大しており、位置間の平均差だけの増大ではない。", "",
        "## 3. 現在課題の学習画像：hard→hard", "",
        "課題1末→29末を、各課題の学習画像上で比較する。両方ともhard（5クラス、2,500枚）だが、"
        "クラスの同一性は保たれない。固定probeによる比較を主とし、こちらは学習分布上の補助確認とする。", "",
        *growth_table(audit["train_hard_growth"]), "",
        "JSONにはeasyの課題30末との比較も残した。hard/easyを混ぜて同一分布と解釈しない。", "",
        "## 4. capとの対応と、まだ確定していない点", "",
        "- capの起点は課題1末。過去の説明で使われた課題2末のf2幅（ReLU約2.70、ELU約2.06）を、"
        "今回の参照幅として引用しない。固定probeの課題1末値とも、以下のEMA参照とも異なる。",
        "- 新specの参照は、元40slotの課題1を再生して測る、各Adam更新直後・同じ学習batch上の分散のEMA。"
        "最初のbatch分散で初期化し、以後0.99 V+0.01 v。固定probeの母分散でも、課題全画像の母分散でもない。"
        "分布・有限batch・平均操作が異なり、既存CSVからこのEMAの伸び率を確定することはできない。",
        "- 新走はReLU/ELU各10seedの同じ20slot構成でnatural/capを再生する。元走は4腕の40slotなので、"
        "束サイズによる数値差を再現検査で評価し、過去のnatural成績を新capの直接対照には使わない。",
        "- 新20slot走は親セッションが別途扱う。本監査は元走の保存結果だけを対象とし、新走の実行状況・"
        "長期幅成長を判定しない。短い再現検査だけでは長期の成長を保証しない。最終報告では新natural自身の"
        "課題1末→後期の固定probe幅、EMA、seed別増減数を併記する。",
        "- 本監査は幅の成長の確認。幅の成長が性能低下の原因であることや、capの有効性を判定するものではない。", "",
        "## 5. 来歴と再生成", "",
        f"元走status：`{audit['source_provenance']['status']}`。元コードcommit："
        + ", ".join(f"`{c}`" for c in audit["source_provenance"]["git_hashes"]) + "。",
        f"selected CSV行数の検査：probe {audit['validation']['probe_selected_rows']}行、"
        f"train {audit['validation']['train_selected_rows']}行。重複・欠落・非有限幅を拒否する。", "",
        "```bash", "python3 analysis/cnn_width_cap_1021/source_width_growth_audit.py", "```", "",
        "引数 `--source` で保存結果ディレクトリ、`--out-dir` で出力先を指定できる。"
        "集計値はCSVに保存された精度に基づく。標準ライブラリのみを使用する。", "",
        "入力ファイルとSHA256：", "",
    ]
    for filename, entry in audit["sources"].items():
        lines += [f"- `{filename}`: `{entry['sha256']}`"]
    lines += [
        "", "幅の定義の実装：`src/cifar5p1_cnn_budget_1019.py::unit_stats2`。",
        "capの定義：`specs/spec_cnn_width_cap_1021.md` §2–5。"
        "これらの意味を監査時に読み合わせた。新specの後の変更は本CSV監査の対象外。", "",
    ]
    return "\n".join(lines)


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=repo / "results/cifar5p1_cnn_budget_1019/b7800")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    source = args.source.resolve()
    provenance = json.loads((source / "provenance.json").read_text(encoding="utf-8"))
    meta = provenance["meta"]
    require(provenance["status"] == "COMPLETED", "Source run is not COMPLETED")
    require(meta["run_id"] == "cifar5p1_cnn_budget_1019", "Unexpected source run")
    require((meta["budget"], meta["n_tasks"], meta["batch"]) == (7800, 30, 32),
            "Unexpected budget/tasks/batch")
    require(len(meta["slots"]) == 40, "Expected original 40-slot bundle")
    for cfg in CONFIGS:
        require(sorted(s for c, s in meta["slots"] if c == cfg) == list(SEEDS),
                f"Unexpected source seeds for {cfg}")
    probe = read_rows(source / "probe_stats.csv", "boundary", ("probe",))
    train = read_rows(source / "train_stats.csv", "task", ("start", "end"))
    audit = {
        "schema": 1,
        "scope": "Saved source-run CSV audit only; no forward, GPU, training, or new-run qualification",
        "aggregation": "Per-seed median across units, then mean across seeds; ratios paired within seed",
        "source_provenance": {
            "status": provenance["status"], "run_id": meta["run_id"],
            "budget": meta["budget"], "n_tasks": meta["n_tasks"], "batch": meta["batch"],
            "original_slots": len(meta["slots"]), "selected_seeds": list(SEEDS),
            "git_hashes": [entry["git_hash"] for entry in provenance["gits"]],
            "probe_sha256": meta["probe_sha256"],
        },
        "sources": {name: {"sha256": sha256(source / name)}
                    for name in ("probe_stats.csv", "train_stats.csv", "provenance.json")},
        "validation": {"probe_selected_rows": len(probe), "train_selected_rows": len(train)},
        "fixed_probe_growth": [comparison(probe, cfg, site, "s_med", 1, 30, "probe")
                               for cfg in CONFIGS for site in SITES],
        "conv_components": [comparison(probe, cfg, site, field, 1, 30, "probe")
                            for cfg in CONFIGS for site in ("c1", "c2")
                            for field in ("s_within_med", "s_between_med")],
        "train_hard_growth": [comparison(train, cfg, site, "s_med", 1, 29, "end")
                              for cfg in CONFIGS for site in SITES],
        "train_easy_growth": [comparison(train, cfg, site, "s_med", 1, 30, "end")
                              for cfg in CONFIGS for site in SITES],
        "fixed_probe_trajectory_seed_means": [
            {"cfg": cfg, "site": site, "boundary": t,
             "s_med_seed_mean": statistics.mean(
                 finite(probe[cfg, seed, t, "probe", site], "s_med") for seed in SEEDS)}
            for cfg in CONFIGS for site in SITES for t in (1, 2, 10, 20, 29, 30)],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"{NAME}.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (args.out_dir / f"{NAME}.md").write_text(markdown(audit), encoding="utf-8")
    print(f"Audited {len(probe)} probe and {len(train)} train rows; wrote {NAME}.md/.json")


if __name__ == "__main__":
    main()
