#!/usr/bin/env python3
"""Aggregate fixed saved-state results; registered J1-J3 never depend on check outcomes."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src import cnn5p1_labelterm_1023 as M
import numpy as np

OUT = M.OUT
ARM_LABEL = {"R": "ReLU", "ELU": "ELU α3.6", "KKT1": "KKT1", "SNAfrz1": "SNAfrz1"}


def stats(values):
    a = np.array(values, dtype=float)
    a = a[np.isfinite(a)]
    return dict(n=len(a), median=float(np.median(a)) if len(a) else None,
                min=float(a.min()) if len(a) else None, max=float(a.max()) if len(a) else None)


def fmt(s, digits=3):
    if s["median"] is None:
        return "未定義"
    return f"{s['median']:.{digits}f} [{s['min']:.{digits}f}, {s['max']:.{digits}f}]"


def read_rows():
    with open(OUT / "rows.csv") as f:
        rows = list(csv.DictReader(f))
    text = {"arm", "cfg", "kind", "site", "classes"}
    integer = {"budget", "boundary", "task", "seed", "slot", "N", "n_alive", "n_units",
               "n_ratio_L", "n_ratio_class", "K_nonpositive"}
    for r in rows:
        for k, v in r.items():
            if k == "excluded":
                r[k] = v == "True"
            elif k in integer:
                r[k] = int(v)
            elif k not in text:
                r[k] = float(v)
    expected = {(budget, b, b + (1 if kind == "easy" else 2), arm, seed, site)
                for budget in M.BUDGETS for b in M.BOUNDARIES
                for kind in (("easy", "hard") if b < 29 else ("easy",))
                for arm in M.ARMS for seed in range(2000, 2010) for site in M.SITES}
    keys = [(r["budget"], r["boundary"], r["task"], r["arm"], r["seed"], r["site"]) for r in rows]
    if set(keys) != expected or len(keys) != len(expected):
        raise RuntimeError("rows.csv does not have exactly the registered states")
    return rows


def aggregate(rows):
    cells, windows = [], []
    for site in M.SITES:
        for kind in ("hard", "easy"):
            for arm in M.ARMS:
                for budget in M.BUDGETS:
                    all_rows = [r for r in rows if (r["site"], r["kind"], r["arm"], r["budget"]) ==
                                (site, kind, arm, budget)]
                    keep = [r for r in all_rows if not r["excluded"]]
                    seed_means = []
                    for seed in range(2000, 2010):
                        rs = [r for r in keep if r["seed"] == seed]
                        if not rs:
                            continue
                        values = {}
                        for k in M.METRICS:
                            vs = [r[k] for r in rs if np.isfinite(r[k])]
                            values[k] = float(np.mean(vs)) if vs else float("nan")
                            values[k + "_n_boundaries"] = len(vs)
                        seed_means.append(dict(seed=seed, boundaries=[r["boundary"] for r in rs], **values))
                    cell = dict(site=site, kind=kind, arm=arm, budget=budget,
                                n_states=len(keep), n_excluded=len(all_rows) - len(keep),
                                n_seeds=len(seed_means), seed_means=seed_means,
                                metrics={k: stats([r[k] for r in seed_means]) for k in M.METRICS})
                    cells.append(cell)
                    for boundary in sorted({r["boundary"] for r in all_rows}):
                        rs = [r for r in keep if r["boundary"] == boundary]
                        windows.append(dict(site=site, kind=kind, arm=arm, budget=budget,
                                            boundary=boundary, n_seeds=len(rs), n_excluded=10 - len(rs),
                                            metrics={k: stats([r[k] for r in rs]) for k in M.METRICS}))
    return cells, windows


def judge(cells):
    hard = [c for c in cells if c["site"] == "f2" and c["kind"] == "hard" and c["arm"] != "R"]
    easy = [c for c in cells if c["site"] == "f2" and c["kind"] == "easy" and c["arm"] != "R"]
    if len(hard) != 9 or len(easy) != 9 or any(c["n_seeds"] != 10 for c in hard + easy):
        raise RuntimeError("the registered nine cells/ten seeds are incomplete")
    j1 = sum(c["metrics"]["agree_rand"]["median"] >= .85 for c in hard)
    j2 = sum(c["metrics"]["ratio_L"]["median"] <= .30 for c in hard)
    j3s = sum(c["metrics"]["class_sink"]["median"] >= .75 for c in easy)
    j3l = sum(c["metrics"]["class_sink"]["median"] <= .25 for c in easy)
    return dict(J1=dict(verdict="RAND_PREDICTS_SIGN" if j1 >= 8 else "RAND_MISSES_SIGN",
                        qualifying_cells=j1, total_cells=9, required_cells=8, threshold="agree_rand >= 0.85"),
                J2=dict(verdict="LABEL_TERM_SMALL" if j2 >= 8 else "LABEL_TERM_LARGE",
                        qualifying_cells=j2, total_cells=9, required_cells=8, threshold="ratio_L <= 0.30"),
                J3=dict(verdict="UNSEEN_ROWS_SINK" if j3s >= 8 else "UNSEEN_ROWS_LIFT" if j3l >= 8 else "MIXED",
                        sink_cells=j3s, lift_cells=j3l, total_cells=9, required_cells=8,
                        threshold_sink="class_sink >= 0.75", threshold_lift="class_sink <= 0.25"))


def table(cells, site, kind, metrics, boundaries=False):
    selected = [c for c in cells if c["site"] == site and c["kind"] == kind]
    header = ["腕", "予算"] + (["境界"] if boundaries else []) + ["seed数", "除外"] + list(metrics)
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for c in selected:
        vs = [ARM_LABEL[c["arm"]], str(c["budget"])] + ([str(c["boundary"])] if boundaries else [])
        vs += [str(c["n_seeds"]), str(c["n_excluded"])]
        vs += [fmt(c["metrics"][k], 1 if k.startswith("n_") else 3) for k in metrics]
        lines.append("| " + " | ".join(vs) + " |")
    return "\n".join(lines)


def verdict_table(verdict):
    lines = ["| 判定 | 結果 | 9マス中 |", "|---|---|---|"]
    for k, v in verdict.items():
        count = f"沈める {v['sink_cells']}/9、持ち上げる {v['lift_cells']}/9" if k == "J3" else f"{v['qualifying_cells']}/9"
        lines.append(f"| {k} | {v['verdict']} | {count} |")
    return "\n".join(lines)


def check_table(checks):
    if not all(k in checks for k in "abcde"):
        lines = ["| 検査 | 結果 | 数値 |", "|---|---|---|"]
        for r in checks.get("checks", []):
            nums = {k: v for k, v in r.items() if k.endswith("_rel") or k == "relative_error"}
            lines.append(f"| {r['name']} {r.get('site', '')} | {r['status']} | {nums} |")
        lines.append("| 全状態の (b)・(d)・(e) | 未実行 | preflight 未通過 |")
        return "\n".join(lines)
    a, b, c, d, e = [checks[k] for k in "abcde"]
    av = max(a[k] for k in a if k.startswith("max_"))
    ev = max(e[k] for k in e if k.startswith("center_"))
    return "\n".join([
        "| 検査 | 結果 | 数値 |", "|---|---|---|",
        f"| (a) 順伝播 | {a['status']} | {a['n_states']}状態、最大相対誤差 {av:.3e} |",
        f"| (b) 2恒等式 | {b['status']} | {b['n_rows']}行、{b['real_identity_relmax']:.3e} / {b['rand_identity_relmax']:.3e}、easy L最大 {b['easy_L_absmax']:.1e} |",
        f"| (c) autograd符号 | {c['status']} | FC2 {c['f2_relative_error']:.3e}、FC1 {c['f1_relative_error']:.3e} |",
        f"| (d) 5回のラベル置換 | {d['status']} | FC2: 0付近 {d['f2']['n_near_zero_pass']}/{d['f2']['n_states']}、agree改善・同等 {d['f2']['n_agree_pass']}/{d['f2']['n_states']}、差の中央値 {d['f2']['agree_difference_median']:.4f} |",
        f"| (e) head中心化 | {e['status']} | {e['n_rows']}行、最大相対誤差 {ev:.3e} |"])


def incomplete(checks):
    inventory = checks.get("preflight", {}).get("checkpoints", [])
    provfile = OUT / "provenance.json"
    if not provfile.exists():
        prov = M.provenance(torch_device(checks), inventory, checks.get("preflight", {}).get("chunk", 50),
                            checks.get("started_utc", M.utc_now()))
        prov.update(status="STOPPED_PREFLIGHT", finished_utc=checks.get("finished_utc"),
                    error=checks.get("error"), n_rows=0)
        M.write_json(provfile, prov)
    verdict = dict(status="NOT_EVALUATED", reason="preflight未通過または全状態の計測未完了",
                   **{j: dict(verdict="NOT_EVALUATED", qualifying_cells=None, total_cells=9)
                      for j in ("J1", "J2", "J3")})
    M.write_json(OUT / "verdict.json", verdict)
    (OUT / "tables.md").write_text("# 集約未実施\n\n全状態の解析を完了していないため、判定用の9マス・seed中央値と範囲は未計算。\n")
    text = ("# cnn5p1_labelterm_1023 — 実行停止\n\n"
            "一行の結論: 必須の preflight を通過していない、または全状態の計測が完了していないため、J1〜J3 は判定していない。\n\n"
            f"停止の記録: `{checks.get('error', 'provenance.json参照')}`\n\n"
            + check_table(checks) + "\n\n"
            "計算定義・判定・閾値は変更していない。GPU が見えないため CPU で実行。Git 管理領域が読み取り専用のため commit は未作成。\n\n"
            "未測定の値・除外数を推測して埋めていない。固定状態の期待勾配であり、Adam の実更新・上流の変化・課題中の効果は扱わない。\n")
    (OUT / "summary.md").write_text(text)


def torch_device(checks):
    import torch
    return torch.device(checks.get("preflight", {}).get("device", "cpu"))


def audit_alive_definition():
    """Descriptive audit added after measurement; never changes the registered mask."""
    records = []
    for fn in sorted((OUT / "units").glob("*.npz")):
        with np.load(fn, allow_pickle=False) as ar:
            meta = json.loads(str(ar["metadata"]))
            alive, real, weight = ar["alive"], ar["S_real"], ar["live_weight"]
            for i, state in enumerate(meta):
                bad = (~alive[i]) & (real[i] != 0)
                if bad.any():
                    records.append(dict(**state, units=np.where(bad)[0].tolist(),
                                        live_weight=weight[i, bad].tolist(), S_real=real[i, bad].tolist()))
    audit = dict(row_count=len(records), unit_count=sum(len(r["units"]) for r in records),
                 f2_rows=sum(r["site"] == "f2" for r in records),
                 f2_units=sum(len(r["units"]) for r in records if r["site"] == "f2"),
                 note="Post-measurement descriptive audit: registered sum K abs(phi_prime)>0 kept unchanged. "
                      "Signed K makes the S=0 parenthetical false.", records=records)
    M.write_json(OUT / "alive_definition_audit.json", audit)
    return audit


def main():
    checks = json.loads((OUT / "checks.json").read_text())
    prov_path = OUT / "provenance.json"
    prov = json.loads(prov_path.read_text()) if prov_path.exists() else {}
    if not checks["preflight"]["passed"] or prov.get("status") not in ("MEASURED", "COMPLETED"):
        incomplete(checks)
        return
    rows = read_rows()
    cells, windows = aggregate(rows)
    verdict = judge(cells)
    audit = audit_alive_definition()
    excluded = json.loads((OUT / "excluded_states.json").read_text())
    M.write_json(OUT / "verdict.json", dict(status="EVALUATED", **verdict, cells=cells,
                                           aggregation="equal boundary mean within seed, then seed median and min/max",
                                           checks_all_passed=checks["all_checks_passed"]))
    M.write_json(OUT / "window_summary.json", windows)
    flattened = []
    for w in windows:
        flat = {k: v for k, v in w.items() if k != "metrics"}
        for k, st in w["metrics"].items():
            flat.update({f"{k}_{q}": v for q, v in st.items()})
        flattened.append(flat)
    M.write_csv(OUT / "window_summary.csv", flattened)
    hard_metrics = ("agree_rand", "ratio_L", "L_sink", "sink_real", "sink_rand", "n_alive")
    easy_metrics = ("class_sink", "ratio_class", "agree_rand", "sink_real", "sink_unif", "n_alive")
    tables = ["# 全集約表", "値は seed 中央値 [最小, 最大]。判定用の表では先に seed 内で境界を平均。",
              verdict_table(verdict)]
    for site in M.SITES:
        for kind in ("hard", "easy"):
            tables += [f"## {site} {kind}: seed 内の境界平均", table(cells, site, kind, M.METRICS),
                       f"## {site} {kind}: 境界別", table(windows, site, kind, M.METRICS, boundaries=True)]
    (OUT / "tables.md").write_text("\n\n".join(tables) + "\n")
    n_switch_excluded = sum(r["excluded"] for r in rows if r["site"] == "f2")
    n_live_empty = sum(not r["excluded"] and r["n_alive"] == 0 for r in rows)
    f1hard = [c for c in cells if c["site"] == "f1" and c["kind"] == "hard" and c["arm"] != "R"]
    f1easy = [c for c in cells if c["site"] == "f1" and c["kind"] == "easy" and c["arm"] != "R"]
    def extent(cs, metric):
        vs = [c["metrics"][metric]["median"] for c in cs]
        return f"{min(vs):.3f}〜{max(vs):.3f}"
    nonalive_nonzero = sum(r["nonalive_S_absmax"] > 0 for r in rows)
    summary = ["# cnn5p1_labelterm_1023 — 保存状態の切替時の押し",
               f"一行の結論: J1={verdict['J1']['verdict']} ({verdict['J1']['qualifying_cells']}/9)、"
               f"J2={verdict['J2']['verdict']} ({verdict['J2']['qualifying_cells']}/9)、"
               f"J3={verdict['J3']['verdict']} (沈める {verdict['J3']['sink_cells']}/9、持ち上げる {verdict['J3']['lift_cells']}/9)。",
               "**spec からの逸脱・明確化**: 計算定義、J1〜J3、閾値、窓、予測の変更はない。"
               "除外規則は旧kappa実装の train_acc < 0.3 を使用した。spec括弧の『train精度0』には例外があり、"
               "7800・境界19・ReLU seed2009 は精度0.2かつFC2全死で、旧解析でも除外されている。"
               "初回は余分に付けた精度==0の確認で停止し、この確認だけを削除してpreflightから再計算した。"
               f"生存条件 sum K|φ′|>0 は変更していないが、その補足『それ以外はS=0』は "
               f"{audit['row_count']}行・延べ{audit['unit_count']} unitで不成立だった。"
               f"Kが負を取り得るためで、FC2はeasyの{audit['f2_rows']}状態・{audit['f2_units']} unit、残りはFC1。"
               "事後の定義監査として alive_definition_audit.json に反例を保存した。"
               "(d) の『0付近』と5置換の集約は spec に数値の指定がないため、結果を見る前に implementation.md の方法を固定した。"
               "この追加の検査基準を登録済みの閾値とは呼ばない。CUDA がこの環境から見えないため CPU で実行した。"
               "Git 管理領域が読み取り専用で commit が拒否され、実装は未commit。実行時HEADとソースSHA256を provenance.json に記録した。",
               "## 判定", verdict_table(verdict),
               "非ReLU 3腕×3予算の9マス。hard は境界1・9・19、easy は境界1・9・19・29。"
               "各 seed 内で境界を等重み平均し、その10 seedの中央値で判定した。同じ保存状態の再集計を独立な追試とは数えない。",
               "## FC2 hard", "中央値 [seed間の最小, 最大]。S>0が沈める側、L<0が実ラベルの押しへ沈める向きに加わる。",
               table(cells, "f2", "hard", hard_metrics),
               "## FC2 easy", table(cells, "f2", "easy", easy_metrics),
               "## 境界ごとの FC2 hard", table(windows, "f2", "hard", hard_metrics, True),
               "## 境界ごとの FC2 easy", table(windows, "f2", "easy", ("class_sink", "ratio_class", "n_alive"), True),
               "## ReLU・FC1 の記録",
               f"ReLU を判定の9マスへ入れない。崩壊は {len(excluded)} checkpoint状態、"
               f"次課題との組は {n_switch_excluded}/840 状態、層を分けた行では {2*n_switch_excluded}/1680 行を集計から除外。"
               "除外行は rows.csv と unit配列に保持。ReLU の境界平均は残った境界だけを使い、その数を verdict.json の seed_means に保存した。",
               f"FC1 の非ReLU9マスの中央値の範囲: hard agree_rand {extent(f1hard, 'agree_rand')}、"
               f"ratio_L {extent(f1hard, 'ratio_L')}、L_sink {extent(f1hard, 'L_sink')}。"
               f"easy class_sink {extent(f1easy, 'class_sink')}、ratio_class {extent(f1easy, 'ratio_class')}。"
               "FC1は同じ局所重み経路の連鎖律で計算し、判定には使わない。全量と全窓は tables.md・window_summary.csv。",
               f"生存 unit 数は各表に併記。非除外で生存0の行は {n_live_empty}。"
               f"生存条件を満たさない unit に非ゼロの S がある行は {nonalive_nonzero}（K の符号も rows.csv に記録）。",
               "## 検査", check_table(checks),
               "(a) は訓練と旧kappa解析の同じ自分の課題の画像で照合。主計算のφ′は訓練のautograd式。"
               "(c) は指定の ELU 780 seed2000 境界9→hard課題11 の全2500枚、η=1。更新は適用していない。",
               "(d) は方向性を含む置換対照であり、数値恒等式の検査とは性質が違う。"
               "各状態5回のagree平均が本物以上かを厳密に記録し、一部の不成立を全体平均で隠していない。"
               "Lの生値・3σ帯・5回の値は permutation_checks.json。"
               "(d)がFAILなので、置換対照が全状態で実ラベル項の非偶然性を確認したとは言わない。",
               "## 言わないこと",
               "固定状態の1歩の期待勾配の帳簿であり、Adamの分母を使った実更新ではない。上流の入力aの変化を含まない。"
               "課題内の戻り・長期の学習性能・新しい独立な追試は扱わない。実ラベルが乱数ラベルと同じとは言わない。",
               "## 来歴と生成物",
               f"実行時HEAD: `{prov['git_hash']}`。Python {prov['python']}、torch {prov['torch']}、device={prov['device']}。"
               f"{prov['started_utc']}〜{prov['finished_utc']}。checkpointはarchiveから読み取り専用で使用し、15ファイルのSHA256をmanifestと照合。",
               "実装・検査・集約コードと小さい結果ファイルはcommit対象だが、環境の書込制限でcommit未作成。"
               "意図してgit外に置く生成物は units/*.npz（21ファイル）と logs/*.log。"
               "checkpointのコピー、別worktree・clone・vaultへの書込、pushは行っていない。"]
    (OUT / "summary.md").write_text("\n\n".join(summary) + "\n")
    prov.update(status="COMPLETED", summarized_utc=M.utc_now(), all_checks_passed=checks["all_checks_passed"],
                code_committed=False, commit_failure="index.lock: Read-only file system",
                aggregation_source_sha256=M.sha256(Path(__file__)),
                source_changes_after_measurement=[dict(path=str(Path(__file__).relative_to(ROOT)),
                                                       reason="Report-only addition of the alive-mask counterexample audit; J1-J3 unchanged")],
                previous_attempts=[str(p.relative_to(ROOT)) for p in sorted((OUT / "attempts").glob("*.json"))],
                excluded_checkpoint_states=len(excluded), excluded_switch_states=n_switch_excluded,
                excluded_rows=2*n_switch_excluded,
                generated_not_for_git=[dict(path=str(p.relative_to(ROOT)), bytes=p.stat().st_size, sha256=M.sha256(p))
                                       for p in sorted((OUT / "units").glob("*.npz"))] +
                                      [dict(path=str(p.relative_to(ROOT)), bytes=p.stat().st_size, sha256=M.sha256(p))
                                       for p in sorted((OUT / "logs").glob("*.log"))])
    M.write_json(prov_path, prov)
    print(json.dumps(verdict, indent=2))
    print(f"rows={len(rows)}; excluded checkpoint states={len(excluded)}; excluded switch states={n_switch_excluded}")


if __name__ == "__main__":
    main()
