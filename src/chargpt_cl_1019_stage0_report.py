#!/usr/bin/env python3
"""chargpt_cl_1019 段 0 の集計と選択（spec §7・追補 0 §H）。

  python src/chargpt_cl_1019_stage0_report.py --root results/chargpt_cl_1019/stage0 --sec_per_update_M x --sec_per_update_W y

選ぶ規則（数値を見る前に追補 0 で決めた）:
  箱 M: 各 α で「課題 1 の末の超過損失 ≤ 0.1 nat」に 3 seed すべてが届く最小の N_M。
        「更新 100 の超過損失 ≥ 0.5 nat」を 3 seed すべてで満たす α だけを候補にする。
  箱 W: 「課題 1 の末の memo_acc ≥ 0.95」に 3 seed すべてが届く最小の N_W。
  候補の α が複数あれば、本走全体の ETA ≤ 10 時間のものの中で最小の α。
"""
import argparse
import csv
import json
from pathlib import Path

ALPHAS = (0.01, 0.03, 0.1)
NM = (1000, 3000, 10000)
NW = (3000, 7500, 30000)
SEEDS = (100, 101, 102)


def read_csv(p):
    return list(csv.DictReader(open(p)))


def task1(d):
    pt = read_csv(d / "per_task.csv")
    ev = read_csv(d / "eval_trace.csv")
    e100 = [float(r["eval"]) for r in ev if r["task"] == "1" and r["u"] == "100"]
    prov = json.loads((d / "provenance.json").read_text())
    return dict(end1=float(pt[0]["end_metric"]), e100=e100[0] if e100 else None,
                ends=[float(r["end_metric"]) for r in pt], online=[float(r["online_all"]) for r in pt],
                task_secs=prov["task_secs"], wall=prov["wall_sec"], H=[float(r["H"]) for r in pt])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    root = Path(a.root)
    rep = dict(M={}, W={})
    for al in ALPHAS:
        for n in NM:
            cfg = f"a{al}_n{n}"
            rep["M"][cfg] = {s: task1(root / "M" / cfg / f"seed{s}") for s in SEEDS
                             if (root / "M" / cfg / f"seed{s}" / "provenance.json").exists()}
    for n in NW:
        cfg = f"n{n}"
        rep["W"][cfg] = {s: task1(root / "W" / cfg / f"seed{s}") for s in SEEDS
                         if (root / "W" / cfg / f"seed{s}" / "provenance.json").exists()}

    sel = dict(M={}, W=None)
    for al in ALPHAS:
        reach = None
        for n in NM:
            r = rep["M"][f"a{al}_n{n}"]
            if len(r) == 3 and all(v["end1"] <= 0.1 for v in r.values()):
                reach = n
                break
        # 更新 100 の値は N によらず同じ（課題 1 の頭は同じ batch 列）。最小の N の走で読む
        r0 = rep["M"][f"a{al}_n{NM[0]}"]
        hard = len(r0) == 3 and all(v["e100"] is not None and v["e100"] >= 0.5 for v in r0.values())
        sel["M"][str(al)] = dict(min_N_reaching=reach, not_too_easy=hard,
                                 e100=[r0[s]["e100"] for s in r0], candidate=bool(reach and hard))
    for n in NW:
        r = rep["W"][f"n{n}"]
        if len(r) == 3 and all(v["end1"] >= 0.95 for v in r.values()):
            sel["W"] = n
            break

    lines = ["# chargpt_cl_1019 段 0", "", "## 箱 M（C3・seed 100〜102・5 課題）", "",
             "| α | N_M | 課題 1 末の超過損失（3 seed） | 更新 100 の超過損失 | 課題 1〜5 末の超過損失（seed 中央） |",
             "| -- | -- | -- | -- | -- |"]
    import statistics as st
    for al in ALPHAS:
        for n in NM:
            r = rep["M"][f"a{al}_n{n}"]
            if not r:
                continue
            e1 = " ".join(f"{v['end1']:.3f}" for v in r.values())
            e100 = " ".join(f"{v['e100']:.3f}" for v in r.values())
            med = " ".join(f"{st.median(v['ends'][t] for v in r.values()):.3f}" for t in range(len(next(iter(r.values()))["ends"])))
            lines.append(f"| {al} | {n} | {e1} | {e100} | {med} |")
    lines += ["", "## 箱 W（C3・seed 100〜102・5 課題）", "",
              "| N_W | 課題 1 末の memo_acc（3 seed） | 課題 1〜5 末の memo_acc（seed 中央） | 課題 1〜5 の online_acc（seed 中央） |",
              "| -- | -- | -- | -- |"]
    for n in NW:
        r = rep["W"][f"n{n}"]
        if not r:
            continue
        e1 = " ".join(f"{v['end1']:.3f}" for v in r.values())
        med = " ".join(f"{st.median(v['ends'][t] for v in r.values()):.3f}" for t in range(len(next(iter(r.values()))["ends"])))
        onl = " ".join(f"{st.median(v['online'][t] for v in r.values()):.3f}" for t in range(len(next(iter(r.values()))["online"])))
        lines.append(f"| {n} | {e1} | {med} | {onl} |")
    lines += ["", "## 規則による選択（ETA の判定の前）", "", "```", json.dumps(sel, indent=1, ensure_ascii=False), "```"]
    out = Path(a.out) if a.out else root
    (out / "stage0_report.md").write_text("\n".join(lines) + "\n")
    (out / "stage0_selection.json").write_text(json.dumps(dict(selection=sel, raw=rep), indent=1, ensure_ascii=False))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
