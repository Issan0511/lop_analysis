#!/usr/bin/env python3
"""D の検査（spec §7 の D の前半の補い）: データ列・規則・probe の sha を CPU で作り直し、本走 4 ブロック C3 の来歴と照合する。

写し元の関数 C.task_cpu_M（$CG/src/chargpt_cl_1019.py、git blob cc524f85d5f2d9813a272abb6b9476b065f1c17b）を
そのまま呼ぶ（走る網と同じ関数。データ列は CPU の乱数で seed と課題だけから作るので、層の数と腕によらない）。
参照 = $CG/results/chargpt_cl_1019/M/C3/seed<k>/provenance.json の tasks_info（課題 1〜50、全長の sha256）。
GPU の短い試走の per_task.csv の照合は check_d.py が行う。これは課題 1〜50 の全部を CPU で見る補い。
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

CG = Path(os.environ.get("CHARGPT_SRC", "/home/kubo/project/Nakatsuka/lop_analysis_chargpt_cl_1019"))
OUT = Path("/home/kubo/project/Nakatsuka/lop_seed10/results/chargpt_seed10_1022/checks/D")


def one(seed):
    import torch
    torch.set_num_threads(2)
    sys.path.insert(0, str(CG / "src"))
    import chargpt_cl_1019 as C
    ref = json.loads((CG / "results/chargpt_cl_1019/M/C3" / f"seed{seed}" / "provenance.json").read_text())["tasks_info"]
    ref = {r["task"]: r for r in ref}
    rows = []
    for t in sorted(ref):
        c = C.task_cpu_M(seed, t, 10000, 0.01)
        rows.append(dict(seed=seed, task=t, data=c["data_sha"] == ref[t]["data_sha"], rule=c["rule_sha"] == ref[t]["rule_sha"],
                         probe=c["probe_sha"] == ref[t]["probe_sha"], data_sha=c["data_sha"][:16]))
    return rows


def main():
    with ProcessPoolExecutor(max_workers=4) as ex:
        res = [r for rows in ex.map(one, range(10)) for r in rows]
    bad = [r for r in res if not (r["data"] and r["rule"] and r["probe"])]
    summ = dict(n=len(res), n_seeds=len({r["seed"] for r in res}), tasks=[min(r["task"] for r in res), max(r["task"] for r in res)],
                n_data_eq=sum(r["data"] for r in res), n_rule_eq=sum(r["rule"] for r in res), n_probe_eq=sum(r["probe"] for r in res),
                bad=bad, ok=not bad, ref="lop_analysis_chargpt_cl_1019/results/chargpt_cl_1019/M/C3/seed<k>/provenance.json tasks_info")
    (OUT / "streams_cpu.json").write_text(json.dumps(dict(summary=summ, rows=res), indent=1))
    print(json.dumps({k: v for k, v in summ.items() if k != "bad"}), "n_bad", len(bad))


if __name__ == "__main__":
    main()
