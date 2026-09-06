# -*- coding: utf-8 -*-
"""offset_grid_0906 の短縮走行（spec §4）: 27 腕 × 30k を回し、有限・lr_used・freeze_v・batch_mode を確認する。

    OMP_NUM_THREADS=1 PYTHONPATH=. python3 -m src.offset_grid_preflight_0906 [--steps 30000]

act_offset_preflight_0906 の run_short / check_run をそのまま使う（config だけ差し替え）。
結果は results/_preflight_offset_grid_0906/preflight.json。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import act_offset_preflight_0906 as P
from . import edge_law_0905 as E
from .common import ROOT, load_config

CONFIG = Path(ROOT) / "configs" / "offset_grid_0906.yaml"
EXPERIMENT = "offset_grid_0906"
# 30k 前検査で判明し spec の Log に登録した発散: b_offset +2 は全ユニットの出力を一様に +2 ずらすので
# c=+2 と同じコヒーレント曲率 λ ≈ 2h·2² = 800 が立ち、lr 0.01 で step 1,000 に落ちる（edge_law の
# ±5/±4 も同じ機構）。ブロック 7 は NOT_RUN として記録する。
EXPECTED_DIVERGENT = ("LRbp2_1216", "Ebp2_1216")


def hook_expectations(row: dict) -> dict:
    h = row["hook"] or {}
    return dict(lr=float(h["value"]) if h.get("type") == "lr" else 0.01,
                freeze_v=bool(h.get("type") == "v_freeze"),
                batch_mode="full32" if h.get("type") == "full_batch" else "online")


def check_hooks(outdir: Path, arm: str, want: dict, seeds=range(10)) -> dict:
    rows, ok = [], True
    for s in seeds:
        with np.load(Path(outdir) / "logs" / f"{arm}_seed{s}.npz", allow_pickle=True) as z:
            got = dict(lr=float(z["lr_used"]), freeze_v=bool(z["freeze_v"]),
                       batch_mode=str(z["batch_mode"]))
        good = all(got[k] == want[k] for k in want)
        ok &= good
        rows.append(dict(seed=int(s), got=got, ok=good))
    return dict(pass_=bool(ok), arm=arm, want=want, rows=rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=P.SHORT_STEPS)
    ap.add_argument("--outdir", default=str(Path(ROOT) / f"results/_preflight_{EXPERIMENT}"))
    a = ap.parse_args()
    out = Path(a.outdir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    table = E.arm_table(load_config(str(CONFIG)))
    report = dict(experiment=EXPERIMENT, steps=int(a.steps), config=str(CONFIG),
                  git_head=E._git_head(), runs=[], checks=[], diverged_arms=[])
    for arm, row in table.items():
        r = P.run_short(arm, out / "arms", a.steps, CONFIG)
        report["runs"].append(r)
        if r["status"] == "NUMERIC_DIVERGENCE":
            report["diverged_arms"].append(arm)
            report["checks"].append(dict(check="short_run", pass_=False, arm=arm, diverged=True,
                                         detail=P.divergence_detail(out / "arms", arm)))
            print(f"[short {arm}] DIVERGED {report['checks'][-1]['detail']}", flush=True)
            continue
        want = hook_expectations(row)
        c1 = P.check_run(out / "arms", arm, a.steps, want["lr"])
        c2 = check_hooks(out / "arms", arm, want)
        report["checks"].append(dict(check="short_run", diverged=False, arm=arm,
                                     pass_=bool(c1["pass_"] and c2["pass_"]),
                                     finite_lr=c1, hooks=c2))
        print(f"[short {arm}] {'PASS' if report['checks'][-1]['pass_'] else 'FAIL'} "
              f"lr={want['lr']} freeze_v={want['freeze_v']} batch={want['batch_mode']}", flush=True)
    # 合否は act_offset の前検査と同じ両方向の規則: 登録済みの発散腕（spec Log・ブロック 7）が
    # ちょうど落ち、それ以外に失敗が無いこと。落ちるはずの腕が落ちなければ FAIL。
    unexpected = [c for c in report["checks"]
                  if not c["pass_"] and not (c.get("diverged") and c.get("arm") in EXPECTED_DIVERGENT)]
    missing = [a for a in EXPECTED_DIVERGENT if a not in report["diverged_arms"]]
    report["expected_divergent"] = list(EXPECTED_DIVERGENT)
    report["unexpected_failures"] = [c.get("arm", c.get("check")) for c in unexpected]
    report["missing_expected_divergence"] = missing
    report["all_checks_pass"] = bool(all(c["pass_"] for c in report["checks"]))
    report["pass_"] = bool(not unexpected and not missing)
    (out / "preflight.json").write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str),
                                        encoding="utf-8")
    print(f"[preflight] {'PASS' if report['pass_'] else 'FAIL'} (diverged {report['diverged_arms']} / "
          f"登録済みの発散を除く失敗 {report['unexpected_failures']} / 落ちるはずが落ちなかった {missing}) "
          f"-> {out / 'preflight.json'}")
    raise SystemExit(0 if report["pass_"] else 1)


if __name__ == "__main__":
    main()
