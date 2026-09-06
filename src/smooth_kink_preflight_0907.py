# -*- coding: utf-8 -*-
"""smooth_kink_0907 の短縮走行（spec §7）: 11 腕 × 30k を回し、有限性と S-null を確認する。

    OMP_NUM_THREADS=1 PYTHONPATH=. python3 -m src.smooth_kink_preflight_0907 [--steps 30000]

`act_offset_preflight_0906` の run_short / check_run / compare_logs をそのまま使う（config を
差し替えるだけ）。S-null は **`offset_grid_0906` の 30k 前検査の logs とバイト一致**すること——
`nets.py` に足した `SMOOTH_LEAKY` が既存の経路を 1 bit も動かしていないことの検査で、
「変異体が落ちる」対（平滑化腕は一致しない）も同時に置く。

合否は両側の規則: **登録済みの発散腕がちょうど落ち、それ以外に失敗が無いこと**。
`EXPECTED_DIVERGENT` は空で始める。s=3 は λ ≈ 2h((1-a)s·log2)² ≈ 700 で lr 0.01 の
安定域（< 2/λ ≈ 0.003）を超えるおそれがあり（spec §2）、落ちたら spec の Log に登録して
`NOT_RUN` にするか、追補で lr を下げる。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import act_offset_preflight_0906 as P
from . import edge_law_0905 as E
from .common import ROOT, load_config

CONFIG = Path(ROOT) / "configs" / "smooth_kink_0907.yaml"
EXPERIMENT = "smooth_kink_0907"
GRID_PREFLIGHT = Path(ROOT) / "results/_preflight_offset_grid_0906/arms"
# 30k 前検査（2026-09-07 朝）で判明し spec §8 の Log に登録した発散: a=0.1・s=3 は
# 初期値でほぼ全ユニットが折れ目付近にいるため隆起 (1-a)s·log2 = 1.87 が一様シフトとして
# 働き、λ ≈ 2h·1.87² ≈ 700 > 2/lr = 200 で step 1,000 に 10/10 落ちる（c=±2・b+2 と同機構）。
EXPECTED_DIVERGENT: tuple[str, ...] = ("SKs3_1216",)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=P.SHORT_STEPS)
    ap.add_argument("--outdir", default=str(Path(ROOT) / f"results/_preflight_{EXPERIMENT}"))
    a = ap.parse_args()
    out = Path(a.outdir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(str(CONFIG))
    table = E.arm_table(cfg)
    pairs = cfg["analysis"]["s_null_pairs"]
    report = dict(experiment=EXPERIMENT, steps=int(a.steps), config=str(CONFIG),
                  git_head=E._git_head(), runs=[], checks=[], diverged_arms=[])
    for arm in table:
        r = P.run_short(arm, out / "arms", a.steps, CONFIG)
        report["runs"].append(r)
        if r["status"] == "NUMERIC_DIVERGENCE":
            report["diverged_arms"].append(arm)
            report["checks"].append(dict(check="short_run", pass_=False, arm=arm, diverged=True,
                                         detail=P.divergence_detail(out / "arms", arm)))
            print(f"[short {arm}] DIVERGED {report['checks'][-1]['detail']}", flush=True)
            continue
        c1 = P.check_run(out / "arms", arm, a.steps, 0.01)
        report["checks"].append(dict(check="short_run", diverged=False, arm=arm,
                                     pass_=bool(c1["pass_"]), finite_lr=c1))
        print(f"[short {arm}] {'PASS' if c1['pass_'] else 'FAIL'}", flush=True)
    # S-null: 参照腕は offset_grid の 30k 前検査とバイト一致（既存経路が動いていない）
    for arm, other in pairs.items():
        c = P.compare_logs(out / "arms", arm, GRID_PREFLIGHT, other)
        c.update(check="S-null", arm=arm)
        report["checks"].append(c)
        print(f"[S-null {arm} vs {other}] {'PASS' if c['pass_'] else 'FAIL'} "
              f"(mismatch {c['n_mismatched_total']})", flush=True)
    # 対（変異体）: 平滑化腕は同じ相手と一致しては**いけない**
    mut = P.compare_logs(out / "arms", "SKs0p01_1216", GRID_PREFLIGHT, "LRoff0_1216")
    report["checks"].append(dict(check="S-null-mutant", arm="SKs0p01_1216",
                                 pass_=bool(not mut["pass_"]),
                                 n_mismatched_total=mut["n_mismatched_total"],
                                 state_hash_all_equal=mut["state_hash_all_equal"]))
    print(f"[S-null-mutant SKs0p01_1216] "
          f"{'PASS' if report['checks'][-1]['pass_'] else 'FAIL'} (差が {mut['n_mismatched_total']} 列)",
          flush=True)

    unexpected = [c for c in report["checks"]
                  if not c["pass_"] and not (c.get("diverged") and c.get("arm") in EXPECTED_DIVERGENT)]
    missing = [x for x in EXPECTED_DIVERGENT if x not in report["diverged_arms"]]
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
