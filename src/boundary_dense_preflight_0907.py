# -*- coding: utf-8 -*-
"""boundary_dense_0907 の短縮走行（spec §7）: 5 腕 × 30k ＋ **S-null**。

    OMP_NUM_THREADS=1 PYTHONPATH=. python3 -m src.boundary_dense_preflight_0907 [--steps 30000]

この実験の生命線は「記録格子を変えても軌道が 1 bit も動かない」こと。それを仮定でなく
**実測**する:

  S-null      `BDref_1216`（境界密・30k で 7 記録点）の `state_hash_final` が、
              `offset_grid_0906` の 30k 前検査の `LRoff0_1216`（標準格子・31 記録点）と一致。
              共有 step（タスク終端）では共通列がバイト一致すること。
  S-null-対   `BDa0p5_1216`（a=0.5）は同じ相手と一致しては**いけない**（空虚な検査よけ）。
  S-grid      実ログの `step` 列が `probe_steps` と一致し、密なオフセットが入っていること。
  S-col       `layer1_b` が入っていて、既存列が 1 つも消えていないこと。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import act_offset_preflight_0906 as P
from . import boundary_dense_0907 as B
from . import edge_law_0905 as E
from .common import ROOT, load_config

EXPERIMENT = "boundary_dense_0907"
GRID_PREFLIGHT = Path(ROOT) / "results/_preflight_offset_grid_0906/arms"
EXPECTED_DIVERGENT: tuple[str, ...] = ()
# 記録格子に依存する列（軌道の性質ではない）。30k 前検査で判明し spec §8 に登録:
# `EluRecorder` は dzbar を「直前の記録点がちょうど `interval`(=1000) step 前」のときだけ
# 埋め、そうでなければ NaN を書く。境界密格子では隣接条件がほぼ成り立たないので dzbar は
# 定義されない。**本走で dzbar を読んではいけない**（解析は z̄ と b の差分だけを使う）。
GRID_DEPENDENT_COLUMNS = ("layer1_dzbar",)


def _load(d: Path, arm: str, seed: int) -> dict:
    with np.load(Path(d) / "logs" / f"{arm}_seed{seed}.npz", allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def s_null(out: Path, arm: str, other: str, seeds=range(10)) -> dict:
    """共有 step（タスク終端）で共通列がバイト一致し、state_hash も一致すること。"""
    rows, ok = [], True
    for s in seeds:
        a, b = _load(out / "arms", arm, s), _load(GRID_PREFLIGHT, other, s)
        ia = {int(v): i for i, v in enumerate(a["step"])}
        shared = [int(v) for v in b["step"] if int(v) in ia]
        sel_a = np.array([ia[v] for v in shared])
        sel_b = np.array([i for i, v in enumerate(b["step"]) if int(v) in ia])
        bad = []
        for k in sorted((set(a) & set(b)) - set(P.SKIP_COLUMNS) - set(GRID_DEPENDENT_COLUMNS)):
            xa, xb = np.asarray(a[k]), np.asarray(b[k])
            if xa.ndim >= 1 and xa.shape[0] == len(a["step"]) and xb.shape[0] == len(b["step"]):
                xa, xb = xa[sel_a], xb[sel_b]
            if xa.dtype != xb.dtype or xa.shape != xb.shape or xa.tobytes() != xb.tobytes():
                bad.append(k)
        h = str(a["state_hash_final"]) == str(b["state_hash_final"])
        ok &= h and not bad
        rows.append(dict(seed=int(s), state_hash_equal=bool(h), n_shared=len(shared),
                         mismatched=bad[:12], n_mismatched=len(bad)))
    return dict(pass_=bool(ok), arm=arm, other=other, rows=rows,
                n_mismatched_total=int(sum(r["n_mismatched"] for r in rows)),
                state_hash_all_equal=bool(all(r["state_hash_equal"] for r in rows)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=P.SHORT_STEPS)
    ap.add_argument("--outdir", default=str(Path(ROOT) / f"results/_preflight_{EXPERIMENT}"))
    a = ap.parse_args()
    out = Path(a.outdir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    reg = B.registered()
    arms = [str(r["name"]) for r in reg["arms"]]
    report = dict(experiment=EXPERIMENT, steps=int(a.steps), config=str(B.CONFIG),
                  git_head=E._git_head(), runs=[], checks=[], diverged_arms=[])

    for arm in arms:
        r = B.run_arm(arm, steps=int(a.steps), outdir=out / "arms")
        report["runs"].append({k: v for k, v in r.items() if k != "sanity"})
        if r["status"] == E.NUMERIC_DIVERGENCE:
            report["diverged_arms"].append(arm)
            report["checks"].append(dict(check="short_run", pass_=False, arm=arm, diverged=True))
            print(f"[short {arm}] DIVERGED", flush=True)
            continue
        c1 = P.check_run(out / "arms", arm, a.steps, 0.01)
        # 格子依存列（dzbar）の NaN は設計どおりなので合否から外す。それ以外の
        # 非有限・欠け・lr 不一致は落とす（規則を緩めたのはこの 1 列だけ）。
        relaxed = all(not (set(r["nonfinite"]) - set(GRID_DEPENDENT_COLUMNS))
                      and not r["missing"] and float(r["lr_used"]) == 0.01
                      for r in c1["rows"])
        report["checks"].append(dict(check="short_run", diverged=False, arm=arm,
                                     pass_=bool(relaxed), raw_pass=bool(c1["pass_"]),
                                     grid_dependent_nan=list(GRID_DEPENDENT_COLUMNS),
                                     finite_lr=c1))
        print(f"[short {arm}] {'PASS' if relaxed else 'FAIL'}", flush=True)

    # S-grid / S-col
    want = B.probe_steps(int(a.steps), [min(100_000, max(0, int(a.steps) - B.PERIOD)),
                                        max(0, int(a.steps) - B.PERIOD)],
                         B.grid_of(reg)["offsets"])
    z = _load(out / "arms", "BDref_1216", 0)
    grid_ok = [int(v) for v in z["step"]] == want
    dense = [s for s in want if s % B.PERIOD not in (0,)]
    report["checks"].append(dict(check="S-grid", pass_=bool(grid_ok and len(dense) >= 3),
                                 arm="BDref_1216", n_probes=len(want),
                                 n_dense=len(dense), want_head=want[:8]))
    print(f"[S-grid] {'PASS' if report['checks'][-1]['pass_'] else 'FAIL'} "
          f"({len(want)} 点・うち密 {len(dense)} 点)", flush=True)
    ref = _load(GRID_PREFLIGHT, "LRoff0_1216", 0)
    missing = (set(ref) - set(z)) - set(P.SKIP_COLUMNS)
    col_ok = ("layer1_b" in z) and not missing
    report["checks"].append(dict(check="S-col", pass_=bool(col_ok), has_b=("layer1_b" in z),
                                 missing=sorted(missing)[:12],
                                 b_shape=list(np.asarray(z["layer1_b"]).shape)))
    print(f"[S-col] {'PASS' if col_ok else 'FAIL'} (layer1_b {list(np.asarray(z['layer1_b']).shape)}"
          f"・欠け {sorted(missing)[:6]})", flush=True)

    # S-null と、その変異体の対
    c = s_null(out, "BDref_1216", "LRoff0_1216")
    c["check"] = "S-null"
    report["checks"].append(c)
    print(f"[S-null BDref_1216 vs LRoff0_1216] {'PASS' if c['pass_'] else 'FAIL'} "
          f"(hash 一致 {c['state_hash_all_equal']}・不一致列 {c['n_mismatched_total']})", flush=True)
    m = s_null(out, "BDa0p5_1216", "LRoff0_1216")
    report["checks"].append(dict(check="S-null-mutant", arm="BDa0p5_1216",
                                 pass_=bool(not m["pass_"]),
                                 n_mismatched_total=m["n_mismatched_total"],
                                 state_hash_all_equal=m["state_hash_all_equal"]))
    print(f"[S-null-mutant BDa0p5_1216] "
          f"{'PASS' if report['checks'][-1]['pass_'] else 'FAIL'} "
          f"(差が {m['n_mismatched_total']} 列)", flush=True)

    unexpected = [x for x in report["checks"]
                  if not x["pass_"] and not (x.get("diverged") and x.get("arm") in EXPECTED_DIVERGENT)]
    missing_div = [x for x in EXPECTED_DIVERGENT if x not in report["diverged_arms"]]
    report["expected_divergent"] = list(EXPECTED_DIVERGENT)
    report["unexpected_failures"] = [x.get("arm", x.get("check")) for x in unexpected]
    report["missing_expected_divergence"] = missing_div
    report["all_checks_pass"] = bool(all(x["pass_"] for x in report["checks"]))
    report["pass_"] = bool(not unexpected and not missing_div)
    (out / "preflight.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"[preflight] {'PASS' if report['pass_'] else 'FAIL'} "
          f"(失敗 {report['unexpected_failures']}) -> {out / 'preflight.json'}")
    raise SystemExit(0 if report["pass_"] else 1)


if __name__ == "__main__":
    main()
