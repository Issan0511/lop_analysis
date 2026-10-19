#!/usr/bin/env python3
"""chargpt_cl_1019 の S-cost（spec §8・§9）: 実行機で 1 腕 1 seed 2 課題の壁時計を測り、本走全体を見積もる。

  python src/chargpt_cl_1019_cost.py --nm 3000 --alpha 0.03 --nw 7500 --out results/chargpt_cl_1019/cost

測る走（GPU にほかの走が無い状態で、1 本ずつ）:
  full : 課題 1・2 とも §4.1 の細かい時点で測る（本走の課題 1・2・10・11・25・26・49・50 と同じ）
  light: 課題 1・2 とも更新 0・100・課題末だけ測る（本走のほかの 42 課題と同じ）
  fresh: 測定なしで 2 課題（fresh 3 回の費用）
本走 1 走 = 8·full + 42·light + 3·fresh（1 課題あたり）＋ 起動の固定費。全体 = 4 腕 × 10 seed × 両箱、1 本ずつ直列。
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PY = sys.executable
SRC = Path(__file__).resolve().parent / "chargpt_cl_1019.py"
ARMS = ("C3", "C4", "B3", "N4")


def run(out, box, arm, n, alpha, extra):
    cmd = [PY, str(SRC), "run", "--box", box, "--arm", arm, "--seed", "0", "--tasks", "2", "--n_per_task", str(n),
           "--out", str(out), "--ckpt", "0", "--fresh_tasks"] + (["--alpha", str(alpha)] if box == "M" else []) + extra
    t0 = time.time()
    subprocess.run(cmd, check=True, capture_output=True)
    wall = time.time() - t0
    prov = json.loads((Path(out) / "provenance.json").read_text())
    return dict(wall=wall, task_secs=prov["task_secs"], inner=prov["wall_sec"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nm", type=int, required=True)
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--nw", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", nargs="*", default=["C3", "B3"])
    ap.add_argument("--boxes", nargs="+", default=["M", "W"], choices=["M", "W"])
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = {}
    for box, n in (("M", a.nm), ("W", a.nw)):
        if box not in a.boxes:
            continue
        for arm in a.arms:
            r = {}
            r["full"] = run(out / f"{box}_{arm}_full", box, arm, n, a.alpha, [])
            r["light"] = run(out / f"{box}_{arm}_light", box, arm, n, a.alpha, ["--within_tasks"])
            r["fresh"] = run(out / f"{box}_{arm}_fresh", box, arm, n, a.alpha, ["--within_tasks", "--no_measure"])
            t_full = sum(r["full"]["task_secs"]) / 2
            t_light = sum(r["light"]["task_secs"]) / 2
            t_fresh = sum(r["fresh"]["task_secs"]) / 2
            fixed = r["light"]["wall"] - sum(r["light"]["task_secs"])     # 起動・データ・保存の固定費
            per_run = 8 * t_full + 42 * t_light + 3 * (t_fresh + 5.0) + fixed
            r.update(t_full=t_full, t_light=t_light, t_fresh=t_fresh, fixed=fixed, per_run_sec=per_run,
                     sec_per_update=t_fresh / n)
            res[f"{box}_{arm}"] = r
            print(box, arm, json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()
                                        if k not in ("full", "light", "fresh")}), flush=True)
    # 腕ごとの費用: 測った腕はその値、測っていない腕は同じ形の腕（C4・N4 は C3）で代える
    total = 0.0
    per_box = {}
    for box in a.boxes:
        sb = 0.0
        for arm in ARMS:
            key = f"{box}_{arm}" if f"{box}_{arm}" in res else f"{box}_C3"
            sb += 10 * res[key]["per_run_sec"]
        per_box[box] = sb
        total += sb
    summary = dict(host=__import__("socket").gethostname(), boxes=a.boxes, nm=a.nm, alpha=a.alpha, nw=a.nw, per_box_hours={k: v / 3600 for k, v in per_box.items()},
                   total_hours=total / 3600, rule="10 時間以内なら本走へ、超えたら報告（spec §9）",
                   over_10h=total / 3600 > 10, measured=res)
    (out / "cost.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in summary.items() if k != "measured"}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
