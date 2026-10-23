#!/usr/bin/env python3
"""chargpt_seed10_1022 C の本走の駆動（spec §5・追補 1、手順書 §3.3・§5）。
seed × 腕 を 1 単位とし、出力に provenance.json（complete=true）がある単位は飛ばす（同じコマンドで再開できる）。
各単位の開始・終了・秒・終了コードを results/chargpt_seed10_1022/revival/<部分>/drive_log.jsonl に追記し、秒は単位の provenance.json にも入る。
  CHARGPT_SRC=$CG $PY revival10_drive.py --parts C1 C2 --jobs 2          # 本走（C1・C2）
  CHARGPT_SRC=$CG $PY revival10_drive.py --parts C3 --jobs 2             # C3（記録・任意）
  ... --dry_run で単位の一覧と課題数だけを出す。--seeds・--arms で絞れる。
"""
import argparse, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import revival10_plan as P  # noqa: E402

S10 = HERE.parents[1]
RES = S10 / "results/chargpt_seed10_1022/revival"
CG = P.CG
PY = os.environ.get("PY", str(CG / ".venv/bin/python"))


def units(parts, seeds=None, arms=None):
    out = []
    for part in parts:
        for seed, fr, to, march, arm_list in P.jobs(part):
            if seeds is not None and seed not in seeds:
                continue
            for arm in arm_list:
                if arms is not None and arm not in arms:
                    continue
                out.append(dict(part=part, seed=seed, arm=arm, from_task=fr, to_task=to, model_arm=march,
                                src=str(P.M2 / P.SRC_ARM[part] / f"seed{seed}"),
                                out=str(RES / part / f"seed{seed}" / arm),
                                raw_out=str(RES / "raw" / part / f"seed{seed}" / arm)))
    return out


def done(u):
    pv = Path(u["out"]) / "provenance.json"
    if not pv.exists():
        return False
    try:
        return bool(json.loads(pv.read_text()).get("complete"))
    except Exception:
        return False


def run_unit(u):
    out = Path(u["out"]); out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CHARGPT_NLAYER="2", CHARGPT_SRC=str(CG), OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "2"))
    cmd = [PY, str(HERE / "revival10.py"), "--arm", u["arm"], "--from_task", str(u["from_task"]), "--to_task", str(u["to_task"]),
           "--seed", str(u["seed"]), "--src", u["src"], "--model_arm", u["model_arm"], "--out", u["out"], "--raw_out", u["raw_out"]]
    t0 = time.time(); start = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    with open(out / "run.log", "w") as lf:
        rc = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
    rec = dict(part=u["part"], seed=u["seed"], arm=u["arm"], from_task=u["from_task"], to_task=u["to_task"], start=start,
               end=time.strftime("%Y-%m-%dT%H:%M:%S%z"), seconds=time.time() - t0, returncode=rc, complete=done(u))
    log = RES / u["part"] / "drive_log.jsonl"
    with open(log, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[{u['part']} seed{u['seed']} {u['arm']}] rc={rc} {rec['seconds']:.0f}s complete={rec['complete']}", flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", required=True, choices=["C1", "C2", "C3"])
    ap.add_argument("--jobs", type=int, default=2, help="GPU に同時に載せる本数（手順書 §5 は 2 まで）")
    ap.add_argument("--seeds", type=int, nargs="*", default=None); ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()
    us = units(a.parts, set(a.seeds) if a.seeds else None, set(a.arms) if a.arms else None)
    todo = [u for u in us if not done(u)]
    ntask = sum(u["to_task"] - u["from_task"] for u in todo)
    print(f"単位 {len(us)}・完走済み {len(us) - len(todo)}・残り {len(todo)}（課題 {ntask}）・同時 {a.jobs}", flush=True)
    if a.dry_run:
        for u in todo:
            print(u["part"], u["seed"], u["arm"], u["from_task"], "->", u["to_task"])
        return
    # 長い単位から先に入れる（同時に回す本数が詰まらないように）
    todo.sort(key=lambda u: -(u["to_task"] - u["from_task"]))
    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        recs = list(ex.map(run_unit, todo))
    bad = [r for r in recs if not r["complete"]]
    print(f"終わり: {len(recs)} 単位・未完 {len(bad)}", flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
