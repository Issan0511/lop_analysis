#!/usr/bin/env python3
"""sink_roots_r11b_0930 check C1 (spec_sink_roots_r11b_0930.md §3): the R11b engine's step is cifar5p1_mlp_0920's.
Two tasks of one arm and seed: cifar5p1_mlp_0920.run(seeds=[seed], CPU, eager, no fresh control, 1 thread) against
src/sink_roots_c5p1_0930.py.  Compared per task: online_acc (|diff| < 1e-9), and the continuous task-end statistics of
0920's evaluate() on the task's own training images -- zbar (torch's lower median over units of the per-unit mean z) and
zbar_min, both layers -- recomputed from the engine's u_m1 / u_m2 at update 780 (relative 1e-8: 0920 prints 10 digits).
Mutation control: the engine with lr 1.0001e-4 must disagree.  online_acc alone does not see that change over 2 tasks
(found on the first try), so the control is judged on the continuous statistics."""
import csv, json, os, subprocess, sys, tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
torch.set_num_threads(1)
from src import cifar5p1_mlp_0920 as C       # noqa: E402

KEYS = ("zbar_l1", "zbar_l2", "zbar_min_l1", "zbar_min_l2")


def engine(arm, seed, tasks, lr, out):
    subprocess.run([sys.executable, str(ROOT / "src" / "sink_roots_c5p1_0930.py"), "--arm", arm, "--seed", str(seed),
                    "--tasks", str(tasks), "--lr", repr(lr), "--out", str(out)], check=True, cwd=ROOT,
                   env=dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"))
    rows = {r["task"]: {"online_acc": r["online_acc"]} for r in json.loads((out / "rows.json").read_text()) if "online_acc" in r}
    a = np.load(out / "arrays.npz", allow_pickle=True)
    for i, t in enumerate(a["tasks_m1"]):
        for L in ("1", "2"):
            mu = np.sort(a["u_m" + L][i, -1].astype(np.float32))           # per-unit mean z at update 780
            rows[int(t)][f"zbar_l{L}"] = float(mu[(len(mu) - 1) // 2])   # torch.median: the lower middle value
            rows[int(t)][f"zbar_min_l{L}"] = float(mu[0])
    return rows


def compare(mine, ref):
    worst = {"online_acc": 0.0, **{k: 0.0 for k in KEYS}}
    for t in ref:
        worst["online_acc"] = max(worst["online_acc"], abs(mine[t]["online_acc"] - ref[t]["online_acc"]))
        for k in KEYS:
            worst[k] = max(worst[k], abs(mine[t][k] / ref[t][k] - 1))
    return worst


def main():
    res = {}
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for arm in ("ELU", "KKT1"):
            C.run(arm, [0], "std", 2, torch.device("cpu"), td / f"ref_{arm}", fresh=False, graph=False)
            ref = {int(r["task"]): {k: float(r[k]) for k in ("online_acc",) + KEYS}
                   for r in csv.DictReader(open(td / f"ref_{arm}" / "per_task.csv")) if r.get("online_acc") not in (None, "")}
            mine = engine(arm, 0, 2, C.LR, td / f"eng_{arm}")
            w = compare(mine, ref)
            res[arm] = {"ref": ref, "engine": mine, "worst": w,
                        "pass": w["online_acc"] < 1e-9 and max(w[k] for k in KEYS) < 1e-8}
        mut = engine("ELU", 0, 2, 1.0001e-4, td / "eng_mut")
        w = compare(mut, res["ELU"]["ref"])
        res["mutation_lr"] = {"worst": w, "detected": max(w[k] for k in KEYS) >= 1e-8}
    print(json.dumps(res, indent=1))
    out = ROOT / "results" / "sink_roots_r11b_0930"
    out.mkdir(parents=True, exist_ok=True)
    (out / "C1_step_check.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
