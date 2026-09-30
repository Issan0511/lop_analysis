#!/usr/bin/env python3
"""sink_roots_r11b_0930 check C1 (spec_sink_roots_r11b_0930.md §3): the R11b engine's step is cifar5p1_mlp_0920's.
Two tasks of one arm and seed: cifar5p1_mlp_0920.run(seeds=[seed], CPU, eager, no fresh control, 1 thread) against
src/sink_roots_c5p1_0930.py; the per-task online_acc must agree (|diff| < 1e-9: one flipped prediction moves it by
1/(780*32) = 4e-5).  Mutation control: the engine with lr 1.0001e-4 must disagree."""
import csv, json, os, subprocess, sys, tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
torch.set_num_threads(1)
from src import cifar5p1_mlp_0920 as C       # noqa: E402


def engine(arm, seed, tasks, lr, out):
    subprocess.run([sys.executable, str(ROOT / "src" / "sink_roots_c5p1_0930.py"), "--arm", arm, "--seed", str(seed),
                    "--tasks", str(tasks), "--lr", repr(lr), "--out", str(out)], check=True, cwd=ROOT,
                   env=dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"))
    return {r["task"]: r["online_acc"] for r in json.loads((out / "rows.json").read_text()) if "online_acc" in r}


def main():
    res = {}
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for arm in ("ELU", "KKT1"):
            C.run(arm, [0], "std", 2, torch.device("cpu"), td / f"ref_{arm}", fresh=False, graph=False)
            ref = {int(r["task"]): float(r["online_acc"]) for r in csv.DictReader(open(td / f"ref_{arm}" / "per_task.csv"))
                   if r.get("online_acc") not in (None, "")}
            mine = engine(arm, 0, 2, C.LR, td / f"eng_{arm}")
            diff = {t: abs(mine[t] - ref[t]) for t in ref}
            res[arm] = {"ref": ref, "engine": mine, "max_abs_diff": max(diff.values()), "pass": max(diff.values()) < 1e-9}
        mut = engine("ELU", 0, 2, 1.0001e-4, td / "eng_mut")
        res["mutation_lr"] = {"engine": mut, "max_abs_diff_vs_ref": max(abs(mut[t] - res["ELU"]["ref"][t]) for t in mut)}
        res["mutation_lr"]["detected"] = res["mutation_lr"]["max_abs_diff_vs_ref"] >= 1e-9
    print(json.dumps(res, indent=1))
    out = ROOT / "results" / "sink_roots_r11b_0930"
    out.mkdir(parents=True, exist_ok=True)
    (out / "C1_step_check.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
