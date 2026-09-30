#!/usr/bin/env python3
"""R7 on the GPU (spec_sink_roots_0930_R7_gpu.md): K1 over all 50 tasks.  Arm A must reproduce the S5 ref arm
(cap_cifar_ee_0920, GPU, the same box): online_acc and memo_acc of every (seed, task) as printed in the two CSVs."""
import csv, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
A = Path("/home/issan/Projects/obsidian-research-data/sink_roots_0930/r7gpu/A/per_task.csv")
REF = ROOT / "results" / "cap_cifar_ee_0920" / "per_task.csv"


def main():
    ref = {(int(r["seed"]), int(r["task"])): r for r in csv.DictReader(open(REF)) if r["arm"] == "ref"}
    a = {(int(r["seed"]), int(r["task"])): r for r in csv.DictReader(open(A))}
    worst = {"online_acc": 0.0, "memo_acc": 0.0}
    missing = sorted(set(ref) - set(a))
    for k in sorted(set(ref) & set(a)):
        for c in worst:
            worst[c] = max(worst[c], abs(float(a[k][c]) - float(ref[k][c])))
    out = {"pairs": len(set(ref) & set(a)), "missing_in_A": len(missing), "max_abs_diff": worst,
           "K1_pass": not missing and max(worst.values()) < 1e-9}
    print(json.dumps(out, indent=1))
    (ROOT / "results" / "sink_roots_0930" / "R7gpu_K1_full.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
