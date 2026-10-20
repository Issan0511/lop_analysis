#!/usr/bin/env python3
"""S-repro1 on the production runs (spec section 6): A1 and R1, seeds 0 and 1, tasks 1-30 of the 50-task runs
(CUDA hidden) against the saved gen_states runs, exact."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "raw" / "depth_silu_rlmnist_1020" / "runs"
STATES = Path("/home/issan/Projects/obsidian-research-data/push_direction_proof_0930/states")
OUT = ROOT / "results" / "depth_silu_rlmnist_1020" / "checks"
allok = True
for act in ("SILU", "R"):
    for s in (0, 1):
        d = np.load(RUNS / f"{act}_L1_none_s{s}.npz"); g = np.load(STATES / f"{act}_K10_s{s}.npz")
        diffs = {}
        for j in range(4):
            for key in ("end", "mid"):
                diffs[f"{key}{j}"] = float(np.abs(d[f"{key}{j}"][:30].astype(np.float64) - g[f"{key}{j}"].astype(np.float64)).max())
            diffs[f"init{j}"] = float(np.abs(d[f"init{j}"].astype(np.float64) - g[f"init{j}"].astype(np.float64)).max())
        diffs["y"] = float(np.abs(d["y"][:30] - g["y"]).max()); diffs["acc"] = float(np.abs(d["acc"][:30] - g["acc"]).max())
        ok = all(v == 0.0 for v in diffs.values())
        allok &= ok
        rec = {"check": f"S-repro1-main_{act}_s{s}", "pass": ok, "max_abs_diff": diffs, "tasks_compared": 30,
               "note": "production run (CUDA_VISIBLE_DEVICES empty, 50 tasks) vs saved gen_states run"}
        (OUT / f"S-repro1-main_{act}_s{s}.json").write_text(json.dumps(rec, indent=1))
        print(rec["check"], "PASS" if ok else "FAIL", max(diffs.values()))
print("ALL PASS" if allok else "SOME FAIL")
