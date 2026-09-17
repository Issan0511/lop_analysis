#!/usr/bin/env python3
"""Write results/rlcifar_mlp_battle_0918/_launch/plan.json (spec §7 order and caps; RSS from S-cost)."""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.rlcifar_mlp_battle_0918 import ARM_ORDER

rss = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
own_max = int(sys.argv[2]) if len(sys.argv) > 2 else 6
plan = {"gpu_max": 11, "own_max": own_max, "reserve_gb": 2.5, "min_gpu_mb": 500, "retries": 2,
        "jobs": [{"arm": a, "rss_gb": rss, "threads": 2} for a in ARM_ORDER]}
out = REPO / "results" / "rlcifar_mlp_battle_0918" / "_launch"
out.mkdir(parents=True, exist_ok=True)
(out / "plan.json").write_text(json.dumps(plan, indent=1))
print(json.dumps({k: v for k, v in plan.items() if k != "jobs"}), [j["arm"] for j in plan["jobs"]])
