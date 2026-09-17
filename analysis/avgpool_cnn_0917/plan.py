#!/usr/bin/env python3
"""Write the job plan for launch.py (spec §3, §7): SNA_avg and SN3_avg, interleaved by seed.
rss_gb 2.2 is the CNN box's measured peak (swish_battle_0917); the live copy under
_launch/ is re-read by the launcher every cycle."""
import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ARMS = ("SNA_avg", "SN3_avg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-max", type=int, default=5)
    ap.add_argument("--reserve-gb", type=float, default=6.0)
    ap.add_argument("--min-gpu-mb", type=int, default=500)
    ap.add_argument("--out", default=str(REPO / "results/avgpool_cnn_0917/_launch/plan.json"))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    jobs = [dict(arm=x, seed=s, device="cuda", threads=1, rss_gb=2.2)
            for s in range(10) for x in ARMS]
    plan = {"gpu_max": a.gpu_max, "reserve_gb": a.reserve_gb, "min_gpu_mb": a.min_gpu_mb,
            "jobs": jobs}
    if Path(a.out).exists() and not a.force:
        raise SystemExit(f"{a.out} exists (the live plan); --force to overwrite")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(plan, indent=1))
    print(f"{len(jobs)} jobs -> {a.out}")


if __name__ == "__main__":
    main()
