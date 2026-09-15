"""Independent raw-log aggregation audit, written before production results.

Uses no experiment analysis functions. Outputs estimates for comparison with the
registered report; estimates use a fresh, common bootstrap index matrix for each
paired seed set, so a report using a sequential RNG may differ by Monte Carlo error.
"""
import argparse
import json
from pathlib import Path

import numpy as np

ARMS = ["LRoff0_1216", "LRwf21_1216", "FB21LRoff0_1216",
        "FB21LRwf21_1216", "LRwi21_1216", "FB21LRwi21_1216"]


def aggregate(step, values, first, last):
    mask = (step % 10000 == 0) & (step >= first * 10000) & (step <= last * 10000)
    expected = np.arange(first, last + 1) * 10000
    if not np.array_equal(step[mask], expected):
        raise ValueError(f"Missing or repeated task endpoints: {first}..{last}")
    return float(np.median(values[mask].astype(np.float64), axis=1).mean())


def bootstrap(values):
    values = np.asarray(values, dtype=np.float64)
    if not values.size or not np.isfinite(values).all():
        raise ValueError("Nonfinite or empty seed estimates")
    indices = np.random.default_rng(20260915).integers(0, len(values), (2000, len(values)))
    return {"mean": float(values.mean()),
            "ci": np.percentile(values[indices].mean(axis=1), [2.5, 97.5]).tolist(),
            "n": len(values), "seed_values": values.tolist()}


def audit(logdir):
    rows, arm_stats = {}, {}
    for arm in ARMS:
        rows[arm] = {}
        excluded = []
        for seed in range(10):
            path = logdir / f"{arm}_seed{seed}.npz"
            if not path.exists():
                raise ValueError(f"Missing raw log: {path}; divergence must be audited explicitly")
            with np.load(path, allow_pickle=False) as log:
                step, zbar, zmax = log["step"], log["layer1_zbar"], log["layer1_zmax"]
                if not np.isfinite(zbar).all() or not np.isfinite(zmax).all():
                    excluded.append(seed)
                    continue
                early = aggregate(step, zbar, 251, 300)
                late = aggregate(step, zbar, 451, 500)
                wf, wfstep = log["layer1_w_free"].astype(np.float64), log["layer1_w_free_step"]
                norm = np.linalg.norm(wf, axis=-1)
                base = norm[wfstep == 200000]
                if base.shape != (1, 100) or np.any(base <= 0):
                    raise ValueError(f"Invalid t20 norms: {arm} seed {seed}")
                tail = (wfstep >= 4510000) & (wfstep <= 5000000) & (wfstep % 10000 == 0)
                post = (wfstep >= 210000) & (wfstep <= 5000000) & (wfstep % 10000 == 0)
                if post.sum() != 480 or tail.sum() != 50:
                    raise ValueError("Incomplete width trajectory")
                rows[arm][seed] = {
                    "early_zbar": early, "late_zbar": late, "delta_zbar": late - early,
                    "late_zmax": aggregate(step, zmax, 451, 500),
                    "growth_ratio": float(np.median(norm[tail], axis=1).mean() / np.median(base)),
                    "max_norm_relative_error": float(np.max(np.abs(norm[post] / base - 1))),
                }
        arm_stats[arm] = {"excluded_seeds": excluded,
                          "delta_zbar": bootstrap([v["delta_zbar"] for v in rows[arm].values()]),
                          "late_zmax": bootstrap([v["late_zmax"] for v in rows[arm].values()])}
    a, b = "LRwf21_1216", "FB21LRwf21_1216"
    paired = sorted(rows[a].keys() & rows[b].keys())
    q1 = bootstrap([rows[a][s]["late_zmax"] - rows[b][s]["late_zmax"] for s in paired])
    q1["seeds"] = paired
    return {"source": "Independent audit of raw NPZ arrays", "per_seed": rows,
            "arms": arm_stats, "q1_paired_delta_zmax": q1,
            "bootstrap_note": "Fresh common index matrix per estimate; seed mean statistic"}


def self_test():
    # Distinguishes unit-first aggregation from time-first aggregation.
    values = np.array([[0, 0, 100], [0, 100, 0], [100, 0, 0]])
    assert aggregate(np.array([10000, 20000, 30000]), values, 1, 3) == 0
    assert np.median(values.mean(axis=0)) != 0
    assert bootstrap(np.zeros(10))["ci"] == [0, 0]
    assert bootstrap(np.ones(10))["mean"] == 1
    try:
        aggregate(np.array([10000, 30000]), values[:2], 1, 3)
    except ValueError:
        pass
    else:
        raise AssertionError("Missing endpoint was accepted")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    self_test()
    if args.logs:
        result = audit(args.logs)
        content = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.output:
            args.output.write_text(content)
        else:
            print(content)
    else:
        print("Independent audit self-tests passed")
