#!/usr/bin/env python3
"""Independently recompute the four handoff quantities, without importing verdict code."""
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/neffdir_ee_0918"


def integrate(f, a, b, tol=1e-13):
    """Adaptive Simpson integration, independently of the registered t implementation."""
    def rec(a, b, fa, fm, fb, whole, eps, depth):
        m = (a + b) / 2
        fl, fr = f((a + m) / 2), f((m + b) / 2)
        left = (m - a) * (fa + 4 * fl + fm) / 6
        right = (b - m) * (fm + 4 * fr + fb) / 6
        delta = left + right - whole
        if depth == 0 or abs(delta) <= 15 * eps:
            return left + right + delta / 15
        return rec(a, m, fa, fl, fm, left, eps / 2, depth - 1) + rec(m, b, fm, fr, fb, right, eps / 2, depth - 1)
    fa, fm, fb = f(a), f((a + b) / 2), f(b)
    return rec(a, b, fa, fm, fb, (b - a) * (fa + 4 * fm + fb) / 6, tol, 24)


def tcritical(level, df):
    c = math.gamma((df + 1) / 2) / (math.sqrt(df * math.pi) * math.gamma(df / 2))
    density = lambda x: c * (1 + x * x / df) ** (-(df + 1) / 2)
    lo, hi = 0., 16.
    for _ in range(52):
        mid = (lo + hi) / 2
        if integrate(density, 0., mid) < level / 2:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def main():
    paired = {}
    with (OUT / "paired.csv").open() as f:
        for r in csv.DictReader(f):
            paired[(r["quantity"], int(r["k"]), int(r["seed"]))] = float(r["value"])
    with (OUT / "verdict.csv").open() as f:
        verdict = {(r["quantity"], int(r["k"]), float(r["level"])): r for r in csv.DictReader(f)}
    values = {q: [] for q in ("dR_add_h", "phi_add_h", "dR_drop", "comp")}
    hashes, per_seed = {}, []
    for seed in range(30, 40):
        p = OUT / f"runs/s{seed}/arms.csv"
        hashes[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
        with p.open() as f:
            rows = {r["arm"]: r for r in csv.DictReader(f) if int(r["k"]) == 1}
        e = lambda a: float(rows[a]["online_acc"])
        n = lambda a: float(rows[a]["nbar_neffS2"])
        raw = lambda a: float(rows[a]["nbar_neffT2"])
        floor = e("S2u30r")
        base, added = e("S2dyn_c12") - floor, e("S2dyn_c12_a02") - floor
        v = {"dR_add_h": e("S2dyn_c12_a02") - e("S2dyn_c12"),
             "phi_add_h": n("S2dyn_c12_a02") / n("S2dyn_c12") * base - added,
             "dR_drop": e("S2dyn_10r") - e("S2dyn_m50"),
             "comp": raw("S2dyn_m50") - raw("S2dyn_10r")}
        for q, x in v.items():
            assert math.isfinite(x)
            assert abs(x - paired[(q, 1, seed)]) < 1e-12, (seed, q)
            values[q].append(x)
        per_seed.append({"seed": seed, **v})
    crit = {level: tcritical(level, 9) for level in (.95, .975)}
    comparisons = []
    for q, xs in values.items():
        avg, sd = statistics.mean(xs), statistics.stdev(xs)
        for level, tc in crit.items():
            half = tc * sd / math.sqrt(10)
            calc = {"mean": avg, "sd": sd, "lo": avg - half, "hi": avg + half}
            expected = verdict[(q, 1, level)]
            errors = {k: abs(x - float(expected[k])) for k, x in calc.items()}
            assert max(errors.values()) < 1e-9, (q, level, errors)
            comparisons.append({"quantity": q, "level": level, **calc, "max_abs_error": max(errors.values())})
    result = {"all_pass": True, "method": "arms.csv only; stdlib paired statistics and adaptive Simpson t-density integration; no verdict imports",
              "t_critical_df9": crit, "source_sha256": hashes, "per_seed": per_seed, "comparisons": comparisons}
    (OUT / "independent_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"all_pass": True, "comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()
