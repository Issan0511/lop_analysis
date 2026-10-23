"""Seven simultaneous paired intervals; no hypothesis-dependent exclusions."""
from __future__ import annotations

import math
import json
import numpy as np
from analysis.chb_decompose_200_1018 import common as C


def student9_quantile(p):
    """Invert the df=9 CDF using its exact integral of cos(theta)^8.

    x=3*tan(theta) reduces the t density to a normalized cos^8.
    Bisection avoids any optional statistics-package dependency.
    """
    if not .5 < p < 1:
        raise ValueError("this registered inverse is for upper df9 quantiles")
    def primitive(theta):
        return (35*theta/128 + 7*math.sin(2*theta)/32 +
                7*math.sin(4*theta)/128 + math.sin(6*theta)/96 + math.sin(8*theta)/1024)
    normalizer = primitive(math.pi/2)
    lo, hi = 0., 100.
    for _ in range(80):
        mid = (lo+hi)/2
        cdf = .5 + .5*primitive(math.atan(mid/3))/normalizer
        if cdf < p: lo = mid
        else: hi = mid
    return (lo+hi)/2


def interval(d):
    d = np.asarray(d, dtype=np.float64)
    if d.shape != (10,) or not np.isfinite(d).all():
        raise ValueError("exactly ten finite paired differences required")
    constant = bool(np.all(d == d[0]))
    mean = float(d[0]) if constant else float(d.mean())
    sd = 0. if constant else float(d.std(ddof=1))
    half = student9_quantile(1 - .05 / (2 * 7)) * sd / math.sqrt(10)
    half95 = student9_quantile(.975) * sd / math.sqrt(10)
    return dict(mean=mean, median=float(np.median(d)), sd=sd,
                ci=[mean-half, mean+half], ci95=[mean-half95, mean+half95],
                differences=d.tolist(), positive=int((d > 0).sum()),
                negative=int((d < 0).sum()), zero=int((d == 0).sum()),
                flags=["DEGENERATE_SD"] if sd == 0 else [])


def pos(x):
    return x["ci"][0] > 0


def ni(x):
    return x["ci"][0] > -.005


def eq(x):
    return x["ci"][0] >= -.005 and x["ci"][1] <= .005


def classify(values):
    if set(values) != set(C.ARMS):
        raise ValueError("all five arms required")
    v = {a: np.asarray(values[a], dtype=np.float64) for a in C.ARMS}
    if any(x.shape != (10,) or not np.isfinite(x).all() or (x < 0).any() or (x > 1).any()
           for x in v.values()):
        raise ValueError("invalid per-seed means")
    intervals = {f"{a}-{b}": interval(v[a]-v[b])
                 for a in C.ARMS[2:] for b in ("CH", "CHB")}
    intervals["CHB-CH"] = interval(v["CHB"]-v["CH"])
    output = dict(status="COMPLETE", intervals=intervals, m=7, n=10, df=9,
                  epsilon_eq=.005, seeds=C.SEEDS,
                  per_arm={a: dict(mean=float(x.mean()), median=float(np.median(x)),
                                   per_seed=x.tolist()) for a, x in v.items()})
    if not pos(intervals["CHB-CH"]):
        return dict(output, label="NOT_REPRODUCED", candidates=[])
    candidates = []
    if (all(pos(intervals[f"{a}-CH"]) and ni(intervals[f"{a}-CHB"])
            for a in ("CH_bfreeze", "CH_bWD")) and eq(intervals["CH_wWD-CH"])):
        candidates.append("BIAS_STOP_SUFFICES")
    if pos(intervals["CH_wWD-CH"]) and ni(intervals["CH_wWD-CHB"]):
        candidates.append("WD_EXPLAINS")
    return dict(output, label=candidates[0] if len(candidates) == 1 else "MIXED",
                candidates=candidates, reason=None if len(candidates) == 1 else
                "OVERLAPPING_EVIDENCE" if candidates else "NO_UNIQUE_PATTERN")


def aggregate(data):
    indexed = {arm: C.validate_rows(data[arm], arm) for arm in C.ARMS}
    values = {a: [float(np.mean([float(indexed[a][s, t]["online_acc"])
                                for t in range(181, 201)])) for s in C.SEEDS] for a in C.ARMS}
    result = classify(values)
    result["early"] = {a: [float(np.mean([float(indexed[a][s, t]["online_acc"])
                                         for t in range(31, 51)])) for s in C.SEEDS] for a in C.ARMS}
    return result


def report(root=C.OUT):
    from analysis.chb_decompose_200_1018.launch import validate_checkpoint
    import torch
    manifest = json.loads((root / "launch_manifest.json").read_text())
    if manifest["config"] != C.config() or manifest["source_sha256"] != C.sources():
        raise ValueError("CHECK_FAILED: report source/configuration changed")
    cpu_checks = json.loads((root / "checks.json").read_text())
    cuda_checks = json.loads((root / "checks_cuda.json").read_text())
    if (not cpu_checks.get("cpu_all_pass") or cpu_checks.get("source_sha256") != C.sources() or
        C.sha(root / "checks.json") != manifest["checks_sha256"] or
        not cuda_checks.get("all_pass") or cuda_checks.get("source_sha256") != C.sources()):
        raise ValueError("CHECK_FAILED: missing/stale CPU or CUDA checks")
    data = {a: C.read_rows(root / a / "per_task.csv") for a in C.ARMS}
    result = aggregate(data)
    diagnostics = {}
    file_hashes = {}
    stream_reference = None
    initial_reference = None
    for arm in C.ARMS:
        out = root / arm
        cp = torch.load(out / "ckpt.pt", map_location="cpu", weights_only=False)
        if validate_checkpoint(cp, arm, manifest["registration"]["commit"]) != 200:
            raise ValueError("INCOMPLETE: arm did not finish")
        checkpoint_rows = C.validate_rows(cp["rows"], arm)
        for row in data[arm]:
            key = (int(row["seed"]), int(row["task"]))
            for metric in ("online_acc", "memo_acc"):
                a,b = float(row[metric]), float(checkpoint_rows[key][metric])
                if abs(a-b) > 6e-10*max(1,abs(b)):
                    raise ValueError("CHECK_FAILED: CSV/checkpoint performance disagreement")
        streams = json.loads((out / "streams.json").read_text())
        if len(streams) != 200 or [r["task"] for r in streams] != list(range(1,201)):
            raise ValueError("INCOMPLETE: stream provenance")
        if stream_reference is None: stream_reference = streams
        if streams != stream_reference:
            raise ValueError("CHECK_FAILED: unpaired labels/batch streams")
        import hashlib
        initial_digest = hashlib.sha256()
        for seed in C.SEEDS:
            with np.load(out / "snap" / f"{arm}_raw_seed{seed}" / "t00.npz") as d:
                for k in ("W1","b1","W2","b2","W3","b3","m1","m2"):
                    initial_digest.update(k.encode()); initial_digest.update(str(d[k].dtype).encode())
                    initial_digest.update(str(d[k].shape).encode()); initial_digest.update(d[k].tobytes())
        # Initialization, including CHB's nonzero b1, is paired before any update.
        initial_hash = initial_digest.hexdigest()
        if initial_reference is None: initial_reference = initial_hash
        if initial_hash != initial_reference:
            raise ValueError("CHECK_FAILED: initial parameters/EMA are not paired")
        dr = C.read_rows(out / "diagnostics.csv")
        keys = [(int(r["seed"]),int(r["task"])) for r in dr]
        if len(keys) != 2000 or set(keys) != {(s,t) for s in C.SEEDS for t in range(1,201)}:
            raise ValueError("INCOMPLETE: unit diagnostic rows")
        for t in range(1,201):
            p = out / "diagnostic_units" / f"t{t:03d}.npz"
            with np.load(p) as u:
                for layer in (1,2):
                    for k in ("b","m","s","derivative_mean","derivative_var","w_norm"):
                        if u[f"l{layer}_{k}"].shape != (10,100) or not np.isfinite(u[f"l{layer}_{k}"]).all():
                            raise ValueError("DIVERGED: invalid unit diagnostics")
                if int(u["adam_update_count"]) != 30000:
                    raise ValueError("CHECK_FAILED: update diagnostic count")
            file_hashes[str(p.relative_to(root))] = C.sha(p)
        diagnostics[arm] = {}
        metrics = [k for k in dr[0] if k not in ("arm","seed","task")]
        for seed in C.SEEDS:
            diagnostics[arm][str(seed)] = {}
            for metric in metrics:
                windows = []
                for lo,hi in ((31,50),(181,200)):
                    raw = [r[metric] for r in dr if int(r["seed"]) == seed and lo <= int(r["task"]) <= hi]
                    vals = [float(v) for v in raw if v != ""]
                    if not all(math.isfinite(v) for v in vals):
                        raise ValueError("DIVERGED: nonfinite aggregate diagnostics")
                    windows.append(dict(mean=float(np.mean(vals)) if vals else None, finite_tasks=len(vals), total_tasks=20))
                diagnostics[arm][str(seed)][metric] = dict(early=windows[0], late=windows[1],
                    difference=windows[1]["mean"]-windows[0]["mean"] if all(w["mean"] is not None for w in windows) else None)
    C.put(root / "diagnostic_windows.json", diagnostics)
    C.put(root / "diagnostic_manifest.json", dict(files=file_hashes, initialization_sha256=initial_reference,
            s_correction=1, derivative_variance_correction=0, diagnostic_reductions="float64 on native float32 forward outputs",
            zero_width_policy="raw r with signed infinity/NaN retained in NPZ; explicit zero/nonfinite counts in CSV"))
    C.put(root / "verdict.json", result)
    lines = [f"# {C.RUN}", "", f"Status: COMPLETE. Registered label: {result['label']}.", "",
             "Mean online accuracy over t181–200; n=10 paired seeds; seven Bonferroni t intervals.", "",
             "| Arm | Mean | Median |", "|---|---:|---:|"]
    for arm, value in result["per_arm"].items():
        lines.append(f"| {arm} | {value['mean']:.6f} | {value['median']:.6f} |")
    lines += ["", "Bias sufficiency is limited to these preventive interventions; a weight-WD benefit does not identify learning rate as its only mechanism.", ""]
    (root / "summary.md").write_text("\n".join(lines))
    C.write_rows(root / "per_task.csv", [r for a in C.ARMS for r in data[a]])
    return result
