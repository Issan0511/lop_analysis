"""T0: separate the zero of expected force from the zero of displacement.

Frozen specification: obsidian-research commit 5b9805a,
``可塑性喪失/spec/天井プログラムT0_spec_0827.md``.

The logical row population contains 35,640,000 registered main pairs.  The
implementation never materialises that table: it constructs the exact masks
per seed and k, then stores the sufficient statistics at seed x band x k.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


SPEC_VAULT_COMMIT = "5b9805a"
KS = (1, 2, 5, 10, 20, 40)
POPS = ("A", "S", "L", "U")
PERIOD = 10_000
WIDTH = 100
BOOT_N = 10_000
BOOT_SEED = 20260828
MIN_MAIN_SEED = 8
MIN_DESC_SEED = 6
MIN_PAIRS = 300
F32_RTOL = 8 * np.finfo(np.float32).eps
F32_ATOL = 8 * np.finfo(np.float32).tiny
F64_RTOL = 64 * np.finfo(np.float64).eps
F64_ATOL = 64 * np.finfo(np.float64).tiny
REQUIRED = {
    "step", "seed", "period", "width", "cos_u_mu", "p_hat", "w_norm",
    "b", "v", "F_self", "F_rest", "F_gate", "G", "flip_state",
    "E_delta", "mu_norm", "eval_loss_exact",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_info(root: Path) -> dict:
    def run(*args: str) -> str:
        return subprocess.check_output(args, cwd=root, text=True).strip()

    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "branch": run("git", "branch", "--show-current"),
        "tracked_clean": not bool(run("git", "status", "--porcelain", "--untracked-files=no")),
        "status_porcelain": run("git", "status", "--porcelain").splitlines(),
    }


def load_one(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: np.array(z[k]) for k in z.files}


def mixed_close(lhs: np.ndarray, rhs: np.ndarray, atol: float, rtol: float,
                scale: np.ndarray) -> np.ndarray:
    return np.abs(lhs - rhs) <= atol + rtol * scale


def aggregate_mask(band: np.ndarray, mask: np.ndarray,
                   values: dict[str, np.ndarray]) -> list[dict]:
    """Return exact band counts and sums for a 2-D logical row population."""
    b = band.ravel()
    m = mask.ravel()
    b = b[m]
    if b.size == 0:
        return []
    uniq, inv = np.unique(b, return_inverse=True)
    count = np.bincount(inv)
    sums = {name: np.bincount(inv, weights=value.ravel()[m])
            for name, value in values.items()}
    return [{"band": int(j), "count": int(count[i]),
             **{f"sum_{name}": float(sums[name][i]) for name in values}}
            for i, j in enumerate(uniq)]


def append_records(dest: list[dict], seed: int, k: int, pop: str,
                   band: np.ndarray, mask: np.ndarray,
                   values: dict[str, np.ndarray]) -> None:
    for row in aggregate_mask(band, mask, values):
        n = row["count"]
        dest.append({"seed": seed, "k": k, "population": pop, **row,
                     **{f"mean_{name}": row[f"sum_{name}"] / n for name in values}})


def source_checks(d: dict[str, np.ndarray]) -> tuple[bool, str]:
    fg = d["F_gate"].astype(np.float64)
    fs = d["F_self"].astype(np.float64)
    fr = d["F_rest"].astype(np.float64)
    ok = mixed_close(fg, fs + fr, F32_ATOL, F32_RTOL,
                     np.abs(fg) + np.abs(fs) + np.abs(fr))
    return bool(ok.all()), f"max_abs={np.max(np.abs(fg-fs-fr)):.17g}, bad={int((~ok).sum())}"


def primary_indices(d: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    step = d["step"]
    fs = d["flip_state"]
    changed = np.any(np.diff(fs, axis=0) != 0, axis=1)
    boundary_idx = np.flatnonzero(changed)
    boundary_steps = step[boundary_idx]
    lookup = {int(s): i for i, s in enumerate(step)}
    starts_step = np.concatenate([np.arange(int(b)-100, int(b)-40) for b in boundary_steps])
    starts_idx = np.array([lookup[int(s)] for s in starts_step], dtype=np.int64)
    owner = np.repeat(boundary_steps, 60)
    return boundary_idx, starts_idx, owner


def seed_statistics(d: dict[str, np.ndarray]) -> tuple[list[dict], list[dict],
                                                        list[dict], list[dict], list[dict],
                                                        list[dict], dict]:
    seed = int(d["seed"])
    step = d["step"]
    period = int(d["period"])
    x = d["cos_u_mu"].astype(np.float64) * d["w_norm"].astype(np.float64)
    p = d["p_hat"].astype(np.float64)
    fg = d["F_gate"].astype(np.float64)
    fs = d["F_self"].astype(np.float64)
    fr = d["F_rest"].astype(np.float64)
    flip = d["flip_state"]
    boundary_idx, starts, owner = primary_indices(d)
    starts_step = step[starts]
    x0, p0, f0 = x[starts], p[starts], fg[starts]
    band = np.floor(x0 * 10.0).astype(np.int64)
    legacy_band = np.digitize(x0, [-0.6, -0.3, -0.1, 0.1, 0.3, 0.6]).astype(np.int64)

    curve_records: list[dict] = []
    mix_seed_records: list[dict] = []
    sensitivity_records: list[dict] = []
    legacy_records: list[dict] = []
    bulk_records: list[dict] = []
    structure: list[dict] = []
    s_one = np.nan
    previous_on_end = None
    all_absorb = True
    all_flip_equal = True
    axis_max = 0.0
    all_finite = True
    common_equal = True
    endpoint_ok = True
    population_ok = True
    monotone_ok = True

    for k in KS:
        ends_step = starts_step + k
        ends = starts + k
        if not np.array_equal(step[ends], ends_step):
            raise RuntimeError(f"non-contiguous record window seed={seed}, k={k}")
        x1, p1 = x[ends], p[ends]
        dx = x1 - x0
        disp = dx / k
        path = np.zeros_like(f0)
        for u in range(k):
            path += fg[starts + u]
        path /= k
        h = disp - f0
        on0, on1 = p0 > 0, p1 > 0
        pops = {"U": np.ones_like(on0, dtype=bool), "A": on0,
                "S": on0 & on1, "L": on0 & ~on1}
        values = {"F": f0, "D": disp, "H": h, "F_path": path}
        for pop in POPS:
            append_records(curve_records, seed, k, pop, band, pops[pop], values)
        append_records(sensitivity_records, seed, k, "A_no_near", band,
                       p0 >= 0.05, values)
        append_records(legacy_records, seed, k, "A", legacy_band, on0, values)

        if k == 1:
            s_one = float(h[on0].mean())
        if previous_on_end is not None and np.any(on1 & ~previous_on_end):
            monotone_ok = False
        previous_on_end = on1
        population_ok &= (not np.any(pops["S"] & pops["L"]) and
                          np.array_equal(pops["S"] | pops["L"], pops["A"]))
        off0 = ~on0
        all_absorb &= bool((~on1[off0]).all() and (f0[off0] == 0).all()
                           and (fs[starts][off0] == 0).all()
                           and (fr[starts][off0] == 0).all()
                           and (dx[off0] == 0).all())
        all_flip_equal &= bool((flip[starts] == flip[ends]).all())
        for ii, jj in zip(starts[::60], ends[::60]):
            old = np.concatenate([flip[ii].astype(np.float64), np.full(5, 0.5)])
            new = np.concatenate([flip[jj].astype(np.float64), np.full(5, 0.5)])
            cos = float(np.dot(old, new) / (np.linalg.norm(old)*np.linalg.norm(new)))
            axis_max = max(axis_max, abs(1-cos))
        all_finite &= bool(np.isfinite(np.stack([x0, x1, dx, f0, fs[starts],
                                                 fr[starts], p0, p1])).all())
        common_equal &= bool(np.array_equal(x0, x[starts]) and
                             np.array_equal(p0, p[starts]) and
                             np.array_equal(f0, fg[starts]))
        endpoint_ok &= bool((ends_step <= owner-1).all())
        structure.append({"seed": seed, "k": k, "n_U": int(pops["U"].sum()),
                          "n_A": int(pops["A"].sum()), "n_S": int(pops["S"].sum()),
                          "n_L": int(pops["L"].sum()),
                          "flip_cross": int(np.any(flip[starts] != flip[ends], axis=1).sum())})

    # Build seed-level mixture contributions from exact S/L sums divided by n_A.
    seed_curve = pd.DataFrame(curve_records)
    idx_cols = ["seed", "k", "band"]
    for keys, group in seed_curve.groupby(idx_cols, sort=True):
        by_pop = {r.population: r for r in group.itertuples()}
        if "A" not in by_pop:
            continue
        a = by_pop["A"]
        row = {"seed": int(keys[0]), "k": int(keys[1]), "band": int(keys[2]),
               "n_A": int(a.count), "n_S": 0, "n_L": 0}
        for name in ("F", "D", "H"):
            row[f"A_{name}"] = getattr(a, f"mean_{name}")
            for pop in ("S", "L"):
                rec = by_pop.get(pop)
                row[f"C_{pop}_{name}"] = (getattr(rec, f"sum_{name}") / a.count
                                            if rec is not None else 0.0)
        for pop in ("S", "L"):
            rec = by_pop.get(pop)
            row[f"n_{pop}"] = int(rec.count) if rec is not None else 0
        row["off_rate"] = 1.0 - row["n_S"] / row["n_A"]
        mix_seed_records.append(row)

    # Bulk k=1000 reference: same-task phase 1000..8000, initialization excluded.
    gap = np.diff(step)
    candidates = np.flatnonzero(gap == 1000)
    s = step[candidates]
    task_id = np.where(s == 0, 0, (s-1)//period)
    tau = s - task_id*period
    keep = (s > 0) & np.isin(tau, np.arange(1000, 9000, 1000))
    bs, be = candidates[keep], candidates[keep] + 1
    bx0, bx1 = x[bs], x[be]
    bband = np.floor(bx0*10).astype(np.int64)
    bdisp = (bx1-bx0)/1000.0
    bvals = {"F": fg[bs], "D": bdisp, "H": bdisp-fg[bs]}
    append_records(bulk_records, seed, 1000, "A", bband, p[bs] > 0, bvals)
    bulk_flip_equal = bool((flip[bs] == flip[be]).all())

    checks = {
        "n_step": len(step), "gap_counts": {int(v): int((np.diff(step)==v).sum())
                                                for v in np.unique(np.diff(step))},
        "n_boundary": len(boundary_idx), "boundary_mask_match": bool(np.array_equal(
            np.flatnonzero((step[:-1] % period == 0) & (np.diff(step) == 1) & (step[:-1] > 0)),
            boundary_idx)),
        "n_start": len(starts), "structure": structure, "common_equal": common_equal,
        "endpoint_ok": endpoint_ok, "all_finite": all_finite,
        "population_ok": population_ok, "monotone_ok": monotone_ok,
        "absorption_ok": all_absorb, "flip_equal": all_flip_equal,
        "axis_max_abs_1_minus_cos": axis_max, "bulk_n_pair": len(bs),
        "bulk_flip_equal": bulk_flip_equal, "s_one": s_one,
    }
    return (curve_records, mix_seed_records, sensitivity_records, legacy_records,
            bulk_records, structure, checks)


def aggregate_curves(seed_curve: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (pop, k, band), g in seed_curve.groupby(["population", "k", "band"], sort=True):
        support = tuple(sorted(map(int, g.seed.unique())))
        rows.append({
            "population": pop, "k": int(k), "band": int(band),
            "band_left": band/10.0, "band_right": (band+1)/10.0,
            "band_center": (band+0.5)/10.0,
            "n_pair": int(g["count"].sum()), "n_seed": len(support),
            "seed_ids": ",".join(map(str, support)),
            "guard_main": len(support) >= MIN_MAIN_SEED and g["count"].sum() >= MIN_PAIRS,
            "guard_descriptive": len(support) >= MIN_DESC_SEED and g["count"].sum() >= MIN_PAIRS,
            "F": float(g.mean_F.mean()), "D": float(g.mean_D.mean()),
            "H": float(g.mean_H.mean()), "F_path": float(g.mean_F_path.mean()),
        })
    return pd.DataFrame(rows)


def aggregate_simple(seed_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if seed_rows.empty:
        return seed_rows
    mean_cols = [c for c in seed_rows.columns if c.startswith("mean_")]
    for keys, g in seed_rows.groupby(["population", "k", "band"], sort=True):
        support = tuple(sorted(map(int, g.seed.unique())))
        row = {"population": keys[0], "k": int(keys[1]), "band": int(keys[2]),
               "n_pair": int(g["count"].sum()), "n_seed": len(support),
               "seed_ids": ",".join(map(str, support))}
        row.update({c.removeprefix("mean_"): float(g[c].mean()) for c in mean_cols})
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_mixture(seed_mix: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in seed_mix.columns if c.startswith("A_") or c.startswith("C_")
            or c == "off_rate"]
    rows = []
    for (k, band), g in seed_mix.groupby(["k", "band"], sort=True):
        support = tuple(sorted(map(int, g.seed.unique())))
        row = {"k": int(k), "band": int(band), "band_center": (band+0.5)/10.0,
               "n_A": int(g.n_A.sum()), "n_S": int(g.n_S.sum()),
               "n_L": int(g.n_L.sum()), "n_seed": len(support),
               "seed_ids": ",".join(map(str, support))}
        row.update({c: float(g[c].mean()) for c in cols})
        rows.append(row)
    return pd.DataFrame(rows)


def root_context(curves: pd.DataFrame, pop: str, k: int) -> dict:
    g = curves[(curves.population == pop) & (curves.k == k)].sort_values("band")
    bands = g.band.to_numpy(dtype=np.int64)
    centers = g.band_center.to_numpy(dtype=np.float64)
    valid = g.guard_main.to_numpy(dtype=bool)
    support = g.seed_ids.to_numpy(dtype=str)
    edge = ((np.diff(bands) == 1) & valid[:-1] & valid[1:]
            & (support[:-1] == support[1:]))
    return {"bands": bands, "centers": centers, "valid": valid,
            "support": support, "edge": edge}


def root_detail(ctx: dict, values: np.ndarray) -> dict:
    x = ctx["centers"]
    valid, edge = ctx["valid"], ctx["edge"]
    y = np.asarray(values, dtype=np.float64)
    down, up = [], []
    for i, ok in enumerate(edge):
        if not ok or not (np.isfinite(y[i]) and np.isfinite(y[i+1])):
            continue
        if y[i] > 0 and y[i+1] < 0:
            down.append(float(x[i] - y[i]*(x[i+1]-x[i])/(y[i+1]-y[i])))
        elif y[i] < 0 and y[i+1] > 0:
            up.append(float(x[i] - y[i]*(x[i+1]-x[i])/(y[i+1]-y[i])))

    zeros = valid & np.isfinite(y) & (y == 0)
    zero_intervals = []
    visited = np.zeros(len(y), dtype=bool)
    for i in np.flatnonzero(zeros):
        if visited[i]:
            continue
        run = [i]
        visited[i] = True
        j = i
        while j < len(y)-1 and edge[j] and zeros[j+1]:
            j += 1
            visited[j] = True
            run.append(j)
        if len(run) > 1:
            zero_intervals.append((float(x[run[0]]), float(x[run[-1]])))
        else:
            z = run[0]
            if z > 0 and z < len(y)-1 and edge[z-1] and edge[z] \
                    and np.isfinite(y[z-1]) and np.isfinite(y[z+1]):
                if y[z-1] > 0 and y[z+1] < 0:
                    down.append(float(x[z]))
                elif y[z-1] < 0 and y[z+1] > 0:
                    up.append(float(x[z]))
    down.sort()
    up.sort()
    return {"root": float(down[-1]) if down else np.nan,
            "down_roots": down, "up_roots": up,
            "zero_intervals": zero_intervals, "n_down": len(down), "n_up": len(up)}


def curve_vector(curves: pd.DataFrame, ctx: dict, pop: str, k: int, metric: str) -> np.ndarray:
    g = curves[(curves.population == pop) & (curves.k == k)].set_index("band")
    return np.array([g.at[int(b), metric] for b in ctx["bands"]], dtype=np.float64)


def seed_matrix(seed_curve: pd.DataFrame, ctx: dict, pop: str, k: int,
                metric: str) -> np.ndarray:
    col = f"mean_{metric}"
    out = np.full((10, len(ctx["bands"])), np.nan)
    sub = seed_curve[(seed_curve.population == pop) & (seed_curve.k == k)]
    pos = {int(b): i for i, b in enumerate(ctx["bands"])}
    for r in sub.itertuples():
        out[int(r.seed), pos[int(r.band)]] = float(getattr(r, col))
    return out


def bootstrap_curve(seed_values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    finite = np.isfinite(seed_values)
    numerator = weights @ np.nan_to_num(seed_values, nan=0.0)
    denominator = weights @ finite.astype(np.float64)
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan),
                     where=denominator > 0)


def root_bootstrap(seed_curve: pd.DataFrame, curves: pd.DataFrame,
                   weights: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    point_rows = []
    root_boot: dict[tuple[str, int, str], np.ndarray] = {}
    point: dict[tuple[str, int, str], float] = {}
    for pop in POPS:
        for k in KS:
            ctx = root_context(curves, pop, k)
            for metric in ("F", "D"):
                vals = curve_vector(curves, ctx, pop, k, metric)
                detail = root_detail(ctx, vals)
                point[(pop, k, metric)] = detail["root"]
                point_rows.append({
                    "population": pop, "k": k, "metric": metric,
                    "primary_down_root": detail["root"],
                    "n_down": detail["n_down"], "n_up": detail["n_up"],
                    "down_roots": json.dumps(detail["down_roots"]),
                    "up_roots": json.dumps(detail["up_roots"]),
                    "zero_intervals": json.dumps(detail["zero_intervals"]),
                })
                seed_vals = seed_matrix(seed_curve, ctx, pop, k, metric)
                boot_curves = bootstrap_curve(seed_vals, weights)
                roots = np.full(BOOT_N, np.nan)
                for b in range(BOOT_N):
                    roots[b] = root_detail(ctx, boot_curves[b])["root"]
                root_boot[(pop, k, metric)] = roots

    contrast_rows = []
    boot_rows = []
    g_boot: dict[tuple[str, int], np.ndarray] = {}
    g_point: dict[tuple[str, int], float] = {}
    for pop in POPS:
        for k in KS:
            zf, zd = point[(pop, k, "F")], point[(pop, k, "D")]
            gp = zd-zf if np.isfinite(zf) and np.isfinite(zd) else np.nan
            gb = root_boot[(pop, k, "D")] - root_boot[(pop, k, "F")]
            exist = np.isfinite(root_boot[(pop, k, "D")]) & np.isfinite(root_boot[(pop, k, "F")])
            rate = float(exist.mean())
            lo = hi = np.nan
            if np.isfinite(gp) and rate >= 0.95:
                lo, hi = np.percentile(gb[exist], [2.5, 97.5])
            g_boot[(pop, k)] = gb
            g_point[(pop, k)] = gp
            row = {"estimate": "g", "population": pop, "k": k, "value": gp,
                   "ci_lo": lo, "ci_hi": hi, "root_exist_rate": rate,
                   "n_exist": int(exist.sum())}
            contrast_rows.append(row)
            boot_rows.append(row.copy())

    for k in KS:
        needed = [root_boot[(p, k, m)] for p in ("A", "S") for m in ("F", "D")]
        exist = np.logical_and.reduce([np.isfinite(v) for v in needed])
        val = g_point[("S", k)] - g_point[("A", k)] \
            if np.isfinite(g_point[("S", k)]) and np.isfinite(g_point[("A", k)]) else np.nan
        arr = g_boot[("S", k)] - g_boot[("A", k)]
        rate = float(exist.mean())
        lo = hi = np.nan
        if np.isfinite(val) and rate >= 0.95:
            lo, hi = np.percentile(arr[exist], [2.5, 97.5])
        row = {"estimate": "C_surv", "population": "S-A", "k": k,
               "value": val, "ci_lo": lo, "ci_hi": hi,
               "root_exist_rate": rate, "n_exist": int(exist.sum())}
        contrast_rows.append(row)
        boot_rows.append(row.copy())

    needed = [root_boot[("A", k, m)] for k in (1, 40) for m in ("F", "D")]
    exist = np.logical_and.reduce([np.isfinite(v) for v in needed])
    pg = g_point[("A", 40)] - g_point[("A", 1)] \
        if np.isfinite(g_point[("A", 40)]) and np.isfinite(g_point[("A", 1)]) else np.nan
    pg_arr = g_boot[("A", 40)] - g_boot[("A", 1)]
    rate = float(exist.mean())
    lo = hi = np.nan
    if np.isfinite(pg) and rate >= 0.95:
        lo, hi = np.percentile(pg_arr[exist], [2.5, 97.5])
    row = {"estimate": "P_g", "population": "A", "k": "40-1", "value": pg,
           "ci_lo": lo, "ci_hi": hi, "root_exist_rate": rate,
           "n_exist": int(exist.sum())}
    contrast_rows.append(row)
    boot_rows.append(row.copy())
    return pd.DataFrame(point_rows), pd.DataFrame(contrast_rows), {
        "root_boot": root_boot, "g_boot": g_boot, "point": point,
        "g_point": g_point, "P_g": pg, "P_g_exist": exist,
        "P_g_boot": pg_arr, "P_g_rate": rate,
        "bootstrap_summary": pd.DataFrame(boot_rows),
    }


def sanity_rows(paths: list[Path], schemas: list[dict], checks: list[dict],
                f_checks: list[tuple[bool, str]], seed_curve: pd.DataFrame,
                seed_mix: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    def add(i: str, ok: bool, note: str) -> None:
        rows.append({"id": i, "pass": bool(ok), "note": note})

    add("S1_sources", len(paths) == 10 and all(s["n_step"] == 20901 for s in schemas),
        f"n_seed={len(paths)}, n_step={[s['n_step'] for s in schemas]}")
    add("S2_schema", all(set(s["keys"]) >= REQUIRED for s in schemas), "required fields")
    add("S3_grid", all(c["gap_counts"] == {1: 19900, 900: 199, 1000: 801}
                       for c in checks), str([c["gap_counts"] for c in checks]))
    add("S4_boundaries", all(c["n_boundary"] == 99 and c["boundary_mask_match"] for c in checks),
        str([(c["n_boundary"], c["boundary_mask_match"]) for c in checks]))
    add("S5_common_starts", all(c["n_start"] == 5940 and c["common_equal"] for c in checks),
        "5940 starts and invariant start columns")
    n_u_ok = all(all(r["n_U"] == 594000 for r in c["structure"]) for c in checks)
    add("S6_U_rows", n_u_ok, "594000 per seed x k")
    add("S7_endpoints", all(c["endpoint_ok"] for c in checks), "end <= B-1")
    add("S8_flip_cross", all(c["flip_equal"] and c["bulk_flip_equal"] for c in checks),
        "main and bulk flip_state byte equal")
    add("S9_axis_cos", all(c["axis_max_abs_1_minus_cos"] <= 64*np.finfo(float).eps
                            for c in checks),
        f"max={max(c['axis_max_abs_1_minus_cos'] for c in checks):.17g}")
    add("S10_finite", all(c["all_finite"] for c in checks), "required pair columns")
    add("S11_population", all(c["population_ok"] and c["monotone_ok"] for c in checks),
        "A=S union L and S monotone")
    add("S12_absorption", all(c["absorption_ok"] for c in checks),
        "start strict-off remains off with zero force and displacement")
    add("S13_force_decomposition", all(x[0] for x in f_checks), str([x[1] for x in f_checks]))
    add("S14_bulk_count", all(c["bulk_n_pair"] == 800 for c in checks),
        str([c["bulk_n_pair"] for c in checks]))

    mix_ok = True
    max_err = 0.0
    for metric in ("F", "D", "H"):
        lhs = seed_mix[f"A_{metric}"].to_numpy()
        rhs = seed_mix[f"C_S_{metric}"].to_numpy()+seed_mix[f"C_L_{metric}"].to_numpy()
        scale = np.abs(lhs)+np.abs(seed_mix[f"C_S_{metric}"])+np.abs(seed_mix[f"C_L_{metric}"])
        ok = mixed_close(lhs, rhs, F64_ATOL, F64_RTOL, scale)
        mix_ok &= bool(ok.all())
        max_err = max(max_err, float(np.max(np.abs(lhs-rhs))))
    add("S15_mixture", mix_ok, f"max_abs={max_err:.17g}")
    return pd.DataFrame(rows)


def verdicts(sanity_pass: bool, stat_pass: bool, contrast: pd.DataFrame) -> dict:
    if not sanity_pass:
        return {"sanity_status": "SANITY_FAIL", "T0_root": "SUPPRESSED",
                "T0_horizon": "SUPPRESSED"}
    if not stat_pass:
        return {"sanity_status": "STAT_SANITY_FAIL_ONE_STEP",
                "T0_root": "SUPPRESSED", "T0_horizon": "SUPPRESSED"}
    g = contrast[(contrast.estimate == "g") & (contrast.population == "A")
                 & (contrast.k.astype(str) == "40")].iloc[0]
    pg = contrast[contrast.estimate == "P_g"].iloc[0]
    if not np.isfinite(g.value):
        root_v = "INCONCLUSIVE_NO_ROOT"
    elif g.root_exist_rate < 0.95:
        root_v = "INCONCLUSIVE_UNSTABLE_ROOT"
    elif g.ci_lo > 0:
        root_v = "SEPARATION_POSITIVE"
    elif g.ci_hi < 0:
        root_v = "SEPARATION_NEGATIVE"
    else:
        root_v = "NO_RESOLVED_SEPARATION"
    if not np.isfinite(pg.value):
        horizon_v = "INCONCLUSIVE_NO_ROOT"
    elif pg.root_exist_rate < 0.95:
        horizon_v = "INCONCLUSIVE_UNSTABLE_ROOT"
    elif pg.ci_lo > 0:
        horizon_v = "GAP_MOVES_POSITIVE_WITH_HORIZON"
    elif pg.ci_hi < 0:
        horizon_v = "GAP_MOVES_NEGATIVE_WITH_HORIZON"
    else:
        horizon_v = "NO_RESOLVED_HORIZON_SHIFT"
    return {"sanity_status": "PASS", "T0_root": root_v,
            "g_A_40": g.value, "g_A_40_ci_lo": g.ci_lo,
            "g_A_40_ci_hi": g.ci_hi, "g_A_40_exist_rate": g.root_exist_rate,
            "T0_horizon": horizon_v, "P_g": pg.value, "P_g_ci_lo": pg.ci_lo,
            "P_g_ci_hi": pg.ci_hi, "P_g_exist_rate": pg.root_exist_rate}


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", default="results/ratchet_log_0819")
    parser.add_argument("--outdir", default="results/ceiling_t0_0828")
    parser.add_argument("--sanity-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    inp = (root / args.input).resolve() if not Path(args.input).is_absolute() else Path(args.input)
    out = (root / args.outdir).resolve() if not Path(args.outdir).is_absolute() else Path(args.outdir)
    git_state = git_info(root)
    out.mkdir(parents=True, exist_ok=True)
    paths = sorted((inp / "logs").glob("seed*.npz"),
                   key=lambda p: int(p.stem.removeprefix("seed")))
    all_curve, all_mix, all_sens, all_legacy, all_bulk, all_struct = [], [], [], [], [], []
    checks, schemas, f_checks, sources = [], [], [], []
    for path in paths:
        d = load_one(path)
        missing = REQUIRED-set(d)
        schemas.append({"seed": int(d["seed"]), "n_step": len(d["step"]), "keys": sorted(d)})
        sources.append({"path": str(path.relative_to(root)), "sha256": sha256(path),
                        "keys": sorted(d), "dtypes": {k: str(v.dtype) for k, v in d.items()},
                        "shapes": {k: list(v.shape) for k, v in d.items()}})
        if missing:
            raise RuntimeError(f"{path}: missing {sorted(missing)}")
        if int(d["period"]) != PERIOD or int(d["width"]) != WIDTH:
            raise RuntimeError(f"{path}: unexpected period/width")
        f_checks.append(source_checks(d))
        curve, mix, sens, legacy, bulk, struct, chk = seed_statistics(d)
        all_curve.extend(curve); all_mix.extend(mix); all_sens.extend(sens)
        all_legacy.extend(legacy); all_bulk.extend(bulk); all_struct.extend(struct)
        checks.append(chk)

    seed_curve = pd.DataFrame(all_curve)
    seed_mix = pd.DataFrame(all_mix)
    sanity = sanity_rows(paths, schemas, checks, f_checks, seed_curve, seed_mix)
    sanity.to_csv(out / "sanity.csv", index=False)
    pd.DataFrame(all_struct).to_csv(out / "structure_counts.csv", index=False)
    config = {"analysis": "ceiling_t0_0828", "spec_vault_commit": SPEC_VAULT_COMMIT,
              "k": list(KS), "period": PERIOD, "width": WIDTH,
              "start_offsets": [-100, -41], "band_width": 0.1,
              "main_guard": {"n_seed": MIN_MAIN_SEED, "n_pair": MIN_PAIRS},
              "descriptive_guard": {"n_seed": MIN_DESC_SEED, "n_pair": MIN_PAIRS},
              "bootstrap_B": BOOT_N, "bootstrap_seed": BOOT_SEED,
              "omp_num_threads": os.environ.get("OMP_NUM_THREADS")}
    write_json(out / "config.json", config)
    write_json(out / "provenance.json", {
        "git": git_state, "spec_vault_commit": SPEC_VAULT_COMMIT,
        "python": sys.version, "platform": platform.platform(),
        "numpy": np.__version__, "pandas": pd.__version__, "sources": sources,
    })
    structural_pass = bool(sanity["pass"].all())
    if not structural_pass:
        pd.DataFrame([{"sanity_status": "SANITY_FAIL", "T0_root": "SUPPRESSED",
                       "T0_horizon": "SUPPRESSED"}]).to_csv(out / "verdict.csv", index=False)
        raise SystemExit("T0 structural sanity failed")
    if args.sanity_only:
        print(f"T0 structural sanity PASS -> {out}")
        return

    curves = aggregate_curves(seed_curve)
    mixture = aggregate_mixture(seed_mix)
    sensitivity = aggregate_simple(pd.DataFrame(all_sens))
    legacy = aggregate_simple(pd.DataFrame(all_legacy))
    bulk = aggregate_simple(pd.DataFrame(all_bulk))
    curves.to_csv(out / "curves.csv", index=False)
    curves[["population", "k", "band", "n_pair", "n_seed", "seed_ids",
            "guard_main", "guard_descriptive"]].to_csv(out / "band_counts.csv", index=False)
    mixture.to_csv(out / "mixture.csv", index=False)
    sensitivity.to_csv(out / "sensitivity.csv", index=False)
    legacy.to_csv(out / "legacy_bands.csv", index=False)
    bulk.to_csv(out / "bulk_reference.csv", index=False)

    rng = np.random.default_rng(BOOT_SEED)
    draws = rng.integers(0, 10, size=(BOOT_N, 10))
    weights = np.zeros((BOOT_N, 10), dtype=np.float64)
    for s in range(10):
        weights[:, s] = (draws == s).sum(axis=1)
    roots, contrast, boot_pack = root_bootstrap(seed_curve, curves, weights)
    roots.to_csv(out / "roots.csv", index=False)
    contrast.to_csv(out / "contrasts.csv", index=False)

    s_one_seed = np.array([c["s_one"] for c in checks], dtype=np.float64)
    s_one_boot = weights @ s_one_seed / 10.0
    nonfinite_one = int((~np.isfinite(s_one_boot)).sum())
    if nonfinite_one:
        s_lo = s_hi = np.nan
        stat_pass = False
    else:
        s_lo, s_hi = np.percentile(s_one_boot, [2.5, 97.5])
        stat_pass = bool(s_lo <= 0 <= s_hi)
    boot_summary = boot_pack["bootstrap_summary"]
    boot_summary = pd.concat([boot_summary, pd.DataFrame([{
        "estimate": "S_one", "population": "A", "k": 1,
        "value": float(s_one_seed.mean()), "ci_lo": s_lo, "ci_hi": s_hi,
        "root_exist_rate": np.nan, "n_exist": BOOT_N-nonfinite_one,
    }])], ignore_index=True)
    boot_summary.to_csv(out / "bootstrap_summary.csv", index=False)

    verdict = verdicts(structural_pass, stat_pass, contrast)
    verdict.update({"S_one": float(s_one_seed.mean()), "S_one_ci_lo": s_lo,
                    "S_one_ci_hi": s_hi, "bootstrap_B": BOOT_N,
                    "bootstrap_seed": BOOT_SEED})
    pd.DataFrame([verdict]).to_csv(out / "verdict.csv", index=False)
    summary = f"""# ceiling_t0_0828

## 主判定

- 構造・統計 sanity: **{verdict['sanity_status']}**
- T0_root: **{verdict['T0_root']}**
- `g_A(40)`: {verdict.get('g_A_40', np.nan):+.8g} [{verdict.get('g_A_40_ci_lo', np.nan):+.8g}, {verdict.get('g_A_40_ci_hi', np.nan):+.8g}]
- 同時根存在率: {verdict.get('g_A_40_exist_rate', np.nan):.4f}
- T0_horizon: **{verdict['T0_horizon']}**
- `P_g=g_A(40)-g_A(1)`: {verdict.get('P_g', np.nan):+.8g} [{verdict.get('P_g_ci_lo', np.nan):+.8g}, {verdict.get('P_g_ci_hi', np.nan):+.8g}]
- 4根同時存在率: {verdict.get('P_g_exist_rate', np.nan):.4f}

## 1-step sanity

- `S_one=E[D_1-F_gate,start]`: {verdict['S_one']:+.8g} [{s_lo:+.8g}, {s_hi:+.8g}]
- 95% CI が 0 を含む: **{stat_pass}**

## 適用範囲

condA・w100・T=10,000・batch=1・std の境界前窓。同一開始コホートの k<=40 に限定する。
bulk Δ1000 は `bulk_reference.csv` の別領域参考値であり、主判定には含めない。
"""
    (out / "summary.md").write_text(summary)
    print(f"T0 {verdict['T0_root']} / {verdict['T0_horizon']} -> {out}")


if __name__ == "__main__":
    main()
