"""T0b: window mean of H on the upper window [+0.1, +0.9).

Frozen specification: obsidian-research commit b44078c,
``可塑性喪失/spec/天井T0b_spec_0828.md``.

The primary statistic is ``M(40)``: the seed-equal-weight mean over contributing
seeds of the per-seed pair-weighted mean of ``H_k = D_k - F_0`` over population A
inside the window.  The window is defined directly on the start coordinate
``x = cos_u_mu * w_norm``; bands are descriptive only (spec §2.3).

Pair-level definitions are identical to the predecessor
``analysis/ceiling_t0/ceiling_t0.py`` (spec §2.1); the shared helpers are
imported from it rather than re-typed.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.ceiling_t0.ceiling_t0 import (
    REQUIRED,
    bootstrap_curve,
    git_info,
    load_one,
    primary_indices,
    sha256,
    source_checks,
    write_json,
)

SPEC_VAULT_COMMIT = "b44078c"          # pre-registration freeze (judgement criteria)
SPEC_VAULT_AMENDMENT = "7f6b7d7"        # §4 CI: percentile -> studentized (spec §12)
CI_METHOD = "studentized"
KS = (1, 2, 5, 10, 20, 40)
K_MAIN = 40
K_BASE = 1
WINDOW_LO = 0.1
WINDOW_HI = 0.9
BAND_W = 0.1
BLIP_BAND = 6
PERIOD = 10_000
WIDTH = 100
BOOT_N = 10_000
BOOT_SEED = 20260829
MIN_MAIN_SEED = 8
MIN_PAIRS = 300
MU_NORM_MEDIAN_MIN = 1.0
NEAR_TAU = 0.05

# Structural constants of the 0819 recording grid (spec §7 B2/B5).
N_STEP = 20_901
N_BOUNDARY = 99
N_START_PER_BOUNDARY = 60
GAP_COUNTS = {1: 19_900, 900: 199, 1000: 801}

# Preflight constants (spec §6).
PREFLIGHT_SEED = 20260829
PLANT_H = 1.0e-4
P1_TOL = 0.10
P2_REPS = 200
P2_MAX_REJECT = 0.07
P3_BLIP_MAG = 3.0e-4
P3_MAX_SHIFT = 0.15
P5_RTOL = 1.0e-9
SYN_PAIR_SIGMA = 5.0e-4
SYN_SEED_TAU = 1.0e-5
# Support-only inputs used to shape the synthetic preflight data (spec §3.1/§3.2).
SYN_SEED_PAIRS = (10713, 12622, 14533, 16443, 18354, 20265, 22175, 24086, 25996, 27899)
SYN_BAND_SHARE = (80087, 39741, 18550, 12311, 6266, 3346, 1663, 1108)

QUANTITIES = ("H", "F0", "D", "F_path", "T1", "T2")
VARIANTS = ("main", "win_lo_m1", "win_lo_p1", "win_hi_m1", "win_hi_p1",
            "a_no_near", "no_blip", "no_strict_off")
VARIANT_WINDOW = {
    "main": (WINDOW_LO, WINDOW_HI),
    "win_lo_m1": (WINDOW_LO - BAND_W, WINDOW_HI),
    "win_lo_p1": (WINDOW_LO + BAND_W, WINDOW_HI),
    "win_hi_m1": (WINDOW_LO, WINDOW_HI - BAND_W),
    "win_hi_p1": (WINDOW_LO, WINDOW_HI + BAND_W),
    "a_no_near": (WINDOW_LO, WINDOW_HI),
    "no_blip": (WINDOW_LO, WINDOW_HI),
    "no_strict_off": (WINDOW_LO, WINDOW_HI),
}
WINDOW_BANDS = tuple(range(int(round(WINDOW_LO * 10)), int(round(WINDOW_HI * 10))))


# --------------------------------------------------------------------------
# pair construction
# --------------------------------------------------------------------------


@dataclass
class SeedPairs:
    """Pair table for one seed.  Rows are (start, unit) in C order."""

    seed: int
    x0: np.ndarray
    p0: np.ndarray
    f0: np.ndarray
    band: np.ndarray
    per_k: dict


def pair_coordinate(d: dict[str, np.ndarray]) -> np.ndarray:
    """Spec §2.1: x := cos_u_mu * w_norm (predecessor ceiling_t0.py L131)."""
    return d["cos_u_mu"].astype(np.float64) * d["w_norm"].astype(np.float64)


def pairs_from_npz(d: dict[str, np.ndarray]) -> SeedPairs:
    step = d["step"]
    x = pair_coordinate(d)
    p = d["p_hat"].astype(np.float64)
    fg = d["F_gate"].astype(np.float64)
    flip = d["flip_state"]
    _, starts, _ = primary_indices(d)
    n_unit = x.shape[1]
    x_start = x[starts]
    f_start = fg[starts]
    x0 = x_start.ravel()
    f0 = f_start.ravel()
    per_k: dict[int, dict[str, np.ndarray]] = {}
    for k in KS:
        ends = starts + k
        if not np.array_equal(step[ends], step[starts] + k):
            raise RuntimeError(f"non-contiguous record window seed={int(d['seed'])}, k={k}")
        disp = ((x[ends] - x_start) / k).ravel()
        path = np.zeros_like(x_start)
        for u in range(k):
            path += fg[starts + u]
        path = (path / k).ravel()
        per_k[k] = {
            "p1": p[ends].ravel(),
            "disp": disp,
            "f_path": path,
            "h": disp - f0,
            "flip_ok": np.repeat((flip[starts] == flip[ends]).all(axis=1), n_unit),
        }
    return SeedPairs(seed=int(d["seed"]), x0=x0, p0=p[starts].ravel(), f0=f0,
                     band=np.floor(x0 * 10.0).astype(np.int64), per_k=per_k)


def in_window(x0: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (x0 >= lo) & (x0 < hi)


def variant_masks(sp: SeedPairs, k: int) -> dict[str, np.ndarray]:
    """Spec §3 population/guard rules plus the §5-5/§5-6 variants."""
    dk = sp.per_k[k]
    same_axis = dk["flip_ok"]
    out: dict[str, np.ndarray] = {}
    for name in VARIANTS:
        lo, hi = VARIANT_WINDOW[name]
        m = in_window(sp.x0, lo, hi) & same_axis
        m &= (sp.p0 >= NEAR_TAU) if name == "a_no_near" else (sp.p0 > 0)
        if name == "no_blip":
            m &= sp.band != BLIP_BAND
        if name == "no_strict_off":
            m &= dk["p1"] > 0
        out[name] = m
    return out


def quantities(sp: SeedPairs, k: int) -> dict[str, np.ndarray]:
    dk = sp.per_k[k]
    return {"H": dk["h"], "F0": sp.f0, "D": dk["disp"], "F_path": dk["f_path"],
            "T1": dk["f_path"] - sp.f0, "T2": dk["disp"] - dk["f_path"]}


def seed_summary(sp: SeedPairs, ks: tuple[int, ...] | None = None
                 ) -> tuple[list[dict], list[dict], dict]:
    """Per-seed variant means and per-band sufficient statistics."""
    ks = tuple(sp.per_k) if ks is None else ks
    vrows: list[dict] = []
    brows: list[dict] = []
    strict_off = {}
    for k in ks:
        q = quantities(sp, k)
        for name, m in variant_masks(sp, k).items():
            n = int(m.sum())
            row = {"seed": sp.seed, "k": k, "variant": name, "n": n}
            for qn, qv in q.items():
                s = float(qv[m].sum()) if n else np.nan
                row[f"sum_{qn}"] = s
                row[f"mean_{qn}"] = s / n if n else np.nan
            vrows.append(row)
        base = (sp.p0 > 0) & sp.per_k[k]["flip_ok"]
        win = base & in_window(sp.x0, WINDOW_LO, WINDOW_HI)
        strict_off[k] = int((win & (sp.per_k[k]["p1"] <= 0)).sum())
        bb = sp.band[base]
        uniq, inv = np.unique(bb, return_inverse=True)
        cnt = np.bincount(inv)
        sums = {qn: np.bincount(inv, weights=qv[base]) for qn, qv in q.items()}
        for i, b in enumerate(uniq):
            row = {"seed": sp.seed, "k": k, "band": int(b), "count": int(cnt[i])}
            for qn in q:
                row[f"sum_{qn}"] = float(sums[qn][i])
                row[f"mean_{qn}"] = float(sums[qn][i]) / cnt[i]
            brows.append(row)
    checks = {
        "seed": sp.seed,
        "strict_off_in_window": strict_off,
        "flip_all_equal": all(bool(sp.per_k[k]["flip_ok"].all()) for k in ks),
        "flip_dropped": {k: int((~sp.per_k[k]["flip_ok"]).sum()) for k in ks},
        "h_identity_max": max(
            float(np.max(np.abs(sp.per_k[k]["h"] - (sp.per_k[k]["disp"] - sp.f0))))
            for k in ks),
        "h_finite": all(bool(np.isfinite(sp.per_k[k]["h"]).all()) for k in ks),
        "band_window_agrees": bool(np.array_equal(
            in_window(sp.x0, WINDOW_LO, WINDOW_HI),
            (sp.band >= WINDOW_BANDS[0]) & (sp.band <= WINDOW_BANDS[-1]))),
        "n_pair_window": int(in_window(sp.x0, WINDOW_LO, WINDOW_HI).sum()),
    }
    return vrows, brows, checks


# --------------------------------------------------------------------------
# seed-cluster bootstrap (spec §3 / §4)
# --------------------------------------------------------------------------


def bootstrap_weights(n_seed: int, boot_n: int = BOOT_N,
                      seed: int = BOOT_SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_seed, size=(boot_n, n_seed))
    weights = np.zeros((boot_n, n_seed), dtype=np.float64)
    for s in range(n_seed):
        weights[:, s] = (draws == s).sum(axis=1)
    return weights


def cluster_interval(values: np.ndarray, weights: np.ndarray,
                     method: str) -> dict:
    """95% CI for a seed-equal-weight mean under the seed-cluster bootstrap.

    ``values`` carries NaN for seeds that do not contribute (spec §3 guard).
    ``method`` is ``percentile`` (spec §4 as frozen) or ``studentized``.
    """
    finite = np.isfinite(values)
    n_c = int(finite.sum())
    out = {"point": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
           "n_seed": n_c, "boot_ok": 0, "ci_method": method}
    if n_c == 0:
        return out
    v = np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0)
    cnt = weights @ finite.astype(np.float64)
    s1 = weights @ v
    s2 = weights @ (v * v)
    have = cnt >= 1
    bm = np.full(weights.shape[0], np.nan)
    bm[have] = s1[have] / cnt[have]
    point = float(values[finite].mean())
    out["point"] = point
    if method == "percentile":
        good = np.isfinite(bm)
        if good.any():
            lo, hi = np.percentile(bm[good], [2.5, 97.5])
            out.update(ci_lo=float(lo), ci_hi=float(hi), boot_ok=int(good.sum()))
        return out
    if n_c < 2:
        return out
    two = cnt >= 2
    var = np.full(weights.shape[0], np.nan)
    var[two] = (s2[two] - cnt[two] * bm[two] ** 2) / (cnt[two] - 1.0)
    se = np.full(weights.shape[0], np.nan)
    se[two] = np.sqrt(np.maximum(var[two], 0.0) / cnt[two])
    good = np.isfinite(bm) & np.isfinite(se) & (se > 0)
    se0 = float(values[finite].std(ddof=1)) / np.sqrt(n_c)
    if not good.any() or not np.isfinite(se0):
        return out
    t = (bm[good] - point) / se[good]
    t_lo, t_hi = np.percentile(t, [2.5, 97.5])
    out.update(ci_lo=float(point - t_hi * se0), ci_hi=float(point - t_lo * se0),
               boot_ok=int(good.sum()))
    return out


def pair_weight_interval(sums: np.ndarray, counts: np.ndarray,
                         weights: np.ndarray) -> dict:
    """Spec §5-5: pair-equal weighting, reported as a sensitivity only."""
    keep = np.isfinite(sums) & (counts > 0)
    if not keep.any():
        return {"point": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                "n_seed": 0, "boot_ok": 0, "ci_method": "percentile"}
    s = np.where(keep, np.nan_to_num(sums), 0.0)
    c = np.where(keep, counts, 0.0).astype(np.float64)
    num = weights @ s
    den = weights @ c
    good = den > 0
    boot = np.full(weights.shape[0], np.nan)
    boot[good] = num[good] / den[good]
    lo, hi = np.percentile(boot[good], [2.5, 97.5])
    return {"point": float(s[keep].sum() / c[keep].sum()), "ci_lo": float(lo),
            "ci_hi": float(hi), "n_seed": int(keep.sum()),
            "boot_ok": int(good.sum()), "ci_method": "percentile"}


def seed_series(seed_variant: pd.DataFrame, seed_ids: list[int], variant: str,
                k: int, column: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (values, counts) aligned to ``seed_ids``; NaN below the pair guard."""
    sub = seed_variant[(seed_variant.variant == variant) & (seed_variant.k == k)]
    lookup = {int(r.seed): r for r in sub.itertuples()}
    values = np.full(len(seed_ids), np.nan)
    counts = np.zeros(len(seed_ids))
    for i, sid in enumerate(seed_ids):
        r = lookup.get(int(sid))
        if r is None:
            continue
        counts[i] = float(r.n)
        if int(r.n) >= MIN_PAIRS:
            values[i] = float(getattr(r, column))
    return values, counts


# --------------------------------------------------------------------------
# isotonic zeros (spec §5-7, descriptive only)
# --------------------------------------------------------------------------


def pava_decreasing(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted pool-adjacent-violators fit under a non-increasing constraint."""
    vals = [float(v) for v in y]
    wts = [float(v) for v in w]
    cnt = [1] * len(vals)
    i = 0
    while i < len(vals) - 1:
        if vals[i] >= vals[i + 1]:
            i += 1
            continue
        total = wts[i] + wts[i + 1]
        vals[i] = (vals[i] * wts[i] + vals[i + 1] * wts[i + 1]) / total
        wts[i] = total
        cnt[i] += cnt[i + 1]
        del vals[i + 1], wts[i + 1], cnt[i + 1]
        if i > 0:
            i -= 1
    return np.repeat(np.array(vals), cnt)


def descending_crossing(x: np.ndarray, y: np.ndarray) -> float:
    """Unique descending zero of a non-increasing step profile (spec §5-7)."""
    for i in range(len(y) - 1):
        if y[i] > 0 and y[i + 1] < 0:
            return float(x[i] - y[i] * (x[i + 1] - x[i]) / (y[i + 1] - y[i]))
    zero = np.flatnonzero(y == 0)
    if zero.size and zero[0] > 0 and zero[-1] < len(y) - 1 \
            and (y[:zero[0]] > 0).all() and (y[zero[-1] + 1:] < 0).all():
        return float(0.5 * (x[zero[0]] + x[zero[-1]]))
    return float("nan")


def band_matrix(seed_band: pd.DataFrame, seed_ids: list[int], k: int,
                bands: list[int], column: str) -> np.ndarray:
    out = np.full((len(seed_ids), len(bands)), np.nan)
    pos = {int(b): j for j, b in enumerate(bands)}
    row = {int(s): i for i, s in enumerate(seed_ids)}
    for r in seed_band[seed_band.k == k].itertuples():
        if int(r.band) in pos:
            out[row[int(r.seed)], pos[int(r.band)]] = float(getattr(r, column))
    return out


def isotonic_table(seed_band: pd.DataFrame, seed_ids: list[int],
                   weights: np.ndarray) -> pd.DataFrame:
    """Spec §5-7.  Descriptive geometry; never enters a verdict."""
    rows = []
    for k in KS:
        counts = band_matrix(seed_band, seed_ids, k, list(WINDOW_BANDS), "count")
        n_pair = np.nansum(np.where(np.isfinite(counts), counts, 0.0), axis=0)
        n_seed = np.isfinite(counts).sum(axis=0)
        keep = (n_seed >= MIN_MAIN_SEED) & (n_pair >= MIN_PAIRS)
        bands = [b for b, ok in zip(WINDOW_BANDS, keep) if ok]
        if len(bands) < 2:
            rows.append({"k": k, "bands_used": "", "n_band": len(bands),
                         "role": "descriptive_not_used_in_verdict"})
            continue
        centers = np.array([(b + 0.5) * BAND_W for b in bands])
        w = n_pair[keep]
        crossing = {}
        boot = {}
        sign = {}
        for metric, column in (("F", "mean_F0"), ("D", "mean_D")):
            seed_vals = band_matrix(seed_band, seed_ids, k, bands, column)
            point = np.array([np.nanmean(seed_vals[:, j]) for j in range(len(bands))])
            fit = pava_decreasing(point, w)
            crossing[metric] = descending_crossing(centers, fit)
            sign[metric] = ("all_negative" if (fit < 0).all() else
                            "all_positive" if (fit > 0).all() else "mixed")
            curves = bootstrap_curve(seed_vals, weights)
            boot[metric] = np.array([
                descending_crossing(centers, pava_decreasing(row, w))
                if np.isfinite(row).all() else np.nan for row in curves])
        g = crossing["D"] - crossing["F"]
        both = np.isfinite(boot["D"]) & np.isfinite(boot["F"])
        rate = float(both.mean())
        row = {"k": k, "bands_used": ",".join(map(str, bands)), "n_band": len(bands),
               "z_tilde_F": crossing["F"], "z_tilde_D": crossing["D"],
               "g_tilde": g, "exist_rate": rate,
               "F_window_sign": sign["F"], "D_window_sign": sign["D"],
               "note": ("" if np.isfinite(g) else
                        "no descending crossing inside the window: the smoothed F is "
                        f"{sign['F']} and the smoothed D is {sign['D']} "
                        "on the guarded window bands"),
               "role": "descriptive_not_used_in_verdict",
               "ci_method": "percentile"}
        for name, arr in (("z_tilde_F", boot["F"]), ("z_tilde_D", boot["D"]),
                          ("g_tilde", boot["D"] - boot["F"])):
            ok = np.isfinite(arr)
            if ok.sum() and rate >= 0.95:
                lo, hi = np.percentile(arr[ok], [2.5, 97.5])
            else:
                lo = hi = np.nan
            row[f"{name}_ci_lo"] = float(lo)
            row[f"{name}_ci_hi"] = float(hi)
        rows.append(row)
    return pd.DataFrame(rows)


def band_table(seed_band: pd.DataFrame, seed_ids: list[int]) -> pd.DataFrame:
    """Spec §5-3: per-band breakdown of the window (descriptive)."""
    rows = []
    sub = seed_band[seed_band.band.isin(WINDOW_BANDS)]
    for (k, band), g in sub.groupby(["k", "band"], sort=True):
        support = tuple(sorted(map(int, g.seed.unique())))
        row = {"population": "A", "k": int(k), "band": int(band),
               "band_left": band * BAND_W, "band_right": (band + 1) * BAND_W,
               "band_center": (band + 0.5) * BAND_W,
               "n_pair": int(g["count"].sum()), "n_seed": len(support),
               "seed_ids": ",".join(map(str, support)),
               "guard_main": bool(len(support) >= MIN_MAIN_SEED
                                  and g["count"].sum() >= MIN_PAIRS)}
        for q in QUANTITIES:
            row[q] = float(g[f"mean_{q}"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def window_estimates(seed_variant: pd.DataFrame, seed_ids: list[int],
                     weights: np.ndarray, ci_method: str,
                     ks: tuple[int, ...] = KS) -> dict:
    """All window-level point estimates and CIs, keyed by (variant, k, quantity)."""
    out: dict[tuple[str, int, str], dict] = {}
    for variant in VARIANTS:
        for k in ks:
            for q in QUANTITIES:
                vals, cnts = seed_series(seed_variant, seed_ids, variant, k,
                                         f"mean_{q}")
                res = cluster_interval(vals, weights, ci_method)
                res.update(variant=variant, k=k, quantity=q,
                           n_pair=int(cnts.sum()))
                out[(variant, k, q)] = res
    for k in ks:
        sums, cnts = seed_series(seed_variant, seed_ids, "main", k, "sum_H")
        res = pair_weight_interval(sums, cnts, weights)
        res.update(variant="pair_weight", k=k, quantity="H",
                   n_pair=int(cnts.sum()))
        out[("pair_weight", k, "H")] = res
    return out


def curve_frame(est: dict) -> pd.DataFrame:
    rows = []
    for k in KS:
        r = est[("main", k, "H")]
        rows.append({"k": k, "M": r["point"], "ci_lo": r["ci_lo"],
                     "ci_hi": r["ci_hi"], "n_pair": r["n_pair"],
                     "n_seed_contrib": r["n_seed"], "boot_ok": r["boot_ok"],
                     "ci_method": r["ci_method"]})
    return pd.DataFrame(rows)


def decomposition_frame(est: dict) -> pd.DataFrame:
    rows = []
    for k in KS:
        row = {"k": k}
        for q in ("H", "T1", "T2", "F0", "D", "F_path"):
            r = est[("main", k, q)]
            row[q] = r["point"]
            row[f"{q}_ci_lo"] = r["ci_lo"]
            row[f"{q}_ci_hi"] = r["ci_hi"]
        row["T1_plus_T2_minus_H"] = row["T1"] + row["T2"] - row["H"]
        row["term1_meaning"] = "F_path - F_0 (force changes along the path)"
        row["term2_meaning"] = "D_k - F_path (gate, absorption, sign cancellation)"
        rows.append(row)
    return pd.DataFrame(rows)


def sensitivity_frame(est: dict) -> pd.DataFrame:
    labels = {
        "win_lo_m1": "window lower edge -1 band: x in [0.0, 0.9)",
        "win_lo_p1": "window lower edge +1 band: x in [0.2, 0.9)",
        "win_hi_m1": "window upper edge -1 band: x in [0.1, 0.8)",
        "win_hi_p1": "window upper edge +1 band: x in [0.1, 1.0)",
        "a_no_near": "population A_no_near: p_hat_start >= 0.05",
        "pair_weight": "pair-equal weighting instead of seed-equal",
        "no_blip": "window minus band [0.6,0.7) (spec 5-4 blip contribution)",
        "no_strict_off": "window minus pairs that are strict-off at the end (spec 5-6)",
    }
    rows = []
    for k in KS:
        base = est[("main", k, "H")]
        for variant, note in labels.items():
            r = est[(variant, k, "H")]
            rows.append({"variant": variant, "k": k, "M": r["point"],
                         "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"],
                         "n_pair": r["n_pair"], "n_seed_contrib": r["n_seed"],
                         "delta_vs_main": r["point"] - base["point"],
                         "rel_delta_vs_main": ((r["point"] - base["point"]) / base["point"]
                                               if base["point"] else np.nan),
                         "ci_method": r["ci_method"], "note": note})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# structural assertions B1-B10 (spec §7)
# --------------------------------------------------------------------------


def mu_norm_stats(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        mu = np.asarray(z["mu_norm"], dtype=np.float64)
    return {"path": str(path), "median": float(np.median(mu)),
            "q05": float(np.quantile(mu, 0.05)),
            "frac_below_half": float((mu < 0.5).mean())}


def seed_evidence(d: dict[str, np.ndarray], checks: dict) -> dict:
    """Everything B1-B7 and B10 need from one npz plus its pair summary."""
    step = d["step"]
    gaps = np.diff(step)
    boundary_idx, starts, _ = primary_indices(d)
    starts_step = step[starts]
    expected = np.concatenate(
        [np.arange(int(b) - WIDTH, int(b) - (WIDTH - N_START_PER_BOUNDARY))
         for b in step[boundary_idx]])
    x = pair_coordinate(d)
    force_ok, force_note = source_checks(d)
    mu = d["mu_norm"].astype(np.float64)
    return {
        "seed": int(d["seed"]),
        "keys": sorted(d),
        "missing": sorted(REQUIRED - set(d)),
        "n_step": int(len(step)),
        "gap_counts": {int(v): int((gaps == v).sum()) for v in np.unique(gaps)},
        "force_ok": force_ok,
        "force_note": force_note,
        "coord_ok": bool(np.array_equal(
            x, d["cos_u_mu"].astype(np.float64) * d["w_norm"].astype(np.float64))
            and np.isfinite(x).all()),
        "n_boundary": int(len(boundary_idx)),
        "n_start": int(len(starts)),
        "starts_match": bool(np.array_equal(starts_step, expected)),
        "contiguous": bool(all(np.array_equal(step[starts + k], starts_step + k)
                               for k in KS)),
        "flip_all_equal": checks["flip_all_equal"],
        "flip_dropped": checks["flip_dropped"],
        "mu_median": float(np.median(mu)),
        "mu_q05": float(np.quantile(mu, 0.05)),
        "mu_frac_below_half": float((mu < 0.5).mean()),
        "h_identity_max": checks["h_identity_max"],
        "h_finite": checks["h_finite"],
        "band_window_agrees": checks["band_window_agrees"],
        "n_pair_window": checks["n_pair_window"],
    }


def runner_supports_arm_c(root: Path) -> tuple[bool, str]:
    """B9: the arm-C generator exposes --seeds and --outdir."""
    src = (root / "src" / "ratchet_log.py").read_text()
    have = {flag: (f'"{flag}"' in src) for flag in ("--seeds", "--outdir")}
    return all(have.values()), json.dumps(have)


def assertion_frame(evidence: list[dict], centered: list[dict],
                    root: Path) -> pd.DataFrame:
    rows: list[dict] = []

    def add(i: str, ok: bool, what: str, note: str) -> None:
        rows.append({"id": i, "pass": bool(ok), "assertion": what, "note": note})

    add("B1", bool(evidence) and all(not e["missing"] for e in evidence),
        "set(REQUIRED) is a subset of npz.files",
        f"n_seed={len(evidence)}, missing={[e['missing'] for e in evidence]}")
    add("B2", bool(evidence) and all(e["gap_counts"] == GAP_COUNTS
                                     and e["n_step"] == N_STEP for e in evidence),
        "np.diff(step) distribution and record count are the known grid",
        f"expected={GAP_COUNTS} n_step={N_STEP}; "
        f"seen={[(e['n_step'], e['gap_counts']) for e in evidence]}")
    add("B3", bool(evidence) and all(e["force_ok"] for e in evidence),
        "F_gate == F_self + F_rest (float32 tolerance)",
        "; ".join(e["force_note"] for e in evidence))
    add("B4", bool(evidence) and all(e["coord_ok"] for e in evidence),
        "x == cos_u_mu * w_norm and finite",
        f"seeds_ok={[e['coord_ok'] for e in evidence]}")
    add("B5", bool(evidence) and all(
        e["n_boundary"] == N_BOUNDARY
        and e["n_start"] == N_BOUNDARY * N_START_PER_BOUNDARY
        and e["starts_match"] and e["contiguous"] for e in evidence),
        "60 starts per boundary on [b-100, b-41] with contiguous records",
        f"n_boundary={[e['n_boundary'] for e in evidence]}, "
        f"n_start={[e['n_start'] for e in evidence]}")
    add("B6", bool(evidence) and all(e["flip_all_equal"] for e in evidence),
        "flip_state_start == flip_state_end on every main pair",
        f"dropped={[e['flip_dropped'] for e in evidence]}")
    add("B7", bool(evidence) and all(e["mu_median"] >= MU_NORM_MEDIAN_MIN
                                     for e in evidence),
        f"median(mu_norm) >= {MU_NORM_MEDIAN_MIN} on the analysed logs",
        f"median={[round(e['mu_median'], 4) for e in evidence]}, "
        f"q05={[round(e['mu_q05'], 4) for e in evidence]}, "
        f"frac<0.5={[round(e['mu_frac_below_half'], 4) for e in evidence]}")
    add("B8", bool(centered) and all(c["median"] < MU_NORM_MEDIAN_MIN
                                     for c in centered),
        "the same threshold rejects centered (out of scope for this spec)",
        f"median={[round(c['median'], 4) for c in centered]}, "
        f"q05={[round(c['q05'], 4) for c in centered]}, "
        f"frac<0.5={[round(c['frac_below_half'], 4) for c in centered]}")
    ok9, note9 = runner_supports_arm_c(root)
    add("B9", ok9, "src/ratchet_log.py exposes --seeds and --outdir", note9)
    add("B10", bool(evidence) and all(e["h_identity_max"] == 0.0 and e["h_finite"]
                                      for e in evidence),
        "implemented per-pair H equals disp - f0 and is finite",
        f"max_abs={[e['h_identity_max'] for e in evidence]}")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# preflight P1-P5 (spec §6)
# --------------------------------------------------------------------------


def synthetic_x(n_pair: int) -> np.ndarray:
    """Window coordinates laid out on the recorded band support (spec §3.1)."""
    share = np.array(SYN_BAND_SHARE, dtype=np.float64)
    share /= share.sum()
    counts = np.floor(share * n_pair).astype(np.int64)
    counts[0] += n_pair - int(counts.sum())
    parts = []
    for band, c in zip(WINDOW_BANDS, counts):
        if c <= 0:
            continue
        parts.append(band * BAND_W + (np.arange(c) + 0.5) / c * BAND_W)
    return np.concatenate(parts)


def synthetic_dataset(seed: int, plant: float = PLANT_H,
                      sigma: float = SYN_PAIR_SIGMA, tau: float = SYN_SEED_TAU,
                      blip_band: int | None = None,
                      blip_value: float | None = None) -> list[SeedPairs]:
    """Synthetic seed group whose window mean of H is exactly ``plant``.

    Pair noise is centred exactly inside each seed and the seed offsets are
    centred exactly across seeds, so P1 tests the estimator rather than a draw.
    """
    rng = np.random.default_rng(seed)
    offsets = rng.normal(size=len(SYN_SEED_PAIRS))
    offsets -= offsets.mean()
    if offsets.std() > 0:
        offsets = offsets / offsets.std() * tau
    dataset = []
    for i, n_pair in enumerate(SYN_SEED_PAIRS):
        x0 = synthetic_x(n_pair)
        band = np.floor(x0 * 10.0).astype(np.int64)
        noise = rng.normal(size=len(x0))
        noise -= noise.mean()
        if noise.std() > 0:
            noise = noise / noise.std() * sigma
        h = plant + offsets[i] + noise
        if blip_band is not None:
            h = np.where(band == blip_band, blip_value, h)
        zeros = np.zeros(len(x0))
        dataset.append(SeedPairs(
            seed=i, x0=x0, p0=np.ones(len(x0)), f0=zeros, band=band,
            per_k={K_MAIN: {"p1": np.ones(len(x0)), "disp": h, "f_path": zeros,
                            "h": h, "flip_ok": np.ones(len(x0), dtype=bool)}}))
    return dataset


def estimate_dataset(dataset: list[SeedPairs], weights: np.ndarray,
                     ci_method: str) -> dict:
    vrows: list[dict] = []
    for sp in dataset:
        v, _b, _c = seed_summary(sp, ks=(K_MAIN,))
        vrows.extend(v)
    ids = [sp.seed for sp in dataset]
    return window_estimates(pd.DataFrame(vrows), ids, weights, ci_method,
                            ks=(K_MAIN,))


def seed_window_means(dataset: list[SeedPairs]) -> tuple[np.ndarray, np.ndarray]:
    vals, counts = [], []
    for sp in dataset:
        m = variant_masks(sp, K_MAIN)["main"]
        h = sp.per_k[K_MAIN]["h"][m]
        vals.append(float(h.mean()) if h.size else np.nan)
        counts.append(float(h.size))
    return np.array(vals), np.array(counts)


def null_reject_rate(dataset: list[SeedPairs], weights: np.ndarray, method: str,
                     reps: int, seed: int) -> float:
    """P2: per-pair sign randomisation, share of runs that do not return
    NO_RESOLVED_WINDOW_H."""
    rng = np.random.default_rng(seed)
    parts = []
    for sp in dataset:
        m = variant_masks(sp, K_MAIN)["main"]
        parts.append(sp.per_k[K_MAIN]["h"][m])
    reject = 0
    for _ in range(reps):
        means = np.array([float((h * (rng.integers(0, 2, size=h.size) * 2 - 1)).mean())
                          if h.size >= MIN_PAIRS else np.nan for h in parts])
        res = cluster_interval(means, weights, method)
        if not (res["ci_lo"] <= 0.0 <= res["ci_hi"]):
            reject += 1
    return reject / reps


def preflight_pre_data(weights: np.ndarray, ci_method: str) -> list[dict]:
    rows: list[dict] = []

    def add(i, stage, check, statistic, threshold, value, ok, gating, note):
        rows.append({"id": i, "stage": stage, "check": check,
                     "statistic": statistic, "threshold": threshold,
                     "value": value, "pass": bool(ok), "gating": bool(gating),
                     "note": note})

    planted = synthetic_dataset(PREFLIGHT_SEED)
    est = estimate_dataset(planted, weights, ci_method)[("main", K_MAIN, "H")]
    rel = abs(est["point"] / PLANT_H - 1.0)
    covered = bool(est["ci_lo"] <= PLANT_H <= est["ci_hi"])
    add("P1", "pre_data", "planted recovery", "|M(40)/h*-1|", P1_TOL, rel,
        rel <= P1_TOL and covered, True,
        f"h*={PLANT_H:.6g}, M(40)={est['point']:.10g}, "
        f"CI=[{est['ci_lo']:.6g}, {est['ci_hi']:.6g}], ci_covers_h*={covered}")

    rate = null_reject_rate(planted, weights, ci_method, P2_REPS, PREFLIGHT_SEED + 1)
    add("P2", "pre_data", "sign-randomised null", "reject rate", P2_MAX_REJECT,
        rate, rate <= P2_MAX_REJECT, True,
        f"reps={P2_REPS}, ci_method={ci_method}, nominal=0.05")
    alt = "studentized" if ci_method == "percentile" else "percentile"
    rate_alt = null_reject_rate(planted, weights, alt, P2_REPS, PREFLIGHT_SEED + 1)
    add("P2_alt", "pre_data", f"sign-randomised null under {alt} CI",
        "reject rate", P2_MAX_REJECT, rate_alt, rate_alt <= P2_MAX_REJECT, False,
        f"diagnostic only, not a gate; reps={P2_REPS}. spec §12 の CI 差し替えは "
        f"この対比に基づく")

    blipped = synthetic_dataset(PREFLIGHT_SEED, blip_band=BLIP_BAND,
                                blip_value=-P3_BLIP_MAG)
    est_b = estimate_dataset(blipped, weights, ci_method)[("main", K_MAIN, "H")]
    shift = abs(est_b["point"] - est["point"]) / abs(est["point"])
    add("P3", "pre_data", "single-band blip robustness", "|dM|/|M|",
        P3_MAX_SHIFT, shift, shift < P3_MAX_SHIFT, True,
        f"band {BLIP_BAND} forced to {-P3_BLIP_MAG:.6g}; "
        f"M_base={est['point']:.10g}, M_blip={est_b['point']:.10g}")

    means, counts = seed_window_means(planted)
    ok4, loo = leave_one_out_sign(means, counts)
    add("P4", "pre_data", "leave-one-seed-out sign stability", "signs agree",
        "all equal", int(len(set(np.sign(loo)))), ok4, True,
        f"full={means.mean():.10g}, loo={[float(f'{v:.6g}') for v in loo]}")
    return rows


def leave_one_out_sign(means: np.ndarray, counts: np.ndarray
                       ) -> tuple[bool, np.ndarray]:
    keep = np.isfinite(means) & (counts >= MIN_PAIRS)
    idx = np.flatnonzero(keep)
    loo = np.array([means[np.setdiff1d(idx, [i])].mean() for i in idx])
    full = float(means[keep].mean())
    signs = set(np.sign(loo).tolist()) | {float(np.sign(full))}
    return len(signs) == 1, loo


def preflight_p5(bands_new: pd.DataFrame, legacy: Path) -> dict:
    """P5: window mean reconstructed from the predecessor band table."""
    if not legacy.exists():
        return {"pass": None, "value": np.nan,
                "note": f"no predecessor band table at {legacy}"}
    old = pd.read_csv(legacy)
    old = old[(old.population == "A") & (old.band.isin(WINDOW_BANDS))]
    worst_m, worst_band = 0.0, 0.0
    for k in KS:
        o = old[old.k == k].sort_values("band")
        n = bands_new[bands_new.k == k].sort_values("band")
        if len(o) != len(n) or not np.array_equal(o.band.to_numpy(), n.band.to_numpy()):
            return {"pass": False, "value": np.nan,
                    "note": f"band grid mismatch at k={k}"}
        m_old = float((o.n_pair * o.H).sum() / o.n_pair.sum())
        m_new = float((n.n_pair * n.H).sum() / n.n_pair.sum())
        worst_m = max(worst_m, abs(m_new - m_old) / abs(m_old) if m_old else abs(m_new))
        scale = np.maximum(np.abs(o.H.to_numpy()), 1e-300)
        worst_band = max(worst_band,
                         float(np.max(np.abs(n.H.to_numpy() - o.H.to_numpy()) / scale)))
        if not np.array_equal(o.n_pair.to_numpy(), n.n_pair.to_numpy()):
            return {"pass": False, "value": np.nan,
                    "note": f"pair counts differ at k={k}"}
    return {"pass": bool(worst_m < P5_RTOL), "value": worst_m,
            "note": f"max relative band-H difference={worst_band:.3g}"}


# --------------------------------------------------------------------------
# verdict and driver (spec §4, §8, §9)
# --------------------------------------------------------------------------


ARMS = {
    "E": {"input": "results/ratchet_log_0819",
          "outdir": "results/ceiling_t0b_E_0828",
          "prefix": "EXPLORATORY_",
          "legacy": "results/ceiling_t0_0828/curves.csv",
          "kind": "exploratory (existing std seeds)"},
    "C": {"input": "results/ratchet_log_0829c",
          "outdir": "results/ceiling_t0b_C_0829",
          "prefix": "",
          "legacy": None,
          "kind": "pre-registered (new seeds)"},
}


def verdict_word(guard_ok: bool, baseline_ok: bool, lo: float, hi: float) -> str:
    if not guard_ok:
        return "INCONCLUSIVE_GUARD"
    if not baseline_ok:
        return "INVALID_BASELINE"
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "INCONCLUSIVE_GUARD"
    if lo > 0:
        return "H_POSITIVE"
    if hi < 0:
        return "H_NEGATIVE"
    return "NO_RESOLVED_WINDOW_H"


def resolve(root: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (root / p).resolve()


def suppress(out: Path, prefix: str, reason: str, note: str) -> None:
    pd.DataFrame([{"T0b_window": prefix + "SUPPRESSED", "reason": reason,
                   "note": note}]).to_csv(out / "verdict.csv", index=False)
    (out / "summary.md").write_text(
        f"# {out.name}\n\n判定は出していない。理由: **{reason}**\n\n{note}\n")


def fmt(v: float, digits: int = 8) -> str:
    return "nan" if v is None or not np.isfinite(v) else f"{v:+.{digits}g}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=tuple(ARMS), required=True)
    ap.add_argument("--input", default=None)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--centered", default="results/ratchet_centered_0822")
    ap.add_argument("--ci-method", choices=("percentile", "studentized"),
                    default=CI_METHOD)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    cfg = ARMS[args.arm]
    prefix = cfg["prefix"]
    inp = resolve(root, args.input or cfg["input"])
    out = resolve(root, args.outdir or cfg["outdir"])
    out.mkdir(parents=True, exist_ok=True)
    git_state = git_info(root)

    # ---- §6 preflight, before any real data is opened -------------------
    weights_pre = bootstrap_weights(len(SYN_SEED_PAIRS))
    pre_rows = preflight_pre_data(weights_pre, args.ci_method)
    pd.DataFrame(pre_rows).to_csv(out / "preflight.csv", index=False)
    bad = [r["id"] for r in pre_rows if r["gating"] and not r["pass"]]
    if bad:
        suppress(out, prefix, "PREFLIGHT_FAIL",
                 f"preflight.csv の {', '.join(bad)} が不合格。"
                 "spec §6 により実データへ進まない。")
        raise SystemExit(f"preflight failed: {bad} -> {out}")

    # ---- §7 assertions on the real logs ---------------------------------
    paths = sorted((inp / "logs").glob("seed*.npz"),
                   key=lambda p: int(p.stem.removeprefix("seed")))
    if not paths:
        suppress(out, prefix, "NO_INPUT", f"{inp}/logs に seed*.npz がない。")
        raise SystemExit(f"no logs under {inp}")
    evidence, all_v, all_b, all_checks, sources = [], [], [], [], []
    for path in paths:
        d = load_one(path)
        if int(d["period"]) != PERIOD or int(d["width"]) != WIDTH:
            raise RuntimeError(f"{path}: unexpected period/width")
        sources.append({"path": str(path.relative_to(root)), "sha256": sha256(path),
                        "seed": int(d["seed"]), "n_step": int(len(d["step"]))})
        sp = pairs_from_npz(d)
        v, b, chk = seed_summary(sp)
        evidence.append(seed_evidence(d, chk))
        all_v.extend(v)
        all_b.extend(b)
        all_checks.append(chk)
        del sp, d
    centered_dir = resolve(root, args.centered)
    centered = [mu_norm_stats(p) for p in
                sorted((centered_dir / "logs").glob("seed*.npz"),
                       key=lambda p: int(p.stem.removeprefix("seed")))]
    assertions = assertion_frame(evidence, centered, root)
    assertions.to_csv(out / "assertions.csv", index=False)
    if not bool(assertions["pass"].all()):
        failed = assertions[~assertions["pass"]].id.tolist()
        suppress(out, prefix, "ASSERTION_FAIL",
                 f"assertions.csv の {', '.join(failed)} が不合格。")
        raise SystemExit(f"assertions failed: {failed} -> {out}")

    # ---- estimates ------------------------------------------------------
    seed_variant = pd.DataFrame(all_v)
    seed_band = pd.DataFrame(all_b)
    seed_ids = [int(e["seed"]) for e in evidence]
    weights = bootstrap_weights(len(seed_ids))
    est = window_estimates(seed_variant, seed_ids, weights, args.ci_method)
    bands = band_table(seed_band, seed_ids)
    curve = curve_frame(est)
    decomposition = decomposition_frame(est)
    sensitivity = sensitivity_frame(est)
    isotonic = isotonic_table(seed_band, seed_ids, weights)

    seed_variant.to_csv(out / "seed_variant_means.csv", index=False)
    bands.to_csv(out / "bands.csv", index=False)
    curve.to_csv(out / "window_curve.csv", index=False)
    decomposition.to_csv(out / "decomposition.csv", index=False)
    sensitivity.to_csv(out / "sensitivity.csv", index=False)
    isotonic.to_csv(out / "isotonic.csv", index=False)

    # ---- post-data preflight rows ---------------------------------------
    means, counts = seed_series(seed_variant, seed_ids, "main", K_MAIN, "mean_H")
    ok4, loo = leave_one_out_sign(means, np.where(np.isfinite(means), MIN_PAIRS, 0))
    post = [{"id": "P4_real", "stage": "post_data",
             "check": "leave-one-seed-out sign stability on the analysed logs",
             "statistic": "signs agree", "threshold": "all equal",
             "value": int(len(set(np.sign(loo).tolist()))), "pass": bool(ok4),
             "gating": False,
             "note": "spec §6 P4 が記述格下げの判断に使う実データ版。"
                     f"loo={[float(f'{v:.6g}') for v in loo]}"}]
    if cfg["legacy"]:
        p5 = preflight_p5(bands, resolve(root, cfg["legacy"]))
        post.append({"id": "P5", "stage": "post_data",
                     "check": "predecessor band table consistency",
                     "statistic": "max relative window-mean difference",
                     "threshold": P5_RTOL, "value": p5["value"],
                     "pass": bool(p5["pass"]), "gating": True, "note": p5["note"]})
    else:
        post.append({"id": "P5", "stage": "post_data",
                     "check": "predecessor band table consistency",
                     "statistic": "max relative window-mean difference",
                     "threshold": P5_RTOL, "value": np.nan, "pass": None,
                     "gating": False,
                     "note": "SKIP: この腕には前身 curves.csv が存在しない"})
    pd.DataFrame(pre_rows + post).to_csv(out / "preflight.csv", index=False)
    bad_post = [r["id"] for r in post if r["gating"] and not r["pass"]]
    if bad_post:
        suppress(out, prefix, "PREFLIGHT_FAIL",
                 f"preflight.csv の {', '.join(bad_post)} が不合格。")
        raise SystemExit(f"post-data preflight failed: {bad_post} -> {out}")

    # ---- §4 verdict -----------------------------------------------------
    main40 = est[("main", K_MAIN, "H")]
    main1 = est[("main", K_BASE, "H")]
    guard_ok = main40["n_seed"] >= MIN_MAIN_SEED
    baseline_ok = bool(np.isfinite(main1["ci_lo"]) and np.isfinite(main1["ci_hi"])
                       and main1["ci_lo"] <= 0.0 <= main1["ci_hi"])
    word = prefix + verdict_word(guard_ok, baseline_ok, main40["ci_lo"], main40["ci_hi"])
    strict_off = {k: int(sum(c["strict_off_in_window"][k] for c in all_checks))
                  for k in KS}
    verdict = {
        "arm": args.arm, "arm_kind": cfg["kind"], "T0b_window": word,
        "M_40": main40["point"], "M_40_ci_lo": main40["ci_lo"],
        "M_40_ci_hi": main40["ci_hi"], "M_1": main1["point"],
        "M_1_ci_lo": main1["ci_lo"], "M_1_ci_hi": main1["ci_hi"],
        "baseline_gate": "PASS" if baseline_ok else "INVALID_BASELINE",
        "guard": "PASS" if guard_ok else "FAIL",
        "n_seed_contrib": main40["n_seed"], "n_seed_total": len(seed_ids),
        "n_pair_window": main40["n_pair"],
        "strict_off_in_window_k40": strict_off[K_MAIN],
        "ci_method": args.ci_method, "bootstrap_B": BOOT_N,
        "bootstrap_seed": BOOT_SEED, "window_lo": WINDOW_LO, "window_hi": WINDOW_HI,
        "spec_vault_commit": SPEC_VAULT_COMMIT,
    }
    pd.DataFrame([verdict]).to_csv(out / "verdict.csv", index=False)

    write_json(out / "config.json", {
        "analysis": out.name, "arm": args.arm, "spec_vault_commit": SPEC_VAULT_COMMIT,
        "spec_vault_amendment_commit": SPEC_VAULT_AMENDMENT,
        "input": str(inp.relative_to(root)), "k": list(KS), "k_main": K_MAIN,
        "window": [WINDOW_LO, WINDOW_HI], "band_width": BAND_W,
        "guard": {"n_seed": MIN_MAIN_SEED, "n_pair": MIN_PAIRS},
        "weighting": "seed-equal (pair-equal reported as sensitivity only)",
        "bootstrap_B": BOOT_N, "bootstrap_seed": BOOT_SEED,
        "ci_method": args.ci_method,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
    })
    write_json(out / "provenance.json", {
        "spec_vault_commit": SPEC_VAULT_COMMIT,
        "spec_vault_amendment_commit": SPEC_VAULT_AMENDMENT,
        "spec_path": "可塑性喪失/spec/天井T0b_spec_0828.md",
        "git": git_state, "arm": args.arm, "seeds": seed_ids,
        "input": str(inp.relative_to(root)), "sources": sources,
        "centered_reference": [c["path"] for c in centered],
        "python": sys.version, "platform": platform.platform(),
        "numpy": np.__version__, "pandas": pd.__version__,
    })

    d40 = decomposition[decomposition.k == K_MAIN].iloc[0]
    no_blip = sensitivity[(sensitivity.variant == "no_blip")
                          & (sensitivity.k == K_MAIN)].iloc[0]
    iso40 = isotonic[isotonic.k == K_MAIN].iloc[0]
    lines = [
        f"# {out.name}", "",
        f"腕{args.arm}（{cfg['kind']}）。凍結 spec: vault `{SPEC_VAULT_COMMIT}` "
        "`可塑性喪失/spec/天井T0b_spec_0828.md`。", "",
        "## 主判定", "",
        f"- preflight: **PASS**（`preflight.csv`）/ 構造アサーション B1–B10: "
        f"**PASS**（`assertions.csv`）",
        f"- ベースラインゲート `M(1)`: {fmt(main1['point'])} "
        f"[{fmt(main1['ci_lo'])}, {fmt(main1['ci_hi'])}] → "
        f"**{verdict['baseline_gate']}**",
        f"- ガード: 寄与 seed **{main40['n_seed']}/{len(seed_ids)}**、"
        f"窓内ペア {main40['n_pair']:,}",
        f"- **T0b_window: {word}**",
        f"- `M(40)`: {fmt(main40['point'])} "
        f"[{fmt(main40['ci_lo'])}, {fmt(main40['ci_hi'])}]"
        f"（seed クラスタ bootstrap B={BOOT_N}, seed={BOOT_SEED}, "
        f"{args.ci_method} CI）", "",
        "## 副次量（判定に使わない）", "",
        f"- 経路分解 k=40: `F_path-F_0` = {fmt(d40.T1)} "
        f"[{fmt(d40.T1_ci_lo)}, {fmt(d40.T1_ci_hi)}] / "
        f"`D-F_path` = {fmt(d40.T2)} [{fmt(d40.T2_ci_lo)}, {fmt(d40.T2_ci_hi)}]",
        f"- ブリップ寄与（帯 [0.6,0.7) 除外 − 窓全体）: {fmt(no_blip.delta_vs_main)}"
        f"（相対 {no_blip.rel_delta_vs_main:+.4f}）",
        f"- 窓内 strict off: k 別 {strict_off}",
        f"- CI: {args.ci_method}（spec §4 改訂 vault `{SPEC_VAULT_AMENDMENT}`）。"
        "§5-7 の等調零点のみ percentile（位置量・判定外）",
        f"- 等調零点 k=40（**記述であり判定していない**）: "
        f"`z̃_F` = {fmt(iso40.z_tilde_F, 6)}, `z̃_D` = {fmt(iso40.z_tilde_D, 6)}, "
        f"`g̃` = {fmt(iso40.g_tilde, 6)} "
        f"[{fmt(iso40.g_tilde_ci_lo, 6)}, {fmt(iso40.g_tilde_ci_hi, 6)}]、"
        f"同時存在率 {iso40.exist_rate:.4f}",
        "  - 交点座標は準 max 型であり、`F` のゼロ近傍を横切る脆さは isotonic では"
        "消えない。`g̃` の CI が 0 を外しても「分離が示された」とは書かない（spec §5-7）。",
        ("  - " + str(iso40.note) + "。すなわち上側窓の内側には下降零点が無い。"
         "前身の `z_F` = 0.6630 を支配していた帯 [0.6,0.7) の孤立した正は "
         "isotonic で吸収された（spec §5-7 が予告した挙動）。"
         if str(iso40.note) else "  - 窓内に下降零点が存在する。"),
        f"- k プロファイル・感度は `window_curve.csv` / `sensitivity.csv`。", "",
        "## 適用範囲（spec §10）", "",
        "condA・w100・T=10,000・batch=1・std の境界前窓、上側窓 `[+0.1,+0.9)`、"
        "同一開始コホートの k<=40 に限定する。",
        "力のゼロ点と変位のゼロ点の**位置**の主張、天井が "
        "`|F_self|/|F_rest|=1` であること、Δ1000 の符号逆転の説明、"
        "選抜と道のりの因果的二分、Q17 の `F_rest` 対称性、"
        "centered・他の w・T・batch・データセットへの一般化、真の介入効果は"
        "いずれも本結果からは言えない。",
    ]
    if prefix:
        lines += ["", "## 格", "",
                  "本腕は**探索**である（spec §8.1）。判定語の `EXPLORATORY_` 接頭辞は"
                  "この格を示す。論文の主張の根拠に単独で使わない。"]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"T0b {word} -> {out}")


if __name__ == "__main__":
    main()
