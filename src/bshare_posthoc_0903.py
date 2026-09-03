"""Post-hoc b-share until first submersion (registered = 0).

Vault: ``現象3主張v1_0902`` §1b (補題 1(b)) predicts that, until a unit first
submerges (``max_x z_i <= 0``, i.e. ``p_hat_i == 0`` on the 32-pattern support),
the bias route carries a fixed share of the descent that is set by the input
geometry alone::

    share_b := Δb / (Δ(w·µ') + Δb)  ≈  1 / (1 + ||µ'||²)

because every plain-SGD step moves ``w`` along the input itself
(``Δw = c·x'``, ``Δb = c``) and ``E[x'·µ'] = ||µ'||²``.  Oracle-dose arms hold
``||µ'||`` at ``target_mu_norm`` for every ``k`` (3.041 at dose 12.16, 2.333 at
dose 9.33 — note ``target_mu_norm`` is what the logs carry and what this module
uses, and it differs from ``dose/4`` in the third decimal).  Raw ("off") arms
have ``||µ'||² = k + 1.25`` exactly, with ``k`` the number of active flip bits.

補題 1(b) is a *per-step* identity.  Cumulated over a window it becomes
``Σc_t / Σc_t(1 + ||µ'_t||²)``, i.e. the reciprocal of a ``|c_t|``-weighted mean
of ``||µ'||²`` — NOT the mean of the per-step shares.  This module therefore
predicts ``1/(1 + mean_t k_t + 1.25)`` for raw arms and also emits the
mean-of-reciprocals convention (``pred_meanrecip``) that the vault brackets as
the other reading.  Both are taken over the window's step *intervals*
(``k_rec[start:j]``, left endpoints), so ``pred_meanrecip`` is the v1 convention
on the v2 window, not the v1 number.  ``c_t`` is not recoverable from the logs,
so every window mean here is unweighted over the (uniformly spaced) records.

REGISTRATION.  This is not a preregistration: the 12.16 value (0.0976) was
compared with the C1 figures in chat on 2026-09-03 before any spec was written,
so every emitted row carries ``registered = 0``.  That label is about *this*
window/statistic only — a registered rule for the same functional form already
exists and PASSed (命題リスト Q19 C2, run ``mu_titration_0823``: slope of bias
share against ``1/(1+||µ||²)`` = +0.811 [+0.764, +0.856]).  That rule registers
the DIRECTION only (slope CI above 0 plus a per-seed Spearman floor); it does
not register numerical agreement, and it measures a different quantity (a
one-step expected-gradient magnitude ratio over a whole 1M run) at a ``||µ||``
that is a run median of a decaying transient, not an oracle-fixed dose.  Its
committed
``results/mu_titration_0823/dose_response.csv`` carries measured shares at both
of these ``||µ||`` (0.10137 [0.09642, 0.10744] at 3.0414; 0.13356 [0.12820,
0.13856] at 2.3333).  Compare against those rather than treating this as the
first look at the functional form.

WHAT THE MEASURED Δ(w·µ') IS NOT.  ``µ'`` is re-derived from the current
``flip_state`` at every probe, so the endpoint difference
``w_j·µ'_j − w_s·µ'_s`` contains ``w_s·(µ'_j − µ'_s)`` — pure reorientation of
the input mean at task boundaries, with no weight update and no ``Δb`` partner.
On these runs that term is ~2/5 of the measured weight route (measured on
bias-frozen units in a side script, not by this module).  The module therefore
also emits a rotation-free estimator (``*_ff``) that accumulates both routes
only over record intervals in which ``flip_state`` (and ``gamma``) do not
change, where ``µ'`` is provably constant.  The price is a change of estimand,
not just of noise: those intervals carry only part of the descent
(``med_frac_db_ff`` / ``med_frac_dwmu_ff`` report how much), and on the fast
arms much of what survives comes from the first probe interval, so ``share_ff``
moves more than ``share`` when ``--start-record`` changes.

Inputs are the per-seed ``logs/*.npz`` of the parent runs.  Those logs are not
committed (``.gitignore``: raw trajectories stay local), so this script must be
run on the machine that holds them.  It is numpy-only so that it can run
without torch.

Usage (from the repository root)::

    python -m src.bshare_posthoc_0903 --source results/gate_dose_0830
    python -m src.bshare_posthoc_0903 --source results/gate_dial_0902
    python -m src.bshare_posthoc_0903 --source results/gate_dose_0830 \\
        --start-record 1 --out results/gate_dose_0830/posthoc_bshare_0903_s1
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

EXPERIMENT = "bshare_posthoc_0903"
ANALYSIS_GRADE = "posthoc_not_preregistered"
ANALYSIS_VERSION = 3
REGISTERED = 0

# Log layout written by ``gate_dose.write_arm_logs_gate`` /
# ``gate_dial_0902`` (layer 1 only).  ``M`` and ``B`` are normalised by
# ``denom`` (= sd of the centred preactivation), so ``w·µ' = M*denom`` and
# ``b = B*denom`` recover the raw values (``現象3主張v1_0902`` C1: 正規化を戻した生値).
KEY_M, KEY_B, KEY_DENOM, KEY_PHAT = ("layer1_M", "layer1_B", "layer1_denom",
                                     "layer1_p_hat")
KEY_ZBAR, KEY_ZMAX = "layer1_zbar", "layer1_zmax"
KEY_FLIP, KEY_STEP, KEY_TARGET = "flip_state", "step", "target_mu_norm"
KEY_GAMMA = "gamma"

# Sanity: (M+B)*denom must reproduce the logged zbar (float32 logs).  On the
# real logs this passes through ``atol`` (max abs residual ~2e-6 against
# |zbar| up to ~12); ``rtol`` alone would not, because |zbar| crosses zero.
ZBAR_RTOL, ZBAR_ATOL = 1e-4, 1e-4
# Relative floor on |Δ(w·µ')_ff + Δb_ff| before a flip-free share is emitted.
FF_DEN_REL = 1e-6
# Free (per-step random) input dimensions in condA: m - f = 20 - 15.
N_FREE_BITS = 5
FREE_VAR_TERM = 0.25 * N_FREE_BITS          # ||µ'||² = k + 1.25 on raw arms

EXCLUSIONS = ("ok", "never_submerged", "submerged_at_start", "degenerate_denom",
              "no_descent")


# --------------------------------------------------------------- helpers

def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def predicted_share_oracle(target_mu_norm: float) -> float:
    """1 / (1 + ||µ'||²) for an oracle-dose arm (||µ'|| fixed for every k)."""
    return 1.0 / (1.0 + float(target_mu_norm) ** 2)


def predicted_share_raw(k) -> np.ndarray:
    """Per-step share 1 / (1 + k + 1.25) for a raw arm."""
    return 1.0 / (1.0 + np.asarray(k, dtype=np.float64) + FREE_VAR_TERM)


def predicted_share_raw_window(k_window) -> float:
    """Cumulative share over a window: reciprocal of the mean ||µ'||².

    This is what ``Σc_t / Σc_t(1 + ||µ'_t||²)`` reduces to when the per-step
    contributions ``c_t`` are exchangeable; it is the estimand-matched
    counterpart of the per-step ``predicted_share_raw``.
    """
    k = np.asarray(k_window, dtype=np.float64)
    if k.size == 0:
        return float("nan")
    return 1.0 / (1.0 + float(k.mean()) + FREE_VAR_TERM)


def _git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:  # pragma: no cover - provenance only
        return "unknown"


def _git_dirty() -> str:
    """Tracked files modified at run time — a bare HEAD is not enough to
    reproduce a run whose analysis code was still uncommitted."""
    try:
        out = subprocess.check_output(["git", "status", "--porcelain",
                                       "--untracked-files=no"], text=True)
        return out.strip() or "clean"
    except Exception:  # pragma: no cover - provenance only
        return "unknown"


def _ratio(num: float, den: float) -> float:
    """num/den, nan on an exactly cancelling or non-finite denominator."""
    if not (np.isfinite(num) and np.isfinite(den)) or den == 0.0:
        return float("nan")
    return float(num) / float(den)


def _median(values) -> float:
    v = np.asarray(values, np.float64)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else float("nan")


# ---------------------------------------------------------- per-seed core

class SanityError(RuntimeError):
    pass


def analyse_seed(z: dict, *, min_descent: float = 0.0,
                 start_record: int = 0) -> dict:
    """Per-unit descent until first submersion for one ``arm_seed*.npz``.

    Window: record ``start_record`` → the first later record with
    ``p_hat == 0``.  ``start_record = 0`` is step 0 (initialisation, before any
    update), so the default window includes the first probe interval, during
    which ``b`` leaves its initial value.  ``--start-record 1`` drops that
    interval; the two differ materially on these runs.

    Returns per-unit arrays (NaN where the unit is excluded) plus counts.
    Excluded units: never submerged after ``start_record``, already submerged at
    ``start_record``, degenerate ``denom`` (NaN ``M``/``B``) at either endpoint,
    or total descent ``Δ(w·µ') + Δb >= -min_descent`` (no descent to
    apportion).  ``d_wmu_all``/``d_b_all`` keep the endpoint differences for the
    ``no_descent`` units as well, so the caller can measure what that exclusion
    does.
    """
    M, B, denom = (np.asarray(z[KEY_M], np.float64), np.asarray(z[KEY_B], np.float64),
                   np.asarray(z[KEY_DENOM], np.float64))
    p_hat = np.asarray(z[KEY_PHAT], np.float64)
    step = np.asarray(z[KEY_STEP], np.int64)
    n_rec, width = p_hat.shape
    if M.shape != (n_rec, width) or B.shape != (n_rec, width):
        raise SanityError("M/B/p_hat shape mismatch")
    if not 0 <= start_record < n_rec - 1:
        raise SanityError(f"start_record {start_record} outside 0..{n_rec - 2}")

    wmu, b = M * denom, B * denom
    if KEY_ZBAR in z:
        zbar = np.asarray(z[KEY_ZBAR], np.float64)
        ok = np.isfinite(wmu) & np.isfinite(b)
        if not np.allclose((wmu + b)[ok], zbar[ok], rtol=ZBAR_RTOL, atol=ZBAR_ATOL):
            raise SanityError("(M+B)*denom does not reproduce layer1_zbar")

    submerged = p_hat <= 0.0
    zmax_checked = False
    if KEY_ZMAX in z:                       # not written by these runs
        zmax = np.asarray(z[KEY_ZMAX], np.float64)
        if not np.array_equal(zmax <= 0.0, submerged):
            raise SanityError("zmax<=0 and p_hat==0 disagree")
        zmax_checked = True

    after = submerged[start_record + 1:]
    ever = after.any(axis=0)
    first = np.where(ever, after.argmax(axis=0) + start_record + 1, -1)

    # Intervals in which µ' is provably constant (no task flip, no re-solved
    # oracle offset), so that Δ(w·µ') over them carries no reorientation term.
    flip = np.asarray(z[KEY_FLIP], np.float64) if KEY_FLIP in z else None
    ff_ok = None
    if flip is not None:
        ff_ok = np.all(flip[1:] == flip[:-1], axis=1)
        if KEY_GAMMA in z:
            g = np.asarray(z[KEY_GAMMA], np.float64)
            if g.shape == (n_rec,):
                # gamma is all-NaN on the raw ("off") arms, where there is no
                # oracle offset to re-solve; NaN == NaN must read as "unchanged"
                # or the mask would reject every interval.
                nan_pair = np.isnan(g[1:]) & np.isnan(g[:-1])
                ff_ok &= (g[1:] == g[:-1]) | nan_pair

    nan = np.nan
    d_wmu = np.full(width, nan)
    d_b = np.full(width, nan)
    d_wmu_all = np.full(width, nan)
    d_b_all = np.full(width, nan)
    d_wmu_ff = np.full(width, nan)
    d_b_ff = np.full(width, nan)
    n_ff = np.zeros(width, np.int64)
    t_sub = np.full(width, -1, np.int64)
    win_len = np.full(width, -1, np.int64)
    reason = np.full(width, "ok", dtype=object)

    for i in range(width):
        # order matters: a unit already down at ``start_record`` that resurfaces
        # and never re-submerges is "submerged_at_start", not "never_submerged"
        if submerged[start_record, i]:
            reason[i] = "submerged_at_start"
            continue
        if first[i] < 0:
            reason[i] = "never_submerged"
            continue
        j = int(first[i])
        vals = (wmu[start_record, i], wmu[j, i], b[start_record, i], b[j, i])
        if not np.all(np.isfinite(vals)):
            reason[i] = "degenerate_denom"
            continue
        dw = wmu[j, i] - wmu[start_record, i]
        db = b[j, i] - b[start_record, i]
        d_wmu_all[i], d_b_all[i] = dw, db
        t_sub[i], win_len[i] = step[j], j - start_record
        if ff_ok is not None:
            m = ff_ok[start_record:j]
            n_ff[i] = int(m.sum())
            if n_ff[i]:
                dwv = np.diff(wmu[start_record:j + 1, i])
                dbv = np.diff(b[start_record:j + 1, i])
                d_wmu_ff[i] = float(np.where(m, dwv, 0.0).sum())
                d_b_ff[i] = float(np.where(m, dbv, 0.0).sum())
        if dw + db >= -float(min_descent):
            reason[i] = "no_descent"
            continue
        d_wmu[i], d_b[i] = dw, db

    good = np.isfinite(d_wmu)
    share_unit = np.full(width, nan)
    share_unit[good] = d_b[good] / (d_wmu[good] + d_b[good])
    # A "share" only means anything when both routes descend; 7% of units on
    # these runs have one route climbing, which puts the ratio outside [0,1].
    same_sign = np.zeros(width, bool)
    same_sign[good] = (d_wmu[good] < 0) & (d_b[good] < 0)

    share_ff = np.full(width, nan)
    ff_good = good & np.isfinite(d_wmu_ff) & np.isfinite(d_b_ff)
    den_ff = d_wmu_ff + d_b_ff
    # relative floor: an exact-zero test lets |den| ~ 1e-12 through and puts a
    # 1e12-sized "share" into the per-unit CSV
    scale = np.abs(d_wmu_ff) + np.abs(d_b_ff)
    with np.errstate(invalid="ignore"):
        big_enough = np.abs(den_ff) >= FF_DEN_REL * scale
    ff_dropped = int((ff_good & ~np.where(np.isfinite(den_ff), big_enough, False)).sum())
    ff_good &= np.where(np.isfinite(den_ff), big_enough, False)
    share_ff[ff_good] = d_b_ff[ff_good] / den_ff[ff_good]
    # What fraction of the descent survives the flip-free mask?  The mask keeps
    # ~90% of the intervals but far less of the movement, because the boundary
    # intervals it drops are where most of the descent happens.
    frac_db_ff = np.full(width, nan)
    frac_dwmu_ff = np.full(width, nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac_db_ff[ff_good] = d_b_ff[ff_good] / d_b[ff_good]
        frac_dwmu_ff[ff_good] = d_wmu_ff[ff_good] / d_wmu[ff_good]

    # ------------------------------------------------ prediction per unit
    target = float(z[KEY_TARGET]) if KEY_TARGET in z else nan
    pred_unit = np.full(width, nan)
    pred_meanrecip = np.full(width, nan)
    pred_ff = np.full(width, nan)
    if np.isfinite(target):
        pred_kind = "oracle"
        p = predicted_share_oracle(target)
        pred_unit[good] = p
        pred_meanrecip[good] = p
        pred_ff[ff_good] = p
    else:
        pred_kind = "raw"
        if flip is None:
            raise SanityError("raw arm without flip_state")
        k_rec = flip.sum(axis=1)
        for i in np.flatnonzero(good):
            j = int(first[i])
            k_int = k_rec[start_record:j]      # k in force during each interval
            pred_unit[i] = predicted_share_raw_window(k_int)
            pred_meanrecip[i] = float(np.mean(predicted_share_raw(k_int)))
            if ff_good[i]:
                m = ff_ok[start_record:j]
                pred_ff[i] = predicted_share_raw_window(k_int[m])

    counts = {r: int((reason == r).sum()) for r in EXCLUSIONS}
    return dict(d_wmu=d_wmu, d_b=d_b, d_wmu_all=d_wmu_all, d_b_all=d_b_all,
                d_wmu_ff=d_wmu_ff, d_b_ff=d_b_ff, n_ff=n_ff,
                frac_db_ff=frac_db_ff, frac_dwmu_ff=frac_dwmu_ff,
                n_ff_dropped=ff_dropped,
                t_sub=t_sub, win_len=win_len, share_unit=share_unit,
                share_ff=share_ff, same_sign=same_sign, pred_unit=pred_unit,
                pred_meanrecip=pred_meanrecip, pred_ff=pred_ff,
                pred_kind=pred_kind, counts=counts, width=width,
                n_records=n_rec, target_mu_norm=target,
                start_record=start_record, zmax_checked=zmax_checked)


def summarise_seed(res: dict) -> dict:
    g = np.isfinite(res["share_unit"])
    empty = dict(n_units=0, med_d_wmu=np.nan, med_d_b=np.nan,
                 share_of_medians=np.nan, med_share_unit=np.nan,
                 share_of_medians_ff=np.nan, med_share_unit_ff=np.nan,
                 share_of_medians_incl_nodesc=np.nan, med_pred=np.nan,
                 med_pred_meanrecip=np.nan, med_pred_ff=np.nan,
                 med_t_sub=np.nan, med_win_len=np.nan,
                 frac_routes_same_sign=np.nan, frac_share_outside01=np.nan,
                 n_units_ff=0, med_frac_db_ff=np.nan, med_frac_dwmu_ff=np.nan,
                 frac_share_ff_outside01=np.nan)
    if not g.any():
        return empty
    med_dw, med_db = _median(res["d_wmu"][g]), _median(res["d_b"][g])
    gf = np.isfinite(res["share_ff"])
    ga = np.isfinite(res["d_wmu_all"])
    dw_a, db_a = _median(res["d_wmu_all"][ga]), _median(res["d_b_all"][ga])
    s = res["share_unit"][g]
    sff = res["share_ff"][gf]
    return dict(
        n_units=int(g.sum()),
        med_d_wmu=med_dw, med_d_b=med_db,
        # C1's "分担" as ratio of unit medians (matches how C1 quotes Δ(w·µ)/Δb)
        share_of_medians=_ratio(med_db, med_dw + med_db),
        med_share_unit=_median(s),
        # rotation-free: both routes accumulated only over flip-free intervals
        share_of_medians_ff=(_ratio(_median(res["d_b_ff"][gf]),
                                    _median(res["d_wmu_ff"][gf])
                                    + _median(res["d_b_ff"][gf]))
                             if gf.any() else np.nan),
        med_share_unit_ff=_median(sff) if gf.any() else np.nan,
        # how much of the descent survives the mask (the mask keeps ~90% of the
        # intervals but far less of the movement)
        med_frac_db_ff=_median(res["frac_db_ff"][gf]) if gf.any() else np.nan,
        med_frac_dwmu_ff=_median(res["frac_dwmu_ff"][gf]) if gf.any() else np.nan,
        frac_share_ff_outside01=(float(((sff < 0) | (sff > 1)).mean())
                                 if gf.any() else np.nan),
        # sensitivity: put the no_descent units back (selection on the very
        # denominator of the share)
        share_of_medians_incl_nodesc=_ratio(db_a, dw_a + db_a),
        med_pred=_median(res["pred_unit"][g]),
        med_pred_meanrecip=_median(res["pred_meanrecip"][g]),
        med_pred_ff=_median(res["pred_ff"][gf]) if gf.any() else np.nan,
        med_t_sub=_median(res["t_sub"][g]),
        med_win_len=_median(res["win_len"][g]),
        frac_routes_same_sign=float(res["same_sign"][g].mean()),
        frac_share_outside01=float(((s < 0) | (s > 1)).mean()),
        n_units_ff=int(gf.sum()),
    )


# ------------------------------------------------------------------ main

def _load_logs(source: Path) -> dict[str, list[Path]]:
    logdir = source / "logs"
    if not logdir.is_dir():
        raise SanityError(f"no logs directory under {source} (raw logs are "
                          "not committed; run on the machine that holds them)")
    arms: dict[str, list[Path]] = {}
    for p in sorted(logdir.glob("*_seed*.npz")):
        arm = p.name.rsplit("_seed", 1)[0]
        arms.setdefault(arm, []).append(p)
    if not arms:
        raise SanityError(f"no *_seed*.npz under {logdir}")
    return arms


SEED_STATS = ("n_units", "n_units_ff", "med_d_wmu", "med_d_b",
              "share_of_medians", "med_share_unit", "share_of_medians_ff",
              "med_share_unit_ff", "share_of_medians_incl_nodesc", "med_pred",
              "med_pred_meanrecip", "med_pred_ff", "med_t_sub", "med_win_len",
              "frac_routes_same_sign", "frac_share_outside01",
              "med_frac_db_ff", "med_frac_dwmu_ff", "frac_share_ff_outside01")


def run(source: Path, out: Path, *, min_descent: float,
        arms_filter: list[str] | None, start_record: int = 0) -> dict:
    started = time.time()
    arms = _load_logs(source)
    if arms_filter:
        arms = {a: v for a, v in arms.items() if a in set(arms_filter)}
        if not arms:
            raise SanityError(f"--arms matched nothing under {source}/logs")
    unit_rows, seed_rows, arm_rows, inputs = [], [], [], {}
    n_zmax_checked, n_seeds_total, n_ff_dropped_total = 0, 0, 0
    for arm, paths in sorted(arms.items()):
        per_seed, pred_kinds, counts_tot = [], set(), {r: 0 for r in EXCLUSIONS}
        for p in paths:
            with np.load(p, allow_pickle=False) as z:
                zd = {k: z[k] for k in z.files}
            seed = int(zd["seed"])
            res = analyse_seed(zd, min_descent=min_descent,
                               start_record=start_record)
            summ = summarise_seed(res)
            per_seed.append(summ)
            pred_kinds.add(res["pred_kind"])
            n_seeds_total += 1
            n_zmax_checked += int(res["zmax_checked"])
            n_ff_dropped_total += int(res["n_ff_dropped"])
            for r, n in res["counts"].items():
                counts_tot[r] += n
            inputs[str(p.relative_to(source))] = sha_file(p)
            seed_rows.append(dict(registered=REGISTERED, arm=arm, seed=seed,
                                  pred_kind=res["pred_kind"],
                                  target_mu_norm=res["target_mu_norm"],
                                  **summ,
                                  **{f"n_{k}": v for k, v in res["counts"].items()}))
            for i in range(res["width"]):
                if np.isfinite(res["share_unit"][i]):
                    unit_rows.append(dict(
                        registered=REGISTERED, arm=arm, seed=seed, unit=i,
                        t_sub=int(res["t_sub"][i]), win_len=int(res["win_len"][i]),
                        d_wmu=res["d_wmu"][i], d_b=res["d_b"][i],
                        share_unit=res["share_unit"][i],
                        routes_same_sign=int(res["same_sign"][i]),
                        share_in_unit_interval=int(0.0 <= res["share_unit"][i] <= 1.0),
                        d_wmu_ff=res["d_wmu_ff"][i], d_b_ff=res["d_b_ff"][i],
                        n_ff_intervals=int(res["n_ff"][i]),
                        share_unit_ff=res["share_ff"][i],
                        frac_db_ff=res["frac_db_ff"][i],
                        frac_dwmu_ff=res["frac_dwmu_ff"][i],
                        pred=res["pred_unit"][i],
                        pred_meanrecip=res["pred_meanrecip"][i],
                        pred_ff=res["pred_ff"][i]))
        if len(pred_kinds) != 1:
            raise SanityError(f"arm {arm} mixes prediction kinds: {pred_kinds}")
        agg = {f"seedmed_{k}": _median([s[k] for s in per_seed]) for k in SEED_STATS}
        n_with = int(np.isfinite([s["share_of_medians"] for s in per_seed]).sum())
        arm_rows.append(dict(
            registered=REGISTERED, arm=arm, n_seeds=len(per_seed),
            n_seeds_with_units=n_with, pred_kind=pred_kinds.pop(),
            start_record=start_record, **agg,
            diff_share_minus_pred=agg["seedmed_share_of_medians"] - agg["seedmed_med_pred"],
            diff_medshare_minus_pred=agg["seedmed_med_share_unit"] - agg["seedmed_med_pred"],
            diff_ff_minus_pred=agg["seedmed_share_of_medians_ff"] - agg["seedmed_med_pred_ff"],
            **{f"n_{k}_total": v for k, v in counts_tot.items()},
            seed_share_of_medians=json.dumps(
                [round(s["share_of_medians"], 4)
                 if np.isfinite(s["share_of_medians"]) else None for s in per_seed])))

    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "bshare_by_arm.csv", arm_rows)
    write_csv(out / "bshare_by_seed.csv", seed_rows)
    if unit_rows:
        write_csv(out / "bshare_by_unit.csv", unit_rows)
    prov = dict(
        experiment=EXPERIMENT, analysis_version=ANALYSIS_VERSION,
        analysis_grade=ANALYSIS_GRADE, registered=REGISTERED,
        source=str(source), out=str(out), min_descent=min_descent,
        start_record=start_record, arms_filter=arms_filter,
        submersion="p_hat == 0 on the exact 32-pattern support",
        zmax_cross_check=(f"layer1_zmax present and checked on "
                          f"{n_zmax_checked}/{n_seeds_total} seeds"
                          if n_zmax_checked else
                          f"layer1_zmax absent from all {n_seeds_total} seeds; "
                          "p_hat==0 used alone"),
        ff_units_dropped_by_denominator_floor=n_ff_dropped_total,
        window=f"record {start_record} -> first later record with p_hat == 0",
        weight_route=("endpoint difference w_j.mu'_j - w_s.mu'_s; contains the "
                      "mu'-reorientation term w_s.(mu'_j - mu'_s). The *_ff "
                      "columns accumulate both routes only over intervals with "
                      "flip_state and gamma unchanged, where mu' is constant."),
        prediction=("oracle: 1/(1+target_mu_norm^2); raw: 1/(1+mean_t k_t+1.25) "
                    "over the window's step intervals (estimand-matched). "
                    "pred_meanrecip is the mean-of-reciprocals convention."),
        prior_registered_rule=("命題リスト Q19 C2 (run mu_titration_0823) is a "
                               "REGISTERED, PASSed rule for the same functional "
                               "form; results/mu_titration_0823/dose_response.csv "
                               "carries bias_share 0.10137 [0.09642,0.10744] at "
                               "mu_norm 3.0414 and 0.13356 [0.12820,0.13856] at "
                               "2.3333. registered=0 applies to THIS window/statistic."),
        git_head=_git_head(), git_dirty=_git_dirty(),
        module_sha256=sha_file(Path(__file__)),
        python=platform.python_version(),
        numpy=np.__version__, inputs_sha256=inputs,
        wall_seconds=round(time.time() - started, 2),
        vault="現象3主張v1_0902 §1b / §7 F")
    (out / "provenance.json").write_text(json.dumps(prov, indent=2, ensure_ascii=False))
    (out / "summary.md").write_text(_summary_md(source, start_record, arm_rows))
    return dict(arm_rows=arm_rows, seed_rows=seed_rows, provenance=prov)


def _f(v, nd=4) -> str:
    return "nan" if not np.isfinite(v) else f"{v:.{nd}f}"


def _summary_md(source: Path, start_record: int, arm_rows: list[dict]) -> str:
    lines = [f"# {EXPERIMENT} v{ANALYSIS_VERSION} (registered = 0, {ANALYSIS_GRADE})", "",
             f"source: `{source}`  window: record {start_record} → first later "
             "record with `p_hat == 0`", "",
             "`share` = median(Δb)/(median(Δ(w·µ'))+median(Δb)) over units, then "
             "median over seeds (C1's 取り方). `unit share` = median of the "
             "per-unit ratios. `share_ff` = the same ratio of medians with both "
             "routes accumulated only over flip-free intervals (µ' constant, so "
             "no reorientation term).", "",
             "| arm | seeds | share | diff | unit share | diff | share_ff | diff_ff | predicted | Δ(w·µ') | Δb | t_sub |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in arm_rows:
        lines.append(
            f"| {r['arm']} | {r['n_seeds_with_units']}/{r['n_seeds']} | "
            f"{_f(r['seedmed_share_of_medians'])} | {_f(r['diff_share_minus_pred'])} | "
            f"{_f(r['seedmed_med_share_unit'])} | {_f(r['diff_medshare_minus_pred'])} | "
            f"{_f(r['seedmed_share_of_medians_ff'])} | {_f(r['diff_ff_minus_pred'])} | "
            f"{_f(r['seedmed_med_pred'])} | {_f(r['seedmed_med_d_wmu'])} | "
            f"{_f(r['seedmed_med_d_b'])} | {r['seedmed_med_t_sub']:.0f} |")
    lines += ["", "## diagnostics (what the headline number hides)", "",
              "| arm | units/seed | submerged at start | no_descent | share incl. no_descent | routes same sign | share outside [0,1] | ff share outside [0,1] | ff keeps Δb | ff keeps Δ(w·µ') | pred (mean-of-recip) | window (records) |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in arm_rows:
        tot = sum(r[f"n_{k}_total"] for k in EXCLUSIONS)
        lines.append(
            f"| {r['arm']} | {r['seedmed_n_units']:.0f} | "
            f"{r['n_submerged_at_start_total']}/{tot} | {r['n_no_descent_total']}/{tot} | "
            f"{_f(r['seedmed_share_of_medians_incl_nodesc'])} | "
            f"{_f(r['seedmed_frac_routes_same_sign'], 3)} | "
            f"{_f(r['seedmed_frac_share_outside01'], 3)} | "
            f"{_f(r['seedmed_frac_share_ff_outside01'], 3)} | "
            f"{_f(r['seedmed_med_frac_db_ff'], 3)} | "
            f"{_f(r['seedmed_med_frac_dwmu_ff'], 3)} | "
            f"{_f(r['seedmed_med_pred_meanrecip'])} | {r['seedmed_med_win_len']:.0f} |")
    lines += ["",
              "No verdict label: the prediction was not frozen before the 12.16 "
              "comparison; treat as descriptive (`registered = 0`). A registered "
              "PASSed rule for the same functional form exists (命題リスト Q19 C2, "
              "`results/mu_titration_0823/dose_response.csv`: 0.10137 "
              "[0.09642, 0.10744] at ‖µ‖ = 3.0414 and 0.13356 [0.12820, 0.13856] "
              "at 2.3333) — compare against that, not against nothing.", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", type=Path, default=Path("results/gate_dose_0830"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--min-descent", type=float, default=0.0)
    ap.add_argument("--start-record", type=int, default=0,
                    help="first record of the descent window (0 = step 0)")
    a = ap.parse_args(argv)
    out = a.out or (a.source / "posthoc_bshare_0903")
    res = run(a.source, out, min_descent=a.min_descent, arms_filter=a.arms,
              start_record=a.start_record)
    for r in res["arm_rows"]:
        print(f"{r['arm']:16s} share={_f(r['seedmed_share_of_medians'])} "
              f"unit={_f(r['seedmed_med_share_unit'])} "
              f"ff={_f(r['seedmed_share_of_medians_ff'])} "
              f"pred={_f(r['seedmed_med_pred'])} "
              f"diff={r['diff_share_minus_pred']:+.4f} "
              f"({r['n_seeds_with_units']}/{r['n_seeds']} seeds)")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
