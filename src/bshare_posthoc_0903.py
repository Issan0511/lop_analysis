"""Post-hoc b-share until first submersion (registered = 0).

Vault: ``現象3主張v1_0902`` §1b (補題 1(b)) predicts that, until a unit first
submerges (``max_x z_i <= 0``, i.e. ``p_hat_i == 0`` on the 32-pattern support),
the bias route carries a fixed share of the descent that is set by the input
geometry alone::

    share_b := Δb / (Δ(w·µ') + Δb)  ≈  1 / (1 + ||µ'||²)  =  1 / (1 + (dose/4)²)

because every plain-SGD step moves ``w`` along the input itself
(``Δw = c·x'``, ``Δb = c``) and ``E[x'·µ'] = ||µ'||²``.  Oracle-dose arms
(``target_mu_norm`` fixed) give one number per dose; raw arms give
``1/(1 + k + 1.25)`` with ``k`` the number of active flip bits.

This module is NOT a preregistration: the 12.16 value (0.0976) was compared
with the C1 figures in chat on 2026-09-03 before any spec was written.  Every
emitted row carries ``registered = 0`` and the provenance records
``analysis_grade = posthoc_not_preregistered``.  The 9.33 arms are the only
cells not looked at before this script existed; they are still reported as
``registered = 0`` because no verdict rule was frozen in advance.

Inputs are the per-seed ``logs/*.npz`` of the parent runs.  Those logs are
not committed (``.gitignore``: raw trajectories stay local), so this script
must be run on the machine that holds them.  It is numpy-only so that it can
run without torch.

Usage (from the repository root)::

    python -m src.bshare_posthoc_0903 --source results/gate_dose_0830 \\
        --out results/gate_dose_0830/posthoc_bshare_0903
    python -m src.bshare_posthoc_0903 --source results/gate_dial_0902 \\
        --out results/gate_dial_0902/posthoc_bshare_0903
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
REGISTERED = 0

# Log layout written by ``gate_dose.write_arm_logs_gate`` /
# ``gate_dial_0902`` (layer 1 only).  ``M`` and ``B`` are normalised by
# ``denom`` (= sd of the centred preactivation), so ``w·µ' = M*denom`` and
# ``b = B*denom`` recover the raw values (``現象3主張v1_0902`` C1: 正規化を戻した生値).
KEY_M, KEY_B, KEY_DENOM, KEY_PHAT = ("layer1_M", "layer1_B", "layer1_denom",
                                     "layer1_p_hat")
KEY_ZBAR, KEY_ZMAX = "layer1_zbar", "layer1_zmax"
KEY_FLIP, KEY_STEP, KEY_TARGET = "flip_state", "step", "target_mu_norm"

# Sanity: (M+B)*denom must reproduce the logged zbar (float32 logs).
ZBAR_RTOL, ZBAR_ATOL = 1e-4, 1e-4
# Free (per-step random) input dimensions in condA: m - f = 20 - 15.
N_FREE_BITS = 5


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


def predicted_share_raw(k: np.ndarray) -> np.ndarray:
    """1 / (1 + k + 1.25) for a raw arm; ``k`` = active flip bits per record."""
    return 1.0 / (1.0 + np.asarray(k, dtype=np.float64) + 0.25 * N_FREE_BITS)


def _git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:  # pragma: no cover - provenance only
        return "unknown"


# ---------------------------------------------------------- per-seed core

class SanityError(RuntimeError):
    pass


def analyse_seed(z: dict, *, min_descent: float = 0.0) -> dict:
    """Per-unit descent until first submersion for one ``arm_seed*.npz``.

    Returns per-unit arrays (NaN where the unit is excluded) plus counts.
    A unit is *submerged* at the first record where ``p_hat == 0``
    (``zmax <= 0`` when the log carries ``zmax``; the two coincide on the
    exact 32-pattern support, and the script checks that they do).
    Excluded units: never submerged in the log, submerged at record 0,
    degenerate ``denom`` (NaN ``M``/``B``) at either endpoint, or total
    descent ``Δ(w·µ') + Δb >= -min_descent`` (no descent to apportion).
    """
    M, B, denom = (np.asarray(z[KEY_M], np.float64), np.asarray(z[KEY_B], np.float64),
                   np.asarray(z[KEY_DENOM], np.float64))
    p_hat = np.asarray(z[KEY_PHAT], np.float64)
    step = np.asarray(z[KEY_STEP], np.int64)
    n_rec, width = p_hat.shape
    if M.shape != (n_rec, width) or B.shape != (n_rec, width):
        raise SanityError("M/B/p_hat shape mismatch")

    wmu, b = M * denom, B * denom
    if KEY_ZBAR in z:
        zbar = np.asarray(z[KEY_ZBAR], np.float64)
        ok = np.isfinite(wmu) & np.isfinite(b)
        if not np.allclose((wmu + b)[ok], zbar[ok], rtol=ZBAR_RTOL, atol=ZBAR_ATOL):
            raise SanityError("(M+B)*denom does not reproduce layer1_zbar")

    submerged = p_hat <= 0.0
    if KEY_ZMAX in z:
        zmax = np.asarray(z[KEY_ZMAX], np.float64)
        if not np.array_equal(zmax <= 0.0, submerged):
            raise SanityError("zmax<=0 and p_hat==0 disagree")

    ever = submerged.any(axis=0)
    first = np.where(ever, submerged.argmax(axis=0), -1)

    d_wmu = np.full(width, np.nan)
    d_b = np.full(width, np.nan)
    t_sub = np.full(width, -1, np.int64)
    reason = np.full(width, "ok", dtype=object)
    for i in range(width):
        if first[i] < 0:
            reason[i] = "never_submerged"
            continue
        if first[i] == 0:
            reason[i] = "submerged_at_record0"
            continue
        j = int(first[i])
        vals = (wmu[0, i], wmu[j, i], b[0, i], b[j, i])
        if not np.all(np.isfinite(vals)):
            reason[i] = "degenerate_denom"
            continue
        dw, db = wmu[j, i] - wmu[0, i], b[j, i] - b[0, i]
        if dw + db >= -float(min_descent):
            reason[i] = "no_descent"
            continue
        d_wmu[i], d_b[i], t_sub[i] = dw, db, step[j]

    good = np.isfinite(d_wmu)
    share_unit = np.full(width, np.nan)
    share_unit[good] = d_b[good] / (d_wmu[good] + d_b[good])

    # Prediction for the same units/windows.
    target = float(z[KEY_TARGET]) if KEY_TARGET in z else float("nan")
    if np.isfinite(target):
        pred_unit = np.where(good, predicted_share_oracle(target), np.nan)
        pred_kind = "oracle"
    else:
        flip = np.asarray(z[KEY_FLIP], np.float64)   # (n_rec, f)
        k_rec = flip.sum(axis=1)
        pred_unit = np.full(width, np.nan)
        for i in np.flatnonzero(good):
            j = int(first[i])
            # average of the record-wise prediction over the descent window
            pred_unit[i] = float(np.mean(predicted_share_raw(k_rec[:j + 1])))
        pred_kind = "raw"

    counts = {r: int((reason == r).sum()) for r in
              ("ok", "never_submerged", "submerged_at_record0",
               "degenerate_denom", "no_descent")}
    return dict(d_wmu=d_wmu, d_b=d_b, t_sub=t_sub, share_unit=share_unit,
                pred_unit=pred_unit, pred_kind=pred_kind, counts=counts,
                width=width, n_records=n_rec, target_mu_norm=target)


def summarise_seed(res: dict) -> dict:
    g = np.isfinite(res["share_unit"])
    if not g.any():
        return dict(n_units=0, med_d_wmu=np.nan, med_d_b=np.nan,
                    share_of_medians=np.nan, med_share_unit=np.nan,
                    med_pred=np.nan, med_t_sub=np.nan)
    med_dw = float(np.median(res["d_wmu"][g]))
    med_db = float(np.median(res["d_b"][g]))
    return dict(
        n_units=int(g.sum()),
        med_d_wmu=med_dw, med_d_b=med_db,
        # C1's "分担" as ratio of unit medians (matches how C1 quotes Δ(w·µ)/Δb)
        share_of_medians=med_db / (med_dw + med_db),
        med_share_unit=float(np.median(res["share_unit"][g])),
        med_pred=float(np.median(res["pred_unit"][g])),
        med_t_sub=float(np.median(res["t_sub"][g])),
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


def run(source: Path, out: Path, *, min_descent: float, arms_filter: list[str] | None) -> dict:
    started = time.time()
    arms = _load_logs(source)
    if arms_filter:
        arms = {a: v for a, v in arms.items() if a in set(arms_filter)}
    unit_rows, seed_rows, arm_rows, inputs = [], [], [], {}
    for arm, paths in arms.items():
        per_seed = []
        for p in paths:
            with np.load(p, allow_pickle=False) as z:
                zd = {k: z[k] for k in z.files}
            seed = int(zd["seed"])
            res = analyse_seed(zd, min_descent=min_descent)
            summ = summarise_seed(res)
            per_seed.append(summ)
            inputs[str(p.relative_to(source))] = sha_file(p)
            seed_rows.append(dict(registered=REGISTERED, arm=arm, seed=seed,
                                  pred_kind=res["pred_kind"],
                                  target_mu_norm=res["target_mu_norm"],
                                  **summ, **{f"n_{k}": v for k, v in res["counts"].items()}))
            for i in range(res["width"]):
                if np.isfinite(res["share_unit"][i]):
                    unit_rows.append(dict(
                        registered=REGISTERED, arm=arm, seed=seed, unit=i,
                        t_sub=int(res["t_sub"][i]), d_wmu=res["d_wmu"][i],
                        d_b=res["d_b"][i], share_unit=res["share_unit"][i],
                        pred=res["pred_unit"][i]))
        def _med(key):
            v = np.array([s[key] for s in per_seed], np.float64)
            v = v[np.isfinite(v)]
            return (float(np.median(v)) if v.size else np.nan), int(v.size)
        som, n1 = _med("share_of_medians")
        msu, _ = _med("med_share_unit")
        mpr, _ = _med("med_pred")
        mdw, _ = _med("med_d_wmu")
        mdb, _ = _med("med_d_b")
        mts, _ = _med("med_t_sub")
        arm_rows.append(dict(
            registered=REGISTERED, arm=arm, n_seeds=len(per_seed),
            n_seeds_with_units=n1, pred_kind=res["pred_kind"],
            seedmed_share_of_medians=som, seedmed_med_share_unit=msu,
            seedmed_pred=mpr, diff_share_minus_pred=(som - mpr) if np.isfinite(som) else np.nan,
            seedmed_d_wmu=mdw, seedmed_d_b=mdb, seedmed_t_sub=mts,
            seed_share_of_medians=json.dumps([round(s["share_of_medians"], 4)
                                              if np.isfinite(s["share_of_medians"]) else None
                                              for s in per_seed])))

    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "bshare_by_arm.csv", arm_rows)
    write_csv(out / "bshare_by_seed.csv", seed_rows)
    if unit_rows:
        write_csv(out / "bshare_by_unit.csv", unit_rows)
    prov = dict(experiment=EXPERIMENT, analysis_grade=ANALYSIS_GRADE,
                registered=REGISTERED, source=str(source), out=str(out),
                min_descent=min_descent, submersion="p_hat==0 (== zmax<=0)",
                window="record 0 -> first submerged record", git_head=_git_head(),
                python=platform.python_version(), numpy=np.__version__,
                inputs_sha256=inputs, wall_seconds=round(time.time() - started, 2),
                vault="現象3主張v1_0902 §1b / §7 F",
                prediction="share_b = 1/(1+||mu'||^2) oracle; 1/(1+k+1.25) raw")
    (out / "provenance.json").write_text(json.dumps(prov, indent=2, ensure_ascii=False))
    lines = [f"# {EXPERIMENT} (registered = 0, {ANALYSIS_GRADE})", "",
             f"source: `{source}`  window: record 0 → first `p_hat == 0`", "",
             "| arm | n seeds | share (median-of-medians) | median unit share | predicted | diff | Δ(w·µ') | Δb | t_sub |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in arm_rows:
        f = lambda v: "nan" if not np.isfinite(v) else f"{v:.4f}"
        lines.append(f"| {r['arm']} | {r['n_seeds_with_units']}/{r['n_seeds']} | "
                     f"{f(r['seedmed_share_of_medians'])} | {f(r['seedmed_med_share_unit'])} | "
                     f"{f(r['seedmed_pred'])} | {f(r['diff_share_minus_pred'])} | "
                     f"{f(r['seedmed_d_wmu'])} | {f(r['seedmed_d_b'])} | {r['seedmed_t_sub']:.0f} |")
    lines += ["", "No verdict label: the prediction was not frozen before the 12.16 "
              "comparison; treat as descriptive (`registered=0`)."]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    return dict(arm_rows=arm_rows, seed_rows=seed_rows, provenance=prov)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", type=Path, default=Path("results/gate_dose_0830"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--min-descent", type=float, default=0.0)
    a = ap.parse_args(argv)
    out = a.out or (a.source / "posthoc_bshare_0903")
    res = run(a.source, out, min_descent=a.min_descent, arms_filter=a.arms)
    for r in res["arm_rows"]:
        print(f"{r['arm']:10s} share={r['seedmed_share_of_medians']:.4f} "
              f"pred={r['seedmed_pred']:.4f} diff={r['diff_share_minus_pred']:+.4f} "
              f"({r['n_seeds_with_units']}/{r['n_seeds']} seeds)")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
