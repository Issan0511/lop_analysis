"""Report-only analyses frozen for ``fb_width_seat_0915``.

This module deliberately contains no verdict logic.  Its public entry point is
``compute_report(logs_by_arm)`` where values are sequences of loaded npz-like
mappings.  ``report_from_directory`` is the filesystem adapter used after the
registered run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

SEED = 20260915
TASK_STEPS = np.arange(451, 501, dtype=np.int64) * 10_000
ARMS = (
    "LRoff0_1216", "LRwf21_1216", "FB21LRoff0_1216",
    "FB21LRwf21_1216", "LRwi21_1216", "FB21LRwi21_1216",
)
FIXED_PAIR = ("LRwf21_1216", "FB21LRwf21_1216")
FREE_PAIR = ("LRoff0_1216", "FB21LRoff0_1216")
DIAGNOSTIC_PAIR = ("LRwi21_1216", "FB21LRwi21_1216")


def _at_steps(values, steps, wanted, key):
    steps = np.asarray(steps, dtype=np.int64)
    pos = {int(s): i for i, s in enumerate(steps)}
    missing = [int(s) for s in wanted if int(s) not in pos]
    if missing:
        raise ValueError(f"{key}: missing steps {missing[:4]}")
    return np.asarray(values)[[pos[int(s)] for s in wanted]]


def _mean_ci(x, rng, n_boot=2000):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"estimate": None, "ci": [None, None], "n": 0}
    draws = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(1)
    return {"estimate": float(x.mean()),
            "ci": [float(v) for v in np.quantile(draws, [.025, .975])],
            "n": int(len(x))}


def _seed_window(z, key, start, stop, *, reverse=False, alive=False):
    wanted = np.arange(start, stop + 1, dtype=np.int64) * 10_000
    x = _at_steps(z[key], z["step"], wanted, key).astype(np.float64)
    if x.ndim == 1:
        return float(x.mean())
    if alive:
        den = _at_steps(z["layer1_denom"], z["step"], wanted,
                        "layer1_denom").astype(np.float64)
        x = np.where(den > .25, x, np.nan)
    # Registered order: unit median at each task, then task mean.  The
    # sensitivity reverses those two operations.
    with np.errstate(all="ignore"):
        ans = (np.nanmedian(np.nanmean(x, axis=0)) if reverse
               else np.nanmean(np.nanmedian(x, axis=1)))
    return float(ans)


def task_internal_displacement(z: Mapping[str, np.ndarray]):
    """Return the 49 registered t451..500 boundary transitions.

    For transition j -> j+1, the audit-v1b quantity is
    ``dzbar - kick - offset`` with weights at the left boundary,
    ``kick = dflip @ W_flip`` and
    ``offset = -dgamma/2 * sum(W_raw)``.  W_raw is reconstructed by joining
    the logged flip and free input weights.  k_on is also the left-boundary
    value, so grouping cannot see the destination state.
    """
    bs = np.asarray(z["layer1_branch_step"], dtype=np.int64)
    fs = np.asarray(z["layer1_w_free_step"], dtype=np.int64)
    wflip = _at_steps(z["layer1_w_flip"], bs, TASK_STEPS,
                      "layer1_w_flip").astype(np.float64)
    wfree = _at_steps(z["layer1_w_free"], fs, TASK_STEPS,
                      "layer1_w_free").astype(np.float64)
    kon = _at_steps(z["layer1_k_on"], bs, TASK_STEPS,
                    "layer1_k_on").astype(np.int64)
    zbar = _at_steps(z["layer1_zbar"], z["step"], TASK_STEPS,
                     "layer1_zbar").astype(np.float64)
    flip = _at_steps(z["flip_state"], z["step"], TASK_STEPS,
                     "flip_state").astype(np.float64)
    gamma = _at_steps(z["gamma"], z["step"], TASK_STEPS,
                      "gamma").astype(np.float64).reshape(50)
    dflip = np.diff(flip, axis=0)
    dgamma = np.diff(gamma)
    kick = np.einsum("tf,thf->th", dflip, wflip[:-1])
    raw_sum = wflip[:-1].sum(-1) + wfree[:-1].sum(-1)
    offset = -.5 * dgamma[:, None] * raw_sum
    delta = np.diff(zbar, axis=0)
    return {"task_from": np.arange(451, 500, dtype=np.int64),
            "k_on": kon[:-1], "delta_zbar": delta, "kick": kick,
            "offset": offset, "delta_prime": delta - kick - offset}


def _group_k_on(logs):
    per_seed = []
    pooled = {k: [] for k in range(33)}
    for z in logs:
        d = task_internal_displacement(z)
        row = {}
        for k in range(33):
            x = d["delta_prime"][d["k_on"] == k]
            row[k] = float(x.mean()) if x.size else np.nan
            if x.size:
                pooled[k].append(x.ravel())
        per_seed.append(row)
    rng = np.random.default_rng(SEED)
    out = {}
    for k in range(33):
        seed_values = [r[k] for r in per_seed]
        p = np.concatenate(pooled[k]) if pooled[k] else np.empty(0)
        out[str(k)] = {**_mean_ci(seed_values, rng), "n_unit_intervals": int(p.size),
                       "pooled_mean": (float(p.mean()) if p.size else None)}
    return out


def _trajectory(logs, key, steps_key="step", transform=None):
    curves = []
    for z in logs:
        x = _at_steps(z[key], z[steps_key], TASK_STEPS, key).astype(np.float64)
        if transform is not None:
            x = transform(x)
        curves.append(np.nanmedian(x, axis=1) if x.ndim > 1 else x)
    if not curves:
        return {"tasks": list(range(451, 501)), "seed_curves": [],
                "seed_mean": [None] * 50}
    a = np.asarray(curves)
    return {"tasks": list(range(451, 501)), "seed_curves": a.tolist(),
            "seed_mean": np.nanmean(a, axis=0).tolist()}


def _arm_report(logs):
    rng = np.random.default_rng(SEED)
    neg, unfit, dead = [], [], []
    nb = np.zeros(33, dtype=np.int64)
    kappa_t20, kappa_late = [], []
    shape_signed, shape_abs_typical, shape_abs_mean, shape_abs_max = [], [], [], []
    for z in logs:
        zm = _at_steps(z["layer1_zmax"], z["step"], TASK_STEPS,
                       "layer1_zmax")
        neg.append(float((zm < 0).mean()))
        unfit.append(_seed_window(z, "unfit", 451, 500))
        dead.append(_seed_window(z, "layer1_strict_dead", 451, 500) /
                    float(z["layer1_zmax"].shape[1]))
        nband = _at_steps(z["layer1_n_band"], z["layer1_branch_step"],
                          TASK_STEPS, "layer1_n_band").astype(np.int64)
        nb += np.bincount(nband.ravel(), minlength=33)[:33]
        W = np.asarray(z["layer1_w_free"], dtype=np.float64)
        ws = np.asarray(z["layer1_w_free_step"], dtype=np.int64)
        w20 = _at_steps(W, ws, [200_000], "layer1_w_free")[0]
        wl = _at_steps(W, ws, TASK_STEPS, "layer1_w_free")
        n20 = np.linalg.norm(w20, axis=-1)
        def kap(a):
            n = np.linalg.norm(a, axis=-1)
            return np.divide(np.abs(a).sum(-1), n,
                             out=np.full_like(n, np.nan), where=n != 0)
        k20, kt = kap(w20), kap(wl)
        kappa_t20.append(float(np.nanmedian(k20)))
        kappa_late.append(float(np.nanmean(np.nanmedian(kt, axis=1))))
        contribution = .5 * (kt - k20[None, :]) * n20[None, :]
        # The registered aggregate follows the main task->unit order.  Absolute
        # summaries describe typical size; max is the literal per-unit bound.
        shape_signed.append(float(np.nanmean(np.nanmedian(contribution, axis=1))))
        shape_abs_typical.append(float(np.nanmean(np.nanmedian(np.abs(contribution), axis=1))))
        shape_abs_mean.append(float(np.nanmean(np.abs(contribution))))
        shape_abs_max.append(float(np.nanmax(np.abs(contribution))))
    total = int(nb.sum())
    return {
        "all_negative_fraction": _mean_ci(neg, rng),
        "unfit": _mean_ci(unfit, rng),
        "strict_dead_fraction": _mean_ci(dead, rng),
        "n_band": {"counts": nb.tolist(),
                   "fractions": (nb / total).tolist() if total else [None] * 33,
                   "n": total},
        "abs_v_trajectory": _trajectory(logs, "layer1_v_unit",
                                         transform=np.abs),
        "w_flip_norm_trajectory": _trajectory(logs, "layer1_w_flip_norm",
                                               "layer1_branch_step"),
        "kappa": {"t20": _mean_ci(kappa_t20, rng),
                  "late": _mean_ci(kappa_late, rng),
                  "delta": _mean_ci(np.asarray(kappa_late)-kappa_t20, rng),
                  "zbar_shape_signed_registered_aggregate":
                      _mean_ci(shape_signed, rng),
                  "zbar_shape_abs_typical_task_median":
                      _mean_ci(shape_abs_typical, rng),
                  "zbar_shape_abs_mean_all_unit_tasks":
                      _mean_ci(shape_abs_mean, rng),
                  "zbar_shape_abs_max_unit_task_upper_bound":
                      _mean_ci(shape_abs_max, rng)},
        "k_on_delta_prime": _group_k_on(logs),
    }


def _paired(logs_by_arm, pair, key="layer1_zmax", *, reverse=False,
            alive=False, change=False):
    left, right = pair
    lm = {int(np.asarray(z["seed"]).item()): z for z in logs_by_arm[left]}
    rm = {int(np.asarray(z["seed"]).item()): z for z in logs_by_arm[right]}
    common = sorted(set(lm) & set(rm))
    values = []
    for seed in common:
        a, b = lm[seed], rm[seed]
        if change:
            va = _seed_window(a, key, 451, 500, reverse=reverse, alive=alive) - _seed_window(a, key, 251, 300, reverse=reverse, alive=alive)
            vb = _seed_window(b, key, 451, 500, reverse=reverse, alive=alive) - _seed_window(b, key, 251, 300, reverse=reverse, alive=alive)
            values.append([va, vb])
        else:
            values.append(_seed_window(a, key, 451, 500, reverse=reverse, alive=alive) - _seed_window(b, key, 451, 500, reverse=reverse, alive=alive))
    rng = np.random.default_rng(SEED)
    if change:
        a = _mean_ci([v[0] for v in values], rng)
        b = _mean_ci([v[1] for v in values], rng)
        return {left: a, right: b, "paired_seeds": common}
    return {**_mean_ci(values, rng), "seed_differences": values,
            "paired_seeds": common, "contrast": f"{left} - {right}"}


def compute_report(logs_by_arm: Mapping[str, Sequence[Mapping[str, np.ndarray]]]):
    """Compute every report-only item in spec section 6."""
    missing = [a for a in ARMS if a not in logs_by_arm]
    if missing:
        raise ValueError(f"missing arms: {missing}")
    arm = {a: _arm_report(logs_by_arm[a]) for a in ARMS}
    # The registered all-negative contrast uses the same per-seed aggregation.
    negdiff = []
    lm = {int(np.asarray(z["seed"]).item()): z for z in logs_by_arm[FIXED_PAIR[0]]}
    rm = {int(np.asarray(z["seed"]).item()): z for z in logs_by_arm[FIXED_PAIR[1]]}
    negseeds = sorted(set(lm) & set(rm))
    for seed in negseeds:
        x, y = lm[seed], rm[seed]
        ax = _at_steps(x["layer1_zmax"], x["step"], TASK_STEPS, "zmax")
        ay = _at_steps(y["layer1_zmax"], y["step"], TASK_STEPS, "zmax")
        negdiff.append(float((ax < 0).mean() - (ay < 0).mean()))
    rng = np.random.default_rng(SEED)
    comparisons = {
        "fixed_all_negative_delta": {**_mean_ci(negdiff, rng),
                                     "seed_differences": negdiff,
                                     "paired_seeds": negseeds,
                                     "contrast": "LRwf21_1216 - FB21LRwf21_1216"},
        "diagnostic_delta_zmax": _paired(logs_by_arm, DIAGNOSTIC_PAIR),
        "diagnostic_q2_analog": _paired(logs_by_arm, DIAGNOSTIC_PAIR,
                                        "layer1_zbar", change=True),
        "free_delta_zmax": _paired(logs_by_arm, FREE_PAIR),
        "sensitivity": {
            "reverse_q1": _paired(logs_by_arm, FIXED_PAIR, reverse=True),
            "reverse_q2_fixed": _paired(logs_by_arm, FIXED_PAIR,
                                         "layer1_zbar", reverse=True,
                                         change=True),
            "reverse_q2_free": _paired(logs_by_arm, FREE_PAIR,
                                        "layer1_zbar", reverse=True,
                                        change=True),
            "alive_q1_denom_gt_0p25": _paired(logs_by_arm, FIXED_PAIR,
                                               alive=True),
        },
    }
    return {"schema": "fb_width_seat_0915_report_v1", "bootstrap_seed": SEED,
            "bootstrap_n": 2000, "task_internal_intervals": 49,
            "aggregation": {
                "unit_metrics": "unit median per task, then window mean per seed",
                "all_negative_fraction": "unit fraction per task, then window mean per seed",
                "alive": "at each task/unit, layer1_denom > 0.25",
                "paired_ci": "mean paired seed difference; percentile seed bootstrap",
            },
            "arms": arm, "comparisons": comparisons}


def report_from_directory(out):
    out = Path(out)
    logs = {a: [] for a in ARMS}
    required = ("layer1_zbar", "layer1_zmax", "layer1_w_free",
                "layer1_w_flip")
    for a in ARMS:
        for s in range(10):
            p = out / "logs" / f"{a}_seed{s}.npz"
            if not p.exists():
                continue
            z = np.load(p, allow_pickle=False)
            divergent = bool(np.asarray(z["numeric_divergence"]).item()) if "numeric_divergence" in z else False
            finite = all(np.isfinite(z[k]).all() for k in required)
            if not divergent and finite:
                logs[a].append(z)
    return compute_report(logs)


def synthetic_selftest():
    """Small direct test of the error-prone audit identity and grouping."""
    n, h = 50, 3
    steps = TASK_STEPS.copy()
    flip = np.zeros((n, 15)); flip[:, 0] = np.arange(n) % 2
    gamma = np.linspace(.2, .4, n)
    wf = np.ones((n, h, 15)) * .1
    wr = np.ones((n, h, 5)) * .2
    kick = np.einsum("tf,thf->th", np.diff(flip, axis=0), wf[:-1])
    off = -.5*np.diff(gamma)[:, None]*(wf[:-1].sum(-1)+wr[:-1].sum(-1))
    prime = np.arange(49)[:, None]*.001 + np.arange(h)[None]*.01
    zb = np.zeros((n, h)); zb[1:] = np.cumsum(kick+off+prime, axis=0)
    z = {"step": steps, "layer1_branch_step": steps,
         "layer1_w_free_step": steps, "layer1_w_flip": wf,
         "layer1_w_free": wr, "layer1_k_on": np.tile(np.arange(h),(n,1)),
         "layer1_zbar": zb, "flip_state": flip, "gamma": gamma}
    got = task_internal_displacement(z)
    err = float(np.max(np.abs(got["delta_prime"] - prime)))
    return {"pass": bool(err < 1e-12 and got["delta_prime"].shape == (49, h)),
            "max_abs_error": err, "shape": list(got["delta_prime"].shape)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path)
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()
    result = synthetic_selftest() if args.selftest else report_from_directory(args.out)
    if not args.selftest:
        (args.out / "report_only.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result if args.selftest else {"written": str(args.out / "report_only.json")}, indent=2))


if __name__ == "__main__":
    main()
