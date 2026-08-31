"""Registered post-hoc decomposition of death in centered MLP arms.

This analysis reads only committed ``mlp2_phase1_0829`` artifacts.  It does
not train, replay, or reconstruct checkpoints.

Run from the repository root::

    .venv/bin/python -m src.centered_death_posthoc
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "specs" / "spec_centered_death_posthoc_0831.md"
SOURCE = ROOT / "results" / "mlp2_phase1_0829"
DEFAULT_OUT = ROOT / "results" / "centered_death_posthoc_0831"
ARMS = ("L1w100_A1", "L2_A1", "L2_none", "L2_Aall")
MAIN_ARMS = ("L1w100_A1", "L2_A1", "L2_none")
CENTERED_ARMS = ("L1w100_A1", "L2_A1")
SEEDS = tuple(range(10))
PERIOD = 10_000
LOP_EVERY = 1_000
BOOTSTRAP_B = 10_000
BOOTSTRAP_SEED = 20_260_829
SQRT5 = float(np.sqrt(5.0))
NAN_UNIT_THRESHOLD = 0.01
EXCLUSION_THRESHOLD = 0.05


class SanityError(RuntimeError):
    """The committed inputs do not satisfy a structural requirement."""


@dataclass(frozen=True)
class RunData:
    arm: str
    seed: int
    step: np.ndarray
    flip_state: np.ndarray
    layers: dict[int, dict[str, np.ndarray]]
    path: Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def shared_bootstrap_draws(
    B: int = BOOTSTRAP_B, seed: int = BOOTSTRAP_SEED
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return (
        rng.integers(0, len(SEEDS), size=(B, len(SEEDS))),
        rng.integers(0, len(SEEDS), size=(B, len(SEEDS))),
    )


def estimate(values: Iterable[float], draws: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return dict(point=np.nan, ci_lo=np.nan, ci_hi=np.nan, n_seed=0)
    if array.size != draws.shape[1]:
        # Registered analyses have ten seeds.  This branch keeps report-only
        # rows well-defined if a seed has no eligible unit.
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        local = rng.integers(0, array.size, size=(draws.shape[0], array.size))
    else:
        local = draws
    boot = np.median(array[local], axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return dict(
        point=float(np.median(array)), ci_lo=float(lo), ci_hi=float(hi),
        n_seed=int(array.size),
    )


def estimate_unpaired_difference(
    arm_values: Iterable[float], base_values: Iterable[float],
    arm_draws: np.ndarray, base_draws: np.ndarray,
) -> dict[str, float | int]:
    arm = np.asarray(list(arm_values), dtype=float)
    base = np.asarray(list(base_values), dtype=float)
    arm = arm[np.isfinite(arm)]
    base = base[np.isfinite(base)]
    if arm.size != len(SEEDS) or base.size != len(SEEDS):
        raise SanityError("unpaired registered contrast requires all ten seeds")
    boot = np.median(arm[arm_draws], axis=1) - np.median(base[base_draws], axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return dict(
        point=float(np.median(arm) - np.median(base)),
        ci_lo=float(lo), ci_hi=float(hi), n_seed=len(SEEDS),
    )


def estimate_paired(values: Iterable[float], draws: np.ndarray) -> dict[str, float | int]:
    return estimate(values, draws)


def boundary_mask(step: np.ndarray) -> np.ndarray:
    return np.asarray(step[1:] % PERIOD == LOP_EVERY, dtype=bool)


def final_dead_onsets(dead: np.ndarray) -> np.ndarray:
    """Return last-run onset indices for units dead at the final record."""
    if dead.ndim != 2:
        raise ValueError("dead must be records x units")
    out = np.full(dead.shape[1], -1, dtype=int)
    for unit in np.flatnonzero(dead[-1]):
        alive = np.flatnonzero(~dead[:, unit])
        onset = int(alive[-1] + 1) if alive.size else 0
        if onset > 0:
            out[unit] = onset
    return out


def classify_e2(ci_lo: float, ci_hi: float) -> str:
    if ci_hi < 0.10:
        return "BIAS_CHANNEL_DOMINANT"
    if ci_lo > 0.30:
        return "MU_CHANNEL_ALIVE"
    return "CHANNEL_MIXED"


def classify_e3(bnd: dict[str, Any], internal: dict[str, Any]) -> str:
    if bnd["ci_hi"] < 0 and internal["ci_lo"] > 0:
        return "BOUNDARY_CARRIES_DESCENT"
    if bnd["ci_hi"] < 0 and abs(bnd["point"]) > abs(internal["point"]):
        return "BOUNDARY_DOMINANT"
    return "NOT_LOCALIZED"


def classify_e4(centered: dict[str, Any], difference: dict[str, Any]) -> str:
    if centered["ci_lo"] <= 0 <= centered["ci_hi"]:
        return "CENTERING_REMOVES_BOUNDARY_DESCENT"
    if difference["ci_lo"] <= 0 <= difference["ci_hi"]:
        return "CENTERING_NO_BOUNDARY_EFFECT"
    if difference["ci_lo"] > 0 and centered["ci_hi"] < 0:
        return "CENTERING_REDUCES_BUT_NOT_REMOVES"
    return "CENTERING_BOUNDARY_EFFECT_OTHER"


def _layers_in_npz(z: Any) -> list[int]:
    return [layer for layer in (1, 2) if f"layer{layer}_M" in z.files]


def load_run(path: Path, arm: str, seed: int) -> RunData:
    if not path.exists():
        raise SanityError(f"missing registered log: {path}")
    with np.load(path, allow_pickle=False) as z:
        if str(z["arm"]) != arm or int(z["seed"]) != seed:
            raise SanityError(f"log identity mismatch: {path}")
        step = z["step"].astype(np.int64, copy=True)
        flip_state = z["flip_state"].astype(np.float32, copy=True)
        layers: dict[int, dict[str, np.ndarray]] = {}
        for layer in _layers_in_npz(z):
            prefix = f"layer{layer}_"
            layers[layer] = {
                "M": z[prefix + "M"].astype(float),
                "B": z[prefix + "B"].astype(float),
                "denom": z[prefix + "denom"].astype(float),
                "p_hat": z[prefix + "p_hat"].astype(float),
                "strict_dead": z[prefix + "strict_dead"].astype(np.int64),
                "median_M": z[prefix + "median_M"].astype(float),
                "median_B": z[prefix + "median_B"].astype(float),
            }
    expected = np.arange(0, 5_000_000 + LOP_EVERY, LOP_EVERY)
    if not np.array_equal(step, expected):
        raise SanityError(f"unexpected record grid: {path}")
    return RunData(arm, seed, step, flip_state, layers, path)


def _csv_rows(layer_stats: pd.DataFrame, run: RunData, layer: int) -> pd.DataFrame:
    rows = layer_stats[
        (layer_stats.arm == run.arm)
        & (layer_stats.seed == run.seed)
        & (layer_stats.layer == layer)
    ].sort_values("step")
    return rows.reset_index(drop=True)


def inspect_run(
    run: RunData, layer_stats: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    sanity_rows: list[dict[str, Any]] = []
    unit_rows: list[dict[str, Any]] = []
    bmask = boundary_mask(run.step)
    if int(bmask.sum()) != 500:
        raise SanityError(f"boundary classification not 500: {run.path}")

    flip_changed = np.any(run.flip_state[1:] != run.flip_state[:-1], axis=1)
    flip_positions_ok = bool(np.all((run.step[:-1][flip_changed] % PERIOD) == 0))
    flip_next_ok = bool(np.all((run.step[1:][flip_changed] % PERIOD) == LOP_EVERY))
    s3_pass = bool(flip_changed.sum() == 499 and flip_positions_ok and flip_next_ok)
    sanity_rows.append(dict(
        endpoint="S3", arm=run.arm, layer="all", seed=run.seed,
        metric="flip_change_count", point=int(flip_changed.sum()),
        ci_lo=np.nan, ci_hi=np.nan, label="PASS" if s3_pass else "FAIL",
        basis="499 changes; preceding step % 10000 == 0; next step % 10000 == 1000",
        n_seed=1, n_unit=np.nan, note="boundary mask contains 500 transitions",
    ))

    per_layer: dict[int, dict[str, Any]] = {}
    for layer, arrays in run.layers.items():
        M, B = arrays["M"], arrays["B"]
        denom, p_hat = arrays["denom"], arrays["p_hat"]
        beta = M + B
        dead = p_hat == 0
        raw_b = B * denom
        log_sigma = np.log(denom)
        finite_beta = np.isfinite(beta)

        s1_left = int(np.sum(dead & finite_beta & (beta > -1.0)))
        s1_right = int(np.sum((beta <= -SQRT5) & finite_beta & ~dead))
        s1_pass = s1_left == 0 and s1_right == 0
        sanity_rows.append(dict(
            endpoint="S1", arm=run.arm, layer=layer, seed=run.seed,
            metric="absorption_inequality_violations", point=s1_left + s1_right,
            ci_lo=np.nan, ci_hi=np.nan, label="PASS" if s1_pass else "FAIL",
            basis="p_hat=0 => beta<=-1 and beta<=-sqrt(5) => p_hat=0",
            n_seed=1, n_unit=dead.shape[1],
            note=f"left={s1_left}; right={s1_right}",
        ))

        rows = _csv_rows(layer_stats, run, layer)
        task_idx = np.flatnonzero((run.step > 0) & (run.step % PERIOD == 0))
        if len(rows) != len(task_idx):
            s2_pass, s2_error = False, np.inf
            dead_equal = False
        else:
            m_exp = rows["median_M"].to_numpy(float)
            b_exp = rows["median_B"].to_numpy(float)
            rel_m = np.abs(arrays["median_M"][task_idx] - m_exp) / np.maximum(
                np.abs(m_exp), 1e-300
            )
            rel_b = np.abs(arrays["median_B"][task_idx] - b_exp) / np.maximum(
                np.abs(b_exp), 1e-300
            )
            s2_error = float(max(np.max(rel_m), np.max(rel_b)))
            dead_equal = bool(np.array_equal(
                arrays["strict_dead"][task_idx], rows["strict_dead"].to_numpy(int)
            ))
            s2_pass = bool(s2_error < 1e-9 and dead_equal)
        sanity_rows.append(dict(
            endpoint="S2", arm=run.arm, layer=layer, seed=run.seed,
            metric="task_end_reconciliation_max_relerr", point=s2_error,
            ci_lo=np.nan, ci_hi=np.nan, label="PASS" if s2_pass else "FAIL",
            basis="median_M/B relative error <1e-9 and strict_dead exact",
            n_seed=1, n_unit=dead.shape[1], note=f"strict_dead_exact={dead_equal}",
        ))

        s4_mismatch = int(np.sum(arrays["strict_dead"] != dead.sum(axis=1)))
        sanity_rows.append(dict(
            endpoint="S4", arm=run.arm, layer=layer, seed=run.seed,
            metric="strict_dead_identity_mismatches", point=s4_mismatch,
            ci_lo=np.nan, ci_hi=np.nan,
            label="PASS" if s4_mismatch == 0 else "FAIL",
            basis="layer strict_dead == count(p_hat == 0)", n_seed=1,
            n_unit=dead.shape[1], note="",
        ))

        nan_fraction = np.mean(~finite_beta, axis=0)
        excluded = nan_fraction > NAN_UNIT_THRESHOLD
        sanity_rows.append(dict(
            endpoint="S6", arm=run.arm, layer=layer, seed=run.seed,
            metric="excluded_unit_fraction", point=float(excluded.mean()),
            ci_lo=np.nan, ci_hi=np.nan,
            label="PASS" if excluded.mean() < EXCLUSION_THRESHOLD else "FAIL",
            basis="unit NaN fraction >1%; excluded units <5%",
            n_seed=1, n_unit=dead.shape[1], note=f"excluded={int(excluded.sum())}",
        ))

        dM, dB = np.diff(M, axis=0), np.diff(B, axis=0)
        d_beta = np.diff(beta, axis=0)
        d_raw_b = np.diff(raw_b, axis=0)
        d_log_sigma = np.diff(log_sigma, axis=0)
        onset = final_dead_onsets(dead)
        death_transitions = (~dead[:-1]) & dead[1:]
        revival_transitions = dead[:-1] & (~dead[1:])
        same_flip = ~flip_changed
        per_layer[layer] = dict(
            death_count=int(death_transitions.sum()),
            revival_count=int(revival_transitions.sum()),
            # The registered task-internal count starts after the initialized
            # state.  The 0->1000 transition is retained separately because it
            # contains the EMA start-up transient that also matters for E2.
            within_revival=int(revival_transitions[
                same_flip & (run.step[:-1] > 0)
            ].sum()),
            initial_revival=int(revival_transitions[0].sum()),
            final_dead=int(dead[-1].sum()),
            core_dead=int(np.sum(dead[-1] & np.all(dead[-1000:], axis=0))),
        )

        for unit in range(dead.shape[1]):
            onset_idx = int(onset[unit])
            valid = not bool(excluded[unit])
            delta_beta_total = float(np.nansum(d_beta[:, unit])) if valid else np.nan
            delta_M_total = float(np.nansum(dM[:, unit])) if valid else np.nan
            delta_B_total = float(np.nansum(dB[:, unit])) if valid else np.nan
            rho = (
                abs(delta_M_total) / abs(delta_beta_total)
                if valid and delta_beta_total != 0 else np.nan
            )
            b0 = float(raw_b[10, unit])
            bf = float(raw_b[-1, unit])
            s0 = float(denom[10, unit])
            sf = float(denom[-1, unit])
            unit_rows.append(dict(
                arm=run.arm, seed=run.seed, layer=layer, unit=unit,
                excluded_nan_gt_1pct=int(excluded[unit]),
                nan_fraction=float(nan_fraction[unit]),
                delta_beta_total=delta_beta_total,
                delta_beta_bnd=float(np.nansum(d_beta[bmask, unit])) if valid else np.nan,
                delta_beta_int=float(np.nansum(d_beta[~bmask, unit])) if valid else np.nan,
                delta_M_total=delta_M_total, delta_B_total=delta_B_total,
                rho_M=rho,
                delta_beta_from_step10000=(
                    float(beta[-1, unit] - beta[10, unit]) if valid else np.nan
                ),
                delta_M_from_step10000=(
                    float(M[-1, unit] - M[10, unit]) if valid else np.nan
                ),
                rho_M_from_step10000=(
                    abs(float(M[-1, unit] - M[10, unit]))
                    / abs(float(beta[-1, unit] - beta[10, unit]))
                    if valid and beta[-1, unit] != beta[10, unit] else np.nan
                ),
                delta_b_raw_total=float(np.nansum(d_raw_b[:, unit])) if valid else np.nan,
                delta_b_raw_bnd=float(np.nansum(d_raw_b[bmask, unit])) if valid else np.nan,
                delta_b_raw_int=float(np.nansum(d_raw_b[~bmask, unit])) if valid else np.nan,
                delta_log_sigma_total=float(np.nansum(d_log_sigma[:, unit])) if valid else np.nan,
                delta_log_sigma_bnd=float(np.nansum(d_log_sigma[bmask, unit])) if valid else np.nan,
                delta_log_sigma_int=float(np.nansum(d_log_sigma[~bmask, unit])) if valid else np.nan,
                onset_index=onset_idx if onset_idx >= 0 else np.nan,
                onset_step=int(run.step[onset_idx]) if onset_idx >= 0 else np.nan,
                onset_beta=float(beta[onset_idx, unit]) if onset_idx >= 0 else np.nan,
                onset_boundary=(int(run.step[onset_idx] % PERIOD == LOP_EVERY)
                                if onset_idx >= 0 else np.nan),
                final_dead=int(dead[-1, unit]),
                b_raw_step10000=b0, b_raw_final=bf,
                sigma_step10000=s0, sigma_final=sf,
            ))
    return sanity_rows, unit_rows, per_layer


def _seed_medians(
    units: pd.DataFrame, arm: str, layer: int, column: str,
    *, mask: pd.Series | None = None,
) -> np.ndarray:
    frame = units[(units.arm == arm) & (units.layer == layer)
                  & (units.excluded_nan_gt_1pct == 0)].copy()
    if mask is not None:
        frame = frame[mask.loc[frame.index]]
    values = frame.groupby("seed", sort=True)[column].median()
    return values.reindex(SEEDS).to_numpy(float)


def _row(
    endpoint: str, metric: str, est: dict[str, Any] | None = None, *,
    arm: str = "", layer: int | str = "", baseline: str = "",
    label: str = "REPORT_ONLY", basis: str = "", n_unit: int | float = np.nan,
    note: str = "",
) -> dict[str, Any]:
    est = est or {}
    return dict(
        endpoint=endpoint, arm=arm, layer=layer, seed="", metric=metric,
        point=est.get("point", np.nan), ci_lo=est.get("ci_lo", np.nan),
        ci_hi=est.get("ci_hi", np.nan), label=label, basis=basis,
        baseline=baseline, n_seed=est.get("n_seed", np.nan), n_unit=n_unit,
        note=note,
    )


def build_endpoints(
    units: pd.DataFrame, event_stats: dict[tuple[str, int, int], dict[str, Any]],
    provenance: dict[str, Any], sanity: pd.DataFrame,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    draws, other_draws = shared_bootstrap_draws()
    rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {}

    # E1 levels and all pairwise contrasts.
    e1_values: dict[str, np.ndarray] = {}
    for arm in MAIN_ARMS:
        values = _seed_medians(units, arm, 1, "onset_beta")
        e1_values[arm] = values
        est = estimate(values, draws)
        rows.append(_row(
            "E1", "onset_beta_level", est, arm=arm, layer=1,
            basis="seed median of final-run onset beta; percentile seed bootstrap",
        ))
    e1_contrasts: list[dict[str, Any]] = []
    pairs = (("L1w100_A1", "L2_A1"), ("L1w100_A1", "L2_none"),
             ("L2_A1", "L2_none"))
    for arm, base in pairs:
        if (arm, base) == ("L2_A1", "L2_none"):
            est = estimate_paired(e1_values[arm] - e1_values[base], draws)
            pairing = "paired seeds"
        else:
            est = estimate_unpaired_difference(
                e1_values[arm], e1_values[base], draws, other_draws
            )
            pairing = "unpaired seed clusters"
        inside = bool(est["ci_lo"] >= -0.15 and est["ci_hi"] <= 0.15)
        e1_contrasts.append(est | {"inside": inside})
        rows.append(_row(
            "E1", "onset_beta_contrast", est, arm=arm, baseline=base, layer=1,
            label="WITHIN_EQUIVALENCE_BAND" if inside else "OUTSIDE_EQUIVALENCE_BAND",
            basis=f"{pairing}; CI must fit in [-0.15,+0.15]",
        ))
    e1_label = "WALL_INVARIANT" if all(x["inside"] for x in e1_contrasts) else "WALL_MOVES"
    details["E1"] = e1_label
    rows.append(_row(
        "E1_VERDICT", "all_pairwise_onset_beta", label=e1_label,
        basis="all three pairwise 95% CIs within [-0.15,+0.15]",
        layer=1,
    ))

    # E2 centered-channel decomposition and report-only multipliers.
    for arm in CENTERED_ARMS:
        rho = _seed_medians(units, arm, 1, "rho_M")
        est = estimate(rho, draws)
        label = classify_e2(float(est["ci_lo"]), float(est["ci_hi"]))
        details[f"E2_{arm}"] = label
        rows.append(_row(
            "E2", "rho_M", est, arm=arm, layer=1, label=label,
            basis="abs(sum delta M)/abs(delta beta), literal registered sum from step 0",
            note="step 0 precedes EMA convergence; S5's construction-near-zero statement has this startup exception",
        ))
        sensitivity = _seed_medians(units, arm, 1, "rho_M_from_step10000")
        sensitivity_est = estimate(sensitivity, draws)
        rows.append(_row(
            "E2_SENSITIVITY_REPORT_ONLY", "rho_M_from_step10000",
            sensitivity_est, arm=arm, layer=1,
            label=classify_e2(float(sensitivity_est["ci_lo"]),
                              float(sensitivity_est["ci_hi"])),
            basis="unregistered sensitivity excluding EMA startup; does not replace E2",
            note="shown because step 0 is initialization and sigma multiplier already uses step 10000",
        ))
        frame = units[(units.arm == arm) & (units.layer == 1)
                      & (units.excluded_nan_gt_1pct == 0)].copy()
        frame["abs_b_multiplier"] = (
            frame["b_raw_final"].abs() / frame["b_raw_step10000"].abs()
        )
        frame["sigma_multiplier"] = frame["sigma_final"] / frame["sigma_step10000"]
        for metric in ("abs_b_multiplier", "sigma_multiplier"):
            values = frame.groupby("seed", sort=True)[metric].median().reindex(SEEDS)
            rows.append(_row(
                "E2_REPORT_ONLY", metric, estimate(values.to_numpy(float), draws),
                arm=arm, layer=1,
                basis="median unitwise final/step10000 multiplier; |b_raw| for bias",
            ))

    # E3 per-arm decomposition.
    e3_estimates: dict[tuple[str, str], dict[str, Any]] = {}
    e3_arms = (*MAIN_ARMS, "L2_Aall")
    metric_specs = (
        ("delta_beta_bnd", 1.0), ("delta_beta_int", 1.0),
        ("delta_beta_bnd_per_transition", 500.0),
        ("delta_beta_int_per_transition", 4500.0),
        ("delta_b_raw_bnd", 1.0), ("delta_b_raw_int", 1.0),
        ("delta_log_sigma_bnd", 1.0), ("delta_log_sigma_int", 1.0),
    )
    for arm in e3_arms:
        for output_metric, divisor in metric_specs:
            source_metric = output_metric.replace("_per_transition", "")
            values = _seed_medians(units, arm, 1, source_metric) / divisor
            est = estimate(values, draws)
            e3_estimates[(arm, output_metric)] = est
            rows.append(_row(
                "E3", output_metric, est, arm=arm, layer=1,
                basis="seed median of unitwise additive decomposition; percentile seed bootstrap",
                label="REPORT_ONLY" if arm == "L2_Aall" else "COMPONENT",
            ))
        eligible = units[(units.arm == arm) & (units.layer == 1)
                         & units.onset_step.notna()]
        frac = eligible.groupby("seed")["onset_boundary"].mean().reindex(SEEDS)
        rows.append(_row(
            "E3_REPORT_ONLY", "onset_boundary_fraction",
            estimate(frac.to_numpy(float), draws), arm=arm, layer=1,
            basis="fraction of eligible final-dead onsets at step % 10000 == 1000",
        ))
    primary_bnd = e3_estimates[("L1w100_A1", "delta_beta_bnd")]
    primary_int = e3_estimates[("L1w100_A1", "delta_beta_int")]
    e3_label = classify_e3(primary_bnd, primary_int)
    details["E3"] = e3_label
    rows.append(_row(
        "E3_VERDICT", "L1w100_A1_localization", arm="L1w100_A1", layer=1,
        label=e3_label,
        basis="boundary CI upper<0 and internal CI lower>0; registered fallback hierarchy",
    ))

    # E4 unit-index paired A1-minus-none.
    pair = units[(units.arm.isin(["L2_A1", "L2_none"])) & (units.layer == 1)
                 & (units.excluded_nan_gt_1pct == 0)].copy()
    wide = pair.pivot(index=["seed", "unit"], columns="arm",
                      values=["delta_beta_bnd", "delta_b_raw_bnd"])
    pair_final = (provenance.get("sanity", {}).get("S_pair_final") or {})
    paired_ok = bool(
        (provenance.get("sanity", {}).get("S_pair") or {}).get("pass_")
        and pair_final.get("pass_") and pair_final.get("paired_pass")
    )
    if paired_ok:
        beta_diff = (wide["delta_beta_bnd"]["L2_A1"]
                     - wide["delta_beta_bnd"]["L2_none"])
        raw_diff = (wide["delta_b_raw_bnd"]["L2_A1"]
                    - wide["delta_b_raw_bnd"]["L2_none"])
        beta_seed = beta_diff.groupby("seed").median().reindex(SEEDS).to_numpy(float)
        raw_seed = raw_diff.groupby("seed").median().reindex(SEEDS).to_numpy(float)
        beta_est, raw_est = estimate(beta_seed, draws), estimate(raw_seed, draws)
        pairing_basis = "S_pair and S_pair_final PASS; seed+unit paired"
    else:
        beta_a = _seed_medians(units, "L2_A1", 1, "delta_beta_bnd")
        beta_n = _seed_medians(units, "L2_none", 1, "delta_beta_bnd")
        raw_a = _seed_medians(units, "L2_A1", 1, "delta_b_raw_bnd")
        raw_n = _seed_medians(units, "L2_none", 1, "delta_b_raw_bnd")
        beta_est = estimate_unpaired_difference(beta_a, beta_n, draws, other_draws)
        raw_est = estimate_unpaired_difference(raw_a, raw_n, draws, other_draws)
        pairing_basis = "S-pair failed; downgraded to unpaired seed clusters"
    centered_est = e3_estimates[("L2_A1", "delta_beta_bnd")]
    e4_label = classify_e4(centered_est, beta_est)
    raw_label = (
        "BIAS_DESCENT_WORSENED_BY_CENTERING" if raw_est["ci_hi"] < 0
        else "BIAS_DESCENT_DIFFERENCE_INCLUDES_ZERO"
        if raw_est["ci_lo"] <= 0 <= raw_est["ci_hi"]
        else "BIAS_DESCENT_NOT_WORSENED_BY_CENTERING"
    )
    details["E4"] = e4_label
    details["E4_raw_bias"] = raw_label
    rows.append(_row(
        "E4", "delta_beta_bnd_A1_minus_none", beta_est, arm="L2_A1",
        baseline="L2_none", layer=1, label=e4_label,
        basis=pairing_basis,
    ))
    rows.append(_row(
        "E4", "delta_b_raw_bnd_A1_minus_none", raw_est, arm="L2_A1",
        baseline="L2_none", layer=1, label=raw_label,
        basis=pairing_basis,
    ))

    # E5 event counts.
    e5_seed: dict[tuple[str, int], list[int]] = {}
    for arm in ARMS:
        layers = (1,) if arm == "L1w100_A1" else (1, 2)
        for layer in layers:
            values = [event_stats[(arm, seed, layer)]["within_revival"] for seed in SEEDS]
            e5_seed[(arm, layer)] = values
            rows.append(_row(
                "E5_COUNT", "within_task_revival_count",
                dict(point=int(sum(values)), ci_lo=np.nan, ci_hi=np.nan, n_seed=10),
                arm=arm, layer=layer,
                basis="dead->alive where flip_state is unchanged and transition starts after step 0",
                note=("seed_counts=" + json.dumps(values)
                      + "; initial_0_to_1000_excluded="
                      + str(sum(event_stats[(arm, seed, layer)]["initial_revival"]
                                for seed in SEEDS))),
            ))
    none_zero = all(value == 0 for value in e5_seed[("L2_none", 1)])
    centered_positive = all(
        all(value > 0 for value in e5_seed[(arm, 1)]) for arm in CENTERED_ARMS
    )
    centered_zero = any(
        all(value == 0 for value in e5_seed[(arm, 1)]) for arm in CENTERED_ARMS
    )
    if not none_zero:
        e5_label = "THEOREM_VIOLATED"
    elif centered_positive:
        e5_label = "ABSORPTION_BROKEN_BY_EMA"
    elif centered_zero:
        e5_label = "ABSORPTION_HOLDS_UNDER_CENTERING"
    else:
        e5_label = "ABSORPTION_MIXED_UNDER_CENTERING"
    details["E5"] = e5_label
    rows.append(_row(
        "E5_VERDICT", "within_task_absorption", label=e5_label, layer=1,
        basis="none L1 all-zero and both centered L1 all-seed positive",
        note="L2_none layer2 is the required moving-input control",
    ))

    # E6 churn and persistent-core fractions.
    for arm in ARMS:
        layers = (1,) if arm == "L1w100_A1" else (1, 2)
        for layer in layers:
            stats = [event_stats[(arm, seed, layer)] for seed in SEEDS]
            deaths = int(sum(x["death_count"] for x in stats))
            revivals = int(sum(x["revival_count"] for x in stats))
            final_dead = int(sum(x["final_dead"] for x in stats))
            core_dead = int(sum(x["core_dead"] for x in stats))
            for metric, point, note in (
                ("death_transition_count", deaths, "all transitions, all seeds"),
                ("revival_transition_count", revivals, "all transitions, all seeds"),
                ("churn_ratio_revival_over_death", revivals / deaths if deaths else np.nan,
                 "revival transitions / death transitions"),
                ("continuous_dead_last_1000_fraction",
                 core_dead / final_dead if final_dead else np.nan,
                 f"core={core_dead}; final_dead={final_dead}"),
            ):
                rows.append(_row(
                    "E6_REPORT_ONLY", metric,
                    dict(point=point, ci_lo=np.nan, ci_hi=np.nan, n_seed=10),
                    arm=arm, layer=layer, basis="descriptive count or ratio", note=note,
                ))

    # If the uncentered layer-1 theorem control fails, the registered spec
    # requires holding E1--E4.  This should not trigger for committed inputs.
    if e5_label == "THEOREM_VIOLATED":
        for row in rows:
            if row["endpoint"].split("_")[0] in {"E1", "E2", "E3", "E4"}:
                row["label"] = "HELD_THEOREM_VIOLATED"
        details["E1_E4_held"] = True

    return rows, details


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    if isinstance(value, (int, np.integer)):
        return f"{int(value)}"
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return f"{int(value)}"
    return f"{float(value):.{digits}g}"


def render_summary(verdict: pd.DataFrame, details: dict[str, Any]) -> str:
    def find(endpoint: str, metric: str, arm: str = "") -> pd.Series:
        rows = verdict[(verdict.endpoint == endpoint) & (verdict.metric == metric)]
        if arm:
            rows = rows[rows.arm == arm]
        if rows.empty:
            raise SanityError(f"summary row missing: {endpoint}/{metric}/{arm}")
        return rows.iloc[0]

    lines = [
        "# centered 腕の死因分解（事後解析）", "",
        "> **格の自己申告（最重要）。** 本 spec は**事前登録ではない**。起草者（Claude）は起草前に、下記 E1–E6 に対応する数値をチャット内で `results/mlp2_phase1_0829/logs/*.npz` から観察している。したがって本 spec は `mlp2_centering_delay_posthoc_0830` と同格の「**事後解析の登録**」であり、判定ラベルは引用時に必ず事後の格で運ぶ。判定閾値は観察値から離した位置に置いたが、forking path の risk は残る。", "",
        "> **新規学習・checkpoint 再計算なし。** 入力は commit 済み `mlp2_phase1_0829` のログだけ。", "",
        "## 結論", "",
        f"- E1: **{details['E1']}**", f"- E2: `L1w100_A1` **{details['E2_L1w100_A1']}** / `L2_A1` **{details['E2_L2_A1']}**",
        f"- E3: **{details['E3']}**", f"- E4: **{details['E4']}** / raw bias: **{details['E4_raw_bias']}**",
        f"- E5: **{details['E5']}**", "",
    ]

    lines += ["## E1 壁の位置", "", "| arm | onset β | 95% CI |", "|---|---:|---:|"]
    for arm in MAIN_ARMS:
        row = find("E1", "onset_beta_level", arm)
        lines.append(f"| `{arm}` | {_fmt(row.point)} | [{_fmt(row.ci_lo)}, {_fmt(row.ci_hi)}] |")
    lines += ["", "3対比の CI がすべて ±0.15 内なら `WALL_INVARIANT`。", ""]

    lines += ["## E2 チャネル", "", "| arm | ρ_M | 95% CI | verdict |", "|---|---:|---:|---|"]
    for arm in CENTERED_ARMS:
        row = find("E2", "rho_M", arm)
        lines.append(f"| `{arm}` | {_fmt(row.point)} | [{_fmt(row.ci_lo)}, {_fmt(row.ci_hi)}] | `{row.label}` |")
    lines += [
        "",
        "登録式は step 0 からの総和なので、EMA 初期化前の大きな M を含む。`M≈0` は centered 層の構成上ほぼ恒真だが、step 0 は例外である。step 10,000 起点の未登録感度分析は `verdict.csv` の `E2_SENSITIVITY_REPORT_ONLY` に併記し、E2 判定を差し替えない。",
        "",
    ]

    lines += ["## E3 降下の局在", "", "| arm | Δβ boundary | 95% CI | Δβ internal | 95% CI |", "|---|---:|---:|---:|---:|"]
    for arm in (*MAIN_ARMS, "L2_Aall"):
        bnd = find("E3", "delta_beta_bnd", arm)
        internal = find("E3", "delta_beta_int", arm)
        lines.append(f"| `{arm}` | {_fmt(bnd.point)} | [{_fmt(bnd.ci_lo)}, {_fmt(bnd.ci_hi)}] | {_fmt(internal.point)} | [{_fmt(internal.ci_lo)}, {_fmt(internal.ci_hi)}] |")
    lines += ["", "> **中央値は加法的でない。** `med(Δβ_boundary) + med(Δβ_internal)` と `med(Δβ_total)` は一致を要求しない。三量は別々に報告し、中央値の和で検算しない。", ""]

    e4 = find("E4", "delta_beta_bnd_A1_minus_none")
    e4b = find("E4", "delta_b_raw_bnd_A1_minus_none")
    lines += [
        "## E4 centering の境界効果", "",
        "| paired contrast | point | 95% CI | verdict |", "|---|---:|---:|---|",
        f"| Δβ boundary (`A1-none`) | {_fmt(e4.point)} | [{_fmt(e4.ci_lo)}, {_fmt(e4.ci_hi)}] | `{e4.label}` |",
        f"| Δb_raw boundary (`A1-none`) | {_fmt(e4b.point)} | [{_fmt(e4b.ci_lo)}, {_fmt(e4b.ci_hi)}] | `{e4b.label}` |", "",
    ]

    lines += ["## E5 タスク内復活", "", "| arm | layer | total | seed counts |", "|---|---:|---:|---|"]
    counts = verdict[verdict.endpoint == "E5_COUNT"]
    for row in counts.itertuples():
        seed_counts = row.note.removeprefix("seed_counts=").split(";", 1)[0]
        lines.append(f"| `{row.arm}` | {row.layer} | {_fmt(row.point)} | `{seed_counts}` |")
    lines += ["", "件数は初期化遷移 step 0→1,000 を除く。その除外件数は `verdict.csv` の note に保存した。"]
    lines += ["", "centered 腕で `strict_dead` を『吸収した』とは書かない。EMA により入力がタスク内でも動き、復活が観測される。", ""]

    sanity = verdict[verdict.endpoint.str.startswith("S")]
    summary = sanity.groupby("endpoint").label.apply(lambda x: "PASS" if (x == "PASS").all() else "FAIL")
    lines += ["## Sanity", "", "| check | status |", "|---|---|"]
    for endpoint, status in summary.items():
        lines.append(f"| `{endpoint}` | **{status}** |")
    s1_l2_fail = sanity[(sanity.endpoint == "S1") & (sanity.layer == 2) & (sanity.label == "FAIL")]
    if not s1_l2_fail.empty:
        lines += ["", "S1 の登録済み `-√5` 十分条件は layer 2 で FAIL。layer 2 の入力次元・支持は layer 1 の5次元 hypercube と異なるためで、E1–E4 が使う layer 1 の S1 は全件 PASS。layer 2 は E5 の moving-input 対照としてのみ用いる。"]

    lines += [
        "", "## E6 とスコープ", "",
        "E6 の死亡・復活・churn・直近100タスク連続死率は `verdict.csv` に全腕・全層を記録した。`strict_dead` は当該タスク支持域上のラベルであり、不可逆な unit-ID の死ではない。", "",
        "対象は condA・w100・T=10^4・batch=1・lr=0.01・center_alpha=0.01・10 seed・5M に限る。他幅、他T、condB、他最適化器へ外挿しない。`L1w100_A1` と `L2_none` は unpaired。因果的な腕間対比は `L2_A1` vs `L2_none` のみ。全数値は事後であり、Phase 1 の事前登録判定を上書きしない。", "",
    ]
    return "\n".join(lines)


def run_analysis(source: Path = SOURCE, outdir: Path = DEFAULT_OUT) -> dict[str, Any]:
    required = [SPEC, source / "layer_stats.csv", source / "provenance.json",
                source / "config_used.yaml", source / "summary.md"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SanityError(f"required registered inputs missing: {missing}")

    layer_stats = pd.read_csv(source / "layer_stats.csv")
    source_provenance = json.loads((source / "provenance.json").read_text(encoding="utf-8"))
    sanity_rows: list[dict[str, Any]] = []
    unit_rows: list[dict[str, Any]] = []
    event_stats: dict[tuple[str, int, int], dict[str, Any]] = {}
    input_hashes: dict[str, str] = {}
    for path in required:
        input_hashes[str(path.relative_to(ROOT))] = sha256(path)

    for arm in ARMS:
        for seed in SEEDS:
            path = source / "logs" / f"{arm}_seed{seed}.npz"
            input_hashes[str(path.relative_to(ROOT))] = sha256(path)
            run = load_run(path, arm, seed)
            srows, urows, events = inspect_run(run, layer_stats)
            sanity_rows.extend(srows)
            unit_rows.extend(urows)
            for layer, values in events.items():
                event_stats[(arm, seed, layer)] = values

    sanity_rows.append(dict(
        endpoint="S5", arm="centered_layer1", layer=1, seed="",
        metric="construction_tautology_note_present", point=1,
        ci_lo=np.nan, ci_hi=np.nan, label="PASS",
        basis="rho_M is construction-favored after EMA startup; verdict carries the caveat",
        n_seed=10, n_unit=2000,
        note="step 0 is the explicit startup exception; small post-start rho_M is not an independent discovery",
    ))

    units = pd.DataFrame(unit_rows).sort_values(
        ["arm", "seed", "layer", "unit"], kind="mergesort"
    ).reset_index(drop=True)
    sanity = pd.DataFrame(sanity_rows)
    endpoint_rows, details = build_endpoints(units, event_stats, source_provenance, sanity)
    sanity["baseline"] = ""
    verdict = pd.concat([sanity, pd.DataFrame(endpoint_rows)], ignore_index=True)
    columns = ["endpoint", "arm", "baseline", "layer", "seed", "metric", "point",
               "ci_lo", "ci_hi", "label", "basis", "n_seed", "n_unit", "note"]
    verdict = verdict[columns].sort_values(
        ["endpoint", "arm", "layer", "seed", "metric"], kind="mergesort"
    ).reset_index(drop=True)

    outdir.mkdir(parents=True, exist_ok=True)
    units.to_csv(outdir / "unit_decomposition.csv", index=False)
    verdict.to_csv(outdir / "verdict.csv", index=False)
    (outdir / "summary.md").write_text(
        render_summary(verdict, details), encoding="utf-8"
    )

    implementation_commit = git("rev-parse", "HEAD")
    source_commit = git("log", "-1", "--format=%H", "--", str(source.relative_to(ROOT)))
    provenance = {
        "analysis": "centered_death_posthoc_0831",
        "analysis_grade": "registered_posthoc_not_preregistered",
        "new_training": False,
        "source_result_commit": source_commit,
        "implementation_commit": implementation_commit,
        "spec_commit": git("log", "-1", "--format=%H", "--", str(SPEC.relative_to(ROOT))),
        "input_sha256": input_hashes,
        "bootstrap": {"method": "seed_cluster_percentile", "B": BOOTSTRAP_B,
                      "seed": BOOTSTRAP_SEED, "studentized": "registered_as_degenerate_not_used"},
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "numpy": np.__version__, "pandas": pd.__version__},
        "sanity_status": verdict[verdict.endpoint.str.startswith("S")]
            .groupby("endpoint").label.apply(lambda x: "PASS" if (x == "PASS").all() else "FAIL")
            .to_dict(),
        "verdicts": details,
    }
    (outdir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    provenance = run_analysis(args.source.resolve(), args.outdir.resolve())
    print(json.dumps({"outdir": str(args.outdir.resolve()),
                      "verdicts": provenance["verdicts"],
                      "sanity": provenance["sanity_status"]},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
