"""Analysis for the preregistered dynrepair_0826 continuation.

The runner writes one fixed-shape NPZ per arm.  This module is deliberately
free of torch: the scientific tables can be rebuilt from the raw arrays
without rerunning training.  All confidence intervals resample seed labels,
so every unit and time point from a selected seed moves as one block.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


UNIT_KEYS = (
    "p_hat", "p_count", "pre_max", "x", "r", "w_norm", "b", "v",
    "strict_dead", "utility_nmse",
)
REQUIRED_KEYS = (
    "step", "seed", "unit", "is_traj", "is_task_head", "treated",
    "pre_p_hat", "pre_pre_max", "eval_nmse", *UNIT_KEYS,
)
SCIENTIFIC_CSVS = (
    "traj.csv", "utility.csv", "units.csv", "km.csv", "verdict.csv",
    "placebo.csv", "sanity.csv", "manifest.csv", "raw_sha256.csv",
)


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    """Canonical CSV representation used by the analysis determinism check."""
    return frame.to_csv(index=False, lineterminator="\n", float_format="%.17g").encode("utf-8")


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _shape_text(shape: Iterable[int]) -> str:
    return "x".join(str(int(x)) for x in shape)


def load_raw(path: Path) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Load and validate one arm's raw contract.

    Returns the arrays and the per-key non-finite counts.  Non-finite values
    are counted and carried, never dropped: §5 of the spec registers "report
    the count, do not exclude".  A diverged descriptive-only arm must still
    appear in traj.csv/utility.csv as the NaN it is.
    """
    with np.load(path, allow_pickle=False) as z:
        missing = sorted(set(REQUIRED_KEYS) - set(z.files))
        if missing:
            raise ValueError(f"{path}: missing NPZ keys {missing}")
        raw = {key: np.asarray(z[key]) for key in z.files}

    step = raw["step"]
    seed = raw["seed"]
    unit = raw["unit"]
    if step.ndim != 1 or seed.ndim != 1 or unit.ndim != 1:
        raise ValueError(f"{path}: step/seed/unit must be one-dimensional")
    if len(np.unique(step)) != len(step) or np.any(np.diff(step) <= 0):
        raise ValueError(f"{path}: step must be unique and strictly increasing")
    T, R, H = len(step), len(seed), len(unit)
    for key in UNIT_KEYS:
        if raw[key].shape != (T, R, H):
            raise ValueError(f"{path}: {key} has {raw[key].shape}, expected {(T, R, H)}")
    for key in ("treated", "pre_p_hat", "pre_pre_max"):
        if raw[key].shape != (R, H):
            raise ValueError(f"{path}: {key} has {raw[key].shape}, expected {(R, H)}")
    if raw["eval_nmse"].shape != (T, R):
        raise ValueError(f"{path}: eval_nmse shape mismatch")
    if raw["is_traj"].shape != (T,) or raw["is_task_head"].shape != (T,):
        raise ValueError(f"{path}: record-kind masks must have shape {(T,)}")

    numeric = ("p_hat", "p_count", "pre_max", "x", "r", "w_norm", "b", "v",
               "utility_nmse", "eval_nmse", "pre_p_hat", "pre_pre_max")
    nonfinite = {
        key: int((~np.isfinite(raw[key])).sum())
        for key in numeric
        if int((~np.isfinite(raw[key])).sum())
    }
    return raw, nonfinite


def _index_at(raw: dict[str, np.ndarray], step: int) -> int:
    where = np.flatnonzero(raw["step"] == int(step))
    if len(where) != 1:
        raise ValueError(f"expected one record at step {step}, got {len(where)}")
    return int(where[0])


def _bootstrap_indices(n_seed: int, n_boot: int, seed: int) -> np.ndarray:
    if n_seed <= 0 or n_boot <= 0:
        raise ValueError("bootstrap dimensions must be positive")
    return np.random.default_rng(seed).integers(
        0, n_seed, size=(n_boot, n_seed), endpoint=False
    )


def paired_mean_ci(
    values: np.ndarray, boot_indices: np.ndarray
) -> tuple[float, float, float]:
    """Mean and percentile CI for one paired value per seed."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size != boot_indices.shape[1]:
        raise ValueError("paired values and bootstrap seed width disagree")
    if not np.isfinite(values).all():
        raise ValueError("paired effect contains non-finite values")
    draws = values[boot_indices].mean(axis=1)
    lo, hi = np.quantile(draws, (0.025, 0.975))
    return float(values.mean()), float(lo), float(hi)


def equivalence_verdict(
    lo: float,
    hi: float,
    margin: float,
    *,
    low: str,
    equivalent: str,
    high: str,
) -> str:
    if hi < -margin:
        return low
    if lo >= -margin and hi <= margin:
        return equivalent
    if lo > margin:
        return high
    return "INCONCLUSIVE"


def _quantile_codes(values: pd.Series, q: int) -> pd.Series:
    """Stable rank-based quantiles; works even when many values tie."""
    if len(values) == 0:
        return pd.Series(dtype="int64", index=values.index)
    ranks = values.rank(method="first").to_numpy(dtype=np.float64)
    codes = np.floor((ranks - 1.0) * q / len(values)).astype(np.int64)
    return pd.Series(np.clip(codes, 0, q - 1), index=values.index, dtype="int64")


def _f1_seed_components(units: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, int]:
    """Per-seed numerator/denominator for the cell-adjusted high-low RD."""
    numer, denom, retained = [], [], 0
    for _, seed_frame in units.groupby("seed", sort=True):
        seed_num = 0.0
        seed_den = 0.0
        for _, cell in seed_frame.groupby("premax_cell", sort=True):
            low = cell[cell.utility_group == "low"].strict_dead_short.to_numpy(dtype=float)
            high = cell[cell.utility_group == "high"].strict_dead_short.to_numpy(dtype=float)
            if len(low) and len(high):
                weight = float(min(len(low), len(high)))
                seed_num += weight * (float(high.mean()) - float(low.mean()))
                seed_den += weight
                retained += len(low) + len(high)
        numer.append(seed_num)
        denom.append(seed_den)
    return np.asarray(numer), np.asarray(denom), int(retained)


def adjusted_rd_ci(
    numer: np.ndarray, denom: np.ndarray, boot_indices: np.ndarray
) -> tuple[float, float, float, int]:
    """Point estimate, percentile CI, and the count of unusable draws.

    A resample whose selected seeds contribute no valid cell has a zero
    denominator; those draws are counted and reported, matching the
    ``n_boot_nonfinite`` column of the work-6 confirmation [spec §5, §10].
    """
    numer = np.asarray(numer, dtype=np.float64)
    denom = np.asarray(denom, dtype=np.float64)
    if numer.shape != denom.shape or numer.size != boot_indices.shape[1]:
        raise ValueError("adjusted RD components and bootstrap width disagree")
    total_den = float(denom.sum())
    if total_den <= 0:
        return np.nan, np.nan, np.nan, int(boot_indices.shape[0])
    point = float(numer.sum() / total_den)
    bn = numer[boot_indices].sum(axis=1)
    bd = denom[boot_indices].sum(axis=1)
    valid = bd > 0
    n_invalid = int((~valid).sum())
    if not valid.any():
        return point, np.nan, np.nan, n_invalid
    lo, hi = np.quantile(bn[valid] / bd[valid], (0.025, 0.975))
    return point, float(lo), float(hi), n_invalid


def kaplan_meier(durations: np.ndarray, events: np.ndarray) -> pd.DataFrame:
    """Small deterministic Kaplan-Meier table, including time zero."""
    durations = np.asarray(durations, dtype=np.int64)
    events = np.asarray(events, dtype=bool)
    if durations.shape != events.shape:
        raise ValueError("KM durations/events shape mismatch")
    rows = [dict(tau=0, n_risk=int(len(durations)), n_event=0, n_censor=0,
                 survival=1.0)]
    survival = 1.0
    for time in np.unique(durations):
        at_risk = int((durations >= time).sum())
        n_event = int(((durations == time) & events).sum())
        n_censor = int(((durations == time) & ~events).sum())
        if at_risk and n_event:
            survival *= 1.0 - n_event / at_risk
        rows.append(dict(tau=int(time), n_risk=at_risk, n_event=n_event,
                         n_censor=n_censor, survival=float(survival)))
    return pd.DataFrame(rows)


def km_median(km: pd.DataFrame) -> float:
    hit = km[km.survival <= 0.5]
    return float(hit.tau.iloc[0]) if len(hit) else np.nan


def _trajectory_frame(raw_by_arm: dict[str, dict[str, np.ndarray]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for arm, raw in raw_by_arm.items():
        for ti in np.flatnonzero(raw["is_traj"].astype(bool)):
            p = raw["p_hat"][ti]
            # p_hat cannot carry the divergence: it is mean(pre > 0), and
            # `nan > 0` is False, so a diverged unit reports a finite p_hat=0
            # and would be counted as frozen.  pre_max is the quantity that
            # actually goes non-finite, so the guard reads that instead.
            pre_max = raw["pre_max"][ti]
            for ri, seed in enumerate(raw["seed"]):
                defined = bool(np.isfinite(pre_max[ri]).all())
                rows.append(dict(
                    arm=arm, seed=int(seed), step=int(raw["step"][ti]),
                    U=float(raw["eval_nmse"][ti, ri]),
                    dead_frac_0_05=(
                        float((p[ri] < 0.05).mean()) if defined else np.nan
                    ),
                    frozen_frac=(
                        float((p[ri] == 0.0).mean()) if defined else np.nan
                    ),
                ))
    return pd.DataFrame(rows).sort_values(["arm", "seed", "step"]).reset_index(drop=True)


def _utility_frame(raw_by_arm: dict[str, dict[str, np.ndarray]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for arm, raw in raw_by_arm.items():
        for ti in np.flatnonzero(raw["is_task_head"].astype(bool)):
            step = int(raw["step"][ti])
            for ri, seed in enumerate(raw["seed"]):
                for ui, unit in enumerate(raw["unit"]):
                    rows.append(dict(
                        arm=arm, seed=int(seed), unit=int(unit), step=step,
                        utility_nmse=float(raw["utility_nmse"][ti, ri, ui]),
                    ))
    return pd.DataFrame(rows).sort_values(
        ["arm", "seed", "unit", "step"]
    ).reset_index(drop=True)


def _units_frame(
    a1: dict[str, np.ndarray], t_int: int, short_step: int, total: int
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    i0 = _index_at(a1, t_int)
    ish = _index_at(a1, short_step)
    treated = a1["treated"].astype(bool)
    practical = treated & (a1["p_hat"][i0] >= 0.05)
    traj_idx = np.flatnonzero(a1["is_traj"].astype(bool) & (a1["step"] >= t_int))

    rows: list[dict[str, Any]] = []
    for ri, seed in enumerate(a1["seed"]):
        for ui, unit in enumerate(a1["unit"]):
            if not practical[ri, ui]:
                continue
            later = [
                int(ti) for ti in traj_idx
                if int(a1["step"][ti]) > t_int and bool(a1["strict_dead"][ti, ri, ui])
            ]
            event = bool(later)
            event_step = int(a1["step"][later[0]]) if event else int(total)
            rows.append(dict(
                seed=int(seed), unit=int(unit),
                utility_nmse=float(a1["utility_nmse"][i0, ri, ui]),
                tau=int(event_step - t_int), censored=not event,
                pre_max_pre=float(a1["pre_pre_max"][ri, ui]),
                pre_max_post=float(a1["pre_max"][i0, ri, ui]),
                p_count_post=int(round(float(a1["p_count"][i0, ri, ui]))),
                strict_dead_short=bool(a1["strict_dead"][ish, ri, ui]),
            ))
    units = pd.DataFrame(rows)
    if len(units):
        units["premax_cell"] = (
            units.groupby("seed", group_keys=False)["pre_max_post"]
            .apply(lambda s: _quantile_codes(s, 5))
            .astype(int)
        )
        units["utility_code"] = (
            units.groupby(["seed", "premax_cell"], group_keys=False)["utility_nmse"]
            .apply(lambda s: _quantile_codes(s, 3))
            .astype(int)
        )
        units["utility_group"] = units.utility_code.map(
            {0: "low", 1: "mid", 2: "high"}
        )
        units = units.sort_values(["seed", "unit"]).reset_index(drop=True)
    else:
        units = pd.DataFrame(columns=[
            "seed", "unit", "utility_nmse", "tau", "censored",
            "pre_max_pre", "pre_max_post", "p_count_post",
            "strict_dead_short", "premax_cell", "utility_code", "utility_group",
        ])

    km = kaplan_meier(
        units.tau.to_numpy(dtype=np.int64),
        ~units.censored.to_numpy(dtype=bool),
    )
    stats = dict(
        n_treated=int(treated.sum()),
        n_practical=int(practical.sum()),
        practical_rate=float(practical.sum() / treated.sum()) if treated.any() else np.nan,
        refreeze_rate=float((~units.censored).mean()) if len(units) else np.nan,
        km_median_tau=km_median(km),
    )
    return units, km, stats


def _endpoint_seed_values(
    traj: pd.DataFrame, arm: str, baseline: str, step: int, column: str
) -> np.ndarray:
    a = traj[(traj.arm == arm) & (traj.step == step)].sort_values("seed")
    b = traj[(traj.arm == baseline) & (traj.step == step)].sort_values("seed")
    if not np.array_equal(a.seed.to_numpy(), b.seed.to_numpy()):
        raise ValueError(f"seed pairing failed for {arm}, {baseline}, step {step}")
    return a[column].to_numpy(dtype=np.float64) - b[column].to_numpy(dtype=np.float64)


def _raw_hash_frame(raw_paths: dict[str, Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for arm, path in sorted(raw_paths.items()):
        with np.load(path, allow_pickle=False) as z:
            rows.append(dict(
                arm=arm, path=str(Path("logs") / path.name), record="FILE",
                dtype="", shape="", nbytes=int(path.stat().st_size),
                sha256=_sha_file(path),
            ))
            for key in sorted(z.files):
                arr = np.ascontiguousarray(z[key])
                rows.append(dict(
                    arm=arm, path=str(Path("logs") / path.name), record=key,
                    dtype=str(arr.dtype), shape=_shape_text(arr.shape),
                    nbytes=int(arr.nbytes), sha256=_sha_bytes(arr.tobytes()),
                ))
    return pd.DataFrame(rows)


def build_frames(
    raw_by_arm: dict[str, dict[str, np.ndarray]],
    raw_paths: dict[str, Path],
    cfg: dict[str, Any],
    runner_meta: dict[str, Any],
    *,
    nonfinite_by_arm: dict[str, dict[str, int]] | None = None,
    smoke: bool,
) -> dict[str, pd.DataFrame]:
    dc = cfg["dynrepair"]
    t_int = int(dc["t_int"])
    total = int(runner_meta["total_steps"])
    short_step = min(total, t_int + int(runner_meta["fine_window"]))
    seeds = np.asarray(raw_by_arm["A0"]["seed"], dtype=np.int64)
    boot = _bootstrap_indices(
        len(seeds), int(dc["bootstrap_n"]), int(dc["bootstrap_seed"])
    )
    delta = float(dc["delta"])

    traj = _trajectory_frame(raw_by_arm)
    utility = _utility_frame(raw_by_arm)
    units, km, ustats = _units_frame(raw_by_arm["A1"], t_int, short_step, total)
    verdict_rows: list[dict[str, Any]] = []

    n_boot = int(dc["bootstrap_n"])
    boot_seed = int(dc["bootstrap_seed"])

    def add_effect(test: str, window: str, comparison: str, values: np.ndarray,
                   margin: float, labels: tuple[str, str, str], note: str) -> str:
        point, lo, hi = paired_mean_ci(values, boot)
        scientific = equivalence_verdict(
            lo, hi, margin, low=labels[0], equivalent=labels[1], high=labels[2]
        )
        verdict_rows.append(dict(
            test=test, window=window, comparison=comparison, estimate=point,
            ci_lo=lo, ci_hi=hi, threshold=margin,
            verdict="SMOKE_ONLY" if smoke else scientific,
            scientific_verdict=scientific,
            precision="INSUFFICIENT" if (hi - lo) / 2 > float(dc["ci_halfwidth_max"]) else "OK",
            rule_order=">".join((*labels, "INCONCLUSIVE")),
            n_boot=n_boot, bootstrap_seed=boot_seed, n_boot_nonfinite=0,
            note=note,
        ))
        return scientific

    for window, step in (("short", short_step), ("long", total)):
        add_effect(
            "O-1", window, "A1-A0",
            _endpoint_seed_values(traj, "A1", "A0", step, "U"),
            delta, ("PERSISTENT", "FOLD", "HARMFUL"),
            "short is primary; A1 is an oracle intervention, not a learning method",
        )
        add_effect(
            "C-1", window, "A1-A2",
            _endpoint_seed_values(traj, "A1", "A2", step, "U"),
            delta, ("SWITCH_SUPERIOR", "SWITCH_SUFFICIENT", "REPLACEMENT_NEEDED"),
            "one-shot, same treated set, condA only",
        )

    numer, denom, retained = _f1_seed_components(units)
    fpoint, flo, fhi, f_invalid_draws = adjusted_rd_ci(numer, denom, boot)
    retention = float(retained / len(units)) if len(units) else 0.0
    g1 = retention < float(dc["f1_valid_retention_min"])
    g3 = (
        not np.isfinite(ustats["practical_rate"])
        or ustats["practical_rate"] < float(dc["practical_revival_min"])
    )
    if np.isfinite(flo) and np.isfinite(fhi):
        if fhi < -float(dc["f1_margin"]):
            fscientific = "PROTECTIVE"
        elif flo >= -float(dc["f1_margin"]) and fhi <= float(dc["f1_margin"]):
            fscientific = "NULL"
        else:
            fscientific = "INCONCLUSIVE"
    else:
        fscientific = "INCONCLUSIVE"
    fshown = "SMOKE_ONLY" if smoke else (
        "DESCRIPTIVE_ONLY" if (g1 or g3) else fscientific
    )
    verdict_rows.append(dict(
        test="F-1", window="short", comparison="utility high-low adjusted RD",
        estimate=fpoint, ci_lo=flo, ci_hi=fhi, threshold=float(dc["f1_margin"]),
        verdict=fshown, scientific_verdict=fscientific,
        precision=(
            "INSUFFICIENT" if np.isfinite(flo) and np.isfinite(fhi)
            and (fhi - flo) / 2 > float(dc["ci_halfwidth_max"]) else "OK"
        ),
        rule_order="PROTECTIVE>NULL>INCONCLUSIVE",
        n_boot=n_boot, bootstrap_seed=boot_seed,
        n_boot_nonfinite=f_invalid_draws,
        note=(
            "G1/G3 descriptive guard applies" if (g1 or g3) else
            "half-step only: p_hat was intervened on; utility was not randomized"
        ),
    ))

    ch_a_scientific = (
        "PASS" if np.isfinite(ustats["refreeze_rate"])
        and ustats["refreeze_rate"] >= float(dc["ch_refreeze_rate_min"])
        and np.isfinite(ustats["km_median_tau"])
        and ustats["km_median_tau"] < float(dc["ch_median_tau_max"])
        else "FAIL"
    )
    ch_a_shown = "SMOKE_ONLY" if smoke else (
        "DESCRIPTIVE_ONLY" if g3 else ch_a_scientific
    )
    verdict_rows.append(dict(
        test="Ch-a", window="long", comparison="A1 practical revivals",
        estimate=ustats["refreeze_rate"], ci_lo=np.nan, ci_hi=np.nan,
        threshold=float(dc["ch_refreeze_rate_min"]), verdict=ch_a_shown,
        scientific_verdict=ch_a_scientific, precision="NOT_APPLICABLE",
        rule_order="PASS>FAIL", n_boot=n_boot, bootstrap_seed=boot_seed,
        n_boot_nonfinite=0,
        note=f"KM median tau={ustats['km_median_tau']}; G3={g3}",
    ))

    chb_results: dict[str, str] = {}
    for column, suffix in (("dead_frac_0_05", "dead"), ("frozen_frac", "frozen")):
        chb_results[suffix] = add_effect(
            f"Ch-b-{suffix}", "long", f"A1-A0 {column}",
            _endpoint_seed_values(traj, "A1", "A0", total, column),
            float(dc["ch_dead_equiv_margin"]),
            ("REJECTED_LOW", "PASS", "REJECTED_HIGH"),
            "dead is primary; frozen is secondary",
        )

    if ch_a_scientific == "PASS" and chb_results["dead"] == "PASS":
        joint = "PASS"
    elif ch_a_scientific == "PASS" or chb_results["dead"] == "PASS":
        joint = "PARTIAL"
    elif chb_results["dead"].startswith("REJECTED"):
        joint = chb_results["dead"]
    else:
        joint = "INCONCLUSIVE"
    verdict_rows.append(dict(
        test="Ch-1", window="long", comparison="Ch-a and Ch-b-dead",
        estimate=np.nan, ci_lo=np.nan, ci_hi=np.nan, threshold=np.nan,
        verdict="SMOKE_ONLY" if smoke else ("DESCRIPTIVE_ONLY" if g3 else joint),
        scientific_verdict=joint, precision="NOT_APPLICABLE",
        rule_order="PASS>PARTIAL>REJECTED_LOW|REJECTED_HIGH>INCONCLUSIVE",
        n_boot=n_boot, bootstrap_seed=boot_seed, n_boot_nonfinite=0,
        note=f"Ch-a={ch_a_scientific}; Ch-b-dead={chb_results['dead']}; G3={g3}",
    ))
    verdict = pd.DataFrame(verdict_rows)

    placebo = pd.DataFrame(runner_meta.get("placebo", []))
    manifest = pd.DataFrame(runner_meta.get("manifest", []))
    runner_sanity = pd.DataFrame(runner_meta.get("sanity", []))
    guard_rows = [
        dict(check="G1", status="TRIGGERED" if g1 else "CLEAR",
             value=retention, threshold=float(dc["f1_valid_retention_min"]),
             detail="F-1 valid-cell retained fraction"),
        dict(check="G2", status="TRIGGERED" if (verdict.precision == "INSUFFICIENT").any() else "CLEAR",
             value=int((verdict.precision == "INSUFFICIENT").sum()),
             threshold=float(dc["ci_halfwidth_max"]), detail="number of imprecise CIs"),
        dict(check="G3", status="TRIGGERED" if g3 else "CLEAR",
             value=ustats["practical_rate"], threshold=float(dc["practical_revival_min"]),
             detail="A1 practical revival rate"),
        dict(check="G4", status=(
            "TRIGGERED" if len(placebo) and
            (placebo["death_rate"].astype(float) > float(dc["placebo_death_guard"])).any()
            else "CLEAR"
        ), value=(
            float(placebo.death_rate.max()) if len(placebo) else np.nan
        ), threshold=float(dc["placebo_death_guard"]),
             detail="maximum seed-level A3-induced death rate"),
        # §5: report non-finite counts, never drop the rows that carry them.
        dict(check="analysis-finite",
             status="PASS" if not (nonfinite_by_arm or {}) else "REPORT",
             value=sum(
                 sum(counts.values()) for counts in (nonfinite_by_arm or {}).values()
             ),
             threshold=np.nan,
             detail=(
                 "non-finite values carried into traj/utility as NaN; "
                 f"by_arm={nonfinite_by_arm or {}}"
             )),
    ]
    sanity = pd.concat([runner_sanity, pd.DataFrame(guard_rows)], ignore_index=True)
    raw_hash = _raw_hash_frame(raw_paths)
    return dict(
        traj=traj, utility=utility, units=units, km=km, verdict=verdict,
        placebo=placebo, manifest=manifest, sanity=sanity, raw_sha256=raw_hash,
    )


def _summary_text(
    frames: dict[str, pd.DataFrame], runner_meta: dict[str, Any], smoke: bool
) -> str:
    verdict = frames["verdict"]
    lines = [
        "# dynrepair_0826 result",
        "",
        "This is a smoke-only implementation check." if smoke else
        "This report applies the preregistered dynamic-repair decision rules.",
        "",
        "A1 is an oracle intervention using the exact current-task loss; it is not a proposed learning method. "
        "A2 is a one-shot reset and does not represent continuous CBP.",
        "",
        "## Verdicts",
        "",
        "| test | window | estimate | 95% CI | verdict | precision |",
        "|---|---|---:|---:|---|---|",
    ]
    for row in verdict.itertuples(index=False):
        estimate = "NA" if pd.isna(row.estimate) else f"{row.estimate:+.6f}"
        ci = (
            "NA" if pd.isna(row.ci_lo) else
            f"[{row.ci_lo:+.6f}, {row.ci_hi:+.6f}]"
        )
        lines.append(
            f"| {row.test} | {row.window} | {estimate} | {ci} | "
            f"{row.verdict} | {row.precision} |"
        )
    lines += [
        "",
        "## Guards and sanity",
        "",
    ]
    for row in frames["sanity"].itertuples(index=False):
        detail = getattr(row, "detail", "")
        lines.append(f"- {row.check}: {row.status} {detail}".rstrip())
    lines += [
        "",
        "## Scope",
        "",
        "- condA, width 100, period 10,000, batch 1, std encoding, snapshot step 500,000 only.",
        "- F-1 remains observational because functional utility was not randomized.",
        "- A3 matches total bias-space displacement, not unit count or per-unit displacement; G4 decides whether it can be read as an undirected control.",
        "- A1_lo / A1_hi are kick-width sensitivity arms: descriptive only, never used by a verdict.",
        f"- source commit: {runner_meta.get('git_hash', 'unknown')}",
        "",
    ]
    diverged = frames["sanity"]
    diverged = diverged[
        (diverged.check == "analysis-finite") & (diverged.status == "REPORT")
    ]
    if len(diverged):
        lines[-1:-1] = [
            "- Non-finite records were kept, not dropped (§5). "
            f"{diverged.detail.iloc[0]}",
        ]
    return "\n".join(lines)


def run_analysis(
    outdir: Path,
    cfg: dict[str, Any],
    runner_meta: dict[str, Any],
    *,
    smoke: bool = False,
) -> dict[str, pd.DataFrame]:
    """Build every aggregate artifact and verify a second in-memory build."""
    outdir = Path(outdir)
    arms = list(cfg["dynrepair"]["arms"])
    raw_paths = {
        arm: outdir / "logs" / f"unit_traj_{arm}.npz" for arm in arms
    }
    missing = [str(path) for path in raw_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing raw trajectories: {missing}")
    loaded = {arm: load_raw(path) for arm, path in raw_paths.items()}
    raw_by_arm = {arm: value[0] for arm, value in loaded.items()}
    nonfinite_by_arm = {
        arm: value[1] for arm, value in loaded.items() if value[1]
    }

    first = build_frames(raw_by_arm, raw_paths, cfg, runner_meta,
                         nonfinite_by_arm=nonfinite_by_arm, smoke=smoke)
    second = build_frames(raw_by_arm, raw_paths, cfg, runner_meta,
                          nonfinite_by_arm=nonfinite_by_arm, smoke=smoke)
    digest_rows = []
    for name in SCIENTIFIC_CSVS:
        key = name[:-4]
        a = _csv_bytes(first[key])
        b = _csv_bytes(second[key])
        if a != b:
            raise RuntimeError(f"analysis determinism failed for {name}")
        digest_rows.append((name, _sha_bytes(a)))
        (outdir / name).write_bytes(a)

    summary = _summary_text(first, runner_meta, smoke)
    (outdir / "summary.md").write_text(summary, encoding="utf-8", newline="\n")
    det_lines = [
        "# Determinism check",
        "",
        "Two independent in-memory analysis builds produced identical canonical CSV bytes.",
        "Per-arm replay checks (S8-<arm>) are listed in sanity.csv and runner_meta.json.",
        "",
        "The cross-run half of S8 is a separate operator step, because it re-executes",
        "the same command from the same commit into a second output directory:",
        "",
        "```",
        "OMP_NUM_THREADS=1 .venv/bin/python -m src.dynrepair --config <cfg> --outdir <dir2>",
        "OMP_NUM_THREADS=1 .venv/bin/python -m src.dynrepair --compare-outdirs <dir1> <dir2>",
        "```",
        "",
        "| artifact | SHA-256 |",
        "|---|---|",
    ]
    det_lines += [f"| {name} | {digest} |" for name, digest in digest_rows]
    (outdir / "determinism_check.md").write_text(
        "\n".join(det_lines) + "\n", encoding="utf-8", newline="\n"
    )
    analysis_meta = dict(
        smoke=bool(smoke),
        canonical_csv_sha256=dict(digest_rows),
        bootstrap_n=int(cfg["dynrepair"]["bootstrap_n"]),
        bootstrap_seed=int(cfg["dynrepair"]["bootstrap_seed"]),
    )
    (outdir / "analysis_meta.json").write_text(
        json.dumps(analysis_meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8", newline="\n",
    )
    return first


def self_test() -> None:
    """Fast tests of the decision, bootstrap and survival primitives."""
    idx = _bootstrap_indices(3, 32, 7)
    p, lo, hi = paired_mean_ci(np.array([-0.2, -0.2, -0.2]), idx)
    assert np.allclose([p, lo, hi], -0.2, rtol=0.0, atol=1e-15)
    assert equivalence_verdict(
        -0.02, -0.014, 0.0134, low="L", equivalent="E", high="H"
    ) == "L"
    assert equivalence_verdict(
        -0.01, 0.01, 0.0134, low="L", equivalent="E", high="H"
    ) == "E"
    assert equivalence_verdict(
        0.02, 0.03, 0.0134, low="L", equivalent="E", high="H"
    ) == "H"
    km = kaplan_meier(np.array([1, 2, 2]), np.array([True, True, False]))
    assert km_median(km) == 2.0
    n = np.array([-1.0, -1.0, -1.0])
    d = np.array([2.0, 2.0, 2.0])
    point, rlo, rhi, n_invalid = adjusted_rd_ci(n, d, idx)
    assert np.allclose([point, rlo, rhi], -0.5, rtol=0.0, atol=1e-15)
    assert n_invalid == 0
    empty_point, empty_lo, empty_hi, empty_invalid = adjusted_rd_ci(
        np.zeros(3), np.zeros(3), idx
    )
    assert np.isnan([empty_point, empty_lo, empty_hi]).all()
    assert empty_invalid == idx.shape[0]


if __name__ == "__main__":
    self_test()
    print("dynrepair analysis self-test: PASS")
