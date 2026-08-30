"""Registered post-hoc reanalysis of the completed 5M MLP2 centering run.

This module only reads the committed ``mlp2_phase1_0829`` task-end table.  It
does not load checkpoints, replay randomness, or update a network.

Run from the repository root::

    .venv/bin/python -m analysis.mlp2_centering_delay_posthoc.analyze
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "specs" / "spec_mlp2_centering_delay_posthoc_0830.md"
SOURCE = ROOT / "results" / "mlp2_phase1_0829"
DEFAULT_OUT = ROOT / "results" / "mlp2_centering_delay_posthoc_0830"
ARMS = ("L2_none", "L2_A1", "L2_Aall")
SEEDS = tuple(range(10))
LAYERS = (1, 2)
BLOCKS = tuple(range(1, 11))
PERIOD = 10_000
TASKS_PER_BLOCK = 50
UNFIT_FLOOR = 1e-23
EPSILON_D = 0.05
EPSILON_U = 0.10
BOOTSTRAP_B = 20_000
BOOTSTRAP_SEED = 20_260_829
DECISION_TOL = 1e-12
COLORS = {
    "L2_none": "#4c566a",
    "L2_A1": "#d08770",
    "L2_Aall": "#5e81ac",
}
LEVEL_METRICS = (
    "strict_dead_frac", "strict_dead", "alive", "eff_rank",
    "eff_rank_per_alive", "dose", "log10_unfit", "unfit",
    "eval_loss_exact",
)
OUTPUT_LEVEL_METRICS = (*LEVEL_METRICS, "log10_mean_unfit")


class PairingInvalid(RuntimeError):
    """The registered paired analysis is not allowed to proceed."""


class SanityFail(RuntimeError):
    """One or more registered input invariants failed."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _pairing_gate(provenance: dict[str, Any]) -> None:
    sanity = provenance.get("sanity") or {}
    s_pair = sanity.get("S_pair") or {}
    s_final = sanity.get("S_pair_final") or {}
    if not (
        s_pair.get("pass_") is True
        and s_final.get("pass_") is True
        and s_final.get("paired_pass") is True
    ):
        raise PairingInvalid("S-pair and S-pair-final must both PASS")


def load_and_validate(
    source: Path = SOURCE,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load the registered arms and enforce every pre-estimate sanity gate."""
    paths = {
        "layer_stats": source / "layer_stats.csv",
        "provenance": source / "provenance.json",
        "config": source / "config_used.yaml",
        "summary": source / "summary.md",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise SanityFail(f"required input missing: {missing}")

    provenance = json.loads(paths["provenance"].read_text(encoding="utf-8"))
    _pairing_gate(provenance)
    with paths["config"].open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    raw = pd.read_csv(paths["layer_stats"])
    data = raw[raw["arm"].isin(ARMS)].copy()
    data = data.sort_values(["arm", "seed", "task", "layer"], kind="mergesort")

    expected_keys = pd.MultiIndex.from_product(
        [ARMS, SEEDS, range(1, 501), LAYERS],
        names=["arm", "seed", "task", "layer"],
    )
    actual_keys = pd.MultiIndex.from_frame(data[["arm", "seed", "task", "layer"]])
    key_set_ok = len(actual_keys) == len(expected_keys) and set(actual_keys) == set(expected_keys)
    unique_ok = not data.duplicated(["arm", "seed", "task", "layer"]).any()

    arms_cfg = {arm["name"]: arm for arm in cfg.get("arms", [])}
    config_ok = bool(
        int(cfg.get("common", {}).get("total_steps", -1)) == 5_000_000
        and list(cfg.get("common", {}).get("seeds", [])) == list(SEEDS)
        and float(cfg.get("common", {}).get("lr_main", np.nan)) == 0.01
        and list(cfg.get("condA", {}).get("T_values", [])) == [PERIOD]
        and int(cfg.get("phase1", {}).get("task_period", -1)) == PERIOD
        and float(cfg.get("phase1", {}).get("unfit_floor", np.nan)) == UNFIT_FLOOR
        and all(arms_cfg.get(arm, {}).get("hidden") == [100, 100] for arm in ARMS)
    )

    centered_expected = {
        ("L2_none", 1): 0, ("L2_none", 2): 0,
        ("L2_A1", 1): 1, ("L2_A1", 2): 0,
        ("L2_Aall", 1): 1, ("L2_Aall", 2): 1,
    }
    centered_ok = all(
        bool((data.loc[(data.arm == arm) & (data.layer == layer), "centered"] == flag).all())
        for (arm, layer), flag in centered_expected.items()
    )

    identity_error = float(np.max(np.abs(
        data["strict_dead_frac"].to_numpy(float)
        - data["strict_dead"].to_numpy(float) / 100.0
    )))
    layer1 = data[data.layer == 1].sort_values(["arm", "seed", "task"])
    layer2 = data[data.layer == 2].sort_values(["arm", "seed", "task"])
    layer_keys_equal = np.array_equal(
        layer1[["arm", "seed", "task"]].to_numpy(),
        layer2[["arm", "seed", "task"]].to_numpy(),
    )
    duplicated_run_metrics = bool(
        layer_keys_equal
        and np.array_equal(layer1["unfit"].to_numpy(), layer2["unfit"].to_numpy())
        and np.array_equal(
            layer1["eval_loss_exact"].to_numpy(),
            layer2["eval_loss_exact"].to_numpy(),
        )
    )
    decision_columns = [
        "strict_dead_frac", "strict_dead", "alive", "unfit", "eval_loss_exact",
    ]
    finite_ok = bool(np.isfinite(data[decision_columns].to_numpy(float)).all())

    checks: dict[str, Any] = {
        "target_rows_30000": len(data) == 30_000,
        "unique_arm_seed_task_layer": unique_ok,
        "complete_registered_grid": key_set_ok,
        "config_registered": config_ok,
        "all_task_end": bool((data["task_end"] == 1).all()),
        "step_matches_task": bool((data["step"] == data["task"] * PERIOD).all()),
        "strict_dead_plus_alive_100": bool(
            ((data["strict_dead"] + data["alive"]) == 100).all()
        ),
        "strict_dead_fraction_identity": identity_error < 1e-12,
        "strict_dead_fraction_max_error": identity_error,
        "centered_flags": centered_ok,
        "decision_columns_finite": finite_ok,
        "strict_dead_fraction_in_unit_interval": bool(
            data["strict_dead_frac"].between(0.0, 1.0).all()
        ),
        "unfit_positive": bool((data["unfit"] > 0.0).all()),
        "layer_run_metrics_bit_equal": duplicated_run_metrics,
        "S_pair": True,
        "S_pair_final": True,
    }
    failed = [key for key, value in checks.items()
              if key != "strict_dead_fraction_max_error" and value is not True]
    if failed:
        raise SanityFail("registered sanity failed: " + ", ".join(failed))
    return data.reset_index(drop=True), cfg, provenance, checks


def block_levels(data: pd.DataFrame) -> pd.DataFrame:
    frame = data.copy()
    frame["block"] = ((frame["task"] - 1) // TASKS_PER_BLOCK + 1).astype(int)
    frame["log10_unfit"] = np.log10(
        np.maximum(frame["unfit"].to_numpy(float), UNFIT_FLOOR)
    )
    grouped = frame.groupby(["arm", "seed", "layer", "block"], sort=True)
    levels = grouped[list(LEVEL_METRICS)].mean().reset_index()
    # Reconciliation only: the original Phase 1 endpoint transformed the
    # arithmetic block mean, whereas this post-hoc spec registers mean(log U).
    levels["log10_mean_unfit"] = np.log10(
        np.maximum(levels["unfit"].to_numpy(float), UNFIT_FLOOR)
    )
    levels["block_name"] = levels["block"].map(lambda value: f"B{int(value):02d}")
    levels["task_start"] = (levels["block"] - 1) * TASKS_PER_BLOCK + 1
    levels["task_end"] = levels["block"] * TASKS_PER_BLOCK
    levels["n_task"] = grouped.size().to_numpy()
    columns = [
        "arm", "seed", "layer", "block", "block_name", "task_start",
        "task_end", "n_task", *OUTPUT_LEVEL_METRICS,
    ]
    return levels[columns].sort_values(
        ["arm", "seed", "layer", "block"], kind="mergesort"
    ).reset_index(drop=True)


def _append_gap_rows(
    rows: list[dict[str, Any]], frame: pd.DataFrame, *, metric: str,
    arm: str, baseline: str, window: str, scope: str, layer: int | None,
) -> None:
    keys = ["seed"]
    arm_values = frame[frame.arm == arm].sort_values(keys)
    base_values = frame[frame.arm == baseline].sort_values(keys)
    if not np.array_equal(arm_values[keys].to_numpy(), base_values[keys].to_numpy()):
        raise SanityFail(f"paired seed mismatch: {arm} vs {baseline}, {window}")
    for a, b in zip(arm_values.itertuples(), base_values.itertuples()):
        av = float(getattr(a, metric))
        bv = float(getattr(b, metric))
        rows.append(dict(
            scope=scope, window=window, metric=metric, arm=arm,
            baseline=baseline, layer="" if layer is None else int(layer),
            seed=int(a.seed), arm_level=av, baseline_level=bv, gap=av - bv,
        ))


def paired_gaps(levels: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    comparisons = (
        ("L2_A1", "L2_none"),
        ("L2_Aall", "L2_none"),
        ("L2_Aall", "L2_A1"),
    )
    for block in BLOCKS:
        window = f"B{block:02d}"
        for layer in LAYERS:
            part = levels[(levels.block == block) & (levels.layer == layer)]
            for arm, baseline in comparisons:
                _append_gap_rows(
                    rows, part, metric="strict_dead_frac", arm=arm,
                    baseline=baseline, window=window, scope="block", layer=layer,
                )
        part = levels[(levels.block == block) & (levels.layer == 1)]
        for arm, baseline in comparisons:
            _append_gap_rows(
                rows, part, metric="log10_unfit", arm=arm,
                baseline=baseline, window=window, scope="block", layer=None,
            )
            _append_gap_rows(
                rows, part, metric="log10_mean_unfit", arm=arm,
                baseline=baseline, window=window,
                scope="block_reconciliation", layer=None,
            )

    # Registered B02 -> B10 closure and requested Aall functional stability.
    for metric, arm, baseline, layer in (
        ("strict_dead_frac", "L2_A1", "L2_none", 2),
        ("log10_unfit", "L2_Aall", "L2_Aall", None),
    ):
        li = 1 if layer is None else layer
        early = levels[(levels.block == 2) & (levels.layer == li) & (levels.arm == arm)]
        late = levels[(levels.block == 10) & (levels.layer == li) & (levels.arm == arm)]
        if baseline != arm:
            early_b = levels[
                (levels.block == 2) & (levels.layer == li) & (levels.arm == baseline)
            ]
            late_b = levels[
                (levels.block == 10) & (levels.layer == li) & (levels.arm == baseline)
            ]
            early_value = early.set_index("seed")[metric] - early_b.set_index("seed")[metric]
            late_value = late.set_index("seed")[metric] - late_b.set_index("seed")[metric]
            values = late_value - early_value
            arm_level = late_value
            base_level = early_value
            out_metric = f"{metric}_gap_change_B10_minus_B02"
        else:
            early_value = early.set_index("seed")[metric]
            late_value = late.set_index("seed")[metric]
            values = late_value - early_value
            arm_level = late_value
            base_level = early_value
            out_metric = f"{metric}_change_B10_minus_B02"
        for seed in SEEDS:
            rows.append(dict(
                scope="change", window="B10-B02", metric=out_metric,
                arm=arm, baseline=baseline, layer="" if layer is None else layer,
                seed=seed, arm_level=float(arm_level.loc[seed]),
                baseline_level=float(base_level.loc[seed]), gap=float(values.loc[seed]),
            ))

    # Exact task 100/500 sensitivity, never used by P1-P4.
    point = data[data.task.isin((100, 500))].copy()
    point["log10_unfit"] = np.log10(
        np.maximum(point["unfit"].to_numpy(float), UNFIT_FLOOR)
    )
    for task in (100, 500):
        window = f"T{task}"
        for layer in LAYERS:
            part = point[(point.task == task) & (point.layer == layer)]
            for arm, baseline in comparisons:
                _append_gap_rows(
                    rows, part, metric="strict_dead_frac", arm=arm,
                    baseline=baseline, window=window, scope="exact_task", layer=layer,
                )
        part = point[(point.task == task) & (point.layer == 1)]
        for arm, baseline in comparisons:
            _append_gap_rows(
                rows, part, metric="log10_unfit", arm=arm,
                baseline=baseline, window=window, scope="exact_task", layer=None,
            )
    return pd.DataFrame(rows).sort_values(
        ["scope", "window", "metric", "arm", "baseline", "layer", "seed"],
        kind="mergesort",
    ).reset_index(drop=True)


def shared_bootstrap_draws(
    *, B: int = BOOTSTRAP_B, seed: int = BOOTSTRAP_SEED,
) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, len(SEEDS), (B, len(SEEDS)))


def estimate(values: np.ndarray, draws: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (len(SEEDS),) or not np.isfinite(values).all():
        raise SanityFail(f"estimate requires ten finite paired seed values, got {values}")
    boot = np.median(values[draws], axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return dict(
        point=float(np.median(values)), ci_lo=float(lo), ci_hi=float(hi),
        n_seed=len(values), n_negative=int(np.sum(values < 0.0)),
        n_zero=int(np.sum(values == 0.0)), n_positive=int(np.sum(values > 0.0)),
        bootstrap_B=len(draws), seed_values=json.dumps(values.tolist()),
    )


def equivalence_status(ci_lo: float, ci_hi: float, point: float, epsilon: float) -> str:
    # CSV arithmetic can put an exact registered boundary at
    # 0.050000000000000044.  This tolerance is far below one unit (0.01) of
    # strict_dead_frac and changes no substantively non-boundary decision.
    if ci_lo >= -epsilon - DECISION_TOL and ci_hi <= epsilon + DECISION_TOL:
        return "EQUIVALENT"
    if -epsilon - DECISION_TOL <= point <= epsilon + DECISION_TOL:
        return "POINT_NEAR"
    return "NOT_EQUIVALENT"


def classify_p1(early: dict[str, Any], closure: dict[str, Any], late: dict[str, Any]) -> str:
    early_protection = early["ci_hi"] < 0.0
    closes = closure["ci_lo"] > 0.0
    equivalent = equivalence_status(
        late["ci_lo"], late["ci_hi"], late["point"], EPSILON_D
    ) == "EQUIVALENT"
    if early_protection and closes and equivalent:
        return "MORPHOLOGICAL_DELAY_AND_CATCHUP"
    if early_protection and closes and late["ci_hi"] < -EPSILON_D:
        return "DURABLE_MORPHOLOGICAL_PROTECTION"
    if early_protection and closes and late["ci_lo"] > EPSILON_D:
        return "LATE_OVERSHOOT"
    if early_protection and not closes:
        return "EARLY_GAP_WITHOUT_CLOSURE"
    if not early_protection:
        return "NO_EARLY_MORPHOLOGICAL_ADVANTAGE"
    return "INCONCLUSIVE_MORPHOLOGY"


def classify_p2(p1: str, function: dict[str, Any]) -> str:
    function_equivalent = equivalence_status(
        function["ci_lo"], function["ci_hi"], function["point"], EPSILON_U
    ) == "EQUIVALENT"
    catches = p1 == "MORPHOLOGICAL_DELAY_AND_CATCHUP"
    if catches and function_equivalent:
        return "DELAY_ONLY_ACROSS_MORPHOLOGY_AND_FUNCTION"
    if catches and function["ci_hi"] < 0.0:
        return "MORPHOLOGICAL_CATCHUP_WITH_DURABLE_FUNCTIONAL_BENEFIT"
    if not catches and function["ci_hi"] < 0.0:
        return "DURABLE_PROTECTION"
    return "INCONCLUSIVE_FUNCTIONAL_KINETICS"


def classify_p3(
    aall_none: dict[str, Any], aall_a1: dict[str, Any], absolute_median: float,
) -> str:
    relative = aall_none["ci_hi"] < -0.10 and aall_a1["ci_hi"] < -0.10
    absolute = absolute_median <= 0.25
    if relative and absolute:
        return "AALL_ABSOLUTE_AND_RELATIVE_LOW"
    if relative:
        return "AALL_RELATIVE_ONLY"
    if absolute:
        return "AALL_ABSOLUTE_ONLY"
    return "AALL_NOT_LOW"


def classify_p4(p1: str, layer1: dict[str, Any]) -> str:
    protected = layer1["ci_hi"] < -0.10
    equivalent = equivalence_status(
        layer1["ci_lo"], layer1["ci_hi"], layer1["point"], EPSILON_D
    ) == "EQUIVALENT"
    worsened = layer1["ci_lo"] > EPSILON_D
    if worsened:
        return "A1_LAYER1_PARADOX"
    if p1 == "MORPHOLOGICAL_DELAY_AND_CATCHUP" and protected:
        return "LAYER2_LOCALIZED_CATCHUP"
    if p1 == "MORPHOLOGICAL_DELAY_AND_CATCHUP" and equivalent:
        return "GLOBAL_CATCHUP"
    return "INCONCLUSIVE_LAYER_LOCALIZATION"


def _values(
    gaps: pd.DataFrame, *, metric: str, arm: str, baseline: str,
    window: str, layer: int | None,
) -> np.ndarray:
    match = (
        (gaps.metric == metric) & (gaps.arm == arm)
        & (gaps.baseline == baseline) & (gaps.window == window)
    )
    if layer is None:
        match &= gaps.layer.astype(str).eq("")
    else:
        match &= gaps.layer.astype(str).eq(str(layer))
    part = gaps.loc[match].sort_values("seed")
    if part.seed.tolist() != list(SEEDS):
        raise SanityFail(
            f"gap lookup failed: {metric}, {arm}, {baseline}, {window}, layer={layer}"
        )
    return part.gap.to_numpy(float)


def _estimate_row(
    endpoint: str, component: str, values: np.ndarray, draws: np.ndarray, *,
    metric: str, arm: str, baseline: str, layer: int | None, window: str,
    epsilon: float | None = None, verdict: str = "REPORT_ONLY",
) -> dict[str, Any]:
    result = estimate(values, draws)
    status = ""
    if epsilon is not None:
        status = equivalence_status(
            result["ci_lo"], result["ci_hi"], result["point"], epsilon
        )
    return dict(
        endpoint=endpoint, component=component, metric=metric, arm=arm,
        baseline=baseline, layer="" if layer is None else layer, window=window,
        epsilon="" if epsilon is None else epsilon, component_status=status,
        verdict=verdict, **result,
    )


def build_verdicts(
    levels: pd.DataFrame, gaps: pd.DataFrame, draws: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def gap_est(metric: str, arm: str, baseline: str, window: str,
                layer: int | None) -> tuple[np.ndarray, dict[str, Any]]:
        values = _values(
            gaps, metric=metric, arm=arm, baseline=baseline,
            window=window, layer=layer,
        )
        return values, estimate(values, draws)

    early_values, early = gap_est("strict_dead_frac", "L2_A1", "L2_none", "B02", 2)
    late_values, late = gap_est("strict_dead_frac", "L2_A1", "L2_none", "B10", 2)
    closure_values, closure = gap_est(
        "strict_dead_frac_gap_change_B10_minus_B02",
        "L2_A1", "L2_none", "B10-B02", 2,
    )
    p1 = classify_p1(early, closure, late)
    rows += [
        _estimate_row(
            "P1", "early_gap", early_values, draws, metric="strict_dead_frac",
            arm="L2_A1", baseline="L2_none", layer=2, window="B02",
            epsilon=EPSILON_D, verdict=p1,
        ),
        _estimate_row(
            "P1", "late_gap", late_values, draws, metric="strict_dead_frac",
            arm="L2_A1", baseline="L2_none", layer=2, window="B10",
            epsilon=EPSILON_D, verdict=p1,
        ),
        _estimate_row(
            "P1", "gap_closure", closure_values, draws,
            metric="strict_dead_frac_gap_change_B10_minus_B02",
            arm="L2_A1", baseline="L2_none", layer=2, window="B10-B02",
            verdict=p1,
        ),
    ]

    # Reproduce the transform order used by the original Phase 1 report.  It
    # is not substituted for the newly registered mean(log10 U) endpoint.
    for arm in ("L2_A1", "L2_Aall"):
        legacy_values, _ = gap_est(
            "log10_mean_unfit", arm, "L2_none", "B10", None
        )
        rows.append(_estimate_row(
            "RECONCILIATION_REPORT_ONLY", f"{arm}_phase1_transform",
            legacy_values, draws, metric="log10_mean_unfit", arm=arm,
            baseline="L2_none", layer=None, window="B10",
        ))

    trajectory_status: dict[int, str] = {}
    for block in BLOCKS:
        values, result = gap_est(
            "strict_dead_frac", "L2_A1", "L2_none", f"B{block:02d}", 2
        )
        trajectory_status[block] = equivalence_status(
            result["ci_lo"], result["ci_hi"], result["point"], EPSILON_D
        )
        rows.append(_estimate_row(
            "P1_TRAJECTORY", f"B{block:02d}_gap", values, draws,
            metric="strict_dead_frac", arm="L2_A1", baseline="L2_none",
            layer=2, window=f"B{block:02d}", epsilon=EPSILON_D,
        ))
    catchup = "NOT_OBSERVED_BY_5M"
    for block in range(2, 10):
        if trajectory_status[block] == trajectory_status[block + 1] == "EQUIVALENT":
            catchup = f"CATCHUP_AT_B{block:02d}_END_{block * 500_000}"
            break
    if catchup == "NOT_OBSERVED_BY_5M" and trajectory_status[10] == "EQUIVALENT":
        catchup = "CATCHUP_BY_5M_SINGLE_BLOCK"

    function_values, function = gap_est(
        "log10_unfit", "L2_A1", "L2_none", "B10", None
    )
    p2 = classify_p2(p1, function)
    rows.append(_estimate_row(
        "P2", "functional_gap", function_values, draws, metric="log10_unfit",
        arm="L2_A1", baseline="L2_none", layer=None, window="B10",
        epsilon=EPSILON_U, verdict=p2,
    ))

    aall_none_values, aall_none = gap_est(
        "strict_dead_frac", "L2_Aall", "L2_none", "B10", 2
    )
    aall_a1_values, aall_a1 = gap_est(
        "strict_dead_frac", "L2_Aall", "L2_A1", "B10", 2
    )
    absolute_values = levels[
        (levels.arm == "L2_Aall") & (levels.layer == 2) & (levels.block == 10)
    ].sort_values("seed").strict_dead_frac.to_numpy(float)
    absolute = estimate(absolute_values, draws)
    p3 = classify_p3(aall_none, aall_a1, absolute["point"])
    rows += [
        _estimate_row(
            "P3", "aall_vs_none", aall_none_values, draws,
            metric="strict_dead_frac", arm="L2_Aall", baseline="L2_none",
            layer=2, window="B10", epsilon=EPSILON_D, verdict=p3,
        ),
        _estimate_row(
            "P3", "aall_vs_a1", aall_a1_values, draws,
            metric="strict_dead_frac", arm="L2_Aall", baseline="L2_A1",
            layer=2, window="B10", epsilon=EPSILON_D, verdict=p3,
        ),
        dict(
            endpoint="P3", component="aall_absolute_level", metric="strict_dead_frac",
            arm="L2_Aall", baseline="", layer=2, window="B10", epsilon=0.25,
            component_status="ABSOLUTE_LOW" if absolute["point"] <= 0.25 else "NOT_ABSOLUTE_LOW",
            verdict=p3, **absolute,
        ),
    ]

    layer1_values, layer1 = gap_est(
        "strict_dead_frac", "L2_A1", "L2_none", "B10", 1
    )
    p4 = classify_p4(p1, layer1)
    rows.append(_estimate_row(
        "P4", "layer1_gap", layer1_values, draws, metric="strict_dead_frac",
        arm="L2_A1", baseline="L2_none", layer=1, window="B10",
        epsilon=EPSILON_D, verdict=p4,
    ))

    # Requested direct check of whether Aall retains evaluation performance.
    # It is deliberately REPORT_ONLY because the registered P3 is morphological.
    aall_u_values, aall_u = gap_est(
        "log10_unfit", "L2_Aall", "L2_none", "B10", None
    )
    if aall_u["ci_hi"] < 0.0:
        p3f_relative = "AALL_DURABLE_FUNCTIONAL_BENEFIT"
    elif equivalence_status(
        aall_u["ci_lo"], aall_u["ci_hi"], aall_u["point"], EPSILON_U
    ) == "EQUIVALENT":
        p3f_relative = "AALL_FUNCTION_EQUIVALENT_TO_BASELINE"
    elif aall_u["ci_lo"] > 0.0:
        p3f_relative = "AALL_FUNCTION_WORSE_THAN_BASELINE"
    else:
        p3f_relative = "INCONCLUSIVE_AALL_FUNCTION"
    aall_change_values, aall_change = gap_est(
        "log10_unfit_change_B10_minus_B02",
        "L2_Aall", "L2_Aall", "B10-B02", None,
    )
    stability = equivalence_status(
        aall_change["ci_lo"], aall_change["ci_hi"],
        aall_change["point"], EPSILON_U,
    )
    if stability == "EQUIVALENT":
        p3f_stability = "AALL_FUNCTION_STABLE_B02_TO_B10"
    elif aall_change["ci_lo"] > EPSILON_U:
        p3f_stability = "AALL_FUNCTION_DETERIORATED_BY_5M"
    elif aall_change["ci_hi"] < -EPSILON_U:
        p3f_stability = "AALL_FUNCTION_IMPROVED_BY_5M"
    else:
        p3f_stability = "INCONCLUSIVE_AALL_FUNCTION_STABILITY"
    rows += [
        _estimate_row(
            "P3F_REPORT_ONLY", "aall_function_vs_none", aall_u_values, draws,
            metric="log10_unfit", arm="L2_Aall", baseline="L2_none",
            layer=None, window="B10", epsilon=EPSILON_U,
            verdict=p3f_relative,
        ),
        _estimate_row(
            "P3F_REPORT_ONLY", "aall_function_B10_minus_B02",
            aall_change_values, draws,
            metric="log10_unfit_change_B10_minus_B02",
            arm="L2_Aall", baseline="L2_Aall", layer=None,
            window="B10-B02", epsilon=EPSILON_U, verdict=p3f_stability,
        ),
    ]

    # Registered exact-task sensitivities.
    for task in (100, 500):
        window = f"T{task}"
        for metric, arm, baseline, layer in (
            ("strict_dead_frac", "L2_A1", "L2_none", 2),
            ("strict_dead_frac", "L2_Aall", "L2_none", 2),
            ("log10_unfit", "L2_A1", "L2_none", None),
            ("log10_unfit", "L2_Aall", "L2_none", None),
        ):
            values, _ = gap_est(metric, arm, baseline, window, layer)
            rows.append(_estimate_row(
                "SENSITIVITY", f"{arm}_vs_{baseline}_{metric}_{window}",
                values, draws, metric=metric, arm=arm, baseline=baseline,
                layer=layer, window=window,
                epsilon=EPSILON_D if metric == "strict_dead_frac" else EPSILON_U,
            ))

    verdict = pd.DataFrame(rows)
    details = dict(
        P1=p1, catchup_time=catchup, P2=p2, P3=p3, P4=p4,
        P3F_relative=p3f_relative, P3F_stability=p3f_stability,
    )
    return verdict, details


def _fmt(row: pd.Series) -> str:
    return f"{row.point:.4g} [{row.ci_lo:.4g}, {row.ci_hi:.4g}]"


def write_summary(
    levels: pd.DataFrame, verdict: pd.DataFrame, details: dict[str, Any], out: Path,
) -> None:
    def component(endpoint: str, name: str) -> pd.Series:
        return verdict[(verdict.endpoint == endpoint) & (verdict.component == name)].iloc[0]

    p1e = component("P1", "early_gap")
    p1l = component("P1", "late_gap")
    p1c = component("P1", "gap_closure")
    p2 = component("P2", "functional_gap")
    p3n = component("P3", "aall_vs_none")
    p3a = component("P3", "aall_vs_a1")
    p3abs = component("P3", "aall_absolute_level")
    p4 = component("P4", "layer1_gap")
    p3fu = component("P3F_REPORT_ONLY", "aall_function_vs_none")
    p3ft = component("P3F_REPORT_ONLY", "aall_function_B10_minus_B02")
    old_a1 = component("RECONCILIATION_REPORT_ONLY", "L2_A1_phase1_transform")
    old_aall = component("RECONCILIATION_REPORT_ONLY", "L2_Aall_phase1_transform")

    late = levels[levels.block == 10]
    level_rows = []
    for arm in ARMS:
        d2 = late[(late.arm == arm) & (late.layer == 2)].strict_dead_frac.median()
        u = late[(late.arm == arm) & (late.layer == 1)].log10_unfit.median()
        level_rows.append(f"| {arm} | {d2:.4f} | {u:.4f} | {10 ** u:.5g} |")

    lines = [
        "# MLP2 5M centering-delay post-hoc reanalysis", "",
        "> Existing `mlp2_phase1_0829` logs only; no new training or checkpoint replay.",
        "> This is a post-hoc reanalysis registration, not a preregistration or independent replication.", "",
        "## Main verdicts", "",
        f"- **P1 morphology:** `{details['P1']}`; catch-up timing: `{details['catchup_time']}`.",
        f"- **P2 morphology + function:** `{details['P2']}`.",
        f"- **P3 Aall low state:** `{details['P3']}`.",
        f"- **P4 localization:** `{details['P4']}`.", "",
        "## Registered components", "",
        "| component | paired median gap [95% percentile CI] | interpretation |",
        "|---|---:|---|",
        f"| A1−none, layer 2, B02 strict_dead_frac | {_fmt(p1e)} | early protection if CI upper < 0 |",
        f"| A1−none, layer 2, B10 strict_dead_frac | {_fmt(p1l)} | equivalent only if CI within ±0.05 |",
        f"| B10 gap−B02 gap | {_fmt(p1c)} | closure if CI lower > 0 |",
        f"| A1−none, B10 log10(unfit) | {_fmt(p2)} | negative means functional benefit |",
        f"| Aall−none, layer 2, B10 strict_dead_frac | {_fmt(p3n)} | P3 relative component |",
        f"| Aall−A1, layer 2, B10 strict_dead_frac | {_fmt(p3a)} | P3 relative component |",
        f"| Aall layer 2 B10 absolute strict_dead_frac | {_fmt(p3abs)} | absolute-low cutoff 0.25 |",
        f"| A1−none, layer 1, B10 strict_dead_frac | {_fmt(p4)} | P4 negative control |", "",
        "## Aall evaluation check (requested, REPORT_ONLY)", "",
        "The registered P3 is morphological.  The following functional checks answer the requested",
        "Aall evaluation question without changing P1–P4 after registration.", "",
        f"- B10 Aall−none log10(unfit): **{_fmt(p3fu)}** — `{details['P3F_relative']}`.",
        f"- Aall B10−B02 log10(unfit): **{_fmt(p3ft)}** — `{details['P3F_stability']}`.", "",
        "## Reconciliation with the original Phase 1 summary", "",
        "The original report used `log10(mean unfit)`; this post-hoc spec registers",
        "`mean(log10 unfit)`.  The former weights rare high-unfit tasks much more strongly.",
        "It is reproduced here only to verify that the source data and transform order agree:", "",
        f"- Phase-1 transform, A1−none B10: **{_fmt(old_a1)}**.",
        f"- Phase-1 transform, Aall−none B10: **{_fmt(old_aall)}**.",
        "- These rows do not replace P2 or P3F and do not change any verdict.", "",
        "## B10 levels (tasks 451–500)", "",
        "`geometric unfit = 10^(mean log10 unfit)`; the table reports the median seed level.", "",
        "| arm | layer-2 strict_dead_frac | mean log10(unfit) | geometric unfit |",
        "|---|---:|---:|---:|", *level_rows, "",
        "## Limits", "",
        "- `strict_dead` is a current-task support label, not permanent unit-ID death or an absorption time.",
        "- Morphological catch-up does not imply functional catch-up; P1 and P2 remain separate.",
        "- P3F is explicitly report-only because the registered P3 did not include an Aall functional verdict.",
        "- Conclusions are limited to paired condA, depth 2, width 100, T=10,000, batch 1, lr=0.01, and 5M steps.",
        "",
    ]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def make_figure(levels: pd.DataFrame, verdict: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), sharex=True)
    panel_specs = (
        (axes[0, 0], 1, "strict_dead_frac", "Layer 1 strict_dead_frac"),
        (axes[0, 1], 2, "strict_dead_frac", "Layer 2 strict_dead_frac"),
        (axes[1, 0], 1, "log10_unfit", "Network function: mean log10(unfit)"),
    )
    for ax, layer, metric, title in panel_specs:
        for arm in ARMS:
            part = levels[(levels.arm == arm) & (levels.layer == layer)]
            pivot = part.pivot(index="block", columns="seed", values=metric)
            for seed in pivot.columns:
                ax.plot(pivot.index, pivot[seed], color=COLORS[arm], alpha=0.12, lw=0.7)
            ax.plot(pivot.index, pivot.median(axis=1), color=COLORS[arm],
                    lw=2.2, label=arm)
        ax.set_title(title)
        ax.grid(alpha=0.2)
    axes[0, 0].set_ylabel("fraction")
    axes[0, 1].set_ylabel("fraction")
    axes[1, 0].set_ylabel("log10 unit")
    axes[0, 0].legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    trajectory = verdict[verdict.endpoint == "P1_TRAJECTORY"].copy()
    trajectory["block"] = trajectory.window.str[1:].astype(int)
    trajectory = trajectory.sort_values("block")
    ax.axhspan(-EPSILON_D, EPSILON_D, color="#a3be8c", alpha=0.18,
               label="equivalence band")
    ax.axhline(0.0, color="#777777", lw=0.8)
    ax.errorbar(
        trajectory.block, trajectory.point,
        yerr=np.vstack([
            trajectory.point - trajectory.ci_lo,
            trajectory.ci_hi - trajectory.point,
        ]),
        marker="o", color=COLORS["L2_A1"], capsize=3,
        label="A1−none, layer 2",
    )
    ax.set_title("Paired morphology gap (95% CI)")
    ax.set_ylabel("strict_dead_frac gap")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.2)
    for axis in axes[1]:
        axis.set_xlabel("50-task block")
        axis.set_xticks(BLOCKS)
    fig.suptitle("MLP2 centering delay: 5M post-hoc block trajectories")
    fig.tight_layout()
    fig.savefig(out / "centering_delay_layers.png", dpi=180)
    fig.savefig(out / "centering_delay_layers.pdf")
    plt.close(fig)


def write_provenance(
    source: Path, out: Path, checks: dict[str, Any], details: dict[str, Any],
) -> None:
    input_names = ("layer_stats.csv", "provenance.json", "config_used.yaml", "summary.md")
    implementation = Path(__file__).resolve()
    implementation_commit = _git("log", "-1", "--format=%H", "--", str(implementation))
    spec_commit = _git("log", "-1", "--format=%H", "--", str(SPEC))
    output_names = (
        "block_levels.csv", "paired_gaps.csv", "verdict.csv", "summary.md",
        "centering_delay_layers.png", "centering_delay_layers.pdf",
    )
    payload = dict(
        analysis="mlp2_centering_delay_posthoc_0830",
        status="POST_HOC_REANALYSIS",
        source=str(source.relative_to(ROOT)),
        spec=str(SPEC.relative_to(ROOT)),
        spec_commit=spec_commit or "UNKNOWN",
        implementation=str(implementation.relative_to(ROOT)),
        implementation_commit=implementation_commit or "UNCOMMITTED",
        implementation_sha256=sha256(implementation),
        git_head=_git("rev-parse", "HEAD") or "UNKNOWN",
        git_status=_git("status", "--short").splitlines(),
        inputs={name: sha256(source / name) for name in input_names},
        outputs={name: sha256(out / name) for name in output_names},
        bootstrap=dict(
            method="paired seed-cluster percentile",
            B=BOOTSTRAP_B,
            seed=BOOTSTRAP_SEED,
            statistic="median paired gap",
        ),
        constants=dict(
            epsilon_D=EPSILON_D, epsilon_U=EPSILON_U,
            unfit_floor=UNFIT_FLOOR, tasks_per_block=TASKS_PER_BLOCK,
        ),
        sanity=checks,
        verdicts=details,
        environment=dict(
            python=sys.version, platform=platform.platform(),
            numpy=np.__version__, pandas=pd.__version__,
        ),
    )
    (out / "provenance.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _write_failure(out: Path, status: str, detail: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dict(endpoint="GATE", verdict=status, detail=detail)]).to_csv(
        out / "verdict.csv", index=False, lineterminator="\n"
    )


def run(source: Path = SOURCE, out: Path = DEFAULT_OUT) -> dict[str, Any]:
    try:
        data, _cfg, _provenance, checks = load_and_validate(source)
    except PairingInvalid as error:
        _write_failure(out, "PAIRING_INVALID", str(error))
        raise
    except SanityFail as error:
        _write_failure(out, "SANITY_FAIL", str(error))
        raise

    out.mkdir(parents=True, exist_ok=True)
    levels = block_levels(data)
    gaps = paired_gaps(levels, data)
    draws = shared_bootstrap_draws()
    verdict, details = build_verdicts(levels, gaps, draws)
    levels.to_csv(
        out / "block_levels.csv", index=False, lineterminator="\n", float_format="%.17g"
    )
    gaps.to_csv(
        out / "paired_gaps.csv", index=False, lineterminator="\n", float_format="%.17g"
    )
    verdict.to_csv(
        out / "verdict.csv", index=False, lineterminator="\n", float_format="%.17g"
    )
    write_summary(levels, verdict, details, out)
    make_figure(levels, verdict, out)
    write_provenance(source, out, checks, details)
    return details


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    details = run(args.source, args.out)
    print(json.dumps(details, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
