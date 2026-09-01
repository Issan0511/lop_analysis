"""P2b width-5 gate experiment: paired seed-sign tests against LIN5.

The training harness, eight arms, RNG allocation, and exact-support logger are
reused from src.width5_gate_0901. P2b changes only the registered endpoint and
preflight semantics:

* S-cap becomes non-blocking floor characterization;
* LIN5 finiteness is a blocking reference sanity;
* G0 counts, seed by seed, whether each width-5 nonlinear arm has larger
  terminal-window unfit than LIN5; and
* G0b reports sign crossings from below LIN5 early to above LIN5 at 5M.

Issa requested implementation before entering the three new predictions. The
predictions and repo spec are now frozen, and every result-bearing CLI stage
checks their exact values before running.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from . import width5_gate_0901 as base
from .common import ROOT, load_config, pick_device
from .dose_const_5m import clopper_pearson
from .mlp2_phase0 import _sha_file, require_omp, write_csv
from .mlp2_phase0b import _complete_arm_logs, _window_indices
from .mlp2_phase1 import NUMERIC_DIVERGENCE


EXPERIMENT = "width5_gate_b_0901"
PREFLIGHT_DIR = "results/_preflight_width5_gate_b_0901"
SMOKE_DIR = "results/_smoke_width5_gate_b_0901"
ARM_ORDER = base.ARM_ORDER
WIDTH5_ARMS = base.WIDTH5_ARMS
WIDTH100_ARMS = base.WIDTH100_ARMS
REGISTERED_ARMS = base.REGISTERED_ARMS
RUN_KEYS = base.RUN_KEYS
LAYER_KEYS = base.LAYER_KEYS
SMOKE_STEPS = base.SMOKE_STEPS
SMOKE_SEEDS = base.SMOKE_SEEDS

_arm = base._arm
_run_arm = base._run_arm
_load_arm = base._load_arm
_window = base._window
_pair_check_final = base._pair_check_final
_collect_divergences = base._collect_divergences
_ci = base._ci

_BASE_PREREGISTRATION = {
    "decisions_complete": True,
    "predictions_confirmed": True,
    "frozen": True,
    "repo_spec_committed": True,
    "execution_authorized": True,
    "implementation_before_predictions_authorized": True,
    "repo_spec_sha256":
        "0710b8919cc411c98a7375bad658e71dafa9e5486b089966b55dbe4c4484d651",
    "prediction_provenance":
        "draft_candidates_proposed_first_then_approved_by_Issa",
    "predictions": {
        "R5_n_onset_5m": "20_of_20",
        "LR5_n_onset_5m": "unknown",
        "E5_n_onset_5m": "unknown",
        "LR5_vs_LIN5_log10U_position": "above",
        "LR5_submerged_fraction_5m": "same_as_w100_0.63_to_0.67",
        "E5_submerged_fraction_5m": "same_as_w100_0.36_to_0.45",
        "Rc_LR5_vs_LIN5": "lower",
        "failure_cause_if_prediction_misses": "unknown",
    },
    "level_equivalence_resolution_confirmed": True,
    "generator_offset_confirmed": True,
    "bootstrap_seed_confirmed": True,
}

_CARRY_OVER = {
    "LR5_vs_LIN5_log10U_position": "above",
    "LR5_submerged_fraction_5m": "same_as_w100_0.63_to_0.67",
    "E5_submerged_fraction_5m": "same_as_w100_0.36_to_0.45",
    "Rc_LR5_vs_LIN5": "lower",
    "failure_cause_if_prediction_misses": "unknown",
}
_NEW_PREDICTION_KEYS = (
    "R5_k_above_LIN5_5m",
    "LR5_k_above_LIN5_5m",
    "E5_k_above_LIN5_5m",
)
_FROZEN_NEW_PREDICTIONS = {
    "R5_k_above_LIN5_5m": "at_least_15_of_20",
    "LR5_k_above_LIN5_5m": "at_least_15_of_20",
    "E5_k_above_LIN5_5m": "5_to_15_of_20",
}


def preregistration_missing(cfg: dict) -> list[str]:
    pre = cfg["preregistration"]
    missing = []
    for key in ("new_predictions_confirmed", "frozen",
                "repo_spec_committed", "execution_authorized"):
        if pre.get(key) is not True:
            missing.append(f"preregistration.{key}")
    return missing


def _base_structural_proxy(cfg: dict) -> dict:
    """Map P2b to P2's frozen validator without changing the real config."""
    proxy = copy.deepcopy(cfg)
    proxy["spec"] = "specs/spec_width5_gate_0901.md"
    proxy["preregistration"] = copy.deepcopy(_BASE_PREREGISTRATION)
    return proxy


def validate_config(cfg: dict, *, stage: str) -> None:
    """Validate unchanged P2 structure plus the P2b endpoint registration."""
    if stage not in {"implementation", "preflight", "smoke", "full", "analyze"}:
        raise ValueError(f"unknown stage {stage!r}")
    base.validate_config(_base_structural_proxy(cfg), stage="implementation")
    B, S, pre = cfg["width5_gate_b"], cfg["sanity"], cfg["preregistration"]
    expected_b = {
        "primary_verdict": "G0_seed_sign_vs_LIN5",
        "comparison_arms": ["R5", "LR5", "E5"],
        "phenomenon_arms": ["LR5", "E5"],
        "linear_baseline": "LIN5",
        "comparison_operator": "strict_greater",
        "ties_policy": "count_as_not_above_keep_in_n",
        "cp_alpha": 0.05,
        "null_probability": 0.5,
        "tight_band": [0.20, 0.80],
        "crossing_registered_secondary": True,
        "crossing_early_better_min_count": 15,
        "crossing_early_operator": "strict_less",
        "crossing_late_operator": "strict_greater",
        "level_report_only": True,
        "width_report_only": True,
        "mechanism_report_only": True,
    }
    if B != expected_b:
        raise ValueError("P2b G0/G0b/G1/G2/G3 design changed")
    if (list(S["s_floor_char_arms"]) != ["R5", "LIN5"]
            or list(S["s_floor_char_tasks"]) != [2, 11]
            or S["s_floor_char_rule"]
            != "report_per_seed_early_window_minima_no_gate"
            or list(S["s_lin_ref_preflight_tasks"]) != [2, 11]
            or list(S["s_lin_ref_full_windows"]) != ["early", "5M"]):
        raise ValueError("P2b sanity design changed")
    if (pre.get("decisions_complete") is not True
            or pre.get("carry_over_confirmed") is not True
            or pre.get("implementation_before_predictions_authorized") is not True
            or pre.get("carry_over_predictions") != _CARRY_OVER):
        raise ValueError("P2 carry-over decisions or predictions changed")
    predictions = pre.get("new_predictions")
    if not isinstance(predictions, dict) or tuple(predictions) != _NEW_PREDICTION_KEYS:
        raise ValueError("P2b prediction fields changed")
    if pre.get("new_predictions_confirmed") is not True:
        if any(predictions[key] is not None for key in _NEW_PREDICTION_KEYS):
            raise ValueError("unconfirmed P2b predictions must remain empty")
        if pre.get("prediction_provenance") != "pending_Issa_entry":
            raise ValueError("unconfirmed prediction provenance changed")
    elif (predictions != _FROZEN_NEW_PREDICTIONS
          or pre.get("prediction_provenance")
          != "draft_candidates_proposed_first_then_approved_by_Issa"):
        raise ValueError("Issa's three frozen P2b predictions changed")
    if stage != "implementation":
        missing = preregistration_missing(cfg)
        if missing:
            raise ValueError("width5_gate_b preregistration is not frozen: "
                             + ", ".join(missing))
        spec_path = Path(ROOT) / str(cfg["spec"])
        if (not spec_path.is_file()
                or _sha_file(spec_path) != pre.get("repo_spec_sha256")):
            raise ValueError("frozen P2b repo spec is missing or changed")


def _floor_characterization(cfg: dict, device: str, outdir: Path) -> dict:
    """Run and report the predecessor's floor calibration without gating."""
    S, P = cfg["sanity"], cfg["phase1"]
    tasks = [int(v) for v in S["s_floor_char_tasks"]]
    total = tasks[1] * int(P["task_period"])
    seeds = [int(v) for v in cfg["common"]["seeds"]]
    rows = []
    for arm in S["s_floor_char_arms"]:
        result = _run_arm(cfg, str(arm), device, outdir, seeds, total)
        if result["status"] != "COMPLETE":
            rows.append(dict(
                arm=arm, status=result["status"], finite=False,
                per_seed_min=[], min=None, median=None, max=None))
            continue
        minima = []
        for seed in seeds:
            with np.load(outdir / "logs" / f"{arm}_seed{seed}.npz",
                         allow_pickle=False) as z:
                idx = _window_indices(z["step"], int(P["task_period"]), tasks)
                minima.append(float(np.min(np.asarray(z["unfit"])[idx])))
        arr = np.asarray(minima, dtype=np.float64)
        rows.append(dict(
            arm=arm, status="COMPLETE", finite=bool(np.isfinite(arr).all()),
            per_seed_min=minima, min=float(np.min(arr)),
            median=float(np.median(arr)), max=float(np.max(arr)),
            below_predecessor_threshold=int(
                np.sum(arr < float(P["onset_threshold"])))))
    return dict(pass_=True, gate=False, tasks=tasks,
                rule=S["s_floor_char_rule"], rows=rows,
                note="characterization_only_never_blocks_main_run")


def _lin_ref_preflight(cfg: dict, floor_dir: Path) -> dict:
    """Blocking early-window finiteness check for the comparison baseline."""
    P, S = cfg["phase1"], cfg["sanity"]
    seeds = [int(v) for v in cfg["common"]["seeds"]]
    tasks = [int(v) for v in S["s_lin_ref_preflight_tasks"]]
    missing, nonfinite, values = [], [], []
    for seed in seeds:
        path = floor_dir / "logs" / f"LIN5_seed{seed}.npz"
        if not path.exists():
            missing.append(seed)
            continue
        with np.load(path, allow_pickle=False) as z:
            idx = _window_indices(z["step"], int(P["task_period"]), tasks)
            arr = np.asarray(z["unfit"], dtype=np.float64)[idx]
        if not np.isfinite(arr).all():
            nonfinite.append(seed)
        values.append(dict(seed=seed, finite=bool(np.isfinite(arr).all()),
                           window_mean=(float(np.mean(arr))
                                        if np.isfinite(arr).all() else None)))
    return dict(pass_=not missing and not nonfinite, tasks=tasks,
                missing_seeds=missing, nonfinite_seeds=nonfinite,
                n_valid=len(seeds) - len(set(missing + nonfinite)),
                per_seed=values)


def preflight(cfg: dict, device: str, outdir: Path) -> dict:
    """Re-run reusable checks and characterize, rather than gate, floor."""
    outdir.mkdir(parents=True, exist_ok=True)
    checks = {
        "S_omp": require_omp(cfg),
        "S_offset": base._s_offset(cfg),
        "S_linear": base._s_linear(cfg),
        "S_mobility": base._s_mobility(cfg),
        "S_rank": base._s_rank(cfg),
        "S_floor": base._s_floor(cfg),
        "S_initial_pairing": base._s_initial_pairing(cfg),
    }
    checks["S0prime"] = base._s0_replay(cfg, device, outdir / "s0prime")
    floor = _floor_characterization(cfg, device, outdir / "floor_char")
    checks["S_floor_char"] = floor
    checks["S_lin_ref"] = _lin_ref_preflight(cfg, outdir / "floor_char")
    gating = {name: value for name, value in checks.items()
              if name != "S_floor_char"}
    result = dict(pass_=bool(all(v.get("pass_") for v in gating.values())),
                  **checks)
    (outdir / "preflight.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")
    for name, value in checks.items():
        label = ("CHARACTERIZED" if name == "S_floor_char"
                 else "PASS" if value.get("pass_") else "FAIL")
        print(f"[{name}] {label}", flush=True)
    if not result["pass_"]:
        failed = [k for k, v in gating.items() if not v.get("pass_")]
        raise RuntimeError(f"preflight failed: {failed}")
    return result


def classify_seed_sign(arm: str, successes: int, trials: int,
                       *, alpha: float = 0.05,
                       tight_band: tuple[float, float] = (0.20, 0.80)) -> dict:
    """Apply the registered exact-binomial label to one arm."""
    if trials <= 0:
        return dict(arm=arm, k=int(successes), n=int(trials),
                    cp95_lo=None, cp95_hi=None,
                    status=f"{arm}_INCONCLUSIVE_NO_VALID_SEEDS")
    lo, hi = clopper_pearson(successes, trials, alpha)
    if lo > 0.5:
        status = f"{arm}_ABOVE_LINEAR"
    elif hi < 0.5:
        status = f"{arm}_BELOW_LINEAR"
    elif lo >= tight_band[0] and hi <= tight_band[1]:
        status = f"{arm}_NOT_SEPARATED_TIGHT"
    else:
        status = f"{arm}_INCONCLUSIVE_WIDE"
    return dict(arm=arm, k=int(successes), n=int(trials),
                rate=float(successes / trials), cp95_lo=float(lo),
                cp95_hi=float(hi), status=status)


def classify_phenomenon(signs: dict[str, dict]) -> str:
    labels = [signs[a]["status"] for a in ("LR5", "E5")]
    if any(label.endswith("_ABOVE_LINEAR") for label in labels):
        return "PHENOMENON3_REPRODUCED"
    allowed = ("_BELOW_LINEAR", "_NOT_SEPARATED_TIGHT")
    if all(any(label.endswith(suffix) for suffix in allowed)
           for label in labels):
        return "PHENOMENON3_NOT_REPRODUCED"
    return "PHENOMENON3_INCONCLUSIVE"


def _paired_endpoint(cfg: dict, arm: str, windows: dict) -> tuple[dict, dict]:
    """Build G0 sign and registered G0b crossing records."""
    B = cfg["width5_gate_b"]
    early_a = np.asarray(windows[arm]["early"]["raw_u"], dtype=np.float64)
    late_a = np.asarray(windows[arm]["5M"]["raw_u"], dtype=np.float64)
    early_l = np.asarray(windows["LIN5"]["early"]["raw_u"], dtype=np.float64)
    late_l = np.asarray(windows["LIN5"]["5M"]["raw_u"], dtype=np.float64)

    valid_late = np.isfinite(late_a) & np.isfinite(late_l)
    late_above = late_a > late_l
    k = int(np.sum(late_above & valid_late))
    sign = classify_seed_sign(
        arm, k, int(np.sum(valid_late)), alpha=float(B["cp_alpha"]),
        tight_band=tuple(float(v) for v in B["tight_band"]))
    sign.update(
        excluded_seed_indices=np.flatnonzero(~valid_late).astype(int).tolist(),
        ties=int(np.sum(valid_late & (late_a == late_l))),
        late_arm_values=late_a.tolist(), late_linear_values=late_l.tolist(),
        late_differences=(late_a - late_l).tolist())

    valid_cross = (np.isfinite(early_a) & np.isfinite(early_l)
                   & np.isfinite(late_a) & np.isfinite(late_l))
    early_better = early_a < early_l
    crossed = early_better & late_above & valid_cross
    n_cross = int(np.sum(valid_cross))
    c = int(np.sum(crossed))
    cp = (clopper_pearson(c, n_cross, float(B["cp_alpha"]))
          if n_cross > 0 else (None, None))
    early_better_count = int(np.sum(early_better & valid_cross))
    defined = early_better_count >= int(B["crossing_early_better_min_count"])
    crossing = dict(
        arm=arm, c=c, n=n_cross,
        rate=(float(c / n_cross) if n_cross else None),
        cp95_lo=(float(cp[0]) if cp[0] is not None else None),
        cp95_hi=(float(cp[1]) if cp[1] is not None else None),
        early_better_count=early_better_count,
        eligible_min_count=int(B["crossing_early_better_min_count"]),
        status=(f"{arm}_CROSSING_DEFINED" if defined
                else f"{arm}_CROSSING_UNDEFINED"),
        excluded_seed_indices=np.flatnonzero(~valid_cross).astype(int).tolist(),
        crossing_seed_indices=np.flatnonzero(crossed).astype(int).tolist(),
    )
    return sign, crossing


def _pearson(values_a: np.ndarray, values_b: np.ndarray) -> float | None:
    valid = np.isfinite(values_a) & np.isfinite(values_b)
    a, b = values_a[valid], values_b[valid]
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _draws(cfg: dict, n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, n, size=(int(cfg["phase1"]["bootstrap_B"]), n))


def _rectangular(rows: list[dict]) -> list[dict]:
    """Give CSV rows a stable union schema, including divergence branches."""
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    return [{key: row.get(key, "") for key in keys} for row in rows]


def analyze(cfg: dict, outdir: Path, sanity: dict, elapsed: dict,
            divergences: dict[str, dict]) -> dict:
    """Analyze P2b G0/G0b and retain P2's report-only G1/G2/G3."""
    P, G = cfg["phase1"], cfg["width5_gate"]
    complete = [arm for arm in ARM_ORDER if arm not in divergences]
    data = {arm: _load_arm(cfg, outdir, arm) for arm in complete}
    windows = {arm: {
        "early": _window(data[arm], cfg, list(P["early_tasks"])),
        "1M": _window(data[arm], cfg, list(P["window_1m_tasks"])),
        "5M": _window(data[arm], cfg, list(P["late_tasks_5m"])),
    } for arm in complete}

    signs, crossings = {}, {}
    for arm in cfg["width5_gate_b"]["comparison_arms"]:
        if arm not in complete or "LIN5" not in complete:
            signs[arm] = dict(
                arm=arm, k=0, n=0, cp95_lo=None, cp95_hi=None,
                status=f"{arm}_INCONCLUSIVE_DIVERGENCE")
            crossings[arm] = dict(
                arm=arm, c=0, n=0, cp95_lo=None, cp95_hi=None,
                early_better_count=0, status=f"{arm}_CROSSING_UNDEFINED")
        else:
            signs[arm], crossings[arm] = _paired_endpoint(
                cfg, arm, windows)
    main_verdict = classify_phenomenon(signs)
    if "LIN5" in complete:
        lin_ref_full = {}
        for window_name in cfg["sanity"]["s_lin_ref_full_windows"]:
            values = np.asarray(
                windows["LIN5"][window_name]["raw_u"], dtype=np.float64)
            lin_ref_full[window_name] = dict(
                n_valid=int(np.sum(np.isfinite(values))),
                nonfinite_seed_indices=np.flatnonzero(
                    ~np.isfinite(values)).astype(int).tolist())
    else:
        lin_ref_full = dict(
            status=NUMERIC_DIVERGENCE,
            early=dict(n_valid=0, nonfinite_seed_indices=list(range(20))),
            **{"5M": dict(n_valid=0,
                          nonfinite_seed_indices=list(range(20)))})

    rng = np.random.default_rng(int(P["bootstrap_seed"]))
    levels, arm_levels, level_rows = {}, {}, []
    for high, low in G["level_contrasts"]:
        key = f"{high}_minus_{low}"
        if high not in complete or low not in complete:
            levels[key] = dict(status=NUMERIC_DIVERGENCE)
            level_rows.append(dict(
                kind="G1_CONTRAST", contrast=key,
                status=NUMERIC_DIVERGENCE, registered=0))
            continue
        high_values = windows[high]["5M"]["log_u"]
        low_values = windows[low]["5M"]["log_u"]
        valid = np.isfinite(high_values) & np.isfinite(low_values)
        values = high_values[valid] - low_values[valid]
        if not len(values):
            levels[key] = dict(status="NO_VALID_SEEDS")
            level_rows.append(dict(
                kind="G1_CONTRAST", contrast=key,
                status="NO_VALID_SEEDS", registered=0))
            continue
        ci = _ci(cfg, values, _draws(cfg, len(values), rng))
        corr = _pearson(high_values, low_values)
        levels[key] = dict(
            status="REPORT_ONLY", seed_values=values.tolist(), ci=ci,
            terminal_pearson=corr, n_valid=int(len(values)))
        level_rows.append(dict(
            kind="G1_CONTRAST", contrast=key, status="REPORT_ONLY",
            registered=0, point=ci["point"],
            percentile_ci_lo=ci["percentile_ci_lo"],
            percentile_ci_hi=ci["percentile_ci_hi"],
            studentized_ci_lo=ci["studentized_ci_lo"],
            studentized_ci_hi=ci["studentized_ci_hi"],
            ci_degenerate=ci["ci_degenerate"],
            terminal_pearson=("" if corr is None else corr),
            n_valid=len(values), seed_values=json.dumps(values.tolist())))

    for arm in complete:
        values = np.asarray(windows[arm]["5M"]["log_u"], dtype=np.float64)
        values = values[np.isfinite(values)]
        if not len(values):
            continue
        ci = _ci(cfg, values, _draws(cfg, len(values), rng))
        arm_levels[arm] = dict(seed_values=values.tolist(), ci=ci)
        level_rows.append(dict(
            kind="G2_ARM_LEVEL", contrast=arm, status="REPORT_ONLY",
            registered=0, point=ci["point"],
            percentile_ci_lo=ci["percentile_ci_lo"],
            percentile_ci_hi=ci["percentile_ci_hi"],
            studentized_ci_lo=ci["studentized_ci_lo"],
            studentized_ci_hi=ci["studentized_ci_hi"],
            ci_degenerate=ci["ci_degenerate"], terminal_pearson="",
            n_valid=len(values), seed_values=json.dumps(values.tolist())))

    verdict_rows, mechanism_rows = [], []
    for arm in ARM_ORDER:
        width, label, _, alpha = REGISTERED_ARMS[arm]
        if arm not in complete:
            verdict_rows.append(dict(
                arm=arm, width=width, activation=label, act_alpha=alpha,
                status=NUMERIC_DIVERGENCE, main_verdict=main_verdict))
            continue
        w5 = windows[arm]["5M"]
        strict = w5["metrics"]["layer1_strict_dead"] / width
        submerged = w5["metrics"]["layer1_submerged"] / width
        sign = signs.get(arm, {})
        verdict_rows.append(dict(
            arm=arm, width=width, activation=label, act_alpha=alpha,
            status="COMPLETE",
            k_above_LIN5_5m=sign.get("k", ""),
            n_valid_sign_5m=sign.get("n", ""),
            cp95_sign_lo=sign.get("cp95_lo", ""),
            cp95_sign_hi=sign.get("cp95_hi", ""),
            sign_status=sign.get("status", ""),
            U_1m_seed_values=json.dumps(windows[arm]["1M"]["u"].tolist()),
            U_5m_seed_values=json.dumps(w5["u"].tolist()),
            median_log10_U_1m=float(np.median(windows[arm]["1M"]["log_u"])),
            median_log10_U_5m=float(np.median(w5["log_u"])),
            median_submerged_frac_5m=float(np.median(submerged)),
            median_strict_dead_frac_5m=(
                float(np.median(strict)) if label == "relu" else ""),
            main_verdict=main_verdict))
        for window_name in G["report_windows"]:
            w = windows[arm][window_name]
            row = dict(
                arm=arm, width=width, activation=label, act_alpha=alpha,
                window=window_name, registered=0, n_records=w["n_records"],
                median_log10_U=float(np.median(w["log_u"])),
                median_unfit_sum=float(np.median(w["unfit_sum"])),
                median_unfit_rate=float(np.median(w["unfit_rate"])))
            for key, values in w["metrics"].items():
                row[f"median_{key.removeprefix('layer1_')}"] = (
                    float(np.nanmedian(values))
                    if np.isfinite(values).any() else "")
            row["median_submerged_frac"] = float(np.median(
                w["metrics"]["layer1_submerged"] / width))
            mechanism_rows.append(row)

    crossing_rows = [crossings[arm] for arm in
                     cfg["width5_gate_b"]["comparison_arms"]]
    write_csv(outdir / "verdict.csv", _rectangular(verdict_rows))
    write_csv(outdir / "crossing.csv", _rectangular(crossing_rows))
    write_csv(outdir / "levels.csv", _rectangular(level_rows))
    write_csv(outdir / "mechanism.csv", _rectangular(mechanism_rows))
    _write_summary(
        cfg, outdir, main_verdict, verdict_rows, signs, crossings,
        levels, lin_ref_full, sanity, divergences)
    return dict(
        main_verdict=main_verdict, signs=signs,
        crossings_registered_secondary=crossings,
        lin_ref_full=lin_ref_full,
        levels_report_only=levels,
        width_arm_levels_report_only=arm_levels,
        divergences=divergences, elapsed_sec=elapsed)


def _write_summary(cfg: dict, outdir: Path, verdict: str,
                   rows: list[dict], signs: dict, crossings: dict,
                   levels: dict, lin_ref_full: dict, sanity: dict,
                   divergences: dict) -> None:
    lines = [
        f"# {EXPERIMENT} summary", "",
        "## Registered verdict", "",
        f"- G0: **{verdict}**",
        "- G0 compares each width-5 nonlinear arm with LIN5 seed by seed; "
        "it is not an absolute LoP onset rate.",
        "- G0b crossing is a registered secondary endpoint.",
        "- G1 levels, G2 width levels, and all G3 mechanism metrics are "
        "REPORT_ONLY.",
        f"- Numeric divergence: {', '.join(sorted(divergences)) or 'none'}",
        "", "## G0 paired signs (terminal task 491–500)", "",
        "| arm | k above LIN5 | valid n | CP95 | label |",
        "|---|---:|---:|---|---|",
    ]
    for arm in cfg["width5_gate_b"]["comparison_arms"]:
        row = signs[arm]
        interval = ("—" if row.get("cp95_lo") is None else
                    f"[{row['cp95_lo']:.6g}, {row['cp95_hi']:.6g}]")
        lines.append(
            f"| {arm} | {row['k']} | {row['n']} | {interval} | "
            f"**{row['status']}** |")
    lines += [
        "", "## G0b crossing (early task 2–11 → terminal task 491–500)", "",
        "| arm | early better than LIN5 | crossings | valid n | CP95 | status |",
        "|---|---:|---:|---:|---|---|",
    ]
    for arm in cfg["width5_gate_b"]["comparison_arms"]:
        row = crossings[arm]
        interval = ("—" if row.get("cp95_lo") is None else
                    f"[{row['cp95_lo']:.6g}, {row['cp95_hi']:.6g}]")
        lines.append(
            f"| {arm} | {row['early_better_count']} | {row['c']} | "
            f"{row['n']} | {interval} | {row['status']} |")
    lines += [
        "", "## Endpoints", "",
        "| arm | width | activation | median log10 U 5M | submerged frac 5M |",
        "|---|---:|---|---:|---:|",
    ]
    for row in rows:
        if row["status"] != "COMPLETE":
            lines.append(
                f"| {row['arm']} | {row['width']} | {row['activation']} | "
                f"— | {row['status']} |")
        else:
            lines.append(
                f"| {row['arm']} | {row['width']} | {row['activation']} | "
                f"{row['median_log10_U_5m']:.6g} | "
                f"{row['median_submerged_frac_5m']:.6g} |")
    lines += ["", "## G1 paired levels (REPORT_ONLY)", ""]
    for name, value in levels.items():
        corr = value.get("terminal_pearson")
        corr_text = "undefined" if corr is None else f"{corr:.6g}"
        lines.append(
            f"- {name}: terminal Pearson = {corr_text}; "
            f"status = {value['status']}")
    lines += ["", "## Floor characterization (not a gate)", ""]
    floor = sanity.get("preflight", sanity).get("S_floor_char", {})
    for row in floor.get("rows", []):
        if row.get("status") == "COMPLETE":
            lines.append(
                f"- {row['arm']} early-window per-seed minima: "
                f"min {row['min']:.6g}, median {row['median']:.6g}, "
                f"max {row['max']:.6g}; below 0.05 = "
                f"{row['below_predecessor_threshold']}/20.")
        else:
            lines.append(f"- {row.get('arm')}: {row.get('status')}")
    lines += ["", "## LIN5 reference finiteness", ""]
    for window_name in cfg["sanity"]["s_lin_ref_full_windows"]:
        row = lin_ref_full[window_name]
        lines.append(
            f"- LIN5 {window_name}: finite {row['n_valid']}/20; "
            f"nonfinite seeds: {row['nonfinite_seed_indices'] or 'none'}")
    lost = {arm: signs[arm].get("excluded_seed_indices", [])
            for arm in cfg["width5_gate_b"]["comparison_arms"]}
    for arm, indices in lost.items():
        lines.append(f"- {arm} vs LIN5 excluded seeds: {indices or 'none'}")
    lines += ["", "## Prediction provenance", "",
              "- Carry-over predictions retain the status frozen in P2.",
              "- New N1–N3 entries are reported with the provenance stored in "
              "config_used.yaml.", "", "## Sanity", ""]
    for key, value in sanity.items():
        if isinstance(value, dict) and "pass_" in value:
            label = ("CHARACTERIZED" if key == "S_floor_char"
                     else "PASS" if value["pass_"] else "FAIL")
            lines.append(f"- {key}: **{label}**")
    (outdir / "summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")


def _provenance(cfg_path: Path, cfg: dict, outdir: Path, sanity: dict,
                analysis_result: dict, elapsed: dict, started: float) -> dict:
    spec_path = Path(ROOT) / str(cfg["spec"])
    if not spec_path.exists():
        raise FileNotFoundError(f"frozen repo spec missing: {spec_path}")
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--short"], cwd=ROOT,
            text=True).splitlines()
    except (OSError, subprocess.CalledProcessError):
        git_hash, dirty = None, []
    names = ("verdict.csv", "crossing.csv", "levels.csv", "mechanism.csv",
             "summary.md", "config_used.yaml")
    hashes = {name: _sha_file(outdir / name) for name in names
              if (outdir / name).exists()}
    hashes.update({f"logs/{p.name}": _sha_file(p)
                   for p in sorted((outdir / "logs").glob("*.npz"))})
    return dict(
        experiment=EXPERIMENT,
        created=time.strftime("%Y-%m-%d %H:%M:%S %z"),
        command=sys.argv, elapsed_sec=round(time.time() - started, 3),
        arm_elapsed_sec=elapsed, cwd=os.getcwd(), python=sys.version,
        platform=platform.platform(), device=cfg["common"]["device"],
        git_hash=git_hash, git_dirty=dirty,
        config=str(cfg_path), config_sha256=_sha_file(cfg_path),
        spec=str(spec_path), spec_sha256=_sha_file(spec_path),
        sanity=sanity, analysis=analysis_result, output_sha256=hashes)


def run_full(cfg_path: Path, cfg: dict, device: str, outdir: Path, *,
             smoke: bool) -> dict:
    started = time.time()
    outdir.mkdir(parents=True, exist_ok=True)
    require_omp(cfg)
    total = SMOKE_STEPS if smoke else int(cfg["common"]["total_steps"])
    seeds = SMOKE_SEEDS if smoke else [int(v) for v in cfg["common"]["seeds"]]
    with (outdir / "config_used.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, allow_unicode=True, sort_keys=False)
    if smoke:
        preflight_result = {"pass_": True, "smoke": True}
    else:
        path = Path(ROOT) / PREFLIGHT_DIR / "preflight.json"
        if not path.exists():
            raise FileNotFoundError(
                "run --preflight after freezing the three predictions")
        preflight_result = json.loads(path.read_text(encoding="utf-8"))
        if not preflight_result.get("pass_"):
            raise RuntimeError("saved P2b preflight did not pass")

    elapsed, identities, divergences = {}, {}, {}
    for arm in ARM_ORDER:
        if _complete_arm_logs(
                outdir, arm, seeds, total,
                int(cfg["common"]["lop_every"])):
            elapsed[arm] = 0.0
            identities[arm] = dict(pass_=True, resumed_from_logs=True)
            continue
        result = _run_arm(cfg, arm, device, outdir, seeds, total)
        elapsed[arm], identities[arm] = (
            result["elapsed_sec"], result["sanity"])
        if result["status"] == NUMERIC_DIVERGENCE:
            divergences[arm] = result["divergence"]
    if smoke:
        payload = dict(
            pass_=bool(all(v.get("pass_") for v in identities.values())),
            identities=identities, divergences=divergences,
            elapsed_sec=elapsed)
        (outdir / "smoke_sanity.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8")
        return payload

    complete = set(ARM_ORDER) - set(divergences)
    sanity = dict(
        preflight=preflight_result,
        final_pairing=_pair_check_final(cfg, outdir, complete),
        exact_recorders=identities)
    if not sanity["final_pairing"]["pass_"]:
        raise RuntimeError("final within-width pairing sanity failed")
    result = analyze(cfg, outdir, sanity, elapsed, divergences)
    provenance = _provenance(
        cfg_path, cfg, outdir, sanity, result, elapsed, started)
    (outdir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")
    print(f"ALL DONE -> {outdir}", flush=True)
    return dict(sanity=sanity, analysis=result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/width5_gate_b_0901.yaml")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--outdir")
    args = parser.parse_args()
    if sum((args.preflight, args.smoke, args.analyze_only)) > 1:
        parser.error("stage flags are mutually exclusive")
    cfg_path = Path(args.config).resolve()
    cfg = load_config(str(cfg_path))
    device = pick_device(cfg)
    if device != "cpu":
        raise ValueError("width5_gate_b is CPU-only")
    stage = ("preflight" if args.preflight else
             "smoke" if args.smoke else
             "analyze" if args.analyze_only else "full")
    validate_config(cfg, stage=stage)
    main_dir = Path(ROOT) / str(cfg["output"]["dir"])
    outdir = (Path(args.outdir).resolve() if args.outdir else
              Path(ROOT) / PREFLIGHT_DIR if args.preflight else
              Path(ROOT) / SMOKE_DIR if args.smoke else main_dir)
    if args.preflight:
        preflight(cfg, device, outdir)
    elif args.analyze_only:
        preflight_result = json.loads(
            (Path(ROOT) / PREFLIGHT_DIR / "preflight.json").read_text(
                encoding="utf-8"))
        divergences = _collect_divergences(cfg, outdir)
        complete = set(ARM_ORDER) - set(divergences)
        sanity = dict(
            preflight=preflight_result,
            final_pairing=_pair_check_final(cfg, outdir, complete))
        analyze(cfg, outdir, sanity, {}, divergences)
    else:
        run_full(cfg_path, cfg, device, outdir, smoke=args.smoke)


if __name__ == "__main__":
    main()
