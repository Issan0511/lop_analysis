"""Post hoc CondA dimension and mean-channel diagnostic; no training or P verdicts.

Reads the completed 0924 task-end snapshots and partial_0053 bookkeeping.
Writes only CSV/Markdown under dimension_diagnostic. Task t means W_(t-1)
to W_t. The parameter-induced mean shift always uses mu_(t-1).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ARCHIVE = Path("/home/issan/Projects/obsidian-research-data/effdisp_validation_0924")
ARMS = (
    "k1", "k1_r10", "k1_sna06", "m40_r5_k1", "m40_r10_k1",
    "m40_r10_k2", "m40_r10_k1_sna06", "m40_r10_k2_sna06",
)
SEEDS = range(100, 105)
WINDOWS = (("early", 1, 20), ("middle", 41, 60),
           ("late", 81, 100), ("full", 1, 100))
COMPARISONS = (
    ("m20_r5_to_m40_r5", "k1", "m40_r5_k1", "different_teacher_and_initialization"),
    ("m20_r10_to_m40_r10_k1", "k1_r10", "m40_r10_k1", "different_teacher_and_initialization"),
    ("m20_r10_to_m40_r10_k2", "k1_r10", "m40_r10_k2", "different_teacher_and_initialization"),
    ("m40_r10_k1_to_k2", "m40_r10_k1", "m40_r10_k2", "shared_m40_teacher_and_initialization"),
    ("m20_r5_leaky_to_SNA06", "k1", "k1_sna06", "shared_m20_teacher_and_initialization"),
    ("m40_r10_k1_leaky_to_SNA06", "m40_r10_k1", "m40_r10_k1_sna06", "shared_m40_teacher_and_initialization"),
    ("m40_r10_k2_leaky_to_SNA06", "m40_r10_k2", "m40_r10_k2_sna06", "shared_m40_teacher_and_initialization"),
)


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else math.nan


def _snapshot(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        required = ("W", "b", "mu", "m", "r", "task", "mse", "task_start_mse",
                    "activeunit_frac", "derivative_absmean")
        return {key: np.asarray(data[key], dtype=np.float64) for key in required}


def _arm_metadata(raw: Path) -> dict[str, dict]:
    metadata = json.loads((raw / "metadata.json").read_text())
    arms = {}
    for group in ("primary_arms", "extra_arms", "frequency_arms", "m40_arms"):
        for name, k, optimizer, lr, eps, r, activation in metadata[group]:
            arms[name] = {"k": int(k), "optimizer": optimizer, "lr": lr,
                          "eps": eps, "r_meta": int(r), "activation": activation}
    if not set(ARMS).issubset(arms):
        raise ValueError(f"metadata lacks arms: {set(ARMS) - set(arms)}")
    return arms


def _one_transition(prev: dict, current: dict, meta: dict, *, arm: str, seed: int,
                    task: int, source_prev: Path, source_next: Path) -> dict:
    w, wn = prev["W"], current["W"]
    b, bn = prev["b"], current["b"]
    mu, mun = prev["mu"], current["mu"]
    m, r = int(current["m"]), int(current["r"])
    if (w.shape != (100, m) or wn.shape != w.shape or mu.shape != (m,)
            or mun.shape != (m,) or b.shape != (100,) or bn.shape != (100,)
            or int(prev["m"]) != m or int(prev["r"]) != r or r != meta["r_meta"]
            or int(prev["task"]) != task - 1 or int(current["task"]) != task):
        raise ValueError(f"shape or task mismatch: {arm}/seed{seed}/t{task}")
    arrays = (w, wn, b, bn, mu, mun)
    if not all(np.all(np.isfinite(a)) for a in arrays):
        raise ValueError(f"nonfinite snapshot: {arm}/seed{seed}/t{task}")
    d = wn - w
    var = np.r_[np.zeros(m-r), np.full(r, .25)]
    # All sums and products below are float64. These are centered on the
    # conditional input distribution of the previous task.
    v = float(np.sum(w*w*var))
    q = float(np.sum(d*d*var))
    x = float(np.sum(w*d*var))
    vn = float(np.sum(wn*wn*var))
    vr = float(np.sum(w*w))
    qr = float(np.sum(d*d))
    xr = float(np.sum(w*d))
    vnr = float(np.sum(wn*wn))
    parameter_mean_step = d @ mu + (bn - b)
    environment_mean_step = wn @ (mun - mu)
    full_mean_step = parameter_mean_step + environment_mean_step
    param_sq = float(parameter_mean_step @ parameter_mean_step)
    env_sq = float(environment_mean_step @ environment_mean_step)
    full_sq = float(full_mean_step @ full_mean_step)
    cross = float(2 * (parameter_mean_step @ environment_mean_step))
    start_mse = float(current["task_start_mse"])
    end_mse = float(current["mse"])
    if not all(np.isfinite(z) for z in (v,q,x,vn,vr,qr,xr,vnr,param_sq,env_sq,
                                       full_sq,cross,start_mse,end_mse)):
        raise ValueError(f"nonfinite diagnostic: {arm}/seed{seed}/t{task}")
    return {
        "arm": arm, "seed": seed, "task": task, "m": m, "r": r, "f": m-r,
        "k": meta["k"], "k_over_f": meta["k"]/(m-r),
        "activation": meta["activation"], "conditional_rank": r,
        "conditional_trace": .25*r, "conditional_rank_fraction": r/m,
        "V_eff_prev": v, "V_eff_next": vn, "Q_eff": q, "X_eff": x,
        "R_eff_prev": math.sqrt(v), "R_eff_next": math.sqrt(vn),
        "D_eff": math.sqrt(q), "c_eff": _safe_ratio(x, math.sqrt(v*q)),
        "rho_eff": _safe_ratio(math.sqrt(q), math.sqrt(v)),
        "balance_eff_transition": _safe_ratio(-2*x, q),
        "eff_identity_residual": vn-v-q-2*x,
        "V_raw_prev": vr, "V_raw_next": vnr, "Q_raw": qr, "X_raw": xr,
        "R_raw_prev": math.sqrt(vr), "R_raw_next": math.sqrt(vnr),
        "D_raw": math.sqrt(qr), "c_raw": _safe_ratio(xr, math.sqrt(vr*qr)),
        "rho_raw": _safe_ratio(math.sqrt(qr), math.sqrt(vr)),
        "raw_identity_residual": vnr-vr-qr-2*xr,
        "D_mean_parameter_sq": param_sq, "D_mean_parameter": math.sqrt(param_sq),
        "D_mean_environment_sq": env_sq, "D_mean_environment": math.sqrt(env_sq),
        "D_mean_cross": cross, "D_mean_full_sq": full_sq,
        "D_mean_full": math.sqrt(full_sq),
        "mean_identity_residual": full_sq-param_sq-env_sq-cross,
        "mean_z_prev_norm": float(np.linalg.norm(w @ mu + b)),
        "mean_z_next_norm": float(np.linalg.norm(wn @ mun + bn)),
        "parameter_mean_sq_over_Q_eff": _safe_ratio(param_sq, q),
        "start_mse": start_mse, "end_mse": end_mse,
        "mse_gain": start_mse-end_mse,
        "mse_gain_fraction": _safe_ratio(start_mse-end_mse, start_mse),
        "activeunit_frac": float(current["activeunit_frac"]),
        "derivative_absmean": float(current["derivative_absmean"]),
        "source_prev": str(source_prev), "source_next": str(source_next),
    }


def load_transitions(raw: Path, partial: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata = _arm_metadata(raw)
    completion = pd.read_csv(partial / "completion.csv")
    selected = completion[(completion.environment == "conda") & completion.arm.isin(ARMS)]
    for arm in ARMS:
        cases = selected[selected.arm == arm]
        if len(cases) != 5 or set(cases.seed) != set(SEEDS) or not cases.status.eq("COMPLETE").all():
            raise ValueError(f"{arm}: five completed seeds required")
    rows = []
    for arm in ARMS:
        for seed in SEEDS:
            directory = raw / arm / f"seed{seed}"
            status = json.loads((directory / "status.json").read_text())
            if status.get("status") != "COMPLETED" or status.get("last_saved_task") != 100:
                raise ValueError(f"{arm}/seed{seed}: not complete")
            old_path = directory / "t000.npz"
            old = _snapshot(old_path)
            for task in range(1, 101):
                new_path = directory / f"t{task:03d}.npz"
                new = _snapshot(new_path)
                rows.append(_one_transition(old, new, metadata[arm], arm=arm, seed=seed,
                                            task=task, source_prev=old_path,
                                            source_next=new_path))
                old_path, old = new_path, new
    frame = pd.DataFrame(rows)
    # Cross-check independent recomputation against the existing partial
    # ledger. This detects time-index or covariance mistakes without treating
    # the algebraic identity as scientific support.
    ledger = pd.read_csv(partial / "transitions.csv")
    ledger = ledger[(ledger.environment == "conda") & ledger.arm.isin(ARMS)
                    & ledger.metric.isin(["effective", "raw"])]
    checks = []
    for metric, suffix in (("effective", "eff"), ("raw", "raw")):
        original = ledger[ledger.metric == metric][["arm","seed","task","V","Q","X","V_next"]]
        merged = frame.merge(original, on=["arm","seed","task"], validate="one_to_one")
        if len(merged) != len(frame):
            raise ValueError(f"{metric}: ledger row count mismatch")
        for ours, theirs in ((f"V_{suffix}_prev", "V"), (f"Q_{suffix}", "Q"),
                             (f"X_{suffix}", "X"), (f"V_{suffix}_next", "V_next")):
            scale = np.maximum(1., np.maximum(merged[ours].abs(), merged[theirs].abs()))
            rel = (merged[ours] - merged[theirs]).abs() / scale
            checks.append({"metric": metric, "quantity": theirs,
                           "max_abs_difference": float((merged[ours]-merged[theirs]).abs().max()),
                           "max_relative_difference": float(rel.max())})
    check_frame = pd.DataFrame(checks)
    if check_frame.max_relative_difference.max() > 1e-10:
        raise ValueError("recomputed metrics disagree with partial_0053 ledger")
    return frame, check_frame


def aggregate_windows(transitions: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (arm, seed), block in transitions.groupby(["arm", "seed"], sort=True):
        for window, lo, hi in WINDOWS:
            part = block[(block.task >= lo) & (block.task <= hi)].sort_values("task")
            if len(part) != hi-lo+1:
                raise ValueError(f"missing {window} tasks for {arm}/seed{seed}")
            sv, sq, sx = (float(part[column].sum()) for column in
                          ("V_eff_prev", "Q_eff", "X_eff"))
            sr_v, sr_q, sr_x = (float(part[column].sum()) for column in
                               ("V_raw_prev", "Q_raw", "X_raw"))
            first, last = part.iloc[0], part.iloc[-1]
            records.append({
                "arm": arm, "seed": seed, "window": window, "first_task": lo,
                "last_task": hi, "m": int(first.m), "r": int(first.r),
                "f": int(first.f), "k": int(first.k),
                "conditional_rank": int(first.conditional_rank),
                "conditional_trace": float(first.conditional_trace),
                "mean_R_eff": float(part.R_eff_prev.mean()),
                "mean_D_eff": float(part.D_eff.mean()),
                "mean_R_raw": float(part.R_raw_prev.mean()),
                "mean_D_raw": float(part.D_raw.mean()),
                "raw_to_effective_R": _safe_ratio(float(part.R_raw_prev.mean()), float(part.R_eff_prev.mean())),
                "raw_to_effective_D": _safe_ratio(float(part.D_raw.mean()), float(part.D_eff.mean())),
                "B_eff_summed": _safe_ratio(-2*sx, sq),
                "c_eff_summed": _safe_ratio(sx, math.sqrt(sv*sq)),
                "rho_eff_summed": _safe_ratio(math.sqrt(sq), math.sqrt(sv)),
                "B_raw_summed": _safe_ratio(-2*sr_x, sr_q),
                "c_raw_summed": _safe_ratio(sr_x, math.sqrt(sr_v*sr_q)),
                "rho_raw_summed": _safe_ratio(math.sqrt(sr_q), math.sqrt(sr_v)),
                "activity_eff": _safe_ratio(sq, float(part.V_eff_prev.mean())),
                "endpoint_V_eff_ratio": _safe_ratio(float(last.V_eff_next), float(first.V_eff_prev)),
                "mean_D_mean_parameter": float(part.D_mean_parameter.mean()),
                "rms_D_mean_parameter": math.sqrt(float(part.D_mean_parameter_sq.mean())),
                "mean_D_mean_environment": float(part.D_mean_environment.mean()),
                "rms_D_mean_environment": math.sqrt(float(part.D_mean_environment_sq.mean())),
                "mean_D_mean_full": float(part.D_mean_full.mean()),
                "mean_parameter_sq_over_Q_eff": _safe_ratio(float(part.D_mean_parameter_sq.sum()), sq),
                "mean_environment_sq_over_Q_eff": _safe_ratio(float(part.D_mean_environment_sq.sum()), sq),
                "mean_start_mse": float(part.start_mse.mean()),
                "mean_end_mse": float(part.end_mse.mean()),
                "mean_mse_gain": float(part.mse_gain.mean()),
                "mean_mse_gain_fraction": float(part.mse_gain_fraction.mean()),
                "median_mse_gain_fraction": float(part.mse_gain_fraction.median()),
                "fraction_tasks_mse_gain_positive": float((part.mse_gain > 0).mean()),
                "mean_activeunit_frac": float(part.activeunit_frac.mean()),
                "mean_derivative_absmean": float(part.derivative_absmean.mean()),
            })
    return pd.DataFrame(records)


def compare_pairs(seed_windows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    indexed = seed_windows.set_index(["arm", "seed", "window"])
    fields = ("mean_R_eff", "mean_D_eff", "mean_R_raw", "mean_D_raw",
              "raw_to_effective_R", "raw_to_effective_D", "B_eff_summed",
              "c_eff_summed", "rho_eff_summed", "activity_eff",
              "endpoint_V_eff_ratio", "mean_D_mean_parameter",
              "rms_D_mean_parameter", "mean_D_mean_environment",
              "rms_D_mean_environment", "mean_D_mean_full",
              "mean_parameter_sq_over_Q_eff", "mean_environment_sq_over_Q_eff",
              "mean_start_mse", "mean_end_mse", "mean_mse_gain",
              "mean_mse_gain_fraction", "median_mse_gain_fraction",
              "fraction_tasks_mse_gain_positive")
    rows = []
    for comparison, reference, other, design in COMPARISONS:
        for seed in SEEDS:
            for window, _, _ in WINDOWS:
                a, b = indexed.loc[(reference,seed,window)], indexed.loc[(other,seed,window)]
                row = {"comparison": comparison, "reference_arm": reference,
                       "other_arm": other, "design_note": design,
                       "seed": seed, "window": window,
                       "reference_m": int(a.m), "other_m": int(b.m),
                       "reference_rank": int(a.conditional_rank),
                       "other_rank": int(b.conditional_rank),
                       "reference_trace": float(a.conditional_trace),
                       "other_trace": float(b.conditional_trace)}
                for field in fields:
                    av, bv = float(a[field]), float(b[field])
                    row[f"reference_{field}"] = av
                    row[f"other_{field}"] = bv
                    row[f"difference_{field}"] = bv-av
                    row[f"ratio_{field}"] = _safe_ratio(bv, av)
                rows.append(row)
    paired = pd.DataFrame(rows)
    med = paired.groupby(["comparison", "reference_arm", "other_arm", "design_note", "window"],
                         as_index=False).median(numeric_only=True)
    med["n_seed"] = 5
    return paired, med


def _md_table(data: pd.DataFrame, columns: list[str]) -> str:
    head = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for row in data[columns].itertuples(index=False, name=None):
        rows.append("| " + " | ".join(f"{v:.3g}" if isinstance(v, (float, np.floating)) and np.isfinite(v)
                                        else str(v) for v in row) + " |")
    return "\n".join([head, sep, *rows])


def write_report(path: Path, windows: pd.DataFrame, med: pd.DataFrame,
                 checks: pd.DataFrame, transitions: pd.DataFrame) -> None:
    late = windows[windows.window == "late"].groupby("arm", as_index=False).median(numeric_only=True)
    late_pairs = med[med.window == "late"].copy()
    phase_pairs = med[med.window.isin(["early", "late"])].copy()
    # Numeric median over seeds is descriptive; m40 snapshots have a different
    # teacher/initialization dimension and cannot isolate a dimension effect.
    lines = [
        "# CondA dimension and mean-channel diagnostic — post hoc",
        "",
        "This uses the completed CondA snapshots only. It adds no training and changes no preregistered P1–P6 criterion or verdict. All values below are medians of five seed-level values; `transition.csv`, `seed_windows.csv`, and `paired_seed.csv` retain every value and source snapshot path.",
        "",
        "## Late window (tasks 81–100)", "",
        "D and R are square roots of centered Q and V. The raw columns use Σ=I. B uses -2 sum X/sum Q; c and rho also use sums, never averages of per-task ratios.", "",
        _md_table(late, ["arm", "m", "r", "conditional_trace", "mean_D_eff", "mean_R_eff",
                         "c_eff_summed", "rho_eff_summed", "B_eff_summed",
                         "mean_D_raw", "mean_R_raw", "raw_to_effective_R",
                         "mean_D_mean_parameter", "mean_D_mean_environment",
                         "mean_parameter_sq_over_Q_eff", "mean_start_mse", "mean_end_mse",
                         "mean_mse_gain", "mean_mse_gain_fraction"]), "",
        "## Seed-indexed comparisons, late window", "",
        "Ratios are calculated within each seed and then medianed. m20–m40 rows match only seed indices: changing input dimension also changes the teacher and initialization. They are not pure dimension-causal estimates.", "",
        _md_table(late_pairs, ["comparison", "reference_m", "other_m", "reference_rank",
                               "other_rank", "ratio_mean_D_eff", "ratio_mean_R_eff",
                               "ratio_mean_D_raw", "ratio_mean_R_raw",
                               "ratio_raw_to_effective_R",
                               "ratio_mean_D_mean_parameter", "ratio_mean_D_mean_environment",
                               "difference_B_eff_summed", "difference_mean_mse_gain_fraction"]), "",
        "## Early and late paired trajectories", "",
        "The two 20-transition windows show whether a ratio is persistent over this observed horizon; they do not establish an asymptotic trend.", "",
        _md_table(phase_pairs, ["comparison", "window", "ratio_mean_D_eff",
                                "ratio_mean_R_eff", "ratio_mean_D_raw", "ratio_mean_R_raw",
                                "difference_mean_mse_gain_fraction"]), "",
        "## Mean channel and interpretation", "",
        "For each transition, the parameter-induced mean change is ||D μ_prev + Δb||². The environment change ||W_next(μ_next−μ_prev)||² and their cross term are separate columns; their sum is the full mean-preactivation shift. This prevents treating persistent-bit directions invisible to centered Σ as harmless weight.",
        "A low sum Q/mean V or STOPPED label is a relative-update diagnostic, not proof that the learner cannot improve. Task-start and task-end support MSE and their signed gain are reported beside displacement. MSE differences across m20/m40 may reflect different target functions and initializations.",
        "The raw/centered algebraic identities and ledger cross-check establish bookkeeping only; they are not independent evidence for a fixed point, causal independence, or asymptotic stationarity.",
        "", "## Audit", "",
        f"Rows: {len(transitions)} transitions; maximum |centered identity residual| = {transitions.eff_identity_residual.abs().max():.3g}, maximum |raw identity residual| = {transitions.raw_identity_residual.abs().max():.3g}, maximum |mean decomposition residual| = {transitions.mean_identity_residual.abs().max():.3g}.",
        f"Maximum relative difference from `partial_0053/transitions.csv` = {checks.max_relative_difference.max():.3g}.",
        "Source: `/home/issan/Projects/obsidian-research-data/effdisp_validation_0924/raw/conda`; previous and next snapshot paths are recorded per transition.", "",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ARCHIVE / "raw" / "conda")
    parser.add_argument("--partial", type=Path, default=ARCHIVE / "partial_0053")
    parser.add_argument("--out", type=Path, default=ARCHIVE / "dimension_diagnostic")
    args = parser.parse_args()
    # All reads and checks complete before any output is written.
    transitions, checks = load_transitions(args.raw, args.partial)
    windows = aggregate_windows(transitions)
    paired, med = compare_pairs(windows)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    transitions.to_csv(out / "transition.csv", index=False)
    windows.to_csv(out / "seed_windows.csv", index=False)
    paired.to_csv(out / "paired_seed.csv", index=False)
    med.to_csv(out / "paired_median.csv", index=False)
    checks.to_csv(out / "ledger_crosscheck.csv", index=False)
    write_report(out / "summary.md", windows, med, checks, transitions)


if __name__ == "__main__":
    main()
