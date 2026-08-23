"""Independent confirmation analysis for direct functional utility (work 6).

The registered invocation is::

  OMP_NUM_THREADS=1 .venv/bin/python -m analysis.function_blind_direct.confirm \
    --logs results/function_blind_direct_0823_confirm/logs \
    --outdir results/function_blind_direct_0823_confirm

The analysis implements ``specs/spec_function_blind_direct_0823_confirm.md``.
In particular, all geometry cells and utility labels are fixed once on the
complete confirmation data before a seed-block bootstrap is run.  Pilot files
are neither read nor modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SPEC = "specs/spec_function_blind_direct_0823_confirm.md"
SWITCHES = tuple(range(200_000, 800_001, 10_000))
EXPECTED_STEPS = np.asarray(
    [step for boundary in SWITCHES for step in (boundary + 1, boundary + 10_000)],
    dtype=np.int64,
)
SEEDS = tuple(range(20))
WIDTH = 100
PERIOD = 10_000
TOTAL_STEPS = 810_000
SUPPORT_SIZE = 32
GENERATOR_OFFSET = 20_260_830
TAU = 0.05
GROUPS = ("low", "mid", "high")
BOOT_N = 10_000
BOOT_SEED = 20_260_831
BOOT_BATCH = 100
EQUIV_MARGIN = 0.05
REQUIRED_OFFSET_KEYS = (
    "net.W", "net.v", "env.flip_state", "teacher.W", "teacher.v",
)
UNIT_KEYS = (
    "p_hat", "pre_max", "x", "r", "w_norm", "b", "v",
    "utility_raw", "utility_nmse",
)
RUN_KEYS = ("eval_nmse", "y_var")
CSV_NAMES = (
    "exposures.csv", "primary_cells.csv", "primary_rates.csv", "verdict.csv",
    "secondary_results.csv", "repeat_exposure.csv", "sanity.csv",
)


def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _scalar(value: np.ndarray | Any) -> Any:
    return np.asarray(value).item()


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"))


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(
        index=False, lineterminator="\n", float_format="%.17g"
    ).encode("utf-8")


def _epoch(boundary: int) -> str:
    if boundary <= 390_000:
        return "200-390k"
    if boundary <= 590_000:
        return "400-590k"
    return "600-800k"


def validate_instrumentation_meta(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate runner-side prerequisites before looking at effect estimates."""
    if not path.exists():
        raise SystemExit(f"instrumentation metaがない: {path}")
    meta = json.loads(path.read_text())
    sanity = meta.get("sanity", {})
    instrument = sanity.get("instrumentation") or {}
    default_zero = sanity.get("default_zero_offset") or {}
    s2 = sanity.get("S2") or {}
    offset = sanity.get("actual_offset_vs_zero") or {}
    key_differences = offset.get("required_key_differences") or {}

    checks = {
        "mode": meta.get("mode") == "confirmation" and meta.get("pilot_only") is False
                and meta.get("smoke") is False,
        "registered_shape": meta.get("R") == 20 and meta.get("seeds") == list(SEEDS)
                            and meta.get("width") == WIDTH
                            and meta.get("total_steps") == TOTAL_STEPS
                            and meta.get("n_records") == len(EXPECTED_STEPS)
                            and meta.get("n_npz") == len(SEEDS),
        "registered_offset": meta.get("generator_offset") == GENERATOR_OFFSET,
        "runner_all_required": sanity.get("all_required_pass") is True,
        "delta_formula": instrument.get("delta_formula_pass") is True
                         and int(instrument.get("delta_formula_n", -1)) == 20
                         and float(instrument.get("delta_formula_max_abs_error", np.inf)) < 1e-12,
        "runner_geometry": instrument.get("p_hat_quantization_pass") is True
                           and float(instrument.get(
                               "p_hat_quantization_max_abs_error", np.inf)) < 1e-12
                           and instrument.get("geometry_pass") is True
                           and float(instrument.get(
                               "geometry_max_relative_error", np.inf)) < 1e-10
                           and instrument.get("strict_dead_pre_max_identity_pass") is True
                           and int(instrument.get(
                               "strict_dead_pre_max_mismatches", -1)) == 0
                           and instrument.get("finite_pass") is True
                           and int(instrument.get("n_nonfinite", -1)) == 0
                           and instrument.get("support_size_pass") is True,
        "default_zero_offset": default_zero.get("pass_") is True,
        "s2": s2.get("pass_") is True and s2.get("steps") == 100_000,
        "offset_comparison": offset.get("pass_") is True
                             and offset.get("mode") == "confirmation"
                             and offset.get("required") is True
                             and offset.get("actual_offset") == GENERATOR_OFFSET
                             and offset.get("zero_offset") == 0
                             and tuple(offset.get("required_keys", ())) == REQUIRED_OFFSET_KEYS
                             and all(key_differences.get(key) is True
                                     for key in REQUIRED_OFFSET_KEYS),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise SystemExit(
            "instrumentation meta必須sanity/offset検査FAIL: " + ", ".join(failed)
        )
    rows = [
        dict(id="C-S2", status="PASS",
             value=float(instrument["delta_formula_max_abs_error"]),
             threshold="<1e-12", detail=f"n={instrument['delta_formula_n']}"),
        dict(id="C-S5", status="PASS", value=int(s2["steps"]),
             threshold="100000-step exact state/hash match", detail="differences=0"),
        dict(id="C-S6", status="PASS", value=len(REQUIRED_OFFSET_KEYS),
             threshold="all 5 required initial hashes differ from offset=0",
             detail=_json(dict(default_zero_offset_pass=True,
                               required_key_differences=key_differences))),
    ]
    return meta, rows


def load_logs(logdir: Path) -> list[dict[str, Any]]:
    paths = sorted(logdir.glob("seed*.npz"), key=lambda p: p.name)
    if len(paths) != len(SEEDS):
        raise SystemExit(f"confirmationは20 NPZを要求: found={len(paths)} in {logdir}")
    logs: list[dict[str, Any]] = []
    required = (
        "step", "seed", "run_id", "condition", "encoding", "batch", "width",
        "period", "lr", "generator_offset", "mode", "total_steps", "support_size",
        "spec",
    ) + UNIT_KEYS + RUN_KEYS
    for path in paths:
        with np.load(path, allow_pickle=False) as z:
            missing = [key for key in required if key not in z.files]
            if missing:
                raise SystemExit(f"{path}: missing keys {missing}")
            data = {key: np.asarray(z[key]) for key in required}
        data["path"] = path
        logs.append(data)
    logs.sort(key=lambda data: int(_scalar(data["seed"])))
    found_seeds = [int(_scalar(data["seed"])) for data in logs]
    if found_seeds != list(SEEDS):
        raise SystemExit(f"seed labelは0..19を要求: {found_seeds}")
    return logs


def log_sanity(logs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    per_seed: list[dict[str, Any]] = []
    max_quant = 0.0
    max_geometry = 0.0
    total_mismatch = 0
    total_nonfinite = 0
    for data in logs:
        seed = int(_scalar(data["seed"]))
        step = np.asarray(data["step"], dtype=np.int64)
        scalar_ok = (
            int(_scalar(data["width"])) == WIDTH
            and int(_scalar(data["period"])) == PERIOD
            and int(_scalar(data["generator_offset"])) == GENERATOR_OFFSET
            and int(_scalar(data["total_steps"])) == TOTAL_STEPS
            and int(_scalar(data["support_size"])) == SUPPORT_SIZE
            and str(_scalar(data["mode"])) == "confirmation"
            and str(_scalar(data["condition"])) == "condA"
            and str(_scalar(data["encoding"])) == "std"
            and str(_scalar(data["batch"])) == "1"
            and abs(float(_scalar(data["lr"])) - 0.01) < 1e-15
            and str(_scalar(data["spec"])) == SPEC
        )
        step_ok = np.array_equal(step, EXPECTED_STEPS)
        shape_ok = all(np.asarray(data[key]).shape == (len(EXPECTED_STEPS), WIDTH)
                       for key in UNIT_KEYS)
        shape_ok = shape_ok and all(
            np.asarray(data[key]).shape == (len(EXPECTED_STEPS),) for key in RUN_KEYS
        )
        if not shape_ok:
            raise SystemExit(f"seed={seed}: array shape mismatch")

        p_hat = np.asarray(data["p_hat"], dtype=np.float64)
        pre_max = np.asarray(data["pre_max"], dtype=np.float64)
        x = np.asarray(data["x"], dtype=np.float64)
        r = np.asarray(data["r"], dtype=np.float64)
        w_norm = np.asarray(data["w_norm"], dtype=np.float64)
        quant = float(np.max(np.abs(SUPPORT_SIZE * p_hat - np.rint(SUPPORT_SIZE * p_hat))))
        numerator = np.abs(x * x + r * r - w_norm * w_norm)
        denominator = w_norm * w_norm
        relative = np.divide(numerator, denominator, out=numerator.copy(),
                             where=denominator > 0)
        geometry = float(relative.max(initial=0.0))
        mismatch = int(np.count_nonzero((p_hat == 0.0) != (pre_max <= 0.0)))
        nonfinite = int(sum(np.count_nonzero(~np.isfinite(np.asarray(data[key], float)))
                            for key in UNIT_KEYS + RUN_KEYS))
        passed = (scalar_ok and step_ok and quant < 1e-12 and geometry < 1e-10
                  and mismatch == 0 and nonfinite == 0)
        per_seed.append(dict(seed=seed, scalar_ok=scalar_ok, step_ok=step_ok,
                             quant_error=quant, geometry_relative_error=geometry,
                             strict_dead_pre_max_mismatch=mismatch,
                             nonfinite=nonfinite, pass_=passed))
        max_quant = max(max_quant, quant)
        max_geometry = max(max_geometry, geometry)
        total_mismatch += mismatch
        total_nonfinite += nonfinite
    summary = dict(
        pass_all=all(row["pass_"] for row in per_seed),
        max_quantization_error=max_quant,
        max_geometry_relative_error=max_geometry,
        strict_dead_pre_max_mismatches=total_mismatch,
        nonfinite=total_nonfinite,
    )
    return per_seed, summary


def build_exposures(logs: list[dict[str, Any]]) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    for data in logs:
        seed = int(_scalar(data["seed"]))
        lookup = {int(step): index for index, step in enumerate(data["step"])}
        for boundary in SWITCHES:
            t0, t1 = boundary + 1, boundary + PERIOD
            i0, i1 = lookup[t0], lookup[t1]
            p0 = np.asarray(data["p_hat"][i0], dtype=np.float64)
            p1 = np.asarray(data["p_hat"][i1], dtype=np.float64)
            units = np.flatnonzero(p0 >= TAU)
            if units.size == 0:
                continue
            values: dict[str, Any] = {
                "seed": np.full(units.size, seed, dtype=np.int64),
                "unit": units.astype(np.int64),
                "switch": np.full(units.size, boundary, dtype=np.int64),
                "t0": np.full(units.size, t0, dtype=np.int64),
                "t1": np.full(units.size, t1, dtype=np.int64),
                "epoch": np.full(units.size, _epoch(boundary), dtype=object),
                "p_hat": p0[units],
                "p_count": np.rint(SUPPORT_SIZE * p0[units]).astype(np.int64),
                "end_strict_dead": (p1[units] == 0.0).astype(np.int64),
                "end_dead_0_05": (p1[units] < TAU).astype(np.int64),
                "eval_nmse": np.full(units.size, float(data["eval_nmse"][i0])),
                "y_var": np.full(units.size, float(data["y_var"][i0])),
            }
            for key in ("pre_max", "x", "r", "w_norm", "b", "v",
                        "utility_raw", "utility_nmse"):
                values[key] = np.asarray(data[key][i0], dtype=np.float64)[units]
            chunks.append(pd.DataFrame(values))
    if not chunks:
        raise SystemExit("t0 p_hat>=0.05のrisk exposureが0件")
    frame = pd.concat(chunks, ignore_index=True)
    frame = frame.sort_values(["seed", "t0", "unit"], kind="mergesort").reset_index(drop=True)
    return frame


def _quantile_cuts(values: np.ndarray, probabilities: Iterable[float]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("quantile input must be nonempty and finite")
    return np.asarray(np.quantile(values, tuple(probabilities)), dtype=np.float64)


def _utility_labels(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cuts = _quantile_cuts(values, (1 / 3, 2 / 3))
    bins = np.searchsorted(cuts, np.asarray(values, np.float64), side="left")
    labels = np.asarray(GROUPS, dtype=object)[bins]
    return labels, cuts


def assign_cells(frame: pd.DataFrame, *, margin_bins: int,
                 utility_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign exact-p_count geometry cells and within-cell utility tertiles."""
    out = frame.copy()
    out["margin_bin"] = -1
    margin_cut_map: dict[tuple[int, int], np.ndarray] = {}
    probabilities = tuple(i / margin_bins for i in range(1, margin_bins))
    for key, indices in out.groupby(["t0", "p_count"], sort=True).groups.items():
        idx = np.asarray(indices, dtype=np.int64)
        cuts = _quantile_cuts(out.loc[idx, "pre_max"].to_numpy(float), probabilities)
        out.loc[idx, "margin_bin"] = np.searchsorted(
            cuts, out.loc[idx, "pre_max"].to_numpy(float), side="left"
        )
        margin_cut_map[(int(key[0]), int(key[1]))] = cuts
    out["margin_bin"] = out["margin_bin"].astype(np.int64)

    tuples = list(zip(out.t0.astype(int), out.p_count.astype(int),
                      out.margin_bin.astype(int)))
    unique = sorted(set(tuples))
    code = {key: index for index, key in enumerate(unique)}
    out["cell_id"] = np.asarray([code[key] for key in tuples], dtype=np.int64)
    out["utility_group"] = ""
    utility_cut_map: dict[int, np.ndarray] = {}
    for cell_id, indices in out.groupby("cell_id", sort=True).groups.items():
        idx = np.asarray(indices, dtype=np.int64)
        labels, cuts = _utility_labels(out.loc[idx, utility_col].to_numpy(float))
        out.loc[idx, "utility_group"] = labels
        utility_cut_map[int(cell_id)] = cuts

    cell_rows: list[dict[str, Any]] = []
    valid_map: dict[int, bool] = {}
    for cell_id, part in out.groupby("cell_id", sort=True):
        t0 = int(part.t0.iloc[0])
        p_count = int(part.p_count.iloc[0])
        margin_bin = int(part.margin_bin.iloc[0])
        counts = {group: int((part.utility_group == group).sum()) for group in GROUPS}
        valid = counts["low"] > 0 and counts["high"] > 0
        valid_map[int(cell_id)] = valid
        mcuts = margin_cut_map[(t0, p_count)]
        ucuts = utility_cut_map[int(cell_id)]
        row: dict[str, Any] = dict(
            cell_id=int(cell_id), t0=t0, p_count=p_count,
            margin_bin=margin_bin, margin_bins=margin_bins,
            utility=utility_col, n_total=int(len(part)),
            n_low=counts["low"], n_mid=counts["mid"], n_high=counts["high"],
            utility_cut_1=float(ucuts[0]), utility_cut_2=float(ucuts[1]),
            valid_low_high=valid,
        )
        for index in range(margin_bins - 1):
            row[f"margin_cut_{index + 1}"] = float(mcuts[index])
        for outcome in ("end_strict_dead", "end_dead_0_05"):
            for group in GROUPS:
                values = part.loc[part.utility_group == group, outcome]
                row[f"{outcome}_events_{group}"] = int(values.sum())
                row[f"{outcome}_risk_{group}"] = (
                    float(values.mean()) if len(values) else np.nan
                )
        cell_rows.append(row)
    out["cell_valid"] = out.cell_id.map(valid_map).astype(bool)
    cells = pd.DataFrame(cell_rows).sort_values("cell_id").reset_index(drop=True)
    return out, cells


def assign_existing_geometry(frame: pd.DataFrame, *, utility_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Re-tertile utility while preserving the main geometry-cell labels."""
    out = frame.copy()
    out["utility_group"] = ""
    rows: list[dict[str, Any]] = []
    valid_map: dict[int, bool] = {}
    for cell_id, indices in out.groupby("cell_id", sort=True).groups.items():
        idx = np.asarray(indices, dtype=np.int64)
        labels, cuts = _utility_labels(out.loc[idx, utility_col].to_numpy(float))
        out.loc[idx, "utility_group"] = labels
        part = out.loc[idx]
        counts = {group: int((part.utility_group == group).sum()) for group in GROUPS}
        valid = counts["low"] > 0 and counts["high"] > 0
        valid_map[int(cell_id)] = valid
        rows.append(dict(cell_id=int(cell_id), utility=utility_col,
                         utility_cut_1=float(cuts[0]), utility_cut_2=float(cuts[1]),
                         n_total=int(len(part)), n_low=counts["low"],
                         n_mid=counts["mid"], n_high=counts["high"],
                         valid_low_high=valid))
    out["cell_valid"] = out.cell_id.map(valid_map).astype(bool)
    return out, pd.DataFrame(rows)


def assign_t0_tertiles(frame: pd.DataFrame, *, utility_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Unadjusted sensitivity: only t0-specific utility tertiles."""
    out = frame.copy()
    out["cell_id"] = out.t0.map({value: index for index, value in
                                  enumerate(sorted(out.t0.unique()))}).astype(int)
    out["margin_bin"] = -1
    return assign_existing_geometry(out, utility_col=utility_col)


def _count_arrays(assigned: pd.DataFrame, outcome: str,
                  seeds: tuple[int, ...] = SEEDS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = assigned[assigned.cell_valid & assigned.utility_group.isin(("low", "high"))]
    cells = np.asarray(sorted(data.cell_id.unique()), dtype=np.int64)
    if cells.size == 0:
        raise ValueError("no valid low/high cells")
    seed_map = {seed: index for index, seed in enumerate(seeds)}
    cell_map = {cell: index for index, cell in enumerate(cells)}
    n = np.zeros((len(seeds), len(cells), 2), dtype=np.float64)
    e = np.zeros_like(n)
    for row in data[["seed", "cell_id", "utility_group", outcome]].itertuples(index=False):
        si = seed_map[int(row.seed)]
        ci = cell_map[int(row.cell_id)]
        gi = 0 if row.utility_group == "low" else 1
        n[si, ci, gi] += 1.0
        e[si, ci, gi] += float(getattr(row, outcome))
    return n, e, cells


def _rd_from_counts(n: np.ndarray, e: np.ndarray) -> np.ndarray:
    valid = (n[..., 0] > 0) & (n[..., 1] > 0)
    weight = np.minimum(n[..., 0], n[..., 1]) * valid
    low = np.divide(e[..., 0], n[..., 0], out=np.zeros_like(e[..., 0]),
                    where=n[..., 0] > 0)
    high = np.divide(e[..., 1], n[..., 1], out=np.zeros_like(e[..., 1]),
                     where=n[..., 1] > 0)
    numerator = (weight * (high - low)).sum(axis=-1)
    denominator = weight.sum(axis=-1)
    return np.divide(numerator, denominator,
                     out=np.full_like(numerator, np.nan, dtype=np.float64),
                     where=denominator > 0)


def classify(ci_lo: float, ci_hi: float) -> str:
    """Mutually exclusive preregistered decision rules, in registered order."""
    if ci_lo >= -EQUIV_MARGIN and ci_hi <= EQUIV_MARGIN:
        return "EQUIV"
    if ci_hi < -EQUIV_MARGIN:
        return "PROTECTIVE"
    if ci_lo > EQUIV_MARGIN:
        return "HARMFUL"
    return "INCONCLUSIVE"


def adjusted_effect(assigned: pd.DataFrame, outcome: str,
                    bootstrap_indices: np.ndarray) -> dict[str, Any]:
    n, e, cells = _count_arrays(assigned, outcome)
    point = float(_rd_from_counts(n.sum(axis=0), e.sum(axis=0)))
    boot = np.empty(len(bootstrap_indices), dtype=np.float64)
    for start in range(0, len(bootstrap_indices), BOOT_BATCH):
        stop = min(start + BOOT_BATCH, len(bootstrap_indices))
        indices = bootstrap_indices[start:stop]
        boot[start:stop] = _rd_from_counts(
            n[indices].sum(axis=1), e[indices].sum(axis=1)
        )
    finite = np.isfinite(boot)
    nonfinite = int((~finite).sum())
    if not finite.any():
        lo = hi = np.nan
    else:
        lo, hi = np.quantile(boot[finite], (0.025, 0.975))
    return dict(
        rd=point, ci_lo=float(lo), ci_hi=float(hi),
        verdict=(classify(float(lo), float(hi)) if nonfinite == 0 else "NOT_ISSUED"),
        n_boot=int(len(boot)), n_boot_nonfinite=nonfinite,
        n_cell=int(len(cells)), weight=float(
            np.minimum(n.sum(axis=0)[:, 0], n.sum(axis=0)[:, 1]).sum()
        ),
        n_low=int(n[:, :, 0].sum()), n_high=int(n[:, :, 1].sum()),
        events_low=int(e[:, :, 0].sum()), events_high=int(e[:, :, 1].sum()),
        risk_low=float(e[:, :, 0].sum() / n[:, :, 0].sum()),
        risk_high=float(e[:, :, 1].sum() / n[:, :, 1].sum()),
    )


def rate_table(assigned: pd.DataFrame, outcome: str) -> pd.DataFrame:
    data = assigned[assigned.cell_valid]
    rows: list[dict[str, Any]] = []
    for seed, part, scope in [(-1, data, "pooled")]:
        for group in GROUPS:
            values = part.loc[part.utility_group == group, outcome]
            rows.append(dict(scope=scope, seed=seed, group=group, outcome=outcome,
                             n=int(len(values)), n_event=int(values.sum()),
                             risk=float(values.mean()) if len(values) else np.nan))
    for seed in SEEDS:
        part = data[data.seed == seed]
        for group in GROUPS:
            values = part.loc[part.utility_group == group, outcome]
            rows.append(dict(scope="seed", seed=seed, group=group, outcome=outcome,
                             n=int(len(values)), n_event=int(values.sum()),
                             risk=float(values.mean()) if len(values) else np.nan))
    return pd.DataFrame(rows)


def repeat_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (seed, unit), part in frame.groupby(["seed", "unit"], sort=True):
        rows.append(dict(
            seed=int(seed), unit=int(unit), n_exposure=int(len(part)),
            n_strict_dead=int(part.end_strict_dead.sum()),
            n_dead_0_05=int(part.end_dead_0_05.sum()),
            first_t0=int(part.t0.min()), last_t0=int(part.t0.max()),
        ))
    return pd.DataFrame(rows).sort_values(["seed", "unit"]).reset_index(drop=True)


def _effect_row(name: str, outcome: str, utility: str, geometry: str,
                effect: dict[str, Any], *, epoch: str = "all") -> dict[str, Any]:
    return dict(
        analysis=name, role="secondary_not_main", outcome=outcome,
        utility=utility, geometry=geometry, epoch=epoch, group="high-low",
        n=effect["n_low"] + effect["n_high"], n_event=effect["events_low"] + effect["events_high"],
        risk=np.nan, rd=effect["rd"], ci_lo=effect["ci_lo"], ci_hi=effect["ci_hi"],
        classification=effect["verdict"], n_boot=effect["n_boot"],
        n_boot_nonfinite=effect["n_boot_nonfinite"], n_cell=effect["n_cell"],
        weight=effect["weight"], n_low=effect["n_low"], n_high=effect["n_high"],
        events_low=effect["events_low"], events_high=effect["events_high"],
        risk_low=effect["risk_low"], risk_high=effect["risk_high"],
    )


def secondary_analyses(frame: pd.DataFrame, primary: pd.DataFrame,
                       bootstrap_indices: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    # §6.1: same primary cells, secondary endpoint.
    effect = adjusted_effect(primary, "end_dead_0_05", bootstrap_indices)
    rows.append(_effect_row("S1_primary_cells_dead_0_05", "end_dead_0_05",
                            "utility_nmse", "p_count_x_margin5", effect))

    # §6.2: margin deciles, otherwise identical.
    decile, _ = assign_cells(frame, margin_bins=10, utility_col="utility_nmse")
    effect = adjusted_effect(decile, "end_strict_dead", bootstrap_indices)
    rows.append(_effect_row("S2_margin10_sensitivity", "end_strict_dead",
                            "utility_nmse", "p_count_x_margin10", effect))

    # §6.3: t0-specific utility tertiles only.
    unadjusted, _ = assign_t0_tertiles(frame, utility_col="utility_nmse")
    effect = adjusted_effect(unadjusted, "end_strict_dead", bootstrap_indices)
    rows.append(_effect_row("S3_unadjusted_t0_tertiles", "end_strict_dead",
                            "utility_nmse", "t0_only", effect))

    # §6.4: raw utility tertiles in the primary geometry cells.
    raw, _ = assign_existing_geometry(primary, utility_col="utility_raw")
    effect = adjusted_effect(raw, "end_strict_dead", bootstrap_indices)
    rows.append(_effect_row("S4_utility_raw", "end_strict_dead",
                            "utility_raw", "p_count_x_margin5", effect))

    # §6.5: exact sign groups, descriptive rates only.
    sign = np.where(frame.utility_nmse < 0, "negative",
                    np.where(frame.utility_nmse == 0, "zero", "positive"))
    for group in ("negative", "zero", "positive"):
        values = frame.loc[sign == group, "end_strict_dead"]
        rows.append(dict(
            analysis="S5_utility_sign_rate", role="secondary_not_main",
            outcome="end_strict_dead", utility="utility_nmse",
            geometry="none", epoch="all", group=group,
            n=int(len(values)), n_event=int(values.sum()),
            risk=float(values.mean()) if len(values) else np.nan,
            rd=np.nan, ci_lo=np.nan, ci_hi=np.nan, classification="DESCRIPTIVE",
            n_boot=0, n_boot_nonfinite=0, n_cell=0, weight=np.nan,
            n_low=np.nan, n_high=np.nan, events_low=np.nan, events_high=np.nan,
            risk_low=np.nan, risk_high=np.nan,
        ))

    # §6.6: primary labels/cells retained, then partitioned by fixed epochs.
    for epoch in ("200-390k", "400-590k", "600-800k"):
        subset = primary[primary.epoch == epoch]
        effect = adjusted_effect(subset, "end_strict_dead", bootstrap_indices)
        rows.append(_effect_row("S6_epoch_primary_rd", "end_strict_dead",
                                "utility_nmse", "p_count_x_margin5", effect,
                                epoch=epoch))
    return pd.DataFrame(rows)


def primary_cell_table(cells: pd.DataFrame) -> pd.DataFrame:
    ordered = [
        "cell_id", "t0", "p_count", "margin_bin", "margin_bins", "utility",
        "n_total", "n_low", "n_mid", "n_high", "valid_low_high",
    ]
    ordered += [column for column in cells.columns if column.startswith("margin_cut_")]
    ordered += ["utility_cut_1", "utility_cut_2"]
    ordered += [column for column in cells.columns if column not in ordered]
    return cells[ordered]


def make_sanity_rows(meta_rows: list[dict[str, Any]], log_summary: dict[str, Any],
                     frame: pd.DataFrame, primary: pd.DataFrame,
                     cells: pd.DataFrame, effect: dict[str, Any],
                     repeat: pd.DataFrame, *, determinism_pass: bool) -> pd.DataFrame:
    seed_stats = frame.groupby("seed", sort=True).end_strict_dead.agg(["size", "sum"])
    n_invalid_cells = int((~cells.valid_low_high).sum())
    excluded = int(cells.loc[~cells.valid_low_high, "n_total"].sum())
    low_n = int((primary.cell_valid & (primary.utility_group == "low")).sum())
    high_n = int((primary.cell_valid & (primary.utility_group == "high")).sum())
    rows = [
        dict(id="C-S1", status="PASS" if log_summary["pass_all"] else "FAIL",
             value=len(frame), threshold="R=20; 122 records; 61 t0/t1 pairs",
             detail=f"seeds={frame.seed.nunique()},t0={frame.t0.nunique()}"),
        *meta_rows,
        dict(id="C-S3", status="PASS" if (
                 log_summary["max_quantization_error"] < 1e-12
                 and log_summary["max_geometry_relative_error"] < 1e-10) else "FAIL",
             value=log_summary["max_geometry_relative_error"],
             threshold="p quant <1e-12; geometry <1e-10",
             detail=f"p_quant={log_summary['max_quantization_error']:.17g}"),
        dict(id="C-S4", status="PASS" if (
                 log_summary["strict_dead_pre_max_mismatches"] == 0
                 and log_summary["nonfinite"] == 0) else "FAIL",
             value=log_summary["strict_dead_pre_max_mismatches"],
             threshold="identity mismatches=0; nonfinite=0",
             detail=f"nonfinite={log_summary['nonfinite']}"),
        dict(id="C-S7", status="PASS" if (
                 effect["n_boot_nonfinite"] == 0 and effect["n_cell"] > 0
                 and low_n > 0 and high_n > 0) else "FAIL",
             value=effect["n_cell"], threshold="valid cells and finite bootstrap",
             detail=_json(dict(
                 total_cells=int(len(cells)), invalid_cells=n_invalid_cells,
                 excluded_exposures=excluded, low=low_n, high=high_n,
                 seed_risk_min=int(seed_stats["size"].min()),
                 seed_risk_max=int(seed_stats["size"].max()),
                 seed_event_min=int(seed_stats["sum"].min()),
                 seed_event_max=int(seed_stats["sum"].max()),
                 repeat_min=int(repeat.n_exposure.min()),
                 repeat_median=float(repeat.n_exposure.median()),
                 repeat_max=int(repeat.n_exposure.max()),
                 bootstrap_nonfinite=effect["n_boot_nonfinite"],
             ))),
        dict(id="C-S8", status="PASS" if determinism_pass else "FAIL",
             value=len(CSV_NAMES), threshold="all CSV byte-identical",
             detail="two independent analysis builds with RNG reset"),
    ]
    result = pd.DataFrame(rows)[["id", "status", "value", "threshold", "detail"]]
    order = {f"C-S{index}": index for index in range(1, 9)}
    result["_order"] = result.id.map(order)
    return result.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def build_outputs(logs: list[dict[str, Any]], meta_rows: list[dict[str, Any]],
                  *, bootstrap_n: int = BOOT_N,
                  assume_deterministic: bool = True) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    _, log_summary = log_sanity(logs)
    if not log_summary["pass_all"]:
        raise SystemExit(f"NPZ sanity FAIL: {log_summary}")
    frame = build_exposures(logs)
    primary, cells = assign_cells(frame, margin_bins=5, utility_col="utility_nmse")
    rng = np.random.default_rng(BOOT_SEED)
    bootstrap_indices = rng.integers(0, len(SEEDS), size=(bootstrap_n, len(SEEDS)))
    effect = adjusted_effect(primary, "end_strict_dead", bootstrap_indices)
    if effect["n_boot_nonfinite"]:
        raise SystemExit(
            f"primary bootstrap nonfinite={effect['n_boot_nonfinite']}; verdictを出さない"
        )
    repeat = repeat_table(frame)
    sanity = make_sanity_rows(meta_rows, log_summary, frame, primary, cells,
                              effect, repeat, determinism_pass=assume_deterministic)
    if not (sanity.status == "PASS").all():
        raise SystemExit("sanity FAIL; verdictを出さない: " +
                         _json(sanity.to_dict(orient="records")))

    verdict = pd.DataFrame([dict(
        analysis="PRIMARY", role="primary", outcome="end_strict_dead",
        utility="utility_nmse", geometry="t0_x_exact_p_count_x_margin5",
        rd=effect["rd"], ci_lo=effect["ci_lo"], ci_hi=effect["ci_hi"],
        equiv_margin=EQUIV_MARGIN, verdict=effect["verdict"],
        rule_order="EQUIV>PROTECTIVE>HARMFUL>INCONCLUSIVE",
        n_boot=effect["n_boot"], bootstrap_seed=BOOT_SEED,
        n_boot_nonfinite=effect["n_boot_nonfinite"], n_cell=effect["n_cell"],
        weight=effect["weight"], n_low=effect["n_low"], n_high=effect["n_high"],
        events_low=effect["events_low"], events_high=effect["events_high"],
        risk_low=effect["risk_low"], risk_high=effect["risk_high"],
    )])
    exposures = primary.rename(columns={
        "margin_bin": "margin5_bin",
        "utility_group": "utility_nmse_group",
        "cell_valid": "primary_cell_valid",
    })
    exposures = exposures.sort_values(["seed", "t0", "unit"], kind="mergesort")
    secondary = secondary_analyses(frame, primary, bootstrap_indices)
    frames = {
        "exposures.csv": exposures.reset_index(drop=True),
        "primary_cells.csv": primary_cell_table(cells),
        "primary_rates.csv": rate_table(primary, "end_strict_dead"),
        "verdict.csv": verdict,
        "secondary_results.csv": secondary,
        "repeat_exposure.csv": repeat,
        "sanity.csv": sanity,
    }
    diagnostics = dict(
        n_exposure=int(len(frame)), n_seed=int(frame.seed.nunique()),
        n_t0=int(frame.t0.nunique()), n_cell=int(len(cells)),
        n_valid_cell=int(cells.valid_low_high.sum()),
        n_invalid_cell=int((~cells.valid_low_high).sum()),
        excluded_exposures=int(cells.loc[~cells.valid_low_high, "n_total"].sum()),
        primary=effect, log_sanity=log_summary,
    )
    return frames, diagnostics


def write_summary(outdir: Path, diagnostics: dict[str, Any], verdict: pd.Series,
                  secondary: pd.DataFrame, runner_meta: dict[str, Any]) -> None:
    primary = diagnostics["primary"]
    epoch_rows = secondary[secondary.analysis == "S6_epoch_primary_rd"]
    lines = [
        "# function_blind_direct_0823 confirmation",
        "",
        "> generator_offset=20260830の独立20系列。pilotとは合算していない。",
        "",
        "## 主判定",
        "",
        f"- **{verdict['verdict']}**",
        f"- 調整RD (high−low): {verdict['rd']:+.4f} "
        f"[{verdict['ci_lo']:+.4f}, {verdict['ci_hi']:+.4f}]",
        f"- 意味のある差の境界: ±{EQUIV_MARGIN:.2f}",
        f"- low/high pooled率（記述値）: {primary['risk_low']:.4f} / "
        f"{primary['risk_high']:.4f}",
        "",
        "判定規則は EQUIV → PROTECTIVE → HARMFUL → INCONCLUSIVE の順に"
        "排他的に適用した。主判定は strict_dead × utility_nmse × "
        "exact p_count・pre_max五分位だけである。",
        "",
        "## データとセル",
        "",
        f"- 曝露: {diagnostics['n_exposure']:,} "
        f"（seed={diagnostics['n_seed']}, t0={diagnostics['n_t0']}）",
        f"- 幾何セル: {diagnostics['n_cell']:,}、有効 "
        f"{diagnostics['n_valid_cell']:,}、除外 {diagnostics['n_invalid_cell']:,}",
        f"- 無効セルに属する除外曝露: {diagnostics['excluded_exposures']:,}",
        f"- bootstrap: seed block B={BOOT_N:,}, RNG seed={BOOT_SEED}, "
        f"nonfinite={primary['n_boot_nonfinite']}",
        "",
        "## 固定副次解析",
        "",
    ]
    for row in secondary.itertuples(index=False):
        if row.group == "high-low":
            suffix = f"/{row.epoch}" if row.epoch != "all" else ""
            lines.append(
                f"- {row.analysis}{suffix}: RD={row.rd:+.4f} "
                f"[{row.ci_lo:+.4f}, {row.ci_hi:+.4f}] "
                f"({row.classification}; secondary)"
            )
        elif row.analysis == "S5_utility_sign_rate":
            lines.append(f"- utility sign {row.group}: rate={row.risk:.4f} "
                         f"({int(row.n_event)}/{int(row.n)})")
    lines += [
        "",
        "副次解析は主結果の差し替えに使わない。",
        "",
        "## サニティ",
        "",
        "- C-S1〜C-S8: **PASS**",
        f"- runner implementation: `{runner_meta.get('implementation_git', 'unknown')}`",
        "- 全CSVは入力・commit・RNGを固定した独立二重集計でbyte一致。",
        "",
        "## 解釈範囲",
        "",
        "- ΔLは現在タスク32入力上の単独消音損失であり、普遍的価値や"
        "将来タスク価値ではない。",
        "- 観察解析であり、機能を人工的に入れ替えた因果介入ではない。",
        "- exact p_countとpre_max分位で層別した範囲を越えて交絡消失を主張しない。",
        "- condB、他幅、他教師、長期将来へ外挿しない。",
    ]
    (outdir / "summary.md").write_text("\n".join(lines) + "\n")


def run(logdir: Path, outdir: Path, *, bootstrap_n: int = BOOT_N) -> None:
    if os.environ.get("OMP_NUM_THREADS") != "1":
        raise SystemExit("OMP_NUM_THREADS=1 is required")
    if bootstrap_n != BOOT_N:
        raise SystemExit(f"registered confirmation requires bootstrap_n={BOOT_N}")
    outdir.mkdir(parents=True, exist_ok=True)
    meta_path = outdir / "instrumentation_meta.json"
    runner_meta, meta_rows = validate_instrumentation_meta(meta_path)
    logs = load_logs(logdir)

    # Two independent builds reset the bootstrap RNG and redo every assignment.
    first, diagnostics = build_outputs(logs, meta_rows, bootstrap_n=bootstrap_n)
    second, _ = build_outputs(logs, meta_rows, bootstrap_n=bootstrap_n)
    first_bytes = {name: _csv_bytes(first[name]) for name in CSV_NAMES}
    second_bytes = {name: _csv_bytes(second[name]) for name in CSV_NAMES}
    mismatch = [name for name in CSV_NAMES if first_bytes[name] != second_bytes[name]]
    if mismatch:
        raise SystemExit("C-S8 determinism FAIL; verdictを出さない: " + ", ".join(mismatch))

    # Serialize through a temporary directory too, guarding the actual writer.
    with tempfile.TemporaryDirectory(prefix="function_blind_direct_confirm_") as temp:
        tempdir = Path(temp)
        for name, payload in first_bytes.items():
            (tempdir / name).write_bytes(payload)
            if (tempdir / name).read_bytes() != second_bytes[name]:
                raise SystemExit(f"C-S8 serialization FAIL: {name}")
    for name, payload in first_bytes.items():
        (outdir / name).write_bytes(payload)

    hashes = {name: hashlib.sha256(first_bytes[name]).hexdigest() for name in CSV_NAMES}
    det_lines = [
        "# determinism check", "",
        "- result: **PASS**",
        "- method: same NPZ inputsから、セル割当とRNGをリセットして解析全体を2回構築",
        f"- bootstrap: B={BOOT_N}, seed={BOOT_SEED}", "",
        "| CSV | SHA-256 |", "|---|---|",
    ] + [f"| {name} | `{hashes[name]}` |" for name in CSV_NAMES]
    (outdir / "determinism_check.md").write_text("\n".join(det_lines) + "\n")

    verdict = first["verdict.csv"].iloc[0]
    write_summary(outdir, diagnostics, verdict, first["secondary_results.csv"], runner_meta)
    meta = dict(
        spec=SPEC, analysis_git=git_hash(),
        instrumentation_git=runner_meta.get("implementation_git"),
        mode="confirmation", generator_offset=GENERATOR_OFFSET,
        seeds=list(SEEDS), n_seed=len(SEEDS), width=WIDTH, period=PERIOD,
        support_size=SUPPORT_SIZE, bootstrap_n=BOOT_N,
        bootstrap_seed=BOOT_SEED, equiv_margin=EQUIV_MARGIN,
        grouping=dict(
            p_count="round(32*p_hat)", margin_bins=5,
            quantile="numpy default linear", ties="searchsorted(side=left)",
            utility="utility_nmse within geometry cell",
            weight="min(n_high,n_low)", labels_fixed_before_bootstrap=True,
        ),
        input_sha256={data["path"].name: sha256(data["path"]) for data in logs},
        instrumentation_meta_sha256=sha256(meta_path),
        csv_sha256=hashes, diagnostics=diagnostics,
        primary_verdict=verdict.to_dict(),
        restrictions=[
            "do not combine with pilot", "observational, not causal intervention",
            "do not extrapolate beyond condA/w100/current-task delta-L",
        ],
    )
    (outdir / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False, allow_nan=False,
                   default=lambda value: value.item() if isinstance(value, np.generic) else str(value))
        + "\n"
    )
    print((outdir / "summary.md").read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--logs", type=Path,
        default=ROOT / "results/function_blind_direct_0823_confirm/logs",
    )
    parser.add_argument(
        "--outdir", type=Path,
        default=ROOT / "results/function_blind_direct_0823_confirm",
    )
    args = parser.parse_args()
    run(args.logs, args.outdir)


if __name__ == "__main__":
    main()
