"""Registered M2 analysis of mean ReLU-opening trajectories.

Invocation::

  OMP_NUM_THREADS=1 .venv/bin/python -m \
    analysis.function_blind_m2.function_blind_m2 \
    --replay results/function_blind_m2_0828 \
    --reference results/function_blind_direct_0823_confirm
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SPEC = "specs/spec_function_blind_m2_0828.md"
SEEDS = tuple(range(20))
GROUPS = ("low", "mid", "high")
METRICS = ("S0", "S1", "delta_S")
BOOT_N = 10_000
BOOT_SEED = 20_260_901
EQUIV_MARGIN = 0.05
EXPECTED_EXPOSURES_SHA256 = (
    "2edc9aa82185843d8fd7f9663380b60590cd75027b27601f19546b39ef7b126b"
)
EXPECTED_INSTRUMENT_META_SHA256 = (
    "a191e440fb9da5ed7a61c3491100911c3f0c09848fffa088160df4d96c6cd8b3"
)
CSV_NAMES = (
    "joined_exposures.csv", "group_distributions.csv",
    "per_seed_distributions.csv", "cell_effects.csv", "estimates.csv",
    "bootstrap.csv", "sanity.csv", "verdict.csv",
)


def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(
        index=False, lineterminator="\n", float_format="%.17g"
    ).encode("utf-8")


def pair_score(high: np.ndarray, low: np.ndarray) -> np.ndarray:
    """Return I(high>low)+0.5*I(equal) for all high/low pairs."""
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    return ((high[:, None] > low[None, :]).astype(np.float64)
            + 0.5 * (high[:, None] == low[None, :]))


def classify_dynamics(lo: float, hi: float, margin: float = EQUIV_MARGIN) -> str:
    if lo >= -margin and hi <= margin:
        return "EQUIV_DYNAMICS"
    if lo > margin:
        return "HIGH_LESS_PUSHED"
    if hi < -margin:
        return "HIGH_MORE_PUSHED"
    return "INCONCLUSIVE"


def classify_baseline(lo: float, hi: float, margin: float = EQUIV_MARGIN) -> str:
    if lo > margin:
        return "HIGH_STARTS_MORE_OPEN"
    if hi < -margin:
        return "HIGH_STARTS_LESS_OPEN"
    if lo >= -margin and hi <= margin:
        return "EQUIV_BASELINE"
    return "INCONCLUSIVE_BASELINE"


def validate_inputs(replay_dir: Path, reference_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    replay_meta_path = replay_dir / "replay_meta.json"
    instrument_path = reference_dir / "instrumentation_meta.json"
    exposures_path = reference_dir / "exposures.csv"
    original_meta_path = reference_dir / "meta.json"
    for path in (replay_meta_path, instrument_path, exposures_path, original_meta_path):
        if not path.exists():
            raise SystemExit(f"required M2 input is missing: {path}")
    replay_meta = json.loads(replay_meta_path.read_text())
    original_meta = json.loads(original_meta_path.read_text())
    checks = dict(
        exposures_hash=sha256(exposures_path) == EXPECTED_EXPOSURES_SHA256,
        instrument_hash=(sha256(instrument_path)
                         == EXPECTED_INSTRUMENT_META_SHA256),
        replay_sanity=(replay_meta.get("sanity", {}).get("all_required_pass") is True),
        replay_shape=(replay_meta.get("seeds") == list(SEEDS)
                      and replay_meta.get("R") == 20
                      and replay_meta.get("n_records") == 122
                      and replay_meta.get("n_npz") == 20
                      and replay_meta.get("total_steps") == 810_000
                      and replay_meta.get("generator_offset") == 20_260_830),
        replay_not_smoke=replay_meta.get("smoke") is False,
        replay_spec=replay_meta.get("spec") == SPEC,
    )
    failed = [key for key, passed in checks.items() if not passed]
    if failed:
        raise SystemExit("M2 input/replay gate FAIL: " + ", ".join(failed))
    return replay_meta, original_meta


def load_s_logs(replay_dir: Path) -> dict[int, dict[str, np.ndarray]]:
    paths = sorted((replay_dir / "logs").glob("seed*.npz"), key=lambda p: p.name)
    if len(paths) != len(SEEDS):
        raise SystemExit(f"M2 replay requires 20 S logs, found={len(paths)}")
    out: dict[int, dict[str, np.ndarray]] = {}
    for path in paths:
        with np.load(path, allow_pickle=False) as z:
            required = ("step", "seed", "S", "support_size", "spec")
            missing = [key for key in required if key not in z.files]
            if missing:
                raise SystemExit(f"{path}: missing keys={missing}")
            seed = int(np.asarray(z["seed"]).item())
            step = np.asarray(z["step"], dtype=np.int64)
            S = np.asarray(z["S"], dtype=np.float64)
            support_size = int(np.asarray(z["support_size"]).item())
            spec = str(np.asarray(z["spec"]).item())
        if S.shape != (122, 100) or step.shape != (122,):
            raise SystemExit(f"{path}: unexpected shape step={step.shape}, S={S.shape}")
        if support_size != 32 or spec != SPEC:
            raise SystemExit(f"{path}: support/spec mismatch")
        if seed in out:
            raise SystemExit(f"duplicate seed log: {seed}")
        out[seed] = dict(step=step, S=S, path=np.asarray(str(path)))
    if sorted(out) != list(SEEDS):
        raise SystemExit(f"M2 S log seeds differ: {sorted(out)}")
    return out


def join_opening(exposures: pd.DataFrame,
                 logs: dict[int, dict[str, np.ndarray]]) -> pd.DataFrame:
    required = {
        "seed", "unit", "t0", "t1", "cell_id", "utility_nmse_group",
        "primary_cell_valid", "end_strict_dead",
    }
    missing = sorted(required - set(exposures.columns))
    if missing:
        raise SystemExit(f"exposures missing M2 columns: {missing}")
    if exposures.duplicated(["seed", "unit", "t0", "t1"]).any():
        raise SystemExit("exposure key is not unique")

    chunks: list[pd.DataFrame] = []
    for seed, part in exposures.groupby("seed", sort=True):
        seed = int(seed)
        data = logs[seed]
        lookup = {int(step): index for index, step in enumerate(data["step"])}
        t0 = part["t0"].to_numpy(dtype=np.int64)
        t1 = part["t1"].to_numpy(dtype=np.int64)
        try:
            i0 = np.asarray([lookup[int(step)] for step in t0], dtype=np.int64)
            i1 = np.asarray([lookup[int(step)] for step in t1], dtype=np.int64)
        except KeyError as error:
            raise SystemExit(f"seed={seed}: S landmark missing {error}") from error
        units = part["unit"].to_numpy(dtype=np.int64)
        S = data["S"]
        piece = part.copy()
        piece["S0"] = S[i0, units]
        piece["S1"] = S[i1, units]
        piece["delta_S"] = piece["S1"] - piece["S0"]
        chunks.append(piece)
    joined = pd.concat(chunks, ignore_index=True)
    return joined.sort_values(
        ["seed", "t0", "unit"], kind="mergesort"
    ).reset_index(drop=True)


def summarize(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    quantiles = np.quantile(values, (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95))
    return dict(
        n=int(values.size), mean=float(values.mean()), sd=float(values.std(ddof=0)),
        q05=float(quantiles[0]), q10=float(quantiles[1]), q25=float(quantiles[2]),
        median=float(quantiles[3]), q75=float(quantiles[4]),
        q90=float(quantiles[5]), q95=float(quantiles[6]),
        p_negative=float(np.mean(values < 0.0)),
        p_zero=float(np.mean(values == 0.0)),
    )


def distribution_tables(joined: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    for group in GROUPS:
        part = joined[joined.utility_nmse_group == group]
        for metric in METRICS:
            group_rows.append(dict(
                group=group, metric=metric, n_seed=int(part.seed.nunique()),
                **summarize(part[metric].to_numpy(float)),
            ))
        for seed, seed_part in part.groupby("seed", sort=True):
            for metric in METRICS:
                seed_rows.append(dict(
                    seed=int(seed), group=group, metric=metric,
                    **summarize(seed_part[metric].to_numpy(float)),
                ))
    return pd.DataFrame(group_rows), pd.DataFrame(seed_rows)


def _cell_data(joined: pd.DataFrame) -> list[dict[str, Any]]:
    primary = joined[
        joined.primary_cell_valid
        & joined.utility_nmse_group.isin(("low", "high"))
    ]
    cells: list[dict[str, Any]] = []
    for cell_id, part in primary.groupby("cell_id", sort=True):
        high = part[part.utility_nmse_group == "high"]
        low = part[part.utility_nmse_group == "low"]
        if high.empty or low.empty:
            raise ValueError(f"registered valid cell lost a group: {cell_id}")
        cell: dict[str, Any] = dict(
            cell_id=int(cell_id),
            high_seed=high.seed.to_numpy(dtype=np.int64),
            low_seed=low.seed.to_numpy(dtype=np.int64),
        )
        for metric in METRICS:
            high_values = high[metric].to_numpy(dtype=np.float64)
            low_values = low[metric].to_numpy(dtype=np.float64)
            cell[f"{metric}_high"] = high_values
            cell[f"{metric}_low"] = low_values
            cell[f"{metric}_score"] = pair_score(high_values, low_values)
        cells.append(cell)
    return cells


def estimate_for_multiplicities(cells: list[dict[str, Any]],
                                multiplicities: np.ndarray) -> dict[str, np.ndarray]:
    """Estimate adjusted A and raw mean differences for K seed weights."""
    multiplicities = np.asarray(multiplicities, dtype=np.float64)
    if multiplicities.ndim == 1:
        multiplicities = multiplicities[None, :]
    if multiplicities.shape[1] != len(SEEDS):
        raise ValueError("multiplicity matrix must have 20 seed columns")
    k = multiplicities.shape[0]
    weights = np.zeros(k, dtype=np.float64)
    numerators = {
        f"A_{metric}": np.zeros(k, dtype=np.float64) for metric in METRICS
    }
    numerators.update({
        f"D_{metric}": np.zeros(k, dtype=np.float64) for metric in METRICS
    })

    for cell in cells:
        wh = multiplicities[:, cell["high_seed"]]
        wl = multiplicities[:, cell["low_seed"]]
        nh = wh.sum(axis=1)
        nl = wl.sum(axis=1)
        valid = (nh > 0.0) & (nl > 0.0)
        weight = np.minimum(nh, nl) * valid
        weights += weight
        for metric in METRICS:
            yh = cell[f"{metric}_high"]
            yl = cell[f"{metric}_low"]
            high_mean = np.divide(
                wh @ yh, nh, out=np.zeros(k, dtype=np.float64), where=nh > 0
            )
            low_mean = np.divide(
                wl @ yl, nl, out=np.zeros(k, dtype=np.float64), where=nl > 0
            )
            numerators[f"D_{metric}"] += weight * (high_mean - low_mean)

            score = cell[f"{metric}_score"]
            pair_numerator = np.einsum(
                "bi,ij,bj->b", wh, score, wl, optimize=True
            )
            pair_denominator = nh * nl
            ps = np.divide(
                pair_numerator, pair_denominator,
                out=np.full(k, 0.5, dtype=np.float64),
                where=pair_denominator > 0,
            )
            numerators[f"A_{metric}"] += weight * (ps - 0.5)

    out = {"weight": weights}
    for key, numerator in numerators.items():
        out[key] = np.divide(
            numerator, weights,
            out=np.full(k, np.nan, dtype=np.float64), where=weights > 0,
        )
    return out


def point_cell_effects(cells: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cell in cells:
        nh = int(len(cell["high_seed"]))
        nl = int(len(cell["low_seed"]))
        row: dict[str, Any] = dict(
            cell_id=cell["cell_id"], n_low=nl, n_high=nh, weight=min(nh, nl)
        )
        for metric in METRICS:
            high = cell[f"{metric}_high"]
            low = cell[f"{metric}_low"]
            ps = float(cell[f"{metric}_score"].mean())
            row[f"PS_{metric}"] = ps
            row[f"A_{metric}"] = ps - 0.5
            row[f"mean_low_{metric}"] = float(low.mean())
            row[f"mean_high_{metric}"] = float(high.mean())
            row[f"D_{metric}"] = float(high.mean() - low.mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("cell_id").reset_index(drop=True)


def bootstrap(cells: list[dict[str, Any]], *, B: int = BOOT_N,
              seed: int = BOOT_SEED, batch_size: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(SEEDS), size=(int(B), len(SEEDS)))
    multiplicities = np.zeros((int(B), len(SEEDS)), dtype=np.float64)
    rows = np.repeat(np.arange(int(B)), len(SEEDS))
    np.add.at(multiplicities, (rows, draws.ravel()), 1.0)

    keys = ("weight",) + tuple(
        name for prefix in ("A", "D") for name in
        (f"{prefix}_S0", f"{prefix}_S1", f"{prefix}_delta_S")
    )
    output = {key: np.empty(int(B), dtype=np.float64) for key in keys}
    for start in range(0, int(B), int(batch_size)):
        stop = min(start + int(batch_size), int(B))
        estimates = estimate_for_multiplicities(cells, multiplicities[start:stop])
        for key in keys:
            output[key][start:stop] = estimates[key]
    return pd.DataFrame(dict(replicate=np.arange(int(B), dtype=np.int64), **output))


def build_outputs(joined: pd.DataFrame, original_meta: dict[str, Any],
                  replay_meta: dict[str, Any]) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    sanity_rows: list[dict[str, Any]] = []
    diagnostics = original_meta["diagnostics"]
    primary = diagnostics["primary"]

    structure_checks = dict(
        n_exposure=len(joined) == 15_582,
        n_seed=joined.seed.nunique() == 20,
        n_t0=joined.t0.nunique() == 61,
        n_cell=joined.cell_id.nunique() == 6_002,
        n_valid_cell=joined.loc[joined.primary_cell_valid, "cell_id"].nunique() == 2_839,
    )
    finite = bool(np.isfinite(joined[list(METRICS)].to_numpy(float)).all())
    opening = bool(
        (joined.S0 > 0.0).all()
        and (joined.S1 >= 0.0).all()
        and np.array_equal(
            (joined.S1 == 0.0).to_numpy(),
            (joined.end_strict_dead == 1).to_numpy(),
        )
    )
    primary_rows = joined[
        joined.primary_cell_valid
        & joined.utility_nmse_group.isin(("low", "high"))
    ]
    n_low = int((primary_rows.utility_nmse_group == "low").sum())
    n_high = int((primary_rows.utility_nmse_group == "high").sum())
    fixed_counts = bool(
        n_low == int(primary["n_low"])
        and n_high == int(primary["n_high"])
        and primary_rows.cell_id.nunique() == int(primary["n_cell"])
    )

    sanity_rows += [
        dict(id="M2-S1", status="PASS" if all(structure_checks.values()) else "FAIL",
             value=json.dumps(structure_checks, sort_keys=True),
             threshold="registered exposure/cell structure", detail=""),
        dict(id="M2-S2", status="PASS",
             value=bool(replay_meta["sanity"]["final_state_match"]["pass_"]),
             threshold="exact original final-state hashes", detail="runner gate"),
        dict(id="M2-S3", status="PASS",
             value=bool(replay_meta["sanity"]["anchor_match"]["pass_"]),
             threshold="all original UNIT/RUN arrays exactly equal", detail="runner gate"),
        dict(id="M2-S4", status="PASS" if replay_meta["sanity"]["S_probe"]["pass_"] else "FAIL",
             value=replay_meta["sanity"]["S_probe"]["scalar_max_abs_error"],
             threshold="finite/nonnegative/S=0 iff p_hat=0/scalar error<1e-12", detail=""),
        dict(id="M2-S5", status="PASS" if len(joined) == 15_582 else "FAIL",
             value=len(joined), threshold="1:1 join, n=15582", detail="unique source key"),
        dict(id="M2-S6", status="PASS" if finite and opening else "FAIL",
             value=json.dumps(dict(finite=finite, opening_identity=opening)),
             threshold="S0>0; S1>=0; S1=0 iff strict_dead", detail=""),
        dict(id="M2-S7", status="PASS" if fixed_counts else "FAIL",
             value=json.dumps(dict(n_low=n_low, n_high=n_high,
                                   n_cell=int(primary_rows.cell_id.nunique()))),
             threshold="work-6 fixed primary rows/cells", detail=""),
    ]
    if not all(row["status"] == "PASS" for row in sanity_rows):
        failed = [row["id"] for row in sanity_rows if row["status"] != "PASS"]
        raise SystemExit("M2 pre-estimate sanity FAIL: " + ", ".join(failed))

    distributions, per_seed = distribution_tables(joined)
    cells = _cell_data(joined)
    cell_effects = point_cell_effects(cells)
    point = estimate_for_multiplicities(cells, np.ones(len(SEEDS)))
    boot = bootstrap(cells)
    finite_boot = bool(np.isfinite(boot.drop(columns="replicate").to_numpy(float)).all())
    sanity_rows.append(dict(
        id="M2-S8", status="PASS" if finite_boot else "FAIL",
        value=int(np.count_nonzero(~np.isfinite(
            boot.drop(columns="replicate").to_numpy(float)
        ))), threshold="0 nonfinite bootstrap values", detail=f"B={BOOT_N}",
    ))
    if not finite_boot:
        raise SystemExit("M2 bootstrap has nonfinite values; verdict suppressed")

    estimate_rows: list[dict[str, Any]] = []
    for prefix in ("A", "D"):
        for metric in METRICS:
            key = f"{prefix}_{metric}"
            lo, hi = np.quantile(boot[key].to_numpy(float), (0.025, 0.975))
            estimate_rows.append(dict(
                statistic=prefix, metric=metric,
                estimate=float(point[key][0]), ci_lo=float(lo), ci_hi=float(hi),
                weight=float(point["weight"][0]), n_boot=BOOT_N,
                bootstrap_seed=BOOT_SEED, n_nonfinite=0,
            ))
    estimates = pd.DataFrame(estimate_rows)
    delta = estimates[(estimates.statistic == "A") & (estimates.metric == "delta_S")].iloc[0]
    base = estimates[(estimates.statistic == "A") & (estimates.metric == "S0")].iloc[0]
    dynamics = classify_dynamics(float(delta.ci_lo), float(delta.ci_hi))
    baseline = classify_baseline(float(base.ci_lo), float(base.ci_hi))
    if dynamics == "EQUIV_DYNAMICS" and baseline == "HIGH_STARTS_MORE_OPEN":
        combined = "START_FARTHER_ONLY_SUPPORTED"
    elif dynamics == "HIGH_LESS_PUSHED":
        combined = "DYNAMIC_DIFFERENCE_HIGH_LESS_PUSHED"
    elif dynamics == "HIGH_MORE_PUSHED":
        combined = "DYNAMIC_DIFFERENCE_OPPOSITE"
    else:
        combined = "COMBINED_INCONCLUSIVE"
    verdict = pd.DataFrame([
        dict(analysis="M2_dynamics", estimate=float(delta.estimate),
             ci_lo=float(delta.ci_lo), ci_hi=float(delta.ci_hi),
             equivalence_margin=EQUIV_MARGIN, verdict=dynamics),
        dict(analysis="M2_baseline", estimate=float(base.estimate),
             ci_lo=float(base.ci_lo), ci_hi=float(base.ci_hi),
             equivalence_margin=EQUIV_MARGIN, verdict=baseline),
        dict(analysis="M2_combined", estimate=np.nan, ci_lo=np.nan, ci_hi=np.nan,
             equivalence_margin=EQUIV_MARGIN, verdict=combined),
    ])
    outputs = {
        "joined_exposures.csv": joined,
        "group_distributions.csv": distributions,
        "per_seed_distributions.csv": per_seed,
        "cell_effects.csv": cell_effects,
        "estimates.csv": estimates,
        "bootstrap.csv": boot,
        "sanity.csv": pd.DataFrame(sanity_rows),
        "verdict.csv": verdict,
    }
    details = dict(
        dynamics=dynamics, baseline=baseline, combined=combined,
        point={key: float(value[0]) for key, value in point.items()},
        n_exposure=len(joined), n_cell=len(cells), n_low=n_low, n_high=n_high,
    )
    return outputs, details


def write_summary(outdir: Path, outputs: dict[str, pd.DataFrame],
                  details: dict[str, Any], replay_meta: dict[str, Any]) -> None:
    estimates = outputs["estimates.csv"]
    verdict = outputs["verdict.csv"].set_index("analysis")

    def est(prefix: str, metric: str) -> pd.Series:
        return estimates[
            (estimates.statistic == prefix) & (estimates.metric == metric)
        ].iloc[0]

    a_delta, a_s0 = est("A", "delta_S"), est("A", "S0")
    d_delta, d_s0 = est("D", "delta_S"), est("D", "S0")
    lines = [
        "# M2: Delta-L 群と開口量の動的変化", "",
        f"- **M2_dynamics: {verdict.loc['M2_dynamics', 'verdict']}**",
        f"- **M2_baseline: {verdict.loc['M2_baseline', 'verdict']}**",
        f"- **M2_combined: {verdict.loc['M2_combined', 'verdict']}**", "",
        "## 主結果", "",
        f"- A_deltaS = {a_delta.estimate:+.6f} "
        f"[{a_delta.ci_lo:+.6f}, {a_delta.ci_hi:+.6f}]",
        f"- A_S0 = {a_s0.estimate:+.6f} "
        f"[{a_s0.ci_lo:+.6f}, {a_s0.ci_hi:+.6f}]",
        f"- raw D_deltaS = {d_delta.estimate:+.9g} "
        f"[{d_delta.ci_lo:+.9g}, {d_delta.ci_hi:+.9g}]",
        f"- raw D_S0 = {d_s0.estimate:+.9g} "
        f"[{d_s0.ci_lo:+.9g}, {d_s0.ci_hi:+.9g}]", "",
        "A は同じ作業6幾何セル内の優越確率から0.5を引いた量。"
        "等価域は +/-0.05。", "",
        "## 固定集合と再生", "",
        f"- risk exposure: {details['n_exposure']:,}",
        f"- valid cells: {details['n_cell']:,}",
        f"- low/high: {details['n_low']:,} / {details['n_high']:,}",
        f"- replay implementation: `{replay_meta.get('implementation_git', 'unknown')}`",
        "- 元走の最終 complete-state hash と全既存 landmark 列: **完全一致**",
        "- M2-S1..S9: **PASS**", "",
        "## 解釈上限", "",
        "- 同じ軌道の読み取り専用再計装であり、独立 replication ではない。",
        "- 固定幾何セル内の観察的関連であり、Delta-L の因果効果ではない。",
        "- 作業6 PROTECTIVE と r-swap SPECIFIC の判定を差し替えない。",
    ]
    (outdir / "summary.md").write_text("\n".join(lines) + "\n")


def run(replay_dir: Path, reference_dir: Path, outdir: Path) -> None:
    if os.environ.get("OMP_NUM_THREADS") != "1":
        raise SystemExit("OMP_NUM_THREADS=1 is required")
    replay_meta, original_meta = validate_inputs(replay_dir, reference_dir)
    exposures = pd.read_csv(reference_dir / "exposures.csv")
    logs = load_s_logs(replay_dir)
    joined = join_opening(exposures, logs)

    first, details = build_outputs(joined, original_meta, replay_meta)
    second, _ = build_outputs(joined, original_meta, replay_meta)
    first_bytes = {name: _csv_bytes(first[name]) for name in CSV_NAMES}
    second_bytes = {name: _csv_bytes(second[name]) for name in CSV_NAMES}
    mismatches = [name for name in CSV_NAMES if first_bytes[name] != second_bytes[name]]
    if mismatches:
        raise SystemExit("M2-S9 determinism FAIL: " + ", ".join(mismatches))

    sanity = first["sanity.csv"].copy()
    sanity = pd.concat([sanity, pd.DataFrame([dict(
        id="M2-S9", status="PASS", value=len(CSV_NAMES),
        threshold="all CSV byte-identical across two full builds", detail="",
    )])], ignore_index=True)
    first["sanity.csv"] = sanity
    # Rebuild only the sanity payload after adding the deterministic gate to both copies.
    first_bytes["sanity.csv"] = _csv_bytes(sanity)
    second_bytes["sanity.csv"] = _csv_bytes(sanity.copy())

    outdir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="function_blind_m2_") as temp:
        tempdir = Path(temp)
        for name, payload in first_bytes.items():
            (tempdir / name).write_bytes(payload)
            if (tempdir / name).read_bytes() != second_bytes[name]:
                raise SystemExit(f"M2-S9 serialization FAIL: {name}")
    for name, payload in first_bytes.items():
        (outdir / name).write_bytes(payload)

    hashes = {name: hashlib.sha256(payload).hexdigest()
              for name, payload in first_bytes.items()}
    det_lines = [
        "# determinism check", "", "- result: **PASS**",
        "- method: 同じ replay S / exposure / RNG から解析全体を2回再構成",
        f"- bootstrap: B={BOOT_N}, seed={BOOT_SEED}", "",
        "| CSV | SHA-256 |", "|---|---|",
    ] + [f"| {name} | `{hashes[name]}` |" for name in CSV_NAMES]
    (outdir / "determinism_check.md").write_text("\n".join(det_lines) + "\n")

    write_summary(outdir, first, details, replay_meta)
    meta = dict(
        spec=SPEC, preregistration_commit="1096be3", analysis_git=git_hash(),
        replay_git=replay_meta.get("implementation_git"),
        seeds=list(SEEDS), bootstrap_n=BOOT_N, bootstrap_seed=BOOT_SEED,
        equivalence_margin=EQUIV_MARGIN,
        reference_sha256=dict(
            exposures=sha256(reference_dir / "exposures.csv"),
            instrumentation_meta=sha256(reference_dir / "instrumentation_meta.json"),
        ),
        replay_meta_sha256=sha256(replay_dir / "replay_meta.json"),
        replay_npz_sha256={
            path.name: sha256(path)
            for path in sorted((replay_dir / "logs").glob("seed*.npz"))
        },
        csv_sha256=hashes, diagnostics=details,
        # pandas' JSON encoder maps the intentionally blank combined-row
        # estimates to null; the strict stdlib encoder rightly rejects NaN.
        verdict=json.loads(first["verdict.csv"].to_json(orient="records")),
        restrictions=[
            "same-trajectory re-instrumentation, not independent replication",
            "observational within frozen geometry cells, not causal",
            "do not replace work-6 or r-swap verdicts",
        ],
    )
    (outdir / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    print((outdir / "summary.md").read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--replay", type=Path,
        default=ROOT / "results/function_blind_m2_0828",
    )
    parser.add_argument(
        "--reference", type=Path,
        default=ROOT / "results/function_blind_direct_0823_confirm",
    )
    parser.add_argument(
        "--outdir", type=Path,
        default=ROOT / "results/function_blind_m2_0828",
    )
    args = parser.parse_args()
    run(args.replay, args.reference, args.outdir)


if __name__ == "__main__":
    main()
