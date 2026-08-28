"""Q17: camp symmetry of the gate-included ``F_rest`` drive.

The frozen preregistration is obsidian-research commit ``de2fff2`` at
``可塑性喪失/spec/Q17駆動対称性_spec_0828.md``.  Revision 3 fixes the
magnitude equivalence margin at ``epsilon_M=0.05`` before any Q17 statistic is
computed.  This is a drive-symmetry margin, not a wall-reachability claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SPEC_VAULT_COMMIT = "de2fff2"
PERIOD = 10_000
WIDTH = 100
SEEDS = tuple(range(10))
BOOT_N = 10_000
BOOT_SEED = 20260828
PROB_MARGIN = 0.05
EPSILON_M = 0.05
MIN_VALID_SEED = 8
MIN_ROWS = 300
F32_RTOL = 8 * np.finfo(np.float32).eps
F32_ATOL = 8 * np.finfo(np.float32).tiny
REGIONS = {
    "FULL": (-100, 100),
    "PRE": (-100, -1),
    "POST": (1, 100),
    "EDGE": (0, 0),
}
PRIMARY_REGIONS = ("FULL", "PRE", "POST")
EXPECTED_GAPS = {1: 19_900, 900: 199, 1000: 801}
EXPECTED_RECORDS = {"FULL": 19_899, "PRE": 9_900, "POST": 9_900, "EDGE": 99}
REQUIRED = {
    "step", "seed", "period", "width", "cos_u_mu", "p_hat", "w_norm",
    "b", "v", "F_self", "F_rest", "F_gate", "G", "flip_state",
    "E_delta", "mu_norm", "eval_loss_exact",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_one(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.array(archive[key]) for key in archive.files}


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")


def git_info(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(args, cwd=root, text=True).strip()

    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "branch": run("git", "branch", "--show-current"),
        "tracked_clean": not bool(run(
            "git", "status", "--porcelain", "--untracked-files=no")),
        "status_porcelain": run("git", "status", "--porcelain").splitlines(),
    }


def mixed_close(lhs: np.ndarray, rhs: np.ndarray,
                scale: np.ndarray | None = None) -> np.ndarray:
    if scale is None:
        scale = np.abs(lhs) + np.abs(rhs)
    return np.abs(lhs - rhs) <= F32_ATOL + F32_RTOL * scale


def scalar_close(lhs: float, rhs: float) -> bool:
    return bool(abs(lhs - rhs) <= F32_ATOL + F32_RTOL * (abs(lhs) + abs(rhs)))


def find_boundaries(step: np.ndarray, flip_state: np.ndarray,
                    period: int = PERIOD) -> np.ndarray:
    """Return left indices B for transitions B -> B+1."""
    changed = (np.diff(flip_state, axis=0) != 0).any(axis=1)
    actual = np.flatnonzero(changed)
    gaps = np.diff(step)
    expected = np.flatnonzero(
        (step[:-1] % period == 0) & (gaps == 1) & (step[:-1] > 0))
    if not np.array_equal(actual, expected):
        raise ValueError("flip_state transitions do not match registered boundaries")
    return actual


def region_indices(step: np.ndarray, boundary_steps: np.ndarray,
                   lo: int, hi: int) -> np.ndarray:
    """Map registered relative steps to record indices, failing on a gap."""
    lookup = {int(value): index for index, value in enumerate(step)}
    wanted = np.concatenate([
        np.arange(int(boundary) + lo, int(boundary) + hi + 1, dtype=np.int64)
        for boundary in boundary_steps
    ])
    try:
        return np.fromiter((lookup[int(value)] for value in wanted),
                           dtype=np.int64, count=len(wanted))
    except KeyError as exc:
        raise ValueError(f"registered boundary-window step is missing: {exc}") from exc


def camp_metrics(v: np.ndarray, f_rest: np.ndarray, p_hat: np.ndarray,
                 camp: int) -> dict[str, float | int]:
    """Sufficient statistics for one seed x region x camp.

    ``N`` retains physical x direction.  ``M=-camp*N`` removes the exact
    camp sign reversal in F_rest = -2 eta v h.
    """
    if camp not in (-1, 1):
        raise ValueError("camp must be -1 or +1")
    mask = v > 0 if camp == 1 else v < 0
    f = np.asarray(f_rest[mask], dtype=np.float64)
    ph = np.asarray(p_hat[mask], dtype=np.float64)
    nonzero = f != 0
    n = int(f.size)
    n_nonzero = int(nonzero.sum())
    n_zero = n - n_nonzero
    n_pos = int((f > 0).sum())
    n_neg = int((f < 0).sum())
    sum_f = float(f.sum(dtype=np.float64))
    sum_abs = float(np.abs(f).sum(dtype=np.float64))
    q = float((f[nonzero] < 0).mean()) if n_nonzero else np.nan
    h_pos = (-camp * np.sign(f[nonzero])) > 0
    p = float(h_pos.mean()) if n_nonzero else np.nan
    n_ratio = sum_f / sum_abs if sum_abs > 0 else np.nan
    m_ratio = -camp * n_ratio if np.isfinite(n_ratio) else np.nan
    zero_phat = float((ph[f == 0] == 0).mean()) if n_zero else np.nan
    return {
        "camp": camp,
        "n_rows": n,
        "n_nonzero": n_nonzero,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "n_zero": n_zero,
        "p": p,
        "q": q,
        "z": n_zero / n if n else np.nan,
        "zero_phat0_fraction": zero_phat,
        "sum_f_rest": sum_f,
        "sum_abs_f_rest": sum_abs,
        "mean_f_rest": sum_f / n if n else np.nan,
        "N": n_ratio,
        "M": m_ratio,
    }


def paired_metrics(plus: dict[str, Any], minus: dict[str, Any]) -> dict[str, float]:
    """Registered camp contrasts after undoing the exact camp sign flip."""
    p_plus, p_minus = float(plus["p"]), float(minus["p"])
    n_plus, n_minus = float(plus["N"]), float(minus["N"])
    m_plus, m_minus = float(plus["M"]), float(minus["M"])
    return {
        "B": p_plus - p_minus,
        "C": p_plus + p_minus - 1.0,
        "B_M": m_plus - m_minus,
        "A_M": (m_plus + m_minus) / 2.0,
        "delta_N": n_plus - n_minus,
        "delta_z": float(plus["z"]) - float(minus["z"]),
    }


def seed_region_metrics(d: dict[str, np.ndarray]) -> tuple[list[dict[str, Any]],
                                                            list[dict[str, Any]],
                                                            list[dict[str, Any]]]:
    seed = int(d["seed"])
    step = d["step"]
    boundaries = find_boundaries(step, d["flip_state"])
    boundary_steps = step[boundaries]
    long_rows: list[dict[str, Any]] = []
    wide_rows: list[dict[str, Any]] = []
    identity_rows: list[dict[str, Any]] = []
    for region, (lo, hi) in REGIONS.items():
        indices = region_indices(step, boundary_steps, lo, hi)
        v = d["v"][indices].astype(np.float64, copy=False).reshape(-1)
        f_rest = d["F_rest"][indices].astype(np.float64, copy=False).reshape(-1)
        p_hat = d["p_hat"][indices].astype(np.float64, copy=False).reshape(-1)
        plus = camp_metrics(v, f_rest, p_hat, 1)
        minus = camp_metrics(v, f_rest, p_hat, -1)
        pair = paired_metrics(plus, minus)
        v_zero_rate = float((v == 0).mean())
        for values in (plus, minus):
            long_rows.append({"seed": seed, "region": region,
                              **values, **pair, "v_zero_rate": v_zero_rate,
                              "camp_fraction": values["n_rows"] / len(v)})
        wide = {"seed": seed, "region": region, "v_zero_rate": v_zero_rate}
        for label, values in (("plus", plus), ("minus", minus)):
            for key, value in values.items():
                if key != "camp":
                    wide[f"{key}_{label}"] = value
        wide.update(pair)
        wide["valid_W"] = bool(
            plus["n_rows"] >= MIN_ROWS and minus["n_rows"] >= MIN_ROWS
            and plus["n_nonzero"] >= MIN_ROWS and minus["n_nonzero"] >= MIN_ROWS)
        wide["valid_N"] = bool(
            plus["n_rows"] >= MIN_ROWS and minus["n_rows"] >= MIN_ROWS
            and plus["sum_abs_f_rest"] > 0 and minus["sum_abs_f_rest"] > 0)
        wide_rows.append(wide)

        identities: list[bool] = []
        if np.isfinite(plus["p"]):
            identities.append(scalar_close(plus["p"], plus["q"]))
        if np.isfinite(minus["p"]):
            identities.append(scalar_close(minus["p"], 1.0 - minus["q"]))
        if np.isfinite(plus["p"]) and np.isfinite(minus["p"]):
            identities.extend([
                scalar_close(pair["B"], plus["q"] + minus["q"] - 1.0),
                scalar_close(pair["C"], plus["q"] - minus["q"]),
            ])
        if np.isfinite(plus["N"]):
            identities.append(scalar_close(plus["M"], -plus["N"]))
        if np.isfinite(minus["N"]):
            identities.append(scalar_close(minus["M"], minus["N"]))
        if np.isfinite(plus["N"]) and np.isfinite(minus["N"]):
            identities.extend([
                scalar_close(pair["B_M"], -(plus["N"] + minus["N"])),
                scalar_close(pair["A_M"], -pair["delta_N"] / 2.0),
            ])
        identity_rows.append({
            "id": "S6_algebra", "seed": seed, "region": region,
            "pass": bool(all(identities)),
            "note": "p/q and M/N camp-sign identities",
        })
    return long_rows, wide_rows, identity_rows


def source_sanity(d: dict[str, np.ndarray], path: Path) -> list[dict[str, Any]]:
    seed = int(d.get("seed", -1))
    rows: list[dict[str, Any]] = []

    def add(identifier: str, passed: bool, note: str) -> None:
        rows.append({"id": identifier, "seed": seed, "region": "ALL",
                     "pass": bool(passed), "note": note})

    keys_ok = REQUIRED <= set(d)
    add("S1_required_columns", keys_ok,
        f"missing={sorted(REQUIRED-set(d))}")
    if not keys_ok:
        return rows
    step = d["step"]
    unit_shape = (20_901, 100)
    shape_ok = (
        step.shape == (20_901,)
        and d["flip_state"].shape == (20_901, 15)
        and all(d[key].shape == unit_shape for key in
                ("p_hat", "v", "F_self", "F_rest", "F_gate")))
    add("S1_shapes", shape_ok,
        f"step={step.shape}, flip={d['flip_state'].shape}, Frest={d['F_rest'].shape}")
    dtype_ok = (step.dtype == np.int64 and d["flip_state"].dtype == np.float32
                and all(d[key].dtype == np.float32 for key in
                        ("p_hat", "v", "F_self", "F_rest", "F_gate")))
    add("S1_dtypes", dtype_ok,
        f"step={step.dtype}, flip={d['flip_state'].dtype}, Frest={d['F_rest'].dtype}")
    finite_ok = all(np.isfinite(d[key]).all() for key in REQUIRED
                    if np.issubdtype(d[key].dtype, np.number))
    add("S1_finite", finite_ok, f"source={path.name}")
    gaps = {int(value): int((np.diff(step) == value).sum())
            for value in np.unique(np.diff(step))}
    add("S1_grid", gaps == EXPECTED_GAPS, f"gaps={gaps}")
    add("S1_seed_period_width",
        seed in SEEDS and int(d["period"]) == PERIOD and int(d["width"]) == WIDTH,
        f"seed={seed}, period={int(d['period'])}, width={int(d['width'])}")

    try:
        boundaries = find_boundaries(step, d["flip_state"])
        boundary_ok = len(boundaries) == 99
        add("S2_boundaries", boundary_ok, f"n={len(boundaries)}")
        boundary_steps = step[boundaries]
        region_sets: dict[str, set[int]] = {}
        for region, (lo, hi) in REGIONS.items():
            idx = region_indices(step, boundary_steps, lo, hi)
            region_sets[region] = set(map(int, idx))
            add(f"S2_rows_{region}", len(idx) == EXPECTED_RECORDS[region],
                f"records={len(idx)}, logical_rows={len(idx)*WIDTH}")
        partition_ok = (
            not (region_sets["PRE"] & region_sets["POST"])
            and not (region_sets["PRE"] & region_sets["EDGE"])
            and not (region_sets["POST"] & region_sets["EDGE"])
            and region_sets["PRE"] | region_sets["POST"] | region_sets["EDGE"]
            == region_sets["FULL"])
        add("S2_partition", partition_ok, "PRE, POST, EDGE partition FULL")
    except ValueError as exc:
        add("S2_boundaries", False, str(exc))

    fg = d["F_gate"].astype(np.float64)
    fs = d["F_self"].astype(np.float64)
    fr = d["F_rest"].astype(np.float64)
    force_ok = mixed_close(fg, fs + fr,
                           np.abs(fg) + np.abs(fs) + np.abs(fr))
    add("S4_force_decomposition", bool(force_ok.all()),
        f"bad={int((~force_ok).sum())}, max_abs={np.max(np.abs(fg-fs-fr)):.17g}")
    v_zero = d["v"] == 0
    p_zero = d["p_hat"] == 0
    add("S5_vzero_frest", bool((fr[v_zero] == 0).all()),
        f"vzero={int(v_zero.sum())}")
    three_zero = ((fg[p_zero] == 0) & (fs[p_zero] == 0) & (fr[p_zero] == 0))
    add("S5_phat0_forces", bool(three_zero.all()),
        f"phat0={int(p_zero.sum())}, bad={int((~three_zero).sum())}")
    return rows


def bootstrap_indices(n_seed: int, n_boot: int = BOOT_N,
                      seed: int = BOOT_SEED) -> np.ndarray:
    if n_seed <= 0 or n_boot <= 0:
        raise ValueError("bootstrap dimensions must be positive")
    return np.random.default_rng(seed).integers(
        0, n_seed, size=(n_boot, n_seed), endpoint=False)


def bootstrap_metric(values: np.ndarray, draws: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    chosen = values[draws]
    finite = np.isfinite(chosen)
    counts = finite.sum(axis=1)
    boot = np.divide(np.where(finite, chosen, 0.0).sum(axis=1), counts,
                     out=np.full(len(draws), np.nan), where=counts > 0)
    valid = np.isfinite(values)
    point = float(values[valid].mean()) if valid.any() else np.nan
    n_bad = int((~np.isfinite(boot)).sum())
    if n_bad:
        lo = hi = np.nan
    else:
        lo, hi = map(float, np.percentile(boot, [2.5, 97.5]))
    return {"point": point, "ci_lo": lo, "ci_hi": hi,
            "half_width": (hi - lo) / 2.0 if np.isfinite(lo + hi) else np.nan,
            "n_valid_seed": int(valid.sum()), "n_boot_nonfinite": n_bad}


def component_status(stat: dict[str, Any], center: float,
                     margin: float) -> str:
    if stat["n_valid_seed"] < MIN_VALID_SEED:
        return "INCONCLUSIVE_GUARD"
    if stat["n_boot_nonfinite"]:
        return "SANITY_FAIL_NONFINITE"
    if stat["half_width"] > margin:
        return "INCONCLUSIVE_PRECISION"
    lower, upper = center - margin, center + margin
    lo, hi = stat["ci_lo"], stat["ci_hi"]
    if hi < lower or lo > upper:
        return "MATERIAL"
    if lo >= lower and hi <= upper:
        return "EQUIV"
    return "INCONCLUSIVE"


def verdict_table(wide: pd.DataFrame, epsilon_m: float,
                  n_boot: int = BOOT_N) -> pd.DataFrame:
    if not 0 < epsilon_m < 1:
        raise ValueError("epsilon_m must be in (0, 1)")
    seed_ids = np.array(SEEDS)
    draws = bootstrap_indices(len(seed_ids), n_boot=n_boot)
    definitions = {
        "Q17-W": (("p_plus", 0.5, PROB_MARGIN, "valid_W"),
                  ("p_minus", 0.5, PROB_MARGIN, "valid_W")),
        "Q17-B": (("B", 0.0, PROB_MARGIN, "valid_W"),),
        "Q17-N": (("M_plus", 0.0, epsilon_m, "valid_N"),
                  ("M_minus", 0.0, epsilon_m, "valid_N"),
                  ("B_M", 0.0, epsilon_m, "valid_N")),
    }
    rows: list[dict[str, Any]] = []
    family_statuses: list[str] = []
    for region in PRIMARY_REGIONS:
        sub = wide[wide.region == region].set_index("seed")
        for family, components in definitions.items():
            statuses = []
            for metric, center, margin, valid_column in components:
                values = np.full(len(seed_ids), np.nan)
                for index, seed in enumerate(seed_ids):
                    if seed in sub.index and bool(sub.loc[seed, valid_column]):
                        values[index] = float(sub.loc[seed, metric])
                stat = bootstrap_metric(values, draws)
                status = component_status(stat, center, margin)
                statuses.append(status)
                rows.append({"region": region, "family": family, "metric": metric,
                             "center": center, "margin": margin, **stat,
                             "status": status})
            if any(value == "INCONCLUSIVE_GUARD" for value in statuses):
                family_status = "INCONCLUSIVE_GUARD"
            elif any(value == "SANITY_FAIL_NONFINITE" for value in statuses):
                family_status = "SANITY_FAIL_NONFINITE"
            elif any(value == "INCONCLUSIVE_PRECISION" for value in statuses):
                family_status = "INCONCLUSIVE_PRECISION"
            elif any(value == "MATERIAL" for value in statuses):
                family_status = "MATERIAL"
            elif all(value == "EQUIV" for value in statuses):
                family_status = "EQUIV"
            else:
                family_status = "INCONCLUSIVE"
            family_statuses.append(family_status)
            rows.append({"region": region, "family": family, "metric": "FAMILY",
                         "center": np.nan, "margin": components[0][2],
                         "point": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                         "half_width": np.nan, "n_valid_seed": np.nan,
                         "n_boot_nonfinite": np.nan, "status": family_status})

    if any(value == "INCONCLUSIVE_GUARD" for value in family_statuses):
        overall = "INCONCLUSIVE_GUARD"
    elif any(value == "SANITY_FAIL_NONFINITE" for value in family_statuses):
        overall = "SANITY_FAIL_NONFINITE"
    elif any(value == "INCONCLUSIVE_PRECISION" for value in family_statuses):
        overall = "INCONCLUSIVE_PRECISION"
    elif any(value == "MATERIAL" for value in family_statuses):
        overall = "MATERIAL_REST_ASYMMETRY"
    elif all(value == "EQUIV" for value in family_statuses):
        overall = "SYMMETRIC_REST_DRIVE"
    else:
        overall = "INCONCLUSIVE_REST_SYMMETRY"
    rows.append({"region": "ALL", "family": "Q17", "metric": "OVERALL",
                 "center": np.nan, "margin": np.nan, "point": np.nan,
                 "ci_lo": np.nan, "ci_hi": np.nan, "half_width": np.nan,
                 "n_valid_seed": np.nan, "n_boot_nonfinite": np.nan,
                 "status": overall})
    return pd.DataFrame(rows)


def make_figures(verdict: pd.DataFrame, wide: pd.DataFrame, out: Path) -> None:
    import matplotlib.pyplot as plt

    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    components = verdict[verdict.metric != "FAMILY"]

    def ci_plot(metrics: tuple[str, ...], name: str, ylabel: str) -> None:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        selected = components[components.metric.isin(metrics)]
        for offset, metric in enumerate(metrics):
            sub = selected[selected.metric == metric]
            x = np.arange(len(sub)) + (offset - (len(metrics)-1)/2) * 0.16
            y = sub.point.to_numpy(float)
            err = np.vstack([y-sub.ci_lo.to_numpy(float), sub.ci_hi.to_numpy(float)-y])
            ax.errorbar(x, y, yerr=err, fmt="o", capsize=3, label=metric)
        ax.axhline(0.5 if metrics[0].startswith("p_") else 0.0,
                   color="black", linewidth=0.8)
        ax.set_xticks(np.arange(len(PRIMARY_REGIONS)), PRIMARY_REGIONS)
        ax.set_ylabel(ylabel)
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures / name, dpi=160)
        plt.close(fig)

    ci_plot(("p_plus", "p_minus"), "fig_sign_composition.png", "P(h > 0)")
    ci_plot(("M_plus", "M_minus", "B_M"), "fig_net_force_ratio.png", "M ratio")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for region, marker in zip(PRIMARY_REGIONS, ("o", "s", "^")):
        sub = wide[wide.region == region]
        axes[0].scatter(sub.seed, sub.B, label=region, marker=marker)
        axes[1].scatter(sub.seed, sub.B_M, label=region, marker=marker)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[0].set_ylabel("B = p+ - p-")
    axes[1].set_ylabel("B_M = M+ - M-")
    for ax in axes:
        ax.set_xlabel("seed")
        ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "fig_seed_metrics.png", dpi=160)
    plt.close(fig)


def make_summary(verdict: pd.DataFrame, sanity: pd.DataFrame,
                 epsilon_m: float) -> str:
    overall = verdict.loc[verdict.metric == "OVERALL", "status"].iloc[0]
    family = verdict[verdict.metric == "FAMILY"][
        ["region", "family", "status"]]
    family_table = ["| region | family | status |", "|---|---|---|"]
    family_table.extend(
        f"| {row.region} | {row.family} | {row.status} |"
        for row in family.itertuples(index=False))
    lines = [
        "# Q17 rest-drive symmetry",
        "",
        f"- spec vault commit: `{SPEC_VAULT_COMMIT}`",
        f"- epsilon_M: `{epsilon_m:.17g}` (CLI freeze confirmation required)",
        f"- sanity: **{'PASS' if bool(sanity['pass'].all()) else 'FAIL'}**",
        f"- overall: **{overall}**",
        "",
        "## Family verdicts",
        "",
        *family_table,
        "",
        "## Interpretation ceiling",
        "",
        "Q17 tests F_rest only. It does not establish that the trap is the sole cause,",
        "nor does a coherence margin prove inability to reach the wall over 10^6 steps.",
        "F_self is assigned to the constraint/damping ledger.",
        "",
    ]
    return "\n".join(lines)


def run(input_dir: Path, out: Path, n_boot: int = BOOT_N,
        sanity_only: bool = False) -> None:
    root = Path(__file__).resolve().parents[2]
    paths = sorted((input_dir / "logs").glob("seed*.npz"),
                   key=lambda path: int(path.stem.removeprefix("seed")))
    sanity_rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    out.mkdir(parents=True, exist_ok=True)
    sanity_rows.append({
        "id": "S1_seed_files", "seed": -1, "region": "ALL",
        "pass": len(paths) == len(SEEDS)
                and [int(path.stem.removeprefix("seed")) for path in paths] == list(SEEDS),
        "note": f"n={len(paths)}",
    })
    sanity_rows.append({
        "id": "S7_scope", "seed": -1, "region": "ALL", "pass": True,
        "note": "primary estimands use only v, F_rest, p_hat, step, flip_state",
    })

    # Pass 1 is structural only. No Q17 estimand is formed until every source
    # passes the registered schema, grid, decomposition and identity gates.
    for path in paths:
        data = load_one(path)
        sanity_rows.extend(source_sanity(data, path))
        sources.append({
            "path": str(path.relative_to(root)), "sha256": sha256(path),
            "keys": sorted(data),
            "shapes": {key: list(data[key].shape) for key in sorted(data)},
            "dtypes": {key: str(data[key].dtype) for key in sorted(data)},
        })
    config = {
        "analysis": "q17_symmetry_0828", "spec_vault_commit": SPEC_VAULT_COMMIT,
        "regions": REGIONS, "probability_margin": PROB_MARGIN,
        "magnitude_criterion": "h-space coherence ratio M=-camp*N",
        "epsilon_M": EPSILON_M, "bootstrap_B": n_boot,
        "bootstrap_seed": BOOT_SEED, "min_valid_seed": MIN_VALID_SEED,
        "min_rows": MIN_ROWS, "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
    }
    write_json(out / "config.json", config)
    write_json(out / "provenance.json", {
        "git": git_info(root), "spec_vault_commit": SPEC_VAULT_COMMIT,
        "python": sys.version, "platform": platform.platform(),
        "numpy": np.__version__, "pandas": pd.__version__, "sources": sources,
    })
    sanity = pd.DataFrame(sanity_rows)
    sanity.to_csv(out / "sanity.csv", index=False)
    if not bool(sanity["pass"].all()):
        pd.DataFrame([{"region": "ALL", "family": "Q17", "metric": "OVERALL",
                       "status": "INCONCLUSIVE_GUARD"}]).to_csv(
                           out / "verdict.csv", index=False)
        raise SystemExit("Q17 structural sanity failed; verdict suppressed")
    if sanity_only:
        print(f"Q17 structural sanity PASS -> {out}")
        return

    # Pass 2 forms the registered estimands only after the structural gate.
    long_rows: list[dict[str, Any]] = []
    wide_rows: list[dict[str, Any]] = []
    identity_rows: list[dict[str, Any]] = []
    for path in paths:
        data = load_one(path)
        long_part, wide_part, identities = seed_region_metrics(data)
        long_rows.extend(long_part)
        wide_rows.extend(wide_part)
        identity_rows.extend(identities)
    sanity = pd.concat([sanity, pd.DataFrame(identity_rows)], ignore_index=True)
    sanity.to_csv(out / "sanity.csv", index=False)
    if not bool(sanity["pass"].all()):
        pd.DataFrame([{"region": "ALL", "family": "Q17", "metric": "OVERALL",
                       "status": "INCONCLUSIVE_GUARD"}]).to_csv(
                           out / "verdict.csv", index=False)
        raise SystemExit("Q17 algebra sanity failed; verdict suppressed")

    long = pd.DataFrame(long_rows).sort_values(["seed", "region", "camp"])
    wide = pd.DataFrame(wide_rows).sort_values(["seed", "region"])
    verdict = verdict_table(wide, epsilon_m=EPSILON_M, n_boot=n_boot)
    long.to_csv(out / "per_seed_metrics.csv", index=False)
    wide.to_csv(out / "per_seed_wide.csv", index=False)
    verdict.to_csv(out / "verdict.csv", index=False)
    (out / "summary.md").write_text(
        make_summary(verdict, sanity, EPSILON_M), encoding="utf-8")
    make_figures(verdict, wide, out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", default="results/ratchet_log_0819")
    parser.add_argument("--outdir", default="results/q17_symmetry_0828")
    parser.add_argument("--sanity-only", action="store_true")
    args = parser.parse_args()
    if os.environ.get("OMP_NUM_THREADS") != "1":
        raise SystemExit("OMP_NUM_THREADS must be exactly 1")
    root = Path(__file__).resolve().parents[2]
    input_dir = Path(args.input)
    out = Path(args.outdir)
    if not input_dir.is_absolute():
        input_dir = (root / input_dir).resolve()
    if not out.is_absolute():
        out = (root / out).resolve()
    run(input_dir, out, n_boot=BOOT_N, sanity_only=args.sanity_only)


if __name__ == "__main__":
    main()
