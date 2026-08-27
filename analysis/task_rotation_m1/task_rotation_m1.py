"""M1: task-direction rotation versus boundary reclassification death.

The frozen specification lives in obsidian-research commit 5b9805a at
``可塑性喪失/spec/タスク方向回転M1_spec_0828.md``.  This module consumes only
the committed ratchet_log_0819 npz files; it does not retrain the model.
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

import numpy as np
import pandas as pd


SPEC_VAULT_COMMIT = "5b9805a"
BOOT_N = 10_000
BOOT_SEED = 20260829
TAU = 0.05
MIN_RISK = 10
MIN_BOUNDARY = 50
MIN_C_LEVEL = 3
MIN_VALID_SEED = 8


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_info(root: Path) -> dict:
    def run(*args: str) -> str:
        return subprocess.check_output(args, cwd=root, text=True).strip()

    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "branch": run("git", "branch", "--show-current"),
        "tracked_clean": not bool(run("git", "status", "--porcelain", "--untracked-files=no")),
        "status_porcelain": run("git", "status", "--porcelain").splitlines(),
    }


def rank_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman correlation with average ties, without a scipy dependency."""
    rx = pd.Series(x).rank(method="average").to_numpy(dtype=np.float64)
    ry = pd.Series(y).rank(method="average").to_numpy(dtype=np.float64)
    if np.ptp(rx) == 0 or np.ptp(ry) == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def load_one(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: np.array(z[k]) for k in z.files}


def boundary_rows(d: dict[str, np.ndarray]) -> tuple[list[dict], list[tuple]]:
    step = d["step"]
    fs = d["flip_state"].astype(np.float64)
    p = d["p_hat"].astype(np.float64)
    seed = int(d["seed"])
    changed = np.any(np.diff(fs, axis=0) != 0, axis=1)
    js = np.flatnonzero(changed)
    rows: list[dict] = []
    events: list[tuple] = []
    for j in js:
        old, new = fs[j], fs[j + 1]
        diff = np.flatnonzero(old != new)
        n_old, n_new = int(old.sum()), int(new.sum())
        overlap = int(np.logical_and(old == 1, new == 1).sum())
        denom = np.sqrt((n_old + 1.25) * (n_new + 1.25))
        c_formula = float((overlap + 1.25) / denom)
        mu_old = np.concatenate([old, np.full(5, 0.5)])
        mu_new = np.concatenate([new, np.full(5, 0.5)])
        c_direct = float(np.dot(mu_old, mu_new) /
                         (np.linalg.norm(mu_old) * np.linalg.norm(mu_new)))

        risk005 = p[j] >= TAU
        death005 = risk005 & (p[j + 1] < TAU)
        risk_strict = p[j] > 0
        death_strict = risk_strict & (p[j + 1] == 0)
        for unit in np.flatnonzero(death005):
            events.append((seed, int(unit), int(step[j]), int(step[j + 1])))
        rows.append({
            "seed": seed,
            "boundary_step": int(step[j]),
            "step_post": int(step[j + 1]),
            "pair_index": int(j),
            "n_old": n_old,
            "n_new": n_new,
            "overlap": overlap,
            "flipped_bit": int(diff[0]) if len(diff) == 1 else -1,
            "n_flipped_bits": int(len(diff)),
            "flip_direction": int(n_new - n_old),
            "q_other_ones": int(overlap if n_new > n_old else n_new),
            "c_formula": c_formula,
            "c_direct": c_direct,
            "n_risk_005": int(risk005.sum()),
            "n_death_005": int(death005.sum()),
            "r005": float(death005.sum() / risk005.sum()) if risk005.any() else np.nan,
            "n_risk_strict": int(risk_strict.sum()),
            "n_death_strict": int(death_strict.sum()),
            "r_strict": (float(death_strict.sum() / risk_strict.sum())
                         if risk_strict.any() else np.nan),
        })
    return rows, events


def sanity_table(boundaries: pd.DataFrame, paths: list[Path], event_match: bool) -> pd.DataFrame:
    eps = np.finfo(np.float64).eps
    counts = boundaries.groupby("seed").size()
    checks = [
        ("S1_seed_and_boundary_count", len(paths) == 10 and len(counts) == 10
         and bool((counts == 99).all()), f"paths={len(paths)}, counts={counts.to_dict()}"),
        ("S2_exactly_one_flip_bit", bool((boundaries.n_flipped_bits == 1).all()),
         f"max={int(boundaries.n_flipped_bits.max())}"),
        ("S3_flip_direction", bool(boundaries.flip_direction.isin([-1, 1]).all()),
         f"values={sorted(boundaries.flip_direction.unique().tolist())}"),
        ("S4_formula_direct", bool((np.abs(boundaries.c_formula-boundaries.c_direct)
                                     <= 64*eps).all()),
         f"max_abs={np.abs(boundaries.c_formula-boundaries.c_direct).max():.17g}"),
        ("S5_c_range", bool(boundaries.c_formula.between(
            np.sqrt(1.25/2.25), np.sqrt(15.25/16.25)).all()),
         f"range=[{boundaries.c_formula.min():.17g},{boundaries.c_formula.max():.17g}]"),
        ("S6_dead2path_event_match", event_match, "computed set equals path=reclass set"),
        ("S7_rate_bounds", bool(((boundaries.n_death_005 >= 0)
                                  & (boundaries.n_death_005 <= boundaries.n_risk_005)
                                  & (boundaries.n_risk_005 <= 100)
                                  & boundaries.r005.between(0, 1)).all()), "r005 and counts"),
        ("S8_finite_source", bool(np.isfinite(boundaries[
            ["c_formula", "c_direct", "r005", "r_strict"]].to_numpy()).all()),
         "c and rates"),
    ]
    return pd.DataFrame([{"id": i, "pass": bool(ok), "note": note}
                         for i, ok, note in checks])


def per_seed_stats(boundaries: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seed, sub in boundaries.groupby("seed", sort=True):
        valid = sub[(sub.n_risk_005 >= MIN_RISK) & sub.r005.notna()]
        enough_n = len(valid) >= MIN_BOUNDARY
        enough_c = valid.c_formula.nunique() >= MIN_C_LEVEL
        nonconstant = valid.r005.nunique() >= 2
        rho = rank_corr(valid.c_formula.to_numpy(), valid.r005.to_numpy()) \
            if enough_n and enough_c and nonconstant else np.nan
        vstrict = sub[(sub.n_risk_strict >= MIN_RISK) & sub.r_strict.notna()]
        rho_strict = rank_corr(vstrict.c_formula.to_numpy(), vstrict.r_strict.to_numpy()) \
            if len(vstrict) >= MIN_BOUNDARY and vstrict.c_formula.nunique() >= MIN_C_LEVEL \
            and vstrict.r_strict.nunique() >= 2 else np.nan
        rows.append({
            "seed": int(seed), "n_boundary": len(sub), "n_valid_005": len(valid),
            "n_c_level_005": int(valid.c_formula.nunique()),
            "r005_nonconstant": bool(nonconstant), "rho_005": rho,
            "n_valid_strict": len(vstrict), "rho_strict": rho_strict,
        })
    return pd.DataFrame(rows)


def bootstrap_rho(per_seed: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(BOOT_SEED)
    draws = rng.integers(0, len(per_seed), size=(BOOT_N, len(per_seed)))
    out = {"rho_005": np.full(BOOT_N, np.nan),
           "rho_strict": np.full(BOOT_N, np.nan)}
    for key in out:
        vals = per_seed[key].to_numpy(dtype=np.float64)
        chosen = vals[draws]
        count = np.isfinite(chosen).sum(axis=1)
        out[key] = np.divide(np.nansum(chosen, axis=1), count,
                             out=np.full(BOOT_N, np.nan), where=count > 0)
    boot = pd.DataFrame({"replicate": np.arange(BOOT_N), **out})
    main = out["rho_005"]
    valid_seed = int(per_seed.rho_005.notna().sum())
    finite = bool(np.isfinite(main).all())
    point = float(per_seed.rho_005.mean()) if valid_seed else np.nan
    if valid_seed < MIN_VALID_SEED:
        verdict, lo, hi = "INCONCLUSIVE_GUARD", np.nan, np.nan
    elif not finite:
        verdict, lo, hi = "SANITY_FAIL_NONFINITE", np.nan, np.nan
    else:
        lo, hi = np.percentile(main, [2.5, 97.5])
        verdict = ("NEGATIVE_ASSOCIATION" if hi < 0 else
                   "OPPOSITE_ASSOCIATION" if lo > 0 else "INCONCLUSIVE")
    return boot, {"M1": verdict, "rho_mean": point, "ci_lo": float(lo),
                  "ci_hi": float(hi), "n_valid_seed": valid_seed,
                  "n_boot_nonfinite": int((~np.isfinite(main)).sum())}


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", default="results/ratchet_log_0819")
    parser.add_argument("--outdir", default="results/task_rotation_m1_0828")
    parser.add_argument("--sanity-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    inp = (root / args.input).resolve() if not Path(args.input).is_absolute() else Path(args.input)
    out = (root / args.outdir).resolve() if not Path(args.outdir).is_absolute() else Path(args.outdir)
    git_state = git_info(root)
    out.mkdir(parents=True, exist_ok=True)
    paths = sorted((inp / "logs").glob("seed*.npz"),
                   key=lambda p: int(p.stem.removeprefix("seed")))
    all_rows, computed_events = [], []
    source = []
    for path in paths:
        d = load_one(path)
        rows, events = boundary_rows(d)
        all_rows.extend(rows)
        computed_events.extend(events)
        source.append({"path": str(path.relative_to(root)), "sha256": sha256(path),
                       "keys": sorted(d), "shape_step": list(d["step"].shape)})
    boundaries = pd.DataFrame(all_rows).sort_values(["seed", "boundary_step"])

    event_path = root / "results/dead2path_0821/events.csv"
    ev = pd.read_csv(event_path)
    expected_events = set(tuple(map(int, row)) for row in ev.loc[ev.path == "reclass",
        ["seed", "unit", "step_prev", "step_post"]].itertuples(index=False, name=None))
    event_match = set(computed_events) == expected_events
    sanity = sanity_table(boundaries, paths, event_match)
    boundaries.to_csv(out / "boundary_table.csv", index=False)
    sanity.to_csv(out / "sanity.csv", index=False)
    config = {"analysis": "task_rotation_m1_0828", "spec_vault_commit": SPEC_VAULT_COMMIT,
              "tau": TAU, "min_risk": MIN_RISK, "min_boundary": MIN_BOUNDARY,
              "min_c_level": MIN_C_LEVEL, "min_valid_seed": MIN_VALID_SEED,
              "bootstrap_B": BOOT_N, "bootstrap_seed": BOOT_SEED,
              "omp_num_threads": os.environ.get("OMP_NUM_THREADS")}
    write_json(out / "config.json", config)
    write_json(out / "provenance.json", {
        "git": git_state, "spec_vault_commit": SPEC_VAULT_COMMIT,
        "python": sys.version, "platform": platform.platform(),
        "numpy": np.__version__, "pandas": pd.__version__, "sources": source,
        "dead2path_events_sha256": sha256(event_path),
    })
    if not bool(sanity["pass"].all()):
        pd.DataFrame([{"M1": "SANITY_FAIL"}]).to_csv(out / "verdict.csv", index=False)
        raise SystemExit("M1 structural sanity failed")
    if args.sanity_only:
        print(f"M1 structural sanity PASS -> {out}")
        return

    per_seed = per_seed_stats(boundaries)
    boot, verdict = bootstrap_rho(per_seed)
    per_seed.to_csv(out / "per_seed.csv", index=False)
    boot.to_csv(out / "bootstrap.csv", index=False)
    pd.DataFrame([{**verdict, "bootstrap_B": BOOT_N,
                   "bootstrap_seed": BOOT_SEED}]).to_csv(out / "verdict.csv", index=False)
    summary = f"""# task_rotation_m1_0828

## 主判定

- **{verdict['M1']}**
- seed 等重み Spearman 平均: {verdict['rho_mean']:+.6f}
- 95% seed-cluster bootstrap CI: [{verdict['ci_lo']:+.6f}, {verdict['ci_hi']:+.6f}]
- 有効 seed: {verdict['n_valid_seed']}/10

## 構造確認

- 隣接タスクは全境界で 1-bit flip
- 先生提示の一般余弦式と直接内積は機械精度内で一致
- `dead2path_0821` の再分類死イベント集合と完全一致

## スコープ

condA・w100・T=10,000 の既存ログにおける観察的な境界別関連であり、因果効果ではない。
"""
    (out / "summary.md").write_text(summary)
    print(f"M1 {verdict['M1']} -> {out}")


if __name__ == "__main__":
    main()
