"""function_blind_direct_0823 pilot 集計。

この出力は confirmation の設計専用で、機能盲目性の支持・否定には使わない。

実行::

  OMP_NUM_THREADS=1 .venv/bin/python -m analysis.function_blind_direct.pilot \
    --logs results/function_blind_direct_0823_pilot/logs \
    --outdir results/function_blind_direct_0823_pilot
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SWITCHES = tuple(range(200_000, 800_001, 10_000))
TAU = 0.05
GROUPS = ("low", "mid", "high")
BOOT_N = 10_000
BOOT_SEED = 20260828
EQUIV_MARGIN = 0.05
REQUIRED_UNIT_KEYS = (
    "p_hat", "pre_max", "x", "r", "w_norm", "b", "v",
    "utility_raw", "utility_nmse",
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
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def qlabels(values: pd.Series) -> pd.Series:
    a = values.to_numpy(dtype=np.float64)
    if a.size < 3 or not np.isfinite(a).all():
        return pd.Series([None] * a.size, index=values.index, dtype="object")
    q1, q2 = np.quantile(a, (1 / 3, 2 / 3))
    lab = np.where(a <= q1, "low", np.where(a <= q2, "mid", "high"))
    return pd.Series(lab, index=values.index, dtype="object")


def load_logs(logdir: Path) -> list[dict]:
    logs = []
    for path in sorted(logdir.glob("seed*.npz")):
        with np.load(path, allow_pickle=False) as z:
            required = ("step", "seed", "width", "period", "generator_offset",
                        "eval_nmse", "y_var") + REQUIRED_UNIT_KEYS
            missing = [k for k in required if k not in z.files]
            if missing:
                raise SystemExit(f"{path}: missing keys {missing}")
            d = {k: np.array(z[k]) for k in required}
            d["run_id"] = str(z["run_id"]) if "run_id" in z.files else path.stem
            d["path"] = path
        logs.append(d)
    logs.sort(key=lambda d: int(d["seed"]))
    if not logs:
        raise SystemExit(f"ログがない: {logdir}")
    seeds = [int(d["seed"]) for d in logs]
    if seeds != list(range(10)):
        raise SystemExit(f"pilotはseed label 0..9を要求: {seeds}")
    if any(int(d["generator_offset"]) != 0 for d in logs):
        raise SystemExit("pilotはgenerator_offset=0だけを読む")
    return logs


def sanity(logs: list[dict]) -> dict:
    max_quant = 0.0
    max_geom = 0.0
    max_dead_margin_mismatch = 0
    finite = True
    rows = []
    need_steps = {s + 1 for s in SWITCHES} | {s + 10_000 for s in SWITCHES}
    for d in logs:
        step = d["step"].astype(np.int64)
        p = d["p_hat"].astype(np.float64)
        x = d["x"].astype(np.float64)
        r = d["r"].astype(np.float64)
        wn = d["w_norm"].astype(np.float64)
        margin = d["pre_max"].astype(np.float64)
        quant = float(np.max(np.abs(p * 32.0 - np.rint(p * 32.0))))
        geom = float(np.max(np.abs(x * x + r * r - wn * wn)
                            / np.maximum(wn * wn, 1e-30)))
        mismatch = int(np.count_nonzero((p == 0.0) != (margin <= 0.0)))
        fin = all(np.isfinite(d[k]).all() for k in REQUIRED_UNIT_KEYS +
                  ("eval_nmse", "y_var"))
        present = need_steps.issubset(set(int(v) for v in step))
        ok = (int(d["width"]) == 100 and int(d["period"]) == 10_000
              and present and quant < 1e-7 and geom < 1e-10
              and mismatch == 0 and fin)
        max_quant = max(max_quant, quant)
        max_geom = max(max_geom, geom)
        max_dead_margin_mismatch = max(max_dead_margin_mismatch, mismatch)
        finite &= fin
        rows.append(dict(seed=int(d["seed"]), n_step=int(step.size),
                         required_steps_present=present, quant_error=quant,
                         geometry_rel_error=geom, dead_margin_mismatch=mismatch,
                         finite=fin, ok=ok))
    return dict(
        pass_all=bool(all(r["ok"] for r in rows)), rows=rows,
        phat_quant_maxerr=max_quant, geometry_max_relerr=max_geom,
        dead_margin_mismatch=max_dead_margin_mismatch, finite=bool(finite),
    )


def build_exposures(logs: list[dict]) -> pd.DataFrame:
    rows = []
    for d in logs:
        step_to_i = {int(s): i for i, s in enumerate(d["step"])}
        seed = int(d["seed"])
        for switch in SWITCHES:
            t0, t1 = switch + 1, switch + 10_000
            if t0 not in step_to_i or t1 not in step_to_i:
                raise SystemExit(f"seed={seed}: missing t0/t1 {t0}/{t1}")
            i0, i1 = step_to_i[t0], step_to_i[t1]
            p0 = d["p_hat"][i0].astype(np.float64)
            risk = p0 >= TAU
            p1 = d["p_hat"][i1].astype(np.float64)
            for unit in np.flatnonzero(risk):
                rows.append(dict(
                    seed=seed, unit=int(unit), switch=int(switch),
                    t0=t0, t1=t1, p_hat=float(p0[unit]),
                    pre_max=float(d["pre_max"][i0, unit]),
                    x=float(d["x"][i0, unit]), r=float(d["r"][i0, unit]),
                    w_norm=float(d["w_norm"][i0, unit]),
                    b=float(d["b"][i0, unit]), v=float(d["v"][i0, unit]),
                    utility_raw=float(d["utility_raw"][i0, unit]),
                    utility_nmse=float(d["utility_nmse"][i0, unit]),
                    eval_nmse=float(d["eval_nmse"][i0]),
                    y_var=float(d["y_var"][i0]),
                    end_strict_dead=int(p1[unit] == 0.0),
                    end_dead_0_05=int(p1[unit] < TAU),
                ))
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("risk exposureが0件")

    # 探索用の2通り。confirmationではどちらを主にするか別specで固定する。
    for _, idx in df.groupby("t0", sort=True).groups.items():
        df.loc[idx, "utility_group"] = qlabels(df.loc[idx, "utility_nmse"])
        df.loc[idx, "p_bin"] = qlabels(df.loc[idx, "p_hat"])
        df.loc[idx, "margin_bin"] = qlabels(df.loc[idx, "pre_max"])
    for _, idx in df.groupby(["t0", "p_bin", "margin_bin"], sort=True,
                             dropna=False).groups.items():
        df.loc[idx, "utility_cell_group"] = qlabels(df.loc[idx, "utility_nmse"])
    for col in ("utility_group", "p_bin", "margin_bin", "utility_cell_group"):
        df[col] = pd.Categorical(df[col], categories=GROUPS, ordered=True)
    return df.sort_values(["seed", "t0", "unit"]).reset_index(drop=True)


def seed_equal_rates(df: pd.DataFrame, group_col: str, outcome: str) -> pd.DataFrame:
    rows = []
    for seed in sorted(df.seed.unique()):
        ds = df[df.seed == seed]
        for group in GROUPS:
            y = ds.loc[ds[group_col] == group, outcome]
            rows.append(dict(seed=int(seed), group=group, n=int(y.size),
                             n_event=int(y.sum()), risk=float(y.mean()) if y.size else np.nan))
    return pd.DataFrame(rows)


def bootstrap_seed_rd(rates: pd.DataFrame, B: int = BOOT_N,
                      rng_seed: int = BOOT_SEED) -> tuple[float, float, float, np.ndarray]:
    seeds = sorted(rates.seed.unique())
    wide = rates.pivot(index="seed", columns="group", values="risk").reindex(seeds)
    low = wide["low"].to_numpy(float)
    high = wide["high"].to_numpy(float)
    valid = np.isfinite(low) & np.isfinite(high)
    if not valid.all():
        low, high = low[valid], high[valid]
    point = float(np.mean(high - low))
    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(0, len(low), size=(B, len(low)))
    boot = (high[idx] - low[idx]).mean(axis=1)
    lo, hi = np.quantile(boot, (0.025, 0.975))
    return point, float(lo), float(hi), boot


def adjusted_arrays(df: pd.DataFrame, outcome: str) -> tuple[np.ndarray, np.ndarray]:
    """seed×(t0,p_bin,margin_bin)×{low,high} の件数・event数。"""
    d = df.dropna(subset=["utility_cell_group", "p_bin", "margin_bin"]).copy()
    d = d[d.utility_cell_group.isin(["low", "high"])]
    seeds = sorted(d.seed.unique())
    seed_map = {s: i for i, s in enumerate(seeds)}
    keys = list(zip(d.t0.astype(int), d.p_bin.astype(str), d.margin_bin.astype(str)))
    cell_codes, uniques = pd.factorize(keys, sort=True)
    n = np.zeros((len(seeds), len(uniques), 2), dtype=np.float64)
    e = np.zeros_like(n)
    for row, cell in zip(d.itertuples(index=False), cell_codes):
        gi = 0 if str(row.utility_cell_group) == "low" else 1
        si = seed_map[int(row.seed)]
        n[si, cell, gi] += 1
        e[si, cell, gi] += int(getattr(row, outcome))
    return n, e


def rd_from_counts(n: np.ndarray, e: np.ndarray) -> np.ndarray:
    """末尾3軸が cell×{low,high} のとき、先行軸ごとの調整RDを返す。"""
    valid = (n[..., 0] > 0) & (n[..., 1] > 0)
    w = np.minimum(n[..., 0], n[..., 1]) * valid
    low = np.divide(e[..., 0], n[..., 0], out=np.zeros_like(e[..., 0]),
                    where=n[..., 0] > 0)
    high = np.divide(e[..., 1], n[..., 1], out=np.zeros_like(e[..., 1]),
                     where=n[..., 1] > 0)
    num = (w * (high - low)).sum(axis=-1)
    den = w.sum(axis=-1)
    return np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)


def bootstrap_adjusted(df: pd.DataFrame, outcome: str, B: int = BOOT_N,
                       rng_seed: int = BOOT_SEED + 1) -> tuple[float, float, float, np.ndarray]:
    n, e = adjusted_arrays(df, outcome)
    point = float(rd_from_counts(n.sum(axis=0), e.sum(axis=0)))
    rng = np.random.default_rng(rng_seed)
    boot = np.empty(B, dtype=np.float64)
    batch = 500
    for start in range(0, B, batch):
        stop = min(B, start + batch)
        idx = rng.integers(0, n.shape[0], size=(stop - start, n.shape[0]))
        nb = n[idx].sum(axis=1)
        eb = e[idx].sum(axis=1)
        boot[start:stop] = rd_from_counts(nb, eb)
    valid = boot[np.isfinite(boot)]
    lo, hi = np.quantile(valid, (0.025, 0.975))
    return point, float(lo), float(hi), boot


def rough_n_seed(n_seed: int, point: float, lo: float, hi: float,
                 margin: float = EQUIV_MARGIN) -> int | None:
    target_half = margin - abs(point)
    if target_half <= 0:
        return None
    half = (hi - lo) / 2
    return max(n_seed, int(math.ceil(n_seed * (half / target_half) ** 2)))


def rank_corr(a: pd.Series, b: pd.Series) -> float:
    x = a.rank(method="average").to_numpy(float)
    y = b.rank(method="average").to_numpy(float)
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def write_summary(outdir: Path, df: pd.DataFrame, san: dict,
                  verdicts: pd.DataFrame, diagnostics: pd.DataFrame,
                  runner_meta: dict | None) -> None:
    lines = [
        "# function_blind_direct_0823 pilot",
        "",
        "> **パイロット専用。以下の効果量・CIを機能盲目性の確認結果として引用しない。**",
        "",
        "## 1. サニティ",
        "",
        f"- 集計sanity: **{'PASS' if san['pass_all'] else 'FAIL'}**",
        f"- p̂量子化最大誤差: {san['phat_quant_maxerr']:.3g}",
        f"- x/r幾何最大相対誤差: {san['geometry_max_relerr']:.3g}",
        f"- strict_deadとpre_max符号の不一致: {san['dead_margin_mismatch']}",
    ]
    if runner_meta:
        rs = runner_meta.get("sanity", {})
        lines += [f"- runner sanity: `{json.dumps(rs, ensure_ascii=False)}`"]
    lines += [
        "", "## 2. リスク集合と転帰", "",
        f"- 曝露: {len(df):,}（seed={df.seed.nunique()}, task={df.t0.nunique()}）",
        f"- endpoint strict_dead: {df.end_strict_dead.mean():.4f} "
        f"({int(df.end_strict_dead.sum()):,}/{len(df):,})",
        f"- endpoint dead_0.05: {df.end_dead_0_05.mean():.4f}",
        "", "## 3. ΔL 診断", "",
    ]
    for name in ("utility_nmse", "utility_raw"):
        q = df[name].quantile([0, .1, .25, .5, .75, .9, 1]).to_dict()
        lines.append(f"- {name}: min={q[0.0]:+.4g}, p10={q[0.1]:+.4g}, "
                     f"median={q[0.5]:+.4g}, p90={q[0.9]:+.4g}, max={q[1.0]:+.4g}")
    lines += ["", "## 4. 探索的候補（confirmationの結果ではない）", ""]
    for row in verdicts.itertuples(index=False):
        nrough = "NA" if pd.isna(row.rough_n_seed) else str(int(row.rough_n_seed))
        lines.append(f"- {row.analysis}/{row.outcome}: RD(high−low)="
                     f"{row.rd:+.4f} [{row.ci_lo:+.4f}, {row.ci_hi:+.4f}], "
                     f"rough seed={nrough}")
    lines += [
        "", "この節を見てconfirmation specを固定する。好都合な候補だけを主解析にしない。",
        "", "## 5. 限界", "",
        "- pilotはgenerator_offset=0で旧軌道を再計装したもので、独立確認ではない。",
        "- ΔLは現在タスク上の単独消音効果。冗長性、相互作用、将来タスク価値を表さない。",
        "- 同一unitの反復曝露がある。unit独立のSEを使わない。",
    ]
    (outdir / "pilot_summary.md").write_text("\n".join(lines) + "\n")


def run(logdir: Path, outdir: Path, bootstrap_n: int = BOOT_N) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    logs = load_logs(logdir)
    san = sanity(logs)
    if not san["pass_all"]:
        raise SystemExit(f"sanity FAIL: {san}")
    df = build_exposures(logs)
    df.to_csv(outdir / "exposures.csv", index=False)

    rate_frames = []
    verdict_rows = []
    for group_col, label in (("utility_group", "unadjusted_t0_tertile"),):
        for outcome in ("end_strict_dead", "end_dead_0_05"):
            rates = seed_equal_rates(df, group_col, outcome)
            rates["analysis"] = label
            rates["outcome"] = outcome
            rate_frames.append(rates)
            rd, lo, hi, _ = bootstrap_seed_rd(rates, B=bootstrap_n)
            verdict_rows.append(dict(
                analysis=label, outcome=outcome, rd=rd, ci_lo=lo, ci_hi=hi,
                rough_n_seed=rough_n_seed(df.seed.nunique(), rd, lo, hi),
            ))
    for outcome in ("end_strict_dead", "end_dead_0_05"):
        rd, lo, hi, _ = bootstrap_adjusted(df, outcome, B=bootstrap_n)
        verdict_rows.append(dict(
            analysis="phat_margin_3x3_adjusted", outcome=outcome,
            rd=rd, ci_lo=lo, ci_hi=hi,
            rough_n_seed=rough_n_seed(df.seed.nunique(), rd, lo, hi),
        ))
    pd.concat(rate_frames, ignore_index=True).to_csv(outdir / "pilot_rates.csv", index=False)
    verdicts = pd.DataFrame(verdict_rows)
    verdicts.to_csv(outdir / "pilot_verdict_candidates.csv", index=False)

    diag = []
    for name in REQUIRED_UNIT_KEYS:
        a = df[name]
        diag += [
            dict(metric=f"{name}.min", value=float(a.min())),
            dict(metric=f"{name}.median", value=float(a.median())),
            dict(metric=f"{name}.max", value=float(a.max())),
        ]
    diag += [
        dict(metric="utility_nmse.frac_positive", value=float((df.utility_nmse > 0).mean())),
        dict(metric="utility_nmse.frac_negative", value=float((df.utility_nmse < 0).mean())),
        dict(metric="event.strict_dead", value=float(df.end_strict_dead.mean())),
        dict(metric="event.dead_0_05", value=float(df.end_dead_0_05.mean())),
    ]
    for cov in ("p_hat", "pre_max", "x", "r", "w_norm", "b", "v"):
        diag.append(dict(metric=f"spearman.utility_nmse.{cov}",
                         value=rank_corr(df.utility_nmse, df[cov])))
    diagnostics = pd.DataFrame(diag)
    diagnostics.to_csv(outdir / "pilot_diagnostics.csv", index=False)
    pd.DataFrame(san["rows"]).to_csv(outdir / "pilot_sanity.csv", index=False)

    runner_meta_path = outdir / "instrumentation_meta.json"
    runner_meta = json.loads(runner_meta_path.read_text()) if runner_meta_path.exists() else None
    meta = dict(
        git=git_hash(), spec="specs/spec_function_blind_direct_0823_pilot.md",
        logdir=str(logdir.resolve()), n_seed=int(df.seed.nunique()),
        n_exposure=int(len(df)), bootstrap_n=int(bootstrap_n),
        bootstrap_seed=BOOT_SEED, equiv_margin=EQUIV_MARGIN,
        input_sha256={p.name: sha256(p) for p in sorted(logdir.glob("seed*.npz"))},
        sanity=san, runner_meta=runner_meta,
        prohibition="pilot only; do not promote effect estimates",
    )
    (outdir / "pilot_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    write_summary(outdir, df, san, verdicts, diagnostics, runner_meta)
    print((outdir / "pilot_summary.md").read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", type=Path,
                    default=ROOT / "results/function_blind_direct_0823_pilot/logs")
    ap.add_argument("--outdir", type=Path,
                    default=ROOT / "results/function_blind_direct_0823_pilot")
    ap.add_argument("--bootstrap-n", type=int, default=BOOT_N)
    args = ap.parse_args()
    run(args.logs, args.outdir, args.bootstrap_n)


if __name__ == "__main__":
    main()
