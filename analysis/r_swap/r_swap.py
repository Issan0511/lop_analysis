"""作業6 の r 差し替え解析 (spec_r_swap_0824)。

登録実行コマンド::

  OMP_NUM_THREADS=1 .venv/bin/python -m analysis.r_swap.r_swap \
    --exposures results/function_blind_direct_0823_confirm/exposures.csv \
    --outdir results/r_swap_0824

推定量・重み規約・bootstrap の構成は作業6 の
``analysis/function_blind_direct/confirm.py`` から直接 import して共有する。
本解析が変えるのは**層別**（``cell_id`` → ``cell_id × r_half`` など）と
**判定規則**（spec §7）だけである。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analysis.function_blind_direct import confirm as fb


ROOT = Path(__file__).resolve().parents[2]
SPEC = "specs/spec_r_swap_0824.md"
SOURCE_COMMIT = "ac29d87"          # spec §3: exposures.csv の出所
BOOT_SEED = 20_260_824             # spec §6: 本解析 (R1/R2/R3)
FB_BOOT_SEED = fb.BOOT_SEED        # 20260831: S2 の作業6 再現のみ (spec §0)
BOOT_N = fb.BOOT_N                 # 10000
SEEDS = fb.SEEDS                   # 0..19
GROUPS = fb.GROUPS                 # low / mid / high
EQUIV = fb.EQUIV_MARGIN            # 0.05
SPECIFIC_POINT = -0.15             # spec §7.1
G1_MIN_RETENTION = 0.40            # spec §7.3
G2_MAX_HALFWIDTH = 0.10            # spec §7.3
FB_N_LOW = 5023                    # spec §7.3 の分母 (作業6 主解析)
FB_N_HIGH = 4431
FB_REGISTERED = dict(              # 作業6 verdict.csv の登録値 (spec §9 S2)
    rd=-0.23527554781786336,
    ci_lo=-0.28122021273155479,
    ci_hi=-0.23246969586791047,
)
OUTCOMES = ("end_strict_dead", "end_dead_0_05")
PRIMARY_OUTCOME = "end_strict_dead"
CSV_NAMES = ("verdict.csv", "cells.csv", "rates.csv", "sanity.csv")

# 層数が作業6 の 2 倍近くになるので bootstrap のチャンクを絞る (メモリのみの都合)。
# 各 replicate は独立に計算されるためチャンク幅は数値へ影響しない。
fb.BOOT_BATCH = 25


# --------------------------------------------------------------------------
# 補助
# --------------------------------------------------------------------------
def git_hash() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:  # pragma: no cover
        return "unknown"


def git_dirty() -> bool:
    try:
        out = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                             capture_output=True, text=True, check=True).stdout
        tracked = [line for line in out.splitlines() if not line.startswith("??")]
        return bool(tracked)
    except Exception:  # pragma: no cover
        return True


def rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman = average-rank Pearson (scipy 非依存、pilot.py と同一定義)。"""
    x = pd.Series(a).rank(method="average").to_numpy(float)
    y = pd.Series(b).rank(method="average").to_numpy(float)
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def boot_indices(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, len(SEEDS), size=(BOOT_N, len(SEEDS)))


# --------------------------------------------------------------------------
# 層別 (spec §4.2 / §4.3)
# --------------------------------------------------------------------------
def median_half(frame: pd.DataFrame, column: str) -> np.ndarray:
    """各幾何セル内で中央値分割。同値は上位側 (=1) へ (spec §4.2)。"""
    median = frame.groupby("cell_id")[column].transform("median").to_numpy(float)
    return (frame[column].to_numpy(float) >= median).astype(np.int64)


def stratify(frame: pd.DataFrame, *, group_col: str, half_col: str) -> pd.DataFrame:
    """cell_id × half を新しい層 (cell_id) として組み直す。

    ``fb._count_arrays`` / ``fb.adjusted_effect`` が読む列名
    (``cell_id`` / ``cell_valid`` / ``utility_group``) に合わせて詰め替える。
    有効セル規則は spec §4.3 (low と high の両方に 1 曝露以上)。
    """
    out = frame.copy()
    out["base_cell_id"] = out.cell_id.astype(np.int64)
    out["half"] = out[half_col].astype(np.int64)
    keys = list(zip(out.base_cell_id.astype(int), out.half.astype(int)))
    code = {key: index for index, key in enumerate(sorted(set(keys)))}
    out["cell_id"] = np.asarray([code[key] for key in keys], dtype=np.int64)
    out["utility_group"] = out[group_col].astype(str)
    counts = out.groupby("cell_id").utility_group.value_counts().unstack(fill_value=0)
    for group in GROUPS:
        if group not in counts:
            counts[group] = 0
    valid = (counts["low"] > 0) & (counts["high"] > 0)
    out["cell_valid"] = out.cell_id.map(valid).astype(bool)
    return out


def cell_rows(name: str, assigned: pd.DataFrame, *, half_col: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cell_id, part in assigned.groupby("cell_id", sort=True):
        row: dict[str, Any] = dict(
            analysis=name, stratum_id=int(cell_id),
            base_cell_id=int(part.base_cell_id.iloc[0]),
            half_col=half_col, half=int(part.half.iloc[0]),
            t0=int(part.t0.iloc[0]), p_count=int(part.p_count.iloc[0]),
            margin5_bin=int(part.margin5_bin.iloc[0]),
            n_total=int(len(part)),
        )
        for group in GROUPS:
            row[f"n_{group}"] = int((part.utility_group == group).sum())
        row["valid_low_high"] = bool(row["n_low"] > 0 and row["n_high"] > 0)
        for outcome in OUTCOMES:
            for group in GROUPS:
                values = part.loc[part.utility_group == group, outcome]
                row[f"{outcome}_events_{group}"] = int(values.sum())
                row[f"{outcome}_risk_{group}"] = (
                    float(values.mean()) if len(values) else np.nan
                )
        rows.append(row)
    return pd.DataFrame(rows)


def rate_rows(name: str, assigned: pd.DataFrame, outcome: str) -> pd.DataFrame:
    table = fb.rate_table(assigned, outcome)
    table.insert(0, "analysis", name)
    table["value"] = np.nan
    return table


# --------------------------------------------------------------------------
# 判定 (spec §7)
# --------------------------------------------------------------------------
def classify_r1(rd: float, ci_lo: float, ci_hi: float) -> str:
    """spec §7.1、規則順 SPECIFIC > REDUCIBLE > INCONCLUSIVE。"""
    excludes_zero = (ci_lo > 0.0) or (ci_hi < 0.0)
    if excludes_zero and rd <= SPECIFIC_POINT:
        return "SPECIFIC"
    if ci_lo >= -EQUIV and ci_hi <= EQUIV:
        return "REDUCIBLE"
    return "INCONCLUSIVE"


def classify_r2(ci_lo: float, ci_hi: float) -> tuple[str, str]:
    """spec §7.2、記載順に適用。重複が起きたら note で明示する。"""
    protective = ci_hi < 0.0
    equivalent = (ci_lo >= -EQUIV) and (ci_hi <= EQUIV)
    if protective:
        note = ("同時に等価域にも収まる (規則が重複)。記載順で PROTECTIVE_R を採る"
                if equivalent else "")
        return "PROTECTIVE_R", note
    if equivalent:
        return "NULL_R", ""
    return "INCONCLUSIVE", ""


def effect_row(*, analysis: str, role: str, outcome: str, group_col: str,
               half_col: str, effect: dict[str, Any], verdict: str,
               criterion: str, rule_order: str, boot_seed: int,
               note: str = "") -> dict[str, Any]:
    half_width = 0.5 * (effect["ci_hi"] - effect["ci_lo"])
    return dict(
        analysis=analysis, role=role, outcome=outcome,
        group=group_col, stratum=half_col,
        rd=effect["rd"], ci_lo=effect["ci_lo"], ci_hi=effect["ci_hi"],
        ci_halfwidth=half_width, equiv_margin=EQUIV,
        verdict=verdict, criterion=criterion, rule_order=rule_order,
        n_boot=effect["n_boot"], bootstrap_seed=boot_seed,
        n_boot_nonfinite=effect["n_boot_nonfinite"], n_cell=effect["n_cell"],
        weight=effect["weight"], n_low=effect["n_low"], n_high=effect["n_high"],
        events_low=effect["events_low"], events_high=effect["events_high"],
        risk_low=effect["risk_low"], risk_high=effect["risk_high"],
        note=note,
    )


# --------------------------------------------------------------------------
# サニティ (spec §9)
# --------------------------------------------------------------------------
def sanity_rows(frame: pd.DataFrame, rebuilt: pd.DataFrame,
                s2_effect: dict[str, Any],
                half_stats: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    # S1: cell_id の再構成が exposures.csv の既存列と完全一致
    checks = {
        "cell_id": np.array_equal(rebuilt.cell_id.to_numpy(np.int64),
                                  frame.cell_id.to_numpy(np.int64)),
        "margin5_bin": np.array_equal(rebuilt.margin_bin.to_numpy(np.int64),
                                      frame.margin5_bin.to_numpy(np.int64)),
        "utility_nmse_group": bool((rebuilt.utility_group.to_numpy()
                                    == frame.utility_nmse_group.to_numpy()).all()),
        "primary_cell_valid": bool((rebuilt.cell_valid.to_numpy()
                                    == frame.primary_cell_valid.to_numpy()).all()),
    }
    rows.append(dict(
        check="S1", name="cell_id 等の再構成一致",
        value=_json_compact(checks),
        criterion="4 列すべて完全一致",
        status="PASS" if all(checks.values()) else "FAIL",
        note="生列 (t0, p_count, pre_max, utility_nmse) から assign_cells で再構成",
    ))

    # S2: 作業6 主解析の再現
    diffs = {key: abs(s2_effect[key] - FB_REGISTERED[key])
             for key in ("rd", "ci_lo", "ci_hi")}
    exact = all(s2_effect[key] == FB_REGISTERED[key]
                for key in ("rd", "ci_lo", "ci_hi"))
    rows.append(dict(
        check="S2", name="作業6 主解析の再現",
        value=_json_compact(dict(
            rd=s2_effect["rd"], ci_lo=s2_effect["ci_lo"], ci_hi=s2_effect["ci_hi"],
            n_low=s2_effect["n_low"], n_high=s2_effect["n_high"],
            verdict=s2_effect["verdict"], max_abs_diff=max(diffs.values()),
        )),
        criterion=("登録値 rd/ci_lo/ci_hi と float 完全一致 "
                   f"(bootstrap_seed={FB_BOOT_SEED})"),
        status="PASS" if exact else "FAIL",
        note="不一致なら spec §9 により中止",
    ))

    # S3: x^2 + r^2 = w_norm^2
    lhs = frame.x.to_numpy(float) ** 2 + frame.r.to_numpy(float) ** 2
    rhs = frame.w_norm.to_numpy(float) ** 2
    rel = np.abs(lhs - rhs) / np.maximum(np.abs(rhs), 1e-300)
    max_rel = float(rel.max())
    rows.append(dict(
        check="S3", name="x^2 + r^2 = ||w||^2 の最大相対誤差",
        value=f"{max_rel:.6e}", criterion="< 1e-10",
        status="PASS" if max_rel < 1e-10 else "FAIL", note="",
    ))

    # S4: p_hat == 0 <=> pre_max <= 0
    left = frame.p_hat.to_numpy(float) == 0.0
    right = frame.pre_max.to_numpy(float) <= 0.0
    mismatch = int((left != right).sum())
    rows.append(dict(
        check="S4", name="p_hat==0 <=> pre_max<=0 の不一致件数",
        value=str(mismatch), criterion="= 0",
        status="PASS" if mismatch == 0 else "FAIL",
        note=(f"n_p_hat_zero={int(left.sum())}, n_pre_max_nonpos={int(right.sum())}。"
              "リスク集合が p_hat>=0.05 なので両辺とも 0 件になる "
              "**空虚に真の検査**であり、成立しても情報量は無い"),
    ))

    # S5: 中央値分割の群サイズ偏り (報告)
    for half_col, stats in half_stats.items():
        rows.append(dict(
            check="S5", name=f"{half_col} の少数側シェア",
            value=_json_compact(stats),
            criterion="報告のみ (spec §9 S5: 各セルで少数側 >= 1/4 を報告)",
            status="REPORT", note="n=1 のセルは定義上 minority_share=0 になる",
        ))

    # 非有限値
    columns = ["r", "utility_nmse", "pre_max", "p_hat", "w_norm", "x"]
    nonfinite = {c: int((~np.isfinite(frame[c].to_numpy(float))).sum())
                 for c in columns}
    rows.append(dict(
        check="S0", name="入力列の非有限値",
        value=_json_compact(nonfinite), criterion="報告 (spec §6: 除外しない)",
        status="REPORT" if sum(nonfinite.values()) else "PASS", note="",
    ))
    return pd.DataFrame(rows)


def _json_compact(value: Any) -> str:
    def default(obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return str(obj)
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=default)


def half_split_stats(frame: pd.DataFrame, half_col: str) -> dict[str, Any]:
    shares: list[float] = []
    for _, part in frame.groupby("cell_id", sort=True):
        n = len(part)
        upper = int((part[half_col] == 1).sum())
        shares.append(min(upper, n - upper) / n)
    array = np.asarray(shares, dtype=np.float64)
    return dict(
        n_cell=int(array.size),
        median_minority_share=float(np.median(array)),
        mean_minority_share=float(array.mean()),
        frac_cells_below_quarter=float((array < 0.25).mean()),
        n_cell_below_quarter=int((array < 0.25).sum()),
    )


# --------------------------------------------------------------------------
# 本体
# --------------------------------------------------------------------------
def build_frames(exposures: Path) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    frame = pd.read_csv(exposures)
    frame = frame.sort_values(["seed", "t0", "unit"],
                              kind="mergesort").reset_index(drop=True)

    # --- S1: 生列から幾何セルを再構成 ---
    rebuilt, _ = fb.assign_cells(frame, margin_bins=5, utility_col="utility_nmse")

    # --- S2: 作業6 主解析の再現 (登録 seed 20260831) ---
    s2_assigned, _ = fb.assign_existing_geometry(frame, utility_col="utility_nmse")
    s2_effect = fb.adjusted_effect(s2_assigned, PRIMARY_OUTCOME,
                                   boot_indices(FB_BOOT_SEED))

    # --- 追加層 (spec §4.2) ---
    frame["r_half"] = median_half(frame, "r")
    frame["u_half"] = median_half(frame, "utility_nmse")
    half_stats = {"r_half": half_split_stats(frame, "r_half"),
                  "u_half": half_split_stats(frame, "u_half")}

    indices = boot_indices(BOOT_SEED)
    verdict_rows: list[dict[str, Any]] = []
    cells: list[pd.DataFrame] = []
    rates: list[pd.DataFrame] = []

    # ---------------- R1 (主): ΔL 効果を r で条件付け ----------------
    r1 = stratify(frame, group_col="utility_nmse_group", half_col="r_half")
    r1_effects = {outcome: fb.adjusted_effect(r1, outcome, indices)
                  for outcome in OUTCOMES}
    primary = r1_effects[PRIMARY_OUTCOME]

    retention_low = primary["n_low"] / FB_N_LOW
    retention_high = primary["n_high"] / FB_N_HIGH
    retention = min(retention_low, retention_high)
    g1_fired = bool(retention < G1_MIN_RETENTION)
    half_width = 0.5 * (primary["ci_hi"] - primary["ci_lo"])
    g2_fired = bool(half_width > G2_MAX_HALFWIDTH)

    r1_verdict = classify_r1(primary["rd"], primary["ci_lo"], primary["ci_hi"])
    guard_note = []
    if g1_fired:
        guard_note.append(
            f"G1 発火: 有効曝露保持率 low={retention_low:.4f} / high={retention_high:.4f} "
            f"< {G1_MIN_RETENTION:.2f} -> R1 は記述的扱い、判定に用いない")
    if g2_fired:
        guard_note.append(
            f"G2 発火: CI 半幅 {half_width:.4f} > {G2_MAX_HALFWIDTH:.2f} -> 精度不足")
    role = "descriptive_g1" if g1_fired else "primary"

    verdict_rows.append(effect_row(
        analysis="R1", role=role, outcome=PRIMARY_OUTCOME,
        group_col="utility_nmse_tertile(作業6流用)", half_col="cell_id x r_half",
        effect=primary, verdict=r1_verdict,
        criterion=f"CI が 0 を含まず点推定 <= {SPECIFIC_POINT} -> SPECIFIC / "
                  f"CI ⊂ [-{EQUIV}, +{EQUIV}] -> REDUCIBLE / 他 INCONCLUSIVE",
        rule_order="SPECIFIC>REDUCIBLE>INCONCLUSIVE", boot_seed=BOOT_SEED,
        note="; ".join(guard_note)))
    verdict_rows.append(effect_row(
        analysis="R1", role="secondary_not_main", outcome="end_dead_0_05",
        group_col="utility_nmse_tertile(作業6流用)", half_col="cell_id x r_half",
        effect=r1_effects["end_dead_0_05"],
        verdict=classify_r1(r1_effects["end_dead_0_05"]["rd"],
                            r1_effects["end_dead_0_05"]["ci_lo"],
                            r1_effects["end_dead_0_05"]["ci_hi"]),
        criterion="R1 と同じ規則を副次転帰へ適用 (主判定の差し替えに使わない)",
        rule_order="SPECIFIC>REDUCIBLE>INCONCLUSIVE", boot_seed=BOOT_SEED))
    cells.append(cell_rows("R1", r1, half_col="r_half"))
    for outcome in OUTCOMES:
        rates.append(rate_rows("R1", r1, outcome))

    # ---------------- R2 (副): r 効果を ΔL で条件付け ----------------
    r_tertile, _ = fb.assign_existing_geometry(frame, utility_col="r")
    frame["r_group"] = r_tertile.utility_group.to_numpy()
    r2 = stratify(frame, group_col="r_group", half_col="u_half")
    r2_effects = {outcome: fb.adjusted_effect(r2, outcome, indices)
                  for outcome in OUTCOMES}
    r2_primary = r2_effects[PRIMARY_OUTCOME]
    r2_verdict, r2_note = classify_r2(r2_primary["ci_lo"], r2_primary["ci_hi"])
    r2_retention = min(r2_primary["n_low"] / FB_N_LOW,
                       r2_primary["n_high"] / FB_N_HIGH)
    verdict_rows.append(effect_row(
        analysis="R2", role="secondary_registered", outcome=PRIMARY_OUTCOME,
        group_col="r_tertile(cell_id 内)", half_col="cell_id x u_half",
        effect=r2_primary, verdict=r2_verdict,
        criterion=f"CI が 0 を含まず負 -> PROTECTIVE_R / CI ⊂ [-{EQUIV}, +{EQUIV}] "
                  "-> NULL_R / 他 INCONCLUSIVE",
        rule_order="PROTECTIVE_R>NULL_R>INCONCLUSIVE", boot_seed=BOOT_SEED,
        note="; ".join(x for x in [r2_note,
                                   f"参考: 曝露保持率 {r2_retention:.4f} "
                                   "(G1 は R1 のみに掛かる規定)"] if x)))
    verdict_rows.append(effect_row(
        analysis="R2", role="secondary_not_main", outcome="end_dead_0_05",
        group_col="r_tertile(cell_id 内)", half_col="cell_id x u_half",
        effect=r2_effects["end_dead_0_05"],
        verdict=classify_r2(r2_effects["end_dead_0_05"]["ci_lo"],
                            r2_effects["end_dead_0_05"]["ci_hi"])[0],
        criterion="R2 と同じ規則を副次転帰へ適用 (主判定の差し替えに使わない)",
        rule_order="PROTECTIVE_R>NULL_R>INCONCLUSIVE", boot_seed=BOOT_SEED))
    cells.append(cell_rows("R2", r2, half_col="u_half"))
    for outcome in OUTCOMES:
        rates.append(rate_rows("R2", r2, outcome))

    # ---------------- R3 (記述): 交絡の程度 ----------------
    r3 = r_tertile.copy()
    r3["base_cell_id"] = r3.cell_id.astype(np.int64)
    r3["half"] = -1
    r3_effects = {outcome: fb.adjusted_effect(r3, outcome, indices)
                  for outcome in OUTCOMES}
    for outcome in OUTCOMES:
        verdict_rows.append(effect_row(
            analysis="R3_marginal_r", role="report", outcome=outcome,
            group_col="r_tertile(cell_id 内)", half_col="cell_id のみ",
            effect=r3_effects[outcome], verdict="—",
            criterion="判定なし (spec §5 R3 は記述)",
            rule_order="—", boot_seed=BOOT_SEED,
            note="作業6 と同一層別で群だけ r に差し替えた周辺 RD"))
    cells.append(cell_rows("R3_marginal_r", r3, half_col="(none)"))
    for outcome in OUTCOMES:
        rates.append(rate_rows("R3_marginal_r", r3, outcome))

    spearman = []
    for seed in SEEDS:
        part = frame[frame.seed == seed]
        spearman.append(dict(
            analysis="R3_spearman", scope="seed", seed=int(seed), group="all",
            outcome="-", n=int(len(part)), n_event=np.nan, risk=np.nan,
            value=rank_corr(part.utility_nmse.to_numpy(float),
                            part.r.to_numpy(float))))
    rates.append(pd.DataFrame(spearman))
    rho = np.asarray([row["value"] for row in spearman], dtype=np.float64)
    rates.append(pd.DataFrame([dict(
        analysis="R3_spearman", scope="pooled", seed=-1, group="all",
        outcome="-", n=int(len(frame)), n_event=np.nan, risk=np.nan,
        value=rank_corr(frame.utility_nmse.to_numpy(float),
                        frame.r.to_numpy(float)))]))
    verdict_rows.append(dict(
        analysis="R3_spearman", role="report", outcome="-",
        group="Spearman(utility_nmse, r)", stratum="seed 別",
        rd=float(np.median(rho)), ci_lo=float(rho.min()), ci_hi=float(rho.max()),
        ci_halfwidth=np.nan, equiv_margin=np.nan, verdict="—",
        criterion="判定なし (spec §5 R3)。rd 列=seed 中央値、ci 列=最小/最大",
        rule_order="—", n_boot=0, bootstrap_seed=BOOT_SEED,
        n_boot_nonfinite=0, n_cell=len(SEEDS), weight=np.nan,
        n_low=np.nan, n_high=np.nan, events_low=np.nan, events_high=np.nan,
        risk_low=float(np.quantile(rho, 0.25)),
        risk_high=float(np.quantile(rho, 0.75)),
        note="risk_low/risk_high 列は seed 別 rho の四分位 (Q1/Q3)"))

    # ---------------- §8 分岐: R1 SPECIFIC かつ R2 有意なら 2x2 記述表 ----------------
    branch_2x2 = (r1_verdict == "SPECIFIC" and not g1_fired
                  and r2_verdict == "PROTECTIVE_R")
    if branch_2x2:
        for r_half in (0, 1):
            for u_half in (0, 1):
                part = frame[(frame.r_half == r_half) & (frame.u_half == u_half)]
                for outcome in OUTCOMES:
                    rates.append(pd.DataFrame([dict(
                        analysis="R4_2x2_descriptive", scope="2x2", seed=-1,
                        group=f"r_half={r_half},u_half={u_half}", outcome=outcome,
                        n=int(len(part)), n_event=int(part[outcome].sum()),
                        risk=float(part[outcome].mean()) if len(part) else np.nan,
                        value=np.nan)]))

    verdict = pd.DataFrame(verdict_rows)
    cells_frame = pd.concat(cells, ignore_index=True)
    rates_frame = pd.concat(rates, ignore_index=True)
    sanity = sanity_rows(frame, rebuilt, s2_effect, half_stats)

    diagnostics = dict(
        n_exposure=int(len(frame)), n_seed=int(frame.seed.nunique()),
        n_t0=int(frame.t0.nunique()),
        n_base_cell=int(frame.cell_id.nunique()),
        r1=primary, r1_secondary=r1_effects["end_dead_0_05"],
        r1_verdict=r1_verdict, r1_role=role,
        r2=r2_primary, r2_verdict=r2_verdict, r2_retention=float(r2_retention),
        r3=r3_effects, spearman_median=float(np.median(rho)),
        spearman_min=float(rho.min()), spearman_max=float(rho.max()),
        retention_low=float(retention_low), retention_high=float(retention_high),
        retention=float(retention), g1_fired=g1_fired, g2_fired=g2_fired,
        ci_halfwidth=float(half_width), half_stats=half_stats,
        s2=s2_effect, branch_2x2=bool(branch_2x2),
        n_stratum_r1=int(r1.cell_id.nunique()),
        n_valid_stratum_r1=int(r1.loc[r1.cell_valid, "cell_id"].nunique()),
        n_stratum_r2=int(r2.cell_id.nunique()),
        n_valid_stratum_r2=int(r2.loc[r2.cell_valid, "cell_id"].nunique()),
        sanity_pass=bool((sanity.status != "FAIL").all()),
        s2_pass=bool(sanity.loc[sanity.check == "S2", "status"].iloc[0] == "PASS"),
    )
    frames = {
        "verdict.csv": verdict,
        "cells.csv": cells_frame,
        "rates.csv": rates_frame,
        "sanity.csv": sanity,
    }
    return frames, diagnostics


def write_summary(outdir: Path, diagnostics: dict[str, Any],
                  verdict: pd.DataFrame, meta: dict[str, Any]) -> None:
    r1 = diagnostics["r1"]
    r2 = diagnostics["r2"]
    lines = [
        "# r_swap_0824: 作業6 の r 差し替え解析",
        "",
        f"> 事前登録 spec: `{SPEC}`（実行前 commit `{meta['spec_commit']}`）。"
        f"入力は `{meta['exposures']}`（作業6 commit `{SOURCE_COMMIT}`）の再解析のみで、"
        "新規学習走はゼロ。",
        "",
        "## 主判定 (R1)",
        "",
        f"- **{diagnostics['r1_verdict']}**"
        + ("（**G1 発火により記述的扱い**。判定に用いない）"
           if diagnostics["g1_fired"] else ""),
        f"- 調整 RD (ΔL high−low, `cell_id × r_half` 層別): "
        f"{r1['rd']:+.4f} [{r1['ci_lo']:+.4f}, {r1['ci_hi']:+.4f}]",
        f"- 比較対象（作業6 主解析・`cell_id` のみ）: "
        f"{FB_REGISTERED['rd']:+.4f} [{FB_REGISTERED['ci_lo']:+.4f}, "
        f"{FB_REGISTERED['ci_hi']:+.4f}]",
        f"- low/high pooled 率: {r1['risk_low']:.4f} / {r1['risk_high']:.4f}",
        f"- 判定規則: CI が 0 を含まず点推定 ≤ {SPECIFIC_POINT} → SPECIFIC / "
        f"CI ⊂ [−{EQUIV}, +{EQUIV}] → REDUCIBLE / 他 INCONCLUSIVE",
        "",
        "### 検出力ガード (spec §7.3)",
        "",
        f"- **G1**: 有効曝露保持率 low {r1['n_low']}/{FB_N_LOW} = "
        f"{diagnostics['retention_low']:.4f}、high {r1['n_high']}/{FB_N_HIGH} = "
        f"{diagnostics['retention_high']:.4f}（閾値 {G1_MIN_RETENTION:.2f}）→ "
        f"**{'発火' if diagnostics['g1_fired'] else '不発'}**",
        f"- **G2**: CI 半幅 {diagnostics['ci_halfwidth']:.4f}"
        f"（閾値 {G2_MAX_HALFWIDTH:.2f}）→ "
        f"**{'発火（精度不足）' if diagnostics['g2_fired'] else '不発'}**",
        f"- R1 の層: {diagnostics['n_stratum_r1']:,}（有効 "
        f"{diagnostics['n_valid_stratum_r1']:,}）"
        f" ← 元の幾何セル {diagnostics['n_base_cell']:,}",
        "",
        "## 副次判定 (R2): r 自体の保護効果",
        "",
        f"- **{diagnostics['r2_verdict']}**",
        f"- 調整 RD (r high−low, `cell_id × u_half` 層別): "
        f"{r2['rd']:+.4f} [{r2['ci_lo']:+.4f}, {r2['ci_hi']:+.4f}]",
        f"- low/high pooled 率: {r2['risk_low']:.4f} / {r2['risk_high']:.4f}",
        f"- 曝露 low/high: {r2['n_low']:,} / {r2['n_high']:,}"
        f"（保持率 {diagnostics['r2_retention']:.4f}）",
        "",
        "## 記述 (R3): 交絡の程度",
        "",
        f"- seed 別 Spearman(ΔL, r) 中央値 "
        f"**{diagnostics['spearman_median']:+.4f}**"
        f"（範囲 {diagnostics['spearman_min']:+.4f} 〜 "
        f"{diagnostics['spearman_max']:+.4f}、seed=20）",
    ]
    for outcome in OUTCOMES:
        eff = diagnostics["r3"][outcome]
        lines.append(
            f"- `cell_id` のみで層別した r 三分位の周辺 RD ({outcome}): "
            f"{eff['rd']:+.4f} [{eff['ci_lo']:+.4f}, {eff['ci_hi']:+.4f}]"
            "（判定なし）")
    lines += [
        "",
        "## 副次転帰 (`end_dead_0_05`)",
        "",
    ]
    for row in verdict.itertuples(index=False):
        if row.outcome == "end_dead_0_05":
            lines.append(
                f"- {row.analysis}: RD={row.rd:+.4f} "
                f"[{row.ci_lo:+.4f}, {row.ci_hi:+.4f}]（{row.verdict}; 副次）")
    lines += [
        "",
        "副次解析は主結果の差し替えに使わない（spec §5）。",
        "",
        "## サニティ (spec §9)",
        "",
    ]
    lines += [
        f"- S2（作業6 主解析の再現、bootstrap_seed={FB_BOOT_SEED}）: "
        f"**{'PASS' if diagnostics['s2_pass'] else 'FAIL'}** — "
        f"{diagnostics['s2']['rd']:+.17g} "
        f"[{diagnostics['s2']['ci_lo']:+.17g}, {diagnostics['s2']['ci_hi']:+.17g}]",
        "- S4 は本フレームでは**空虚に真**（リスク集合が `p_hat >= 0.05` なので "
        "`p_hat == 0` の曝露が 0 件）。成立しても情報量は無い旨を sanity.csv に明記した",
        "- 詳細は `sanity.csv`",
        "",
        "## 出所と再現",
        "",
        f"- 実装 commit: `{meta['implementation_git']}`"
        + ("（**dirty**）" if meta["git_dirty"] else ""),
        f"- bootstrap: seed block B={BOOT_N:,}、本解析 RNG seed={BOOT_SEED}、"
        f"S2 のみ {FB_BOOT_SEED}",
        f"- `OMP_NUM_THREADS={meta['omp_num_threads']}`",
        f"- S6（同一入力・同一 commit での二重集計 byte 一致）: "
        f"**{meta['s6_status']}**",
        "",
        "## スコープ (spec §11)",
        "",
        "condA・w100・T=10,000・std・`generator_offset = 20260830` の 20 系列限定。"
        "condB へ外挿しない。**層別後の観察的関連であり、因果介入ではない**。",
        "",
    ]
    (outdir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def run(exposures: Path, outdir: Path) -> None:
    frames_a, diagnostics = build_frames(exposures)
    frames_b, _ = build_frames(exposures)
    mismatches = [name for name in CSV_NAMES
                  if fb._csv_bytes(frames_a[name]) != fb._csv_bytes(frames_b[name])]
    s6_status = "PASS" if not mismatches else f"FAIL: {mismatches}"

    if not diagnostics["s2_pass"]:
        raise SystemExit(
            "S2 FAIL（作業6 主解析を再現できない）。spec §9 により中止する: "
            + _json_compact(diagnostics["s2"]))

    sanity = frames_a["sanity.csv"]
    sanity = pd.concat([sanity, pd.DataFrame([dict(
        check="S6", name="同一 commit・同一入力での二重集計 byte 一致",
        value=s6_status, criterion="全 CSV が byte 一致",
        status="PASS" if not mismatches else "FAIL",
        note="in-process の独立再構築どうしを比較")])], ignore_index=True)
    frames_a["sanity.csv"] = sanity

    outdir.mkdir(parents=True, exist_ok=True)
    for name in CSV_NAMES:
        (outdir / name).write_bytes(fb._csv_bytes(frames_a[name]))

    meta = dict(
        spec=SPEC, spec_commit="3e22b6b", exposures=str(exposures),
        source_commit=SOURCE_COMMIT, implementation_git=git_hash(),
        git_dirty=git_dirty(), bootstrap_seed=BOOT_SEED,
        s2_bootstrap_seed=FB_BOOT_SEED, n_boot=BOOT_N,
        omp_num_threads=os.environ.get("OMP_NUM_THREADS", "unset"),
        s6_status=s6_status, numpy=np.__version__, pandas=pd.__version__,
        diagnostics={k: v for k, v in diagnostics.items() if k != "half_stats"},
        half_stats=diagnostics["half_stats"],
    )
    (outdir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    write_summary(outdir, diagnostics, frames_a["verdict.csv"], meta)

    print(f"[r_swap_0824] R1 {diagnostics['r1_verdict']} "
          f"({diagnostics['r1_role']}) rd={diagnostics['r1']['rd']:+.4f} "
          f"[{diagnostics['r1']['ci_lo']:+.4f}, {diagnostics['r1']['ci_hi']:+.4f}]")
    print(f"[r_swap_0824] R2 {diagnostics['r2_verdict']} "
          f"rd={diagnostics['r2']['rd']:+.4f} "
          f"[{diagnostics['r2']['ci_lo']:+.4f}, {diagnostics['r2']['ci_hi']:+.4f}]")
    print(f"[r_swap_0824] G1={'FIRED' if diagnostics['g1_fired'] else 'ok'} "
          f"G2={'FIRED' if diagnostics['g2_fired'] else 'ok'} "
          f"S6={s6_status}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exposures", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    run(args.exposures, args.outdir)


if __name__ == "__main__":
    main()
