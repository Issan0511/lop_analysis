"""Post-hoc registration of the censoring correction for ``center_oracle_0831``.

Registered by ``specs/spec_center_oracle_0831_followup2.md`` (vault
``可塑性喪失/spec/オラクル中心化_spec_0831_追補2.md``).  This is NOT a
preregistration: the numbers were computed in chat on 2026-08-31 *after* the
run was read.  Every emitted row carries ``registered = 0`` and the provenance
records ``analysis_grade = registered_posthoc_not_preregistered``.

The parent registered ``dbeta_boundary_499`` as a **sum over 5M**.  Under the
oracle arm every unit freezes at a median of 36 boundaries, so the sum ratio
measures surviving-boundary counts rather than descent depth.  This module
re-expresses the same quantity, on the same 499-boundary mask, as a rate per
*surviving* boundary.  No new run, no re-instrumentation: the only inputs are
the committed per-seed logs of both arms.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from .center_oracle_0831 import (
    OracleSanityError,
    estimate,
    shared_draws,
    transition_masks,
)
from .common import ROOT
from .mlp2_phase0 import _sha_file, write_csv
from .width5_gate_b_0901 import _rectangular


EXPERIMENT = "center_oracle_0831_followup2"
SPEC = "specs/spec_center_oracle_0831_followup2.md"
ORACLE_SOURCE = "results/center_oracle_0831"
REFERENCE_SOURCE = "results/mlp2_phase1_0829"
DEFAULT_OUT = "results/center_oracle_0831_followup2"

ORACLE_ARM = "L1w100_Aexact"
REFERENCE_ARM = "L1w100_A1"
SEEDS = tuple(range(10))
TASK_PERIOD = 10_000

# spec §3.  A unit needs this many surviving boundaries in the window to enter
# the aggregation; the exclusion is judged per window, not once globally.
MIN_SURVIVING_BOUNDARIES = 10
# Inherited from the parent run's analysis block (config_used.yaml).
BOOTSTRAP_B = 10_000
BOOTSTRAP_SEED = 20260829

# spec §2.  A mismatch aborts before anything is written.
INPUT_SHA256 = {
    f"{ORACLE_SOURCE}/logs/{ORACLE_ARM}_seed0.npz":
        "0d17868743d6b3761a0a9bd07011efaad719acee8f0cbe0ac1db1cddc56b6e79",
    f"{ORACLE_SOURCE}/logs/{ORACLE_ARM}_seed1.npz":
        "6ea73838b15233030a0e4661df9b14933b3c7523b697fe20b6d0ba235260aeb8",
    f"{ORACLE_SOURCE}/logs/{ORACLE_ARM}_seed2.npz":
        "0ac8809fae3908c026004bd8c66d4e5710247b95374834e3fbf1efcbfa298526",
    f"{ORACLE_SOURCE}/logs/{ORACLE_ARM}_seed3.npz":
        "28245f4a5c9713d3be66e3ba401728664c5a33e06c9d95b197caa1dcd047c793",
    f"{ORACLE_SOURCE}/logs/{ORACLE_ARM}_seed4.npz":
        "9fc582c61bb6339aba04897adaeeb4cc0ea78751a10f72973838edd185fb2737",
    f"{ORACLE_SOURCE}/logs/{ORACLE_ARM}_seed5.npz":
        "ca01b2ef1d7c18fbb9aabb1e5e9f6eab04b24f3143df3537c54ce0597c7f1f65",
    f"{ORACLE_SOURCE}/logs/{ORACLE_ARM}_seed6.npz":
        "93d71cd3ec6d1f9ba275f40cfd1e9f3369c4f3cfcb69f0fed0d60887a8d4b3d5",
    f"{ORACLE_SOURCE}/logs/{ORACLE_ARM}_seed7.npz":
        "daf91c8261e7e3ad285e185acb837faba9350c6ec10e92b2bf3e118c3375fe15",
    f"{ORACLE_SOURCE}/logs/{ORACLE_ARM}_seed8.npz":
        "9b3a56fe410bd7d79f80ab436ef0e6789ac55519dae258526663111bef2cacf9",
    f"{ORACLE_SOURCE}/logs/{ORACLE_ARM}_seed9.npz":
        "a76b67e87e239f35c3594a1f1c41a299c6a7612addb9bb3c26c482cfad957ece",
    f"{REFERENCE_SOURCE}/logs/{REFERENCE_ARM}_seed0.npz":
        "5f8aac6a3b57254fcf85f971b94093d0febc4b8171cfa02f1e6b9efd9235fb93",
    f"{REFERENCE_SOURCE}/logs/{REFERENCE_ARM}_seed1.npz":
        "70c1e00acdd3367939c8ff271ac0af435199ecc761dcb84421e1d723d0a51f14",
    f"{REFERENCE_SOURCE}/logs/{REFERENCE_ARM}_seed2.npz":
        "2d60c146f6d1a9104e003920a412763414926d30739e78eb8c684b5635394cea",
    f"{REFERENCE_SOURCE}/logs/{REFERENCE_ARM}_seed3.npz":
        "625788083e0e0cc0389a38892d102b45487fda221fed4929217713733d0455f9",
    f"{REFERENCE_SOURCE}/logs/{REFERENCE_ARM}_seed4.npz":
        "c0cf3af5ae82e8283fe4fe9391c2c23a5f6e09405af803fd9292f5333b020c47",
    f"{REFERENCE_SOURCE}/logs/{REFERENCE_ARM}_seed5.npz":
        "0796f300481e4b1e0671e68231c2ce1ec8fd63c2b72f86d5da39a686145ca9bf",
    f"{REFERENCE_SOURCE}/logs/{REFERENCE_ARM}_seed6.npz":
        "54e1a535bae743f386da7db7d59f067e97a1941e1a32330478a7c55b04147c74",
    f"{REFERENCE_SOURCE}/logs/{REFERENCE_ARM}_seed7.npz":
        "d7d213c5a4ae51dc9a2c6559c636ea2438006eee572ba3787248de6a1f085729",
    f"{REFERENCE_SOURCE}/logs/{REFERENCE_ARM}_seed8.npz":
        "f0634d357946d19a217692adcd1b2a4f2d760384cfa0dd8df927e95ee229a71c",
    f"{REFERENCE_SOURCE}/logs/{REFERENCE_ARM}_seed9.npz":
        "261d39937425745cf7808b78ad505ef2c9f2692d81d5813871b75776c97c5c7d",
    f"{ORACLE_SOURCE}/verdict.csv":
        "4d9ec5bbf6630d36c64eb533c9961f5769a9eb6e9878f9667521debdf3dec5ed",
}

# spec §5 S_reproduce: the parent's own registered P1 numbers.
REPRODUCE_TOL = 1e-9
# spec §5 S_known: the values written in the vault addendum, cross-check only.
KNOWN_TOL = 5e-4
KNOWN = {
    "K": 36,
    "rate_all_499": {ORACLE_ARM: -0.0509, REFERENCE_ARM: -0.0211},
    "rate_first150": {ORACLE_ARM: -0.0509, REFERENCE_ARM: -0.0234},
    "rate_first_K": {ORACLE_ARM: -0.0489, REFERENCE_ARM: -0.0279},
    "revival_total": {ORACLE_ARM: 0, REFERENCE_ARM: 34_222},
    "revival_within_task": {ORACLE_ARM: 0, REFERENCE_ARM: 25_723},
}

WINDOW_ALL = "all_499"
WINDOW_150 = "first150"
WINDOW_K = "first_K"


def log_path(arm: str, seed: int, root: Path | None = None) -> Path:
    root = Path(root) if root else Path(ROOT)
    source = ORACLE_SOURCE if arm == ORACLE_ARM else REFERENCE_SOURCE
    return root / source / "logs" / f"{arm}_seed{seed}.npz"


def check_inputs(root: Path | None = None) -> dict[str, str]:
    """S_input: every registered input must hash to its pinned value."""
    root = Path(root) if root else Path(ROOT)
    checked: dict[str, str] = {}
    for name, want in INPUT_SHA256.items():
        path = root / name
        if not path.exists():
            raise OracleSanityError(f"registered input missing: {path}")
        got = _sha_file(path)
        if got != want:
            raise OracleSanityError(
                f"S_input FAIL: {name} sha256 {got} != registered {want}")
        checked[name] = got
    return checked


def load_log(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as z:
        return {
            "step": z["step"].astype(int),
            "flip_state": np.asarray(z["flip_state"]),
            "beta": z["layer1_M"].astype(float) + z["layer1_B"].astype(float),
            "p_hat": np.asarray(z["layer1_p_hat"]),
            "unfit": np.asarray(z["unfit"], dtype=float),
        }


def per_unit_rates(step, flip_state, beta, p_hat, k: int | None = None):
    """Mean Δβ per boundary the unit was still alive for (spec §3).

    ``k`` restricts the aggregation to the first ``k`` of the 499 true-switch
    transitions.  Returns ``(rate, n_surv)``, both per unit; ``rate`` is NaN
    where the unit survived no boundary in the window.
    """
    masks = transition_masks(step, flip_state)
    beta = np.asarray(beta, dtype=float)
    if not np.isfinite(beta).all():
        raise OracleSanityError("beta contains non-finite entries")
    dbeta = np.diff(beta, axis=0)
    index = np.flatnonzero(masks["boundary_499"])
    if k is not None:
        if k <= 0:
            raise OracleSanityError(f"window size must be positive, got {k}")
        index = index[:k]
    selected = np.zeros(dbeta.shape[0], dtype=bool)
    selected[index] = True
    alive_before = np.asarray(p_hat)[:-1] > 0
    use = selected[:, None] & alive_before
    n_surv = use.sum(axis=0).astype(int)
    total = np.where(use, dbeta, 0.0).sum(axis=0)
    rate = np.divide(total, n_surv, out=np.full(n_surv.shape, np.nan),
                     where=n_surv > 0)
    return rate, n_surv


def seed_rate(rate: np.ndarray, n_surv: np.ndarray) -> dict[str, Any]:
    """Seed representative = median rate over units that clear the exclusion."""
    keep = n_surv >= MIN_SURVIVING_BOUNDARIES
    if not keep.any():
        raise OracleSanityError("no unit clears the survival exclusion")
    return {
        "value": float(np.median(rate[keep])),
        "n_kept": int(keep.sum()),
        "n_unit": int(keep.size),
        "excluded_frac": float(1.0 - keep.sum() / keep.size),
    }


def boundary_sum(step, flip_state, beta) -> np.ndarray:
    """The parent's registered per-unit sum over the same 499 mask."""
    masks = transition_masks(step, flip_state)
    dbeta = np.diff(np.asarray(beta, dtype=float), axis=0)
    return np.nansum(dbeta[masks["boundary_499"]], axis=0)


def survival_K(n_surv_by_seed: list[np.ndarray]) -> tuple[int, list[float]]:
    """K = round(median over seeds of the per-seed median survival) (spec §3).

    Ties round half up, as the spec's 四捨五入 says; ``round`` would take
    36.5 down to 36.
    """
    per_seed = [float(np.median(n)) for n in n_surv_by_seed]
    return int(np.floor(float(np.median(per_seed)) + 0.5)), per_seed


def revival_counts(step, flip_state, p_hat) -> dict[str, int]:
    """Dead -> alive transitions, in total and away from task boundaries."""
    masks = transition_masks(step, flip_state)
    dead = np.asarray(p_hat) == 0
    revived = dead[:-1] & ~dead[1:]
    return {
        "total": int(revived.sum()),
        "within_task": int(revived[masks["internal_4500"]].sum()),
        "at_boundary": int(revived[masks["boundary_499"]].sum()),
    }


def extinction_task(step, p_hat, task_period: int = TASK_PERIOD) -> int | None:
    """First task index at which every unit is strictly dead, or None."""
    all_dead = (np.asarray(p_hat) == 0).all(axis=1)
    if not all_dead.any():
        return None
    return int(np.asarray(step, dtype=int)[int(np.argmax(all_dead))]
               // task_period)


def label_ratio(est: dict[str, Any]) -> str:
    """spec §4 P1'-a: the CI must clear 1 in one direction or the other."""
    if est["ci_lo"] > 1.0:
        return "LAG_IS_PROTECTIVE"
    if est["ci_hi"] < 1.0:
        return "LAG_IS_HARMFUL"
    return "RATE_INCONCLUSIVE"


def window_dependence(label_a: str, label_b: str) -> str | None:
    """spec §4 P1'-b: only an outright sign split is WINDOW_DEPENDENT."""
    split = {"LAG_IS_PROTECTIVE", "LAG_IS_HARMFUL"}
    return "WINDOW_DEPENDENT" if {label_a, label_b} == split else None


def collect(root: Path | None = None) -> dict[str, Any]:
    """Read both arms once and derive every registered quantity."""
    logs = {arm: {seed: load_log(log_path(arm, seed, root)) for seed in SEEDS}
            for arm in (ORACLE_ARM, REFERENCE_ARM)}

    mask_rows: list[dict[str, Any]] = []
    for arm, per_seed in logs.items():
        for seed, log in per_seed.items():
            masks = transition_masks(log["step"], log["flip_state"])
            mask_rows.append({
                "arm": arm, "seed": seed,
                "n_flip_changed": int(masks["changed"].sum()),
                "n_boundary_499": int(masks["boundary_499"].sum()),
                "n_internal_4500": int(masks["internal_4500"].sum()),
                "n_startup_0to1000": int(masks["startup_0to1000"].sum()),
            })
    s_mask = {
        "name": "S_mask", "gating": True,
        "pass_": all(r["n_flip_changed"] == 499 and r["n_boundary_499"] == 499
                     and r["n_internal_4500"] == 4500
                     and r["n_startup_0to1000"] == 1 for r in mask_rows),
        "rows": mask_rows,
    }
    if not s_mask["pass_"]:
        raise OracleSanityError(f"S_mask FAIL: {mask_rows}")

    full = {arm: {seed: per_unit_rates(log["step"], log["flip_state"],
                                       log["beta"], log["p_hat"])
                  for seed, log in per_seed.items()}
            for arm, per_seed in logs.items()}
    K, per_seed_survival = survival_K(
        [full[ORACLE_ARM][seed][1] for seed in SEEDS])

    windows = {WINDOW_ALL: None, WINDOW_150: 150, WINDOW_K: K}
    unit_rows: list[dict[str, Any]] = []
    seed_values: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for window, k in windows.items():
        for arm in (ORACLE_ARM, REFERENCE_ARM):
            entries = []
            for seed in SEEDS:
                log = logs[arm][seed]
                rate, n_surv = (full[arm][seed] if k is None else
                                per_unit_rates(log["step"], log["flip_state"],
                                               log["beta"], log["p_hat"], k))
                entries.append(seed_rate(rate, n_surv))
                for unit in range(rate.size):
                    unit_rows.append({
                        "arm": arm, "seed": seed, "unit": unit,
                        "window": window,
                        "window_boundaries": 499 if k is None else k,
                        "n_surviving_boundaries": int(n_surv[unit]),
                        "rate": float(rate[unit]),
                        "kept": int(n_surv[unit] >= MIN_SURVIVING_BOUNDARIES),
                    })
            seed_values[(window, arm)] = entries

    sums = {arm: np.array([float(np.median(boundary_sum(
        logs[arm][seed]["step"], logs[arm][seed]["flip_state"],
        logs[arm][seed]["beta"]))) for seed in SEEDS])
        for arm in (ORACLE_ARM, REFERENCE_ARM)}

    revivals = {arm: {"total": 0, "within_task": 0, "at_boundary": 0}
                for arm in (ORACLE_ARM, REFERENCE_ARM)}
    for arm, per_seed in logs.items():
        for log in per_seed.values():
            counts = revival_counts(log["step"], log["flip_state"],
                                    log["p_hat"])
            for key, value in counts.items():
                revivals[arm][key] += value

    extinction = {arm: [extinction_task(logs[arm][seed]["step"],
                                        logs[arm][seed]["p_hat"])
                        for seed in SEEDS]
                  for arm in (ORACLE_ARM, REFERENCE_ARM)}
    unfit_final = {arm: [float(logs[arm][seed]["unfit"][-1]) for seed in SEEDS]
                   for arm in (ORACLE_ARM, REFERENCE_ARM)}

    return {
        "K": K, "per_seed_survival": per_seed_survival, "windows": windows,
        "seed_values": seed_values, "unit_rows": unit_rows,
        "boundary_sums": sums, "revivals": revivals,
        "extinction": extinction, "unfit_final": unfit_final,
        "s_mask": s_mask,
    }


def check_reproduce(sums: dict[str, np.ndarray], source: Path) -> dict:
    """S_reproduce: the parent's registered P1 must come back off these logs."""
    draws = shared_draws(BOOTSTRAP_B, BOOTSTRAP_SEED)
    recomputed = {
        "Aexact_dbeta_boundary_499": estimate(sums[ORACLE_ARM], draws),
        "A1_dbeta_boundary_499": estimate(sums[REFERENCE_ARM], draws),
        "R_abs_boundary_ratio_499": estimate(
            np.abs(sums[ORACLE_ARM]) / np.abs(sums[REFERENCE_ARM]), draws),
    }
    with (source / "verdict.csv").open(newline="", encoding="utf-8") as fh:
        recorded = {row["metric"]: row for row in csv.DictReader(fh)}
    rows, ok = [], True
    for metric, est in recomputed.items():
        want = float(recorded[metric]["point"])
        delta = abs(est["point"] - want)
        match = delta <= REPRODUCE_TOL
        ok = ok and match
        rows.append({"metric": metric, "recorded": want,
                     "recomputed": est["point"], "abs_delta": delta,
                     "match": bool(match)})
    return {"name": "S_reproduce", "gating": True, "pass_": ok, "rows": rows,
            "tolerance": REPRODUCE_TOL, "recomputed": recomputed}


def check_known(rates: dict[tuple[str, str], dict], K: int,
                revivals: dict) -> dict:
    """S_known: cross-check against the vault addendum, not a gate."""
    rows = [{"quantity": "K", "vault": KNOWN["K"], "recomputed": K,
             "abs_delta": abs(K - KNOWN["K"]),
             "match": bool(K == KNOWN["K"])}]
    for window, key in ((WINDOW_ALL, "rate_all_499"),
                        (WINDOW_150, "rate_first150"),
                        (WINDOW_K, "rate_first_K")):
        for arm in (ORACLE_ARM, REFERENCE_ARM):
            got = rates[(window, arm)]["point"]
            want = KNOWN[key][arm]
            rows.append({"quantity": f"{key}:{arm}", "vault": want,
                         "recomputed": got, "abs_delta": abs(got - want),
                         "match": bool(abs(got - want) <= KNOWN_TOL)})
    for key in ("revival_total", "revival_within_task"):
        field = "total" if key == "revival_total" else "within_task"
        for arm in (ORACLE_ARM, REFERENCE_ARM):
            got = revivals[arm][field]
            want = KNOWN[key][arm]
            rows.append({"quantity": f"{key}:{arm}", "vault": want,
                         "recomputed": got, "abs_delta": abs(got - want),
                         "match": bool(got == want)})
    return {"name": "S_known", "gating": False, "tolerance": KNOWN_TOL,
            "note": "cross-check against the vault addendum; not a gate",
            "pass_": all(r["match"] for r in rows), "rows": rows}


def _fmt(value: float, digits: int = 4) -> str:
    return f"{value:+.{digits}f}"


def render_summary(data: dict, rates: dict, ratios: dict, labels: dict,
                   sanity: dict) -> str:
    K = data["K"]
    extinct = [t for t in data["extinction"][ORACLE_ARM] if t is not None]
    lines = [
        f"# {EXPERIMENT} summary", "",
        "> **事後解析の登録。事前登録ではない。** vault `オラクル中心化_spec_0831_追補2` "
        "§2 の数値は本走の結果を見た後にチャットで算出済みである（spec §0）。全行 "
        "`registered = 0`、`analysis_grade = "
        "registered_posthoc_not_preregistered`。",
        "",
        "入力は committed の per-seed ログ 20 本と親 `verdict.csv` のみ。"
        "新しい走・再計装はしていない。マスクは追補1 の 499 点をそのまま使う。",
        "",
        f"- **P1'-a（主）**: **{labels['P1a']}**",
        f"- **P1'-b**: **{labels['P1b']}**"
        + (f" / **{labels['window']}**" if labels["window"] else
           "（P1'-a と同符号なので `WINDOW_DEPENDENT` は出ない）"),
        f"- **P1'-c**: **{labels['P1c']}**",
        "",
        "## P1'-a / P1'-b 生存中 1 境界あたりの Δβ", "",
        "ユニットごとに、**直前に alive だった真の切替遷移のみ**を平均する。"
        f"除外は当該窓の生存境界が {MIN_SURVIVING_BOUNDARIES} 本未満のユニット。"
        f"$K$ = {K}（Aexact の生存境界数: per-seed 中央値の中央値 "
        f"{np.median(data['per_seed_survival']):.1f} を四捨五入）。", "",
        "| 集計範囲 | `L1w100_Aexact` | `L1w100_A1` | $R'$ | ラベル |",
        "|---|---:|---:|---:|---|",
    ]
    window_title = {WINDOW_ALL: "全 499 境界", WINDOW_150: "最初の 150 境界",
                    WINDOW_K: f"最初の {K} 境界（生存本数を揃えた対照）"}
    for window in (WINDOW_ALL, WINDOW_150, WINDOW_K):
        exact, a1 = rates[(window, ORACLE_ARM)], rates[(window, REFERENCE_ARM)]
        ratio = ratios[window]
        label = {WINDOW_ALL: labels["P1a"], WINDOW_K: labels["P1b"]}.get(
            window, "REPORT_ONLY")
        lines.append(
            f"| {window_title[window]} | **{_fmt(exact['point'])}** "
            f"[{_fmt(exact['ci_lo'])}, {_fmt(exact['ci_hi'])}] | "
            f"{_fmt(a1['point'])} [{_fmt(a1['ci_lo'])}, {_fmt(a1['ci_hi'])}] | "
            f"**{ratio['point']:.3f}** [{ratio['ci_lo']:.3f}, "
            f"{ratio['ci_hi']:.3f}] | {label} |")
    lines += [
        "",
        "**最初の 150 境界は REPORT_ONLY**（窓依存の目視用）。"
        "$R' = |{\\rm rate}_{\\rm Aexact}| / |{\\rm rate}_{\\rm A1}|$ を seed ごとに "
        "作り、seed クラスタ bootstrap（$B$=10,000・percentile・seed 20260829）で "
        "CI を出す。**生存本数を揃えても Aexact のほうが速く落ちる。**",
        "",
        "## P1'-c 復活件数", "",
        "| 腕 | 総数 | タスク内 | 境界 |",
        "|---|---:|---:|---:|",
    ]
    for arm in (ORACLE_ARM, REFERENCE_ARM):
        r = data["revivals"][arm]
        lines.append(f"| `{arm}` | {r['total']:,} | {r['within_task']:,} | "
                     f"{r['at_boundary']:,} |")
    lines += [
        "",
        "10 seed 合計。`p_hat == 0` の記録点から次の記録点で `p_hat > 0` に戻った"
        "遷移を数える。",
        "",
        "## P1'-d 全滅と unfit（REPORT_ONLY）", "",
        f"- `{ORACLE_ARM}` 全滅到達 task: "
        f"**{min(extinct)}–{max(extinct)}（中央値 {np.median(extinct):.0f}）** "
        f"/ per-seed {data['extinction'][ORACLE_ARM]}",
        f"- `{REFERENCE_ARM}` は 5M までに全滅した seed なし",
        f"- `unfit` 最終値の中央値: `{ORACLE_ARM}` "
        f"{np.median(data['unfit_final'][ORACLE_ARM]):.4f} / `{REFERENCE_ARM}` "
        f"{np.median(data['unfit_final'][REFERENCE_ARM]):.4f}",
        "",
        "> **vault §1 の errata**: 全滅到達の中央値は「≈ 210」ではなく "
        f"**{np.median(extinct):.0f}**（range は 154–454 で正しい）。210 は seed 7 "
        "の値である。",
        "",
        "## Sanity", "",
    ]
    for key in ("S_input", "S_mask", "S_reproduce", "S_known"):
        rec = sanity[key]
        gate = "gate" if rec.get("gating") else "照合のみ"
        lines.append(
            f"- **{key}**: {'PASS' if rec['pass_'] else 'FAIL'} ({gate})")
    lines += [
        "",
        "S_reproduce は同じログから親の P1（`dbeta_boundary_499` の seed 中央値と "
        "$R_{499}$）を再計算して `results/center_oracle_0831/verdict.csv` と "
        f"{REPRODUCE_TOL:g} 以内で一致することを確認する。**rate は同じマスクの"
        "厳密な細分である。**",
        "",
        "## 引用上の注意", "",
        "- **P1 の登録判定 `BOTH_CONTRIBUTE` を上書きしない。** 併記の形は「総和では "
        "`BOTH_CONTRIBUTE`（$R_{499}=0.5167$）だが、これは生存境界数 "
        f"{data['K']} 対 387 の打ち切りを含む。率に直すと符号が反転する（P1'）」",
        "- **`R_{499}=0.5167` を単独で引かない。** 必ず率の行を併記する",
        "- 可識別性の交絡は生きている。書いてよいのは「**EMA 遅れ窓を外すと悪化する**」"
        "までで、「$\\mu$ が保護的である」ではない",
        "- 「centering を改善すれば LoP を防げる」と書かない",
        "- 0/10・10/10 は「観測しなかった」の強さ。`strict_dead = 1.0` が 10 seed で"
        "揃ったことは「必ず全滅する」ではない",
        "",
    ]
    return "\n".join(lines)


def run_analysis(root: Path | None = None,
                 outdir: Path | None = None) -> dict[str, Any]:
    started = time.time()
    root = Path(root) if root else Path(ROOT)
    outdir = Path(outdir) if outdir else root / DEFAULT_OUT
    spec_path = root / SPEC
    if not spec_path.exists():
        raise OracleSanityError(f"frozen repo spec missing: {spec_path}")

    input_sha = check_inputs(root)
    data = collect(root)

    draws = shared_draws(BOOTSTRAP_B, BOOTSTRAP_SEED)
    rates, ratios = {}, {}
    for window in data["windows"]:
        for arm in (ORACLE_ARM, REFERENCE_ARM):
            entries = data["seed_values"][(window, arm)]
            rates[(window, arm)] = estimate(
                [e["value"] for e in entries], draws)
        exact = np.array([e["value"]
                          for e in data["seed_values"][(window, ORACLE_ARM)]])
        a1 = np.array([e["value"]
                       for e in data["seed_values"][(window, REFERENCE_ARM)]])
        ratios[window] = estimate(np.abs(exact) / np.abs(a1), draws)

    s_reproduce = check_reproduce(data["boundary_sums"],
                                  root / ORACLE_SOURCE)
    if not s_reproduce["pass_"]:
        raise OracleSanityError(
            f"S_reproduce FAIL: {s_reproduce['rows']}")
    s_known = check_known(rates, data["K"], data["revivals"])
    sanity = {
        "S_input": {"name": "S_input", "gating": True, "pass_": True,
                    "sha256": input_sha},
        "S_mask": data["s_mask"],
        "S_reproduce": s_reproduce,
        "S_known": s_known,
    }

    labels = {
        "P1a": label_ratio(ratios[WINDOW_ALL]),
        "P1b": label_ratio(ratios[WINDOW_K]),
        "P1c": ("ABSORPTION_EXACT_UNDER_ORACLE"
                if data["revivals"][ORACLE_ARM]["total"] == 0
                else "ABSORPTION_NOT_EXACT"),
    }
    labels["window"] = window_dependence(labels["P1a"], labels["P1b"])

    outdir.mkdir(parents=True, exist_ok=True)
    verdict_rows: list[dict[str, Any]] = []
    endpoint_of = {WINDOW_ALL: "P1prime_a", WINDOW_K: "P1prime_b",
                   WINDOW_150: "P1prime_REPORT_ONLY"}
    for window in (WINDOW_ALL, WINDOW_150, WINDOW_K):
        entries_exact = data["seed_values"][(window, ORACLE_ARM)]
        entries_a1 = data["seed_values"][(window, REFERENCE_ARM)]
        label = {WINDOW_ALL: labels["P1a"],
                 WINDOW_K: labels["P1b"]}.get(window, "REPORT_ONLY")
        ratio = ratios[window]
        verdict_rows.append({
            "endpoint": endpoint_of[window], "registered": 0,
            "metric": f"R_rate_ratio_{window}", "arm": "", "window": window,
            "window_boundaries": 499 if data["windows"][window] is None
            else data["windows"][window],
            "point": ratio["point"], "ci_lo": ratio["ci_lo"],
            "ci_hi": ratio["ci_hi"], "n_seed": ratio["n_seed"],
            "label": label,
            "basis": "paired seed ratio of |median unit rate|; "
                     "alive-before-transition boundaries only",
        })
        for arm, entries in ((ORACLE_ARM, entries_exact),
                             (REFERENCE_ARM, entries_a1)):
            est = rates[(window, arm)]
            verdict_rows.append({
                "endpoint": endpoint_of[window], "registered": 0,
                "metric": f"rate_per_surviving_boundary_{window}", "arm": arm,
                "window": window,
                "window_boundaries": 499 if data["windows"][window] is None
                else data["windows"][window],
                "point": est["point"], "ci_lo": est["ci_lo"],
                "ci_hi": est["ci_hi"], "n_seed": est["n_seed"],
                "label": "REPORT_ONLY" if arm == REFERENCE_ARM else label,
                "basis": "seed median over units with >= "
                         f"{MIN_SURVIVING_BOUNDARIES} surviving boundaries; "
                         f"excluded {np.mean([e['excluded_frac'] for e in entries]):.3f}",
            })
    for arm in (ORACLE_ARM, REFERENCE_ARM):
        counts = data["revivals"][arm]
        for field, metric in (("total", "revival_total"),
                              ("within_task", "revival_within_task"),
                              ("at_boundary", "revival_at_boundary")):
            verdict_rows.append({
                "endpoint": "P1prime_c", "registered": 0, "metric": metric,
                "arm": arm, "window": WINDOW_ALL, "window_boundaries": 499,
                "point": counts[field], "ci_lo": "", "ci_hi": "",
                "n_seed": len(SEEDS),
                "label": labels["P1c"] if arm == ORACLE_ARM else "REFERENCE",
                "basis": "dead -> alive record transitions, summed over seeds",
            })
    extinct = [t for t in data["extinction"][ORACLE_ARM] if t is not None]
    verdict_rows.append({
        "endpoint": "P1prime_d", "registered": 0,
        "metric": "extinction_task_median", "arm": ORACLE_ARM,
        "window": WINDOW_ALL, "window_boundaries": 499,
        "point": float(np.median(extinct)), "ci_lo": float(min(extinct)),
        "ci_hi": float(max(extinct)), "n_seed": len(extinct),
        "label": "REPORT_ONLY",
        "basis": "first task index with every unit strictly dead; "
                 "ci columns carry the min/max, not a bootstrap",
    })
    for arm in (ORACLE_ARM, REFERENCE_ARM):
        verdict_rows.append({
            "endpoint": "P1prime_d", "registered": 0,
            "metric": "unfit_final_median", "arm": arm, "window": WINDOW_ALL,
            "window_boundaries": 499,
            "point": float(np.median(data["unfit_final"][arm])),
            "ci_lo": "", "ci_hi": "", "n_seed": len(SEEDS),
            "label": "REPORT_ONLY", "basis": "seed median of the final record",
        })
    write_csv(outdir / "verdict.csv", _rectangular(verdict_rows))
    write_csv(outdir / "unit_rates.csv", _rectangular(data["unit_rows"]))
    write_csv(outdir / "sanity_reproduce.csv", _rectangular(s_reproduce["rows"]))
    write_csv(outdir / "sanity_known.csv", _rectangular(s_known["rows"]))
    (outdir / "summary.md").write_text(
        render_summary(data, rates, ratios, labels, sanity), encoding="utf-8")

    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--short"], cwd=ROOT, text=True).splitlines()
    except (OSError, subprocess.CalledProcessError):
        git_hash, dirty = None, []
    provenance = {
        "experiment": EXPERIMENT,
        "analysis_grade": "registered_posthoc_not_preregistered",
        "grade_note": ("values computed in chat 2026-08-31 after the parent "
                       "run was read; see spec §0 and vault "
                       "オラクル中心化_spec_0831_追補2"),
        "created": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "command": sys.argv, "elapsed_sec": round(time.time() - started, 3),
        "cwd": os.getcwd(), "python": sys.version,
        "platform": platform.platform(), "git_hash": git_hash,
        "git_dirty": dirty, "spec": str(spec_path),
        "spec_sha256": _sha_file(spec_path),
        "sources": [ORACLE_SOURCE, REFERENCE_SOURCE],
        "input_sha256": input_sha,
        "K": data["K"], "per_seed_survival_median": data["per_seed_survival"],
        "min_surviving_boundaries": MIN_SURVIVING_BOUNDARIES,
        "bootstrap": {"B": BOOTSTRAP_B, "seed": BOOTSTRAP_SEED,
                      "kind": "seed cluster percentile"},
        "labels": labels, "sanity": sanity,
        "output_sha256": {name: _sha_file(outdir / name)
                          for name in ("verdict.csv", "unit_rates.csv",
                                       "sanity_reproduce.csv",
                                       "sanity_known.csv", "summary.md")},
    }
    (outdir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")
    print(f"ALL DONE -> {outdir}", flush=True)
    return {"K": data["K"], "rates": rates, "ratios": ratios,
            "labels": labels, "revivals": data["revivals"],
            "extinction": data["extinction"], "sanity": sanity}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=None)
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args()
    run_analysis(args.root, args.outdir)


if __name__ == "__main__":
    main()
