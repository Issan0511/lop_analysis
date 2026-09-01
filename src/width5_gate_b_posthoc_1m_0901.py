"""Post-hoc registration of the 1M-window sign test for width5_gate_b_0901.

Registered by ``specs/spec_width5_gate_b_0901_posthoc_1m.md``.  This is NOT a
preregistration: the numbers were computed in chat on 2026-09-01 before the
spec was frozen (vault ``原典条件照合_0901`` §5).  Every emitted row carries
``registered = 0`` and the provenance records
``analysis_grade = registered_posthoc_not_preregistered``.

The only inputs are two committed files from the parent run.  The parent's raw
per-seed logs are not in the repository, so restricting the input to
``verdict.csv`` keeps this analysis fully recomputable from a fresh clone.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

from .common import ROOT
from .dose_const_5m import clopper_pearson
from .mlp2_phase0 import _sha_file, write_csv
from .width5_gate_b_0901 import _rectangular, classify_seed_sign


EXPERIMENT = "width5_gate_b_0901_posthoc_1m"
SPEC = "specs/spec_width5_gate_b_0901_posthoc_1m.md"
SOURCE = "results/width5_gate_b_0901"
DEFAULT_OUT = "results/width5_gate_b_0901/posthoc_1m"

# Pinned by the spec §2.  A mismatch aborts before anything is written.
INPUT_SHA256 = {
    "verdict.csv":
        "a9a89b32e4cf6dd2c46a8d65c282d31b61363957f99f873453463f1e9e0a09d3",
    "provenance.json":
        "e9c845e66b619b7246544ee49e1776f4349e1b9ab0b784e5b8195d8e345c8bc0",
}

# Frozen in the parent spec §5; S_reproduce below re-derives the recorded 5M
# labels with these values, so a drift here fails loudly instead of silently.
ALPHA = 0.05
TIGHT_BAND = (0.20, 0.80)
COMPARISON_ARMS = ("R5", "LR5", "E5")
BASELINE = "LIN5"
ARM_ORDER = ("R5", "LR5", "E5", "LIN5", "R100", "LR100", "E100", "LIN100")
WINDOWS = {"1m": "task 91-100", "5m": "task 491-500"}

# unfit = residual_var / signal_var, so 1.0 is the degeneracy point where the
# residual carries the whole signal variance.  0.999 is a finite-precision
# margin on that point, not a calibrated onset threshold.
COLLAPSE_THRESHOLD = 0.999

# handoff W1-3: the values computed in chat, used as a cross-check only.
KNOWN_1M_K = {"R5": 9, "LR5": 0, "E5": 0}


class SanityError(RuntimeError):
    """Raised when a registered pre-execution check fails."""


def _finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


def load_source(source: Path) -> tuple[dict, dict]:
    """Verify input hashes and read the per-seed unfit values."""
    checked = {}
    for name, want in INPUT_SHA256.items():
        path = source / name
        if not path.exists():
            raise SanityError(f"registered input missing: {path}")
        got = _sha_file(path)
        if got != want:
            raise SanityError(
                f"S_input FAIL: {name} sha256 {got} != registered {want}")
        checked[name] = got

    import csv
    with (source / "verdict.csv").open(newline="", encoding="utf-8") as fh:
        rows = {row["arm"]: row for row in csv.DictReader(fh)}
    values = {
        arm: {w: [float(v) for v in
                  ast.literal_eval(rows[arm][f"U_{w}_seed_values"])]
              for w in WINDOWS}
        for arm in ARM_ORDER
    }
    recorded = {
        arm: dict(k=rows[arm]["k_above_LIN5_5m"],
                  status=rows[arm]["sign_status"])
        for arm in COMPARISON_ARMS
    }
    return values, dict(input_sha256=checked, recorded_5m=recorded)


def sign_test(values: dict, arm: str, window: str) -> dict:
    """Apply the parent spec's G0 rule to one arm in one window."""
    a = values[arm][window]
    base = values[BASELINE][window]
    valid = [i for i in range(len(a))
             if _finite(a[i]) and _finite(base[i])]
    k = sum(1 for i in valid if a[i] > base[i])
    record = classify_seed_sign(arm, k, len(valid), alpha=ALPHA,
                                tight_band=TIGHT_BAND)
    record.update(
        registered=0, window=window, window_tasks=WINDOWS[window],
        ties=sum(1 for i in valid if a[i] == base[i]),
        excluded_seed_indices=[i for i in range(len(a)) if i not in valid],
        above_seed_indices=[i for i in valid if a[i] > base[i]],
    )
    return record


def collapse_counts(values: dict) -> list[dict]:
    """Threshold-free separator: how many seeds reach the degeneracy point."""
    rows = []
    for arm in ARM_ORDER:
        late = values[arm]["5m"]
        valid = [v for v in late if _finite(v)]
        k = sum(1 for v in valid if v >= COLLAPSE_THRESHOLD)
        lo, hi = clopper_pearson(k, len(valid), ALPHA)
        rows.append(dict(
            metric="collapse_count_5m", registered=0, arm=arm,
            threshold=COLLAPSE_THRESHOLD, k=k, n=len(valid),
            rate=k / len(valid), cp95_lo=lo, cp95_hi=hi,
            collapse_seed_indices=[i for i, v in enumerate(late)
                                   if _finite(v) and v >= COLLAPSE_THRESHOLD]))
    return rows


def baseline_drift(values: dict) -> dict:
    """Does LIN5 itself get worse between the 1M and the terminal window?"""
    one, five = values[BASELINE]["1m"], values[BASELINE]["5m"]
    pairs = [(a, b) for a, b in zip(one, five) if _finite(a) and _finite(b)]
    diffs = [b - a for a, b in pairs]
    worse = sum(1 for d in diffs if d > 0)
    return dict(
        metric="LIN5_1m_to_5m", registered=0, arm=BASELINE, n=len(pairs),
        within_seed_diff_median=statistics.median(diffs),
        worse_seed_count=worse,
        arm_median_1m=statistics.median([a for a, _ in pairs]),
        arm_median_5m=statistics.median([b for _, b in pairs]),
        arm_median_diff=(statistics.median([b for _, b in pairs])
                         - statistics.median([a for a, _ in pairs])),
        degraded=bool(worse > len(pairs) / 2
                      and statistics.median(diffs) > 0))


def bimodality(values: dict, arm: str = "R5") -> list[dict]:
    """Split the terminal sign count by whether the seed fully collapsed."""
    late, base = values[arm]["5m"], values[BASELINE]["5m"]
    rows = []
    for label, keep in (("collapsed", True), ("not_collapsed", False)):
        idx = [i for i in range(len(late))
               if _finite(late[i]) and _finite(base[i])
               and (late[i] >= COLLAPSE_THRESHOLD) == keep]
        rows.append(dict(
            metric="R5_bimodality_5m", registered=0, arm=arm, group=label,
            threshold=COLLAPSE_THRESHOLD, n=len(idx),
            above_LIN5=sum(1 for i in idx if late[i] > base[i]),
            seed_indices=idx))
    return rows


def check_reproduce(signs: dict, recorded: dict) -> dict:
    """S-reproduce: the 5M window must reproduce the parent's own record."""
    rows, ok = [], True
    for arm in COMPARISON_ARMS:
        got, want = signs[(arm, "5m")], recorded[arm]
        match = (str(got["k"]) == str(want["k"])
                 and got["status"] == want["status"])
        ok = ok and match
        rows.append(dict(arm=arm, recorded_k=want["k"],
                         recomputed_k=got["k"], recorded_status=want["status"],
                         recomputed_status=got["status"], match=bool(match)))
    return dict(name="S_reproduce", pass_=ok, gating=True, rows=rows)


def check_known(signs: dict) -> dict:
    """S-known: cross-check against the values computed in chat (handoff W1-3)."""
    rows, ok = [], True
    for arm, want in KNOWN_1M_K.items():
        got = signs[(arm, "1m")]["k"]
        ok = ok and got == want
        rows.append(dict(arm=arm, handoff_k=want, recomputed_k=got,
                         match=bool(got == want)))
    return dict(name="S_known", pass_=ok, gating=False,
                note="cross-check only; not a gate", rows=rows)


def _fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def render_summary(signs: dict, collapses: list[dict], drift: dict,
                   bimodal: list[dict], sanity: dict) -> str:
    lines = [
        f"# {EXPERIMENT} summary", "",
        "> **1M 窓は事後登録。事前登録は末尾窓のみ。** 本解析の数値は spec 凍結前に "
        "チャットで算出済みであり、事前登録ではない（spec §0）。全行 "
        "`registered = 0`、`analysis_grade = "
        "registered_posthoc_not_preregistered`。",
        "",
        "入力は committed の `results/width5_gate_b_0901/"
        "{verdict.csv,provenance.json}` のみ。新しい走・再計装はしていない。",
        "",
        "## P1 1M 窓の符号検定（事後）", "",
        "$k_A$ = 当該窓で腕 $A$ の `unfit` が同 seed の `LIN5` より大きい seed 数。"
        "判定規則は親 spec §5 G0 と同一で、窓だけ付け替えた。"
        "`LIN5` は leaky($a$=1.0)・隠れ 5 ユニットの実装である。", "",
        "| 腕 | 窓 | k | n | 除外 | 同値 | CP95 | ラベル |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for window in WINDOWS:
        for arm in COMPARISON_ARMS:
            r = signs[(arm, window)]
            lines.append(
                f"| `{arm}` | {r['window_tasks']} | **{r['k']}** | {r['n']} | "
                f"{len(r['excluded_seed_indices'])} | {r['ties']} | "
                f"[{_fmt(r['cp95_lo'])}, {_fmt(r['cp95_hi'])}] | "
                f"**{r['status']}** |")
    lines += [
        "",
        "**主判定は出していない。** 親 spec の `PHENOMENON3_NOT_REPRODUCED` は "
        "末尾窓で確定済みであり、事後の窓で上書きしない（spec §3 P1）。",
        "",
        "## P2-1 `LIN5` は 5M で劣化しているか（事後）", "",
        f"- seed 内差 `U_5m − U_1m` の中央値: **{drift['within_seed_diff_median']:+.4f}**",
        f"- 悪化した seed: **{drift['worse_seed_count']}/{drift['n']}**",
        f"- 腕中央値: 1M {_fmt(drift['arm_median_1m'])} → 5M "
        f"{_fmt(drift['arm_median_5m'])}（差 {drift['arm_median_diff']:+.4f}）",
        f"- 判定: **{'劣化していると読める' if drift['degraded'] else '劣化しているとは読めない'}**",
        "",
        "## P2-2 `R5` の二峰性（事後）", "",
        "| 群 | n | `LIN5` より上 |",
        "|---|---:|---:|",
    ]
    for row in bimodal:
        lines.append(f"| {row['group']} | {row['n']} | "
                     f"**{row['above_LIN5']}**/{row['n']} |")
    lines += [
        "",
        f"完全崩壊は `unfit` >= {COLLAPSE_THRESHOLD}。"
        "**符号検定は「1.0 対 0.51」と「0.53 対 0.49」を同じ 1 票にする。**",
        "",
        "## P2-3 完全崩壊カウント（事後・8 腕）", "",
        f"`unfit` >= {COLLAPSE_THRESHOLD}（残差分散が信号分散に等しい縮退点。"
        "較正の要る閾値ではない）。**これを「LoP の発症率」と呼ばない。**", "",
        "| 腕 | k/n | CP95 |",
        "|---|---:|---|",
    ]
    for row in collapses:
        lines.append(f"| `{row['arm']}` | {row['k']}/{row['n']} | "
                     f"[{_fmt(row['cp95_lo'])}, {_fmt(row['cp95_hi'])}] |")
    lines += ["", "## Sanity", ""]
    for key in ("S_input", "S_reproduce", "S_known"):
        rec = sanity[key]
        gate = "gate" if rec.get("gating") else "照合のみ"
        lines.append(f"- **{key}**: {'PASS' if rec['pass_'] else 'FAIL'} ({gate})")
    lines += [
        "", "## 引用上の注意", "",
        "- **1M 窓は事後登録。** 事前登録は末尾窓のみ",
        "- **1M 窓のラベルで `PHENOMENON3_NOT_REPRODUCED` を上書きしない**",
        "- **完全崩壊カウントを「LoP の発症率」と呼ばない**",
        "- **`LIN5` は原典の Linear ベースラインではない。** 原典は隠れ層ゼロの"
        "単層線形回帰で、当方は leaky($a$=1.0)・隠れ 5。差は `LIN0` 腕が入るまで"
        "閉じない",
        "- **1M 窓を「原典の図と同じ条件」と書かない。** 揃うのはホライズンだけ",
        "- スコープは condA・$T=10^4$・batch 1・lr 0.01・seed 20",
        "",
    ]
    return "\n".join(lines)


def run_analysis(source: Path | None = None,
                 outdir: Path | None = None) -> dict:
    started = time.time()
    source = Path(source) if source else Path(ROOT) / SOURCE
    outdir = Path(outdir) if outdir else Path(ROOT) / DEFAULT_OUT
    spec_path = Path(ROOT) / SPEC
    if not spec_path.exists():
        raise SanityError(f"frozen repo spec missing: {spec_path}")

    values, meta = load_source(source)
    signs = {(arm, window): sign_test(values, arm, window)
             for window in WINDOWS for arm in COMPARISON_ARMS}

    s_input = dict(name="S_input", pass_=True, gating=True,
                   sha256=meta["input_sha256"])
    s_reproduce = check_reproduce(signs, meta["recorded_5m"])
    if not s_reproduce["pass_"]:
        raise SanityError(
            "S_reproduce FAIL: recomputed 5M labels differ from the parent "
            f"record: {s_reproduce['rows']}")
    s_known = check_known(signs)
    sanity = dict(S_input=s_input, S_reproduce=s_reproduce, S_known=s_known)

    collapses = collapse_counts(values)
    drift = baseline_drift(values)
    bimodal = bimodality(values)

    outdir.mkdir(parents=True, exist_ok=True)
    verdict_rows = []
    for window in WINDOWS:
        for arm in COMPARISON_ARMS:
            r = signs[(arm, window)]
            verdict_rows.append(dict(
                metric="G0_sign", registered=0, arm=arm, window=window,
                window_tasks=r["window_tasks"], k=r["k"], n=r["n"],
                rate=r["rate"], cp95_lo=r["cp95_lo"], cp95_hi=r["cp95_hi"],
                status=r["status"], ties=r["ties"],
                excluded_seed_indices=r["excluded_seed_indices"],
                above_seed_indices=r["above_seed_indices"]))
    verdict_rows += [
        dict(metric=r["metric"], registered=0, arm=r["arm"], window="5m",
             window_tasks=WINDOWS["5m"], k=r["k"], n=r["n"], rate=r["rate"],
             cp95_lo=r["cp95_lo"], cp95_hi=r["cp95_hi"],
             status=f"{r['arm']}_COLLAPSE_COUNT", ties="",
             excluded_seed_indices=[],
             above_seed_indices=r["collapse_seed_indices"])
        for r in collapses]
    write_csv(outdir / "verdict.csv", _rectangular(verdict_rows))
    write_csv(outdir / "bimodality.csv", _rectangular([drift] + bimodal))
    (outdir / "summary.md").write_text(
        render_summary(signs, collapses, drift, bimodal, sanity),
        encoding="utf-8")

    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--short"], cwd=ROOT, text=True).splitlines()
    except (OSError, subprocess.CalledProcessError):
        git_hash, dirty = None, []
    provenance = dict(
        experiment=EXPERIMENT,
        analysis_grade="registered_posthoc_not_preregistered",
        grade_note=("values computed in chat 2026-09-01 before the spec was "
                    "frozen; see spec §0 and vault 原典条件照合_0901 §5"),
        created=time.strftime("%Y-%m-%d %H:%M:%S %z"), command=sys.argv,
        elapsed_sec=round(time.time() - started, 3), cwd=os.getcwd(),
        python=sys.version, platform=platform.platform(),
        git_hash=git_hash, git_dirty=dirty,
        spec=str(spec_path), spec_sha256=_sha_file(spec_path),
        source=str(source), input_sha256=meta["input_sha256"],
        parent_recorded_5m=meta["recorded_5m"], sanity=sanity,
        output_sha256={name: _sha_file(outdir / name)
                       for name in ("verdict.csv", "bimodality.csv",
                                    "summary.md")})
    (outdir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")
    print(f"ALL DONE -> {outdir}", flush=True)
    return dict(signs=signs, collapses=collapses, drift=drift,
                bimodality=bimodal, sanity=sanity)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=None)
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args()
    run_analysis(args.source, args.outdir)


if __name__ == "__main__":
    main()
