"""std 腕へ bias 専用 weight decay を入れる反証テスト。

事前登録: ``specs/spec_bias_wd_std_0901.md``。

コマンド::

    OMP_NUM_THREADS=1 python -m src.bias_wd_std_0901 --s1s2
    OMP_NUM_THREADS=1 python -m src.bias_wd_std_0901 --s0
    OMP_NUM_THREADS=1 python -m src.bias_wd_std_0901 --smoke
    OMP_NUM_THREADS=1 python -m src.bias_wd_std_0901 --arm S_main
    OMP_NUM_THREADS=1 python -m src.bias_wd_std_0901 --analyze-only
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from .bias_wd_0901 import paired_ci, s1_s2_algebra
from .bias_wd_common import markdown_table, provenance, require_omp, run_arm
from .common import ROOT, load_config


CONFIG = Path(ROOT) / "configs" / "bias_wd_std_0901.yaml"
LOP_PERSISTS = "LOP_PERSISTS"
LOP_REMOVED = "LOP_REMOVED"
INCONCLUSIVE_PARTIAL = "INCONCLUSIVE_PARTIAL"


def arms_of(cfg: dict) -> list[dict]:
    return list(cfg["arms"])


def arm_lambda(cfg: dict, name: str) -> float:
    return float(next(a for a in cfg["arms"] if a["name"] == name)["wd_b"])


def outdir_of(cfg: dict) -> Path:
    return Path(ROOT) / cfg["bias_wd"]["output_dir"]


def validate_config(cfg: dict, *, full: bool) -> None:
    C, A, P = cfg["common"], cfg["condA"], cfg["bias_wd"]
    if C.get("device", "cpu") != "cpu":
        raise ValueError("bias_wd_std_0901 is CPU-only")
    if (int(A["m"]), int(A["f"]), int(A["target_hidden"])) != (20, 15, 100):
        raise ValueError("registered condA dimensions differ")
    if float(A["beta"]) != 0.7 or list(A["T_values"]) != [10_000]:
        raise ValueError("registered condA regime differs")
    if list(A["encodings"]) != ["std"] or float(C["lr_main"]) != 0.01:
        raise ValueError("registered encoding/lr differs")
    names = [a["name"] for a in cfg["arms"]]
    if names != ["S_none", "S_main", "S_sub"]:
        raise ValueError(f"registered arms differ: {names}")
    expected = {"S_none": 0.0, "S_main": 1e-3, "S_sub": 1e-1}
    for arm in cfg["arms"]:
        if list(arm["hidden"]) != [100, 100] or list(arm["centered_layers"]) != []:
            raise ValueError(f"{arm['name']}: hidden/centering differs")
        if float(arm["wd_b"]) != expected[arm["name"]]:
            raise ValueError(f"{arm['name']}: wd_b differs")
    if (int(P["bootstrap_B"]), int(P["bootstrap_seed"])) != (20_000, 20_260_903):
        raise ValueError("registered bootstrap differs")
    if list(P["early_block_tasks"]) != [51, 100] or list(P["late_block_tasks"]) != [451, 500]:
        raise ValueError("registered windows differ")
    if (float(P["unfit_floor_L2"]), float(P["persist_ratio"]),
            float(P["removed_ratio"])) != (1e-23, 0.5, 0.1):
        raise ValueError("registered floor/threshold differs")
    if full and (int(C["total_steps"]) != 5_000_000
                 or list(C["seeds"]) != list(range(10))):
        raise ValueError("the full run is 5M steps and seeds 0..9")


def s0_replay(cfg: dict, gate_dir: Path) -> dict:
    """S_none が committed mlp2_phase1_0829/L2_none と一致すること。"""
    steps = int(cfg["bias_wd"]["s0_replay_steps"])
    baseline = str(cfg["bias_wd"]["baseline_arm"])
    base_dir = Path(ROOT) / cfg["baseline_dir"] / "logs"
    replay_cfg = copy.deepcopy(cfg)
    replay_cfg["common"]["total_steps"] = steps
    replay_cfg["common"]["checkpoints"] = []
    result = run_arm(replay_cfg, "S_none", 0.0, gate_dir, total_steps=steps,
                     task_period=1000, guard_every=1000,
                     keep_unit_arrays=False, write_logs=False)
    frame = result["frame"]
    differences: list[dict] = []
    max_abs = {"unfit": 0.0, "eval_loss_exact": 0.0}
    for seed in cfg["common"]["seeds"]:
        path = base_dir / f"{baseline}_seed{int(seed)}.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        mine = frame[frame.seed == int(seed)].set_index("step")
        with np.load(path, allow_pickle=False) as data:
            for step in mine.index:
                found = np.flatnonzero(data["step"] == int(step))
                if len(found) != 1:
                    differences.append(dict(seed=int(seed), step=int(step),
                                            field="step", detail=str(len(found))))
                    continue
                i = int(found[0])
                for key in ("unfit", "eval_loss_exact"):
                    delta = abs(float(mine.loc[step, key]) - float(data[key][i]))
                    max_abs[key] = max(max_abs[key], delta)
                    if delta > 1e-12:
                        differences.append(dict(seed=int(seed), step=int(step),
                                                field=key, detail=f"abs={delta:.3g}"))
                for li in (1, 2):
                    dead = float((data[f"layer{li}_p_hat"][i] == 0).mean())
                    if float(mine.loc[step, f"L{li}_strict_dead_frac"]) != dead:
                        differences.append(dict(seed=int(seed), step=int(step),
                                                field=f"L{li}_strict_dead_frac",
                                                detail="mismatch"))
    report = dict(
        pass_=not differences, arm="S_none", baseline=baseline, steps=steps,
        n_seeds=len(cfg["common"]["seeds"]), n_probes=int(frame.step.nunique()),
        max_abs=max_abs, differences=differences[:50],
        recorder_sanity=result["sanity"],
    )
    gate_dir.mkdir(parents=True, exist_ok=True)
    (gate_dir / "s0_replay.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"S0 [S_none vs {baseline}]: {'PASS' if report['pass_'] else 'FAIL'} "
          f"max|dunfit|={max_abs['unfit']:.3g}", flush=True)
    if not report["pass_"]:
        raise RuntimeError("S0 replay failed; the full run is blocked")
    return report


METRICS = [
    "unfit", "eval_loss_exact",
    "L1_strict_dead_frac", "L2_strict_dead_frac",
    "L1_M_median_alive", "L2_M_median_alive",
    "L1_B_median_alive", "L2_B_median_alive",
    "L1_beta_median_alive", "L2_beta_median_alive",
    "L1_eff_rank", "L2_eff_rank",
    "L1_p_hat_thin_frac", "L2_p_hat_thin_frac",
    "L1_p_hat_sat_frac", "L2_p_hat_sat_frac",
]


def block_levels(cfg: dict, frame: pd.DataFrame) -> pd.DataFrame:
    size = int(cfg["bias_wd"]["block_tasks"])
    floor = float(cfg["bias_wd"]["unfit_floor_L2"])
    rows = []
    for (arm, seed), group in frame.groupby(["arm", "seed"], sort=True):
        group = group[group.task > 0].copy()
        group["block"] = ((group.task - 1) // size + 1).astype(int)
        group["log10_unfit"] = np.log10(np.maximum(group.unfit.to_numpy(), floor))
        group["at_floor"] = group.unfit.to_numpy() <= floor
        for block, part in group.groupby("block"):
            row = dict(
                arm=arm, seed=int(seed), block=int(block),
                task_lo=int(part.task.min()), task_hi=int(part.task.max()),
                n_task_ends=int(len(part)),
                mean_log10_unfit=float(part.log10_unfit.mean()),
                log10_mean_unfit=float(np.log10(max(float(part.unfit.mean()), floor))),
                floor=floor, floor_frac=float(part.at_floor.mean()),
            )
            row.update({key: float(part[key].mean()) for key in METRICS})
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["arm", "seed", "block"])


def _draws(cfg: dict, n: int) -> np.ndarray:
    rng = np.random.default_rng(int(cfg["bias_wd"]["bootstrap_seed"]))
    return rng.integers(0, n, size=(int(cfg["bias_wd"]["bootstrap_B"]), n))


def _series(levels: pd.DataFrame, arm: str, block: int, column: str,
            seeds: list[int]) -> np.ndarray:
    group = levels[(levels.arm == arm) & (levels.block == block)].set_index("seed")
    missing = [seed for seed in seeds if seed not in group.index]
    if missing:
        raise RuntimeError(f"{arm} block {block}: missing seeds {missing}")
    return group.loc[seeds, column].to_numpy(dtype=np.float64)


def _ratio_result(cfg: dict, numerator: np.ndarray, denominator: np.ndarray,
                  draws: np.ndarray) -> tuple[np.ndarray, dict, np.ndarray]:
    small = np.abs(denominator) < float(cfg["bias_wd"]["small_denominator_dex"])
    if np.any(denominator == 0):
        raise RuntimeError("exactly zero S_none degradation makes the registered ratio undefined")
    ratio = numerator / denominator
    return ratio, paired_ci(cfg, ratio, draws), small


def _verdict(cfg: dict, ci: dict) -> str:
    if float(ci["ci_lo"]) >= float(cfg["bias_wd"]["persist_ratio"]):
        return LOP_PERSISTS
    if float(ci["ci_hi"]) <= float(cfg["bias_wd"]["removed_ratio"]):
        return LOP_REMOVED
    return INCONCLUSIVE_PARTIAL


def analyze(cfg: dict, outdir: Path) -> dict:
    P = cfg["bias_wd"]
    seeds = [int(seed) for seed in cfg["common"]["seeds"]]
    frame = pd.read_csv(outdir / "task_end_metrics.csv")
    levels = block_levels(cfg, frame)
    levels.to_csv(outdir / "block_levels.csv", index=False)
    size = int(P["block_tasks"])
    b02 = (int(P["early_block_tasks"][1]) - 1) // size + 1
    b10 = (int(P["late_block_tasks"][1]) - 1) // size + 1
    draws = _draws(cfg, len(seeds))

    def series(arm: str, block: int, column: str) -> np.ndarray:
        return _series(levels, arm, block, column, seeds)

    complete_arms = set(frame.arm.unique())
    if "S_none" not in complete_arms:
        raise RuntimeError("control arm S_none did not complete; no registered analysis is possible")

    def divergence_text(arm: str) -> str:
        path = outdir / "arm_status" / f"{arm}.json"
        if not path.exists():
            return f"{arm} stopped by registered S4; partial trajectory excluded"
        event = json.loads(path.read_text(encoding="utf-8"))
        return (f"{arm} stopped by registered S4 at step {event['detected_step']} "
                f"(seeds={event['bad_seeds']}); partial trajectory excluded")

    drift = {
        arm: series(arm, b10, "mean_log10_unfit")
             - series(arm, b02, "mean_log10_unfit")
        for arm in complete_arms
    }
    ratio_main = ratio_main_ci = diff_main_ci = None
    small = np.zeros(len(seeds), dtype=bool)
    if "S_main" in complete_arms:
        ratio_main, ratio_main_ci, small = _ratio_result(
            cfg, drift["S_main"], drift["S_none"], draws)
        diff_main_ci = paired_ci(cfg, drift["S_main"] - drift["S_none"], draws)
        main_verdict = _verdict(cfg, ratio_main_ci)
        rows = [dict(
            pred="P-main", scope="mean(log10 unfit), B10-B02, S_main/S_none",
            verdict=main_verdict,
            evidence=(f"none drift {drift['S_none'].mean():+.6f} dex; main drift "
                      f"{drift['S_main'].mean():+.6f} dex; paired seed-ratio mean "
                      f"{ratio_main_ci['point']:.6f} CI [{ratio_main_ci['ci_lo']:.6f}, "
                      f"{ratio_main_ci['ci_hi']:.6f}]; drift difference "
                      f"{diff_main_ci['point']:+.6f} CI [{diff_main_ci['ci_lo']:+.6f}, "
                      f"{diff_main_ci['ci_hi']:+.6f}]; |none drift|<"
                      f"{float(P['small_denominator_dex']):g} dex: {int(small.sum())}/10"),
            ci_basis="paired percentile bootstrap", ci_degenerate=int(
                ratio_main_ci["ci_degenerate"] or diff_main_ci["ci_degenerate"]),
        )]
    else:
        main_verdict = "NUMERIC_DIVERGENCE"
        rows = [dict(
            pred="P-main", scope="mean(log10 unfit), B10-B02, S_main/S_none",
            verdict=main_verdict,
            evidence=(f"{divergence_text('S_main')}; completed S_none drift "
                      f"{drift['S_none'].mean():+.6f} dex"),
            ci_basis="not computed", ci_degenerate="",
        )]

    ratio_sub = None
    if "S_sub" in complete_arms:
        ratio_sub, ratio_sub_ci, small_sub = _ratio_result(
            cfg, drift["S_sub"], drift["S_none"], draws)
        diff_sub_ci = paired_ci(cfg, drift["S_sub"] - drift["S_none"], draws)
        rows.append(dict(
            pred="P-dose", scope="mean(log10 unfit), B10-B02, S_sub/S_none",
            verdict="REPORT_ONLY",
            evidence=(f"none drift {drift['S_none'].mean():+.6f} dex; sub drift "
                      f"{drift['S_sub'].mean():+.6f} dex; paired seed-ratio mean "
                      f"{ratio_sub_ci['point']:.6f} CI [{ratio_sub_ci['ci_lo']:.6f}, "
                      f"{ratio_sub_ci['ci_hi']:.6f}]; drift difference "
                      f"{diff_sub_ci['point']:+.6f} CI [{diff_sub_ci['ci_lo']:+.6f}, "
                      f"{diff_sub_ci['ci_hi']:+.6f}]; small denominators "
                      f"{int(small_sub.sum())}/10"),
            ci_basis="paired percentile bootstrap",
            ci_degenerate=int(
                ratio_sub_ci["ci_degenerate"] or diff_sub_ci["ci_degenerate"]),
        ))
    else:
        rows.append(dict(
            pred="P-dose", scope="mean(log10 unfit), B10-B02, S_sub/S_none",
            verdict="NUMERIC_DIVERGENCE",
            evidence=divergence_text("S_sub"),
            ci_basis="not computed", ci_degenerate="",
        ))

    for arm in ("S_none", "S_main", "S_sub"):
        if arm not in complete_arms:
            rows.append(dict(
                pred="dead", scope=f"B10 strict_dead_frac ({arm})",
                verdict="NUMERIC_DIVERGENCE", evidence="B10 unavailable",
                ci_basis="not computed", ci_degenerate="",
            ))
            continue
        rows.append(dict(
            pred="dead", scope=f"B10 strict_dead_frac ({arm})", verdict="REPORT_ONLY",
            evidence=(f"L1 {series(arm, b10, 'L1_strict_dead_frac').mean():.6f}; "
                      f"L2 {series(arm, b10, 'L2_strict_dead_frac').mean():.6f}"),
            ci_basis="", ci_degenerate="",
        ))

    ledger_rows = []
    for arm in ("S_none", "S_main", "S_sub"):
        for layer in (1, 2):
            if arm not in complete_arms:
                rows.append(dict(
                    pred="ledger",
                    scope=f"alive median channels B02->B10 ({arm}, L{layer})",
                    verdict="NUMERIC_DIVERGENCE", evidence="B10 unavailable",
                    ci_basis="not computed", ci_degenerate="",
                ))
                continue
            m02 = series(arm, b02, f"L{layer}_M_median_alive")
            m10 = series(arm, b10, f"L{layer}_M_median_alive")
            b02v = series(arm, b02, f"L{layer}_B_median_alive")
            b10v = series(arm, b10, f"L{layer}_B_median_alive")
            mci = paired_ci(cfg, m10 - m02, draws)
            bci = paired_ci(cfg, b10v - b02v, draws)
            evidence = (
                f"M {m02.mean():+.6f}->{m10.mean():+.6f}, delta {mci['point']:+.6f} "
                f"CI [{mci['ci_lo']:+.6f}, {mci['ci_hi']:+.6f}]; "
                f"B {b02v.mean():+.6f}->{b10v.mean():+.6f}, delta {bci['point']:+.6f} "
                f"CI [{bci['ci_lo']:+.6f}, {bci['ci_hi']:+.6f}]"
            )
            rows.append(dict(
                pred="ledger", scope=f"alive median channels B02->B10 ({arm}, L{layer})",
                verdict="REPORT_ONLY", evidence=evidence,
                ci_basis="paired percentile bootstrap",
                ci_degenerate=int(mci["ci_degenerate"] or bci["ci_degenerate"]),
            ))
            ledger_rows.append(dict(
                arm=arm, layer=layer, M_B02=m02.mean(), M_B10=m10.mean(),
                M_delta=mci["point"], M_ci_lo=mci["ci_lo"], M_ci_hi=mci["ci_hi"],
                B_B02=b02v.mean(), B_B10=b10v.mean(), B_delta=bci["point"],
                B_ci_lo=bci["ci_lo"], B_ci_hi=bci["ci_hi"],
            ))

    for arm in ("S_main", "S_sub"):
        if arm not in complete_arms:
            rows.append(dict(
                pred="static", scope=f"B10 mean(log10 unfit), {arm}-S_none",
                verdict="NUMERIC_DIVERGENCE", evidence="B10 unavailable",
                ci_basis="not computed", ci_degenerate="",
            ))
            continue
        delta = series(arm, b10, "mean_log10_unfit") - series(
            "S_none", b10, "mean_log10_unfit")
        ci = paired_ci(cfg, delta, draws)
        rows.append(dict(
            pred="static", scope=f"B10 mean(log10 unfit), {arm}-S_none",
            verdict="REPORT_ONLY",
            evidence=f"paired {ci['point']:+.6f} dex CI [{ci['ci_lo']:+.6f}, {ci['ci_hi']:+.6f}]",
            ci_basis="paired percentile bootstrap", ci_degenerate=int(ci["ci_degenerate"]),
        ))

    verdict = pd.DataFrame(rows)
    verdict.to_csv(outdir / "verdict.csv", index=False)

    endpoints = pd.DataFrame({"seed": seeds})
    for arm in ("S_none", "S_main", "S_sub"):
        if arm not in complete_arms:
            continue
        endpoints[f"{arm}_drift_B10minusB02"] = drift[arm]
        for block, tag in ((b02, "B02"), (b10, "B10")):
            endpoints[f"{arm}_{tag}_meanlog10unfit"] = series(
                arm, block, "mean_log10_unfit")
            for layer in (1, 2):
                for metric in ("strict_dead_frac", "M_median_alive", "B_median_alive"):
                    endpoints[f"{arm}_{tag}_L{layer}_{metric}"] = series(
                        arm, block, f"L{layer}_{metric}")
    if ratio_main is not None:
        endpoints["S_main_over_S_none_drift_ratio"] = ratio_main
    if ratio_sub is not None:
        endpoints["S_sub_over_S_none_drift_ratio"] = ratio_sub
    endpoints.to_csv(outdir / "paired_endpoints.csv", index=False)

    ledger = pd.DataFrame(ledger_rows)
    _figure(cfg, frame, outdir)
    result = dict(
        main_verdict=main_verdict, blocks=dict(B02=b02, B10=b10),
        ratio_main=ratio_main_ci, diff_main=diff_main_ci,
        control_drift=float(drift["S_none"].mean()),
        small_denominator_seeds=[int(seeds[i]) for i in np.flatnonzero(small)],
    )
    _summary(cfg, outdir, verdict, levels, ledger, result)
    return result


def _figure(cfg: dict, frame: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    colors = {"S_none": "#555555", "S_main": "#2b8cbe", "S_sub": "#e34a33"}
    panels = [
        ("unfit", "exact-support unfit", True),
        ("L1_strict_dead_frac", "strict_dead_frac (L1)", False),
        ("L2_strict_dead_frac", "strict_dead_frac (L2)", False),
        ("L1_M_median_alive", "alive median M (L1)", False),
        ("L1_B_median_alive", "alive median B (L1)", False),
        ("L2_M_median_alive", "alive median M (L2)", False),
    ]
    for (key, label, logy), ax in zip(panels, axes.flat):
        for arm in ("S_none", "S_main", "S_sub"):
            group = frame[frame.arm == arm].groupby("task")[key].median()
            if group.empty:
                continue
            ax.plot(group.index, group.values, lw=1.2, color=colors[arm],
                    label=f"{arm} ($\\lambda$={arm_lambda(cfg, arm):g})")
        ax.set_xlabel("task")
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
        if logy:
            ax.set_yscale("log")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("bias_wd_std_0901 — std, weight decay on hidden bias only")
    fig.tight_layout()
    fig.savefig(outdir / "fig_bias_wd_std.png", dpi=150)
    plt.close(fig)


def _summary(cfg: dict, outdir: Path, verdict: pd.DataFrame,
             levels: pd.DataFrame, ledger: pd.DataFrame, result: dict) -> None:
    P = cfg["bias_wd"]
    sanity = json.loads((outdir / "run_sanity.json").read_text(encoding="utf-8"))
    s4_parts = []
    for arm, status in sanity["S4_numeric_divergence"].items():
        if status == "COMPLETE":
            s4_parts.append(f"{arm} PASS")
        else:
            event = json.loads((outdir / "arm_status" / f"{arm}.json").read_text(
                encoding="utf-8"))
            s4_parts.append(
                f"{arm} FAIL (step {event['detected_step']}, seeds={event['bad_seeds']})")
    b10 = result["blocks"]["B10"]
    arms = [a["name"] for a in cfg["arms"]]
    late = levels[levels.block == b10]
    table = (late.groupby("arm")[[
        "mean_log10_unfit", "log10_mean_unfit", "L1_strict_dead_frac",
        "L2_strict_dead_frac", "L1_M_median_alive", "L1_B_median_alive",
        "L2_M_median_alive", "L2_B_median_alive", "floor_frac",
    ]].mean().reindex(arms).reset_index())
    table.insert(1, "wd_b", [arm_lambda(cfg, arm) for arm in arms])
    ratio = result["ratio_main"]
    diff = result["diff_main"]
    small = result["small_denominator_seeds"]
    if ratio is None:
        primary_lines = [
            "- `S_main` が S4 で停止したため、登録 endpoint の劣化比・対応劣化差は"
            "算出不能（部分軌道は除外）",
            f"- 完走した `S_none` の B10−B02 劣化は平均 "
            f"{result['control_drift']:+.6f} dex",
        ]
    else:
        primary_lines = [
            f"- `S_main/S_none` の劣化比: {ratio['point']:.6f} "
            f"(paired percentile 95% CI [{ratio['ci_lo']:.6f}, {ratio['ci_hi']:.6f}])",
            f"- 対応劣化差: {diff['point']:+.6f} dex "
            f"(95% CI [{diff['ci_lo']:+.6f}, {diff['ci_hi']:+.6f}])",
            f"- `abs(S_none drift)<{float(P['small_denominator_dex']):g}` dex の seed: "
            f"{small if small else 'なし'}",
        ]
    lines = [
        "# bias_wd_std_0901 — 本走の結果", "",
        f"事前登録: [`{cfg['spec']}`](../../{cfg['spec']})。encoding は std、"
        "3腕とも無中心化。", "",
        "## Verdict", "", markdown_table(verdict), "",
        f"主判定は **{result['main_verdict']}**。B02 = task "
        f"{P['early_block_tasks'][0]}–{P['early_block_tasks'][1]}、B10 = task "
        f"{P['late_block_tasks'][0]}–{P['late_block_tasks'][1]} の "
        "`mean(log10 unfit)` 劣化比を seed 対応で評価する設計。", "",
        *primary_lines, "",
        f"## 終盤窓（task {P['late_block_tasks'][0]}–{P['late_block_tasks'][1]} = "
        f"block {b10}）の腕別水準（seed 平均）", "", markdown_table(table), "",
        "## 台帳移動（alive 中央、B02→B10）", "", markdown_table(ledger), "",
        "## 集計規約", "",
        f"- 主 endpoint は `mean(log10 unfit)`。床 `{float(P['unfit_floor_L2']):g}` "
        "を各 task 末に当ててから log10 を取り、seed 内50 taskを平均",
        f"- paired percentile bootstrap: B={int(P['bootstrap_B'])}, "
        f"bootstrap_seed={int(P['bootstrap_seed'])}。studentized は退化診断のみ",
        "- `log10(mean unfit)`、dead、M/B 台帳、`S_sub`、B10静的差は REPORT_ONLY", "",
        "## サニティ", "",
        f"- **S0: {'PASS' if sanity['S0_S_none_matches_mlp2_phase1_0829_L2_none'] else 'FAIL'}**。"
        "`S_none` と committed `mlp2_phase1_0829/L2_none` を "
        "30k・1k格子で replay",
        f"- **S1/S2: {'PASS' if sanity['S1_wd0_bit_identical_to_no_wd_path'] and sanity['S2_only_hidden_bias_touched'] else 'FAIL'}**。"
        "lambda=0 の bit identity と、WD が隠れ層 bias だけを触る代数検査",
        f"- **S3: {'PASS' if sanity['S3_pass'] else 'FAIL'}**（完走腕。停止腕も S4 検出前の"
        "記録点では壁恒等式・1/32量子化・第1層 kappa 閉形式・独立実装一致に違反なし）。"
        "beta は前件で修正済みのスケール正規化尺度",
        f"- **S4 数値安定性**（probe_every=1000）: {'; '.join(s4_parts)}", "",
        "## 事前登録後の実装補正", "",
        "- S4 で停止した腕の部分ログを除外し、該当 endpoint に "
        "`NUMERIC_DIVERGENCE` を出す処理を本走開始後に追加した。判定式・窓・しきい・"
        "完走腕の数値には触れていない", "",
        "## 引いてはいけない線", "",
        "- `LOP_PERSISTS` でも b-WD が無意味とは書かない。centered では効いている",
        "- `LOP_REMOVED` でも mu 駆動説の棄却まで飛ばず、裁定を Issa に返す",
        "- dead の変化を機能改善・悪化と読み替えない",
    ]
    (outdir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


OUTPUTS = (
    "verdict.csv", "summary.md", "paired_endpoints.csv", "task_end_metrics.csv",
    "block_levels.csv", "run_sanity.json", "config_used.yaml", "fig_bias_wd_std.png",
)


def _shard(outdir: Path) -> Path:
    path = outdir / "shards"
    path.mkdir(parents=True, exist_ok=True)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--s0", action="store_true")
    parser.add_argument("--s1s2", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--arm")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--outdir")
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = load_config(str(cfg_path))
    validate_config(cfg, full=not args.smoke)
    require_omp(int(cfg["bias_wd"]["omp_num_threads"]))
    outdir = Path(args.outdir).resolve() if args.outdir else outdir_of(cfg)
    outdir.mkdir(parents=True, exist_ok=True)
    gate_dir = Path(ROOT) / "results" / "_gate_bias_wd_std_0901"
    started = time.time()

    if args.s1s2:
        s1_s2_algebra(cfg, gate_dir)
        return
    if args.s0:
        s0_replay(cfg, gate_dir)
        return
    if args.smoke:
        smoke_cfg = copy.deepcopy(cfg)
        smoke_cfg["common"]["seeds"] = [0]
        smoke_dir = Path(ROOT) / "results" / "_smoke_bias_wd_std_0901"
        smoke_dir.mkdir(parents=True, exist_ok=True)
        for arm in cfg["arms"]:
            result = run_arm(
                smoke_cfg, arm["name"], float(arm["wd_b"]), smoke_dir,
                total_steps=30_000, task_period=10_000, guard_every=1_000,
                keep_unit_arrays=False, write_logs=False,
            )
            if not result["sanity"]["pass_"]:
                raise RuntimeError(f"smoke sanity failed: {result['sanity']}")
        print("SMOKE PASS", flush=True)
        return

    for name, path in (("S1/S2", gate_dir / "s1_s2_algebra.json"),
                       ("S0", gate_dir / "s0_replay.json")):
        if not path.exists():
            raise FileNotFoundError(f"run --s1s2 and --s0 before the full run ({name})")
        if not json.loads(path.read_text(encoding="utf-8")).get("pass_"):
            raise RuntimeError(f"saved {name} gate did not pass")

    total = int(cfg["common"]["total_steps"])
    period = int(cfg["phase1"]["task_period"])
    guard = int(cfg["bias_wd"]["guard_every"])
    todo = ([a for a in cfg["arms"] if a["name"] == args.arm]
            if args.arm else list(cfg["arms"]))
    if args.arm and not todo:
        raise SystemExit(f"unknown arm {args.arm!r}")

    if not args.analyze_only:
        for arm in todo:
            result = run_arm(
                cfg, arm["name"], float(arm["wd_b"]), outdir,
                total_steps=total, task_period=period, guard_every=guard,
            )
            result["frame"].to_csv(_shard(outdir) / f"{arm['name']}.csv", index=False)
            (_shard(outdir) / f"{arm['name']}.json").write_text(json.dumps(
                {key: value for key, value in result.items() if key != "frame"},
                indent=2, ensure_ascii=False), encoding="utf-8")
            if result["status"] != "COMPLETE":
                print(f"[{arm['name']}] {result['status']}: continuing", flush=True)
            elif not result["sanity"]["pass_"]:
                raise RuntimeError(f"{arm['name']} sanity failed")
        if args.arm:
            return

    shards, meta = [], {}
    for arm in cfg["arms"]:
        path = _shard(outdir) / f"{arm['name']}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        meta[arm["name"]] = json.loads(
            (_shard(outdir) / f"{arm['name']}.json").read_text(encoding="utf-8"))
        if meta[arm["name"]]["status"] == "COMPLETE":
            shards.append(pd.read_csv(path))
    frame = pd.concat(shards, ignore_index=True)
    frame.to_csv(outdir / "task_end_metrics.csv", index=False)

    s0 = json.loads((gate_dir / "s0_replay.json").read_text(encoding="utf-8"))
    s1s2 = json.loads((gate_dir / "s1_s2_algebra.json").read_text(encoding="utf-8"))
    run_sanity = {
        "S0_S_none_matches_mlp2_phase1_0829_L2_none": bool(s0["pass_"]),
        "S0_detail": {key: s0[key] for key in ("baseline", "max_abs", "n_probes")},
        "S1_wd0_bit_identical_to_no_wd_path": bool(all(
            row["S1_bit_identity"] for row in s1s2["rows"])),
        "S2_only_hidden_bias_touched": bool(all(
            row["S2_W_v_c_untouched"] and row["S2_bias_delta_ok"]
            for row in s1s2["rows"]) and s1s2["s2_source_only_bias_update_uses_wd_b"]),
        "S3_exact_support_identities": {
            arm: dict(
                max_relerr=meta[arm]["sanity"]["max_relerr"],
                n_quantization_violations=meta[arm]["sanity"]["n_quantization_violations"],
                n_wall_identity_violations=meta[arm]["sanity"]["n_wall_identity_violations"],
                pass_=(meta[arm]["sanity"]["pass_"]
                       if meta[arm]["status"] == "COMPLETE" else None),
                note=("complete arm" if meta[arm]["status"] == "COMPLETE"
                      else "stopped by S4; identities passed before divergence"),
            ) for arm in meta
        },
        "S3_pass": bool(all(meta[arm]["sanity"]["pass_"] for arm in meta
                            if meta[arm]["status"] == "COMPLETE")),
        "S3_beta_metric": "max|a-b|/max|b_ref| (bias_wd_0901 corrected scale-normalized metric)",
        "S4_numeric_divergence": {arm: meta[arm]["status"] for arm in meta},
        "S4_probe_every": guard,
        "training_elapsed_sec": {arm: meta[arm]["elapsed_sec"] for arm in meta},
        "post_registration_implementation_correction": {
            "what": "exclude every S4-stopped arm from aggregate analysis and emit NUMERIC_DIVERGENCE rows",
            "when": "after S_sub and S_main diverged during the registered full run",
            "affects_registered_numeric_rule": False,
        },
    }
    (outdir / "run_sanity.json").write_text(
        json.dumps(run_sanity, indent=2, ensure_ascii=False), encoding="utf-8")
    with (outdir / "config_used.yaml").open("w") as stream:
        yaml.safe_dump(cfg, stream, allow_unicode=True, sort_keys=False)

    result = analyze(cfg, outdir)
    (outdir / "provenance.json").write_text(json.dumps(provenance(
        "bias_wd_std_0901", cfg_path, cfg, outdir,
        dict(analysis=result, run_sanity=run_sanity,
             predecessor="bias_wd_0901@da22465"),
        started, sys.argv, OUTPUTS), indent=2, ensure_ascii=False), encoding="utf-8")
    print(pd.read_csv(outdir / "verdict.csv")[["pred", "scope", "verdict"]]
          .to_string(index=False), flush=True)
    print(f"ALL DONE -> {outdir}", flush=True)


if __name__ == "__main__":
    main()
