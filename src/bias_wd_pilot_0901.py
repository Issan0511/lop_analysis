"""段階 A: bias 専用 weight decay の**グリッド決定専用**パイロット。

HANDOFF §4。**判定を出さない。** `verdict.csv` は書かず、書こうとしたら落ちる。
出力は記述統計 CSV と `summary.md` と `grid_selection.json` だけで、冒頭に
「本走のグリッド決定専用。判定を含まない。ここの数値を結果として引用しては
ならない」を必ず書く。

コマンド::

    OMP_NUM_THREADS=1 .venv/bin/python -m src.bias_wd_pilot_0901 --s0
    OMP_NUM_THREADS=1 .venv/bin/python -m src.bias_wd_pilot_0901 --smoke
    OMP_NUM_THREADS=1 .venv/bin/python -m src.bias_wd_pilot_0901 --arm P_1em3
    OMP_NUM_THREADS=1 .venv/bin/python -m src.bias_wd_pilot_0901 --analyze-only
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

from .bias_wd_common import (
    markdown_table, provenance, require_omp, run_arm,
)
from .common import ROOT, load_config

CONFIG = Path(ROOT) / "configs" / "bias_wd_pilot_0901.yaml"

BANNER = (
    "> **本走 (`bias_wd_0901`) の lambda グリッド決定専用。判定を含まない。**\n"
    "> ここの数値を結果として引用してはならない (HANDOFF §4「制約」/ §7-6)。\n"
)


# ------------------------------------------------------------- config

def arm_lambdas(cfg: dict) -> list[tuple[str, float]]:
    return [(a["name"], float(a["wd_b"])) for a in cfg["arms"]]


def validate_config(cfg: dict, *, full: bool) -> None:
    C, A, P = cfg["common"], cfg["condA"], cfg["pilot"]
    if cfg["common"].get("device", "cpu") != "cpu":
        raise ValueError("pilot is CPU-only")
    if (int(A["m"]), int(A["f"]), int(A["target_hidden"])) != (20, 15, 100):
        raise ValueError("registered condA dimensions differ")
    if float(A["beta"]) != 0.7 or list(A["T_values"]) != [10_000]:
        raise ValueError("registered condA regime differs")
    if float(cfg["intervention"]["center_alpha"]) != 0.01:
        raise ValueError("registered center_alpha differs")
    if float(C["lr_main"]) != 0.01:
        raise ValueError("registered lr differs")
    if P.get("emit_verdict") is not False:
        raise ValueError("the pilot must never emit a verdict")
    for arm in cfg["arms"]:
        if list(arm["hidden"]) != [100] or list(arm["centered_layers"]) != [1]:
            raise ValueError(f"{arm['name']}: pilot arms are w100 centered layer 1")
    lams = [lam for _, lam in arm_lambdas(cfg)]
    if lams != sorted(lams) or len(set(lams)) != len(lams):
        raise ValueError("lambda levels must be strictly increasing")
    rule = P["grid_rule"]
    if int(rule["read_at_step"]) != int(C["total_steps"]):
        raise ValueError("grid_rule.read_at_step must be the last recorded step")
    if full and (int(C["total_steps"]) != 500_000
                 or list(C["seeds"]) != list(range(10))):
        raise ValueError("the full pilot is 500k steps and seeds 0..9")


def outdir_of(cfg: dict) -> Path:
    return Path(ROOT) / cfg["pilot"]["output_dir"]


# ------------------------------------------------------------- S0 replay

def s0_replay(cfg: dict, outdir: Path) -> dict:
    """wd_b=0 腕が committed `L1w100_A1` と task 末指標で一致することの確認。

    WD コード経路 (`gb + wd_b*b`) を通したうえで、無 WD 実装の committed 軌道と
    `p_hat` 配列・`unfit`・`eval_loss_exact` が一致することを見る。これが通れば
    「wd_b は b にしか効かない」と「乱数消費が変わっていない」が同時に言える。
    """
    steps = int(cfg["pilot"]["s0_replay_steps"])
    period = int(cfg["phase1"]["task_period"])
    replay_cfg = copy.deepcopy(cfg)
    replay_cfg["common"]["total_steps"] = steps
    replay_cfg["common"]["checkpoints"] = []
    result = run_arm(replay_cfg, "P_none", 0.0, outdir, total_steps=steps,
                     task_period=1000, guard_every=1000,
                     keep_unit_arrays=True, write_logs=False)
    if result["status"] != "COMPLETE":
        raise RuntimeError(f"S0 replay did not complete: {result['status']}")
    frame = result["frame"]

    diffs, max_abs = [], {"unfit": 0.0, "eval_loss_exact": 0.0}
    base_dir = Path(ROOT) / cfg["baseline_dir"] / "logs"
    arm = cfg["pilot"]["baseline_arm"]
    for seed in cfg["common"]["seeds"]:
        path = base_dir / f"{arm}_seed{int(seed)}.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        mine = frame[frame.seed == int(seed)].set_index("step")
        with np.load(path, allow_pickle=False) as z:
            for step in mine.index:
                idx = np.flatnonzero(z["step"] == int(step))
                if len(idx) != 1:
                    diffs.append(dict(seed=int(seed), step=int(step),
                                      field="step", detail=str(len(idx))))
                    continue
                i = int(idx[0])
                for key in ("unfit", "eval_loss_exact"):
                    delta = abs(float(mine.loc[step, key]) - float(z[key][i]))
                    max_abs[key] = max(max_abs[key], delta)
                    if delta > 1e-12:
                        diffs.append(dict(seed=int(seed), step=int(step),
                                          field=key, detail=f"abs={delta:.3g}"))
                dead = int((z["layer1_p_hat"][i] == 0).sum()) / z["layer1_p_hat"].shape[1]
                if abs(float(mine.loc[step, "L1_strict_dead_frac"]) - dead) > 0:
                    diffs.append(dict(seed=int(seed), step=int(step),
                                      field="strict_dead_frac", detail="mismatch"))

    out = dict(pass_=not diffs, steps=steps, probe_every=1000, period=period,
               baseline=str((base_dir / f"{arm}_seed0.npz").relative_to(ROOT)),
               n_seeds=len(cfg["common"]["seeds"]),
               n_probes=int(frame.step.nunique()), max_abs=max_abs,
               differences=diffs[:50], recorder_sanity=result["sanity"])
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "s0_replay.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    if not out["pass_"]:
        raise RuntimeError(f"S0 replay failed: {diffs[:3]}")
    print(f"S0 PASS: {out['n_seeds']} seeds x {out['n_probes']} probes, "
          f"max|dunfit|={max_abs['unfit']:.3g}", flush=True)
    return out


# ------------------------------------------------------------- grid rule

def select_grid(table: pd.DataFrame, rule: dict) -> dict:
    """凍結済み規則で main / sub lambda を選ぶ。実測を見た後の裁量は入らない。

    ``table``: 列 ``wd_b`` と ``metric``（seed 統計を取ったあとの 1 水準 1 行）。
    """
    excluded = {float(v) for v in rule.get("exclude_lambda", [])}
    pool = [(float(r.wd_b), float(r.metric)) for r in table.itertuples()
            if float(r.wd_b) not in excluded and np.isfinite(r.metric)]
    pool.sort(key=lambda kv: kv[0])
    values = [v for _, v in pool]
    lo, hi = (float(v) for v in rule["span_required"])
    spans = bool(values) and min(values) <= lo and max(values) >= hi

    targets = [("main", float(rule["targets"]["main"]))]
    targets += [(f"sub{i + 1}", float(v))
                for i, v in enumerate(rule["targets"]["sub"])]
    taken: set[float] = set()
    picks = {}
    for name, target in targets:
        remaining = [(lam, val) for lam, val in pool if lam not in taken]
        if not remaining:
            picks[name] = None
            continue
        # 目標に最も近い水準。同点は小さい lambda（tie_break: smaller_lambda）
        lam, val = min(remaining, key=lambda kv: (abs(kv[1] - target), kv[0]))
        taken.add(lam)
        picks[name] = dict(wd_b=lam, metric=val, target=target)
    return dict(picks=picks, spans_required_range=spans,
                observed_range=[min(values), max(values)] if values else None,
                required_range=[lo, hi], pool=[dict(wd_b=l, metric=v)
                                               for l, v in pool],
                rule=copy.deepcopy(rule))


# ------------------------------------------------------------- analysis

def analyze(cfg: dict, outdir: Path) -> dict:
    frame = pd.read_csv(outdir / "task_end_metrics.csv")
    rule = cfg["pilot"]["grid_rule"]
    read_at = int(rule["read_at_step"])
    metric = str(rule["metric"])
    lam_of = dict(arm_lambdas(cfg))

    at_end = frame[frame.step == read_at]
    if at_end.empty:
        raise RuntimeError(f"no rows at step {read_at}")
    columns = ["L1_wall_frac", "L1_b_median_alive", "L1_beta_median_alive",
               "L1_kappa_median_alive", "L1_margin_median_alive",
               "L1_strict_dead_frac", "L1_p_hat_median_alive",
               "L1_p_hat_thin_frac", "L1_p_hat_sat_frac", "L1_eff_rank",
               "unfit"]
    table = (at_end.groupby("arm")[columns].median()
             .reindex([a for a, _ in arm_lambdas(cfg)]).reset_index())
    table.insert(1, "wd_b", table["arm"].map(lam_of))
    table["log10_unfit_mean_over_seeds"] = [
        float(np.mean(np.log10(np.maximum(
            at_end[at_end.arm == arm]["unfit"].to_numpy(), 1e-16))))
        for arm in table["arm"]]
    table.to_csv(outdir / "grid_table.csv", index=False)

    selection = select_grid(table.rename(columns={metric: "metric"})
                            [["wd_b", "metric"]], rule)
    selection["read_at_step"] = read_at
    selection["metric"] = metric
    selection["seed_statistic"] = str(rule["seed_statistic"])
    (outdir / "grid_selection.json").write_text(
        json.dumps(selection, indent=2, ensure_ascii=False), encoding="utf-8")

    # 全 task 末の推移 (記述のみ)
    trend = (frame.groupby(["arm", "wd_b", "task"])[columns]
             .median().reset_index())
    trend.to_csv(outdir / "trend_by_task.csv", index=False)
    _figure(frame, cfg, outdir)
    _summary(cfg, outdir, table, selection, frame)
    return selection


def _figure(frame: pd.DataFrame, cfg: dict, outdir: Path) -> None:
    keys = [("L1_wall_frac", "wall_frac = |beta|/kappa (alive median)", False),
            ("L1_b_median_alive", "b (alive median)", False),
            ("L1_strict_dead_frac", "strict_dead_frac", False),
            ("unfit", "exact-support unfit", True)]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(cfg["arms"])))
    for (key, label, logy), ax in zip(keys, axes.ravel()):
        for color, (arm, lam) in zip(colors, arm_lambdas(cfg)):
            g = frame[frame.arm == arm].groupby("step")[key].median()
            ax.plot(g.index, g.values, lw=1.4, color=color,
                    label=f"$\\lambda$={lam:g}")
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
        if logy:
            ax.set_yscale("log")
    axes[0, 0].axhline(0.5, color="gray", ls="--", lw=1)
    for ax in axes[1]:
        ax.set_xlabel("step")
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.suptitle("bias_wd_pilot_0901 — grid selection only, no verdict")
    fig.tight_layout()
    fig.savefig(outdir / "fig_bias_wd_pilot.png", dpi=150)
    plt.close(fig)


def _summary(cfg: dict, outdir: Path, table: pd.DataFrame, selection: dict,
             frame: pd.DataFrame) -> None:
    rule = cfg["pilot"]["grid_rule"]
    picks = selection["picks"]
    chosen = pd.DataFrame([
        dict(role=name, target=p["target"] if p else float("nan"),
             wd_b=p["wd_b"] if p else float("nan"),
             observed_wall_frac=p["metric"] if p else float("nan"))
        for name, p in picks.items()])
    lines = [
        "# bias_wd_pilot_0901 — 段階 A（グリッド決定専用パイロット）", "",
        BANNER, "",
        "## 設計", "",
        f"- 腕: `L1w100_A1` と同一設定で `wd_b` だけを振った {len(cfg['arms'])} 水準",
        "- $\\lambda$ = " + ", ".join(f"{lam:g}" for _, lam in arm_lambdas(cfg)),
        f"- seed 0–9、{int(cfg['common']['total_steps']):,} step、CPU、"
        f"`OMP_NUM_THREADS=1`", "",
        f"## step {int(rule['read_at_step']):,} の水準別要約（seed 中央値）", "",
        markdown_table(table), "",
        "## 凍結済み規則による選択", "", markdown_table(chosen), "",
        f"- 実測範囲 {selection['observed_range']}（要求 "
        f"{selection['required_range']}）、跨いだか "
        f"**{selection['spans_required_range']}**", "",
        "## 注意", "",
        "- **判定は含まない。** `verdict.csv` は書いていない",
        "- ここの数値を結果として引用してはならない",
    ]
    (outdir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ------------------------------------------------------------- driver

OUTPUTS = ("task_end_metrics.csv", "grid_table.csv", "trend_by_task.csv",
           "grid_selection.json", "summary.md", "run_sanity.json",
           "config_used.yaml", "s0_replay.json", "fig_bias_wd_pilot.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--s0", action="store_true", help="S0 replay only")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--arm", help="run a single arm and write its shard")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--outdir")
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = load_config(str(cfg_path))
    full = not (args.smoke or args.s0)
    validate_config(cfg, full=full)
    require_omp(int(cfg["pilot"]["omp_num_threads"]))
    outdir = Path(args.outdir).resolve() if args.outdir else outdir_of(cfg)
    outdir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    if args.s0:
        s0_replay(cfg, Path(ROOT) / "results" / "_s0_bias_wd_pilot_0901")
        return

    if args.smoke:
        smoke_cfg = copy.deepcopy(cfg)
        smoke_cfg["common"]["seeds"] = [0]
        smoke_cfg["common"]["total_steps"] = 30_000
        smoke_dir = Path(ROOT) / "results" / "_smoke_bias_wd_pilot_0901"
        smoke_dir.mkdir(parents=True, exist_ok=True)
        for arm, lam in arm_lambdas(smoke_cfg):
            r = run_arm(smoke_cfg, arm, lam, smoke_dir, total_steps=30_000,
                        task_period=10_000, guard_every=1_000,
                        keep_unit_arrays=False, write_logs=False)
            if not r["sanity"]["pass_"]:
                raise RuntimeError(f"smoke sanity failed: {r['sanity']}")
        print("SMOKE PASS", flush=True)
        return

    shard_dir = outdir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    if args.arm:
        lam = dict(arm_lambdas(cfg))[args.arm]
        result = run_arm(cfg, args.arm, lam, outdir,
                         total_steps=int(cfg["common"]["total_steps"]),
                         task_period=int(cfg["phase1"]["task_period"]),
                         guard_every=int(cfg["pilot"]["guard_every"]))
        result["frame"].to_csv(shard_dir / f"{args.arm}.csv", index=False)
        (shard_dir / f"{args.arm}.json").write_text(json.dumps(
            {k: v for k, v in result.items() if k != "frame"},
            indent=2, ensure_ascii=False), encoding="utf-8")
        if not result["sanity"]["pass_"]:
            raise RuntimeError(f"{args.arm} sanity failed")
        return

    if not args.analyze_only:
        for arm, lam in arm_lambdas(cfg):
            result = run_arm(cfg, arm, lam, outdir,
                             total_steps=int(cfg["common"]["total_steps"]),
                             task_period=int(cfg["phase1"]["task_period"]),
                             guard_every=int(cfg["pilot"]["guard_every"]))
            result["frame"].to_csv(shard_dir / f"{arm}.csv", index=False)
            (shard_dir / f"{arm}.json").write_text(json.dumps(
                {k: v for k, v in result.items() if k != "frame"},
                indent=2, ensure_ascii=False), encoding="utf-8")

    shards, sanity = [], {}
    for arm, _ in arm_lambdas(cfg):
        path = shard_dir / f"{arm}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        shards.append(pd.read_csv(path))
        sanity[arm] = json.loads((shard_dir / f"{arm}.json").read_text())
    frame = pd.concat(shards, ignore_index=True)
    frame.to_csv(outdir / "task_end_metrics.csv", index=False)

    s0_path = Path(ROOT) / "results" / "_s0_bias_wd_pilot_0901" / "s0_replay.json"
    s0 = json.loads(s0_path.read_text(encoding="utf-8")) if s0_path.exists() else None
    if s0 is not None and not s0.get("pass_"):
        raise RuntimeError("saved S0 replay did not pass")
    run_sanity = {
        "S0_wd0_matches_L1w100_A1": bool(s0["pass_"]) if s0 else None,
        "S0_max_abs": (s0 or {}).get("max_abs"),
        "arms": {arm: sanity[arm]["sanity"] for arm, _ in arm_lambdas(cfg)},
        "all_arms_complete": all(sanity[a]["status"] == "COMPLETE"
                                 for a, _ in arm_lambdas(cfg)),
        "all_arms_sanity_pass": all(sanity[a]["sanity"]["pass_"]
                                    for a, _ in arm_lambdas(cfg)),
        "training_elapsed_sec": {a: sanity[a]["elapsed_sec"]
                                 for a, _ in arm_lambdas(cfg)},
    }
    (outdir / "run_sanity.json").write_text(
        json.dumps(run_sanity, indent=2, ensure_ascii=False), encoding="utf-8")
    with (outdir / "config_used.yaml").open("w") as fh:
        yaml.safe_dump(cfg, fh, allow_unicode=True, sort_keys=False)
    if s0 is not None:
        (outdir / "s0_replay.json").write_text(
            json.dumps(s0, indent=2, ensure_ascii=False), encoding="utf-8")

    selection = analyze(cfg, outdir)
    if (outdir / "verdict.csv").exists():
        raise RuntimeError("the pilot must not emit verdict.csv")
    (outdir / "provenance.json").write_text(json.dumps(provenance(
        "bias_wd_pilot_0901", cfg_path, cfg, outdir,
        dict(grid_selection=selection, run_sanity=run_sanity,
             stage="A_pilot_no_verdict"),
        started, sys.argv, OUTPUTS), indent=2, ensure_ascii=False),
        encoding="utf-8")
    print(json.dumps(selection["picks"], indent=2), flush=True)
    print(f"PILOT DONE -> {outdir}", flush=True)


if __name__ == "__main__":
    main()
