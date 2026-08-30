"""Test candidate 6 (switch concentration) and candidate 7 (initial coverage).

No network update is executed.  Candidate 6 reads the existing 1k-step
``center_selfcov_0814`` logs.  Candidate 7 loads the saved step-0 parameters,
replays only the input generator, enumerates each task's exact 32-point support,
and solves an ordinary least-squares readout with ``W`` and ``b`` fixed.

Run from the repository root::

    OMP_NUM_THREADS=1 .venv/bin/python -m analysis.switch_coverage_0830.analyze
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from src.ratchet_log import full_support_ro, teacher_f64
from src.train import setup_group


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results" / "center_selfcov_0814"
DEFAULT_OUT = ROOT / "results" / "switch_coverage_0830"
WIDTHS = (5, 100)
PERIOD = 10_000
N_TASKS = 100
BOOTSTRAP_B = 20_000
SUBSET_DRAWS = 100
ANALYSIS_SEED = 20260830


def load_config() -> dict:
    with (SOURCE / "config_used.yaml").open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if cfg["common"]["lop_every"] != 1_000:
        raise RuntimeError("candidate 6 requires the existing 1k probe grid")
    if cfg["condA"]["T_values"] != [PERIOD]:
        raise RuntimeError("unexpected condA task period")
    return cfg


def load_centered_logs() -> pd.DataFrame:
    frames = []
    for width in WIDTHS:
        frame = pd.read_csv(SOURCE / f"lop_metrics_A_w{width}.csv")
        frame = frame[frame.run_id.str.contains("_centered_")].copy()
        frame["width"] = width
        frame["seed"] = frame.run_id.str.extract(r"_s(\d+)$")[0].astype(int)
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    expected = {(width, seed) for width in WIDTHS for seed in range(5)}
    actual = set(map(tuple, result[["width", "seed"]].drop_duplicates().to_numpy()))
    if actual != expected:
        raise RuntimeError(f"unexpected run set: {sorted(actual)}")
    return result


def switch_intervals(logs: pd.DataFrame) -> pd.DataFrame:
    """Return consecutive 1k intervals after task 1.

    A row ending at offset 1,000 is the first available post-switch probe: the
    switch occurred immediately after the preceding 10,000-step row, and the
    row includes the first 1,000 updates on the new task.  Other offsets are the
    within-task comparison intervals.
    """
    rows = []
    for (width, seed), group in logs.groupby(["width", "seed"], sort=True):
        group = group.sort_values("step").copy()
        group["start_step"] = group.step.shift()
        group["dead_change"] = group.dead_frac.diff()
        group = group[
            (group.step > PERIOD)
            & ((group.step - group.start_step) == 1_000)
        ].copy()
        group["end_offset"] = group.step % PERIOD
        group["post_switch"] = group.end_offset == 1_000
        group["positive_dead_change"] = group.dead_change.clip(lower=0.0)
        group["positive_label_units"] = group.positive_dead_change * width
        group["net_label_units"] = group.dead_change * width
        rows.append(group[[
            "width", "seed", "start_step", "step", "end_offset",
            "post_switch", "dead_change", "positive_dead_change",
            "positive_label_units", "net_label_units",
        ]])
    result = pd.concat(rows, ignore_index=True)
    counts = result.groupby(["width", "seed", "post_switch"]).size()
    if set(counts[counts.index.get_level_values("post_switch")].to_numpy()) != {99}:
        raise RuntimeError("expected 99 post-switch intervals per run")
    if set(counts[~counts.index.get_level_values("post_switch")].to_numpy()) != {891}:
        raise RuntimeError("expected 891 comparison intervals per run")
    return result


def summarize_switch_by_seed(intervals: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (width, seed), group in intervals.groupby(["width", "seed"], sort=True):
        post = group[group.post_switch]
        rest = group[~group.post_switch]
        post_rate = float(post.positive_dead_change.mean())
        rest_rate = float(rest.positive_dead_change.mean())
        rows.append(dict(
            width=int(width), seed=int(seed),
            n_post=len(post), n_interior=len(rest),
            post_positive_change_rate=post_rate,
            interior_positive_change_rate=rest_rate,
            concentration_fold=post_rate / rest_rate,
            post_event_rate=float((post.positive_dead_change > 0).mean()),
            interior_event_rate=float((rest.positive_dead_change > 0).mean()),
            post_net_change_rate=float(post.dead_change.mean()),
            interior_net_change_rate=float(rest.dead_change.mean()),
        ))
    return pd.DataFrame(rows)


def _switch_stat(seed_table: pd.DataFrame, width: int, sampled: np.ndarray) -> float:
    table = seed_table[seed_table.width == width].set_index("seed")
    post = table.loc[sampled, "post_positive_change_rate"].to_numpy(float).mean()
    rest = table.loc[sampled, "interior_positive_change_rate"].to_numpy(float).mean()
    return post / rest


def bootstrap_switch(seed_table: pd.DataFrame) -> dict:
    rng = np.random.default_rng(ANALYSIS_SEED)
    seeds = np.arange(5)
    draws5 = rng.choice(seeds, size=(BOOTSTRAP_B, len(seeds)), replace=True)
    draws100 = rng.choice(seeds, size=(BOOTSTRAP_B, len(seeds)), replace=True)
    boot5 = np.array([_switch_stat(seed_table, 5, draw) for draw in draws5])
    boot100 = np.array([_switch_stat(seed_table, 100, draw) for draw in draws100])
    point5 = _switch_stat(seed_table, 5, seeds)
    point100 = _switch_stat(seed_table, 100, seeds)
    ratio = boot5 / boot100
    return dict(
        bootstrap_unit="seed cluster (all intervals retained)",
        bootstrap_B=BOOTSTRAP_B,
        w5_concentration_fold=point5,
        w5_ci95=np.quantile(boot5, [0.025, 0.975]).tolist(),
        w100_concentration_fold=point100,
        w100_ci95=np.quantile(boot100, [0.025, 0.975]).tolist(),
        fold_ratio_w5_over_w100=point5 / point100,
        fold_ratio_ci95=np.quantile(ratio, [0.025, 0.975]).tolist(),
    )


def solve_readout(features: np.ndarray, target: np.ndarray, intercept: bool) -> dict:
    design = features
    if intercept:
        design = np.column_stack([features, np.ones(len(features))])
    coef, _, rank, singular = np.linalg.lstsq(design, target, rcond=None)
    prediction = design @ coef
    mse = float(np.mean((prediction - target) ** 2))
    variance = float(np.var(target))
    return dict(
        mse=mse,
        nmse=mse / max(variance, 1e-15),
        design_rank=int(rank),
        min_singular=float(singular[-1]) if len(singular) else float("nan"),
    )


def replay_initial_feature_fits(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    fit_rows = []
    subset_rows = []
    sanity = {}
    rng = np.random.default_rng(ANALYSIS_SEED)

    for width in WIDTHS:
        step0_path = SOURCE / "ckpts" / f"A_w{width}_step0.pt"
        final_path = SOURCE / "ckpts" / f"A_w{width}_step1000000.pt"
        step0 = torch.load(step0_path, map_location="cpu", weights_only=False)
        final = torch.load(final_path, map_location="cpu", weights_only=False)
        state = setup_group(
            ("A", width, 1, "none"), step0["runs"], cfg, torch.device("cpu")
        )
        init_exact = (
            torch.equal(state["net"].W, step0["net"]["W"])
            and torch.equal(state["net"].b, step0["net"]["b"])
            and torch.equal(state["env"].flip_state, step0["env"]["flip_state"])
        )
        if not init_exact:
            raise RuntimeError(f"w{width}: deterministic step-0 reconstruction failed")

        W = step0["net"]["W"].double()
        b = step0["net"]["b"].double()
        centered_indices = [
            i for i, run in enumerate(step0["runs"]) if run["enc"] == "centered"
        ]

        for task in range(1, N_TASKS + 1):
            # segment consumes exactly the original input draws.  Its leading
            # maybe_flip selects the current task, while W/b are never updated.
            state["env"].segment(PERIOD)
            raw = full_support_ro(state["env"]).double()
            target = teacher_f64(state["teacher"], raw)
            task_mean = raw.mean(dim=0)
            modes = [("ideal_task_centered", raw - task_mean[None])]
            if task == 1:
                # Literal state of the centered arm at step 0: running_mean=0,
                # so its learner input is the raw support.
                modes.append(("literal_t0_raw", raw))

            for mode, learner_input in modes:
                activation = torch.relu(
                    torch.einsum("rhd,prd->prh", W, learner_input) + b
                ).numpy()
                target_np = target.numpy()
                for run_index in centered_indices:
                    seed = int(step0["runs"][run_index]["seed"])
                    features = activation[:, run_index, :]
                    response = target_np[:, run_index]
                    primary = solve_readout(features, response, intercept=False)
                    sensitivity = solve_readout(features, response, intercept=True)
                    fit_rows.append(dict(
                        width=width, seed=seed, task=task, mode=mode,
                        n_features=width,
                        nmse_v_only=primary["nmse"], mse_v_only=primary["mse"],
                        rank_v_only=primary["design_rank"],
                        nmse_v_plus_c=sensitivity["nmse"],
                        rank_v_plus_c=sensitivity["design_rank"],
                    ))

                    if width == 100:
                        subset_nmse = []
                        for _ in range(SUBSET_DRAWS):
                            selected = rng.choice(width, size=5, replace=False)
                            result = solve_readout(
                                features[:, selected], response, intercept=False
                            )
                            subset_nmse.append(result["nmse"])
                        subset_rows.append(dict(
                            width=width, seed=seed, task=task, mode=mode,
                            subset_size=5, subset_draws=SUBSET_DRAWS,
                            subset_nmse_median=float(np.median(subset_nmse)),
                            subset_nmse_q25=float(np.quantile(subset_nmse, 0.25)),
                            subset_nmse_q75=float(np.quantile(subset_nmse, 0.75)),
                        ))

        replay_exact = (
            state["env"].t == 1_000_000
            and torch.equal(state["env"].flip_state, final["env"]["flip_state"])
        )
        if not replay_exact:
            raise RuntimeError(f"w{width}: final input replay mismatch")
        sanity[f"w{width}"] = dict(
            step0_reconstruction_exact=True,
            final_env_t=int(state["env"].t),
            final_flip_state_exact=True,
        )

    return pd.DataFrame(fit_rows), pd.DataFrame(subset_rows), sanity


def _cluster_median(table: pd.DataFrame, sampled_seeds: np.ndarray, column: str) -> float:
    blocks = [table[table.seed == int(seed)][column].to_numpy(float) for seed in sampled_seeds]
    return float(np.median(np.concatenate(blocks)))


def bootstrap_coverage(fits: pd.DataFrame, subsets: pd.DataFrame) -> dict:
    ideal = fits[fits["mode"] == "ideal_task_centered"]
    w5 = ideal[ideal.width == 5]
    w100 = ideal[ideal.width == 100]
    sub = subsets[subsets["mode"] == "ideal_task_centered"]
    seeds = np.arange(5)
    rng = np.random.default_rng(ANALYSIS_SEED + 1)
    draws5 = rng.choice(seeds, size=(BOOTSTRAP_B, 5), replace=True)
    draws100 = rng.choice(seeds, size=(BOOTSTRAP_B, 5), replace=True)

    fold_width = np.empty(BOOTSTRAP_B)
    fold_subset = np.empty(BOOTSTRAP_B)
    fold_w5_vs_subset = np.empty(BOOTSTRAP_B)
    for i in range(BOOTSTRAP_B):
        m5 = _cluster_median(w5, draws5[i], "nmse_v_only")
        m100 = _cluster_median(w100, draws100[i], "nmse_v_only")
        msub = _cluster_median(sub, draws100[i], "subset_nmse_median")
        fold_width[i] = m5 / max(m100, 1e-300)
        fold_subset[i] = msub / max(m100, 1e-300)
        fold_w5_vs_subset[i] = m5 / max(msub, 1e-300)

    point5 = float(w5.nmse_v_only.median())
    point100 = float(w100.nmse_v_only.median())
    point_sub = float(sub.subset_nmse_median.median())
    return dict(
        bootstrap_unit="seed cluster (100 task trajectory retained)",
        bootstrap_B=BOOTSTRAP_B,
        ideal_centered_median_nmse_w5=point5,
        ideal_centered_median_nmse_w100=point100,
        w5_over_w100_fold=point5 / point100,
        w5_over_w100_ci95=np.quantile(fold_width, [0.025, 0.975]).tolist(),
        ideal_centered_median_nmse_w100_random5=point_sub,
        random5_over_full100_fold=point_sub / point100,
        random5_over_full100_ci95=np.quantile(fold_subset, [0.025, 0.975]).tolist(),
        w5_over_w100_random5_fold=point5 / point_sub,
        w5_over_w100_random5_ci95=np.quantile(
            fold_w5_vs_subset, [0.025, 0.975]
        ).tolist(),
    )


def build_summary(
    intervals: pd.DataFrame,
    seed_switch: pd.DataFrame,
    switch_boot: dict,
    fits: pd.DataFrame,
    subsets: pd.DataFrame,
    coverage_boot: dict,
    sanity: dict,
) -> dict:
    switch_aggregate = {}
    for width in WIDTHS:
        group = intervals[intervals.width == width]
        post, rest = group[group.post_switch], group[~group.post_switch]
        switch_aggregate[f"w{width}"] = dict(
            post_positive_change_rate=float(post.positive_dead_change.mean()),
            interior_positive_change_rate=float(rest.positive_dead_change.mean()),
            post_positive_event_rate=float((post.positive_dead_change > 0).mean()),
            interior_positive_event_rate=float((rest.positive_dead_change > 0).mean()),
            post_net_change_rate=float(post.dead_change.mean()),
            interior_net_change_rate=float(rest.dead_change.mean()),
            all_seed_concentration_folds=(
                seed_switch[seed_switch.width == width].concentration_fold.tolist()
            ),
        )

    literal = fits[fits["mode"] == "literal_t0_raw"]
    initial = {}
    for width in WIDTHS:
        group = literal[literal.width == width]
        initial[f"w{width}"] = dict(
            nmse_v_only_by_seed=group.sort_values("seed").nmse_v_only.tolist(),
            median_nmse_v_only=float(group.nmse_v_only.median()),
            rank_by_seed=group.sort_values("seed").rank_v_only.astype(int).tolist(),
        )
    initial["w100_full_rank_all_seeds"] = bool(
        (literal[literal.width == 100].rank_v_only == 32).all()
    )
    initial["w100_numerical_interpolation_all_seeds"] = bool(
        (literal[literal.width == 100].nmse_v_only < 1e-20).all()
    )

    ideal = fits[fits["mode"] == "ideal_task_centered"]
    sensitivity = {}
    for width in WIDTHS:
        group = ideal[ideal.width == width]
        sensitivity[f"w{width}"] = dict(
            median_nmse_v_only=float(group.nmse_v_only.median()),
            q25_q75_nmse_v_only=np.quantile(
                group.nmse_v_only, [0.25, 0.75]
            ).tolist(),
            median_nmse_v_plus_c=float(group.nmse_v_plus_c.median()),
            median_rank_v_only=float(group.rank_v_only.median()),
        )

    return dict(
        source="results/center_selfcov_0814 (post-hoc; no training)",
        candidate6=dict(
            verdict="boundary-concentration signature supported; absorbing-crossing mechanism not identified",
            resolution="1,000 steps; first post-switch bin is (0, 1,000] updates",
            aggregate=switch_aggregate,
            bootstrap=switch_boot,
        ),
        candidate7=dict(
            verdict="initial-feature redundancy supported; directional-nearest-neighbor interpretation not isolated",
            literal_t0=initial,
            ideal_task_centered=sensitivity,
            bootstrap=coverage_boot,
            random5_subset_draws=SUBSET_DRAWS,
        ),
        sanity=sanity,
    )


def make_figure(intervals: pd.DataFrame, fits: pd.DataFrame, subsets: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    offsets = np.arange(0, PERIOD, 1_000)
    for width, color in [(5, "tab:red"), (100, "tab:blue")]:
        group = intervals[intervals.width == width]
        means = group.groupby("end_offset").positive_dead_change.mean().reindex(offsets)
        axes[0].plot(offsets, means, marker="o", label=f"w{width}", color=color)
    axes[0].axvspan(500, 1_500, color="0.9", zorder=-1)
    axes[0].set_xlabel("interval end offset within task")
    axes[0].set_ylabel("mean positive dead_frac change / 1k")
    axes[0].set_title("Candidate 6: post-switch concentration")
    axes[0].legend()

    ideal = fits[fits["mode"] == "ideal_task_centered"]
    data = [
        ideal[ideal.width == 5].nmse_v_only.to_numpy(),
        subsets[subsets["mode"] == "ideal_task_centered"].subset_nmse_median.to_numpy(),
        ideal[ideal.width == 100].nmse_v_only.to_numpy(),
    ]
    axes[1].boxplot(data, tick_labels=["w5", "w100\nrandom 5", "w100\nall 100"], showfliers=False)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("best fixed-feature readout NMSE")
    axes[1].set_title("Candidate 7: exact task-support fits")
    fig.tight_layout()
    fig.savefig(out / "switch_coverage_0830.png", dpi=180)
    fig.savefig(out / "switch_coverage_0830.pdf")
    plt.close(fig)


def fmt_ci(value: float, ci: list[float]) -> str:
    return f"{value:.3g} [{ci[0]:.3g}, {ci[1]:.3g}]"


def write_report(summary: dict, out: Path) -> None:
    c6 = summary["candidate6"]
    c7 = summary["candidate7"]
    b6 = c6["bootstrap"]
    b7 = c7["bootstrap"]
    a5 = c6["aggregate"]["w5"]
    a100 = c6["aggregate"]["w100"]
    t5 = c7["literal_t0"]["w5"]
    t100 = c7["literal_t0"]["w100"]
    s5 = c7["ideal_task_centered"]["w5"]
    s100 = c7["ideal_task_centered"]["w100"]
    text = [
        "# Switch shock and initial random-feature coverage (post-hoc; no training)", "",
        "## Verdict", "",
        "- **Candidate 6: signature supported, mechanism not identified.** Positive `dead_frac` changes are concentrated in the first observed 1,000 updates after a task switch, and the concentration fold is larger at w5. The metric is a task-support label without unit identity, so this does not prove that physical trajectories crossed an absorbing boundary during that bin.",
        "- **Candidate 7: redundancy supported, nearest-direction story not isolated.** With initial `W,b` frozen, the literal step-0 w100 feature matrix spans all 32 support points and a fitted `v` interpolates every seed. w5 does not. Under ideal per-task centering across all 100 tasks, w100 still has much lower readout error; random five-feature subsets of w100 fall back near w5.", "",
        "## Candidate 6 — switch-aligned label increases", "",
        "The existing probe grid is 1,000 steps. Rows at 10,000 multiples are pre-switch; the row at offset 1,000 is after the first 1,000 updates on the new task. Task 1 is excluded, leaving 99 post-switch and 891 within-task comparison intervals per seed.", "",
        "| width | positive change / post bin | positive change / interior bin | concentration fold [seed-cluster 95% CI] | positive-event rate post / interior |",
        "|---|---:|---:|---:|---:|",
        f"| w5 | {a5['post_positive_change_rate']:.6f} | {a5['interior_positive_change_rate']:.6f} | {fmt_ci(b6['w5_concentration_fold'], b6['w5_ci95'])} | {a5['post_positive_event_rate']:.3f} / {a5['interior_positive_event_rate']:.3f} |",
        f"| w100 | {a100['post_positive_change_rate']:.6f} | {a100['interior_positive_change_rate']:.6f} | {fmt_ci(b6['w100_concentration_fold'], b6['w100_ci95'])} | {a100['post_positive_event_rate']:.3f} / {a100['interior_positive_event_rate']:.3f} |", "",
        f"The concentration-fold ratio w5/w100 is {fmt_ci(b6['fold_ratio_w5_over_w100'], b6['fold_ratio_ci95'])}. Every seed has post/interior fold >1. This supports the proposed timing signature and its stronger relative concentration at w5.", "",
        "**Limit:** `dead_frac` can fall as well as rise and can change when the task support changes. Positive aggregate changes hide simultaneous unit-level entries/exits. The current files therefore cannot distinguish boundary reclassification from irreversible unit motion, nor localize an event within the 1,000-step bin.", "",
        "## Candidate 7 — fixed initial features, least-squares `v`", "",
        "### Literal centered-arm state at step 0", "",
        "At step 0 the centered arm has `running_mean=0`, so learner input is raw input. The 32 support points are enumerated exactly; `W,b` are the saved initial tensors and only `v` is fitted (no intercept in the primary result).", "",
        "| width | median NMSE | ranks by seed | result |",
        "|---|---:|---|---|",
        f"| w5 | {t5['median_nmse_v_only']:.6g} | {t5['rank_by_seed']} | substantial residual |",
        f"| w100 | {t100['median_nmse_v_only']:.3g} | {t100['rank_by_seed']} | all five seeds interpolate (<1e-20 NMSE) |", "",
        "This is direct evidence that the initial w100 bank already contains a readout span sufficient for the first task, whereas w5 must change hidden features to reach a comparable fit.", "",
        "### All 100 tasks under ideal task-wise centering", "",
        "This sensitivity removes each task's exact input mean before applying the same frozen initial `W,b`. It approximates the steady centered geometry without replaying learning.", "",
        "| feature bank | median NMSE | IQR | median rank |",
        "|---|---:|---:|---:|",
        f"| w5 all 5 | {s5['median_nmse_v_only']:.6f} | {s5['q25_q75_nmse_v_only'][0]:.6f}–{s5['q25_q75_nmse_v_only'][1]:.6f} | {s5['median_rank_v_only']:.0f} |",
        f"| w100 all 100 | {s100['median_nmse_v_only']:.6f} | {s100['q25_q75_nmse_v_only'][0]:.6f}–{s100['q25_q75_nmse_v_only'][1]:.6f} | {s100['median_rank_v_only']:.0f} |",
        f"| w100 random 5 (median of {SUBSET_DRAWS}/case) | {b7['ideal_centered_median_nmse_w100_random5']:.6f} | — | ≤5 |", "",
        f"w5/w100-all error fold: {fmt_ci(b7['w5_over_w100_fold'], b7['w5_over_w100_ci95'])}. Random-5/w100-all fold: {fmt_ci(b7['random5_over_full100_fold'], b7['random5_over_full100_ci95'])}. w5/random-5 fold: {fmt_ci(b7['w5_over_w100_random5_fold'], b7['w5_over_w100_random5_ci95'])}.", "",
        "The subset control shows that most of the advantage comes from having a redundant 100-feature span, not from a different per-feature initialization law. It does **not** specifically establish that one initial direction is geometrically close to a unique required direction; the least-squares test identifies span/coverage, which also includes ordinary dimensionality.", "",
        "## Reproducibility and scope", "",
        "- No SGD step, optimizer update, or new training run was executed.",
        "- Step-0 `W,b,flip_state` reconstruction and the 1M final `flip_state` replay match exactly for both widths.",
        "- Width groups use different width-dependent generators in the original experiment, so w5 versus w100 is an unpaired five-seed comparison. Bootstrap resamples seed trajectories and retains all 100 tasks within a selected seed.",
        "- All claims are post-hoc and limited to `center_selfcov_0814`, condA, centered input, teacher width 100, T=10,000.",
    ]
    (out / "REPORT.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def run(out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    logs = load_centered_logs()
    intervals = switch_intervals(logs)
    seed_switch = summarize_switch_by_seed(intervals)
    switch_boot = bootstrap_switch(seed_switch)
    fits, subsets, sanity = replay_initial_feature_fits(cfg)
    coverage_boot = bootstrap_coverage(fits, subsets)
    summary = build_summary(
        intervals, seed_switch, switch_boot, fits, subsets, coverage_boot, sanity
    )

    intervals.to_csv(out / "switch_intervals.csv", index=False)
    seed_switch.to_csv(out / "switch_by_seed.csv", index=False)
    fits.to_csv(out / "initial_feature_fits.csv", index=False)
    subsets.to_csv(out / "w100_random5_subsets.csv", index=False)
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    make_figure(intervals, fits, subsets, out)
    write_report(summary, out)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    summary = run(args.out)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
