"""Test whether residual structure recreates a nonzero mean force after centering.

This reads existing checkpoints and 1k-step CSV logs.  The checkpoint analysis
uses the exact 32-point condA support; it does not train or advance an environment.

Run from the repository root::

    OMP_NUM_THREADS=1 .venv/bin/python -m analysis.residual_force_recreation.analyze
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
DEFAULT_OUT = ROOT / "results" / "residual_force_recreation_0830"
WIDTHS = (5, 100)
STEPS = (0, 10_000, 50_000, 100_000, 300_000, 1_000_000)
BOOTSTRAP_B = 20_000


def load_config() -> dict:
    with (SOURCE / "config_used.yaml").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_state(width: int, step: int, cfg: dict) -> tuple[dict, dict]:
    ckpt = torch.load(
        SOURCE / "ckpts" / f"A_w{width}_step{step}.pt",
        map_location="cpu", weights_only=False,
    )
    st = setup_group(("A", width, 1, "none"), ckpt["runs"], cfg, torch.device("cpu"))
    st["net"].load_state(ckpt["net"])
    st["env"].load_state(ckpt["env"])
    st["teacher"].load_state(ckpt["teacher"])
    st["running_mean"].copy_(ckpt["running_mean"])
    return st, ckpt


def exact_checkpoint_rows(width: int, step: int, cfg: dict) -> list[dict]:
    st, ckpt = load_state(width, step, cfg)
    X = full_support_ro(st["env"]).double()
    y = teacher_f64(st["teacher"], X)
    x_in = X - st["centered"][None, :, None].double() * st["running_mean"][None].double()
    net = st["net"]
    W, b = net.W.double(), net.b.double()
    v, c = net.v.double(), net.c.double()
    pre = torch.einsum("rhd,prd->prh", W, x_in) + b
    gate = (pre > 0).double()
    activation = torch.relu(pre)
    prediction = (activation * v).sum(dim=2) + c
    delta = prediction - y

    mu = x_in.mean(dim=0)
    mean_delta = delta.mean(dim=0)
    G = (delta[:, :, None] * x_in).mean(dim=0)  # E[delta x]
    mean_term = mean_delta[:, None] * mu
    covariance_term = G - mean_term

    # Candidate-5 proxy: E[delta v_i x], without the ReLU gate.
    ungated_force = v[:, :, None] * G[:, None, :]
    # Actual half-gradient: E[delta v_i gate_i x].  Multiplying by 2 gives
    # exactly the W gradient used by training and by CSV drift_sq_W.
    gated_force = (
        delta[:, :, None, None]
        * v[None, :, :, None]
        * gate[:, :, :, None]
        * x_in[:, :, None, :]
    ).mean(dim=0)
    bias_force = (delta[:, :, None] * v[None] * gate).mean(dim=0)
    p_hat = gate.mean(dim=0)
    signal_var = y.var(dim=0, unbiased=False)
    residual_var = delta.var(dim=0, unbiased=False)
    input_trace_cov = x_in.var(dim=0, unbiased=False).sum(dim=1)

    rows = []
    for run_index, run in enumerate(ckpt["runs"]):
        if run["enc"] != "centered":
            continue
        g2 = float(G[run_index].square().sum())
        cov2 = float(covariance_term[run_index].square().sum())
        mse = float(delta[:, run_index].square().mean())
        tr_x = float(input_trace_cov[run_index])
        rows.append(dict(
            width=width, step=step, seed=int(run["seed"]),
            dead_frac_exact=float((p_hat[run_index] < 0.05).double().mean()),
            mse=mse, signal_var=float(signal_var[run_index]),
            nmse=mse / max(float(signal_var[run_index]), 1e-15),
            residual_var=float(residual_var[run_index]),
            mu_norm=float(mu[run_index].norm()),
            mean_delta=float(mean_delta[run_index]),
            G_norm=float(G[run_index].norm()),
            covariance_term_norm=float(covariance_term[run_index].norm()),
            mean_input_term_norm=float(mean_term[run_index].norm()),
            covariance_share_norm=(
                float(covariance_term[run_index].norm())
                / max(float(covariance_term[run_index].norm()
                            + mean_term[run_index].norm()), 1e-15)
            ),
            residual_input_coherence=g2 / max(mse * tr_x, 1e-15),
            covariance_coherence=cov2 / max(float(residual_var[run_index]) * tr_x, 1e-15),
            ungated_force_sq_total=float(ungated_force[run_index].square().sum()),
            ungated_force_sq_per_unit=(
                float(ungated_force[run_index].square().sum()) / width
            ),
            gated_force_sq_total=float(gated_force[run_index].square().sum()),
            gated_force_sq_per_unit=float(gated_force[run_index].square().sum()) / width,
            bias_force_sq_per_unit=float(bias_force[run_index].square().sum()) / width,
            drift_sq_W_exact=4.0 * float(gated_force[run_index].square().sum()),
        ))
    return rows


def checkpoint_table(cfg: dict) -> pd.DataFrame:
    rows = []
    for width in WIDTHS:
        for step in STEPS:
            rows.extend(exact_checkpoint_rows(width, step, cfg))
    return pd.DataFrame(rows)


def load_centered_csv() -> pd.DataFrame:
    frames = []
    for width in WIDTHS:
        data = pd.read_csv(SOURCE / f"lop_metrics_A_w{width}.csv")
        data = data[data.run_id.str.contains("_centered_")].copy()
        data["width"] = width
        data["seed"] = data.run_id.str.extract(r"_s(\d+)$")[0].astype(int)
        frames.append(data)
    return pd.concat(frames, ignore_index=True)


def validate_exact_against_csv(exact: pd.DataFrame, csv: pd.DataFrame) -> pd.DataFrame:
    logged = csv[csv.step.isin(STEPS)][
        ["width", "seed", "step", "drift_sq_W", "dead_frac", "eval_loss", "snr_drift"]
    ]
    merged = exact.merge(logged, on=["width", "seed", "step"], validate="one_to_one")
    merged["drift_log10_error"] = np.abs(
        np.log10(merged.drift_sq_W_exact.clip(lower=1e-30))
        - np.log10(merged.drift_sq_W.clip(lower=1e-30))
    )
    return merged


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(pd.Series(x).rank().corr(pd.Series(y).rank()))


def event_analysis(csv: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    event_rows = []
    onset_rows = []
    for (width, seed), group in csv.groupby(["width", "seed"], sort=True):
        group = group.sort_values("step").copy()
        group["prev_step"] = group.step.shift()
        group["prev_dead"] = group.dead_frac.shift()
        group["prev_drift"] = group.drift_sq_W.shift()
        group["dead_change"] = group.dead_frac.diff()
        group["task"] = np.ceil(group.step / 10_000).astype(int)
        group["prev_task"] = np.ceil(group.prev_step / 10_000)
        usable = group[
            (group.step > 10_000)
            & (group.task == group.prev_task)
            & (group.prev_drift > 0)
        ].copy()
        event = usable[usable.dead_change > 0]
        nonevent = usable[usable.dead_change <= 0]
        event_log = np.log10(event.prev_drift.to_numpy(float))
        nonevent_log = np.log10(nonevent.prev_drift.to_numpy(float))
        event_rows.append(dict(
            width=int(width), seed=int(seed), n_interval=len(usable),
            n_dead_rise=len(event),
            spearman_prev_drift_vs_dead_change=spearman(
                np.log10(usable.prev_drift.to_numpy(float)),
                usable.dead_change.to_numpy(float),
            ),
            event_minus_nonevent_median_log10_drift=(
                float(np.median(event_log) - np.median(nonevent_log))
                if len(event_log) and len(nonevent_log) else float("nan")
            ),
        ))

        after = group[(group.step >= 10_000) & (group.dead_frac > 0)]
        if len(after):
            onset = after.iloc[0]
            previous = group[group.step < onset.step].iloc[-1]
            onset_rows.append(dict(
                width=int(width), seed=int(seed), onset_step=int(onset.step),
                previous_step=int(previous.step),
                previous_dead=float(previous.dead_frac), onset_dead=float(onset.dead_frac),
                previous_drift_sq_W=float(previous.drift_sq_W),
                onset_drift_sq_W=float(onset.drift_sq_W),
                same_task_as_previous=(
                    int(np.ceil(onset.step / 10_000))
                    == int(np.ceil(previous.step / 10_000))
                ),
            ))
    return pd.DataFrame(event_rows), pd.DataFrame(onset_rows)


def bootstrap_log_ratio(step10: pd.DataFrame, metric: str, seed: int) -> dict:
    a = step10[step10.width == 5][metric].to_numpy(float)
    b = step10[step10.width == 100][metric].to_numpy(float)
    point = float(np.median(np.log10(a)) - np.median(np.log10(b)))
    rng = np.random.default_rng(seed)
    da = np.median(np.log10(a[rng.integers(0, len(a), (BOOTSTRAP_B, len(a)))]), axis=1)
    db = np.median(np.log10(b[rng.integers(0, len(b), (BOOTSTRAP_B, len(b)))]), axis=1)
    lo, hi = np.quantile(da - db, [0.025, 0.975])
    return dict(
        metric=metric, median_w5=float(np.median(a)), median_w100=float(np.median(b)),
        fold_w5_over_w100=10.0 ** point,
        fold_ci95=[10.0 ** float(lo), 10.0 ** float(hi)],
        bootstrap_B=BOOTSTRAP_B,
    )


def summarize(
    exact: pd.DataFrame, validation: pd.DataFrame, events: pd.DataFrame,
    onsets: pd.DataFrame, csv: pd.DataFrame,
) -> dict:
    step10 = exact[exact.step == 10_000]
    if not np.allclose(step10.dead_frac_exact, 0):
        raise RuntimeError("task-1 checkpoint is not all-alive")
    metrics = [
        "mse", "nmse", "mu_norm", "G_norm", "covariance_term_norm", "mean_input_term_norm",
        "ungated_force_sq_per_unit", "gated_force_sq_per_unit",
        "residual_input_coherence",
    ]
    contrasts = {
        metric: bootstrap_log_ratio(step10, metric, 20260830 + i)
        for i, metric in enumerate(metrics)
    }
    final_dead = (csv[csv.step == 1_000_000]
                  [["width", "seed", "dead_frac"]])
    prospective = step10.merge(final_dead, on=["width", "seed"], validate="one_to_one")
    w5 = prospective[prospective.width == 5]
    event_medians = events.groupby("width").agg(
        median_spearman=("spearman_prev_drift_vs_dead_change", "median"),
        median_event_log10_difference=("event_minus_nonevent_median_log10_drift", "median"),
        total_dead_rises=("n_dead_rise", "sum"),
    )
    csv10 = csv[csv.step == 10_000]
    summary = dict(
        step10_all_dead_frac_zero=True,
        step10_contrasts=contrasts,
        step10_covariance_share_norm_median_w5=float(
            step10[step10.width == 5].covariance_share_norm.median()
        ),
        step10_covariance_share_norm_median_w100=float(
            step10[step10.width == 100].covariance_share_norm.median()
        ),
        step10_snr_drift_median_w5=float(csv10[csv10.width == 5].snr_drift.median()),
        step10_snr_drift_median_w100=float(csv10[csv10.width == 100].snr_drift.median()),
        step10_w5_spearman_ungated_force_vs_final_dead=spearman(
            w5.ungated_force_sq_per_unit.to_numpy(float), w5.dead_frac.to_numpy(float)
        ),
        step10_w5_spearman_gated_force_vs_final_dead=spearman(
            w5.gated_force_sq_per_unit.to_numpy(float), w5.dead_frac.to_numpy(float)
        ),
        exact_vs_csv_median_abs_log10_drift_error=float(
            validation[validation.step > 0].drift_log10_error.median()
        ),
        event_analysis={
            f"w{width}": {key: float(value) for key, value in row.items()}
            for width, row in event_medians.to_dict(orient="index").items()
        },
        onset_same_task_preobservation_count={
            f"w{width}": int(group.same_task_as_previous.sum())
            for width, group in onsets.groupby("width")
        },
    )
    return summary


def plot(exact: pd.DataFrame, events: pd.DataFrame, out: Path) -> None:
    colors = {5: "#bf616a", 100: "#5e81ac"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.6))
    step10 = exact[exact.step == 10_000]
    for j, metric in enumerate(("ungated_force_sq_per_unit", "gated_force_sq_per_unit")):
        ax = axes[0, j]
        for xi, width in enumerate(WIDTHS):
            values = step10[step10.width == width][metric].to_numpy(float)
            ax.scatter(np.full(len(values), xi), values, color=colors[width], s=50)
            ax.plot([xi - .18, xi + .18], [np.median(values)] * 2, color="black", lw=2)
        ax.set_yscale("log")
        ax.set_xticks([0, 1], ["w5", "w100"])
        ax.set_ylabel(metric.replace("_", " "))
        ax.set_title("task 1 end; all units alive")

    ax = axes[1, 0]
    for width in WIDTHS:
        group = exact[(exact.width == width) & (exact.step > 0)]
        pivot = group.pivot(index="step", columns="seed", values="G_norm")
        ax.plot(pivot.index / 1000, pivot.median(axis=1), marker="o",
                color=colors[width], label=f"w{width}")
        ax.fill_between(pivot.index / 1000, pivot.quantile(.25, axis=1),
                        pivot.quantile(.75, axis=1), color=colors[width], alpha=.18)
    ax.set_yscale("log")
    ax.set_xlabel("step (thousands)")
    ax.set_ylabel(r"$\|E[\delta x_{in}]\|$")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    for xi, width in enumerate(WIDTHS):
        values = events[events.width == width].event_minus_nonevent_median_log10_drift
        ax.scatter(np.full(len(values), xi), values, color=colors[width], s=50)
        ax.plot([xi - .18, xi + .18], [np.nanmedian(values)] * 2, color="black", lw=2)
    ax.axhline(0, color="#777777", lw=.8)
    ax.set_xticks([0, 1], ["w5", "w100"])
    ax.set_ylabel("pre-event log10 drift: rise - no rise")
    ax.set_title("same-task 1k intervals")
    fig.suptitle("Residual-induced mean force after input centering", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "residual_force_recreation.png", dpi=180)
    fig.savefig(out / "residual_force_recreation.pdf")
    plt.close(fig)


def write_report(exact: pd.DataFrame, events: pd.DataFrame, summary: dict, out: Path) -> None:
    c = summary["step10_contrasts"]
    lines = [
        "# Residual-induced force after centering (post-hoc; no training)", "",
        "## Verdict", "",
        "- **Cross-moment regeneration is confirmed:** after input centering, E[delta x] remains nonzero and is dominated by Cov(delta, x), not by residual input mean.",
        "- **Causal killing is not established:** force magnitude has no robust 1k-step event association with increases in the aggregate, task-dependent dead_frac label.",
        "- This is analogous to layer-2 mean regeneration, but not mathematically identical: the recreated object is a residual-input cross-moment (mean force), not the input mean itself.", "",
        "## Clean pre-death checkpoint: step 10k", "",
        "All centered runs have exact dead_frac=0 here.", "",
        "| metric | w5 median | w100 median | w5/w100 fold [95% CI] |",
        "|---|---:|---:|---:|",
    ]
    for metric in ("nmse", "mu_norm", "G_norm", "covariance_term_norm",
                   "mean_input_term_norm", "ungated_force_sq_per_unit",
                   "gated_force_sq_per_unit", "residual_input_coherence"):
        row = c[metric]
        lines.append(
            f"| {metric} | {row['median_w5']:.6g} | {row['median_w100']:.6g} | "
            f"{row['fold_w5_over_w100']:.2f} [{row['fold_ci95'][0]:.2f}, {row['fold_ci95'][1]:.2f}] |"
        )
    lines += ["",
        f"Covariance share ||Cov(delta,x)||/(||Cov||+||Edelta mu||): w5={summary['step10_covariance_share_norm_median_w5']:.3f}, w100={summary['step10_covariance_share_norm_median_w100']:.3f}.",
        f"CSV gradient SNR at the same point: w5={summary['step10_snr_drift_median_w5']:.4f}, w100={summary['step10_snr_drift_median_w100']:.4f}. The mean force is larger per unit at w5 but less dominant relative to per-sample gradient variance. Residual-input coherence is also lower at w5, so the force increase is primarily an absolute residual-scale effect, not stronger normalized directionality.", "",
        "## Does it predict death?", "",
        f"- At w5, Spearman(step-10k force, final dead_frac): ungated={summary['step10_w5_spearman_ungated_force_vs_final_dead']:.3f}, gated={summary['step10_w5_spearman_gated_force_vs_final_dead']:.3f} (n=5; descriptive).",
        f"- Same-task interval Spearman(previous drift, next 1k net dead change), seed median: w5={summary['event_analysis']['w5']['median_spearman']:.3f}, w100={summary['event_analysis']['w100']['median_spearman']:.3f}.",
        f"- Only {summary['onset_same_task_preobservation_count']['w5']}/5 w5 first onsets have a 1k observation in the same task immediately before detection; the other onsets occur at the first probe after a task switch.", "",
        "The logs therefore confirm the proposed nonzero force before final gate closure, but they do not show that this force points toward the gate-closing boundary. A unit-ID logger must record the signed force projection onto each unit's gate margin immediately before onset.",
    ]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    cfg = load_config()
    exact = checkpoint_table(cfg)
    csv = load_centered_csv()
    validation = validate_exact_against_csv(exact, csv)
    events, onsets = event_analysis(csv)
    summary = summarize(exact, validation, events, onsets, csv)
    exact.to_csv(args.out / "checkpoint_exact_force.csv", index=False)
    validation.to_csv(args.out / "checkpoint_csv_validation.csv", index=False)
    events.to_csv(args.out / "event_association_by_seed.csv", index=False)
    onsets.to_csv(args.out / "first_dead_onsets.csv", index=False)
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    plot(exact, events, args.out)
    write_report(exact, events, summary, args.out)


if __name__ == "__main__":
    main()
