"""Validate the three capacity-regime explanations on existing checkpoints.

This is a read-only, post-hoc analysis.  It never advances an environment or
starts an online training run.  The only optimisation is a static fit to the
32 points in the current condA support; its result is an *achievable upper
bound* on approximation error, not a certified global minimum.

Run from the repository root with::

    OMP_NUM_THREADS=1 .venv/bin/python -m analysis.capacity_regime.capacity_regime
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.ratchet_log import exact_record, full_support_ro, teacher_f64
from src.train import setup_group


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results" / "center_selfcov_0814"
DEFAULT_OUT = ROOT / "results" / "capacity_regime_0830"


def _load_state(width: int, cfg: dict) -> tuple[dict, dict]:
    ckpt = torch.load(
        SOURCE / "ckpts" / f"A_w{width}_step1000000.pt",
        map_location="cpu",
        weights_only=False,
    )
    st = setup_group(("A", width, 1, "none"), ckpt["runs"], cfg, torch.device("cpu"))
    st["net"].load_state(ckpt["net"])
    st["env"].load_state(ckpt["env"])
    st["teacher"].load_state(ckpt["teacher"])
    st["running_mean"].copy_(ckpt["running_mean"])
    return st, ckpt


def _student_view(st: dict) -> dict[str, torch.Tensor]:
    X = full_support_ro(st["env"]).double()
    y = teacher_f64(st["teacher"], X)
    x_in = X - st["centered"][None, :, None].double() * st["running_mean"][None].double()
    net = st["net"]
    W, b = net.W.double(), net.b.double()
    v, c = net.v.double(), net.c.double()
    pre = torch.einsum("rhd,prd->prh", W, x_in) + b
    a = torch.relu(pre)
    yhat = (a * v).sum(dim=-1) + c
    delta = yhat - y
    return dict(X=X, x_in=x_in, y=y, pre=pre, a=a, yhat=yhat, delta=delta)


def _target_on_free_bits(st: dict, run: int) -> tuple[torch.Tensor, torch.Tensor]:
    patterns = st["env"].patterns.double()
    flip = st["env"].flip_state[run].double().expand(len(patterns), -1)
    X = torch.cat([flip, patterns], dim=1)
    teacher = st["teacher"]
    pre = torch.einsum("hm,nm->nh", teacher.W[run].double(), X) + teacher.b[run].double()
    y = ((pre >= teacher.tau[run].double()) * teacher.v[run].double()).sum(dim=1)
    y = y + teacher.cout[run].double()
    return 2.0 * patterns - 1.0, y


def _linear_floor(X: torch.Tensor, y: torch.Tensor) -> float:
    Z = torch.cat([X, torch.ones(len(X), 1, dtype=torch.float64)], dim=1)
    pred = Z @ (torch.linalg.pinv(Z, rcond=1e-12) @ y)
    return float(((pred - y) ** 2).mean())


def _constructive_width31_error(y: torch.Tensor) -> float:
    """Exact 1-D piecewise-linear interpolation on binary code 0..31."""
    slopes = y[1:] - y[:-1]
    code = torch.arange(32, dtype=torch.float64)
    pred = y[0] + slopes[0] * torch.relu(code)
    for knot in range(1, 31):
        pred = pred + (slopes[knot] - slopes[knot - 1]) * torch.relu(code - knot)
    return float((pred - y).abs().max())


def _fit_relu_upper(
    X: torch.Tensor,
    y: torch.Tensor,
    width: int,
    restarts: int,
    steps: int,
    seed: int,
) -> float:
    """Return the best achieved MSE across independent Adam restarts."""
    if width == 0:
        return float(y.var(unbiased=False))
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    scale = y.std(unbiased=False).clamp_min(1e-12)
    yn = (y - y.mean()) / scale
    d = X.shape[1]
    W = (torch.randn(restarts, width, d, generator=gen, dtype=torch.float64)
         * math.sqrt(2.0 / d)).requires_grad_()
    b = torch.zeros(restarts, width, dtype=torch.float64, requires_grad=True)
    v = (torch.randn(restarts, width, generator=gen, dtype=torch.float64)
         * math.sqrt(1.0 / width)).requires_grad_()
    c = torch.zeros(restarts, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([W, b, v, c], lr=0.03)
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        a = torch.relu(torch.einsum("rhd,nd->rnh", W, X) + b[:, None, :])
        pred = (a * v[:, None, :]).sum(dim=2) + c[:, None]
        loss = ((pred - yn) ** 2).mean(dim=1)
        loss.sum().backward()
        opt.step()
        if step + 1 == steps // 2:
            for group in opt.param_groups:
                group["lr"] = 0.01
    with torch.no_grad():
        a = torch.relu(torch.einsum("rhd,nd->rnh", W, X) + b[:, None, :])
        pred = (a * v[:, None, :]).sum(dim=2) + c[:, None]
        loss = ((pred - yn) ** 2).mean(dim=1) * scale ** 2
    return float(loss.min())


def _refit_mse(a: torch.Tensor, y: torch.Tensor) -> float:
    Z = torch.cat([a, torch.ones(len(a), 1, dtype=torch.float64)], dim=1)
    pred = Z @ (torch.linalg.pinv(Z, rcond=1e-12) @ y)
    return float(((pred - y) ** 2).mean())


def candidate12(
    st: dict,
    ckpt: dict,
    restarts: int,
    oracle_restarts: int,
    steps: int,
) -> pd.DataFrame:
    view = _student_view(st)
    rows = []
    for run in range(5, 10):  # the five centered rows
        seed = int(ckpt["runs"][run]["seed"])
        X, y = _target_on_free_bits(st, run)
        pre = view["pre"][:, run]
        a = view["a"][:, run]
        p_hat = (pre > 0).double().mean(dim=0)
        alive = p_hat >= 0.05
        observed = float((view["delta"][:, run] ** 2).mean())
        alive_only = (a[:, alive] * st["net"].v[run, alive].double()).sum(dim=1)
        alive_only = alive_only + st["net"].c[run].double()
        width5 = min(
            _fit_relu_upper(X, y, 5, restarts, steps, 20260830 + seed),
            _fit_relu_upper(X, y, 5, restarts, steps, 20360830 + seed),
        )
        alive_upper = _fit_relu_upper(
            X, y, int(alive.sum()), oracle_restarts, steps,
            20260830 + 100 * seed + int(alive.sum()),
        )
        rows.append(dict(
            seed=seed,
            alive=int(alive.sum()),
            observed_mse=observed,
            alive_only_mse=float(((alive_only - y) ** 2).mean()),
            refit_alive_mse=_refit_mse(a[:, alive], y),
            constant_floor=float(y.var(unbiased=False)),
            linear_floor=_linear_floor(X, y),
            width5_achieved_upper=width5,
            alive_width_achieved_upper=alive_upper,
            excess_over_alive_upper=observed - alive_upper,
            width31_max_abs_error=_constructive_width31_error(y),
        ))
    return pd.DataFrame(rows)


def _spearman(x: np.ndarray, y: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is not None:
        x, y = x[mask], y[mask]
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(pd.Series(x).rank().corr(pd.Series(y).rank()))


def candidate3(st: dict, ckpt: dict) -> pd.DataFrame:
    rec = exact_record(st, as_f64=True)
    view = _student_view(st)
    x_in, y = view["x_in"], view["y"]
    pre, a, delta = view["pre"], view["a"], view["delta"]
    v = st["net"].v.double()
    gate = (pre > 0).double()
    gW = (2.0 * delta[:, :, None, None] * v[None, :, :, None]
          * gate[:, :, :, None] * x_in[:, :, None, :])
    drift = gW.mean(dim=0).pow(2).sum(dim=(1, 2))
    noise = gW.var(dim=0, unbiased=False).sum(dim=(1, 2))
    yvar = y.var(dim=0, unbiased=False)
    q = a * v[None]
    utility = (q * q - 2.0 * delta[:, :, None] * q).mean(dim=0)
    utility = utility / yvar[:, None].clamp_min(1e-12)
    rows = []
    for run, run_cfg in enumerate(ckpt["runs"]):
        fs = np.asarray(rec["F_self"][run])
        fr = np.asarray(rec["F_rest"][run])
        fg = np.asarray(rec["F_gate"][run])
        denom = np.abs(fs) + np.abs(fr)
        valid = denom > 1e-15
        cancellation = (1.0 - np.abs(fg[valid]) / denom[valid]
                        if valid.any() else np.asarray([np.nan]))
        p_hat = np.asarray(rec["p_hat"][run])
        alive = p_hat >= 0.05
        resid = delta[:, run].numpy()
        util = utility[run].numpy()
        mse = float((delta[:, run] ** 2).mean())
        signal = float(yvar[run])
        rows.append(dict(
            width=int(run_cfg["width"]), enc=run_cfg["enc"], seed=int(run_cfg["seed"]),
            alive=int(alive.sum()), mse=mse, signal_var=signal,
            nmse=mse / max(signal, 1e-12),
            unfit=float(np.var(resid) / max(signal, 1e-12)),
            residual_mean=float(np.mean(resid)),
            cancellation_median=(float(np.median(cancellation))
                                 if valid.any() else float("nan")),
            balance_abs_self_over_rest=float(
                np.median(np.abs(fs[valid]) / np.maximum(np.abs(fr[valid]), 1e-300))
            ) if valid.any() else float("nan"),
            gradient_snr=float(drift[run] / noise[run].clamp_min(1e-30)),
            rho_utility_Fgate=_spearman(util, fg),
            rho_utility_Fgate_alive=_spearman(util, fg, alive),
        ))
    return pd.DataFrame(rows)


def _summary(floors: pd.DataFrame, signatures: pd.DataFrame) -> str:
    centered = signatures[signatures.enc == "centered"]
    arm = centered.groupby("width").agg(
        alive=("alive", "mean"), mse=("mse", "mean"), unfit=("unfit", "median"),
        cancellation=("cancellation_median", "median"),
        gradient_snr=("gradient_snr", "median"),
    )
    table = [
        "| width | mean alive | mean MSE | median unfit | median cancellation | median gradient SNR |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for width, row in arm.iterrows():
        table.append(
            f"| {int(width)} | {row.alive:.3g} | {row.mse:.6g} | {row.unfit:.6g} | "
            f"{row.cancellation:.6g} | {row.gradient_snr:.6g} |"
        )
    lines = [
        "# capacity_regime_0830 — post-hoc validation",
        "",
        "> Existing final checkpoints only; no new online learning run. Static ReLU fits are",
        "> achievable upper bounds from multistart Adam, not certified global minima.",
        "",
        "## Candidate 1 — native width-5 approximation floor",
        "",
        "**REJECTED.** On the five exact final w5-centered tasks, the best achieved width-5",
        f"MSE has median {floors.width5_achieved_upper.median():.6g} and maximum "
        f"{floors.width5_achieved_upper.max():.6g}, versus observed median "
        f"{floors.observed_mse.median():.6g}.  A width-31 constructive interpolant has maximum",
        f"absolute error {floors.width31_max_abs_error.max():.3g}, hence width 100 has exact",
        "zero approximation error on the 32-point support.",
        "",
        "## Candidate 2 — absolute alive count / effective capacity",
        "",
        "**PARTIALLY SUPPORTED.** Final alive counts are "
        f"{floors.alive.tolist()} (mean {floors.alive.mean():.1f}).  Dead-unit removal changes",
        "no prediction on this support, so these counts are the exact active feature counts here.",
        "However, the observed loss exceeds the best achieved network of the same alive width by",
        f"{floors.excess_over_alive_upper.tolist()}; count alone is insufficient, especially for",
        "the 2- and 3-alive runs.  Feature placement/conditioning remains part of the failure.",
        "",
        "## Candidate 3 — large residual changes the force-field phase",
        "",
        "**RESIDUAL PREMISE CONFIRMED; PREDICTED SIGNATURE NOT SUPPORTED.**",
        "",
        *table,
        "",
        "The w5-centered residual is large, but its self/rest cancellation is not weaker and its",
        "gradient SNR is smaller than w100-centered.  Thus large residual alone does not imply a",
        "drift-dominated, function-seeing phase in these final static snapshots.  A time-resolved",
        "w5 logger would still be required to test the exact temporal rho used in the pillar.",
        "",
        "## Overall",
        "",
        "The supported *proximate description* is an **acquired effective-capacity collapse**,",
        "not native width-5 inexpressivity: width 5 can fit the current task, but only 0–3 active",
        "features remain and, when 2–3 survive, their geometry is materially worse than an",
        "alive-matched static oracle.  This does not yet explain why centering fails to prevent",
        "that collapse at width 5.  The mechanism therefore remains unresolved, and Candidate 3",
        "should not be promoted without a dedicated temporal test.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--restarts", type=int, default=256)
    parser.add_argument("--oracle-restarts", type=int, default=512)
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--summary-only", action="store_true",
                        help="rebuild summary.md from existing CSV outputs")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.summary_only:
        floors = pd.read_csv(args.out / "task_floors.csv")
        signatures = pd.read_csv(args.out / "final_signatures.csv")
        summary = _summary(floors, signatures)
        (args.out / "summary.md").write_text(summary, encoding="utf-8")
        print(summary)
        return
    cfg = yaml.safe_load((SOURCE / "config_used.yaml").read_text())
    st5, ck5 = _load_state(5, cfg)
    st100, ck100 = _load_state(100, cfg)
    floors = candidate12(st5, ck5, args.restarts, args.oracle_restarts, args.steps)
    signatures = pd.concat([candidate3(st5, ck5), candidate3(st100, ck100)], ignore_index=True)
    args.out.mkdir(parents=True, exist_ok=True)
    floors.to_csv(args.out / "task_floors.csv", index=False)
    signatures.to_csv(args.out / "final_signatures.csv", index=False)
    (args.out / "summary.md").write_text(_summary(floors, signatures), encoding="utf-8")
    print(_summary(floors, signatures))


if __name__ == "__main__":
    main()
