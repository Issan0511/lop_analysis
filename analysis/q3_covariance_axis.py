"""Q3 centered の共分散主空間仮説を既存 checkpoint / probe log で検証する。

これは `specs/spec_q3_covariance_axis_posthoc_0822.md` に固定した、結果観察後の
探索解析である。新しい学習走は行わない。

実行:
  OMP_NUM_THREADS=1 .venv/bin/python -m analysis.q3_covariance_axis
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from src.envs import SCREnv
from src.train import make_gens


ROOT = Path(__file__).resolve().parents[1]
STD_DIR = ROOT / "results" / "ratchet_log_0819"
CENTERED_DIR = ROOT / "results" / "ratchet_centered_0822"
DEFAULT_OUT = CENTERED_DIR / "exploratory_covaxis"

D = 20
F = 15
TOP_DIM = D - F
RANDOM_FLOOR = TOP_DIM / D
FINAL_STEP = 1_000_000
P_HAT_TAU = 0.05
BOOTSTRAP_B = 10_000
BOOTSTRAP_SEED = 20260822


def _np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy().astype(np.float64, copy=False)


def support_patterns(n_bits: int = TOP_DIM) -> np.ndarray:
    ids = np.arange(2**n_bits, dtype=np.int64)[:, None]
    shifts = np.arange(n_bits, dtype=np.int64)[None, :]
    return ((ids >> shifts) & 1).astype(np.float64)


def top_projection(v: np.ndarray) -> np.ndarray:
    """S_max=span(last five coordinates) への射影エネルギー比。"""
    v = np.asarray(v, dtype=np.float64)
    den = np.sum(v * v, axis=-1)
    num = np.sum(v[..., F:] ** 2, axis=-1)
    return np.divide(num, den, out=np.full_like(den, np.nan), where=den > 1e-30)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return np.nan
    rx = pd.Series(x[ok]).rank(method="average").to_numpy(dtype=np.float64)
    ry = pd.Series(y[ok]).rank(method="average").to_numpy(dtype=np.float64)
    if np.std(rx) < 1e-15 or np.std(ry) < 1e-15:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def axis_metrics(W: np.ndarray, mu: np.ndarray, alive: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    wn = np.linalg.norm(W, axis=1)
    U = np.divide(W, wn[:, None], out=np.zeros_like(W), where=wn[:, None] > 1e-30)
    for tag, base in (("e1W", W), ("e1U", U)):
        for subset, mask in (("all", np.ones(len(W), dtype=bool)), ("alive", alive)):
            A = base[mask]
            prefix = f"{tag}_{subset}"
            out[f"{prefix}_n"] = float(len(A))
            if len(A) < 2 or not np.isfinite(A).all():
                out[f"{prefix}_top_proj"] = np.nan
                out[f"{prefix}_mu_abs_cos"] = np.nan
                out[f"{prefix}_top1_frac"] = np.nan
                continue
            _, s, vt = np.linalg.svd(A, full_matrices=False)
            e1 = vt[0]
            out[f"{prefix}_top_proj"] = float(top_projection(e1))
            nmu = float(np.linalg.norm(mu))
            out[f"{prefix}_mu_abs_cos"] = (
                float(abs(e1 @ mu) / nmu) if nmu > 1e-30 else np.nan
            )
            out[f"{prefix}_top1_frac"] = float(s[0] ** 2 / max(np.sum(s * s), 1e-30))
    return out


def load_checkpoint_metrics(arm: str, result_dir: Path, centered: bool, step: int):
    ckpt_path = result_dir / "ckpts" / f"A_w100_step{step}.pt"
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    W_all = _np(ck["net"]["W"])
    b_all = _np(ck["net"]["b"])
    rm_all = _np(ck["running_mean"])
    flip_all = _np(ck["env"]["flip_state"])
    patterns = support_patterns()
    sigma_expected = np.diag(np.r_[np.zeros(F), np.full(TOP_DIM, 0.25)])

    unit_rows: list[dict] = []
    seed_rows: list[dict] = []
    for r, run in enumerate(ck["runs"]):
        seed = int(run["seed"])
        raw = np.concatenate(
            [np.broadcast_to(flip_all[r], (len(patterns), F)), patterns], axis=1
        )
        x_in = raw - rm_all[r][None, :] if centered else raw
        mu = x_in.mean(axis=0)
        xc = x_in - mu[None, :]
        sigma_emp = xc.T @ xc / len(x_in)
        cov_err = float(np.max(np.abs(sigma_emp - sigma_expected)))
        if cov_err > 1e-12:
            raise AssertionError(f"{arm} step={step} seed={seed}: Sigma mismatch {cov_err}")

        W = W_all[r]
        b = b_all[r]
        wnorm = np.linalg.norm(W, axis=1)
        pre = x_in @ W.T + b[None, :]
        p_hat = (pre > 0).mean(axis=0)
        alive = p_hat >= P_HAT_TAU
        q_w = top_projection(W)
        nmu = float(np.linalg.norm(mu))
        w_dot_mu = W @ mu
        cos = np.divide(
            w_dot_mu,
            wnorm * nmu,
            out=np.full(len(W), np.nan),
            where=(wnorm * nmu) > 1e-30,
        )
        sigma_z = 0.5 * np.linalg.norm(W[:, F:], axis=1)
        mean_pre = w_dot_mu + b
        beta = np.divide(
            mean_pre,
            sigma_z,
            out=np.full(len(W), np.nan),
            where=sigma_z > 1e-30,
        )

        for i in range(len(W)):
            unit_rows.append(
                dict(
                    arm=arm,
                    step=step,
                    seed=seed,
                    unit=i,
                    alive=bool(alive[i]),
                    p_hat=float(p_hat[i]),
                    w_norm=float(wnorm[i]),
                    top_projection=float(q_w[i]),
                    cos_u_mu=float(cos[i]),
                    w_dot_mu=float(w_dot_mu[i]),
                    bias=float(b[i]),
                    mean_pre=float(mean_pre[i]),
                    sigma_pre=float(sigma_z[i]),
                    beta=float(beta[i]),
                )
            )

        row = dict(
            arm=arm,
            step=step,
            seed=seed,
            n_alive=int(alive.sum()),
            dead_frac=float(1.0 - alive.mean()),
            mu_norm=nmu,
            mu_top_projection=float(top_projection(mu)),
            sigma_max=0.25,
            sigma_top_dim=TOP_DIM,
            sigma_cov_max_error=cov_err,
            unit_top_proj_mean_all=float(np.nanmean(q_w)),
            unit_top_proj_median_all=float(np.nanmedian(q_w)),
            unit_top_proj_mean_alive=float(np.nanmean(q_w[alive])) if alive.any() else np.nan,
            unit_top_proj_median_alive=float(np.nanmedian(q_w[alive])) if alive.any() else np.nan,
            rho_p_cos=spearman(p_hat, cos),
            rho_p_meanpre=spearman(p_hat, mean_pre),
            rho_p_beta=spearman(p_hat, beta),
            median_abs_w_dot_mu=float(np.median(np.abs(w_dot_mu))),
            median_abs_bias=float(np.median(np.abs(b))),
            median_sigma_pre=float(np.median(sigma_z)),
        )
        row.update(axis_metrics(W, mu, alive))
        seed_rows.append(row)
    return unit_rows, seed_rows, ck


def paired_bootstrap(
    left: pd.DataFrame,
    right: pd.DataFrame,
    metric_left: str,
    metric_right: str,
    rng: np.random.Generator,
    B: int,
) -> tuple[float, float, float, int]:
    a = left.set_index("seed")[metric_left]
    b = right.set_index("seed")[metric_right]
    common = a.index.intersection(b.index)
    diff = (a.loc[common] - b.loc[common]).to_numpy(dtype=np.float64)
    diff = diff[np.isfinite(diff)]
    if len(diff) == 0:
        return np.nan, np.nan, np.nan, 0
    draws = rng.integers(0, len(diff), size=(B, len(diff)))
    boot = diff[draws].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(diff.mean()), float(lo), float(hi), int(len(diff))


def make_comparisons(seed_df: pd.DataFrame, B: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    c0 = seed_df[(seed_df.arm == "centered") & (seed_df.step == 0)]
    c1 = seed_df[(seed_df.arm == "centered") & (seed_df.step == FINAL_STEP)]
    s1 = seed_df[(seed_df.arm == "std") & (seed_df.step == FINAL_STEP)]
    specs = [
        ("centered_final_minus_initial", c1, c0, "unit_top_proj_mean_all", "unit_top_proj_mean_all"),
        ("centered_final_minus_std_final", c1, s1, "unit_top_proj_mean_all", "unit_top_proj_mean_all"),
        ("centered_final_minus_initial_alive", c1, c0, "unit_top_proj_mean_alive", "unit_top_proj_mean_alive"),
        ("centered_final_minus_std_final_alive", c1, s1, "unit_top_proj_mean_alive", "unit_top_proj_mean_alive"),
    ]
    for axis in ("e1W_all_top_proj", "e1U_all_top_proj", "e1W_alive_top_proj", "e1U_alive_top_proj"):
        specs.append((f"{axis}:centered_final_minus_initial", c1, c0, axis, axis))
        specs.append((f"{axis}:centered_final_minus_std_final", c1, s1, axis, axis))
    specs.append(("centered_final_rho_beta_minus_cos", c1, c1, "rho_p_beta", "rho_p_cos"))

    rows = []
    for contrast, left, right, ml, mr in specs:
        est, lo, hi, n = paired_bootstrap(left, right, ml, mr, rng, B)
        rows.append(
            dict(
                contrast=contrast,
                left_metric=ml,
                right_metric=mr,
                estimate=est,
                ci_low=lo,
                ci_high=hi,
                n_seed=n,
                bootstrap_B=B,
                bootstrap_seed=seed,
            )
        )
    return pd.DataFrame(rows)


def replay_centered_mu(centered_dir: Path, centered_ckpt: dict, cfg_path: Path):
    with cfg_path.open() as fh:
        cfg = yaml.safe_load(fh)
    alpha = float(cfg["condA"]["center_alpha"])
    total = int(cfg["common"]["total_steps"])
    period = int(cfg["condA"]["T_values"][0])
    seeds = [int(s) for s in cfg["common"]["seeds"]]
    R = len(seeds)

    logs = []
    for seed in seeds:
        with np.load(centered_dir / "logs" / f"seed{seed}.npz") as z:
            logs.append(
                dict(
                    step=z["step"].astype(np.int64),
                    mu_norm=z["mu_norm"].astype(np.float64),
                    flip_state=z["flip_state"].astype(np.float64),
                )
            )
    steps = logs[0]["step"]
    if any(not np.array_equal(z["step"], steps) for z in logs[1:]):
        raise AssertionError("probe log step grids differ across seeds")

    gens = make_gens("A", 100, "cpu")
    T = torch.tensor([period] * R, dtype=torch.long)
    env = SCREnv(R, D, F, T, gens["input"], "cpu")
    running_mean = torch.zeros(R, D, dtype=torch.float32)
    mu_all = np.empty((len(steps), R, D), dtype=np.float64)
    flip_all = np.empty((len(steps), R, F), dtype=np.float64)
    ptr = 0
    for t in range(total + 1):
        if ptr < len(steps) and t == int(steps[ptr]):
            raw_mu = torch.cat(
                [env.flip_state, torch.full((R, TOP_DIM), 0.5, dtype=torch.float32)], dim=1
            )
            mu_all[ptr] = _np(raw_mu - running_mean)
            flip_all[ptr] = _np(env.flip_state)
            ptr += 1
        if t == total:
            break
        x_raw = env.step()
        running_mean.mul_(1.0 - alpha).add_(alpha * x_raw)
    if ptr != len(steps):
        raise AssertionError(f"replay captured {ptr}/{len(steps)} probe points")

    logged_mu_norm = np.column_stack([z["mu_norm"] for z in logs])
    replay_mu_norm = np.linalg.norm(mu_all, axis=2)
    log_mu_err = float(np.max(np.abs(replay_mu_norm - logged_mu_norm)))
    log_flip_equal = all(np.array_equal(flip_all[:, r], logs[r]["flip_state"]) for r in range(R))
    ck_rm = centered_ckpt["running_mean"].detach().cpu()
    rm_bit_equal = bool(torch.equal(running_mean, ck_rm))
    rm_max_error = float(torch.max(torch.abs(running_mean - ck_rm)).item())
    if not log_flip_equal or log_mu_err > 2e-6 or not rm_bit_equal:
        raise AssertionError(
            f"replay mismatch: flip={log_flip_equal}, mu_err={log_mu_err}, "
            f"rm_bit_equal={rm_bit_equal}, rm_err={rm_max_error}"
        )

    q_mu = top_projection(mu_all)
    angles = np.degrees(np.arccos(np.sqrt(np.clip(q_mu, 0.0, 1.0))))
    rem = steps % period
    post = (steps >= period + 1) & (rem >= 1) & (rem <= 100)
    pre = (steps >= period) & ((rem >= period - 100) | (rem == 0))
    bulk = ~(post | pre)
    regions = {"post_boundary_1_100": post, "pre_boundary_-100_0": pre, "bulk": bulk, "all": np.ones(len(steps), bool)}

    region_rows = []
    for r, seed in enumerate(seeds):
        for region, mask in regions.items():
            vals_q = q_mu[mask, r]
            vals_a = angles[mask, r]
            region_rows.append(
                dict(
                    seed=seed,
                    region=region,
                    n_points=int(mask.sum()),
                    mu_top_projection_mean=float(np.mean(vals_q)),
                    mu_top_projection_median=float(np.median(vals_q)),
                    angle_to_top_space_mean_deg=float(np.mean(vals_a)),
                    angle_to_top_space_median_deg=float(np.median(vals_a)),
                    angle_to_top_space_q25_deg=float(np.percentile(vals_a, 25)),
                    angle_to_top_space_q75_deg=float(np.percentile(vals_a, 75)),
                )
            )

    offsets = np.full(len(steps), 9999, dtype=np.int64)
    near_pre = rem >= period - 100
    near_post = (rem <= 100) & (steps >= period)
    offsets[near_pre] = rem[near_pre] - period
    offsets[near_post] = rem[near_post]
    offset_rows = []
    for off in range(-100, 101):
        vals_q = q_mu[offsets == off].ravel()
        vals_a = angles[offsets == off].ravel()
        if len(vals_q) == 0:
            continue
        offset_rows.append(
            dict(
                offset=off,
                n_samples=len(vals_q),
                mu_top_projection_median=float(np.median(vals_q)),
                mu_top_projection_q25=float(np.percentile(vals_q, 25)),
                mu_top_projection_q75=float(np.percentile(vals_q, 75)),
                angle_median_deg=float(np.median(vals_a)),
                angle_q25_deg=float(np.percentile(vals_a, 25)),
                angle_q75_deg=float(np.percentile(vals_a, 75)),
            )
        )
    validation = dict(
        n_record_steps=len(steps),
        log_flip_state_exact=log_flip_equal,
        log_mu_norm_max_abs_error=log_mu_err,
        checkpoint_running_mean_bit_exact=rm_bit_equal,
        checkpoint_running_mean_max_abs_error=rm_max_error,
    )
    return pd.DataFrame(region_rows), pd.DataFrame(offset_rows), validation


def _mean(seed_df: pd.DataFrame, arm: str, step: int, col: str) -> float:
    z = seed_df[(seed_df.arm == arm) & (seed_df.step == step)][col].to_numpy(float)
    return float(np.nanmean(z))


def _comparison(comp: pd.DataFrame, name: str) -> pd.Series:
    z = comp[comp.contrast == name]
    if len(z) != 1:
        raise KeyError(name)
    return z.iloc[0]


def write_summary(
    outdir: Path,
    seed_df: pd.DataFrame,
    comp: pd.DataFrame,
    mu_regions: pd.DataFrame,
    validation: dict,
):
    h1a = _comparison(comp, "centered_final_minus_initial")
    h1b = _comparison(comp, "centered_final_minus_std_final")
    h3 = _comparison(comp, "centered_final_rho_beta_minus_cos")
    post = mu_regions[mu_regions.region == "post_boundary_1_100"]
    pre = mu_regions[mu_regions.region == "pre_boundary_-100_0"]

    def ci(row):
        return f"{row.estimate:.4f} [{row.ci_low:.4f}, {row.ci_high:.4f}]"

    c_final = seed_df[(seed_df.arm == "centered") & (seed_df.step == FINAL_STEP)]
    s_final = seed_df[(seed_df.arm == "std") & (seed_df.step == FINAL_STEP)]
    lines = [
        "# centered 共分散主空間仮説 — 探索解析",
        "",
        "> 結果観察後に立てた仮説の post-hoc 解析。確認的検定ではない。",
        "",
        "## 構造と再生サニティ",
        "",
        f"- 周期内 Sigma の固有値: 0（15重）/ 0.25（5重）。最大固有空間のランダム射影期待値は {RANDOM_FLOOR:.2f}。",
        f"- probe 記録点: {validation['n_record_steps']:,}。flip_state 完全一致={validation['log_flip_state_exact']}、mu_norm 最大誤差={validation['log_mu_norm_max_abs_error']:.3g}。",
        f"- 最終 running_mean bit一致={validation['checkpoint_running_mean_bit_exact']}（最大誤差 {validation['checkpoint_running_mean_max_abs_error']:.3g}）。",
        "",
        "## H1: W は最大固有空間へ移ったか",
        "",
        f"- 全unitの平均射影率（seed平均）: 初期 centered {_mean(seed_df, 'centered', 0, 'unit_top_proj_mean_all'):.4f}、最終 centered {_mean(seed_df, 'centered', FINAL_STEP, 'unit_top_proj_mean_all'):.4f}、最終 std {_mean(seed_df, 'std', FINAL_STEP, 'unit_top_proj_mean_all'):.4f}。",
        f"- centered 最終−初期: {ci(h1a)}（paired seed bootstrap 95% CI）。",
        f"- centered 最終−std 最終: {ci(h1b)}。",
        f"- centered 最終の方向正規化 W 主軸 q(e1U): 全unit {_mean(seed_df, 'centered', FINAL_STEP, 'e1U_all_top_proj'):.4f}、alive {_mean(seed_df, 'centered', FINAL_STEP, 'e1U_alive_top_proj'):.4f}。ただし第1軸のエネルギー比は全unit {_mean(seed_df, 'centered', FINAL_STEP, 'e1U_all_top1_frac'):.4f}、alive {_mean(seed_df, 'centered', FINAL_STEP, 'e1U_alive_top1_frac'):.4f}。",
        f"- std 最終の方向正規化 W 主軸 q(e1U): 全unit {_mean(seed_df, 'std', FINAL_STEP, 'e1U_all_top_proj'):.4f}、alive {_mean(seed_df, 'std', FINAL_STEP, 'e1U_alive_top_proj'):.4f}。第1軸のエネルギー比は全unit {_mean(seed_df, 'std', FINAL_STEP, 'e1U_all_top1_frac'):.4f}、alive {_mean(seed_df, 'std', FINAL_STEP, 'e1U_alive_top1_frac'):.4f}。",
        "- 結論: centered 内で最大固有空間への傾きは増えた。一方、個々の W の射影は std の方が大きく、centered の第1軸エネルギーも小さいため、「centered だけが単一の最大固有ベクトルへ強く整列した」という強い形は支持されない。",
        "",
        "## H2: 境界直後の mu は最大固有空間と直交するか",
        "",
        f"- 境界直後 +1..+100 の q(mu): seed別中央値の平均 {post.mu_top_projection_median.mean():.4f}。部分空間への角度: seed別中央値の平均 {post.angle_to_top_space_median_deg.mean():.2f}°。",
        f"- 境界前 -100..0 の q(mu): seed別中央値の平均 {pre.mu_top_projection_median.mean():.4f}。角度: {pre.angle_to_top_space_median_deg.mean():.2f}°。",
        "- 角度90°が直交、0°が最大固有空間内を表す。",
        "- 結論: 境界で flip bit の残差が立つと mu は最大固有空間にほぼ直交し、その後EMAとともに戻る、という幾何は強く支持される。",
        "",
        "## H3: cos(u,mu) と規格化マージン beta",
        "",
        f"- centered 最終 Spearman rho(p, cos): seed平均 {c_final.rho_p_cos.mean():.4f}。rho(p, beta): {c_final.rho_p_beta.mean():.4f}。",
        f"- rho_beta−rho_cos: {ci(h3)}。",
        f"- std 最終 rho(p, cos): seed平均 {s_final.rho_p_cos.mean():.4f}。rho(p, beta): {s_final.rho_p_beta.mean():.4f}。",
        "- 結論: centered の最終 checkpoint では、ゲート率の順位は cos(u,mu) ではなく bias と共分散スケールを含む beta がほぼ完全に説明する。",
        "",
        "## 解釈上の制限",
        "",
        "- W の全方向は step 0 と 1M にしか保存されていない。したがって、H2 は各時点の mu と既知の最大固有空間の角度を厳密に測るが、同時刻の W 主軸との角度ではない。",
        "- beta は5ビット離散入力の完全な十分統計ではない。順位相関の改善は、bias と分散を無視できないことの診断である。",
        "- 対象は condA・w100・T=10,000・center_alpha=0.01・seed 10本に限定される。",
    ]
    (outdir / "summary.md").write_text("\n".join(lines) + "\n")


def make_figure(outdir: Path, seed_df: pd.DataFrame, mu_offset: pd.DataFrame):
    fig, ax = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)

    # A: individual W projection
    a = ax[0, 0]
    groups = [("std", 0), ("std", FINAL_STEP), ("centered", 0), ("centered", FINAL_STEP)]
    labels = ["std\n0", "std\n1M", "centered\n0", "centered\n1M"]
    rng = np.random.default_rng(7)
    for x, (arm, step) in enumerate(groups):
        vals = seed_df[(seed_df.arm == arm) & (seed_df.step == step)]["unit_top_proj_mean_all"].to_numpy()
        a.scatter(x + rng.uniform(-0.06, 0.06, len(vals)), vals, s=25, alpha=0.8)
        a.plot([x - 0.18, x + 0.18], [np.mean(vals)] * 2, color="black", lw=2)
    a.axhline(RANDOM_FLOOR, color="0.45", ls="--", label="random expectation 5/20")
    a.set_xticks(range(4), labels)
    a.set_ylabel(r"mean $||P_{max}w_i||^2/||w_i||^2$")
    a.set_title("A. Individual weight projection")
    a.legend(frameon=False, fontsize=9)

    # B: population direction axis
    a = ax[0, 1]
    final = seed_df[seed_df.step == FINAL_STEP]
    xlabels = ["std all", "std alive", "cent all", "cent alive"]
    selectors = [
        ("std", "e1U_all_top_proj"),
        ("std", "e1U_alive_top_proj"),
        ("centered", "e1U_all_top_proj"),
        ("centered", "e1U_alive_top_proj"),
    ]
    for x, (arm, col) in enumerate(selectors):
        vals = final[final.arm == arm][col].dropna().to_numpy()
        a.scatter(x + rng.uniform(-0.06, 0.06, len(vals)), vals, s=25, alpha=0.8)
        a.plot([x - 0.18, x + 0.18], [np.mean(vals)] * 2, color="black", lw=2)
    a.axhline(RANDOM_FLOOR, color="0.45", ls="--")
    a.set_xticks(range(4), xlabels, rotation=15)
    a.set_ylim(-0.03, 1.03)
    a.set_ylabel(r"$||P_{max}e_1^U||^2$")
    a.set_title("B. Population direction axis at 1M")

    # C: residual mean geometry around a boundary
    a = ax[1, 0]
    x = mu_offset.offset.to_numpy()
    y = mu_offset.angle_median_deg.to_numpy()
    a.fill_between(x, mu_offset.angle_q25_deg, mu_offset.angle_q75_deg, alpha=0.25)
    a.plot(x, y, lw=2)
    a.axvline(0, color="0.3", ls="--")
    a.set_xlabel("probe offset from task boundary")
    a.set_ylabel("angle(mu, top eigenspace) [deg]")
    a.set_ylim(0, 92)
    a.set_title("C. Residual mean after centering")

    # D: gate coordinate rank correlation
    a = ax[1, 1]
    for x0, arm in enumerate(("std", "centered")):
        z = final[final.arm == arm].sort_values("seed")
        for _, row in z.iterrows():
            a.plot([x0 * 3, x0 * 3 + 1], [row.rho_p_cos, row.rho_p_beta], color="0.75", lw=0.8)
        a.scatter(np.full(len(z), x0 * 3), z.rho_p_cos, s=25, label=f"{arm}: cos")
        a.scatter(np.full(len(z), x0 * 3 + 1), z.rho_p_beta, s=25, label=f"{arm}: beta")
    a.set_xticks([0, 1, 3, 4], ["std cos", "std beta", "cent cos", "cent beta"], rotation=15)
    a.set_ylabel("Spearman rho with exact gate rate")
    a.set_ylim(-1.03, 1.03)
    a.axhline(0, color="0.6", lw=0.8)
    a.set_title("D. Gate coordinate at 1M")

    fig.savefig(outdir / "fig_covariance_axis.png", dpi=180)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--std-dir", type=Path, default=STD_DIR)
    ap.add_argument("--centered-dir", type=Path, default=CENTERED_DIR)
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--bootstrap-B", type=int, default=BOOTSTRAP_B)
    ap.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    all_units: list[dict] = []
    all_seeds: list[dict] = []
    checkpoints = {}
    for arm, result_dir, centered in (
        ("std", args.std_dir, False),
        ("centered", args.centered_dir, True),
    ):
        for step in (0, FINAL_STEP):
            units, seeds, ck = load_checkpoint_metrics(arm, result_dir, centered, step)
            all_units.extend(units)
            all_seeds.extend(seeds)
            checkpoints[(arm, step)] = ck

    # Paired design sanity at initialization and environment trajectory at final.
    for key in ("W", "b", "v", "c"):
        if not torch.equal(checkpoints[("std", 0)]["net"][key], checkpoints[("centered", 0)]["net"][key]):
            raise AssertionError(f"initial {key} differs between arms")
    if not torch.equal(
        checkpoints[("std", FINAL_STEP)]["env"]["flip_state"],
        checkpoints[("centered", FINAL_STEP)]["env"]["flip_state"],
    ):
        raise AssertionError("final flip_state differs between paired arms")

    unit_df = pd.DataFrame(all_units)
    seed_df = pd.DataFrame(all_seeds)
    comparisons = make_comparisons(seed_df, args.bootstrap_B, args.bootstrap_seed)
    mu_regions, mu_offset, validation = replay_centered_mu(
        args.centered_dir,
        checkpoints[("centered", FINAL_STEP)],
        ROOT / "configs" / "ratchet_centered_0822.yaml",
    )

    unit_df.to_csv(args.outdir / "checkpoint_unit_metrics.csv", index=False)
    seed_df.to_csv(args.outdir / "checkpoint_seed_metrics.csv", index=False)
    comparisons.to_csv(args.outdir / "comparisons.csv", index=False)
    mu_regions.to_csv(args.outdir / "mu_geometry_seed_summary.csv", index=False)
    mu_offset.to_csv(args.outdir / "mu_geometry_by_offset.csv", index=False)
    write_summary(args.outdir, seed_df, comparisons, mu_regions, validation)
    make_figure(args.outdir, seed_df, mu_offset)
    print((args.outdir / "summary.md").read_text())
    print(f"wrote {args.outdir}")


if __name__ == "__main__":
    main()
