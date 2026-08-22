"""q3_gate_curve_ci: centered 消灯点の事前登録集計 [spec_ratchet_centered_0822 §5–§6]。

実行順序:

  1. 本ファイルを commit する（集計コードの凍結）
  2. OMP_NUM_THREADS=1 .venv/bin/python analysis/q3_gate_curve_ci.py

入力は commit 済み std ログ ``results/ratchet_log_0819/logs/seed*.npz`` と、
事前登録後に取得した centered ログ
``results/ratchet_centered_0822/logs/seed*.npz``。再学習は行わない。

主推定量は、cos 幅 0.05 の各ビンで p_hat のプール中央値を取り、低 cos 側から
中央値ゼロが連続する領域の最大上端を消灯点 theta_med とするもの。p_hat は
32 パターンの厳密値 k/32 なので、seed bootstrap は各 seed・ビンの 33 カテゴリ
度数を再集計して中央値を厳密に復元する。2,000 万サンプルを bootstrap ごとに
展開しない。

出力（centered results 内）:
  verdict.csv / gate_curve.csv / theta_estimates.csv / per_seed_metrics.csv /
  analysis_meta.json / summary.md / figures/fig_q3_gate_curves.png /
  figures/fig_q3_mu_norm_boundary.png
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import ROOT  # noqa: E402


STD_DIR = Path(ROOT) / "results" / "ratchet_log_0819"
CENTERED_DIR = Path(ROOT) / "results" / "ratchet_centered_0822"
SPEC = "specs/spec_ratchet_centered_0822.md"

COS_LO, COS_HI, BIN_W = -0.60, 0.60, 0.05
BIN_EDGES = np.linspace(COS_LO, COS_HI, 25, dtype=np.float64)
BIN_UPPER = BIN_EDGES[1:]
N_BIN = 24
N_P = 33                         # p_hat = k/32, k=0..32
P_VALUES = np.arange(N_P, dtype=np.float64) / 32.0
MIN_BIN_N = 1000
BOOT_B = 10_000
BOOT_SEED = 20260822
HALF_STEP = 500_000
SCOPES = ["all", "w_q1", "w_q2", "w_q3", "w_q4"]
HALVES = ["t_lt_500k", "t_ge_500k"]

plt.rcParams["font.family"] = ["Noto Sans CJK JP", "Noto Sans CJK TC",
                               "Noto Sans CJK KR", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def git_hash(paths: list[str] | None = None) -> str:
    cmd = ["git", "log", "-1", "--format=%h"]
    if paths:
        cmd += ["--", *paths]
    try:
        return subprocess.check_output(cmd, cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def seed_paths(resdir: Path) -> list[Path]:
    paths = list((resdir / "logs").glob("seed*.npz"))
    paths.sort(key=lambda p: int(p.stem.removeprefix("seed")))
    if len(paths) != 10:
        raise SystemExit(f"{resdir}: seed log は10本必要、実際は {len(paths)} 本")
    return paths


def check_source_run(resdir: Path, expected_spec: str | None = None) -> dict:
    """本走の構造サニティを再確認し、FAIL 成果物の集計を止める。"""
    meta_path = resdir / "meta.json"
    if not meta_path.exists():
        raise SystemExit(f"{resdir}: meta.json がない")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    sanity = meta.get("sanity", {})
    for key, flag in (("S2", "s2_pass"), ("S3", "s3_pass"), ("S4", "s4_pass")):
        if key not in sanity or sanity[key].get(flag) is not True:
            raise SystemExit(f"{resdir}: {key} が PASS でない")
    if meta.get("n_record_steps") != 20_901 or meta.get("n_realized_flips") != 99:
        raise SystemExit(f"{resdir}: record/flip 数が仕様外")
    if expected_spec is not None and meta.get("spec") != expected_spec:
        raise SystemExit(f"{resdir}: spec 参照が不一致: {meta.get('spec')}")
    return meta


def hist_median(hist: np.ndarray) -> np.ndarray:
    """最終軸が p=k/32 の度数である配列の通常の中央値（偶数 n は中央2値の平均）。"""
    hist = np.asarray(hist)
    n = hist.sum(axis=-1)
    cum = np.cumsum(hist, axis=-1)
    lo_rank = (n - 1) // 2 + 1
    hi_rank = n // 2 + 1
    lo_idx = np.argmax(cum >= lo_rank[..., None], axis=-1)
    hi_idx = np.argmax(cum >= hi_rank[..., None], axis=-1)
    out = (P_VALUES[lo_idx] + P_VALUES[hi_idx]) / 2.0
    return np.where(n > 0, out, np.nan)


def theta_one(med: np.ndarray, valid: np.ndarray) -> float:
    """低 cos 側から中央値ゼロが連続する領域の最大ビン上端 [spec §5.2]。"""
    idx = np.flatnonzero(valid)
    if idx.size == 0 or not np.isfinite(med[idx[0]]) or med[idx[0]] != 0:
        return np.nan
    theta = np.nan
    for j in idx:
        if not np.isfinite(med[j]) or med[j] != 0:
            break
        theta = float(BIN_UPPER[j])
    return theta


def theta_all_one(hist: np.ndarray, valid: np.ndarray) -> float:
    """低 cos 側から全サンプル p_hat=0 が連続する厳格版消灯点。"""
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        return np.nan
    nonzero = hist[:, 1:].sum(axis=1)
    if nonzero[idx[0]] != 0:
        return np.nan
    theta = np.nan
    for j in idx:
        if nonzero[j] != 0:
            break
        theta = float(BIN_UPPER[j])
    return theta


def theta_many(med: np.ndarray, hist: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """bootstrap 行ごとに有効ビン n>=1000 を再判定し theta_med/all を返す。"""
    n = hist.sum(axis=-1)
    out_med = np.full(len(med), np.nan)
    out_all = np.full(len(med), np.nan)
    for i in range(len(med)):
        valid = n[i] >= MIN_BIN_N
        out_med[i] = theta_one(med[i], valid)
        out_all[i] = theta_all_one(hist[i], valid)
    return out_med, out_all


def count_hist(bin_idx: np.ndarray, p_idx: np.ndarray, mask: np.ndarray) -> np.ndarray:
    code = bin_idx[mask].astype(np.int64) * N_P + p_idx[mask].astype(np.int64)
    return np.bincount(code, minlength=N_BIN * N_P).reshape(N_BIN, N_P)


def summarize_arm(label: str, resdir: Path) -> dict:
    """arm を2パスで度数化する。結合集計値（theta）はここでは計算しない。"""
    paths = seed_paths(resdir)

    # arm 内の pooled ||w|| 四分位（84 MB 程度）。境界を得たら直ちに解放する。
    w_chunks = []
    for path in paths:
        with np.load(path) as z:
            w_chunks.append(np.asarray(z["w_norm"], dtype=np.float32).reshape(-1))
    pooled_w = np.concatenate(w_chunks)
    qbounds = np.quantile(pooled_w, [0.25, 0.50, 0.75])
    w_min, w_max = float(pooled_w.min()), float(pooled_w.max())
    del pooled_w, w_chunks

    counts = np.zeros((10, len(SCOPES), N_BIN, N_P), dtype=np.int64)
    outside = np.zeros((10, len(SCOPES)), dtype=np.int64)
    half_counts = np.zeros((10, 2, N_BIN, N_P), dtype=np.int64)
    mu_chunks: list[np.ndarray] = []
    boundary_mu: list[list[np.ndarray]] = [[] for _ in range(101)]
    per_seed = []
    seeds = []
    common_step = None

    for si, path in enumerate(paths):
        with np.load(path) as z:
            step = np.asarray(z["step"], dtype=np.int64)
            p = np.asarray(z["p_hat"], dtype=np.float32)
            cos = np.asarray(z["cos_u_mu"], dtype=np.float32)
            w = np.asarray(z["w_norm"], dtype=np.float32)
            mu = np.asarray(z["mu_norm"], dtype=np.float32)
            fs = np.asarray(z["flip_state"], dtype=np.float32)
            seed = int(z["seed"])
            run_id = str(z["run_id"])

        if p.shape != (20_901, 100) or cos.shape != p.shape or w.shape != p.shape:
            raise SystemExit(f"{path}: unit 配列 shape 不正 {p.shape}/{cos.shape}/{w.shape}")
        if common_step is None:
            common_step = step
        elif not np.array_equal(common_step, step):
            raise SystemExit(f"{path}: step grid が seed 間で不一致")
        if not np.isfinite(p).all() or not np.isfinite(cos).all() or not np.isfinite(w).all():
            raise SystemExit(f"{path}: NaN/Inf")

        p_scaled = p * 32.0
        if not np.allclose(p_scaled, np.rint(p_scaled), atol=1e-6):
            raise SystemExit(f"{path}: p_hat が k/32 でない")
        p_idx = np.rint(p_scaled).astype(np.int8)
        in_range = (cos >= COS_LO) & (cos < COS_HI)
        bin_idx = np.floor((cos - COS_LO) / BIN_W).astype(np.int16)
        layer = np.digitize(w, qbounds, right=True).astype(np.int8)  # 0..3

        flat_in = in_range.reshape(-1)
        flat_bin = bin_idx.reshape(-1)
        flat_p = p_idx.reshape(-1)
        counts[si, 0] = count_hist(flat_bin, flat_p, flat_in)
        outside[si, 0] = int((~flat_in).sum())
        for qi in range(4):
            qmask = (layer == qi).reshape(-1)
            counts[si, qi + 1] = count_hist(flat_bin, flat_p, flat_in & qmask)
            outside[si, qi + 1] = int((~flat_in & qmask).sum())

        for hi, row_sel in enumerate((step < HALF_STEP, step >= HALF_STEP)):
            hm = in_range[row_sel].reshape(-1)
            half_counts[si, hi] = count_hist(bin_idx[row_sel].reshape(-1),
                                              p_idx[row_sel].reshape(-1), hm)

        mu_chunks.append(mu.astype(np.float64, copy=False))
        changed = (np.abs(np.diff(fs, axis=0)) > 0).any(axis=1)
        left_steps = step[:-1][changed]
        if left_steps.size != 99:
            raise SystemExit(f"{path}: realized flip {left_steps.size} != 99")
        for off in range(101):
            target = left_steps + off
            ix = np.searchsorted(step, target)
            if not np.array_equal(step[ix], target):
                raise SystemExit(f"{path}: boundary offset +{off} が grid にない")
            boundary_mu[off].append(mu[ix].astype(np.float64, copy=False))

        dead_frac = float((p[-1] < 0.05).mean())
        per_seed.append(dict(arm=label, seed=seed, run_id=run_id,
                             final_dead_frac=dead_frac,
                             mu_norm_median=float(np.median(mu)),
                             mu_norm_q1=float(np.quantile(mu, 0.25)),
                             mu_norm_q3=float(np.quantile(mu, 0.75))))
        seeds.append(seed)

    if seeds != list(range(10)):
        raise SystemExit(f"{label}: seeds が 0..9 でない: {seeds}")

    mu_all = np.concatenate(mu_chunks)
    post = np.array([np.concatenate(v) for v in boundary_mu], dtype=np.float64)
    return dict(label=label, resdir=str(resdir), seeds=seeds, step=common_step,
                qbounds=qbounds, w_min=w_min, w_max=w_max,
                counts=counts, outside=outside, half_counts=half_counts,
                mu_all=mu_all, boundary_mu=post, per_seed=per_seed)


def bootstrap_arm(arm: dict, weights: np.ndarray) -> dict:
    B = weights.shape[0]
    curve = np.full((B, len(SCOPES), N_BIN), np.nan)
    theta_med = np.full((B, len(SCOPES)), np.nan)
    theta_all = np.full((B, len(SCOPES)), np.nan)

    for si in range(len(SCOPES)):
        flat = arm["counts"][:, si].reshape(10, -1)
        hist = (weights @ flat).reshape(B, N_BIN, N_P)
        med = hist_median(hist)
        tm, ta = theta_many(med, hist)
        curve[:, si] = med
        theta_med[:, si] = tm
        theta_all[:, si] = ta
        del hist

    half_theta = np.full((B, 2), np.nan)
    for hi in range(2):
        flat = arm["half_counts"][:, hi].reshape(10, -1)
        hist = (weights @ flat).reshape(B, N_BIN, N_P)
        med = hist_median(hist)
        half_theta[:, hi], _ = theta_many(med, hist)
        del hist
    return dict(curve=curve, theta_med=theta_med,
                theta_all=theta_all, half_theta=half_theta)


def finite_ci(v: np.ndarray) -> tuple[float, float, int]:
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan, np.nan, 0
    q = np.quantile(v, [0.025, 0.975])
    return float(q[0]), float(q[1]), int(v.size)


def point_estimates(arm: dict) -> dict:
    pooled = arm["counts"].sum(axis=0)
    n = pooled.sum(axis=-1)
    med = hist_median(pooled)
    valid = n >= MIN_BIN_N
    theta_med, theta_all, theta_strat, nonzero_below = [], [], [], []

    seed_med = hist_median(arm["counts"])     # [seed, scope, bin]
    strat_med = np.nanmedian(seed_med, axis=0)
    for si in range(len(SCOPES)):
        tm = theta_one(med[si], valid[si])
        ta = theta_all_one(pooled[si], valid[si])
        ts = theta_one(strat_med[si], valid[si])
        theta_med.append(tm)
        theta_all.append(ta)
        theta_strat.append(ts)
        if np.isfinite(tm):
            below = BIN_UPPER <= tm + 1e-12
            nonzero_below.append(int(pooled[si, below, 1:].sum()))
        else:
            nonzero_below.append(0)

    half_pooled = arm["half_counts"].sum(axis=0)
    half_med = hist_median(half_pooled)
    half_theta = [theta_one(half_med[i], half_pooled[i].sum(axis=-1) >= MIN_BIN_N)
                  for i in range(2)]
    return dict(pooled=pooled, n=n, med=med, valid=valid,
                theta_med=np.asarray(theta_med), theta_all=np.asarray(theta_all),
                theta_strat=np.asarray(theta_strat),
                nonzero_below=np.asarray(nonzero_below),
                half_theta=np.asarray(half_theta))


def build_tables(arms: list[dict], points: dict, boots: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    curve_rows, theta_rows = [], []
    for arm in arms:
        label = arm["label"]
        pt, bt = points[label], boots[label]
        for si, scope in enumerate(SCOPES):
            for bi in range(N_BIN):
                lo, hi, nf = finite_ci(bt["curve"][:, si, bi])
                curve_rows.append(dict(
                    arm=label, scope=scope, bin_index=bi,
                    cos_lo=BIN_EDGES[bi], cos_hi=BIN_EDGES[bi + 1],
                    n=int(pt["n"][si, bi]), valid=bool(pt["valid"][si, bi]),
                    p_median=float(pt["med"][si, bi]) if pt["n"][si, bi] else np.nan,
                    ci_lo=lo, ci_hi=hi, bootstrap_finite=nf,
                    outside_scope_n=int(arm["outside"][:, si].sum())))
            for kind, vals, reps in (
                ("theta_med", pt["theta_med"], bt["theta_med"]),
                ("theta_all", pt["theta_all"], bt["theta_all"]),
            ):
                lo, hi, nf = finite_ci(reps[:, si])
                theta_rows.append(dict(
                    arm=label, scope=scope, estimate=kind,
                    point=float(vals[si]), ci_lo=lo, ci_hi=hi,
                    bootstrap_finite=nf,
                    theta_med_strat=(float(pt["theta_strat"][si])
                                     if kind == "theta_med" else np.nan),
                    nonzero_samples_below_theta_med=(int(pt["nonzero_below"][si])
                                                     if kind == "theta_med" else np.nan)))
        for hi_idx, half in enumerate(HALVES):
            lo, hi, nf = finite_ci(bt["half_theta"][:, hi_idx])
            theta_rows.append(dict(
                arm=label, scope=half, estimate="theta_med_time_half",
                point=float(pt["half_theta"][hi_idx]), ci_lo=lo, ci_hi=hi,
                bootstrap_finite=nf, theta_med_strat=np.nan,
                nonzero_samples_below_theta_med=np.nan))
    return pd.DataFrame(curve_rows), pd.DataFrame(theta_rows)


def verdict_tables(arms: list[dict], points: dict, boots: dict,
                   theta_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    std, ctr = points["std"], points["centered"]
    bstd, bctr = boots["std"], boots["centered"]
    delta_rows = []
    for si, scope in enumerate(SCOPES):
        point = float(ctr["theta_med"][si] - std["theta_med"][si])
        reps = bctr["theta_med"][:, si] - bstd["theta_med"][:, si]
        lo, hi, nf = finite_ci(reps)
        delta_rows.append(dict(scope=scope, delta_theta_med=point,
                               ci_lo=lo, ci_hi=hi, bootstrap_finite=nf))
    delta_df = pd.DataFrame(delta_rows)

    ctr_all = ctr["theta_med"][0]
    n_neg_valid = int((ctr["valid"][0] & (BIN_UPPER <= 0)).sum())
    c1 = bool(np.isfinite(ctr_all) and n_neg_valid >= 4)
    d0 = delta_df.iloc[0]
    if not c1 or not np.isfinite(d0.ci_lo) or not np.isfinite(d0.ci_hi):
        c2 = "不可比（C2を実施しない）"
    elif d0.ci_hi < 0 or d0.ci_lo > 0:
        c2 = "追随"
    elif d0.ci_lo >= -0.05 - 1e-12 and d0.ci_hi <= 0.05 + 1e-12 \
            and d0.ci_lo <= 0 <= d0.ci_hi:
        c2 = "固有"
    else:
        c2 = "保留"

    if c1 and np.isfinite(d0.delta_theta_med):
        direction = int(np.sign(d0.delta_theta_med))
        layer_signs = np.sign(delta_df.iloc[1:].delta_theta_med.to_numpy()).astype(int)
        agree = int((layer_signs == direction).sum())
        c3_result = f"{agree}/4層が全体方向と一致"
    else:
        agree = 0
        c3_result = "不可比のため未実施"

    verdict = pd.DataFrame([
        dict(id="C1", question="centered曲線はstdと可比か",
             result="可比" if c1 else "不可比",
             theta_centered=ctr_all, theta_std=std["theta_med"][0],
             delta_theta=d0.delta_theta_med, ci_lo=d0.ci_lo, ci_hi=d0.ci_hi,
             detail=f"centered theta_med finite={np.isfinite(ctr_all)}, "
                    f"cos<0有効ビン={n_neg_valid}"),
        dict(id="C2", question="消灯点は動いたか（主判定）", result=c2,
             theta_centered=ctr_all, theta_std=std["theta_med"][0],
             delta_theta=d0.delta_theta_med, ci_lo=d0.ci_lo, ci_hi=d0.ci_hi,
             detail="paired seed bootstrap 95%CI; 分解能0.05"),
        dict(id="C3", question="四分位層別で一貫するか", result=c3_result,
             theta_centered=np.nan, theta_std=np.nan, delta_theta=np.nan,
             ci_lo=np.nan, ci_hi=np.nan,
             detail="報告のみ。C2を覆さない"),
    ])
    return verdict, delta_df


def make_figures(outdir: Path, arms: list[dict], curve_df: pd.DataFrame) -> None:
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    colors = {"std": "tab:blue", "centered": "tab:orange"}

    fig, axes = plt.subplots(1, 5, figsize=(17.5, 3.8), sharex=True, sharey=True)
    x = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
    for ax, scope in zip(axes, SCOPES):
        for arm in ("std", "centered"):
            d = curve_df[(curve_df.arm == arm) & (curve_df.scope == scope)].sort_values("bin_index")
            good = d.valid.to_numpy(dtype=bool)
            y = d.p_median.to_numpy(dtype=float)
            lo = d.ci_lo.to_numpy(dtype=float)
            hi = d.ci_hi.to_numpy(dtype=float)
            ax.plot(x[good], y[good], marker="o", ms=3, lw=1.2,
                    color=colors[arm], label=arm)
            ax.fill_between(x[good], lo[good], hi[good], color=colors[arm], alpha=0.16)
        ax.axhline(0, color="black", lw=0.6)
        ax.axvline(0, color="gray", lw=0.6, ls="--")
        ax.set_title(scope)
        ax.grid(alpha=0.25)
        ax.set_xlabel("cos(u, mu)")
    axes[0].set_ylabel("bin median p_hat")
    axes[0].legend(fontsize=8)
    fig.suptitle("Q3 gate curves: std vs centered (seed-bundle bootstrap 95% CI)")
    fig.tight_layout()
    fig.savefig(figdir / "fig_q3_gate_curves.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))
    for arm in arms:
        post = arm["boundary_mu"]
        med = np.median(post, axis=1)
        q1 = np.quantile(post, 0.25, axis=1)
        q3 = np.quantile(post, 0.75, axis=1)
        off = np.arange(101)
        col = colors[arm["label"]]
        axes[0].plot(off, med, color=col, label=arm["label"])
        axes[0].fill_between(off, q1, q3, color=col, alpha=0.16)
        axes[1].hist(arm["mu_all"], bins=60, histtype="step", density=True,
                     color=col, label=arm["label"])
    axes[0].axvline(1, color="gray", ls="--", lw=0.8,
                    label="new flip visible (+1)")
    axes[0].set(xlabel="offset from realized boundary", ylabel="mu_norm",
                title="Boundary [0,+100]: median and IQR")
    axes[1].set(xlabel="mu_norm", ylabel="density", title="All recorded points")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figdir / "fig_q3_mu_norm_boundary.png", dpi=150)
    plt.close(fig)


def fnum(v: float, digits: int = 4) -> str:
    return "NA" if not np.isfinite(v) else f"{v:.{digits}f}"


def md_table(df: pd.DataFrame) -> str:
    """tabulate 追加依存なしの小さな Markdown table writer。"""
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                cells.append("NA" if not np.isfinite(v) else f"{float(v):.6g}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_summary(outdir: Path, arms: list[dict], points: dict,
                  theta_df: pd.DataFrame, delta_df: pd.DataFrame,
                  verdict: pd.DataFrame) -> None:
    lines = [
        "# ratchet_centered_0822: centered消灯点の判定", "",
        f"仕様: `{SPEC}`。事前登録コミット: `{git_hash([SPEC, 'configs/ratchet_centered_0822.yaml', 'src/ratchet_log.py'])}`。"
        f"集計コードコミット: `{git_hash(['analysis/q3_gate_curve_ci.py'])}`。", "",
        "## 0. 一行", "",
        f"C1 **{verdict.iloc[0].result}** / C2 **{verdict.iloc[1].result}** / "
        f"C3 **{verdict.iloc[2].result}**。", "",
        "## 1. 主判定", "",
        "| arm | theta_med | 95% CI | theta_all | theta_med_strat |", "|---|---:|---:|---:|---:|",
    ]
    for arm in ("std", "centered"):
        tm = theta_df[(theta_df.arm == arm) & (theta_df.scope == "all") &
                      (theta_df.estimate == "theta_med")].iloc[0]
        ta = theta_df[(theta_df.arm == arm) & (theta_df.scope == "all") &
                      (theta_df.estimate == "theta_all")].iloc[0]
        lines.append(f"| {arm} | {fnum(tm.point, 2)} | [{fnum(tm.ci_lo, 2)}, {fnum(tm.ci_hi, 2)}] | "
                     f"{fnum(ta.point, 2)} | {fnum(tm.theta_med_strat, 2)} |")
    d = delta_df.iloc[0]
    lines += ["", f"paired Δtheta = centered − std = **{fnum(d.delta_theta_med, 2)}** "
              f"[95% CI {fnum(d.ci_lo, 2)}, {fnum(d.ci_hi, 2)}]。", "",
              md_table(verdict), "", "## 2. ||w|| 四分位", "",
              "四分位境界はarm内のpooled ||w||から別々に計算した（相対層）。", "",
              "| arm | min | Q1 | Q2 | Q3 | max |", "|---|---:|---:|---:|---:|---:|"]
    for arm in arms:
        q = arm["qbounds"]
        lines.append(f"| {arm['label']} | {arm['w_min']:.4f} | {q[0]:.4f} | {q[1]:.4f} | "
                     f"{q[2]:.4f} | {arm['w_max']:.4f} |")
    lines += ["", md_table(delta_df), "", "## 3. 時間半割", "",
              md_table(theta_df[theta_df.estimate == "theta_med_time_half"]), "",
              "## 4. E1（報告のみ）", ""]
    for arm in arms:
        mu = arm["mu_all"]
        post = arm["boundary_mu"]
        df = pd.DataFrame(arm["per_seed"])
        lines += [
            f"- **{arm['label']} mu_norm**: median {np.median(mu):.6f}, "
            f"IQR [{np.quantile(mu, .25):.6f}, {np.quantile(mu, .75):.6f}]。"
            f"境界offset 0 / +1 / +100 の中央値は "
            f"{np.median(post[0]):.6f} / {np.median(post[1]):.6f} / {np.median(post[100]):.6f}。",
            f"- **{arm['label']} final dead_frac**: pooled "
            f"{df.final_dead_frac.mean():.4f}、seed中央値 {df.final_dead_frac.median():.4f}、"
            f"seed IQR [{df.final_dead_frac.quantile(.25):.4f}, {df.final_dead_frac.quantile(.75):.4f}]。",
        ]
    lines += ["", "境界offset 0はprobe順序上flip前、変更後のflip_stateが最初に見えるのは+1。"
              "dead_fracは判定に使用していない。", "",
              "## 5. 集計規約とスコープ", "",
              f"- cosビンは [{COS_LO:.2f}, {COS_HI:.2f})、幅 {BIN_W:.2f}。"
              f"有効ビンはpooled n >= {MIN_BIN_N}。範囲外件数は `gate_curve.csv` に記録。",
              f"- bootstrapはseed束ね B={BOOT_B:,}、`np.random.default_rng({BOOT_SEED})`。"
              "同じseed復元抽出を両armへ適用したpaired比較。",
              "- 主判定はtheta_med全体のみ。theta_all、四分位、時間半割、"
              "theta_med_strat、mu_norm、dead_fracは副次または報告のみ。",
              "- スコープはcondA・w100・T=1e4・batch=1・center_alpha=0.01。"
              "condBおよびalpha依存性へ外挿しない。", "",
              "## 6. 出力", "",
              "- `verdict.csv`: C1–C3", "- `gate_curve.csv`: 全arm・全層・全ビンの曲線とCI",
              "- `theta_estimates.csv`: theta_med/all・時間半割",
              "- `per_seed_metrics.csv`: final dead_frac / mu_norm",
              "- `analysis_meta.json`: provenance・bootstrap設定",
              "- `figures/fig_q3_gate_curves.png`, `fig_q3_mu_norm_boundary.png`", ""]
    (outdir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def self_test() -> None:
    h = np.zeros((2, N_P), dtype=np.int64)
    h[0, 0], h[0, 2] = 1, 1
    h[1, 3] = 3
    m = hist_median(h)
    assert np.allclose(m, [1 / 32, 3 / 32])
    med = np.full(N_BIN, 0.25)
    med[:9] = 0
    valid = np.ones(N_BIN, dtype=bool)
    assert np.isclose(theta_one(med, valid), BIN_UPPER[8])
    hist = np.zeros((N_BIN, N_P), dtype=np.int64)
    hist[:5, 0] = 1000
    hist[5:, 1] = 1000
    assert np.isclose(theta_all_one(hist, valid), BIN_UPPER[4])
    draws = np.array([[0, 0, 1], [2, 1, 2]])
    weights = np.zeros((2, 3), dtype=np.int16)
    np.add.at(weights, (np.arange(2)[:, None], draws), 1)
    assert np.array_equal(weights, [[2, 1, 0], [0, 1, 2]])
    print("q3_gate_curve_ci synthetic self-test: PASS")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--std", type=Path, default=STD_DIR)
    ap.add_argument("--centered", type=Path, default=CENTERED_DIR)
    ap.add_argument("--outdir", type=Path, default=CENTERED_DIR)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return

    t0 = time.time()
    check_source_run(args.std, "specs/spec_ratchet_log_0819.md")
    check_source_run(args.centered, SPEC)
    print("loading and histogramming std ...", flush=True)
    std = summarize_arm("std", args.std)
    print("loading and histogramming centered ...", flush=True)
    centered = summarize_arm("centered", args.centered)
    arms = [std, centered]
    if std["seeds"] != centered["seeds"]:
        raise SystemExit("paired bootstrap不能: seed対応が一致しない")

    rng = np.random.default_rng(BOOT_SEED)
    draws = rng.integers(0, 10, size=(BOOT_B, 10))
    weights = np.zeros((BOOT_B, 10), dtype=np.int16)
    np.add.at(weights, (np.arange(BOOT_B)[:, None], draws), 1)

    points = {a["label"]: point_estimates(a) for a in arms}
    boots = {}
    for arm in arms:
        print(f"bootstrap {arm['label']} ...", flush=True)
        boots[arm["label"]] = bootstrap_arm(arm, weights)

    curve_df, theta_df = build_tables(arms, points, boots)
    verdict, delta_df = verdict_tables(arms, points, boots, theta_df)

    args.outdir.mkdir(parents=True, exist_ok=True)
    curve_df.to_csv(args.outdir / "gate_curve.csv", index=False)
    theta_df.to_csv(args.outdir / "theta_estimates.csv", index=False)
    pd.DataFrame(std["per_seed"] + centered["per_seed"]).to_csv(
        args.outdir / "per_seed_metrics.csv", index=False)
    verdict.to_csv(args.outdir / "verdict.csv", index=False)
    delta_df.to_csv(args.outdir / "delta_theta.csv", index=False)
    make_figures(args.outdir, arms, curve_df)
    write_summary(args.outdir, arms, points, theta_df, delta_df, verdict)

    meta = dict(
        date=time.strftime("%Y-%m-%d %H:%M:%S"), elapsed_sec=round(time.time() - t0, 1),
        spec=SPEC, analysis_git_hash=git_hash(["analysis/q3_gate_curve_ci.py"]),
        prereg_git_hash=git_hash([SPEC, "configs/ratchet_centered_0822.yaml",
                                  "src/ratchet_log.py"]),
        std_source=str(args.std), centered_source=str(args.centered),
        seeds=std["seeds"], n_record_steps=len(std["step"]),
        cos_range=[COS_LO, COS_HI], bin_width=BIN_W, min_bin_n=MIN_BIN_N,
        bootstrap_B=BOOT_B, bootstrap_seed=BOOT_SEED,
        python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
        verdict=verdict.to_dict("records"), delta=delta_df.to_dict("records"),
    )
    (args.outdir / "analysis_meta.json").write_text(
        json.dumps(meta, indent=1, ensure_ascii=False, default=str), encoding="utf-8")

    print(verdict.to_string(index=False), flush=True)
    print(f"Q3 AGGREGATION DONE -> {args.outdir}", flush=True)


if __name__ == "__main__":
    main()
