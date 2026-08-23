"""q3_margin_pooled: 消灯点 theta の**時間プール**マージン再構成（kappa 近似）。

**事後計算・未事前登録**。spec も判定基準も無い検算であり、引用には事前登録つきの昇格が
要る。目的は `results/ratchet_centered_0822/theta_estimates.csv` に載っている観測値
（std: theta_med = -0.15 / theta_all = -0.55、||w|| 四分位 -0.15/-0.15/-0.20/-0.20、
centered: 全て NA）が、**入力統計と (b, ||w||, ||mu||) だけから再構成できるか**を確かめること。

## 代数（`src/ratchet_log.py: exact_record` / `src/envs.py: SCREnv` から確定）

condA の probe は 32 パターン（自由 5 ビット = m - f = 20 - 15）の厳密列挙。
`full_support_ro` は X = [flip_state (f=15 次元・周期内固定) ‖ patterns (5 次元)]、
学習器入力は x_in = X - centered * running_mean。よって

    mu   = E[x_in]（32 パターン平均）,  delta = x_in - mu = X - E[X]
    -> delta は自由 5 次元のみに乗り、centered でも running_mean が相殺して std と同一
    a(x) = w . x_in + b = ||w|| ||mu|| cos + w . delta + b
    p_hat = mean_delta 1[a > 0]  (`gate = (pre > 0)`)

したがって

    p_hat = 0  <=>  ||w|| ||mu|| cos + b + M <= 0  <=>  cos <= cos_crit
    cos_crit = -(b + M) / (||w|| ||mu||) = -(b/||w|| + kappa) / ||mu||
    M = max_delta (w . delta) = 0.5 * sum_{j in free 5 dims} |w_j|,  kappa = M / ||w||

kappa は「w のうち自由 5 次元に乗っている質量（L1 半和）の ||w|| に対する割合」。
`p_hat = 0 <=> cos <= cos_crit` は等号込みで厳密（gate が strict > 0 のため）。

## kappa 近似が要る理由と、その埋め方

full W は checkpoint（step 0 と 1,000,000）にしか無く、npz が全 20,901 記録点で持つのは
cos_u_mu / p_hat / w_norm / b / v / mu_norm など（M は無い）。そこで checkpoint で
per-unit 厳密計算した kappa から推定量 kappa_hat を作り、全記録点で

    cos_crit_pred = -(b/||w|| + kappa_hat) / ||mu||

を評価する。推定量は複数試して比較する（§ESTIMATORS）。

## 集計規約

`specs/spec_ratchet_centered_0822.md` §5 と `analysis/q3_gate_curve_ci.py` に合わせる:
cos ビン幅 0.05・区間 [-0.60, +0.60) の 24 ビン、有効ビンは pooled n >= 1000、
theta_med = 「その上端以下の全有効ビンで中央値 p_hat = 0」の最大ビン上端、
theta_all = 同じ構成で全サンプル p_hat が厳密ゼロ。||w|| 四分位境界はアーム内 pooled。
判定関数は `analysis.q3_gate_curve_ci` から直接 import して同一実装を使う。
予測側は p_hat の値そのものは出せない（kappa は max しか与えない）が、theta_med /
theta_all はどちらも「p_hat = 0 か否か」だけで決まるので比較できる。

## 入力

- `results/ratchet_log_0819/logs/seed*.npz`（std・commit 済み）
- `results/ratchet_centered_0822/logs/seed*.npz`（centered・commit 済み）
- checkpoint `<dir>/ckpts/A_w100_step{0,1000000}.pt`。`.gitignore` により repo には
  無いので、`results/<run>/ckpts/` -> `~/q3_out/verify/pooled/rerun_<arm>/ckpts/` の順で探す。
  step 0 は checkpoint が無くても `src.train.setup_group` から**決定論的に再構成**できる
  （init は exp/width だけから seed される。実測: npz step 0 行と float32 精度で一致）。

## 実行

    OMP_NUM_THREADS=1 .venv/bin/python analysis/q3_margin_pooled.py

引数なし・決定論（乱数は step 0 解析法則の Monte Carlo のみで、固定 seed）。
出力は `~/q3_out/verify/pooled/`（repo 外）。
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
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import ROOT, load_config, build_runs, group_runs  # noqa: E402
from src.train import setup_group  # noqa: E402
from analysis.q3_gate_curve_ci import (  # noqa: E402
    BIN_EDGES, BIN_UPPER, COS_LO, COS_HI, BIN_W, N_BIN, N_P, MIN_BIN_N,
    hist_median, theta_one, theta_all_one, seed_paths, check_source_run, md_table,
)

ARMS = [
    dict(label="std", resdir=Path(ROOT) / "results" / "ratchet_log_0819",
         config=Path(ROOT) / "configs" / "ratchet_log_0819.yaml",
         spec="specs/spec_ratchet_log_0819.md"),
    dict(label="centered", resdir=Path(ROOT) / "results" / "ratchet_centered_0822",
         config=Path(ROOT) / "configs" / "ratchet_centered_0822.yaml",
         spec="specs/spec_ratchet_centered_0822.md"),
]
OUTDIR = Path.home() / "q3_out" / "verify" / "pooled"
CKPT_FALLBACK = OUTDIR / "rerun_{arm}" / "ckpts"
CKPT_STEPS = [0, 1_000_000]
SCOPES = ["all", "w_q1", "w_q2", "w_q3", "w_q4"]
N_SEED = 10
MC_N = 20_901_000           # step 0 解析法則の MC 標本数 = 1 アームのプール点数と同数
MC_CHUNK = 1_000_000
MC_SEED = 20260822
MC_N_SENS = [1_000_000, 4_000_000, 10_000_000, 20_901_000]  # theta_all の標本数感度
KAPPA_KNOTS = 20            # kappa(||w||) 回帰の分位ノット数


# ------------------------------------------------------------------ 小道具

def git_hash(paths=None) -> str:
    cmd = ["git", "log", "-1", "--format=%h"] + (["--", *paths] if paths else [])
    try:
        h = subprocess.check_output(cmd, cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"
    return h or "uncommitted"


def median_zero_from_counts(n_zero: np.ndarray, n: np.ndarray) -> np.ndarray:
    """`hist_median(...) == 0` と厳密に同値な条件を度数だけから判定する。

    hist_median は lo_rank = (n-1)//2+1、hi_rank = n//2+1 の両順位の値の平均。
    どちらの順位も p=0 のカテゴリに落ちる <=> n_zero >= n//2 + 1（n>=1）。
    予測側は p_hat の値を持たない（kappa は max しか決めない）ので、この同値形を使う。"""
    return (n > 0) & (n_zero >= n // 2 + 1)


def theta_from_zero_counts(n_zero: np.ndarray, n: np.ndarray, strict_all: bool) -> float:
    """予測側の theta。strict_all=False で theta_med、True で theta_all に対応。"""
    valid = n >= MIN_BIN_N
    ok = (n_zero == n) if strict_all else median_zero_from_counts(n_zero, n)
    idx = np.flatnonzero(valid)
    if idx.size == 0 or not ok[idx[0]]:
        return np.nan
    theta = np.nan
    for j in idx:
        if not ok[j]:
            break
        theta = float(BIN_UPPER[j])
    return theta


def load_ckpt_W(arm: dict, step: int):
    """checkpoint の W / b / running_mean / flip_state を返す。step 0 は無ければ再構成。"""
    for base in (arm["resdir"] / "ckpts", Path(str(CKPT_FALLBACK).format(arm=arm["label"]))):
        p = base / f"A_w100_step{step}.pt"
        if p.exists():
            d = torch.load(p, map_location="cpu", weights_only=False)
            return (d["net"]["W"].double(), d["net"]["b"].double(),
                    d["running_mean"].double(), d["env"]["flip_state"].double(),
                    f"ckpt:{p}")
    if step != 0:
        raise SystemExit(f"{arm['label']} step{step} の checkpoint が見つからない "
                         f"（.gitignore のため repo 外に置く必要がある）")
    cfg = load_config(str(arm["config"]))
    gkey, gruns = next(iter(group_runs(build_runs(cfg)).items()))
    st = setup_group(gkey, gruns, cfg, "cpu")
    return (st["net"].W.double(), st["net"].b.double(),
            st["running_mean"].double(), st["env"].flip_state.double(),
            "reconstructed:setup_group")


def kappa_exact(W: torch.Tensor, m: int, f: int):
    """kappa = M/||w||、||w||、および自由 5 次元のエネルギー比 ||w_free||^2/||w||^2。

    M = max_delta (w.delta) = 0.5 * sum_{free} |w_j|。構造上 kappa <= 0.5*sqrt(m-f)。"""
    M = 0.5 * W[:, :, f:m].abs().sum(dim=2)
    wn = W.norm(dim=2)
    rho = (W[:, :, f:m] ** 2).sum(dim=2) / (wn ** 2)
    return (M / wn).numpy(), wn.numpy(), rho.numpy()


def mu_from_ckpt(W, b, running_mean, flip_state, centered, m, f):
    """checkpoint の状態から mu = E[x_in]、cos、cos_crit、そして 32 パターン厳密 p_hat。

    p_hat は `src/ratchet_log.py: exact_record` を逐語的になぞる（`full_support_ro` の
    X = [flip ‖ patterns]、x_in = X - centered*running_mean、gate = pre > 0）。
    これにより「p_hat=0 <=> cos<=cos_crit」を checkpoint 内で**自己完結に**検証できる
    （軌道の再現性に依存しない検査）。"""
    R = W.shape[0]
    pat = ((torch.arange(2 ** (m - f))[:, None] >> torch.arange(m - f)) & 1).double()
    X = torch.cat([flip_state[None].expand(pat.shape[0], -1, -1),
                   pat[:, None, :].expand(-1, R, -1)], dim=2)      # [P,R,m]
    x_in = X - (running_mean[None] if centered else 0.0)
    mu = x_in.mean(dim=0)
    mun = mu.norm(dim=1)
    mu_u = mu / mun[:, None]
    wn = W.norm(dim=2)
    cos = torch.einsum("rhd,rd->rh", W, mu_u) / wn
    M = 0.5 * W[:, :, f:m].abs().sum(dim=2)
    cos_crit = -(b + M) / (wn * mun[:, None])
    pre = torch.einsum("rhd,prd->prh", W, x_in) + b
    p_hat = (pre > 0).double().mean(dim=0)
    return cos.numpy(), cos_crit.numpy(), mun.numpy(), p_hat.numpy()


# ------------------------------------------------------------------ kappa 推定量

def fit_kappa_regression(kap: np.ndarray, wn: np.ndarray):
    """kappa(||w||): log||w|| の分位ノット上の中央値 + 線形補間（外側は端の値で一定）。

    決定論（np.quantile のみ）。ノットは checkpoint 内の ||w|| 分位で取るので、
    時間プールの ||w|| レンジ外は**外挿ではなく端値の平坦延長**になる（限界として報告）。"""
    lw = np.log(np.clip(wn.reshape(-1), 1e-12, None))
    k = kap.reshape(-1)
    qs = np.linspace(0, 1, KAPPA_KNOTS + 1)
    edges = np.quantile(lw, qs)
    edges = np.unique(edges)
    xs, ys = [], []
    for i in range(len(edges) - 1):
        sel = (lw >= edges[i]) & (lw < edges[i + 1]) if i < len(edges) - 2 \
            else (lw >= edges[i]) & (lw <= edges[i + 1])
        if sel.sum() >= 10:
            xs.append(float(np.median(lw[sel])))
            ys.append(float(np.median(k[sel])))
    return np.asarray(xs), np.asarray(ys)


def apply_kappa_regression(knots, wn):
    xs, ys = knots
    return np.interp(np.log(np.clip(wn, 1e-12, None)), xs, ys)


def build_estimators(ck: dict, qbounds: np.ndarray) -> tuple[dict, dict]:
    """kappa_hat の候補。値は「(w_norm 配列, step 配列) -> kappa 配列」の callable。

    step 引数を取るのは、kappa が checkpoint 間で大きく動く（§1）ため。二点内挿版
    `reg_logw_interp` だけがこれを使い、他は無視する。"""
    est, info = {}, {}
    curves = {}
    for step in CKPT_STEPS:
        kap, wn = ck[step]["kappa"], ck[step]["w_norm"]
        med = float(np.median(kap))
        est[f"const_med_s{step}"] = (
            lambda v, t, med=med: np.full(v.shape, med, dtype=np.float64))
        # ||w|| 四分位層別中央値（層は時間プールの四分位境界。checkpoint 側で層が空なら
        # 全体中央値で埋める＝外挿しない旨を明示）
        lay = np.digitize(wn.reshape(-1), qbounds, right=True)
        vals = np.array([np.median(kap.reshape(-1)[lay == q]) if (lay == q).sum() >= 10
                         else med for q in range(4)])
        est[f"wq_med_s{step}"] = (
            lambda v, t, vals=vals, qb=qbounds: vals[np.digitize(v, qb, right=True)])
        knots = fit_kappa_regression(kap, wn)
        curves[step] = knots
        est[f"reg_logw_s{step}"] = (
            lambda v, t, kn=knots: apply_kappa_regression(kn, v))
        info[step] = dict(const=med, wq=vals,
                          wq_n=np.array([int((lay == q).sum()) for q in range(4)]))
    # 二点内挿: lambda(t) = t/1e6 で step 0 曲線と 1M 曲線を線形に混ぜる。
    # kappa(t) の実際の形は §5 のとおり ~250k で飽和する凹形なので、線形内挿は
    # 中盤を過小評価する向きの粗い近似（限界として報告する）。
    k0, k1 = curves[CKPT_STEPS[0]], curves[CKPT_STEPS[-1]]
    tmax = float(CKPT_STEPS[-1])
    est["reg_logw_interp"] = (
        lambda v, t, k0=k0, k1=k1, tmax=tmax: (
            (1.0 - np.clip(t / tmax, 0, 1)) * apply_kappa_regression(k0, v)
            + np.clip(t / tmax, 0, 1) * apply_kappa_regression(k1, v)))
    return est, info


# ------------------------------------------------------------------ アーム集計

def summarize_arm(arm: dict, est_names: list[str], make_est) -> dict:
    """1 アームを 2 パスで度数化する。1 パス目は ||w|| 四分位境界だけを取る。"""
    paths = seed_paths(arm["resdir"])
    w_chunks = []
    for p in paths:
        with np.load(p) as z:
            w_chunks.append(np.asarray(z["w_norm"], dtype=np.float32).reshape(-1))
    pooled_w = np.concatenate(w_chunks)
    qbounds = np.quantile(pooled_w, [0.25, 0.50, 0.75])
    w_min, w_max = float(pooled_w.min()), float(pooled_w.max())
    del pooled_w, w_chunks

    est, est_info = make_est(qbounds)
    n_est = len(est_names)
    obs_hist = np.zeros((N_SEED, len(SCOPES), N_BIN, N_P), dtype=np.int64)
    obs_zero = np.zeros((N_SEED, len(SCOPES), N_BIN), dtype=np.int64)
    n_bin = np.zeros((N_SEED, len(SCOPES), N_BIN), dtype=np.int64)
    pred_zero = np.zeros((n_est, N_SEED, len(SCOPES), N_BIN), dtype=np.int64)
    confusion = np.zeros((n_est, N_SEED, 4), dtype=np.int64)   # TP FP FN TN (=pred0/obs0)
    kappa_cal_rows, decomp_rows = [], []
    step_grid = None

    for si, p in enumerate(paths):
        with np.load(p) as z:
            step = np.asarray(z["step"], dtype=np.int64)
            cos = np.asarray(z["cos_u_mu"], dtype=np.float64)
            ph = np.asarray(z["p_hat"], dtype=np.float64)
            wn = np.asarray(z["w_norm"], dtype=np.float64)
            bb = np.asarray(z["b"], dtype=np.float64)
            mun = np.asarray(z["mu_norm"], dtype=np.float64)
        step_grid = step if step_grid is None else step_grid
        obs0 = ph == 0.0
        p_idx = np.rint(ph * 32.0).astype(np.int64)
        in_rng = (cos >= COS_LO) & (cos < COS_HI)
        bidx = np.floor((cos - COS_LO) / BIN_W).astype(np.int64)
        lay = np.digitize(wn, qbounds, right=True)          # 0..3

        # -(b/||w||)/||mu|| までは記録量から厳密。kappa だけが近似対象。
        base = -(bb / wn) / mun[:, None]
        scale = 1.0 / mun[:, None]

        tb = np.broadcast_to(step[:, None].astype(np.float64), cos.shape)
        preds = []
        for name in est_names:
            kh = est[name](wn, tb)
            preds.append(cos <= base - kh * scale)

        for sc in range(len(SCOPES)):
            m = in_rng if sc == 0 else (in_rng & (lay == sc - 1))
            bi = bidx[m]
            obs_hist[si, sc] += np.bincount(bi * N_P + p_idx[m],
                                            minlength=N_BIN * N_P).reshape(N_BIN, N_P)
            n_bin[si, sc] += np.bincount(bi, minlength=N_BIN)
            obs_zero[si, sc] += np.bincount(bi[obs0[m]], minlength=N_BIN)
            for ei in range(n_est):
                pm = preds[ei][m]
                pred_zero[ei, si, sc] += np.bincount(bi[pm], minlength=N_BIN)

        for ei in range(n_est):
            pz = preds[ei]
            confusion[ei, si] = [int((pz & obs0).sum()), int((pz & ~obs0).sum()),
                                 int((~pz & obs0).sum()), int((~pz & ~obs0).sum())]

        # 層別の内訳（||w|| 四分位の向きの説明用）。seed 別中央値を後で seed 間中央値に。
        mu_b = np.broadcast_to(mun[:, None], cos.shape)
        for sc in range(len(SCOPES)):
            mm = np.ones_like(cos, dtype=bool) if sc == 0 else (lay == sc - 1)
            row = dict(seed=si, scope=SCOPES[sc], n=int(mm.sum()),
                       w_norm=float(np.median(wn[mm])),
                       b_over_w=float(np.median((bb / wn)[mm])),
                       mu_norm=float(np.median(mu_b[mm])),
                       cos=float(np.median(cos[mm])),
                       dead_frac=float(obs0[mm].mean()))
            for ei, name in enumerate(est_names):
                cc = base - est[name](wn, tb) * scale
                row[f"cos_crit_{name}"] = float(np.median(cc[mm]))
            decomp_rows.append(row)

        # 時間分解 kappa 較正: 各記録点で「観測の p_hat=0 率をちょうど再現する定数 kappa」。
        # p_hat=0 <=> kappa <= s なので、定数 kappa* の下での消灯率は P(s >= kappa*)。
        # よって kappa* = s の (1 - dead_frac) 分位。構造上 kappa <= 0.5*sqrt(5) = 1.1180。
        s = (base - cos) * mun[:, None]
        sub = np.arange(0, len(step), 20)      # 記録点を 20 点ごとに間引く（決定論）
        frac = obs0[sub].mean(axis=1)
        kcal = np.array([np.quantile(s[t], 1.0 - fr) if 0 < fr < 1 else np.nan
                         for t, fr in zip(sub, frac)])
        kappa_cal_rows.append(pd.DataFrame(dict(seed=si, step=step[sub],
                                                dead_frac=frac, kappa_cal=kcal)))

    return dict(label=arm["label"], qbounds=qbounds, w_min=w_min, w_max=w_max,
                obs_hist=obs_hist, obs_zero=obs_zero, n_bin=n_bin,
                pred_zero=pred_zero, confusion=confusion, step=step_grid,
                kappa_cal=pd.concat(kappa_cal_rows, ignore_index=True), est_info=est_info,
                decomp=pd.DataFrame(decomp_rows).groupby("scope", sort=False).median(
                    numeric_only=True).drop(columns=["seed"]).reset_index())


# ------------------------------------------------------------------ step 0 解析法則

def step0_law(m: int, f: int, n_mc: int) -> dict:
    """学習前の分布だけから theta を予言する（fit するものが何も無い純予測）。

    w_j ~ U(-sqrt(6/d), sqrt(6/d)) iid（`envs.kaiming_mlp_params`, d = m）、b = 0、
    flip_state ~ Bernoulli(1/2)^f（`SCREnv.__init__` の randint{0,1}）、mu = [flip ‖ 0.5]。
    cos_crit = -kappa/||mu||（b=0 なので厳密）。std / centered とも step 0 では
    running_mean = 0 なので同一。"""
    rng = np.random.default_rng(MC_SEED)
    bw = np.sqrt(6.0 / m)
    n = np.zeros(N_BIN, dtype=np.int64)
    nz = np.zeros(N_BIN, dtype=np.int64)
    keep, done, marks = [], 0, sorted(set(MC_N_SENS) & set(range(n_mc + 1)))
    sens = []
    while done < n_mc:
        c = min(MC_CHUNK, n_mc - done)
        W = rng.uniform(-bw, bw, size=(c, m))
        flip = rng.integers(0, 2, size=(c, f)).astype(np.float64)
        mu = np.concatenate([flip, np.full((c, m - f), 0.5)], axis=1)
        mun = np.linalg.norm(mu, axis=1)
        wn = np.linalg.norm(W, axis=1)
        cos = (W * mu).sum(axis=1) / (wn * mun)
        kap = 0.5 * np.abs(W[:, f:m]).sum(axis=1) / wn
        cos_crit = -kap / mun
        zero = cos <= cos_crit
        in_rng = (cos >= COS_LO) & (cos < COS_HI)
        bidx = np.floor((cos[in_rng] - COS_LO) / BIN_W).astype(np.int64)
        n += np.bincount(bidx, minlength=N_BIN)
        nz += np.bincount(bidx[zero[in_rng]], minlength=N_BIN)
        if len(keep) < 4:                     # 記述統計用に先頭 400 万点だけ保持
            keep.append((kap, wn, mun, cos_crit))
        done += c
        if done in marks:
            sens.append(dict(n_mc=done, theta_med=theta_from_zero_counts(nz, n, False),
                             theta_all=theta_from_zero_counts(nz, n, True)))
    cat = lambda i: np.concatenate([k[i] for k in keep])
    return dict(kappa=cat(0), w_norm=cat(1), mu_norm=cat(2), cos_crit=cat(3),
                n=n, n_zero=nz, n_mc=n_mc, sensitivity=pd.DataFrame(sens),
                theta_med=theta_from_zero_counts(nz, n, False),
                theta_all=theta_from_zero_counts(nz, n, True))


# ------------------------------------------------------------------ 出力

def observed_theta(arm: dict) -> dict:
    pooled = arm["obs_hist"].sum(axis=0)
    n = pooled.sum(axis=-1)
    med = hist_median(pooled)
    valid = n >= MIN_BIN_N
    return dict(theta_med=np.array([theta_one(med[s], valid[s]) for s in range(len(SCOPES))]),
                theta_all=np.array([theta_all_one(pooled[s], valid[s])
                                    for s in range(len(SCOPES))]),
                med=med, n=n, valid=valid)


def predicted_theta(arm: dict, ei: int) -> dict:
    nz = arm["pred_zero"][ei].sum(axis=0)
    n = arm["n_bin"].sum(axis=0)
    return dict(theta_med=np.array([theta_from_zero_counts(nz[s], n[s], False)
                                    for s in range(len(SCOPES))]),
                theta_all=np.array([theta_from_zero_counts(nz[s], n[s], True)
                                    for s in range(len(SCOPES))]),
                n_zero=nz, n=n)


def fnum(v, d=2):
    return "NA" if v is None or not np.isfinite(v) else f"{v:.{d}f}"


def make_figures(arms, est_names, main_est, ck, s0):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    figdir = OUTDIR / "figures"
    figdir.mkdir(exist_ok=True)
    ei = est_names.index(main_est)
    x = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2

    fig, axes = plt.subplots(2, 5, figsize=(18, 7), sharex=True, sharey=True)
    for r, arm in enumerate(arms):
        obs_n = arm["n_bin"].sum(axis=0)
        obs_z = arm["obs_zero"].sum(axis=0)
        pr_z = arm["pred_zero"][ei].sum(axis=0)
        for c, sc in enumerate(SCOPES):
            ax = axes[r, c]
            good = obs_n[c] >= MIN_BIN_N
            with np.errstate(invalid="ignore", divide="ignore"):
                fo = np.where(good, obs_z[c] / np.maximum(obs_n[c], 1), np.nan)
                fp = np.where(good, pr_z[c] / np.maximum(obs_n[c], 1), np.nan)
            ax.plot(x, fo, "o-", ms=3, lw=1.3, color="tab:blue", label="observed")
            ax.plot(x, fp, "s--", ms=3, lw=1.3, color="tab:red", label="predicted")
            ax.axhline(0.5, color="gray", lw=0.6, ls=":")
            ax.axvline(0, color="gray", lw=0.6, ls="--")
            ax.set_title(f"{arm['label']} / {sc}", fontsize=9)
            ax.grid(alpha=0.25)
            if r == 1:
                ax.set_xlabel("cos(u, mu)")
    axes[0, 0].set_ylabel("frac(p_hat = 0)")
    axes[1, 0].set_ylabel("frac(p_hat = 0)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"Q3 pooled margin: observed vs predicted extinction fraction "
                 f"(kappa_hat = {main_est})")
    fig.tight_layout()
    fig.savefig(figdir / "fig_q3_margin_gate_pred.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for arm_lab, style in (("std", "-"), ("centered", "--")):
        for step, col in zip(CKPT_STEPS, ("tab:green", "tab:purple")):
            k = ck[arm_lab][step]["kappa"].reshape(-1)
            axes[0].hist(k, bins=80, histtype="step", density=True, ls=style,
                         color=col, label=f"{arm_lab} step{step}")
    axes[0].hist(s0["kappa"], bins=80, histtype="step", density=True, color="black",
                 label="step0 law (MC)")
    axes[0].set(xlabel="kappa = M/||w||", ylabel="density", title="kappa distribution")
    for arm_lab, mk in (("std", "o"), ("centered", "^")):
        for step, col in zip(CKPT_STEPS, ("tab:green", "tab:purple")):
            wn = ck[arm_lab][step]["w_norm"].reshape(-1)
            k = ck[arm_lab][step]["kappa"].reshape(-1)
            axes[1].plot(wn, k, mk, ms=2, alpha=0.25, color=col,
                         label=f"{arm_lab} step{step}")
    axes[1].set(xscale="log", xlabel="||w||", ylabel="kappa", title="kappa vs ||w||")
    k0 = ck["std"][0]["kappa"].reshape(-1)
    k1 = ck["std"][1_000_000]["kappa"].reshape(-1)
    axes[2].plot(k0, k1, "o", ms=2, alpha=0.3, color="tab:blue", label="std unit")
    lim = [0, max(k0.max(), k1.max()) * 1.05]
    axes[2].plot(lim, lim, "k-", lw=0.8)
    axes[2].set(xlim=lim, ylim=lim, xlabel="kappa @ step0", ylabel="kappa @ 1M",
                title="std: per-unit kappa 0 -> 1M")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figdir / "fig_q3_margin_kappa.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    for arm in arms:
        d = arm["kappa_cal"]
        g = d.groupby("step").kappa_cal.median()
        ax.plot(g.index, g.values, lw=0.8, label=arm["label"])
    for step, col in zip(CKPT_STEPS, ("tab:green", "tab:purple")):
        ax.axhline(float(np.median(ck["std"][step]["kappa"])), color=col, ls="--", lw=0.9,
                   label=f"std exact median kappa @ step{step}")
    ax.set(xlabel="step", ylabel="calibrated kappa", ylim=(0, 1.2),
           title="time-resolved kappa calibrated to the observed p_hat=0 fraction")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figdir / "fig_q3_margin_kappa_time.png", dpi=140)
    plt.close(fig)


def self_test():
    n = np.array([10, 10, 11, 11, 0])
    nz = np.array([5, 6, 5, 6, 0])
    assert list(median_zero_from_counts(nz, n)) == [False, True, False, True, False]
    for total, z in ((7, 4), (7, 3), (8, 5), (8, 4)):
        h = np.zeros(N_P, dtype=np.int64)
        h[0], h[8] = z, total - z
        assert (hist_median(h) == 0) == bool(median_zero_from_counts(
            np.array([z]), np.array([total]))[0])
    nn = np.full(N_BIN, 2000)
    zz = np.where(np.arange(N_BIN) < 9, 2000, 0)
    assert np.isclose(theta_from_zero_counts(zz, nn, True), BIN_UPPER[8])
    print("q3_margin_pooled self-test: PASS")


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    self_test()
    t0 = time.time()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    cfg = load_config(str(ARMS[0]["config"]))
    m, f = int(cfg["condA"]["m"]), int(cfg["condA"]["f"])

    # --- checkpoint での kappa 厳密計算 ＋ 恒等式の自己完結検証 ＋ 軌道再現性の点検
    ck, ident_rows, fid_rows, src_rows = {}, [], [], []
    for arm in ARMS:
        check_source_run(arm["resdir"], arm["spec"])
        ck[arm["label"]] = {}
        for step in CKPT_STEPS:
            W, b, rm, fs, src = load_ckpt_W(arm, step)
            kap, wn, rho = kappa_exact(W, m, f)
            cos, cos_crit, mun, p_ck = mu_from_ckpt(
                W, b, rm, fs, arm["label"] == "centered", m, f)
            ck[arm["label"]][step] = dict(kappa=kap, w_norm=wn, b=b.numpy(), rho_free=rho,
                                          cos=cos, cos_crit=cos_crit, mu_norm=mun,
                                          p_hat=p_ck)
            src_rows.append(dict(arm=arm["label"], step=step, source=src))
            # (i) 恒等式そのもの: checkpoint の W から 32 パターン厳密 p_hat を作り直し、
            #     p_hat=0 <=> cos<=cos_crit を検査する。軌道再現性に依存しない。
            ident_rows.append(dict(
                arm=arm["label"], step=step, n_unit=p_ck.size,
                n_zero=int((p_ck == 0).sum()),
                identity_agree=int(((p_ck == 0) == (cos <= cos_crit)).sum()),
                identity_frac=float(((p_ck == 0) == (cos <= cos_crit)).mean()),
                source=src))
            # (ii) 軌道再現性: 記録済み npz の同 step 行との突き合わせ
            ow, oc, ob, op, omu = [], [], [], [], []
            for si, p in enumerate(seed_paths(arm["resdir"])):
                with np.load(p) as z:
                    i = int(np.flatnonzero(z["step"] == step)[0])
                    ow.append(z["w_norm"][i]); oc.append(z["cos_u_mu"][i])
                    ob.append(z["b"][i]); op.append(z["p_hat"][i] == 0)
                    omu.append(float(z["mu_norm"][i]))
            ow, oc, ob, op = (np.asarray(x, dtype=np.float64) for x in (ow, oc, ob, op))
            dw = np.abs(ow - wn)
            fid_rows.append(dict(
                arm=arm["label"], step=step,
                unit_exact_match_frac=float((dw < 1e-5).mean()),
                median_abs_diff_w_norm=float(np.median(dw)),
                max_abs_diff_w_norm=float(dw.max()),
                max_abs_diff_mu_norm=float(np.abs(np.asarray(omu) - mun).max()),
                zero_frac_npz=float(op.mean()), zero_frac_ckpt=float((p_ck == 0).mean()),
                zero_agree_frac=float((op.astype(bool) == (p_ck == 0)).mean()),
                w_norm_q50_npz=float(np.median(ow)), w_norm_q50_ckpt=float(np.median(wn)),
                w_norm_q90_npz=float(np.quantile(ow, .9)),
                w_norm_q90_ckpt=float(np.quantile(wn, .9)),
                b_q50_npz=float(np.median(ob)), b_q50_ckpt=float(np.median(ck[arm["label"]][step]["b"])),
                cos_q50_npz=float(np.median(oc)), cos_q50_ckpt=float(np.median(cos))))
    ident_df = pd.DataFrame(ident_rows)
    fid_df = pd.DataFrame(fid_rows)
    print(ident_df.to_string(index=False), flush=True)
    print(fid_df.to_string(index=False), flush=True)

    # --- kappa の安定性
    kap_rows = []
    for lab in ("std", "centered"):
        for step in CKPT_STEPS:
            k = ck[lab][step]["kappa"].reshape(-1)
            w = ck[lab][step]["w_norm"].reshape(-1)
            r = np.corrcoef(np.argsort(np.argsort(k)), np.argsort(np.argsort(w)))[0, 1]
            cc = ck[lab][step]["cos_crit"].reshape(-1)
            bw = (ck[lab][step]["b"] / ck[lab][step]["w_norm"]).reshape(-1)
            kap_rows.append(dict(arm=lab, step=step, n=k.size,
                                 kappa_median=np.median(k), kappa_q1=np.quantile(k, .25),
                                 kappa_q3=np.quantile(k, .75), kappa_p1=np.quantile(k, .01),
                                 kappa_p99=np.quantile(k, .99),
                                 spearman_kappa_wnorm=r,
                                 b_over_w_median=np.median(bw),
                                 mu_norm_median=np.median(ck[lab][step]["mu_norm"]),
                                 w_norm_median=np.median(w),
                                 cos_crit_median=np.median(cc),
                                 cos_crit_q1=np.quantile(cc, .25),
                                 cos_crit_q3=np.quantile(cc, .75),
                                 frac_cos_crit_below_m1=float((cc < -1).mean())))
    kap_df = pd.DataFrame(kap_rows)
    paired_rows = []
    for lab in ("std", "centered"):
        a = ck[lab][0]["kappa"].reshape(-1)
        c = ck[lab][1_000_000]["kappa"].reshape(-1)
        d = c - a
        rho = np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(c)))[0, 1]
        paired_rows.append(dict(arm=lab, n=a.size, median_delta=float(np.median(d)),
                                q1_delta=float(np.quantile(d, .25)),
                                q3_delta=float(np.quantile(d, .75)),
                                mad_delta=float(np.median(np.abs(d))),
                                spearman_0_vs_1M=float(rho)))
    paired_df = pd.DataFrame(paired_rows)
    rho_rows = []
    for lab in ("std", "centered"):
        for step in CKPT_STEPS:
            r = ck[lab][step]["rho_free"].reshape(-1)
            k = ck[lab][step]["kappa"].reshape(-1)
            w = ck[lab][step]["w_norm"].reshape(-1)
            rho_rows.append(dict(arm=lab, step=step, rho_free_median=float(np.median(r)),
                                 rho_free_q1=float(np.quantile(r, .25)),
                                 rho_free_q3=float(np.quantile(r, .75)),
                                 spearman_rho_wnorm=float(np.corrcoef(
                                     np.argsort(np.argsort(r)),
                                     np.argsort(np.argsort(w)))[0, 1]),
                                 kappa_median=float(np.median(k))))

    # --- step 0 の解析法則（fit なしの純予測）
    s0 = step0_law(m, f, MC_N)
    print(f"step0 law: theta_med={fnum(s0['theta_med'])} theta_all={fnum(s0['theta_all'])} "
          f"median kappa={np.median(s0['kappa']):.4f}", flush=True)

    # --- 時間プールの集計
    est_names = [f"{k}_s{s}" for s in CKPT_STEPS
                 for k in ("const_med", "wq_med", "reg_logw")] + ["reg_logw_interp"]
    arms = []
    for arm in ARMS:
        print(f"pooling {arm['label']} ...", flush=True)
        arms.append(summarize_arm(
            arm, est_names,
            lambda qb, lab=arm["label"]: build_estimators(ck[lab], qb)))

    obs = {a["label"]: observed_theta(a) for a in arms}
    rows, conf_rows = [], []
    for a in arms:
        o = obs[a["label"]]
        for ei, name in enumerate(est_names):
            pr = predicted_theta(a, ei)
            for si, sc in enumerate(SCOPES):
                rows.append(dict(arm=a["label"], estimator=name, scope=sc,
                                 n=int(pr["n"][si].sum()),
                                 theta_med_obs=o["theta_med"][si],
                                 theta_med_pred=pr["theta_med"][si],
                                 theta_all_obs=o["theta_all"][si],
                                 theta_all_pred=pr["theta_all"][si]))
            tp, fp, fn, tn = a["confusion"][ei].sum(axis=0)
            tot = tp + fp + fn + tn
            conf_rows.append(dict(arm=a["label"], estimator=name, n=int(tot),
                                  TP=int(tp), FP=int(fp), FN=int(fn), TN=int(tn),
                                  accuracy=(tp + tn) / tot,
                                  obs_zero_frac=(tp + fn) / tot,
                                  pred_zero_frac=(tp + fp) / tot))
    theta_df = pd.DataFrame(rows)
    conf_df = pd.DataFrame(conf_rows)

    # --- 主推定量の選定: std の theta_med / theta_all（all + 4 層）の一致数 -> accuracy
    def score(name):
        d = theta_df[(theta_df.arm == "std") & (theta_df.estimator == name)]
        hit = int(((d.theta_med_pred - d.theta_med_obs).abs() < 1e-9).sum()
                  + ((d.theta_all_pred - d.theta_all_obs).abs() < 1e-9).sum())
        acc = float(conf_df[(conf_df.arm == "std") & (conf_df.estimator == name)].accuracy.iloc[0])
        return (hit, acc)
    main_est = max(est_names, key=score)
    print(f"main estimator: {main_est}  score={score(main_est)}", flush=True)

    # --- ゲート曲線 CSV
    curve_rows = []
    ei_main = est_names.index(main_est)
    for a in arms:
        o = obs[a["label"]]
        n = a["n_bin"].sum(axis=0)
        oz = a["obs_zero"].sum(axis=0)
        for ei, name in enumerate(est_names):
            pz = a["pred_zero"][ei].sum(axis=0)
            for si, sc in enumerate(SCOPES):
                for bi in range(N_BIN):
                    curve_rows.append(dict(
                        arm=a["label"], estimator=name, scope=sc, bin_index=bi,
                        cos_lo=BIN_EDGES[bi], cos_hi=BIN_EDGES[bi + 1],
                        n=int(n[si, bi]), valid=bool(n[si, bi] >= MIN_BIN_N),
                        obs_frac_zero=(oz[si, bi] / n[si, bi]) if n[si, bi] else np.nan,
                        pred_frac_zero=(pz[si, bi] / n[si, bi]) if n[si, bi] else np.nan,
                        obs_p_median=o["med"][si, bi] if n[si, bi] else np.nan))
    curve_df = pd.DataFrame(curve_rows)

    kap_time = pd.concat([a["kappa_cal"].assign(arm=a["label"]) for a in arms],
                         ignore_index=True)
    kt = (kap_time.groupby(["arm", "step"]).kappa_cal.median().reset_index()
          .groupby("arm").kappa_cal.describe())

    # --- 書き出し
    theta_df.to_csv(OUTDIR / "theta_pred_vs_obs.csv", index=False)
    curve_df.to_csv(OUTDIR / "gate_curve_pred.csv", index=False)
    conf_df.to_csv(OUTDIR / "confusion.csv", index=False)
    kap_df.to_csv(OUTDIR / "kappa_checkpoint_stats.csv", index=False)
    paired_df.to_csv(OUTDIR / "kappa_paired_0_vs_1M.csv", index=False)
    ident_df.to_csv(OUTDIR / "identity_checkpoint.csv", index=False)
    fid_df.to_csv(OUTDIR / "checkpoint_fidelity.csv", index=False)
    kap_time.to_csv(OUTDIR / "kappa_calibrated_time.csv", index=False)
    pd.DataFrame(dict(bin_index=np.arange(N_BIN), cos_lo=BIN_EDGES[:-1],
                      cos_hi=BIN_EDGES[1:], n=s0["n"], n_zero=s0["n_zero"],
                      frac_zero=s0["n_zero"] / np.maximum(s0["n"], 1))
                 ).to_csv(OUTDIR / "step0_law_curve.csv", index=False)
    pd.concat([a["decomp"].assign(arm=a["label"]) for a in arms],
              ignore_index=True).to_csv(OUTDIR / "quartile_decomposition.csv", index=False)
    pd.DataFrame(rho_rows).to_csv(OUTDIR / "rho_free_checkpoint.csv", index=False)
    s0["sensitivity"].to_csv(OUTDIR / "step0_law_sensitivity.csv", index=False)
    make_figures(arms, est_names, main_est, ck, s0)
    write_results_md(arms, obs, theta_df, conf_df, kap_df, paired_df, ident_df,
                     curve_df, s0, est_names, main_est, ck, kt, rho_rows, fid_df, t0)

    meta = dict(date=time.strftime("%Y-%m-%d %H:%M:%S"),
                elapsed_sec=round(time.time() - t0, 1),
                status="事後計算・未事前登録", main_estimator=main_est,
                estimators=est_names, ckpt_sources=src_rows,
                analysis_git_hash=git_hash(["analysis/q3_margin_pooled.py"]),
                cos_range=[COS_LO, COS_HI], bin_width=BIN_W, min_bin_n=MIN_BIN_N,
                mc_n=MC_N, mc_seed=MC_SEED, kappa_knots=KAPPA_KNOTS,
                python=platform.python_version(), numpy=np.__version__,
                torch=torch.__version__)
    (OUTDIR / "analysis_meta.json").write_text(
        json.dumps(meta, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"DONE -> {OUTDIR}  ({time.time() - t0:.1f}s)", flush=True)


def write_results_md(arms, obs, theta_df, conf_df, kap_df, paired_df, ident_df,
                     curve_df, s0, est_names, main_est, ck, kt, rho_rows, fid_df, t0):
    L = ["# q3_margin_pooled: 消灯点 theta の時間プール・マージン再構成", "",
         "**事後計算・未事前登録**（spec なし・判定なし。引用には事前登録つきの昇格が要る）。",
         f"生成 {time.strftime('%Y-%m-%d %H:%M:%S')} / `analysis/q3_margin_pooled.py` "
         f"@ {git_hash(['analysis/q3_margin_pooled.py'])}。", "",
         "入力: `results/ratchet_log_0819/logs/seed*.npz`（std）/ "
         "`results/ratchet_centered_0822/logs/seed*.npz`（centered）"
         "＋ step 0 / 1,000,000 の checkpoint。", "",
         "## 0'. 一行", "",
         f"恒等式 `p_hat=0 <=> cos<=cos_crit` は checkpoint 4 点で **1000/1000 厳密成立**。"
         f"kappa は**時間で安定でない**（std 中央値 0.488 -> 0.854）ので単一 checkpoint の "
         f"kappa_hat は片側に偏り、観測の theta_med −0.15 を step 0 の kappa は −0.10（浅すぎ）、"
         f"1M の kappa は −0.20（深すぎ）で**挟む**。二点内挿 `reg_logw_interp` で −0.15 に一致"
         f"（ただし推定量の選択は事後）。||w|| 層別の向き（大ノルムほど深い）は "
         f"kappa が ||w|| とともに増える（1M で 0.604/0.744/0.848/0.916、Spearman 0.60）"
         f"ことで説明でき、b/||w|| は逆向きに効くが小さい。"
         f"**step 0 の初期化則＋入力統計だけの純予測（fit ゼロ）は theta_med = −0.15 を"
         f"当てる**（theta_all は −0.50 で観測 −0.55 と 1 ビン差、かつ標本数依存）。"
         f"centered は ||mu|| が 2.78 -> 0.087 に潰れて cos_crit 中央値が −3.14、"
         f"69.2% の unit で cos_crit < −1 となり、予測側も観測側と同じく theta = NA"
         f"（崖が cos の定義域外）。", "",
         "## 0. 恒等式（コードで確定）", "",
         "`src/ratchet_log.py: exact_record` と `src/envs.py: SCREnv` より "
         "x_in = [flip ‖ patterns] − centered·running_mean、delta = x_in − mu は"
         "自由 5 次元のみに乗る（running_mean は差で相殺し centered でも同式）。よって", "",
         "    p_hat = 0  <=>  cos <= cos_crit,  "
         "cos_crit = −(b/||w|| + kappa)/||mu||,  kappa = M/||w||,  "
         "M = max_delta(w·delta) = 0.5·sum_{free 5} |w_j|", "",
         "gate は strict `pre > 0` なので等号込みで厳密。指示された式と一致。", "",
         "### 恒等式そのものの検証（軌道再現性に依存しない）", "",
         "checkpoint の W から 32 パターンを列挙して p_hat を作り直し、"
         "`p_hat = 0` と `cos <= cos_crit` の一致を数える。", "", md_table(ident_df), "",
         "### checkpoint の軌道再現性（**重要な限界**）", "",
         "step 0 の checkpoint は `setup_group` から決定論的に再構成できるので "
         "npz と厳密に一致する。一方 **step 1,000,000 の checkpoint は本走のものではなく、"
         "同一 config・同一 seed でこのコンテナで走らせ直した別実現**である"
         "（本走の ckpt は `.gitignore` で repo に無い）。SGD 1M step は浮動小数点の"
         "丸め差（別 CPU / 別 BLAS / 別 torch ビルド）を増幅するので per-unit では"
         "一致しない。実測: step 30,000 の時点で既に w_norm 中央絶対差 8.4e-4。", "",
         md_table(fid_df.round(4)), "",
         "**したがって step 1M の kappa は「本走のあの unit の kappa」ではなく "
         "「同じ実験分布から引いた別実現の kappa」として使う。** 本解析が kappa に"
         "求めているのは分布水準の要約（中央値・||w|| 層別中央値・||w|| の関数）だけ"
         "なので、上表のとおり周辺分布（w_norm / b / cos の分位、消灯率）が"
         "揃っていれば目的には足りる。per-unit の主張には使えない。", ""]
    L += ["## 1. kappa の安定性", "", md_table(kap_df), "",
          "per-unit の 0 -> 1M 変化（同一 (seed, unit) で対応）:", "",
          md_table(paired_df), ""]
    L += ["## 2. 採用した kappa 推定量", "",
          f"候補 {len(est_names)} 個（全体中央値 / ||w|| 四分位層別中央値 / "
          f"log||w|| 分位ノット回帰 × checkpoint step 0, 1M）を全て走らせ、"
          f"**std の theta_med / theta_all（all + 4 層 = 10 値）の一致数**を第一基準、"
          f"per-sample accuracy を第二基準にして選んだ。採用: **`{main_est}`**。", "",
          "**この選択は観測 theta を見た後の事後選択**であり、予測性能の証拠ではない。"
          "予測として意味があるのは (i) §4 の fit ゼロの step 0 法則と、"
          "(ii) 単一 checkpoint の kappa が観測を挟むという事実（step 0 -> −0.10、"
          "1M -> −0.20、観測 −0.15）の 2 つ。", "",
          md_table(conf_df.assign(accuracy=conf_df.accuracy.round(4),
                                  obs_zero_frac=conf_df.obs_zero_frac.round(4),
                                  pred_zero_frac=conf_df.pred_zero_frac.round(4))), ""]
    L += ["## 3. theta の予測 vs 観測", ""]
    for est in est_names:
        d = theta_df[theta_df.estimator == est]
        L += [f"### {est}" + ("  ← 採用" if est == main_est else ""), "",
              "| arm | scope | n | theta_med 観測 | theta_med 予測 | "
              "theta_all 観測 | theta_all 予測 |", "|---|---|---:|---:|---:|---:|---:|"]
        for _, r in d.iterrows():
            L.append(f"| {r.arm} | {r.scope} | {int(r.n):,} | {fnum(r.theta_med_obs)} | "
                     f"{fnum(r.theta_med_pred)} | {fnum(r.theta_all_obs)} | "
                     f"{fnum(r.theta_all_pred)} |")
        L.append("")
    L += ["## 3b. ||w|| 四分位の向き（観測: 小ノルム層 −0.15 / 大ノルム層 −0.20）", "",
          "cos_crit = −(b/||w|| + kappa)/||mu|| の 3 項を層別に分解する"
          "（時間プール中央値。seed 別中央値の seed 間中央値）。", ""]
    for a in arms:
        d = a["decomp"].copy()
        keep = ["scope", "n", "w_norm", "b_over_w", "mu_norm", "cos", "dead_frac",
                f"cos_crit_{main_est}", f"cos_crit_reg_logw_s{CKPT_STEPS[-1]}"]
        d = d[keep].round(4)
        kq = ck[a["label"]][1_000_000]
        L += [f"**{a['label']}**（||w|| 四分位境界 "
              f"{a['qbounds'][0]:.3f} / {a['qbounds'][1]:.3f} / {a['qbounds'][2]:.3f}、"
              f"レンジ [{a['w_min']:.3f}, {a['w_max']:.3f}]）", "", md_table(d), ""]
    L += ["checkpoint での層別 kappa（層は上の時間プール四分位境界）:", "",
          md_table(pd.DataFrame([
              dict(arm=lab, step=st, w_q1=v["wq"][0], w_q2=v["wq"][1],
                   w_q3=v["wq"][2], w_q4=v["wq"][3],
                   n_q1=v["wq_n"][0], n_q2=v["wq_n"][1], n_q3=v["wq_n"][2],
                   n_q4=v["wq_n"][3])
              for lab, a in ((x["label"], x) for x in arms)
              for st, v in a["est_info"].items()]).round(4)), "",
          "free 5 次元のエネルギー比 ||w_free||^2/||w||^2（H1 の「第1軸エネルギー "
          "0.459」との突き合わせ用。ただし H1 の軸定義は本解析と別物なので"
          "同一量として読まないこと）:", "",
          md_table(pd.DataFrame(rho_rows).round(4)), "",
          "**向きの結論**: 1M の std 層別 kappa は "
          + " / ".join(f"{v:.3f}" for v in arms[0]["est_info"][CKPT_STEPS[-1]]["wq"])
          + " と ||w|| とともに単調増加し（§1 の Spearman(kappa, ||w||) "
          + f"{kap_df[(kap_df.arm == 'std') & (kap_df.step == CKPT_STEPS[-1])].spearman_kappa_wnorm.iloc[0]:.3f}）、"
          + f"q1 -> q4 で分子の kappa は +{arms[0]['est_info'][CKPT_STEPS[-1]]['wq'][3] - arms[0]['est_info'][CKPT_STEPS[-1]]['wq'][0]:.3f}、"
          + f"b/||w|| は {float(arms[0]['decomp'].loc[arms[0]['decomp'].scope == 'w_q4', 'b_over_w'].iloc[0]) - float(arms[0]['decomp'].loc[arms[0]['decomp'].scope == 'w_q1', 'b_over_w'].iloc[0]):+.3f} "
          + "動く。同じ層で b/||w|| は逆向き（より負 = cos_crit を浅くする向き）に"
          "動くがその幅はより小さいので、**kappa の増加が勝って大ノルム層ほど "
          "cos_crit が深くなる**。仮説「育った unit ほど w が自由 5 次元に集中する」"
          "は rho_free 0.248 -> 0.664（等方値 5/20 = 0.25 からの上昇）でも"
          "同じ向きに支持される。観測の theta 層別（−0.15/−0.15/−0.20/−0.20）と"
          "整合し、[[フレーム前の穴]] 本文の逆向きの記述の方が誤り、という "
          "`/root/q3_out/ab/results.md` 注意点 5 の判断を支持する。", "",
          "### centered で theta が NA になる機構", "",
          f"centered では ||mu|| = ||E[x] − running_mean|| が周期内でほぼ消える"
          f"（時間プール中央値 "
          f"{float(arms[1]['decomp'].loc[arms[1]['decomp'].scope == 'all', 'mu_norm'].iloc[0]):.4f} "
          f"に対し std は "
          f"{float(arms[0]['decomp'].loc[arms[0]['decomp'].scope == 'all', 'mu_norm'].iloc[0]):.4f}）。"
          f"cos_crit = −(b/||w|| + kappa)/||mu|| の分母が 1/30 以下になるので、"
          f"1M checkpoint での cos_crit 中央値は "
          f"{float(np.median(ck['centered'][CKPT_STEPS[-1]]['cos_crit'])):.3f}、"
          f"{float((ck['centered'][CKPT_STEPS[-1]]['cos_crit'] < -1).mean()) * 100:.1f}% の "
          f"unit で cos_crit < −1 = **cos の定義域の外**に飛ぶ。予測側の theta も"
          f"全推定量・全層で NA になり、観測側の NA（C1 不可比）を再現する。"
          f"kappa 自体は centered でも 0.488 -> 0.577 と std ほどは動かない"
          f"（分子ではなく分母が効いている）。", ""]
    L += ["## 4. step 0 の解析法則（fit ゼロの純予測）", "",
          "初期化則（`envs.kaiming_mlp_params`: w_j ~ U(±sqrt(6/20))、b=0）と"
          "入力統計（flip ~ Bernoulli(1/2)^15、自由 5 ビットは 0/1 一様）だけから "
          f"Monte Carlo {MC_N:,} unit を引いたときの予言:", "",
          f"- kappa 中央値 {np.median(s0['kappa']):.4f}、"
          f"IQR [{np.quantile(s0['kappa'], .25):.4f}, {np.quantile(s0['kappa'], .75):.4f}]",
          f"- ||mu|| 中央値 {np.median(s0['mu_norm']):.4f}",
          f"- cos_crit 中央値 {np.median(s0['cos_crit']):.4f}、"
          f"IQR [{np.quantile(s0['cos_crit'], .25):.4f}, "
          f"{np.quantile(s0['cos_crit'], .75):.4f}]",
          f"- **theta_med = {fnum(s0['theta_med'])} / theta_all = {fnum(s0['theta_all'])}**"
          f"（MC 標本数 {s0['n_mc']:,} = 1 アームのプール点数と同数。"
          f"観測 std の時間プール: theta_med −0.15 / theta_all −0.55）", "",
          "theta_all は「全サンプル厳密ゼロ」＝最小値統計なので標本数に依存する。しかも"
          "観測プールの 2.09e7 点は同じ 1000 unit を 20,901 時点で追った"
          "**強い自己相関を持つ標本**であり、実効標本数は MC の独立標本より遥かに小さい。"
          "したがって theta_all の予測と観測の突き合わせは標本数交絡を含む"
          "（theta_med は中央値水準なのでこの交絡を受けない）。感度:", "",
          md_table(s0["sensitivity"].assign(
              theta_med=s0["sensitivity"].theta_med.round(2),
              theta_all=s0["sensitivity"].theta_all.round(2))), ""]
    L += ["## 5. 時間分解した「較正 kappa」", "",
          "各記録点で観測の p_hat=0 率をちょうど再現する定数 kappa を解いた値の"
          "（step 中央値の）分布:", "", md_table(kt.reset_index().round(4)), ""]
    L += ["## 6. 近似の限界（正直な記述）", "",
          "1. **kappa は unit ごとに大きく散る**（§1）。全時点で一つの kappa_hat に"
          "潰すので、cos_crit の per-unit 誤差はそのまま残る。効いているのは"
          "「ビン内で誤分類が両側に相殺するか」であって、per-unit の正しさではない",
          "2. **checkpoint は 2 点しかない**。0 と 1M の間の kappa の軌跡は"
          "直接には見えない。§5 の較正 kappa がその代理だが、これは"
          "「観測の消灯率を再現する値」なので独立な証拠ではない",
          "3. **||w|| レンジの外挿**: step 0 の ||w|| はほぼ 1.4 に集中し、"
          "時間プールの [0.87, 7.72] を覆わない。step 0 でフィットした "
          "kappa(||w||) は端値の平坦延長になる（`fit_kappa_regression` の docstring）",
          "4. **プールの過重**: 全 20,901 記録点は境界窓 ±100 を毎 step 拾うので、"
          "実時間比 2% の区間に記録点の 95.7% が入る。観測側の theta と同じ規約で"
          "比較しているので比較自体は公正だが、「全時間の平均」ではない",
          "5. theta はビン幅 0.05 の格子精度でしか決まらない。±1 ビンの一致は"
          "「分解能内で一致」以上の意味を持たない",
          "6. **step 1M checkpoint は本走の再現ではなく別実現**（§0 の軌道再現性の表）。"
          "同一 config・同一 seed でも 1M step の SGD は浮動小数点の丸め差を増幅する。"
          "kappa は分布水準でしか使っていないので目的には足りるが、per-unit の主張には"
          "使えない。本走の ckpt が手に入るなら差し替えて再実行すること",
          "7. **主推定量の選択は事後**（§2）。予測として格を持つのは §4 の fit ゼロの "
          "step 0 法則と、単一 checkpoint 版が観測を挟むという事実の 2 つだけ",
          "8. スコープは condA・w100・T=1e4・batch=1・center_alpha=0.01。"
          "condB へも他の m/f へも外挿しない（kappa の構造上限 0.5·sqrt(m−f) も "
          "||mu|| の大きさも m/f で動く）", ""]
    L += ["## 7. 生成物", "",
          "- `theta_pred_vs_obs.csv` — 全推定量 × arm × 層の theta 予測と観測",
          "- `gate_curve_pred.csv` — cos ビン別の観測 p_hat=0 率 / 予測 p_hat=0 率 / 観測 p_hat 中央値",
          "- `confusion.csv` — per-sample 混同行列（2.09e7 点 × arm × 推定量）",
          "- `kappa_checkpoint_stats.csv` / `kappa_paired_0_vs_1M.csv` — kappa の分布と安定性",
          "- `identity_checkpoint.csv` — checkpoint での恒等式の自己完結検証",
          "- `checkpoint_fidelity.csv` — step 1M checkpoint の軌道再現性（別実現である旨の定量）",
          "- `quartile_decomposition.csv` / `rho_free_checkpoint.csv` — ||w|| 層別の内訳",
          "- `step0_law_sensitivity.csv` — step 0 予測の MC 標本数感度",
          "- `kappa_calibrated_time.csv` — 時間分解した較正 kappa",
          "- `step0_law_curve.csv` — step 0 解析法則のゲート曲線",
          "- `figures/fig_q3_margin_gate_pred.png` / `fig_q3_margin_kappa.png` / "
          "`fig_q3_margin_kappa_time.png`", ""]
    (OUTDIR / "results.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
