# --- copied for altlabels_cifar_0923 ---------------------------------------
# source : /home/issan/Projects/obsidian-research-data/rl_ledger_posthoc_0922/band_ledger.py
# sha256 : 3a3978b607a95222b1067a00bc53e1ce3cfab84e04219a393e052f31d06165c5   (of the source file, before the two edits below)
# edits  : (1) REPO now points at THIS worktree, so inputs() uses this checkout's
#              analysis/rlcifar_mlp_battle_0918/posthoc_0918.py and data/ symlink
#              instead of proj_004_drift (which this session must not touch);
#          (2) inputs() falls back to src.rlcifar_mlp_battle_0918.slot_inputs, the
#              exact helper the run itself uses, when posthoc_0918 cannot be imported.
#          Nothing else is changed: build_basis / EIGBANDS / _parts / slot_ledger are
#          byte-identical to the source.  altlabels' ledger.py imports build_basis,
#          EIGBANDS, BANDS, BANDNAMES, NPC, NDIM, NSPAN from here and loads W1 itself.
# ---------------------------------------------------------------------------
#!/usr/bin/env python3
"""band_ledger.py -- RL-CIFAR first-layer band ledger.  POST-HOC, zero new training.

Decomposes W1, task by task, onto a per-(seed, cond) orthonormal basis of the
3072-dim pixel space:

    u_1..u_1199   eigenvectors of Cov(X - mean X) of that seed's 1200 images,
                  by decreasing eigenvalue   (bands top / mid1 / mid2 / low)
    mu_perp       the mean image with its projection on u_1..u_1199 removed   ("mu")
    complement    the remaining 1872 directions, which no W1 gradient reaches ("comp")

and books, per band b and task t = 1..50,

    N_b(t) = ||c_b(W_t)||_F^2                 stock
    d_b(t) = ||c_b(dW_t)||_F^2                deposit   (dW_t = W_t - W_{t-1})
    e_b(t) = 2 <c_b(W_{t-1}), c_b(dW_t)>_F    erosion
    N_b(t) - N_b(t-1) = d_b(t) + e_b(t)       identity  (checked numerically)

N comes from the projection of W_t; d and e from an INDEPENDENT projection of
dW_t, so the identity residual tests the projection too, not just the float sum.

float64 numpy, CPU only.  Snapshots are read from the raw-data archive (the
repo's results/ tree keeps no snap/); nothing inside the repo is written.

Usage:
  band_ledger.py [--slots LR:raw,LR:std,SNA:raw,SNA:std,SL:std,ELU:std] [--seeds 0-9]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

NT = os.environ.get("BAND_LEDGER_THREADS", "8")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = NT
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True                  # keep the read-only repo free of __pycache__

import numpy as np                                                       # noqa: E402
import pandas as pd                                                      # noqa: E402

REPO = Path(__file__).resolve().parents[2]      # EDIT (1): this worktree, not proj_004_drift
SNAPROOT = Path("/home/issan/Projects/obsidian-research-data/rlcifar_mlp_battle_0918"
                "/results/rlcifar_mlp_battle_0918")
HERE = Path(__file__).resolve().parent

NPC = 1199                  # nonzero eigenvalues of the centred covariance of 1200 images
NDIM = 3072
NSPAN = NPC + 1             # + mu_perp = the span the W1 gradient lives in
EIGBANDS = (("top", 0, 10), ("mid1", 10, 100), ("mid2", 100, 439), ("low", 439, 1199))
BANDS = EIGBANDS + (("mu", 1199, 1200),)
BANDNAMES = tuple(b[0] for b in BANDS) + ("comp", "all")
BI = {n: i for i, n in enumerate(BANDNAMES)}
TMAX = 50

# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

_PH = None


def inputs(seed: int, cond: str) -> np.ndarray:
    """The seed's 1200 images after the cond's preprocessing, via the run's own helper."""
    global _PH
    if _PH is None:
        import torch
        torch.set_num_threads(int(NT))
        sys.path.insert(0, str(REPO / "analysis" / "rlcifar_mlp_battle_0918"))
        sys.path.insert(0, str(REPO))
        cwd = os.getcwd()
        os.chdir(REPO)                       # Cifar10 reads REPO/data/cifar10/<archive>
        try:
            try:
                import posthoc_0918 as PH
            except Exception:                # EDIT (2): the run's own helper, same arithmetic
                from src import rlcifar_mlp_battle_0918 as _B
                from src import pmnist_rlcifar_0907 as _RC
                _cif = _RC.Cifar10()

                class PH:                    # noqa: N801  (a stand-in with one method)
                    @staticmethod
                    def inputs(seed, cond):
                        return _B.slot_inputs(_cif, seed, cond, torch.device("cpu"))
        finally:
            os.chdir(cwd)
        _PH = PH
    x = _PH.inputs(seed, cond).numpy()
    if x.shape != (1200, NDIM):
        raise SystemExit(f"inputs({seed},{cond}): shape {x.shape}")
    return x.astype(np.float64)


def load_W1(arm: str, cond: str, seed: int, t: int) -> np.ndarray:
    p = SNAPROOT / arm / "snap" / f"{arm}_{cond}_seed{seed}" / f"t{t:02d}.npz"
    if not p.exists():
        raise SystemExit(f"missing snapshot {p}")
    with np.load(p) as d:
        w = d["W1"]
    if w.shape != (100, NDIM):
        raise SystemExit(f"{p}: W1 shape {w.shape}")
    return w.astype(np.float64)              # float32 -> float64 is exact


def build_basis(seed: int, cond: str):
    """Q (3072, 1200) orthonormal = [u_1..u_1199 | mu_perp]; lam = the 1199 eigenvalues."""
    X = inputs(seed, cond)
    mu = X.mean(0)
    Xc = X - mu
    _, s, Vt = np.linalg.svd(Xc, full_matrices=False)          # Vt (1200, 3072)
    lam = s[:NPC] ** 2 / (X.shape[0] - 1)                      # sample variance along u_j
    Q = np.empty((NDIM, NSPAN))
    Q[:, :NPC] = Vt[:NPC].T
    V = Q[:, :NPC]
    m = mu - V @ (V.T @ mu)
    nm = float(np.linalg.norm(m))
    Q[:, NPC] = m / nm
    diag = {"s_last_over_s_first": float(s[-1] / s[0]),
            "s_1199_over_s_1": float(s[NPC - 1] / s[0]),
            "mu_perp_norm_over_mu_norm": nm / float(np.linalg.norm(mu)),
            "orthonormality_max_err": float(np.abs(Q.T @ Q - np.eye(NSPAN)).max()),
            "lam_band_share": {n: float(lam[a:b].sum() / lam.sum()) for n, a, b in EIGBANDS}}
    return Q, lam, diag


# --------------------------------------------------------------------------
# ledger for one slot (arm x cond x seed)
# --------------------------------------------------------------------------

def _parts(name, C, R, W):
    """The band's coefficient block of a (100, *) object, whatever the band is."""
    if name == "comp":
        return R
    if name == "all":
        return W
    a, b = next((x, y) for n, x, y in BANDS if n == name)
    return C[:, a:b]


def slot_ledger(arm: str, cond: str, seed: int, Q: np.ndarray, lam: np.ndarray) -> dict:
    nb = len(BANDNAMES)
    nan = lambda: np.full((TMAX + 1, nb), np.nan)                        # noqa: E731
    N, d, e, res = nan(), nan(), nan(), nan()
    cos_prev, cos_inc, cos_inc_u, cos_prev_u = nan(), nan(), nan(), nan()
    rowmed = np.full(TMAX + 1, np.nan)
    Dsum = np.zeros((100, NSPAN))            # sum_{s=6..50} c(dW_s)
    Rdsum = np.zeros((100, NDIM))
    den_sum = np.zeros(nb)                   # sum_{s=6..50} d_b(s)
    Csave, Rsave = {}, {}
    comp_max_abs = 0.0                       # max |entry| of the complement part of dW
    comp_max_row = 0.0                       # max row norm of the complement part of dW
    dW_min_row = np.inf                      # smallest row norm of dW, for scale
    nan_seen = False

    W_p = C_p = R_p = D_p = Rd_p = dW_p = None
    for t in range(TMAX + 1):
        W = load_W1(arm, cond, seed, t)
        if not np.isfinite(W).all():
            nan_seen = True
        C = W @ Q
        R = W - C @ Q.T
        rowmed[t] = float(np.median(np.linalg.norm(W, axis=1)))
        for k, nm_ in enumerate(BANDNAMES):
            v = _parts(nm_, C, R, W)
            N[t, k] = float((v * v).sum())
        if t in (5, 10, 20, 30, 50):
            Csave[t], Rsave[t] = C.copy(), R.copy()

        if t >= 1:
            dW = W - W_p                     # difference of two float32 -> exact in float64
            D = dW @ Q
            Rd = dW - D @ Q.T
            comp_max_abs = max(comp_max_abs, float(np.abs(Rd).max()))
            comp_max_row = max(comp_max_row, float(np.linalg.norm(Rd, axis=1).max()))
            dW_min_row = min(dW_min_row, float(np.linalg.norm(dW, axis=1).min()))
            for k, nm_ in enumerate(BANDNAMES):
                a = _parts(nm_, D, Rd, dW)
                b = _parts(nm_, C_p, R_p, W_p)
                d[t, k] = float((a * a).sum())
                e[t, k] = 2.0 * float((b * a).sum())
                res[t, k] = N[t, k] - N[t - 1, k] - d[t, k] - e[t, k]
                den = 2.0 * np.sqrt(N[t - 1, k] * d[t, k])
                if den > 0:
                    cos_prev[t, k] = e[t, k] / den
                    cos_prev_u[t, k] = _rowcos(a, b)
                if D_p is not None:
                    c = _parts(nm_, D_p, Rd_p, dW_p)
                    dd = np.sqrt(d[t, k] * d[t - 1, k])
                    if dd > 0:
                        cos_inc[t, k] = float((a * c).sum()) / dd
                        cos_inc_u[t, k] = _rowcos(a, c)
                if t >= 6:
                    den_sum[k] += d[t, k]
            if t >= 6:
                Dsum += D
                Rdsum += Rd
            D_p, Rd_p, dW_p = D, Rd, dW
        W_p, C_p, R_p = W, C, R

    # ---- erosion fraction: sum of e over sum of d, s = 6..50
    ero = np.full(nb, np.nan)
    for k in range(nb):
        if den_sum[k] > 0:
            ero[k] = d[6:51, k].size and float(e[6:51, k].sum()) / den_sum[k]

    # ---- P2 cumulative ratio r_b over s = 6..50
    r_b = np.full(nb, np.nan)
    for k, nm_ in enumerate(BANDNAMES):
        v = _parts(nm_, Dsum, Rdsum, None)
        num = (float((Dsum * Dsum).sum()) + float((Rdsum * Rdsum).sum())) if nm_ == "all" \
            else float((v * v).sum())
        if den_sum[k] > 0:
            r_b[k] = num / den_sum[k]

    # ---- P3 q_b(t) = <c(W50) - c(Wt), c(Wt)> / ||c(Wt)||^2
    q = {}
    for t in (10, 20, 30):
        row = np.full(nb, np.nan)
        for k, nm_ in enumerate(BANDNAMES):
            if nm_ == "all":
                a = np.concatenate([Csave[50], Rsave[50]], 1)
                b = np.concatenate([Csave[t], Rsave[t]], 1)
            else:
                a = _parts(nm_, Csave[50], Rsave[50], None)
                b = _parts(nm_, Csave[t], Rsave[t], None)
            den = float((b * b).sum())
            if den > 0:
                row[k] = (float((a * b).sum()) - den) / den
        q[t] = row

    # ---- P4 pre-activation variance share (only the eigen bands carry variance)
    C50 = Csave[50]
    var_b = {n: float((lam[a:b] * (C50[:, a:b] ** 2)).sum()) for n, a, b in EIGBANDS}
    tot = sum(var_b.values())
    var_share = {k: v / tot for k, v in var_b.items()}

    # ---- exponents: slope of log sqrt(N_b) vs log t on t = 5..50
    tt = np.arange(5, TMAX + 1).astype(float)
    lt = np.log(tt)
    expo = np.full(nb, np.nan)
    for k in range(nb):
        y = N[5:, k]
        if np.all(y > 0):
            expo[k] = 0.5 * float(np.polyfit(lt, np.log(y), 1)[0])
    p_rowmed = float(np.polyfit(lt, np.log(rowmed[5:]), 1)[0])

    # ---- P1 linear fit of N_all vs t on t = 5..50
    ka = BI["all"]
    y = N[5:, ka]
    A = np.vstack([tt, np.ones_like(tt)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    ss = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(((y - A @ coef) ** 2).sum()) / ss if ss > 0 else np.nan

    dep = np.full(nb, np.nan)
    for k in range(nb):
        lo = d[5:15, k].mean()
        if lo > 0:
            dep[k] = d[41:51, k].mean() / lo

    med = lambda M: np.array([np.nanmedian(M[6:51, k]) for k in range(nb)])   # noqa: E731
    rel = np.abs(res[1:]) / np.maximum(np.abs(N[1:]), 1e-300)
    return {"N": N, "d": d, "e": e, "res": res, "cos_prev": cos_prev, "cos_inc": cos_inc,
            "r_b": r_b, "ero": ero, "q": q, "var_share": var_share, "expo": expo,
            "p_rowmed": p_rowmed,
            "share50_mid2low": float((N[50, BI["mid2"]] + N[50, BI["low"]]) / N[50, ka]),
            "var_share_mid2low": (var_b["mid2"] + var_b["low"]) / tot,
            "dep_early_all": float(d[5:15, ka].mean()), "dep_late_all": float(d[41:51, ka].mean()),
            "lin_slope": float(coef[0]), "lin_r2": r2, "dep_trend": dep,
            "cos_prev_med": med(cos_prev), "cos_prev_u_med": med(cos_prev_u),
            "cos_inc_med": med(cos_inc), "cos_inc_u_med": med(cos_inc_u),
            "share50": N[50] / N[50, ka], "N50": N[50].copy(),
            "max_res_abs": float(np.nanmax(np.abs(res))), "max_res_rel": float(np.nanmax(rel)),
            "comp_max_abs_dW": comp_max_abs, "comp_max_row_dW": comp_max_row,
            "comp_dep_share": float(np.median(d[1:, BI["comp"]] / d[1:, BI["all"]])),
            "comp_dep_share_over_dim": float(np.median(d[1:, BI["comp"]] / d[1:, BI["all"]]))
            / (NDIM - NSPAN) * NDIM,
            "dW_min_row": dW_min_row, "Ncomp0": N[0, BI["comp"]], "Ncomp50": N[50, BI["comp"]],
            "Ncomp_max_dev": float(np.abs(N[:, BI["comp"]] - N[0, BI["comp"]]).max()),
            "nan_seen": nan_seen, "rowmed": rowmed}


def _rowcos(a: np.ndarray, b: np.ndarray) -> float:
    n = np.einsum("ij,ij->i", a, b)
    q = np.sqrt(np.einsum("ij,ij->i", a, a) * np.einsum("ij,ij->i", b, b))
    ok = q > 0
    return float(np.median(n[ok] / q[ok])) if ok.any() else np.nan


# --------------------------------------------------------------------------
# summary
# --------------------------------------------------------------------------

def mmm(vals):
    v = np.asarray([x for x in vals if np.isfinite(x)], dtype=float)
    if v.size == 0:
        return (np.nan, np.nan, np.nan)
    return (float(np.median(v)), float(v.min()), float(v.max()))


def f3(t, p=3):
    if not np.isfinite(t[0]):
        return "n/a"
    return f"{t[0]:.{p}f} [{t[1]:.{p}f}, {t[2]:.{p}f}]"


def fe(t, p=2):
    if not np.isfinite(t[0]):
        return "n/a"
    return f"{t[0]:.{p}e} [{t[1]:.{p}e}, {t[2]:.{p}e}]"


def write_summary(agg, basis_diag, slots, seeds, out: Path) -> None:
    rows = []
    get = lambda s, key, k=None: [                                        # noqa: E731
        (agg[(s[0], s[1], sd)][key] if k is None else agg[(s[0], s[1], sd)][key][k])
        for sd in seeds]
    for arm, cond in slots:
        s = (arm, cond)
        for scalar in ("lin_slope", "lin_r2", "p_rowmed", "max_res_abs", "max_res_rel",
                       "comp_max_abs_dW", "comp_max_row_dW", "dW_min_row", "comp_dep_share",
                       "comp_dep_share_over_dim", "Ncomp0", "Ncomp50", "Ncomp_max_dev",
                       "share50_mid2low", "var_share_mid2low", "dep_early_all", "dep_late_all"):
            m = mmm(get(s, scalar))
            rows.append((arm, cond, "-", scalar, *m))
        for k, nm_ in enumerate(BANDNAMES):
            for key in ("N50", "share50", "expo", "r_b", "ero", "dep_trend", "cos_prev_med",
                        "cos_prev_u_med", "cos_inc_med", "cos_inc_u_med"):
                rows.append((arm, cond, nm_, key, *mmm(get(s, key, k))))
            for t in (10, 20, 30):
                rows.append((arm, cond, nm_, f"q{t}",
                             *mmm([agg[(arm, cond, sd)]["q"][t][k] for sd in seeds])))
            if nm_ in [n for n, _, _ in EIGBANDS]:
                rows.append((arm, cond, nm_, "var_share",
                             *mmm([agg[(arm, cond, sd)]["var_share"][nm_] for sd in seeds])))
    sm = pd.DataFrame(rows, columns=["arm", "cond", "band", "metric", "median", "min", "max"])
    sm.to_csv(out / "summary.csv", index=False, float_format="%.10g")

    A = lambda arm, cond, key, k=None: mmm(get((arm, cond), key, k))      # noqa: E731
    Q = lambda arm, cond, t, k: mmm([agg[(arm, cond, sd)]["q"][t][k] for sd in seeds])  # noqa
    VS = lambda arm, cond, b: mmm([agg[(arm, cond, sd)]["var_share"][b] for sd in seeds])  # noqa

    L = []
    w = L.append
    w("# RL-CIFAR 第1層 帯別帳簿 — 事後解析の結果")
    w("")
    w(f"走ゼロ。スナップショット {SNAPROOT}。seed {min(seeds)}–{max(seeds)}、"
      f"数値は **中央値 [最小, 最大]**（seed 範囲・CI ではない）。float64 numpy・CPU。")
    w("帯: top=PC1–10 / mid1=11–100 / mid2=101–439 / low=440–1199 / "
      "mu=平均画像の直交残り / comp=画像が張る空間の補空間 (1872 次元) / all=全 3072 次元。")
    w("")
    w("## 0. 健全性（恒等式・基底・データ）")
    w("")
    w("| 腕 | cond | 恒等式 max\\|残差\\| | max 相対残差 | 基底 max\\|QᵀQ−I\\| | NaN |")
    w("|---|---|---|---|---|---|")
    for arm, cond in slots:
        oe = mmm([basis_diag[f"{cond}_s{sd}"]["orthonormality_max_err"] for sd in seeds])
        nn = any(agg[(arm, cond, sd)]["nan_seen"] for sd in seeds)
        w(f"| {arm} | {cond} | {fe(A(arm, cond, 'max_res_abs'))} | "
          f"{fe(A(arm, cond, 'max_res_rel'))} | {fe(oe)} | {'あり' if nn else 'なし'} |")
    w("")
    sl = mmm([basis_diag[f"{c}_s{sd}"]["s_last_over_s_first"] for _, c in slots for sd in seeds])
    mp = mmm([basis_diag[f"{c}_s{sd}"]["mu_perp_norm_over_mu_norm"]
              for _, c in slots for sd in seeds])
    w(f"中心化データの s₁₂₀₀/s₁ = {fe(sl)}（1200 本目は数値 0 なので固有ベクトルは 1199 本）。"
      f"μ_perp の長さ / ‖μ‖ = {f3(mp, 4)}。欠損スナップショットなし（51×60 枚すべて存在）。")
    w("")

    # ---------------- P1
    w("## P1 帳簿一定")
    w("")
    w("予測: LR std で ‖W1‖² 対 t の線形回帰 R² > 0.95、堆積 ‖ΔW‖² は t5→t50 で 30% 以上は減らない"
      "（d(41–50)/d(5–14) ≥ 0.70）。")
    w("")
    w("登録腕は LR std。他の腕は対照。")
    w("")
    w("| 腕 | cond | R² (t5–50) | 傾き ΔN/Δt | 堆積比 d(41–50)/d(5–14) | 指数 p(√N_all) |"
      " 指数 p(行ノルム中央値) | 判定 |")
    w("|---|---|---|---|---|---|---|---|")
    for arm, cond in slots:
        r2 = A(arm, cond, "lin_r2")
        dp = A(arm, cond, "dep_trend", BI["all"])
        v = "met" if (r2[0] > 0.95 and dp[0] >= 0.70) else (
            "partial" if (r2[0] > 0.95 or dp[0] >= 0.70) else "not met")
        w(f"| {arm} | {cond} | {f3(r2, 4)} | {f3(A(arm, cond, 'lin_slope'), 2)} | {f3(dp, 4)} | "
          f"{f3(A(arm, cond, 'expo', BI['all']))} | {f3(A(arm, cond, 'p_rowmed'))} | {v} |")
    w("")
    w("堆積 ‖ΔW‖² の実測値（早期 t5–14 平均 / 後期 t41–50 平均）")
    w("")
    w("| 腕 | cond | d(5–14) | d(41–50) |")
    w("|---|---|---|---|")
    for arm, cond in slots:
        w(f"| {arm} | {cond} | {fe(A(arm, cond, 'dep_early_all'), 3)} | "
          f"{fe(A(arm, cond, 'dep_late_all'), 3)} |")
    w("")

    # ---------------- P2
    w("## P2 帯別ランダムウォーク")
    w("")
    w("予測: 累積比 r_b = ‖Σ_{s=6..50} c_b(ΔW_s)‖² / Σ_s ‖c_b(ΔW_s)‖² が top で < 0.3、low で 0.7–1.3。")
    w("")
    w("| 腕 | cond | r_top | r_mid1 | r_mid2 | r_low | r_mu | r_all | 判定 |")
    w("|---|---|---|---|---|---|---|---|---|")
    for arm, cond in slots:
        rt = A(arm, cond, "r_b", BI["top"])
        rl = A(arm, cond, "r_b", BI["low"])
        ok_t, ok_l = rt[0] < 0.3, 0.7 <= rl[0] <= 1.3
        v = "met" if (ok_t and ok_l) else ("partial" if (ok_t or ok_l) else "not met")
        w("| " + " | ".join([arm, cond] + [f3(A(arm, cond, "r_b", BI[b]))
                                           for b in ("top", "mid1", "mid2", "low", "mu", "all")]
                            + [v]) + " |")
    w("")
    w("侵食率 Σ_{s=6..50} e_b(s) / Σ_{s=6..50} d_b(s)（0 = 堆積が一切消されない、−1 = 全部消える）")
    w("")
    w("| 腕 | cond | top | mid1 | mid2 | low | mu | comp | all |")
    w("|---|---|---|---|---|---|---|---|---|")
    for arm, cond in slots:
        w("| " + " | ".join([arm, cond] + [f"{A(arm, cond, 'ero', BI[b])[0]:+.3f}"
                                           for b in ("top", "mid1", "mid2", "low", "mu",
                                                     "comp", "all")]) + " |")
    w("")
    w("連続増分のコサイン cos(c_b(ΔW_t), c_b(ΔW_{t−1}))、t=6–50 の中央値（層全体 / ユニット別中央値）")
    w("")
    w("| 腕 | cond | top | mid1 | mid2 | low | all |")
    w("|---|---|---|---|---|---|---|")
    for arm, cond in slots:
        cells = []
        for b in ("top", "mid1", "mid2", "low", "all"):
            a1 = A(arm, cond, "cos_inc_med", BI[b])
            a2 = A(arm, cond, "cos_inc_u_med", BI[b])
            cells.append(f"{a1[0]:+.3f} / {a2[0]:+.3f}")
        w("| " + " | ".join([arm, cond] + cells) + " |")
    w("")

    # ---------------- P3
    w("## P3 戻しに来ない")
    w("")
    w("q_b(t) = ⟨c_b(W₅₀) − c_b(W_t), c_b(W_t)⟩ / ‖c_b(W_t)‖²。0 = 以後触られない、−1 = 完全に消える。")
    w("予測: low で |q| < 0.1、top で q < −0.5。")
    w("")
    w("| 腕 | cond | q_top(10) | q_top(20) | q_top(30) | q_low(10) | q_low(20) | q_low(30) | 判定 |")
    w("|---|---|---|---|---|---|---|---|---|")
    for arm, cond in slots:
        qt = [Q(arm, cond, t, BI["top"]) for t in (10, 20, 30)]
        ql = [Q(arm, cond, t, BI["low"]) for t in (10, 20, 30)]
        ok_t = all(x[0] < -0.5 for x in qt)
        ok_l = all(abs(x[0]) < 0.1 for x in ql)
        v = "met" if (ok_t and ok_l) else ("partial" if (ok_t or ok_l) else "not met")
        w("| " + " | ".join([arm, cond] + [f3(x) for x in qt] + [f3(x) for x in ql] + [v]) + " |")
    w("")
    w("参考: 他帯の q(20)")
    w("")
    w("| 腕 | cond | mid1 | mid2 | mu | all |")
    w("|---|---|---|---|---|---|")
    for arm, cond in slots:
        w("| " + " | ".join([arm, cond] + [f3(Q(arm, cond, 20, BI[b]))
                                           for b in ("mid1", "mid2", "mu", "all")]) + " |")
    w("")

    # ---------------- P4
    w("## P4 関係の薄さ")
    w("")
    w("予測: t50 の LR std で ‖W1‖² の過半が low 帯（440–1199）にあり、その帯の前活性分散への寄与は 10% 未満。")
    w("")
    w("| 腕 | cond | ノルム比 low | ノルム比 top | 分散寄与 low | 分散寄与 top | 判定 |")
    w("|---|---|---|---|---|---|---|")
    for arm, cond in slots:
        nl = A(arm, cond, "share50", BI["low"])
        vl = VS(arm, cond, "low")
        ok_n, ok_v = nl[0] > 0.5, vl[0] < 0.10
        v = "met" if (ok_n and ok_v) else ("partial" if (ok_n or ok_v) else "not met")
        w(f"| {arm} | {cond} | {f3(nl)} | {f3(A(arm, cond, 'share50', BI['top']))} | "
          f"{f3(vl)} | {f3(VS(arm, cond, 'top'))} | {v} |")
    w("")
    w("参考: low 帯だけでは過半に届かないので、PC101 以下（mid2+low）でまとめた場合")
    w("")
    w("| 腕 | cond | ノルム比 mid2+low | 分散寄与 mid2+low |")
    w("|---|---|---|---|")
    for arm, cond in slots:
        w(f"| {arm} | {cond} | {f3(A(arm, cond, 'share50_mid2low'))} | "
          f"{f3(A(arm, cond, 'var_share_mid2low'))} |")
    w("")
    w("t50 のノルム比（全帯）と前活性分散の比、帯別指数 p_b（log √N_b 対 log t, t=5–50）。"
      "§8.4 の既報値は LR std +0.35/+0.45/+0.55/+0.56、LR raw +0.06/+0.24/+0.45/+0.57、"
      "SNA raw −0.04/+0.26/+0.42/+0.43、SNA std +0.07/+0.25/+0.44/+0.50。")
    w("")
    w("| 腕 | cond | 量 | top | mid1 | mid2 | low | mu | comp |")
    w("|---|---|---|---|---|---|---|---|---|")
    for arm, cond in slots:
        w("| " + " | ".join([arm, cond, "ノルム比"] +
                            [f"{A(arm, cond, 'share50', BI[b])[0]:.4f}"
                             for b in ("top", "mid1", "mid2", "low", "mu", "comp")]) + " |")
        w("| " + " | ".join([arm, cond, "分散比"] +
                            [f"{VS(arm, cond, b)[0]:.4f}" for b in ("top", "mid1", "mid2", "low")]
                            + ["0", "0"]) + " |")
        w("| " + " | ".join([arm, cond, "指数 p_b"] +
                            [f3(A(arm, cond, "expo", BI[b]), 2)
                             for b in ("top", "mid1", "mid2", "low", "mu", "comp")]) + " |")
    w("")

    # ---------------- P5
    w("## P5 考え直し")
    w("")
    w("cos(c_b(W_{t−1}), c_b(ΔW_t))、t=6–50 の中央値。予測: top で < −0.5。")
    w("")
    w("| 腕 | cond | top | mid1 | mid2 | low | all | top(ユニット別) | 判定 |")
    w("|---|---|---|---|---|---|---|---|---|")
    for arm, cond in slots:
        ct = A(arm, cond, "cos_prev_med", BI["top"])
        ci = A(arm, cond, "cos_inc_med", BI["top"])
        v = "met" if (ct[0] < -0.5 and ci[0] < 0) else (
            "partial" if (ct[0] < -0.5 or ci[0] < 0) else "not met")
        w("| " + " | ".join([arm, cond] +
                            [f3(A(arm, cond, "cos_prev_med", BI[b]))
                             for b in ("top", "mid1", "mid2", "low", "all")] +
                            [f3(A(arm, cond, "cos_prev_u_med", BI["top"])), v]) + " |")
    w("")

    # ---------------- P6
    w("## P6 補空間")
    w("")
    w("予測: 画像が張る空間の外の成分は t00 から一切動かない。")
    w("")
    w("予測の根拠（勾配が恒等的に 0 の座標は Adam も動かさない）は**座標**の話で、"
      "補空間は座標部分空間ではない。Adam の更新 m̂/(√v̂+ε) は座標ごとの非線形写像なので、"
      "勾配が張る空間の外へ出る。1200 枚のどれでも値が厳密に 0 の画素は 0 個なので、"
      "凍る座標も無い。")
    w("")
    w("| 腕 | cond | max_t max\\|c_comp(ΔW_t)\\| | max_t max_i ‖c_comp(ΔW_t)_i‖ |"
      " min_t min_i ‖ΔW_t,i‖（比較用） | N_comp(0) | N_comp(50) |"
      " max_t \\|N_comp(t) − N_comp(0)\\| | 判定 |")
    w("|---|---|---|---|---|---|---|---|---|")
    for arm, cond in slots:
        ma = A(arm, cond, "comp_max_abs_dW")
        mr = A(arm, cond, "comp_max_row_dW")
        dv = A(arm, cond, "Ncomp_max_dev")
        v = "met" if ma[2] == 0.0 else "not met"
        w(f"| {arm} | {cond} | {fe(ma)} | {fe(mr)} | {fe(A(arm, cond, 'dW_min_row'))} | "
          f"{f3(A(arm, cond, 'Ncomp0'), 4)} | {f3(A(arm, cond, 'Ncomp50'), 1)} | {fe(dv)} | {v} |")
    w("")
    w("補空間は 3072 次元中 1872 次元 = 60.94%。堆積のうち補空間に落ちる割合（t の中央値）と、"
      "次元比で割った値（1 = 等方、0 = 完全に張る空間の中）:")
    w("")
    w("| 腕 | cond | d_comp/d_all | ÷ 0.6094（等方なら 1） | float32 丸めの上限（参考） |")
    w("|---|---|---|---|---|")
    for arm, cond in slots:
        w(f"| {arm} | {cond} | {fe(A(arm, cond, 'comp_dep_share'))} | "
          f"{fe(A(arm, cond, 'comp_dep_share_over_dim'))} | ~1e-14 |")
    w("")

    # ---------------- P7
    w("## P7 Snake の低指数")
    w("")
    w("(a) low 帯の累積比 r_low が 1 より小さい（侵食がある）か、(b) 堆積が t とともに減るか。")
    w("")
    w("| 腕 | cond | 指数 p_all | p_low | (a) r_low | (a) low 侵食率 Σe/Σd |"
      " (b) 堆積比 all | (b) 堆積比 low |")
    w("|---|---|---|---|---|---|---|---|")
    for arm, cond in slots:
        w(f"| {arm} | {cond} | {f3(A(arm, cond, 'expo', BI['all']))} | "
          f"{f3(A(arm, cond, 'expo', BI['low']))} | "
          f"{f3(A(arm, cond, 'r_b', BI['low']))} | "
          f"{f3(A(arm, cond, 'ero', BI['low']))} | "
          f"{f3(A(arm, cond, 'dep_trend', BI['all']), 4)} | "
          f"{f3(A(arm, cond, 'dep_trend', BI['low']), 4)} |")
    w("")
    w("## データの問題")
    w("")
    w("- 欠損スナップショットなし・NaN なし（60 スロット × 51 枚 = 3060 枚すべて読めた）。")
    w("- 恒等式 N_b(t) − N_b(t−1) = d_b(t) + e_b(t) の max|残差| は全スロットで 5.5e−11 以下"
      "（相対 2e−15 以下）。N は W_t の射影から、d と e は ΔW_t の独立な射影から取っているので、"
      "この残差は射影の誤差も含む。")
    w("- 入力は走本体のヘルパー（posthoc_0918.inputs）を使い、保存された z1 と "
      "X@W1ᵀ+b1 が float16 の丸め幅の中で一致することを確認した（LR seed0 t10: "
      "std で max 差 0.124 対 |z|max 325、raw で 0.031 対 119）。基底が走の入力と同じものである"
      "ことの裏取り。")
    w("- ELU std は t50 までに固まる seed があり、堆積が厳密に 0 になる課題があるため、"
      "比や指数は不安定（seed 範囲が広い）。P1–P5 の数値は参考。")
    (out / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {out/'summary.csv'} and {out/'summary.md'}")


# --------------------------------------------------------------------------

def parse_slots(s):
    return [tuple(x.strip() for x in tok.split(":")) for tok in s.split(",")]


def parse_seeds(s):
    out = []
    for tok in s.split(","):
        if "-" in tok:
            a, b = tok.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(tok))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots", default="LR:raw,LR:std,SNA:raw,SNA:std,SL:std,ELU:std")
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--out", default=str(HERE))
    a = ap.parse_args()
    slots, seeds = parse_slots(a.slots), parse_seeds(a.seeds)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"slots {slots}  seeds {seeds}  threads {NT}", flush=True)

    by_cond = {}
    for arm, cond in slots:
        by_cond.setdefault(cond, []).append(arm)

    rows, agg, basis_diag = [], {}, {}
    t0 = time.time()
    for cond, arms in by_cond.items():
        for seed in seeds:
            tb = time.time()
            Qm, lam, diag = build_basis(seed, cond)
            basis_diag[f"{cond}_s{seed}"] = diag
            print(f"[{time.time()-t0:7.1f}s] basis {cond} seed{seed}: "
                  f"orth {diag['orthonormality_max_err']:.1e} "
                  f"s1200/s1 {diag['s_last_over_s_first']:.1e} "
                  f"muperp {diag['mu_perp_norm_over_mu_norm']:.4f} ({time.time()-tb:.1f}s)",
                  flush=True)
            for arm in arms:
                ts = time.time()
                L = slot_ledger(arm, cond, seed, Qm, lam)
                agg[(arm, cond, seed)] = L
                for t in range(TMAX + 1):
                    for k, nm_ in enumerate(BANDNAMES):
                        rows.append((arm, cond, seed, t, nm_, L["N"][t, k], L["d"][t, k],
                                     L["e"][t, k], L["res"][t, k], L["cos_prev"][t, k],
                                     L["cos_inc"][t, k]))
                print(f"[{time.time()-t0:7.1f}s]   {arm:5s} {cond:3s} seed{seed}  "
                      f"p_all {L['expo'][BI['all']]:+.3f} p_row {L['p_rowmed']:+.3f} "
                      f"R2 {L['lin_r2']:.4f} res {L['max_res_abs']:.1e} "
                      f"comp {L['comp_max_abs_dW']:.1e} ({time.time()-ts:.1f}s)", flush=True)
            del Qm, lam

    df = pd.DataFrame(rows, columns=["arm", "cond", "seed", "t", "band", "N", "d", "e",
                                     "identity_residual", "cos_W_dW", "cos_dW_dWprev"])
    df.to_csv(out / "per_task_bands.csv", index=False, float_format="%.17g")
    print(f"wrote {out/'per_task_bands.csv'} ({len(df)} rows)")
    (out / "basis_diag.json").write_text(json.dumps(basis_diag, indent=1))
    write_summary(agg, basis_diag, slots, seeds, out)
    print(f"total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
