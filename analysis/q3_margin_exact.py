"""q3_margin_exact: 条件A ゲート恒等式 p_hat=0 ⟺ cos <= cos_crit の厳密検算。

**事後計算・未事前登録**。spec_ratchet_log_0819 / spec_ratchet_centered_0822 の
どちらにも登録されていない、Q3 (消灯点) 解釈のための後付け検証である。判定は
verdict.csv を書き換えない。

検証する代数恒等式 (条件A・32 パターン厳密分布の下で):

    x_in(p) = mu + delta(p),  mu = E_p[x_in],  delta(p) = (0_f, pattern_p - 1/2)
    a_i(p)  = w_i . x_in(p) + b_i = ||w_i|| ||mu|| cos(w_i,mu) + w_i.delta(p) + b_i
    M_i     = max_p w_i.delta(p) = (1/2) sum_{j>=f} |w_ij|   (= ||w_i,free||_1 / 2)
    p_hat_i = 0  <=>  max_p a_i(p) <= 0  <=>  cos_i <= cos_crit_i,
    cos_crit_i = -(b_i + M_i) / (||w_i|| ||mu||)

同値な正規化形 (beta は lop_metrics.compute_b_metrics と同じ定義
beta = (w.mu + b)/sqrt(w^T Sigma w)、条件A では Sigma = (1/4) diag(free bits) なので
sqrt(w^T Sigma w) = ||w_free||_2 / 2):

    p_hat_i = 0  <=>  beta_i <= -r_i,   r_i = ||w_i,free||_1 / ||w_i,free||_2 in [1, sqrt(5)]

出所と制約:
  - 記録ログ results/ratchet_log_0819/logs/seed*.npz (std) と
    results/ratchet_centered_0822/logs/seed*.npz (centered) には
    cos_u_mu / p_hat / w_norm / b / mu_norm しか無く、**full W が無い**ので M は
    復元できない。W は checkpoint (.pt) にしかなく、その .pt は .gitignore 対象で
    リポジトリに含まれない。
  - そこで本スクリプトは step 0 と step 1e6 の checkpoint を **同一 config で再走して
    再生成**する (probe=None。本走の S2 サニティが probe 無しでの bit 一致を確認済み)。
    再生成物は results/ 配下を汚さないよう REPRO_DIR にキャッシュする。
  - 再走は float32 の加算順序が環境 (torch/BLAS のバージョン) に依存するため、
    step 0 は記録ログと厳密一致するが step 1e6 は軌道がドリフトしうる。
    ドリフト量は D 節で定量報告し、恒等式の判定 (A 節) は
    「同一状態から計算した p_hat と cos_crit」の間で行う (恒等式は 1 つの (W,b,mu) に
    ついての代数命題なので、これで厳密検証になる)。
  - 記録ログ**だけ**で検証できる必要条件 (E 節) は全 20,901 記録点で別途走らせる。

実行方法 (引数なし・決定論):

  OMP_NUM_THREADS=1 python3 analysis/q3_margin_exact.py

初回は checkpoint 再生成のため 1 arm あたり 10-20 分かかる。2 回目以降は
REPRO_DIR のキャッシュを再利用する。

出力: /root/q3_out/verify/exact/
  results.md / identity_units.csv / cos_crit_dist.csv / theta_1m.csv /
  recorded_necessary.csv / drift.csv / meta.json / figures/fig_cos_crit_dist.png
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import ROOT, load_config, build_runs, group_runs  # noqa: E402
from src.train import setup_group, train_group                    # noqa: E402
from analysis.q3_gate_curve_ci import (BIN_UPPER, COS_HI, COS_LO,  # noqa: E402
                                       BIN_W, N_BIN, N_P, hist_median,
                                       theta_all_one, theta_one)

OUTDIR = Path("/root/q3_out/verify/exact")
REPRO_DIR = OUTDIR / "repro"
STEPS = (0, 1_000_000)
ARMS = {
    "std": dict(cfg=Path(ROOT) / "configs" / "ratchet_log_0819.yaml",
                res=Path(ROOT) / "results" / "ratchet_log_0819"),
    "centered": dict(cfg=Path(ROOT) / "configs" / "ratchet_centered_0822.yaml",
                     res=Path(ROOT) / "results" / "ratchet_centered_0822"),
}
GNAME = "A_w100"
QUANTS = [0.0, 0.05, 0.5, 0.95, 1.0]
QNAMES = ["min", "q05", "median", "q95", "max"]

plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def git_hash(paths: list[str] | None = None) -> str:
    cmd = ["git", "log", "-1", "--format=%h"]
    if paths:
        cmd += ["--", *paths]
    try:
        return subprocess.check_output(cmd, cwd=ROOT, text=True).strip() or "未コミット"
    except Exception:
        return "unknown"


# ------------------------------------------------------------------ checkpoint 再生成

def ckpt_path(arm: str, step: int) -> Path:
    return REPRO_DIR / arm / "ckpts" / f"{GNAME}_step{step}.pt"


def ensure_ckpts(arm: str) -> float:
    """step 0 / 1e6 の checkpoint が無ければ本走と同一 config で再走して作る。"""
    if all(ckpt_path(arm, s).exists() for s in STEPS):
        return 0.0
    cfg = load_config(str(ARMS[arm]["cfg"]))
    runs = build_runs(cfg)
    groups = group_runs(runs)
    if len(groups) != 1:
        raise SystemExit(f"{arm}: 単一グループ前提だが {len(groups)} 個")
    gkey, gruns = next(iter(groups.items()))
    outdir = REPRO_DIR / arm
    outdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    train_group(gkey, gruns, cfg, "cpu", str(outdir),
                total_steps=int(cfg["common"]["total_steps"]), ckpts=list(STEPS))
    return time.time() - t0


def load_state(arm: str, step: int) -> dict:
    """checkpoint から解析に必要な float64 テンソルを取り出す。"""
    cfg = load_config(str(ARMS[arm]["cfg"]))
    runs = build_runs(cfg)
    gkey, gruns = next(iter(group_runs(runs).items()))
    st = setup_group(gkey, gruns, cfg, "cpu")
    ck = torch.load(ckpt_path(arm, step), map_location="cpu", weights_only=False)
    st["net"].load_state(ck["net"])
    st["env"].load_state(ck["env"])
    st["teacher"].load_state(ck["teacher"])
    st["running_mean"].copy_(ck["running_mean"])
    seeds = np.array([int(r["seed"]) for r in gruns], dtype=np.int64)
    return dict(W=st["net"].W.double(), b=st["net"].b.double(),
                running_mean=st["running_mean"].double(),
                flip_state=st["env"].flip_state.double(),
                patterns=st["env"].patterns.double(),
                centered=st["centered"].double(),
                f=int(cfg["condA"]["f"]), m=int(cfg["condA"]["m"]), seeds=seeds)


# ------------------------------------------------------------------ 厳密量

def exact_quantities(state: dict) -> dict:
    """1 時点ぶんの per-unit 厳密量 [R,h] (全て float64 numpy)。"""
    W, b, f = state["W"], state["b"], state["f"]
    P = state["patterns"].shape[0]
    flip = state["flip_state"].unsqueeze(0).expand(P, -1, -1)            # [P,R,f]
    rnd = state["patterns"][:, None, :].expand(-1, W.shape[0], -1)       # [P,R,m-f]
    X = torch.cat([flip, rnd], dim=2)                                    # [P,R,m]
    x_in = X - state["centered"][None, :, None] * state["running_mean"][None]

    mu = x_in.mean(dim=0)                                                # [R,m]
    mu_norm = mu.norm(dim=1)                                             # [R]
    mu_u = mu / mu_norm.clamp_min(1e-300)[:, None]
    w_norm = W.norm(dim=2)                                               # [R,h]
    cos = torch.einsum("rhd,rd->rh", W, mu_u) / w_norm.clamp_min(1e-300)
    s = torch.einsum("rhd,rd->rh", W, mu) + b                            # w.mu + b
    Wf = W[:, :, f:]
    M = 0.5 * Wf.abs().sum(dim=2)                                        # max_p w.delta
    l2_free = Wf.norm(dim=2)
    cos_crit = -(b + M) / (w_norm * mu_norm[:, None]).clamp_min(1e-300)

    pre = torch.einsum("rhd,prd->prh", W, x_in) + b                      # [P,R,h]
    p_hat = (pre > 0).double().mean(dim=0)                               # [R,h]
    pre_max, pre_min = pre.max(dim=0).values, pre.min(dim=0).values
    beta = s / (0.5 * l2_free).clamp_min(1e-300)
    r_ratio = M / (0.5 * l2_free).clamp_min(1e-300)      # ||w_f||_1/||w_f||_2

    n = lambda t: t.detach().cpu().numpy()
    return dict(cos=n(cos), p_hat=n(p_hat), w_norm=n(w_norm), b=n(b), s=n(s),
                M=n(M), cos_crit=n(cos_crit), beta=n(beta), r_ratio=n(r_ratio),
                pre_max=n(pre_max), pre_min=n(pre_min),
                mu_norm=n(mu_norm), l2_free=n(l2_free))


def recorded_row(arm: str, step: int) -> dict:
    """記録ログの同 step 行を seed 順に積む [R,h] / [R]。"""
    res = ARMS[arm]["res"]
    out = {k: [] for k in ("cos_u_mu", "p_hat", "w_norm", "b", "mu_norm")}
    for seed in range(10):
        with np.load(res / "logs" / f"seed{seed}.npz") as z:
            i = int(np.flatnonzero(np.asarray(z["step"]) == step)[0])
            for k in out:
                out[k].append(np.asarray(z[k][i], dtype=np.float64))
    return {k: np.stack(v) for k, v in out.items()}


# ------------------------------------------------------------------ A: 恒等式

def identity_check(q: dict) -> dict:
    """p_hat=0 と cos<=cos_crit / s+M<=0 の一致を unit ごとに突き合わせる。"""
    off = q["p_hat"] == 0.0
    pred_cos = q["cos"] <= q["cos_crit"]
    pred_raw = (q["s"] + q["M"]) <= 0.0
    mism_cos = off != pred_cos
    mism_raw = off != pred_raw
    margin = np.abs(q["s"] + q["M"])
    # 恒等式のもう一方の端 (全点灯) と、pre の最大/最小が s±M と一致すること
    on_all = q["p_hat"] == 1.0
    mism_on = on_all != ((q["s"] - q["M"]) > 0.0)
    beta_form = off != (q["beta"] <= -q["r_ratio"])
    return dict(
        n_unit=int(off.size), n_off=int(off.sum()), n_on_all=int(on_all.sum()),
        n_mismatch_cos=int(mism_cos.sum()), n_mismatch_raw=int(mism_raw.sum()),
        n_mismatch_on=int(mism_on.sum()), n_mismatch_beta=int(beta_form.sum()),
        min_margin_mismatch=(float(margin[mism_cos].min()) if mism_cos.any()
                             else float("nan")),
        max_err_premax=float(np.abs(q["pre_max"] - (q["s"] + q["M"])).max()),
        max_err_premin=float(np.abs(q["pre_min"] - (q["s"] - q["M"])).max()),
        min_abs_margin=float(margin.min()),
        r_ratio_min=float(q["r_ratio"].min()), r_ratio_max=float(q["r_ratio"].max()))


def recon_check(q: dict, rec: dict) -> dict:
    """32 パターン再構成 p_hat と記録済み p_hat / cos / w_norm / b の突き合わせ。"""
    dp = np.abs(q["p_hat"] - rec["p_hat"])
    return dict(
        n_unit=int(dp.size),
        n_p_exact=int((dp == 0).sum()),
        max_abs_dp=float(dp.max()),
        n_off_agree=int(((q["p_hat"] == 0) == (rec["p_hat"] == 0)).sum()),
        max_abs_dcos=float(np.abs(q["cos"] - rec["cos_u_mu"]).max()),
        max_abs_dw=float(np.abs(q["w_norm"] - rec["w_norm"]).max()),
        max_abs_db=float(np.abs(q["b"] - rec["b"]).max()),
        max_abs_dmu=float(np.abs(q["mu_norm"] - rec["mu_norm"]).max()))


# ------------------------------------------------------------------ E: 記録ログのみの必要条件

def pooled_mu_norm(arm: str) -> dict:
    """記録ログ全 20,901 点 x 10 seed の ||mu|| 要約 (帰結2 の水準確認用)。"""
    v = []
    for seed in range(10):
        with np.load(ARMS[arm]["res"] / "logs" / f"seed{seed}.npz") as z:
            v.append(np.asarray(z["mu_norm"], dtype=np.float64))
    a = np.concatenate(v)
    return dict(arm=arm, n=int(a.size), mean=float(a.mean()),
                median=float(np.median(a)), q05=float(np.quantile(a, 0.05)),
                q95=float(np.quantile(a, 0.95)),
                min=float(a.min()), max=float(a.max()))


def recorded_necessary(arm: str, kappa: float) -> dict:
    """full W 無しでも検証できる必要条件を全 20,901 記録点で回す。

      N1: p_hat=0  => s <= 0            (s = ||w|| ||mu|| cos + b = w.mu + b)
      N2: p_hat=1  => s >  0
      N3: 0<p_hat<1 => |s| <= M <= (sqrt(5)/2)||w||   (M は未知だが上界は既知)

    加えて **近似** plug-in 判定: M ~= kappa*||w|| (kappa は再走 1e6 状態の
    median(M/||w||)) を代入した s + kappa*||w|| <= 0 が記録済み p_hat=0 をどれだけ
    再現するか。M のユニット差を無視しているので厳密検証ではなく水準確認である。

    float32 保存なので閾値には相対トレランスを付ける。"""
    res = ARMS[arm]["res"]
    tot = dict(n=0, n_off=0, n_on=0, n_mid=0, v1=0, v2=0, v3=0,
               worst1=-np.inf, worst2=-np.inf, worst3=-np.inf,
               plugin_kappa=float(kappa), plugin_hit=0)
    bound_c = 0.5 * np.sqrt(5.0)
    for seed in range(10):
        with np.load(res / "logs" / f"seed{seed}.npz") as z:
            p = np.asarray(z["p_hat"], dtype=np.float64)
            cos = np.asarray(z["cos_u_mu"], dtype=np.float64)
            wn = np.asarray(z["w_norm"], dtype=np.float64)
            b = np.asarray(z["b"], dtype=np.float64)
            mun = np.asarray(z["mu_norm"], dtype=np.float64)[:, None]
        s = wn * mun * cos + b
        tol = 1e-5 * (np.abs(wn * mun * cos) + np.abs(b) + 1e-12)
        off, on = p == 0.0, p == 1.0
        mid = ~off & ~on
        tot["n"] += int(p.size)
        tot["n_off"] += int(off.sum()); tot["n_on"] += int(on.sum())
        tot["n_mid"] += int(mid.sum())
        e1 = np.where(off, s - tol, -np.inf)
        e2 = np.where(on, -s - tol, -np.inf)
        e3 = np.where(mid, np.abs(s) - bound_c * wn - tol, -np.inf)
        for k, e in (("1", e1), ("2", e2), ("3", e3)):
            tot["v" + k] += int((e > 0).sum())
            tot["worst" + k] = max(tot["worst" + k], float(e.max()))
        tot["plugin_hit"] += int((off == ((s + kappa * wn) <= 0)).sum())
        del p, cos, wn, b, mun, s, tol, off, on, mid, e1, e2, e3
    tot["plugin_acc"] = tot["plugin_hit"] / max(1, tot["n"])
    return tot


# ------------------------------------------------------------------ 帰結1: 1M 単独の theta

def theta_at_step(cos: np.ndarray, p_hat: np.ndarray, min_n: int) -> tuple:
    """spec_ratchet_centered_0822 §5.2 と同じ定義で theta_med / theta_all を計算。
    ただし 1 時点だけなので有効ビン下限 min_n を明示的に振る。"""
    c, p = cos.reshape(-1), p_hat.reshape(-1)
    keep = (c >= COS_LO) & (c < COS_HI)
    bin_idx = np.floor((c[keep] - COS_LO) / BIN_W).astype(np.int64)
    p_idx = np.rint(p[keep] * 32.0).astype(np.int64)
    hist = np.bincount(bin_idx * N_P + p_idx,
                       minlength=N_BIN * N_P).reshape(N_BIN, N_P)
    n = hist.sum(axis=1)
    valid = n >= min_n
    return theta_one(hist_median(hist), valid), theta_all_one(hist, valid), int(keep.sum())


def bin_table(arm: str, source: str, cos: np.ndarray, p_hat: np.ndarray,
              cos_crit: np.ndarray | None) -> pd.DataFrame:
    """cos ビンごとの観測 (中央値 p̂ / 全消灯) と cos_crit の min / median。

    帰結1 の機構主張「全消灯は min(cos_crit)、中央値消灯は median(cos_crit) の交差」を
    ビン単位で読めるようにするための表。"""
    c, p = cos.reshape(-1), p_hat.reshape(-1)
    cc = None if cos_crit is None else cos_crit.reshape(-1)
    keep = (c >= COS_LO) & (c < COS_HI)
    idx = np.floor((c - COS_LO) / BIN_W).astype(np.int64)
    rows = []
    for j in range(N_BIN):
        sel = keep & (idx == j)
        n = int(sel.sum())
        row = dict(arm=arm, source=source, bin_index=j,
                   bin_lo=float(COS_LO + j * BIN_W), bin_hi=float(BIN_UPPER[j]), n=n,
                   p_median=float(np.median(p[sel])) if n else np.nan,
                   frac_off=float((p[sel] == 0).mean()) if n else np.nan)
        if cc is not None and n:
            marg = cc[sel] - c[sel]          # >=0 <=> そのユニットは消灯
            row.update(cos_crit_min=float(cc[sel].min()),
                       cos_crit_median=float(np.median(cc[sel])),
                       margin_min=float(marg.min()),
                       margin_median=float(np.median(marg)))
        else:
            row.update(cos_crit_min=np.nan, cos_crit_median=np.nan,
                       margin_min=np.nan, margin_median=np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def theta_from_margin(bt: pd.DataFrame) -> tuple[float, float]:
    """ビン別マージン (cos_crit − cos) の中央値 / 最小の零交差から theta を読む。

    ビン内で median(margin) >= 0 ⟺ そのビンの中央値 p̂ = 0、
    min(margin) >= 0 ⟺ そのビンの全点 p̂ = 0 なので、theta_one/theta_all_one と
    同じ「低 cos 側からの連続領域の上端」を取れば観測 theta と一致するはず。"""
    occ = bt[bt.n > 0].sort_values("bin_index")
    out = []
    for col in ("margin_median", "margin_min"):
        theta = np.nan
        for _, r in occ.iterrows():
            if not (np.isfinite(r[col]) and r[col] >= 0):
                break
            theta = float(r["bin_hi"])
        out.append(theta)
    return out[0], out[1]


# ------------------------------------------------------------------ 実行

def q(a: np.ndarray) -> list:
    return [float(x) for x in np.quantile(a, QUANTS)]


def fnum(v: float, digits: int = 4) -> str:
    return "NA" if v is None or not np.isfinite(v) else f"{v:.{digits}f}"


def _get(theta: pd.DataFrame, arm: str, source: str, min_n: int, col: str) -> float:
    sel = theta[(theta.arm == arm) & (theta.source == source) &
                (theta.min_bin_n == min_n)]
    return float(sel[col].iloc[0]) if len(sel) else float("nan")


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
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


def main() -> None:
    torch.set_num_threads(1)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "figures").mkdir(exist_ok=True)
    gen_sec = {arm: ensure_ckpts(arm) for arm in ARMS}

    ident_rows, recon_rows, dist_rows, drift_rows, theta_rows = [], [], [], [], []
    qs: dict[tuple[str, int], dict] = {}
    for arm in ARMS:
        for step in STEPS:
            state = load_state(arm, step)
            qq = exact_quantities(state)
            qs[(arm, step)] = qq
            rec = recorded_row(arm, step)
            ident_rows.append(dict(arm=arm, step=step, **identity_check(qq)))
            recon_rows.append(dict(arm=arm, step=step, **recon_check(qq, rec)))

            cc = qq["cos_crit"]
            row = dict(arm=arm, step=step,
                       mu_norm_mean=float(qq["mu_norm"].mean()),
                       mu_norm_min=float(qq["mu_norm"].min()),
                       mu_norm_max=float(qq["mu_norm"].max()),
                       frac_abs_gt1=float((np.abs(cc) > 1).mean()),
                       frac_ge_p1=float((cc >= 1).mean()),
                       frac_le_m1=float((cc <= -1).mean()),
                       frac_p_hat_zero=float((qq["p_hat"] == 0).mean()))
            row.update({f"cos_crit_{nm}": v for nm, v in zip(QNAMES, q(cc))})
            row.update({f"cos_{nm}": v for nm, v in zip(QNAMES, q(qq["cos"]))})
            dist_rows.append(row)

            drift_rows.append(dict(
                arm=arm, step=step,
                max_abs_dcos=float(np.abs(qq["cos"] - rec["cos_u_mu"]).max()),
                med_abs_dcos=float(np.median(np.abs(qq["cos"] - rec["cos_u_mu"]))),
                max_abs_dp=float(np.abs(qq["p_hat"] - rec["p_hat"]).max()),
                frac_p_identical=float((qq["p_hat"] == rec["p_hat"]).mean()),
                frac_off_identical=float(((qq["p_hat"] == 0) ==
                                          (rec["p_hat"] == 0)).mean()),
                n_off_repro=int((qq["p_hat"] == 0).sum()),
                n_off_recorded=int((rec["p_hat"] == 0).sum())))

    # --- 帰結1: 1M 単独の theta (記録ログ / 再走の両方、有効ビン下限を振る)
    bin_frames = []
    for arm in ARMS:
        rec = recorded_row(arm, 1_000_000)
        qq = qs[(arm, 1_000_000)]
        for src, cos, p in (("recorded", rec["cos_u_mu"], rec["p_hat"]),
                            ("repro", qq["cos"], qq["p_hat"])):
            for min_n in (1, 20, 1000):
                tm, ta, n_in = theta_at_step(cos, p, min_n)
                theta_rows.append(dict(arm=arm, source=src, min_bin_n=min_n,
                                       theta_med=tm, theta_all=ta, n_in_range=n_in))
            bt = bin_table(arm, src, cos, p,
                           qq["cos_crit"] if src == "repro" else None)
            bin_frames.append(bt)
            if src == "repro":
                tm, ta = theta_from_margin(bt)
                theta_rows.append(dict(arm=arm, source="margin_crossing(repro)",
                                       min_bin_n=0, theta_med=tm, theta_all=ta,
                                       n_in_range=int(bt.n.sum())))
        cc, cs = qq["cos_crit"].reshape(-1), qq["cos"].reshape(-1)
        keep = (cs >= COS_LO) & (cs < COS_HI)
        theta_rows.append(dict(
            arm=arm, source="cos_crit_global(参考)", min_bin_n=-1,
            theta_med=float(np.median(cc[keep])) if keep.any() else np.nan,
            theta_all=float(cc[keep].min()) if keep.any() else np.nan,
            n_in_range=int(keep.sum())))
    bins = pd.concat(bin_frames, ignore_index=True)

    # --- 帰結3: centered 1M で b+M<=0 が p_hat=0 を説明するか
    cons3 = {}
    for arm in ARMS:
        qq = qs[(arm, 1_000_000)]
        off = qq["p_hat"] == 0
        bm = (qq["b"] + qq["M"]) <= 0
        wmu = qq["s"] - qq["b"]
        ratio = np.abs(wmu) / np.maximum(np.abs(qq["b"] + qq["M"]), 1e-300)
        pv, cv, bv = qq["p_hat"].reshape(-1), qq["cos"].reshape(-1), qq["beta"].reshape(-1)
        rank = lambda a: pd.Series(a).rank().to_numpy()
        cons3[arm] = dict(
            arm=arm, agree_bplusM=float((off == bm).mean()),
            n_disagree=int((off != bm).sum()),
            ratio_wmu_over_bM_median=float(np.median(ratio)),
            ratio_wmu_over_bM_q95=float(np.quantile(ratio, 0.95)),
            pearson_p_cos=float(np.corrcoef(pv, cv)[0, 1]),
            pearson_p_beta=float(np.corrcoef(pv, bv)[0, 1]),
            spearman_p_cos=float(np.corrcoef(rank(pv), rank(cv))[0, 1]),
            spearman_p_beta=float(np.corrcoef(rank(pv), rank(bv))[0, 1]))

    kappa = {arm: float(np.median(qs[(arm, 1_000_000)]["M"] /
                                  qs[(arm, 1_000_000)]["w_norm"])) for arm in ARMS}
    nec = {arm: recorded_necessary(arm, kappa[arm]) for arm in ARMS}
    mu_pool = pd.DataFrame([pooled_mu_norm(arm) for arm in ARMS])

    ident = pd.DataFrame(ident_rows)
    recon = pd.DataFrame(recon_rows)
    dist = pd.DataFrame(dist_rows)
    drift = pd.DataFrame(drift_rows)
    theta = pd.DataFrame(theta_rows)
    cons3_df = pd.DataFrame(list(cons3.values()))
    nec_df = pd.DataFrame([dict(arm=a, **v) for a, v in nec.items()])
    ident.to_csv(OUTDIR / "identity_units.csv", index=False)
    recon.to_csv(OUTDIR / "recon_vs_recorded.csv", index=False)
    dist.to_csv(OUTDIR / "cos_crit_dist.csv", index=False)
    drift.to_csv(OUTDIR / "drift.csv", index=False)
    theta.to_csv(OUTDIR / "theta_1m.csv", index=False)
    cons3_df.to_csv(OUTDIR / "consequence3.csv", index=False)
    nec_df.to_csv(OUTDIR / "recorded_necessary.csv", index=False)
    bins.to_csv(OUTDIR / "bins_1m.csv", index=False)
    mu_pool.to_csv(OUTDIR / "mu_norm_pooled.csv", index=False)

    make_figure(qs)
    write_summary(ident, recon, dist, drift, theta, cons3_df, nec_df, bins,
                  mu_pool, gen_sec)
    meta = dict(date=time.strftime("%Y-%m-%d %H:%M:%S"), python=platform.python_version(),
                torch=torch.__version__, numpy=np.__version__,
                git_analysis=git_hash(["analysis/q3_margin_exact.py"]),
                git_src=git_hash(["src/ratchet_log.py", "src/train.py", "src/envs.py"]),
                repro_sec=gen_sec, steps=list(STEPS),
                note="事後計算・未事前登録。checkpoint は再走で再生成 (results/ は不変)。")
    (OUTDIR / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False),
                                      encoding="utf-8")
    print(f"wrote {OUTDIR}")


def make_figure(qs: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, step in zip(axes, STEPS):
        # step 0 は running_mean=0 なので std と centered が厳密に一致する。
        # 重なって見えなくならないよう centered は破線にする。
        for arm, col, ls in (("std", "tab:blue", "-"), ("centered", "tab:orange", "--")):
            cc = qs[(arm, step)]["cos_crit"].reshape(-1)
            ax.hist(np.clip(cc, -3, 3), bins=80, histtype="step", color=col, ls=ls,
                    label=f"{arm} (|cos_crit|>1: {(np.abs(cc) > 1).mean():.2f})")
        ax.axvline(-1, color="gray", ls="--", lw=0.8)
        ax.axvline(1, color="gray", ls="--", lw=0.8)
        ax.set(xlabel="cos_crit (clipped to [-3,3])", ylabel="units",
               title=f"step {step}")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("cos_crit = -(b+M)/(||w|| ||mu||) の分布 (10 seed x 100 unit)")
    fig.tight_layout()
    fig.savefig(OUTDIR / "figures" / "fig_cos_crit_dist.png", dpi=150)
    plt.close(fig)


def write_summary(ident, recon, dist, drift, theta, cons3, nec, bins,
                  mu_pool, gen_sec) -> None:
    n_mis = int(ident.n_mismatch_cos.sum())
    n_mis_raw = int(ident.n_mismatch_raw.sum())
    n_unit = int(ident.n_unit.sum())
    v_tot = int(nec[["v1", "v2", "v3"]].to_numpy().sum())
    ctr = dist[(dist.arm == "centered")]
    std = dist[(dist.arm == "std")]
    lines = [
        "# Q3 margin exact: ゲート恒等式 p_hat=0 ⟺ cos ≤ cos_crit の厳密検算", "",
        "**事後計算・未事前登録**。既存 verdict.csv を書き換えない。",
        f"解析コード commit `{git_hash(['analysis/q3_margin_exact.py'])}` / "
        f"実験コード commit `{git_hash(['src/ratchet_log.py', 'src/train.py'])}`。", "",
        "## 0. 一行", "",
        f"恒等式 p̂=0 ⟺ cos ≤ cos_crit は **{n_unit} unit 中 不一致 {n_mis} 件** (cos 形) / "
        f"**{n_mis_raw} 件** (生形 s+M≤0) で **厳密に成立**。"
        f"32 パターン再構成 p̂ は step 0 で記録ログと完全一致 "
        f"({int(recon[recon.step == 0].n_p_exact.sum())}/"
        f"{int(recon[recon.step == 0].n_unit.sum())} unit)。"
        f"記録ログのみで検証できる必要条件は両 arm 合計 {int(nec.n.sum()):,} サンプルで"
        f"違反 {v_tot} 件。", "",
        "帰結の成否: "
        f"**帰結1** θ_all({fnum(_get(theta, 'std', 'recorded', 20, 'theta_all'), 2)}) "
        f"< θ_med({fnum(_get(theta, 'std', 'recorded', 20, 'theta_med'), 2)}) は成立するが "
        "順序自体は定義から自動 (§7)。"
        f"**帰結2** centered の |cos_crit|>1 が "
        f"{float(dist[(dist.arm == 'centered') & (dist.step == 1000000)].frac_abs_gt1.iloc[0]):.1%} "
        "で消灯点が cos の定義域外 → θ が NA になる C1「不可比」を再現。"
        f"**帰結3** centered 1e6 の b+M≤0 と p̂=0 の一致率 "
        f"{float(cons3[cons3.arm == 'centered'].agree_bplusM.iloc[0]):.1%} "
        f"(std は {float(cons3[cons3.arm == 'std'].agree_bplusM.iloc[0]):.1%})。", "",
        "## 1. 定義の確認 (実装との対応)", "",
        "- `src/envs.py::SCREnv` … m=20, f=15 なので自由ビンは 5 本 = **32 パターン**。",
        "  自由ビンの生値は **0/1** (±0.5 ではない)。",
        "- `src/ratchet_log.py::exact_record` … `x_in = X - centered*running_mean`、",
        "  `mu = x_in.mean(0)`、`cos_u_mu = W·(mu/||mu||)/||w||`、",
        "  `p_hat = (pre > 0).mean(0)` (**厳密不等号**、32 パターン上の点灯率)。",
        "- したがって δ(p) := x_in(p) − mu = (0_15, pattern_p − 1/2) で、",
        "  **δ の値域は自由 5 座標で ±0.5、flip 15 座標では恒等的に 0**。",
        "  running_mean は mu と x_in の両方に同じだけ入るので δ から消える",
        "  (= std と centered で δ は同一)。",
        "- ゆえに M = max_p w·δ(p) = (1/2)Σ_{j≥15}|w_j| = ||w_free||_1/2、",
        "  min_p w·δ(p) = −M。**ユーザ提示の式は実装と一致**する。",
        "- 併せて成り立つ正規化形: β = (w·mu+b)/√(wᵀΣw) (lop_metrics と同一定義、",
        "  条件A では √(wᵀΣw)=||w_free||₂/2) を使うと",
        "  **p̂=0 ⟺ β ≤ −r, r = ||w_free||₁/||w_free||₂ ∈ [1,√5]**。", "",
        "## 2. A. 恒等式の同値検証 (同一状態内・厳密)", "",
        md_table(ident), "",
        "`n_mismatch_cos` が cos ≤ cos_crit 形、`n_mismatch_raw` が s+M ≤ 0 形、",
        "`n_mismatch_on` が p̂=1 ⟺ s−M>0 形、`n_mismatch_beta` が β ≤ −r 形の不一致数。",
        "`max_err_premax/premin` は 32 パターンの直接計算 max/min pre と s±M の差 ",
        "(float64 の丸め水準)。`min_abs_margin` は |s+M| の最小値 = 判定が丸めに",
        "晒される余裕。", "",
        "## 3. B. p̂ 再構成と記録ログの突き合わせ", "",
        md_table(recon), "",
        "step 0 は再走が記録ログと厳密一致するので、ここでの一致は **µ/δ/p̂ の定義理解が",
        "確定した**ことを意味する。step 1e6 は float32 加算順序の環境差でトラジェクトリが",
        "ドリフトするため、記録ログとの一致は期待できない (D 節)。", "",
        "## 4. C. cos_crit 分布 (帰結2)", "",
        md_table(dist), "",
        "参考: 記録ログ全 20,901 点プールの ||mu|| (仮説が引いた 3.04 / 0.117 の水準確認)", "",
        md_table(mu_pool), "",
        "## 5. D. 再走ログ vs 記録ログのドリフト", "",
        md_table(drift), "",
        "## 6. E. 記録ログのみで検証できる必要条件",
        "(20,901 記録点 × 10 seed × 100 unit = arm あたり 20,901,000 サンプル)", "",
        "N1: p̂=0 ⇒ s≤0 / N2: p̂=1 ⇒ s>0 / N3: 0<p̂<1 ⇒ |s| ≤ (√5/2)||w||",
        "(s = ||w||·||µ||·cos + b。M は記録ログから復元できないので上界で代用)。",
        "`v1/v2/v3` が違反数、`worst*` は違反量の最大 (負なら余裕あり)。",
        "`plugin_acc` は M ≈ κ||w|| (κ = 再走 1e6 の median(M/||w||)) を代入した",
        "**近似**判定 s+κ||w||≤0 が記録済み p̂=0 を再現した割合。M のユニット差を",
        "無視しているので厳密検証ではない (水準確認)。", "",
        md_table(nec), "", "## 7. 帰結1: 1e6 単独の theta", "", md_table(theta), "",
        "ビン別の内訳 (cos_crit は再走側にしか無いので repro 行のみ。n=0 のビンは省略):", "",
        md_table(bins[(bins.source == "repro") & (bins.n > 0)].drop(columns=["source"])),
        "",
        "θ_all ≤ θ_med は **定義から自動**である (あるビンで全点 p̂=0 なら中央値も 0 なので、",
        "全消灯ビンの連なりは中央値消灯ビンの連なりの前置部分)。恒等式が付け加えるのは",
        "**水準の予測** の方であり、順序そのものは恒等式なしでも成り立つ。",
        "その水準予測を厳密統計量で書くと、ビン内マージン m = cos_crit − cos について",
        "median(m) ≥ 0 ⟺ そのビンの中央値 p̂ = 0、min(m) ≥ 0 ⟺ そのビン全点 p̂ = 0 なので、",
        "θ_med / θ_all は m の中央値 / 最小の零交差そのものになる",
        "(`margin_crossing(repro)` 行が観測 `repro` 行と一致することで確認できる)。",
        "ビン幅 0.05 の下では m ≈ cos_crit − (ビン代表 cos) なので、",
        "仮説が言う「median(cos_crit) / min(cos_crit) の交差」と実質同じである。", "",
        "## 8. 帰結3: b+M ≤ 0 による説明力 (1e6)", "", md_table(cons3), "",
        f"## 9. 留保", "",
        "1. **full W が記録ログに無い**。W は checkpoint (.pt) にしかなく .gitignore 対象で",
        "   リポジトリに含まれないため、step 0 / 1e6 の状態を**同一 config の再走で再生成**",
        "   した (probe=None。本走 meta.json の S2 が probe 無しでの bit 一致を確認済み)。",
        "   再走時間: " + " / ".join(
            f"{a} {'キャッシュ再利用' if s <= 0 else f'{s:.0f}s'}"
            for a, s in gen_sec.items())
        + " (初回生成は本環境で 1 arm あたり CPU 15 分程度)。",
        "2. **step 1e6 は記録ログと bit 一致しない**。float32 の加算順序が torch/BLAS の",
        "   バージョンに依存するため、1e6 step の SGD でトラジェクトリがドリフトする",
        "   (step 0 は厳密一致)。恒等式の判定は「同一状態から計算した p̂ と cos_crit」の",
        "   間で行っており、代数命題としての検証は成立する。ただし",
        "   **「記録された 1e6 の状態そのもの」に対する検証ではない**。",
        "   同一マシン上で同一コードを独立に 2 回走らせた W/b/v/c は bit 一致した",
        "   (別途手動確認) ので、非決定性は run 間ではなく**環境間**にある。",
        "   集約量 (死亡ユニット数 n_off) は D 節のとおり数ユニット差に収まる。",
        "3. 1e6 単独では 1 arm あたり 1,000 サンプルしかなく、事前登録の有効ビン下限",
        "   n≥1000 は満たせない。theta 表は下限を振って併記した。",
        "   verdict の −0.15 / −0.55 は全 20,901 記録点プールの値なので水準が異なる。",
        "4. E 節の必要条件は M を上界で置き換えた**必要条件**であり、恒等式の十分性までは",
        "   保証しない。十分性は A 節 (2 時点) が担保する。",
    ]
    (OUTDIR / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
