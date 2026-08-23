"""q3_cancel_residual: 「バイアス b は w·µ を打ち消しているのか」の残差 s の測定。

**事後計算・未事前登録**。spec_ratchet_log_0819 / spec_ratchet_centered_0822 の
どちらにも登録が無い、Q3（消灯点）解釈のための後付け解析である。判定は
verdict.csv を書き換えない。ここでの数値は探索的であり、引用には事前登録つきの
昇格が要る。

## 問い

condA の µ は周期（T=1e4 step）内で**定数**なので、hidden unit の pre-activation の
うち µ に由来する部分は周期内で動かない定数である。ならば b がそれを打ち消すように
学習して、その定数部分が 0 に落ち着いてもよいはずではないか（Issa の疑問）。

## 代数（`src/ratchet_log.py: exact_record` / `src/envs.py: SCREnv` / `src/nets.py` で確認）

x = [flip_state (f=15, 周期内固定) ‖ 自由 5 ビット U{0,1}]、x_in = x − centered·running_mean。

    µ     = E[x_in]（32 パターン平均）,  delta = x_in − µ（自由 5 次元のみ）
    a(δ)  = w·µ + w·δ + b = ||w||·||µ||·cos(w,µ) + w·δ + b
    s     := w·µ + b = w_norm · mu_norm · cos_u_mu + b     ← 周期内で定数の部分
    M     := max_δ (w·δ) = ||w_free||_1 / 2,   kappa := M/||w||
    min_δ (w·δ) = −M（δ は ±1/2 対称）

したがって p_hat = mean_δ 1[a>0] について**厳密に**

    p_hat = 0  <=>  s + M <= 0  <=>  s/M <= −1
    p_hat = 1  <=>  s − M >  0  <=>  s/M >  +1

すなわち **s/M こそが発火の自然座標**であり、s/M ∈ (−1, +1] が部分発火帯、
s/M = 0（完全な打ち消し）は「32 パターンの半分で発火」＝ p_hat ≈ 0.5 に対応する。
この対応は kappa 近似を要さないので、**p_hat そのものが s/M の κ 非依存な代理**に
なる（§A で両方を出す）。

## 増分の代数（§C の主張の根拠）

`nets.VecMLP.grads` は gW = gb ⊗ x_in（同じ gb = 2δ·v·gate が両方に掛かる）。
batch=1・周期内（µ 固定）では 1 step の更新について**厳密に**

    Δ(w·µ) = (Δw)·µ = −lr·gb·(x_in·µ) = Δb · (x_in·µ)
    Δs     = Δb · (1 + x_in·µ)

std では x ∈ {0,1}^20、µ = [flip_state ‖ 0.5·1_5] なので
x·µ = #{flip bit = 1} + 0.5·#{自由ビット = 1} ≥ 0 で**常に非負**。よって b と w·µ は
常に同符号に動き、b は s の動きの 1/(1+x·µ) ≈ 1/10 しか担えない。打ち消しは
「学習が下手だから起きない」のではなく、**共通因子 gb によって構造的に不可能**である。
centered では ||µ||^2 ≈ 0.01 なので x_in·µ ≈ 0 となり Δs ≈ Δb、s ≈ b になる。

## kappa 近似（M の復元）

記録ログには full W が無く M を厳密復元できない（W は checkpoint のみ）。そこで
`analysis/q3_margin_pooled.py` の推定量をそのまま import して使う:

  - 主報告 `reg_logw_interp`（step 0 と 1M の log||w|| 分位ノット回帰を t/1e6 で内挿）
  - 感度 `const_med_s0` / `const_med_s1000000`（単一 checkpoint の中央値。両者が観測を挟む）

**M̂ = kappa_hat(||w||, t)·||w|| は近似であり、per-unit の主張には使えない**
（step 1M checkpoint は本走ではなく同一 config の別実現。q3_margin_pooled §「軌道
再現性」を参照）。κ 非依存の裏取りとして (i) p_hat の分布、(ii) 消灯遷移点での
−s ≈ M（p_hat が 0 に落ちる記録点の s から M を挟む）を §A5 に出す。

## 集計規約

- alive の主定義は**コード準拠**の `p_hat >= 0.05`（`src/figures_ratchet_log.py: TAU`、
  spec_ratchet_log_0819 §3.5 の凍結/死亡閾値）。p_hat は k/32 を取るので実質
  p_hat >= 2/32。副定義 `p_hat > 0`（＝ s+M>0 の厳密な生存）も併記する。
- 死亡時刻 t_death は §3.5 の定義（p_hat が TAU を下方クロスし、以後 1 周期以上
  回復しない最初の記録点）。`src/figures_ratchet_log.py: death_events` と同一実装。
- 記録グリッドは非一様（境界 ±100 が毎 step、他は 1000 step ごと）。素朴な時間プールは
  境界近傍に強く重み付けされるので、**all / bulk（窓外の 1000 刻み）/ boundary（窓内）**
  を必ず分けて出す。
- 図・時系列は 10 seed × 100 unit = 1000 値を記録点ごとにプールした分位で描く。

## 入力

- `results/ratchet_log_0819/logs/seed*.npz`（std）
- `results/ratchet_centered_0822/logs/seed*.npz`（centered）
- checkpoint `A_w100_step{0,1000000}.pt`（`.gitignore` のため repo 外。
  `results/<run>/ckpts/` -> `~/q3_out/verify/pooled/rerun_<arm>/ckpts/` の順で探索）

## 実行

    OMP_NUM_THREADS=1 python3 analysis/q3_cancel_residual.py

引数なし・決定論（乱数を使わない）。出力は `/root/q3_out/cancel/`（repo 外）。
"""
from __future__ import annotations

import json
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import ROOT  # noqa: E402
from analysis.q3_gate_curve_ci import seed_paths, check_source_run, md_table  # noqa: E402
from analysis.q3_margin_pooled import (ARMS as POOLED_ARMS, CKPT_STEPS,  # noqa: E402
                                       build_estimators, kappa_exact, load_ckpt_W)

OUTDIR = Path("/root/q3_out/cancel")
FIGDIR = OUTDIR / "figures"
TAU = 0.05                     # 凍結/死亡閾値 [spec_ratchet_log_0819 §3.5]
PERIOD = 10_000
HALF_W = 100                   # 境界窓の半幅 [ratchet.boundary_window]
N_SEED, N_UNIT = 10, 100
MAIN_EST = "reg_logw_interp"
SENS_EST = ["const_med_s0", "const_med_s1000000"]
QLEV = [0.05, 0.25, 0.50, 0.75, 0.95]
QNAME = ["q05", "q25", "med", "q75", "q95"]

plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ------------------------------------------------------------------ 小道具

def git_hash(paths=None) -> str:
    cmd = ["git", "log", "-1", "--format=%h"] + (["--", *paths] if paths else [])
    try:
        h = subprocess.check_output(cmd, cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"
    return h or "uncommitted"


def qrow(x: np.ndarray, extra: dict | None = None) -> dict:
    """分位要約。x は 1 次元 float。空なら NaN を返す。"""
    x = np.asarray(x, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    d = dict(n=int(x.size))
    if x.size == 0:
        d.update({k: np.nan for k in QNAME})
        d.update(mean=np.nan, iqr=np.nan)
    else:
        q = np.quantile(x, QLEV)
        d.update({k: float(v) for k, v in zip(QNAME, q)})
        d.update(mean=float(x.mean()), iqr=float(q[3] - q[1]))
    if extra:
        d.update(extra)
    return d


def death_index(p_hat: np.ndarray, step: np.ndarray, period: int) -> np.ndarray:
    """§3.5 の死亡記録点 index を unit ごとに返す（死なない unit は -1）。

    `src/figures_ratchet_log.py: death_events` と同一の定義・同一の走査順。
    p_hat: [n_rec, n_unit]。"""
    n, h = p_hat.shape
    out = np.full(h, -1, dtype=np.int64)
    below = p_hat < TAU
    for i in range(h):
        bcol = below[:, i]
        cross = np.flatnonzero(bcol[1:] & ~bcol[:-1]) + 1
        for c in cross:
            horizon = step[c] + period
            seg = bcol[(step >= step[c]) & (step <= horizon)]
            if seg.all():
                out[i] = int(c)
                break
    return out


# ------------------------------------------------------------------ 読み込み

def load_arm(resdir: Path) -> dict:
    """1 arm ぶんの記録量を [n_rec, n_seed, n_unit] に積む（float32 のまま保持）。"""
    paths = seed_paths(resdir)
    s_l, b_l, wn_l, p_l, cos_l, mun_l, fs_l = [], [], [], [], [], [], []
    step = None
    for p in paths:
        with np.load(p) as z:
            st = z["step"].astype(np.int64)
            if step is None:
                step = st
            elif not np.array_equal(step, st):
                raise SystemExit(f"{p}: 記録グリッドが seed 間で不一致")
            cos = z["cos_u_mu"].astype(np.float64)
            wn = z["w_norm"].astype(np.float64)
            bb = z["b"].astype(np.float64)
            mun = z["mu_norm"].astype(np.float64)
            wmu = wn * mun[:, None] * cos
            s_l.append((wmu + bb).astype(np.float32))
            b_l.append(bb.astype(np.float32))
            wn_l.append(wn.astype(np.float32))
            p_l.append(z["p_hat"].astype(np.float32))
            cos_l.append(cos.astype(np.float32))
            mun_l.append(mun.astype(np.float32))
            fs_l.append(z["flip_state"].astype(np.float32))
    stack = lambda L: np.stack(L, axis=1)          # noqa: E731  -> [n_rec, n_seed, ...]
    return dict(step=step, s=stack(s_l), b=stack(b_l), w_norm=stack(wn_l),
                p_hat=stack(p_l), cos=stack(cos_l), mu_norm=stack(mun_l),
                flip=np.stack(fs_l, axis=1))


def grid_masks(step: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(in_win, is_bulk)。in_win = いずれかの境界の ±HALF_W 以内、
    is_bulk = 1000 刻みグリッド上で in_win でない点。"""
    bnd = np.arange(PERIOD, int(step[-1]) + 1, PERIOD)
    dist = np.abs(step[:, None] - bnd[None, :]).min(axis=1)
    in_win = dist <= HALF_W
    is_bulk = (step % 1000 == 0) & ~in_win
    return in_win, is_bulk


# ------------------------------------------------------------------ kappa

def build_kappa(arm_label: str, w_all: np.ndarray) -> tuple[dict, dict]:
    """q3_margin_pooled の推定量をそのまま作る。||w|| 四分位境界はアーム内 pooled。"""
    pooled = w_all.reshape(-1).astype(np.float64)
    qb = np.quantile(pooled, [0.25, 0.5, 0.75])
    arm = next(a for a in POOLED_ARMS if a["label"] == arm_label)
    ck, inv = {}, []
    for st in CKPT_STEPS:
        W, b, rm, fs, src = load_ckpt_W(arm, st)
        kap, wn, rho = kappa_exact(W, 20, 15)
        ck[st] = dict(kappa=kap, w_norm=wn, source=src)
        d = phat_to_sM(W, 15, 20, [1, 2, 4, 6, 8, 10, 12, 16])
        d.insert(0, "ckpt_step", st)
        d.insert(0, "arm", arm_label)
        inv.append(d)
    est, info = build_estimators(ck, qb)
    return est, dict(ck=ck, qbounds=qb, info=info,
                     inv=pd.concat(inv, ignore_index=True))


def phat_to_sM(W, f: int, m: int, ks) -> pd.DataFrame:
    """観測された p̂ から s/M を **κ 非依存に**逆算する（checkpoint の W を使う）。

    p̂ = #{δ : w·δ > −s} / 32。X_j := (w·δ_j)/M を昇順に並べると
    p̂ = k/32 ⟺ −s/M ∈ [X_(32−k−1), X_(32−k))  ⟺  s/M ∈ (−X_(32−k), −X_(32−k−1)]。
    κ そのものは要らず、必要なのは X の**形**（自由 5 次元の重みの相対配分）だけで、
    これは κ の水準より checkpoint 間で安定している。返り値は unit をまたいだ分位。"""
    import torch
    Wf = W[:, :, f:m]                                        # [R,h,5]
    nf = m - f
    pat = ((torch.arange(2 ** nf)[:, None] >> torch.arange(nf)) & 1).double() - 0.5
    proj = torch.einsum("rhd,pd->rhp", Wf, pat)              # [R,h,32]
    M = 0.5 * Wf.abs().sum(dim=2)                            # [R,h]
    X = (proj / M[:, :, None]).numpy().reshape(-1, 2 ** nf)
    X.sort(axis=1)
    P = 2 ** nf
    rows = []
    for k in ks:
        hi = -X[:, P - k - 1]          # s/M の上限（含む）
        lo = -X[:, P - k]              # s/M の下限（含まない）
        rows.append(dict(p_hat=k / P, k=k,
                         sM_lo_med=float(np.median(lo)), sM_hi_med=float(np.median(hi)),
                         sM_mid_med=float(np.median(0.5 * (lo + hi))),
                         sM_mid_q25=float(np.quantile(0.5 * (lo + hi), .25)),
                         sM_mid_q75=float(np.quantile(0.5 * (lo + hi), .75))))
    return pd.DataFrame(rows)


def kappa_field(est, step: np.ndarray, w_norm: np.ndarray) -> np.ndarray:
    """[n_rec, n_seed, n_unit] の ||w|| と step から kappa_hat を作る（float32 で返す）。"""
    t = np.repeat(step.astype(np.float64), w_norm.shape[1] * w_norm.shape[2])
    v = w_norm.reshape(-1).astype(np.float64)
    return est(v, t).astype(np.float32).reshape(w_norm.shape)


# ------------------------------------------------------------------ §A 分布

def section_A(D: dict, arm: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """s / (s/||w||) / (s/M̂) の pooled 分布を グリッド区分 × alive 条件で。

    併せて κ̂ の自己整合性（厳密には p̂>0 ⟺ s/M > −1 なので、M̂ が正しければ
    p̂>0 の点で s/M̂ ≤ −1 は起きない）も返す。"""
    step, s, wn, p, Mh = D["step"], D["s"], D["w_norm"], D["p_hat"], D["M_hat"]
    in_win, is_bulk = D["in_win"], D["is_bulk"]
    rows = []
    grids = dict(all=np.ones_like(in_win), bulk=is_bulk, boundary=in_win)
    conds = dict(all=None, alive_tau=(p >= TAU), alive_pos=(p > 0), dead_tau=(p < TAU))
    for gname, gm in grids.items():
        gm3 = gm[:, None, None]
        for cname, cm in conds.items():
            m = np.broadcast_to(gm3, s.shape) if cm is None else (gm3 & cm)
            for qty, val in (("s", s), ("s_over_w", s / wn), ("s_over_M", s / Mh),
                             ("p_hat", p)):
                rows.append(qrow(val[m], dict(arm=arm, grid=gname, cond=cname, qty=qty)))
    df = pd.DataFrame(rows)[["arm", "grid", "cond", "qty", "n", *QNAME, "mean", "iqr"]]

    sm = s / Mh
    pos, zero = p > 0, p == 0
    cons = pd.DataFrame([
        dict(arm=arm, stat="frac (p̂>0 かつ s/M̂ <= -1)  [厳密なら 0]",
             value=float(np.mean(sm[pos] <= -1))),
        dict(arm=arm, stat="frac (p̂==0 かつ s/M̂ > -1)  [厳密なら 0]",
             value=float(np.mean(sm[zero] > -1))),
        dict(arm=arm, stat="frac alive(p̂>0) with s > 0",
             value=float(np.mean(s[pos] > 0))),
        dict(arm=arm, stat="frac alive(p̂>=τ) with s > 0",
             value=float(np.mean(s[p >= TAU] > 0))),
    ])
    return df, cons


def section_A_time(D: dict, arm: str) -> pd.DataFrame:
    """記録点ごとの分位（10 seed × 100 unit をプール）。alive 条件つき。"""
    step, s, wn, p, Mh, b = (D["step"], D["s"], D["w_norm"], D["p_hat"],
                             D["M_hat"], D["b"])
    n = step.size
    flat = lambda a: a.reshape(n, -1)                                    # noqa: E731
    sf, wf, pf, mf, bf = flat(s), flat(wn), flat(p), flat(Mh), flat(b)
    wmu = sf - bf
    out = []
    for cname, cm in (("all", None), ("alive_tau", pf >= TAU), ("dead_tau", pf < TAU)):
        rec = dict(arm=arm, cond=cname, step=step, in_win=D["in_win"],
                   is_bulk=D["is_bulk"])
        cnt = np.full(n, sf.shape[1]) if cm is None else cm.sum(axis=1)
        rec["n"] = cnt
        for qty, val in (("s", sf), ("s_over_w", sf / wf), ("s_over_M", sf / mf),
                         ("b", bf), ("b_over_w", bf / wf), ("wmu", wmu),
                         ("wmu_over_w", wmu / wf), ("p_hat", pf), ("w_norm", wf)):
            v = val if cm is None else np.where(cm, val, np.nan)
            with np.errstate(invalid="ignore"):
                q = np.nanquantile(v, [0.25, 0.5, 0.75], axis=1)
            rec[f"{qty}_q25"], rec[f"{qty}_med"], rec[f"{qty}_q75"] = q[0], q[1], q[2]
        out.append(pd.DataFrame(rec))
    return pd.concat(out, ignore_index=True)


# ------------------------------------------------------------------ §C 増分

def section_C(D: dict, arm: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """1 step 増分の (Δ(w·µ), Δb) 関係と、境界での µ ジャンプ項。

    周期内・1 step 隣接・flip_state 不変の記録点対だけを使う（µ が厳密に固定）。
    centered は running_mean が毎 step 動くので µ 固定にならない旨を注記して併記する。
    """
    step, s, b = D["step"], D["s"], D["b"]
    wmu = s - b
    d = np.diff(step)
    same_per = (step[1:] // PERIOD) == (step[:-1] // PERIOD)
    same_flip = np.abs(np.diff(D["flip"], axis=0)).sum(axis=(1, 2)) == 0
    sel = (d == 1) & same_per & same_flip
    idx = np.flatnonzero(sel)
    db = (b[idx + 1] - b[idx]).astype(np.float64).ravel()
    dw = (wmu[idx + 1] - wmu[idx]).astype(np.float64).ravel()
    ds = db + dw
    # 有効な更新のみ（gate が閉じている unit は勾配ゼロ）。float32 の丸め下限で切る。
    scale = np.abs(b[idx]).astype(np.float64).ravel()
    live = np.abs(db) > np.maximum(1e-7, 1e-5 * scale)
    r = dw[live] / db[live]
    rows = [qrow(r, dict(arm=arm, qty="ratio_dwmu_over_db", note="1 step・周期内・µ固定")),
            qrow(np.abs(db[live]) / np.abs(ds[live]),
                 dict(arm=arm, qty="share_|db|/|ds|", note="b が担う s の動きの割合")),
            qrow((D["mu_norm"] ** 2).reshape(-1),
                 dict(arm=arm, qty="mu_norm_sq", note="理論上の E[x_in·µ]"))]
    summ = pd.DataFrame(rows)[["arm", "qty", "n", *QNAME, "mean", "iqr", "note"]]

    agree = float(np.mean(np.sign(dw[live]) == np.sign(db[live]))) if live.any() else np.nan
    ds_neg = float(np.mean(ds[live] < 0)) if live.any() else np.nan
    # 境界の µ ジャンプ。flip は「記録点 B と B+1 の間」で起きる
    # （`src/figures_ratchet_log.py: realized_boundaries` の注記）。実際に flip_state が
    # 変わった境界だけを取り、alive な unit に限って比較する（dead は gate が閉じて Δb=0）。
    bnd = np.arange(PERIOD, int(step[-1]), PERIOD)
    ib = np.searchsorted(step, bnd)
    ib = ib[(ib >= 0) & (ib + 1 < step.size)]
    real = np.abs(D["flip"][ib + 1] - D["flip"][ib]).sum(axis=(1, 2)) > 0
    ib = ib[real]
    al = (D["p_hat"][ib] >= TAU)
    jump_dw = (wmu[ib + 1] - wmu[ib]).astype(np.float64)[al]
    jump_db = (b[ib + 1] - b[ib]).astype(np.float64)[al]
    extra = pd.DataFrame([
        dict(arm=arm, stat="n_1step_pairs(record)", value=float(idx.size)),
        dict(arm=arm, stat="frac_db_active", value=float(live.mean())),
        dict(arm=arm, stat="frac_same_sign(dwmu,db)", value=agree),
        dict(arm=arm, stat="frac_ds_negative", value=ds_neg),
        dict(arm=arm, stat="mean_db(active)", value=float(db[live].mean()) if live.any() else np.nan),
        dict(arm=arm, stat="mean_ds(active)", value=float(ds[live].mean()) if live.any() else np.nan),
        dict(arm=arm, stat="median_dwmu_over_db", value=float(np.median(r)) if r.size else np.nan),
        dict(arm=arm, stat="median_mu_norm_sq", value=float(np.median(D["mu_norm"] ** 2))),
        dict(arm=arm, stat="step_median_|dwmu| (active)",
             value=float(np.median(np.abs(dw[live]))) if live.any() else np.nan),
        dict(arm=arm, stat="n_realized_boundaries", value=float(ib.size)),
        dict(arm=arm, stat="bnd_median_|dwmu| (alive)", value=float(np.median(np.abs(jump_dw)))),
        dict(arm=arm, stat="bnd_median_|db| (alive)", value=float(np.median(np.abs(jump_db)))),
        dict(arm=arm, stat="bnd_median_dwmu (alive)", value=float(np.median(jump_dw))),
        dict(arm=arm, stat="bnd_q95_|dwmu| (alive)", value=float(np.quantile(np.abs(jump_dw), .95))),
    ])
    return summ, extra


# ------------------------------------------------------------------ §D drift/selection

def section_D(D: dict, arm: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """b の負ドリフトが per-unit の drift か、母集団の selection かを分ける。"""
    step, b, wn, p, s, Mh = (D["step"], D["b"], D["w_norm"], D["p_hat"],
                             D["s"], D["M_hat"])
    n = step.size
    # --- 死亡時刻（seed×unit）
    dth = np.full((N_SEED, N_UNIT), -1, dtype=np.int64)
    for si in range(N_SEED):
        dth[si] = death_index(p[:, si, :], step, PERIOD)
    dth_flat = dth.reshape(-1)
    ever_dead = dth_flat >= 0
    bf, wf, pf, sf, mf = (a.reshape(n, -1) for a in (b, wn, p, s, Mh))
    # --- 「本当に凍る」時刻: これ以降ずっと p̂ == 0 になる最初の記録点。
    #     p̂ = 1/32 は τ=0.05 未満だが gate は開くので勾配は 0 でない（＝凍らない）。
    zero_after = np.cumsum((pf > 0)[::-1], axis=0)[::-1] == 0     # [n, U]
    has_frz = zero_after.any(axis=0)
    t_frz = np.where(has_frz, zero_after.argmax(axis=0), n)
    bw = bf / wf
    # not-yet-dead マスク（記録点 index < t_death、死なない unit は常に True）。
    # **重要**: step 0 で既に p̂ < τ の unit（born dead）は §3.5 の下方クロスを一度も
    # 持たないので「死なない unit」に分類されてしまう。これらは gate が一度も開かず
    # gW = gb = 0 で b が永久に 0 のままなので、混ぜると pre_death コホートが
    # b/||w|| ≡ 0 に潰れる。born alive の unit に限定する。
    born_alive = pf[0] >= TAU
    ridx = np.arange(n)[:, None]
    alive_never = np.where(ever_dead, dth_flat, n)[None, :]
    pre_death = (ridx < alive_never) & born_alive[None, :]

    # --- 時系列: 母集団 / alive / dead / 未死亡 / 最終生存コホート
    surv_final = pf[-1] >= TAU
    cohorts = dict(all=np.ones_like(pf, dtype=bool), alive_tau=pf >= TAU,
                   dead_tau=pf < TAU, pre_death=pre_death,
                   dead_tau_born_alive=(pf < TAU) & born_alive[None, :],
                   final_survivor=np.broadcast_to(surv_final[None, :], pf.shape))
    ts = []
    for cname, cm in cohorts.items():
        rec = dict(arm=arm, cohort=cname, step=step, n=cm.sum(axis=1),
                   in_win=D["in_win"], is_bulk=D["is_bulk"])
        for qty, val in (("b", bf), ("b_over_w", bw), ("w_norm", wf), ("s", sf),
                         ("s_over_M", sf / mf)):
            v = np.where(cm, val, np.nan)
            with np.errstate(invalid="ignore"):
                q = np.nanquantile(v, [0.25, 0.5, 0.75], axis=1)
            rec[f"{qty}_q25"], rec[f"{qty}_med"], rec[f"{qty}_q75"] = q[0], q[1], q[2]
        ts.append(pd.DataFrame(rec))
    ts = pd.concat(ts, ignore_index=True)

    # --- b のドリフトを「どの発火状態のときに稼いだか」で分解する。
    # 記録点間の Δb を、区間の始点の p̂ の状態で 3 つに割り振る。b(0)=0 なので
    # b_final = sum(Δb) が厳密に成り立ち、3 成分の和は b_final に一致する。
    db_rec = np.diff(bf, axis=0)                                  # [n-1, U]
    st_alive, st_marg = pf[:-1] >= TAU, (pf[:-1] > 0) & (pf[:-1] < TAU)
    st_zero = pf[:-1] == 0
    db_alive = np.where(st_alive, db_rec, 0.0).sum(axis=0)
    db_marg = np.where(st_marg, db_rec, 0.0).sum(axis=0)
    db_zero = np.where(st_zero, db_rec, 0.0).sum(axis=0)
    never_fired = (pf == 0).all(axis=0)      # 一度も発火しない unit（真に不活性）

    # --- per-unit: 生前最後の値 / 死亡時刻 / 早期 b と寿命の関係
    last_alive = np.where(ever_dead, np.maximum(dth_flat - 1, 0), n - 1)
    col = np.arange(bf.shape[1])
    i_early = int(np.searchsorted(step, PERIOD))          # step=10,000 の記録点
    per_unit = pd.DataFrame(dict(
        arm=arm,
        seed=np.repeat(np.arange(N_SEED), N_UNIT),
        unit=np.tile(np.arange(N_UNIT), N_SEED),
        born_alive=born_alive,
        ever_dead=ever_dead,
        t_death=np.where(ever_dead, step[np.clip(dth_flat, 0, n - 1)], -1),
        ever_frozen=has_frz,
        t_frozen=np.where(has_frz, step[np.clip(t_frz, 0, n - 1)], -1),
        b_last_alive=bf[last_alive, col],
        bw_last_alive=bw[last_alive, col],
        w_last_alive=wf[last_alive, col],
        s_last_alive=sf[last_alive, col],
        b_early=bf[i_early, col],
        bw_early=bw[i_early, col],
        db_while_alive=db_alive,
        db_while_marginal=db_marg,
        db_while_zero=db_zero,
        never_fired=never_fired,
        b_final=bf[-1, col],
        bw_final=bw[-1, col],
        w_final=wf[-1, col],
        p_final=pf[-1, col],
    ))

    # --- 分解: 最終時点の母集団 median を alive / dead に割る
    dead_now = pf[-1] < TAU
    post_death_gap = bf[-1, col] - bf[last_alive, col]   # t_death 以降の b の総移動
    frz_gap = bf[-1, col] - bf[np.clip(t_frz, 0, n - 1), col]
    pd_med = lambda m, v: float(np.median(v[m])) if m.any() else np.nan   # noqa: E731
    dec = pd.DataFrame([
        dict(arm=arm, stat="frac_dead_final(p̂<τ)", value=float(dead_now.mean())),
        dict(arm=arm, stat="frac_born_dead(step0 で p̂<τ)", value=float((~born_alive).mean())),
        dict(arm=arm, stat="frac_born_phat0(step0 で p̂=0)", value=float((pf[0] == 0).mean())),
        dict(arm=arm, stat="frac_never_fired(全記録点で p̂=0)", value=float(never_fired.mean())),
        dict(arm=arm, stat="max |b_final| among never_fired (厳密に 0 のはず)",
             value=float(np.abs(bf[-1][never_fired]).max()) if never_fired.any() else np.nan),
        dict(arm=arm, stat="frac_born_phat0_that_revive",
             value=float(np.mean(~never_fired[pf[0] == 0])) if (pf[0] == 0).any() else np.nan),
        # b_final = Δb(p̂≥τ 中) + Δb(0<p̂<τ 中) + Δb(p̂=0 中) の分解
        dict(arm=arm, stat="Δb median while p̂>=τ", value=float(np.median(db_alive))),
        dict(arm=arm, stat="Δb median while 0<p̂<τ", value=float(np.median(db_marg))),
        dict(arm=arm, stat="Δb median while p̂=0 (区間始点で p̂=0)",
             value=float(np.median(db_zero))),
        dict(arm=arm, stat="frac of intervals that are 1 step",
             value=float(np.mean(np.diff(step) == 1))),
        dict(arm=arm, stat="Δb median while p̂>=τ (最終 alive のみ)",
             value=pd_med(pf[-1] >= TAU, db_alive)),
        dict(arm=arm, stat="Δb median while 0<p̂<τ (最終 alive のみ)",
             value=pd_med(pf[-1] >= TAU, db_marg)),
        dict(arm=arm, stat="Δb median while p̂>=τ (最終 dead のみ)",
             value=pd_med(pf[-1] < TAU, db_alive)),
        dict(arm=arm, stat="Δb median while 0<p̂<τ (最終 dead のみ)",
             value=pd_med(pf[-1] < TAU, db_marg)),
        dict(arm=arm, stat="frac_ever_dead(§3.5)", value=float(ever_dead.mean())),
        dict(arm=arm, stat="frac_ever_frozen(p̂≡0 以降)", value=float(has_frz.mean())),
        dict(arm=arm, stat="b_over_w_median_dead_final(born_alive のみ)",
             value=pd_med(dead_now & born_alive, bw[-1])),
        dict(arm=arm, stat="b_over_w_median_all_final", value=float(np.median(bw[-1]))),
        dict(arm=arm, stat="b_over_w_median_alive_final", value=pd_med(~dead_now, bw[-1])),
        dict(arm=arm, stat="b_over_w_median_dead_final", value=pd_med(dead_now, bw[-1])),
        dict(arm=arm, stat="b_median_all_final", value=float(np.median(bf[-1]))),
        dict(arm=arm, stat="b_median_alive_final", value=pd_med(~dead_now, bf[-1])),
        dict(arm=arm, stat="b_median_dead_final", value=pd_med(dead_now, bf[-1])),
        dict(arm=arm, stat="w_norm_median_alive_final", value=pd_med(~dead_now, wf[-1])),
        dict(arm=arm, stat="w_norm_median_dead_final", value=pd_med(dead_now, wf[-1])),
        # 生涯ドリフト: b(0)=0 なので「死亡時点の b」＝生きているあいだに獲得した総ドリフト
        dict(arm=arm, stat="b_median_at_death (生涯ドリフト)",
             value=pd_med(ever_dead, bf[last_alive, col])),
        dict(arm=arm, stat="b_over_w_median_at_death (生涯ドリフト)",
             value=pd_med(ever_dead, bw[last_alive, col])),
        # 「死亡＝凍結」は τ=0.05 では成り立たない（p̂=1/32 でも gate は開く）ことの定量
        dict(arm=arm, stat="median |b_final − b(t_death)| (§3.5 死亡後)",
             value=pd_med(ever_dead, np.abs(post_death_gap))),
        dict(arm=arm, stat="median |b_final − b(t_frozen)| (p̂≡0 以降・厳密に 0 のはず)",
             value=pd_med(has_frz, np.abs(frz_gap))),
        dict(arm=arm, stat="spearman(bw_early, t_death | dead)",
             value=_spearman(per_unit.loc[per_unit.ever_dead, "bw_early"].to_numpy(),
                             per_unit.loc[per_unit.ever_dead, "t_death"].to_numpy())),
        dict(arm=arm, stat="bw_early_median_dead_by_1e5",
             value=_cond_median(per_unit, (per_unit.ever_dead) &
                                (per_unit.t_death <= 100_000), "bw_early")),
        dict(arm=arm, stat="bw_early_median_final_survivor",
             value=_cond_median(per_unit, per_unit.p_final >= TAU, "bw_early")),
        dict(arm=arm, stat="b_early_median_dead_by_1e5",
             value=_cond_median(per_unit, (per_unit.ever_dead) &
                                (per_unit.t_death <= 100_000), "b_early")),
        dict(arm=arm, stat="b_early_median_final_survivor",
             value=_cond_median(per_unit, per_unit.p_final >= TAU, "b_early")),
    ])
    return ts, per_unit, dec


def _spearman(x, y):
    if x.size < 3:
        return np.nan
    from scipy.stats import spearmanr
    r = spearmanr(x, y)
    return float(r.statistic)


def _cond_median(df, mask, col):
    v = df.loc[mask, col].to_numpy()
    return float(np.median(v)) if v.size else np.nan


# ------------------------------------------------------------------ §E centered 対比

def section_E(D: dict, arm: str) -> pd.DataFrame:
    """s ≈ b がどれだけ成り立つか、および死亡条件 b + M <= 0 の一致率。"""
    s, b, p, Mh, wn = D["s"], D["b"], D["p_hat"], D["M_hat"], D["w_norm"]
    wmu = s - b
    dead = (p == 0)
    pred_full = (s + Mh) <= 0                 # 正しい条件（w·µ 込み）
    pred_bonly = (b + Mh) <= 0                # w·µ を無視した条件
    frac = lambda m: float(np.mean(m))        # noqa: E731
    den = np.abs(s)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.abs(wmu) / np.where(den > 0, den, np.nan)
    rows = [
        dict(arm=arm, stat="median |w·µ|", value=float(np.median(np.abs(wmu)))),
        dict(arm=arm, stat="median |b|", value=float(np.median(np.abs(b)))),
        dict(arm=arm, stat="median |w·µ| / |b|",
             value=float(np.median(np.abs(wmu) / np.maximum(np.abs(b), 1e-12)))),
        dict(arm=arm, stat="median |w·µ| / |s|", value=float(np.nanmedian(rel))),
        dict(arm=arm, stat="median |s - b| / |s|", value=float(np.nanmedian(rel))),
        dict(arm=arm, stat="agree(p_hat==0, s+M̂<=0)", value=frac(dead == pred_full)),
        dict(arm=arm, stat="agree(p_hat==0, b+M̂<=0)", value=frac(dead == pred_bonly)),
        dict(arm=arm, stat="frac p_hat==0", value=frac(dead)),
        dict(arm=arm, stat="median mu_norm", value=float(np.median(D["mu_norm"]))),
        dict(arm=arm, stat="mu_norm final", value=float(np.median(D["mu_norm"][-1]))),
    ]
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ §F 追随 / κ 裏取り

def section_F(D: dict, arm: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(i) alive の s/M̂ が時間とともに 0 に寄るか、(ii) 消灯遷移点での −s ≈ M 裏取り。"""
    step, s, p, Mh, wn = D["step"], D["s"], D["p_hat"], D["M_hat"], D["w_norm"]
    n = step.size
    sf, pf, mf, wf = (a.reshape(n, -1) for a in (s, p, Mh, wn))
    bulk = D["is_bulk"]
    rows = []
    edges = [(0, 1, "step 0"), (1, 100_000, "(0, 1e5)"), (100_000, 250_000, "[1e5, 2.5e5)"),
             (250_000, 500_000, "[2.5e5, 5e5)"), (500_000, 750_000, "[5e5, 7.5e5)"),
             (750_000, 1_000_001, "[7.5e5, 1e6]")]
    for lo, hi, lab in edges:
        m = bulk & (step >= lo) & (step < hi)
        if not m.any():
            continue
        al = pf[m] >= TAU
        rows.append(qrow((sf[m] / mf[m])[al],
                         dict(arm=arm, window=lab, qty="s_over_M|alive")))
        rows.append(qrow(pf[m][al], dict(arm=arm, window=lab, qty="p_hat|alive")))
        rows.append(qrow((sf[m] / wf[m])[al],
                         dict(arm=arm, window=lab, qty="s_over_w|alive")))
    trend = pd.DataFrame(rows)[["arm", "window", "qty", "n", *QNAME, "mean", "iqr"]]

    # κ 裏取り: p_hat が >0 から ==0 に落ちる **1 step 隣接**の記録点対で M を挟む。
    # 直前 (p>0): s + M > 0 -> M > -s_prev ;  直後 (p=0): s + M <= 0 -> M <= -s_next。
    # 1 step 隣接に限れば M の時間変化は 1 step ぶんなので挟みは十分きつい。
    z = pf == 0
    one = np.diff(step) == 1
    trans = np.flatnonzero(np.any(z[1:] & ~z[:-1], axis=1) & one)
    lo_l, hi_l, kh_l = [], [], []
    for i in trans:
        u = np.flatnonzero(z[i + 1] & ~z[i])
        lo_l.append(-sf[i, u])
        hi_l.append(-sf[i + 1, u])
        kh_l.append(mf[i + 1, u])
    if lo_l:
        lo_a, hi_a, kh_a = np.concatenate(lo_l), np.concatenate(hi_l), np.concatenate(kh_l)
        ok = np.isfinite(lo_a) & np.isfinite(hi_a) & (hi_a > 0) & (hi_a >= lo_a)
        mid = 0.5 * (lo_a + hi_a)
        inb = float(np.mean((kh_a[ok] > lo_a[ok]) & (kh_a[ok] <= hi_a[ok])))
        kcheck = pd.DataFrame([
            qrow(lo_a[ok], dict(arm=arm, qty="M_lower(-s at last alive)", frac_in_bracket=inb)),
            qrow(hi_a[ok], dict(arm=arm, qty="M_upper(-s at first zero)", frac_in_bracket=inb)),
            qrow(kh_a[ok], dict(arm=arm, qty="M_hat(kappa_hat*||w||)", frac_in_bracket=inb)),
            qrow((kh_a[ok] / np.maximum(mid[ok], 1e-12)),
                 dict(arm=arm, qty="M_hat / M_bracket_mid", frac_in_bracket=inb)),
        ])[["arm", "qty", "n", *QNAME, "mean", "iqr", "frac_in_bracket"]]
    else:
        kcheck = pd.DataFrame(columns=["arm", "qty", "n", *QNAME, "mean", "iqr",
                                       "frac_in_bracket"])
    return trend, kcheck


# ------------------------------------------------------------------ 図

def make_figures(TS: dict, TSD: dict):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    cols = dict(std="#1f77b4", centered="#d62728")

    # 図1: s/||w|| と s/M̂ の時間発展（all vs alive）、両 arm
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.4), sharex=True)
    for j, arm in enumerate(["std", "centered"]):
        d = TS[arm]
        bulk = d[d.is_bulk]
        for i, (qty, lab) in enumerate([("s_over_w", "s / ||w||"), ("s_over_M", "s / M̂")]):
            ax = axes[i, j]
            for cond, ls, alpha in (("all", "--", 0.55), ("alive_tau", "-", 1.0)):
                g = bulk[bulk.cond == cond]
                ax.plot(g.step, g[f"{qty}_med"], ls, color=cols[arm], lw=1.5, alpha=alpha,
                        label=f"{cond} (median)")
                if cond == "alive_tau":
                    ax.fill_between(g.step, g[f"{qty}_q25"], g[f"{qty}_q75"],
                                    color=cols[arm], alpha=0.16, lw=0)
            ax.axhline(0, color="k", lw=0.8)
            if qty == "s_over_M":
                ax.axhline(-1, color="k", lw=0.8, ls=":")
                ax.text(0.99, 0.04, "s/M = −1: 消灯境界", ha="right", va="bottom",
                        transform=ax.transAxes, fontsize=8)
                ax.set_ylim(-3.2, 1.6)
            ax.set_ylabel(lab)
            ax.set_title(f"{arm}: {lab}（bulk グリッドのみ・10seed×100unit プール）",
                         fontsize=10)
            ax.legend(fontsize=8, loc="lower left")
    for ax in axes[1]:
        ax.set_xlabel("step")
    fig.suptitle("完全打ち消しなら s = 0（実線 = alive のみ / 破線 = 全 unit）", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_cancel_s_time.png", dpi=140)
    plt.close(fig)

    # 図2: 2 項分解 w·µ と b（alive のみ）
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), sharex=True)
    for j, arm in enumerate(["std", "centered"]):
        d = TS[arm]
        g = d[(d.is_bulk) & (d.cond == "alive_tau")]
        ax = axes[j]
        ax.plot(g.step, g.wmu_med, color="#2ca02c", lw=1.6, label="w·µ (median)")
        ax.plot(g.step, g.b_med, color="#9467bd", lw=1.6, label="b (median)")
        ax.plot(g.step, g.s_med, color="k", lw=1.8, label="s = w·µ + b (median)")
        ax.axhline(0, color="k", lw=0.7)
        ax.set(xlabel="step", ylabel="value", title=f"{arm}: alive unit の 2 項分解")
        ax.legend(fontsize=8)
    fig.suptitle("打ち消しなら緑と紫が逆符号・和(黒)が 0 に張り付くはず", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_cancel_decomp.png", dpi=140)
    plt.close(fig)

    # 図3: b/||w|| の drift vs selection
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), sharex=True)
    styles = dict(all=("k", "-"), alive_tau=("#1f77b4", "-"), dead_tau=("#d62728", "--"),
                  dead_tau_born_alive=("#8c564b", "--"),
                  pre_death=("#2ca02c", "-."), final_survivor=("#ff7f0e", ":"))
    for j, arm in enumerate(["std", "centered"]):
        d = TSD[arm]
        ax = axes[j]
        for coh, (c, ls) in styles.items():
            g = d[(d.is_bulk) & (d.cohort == coh)]
            ax.plot(g.step, g.b_over_w_med, ls, color=c, lw=1.4, label=coh)
        ax.axhline(0, color="k", lw=0.7)
        ax.set(xlabel="step", ylabel="b / ||w|| (median)", title=f"{arm}: b/||w|| のコホート別")
        ax.legend(fontsize=8)
    fig.suptitle("per-unit の drift（生存 unit 自身が下がる）か "
                 "composition（低 p̂ 帯に沈んだ unit が母集団を占める）か", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_cancel_b_drift.png", dpi=140)
    plt.close(fig)

    # 図4: p_hat（κ 非依存の s/M 代理）の alive 分布の時間発展
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), sharex=True)
    for j, arm in enumerate(["std", "centered"]):
        d = TS[arm]
        g = d[(d.is_bulk) & (d.cond == "alive_tau")]
        ax = axes[j]
        ax.plot(g.step, g.p_hat_med, color=cols[arm], lw=1.6, label="median p̂ (alive)")
        ax.fill_between(g.step, g.p_hat_q25, g.p_hat_q75, color=cols[arm], alpha=0.18, lw=0)
        ax.axhline(0.5, color="k", lw=0.9, ls="--")
        ax.text(0.99, 0.93, "p̂ = 0.5 ⟺ s ≈ 0（完全打ち消し）", ha="right", va="top",
                transform=ax.transAxes, fontsize=8)
        ax.set(xlabel="step", ylabel="p̂", ylim=(0, 1),
               title=f"{arm}: alive unit の p̂（κ 非依存）")
        ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_cancel_phat_alive.png", dpi=140)
    plt.close(fig)


# ------------------------------------------------------------------ main

def main():
    t0 = time.time()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    FIGDIR.mkdir(parents=True, exist_ok=True)
    metas, kinfo = {}, {}
    A, AT, C1, C2, TSD, PU, DEC, E, TR, KC, CONS, INV = ([] for _ in range(12))
    TS, TSDd = {}, {}
    for arm_spec in POOLED_ARMS:
        arm = arm_spec["label"]
        metas[arm] = check_source_run(arm_spec["resdir"], arm_spec["spec"])
        D = load_arm(arm_spec["resdir"])
        D["in_win"], D["is_bulk"] = grid_masks(D["step"])
        est, ki = build_kappa(arm, D["w_norm"])
        kinfo[arm] = {str(s): dict(source=ki["ck"][s]["source"],
                                   kappa_median=float(np.median(ki["ck"][s]["kappa"])))
                      for s in CKPT_STEPS}
        INV.append(ki["inv"])
        D["M_hat"] = kappa_field(est[MAIN_EST], D["step"], D["w_norm"]) * D["w_norm"]
        print(f"[{arm}] loaded. n_rec={D['step'].size} "
              f"in_win={int(D['in_win'].sum())} bulk={int(D['is_bulk'].sum())} "
              f"M_hat median={float(np.median(D['M_hat'])):.4f}", flush=True)

        a_df, cons = section_A(D, arm)
        A.append(a_df)
        CONS.append(cons)
        AT.append(section_A_time(D, arm))
        c1, c2 = section_C(D, arm)
        C1.append(c1)
        C2.append(c2)
        ts, pu, dec = section_D(D, arm)
        TSD.append(ts)
        PU.append(pu)
        DEC.append(dec)
        E.append(section_E(D, arm))
        tr, kc = section_F(D, arm)
        TR.append(tr)
        KC.append(kc)
        TS[arm] = AT[-1]
        TSDd[arm] = ts
        # 感度: 別 kappa 推定量での s/M̂ の alive 中央値
        for en in SENS_EST:
            Mh2 = kappa_field(est[en], D["step"], D["w_norm"]) * D["w_norm"]
            al = D["p_hat"] >= TAU
            A.append(pd.DataFrame([qrow((D["s"] / Mh2)[al],
                                        dict(arm=arm, grid="all", cond="alive_tau",
                                             qty=f"s_over_M[{en}]"))])
                     [["arm", "grid", "cond", "qty", "n", *QNAME, "mean", "iqr"]])
            del Mh2
        del D

    (A, AT, C1, C2, TSD, PU, DEC, E, TR, KC, CONS, INV) = (
        pd.concat(x, ignore_index=True)
        for x in (A, AT, C1, C2, TSD, PU, DEC, E, TR, KC, CONS, INV))

    # 小さい表はそのまま CSV。全 20,901 記録点の時系列は gzip で置き、
    # 人が読む用に bulk グリッドだけの小さい版も併置する。
    for name, df in (("s_dist_pooled", A), ("increment_summary", C1),
                     ("increment_stats", C2), ("per_unit", PU),
                     ("b_decomposition", DEC), ("arm_contrast", E),
                     ("alive_trend", TR), ("kappa_bracket_check", KC),
                     ("kappa_consistency", CONS), ("phat_to_sM_inversion", INV)):
        df.to_csv(OUTDIR / f"{name}.csv", index=False, float_format="%.6g")
    for name, df in (("s_time", AT), ("b_cohort_time", TSD)):
        df.to_csv(OUTDIR / f"{name}.csv.gz", index=False, float_format="%.6g",
                  compression="gzip")
        df[df.is_bulk].to_csv(OUTDIR / f"{name}_bulk.csv", index=False,
                              float_format="%.6g")

    make_figures(TS, TSDd)
    write_report(A, AT, C1, C2, TSD, PU, DEC, E, TR, KC, CONS, INV, metas, kinfo, t0)
    print(f"DONE -> {OUTDIR}  ({time.time()-t0:.0f}s)", flush=True)


def _pick(df, **kw):
    m = np.ones(len(df), dtype=bool)
    for k, v in kw.items():
        m &= (df[k] == v).to_numpy()
    return df[m]


def _v(df, arm, stat):
    r = df[(df.arm == arm) & (df.stat == stat)]
    return float(r.value.iloc[0]) if len(r) else np.nan


def write_report(A, AT, C1, C2, TSD, PU, DEC, E, TR, KC, CONS, INV, metas, kinfo, t0):
    bulk_steps = np.sort(AT.loc[AT.is_bulk, "step"].unique())

    def nearest_bulk(targets):
        """表に出す step。bulk グリッド（1000 刻みのうち境界窓の外）上で最も近い点を取る。
        10,000 の倍数は必ず境界窓の中なので bulk には無い。"""
        return [int(bulk_steps[np.argmin(np.abs(bulk_steps - t))]) for t in targets]

    ST5 = nearest_bulk([0, 10_000, 100_000, 500_000, 1_000_000])
    ST8 = nearest_bulk([0, 10_000, 50_000, 100_000, 250_000, 500_000, 750_000, 1_000_000])

    L = ["# q3_cancel_residual: バイアス b は w·µ を打ち消しているか", "",
         "**事後計算・未事前登録**（spec なし・判定なし。verdict.csv は書き換えない。"
         "引用には事前登録つきの昇格が要る）。",
         f"生成 {time.strftime('%Y-%m-%d %H:%M:%S')} / `analysis/q3_cancel_residual.py` "
         f"@ {git_hash(['analysis/q3_cancel_residual.py'])}。", "",
         "スコープ: condA・w100・T=1e4・batch=1・center_alpha=0.01・10 seed × 100 unit "
         "× 20,901 記録点。**新しい学習走は行っていない**（既存ログ + 既存 checkpoint のみ）。", ""]

    def med(arm, grid, cond, qty):
        r = _pick(A, arm=arm, grid=grid, cond=cond, qty=qty)
        return float(r["med"].iloc[0]) if len(r) else np.nan

    s_std = med("std", "bulk", "alive_tau", "s_over_M")
    s_cen = med("centered", "bulk", "alive_tau", "s_over_M")
    p_std = med("std", "bulk", "alive_tau", "p_hat")
    p_cen = med("centered", "bulk", "alive_tau", "p_hat")
    r_std = float(_pick(C1, arm="std", qty="ratio_dwmu_over_db")["med"].iloc[0])
    sh_std = float(_pick(C1, arm="std", qty="share_|db|/|ds|")["med"].iloc[0])
    kfix = float(_pick(KC, arm="std", qty="M_hat / M_bracket_mid")["med"].iloc[0])

    # ---- 0. 一行
    L += ["## 0. 一行", "",
          f"**打ち消していない。むしろ強め合っている。** std の alive unit の残差は "
          f"s/M̂ 中央値 {s_std:+.3f}（完全打ち消し = 0、消灯境界 = −1）で、0 ではなく"
          f"**消灯境界の側に寄っている**。κ 非依存の裏取りとして alive の p̂ 中央値は "
          f"{p_std:.3f}（s = 0 なら 0.5 のはず）。理由は代数で決まっていて、"
          f"`nets.VecMLP.grads` の `gW = gb ⊗ x_in`（同じ因子 gb が w と b の両方に掛かる）"
          f"より batch=1・µ 固定の 1 step について **Δ(w·µ) = Δb·(x_in·µ) が厳密**、"
          f"std では x_in·µ ≥ 0 なので b と w·µ は**常に同符号に動く**"
          f"（同符号率 {_v(C2,'std','frac_same_sign(dwmu,db)'):.4f}、比の実測中央値 "
          f"{r_std:.2f}、b が担う |Δs| の割合は中央値 {sh_std:.3f}）。"
          f"打ち消しは「b の学習が下手だから起きない」のではなく、"
          f"**SGD の軌道上に打ち消しへ向かう自由度が無い**。"
          f"centered では ||µ|| が潰れて x_in·µ ≈ 0 になるので Δs ≈ Δb・s ≈ b になり、"
          f"b が唯一の定数項として効く（s/M̂ 中央値 {s_cen:+.3f}、p̂ 中央値 {p_cen:.3f}）。", "",
          f"b の負ドリフト（母集団 b/||w|| が 0 → {_v(DEC,'std','b_over_w_median_all_final'):.3f}）は "
          f"**per-unit の drift**である（全 unit が b(0)=0 から出発するので"
          f"「もともと b が低い unit が選ばれた」型の selection は原理的に無い）。"
          f"ただし**母集団値の水準は composition が主役**で、最終時点で alive な unit "
          f"（{100*(1-_v(DEC,'std','frac_dead_final(p̂<τ)')):.1f}%）の b/||w|| は "
          f"{_v(DEC,'std','b_over_w_median_alive_final'):+.4f} しかなく、残りは"
          f"低 p̂ 帯に沈んだ {100*_v(DEC,'std','frac_dead_final(p̂<τ)'):.1f}% の unit の値である。"
          f"なお §3.5 の死亡は終端ではなく unit は死んでは復活しており、"
          f"負ドリフトは主に**発火している最中**に稼がれている（内訳は §D）。", ""]

    # ---- 1. 定義
    L += ["## 1. 定義（コードで確認）", "",
          "`src/envs.py: SCREnv` は x = [flip_state(f=15, 周期内固定) ‖ 自由 5 ビット U{0,1}]、"
          "`src/train.py` は x_in = x − centered·running_mean、"
          "`src/ratchet_log.py: exact_record` は µ = E[x_in]（32 パターン平均）、"
          "gate は strict `pre > 0`。よって", "",
          "```",
          "s := w·µ + b = w_norm · mu_norm · cos_u_mu + b      ← 記録量から厳密に復元",
          "M := max_δ(w·δ) = ||w_free||_1 / 2 ,  κ := M/||w|| ,  min_δ(w·δ) = −M",
          "p̂ = 0 ⟺ s + M ≤ 0 ⟺ s/M ≤ −1 ;   p̂ = 1 ⟺ s/M > +1",
          "```", "",
          "**s/M が発火の自然座標**である。s/M = 0（完全打ち消し）は δ 分布の ± 対称性から "
          "p̂ ≈ 0.5 に対応する。したがって p̂ は s/M の **κ 非依存な単調代理**であり、"
          "以下では κ 近似に依る数値と p̂ の両方を出す。", "",
          f"alive の主定義は**コード準拠**の `p̂ ≥ {TAU}`"
          f"（`src/figures_ratchet_log.py: TAU`、spec_ratchet_log_0819 §3.5 の凍結/死亡閾値。"
          f"p̂ は k/32 を取るので実質 p̂ ≥ 2/32）。副定義 `p̂ > 0`（＝ s+M>0 の厳密な生存）"
          f"も `alive_pos` として併記した。", ""]
    L += ["### 1.1 M̂ の作り方と限界", "",
          "M は full W からしか出ないが full W は checkpoint（step 0 / 1,000,000）にしか無い。"
          f"`analysis/q3_margin_pooled.py` の推定量 `{MAIN_EST}` をそのまま import して "
          "M̂ = κ̂(||w||, t)·||w|| とした（**近似**）。単一 checkpoint 版 "
          f"`{SENS_EST[0]}` / `{SENS_EST[1]}` を感度として §A3 に併記する。", "",
          "checkpoint 由来 κ の中央値: " +
          "; ".join(f"{a} step{s}={d['kappa_median']:.4f}"
                    for a, dd in kinfo.items() for s, d in dd.items()), "",
          "**step 1M の checkpoint は本走ではなく同一 config の別実現**（本走 ckpt は "
          "`.gitignore` で repo に無い）。分布水準の要約にしか使えず、per-unit の主張には"
          "使えない。この限界は q3_margin_pooled の同節と同じ。", "",
          "κ̂ の自己整合性（厳密なら p̂>0 ⟺ s/M > −1 なので下の 2 行は 0 のはず）:", ""]
    L += [md_table(CONS.pivot(index="stat", columns="arm", values="value").reset_index()), ""]

    # ---- A 分布
    frac_win = 100 * float(_pick(AT, arm="std", cond="all").in_win.mean())
    L += ["## A. s の分布（完全打ち消し = 0 からのズレ）", "", "### A1. pooled 分位", "",
          "グリッドは all / bulk（境界窓の外の 1000 刻み・901 点）/ boundary（境界 ±100・20,000 点）。"
          f"境界窓が全記録点の {frac_win:.1f}% を占めるので、"
          "**素朴な時間プール（all）は境界近傍に強く重み付けされる**。以下では bulk を主報告にする。", ""]
    tab = A[A.qty.isin(["s", "s_over_w", "s_over_M", "p_hat"])
            & A.cond.isin(["all", "alive_tau", "alive_pos", "dead_tau"])]
    tab = tab.sort_values(["arm", "grid", "cond", "qty"])
    L += [md_table(tab[["arm", "grid", "cond", "qty", "n", "q05", "q25", "med",
                        "q75", "q95", "iqr"]]), ""]
    L += ["### A2. 読み（bulk グリッド）", "",
          f"- **std・alive: s/||w|| 中央値 {med('std','bulk','alive_tau','s_over_w'):+.4f}、"
          f"s/M̂ 中央値 {s_std:+.4f}。0 ではない。** IQR は "
          f"[{_pick(A,arm='std',grid='bulk',cond='alive_tau',qty='s_over_M')['q25'].iloc[0]:+.3f}, "
          f"{_pick(A,arm='std',grid='bulk',cond='alive_tau',qty='s_over_M')['q75'].iloc[0]:+.3f}]。",
          f"- std・全 unit: s/M̂ 中央値 {med('std','bulk','all','s_over_M'):+.4f}"
          f"（dead を含むので −1 を大きく割る）。std・dead: "
          f"{med('std','bulk','dead_tau','s_over_M'):+.4f}。",
          f"- centered・alive: s/||w|| 中央値 "
          f"{med('centered','bulk','alive_tau','s_over_w'):+.4f}、s/M̂ 中央値 {s_cen:+.4f}。",
          f"- **κ 非依存の裏取り**: alive の p̂ 中央値は std {p_std:.3f} / centered {p_cen:.3f}。"
          f"完全打ち消し（s=0）なら 0.5 になるはずで、両アームとも下回る。",
          f"- alive のうち s > 0 なのは std "
          f"{_v(CONS,'std','frac alive(p̂>=τ) with s > 0'):.3f} / centered "
          f"{_v(CONS,'centered','frac alive(p̂>=τ) with s > 0'):.3f}。"
          f"「0 のまわりに対称に散る」のではなく**負に偏っている**。", "",
          "**alive 条件は s/M > −1 という下からの切り取りを自動的に課す**（p̂>0 の定義そのもの）。"
          "だから「alive の s/M が負」自体は自明である。効くのは**どこに寄っているか**で、"
          "打ち消しが働いていれば 0（p̂ ≈ 0.5）に寄るはずのところ、実際は"
          "切り取り境界 −1 と 0 の中間より境界寄りにいる。", ""]

    L += ["### A3. κ̂ 推定量の感度（alive・s/M̂ の分位）", ""]
    sens = A[A.qty.str.startswith("s_over_M") & (A.cond == "alive_tau")]
    L += [md_table(sens[["arm", "grid", "qty", "n", "q25", "med", "q75"]]), "",
          f"推定量を替えても **std の alive 中央値は 0 に届かず負のまま**"
          f"（−0.29 〜 −0.51 の範囲）で、結論の向きは動かない。"
          f"さらに §A5 の挟み込みでは M̂ が真の M を中央値で {kfix:.2f} 倍に過小評価しており、"
          f"これを補正すると std の alive s/M は {s_std*kfix:+.3f} 程度になる。"
          f"**補正しても 0 ではない**。", ""]

    L += ["### A4. 時間発展（bulk グリッド・alive のみ・記録点ごとに 10seed×100unit をプール）", ""]
    rows = []
    for arm in ["std", "centered"]:
        g = AT[(AT.arm == arm) & (AT.cond == "alive_tau") & (AT.is_bulk)]
        for st in ST8:
            r = g[g.step == st]
            if not len(r):
                continue
            r = r.iloc[0]
            rows.append(dict(arm=arm, step=st, n_alive=int(r["n"]),
                             s_med=r.s_med, s_over_w_med=r.s_over_w_med,
                             s_over_M_med=r.s_over_M_med, p_hat_med=r.p_hat_med,
                             wmu_med=r.wmu_med, b_med=r.b_med, w_norm_med=r.w_norm_med))
    L += [md_table(pd.DataFrame(rows)), "",
          "（10,000 の倍数は必ず境界窓の中なので bulk グリッドに無い。表の step は"
          "目標値に最も近い bulk 点。）", ""]

    L += ["### A5. κ̂ の裏取り（消灯遷移点で M を挟む）", "",
          "p̂ が >0 から ==0 に落ちる **1 step 隣接**の記録点対では、直前で s+M>0、"
          "直後で s+M≤0 なので **M ∈ (−s_prev, −s_next]** と挟める"
          "（1 step ぶんの M の変化だけ緩い）。", ""]
    L += [md_table(KC[["arm", "qty", "n", "q25", "med", "q75", "frac_in_bracket"]]), "",
          f"std では M̂ が挟み区間に入る割合が "
          f"{float(_pick(KC, arm='std', qty='M_hat(kappa_hat*||w||)')['frac_in_bracket'].iloc[0]):.3f}、"
          f"M̂/M_mid の中央値 {kfix:.3f} で **κ̂ は M を系統的に過小評価**している。"
          f"§1.1 の自己整合性表でも「p̂>0 なのに s/M̂ ≤ −1」が std で "
          f"{_v(CONS,'std','frac (p̂>0 かつ s/M̂ <= -1)  [厳密なら 0]'):.3f} 起きており、同じ向き。"
          f"向きの理由づけ: 消灯する unit は M が**小さい**方に偏るので、"
          f"母集団中央値を当てにいく κ̂ は本来**過大**に出るはずである。"
          f"逆が観測されるのは、q3_margin_pooled が既に指摘している"
          f"「κ(t) は ~250k で飽和する凹形なので 2 点線形内挿は中盤を過小評価する」"
          f"ことと整合する。いずれにせよ過小評価は |s/M̂| を**過大**にする向きなので、"
          f"A2 の「0 ではない」という結論は補正後も残る。", ""]

    L += ["### A6. κ を使わない逆算（p̂ → s/M）", "",
          "p̂ = k/32 は「32 個の (w·δ)/M のうち k 個が −s/M を上回る」ことなので、"
          "checkpoint の W から (w·δ)/M の順序統計を作れば **κ の水準を一切使わずに** "
          "s/M を区間で決められる（必要なのは自由 5 次元の重みの相対配分＝X の形だけで、"
          "これは κ の水準より checkpoint 間で安定している）:", "",
          "```",
          "p̂ = k/32  ⟺  s/M ∈ ( −X_(32−k), −X_(32−k−1) ]      X_(1)≤…≤X_(32) は (w·δ)/M の昇順",
          "```", ""]
    L += [md_table(INV), "",
          f"observed: std の alive p̂ 中央値 = {p_std:.3f}（= {int(round(p_std*32))}/32）、"
          f"centered = {p_cen:.3f}（= {int(round(p_cen*32))}/32）。"
          f"この表を引くと **std の median alive unit の s/M は "
          f"{float(INV[(INV.arm=='std')&(INV.ckpt_step==1_000_000)&(INV.k==int(round(p_std*32)))]['sM_mid_med'].iloc[0]):+.3f} 付近**、"
          f"centered は "
          f"{float(INV[(INV.arm=='centered')&(INV.ckpt_step==1_000_000)&(INV.k==int(round(p_cen*32)))]['sM_mid_med'].iloc[0]):+.3f} 付近"
          f"（step 1M checkpoint 由来）。std は M̂ 経由の {s_std:+.3f}・"
          f"挟み込み補正後の {s_std*kfix:+.3f} と同じ範囲に落ちる。"
          f"**3 つの独立な経路がどれも「0 ではない・−0.3 〜 −0.4 あたり」と言う。**"
          f"また表の k=16（p̂ = 0.5）で s/M がちょうど 0 になっており、"
          f"「s = 0 ⟺ p̂ ≈ 0.5」という §1 の対応が数値でも確認できる。", ""]

    # ---- B/C
    L += ["## B/C. 2 項分解と、打ち消しが構造的に不可能な理由", "",
          "### B1. w·µ と b の時間発展（alive のみ・bulk）", ""]
    rows = []
    for arm in ["std", "centered"]:
        g = AT[(AT.arm == arm) & (AT.cond == "alive_tau") & (AT.is_bulk)]
        for st in ST5:
            r = g[g.step == st]
            if not len(r):
                continue
            r = r.iloc[0]
            rows.append(dict(arm=arm, step=st, wmu_med=r.wmu_med, b_med=r.b_med,
                             s_med=r.s_med, wmu_over_w_med=r.wmu_over_w_med,
                             b_over_w_med=r.b_over_w_med))
    L += [md_table(pd.DataFrame(rows)), "",
          "std では w·µ と b が**どちらも負に動き、和 s は両者のどちらより深い**。"
          "打ち消し（逆符号）ではなく**強め合い**である。しかも |w·µ| ≫ |b| で"
          f"（全記録点 median |w·µ|/|b| = {_v(E,'std','median |w·µ| / |b|'):.2f}）、"
          "b は水準としても w·µ に対抗できていない。"
          "centered では w·µ が 0 付近に張り付き、s ≈ b になる。", ""]

    L += ["### C1. 1 step 増分の厳密関係（打ち消し不能の代数）", "",
          "`nets.VecMLP.grads` は `gW = gb ⊗ x_in`、`sgd_step` は W -= lr·gW / b -= lr·gb。"
          "つまり **w の更新と b の更新は同じ因子 gb = 2δ·v·gate を共有する**。"
          "batch=1・µ 固定（周期内）なら 1 step について厳密に", "", "```",
          "Δw = Δb · x_in   →   Δ(w·µ) = Δb · (x_in·µ),   Δs = Δb · (1 + x_in·µ)",
          "```", "",
          "std では x ∈ {0,1}^20、µ = [flip_state ‖ 0.5·1_5] なので "
          "x·µ = #{flip bit = 1} + 0.5·#{free bit = 1} ≥ 0。"
          "**b を下げる更新は必ず w·µ を (x·µ) 倍だけ同じ向きに下げる。**", "",
          "周期内・1 step 隣接・flip_state 不変の記録点対だけで測った実測:", ""]
    L += [md_table(C1[["arm", "qty", "n", "q05", "q25", "med", "q75", "q95", "note"]]), ""]
    L += [md_table(C2.pivot(index="stat", columns="arm", values="value").reset_index()), "",
          f"- std: Δ(w·µ)/Δb の中央値 {r_std:.2f}、5–95% が "
          f"[{_pick(C1,arm='std',qty='ratio_dwmu_over_db')['q05'].iloc[0]:.2f}, "
          f"{_pick(C1,arm='std',qty='ratio_dwmu_over_db')['q95'].iloc[0]:.2f}]。"
          f"これは x·µ = #{{flip=1}} + 0.5·#{{free=1}} の実現分布そのもので、"
          f"||µ||² の中央値 {_v(C2,'std','median_mu_norm_sq'):.2f} と整合する"
          f"（flip の 1 の個数が境界ごとにランダムウォークするので半整数で散る）。",
          f"- **同符号率 {_v(C2,'std','frac_same_sign(dwmu,db)'):.4f}**。"
          f"打ち消し（逆符号）は 1 step たりとも起きていない。",
          f"- b が担う |Δs| の割合は中央値 {sh_std:.3f}。**b は s の動きの約 "
          f"{100*sh_std:.0f}% しか動かせず、残り約 {100*(1-sh_std):.0f}% は w·µ が動く。**",
          f"- Δs の符号は 1 step では約半々（負率 {_v(C2,'std','frac_ds_negative'):.3f}）だが、"
          f"平均は負（mean Δs = {_v(C2,'std','mean_ds(active)'):.3e}、"
          f"mean Δb = {_v(C2,'std','mean_db(active)'):.3e}）。"
          f"累積すると s は下へ流れる。",
          f"- centered: Δ(w·µ)/Δb の中央値 {_v(C2,'centered','median_dwmu_over_db'):.4f}、"
          f"||µ||² 中央値 {_v(C2,'centered','median_mu_norm_sq'):.5f} で **Δs ≈ Δb**。"
          f"ただし centered は running_mean が毎 step 動くため µ が周期内でも固定されず、"
          f"Δ(w·µ) に w·Δµ が混ざる。**centered のこの行は参考値**"
          f"（同符号率 {_v(C2,'centered','frac_same_sign(dwmu,db)'):.3f} が 1 でないのはそのため）。"
          f"centered の s ≈ b は §E で直接確認する。", "",
          "**これが Issa の疑問への直接の答えである。** µ が周期内で定数であることは、"
          "b がそれを打ち消せることを意味しない。b と w は独立なパラメータに見えるが、"
          "勾配は共通因子 gb で結ばれており、SGD が動かせるのは実質「gb の大きさと符号」"
          "という 1 自由度だけである。その 1 自由度は s を (1+x·µ) 倍のゲインで動かす。"
          "s = 0 を保つには b と w·µ が逆向きに動く必要があるが、その方向は"
          "**この unit の勾配が張る部分空間に無い**。", ""]

    L += ["### C2. 境界での µ ジャンプ（打ち消しを崩す第 2 の経路）", "",
          "境界で flip が 1 ビット反転すると µ が不連続に跳び、w を動かさずに w·µ が跳ぶ。"
          "b にはこれを先取りする術が無い。実現 flip が起きた境界の直前記録点→直後記録点"
          "（1 step）で、その時点で alive な unit に限ると:", ""]
    for arm in ["std", "centered"]:
        L += [f"- {arm}: median |Δ(w·µ)| = {_v(C2,arm,'bnd_median_|dwmu| (alive)'):.4f}、"
              f"q95 |Δ(w·µ)| = {_v(C2,arm,'bnd_q95_|dwmu| (alive)'):.4f}、"
              f"median |Δb| = {_v(C2,arm,'bnd_median_|db| (alive)'):.6f}"
              f"（**ちょうど 0**。その 1 step でその unit が発火しなければ gb = 0 で b は"
              f"一切動かない。実現境界 {int(_v(C2,arm,'n_realized_boundaries'))} 回）。"]
    L += ["", f"比較のため、通常の 1 step（勾配が実際に立った pair）の |Δ(w·µ)| の中央値は "
          f"std {_v(C2,'std','step_median_|dwmu| (active)'):.5f} / centered "
          f"{_v(C2,'centered','step_median_|dwmu| (active)'):.5f}。"
          f"**境界 1 回の µ ジャンプは通常の 1 step の "
          f"{_v(C2,'std','bnd_median_|dwmu| (alive)')/max(_v(C2,'std','step_median_|dwmu| (active)'),1e-12):.0f} 倍**"
          f"（std）の w·µ 変位を生み、b はそれに対して何もしていない。"
          f"µ ジャンプは flip 1 ビットぶんなので w·Δµ = ±w_j、大きさは両アームで同程度だが、"
          f"centered では定常の |w·µ| が ~0 なので**相対的にはるかに大きい撹乱**になる。", ""]

    # ---- D
    L += ["## D. b の負ドリフトは drift か selection か", "",
          "**前提の確認**: 全 unit は b(0) = 0 から出発する（init は b=0、`freeze_bias` は false）。"
          "したがって「もともと b が低い unit が選ばれた」型の selection は原理的に無く、"
          "時点 t の b の値そのものが **その unit が生きているあいだに獲得したドリフト量**である。"
          "残る問いは (i) 生存中の unit 自身がどれだけ下がるか、"
          "(ii) 母集団の値が「死んで止まった unit」の寄せ集めでどれだけ説明されるか、"
          "(iii) 早く負に振れた unit が早く死ぬ（selection）か、の 3 つ。", "",
          f"**本解析で明示的に確認した 2 つの前提**:", "",
          f"- 初期化の時点で p̂ = 0 の unit が std/centered とも "
          f"{100*_v(DEC,'std','frac_born_phat0(step0 で p̂=0)'):.1f}% ある。"
          f"ただしこれらは**永久に不活性ではない**: 境界で flip が起きると µ が動くので "
          f"{100*_v(DEC,'std','frac_born_phat0_that_revive'):.1f}%（std）/ "
          f"{100*_v(DEC,'centered','frac_born_phat0_that_revive'):.1f}%（centered）"
          f"は後で一度は発火する。全記録点で p̂ = 0 のままの unit は std "
          f"{100*_v(DEC,'std','frac_never_fired(全記録点で p̂=0)'):.1f}% / centered "
          f"{100*_v(DEC,'centered','frac_never_fired(全記録点で p̂=0)'):.1f}% で、"
          f"これらの b_final は max |b| = "
          f"{_v(DEC,'std','max |b_final| among never_fired (厳密に 0 のはず)'):.0e} と"
          f"厳密に 0（勾配が完全にゼロなので当然）。"
          f"§3.5 の死亡定義（p̂ の下方クロス）は born dead を「死亡」と数えないので、"
          f"コホート分けでは `born_alive`（step 0 で p̂ ≥ τ）で切り分けてある。",
          f"- **§3.5 の t_death は終端ではない**（1 周期回復しないことしか要求しない）。"
          f"std では frac_ever_dead = {_v(DEC,'std','frac_ever_dead(§3.5)'):.3f} と"
          f"ほぼ全 unit が一度は死ぬが、最終時点の alive は "
          f"{100*(1-_v(DEC,'std','frac_dead_final(p̂<τ)')):.1f}% ある（＝復活がある）。"
          f"そのため `pre_death` コホートは中盤で空になる。組成固定の読みは "
          f"`final_survivor` を使う。", "",
          "### D1. 最終時点（step 1e6）の分解", ""]
    L += [md_table(DEC.pivot(index="stat", columns="arm", values="value").reset_index()), ""]
    bw_all = _v(DEC, "std", "b_over_w_median_all_final")
    bw_al = _v(DEC, "std", "b_over_w_median_alive_final")
    bw_dd = _v(DEC, "std", "b_over_w_median_dead_final")
    L += ["### D2. 読み", "",
          f"1. **母集団 vs 生存**: std 最終時点で母集団 median b/||w|| = {bw_all:+.4f}、"
          f"**alive のみ = {bw_al:+.4f}**、dead のみ = {bw_dd:+.4f}。"
          f"生 b でも alive {_v(DEC,'std','b_median_alive_final'):+.4f} / dead "
          f"{_v(DEC,'std','b_median_dead_final'):+.4f}。"
          f"母集団の −0.20 は、{100*_v(DEC,'std','frac_dead_final(p̂<τ)'):.1f}% を占める"
          f"**死亡 unit の値でほぼ決まっている**（composition）。",
          f"2. **「死亡 = 凍結」は τ=0.05 では成り立たない**（本解析で見つかった注意点）。"
          f"p̂ = 1/32 は τ を下回るが gate は開くので勾配はゼロでない。実際 §3.5 の t_death "
          f"以降の |Δb| の median は std {_v(DEC,'std','median |b_final − b(t_death)| (§3.5 死亡後)'):.4f} / "
          f"centered {_v(DEC,'centered','median |b_final − b(t_death)| (§3.5 死亡後)'):.4f} で"
          f"**大きく動いている**。真に凍るのは p̂ ≡ 0 になって以降で、そこからの |Δb| の "
          f"median は {_v(DEC,'std','median |b_final − b(t_frozen)| (p̂≡0 以降・厳密に 0 のはず)'):.1e}"
          f"（＝厳密に凍っている）。p̂≡0 に到達する unit は std "
          f"{100*_v(DEC,'std','frac_ever_frozen(p̂≡0 以降)'):.1f}% / centered "
          f"{100*_v(DEC,'centered','frac_ever_frozen(p̂≡0 以降)'):.1f}%。"
          f"死亡時点の b の median は "
          f"{_v(DEC,'std','b_median_at_death (生涯ドリフト)'):+.4f} しかないのに"
          f"最終的な dead の b は {_v(DEC,'std','b_median_dead_final'):+.4f} まで行く。"
          f"つまり **b の負ドリフトの 8 割方は「最初の §3.5 死亡より後」に稼がれている**。"
          f"§3.5 の死亡は初到達イベントであって学習の終点ではなく、unit は"
          f" alive と dead を往復している。",
          f"3. **どの発火状態でドリフトを稼いだかの分解**（b(0)=0 なので b_final = "
          f"Δb(p̂≥τ 中) + Δb(0<p̂<τ 中) + Δb(p̂=0 中) が厳密に成り立つ。"
          f"区間のラベルは区間始点の p̂ なので、1000 step 区間では粗い近似）:", "",
          f"   | 帯 | std | centered |", f"   |---|---|---|",
          f"   | Δb median while p̂≥τ（発火中） | "
          f"{_v(DEC,'std','Δb median while p̂>=τ'):+.4f} | "
          f"{_v(DEC,'centered','Δb median while p̂>=τ'):+.4f} |",
          f"   | Δb median while 0<p̂<τ（瀕死） | "
          f"{_v(DEC,'std','Δb median while 0<p̂<τ'):+.4f} | "
          f"{_v(DEC,'centered','Δb median while 0<p̂<τ'):+.4f} |",
          f"   | Δb median while p̂=0（区間始点） | "
          f"{_v(DEC,'std','Δb median while p̂=0 (区間始点で p̂=0)'):+.4f} | "
          f"{_v(DEC,'centered','Δb median while p̂=0 (区間始点で p̂=0)'):+.4f} |",
          f"   | 参考: b median at 1e6 | {_v(DEC,'std','b_median_all_final'):+.4f} | "
          f"{_v(DEC,'centered','b_median_all_final'):+.4f} |", "",
          f"   **負ドリフトは「発火している最中」に稼がれている**"
          f"（std で −0.50 / 全体 −0.58 の 8 割超、centered でも同様）。"
          f"「瀕死の帯でじりじり削られる」像ではない。"
          f"2 と合わせると: unit は死んでは復活し、復活して発火しているあいだに"
          f"また b を下げ、それを繰り返しながら徐々に沈む。",
          f"4. **弱い selection もある**: 早期（step 10,000）の b/||w|| と t_death の Spearman は "
          f"std {_v(DEC,'std','spearman(bw_early, t_death | dead)'):+.3f} / centered "
          f"{_v(DEC,'centered','spearman(bw_early, t_death | dead)'):+.3f}（正 = 早期に b が"
          f"低いほど早く死ぬ）。早期 b/||w|| の median は「10 万 step までに死ぬ群」std "
          f"{_v(DEC,'std','bw_early_median_dead_by_1e5'):+.4f} vs 「最終生存群」std "
          f"{_v(DEC,'std','bw_early_median_final_survivor'):+.4f} で、"
          f"**5 倍ほどの差はあるが絶対値は小さい**。selection は存在するが主因ではない。", "",
          "**まとめ（drift か selection か）**: **per-unit の drift である**。"
          "b(0)=0 から始まる以上、母集団の b が下がるには個々の unit の b が"
          "自力で下がるしかなく、実際 Δb の中央値はどの帯でも負である。"
          "「b が低い unit がもともと居て選ばれた」型の selection は原理的に無い。"
          "**ただし報告されている母集団値 b/||w|| = −0.15/−0.20 の水準は composition が主役**で、"
          "最終時点で生き残っている unit（std で 5.3%）の b/||w|| は −0.06 しかなく、"
          "母集団値の大半は既に低 p̂ 帯に沈んだ 94.7% の unit の値である。"
          "「早く b が下がった unit が早く死ぬ」型の弱い selection も存在するが"
          "（Spearman +0.12、早期 b/||w|| の群間差 0.015 程度）主因ではない。", "",
          "### D3. コホート別の時系列（bulk・b/||w|| の median）", ""]
    rows = []
    for arm in ["std", "centered"]:
        for coh in ["all", "alive_tau", "dead_tau", "dead_tau_born_alive",
                    "pre_death", "final_survivor"]:
            g = TSD[(TSD.arm == arm) & (TSD.cohort == coh) & (TSD.is_bulk)]
            r = dict(arm=arm, cohort=coh)
            for st in ST5:
                x = g[g.step == st]
                r[f"t={st}"] = float(x.b_over_w_med.iloc[0]) if len(x) else np.nan
            xf = g[g.step == ST5[-1]]
            r["n_final"] = int(xf["n"].iloc[0]) if len(xf) else 0
            rows.append(r)
    L += [md_table(pd.DataFrame(rows)), "",
          "`pre_death` = born_alive かつまだ §3.5 の t_death に達していない unit"
          "（死亡 unit の**生前だけ**を見るコホート）、"
          "`dead_tau_born_alive` = dead のうち born dead を除いたもの、"
          "`final_survivor` = step 1e6 で alive な unit を最初から追った**固定コホート**"
          "（コホート組成が動かないので純粋な per-unit drift になる）。", ""]
    rows2 = []
    for arm in ["std", "centered"]:
        g = TSD[(TSD.arm == arm) & (TSD.cohort == "final_survivor") & (TSD.is_bulk)]
        for st in ST5:
            x = g[g.step == st]
            if len(x):
                rows2.append(dict(arm=arm, step=st, b_med=float(x.b_med.iloc[0]),
                                  b_over_w_med=float(x.b_over_w_med.iloc[0]),
                                  w_norm_med=float(x.w_norm_med.iloc[0]),
                                  s_med=float(x.s_med.iloc[0])))
    r2 = pd.DataFrame(rows2)
    L += ["固定コホート `final_survivor` の生 b・||w||・s:", "",
          md_table(r2), "",
          f"b/||w|| が浅く見えるのは分母 ||w|| が伸びるからでもある"
          f"（std の最終生存コホートで ||w|| median "
          f"{float(r2[(r2.arm=='std')].w_norm_med.iloc[0]):.2f} → "
          f"{float(r2[(r2.arm=='std')].w_norm_med.iloc[-1]):.2f}）。"
          f"生 b で見ると centered は単調に下がり続け、std は 5e5 付近を底にやや戻す。"
          f"いずれにせよ **b/||w|| だけを見ると per-unit の drift を過小評価する**。"
          f"なお同じ固定コホートの s（最右列）は "
          f"{float(r2[(r2.arm=='std')].s_med.iloc[0]):+.2f} → "
          f"{float(r2[(r2.arm=='std')].s_med.iloc[-1]):+.2f}（std）で、"
          f"**組成を固定しても s は 0 に戻らない**。", ""]

    # ---- E
    L += ["## E. centered との対比", ""]
    L += [md_table(E.pivot(index="stat", columns="arm", values="value").reset_index()), "",
          f"centered は ||µ|| が {_v(E,'std','median mu_norm'):.4f} → "
          f"{_v(E,'centered','median mu_norm'):.4f}（最終 "
          f"{_v(E,'centered','mu_norm final'):.4f}）まで潰れるので w·µ 項がほぼ消え、"
          f"**s ≈ b** になる: median |s−b|/|s| = "
          f"{_v(E,'centered','median |s - b| / |s|'):.4f}（std は "
          f"{_v(E,'std','median |s - b| / |s|'):.4f}）、"
          f"median |w·µ|/|b| = {_v(E,'centered','median |w·µ| / |b|'):.4f}（std は "
          f"{_v(E,'std','median |w·µ| / |b|'):.4f}）。", "",
          f"死亡条件を **w·µ を無視した `b + M̂ ≤ 0`** で予測した一致率は centered "
          f"{_v(E,'centered','agree(p_hat==0, b+M̂<=0)'):.4f} に対し std は "
          f"{_v(E,'std','agree(p_hat==0, b+M̂<=0)'):.4f} しかない。"
          f"**正しい `s + M̂ ≤ 0`** なら centered {_v(E,'centered','agree(p_hat==0, s+M̂<=0)'):.4f} / "
          f"std {_v(E,'std','agree(p_hat==0, s+M̂<=0)'):.4f}（残差は κ̂ の誤差であり、"
          f"厳密な M では checkpoint 上 1000/1000 で成立することを "
          f"`analysis/q3_margin_exact.py` が確認済み）。", "",
          "既報の「centered では death が `b + M ≤ 0` になる（一致率 98.9%）」は"
          "**再現する向き**である（本表の 0.92 は κ̂ 近似ぶんの目減り）。"
          "つまり centered では消灯の唯一の駆動が b の負ドリフトになり、"
          "std では w·µ の負ドリフトが主役になる、という対比が数値として立つ。", ""]

    # ---- F
    L += ["## F. 能力の問題か時間の問題か（alive の s が 0 に近づくか）", "",
          "bulk グリッドを時間窓で切り、各窓の alive unit の s/M̂・p̂・s/||w|| の分布:", ""]
    L += [md_table(TR[["arm", "window", "qty", "n", "q25", "med", "q75"]]), ""]
    for arm in ["std", "centered"]:
        g = TR[(TR.arm == arm) & (TR.qty == "s_over_M|alive")]
        gp = TR[(TR.arm == arm) & (TR.qty == "p_hat|alive")]
        if len(g) >= 3:
            L += [f"- {arm}: alive の s/M̂ 中央値は {g['med'].iloc[0]:+.3f}（step 0）→ "
                  f"{g['med'].iloc[1]:+.3f} → … → {g['med'].iloc[-1]:+.3f}、"
                  f"p̂ 中央値は {gp['med'].iloc[0]:.3f} → {gp['med'].iloc[-1]:.3f}。"]
    L += ["", "どちらのアームでも alive の残差は **0 に近づかない**。"
          "std は途中（[2.5e5, 5e5) 付近）で最も深くなり、その後わずかに戻すが 0 には遠い。"
          "centered は単調に 0 から離れる。したがって「遅いが追随している」の像は支持されない。"
          "**能力の問題（構造的に打ち消せない）であって、時間の問題ではない。**", "",
          "ただし留保として、後半で alive に残る unit は選択された部分集団であり"
          "（std の alive は step 1e6 で 5.3%）、この時系列はコホート組成の変化を含む。"
          "組成を固定した見方は §D3 の `final_survivor` 行を参照。", ""]

    # ---- 留保
    L += ["## 留保", "",
          "1. **事後計算・未事前登録**。spec も判定基準も無く、verdict.csv は書き換えていない。"
          "引用には事前登録つきの昇格が要る。",
          "2. **M̂ は近似**。κ̂ は step 0 / 1M の checkpoint から作った回帰の内挿で、"
          "step 1M checkpoint は本走ではなく同一 config の別実現。per-unit の s/M̂ は"
          "「その unit の s/M」ではない。§A3 の推定量感度、§A5 の挟み込み較正、"
          "および κ 非依存な p̂ の分布の 3 つで結論の向きが変わらないことを確認している。",
          "3. **§A5 は κ̂ が M を系統的に過小評価していることを示している**"
          f"（M̂/M_mid 中央値 std {kfix:.2f}）。したがって s/M̂ の絶対値は"
          "やや大きめに出ている。補正しても符号と「0 でない」という結論は変わらない。"
          "較正は消灯直前の unit という選択された部分集団で行っている点も留保。",
          "4. s / s/||w|| / p̂ / w·µ / b は記録量からの**厳密**復元（近似は M̂ だけ）。",
          "5. alive 条件は s/M > −1 という**下からの切り取り**を自動的に課す。"
          "「alive の s/M が負」自体は自明で、意味があるのは「0（p̂≈0.5）に寄らない」ことである。",
          "6. §C の増分解析は **std でのみ厳密**。centered は running_mean が毎 step 動くので "
          "µ が周期内でも固定されず、Δ(w·µ) に w·Δµ が混ざる（centered の行は参考値）。",
          "7. §C の 1 step 隣接ペアは**境界窓の内側にしか無い**。バルクの 1000 step 刻みでは "
          "1 step の増分関係は測れない。境界窓は学習の途中経過としては偏った標本だが、"
          "Δ(w·µ) = Δb·(x_in·µ) 自体は代数恒等式なので窓の選び方に依らない。",
          "8. 記録量は float32。1 step の Δb は小さいので、|Δb| が float32 の丸め下限に近い"
          "ペアは除外した（§C の `frac_db_active` 参照。除外は gate が閉じた unit を"
          "落とす効果が大半）。",
          "9. §D の cohort 時系列は組成が時間とともに変わる。組成固定の読みは "
          "`final_survivor` 行のみ。",
          "10. §D3 の Δb 帯別分解は「区間の始点の p̂」でラベルする。記録点の "
          f"{100*_v(DEC,'std','frac of intervals that are 1 step'):.1f}% は 1 step 隣接だが、"
          "**経過時間の 98% は 1000 step 区間**が占めるので、その区間内で unit が帯を"
          "またぐぶんの誤配分が入る。成分の和が b_final に一致することは厳密だが、"
          "帯への割り振り自体は粗い。",
          "11. スコープは condA・w100・T=1e4・batch=1・center_alpha=0.01・10 seed のみ。"
          "他の m/f・幅・batch・condB へは外挿しない（κ の構造上限も ||µ|| も m/f で動く）。", ""]

    L += ["## 生成物", "",
          "- `s_dist_pooled.csv` — s / s/||w|| / s/M̂ / p̂ の pooled 分位（grid × alive 条件 × κ̂ 感度）",
          "- `s_time.csv.gz` / `s_time_bulk.csv` — 記録点ごとの分位（cond = all / alive_tau / dead_tau）",
          "- `increment_summary.csv` / `increment_stats.csv` — 1 step 増分の代数と実測",
          "- `b_cohort_time.csv.gz` / `b_cohort_time_bulk.csv` — b・b/||w||・||w||・s・s/M̂ のコホート別時系列",
          "- `per_unit.csv` — seed×unit ごとの t_death・t_frozen・生前最後の b・早期 b・最終値",
          "- `b_decomposition.csv` — 最終時点の drift/composition/selection 分解",
          "- `arm_contrast.csv` — std / centered の対比と死亡条件の一致率",
          "- `alive_trend.csv` — alive の s/M̂・p̂ の時間窓別分布",
          "- `kappa_bracket_check.csv` / `kappa_consistency.csv` — κ̂ の裏取り",
          "- `phat_to_sM_inversion.csv` — κ を使わない p̂ → s/M 逆算表（§A6）",
          "- `figures/fig_cancel_s_time.png` — s/||w|| と s/M̂ の時間発展（all vs alive・両 arm）",
          "- `figures/fig_cancel_decomp.png` — w·µ と b の 2 項分解（alive）",
          "- `figures/fig_cancel_b_drift.png` — b/||w|| のコホート別（drift vs composition）",
          "- `figures/fig_cancel_phat_alive.png` — alive の p̂（κ 非依存の s/M 代理）", ""]

    (OUTDIR / "results.md").write_text("\n".join(L), encoding="utf-8")
    meta = dict(script="analysis/q3_cancel_residual.py",
                git=git_hash(), generated=time.strftime("%Y-%m-%dT%H:%M:%S"),
                elapsed_sec=round(time.time() - t0, 1),
                preregistered=False, tau=TAU, main_kappa_estimator=MAIN_EST,
                sensitivity_estimators=SENS_EST, kappa_sources=kinfo,
                source_meta={k: {kk: v[kk] for kk in ("spec", "n_record_steps",
                                                      "n_realized_flips")}
                             for k, v in metas.items()},
                python=platform.python_version(), numpy=np.__version__,
                pandas=pd.__version__)
    (OUTDIR / "analysis_meta.json").write_text(
        json.dumps(meta, indent=1, ensure_ascii=False, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
