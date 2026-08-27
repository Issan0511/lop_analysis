"""xr_recheck_0827: 8/22 の x/r 分解チャット結果の再現登録 [specs/spec_xr_recheck_0827.md]。

  OMP_NUM_THREADS=1 .venv/bin/python -m analysis.xr_recheck.xr_recheck \
      [--outdir results/xr_recheck_0827]

**新規走行なし**。`results/ratchet_log_0819/logs/seed*.npz` と
`results/posreset_0819/snapshots/A_w100_cont_step500000.pt` だけを読む。

spec §1 の格の宣言に従い、各行に tier を付ける:

  repro   旧値があるもの。許容幅は spec §5 で凍結済み。外れたら旧値を撤回する
  prereg  旧値に無いもの (CI・等価幅判定・符号分率の区間)。真の事前登録として成立する
  report  判定を置かない記述 (A3-c・A6・付随量)

判定は spec §5 の A1–A6 と §6 の S0–S5 が唯一の正で、本モジュールはそれを実装するだけ。
実施順序は spec §9 の `A1 -> A4 -> A2 -> A5 -> A3 -> A6` で、**A1 が FAIL したら以降を
計算せずに中止**する (SystemExit)。

--------------------------------------------------------------------------------
spec に無く、本モジュールで確定させた定義 (すべて 8/22 の測定条件に合わせたもの)
--------------------------------------------------------------------------------

spec §5 の A2・A5 は 8/22 チャットの数値を repro 対象に挙げているが、その数値が
**どの母集団の上で計算されたか**を書いていない。母集団を変えると同じ統計量が別物に
なるため (下記)、vault の記述から母集団を復元して固定した。復元の出所も併記する。

  A2-a/A2-c  [[xr分解と生存機構_0822]] §19「タスク内・**発火中**限定・Δstep=1000、
             n = 149,492」。したがって
               母集団 = 非境界ペア かつ Δstep == 1000 かつ ペア始点で **p̂ > 0**
             閾値ではなく**離散の非ゼロ**である (p̂ = k/32 なので p̂>0 <=> k>=1)。
             p̂ >= TAU (= k>=2) にすると 119,046 件になり 30,446 件ずれる。
             ゲートを外すと消灯ユニットの Δx = 0 が流れ込んで p₊ は別物になる。
             この母集団は S6 の加法恒等式で検算する (下記)。
  A2-b       同 §19「10 seed × 100 unit (記録点ベース)」で `x_max > 0` の割合 0.9490。
             分母は**ユニット 1000 本**であってペアではない。
  A3         同 §7「posreset_0819 スナップショット (step=500k)、condA の入力サポート
             32 パターンで厳密評価」。alive = p̂ > 0 を snapshot 自身から再計算する
             (S5: logs 側の p̂ を使うと別 run の seed 対応を仮定することになる)。
             総数 alive 120 / dead 880 を S0 の中止条件に置く (§7 の seed あたり
             12.0 / 88.0、§17 の n=120 / 880 と一致すべき)。
  A5         母集団は A2 と同じ。窓の残差は窓内記録点の `eval_loss_exact` の平均。
             ゲートを外すと median|Δx| が全窓で 0 になる。

`--pair-population` / `--a5-loss-agg` で母集団を切り替えられるようにしてあるが、
**既定値が spec の判定に使う正**であり、切り替えは逸脱節に記録される。

--------------------------------------------------------------------------------
推定量 (spec §7 の「seed クラスタ」の 2 通りの読みを項目ごとに割り当てる)
--------------------------------------------------------------------------------

  pooled  seed を復元抽出し、選ばれた seed の分子・分母を合算して比を作り直す
  mean    seed ごとの値を作ってから平均する (`src.figures_ratchet_log.boot_ci` と同じ)

  A2-a (repro)         **pooled**。旧値 0.4971 / 0.5008 が pooled なので、同じ推定量で
                       なければ再現の検定にならない
  A2-c (prereg・主判定) **mean**。seed が独立実現 = 推論の単位。pooled は「発火中
                       ユニット数」というノイズ変数で seed を重み付けてしまう
  乖離 (pooled − mean)  report tier で必ず出す。大きければ「seed 間で対称性が揃って
                       いない」という別の所見になる
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from src.common import ROOT                                        # noqa: E402
from src.figures_ratchet_log import TAU                            # noqa: E402
from src.figures_teachw import spearman                            # noqa: E402
from analysis.q3_boundary_race import freeze_events_all            # noqa: E402

BOOT_N = 20000
BOOT_SEED = 20260827            # spec §7 の事前登録の抽選列
PERIOD = 10_000                 # condA T
HALF_W = 100                    # ratchet.boundary_window (A4 の窓)
BULK = 1000                     # A2/A5 が使う Δstep
M_DIM, F_DIM = 20, 15           # condA: 自由 5 bit -> 32 パターン
N_SEED = 10
N_UNIT = 100

LOGDIR = os.path.join(ROOT, "results", "ratchet_log_0819", "logs")
SNAPSHOT = os.path.join(ROOT, "results", "posreset_0819", "snapshots",
                        "A_w100_cont_step500000.pt")
A4_SCRIPT = os.path.join(ROOT, "analysis", "q3_boundary_race.py")

# spec §5 の repro 目標値 (許容幅は spec / 追補が凍結したもの。ここで緩めない)
T_A1A_PAIRS = 16_475_293        # 非境界・off_prev ペア数 (厳密一致)
T_A1D_PAIRS = 80_753            # 境界・off_prev ペア数 (厳密一致・追補 §9 で repro に格下げ)
T_A1D_REVIVE = 9_318            # 同ペアでの復活数 (厳密一致)。11.54%
T_A2A_PLUS, T_A2A_MINUS = 0.4971, 0.5008        # ±0.5pt (pooled)
T_A2B_VISIT = 0.9490            # ±1.0pt
T_A4_LO, T_A4_HI = 0.099, 0.186 # 境界あたりハザードのレンジ (両端 ±0.0005)
T_A4_TOL = 5e-4

# S6: 母集団定義の加法恒等式 (追補 §1.2)。ゼロ許容。
T_S6_TOTAL = 801_000            # Δstep=1000・タスク内の全ペア
T_S6_ON = 149_492               # うち始点で p̂ > 0  -> §19/§3 の母集団
T_S6_OFF = 651_508              # うち始点で p̂ = 0  -> §14 の「1000 step ペア、復活 0」

# S0: snapshot から再計算した生死の総数 (追補 §7)。ゼロ許容。
T_A3_ALIVE, T_A3_DEAD = 120, 880

# 判定に使わない旧値 (summary の突き合わせ欄にだけ出す)
OLD = dict(a3_alive_signed=0.034, a3_dead_signed=0.180, a3_alive_abs=0.277,
           a3_dead_abs=0.259, a3_eff_rank_a=6.61, a3_eff_rank_w=7.53,
           a3_n_alive=12.0)

EQ_A2 = (0.48, 0.52)            # A2-c 等価幅
EQ_A3 = (-0.10, 0.10)           # A3-a 等価幅
A5_FLOOR = 0.5                  # A5-a: CI 下限 > 0.5

plt.rcParams["font.family"] = ["Noto Sans CJK JP", "Noto Sans CJK TC",
                               "Noto Sans CJK KR", "IPAGothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ------------------------------------------------------------------ 読み込み・下ごしらえ

UNIT_KEYS = ("p_hat", "cos_u_mu", "w_norm", "b", "F_self", "F_rest")
RUN_KEYS = ("mu_norm", "eval_loss_exact")


def load_seed(path):
    """1 seed の npz から本 spec が使う列だけを取る (全列を持つと 10 seed で数 GB)。"""
    with np.load(path) as z:
        d = {k: z[k] for k in UNIT_KEYS + RUN_KEYS + ("step", "flip_state",
                                                      "seed", "run_id")}
    d["step"] = d["step"].astype(np.int64)
    # spec §4 の x = ||w||·||µ||·cos + b (= 前活性の µ̂ 平行成分。w·µ+b と同値)
    d["x"] = (d["w_norm"] * d["mu_norm"][:, None] * d["cos_u_mu"] + d["b"])
    return d


def seed_paths():
    ps = sorted((p for p in os.listdir(LOGDIR) if p.endswith(".npz")),
                key=lambda p: int(p[4:-4]))
    return [os.path.join(LOGDIR, p) for p in ps]


def task_id(step):
    """記録点が属するタスク番号。

    probe はループ本体先頭で呼ばれるので境界 B の flip は記録点 B と B+1 の**間**で
    起きる (spec §2.2 の mu_norm の実測もそうなっている)。したがって step=B はまだ
    前のタスクであり、素朴な `step // T` は境界ペアをタスク内に混ぜる
    ([[xr分解と生存機構_0822]] §18 で自己検出されたマスクのバグ)。spec §4 の
    「`step[s] // T` は使わない」はこれを指す。"""
    return np.where(step == 0, 0, (step - 1) // PERIOD)


def pair_masks(d):
    """隣接記録ペアの分類 [spec §4]。

    is_bnd: flip_state がそのペアで変化した = 境界ペア。**ハードコードしない** (S2)。
    dstep:  ペアの step 差。A2/A5 の母集団は dstep == 1000 に限る。
    """
    is_bnd = (np.abs(np.diff(d["flip_state"], axis=0)) > 0).any(axis=1)
    dstep = np.diff(d["step"])
    return is_bnd, dstep


# ------------------------------------------------------------------ S0 / S2 / S3

def check_s0():
    """S0: 入力の実在と形状。1 件でも外れたら中止 [spec §6]。"""
    import torch
    rows = []

    def add(name, ok, detail):
        rows.append(dict(check=name, ok=bool(ok), detail=detail))

    paths = seed_paths() if os.path.isdir(LOGDIR) else []
    add("logs/seed*.npz が 10 本", len(paths) == N_SEED, f"{len(paths)} 本")
    tracked = git_tracked(paths + [SNAPSHOT])
    add("logs が git 追跡下", all(tracked.get(p, False) for p in paths),
        f"{sum(tracked.get(p, False) for p in paths)}/{len(paths)}")
    add("snapshot が存在・追跡下",
        os.path.exists(SNAPSHOT) and tracked.get(SNAPSHOT, False), SNAPSHOT)
    add("q3_boundary_race.py が存在", os.path.exists(A4_SCRIPT), A4_SCRIPT)

    snap = torch.load(SNAPSHOT, map_location="cpu", weights_only=False)
    W = snap["net"]["W"]
    add("net/W の shape が (10,100,20)", tuple(W.shape) == (N_SEED, N_UNIT, M_DIM),
        str(tuple(W.shape)))
    add("running_mean の shape が (10,20)",
        tuple(snap["running_mean"].shape) == (N_SEED, M_DIM),
        str(tuple(snap["running_mean"].shape)))
    add("step == 500000", int(snap["step"]) == 500_000, str(snap["step"]))
    seeds = sorted(int(r["seed"]) for r in snap["runs"])
    add("runs の seed 集合が 0..9", seeds == list(range(N_SEED)), str(seeds))

    # A3 の母集団 [追補 §7]。外れたら A3 の生死定義が違うので中止する。
    gate = snapshot_gate(snap)
    add("§9 の閉形式が 32 パターン列挙の max と一致",
        gate["max_closed_err"] <= 1e-12, f"max abs err {gate['max_closed_err']:.3e}")
    n_alive = int((gate["p_hat"] > 0).sum())
    n_dead = int((gate["p_hat"] == 0).sum())
    add(f"snapshot の alive (p̂>0) 総数 = {T_A3_ALIVE}", n_alive == T_A3_ALIVE,
        f"{n_alive} (seed 平均 {n_alive / N_SEED:.1f})")
    add(f"snapshot の dead (p̂=0) 総数 = {T_A3_DEAD}", n_dead == T_A3_DEAD,
        f"{n_dead} (seed 平均 {n_dead / N_SEED:.1f})")
    add("参考: p̂ >= TAU の総数 (判定に使わない)", True,
        f"{int((gate['p_hat'] >= TAU).sum())}")
    return all(r["ok"] for r in rows), pd.DataFrame(rows), snap, gate


def check_s2(seeds):
    """S2: 境界を flip_state の差分から機械的に決め、99/99 が step ≡ 0 (mod T) [spec §6]。"""
    rows, ok = [], True
    for d in seeds:
        is_bnd, dstep = pair_masks(d)
        left = d["step"][:-1][is_bnd]
        aligned = int((left % PERIOD == 0).sum())
        adj = int((dstep[is_bnd] == 1).sum())
        n = int(is_bnd.sum())
        good = (n == 99 == aligned == adj)
        ok &= good
        rows.append(dict(seed=int(d["seed"]), run_id=str(d["run_id"]),
                         n_boundary_pairs=n, n_left_aligned=aligned,
                         n_adjacent=adj, ok=good,
                         bad_steps=[int(s) for s in left[left % PERIOD != 0]][:5]))
    return ok, pd.DataFrame(rows)


def check_s3(seeds):
    """S3: mu_norm が各タスク内で厳密に一定 (unique が 1 個) [spec §6・§2.2]。"""
    rows, ok = [], True
    for d in seeds:
        tid = task_id(d["step"])
        bad = [int(t) for t in np.unique(tid)
               if np.unique(d["mu_norm"][tid == t]).size != 1]
        good = not bad
        ok &= good
        rows.append(dict(seed=int(d["seed"]), n_task=int(np.unique(tid).size),
                         n_task_with_multiple_mu=len(bad), ok=good,
                         bad_tasks=bad[:5]))
    return ok, pd.DataFrame(rows)


# ------------------------------------------------------------------ A1

def a1_seed(d):
    """A1: 片側吸収の厳密性 [spec §5]。p̂=0 のユニットは非境界ペアで動けない。"""
    is_bnd, _ = pair_masks(d)
    p0, p1 = d["p_hat"][:-1], d["p_hat"][1:]
    off = (p0 == 0)
    dcos = np.abs(np.diff(d["cos_u_mu"], axis=0))
    dwn = np.abs(np.diff(d["w_norm"], axis=0))

    nb = off & ~is_bnd[:, None]                       # 非境界・off_prev
    bd = off & is_bnd[:, None]                        # 境界・off_prev (A1-d の対照)
    rev = (p1 != 0)
    return dict(
        seed=int(d["seed"]),
        a1_pairs=int(nb.sum()), a1_revive=int((nb & rev).sum()),
        a1_max_dcos=float(dcos[nb].max()) if nb.any() else 0.0,
        a1_max_dwnorm=float(dwn[nb].max()) if nb.any() else 0.0,
        a1d_pairs=int(bd.sum()), a1d_revive=int((bd & rev).sum()))


# ------------------------------------------------------------------ A4

def hazard_levels(haz):
    """A4 のハザードを 2 つの集約粒度で出す。

    親 spec A4-a の「レンジ 0.099–0.186」は `q3_boundary_race.hazard_table` が返す
    **ブロック単位** (境界 11 本ずつ・`range(1, 100, 11)`) の `hazard_state` のレンジ
    である。境界ごと (99 点) のレンジは 1 境界あたりの分母が ~1000 本しかないぶん
    はるかに広くなるので、判定にはブロック版を使う。
    """
    h = haz.groupby("b_index", as_index=False).agg(
        n_alive=("n_alive", "sum"), n_freeze_win=("n_freeze_win", "sum"))
    h["hazard_state"] = h.n_freeze_win / h.n_alive.replace(0, np.nan)
    rows = []
    for lo in range(1, 100, 11):
        hi = min(lo + 10, 99)
        m = (h.b_index >= lo) & (h.b_index <= hi)
        nl, kf = int(h.n_alive[m].sum()), int(h.n_freeze_win[m].sum())
        rows.append(dict(block=f"{lo}-{hi}", n_alive=nl, n_freeze_win=kf,
                         hazard_state=kf / nl if nl else np.nan))
    return h, pd.DataFrame(rows)


def a4_seed(d):
    """A4: 境界 1 回あたりのハザード。

    判定規則は `analysis/q3_boundary_race.py` の `hazard_state` と同一 —
    分母 = 境界 B の記録点で alive (p̂ >= TAU) な unit、分子 = そのうち窓 [B+1, B+100]
    に `death_events` と同じ規則の消灯イベントを持つ unit。`freeze_events_all` を
    当該モジュールから import して使うので、判定規則はコード上も一本化されている。
    """
    step, p = d["step"], d["p_hat"]
    fz = [np.array([step[i] for i in ev], dtype=np.int64)
          for ev in freeze_events_all(step, p)]
    index = {int(v): i for i, v in enumerate(step)}
    is_bnd, _ = pair_masks(d)
    rows = []
    for k, B in enumerate(sorted(int(s) for s in step[:-1][is_bnd])):
        i0 = index[B]
        alive0 = p[i0] >= TAU
        nfz = sum(1 for u in np.flatnonzero(alive0)
                  if fz[u].size and ((fz[u] >= B + 1) & (fz[u] <= B + HALF_W)).any())
        rows.append(dict(seed=int(d["seed"]), b_index=k + 1, step=B,
                         n_alive=int(alive0.sum()), n_freeze_win=int(nfz)))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ A2 / A5 / A6

def stir_masks(d, population="on_bulk"):
    """A2/A5/A6 が使う「撹拌」ペアの母集団 (モジュール docstring・追補 §1 参照)。

    既定 `on_bulk` = 非境界 かつ Δstep == 1000 かつ ペア始点で **p̂ > 0**（発火中）。
    [[xr分解と生存機構_0822]] §19 の n = 149,492 の母集団。p̂ = k/32 なので
    `p̂ > 0` は閾値ではなく離散の非ゼロ (k >= 1) であり、`p̂ >= TAU` (k >= 2) とは
    30,446 件ずれる。S6 の加法恒等式で検算する。
    """
    is_bnd, dstep = pair_masks(d)
    m = ~is_bnd
    if population.endswith("_bulk"):
        m = m & (dstep == BULK)
    m = np.broadcast_to(m[:, None], d["p_hat"][:-1].shape)
    if population.startswith("on_"):
        m = m & (d["p_hat"][:-1] > 0)
    elif population.startswith("alive_"):
        m = m & (d["p_hat"][:-1] >= TAU)
    return m


def s6_counts(d):
    """S6: 母集団定義の加法恒等式 [追補 §1.2]。

        Δstep=1000・タスク内の全ペア = (始点 p̂>0) + (始点 p̂=0)
                             801,000 =    149,492  +   651,508

    右辺の 2 項はそれぞれ [[xr分解と生存機構_0822]] §19（撹拌の母集団）と §14
    （1000 step ペアで復活 0）が別々に報告している値である。加法的に閉じることが
    母集団定義のゼロ許容チェックになる。"""
    is_bnd, dstep = pair_masks(d)
    base = np.broadcast_to((~is_bnd & (dstep == BULK))[:, None], d["p_hat"][:-1].shape)
    on = base & (d["p_hat"][:-1] > 0)
    return dict(seed=int(d["seed"]), s6_total=int(base.sum()), s6_on=int(on.sum()),
                s6_off=int((base & ~on).sum()),
                s6_alive_tau=int((base & (d["p_hat"][:-1] >= TAU)).sum()))


def a2_seed(d, population):
    """A2: 撹拌はほぼ対称 [spec §5]。"""
    m = stir_masks(d, population)
    dx = np.diff(d["x"], axis=0)[m]
    n = int(dx.size)
    xmax = d["x"].max(axis=0)                          # ユニットごとの生涯最高到達点
    # A2-d: 最終記録点での復活 (終端の隣接ペアで p̂=0 -> p̂>0)
    term = int(((d["p_hat"][-2] == 0) & (d["p_hat"][-1] != 0)).sum())
    return dict(
        seed=int(d["seed"]), a2_n=n,
        a2_n_plus=int((dx > 0).sum()), a2_n_minus=int((dx < 0).sum()),
        a2_n_zero=int((dx == 0).sum()),
        a2_p_plus=float((dx > 0).mean()) if n else np.nan,
        a2_p_minus=float((dx < 0).mean()) if n else np.nan,
        a2_n_unit=int(xmax.size), a2_n_visit=int((xmax > 0).sum()),
        a2_visit=float((xmax > 0).mean()),
        a2_xmax_med=float(np.median(xmax)), a2_xmax_max=float(xmax.max()),
        a2d_terminal_revive=term,
        a2_dx_mean=float(dx.mean()) if n else np.nan,
        a2_dx_sd=float(dx.std(ddof=1)) if n > 1 else np.nan)


def window_table(d, population):
    """窓 (= タスク) ごとの median|Δx| / 残差 / |F_self|/|F_rest| / E[Δx]。A5・A6 用。"""
    m = stir_masks(d, population)
    dx = np.diff(d["x"], axis=0)
    tid = task_id(d["step"])
    tid_pair = tid[:-1]                                # ペアは始点のタスクに属する
    fs, fr = np.abs(d["F_self"]), np.abs(d["F_rest"])
    alive_rec = d["p_hat"] >= TAU
    rows = []
    for t in np.unique(tid):
        pm = (tid_pair == t)[:, None] & m
        if not pm.any():
            continue
        v = np.abs(dx[pm])
        rm = (tid == t)
        am = rm[:, None] & alive_rec
        ratio = (fs[am] / np.where(fr[am] > 0, fr[am], np.nan)) if am.any() else np.array([])
        rows.append(dict(
            seed=int(d["seed"]), task=int(t), n_pair=int(pm.sum()),
            med_abs_dx=float(np.median(v)), mean_dx=float(dx[pm].mean()),
            loss_mean=float(d["eval_loss_exact"][rm].mean()),
            loss_first=float(d["eval_loss_exact"][rm][0]),
            ratio_med=float(np.nanmedian(ratio)) if ratio.size else np.nan,
            frac_ratio_gt1=float(np.nanmean(ratio > 1.0)) if ratio.size else np.nan))
    return pd.DataFrame(rows)


def a5_seed(win, loss_col):
    """A5: 窓ごとの median|Δx| と残差の Spearman 相関 [spec §5]。"""
    w = win.dropna(subset=["med_abs_dx", loss_col])
    if len(w) < 3:
        return dict(a5_rho=np.nan, a5_n_window=len(w))
    return dict(a5_rho=spearman(w["med_abs_dx"].values, w[loss_col].values),
                a5_n_window=int(len(w)))


def a6_seed(win):
    """A6: 天井の記述 (report tier のみ。spec §5 A6 により判定を置かない)。"""
    w = win.dropna(subset=["ratio_med", "mean_dx"])
    rho = (spearman(w["ratio_med"].values, w["mean_dx"].values)
           if len(w) >= 3 else np.nan)
    return dict(a6_rho_ratio_vs_edx=rho,
                a6_ratio_med=float(np.nanmedian(win["ratio_med"])),
                a6_ratio_q05=float(np.nanquantile(win["ratio_med"], 0.05)),
                a6_ratio_q95=float(np.nanquantile(win["ratio_med"], 0.95)),
                a6_frac_window_ratio_gt1=float(np.nanmean(win["ratio_med"] > 1.0)),
                a6_frac_unit_ratio_gt1=float(np.nanmean(win["frac_ratio_gt1"])))


# ------------------------------------------------------------------ A3 (snapshot)

def support_patterns():
    """condA の入力サポート: 自由 5 bit の 32 パターン (`envs.SCREnv.patterns` と同型)。"""
    k = M_DIM - F_DIM
    return ((np.arange(2 ** k)[:, None] >> np.arange(k)) & 1).astype(np.float64)


def snapshot_gate(snap):
    """snapshot から 32 パターン厳密ゲート率 p̂ を再計算する [追補 §7]。

    式は `src/ratchet_log.py` の `exact_record` と同じ (std なので centering 補正なし、
    x_in = X)。あわせて [[xr分解と生存機構_0822]] §9 の閉形式

        dead(flip) <=> flip·w[:15] + b + Σ_{j∈free} max(w_j, 0) <= 0

    を計算し、32 パターン列挙の max と一致することを検算する (S0)。free bit は {0,1}
    なので、上式の右辺が pre の 32 パターン最大値そのものになる。
    """
    W = snap["net"]["W"].double().numpy()
    b = snap["net"]["b"].double().numpy()
    flip = snap["env"]["flip_state"].double().numpy()
    pat = support_patterns()
    pre, p_hat, closed = [], [], []
    for r in range(W.shape[0]):
        X = np.concatenate([np.broadcast_to(flip[r], (pat.shape[0], F_DIM)), pat], axis=1)
        P = X @ W[r].T + b[r]                          # [32,h]
        pre.append(P)
        p_hat.append((P > 0).mean(axis=0))
        closed.append(flip[r] @ W[r][:, :F_DIM].T + b[r]
                      + np.maximum(W[r][:, F_DIM:], 0.0).sum(axis=1))
    pre, p_hat, closed = np.stack(pre), np.stack(p_hat), np.stack(closed)
    return dict(W=W, b=b, pre=pre, p_hat=p_hat, closed=closed,
                enum_max=pre.max(axis=1),
                max_closed_err=float(np.abs(closed - pre.max(axis=1)).max()))


def a3_from_snapshot(gate):
    """A3: 生存者は整列していない [spec §5]。

    alive/dead は **snapshot 自身**から再計算する (S5: logs 側の p̂ を使うと別 run の
    seed 対応を仮定することになる)。alive = **p̂ > 0**（発火中。追補 §1・§7 で母集団の
    ゲートを p̂ > 0 に統一した。閉形式 dead <=> max_pre <= 0 の裏返しでもある）。
    """
    W, p_hat_all, pre_all = gate["W"], gate["p_hat"], gate["pre"]
    rows, pairs = [], []
    for r in range(W.shape[0]):
        pre, p_hat = pre_all[r], p_hat_all[r]
        alive = p_hat > 0
        Wn = W[r] / np.maximum(np.linalg.norm(W[r], axis=1, keepdims=True), 1e-300)
        C = Wn @ Wn.T
        out = dict(seed=r, n_alive=int(alive.sum()), n_dead=int((~alive).sum()))
        for lab, sel in (("alive", alive), ("dead", ~alive), ("all", np.ones_like(alive))):
            idx = np.flatnonzero(sel)
            if idx.size < 2:
                out.update({f"a3_signed_{lab}": np.nan, f"a3_abs_{lab}": np.nan,
                            f"a3_effrank_w_{lab}": np.nan, f"a3_effrank_a_{lab}": np.nan})
                continue
            iu = np.triu_indices(idx.size, k=1)
            c = C[np.ix_(idx, idx)][iu]
            out[f"a3_signed_{lab}"] = float(c.mean())
            out[f"a3_abs_{lab}"] = float(np.abs(c).mean())
            out[f"a3_effrank_w_{lab}"] = eff_rank(W[r][idx])
            out[f"a3_effrank_a_{lab}"] = eff_rank(np.maximum(pre[:, idx], 0.0))
            if lab in ("alive", "dead"):
                pairs.append(pd.DataFrame(dict(seed=r, group=lab, cos=c)))
        rows.append(out)
    return pd.DataFrame(rows), pd.concat(pairs, ignore_index=True)


def eff_rank(A):
    """エントロピー実効ランク exp(-Σ p log p), p = s/Σs (`src/lop_metrics.py` と同式)。"""
    s = np.linalg.svd(A, compute_uv=False)
    tot = s.sum()
    if not np.isfinite(tot) or tot <= 0:
        return float("nan")
    p = s / tot
    return float(np.exp(-(p * np.log(np.maximum(p, 1e-12))).sum()))


# ------------------------------------------------------------------ bootstrap

def boot_mean(rng, vec, B=BOOT_N):
    """seed クラスタ bootstrap (mean 版): seed ごとの値を復元抽出して平均。"""
    v = np.asarray(vec, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan, np.nan, np.nan
    bs = v[rng.integers(0, v.size, (B, v.size))].mean(axis=1)
    return float(v.mean()), float(np.quantile(bs, .025)), float(np.quantile(bs, .975))


def boot_pooled(rng, num, den, B=BOOT_N):
    """seed クラスタ bootstrap (pooled 版・**主**): seed を復元抽出し、選ばれた seed の
    分子・分母を合算してから比を作り直す。「ユニットはプール」の字義 (spec §7)。"""
    num, den = np.asarray(num, float), np.asarray(den, float)
    m = np.isfinite(num) & np.isfinite(den) & (den > 0)
    num, den = num[m], den[m]
    if num.size == 0:
        return np.nan, np.nan, np.nan
    idx = rng.integers(0, num.size, (B, num.size))
    bs = num[idx].sum(axis=1) / den[idx].sum(axis=1)
    return float(num.sum() / den.sum()), float(np.quantile(bs, .025)), \
        float(np.quantile(bs, .975))


def contains(ci, lo, hi):
    return bool(np.isfinite(ci[0]) and np.isfinite(ci[1]) and ci[0] >= lo and ci[1] <= hi)


# ------------------------------------------------------------------ 判定

def judge(df, tot, haz, a3, rng, population, loss_col):
    V = []

    def add(id, tier, statistic, value, ci=(np.nan, np.nan), criterion="", verdict="—",
            note=""):
        V.append(dict(id=id, tier=tier, statistic=statistic, value=value,
                      ci_lo=ci[0], ci_hi=ci[1], criterion=criterion, verdict=verdict,
                      note=note))

    # ---------------- A1
    ok_a = tot["a1_pairs"] == T_A1A_PAIRS
    add("A1-a", "repro", "非境界・off_prev ペア数", tot["a1_pairs"],
        criterion=f"= {T_A1A_PAIRS} に厳密一致",
        verdict="PASS" if ok_a else "FAIL",
        note=f"差 {tot['a1_pairs'] - T_A1A_PAIRS:+d}。母集団 = 全ての非境界隣接ペア "
             f"(Δstep を問わない)")
    add("A1-b", "prereg", "同ペアでの復活数", tot["a1_revive"], criterion="= 0 (厳密)",
        verdict="PASS" if tot["a1_revive"] == 0 else "FAIL")
    add("A1-c", "prereg", "同ペアでの max|Δcos_u_mu|", tot["a1_max_dcos"],
        criterion="= 0.0 (float32 厳密)",
        verdict="PASS" if tot["a1_max_dcos"] == 0.0 else "FAIL")
    add("A1-c'", "report", "同ペアでの max|Δ‖w‖|", tot["a1_max_dwnorm"],
        criterion="(判定なし。[[xr分解と生存機構_0822]] §14 が併記していた量)",
        note="spec §5 A1-c は Δcos のみを判定対象にしている")
    # A1-d は追補 §9 で prereg -> repro に格下げした。spec 執筆時に
    # [[xr分解と生存機構_0822]] §14 を読んでおり 11.54% が文脈に入っていたため、
    # prereg を名乗れない。格は落ちるが判定は「厳密一致」になる。
    frac = tot["a1d_revive"] / tot["a1d_pairs"] if tot["a1d_pairs"] else np.nan
    ok_dp = tot["a1d_pairs"] == T_A1D_PAIRS
    add("A1-d", "repro", "境界・off_prev ペア数", tot["a1d_pairs"],
        criterion=f"= {T_A1D_PAIRS} に厳密一致", verdict="PASS" if ok_dp else "FAIL",
        note=f"差 {tot['a1d_pairs'] - T_A1D_PAIRS:+d}")
    ok_dr = tot["a1d_revive"] == T_A1D_REVIVE
    add("A1-d", "repro", "同ペアでの復活数", tot["a1d_revive"],
        criterion=f"= {T_A1D_REVIVE} に厳密一致 (11.54%)",
        verdict="PASS" if ok_dr else "FAIL",
        note=f"差 {tot['a1d_revive'] - T_A1D_REVIVE:+d}、実測 {frac:.4f}。"
             f"> 0 なので「復活は境界でのみ起きる」")

    # ---------------- A4
    hb, blocks = hazard_levels(haz)
    lo, hi = float(blocks.hazard_state.min()), float(blocks.hazard_state.max())
    ok4 = (abs(lo - T_A4_LO) < T_A4_TOL) and (abs(hi - T_A4_HI) < T_A4_TOL)
    bnote = ("集約粒度は `q3_boundary_race.hazard_table` と同じ**ブロック単位** "
             "(境界 11 本ずつ・`range(1, 100, 11)`)。10 seed を合算した hazard_state")
    add("A4-a", "repro", "境界あたりハザードのレンジ (min・ブロック)", lo,
        criterion=f"{T_A4_LO}–{T_A4_HI} が再現 (両端 ±{T_A4_TOL})",
        verdict="PASS" if ok4 else "FAIL", note=f"max = {hi:.4f}。{bnote}")
    add("A4-a", "repro", "境界あたりハザードのレンジ (max・ブロック)", hi,
        criterion=f"{T_A4_LO}–{T_A4_HI} が再現 (両端 ±{T_A4_TOL})",
        verdict="PASS" if ok4 else "FAIL", note=f"min = {lo:.4f}。{bnote}")
    plo, phi = float(hb.hazard_state.min()), float(hb.hazard_state.max())
    add("A4-a'", "report", "同・境界ごと (99 点) のレンジ", phi - plo,
        criterion="(判定なし。集約粒度の違いを可視化する)",
        note=f"{plo:.4f} – {phi:.4f}。ブロック集約より広いのは 1 境界あたりの "
             f"alive が ~1000 本しかないため。**判定に使うのはブロック版**")
    hs = haz.groupby("seed", as_index=False).agg(
        n_alive=("n_alive", "sum"), n_freeze_win=("n_freeze_win", "sum"))
    m4 = boot_mean(rng, (hs.n_freeze_win / hs.n_alive).values)
    p4 = boot_pooled(rng, hs.n_freeze_win.values, hs.n_alive.values)
    add("A4-b", "prereg", "境界あたりハザード (seed クラスタ)", m4[0], m4[1:],
        criterion="(新規に CI を付す。判定閾値は置かない)",
        note=f"推論の単位は seed なので mean 版を主に置く (追補 §5)。"
             f"pooled 版 {p4[0]:.4f} [{p4[1]:.4f}, {p4[2]:.4f}]")

    # ---------------- A2
    # A2-a は **pooled**。旧値 0.4971 / 0.5008 が pooled なので、同じ推定量でなければ
    # 再現の検定にならない (追補 §5)。
    pp = tot["a2_n_plus"] / tot["a2_n"]
    pm = tot["a2_n_minus"] / tot["a2_n"]
    ok2a = (abs(pp - T_A2A_PLUS) <= 0.005) and (abs(pm - T_A2A_MINUS) <= 0.005)
    add("A2-a", "repro", "符号比 Δx>0 の割合 (pooled)", pp,
        criterion=f"{T_A2A_PLUS} を ±0.5pt で再現", verdict="PASS" if ok2a else "FAIL",
        note=f"Δx<0 {pm:.4f} (目標 {T_A2A_MINUS}), Δx=0 "
             f"{tot['a2_n_zero'] / tot['a2_n']:.4f}, n={tot['a2_n']} "
             f"(旧 n={T_S6_ON})。母集団 = {population}")
    add("A2-a", "repro", "符号比 Δx<0 の割合 (pooled)", pm,
        criterion=f"{T_A2A_MINUS} を ±0.5pt で再現", verdict="PASS" if ok2a else "FAIL",
        note=f"Δx>0 {pp:.4f}")
    vis = tot["a2_n_visit"] / tot["a2_n_unit"]
    ok2b = abs(vis - T_A2B_VISIT) <= 0.010
    add("A2-b", "repro", "+µ 側訪問率 (x_max > 0 のユニット割合)", vis,
        criterion=f"{T_A2B_VISIT} を ±1.0pt で再現", verdict="PASS" if ok2b else "FAIL",
        note=f"{tot['a2_n_visit']}/{tot['a2_n_unit']} ユニット。分母はペアではなく"
             f"ユニット (記録点ベース)")
    # A2-c は **mean**。seed が独立実現 = 推論の単位で、pooled は「発火中ユニット数」
    # というノイズ変数で seed を重み付けてしまう (追補 §5)。
    m2, l2, h2 = boot_mean(rng, df.a2_p_plus.values)
    p2 = boot_pooled(rng, df.a2_n_plus.values, df.a2_n.values)
    in_eq = contains((l2, h2), *EQ_A2)
    has_half = bool(l2 <= 0.5 <= h2)
    v2 = "PASS" if in_eq else ("INCONCLUSIVE" if has_half else "対称を撤回")
    add("A2-c", "prereg", "符号分率 p₊ (seed クラスタ bootstrap・mean)", m2, (l2, h2),
        criterion=f"CI ⊂ [{EQ_A2[0]}, {EQ_A2[1]}] -> PASS / 0.5 を含む -> INCONCLUSIVE "
                  f"/ 含まない -> 対称を撤回",
        verdict=v2,
        note=f"pooled 版 {p2[0]:.4f} [{p2[1]:.4f}, {p2[2]:.4f}] (A2-a と同じ推定量)")
    add("A2-c'", "report", "推定量の乖離 (pooled − mean)", p2[0] - m2,
        criterion="(判定なし・追補 §5)",
        note="大きければ「seed 間で対称性が揃っていない」という別の所見になる。"
             f"seed 別 p₊ の SD {np.nanstd(df.a2_p_plus.values, ddof=1):.4f}、"
             f"seed 別ペア数 {int(df.a2_n.min())}–{int(df.a2_n.max())}")
    add("A2-d", "prereg", "終端 (最終記録点) での復活数", tot["a2d_terminal_revive"],
        criterion="= 0",
        verdict="PASS" if tot["a2d_terminal_revive"] == 0 else "FAIL",
        note="最終隣接ペアで p̂=0 -> p̂>0 になったユニット数")

    # ---------------- A5
    p5, l5, h5 = boot_mean(rng, df.a5_rho.values)
    ok5 = bool(np.isfinite(l5) and l5 > A5_FLOOR)
    add("A5-a", "prereg", f"窓ごとの median|Δx| と {loss_col} の Spearman", p5, (l5, h5),
        criterion=f"CI 下限 > {A5_FLOOR} -> PASS", verdict="PASS" if ok5 else "FAIL",
        note=f"seed ごとに窓 (タスク) 上で相関を取り、seed クラスタで CI。"
             f"窓数/seed = {int(np.nanmedian(df.a5_n_window))}。旧値 ≈0.96 は出所不明の"
             f"ため repro tier を与えていない (spec §5 A5)")

    # ---------------- A3
    p3, l3, h3 = boot_mean(rng, a3.a3_signed_alive.values)
    ok3 = contains((l3, h3), *EQ_A3)
    add("A3-a", "prereg", "生存ユニット間の符号付きペア cos の平均", p3, (l3, h3),
        criterion=f"CI ⊂ [{EQ_A3[0]}, {EQ_A3[1]}] -> PASS (整列していない)",
        verdict="PASS" if ok3 else "FAIL",
        note=f"alive 本数 seed 平均 {a3.n_alive.mean():.1f} (旧 {OLD['a3_n_alive']})。"
             f"旧値 (判定に使わない) 符号付き +{OLD['a3_alive_signed']}")
    d3 = boot_mean(rng, a3.a3_signed_dead.values)
    diff = boot_mean(rng, (a3.a3_signed_alive - a3.a3_signed_dead).values)
    add("A3-b", "prereg", "死亡ユニット間の符号付きペア cos の平均", d3[0], d3[1:],
        criterion="生存 < 死亡 が CI で分離するか",
        verdict="分離" if (np.isfinite(diff[2]) and diff[2] < 0) else "非分離",
        note=f"差 (生存 − 死亡) {diff[0]:+.4f} [{diff[1]:+.4f}, {diff[2]:+.4f}]。"
             f"旧値 (判定に使わない) 死亡 +{OLD['a3_dead_signed']}")
    for lab, key, old in (("生存", "alive", OLD["a3_eff_rank_a"]),
                          ("死亡", "dead", None), ("全体", "all", None)):
        ea = boot_mean(rng, a3[f"a3_effrank_a_{key}"].values)
        ew = boot_mean(rng, a3[f"a3_effrank_w_{key}"].values)
        anote = ("旧値 %.2f" % old) if old else ""
        if key == "dead":
            anote = ("**定義上 NaN**。dead は p̂=0 = 32 パターンすべてでゲートが閉じて"
                     "いるので、活性行列が恒等的にゼロで特異値が定義できない")
        elif key == "all":
            anote = ("生存と一致するのは定義どおり — dead の列が全ゼロなので"
                     "特異値に寄与しない")
        add("A3-c", "report", f"eff_rank 活性側 [{lab}]", ea[0], ea[1:],
            criterion="(判定なし)", note=anote)
        add("A3-c", "report", f"eff_rank 重み側 [{lab}]", ew[0], ew[1:],
            criterion="(判定なし)",
            note=("旧値 %.2f" % OLD["a3_eff_rank_w"]) if key == "alive" else "")
    for lab, key in (("生存", "alive"), ("死亡", "dead")):
        ab = boot_mean(rng, a3[f"a3_abs_{key}"].values)
        add("A3-c", "report", f"ペア |cos| の平均 [{lab}]", ab[0], ab[1:],
            criterion="(判定なし)",
            note=f"旧値 {OLD['a3_alive_abs'] if key == 'alive' else OLD['a3_dead_abs']}")

    # ---------------- A6 (report のみ)
    for k, lab in (("a6_ratio_med", "|F_self|/|F_rest| の窓中央値"),
                   ("a6_frac_window_ratio_gt1", "比 > 1 の窓の割合"),
                   ("a6_frac_unit_ratio_gt1", "比 > 1 の (窓,ユニット) 割合"),
                   ("a6_rho_ratio_vs_edx", "比と E[Δx] の Spearman (窓上)")):
        m6 = boot_mean(rng, df[k].values)
        add("A6", "report", lab, m6[0], m6[1:], criterion="(判定なし・spec §5 A6)",
            note="因果の向き・可動域・符号不一致の 3 点により判定を置かない")
    return pd.DataFrame(V)


# ------------------------------------------------------------------ 図

def make_figures(outdir, df, win, pairs, haz):
    fig_dir = os.path.join(outdir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # 1) 符号分率の seed 別分布
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.bar(df.seed.astype(str), df.a2_p_plus, color="tab:blue", alpha=.8)
    ax.axhline(0.5, color="k", lw=1)
    ax.axhspan(EQ_A2[0], EQ_A2[1], color="tab:green", alpha=.15,
               label=f"等価幅 [{EQ_A2[0]}, {EQ_A2[1]}]")
    ax.axhline(T_A2A_PLUS, color="tab:red", ls="--", lw=1.2, label=f"旧値 {T_A2A_PLUS}")
    ax.set_ylim(0.40, 0.60)
    ax.set_xlabel("seed")
    ax.set_ylabel("p₊ = P(Δx > 0)")
    ax.set_title("A2-c: 符号分率の seed 別分布")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig_sign_fraction_by_seed.png"), dpi=140)
    plt.close(fig)

    # 2) median|Δx| vs eval_loss (窓)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    sc = ax.scatter(win.loss_mean, win.med_abs_dx, c=win.seed, cmap="tab10", s=8,
                    alpha=.7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("窓内 eval_loss_exact の平均")
    ax.set_ylabel("窓ごとの median|Δx| (alive・Δstep=1000)")
    ax.set_title("A5: 揺れの大きさは残差に比例するか")
    fig.colorbar(sc, ax=ax, label="seed")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig_absdx_vs_loss.png"), dpi=140)
    plt.close(fig)

    # 3) 生存/死亡のペア cos 分布
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    bins = np.linspace(-1, 1, 61)
    for g, c, lab in (("alive", "tab:blue", "生存"), ("dead", "tab:orange", "死亡")):
        v = pairs[pairs.group == g].cos.values
        if v.size:
            ax.hist(v, bins=bins, alpha=.55, color=c, density=True,
                    label=f"{lab} (n={v.size}, 符号付き平均={v.mean():+.3f})")
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("重み行ベクトルのペア cos")
    ax.set_ylabel("密度")
    ax.set_title("A3: 生存者は整列していないか (snapshot step=500k)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig_pair_cos_alive_dead.png"), dpi=140)
    plt.close(fig)

    # 4) 境界 index あたりのハザード (A4)
    hb, blocks = hazard_levels(haz)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(hb.b_index, hb.hazard_state, "o-", ms=3, lw=1, color="tab:purple",
            alpha=.45, label="境界ごと (99 点・判定に使わない)")
    bx = [int(s.split("-")[0]) + 5 for s in blocks.block]
    ax.plot(bx, blocks.hazard_state, "s-", ms=6, lw=2, color="tab:red",
            label="ブロック (11 境界ずつ・判定に使う)")
    ax.axhline(T_A4_LO, color="tab:gray", ls="--", lw=1, label=f"旧レンジ {T_A4_LO}")
    ax.axhline(T_A4_HI, color="tab:gray", ls=":", lw=1, label=f"旧レンジ {T_A4_HI}")
    ax.set_xlabel("境界 index")
    ax.set_ylabel("hazard_state")
    ax.set_title("A4: 境界あたりハザードは一定か")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig_hazard_by_boundary.png"), dpi=140)
    plt.close(fig)
    return fig_dir


# ------------------------------------------------------------------ summary.md

def _md(df, fmt=".6f"):
    cols = list(df.columns)
    def f(v):
        if isinstance(v, (float, np.floating)):
            return format(v, fmt) if np.isfinite(v) else ""
        return str(v)
    out = ["| " + " | ".join(cols) + " |",
           "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(f(r[c]) for c in cols) + " |")
    return "\n".join(out)


def write_summary(outdir, V, df, s0, s2, s3, s6, s6_tot, meta, args):
    show = ["id", "tier", "statistic", "value", "ci_lo", "ci_hi", "criterion",
            "verdict", "note"]
    L = ["# xr_recheck_0827: xr 分解チャット結果の再現登録", "",
         f"仕様: `specs/spec_xr_recheck_0827.md` (再現登録)。生成: `{meta['date']}`、"
         f"git `{meta['git_hash']}`。**新規走行なし** — "
         f"`results/ratchet_log_0819/logs/seed*.npz` と "
         f"`results/posreset_0819/snapshots/A_w100_cont_step500000.pt` の再解析。", "",
         "## 0. 格の宣言 (spec §1)", "",
         "対象の数値は 8/22 のチャットで既に見ているため、真の事前登録は原理的に"
         "不可能である。`tier` 列で格を分ける — `repro` は旧値があり許容幅を凍結した"
         "もの、`prereg` は旧値に無く真の事前登録として成立するもの、`report` は判定を"
         "置かない記述。**論文に引用するときは tier を落とさないこと。**", "",
         "## 1. サニティ", "",
         f"**S0** (実行前の前提ゲート): "
         f"{'PASS' if bool(s0.ok.all()) else '**FAIL**'}", "",
         _md(s0[["check", "ok", "detail"]]), "",
         f"**S1**: `OMP_NUM_THREADS={meta['omp_num_threads']}`、"
         f"python {meta['python']} / numpy {meta['numpy']} / pandas {meta['pandas']} / "
         f"matplotlib {meta['matplotlib']} / torch {meta['torch']} / "
         f"{meta['platform']}。Spearman は `src/figures_teachw.spearman` "
         f"(venv に scipy が無いため repo 内実装を再利用)。", "",
         f"**S2** (境界を flip_state から機械的に決定): "
         f"{'PASS' if bool(s2.ok.all()) else '**FAIL**'} — "
         f"全 {len(s2)} seed で 99/99 が step ≡ 0 (mod 10⁴) かつ隣接。", "",
         _md(s2[["seed", "n_boundary_pairs", "n_left_aligned", "n_adjacent", "ok"]]), "",
         f"**S3** (mu_norm がタスク内で厳密に一定): "
         f"{'PASS' if bool(s3.ok.all()) else '**FAIL**'}", "",
         _md(s3[["seed", "n_task", "n_task_with_multiple_mu", "ok"]]), "",
         "**S4**: 2 回実行して `verdict.csv` が byte 一致することを確認する "
         "(実行手順は下記 §6)。", "",
         f"**S6** (母集団定義の加法恒等式・ゼロ許容 [追補 §1.2]): "
         f"{s6_tot['total']} = {s6_tot['on']} + {s6_tot['off']} "
         f"(目標 {T_S6_TOTAL} = {T_S6_ON} + {T_S6_OFF})", "",
         "Δstep=1000・タスク内の全ペアが「始点 p̂>0」と「始点 p̂=0」に過不足なく分かれる。"
         "右辺の 2 項は [[xr分解と生存機構_0822]] §19 (撹拌の母集団) と §14 "
         "(1000 step ペアで復活 0) が別々に報告している値であり、加法的に閉じることが"
         "母集団定義の検算になる。参考: 同じ母集団を `p̂ >= 0.05` で切ると "
         f"{s6_tot['alive_tau']} 件で、{T_S6_ON - s6_tot['alive_tau']} 件ずれる。", "",
         _md(s6[["seed", "s6_total", "s6_on", "s6_off", "s6_alive_tau"]], ".0f"), "",
         "**S5**: A3 の snapshot (`posreset_0819` cont アーム) と A1/A2/A4/A5/A6 の "
         "logs (`ratchet_log_0819`) は**別 run** である。本 summary は A3 の alive/dead を"
         "snapshot 自身の 32 パターン厳密ゲート率から再計算しており、logs 側の p̂ を"
         "参照していない。**A3 の結論を A1/A2 と同一個体についての主張として読まないこと。**",
         "", "## 2. 判定 (A1–A5)", "",
         _md(V[V.id.str.startswith(("A1", "A2", "A3", "A4", "A5"))][show]), "",
         "## 3. A6 の報告 (判定なし)", "",
         _md(V[V.id == "A6"][["statistic", "value", "ci_lo", "ci_hi", "note"]]), "",
         "spec §5 A6 の決定により、天井については判定を置かない。因果の向きが観察では"
         "決まらないこと、比の可動域が 5% しかないこと、`E[F_gate]` と `E[Δx]` の符号"
         "不一致が未解決であることの 3 点による。**この決定に伴い v3 柱2 から §20 由来の"
         "記述を降ろし、天井の記述は A1-d に置き換える。**", "",
         "## 4. seed ごとの値", "", _md(df, ".6f"), "",
         "## 5. 逸脱・留保", "",
         "すべて `specs/spec_xr_recheck_0827_addendum.md` に**結果を見る前に**固定した。", "",
         "1. **A2/A5/A6 の母集団は親 spec §4 の字義では判定が壊れるため確定させた。** "
         "[[xr分解と生存機構_0822]] §19 の「タスク内・発火中限定・Δstep=1000、"
         f"n = 149,492」に合わせ、本走の母集団は `{args.pair_population}` "
         "(非境界 かつ Δstep == 1000 かつ ペア始点で **p̂ > 0**) とした。"
         "**閾値ではなく離散の非ゼロ**である — p̂ = k/32 なので p̂>0 は k>=1 を意味し、"
         "`p̂ >= 0.05` (k>=2) にすると 30,446 件ずれて A2-a は定義の食い違いだけで "
         "FAIL する。S6 で加法恒等式を検算した。",
         "2. **A2-b の分母はユニットである。** 「+µ 側訪問率」は x_max > 0 となる"
         "ユニットの割合 (10 seed × 100 unit = 1000 本) であって、ペアの割合ではない。",
         "3. **推定量を項目ごとに分けた。** A2-a (repro) は **pooled** — 旧値 0.4971 / "
         "0.5008 が pooled なので、同じ推定量でなければ再現の検定にならない。"
         "A2-c (prereg・主判定) は **mean** — seed が独立実現 = 推論の単位であり、"
         "pooled は「発火中ユニット数」というノイズ変数で seed を重み付けてしまう。"
         "両者の乖離は A2-c' 行に report tier で出した。",
         "4. **A4 は `q3_boundary_race.py` をそのまま実行したものではない。** 判定規則を"
         "コード上一本化するため `freeze_events_all` を当該モジュールから import して"
         "`hazard_state` を再計算している (分母 = 境界 B で alive、分子 = 窓 [B+1, B+100] に"
         "同一規則の消灯イベント)。レンジの許容幅は親 spec に無いため両端 ±0.0005 "
         "(3 桁表示の丸め半幅) を凍結した。スクリプト自体の 2 回実行と byte 一致確認は別手順。",
         "4b. **A4-a の集約粒度を初回実行後に訂正した (逸脱)。** 初回は境界ごと (99 点) の"
         "レンジを取り 0.0323–0.2647 で FAIL したが、これは実装の読み違いだった。"
         "親 spec の 0.099–0.186 は `q3_boundary_race.hazard_table` が返す**ブロック単位** "
         "(境界 11 本ずつ・`range(1, 100, 11)`) のレンジであり、元コードに合わせて"
         "取り直した。**訂正したのは統計量の集約粒度であって判定基準ではない** "
         "(許容幅 ±0.0005 は据え置き)。境界ごとのレンジは A4-a' に report tier で残した。",
         "5. **A1-d を prereg から repro に格下げした。** [[命題リスト]] Q3 に旧値 "
         f"({T_A1D_PAIRS} 件中 {T_A1D_REVIVE} 件 = 11.54%) があり、親 spec 執筆時に "
         "[[xr分解と生存機構_0822]] §14 を読んで 11.54% が文脈に入っていた。"
         "**見た数値を prereg と名乗れない**ので格を落とし、代わりに判定を"
         "「厳密一致」に強めた。",
         "6. **A5 の旧値 ≈0.96 は出所不明のまま。** 親 spec の指示どおり repro tier を"
         f"与えず、新規測定として扱っている。窓の残差は `{args.a5_loss_agg}` で集約した。",
         "7. **親 spec §2.3 の「§7 は n=12・1 seed」は誤りだった。** §7 は見出しに "
         "「10 seed」と明記しており、12.0 / 88.0 / 100.0 は seed あたりの平均本数である "
         "(§17 の n=120 / 880 がその 10 倍で整合)。10 seed で seed クラスタ CI を付ける"
         "という設計は変わらず、§7 に CI が無い以上 A3-a の prereg tier も成立する。", "",
         "## 6. 出力と再実行", "",
         "```",
         "OMP_NUM_THREADS=1 .venv/bin/python -m analysis.xr_recheck.xr_recheck \\",
         f"    --outdir {os.path.relpath(outdir, ROOT)}",
         "```", "",
         "- `verdict.csv` — A1–A6 の全行 (`id, tier, statistic, value, ci_lo, ci_hi, "
         "criterion, verdict, note`)",
         "- `per_seed_metrics.csv` — seed ごとの全指標",
         "- `window_metrics.csv` — 窓 (タスク) ごとの median|Δx| / 残差 / 比",
         "- `hazard_by_boundary.csv` — A4 の境界別ハザード",
         "- `a3_snapshot.csv` / `a3_pair_cos.csv` — A3 の seed 別値とペア cos",
         "- `sanity.csv` — S0/S2/S3 の全行",
         "- `figures/fig_sign_fraction_by_seed.png`, `fig_absdx_vs_loss.png`, "
         "`fig_pair_cos_alive_dead.png`, `fig_hazard_by_boundary.png`", "",
         "## 7. スコープ・禁止事項 (spec §8 の転記)", "",
         "- スコープ: **condA・w100・T=10⁴・batch=1 限定**。condB へ外挿しない",
         "- **A2 の符号は 1000 step 積分の符号である** ([[xr分解と生存機構_0822]] §19 の"
         "留保の転記): 「Δstep=1000 の符号は 1000 step 積分の符号で 1 step の符号とは"
         "別物。x_max は記録点ベースなので未記録 step のピークは拾えていない」。"
         "A2 の PASS は「1 step の増分が対称」を意味しない",
         "- **κ = ‖w_free‖₁/‖w_free‖₂ は logs から復元できない** (full W が無い)。"
         "吸収不等式の命題検証は本 spec の対象外 (作業3 の領分)",
         "- A6 は報告のみ。ここから機構の深掘りに入らない",
         "- null 結果も PASS と同じ形式で報告する",
         "- **再現しなかった旧値は消さず**、vault に「再現せず・撤回」として格を落として残す",
         "- 追加したくなった項目は追補 (addendum) として別 commit で、**結果を見る前に**固定する",
         ""]
    p = os.path.join(outdir, "summary.md")
    with open(p, "w") as fh:
        fh.write("\n".join(L) + "\n")
    return p


# ------------------------------------------------------------------ main

def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def git_tracked(paths):
    try:
        out = subprocess.check_output(["git", "ls-files", "-z", "--"] + list(paths),
                                      cwd=ROOT, text=True)
        listed = {os.path.join(ROOT, p) for p in out.split("\0") if p}
    except Exception:
        return {p: False for p in paths}
    return {p: os.path.abspath(p) in listed for p in paths}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.join(ROOT, "results", "xr_recheck_0827"))
    ap.add_argument("--pair-population", default="on_bulk",
                    choices=["on_bulk", "alive_bulk", "any_bulk", "on_all", "any_all"],
                    help="A2/A5/A6 の撹拌ペア母集団。既定 (on_bulk = 非境界・"
                         "Δstep=1000・始点で p̂>0) が spec の判定に使う正")
    ap.add_argument("--a5-loss-agg", default="loss_mean",
                    choices=["loss_mean", "loss_first"],
                    help="A5 の窓ごとの残差の集約")
    args = ap.parse_args()

    t0 = time.time()
    rng = np.random.default_rng(BOOT_SEED)

    # ---- S0 (実行前の前提ゲート)
    s0_ok, s0, snap, gate = check_s0()
    print(s0.to_string(index=False), flush=True)
    if not s0_ok:
        raise SystemExit("[xr_recheck] S0 FAIL — 入力の前提が崩れている。中止 (§6 S0)")

    # ---- seed ごとの計算 (1 seed ずつ読み、要らなくなったら捨てる)
    rows, wins, hazs, s6s = [], [], [], []
    for p in seed_paths():
        d = load_seed(p)
        r = dict(run_id=str(d["run_id"]))
        r.update(a1_seed(d))
        r.update(a2_seed(d, args.pair_population))
        s6s.append(s6_counts(d))
        w = window_table(d, args.pair_population)
        r.update(a5_seed(w, args.a5_loss_agg))
        r.update(a6_seed(w))
        hazs.append(a4_seed(d))
        wins.append(w)
        rows.append(r)
        print(f"seed {r['seed']}: A1 pairs={r['a1_pairs']} revive={r['a1_revive']} "
              f"| A2 n={r['a2_n']} p+={r['a2_p_plus']:.4f} | A5 rho={r['a5_rho']:.4f}",
              flush=True)
        del d
    df = pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)
    win = pd.concat(wins, ignore_index=True)
    haz = pd.concat(hazs, ignore_index=True)

    # ---- S2 / S3 は step / flip_state / mu_norm だけで足りるので軽く読み直す
    light = []
    for p in seed_paths():
        with np.load(p) as z:
            light.append({k: z[k] for k in ("step", "flip_state", "mu_norm",
                                            "seed", "run_id")})
        light[-1]["step"] = light[-1]["step"].astype(np.int64)
    s2_ok, s2 = check_s2(light)
    s3_ok, s3 = check_s3(light)

    # ---- S6: 母集団定義の加法恒等式 [追補 §1.2]。ゼロ許容。
    s6 = pd.DataFrame(s6s).sort_values("seed").reset_index(drop=True)
    s6_tot = dict(total=int(s6.s6_total.sum()), on=int(s6.s6_on.sum()),
                  off=int(s6.s6_off.sum()), alive_tau=int(s6.s6_alive_tau.sum()))
    s6_ok = (s6_tot["total"] == T_S6_TOTAL and s6_tot["on"] == T_S6_ON
             and s6_tot["off"] == T_S6_OFF
             and s6_tot["on"] + s6_tot["off"] == s6_tot["total"])
    print(f"S2: {'PASS' if s2_ok else 'FAIL'} / S3: {'PASS' if s3_ok else 'FAIL'} / "
          f"S6: {'PASS' if s6_ok else 'FAIL'} "
          f"({s6_tot['total']} = {s6_tot['on']} + {s6_tot['off']})", flush=True)
    if not s2_ok:
        raise SystemExit("[xr_recheck] S2 FAIL — 境界の同定が壊れている。中止 (§6 S2)")
    if not s3_ok:
        raise SystemExit("[xr_recheck] S3 FAIL — A1 の前提が壊れている。中止 (§6 S3)")
    if not s6_ok:
        raise SystemExit(
            f"[xr_recheck] S6 FAIL — 母集団定義の加法恒等式が閉じない。中止 (追補 §1.2)。"
            f"実測 {s6_tot['total']} = {s6_tot['on']} + {s6_tot['off']}、"
            f"目標 {T_S6_TOTAL} = {T_S6_ON} + {T_S6_OFF}")

    # ---- A1 の中止条件を判定より先に見る (spec §9: A1 FAIL なら以降を計算しない)
    tot = dict(
        a1_pairs=int(df.a1_pairs.sum()), a1_revive=int(df.a1_revive.sum()),
        a1_max_dcos=float(df.a1_max_dcos.max()),
        a1_max_dwnorm=float(df.a1_max_dwnorm.max()),
        a1d_pairs=int(df.a1d_pairs.sum()), a1d_revive=int(df.a1d_revive.sum()),
        a2_n=int(df.a2_n.sum()), a2_n_plus=int(df.a2_n_plus.sum()),
        a2_n_minus=int(df.a2_n_minus.sum()), a2_n_zero=int(df.a2_n_zero.sum()),
        a2_n_unit=int(df.a2_n_unit.sum()), a2_n_visit=int(df.a2_n_visit.sum()),
        a2d_terminal_revive=int(df.a2d_terminal_revive.sum()))
    if tot["a1_revive"] != 0 or tot["a1_max_dcos"] != 0.0:
        raise SystemExit(
            f"[xr_recheck] A1 FAIL — 復活 {tot['a1_revive']} 件 / "
            f"max|Δcos| {tot['a1_max_dcos']:.3e}。spec §9 により以降を中止し、"
            f"v3 柱2 を作り直す")

    # ---- A3 (snapshot)
    a3, pairs = a3_from_snapshot(gate)

    V = judge(df, tot, haz, a3, rng, args.pair_population, args.a5_loss_agg)

    os.makedirs(args.outdir, exist_ok=True)
    V.to_csv(os.path.join(args.outdir, "verdict.csv"), index=False)
    df.to_csv(os.path.join(args.outdir, "per_seed_metrics.csv"), index=False)
    win.to_csv(os.path.join(args.outdir, "window_metrics.csv"), index=False)
    haz.to_csv(os.path.join(args.outdir, "hazard_by_boundary.csv"), index=False)
    a3.to_csv(os.path.join(args.outdir, "a3_snapshot.csv"), index=False)
    pairs.to_csv(os.path.join(args.outdir, "a3_pair_cos.csv"), index=False)
    sanity = pd.concat([s0.assign(block="S0"),
                        s2.assign(block="S2", check="境界の機械的同定",
                                  detail=s2.apply(
                                      lambda r: f"seed {r.seed}: "
                                                f"{r.n_boundary_pairs} 境界", axis=1)),
                        s3.assign(block="S3", check="mu_norm のタスク内不変",
                                  detail=s3.apply(
                                      lambda r: f"seed {r.seed}: "
                                                f"{r.n_task} タスク", axis=1)),
                        s6.assign(block="S6", check="母集団の加法恒等式", ok=True,
                                  detail=s6.apply(
                                      lambda r: f"seed {r.seed}: {r.s6_total} = "
                                                f"{r.s6_on} + {r.s6_off} "
                                                f"(p̂>=TAU なら {r.s6_alive_tau})",
                                      axis=1))],
                       ignore_index=True)[["block", "check", "ok", "detail"]]
    sanity.to_csv(os.path.join(args.outdir, "sanity.csv"), index=False)
    fig_dir = make_figures(args.outdir, df, win, pairs, haz)

    import torch
    meta = dict(date=time.strftime("%Y-%m-%d %H:%M:%S"), git_hash=git_hash(),
                spec="specs/spec_xr_recheck_0827.md",
                sources=[LOGDIR, SNAPSHOT], elapsed_sec=round(time.time() - t0, 1),
                omp_num_threads=os.environ.get("OMP_NUM_THREADS", "(未設定)"),
                python=platform.python_version(), numpy=np.__version__,
                pandas=pd.__version__, matplotlib=matplotlib.__version__,
                torch=torch.__version__, platform=platform.platform(),
                bootstrap_B=BOOT_N, bootstrap_seed=BOOT_SEED,
                pair_population=args.pair_population, a5_loss_agg=args.a5_loss_agg,
                n_seeds=int(len(df)), period=PERIOD,
                s0_pass=bool(s0_ok), s2_pass=bool(s2_ok), s3_pass=bool(s3_ok),
                s6_pass=bool(s6_ok), s6=s6_tot, totals=tot)
    with open(os.path.join(args.outdir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1, default=str, ensure_ascii=False)
    sp = write_summary(args.outdir, V, df, s0, s2, s3, s6, s6_tot, meta, args)

    print(V[["id", "tier", "statistic", "value", "ci_lo", "ci_hi",
             "verdict"]].to_string(index=False), flush=True)
    print(f"-> {args.outdir}/verdict.csv, {sp}, {fig_dir}/", flush=True)
    print("XR_RECHECK DONE", flush=True)
    return df, V


if __name__ == "__main__":
    main()
