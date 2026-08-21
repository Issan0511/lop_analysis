"""dead2path_0821: 死の二経路の分離 (再分類死 / 輸送死) [spec_dead2path_0821]。

  OMP_NUM_THREADS=1 .venv/bin/python -m analysis.dead2path.dead2path \
      [results/ratchet_log_0819] [--outdir results/dead2path_0821]

**再学習なし**。`results/ratchet_log_0819/logs/seed*.npz` だけを読み、p̂ の 0.05 下抜けを
「境界ペア (B, B+1), B ≡ 0 (mod 10⁴) で起きた = **再分類死**」と「それ以外の µ̂ 固定ペアで
起きた = **輸送死**」に分ける。分離が記録構造から一意に決まる理由は spec §2 の通りで、
隣接記録ペアは境界ペアか µ̂ 固定ペアのどちらかに排他的に属する。

判定は spec §5 の D0–D5 が唯一の正で、本モジュールはそれを実装するだけ。特に

  D0  前提ゲート (0.10 <= r <= 0.90 でなければ D1–D3 は解釈しない)
  D1  再分類死は輸送を伴わない (境界ペアの |Δ‖w‖| <= 同一窓内の非境界ペアの 95%点)
  D2  **主判定**: 輸送死のみ・µ̂ 固定ペアの増分のみで P3 の median PE を再計算
  D3  E1 の群別再計算 (self の符号安定率は 3 群とも 1.000 であるべき = 検算)

を先に読むこと。S3 (既出数値の再現) が通らなければ D0 以降は計算しない。

出力 (すべて --outdir の中): verdict.csv / summary.md / per_seed_metrics.csv /
events.csv / meta.json / figures/。
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
from src.common import ROOT, load_config, switch_steps                # noqa: E402
from src.figures_ratchet_log import (TAU, boot_ci, death_events,      # noqa: E402
                                     descent_windows, load_seeds,
                                     pe_permutation_null,
                                     per_seed_metrics, judge)

BOOT_N = 10000
BOOT_SEED = 20260821            # spec §7 の事前登録の抽選列
PERM_N = 1000                   # 符号置換 null の反復数 (P3 と同一)
D0_LO, D0_HI = 0.10, 0.90       # spec §5 D0 のゲート
MIN_PTS = 3                     # 符号安定率を出すのに要る発火中の記録点数 (E1 と同一)

# S3: ratchet_log_0819 verdict.csv の該当行 (spec §6 S3 に転記されている値)
S3_TARGETS = [
    ("P3", "median PE (死亡ユニット)", 0.39592990555641694, 1e-12),
    ("E1", "self 成分の符号安定率", 1.0, 1e-12),
    ("E1", "rest 成分の符号安定率", 0.8059758263149069, 1e-7),
    ("E1", "self/rest の |µ̂ 射影| 時間平均比", 0.6176014192580223, 1e-7),
]

plt.rcParams["font.family"] = ["Noto Sans CJK JP", "Noto Sans CJK TC",
                              "Noto Sans CJK KR", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ------------------------------------------------------------------ 記録ペアの分類 (§2)

def pair_classification(d, period, half_w):
    """隣接記録ペア j = (t_j, t_{j+1}) の種別。

    is_boundary[j]: flip_state がこのペアで変化した = **境界ペア**。境界インデックスは
      ハードコードせず flip_state の差分から機械的に決める (S2)。
    pair_win[j]: そのペアが属する境界窓の番号 (どの境界の ±half_w にも収まらなければ −1)。
      両端が同じ境界の窓に入っているときだけ番号を与える (D1 の窓内対照に使う)。
    """
    step, fs = d["step"], d["flip_state"]
    is_boundary = (np.abs(np.diff(fs, axis=0)) > 0).any(axis=1)        # [n-1]
    b = np.array(switch_steps(int(period), int(step[-1])))
    nb = np.abs(step[:, None] - b[None, :]).argmin(axis=1)             # [n]
    in_win = np.abs(step - b[nb]) <= half_w
    same = (nb[:-1] == nb[1:]) & in_win[:-1] & in_win[1:]
    pair_win = np.where(same, nb[:-1], -1)
    return is_boundary, pair_win, b


def check_s2(seeds, period):
    """S2: 全 seed で「flip が変化するペアの先頭が step ≡ 0 (mod period)」かつ隣接。"""
    rows, ok = [], True
    for d in seeds:
        step = d["step"]
        chg = (np.abs(np.diff(d["flip_state"], axis=0)) > 0).any(axis=1)
        left, right = step[:-1][chg], step[1:][chg]
        n_bnd = len(switch_steps(int(period), int(step[-1])))
        aligned = int((left % period == 0).sum())
        adj = int(((right - left) == 1).sum())
        good = (aligned == int(chg.sum()) == adj)
        ok &= good
        rows.append(dict(seed=int(d["seed"]), run_id=str(d["run_id"]),
                         n_flip_pairs=int(chg.sum()), n_boundaries_in_grid=n_bnd,
                         n_left_aligned=aligned, n_adjacent=adj, ok=good,
                         bad_steps=[int(s) for s in left[left % period != 0]][:5]))
    return ok, pd.DataFrame(rows)


# ------------------------------------------------------------------ 死亡イベント (§3)

def death_event_table(d, is_boundary, pair_win):
    """§3 の死亡イベント: 隣接ペアで p̂ が TAU を上から下へ跨ぐ全ての事象。

    ratchet_log の `death_events` (「下方クロス後 1 周期回復しない最初の 1 個」) とは
    **別の母集団**である点に注意。こちらは全クロスを列挙し、再点灯 / 恒久死のフラグを
    別に持たせる。P3 の再計算 (D2) は比較可能性のため ratchet_log 側の定義を使う。
    """
    step, p, cos, wn = d["step"], d["p_hat"], d["cos_u_mu"], d["w_norm"]
    n, h = p.shape
    below = p < TAU
    cross = below[1:] & ~below[:-1]                                    # [n-1,h]
    # 「以後 1M step まで p̂ >= TAU に戻らない」= 恒久死。suffix OR で一括判定
    alive_after = np.zeros_like(below)
    alive_after[:-1] = np.logical_or.accumulate((~below)[::-1], axis=0)[::-1][1:]
    js, us = np.nonzero(cross)
    return pd.DataFrame(dict(
        seed=int(d["seed"]), run_id=str(d["run_id"]), unit=us.astype(int),
        pair_index=js.astype(int), step_prev=step[js], step_post=step[js + 1],
        path=np.where(is_boundary[js], "reclass", "transport"),
        pair_win=pair_win[js],
        reignited=alive_after[js + 1, us],
        permanent=~alive_after[js + 1, us],
        p_hat_prev=p[js, us].astype(np.float64),
        p_hat_post=p[js + 1, us].astype(np.float64),
        d_cos=(cos[js + 1, us] - cos[js, us]).astype(np.float64),
        d_wnorm=(wn[js + 1, us] - wn[js, us]).astype(np.float64),
        v=d["v"][js, us].astype(np.float64),
        w_norm_prev=wn[js, us].astype(np.float64)))


def unit_groups(ev, h):
    """ユニット単位の群ラベル = 恒久死をもたらした死亡イベントの経路 [§3]。

    恒久死イベントはユニットあたり高々 1 個 (それ以降回復しないため)。無ければ alive。
    """
    lab = np.array(["alive"] * h, dtype=object)
    idx = np.full(h, -1, dtype=int)
    perm = ev[ev.permanent]
    for u, path, j in zip(perm.unit.values, perm.path.values, perm.pair_index.values):
        lab[u], idx[u] = path, j
    return lab, idx


# ------------------------------------------------------------------ D1

def d1_wnorm_contrast(d, ev, is_boundary, pair_win, alive_only=False):
    """D1: 再分類死の境界ペアでの |Δ‖w‖| と、同一窓内の非境界ペアの |Δ‖w‖| 分布。

    対照は「同じユニット・同じ境界窓の、境界ペア以外の隣接ペア」。窓内は全て 1 step
    刻みなので、これは『1 step 分の SGD がどれだけ ‖w‖ を動かすか』の実測上限になる
    (§4-1: 対照が実データの中にある)。

    alive_only=True では対照を「左端で p̂ >= TAU のペア」に絞る。死亡後のユニットは
    勾配が恒等的に 0 で ‖w‖ が動かないため、素の対照には構造的な 0 が混ざり、95%点を
    押し下げる (= D1 を通りにくくする) 向きに効く。事前登録の主判定は**字義どおりの
    素の対照**で、alive 版は副次報告 (§ 逸脱節)。
    """
    wn, p = d["w_norm"], d["p_hat"]
    dwn = np.abs(np.diff(wn, axis=0))                                  # [n-1,h]
    obs, ctrl, per_event = [], [], []
    sub = ev[(ev.path == "reclass") & (ev.pair_win >= 0)]
    for u, j, w in zip(sub.unit.values, sub.pair_index.values, sub.pair_win.values):
        m = (pair_win == w) & ~is_boundary
        if alive_only:
            m = m & (p[:-1, u] >= TAU)
        c = dwn[m, u].astype(np.float64)
        o = float(dwn[j, u])
        obs.append(o)
        ctrl.append(c)
        if c.size:
            per_event.append(float(o <= np.quantile(c, 0.95)))
    obs = np.array(obs)
    ctrl = np.concatenate(ctrl) if ctrl else np.zeros(0)
    return dict(n_event=int(obs.size), n_ctrl=int(ctrl.size),
                obs_median=float(np.median(obs)) if obs.size else np.nan,
                ctrl_q95=float(np.quantile(ctrl, 0.95)) if ctrl.size else np.nan,
                ctrl_median=float(np.median(ctrl)) if ctrl.size else np.nan,
                # batch=1 では「そのユニットのゲートが開いたサンプルが引かれた step」
                # でしか w_i は動かない。厳密 0 の割合を対照側と並べて出さないと、
                # 「境界では動かない」が経路の性質なのか batch=1 の疎さなのか読めない
                obs_frac_zero=float((obs == 0).mean()) if obs.size else np.nan,
                ctrl_frac_zero=float((ctrl == 0).mean()) if ctrl.size else np.nan,
                frac_event_below_own_q95=(float(np.mean(per_event))
                                          if per_event else np.nan))


# ------------------------------------------------------------------ D2

def pe_by_path(d, is_boundary):
    """D2: 経路別に PE を再計算する。

    降下窓の作り方は ratchet_log と同一 (`death_events` -> `descent_windows`) で、
    そこに 2 つのフィルタを掛ける:
      (i) 死亡そのものが µ̂ 固定ペアで起きた窓だけを「輸送死」として残す
      (ii) 窓内の増分のうち**境界ペアを跨ぐもの**を落とす (座標の張り替えであって輸送
           ではない、§4-3)
    返り値は経路ごとの PE 配列と増分列 (null 用)、および D4 用の内訳。
    """
    cos = d["cos_u_mu"]
    deaths = death_events(d, int(d["period"]))
    wins = descent_windows(d, deaths)
    out = {k: dict(pes=[], incs=[]) for k in ("transport", "reclass")}
    all_abs = bnd_abs = 0.0
    n_drop_short = 0
    for u, s0, c in wins:
        path = "reclass" if is_boundary[c - 1] else "transport"
        dc = np.diff(cos[s0:c + 1, u].astype(np.float64))
        bmask = is_boundary[s0:c]
        all_abs += float(np.abs(dc).sum())
        bnd_abs += float(np.abs(dc[bmask]).sum())
        keep = dc[~bmask]
        if keep.size < 2:                    # 増分が 1 本以下は PE 不定 (ratchet_log と同基準)
            n_drop_short += 1
            continue
        tot = np.abs(keep).sum()
        if tot <= 0:
            n_drop_short += 1
            continue
        out[path]["pes"].append(abs(keep.sum()) / tot)
        out[path]["incs"].append(keep)
    for k in out:
        out[k]["pes"] = np.array(out[k]["pes"])
    return out, dict(n_window_total=len(wins), n_drop_short=n_drop_short,
                     sum_abs_dcos=all_abs, sum_abs_dcos_boundary=bnd_abs)


# ------------------------------------------------------------------ D3

def e1_by_group(d, lab, perm_idx):
    """D3: self / rest の µ̂ 射影を輸送死 / 再分類死 / 生存の 3 群別に。

    指標の定義は ratchet_log の `e1_drive_decomposition` と逐語的に同じ (発火中 =
    p̂ >= TAU の記録点に限り、符号安定率 = max(正の割合, 負の割合)、比 = |F_self| と
    |F_rest| の時間平均の比)。違うのは**どの区間で測るか**だけ:

      死亡群 : その恒久死に対応する降下窓 [t_start, t_death] (ratchet_log と同じ作り方)
      生存群 : 全記録区間 (降下窓が定義できないため)。区間長が死亡群より長い点は
               §逸脱節に明記する。符号安定率は区間長に対して単調でない量なので
               直接の比較には注意が要る
    """
    p, Fs, Fr, cos = d["p_hat"], d["F_self"], d["F_rest"], d["cos_u_mu"]
    n, h = p.shape
    res = {g: dict(ratio=[], stab_s=[], stab_r=[]) for g in
           ("transport", "reclass", "alive")}
    sgn = lambda z: float(max((z > 0).mean(), (z < 0).mean()))
    for u in range(h):
        g = lab[u]
        if g == "alive":
            s0, c = 0, n - 1
        else:
            c = int(perm_idx[u]) + 1
            pos = np.flatnonzero(cos[:c + 1, u] > 0)
            if not pos.size:
                continue
            s0 = int(pos[-1])
            if c - s0 < 2:
                continue
        m = p[s0:c + 1, u] >= TAU
        if m.sum() < MIN_PTS:
            continue
        a = Fs[s0:c + 1, u][m].astype(np.float64)
        b = Fr[s0:c + 1, u][m].astype(np.float64)
        mb = np.abs(b).mean()
        if mb > 0:
            res[g]["ratio"].append(np.abs(a).mean() / mb)
        res[g]["stab_s"].append(sgn(a))
        res[g]["stab_r"].append(sgn(b))
    f = lambda v: float(np.median(v)) if len(v) else np.nan
    return {g: dict(n_unit=len(res[g]["stab_s"]), ratio=f(res[g]["ratio"]),
                    stab_self=f(res[g]["stab_s"]), stab_rest=f(res[g]["stab_r"]),
                    stab_self_min=(float(np.min(res[g]["stab_s"]))
                                   if res[g]["stab_s"] else np.nan))
            for g in res}


# ------------------------------------------------------------------ seed ごとの集計

def per_seed(seeds, period, half_w):
    rows, pe_pack, d1_pack, d1a_pack, ev_all, e1_pack = [], [], [], [], [], []
    for d in seeds:
        is_b, pw, _ = pair_classification(d, period, half_w)
        ev = death_event_table(d, is_b, pw)
        ev_all.append(ev)
        h = d["p_hat"].shape[1]
        lab, pidx = unit_groups(ev, h)
        pe, brk = pe_by_path(d, is_b)
        pe_pack.append(pe)
        d1 = d1_wnorm_contrast(d, ev, is_b, pw, alive_only=False)
        d1a = d1_wnorm_contrast(d, ev, is_b, pw, alive_only=True)
        d1_pack.append(d1)
        d1a_pack.append(d1a)
        e1 = e1_by_group(d, lab, pidx)
        e1_pack.append(e1)

        n_perm_unit = int((lab != "alive").sum())
        n_rec_unit = int((lab == "reclass").sum())
        wn, v = d["w_norm"], np.abs(d["v"])
        row = dict(seed=int(d["seed"]), run_id=str(d["run_id"]), n_unit=h,
                   n_event=len(ev), n_event_reclass=int((ev.path == "reclass").sum()),
                   n_event_transport=int((ev.path == "transport").sum()),
                   n_perm_unit=n_perm_unit, n_perm_reclass=n_rec_unit,
                   n_perm_transport=int((lab == "transport").sum()),
                   n_alive_unit=int((lab == "alive").sum()),
                   r_reclass=(n_rec_unit / n_perm_unit if n_perm_unit else np.nan),
                   PE_median_transport=(float(np.median(pe["transport"]["pes"]))
                                        if pe["transport"]["pes"].size else np.nan),
                   PE_median_reclass=(float(np.median(pe["reclass"]["pes"]))
                                      if pe["reclass"]["pes"].size else np.nan),
                   n_pe_transport=int(pe["transport"]["pes"].size),
                   n_pe_reclass=int(pe["reclass"]["pes"].size),
                   frac_abs_dcos_boundary=(brk["sum_abs_dcos_boundary"]
                                           / brk["sum_abs_dcos"]
                                           if brk["sum_abs_dcos"] > 0 else np.nan),
                   n_window_total=brk["n_window_total"],
                   n_window_dropped=brk["n_drop_short"],
                   d1_obs_median=d1["obs_median"], d1_ctrl_q95=d1["ctrl_q95"],
                   d1_frac_below=d1["frac_event_below_own_q95"],
                   d1_obs_frac_zero=d1["obs_frac_zero"],
                   d1_ctrl_frac_zero=d1["ctrl_frac_zero"],
                   d1a_obs_median=d1a["obs_median"], d1a_ctrl_q95=d1a["ctrl_q95"],
                   d1a_obs_frac_zero=d1a["obs_frac_zero"],
                   d1a_ctrl_frac_zero=d1a["ctrl_frac_zero"],
                   # D5 (探索・判定なし)
                   reignite_rate_reclass=(float(ev[ev.path == "reclass"].reignited.mean())
                                          if (ev.path == "reclass").any() else np.nan),
                   reignite_rate_transport=(float(ev[ev.path == "transport"].reignited.mean())
                                            if (ev.path == "transport").any() else np.nan),
                   p_hat_prev_med_reclass=(float(ev[ev.path == "reclass"].p_hat_prev.median())
                                           if (ev.path == "reclass").any() else np.nan),
                   p_hat_prev_med_transport=(float(ev[ev.path == "transport"].p_hat_prev.median())
                                             if (ev.path == "transport").any() else np.nan))
        for g in ("transport", "reclass", "alive"):
            sel = lab == g
            row[f"wnorm_final_{g}"] = float(np.median(wn[-1, sel])) if sel.any() else np.nan
            row[f"absv_final_{g}"] = float(np.median(v[-1, sel])) if sel.any() else np.nan
            for k in ("ratio", "stab_self", "stab_rest", "n_unit", "stab_self_min"):
                row[f"e1_{k}_{g}"] = e1[g][k]
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("run_id").reset_index(drop=True)
    return df, pe_pack, d1_pack, d1a_pack, pd.concat(ev_all, ignore_index=True), e1_pack


# ------------------------------------------------------------------ S3 (既出数値の再現)

def check_s3(resdir, seeds, period, half_w, P):
    """S3: 分離前の全体で ratchet_log_0819 の verdict.csv を再現する。

    `figures_ratchet_log` の per_seed_metrics + judge をそのまま呼ぶ (再実装しない)。
    乱数消費の順序まで同一なので、bootstrap CI や符号置換 null の 95%点も一致するはず。
    """
    df, pe_inc, bc_all, stair, _ = per_seed_metrics(seeds, period, half_w)
    V, _ = judge(df, pe_inc, bc_all, stair, P, seeds, period, half_w)
    ref = pd.read_csv(os.path.join(resdir, "verdict.csv"))
    rows, ok = [], True
    for vid, stat, want, tol in S3_TARGETS:
        m = V[(V.id == vid) & (V.statistic == stat)]
        got = float(m.point.iloc[0]) if len(m) else np.nan
        rm = ref[(ref.id == vid) & (ref.statistic == stat)]
        refv = float(rm.point.iloc[0]) if len(rm) else np.nan
        good = bool(np.isfinite(got) and abs(got - want) <= tol
                    and np.isfinite(refv) and abs(refv - want) <= tol)
        ok &= good
        rows.append(dict(id=vid, statistic=stat, spec_value=want, recomputed=got,
                         verdict_csv=refv, tol=tol, ok=good))
    # n_window と null 95%点は note 文字列にしか無いので文字列で控える
    note = V[(V.id == "P3")].note.iloc[0] if (V.id == "P3").any() else ""
    thr = V[(V.id == "P3")].threshold.iloc[0] if (V.id == "P3").any() else ""
    return ok, pd.DataFrame(rows), dict(p3_note=str(note), p3_threshold=str(thr))


# ------------------------------------------------------------------ 判定 (§5)

def judge_d(df, pe_pack, d1_pack, d1a_pack, e1_pack, rng):
    V = []
    add = lambda **kw: V.append(kw)

    # --- D0: 前提ゲート
    m, lo, hi = boot_ci(rng, df.r_reclass, BOOT_N)
    pooled = float(df.n_perm_reclass.sum() / max(df.n_perm_unit.sum(), 1))
    gate = bool(np.isfinite(m) and D0_LO <= m <= D0_HI)
    add(id="D0", statistic="恒久死に占める再分類死の割合 r (seed 平均)", point=m,
        ci_lo=lo, ci_hi=hi, threshold=f"{D0_LO:.2f} <= r <= {D0_HI:.2f} なら D1–D3 へ",
        result="ゲート通過" if gate else ("実質単一経路 (輸送死)" if m < D0_LO
                                      else "実質単一経路 (再分類死)"),
        note=f"pooled r={pooled:.4f}; 恒久死ユニット {int(df.n_perm_unit.sum())} / "
             f"うち再分類 {int(df.n_perm_reclass.sum())} (10 seed x 100 unit)")

    # --- D1: 再分類死は輸送を伴わない
    for tag, pack, lab in (("D1", d1_pack, "素の窓内対照 (事前登録)"),
                           ("D1-alive", d1a_pack, "対照を左端 alive に限定 (副次)")):
        obs = np.array([x["obs_median"] for x in pack])
        q95 = np.array([x["ctrl_q95"] for x in pack])
        mo, loo, hio = boot_ci(rng, obs, BOOT_N)
        mq = float(np.nanmean(q95))
        p1 = bool(np.isfinite(mo) and np.isfinite(mq) and mo <= mq)
        fb, _, _ = boot_ci(rng, [x["frac_event_below_own_q95"] for x in pack], BOOT_N)
        add(id=tag, statistic=f"境界ペアの |Δ‖w‖| 中央値 ({lab})", point=mo,
            ci_lo=loo, ci_hi=hio,
            threshold=f"同一窓内の非境界ペアの 95%点 ({mq:.3e}) 以下",
            result=("PASS" if p1 else "FAIL") if tag == "D1" else
                   ("<= q95" if p1 else "> q95"),
            note=f"窓内対照 n={int(np.sum([x['n_ctrl'] for x in pack]))}, "
                 f"イベント n={int(np.sum([x['n_event'] for x in pack]))}; "
                 f"自分の窓の 95%点以下だったイベントの割合 {fb:.4f}; "
                 f"|Δ‖w‖|=0 の割合 境界 {np.nanmean([x['obs_frac_zero'] for x in pack]):.4f} "
                 f"vs 対照 {np.nanmean([x['ctrl_frac_zero'] for x in pack]):.4f} "
                 f"(batch=1 の疎さの目安)")

    # --- D2: 主判定
    inc_t = [pe_pack[i]["transport"]["incs"] for i in range(len(pe_pack))]
    pes_t = np.concatenate([p["transport"]["pes"] for p in pe_pack]) \
        if pe_pack else np.zeros(0)
    obs = float(np.median(pes_t)) if pes_t.size else np.nan
    null = pe_permutation_null(rng, inc_t, PERM_N)
    q95 = float(np.quantile(null, 0.95)) if np.ndim(null) else np.nan
    d2 = bool(np.isfinite(obs) and np.isfinite(q95) and obs > q95)
    add(id="D2", statistic="median PE (輸送死・µ̂ 固定ペアのみ)", point=obs,
        ci_lo=np.nan, ci_hi=np.nan,
        threshold=f"帰無 95 パーセンタイル ({q95:.4f}) 超",
        result="PASS" if d2 else "FAIL",
        note=f"n_window={pes_t.size}, 符号置換 null {PERM_N} 回, "
             f"null 中央値 {float(np.median(null)):.4f}"
             if np.ndim(null) else "null 不定")
    pes_r = np.concatenate([p["reclass"]["pes"] for p in pe_pack]) \
        if pe_pack else np.zeros(0)
    add(id="D2-reclass", statistic="median PE (再分類死・µ̂ 固定ペアのみ)",
        point=float(np.median(pes_r)) if pes_r.size else np.nan,
        ci_lo=np.nan, ci_hi=np.nan, threshold="(参考・判定に使わない)", result="—",
        note=f"n_window={pes_r.size}。降下窓の増分は µ̂ 固定ペアだけに絞ってあるので、"
             f"この群でも張り替えは入っていない (違うのは死亡の起き方だけ)")

    # --- D3: E1 の群別再計算
    for g, lab in (("transport", "輸送死"), ("reclass", "再分類死"), ("alive", "生存")):
        ms, los, his = boot_ci(rng, df[f"e1_stab_self_{g}"], BOOT_N)
        mr, lor, hir = boot_ci(rng, df[f"e1_stab_rest_{g}"], BOOT_N)
        mq, loq, hiq = boot_ci(rng, df[f"e1_ratio_{g}"], BOOT_N)
        smin = float(np.nanmin(df[f"e1_stab_self_min_{g}"])) \
            if np.isfinite(df[f"e1_stab_self_min_{g}"]).any() else np.nan
        okg = bool(np.isfinite(ms) and abs(ms - 1.0) <= 1e-12)
        add(id="D3", statistic=f"self 符号安定率 [{lab}]", point=ms, ci_lo=los,
            ci_hi=his, threshold="1.000 (Q2 の定理の帰結。崩れたらコードのバグ)",
            result="PASS" if okg else "FAIL",
            note=f"ユニット最小値 {smin:.6f}, n_unit={int(np.nansum(df[f'e1_n_unit_{g}']))}")
        add(id="D3", statistic=f"rest 符号安定率 [{lab}]", point=mr, ci_lo=lor,
            ci_hi=hir, threshold="(群差の記述・判定なし)", result="—", note="")
        add(id="D3", statistic=f"self/rest の |µ̂ 射影| 時間平均比 [{lab}]", point=mq,
            ci_lo=loq, ci_hi=hiq, threshold="(群差の記述・判定なし)", result="—", note="")

    # --- D4: 混在の量 (報告のみ)
    mall, loa, hia = boot_ci(rng, df.frac_abs_dcos_boundary, BOOT_N)
    add(id="D4", statistic="境界跨ぎ増分が Σ|Δcos| に占める割合", point=mall,
        ci_lo=loa, ci_hi=hia, threshold="(報告のみ)", result="—",
        note="降下窓内で、境界ペアを跨ぐ |Δcos| の総和 / 全 |Δcos| の総和")
    add(id="D4", statistic="分離前後の median PE の差 (D2 − S3 の 0.3959)",
        point=(obs - S3_TARGETS[0][2]) if np.isfinite(obs) else np.nan,
        ci_lo=np.nan, ci_hi=np.nan, threshold="(報告のみ)", result="—",
        note=f"分離前 {S3_TARGETS[0][2]:.6f} (n=766) -> 分離後 {obs:.6f} "
             f"(n={pes_t.size})")
    for k, lab in (("n_event_reclass", "死亡イベント数 [再分類死]"),
                   ("n_event_transport", "死亡イベント数 [輸送死]"),
                   ("n_perm_reclass", "恒久死ユニット数 [再分類死]"),
                   ("n_perm_transport", "恒久死ユニット数 [輸送死]"),
                   ("n_alive_unit", "生存ユニット数")):
        mk, lok, hik = boot_ci(rng, df[k].astype(float), BOOT_N)
        add(id="D4", statistic=f"{lab} (seed 平均)", point=mk, ci_lo=lok, ci_hi=hik,
            threshold="(報告のみ)", result="—",
            note=f"10 seed 合計 {int(df[k].sum())}")

    # --- D5: 探索 (判定なし・追跡しない)
    for k, lab in (("reignite_rate_reclass", "再点灯率 [再分類死イベント]"),
                   ("reignite_rate_transport", "再点灯率 [輸送死イベント]"),
                   ("p_hat_prev_med_reclass", "死亡直前 p̂ の中央値 [再分類死]"),
                   ("p_hat_prev_med_transport", "死亡直前 p̂ の中央値 [輸送死]"),
                   ("wnorm_final_reclass", "最終 ‖w‖ 中央値 [再分類死]"),
                   ("wnorm_final_transport", "最終 ‖w‖ 中央値 [輸送死]"),
                   ("wnorm_final_alive", "最終 ‖w‖ 中央値 [生存]"),
                   ("absv_final_reclass", "最終 |v| 中央値 [再分類死]"),
                   ("absv_final_transport", "最終 |v| 中央値 [輸送死]"),
                   ("absv_final_alive", "最終 |v| 中央値 [生存]")):
        mk, lok, hik = boot_ci(rng, df[k], BOOT_N)
        add(id="D5", statistic=lab, point=mk, ci_lo=lok, ci_hi=hik,
            threshold="(探索・判定なし・追跡しない)", result="—", note="")
    return pd.DataFrame(V), dict(gate=gate, d2=d2, null=null, q95=q95, obs=obs,
                                 pes_t=pes_t, pes_r=pes_r, pooled_r=pooled)


# ------------------------------------------------------------------ 図

def make_figures(outdir, df, ev, pe_pack, d1_pack, extra):
    fig_dir = os.path.join(outdir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # 1) 群別 PE 分布
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    bins = np.linspace(0, 1, 41)
    for k, c, lab in (("pes_t", "tab:blue", "輸送死"), ("pes_r", "tab:orange", "再分類死")):
        v = extra[k]
        if v.size:
            ax.hist(v, bins=bins, alpha=.55, color=c, density=True,
                    label=f"{lab} (n={v.size}, med={np.median(v):.3f})")
    if np.isfinite(extra["q95"]):
        ax.axvline(extra["q95"], color="k", ls="--", lw=1.2,
                   label=f"符号置換 null 95%点 = {extra['q95']:.3f}")
    ax.set_xlabel("PE = |Σ Δcos| / Σ|Δcos| (µ̂ 固定ペアのみ)")
    ax.set_ylabel("密度")
    ax.set_title("D2: 群別の path efficiency")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig_pe_by_path.png"), dpi=140)
    plt.close(fig)

    # 2) 境界ペア vs 非境界ペアの |Δ‖w‖|
    obs = np.array([x["obs_median"] for x in d1_pack])
    q95 = np.array([x["ctrl_q95"] for x in d1_pack])
    med = np.array([x["ctrl_median"] for x in d1_pack])
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    x = np.arange(len(obs))
    ax.plot(x, obs, "o-", color="tab:red", label="境界ペア |Δ‖w‖| の中央値")
    ax.plot(x, q95, "s--", color="tab:gray", label="窓内の非境界ペア 95%点")
    ax.plot(x, med, "^:", color="tab:green", label="窓内の非境界ペア 中央値")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(df.seed.astype(str))
    ax.set_xlabel("seed")
    ax.set_ylabel("|Δ‖w‖| (log)")
    ax.set_title("D1: 再分類死の境界ペアは窓内対照より動いていないか")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig_dwnorm_boundary_vs_window.png"), dpi=140)
    plt.close(fig)

    # 3) 群別の死亡時刻 (記述のみ。S4 により群間の時間比較はしない)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    bins = np.linspace(0, float(ev.step_post.max()), 51)
    for path, c, lab in (("transport", "tab:blue", "輸送死"),
                         ("reclass", "tab:orange", "再分類死")):
        s = ev[(ev.path == path) & ev.permanent].step_post.values
        if s.size:
            ax.hist(s, bins=bins, alpha=.55, color=c, label=f"{lab} (n={s.size})")
    ax.set_xlabel("恒久死の記録 step (右端の記録点)")
    ax.set_ylabel("ユニット数")
    ax.set_title("恒久死の時刻分布 (記述のみ)")
    ax.text(.02, .96, "S4: 時刻分解能が群で非対称 (再分類死は 1 step / 輸送死は\n"
                      "バルクでは 1000 step)。ハザード・t50 の群間比較はしない",
            transform=ax.transAxes, va="top", fontsize=7,
            bbox=dict(fc="w", ec="0.6", alpha=.85))
    ax.legend(fontsize=8, loc="center right")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig_death_time_by_path.png"), dpi=140)
    plt.close(fig)
    return fig_dir


# ------------------------------------------------------------------ summary.md

def _md(df, fmt=".4f"):
    cols = list(df.columns)
    f = lambda v: (format(v, fmt) if isinstance(v, (float, np.floating))
                   and np.isfinite(v) else ("" if isinstance(v, float)
                                            and not np.isfinite(v) else str(v)))
    out = ["| " + " | ".join(cols) + " |",
           "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(f(r[c]) for c in cols) + " |")
    return "\n".join(out)


def write_summary(outdir, V, df, s3_df, s3_note, s2_df, extra, meta):
    L = ["# dead2path_0821: 死の二経路の分離 (再分類死 / 輸送死)", "",
         f"仕様: `specs/spec_dead2path_0821.md` (事前登録)。生成: `{meta['date']}`、"
         f"git `{meta['git_hash']}`。**再学習なし** — "
         f"`results/ratchet_log_0819/logs/seed*.npz` の事後解析。", "",
         "## 0. 一行", "",
         "死 (p̂ の 0.05 下抜け) を、境界の実効閾値ジャンプによる**再分類死**と、"
         "タスク内の力による**輸送死**に記録構造から一意に分離し、分離後に P3 "
         "(単調輸送) と E1 (駆動分解) を再判定した。", "",
         "## 1. サニティ", "",
         f"**S1**: `OMP_NUM_THREADS={meta['omp_num_threads']}`、"
         f"python {meta['python']} / numpy {meta['numpy']} / pandas {meta['pandas']}。", "",
         f"**S2** (境界インデックスを flip_state から機械的に決定): "
         f"{'PASS' if bool(s2_df.ok.all()) else '**FAIL**'} — 全 "
         f"{len(s2_df)} seed で「flip 変化ペアの先頭が step ≡ 0 (mod 10⁴)」かつ隣接。", "",
         _md(s2_df[["seed", "n_flip_pairs", "n_boundaries_in_grid",
                    "n_left_aligned", "n_adjacent", "ok"]]), "",
         "**S3** (既出数値の再現。これが通らなければ分離に進まない):", "",
         _md(s3_df[["id", "statistic", "spec_value", "recomputed", "verdict_csv", "ok"]],
             ".8f"), "",
         f"P3 の再計算 note: `{s3_note['p3_note']}` / 閾値 `{s3_note['p3_threshold']}`。", "",
         "**S4**: 死亡時刻の分解能は群で非対称である (再分類死は必ず 1 step、輸送死は"
         "バルク区間なら 1000 step)。したがって**時間あたりのハザード・t50 の群間比較は"
         "本 summary に含めない**。図 `fig_death_time_by_path.png` は記述のみ。", "",
         "## 2. D0 ゲート", "",
         _md(V[V.id == "D0"][["statistic", "point", "ci_lo", "ci_hi",
                              "threshold", "result", "note"]]), ""]
    if extra["gate"]:
        L += ["ゲート**通過**。二経路がともに非自明に存在するので D1–D3 を解釈する。", ""]
    else:
        L += ["ゲート**不通過**。spec §5 D0 の規定により、以下の D1–D3 は"
              "**参考値として記録するだけ**で、Q16 の結論は D0 の行が担う。", ""]
    L += ["## 3. 判定 (D1–D3)", "",
          _md(V[V.id.isin(["D1", "D1-alive", "D2", "D2-reclass", "D3"])]
              [["id", "statistic", "point", "ci_lo", "ci_hi", "threshold",
                "result", "note"]]), "",
          "## 4. 混在の量 (D4・判定なし)", "",
          _md(V[V.id == "D4"][["statistic", "point", "ci_lo", "ci_hi", "note"]]), "",
          "## 5. 探索 (D5・判定なし・追跡しない)", "",
          _md(V[V.id == "D5"][["statistic", "point", "ci_lo", "ci_hi"]]), "",
          "spec §8 の線引きにより、D5 はここから機構の深掘りに入らない。", "",
          "## 6. スコープ・禁止事項 (spec §8 の転記)", "",
          "- スコープ: **condA・w100・T=1e4・batch=1・ratchet_log_0819 のログ限定**。"
          "condB へ外挿しない",
          "- 再学習しない。アームを足したくなったら別 spec を起案する",
          "- **台帳の代数 (縮退・増幅・二速修理) を本 spec の結果に混ぜない** "
          "(論文スコープ分離)。現論文へ渡すのは spec §1 の一文と群別の数値のみ",
          "- D5 は記録のみ。ここから機構の深掘りに入らない (8/21 の線引き)",
          "- D2 が FAIL した場合、代替仮説 (拡散到着) の検討は**別 spec**とする。"
          "本 spec では撤回の事実と数値だけを記録する",
          "- null 結果も PASS と同じ形式で報告する", "",
          "## 7. 逸脱・留保", "",
          "1. **D1 の対照の作り方**: spec §5 の字義は「同一窓内の非境界ペアの |Δ‖w‖| 分布」"
          "で、これを主判定に据えた (`D1` 行)。ただし死亡後のユニットは勾配が恒等的に 0 で "
          "‖w‖ が動かないため、素の対照には構造的な 0 が混ざり 95%点を押し下げる "
          "(= 判定を通りにくくする) 向きに効く。対照を「左端で p̂ >= TAU のペア」に"
          "限定した副次値を `D1-alive` 行に併記した。あわせて **|Δ‖w‖| が厳密に 0 の"
          "割合**を境界側と対照側で並べた (note 欄)。batch=1 では「そのユニットのゲートが"
          "開いたサンプルが引かれた step」でしか w_i は動かないので、境界側の中央値が 0 に"
          "なること自体は経路の性質とは限らない。両者の 0 の割合を見て読むこと。",
          "2. **D2 の母集団**: 降下窓の作り方は比較可能性のため ratchet_log_0819 の "
          "`death_events` (下方クロス後 1 周期回復しない最初の 1 個) をそのまま使い、"
          "そこに経路ラベルと µ̂ 固定ペアのフィルタを掛けた。spec §3 の「死亡イベント = "
          "全ての下方クロス」の母集団 (events.csv・D0・D1・D5 が使う) とは別である。"
          "両者を混ぜていない。",
          "3. **D3 の生存群の区間**: 死亡群は恒久死に対応する降下窓で測るが、生存群には"
          "降下窓が定義できないため全記録区間で測った。区間長が群間で違うので、rest の"
          "符号安定率の群差は「同じ長さでの比較」ではない。self の符号安定率 (= 1.000 の"
          "検算) は区間長に依らないので D3 の判定自体はこの非対称の影響を受けない。",
          "4. **D2 の帰無の乱数列**: 符号置換の手続きは P3 と同一 (窓ごと 1000 回) だが、"
          "抽選列は本 spec の `default_rng(20260821)` を使っている。P3 の 0.2177 とは"
          "手続きが同じで乱数列が違う。", "",
          "## 8. 出力", "",
          "- `verdict.csv` — D0–D5 の全行",
          "- `per_seed_metrics.csv` — seed ごとの全指標",
          "- `events.csv` — 死亡イベント (seed, unit, pair, path, reignited, permanent, "
          "p̂_prev, Δcos, Δ‖w‖)",
          "- `figures/fig_pe_by_path.png`, `fig_dwnorm_boundary_vs_window.png`, "
          "`fig_death_time_by_path.png`", ""]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="?",
                    default=os.path.join(ROOT, "results", "ratchet_log_0819"),
                    help="ratchet_log_0819 の実験ディレクトリ")
    ap.add_argument("--outdir", default=os.path.join(ROOT, "results", "dead2path_0821"))
    ap.add_argument("--skip-s3", action="store_true",
                    help="S3 の再現をスキップ (デバッグ専用。本走では使わない)")
    args = ap.parse_args()

    t0 = time.time()
    resdir = args.results
    cfg = load_config(os.path.join(resdir, "config_used.yaml"))
    P = cfg["ratchet"]
    period = int(cfg["condA"]["T_values"][0])
    half_w = int(P["boundary_window"])

    seeds = load_seeds(resdir)
    seeds.sort(key=lambda d: str(d["run_id"]))                 # run_id ソート [§7]
    print(f"loaded {len(seeds)} seeds, {len(seeds[0]['step'])} 記録点, "
          f"period={period}, half_w={half_w}", flush=True)

    s2_ok, s2_df = check_s2(seeds, period)
    print(f"S2: {'PASS' if s2_ok else 'FAIL'}", flush=True)
    if not s2_ok:
        print(s2_df.to_string(index=False))
        raise SystemExit("[dead2path] S2 FAIL — 境界の同定が壊れている。中止 (§6 S2)")

    if args.skip_s3:
        s3_ok, s3_df, s3_note = True, pd.DataFrame(), dict(p3_note="(skipped)",
                                                           p3_threshold="(skipped)")
    else:
        print("S3: ratchet_log_0819 の既出数値を再現中 ...", flush=True)
        s3_ok, s3_df, s3_note = check_s3(resdir, seeds, period, half_w, P)
        print(s3_df.to_string(index=False), flush=True)
        if not s3_ok:
            raise SystemExit("[dead2path] S3 FAIL — 解析コードのバグ。分離に進まない (§6 S3)")

    df, pe_pack, d1_pack, d1a_pack, ev, e1_pack = per_seed(seeds, period, half_w)
    rng = np.random.default_rng(BOOT_SEED)
    V, extra = judge_d(df, pe_pack, d1_pack, d1a_pack, e1_pack, rng)

    os.makedirs(args.outdir, exist_ok=True)
    df.to_csv(os.path.join(args.outdir, "per_seed_metrics.csv"), index=False)
    ev.to_csv(os.path.join(args.outdir, "events.csv"), index=False)
    V.to_csv(os.path.join(args.outdir, "verdict.csv"), index=False)
    fig_dir = make_figures(args.outdir, df, ev, pe_pack, d1_pack, extra)

    meta = dict(date=time.strftime("%Y-%m-%d %H:%M:%S"), git_hash=git_hash(),
                spec="specs/spec_dead2path_0821.md", source=resdir,
                elapsed_sec=round(time.time() - t0, 1),
                omp_num_threads=os.environ.get("OMP_NUM_THREADS", "(未設定)"),
                python=platform.python_version(), numpy=np.__version__,
                pandas=pd.__version__, bootstrap_B=BOOT_N, bootstrap_seed=BOOT_SEED,
                perm_null_n=PERM_N, n_seeds=len(seeds), period=period,
                half_window=half_w,
                s2_pass=bool(s2_ok), s3_pass=bool(s3_ok),
                s2=s2_df.to_dict("records"),
                s3=s3_df.to_dict("records") if len(s3_df) else [],
                d0_gate=bool(extra["gate"]), d2_pass=bool(extra["d2"]))
    with open(os.path.join(args.outdir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1, default=str, ensure_ascii=False)
    sp = write_summary(args.outdir, V, df, s3_df, s3_note, s2_df, extra, meta)

    print(V.to_string(index=False), flush=True)
    print(f"-> {args.outdir}/verdict.csv, per_seed_metrics.csv, events.csv, "
          f"{sp}, {fig_dir}/", flush=True)
    print("DEAD2PATH DONE", flush=True)
    return df, V


if __name__ == "__main__":
    main()
