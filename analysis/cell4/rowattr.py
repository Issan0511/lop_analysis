"""cell4_0821: G の µ̂ 整列の行帰属 — 税 (self 行) vs それ以外 (rest 行) [spec_cell4_0821]。

  OMP_NUM_THREADS=1 .venv/bin/python -m analysis.cell4.rowattr \
      [results/ratchet_log_0819] [--outdir results/cell4_0821]

**再学習なし・再走なし**。`results/ratchet_log_0819/logs/seed*.npz` だけを読む。

核になる恒等式 [spec §2]。probe の定義 (`src/ratchet_log.py` L119-126) で
a_i·gate_i = a_i (relu は gate=0 の所で厳密に 0) なので

    F_self,i = −2η·v_i²·(E[a_i·x]·µ̂)
      ⇒ self 行の µ̂ 射影 = v_i·E[a_i·x]·µ̂ = −F_self,i/(2η·v_i)   … ゲート不変
    rest 行の µ̂ 射影 (ゲート抜き) = G·µ̂ − (self 行)
    rest 行の µ̂ 射影 (ゲート版)   = −F_rest,i/(2η·v_i)

判定は spec §5 の P0-P4 が唯一の正で、本モジュールはそれを実装するだけ。特に

  P0  前提ゲート (S1-S4)。FAIL なら判定に進まない
  P1  **主判定**: σ·(rest 射影) > σ·(self 射影)、σ = sign(G·µ̂)。**‖G‖ を含まない** [§3.1]
  P1b 正規化版の floor ∈ {1e-3,1e-2,1e-1} 感度。符号が割れたら P1 を判定保留へ [§3.2]
  P2  ゲート版で同じ比較
  P3  層B / 層C / eval_loss 四分位での同じ量 (報告のみ)
  P4  総寄与シェア |self|/(|self|+|rest|) (報告のみ・主判定にしない)

を先に読むこと。S2/S3 (既出数値の再現) が通らなければ判定を計算しない。

出力 (すべて --outdir の中): verdict.csv / summary.md / per_seed_metrics.csv /
meta.json / figures/。
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
from src.common import ROOT, load_config                                  # noqa: E402
from src.figures_ratchet_log import (TAU, death_events, descent_windows,  # noqa: E402
                                     e1_drive_decomposition, load_seeds)

BOOT_N = 10000
BOOT_SEED = 20260821            # spec §5 の事前登録の抽選列
VMIN = 1e-2                     # |v_i| の下限ゲート [spec §3 Phase 0]
FLOORS = (1e-3, 1e-2, 1e-1)     # P1b の ‖G‖ 下限 3 点 [spec §3.2]
SIGN_MIN = 9                    # P1 の符号数の下限 (9/10) [spec §5]

# S2: ratchet_log_0819 verdict.csv の E1 行 (spec §9 S2 に転記されている値)
S2_REF = [
    ("e1_sign_stab_self", "self 成分の符号安定率", 1.0, 1e-9),
    ("e1_sign_stab_rest", "rest 成分の符号安定率", 0.8059758263149069, 1e-9),
    ("e1_ratio_self_rest", "self/rest の |µ̂ 射影| 時間平均比", 0.6176014192580223, 1e-9),
]

# S3: spec §1 の再現表
S3_SCOPE_SEEDS = (1, 2, 3, 4, 5, 9)          # t=500k の |E[δ]| 上位 6 seed
S3_REF = dict(cos_lo=0.9656, cos_hi=0.9945, rat_lo=3.70, rat_hi=9.57, rat_med=4.252)

# 層C の固定集合 [spec §4]。記録点ごとの outcome では条件付けない
LAYER_C_IN, LAYER_C_OUT = set(S3_SCOPE_SEEDS), {0, 6, 7, 8}

plt.rcParams["font.family"] = ["Noto Sans CJK JP", "Noto Sans CJK TC",
                              "Noto Sans CJK KR", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------------- 行射影

def row_projections(d):
    """1 seed ぶんの行射影と各種マスク。全て float64 で作る。

    返り値の [n,h] 配列は unit_ok でない要素を nan にしてあるので、以降は
    np.nanmedian / 有限マスクで素直に集計できる。"""
    eta = float(d["lr"])
    v = d["v"].astype(np.float64)
    p = d["p_hat"].astype(np.float64)
    Gdm = d["G_dot_mu"].astype(np.float64)                 # [n]
    Gn = np.linalg.norm(d["G"].astype(np.float64), axis=1)  # [n]

    unit_ok = (p >= TAU) & (np.abs(v) >= VMIN)             # [n,h]
    vs = np.where(unit_ok, v, np.nan)

    self_proj = -d["F_self"].astype(np.float64) / (2 * eta * vs)      # ゲート不変
    rest_proj = Gdm[:, None] - self_proj                              # ゲート抜き
    rest_g = -d["F_rest"].astype(np.float64) / (2 * eta * vs)         # ゲート版
    tot_g = -d["F_gate"].astype(np.float64) / (2 * eta * vs)          # ゲート版の合計

    sigma = np.sign(Gdm)                                   # [n]
    # σ=0 (G·µ̂ が float32 で厳密に 0) の記録点は向きが定義できないので落とす
    rec_ok = sigma != 0
    sg = np.where(rec_ok, sigma, np.nan)[:, None]

    # §3.1 の主判定量。σ·(rest) − σ·(self) = σ·(G·µ̂) − 2σ·(self) で ‖G‖ を含まない
    d_raw = sg * (rest_proj - self_proj)
    # §3.2 の正規化版 (解釈用・P1b の感度用)
    c_self = sg * self_proj / Gn[:, None]
    c_rest = sg * rest_proj / Gn[:, None]

    # P2: ゲート版。合計の符号を σ_g とするのが §3.1 の直接の類推
    sg_g = np.sign(tot_g)
    d_g = np.where(sg_g != 0, sg_g * (rest_g - self_proj), np.nan)

    # P4: 有界・‖G‖ 非依存のシェア
    den = np.abs(self_proj) + np.abs(rest_proj)
    share_self = np.where(den > 0, np.abs(self_proj) / den, np.nan)

    return dict(step=d["step"], seed=int(d["seed"]), eta=eta,
                Gdm=Gdm, Gn=Gn, sigma=sigma, rec_ok=rec_ok, unit_ok=unit_ok,
                self_proj=self_proj, rest_proj=rest_proj, rest_g=rest_g,
                d_raw=d_raw, d_g=d_g, c_self=c_self, c_rest=c_rest,
                share_self=share_self,
                eval_loss=d["eval_loss_exact"].astype(np.float64),
                cos_G_mu=np.abs(d["cos_G_mu"].astype(np.float64)))


def grid_layers(step, period, half_w, bulk_every):
    """層A (バルク) / 層B (境界窓) の記録点マスク [spec §4]。"""
    bnd = np.arange(period, int(step[-1]) + 1, period)
    dist = np.min(np.abs(step[:, None] - bnd[None, :]), axis=1)
    layer_b = dist <= half_w
    layer_a = (step % bulk_every == 0) & ~layer_b
    return layer_a, layer_b


def _agg(vals, rec_mask, R):
    """(記録点 × alive) を 1 スカラーに: 勝率と中央値。rec_mask は [n] の層マスク。"""
    x = vals[rec_mask & R]
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan, 0
    return float((x > 0).mean()), float(np.median(x)), int(x.size)


# ---------------------------------------------------------------- サニティ

def check_s2(seeds):
    """S2: figures_ratchet_log の E1 を再現する [spec §9]。

    参照値は ratchet_log_0819 verdict.csv の E1 行 = boot_ci の点推定 = seed 平均。"""
    period = 10000
    rows = []
    vals = {k: [] for k, _, _, _ in S2_REF}
    for d in seeds:
        e1 = e1_drive_decomposition(d, descent_windows(d, death_events(d, period)))
        for k in vals:
            vals[k].append(e1[k])
    ok = True
    for key, name, ref, tol in S2_REF:
        got = float(np.mean([x for x in vals[key] if np.isfinite(x)]))
        good = bool(abs(got - ref) <= tol)
        ok &= good
        rows.append(dict(id="S2", statistic=name, spec_value=ref, recomputed=got,
                         abs_diff=abs(got - ref), tol=tol, ok=good))
    return ok, pd.DataFrame(rows)


def check_s3(seeds):
    """S3: spec §1 の再現表 (t=500k・|E[δ]| 上位 6 seed) を再計算する。"""
    step = seeds[0]["step"]
    i5 = int(np.argmin(np.abs(step - 500000)))
    order = sorted(range(len(seeds)), key=lambda j: -abs(float(seeds[j]["E_delta"][i5])))
    top6 = sorted(int(seeds[j]["seed"]) for j in order[:6])
    cos = np.array([abs(float(s["cos_G_mu"][i5])) for s in seeds])
    rat = np.array([float(s["ratio_mu_cov"][i5]) for s in seeds])
    sel = np.array([int(s["seed"]) in set(top6) for s in seeds])
    got = dict(seeds=top6, cos_lo=cos[sel].min(), cos_hi=cos[sel].max(),
               rat_lo=rat[sel].min(), rat_hi=rat[sel].max(), rat_med=float(np.median(rat)))
    rows = [dict(id="S3", statistic="|E[δ]| 上位 6 seed", spec_value=str(list(S3_SCOPE_SEEDS)),
                 recomputed=str(top6), ok=bool(top6 == sorted(S3_SCOPE_SEEDS)))]
    for k, name, dec in [("cos_lo", "6 seed の |cos| 下端", 4), ("cos_hi", "6 seed の |cos| 上端", 4),
                         ("rat_lo", "6 seed の ratio 下端", 2), ("rat_hi", "6 seed の ratio 上端", 2),
                         ("rat_med", "全 10 seed の ratio 中央値", 3)]:
        good = bool(round(got[k], dec) == round(S3_REF[k], dec))
        rows.append(dict(id="S3", statistic=name, spec_value=S3_REF[k],
                         recomputed=round(float(got[k]), 6), ok=good))
    df = pd.DataFrame(rows)
    return bool(df["ok"].all()), df, got


def check_s4(seeds, period, half_w, bulk_every):
    """S4: p̂=1 のユニットで −F_gate/(2ηv) == G·µ̂ [spec §9]。

    p̂=1 ではゲートが全 32 パターンで開くので F_gate はゲート抜きの G·µ̂ に一致するはず。
    F_gate は F_self/F_rest とは別に保存された量なので、これは**独立な 2 経路の一致**
    であり定義から従わない。被覆は前半に偏るので時間層別も返す (spec §9)。"""
    bins = [(0, 1), (1, 10000), (10000, 100000), (100000, 500000), (500000, 10 ** 9)]
    errs, per_bin = [], {b: [[], 0, 0] for b in bins}
    n_hit = n_tot = n_pts_hit = n_pts = 0
    for d in seeds:
        eta = float(d["lr"])
        v = d["v"].astype(np.float64)
        m = (d["p_hat"] == 1.0) & (np.abs(v) >= VMIN)
        step, Gdm = d["step"], d["G_dot_mu"].astype(np.float64)
        n_tot += m.size
        n_hit += int(m.sum())
        n_pts += m.shape[0]
        n_pts_hit += int(m.any(axis=1).sum())
        if not m.any():
            continue
        lhs = -d["F_gate"].astype(np.float64)[m] / (2 * eta * v[m])
        rhs = np.repeat(Gdm[:, None], m.shape[1], axis=1)[m]
        e = np.abs(lhs - rhs) / np.maximum(np.abs(rhs), 1e-300)
        errs.append(e)
        ridx = np.repeat(step[:, None], m.shape[1], axis=1)[m]
        for b in bins:
            sel = (ridx >= b[0]) & (ridx < b[1])
            if sel.any():
                per_bin[b][0].append(e[sel])
            rsel = (step >= b[0]) & (step < b[1])
            per_bin[b][1] += int(m[rsel].any(axis=1).sum())
            per_bin[b][2] += int(rsel.sum())
    e = np.concatenate(errs)
    rows = []
    for b in bins:
        eb = np.concatenate(per_bin[b][0]) if per_bin[b][0] else np.zeros(0)
        rows.append(dict(step_lo=b[0], step_hi=b[1], n_units=int(eb.size),
                         cover_frac=per_bin[b][1] / max(per_bin[b][2], 1),
                         max_rel_err=float(eb.max()) if eb.size else np.nan))
    ok = bool(e.max() < 1e-6)
    return ok, pd.DataFrame(rows), dict(
        s4_pass=ok, s4_n_units=n_hit, s4_frac_units=n_hit / n_tot,
        s4_n_points=n_pts_hit, s4_frac_points=n_pts_hit / n_pts,
        s4_med_rel_err=float(np.median(e)), s4_p99_rel_err=float(np.percentile(e, 99)),
        s4_max_rel_err=float(e.max()), s4_threshold=1e-6)


# ---------------------------------------------------------------- seed 集計

def per_seed(seeds, period, half_w, bulk_every):
    """seed ごとに層別の指標を作る [spec §5]。

    集計順序は spec §5 の明記どおり: 記録点 × alive ユニットごとに対応量 → run 単位で
    中央値 (と勝率) → seed ベクトル。`median(c_rest − c_self)` を採り
    `median(c_rest) − median(c_self)` は使わない。"""
    rows, series = [], []
    for d in seeds:
        P = row_projections(d)
        step = P["step"]
        la, lb = grid_layers(step, period, half_w, bulk_every)
        R = P["rec_ok"]
        r = dict(seed=P["seed"], n_records=len(step),
                 n_layerA=int((la & R).sum()), n_layerB=int((lb & R).sum()),
                 n_sigma0=int((~R).sum()),
                 layerC="in" if P["seed"] in LAYER_C_IN else "out",
                 frac_unit_dropped=float(1 - P["unit_ok"].mean()),
                 frac_alive0=float(((P["unit_ok"].sum(axis=1) == 0) & R).mean()))

        for tag, mask in (("A", la), ("B", lb), ("all", np.ones_like(la))):
            w, m, n = _agg(P["d_raw"], mask, R)
            r[f"P1_win_{tag}"], r[f"P1_med_{tag}"], r[f"P1_n_{tag}"] = w, m, n
            wg, mg, ng = _agg(P["d_g"], mask, R)
            r[f"P2_win_{tag}"], r[f"P2_med_{tag}"], r[f"P2_n_{tag}"] = wg, mg, ng
            for nm, arr in (("cself", P["c_self"]), ("crest", P["c_rest"]),
                            ("share", P["share_self"])):
                x = arr[mask & R]
                x = x[np.isfinite(x)]
                r[f"{nm}_{tag}"] = float(np.median(x)) if x.size else np.nan

        # P1b: 正規化版の対応差を ‖G‖ floor 3 点で (層A)
        cdiff = P["c_rest"] - P["c_self"]
        for f in FLOORS:
            sel = la & R & (P["Gn"] >= f)
            x = cdiff[sel]
            x = x[np.isfinite(x)]
            r[f"P1b_med_{f:g}"] = float(np.median(x)) if x.size else np.nan
            r[f"P1b_n_{f:g}"] = int(x.size)

        # 事後診断 (事前登録外): self の符号をランダム化した null の期待勝率。
        # self_proj = v_i·E[a_i x]·µ̂ で E[a_i x]·µ̂ >= 0 (condA の非負性) なので
        # sign(self_proj) = sign(v_i)。これが ± 対称なら d = |G·µ̂| − 2σ·self の勝率は
        # signal ゼロでも 0.5 を超える。ε=±1 を等確率で振ると
        #   P(d_null > 0) = 0.5 + 0.5·P(2|self| < |G·µ̂|)
        # と解析的に出る (どちらの符号でも片方は必ず勝つため)。
        gabs = np.repeat(np.abs(P["Gdm"])[:, None], P["self_proj"].shape[1], axis=1)
        sel = la & R
        u, g2 = P["self_proj"][sel], gabs[sel]
        fin = np.isfinite(u)
        r["P1_null_A"] = float(0.5 + 0.5 * (2 * np.abs(u[fin]) < g2[fin]).mean())
        r["P1_excess_A"] = r["P1_win_A"] - r["P1_null_A"]
        r["frac_self_pos_A"] = float((u[fin] > 0).mean())

        # P3 診断: eval_loss の seed 内四分位 (判定なし・regime を見るだけ)
        q = np.quantile(P["eval_loss"][la & R], [0.25, 0.5, 0.75]) if (la & R).any() else None
        for qi, lo, hi in ([(1, -np.inf, q[0]), (2, q[0], q[1]), (3, q[1], q[2]),
                            (4, q[2], np.inf)] if q is not None else []):
            sel = la & R & (P["eval_loss"] > lo) & (P["eval_loss"] <= hi)
            w, m, n = _agg(P["d_raw"], sel, np.ones_like(R))
            r[f"P1_win_lossQ{qi}"], r[f"P1_n_lossQ{qi}"] = w, n
        rows.append(r)
        # alive=0 の記録点 (1.65%) は全 NaN 列になるが仕様どおりの挙動なので警告を抑止
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                series.append(dict(
                    seed=P["seed"], step=step, la=la, R=R, cos=P["cos_G_mu"],
                    c_self=np.nanmedian(np.where(P["unit_ok"], P["c_self"], np.nan), axis=1),
                    c_rest=np.nanmedian(np.where(P["unit_ok"], P["c_rest"], np.nan), axis=1),
                    share=np.nanmedian(np.where(P["unit_ok"], P["share_self"], np.nan), axis=1)))
    return pd.DataFrame(rows), series


def boot_median(rng, vec, B=BOOT_N):
    """seed 単位 bootstrap の**中央値**版 (spec §5 が「seed 中央値」を要求するため)。

    `figures_ratchet_log.boot_ci` は点推定に平均を使う。ratchet_log_0819 の verdict.csv
    は「seed 中央値」を点推定にしつつ CI は平均のもの、という不整合を注記で処理していた。
    ここは spec の字義に合わせて中央値で統一する (逸脱節に記載)。"""
    v = np.asarray(vec, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan, np.nan, np.nan
    bs = np.median(v[rng.integers(0, v.size, (B, v.size))], axis=1)
    return float(np.median(v)), float(np.quantile(bs, .025)), float(np.quantile(bs, .975))


# ---------------------------------------------------------------- 判定 [spec §5]

def judge(df, rng, s_ok):
    V, extra = [], {}

    def row(i, stat, pt, lo, hi, thr, res, note=""):
        V.append(dict(id=i, statistic=stat, point=pt, ci_lo=lo, ci_hi=hi,
                      threshold=thr, result=res, note=note))

    row("P0", "前提ゲート S1-S4", np.nan, np.nan, np.nan, "S1-S4 全 PASS",
        "PASS" if s_ok else "FAIL",
        "FAIL なら P1 以降を計算しない [spec §5]")

    # --- P1: 3 条件すべて (spec §5 の「対応差」が勝率か生の差か一意でないため両方を課す)
    win, wlo, whi = boot_median(rng, df["P1_win_A"])
    med, mlo, mhi = boot_median(rng, df["P1_med_A"])
    nsign = int((df["P1_med_A"] > 0).sum())
    n_ok = int(np.isfinite(df["P1_med_A"]).sum())
    p1 = bool(win > 0.5 and mlo > 0 and nsign >= SIGN_MIN)
    row("P1", "層A の勝率 σ·(rest) > σ·(self) (seed 中央値)", win, wlo, whi,
        "seed 中央値 > 0.5", ("PASS (字義)" if p1 else "FAIL"),
        f"3 条件の連言。勝率>0.5={win > 0.5}, 対応差 CI 下端>0={mlo > 0}, "
        f"符号数 {nsign}/{n_ok} >= {SIGN_MIN}={nsign >= SIGN_MIN}")
    row("P1-diff", "層A の対応差 σ·(rest−self) の中央値 (‖G‖ 非依存)", med, mlo, mhi,
        "CI 下端 > 0", "—", "spec §3.1 の主判定量。‖G‖ を含まない")
    row("P1-sign", "対応差が正だった seed 数", nsign, np.nan, np.nan,
        f">= {SIGN_MIN}/10 (符号検定 p<=0.011)", "—",
        "10 seed の中央値 bootstrap は階段状になるので併記 [spec §5]")
    extra["p1"] = p1

    # --- 事後追加 (事前登録外): 符号ランダム化 null。P1 の勝率が signal か下駄か
    nul, nlo, nhi = boot_median(rng, df["P1_null_A"])
    exc, elo, ehi = boot_median(rng, df["P1_excess_A"])
    n_exc = int((df["P1_excess_A"] > 0).sum())
    row("P1-null", "符号ランダム化 null の期待勝率 (層A)", nul, nlo, nhi,
        "(事後追加の診断)", "—",
        "self_proj = v_i·E[a_i x]·µ̂ で E[a_i x]·µ̂ >= 0 (condA の非負性) なので "
        "sign(self)=sign(v_i)。これが ± 対称なら勝率は signal ゼロでも 0.5 を超える。"
        "null 期待値 = 0.5 + 0.5·P(2|self| < |G·µ̂|)")
    row("P1-excess", "観測勝率 − null 期待勝率 (層A)", exc, elo, ehi,
        "CI がゼロ非含有 (事後追加)", "—",
        f"正だった seed 数 {n_exc}/10。ゼロを含むなら P1 の PASS は符号対称性の下駄で "
        f"説明され、**測るべきものを測っていない**")
    extra["p1_excess_ok"] = bool(elo > 0 or ehi < 0)
    extra["p1_null"], extra["p1_excess"], extra["n_exc"] = nul, exc, n_exc
    extra["p1_win"] = win

    # --- P1b: floor 3 点で符号一致
    sgns, det = [], []
    for f in FLOORS:
        v, lo, hi = boot_median(rng, df[f"P1b_med_{f:g}"])
        sgns.append(np.sign(v))
        det.append(f"floor={f:g}: {v:+.4f} (CI[{lo:+.4f},{hi:+.4f}], n={int(df[f'P1b_n_{f:g}'].sum())})")
        row(f"P1b-{f:g}", f"正規化版 対応差の中央値 (‖G‖>={f:g})", v, lo, hi,
            "(P1b の構成要素)", "—", "")
    agree = bool(len(set(sgns)) == 1 and sgns[0] != 0)
    row("P1b", "floor 3 点での符号一致", float(sgns[0]) if agree else np.nan,
        np.nan, np.nan, "3 点すべてで符号一致", "PASS" if agree else "FAIL",
        "; ".join(det) + ("" if agree else " — 不一致なので P1 は判定保留 [spec §3.2]"))
    extra["p1b"] = agree

    # --- P2: ゲート版
    w2, w2lo, w2hi = boot_median(rng, df["P2_win_A"])
    m2, m2lo, m2hi = boot_median(rng, df["P2_med_A"])
    n2 = int((df["P2_med_A"] > 0).sum())
    p2 = bool(w2 > 0.5 and m2lo > 0 and n2 >= SIGN_MIN)
    row("P2", "ゲート版 層A の勝率 (seed 中央値)", w2, w2lo, w2hi,
        "P1 と同基準", "PASS" if p2 else "FAIL",
        f"σ_g = sign(−F_gate/(2ηv)) を使用 (§3.1 のゲート版の直接の類推)。"
        f"対応差 {m2:+.4g} CI[{m2lo:+.4g},{m2hi:+.4g}], 符号数 {n2}/10")
    extra["p2"] = p2

    # --- 最終判定: P1・P1b・P2 の連言 [spec §5]
    if not s_ok:
        final = "中止 (P0 FAIL)"
    elif not extra["p1_excess_ok"]:
        final = "無効 (指標が符号対称性で説明される)"
    elif not extra["p1b"] or (p1 != p2):
        final = "判定保留"
    elif p1:
        final = "PASS"
    else:
        final = "FAIL"
    row("P1-final", "主判定の帰趨", np.nan, np.nan, np.nan,
        "P1 かつ P1b かつ P1・P2 が一致 (+ 事後の P1-excess)", final,
        "判定保留 = P1b の符号が floor で割れた、または P1 と P2 が食い違った [spec §5]。"
        "**無効** = 事後の符号ランダム化 null と区別できない (事前登録外の上書き。"
        "ratchet_log_0819 の P4 と同じ扱い)")
    extra["final"] = final

    # --- P3 (報告のみ): 層B / 層C / eval_loss 四分位
    for tag, lab in (("B", "層B (境界窓)"), ("all", "全記録点")):
        w, lo, hi = boot_median(rng, df[f"P1_win_{tag}"])
        row("P3", f"{lab} の勝率 (seed 中央値)", w, lo, hi, "(報告のみ)", "—", "")
    for grp, lab in (("in", "層C in {1,2,3,4,5,9}"), ("out", "層C out {0,6,7,8}")):
        sub = df[df["layerC"] == grp]["P1_win_A"]
        row("P3", f"{lab} の勝率 (seed 中央値)", float(np.median(sub)), np.nan, np.nan,
            "(報告のみ)", "—", f"n_seed={len(sub)}。seed 単位の固定集合 [spec §4]")
    for qi in (1, 2, 3, 4):
        c = f"P1_win_lossQ{qi}"
        if c in df:
            row("P3", f"eval_loss 第 {qi} 四分位 (層A) の勝率", float(np.nanmedian(df[c])),
                np.nan, np.nan, "(報告のみ・条件付き判定はしない)", "—", "")

    # --- P4 (報告のみ)
    s, slo, shi = boot_median(rng, df["share_A"])
    extra["share"] = s
    row("P4", "総寄与シェア |self|/(|self|+|rest|) (層A)", s, slo, shi,
        "(報告のみ)", "—",
        "有界・‖G‖ 非依存。**主判定にしない** — 「大きく寄与して正味ゼロ」は Q2b と両立 [spec §5]")
    return pd.DataFrame(V), extra


# ---------------------------------------------------------------- 図

def make_figures(outdir, df, series, s4_df, extra):
    fig_dir = os.path.join(outdir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    ax = axes[0, 0]
    for s in series:
        m = s["la"] & s["R"]
        ax.plot(s["step"][m] / 1e6, s["share"][m], lw=0.6, alpha=0.45)
    allsh = np.nanmedian(np.vstack([s["share"][s["la"] & s["R"]] for s in series]), axis=0)
    st0 = series[0]["step"][series[0]["la"] & series[0]["R"]]
    ax.plot(st0 / 1e6, allsh, "k-", lw=1.6, label="seed 中央値")
    ax.axhline(0.5, color="r", ls="--", lw=1, label="0.5 (self=rest)")
    ax.set_xlabel("step (×10⁶)"); ax.set_ylabel("|self| / (|self|+|rest|)")
    ax.set_title("P4 総寄与シェア (層A・seed 線)"); ax.legend(fontsize=8)

    ax = axes[0, 1]
    x = np.arange(len(df))
    ax.bar(x - 0.2, df["P1_win_A"], 0.4, label="層A (主判定)")
    ax.bar(x + 0.2, df["P1_win_B"], 0.4, label="層B (境界窓)")
    ax.axhline(0.5, color="r", ls="--", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(df["seed"])
    ax.set_xlabel("seed"); ax.set_ylabel("σ·(rest) > σ·(self) の割合")
    ax.plot(x, df["P1_null_A"], "kv", ms=6, label="符号ランダム化 null")
    ax.set_title("P1 勝率 — 観測は null と区別できない (事後)"); ax.legend(fontsize=7)

    ax = axes[1, 0]
    for j, f in enumerate(FLOORS):
        ax.scatter(np.full(len(df), j) + np.linspace(-.15, .15, len(df)),
                   df[f"P1b_med_{f:g}"], s=18, alpha=.7)
        ax.scatter([j], [np.median(df[f"P1b_med_{f:g}"])], marker="_", s=600, c="k")
    ax.axhline(0, color="r", ls="--", lw=1)
    ax.set_xticks(range(len(FLOORS))); ax.set_xticklabels([f"‖G‖≥{f:g}" for f in FLOORS])
    ax.set_ylabel("median(c_rest − c_self)")
    ax.set_title(f"P1b floor 感度 (符号一致: {extra['p1b']})")

    ax = axes[1, 1]
    lab = [f"[{int(r.step_lo/1000)}k,{int(min(r.step_hi,1000000)/1000)}k)"
           for r in s4_df.itertuples()]
    ax.bar(range(len(s4_df)), s4_df["cover_frac"], color="tab:green", alpha=.75)
    ax.set_xticks(range(len(s4_df))); ax.set_xticklabels(lab, fontsize=7, rotation=20)
    ax.set_ylabel("p̂=1 ユニットを含む記録点の割合")
    ax.set_title("S4 の被覆 (時間層別) [spec §9]")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "rowattr.png"), dpi=140)
    plt.close(fig)
    return fig_dir


# ---------------------------------------------------------------- summary.md

def _md(d, fmt=".4f"):
    if not len(d):
        return "(なし)\n"
    cols = list(d.columns)
    f = lambda v: (format(v, fmt) if isinstance(v, float) and np.isfinite(v)
                   else ("" if isinstance(v, float) else str(v)))
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in d.iterrows():
        out.append("| " + " | ".join(f(r[c]) for c in cols) + " |")
    return "\n".join(out) + "\n"


def write_summary(outdir, V, df, s2_df, s3_df, s3_got, s4_df, s4_meta, extra, meta):
    L = [f"# cell4_0821: G の µ̂ 整列の行帰属 (税 vs それ以外)", "",
         f"spec: `specs/spec_cell4_0821.md` / 生成 {meta['date']} / git `{meta['git_hash']}`", "",
         f"入力: `{meta['source']}/logs/seed*.npz` のみ (**再走なし**)。",
         f"seed {meta['n_seeds']} 本 × 記録点 {int(df['n_records'].iloc[0])}。", "",
         f"## 最終判定: **{extra['final']}**", "",
         "> **事前登録の P1 は字義では PASS するが、PASS と書いてはいけない。**",
         "> `self_proj = v_i·E[a_i·x]·µ̂` で、condA の非負性 (x ∈ {0,1}²⁰, a_i ≥ 0) により",
         "> `E[a_i·x]·µ̂ ≥ 0` なので **sign(self_proj) = sign(v_i)**。v の符号はユニット間で",
         f"> ほぼ対称 (層A で正の割合 {df['frac_self_pos_A'].mean():.3f}) なので、",
         "> 対応比較 `d = |G·µ̂| − 2σ·self` の勝率は **signal がゼロでも 0.5 を超える**",
         "> (どちらの符号でも片方は必ず勝つため)。符号ランダム化 null の期待勝率は解析的に",
         "> `0.5 + 0.5·P(2|self| < |G·µ̂|)` で、実測は",
         f"> **観測 {extra['p1_win']:.4f} vs null {extra['p1_null']:.4f} "
         f"(差 {extra['p1_excess']:+.4f}、CI がゼロを含む、正の seed {extra['n_exc']}/10)**。",
         "> 事前登録指標は**測るべきものを測っていない**。ratchet_log_0819 の P4 と同じ扱いで、",
         "> `verdict.csv` の `P1-final` を無効へ上書きした (事後・事前登録外)。", "",
         "> **代わりに読める事実**: 総寄与シェア `|self|/(|self|+|rest|)` は層A で",
         f"> **{extra['share']:.3f}** — 税は絶対値では rest とほぼ同等の大きさを出している。",
         "> しかし符号が v_i に従うためユニット間で相殺し、符号付き中央値はほぼ 0 になる",
         f"> (層A の median c_self = {df['cself_A'].median():+.4f})。これは Q2b の",
         "> 「大きく寄与しているのに正味ゼロ」と整合する。**Q12 の防衛にはこの per-unit の",
         "> 枠組みでは足りず、amended 指標の設計が要る。**", "",
         "## サニティ", "", "### S2 — E1 の再現", "", _md(s2_df, ".10f"),
         "", "### S3 — spec §1 の再現表", "", _md(s3_df, ".4f"),
         f"\n再計算した上位 6 seed = {s3_got['seeds']}\n",
         "", "### S4 — p̂=1 ユニットでの独立 2 経路一致", "",
         f"- 対象ユニット {s4_meta['s4_n_units']} 個 "
         f"({s4_meta['s4_frac_units']:.4%})、該当記録点 {s4_meta['s4_frac_points']:.2%}",
         f"- 相対誤差 中央値 {s4_meta['s4_med_rel_err']:.2e} / p99 "
         f"{s4_meta['s4_p99_rel_err']:.2e} / **max {s4_meta['s4_max_rel_err']:.2e}** "
         f"(閾値 {s4_meta['s4_threshold']:g})", "",
         "被覆は前半に偏る (spec §9 の指示どおり時間層別で掲載):", "", _md(s4_df, ".2e"),
         "", "## 判定表", "", _md(V, ".4f"),
         "", "## seed 別", "",
         _md(df[["seed", "layerC", "n_layerA", "n_layerB", "P1_win_A", "P1_med_A",
                 "P2_win_A", "cself_A", "crest_A", "share_A"]], ".4f"),
         "", "## 除外率 [spec §3 Phase 0]", "",
         f"- ユニット除外 (p̂<{TAU} または |v|<{VMIN}): "
         f"{df['frac_unit_dropped'].mean():.4%} (seed 平均)",
         f"- alive=0 の記録点: {df['frac_alive0'].mean():.4%} (seed 平均)",
         f"- σ=0 (G·µ̂ が厳密に 0) の記録点: {int(df['n_sigma0'].sum())} 個 (全 seed 合計)", "",
         "## 逸脱節", "",
         "1. **spec §5 の「対応差」の解釈**。P1 の統計量欄は勝率、基準欄は「対応差の",
         "   bootstrap CI 下端 > 0」と書かれており、対応差が勝率のことか σ·(rest−self) の",
         "   生の差のことか一意に読めない。**両方を計算し、3 条件 (勝率 > 0.5・生の差の",
         "   CI 下端 > 0・符号数 >= 9/10) の連言を P1 の PASS 条件とした**。どちらの読みでも",
         "   PASS が緩まない向きの解釈である。",
         "2. **bootstrap の点推定を中央値にした**。`figures_ratchet_log.boot_ci` は平均を",
         "   返すが、spec §5 は「seed 中央値」を要求している。ratchet_log_0819 は点推定に",
         "   中央値・CI に平均版を使い注記で処理していたが、ここは中央値で統一した",
         "   (`boot_median`)。",
         "3. **P2 の σ**。spec は「§3.1 と同じ比較」とだけ書いており σ の取り方を指定して",
         "   いない。ゲート版の合計は per-unit なので σ_g = sign(−F_gate/(2ηv)) を採った",
         "   (G·µ̂ の符号を流用すると、ゲート版の合計と符号が食い違う点で比較が反転する)。",
         "4. **事後追加 (事前登録外)**: `P1-null` / `P1-excess` と、それによる `P1-final` の",
         "   無効化。spec §5 は符号ランダム化 null を要求していない。P1 の統計量が",
         "   sign(self)=sign(v_i) の対称性だけで 0.5 を超えることに実行後に気付いたため、",
         "   ratchet_log_0819 の P4 (「閾値は満たすが PASS と書いてはいけない」) と同じ",
         "   扱いで追加した。**事前登録の P1 行はそのまま残してある** (字義の結果は",
         "   `PASS (字義)`)。",
         "5. 新たな生ログを作らないので出力は全て commit 可能 [spec §10]。", "",
         "## 出力", "",
         "- `verdict.csv` — P0–P4 の全行", "- `per_seed_metrics.csv` — seed ごとの全指標",
         "- `figures/rowattr.png` — シェア時系列 / P1 勝率 / P1b floor 感度 / S4 被覆", ""]
    p = os.path.join(outdir, "summary.md")
    with open(p, "w") as fh:
        fh.write("\n".join(L))
    return p


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="?",
                    default=os.path.join(ROOT, "results", "ratchet_log_0819"))
    ap.add_argument("--outdir", default=os.path.join(ROOT, "results", "cell4_0821"))
    args = ap.parse_args()

    t0 = time.time()
    cfg = load_config(os.path.join(args.results, "config_used.yaml"))
    P = cfg["ratchet"]
    period = int(cfg["condA"]["T_values"][0])
    half_w, bulk_every = int(P["boundary_window"]), int(P["bulk_every"])

    seeds = load_seeds(args.results)
    seeds.sort(key=lambda d: str(d["run_id"]))                 # run_id ソート [spec §5]
    print(f"loaded {len(seeds)} seeds, {len(seeds[0]['step'])} 記録点, "
          f"period={period}, half_w={half_w}", flush=True)

    print("S2: E1 を再現中 ...", flush=True)
    s2_ok, s2_df = check_s2(seeds)
    print(s2_df.to_string(index=False), flush=True)
    if not s2_ok:
        raise SystemExit("[cell4] S2 FAIL — 解析コードのバグ。判定に進まない (spec §9)")

    s3_ok, s3_df, s3_got = check_s3(seeds)
    print("S3:", "PASS" if s3_ok else "FAIL", flush=True)
    print(s3_df.to_string(index=False), flush=True)
    if not s3_ok:
        raise SystemExit("[cell4] S3 FAIL — §1 の再現表が出ない。中止 (spec §9)")

    s4_ok, s4_df, s4_meta = check_s4(seeds, period, half_w, bulk_every)
    print(f"S4: {'PASS' if s4_ok else 'FAIL'} "
          f"(max 相対誤差 {s4_meta['s4_max_rel_err']:.2e} < 1e-6)", flush=True)

    s_ok = bool(s2_ok and s3_ok and s4_ok)
    if not s_ok:
        raise SystemExit("[cell4] P0 FAIL — 判定に進まない (spec §5)")

    df, series = per_seed(seeds, period, half_w, bulk_every)
    rng = np.random.default_rng(BOOT_SEED)
    V, extra = judge(df, rng, s_ok)

    os.makedirs(args.outdir, exist_ok=True)
    df.to_csv(os.path.join(args.outdir, "per_seed_metrics.csv"), index=False)
    V.to_csv(os.path.join(args.outdir, "verdict.csv"), index=False)
    fig_dir = make_figures(args.outdir, df, series, s4_df, extra)

    meta = dict(date=time.strftime("%Y-%m-%d %H:%M:%S"), git_hash=git_hash(),
                spec="specs/spec_cell4_0821.md", source=args.results,
                elapsed_sec=round(time.time() - t0, 1),
                omp_num_threads=os.environ.get("OMP_NUM_THREADS", "(未設定)"),
                python=platform.python_version(), numpy=np.__version__,
                pandas=pd.__version__, bootstrap_B=BOOT_N, bootstrap_seed=BOOT_SEED,
                n_seeds=len(seeds), period=period, half_window=half_w,
                bulk_every=bulk_every, tau=TAU, v_min=VMIN, floors=list(FLOORS),
                s2_pass=bool(s2_ok), s3_pass=bool(s3_ok), **s4_meta,
                final=extra["final"], p1=extra["p1"], p1b=extra["p1b"], p2=extra["p2"])
    with open(os.path.join(args.outdir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1, default=str, ensure_ascii=False)
    sp = write_summary(args.outdir, V, df, s2_df, s3_df, s3_got, s4_df, s4_meta,
                       extra, meta)

    print(V.to_string(index=False), flush=True)
    print(f"-> {args.outdir}/verdict.csv, per_seed_metrics.csv, {sp}, {fig_dir}/",
          flush=True)
    print(f"CELL4 DONE — 最終判定: {extra['final']}", flush=True)
    return df, V


if __name__ == "__main__":
    main()
