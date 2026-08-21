"""cell4_0821 追補: G の µ̂ 整列を全体レベルで G_out / G_teach に割る [spec_cell4_0821_addendum]。

  OMP_NUM_THREADS=1 .venv/bin/python -m analysis.cell4.global_split \
      [results/ratchet_log_0819] [--outdir results/cell4_0821/addendum]

**再学習なし・再走なし**。本体 `analysis/cell4/rowattr.py` の per-unit 帰属が
「sign(self 射影) = sign(v_i) の対称性」で潰れて無効だったため (spec 追補 §1-2)、
per-unit を捨てて**全体レベル**で割り直す。

    δ = Σ_j v_j·a_j + c − y  より恒等的に
    G = E[δx] = Σ_j v_j·E[a_j·x]  +  (c·µ − E[y·x])
                └──── G_out ────┘     └─── G_teach ───┘

    F_self,j = −2η·v_j²·(E[a_j·x]·µ̂)  なので  b_j := v_j·(E[a_j·x]·µ̂) = −F_self,j/(2η·v_j)
    G_out·µ̂ = Σ_j b_j          （j は全 100 ユニット。|v| の下限ゲートは不要）
    G_teach·µ̂ = G·µ̂ − G_out·µ̂

判定は spec 追補 §5 の A0-A5 が唯一の正で、本モジュールはそれを実装するだけ。特に

  A0  前提ゲート (S1-S4)。FAIL なら判定に進まない
  A1  **主判定**: σ·(G_out·µ̂) − σ·(G_teach·µ̂)、σ = sign(G·µ̂)。‖G‖ を含まない
  A2  **符号ランダム化 null との比較**。A2 FAIL なら A1 の符号を解釈しない
  A3  k_out / k_teach の floor 3 点感度 (報告のみ)
  A4  集中度 (報告のみ)   A5  数値条件 κ (報告のみ・主判定では除外しない)

を先に読むこと。本追補の全数値は **amended (事後追加)** の格で運ぶ [追補 §8]。

出力 (すべて --outdir の中): verdict_addendum.csv / summary_addendum.md /
per_seed_metrics_addendum.csv / meta.json / figures/。
"""
import argparse
import json
import os
import platform
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from src.common import ROOT, load_config                                  # noqa: E402
from src.figures_ratchet_log import TAU, load_seeds                       # noqa: E402
from analysis.cell4.rowattr import (BOOT_SEED, BOOT_N, FLOORS, SIGN_MIN,   # noqa: E402
                                    _md, git_hash, grid_layers, boot_median)

PERM_N = 1000                   # 符号ランダム化 null の反復数 [追補 §4]
KAPPA_HI = 1e4                  # A5: float32 相対誤差が ~1e-3 を超える目安
LAYER_C_IN = {1, 2, 3, 4, 5, 9}


# ---------------------------------------------------------------- 全体レベル分割

def global_terms(d):
    """1 seed ぶんの G_out / G_teach と補助量。全て float64。

    b_j = v_j·(E[a_j·x]·µ̂) = −F_self,j/(2η·v_j)。E[a_j·x]·µ̂ >= 0 (condA の非負性) なので
    |b_j| = |v_j|·(E[a_j·x]·µ̂) であり、符号ランダム化 null は Σ_j ε_j·|b_j| で作れる。
    dead (p̂=0) は F_self = 0 なので b_j = 0 で寄与しない (S4 で厳密性を確認)。"""
    eta = float(d["lr"])
    v = d["v"].astype(np.float64)
    term = d["F_self"].astype(np.float64) / v          # v に厳密ゼロなし (S4 で確認)
    b = -term / (2.0 * eta)                            # [n,h]
    Gdm = d["G_dot_mu"].astype(np.float64)
    Gn = np.linalg.norm(d["G"].astype(np.float64), axis=1)
    G_out = b.sum(axis=1)
    G_teach = Gdm - G_out
    sigma = np.sign(Gdm)
    rec_ok = sigma != 0                                # σ=0 は向きが定義できない
    # A1 の主判定量: σ·G_out·µ̂ − σ·G_teach·µ̂ = σ·(2·G_out·µ̂ − G·µ̂)。‖G‖ を含まない
    D = np.where(rec_ok, sigma * (2.0 * G_out - Gdm), np.nan)
    absb = np.abs(b)
    kappa = absb.sum(axis=1) / np.maximum(np.abs(b.sum(axis=1)), 1e-300)
    srt = -np.sort(-absb, axis=1)
    den = absb.sum(axis=1)
    safe = np.where(den > 0, den, np.nan)          # 全ユニット dead の記録点は寄与ゼロ
    return dict(seed=int(d["seed"]), step=d["step"], eta=eta, Gdm=Gdm, Gn=Gn,
                sigma=sigma, rec_ok=rec_ok, G_out=G_out, G_teach=G_teach, D=D,
                absb=absb, kappa=kappa,
                k_out=np.where(rec_ok, sigma * G_out / Gn, np.nan),
                k_teach=np.where(rec_ok, sigma * G_teach / Gn, np.nan),
                top1=srt[:, 0] / safe, top3=srt[:, :3].sum(axis=1) / safe,
                n_alive=(d["p_hat"] >= TAU).sum(axis=1),
                cos=np.abs(d["cos_G_mu"].astype(np.float64)))


def _med(x):
    x = x[np.isfinite(x)]
    return float(np.median(x)) if x.size else np.nan


# ---------------------------------------------------------------- サニティ [追補 §7]

def check_s2(outdir_main, df):
    """S2: 本体 results/cell4_0821/verdict.csv の値を再現する。"""
    ref = pd.read_csv(os.path.join(outdir_main, "verdict.csv"))
    get = lambda i: float(ref[ref["id"] == i]["point"].iloc[0])
    rows = []
    for i, name, got in (("P1", "層A 勝率", df["ref_P1_win_A"].median()),
                         ("P1-null", "符号ランダム化 null の期待勝率", df["ref_P1_null_A"].median()),
                         ("P4", "総寄与シェア", df["ref_share_A"].median())):
        exp = get(i)
        ok = bool(abs(got - exp) <= 1e-9)
        rows.append(dict(id="S2", statistic=name, main_verdict=exp, recomputed=got,
                         abs_diff=abs(got - exp), ok=ok))
    df2 = pd.DataFrame(rows)
    return bool(df2["ok"].all()), df2


def check_s3(seeds):
    """S3: 独立 2 経路の一致 (恒真でない) [追補 §7]。

    (a) p̂=1 のユニットで −F_gate/(2ηv) == G·µ̂
    (b) G_out の 2 経路: F_self と (F_gate − F_rest)。**残差はスケール付きで測る** (修正1)。
        /|F_self| で測ると F_self が F_rest に比べ極端に小さい点で桁落ちが支配し、
        測っているのが probe の一貫性ではなく float32 の丸めになる。"""
    bins = [(0, 1), (1, 10000), (10000, 100000), (100000, 500000), (500000, 10 ** 9)]
    ea, per, nb = [], {b: [[], 0, 0] for b in bins}, 0
    sc, rel = [], []
    for d in seeds:
        eta = float(d["lr"])
        v = d["v"].astype(np.float64)
        Fs, Fr, Fg = [d[k].astype(np.float64) for k in ("F_self", "F_rest", "F_gate")]
        step, Gdm = d["step"], d["G_dot_mu"].astype(np.float64)
        m = (d["p_hat"] == 1.0) & (np.abs(v) >= 1e-2)
        if m.any():
            lhs = -Fg[m] / (2 * eta * v[m])
            rhs = np.repeat(Gdm[:, None], m.shape[1], axis=1)[m]
            e = np.abs(lhs - rhs) / np.maximum(np.abs(rhs), 1e-300)
            ea.append(e)
            ridx = np.repeat(step[:, None], m.shape[1], axis=1)[m]
            for b in bins:
                s_ = (ridx >= b[0]) & (ridx < b[1])
                if s_.any():
                    per[b][0].append(e[s_])
                rs = (step >= b[0]) & (step < b[1])
                per[b][1] += int(m[rs].any(axis=1).sum())
                per[b][2] += int(rs.sum())
        mm = np.abs(Fs) > 0
        nb += int(mm.sum())
        r = np.abs((Fg - Fr) - Fs)[mm]
        sc.append(r / (np.abs(Fg[mm]) + np.abs(Fr[mm])))
        rel.append(r / np.abs(Fs[mm]))
    ea, sc, rel = map(np.concatenate, (ea, sc, rel))
    cov = pd.DataFrame([dict(step_lo=b[0], step_hi=b[1],
                             cover_frac=per[b][1] / max(per[b][2], 1),
                             max_rel_err=float(np.concatenate(per[b][0]).max())
                             if per[b][0] else np.nan) for b in bins])
    ok_a, ok_b = bool(ea.max() < 1e-6), bool(sc.max() < 1e-6)
    rows = [dict(id="S3a", statistic="p̂=1 で −F_gate/(2ηv) == G·µ̂ (相対)",
                 p50=float(np.median(ea)), p99=float(np.percentile(ea, 99)),
                 max=float(ea.max()), tol=1e-6, ok=ok_a),
            dict(id="S3b", statistic="G_out 2 経路 スケール付き残差 (採用・修正1)",
                 p50=float(np.median(sc)), p99=float(np.percentile(sc, 99)),
                 max=float(sc.max()), tol=1e-6, ok=ok_b),
            dict(id="S3b-ref", statistic="同 /|F_self| 版 (参考・判定に使わない)",
                 p50=float(np.median(rel)), p99=float(np.percentile(rel, 99)),
                 max=float(rel.max()), tol=np.nan, ok=None)]
    return bool(ok_a and ok_b), pd.DataFrame(rows), cov, int(nb)


def check_s4(seeds):
    """S4: dead (p̂=0) の F_self が厳密に 0 (G_out の和に不定項が入らないこと)。"""
    n, mx, nz, vmin = 0, 0.0, 0, np.inf
    for d in seeds:
        v, Fs = d["v"].astype(np.float64), d["F_self"].astype(np.float64)
        dead = d["p_hat"] == 0.0
        n += int(dead.sum())
        nz += int((v == 0).sum())
        vmin = min(vmin, float(np.abs(v[v != 0]).min()))
        if dead.any():
            mx = max(mx, float(np.abs(Fs[dead]).max()))
    ok = bool(mx == 0.0 and nz == 0)
    return ok, pd.DataFrame([
        dict(id="S4", statistic="p̂=0 での max|F_self|", value=mx, tol=0.0, ok=mx == 0.0),
        dict(id="S4", statistic="v==0 の要素数 (0/0 が起きないこと)", value=float(nz),
             tol=0.0, ok=nz == 0),
        dict(id="S4", statistic="min|v| (非ゼロ)", value=vmin, tol=np.nan, ok=None)])


# ---------------------------------------------------------------- seed 集計

def per_seed(seeds, period, half_w, bulk_every, main_dir):
    """seed ごとに層別の指標を作る [追補 §5]。集計は run 単位で中央値。"""
    from analysis.cell4.rowattr import row_projections
    rows, series, packs = [], [], []
    for d in seeds:
        T = global_terms(d)
        la, lb = grid_layers(T["step"], period, half_w, bulk_every)
        R = T["rec_ok"]
        r = dict(seed=T["seed"], layerC="in" if T["seed"] in LAYER_C_IN else "out",
                 n_layerA=int((la & R).sum()), n_layerB=int((lb & R).sum()),
                 n_sigma0=int((~R).sum()))
        for tag, mk in (("A", la), ("B", lb), ("all", np.ones_like(la))):
            r[f"A1_med_{tag}"] = _med(T["D"][mk & R])
            r[f"kout_{tag}"] = _med(T["k_out"][mk & R])
            r[f"kteach_{tag}"] = _med(T["k_teach"][mk & R])
        for f in FLOORS:                                   # A3
            sel = la & R & (T["Gn"] >= f)
            r[f"kout_f{f:g}"] = _med(T["k_out"][sel])
            r[f"kteach_f{f:g}"] = _med(T["k_teach"][sel])
        sel = la & R                                       # A4 / A5
        r["top1_A"], r["top3_A"] = _med(T["top1"][sel]), _med(T["top3"][sel])
        r["n_alive_A"] = _med(T["n_alive"][sel].astype(float))
        r["kappa_p50"], r["kappa_p95"] = _med(T["kappa"][sel]), \
            float(np.percentile(T["kappa"][sel], 95))
        r["kappa_max"] = float(T["kappa"][sel].max())
        r["frac_kappa_hi"] = float((T["kappa"][sel] > KAPPA_HI).mean())
        r["A1_med_A_lowkappa"] = _med(T["D"][sel & (T["kappa"] <= KAPPA_HI)])

        # S2 用: 本体 rowattr の 3 値を同じ経路で再計算する
        Pm = row_projections(d)
        mk = la & Pm["rec_ok"]
        x = Pm["d_raw"][mk]
        x = x[np.isfinite(x)]
        r["ref_P1_win_A"] = float((x > 0).mean())
        gabs = np.repeat(np.abs(Pm["Gdm"])[:, None], Pm["self_proj"].shape[1], axis=1)
        u, g2 = Pm["self_proj"][mk], gabs[mk]
        fin = np.isfinite(u)
        r["ref_P1_null_A"] = float(0.5 + 0.5 * (2 * np.abs(u[fin]) < g2[fin]).mean())
        sh = Pm["share_self"][mk]
        r["ref_share_A"] = _med(sh)

        rows.append(r)
        packs.append(dict(seed=T["seed"], absb=T["absb"][la & R], Gdm=T["Gdm"][la & R],
                          sigma=T["sigma"][la & R]))
        series.append(dict(seed=T["seed"], step=T["step"], m=la & R, Gn=T["Gn"],
                           k_out=T["k_out"], k_teach=T["k_teach"], cos=T["cos"]))
    return pd.DataFrame(rows), series, packs


def null_distribution(packs, n_perm=PERM_N, seed=BOOT_SEED):
    """符号ランダム化 null [追補 §4]。σ は固定 (修正2)。

    G_out^null·µ̂ = Σ_j ε_j·|b_j| なので E[σ·G_out^null·µ̂] = 0、
    したがって null の A1 統計量の期待値は −|G·µ̂| になる。
    観測と同じ集計 (run 中央値 → seed 中央値) を各置換で行う。"""
    rng = np.random.default_rng(seed)
    out = np.empty(n_perm)
    for p in range(n_perm):
        med = []
        for pk in packs:
            eps = rng.integers(0, 2, pk["absb"].shape) * 2 - 1
            g_out = (eps * pk["absb"]).sum(axis=1)
            med.append(np.median(pk["sigma"] * (2.0 * g_out - pk["Gdm"])))
        out[p] = np.median(med)
    return out


# ---------------------------------------------------------------- 判定 [追補 §5]

def judge(df, nulls, rng, s_ok):
    V, extra = [], {}

    def row(i, stat, pt, lo, hi, thr, res, note=""):
        V.append(dict(id=i, statistic=stat, point=pt, ci_lo=lo, ci_hi=hi,
                      threshold=thr, result=res, note=note))

    row("A0", "前提ゲート S1-S4", np.nan, np.nan, np.nan, "S1-S4 全 PASS",
        "PASS" if s_ok else "FAIL", "FAIL なら A1 以降を計算しない [追補 §5]")

    # --- A1
    m, lo, hi = boot_median(rng, df["A1_med_A"])
    nsign = int((df["A1_med_A"] > 0).sum())
    nneg = int((df["A1_med_A"] < 0).sum())
    a1_nonzero = bool(lo > 0 or hi < 0)
    a1_sign = bool(max(nsign, nneg) >= SIGN_MIN)
    a1 = bool(a1_nonzero and a1_sign)
    who = "G_out 優位" if m > 0 else "G_teach 優位"
    row("A1", "層A の σ·(G_out·µ̂) − σ·(G_teach·µ̂) (seed 中央値)", m, lo, hi,
        f"CI がゼロ非含有 かつ 符号数 >= {SIGN_MIN}/10",
        f"{who}" if a1 else "判別不能",
        f"CI ゼロ非含有={a1_nonzero}, 正 {nsign}/10・負 {nneg}/10 (>= {SIGN_MIN}={a1_sign})。"
        f"符号が正 = G_out 優位、負 = G_teach 優位 [追補 §5]")
    extra["a1"], extra["a1_point"], extra["who"] = a1, m, who

    # --- A2: 符号ランダム化 null
    nlo, nhi = float(np.percentile(nulls, 2.5)), float(np.percentile(nulls, 97.5))
    a2 = bool(m < nlo or m > nhi)
    p_two = float(2 * min((nulls <= m).mean(), (nulls >= m).mean()))
    row("A2", "A1 統計量 vs 符号ランダム化 null (両側 95%)", m, nlo, nhi,
        "観測が null の 95 パーセンタイル外", "PASS" if a2 else "FAIL",
        f"null は v_j の符号を i.i.d. ±1 に引き直し {PERM_N} 回、σ は固定 (修正2)。"
        f"null 中央値 {np.median(nulls):+.4g}、両側 p={p_two:.4f}。"
        f"**FAIL なら A1 の符号は解釈しない** [追補 §5]")
    extra["a2"], extra["null_lo"], extra["null_hi"] = a2, nlo, nhi
    extra["null_med"], extra["p_two"] = float(np.median(nulls)), p_two

    # --- 最終判定 [追補 §6]
    if not s_ok:
        final = "中止 (A0 FAIL)"
    elif not a2:
        final = "判定保留 (null と区別できない)"
    elif not a1:
        final = "判別不能"
    else:
        final = who
    if not s_ok:
        fnote = "A0 で中止"
    elif not a2:
        fnote = ("観測が符号ランダム化 null と区別できない。追補 §6 の第3行 "
                 "(Q12 を判定保留で閉じる・新規アームは起案しない)")
    elif not a1:
        fnote = ("**この結末は追補 §6 の処置表に無い**。A2 は PASS (G_out·µ̂ は "
                 "ランダム符号では説明できない) だが、A1 は G_out と G_teach の大小が "
                 "seed をまたいで一定しない。§6 の 3 行はいずれも A1 に符号が付くか "
                 "A2 が FAIL であることを前提にしている。処置の決定は台帳側に差し戻す")
    else:
        fnote = f"追補 §6 の {'第1行' if extra['a1_point'] > 0 else '第2行'} に対応"
    row("A-final", "主判定の帰趨", np.nan, np.nan, np.nan,
        "A1 かつ A2", final, fnote)
    extra["fnote"] = fnote
    extra["final"] = final

    # --- A3 (報告のみ)
    sg = []
    for f in FLOORS:
        ko, klo, khi = boot_median(rng, df[f"kout_f{f:g}"])
        kt, tlo, thi = boot_median(rng, df[f"kteach_f{f:g}"])
        sg.append(np.sign(ko - kt))
        row("A3", f"k_out (‖G‖>={f:g})", ko, klo, khi, "(報告のみ)", "—", "")
        row("A3", f"k_teach (‖G‖>={f:g})", kt, tlo, thi, "(報告のみ)", "—", "")
    agree = bool(len(set(sg)) == 1)
    row("A3", "floor 3 点で k_out − k_teach の符号一致", float(sg[0]), np.nan, np.nan,
        "(報告のみ)", "—", "割れたら解釈不能と明記 [追補 §5]" if not agree else "一致")
    extra["a3_agree"] = agree

    # --- A4 / A5 (報告のみ)
    for c, name in (("top1_A", "G_out·µ̂ の上位 1 ユニット寄与割合"),
                    ("top3_A", "同 上位 3 ユニット"),
                    ("n_alive_A", "alive 数 (層A 中央値)")):
        pt, lo2, hi2 = boot_median(rng, df[c])
        row("A4", name, pt, lo2, hi2, "(報告のみ)", "—",
            "alive 中央値 9 の系なので少数支配の可能性を明示 [追補 §5]")
    row("A5", "κ = Σ|項|/|Σ項| (層A・seed 中央値の中央値)", float(df["kappa_p50"].median()),
        np.nan, np.nan, "(報告のみ)", "—",
        f"p95 中央値 {df['kappa_p95'].median():.2f}、max {df['kappa_max'].max():.3e}。"
        f"κ·1.2e-7 が float32 由来の相対誤差の目安")
    kh, khlo, khhi = boot_median(rng, df["A1_med_A_lowkappa"])
    row("A5", f"κ<={KAPPA_HI:g} に限った A1 (感度)", kh, khlo, khhi,
        "(報告のみ・主判定では除外しない)", "—",
        f"κ>{KAPPA_HI:g} の記録点は層A の {df['frac_kappa_hi'].mean():.4%} (seed 平均)")
    return pd.DataFrame(V), extra


# ---------------------------------------------------------------- 図

def make_figures(outdir, df, series, nulls, extra):
    fig_dir = os.path.join(outdir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    fig, ax = plt.subplots(2, 2, figsize=(13, 8.5))

    a = ax[0, 0]
    # k は記録点ごとの揺れが大きく時系列では読めない。A3 が実際に報告しているのは
    # 中央値なので分布で見せる。‖G‖→0 の発散を避けるため A3 の中間 floor を掛ける。
    fl = FLOORS[1]
    ko = np.concatenate([s["k_out"][s["m"] & (s["Gn"] >= fl)] for s in series])
    kt = np.concatenate([s["k_teach"][s["m"] & (s["Gn"] >= fl)] for s in series])
    ko, kt = ko[np.isfinite(ko)], kt[np.isfinite(kt)]
    bins = np.linspace(-3, 4, 120)
    a.hist(ko, bins=bins, alpha=.55, color="tab:blue", label=f"k_out (中央値 {np.median(ko):+.3f})")
    a.hist(kt, bins=bins, alpha=.55, color="tab:orange", label=f"k_teach (中央値 {np.median(kt):+.3f})")
    a.axvline(np.median(ko), color="tab:blue", lw=1.6)
    a.axvline(np.median(kt), color="tab:orange", lw=1.6)
    a.axvline(0, color="k", lw=.8)
    a.set_xlabel("σ·(項·µ̂)/‖G‖    (k_out + k_teach = |cos(G,µ̂)|)")
    a.set_ylabel("記録点数")
    a.set_title(f"A3 分布 (層A・‖G‖≥{fl:g}・全 seed プール)")
    a.legend(fontsize=8)

    a = ax[0, 1]
    a.hist(nulls, bins=50, color="0.75", label=f"符号ランダム化 null ({PERM_N})")
    a.axvline(extra["a1_point"], color="r", lw=2, label=f"観測 {extra['a1_point']:+.4g}")
    a.axvline(extra["null_lo"], color="k", ls="--", lw=1)
    a.axvline(extra["null_hi"], color="k", ls="--", lw=1, label="null 95%")
    a.set_xlabel("σ·G_out·µ̂ − σ·G_teach·µ̂ (seed 中央値)")
    a.set_title(f"A2 (両側 p={extra['p_two']:.4f}) → {extra['final']}"); a.legend(fontsize=7)

    a = ax[1, 0]
    x = np.arange(len(df))
    a.bar(x, df["A1_med_A"], color=["tab:green" if u > 0 else "tab:red"
                                   for u in df["A1_med_A"]])
    a.axhline(0, color="k", lw=.8)
    a.set_xticks(x); a.set_xticklabels(df["seed"])
    a.set_xlabel("seed"); a.set_ylabel("A1 統計量 (run 中央値)")
    a.set_title("A1 seed 別 (緑 = G_out 優位 / 赤 = G_teach 優位)")

    a = ax[1, 1]
    a.bar(x - .2, df["top1_A"], .4, label="上位 1 ユニット")
    a.bar(x + .2, df["top3_A"], .4, label="上位 3 ユニット")
    a.set_xticks(x); a.set_xticklabels(df["seed"])
    a.set_xlabel("seed"); a.set_ylabel("Σ|b_j| に占める割合")
    a.set_title("A4 G_out の集中度"); a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "global_split.png"), dpi=140)
    plt.close(fig)
    return fig_dir


def write_summary(outdir, V, df, s2, s3, cov, s4, extra, meta):
    L = ["# cell4_0821 追補: G の µ̂ 整列の全体レベル分割 (G_out vs G_teach)", "",
         f"spec: `specs/spec_cell4_0821_addendum.md` / 生成 {meta['date']} / git `{meta['git_hash']}`", "",
         "**格: amended (事後追加)。** 本体 `spec_cell4_0821` の P1/P2 の差し替えであり、",
         "引用のたびにそのラベルを運ぶこと [追補 §8]。本体の成果物は上書きしていない。", "",
         f"入力: `{meta['source']}/logs/seed*.npz` のみ (**再走なし**)。", "",
         f"## 最終判定: **{extra['final']}**", "",
         f"- A1 統計量 (層A・seed 中央値) = **{extra['a1_point']:+.4g}** "
         f"→ 点推定としては {extra['who']}",
         f"- 符号ランダム化 null ({PERM_N} 回) の中央値 {extra['null_med']:+.4g}、"
         f"95% 区間 [{extra['null_lo']:+.4g}, {extra['null_hi']:+.4g}]、両側 **p = {extra['p_two']:.4f}**",
         f"- A2 = {'PASS' if extra['a2'] else 'FAIL'}"
         + ("" if extra["a2"] else " → **A1 の符号は解釈しない**（本体 P1 と同じ失敗にあたるため）"),
         f"- A1 の seed 別符号: 正 {int((df['A1_med_A'] > 0).sum())}/10・"
         f"負 {int((df['A1_med_A'] < 0).sum())}/10", "",
         f"> {extra['fnote']}", "",
         "**A2 PASS の読み方**: null では E[σ·G_out^null·µ̂] = 0 なので統計量の期待値は",
         f"−|G·µ̂| になる (実測の null 中央値 {extra['null_med']:+.4g})。観測 "
         f"{extra['a1_point']:+.4g} はそれより有意に**高い**ので、",
         "**v_j の符号と活性の大きさには実際に対応があり、G_out·µ̂ はランダム符号の √N",
         "スケールでは説明できない**。ただしその大きさは G_teach·µ̂ と同程度で、どちらが",
         "優位かは seed ごとに入れ替わる (A1 判別不能)。", "",
         "**A3 (正規化版・報告のみ) は 3 floor すべてで k_teach > k_out** で符号が一致して",
         "いる。A1 (生スケール・‖G‖ 非依存) は大 ‖G‖ の記録点に重みが寄り、A3 は記録点を",
         "等重みで見るため、重みの違いで結論が変わっている。**事前登録の主判定は A1 なので",
         "判定は「判別不能」のまま**であり、A3 の向きを判定として引いてはいけない。", "",
         "## サニティ", "", "### S2 — 本体 verdict.csv の再現", "", _md(s2, ".10f"),
         "", "### S3 — 独立 2 経路の一致", "", _md(s3, ".2e"),
         "", "S3a の被覆 (時間層別):", "", _md(cov, ".2e"),
         "", "### S4 — dead の F_self が厳密 0 / v の非ゼロ性", "", _md(s4, ".3e"),
         "", "## 判定表", "", _md(V, ".4f"),
         "", "## seed 別", "",
         _md(df[["seed", "layerC", "n_layerA", "A1_med_A", "kout_A", "kteach_A",
                 "top1_A", "top3_A", "n_alive_A", "kappa_p50", "kappa_max"]], ".4f"),
         "", "## 逸脱節", "",
         "1. **§7 S3(b) の残差の分母** を `/|F_self|` から `/(|F_gate|+|F_rest|)` へ変更",
         "   (spec に修正1 として記載済み、commit 前レビューによる)。字義では max 6.31e-2 で",
         "   A0 中止になり、測っているのが probe の一貫性ではなく float32 の丸めになるため。",
         "   参考として `/|F_self|` 版も判定表外に併記した。",
         "2. **§4 の σ を null の下でも固定** (修正2)。G·µ̂ を観測値に固定する設計から従う。",
         "3. **A5 (条件数の報告) を追加** (修正3)。主判定では除外していない。",
         "4. `boot_median` は本体 `rowattr.py` と同じく点推定・CI とも中央値で統一。",
         "5. **実現した結末 (A1 判別不能 かつ A2 PASS) が §6 の処置表に無い**。§6 の 3 行は",
         "   いずれも「A1 に符号が付く」か「A2 が FAIL」を前提としている。本モジュールは",
         "   判定を `判別不能` と記録するにとどめ、現論文への処置は決定していない。", "",
         "## 出力", "", "- `verdict_addendum.csv` — A0–A5 の全行",
         "- `per_seed_metrics_addendum.csv` — seed ごとの全指標",
         "- `figures/global_split.png` — k_out/k_teach 時系列 / A2 null / A1 seed 別 / A4 集中度", ""]
    p = os.path.join(outdir, "summary_addendum.md")
    with open(p, "w") as fh:
        fh.write("\n".join(L))
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="?",
                    default=os.path.join(ROOT, "results", "ratchet_log_0819"))
    ap.add_argument("--outdir",
                    default=os.path.join(ROOT, "results", "cell4_0821", "addendum"))
    ap.add_argument("--main", default=os.path.join(ROOT, "results", "cell4_0821"))
    args = ap.parse_args()

    t0 = time.time()
    cfg = load_config(os.path.join(args.results, "config_used.yaml"))
    P = cfg["ratchet"]
    period = int(cfg["condA"]["T_values"][0])
    half_w, bulk_every = int(P["boundary_window"]), int(P["bulk_every"])

    seeds = load_seeds(args.results)
    seeds.sort(key=lambda d: str(d["run_id"]))
    print(f"loaded {len(seeds)} seeds, {len(seeds[0]['step'])} 記録点", flush=True)

    s4_ok, s4_df = check_s4(seeds)
    print(f"S4: {'PASS' if s4_ok else 'FAIL'}", flush=True)
    print(s4_df.to_string(index=False), flush=True)
    s3_ok, s3_df, cov, n_b = check_s3(seeds)
    print(f"S3: {'PASS' if s3_ok else 'FAIL'} (n={n_b})", flush=True)
    print(s3_df.to_string(index=False), flush=True)

    df, series, packs = per_seed(seeds, period, half_w, bulk_every, args.main)
    s2_ok, s2_df = check_s2(args.main, df)
    print(f"S2: {'PASS' if s2_ok else 'FAIL'}", flush=True)
    print(s2_df.to_string(index=False), flush=True)

    s_ok = bool(s2_ok and s3_ok and s4_ok)
    if not s_ok:
        raise SystemExit("[cell4-addendum] A0 FAIL — 判定に進まない (追補 §5)")

    print(f"符号ランダム化 null を {PERM_N} 回 ...", flush=True)
    nulls = null_distribution(packs)
    rng = np.random.default_rng(BOOT_SEED)
    V, extra = judge(df, nulls, rng, s_ok)

    os.makedirs(args.outdir, exist_ok=True)
    df.to_csv(os.path.join(args.outdir, "per_seed_metrics_addendum.csv"), index=False)
    V.to_csv(os.path.join(args.outdir, "verdict_addendum.csv"), index=False)
    fig_dir = make_figures(args.outdir, df, series, nulls, extra)

    meta = dict(date=time.strftime("%Y-%m-%d %H:%M:%S"), git_hash=git_hash(),
                spec="specs/spec_cell4_0821_addendum.md", source=args.results,
                main=args.main, grade="amended (事後追加)",
                elapsed_sec=round(time.time() - t0, 1),
                omp_num_threads=os.environ.get("OMP_NUM_THREADS", "(未設定)"),
                python=platform.python_version(), numpy=np.__version__,
                pandas=pd.__version__, bootstrap_B=BOOT_N, bootstrap_seed=BOOT_SEED,
                perm_n=PERM_N, kappa_hi=KAPPA_HI, n_seeds=len(seeds),
                s2_pass=bool(s2_ok), s3_pass=bool(s3_ok), s4_pass=bool(s4_ok),
                a1=extra["a1"], a2=extra["a2"], a1_point=extra["a1_point"],
                null_med=extra["null_med"], null_lo=extra["null_lo"],
                null_hi=extra["null_hi"], p_two=extra["p_two"], final=extra["final"])
    with open(os.path.join(args.outdir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1, default=str, ensure_ascii=False)
    sp = write_summary(args.outdir, V, df, s2_df, s3_df, cov, s4_df, extra, meta)

    print(V.to_string(index=False), flush=True)
    print(f"-> {args.outdir}/  ({sp}, {fig_dir}/)", flush=True)
    print(f"CELL4-ADDENDUM DONE — 最終判定: {extra['final']}", flush=True)
    return df, V


if __name__ == "__main__":
    main()
