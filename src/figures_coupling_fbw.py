"""coupling_fbw_0813 (spec_coupling_fbw_0813) の追加解析。

  python -m src.figures_coupling_fbw results/coupling_fbw_0813

基本図 (トレンド/境界整列/全ペア順序検定) は src.figures_coupling (batch 対応済み) が出す。
本モジュールは仕様 §2 の6解析と §3 の事前登録判定 (P1〜P4) を出力する:

  pulse_amplitudes.csv     — 境界別 Δdead_norm (= Δdead/(1−dead_pre)、dead_pre>0.9 除外)
  amplitude_ratio.csv      — セル×相別の振幅比 r = Δdead_norm(full)/Δdead_norm(b=1)、seed bootstrap CI
  crossing.csv             — 幅別 Δt50 = t50(dead) − t50(srank_alive_drop) と w* 推定
  fig_fbw_amp.png          — 正規化パルス振幅 (early/mid/late × batch)
  fig_fbw_pulse_shape.png  — 境界後 Δdead 累積曲線 (batch 重ね描き × 相)
  fig_fbw_crossing.png     — Δt50(w) の2系列 (b=1 / full) と符号反転
  fig_fbw_buffer.png       — m_alive = w(1−dead_frac) vs eval_loss 散布図 (P3 探索的)
  fig_fbw_init_dead.png    — step=0 の dead_frac の幅依存 (P4 サニティ)
  summary_fbw.md           — P1〜P4 の事前登録判定
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .figures_coupling import (load_all, cells, cell_mask, cell_title,
                               series_for_run, t_half, MIN_CHANGE)

PHASES = ["early", "mid", "late"]
PHASE_COLOR = {"early": "tab:blue", "mid": "tab:orange", "late": "tab:red"}
BATCH_COLOR = {"1": "tab:gray", "full": "tab:green"}
DEAD_PRE_MAX = 0.9      # 仕様: dead_frac > 0.9 のセルはパルス解析から除外
N_BOOT = 4000


def add_gk(df):
    """batch を除いたセルキー列 gk (例 "A_w100", "B_w100_c2") を付ける。
    condA の c は NaN なので、タプル/集合で扱うと NaN != NaN で行ごとに別セルに
    化ける。文字列キーに正規化してその事故を防ぐ。"""
    df = df.copy()
    df["gk"] = (df.exp.astype(str) + "_w" + df.width.astype(str)
                + df.c.map(lambda v: "" if pd.isna(v) else f"_c{v:g}"))
    return df


def add_offsets(lop, period, pre, post):
    """各行に最寄り境界 s0 とオフセットを付与し、窓内 (s0>0) の行だけ返す。"""
    lop = add_gk(lop)
    s0 = (lop.step / period).round() * period
    lop["s0"] = s0
    lop["offset"] = lop.step - s0
    total = lop.step.max()
    win = lop[(lop.offset >= -pre) & (lop.offset <= post) & (lop.s0 > 0)
              & (lop.s0 + post <= total)].copy()
    win["phase"] = pd.cut(win.s0, [0, total / 3, 2 * total / 3, total + 1],
                          labels=PHASES)
    return win


# ------------------------------------------------------- (1)(2) パルス振幅と振幅比

def pulse_amplitudes(win, stride):
    """境界別の Δdead と正規化振幅。pre 基準 = offset<=0 平均、post 終端 = 末尾3点平均。
    Δdead_norm = Δdead·w/n_alive = Δdead/(1−dead_pre)。dead_pre>0.9 の境界は除外。
    late_inc は full の形状補助判定用: 窓後半 (offset>post/2) の増分。"""
    rows = []
    post = win.offset.max()
    for (rid, s0), g in win.groupby(["run_id", "s0"]):
        g = g.sort_values("offset")
        pre_m = g[g.offset <= 0].dead_frac.mean()
        tail = g[g.offset >= post - 2 * stride].dead_frac.mean()
        mid_v = g[g.offset <= post / 2].dead_frac.iloc[-1]
        if not np.isfinite(pre_m) or not np.isfinite(tail):
            continue
        r0 = g.iloc[0]
        excluded = pre_m > DEAD_PRE_MAX
        rows.append(dict(exp=r0.exp, width=r0.width, batch=r0.batch, c=r0.c, gk=r0.gk,
                         run_id=rid, seed=r0.seed, s0=s0, phase=r0.phase,
                         dead_pre=pre_m, ddead=tail - pre_m,
                         # 飽和セル (1-pre_m ≈ 0) の正規化は発散するので除外側に倒す
                         ddead_norm=np.nan if excluded else (tail - pre_m) / (1 - pre_m),
                         late_inc=tail - mid_v,
                         excluded=excluded))
    return pd.DataFrame(rows)


def amplitude_ratio(amp, rng):
    """セル (exp,width,c) × 相ごとに r = mean_seed Δdead_norm(full) / mean_seed(b=1)。
    seed は matched (同 seed = 同初期化・同教師系列) なので paired bootstrap。"""
    ok = amp[~amp.excluded]
    rows = []
    for (exp, width, c, phase), g in ok.groupby(["exp", "width", "c", "phase"],
                                                dropna=False, observed=True):
        by = g.groupby(["batch", "seed"]).ddead_norm.mean().unstack("seed")
        if "1" not in by.index or "full" not in by.index:
            continue
        seeds = by.columns[by.loc["1"].notna() & by.loc["full"].notna()]
        a = by.loc["1", seeds].values      # b=1
        f = by.loc["full", seeds].values   # full
        row = dict(exp=exp, width=width, c=c, phase=phase, n_seed=len(seeds),
                   ddn_b1=np.nan, ddn_full=np.nan, r=np.nan,
                   r_lo=np.nan, r_hi=np.nan)
        if len(seeds) >= 3 and a.mean() > 1e-6:
            bs = rng.choice(len(seeds), (N_BOOT, len(seeds)), replace=True)
            den = a[bs].mean(axis=1)
            row.update(ddn_b1=a.mean(), ddn_full=f.mean(), r=f.mean() / a.mean())
            # 分母 (b=1 の振幅) が 0 を跨ぐと比の CI は発散して無意味になる。
            # ブートストラップ標本の 95% 超で分母が正のときだけ CI を報告する。
            pos = den > 1e-6
            if pos.mean() > 0.95:
                rb = f[bs][pos].mean(axis=1) / den[pos]
                row.update(r_lo=float(np.quantile(rb, 0.025)),
                           r_hi=float(np.quantile(rb, 0.975)))
        elif len(seeds):
            row.update(ddn_b1=a.mean(), ddn_full=f.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def fig_amp(amp, figdir):
    ok = amp[~amp.excluded]
    gks = sorted(ok.gk.unique())
    fig, axes = plt.subplots(1, len(gks), figsize=(2.6 * len(gks), 3.4),
                             squeeze=False, sharey=False)
    for k, gk in enumerate(gks):
        ax = axes[0][k]
        sub = ok[ok.gk == gk]
        for j, batch in enumerate(["1", "full"]):
            m = sub[sub.batch == batch].groupby("phase", observed=True).ddead_norm
            mean, sem = m.mean(), m.sem()
            xs = np.arange(len(PHASES)) + (j - 0.5) * 0.35
            ys = [mean.get(p, np.nan) for p in PHASES]
            es = [sem.get(p, np.nan) for p in PHASES]
            ax.bar(xs, ys, width=0.32, yerr=es, color=BATCH_COLOR[batch],
                   label=("b=1" if batch == "1" else "full"), capsize=2)
        ax.set_xticks(range(len(PHASES)))
        ax.set_xticklabels(PHASES, fontsize=8)
        ax.set_title(gk, fontsize=9)
        ax.grid(alpha=0.3, axis="y")
        if k == 0:
            ax.set_ylabel("Δdead_norm per boundary")
            ax.legend(fontsize=7)
    fig.suptitle("normalized boundary dead pulse (mean ± SEM over boundaries+seeds; "
                 f"dead_pre>{DEAD_PRE_MAX} excluded)")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "fig_fbw_amp.png"), dpi=150)
    plt.close(fig)


# ------------------------------------------------------------ (3) パルス形状

def fig_pulse_shape(win, figdir):
    """境界後の Δdead 累積曲線 (pre 基準からの偏差) を batch で重ね描き。相ごとに列。"""
    gks = sorted(win.gk.unique())
    fig, axes = plt.subplots(len(gks), len(PHASES),
                             figsize=(3.6 * len(PHASES), 2.6 * len(gks)),
                             squeeze=False, sharex=True)
    for i, gk in enumerate(gks):
        sub = win[win.gk == gk]
        for j, phase in enumerate(PHASES):
            ax = axes[i][j]
            for batch in ["1", "full"]:
                s = sub[(sub.phase == phase) & (sub.batch == batch)]
                if s.empty:
                    continue
                # 境界別に pre 基準を引いてから offset 平均 (dead_pre>0.9 の境界は除外)
                curves = []
                for (rid, s0), g in s.groupby(["run_id", "s0"]):
                    g = g.sort_values("offset")
                    pre_m = g[g.offset <= 0].dead_frac.mean()
                    if not np.isfinite(pre_m) or pre_m > DEAD_PRE_MAX:
                        continue
                    curves.append(pd.Series(g.dead_frac.values - pre_m,
                                            index=g.offset.values))
                if not curves:
                    continue
                m = pd.concat(curves, axis=1).mean(axis=1)
                ax.plot(m.index, m.values, lw=1.4, color=BATCH_COLOR[batch],
                        label=("b=1" if batch == "1" else "full"))
            ax.axvline(0, color="gray", lw=0.8, ls="--")
            ax.grid(alpha=0.3)
            if i == 0:
                ax.set_title(phase)
            if j == 0:
                ax.set_ylabel(f"{gk}\nΔdead")
            if i == len(gks) - 1:
                ax.set_xlabel("step - switch")
            if i == 0 and j == 0:
                ax.legend(fontsize=7)
    fig.suptitle("boundary dead pulse shape: cumulative Δdead vs offset (batch overlay)")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "fig_fbw_pulse_shape.png"), dpi=150)
    plt.close(fig)


# ------------------------------------------------- (4) 交差プロット Δt50(w) と w*

def compute_t50(d, srank_col):
    """run 別の t50 (dead / srank_alive_drop / eval_loss 用に系列を拡張)。"""
    lop, post = d["lop"], d["post"]
    rows = []
    for rid, lr in lop.groupby("run_id"):
        pr = post[post.run_id == rid] if len(post) else post
        ss = series_for_run(lr, pr, srank_col)
        lr_s = lr.sort_values("step")
        ss["eval_loss_rise"] = (lr_s.step.values, lr_s.eval_loss.values)
        r0 = lr.iloc[0]
        out = dict(exp=r0.exp, width=r0.width, batch=r0.batch, c=r0.c,
                   seed=r0.seed, run_id=rid)
        for met, mc in [("dead", MIN_CHANGE["dead"]),
                        ("srank_alive_drop", MIN_CHANGE["srank_alive_drop"]),
                        ("eval_loss_rise", 0.05)]:
            x, y = ss[met]
            out[f"t50_{met}"] = t_half(np.asarray(x, float), np.asarray(y, float), mc)
        rows.append(out)
    df = pd.DataFrame(rows)
    df["dt50"] = df.t50_dead - df.t50_srank_alive_drop
    return df


def wstar_from_curve(widths, dts):
    """seed 平均 Δt50(w) 曲線の符号反転点を log2(w) 線形内挿。反転なしは NaN。"""
    ok = np.isfinite(dts)
    w, y = np.asarray(widths, float)[ok], np.asarray(dts, float)[ok]
    if len(w) < 2:
        return np.nan
    o = np.argsort(w)
    w, y = w[o], y[o]
    for i in range(len(w) - 1):
        if y[i] > 0 >= y[i + 1] or y[i] >= 0 > y[i + 1]:
            lw = np.log2(w[i]) + (np.log2(w[i + 1]) - np.log2(w[i])) * y[i] / (y[i] - y[i + 1])
            return float(2 ** lw)
    return np.nan


def crossing(t50, rng):
    """condA の Δt50(w) を batch 別に集計し w* を seed bootstrap で推定。"""
    sub = t50[t50.exp == "A"]
    rows, west, mono = [], {}, []
    for batch, g in sub.groupby("batch"):
        piv = g.pivot_table(index="seed", columns="width", values="dt50")
        widths = sorted(piv.columns)
        mean = piv.mean(axis=0)
        for w in widths:
            col = piv[w].dropna()
            rows.append(dict(batch=batch, width=w, n=len(col),
                             dt50_mean=col.mean() if len(col) else np.nan,
                             dt50_sem=col.sem() if len(col) > 1 else np.nan))
        ws = wstar_from_curve(widths, [mean[w] for w in widths])
        # P2 の単調減少性: 隣接幅の増分 Δt50(w_{i+1}) − Δt50(w_i) を paired bootstrap
        # (seed が matched なので対応のある差)。正の増分は単調減少に反する。
        for i in range(len(widths) - 1):
            w0, w1 = widths[i], widths[i + 1]
            pair = piv[[w0, w1]].dropna()
            if len(pair) < 3:
                continue
            dd = (pair[w1] - pair[w0]).values
            pick = rng.choice(len(dd), (N_BOOT, len(dd)), replace=True)
            bm = dd[pick].mean(axis=1)
            mono.append(dict(batch=batch, w_from=w0, w_to=w1, n=len(dd),
                             delta=float(dd.mean()),
                             lo=float(np.quantile(bm, 0.025)),
                             hi=float(np.quantile(bm, 0.975)),
                             violates=bool(np.quantile(bm, 0.025) > 0)))
        # paired bootstrap: seed 行を復元抽出して曲線→w* を再計算
        seeds = piv.index.values
        wbs = []
        for _ in range(N_BOOT):
            pick = rng.choice(len(seeds), len(seeds), replace=True)
            bm = piv.iloc[pick].mean(axis=0)
            wbs.append(wstar_from_curve(widths, [bm[w] for w in widths]))
        wbs = np.array(wbs)
        fin = np.isfinite(wbs)
        west[batch] = dict(wstar=ws,
                           wstar_lo=float(np.quantile(wbs[fin], 0.025)) if fin.sum() > 40 else np.nan,
                           wstar_hi=float(np.quantile(wbs[fin], 0.975)) if fin.sum() > 40 else np.nan,
                           frac_crossing=float(fin.mean()))
    return pd.DataFrame(rows), west, pd.DataFrame(mono)


def fig_crossing(cross_df, west, figdir):
    """b=1 と full は Δt50 のスケールが1桁以上違う (full は dead が大幅遅延) ため、
    共有軸だと b=1 の符号反転が潰れる。パネルを分けて各系列を自前のスケールで描く。"""
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    for ax, batch in zip(axes, ["1", "full"]):
        g = cross_df[cross_df.batch == batch].sort_values("width")
        lbl = "b=1" if batch == "1" else "full"
        ax.errorbar(g.width, g.dt50_mean, yerr=g.dt50_sem, marker="o", ms=4,
                    lw=1.4, capsize=3, color=BATCH_COLOR[batch], label=lbl)
        ws = west.get(batch, {}).get("wstar", np.nan)
        if np.isfinite(ws):
            ax.axvline(ws, color="tab:red", lw=1, ls=":", label=f"w* = {ws:.0f}")
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("width")
        ax.set_title(f"{lbl}: " + ("sign reversal present" if np.isfinite(ws)
                                   else "no sign reversal"), fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Δt50 = t50(dead) − t50(srank_alive_drop)")
    fig.suptitle("coupling direction crossover (condA): Δt50 > 0 → B→A, < 0 → A→B")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "fig_fbw_crossing.png"), dpi=150)
    plt.close(fig)


# ------------------------------------------------------- (5) バッファ散布図 (P3)

def fig_buffer(d, t50, lop_every, figdir):
    lop = d["lop"]
    sub = lop[(lop.exp == "A") & (lop.step % lop_every == 0)].copy()
    sub["m_alive"] = sub.width * (1 - sub.dead_frac)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    cmap = plt.get_cmap("viridis")
    widths = sorted(sub.width.unique())
    wcol = {w: cmap(i / max(1, len(widths) - 1)) for i, w in enumerate(widths)}
    for j, batch in enumerate(["1", "full"]):
        ax = axes[j]
        for w in widths:
            s = sub[(sub.batch == batch) & (sub.width == w)]
            ax.scatter(s.m_alive, s.eval_loss, s=3, alpha=0.15, color=wcol[w],
                       label=f"w={w}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("m_alive = w·(1−dead_frac)")
        ax.set_title("b=1" if batch == "1" else "full")
        ax.grid(alpha=0.3)
        if j == 0:
            ax.set_ylabel("eval_loss")
            leg = ax.legend(fontsize=7, markerscale=3)
            for lh in leg.legend_handles:
                lh.set_alpha(1)
    fig.suptitle("P3 (exploratory): alive-buffer vs eval_loss, all condA widths pooled")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "fig_fbw_buffer.png"), dpi=150)
    plt.close(fig)

    # 探索的: eval_loss 離陸時 (t50) の m_alive を run 別に補間
    rows = []
    for r in t50[t50.exp == "A"].itertuples():
        t = r.t50_eval_loss_rise
        if not np.isfinite(t):
            continue
        s = lop[lop.run_id == r.run_id].sort_values("step")
        m_alive = r.width * (1 - np.interp(t, s.step.values, s.dead_frac.values))
        rows.append(dict(width=r.width, batch=r.batch, seed=r.seed,
                         t_takeoff=t, m_alive_takeoff=m_alive))
    return pd.DataFrame(rows)


# ------------------------------------------------------ (6) 初期 dead (P4)

def fig_init_dead(d, figdir):
    """step=0 の dead_frac。b=1 と full は同一初期化 (make_gens は batch を含まない) なので
    seed で重複排除する。幅ごとの推定精度は unit 数 w·n_seed の二項ばらつきで決まり、
    w=5 では 0.2 刻みに量子化されるため、幅依存の判定は SE 込みで行う。"""
    lop = d["lop"]
    sub = lop[(lop.exp == "A") & (lop.step == 0)].drop_duplicates(["width", "seed"])
    g = sub.groupby("width").dead_frac
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for w, s in sub.groupby("width"):
        ax.scatter([w] * len(s), s.dead_frac, s=14, color="tab:blue", alpha=0.6)
    ax.plot(sorted(g.mean().index), g.mean().sort_index().values, "-o",
            color="tab:red", ms=4, label="mean")
    ax.axhline(0.32, color="gray", lw=0.8, ls="--", label="anchor 0.32")
    ax.set_xscale("log")
    ax.set_xlabel("width")
    ax.set_ylabel("dead_frac @ step=0")
    ax.set_title("P4 sanity: initial dead fraction vs width (condA)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "fig_fbw_init_dead.png"), dpi=150)
    plt.close(fig)
    st = g.agg(["mean", "std", "count"]).reset_index()
    # 幅 w・seed 数 n の run 群での per-unit 二項 SE (seed 内相関は無視した下限見積り)
    st["n_units"] = st.width * st["count"]
    p = st["mean"]
    st["se_binom"] = np.sqrt(p * (1 - p) / st.n_units)
    pooled = float((p * st.n_units).sum() / st.n_units.sum())
    st["z_vs_pooled"] = (p - pooled) / st.se_binom
    st.attrs["pooled"] = pooled
    return st


# ---------------------------------------------------------------- 判定サマリ

def p1_verdict(r):
    if not np.isfinite(r):
        return "判定不能 (r 計算不可: b=1 パルス不発または seed 不足)"
    if r > 0.5:
        return f"r={r:.3f} > 0.5 → **drift 主導**。Path A を「境界 drift 過渡 + diffusion 増幅」と再定式化"
    if r < 0.2:
        return f"r={r:.3f} < 0.2 → **diffusion 主導**。③→④ リンクを diffusion 経路として支持"
    return f"r={r:.3f} ∈ [0.2, 0.5] → **混合**。両成分を定量報告"


def final_state(d):
    """最終ステップの主要指標を (セル × batch) で集計。P1 の独立証拠用。"""
    lop = add_gk(d["lop"])
    last = lop[lop.step == lop.step.max()]
    cols = [c for c in ["dead_frac", "eval_loss", "stable_rank_W_alive", "trC_W"]
            if c in last.columns]
    return last.groupby(["gk", "batch"])[cols].mean()


def t50_breakdown(t50):
    """Δt50 の内訳 (dead / srank_alive それぞれの t50)。P2′ の解釈に必要。"""
    return t50[t50.exp == "A"].groupby(["width", "batch"])[
        ["t50_dead", "t50_srank_alive_drop"]].mean()


def write_summary(resdir, ratio, cross_df, west, mono, take, init_stats, amp,
                  fin_state, brk):
    lines = ["# coupling_fbw_0813 追加解析 (spec_coupling_fbw_0813 §2–3)\n"]

    lines.append("## P1: 境界 dead パルスの帰属 (主判定セル A_w100 early)\n")
    main = ratio[(ratio.exp == "A") & (ratio.width == 100) & (ratio.phase == "early")]
    if len(main):
        m = main.iloc[0]
        lines.append(f"- Δdead_norm: b=1 = {m.ddn_b1:.4g}, full = {m.ddn_full:.4g}")
        lines.append(f"- 振幅比 r = {m.r:.3f} (bootstrap 95%CI [{m.r_lo:.3f}, {m.r_hi:.3f}], "
                     f"n_seed={m.n_seed})")
        lines.append(f"- **判定**: {p1_verdict(m.r)}")
    else:
        lines.append("- 主判定セルのデータ無し")
    ok = amp[~amp.excluded]
    if len(ok):
        li = ok.groupby(["gk", "batch"]).late_inc.mean().unstack("batch")
        lines.append("\n- 形状補助判定: 窓後半 (offset > post/2) の Δdead 増分。"
                     "full は境界直後の過渡に集中し後半はほぼ 0、b=1 は窓全体でトリクル継続、"
                     "が事前登録の予測。full の後半増分が b=1 と同程度なら実装バグを疑う。\n")
        lines.append(li.round(5).to_string())
    excl = amp.groupby(["gk", "batch", "phase"], observed=True).excluded.mean() \
              .unstack(["batch", "phase"])
    lines.append(f"\n- パルス解析からの除外率 (dead_pre > {DEAD_PRE_MAX})。"
                 "主判定は early セルなので、そこでの除外率が低いことが前提:\n")
    lines.append(excl.round(3).to_string())

    lines.append("\n\n### P1 の独立証拠: 診断 trC と訓練ノイズの分離 (@最終ステップ)\n")
    lines.append("本実験の肝は「診断の trC_W (eval バッチの per-sample 勾配分散) は測れるが、"
                 "full では訓練に注入されるノイズが厳密ゼロ」という状況。"
                 "full の trC_W が b=1 以上なのに dead が遅い/浅いなら、"
                 "「trC が大きいこと」ではなく「実際に注入されるノイズ」が dead の駆動因である。\n")
    lines.append(fin_state.round(4).to_string())
    lines.append("\n全セル×相の振幅比:\n")
    lines.append(ratio.to_string(index=False))

    lines.append("\n\n## P2 / P2′: 交差 Δt50(w) = t50(dead) − t50(srank_alive_drop)\n")
    lines.append(cross_df.to_string(index=False))
    lines.append("\n\n内訳 (Δt50 の符号がどちらの指標の動きで決まっているか):\n")
    lines.append(brk.round(0).to_string())
    for batch in ["1", "full"]:
        wd = west.get(batch)
        if not wd:
            continue
        lbl = "b=1 (P2)" if batch == "1" else "full (P2′)"
        if np.isfinite(wd["wstar"]):
            lines.append(f"\n- {lbl}: w* = {wd['wstar']:.1f} "
                         f"(95%CI [{wd['wstar_lo']:.1f}, {wd['wstar_hi']:.1f}], "
                         f"bootstrap 反転出現率 {wd['frac_crossing']:.2f})")
        else:
            lines.append(f"\n- {lbl}: 符号反転なし (bootstrap 反転出現率 "
                         f"{wd['frac_crossing']:.2f}) — 予測 P2 の反証条件に該当するか要確認")
    if len(mono):
        lines.append("\n\n単調減少性の検定 (隣接幅の増分 Δt50(w_to) − Δt50(w_from)、"
                     "paired seed bootstrap 95%CI。violates=True は CI 下端が正 "
                     "= 単調減少に反する有意な増加):\n")
        lines.append(mono.to_string(index=False))
        v = mono[(mono.batch == "1") & mono.violates]
        lines.append(f"\n- b=1 の有意な非単調区間: "
                     + (", ".join(f"w{r.w_from}→w{r.w_to}" for r in v.itertuples())
                        if len(v) else "なし (P2 の単調性は反証されず)"))

    lines.append("\n## P3 (探索的・傾向報告のみ): eval_loss 離陸時の m_alive\n")
    if len(take):
        tt = take.groupby(["width", "batch"]).m_alive_takeoff.agg(["mean", "std", "count"])
        lines.append(tt.to_string())
    else:
        lines.append("- 離陸検出なし")

    lines.append("\n## P4 (サニティ): step=0 の dead_frac 幅依存 (per-unit 確率なら幅非依存のはず)\n")
    lines.append("b=1 と full は同一初期化なので seed で重複排除済み。"
                 "se_binom は unit 数 w·n_seed の二項 SE、z_vs_pooled は全幅プール値からの乖離。\n")
    lines.append(init_stats.to_string(index=False))
    pooled = init_stats.attrs["pooled"]
    zmax = init_stats.z_vs_pooled.abs().max()
    lines.append(f"\n- 全幅プール per-unit dead 率 = {pooled:.3f} (アンカー 0.32)")
    lines.append(f"- プール値からの最大 |z| = {zmax:.2f}"
                 + (" → 二項ばらつきを超える幅依存あり。dead 判定仕様 "
                    "(dead_tau=0.95, eval 2000) を再点検" if zmax > 2.5
                    else " → 幅依存は二項ばらつきの範囲内、P4 は満たされる"))

    with open(os.path.join(resdir, "summary_fbw.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    args = ap.parse_args()
    figdir = os.path.join(args.results, "figures")
    os.makedirs(figdir, exist_ok=True)
    rng = np.random.default_rng(0)

    import yaml
    with open(os.path.join(args.results, "config_used.yaml")) as fh:
        cfg = yaml.safe_load(fh)
    cp = cfg.get("coupling", {})
    pre, post = cp.get("pre_window", 0), cp.get("post_window", 0)
    stride = cp.get("fine_stride", 100)

    d = load_all(args.results)
    srank_col = "stable_rank_W" if "stable_rank_W" in d["lop"].columns else "eff_rank_W"
    period = int(d["runs"].period.iloc[0])

    win = add_offsets(d["lop"], period, pre, post)
    amp = pulse_amplitudes(win, stride)
    amp.to_csv(os.path.join(args.results, "pulse_amplitudes.csv"), index=False)
    ratio = amplitude_ratio(amp, rng)
    ratio.to_csv(os.path.join(args.results, "amplitude_ratio.csv"), index=False)
    fig_amp(amp, figdir)
    fig_pulse_shape(win, figdir)

    t50 = compute_t50(d, srank_col)
    t50.to_csv(os.path.join(args.results, "t50_runs.csv"), index=False)
    cross_df, west, mono = crossing(t50, rng)
    cross_df.to_csv(os.path.join(args.results, "crossing.csv"), index=False)
    fig_crossing(cross_df, west, figdir)

    take = fig_buffer(d, t50, cfg["common"]["lop_every"], figdir)
    init_stats = fig_init_dead(d, figdir)

    write_summary(args.results, ratio, cross_df, west, mono, take, init_stats, amp,
                  final_state(d), t50_breakdown(t50))
    print(ratio.to_string(index=False))
    print(cross_df.to_string(index=False))
    print(f"figures -> {figdir}")


if __name__ == "__main__":
    main()
