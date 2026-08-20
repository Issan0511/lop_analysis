"""teachw_0820 の判定と図 [spec_teachw_0820 §6, §8]。

  OMP_NUM_THREADS=1 .venv/bin/python -m src.figures_teachw [results/teachw_0820]

出力: verdict.csv (P0–P2 + S1–S4) / summary.md / per_seed_metrics.csv / runs.csv /
figures/。事前登録の判定基準は §6 が唯一の正で、本モジュールはそれを実装するだけ。

**主判定量**: alive_final = #{i: p̂_i ≥ 0.05}。p̂ は t=1M における入力分布の
**全サポート (2^(m−f)=32 パターン) 上の厳密ゲート率**で、有限 eval バッチの推定では
ない (`src/teachw.exact_record`)。dead_frac_final = 1 − alive_final/100 は
ratchet_log_0819 の同名量と同一定義なので seed 別に厳密一致するはず (§7 S3)。

**仕様が決めていない集計の選択 (逸脱として summary に明記)**:
- P0 のレベル内 seed 集計は **中央値**。平均も併記し、判定が割れたら逸脱節に書く。
- t50 は初期値補正版 = dead が d0 + (d_final − d0)/2 に初到達する step。condA は
  t=0 で既に dead ≈ 0.25 あるので、生の「final の半分」だと t=0 で満たされてしまう。
"""
import json
import os
import sys

import numpy as np
import pandas as pd

from .common import ROOT

ANCHOR = "results/ratchet_log_0819/per_seed_metrics.csv"


# ---------------------------------------------------------------- 読み込み

def load_arms(resdir):
    """results/teachw_0820/H*/ を H_T 昇順で読む。"""
    arms = []
    for name in sorted(os.listdir(resdir)):
        adir = os.path.join(resdir, name)
        if not (name.startswith("H") and os.path.isdir(adir)
                and os.path.exists(os.path.join(adir, "meta.json"))):
            continue
        meta = json.load(open(os.path.join(adir, "meta.json")))
        logdir = os.path.join(adir, "logs")
        paths = sorted((p for p in os.listdir(logdir) if p.endswith(".npz")),
                       key=lambda p: int(p[4:-4]))
        seeds = []
        for p in paths:
            d = np.load(os.path.join(logdir, p), allow_pickle=False)
            seeds.append({k: d[k] for k in d.files})
        arms.append(dict(H_T=int(meta["target_hidden"]), dir=adir, meta=meta, seeds=seeds))
    arms.sort(key=lambda a: a["H_T"])
    if not arms:
        raise SystemExit(f"アームが見つからない: {resdir}/H*/")
    return arms


def t50_dead(step, dead):
    """dead が d0 + (d_final − d0)/2 に初到達する step (初期値補正版)。

    condA は t=0 で既に dead ≈ 0.25 あるので、生の「final の半分」定義だと多くの
    seed で t=0 が答えになり時間情報が消える。単調増加しない系列でも「初到達」なので
    定義は well-defined (到達しなければ NaN)。"""
    d0, df = float(dead[0]), float(dead[-1])
    if not np.isfinite(d0) or not np.isfinite(df) or df <= d0:
        return np.nan
    tgt = d0 + 0.5 * (df - d0)
    idx = np.flatnonzero(np.asarray(dead) >= tgt)
    return float(step[idx[0]]) if idx.size else np.nan


def per_seed_metrics(arms, tail_frac=0.1):
    """(H_T, seed) 1 行の判定素材。tail_frac は eval_loss の plateau 窓 (末尾割合)。"""
    rows = []
    for a in arms:
        for d in a["seeds"]:
            step = d["step"].astype(np.int64)
            alive = d["alive"].astype(float)
            n_tail = max(1, int(len(step) * tail_frac))
            rows.append(dict(
                H_T=a["H_T"], log2H=float(np.log2(a["H_T"])),
                seed=int(d["seed"]), run_id=str(d["run_id"]),
                out_scale=float(d["out_scale"]),
                alive_final=float(alive[-1]),
                dead_frac_final=float(1.0 - alive[-1] / float(d["width"])),
                width=int(d["width"]),
                p_hat_median_alive_final=float(d["p_hat_median_alive"][-1]),
                eval_loss_exact_final=float(d["eval_loss_exact"][-1]),
                eval_loss_exact_plateau=float(np.median(d["eval_loss_exact"][-n_tail:])),
                eval_loss_exact_t0=float(d["eval_loss_exact"][0]),
                var_y_t0=float(d["var_y"][0]),
                var_y_median=float(np.median(d["var_y"])),
                alive_t0=float(alive[0]),
                t50_dead=t50_dead(step, float(d["width"]) - alive),
                n_rec=int(len(step))))
    df = pd.DataFrame(rows)
    # §8: run_id ソートで seed ペアリング (run_id は全アーム共通なので順序も共通)
    return df.sort_values(["H_T", "run_id"]).reset_index(drop=True)


# ---------------------------------------------------------------- 統計

def boot_ci(rng, vec, B):
    v = np.asarray(vec, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan, np.nan, np.nan
    bs = v[rng.integers(0, v.size, (B, v.size))].mean(axis=1)
    return float(v.mean()), float(np.quantile(bs, .025)), float(np.quantile(bs, .975))


def boot_ci_median(rng, vec, B):
    v = np.asarray(vec, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan, np.nan, np.nan
    bs = np.median(v[rng.integers(0, v.size, (B, v.size))], axis=1)
    return float(np.median(v)), float(np.quantile(bs, .025)), float(np.quantile(bs, .975))


def _rank(x):
    """平均順位 (同順位は平均)。scipy 非依存 [memory: venv に scipy なし]。"""
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), float)
    r[order] = np.arange(1, len(x) + 1, dtype=float)
    for v in np.unique(x):
        m = x == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def spearman(x, y):
    """Spearman ρ (順位 Pearson)。分散 0 なら NaN。"""
    rx, ry = _rank(x), _rank(y)
    sx, sy = rx.std(), ry.std()
    if sx == 0 or sy == 0:
        return np.nan
    return float(((rx - rx.mean()) * (ry - ry.mean())).mean() / (sx * sy))


def ols_slope(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    vx = ((x - x.mean()) ** 2).sum()
    if vx == 0:
        return np.nan
    return float(((x - x.mean()) * (y - y.mean())).sum() / vx)


def per_seed_slopes(df, levels):
    """seed ごとの (i) alive~log2(H_T) の OLS 傾き と (ii) Spearman ρ。

    レベル集合が全 seed で同一なので、seed 内傾きの平均 = seed 固定効果つき
    プール回帰の傾きと厳密に一致する (summary に併記)。"""
    sub = df[df.H_T.isin(levels)]
    out = []
    for sd, g in sub.groupby("seed"):
        g = g.sort_values("log2H")
        out.append(dict(seed=int(sd), slope=ols_slope(g.log2H, g.alive_final),
                        rho=spearman(g.log2H.to_numpy(), g.alive_final.to_numpy()),
                        n_level=len(g)))
    return pd.DataFrame(out).sort_values("seed").reset_index(drop=True)


# ---------------------------------------------------------------- サニティ

def check_s2(arms):
    """S2: flip_state 軌跡 hash が seed ごとに全アームで一致 [§7]。"""
    seeds = sorted(int(s) for s in arms[0]["meta"]["flip_hash"])
    per = {s: sorted({a["meta"]["flip_hash"][str(s)] for a in arms}) for s in seeds}
    return dict(pass_=all(len(v) == 1 for v in per.values()), n_seed=len(seeds),
                n_arm=len(arms), n_distinct={s: len(v) for s, v in per.items()},
                hash_per_seed={s: v[0] for s, v in per.items()})


def check_s3(df, ref_H=100):
    """S3: H_T=ref_H の dead_frac_final が ratchet_log_0819 と seed 別に厳密一致 [§7]。"""
    p = os.path.join(ROOT, ANCHOR)
    if not os.path.exists(p):
        return dict(pass_=False, note=f"アンカー欠落: {ANCHOR}")
    ref = pd.read_csv(p).set_index("seed")["dead_frac_final"]
    got = df[df.H_T == ref_H].set_index("seed")["dead_frac_final"]
    common = sorted(set(ref.index) & set(got.index))
    diff = {int(s): (float(ref[s]), float(got[s])) for s in common
            if float(ref[s]) != float(got[s])}
    return dict(pass_=bool(common and not diff), n_seed=len(common),
                n_mismatch=len(diff), mismatch=diff, anchor=ANCHOR,
                max_abs_diff=float(max((abs(a - b) for a, b in
                                        ((float(ref[s]), float(got[s])) for s in common)),
                                       default=np.nan)))


def check_s4(df, band, ref_H=100):
    """S4: Var[y_scaled] の帯を本走ログで確認 [§7]。t=0 の記録点を使う。

    字義 (中央値) と意図 (平均) を並記する。LTU は β=0.7 のしきい値が高いため
    Var[y_raw] ∝ Binom(H_T, ≈0.151) の実現値で、期待値は O(H_T) だが低 H_T では
    ゼロ過剰になる (周期内で定数の教師)。中央値はゼロ質量を拾うので、乗法スケーリング
    則をどう選んでも直らない。詳細は phase0_summary.md 0-2。"""
    med = df.groupby("H_T")["var_y_t0"].median()
    avg = df.groupby("H_T")["var_y_t0"].mean()
    ref, ref_m = float(med.get(ref_H, np.nan)), float(avg.get(ref_H, np.nan))
    rows = []
    for h, v in med.items():
        ratio = float(v / ref) if ref else np.nan
        ratio_m = float(avg[h] / ref_m) if ref_m else np.nan
        rows.append(dict(H_T=int(h), var_y_t0_median=float(v), ratio_vs_ref=ratio,
                         in_band=bool(band[0] <= ratio <= band[1]),
                         var_y_t0_mean=float(avg[h]), ratio_mean=ratio_m,
                         in_band_mean=bool(band[0] <= ratio_m <= band[1]),
                         n_zero_var=int((df[(df.H_T == h)].var_y_t0 == 0).sum())))
    return dict(pass_=all(r["in_band"] for r in rows),
                pass_intent=all(r["in_band_mean"] for r in rows), band=list(band),
                ref_var=ref, ref_var_mean=ref_m, rows=rows)


# ---------------------------------------------------------------- 判定 (§6)

def judge(df, P, arms):
    B = int(P["bootstrap_B"])
    rng = np.random.default_rng(int(P["bootstrap_seed"]))
    levels = sorted(int(h) for h in df.H_T.unique())
    V = []
    def add(**k):
        row = dict(id="", statistic="", point=np.nan, ci_lo=np.nan, ci_hi=np.nan,
                   threshold="", result="", note="")
        row.update(k)
        V.append(row)

    # --- P0: 前提ゲート (レベルごと)
    gate, g_rows = {}, []
    for h in levels:
        v = df[df.H_T == h]["dead_frac_final"].to_numpy()
        med, mean = float(np.median(v)), float(v.mean())
        ok = med >= float(P["p0_dead_min"])
        gate[h] = ok
        g_rows.append(dict(H_T=int(h), dead_median=med, dead_mean=mean,
                           alive_median=float(np.median(df[df.H_T == h].alive_final)),
                           n_seed_ge=int((v >= float(P["p0_dead_min"])).sum()),
                           n_seed=int(v.size), valid=ok,
                           agree_mean=bool(ok == (mean >= float(P["p0_dead_min"])))))
        add(id=f"P0_H{h}", statistic=f"dead_frac_final (H_T={h}, seed 中央値)",
            point=med, threshold=f">= {P['p0_dead_min']}",
            result="PASS" if ok else "FAIL",
            note=f"平均 {mean:.3f}; 基準以上の seed {int((v >= float(P['p0_dead_min'])).sum())}"
                 f"/{int(v.size)}; FAIL なら P1 から除外 (void)")
    valid = [h for h in levels if gate[h]]
    enough = len(valid) >= int(P["p0_min_levels"])
    add(id="P0", statistic="有効レベル数", point=float(len(valid)),
        threshold=f">= {P['p0_min_levels']}", result="PASS" if enough else "保留",
        note=f"有効レベル {valid} / 全レベル {levels}"
             + ("" if enough else "; 有効レベル不足のため P1 は判定保留 (posreset G0 規約)"))

    # --- P1: 主判定
    ps = per_seed_slopes(df, valid) if len(valid) >= 2 else pd.DataFrame()
    if len(valid) >= 2:
        sl, lo, hi = boot_ci(rng, ps.slope.to_numpy(), B)
        rho_med, rlo, rhi = boot_ci_median(rng, ps.rho.to_numpy(), B)
        ok_i = bool(np.isfinite(lo) and sl > 0 and lo > 0)
        ok_ii = bool(np.isfinite(rho_med) and rho_med >= float(P["rho_median_min"]))
        res = ("PASS" if (ok_i and ok_ii) else "FAIL") if enough else "保留"
        pooled = ols_slope(df[df.H_T.isin(valid)].log2H, df[df.H_T.isin(valid)].alive_final)
        add(id="P1_i", statistic="alive_final ~ log2(H_T) の seed 内傾き (平均)",
            point=sl, ci_lo=lo, ci_hi=hi, threshold="傾き > 0 かつ CI ゼロ非含有",
            result=("PASS" if ok_i else "FAIL") if enough else "保留",
            note=f"有効レベル {valid} (n={len(valid)}), seed {len(ps)} 本, B={B}; "
                 f"seed 無視のプール OLS 傾き {pooled:.4f}")
        add(id="P1_ii", statistic="per-seed Spearman ρ の中央値", point=rho_med,
            ci_lo=rlo, ci_hi=rhi, threshold=f">= {P['rho_median_min']}",
            result=("PASS" if ok_ii else "FAIL") if enough else "保留",
            note=f"seed 別 ρ: {[round(float(x), 3) for x in ps.rho]}")
        add(id="P1", statistic="主判定 (i) かつ (ii)", point=np.nan,
            threshold="(i) 傾き>0 かつ CI ゼロ非含有 / (ii) ρ 中央値 >= "
                      f"{P['rho_median_min']}",
            result=res,
            note=f"(i)={'PASS' if ok_i else 'FAIL'} / (ii)={'PASS' if ok_ii else 'FAIL'}"
                 + ("" if enough else "; 有効レベル < "
                    f"{P['p0_min_levels']} のため保留 (統計量は参考値)"))
    else:
        add(id="P1", statistic="主判定", threshold="—", result="保留",
            note="有効レベルが 2 未満で回帰不能")

    # 参考: 全レベルで同じ統計量 (void レベルを混ぜた場合)
    ps_all = per_seed_slopes(df, levels)
    sl_a, lo_a, hi_a = boot_ci(rng, ps_all.slope.to_numpy(), B)
    rho_a, _, _ = boot_ci_median(rng, ps_all.rho.to_numpy(), B)
    add(id="P1_all", statistic="[参考] 全レベルでの傾き (void 混在)", point=sl_a,
        ci_lo=lo_a, ci_hi=hi_a, threshold="判定に使わない", result="参考",
        note=f"ρ 中央値 {rho_a:.3f}; レベル {levels}")

    # --- P2: 効果量 (報告のみ)
    lo_h, hi_h = min(levels), max(levels)
    a = df[df.H_T == hi_h].sort_values("run_id").alive_final.to_numpy()
    b = df[df.H_T == lo_h].sort_values("run_id").alive_final.to_numpy()
    n = min(len(a), len(b))
    idx = rng.integers(0, n, (B, n))
    dif = a[:n] - b[:n]
    add(id="P2", statistic=f"alive(H={hi_h}) − alive(H={lo_h}) の paired 差",
        point=float(dif.mean()), ci_lo=float(np.quantile(dif[idx].mean(axis=1), .025)),
        ci_hi=float(np.quantile(dif[idx].mean(axis=1), .975)),
        threshold="判定なし (報告のみ)", result="報告",
        note=f"seed 別差 {dif.astype(int).tolist()}; "
             f"H={lo_h} は{'有効' if gate.get(lo_h) else '**void (P0 FAIL)**'}")

    # --- サニティ
    s2 = check_s2(arms)
    add(id="S2", statistic="flip_state 軌跡 hash の全アーム一致 (seed 別)",
        point=float(s2["n_seed"]), threshold="seed ごとに異なりハッシュ数 = 1",
        result="PASS" if s2["pass_"] else "FAIL",
        note=f"seed {s2['n_seed']} 本 × アーム {s2['n_arm']} 本; "
             f"異なり数 {sorted(set(s2['n_distinct'].values()))}")
    s3 = check_s3(df)
    add(id="S3", statistic="H_T=100 の dead_frac_final が ratchet_log_0819 と一致",
        point=float(s3.get("n_seed", 0)), threshold="seed 別に厳密一致",
        result="PASS" if s3["pass_"] else "FAIL",
        note=f"不一致 {s3.get('n_mismatch')} / {s3.get('n_seed')} seed; "
             f"max|diff|={s3.get('max_abs_diff')}")
    s4 = check_s4(df, P["var_band"])
    add(id="S4", statistic="Var[y_scaled] (t=0) の H_T=100 比 (字義: 中央値)", point=np.nan,
        threshold=f"中央値比が {list(P['var_band'])} 内",
        result="PASS" if s4["pass_"] else "FAIL",
        note="中央値比 " + "; ".join(f"H{r['H_T']}:{r['ratio_vs_ref']:.3f}"
                                    for r in s4["rows"]))
    add(id="S4_intent", statistic="同上 (意図: 平均。低 H_T のゼロ過剰を回避)",
        point=np.nan, threshold=f"平均比が {list(P['var_band'])} 内",
        result="PASS" if s4["pass_intent"] else "FAIL",
        note="平均比 " + "; ".join(f"H{r['H_T']}:{r['ratio_mean']:.3f}"
                                  for r in s4["rows"])
             + "。字義の中央値は Var[y_raw] ∝ Binom(H_T, ≈0.151) のゼロ過剰を拾うので"
               "乗法スケーリングでは直らない (phase0_summary.md 0-2)")
    omp = {a["meta"].get("omp_num_threads") for a in arms}
    add(id="S1", statistic="OMP_NUM_THREADS", point=np.nan, threshold='= "1"',
        result="PASS" if omp == {"1"} else "FAIL", note=f"アーム横断で {sorted(omp)}")

    return pd.DataFrame(V), pd.DataFrame(g_rows), ps, dict(s2=s2, s3=s3, s4=s4,
                                                           valid=valid, gate=gate,
                                                           enough=enough)


# ---------------------------------------------------------------- 図

def make_figures(arms, df, resdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    have = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ("Noto Sans CJK JP", "Noto Sans CJK TC", "Noto Sans CJK KR",
                 "IPAGothic", "TakaoGothic"):
        if cand in have:
            plt.rcParams["font.family"] = cand
            break
    plt.rcParams["axes.unicode_minus"] = False
    fig_dir = os.path.join(resdir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    levels = sorted(df.H_T.unique())
    cmap = plt.get_cmap("viridis")
    col = {h: cmap(i / max(1, len(levels) - 1)) for i, h in enumerate(levels)}

    # (1) alive vs log2 H_T: seed 線 + 中央値 [§8]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for sd, g in df.groupby("seed"):
        g = g.sort_values("log2H")
        ax[0].plot(g.log2H, g.alive_final, "-o", lw=0.8, ms=3, alpha=0.45,
                   color="0.45", zorder=1)
    med = df.groupby("log2H")["alive_final"].median()
    ax[0].plot(med.index, med.values, "-o", lw=2.4, ms=7, color="crimson",
               label="seed 中央値", zorder=3)
    ax[0].set_xticks([float(np.log2(h)) for h in levels])
    ax[0].set_xticklabels([str(h) for h in levels])
    ax[0].set_xlabel("教師 LTU 幅 $H_T$ (log2 軸)")
    ax[0].set_ylabel(r"alive_final = #{i: $\hat{p}_i \geq 0.05$}")
    ax[0].set_title("生存者数 vs 教師複雑度 (t=1M, 厳密 $\\hat{p}$)")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    for h in levels:
        v = df[df.H_T == h].dead_frac_final
        ax[1].scatter([np.log2(h)] * len(v), v, s=18, color=col[h], alpha=0.8)
    dm = df.groupby("log2H")["dead_frac_final"].median()
    ax[1].plot(dm.index, dm.values, "-o", lw=2.0, color="crimson")
    ax[1].axhline(0.5, ls="--", c="k", lw=1, label="P0 ゲート 0.5")
    ax[1].set_xticks([float(np.log2(h)) for h in levels])
    ax[1].set_xticklabels([str(h) for h in levels])
    ax[1].set_xlabel("教師 LTU 幅 $H_T$ (log2 軸)")
    ax[1].set_ylabel("dead_frac_final")
    ax[1].set_title("P0 前提ゲート")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig_tw_alive_vs_logH.png"), dpi=130)
    plt.close(fig)

    # (2) dead_frac 時系列 (レベル別、seed 中央値) + (3) eval_loss_exact
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for a in arms:
        h = a["H_T"]
        step = a["seeds"][0]["step"]
        dead = np.stack([1.0 - s["alive"] / float(s["width"]) for s in a["seeds"]])
        loss = np.stack([s["eval_loss_exact"] for s in a["seeds"]])
        ax[0].plot(step, np.median(dead, axis=0), color=col[h], lw=1.5, label=f"$H_T$={h}")
        ax[0].fill_between(step, np.quantile(dead, .25, axis=0),
                           np.quantile(dead, .75, axis=0), color=col[h], alpha=0.15, lw=0)
        ax[1].plot(step, np.median(loss, axis=0), color=col[h], lw=1.5, label=f"$H_T$={h}")
    ax[0].set_xlabel("step")
    ax[0].set_ylabel("dead_frac (厳密 $\\hat{p}$ < 0.05)")
    ax[0].set_title("堆積の進行 (seed 中央値 / 帯は IQR)")
    ax[0].legend(fontsize=8, ncol=2)
    ax[0].grid(alpha=0.3)
    ax[1].set_yscale("log")
    ax[1].set_xlabel("step")
    ax[1].set_ylabel("eval_loss_exact")
    ax[1].set_title("厳密損失 (32 パターン全サポート)")
    ax[1].legend(fontsize=8, ncol=2)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig_tw_series.png"), dpi=130)
    plt.close(fig)

    # (3) P3 探索: alive の median p̂ と t50
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.0))
    x = [float(np.log2(h)) for h in levels]
    for key, axi, ylab in (("p_hat_median_alive_final", ax[0], "alive ユニットの median $\\hat{p}$"),
                           ("t50_dead", ax[1], "t50 (dead が中間値へ初到達)")):
        for sd, g in df.groupby("seed"):
            g = g.sort_values("log2H")
            axi.plot(g.log2H, g[key], "-o", lw=0.7, ms=3, alpha=0.4, color="0.5")
        m = df.groupby("log2H")[key].median()
        axi.plot(m.index, m.values, "-o", lw=2.2, ms=6, color="crimson")
        axi.set_xticks(x)
        axi.set_xticklabels([str(h) for h in levels])
        axi.set_xlabel("教師 LTU 幅 $H_T$ (log2 軸)")
        axi.set_ylabel(ylab)
        axi.grid(alpha=0.3)
    ax[0].set_title("P3: 生存者の発火率")
    ax[1].set_title("P3: 堆積の速さ")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig_tw_p3.png"), dpi=130)
    plt.close(fig)
    return fig_dir


# ---------------------------------------------------------------- summary

def _md(dfx, fmt=".4f"):
    cols = [str(c) for c in dfx.columns]
    def cell(v):
        if isinstance(v, (bool, np.bool_)):
            return str(bool(v))
        if isinstance(v, (float, np.floating)):
            return f"{v:{fmt}}" if np.isfinite(v) else "—"
        return str(v)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in dfx.iterrows():
        out.append("| " + " | ".join(cell(v) for v in r.to_list()) + " |")
    return "\n".join(out)


def write_summary(resdir, df, V, gates, ps, extra, cfg, arms):
    P = cfg["teachw"]
    g = lambda i: V[V.id == i].iloc[0]
    levels = sorted(df.H_T.unique())
    valid, enough = extra["valid"], extra["enough"]
    p1 = g("P1")

    if not enough:
        headline = (f"**判定保留**。P0 の有効レベルが {len(valid)} 本 "
                    f"({valid}) で事前登録の下限 {P['p0_min_levels']} に届かないため、"
                    "P1 は posreset G0 規約に従い判定しない。")
    elif p1.result == "PASS":
        headline = ("**P1 PASS = 「生存者数は教師複雑度に単調増加」を支持**"
                    "(ただし §1 のとおり単調性は必要条件であって十分条件ではない)。")
    else:
        headline = ("**P1 FAIL = 「生存者数 = タスクを表現するのに必要なユニット数」"
                    "予言は、このダイヤル・このスコープで棄却**(§1 のキルライン)。")

    piv = df.pivot_table(index="seed", columns="H_T", values="alive_final").astype(int)
    piv.insert(0, "seed", piv.index)
    L = ["# teachw_0820: 教師複雑度スイープ (生存者数 = 必要ユニット数か)", "",
         "仕様: `specs/spec_teachw_0820.md` (実行前にコミット済み = 事前登録)  ",
         f"生成: {pd.Timestamp.now():%Y-%m-%d %H:%M:%S}  ",
         f"レジーム: condA A_w100 / m={cfg['condA']['m']}, f={cfg['condA']['f']}, "
         f"T={cfg['condA']['T_values'][0]}, std, batch=1, lr={cfg['common']['lr_main']} / "
         f"seed {df.seed.nunique()} 本 / {cfg['common']['total_steps']:,} step / "
         f"アーム $H_T$ ∈ {levels}", "",
         "## リポジトリ来歴 (監査用)", "",
         "本実験は `github.com/Issan0511/lop_analysis` の **main ブランチ**上で実施した。"
         "作業ディレクトリ名がローカルで `proj_004_drift` なのは同一 origin の 2 つ目の "
         "clone だからで、別リポジトリではない (`git remote -v` で確認可能)。", "",
         "## 結論 (一行)", "", headline, "",
         "## 主判定量の定義", "",
         "alive_final = #{i: p̂_i ≥ 0.05}。p̂ は t=1M における入力分布の**全サポート "
         "(2^(m−f)=32 パターン) 上の厳密ゲート率**で、有限 eval バッチの推定ではない。"
         "dead_frac_final = 1 − alive_final/100 は ratchet_log_0819 の同名量と同一定義 "
         "(S3 で seed 別に照合)。", "",
         "## 判定表 (§6)", "",
         _md(V[["id", "statistic", "point", "ci_lo", "ci_hi", "threshold", "result",
                "note"]]), "",
         "## P0 前提ゲート (レベル別)", "",
         _md(gates), "",
         f"有効レベル: {valid} ({len(valid)} 本, 下限 {P['p0_min_levels']})。"
         + ("" if enough else " **不足 → 全体判定保留**。"), "",
         "## レベル × seed の alive_final 行列", "",
         _md(piv, fmt=".0f"), "",
         "## P1 の seed 別内訳", "",
         (_md(ps) if len(ps) else "(有効レベル不足で算出せず)"), "",
         "## P3 (探索的, 判定なし)", "",
         _md(df.groupby("H_T").agg(
             alive_final_median=("alive_final", "median"),
             p_hat_median_alive=("p_hat_median_alive_final", "median"),
             loss_plateau_median=("eval_loss_exact_plateau", "median"),
             loss_t0_median=("eval_loss_exact_t0", "median"),
             t50_median=("t50_dead", "median"),
             alive_t0_median=("alive_t0", "median")).reset_index(), fmt=".4g"), "",
         "## サニティ (§7)", "",
         f"- S1 (`OMP_NUM_THREADS=1`): {g('S1').result} — {g('S1').note}",
         f"- S2 (flip_state 軌跡の全アーム一致): {g('S2').result} — {g('S2').note}",
         f"- S3 (H_T=100 アンカー再現): {g('S3').result} — {g('S3').note}",
         f"- S4 (Var[y] 帯, 字義=中央値): {g('S4').result} — {g('S4').note}",
         f"- S4' (同, 意図=平均): {g('S4_intent').result} — {g('S4_intent').note}", "",
         _md(pd.DataFrame(extra["s4"]["rows"]), fmt=".4g"), "",
         "## 逸脱・留保 (§9)", "",
         "1. **Phase 0-1 の照合条件**: 仕様 §4-1 の字義は「H_T=100・seed 0 の 50k "
         "スモークが ratchet_log_0819 と state hash 一致」だが、学習は R 系列を"
         "ベクトル化して回すので `torch.randint(..., (R,...))` の抽選列が R に依存し、"
         "seed 0 単独 (R=1) の軌道は seed 0–9 群 (R=10) の seed 0 成分と原理的に一致"
         "しない。アンカー側の実測ハッシュと**同一条件** (R=10 / 100k step / probe なし) "
         "で照合した (`phase0_summary.md` 0-1)。字義版より強い検査になっている。",
         "2. **probe の常駐**: 仕様 §3 は「t=1M で厳密 p̂ を 1 回計算 (フル probe 常駐は"
         "不要)」だが、P3 の時系列 (dead 進行の t50・loss plateau) と S4 のログ内完結の"
         "ために **1000 step ごとに厳密 p̂ / eval_loss_exact / Var[y] を記録**した。"
         "記録は読み取り専用で、probe あり/なしの最終 state hash が bit 一致することを"
         "Phase 0-4 で実測している (t=1M の値は仕様どおりの 1 点計算と同一)。",
         "3. **P0 のレベル内 seed 集計**: 仕様 §6 は「各レベルで ≥ 0.5」としか書いておらず"
         "seed 集計を指定していない。**中央値**を主とし、平均も併記した "
         "(`agree_mean` 列が判定の一致を示す)。",
         "4. **t50 の定義**: 初期値補正版 (dead が d0 + (d_final − d0)/2 に初到達する "
         "step)。condA は t=0 で既に dead ≈ 0.25 あり、生の「final の半分」定義だと "
         "多くの seed で t=0 が答えになる。P3 は判定に使わない探索的指標。",
         "5. **スナップショット**: `results/**/snapshots/*.pt` は既存 .gitignore の対象"
         "なのでリポジトリには入らない (各アーム `H*/snapshots/` にローカル保存)。"
         "判定に使った生ログ (`H*/logs/seed*.npz`) は commit している。",
         "6. **実装の追加**: `train.train_group` はループ後の `total` で snapshot を"
         "書いていなかった (probe / ckpt は補っていた) ので補完した。既存の呼び出しは"
         "いずれも `t_int < total` なので挙動は不変。", "",
         "## スコープ (§9)", "",
         "**condA・w100・T=1e4・batch=1・LTU 教師族・スケーリング則 y·√(100/H_T)** に"
         "限定。condB・他教師族へ外挿しない。単調性は役立ち説の必要条件であって"
         "十分条件ではない (複雑度と相関する第三因子の可能性)。", ""]
    p = os.path.join(resdir, "summary.md")
    with open(p, "w") as fh:
        fh.write("\n".join(L) + "\n")
    return p


def main():
    resdir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "teachw_0820")
    import yaml
    arms = load_arms(resdir)
    cfg = yaml.safe_load(open(os.path.join(arms[0]["dir"], "config_used.yaml")))
    df = per_seed_metrics(arms)
    V, gates, ps, extra = judge(df, cfg["teachw"], arms)

    df.to_csv(os.path.join(resdir, "per_seed_metrics.csv"), index=False)
    V.to_csv(os.path.join(resdir, "verdict.csv"), index=False)
    gates.to_csv(os.path.join(resdir, "p0_gates.csv"), index=False)
    runs = pd.concat([pd.read_csv(os.path.join(a["dir"], "runs.csv")).assign(
        H_T=a["H_T"], out_scale=a["meta"]["out_scale"]) for a in arms])
    runs.to_csv(os.path.join(resdir, "runs.csv"), index=False)
    with open(os.path.join(resdir, "sanity.json"), "w") as fh:
        json.dump(extra, fh, indent=1, default=str, ensure_ascii=False)
    make_figures(arms, df, resdir)
    p = write_summary(resdir, df, V, gates, ps, extra, cfg, arms)
    print(V[["id", "statistic", "point", "result"]].to_string(index=False))
    print(f"-> {p}")


if __name__ == "__main__":
    main()
