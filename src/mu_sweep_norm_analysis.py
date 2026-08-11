"""ノルム固定 × mu 強度スイープ解析 — SDEノート 2026-08-10「検証可能な予測」の検証。

  python3 -m src.mu_sweep_norm_analysis results/mu_sweep_norm_0811

検証対象:
  (P1) 1/D* は ||mu||^2 = c^2 に対して直線。主判定は norm=fixed アーム (||w|| 交絡なし)。
  (P2) 縦軸 D は pairwise cos cbar から D = (1-1/n)(1-cbar) で読める (恒等式の数値確認)。
  (P3) 問題③: free アームは ||w|| と交絡 -> rnorm 共変量つき二次フィットで分離。
出力: norm_sweep_report.md / norm_sweep_per_run.csv / figures/fig_n1..n4.png
このスクリプトは判定基準を含めて固定されている。閾値・式の変更は行わないこと。
"""
import argparse
import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .mu_sweep_analysis import wls_fit, quad_fit, bootstrap_ci

RNG = np.random.default_rng(0)
N_BOOT = 10000
ARMS = ["fixed", "free"]        # fixed が主判定


def per_run_table(tail, series, smax, t0):
    rows = []
    for rid, g in tail.groupby("run_id"):
        d = g[series].dropna()
        if len(d) < max(3, 0.75 * len(g)):
            continue
        Dstar = d.mean()
        sl = np.polyfit(g.loc[d.index, "step"], d, 1)[0] if len(d) > 2 else np.nan
        rn = g.w_rnorm_alive.mean() if g.w_rnorm_alive.notna().any() else g.w_rnorm_mean.mean()
        rows.append(dict(run_id=rid, norm=g.norm.iloc[0], c=g.c.iloc[0], seed=g.seed.iloc[0],
                         Dstar=Dstar, invD=1.0 / Dstar,
                         rel_drift=abs(sl) * (smax - t0) / Dstar if np.isfinite(sl) else np.nan,
                         n_alive=g.n_alive.mean(), dead=g.dead_frac.mean(),
                         loss=g.eval_loss.mean(), rnorm=rn))
    return pd.DataFrame(rows)


def analyze_arm(per, min_alive):
    agg = per.groupby("c").agg(invD=("invD", "mean"), sd=("invD", "std"), n=("invD", "size"),
                               n_alive=("n_alive", "mean"), dead=("dead", "mean"),
                               loss=("loss", "mean"), rnorm=("rnorm", "mean"),
                               drift=("rel_drift", "median")).reset_index()
    agg["sem"] = agg.sd / np.sqrt(agg.n)
    agg["mu2"] = agg.c ** 2
    agg["used"] = agg.n_alive >= min_alive
    use = agg[agg.used]
    per_seed = {r.c: per[per.c == r.c].invD.values for _, r in use.iterrows()}
    w = 1.0 / use["sem"].clip(lower=1e-9) ** 2
    a, b, r2 = wls_fit(use.mu2.values, use.invD.values, w.values)

    def fitM1(cs, ys, ws):
        aa, bb, _ = wls_fit(cs ** 2, ys, ws); return (aa, bb)
    (a_lo, b_lo), (a_hi, b_hi) = bootstrap_ci(per_seed, fitM1)

    def fitM2(cs, ys, ws):
        return tuple(quad_fit(cs, ys, ws))
    (q_alo, g_lo, q_blo), (q_ahi, g_hi, q_bhi) = bootstrap_ci(per_seed, fitM2)
    _, g_pt, _ = quad_fit(use.c.values, use.invD.values, w.values)

    npts = int(use.shape[0])
    plateau_bad = float((per.rel_drift > 0.10).mean()) if len(per) else np.nan
    return dict(agg=agg, use=use, npts=npts, a=a, b=b, r2=r2,
                a_lo=a_lo, a_hi=a_hi, b_lo=b_lo, b_hi=b_hi,
                g=g_pt, g_lo=g_lo, g_hi=g_hi, plateau_bad=plateau_bad,
                lin_ok=(npts >= 5) and (r2 >= 0.90) and (b_lo > 0),
                align_ok=a_lo > 0, odd_ok=(g_lo <= 0 <= g_hi),
                loss_ratio=agg.loss.max() / max(agg.loss.min(), 1e-12))


def m3_covariate_fit(per):
    """free アーム per-run 行での OLS: invD ~ 1 + c^2 + c + (rnorm-1)。seed 復元抽出 CI。"""
    def design(s):
        return np.stack([np.ones(len(s)), s.c.values ** 2, s.c.values,
                         s.rnorm.values - 1.0], axis=1)
    pt, *_ = np.linalg.lstsq(design(per), per.invD.values, rcond=None)
    groups = {c: g.index.values for c, g in per.groupby("c")}
    out = []
    for _ in range(N_BOOT):
        idx = np.concatenate([RNG.choice(v, size=len(v), replace=True)
                              for v in groups.values()])
        s = per.loc[idx]
        try:
            beta, *_ = np.linalg.lstsq(design(s), s.invD.values, rcond=None)
            out.append(beta)
        except np.linalg.LinAlgError:
            continue
    p = np.array(out)
    return pt, np.percentile(p, 2.5, axis=0), np.percentile(p, 97.5, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--group", default="B_w100")
    ap.add_argument("--tail-frac", type=float, default=0.4)
    ap.add_argument("--min-alive", type=float, default=20.0)
    ap.add_argument("--series", default="w_D_alive", choices=["w_D_alive", "w_D_all"])
    args = ap.parse_args()
    R = args.outdir.rstrip("/")
    figdir = os.path.join(R, "figures"); os.makedirs(figdir, exist_ok=True)
    h = int(re.search(r"_w(\d+)$", args.group).group(1))

    m = pd.read_csv(os.path.join(R, f"lop_metrics_{args.group}.csv"))
    runs = pd.read_csv(os.path.join(R, "runs.csv"))
    if "norm" not in runs.columns:
        runs["norm"] = np.where(runs.run_id.str.endswith("_nfix"), "fixed", "free")
    df = m.merge(runs[["run_id", "c", "seed", "norm"]], on="run_id")
    smax = df.step.max(); t0 = (1.0 - args.tail_frac) * smax
    tail = df[df.step >= t0].copy()
    diverged = sorted(tail[tail.eval_loss.isna()].run_id.unique())
    tail = tail[~tail.run_id.isin(diverged)]

    # --- (P2) 恒等式チェック: D = (1-1/n)(1-cbar)。alive 版 + 全ユニット版
    ok = tail[(tail.n_alive >= 2) & tail.w_D_alive.notna() & tail.w_paircos_alive.notna()]
    id_err_alive = float((ok.w_D_alive - (1 - 1 / ok.n_alive) * (1 - ok.w_paircos_alive))
                         .abs().max()) if len(ok) else np.nan
    ok2 = tail[tail.w_D_all.notna() & tail.w_paircos_all.notna()]
    id_err_all = float((ok2.w_D_all - (1 - 1.0 / h) * (1 - ok2.w_paircos_all))
                       .abs().max()) if len(ok2) else np.nan
    id_ok = (id_err_alive < 1e-4) and (id_err_all < 1e-4)

    # --- (P3-a) 操作チェック: fixed アームの rnorm が学習を通じて不変
    fx = df[df.norm == "fixed"]
    r0 = fx[fx.step == 0].set_index("run_id").w_rnorm_mean
    rel = (fx.set_index("run_id").w_rnorm_mean / r0 - 1.0).abs()
    manip_err = float(rel.max()) if len(rel) else np.nan
    manip_ok = manip_err < 1e-3

    per = per_run_table(tail, args.series, smax, t0)
    per.to_csv(os.path.join(R, "norm_sweep_per_run.csv"), index=False)
    res = {arm: analyze_arm(per[per.norm == arm].reset_index(drop=True), args.min_alive)
           for arm in ARMS if (per.norm == arm).any()}

    # --- (P3-b) free アーム共変量フィット
    m3_pt = m3_lo = m3_hi = None
    if "free" in res:
        m3_pt, m3_lo, m3_hi = m3_covariate_fit(per[per.norm == "free"].reset_index(drop=True))

    # --- 図
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for arm, col in (("fixed", "C0"), ("free", "C1")):
        if arm not in res:
            continue
        r_ = res[arm]
        ax.errorbar(r_["agg"].mu2, r_["agg"].invD, yerr=r_["agg"]["sem"], fmt="o", ms=5,
                    color=col, capsize=3, label=f"{arm}: {r_['a']:.3g}+{r_['b']:.3g}·‖μ‖² (R²w={r_['r2']:.3f})")
        xx = np.linspace(0, r_["agg"].mu2.max() * 1.05, 100)
        ax.plot(xx, r_["a"] + r_["b"] * xx, "-", color=col, alpha=0.7)
    ax.set_xlabel("‖μ‖² = c²"); ax.set_ylabel(f"1/D* ({args.series}, tail {args.tail_frac:.0%})")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(figdir, "fig_n1_invD_vs_mu2_arms.png"), dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax_, arm in zip(axes, ARMS):
        for c, g in df[df.norm == arm].groupby("c"):
            gg = g.groupby("step")[args.series].mean()
            ax_.plot(gg.index, gg.values, label=f"c={c}")
        ax_.axvspan(t0, smax, color="k", alpha=0.06)
        ax_.set_title(arm); ax_.set_xlabel("step")
    axes[0].set_ylabel(args.series); axes[1].legend(fontsize=6, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "fig_n2_D_vs_step_arms.png"), dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for arm, ls in (("fixed", "-"), ("free", "--")):
        for c, g in df[df.norm == arm].groupby("c"):
            gg = g.groupby("step")["w_rnorm_mean"].mean()
            ax.plot(gg.index, gg.values, ls, lw=1, alpha=0.7)
    ax.set_xlabel("step"); ax.set_ylabel("mean ||W_i||  (実線=fixed, 破線=free)")
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "fig_n3_rnorm_check.png"), dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for arm, col in (("fixed", "C0"), ("free", "C1")):
        if arm not in res:
            continue
        a_ = res[arm]["agg"]
        axes[0].plot(a_.c, a_.dead, "o-", color=col, label=arm)
        axes[1].plot(a_.c, a_.n_alive, "o-", color=col, label=arm)
    axes[0].set_xlabel("c"); axes[0].set_ylabel("dead_frac (tail)"); axes[0].legend()
    axes[1].axhline(args.min_alive, ls="--", c="r"); axes[1].set_xlabel("c")
    axes[1].set_ylabel("n_alive (tail mean)")
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "fig_n4_dead_alive_arms.png"), dpi=150); plt.close(fig)

    # --- レポート
    L = ["# norm_sweep 解析結果 (自動生成)\n",
         f"- 系列: {args.series} / group {args.group} / tail {args.tail_frac:.0%} "
         f"/ 発散除外 {len(diverged)} run: {diverged if diverged else 'なし'}\n"]
    L.append("\n## 前提チェック\n\n| 項目 | 値 | 判定 |\n|---|---|---|\n")
    L.append(f"| (P2) 恒等式 D=(1-1/n)(1-c̄) 最大誤差 (alive / all) "
             f"| {id_err_alive:.3g} / {id_err_all:.3g} | {'PASS' if id_ok else 'FAIL'} |\n")
    L.append(f"| (P3-a) fixed アーム rnorm 相対ドリフト最大 "
             f"| {manip_err:.3g} | {'PASS' if manip_ok else 'FAIL'} |\n")
    for arm in ARMS:
        if arm not in res:
            continue
        r_ = res[arm]
        L.append(f"\n## アーム: {arm}{'（主判定）' if arm == 'fixed' else '（参考: ‖w‖ 交絡あり）'}\n\n")
        L.append(f"有効点 {r_['npts']} (n_alive≥{args.min_alive:g})\n\n| 基準 | 値 | 判定 |\n|---|---|---|\n")
        L.append(f"| (P1) 直線性 (点数≥5 & R²w≥0.90 & β>0) | R²w={r_['r2']:.3f}, "
                 f"β={r_['b']:.4g} [{r_['b_lo']:.4g}, {r_['b_hi']:.4g}] | {'PASS' if r_['lin_ok'] else 'FAIL'} |\n")
        L.append(f"| 切片 α>0 (λ_align>0, Phase1) | α={r_['a']:.4g} "
                 f"[{r_['a_lo']:.4g}, {r_['a_hi']:.4g}] | {'PASS' if r_['align_ok'] else 'FAIL'} |\n")
        L.append(f"| 奇数項 γ の CI が 0 を含む (λ_lock 最低次 c²) | γ={r_['g']:.4g} "
                 f"[{r_['g_lo']:.4g}, {r_['g_hi']:.4g}] | {'PASS' if r_['odd_ok'] else 'FAIL'} |\n")
        L.append(f"| plateau (rel_drift>0.1 の run 割合 <50%) | {r_['plateau_bad']:.0%} "
                 f"| {'PASS' if r_['plateau_bad'] < 0.5 else 'HOLD: total_steps 延長を検討'} |\n")
        L.append(f"\n診断: tail eval_loss 比 = {r_['loss_ratio']:.2f}; "
                 f"mean ||W_i|| 範囲 {r_['agg'].rnorm.min():.3g}–{r_['agg'].rnorm.max():.3g}\n")
        L.append("\n" + r_["agg"].round(4).to_markdown(index=False) + "\n")
    if m3_pt is not None:
        L.append("\n## (P3-b) free アーム共変量フィット invD ~ 1 + c² + c + (rnorm−1)\n\n")
        names = ["切片", "β(c²)", "γ(c)", "δ(rnorm−1)"]
        L.append("| 項 | 点推定 | 95% CI |\n|---|---|---|\n")
        for nm, p, lo, hi in zip(names, m3_pt, m3_lo, m3_hi):
            L.append(f"| {nm} | {p:.4g} | [{lo:.4g}, {hi:.4g}] |\n")
        L.append("\n読み方: rnorm 共変量投入後も β の CI が 0 を上回れば、free アームの傾きは"
                 "ノルム交絡だけでは説明できない。γ は fixed アームの γ と突き合わせる"
                 "(fixed で γ≠0 なら 1 次項は交絡ではなく力学)。\n")
    # 報告テンプレ
    L.append("\n## 報告用テンプレ（この節を丸ごと貼り付けて報告する）\n\n```\n")
    L.append(f"[mu_sweep_norm_0811 結果]\n発散除外: {len(diverged)} run\n")
    L.append(f"恒等式チェック: {id_err_alive:.3g} ({'PASS' if id_ok else 'FAIL'}) / "
             f"rnorm固定チェック: {manip_err:.3g} ({'PASS' if manip_ok else 'FAIL'})\n")
    for arm in ARMS:
        if arm not in res:
            continue
        r_ = res[arm]
        L.append(f"{arm}: R2w={r_['r2']:.3f} beta={r_['b']:.4g}[{r_['b_lo']:.4g},{r_['b_hi']:.4g}] "
                 f"alpha={r_['a']:.4g}[{r_['a_lo']:.4g},{r_['a_hi']:.4g}] "
                 f"gamma={r_['g']:.4g}[{r_['g_lo']:.4g},{r_['g_hi']:.4g}] "
                 f"plateau_bad={r_['plateau_bad']:.0%} "
                 f"判定: 直線性{'PASS' if r_['lin_ok'] else 'FAIL'}/切片{'PASS' if r_['align_ok'] else 'FAIL'}/"
                 f"奇数項{'PASS' if r_['odd_ok'] else 'FAIL'}\n")
    if m3_pt is not None:
        L.append(f"free共変量後: beta={m3_pt[1]:.4g}[{m3_lo[1]:.4g},{m3_hi[1]:.4g}] "
                 f"gamma={m3_pt[2]:.4g}[{m3_lo[2]:.4g},{m3_hi[2]:.4g}] "
                 f"delta={m3_pt[3]:.4g}[{m3_lo[3]:.4g},{m3_hi[3]:.4g}]\n")
    L.append("```\n")
    open(os.path.join(R, "norm_sweep_report.md"), "w").write("".join(L))
    print("".join(L))


if __name__ == "__main__":
    main()
