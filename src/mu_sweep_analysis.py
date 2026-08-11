"""mu 強度スイープ解析: 1/D* の ||mu||^2 直線性検証 (SDE ノート 2026-08-10 の予測)。

  python -m src.mu_sweep_analysis results/mu_sweep_0811

予測:  1/D* = 2*lambda_align/sigma^2 + (2*kappa/sigma^2) * ||mu||^2,   ||mu|| = c
D は重み方向 u_i = W_i/||W_i|| の多様性 D = 1 - ||ubar||^2 (E[g] の方向ではない)。
主系列は w_D_alive (生存ユニットのみ)。判定基準は CLAUDE.md / 本ファイル末尾 VERDICT 参照。
"""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(0)
N_BOOT = 10000


def wls_fit(x, y, w):
    """y = a + b x の重み付き最小二乗。(a, b, R2w) を返す。"""
    W = np.asarray(w, float)
    X = np.stack([np.ones_like(x), x], axis=1)
    A = X.T @ (W[:, None] * X)
    beta = np.linalg.solve(A, X.T @ (W * y))
    yhat = X @ beta
    ybar = np.average(y, weights=W)
    ss_res = np.sum(W * (y - yhat) ** 2)
    ss_tot = np.sum(W * (y - ybar) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return beta[0], beta[1], r2


def quad_fit(c, y, w):
    """y = a + g*c + b*c^2 (奇数項つき)。(a, g, b) を返す。"""
    W = np.asarray(w, float)
    X = np.stack([np.ones_like(c), c, c ** 2], axis=1)
    A = X.T @ (W[:, None] * X)
    return np.linalg.solve(A, X.T @ (W * y))


def bootstrap_ci(per_seed, fit_fn, n_boot=N_BOOT):
    """per_seed: {c: array of per-seed y}. シードを c 内で復元抽出して fit を繰り返す。
    fit_fn: (x_arr, y_arr, w_arr) -> params。percentile 95% CI を返す。"""
    cs = sorted(per_seed.keys())
    out = []
    for _ in range(n_boot):
        xs, ys, ws = [], [], []
        for c in cs:
            v = per_seed[c]
            s = RNG.choice(v, size=len(v), replace=True)
            xs.append(c); ys.append(s.mean())
            sem = s.std(ddof=1) / np.sqrt(len(s)) if len(s) > 1 else np.nan
            ws.append(1.0 / max(sem, 1e-9) ** 2 if np.isfinite(sem) and sem > 0 else 1.0)
        try:
            out.append(fit_fn(np.array(xs), np.array(ys), np.array(ws)))
        except np.linalg.LinAlgError:
            continue
    out = np.array(out)
    lo = np.percentile(out, 2.5, axis=0)
    hi = np.percentile(out, 97.5, axis=0)
    return lo, hi


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

    m = pd.read_csv(os.path.join(R, f"lop_metrics_{args.group}.csv"))
    runs = pd.read_csv(os.path.join(R, "runs.csv"))
    df = m.merge(runs[["run_id", "c", "seed"]], on="run_id")
    smax = df.step.max()
    t0 = (1.0 - args.tail_frac) * smax
    tail = df[df.step >= t0].copy()

    # --- 発散ラン除外 (tail に NaN eval_loss)
    diverged = sorted(tail[tail.eval_loss.isna()].run_id.unique())
    tail = tail[~tail.run_id.isin(diverged)]

    # --- run ごとの D*, plateau, 診断量
    rows = []
    for rid, g in tail.groupby("run_id"):
        d = g[args.series].dropna()
        if len(d) < max(3, 0.75 * len(g)):
            continue                     # tail の大半が NaN (alive<2 等) -> 除外
        Dstar = d.mean()
        sl = np.polyfit(g.loc[d.index, "step"], d, 1)[0] if len(d) > 2 else np.nan
        rows.append(dict(run_id=rid, c=g.c.iloc[0], seed=g.seed.iloc[0],
                         Dstar=Dstar, invD=1.0 / Dstar,
                         rel_drift=abs(sl) * (smax - t0) / Dstar if np.isfinite(sl) else np.nan,
                         n_alive=g.n_alive.mean(), dead=g.dead_frac.mean(),
                         loss=g.eval_loss.mean(), rnorm=g.w_rnorm_mean.mean()))
    per = pd.DataFrame(rows)
    per.to_csv(os.path.join(R, "mu_sweep_per_run.csv"), index=False)

    # --- c ごとの集計と点フィルタ
    agg = per.groupby("c").agg(invD=("invD", "mean"), sd=("invD", "std"),
                               n=("invD", "size"), n_alive=("n_alive", "mean"),
                               dead=("dead", "mean"), loss=("loss", "mean"),
                               rnorm=("rnorm", "mean"),
                               drift=("rel_drift", "median")).reset_index()
    agg["sem"] = agg.sd / np.sqrt(agg.n)
    agg["mu2"] = agg.c ** 2
    agg["used"] = agg.n_alive >= args.min_alive
    use = agg[agg.used]
    per_seed = {r.c: per[per.c == r.c].invD.values for _, r in use.iterrows()}

    # --- フィット M1 (線形 in mu^2) と M2 (奇数項つき)
    w = 1.0 / use["sem"].clip(lower=1e-9) ** 2
    a, b, r2 = wls_fit(use.mu2.values, use.invD.values, w.values)
    # bootstrap には x=c を渡すので、二乗して使う fit で包む
    def fitM1(cs, ys, ws): aa, bb, _ = wls_fit(cs ** 2, ys, ws); return (aa, bb)
    (a_lo, b_lo), (a_hi, b_hi) = bootstrap_ci(per_seed, fitM1)
    def fitM2(cs, ys, ws): return tuple(quad_fit(cs, ys, ws))
    (q_alo, g_lo, q_blo), (q_ahi, g_hi, q_bhi) = bootstrap_ci(per_seed, fitM2)

    # --- 図
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for used, mk in ((True, "o"), (False, "s")):
        gsel = agg[agg.used == used]
        if len(gsel):
            ax.errorbar(gsel.mu2, gsel.invD, yerr=gsel["sem"], fmt=mk, ms=5, color="C0",
                        mfc=("C0" if used else "white"), capsize=3,
                        label=None if used else f"除外 (n_alive<{args.min_alive:g})")
    for _, r in per.iterrows():
        ax.plot(r.c ** 2, r.invD, ".", color="gray", alpha=0.35, ms=3)
    xx = np.linspace(0, agg.mu2.max() * 1.05, 100)
    ax.plot(xx, a + b * xx, "C1-", label=f"WLS: {a:.3g} + {b:.3g}·‖μ‖²  (R²w={r2:.3f})")
    ax.set_xlabel("‖μ‖² = c²"); ax.set_ylabel(f"1/D*  ({args.series}, tail {args.tail_frac:.0%})")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(figdir, "fig_m1_invD_vs_mu2.png"), dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for c, g in df.groupby("c"):
        gg = g.groupby("step")[args.series].mean()
        ax.plot(gg.index, gg.values, label=f"c={c}")
    ax.axvspan(t0, smax, color="k", alpha=0.06)
    ax.set_xlabel("step"); ax.set_ylabel(args.series); ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "fig_m2_D_vs_step.png"), dpi=150); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    ax[0].plot(agg.c, agg.dead, "o-"); ax[0].set_xlabel("c"); ax[0].set_ylabel("dead_frac (tail)")
    ax[1].plot(agg.c, agg.n_alive, "o-"); ax[1].axhline(args.min_alive, ls="--", c="r")
    ax[1].set_xlabel("c"); ax[1].set_ylabel("n_alive (tail mean)")
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "fig_m3_dead_alive_vs_c.png"), dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for c, g in df.groupby("c"):
        gg = g.groupby("step")["w_rnorm_mean"].mean()
        ax.plot(gg.index, gg.values, label=f"c={c}")
    ax.set_xlabel("step"); ax.set_ylabel("mean ||W_i|| "); ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "fig_m4_rnorm_vs_step.png"), dpi=150); plt.close(fig)

    # --- VERDICT (CLAUDE.md の判定基準をコード化)
    npts = int(use.shape[0])
    lin_ok = (npts >= 5) and (r2 >= 0.90) and (b_lo > 0)
    align_ok = a_lo > 0
    odd_ok = (g_lo <= 0 <= g_hi)
    plateau_bad = float((per.rel_drift > 0.10).mean()) if len(per) else np.nan
    loss_ratio = agg.loss.max() / max(agg.loss.min(), 1e-12)

    L = []
    L.append("# mu_sweep 解析結果 (自動生成)\n")
    L.append(f"- 系列: {args.series} / group {args.group} / tail {args.tail_frac:.0%} "
             f"/ 有効点 {npts} (n_alive≥{args.min_alive:g}) / 発散除外 {len(diverged)} run\n")
    L.append("## 判定\n")
    L.append(f"| 基準 | 値 | 判定 |\n|---|---|---|\n")
    L.append(f"| 直線性 (点数≥5 & R²w≥0.90 & β>0) | R²w={r2:.3f}, β={b:.4g} "
             f"[{b_lo:.4g}, {b_hi:.4g}] | {'PASS' if lin_ok else 'FAIL'} |\n")
    L.append(f"| 切片 α>0 (λ_align>0, Phase1) | α={a:.4g} [{a_lo:.4g}, {a_hi:.4g}] "
             f"| {'PASS' if align_ok else 'FAIL'} |\n")
    L.append(f"| 奇数項 γ の CI が 0 を含む (λ_lock 偶関数) | γ∈[{g_lo:.4g}, {g_hi:.4g}] "
             f"| {'PASS' if odd_ok else 'FAIL'} |\n")
    L.append(f"| plateau (rel_drift>0.1 の run 割合 <50%) | {plateau_bad:.0%} "
             f"| {'PASS' if plateau_bad < 0.5 else 'HOLD: total_steps 延長を検討'} |\n")
    L.append(f"\n診断: tail eval_loss 比 (max/min across c) = {loss_ratio:.2f}"
             f"{' — σ²一定の仮定に注意 (>3)' if loss_ratio > 3 else ''}; "
             f"mean ||W_i|| 範囲 {agg.rnorm.min():.3g}–{agg.rnorm.max():.3g}\n")
    L.append("\n## c ごとの集計\n\n" + agg.round(4).to_markdown(index=False) + "\n")
    open(os.path.join(R, "mu_sweep_summary.md"), "w").write("".join(L))
    print("".join(L))


if __name__ == "__main__":
    main()
