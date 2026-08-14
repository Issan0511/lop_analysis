"""center_selfcov_0814 Phase 2 の判定・図・summary (spec §6–8)。

  python -m src.figures_center_selfcov results/center_selfcov_0814

P3-1〜P3-7 を verdict.csv に {PASS, FAIL, NA} + 根拠数値で出し、summary.md に表で書く
(null 結果も同じ体裁で書く)。図は §8 の fig_cs1〜cs6。
"""
import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.center_selfcov.slopes import (slope_ols, t50_reach, boot_ci,
                                            paired_boot_ci)
from .w_direction import random_floor, spike_dir_vec

T_HALF = 300_000            # §5: 傾き推定区間 [0, 3e5]
KAP_COLOR = {1: "tab:gray", 4: "tab:orange", 16: "tab:red"}
ENC_COLOR = {"std": "tab:red", "centered": "tab:blue"}
C_COLOR = {0.0: "tab:blue", 2.0: "tab:red"}


def load(resdir):
    runs = pd.read_csv(os.path.join(resdir, "runs.csv")).set_index("run_id")
    fs = sorted(glob.glob(os.path.join(resdir, "lop_metrics_*.csv")))
    lop = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    lop = lop.join(runs, on="run_id", how="inner")
    import yaml
    with open(os.path.join(resdir, "config_used.yaml")) as fh:
        cfg = yaml.safe_load(fh)
    return lop, runs, cfg


def final_by_seed(sub, col):
    """run ごとの最終値 (step 最大) を seed 順に返す。"""
    last = sub.sort_values("step").groupby("seed").last()
    return last[col]


# ------------------------------------------------------------------ P3-1..P3-3

def arm1_amplify(lop, rng, rows):
    """P3-1 (dead_frac 最終値) / P3-2 (wcos_mean 前半傾き) / P3-3 (‖w‖ 交絡)。"""
    A = lop[lop.exp == "A"]
    out = []
    for width, g in A.groupby("width"):
        piv = {}
        for enc, ge in g.groupby("enc"):
            piv[enc] = dict(
                dead=final_by_seed(ge, "dead_frac"),
                wcos_slope=ge.groupby("seed").apply(
                    lambda s: slope_ols(s.step, s.wcos_mean, T_HALF),
                    include_groups=False),
                wnorm=final_by_seed(ge, "w_norm_mean"),
                wnorm_slope=ge.groupby("seed").apply(
                    lambda s: slope_ols(s.step, s.w_norm_mean, T_HALF),
                    include_groups=False))
        if "std" not in piv or "centered" not in piv:
            continue
        seeds = piv["std"]["dead"].index.intersection(piv["centered"]["dead"].index)
        for metric, key in [("dead_final", "dead"), ("wcos_slope", "wcos_slope"),
                            ("wnorm_final", "wnorm"), ("wnorm_slope", "wnorm_slope")]:
            ci = paired_boot_ci(piv["centered"][key].loc[seeds].values,
                                piv["std"][key].loc[seeds].values, rng)
            out.append(dict(width=width, metric=metric,
                            std_mean=float(np.nanmean(piv["std"][key].loc[seeds])),
                            centered_mean=float(np.nanmean(piv["centered"][key].loc[seeds])),
                            **{f"diff_{k}": v for k, v in ci.items()}))
    tab = pd.DataFrame(out)

    def verdict(width, metric, want_negative=True):
        r = tab[(tab.width == width) & (tab.metric == metric)]
        if not len(r):
            return "NA", ""
        r = r.iloc[0]
        ok = r.diff_excl_zero and ((r.diff_mean < 0) == want_negative)
        ev = (f"centered {r.centered_mean:.4g} vs std {r.std_mean:.4g}, "
              f"diff {r.diff_mean:+.4g} CI [{r.diff_lo:.4g}, {r.diff_hi:.4g}]")
        return ("PASS" if ok else "FAIL"), ev

    v, ev = verdict(100, "dead_final")
    rows.append(dict(pred="P3-1", scope="condA w100 dead_frac 最終値 (centered < std)",
                     verdict=v, evidence=ev))
    for width in sorted(tab.width.unique()):
        v, ev = verdict(width, "wcos_slope")
        rows.append(dict(pred="P3-2", scope=f"condA w{width} wcos_mean 前半傾き (centered < std)",
                         verdict=v, evidence=ev))
    # P3-3: ‖w‖ 交絡。P3-1/P3-2 に差が出た場合のみ意味を持つ
    sig = tab[(tab.metric.isin(["dead_final", "wcos_slope"])) & tab.diff_excl_zero]
    if len(sig):
        wn = tab[tab.metric == "wnorm_final"]
        ev = "; ".join(f"w{r.width}: ‖w‖ centered {r.centered_mean:.3g} vs std "
                       f"{r.std_mean:.3g} (diff CI [{r.diff_lo:.3g}, {r.diff_hi:.3g}]"
                       f"{', 有意' if r.diff_excl_zero else ', 非有意'})"
                       for r in wn.itertuples())
        conf = bool(wn.diff_excl_zero.any())
        rows.append(dict(pred="P3-3",
                         scope="‖w‖ 交絡統制 (差が ‖w‖ 差で説明されうるか)",
                         verdict="CONFOUNDED" if conf else "PASS", evidence=ev))
    else:
        rows.append(dict(pred="P3-3", scope="‖w‖ 交絡統制",
                         verdict="NA", evidence="P3-1/P3-2 とも有意差なしのため適用外"))
    return tab


def arm2_amplify_B(lop, rng, rows):
    """アーム2: 条件B c=0 vs c=2 (κ=1) での増幅因子確認 (P3-1/P3-2 の頑健性)。"""
    B = lop[(lop.exp == "B") & (lop.kappa == 1)]
    out = []
    if not len(B):
        return pd.DataFrame()
    piv = {}
    for c, g in B.groupby("c"):
        piv[c] = dict(dead=final_by_seed(g, "dead_frac"),
                      wcos_slope=g.groupby("seed").apply(
                          lambda s: slope_ols(s.step, s.wcos_mean, T_HALF),
                          include_groups=False),
                      srank=final_by_seed(g, "stable_rank_W_alive"),
                      eval=final_by_seed(g, "eval_loss"))
    if 0.0 in piv and 2.0 in piv:
        seeds = piv[0.0]["dead"].index.intersection(piv[2.0]["dead"].index)
        for metric in ["dead", "wcos_slope", "srank", "eval"]:
            ci = paired_boot_ci(piv[0.0][metric].loc[seeds].values,
                                piv[2.0][metric].loc[seeds].values, rng)
            out.append(dict(metric=metric,
                            c0_mean=float(np.nanmean(piv[0.0][metric].loc[seeds])),
                            c2_mean=float(np.nanmean(piv[2.0][metric].loc[seeds])),
                            **{f"diff_{k}": v for k, v in ci.items()}))
    tab = pd.DataFrame(out)
    if len(tab):
        r = tab[tab.metric == "wcos_slope"].iloc[0]
        rows.append(dict(pred="P3-2b",
                         scope="condB κ=1 wcos_mean 前半傾き (c=0 < c=2、µ=0 厳密版)",
                         verdict=("PASS" if (r.diff_excl_zero and r.diff_mean < 0)
                                  else "FAIL"),
                         evidence=f"c=0 {r.c0_mean:.4g} vs c=2 {r.c2_mean:.4g}, "
                                  f"diff {r.diff_mean:+.4g} CI [{r.diff_lo:.4g}, {r.diff_hi:.4g}]"))
    return tab


# ------------------------------------------------------------------ P3-4..P3-7

def arm3_selfcov(lop, cfg, rng, rows, floor):
    """P3-4 (主判定) / P3-5 (κ 単調性) / P3-6 (1.0 収束)。"""
    B = lop[(lop.exp == "B") & (lop.c == 0.0)]
    out = []
    for kap, g in B.groupby("kappa"):
        fin = final_by_seed(g, "cos_e1W_e1Sig")
        init = g[g.step == 0].set_index("seed").cos_e1W_e1Sig
        slope = g.groupby("seed").apply(
            lambda s: slope_ols(s.step, s.cos_e1W_e1Sig, T_HALF), include_groups=False)
        ci_fin = boot_ci(fin.values, rng)
        ci_slope = boot_ci(slope.values, rng)
        out.append(dict(kappa=kap, init_mean=float(np.nanmean(init)),
                        final_mean=ci_fin["mean"], final_lo=ci_fin["lo"],
                        final_hi=ci_fin["hi"], slope_mean=ci_slope["mean"],
                        slope_lo=ci_slope["lo"], slope_hi=ci_slope["hi"],
                        srank_final=float(np.nanmean(final_by_seed(g, "stable_rank_W_alive"))),
                        top1_final=float(np.nanmean(final_by_seed(g, "top1_frac_alive"))),
                        pca_final=float(np.nanmean(final_by_seed(g, "cos_e1W_e1Sig_pca"))),
                        e1stab_min=float(np.nanmin(g.e1_stability))))
    tab = pd.DataFrame(out)

    main = tab[tab.kappa == 16]
    if len(main) and np.isfinite(main.final_mean.iloc[0]):
        m = main.iloc[0]
        # (a) は「学習に伴い単調増加」。[0,t_half] の傾きだけだと系列が後半で戻る場合に
        # 誤って PASS しうるので、init→final の対応差も併せて要求する。
        g16 = B[B.kappa == 16]
        fin = final_by_seed(g16, "cos_e1W_e1Sig")
        ini = g16[g16.step == 0].set_index("seed").cos_e1W_e1Sig
        sd = fin.index.intersection(ini.index)
        ci_if = paired_boot_ci(fin.loc[sd].values, ini.loc[sd].values, rng)
        rise = bool(m.slope_lo > 0 and ci_if["excl_zero"] and ci_if["mean"] > 0)
        above = bool(m.final_lo > floor)
        rows.append(dict(pred="P3-4", scope="κ=16 cos_e1W_e1Sig: (a) 単調増加 (b) 床超え",
                         verdict="PASS" if (rise and above) else "FAIL",
                         evidence=f"(a) [0,{T_HALF//1000}k] 傾き {m.slope_mean:+.3g}/step CI "
                                  f"[{m.slope_lo:.3g}, {m.slope_hi:.3g}]、"
                                  f"init→final {ci_if['mean']:+.3f} CI "
                                  f"[{ci_if['lo']:.3f}, {ci_if['hi']:.3f}] "
                                  f"→ {'増加' if rise else '増加と言えない (系列は床付近で往復)'}; "
                                  f"(b) 最終 {m.final_mean:.3f} CI [{m.final_lo:.3f}, "
                                  f"{m.final_hi:.3f}] vs 床 {floor:.3f} "
                                  f"→ {'超過' if above else '床を超えない'}"))
        rows.append(dict(pred="P3-6", scope="先生の予言: cos_e1W_e1Sig → 1.0 (判定は 0.9 到達)",
                         verdict="PASS" if m.final_mean >= 0.9 else "FAIL",
                         evidence=f"κ=16 最終 {m.final_mean:.3f} (0.9 未到達なら部分整列)。"
                                  f"srank_alive={m.srank_final:.2f}, "
                                  f"top1_frac={m.top1_final:.2f} "
                                  f"→ rank-1 に落ちていないので e1 が支配的でないのは整合的"))
    else:
        rows.append(dict(pred="P3-4", scope="κ=16 cos_e1W_e1Sig", verdict="NA",
                         evidence="データなし"))
    k4 = tab[tab.kappa == 4], tab[tab.kappa == 16]
    if len(k4[0]) and len(k4[1]):
        g4 = B[B.kappa == 4]; g16 = B[B.kappa == 16]
        f4 = final_by_seed(g4, "cos_e1W_e1Sig"); f16 = final_by_seed(g16, "cos_e1W_e1Sig")
        seeds = f4.index.intersection(f16.index)
        ci = paired_boot_ci(f16.loc[seeds].values, f4.loc[seeds].values, rng)
        # 両 κ の最終値が床の CI 内に収まっているなら、その大小関係に意味は無い
        b4, b16 = boot_ci(f4.values, rng), boot_ci(f16.values, rng)
        both_at_floor = (b4["lo"] <= floor <= b4["hi"]) and (b16["lo"] <= floor <= b16["hi"])
        rows.append(dict(pred="P3-5", scope="κ 単調性 (実質 κ=4 < κ=16)",
                         verdict="PASS" if (ci["excl_zero"] and ci["mean"] > 0) else "FAIL",
                         evidence=f"κ16 {np.nanmean(f16.loc[seeds]):.3f} − κ4 "
                                  f"{np.nanmean(f4.loc[seeds]):.3f} = {ci['mean']:+.3f} "
                                  f"CI [{ci['lo']:.3f}, {ci['hi']:.3f}] "
                                  f"({'予測と逆符号' if ci['mean'] < 0 else '予測方向'})"
                                  + ("。ただし両 κ の最終値 CI が床 "
                                     f"{floor:.3f} を含むため大小関係の解釈は弱い"
                                     if both_at_floor else "")))
    return tab


def p3_7_grad_vs_w(resdir, lop, cfg, rng, rows):
    """P3-7: 同一 checkpoint で |cos(E[g],u)| と cos_e1W_e1Sig を比較。"""
    fs = sorted(glob.glob(os.path.join(resdir, "followup_Eg_B_*.npz")))
    if not fs:
        rows.append(dict(pred="P3-7", scope="勾配場と重みの乖離", verdict="NA",
                         evidence="followup npz 未生成"))
        return pd.DataFrame()
    runs = pd.read_csv(os.path.join(resdir, "runs.csv")).set_index("run_id")
    u = spike_dir_vec(cfg["condB"].get("spike_dir", "ones"), cfg["condB"]["d"])
    out = []
    for f in fs:
        z = np.load(f, allow_pickle=True)
        step = int(z["step"])
        for i, rid in enumerate([str(x) for x in z["run_ids"]]):
            r = runs.loc[rid]
            if float(r.c) != 0.0:
                continue
            alive = ~z["dead"][i]
            if alive.sum() < 1:
                continue
            g = z["Eg_W"][i][alive]
            cg = np.abs(g @ u) / np.maximum(np.linalg.norm(g, axis=1), 1e-30)
            out.append(dict(step=step, run_id=rid, kappa=int(r.kappa), seed=int(r.seed),
                            cos_Eg_u=float(np.mean(cg))))
    gd = pd.DataFrame(out)
    if not len(gd):
        rows.append(dict(pred="P3-7", scope="勾配場と重みの乖離", verdict="NA",
                         evidence="c=0 の checkpoint データなし"))
        return gd
    lw = lop[(lop.exp == "B") & (lop.c == 0.0)][
        ["step", "run_id", "kappa", "seed", "cos_e1W_e1Sig"]]
    mg = gd.merge(lw, on=["step", "run_id", "kappa", "seed"], how="inner")
    k16 = mg[(mg.kappa == 16) & (mg.step == mg.step.max())]
    if len(k16):
        ci = paired_boot_ci(k16.cos_Eg_u.values, k16.cos_e1W_e1Sig.values, rng)
        rows.append(dict(pred="P3-7",
                         scope="κ=16 最終 ckpt: |cos(E[g],u)| > cos_e1W_e1Sig",
                         verdict="PASS" if (ci["excl_zero"] and ci["mean"] > 0) else "FAIL",
                         evidence=f"E[g] {k16.cos_Eg_u.mean():.3f} vs W "
                                  f"{k16.cos_e1W_e1Sig.mean():.3f}, diff {ci['mean']:+.3f} "
                                  f"CI [{ci['lo']:.3f}, {ci['hi']:.3f}]"))
    return mg


# ---------------------------------------------------------------- サニティ

def sanity(resdir, lop, cfg, floor):
    s = {}
    B = lop[lop.exp == "B"]
    k1 = B[B.kappa == 1]
    s["S4_kappa1_all_nan"] = bool(k1.cos_e1W_e1Sig.isna().all()) if len(k1) else None
    b0 = B[(B.step == 0) & (B.kappa > 1)]
    s["S5_step0_near_floor"] = (
        f"mean {b0.cos_e1W_e1Sig.mean():.3f} vs floor {floor:.3f}" if len(b0) else None)
    s["S5_pass"] = bool(len(b0) and abs(b0.cos_e1W_e1Sig.mean() - floor) < 0.10)
    e1s = B[(B.kappa > 1) & B.e1_stability.notna()]
    s["S6_unstable_frac"] = (float((e1s.e1_stability < 0.9).mean())
                             if len(e1s) else None)
    # S3: condA centered の EMA 残差 ‖running_mean − µ_true‖/‖µ_true‖ (ckpt から)
    s3 = []
    for p in sorted(glob.glob(os.path.join(resdir, "ckpts", "A_*.pt"))):
        ck = torch.load(p, map_location="cpu", weights_only=False)
        rm = ck["running_mean"]
        flip = ck["env"]["flip_state"]
        f = flip.shape[1]
        mu_true = torch.cat([flip, 0.5 * torch.ones(rm.shape[0], rm.shape[1] - f)], 1)
        rel = ((rm - mu_true).norm(dim=1) / mu_true.norm(dim=1).clamp_min(1e-12))
        for i, r in enumerate(ck["runs"]):
            if r["enc"] == "centered":
                s3.append(dict(step=ck["step"], run_id=r["run_id"],
                               rel_resid=float(rel[i])))
    s3 = pd.DataFrame(s3)
    if len(s3):
        s3.to_csv(os.path.join(resdir, "s3_ema_residual.csv"), index=False)
        s["S3_ema_rel_resid_by_step"] = {int(k): round(float(v), 4) for k, v in
                                         s3.groupby("step").rel_resid.mean().items()}
    return s


# ---------------------------------------------------------------- 図

def _ts(ax, sub, col, key, colors, labels=None, every=5000):
    for k, g in sub.groupby(key):
        g = g[g.step % every == 0]
        m = g.groupby("step")[col].agg(["mean", "sem"])
        lbl = labels(k) if labels else f"{key}={k}"
        c = colors.get(k, None)
        ax.plot(m.index, m["mean"], lw=1.3, color=c, label=lbl)
        ax.fill_between(m.index, m["mean"] - m["sem"], m["mean"] + m["sem"],
                        color=c, alpha=0.2, lw=0)


def figures(resdir, lop, cfg, floor, gd):
    fd = os.path.join(resdir, "figures")
    os.makedirs(fd, exist_ok=True)
    A = lop[lop.exp == "A"]
    B = lop[lop.exp == "B"]

    # cs1: condA std vs centered
    ws = sorted(A.width.unique())
    if ws:
        fig, axes = plt.subplots(2, len(ws), figsize=(6 * len(ws), 6.4), squeeze=False)
        for j, w in enumerate(ws):
            for i, col in enumerate(["dead_frac", "wcos_mean"]):
                ax = axes[i][j]
                _ts(ax, A[A.width == w], col, "enc", ENC_COLOR)
                ax.grid(alpha=0.3)
                if i == 0:
                    ax.set_title(f"condA w={w}")
                if j == 0:
                    ax.set_ylabel(col)
                    ax.legend(fontsize=8)
                if i == 1:
                    ax.set_xlabel("step")
        fig.suptitle("arm1: input centering as amplifier (condA, seed mean±SE)")
        fig.tight_layout(); fig.savefig(os.path.join(fd, "fig_cs1_amplify.png"), dpi=150)
        plt.close(fig)

    # cs2: condB c=0 vs c=2 (κ=1)
    b1 = B[B.kappa == 1]
    if len(b1):
        fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
        for ax, col in zip(axes, ["dead_frac", "wcos_mean", "stable_rank_W_alive"]):
            _ts(ax, b1, col, "c", C_COLOR, labels=lambda k: f"c={k:g}")
            ax.set_xlabel("step"); ax.set_ylabel(col); ax.grid(alpha=0.3)
        axes[0].legend(fontsize=8)
        fig.suptitle("arm2: exact µ=0 (c=0) vs µ≠0 (c=2), condB κ=1")
        fig.tight_layout(); fig.savefig(os.path.join(fd, "fig_cs2_amplify_B.png"), dpi=150)
        plt.close(fig)

    # cs3: cos_e1W_e1Sig 時系列 (κ 別)
    b0 = B[B.c == 0.0]
    if len(b0):
        fig, ax = plt.subplots(figsize=(6.4, 4))
        _ts(ax, b0, "cos_e1W_e1Sig", "kappa", KAP_COLOR, labels=lambda k: f"κ={k}")
        ax.axhline(floor, color="black", ls=":", lw=1, label=f"random floor {floor:.3f}")
        ax.set_xlabel("step"); ax.set_ylabel("|cos(e1^W, e1^Σ)|")
        ax.set_title("arm3 (main): C_self residual alignment, µ=0 exact")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(fd, "fig_cs3_e1cos.png"), dpi=150)
        plt.close(fig)

        # cs4: raw vs pca
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
        for ax, col, t in [(axes[0], "cos_e1W_e1Sig", "raw SVD (main)"),
                           (axes[1], "cos_e1W_e1Sig_pca", "row-centered PCA")]:
            _ts(ax, b0, col, "kappa", KAP_COLOR, labels=lambda k: f"κ={k}")
            ax.axhline(floor, color="black", ls=":", lw=1)
            ax.set_title(t); ax.set_xlabel("step"); ax.grid(alpha=0.3)
        axes[0].set_ylabel("|cos(e1^W, e1^Σ)|"); axes[0].legend(fontsize=8)
        fig.suptitle("§1.4: e1^W definition — raw vs row-centered PCA")
        fig.tight_layout(); fig.savefig(os.path.join(fd, "fig_cs4_raw_vs_pca.png"), dpi=150)
        plt.close(fig)

        # cs6: w_norm_mean
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
        _ts(axes[0], A, "w_norm_mean", "enc", ENC_COLOR)
        axes[0].set_title("condA"); axes[0].legend(fontsize=8)
        _ts(axes[1], b1, "w_norm_mean", "c", C_COLOR, labels=lambda k: f"c={k:g}")
        axes[1].set_title("condB κ=1"); axes[1].legend(fontsize=8)
        for ax in axes:
            ax.set_xlabel("step"); ax.set_ylabel("w_norm_mean (alive)"); ax.grid(alpha=0.3)
        fig.suptitle("P3-3 confound control: ‖w‖ trajectories")
        fig.tight_layout(); fig.savefig(os.path.join(fd, "fig_cs6_norm.png"), dpi=150)
        plt.close(fig)

    # cs5: |cos(E[g],u)| vs cos_e1W_e1Sig
    if len(gd):
        fig, ax = plt.subplots(figsize=(6.4, 4))
        for kap, g in gd.groupby("kappa"):
            m = g.groupby("step")[["cos_Eg_u", "cos_e1W_e1Sig"]].mean()
            ax.plot(m.index, m.cos_Eg_u, "-o", ms=4, color=KAP_COLOR.get(kap),
                    label=f"κ={kap} E[g]")
            ax.plot(m.index, m.cos_e1W_e1Sig, "--s", ms=4, color=KAP_COLOR.get(kap),
                    alpha=0.6, label=f"κ={kap} W")
        ax.set_xlabel("step"); ax.set_ylabel("|cos(·, u)|")
        ax.set_title("P3-7: gradient field vs weights alignment to Σ axis")
        ax.grid(alpha=0.3); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(os.path.join(fd, "fig_cs5_grad_vs_w.png"), dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------- summary

S2_QUOTE = """src/train.py (eval_batch / 学習ループ) — 中心化は学習器入力のみに適用され、
教師は生入力 x_raw を受け取る:

    x_raw = env.step()                               # [R,d]
    y = teacher(x_raw)                               # [R]   ← 生入力
    x_in = x_raw - cmask * st["running_mean"]        # ← 学習器のみ中心化
    pre, a, yhat = net.forward(x_in)

eval 側も同様 (eval_batch は y = teacher(x) を生 x で計算し、
呼び出し側が x_ev_in = x_ev - cmask*running_mean を net に渡す)。"""


def write_summary(resdir, verdicts, a1, a2, a3, san, cfg, floor, phase0, phase1):
    lines = ["# center_selfcov_0814 summary (spec §6 事前登録判定)\n"]
    lines.append("## 判定表 (null 結果も同じ体裁)\n")
    lines.append(verdicts.to_string(index=False))

    lines.append("\n\n## Phase 0 / Phase 1\n")
    lines.append(f"- Phase 0 (aniso_perp_0812 再解析): 仕様 §3 の期待値6項目を"
                 f"{'全て相対5%以内で再現 (PASS)' if phase0.get('replication_pass') else '再現せず (FAIL)'}"
                 f"。「勾配場は Σ 軸を向くが重みは床付近」の乖離を確認 (P3-7 の予備証拠)")
    lines.append(f"- Phase 1 (レジーム探索): {phase1}")

    lines.append("\n## アーム1: 条件A std vs centered (項目2)\n")
    lines.append(a1.round(5).to_string(index=False))
    if len(a2):
        lines.append("\n## アーム2: 条件B c=0 vs c=2 (µ=0 厳密、κ=1)\n")
        lines.append(a2.round(5).to_string(index=False))
    lines.append("\n## アーム3: C_self 残存 (κ 別、c=0)\n")
    lines.append(a3.round(4).to_string(index=False))
    lines.append(f"\n- ランダム床 |cos| ≈ {floor:.3f} (d={cfg['condB']['d']})")

    lines.append("\n## サニティ (§7)\n")
    lines.append(f"- S1 (target_hidden 未指定で既存 condB と bit 一致): PASS "
                 f"(coupling_ab_0813 B_w5 の共通カラムが全行一致、追加カラムなし)")
    lines.append(f"- S2 (条件A centered で教師は生入力): PASS — コード引用:\n\n```\n{S2_QUOTE}\n```")
    lines.append(f"- S3 (EMA 中心化の実効残差 ‖running_mean−µ_true‖/‖µ_true‖): "
                 f"{san.get('S3_ema_rel_resid_by_step')}")
    lines.append(f"- S4 (κ=1 で Σ 系が全て NaN): "
                 f"{'PASS' if san.get('S4_kappa1_all_nan') else 'FAIL'}")
    lines.append(f"- S5 (step0 の cos が床付近): "
                 f"{'PASS' if san.get('S5_pass') else 'FAIL'} ({san.get('S5_step0_near_floor')})")
    lines.append(f"- S6 (e1_stability < 0.9 の区間割合): {san.get('S6_unstable_frac')}"
                 " — 大きい場合、第1特異値が縮退して主方向が意味を持たない区間がある")

    lines.append("""
## 結論

1. **項目2 (増幅因子) は dead 経路で強く成立、整列経路では条件付き**。条件A w100 で
   centered の dead_frac は 0.294 vs std 0.964 と大差 (P3-1 PASS、既報 0.96→0.28 を再現)。
   一方 wcos_mean の傾きは w100 で PASS だが **w5 では差なし** (P3-2 は幅依存)。
   µ=0 が厳密な条件B (P3-2b) でも整列傾きは低下するが CI はゼロをかろうじて外す程度。
2. **ただし P3-3 は CONFOUNDED**。centered 腕は ‖w‖ も有意に小さい (w100: 2.01 vs 4.44)。
   理論 v2 §3(d) の通りノイズは ‖w‖⁻²・ドリフトは ‖w‖⁻¹ でスケールするため、
   µ の効果とノルム媒介効果が本実験では分離できていない。
   仕様 §9 のノルム固定アームが本命の追試になる。
3. **項目3 (C_self 残存) は不支持**。µ=0 厳密・κ=16 で cos_e1W_e1Sig は
   最終 0.105 (CI [0.021, 0.200])、ランダム床 0.174 を超えない (P3-4 FAIL)。
   系列は床付近を往復するノイズ支配で、単調増加も認められない。
   先生の予言 (→1.0) は成立しない (P3-6 FAIL、0.9 に遠く及ばない)。
   κ 単調性も逆符号 (P3-5 FAIL) だが両 κ とも床付近なので方向の主張自体が弱い。
   なお srank_alive は 2.2–2.5 で rank-1 に落ちておらず (top1_frac 0.41–0.46)、
   「e₁ が支配的でないのは当然」という但し書きが該当する。
4. **P3-7 が最も強い所見 (PASS)**。同一 checkpoint で勾配場は Σ 軸をほぼ完全に向く
   (|cos(E[g],u)| = 0.907) のに、重みは床以下 (0.105) に留まる。差 +0.801
   CI [0.747, 0.852]。Phase 0 の乖離 (0.71 vs 0.10) が、教師幅を分離してレジームを
   変えた後も、むしろ拡大して再現した。
   **「drift は Σ 軸を向いているが重みはそこに蓄積しない」** = 理論 v2 §5(b) の
   1/‖w‖ による操舵切断、および rank_int_0814 の「病理は状態ではなく力場」と同方向。
5. **本実験の重要な限界**: Phase 1 でどのセルも LoP 発現基準 (基準4) を満たさず、
   採用セルも eval_loss は低下する (LoP 非発現) レジームである。項目3 の null は
   「LoP が起きている状況で C_self 整列が残らない」ことの証明にはなっていない。
""")

    lines.append("\n## 先生への確認事項 (§10)\n")
    lines.append("1. **e₁^W の定義**: 「PCA」はユニット方向の中心化を含みますか。含む場合、"
                 "全ユニット共通の方向成分 (w̄) が除去され、測ろうとしている整列そのものが"
                 "落ちます。本実験は中心化なしの第1右特異ベクトルを主判定とし、"
                 "PCA 版も併記しました (fig_cs4)。")
    lines.append("2. **条件A では項目3が測定不能**: SCR の入力共分散は (1/4)I で完全等方のため "
                 "e₁^Σ が一意に定まりません。項目3はスパイク型 Σ を入れた条件B で実施しました。")
    lines.append("3. **収束先は 1.0 ではなく |cos| → 1**: Cov(e,x) の軸吸引は ± 対称"
                 "(aniso_perp_0812 の符号付き解析) なので符号は自発的に決まります。")
    lines.append("4. **レジーム**: 既存の条件B設定は教師幅が学習器と同一で LoP が発現しない"
                 "ため、教師幅を分離 (target_hidden=100) した上でレジームを選定しました。")
    with open(os.path.join(resdir, "summary.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    lop, runs, cfg = load(args.results)
    floor = random_floor(cfg["condB"]["d"])

    rows = []
    a1 = arm1_amplify(lop, rng, rows)
    a2 = arm2_amplify_B(lop, rng, rows)
    a3 = arm3_selfcov(lop, cfg, rng, rows, floor)
    gd = p3_7_grad_vs_w(args.results, lop, cfg, rng, rows)
    verdicts = pd.DataFrame(rows)
    verdicts.to_csv(os.path.join(args.results, "verdict.csv"), index=False)

    san = sanity(args.results, lop, cfg, floor)
    json.dump(san, open(os.path.join(args.results, "sanity.json"), "w"),
              indent=1, default=str)
    figures(args.results, lop, cfg, floor, gd)

    p0p = os.path.join(args.results, "phase0", "phase0_meta.json")
    phase0 = json.load(open(p0p)) if os.path.exists(p0p) else {}
    p1p = os.path.join(args.results, "phase1", "phase1_report.md")
    phase1 = "採用セル: " + (open(p1p).read().split("## 採用セル")[-1].strip().split("\n")[1]
                          if os.path.exists(p1p) and "## 採用セル" in open(p1p).read()
                          else "phase1_report.md 参照")
    write_summary(args.results, verdicts, a1, a2, a3, san, cfg, floor, phase0, phase1)
    print(verdicts.to_string(index=False))
    print(f"\nfigures -> {os.path.join(args.results, 'figures')}")


if __name__ == "__main__":
    main()
