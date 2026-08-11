"""Part A 解析 (task_074): followup.py が出力した npz から A1/A2/A4/A5/A6 を算出。

  python -m src.followup_analysis results/drift_0809

出力 (すべて対象ディレクトリ <R> = results/drift_0809 の中):
  <R>/followup_A1_pairwise.csv   ニューロン間 pairwise cos (run×ckpt 集約)
  <R>/followup_A1_hist.csv       pairwise cos ヒストグラム (条件×ckpt, seed 合算)
  <R>/followup_A2_deadcross.csv  cos 符号 × 生死 の分割表 (条件別)
  <R>/followup_A4_ckptcos.csv    連続 ckpt 間の E[g] 方向相関
  <R>/followup_A5_period.csv     周期内平均勾配 ḡ_τ の方向分散
  <R>/followup_A6_splithalf.csv  split-half による cos アーティファクト検定
  <R>/figures/fig_a1_*.png, fig_a2_*.png, fig_a4_*.png, fig_a5_*.png, fig_a6_*.png
"""
import argparse
import glob
import json
import os
import re
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import ROOT

EPS = 1e-12
COND_COLS = ["exp", "width", "period", "enc", "c", "lr"]
NBINS = 40
BIN_EDGES = np.linspace(-1, 1, NBINS + 1)
BIN_CENTERS = 0.5 * (BIN_EDGES[1:] + BIN_EDGES[:-1])


def git_hash():
    try:
        return subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "N/A"


def unitize(a, axis):
    return a / np.maximum(np.linalg.norm(a, axis=axis, keepdims=True), EPS)


def load_npz(resdir):
    """{(exp, width, step): npz-dict} を返す。

    発散系列 (パラメータ NaN/Inf) は finite=False で記録済み。数値配列は 0 に潰して
    線形代数を通し、集計側では finite==False の行を NaN 扱いにする。
    """
    out = {}
    for p in sorted(glob.glob(os.path.join(resdir, "followup_Eg_*.npz"))):
        m = re.match(r"followup_Eg_([AB])_w(\d+)_step(\d+)\.npz", os.path.basename(p))
        exp, width, step = m.group(1), int(m.group(2)), int(m.group(3))
        z = dict(np.load(p, allow_pickle=True))
        for k, v in z.items():
            if isinstance(v, np.ndarray) and v.dtype.kind == "f":
                z[k] = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        out[(exp, width, step)] = z
    return out


NANCOLS = {}


def mark_nonfinite(df, cols):
    """finite==False の行の指標列を NaN にする。"""
    bad = ~df["finite"].astype(bool)
    df.loc[bad, [c for c in cols if c in df.columns]] = np.nan
    return df


def cond_key(runs, rid):
    r = runs.loc[rid]
    return tuple(r[c] for c in COND_COLS)


# --------------------------------------------------------------- A1 pairwise cos

def a1_pairwise(data, runs):
    """各 run×ckpt のニューロン間 pairwise cos 統計 + ヒストグラム。

    3 通りの見方を出す:
      pcos_*        生の符号付き cos(E[g_i], E[g_j])
      vpcos_*       sign(v_i)sign(v_j) を掛けた「出力符号補正つき」cos
                    (g_{W_i} ∝ 2δ v_i 1[pre_i>0] x なので、v の符号だけで向きは反転する。
                     符号クローンの整列を見たいならこちらが素直)
      eig1_frac     単位化 E[g_i] の Gram 行列の最大固有値 / h。
                    1 に近い = 全ニューロンの E[g] が (符号を除いて) 1 本の直線上に乗る
    dead ニューロンは E[g]≈0 で向きが数値雑音になるため *_alive も併記する。
    """
    rows, hist, hist_v = [], [], []
    for (exp, width, step), z in sorted(data.items()):
        Eg, ids, dead, sv = z["Eg_W"], z["run_ids"], z["dead"], np.sign(z["v"])
        Eo, Ee = z["Eg_W_odd"], z["Eg_W_even"]
        U, Uo, Ue = unitize(Eg, 2), unitize(Eo, 2), unitize(Ee, 2)
        # µ̂ 成分を抜いた残差の整列 (µ が整列の原因かを切り分ける対照)
        muh = unitize(z["mu_inter"], 1)
        Up = unitize(Eg - np.einsum("rhd,rd->rh", Eg, muh)[..., None] * muh[:, None, :], 2)
        iu = np.triu_indices(width, k=1)

        def stats(Um, s=None):
            V = Um if s is None else Um * s[:, None]
            G = V @ V.T
            ev = np.linalg.eigvalsh(G)[-1] / max(len(V), 1)
            return G[np.triu_indices(len(V), k=1)], float(ev)

        for i, rid in enumerate(ids):
            pc, e1 = stats(U[i])
            vpc, _ = stats(U[i], sv[i])
            Gx = Uo[i] @ Ue[i].T                     # 共通サンプル雑音を消した cross-half 版
            pcx = 0.5 * (Gx + Gx.T)[iu]
            vGx = (sv[i][:, None] * sv[i][None, :]) * Gx
            vpcx = 0.5 * (vGx + vGx.T)[iu]
            alive = ~dead[i]
            if alive.sum() >= 2:
                pca, e1a = stats(U[i][alive])
                vpca, _ = stats(U[i][alive], sv[i][alive])
                vpcp, e1p = stats(Up[i][alive], sv[i][alive])
            else:
                pca, vpca, vpcp = (np.array([np.nan]),) * 3
                e1a = e1p = np.nan
            rows.append(dict(run_id=str(rid), ckpt=step, n_pairs=len(pc),
                             finite=bool(z["finite"][i]),
                             pcos_mean=float(pc.mean()), pcos_median=float(np.median(pc)),
                             pcos_absmean=float(np.abs(pc).mean()),
                             pcos_frac_pos5=float((pc > 0.5).mean()),
                             pcos_frac_neg5=float((pc < -0.5).mean()),
                             pcos_split_mean=float(pcx.mean()),
                             vpcos_mean=float(vpc.mean()),
                             vpcos_split_mean=float(vpcx.mean()),
                             vpcos_frac_pos5=float((vpc > 0.5).mean()),
                             eig1_frac=e1,
                             pcos_alive_mean=float(np.nanmean(pca)),
                             vpcos_alive_mean=float(np.nanmean(vpca)),
                             vpcos_perp_alive_mean=float(np.nanmean(vpcp)),
                             eig1_frac_alive=float(e1a), eig1_frac_perp_alive=float(e1p),
                             n_alive=int(alive.sum())))
            # ヒストグラムは alive ニューロンのみ (dead は E[g]≈0 で cos が 0 に張り付く)
            hist.append((str(rid), step, np.histogram(pca, bins=BIN_EDGES)[0]))
            hist_v.append((str(rid), step, np.histogram(vpca, bins=BIN_EDGES)[0]))
    df = pd.DataFrame(rows).join(runs, on="run_id")

    def agg_hist(hs, tag):
        hdf = pd.DataFrame([dict(run_id=r, ckpt=s, kind=tag,
                                 **{f"b{k}": v for k, v in enumerate(h)})
                            for r, s, h in hs]).join(runs, on="run_id")
        return hdf.groupby(COND_COLS + ["ckpt", "kind"], dropna=False)[
            [f"b{k}" for k in range(NBINS)]].sum().reset_index()

    agg = pd.concat([agg_hist(hist, "raw"), agg_hist(hist_v, "vsigned")], ignore_index=True)
    return df, agg


# ------------------------------------------------- A2 cos 符号 × 最終的な生死

def a2_dead_cross(data, runs, sign_step=10000, dead_step=1000000):
    rows = []
    for (exp, width, s0), z0 in sorted(data.items()):
        if s0 != sign_step:
            continue
        z1 = data.get((exp, width, dead_step))
        if z1 is None:
            continue
        mu = unitize(z0["mu_inter"], 1)
        cos0 = np.einsum("rhd,rd->rh", unitize(z0["Eg_W"], 2), mu)     # [R,h]
        sgn = np.sign(cos0)
        dead1, sv = z1["dead"], np.sign(z1["v"])
        assert list(z0["run_ids"]) == list(z1["run_ids"])
        for i, rid in enumerate(z0["run_ids"]):
            for sp, lab in [(sgn[i] > 0, "pos"), (sgn[i] < 0, "neg")]:
                rows.append(dict(run_id=str(rid), cos_sign=lab, n=int(sp.sum()),
                                 finite=bool(z0["finite"][i] and z1["finite"][i]),
                                 n_dead=int((sp & dead1[i]).sum()),
                                 n_v_pos=int((sp & (sv[i] > 0)).sum()),
                                 n_dead_v_pos=int((sp & dead1[i] & (sv[i] > 0)).sum()),
                                 sign_step=sign_step, dead_step=dead_step))
    # smoke 実行など sign_step/dead_step の ckpt が無い場合は空表 (列だけ揃える)
    df = pd.DataFrame(rows, columns=["run_id", "cos_sign", "n", "finite", "n_dead",
                                     "n_v_pos", "n_dead_v_pos", "sign_step", "dead_step"])
    if not rows:
        print(f"  [A2] ckpt {sign_step:g}/{dead_step:g} が無いのでスキップ", flush=True)
    df = df.join(runs, on="run_id")
    df["dead_rate"] = df.n_dead / df.n.clip(lower=1)
    return df


# -------------------------------------------- A4 ckpt 間の E[g] 方向相関

def a4_ckpt_cos(data, runs):
    rows = []
    groups = sorted({(e, w) for e, w, _ in data})
    for exp, width in groups:
        steps = sorted(s for e, w, s in data if (e, w) == (exp, width))
        for s0, s1 in zip(steps[:-1], steps[1:]):
            z0, z1 = data[(exp, width, s0)], data[(exp, width, s1)]
            A0 = z0["Eg_W"].reshape(len(z0["run_ids"]), -1)
            A1 = z1["Eg_W"].reshape(len(z1["run_ids"]), -1)
            c_all = (unitize(A0, 1) * unitize(A1, 1)).sum(1)                    # [R]
            c_null = (unitize(A0, 1) * unitize(np.roll(A1, 1, axis=0), 1)).sum(1)
            U0, U1 = unitize(z0["Eg_W"], 2), unitize(z1["Eg_W"], 2)
            c_i = (U0 * U1).sum(2)                                             # [R,h]
            for i, rid in enumerate(z0["run_ids"]):
                rows.append(dict(run_id=str(rid), ckpt_from=s0, ckpt_to=s1,
                                 finite=bool(z0["finite"][i] and z1["finite"][i]),
                                 cos_all=float(c_all[i]), cos_null=float(c_null[i]),
                                 cos_neuron_mean=float(c_i[i].mean()),
                                 cos_neuron_absmean=float(np.abs(c_i[i]).mean())))
    return pd.DataFrame(rows).join(runs, on="run_id")


# ------------------------------------ A5 周期内平均勾配 ḡ_τ の方向分散

def a5_period(data, runs):
    rows = []
    for (exp, width, step), z in sorted(data.items()):
        gn, cnt, per = z["gbar_norm"], z["gbar_count"], z["period"]
        Eg = z["Eg_W"].reshape(len(z["run_ids"]), -1)
        En = np.linalg.norm(Eg, axis=1)
        for i, rid in enumerate(z["run_ids"]):
            ok = cnt[i] >= 0.5 * per[i]                 # ほぼ完全な周期ビンのみ
            n_p = int(ok.sum())
            if n_p == 0:
                continue
            mg = float(gn[i][ok].mean())
            rows.append(dict(run_id=str(rid), ckpt=step, n_periods=n_p,
                             finite=bool(z["finite"][i]),
                             Eg_norm=float(En[i]), gbar_norm_mean=mg,
                             ratio=float(En[i] / max(mg, EPS)),
                             ratio_pred=float(1.0 / np.sqrt(n_p))))
    df = pd.DataFrame(rows).join(runs, on="run_id")
    df["ratio_over_pred"] = df.ratio / df.ratio_pred
    return df


# ------------------------------------------------- A6 split-half cos 検定

def a6_splithalf(data, runs):
    rows = []
    for (exp, width, step), z in sorted(data.items()):
        mu, mo, me = unitize(z["mu_inter"], 1), unitize(z["mu_odd"], 1), unitize(z["mu_even"], 1)
        U, Uo, Ue = unitize(z["Eg_W"], 2), unitize(z["Eg_W_odd"], 2), unitize(z["Eg_W_even"], 2)
        c_full = np.einsum("rhd,rd->rh", U, mu)
        c_x = 0.5 * (np.einsum("rhd,rd->rh", Uo, me) + np.einsum("rhd,rd->rh", Ue, mo))
        # ランダム方向の |cos| 期待値 (d 次元): 「向いていない」の基準線
        chance = float(np.sqrt(2.0 / (np.pi * z["mu_inter"].shape[1])))
        for i, rid in enumerate(z["run_ids"]):
            rows.append(dict(run_id=str(rid), ckpt=step, finite=bool(z["finite"][i]),
                             cos_full_abs=float(np.abs(c_full[i]).mean()),
                             cos_split_abs=float(np.abs(c_x[i]).mean()),
                             cos_full_mean=float(c_full[i].mean()),
                             cos_split_mean=float(c_x[i].mean()),
                             cos_chance=chance))
    return pd.DataFrame(rows).join(runs, on="run_id")


# ----------------------------------------------------------------------- 図

def _condA(df, width, T=10000, lr=0.01):
    return df[(df.exp == "A") & (df.width == width) & (df.period == T) & (df.lr == lr)]


def _condB(df, width, K=10000):
    return df[(df.exp == "B") & (df.width == width) & (df.period == K)]


COLS = {"std": "tab:red", "centered": "tab:blue", 2.0: "tab:red", 0.0: "tab:blue"}
SPECS = [("A", "enc", ["std", "centered"]), ("B", "c", [2.0, 0.0])]


def _pick(df, key, v):
    return df[df[key] == v] if key == "enc" else df[np.isclose(df[key].astype(float), v)]


def fig_a1(a1, hist, figdir, final=1000000, width=100):
    """ヒストグラム (raw / v 符号補正) と ckpt 推移。"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for row, kind in enumerate(["raw", "vsigned"]):
        for col, (exp, key, vals) in enumerate(SPECS):
            ax = axes[row][col]
            h = hist[(hist.exp == exp) & (hist.width == width) & (hist.period == 10000)
                     & (hist.ckpt == final) & (hist.kind == kind)]
            m0 = a1[(a1.exp == exp) & (a1.width == width) & (a1.period == 10000)
                    & (a1.ckpt == final)]
            for v in vals:
                s = _pick(h, key, v)
                if s.empty:
                    continue
                cnt = s[[f"b{k}" for k in range(NBINS)]].sum().to_numpy(dtype=float)
                mv = _pick(m0, key, v)
                mu_ = (mv.pcos_alive_mean.mean() if kind == "raw"
                       else mv.vpcos_alive_mean.mean())
                ax.step(BIN_CENTERS, cnt / max(cnt.sum(), 1), where="mid", color=COLS[v],
                        label=f"{key}={v} (mean={mu_:.2f})")
            ax.axvline(0, color="k", lw=0.8, alpha=0.5)
            lab = "cos(E[g_i], E[g_j])" if kind == "raw" else "s_i s_j cos(E[g_i], E[g_j])"
            ax.set_title(f"cond {exp}, w={width}, period=10^4, step={final:g} "
                         f"[{kind}, alive neurons only]", fontsize=10)
            ax.set_xlabel(lab); ax.set_ylabel("fraction of pairs")
            ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle("A1: neuron-to-neuron alignment of E[g]  (s_i = sign(v_i))")
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "fig_a1_pairwise_hist.png"), dpi=140)
    plt.close(fig)

    metrics = [("pcos_mean", "mean pairwise cos (raw)"),
               ("vpcos_alive_mean", "mean v-signed pairwise cos (alive)"),
               ("vpcos_perp_alive_mean", "same, with µ̂ projected out"),
               ("eig1_frac_alive", "λ₁(Gram)/h  (collinearity, alive)")]
    fig, axes = plt.subplots(len(metrics), 2, figsize=(12, 10), sharex=True)
    for row, (mcol, ylab) in enumerate(metrics):
        for col, (exp, key, vals) in enumerate(SPECS):
            ax = axes[row][col]
            for w, ls in [(5, "--"), (100, "-")]:
                for v in vals:
                    s = _condA(a1, w) if exp == "A" else _condB(a1, w)
                    s = _pick(s, key, v)
                    if s.empty:
                        continue
                    g = s.groupby("ckpt")[mcol]
                    ax.errorbar(np.clip(g.mean().index, 1, None), g.mean(), yerr=g.std(),
                                ls=ls, marker="o", ms=4, color=COLS[v],
                                label=f"w={w}, {key}={v}")
            ax.axhline(0, color="k", lw=0.8); ax.set_xscale("log"); ax.grid(alpha=0.3)
            if row == 0:
                ax.set_title(f"cond {exp} (period=10^4)")
            if row == len(metrics) - 1:
                ax.set_xlabel("checkpoint step")
            if col == 0:
                ax.set_ylabel(ylab, fontsize=9)
            ax.legend(fontsize=7)
    fig.suptitle("A1: alignment vs training step")
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "fig_a1_pairwise_vs_step.png"), dpi=140)
    plt.close(fig)


def fig_a2(a2, figdir):
    sub = a2[(a2.exp == "A")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, width in zip(axes, [5, 100]):
        s = sub[sub.width == width]
        keys, pos, neg = [], [], []
        for (per, enc), g in s.groupby(["period", "enc"]):
            keys.append(f"T={per:g}\n{enc}")
            for lab, arr in [("pos", pos), ("neg", neg)]:
                gg = g[g.cos_sign == lab]
                arr.append(gg.n_dead.sum() / max(gg.n.sum(), 1))
        x = np.arange(len(keys))
        ax.bar(x - 0.2, pos, 0.4, label="sign(cos)>0  (+µ side)", color="tab:red")
        ax.bar(x + 0.2, neg, 0.4, label="sign(cos)<0  (−µ side)", color="tab:blue")
        ax.set_xticks(x); ax.set_xticklabels(keys, fontsize=7)
        ax.set_title(f"cond A, width={width}"); ax.grid(alpha=0.3, axis="y")
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("dead fraction at 1e6 steps")
    axes[0].legend(fontsize=8)
    fig.suptitle("A2: sign of cos(E[g_i], µ̂) at step 1e4  ->  death by step 1e6")
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "fig_a2_dead_cross.png"), dpi=140)
    plt.close(fig)


def fig_a4(a4, figdir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, (exp, key, vals, cols) in zip(axes, [
            ("A", "enc", ["std", "centered"], {"std": "tab:red", "centered": "tab:blue"}),
            ("B", "c", [2.0, 0.0], {2.0: "tab:red", 0.0: "tab:blue"})]):
        for v in vals:
            s = a4[(a4.exp == exp) & (a4.period == 10000) & (a4.lr == 0.01)]
            s = s[s[key] == v] if key == "enc" else s[np.isclose(s[key].astype(float), v)]
            if s.empty:
                continue
            g = s.groupby("ckpt_to").cos_all
            ax.errorbar(g.mean().index, g.mean(), yerr=g.std(), marker="o", ms=4,
                        color=cols[v], label=f"{key}={v}")
            gn = s.groupby("ckpt_to").cos_null
            ax.plot(gn.mean().index, gn.mean(), ":", color=cols[v], lw=1,
                    label=f"{key}={v} (null: other run)")
        ax.axhline(0, color="k", lw=0.8); ax.set_xscale("log"); ax.grid(alpha=0.3)
        ax.set_xlabel("later checkpoint of the pair"); ax.set_title(f"cond {exp}, period=10^4")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("cos(E[g](t_k), E[g](t_k+1))")
    fig.suptitle("A4: persistence of the drift direction across checkpoints")
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "fig_a4_ckpt_cos.png"), dpi=140)
    plt.close(fig)


def fig_a5(a5, figdir):
    s = a5[a5.ckpt == a5.ckpt.max()]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for exp, mk in [("A", "o"), ("B", "s")]:
        e = s[s.exp == exp]
        ax.scatter(e.ratio_pred, e.ratio, s=22, marker=mk, alpha=0.7, label=f"cond {exp}")
    lim = [0, max(s.ratio.max(), s.ratio_pred.max()) * 1.1]
    ax.plot(lim, lim, "k--", lw=1, label="1/√n_periods (incoherent)")
    ax.set_xlabel("1/√n_periods  (prediction if ḡ_τ directions are random)")
    ax.set_ylabel("‖E[g]‖ / mean_τ ‖ḡ_τ‖")
    ax.grid(alpha=0.3); ax.legend(); ax.set_title("A5: within-period mean gradients — coherent?")
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "fig_a5_period_coherence.png"), dpi=140)
    plt.close(fig)


def fig_a6(a6, figdir):
    s = a6[a6.ckpt == a6.ckpt.max()]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, (exp, key, vals, cols) in zip(axes, [
            ("A", "enc", ["std", "centered"], {"std": "tab:red", "centered": "tab:blue"}),
            ("B", "c", [2.0, 0.0], {2.0: "tab:red", 0.0: "tab:blue"})]):
        e = s[s.exp == exp]
        for v in vals:
            q = e[e[key] == v] if key == "enc" else e[np.isclose(e[key].astype(float), v)]
            if q.empty:
                continue
            ax.scatter(q.cos_full_abs, q.cos_split_abs, s=26, color=cols[v], alpha=0.75,
                       label=f"{key}={v}")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ch = e.cos_chance.mean()
        ax.axhline(ch, color="gray", ls=":", lw=1)
        ax.axvline(ch, color="gray", ls=":", lw=1)
        ax.text(0.02, ch + 0.02, f"chance |cos| = {ch:.2f}", fontsize=7, color="gray")
        ax.set_xlabel("|cos| (same samples for µ̂ and E[g])")
        ax.set_ylabel("|cos| (split-half: µ̂ and E[g] from disjoint samples)")
        ax.set_title(f"cond {exp}, step 1e6"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.suptitle("A6: is the µ-alignment a finite-sample artifact?")
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "fig_a6_splithalf.png"), dpi=140)
    plt.close(fig)


# ----------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", help="実験ディレクトリ (例: results/drift_0809)")
    ap.add_argument("--figdir", default=None,
                    help="図の出力先 (既定: <results>/figures)")
    args = ap.parse_args()
    args.figdir = args.figdir or os.path.join(args.results, "figures")
    os.makedirs(args.figdir, exist_ok=True)

    runs = pd.read_csv(os.path.join(args.results, "runs.csv")).set_index("run_id")
    data = load_npz(args.results)
    print(f"loaded {len(data)} npz measurement files", flush=True)

    a1, hist = a1_pairwise(data, runs)
    a2 = a2_dead_cross(data, runs)
    a4 = a4_ckpt_cos(data, runs)
    a5 = a5_period(data, runs)
    a6 = a6_splithalf(data, runs)

    # 発散系列の指標を NaN 化 → 以降の集計・図から自動的に除外
    for df, cols in [(a1, [c for c in a1.columns if c.startswith(("pcos", "vpcos", "eig1"))]),
                     (a2, ["n_dead", "n_v_pos", "n_dead_v_pos", "dead_rate"]),
                     (a4, ["cos_all", "cos_null", "cos_neuron_mean", "cos_neuron_absmean"]),
                     (a5, ["Eg_norm", "gbar_norm_mean", "ratio", "ratio_over_pred"]),
                     (a6, [c for c in a6.columns if c.startswith("cos_")])]:
        mark_nonfinite(df, cols)
    n_bad = int((~a1.finite).sum())
    print(f"  non-finite (diverged) run-checkpoint rows in A1: {n_bad}/{len(a1)}")

    gh = git_hash()
    for name, df in [("A1_pairwise", a1), ("A1_hist", hist), ("A2_deadcross", a2),
                     ("A4_ckptcos", a4), ("A5_period", a5), ("A6_splithalf", a6)]:
        df = df.copy()
        df["git_hash"] = gh
        df.to_csv(os.path.join(args.results, f"followup_{name}.csv"), index=False)
        print(f"  wrote followup_{name}.csv  ({len(df)} rows)")

    fig_a1(a1, hist, args.figdir)
    fig_a2(a2, args.figdir)
    fig_a4(a4, args.figdir)
    fig_a5(a5, args.figdir)
    fig_a6(a6, args.figdir)
    json.dump(dict(git_hash=gh, n_npz=len(data),
                   npz_groups=sorted({f"{e}_w{w}" for e, w, _ in data})),
              open(os.path.join(args.results, "followup_analysis_meta.json"), "w"), indent=1)
    print("FOLLOWUP ANALYSIS DONE")


if __name__ == "__main__":
    main()
