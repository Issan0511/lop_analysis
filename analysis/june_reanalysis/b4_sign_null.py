"""B4 — 符号プリミティブ vs 整列プリミティブの判別（H-sign の検証、仕様書 §6）。

  .venv/bin/python -m analysis.june_reanalysis.b4_sign_null

前処理は Phase 0 で確認した A1 の定義に揃える:
  対象 = Ê[g_{W_i}]、alive のみ、signed、sign(v_i) 補正。
  補正は「ベクトルに先に sv_i を掛ける」形で入れる（ŵ_i = sv_i·w_i）。以降は素の cos。
比較のため obj='W'（仕様書の文面どおりの重み）も同じ手順で回す。

ヌル: 各ペアで w_j を w̃_j に置換 — sign(w_j) を座標ごとに保持し、|w_j| を座標間で
一様ランダムに並べ替える（N=1000）。
"""
import collections

import numpy as np

from . import common as C

STEPS = [100000, 1000000]
N_PERM = 1000
TWO_OVER_PI = 2.0 / np.pi


def sign_perm_null(X, n_perm=N_PERM, seed=C.SEED):
    """符号固定・絶対値置換ヌル。X: [h,d]（すでに sv 補正済み・非正規化）。

    返り値: null_mean, null_std   いずれも [h,h]（行 i = 観測側、列 j = 置換側）
    """
    h, d = X.shape
    g = np.random.default_rng(seed)
    U = C.unit(X, axis=1)
    sgn, mag = np.sign(X), np.abs(X)
    # 各置換で全ユニット j の |w_j| を座標間でシャッフル
    idx = np.argsort(g.random((n_perm, h, d)), axis=2)              # [N,h,d]
    Xt = sgn[None] * np.take_along_axis(mag[None].repeat(n_perm, 0), idx, axis=2)
    Ut = C.unit(Xt, axis=2)                                          # [N,h,d]
    G = np.einsum("id,njd->nij", U, Ut)                              # [N,h,h]
    return G.mean(0), G.std(0)


def sign_agreement(X):
    """s_ij = (1/d) Σ_k 1{sign(x_ik) = sign(x_jk)}  -> [h,h]"""
    S = np.sign(X)
    S[S == 0] = 1.0
    return (S @ S.T) / X.shape[1] * 0.5 + 0.5


def arcsin_pred(rho):
    return 0.5 + np.arcsin(np.clip(rho, -1, 1)) / np.pi


def mag_profile_check(X):
    """置換ヌルの前提チェック: |w| の座標間分布がユニット間で相関していないか。"""
    A = np.abs(X)
    A = A / np.maximum(A.sum(axis=1, keepdims=True), 1e-30)
    Ac = A - A.mean(axis=1, keepdims=True)
    Uc = C.unit(Ac, axis=1)
    G = Uc @ Uc.T
    iu = np.triu_indices(len(G), 1)
    coord_mean = A.mean(axis=0)
    return dict(mag_profile_pairwise_corr_mean=float(G[iu].mean()),
                mag_profile_pairwise_corr_absmean=float(np.abs(G[iu]).mean()),
                mag_coord_cv=float(coord_mean.std() / max(coord_mean.mean(), 1e-30)))


def analyse_run(z, i, obj):
    M, sv = C.get_matrix(z, i, obj, alive_only=True)
    if len(M) < 3:
        return None
    X = M * sv[:, None]                       # v 符号補正をベクトルに畳み込む
    d = X.shape[1]
    U = C.unit(X, axis=1)
    G = U @ U.T
    iu = np.triu_indices(len(U), 1)
    rho = G[iu]

    nm, ns = sign_perm_null(X)
    nm_s, ns_s = 0.5 * (nm + nm.T), 0.5 * (ns + ns.T)     # i<->j 非対称を対称化
    rho_null = nm_s[iu]
    zij = (rho - rho_null) / np.maximum(ns_s[iu], 1e-12)

    s_ij = sign_agreement(X)[iu]
    pred = arcsin_pred(rho)

    out = dict(n_units=int(len(X)), d=int(d), n_pairs=int(len(rho)),
               floor=C.chance_floor(d))
    out.update(C.cos_stats(rho, "obs_"))
    out.update(C.cos_stats(rho_null, "null_"))
    out["z_mean"] = float(zij.mean())
    out["z_absmean"] = float(np.abs(zij).mean())
    out["z_median"] = float(np.median(zij))
    out["frac_z_gt2"] = float((zij > 2).mean())
    out["diff_mean"] = float((rho - rho_null).mean())
    lo, hi = C.boot_ci(rho - rho_null)
    out["diff_ci_lo"], out["diff_ci_hi"] = lo, hi
    out["null_share"] = float(rho_null.mean() / rho.mean()) if abs(rho.mean()) > 1e-9 else np.nan
    out["excess_share"] = float((rho.mean() - rho_null.mean()) / rho.mean()) \
        if abs(rho.mean()) > 1e-9 else np.nan
    out["sign_agree_mean"] = float(s_ij.mean())
    out["sign_agree_minus_arcsin_mean"] = float((s_ij - pred).mean())
    out["frac_sign_agree_gt_95"] = float((s_ij > 0.95).mean())
    out.update(mag_profile_check(X))
    out["_pairs"] = (rho, rho_null, zij, s_ij)
    return out


def main():
    runs = C.load_runs()
    rows = collections.defaultdict(list)
    pairs = collections.defaultdict(list)

    for exp, width, step, z in C.iter_units(steps=STEPS):
        for i, rid in enumerate(z["run_ids"]):
            if not z["finite"][i]:
                continue
            r = runs[rid]
            for obj in ["W", "Eg"]:
                o = analyse_run(z, i, obj)
                if o is None:
                    continue
                p = o.pop("_pairs")
                o["run_id"] = rid
                key = (step, C.cond_label(r), obj)
                rows[key].append(o)
                pairs[key].append(p)

    KEYS = sorted({k for v in rows.values() for e in v for k in e} - {"run_id"})
    summary = {}
    for (step, cond, obj), lst in sorted(rows.items()):
        s = {k: C.agg_seeds([float(e.get(k, np.nan)) for e in lst]) for k in KEYS}
        s["n_runs"] = len(lst)
        summary[f"step{step}|{cond}|{obj}"] = s

    C.save_json(dict(steps=STEPS, n_perm=N_PERM, two_over_pi=TWO_OVER_PI,
                     summary=summary), "B4", "b4.json")
    make_figs(summary, pairs)
    write_verdict(summary)
    return summary


def _sel(summary, step, obj):
    out = []
    for k, v in summary.items():
        s, cond, o = k.split("|")
        if s == f"step{step}" and o == obj and "_w100_" in cond:
            out.append((cond, v))
    return sorted(out)


PRIMARY = ["A_w100_T10000_std_lr0.01", "A_w100_T10000_centered_lr0.01",
           "B_w100_K10000_c2.0_lr0.01", "B_w100_K10000_c0.0_lr0.01"]


def make_figs(summary, pairs):
    plt = C.mpl()

    # (1) 観測 vs ヌル（条件別バー）
    for obj in ["W", "Eg"]:
        for step in STEPS:
            its = _sel(summary, step, obj)
            if not its:
                continue
            fig, ax = plt.subplots(figsize=(max(7, 1.4 * len(its)), 4.8))
            x = np.arange(len(its))
            for j, (k, lab, col) in enumerate([
                    ("obs_signed_mean", "observed mean ρ (signed, v-corrected)", "tab:red"),
                    ("null_signed_mean", "sign-fixed / magnitude-permuted null", "tab:gray")]):
                m = [it[1][k]["mean"] for it in its]
                sd = [it[1][k]["std"] for it in its]
                ax.bar(x + (j - 0.5) * 0.38, m, 0.38, yerr=sd, capsize=2, label=lab, color=col)
            ax.axhline(TWO_OVER_PI, ls="--", color="green", lw=1.2)
            ax.text(0.01, TWO_OVER_PI + 0.02, "2/π = 0.637", fontsize=8, color="green",
                    transform=ax.get_yaxis_transform())
            ax.axhline(0, color="k", lw=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels([it[0].replace("_lr0.01", "") for it in its],
                               rotation=30, ha="right", fontsize=7)
            ax.set_ylabel("mean signed cos over pairs")
            ax.set_title(f"B4: observed alignment vs sign-only null — obj={obj}, step={step:g}",
                         fontsize=10)
            ax.grid(alpha=0.3, axis="y")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(C.figpath("B4", f"fig_b4_null_{obj}_step{step}.png"), dpi=140)
            plt.close(fig)

    # (2) 条件ごとに 1 図: s_ij vs ρ_ij 散布（仕様書 §8: 重ねない）
    for obj in ["W", "Eg"]:
        for cond in PRIMARY:
            key = (1000000, cond, obj)
            if key not in pairs:
                continue
            rho = np.concatenate([p[0] for p in pairs[key]])
            s_ij = np.concatenate([p[3] for p in pairs[key]])
            g = C.rng()
            if len(rho) > 20000:
                sel = g.choice(len(rho), 20000, replace=False)
                rho, s_ij = rho[sel], s_ij[sel]
            fig, ax = plt.subplots(figsize=(6.2, 5.2))
            ax.scatter(rho, s_ij, s=4, alpha=0.15, color="tab:blue", lw=0)
            t = np.linspace(-1, 1, 400)
            ax.plot(t, arcsin_pred(t), "k-", lw=1.6,
                    label="arcsin: ½ + arcsin(ρ)/π  (alignment primitive)")
            ax.axhspan(0.95, 1.0, color="tab:orange", alpha=0.25,
                       label="s ≈ 1 band (sign primitive)")
            ax.axvline(TWO_OVER_PI, color="green", ls="--", lw=1.2, label="2/π = 0.637")
            ax.set_xlabel("ρ_ij = cos(ŵ_i, ŵ_j)   [signed, v-corrected]")
            ax.set_ylabel("s_ij = coordinate-wise sign agreement")
            ax.set_xlim(-1, 1)
            ax.set_ylim(0, 1.02)
            ax.set_title(f"B4: sign agreement vs cos — {cond.replace('_lr0.01','')}, "
                         f"obj={obj}, step=1e6", fontsize=9)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7, loc="upper left")
            fig.tight_layout()
            fig.savefig(C.figpath("B4", f"fig_b4_signscatter_{obj}_{cond}.png"), dpi=140)
            plt.close(fig)

    # (3) z 分布（条件別、obj=Eg）
    its = [(c, k) for c in PRIMARY for k in [(1000000, c, "Eg")] if k in pairs]
    if its:
        fig, axes = plt.subplots(1, len(its), figsize=(3.2 * len(its), 3.6), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, (cond, key) in zip(axes, its):
            zz = np.concatenate([p[2] for p in pairs[key]])
            ax.hist(zz, bins=60, color="tab:purple", alpha=0.85)
            ax.axvline(0, color="k", lw=1)
            ax.set_title(f"{cond.replace('_lr0.01','')}\nmean z = {zz.mean():.1f}", fontsize=8)
            ax.set_xlabel("z_ij = (ρ − null) / std(null)")
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("pairs")
        fig.suptitle("B4: excess of observed alignment over the sign-only null (obj=Eg, step=1e6)",
                     fontsize=10)
        fig.tight_layout()
        fig.savefig(C.figpath("B4", "fig_b4_zdist_Eg.png"), dpi=140)
        plt.close(fig)
    print("  wrote B4 figures")


def write_verdict(summary):
    L = []
    for obj in ["W", "Eg"]:
        its = [it for it in _sel(summary, 1000000, obj) if it[0] in PRIMARY]
        if not its:
            continue
        for cond, v in its:
            L.append(f"obj={obj} {cond.replace('_lr0.01','')}: "
                     f"ρ_obs={v['obs_signed_mean']['mean']:+.3f} "
                     f"ρ_null={v['null_signed_mean']['mean']:+.3f} "
                     f"z={v['z_mean']['mean']:+.1f} "
                     f"excess_share={v['excess_share']['mean']:.2f} "
                     f"s̄={v['sign_agree_mean']['mean']:.3f} "
                     f"(s−arcsin)={v['sign_agree_minus_arcsin_mean']['mean']:+.3f}")
    C.verdict("B4", "verdict.txt", "\n".join(L))


if __name__ == "__main__":
    main()
