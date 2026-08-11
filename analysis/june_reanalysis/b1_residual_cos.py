"""B1 — µ 射影除去後の残差 inter-unit cos（仕様書 §3）。

  .venv/bin/python -m analysis.june_reanalysis.b1_residual_cos

Phase 0 の発見により、対象を 2 つ並走させる:
  obj='W'  学習器 第1層の重み w_i        （仕様書の文面どおり）
  obj='Eg' 凍結測定の期待勾配 E[g_{W_i}] （既報 A1 が実際に使った対象）

出力: results/june_reanalysis/B1/{b1.json, verdict.txt, fig_*.png, v_hat.npz}
"""
import collections
import os

import numpy as np

from . import common as C

STEPS = [100000, 1000000]


def analyse_run(z, i, obj, alive_only=True):
    """1 run × 1 ckpt の B1 統計。"""
    M, sv = C.get_matrix(z, i, obj, alive_only=alive_only)
    d = M.shape[1]
    out = dict(n_units=int(len(M)), d=int(d), floor=C.chance_floor(d))
    if len(M) < 3:
        return None

    muh = C.unit(z["mu_inter"][i])
    U = C.unit(M, axis=1)

    # --- 手順 4: 除去前の再現確認
    out.update(C.cos_stats(C.pair_cos(M), "pre_"))
    out.update(C.cos_stats(C.pair_cos(M, sv), "pre_vsigned_"))
    alpha = U @ muh                                    # cos(w_i, µ̂)  [n]
    out["mu_cos_signed_mean"] = float(alpha.mean())
    out["mu_cos_abs_mean"] = float(np.abs(alpha).mean())
    out["alpha_sq_mean"] = float((alpha ** 2).mean())

    # --- 手順 2-3: µ 射影を抜いた残差の pairwise cos
    Mp = M - (M @ muh)[:, None] * muh[None, :]
    nrm = np.linalg.norm(Mp, axis=1)
    keep = nrm > 1e-10 * max(np.linalg.norm(M, axis=1).max(), 1e-30)
    Mp, svp = Mp[keep], sv[keep]
    out["n_units_perp"] = int(len(Mp))
    if len(Mp) < 3:
        return out
    cperp = C.pair_cos(Mp)
    out.update(C.cos_stats(cperp, "perp_"))
    out.update(C.cos_stats(C.pair_cos(Mp, svp), "perp_vsigned_"))
    out["perp_signed_ci"] = C.boot_ci(cperp)

    # --- 手順 5: 独立残差モデルの予測（実測 α で再計算）
    a = alpha[keep]
    ii = np.triu_indices(len(a), 1)
    out["indep_pred_pre_signed_mean"] = float((np.outer(a, a))[ii].mean())
    abar2 = float((a ** 2).mean())
    out["alpha_sq_mean_perp"] = abar2
    # 除去前 signed 実測から µ 寄与を引いた残り = 第二共通方向の寄与 β²
    beta2 = float(np.mean(C.pair_cos(M[keep])) - (np.outer(a, a))[ii].mean())
    out["beta_sq_from_data"] = beta2
    out["beta_from_data"] = float(np.sqrt(beta2)) if beta2 > 0 else float("nan")
    out["perp_pred_if_second_dir"] = float(beta2 / max(1.0 - abar2, 1e-12))

    # --- 手順 6: 第二共通方向の抽出（残差行列の第1右特異ベクトル）
    Up = C.unit(Mp, axis=1)
    # 符号の任意性を潰すため v 符号補正版でも取る（A1 と同じ規約）
    for tag, X in [("", Up), ("_vsigned", Up * svp[:, None])]:
        _, s, Vt = np.linalg.svd(X, full_matrices=False)
        frac = float(s[0] ** 2 / max((s ** 2).sum(), 1e-30))
        vh = Vt[0]
        cs = X @ vh
        out[f"sv1_frac{tag}"] = frac
        out[f"sv_spectrum{tag}"] = (s ** 2 / max((s ** 2).sum(), 1e-30)).tolist()
        out[f"vhat_cos_absmean{tag}"] = float(np.abs(cs).mean())
        out[f"vhat_cos_signedmean{tag}"] = float(cs.mean())
        out[f"vhat_cos_mu{tag}"] = float(np.abs(vh @ muh))
        out[f"vhat{tag}"] = vh.tolist()
    return out


def main():
    runs = C.load_runs()
    res = collections.defaultdict(list)     # (step, cond, obj) -> [per-run dict]
    vhats = {}

    for exp, width, step, z in C.iter_units(steps=STEPS):
        for i, rid in enumerate(z["run_ids"]):
            if not z["finite"][i]:
                continue
            r = runs[rid]
            for obj in ["W", "Eg"]:
                o = analyse_run(z, i, obj)
                if o is None:
                    continue
                o["run_id"] = rid
                res[(step, C.cond_label(r), obj)].append(o)
                if "vhat_vsigned" in o:
                    vhats[f"{step}|{rid}|{obj}"] = np.array(o["vhat_vsigned"])

    # ---- 条件 × seed 集約
    summary = {}
    KEYS = ["pre_signed_mean", "pre_abs_mean", "pre_vsigned_signed_mean",
            "mu_cos_signed_mean", "mu_cos_abs_mean",
            "perp_signed_mean", "perp_abs_mean", "perp_vsigned_signed_mean",
            "indep_pred_pre_signed_mean", "beta_sq_from_data",
            "perp_pred_if_second_dir", "sv1_frac_vsigned", "sv1_frac",
            "vhat_cos_absmean_vsigned", "vhat_cos_mu_vsigned", "floor"]
    for k, lst in sorted(res.items()):
        step, cond, obj = k
        e = {kk: C.agg_seeds([o.get(kk, np.nan) for o in lst]) for kk in KEYS}
        e["n_units_mean"] = float(np.mean([o["n_units"] for o in lst]))
        e["n_runs"] = len(lst)
        # 全 seed の pair を束ねた bootstrap CI は pair レベルの再計算が要るので
        # seed 平均の CI で代用せず、seed 間 std を主表示にする（§2 の規約どおり）
        summary[f"step{step}|{cond}|{obj}"] = e

    C.save_json(dict(steps=STEPS, reported=C.REPORTED, summary=summary),
                "B1", "b1.json")
    np.savez_compressed(os.path.join(C.OUT, "B1", "v_hat.npz"), **vhats)
    print(f"  wrote B1/v_hat.npz ({len(vhats)} vectors)")

    make_figs(summary)
    write_verdict(summary)
    return summary


def _sel(summary, step, obj, exps=("A", "B")):
    out = []
    for k, v in summary.items():
        s, cond, o = k.split("|")
        if s != f"step{step}" or o != obj or not cond.startswith(tuple(exps)):
            continue
        if "_w100_" not in cond:
            continue
        out.append((cond, v))
    return sorted(out)


def make_figs(summary):
    plt = C.mpl()
    for obj in ["W", "Eg"]:
        for step in STEPS:
            items = _sel(summary, step, obj)
            if not items:
                continue
            fig, ax = plt.subplots(figsize=(max(7, 1.3 * len(items)), 5))
            x = np.arange(len(items))
            series = [("pre_abs_mean", "|cos| before µ removal", "tab:gray"),
                      ("perp_abs_mean", "|cos| after µ removal", "tab:blue"),
                      ("mu_cos_abs_mean", "|cos(·, µ̂)|", "tab:orange"),
                      ("perp_vsigned_signed_mean", "signed cos⊥ (v-corrected)", "tab:red")]
            w = 0.8 / len(series)
            for j, (key, lab, col) in enumerate(series):
                m = [it[1][key]["mean"] for it in items]
                sd = [it[1][key]["std"] for it in items]
                ax.bar(x + (j - (len(series) - 1) / 2) * w, m, w, yerr=sd,
                       capsize=2, label=lab, color=col)
            fl = items[0][1]["floor"]["mean"]
            ax.axhline(fl, ls=":", color="k", lw=1.2)
            ax.text(0.01, fl + 0.01, f"random floor √(2/πd) = {fl:.3f}",
                    fontsize=8, transform=ax.get_yaxis_transform())
            ax.axhline(0, color="k", lw=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels([it[0].replace("_lr0.01", "") for it in items],
                               rotation=30, ha="right", fontsize=7)
            ax.set_ylabel("cos statistic")
            ax.set_title(f"B1: inter-unit cos before/after µ̂ removal — obj={obj}, step={step:g}\n"
                         f"(width=100, alive units only; mean±std over seeds)", fontsize=10)
            ax.grid(alpha=0.3, axis="y")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(C.figpath("B1", f"fig_b1_{obj}_step{step}.png"), dpi=140)
            plt.close(fig)

    # 特異値スペクトル（残差行列, v 符号補正）
    for obj in ["W", "Eg"]:
        items = _sel(summary, 1000000, obj)
        if not items:
            continue
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for cond, v in items:
            ax.bar(cond.replace("_lr0.01", ""), v["sv1_frac_vsigned"]["mean"],
                   yerr=v["sv1_frac_vsigned"]["std"], capsize=3)
        ax.set_ylabel("λ₁ / Σλ  of residual matrix (v-signed)")
        ax.set_title(f"B1: concentration on the 1st residual direction — obj={obj}, step=1e6",
                     fontsize=10)
        ax.tick_params(axis="x", rotation=30, labelsize=7)
        for lab in ax.get_xticklabels():
            lab.set_ha("right")
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(C.figpath("B1", f"fig_b1_spectrum_{obj}.png"), dpi=140)
        plt.close(fig)
    print("  wrote B1 figures")


def write_verdict(summary):
    lines = []
    for obj in ["W", "Eg"]:
        items = _sel(summary, 1000000, obj)
        if not items:
            continue
        pre = np.nanmean([v["pre_abs_mean"]["mean"] for _, v in items])
        perp = np.nanmean([v["perp_abs_mean"]["mean"] for _, v in items])
        mu = np.nanmean([v["mu_cos_abs_mean"]["mean"] for _, v in items])
        fl = items[0][1]["floor"]["mean"]
        sv1 = np.nanmean([v["sv1_frac_vsigned"]["mean"] for _, v in items])
        lines.append(f"obj={obj}: |cos| 除去前 {pre:.3f} -> 除去後 {perp:.3f} "
                     f"(床 {fl:.3f}), |cos(·,µ̂)| {mu:.3f}, 残差第1方向寄与率 {sv1:.3f}")
    C.verdict("B1", "verdict.txt", "\n".join(lines))


if __name__ == "__main__":
    main()
