"""B3 — Σ 最小固有ベクトルとの照合（H-Σ の検証、仕様書 §5）。

  .venv/bin/python -m analysis.june_reanalysis.b3_sigma_eig

仕様書 §5-2 の指示どおり、**スペクトル検査を先に行い**、Σ が等方なら「判定不能」で終了する。
条件B は設計上 Σ = I_21（`envs.py:133-156`, x = µ + z, z~N(0,I)）なので空虚になるはず。
"""
import collections

import numpy as np

from . import common as C
from . import measure as Ms
from .b2_cov_check import vhat_from_npz, perp

STEPS = [100000, 1000000]
N_NULL = 10000
DEGEN_TOL = 0.05          # 仕様書 §5-2: 最小固有値との比が 1±0.05 以内を縮退とみなす


def sigma_analytic(exp, d, cfg):
    """Σ の解析値（仕様書 §5-1「解析値優先」）。`src/envs.py` の定義から。

    条件A: x = [flip(f ビット, 周期内で定数), U{0,1}(d-f ビット)]
           -> 周期内 Σ = diag(0×f, 0.25×(d-f))
    条件B: x = µ + z, z ~ N(0, I_d)  -> Σ = I_d
    """
    if exp == "B":
        return np.eye(d)
    f = cfg["condA"]["f"]
    return np.diag(np.array([0.0] * f + [0.25] * (d - f)))


def degenerate_subspace(evals, evecs, which="min"):
    """λ_min（または λ_max）と比が 1±tol 以内の固有値群 S を返す。"""
    ref = evals[0] if which == "min" else evals[-1]
    if abs(ref) < 1e-12:
        # λ_ref ≈ 0 のときは比が定義できないので絶対スケールで判定する
        scale = max(evals[-1], 1e-12)
        sel = np.abs(evals - ref) <= DEGEN_TOL * scale
    else:
        sel = np.abs(evals / ref - 1.0) <= DEGEN_TOL
    return evecs[:, sel], int(sel.sum())


def proj_frac(P, v):
    """‖P_S v‖² （v は単位ベクトル、P は S の正規直交基底 [d, k]）。"""
    v = C.unit(v)
    return float(np.sum((P.T @ v) ** 2))


def null_pvalue(P, stat, d, rng, n=N_NULL, restrict=None):
    """一様ランダム単位ベクトルの ‖P_S v‖² 帰無分布と片側 p 値。"""
    V = rng.standard_normal((n, d))
    if restrict is not None:                       # µ⊥ 内に限定する場合
        m = C.unit(restrict)
        V = V - (V @ m)[:, None] * m[None, :]
    V = C.unit(V, axis=1)
    s = np.sum((V @ P) ** 2, axis=1)
    return float((s >= stat).mean()), float(s.mean()), float(s.std())


def main():
    runs = C.load_runs()
    rng = C.rng()
    cfg = Ms.load_cfg()
    rows = collections.defaultdict(list)
    spectra = {}

    for exp, width in C.GROUPS:
        for step in STEPS:
            m = Ms.get(exp, width, step)
            z = C.load_npz(exp, width, step)
            if m is None or z is None:
                continue
            for i, rid in enumerate(z["run_ids"]):
                if not z["finite"][i]:
                    continue
                r = runs[rid]
                d = int(m["d"])
                lab = C.cond_label(r)
                # --- 手順 1: Σ は解析値を優先（経験共分散は診断用に併記）
                ev, U = np.linalg.eigh(sigma_analytic(exp, d, cfg))
                Se = 0.5 * (m["Sigma_in"][i] + m["Sigma_in"][i].T)
                eve = np.linalg.eigvalsh(Se)
                spectra.setdefault(f"step{step}|{lab}", []).append(ev.tolist())
                spectra.setdefault(f"emp|step{step}|{lab}", []).append(eve.tolist())

                # --- 手順 2: スペクトル検査（これが先）
                Pmin, kmin = degenerate_subspace(ev, U, "min")
                Pmax, kmax = degenerate_subspace(ev, U, "max")
                _, kmin_e = degenerate_subspace(eve, np.linalg.eigh(Se)[1], "min")
                e = dict(run_id=rid, step=step, d=d,
                         lam_min=float(ev[0]), lam_max=float(ev[-1]),
                         lam_ratio=float(ev[-1] / max(ev[0], 1e-30)),
                         dim_S_min=kmin, dim_S_max=kmax,
                         isotropic=bool(kmin == d),
                         emp_lam_min=float(eve[0]), emp_lam_max=float(eve[-1]),
                         emp_dim_S_min=kmin_e, M_samples=float(m["M"][i]),
                         null_mean_min=kmin / d, null_mean_max=kmax / d)

                mu_in = m["mu_in"][i]
                # --- 手順 3: 共通方向の候補
                vh, sv1 = vhat_from_npz(z, i, mu_in)
                cands = {}
                e["n_alive"] = int((~z["dead"][i]).sum())
                e["has_vhat"] = float(vh is not None)
                if vh is not None:
                    cands["vhat_Eg"] = vh
                    e["sv1_frac"] = sv1
                # (b) c=0 / centered では µ≈0 なので重み行列そのものの第1特異ベクトル
                Wm, _ = C.get_matrix(z, i, "W", alive_only=True)
                if len(Wm) >= 3:
                    _, s_, Vt = np.linalg.svd(C.unit(Wm, axis=1), full_matrices=False)
                    cands["sv1_W"] = Vt[0]
                    e["sv1_frac_W"] = float(s_[0] ** 2 / max((s_ ** 2).sum(), 1e-30))
                Egm, svg = C.get_matrix(z, i, "Eg", alive_only=True)
                if len(Egm) >= 3:
                    _, s_, Vt = np.linalg.svd(C.unit(Egm, axis=1) * svg[:, None],
                                              full_matrices=False)
                    cands["sv1_Eg"] = Vt[0]

                # --- 手順 4-5: 検定統計量と p 値（全空間 / µ⊥ 内）
                for name, v in cands.items():
                    for wtag, P, k in [("min", Pmin, kmin), ("max", Pmax, kmax)]:
                        if k == 0 or k == d:
                            e[f"projfrac_{name}_{wtag}"] = np.nan if k == 0 else 1.0
                            e[f"p_{name}_{wtag}"] = np.nan
                            continue
                        st = proj_frac(P, v)
                        p, nm, ns = null_pvalue(P, st, d, rng)
                        e[f"projfrac_{name}_{wtag}"] = st
                        e[f"p_{name}_{wtag}"] = p
                        e[f"nullmean_{name}_{wtag}"] = nm
                        # µ⊥ 内での比較
                        vp = perp(v, mu_in)
                        mh = C.unit(mu_in)
                        Pp = P - np.outer(mh, mh @ P)      # S を µ⊥ に射影 [d,k]
                        q, rr = np.linalg.qr(Pp)
                        rank = int((np.abs(np.diag(rr)) > 1e-8).sum())
                        q = q[:, :max(rank, 1)]
                        stp = proj_frac(q, vp)
                        pp, _, _ = null_pvalue(q, stp, d, rng, restrict=mu_in)
                        e[f"projfrac_{name}_{wtag}_muperp"] = stp
                        e[f"p_{name}_{wtag}_muperp"] = pp
                rows[(step, lab)].append(e)

    KEYS = sorted({k for v in rows.values() for e in v for k in e} - {"run_id"})
    summary = {}
    for (step, cond), lst in sorted(rows.items()):
        s = {k: C.agg_seeds([float(e.get(k, np.nan))
                             if not isinstance(e.get(k), bool) else float(e[k])
                             for e in lst]) for k in KEYS}
        s["n_runs"] = len(lst)
        summary[f"step{step}|{cond}"] = s

    spec_mean = {k: np.mean(np.array(v), axis=0).tolist() for k, v in spectra.items()}
    C.save_json(dict(steps=STEPS, degen_tol=DEGEN_TOL, n_null=N_NULL,
                     summary=summary, sigma_spectrum_mean=spec_mean), "B3", "b3.json")
    make_figs(summary, spec_mean)
    write_verdict(summary)
    return summary


def _sel(summary, step):
    out = []
    for k, v in summary.items():
        s, cond = k.split("|")
        if s == f"step{step}" and "_w100_" in cond:
            out.append((cond, v))
    return sorted(out)


def make_figs(summary, spec):
    plt = C.mpl()
    # Σ 固有値スペクトル（条件別に分ける — 仕様書 §8）
    items = sorted(k for k in spec if k.startswith("step1000000") and "_w100_" in k)
    n = len(items)
    ncol = min(5, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 2.9 * nrow), sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n:]:
        ax.axis("off")
    for ax, k in zip(axes, items):
        ev = np.array(spec[k])
        ax.plot(np.arange(1, len(ev) + 1), ev, "o-", ms=4, label="analytic")
        ke = "emp|" + k
        if ke in spec:
            ax.plot(np.arange(1, len(spec[ke]) + 1), np.array(spec[ke]), "s--", ms=3,
                    color="tab:gray", alpha=0.8, label="empirical (x_in)")
        ax.set_yscale("symlog", linthresh=1e-4)
        ax.set_title(k.split("|")[1].replace("_lr0.01", ""), fontsize=7)
        ax.set_xlabel("eigenvalue index")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("eigenvalue of Σ")
    axes[0].legend(fontsize=7)
    fig.suptitle("B3: spectrum of Σ — is there anisotropy to test at all?  (step=1e6, w=100)\n"
                 "empirical spread at small windows is finite-sample (Wishart) noise, not anisotropy",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(C.figpath("B3", "fig_b3_sigma_spectrum.png"), dpi=140)
    plt.close(fig)

    for step in STEPS:
        its = _sel(summary, step)
        if not its:
            continue
        fig, ax = plt.subplots(figsize=(max(7, 1.4 * len(its)), 4.8))
        x = np.arange(len(its))
        m = [it[1].get("projfrac_vhat_Eg_min", {"mean": np.nan})["mean"] for it in its]
        sd = [it[1].get("projfrac_vhat_Eg_min", {"std": np.nan})["std"] for it in its]
        nm = [it[1].get("null_mean_min", {"mean": np.nan})["mean"] for it in its]
        ax.bar(x - 0.2, m, 0.4, yerr=sd, capsize=2, label="‖P_S v̂‖²  (observed)",
               color="tab:purple")
        ax.bar(x + 0.2, nm, 0.4, label="null E = dim(S)/d", color="lightgray")
        for xi, it in zip(x, its):
            k = it[1].get("dim_S_min", {"mean": np.nan})["mean"]
            dd = it[1]["d"]["mean"]
            ax.text(xi, 1.02, f"dim S={k:.0f}/{dd:.0f}", ha="center", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels([it[0].replace("_lr0.01", "") for it in its],
                           rotation=30, ha="right", fontsize=7)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("‖P_S v̂‖²   (S = min-eigenvalue subspace)")
        ax.set_title(f"B3: H-Σ test — v̂ vs the minimum-eigenvalue subspace of Σ  step={step:g}",
                     fontsize=10)
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(C.figpath("B3", f"fig_b3_projfrac_step{step}.png"), dpi=140)
        plt.close(fig)
    print("  wrote B3 figures")


def write_verdict(summary):
    its = _sel(summary, 1000000)
    L = []
    for cond, v in its:
        d = v["d"]["mean"]
        k = v["dim_S_min"]["mean"]
        iso = v["isotropic"]["mean"]
        pf = v.get("projfrac_vhat_Eg_min", {"mean": np.nan})["mean"]
        p = v.get("p_vhat_Eg_min", {"mean": np.nan})["mean"]
        L.append(f"{cond.replace('_lr0.01','')}: dim(S)={k:.1f}/{d:.0f} "
                 f"(等方率{iso:.0%}) ‖P_S v̂‖²={pf:.3f} vs null {k/d:.3f}, p={p:.3f}")
    C.verdict("B3", "verdict.txt", "\n".join(L))


if __name__ == "__main__":
    main()
