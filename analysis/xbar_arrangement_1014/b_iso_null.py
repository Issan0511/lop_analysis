"""xbar_arrangement_1014 agent B, follow-up: isotropic-pattern surrogate ("iso") for N_i(T) and C_i(T). Same box, definitions, conditions,
targets and statistics as b_strat_null.py (whose functions are imported; `counts` there is the verbatim copy of cobirth_seeds_1013 verify_counts.counts).
iso surrogate, per phase-2 state T and per alive unit k (candidate role only; the target's own T_i, O_i, k_i stay actual):
  x_bar = mean_n Xt_n (785), K_n = Xt_n . x_bar, K_bar = mean K (= x_bar . x_bar);  actual z_kn = Xt_n . (w_k, b_k);  m_k = mean_n z_kn;
  u_perp_kn = z_kn - m_k K_n / K_bar;  s_perp_k = sd_n(u_perp_kn) (ddof 0).
  g ~ N(0, I_785);  g <- g - x_bar (x_bar . g)/K_bar;  u = Xt g;  u <- u * s_perp_k / sd_n(u);  z_iso_kn = m_k K_n / K_bar + u_n;  o_k = 1[z_iso > 0].
  One g per (unit, state, draw); R = 20 draws, rng = np.random.default_rng(2000 + r), drawn state by state T = 11..100 for all 100 columns.
  Candidate eligibility (alive) is the unit's ACTUAL alive status, as in strat/flat.
iso+common surrogate (mode "isoc"), coordinator follow-up 2: per state, u_bar_n = mean over ALIVE k of u_perp_kn; for each alive candidate the isotropic
  u is drawn as above (fresh g, projected, u = Xt g) but rescaled to sd_n(u) = sqrt(max(s_perp_k^2 - sd_n(u_bar)^2, 0.25 s_perp_k^2)); u_isoc_kn = u_bar_n + u_n;
  z_isoc_kn = m_k K_n / K_bar + u_isoc_kn; gates 1[z_isoc > 0]; rng = np.random.default_rng(3000 + r).
Outputs: results/xbar_arrangement_1014/b_<mode>_<tag>.json, obsidian-research-data/xbar_arrangement_1014/agent_b/b_arrays_<mode>_<tag>.npz
  (keys N, C, N<mode>_mean, C<mode>_mean, N<mode>_draws, C<mode>_draws, T_list).
usage: b_iso_null.py <tag> <npz rel. to ROOT> <gate leaky|E1|E2> [R] [mode iso|isoc]"""
import os, sys, json, time
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(_k, "1")
import numpy as np, warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b_strat_null as B
from common import load_X  # path inserted by b_strat_null


def main():
    tag, npz, gate = sys.argv[1:4]; R = int(sys.argv[4]) if len(sys.argv) > 4 else 20
    mode = sys.argv[5] if len(sys.argv) > 5 else "iso"; assert mode in ("iso", "isoc"); seed0 = 2000 if mode == "iso" else 3000
    t0 = time.time()
    Nc, Cc, kk, store, K = B.counts(npz, gate)
    v = np.load(os.path.join(B.VER, f"vcounts_{tag}.npz"))
    match = bool((v["N"] == Nc).all()) and bool((v["C"][Nc >= 0] == Cc[Nc >= 0]).all())
    d = np.load(os.path.join(B.ROOT, npz)); X = load_X(d["subset"]); Xt = np.c_[X, np.ones(len(X))]
    xbar = Xt.mean(0); Kbar = float(K.mean()); assert abs(Kbar - xbar @ xbar) < 1e-8 * Kbar
    Kn = K / Kbar
    order = np.argsort(K, kind="stable"); n = len(K); blocks = [order[q * n // 10:(q + 1) * n // 10] for q in range(10)]
    H = Nc.shape[1]; Na = Nc[B.PH]; Ca = Cc[B.PH]
    ids = [i for i in range(H) if (Na[:, i] >= 0).sum() >= 30]
    # actual m_k, s_perp_k per phase-2 state
    M = {}; Sp = {}; E0 = d["end0"]; E1 = d["end1"]          # decompress once
    Ub = {}; Sres = {}; common = dict(sd_ratio=[], var_share=[], unit_r2_median=[], floor_frac=[])
    for T in range(11, 101):
        W = E0[T - 1].astype(np.float64); b = E1[T - 1].astype(np.float64); Z = X @ W.T + b[None, :]
        m = Z.mean(0); U = Z - Kn[:, None] * m[None, :]
        assert np.abs(U.mean(0)).max() < 1e-8 * (np.abs(Z).max() + 1)
        M[T] = m; Sp[T] = U.std(0)
        al = store[T][3]; ub = U[:, al].mean(1); sdu = ub.std(); Ub[T] = ub
        sp = Sp[T]; Sres[T] = np.sqrt(np.maximum(sp ** 2 - sdu ** 2, 0.25 * sp ** 2))
        spa = sp[al]; common["sd_ratio"].append(sdu / spa.mean()); common["var_share"].append(sdu ** 2 / (spa ** 2).mean())
        Ua = U[:, al]; cr = (Ua * (ub - ub.mean())[:, None]).mean(0) / (Ua.std(0) * ub.std() + 1e-300)
        common["unit_r2_median"].append(float(np.median(cr ** 2))); common["floor_frac"].append(float((sp[al] ** 2 - sdu ** 2 < 0.25 * sp[al] ** 2).mean()))
    Ns = np.full((R, 90, H), -1, np.int16); Cs = np.full((R, 90, H), -1, np.int16)
    p_act = []; p_iso = []; occ_act = np.zeros(10); occ_iso = np.zeros(10); tot = np.zeros(10); pk_pairs = []
    for r in range(R):
        rng = np.random.default_rng(seed0 + r)
        for T in range(11, 101):
            O, T3, k, alive = store[T]
            G = rng.standard_normal((Xt.shape[1], H))
            G -= np.outer(xbar, xbar @ G) / Kbar
            U = Xt @ G
            if r == 0 and T == 11: assert np.abs(U.mean(0)).max() < 1e-8 * np.abs(U).max()
            if mode == "iso":
                U *= (Sp[T] / U.std(0))[None, :]
            else:
                U *= (Sres[T] / U.std(0))[None, :]; U += Ub[T][:, None]
            Zi = Kn[:, None] * M[T][None, :] + U
            P = Zi > 0
            P[:, ~alive] = False                                    # dead units are never candidates (alive mask also excludes them)
            n_, c_ = B.surr_counts(O, T3, k, alive, P)
            Ns[r, T - 11] = n_; Cs[r, T - 11] = c_
            pa = O[:, alive].mean(0); pi = P[:, alive].mean(0)
            p_iso.append(pi.mean())
            if r == 0: p_act.append(pa.mean()); pk_pairs.append(np.c_[pa, pi])
            for q, idx in enumerate(blocks):
                occ_iso[q] += P[np.ix_(idx, alive)].sum()
                if r == 0: occ_act[q] += O[np.ix_(idx, alive)].sum(); tot[q] += len(idx) * alive.sum()
    occ_iso /= (tot * R); occ_act /= tot
    pk = np.concatenate(pk_pairs)
    res = dict(tag=tag, npz=npz, gate=gate, R=R, mode=mode, seeds=f"default_rng({seed0} + r)", match_cobirth_counts=match, n_targets=len(ids),
               common_component={k_: dict(median=float(np.median(v_)), q05=float(np.quantile(v_, .05)), q95=float(np.quantile(v_, .95))) for k_, v_ in common.items()},
               widths=dict(p_actual_mean_over_alive_units=float(np.mean(p_act)), p_iso_mean_over_alive_units=float(np.mean(p_iso)),
                           ratio_iso_over_actual=float(np.mean(p_iso) / np.mean(p_act)),
                           r0_unit_level=dict(median_p_actual=float(np.median(pk[:, 0])), median_p_iso=float(np.median(pk[:, 1])),
                                              pearson_p=float(np.corrcoef(pk[:, 0], pk[:, 1])[0, 1]), median_abs_diff=float(np.median(np.abs(pk[:, 0] - pk[:, 1]))),
                                              frac_iso_k0=float((pk[:, 1] == 0).mean()), n_unit_states=int(len(pk))),
                           occupancy_by_decile_actual=occ_act.tolist(), occupancy_by_decile_iso=occ_iso.tolist()),
               Kbar=Kbar)
    sN = B.series_stats(Na, ids); sC = B.series_stats(Ca, ids)
    for qty, Mact, sA, S in (("N", Na, sN, Ns), ("C", Ca, sC, Cs)):
        st = [B.series_stats(S[r], ids) for r in range(R)]
        Fm = np.array([s["F"] for s in st]); Mm = np.array([s["mu"] for s in st])
        out = dict(actual={k_: sA[k_] for k_ in B.SCAL}, surr=dict(summary=B.summarize_surr(st), n_F_nan_total=int(sum(s["n_F_nan"] for s in st)),
                   B2_F=B.per_target_band(sA["F"], Fm), B2_mu=B.per_target_band(sA["mu"], Mm), B3=B.common_cause(Mact, S.astype(float), ids),
                   per_target_F_median_over_r=np.nanmedian(Fm, 0).tolist(), per_target_mu_median_over_r=np.median(Mm, 0).tolist()))
        out["surr"]["ratio_mu_pooled"] = out["surr"]["summary"]["mu_pooled"]["median"] / sA["mu_pooled"]
        out["surr"]["ratio_mu_median"] = out["surr"]["summary"]["mu_median"]["median"] / sA["mu_median"] if sA["mu_median"] > 0 else None
        # 0/1-only share (actual and per draw) for reading F_i medians
        def share01(Mx):
            c = 0
            for i in ids:
                x = Mx[Mx[:, i] >= 0, i]
                c += int(x.max() == 1)
            return c / len(ids)
        out["surr"]["share_01"] = B.summarize_surr([dict(F_median=share01(S[r]), mu_median=0, mu_pooled=0, pool_fano=0, pool_fano_pop=0, max=0, n_ge10=0) for r in range(R)])["F_median"]
        out[mode] = out.pop("surr"); res[qty] = out
    def mean(Sx):
        m = Sx.astype(np.float64).mean(0); m[Na < 0] = -1.0; return m
    assert ((Ns < 0) == (Na < 0)[None]).all()
    np.savez_compressed(os.path.join(B.BIG, f"b_arrays_{mode}_{tag}.npz"), N=Na.astype(np.int16), C=Ca.astype(np.int16), **{f"N{mode}_mean": mean(Ns), f"C{mode}_mean": mean(Cs),
                        f"N{mode}_draws": Ns, f"C{mode}_draws": Cs}, T_list=np.arange(11, 101))
    res["elapsed_s"] = time.time() - t0
    json.dump(res, open(os.path.join(B.RES, f"b_{mode}_{tag}.json"), "w"), indent=1)
    w = res["widths"]
    print(f"[{tag} {mode}] common sd ratio {res['common_component']['sd_ratio']['median']:.3f} | match {match} | p act {w['p_actual_mean_over_alive_units']:.4f} iso {w['p_iso_mean_over_alive_units']:.4f} | "
          f"N mu ratio {res['N'][mode]['ratio_mu_pooled']:.3f} pool F act {sN['pool_fano']:.3f} iso {res['N'][mode]['summary']['pool_fano']['median']:.3f} | "
          f"C mu ratio {res['C'][mode]['ratio_mu_pooled']:.3f} pool F act {sC['pool_fano']:.3f} iso {res['C'][mode]['summary']['pool_fano']['median']:.3f} | {res['elapsed_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
