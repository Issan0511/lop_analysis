"""xbar_arrangement_1014 / agent A: occupancy profile f_n vs brightness K_n, and the exchangeable-pattern model.
Read-only on ROOT (push_direction_proof_0930). One file per process.
usage: a_occupancy.py <tag>      writes results/xbar_arrangement_1014/a_occ_<tag>.json

Definitions (task statement, used exactly):
  Xt = [X, 1] (1200x785); xbar = mean_n Xt_n; K_n = Xt_n.xbar; Kbar = xbar.xbar.
  Wt_k = (W1_k, b1_k); z_kn = Xt_n.Wt_k; m_k = Wt_k.xbar; u_kn = z_kn - m_k K_n/Kbar (u-perp);
  s_k = sd_n z_kn; sp_k = sd_n u_kn; r_k = m_k/s_k; open z>0; p_k = mean_n 1[z>0]; alive p_k>0;
  f_n = fraction of alive units open on n.  State T = end0[T-1], end1[T-1]; phase 2 = T 11..100.
  Model: unit k open on n iff u_kn > c_kn := -m_k K_n/Kbar (= |m_k| K_n/Kbar for m_k<0);
  Fhat(n) = mean_{k alive} Phat_k(c_kn), Phat_k(c) = #{n': u_kn' > c}/N  (= permutation expectation);
  FG(n) = mean_k Phibar(c_kn/sp_k);  F_flat = mean_k p_k.
  Threshold inputs T_i (copied from cobirth_seeds_1013/verify_counts.py):
    leaky/E1: score = log K - |x| - 2 log(1+e^-|x|) - log tau, x = z/tau, tau = 0.1 sd_n z + 1e-12
    E2:       score = log K + z on closed (z<0), -inf on open.  top 3 by score (stable argsort).
Extras (not in the task list, labelled as such in the report):
  sim_*  : f drawn from the model itself (independent permutation of each unit's u-perp over inputs), R_SIMF draws per state;
  A5 'unit', 'unit.K' and 'MC' model columns (per-unit normalised density; K-weighted; permutation Monte Carlo through the actual
  selection rule, R_MC draws); gate 'near' = 3 inputs with the smallest |z|; T = 0 (init) row in A3/A4.
"""
import os, sys, json, math, time
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[_k] = "2"
import numpy as np
ROOT = "/home/issan/Projects/obsidian-research-data/push_direction_proof_0930"
sys.path.insert(0, ROOT); from common import load_X
OUT = "/home/issan/Projects/claude/wt/xbar_arrangement_1014/results/xbar_arrangement_1014"
FILES = {
    "LR0.1_s0": "review/opus_bwdfloor/out/LR_ref_s0.npz", "LR0.1_s1": "review/opus_bwdfloor/out/LR_ref_s1.npz",
    **{f"LR{a}_s{s}": f"review/opus_lrslope/out/LRa{a}_s{s}.npz" for a in ("0.03", "0.2", "0.3", "0.5") for s in (0, 1)},
    "ELU_s0": "review/opus_bwdfloor/out/ELU_ref_s0.npz", "ELU_s1": "review/opus_bwdfloor/out/ELU_ref_s1.npz",
}
PHASE2 = list(range(11, 101))
SAMPLED = [11, 21, 31, 41, 51, 61, 71, 81, 91, 100]
TRANS = list(range(0, 13))          # 0 = init (extra reference row), 1..12 as asked
R_MC = 50
R_SIMF = 10
_erfc = np.vectorize(math.erfc)

def pearson(a, b):
    a = a - a.mean(); b = b - b.mean(); den = math.sqrt(float((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / den) if den > 0 else float("nan")

def rankdata(a):
    n = len(a); order = np.argsort(a, kind="mergesort"); sa = a[order]
    new = np.r_[True, sa[1:] != sa[:-1]]; gid = np.cumsum(new) - 1
    cnt = np.bincount(gid); avg = np.cumsum(cnt) - (cnt - 1) / 2.0
    r = np.empty(n); r[order] = avg[gid]; return r

def spearman(a, b): return pearson(rankdata(a), rankdata(b))

def q(v, ps): v = np.asarray(v, float); v = v[np.isfinite(v)]; return [float(np.quantile(v, p)) for p in ps] if len(v) else [float("nan")] * len(ps)

def threshold_inputs(Z, logK, gate, units):
    """top-3 inputs per unit in `units` (columns of Z); returns (3, len(units)) indices, and a mask of usable units."""
    Zu = Z[:, units]
    if gate in ("leaky", "E1"):
        tau = 0.1 * Zu.std(0) + 1e-12; x = Zu / tau[None, :]; ax = np.abs(x)
        score = logK[:, None] - ax - 2.0 * np.log1p(np.exp(-ax)) - np.log(tau)[None, :]
        ok = np.ones(len(units), bool)
    elif gate == "E2":
        score = np.where(Zu < 0, logK[:, None] + Zu, -np.inf)
        ok = (Zu < 0).sum(0) >= 3
    else:                                                                    # "near": extra, not in the task: 3 inputs with smallest |z| (no K term)
        score = -np.abs(Zu); ok = np.ones(len(units), bool)
    T3 = np.argsort(-score, axis=0, kind="stable")[:3]
    return T3, ok

class Box:
    def __init__(self, tag):
        self.tag = tag; self.d = np.load(os.path.join(ROOT, FILES[tag])); d = self.d
        self.act = str(d["act"]); X = load_X(d["subset"]); self.Xt = np.c_[X, np.ones(len(X))]
        self.N = len(X); self.xbar = self.Xt.mean(0); self.K = self.Xt @ self.xbar; self.Kbar = float(self.xbar @ self.xbar)
        self.logK = np.log(self.K)
        edges = np.quantile(self.K, np.arange(1, 8) / 8.0); self.oct = np.searchsorted(edges, self.K, side="right")
        self.oct_count = np.bincount(self.oct, minlength=8); self.medK = float(np.median(self.K))
        self.oct_Kmean = [float(self.K[self.oct == o].mean()) for o in range(8)]
        self.gates = (["E1", "E2"] if self.act == "ELU" else ["leaky"]) + ["near"]
        self.chk = dict(m_max_abs=0.0, umean_max_abs=0.0, zbar_identity_max_abs=0.0)

    def state(self, T):
        d = self.d
        if T == 0: W = d["init0"].astype(np.float64); b = d["init1"].astype(np.float64)
        else: W = d["end0"][T - 1].astype(np.float64); b = d["end1"][T - 1].astype(np.float64)
        Wt = np.c_[W, b]; Z = self.Xt @ Wt.T; m = Wt @ self.xbar
        self.chk["m_max_abs"] = max(self.chk["m_max_abs"], float(np.abs(m - Z.mean(0)).max()))
        U = Z - np.outer(self.K / self.Kbar, m)
        self.chk["umean_max_abs"] = max(self.chk["umean_max_abs"], float(np.abs(U.mean(0)).max()))
        O = Z > 0; p = O.mean(0); alive = p > 0
        return dict(Z=Z, m=m, U=U, O=O, p=p, alive=alive, s=Z.std(0), sp=U.std(0))

    def basic(self, S, rng=None, R=0):
        """quantities used by A1/A2/A4 for one state"""
        K, Kbar, N = self.K, self.Kbar, self.N
        Z, m, U, O, p, alive, s, sp = (S[k] for k in ("Z", "m", "U", "O", "p", "alive", "s", "sp"))
        idx = np.flatnonzero(alive); H = len(idx); res = dict(H_alive=int(H), n_m_pos=int((m[idx] > 0).sum()))
        if H == 0: return res, None
        f = O[:, idx].mean(1)
        C = -np.outer(K / Kbar, m[idx])                                   # c_kn  (N, H)
        Ph = np.empty((N, H))
        for j, k in enumerate(idx):
            us = np.sort(U[:, k]); Ph[:, j] = (N - np.searchsorted(us, C[:, j], side="right")) / N
        Fhat = Ph.mean(1)
        FG = (0.5 * _erfc(C / sp[idx][None, :] / math.sqrt(2.0))).mean(1)
        Za = Z[:, idx]; zbar = Za.mean(1); med = np.median(Za, axis=1)
        mbar = float(m[idx].mean()); ubar = U[:, idx].mean(1)
        self.chk["zbar_identity_max_abs"] = max(self.chk["zbar_identity_max_abs"], float(np.abs(zbar - (mbar * K / Kbar + ubar)).max()))
        rho = ubar / (abs(mbar) * K / Kbar)
        r = m[idx] / s[idx]
        res.update(
            F_flat=float(p[idx].mean()), mean_f=float(f.mean()), max_f=float(f.max()), K_argmax_f=float(K[int(np.argmax(f))]),
            oct_argmax_f=int(self.oct[int(np.argmax(f))]), meanK_at_max_f=float(K[f == f.max()].mean()),
            n_f_ge_half=int((f >= 0.5).sum()),
            pearson_fK=pearson(f, K), spearman_fK=spearman(f, K), pearson_FhatK=pearson(Fhat, K),
            pearson_f_Fhat=pearson(f, Fhat), spearman_f_Fhat=spearman(f, Fhat), pearson_f_FG=pearson(f, FG),
            oct_f=[float(f[self.oct == o].mean()) for o in range(8)],
            oct_Fhat=[float(Fhat[self.oct == o].mean()) for o in range(8)],
            oct_FG=[float(FG[self.oct == o].mean()) for o in range(8)],
            max_Fhat=float(Fhat.max()), n_Fhat_ge_half=int((Fhat >= 0.5).sum()),
            n_zbar_pos=int((zbar > 0).sum()), n_median_pos=int((med > 0).sum()), mbar=float(mbar),
            max_zbar=float(zbar.max()), max_rho=float(rho.max()),
            sd_ubar_over_mean_sp=float(ubar.std() / sp[idx].mean()), inv_sqrtH=float(1 / math.sqrt(H)),
            sd_ubar_over_indep=float(ubar.std() / (math.sqrt(float((sp[idx] ** 2).sum())) / H)),
            n_p_gt_half=int((p[idx] > 0.5).sum()), mean_r=float(r.mean()), median_r=float(np.median(r)),
            mean_p=float(p[idx].mean()),
            Kopen_unitw=float(np.mean([(K * O[:, k]).sum() / O[:, k].sum() for k in idx])),
            Kopen_unitw_model=float(np.mean((K[:, None] * Ph).sum(0) / np.maximum(Ph.sum(0), 1e-300))),
            Kopen_pooled=float((K * f).sum() / f.sum()), Kopen_pooled_model=float((K * Fhat).sum() / Fhat.sum()),
        )
        if R > 0:   # f drawn from the model itself (independent permutation of each unit's u-perp over inputs): reference values
            acc = dict(pearson_fK=[], spearman_fK=[], pearson_f_Fhat=[], max_f=[], n_f_ge_half=[]); octs = []
            Ua = U[:, idx]
            for _ in range(R):
                perm = np.argsort(rng.random((N, H)), axis=0); fs = (np.take_along_axis(Ua, perm, axis=0) > C).mean(1)
                acc["pearson_fK"].append(pearson(fs, K)); acc["spearman_fK"].append(spearman(fs, K)); acc["pearson_f_Fhat"].append(pearson(fs, Fhat))
                acc["max_f"].append(float(fs.max())); acc["n_f_ge_half"].append(int((fs >= 0.5).sum())); octs.append([float(fs[self.oct == o].mean()) for o in range(8)])
            for k2, v2 in acc.items(): res["sim_" + k2] = float(np.mean(v2))
            res["sim_oct_f"] = np.mean(octs, 0).tolist()
        res["ratio_f_Fhat"] = [a / b if b > 0 else float("nan") for a, b in zip(res["oct_f"], res["oct_Fhat"])]
        res["ratio_f_FG"] = [a / b if b > 0 else float("nan") for a, b in zip(res["oct_f"], res["oct_FG"])]
        return res, dict(idx=idx, C=C, Ph=Ph)

    def a3(self, S):
        idx = np.flatnonzero(S["alive"]); Ua = S["U"][:, idx]; sp = S["sp"][idx]
        good = sp > 0; Ua = Ua[:, good]; H = Ua.shape[1]
        Cm = np.corrcoef(Ua.T); iu = np.triu_indices(H, 1); pc = Cm[iu]
        uk = np.array([pearson(Ua[:, j], self.K) for j in range(H)])
        Uc = Ua - Ua.mean(0); sd = Uc.std(0)
        sk = (Uc ** 3).mean(0) / sd ** 3; ku = (Uc ** 4).mean(0) / sd ** 4 - 3.0
        return pc, uk, sk, ku

    def a5(self, S, basic_aux, rng):
        K, Kbar, N = self.K, self.Kbar, self.N
        idx = basic_aux["idx"]; C = basic_aux["C"]; U = S["U"]; sp = S["sp"]; Z = S["Z"]; m = S["m"]
        # KDE (Gaussian, bandwidth 0.3 sp_k) of each unit's own u-perp, evaluated at c_kn
        G = np.empty((N, len(idx)))
        for j, k in enumerate(idx):
            h = 0.3 * sp[k]; D = (C[:, j][:, None] - U[None, :, k]) / h
            G[:, j] = np.exp(-0.5 * D * D).sum(1) / (N * h * math.sqrt(2 * math.pi))
        w_spec = G.mean(1)                                                   # (1/H) sum_k g_k(c_kn)  (task statement)
        Gn = G / np.maximum(G.sum(0, keepdims=True), 1e-300); w_unit = Gn.mean(1)          # each unit normalised (each gives 3 inputs)
        GK = G * K[:, None]; GKn = GK / np.maximum(GK.sum(0, keepdims=True), 1e-300); w_unitK = GKn.mean(1)
        def octfr(w): w = np.asarray(w, float); return [float(w[self.oct == o].sum() / w.sum()) for o in range(8)]
        def meanK(w): return float((w * K).sum() / w.sum())
        def below(w): return float(w[K < self.medK].sum() / w.sum())
        out = {}
        for gate in self.gates:
            T3, ok = threshold_inputs(Z, self.logK, gate, idx)
            sel = T3[:, ok].ravel(); cnt = np.bincount(self.oct[sel], minlength=8)
            # Monte Carlo of the model with the actual selection rule: per unit, independent permutation of u-perp over inputs
            mc_cnt = np.zeros(8); mc_n = 0; mc_meanK = []
            for _ in range(R_MC):
                perm = np.argsort(rng.random((N, len(idx))), axis=0)
                Us = np.take_along_axis(U[:, idx], perm, axis=0); Zs = -C + Us     # z_sim = m K/Kbar + u_perm
                T3s, oks = threshold_inputs(Zs, self.logK, gate, np.arange(len(idx)))
                ss = T3s[:, oks].ravel(); mc_cnt += np.bincount(self.oct[ss], minlength=8); mc_n += len(ss); mc_meanK.append(float(K[ss].mean()))
            out[gate] = dict(
                n_units=int(ok.sum()), n_excluded=int((~ok).sum()), n_inputs=int(len(sel)),
                cnt=cnt.tolist(), frac=(cnt / cnt.sum()).tolist(), meanK=float(K[sel].mean()), frac_below_medK=float((K[sel] < self.medK).mean()),
                n_distinct=int(len(np.unique(sel))),
                pred_spec=octfr(w_spec), pred_unit=octfr(w_unit), pred_unitK=octfr(w_unitK),
                pred_mc=(mc_cnt / mc_cnt.sum()).tolist(), mc_cnt=mc_cnt.tolist(), mc_n=int(mc_n),
                meanK_spec=meanK(w_spec), meanK_unit=meanK(w_unit), meanK_unitK=meanK(w_unitK), meanK_mc=float(np.mean(mc_meanK)),
                below_spec=below(w_spec), below_unit=below(w_unit), below_unitK=below(w_unitK),
            )
        return out

def summarize(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if not len(v): return dict(med=float("nan"), q10=float("nan"), q90=float("nan"), min=float("nan"), max=float("nan"), n=0)
    return dict(med=float(np.median(v)), q10=float(np.quantile(v, 0.1)), q90=float(np.quantile(v, 0.9)), min=float(v.min()), max=float(v.max()), n=int(len(v)))

def main(tag):
    t0 = time.time(); B = Box(tag); rng = np.random.default_rng(1014)
    res = dict(tag=tag, npz=FILES[tag], act=B.act, N=B.N, Kbar=B.Kbar,
               K=dict(min=float(B.K.min()), max=float(B.K.max()), mean=float(B.K.mean()), sd=float(B.K.std()), all_ge_1=bool((B.K >= 1).all()),
                      Kbar_minus_meanK=float(B.Kbar - B.K.mean()), median=B.medK),
               oct_count=B.oct_count.tolist(), oct_Kmean=B.oct_Kmean, sampled=SAMPLED, R_MC=R_MC)
    # A1 / A2 over phase 2
    per = {}
    aux_keep = {}
    for T in PHASE2:
        S = B.state(T); r, aux = B.basic(S, rng, R_SIMF); per[T] = r
        if T in SAMPLED: aux_keep[T] = (S, aux)
    keys_scalar = ["H_alive", "n_m_pos", "F_flat", "max_f", "K_argmax_f", "meanK_at_max_f", "n_f_ge_half", "pearson_fK", "spearman_fK", "pearson_FhatK",
                   "pearson_f_Fhat", "spearman_f_Fhat", "pearson_f_FG", "max_Fhat", "n_Fhat_ge_half", "n_zbar_pos", "n_median_pos", "mbar", "max_zbar",
                   "max_rho", "sd_ubar_over_mean_sp", "inv_sqrtH", "sd_ubar_over_indep", "n_p_gt_half", "mean_r", "median_r",
                   "Kopen_unitw", "Kopen_unitw_model", "Kopen_pooled", "Kopen_pooled_model",
                   "sim_pearson_fK", "sim_spearman_fK", "sim_pearson_f_Fhat", "sim_max_f", "sim_n_f_ge_half"]
    res["phase2_per_state"] = {str(T): per[T] for T in PHASE2}
    res["phase2_summary"] = {k: summarize([per[T][k] for T in PHASE2 if k in per[T]]) for k in keys_scalar}
    res["phase2_summary"]["sd_ubar_x_sqrtH"] = summarize([per[T]["sd_ubar_over_mean_sp"] / per[T]["inv_sqrtH"] for T in PHASE2])
    res["phase2_summary"]["n_states_any_f_ge_half"] = int(sum(per[T]["n_f_ge_half"] > 0 for T in PHASE2))
    res["phase2_summary"]["n_states_any_zbar_pos"] = int(sum(per[T]["n_zbar_pos"] > 0 for T in PHASE2))
    res["phase2_summary"]["n_states_any_median_pos"] = int(sum(per[T]["n_median_pos"] > 0 for T in PHASE2))
    for k in ("oct_f", "oct_Fhat", "oct_FG", "ratio_f_Fhat", "ratio_f_FG", "sim_oct_f"):
        M = np.array([per[T][k] for T in PHASE2], float)
        res["phase2_summary"][k] = dict(med=np.nanmedian(M, 0).tolist(), q10=np.nanquantile(M, 0.1, axis=0).tolist(), q90=np.nanquantile(M, 0.9, axis=0).tolist())
    # A3 on sampled states (+ init reference)
    a3 = {}
    pools = dict(pc=[], uk=[], sk=[], ku=[])
    for T in [0] + SAMPLED:
        S = B.state(T) if T == 0 else aux_keep[T][0]
        pc, uk, sk, ku = B.a3(S)
        a3[str(T)] = dict(n_pairs=int(len(pc)), mean=float(pc.mean()), sd=float(pc.std()), q01=float(np.quantile(pc, 0.01)), q99=float(np.quantile(pc, 0.99)),
                          frac_abs_gt_half=float((np.abs(pc) > 0.5).mean()), mean_abs=float(np.abs(pc).mean()),
                          uK_q=q(uk, (0.1, 0.5, 0.9)), skew_q=q(sk, (0.1, 0.5, 0.9)), exkurt_q=q(ku, (0.1, 0.5, 0.9)))
        if T != 0:
            pools["pc"].append(pc); pools["uk"].append(uk); pools["sk"].append(sk); pools["ku"].append(ku)
    pc = np.concatenate(pools["pc"]); uk = np.concatenate(pools["uk"]); sk = np.concatenate(pools["sk"]); ku = np.concatenate(pools["ku"])
    a3["pooled_sampled"] = dict(n_pairs=int(len(pc)), mean=float(pc.mean()), sd=float(pc.std()), q01=float(np.quantile(pc, 0.01)), q99=float(np.quantile(pc, 0.99)),
                                frac_abs_gt_half=float((np.abs(pc) > 0.5).mean()), mean_abs=float(np.abs(pc).mean()),
                                uK_q=q(uk, (0.1, 0.5, 0.9)), skew_q=q(sk, (0.1, 0.5, 0.9)), exkurt_q=q(ku, (0.1, 0.5, 0.9)), n_units=int(len(uk)))
    res["A3"] = a3
    # A4 transition
    a4 = {}
    for T in TRANS:
        S = B.state(T); r, _ = B.basic(S)
        a4[str(T)] = {k: r.get(k) for k in ("H_alive", "n_p_gt_half", "n_f_ge_half", "n_zbar_pos", "n_median_pos", "mean_r", "median_r", "mean_p", "max_f", "mbar", "n_m_pos", "pearson_fK")}
    res["A4"] = a4
    # A5 on sampled states
    a5 = {}
    for T in SAMPLED:
        S, aux = aux_keep[T]; a5[str(T)] = B.a5(S, aux, rng)
        a5[str(T)]["Kopen_unitw"] = per[T]["Kopen_unitw"]; a5[str(T)]["Kopen_unitw_model"] = per[T]["Kopen_unitw_model"]
    pooled = {}
    for gate in B.gates:
        cnt = np.sum([a5[str(T)][gate]["cnt"] for T in SAMPLED], 0); nu = np.array([a5[str(T)][gate]["n_units"] for T in SAMPLED], float)
        def wavg(key): return (np.array([a5[str(T)][gate][key] for T in SAMPLED], float) * nu[:, None]).sum(0) / nu.sum()
        def wavs(key): return float((np.array([a5[str(T)][gate][key] for T in SAMPLED], float) * nu).sum() / nu.sum())
        mc = np.sum([a5[str(T)][gate]["mc_cnt"] for T in SAMPLED], 0)
        pooled[gate] = dict(n_inputs=int(cnt.sum()), frac=(cnt / cnt.sum()).tolist(),
                            meanK=wavs("meanK"), frac_below_medK=wavs("frac_below_medK"),
                            pred_spec=wavg("pred_spec").tolist(), pred_unit=wavg("pred_unit").tolist(), pred_unitK=wavg("pred_unitK").tolist(),
                            pred_mc=(mc / mc.sum()).tolist(),
                            meanK_spec=wavs("meanK_spec"), meanK_unit=wavs("meanK_unit"), meanK_unitK=wavs("meanK_unitK"), meanK_mc=wavs("meanK_mc"),
                            below_spec=wavs("below_spec"), below_unit=wavs("below_unit"), below_unitK=wavs("below_unitK"))
    pooled["Kopen_unitw"] = float(np.mean([per[T]["Kopen_unitw"] for T in SAMPLED])); pooled["Kopen_unitw_model"] = float(np.mean([per[T]["Kopen_unitw_model"] for T in SAMPLED]))
    pooled["meanK"] = float(B.K.mean())
    res["A5"] = dict(per_state=a5, pooled=pooled, kde="Gaussian KDE, bandwidth 0.3*sd_n(u_perp_k), evaluated at c_kn = -m_k K_n/Kbar")
    res["checks"] = B.chk; res["elapsed_s"] = time.time() - t0
    json.dump(res, open(os.path.join(OUT, f"a_occ_{tag}.json"), "w"), indent=1)
    print(tag, "done", round(time.time() - t0, 1), "s", json.dumps(B.chk))

if __name__ == "__main__":
    main(sys.argv[1])
