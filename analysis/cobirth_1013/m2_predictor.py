#!/usr/bin/env python
"""M2 (cobirth_1013): does readout alignment x open-set overlap predict co-presence / co-birth of lifters?

Box: 784-100-10 MLP, hard leaky ReLU slope a in {0.1, 0.3, 0.5}, CE, Adam, batch 16, 4000 updates/task, iid uniform labels
redrawn each task, 100 tasks, seed 0. Saved task-end weights (push_direction_proof_0930): end0..end3 [100, ...].
Indexing: checkpoint e (0-based) = end0[e] = end of task e+1 (1-based). Phase 2 here = task ends 11..100 = e in 10..99 (90 states).
(The H4/P161/stoch scripts call end0[t-2] "task t"; their t = 10..100 is e = 8..98. We reproduce their row counts as a check.)

Lifter criterion (copied verbatim in spirit from stage14_1007_h4/agent_stoch/stoch_bridges.py bridges_at(); same as P158/P161/P176):
  Z = Xt W^T (Xt = [X, 1]), tau = 0.1 sd_n(z_i) + 1e-12, s = sigma(z/tau), Dp = (1-a) s (1-s)/tau, K_n = Xt . mean(Xt),
  E = K_n Dp_in, T_i = top-3 inputs of E[:, i], O_i = {n: z_in > 0}, k_i = |O_i|, need_i = (k_i+1)//2;
  j is a lifter of i  <=>  j != i, both alive, |O_j ∩ T_i| >= 2 and |O_j ∩ O_i| >= need_i.        L_ij(t) = indicator.

Predictor (THEORY_FABLE.md lemma 1-2, exact for one full-batch SGD step with fresh labels):
  Cov_Y(dz_jn, dz_kn') = (eta^2/(N^2 K)) <v_j^c, v_k^c> sum_m Theta_nm Theta_n'm d_jm d_km,     d = hard gate (a or 1)
  g_jk(T_i) = sum_{n in T_i} sum_m Theta_nm^2 d_jm d_km,  omega_jk(i) = g_jk / sqrt(g_jj g_kk),  rho_jk(i) = cos(v_j^c, v_k^c) omega_jk(i).

Outputs: results/cobirth_1013/m2_predictor.json (summary), obsidian-research-data/cobirth_1013/m2/ (per-arm npz with L, births,
cos_t, J_t, time-averaged |rho|, omega, pair tables, MC check arrays).
usage: m2_predictor.py [--arms a0.1 a0.3 a0.5] [--perms 2000] [--mc 2000] [--quick]
"""
import os, sys, json, math, time, argparse
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
import numpy as np

ROOT = "/home/issan/Projects/obsidian-research-data/push_direction_proof_0930"
REPLAY = "/home/issan/Projects/obsidian-research-data/cumulative_push_1013/replay"
WT = "/home/issan/Projects/claude/wt/cobirth_1013"
OUT_JSON = os.path.join(WT, "results", "cobirth_1013", "m2_predictor.json")
OUT_RAW = "/home/issan/Projects/obsidian-research-data/cobirth_1013/m2"
sys.path.insert(0, ROOT)
from common import load_X  # noqa: E402

ARMS = {"a0.1": ("review/opus_bwdfloor/out/LR_ref_s0.npz", 0.1), "a0.3": ("review/opus_lrslope/out/LRa0.3_s0.npz", 0.3), "a0.5": ("review/opus_lrslope/out/LRa0.5_s0.npz", 0.5)}
K, LR, B1, B2, EPS = 10, 1e-3, 0.9, 0.999, 1e-8
E_ALL = list(range(8, 100))          # states we compute (covers stoch t=10..100 and our phase 2 + one before for births)
E_PH2 = list(range(10, 100))         # phase 2: task ends 11..100
E_STOCH = list(range(8, 99))         # stoch convention t=10..100 -> e=8..98
E_MC = 39                            # task-40 end state
EDGES = {"rho": [0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0001], "cos": [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0001], "J": [0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0001], "omega": [0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0001],
         "rho_s": [-1.0001, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 1.0001], "cos_s": [-1.0001, -0.5, -0.3, -0.1, 0, 0.1, 0.3, 0.5, 1.0001]}
PREDS = ("rho", "cos", "J", "omega", "rho_s", "cos_s")
P = lambda *x: print(*x, flush=True)


def softmax(l):
    l = l - l.max(1, keepdims=True); e = np.exp(l); return e / e.sum(1, keepdims=True)


def rankdata(x):
    x = np.asarray(x, float); n = len(x); order = np.argsort(x, kind="mergesort"); r = np.empty(n); r[order] = np.arange(1, n + 1)
    u, inv, cnt = np.unique(x, return_inverse=True, return_counts=True); s = np.bincount(inv, weights=r); return s[inv] / cnt[inv]


def pearson(a, b):
    a = a - a.mean(); b = b - b.mean(); d = math.sqrt(float((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / d) if d > 0 else float("nan")


def spearman_perm(pred, y, grp, nperm, rng):
    """Spearman(pred, y); permutation p-values for shuffling pred labels globally and within group (target)."""
    rp, ry = rankdata(pred), rankdata(y); obs = pearson(rp, ry)
    if not np.isfinite(obs) or nperm == 0: return dict(rho=obs, p_global=None, p_within=None, nperm=nperm)
    cg = cw = 0; n = len(rp)
    gs = np.argsort(grp, kind="mergesort"); grp_s = grp[gs]; rp_s = rp[gs]; ry_s = ry[gs]
    for _ in range(nperm):
        cg += pearson(rp[rng.permutation(n)], ry) >= obs
        key = rng.random(n); order = np.lexsort((key, grp_s)); cw += pearson(rp_s[order], ry_s) >= obs
    return dict(rho=obs, p_global=(cg + 1) / (nperm + 1), p_within=(cw + 1) / (nperm + 1), nperm=nperm)


def state(dd, e, a, Xt):
    W = np.concatenate([dd["end0"][e].astype(np.float64), dd["end1"][e].astype(np.float64)[:, None]], 1)
    Z = Xt @ W.T; tau = 0.1 * Z.std(0) + 1e-12; s = 0.5 * (1 + np.tanh(0.5 * Z / tau)); Dp = (1 - a) * s * (1 - s) / tau
    O = Z > 0; Dh = np.where(O, 1.0, a); W2 = dd["end2"][e].astype(np.float64); b2 = dd["end3"][e].astype(np.float64)
    Vc = W2 - W2.mean(0, keepdims=True)
    return dict(W=W, Z=Z, tau=tau, Dp=Dp, O=O, Dh=Dh, W2=W2, b2=b2, Vc=Vc)


def lifters(O, Dp, Kv):
    """P158/P161/P176 loose criterion, as in stoch_bridges.py bridges_at(). isb[i, j] = j is a lifter of target i."""
    H = O.shape[1]; alive = O.any(0); k = O.sum(0); need = (k + 1) // 2; Oi = O.astype(np.int32); ov = Oi.T @ Oi
    E = Kv[:, None] * Dp; T = np.argsort(-E, axis=0)[:3, :]; thr = np.stack([O[T[:, i], :].sum(0) for i in range(H)], 0)
    isb = (thr >= 2) & (ov >= need[:, None]); isb[np.diag_indices(H)] = False; isb[~alive, :] = False; isb[:, ~alive] = False
    return isb, T, k, alive, ov


def cos_matrix(Vc):
    nv = np.linalg.norm(Vc, axis=0); return (Vc.T @ Vc) / np.maximum(nv[:, None] * nv[None, :], 1e-300)


def jaccard(ov, k):
    un = k[:, None] + k[None, :] - ov; return np.where(un > 0, ov / np.maximum(un, 1), 0.0)


def pair_table(Lt, abs_rho_bar, om_bar, cos_bar, J_bar, cnt_i, n_i_t, rho_s_bar, cos_s_bar):
    """Lt: [T, H, H] bool (time, target i, unit j). Returns dict of flat arrays over (i, j<k) eligible pairs."""
    T, H, _ = Lt.shape; cols = {c: [] for c in ("i", "j", "k", "C", "Lj", "Lk", "exc", "exc_alive", "exc_state", "rho", "omega", "cos", "J", "rho_s", "cos_s")}
    iu = np.triu_indices(H, 1)
    for i in range(H):
        Li = Lt[:, i, :].astype(np.float64); Ls = Li.sum(0)
        if Ls.max() == 0: continue
        C = Li.T @ Li; exc = C - np.outer(Ls, Ls) / T
        Ti = max(float(cnt_i[i]), 1.0); exc_alive = C - np.outer(Ls, Ls) / Ti
        nt = n_i_t[:, i].astype(np.float64); f = Ls / max(nt.sum(), 1.0); exc_state = C - float((nt * (nt - 1)).sum()) * np.outer(f, f)
        elig = (Ls[:, None] >= 1) & (Ls[None, :] >= 1); elig[i, :] = False; elig[:, i] = False
        m = elig[iu]
        if m.sum() == 0: continue
        jj, kk = iu[0][m], iu[1][m]
        cols["i"].append(np.full(len(jj), i)); cols["j"].append(jj); cols["k"].append(kk); cols["C"].append(C[jj, kk]); cols["Lj"].append(Ls[jj]); cols["Lk"].append(Ls[kk])
        cols["exc"].append(exc[jj, kk]); cols["exc_alive"].append(exc_alive[jj, kk]); cols["exc_state"].append(exc_state[jj, kk])
        cols["rho"].append(abs_rho_bar[i][jj, kk]); cols["omega"].append(om_bar[i][jj, kk]); cols["cos"].append(cos_bar[jj, kk]); cols["J"].append(J_bar[jj, kk])
        cols["rho_s"].append(rho_s_bar[i][jj, kk]); cols["cos_s"].append(cos_s_bar[jj, kk])
    return {c: (np.concatenate(v) if v else np.zeros(0)) for c, v in cols.items()}


def bin_stats(pred, tab, edges):
    b = np.digitize(pred, edges) - 1; out = []
    for q in range(len(edges) - 1):
        m = b == q
        rec = dict(bin=f"[{max(edges[q], -1):g},{min(edges[q+1], 1):g})", n_pairs=int(m.sum()))
        if m.sum():
            rec.update(mean_exc=float(tab["exc"][m].mean()), mean_exc_state=float(tab["exc_state"][m].mean()), mean_exc_alive=float(tab["exc_alive"][m].mean()),
                       mean_C=float(tab["C"][m].mean()), sum_C=int(tab["C"][m].sum()), frac_C_pos=float((tab["C"][m] > 0).mean()), mean_pred=float(pred[m].mean()))
        out.append(rec)
    return out


def analyze_table(tab, label, nperm, rng):
    n = len(tab["i"])
    res = dict(label=label, n_pairs=int(n), n_targets=int(len(np.unique(tab["i"]))) if n else 0, total_C=int(tab["C"].sum()) if n else 0,
               n_pairs_with_C_pos=int((tab["C"] > 0).sum()) if n else 0, mean_exc=float(tab["exc"].mean()) if n else None,
               mean_exc_state=float(tab["exc_state"].mean()) if n else None, mean_exc_alive=float(tab["exc_alive"].mean()) if n else None)
    if n < 10: return res
    res["bins"] = {p: bin_stats(tab[p], tab, EDGES[p]) for p in PREDS}
    res["spearman"] = {}
    for p in PREDS:
        for yk in ("exc", "exc_state", "exc_alive", "C"):
            res["spearman"][f"{p}~{yk}"] = spearman_perm(tab[p], tab[yk], tab["i"], nperm if yk in ("exc", "exc_state") else 0, rng)
    # predictor correlations among themselves (diagnostic)
    res["pred_corr_spearman"] = {"rho~cos": pearson(rankdata(tab["rho"]), rankdata(tab["cos"])), "rho~J": pearson(rankdata(tab["rho"]), rankdata(tab["J"])),
                                 "rho~omega": pearson(rankdata(tab["rho"]), rankdata(tab["omega"])), "cos~J": pearson(rankdata(tab["cos"]), rankdata(tab["J"])), "omega~J": pearson(rankdata(tab["omega"]), rankdata(tab["J"]))}
    # rank-based partial: rho given cos, and cos given J (Spearman partial correlation)
    def partial(x, y, z):
        rx, ry, rz = rankdata(x), rankdata(y), rankdata(z); rxy, rxz, ryz = pearson(rx, ry), pearson(rx, rz), pearson(ry, rz)
        d = math.sqrt(max((1 - rxz ** 2) * (1 - ryz ** 2), 1e-300)); return float((rxy - rxz * ryz) / d)
    res["partial_spearman"] = {"rho~exc|cos": partial(tab["rho"], tab["exc"], tab["cos"]), "rho~exc|J": partial(tab["rho"], tab["exc"], tab["J"]), "omega~exc|J": partial(tab["omega"], tab["exc"], tab["J"]), "omega~exc_state|J": partial(tab["omega"], tab["exc_state"], tab["J"]),
                               "cos~exc|J": partial(tab["cos"], tab["exc"], tab["J"]), "J~exc|cos": partial(tab["J"], tab["exc"], tab["cos"]),
                               "rho~exc_state|cos": partial(tab["rho"], tab["exc_state"], tab["cos"]), "rho~exc_state|J": partial(tab["rho"], tab["exc_state"], tab["J"])}
    # quantile split: top decile of |rho| vs rest
    q90 = float(np.quantile(tab["rho"], 0.9)); hi = tab["rho"] >= q90
    res["top_decile_rho"] = dict(threshold=q90, n_hi=int(hi.sum()), mean_exc_hi=float(tab["exc"][hi].mean()), mean_exc_lo=float(tab["exc"][~hi].mean()),
                                 mean_exc_state_hi=float(tab["exc_state"][hi].mean()), mean_exc_state_lo=float(tab["exc_state"][~hi].mean()),
                                 share_C_hi=float(tab["C"][hi].sum() / max(tab["C"].sum(), 1)))
    return res


def mc_sgd_check(dd, e, a, Xt, Theta, Vc, Dh, W2, b2, Z, T, targets, R, rng):
    """Monte Carlo over fresh labels of one full-batch SGD step (eta = LR) vs exact covariance formula, at inputs in targets' T_i."""
    N, H = Z.shape; A = np.where(Z > 0, Z, a * Z); p = softmax(A @ W2.T + b2); PV = p @ W2
    sel = sorted(set(int(n) for i in targets for n in T[:, i])); S = len(sel); Th = Theta[sel]
    dz = np.zeros((R, S, H)); ch = 100
    for r0 in range(0, R, ch):
        r = min(ch, R - r0); Y = rng.integers(0, K, size=(r, N)); C = PV[None, :, :] - W2[Y]; M = Dh[None, :, :] * C
        dz[r0:r0 + r] = -(LR / N) * np.einsum("sn,rnh->rsh", Th, M)
    coef = LR ** 2 / (N ** 2 * K); VV = Vc.T @ Vc; out = dict(R=R, inputs=sel, targets=[int(i) for i in targets], same_input=[], cross_input=[])
    mean_f = -(LR / N) * Th @ (Dh * (PV - W2.mean(0)[None, :]))
    for si, n in enumerate(sel):
        X_ = dz[:, si, :]; cov_mc = np.cov(X_, rowvar=False); G = Dh.T @ ((Theta[n] ** 2)[:, None] * Dh); cov_f = coef * VV * G
        sd_f = np.sqrt(np.diag(cov_f)); nrm = np.outer(sd_f, sd_f); err_corr = (cov_mc - cov_f) / nrm; corr_f = cov_f / nrm
        off = ~np.eye(H, dtype=bool); rel = np.abs(cov_mc - cov_f) / np.maximum(np.abs(cov_f), 1e-300)
        big = off & (np.abs(corr_f) >= 0.1)
        out["same_input"].append(dict(n=int(n), max_abs_err_corr_units=float(np.abs(err_corr[off]).max()), rms_err_corr_units=float(np.sqrt((err_corr[off] ** 2).mean())),
                                      max_rel_err_diag=float(np.max(np.abs(np.diag(cov_mc) - np.diag(cov_f)) / np.diag(cov_f))),
                                      median_rel_err_offdiag=float(np.median(rel[off])), max_rel_err_offdiag_all=float(rel[off].max()),
                                      max_rel_err_offdiag_corr_ge_0p1=float(rel[big].max()) if big.any() else None, n_pairs_corr_ge_0p1=int(big.sum() // 2),
                                      median_rel_err_offdiag_corr_ge_0p1=float(np.median(rel[big])) if big.any() else None,
                                      max_abs_mean_err_over_sd=float(np.max(np.abs(X_.mean(0) - mean_f[si]) / sd_f)), abs_corr_f_quantiles=[float(np.quantile(np.abs(corr_f[off]), q)) for q in (0.5, 0.9, 0.99)]))
    # cross-input covariances within the same target's threshold set
    for i in targets:
        ns = [int(n) for n in T[:, i]]
        for u in range(3):
            for w in range(u + 1, 3):
                n1, n2 = ns[u], ns[w]; s1, s2 = sel.index(n1), sel.index(n2)
                X1 = dz[:, s1, :] - dz[:, s1, :].mean(0); X2 = dz[:, s2, :] - dz[:, s2, :].mean(0); cov_mc = X1.T @ X2 / (R - 1)
                G = Dh.T @ ((Theta[n1] * Theta[n2])[:, None] * Dh); cov_f = coef * VV * G
                sd1 = np.sqrt(np.diag(coef * VV * (Dh.T @ ((Theta[n1] ** 2)[:, None] * Dh)))); sd2 = np.sqrt(np.diag(coef * VV * (Dh.T @ ((Theta[n2] ** 2)[:, None] * Dh))))
                err = (cov_mc - cov_f) / np.outer(sd1, sd2); rel = np.abs(cov_mc - cov_f) / np.maximum(np.abs(cov_f), 1e-300); cf = cov_f / np.outer(sd1, sd2); big = np.abs(cf) >= 0.1
                out["cross_input"].append(dict(target=int(i), n1=n1, n2=n2, max_abs_err_corr_units=float(np.abs(err).max()), median_rel_err=float(np.median(rel)),
                                               max_rel_err_corr_ge_0p1=float(rel[big].max()) if big.any() else None, n_corr_ge_0p1=int(big.sum())))
    out["mc_sd_corr_units_expected"] = 1 / math.sqrt(R)
    out["summary"] = dict(max_abs_err_corr_units_same=float(max(r["max_abs_err_corr_units"] for r in out["same_input"])),
                          max_rel_err_diag=float(max(r["max_rel_err_diag"] for r in out["same_input"])),
                          max_rel_err_offdiag_corr_ge_0p1=float(max(r["max_rel_err_offdiag_corr_ge_0p1"] for r in out["same_input"] if r["max_rel_err_offdiag_corr_ge_0p1"] is not None)),
                          median_rel_err_offdiag=float(np.median([r["median_rel_err_offdiag"] for r in out["same_input"]])),
                          max_abs_err_corr_units_cross=float(max(r["max_abs_err_corr_units"] for r in out["cross_input"])) if out["cross_input"] else None)
    return out, dz, sel


def mc_adam_check(dd, e, a, Xt, Vc, Dh, W2, b2, Z, T, targets, R, rng):
    """First Adam step of task e+2 (state = end of task e+1, Adam state from replay task_{e+1}.npz, batch from task_{e+2}.npz) with fresh
    labels on the 16 batch inputs: MC covariance of dz_jn across units vs the linearized formula with per-unit preconditioned Gram."""
    tagdir = {0.1: "a0.1", 0.3: "a0.3", 0.5: "a0.5"}[a]; f0 = os.path.join(REPLAY, tagdir, f"task_{e+1:03d}.npz"); f1 = os.path.join(REPLAY, tagdir, f"task_{e+2:03d}.npz")
    if not (os.path.exists(f0) and os.path.exists(f1)): return dict(available=False)
    d0, d1 = np.load(f0), np.load(f1); m0 = np.concatenate([d0["adam_m_0"].astype(np.float64), d0["adam_m_1"].astype(np.float64)[:, None]], 1)
    v0 = np.concatenate([d0["adam_v_0"].astype(np.float64), d0["adam_v_1"].astype(np.float64)[:, None]], 1); k = int(d0["adam_step"]) + 1
    batch = d1["first_batch_idx"].astype(int); y1 = dd["y"][e + 1].astype(int); N, H = Z.shape
    A = np.where(Z > 0, Z, a * Z); p = softmax(A @ W2.T + b2); PV = p @ W2; nb = len(batch)
    # check our gradient against the replayed gradient of the real first batch (labels of task e+2)
    Cb = PV[batch] - W2[y1[batch]]; g_W = (Dh[batch] * Cb).T @ Xt[batch] / nb; g_rep = np.concatenate([d1["g1_0"].astype(np.float64), d1["g1_1"].astype(np.float64)[:, None]], 1)
    grad_rel = float(np.abs(g_W - g_rep).max() / np.abs(g_rep).max())
    adam = lambda g: -(LR / (1 - B1 ** k)) * (B1 * m0 + (1 - B1) * g) / (np.sqrt(B2 * v0 + (1 - B2) * g * g) / math.sqrt(1 - B2 ** k) + EPS)
    dth_rep = np.concatenate([d1["dtheta1_0"].astype(np.float64), d1["dtheta1_1"].astype(np.float64)[:, None]], 1); dth_rel = float(np.abs(adam(g_rep) - dth_rep).max() / np.abs(dth_rep).max())
    sel = sorted(set(int(n) for i in targets for n in T[:, i])); S = len(sel); Xs = Xt[sel]
    dz = np.zeros((R, S, H)); ch = 100
    for r0 in range(0, R, ch):
        r = min(ch, R - r0); Y = rng.integers(0, K, size=(r, nb)); C = PV[None, batch, :] - W2[Y]; M = Dh[None, batch, :] * C   # r x nb x H
        g = np.einsum("rmh,mc->rhc", M, Xt[batch]) / nb                                                                      # r x H x 785
        dw = adam(g); dz[r0:r0 + r] = np.einsum("sc,rhc->rsh", Xs, dw)
    # linearized formula: dz_jn ~ const - lr' sum_c Xt[n,c] g_jc / (sqrt(v0_jc/(1-B2^k)) + EPS), lr' = LR (1-B1)/(1-B1^k)
    pre = 1.0 / (np.sqrt(v0 / (1 - B2 ** k)) + EPS); lrp = LR * (1 - B1) / (1 - B1 ** k); VV = Vc.T @ Vc
    out = dict(available=True, R=R, k=k, batch=[int(b) for b in batch], inputs=sel, grad_check_rel=grad_rel, dtheta_check_rel=dth_rel,
               linear_regime=dict(median_sqrt_v0_over_absg=float(np.median(np.sqrt(v0 / (1 - B2 ** k)) / np.maximum(np.abs(g_rep), 1e-30))),
                                  median_newg2_over_b2v0=float(np.median((1 - B2) * g_rep ** 2 / np.maximum(B2 * v0, 1e-300)))), same_input=[])
    for si, n in enumerate(sel):
        X_ = dz[:, si, :]; cov_mc = np.cov(X_, rowvar=False)
        Tt = np.einsum("c,mc,hc->hm", Xt[n], Xt[batch], pre)                      # H x nb : Theta~^{(j)}_{nm}
        Dm = Dh[batch].T                                                           # H x nb
        G = (Dm * Tt) @ (Dm * Tt).T                                                # sum_m d_jm d_km Tt_jm Tt_km
        cov_f = (lrp ** 2 / (nb ** 2 * K)) * VV * G; sd_f = np.sqrt(np.diag(cov_f)); nrm = np.outer(sd_f, sd_f); err = (cov_mc - cov_f) / nrm
        off = ~np.eye(H, dtype=bool); rel = np.abs(cov_mc - cov_f) / np.maximum(np.abs(cov_f), 1e-300); cf = cov_f / nrm; big = off & (np.abs(cf) >= 0.1)
        # SGD-form formula with the plain Gram (what would be wrong if preconditioning mattered)
        G0 = Dm @ np.diag(Xt[n] @ Xt[batch].T) @ Dm.T if False else (Dm * (Xt[batch] @ Xt[n])[None, :]) @ (Dm * (Xt[batch] @ Xt[n])[None, :]).T
        out["same_input"].append(dict(n=int(n), max_abs_err_corr_units=float(np.abs(err[off]).max()), rms_err_corr_units=float(np.sqrt((err[off] ** 2).mean())),
                                      max_rel_err_diag=float(np.max(np.abs(np.diag(cov_mc) - np.diag(cov_f)) / np.diag(cov_f))), median_rel_err_diag=float(np.median(np.abs(np.diag(cov_mc) - np.diag(cov_f)) / np.diag(cov_f))),
                                      median_rel_err_offdiag=float(np.median(rel[off])), max_rel_err_offdiag_corr_ge_0p1=float(rel[big].max()) if big.any() else None, n_pairs_corr_ge_0p1=int(big.sum() // 2),
                                      corr_mc_vs_corr_f_pearson_offdiag=pearson((cov_mc / np.outer(np.sqrt(np.diag(cov_mc)), np.sqrt(np.diag(cov_mc))))[off], cf[off]),
                                      corr_f_vs_sgdgram_corr_pearson_offdiag=pearson(cf[off], ((VV * G0) / np.outer(np.sqrt(np.diag(VV * G0)), np.sqrt(np.diag(VV * G0))))[off])))
    out["mc_sd_corr_units_expected"] = 1 / math.sqrt(R)
    out["summary"] = dict(max_abs_err_corr_units=float(max(r["max_abs_err_corr_units"] for r in out["same_input"])), max_rel_err_diag=float(max(r["max_rel_err_diag"] for r in out["same_input"])),
                          median_rel_err_diag=float(np.median([r["median_rel_err_diag"] for r in out["same_input"]])), median_rel_err_offdiag=float(np.median([r["median_rel_err_offdiag"] for r in out["same_input"]])),
                          max_rel_err_offdiag_corr_ge_0p1=float(max(r["max_rel_err_offdiag_corr_ge_0p1"] for r in out["same_input"] if r["max_rel_err_offdiag_corr_ge_0p1"] is not None)))
    return out


def run_arm(tag, args):
    t0 = time.time(); rel, a = ARMS[tag]; dd = np.load(os.path.join(ROOT, rel)); X = load_X(dd["subset"]); N = X.shape[0]
    Xt = np.concatenate([X, np.ones((N, 1))], 1); xbar = Xt.mean(0); Kv = Xt @ xbar; Theta = Xt @ Xt.T; Q = Theta ** 2; H = dd["end0"].shape[1]
    rng = np.random.default_rng(1013)
    nE = len(E_ALL); Lall = np.zeros((nE, H, H), bool); Tall = np.zeros((nE, 3, H), np.int64); kall = np.zeros((nE, H), np.int64); alive_all = np.zeros((nE, H), bool)
    cos_t = np.zeros((nE, H, H)); J_t = np.zeros((nE, H, H)); abs_rho_sum = np.zeros((H, H, H)); om_sum = np.zeros((H, H, H)); cnt_i = np.zeros(H); rho_s_sum = np.zeros((H, H, H))
    bc_keys = ("abs_cos", "cos", "J", "abs_rho", "rho", "omega"); bcond = {g: {k_: 0.0 for k_ in bc_keys} | {"n": 0, "events": 0} for g in ("coborn", "alive", "cross_born")}
    heavy = []  # states with n_i >= 7: predictor values among lifter pairs vs alive pairs
    clone = dict(copres_total=0, copres_clone=0, copres_cos=0, copres_J=0, copres_both=0, base_pairs=0, base_clone=0, base_cos=0, base_J=0)
    clone_b = dict(cob_total=0, cob_clone=0, cob_cos=0, cob_J=0); cos_pool = []; J_pool = []; alive_count_ph2 = []
    prev_isb = None; iu = np.triu_indices(H, 1)
    for ei, e in enumerate(E_ALL):
        st = state(dd, e, a, Xt); isb, T, k, alive, ov = lifters(st["O"], st["Dp"], Kv); Lall[ei] = isb; Tall[ei] = T; kall[ei] = k; alive_all[ei] = alive
        cos = cos_matrix(st["Vc"]); Jm = jaccard(ov, k); cos_t[ei] = cos; J_t[ei] = Jm; D = st["Dh"]
        in_ph2 = e in E_PH2
        if in_ph2:
            rho_i_cache = {}
            for i in np.where(alive)[0]:
                q = Q[T[:, i]].sum(0); G = (D * q[:, None]).T @ D; dg = np.sqrt(np.diag(G)); om = G / (dg[:, None] * dg[None, :]); rho = cos * om
                abs_rho_sum[i] += np.abs(rho); om_sum[i] += om; cnt_i[i] += 1; rho_s_sum[i] += rho
                rho_i_cache[i] = (np.abs(rho), om, rho)
            # (3) clone-like share among co-presences (instantaneous cos / J), baseline among alive pairs
            ac = np.abs(cos) > 0.5; aj = Jm > 0.2; am = alive[:, None] & alive[None, :]; bp = am[iu]
            clone["base_pairs"] += int(bp.sum()); clone["base_clone"] += int((ac | aj)[iu][bp].sum()); clone["base_cos"] += int(ac[iu][bp].sum()); clone["base_J"] += int(aj[iu][bp].sum())
            cos_pool.append(np.abs(cos)[iu][bp].astype(np.float32)); J_pool.append(Jm[iu][bp].astype(np.float32)); alive_count_ph2.append(int(alive.sum()))
            born = isb & ~prev_isb if prev_isb is not None else np.zeros_like(isb)
            for i in np.where(isb.any(1))[0]:
                js = np.where(isb[i])[0]
                if len(js) >= 2:
                    jj, kk = np.triu_indices(len(js), 1); jj, kk = js[jj], js[kk]; c_ = ac[jj, kk]; j_ = aj[jj, kk]
                    clone["copres_total"] += len(jj); clone["copres_clone"] += int((c_ | j_).sum()); clone["copres_cos"] += int(c_.sum()); clone["copres_J"] += int(j_.sum()); clone["copres_both"] += int((c_ & j_).sum())
                    if len(js) >= 7:
                        ar, om, _ = rho_i_cache[i]; others = np.where(alive)[0]; others = others[others != i]; oj, ok = np.triu_indices(len(others), 1); oj, ok = others[oj], others[ok]
                        heavy.append(dict(task_end=e + 1, i=int(i), n_lifters=int(len(js)), k_i=int(k[i]), lifters=[int(j) for j in js],
                                          lifter_pairs=dict(mean_abs_cos=float(np.abs(cos[jj, kk]).mean()), mean_J=float(Jm[jj, kk].mean()), mean_abs_rho=float(ar[jj, kk].mean()), mean_omega=float(om[jj, kk].mean()),
                                                            frac_abs_cos_gt_0p5=float((np.abs(cos[jj, kk]) > 0.5).mean()), frac_J_gt_0p2=float((Jm[jj, kk] > 0.2).mean()), max_abs_rho=float(ar[jj, kk].max())),
                                          alive_pairs=dict(mean_abs_cos=float(np.abs(cos[oj, ok]).mean()), mean_J=float(Jm[oj, ok].mean()), mean_abs_rho=float(ar[oj, ok].mean()), mean_omega=float(om[oj, ok].mean()),
                                                           frac_abs_cos_gt_0p5=float((np.abs(cos[oj, ok]) > 0.5).mean()), frac_J_gt_0p2=float((Jm[oj, ok] > 0.2).mean())),
                                          lifter_k=[int(k[j]) for j in js], lifter_J_with_target=[float(Jm[i, j]) for j in js]))
                bs = np.where(born[i])[0]
                if len(bs) >= 2:
                    jj, kk = np.triu_indices(len(bs), 1); jj, kk = bs[jj], bs[kk]; c_ = ac[jj, kk]; j_ = aj[jj, kk]
                    clone_b["cob_total"] += len(jj); clone_b["cob_clone"] += int((c_ | j_).sum()); clone_b["cob_cos"] += int(c_.sum()); clone_b["cob_J"] += int(j_.sum())
                    # state-conditioned comparison: co-born pairs vs all alive pairs (j,k != i) vs (j born for i) x (k born for another target, not for i)
                    ar, om, rs = rho_i_cache[i]
                    def _acc(g, J1, K1):
                        d = bcond[g]; d["n"] += len(J1); d["events"] += 1; d["abs_cos"] += float(np.abs(cos[J1, K1]).sum()); d["cos"] += float(cos[J1, K1].sum()); d["J"] += float(Jm[J1, K1].sum())
                        d["abs_rho"] += float(ar[J1, K1].sum()); d["rho"] += float(rs[J1, K1].sum()); d["omega"] += float(om[J1, K1].sum())
                    _acc("coborn", jj, kk)
                    others = np.where(alive)[0]; others = others[others != i]; oj, ok = np.triu_indices(len(others), 1); _acc("alive", others[oj], others[ok])
                    ob = np.where(born.any(0) & ~born[i] & alive)[0]; ob = ob[ob != i]
                    if len(ob):
                        J1, K1 = np.meshgrid(bs, ob, indexing="ij"); _acc("cross_born", J1.ravel(), K1.ravel())
        prev_isb = isb
        if ei % 20 == 0: P(f"[{tag}] e {e} (task end {e+1}) lifters {int(isb.sum())} alive {int(alive.sum())} ({time.time()-t0:.0f}s)")
    # reproduction check against stoch rows (t = 10..100 -> e = 8..98)
    stoch_idx = [E_ALL.index(e) for e in E_STOCH]; n_rows_stoch = int(Lall[stoch_idx].sum())
    ph2_idx = [E_ALL.index(e) for e in E_PH2]; Lt = Lall[ph2_idx]; Bt = Lall[ph2_idx] & ~Lall[[i - 1 for i in ph2_idx]]
    n_i_t = Lt.sum(2)  # [T, H]
    cnt_safe = np.maximum(cnt_i, 1)[:, None, None]; abs_rho_bar = abs_rho_sum / cnt_safe; om_bar = om_sum / cnt_safe; rho_s_bar = rho_s_sum / cnt_safe
    cos_bar = np.abs(cos_t[ph2_idx]).mean(0); J_bar = J_t[ph2_idx].mean(0); cos_s_bar = cos_t[ph2_idx].mean(0)
    birth_cond = {g: ({k_: (d[k_] / d["n"] if d["n"] else None) for k_ in bc_keys} | {"n_pairs": d["n"], "events": d["events"]}) for g, d in bcond.items()}
    P(f"[{tag}] states done ({time.time()-t0:.0f}s). stoch-convention rows {n_rows_stoch}; phase-2 lifter triples {int(Lt.sum())}; births {int(Bt.sum())}")
    tab_p = pair_table(Lt, abs_rho_bar, om_bar, cos_bar, J_bar, cnt_i, n_i_t, rho_s_bar, cos_s_bar); tab_b = pair_table(Bt, abs_rho_bar, om_bar, cos_bar, J_bar, cnt_i, Bt.sum(2), rho_s_bar, cos_s_bar)
    P(f"[{tag}] pair tables: presence {len(tab_p['i'])} pairs, births {len(tab_b['i'])} pairs ({time.time()-t0:.0f}s)")
    res_p = analyze_table(tab_p, "co-presence", args.perms, rng); res_b = analyze_table(tab_b, "co-birth", args.perms, rng)
    P(f"[{tag}] stats done ({time.time()-t0:.0f}s)")
    # distribution of predictors over all alive pairs (phase 2 average) and over eligible pairs
    cp = np.concatenate(cos_pool); jp = np.concatenate(J_pool)
    pred_dist = dict(abs_cos_alive_pairs_pooled_q=[float(np.quantile(cp, q)) for q in (0.1, 0.5, 0.9, 0.99)], J_alive_pairs_pooled_q=[float(np.quantile(jp, q)) for q in (0.1, 0.5, 0.9, 0.99)],
                     J_alive_pairs_pooled_max=float(jp.max()), frac_alive_pairs_abs_cos_gt_0p5=float((cp > 0.5).mean()), frac_alive_pairs_J_gt_0p2=float((jp > 0.2).mean()), n_alive_pair_slots=int(len(cp)),
                     alive_per_state_min_max=[int(min(alive_count_ph2)), int(max(alive_count_ph2))], units_alive_all_phase2=int(alive_all[ph2_idx].all(0).sum()),
                     abs_rho_bar_eligible_q=[float(np.quantile(tab_p["rho"], q)) for q in (0.1, 0.5, 0.9, 0.99)] if len(tab_p["i"]) else None,
                     omega_bar_eligible_q=[float(np.quantile(tab_p["omega"], q)) for q in (0.1, 0.5, 0.9, 0.99)] if len(tab_p["i"]) else None)
    # (4) MC check at task-40 end state
    ei40 = E_ALL.index(E_MC); st = state(dd, E_MC, a, Xt); isb40 = Lall[ei40]; T40 = Tall[ei40]; nl = isb40.sum(1); order = np.argsort(-nl)
    targets = [int(i) for i in order[:3] if alive_all[ei40][i]]
    mc_sgd, dz_sgd, sel_sgd = mc_sgd_check(dd, E_MC, a, Xt, Theta, st["Vc"], st["Dh"], st["W2"], st["b2"], st["Z"], T40, targets, args.mc, rng)
    P(f"[{tag}] SGD MC check: max |err| corr units {mc_sgd['summary']['max_abs_err_corr_units_same']:.4f} (MC sd {1/math.sqrt(args.mc):.4f}), max rel err diag {mc_sgd['summary']['max_rel_err_diag']:.4f} ({time.time()-t0:.0f}s)")
    mc_adam = mc_adam_check(dd, E_MC, a, Xt, st["Vc"], st["Dh"], st["W2"], st["b2"], st["Z"], T40, targets, args.mc, rng) if not args.no_adam else dict(available=False)
    if mc_adam.get("available"): P(f"[{tag}] Adam MC check: grad check {mc_adam['grad_check_rel']:.2e}, dtheta check {mc_adam['dtheta_check_rel']:.2e}, max |err| corr units {mc_adam['summary']['max_abs_err_corr_units']:.4f}, median rel err diag {mc_adam['summary']['median_rel_err_diag']:.4f} ({time.time()-t0:.0f}s)")
    # heavy cases summary
    heavy_sorted = sorted(heavy, key=lambda r: -r["n_lifters"])
    # save raw
    os.makedirs(OUT_RAW, exist_ok=True)
    np.savez_compressed(os.path.join(OUT_RAW, f"m2_{tag}_states.npz"), E_ALL=np.array(E_ALL), L=Lall, T=Tall, k=kall, alive=alive_all, cos_t=cos_t.astype(np.float32), J_t=J_t.astype(np.float32),
                        abs_rho_bar=abs_rho_bar.astype(np.float32), omega_bar=om_bar.astype(np.float32), rho_s_bar=rho_s_bar.astype(np.float32), cnt_i=cnt_i, n_i_t=n_i_t, cos_bar=cos_bar, J_bar=J_bar, cos_s_bar=cos_s_bar)
    np.savez_compressed(os.path.join(OUT_RAW, f"m2_{tag}_pairs.npz"), **{f"presence_{c}": v for c, v in tab_p.items()}, **{f"birth_{c}": v for c, v in tab_b.items()})
    np.savez_compressed(os.path.join(OUT_RAW, f"m2_{tag}_mc_sgd.npz"), dz=dz_sgd.astype(np.float32), inputs=np.array(sel_sgd), targets=np.array(targets))
    n_alive_ut = int(alive_all[ph2_idx].sum())
    summary = dict(arm=tag, a=a, N=N, H=H, phase2_task_ends=[E_PH2[0] + 1, E_PH2[-1] + 1], n_states=len(E_PH2), n_alive_unit_tasks=n_alive_ut,
                   reproduction=dict(stoch_rows_t10_100=n_rows_stoch, stoch_rows_expected={"a0.1": 1511, "a0.3": 2836, "a0.5": 7712}[tag]),
                   lifter_triples_phase2=int(Lt.sum()), births_phase2=int(Bt.sum()), n_i_max=int(n_i_t.max()), n_targets_tasks_with_lifters=int((n_i_t >= 1).sum()),
                   clone_share=dict(**clone, **clone_b, share_copres_clone=clone["copres_clone"] / max(clone["copres_total"], 1), share_copres_cos=clone["copres_cos"] / max(clone["copres_total"], 1), share_copres_J=clone["copres_J"] / max(clone["copres_total"], 1),
                                    share_base_clone=clone["base_clone"] / max(clone["base_pairs"], 1), share_base_cos=clone["base_cos"] / max(clone["base_pairs"], 1), share_base_J=clone["base_J"] / max(clone["base_pairs"], 1),
                                    share_cob_clone=clone_b["cob_clone"] / max(clone_b["cob_total"], 1), share_cob_cos=clone_b["cob_cos"] / max(clone_b["cob_total"], 1), share_cob_J=clone_b["cob_J"] / max(clone_b["cob_total"], 1)),
                   predictor_distribution=pred_dist, copresence=res_p, cobirth=res_b, birth_conditioned=birth_cond, heavy_states=heavy_sorted[:12], n_heavy_states=len(heavy), mc_sgd=mc_sgd, mc_adam=mc_adam, seconds=time.time() - t0)
    with open(os.path.join(OUT_RAW, f"m2_{tag}_summary.json"), "w") as f: json.dump(summary, f, indent=1, ensure_ascii=False)
    P(f"[{tag}] done ({time.time()-t0:.0f}s)")
    return summary


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--arms", nargs="*", default=list(ARMS)); ap.add_argument("--perms", type=int, default=2000); ap.add_argument("--mc", type=int, default=2000)
    ap.add_argument("--quick", action="store_true"); ap.add_argument("--no-adam", action="store_true"); ap.add_argument("--merge", action="store_true", help="assemble results json from per-arm summaries"); args = ap.parse_args()
    if args.quick: args.perms, args.mc = 100, 400
    if args.merge:
        out = {}
        for tag in ARMS:
            f = os.path.join(OUT_RAW, f"m2_{tag}_summary.json")
            if os.path.exists(f): out[tag] = json.load(open(f))
        os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
        with open(OUT_JSON, "w") as f: json.dump(out, f, indent=1, ensure_ascii=False)
        P(f"merged {list(out)} -> {OUT_JSON}"); return
    out = {}
    for tag in args.arms: out[tag] = run_arm(tag, args)
    if args.quick:
        with open(os.path.join(OUT_RAW, "m2_quick.json"), "w") as f: json.dump(out, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
