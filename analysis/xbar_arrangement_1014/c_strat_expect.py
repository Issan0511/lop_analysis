"""C (xbar_arrangement_1014): P149 (arrangement expectation of the gate part of the capacity push, real multi-class states) + K-decile-stratified permutation.
Copy of block_1003/stage10_1005/p149_expect.py (read-only reference); the original computations are kept unchanged (same rng(3) stream) so they can be checked against the P149 logs/json.
New: brightness K_n = x~_n . x~bar (Kv), K-deciles (10 x 120 inputs). Stratified permutation = one random permutation inside each decile (independently), applied jointly to (e', d) (gate) or (u, f) (readout), M fixed.
  E_strat = exact stratified mean (closed form below), checked against the MC mean of 150 stratified draws; sd_strat = MC sd of the 150 draws (separate rng, seed 14).
  closed form: E = sum_s tr_s(M) <xy>_s + sum_s (Msum_ss - tr_s) (Sx_s Sy_s - Sxy_s)/(n_s(n_s-1)) + sum_{s!=t} Msum_st xbar_s ybar_t   (Msum = B^T M B, B = decile indicator)
usage: c_strat_expect.py <a01|a03|a05> [seed=0]     env C_THREADS (default 8)"""
import os, sys, json, time
NT = os.environ.get("C_THREADS", "8")
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[_k] = NT
import numpy as np
T0 = time.time()
REF = "/home/issan/Projects/obsidian-research-data/push_direction_proof_0930/review/stage4_1002/block_1003"
S9 = os.path.join(REF, "stage9_1004")
exec(open(os.path.join(S9, "p138_multiclass.py")).read().split("mode = sys.argv[1]")[0].replace("os.path.dirname(os.path.abspath(__file__))", repr(S9)))
tag = sys.argv[1]; SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
# seed switch (same sources as stage16_1008_seed1/p138_multiclass.py, P177); seed 0 = the original real_V
SRCS = {("a01", 0): "review/opus_bwdfloor/out/LR_ref_s0.npz", ("a01", 1): "review/opus_bwdfloor/out/LR_ref_s1.npz", ("a03", 0): "review/opus_lrslope/out/LRa0.3_s0.npz", ("a03", 1): "review/opus_lrslope/out/LRa0.3_s1.npz", ("a05", 0): "review/opus_lrslope/out/LRa0.5_s0.npz", ("a05", 1): "review/opus_lrslope/out/LRa0.5_s1.npz"}
REFTASK = {"a01": 40, "a03": 100, "a05": 100}
if SEED != 0:
    def real_V(tag):
        src, task = SRCS[(tag, SEED)], REFTASK[tag]
        d = np.load(os.path.join(ROOT, src)); W1, b1, W2 = [d[f"end{j}"][task - 2].astype(np.float64) for j in range(3)]; Vc = W2.T - W2.T.mean(1, keepdims=True)
        B = np.linalg.svd(np.eye(10) - np.ones((10, 10)) / 10)[0][:, :9]; return Vc @ B, W1, b1, d["subset"]
sys.path.insert(0, ROOT); from common import load_X
V, W1, b1, subset = real_V(tag); X = load_X(subset); N = X.shape[0]; Xt = np.concatenate([X, np.ones((N, 1))], 1); W = np.concatenate([W1, b1[:, None]], 1); a = {"a01": 0.1, "a03": 0.3, "a05": 0.5}[tag]
if SEED == 0:
    chk = np.load(os.path.join(S9, "verify_real", f"real_{tag}.npz")); assert np.abs(chk["W"] - W).max() < 1e-9 and np.abs(chk["g"] - (V * V).sum(1)).max() < 1e-8 * chk["g"].max()
Z = Xt @ W.T; tau = 0.1 * Z.std(0) + 1e-12; big, Th, Z, D, Dp, F = mc_kernel(Xt, W, V, a, tau); Kp = V.shape[1]; Kv = Xt @ Xt.mean(0)
lam = 1e-4 * lmax_power(big, iters=3000); big[np.diag_indices_from(big)] += lam; E4 = np.linalg.inv(big).reshape(Kp, N, Kp, N); del big
print(f"[{time.time() - T0:.0f}s] {tag} s{SEED}: kernel inverted, N={N} K'={Kp} lam={lam:.6g} threads={NT}", flush=True)
Ebar = sum(E4[c, :, c, :] for c in range(Kp)); g = (V * V).sum(1); rng = np.random.default_rng(3); rows = []; Ip = np.eye(Xt.shape[1])
def lemma(M, x, y):
    n = len(x); mud = np.trace(M) / n; tot = M.sum(); muo = (tot - np.trace(M)) / (n * (n - 1)); cov = np.mean(x * y) - x.mean() * y.mean()
    return n * (mud - muo) * cov + tot * x.mean() * y.mean(), n * (mud - muo) * cov, -n * muo / (mud - muo)
def mcsd(M, x, y, k=150):
    v = np.empty(k)
    for t in range(k): pm = rng.permutation(len(x)); v[t] = x[pm] @ M @ y[pm]
    return float(v.std()), float(v.mean())
# ---- new: K-deciles and the stratified permutation ----
NDEC = 10; order = np.argsort(Kv, kind="stable"); dec = np.empty(N, int); dec[order] = np.arange(N) * NDEC // N
groups = [np.where(dec == k)[0] for k in range(NDEC)]; Bm = np.zeros((N, NDEC)); Bm[np.arange(N), dec] = 1.0; ns = Bm.sum(0)
assert (Kv > 0).all() and all(len(G_) == N // NDEC for G_ in groups)
rng_s = np.random.default_rng(14)
def strat_lemma(M, x, y):
    Ms = Bm.T @ M @ Bm; tr = np.bincount(dec, weights=np.diag(M), minlength=NDEC); Sx = Bm.T @ x; Sy = Bm.T @ y; Sxy = Bm.T @ (x * y)
    within_diag = float(tr @ (Sxy / ns)); within_off = float((np.diag(Ms) - tr) @ ((Sx * Sy - Sxy) / (ns * (ns - 1))))
    xb = Sx / ns; yb = Sy / ns; O = Ms.copy(); np.fill_diagonal(O, 0.0); between = float(xb @ O @ yb)
    return within_diag + within_off + between
def strat_perms(k):
    PM = np.empty((k, N), int)
    for t in range(k):
        for G_ in groups: PM[t, G_] = G_[rng_s.permutation(len(G_))]
    return PM
def mc_strat(M, x, y, k=150):
    PM = strat_perms(k); Xp = x[PM]; Yp = y[PM]; v = np.einsum("tn,tn->t", Xp @ M, Yp); return float(v.std()), float(v.mean())
def dec_frac(x): s = np.bincount(dec, weights=x, minlength=NDEC); return (s / s.sum()).tolist()
def dec_mean(x): return (np.bincount(dec, weights=x, minlength=NDEC) / ns).tolist()
rng_r = np.random.default_rng(15)
def mc_unif_r(M, x, y, k=150):
    PM = np.stack([rng_r.permutation(N) for _ in range(k)]); v = np.einsum("tn,tn->t", x[PM] @ M, y[PM]); return float(v.std()), float(v.mean())
alive_units = np.where((Z > 0).any(0))[0]
for j, i in enumerate(alive_units):
    d = D[:, i]; e = Kv * Dp[:, i]; f = F[:, i]; u = Kv * d; P = np.einsum("a,b,anbm->nm", V[i], V[i], E4)          # gamma * P_i
    Mf = Th * P; Sg = float(e @ Mf @ d); Ef, covf, Tf = lemma(Mf, e, d); sdf, mcf = mcsd(Mf, e, d)
    Y = d[:, None] * Xt; PY = P @ Y; G = Y.T @ PY; PB = P + PY @ np.linalg.solve(Ip - G, PY.T); Mb = Th * PB           # gamma * P^B_i (gate path of unit i removed); Q = I - G
    Sw = float(e @ Mb @ d); Eb, covb, Tb = lemma(Mb, e, d); sdb, mcb = mcsd(Mb, e, d)
    Sr = float(u @ Ebar @ f); Er, covr, Tr = lemma(Ebar, u, f)
    row = dict(unit=int(i), Ag=Sg, Ef=float(Ef), covshare_f=float(covf / Ef), Tbar_f=float(Tf), sd_f=sdf, mc_f=mcf, Aw=Sw, Eb=float(Eb), Tbar_b=float(Tb), sd_b=sdb, mc_b=mcb, Ar=Sr, Er=float(Er), covr=float(covr),
               cov_ed=float(np.mean(e * d) - e.mean() * d.mean()), minPB=float(np.linalg.eigvalsh(PB)[0]) if i % 25 == 0 else None)
    # new: stratified expectation (exact + MC check) and MC sd, both M versions; readout part
    Esf = strat_lemma(Mf, e, d); sdsf, mcsf = mc_strat(Mf, e, d); Esb = strat_lemma(Mb, e, d); sdsb, mcsb = mc_strat(Mb, e, d)
    Esr = strat_lemma(Ebar, u, f); sdsr, mcsr = mc_strat(Ebar, u, f); sdur, mcur = mc_unif_r(Ebar, u, f)
    row.update(Es_f=Esf, sd_s_f=sdsf, mc_s_f=mcsf, Es_b=Esb, sd_s_b=sdsb, mc_s_b=mcsb, Es_r=Esr, sd_s_r=sdsr, mc_s_r=mcsr, sd_u_r=sdur, mc_u_r=mcur,
               p_open=float((Z[:, i] > 0).mean()),
               prof=dict(e_frac=dec_frac(e), dma_frac=dec_frac(d - a), Mf_diag=dec_mean(np.diag(Mf)), Mf_row=dec_mean(Mf.sum(1)), Mb_diag=dec_mean(np.diag(Mb)), Mb_row=dec_mean(Mb.sum(1)),
                         Mf_diag_all=float(np.diag(Mf).mean()), Mf_row_all=float(Mf.sum() / N), Mb_diag_all=float(np.diag(Mb).mean()), Mb_row_all=float(Mb.sum() / N),
                         d_mean=dec_mean(d), e_mean=dec_mean(e)))
    rows.append(row)
    if j % 20 == 0: print(f"[{time.time() - T0:.0f}s] unit {j + 1}/{len(alive_units)}", flush=True)
OUT = "/home/issan/Projects/claude/wt/xbar_arrangement_1014/results/xbar_arrangement_1014"
meta = dict(tag=tag, seed=SEED, a=a, N=N, Kp=Kp, lam=lam, threads=NT, n_alive=len(rows), ndec=NDEC, K_dec_mean=dec_mean(Kv), K_dec_min=[float(Kv[G_].min()) for G_ in groups], K_dec_max=[float(Kv[G_].max()) for G_ in groups],
            xnorm2_dec_mean=dec_mean(np.diag(Th)), rng=dict(p149=3, strat=14, readout_uniform=15), draws=150, script="analysis/xbar_arrangement_1014/c_strat_expect.py", elapsed_s=time.time() - T0)
json.dump(dict(meta=meta, rows=rows), open(os.path.join(OUT, f"c_strat_expect_{tag}_s{SEED}.json"), "w"))
# ---- P149 lines (unchanged format) ----
f_ = lambda k: np.array([r[k] for r in rows]); n = len(rows)
print(f"{tag}: units {n} | Cov(e', d) > 0: {int((f_('cov_ed') > 0).sum())} | lemma vs MC mean (full): max |diff|/sd*sqrt(150) {np.max(np.abs(f_('Ef') - f_('mc_f')) / (f_('sd_f') / np.sqrt(150))):.1f}")
for nm, S, E, sd, T in (("full M", "Ag", "Ef", "sd_f", "Tbar_f"), ("probe M (gate path removed)", "Aw", "Eb", "sd_b", "Tbar_b")):
    ra = f_(S) / f_(E); print(f"   gate, {nm}: expectation > 0: {int((f_(E) > 0).sum())}/{n} | actual > 0: {int((f_(S) > 0).sum())}/{n} | actual/expectation median {np.median(ra):.2f} min {ra.min():.2f} max {ra.max():.2f} | margin E/sd median {np.median(f_(E) / f_(sd)):.1f} min {(f_(E) / f_(sd)).min():.1f} | z median {np.median((f_(S) - f_(E)) / f_(sd)):+.1f} | Tbar median {np.median(f_(T)):.3f} max {f_(T).max():.3f}")
print(f"   screening (actual / probe-form actual): median {np.median(f_('Ag') / f_('Aw')):.2f} min {(f_('Ag') / f_('Aw')).min():.2f} max {(f_('Ag') / f_('Aw')).max():.2f}")
rr = f_("Ar") / f_("Er"); print(f"   readout: expectation > 0: {int((f_('Er') > 0).sum())}/{n} | actual > 0: {int((f_('Ar') > 0).sum())}/{n} | actual/expectation median {np.median(rr):.3f} | Cov(u, phi) > 0: {int((f_('covr') > 0).sum())}/{n}")
# ---- new quick lines ----
for nm, S, E, Es, sds, mcs in (("full", "Ag", "Ef", "Es_f", "sd_s_f", "mc_s_f"), ("probe", "Aw", "Eb", "Es_b", "sd_s_b", "mc_s_b"), ("readout", "Ar", "Er", "Es_r", "sd_s_r", "mc_s_r")):
    gap = (f_(E) - f_(Es)) / (f_(E) - f_(S)); zs = (f_(S) - f_(Es)) / f_(sds); chkz = np.abs(f_(Es) - f_(mcs)) / (f_(sds) / np.sqrt(150))
    print(f"   strat {nm}: E_strat>0 {int((f_(Es) > 0).sum())}/{n} | actual/E_strat median {np.median(f_(S) / f_(Es)):.2f} | gap closed median {np.median(gap):.2f} | z_strat median {np.median(zs):+.1f} | exact vs MC max |diff|/(sd/sqrt150) {chkz.max():.1f}")
print(f"[{time.time() - T0:.0f}s] done", flush=True)
