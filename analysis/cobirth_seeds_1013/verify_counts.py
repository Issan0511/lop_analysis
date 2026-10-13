"""TASK V (cobirth_seeds_1013, adversarial verifier, 1014): an independent minimal recomputation of the loose-bridge counts N_i(T)
and of the phase-2 statistics that the four agent reports (SEED1 / SLOPES / ELU / nbeta_time) rest on. Written from the definitions in
cobirth_1013 README section 1, not copied from m1_counts.py / bridges_at:
  state T (M1 convention, T = 1..100) = end0[T-1], end1[T-1];  z_in = w_i.x_n + b_i;  O_i = {n : z_in > 0};  k_i = |O_i|;  alive = k_i > 0.
  Threshold inputs T_i = the 3 inputs with the largest score e'_in, computed in the LOG domain:
    leaky/E1 : log e' = log K_n + log s(1-s) - log tau_i,  s = sigmoid(z/tau),  tau_i = 0.1 sd_n(z_in) + 1e-12  (the factor (1-a) drops out of the ranking)
    E2       : log e' = log K_n + z_in on closed inputs (z_in < 0), -inf on open ones       (e' = K_n e^z 1[z<0])
  K_n = x~_n . mean(x~) > 0 (checked).  j != i (both alive) lifts i iff |O_j cap T_i| >= 2 and |O_j cap O_i| >= ceil(k_i / 2).
  C_i = #{j != i alive : |O_j cap T_i| >= 2}.
Statistics on phase 2 (T 11..100), targets with >= 30 alive tasks (M1 rule): per-target mu_i, s_i^2 (ddof=1), F_i = s_i^2/mu_i; pooled Fano
(ddof=1) over the used targets' alive tasks; max N; N >= 10; break-even n*(p) = mu_i + s_i sqrt((1-p)/p); (C, width) stratified within Fano
(population variances, 12 fine width bins x min(C, 30), all alive (i, T) of phase 2); null 1 = independent circular shift of each candidate's
series over the target's alive tasks (own RNG, seed 777).
usage: verify_counts.py <tag> <npz rel. to ROOT> <gate leaky|E1|E2> <agent npz to compare or -> [R_null]"""
import os, sys, json, math, time
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[_k] = "1"
import numpy as np
ROOT = "/home/issan/Projects/obsidian-research-data/push_direction_proof_0930"
sys.path.insert(0, ROOT); from common import load_X
OUT = "/home/issan/Projects/obsidian-research-data/cobirth_seeds_1013/verify"
FINE = [(1, 1), (2, 2), (3, 3), (4, 5), (6, 8), (9, 12), (13, 20), (21, 40), (41, 80), (81, 160), (161, 320), (321, 1200)]

def counts(npz, gate):
    d = np.load(os.path.join(ROOT, npz)); X = load_X(d["subset"]); Xt = np.c_[X, np.ones(len(X))]; K = Xt @ Xt.mean(0)
    assert (K > 0).all(); logK = np.log(K)
    H = d["end0"].shape[1]; NT = d["end0"].shape[0]
    Nc = np.full((NT, H), -1, int); Cc = np.full((NT, H), -1, int); kk = np.zeros((NT, H), int); BR = np.zeros((NT, H, H), bool)
    for T in range(1, NT + 1):
        W = d["end0"][T - 1].astype(np.float64); b = d["end1"][T - 1].astype(np.float64); Z = Xt[:, :-1] @ W.T + b[None, :]
        O = Z > 0; k = O.sum(0); alive = k > 0
        if gate in ("leaky", "E1"):
            tau = 0.1 * Z.std(0) + 1e-12; x = Z / tau[None, :]; ax = np.abs(x)
            score = logK[:, None] - ax - 2.0 * np.log1p(np.exp(-ax)) - np.log(tau)[None, :]
        else:
            score = np.where(Z < 0, logK[:, None] + Z, -np.inf)
        T3 = np.argsort(-score, axis=0, kind="stable")[:3]            # (3, H)
        Of = O.astype(np.int64)
        thr = Of[T3].sum(0)                                              # thr[i, j] = |T_i cap O_j|
        ov = Of.T @ Of                                                   # ov[i, j] = |O_i cap O_j|
        need = -(-k // 2)                                                # ceil(k/2)
        both = alive[:, None] & alive[None, :]; np.fill_diagonal(both, False)
        br = (thr >= 2) & (ov >= need[:, None]) & both
        BR[T - 1] = br; kk[T - 1] = k
        Nc[T - 1] = np.where(alive, br.sum(1), -1); Cc[T - 1] = np.where(alive, ((thr >= 2) & both).sum(1), -1)
    return Nc, Cc, kk, BR

def stats(Nc, Cc, kk, BR, R, seed=777):
    ph = np.arange(10, 100)                                              # rows T-1 for T = 11..100
    out = {}; mus = []; vs = []; Fs = []; pool = []; ids = []; mats = []
    for i in range(Nc.shape[1]):
        rows = ph[Nc[ph, i] >= 0]
        if len(rows) < 30: continue
        v = Nc[rows, i].astype(float); m = v.mean(); s2 = v.var(ddof=1)
        mus.append(m); vs.append(s2); Fs.append(s2 / m if m > 0 else np.nan); pool.append(v); ids.append(i)
        mats.append(np.delete(BR[rows, i, :], i, axis=1).astype(np.int8))
    mus = np.array(mus); vs = np.array(vs); Fs = np.array(Fs); allv = np.concatenate(pool)
    out["n_targets"] = len(ids); out["n_target_tasks"] = int(len(allv)); out["mu_pooled"] = float(allv.mean())
    out["pool_fano"] = float(allv.var(ddof=1) / allv.mean()); out["fano_median"] = float(np.nanmedian(Fs)); out["fano_q90"] = float(np.nanquantile(Fs, 0.9)); out["fano_max"] = float(np.nanmax(Fs))
    out["fano_max_target"] = int(ids[int(np.nanargmax(Fs))]); out["max_N"] = int(allv.max()); out["n_N_ge10"] = int((allv >= 10).sum()); out["mu_max"] = float(mus.max())
    for p in (0.14, 0.11, 0.08):
        be = mus + np.sqrt(vs) * math.sqrt((1 - p) / p); a = int(np.argmax(be))
        out[f"breakeven_{p:g}"] = dict(max=float(be.max()), target=int(ids[a]), mu=float(mus[a]), sd=float(math.sqrt(vs[a])), median=float(np.median(be)))
    # (C, width) decomposition over all alive (i, T) of phase 2 (population variances)
    sel = Nc[ph] >= 0; NN = Nc[ph][sel].astype(float); CC = Cc[ph][sel].astype(float); KK = kk[ph][sel]
    kb = np.zeros(len(KK), int)
    for bidx, (lo, hi) in enumerate(FINE): kb[(KK >= lo) & (KK <= hi)] = bidx
    g = kb * 100 + np.minimum(CC, 30).astype(int); ug, inv = np.unique(g, return_inverse=True); means = np.bincount(inv, weights=NN) / np.bincount(inv)
    out["CW_within_fano"] = float(((NN - means[inv]) ** 2).mean() / NN.mean()); out["C_fano"] = float(CC.var() / CC.mean()); out["all_alive_n"] = int(len(NN))
    if R > 0:
        rng = np.random.default_rng(seed); med = np.empty(R)
        for r in range(R):
            F = np.empty(len(mats))
            for t, M in enumerate(mats):
                n, m = M.shape; off = rng.integers(0, n, size=m); idx = (np.arange(n)[:, None] - off[None, :]) % n
                v = np.take_along_axis(M, idx, axis=0).sum(1).astype(float); mu = v.mean(); F[t] = v.var(ddof=1) / mu if mu > 0 else np.nan
            med[r] = np.nanmedian(F)
        out["null1_median_F_q025_q975"] = [float(np.quantile(med, 0.025)), float(np.quantile(med, 0.975))]; out["null1_R"] = R
        out["null1_p_two_sided"] = float(min(1.0, 2 * min((1 + (med >= out["fano_median"]).sum()) / (R + 1), (1 + (med <= out["fano_median"]).sum()) / (R + 1))))
    out["per_target"] = dict(i=ids, mu=mus.tolist(), var=vs.tolist())
    return out

if __name__ == "__main__":
    tag, npz, gate, cmp = sys.argv[1:5]; R = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    t0 = time.time(); Nc, Cc, kk, BR = counts(npz, gate); res = stats(Nc, Cc, kk, BR, R)
    res.update(tag=tag, npz=npz, gate=gate)
    if cmp != "-":
        z = np.load(cmp); Na = z["N"].astype(int)
        res["cmp_agent_npz"] = cmp; res["N_equal_all"] = bool((Na == Nc).all()); res["N_n_diff_cells"] = int((Na != Nc).sum())
        res["N_equal_phase2"] = bool((Na[10:] == Nc[10:]).all()); res["isb_equal"] = bool((z["isb"].astype(bool) == BR).all())
        if "C" in z.files:
            Ca = z["C"].astype(int); res["C_equal_alive"] = bool((Ca[Nc >= 0] == Cc[Nc >= 0]).all())
    res["elapsed_s"] = time.time() - t0
    np.savez_compressed(os.path.join(OUT, f"vcounts_{tag}.npz"), N=Nc, C=Cc, k=kk)
    json.dump(res, open(os.path.join(OUT, f"vstats_{tag}.json"), "w"), indent=1)
    show = {k: v for k, v in res.items() if k != "per_target"}
    print(json.dumps(show))
