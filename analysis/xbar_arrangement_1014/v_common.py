"""Verifier (Opus, 1014) common loaders, written from the definitions (THEORY_XBAR.md §0, cobirth_seeds_1013/verify_counts.py docstring).
state T (M1) = end0[T-1], end1[T-1]; Xt = [X, 1]; xbar = mean_n Xt_n; K_n = Xt_n.xbar; Kbar = xbar.xbar;
z_kn = w_k.x_n + b_k; O_k = {z > 0}; alive = |O_k| > 0; m_k = mean_n z_kn; u_kn = z_kn - m_k K_n/Kbar; sperp_k = sd_n u_kn (ddof 0).
Threshold inputs (leaky / E1): top-3 of log K_n - |x| - 2 log(1+e^-|x|) - log tau, x = z/tau, tau = 0.1 sd_n(z) + 1e-12 (stable argsort).
E2: log K_n + z on z < 0, -inf on open. Bridge j->i: j != i, both alive, |O_j cap T_i| >= 2 and |O_j cap O_i| >= ceil(k_i/2).
C_i = #{j != i alive: |O_j cap T_i| >= 2}."""
import os, sys
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[_k] = "2"
import numpy as np
ROOT = "/home/issan/Projects/obsidian-research-data/push_direction_proof_0930"
sys.path.insert(0, ROOT)
from common import load_X  # noqa: E402
RES = "/home/issan/Projects/claude/wt/xbar_arrangement_1014/results/xbar_arrangement_1014"
OBS = "/home/issan/Projects/obsidian-research-data"
CONDS = {"a01s0": ("review/opus_bwdfloor/out/LR_ref_s0.npz", "leaky"), "a01s1": ("review/opus_bwdfloor/out/LR_ref_s1.npz", "leaky"),
         "a003s0": ("review/opus_lrslope/out/LRa0.03_s0.npz", "leaky"), "a02s0": ("review/opus_lrslope/out/LRa0.2_s0.npz", "leaky"),
         "a03s0": ("review/opus_lrslope/out/LRa0.3_s0.npz", "leaky"), "a03s1": ("review/opus_lrslope/out/LRa0.3_s1.npz", "leaky"),
         "a05s0": ("review/opus_lrslope/out/LRa0.5_s0.npz", "leaky"), "a05s1": ("review/opus_lrslope/out/LRa0.5_s1.npz", "leaky"),
         "ELU_s0_E1": ("review/opus_bwdfloor/out/ELU_ref_s0.npz", "E1"), "ELU_s1_E1": ("review/opus_bwdfloor/out/ELU_ref_s1.npz", "E1")}
PH2 = list(range(11, 101))

def load(cond_or_npz):
    npz = CONDS[cond_or_npz][0] if cond_or_npz in CONDS else cond_or_npz
    d = np.load(os.path.join(ROOT, npz)); X = load_X(d["subset"]); Xt = np.c_[X, np.ones(len(X))]
    xbar = Xt.mean(0); K = Xt @ xbar; Kbar = float(xbar @ xbar)
    return d, Xt, xbar, K, Kbar

def weights(d, T):
    if T == 0: return np.c_[d["init0"].astype(np.float64), d["init1"].astype(np.float64)]
    return np.c_[d["end0"][T - 1].astype(np.float64), d["end1"][T - 1].astype(np.float64)]

def thresholds(Z, K, gate):
    if gate in ("leaky", "E1"):
        tau = 0.1 * Z.std(0) + 1e-12; x = Z / tau[None, :]; ax = np.abs(x)
        score = np.log(K)[:, None] - ax - 2.0 * np.log1p(np.exp(-ax)) - np.log(tau)[None, :]
    else:
        score = np.where(Z < 0, np.log(K)[:, None] + Z, -np.inf)
    return np.argsort(-score, axis=0, kind="stable")[:3]          # (3, H)

def bridge_counts(Otgt, T3, Ocand, alive):
    """Otgt (N,H) actual target gates, T3 (3,H) target threshold inputs, Ocand (N,H) candidate gates; alive (H,) actual.
    returns N_i, C_i (H,), -1 where target not alive."""
    Of = Otgt.astype(np.int32); Cf = Ocand.astype(np.int32)
    thr = np.stack([Cf[T3[:, i]].sum(0) for i in range(Otgt.shape[1])])     # thr[i, j] = |T_i cap O_j|
    ov = Of.T @ Cf                                                             # ov[i, j] = |O_i cap O_j|
    k = Of.sum(0); need = -(-k // 2)
    both = alive[:, None] & alive[None, :]; np.fill_diagonal(both, False)
    c = (thr >= 2) & both; br = c & (ov >= need[:, None])
    return np.where(alive, br.sum(1), -1), np.where(alive, c.sum(1), -1)

def phase2_stats(Narr):
    """Narr (90, H) phase-2 counts with -1 = not alive. Targets with >= 30 alive tasks."""
    mus, pool = [], []
    for i in range(Narr.shape[1]):
        v = Narr[:, i]; v = v[v >= 0]
        if len(v) < 30: continue
        mus.append(v.mean()); pool.append(v)
    allv = np.concatenate(pool).astype(float)
    return dict(n_targets=len(mus), mu_median=float(np.median(mus)), pooled_mean=float(allv.mean()),
                pooled_fano=float(allv.var(ddof=1) / allv.mean()), max=int(allv.max()), n_ge10=int((allv >= 10).sum()))
