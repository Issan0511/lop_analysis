import sys, numpy as np
sys.path.insert(0, "/home/issan/Projects/obsidian-research-data/push_direction_proof_0930")
from common import load_X
ROOT="/home/issan/Projects/obsidian-research-data/push_direction_proof_0930/"
def state(path, T):
    d = np.load(ROOT+path); X = load_X(d["subset"]); N = X.shape[0]
    Xt = np.concatenate([X, np.ones((N,1))],1); xbar = Xt.mean(0); K = Xt@xbar; Kb = xbar@xbar
    W = np.concatenate([d["end0"][T-1].astype(np.float64), d["end1"][T-1].astype(np.float64)[:,None]],1)
    Z = Xt@W.T; m = W@xbar; U = Z - np.outer(K/Kb, m); return K, Kb, Z, m, U
for path,label,Ts in [("review/opus_bwdfloor/out/LR_ref_s0.npz","LR a0.1 s0",[1,2,3,50]),("review/opus_lrslope/out/LRa0.5_s0.npz","LR a0.5 s0",[1,50]),("review/opus_bwdfloor/out/ELU_ref_s0.npz","ELU s0",[1,50])]:
    for T in Ts:
        K,Kb,Z,m,U = state(path,T); opn = Z>0; alive = opn.any(0); p = opn.mean(0)
        f = opn[:,alive].mean(1); su = U.std(0)
        # does the pattern itself depend on brightness? per-unit corr(u_perp, K) and the brightness-regression slope share
        cu = np.array([np.corrcoef(U[:,i],K)[0,1] for i in np.where(alive)[0]])
        print(f"{label} T={T}: alive {alive.sum()} p>1/2 units {(p[alive]>0.5).sum()} | f>=1/2 inputs {(f>=0.5).sum()}/{len(f)} f max {f.max():.3f} | mean-unit zbar>0 inputs {(Z[:,alive].mean(1)>0).sum()} | corr(u_perp,K) per unit q10/50/90 {np.quantile(cu,[.1,.5,.9]).round(3)} | mean r {np.mean((m/Z.std(0))[alive]):.2f}")
