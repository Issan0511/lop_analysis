import sys, numpy as np
sys.path.insert(0, "/home/issan/Projects/obsidian-research-data/push_direction_proof_0930")
from common import load_X
from math import erf, sqrt
ROOT="/home/issan/Projects/obsidian-research-data/push_direction_proof_0930/"
Phibar = np.vectorize(lambda x: 0.5*(1-erf(x/sqrt(2))))
def analyze(path, T, label):
    d = np.load(ROOT+path); X = load_X(d["subset"]); N = X.shape[0]
    Xt = np.concatenate([X, np.ones((N,1))],1); xbar = Xt.mean(0); K = Xt@xbar; Kb = xbar@xbar
    W1 = d["end0"][T-1].astype(np.float64); b1 = d["end1"][T-1].astype(np.float64)
    W = np.concatenate([W1, b1[:,None]],1)           # [H, 785]
    Z = Xt @ W.T                                     # [N, H]
    m = W @ xbar                                     # mean over n of z  == Z.mean(0)
    assert np.allclose(m, Z.mean(0))
    U = Z - np.outer(K/Kb, m)                        # u_perp: zero mean over n (exact)
    assert np.allclose(U.sum(0), 0, atol=1e-8*np.abs(Z).max())
    opn = Z > 0; k = opn.sum(0); alive = k > 0; H = alive.sum()
    p = opn.mean(0); s = Z.std(0); su = U.std(0); r = m/np.where(s>0,s,1)
    print(f"\n=== {label} T={T}: alive {H}/{len(k)}  K: min {K.min():.3f} max {K.max():.3f} mean {K.mean():.3f} (=Kbar {Kb:.3f}) sd {K.std():.3f} mean/sd {K.mean()/K.std():.2f}")
    print(f" p: q10/50/90 {np.quantile(p[alive],[.1,.5,.9]).round(4)}  r: {np.quantile(r[alive],[.1,.5,.9]).round(2)}  s: {np.quantile(s[alive],[.1,.5,.9]).round(2)}  s_perp/s: {np.quantile((su/s)[alive],[.1,.5,.9]).round(3)}")
    # occupancy profile vs brightness
    f = opn[:,alive].mean(1)                         # fraction of alive units open on n
    order = np.argsort(K); bins = np.array_split(order, 8)
    pred_gauss = np.zeros(N)
    for n in range(N):
        pred_gauss[n] = Phibar((-m[alive]*K[n]/Kb)/su[alive]).mean()   # P(u_perp > -m K/Kb) with gaussian u_perp of sd su
    # width-preserving null: each unit's gate pattern randomly permuted over inputs -> f_null(n) = mean p (flat)
    print(" bins by K: K_mid | f_actual | f_gauss_pred | mean p (flat null)")
    for b in bins:
        print(f"   {K[b].mean():6.3f} | {f[b].mean():.4f} | {pred_gauss[b].mean():.4f} | {p[alive].mean():.4f}")
    print(f" corr(f, K) = {np.corrcoef(f, K)[0,1]:.3f}; corr(f, pred) = {np.corrcoef(f, pred_gauss)[0,1]:.3f}; f max {f.max():.3f} at K {K[f.argmax()]:.3f}; #inputs f>=0.5: {(f>=0.5).sum()}")
    # mean unit
    zbar = Z[:,alive].mean(1); mbar = m[alive].mean(); ubar = U[:,alive].mean(1)
    ratio = ubar/(-mbar*K/Kb)
    print(f" mean unit: mbar {mbar:.3f}; zbar>0 inputs: {(zbar>0).sum()}; max ubar/(|mbar|K/Kb) = {ratio.max():.3f}; sd(ubar)/mean s_perp = {ubar.std()/su[alive].mean():.3f} (indep expectation 1/sqrt(H)={1/np.sqrt(H):.3f})")
    med = np.median(Z[:,alive],1); print(f" cross-unit median z>0 inputs: {(med>0).sum()}")
    # pairwise pattern correlations (normalized u_perp)
    Un = U[:,alive]/su[alive]; C = (Un.T@Un)/N; off = C[~np.eye(H,dtype=bool)]
    print(f" pairwise corr of u_perp: mean {off.mean():.4f}, sd {off.std():.3f}, q01/q99 {np.quantile(off,[.01,.99]).round(3)}, |corr|>0.5: {(np.abs(off)>0.5).mean():.4f}")
    # gaussianity of u_perp per unit: skew & kurtosis
    sk = ((Un**3).mean(0)); ku = ((Un**4).mean(0))-3
    print(f" u_perp skew q10/50/90 {np.quantile(sk,[.1,.5,.9]).round(2)}, excess kurt {np.quantile(ku,[.1,.5,.9]).round(2)}")
    # per-unit: actual p vs gaussian prediction from (m, su, K profile)
    pg = np.array([Phibar((-m[i]*K/Kb)/su[i]).mean() for i in np.where(alive)[0]])
    print(f" per-unit p actual vs gauss-pred: corr {np.corrcoef(p[alive],pg)[0,1]:.3f}; median ratio actual/pred {np.median(p[alive]/np.maximum(pg,1e-9)):.2f}")
    # threshold inputs (3 smallest |z| among all inputs) brightness vs all
    thrK = []
    for i in np.where(alive)[0]:
        idx = np.argsort(np.abs(Z[:,i]))[:3]; thrK.extend(K[idx])
    thrK = np.array(thrK); opK = K[np.where(opn[:,alive].any(1))[0]]
    print(f" K of threshold inputs: mean {thrK.mean():.3f} (all {K.mean():.3f}); frac below median K: {(thrK<np.median(K)).mean():.2f}; K of open gates (unit-weighted): {np.average(K, weights=opn[:,alive].sum(1)):.3f}")
    return dict(K=K, Kb=Kb, f=f, p=p, m=m, su=su, alive=alive, Z=Z, U=U)
for path, label, Ts in [("review/opus_bwdfloor/out/LR_ref_s0.npz","LR a0.1 s0",[2,5,20,50,90]), ("review/opus_lrslope/out/LRa0.3_s0.npz","LR a0.3 s0",[50]), ("review/opus_lrslope/out/LRa0.5_s0.npz","LR a0.5 s0",[50]), ("review/opus_bwdfloor/out/ELU_ref_s0.npz","ELU s0",[50])]:
    for T in Ts: analyze(path, T, label)
