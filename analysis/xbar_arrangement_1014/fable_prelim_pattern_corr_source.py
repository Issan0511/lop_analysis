import sys, numpy as np
sys.path.insert(0, "/home/issan/Projects/obsidian-research-data/push_direction_proof_0930")
from common import load_X
ROOT="/home/issan/Projects/obsidian-research-data/push_direction_proof_0930/"
rng=np.random.default_rng(0)
def spearman(x,y):
    rx=np.argsort(np.argsort(x)); ry=np.argsort(np.argsort(y)); return np.corrcoef(rx,ry)[0,1]
for path,label,T in [("review/opus_bwdfloor/out/LR_ref_s0.npz","LR a0.1 s0",50),("review/opus_lrslope/out/LRa0.3_s0.npz","LR a0.3 s0",50),("review/opus_lrslope/out/LRa0.5_s0.npz","LR a0.5 s0",50),("review/opus_bwdfloor/out/ELU_ref_s0.npz","ELU s0",50),("review/opus_bwdfloor/out/LR_ref_s0.npz","LR a0.1 s0 (init)",0)]:
    d=np.load(ROOT+path); X=load_X(d["subset"]); N=len(X); Xt=np.c_[X,np.ones(N)]; xbar=Xt.mean(0); K=Xt@xbar; Kb=xbar@xbar
    if T==0: W=np.c_[d["init0"].astype(np.float64), d["init1"].astype(np.float64)]; V=d["init2"].astype(np.float64).T
    else: W=np.c_[d["end0"][T-1].astype(np.float64), d["end1"][T-1].astype(np.float64)]; V=d["end2"][T-1].astype(np.float64).T
    Z=Xt@W.T; m=W@xbar; U=Z-np.outer(K/Kb,m); alive=(Z>0).any(0); H=alive.sum()
    Ua=U[:,alive]; Un=(Ua-Ua.mean(0))/Ua.std(0); C=(Un.T@Un)/N; iu=np.triu_indices(H,1); off=C[iu]
    # data geometry: centered input covariance spectrum, effective dimension
    Xc=Xt-xbar; # pattern = w_perp . x_n; centering over n is automatic for w_perp ⊥ xbar? u_perp has zero mean by construction
    # isotropic null: random w_perp (Gaussian in 785 dims, projected ⊥ xbar), pattern correlations over inputs
    G=rng.standard_normal((785,200)); G-=np.outer(xbar, (xbar@G)/Kb); Ug=Xt@G; Ug=(Ug-Ug.mean(0))/Ug.std(0); Cg=(Ug.T@Ug)/N; offg=Cg[np.triu_indices(200,1)]
    ev=np.linalg.eigvalsh((Xc.T@Xc)/N); ev=ev[ev>0]; PR=ev.sum()**2/(ev**2).sum()
    # readout similarity vs pattern correlation
    Vc=V-V.mean(1,keepdims=True); Va=Vc[alive]; cosv=(Va@Va.T)/np.sqrt(np.outer((Va**2).sum(1),(Va**2).sum(1))); cv=cosv[iu]
    # open-set overlap: actual vs expectation under independence given brightness? skip; report Jaccard tail
    print(f"{label} T={T}: H {H} | pattern corr: mean {off.mean():+.4f} sd {off.std():.3f} q01 {np.quantile(off,.01):+.3f} q99 {np.quantile(off,.99):+.3f} frac>0.3 {(off>0.3).mean():.4f} | isotropic null: mean {offg.mean():+.4f} sd {offg.std():.3f} q99 {np.quantile(offg,.99):+.3f} frac>0.3 {(offg>0.3).mean():.4f} | PR(data) {PR:.1f} 1/sqrt(PR) {1/np.sqrt(PR):.3f}")
    print(f"      corr(pattern corr, cos v^c): Pearson {np.corrcoef(off,cv)[0,1]:+.3f} Spearman {spearman(off,cv):+.3f} | mean pattern corr for cos v^c>0.3: {off[cv>0.3].mean():+.3f} (n={int((cv>0.3).sum())}), for cos<-0.3: {off[cv<-0.3].mean():+.3f} | sd of cos v^c {cv.std():.3f}")
