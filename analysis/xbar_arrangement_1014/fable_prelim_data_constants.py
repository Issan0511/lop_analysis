import sys, numpy as np
sys.path.insert(0, "/home/issan/Projects/obsidian-research-data/push_direction_proof_0930")
from common import load_X
ROOT="/home/issan/Projects/obsidian-research-data/push_direction_proof_0930/"
for path,label in [("review/opus_bwdfloor/out/LR_ref_s0.npz","seed0"),("review/opus_bwdfloor/out/LR_ref_s1.npz","seed1")]:
    d=np.load(ROOT+path); X=load_X(d["subset"]); N=len(X); Xt=np.c_[X,np.ones(N)]; xbar=Xt.mean(0); K=Xt@xbar; Kb=xbar@xbar
    Xp=Xt-np.outer(K/Kb,xbar); Sp=(Xp.T@Xp)/N; ev=np.linalg.eigvalsh(Sp); ev=ev[ev>1e-12]; PR=ev.sum()**2/(ev**2).sum()
    nperp=np.linalg.norm(Xp,axis=1); cot=K/(np.sqrt(Kb)*nperp)
    print(f"{label}: |xbar|={np.sqrt(Kb):.3f} Kbar={Kb:.2f} tr(Sigma_perp)={ev.sum():.2f} sqrt={np.sqrt(ev.sum()):.3f} rho_data=sqrt(tr)/|xbar|={np.sqrt(ev.sum())/np.sqrt(Kb):.3f} PR(Sigma_perp)={PR:.1f} 1/sqrt(PR)={1/np.sqrt(PR):.3f} | cot theta: min {cot.min():.3f} q10 {np.quantile(cot,.1):.3f} median {np.median(cot):.3f} q90 {np.quantile(cot,.9):.3f} max {cot.max():.3f} | |x_perp|: min {nperp.min():.2f} median {np.median(nperp):.2f} max {nperp.max():.2f} | corr(cot,K) {np.corrcoef(cot,K)[0,1]:.3f}")
    # lambda_k for a phase-2 state: |m_k|/s_perp_k * rho_data
    for T in (50,):
        W=np.c_[d["end0"][T-1].astype(np.float64), d["end1"][T-1].astype(np.float64)]; Z=Xt@W.T; m=W@xbar; U=Z-np.outer(K/Kb,m); alive=(Z>0).any(0); su=U.std(0)
        lam=(-m/su)[alive]*np.sqrt(ev.sum())/np.sqrt(Kb)
        print(f"   T={T} a0.1: lambda_k q10/50/90 = {np.quantile(lam,[.1,.5,.9]).round(2)}; implied occupancy at darkest input cot_min: mean_k Phibar(lam*cot_min) = {np.mean(0.5*np.vectorize(__import__('math').erfc)(lam*cot.min()/np.sqrt(2))):.3f}; at median cot: {np.mean(0.5*np.vectorize(__import__('math').erfc)(lam*np.median(cot)/np.sqrt(2))):.4f}")
