import sys, numpy as np
from math import erfc, sqrt
sys.path.insert(0, "/home/issan/Projects/obsidian-research-data/push_direction_proof_0930")
from common import load_X
ROOT="/home/issan/Projects/obsidian-research-data/push_direction_proof_0930/"
xs=np.linspace(-8,8,1601); dx=xs[1]-xs[0]; phi=np.exp(-xs**2/2)/np.sqrt(2*np.pi)
Phibar=lambda x: 0.5*erfc(x/sqrt(2))
Phibar_v=np.vectorize(Phibar, otypes=[float])
def biv_tail(a,b,rho):
    # P(X>a, Y>b) for standard bivariate normal with corr rho; a,b arrays (same shape)
    a=np.asarray(a,float); b=np.asarray(b,float); out=np.empty(a.shape)
    s=np.sqrt(max(1-rho**2,1e-12))
    for idx in np.ndindex(a.shape):
        m=xs>a[idx]; out[idx]=np.sum(phi[m]*Phibar_v((b[idx]-rho*xs[m])/s))*dx if m.any() else 0.0
    return out
def run(path,label,T,npairs=400,seed=0):
    rng=np.random.default_rng(seed)
    d=np.load(ROOT+path); X=load_X(d["subset"]); N=len(X); Xt=np.c_[X,np.ones(N)]; xbar=Xt.mean(0); K=Xt@xbar; Kb=xbar@xbar
    W=np.c_[d["end0"][T-1].astype(np.float64), d["end1"][T-1].astype(np.float64)]; Z=Xt@W.T; m=W@xbar; U=Z-np.outer(K/Kb,m)
    opn=Z>0; k=opn.sum(0); alive=k>0; idx=np.where(alive)[0]; su=U.std(0)
    Un=U/su; C=(Un[:,idx].T@Un[:,idx])/N
    pairs=[(i,j) for ii,i in enumerate(idx) for jj,j in enumerate(idx) if i!=j and k[i]<=k[j] and k[i]>=5]
    sel=[pairs[t] for t in rng.choice(len(pairs), size=min(npairs,len(pairs)), replace=False)]
    rows=[]
    for (i,j) in sel:
        ci=(-m[i]*K/Kb)/su[i]; cj=(-m[j]*K/Kb)/su[j]            # standardized thresholds per input
        rho=C[list(idx).index(i), list(idx).index(j)]
        both=(opn[:,i]&opn[:,j]).sum()
        pred_rho=biv_tail(ci,cj,rho).sum(); pred_0=(Phibar_v(ci)*Phibar_v(cj)).sum()
        # calibration: single-unit predicted open counts
        pi_pred=Phibar_v(ci).sum(); pj_pred=Phibar_v(cj).sum()
        rows.append((i,j,rho,both,pred_rho,pred_0,k[i],k[j],pi_pred,pj_pred))
    R=np.array(rows,float)
    tot_act=R[:,3].sum(); tot_rho=R[:,4].sum(); tot_0=R[:,5].sum()
    print(f"{label} T={T}: pairs {len(R)} | co-open actual {tot_act:.0f} vs Gaussian-copula pred with global rho {tot_rho:.0f} (ratio {tot_act/tot_rho:.2f}) vs independent {tot_0:.0f} (ratio {tot_act/tot_0:.2f}) | single-unit open counts: actual {R[:,6].mean():.1f}/{R[:,7].mean():.1f} vs gauss pred {R[:,8].mean():.1f}/{R[:,9].mean():.1f}")
    hi=R[:,2]>np.quantile(R[:,2],0.9); lo=R[:,2]<np.quantile(R[:,2],0.1)
    print(f"      top-decile rho pairs (rho>{np.quantile(R[:,2],0.9):.2f}): actual {R[hi,3].sum():.0f} vs pred_rho {R[hi,4].sum():.0f} vs indep {R[hi,5].sum():.0f} | bottom decile: actual {R[lo,3].sum():.0f} vs pred_rho {R[lo,4].sum():.0f}")
    # coverage-type statistic: fraction of target's open set covered by candidate, actual vs pred
    cov_act=R[:,3]/R[:,6]; cov_pred=R[:,4]/np.maximum(R[:,8],1e-9)
    print(f"      coverage |O_i∩O_j|/k_i: actual mean {cov_act.mean():.3f}, q90 {np.quantile(cov_act,.9):.3f}, frac>=0.5 {(cov_act>=0.5).mean():.3f} | pred(rho): mean {cov_pred.mean():.3f}, q90 {np.quantile(cov_pred,.9):.3f}, frac>=0.5 {(cov_pred>=0.5).mean():.3f}")
for path,label in [("review/opus_bwdfloor/out/LR_ref_s0.npz","LR a0.1 s0"),("review/opus_lrslope/out/LRa0.3_s0.npz","LR a0.3 s0"),("review/opus_bwdfloor/out/ELU_ref_s0.npz","ELU s0")]:
    run(path,label,50)
