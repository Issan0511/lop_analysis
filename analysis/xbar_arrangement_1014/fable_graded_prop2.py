"""Graded version of 命題 2: F(n) = mean_k Phibar(lam_k cot_n) with SIGNED lam_k = (-m_k/sp_k) sqrt(tr Sperp)/|xbar|.
Per state: premise (all m<0) status, mismatch m<0 vs p>=1/2, model max F vs measured max f, margin decomposition at worst input."""
import os, sys, json, numpy as np
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ[k]="4"
from math import erf, sqrt
ROOT="/home/issan/Projects/obsidian-research-data/push_direction_proof_0930"; sys.path.insert(0,ROOT); from common import load_X
FILES={"LR0.1_s0":"review/opus_bwdfloor/out/LR_ref_s0.npz","LR0.1_s1":"review/opus_bwdfloor/out/LR_ref_s1.npz",
 **{f"LR{a}_s{s}":f"review/opus_lrslope/out/LRa{a}_s{s}.npz" for a in ("0.03","0.2","0.3","0.5") for s in (0,1)},
 "ELU_s0":"review/opus_bwdfloor/out/ELU_ref_s0.npz","ELU_s1":"review/opus_bwdfloor/out/ELU_ref_s1.npz"}
Phibar=np.vectorize(lambda x: 0.5*(1-erf(x/sqrt(2))))
out={}
for tag,path in FILES.items():
    d=np.load(os.path.join(ROOT,path)); X=load_X(d["subset"]); N=len(X)
    Xt=np.c_[X,np.ones(N)]; xbar=Xt.mean(0); K=Xt@xbar; Kb=xbar@xbar; nx=np.sqrt(Kb)
    Xp=Xt-np.outer(K/Kb,xbar); nperp=np.linalg.norm(Xp,axis=1); cot=K/(nx*nperp); trS=(nperp**2).mean()
    rows=[]
    for T in range(0,101):
        if T==0:
            W=np.c_[d["init0"],d["init1"]] if "init0" in d.files else None
            if W is None: continue
        else:
            W=np.c_[d["end0"][T-1],d["end1"][T-1]]
        W=W.astype(np.float64); Z=Xt@W.T; m=W@xbar; U=Z-np.outer(K/Kb,m); sp=U.std(0)
        p=(Z>0).mean(0); alive=p>0; H=int(alive.sum())
        lam=(-m/np.maximum(sp,1e-12))*np.sqrt(trS)/nx            # signed: >0 iff m<0
        Pm=Phibar(np.outer(cot,lam))                                # N x Hall
        F=Pm[:,alive].mean(1); f=(Z[:,alive]>0).mean(1)
        n_mpos=int(((m>=0)&alive).sum()); n_phalf=int(((p>=0.5)&alive).sum())
        mis1=int(((m<0)&(p>=0.5)&alive).sum()); mis2=int(((m>=0)&(p<0.5)&alive).sum())
        nstar=int(F.argmax()); neg=alive&(m<0); pos=alive&(m>=0)
        Dm=float((0.5-Pm[nstar,neg]).sum()); Dp=float((Pm[nstar,pos]-0.5).sum())
        # allowed number of fully-positive units at the worst-cot input before F>=1/2 (each adds at most 1/2 in excess)
        imin=int(cot.argmin()); Dm_min=float((0.5-Pm[imin,neg]).sum()); Hcrit=2*Dm_min
        rows.append(dict(T=T,H=H,n_mpos=n_mpos,n_phalf=n_phalf,mis_mneg_phalf=mis1,mis_mpos_plow=mis2,
            maxF=float(F.max()),maxf=float(f.max()),nF_half=int((F>=0.5).sum()),nf_half=int((f>=0.5).sum()),
            Dminus=Dm,Dplus=Dp,Hcrit_frac=Hcrit/max(H,1),cotmin=float(cot.min()),lam_med_neg=float(np.median(lam[neg])) if neg.any() else None))
    out[tag]=rows
    r={x['T']:x for x in rows}
    ph2=[x for x in rows if x['T']>=11]
    print(f"== {tag}: cot_min {cot.min():.2f}  trS/|xbar| {np.sqrt(trS)/nx:.2f}")
    for T in (0,1,2,3,10):
        if T in r:
            x=r[T]; print(f"  T{T:<3d} H{x['H']:3d} m+ {x['n_mpos']:3d} p>½ {x['n_phalf']:3d} mis(m<0,p≥½) {x['mis_mneg_phalf']:3d} mis(m≥0,p<½) {x['mis_mpos_plow']:3d} | maxF {x['maxF']:.2f} maxf {x['maxf']:.2f} #F≥½ {x['nF_half']:4d} #f≥½ {x['nf_half']:4d} | D- {x['Dminus']:5.1f} D+ {x['Dplus']:5.1f} Hcrit/H {x['Hcrit_frac']:.2f}")
    print(f"  ph2 (90 states): m+>0 in {sum(1 for x in ph2 if x['n_mpos']>0)}, mis(m<0,p≥½)>0 in {sum(1 for x in ph2 if x['mis_mneg_phalf']>0)}, max maxF {max(x['maxF'] for x in ph2):.2f}, max maxf {max(x['maxf'] for x in ph2):.2f}, #F≥½>0 in {sum(1 for x in ph2 if x['nF_half']>0)}, #f≥½>0 in {sum(1 for x in ph2 if x['nf_half']>0)}, Hcrit/H min {min(x['Hcrit_frac'] for x in ph2):.2f} med {np.median([x['Hcrit_frac'] for x in ph2]):.2f}, lam_med {np.median([x['lam_med_neg'] for x in ph2 if x['lam_med_neg'] is not None]):.2f}")
json.dump(out,open("/home/issan/Projects/claude/proj_004_drift/results/xbar_arrangement_1014/fable_graded_prop2.json","w"))
