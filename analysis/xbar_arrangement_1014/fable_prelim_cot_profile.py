import sys, numpy as np
sys.path.insert(0, "/home/issan/Projects/obsidian-research-data/push_direction_proof_0930")
from common import load_X
ROOT="/home/issan/Projects/obsidian-research-data/push_direction_proof_0930/"
rng=np.random.default_rng(1)
def spearman(x,y):
    rx=np.argsort(np.argsort(x)); ry=np.argsort(np.argsort(y)); return np.corrcoef(rx,ry)[0,1]
for path,label,T in [("review/opus_bwdfloor/out/LR_ref_s0.npz","LR a0.1 s0",50),("review/opus_lrslope/out/LRa0.3_s0.npz","LR a0.3 s0",50),("review/opus_lrslope/out/LRa0.5_s0.npz","LR a0.5 s0",50),("review/opus_bwdfloor/out/ELU_ref_s0.npz","ELU s0",50)]:
    d=np.load(ROOT+path); X=load_X(d["subset"]); N=len(X); Xt=np.c_[X,np.ones(N)]; xbar=Xt.mean(0); K=Xt@xbar; Kb=xbar@xbar
    Xp=Xt-np.outer(K/Kb,xbar); nperp=np.linalg.norm(Xp,axis=1); cot=K/(np.sqrt(Kb)*nperp)   # cot theta_n
    W=np.c_[d["end0"][T-1].astype(np.float64), d["end1"][T-1].astype(np.float64)]; Z=Xt@W.T; m=W@xbar; U=Z-np.outer(K/Kb,m)
    opn=Z>0; alive=opn.any(0); f=opn[:,alive].mean(1); su=U.std(0)
    # iso surrogate occupancy: random directions, matched s_perp and actual m
    R=20; fiso=np.zeros(N)
    for r in range(R):
        G=rng.standard_normal((785,alive.sum())); G-=np.outer(xbar,(xbar@G)/Kb); Ug=Xt@G; Ug*= (su[alive]/Ug.std(0))[None,:]
        Zg=np.outer(K/Kb,m[alive])+Ug; fiso+=(Zg>0).mean(1)/R
    # per-unit widths in iso vs actual
    print(f"{label} T={T}: Spearman(f,K) {spearman(f,K):+.3f}  Spearman(f,cot) {spearman(f,cot):+.3f}  Spearman(f, f_iso) {spearman(f,fiso):+.3f} | corr(cot,K) {np.corrcoef(cot,K)[0,1]:.2f} | sd(u_perp)/|x_perp| const? corr over n of sd_k-avg |u| with nperp: {np.corrcoef(np.abs(U[:,alive]).mean(1), nperp)[0,1]:.2f}")
    o=np.argsort(K); bins=np.array_split(o,8)
    print("   octile(K): f_actual | f_iso | ratio :", " ".join(f"{f[b].mean():.4f}/{fiso[b].mean():.4f}/{f[b].mean()/fiso[b].mean():.2f}" for b in bins))
    print(f"   max f actual {f.max():.3f} (K {K[f.argmax()]:.1f}, cot {cot[f.argmax()]:.2f}) | max f_iso {fiso.max():.3f} | #f>=.5 actual {(f>=.5).sum()} iso {(fiso>=.5).sum()} | mean p actual {opn[:,alive].mean():.4f} iso {fiso.mean():.4f}")
