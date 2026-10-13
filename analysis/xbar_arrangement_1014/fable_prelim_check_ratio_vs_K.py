import sys, json, numpy as np
sys.path.insert(0, "/home/issan/Projects/obsidian-research-data/push_direction_proof_0930")
from common import load_X
ROOT="/home/issan/Projects/obsidian-research-data/push_direction_proof_0930/"
S10=ROOT+"review/stage4_1002/block_1003/stage10_1005/"
def spearman(x,y):
    rx=np.argsort(np.argsort(x)); ry=np.argsort(np.argsort(y)); return np.corrcoef(rx,ry)[0,1]
for tag,(src,task,a) in {"a01":("review/opus_bwdfloor/out/LR_ref_s0.npz",40,0.1),"a03":("review/opus_lrslope/out/LRa0.3_s0.npz",100,0.3),"a05":("review/opus_lrslope/out/LRa0.5_s0.npz",100,0.5)}.items():
    rows=json.load(open(S10+f"p149_expect_{tag}.json")); d=np.load(ROOT+src); X=load_X(d["subset"]); N=len(X)
    Xt=np.c_[X,np.ones(N)]; xbar=Xt.mean(0); K=Xt@xbar; Kb=xbar@xbar
    W=np.c_[d["end0"][task-2].astype(np.float64), d["end1"][task-2].astype(np.float64)]; Z=Xt@W.T; m=W@xbar; s=Z.std(0)
    tau=0.1*s+1e-12; sg=1/(1+np.exp(-Z/tau)); Dp=(1-a)*sg*(1-sg)/tau; ep=K[:,None]*Dp
    units=np.array([r["unit"] for r in rows]); Ag=np.array([r["Ag"] for r in rows]); Ef=np.array([r["Ef"] for r in rows]); Aw=np.array([r["Aw"] for r in rows]); Eb=np.array([r["Eb"] for r in rows])
    # e'-weighted mean brightness of the unit's threshold measure, and K of top-3
    Kthr=np.array([ (ep[:,i]@K)/ep[:,i].sum() for i in units]); Ktop3=np.array([K[np.argsort(-ep[:,i])[:3]].mean() for i in units])
    p=(Z>0).mean(0)[units]; r=(m/s)[units]
    # dark-coverage: occupancy of the unit's top-3 threshold inputs by other alive units
    alive=(Z>0).any(0); f=(Z[:,alive]>0).mean(1)
    fthr=np.array([f[np.argsort(-ep[:,i])[:3]].mean() for i in units])
    rf=Ag/Ef; rb=Aw/Eb
    print(f"{tag}: n={len(units)} | ratio full median {np.median(rf):.2f}, probe {np.median(rb):.2f}")
    print(f"   Spearman(ratio_full, K_thr e'-weighted) = {spearman(rf,Kthr):+.2f}; (ratio_full, K top3) = {spearman(rf,Ktop3):+.2f}; (ratio_full, f at top3) = {spearman(rf,fthr):+.2f}; (ratio_full, p) = {spearman(rf,p):+.2f}; (ratio_full, r) = {spearman(rf,r):+.2f}")
    print(f"   Spearman(ratio_probe, K_thr) = {spearman(rb,Kthr):+.2f}; (ratio_probe, f at top3) = {spearman(rb,fthr):+.2f}; (ratio_probe, p) = {spearman(rb,p):+.2f}; (ratio_probe, r) = {spearman(rb,r):+.2f}")
    print(f"   K_thr: mean {Kthr.mean():.2f} (all K mean {K.mean():.2f}); f at top3 mean {fthr.mean():.4f} vs mean occupancy {f.mean():.4f}; ratio by K_thr tercile (full): " + ", ".join(f"{np.median(rf[idx]):.2f}" for idx in np.array_split(np.argsort(Kthr),3)) + " | probe: " + ", ".join(f"{np.median(rb[idx]):.2f}" for idx in np.array_split(np.argsort(Kthr),3)))
