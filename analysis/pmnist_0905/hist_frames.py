import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
from pathlib import Path
SRC=Path("results/_diag_hist_0905"); ARMS=["R","LR","SN3","SN1","LIN"]
C={"R":"#d1495b","LR":"#2e86ab","SN3":"#2a9d5c","SN1":"#8b5cf6","LIN":"#6b7280"}
N={"R":"R  ReLU","LR":"LR  leaky 0.1","SN3":"SN3  Snake α=3","SN1":"SN1  Snake α=1","LIN":"LIN  identity"}
D={a:np.load(SRC/f"{a}.npz") for a in ARMS}; E=D["R"]["edges"]; ctr=(E[:-1]+E[1:])/2
zz=np.linspace(E[0],E[-1],1200)
def dphi(a,z):
    return {"R":(z>0).astype(float),"LR":np.where(z>0,1.0,0.1),
            "SN3":1+np.sin(6*z),"SN1":1+np.sin(2*z)}.get(a,np.ones_like(z))
TS=[0,24,99,199]
fig,ax=plt.subplots(len(TS),5,figsize=(19.5,11.5),sharex=True)
for r,t in enumerate(TS):
    for j,a in enumerate(ARMS):
        A=ax[r,j]; g=A.twinx(); g.plot(zz,dphi(a,zz),color="#c9a227",lw=1.0,alpha=.8)
        g.set_ylim(-.15,2.3); g.set_yticks([])
        h=D[a]["h1"][t].astype(float); h/=max(h.max(),1)
        A.plot(ctr,h,color=C[a],lw=1.0,alpha=.45)
        mh,_=np.histogram(D[a]["m1"][t],bins=E); mh=mh.astype(float)/max(mh.max(),1)
        A.bar(ctr,mh,width=E[1]-E[0],color=C[a],alpha=.85)
        A.axvline(0,color="k",lw=.7,ls=":"); A.set_xlim(-13,6); A.set_ylim(0,1.05); A.set_yticks([])
        zb=np.median(D[a]["m1"][t])
        A.text(.03,.88,f"z̄ med {zb:+.2f}",transform=A.transAxes,fontsize=9,color=C[a])
        if r==0: A.set_title(N[a],fontsize=12,color=C[a],pad=8)
        if j==0: A.set_ylabel(f"task {t+1}",fontsize=12)
        if r==len(TS)-1: A.set_xlabel("preactivation z (layer 1)",fontsize=9)
fig.suptitle("Layer-1 preactivation, Adam lr=0.001, seed 0 — bars: the 100 per-unit means z̄ᵢ | "
             "faint: pooled z | gold: φ'(z)",y=.995,fontsize=13)
fig.tight_layout(); fig.savefig("results/_diag_hist_0905/frames.png",dpi=100,bbox_inches="tight")
print("wrote frames.png")
