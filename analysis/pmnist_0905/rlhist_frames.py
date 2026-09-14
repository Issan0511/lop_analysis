import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
from pathlib import Path
SRC=Path("results/_diag_rlhist_0907")
ARMS=["R","LR","SNA","R_l2","LR_l2","SNA_l2","R_l2init"]
C={"R":"#d1495b","LR":"#2e86ab","SNA":"#2a9d5c","R_l2":"#e07a5f","LR_l2":"#4895ef","SNA_l2":"#52b788","R_l2init":"#9d4edd"}
N={"R":"R  ReLU","LR":"LR  leaky .1","SNA":"SNA  adaptive α","R_l2":"R + l2","LR_l2":"LR + l2","SNA_l2":"SNA + l2","R_l2init":"R + l2init"}
F={"R":0.114,"LR":0.808,"SNA":0.986,"R_l2":0.941,"LR_l2":0.950,"SNA_l2":0.964,"R_l2init":0.958}
D={a:np.load(SRC/f"{a}.npz") for a in ARMS}
E=D["R"]["edges"]; ctr=(E[:-1]+E[1:])/2; zz=np.linspace(E[0],E[-1],1400)
def dphi(a,z):
    if a.startswith("R"): return (z>0).astype(float)
    if a.startswith("LR"): return np.where(z>0,1.0,0.1)
    return 1+np.sin(2*0.6*z)
TS=[0,2,9,49]
fig,ax=plt.subplots(len(TS),len(ARMS),figsize=(3.05*len(ARMS),3.0*len(TS)),sharex=True)
for r,t in enumerate(TS):
    for j,a in enumerate(ARMS):
        A=ax[r,j]; g=A.twinx(); g.plot(zz,dphi(a,zz),color="#c9a227",lw=.9,alpha=.75); g.set_ylim(-.15,2.3); g.set_yticks([])
        h=D[a]["h1"][t].astype(float); h/=max(h.max(),1); A.plot(ctr,h,color=C[a],lw=1.0,alpha=.45)
        mh,_=np.histogram(D[a]["m1"][t],bins=E); mh=mh.astype(float)/max(mh.max(),1)
        A.bar(ctr,mh,width=E[1]-E[0],color=C[a],alpha=.85)
        A.axvline(0,color="k",lw=.7,ls=":"); A.set_xlim(-14,6); A.set_ylim(0,1.05); A.set_yticks([])
        A.text(.03,.86,f"z̄ {np.median(D[a]['m1'][t]):+.2f}\nmemo {D[a]['acc'][t]:.2f}",transform=A.transAxes,fontsize=8.5,color=C[a])
        if r==0: A.set_title(f"{N[a]}\nfinal {F[a]:.3f}",fontsize=10.5,color=C[a],pad=6)
        if j==0: A.set_ylabel(f"task {t+1}",fontsize=11)
        if r==len(TS)-1: A.set_xlabel("preactivation z (layer 1)",fontsize=9)
fig.suptitle("Random Label MNIST — layer-1 preactivation. bars: 100 per-unit means | faint: pooled z | gold: φ'(z).  seed 0",y=.995,fontsize=12)
fig.tight_layout(); fig.savefig("results/_diag_rlhist_0907/frames_rl.png",dpi=100,bbox_inches="tight"); print("wrote frames_rl.png")
