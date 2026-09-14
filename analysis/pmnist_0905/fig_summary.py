import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, pandas as pd, numpy as np, glob, sys
df=pd.concat([pd.read_csv(f) for f in glob.glob("results/pmnist_main_0905/**/per_task.csv",recursive=True)])
df=df[df.acc.notna()]
ARMS=["R","LR","SN3","SN1","LIN"]
C={"R":"#d1495b","LR":"#2e86ab","SN3":"#2a9d5c","SN1":"#8b5cf6","LIN":"#6b7280"}
N={"R":"R (ReLU)","LR":"LR (leaky .1)","SN3":"SN3 (Snake α=3)","SN1":"SN1 (Snake α=1)","LIN":"LIN (identity)"}
SEL={"R":0.1,"LR":0.1,"SN3":0.02,"SN1":0.05,"LIN":0.05}
lrs=sorted(df.lr.unique())
def w(arm,lr,c="acc"):
    g=df[(df.lr==lr)&(df.arm==arm)&(df.task>=151)&(df.task<=200)]
    f=[s for s,gg in g.groupby("seed") if len(gg)==50]
    return g[g.seed.isin(f)].groupby("seed")[c].mean()
def stat(arm,lr,fn):
    r=[g.sort_values("task") for _,g in df[(df.lr==lr)&(df.arm==arm)].groupby("seed") if g.task.max()==200]
    return np.median([fn(x) for x in r]) if r else np.nan

fig,ax=plt.subplots(1,4,figsize=(19,4.4))
for arm in ARMS:
    y=[w(arm,l).median() for l in lrs]; c=C[arm]
    ax[0].plot(lrs,y,"o-",color=c,lw=1.8,ms=5,label=N[arm])
    ax[0].plot([SEL[arm]],[w(arm,SEL[arm]).median()],"o",ms=13,mfc="none",mec=c,mew=2)
    ax[1].plot(lrs,[stat(arm,l,lambda x:x.acc.rolling(10,min_periods=10).mean().max()-x.acc.tail(10).mean()) for l in lrs],"o-",color=c,lw=1.8,ms=5)
    ax[2].plot(lrs,[stat(arm,l,lambda x:x.mob_l1.tail(10).mean()) for l in lrs],"o-",color=c,lw=1.8,ms=5)
    ax[3].plot(lrs,[stat(arm,l,lambda x:x.w_norm_l1.tail(10).mean()/x.w_norm_l1.iloc[0]) for l in lrs],"o-",color=c,lw=1.8,ms=5)
lin=[stat("LIN",l,lambda x:x.acc.rolling(10,min_periods=10).mean().max()-x.acc.tail(10).mean()) for l in lrs]
ax[1].fill_between(lrs,0,lin,color="#6b7280",alpha=.18)
ax[1].text(0.0055,0.004,"the gate-free floor:\nLIN loses this much with\nmobility exactly 1.0000",fontsize=8,color="#374151")
ax[0].set(xscale="log",xlabel="learning rate",ylabel="test accuracy, tasks 151-200",
          title="retention (ring = the lr stage 1 chose)")
ax[1].set(xscale="log",xlabel="learning rate",ylabel="peak − final",title="how much plasticity was lost")
ax[2].set(xscale="log",yscale="log",xlabel="learning rate",ylabel=r"median$_i$E$[\varphi'(z_i)]$",title="mobility, layer 1")
ax[3].set(xscale="log",xlabel="learning rate",ylabel="‖w‖ ratio to task 1",title="weight norm growth, layer 1")
ax[0].legend(fontsize=8.5,loc="lower center")
for a in ax: a.grid(alpha=.25,which="both")
fig.suptitle("pmnist_main_0905 — 5 arms × 5 learning rates × 10 seeds × 200 tasks, medians. "
             "Permuted MNIST, 784-100-100-10, plain SGD, batch 16",y=1.02)
fig.tight_layout(); fig.savefig(sys.argv[1],dpi=130,bbox_inches="tight"); print("wrote",sys.argv[1])
