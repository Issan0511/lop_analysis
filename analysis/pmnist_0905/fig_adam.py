import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, pandas as pd, numpy as np, glob, sys
a=pd.concat([pd.read_csv(f) for f in glob.glob("results/_diag_adam_0905/shard*/per_task.csv")])
a=a[a.acc.notna()]
sgd=pd.concat([pd.read_csv(f) for f in glob.glob("results/pmnist_main_0905/**/per_task.csv",recursive=True)])
sgd=sgd[(sgd.acc.notna())&(sgd.lr==0.1)]
ARMS=["R","LR","SN3","SN1","LIN"]
C={"R":"#d1495b","LR":"#2e86ab","SN3":"#2a9d5c","SN1":"#8b5cf6","LIN":"#6b7280"}
N={"R":"R (ReLU)","LR":"LR (leaky .1)","SN3":"SN3 (Snake α=3)","SN1":"SN1 (Snake α=1)","LIN":"LIN"}
def full(df,arm):
    r=[g.sort_values("task") for _,g in df[df.arm==arm].groupby("seed") if g.task.max()==200]
    return r
fig,ax=plt.subplots(1,4,figsize=(19,4.5))
for arm in ARMS:
    r=full(a,arm)
    if not r: continue
    t=r[0].task.values; c=C[arm]
    acc=np.median([x.acc.values for x in r],axis=0)
    ax[0].plot(t,pd.Series(acc).rolling(10,min_periods=1).mean(),color=c,lw=1.8,label=f"{N[arm]}  (n={len(r)})")
    ax[1].plot(t,np.median([x.mob_l1.values for x in r],axis=0),color=c,lw=1.8)
    ax[2].plot(t,np.median([x.zbar_l1.values for x in r],axis=0),color=c,lw=1.8)
    rs=full(sgd,arm)
    if rs:
        ax[3].plot(t,pd.Series(np.median([x.acc.values for x in rs],axis=0)).rolling(10,min_periods=1).mean(),
                   color=c,lw=1.8)
# gap annotation
rl=full(a,"LR"); rs3=full(a,"SN3")
if rl and rs3:
    gl=pd.Series(np.median([x.acc.values for x in rl],axis=0)).rolling(10,min_periods=1).mean()
    gs=pd.Series(np.median([x.acc.values for x in rs3],axis=0)).rolling(10,min_periods=1).mean()
    ax[0].fill_between(rl[0].task.values,gs,gl,where=(gl>gs),color="#2e86ab",alpha=.10)
    ax[0].annotate("gap LR − SN3\n0.028 → 0.009\n(still closing at task 200)",
                   xy=(185,(gl.iloc[-1]+gs.iloc[-1])/2),xytext=(105,0.868),fontsize=8.5,
                   arrowprops=dict(arrowstyle="->",lw=.9,color="#333"))
ax[0].set(xlabel="task",ylabel="test accuracy (10-task mean)",title="Adam lr=0.001 — accuracy")
ax[1].set(xlabel="task",ylabel=r"median$_i$E$[\varphi'(z_i)]$",yscale="log",title="Adam — mobility, layer 1")
ax[2].set(xlabel="task",ylabel=r"median$_i$E$[z_i]$",title="Adam — preactivation mean, layer 1")
ax[3].set(xlabel="task",ylabel="test accuracy (10-task mean)",title="for comparison: plain SGD lr=0.1")
ax[2].axhline(0,color="k",lw=.7,ls=":")
ax[0].legend(fontsize=8.5,loc="lower left")
for x in ax: x.grid(alpha=.25)
fig.suptitle("Adam lr=0.001 (the regime the reference paper uses) vs the plain-SGD box — "
             "Permuted MNIST, 784-100-100-10, batch 16, 200 tasks, medians. UNREGISTERED diagnostic",y=1.03)
fig.tight_layout(); fig.savefig(sys.argv[1],dpi=130,bbox_inches="tight"); print("wrote",sys.argv[1])
