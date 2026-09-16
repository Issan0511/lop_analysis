"""Figures of preregistered statistics, without additional tests or selection."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/unit_label_demand_0916"
plt.rcParams.update({"font.family":"Noto Sans CJK JP","font.size":11,"axes.unicode_minus":False,
                     "axes.spines.top":False,"axes.spines.right":False})
with open(OUT/"group_summary.csv") as f: rows=list(csv.DictReader(f))
colors={"f_support":"#7057a5","f_suppress":"#cf6429","f_cert":"#20877f"}
labels={"f_support":"旧ラベルを支える割合","f_suppress":"抑制要求が来る割合","f_cert":"十分条件で保証できる割合"}
fig,axes=plt.subplots(1,2,figsize=(12,5.8),sharex=True,layout="constrained")
for layer,ax in enumerate(axes,1):
    names=[]
    for k,(act,group) in enumerate((a,g) for a in ("GELU","SILU") for g in ("P","B","V")):
        rr=[r for r in rows if r["act"]==act and r["layer"]==str(layer) and r["group"]==group
            and r["window"]=="t2-10" and r["step"]=="0"]
        names.append(f"{act}  "+{"P":"正側","B":"帯","V":"谷の奥"}[group])
        for offset,key in zip((-.18,0,.18),colors):
            vals=100*np.array([float(r[key]) for r in rr]); mid=np.median(vals)
            ax.errorbar(mid,k+offset,xerr=[[mid-vals.min()],[vals.max()-mid]],fmt="o",capsize=3,
                        color=colors[key],label=labels[key] if k==0 and layer==1 else None)
    ax.set(yticks=range(6),yticklabels=names,xlim=(-1,101),xlabel="画像×個体の組に対する割合 (%)",title=f"第{layer}隠れ層")
    ax.invert_yaxis();ax.grid(axis="x",alpha=.2);ax.axvline(50,color="#cccccc",linestyle=":")
fig.suptitle("ラベル切替直後、どこに抑制要求が来るか\nRL-MNIST・task 2–10・非ゼロ活性のみ・点=3 seed中央値、線=最小～最大",fontsize=14)
fig.legend(loc="outside lower center",ncol=3,frameon=False)
fig.savefig(OUT/"allocation_at_switch.png",dpi=180)
fig.savefig(OUT/"allocation_at_switch.pdf")
plt.close(fig)

fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True,sharey=True,layout="constrained")
for ai,act in enumerate(("GELU","SILU")):
    for li,layer in enumerate((1,2)):
        ax=axes[ai,li]
        for group,color in zip(("P","B","V"),("#cf6429","#7057a5","#20877f")):
            steps=(0,25,100,300,1000,6000); yy=[]
            for step in steps:
                vals=[float(r["S"]) for r in rows if r["act"]==act and r["layer"]==str(layer) and r["group"]==group
                      and r["window"]=="t2-10" and r["step"]==str(step)]
                yy.append(np.median(vals))
            ax.plot(range(len(steps)),yy,"o-",color=color,label={"P":"正側","B":"帯","V":"谷の奥"}[group])
        ax.set(title=f"{act}・第{layer}隠れ層",xticks=range(6),xticklabels=steps,ylim=(-1.05,1.05))
        ax.axhline(0,color="#999999",linewidth=.8);ax.grid(alpha=.15)
        if ai==1:ax.set_xlabel("切替後の更新数（測定点を等間隔で表示）")
        if li==0:ax.set_ylabel("要求の偏り S\n＋: 抑制優勢 ／ −: 増幅優勢")
axes[0,0].legend(frameon=False,ncol=3)
fig.suptitle("切替後も抑制優勢が続くか\n各時点の群で再集計・task 2–10・3 seed中央値・S = Σ(a u) / Σ|a u|",fontsize=14)
fig.savefig(OUT/"allocation_over_time.png",dpi=180)
plt.close(fig)
