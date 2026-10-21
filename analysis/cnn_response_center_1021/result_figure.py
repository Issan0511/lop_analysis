#!/usr/bin/env python3
"""Static figure from saved CNN response outputs only; no model imports."""
from __future__ import annotations
import os
os.environ.setdefault("MPLCONFIGDIR","/tmp/matplotlib_cnn_response_1021")
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
RESULTS=ROOT/"results/cnn_response_center_1021"
report=json.loads((HERE/"result_report.json").read_text())
BLUE="#2563a6";GREEN="#247b62";GRAY="#616875"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,
                     "axes.spines.right":False,"axes.labelcolor":"#303640","text.color":"#20252d",
                     "xtick.color":GRAY,"ytick.color":GRAY,"pdf.fonttype":42})
fig,(left,right)=plt.subplots(1,2,figsize=(12.6,5.2),gridspec_kw={"width_ratios":[.84,1.6]})
fig.subplots_adjust(left=.07,right=.98,bottom=.16,top=.81,wspace=.28)
fig.suptitle("CNN response-field intervention: one direction supported",x=.07,ha="left",y=.975,fontsize=17,fontweight="bold")
fig.text(.07,.915,"RL-CIFAR task 31 · LeakyReLU(0.1) · 10 paired seeds · c1 only · matched initial features and predictions",fontsize=10.3,color=GRAY)

left.axhline(0,color=GRAY,lw=1,alpha=.7)
for x,key,color in ((0,"rescue",BLUE),(1,"reverse",GREEN)):
    q=report["primary_inference"][key]
    values=np.array([q["seed_differences"][str(s)] for s in range(10,20)])*100
    jitter=np.linspace(-.12,.12,10)
    left.scatter(x+jitter,values,c=color,s=32,alpha=.72,zorder=3,edgecolors="white",linewidths=.4)
    mean=q["mean_difference"]*100;lo,hi=np.array(q["ci95_seed_bootstrap"])*100
    left.errorbar(x+.24,mean,yerr=np.array([[mean-lo],[hi-mean]]),fmt="D",markersize=7,
                  color=color,ecolor=color,elinewidth=2,capsize=5,zorder=4)
    left.text(x,5.52,f"{q['n_positive']}/10 expected direction\nHolm p = {q['p_holm']:.4g}",ha="center",va="top",fontsize=9.2,color=color)
left.set_ylim(-2.65,5.8);left.set_xlim(-.38,1.5)
left.set_xticks([0,1],["Rescue of LR\nswap − self","Reverse on LRc\nself − swap"])
left.set_ylabel("Initial 3,000-update accuracy difference (percentage points)")
left.set_title("A  Registered primary contrasts",loc="left",fontweight="bold",pad=12)
left.grid(axis="y",alpha=.15);left.set_axisbelow(True)
left.text(.02,-.18,"Dots: seeds. Diamonds: means.\nBars: seed-bootstrap 95% CI (20,000 resamples).",transform=left.transAxes,fontsize=8.7,color=GRAY,va="top")

arms=("LR_self","LR_swap","LRc_self","LRc_swap")
colors={"LR_self":BLUE,"LR_swap":BLUE,"LRc_self":GREEN,"LRc_swap":GREEN}
styles={"LR_self":"-","LR_swap":"--","LRc_self":"-","LRc_swap":"--"}
xs=(np.arange(40)*750+375)/1000
ys={}
right.axvspan(0,3,color="#a7b5dc",alpha=.20,lw=0)
right.axvspan(27,30,color="#e8c889",alpha=.22,lw=0)
for a in arms:
    c=np.load(RESULTS/a/"batch_curves.npz")
    assert c["accuracy"].shape==(10,30000)
    ys[a]=c["accuracy"].reshape(10,40,750).mean(axis=2,dtype=np.float64).mean(axis=0)*100
    right.plot(xs,ys[a],styles[a],color=colors[a],lw=2,label=a.replace("_"," "))
right.set_xlim(0,30);right.set_ylim(5,103)
right.set_xlabel("Updates on task 31 (thousands)");right.set_ylabel("Online accuracy (%)")
right.set_title("B  Learning curves and the late cost",loc="left",fontweight="bold",pad=12)
right.grid(axis="y",alpha=.15);right.set_axisbelow(True)
right.legend(loc="lower right",ncol=2,frameon=False,fontsize=9)
right.text(1.5,8,"Primary\nwindow",ha="center",fontsize=8.5,color=GRAY)
right.text(28.5,25,"Late\nwindow",ha="center",fontsize=8.5,color=GRAY)

inset=right.inset_axes([.45,.32,.51,.34])
for a in arms:
    inset.plot(xs[-4:],ys[a][-4:],styles[a],color=colors[a],lw=1.8)
inset.set_xlim(27,30);inset.set_ylim(95.8,100.35)
inset.set_xticks([27,28.5,30]);inset.set_yticks([96,98,100]);inset.tick_params(labelsize=8)
inset.grid(axis="y",alpha=.15);inset.set_title("Last 3,000 updates: LR swap − self = −1.544 pp",fontsize=8.4,pad=5)
right.text(.015,-.18,"Curves: mean of 10 seeds in non-overlapping 750-update bins.\nLate-window difference is descriptive and does not replace the primary test.",transform=right.transAxes,fontsize=8.7,color=GRAY,va="top")
for ext in ("png","pdf"):
    fig.savefig(HERE/f"result_figure.{ext}",dpi=200,facecolor="white",bbox_inches="tight")
plt.close(fig)
print(HERE/"result_figure.png")
