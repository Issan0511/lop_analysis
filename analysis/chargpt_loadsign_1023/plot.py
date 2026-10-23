#!/usr/bin/env python3
"""Descriptive per-seed window plot; no smoothing or inferential intervals."""
import os
os.environ['MPLCONFIGDIR']='/tmp/chargpt_loadsign_1023_matplotlib'
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parents[2]/'results/chargpt_loadsign_1023'
rows=list(csv.DictReader((OUT/'recombination_windows.csv').open()))
series=[('s_other0_residual0_radial0','Self output only','#1e8d68'),
        ('s_other0_residual1_radial0','+ Fixed residual input','#a08a37'),
        ('s_other1_residual1_radial0','+ Other unit outputs','#dc8a35'),
        ('s_other1_residual1_radial1','+ LN variance derivative (observed)','#3765a5')]
fig,axes=plt.subplots(1,2,figsize=(11,5.4),sharey=True)
for ax,arm in zip(axes,['C3','ReLUcap']):
    for j,(key,label,color) in enumerate(series):
        xs=np.arange(3)+(j-1.5)*.09
        vals=np.array([[float(r[key]) for r in rows if r['arm']==arm and r['window']==w] for w in ['early','middle','late']])
        med=np.median(vals,axis=1)
        ax.errorbar(xs,med,yerr=np.stack([med-vals.min(1),vals.max(1)-med]),fmt='o-',ms=4,lw=1.6,capsize=3,color=color,label=label)
        for x,ys in zip(xs,vals):ax.scatter(np.full(10,x),ys,s=8,color=color,alpha=.35,zorder=2)
    ax.axhline(.5,color='#666',ls='--',lw=.8)
    ax.set(title=arm+' / final FFN',xticks=range(3),xticklabels=['Tasks 1–9','Tasks 39–50','Tasks 89–100'],ylim=(-.02,1.04))
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',alpha=.15)
axes[0].set_ylabel('Fraction of alive units with a sinking push')
handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.045),ncol=2,frameon=False,fontsize=9)
fig.suptitle('Prediction-side decomposition of the loading signal',fontsize=13)
fig.text(.5,.015,'Dots: 10 seed-specific window medians. Bars: seed range. Exploratory decomposition; no training.',ha='center',fontsize=8,color='#555')
fig.tight_layout(rect=[0,.19,1,.95])
fig.savefig(OUT/'signal_components.png',dpi=170,bbox_inches='tight')
fig.savefig(OUT/'signal_components.svg',bbox_inches='tight')
