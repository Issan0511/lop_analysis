#!/usr/bin/env python3
"""Standalone figures from the frozen-state carrier CSVs."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/chargpt_phase2carrier_1024'
TASKS=[1,4,5,6,9,39,50,89,100]
rows=list(csv.DictReader((OUT/'carriers.csv').open()))
comps=list(csv.DictReader((OUT/'comparisons.csv').open()))
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,2,figsize=(11,7.5),layout='constrained')
colors={'full':'#222222','own':'#ce6500','upstream':'#17845c'}
for layer in [0,1]:
 for j,metric in enumerate(['SGD','Adam_fixed']):
  ax=axes[layer,j]
  for route,label in [('full','All parameters'),('own','Own up row'),('upstream','Upstream sum')]:
   aa=np.array([[float(r['mean']) for r in rows if (int(r['layer']),r['metric'],r['signal'],r['route'],int(r['task']))==(layer,metric,'switch',route,t)] for t in TASKS])
   assert aa.shape==(9,10)
   ax.plot(TASKS,np.median(aa,axis=1),'-o',markersize=3,color=colors[route],label=label)
   ax.fill_between(TASKS,aa.min(1),aa.max(1),color=colors[route],alpha=.10)
  ax.axhline(0,color='.5',lw=.7)
  ax.set_xscale('log');ax.set_xticks([1,4,9,39,100],labels=['1','4','9','39','100'])
  ax.set_yscale('symlog',linthresh=.01 if metric=='SGD' else 1)
  ax.set_title(f'C3 L{layer} / {metric}')
  ax.set_xlabel('End of task');ax.set_ylabel('Mean S_r over all 512 units\npositive = sinking')
  ax.grid(alpha=.15)
axes[0,0].legend(fontsize=9)
fig.suptitle('Actual expected switch gradient: carrier contributions\nLines: median over 10 seeds; bands: full seed range. No training updates.')
fig.savefig(OUT/'carrier_means.png',dpi=180)
fig.savefig(OUT/'carrier_means.pdf')
plt.close(fig)

fig,axes=plt.subplots(2,2,figsize=(11,7.5),layout='constrained')
for layer in [0,1]:
 for j,metric in enumerate(['SGD','Adam_fixed']):
  ax=axes[layer,j]
  for route,label in [('full','All parameters'),('own','Own up row'),('upstream','Upstream sum')]:
   aa=[]
   for t in TASKS:
    rr=[r for r in comps if (int(r['layer']),r['metric'],r['predictor'],r['actual'],r['route'],r['subset'],float(r['zero_tol']),int(r['task']))==(layer,metric,'P2','switch',route,'all',1e-10,t)]
    vals=np.array([float(r['ba']) if r['ba'] else np.nan for r in rr])
    aa.append(vals)
   aa=np.array(aa)
   # No BA is defined when either actual sign has fewer than 10 units.
   good=np.isfinite(aa).any(1)
   center=np.full(9,np.nan);lo=center.copy();hi=center.copy()
   for i in np.where(good)[0]:
    vv=aa[i,np.isfinite(aa[i])];center[i]=np.median(vv);lo[i]=vv.min();hi[i]=vv.max()
   ax.plot(TASKS,center,'-o',markersize=3,color=colors[route],label=label)
   ax.fill_between(TASKS,lo,hi,color=colors[route],alpha=.10)
  ax.axhline(.5,color='.5',lw=.7,ls='--')
  ax.set_xscale('log');ax.set_xticks([1,4,9,39,100],labels=['1','4','9','39','100'])
  ax.set_ylim(0,1);ax.set_title(f'C3 L{layer} / {metric}')
  ax.set_xlabel('End of task');ax.set_ylabel('Phase-2 vs actual switch\nbalanced accuracy')
  ax.grid(alpha=.15)
axes[0,0].legend(fontsize=9)
fig.suptitle('Direction agreement must be checked on the same carrier\nLines and bands summarize only seeds with defined BA; denominators are in comparisons.csv.')
fig.savefig(OUT/'carrier_agreement.png',dpi=180)
fig.savefig(OUT/'carrier_agreement.pdf')
plt.close(fig)
