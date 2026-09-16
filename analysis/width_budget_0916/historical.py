"""Supplemental horizon dependence from old archived logs; no replay selection."""
import numpy as np
import pandas as pd
from run import OLD,OUT,ARMS

rows=[]
for arm in ARMS:
    for seed in range(10):
        with np.load(OLD/'logs'/f'{arm}_seed{seed}.npz') as d:
            a=float(d['act_alpha']);wn=d['layer1_w_norm'].astype(float);wf=d['layer1_w_free'].astype(float)
            for task in [1,2,5,10,20,50,100,110,200,500]:
                ix=int(np.flatnonzero(d['step']==task*10000)[0]);j=int(np.flatnonzero(d['layer1_w_free_step']==task*10000)[0])
                for geom,n0,n1 in [('full',wn[0]**2,wn[ix]**2),('variance',(wf[0]**2).sum(1)/4,(wf[j]**2).sum(1)/4)]:
                    rows.append(dict(a=a,seed=seed,horizon=task,geometry=geom,
                        mean_net=float(np.mean(n1-n0)),normalized_net=float(np.mean(n1-n0)/np.mean(n0)),
                        median_size_ratio=float(np.median(np.sqrt(n1/n0))),unit_shrink_fraction=float(np.mean(n1<n0))))
pd.DataFrame(rows).to_csv(OUT/'historical_horizons.csv',index=False)
print(pd.DataFrame(rows).groupby(['geometry','horizon','a']).median_size_ratio.median().unstack('a').round(3).to_string())
