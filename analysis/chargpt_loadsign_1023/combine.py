#!/usr/bin/env python3
"""Exploratory recombination of already measured additive signal terms."""
import csv
import json
import itertools
import numpy as np
from report import OUT, WINDOWS, ARMS


def main():
    rows=[]
    for arm in ARMS:
        for seed in range(10):
            a=np.load(OUT/'raw'/f'{arm}_seed{seed}.npz')
            b=np.load(OUT/'raw_components'/f'{arm}_seed{seed}.npz')
            for task in list(range(1,11))+[25]+list(range(39,51))+list(range(89,101)):
                pre=f't{task}_';alive=b[pre+'alive']
                self_=b[pre+'original_self'];other=b[pre+'original_other'];res=b[pre+'original_residual'];rad=a[pre+'radial']
                total=self_+other+res+rad
                orig=a[pre+'original_exact_K']
                err=np.max(np.abs(total-orig))/max(np.max(np.abs(orig)),np.max(np.abs(self_)),1e-300)
                assert err<=1e-10
                assert np.mean(total[alive]>0)==np.mean(orig[alive]>0)
                row=dict(arm=arm,seed=seed,task=task,n_alive=int(alive.sum()),identity_error=float(err))
                for o,r,l in itertools.product([0,1],repeat=3):
                    vec=self_+o*other+r*res+l*rad
                    row[f's_other{o}_residual{r}_radial{l}']=float(np.mean(vec[alive]>0))
                keep=self_+res
                row['other_pos_to_nonpos']=float(np.mean((keep[alive]>0)&((keep+other)[alive]<=0)))
                row['other_nonpos_to_pos']=float(np.mean((keep[alive]<=0)&((keep+other)[alive]>0)))
                rows.append(row)
    def write(path,data):
        with path.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]),lineterminator='\n');w.writeheader();w.writerows(data)
    write(OUT/'recombination_states.csv',rows)
    fields=[k for k in rows[0] if k not in ['arm','seed','task']];wins=[]
    for arm in ARMS:
        for seed in range(10):
            for win,tasks in WINDOWS.items():
                rr=[r for r in rows if r['arm']==arm and r['seed']==seed and r['task'] in tasks]
                wins.append(dict(arm=arm,seed=seed,window=win,**{k:float(np.median([r[k] for r in rr])) for k in fields}))
    write(OUT/'recombination_windows.csv',wins)
    summary={}
    for arm in ARMS:
        summary[arm]={}
        for k in fields:
            summary[arm][k]={w:[min(r[k] for r in wins if r['arm']==arm and r['window']==w),
                                max(r[k] for r in wins if r['arm']==arm and r['window']==w)] for w in WINDOWS}
    (OUT/'recombination_ranges.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
