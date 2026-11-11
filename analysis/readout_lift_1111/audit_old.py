#!/usr/bin/env python3
"""Post-hoc: simultaneous negative-body motion and top growth in old long runs."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from summarize import md

def main(out):
    out=Path(out);root=Path('/home/issan/Projects/obsidian-research-data/push_lift_ladder_1layer_0928/vfreeze/runs_vf')
    rows=[];inputs={}
    for arm in ('EEQ','VF0.1','VF1','VF10'):
        for seed in (0,1):
            p=root/f'{arm}_s{seed}.npz';inputs[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();d=np.load(p)
            for t in range(150,200):
                eligible=d['sw_k'][t]>0
                for a,b,window in (('sw','mid','0-500'),('mid','end','500-4000'),('sw','end','0-4000')):
                    db=d[f'{b}_body'][t]-d[f'{a}_body'][t];dh=d[f'{b}_top'][t]-d[f'{a}_top'][t]
                    rows.append(dict(tier='posthoc',arm=arm,seed=seed,task=t+1,window=window,
                        selected=int(eligible.sum()),body_delta_median=float(np.median(db[eligible])),top_delta_median=float(np.median(dh[eligible])),
                        simultaneous_fraction=float(np.mean(((db<-.1)&(dh>.1))[eligible])),
                        top_level_median=float(np.median(d[f'{b}_top'][t][eligible])),
                        body_level_median=float(np.median(d[f'{b}_body'][t][eligible])),ce_end=float(d['ce_end'][t])))
    df=pd.DataFrame(rows);df.to_csv(out/'old_vfreeze_rows.csv',index=False)
    means=df.groupby(['arm','seed','window']).mean(numeric_only=True).reset_index();means.to_csv(out/'old_vfreeze_seed_means.csv',index=False)
    summary=means.groupby(['arm','window']).mean(numeric_only=True)
    (out/'old_vfreeze_posthoc.md').write_text('# Post-hoc long-run audit\n\nTasks151-200 only; units with opening at task start. Unit medians/fractions, then task mean, then two-seed mean. Original body is median preactivation over all1200 inputs (all_probe4.py), not the new fixed-deep mean.\n\n'+md(summary.round(6).reset_index())+'\n')
    (out/'old_vfreeze_provenance.json').write_text(json.dumps(inputs,indent=2))
    print(summary[['body_delta_median','top_delta_median','simultaneous_fraction']].round(6).to_string())

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--out',required=True);v=a.parse_args();main(v.out)
