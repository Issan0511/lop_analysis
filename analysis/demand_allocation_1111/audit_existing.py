#!/usr/bin/env python3
"""Post-hoc reproduction of existing summaries, with source hashes and windows."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,rows):
    with open(p,'w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);inputs={}
    vf=Path('/home/issan/Projects/obsidian-research-data/push_lift_ladder_1layer_0928/vfreeze/runs_vf')
    rows=[]
    for arm in ['EEQ','VF0.1','VF1','VF10']:
        for seed in [0,1]:
            p=vf/f'{arm}_s{seed}.npz';d=np.load(p);inputs[str(p)]=sha(p)
            seq=[]
            for t in range(150,200):
                m=d['sw_k'][t]>0;s=d['end_s'][t]
                seq.append([np.median((d['end_top'][t]/s)[m]),
                    np.median(s[m]),np.median((d['mid_body'][t]-d['sw_body'][t])[m]),
                    np.median((d['mid_top'][t]-d['sw_top'][t])[m]),
                    np.median((d['end_body'][t]-d['mid_body'][t])[m]),
                    np.median((d['end_top'][t]-d['mid_top'][t])[m]),
                    np.median((1-d['mid_k'][t]/np.maximum(d['sw_k'][t],1))[m]),d['ce_end'][t]])
            vals=np.mean(seq,axis=0)
            row=dict(arm=arm,seed=seed,window='task151-200',unit_selection='sw_k>0',reduction='unit median then task mean')
            row.update(zip(['top_over_sd','sd','body_push_0_500','top_push_0_500','body_lift_500_4000','top_lift_500_4000','lost_open_count_fraction','ce_end'],vals))
            rows.append(row)
    write(out/'existing_vfreeze.csv',rows)
    stage=Path('/home/issan/Projects/obsidian-research-data/stage1_ctrl_1111/results/stage1_ctrl_1111/raw')
    rows=[]
    for act in ['R','ELU','SILU','GELU','LR']:
        for seed in range(10):
            p=stage/f'branch_adam_lr0.001_{act}_s{seed}.npz';base=np.load(p);inputs[str(p)]=sha(p)
            for arm in ['S','Qall']:
                if arm=='S':d=base
                else:
                    p=stage/f'persist_Q_adam_lr0.001_{act}_s{seed}.npz';d=np.load(p);inputs[str(p)]=sha(p)
                u=d[f'{arm}|t30|units'][-1]
                rows.append(dict(act=act,seed=seed,arm=arm,window='body task30 end; deficit task21-30; V task30 start',
                    closed_fraction=np.mean(u[2]==0),mean_abs_dphi=np.mean(u[3]),sd_median=np.median(u[1]),
                    r_median=np.median(u[0]/np.maximum(u[1],1e-12)),Vnorm_task30_start=float(d[f'{arm}|t30|vnorm'][0]),
                    acc_task30_end=float(d[f'{arm}|t30|eval'][-1,0]),
                    deficit=np.mean([base[f'fa|t{t:02d}|eval'][-1,0]-d[f'{arm}|t{t:02d}|eval'][-1,0] for t in range(21,31)])))
    write(out/'existing_stage1.csv',rows)
    (out/'existing_provenance.json').write_text(json.dumps(dict(tier='posthoc reproduction of previously reported summaries',inputs=inputs),indent=2))
    for name,cols in [('existing_vfreeze.csv',['top_over_sd','body_push_0_500']),('existing_stage1.csv',['closed_fraction','deficit'])]:
        import pandas as pd
        d=pd.read_csv(out/name);keys=['arm'] if 'vfreeze' in name else ['act','arm']
        print(name);print(d.groupby(keys)[cols].mean().round(6).to_string())

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--out',required=True);main(a.parse_args().out)
