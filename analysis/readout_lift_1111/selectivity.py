#!/usr/bin/env python3
"""Post-hoc endpoint audit: which inputs became the upper tail?"""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
import stage1_ctrl_1111 as E
import demand_allocation_1111 as D
import readout_lift_1111 as R
from summarize import md
torch.set_num_threads(1)

def measure(P,X,y,initial_top,meta):
    F=D.fields(P,X,y,'ELU');z=F['z'];a=F['a'];V=P[2].detach();vc=V-V.mean(0)
    norm=vc.norm(dim=0).clamp_min(1e-30);score=vc[y]/norm
    top=torch.zeros_like(z,dtype=torch.bool);top.scatter_(0,torch.argsort(z,dim=0,stable=True)[-60:],True)
    targetmax=y[:,None]==V.argmax(0)[None,:]
    return dict(**meta,normalized_target_score=float(score[top].mean()),top_matches_preferred_class=float(targetmax[top].float().mean()),
                initial_top_overlap=float((top&initial_top).sum()/top.sum()),
                label_term_current_top=float(((.1-torch.nn.functional.one_hot(y,10))@V)[top].mean()),
                weighted_alignment=float(((a-a.mean(0))*score).mean()),acc=F['acc'],ce=F['ce'])

def main(raw,follow,out):
    rows=[];inputs={}
    for t in (10,25):
        for s in (0,1,2):
            source=R.ARCH/f'ELU_s{s}'/f'baseline_before_t{t}.pt';base=torch.load(source,weights_only=False,map_location='cpu')
            X,_=E.load_X(s);y=base['y'];z=D.fields(base['P'],X,y,'ELU')['z'];top=torch.zeros_like(z,dtype=torch.bool)
            top.scatter_(0,torch.argsort(z,dim=0,stable=True)[-60:],True)
            rows.append(measure(base['P'],X,y,top,dict(tier='posthoc',task=t,seed=s,arm='initial')))
            for root in (Path(raw),Path(follow)):
                for p in sorted((root/f's{s}_t{t}').glob('*_end.pt')):
                    arm=p.name.removesuffix('_end.pt');ck=torch.load(p,weights_only=False,map_location='cpu');inputs[str(p)]=R.sha(p)
                    rows.append(measure(ck['P'],X,y,top,dict(tier='posthoc',task=t,seed=s,arm=arm)))
    out=Path(out);d=pd.DataFrame(rows);d.to_csv(out/'selectivity_posthoc.csv',index=False)
    mean=d.groupby(['task','arm']).mean(numeric_only=True);mean.to_csv(out/'selectivity_posthoc_means.csv')
    (out/'selectivity_posthoc.md').write_text('# Post-hoc selection of upper-tail inputs\n\nInitial versus endpoint top5% per unit. Score is the correct-class readout coefficient minus its class mean, divided by the column norm. These endpoint audits were added after primary outcomes; no new training or independent significance claim.\n\n'+md(mean.round(6).reset_index())+'\n')
    (out/'selectivity_provenance.json').write_text(json.dumps(inputs,indent=2));print(mean.round(6).to_string())

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--raw',required=True);a.add_argument('--follow',required=True);a.add_argument('--out',required=True);v=a.parse_args();main(v.raw,v.follow,v.out)
