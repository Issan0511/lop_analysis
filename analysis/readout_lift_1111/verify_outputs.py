#!/usr/bin/env python3
"""Independent endpoint/table and constraint audit, without training."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
import stage1_ctrl_1111 as E
import readout_lift_1111 as R
torch.set_num_threads(1)

def main(out):
    out=Path(out);errors=dict(ce=0.,acc=0.,h=0.,body=0.,constraint_trace=0.,readout=0.,source_sum=0.);count=0
    for folder,table in [('raw','ends.csv'),('followup_raw','followup_ends.csv')]:
        df=pd.read_csv(out/table).set_index(['task','seed','arm'])
        for t in (10,25):
            for s in (0,1,2):
                base=torch.load(R.ARCH/f'ELU_s{s}'/f'baseline_before_t{t}.pt',weights_only=False,map_location='cpu')
                X,_=E.load_X(s);y=base['y'];P0=base['P'];z0=X@P0[0].T+P0[1];B=z0<=-3;n=B.sum(0);assert (n>=30).all()
                bx=B.T.double()@X.double()/n[:,None]
                reference=np.load(out/'raw'/f's{s}_t{t}'/'V1_trace.npz')['body']
                for p in sorted((out/folder/f's{s}_t{t}').glob('*_end.pt')):
                    arm=p.name.removesuffix('_end.pt');ck=torch.load(p,weights_only=False,map_location='cpu');W,b,V,o=ck['P']
                    z=X@W.T+b;a=torch.where(z>0,z,torch.expm1(z.clamp_max(0)));ell=a@V.T+o
                    ce=float(torch.nn.functional.cross_entropy(ell,y));acc=float((ell.argmax(1)==y).float().mean())
                    h=float(z.max(0).values.median());body=float(((W.double()*bx).sum(1)+b.double()).mean());row=df.loc[(t,s,arm)]
                    for k,v in [('ce',ce-row.ce),('acc',acc-row.acc),('h',h-row.h_median),('body',body-row.body)]:errors[k]=max(errors[k],abs(v))
                    if arm.startswith('V01') or arm.startswith('C01'):expected=P0[2].mean(0)+.1*(P0[2]-P0[2].mean(0))
                    elif arm in ('V1','C1'):expected=P0[2]
                    else:expected=None
                    if expected is not None:errors['readout']=max(errors['readout'],float((V-expected).abs().max()))
                    tr=np.load(p.with_name(arm+'_trace.npz'))
                    if folder=='followup_raw' or arm.endswith('M'):errors['constraint_trace']=max(errors['constraint_trace'],float(np.abs(tr['body']-reference).max()))
                    assert np.isfinite(ell.numpy()).all();count+=1
    for filename in ('source.csv','followup_source.csv'):
        d=pd.read_csv(out/filename);keys=['task','seed','arm','step','group'];tab=d.pivot(index=keys,columns='source',values='dz')
        err=(tab.drop(columns='total').sum(axis=1)-tab.total).abs().max();errors['source_sum']=max(errors['source_sum'],float(err))
    assert count==78 and errors['ce']<2e-5 and errors['acc']<1e-7 and errors['h']<2e-5 and errors['body']<2e-5
    assert errors['constraint_trace']<2e-5 and errors['readout']==0 and errors['source_sum']<1e-8
    result=dict(endpoints=count,errors=errors,status='PASS');(out/'independent_checks.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();main(a.out)
