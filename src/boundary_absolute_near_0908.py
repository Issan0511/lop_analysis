
from pathlib import Path
import csv
import numpy as np
import torch
from src.boundary_groups_0908 import H,OUT,partition,credits,csvwrite
torch.set_num_threads(1)
def main():
 data=H.Mnist(torch.device('cpu'))
 for f in sorted(OUT.glob('*_states.pt')):
  prefix=f.name.removesuffix('_states.pt');out=OUT/(prefix+'_absolute_near.csv')
  if out.exists():continue
  arm,iv,stag=prefix.split('_');seed=int(stag[1:])
  raw=torch.load(f,weights_only=False);labels=data.test_y[raw['probe_indices']];rows=[]
  for b in raw['boundaries']:
   if b['task']>119:continue
   base=partition(b['end'],b['before']['z']);D=base[0]|base[2];N=abs(b['before']['z'])<=.5
   masks=np.stack([D&~N,N&~D,D&N,~(D|N)])
   cc,ce,_,_,_=credits(b['before'],b['after_625'],masks,labels,arm)
   nd=int(D.sum());nn=int(masks[1].sum())
   rows.append({'arm':arm,'iv':iv,'seed':seed,'task':b['task'],'N_abs_count':int(N.sum()),'N_only_count':nn,
     'D_count':nd,'credit_N_only':float(cc[1]),'credit_D_total':float(cc[0]+cc[2]),
     'N_credit_per_unit':float(cc[1]/nn) if nn else None,'D_credit_per_unit':float((cc[0]+cc[2])/nd) if nd else None})
  csvwrite(out,rows)
 allrows=[]
 for f in sorted(OUT.glob('*_absolute_near.csv')):
  rr=list(csv.DictReader(f.open()));r=rr[0];s={k:r[k] for k in ['arm','iv','seed']}
  for k in ['N_abs_count','N_only_count','D_count','credit_N_only','credit_D_total','N_credit_per_unit','D_credit_per_unit']:
   vals=[float(r[k]) for r in rr if r[k]!=''];s[k]=float(np.mean(vals)) if vals else None
  s['N_only_exceeds_D_boundaries']=sum(float(r['credit_N_only'])>float(r['credit_D_total']) for r in rr)
  allrows.append(s)
 csvwrite(OUT/'absolute_near_seed_verdict.csv',allrows)
 print('completed sensitivity',len(allrows))
 for r in allrows:print(r)
if __name__=='__main__':main()
