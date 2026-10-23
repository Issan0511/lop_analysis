import sys, json, numpy as np
sys.path.insert(0,'/home/kubo/project/Nakatsuka/lop_detlabel/analysis/chargpt_detlabel_1023/j6')
import check_sj6_unit as U
import csv
taus=[0,1e-16,1e-15,1e-14,1e-13,1e-12,1e-11,1e-10,1e-9,1e-8]
ref={}
for f in U.FILES:
    rows=list(csv.DictReader((U.REF/f).open())); ref[f]=({(int(r['seed']),int(r['task'])):r for r in rows if r['arm']=='C3'},U.columns(rows[0]))
need=[]; bycol={}; band={}
for s in range(10):
    z=U.load_side(U.OURS,'A2',s)
    for t in U.TASKS:
        for f,(rc,cols) in ref.items():
            for col in cols:
                v,mask=U.value(f,col,z,t); a=np.abs(v); mx=a[mask].max() if mask.any() else 0
                r=np.where(mask, a/mx if mx else 0, np.inf)
                nz=mask&(a>0)&(r<=1e-10)
                if nz.any():
                    b=bycol.setdefault(col,[0,0,0]); b[0]+=int((nz&(r<=1e-14)).sum()); b[1]+=int((nz&(r>1e-14)&(r<=1e-12)).sum()); b[2]+=int((nz&(r>1e-12)).sum())
                n=int(mask.sum()); rpos=round(float(rc[s,t][col])*n); pos=int((mask&(v>0)).sum())
                if rpos==pos: continue
                for tau in taus:
                    near=mask&(a<=tau*mx); pf=int((mask&~near&(v>0)).sum())
                    if 0<=rpos-pf<=int(near.sum()): need.append(dict(seed=s,task=t,col=col,tau=tau,diff=rpos-pos)); break
                else: need.append(dict(seed=s,task=t,col=col,tau=None,diff=rpos-pos))
from collections import Counter
print('mismatch state-cols',len(need),'minimal tau needed:',sorted(Counter(str(x['tau']) for x in need).items()))
print('cols with nonzero |v|/max<=1e-10 [<=1e-14, (1e-14,1e-12], (1e-12,1e-10]]:',bycol)
json.dump(dict(need=need,bycol=bycol),open(sys.argv[1],'w'),indent=1)
