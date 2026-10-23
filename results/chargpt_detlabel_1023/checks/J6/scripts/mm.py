import json, numpy as np, csv
from pathlib import Path
CHK=Path('/home/kubo/project/Nakatsuka/lop_detlabel/results/chargpt_detlabel_1023/checks/J6'); O=CHK/'A2_full'
mm=json.load(open(CHK/'A2_full_vs_loadsign.json'))['components_states.csv']['mismatches']
raw={}; ok=0; rows=[]; maxunits=0
ntiny_all=[]
for s in range(10):
    z=np.load(O/f'raw_components/A2_seed{s}.npz')
    for t in [1,2,3,4,5,6,7,8,9,10,25]+list(range(39,51))+list(range(89,101)):
        al=z[f't{t}_alive']
        for e in ['original','loaderror']:
            v=z[f't{t}_{e}_other']; sc=np.abs(v[al]).max()
            raw[s,t,e]=(int(al.sum()), int((al&(np.abs(v)<=1e-12*sc)&(v!=0)).sum()))
            ntiny_all.append(raw[s,t,e][1])
for m in mm:
    e='original' if 'original' in m['col'] else 'loaderror'
    n,nt=raw[m['seed'],m['task'],e]
    du=round(abs(m['ours']-m['ref'])*n)
    maxunits=max(maxunits,du); ok+= du<=nt
    rows.append((m['seed'],m['task'],e,n,du,nt))
print('mismatches',len(mm),'original',sum(r[2]=='original' for r in rows),'loaderror',sum(r[2]=='loaderror' for r in rows))
print('max units differing',maxunits,'mismatches with du<=n_tiny_nonzero',ok)
print('states with any rounding-level nonzero other (orig/load):',sum(1 for k,v in raw.items() if v[1]>0 and k[2]=='original'),sum(1 for k,v in raw.items() if v[1]>0 and k[2]=='loaderror'),'of 350; max per state',max(ntiny_all))
print('late (89-100) original mismatches:',[(r[0],r[1],r[4],r[3]) for r in rows if r[2]=='original' and r[1]>=89])
