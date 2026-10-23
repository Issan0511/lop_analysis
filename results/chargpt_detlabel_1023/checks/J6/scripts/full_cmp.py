import csv, math, json, numpy as np
from pathlib import Path
CHK=Path('/home/kubo/project/Nakatsuka/lop_detlabel/results/chargpt_detlabel_1023/checks/J6'); O=CHK/'A2_full'; R=Path('/home/kubo/project/Nakatsuka/lop_detlabel/results/chargpt_loadsign_1023')
out={}
for f in ['states.csv','components_states.csv','loading_sources_states.csv']:
    ours={(int(r['seed']),int(r['task'])):r for r in csv.DictReader((O/f).open())}
    ref={(int(r['seed']),int(r['task'])):r for r in csv.DictReader((R/f).open()) if r['arm']=='C3'}
    assert sorted(ours)==sorted(ref) and len(ours)==350
    cols=[c for c in next(iter(ref.values())) if c not in ('arm','seed','task','layer','previous_error','autograd_error')]
    sm=[];im=0;cf=0;mx=0;ns=0;ni=0;nc=0
    for k in ref:
        for c in cols:
            a,b=float(ours[k][c]),float(ref[k][c])
            if c.endswith('_error'): continue
            if c.startswith('s_'):
                ns+=1
                if not (a==b or (math.isnan(a) and math.isnan(b))): sm.append((k,c,a,b))
            elif c.startswith('n_') or c.endswith('_missing'):
                ni+=1; im+= not (a==b)
            else:
                if math.isnan(a) and math.isnan(b): continue
                nc+=1; r=abs(a-b)/max(abs(b),1e-300); mx=max(mx,r); cf+= r>1e-10
    jcols=sorted({c for _,c,_,_ in sm})
    out[f]=dict(n_sign=ns,n_sign_mismatch=len(sm),mismatch_cols=jcols,n_int=ni,n_int_mismatch=im,n_cont=nc,n_cont_fail=cf,max_rel=mx,
                mismatches=[dict(seed=k[0],task=k[1],col=c,ours=a,ref=b) for k,c,a,b in sm])
(CHK/'A2_full_vs_loadsign.json').write_text(json.dumps(out,indent=1)+'\n')
for f,v in out.items():
    print(f,{k:v[k] for k in v if k!='mismatches'})
    for m in v['mismatches'][:12]: print('  ',m)
# J6 の判定の量（後期の s_original_other・s_original_exact_K の seed 内中央値）が loadsign と同じか
import statistics
for f,col in [('components_states.csv','s_original_other'),('states.csv','s_original_exact_K')]:
    for src,P in [('ours',O/f),('ref',R/f)]:
        rows=[r for r in csv.DictReader(P.open()) if r['arm'] in ('C3','A2')]
        meds=[statistics.median(float(r[col]) for r in rows if int(r['seed'])==s and 89<=int(r['task'])<=100) for s in range(10)]
        print(col,src,[round(x,4) for x in meds])
