#!/usr/bin/env python3
"""Additional provenance check: the old own-row predictor must stay identical."""
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/chargpt_phase2carrier_1024'
OLD=Path('/home/issan/Projects/obsidian-research-data/chargpt_l0phase2_1023/results/chargpt_l0phase2_1023/raw')
records=[]
for seed in range(10):
 for task in [9,39,100]:
  path=OLD/f'headcov_C3_seed{seed}_t{task}_u10000.npz'
  new=OUT/'raw'/f'C3_seed{seed}_task{task}.npz'
  checks={}
  with np.load(path) as old,np.load(new) as cur:
   assert np.array_equal(old['alive'],cur['L0_alive'])
   for metric in ['SGD','Adam_fixed']:
    for name,ok in [('P2','iid'),('P2ctx','ctx')]:
     a,b=old[metric+'_'+ok],cur[f'L0_{metric}_{name}_up0_r']
     err=float(np.max(abs(a-b))/max(np.max(abs(a)),np.max(abs(b)),1e-300))
     checks[metric+'_'+name]=err
     assert err<1e-8,(seed,task,metric,name,err)
  records.append(dict(seed=seed,task=task,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),checks=checks))
res=dict(states=len(records),max_relative_error=max(v for r in records for v in r['checks'].values()),records=records,
         note='Old CSV mean is restricted to uniform-alive units. Current primary mean includes all 512; compare vectors before aggregation.')
(OUT/'parent_audit.json').write_text(json.dumps(res,indent=2)+'\n')
print('Parent own-row reference:',res['states'],'states; max relative error',res['max_relative_error'])
