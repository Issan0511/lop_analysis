"""Cross-check receiver pooling, saved pairs, and compatibility with the earlier audit."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from src import boundary_transport_0908 as T
OUT=T.OUT
def main():
 checks=[];cases=[]
 for arm in ['SNA','LR']:
  for iv in ['none','l2']:
   for seed in range(3):
    prefix=f'{arm}_{iv}_s{seed}';raw=dict(np.load(OUT/'raw'/(prefix+'.npz')))
    counts=pd.read_csv(OUT/(prefix+'_paired_counts.csv'))
    maxpool=0.
    for key in T.COMP+['actual','rounding']:
     # Receiver conditional means weighted back by each fixed population.
     group=raw['receiver_'+key+'_fixed']
     n=raw['receiver_counts']
     reconstructed=np.nansum(group*n[:,None,:,:],axis=(2,3))/(512*100)
     maxpool=max(maxpool,float(abs(reconstructed-raw[key].mean(2)).max()))
    assert maxpool<1e-10,maxpool
    recomputed=((raw['pair_z']<=0)&(raw['pair_local_direction']>1e-8)&(raw['pair_actual']<-1e-8)).sum((2,3))
    assert np.array_equal(recomputed.ravel(),counts.n_neg_local_up_actual_down.to_numpy())
    actual=raw['actual'].sum(1).mean()
    prev=pd.read_csv(T.G.OUT/(prefix+'_credits.csv')) # same source trajectory identity handled via z checks
    before=np.load(T.G.OUT/'raw'/(prefix+'.npz'))['actual'][:,:20].sum(1).mean()
    error=float(abs(actual-before));assert error<1e-10
    checks.append(dict(arm=arm,iv=iv,seed=seed,receiver_pool_maxabs=maxpool,
                       previous_audit_actual_error=error,paired_counts_exact=True))
    cases.append(dict(arm=arm,iv=iv,seed=seed,
      within_cov_sum=float(raw['within_flow_cov'].sum(1).mean()),
      within_square_sum=float(raw['within_flow_square'].sum(1).mean()),
      within_delta=float((raw['within_var'][:,-1]-raw['within_var'][:,0]).mean())))
 (OUT/'report_validation.json').write_text(json.dumps(checks,indent=2))
 pd.DataFrame(cases).to_csv(OUT/'seed_variance_flow.csv',index=False,lineterminator='\n')
 print('CROSS-CHECK PASS',max(x['receiver_pool_maxabs'] for x in checks),max(x['previous_audit_actual_error'] for x in checks))
 print(pd.DataFrame(cases).groupby(['arm','iv']).median(numeric_only=True).drop(columns='seed').to_string())
if __name__=='__main__':main()
