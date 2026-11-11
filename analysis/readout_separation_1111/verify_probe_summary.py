"""Synthetic known contrasts for the crossed frozen-assay aggregation."""
import json
import numpy as np
import pandas as pd
from analysis.readout_separation_1111 import summarize_probe as P

def main():
    records=[]
    for model,seeds,modes in [('mlp',range(10),['native','elu_unity']),('vit',range(10252010,10252015),['native','ln_remove_radial','ln_remove_radial_mean'])]:
        for seed in seeds:
            for label in ('actual','uniform'):
                for f in sorted(P.BANKS):
                    for b in sorted(P.BANKS):
                        for j in modes:
                            sf=.1 if f.endswith('01') else 1.
                            sb=.1 if b.endswith('01') else 1.
                            v=10*sf+sb+(3 if j!='native' else 0)+seed*.001
                            records.append(dict(model=model,seed=seed,label_channel=label,forward_bank=f,backward_bank=b,jacobian=j,g_trunk_norm=v))
    df=pd.DataFrame(records).sample(frac=1,random_state=2)
    estimates=P.summaries(df,P.KEY,['g_trunk_norm'])
    assert len(estimates)==360 and estimates.estimable.all()
    contrast=P.local_contrasts(df)
    for _,r in contrast.iterrows():
        target=9. if r.contrast.endswith('_forward_scale') else .9 if r.contrast.endswith('_backward_scale') else 9.9 if r.contrast.endswith('_physical_scale') else 3. if r.contrast.endswith('_minus_native') else 0.
        assert np.isclose(r.estimate,target,atol=1e-10),(r.contrast,r.estimate,target)
        assert np.isclose(r.ci_low,target,atol=1e-10) and np.isclose(r.ci_high,target,atol=1e-10)
    broken=df.copy();broken.loc[broken.index[0],'g_trunk_norm']=np.nan
    q=P.summaries(broken,P.KEY,['g_trunk_norm'])
    assert (~q.estimable).sum()==1
    try:P.summaries(pd.concat([df,df.iloc[:1]]),P.KEY,['g_trunk_norm'])
    except AssertionError:pass
    else:raise AssertionError('duplicate seed was treated as independent')
    print(json.dumps(dict(passed=True,synthetic_records=len(df),groups=len(estimates),contrasts=len(contrast),checks=['paired_seed_reindex','known_forward_backward_transmission_contrasts','nonfinite_not_dropped','duplicate_seed_rejected'])))

if __name__=='__main__':main()
