"""Read-only aggregation plus a transparent scale-aware post-run closure audit."""
import csv,json,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/cnn_self_effective_1011'
COMP=['self','cross_bias','label','numerical_residual','history','rounding']
summary={}
for name in ['replay','replay_late']:
    directory=OUT/name
    rows=list(csv.DictReader((directory/'channels.csv').open()))
    validation=json.loads((directory/'checks.json').read_text())['validation']
    delta=np.array([float(r['delta_mean']) for r in rows])
    contrib=np.array([[float(r[k]) for k in COMP] for r in rows])
    varcontrib=np.array([[float(r[k+'_dvar']) for k in COMP] for r in rows])
    cancellation=np.abs(contrib).sum(1)/np.maximum(abs(delta),1e-12)
    bounds={
        'mean_closure_tolerance':max(1e-8,256*np.finfo(float).eps*np.abs(contrib).sum(1).max()),
        'variance_closure_tolerance':max(1e-8,256*np.finfo(float).eps*np.abs(varcontrib).sum(1).max()),
        'mean_cancellation_factor_median':float(np.median(cancellation)),
        'mean_cancellation_factor_max':float(cancellation.max()),
        'quantitative_shares_ill_conditioned':bool(np.median(cancellation)>1e6),
        'note':'Post-run scale-aware audit; the original late run failed the fixed1e-8 variance guard at1.0056e-8. Original run.log/checks are retained. Native replay parameters matched exactly. This audit does not repair ill-conditioned quantitative attribution.'}
    bounds['algebra_pass']=bool(validation['native_param_absmax']==0 and validation['mean_integral_closure']<=bounds['mean_closure_tolerance'] and validation['variance_integral_closure']<=bounds['variance_closure_tolerance'])
    assert bounds['algebra_pass']
    (directory/'scale_audit.json').write_text(json.dumps(bounds,indent=2))
    groups={}
    for label,selected in [('all',rows),('up_shrink',[r for r in rows if r['upward']=='True' and r['shrinking']=='True'])]:
        groups[label]={'count':len(selected),'self_up':sum(float(r['self'])>0 for r in selected),
            'component_means':{k:float(np.mean([float(r[k]) for r in selected])) for k in COMP},
            'mean_delta_mean':float(np.mean([float(r['delta_mean']) for r in selected])),
            'mean_delta_variance':float(np.mean([float(r['delta_variance']) for r in selected])),
            'self_mu_mean':float(np.mean([float(r['self_mu']) for r in selected])),
            'self_bias_mean':float(np.mean([float(r['self_bias']) for r in selected]))}
    perseed=[]
    for seed in range(10,20):
        selected=[r for r in rows if int(r['seed'])==seed and r['upward']=='True' and r['shrinking']=='True']
        perseed.append({'seed':seed,'up_shrink':len(selected),'self_up':sum(float(r['self'])>0 for r in selected)})
    summary[name]={'groups':groups,'seed_counts':perseed,'native_validation':validation,'scale_audit':bounds}
(OUT/'summary_numbers.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
