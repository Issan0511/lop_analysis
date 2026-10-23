"""Independent artifact/hash/coverage and closed-form df9 interval verification."""
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

W = Path('/home/issan/Projects/claude/wt/resp_step_controls_1018')
P = W / 'results/resp_step_controls_1018/RM'
OUT = Path('/home/issan/Projects/claude/wt/f51_integration_audit_1018/results/f51_integration_audit_1018/rm_independent_result_check.json')

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def t9_cdf(x):
    # Substitute theta=atan(t/sqrt(9)); integrate cos(theta)^8 by recurrence.
    theta = math.atan(x / 3)
    integral = theta
    for n in (2, 4, 6, 8):
        integral = math.sin(theta) * math.cos(theta)**(n - 1) / n + (n - 1) * integral / n
    return .5 + math.gamma(5) / (math.sqrt(math.pi) * math.gamma(4.5)) * integral

def critical(p):
    a,b = 0.,100.
    for _ in range(100):
        c = (a+b)/2
        if t9_cdf(c) < p:a=c
        else:b=c
    return (a+b)/2

checked = {}
identities = {}
rows_by_phase = {}
for phase,seeds,arms in [('calibration',range(100,110),('N','R')),('evaluation',range(10),('N','R','LR_G','LR_U'))]:
    folder=P/phase
    proof=json.loads((folder/'completion.json').read_text())
    assert proof['status']=='COMPLETE'
    for rel,digest in proof['output_sha256'].items():
        assert Path(rel).name==rel and sha(folder/rel)==digest,(phase,rel)
    identity=json.loads((folder/'provenance.json').read_text())
    assert identity==proof['identity']
    for rel,digest in identity['source_sha256'].items():
        assert sha(W/rel)==digest,rel
    rows=json.loads((folder/'per_task.json').read_text())
    keys=[(int(r['seed']),r['arm'],int(r['task'])) for r in rows]
    expected={(s,a,t) for s in seeds for a in arms for t in (11,12)}
    assert len(keys)==len(expected) and set(keys)==expected
    assert all(r['finite'] and math.isfinite(r['online_acc']) for r in rows)
    keyed=dict(zip(keys,rows))
    for s in seeds:
        for t in (11,12):
            for attr in ('label_hash','order_hash','initial_state_sha256'):
                vals=[keyed[s,a,t][attr] for a in arms]
                assert None not in vals and len(set(vals))==1,(phase,s,t,attr)
    checked[phase]={'rows':len(rows),'output_hashes':len(proof['output_sha256']),'seeds':list(seeds),'arms':list(arms)}
    identities[phase]=identity
    rows_by_phase[phase]=keyed
assert identities['calibration']['dataset_signature']==identities['evaluation']['dataset_signature']
schedule=json.loads((P/'calibration/schedule.json').read_text())
assert sha(P/'calibration/schedule.json')==identities['evaluation']['schedule_sha256']
for metric in ('qG','qU'):
    for layer in (1,2,3):
        records=[r for r in schedule['per_seed'] if r['metric']==metric and r['layer']==layer]
        assert sorted(r['seed'] for r in records)==list(range(100,110))
        ratios=[r['R']/r['N'] for r in records]
        assert all(a==r['ratio'] for a,r in zip(ratios,records))
        assert statistics.median(ratios)==schedule[metric][layer-1]

keyed=rows_by_phase['evaluation']
q=critical(1-.05/(2*15))
q95=critical(.975)
observed=json.loads((P/'verdict.json').read_text())
effects={}
for a,b in [('R','N'),('LR_G','N'),('LR_U','N'),('LR_G','R'),('LR_U','R')]:
    d=[keyed[s,a,11]['online_acc']-keyed[s,b,11]['online_acc'] for s in range(10)]
    mean=statistics.mean(d);se=statistics.stdev(d)/math.sqrt(10)
    vals={'mean':mean,'lo':mean-q*se,'hi':mean+q*se,'lo95':mean-q95*se,'hi95':mean+q95*se}
    existing=observed['boxes']['RM']['effects'][a+'-'+b]
    for k,v in vals.items():assert abs(v-existing[k])<1e-11,(a,b,k,v,existing[k])
    effects[a+'-'+b]=vals
controls={}
for arm in ('LR_G','LR_U'):
    cn,cr=effects[arm+'-N'],effects[arm+'-R']
    controls[arm]='REPRODUCES' if cn['lo']>0 and cr['lo']>-.005 else 'SHORTFALL' if cr['hi']<-.005 else 'UNRESOLVED'
label='NOT_REPRODUCED' if effects['R-N']['lo']<=0 else 'LR_EXPLAINS' if 'REPRODUCES' in controls.values() else 'NOT_LR' if set(controls.values())=={'SHORTFALL'} else 'UNRESOLVED'
assert label==observed['boxes']['RM'].get('label',observed['boxes']['RM']['status'])
assert observed['status']=='INCOMPLETE' and observed['label'] is None
means={arm:statistics.mean(keyed[s,arm,11]['online_acc'] for s in range(10)) for arm in ('N','R','LR_G','LR_U')}
result={'all_pass':True,'reviewer':'Codex /root','method':'independent hashlib/coverage joins; closed-form Student t df9 CDF via cos^8 recurrence, n=10, m=15','source_commit':identities['evaluation']['git_hash'],'critical_value':q,'checked':checked,'locked_qG':schedule['qG'],'locked_qU':schedule['qU'],'t11_arm_means':means,'effects':effects,'controls':controls,'RM_label':label,'overall_status':'INCOMPLETE','overall_label':None}
OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
