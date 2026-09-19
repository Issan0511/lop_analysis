from pathlib import Path
import argparse, csv, hashlib, json, statistics
import numpy as np

parser=argparse.ArgumentParser(description='Independent post-run consistency audit; no registered criteria are changed.')
parser.add_argument('result_root',type=Path)
parser.add_argument('--raw-root',type=Path,help='Raw result mirror in the data archive after cleanup')
args=parser.parse_args()
root=args.result_root
raw=args.raw_root or root
jobs = [f'{a}_base' for a in ('R','GELU','ELU','LR','LK001','SNA')] + ['GELU_early','GELU_late','ELU_early']
seeds = [1001,1002,1003,1004,1005]
plan = json.loads((raw/'_launch/plan.json').read_text())
rows, prov, endpoint, summaries = {}, {}, {}, {}
for job in jobs:
    p = json.loads((root/job/'provenance.json').read_text())
    assert p['completed_steps'] == 60000 and p['max_steps'] is None
    assert p['seeds'] == seeds and p['R'] == 5 and p['observed'] and p['graph']
    assert p['git_hash'] == plan['git_hash'] and p['dirty_src_analysis'] == []
    prov[job] = p
    with (root/job/'diagnostics.csv').open() as f: rr = list(csv.DictReader(f))
    assert len(rr) == 160
    rows[job] = rr
    with (root/job/'per_task.csv').open() as f: er = list(csv.DictReader(f))
    assert len(er) == 10
    endpoint[job] = {(int(r['seed']),int(r['task'])):r for r in er}
    net_error = []
    component_error = []
    for r in rr:
        e,u,n,c,l,h,k,res = [float(r[x]) for x in ('down','up','net','conf','label','hist','correction','roundoff')]
        assert all(np.isfinite(x) for x in (e,u,n,c,l,h,k,res)) and e>=0 and u>=0
        net_error.append(abs(n-(u-e))/(1+e+u))
        component_error.append(abs(n-(c+l+h+k+res))/(1+abs(c)+abs(l)+abs(h)+abs(k)))
        if r['mode']=='base' or int(r['task'])==2:
            assert k == 0
    z = np.load(raw/job/'unit_metrics.npz')
    assert list(z['seeds']) == seeds
    end = np.flatnonzero((z['task']==1)&(z['step']==30000)).item()
    start = np.flatnonzero((z['task']==2)&(z['step']==0)).item()
    for name in z.files:
        if name not in ('task','step','seeds'):
            assert np.array_equal(z[name][end],z[name][start]), (job,name,'task-switch mutation')
    summaries[job] = {'max_normalized_flux_sum_error':max(net_error),
                      'max_normalized_component_sum_error':max(component_error),
                      'rows':len(rr),'tasks':len(er),'task_switch_features_bit_equal':True}

# Compare paired experimental inputs without requiring activation states to match.
ref=prov['R_base']
shared={}
for key in ref:
    if any(x in key for x in ('initial','subset','stream','data_sha')):
        shared[key] = all(prov[j].get(key)==ref[key] for j in jobs)
assert all(shared.values()), shared

v=json.loads((root/'verdict.json').read_text())
comparisons=[('C1','GELU_early','GELU_base',1,'acc'),('C2','GELU_early','GELU_late',1,'acc'),
             ('C3','GELU_early','GELU_base',2,'online_acc'),('C4','ELU_early','ELU_base',1,'acc')]
independent={}
for key,a,b,task,metric in comparisons:
    diffs=[float(endpoint[a][s,task][metric])-float(endpoint[b][s,task][metric]) for s in seeds]
    assert abs(statistics.median(diffs)-v['labels'][key]['median_difference']) < 1e-7
    assert sum(d>0 for d in diffs)==v['labels'][key]['positive_seeds']
    independent[key]={'differences':diffs,'median':statistics.median(diffs)}
result={'pass':True,'git_hash':plan['git_hash'],'paired_inputs_equal':shared,'jobs':summaries,
        'independent_C1_C4':independent,'note':'Independent audit after all runs; changes no registered criterion.'}
(root/'qc.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
