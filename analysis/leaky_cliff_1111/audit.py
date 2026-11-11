#!/usr/bin/env python3
"""Registered contrasts, censored times, integrity audit and descriptive geometry."""
import hashlib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import analyze as A
ROOT=A.ROOT; OUT=A.OUT
torch.set_num_threads(1)

def raw(p):
    p=Path(p)
    if p.exists():return p
    parts=p.parts; idx=parts.index('results');rel=str(Path(*parts[idx:]))
    manifest=json.loads((OUT/'backup_manifest.json').read_text())
    return Path(next(f['backup'] for f in manifest['files'] if f['source_rel']==rel))

def audit():
    complete=json.loads((OUT/'COMPLETE.json').read_text())
    loc=json.loads((OUT/'localization_selection.json').read_text())
    expected={'discovery':21,'confirmation':15,'branches':70 if loc['run'] else 60}
    provenance=[]; rngs={}; checks=[]
    for stage,n in expected.items():
        paths=sorted((OUT/stage).glob('*/s*/provenance.json'));assert len(paths)==n,(stage,len(paths),n)
        for p in paths:
            dr=p.parent;meta=json.loads(p.read_text());args=meta['args']
            assert (dr/'done.json').exists()
            assert meta['code_sha256']==A.sha(ROOT/'src/leaky_cliff_1111.py')
            assert meta['spec_sha256']==A.sha(ROOT/'specs/spec_leaky_cliff_1111.md')
            t=pd.read_csv(dr/'tasks.csv');c=pd.read_csv(dr/'curves.csv');d=pd.read_csv(dr/'diagnostics.csv')
            assert np.isfinite(c[['acc','ce']].to_numpy()).all()
            assert len(t)==(20 if stage!='branches' else 1)
            assert (t.steps==args['steps']).all()
            for l in (1,2):
                if stage!='branches':
                    assert np.max(np.abs(d[f'L{l}_true_gmean']-(args['a']+(1-args['a'])*d[f'L{l}_pplus_unit_mean'])))<1e-12
            if stage!='branches':
                ck=torch.load(raw(dr/'state_t20.pt'),weights_only=False,map_location='cpu')
                val=(ck['g_lab'].numpy().tobytes(),ck['g_batch'].numpy().tobytes())
                k=args['seed'];assert k not in rngs or rngs[k]==val
                rngs[k]=val
            else:
                b=json.loads((dr/'branch_check.json').read_text());assert b['source_sha256']==A.sha(raw(args['source']))
                if args['arm'] in ['B','BO','B1','B2','X','Breverse','Xreverse']:
                    assert b['initial_natural_logit_equal'] and b['initial_maxabs']==0
                checks.append(dict(arm=args['arm'],seed=args['seed'],**b))
            provenance.append(dict(path=str(p.relative_to(ROOT)),**{k:meta[k] for k in ['git_hash','spec_sha256','code_sha256','data_sha256','torch']}))
    assert len({p['git_hash'] for p in provenance})==1
    assert len({p['data_sha256'] for p in provenance})==1
    result=dict(all_pass=True,completed=expected,total=len(provenance),paired_rng_at_task20=True,
                code_and_spec_sha_match=True,natural_mean_identity=True,all_recorded_acc_ce_finite=True,
                initial_B_X_forward_bitwise_equal=True,training_git_hash=provenance[0]['git_hash'],
                code_sha256=provenance[0]['code_sha256'],spec_sha256=provenance[0]['spec_sha256'],
                data_sha256=provenance[0]['data_sha256'],seconds=complete['seconds'])
    (OUT/'integrity.json').write_text(json.dumps(result,indent=2)+'\n')
    pd.DataFrame(checks).to_csv(OUT/'branch_initial_checks.csv',index=False)
    return result

def contrasts():
    s=pd.read_csv(OUT/'natural_seed_summary.csv');sel=json.loads((OUT/'selection.json').read_text())
    rows=[]
    for metric in ['late_online','late_endpoint','late_ce','online_drop']:
        p=s[s.stage=='confirmation'].pivot(index='seed',columns='a',values=metric)
        for hi,lo in [(sel['good'],sel['bad']),(sel['midpoint'],sel['bad']),(sel['good'],sel['midpoint'])]:
            # CSV float parsing may differ from full precision by a final bit.
            hi=min(p.columns,key=lambda v:abs(v-hi));lo=min(p.columns,key=lambda v:abs(v-lo))
            rows.append(dict(metric=metric,a_high=hi,a_low=lo,**A.interval(p[hi]-p[lo])))
    pd.DataFrame(rows).to_csv(OUT/'confirmation_contrasts.csv',index=False)
    b=pd.read_csv(OUT/'branch_seed_summary.csv');budget=[];censor=[]
    for s in range(3,8):
        g=b[b.seed==s].set_index('arm');row=dict(seed=s,N_acc6000=g.loc['N','acc_6000'],N_acc24000=g.loc['N','acc_24000'],N_T90=g.loc['N','T90'])
        for ref in ['Fbad','Fgood']:
            g6=g.loc[ref,'acc_6000']-g.loc['N','acc_6000'];g24=g.loc[ref,'acc_24000']-g.loc['N','acc_24000']
            row.update({ref+'_gap6000':g6,ref+'_gap24000':g24,ref+'_gap_contraction':1-g24/g6 if g6>0 else np.nan})
        budget.append(row)
    pd.DataFrame(budget).to_csv(OUT/'budget_seed_summary.csv',index=False)
    for arm,g in b.groupby('arm'):
        for metric in ['T80','T90','Tce05']:
            v=g[metric].dropna();n=len(g);ordered=np.sort(v)
            censor.append(dict(arm=arm,metric=metric,n=n,reached=len(v),right_censored=n-len(v),budget=int(g.steps.iloc[0]),
                               first=float(v.min()) if len(v) else np.nan,last_observed=float(v.max()) if len(v) else np.nan,
                               KM_median=float(ordered[n//2]) if len(v)>n//2 else np.nan))
    pd.DataFrame(censor).to_csv(OUT/'censored_times.csv',index=False)

def geometry():
    data=[];end=[]
    for stage in ['discovery','confirmation']:
        for p in sorted((OUT/stage).glob('*/s*/provenance.json')):
            dr=p.parent;args=json.loads(p.read_text())['args'];a=args['a'];s=args['seed']
            t=pd.read_csv(dr/'tasks.csv');d=pd.read_csv(dr/'diagnostics.csv');d=d[d.step==6000].copy()
            e=t.merge(d,on='task',suffixes=('','_diag'));e['a']=a;e['seed']=s;e['stage']=stage;end.append(e)
            for task in [2,5,10,15,20]:
                arrays=np.load(raw(dr/f'state_t{task}.npz'))
                for l in [1,2]:
                    z=arrays[f'z{l}'].astype(float);pplus=(z>0).mean(0);m=z.mean(0);sd=z.std(0);r=np.divide(m,sd,out=np.full_like(m,np.nan),where=sd>0)
                    mean=a+(1-a)*pplus;g2=a*a+(1-a*a)*pplus
                    ne=np.divide(mean*mean,g2,out=np.zeros_like(mean),where=g2>0)
                    data.extend(dict(stage=stage,a=a,seed=s,task=task,layer=l,unit=i,pplus=pplus[i],m=m[i],s=sd[i],r=r[i],gmean=mean[i],g2=g2[i],neff=ne[i]) for i in range(100))
    unit=pd.DataFrame(data);unit.to_csv(OUT/'unit_geometry.csv.gz',index=False)
    end=pd.concat(end,ignore_index=True);end.to_csv(OUT/'task_end_geometry.csv.gz',index=False)
    matches=[]
    for (stage,seed,task),g in end.groupby(['stage','seed','task']):
        rs=list(g.to_dict('records'))
        for i,u in enumerate(rs):
            for v in rs[i+1:]:
                if max(abs(u[f'L{l}_true_gmean']-v[f'L{l}_true_gmean']) for l in [1,2])<=.02:
                    matches.append(dict(stage=stage,seed=seed,task=task,a=u['a'],b=v['a'],g1a=u['L1_true_gmean'],g1b=v['L1_true_gmean'],g2a=u['L2_true_gmean'],g2b=v['L2_true_gmean'],acca=u['endpoint_acc'],accb=v['endpoint_acc'],abs_acc_gap=abs(u['endpoint_acc']-v['endpoint_acc'])))
    pd.DataFrame(matches).to_csv(OUT/'matched_means_descriptive.csv',index=False)
    # Per-unit distributions; each seed contributes 100 units. No claim of 300 independent replications.
    fig,ax=A.plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for a,col in zip([.003,.03,.1,.3],['C0','C1','C2','C3']):
        g=unit[(unit.stage=='discovery')&(unit.a==a)&(unit.task==20)]
        for l in [1,2]:
            q=g[g.layer==l]
            ax[l-1,0].hist(q.pplus,bins=np.linspace(0,1,26),weights=np.ones(len(q))/len(q),histtype='step',label=f'a={a:g}',color=col)
            r=np.sort(q.r.dropna());ax[l-1,1].plot(r,np.arange(1,len(r)+1)/len(r),color=col)
            ax[l-1,2].scatter(q.gmean,q.neff,s=7,alpha=.3,color=col)
    for l in [1,2]:
        ax[l-1,0].set(xlabel='Per-unit positive-branch input fraction',ylabel=f'Layer {l}: fraction of units',yscale='log')
        ax[l-1,1].set(xlabel='Per-unit r = m / s',ylabel='Empirical CDF',xlim=(-10,3))
        ax[l-1,2].set(xlabel='Per-unit mean derivative',ylabel='Derivative participation',ylim=(-.02,1.02))
    ax[0,0].legend(fontsize=9);fig.suptitle('Task 20 geometry: three paired discovery seeds, 100 units per layer per seed')
    A.savefig(fig,'unit_distribution')
    return len(matches),max((r['abs_acc_gap'] for r in matches),default=None)

def epsilon_control():
    sys.path.insert(0,str(ROOT))
    from src import leaky_cliff_1111 as C
    d=pd.read_csv(OUT/'branch_diagnostic_points.csv');rows=[]
    for seed in range(3,8):
        x=C.get_x(seed);zero=int((x==0).all(0).sum());fixed=zero/785
        g=d[(d.arm=='N')&(d.seed==seed)&(d.step==6000)].iloc[0]
        rows.append(dict(seed=seed,always_zero_input_pixels=zero,L1_fixed_zero_fraction=fixed,L1_eps_fraction=g.L1_eps_frac,
                         L1_eps_excess_on_nonzero_input_fraction=(g.L1_eps_frac-fixed)/(1-fixed),L2_eps_fraction=g.L2_eps_frac))
    pd.DataFrame(rows).to_csv(OUT/'epsilon_input_control.csv',index=False)

def main():
    result=audit();contrasts();epsilon_control();n,g=geometry();result.update(matched_mean_pairs=n,matched_mean_largest_endpoint_gap=g)
    (OUT/'integrity.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
