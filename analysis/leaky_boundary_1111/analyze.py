#!/usr/bin/env python3
"""Boundary-aligned clocks, fixed-response controls, and registered diagnostics."""
import hashlib,json,math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/leaky_boundary_1111'
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150,'savefig.dpi':180})
def save(fig,name):
    fig.savefig(OUT/(name+'.png'),bbox_inches='tight');fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
def ci(v):
    v=np.asarray(v,float);v=v[np.isfinite(v)];n=len(v)
    t={1:12.7062047364,2:4.30265272975,3:3.18244630528,4:2.7764451052}.get(n-1,np.nan)
    m=float(v.mean()) if n else np.nan;h=t*v.std(ddof=1)/np.sqrt(n) if n>1 else np.nan
    return dict(n=n,mean=m,lo=m-h,hi=m+h,min=float(v.min()) if n else np.nan,max=float(v.max()) if n else np.nan)
def raw(p):
    p=Path(p)
    if p.exists():return p
    rel=str(Path(*p.parts[p.parts.index('results'):]))
    m=json.loads((OUT/'backup_manifest.json').read_text());return Path(next(f['backup'] for f in m['files'] if f['source_rel']==rel))
def load():
    complete=json.loads((OUT/'COMPLETE.json').read_text());sel={int(k):v for k,v in complete['selection'].items()};frames=[];summaries=[];prefix=[];spec=[]
    for seed,s in sel.items():
        t=pd.read_csv(OUT/'prefix'/f's{seed}'/'tasks.csv');t['seed']=seed;prefix.append(t)
    for p in sorted((OUT/'probes').glob('*/s*/age_*/summary.csv')):
        assert (p.parent/'done.json').exists();summary=pd.read_csv(p);r=summary.iloc[0];seed=int(r.seed);age=float(r.age);s=sel[seed]
        role='pre' if age==s['pre_age'] else ('post' if age==s['post_age'] else 'other')
        phase=age-s['pre_age'];summary['role']=role;summary['phase']=phase;summary['path']=str(p.parent);summaries.append(summary)
        d=pd.read_csv(p.parent/'curves_t0.csv').copy();d['arm']=r.arm;d['seed']=seed;d['age']=age;d['role']=role;d['phase']=phase;frames.append(d)
        for budget in [6000,24000,96000]:
            if r.steps>=budget:summary[f'online_first{budget}']=d[(d.step>0)&(d.step%75==0)&(d.step<=budget)].online_acc.mean()
    for p in sorted((OUT/'spectra').glob('s*/age_*/spectrum.csv')):
        assert (p.parent/'done.json').exists();d=pd.read_csv(p);seed=int(d.seed.iloc[0]);d['phase']=d.age-sel[seed]['pre_age'];spec.append(d)
    s=pd.concat(summaries,ignore_index=True);d=pd.concat(frames,ignore_index=True);p=pd.concat(prefix,ignore_index=True);sp=pd.concat(spec,ignore_index=True)
    assert len(s)==70 and len(p)==30
    s.to_csv(OUT/'probe_seed_summary.csv',index=False);d[(d.role.isin(['pre','post']))&d.step.isin([0,1,75,150,300,750,1500,3000,6000,24000,96000])].to_csv(OUT/'paired_dense_points.csv',index=False);d.to_csv(OUT/'all_probe_curves.csv.gz',index=False);p.to_csv(OUT/'prefix_tasks.csv',index=False);sp.to_csv(OUT/'spectral_summary.csv',index=False)
    pd.DataFrame(sel.values()).to_csv(OUT/'boundary_selection.csv',index=False)
    return s,d,p,sp,sel
def clocks(s,d,sel):
    rows=[];normed=[];quarters=[];comparisons=[]
    for seed in range(3,8):
        ss=s[(s.seed==seed)&(s.arm=='N')].set_index('role');pre=ss.loc['pre'];post=ss.loc['post'];row=dict(seed=seed,pre_age=pre.age,post_age=post.age)
        for role,g in [('pre',pre),('post',post)]:
            for k in ['T90','T90_sustained','Tce2.0','Tce1.5','Tce1.0','Tce0.5','Tce0.2']:row[role+'_'+k]=g[k]
            row[role+'_learning_time']=g['Tce0.5']-g['Tce2.0']
        row['T90_ratio']=post.T90/pre.T90;row['learning_time_ratio']=row['post_learning_time']/row['pre_learning_time']
        vals=[];grid=np.linspace(0,1,51);curves={}
        for threshold in ['Tce1.5','Tce1.0','Tce0.5']:
            v=(post[threshold]-post['Tce2.0'])/(pre[threshold]-pre['Tce2.0']) if pre[threshold]>pre['Tce2.0'] else np.nan
            row[threshold+'_interval_ratio']=v;vals.append(v)
        for role,g in [('pre',pre),('post',post)]:
            if np.isfinite(g['Tce2.0']) and g['Tce0.5']>g['Tce2.0']:
                c=d[(d.seed==seed)&(d.arm=='N')&(d.role==role)]
                yy=np.interp(g['Tce2.0']+grid*(g['Tce0.5']-g['Tce2.0']),c.step,c.ce);curves[role]=yy
                normed.extend(dict(seed=seed,role=role,u=u,ce=v) for u,v in zip(grid,yy))
        row['normalized_ce_rmse']=float(np.sqrt(np.mean((curves['post']-curves['pre'])**2))) if len(curves)==2 else np.nan
        row['interval_ratio_range']=max(vals)/min(vals) if all(np.isfinite(vals)) and min(vals)>0 else np.nan
        row['clock_test_available']=np.isfinite(row['normalized_ce_rmse']) and np.isfinite(row['interval_ratio_range'])
        row['single_clock_consistent']=bool(row['normalized_ce_rmse']<=.10 and row['interval_ratio_range']<=2) if row['clock_test_available'] else None
        q=s[(s.seed==seed)&(s.arm=='N')&s.phase.between(0,1)].sort_values('phase').copy();q['learning_time']=q['Tce0.5']-q['Tce2.0']
        for metric in ['T90','learning_time']:
            v=q[metric].to_numpy();rr=v[1:]/v[:-1];row[metric+'_max_quarter_ratio']=np.nanmax(rr) if np.isfinite(rr).any() else np.nan
            row[metric+'_sharp_available']=bool(np.isfinite(v).all() and (v>0).all())
            row[metric+'_sharp']=bool(v[-1]/v[0]>=3 and np.max(rr)>=2) if row[metric+'_sharp_available'] else None
        quarters.append(q);rows.append(row)
    pd.DataFrame(rows).to_csv(OUT/'clock_seed_tests.csv',index=False);pd.concat(quarters).to_csv(OUT/'quarter_source_clocks.csv',index=False);pd.DataFrame(normed).to_csv(OUT/'normalized_ce_curves.csv',index=False)
    selected=s[s.role.isin(['pre','post'])]
    for role in ['pre','post']:
        for arm in ['G','L']:
            a=selected[(selected.arm==arm)&(selected.role==role)].set_index('seed');n=selected[(selected.arm=='N')&(selected.role==role)].set_index('seed')
            for metric in ['acc_6000','acc_24000','acc_96000','ce_6000','ce_24000','ce_96000','T90','T90_sustained']:
                comparisons.append(dict(role=role,arm=arm,reference='N',metric=metric,**ci(a[metric]-n[metric])))
    for arm in ['N','G','L']:
        a=selected[(selected.arm==arm)&(selected.role=='post')].set_index('seed');n=selected[(selected.arm==arm)&(selected.role=='pre')].set_index('seed')
        for metric in ['acc_6000','acc_24000','acc_96000','T90','T90_sustained']:
            comparisons.append(dict(role='post-minus-pre',arm=arm,reference=arm,metric=metric,**ci(a[metric]-n[metric])))
    comparisons=pd.DataFrame(comparisons);comparisons.to_csv(OUT/'paired_contrasts.csv',index=False)
    # Registered interaction: whether fixed-response minus natural benefit changes across the boundary.
    ints=[]
    for arm in ['G','L']:
        for metric in ['acc_6000','acc_24000','acc_96000']:
            p=selected.pivot(index='seed',columns=['arm','role'],values=metric)
            v=(p[(arm,'post')]-p[('N','post')])-(p[(arm,'pre')]-p[('N','pre')])
            ints.append(dict(arm=arm,metric=metric,**ci(v)))
    pd.DataFrame(ints).to_csv(OUT/'control_boundary_interactions.csv',index=False)
    return pd.DataFrame(rows),pd.DataFrame(normed)

def figures(s,d,p,sp,sel,clock,normed):
    fig,ax=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for seed,g in p.groupby('seed'):ax[0].plot(g.task,g.online_acc,marker='o',alpha=.6,label=f's{seed}')
    ax[0].set(xlabel='Natural training task',ylabel='Online accuracy',ylim=(0,1));ax[0].legend(fontsize=8)
    for seed,g in s[s.arm=='N'].groupby('seed'):
        g=g.sort_values('age');ax[1].plot(g.age,g.T90,marker='o',ms=3,alpha=.7)
        h=g[g.T90.isna()];ax[1].scatter(h.age,h.steps,marker='^',s=30)
    for n in [6000,24000,96000]:ax[1].axhline(n,color='gray',ls=':',lw=.7)
    ax[1].set(xlabel='Age of source state (same new task)',ylabel='Updates to 90% (triangles: censored)',yscale='log')
    positions=[];labels=[]
    for i,arm in enumerate(['N','G','L']):
        for j,role in enumerate(['pre','post']):
            g=s[(s.arm==arm)&(s.role==role)];ii=ci(g.acc_6000);x=i*3+j
            ax[2].errorbar(x,ii['mean'],yerr=[[ii['mean']-ii['lo']],[ii['hi']-ii['mean']]],fmt='o',color='C0' if role=='pre' else 'C1',capsize=3);positions.append(x);labels.append(arm+'\n'+role)
    ax[2].set(xticks=positions,xticklabels=labels,ylabel='Same-task accuracy at 6k',ylim=(0,1.05))
    fig.suptitle('Boundary test: a=.003 / independent seeds3–7 / N native, G fixed gates, L fixed Jacobian')
    save(fig,'boundary_overview')
    fig,ax=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for j,arm in enumerate(['N','G','L']):
        for role,col in [('pre','C0'),('post','C1')]:
            g=d[(d.arm==arm)&(d.role==role)];av=g.groupby('step').mean(numeric_only=True)
            for seed,z in g.groupby('seed'):
                ax[0,j].plot(z.step,z.acc,color=col,alpha=.15,lw=.7);ax[1,j].plot(z.step,z.ce,color=col,alpha=.15,lw=.7)
            ax[0,j].plot(av.index,av.acc,label=role,color=col,lw=2);ax[1,j].plot(av.index,av.ce,color=col,lw=2)
        ax[0,j].set_title(arm);ax[0,j].set(ylabel='Accuracy',ylim=(0,1.03));ax[1,j].set(ylabel='Cross-entropy',yscale='log')
        for a in ax[:,j]:a.set_xscale('symlog',linthresh=750);a.set_xticks([0,750,6000,24000,96000],['0','750','6k','24k','96k']);a.set_xlabel('Updates on identical probe task');a.axvline(6000,color='gray',ls=':',lw=.7)
    ax[0,0].legend();fig.suptitle('Identical starting forward and first update; subsequent response evolves differently')
    save(fig,'paired_learning')
    fig,ax=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for (seed,role),g in normed.groupby(['seed','role']):ax[0].plot(g.u,g.ce,color='C0' if role=='pre' else 'C1',alpha=.6,ls='-' if role=='post' else '--')
    ax[0].set(xlabel='Time normalized between CE<=2 and CE<=.5',ylabel='Cross-entropy');ax[0].set_title('Dashed blue: pre; solid orange: post')
    q=s[(s.arm=='N')&s.phase.between(0,1)]
    for seed,g in q.groupby('seed'):
        g=g.sort_values('phase');ax[1].plot(g.phase,g.T90,marker='o',label=f's{seed}');ax[2].plot(g.phase,g['Tce0.5']-g['Tce2.0'],marker='o')
    ax[1].set(xlabel='Actual source-path fraction pre→post',ylabel='T90 (updates)',yscale='log');ax[1].legend(fontsize=8)
    ax[2].set(xlabel='Actual source-path fraction pre→post',ylabel='Time from CE<=2 to CE<=.5',yscale='log')
    fig.suptitle('Is the boundary a smooth clock change? No weight interpolation; held-out labels fixed')
    save(fig,'clock_collapse')
    fig,ax=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for arm,role,col,ls in [('N','pre','C0','--'),('N','post','C1','-'),('G','pre','C2','--'),('G','post','C3','-')]:
        g=d[(d.arm==arm)&(d.role==role)&(d.step<=6000)].groupby('step').mean(numeric_only=True)
        for l in [1,2]:
            for j,key in enumerate(['pplus','zero_positive_input','gate_participation']):
                y=g[f'L{l}_{key}'].dropna();ax[l-1,j].plot(y.index,y,label=arm+' '+role,color=col,ls=ls)
    for l in [1,2]:
        ax[l-1,0].set_ylabel(f'L{l} positive-branch fraction');ax[l-1,1].set_ylabel(f'L{l} inputs with no positive units');ax[l-1,2].set_ylabel(f'L{l} gate-covariance participation')
    for a in ax.flat:a.set_xscale('symlog',linthresh=75);a.set_xticks([0,75,750,6000],['0','75','750','6k']);a.set_xlabel('Updates on common probe')
    ax[0,0].legend(fontsize=8);fig.suptitle('Used activation patterns: G holds source gates fixed; N can close and reopen')
    save(fig,'gate_dynamics')
    fig,ax=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for i,mode in enumerate(['euclid','adam_frozen']):
        q=sp[(sp['mode']==mode)&sp.phase.between(0,1)]
        for seed,g in q.groupby('seed'):
            g=g.sort_values('phase');ax[i,0].plot(g.phase,g.mean_eigenvalue,marker='o',ms=3,label=f's{seed}')
            for j,key in [(1,'remaining_t100'),(2,'ridge_residual_0.0001')]:
                valid=g[key+'_converged'].astype(bool);ax[i,j].plot(g.phase,g[key],marker='o',ms=3,alpha=.7)
                ax[i,j].scatter(g.loc[~valid,'phase'],g.loc[~valid,key],marker='x',color='black',s=50)
        ax[i,0].set_ylabel(mode+' mean eigenvalue');ax[i,0].set_yscale('log');ax[i,1].set_ylabel('Demand remaining at normalized time100');ax[i,2].set_ylabel('Ridge demand residual, alpha=1e-4')
    for a in ax.flat:a.set_xlabel('Source-path fraction pre→post')
    ax[0,0].legend(fontsize=8);fig.suptitle('Task-direction response on all1200 images; x marks unconverged64/128 tail estimates')
    save(fig,'spectral_demand')

def margins(s,sel):
    rows=[];cdfs=[];grid=np.linspace(-8,8,321)
    for seed in range(3,8):
        for phase in [0,.25,.5,.75,1]:
            rr=s[(s.seed==seed)&(s.arm=='N')&np.isclose(s.phase,phase)].iloc[0]
            rawdata=np.load(raw(Path(rr.path)/'logits_t0_s6000.npz'));f=rawdata['logits'];y=rawdata['labels'];other=f.copy();other[np.arange(len(y)),y]=-np.inf
            margin=f[np.arange(len(y)),y]-other.max(1);prob=np.exp(f-f.max(1,keepdims=True));prob/=prob.sum(1,keepdims=True)
            rows.append(dict(seed=seed,phase=phase,age=rr.age,acc=(margin>0).mean(),mean_margin=margin.mean(),near0_fraction=(abs(margin)<.1).mean(),q10=np.quantile(margin,.1),q50=np.quantile(margin,.5),q90=np.quantile(margin,.9)))
            cdfs.extend(dict(seed=seed,phase=phase,margin=x,cdf=(margin<=x).mean()) for x in grid)
    df=pd.DataFrame(rows);cf=pd.DataFrame(cdfs);df.to_csv(OUT/'margin_seed_summary.csv',index=False);cf.to_csv(OUT/'margin_cdfs.csv',index=False)
    fig,ax=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for phase,g in cf.groupby('phase'):
        av=g.groupby('margin').cdf.mean();ax[0].plot(av.index,av,label=f'path {phase:g}')
    ax[0].axvline(0,color='gray',ls=':');ax[0].set(xlabel='True-class margin at6k (display -8..8)',ylabel='Empirical CDF');ax[0].legend(fontsize=8)
    for seed,g in s[(s.arm=='N')&s.phase.between(0,1)].groupby('seed'):
        g=g.sort_values('phase');ax[1].plot(g.phase,g.acc_6000,marker='o');ax[2].plot(g.phase,g.ce_6000,marker='o')
    ax[1].set(xlabel='Source-path fraction pre→post',ylabel='Accuracy at6k',ylim=(0,1.03));ax[2].set(xlabel='Source-path fraction pre→post',ylabel='CE at6k')
    fig.suptitle('Check metric amplification: same new labels, continuous margins and CE alongside accuracy')
    save(fig,'margin_boundary')

def coverage(s,d,p):
    # All metrics and sampling times below were registered; the interpretation of
    # a near-one-positive-unit regime is post hoc and is not a phase classifier.
    q=d[(d.arm=='N')&d.phase.between(0,1)&d.step.isin([0,1,75,150,300,750,1500,3000,6000])].copy()
    q['L2_mean_positive_count']=100*q.L2_pplus
    q.to_csv(OUT/'source_quarter_gate_dynamics.csv',index=False)
    rows=[];pairs=[]
    metrics=['L1_pplus','L2_pplus','L1_gmean','L2_gmean','L1_gvar','L2_gvar','L1_zero_positive_input','L2_zero_positive_input','L2_mean_positive_count','L2_gate_participation','L2_unique_gate_patterns','L2_gate_numerical_rank','L2_feature_participation','acc','ce']
    for (phase,step),g in q.groupby(['phase','step']):
        for metric in metrics:rows.append(dict(phase=phase,step=step,metric=metric,**ci(g[metric])))
    for step in sorted(q.step.unique()):
        z=q[q.step==step]
        for metric in metrics:
            v=z.pivot(index='seed',columns='phase',values=metric)
            pairs.append(dict(step=step,metric=metric,**ci(v[1.]-v[0.])))
    pd.DataFrame(rows).to_csv(OUT/'gate_aggregate.csv',index=False);pd.DataFrame(pairs).to_csv(OUT/'gate_prepost_contrasts.csv',index=False)
    fig,ax=plt.subplots(1,3,figsize=(13,3.9),layout='constrained')
    for seed,g in p.groupby('seed'):ax[0].plot(g.task,g.online_acc,marker='o',ms=4,alpha=.65)
    ax[0].set(xlabel='Sequential task (6k updates each)',ylabel='Online accuracy',ylim=(0,1));ax[0].axvspan(3,4,color='gray',alpha=.12)
    for seed,g in q[q.step==75].groupby('seed'):
        g=g.sort_values('phase');ax[1].plot(g.phase,g.L2_zero_positive_input*100,marker='o',ms=4,alpha=.65)
    ax[1].set(xlabel='Source-path fraction before → after',ylabel='Inputs with no positive L2 unit (%)',ylim=(-2,100));ax[1].set_title('75 updates into the SAME new task')
    for seed,g in s[(s.arm=='N')&s.phase.between(0,1)].groupby('seed'):
        g=g.sort_values('phase');ax[2].plot(g.phase,g.T90,marker='o',ms=4,alpha=.65,label=f'seed {seed}')
    ax[2].axhline(6000,color='black',ls=':',label='6k task budget');ax[2].set(xlabel='Source-path fraction before → after',ylabel='Updates needed for 90% accuracy');ax[2].legend(fontsize=8)
    fig.suptitle('Boundary anatomy: gate coverage, learning time, and the fixed task budget')
    save(fig,'boundary_mechanism')

def main():
    s,d,p,sp,sel=load();c,n=clocks(s,d,sel);figures(s,d,p,sp,sel,c,n);margins(s,sel);coverage(s,d,p)
    report=c.to_string(index=False)+'\n'+s[s.role.isin(['pre','post'])].groupby(['arm','role'])[['acc_6000','acc_24000','acc_96000','T90','T90_sustained']].mean().to_string()
    print('\n'.join(line.rstrip() for line in report.splitlines()))
if __name__=='__main__':main()
